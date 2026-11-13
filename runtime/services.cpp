#include "services.h"
#include "bindings.h"
#include "tls.h"
#include "nvidia-metadata.h"
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <zlib.h>

#include <climits>
#include <fcntl.h>
#include <limits>
#ifdef MOM_EMBEDDED_APP_RESOURCES
#include "app-resources.inc"
#endif

namespace mom::runtime {
namespace {
// Synchronous libuv requests still allocate result storage, which must be cleaned on every exit.
struct FsRequest {
  uv_fs_t request{};
  ~FsRequest() {uv_fs_req_cleanup(&request);}
};

// Own the descriptor through native/JS/allocation failures. Explicit close reports errors once;
// cleanup never closes a recycled descriptor after a failed close or throws while unwinding.
struct FileDescriptor {
  int fd;
  explicit FileDescriptor(int value) : fd(value) {}
  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;
  int close() noexcept {
    const int value = fd; fd = -1;
    if (value < 0) {return 0;}
    FsRequest request;
    return uv_fs_close(nullptr, &request.request, value, nullptr);
  }
  ~FileDescriptor() {(void)close();}
};

// JS exposes uint32 modes. Pass only actual permission bits to the signed OS interface, never
// overflow a conversion (or rely on a Node-version-specific native assertion for extreme values).
int fileMode(Runtime &rt, const Value &value) {
  if (!value.isNumber()) {throw JSError(rt, "Invalid file mode");}
  const double mode = value.asNumber();
  if (!std::isfinite(mode) || mode < 0 || mode > UINT32_MAX || std::floor(mode) != mode) {
    throw JSError(rt, "Invalid file mode");
  }
  return static_cast<int>(static_cast<uint32_t>(mode) & 07777U);
}

std::string filename(Runtime &rt, const Value &value) {
  if (!value.isString()) {throw JSError(rt, "File path must be a string");}
  auto name = value.asString(rt).utf8(rt);
  if (name.find('\0') != std::string::npos) {throw JSError(rt, "File path contains NUL");}
  return name;
}

int nativeIsIP(const std::string &host) {
  // This is a literal classifier, not DNS. Keep the bridge input bounded while retaining Node's
  // scope syntax (ASCII alnum, '-', '.', ':') without probing whether an interface exists.
  constexpr size_t kMaxIpLiteralLength = 4096;
  if (host.empty() || host.size() > kMaxIpLiteralLength || host.find('\0') != std::string::npos) {return 0;}
  char address[16]{};
  if (uv_inet_pton(AF_INET, host.c_str(), address) == 0) {return 4;}

  std::string literal = host;
  const auto zone = literal.find('%');
  if (zone != std::string::npos) {
    if (zone + 1 == literal.size()) {return 0;}
    for (size_t i = zone + 1; i < literal.size(); ++i) {
      const auto c = static_cast<unsigned char>(literal[i]);
      const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
      const bool digit = c >= '0' && c <= '9';
      if (!alpha && !digit && c != '-' && c != '.' && c != ':') {return 0;}
    }
    literal.resize(zone);
  }
  return uv_inet_pton(AF_INET6, literal.c_str(), address) == 0 ? 6 : 0;
}

// A host function can outlive a property lookup's receiver. Keep hash state alive in method
// capabilities, never capture a dangling HostObject pointer or retain JS on a native thread.
class Hash final : public HostObject, public std::enable_shared_from_this<Hash> {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_{EVP_MD_CTX_new(), EVP_MD_CTX_free};
public:
  Hash() {
    if (!context_ || EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1) {
      throw std::runtime_error("Cannot initialize SHA-256");
    }
  }
  Value get(Runtime &rt, const PropNameID &property) override {
    const auto name = property.utf8(rt);
    if (name != "update" && name != "digest") {return Value::undefined();}
    return Function::createFromHostFunction(rt, property, name == "update" ? 1 : 0,
        [state = shared_from_this(), name](Runtime &rt, const Value &, const Value *args, size_t count) -> Value {
      try {
        if (!state->context_) {throw JSError(rt, codedError(rt, "ERR_CRYPTO_HASH_FINALIZED", "Digest already called"));}
        ERR_clear_error();
        if (name == "update") {
          if (!count) {throw JSError(rt, "Missing hash input");}
          const auto bytes = byteView(rt, args[0]);
          // A forged view's getters may reenter digest(); never pass a finalized context to EVP.
          if (!state->context_) {throw JSError(rt, codedError(rt, "ERR_CRYPTO_HASH_FINALIZED", "Digest already called"));}
          if (EVP_DigestUpdate(state->context_.get(), bytes.data(rt), bytes.size) != 1) {
            throw std::runtime_error("SHA-256 update failed");
          }
          return Value::undefined();
        }
        std::vector<uint8_t> bytes(EVP_MAX_MD_SIZE);
        unsigned size = 0;
        const int status = EVP_DigestFinal_ex(state->context_.get(), bytes.data(), &size);
        state->context_.reset();
        if (status != 1 || size != 32) {throw std::runtime_error("SHA-256 digest failed");}
        bytes.resize(size);
        return byteBuffer(rt, std::move(bytes));
      } catch (const JSError &) {throw;}
      catch (const std::exception &error) {throw JSError(rt, error.what());}
    });
  }
};
} // namespace

void installServices(Runtime &rt) {
  initializeCrypto(); // Disable host OpenSSL configuration before any RAND/EVP auto-initialization.
  binding(rt, "__momNvidiaComputeCapabilities", 0, [](Runtime &rt, const Value *) -> Value {
    const auto values = nvidiaComputeCapabilities();
    Array result(rt, values.size());
    for (size_t i = 0; i < values.size(); ++i) {result.setValueAtIndex(rt, i, static_cast<double>(values[i]));}
    return result;
  });
  binding(rt, "__momIsIP", 1, [](Runtime &rt, const Value *args) -> Value {
    if (!args[0].isString()) {return Value(0.0);}
    return Value(static_cast<double>(nativeIsIP(args[0].asString(rt).utf8(rt))));
  });
  binding(rt, "__momHashCreate", 0, [](Runtime &rt, const Value *) -> Value {
    return Object::createFromHostObject(rt, std::make_shared<Hash>());
  });
  binding(rt, "__momRandomBytes", 1, [](Runtime &rt, const Value *args) -> Value {
    const double size = args[0].asNumber();
    if (!std::isfinite(size) || size < 0 || size > INT_MAX || std::floor(size) != size) {
      throw JSError(rt, "Invalid random byte count");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    ERR_clear_error();
    if (size && RAND_bytes(bytes.data(), static_cast<int>(size)) != 1) {
      throw std::runtime_error("Secure random source failed"); // Never substitute predictable bytes.
    }
    return byteBuffer(rt, std::move(bytes));
  });
  binding(rt, "__momGzip", 1, [](Runtime &rt, const Value *args) -> Value {
    const auto bytes = byteView(rt, args[0]);
    if (bytes.size > (std::numeric_limits<uInt>::max)()) {throw JSError(rt, "Gzip input is too large");}
    z_stream stream{};
    if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
      throw std::runtime_error("Cannot initialize gzip");
    }
    struct Release {z_stream &stream; ~Release() {deflateEnd(&stream);}} release{stream};
    const auto capacity = deflateBound(&stream, static_cast<uLong>(bytes.size));
    if (capacity > (std::numeric_limits<uInt>::max)()) {throw JSError(rt, "Gzip output is too large");}
    std::vector<uint8_t> compressed(static_cast<size_t>(capacity));
    stream.next_in = const_cast<Bytef *>(bytes.data(rt));
    stream.avail_in = static_cast<uInt>(bytes.size);
    stream.next_out = compressed.data();
    stream.avail_out = static_cast<uInt>(compressed.size());
    if (deflate(&stream, Z_FINISH) != Z_STREAM_END) {throw std::runtime_error("Gzip compression failed");}
    compressed.resize(stream.total_out);
    return byteBuffer(rt, std::move(compressed));
  });
  binding(rt, "__momCpus", 0, [](Runtime &rt, const Value *) -> Value {
    uv_cpu_info_t *info = nullptr;
    int count = 0;
    checkSystem(rt, uv_cpu_info(&info, &count), "uv_cpu_info");
    struct Release {uv_cpu_info_t *info; int count; ~Release() {uv_free_cpu_info(info, count);}} release{info, count};
    Array cpus(rt, static_cast<size_t>(count));
    for (int i = 0; i < count; i++) {
      Object cpu(rt), times(rt);
      cpu.setProperty(rt, "model", String::createFromUtf8(rt, info[i].model));
      cpu.setProperty(rt, "speed", info[i].speed);
      times.setProperty(rt, "user", static_cast<double>(info[i].cpu_times.user));
      times.setProperty(rt, "nice", static_cast<double>(info[i].cpu_times.nice));
      times.setProperty(rt, "sys", static_cast<double>(info[i].cpu_times.sys));
      times.setProperty(rt, "idle", static_cast<double>(info[i].cpu_times.idle));
      times.setProperty(rt, "irq", static_cast<double>(info[i].cpu_times.irq));
      cpu.setProperty(rt, "times", std::move(times));
      cpus.setValueAtIndex(rt, static_cast<size_t>(i), std::move(cpu));
    }
    return cpus;
  });
  binding(rt, "__momReadFile", 1, [](Runtime &rt, const Value *args) -> Value {
    const auto path = filename(rt, args[0]);
#ifdef MOM_EMBEDDED_APP_RESOURCES
    // Only the built-in policy path is virtual. User configuration and all other filesystem
    // accesses retain normal OS errors; never treat an arbitrary basename as an embedded file.
    if (path == kEmbeddedPolicyPath) {
      const auto bytes = embedded_resource("GPU-CONFIG.md");
      return byteBuffer(rt, std::vector<uint8_t>(bytes.begin(), bytes.end()));
    }
#endif
    FsRequest open;
    const int fd = uv_fs_open(nullptr, &open.request, path.c_str(), O_RDONLY, 0, nullptr);
    checkSystem(rt, fd, "open", path);
    FileDescriptor file(fd);
    std::vector<uint8_t> bytes;
    for (;;) {
      char block[65536];
      uv_buf_t buffer = uv_buf_init(block, sizeof(block));
      FsRequest read;
      const auto size = uv_fs_read(nullptr, &read.request, fd, &buffer, 1, -1, nullptr);
      checkSystem(rt, static_cast<int>(size), "read", path);
      if (!size) {break;}
      if (bytes.size() > 0x7fffffff - static_cast<size_t>(size)) {throw JSError(rt, "File exceeds native Buffer capacity");}
      bytes.insert(bytes.end(), block, block + size);
    }
    return byteBuffer(rt, std::move(bytes));
  });
  binding(rt, "__momWriteFile", 4, [](Runtime &rt, const Value *args) -> Value {
    const auto path = filename(rt, args[0]);
    const auto bytes = byteView(rt, args[1]);
    const int mode = fileMode(rt, args[2]);
    if (!args[3].isBool()) {throw JSError(rt, "Invalid exclusive-create flag");}
    const bool exclusive = args[3].asBool();
    if (bytes.size > INT_MAX) {throw JSError(rt, "File exceeds native Buffer capacity");}
    // Resolve every JS getter/range/mode before a destructive open. Retain the ArrayBuffer handle
    // throughout synchronous writes; zero-length data still creates/truncates the requested file.
    const auto *data = bytes.data(rt);
    FsRequest open;
    const int fd = uv_fs_open(nullptr, &open.request, path.c_str(),
        O_WRONLY | O_CREAT | O_TRUNC | (exclusive ? O_EXCL : 0), mode, nullptr);
    checkSystem(rt, fd, "open", path);
    FileDescriptor file(fd);
    for (size_t offset = 0; offset < bytes.size;) {
      auto buffer = uv_buf_init(reinterpret_cast<char *>(const_cast<uint8_t *>(data + offset)),
          static_cast<unsigned>(bytes.size - offset));
      FsRequest write;
      const auto written = uv_fs_write(nullptr, &write.request, fd, &buffer, 1, -1, nullptr);
      if (written <= 0) {checkSystem(rt, written ? static_cast<int>(written) : UV_EIO, "write", path);}
      if (static_cast<size_t>(written) > bytes.size - offset) {throw JSError(rt, "Invalid file write size");}
      offset += static_cast<size_t>(written);
    }
    checkSystem(rt, file.close(), "close", path);
    return Value::undefined();
  });
  binding(rt, "__momChmod", 2, [](Runtime &rt, const Value *args) -> Value {
    const auto path = filename(rt, args[0]);
    const int mode = fileMode(rt, args[1]);
    FsRequest chmod;
    checkSystem(rt, uv_fs_chmod(nullptr, &chmod.request, path.c_str(), mode, nullptr), "chmod", path);
    return Value::undefined();
  });
  binding(rt, "__momFileExists", 1, [](Runtime &rt, const Value *args) -> Value {
    const auto path = filename(rt, args[0]);
#ifdef MOM_EMBEDDED_APP_RESOURCES
    if (path == kEmbeddedPolicyPath) {return Value(true);}
#endif
    FsRequest stat;
    return Value(uv_fs_stat(nullptr, &stat.request, path.c_str(), nullptr) == 0);
  });
  binding(rt, "__momReadDirectory", 1, [](Runtime &rt, const Value *args) -> Value {
    const auto path = filename(rt, args[0]);
    FsRequest scan;
    const int count = uv_fs_scandir(nullptr, &scan.request, path.c_str(), 0, nullptr);
    checkSystem(rt, count, "scandir", path);
    Array names(rt, static_cast<size_t>(count));
    uv_dirent_t entry;
    for (int i = 0; i < count; i++) {
      checkSystem(rt, uv_fs_scandir_next(&scan.request, &entry), "scandir", path);
      names.setValueAtIndex(rt, static_cast<size_t>(i), String::createFromUtf8(rt, entry.name));
    }
    return names;
  });
  binding(rt, "__momWrite", 2, [](Runtime &rt, const Value *args) -> Value {
    const auto fd = args[0].asNumber();
    const auto bytes = byteView(rt, args[1]);
    if (!std::isfinite(fd) || fd < 0 || fd > INT_MAX || std::floor(fd) != fd || bytes.size > UINT_MAX) {
      throw JSError(rt, "Invalid synchronous write");
    }
    auto buffer = uv_buf_init(reinterpret_cast<char *>(const_cast<uint8_t *>(bytes.data(rt))), static_cast<unsigned>(bytes.size));
    FsRequest write;
    const auto written = uv_fs_write(nullptr, &write.request, static_cast<int>(fd), &buffer, 1, -1, nullptr);
    if (written < 0) {checkSystem(rt, static_cast<int>(written), "write");}
    return Value(static_cast<double>(written));
  });
}
} // namespace mom::runtime
