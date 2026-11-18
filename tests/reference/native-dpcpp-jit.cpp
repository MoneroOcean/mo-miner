// CPU-only execution of the maintained source compiler. No platform, device or queue is created.
#include <RTC.h>
#include <Options.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace jit = jit_compiler;
std::pair<std::string, std::string> nativeKawpowSource(const std::string&, uint64_t);
void nativeEmitKawpowSource(const char*);
static constexpr char source[] = R"SYCL(
#include <sycl/sycl.hpp>
namespace ex = sycl::ext::oneapi::experimental;
extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((ex::single_task_kernel))
void mom_static_jit_probe(int* output) { output[0] = 17; }
)SYCL";

// A successful RTCResult does not own its context's binary blobs. Release all images even when
// a later verifier rejects one; the result itself must outlive this guard and its views.
struct Images {
  const jit::RTCBundleInfo& bundle;
  ~Images() {for (const auto& image : bundle.DevImgInfos) jit::destroyBinary(image.BinaryInfo.BinaryStart);}
};

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

static void configure(const std::string& cpu) {
  jit::resetJITConfiguration();
  if (!cpu.empty()) {
    // These ABI arrays are length-delimited, not C strings (matching the maintained runtime).
    jit::DynArray<char> target(cpu.begin(), cpu.end());
    jit::addToJITConfiguration(jit::option::JITTargetCPU::set(std::move(target)));
    const std::string features = cpu == "sm_61" ? "+sm_61,+ptx64" : "+sm_120,+ptx87";
    jit::DynArray<char> attributes(features.begin(), features.end());
    jit::addToJITConfiguration(jit::option::JITTargetFeatures::set(std::move(attributes)));
  }
}

// Check real target output and registered free-function symbols, not only a success status.
static void verify(const jit::RTCResult& result, jit::BinaryFormat format, const std::string& cpu,
                   const std::string& kernel = "mom_static_jit_probe") {
  if (result.getErrorCode() != jit::RTCResult::RTCErrorCode::SUCCESS) {
    std::fprintf(stderr, "%s\n", result.getBuildLog());
    throw std::runtime_error("Maintained source compilation failed");
  }
  Images cleanup{result.getBundleInfo()};
  require(!cleanup.bundle.DevImgInfos.empty(), "Compiler returned no device images");
  bool found_kernel = false;
  for (const auto& image : cleanup.bundle.DevImgInfos) {
    const auto& binary = image.BinaryInfo;
    require(binary.Format == format && binary.AddressBits == 64 && binary.BinaryStart && binary.BinarySize >= 16,
            "Unexpected device image ABI/format");
    for (const auto& symbol : image.SymbolTable)
      found_kernel |= std::string(symbol.c_str()).find(kernel) != std::string::npos;
    if (format == jit::BinaryFormat::SPIRV) {
      uint32_t magic = 0; std::memcpy(&magic, binary.BinaryStart, sizeof(magic));
      require(magic == 0x07230203, "Invalid SPIR-V image");
    } else {
      const std::string ptx(reinterpret_cast<const char*>(binary.BinaryStart), binary.BinarySize);
      require(ptx.find(".target " + cpu) != std::string::npos && ptx.find(kernel) != std::string::npos,
              "PTX target/kernel does not match the requested specialization");
    }
  }
  require(found_kernel, "Compiler lost the original free-function kernel registration");
}

int main(int argc, char** argv) try {
  if (argc >= 2 && std::strcmp(argv[1], "--emit-source") == 0) {
    require(argc == 3, "Expected --emit-source <algorithm>");
    nativeEmitKawpowSource(argv[2]);
    return 0;
  }
  const std::string target = argc >= 2 ? argv[1] : "spirv";
  require(argc <= 3 && (target == "spirv" || target == "sm_61" || target == "sm_120"), "Expected spirv, sm_61 or sm_120 [algorithm]");
  const auto format = target == "spirv" ? jit::BinaryFormat::SPIRV : jit::BinaryFormat::PTX;
  const std::string cpu = target == "spirv" ? "" : target;
  const auto generated = argc == 3 ? nativeKawpowSource(argv[2], 607046) : std::make_pair(std::string("mom_static_jit_probe"), std::string(source));
  // Keep the in-memory source name relative: Windows VFS rejects the Linux-only root form.
  const jit::InMemoryFile file{"mom-native-jit-probe.cpp", generated.second.c_str()};
  const jit::View<jit::InMemoryFile> includes{nullptr, 0};
  const std::array<const char*, 2> options{{"-std=c++17", "-O2"}};
  const jit::View<const char*> args{options};
  configure(cpu);
  const auto hash = jit::calculateHash(file, includes, args, format);
  require(!hash.failed(), hash.failed() ? hash.getPreprocLog() : "Source hash failed");
  const auto result = jit::compileSYCL(file, includes, args, {nullptr, 0}, true, format);
  verify(result, format, cpu, generated.first);
  const auto& ir = result.getDeviceCodeIR();
  require(!ir.empty(), "Compiler did not preserve its persistent-cache IR");
  const auto cached = jit::compileSYCL(file, includes, args, {ir.begin(), ir.size()}, false, format);
  verify(cached, format, cpu, generated.first);
  const auto malformed = jit::compileSYCL({file.Path, "this is not C++;"}, includes, args, {nullptr, 0}, false, format);
  require(malformed.getErrorCode() != jit::RTCResult::RTCErrorCode::SUCCESS, "Invalid source unexpectedly compiled");
  configure(cpu);
  const auto again = jit::compileSYCL(file, includes, args, {nullptr, 0}, false, format);
  verify(again, format, cpu, generated.first);
  const auto hash_again = jit::calculateHash(file, includes, args, format);
  require(!hash_again.failed() && std::strcmp(hash.getHash(), hash_again.getHash()) == 0, "Failed compilation contaminated source hashing");
  std::printf("Maintained static SYCL JIT: %s%s%s source/cache/error-recovery/hash/binary-lifecycle passed\n",
      target.c_str(), argc == 3 ? " " : "", argc == 3 ? argv[2] : "");
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what()); return 1;
}
