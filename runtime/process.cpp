#include "process.h"
#include "bindings.h"
#include <uv.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>
#include <algorithm>
#ifndef _WIN32
#include <filesystem>
#include <sys/stat.h>
#endif

namespace mom::runtime {
using namespace facebook::jsi;
namespace {
// libuv uses UTF-8 on both systems and reports required capacity when a path does not fit.
template <typename Read> std::string osPath(Runtime &rt, Read read) {
  std::vector<char> buffer(4096);
  for (;;) {
    size_t size = buffer.size();
    const int status = read(buffer.data(), &size);
    if (!status) {return std::string(buffer.data(), size);}
    if (status != UV_ENOBUFS || buffer.size() >= 1024 * 1024 || size > 1024 * 1024) {
      throw JSError(rt, uv_strerror(status));
    }
    buffer.resize(size > buffer.size() ? size : buffer.size() * 2);
  }
}

} // namespace

const char *prepareWorkerDirectory(std::vector<std::string> &environment) {
#ifndef _WIN32
  if (std::find(environment.begin(), environment.end(), "MOM_CLUSTER_WORKER=1") == environment.end()) return nullptr;
  // A system GPU driver's empty RUNPATH entry otherwise imports libraries from the miner's
  // writable cwd, outside our direct dlopen guard. Apply cwd before exec/static constructors.
  struct stat root{};
  if (stat("/", &root) || !S_ISDIR(root.st_mode) || root.st_uid != 0 || (root.st_mode & 0022))
    throw std::runtime_error("Native worker requires a root-owned, non-writable filesystem root");
  const auto caller = std::filesystem::current_path();
  bool dataset = false;
  const char *paths[] = {"HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "MOM_JIT_CACHE_DIR",
    "MOM_VERTHASH_DATA", "ACPP_APPDB_DIR", "HIPSYCL_APPDB_DIR", "SYCL_CACHE_DIR", "CUDA_CACHE_PATH"};
  for (auto &entry : environment) {
    const auto separator = entry.find('=');
    if (separator == std::string::npos) continue;
    const auto key = entry.substr(0, separator), value = entry.substr(separator + 1);
    if (key == "MOM_VERTHASH_DATA" && !value.empty()) dataset = true;
    if (!value.empty() && std::find(std::begin(paths), std::end(paths), key) != std::end(paths)) {
      const std::filesystem::path selected(value);
      if (selected.is_relative()) entry = key + "=" + (caller / selected).string();
    }
  }
  // Verthash's existing default is caller-cwd/verthash.dat, not root/verthash.dat.
  if (!dataset) {
    environment.erase(std::remove(environment.begin(), environment.end(), "MOM_VERTHASH_DATA="), environment.end());
    environment.push_back("MOM_VERTHASH_DATA=" + (caller / "verthash.dat").string());
  }
  return "/";
#else
  (void)environment;
  return nullptr; // Windows driver search needs its own system-directory policy/qualification.
#endif
}

void installProcess(Runtime &rt, int argc, const char *const *argv) {
  Object info(rt);
#ifdef _WIN32
  info.setProperty(rt, "platform", "win32");
#else
  info.setProperty(rt, "platform", "linux");
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
  info.setProperty(rt, "arch", "arm64");
#elif defined(__x86_64__) || defined(_M_X64)
  info.setProperty(rt, "arch", "x64");
#else
#error Native process architecture needs an explicit Node-compatible name.
#endif
  info.setProperty(rt, "pid", static_cast<double>(uv_os_getpid()));
  info.setProperty(rt, "ppid", static_cast<double>(uv_os_getppid()));
  info.setProperty(rt, "execPath", String::createFromUtf8(rt, osPath(rt, uv_exepath)));
  Array arguments(rt, static_cast<size_t>(argc > 1 ? argc - 1 : 0));
  for (int i = 1; i < argc; i++) {
    arguments.setValueAtIndex(rt, static_cast<size_t>(i - 1), String::createFromUtf8(rt, argv[i]));
  }
  info.setProperty(rt, "arguments", std::move(arguments));
  rt.global().setProperty(rt, "__momProcessInfo", std::move(info));
  binding(rt, "__momCwd", 0, [](Runtime &rt, const Value *) -> Value {
    return String::createFromUtf8(rt, osPath(rt, uv_cwd));
  });
  binding(rt, "__momChdir", 1, [](Runtime &rt, const Value *args) -> Value {
    const auto directory = args[0].asString(rt).utf8(rt);
    // Node chdir, unlike fs path APIs, passes a C string to libuv. Preserve its NUL truncation.
    const int status = uv_chdir(directory.c_str());
    if (status) {throw JSError(rt, codedError(rt, uv_err_name(status), uv_strerror(status)));}
    return Value::undefined();
  });
  binding(rt, "__momExitNow", 1, [](Runtime &rt, const Value *args) -> Value {
    const double code = args[0].asNumber();
    if (!integerInRange(code, INT32_MIN, INT32_MAX)) {throw JSError(rt, "Invalid process exit code");}
    // process.exit is explicitly forced, unlike exitCode/event-loop completion. Flush our own
    // synchronous output, then let the OS close handles; never unwind through generated C frames.
    std::fflush(nullptr);
    std::_Exit(static_cast<int>(code));
  });
}

int processExitCode(Runtime &rt) {
  const auto code = rt.global().getPropertyAsObject(rt, "process").getProperty(rt, "exitCode");
  if (code.isUndefined()) {return 0;}
  const double value = code.asNumber();
  // A replaced JS descriptor must not turn NaN/out-of-range input into undefined native casts.
  if (!integerInRange(value, INT32_MIN, INT32_MAX)) {throw JSError(rt, "Invalid final process exit code");}
  return static_cast<int>(value);
}
} // namespace mom::runtime
