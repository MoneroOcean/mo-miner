#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>

// Compile the actual loader class without a CUDA SDK or access to GPU devices.
using nvrtcProgram = void*;
using nvrtcResult = int;
nvrtcResult nvrtcCreateProgram(...), nvrtcCompileProgram(...), nvrtcGetProgramLogSize(...),
    nvrtcGetProgramLog(...), nvrtcGetPTXSize(...), nvrtcGetPTX(...), nvrtcGetCUBINSize(...),
    nvrtcGetCUBIN(...), nvrtcDestroyProgram(...), nvrtcGetNumSupportedArchs(...),
    nvrtcGetSupportedArchs(...);
nvrtcResult nvrtcVersion(int*, int*);
const char* nvrtcGetErrorString(nvrtcResult);
static std::string mode;
static std::vector<std::string> names;
static nvrtcResult compiler_version(int* major, int* minor) {
  *major = mode == "legacy" ? 12 : 13;
  *minor = mode == "legacy" ? 6 : 3;
  return 0;
}
#ifdef MOM_STATIC_NVRTC
// Link-time API symbols stand in for compiler archives, not a second loader implementation.
nvrtcResult nvrtcCreateProgram(...) { return 0; }
nvrtcResult nvrtcCompileProgram(...) { return 0; }
nvrtcResult nvrtcGetProgramLogSize(...) { return 0; }
nvrtcResult nvrtcGetProgramLog(...) { return 0; }
nvrtcResult nvrtcGetPTXSize(...) { return 0; }
nvrtcResult nvrtcGetPTX(...) { return 0; }
nvrtcResult nvrtcGetCUBINSize(...) { return 0; }
nvrtcResult nvrtcGetCUBIN(...) { return 0; }
nvrtcResult nvrtcDestroyProgram(...) { return 0; }
nvrtcResult nvrtcGetNumSupportedArchs(...) { return 0; }
nvrtcResult nvrtcGetSupportedArchs(...) { return 0; }
nvrtcResult nvrtcVersion(int* major, int* minor) { return compiler_version(major, minor); }
const char* nvrtcGetErrorString(nvrtcResult) { return "fixture error"; }
#else
static int unused(...) { return 0; }
static void* mock_dlopen(const char* name, int) {
  names.emplace_back(name);
  if (mode == "none") return nullptr;
  if (std::strcmp(name, "/opt/mom/nvrtc-13.3.33/lib/libnvrtc-builtins.so.13.3") == 0)
    return mode == "missing-builtins" ? nullptr : reinterpret_cast<void*>(2);
  const bool managed = std::strcmp(name,
      "/opt/mom/nvrtc-13.3.33/lib/libnvrtc.so.13.3.33") == 0;
  if (mode == "legacy") return std::strcmp(name, "libnvrtc.so.12") == 0 ?
      reinterpret_cast<void*>(1) : nullptr;
  if (mode == "alias13") return std::strcmp(name, "libnvrtc.so.13") == 0 ?
      reinterpret_cast<void*>(1) : nullptr;
  if (mode == "unversioned") return std::strcmp(name, "libnvrtc.so") == 0 ?
      reinterpret_cast<void*>(1) : nullptr;
  if (!managed) throw std::runtime_error("unnecessary second compiler load");
  return reinterpret_cast<void*>(1);
}
static void* mock_dlsym(void*, const char* name) {
  if (std::strcmp(name, "nvrtcVersion") == 0)
    return mode == "missing-version" ? nullptr : reinterpret_cast<void*>(&compiler_version);
  if ((mode == "missing-cubin" && std::strcmp(name, "nvrtcGetCUBINSize") == 0) ||
      (mode == "missing-required" && std::strcmp(name, "nvrtcCompileProgram") == 0))
    return nullptr;
  return reinterpret_cast<void*>(&unused);
}
#define dlopen mock_dlopen
#define dlsym mock_dlsym
#endif
namespace mom {
#include "loader.inc"
}
static void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  try {
    check(argc == 2, "expected mode"); mode = argv[1];
    using PearlCompiler = mom::NvrtcApi;
    using CnCompiler = mom::NvrtcApi;
    auto& pearl = PearlCompiler::instance();
    const auto first_loads = names.size();
    auto& cn = CnCompiler::instance();
    check(&pearl == &cn && first_loads == names.size(), "algorithms loaded separate compilers");
#ifdef MOM_STATIC_NVRTC
    check(names.empty() && pearl.error.empty(), "static compiler searched for a library");
    check(pearl.ptx_available() && pearl.cubin_available(), "archive API is unavailable");
    int major = 0, minor = 0;
    check(pearl.version(&major, &minor) == 0 && major == 13 && minor == 3, "archive version mismatch");
    check(pearl.compile_program == &nvrtcCompileProgram && pearl.get_ptx == &nvrtcGetPTX &&
          pearl.get_cubin == &nvrtcGetCUBIN && pearl.destroy_program == &nvrtcDestroyProgram,
          "static compiler did not use link-time symbols");
#else
    check(names.front() == "/opt/mom/nvrtc-13.3.33/lib/libnvrtc.so.13.3.33",
          "managed compiler preference changed");
    const unsigned expected = mode == "none" || mode == "legacy" ? 4 :
        mode == "alias13" ? 3 : 2;
    check(names.size() == expected, "unexpected compiler search order");
    const bool present = mode != "none" && mode != "missing-required" && mode != "missing-builtins";
    check(pearl.ptx_available() == present, "required-symbol availability changed");
    check(pearl.cubin_available() == (present && mode != "missing-cubin"),
          "optional CUBIN availability changed");
    if (mode == "missing-version")
      check(!pearl.version && pearl.error.empty(), "optional version degraded healthy runtime");
    else if (present) {
      int major = 0, minor = 0;
      check(pearl.version && pearl.version(&major, &minor) == 0, "version unavailable");
      check(major == (mode == "legacy" ? 12 : 13) && minor == (mode == "legacy" ? 6 : 3),
            "wrong compiler version");
    }
    if (mode == "none") check(pearl.error == "NVRTC is not installed", "missing runtime error");
    if (mode == "missing-builtins") check(pearl.error == "the managed NVRTC builtins library is not installed",
                                           "missing builtins did not fail closed");
    if (mode == "missing-cubin") check(pearl.error == "missing NVRTC symbol nvrtcGetCUBINSize",
                                        "optional version hid CUBIN diagnostic");
#endif
    std::printf("SHARED_NVRTC_LOADER_PASS mode=%s GPU_calls=0\n", mode.c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what()); return 1;
  }
}
