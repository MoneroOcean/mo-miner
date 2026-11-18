// Native-release link boundary for optional loaders retained inside static vendor archives.
// Production only opens system GPU drivers. An explicit portable-CPU test opt-in permits one
// exact OS-registered Intel CPU ICD path; installed SDKs/plugins remain rejected by default.
#include <cstring>
#include <cstdlib>

namespace {
thread_local bool deniedLibrary = false;
}

// ICD files may contain a SONAME or a distro-specific absolute driver path (Intel uses the
// latter). Admit only known GPU ICD identities under OS library roots, never an environment-
// selected SDK/plugin path. Do not normalize traversal into an apparently trusted directory.
static bool openclDriver(const char* name) {
  const char* base = std::strrchr(name, '/');
  if (base) {
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(name); *p; ++p)
      if (*p < 32 || *p == 127 || *p == '\\') return false;
    const char* roots[] = {"/usr/lib/", "/usr/lib64/", "/lib/", "/lib64/"};
    bool system = false;
    for (const char* root : roots)
      if (std::strncmp(name, root, std::strlen(root)) == 0) system = true;
    if (!system) return false;
    for (const char* part = name + 1; *part;) {
      const char* end = std::strchr(part, '/');
      if (!end) break;
      const auto size = end - part;
      if (size == 0 || (size == 1 && part[0] == '.') ||
          (size == 2 && part[0] == '.' && part[1] == '.')) return false;
      part = end + 1;
    }
    ++base;
  } else base = name;
  return std::strcmp(base, "libigdrcl.so") == 0 ||
         std::strcmp(base, "libigdrcl_legacy1.so") == 0 ||
         std::strcmp(base, "libnvidia-opencl.so.1") == 0 ||
         std::strcmp(base, "libamdocl64.so") == 0 ||
         std::strcmp(base, "libamdocl64.so.2") == 0 ||
         std::strcmp(base, "libRusticlOpenCL.so.1") == 0;
}

// Intel's explicitly installed CPU-only OpenCL test ICD is not a GPU driver. Keep it out
// of the production allowlist; the exact package-owned registry path is opt-in for CI only.
static const char* portableCpuOpenclDriver(const char* name) {
  const char* enabled = std::getenv("MOM_REQUIRE_PORTABLE_CPU_TESTS");
  if (!enabled || std::strcmp(enabled, "1") != 0 || !name) return nullptr;
  static constexpr char path[] = "/opt/intel/oneapi/redist/lib/libintelocl.so";
  return std::strcmp(name, path) == 0 ? name : nullptr;
}

extern "C" void* __real_dlopen(const char* name, int flags);
extern "C" char* __real_dlerror();
// Report a refused dlopen once. The refusal cleared the loader's own error, so any error the
// loader holds now came from a later dl call (for example a failed dlsym) and is the newer one.
extern "C" char* __wrap_dlerror() {
  char* const loaderError = __real_dlerror();
  if (!deniedLibrary) return loaderError;
  deniedLibrary = false;
  if (loaderError) return loaderError;
  static thread_local char message[] = "MoM: application library loading is disabled";
  return message;
}
extern "C" void* __wrap_dlopen(const char* name, int flags) {
  deniedLibrary = false;
  if (const char* cpuDriver = portableCpuOpenclDriver(name)) return __real_dlopen(cpuDriver, flags);
  // A null name resolves this executable's own symbols; it loads no external component.
  if (!name || std::strcmp(name, "libcuda.so.1") == 0 ||
      std::strcmp(name, "libnvidia-ptxjitcompiler.so.1") == 0 ||
      std::strcmp(name, "libnvidia-ml.so.1") == 0 ||
      // Pinned Level Zero discovery passes these exact GPU SONAMEs. Do not admit its
      // ZE_ENABLE_ALT_DRIVERS paths, NPU drivers, loaders or application plugins.
      std::strcmp(name, "libze_intel_gpu.so.1") == 0 ||
      std::strcmp(name, "libze_intel_gpu_legacy1.so.1") == 0 || openclDriver(name))
    return __real_dlopen(name, flags);
  // Match dlopen's failure contract without opening any rejected path. Vendor loaders can
  // consume dlerror() unconditionally; a bare null return would make diagnostics crash.
  (void)__real_dlerror();
  deniedLibrary = true;
  return nullptr;
}
