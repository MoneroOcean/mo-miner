#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__) && \
    !defined(MOM_PEARLHASH_ESIMD_TU)
struct PearlHashCudaDriverApi {
#if defined(_WIN32)
  HMODULE library = nullptr;
#else
  void* library = nullptr;
#endif
  decltype(&cuInit) init = nullptr;
  decltype(&cuDeviceGetAttribute) device_attribute = nullptr;
  decltype(&cuDevicePrimaryCtxRetain) retain_primary = nullptr;
  decltype(&cuDevicePrimaryCtxRelease) release_primary = nullptr;
  decltype(&cuCtxSetCurrent) set_current = nullptr;
  decltype(&cuCtxSynchronize) synchronize = nullptr;
  decltype(&cuModuleLoadData) module_load = nullptr;
  decltype(&cuModuleUnload) module_unload = nullptr;
  decltype(&cuModuleGetFunction) module_function = nullptr;
  decltype(&cuFuncSetAttribute) function_attribute = nullptr;
  decltype(&cuLaunchKernel) launch = nullptr;
  decltype(&cuEventCreate) event_create = nullptr;
  decltype(&cuEventDestroy) event_destroy = nullptr;
  decltype(&cuEventRecord) event_record = nullptr;
  decltype(&cuEventQuery) event_query = nullptr;
  decltype(&cuEventElapsedTime) event_elapsed = nullptr;
  decltype(&cuGetErrorString) error_string = nullptr;
  std::string error;

  template <typename T>
  bool symbol(T& output, const char* name) {
#if defined(_WIN32)
    output = reinterpret_cast<T>(GetProcAddress(library, name));
#else
    output = reinterpret_cast<T>(dlsym(library, name));
#endif
    if (output) return true;
    error = std::string("missing CUDA driver symbol ") + name;
    return false;
  }

  PearlHashCudaDriverApi() {
#if defined(_WIN32)
    library = LoadLibraryA("nvcuda.dll");
#else
    library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
    if (!library) {
      error = "the CUDA driver library is not installed";
      return;
    }
    (void)(symbol(init, "cuInit") &&
           symbol(device_attribute, "cuDeviceGetAttribute") &&
           symbol(retain_primary, "cuDevicePrimaryCtxRetain") &&
           symbol(release_primary, "cuDevicePrimaryCtxRelease") &&
           symbol(set_current, "cuCtxSetCurrent") &&
           symbol(synchronize, "cuCtxSynchronize") &&
           symbol(module_load, "cuModuleLoadData") &&
           symbol(module_unload, "cuModuleUnload") &&
           symbol(module_function, "cuModuleGetFunction") &&
           symbol(function_attribute, "cuFuncSetAttribute") &&
           symbol(launch, "cuLaunchKernel") &&
           symbol(event_create, "cuEventCreate") &&
           symbol(event_destroy, "cuEventDestroy") &&
           symbol(event_record, "cuEventRecord") &&
           symbol(event_query, "cuEventQuery") &&
           symbol(event_elapsed, "cuEventElapsedTime") &&
           symbol(error_string, "cuGetErrorString"));
  }

  bool available() const {
    return library && init && device_attribute && retain_primary && release_primary &&
           set_current && synchronize && module_load && module_unload && module_function &&
           function_attribute && launch && event_create && event_destroy && event_record &&
           event_query && event_elapsed && error_string;
  }
  static PearlHashCudaDriverApi& instance() {
    static PearlHashCudaDriverApi api;
    return api;
  }
};

struct PearlHashNvrtcApi {
#if defined(_WIN32)
  HMODULE library = nullptr;
#else
  void* library = nullptr;
#endif
  decltype(&nvrtcCreateProgram) create_program = nullptr;
  decltype(&nvrtcCompileProgram) compile_program = nullptr;
  decltype(&nvrtcGetProgramLogSize) get_log_size = nullptr;
  decltype(&nvrtcGetProgramLog) get_log = nullptr;
  decltype(&nvrtcGetPTXSize) get_ptx_size = nullptr;
  decltype(&nvrtcGetPTX) get_ptx = nullptr;
  decltype(&nvrtcGetCUBINSize) get_cubin_size = nullptr;
  decltype(&nvrtcGetCUBIN) get_cubin = nullptr;
  decltype(&nvrtcGetNumSupportedArchs) get_num_supported_archs = nullptr;
  decltype(&nvrtcGetSupportedArchs) get_supported_archs = nullptr;
  decltype(&nvrtcDestroyProgram) destroy_program = nullptr;
  decltype(&nvrtcGetErrorString) error_string = nullptr;
  std::string error;

  template <typename T>
  bool symbol(T& output, const char* name) {
#if defined(_WIN32)
    output = reinterpret_cast<T>(GetProcAddress(library, name));
#else
    output = reinterpret_cast<T>(dlsym(library, name));
#endif
    if (output) return true;
    error = std::string("missing NVRTC symbol ") + name;
    return false;
  }

  PearlHashNvrtcApi() {
#if defined(_WIN32)
    const char* names[] = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll",
                           "nvrtc64_121_0.dll", "nvrtc64_122_0.dll"};
    for (const char* name : names)
      if ((library = LoadLibraryA(name))) break;
#else
    const char* names[] = {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"};
    for (const char* name : names)
      if ((library = dlopen(name, RTLD_NOW | RTLD_LOCAL))) break;
#endif
    if (!library) {
      error = "NVRTC is not installed";
      return;
    }
    (void)(symbol(create_program, "nvrtcCreateProgram") &&
           symbol(compile_program, "nvrtcCompileProgram") &&
           symbol(get_log_size, "nvrtcGetProgramLogSize") &&
           symbol(get_log, "nvrtcGetProgramLog") &&
           symbol(get_ptx_size, "nvrtcGetPTXSize") &&
           symbol(get_ptx, "nvrtcGetPTX") &&
           symbol(get_cubin_size, "nvrtcGetCUBINSize") &&
           symbol(get_cubin, "nvrtcGetCUBIN") &&
           symbol(get_num_supported_archs, "nvrtcGetNumSupportedArchs") &&
           symbol(get_supported_archs, "nvrtcGetSupportedArchs") &&
           symbol(destroy_program, "nvrtcDestroyProgram") &&
           symbol(error_string, "nvrtcGetErrorString"));
  }

  bool available() const {
    return library && create_program && compile_program && get_log_size && get_log &&
           get_ptx_size && get_ptx && get_cubin_size && get_cubin &&
           get_num_supported_archs && get_supported_archs && destroy_program && error_string;
  }
  static PearlHashNvrtcApi& instance() {
    static PearlHashNvrtcApi api;
    return api;
  }
};

static std::filesystem::path pearlhash_cuda_header_root(
    const char* configured, const std::vector<std::filesystem::path>& candidates,
    const std::filesystem::path& required) {
  if (configured && *configured) {
    const std::filesystem::path path(configured);
    if (std::filesystem::exists(path / required)) return path;
  }
  for (const auto& path : candidates)
    if (std::filesystem::exists(path / required)) return path;
  return {};
}
#include "cuda_search.inc"
#else
struct PearlHashCudaSearch {
  static bool is_cuda(const sycl::device&) { return false; }
  static bool supports(sycl::queue&, std::string* const reason = nullptr) {
    if (reason) *reason = "CUDA source-JIT support is not part of this worker";
    return false;
  }
  bool ensure(sycl::queue&, const Buffers&, int, int, int, int) { return false; }
  void launch(const Buffers&, uint32_t, int, int, bool) {}
  void wait(double) {}
  void release() noexcept {}
  void fail(const std::string&) noexcept {}
};
#endif
