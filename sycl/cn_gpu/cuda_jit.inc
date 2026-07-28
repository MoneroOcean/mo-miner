#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__)
struct CnGpuCudaDriverApi {
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
  decltype(&cuModuleLoadData) module_load = nullptr;
  decltype(&cuModuleUnload) module_unload = nullptr;
  decltype(&cuModuleGetFunction) module_function = nullptr;
  decltype(&cuLaunchKernel) launch = nullptr;
  decltype(&cuEventCreate) event_create = nullptr;
  decltype(&cuEventDestroy) event_destroy = nullptr;
  decltype(&cuEventRecord) event_record = nullptr;
  decltype(&cuEventQuery) event_query = nullptr;
  decltype(&cuGetErrorString) error_string = nullptr;
  std::string error;

  template <typename T> bool symbol(T& output, const char* const name) {
#if defined(_WIN32)
    output = reinterpret_cast<T>(GetProcAddress(library, name));
#else
    output = reinterpret_cast<T>(dlsym(library, name));
#endif
    if (output) return true;
    error = std::string("missing CUDA driver symbol ") + name;
    return false;
  }

  CnGpuCudaDriverApi() {
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
           symbol(module_load, "cuModuleLoadData") &&
           symbol(module_unload, "cuModuleUnload") &&
           symbol(module_function, "cuModuleGetFunction") &&
           symbol(launch, "cuLaunchKernel") &&
           symbol(event_create, "cuEventCreate") &&
           symbol(event_destroy, "cuEventDestroy") &&
           symbol(event_record, "cuEventRecord") &&
           symbol(event_query, "cuEventQuery") &&
           symbol(error_string, "cuGetErrorString"));
  }
  bool available() const {
    return library && init && device_attribute && retain_primary && release_primary &&
           set_current && module_load && module_unload && module_function && launch &&
           event_create && event_destroy && event_record && event_query && error_string;
  }
  static CnGpuCudaDriverApi& instance() {
    static CnGpuCudaDriverApi api;
    return api;
  }
};

struct CnGpuNvrtcApi {
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
  decltype(&nvrtcDestroyProgram) destroy_program = nullptr;
  decltype(&nvrtcGetErrorString) error_string = nullptr;
  std::string error;

  template <typename T> bool symbol(T& output, const char* const name) {
#if defined(_WIN32)
    output = reinterpret_cast<T>(GetProcAddress(library, name));
#else
    output = reinterpret_cast<T>(dlsym(library, name));
#endif
    if (output) return true;
    error = std::string("missing NVRTC symbol ") + name;
    return false;
  }

  CnGpuNvrtcApi() {
#if defined(_WIN32)
    const char* const names[] = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll",
                                 "nvrtc64_121_0.dll", "nvrtc64_122_0.dll"};
    for (const char* const name : names)
      if ((library = LoadLibraryA(name))) break;
#else
    const char* const names[] = {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"};
    for (const char* const name : names)
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
           symbol(destroy_program, "nvrtcDestroyProgram") &&
           symbol(error_string, "nvrtcGetErrorString"));
  }
  bool available() const {
    return library && create_program && compile_program && get_log_size && get_log &&
           get_ptx_size && get_ptx && destroy_program && error_string;
  }
  static CnGpuNvrtcApi& instance() {
    static CnGpuNvrtcApi api;
    return api;
  }
};

struct CnGpuCudaExpand {
  CUmodule module = nullptr;
  CUfunction kernel = nullptr;
  CUevent done = nullptr;
  CUcontext context = nullptr;
  CUdevice device = 0;
  bool retained_primary = false;
  bool checked = false;
  bool enabled = false;

  static void cuda_check(CnGpuCudaDriverApi& api, const CUresult status,
                         const char* const operation) {
    if (status == CUDA_SUCCESS) return;
    const char* detail = nullptr;
    if (api.error_string) (void)api.error_string(status, &detail);
    throw std::string(operation) + ": " + (detail ? detail : "CUDA driver error");
  }
  static std::string rtc_error(CnGpuNvrtcApi& api, const nvrtcResult status,
                               nvrtcProgram program, const char* const operation) {
    size_t size = 0;
    std::string log;
    if (program && api.get_log_size(program, &size) == NVRTC_SUCCESS && size) {
      log.resize(size);
      (void)api.get_log(program, log.data());
    }
    return std::string(operation) + ": " + api.error_string(status) +
           (log.empty() ? "" : "\n" + log);
  }
  static CUdevice native_device(const sycl::device& device) {
#if defined(MOM_SYCL_ADAPTIVECPP)
    (void)device;
    return 0;
#else
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(device);
#endif
  }
  static uint64_t hash_text(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : value) {
      hash ^= byte;
      hash *= 1099511628211ULL;
    }
    return hash;
  }
  static std::filesystem::path cache_dir() {
    if (const char* configured = std::getenv("MOM_JIT_CACHE_DIR"); configured && *configured)
      return configured;
#if defined(_WIN32)
    if (const char* base = std::getenv("LOCALAPPDATA"); base && *base)
      return std::filesystem::path(base) / "mom" / "jit";
#else
    if (const char* base = std::getenv("XDG_CACHE_HOME"); base && *base)
      return std::filesystem::path(base) / "mom" / "jit";
    if (const char* home = std::getenv("HOME"); home && *home)
      return std::filesystem::path(home) / ".cache" / "mom" / "jit";
#endif
    return {};
  }
  static std::vector<char> read_cache(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return {};
    const std::streamsize size = stream.tellg();
    if (size <= 0 || size > 64 * 1024 * 1024) return {};
    std::vector<char> code(static_cast<size_t>(size));
    stream.seekg(0);
    if (!stream.read(code.data(), size)) return {};
    return code;
  }
  static void write_cache(const std::filesystem::path& path,
                          const std::vector<char>& code) noexcept {
    try {
      if (path.empty() || code.empty()) return;
      std::filesystem::create_directories(path.parent_path());
      const std::filesystem::path temporary = path.string() + ".tmp";
      {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream.write(code.data(), static_cast<std::streamsize>(code.size()))) return;
      }
      std::error_code error;
      std::filesystem::rename(temporary, path, error);
      if (error) {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
        if (error) std::filesystem::remove(temporary, error);
      }
    } catch (...) {
      // A read-only cache is not a mining failure; the in-process module remains usable.
    }
  }

  void release() noexcept {
    CnGpuCudaDriverApi& api = CnGpuCudaDriverApi::instance();
    if (api.available() && context) (void)api.set_current(context);
    if (done && api.event_destroy) (void)api.event_destroy(done);
    if (module && api.module_unload) (void)api.module_unload(module);
    if (retained_primary && api.release_primary) (void)api.release_primary(device);
    done = nullptr;
    module = nullptr;
    kernel = nullptr;
    context = nullptr;
    retained_primary = false;
    enabled = false;
  }

  bool ensure(sycl::queue& queue, std::string* const reason = nullptr) {
    const auto unavailable = [&](const std::string& message) {
      if (reason) *reason = message;
      return false;
    };
    if (!mom_is_cuda(queue.get_device()))
      return unavailable("the selected SYCL device is not CUDA");
    if (checked) return enabled;
    checked = true;
    try {
      CnGpuCudaDriverApi& driver = CnGpuCudaDriverApi::instance();
      CnGpuNvrtcApi& rtc = CnGpuNvrtcApi::instance();
      if (!driver.available()) throw driver.error;
      if (!rtc.available()) throw rtc.error;
      cuda_check(driver, driver.init(0), "cuInit(cn/gpu)");
      device = native_device(queue.get_device());
      int major = 0;
      cuda_check(driver, driver.device_attribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
        "cuDeviceGetAttribute(cn/gpu major)");
      cuda_check(driver, driver.retain_primary(&context, device),
                 "cuDevicePrimaryCtxRetain(cn/gpu)");
      retained_primary = true;
      cuda_check(driver, driver.set_current(context), "cuCtxSetCurrent(cn/gpu)");

      const std::string source(
#include "cuda_expand_kernel.inc"
      );
      // Keep the cached program forward-compatible with GPUs newer than the installed NVRTC.
      // CUDA 12.x cannot name Blackwell's compute_120 target, but its compute_80 PTX is accepted
      // and optimized for the actual device by current drivers. Older GPUs retain the SYCL path.
      if (major < 8) throw std::string("the native cn/gpu expansion requires compute capability 8.0+");
      const std::string architecture = "--gpu-architecture=compute_80";
      const std::vector<std::string> option_storage{
        "--std=c++17", architecture, "--device-as-default-execution-space"
      };
      std::vector<const char*> options;
      for (const auto& option : option_storage) options.push_back(option.c_str());

      std::ostringstream key;
      key << source << '\0' << architecture << "\0cn-gpu-expand-cuda-v1";
      std::ostringstream filename;
      filename << "cn-gpu-expand-compute80-" << std::hex
               << std::setw(16) << std::setfill('0') << hash_text(key.str()) << ".ptx";
      const std::filesystem::path root = cache_dir();
      const std::filesystem::path path =
        root.empty() ? std::filesystem::path{} : root / filename.str();
      std::vector<char> code = path.empty() ? std::vector<char>{} : read_cache(path);
      if (!code.empty() && driver.module_load(&module, code.data()) != CUDA_SUCCESS) {
        std::error_code error;
        std::filesystem::remove(path, error);
        code.clear();
      }
      if (code.empty()) {
        nvrtcProgram program = nullptr;
        nvrtcResult status = rtc.create_program(
          &program, source.c_str(), "cn-gpu-expand.cu", 0, nullptr, nullptr);
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, program, "nvrtcCreateProgram(cn/gpu)");
        status = rtc.compile_program(program, static_cast<int>(options.size()), options.data());
        if (status != NVRTC_SUCCESS) {
          const std::string message =
            rtc_error(rtc, status, program, "nvrtcCompileProgram(cn/gpu)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        size_t size = 0;
        status = rtc.get_ptx_size(program, &size);
        if (status != NVRTC_SUCCESS) {
          const std::string message =
            rtc_error(rtc, status, program, "nvrtcGetPTXSize(cn/gpu)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        code.resize(size);
        status = rtc.get_ptx(program, code.data());
        if (status != NVRTC_SUCCESS) {
          const std::string message =
            rtc_error(rtc, status, program, "nvrtcGetPTX(cn/gpu)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        (void)rtc.destroy_program(&program);
        cuda_check(driver, driver.module_load(&module, code.data()),
                   "cuModuleLoadData(cn/gpu)");
        write_cache(path, code);
      }
      cuda_check(driver, driver.module_function(&kernel, module, "cn_gpu_expand_cuda"),
                 "cuModuleGetFunction(cn/gpu)");
      cuda_check(driver, driver.event_create(&done, CU_EVENT_DEFAULT),
                 "cuEventCreate(cn/gpu)");
      enabled = true;
    } catch (const std::string& error) {
      release();
      if (reason) *reason = error;
    } catch (const std::exception& error) {
      release();
      if (reason) *reason = error.what();
    }
    return enabled;
  }

  void launch(uint64_t* states, uint64_t* scratchpads, const unsigned batch) {
    CnGpuCudaDriverApi& api = CnGpuCudaDriverApi::instance();
    cuda_check(api, api.set_current(context), "cuCtxSetCurrent(cn/gpu launch)");
    void* state_arg = states;
    void* scratchpad_arg = scratchpads;
    unsigned batch_arg = batch;
    void* args[] = {&state_arg, &scratchpad_arg, &batch_arg};
    constexpr unsigned threads = 32;
    const uint64_t total = static_cast<uint64_t>(batch) * 4096U;
    cuda_check(api, api.launch(
      kernel, static_cast<unsigned>((total + threads - 1) / threads), 1, 1,
      threads, 1, 1, 0, nullptr, args, nullptr), "cuLaunchKernel(cn/gpu)");
    cuda_check(api, api.event_record(done, nullptr), "cuEventRecord(cn/gpu)");
  }

  void wait() {
    CnGpuCudaDriverApi& api = CnGpuCudaDriverApi::instance();
    for (;;) {
      const CUresult status = api.event_query(done);
      if (status == CUDA_SUCCESS) return;
      if (status != CUDA_ERROR_NOT_READY)
        cuda_check(api, status, "cuEventQuery(cn/gpu)");
#if defined(_WIN32)
      mom_sycl_poll_pause();
#else
      std::this_thread::sleep_for(std::chrono::microseconds(100));
#endif
    }
  }
  void fail(const std::string& error) noexcept {
    release();
    checked = true;
    std::fprintf(stderr, "cn/gpu CUDA source-JIT failed, using SYCL: %s\n", error.c_str());
  }
  ~CnGpuCudaExpand() { release(); }
};
#else
struct CnGpuCudaExpand {
  bool ensure(sycl::queue&, std::string* const reason = nullptr) {
    if (reason) *reason = "CUDA source-JIT support is not part of this worker";
    return false;
  }
  void launch(uint64_t*, uint64_t*, unsigned) {}
  void wait() {}
  void fail(const std::string&) noexcept {}
};
#endif
