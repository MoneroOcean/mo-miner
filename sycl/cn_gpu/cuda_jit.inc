#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__)
using CnGpuCudaDriverApi = mom::CudaDriverApi;
using CnGpuNvrtcApi = mom::NvrtcApi;
using mom::cuda_check;
using mom::rtc_error;

struct CnGpuCudaExpand {
  CUmodule module = nullptr;
  CUfunction kernel = nullptr;
  CUevent done = nullptr;
  CUcontext context = nullptr;
  CUdevice device = 0;
  bool retained_primary = false;
  bool checked = false;
  bool enabled = false;

  static CUdevice native_device(const sycl::device& device) {
#if defined(MOM_SYCL_ADAPTIVECPP)
    (void)device;
    return 0;
#else
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(device);
#endif
  }
  void release() noexcept {
    CnGpuCudaDriverApi& api = CnGpuCudaDriverApi::instance();
    if (api.basic_available() && context)
      (void)api.set_current(context);
    if (done && api.event_destroy)
      (void)api.event_destroy(done);
    if (module && api.module_unload)
      (void)api.module_unload(module);
    if (retained_primary && api.release_primary)
      (void)api.release_primary(device);
    done = nullptr;
    module = nullptr;
    kernel = nullptr;
    context = nullptr;
    retained_primary = false;
    enabled = false;
  }

  bool ensure(sycl::queue& queue, std::string* const reason = nullptr) {
    const auto unavailable = [&](const std::string& message) {
      if (reason)
        *reason = message;
      return false;
    };
    if (!mom_is_cuda(queue.get_device()))
      return unavailable("the selected SYCL device is not CUDA");
    if (checked)
      return enabled;
    checked = true;
    try {
      CnGpuCudaDriverApi& driver = CnGpuCudaDriverApi::instance();
      CnGpuNvrtcApi& rtc = CnGpuNvrtcApi::instance();
      if (!driver.basic_available())
        throw driver.error;
      if (!rtc.ptx_available())
        throw rtc.error;
      cuda_check(driver, driver.init(0), "cuInit(cn/gpu)");
      device = native_device(queue.get_device());
      int major = 0;
      cuda_check(
          driver,
          driver.device_attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
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
      if (major < 8)
        throw std::string("the native cn/gpu expansion requires compute capability 8.0+");
      const std::string architecture = "--gpu-architecture=compute_80";
      const std::vector<std::string> option_storage{"--std=c++17", architecture,
                                                    "--device-as-default-execution-space"};
      std::vector<const char*> options;
      for (const auto& option : option_storage)
        options.push_back(option.c_str());

      std::ostringstream key;
      key << source << '\0' << architecture << "\0cn-gpu-expand-cuda-v1";
      std::ostringstream filename;
      filename << "cn-gpu-expand-compute80-" << std::hex << std::setw(16) << std::setfill('0')
               << mom::jit_cache::hash(key.str()) << ".ptx";
      const std::filesystem::path root = mom::jit_cache::directory();
      const std::filesystem::path path =
          root.empty() ? std::filesystem::path{} : root / filename.str();
      std::vector<char> code = path.empty() ? std::vector<char>{} : mom::jit_cache::read(path);
      if (!code.empty() && driver.module_load(&module, code.data()) != CUDA_SUCCESS) {
        std::error_code error;
        std::filesystem::remove(path, error);
        code.clear();
      }
      if (code.empty()) {
        nvrtcProgram program = nullptr;
        nvrtcResult status =
            rtc.create_program(&program, source.c_str(), "cn-gpu-expand.cu", 0, nullptr, nullptr);
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
          const std::string message = rtc_error(rtc, status, program, "nvrtcGetPTXSize(cn/gpu)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        code.resize(size);
        status = rtc.get_ptx(program, code.data());
        if (status != NVRTC_SUCCESS) {
          const std::string message = rtc_error(rtc, status, program, "nvrtcGetPTX(cn/gpu)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        (void)rtc.destroy_program(&program);
        cuda_check(driver, driver.module_load(&module, code.data()), "cuModuleLoadData(cn/gpu)");
        mom::jit_cache::write(path, code);
      }
      cuda_check(driver, driver.module_function(&kernel, module, "cn_gpu_expand_cuda"),
                 "cuModuleGetFunction(cn/gpu)");
      cuda_check(driver, driver.event_create(&done, CU_EVENT_DEFAULT), "cuEventCreate(cn/gpu)");
      enabled = true;
    } catch (const std::string& error) {
      release();
      if (reason)
        *reason = error;
    } catch (const std::exception& error) {
      release();
      if (reason)
        *reason = error.what();
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
    cuda_check(api,
               api.launch(kernel, static_cast<unsigned>((total + threads - 1) / threads), 1, 1,
                          threads, 1, 1, 0, nullptr, args, nullptr),
               "cuLaunchKernel(cn/gpu)");
    cuda_check(api, api.event_record(done, nullptr), "cuEventRecord(cn/gpu)");
  }

  void wait() {
    CnGpuCudaDriverApi& api = CnGpuCudaDriverApi::instance();
    for (;;) {
      const CUresult status = api.event_query(done);
      if (status == CUDA_SUCCESS)
        return;
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
  ~CnGpuCudaExpand() {
    release();
  }
};
#else
struct CnGpuCudaExpand {
  bool ensure(sycl::queue&, std::string* const reason = nullptr) {
    if (reason)
      *reason = "CUDA source-JIT support is not part of this worker";
    return false;
  }
  void launch(uint64_t*, uint64_t*, unsigned) {
  }
  void wait() {
  }
  void fail(const std::string&) noexcept {
  }
};
#endif
