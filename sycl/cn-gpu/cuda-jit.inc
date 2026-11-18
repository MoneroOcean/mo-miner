#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__)
using CnGpuCudaDriverApi = mom::CudaDriverApi;
#if !defined(MOM_EMBEDDED_CUDA_IMAGES)
using CnGpuNvrtcApi = mom::NvrtcApi;
using mom::rtc_error;
#endif
using mom::cuda_check;

struct CnGpuCudaExpand {
  CUmodule module = nullptr;
  CUfunction kernel = nullptr;
  CUevent done = nullptr;
  CUcontext context = nullptr;
  CUdevice device = 0;
  bool retained_primary = false;
  bool checked = false;
  bool enabled = false;

  // Use the selected SYCL device's native ordinal, never ordinal zero in a multi-GPU worker.
  static CUdevice native_device(const sycl::device& device) {
#if defined(MOM_SYCL_ADAPTIVECPP)
    return sycl::get_native<sycl::backend::cuda>(device);
#else
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(device);
#endif
  }
  // Release owned driver resources after selecting their retained context.
  void release() noexcept {
    // Empty fallback state must not instantiate a CUDA driver loader on another vendor.
    if (context || done || module || retained_primary) {
      CnGpuCudaDriverApi& api = CnGpuCudaDriverApi::instance();
      if (api.basic_available() && context)
        (void)api.set_current(context);
      if (done && api.event_destroy)
        (void)api.event_destroy(done);
      if (module && api.module_unload)
        (void)api.module_unload(module);
      if (retained_primary && api.release_primary)
        (void)api.release_primary(device);
    }
    done = nullptr;
    module = nullptr;
    kernel = nullptr;
    context = nullptr;
    retained_primary = false;
    enabled = false;
  }

  // Try native expansion once; unavailable CUDA acceleration retains the SYCL fallback.
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
#if !defined(MOM_EMBEDDED_CUDA_IMAGES)
      CnGpuNvrtcApi& rtc = CnGpuNvrtcApi::instance();
#endif
      if (!driver.basic_available())
        throw driver.error;
      if (!driver.synchronize)
        throw std::string("missing CUDA driver symbol cuCtxSynchronize");
#if !defined(MOM_EMBEDDED_CUDA_IMAGES)
      if (!rtc.ptx_available())
        throw rtc.error;
#endif
      cuda_check(driver, driver.init(0), "cuInit(cn/gpu)");
      device = native_device(queue.get_device());
      int major = 0, minor = 0;
      cuda_check(
          driver,
          driver.device_attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
          "cuDeviceGetAttribute(cn/gpu major)");
      cuda_check(driver,
          driver.device_attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
          "cuDeviceGetAttribute(cn/gpu minor)");
      cuda_check(driver, driver.retain_primary(&context, device),
                 "cuDevicePrimaryCtxRetain(cn/gpu)");
      retained_primary = true;
      cuda_check(driver, driver.set_current(context), "cuCtxSetCurrent(cn/gpu)");

#if defined(MOM_EMBEDDED_CUDA_IMAGES)
      // Native images keep the tested SM80 floor and reject unknown capability encodings.
      if (major < 8)
        throw std::string("the native cn/gpu expansion requires compute capability 8.0+");
      if (major > 99 || minor < 0 || minor > 9)
        throw std::string("invalid cn/gpu CUDA device capability");
      const int device_arch = major * 10 + minor;
      const auto candidates = mom::cuda_images::select_embedded("cn-expand", device_arch);
      if (!candidates.primary.data)
        throw std::string("no compatible embedded cn/gpu expansion image");
      cuda_check(driver,
                 mom::cuda_load_image_candidates(
                     candidates, &module, [&](CUmodule* loaded, const void* image) {
                       return driver.module_load(loaded, image);
                     }),
                 "cuModuleLoadData(cn/gpu)");
#else
      const std::string source(
#include "cuda-expand-kernel.inc"
      );
      // Native code avoids a newer compiler's PTX ISA requiring a newer driver.
      // Unknown future devices/old compilers retain the existing compute80 PTX
      // route; GPUs below SM80 still use the complete SYCL implementation.
      if (major < 8)
        throw std::string("the native cn/gpu expansion requires compute capability 8.0+");
      if (major > 99 || minor < 0 || minor > 9)
        throw std::string("invalid cn/gpu CUDA device capability");
      const int device_arch = major * 10 + minor;
      bool emit_cubin = false;
      if (rtc.cubin_available()) {
        int count = 0;
        auto status = rtc.get_num_supported_archs(&count);
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, nullptr, "nvrtcGetNumSupportedArchs(cn/gpu)");
        if (count <= 0 || count > 128)
          throw std::string("invalid cn/gpu NVRTC architecture count");
        std::vector<int> architectures(static_cast<size_t>(count));
        status = rtc.get_supported_archs(architectures.data());
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, nullptr, "nvrtcGetSupportedArchs(cn/gpu)");
        emit_cubin = std::find(architectures.begin(), architectures.end(), device_arch) != architectures.end();
      }
      const std::string target = emit_cubin ? "sm_" + std::to_string(device_arch) : "compute_80";
      const std::string architecture = "--gpu-architecture=" + target;
      const std::vector<std::string> option_storage{"--std=c++17", architecture,
                                                    "--device-as-default-execution-space"};
      std::vector<const char*> options;
      for (const auto& option : option_storage)
        options.push_back(option.c_str());

      std::ostringstream key;
      key << source << '\0' << architecture << "\0cn-gpu-expand-cuda-v2";
      std::ostringstream filename;
      filename << "cn-gpu-expand-" << target << '-' << std::hex << std::setw(16) << std::setfill('0')
               << mom::jit_cache::hash(key.str()) << (emit_cubin ? ".cubin" : ".ptx");
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
        mom::NvrtcProgram program(rtc);
        nvrtcResult status =
            rtc.create_program(program.address(), source.c_str(), "cn-gpu-expand.cu", 0, nullptr, nullptr);
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, program.get(), "nvrtcCreateProgram(cn/gpu)");
        status = rtc.compile_program(program.get(), static_cast<int>(options.size()), options.data());
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, program.get(), "nvrtcCompileProgram(cn/gpu)");
        size_t size = 0;
        status = emit_cubin ? rtc.get_cubin_size(program.get(), &size) : rtc.get_ptx_size(program.get(), &size);
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, program.get(), emit_cubin ? "nvrtcGetCUBINSize(cn/gpu)" : "nvrtcGetPTXSize(cn/gpu)");
        if (!size || size > 128 * 1024 * 1024)
          throw std::string("invalid cn/gpu NVRTC output size");
        code.resize(size);
        status = emit_cubin ? rtc.get_cubin(program.get(), code.data()) : rtc.get_ptx(program.get(), code.data());
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, program.get(), emit_cubin ? "nvrtcGetCUBIN(cn/gpu)" : "nvrtcGetPTX(cn/gpu)");
        program.reset();
        cuda_check(driver, driver.module_load(&module, code.data()), "cuModuleLoadData(cn/gpu)");
        mom::jit_cache::write(path, code);
      }
#endif
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

  // Launch native expansion and record the completion event consumed before SYCL resumes.
  void launch(uint64_t* states, uint64_t* scratchpads, const unsigned batch) {
    CnGpuCudaDriverApi& api = CnGpuCudaDriverApi::instance();
    cuda_check(api, api.set_current(context), "cuCtxSetCurrent(cn/gpu launch)");
    void* state_arg = states;
    void* scratchpad_arg = scratchpads;
    unsigned batch_arg = batch;
    void* args[] = {&state_arg, &scratchpad_arg, &batch_arg};
    constexpr unsigned threads = 32;
    const uint64_t total = static_cast<uint64_t>(batch) * 4096U;
    mom::cuda_check_submitted(
        api, api.launch(kernel, static_cast<unsigned>((total + threads - 1) / threads), 1, 1,
                        threads, 1, 1, 0, nullptr, args, nullptr),
        "cuLaunchKernel(cn/gpu)");
    mom::cuda_check_submitted(api, api.event_record(done, nullptr), "cuEventRecord(cn/gpu)");
  }

  // Poll native completion without busy-spinning and propagate driver faults.
  void wait() {
    CnGpuCudaDriverApi& api = CnGpuCudaDriverApi::instance();
    for (;;) {
      const CUresult status = api.event_query(done);
      if (status == CUDA_SUCCESS)
        return;
      if (status != CUDA_ERROR_NOT_READY)
        mom::cuda_check_submitted(api, status, "cuEventQuery(cn/gpu)");
#if defined(_WIN32)
      mom_sycl_poll_pause();
#else
      std::this_thread::sleep_for(std::chrono::microseconds(100));
#endif
    }
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
};
#endif
