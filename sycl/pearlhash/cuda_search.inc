
struct PearlHashCudaSearch {
  CUmodule module = nullptr;
  CUfunction gemm_kernel = nullptr;
  CUfunction hash_kernel = nullptr;
  CUevent started = nullptr;
  CUevent gemm_done = nullptr;
  CUevent done = nullptr;
  CUcontext context = nullptr;
  CUdevice device = 0;
  bool retained_primary = false;
  bool checked = false;
  bool enabled = false;
  unsigned stats_printed = 0;

  static bool is_cuda(const sycl::device& device) {
  #if defined(MOM_SYCL_ADAPTIVECPP)
    return device.get_backend() == sycl::backend::cuda;
  #else
    return device.get_backend() == sycl::backend::ext_oneapi_cuda;
  #endif
  }
  static void cuda_check(PearlHashCudaDriverApi& api, CUresult status,
                         const char* operation) {
    if (status == CUDA_SUCCESS) return;
    const char* detail = nullptr;
    if (api.error_string) (void)api.error_string(status, &detail);
    throw std::string(operation) + ": " + (detail ? detail : "CUDA driver error");
  }
  static std::string rtc_error(PearlHashNvrtcApi& api, nvrtcResult status,
                               nvrtcProgram program, const char* operation) {
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
  static bool supports(sycl::queue& q, std::string* const reason = nullptr) {
    const auto unavailable = [&](const std::string& message) {
      if (reason) *reason = message;
      return false;
    };
    if (!is_cuda(q.get_device()))
      return unavailable("the selected SYCL device is not CUDA");
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
    if (!api.available()) return unavailable(api.error);
    if (api.init(0) != CUDA_SUCCESS) return unavailable("cuInit failed");
    int major = 0;
    if (api.device_attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             native_device(q.get_device())) != CUDA_SUCCESS)
      return unavailable("the CUDA compute capability could not be read");
    if (major < 8) return unavailable("the CUTE int8 mainloop requires compute capability 8.0+");
    return true;
  }

  void release() noexcept {
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
    if (api.available() && context) (void)api.set_current(context);
    if (done && api.event_destroy) (void)api.event_destroy(done);
    if (gemm_done && api.event_destroy) (void)api.event_destroy(gemm_done);
    if (started && api.event_destroy) (void)api.event_destroy(started);
    if (module && api.module_unload) (void)api.module_unload(module);
    if (retained_primary && api.release_primary) (void)api.release_primary(device);
    done = nullptr;
    gemm_done = nullptr;
    started = nullptr;
    module = nullptr;
    gemm_kernel = nullptr;
    hash_kernel = nullptr;
    context = nullptr;
    retained_primary = false;
    enabled = false;
  }

  bool ensure(sycl::queue& q, const Buffers& b, int m, int n, int k, int rank) {
    if (!supports(q) || m % 128 || n % 256 || k != 4096 || rank != 256)
      return false;
    // The two-stage native path requires a complete transcript surface. Never launch the CUTE
    // producer when an oversized or failed USM allocation left it null; the shared SYCL path can
    // still execute without that surface.
    if (!b.transcript) return false;
    if (checked) return enabled;
    checked = true;
    try {
      PearlHashCudaDriverApi& driver = PearlHashCudaDriverApi::instance();
      PearlHashNvrtcApi& rtc = PearlHashNvrtcApi::instance();
      if (!rtc.available()) throw rtc.error;

      device = native_device(q.get_device());
      cuda_check(driver, driver.retain_primary(&context, device),
                 "cuDevicePrimaryCtxRetain(PearlHash)");
      retained_primary = true;
      cuda_check(driver, driver.set_current(context), "cuCtxSetCurrent(PearlHash)");

      std::vector<std::filesystem::path> cuda_candidates;
      if (const char* path = std::getenv("CUDA_PATH"); path && *path)
        cuda_candidates.emplace_back(path);
      if (const char* path = std::getenv("CUDA_HOME"); path && *path)
        cuda_candidates.emplace_back(path);
#if defined(_WIN32)
      cuda_candidates.emplace_back("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0");
      cuda_candidates.emplace_back("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6");
#else
      cuda_candidates.emplace_back("/usr/local/cuda");
      cuda_candidates.emplace_back("/opt/nvidia-cuda-ubuntu/usr");
#endif
      const std::filesystem::path cuda_include = pearlhash_cuda_header_root(
        nullptr, [&] {
          std::vector<std::filesystem::path> paths;
          for (const auto& root : cuda_candidates) paths.push_back(root / "include");
          return paths;
        }(), "cuda_runtime.h");

      std::vector<std::filesystem::path> cutlass_candidates;
#if defined(_WIN32)
      if (const char* program_data = std::getenv("ProgramData"); program_data && *program_data)
        cutlass_candidates.emplace_back(
          std::filesystem::path(program_data) / "mom" / "cutlass" / "include");
#else
      cutlass_candidates.emplace_back("/opt/mom/cutlass/include");
      cutlass_candidates.emplace_back("/usr/local/include/cutlass");
#endif
      const std::filesystem::path cutlass_include = pearlhash_cuda_header_root(
        std::getenv("MOM_CUTLASS_INCLUDE_DIR"), cutlass_candidates, "cute/tensor.hpp");

      std::vector<std::filesystem::path> cccl_candidates;
      if (!cuda_include.empty()) {
        cccl_candidates.push_back(cuda_include);
        cccl_candidates.push_back(cuda_include / "cccl");
      }
      const std::filesystem::path cccl_include = pearlhash_cuda_header_root(
        std::getenv("MOM_CCCL_INCLUDE_DIR"), cccl_candidates, "cuda/std/cstdint");
      if (cuda_include.empty()) throw std::string("CUDA runtime headers are not installed");
      if (cutlass_include.empty()) throw std::string("CUTLASS headers are not installed");
      if (cccl_include.empty()) throw std::string("CCCL headers are not installed");

      int capability_major = 0, capability_minor = 0;
      cuda_check(driver, driver.device_attribute(
        &capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
        "cuDeviceGetAttribute(PearlHash compute capability major)");
      cuda_check(driver, driver.device_attribute(
        &capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
        "cuDeviceGetAttribute(PearlHash compute capability minor)");
      const int device_arch = capability_major * 10 + capability_minor;
      int arch_count = 0;
      nvrtcResult arch_status = rtc.get_num_supported_archs(&arch_count);
      if (arch_status != NVRTC_SUCCESS || arch_count <= 0)
        throw rtc_error(rtc, arch_status, nullptr, "nvrtcGetNumSupportedArchs(PearlHash)");
      std::vector<int> supported_archs(static_cast<size_t>(arch_count));
      arch_status = rtc.get_supported_archs(supported_archs.data());
      if (arch_status != NVRTC_SUCCESS)
        throw rtc_error(rtc, arch_status, nullptr, "nvrtcGetSupportedArchs(PearlHash)");
      int target_arch = 0;
      for (const int arch : supported_archs)
        if (arch <= device_arch && arch > target_arch) target_arch = arch;
      if (target_arch < 80)
        throw std::string("NVRTC does not support PearlHash's required compute capability 8.0+");
      // A cubin is valid only for the exact architecture it names. When NVRTC knows the selected
      // GPU, emit final machine code locally and avoid a second driver-JIT pass; otherwise retain
      // forward-compatible PTX for newer GPUs (for example CUDA 12.6 on a Blackwell card).
      const bool emit_cubin = target_arch == device_arch;

      const std::string source(
#include "cuda_kernel.inc"
      );
      std::vector<std::string> option_storage{
        "--std=c++17",
        "--gpu-architecture=" + std::string(emit_cubin ? "sm_" : "compute_") +
          std::to_string(target_arch),
        "--include-path=" + cutlass_include.generic_string(),
        "--include-path=" + cuda_include.generic_string(),
        "--include-path=" + cccl_include.generic_string(),
        "--device-as-default-execution-space",
        "--use_fast_math"
      };
      std::vector<const char*> options;
      options.reserve(option_storage.size());
      for (const auto& option : option_storage) options.push_back(option.c_str());

      std::ostringstream key_stream;
      key_stream << source;
      for (const auto& option : option_storage) key_stream << '\0' << option;
      key_stream << "\0pearlhash-cuda-cache-v1";
      std::ostringstream filename;
      filename << "pearlhash-cuda-" << (emit_cubin ? "sm" : "compute") << target_arch << "-"
               << std::hex << std::setw(16) << std::setfill('0')
               << pearlhash_jit_hash(key_stream.str()) << (emit_cubin ? ".cubin" : ".ptx");
      const std::filesystem::path cache_dir = pearlhash_jit_cache_dir();
      const std::filesystem::path cache_path =
        cache_dir.empty() ? std::filesystem::path{} : cache_dir / filename.str();
      std::vector<char> code = cache_path.empty()
        ? std::vector<char>{} : pearlhash_read_cache(cache_path);
      if (!code.empty() && driver.module_load(&module, code.data()) != CUDA_SUCCESS) {
        std::error_code error;
        std::filesystem::remove(cache_path, error);
        code.clear();
      }
      if (code.empty()) {
        nvrtcProgram program = nullptr;
        nvrtcResult status = rtc.create_program(
          &program, source.c_str(), "pearlhash-cute.cu", 0, nullptr, nullptr);
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, program, "nvrtcCreateProgram(PearlHash)");
        status = rtc.compile_program(program, static_cast<int>(options.size()), options.data());
        if (status != NVRTC_SUCCESS) {
          const std::string message =
            rtc_error(rtc, status, program, "nvrtcCompileProgram(PearlHash)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        size_t code_size = 0;
        status = emit_cubin
          ? rtc.get_cubin_size(program, &code_size)
          : rtc.get_ptx_size(program, &code_size);
        if (status != NVRTC_SUCCESS) {
          const std::string message =
            rtc_error(rtc, status, program,
                      emit_cubin ? "nvrtcGetCUBINSize(PearlHash)" : "nvrtcGetPTXSize(PearlHash)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        code.resize(code_size);
        status = emit_cubin
          ? rtc.get_cubin(program, code.data())
          : rtc.get_ptx(program, code.data());
        if (status != NVRTC_SUCCESS) {
          const std::string message =
            rtc_error(rtc, status, program,
                      emit_cubin ? "nvrtcGetCUBIN(PearlHash)" : "nvrtcGetPTX(PearlHash)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        (void)rtc.destroy_program(&program);
        cuda_check(driver, driver.module_load(&module, code.data()),
                   "cuModuleLoadData(PearlHash)");
        pearlhash_write_cache(cache_path, code);
      }
      cuda_check(driver, driver.module_function(&gemm_kernel, module, "pearlhash_cuda_gemm"),
                 "cuModuleGetFunction(PearlHash GEMM)");
      cuda_check(driver, driver.module_function(&hash_kernel, module, "pearlhash_cuda_hash"),
                 "cuModuleGetFunction(PearlHash hash)");
      constexpr int shared_bytes = 98304;
      cuda_check(driver, driver.function_attribute(
        gemm_kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes),
        "cuFuncSetAttribute(PearlHash shared memory)");
      cuda_check(driver, driver.function_attribute(
        gemm_kernel, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100),
        "cuFuncSetAttribute(PearlHash carveout)");
      cuda_check(driver, driver.event_create(&started, CU_EVENT_DEFAULT),
                 "cuEventCreate(PearlHash start)");
      cuda_check(driver, driver.event_create(&gemm_done, CU_EVENT_DEFAULT),
                 "cuEventCreate(PearlHash GEMM done)");
      cuda_check(driver, driver.event_create(&done, CU_EVENT_DEFAULT),
                 "cuEventCreate(PearlHash done)");
      enabled = true;
    } catch (const std::string& error) {
      release();
      std::fprintf(stderr, "PearlHash CUDA source-JIT unavailable, using the best SYCL kernel: %s\n",
                   error.c_str());
    } catch (const std::exception& error) {
      release();
      std::fprintf(stderr, "PearlHash CUDA source-JIT unavailable, using the best SYCL kernel: %s\n",
                   error.what());
    }
    return enabled;
  }

  void fail(const std::string& error) noexcept {
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
    if (api.available() && context) {
      (void)api.set_current(context);
      (void)api.synchronize();
    }
    release();
    checked = true;
    std::fprintf(stderr, "PearlHash CUDA source-JIT failed, using the best SYCL kernel: %s\n",
                 error.c_str());
  }

  void launch(const Buffers& b, uint32_t seed, int m, int n, bool debug) {
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
    cuda_check(api, api.set_current(context), "cuCtxSetCurrent(PearlHash launch)");
    void* a = b.Ap;
    void* bp = b.Bp;
    void* transcript = b.transcript;
    void* key = b.cA;
    void* target = b.target;
    void* result = b.result;
    int k = 4096;
    int debug_int = debug ? 1 : 0;
    void* gemm_args[] = {&a, &bp, &m, &n, &k, &transcript};
    void* hash_args[] = {&transcript, &key, &target, &result, &seed,
                         &m, &n, &debug_int};
    constexpr unsigned threads = 256;
    constexpr unsigned shared_bytes = 98304;
    cuda_check(api, api.event_record(started, nullptr),
               "cuEventRecord(PearlHash start)");
    cuda_check(api, api.launch(gemm_kernel, static_cast<unsigned>(m / 128),
                               static_cast<unsigned>(n / 256), 1,
                               threads, 1, 1, shared_bytes, nullptr,
                               gemm_args, nullptr),
               "cuLaunchKernel(PearlHash GEMM)");
    cuda_check(api, api.event_record(gemm_done, nullptr),
               "cuEventRecord(PearlHash GEMM done)");
    const uint64_t tiles = static_cast<uint64_t>(m / 16) *
                           static_cast<uint64_t>(n / 16);
    cuda_check(api, api.launch(hash_kernel,
                               static_cast<unsigned>((tiles + threads - 1) / threads),
                               1, 1, threads, 1, 1, 0, nullptr,
                               hash_args, nullptr),
               "cuLaunchKernel(PearlHash hash)");
    cuda_check(api, api.event_record(done, nullptr), "cuEventRecord(PearlHash done)");
  }

  void wait(double learned_us) {
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
#if !defined(_WIN32)
    const auto poll = learned_us > 20000.0 ? std::chrono::microseconds(500)
                                           : std::chrono::microseconds(100);
#else
    (void)learned_us;
#endif
    for (;;) {
      const CUresult status = api.event_query(done);
      if (status == CUDA_SUCCESS) {
        if (std::getenv("MOM_PEARLHASH_STATS") && stats_printed++ < 4) {
          float gemm_ms = 0.0f, hash_ms = 0.0f;
          cuda_check(api, api.event_elapsed(&gemm_ms, started, gemm_done),
                     "cuEventElapsedTime(PearlHash CUDA GEMM)");
          cuda_check(api, api.event_elapsed(&hash_ms, gemm_done, done),
                     "cuEventElapsedTime(PearlHash CUDA hash)");
          std::fprintf(stderr, "PEARLHASH_CUDA_STATS gemm=%.3fms hash=%.3fms\n",
                       gemm_ms, hash_ms);
        }
        return;
      }
      if (status != CUDA_ERROR_NOT_READY)
        cuda_check(api, status, "cuEventQuery(PearlHash CUDA)");
#if defined(_WIN32)
      mom_sycl_poll_pause();
#else
      std::this_thread::sleep_for(poll);
#endif
    }
  }
  ~PearlHashCudaSearch() { release(); }
};
