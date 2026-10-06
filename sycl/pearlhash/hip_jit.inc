struct PearlHashHipSearch {
#if defined(MOM_SYCL_HAS_HIP)
  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
  hipFunction_t hash_kernel = nullptr;
  hipEvent_t started = nullptr;
  hipEvent_t wmma_done = nullptr;
  hipEvent_t done = nullptr;
  hipStream_t native_stream = nullptr;
  hipDevice_t native_device = -1;
  uint32_t* transcript = nullptr;
  size_t transcript_bytes = 0;
  bool checked = false;
  bool enabled = false;
  unsigned stats_printed = 0;
  int tile_er = 4;
  int tile_ec = 4;
  int k_unroll = 2;
  int compiled_k = 0;
  int compiled_rank = 0;
  unsigned wmma_threads = 256;

  static void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess)
      throw std::string(operation) + ": " + hipGetErrorString(status);
  }
  static std::string rtc_error(mom::HiprtcApi& api, hiprtcResult status, hiprtcProgram program,
                               const char* operation) {
    size_t size = 0;
    std::string log;
    if (program && api.get_log_size(program, &size) == HIPRTC_SUCCESS && size) {
      log.resize(size);
      (void)api.get_log(program, log.data());
    }
    return std::string(operation) + ": " + api.error_string(status) +
           (log.empty() ? "" : "\n" + log);
  }
  static bool supports(sycl::queue& q, std::string* const reason = nullptr) {
    const auto unavailable = [&](const std::string& message) {
      if (reason)
        *reason = message;
      return false;
    };
    if (!mom_is_hip(q.get_device()))
      return unavailable("the selected SYCL device is not HIP");
    hipDevice_t device = -1;
    hipStream_t stream = nullptr;
    hipDeviceProp_t properties{};
    mom::hip_queue_identity(q, device, stream);
    const hipError_t properties_status = hipGetDeviceProperties(&properties, device);
    if (properties_status != hipSuccess)
      return unavailable(std::string("hipGetDeviceProperties: ") +
                         hipGetErrorString(properties_status));
    if (mom::amd::has_gfx12_int8_wmma(properties.gcnArchName))
      return true;
    return unavailable(
        std::string("gfx12 WMMA is unavailable on ") +
        (properties.gcnArchName[0] ? properties.gcnArchName : "an unknown architecture"));
  }
  static bool supports_shape(const int m, const int n, const int k, const int rank) {
    constexpr int rows_per_wave = 16 * 4;
    constexpr int columns_per_wave = 16 * 4;
    constexpr int waves_per_launch = 256 / 32;
    const uint64_t waves = m > 0 && n > 0
        ? static_cast<uint64_t>(m / rows_per_wave) * static_cast<uint64_t>(n / columns_per_wave)
        : 0;
    return m > 0 && n > 0 && k > 0 && rank > 0 && m % rows_per_wave == 0 &&
        n % columns_per_wave == 0 && k % 16 == 0 &&
        rank % 16 == 0 && k % rank == 0 && k / rank == 16 &&
        waves % waves_per_launch == 0;
  }
  void release() noexcept {
    if (native_device >= 0)
      (void)hipSetDevice(native_device);
    if (done)
      (void)hipEventDestroy(done);
    if (wmma_done)
      (void)hipEventDestroy(wmma_done);
    if (started)
      (void)hipEventDestroy(started);
    if (transcript)
      (void)hipFree(transcript);
    if (module)
      (void)hipModuleUnload(module);
    done = nullptr;
    wmma_done = nullptr;
    started = nullptr;
    native_stream = nullptr;
    native_device = -1;
    transcript = nullptr;
    transcript_bytes = 0;
    module = nullptr;
    kernel = nullptr;
    hash_kernel = nullptr;
    compiled_k = 0;
    compiled_rank = 0;
    enabled = false;
    checked = false;
  }
  void activate() const {
    hip_check(hipSetDevice(native_device), "hipSetDevice(PearlHash)");
  }
  bool ensure(sycl::queue& q, int m, int n, int k, int rank) {
    if (!mom_is_hip(q.get_device()))
      return false;
    hipDevice_t queue_device = -1;
    hipStream_t queue_stream = nullptr;
    mom::hip_queue_identity(q, queue_device, queue_stream);
    if (native_device >= 0 && native_device != queue_device)
      return false;
    native_device = queue_device;
    native_stream = queue_stream;
    activate();
    hipDeviceProp_t queue_properties{};
    if (hipGetDeviceProperties(&queue_properties, native_device) != hipSuccess ||
        !mom::amd::has_gfx12_int8_wmma(queue_properties.gcnArchName))
      return false;
    // This source-JIT path implements the ratio-16 transcript. Other valid profiles use the shared
    // SYCL path, so they must be rejected before its duplicate WMMA matrices are allocated.
    if (!supports_shape(m, n, k, rank)) {
      return false;
    }
    if (checked && (compiled_k != k || compiled_rank != rank)) {
      return false;
    }
    if (!checked) {
      try {
        checked = true;
        std::string arch = queue_properties.gcnArchName;
        if (const size_t colon = arch.find(':'); colon != std::string::npos)
          arch.resize(colon);
        const std::string source(
#include "hip_kernel.inc"
        );
        const std::string arch_option = "--offload-arch=" + arch;
        const std::string er_option = "-DPEARLHASH_ER=" + std::to_string(tile_er);
        const std::string ec_option = "-DPEARLHASH_EC=" + std::to_string(tile_ec);
        const std::string k_option = "-DPEARLHASH_K=" + std::to_string(k);
        const std::string rank_option = "-DPEARLHASH_RANK=" + std::to_string(rank);
        const std::string unroll_option = "-DPEARLHASH_K_UNROLL=" + std::to_string(k_unroll);
        const char* options[] = {"-O3",
                                 "--std=c++17",
                                 arch_option.c_str(),
                                 er_option.c_str(),
                                 ec_option.c_str(),
                                 k_option.c_str(),
                                 rank_option.c_str(),
                                 unroll_option.c_str()};
        std::ostringstream key_stream;
        key_stream << source << '\0' << arch;
        for (const char* option : options) key_stream << '\0' << option;
        key_stream << "\0pearlhash-hip-cache-v1";
        std::ostringstream filename;
        filename << "pearlhash-" << arch << '-' << std::hex << std::setw(16) << std::setfill('0')
                 << mom::jit_cache::hash(key_stream.str()) << ".hsaco";
        const std::filesystem::path cache_dir = mom::jit_cache::directory();
        const std::filesystem::path cache_path =
            cache_dir.empty() ? std::filesystem::path{} : cache_dir / filename.str();
        std::vector<char> code =
            cache_path.empty() ? std::vector<char>{} : mom::jit_cache::read(cache_path);
        if (!code.empty() && hipModuleLoadData(&module, code.data()) != hipSuccess) {
          std::error_code error;
          std::filesystem::remove(cache_path, error);
          code.clear();
        }
        if (code.empty()) {
          mom::HiprtcApi& rtc = mom::HiprtcApi::instance();
          if (!rtc.available())
            throw rtc.error;
          mom::HiprtcProgram program(rtc);
          hiprtcResult status = rtc.create_program(program.address(), source.c_str(), "pearlhash-wmma.hip",
                                                   0, nullptr, nullptr);
          if (status != HIPRTC_SUCCESS)
            throw rtc_error(rtc, status, program.get(), "hiprtcCreateProgram(PearlHash)");
          status = rtc.compile_program(program.get(), 8, options);
          if (status != HIPRTC_SUCCESS)
            throw rtc_error(rtc, status, program.get(), "hiprtcCompileProgram(PearlHash)");
          size_t code_size = 0;
          status = rtc.get_code_size(program.get(), &code_size);
          if (status != HIPRTC_SUCCESS)
            throw rtc_error(rtc, status, program.get(), "hiprtcGetCodeSize(PearlHash)");
          code.resize(code_size);
          status = rtc.get_code(program.get(), code.data());
          if (status != HIPRTC_SUCCESS)
            throw rtc_error(rtc, status, program.get(), "hiprtcGetCode(PearlHash)");
          program.reset();
          hip_check(hipModuleLoadData(&module, code.data()), "hipModuleLoadData(PearlHash)");
          mom::jit_cache::write(cache_path, code);
        }
        hip_check(hipModuleGetFunction(&kernel, module, "pearlhash_wmma_transcript"), "hipModuleGetFunction(PearlHash WMMA)");
        hip_check(hipModuleGetFunction(&hash_kernel, module, "pearlhash_hash_search"), "hipModuleGetFunction(PearlHash hash)");
        hip_check(hipEventCreate(&started), "hipEventCreate(PearlHash start)");
        hip_check(hipEventCreate(&wmma_done), "hipEventCreate(PearlHash WMMA done)");
        hip_check(hipEventCreate(&done), "hipEventCreate(PearlHash done)");
        compiled_k = k;
        compiled_rank = rank;
        enabled = true;
      } catch (const std::string& error) {
        release();
        checked = true;
        std::fprintf(stderr, "PearlHash gfx12 WMMA unavailable, using portable dp4a: %s\n",
                     error.c_str());
        return false;
      } catch (const std::exception& error) {
        release();
        checked = true;
        std::fprintf(stderr, "PearlHash gfx12 WMMA unavailable, using portable dp4a: %s\n",
                     error.what());
        return false;
      }
    }
    if (!enabled) return false;
    const size_t bytes = (size_t)(m / 16) * (size_t)(n / 16) * 16u * sizeof(uint32_t);
    if (bytes != transcript_bytes) {
      try {
        if (transcript) hip_check(hipFree(transcript), "hipFree(PearlHash transcript)");
        transcript = nullptr;
        transcript_bytes = 0;
        hip_check(hipMalloc(reinterpret_cast<void**>(&transcript), bytes),
                  "hipMalloc(PearlHash transcript)");
        transcript_bytes = bytes;
      } catch (const std::string& error) {
        release();
        checked = true;
        std::fprintf(stderr,
                     "PearlHash gfx12 transcript allocation unavailable, using portable dp4a: %s\n",
                     error.c_str());
        return false;
      } catch (const std::exception& error) {
        release();
        checked = true;
        std::fprintf(stderr,
                     "PearlHash gfx12 transcript allocation unavailable, using portable dp4a: %s\n",
                     error.what());
        return false;
      }
    }
    return true;
  }
  void launch(const Buffers& b, uint32_t seed, int m, int n, int k, int rank, bool debug) {
    activate();
    const unsigned threads = wmma_threads;
    const unsigned waves_per_block = threads / 32;
    const uint64_t waves = (uint64_t)(m / (16 * tile_er)) * (uint64_t)(n / (16 * tile_ec));
    const unsigned blocks = (unsigned)((waves + waves_per_block - 1) / waves_per_block);
    void* A = b.ApWmma;
    void* B = b.BpWmma;
    void* transcript_arg = transcript;
    void* key = b.cA;
    void* target = b.target;
    void* result = b.result;
    int debug_int = debug ? 1 : 0;
    void* wmma_args[] = {&A, &B, &transcript_arg, &m, &n};
    hip_check(hipEventRecord(started, native_stream), "hipEventRecord(PearlHash start)");
    hip_check(hipModuleLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, native_stream, wmma_args, nullptr),
              "hipModuleLaunchKernel(PearlHash WMMA)");
    hip_check(hipEventRecord(wmma_done, native_stream), "hipEventRecord(PearlHash WMMA done)");
    const uint64_t tiles = (uint64_t)(m / 16) * (uint64_t)(n / 16);
    constexpr unsigned hash_threads = 256;
    const unsigned hash_blocks = (unsigned)((tiles + hash_threads - 1) / hash_threads);
    void* hash_args[] = {&transcript_arg, &key, &target, &result, &seed, &m, &n, &debug_int};
    hip_check(hipModuleLaunchKernel(hash_kernel, hash_blocks, 1, 1, hash_threads, 1, 1, 0, native_stream,
                                    hash_args, nullptr),
              "hipModuleLaunchKernel(PearlHash hash)");
    hip_check(hipEventRecord(done, native_stream), "hipEventRecord(PearlHash)");
  }
  // hipEventSynchronize can spin a complete CPU core for the many-second production GEMM. Querying
  // at a sub-millisecond cadence is effectively zero-cost for hashrate and keeps the miner idle while
  // the GPU works. The caller first sleeps through 90% of its learned attempt duration.
  void wait(double learned_us) {
    activate();
#if !defined(_WIN32)
    const auto poll =
        learned_us > 20000.0 ? std::chrono::microseconds(500) : std::chrono::microseconds(100);
#else
    (void)learned_us;
#endif
    for (;;) {
      const hipError_t status = hipEventQuery(done);
      if (status == hipSuccess) {
        if (std::getenv("MOM_PEARLHASH_STATS") && stats_printed++ < 4) {
          float wmma_ms = 0.0f, hash_ms = 0.0f;
          hip_check(hipEventElapsedTime(&wmma_ms, started, wmma_done),
                    "hipEventElapsedTime(PearlHash WMMA)");
          hip_check(hipEventElapsedTime(&hash_ms, wmma_done, done),
                    "hipEventElapsedTime(PearlHash hash)");
          std::fprintf(stderr, "PEARLHASH_HIP_STATS wmma=%.3fms hash=%.3fms\n", wmma_ms, hash_ms);
        }
        return;
      }
      if (status != hipErrorNotReady)
        hip_check(status, "hipEventQuery(PearlHash)");
#if defined(_WIN32)
      mom_sycl_poll_pause();
#else
      std::this_thread::sleep_for(poll);
#endif
    }
  }
  ~PearlHashHipSearch() {
    release();
  }
#else
  static bool supports(sycl::queue&, std::string* const reason = nullptr) {
    if (reason)
      *reason = "HIP source-JIT support is not part of this worker";
    return false;
  }
  static bool supports_shape(int, int, int, int) {
    return false;
  }
  bool ensure(sycl::queue&, int, int, int, int) {
    return false;
  }
  void wait(double) {
  }
  void release() noexcept {
  }
#endif
};
