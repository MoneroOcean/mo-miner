struct PearlHashHipSearch {
#if defined(MOM_SYCL_HAS_HIP)
  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
  hipFunction_t hash_kernel = nullptr;
  hipEvent_t started = nullptr;
  hipEvent_t wmma_done = nullptr;
  hipEvent_t done = nullptr;
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
    if (status != hipSuccess) throw std::string(operation) + ": " + hipGetErrorString(status);
  }
  static std::string rtc_error(PearlHashHiprtcApi& api, hiprtcResult status,
                               hiprtcProgram program, const char* operation) {
    size_t size = 0;
    std::string log;
    if (program && api.get_log_size(program, &size) == HIPRTC_SUCCESS && size) {
      log.resize(size);
      (void)api.get_log(program, log.data());
    }
    return std::string(operation) + ": " + api.error_string(status) +
           (log.empty() ? "" : "\n" + log);
  }
  static bool is_hip(const sycl::device& device) {
  #if defined(MOM_SYCL_ADAPTIVECPP)
    return device.get_backend() == sycl::backend::hip;
  #else
    return device.get_backend() == sycl::backend::ext_oneapi_hip;
  #endif
  }
  static bool supports(sycl::queue& q, std::string* const reason = nullptr) {
    const auto unavailable = [&](const std::string& message) {
      if (reason) *reason = message;
      return false;
    };
    if (!is_hip(q.get_device())) return unavailable("the selected SYCL device is not HIP");
    int device = 0;
    hipDeviceProp_t properties{};
    const hipError_t device_status = hipGetDevice(&device);
    if (device_status != hipSuccess)
      return unavailable(std::string("hipGetDevice: ") + hipGetErrorString(device_status));
    const hipError_t properties_status = hipGetDeviceProperties(&properties, device);
    if (properties_status != hipSuccess)
      return unavailable(std::string("hipGetDeviceProperties: ") +
                         hipGetErrorString(properties_status));
    if (!std::strncmp(properties.gcnArchName, "gfx1200", 7) ||
        !std::strncmp(properties.gcnArchName, "gfx1201", 7)) return true;
    return unavailable(std::string("gfx12 WMMA is unavailable on ") +
                       (properties.gcnArchName[0] ? properties.gcnArchName : "an unknown architecture"));
  }
  void release() noexcept {
    if (done) (void)hipEventDestroy(done);
    if (wmma_done) (void)hipEventDestroy(wmma_done);
    if (started) (void)hipEventDestroy(started);
    if (transcript) (void)hipFree(transcript);
    if (module) (void)hipModuleUnload(module);
    done = nullptr;
    wmma_done = nullptr;
    started = nullptr;
    transcript = nullptr;
    transcript_bytes = 0;
    module = nullptr;
    kernel = nullptr;
    hash_kernel = nullptr;
    compiled_k = 0;
    compiled_rank = 0;
    enabled = false;
  }
  bool ensure(sycl::queue& q, int m, int n, int k, int rank) {
    if (!supports(q)) return false;
    // The fixed wave tile and cumulative transcript require these divisibilities. All shipped
    // PearlHash shapes satisfy them; retaining the check keeps custom research shapes safe.
    const int rows_per_wave = 16 * tile_er, cols_per_wave = 16 * tile_ec;
    const uint64_t waves = m > 0 && n > 0 ? (uint64_t)(m / rows_per_wave) * (uint64_t)(n / cols_per_wave) : 0;
    if (m % rows_per_wave || n % cols_per_wave || k % 16 || rank % 16 || k % rank ||
        k / rank != 16 || waves % (wmma_threads / 32)) {
      return false;
    }
    if (checked && (compiled_k != k || compiled_rank != rank)) {
      return false;
    }
    if (!checked) {
      try {
        checked = true;
        int device = 0;
        hipDeviceProp_t properties{};
        hip_check(hipGetDevice(&device), "hipGetDevice(PearlHash)");
        hip_check(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties(PearlHash)");
        std::string arch = properties.gcnArchName;
        if (const size_t colon = arch.find(':'); colon != std::string::npos) arch.resize(colon);
        if (arch != "gfx1200" && arch != "gfx1201") {
          return false;
        }

        const std::string source(
#include "hip_kernel.inc"
        );
        const std::string arch_option = "--offload-arch=" + arch;
        const std::string er_option = "-DPEARLHASH_ER=" + std::to_string(tile_er);
        const std::string ec_option = "-DPEARLHASH_EC=" + std::to_string(tile_ec);
        const std::string k_option = "-DPEARLHASH_K=" + std::to_string(k);
        const std::string rank_option = "-DPEARLHASH_RANK=" + std::to_string(rank);
        const std::string unroll_option = "-DPEARLHASH_K_UNROLL=" + std::to_string(k_unroll);
        const char* options[] = {"-O3", "--std=c++17", arch_option.c_str(), er_option.c_str(),
                                 ec_option.c_str(), k_option.c_str(), rank_option.c_str(),
                                 unroll_option.c_str()};
        std::ostringstream key_stream;
        key_stream << source << '\0' << arch;
        for (const char* option : options) key_stream << '\0' << option;
        key_stream << "\0pearlhash-hip-cache-v1";
        std::ostringstream filename;
        filename << "pearlhash-" << arch << '-' << std::hex << std::setw(16)
                 << std::setfill('0') << pearlhash_jit_hash(key_stream.str()) << ".hsaco";
        const std::filesystem::path cache_dir = pearlhash_jit_cache_dir();
        const std::filesystem::path cache_path =
          cache_dir.empty() ? std::filesystem::path{} : cache_dir / filename.str();
        std::vector<char> code = cache_path.empty()
          ? std::vector<char>{} : pearlhash_read_cache(cache_path);
        if (!code.empty() && hipModuleLoadData(&module, code.data()) != hipSuccess) {
          std::error_code error;
          std::filesystem::remove(cache_path, error);
          code.clear();
        }
        if (code.empty()) {
          PearlHashHiprtcApi& rtc = PearlHashHiprtcApi::instance();
          if (!rtc.available()) throw rtc.error;
          hiprtcProgram program = nullptr;
          hiprtcResult status = rtc.create_program(
            &program, source.c_str(), "pearlhash-wmma.hip", 0, nullptr, nullptr);
          if (status != HIPRTC_SUCCESS)
            throw rtc_error(rtc, status, program, "hiprtcCreateProgram(PearlHash)");
          status = rtc.compile_program(program, 8, options);
          if (status != HIPRTC_SUCCESS) {
            const std::string message =
              rtc_error(rtc, status, program, "hiprtcCompileProgram(PearlHash)");
            (void)rtc.destroy_program(&program);
            throw message;
          }
          size_t code_size = 0;
          status = rtc.get_code_size(program, &code_size);
          if (status != HIPRTC_SUCCESS) {
            const std::string message =
              rtc_error(rtc, status, program, "hiprtcGetCodeSize(PearlHash)");
            (void)rtc.destroy_program(&program);
            throw message;
          }
          code.resize(code_size);
          status = rtc.get_code(program, code.data());
          if (status != HIPRTC_SUCCESS) {
            const std::string message =
              rtc_error(rtc, status, program, "hiprtcGetCode(PearlHash)");
            (void)rtc.destroy_program(&program);
            throw message;
          }
          (void)rtc.destroy_program(&program);
          hip_check(hipModuleLoadData(&module, code.data()), "hipModuleLoadData(PearlHash)");
          pearlhash_write_cache(cache_path, code);
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
        std::fprintf(stderr, "PearlHash gfx12 WMMA unavailable, using portable dp4a: %s\n", error.c_str());
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
        hip_check(hipMalloc(reinterpret_cast<void**>(&transcript), bytes), "hipMalloc(PearlHash transcript)");
        transcript_bytes = bytes;
      } catch (const std::string& error) {
        release();
        std::fprintf(stderr, "PearlHash gfx12 transcript allocation unavailable, using portable dp4a: %s\n", error.c_str());
        return false;
      }
    }
    return true;
  }
  void fail(const std::string& error) noexcept {
    (void)hipDeviceSynchronize();
    release();
    checked = true;
    std::fprintf(stderr, "PearlHash HIP source-JIT failed, using the best SYCL kernel: %s\n",
                 error.c_str());
  }
  void launch(const Buffers& b, uint32_t seed, int m, int n, int k, int rank, bool debug) {
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
    hip_check(hipEventRecord(started, nullptr), "hipEventRecord(PearlHash start)");
    hip_check(hipModuleLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, wmma_args, nullptr),
              "hipModuleLaunchKernel(PearlHash WMMA)");
    hip_check(hipEventRecord(wmma_done, nullptr), "hipEventRecord(PearlHash WMMA done)");
    const uint64_t tiles = (uint64_t)(m / 16) * (uint64_t)(n / 16);
    constexpr unsigned hash_threads = 256;
    const unsigned hash_blocks = (unsigned)((tiles + hash_threads - 1) / hash_threads);
    void* hash_args[] = {&transcript_arg, &key, &target, &result, &seed, &m, &n, &debug_int};
    hip_check(hipModuleLaunchKernel(hash_kernel, hash_blocks, 1, 1, hash_threads, 1, 1, 0, nullptr,
                                    hash_args, nullptr),
              "hipModuleLaunchKernel(PearlHash hash)");
    hip_check(hipEventRecord(done, nullptr), "hipEventRecord(PearlHash)");
  }
  // hipEventSynchronize can spin a complete CPU core for the many-second production GEMM. Querying
  // at a sub-millisecond cadence is effectively zero-cost for hashrate and keeps the miner idle while
  // the GPU works. The caller first sleeps through 90% of its learned attempt duration.
  void wait(double learned_us) {
#if !defined(_WIN32)
    const auto poll = learned_us > 20000.0 ? std::chrono::microseconds(500)
                                           : std::chrono::microseconds(100);
#else
    (void)learned_us;
#endif
    for (;;) {
      const hipError_t status = hipEventQuery(done);
      if (status == hipSuccess) {
        if (std::getenv("MOM_PEARLHASH_STATS") && stats_printed++ < 4) {
          float wmma_ms = 0.0f, hash_ms = 0.0f;
          hip_check(hipEventElapsedTime(&wmma_ms, started, wmma_done), "hipEventElapsedTime(PearlHash WMMA)");
          hip_check(hipEventElapsedTime(&hash_ms, wmma_done, done), "hipEventElapsedTime(PearlHash hash)");
          std::fprintf(stderr, "PEARLHASH_HIP_STATS wmma=%.3fms hash=%.3fms\n", wmma_ms, hash_ms);
        }
        return;
      }
      if (status != hipErrorNotReady) hip_check(status, "hipEventQuery(PearlHash)");
#if defined(_WIN32)
      mom_sycl_poll_pause();
#else
      std::this_thread::sleep_for(poll);
#endif
    }
  }
  ~PearlHashHipSearch() { release(); }
#else
  static bool supports(sycl::queue&, std::string* const reason = nullptr) {
    if (reason) *reason = "HIP source-JIT support is not part of this worker";
    return false;
  }
  bool ensure(sycl::queue&, int, int, int, int) { return false; }
  void wait(double) {}
  void release() noexcept {}
  void fail(const std::string&) noexcept {}
#endif
};
