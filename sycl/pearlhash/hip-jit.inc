struct PearlHashHipSearch {
#if defined(MOM_SYCL_HAS_HIP)
  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
  hipEvent_t started = nullptr;
  hipEvent_t done = nullptr;
  hipStream_t native_stream = nullptr;
  hipDevice_t native_device = -1;
  bool checked = false;
  bool enabled = false;
  unsigned stats_printed = 0;
  // One wave's 13x4 WMMA tiles hold exactly the lane openings the host keys and proves.
  static constexpr int tile_er = static_cast<int>(kAmdLaneRows.length[1]);
  static constexpr int tile_ec = static_cast<int>(kAmdLaneCols.length[0]);
  int k_unroll = 1;
  int compiled_k = 0;
  int compiled_rank = 0;
  unsigned wmma_threads = 256;

  static void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess)
      throw std::string(operation) + ": " + hipGetErrorString(status);
  }
  // Attach the available HIPRTC compiler log to a failed compilation operation.
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
  // Require HIP and gfx12 int8 WMMA before enabling the optional runtime-compiled route.
  static bool supports(sycl::queue& q, std::string* const reason = nullptr) {
    const auto unavailable = [&](const std::string& message) {
      if (reason)
        *reason = message;
      return false;
    };
    if (!mom_is_hip(q.get_device()))
      return unavailable("the selected SYCL device is not HIP");
    return mom::hip_gfx12_int8_wmma_supported(q, reason);
  }
  // Admit complete proof rectangles; the WMMA kernel masks partial wave tiles at matrix edges.
  static bool supports_shape(const int m, const int n, const int k, const int rank) {
    constexpr int rows_per_wave = 16 * 12;
    constexpr int columns_per_wave = 16 * 4;
    const uint64_t waves = m > 0 && n > 0
        ? static_cast<uint64_t>(m / rows_per_wave + (m % rows_per_wave != 0)) *
              static_cast<uint64_t>(n / columns_per_wave + (n % columns_per_wave != 0))
        : 0;
    // A bounded row slice keeps local transcript offsets within uint32_t even for large profiles.
    // Complete rank groups beyond the first 16 fold into the same canonical rolling transcript.
    return m >= 128 && n >= 128 && k > 0 && rank > 0 && m % 64 == 0 &&
        n % 64 == 0 && k % 32 == 0 &&
        rank % 32 == 0 && k % rank == 0 && k / rank >= 16 && k / rank <= 64 &&
        waves > 0 && waves <= std::numeric_limits<uint32_t>::max();
  }
  // Best-effort release on the owning HIP device; reset handles so partial setup remains retryable.
  void release() noexcept {
    if (native_device >= 0)
      (void)hipSetDevice(native_device);
    if (done)
      (void)hipEventDestroy(done);
    if (started)
      (void)hipEventDestroy(started);
    if (module)
      (void)hipModuleUnload(module);
    done = nullptr;
    started = nullptr;
    native_stream = nullptr;
    native_device = -1;
    module = nullptr;
    kernel = nullptr;
    compiled_k = 0;
    compiled_rank = 0;
    enabled = false;
    checked = false;
  }
  void activate() const {
    hip_check(hipSetDevice(native_device), "hipSetDevice(PearlHash)");
  }
  // Compile/cache the selected HIP WMMA specialization using the shared matrix buffers.
  bool ensure(sycl::queue& q, const Buffers& b, int m, int n, int k, int rank) {
    // Allocation slices keep the shared 512-row granularity; launch() chooses cache-aligned
    // sub-slices and rounds wave blocks up, masking unused waves in the kernel.
    if (!b.Ap || !b.Bp)
      return false;
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
    // This source-JIT path implements the fixed rolling transcript for ratios 16..64. Profiles
    // outside that bounded range use the shared SYCL path before choosing a matrix layout.
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
#include "hip-kernel.inc"
        );
        const std::string arch_option = "--offload-arch=" + arch;
        const std::string er_option = "-DPEARLHASH_ER=" + std::to_string(tile_er);
        const std::string ec_option = "-DPEARLHASH_EC=" + std::to_string(tile_ec);
        const std::string k_option = "-DPEARLHASH_K=" + std::to_string(k);
        const std::string rank_option = "-DPEARLHASH_RANK=" + std::to_string(rank);
        const std::string unroll_option = "-DPEARLHASH_K_UNROLL=" + std::to_string(k_unroll);
        const char* options[] = {"-O3",
                                 "--std=c++17",
                                 "-mwavefrontsize64",
                                 arch_option.c_str(),
                                 er_option.c_str(),
                                 ec_option.c_str(),
                                 k_option.c_str(),
                                 rank_option.c_str(),
                                 unroll_option.c_str()};
        std::ostringstream key_stream;
        key_stream << source << '\0' << arch;
        for (const char* option : options) key_stream << '\0' << option;
        key_stream << '\0' << "pearlhash-hip-cache-v1";
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
          status = rtc.compile_program(program.get(), static_cast<int>(std::size(options)), options);
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
        hip_check(hipModuleGetFunction(&kernel, module, "pearlhash_wmma_hash_fused"), "hipModuleGetFunction(PearlHash fused)");
        hip_check(hipEventCreate(&started), "hipEventCreate(PearlHash start)");
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
    return enabled;
  }
  // Enqueue the fused WMMA/transcript/hash kernel in native-stream order, then record completion.
  void launch(const Buffers& b, uint32_t seed, int m, int n, int k, int rank, bool debug) {
    activate();
    const unsigned threads = wmma_threads;
    const unsigned waves_per_block = threads / 64;
    void* B = b.Bp;
    void* key = b.cA;
    void* target = b.target;
    void* result = b.result;
    int debug_int = debug ? 1 : 0;
    // Match wave_groups' bounded A band. The fused kernel uses LDS, so its launch slices
    // need not fit the unused global transcript; small jobs and tails remain masked.
    const int cache_rows = std::max(16, std::min(64, 131072 / k));
    const int row_group = cache_rows * tile_er * 16;
    // Slices start at whole row groups so every lane opening keeps its canonical origin.
    const int group_rows = tile_er * 16;
    const int debug_rows = std::min(row_group, std::max(group_rows, (m / 2 / group_rows) * group_rows));
    const int chunk_rows = std::min(m, debug ? debug_rows : row_group);
    try {
      hip_check(hipEventRecord(started, native_stream), "hipEventRecord(PearlHash start)");
      for (int row_offset = 0; row_offset < m; row_offset += chunk_rows) {
        int rows = std::min(chunk_rows, m - row_offset);
        void* A = b.Ap + static_cast<size_t>(row_offset) * k;
        const uint64_t waves =
            static_cast<uint64_t>(rows / (16 * tile_er) + (rows % (16 * tile_er) != 0)) *
            static_cast<uint64_t>(n / (16 * tile_ec) + (n % (16 * tile_ec) != 0));
        const unsigned blocks = static_cast<unsigned>((waves + waves_per_block - 1) / waves_per_block);
        void* fused_args[] = {&A, &B, &key, &target, &result, &seed,
                              &rows, &n, &row_offset, &debug_int};
        hip_check(hipModuleLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, native_stream,
                                        fused_args, nullptr), "hipModuleLaunchKernel(PearlHash fused)");
      }
      hip_check(hipEventRecord(done, native_stream), "hipEventRecord(PearlHash)");
    } catch (...) {
      // SYCL cannot track these native submissions. Retire any earlier slice before its shared
      // buffers can be reused or freed, even if a later launch/completion record was rejected.
      (void)hipStreamSynchronize(native_stream);
      throw;
    }
  }
  // hipEventSynchronize can spin a complete CPU core for the many-second production GEMM. Querying
  // at a sub-millisecond cadence is effectively zero-cost for hashrate and keeps the miner idle while
  // the GPU works. The caller first sleeps through 90% of its learned attempt duration.
  void wait(double learned_us) {
    try {
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
            float search_ms = 0.0f;
            hip_check(hipEventElapsedTime(&search_ms, started, done),
                      "hipEventElapsedTime(PearlHash search)");
            std::fprintf(stderr, "PEARLHASH_HIP_STATS search=%.3fms\n", search_ms);
          }
          return;
        }
        if (status != hipErrorNotReady) {
          // SYCL cannot retire work submitted directly to this HIP stream. The outer handler
          // drains this stream for activation, query, and elapsed-time failures alike.
          hip_check(status, "hipEventQuery(PearlHash)");
        }
#if defined(_WIN32)
        mom_sycl_poll_pause();
#else
        std::this_thread::sleep_for(poll);
#endif
      }
    } catch (...) {
      // SYCL cannot retire native work; preserve the original error while draining this stream.
      (void)hipStreamSynchronize(native_stream);
      throw;
    }
  }
  ~PearlHashHipSearch() {
    release();
  }
#else
  // Require HIP and gfx12 int8 WMMA before enabling the optional runtime-compiled route.
  static bool supports(sycl::queue&, std::string* const reason = nullptr) {
    if (reason)
      *reason = "HIP source-JIT support is not part of this worker";
    return false;
  }
  // Require complete WMMA wave tiles and the kernel's fixed K/rank noise profile.
  static bool supports_shape(int, int, int, int) {
    return false;
  }
  // Compile/cache the selected HIP WMMA specialization using the shared bounded transcript owner.
  bool ensure(sycl::queue&, const Buffers&, int, int, int, int) {
    return false;
  }
  void wait(double) {
  }
  // Best-effort release on the owning HIP device; reset handles so partial setup remains retryable.
  void release() noexcept {
  }
#endif
};
