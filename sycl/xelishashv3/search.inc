// Submit the dependent scratch initialization, mixing, and final-hash stages.
template <bool Native>
static void search(State& s, uint64_t first, unsigned count, bool test, bool use_esimd = false) {
  const uint8_t* input = s.input;
  uint64_t* scratch = s.scratch;
  const uint8_t* target = s.target;
  Result* result = s.result;
  const double* lut = s.lut;
  using Clock = std::chrono::steady_clock;
  const auto total_start = s.profile ? Clock::now() : Clock::time_point{};
  // Profiling needs per-stage barriers for meaningful timings. Portable OpenCL also completes one
  // kernel at a time because some runtimes reject the full queued chain during kernel setup.
  // Windows Level Zero requires the same barriers around ESIMD despite the queue being in order.
  const auto run_stage = [&](const char* stage, bool required, const auto& submit) {
#ifdef MOM_SYCL_PORTABLE_OPENCL
    required = true;
#endif
#ifdef _WIN32
    required = required || use_esimd;
#endif
    try {
      const sycl::event event = submit();
      if (required || s.profile)
        sycl_wait_and_throw(event, s.device);
      return event;
    } catch (...) {
      std::fprintf(stderr, "XelisHashV3 %s failed\n", stage);
      throw;
    }
  };
  const auto stage1_start = s.profile ? Clock::now() : Clock::time_point{};
  // Keep stage events owned until the final wait. This defensively avoids pending-handle reuse
  // in AdaptiveCpp's HIP event pool without adding barriers; it does not establish a stall cause.
  [[maybe_unused]] const auto stage1_event = run_stage("stage 1", false, [&] {
    return s.queue.submit([&](sycl::handler& h) {
      h.parallel_for<Stage1Kernel<Native>>(
          sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        stage1(input, scratch + id[0] * WORDS, first + id[0]);
      });
    });
  });
  const auto stage1_end = s.profile ? Clock::now() : Clock::time_point{};
  const auto stage3_start = s.profile ? Clock::now() : Clock::time_point{};
  // Specialize at dispatch, so the no-FP64 device kernel contains no double arithmetic.
  const auto submit_stage3 = [&]<bool Fp64>() {
#ifdef MOM_XELISHASHV3_ESIMD
    if constexpr (Native) {
      if (use_esimd) {
        return run_stage("stage 3", false, [&] {
          return s.queue.submit([&](sycl::handler& h) {
            h.parallel_for<Stage3EsimdKernel<Native, Fp64>>(
                sycl::range<1>(count),
                [=](sycl::id<1> id) SYCL_ESIMD_KERNEL MOM_SYCL_KERNEL_ARGS_RESTRICT {
                  stage3<Native, true, Fp64>(scratch + id[0] * WORDS, lut);
                });
          });
        });
      }
    }
#endif
    return run_stage("stage 3", false, [&] {
      return s.queue.submit([&](sycl::handler& h) {
        h.parallel_for<Stage3Kernel<Native, Fp64>>(
            sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          stage3<Native, false, Fp64>(scratch + id[0] * WORDS, lut);
        });
      });
    });
  };
  [[maybe_unused]] const auto stage3_event = lut ? submit_stage3.template operator()<true>()
                                                : submit_stage3.template operator()<false>();
  const auto stage3_end = s.profile ? Clock::now() : Clock::time_point{};
  const auto stage4_start = s.profile ? Clock::now() : Clock::time_point{};
  run_stage("stage 4", true, [&] {
    return s.queue.submit([&](sycl::handler& h) {
      h.parallel_for<Stage4Kernel<Native>>(
          sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        uint8_t hash[32];
        stage4(scratch + id[0] * WORDS, hash);
        bool hit = true;
        for (unsigned i = 0; i < 32; ++i)
          if (hash[i] != target[i]) {
            hit = hash[i] < target[i];
            break;
          }
        if (test && id[0] == 0) {
          result->count = 1;
          result->nonce = first;
          for (unsigned i = 0; i < 32; ++i)
            result->hash[i] = hash[i];
        } else if (!test && hit) {
          using Atomic =
              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                               sycl::access::address_space::global_space>;
          if (Atomic(result->count).fetch_add(1) == 0) {
            result->nonce = first + id[0];
            for (unsigned i = 0; i < 32; ++i)
              result->hash[i] = hash[i];
          }
        }
      });
    });
  });
  const auto stage4_end = s.profile ? Clock::now() : Clock::time_point{};
  const auto total_end = s.profile ? Clock::now() : Clock::time_point{};
  if (s.profile && s.profile_reports++ < 8) {
    const double wall_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start).count();
    const double stage1_ms =
        std::chrono::duration<double, std::milli>(stage1_end - stage1_start).count();
    const double stage3_ms =
        std::chrono::duration<double, std::milli>(stage3_end - stage3_start).count();
    const double stage4_ms =
        std::chrono::duration<double, std::milli>(stage4_end - stage4_start).count();
    std::fprintf(
        stderr,
        "XelisHashV3 profile count=%u native=%u esimd=%u wall_ms=%.3f stage_ms=%.3f,%.3f,%.3f\n",
        s.profile_reports, Native ? 1u : 0u, use_esimd ? 1u : 0u, wall_ms, stage1_ms, stage3_ms,
        stage4_ms);
  }
}
