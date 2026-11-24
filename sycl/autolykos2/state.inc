#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
// Learn from six completed, distinct batches: two warmups, then split/mono/mono/split.
// The caller holds the state mutex and observes only a successful dispatch's normal result.
struct AutolykosPipelineSelection {
  uint32_t n = 0, intensity = 0;
  unsigned workgroup = 0, phase = 0;
  double split_us = 0, mono_us = 0;
  bool monolithic = false;

  static bool eligible(const bool cuda, const bool table, const bool test,
                       const bool split_override, const bool profile_override) {
    return cuda && table && !test && !split_override && !profile_override;
  }

  // Relearn only when the workload shape changes; an override never alters the cached choice.
  bool select(const uint32_t next_n, const uint32_t next_intensity,
              const unsigned next_workgroup, const bool allowed, const bool fallback) {
    if (!allowed)
      return fallback;
    if (n != next_n || intensity != next_intensity || workgroup != next_workgroup) {
      *this = {};
      n = next_n;
      intensity = next_intensity;
      workgroup = next_workgroup;
    }
    return phase < 6 ? phase == 0 || phase == 2 || phase == 5 : !monolithic;
  }

  // Count only completed batches and keep split across noise; 5% is a throughput margin.
  void observe(const double elapsed_us) {
    if (phase >= 6 || !std::isfinite(elapsed_us) || elapsed_us <= 0)
      return;
    if (phase >= 2) {
      double& total = phase == 2 || phase == 5 ? split_us : mono_us;
      if (!std::isfinite(total + elapsed_us))
        return;
      total += elapsed_us;
    }
    if (++phase == 6) {
      monolithic = mono_us <= split_us / 1.05;
      std::fprintf(stderr, "Autolykos2: selected %s pipeline (split %.2f, monolithic %.2f MH/s)\n",
                   monolithic ? "monolithic" : "split", 2.0 * intensity / split_us,
                   2.0 * intensity / mono_us);
    }
  }
};
#endif

struct State {
  sycl::device device;
  sycl::queue queue;
  UsmPtr<uint32_t> table;
  UsmPtr<uint32_t> bhashes;
  MomBufferAllocation<AutolykosPortableResult> portable_results;
  UsmPtr<PowHits<>> result;
  UsmPtr<uint32_t> table_mismatches;
  uint32_t table_height = UINT32_MAX;
  uint32_t table_n = 0;
  uint32_t table_cap_n = 0;
  uint32_t bhashes_cap = 0;
  bool subgroup_coop_verified = false;
  unsigned workgroup;
  unsigned prehash_workgroup;
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  AutolykosPipelineSelection pipeline_selection;
#endif
  std::mutex mutex;

  // Background prebuild of the next height's table was tried three ways and measured
  // slower than the synchronous rebuild it replaces on Xe2/GuC (B580, 2026-06): a second
  // queue is timesliced against mining whether paced, flat-out or priority_low (mining
  // drops to 50%/15%/starved-builder respectively), and fusing builder workgroups into the
  // mix kernel halves its occupancy through shared register allocation (~2x the slices'
  // GPU cost). The prehash's GPU-seconds cannot co-execute with mining on this hardware,
  // so the cheapest place to pay them is one concentrated rebuild per height.

  // Bind launch policy to one device; portable builds use buffers instead of shared USM.
  explicit State(const std::string& dev_str)
      : device(get_dev(dev_str)),
        queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
        workgroup(autolykos_workgroup(device)),
        prehash_workgroup(autolykos_prehash_workgroup(device)) {
    if constexpr (!mom_sycl_portable_opencl) {
      if (!mom_has_usm_shared(device)) {
        throw std::string("autolykos2 SYCL device does not support shared allocations");
      }
    }
  }

  // Drain accepted kernels before the members release their allocations.
  ~State() {
    sycl_cleanup_noexcept("autolykos2 wait", [&] { queue.wait_and_throw(); });
  }

  // Pick the search workgroup per backend; MOM_AUTOLYKOS2_WORKGROUP may override it.
  static unsigned autolykos_workgroup(const sycl::device& dev) {
    // CUDA/HIP: 128 is a reliable default for both the scalar diagnostic and the eight-lane
    // subgroup path. The latter is insensitive between 64 and 128 on current NVIDIA hardware.
    const unsigned preferred =
        (mom_is_cuda(dev) || mom_is_hip(dev))
            ? 128
            :
#if defined(_WIN32)
            // Windows Arc Level Zero is consistently faster at local size 32 for the split table path.
            32;
#else
            64;
#endif
    const unsigned fallback = sycl_default_workgroup(dev, {32, 64, 128, 256}, preferred);
    return env_workgroup(dev, "MOM_AUTOLYKOS2_WORKGROUP", fallback);
  }

  // Tune prehash independently of the search workgroup within supported device limits.
  static unsigned autolykos_prehash_workgroup(const sycl::device& dev) {
    const unsigned fallback = sycl_default_workgroup(dev, {32, 64, 128, 256}, 64);
    return env_workgroup(dev, "MOM_AUTOLYKOS2_PREHASH_WORKGROUP", fallback);
  }

  // Accept an env override only if it is one of the supported workgroup sizes.
  static unsigned env_workgroup(const sycl::device& dev, const char* name, const unsigned fallback) {
    uint32_t parsed = 0;
    if (mom_parse_env_u32(name, parsed) && (parsed == 32 || parsed == 64 || parsed == 128 || parsed == 256) &&
        sycl_workgroup_fits(dev, static_cast<unsigned>(parsed)))
      return parsed;
    return fallback;
  }

  // Unlike the kawpow DAG chunking this defaults on: at live N the monolithic build is one
  // multi-second GPU job, the long-running-job shape that tripped xe/GuC scheduler cleanup
  // crashes in June 2026. ~3 s chunks cost nothing on the in-order queue. 0 = single kernel.
  static uint32_t autolykos_table_chunk_rows() {
    constexpr uint32_t fallback = 32U * 1024U * 1024U;
    uint32_t parsed = 0;
    return mom_parse_env_u32("MOM_AUTOLYKOS2_TABLE_CHUNK", parsed) ? parsed : fallback;
  }

  // 0 disables the post-build row check, 1 (default) cross-checks a strided sample of rows
  // against autolykos_prehash_digest_dev, 2 recomputes every row (slow, for debugging).
  static unsigned autolykos_table_verify_mode() {
    uint32_t parsed = 0;
    return mom_parse_env_u32("MOM_AUTOLYKOS2_VERIFY_TABLE", parsed) && parsed <= 2 ? parsed : 1;
  }

  // Keep split search enabled unless a diagnostic override starts with zero.
  static bool autolykos_split_table() {
    const char* const value = std::getenv("MOM_AUTOLYKOS2_SPLIT");
    return !(value && value[0] == '0');
  }

  // CUDA/HIP path: eight lanes own one 256-bit accumulator. This lowers the scalar kernel's register
  // footprint and turns every random table row into one contiguous 32-byte subgroup transaction.
  // The old HIP-only variable remains an alias for existing diagnostic scripts.
  static bool autolykos_subgroup_coop() {
    const char* value = std::getenv("MOM_AUTOLYKOS2_SUBGROUP_COOP");
    if (!value)
      value = std::getenv("MOM_AUTOLYKOS2_HIP_COOP");
    return !value || (std::strcmp(value, "0") && std::strcmp(value, "false"));
  }

  // Opt into one scalar/cooperative cross-check per state, including the legacy HIP alias.
  static bool autolykos_subgroup_coop_verify() {
    const char* value = std::getenv("MOM_AUTOLYKOS2_SUBGROUP_COOP_VERIFY");
    if (!value)
      value = std::getenv("MOM_AUTOLYKOS2_HIP_COOP_VERIFY");
    return value && (!std::strcmp(value, "1") || !std::strcmp(value, "true"));
  }

  // Allocate shared hits lazily; portable builds instead retain per-nonce buffer results.
  void ensure_result() {
    if constexpr (!mom_sycl_portable_opencl) {
      if (!result)
        result = mom_usm_alloc<PowHits<>>(queue, 1, true,
                                          "Can't allocate autolykos2 SYCL result buffer");
    }
  }

  // Drain outstanding users before growing the portable per-nonce output buffer.
  void ensure_portable_results(const uint32_t count) {
    if constexpr (!mom_sycl_portable_opencl)
      return;
    if (portable_results.capacity() >= count)
      return;
    queue.wait_and_throw();
    portable_results.allocate(count);
  }

  // Grow split-search intermediate storage only after its queued users finish.
  void ensure_bhashes(const uint32_t intensity) {
    if (bhashes && bhashes_cap >= intensity)
      return;
    queue.wait_and_throw();
    bhashes.reset();
    bhashes_cap = 0;
    bhashes = mom_usm_alloc<uint32_t>(queue, static_cast<uint64_t>(intensity) * 8U, false,
                                      "Can't allocate autolykos2 intermediate buffer");
    bhashes_cap = intensity;
  }

  // Choose precomputed rows only when backend capabilities and the memory reserve permit.
  bool should_use_table(const bool is_test, const uint32_t n_len) const {
    if constexpr (mom_sycl_portable_opencl) {
      // Some conforming ICDs cap individual buffers at 2 GiB. Recompute rows instead of requiring
      // the ~6.9-GiB fast table in the no-USM profile.
      (void)is_test;
      (void)n_len;
      return false;
    } else {
      if (is_test || device.is_cpu())
        return false;
      const char* const value = std::getenv("MOM_AUTOLYKOS2_TABLE");
      if (value)
        return value[0] != '0'; // Preserve the explicit diagnostic force-on/force-off behavior.
      if (!mom_has_usm_device(device))
        return false;

      // The correct on-the-fly path needs no table. Select it automatically on lower-memory GPUs so
      // the advertised 3-GiB minimum is real; 8-GiB-class devices retain the full-speed table.
      const uint64_t table_bytes = static_cast<uint64_t>(n_len) * TABLE_ENTRY_BYTES;
      const uint64_t global_bytes = device.get_info<sycl::info::device::global_mem_size>();
      constexpr uint64_t reserve = 512ULL * MiB;
      if (table_bytes > global_bytes || global_bytes - table_bytes < reserve)
        return false;
      return mom_is_hip(device) || sycl_is_level_zero_gpu(device) ||
             device.get_info<sycl::info::device::max_mem_alloc_size>() >= table_bytes;
    }
  }

  // Recompute a sample of the freshly built table with the reference digest and throw on
  // any difference: the fast build kernel folds the constant pad, and a silently wrong
  // table would invalidate every share until the next block.
  void verify_table(const uint32_t height, const uint32_t n_len) {
    const unsigned mode = autolykos_table_verify_mode();
    if (!mode)
      return;

    if (!table_mismatches)
      table_mismatches = mom_usm_alloc<uint32_t>(queue, 1, true,
                                                 "Can't allocate autolykos2 verify counter");
    *table_mismatches = 0;

    const uint32_t stride = mode >= 2 ? 1 : 65521; // prime; ~3.3k of 216M rows in default mode
    const uint32_t count = (n_len + stride - 1) / stride;
    const uint32_t* const d_table = table;
    uint32_t* const d_mismatches = table_mismatches;
    const uint32_t local = prehash_workgroup;

    sycl_wait_and_throw(queue.submit([&](sycl::handler& h) {
      h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(mom_round_up_size(count, local)), sycl::range<1>(local)),
          [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
            const uint32_t gid = item.get_global_id(0);
            if (gid >= count)
              return;
            const uint32_t row = gid * stride;
            if (row >= n_len)
              return;

            uint8_t digest[32];
            autolykos_prehash_digest_dev(row, height, digest);
            uint32_t expected[8];
            digest_to_limbs_dev(digest, expected);

            const uint32_t* const stored =
                d_table + static_cast<uint64_t>(row) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
            bool differs = false;
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
              differs |= stored[i] != expected[i];
            if (differs) {
              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                               sycl::access::address_space::global_space>(*d_mismatches)
                  .fetch_add(1);
            }
          });
    }),
                        device);

    if (*table_mismatches) {
      throw std::string("Autolykos2 table verification failed: ") +
          std::to_string(*table_mismatches) + " of " + std::to_string(count) +
          " sampled rows differ";
    }
  }

  // Rebuild and verify the height-specific table before publishing its reuse key.
  void ensure_table(const uint32_t height, const uint32_t n_len, const bool is_test,
                    const bool should_log) {
    if (!should_use_table(is_test, n_len)) {
      if (table) {
        queue.wait_and_throw();
        table.reset();
      }
      table_height = UINT32_MAX;
      table_n = 0;
      table_cap_n = 0;
      return;
    }

    if (table && table_height == height && table_n == n_len)
      return;
    if (!mom_has_usm_device(device)) {
      throw std::string("autolykos2 SYCL GPU device does not support device allocations");
    }

    const uint64_t start_ms = mom_now_ms();
    table_height = UINT32_MAX;
    if (!table || table_cap_n != n_len) {
      queue.wait_and_throw();
      table.reset();
      table_cap_n = 0;
      table = mom_usm_alloc<uint32_t>(
          queue, static_cast<uint64_t>(n_len) * (TABLE_ENTRY_BYTES / sizeof(uint32_t)), false,
          "Can't allocate autolykos2 table");
      table_cap_n = n_len;
    }

    uint32_t* const d_table = table;
    sycl::queue& q = queue;
    const uint32_t local = prehash_workgroup;

    const uint32_t chunk_rows = autolykos_table_chunk_rows();
    sycl::event build_event;
    for (uint32_t start_row = 0; start_row < n_len;) {
      const uint32_t current_rows = chunk_rows ? std::min(chunk_rows, n_len - start_row) : n_len;
      const uint32_t chunk_start = start_row;
      build_event = q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(mom_round_up_size(current_rows, local)),
                              sycl::range<1>(local)),
            [=](sycl::nd_item<1> item) {
              const uint32_t local_row = item.get_global_id(0);
              if (local_row >= current_rows)
                return;
              autolykos_table_row_store_dev(d_table, chunk_start + local_row, height);
            });
      });
      if (!chunk_rows)
        break;
      start_row += current_rows;
    }
    sycl_wait_and_throw(build_event, device);
    verify_table(height, n_len);

    table_height = height;
    table_n = n_len;
    if (should_log || std::getenv("MOM_LOOP_STATS")) {
      char elapsed[32];
      mom_format_duration_ms(elapsed, sizeof(elapsed), mom_now_ms() - start_ms);
      std::fprintf(stderr, "Autolykos2 table for height %u N %u calculated (%s)\n", height, n_len,
                   elapsed);
    }
  }
};
