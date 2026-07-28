struct AutolykosState {
  sycl::device device;
  sycl::queue queue;
  std::unique_ptr<MomKernelBundle> bundle;
  uint32_t* table = nullptr;
  uint32_t* bhashes = nullptr;
  MomBufferAllocation<AutolykosPortableResult> portable_results;
  uint32_t portable_result_cap = 0;
  AutolykosResult* result = nullptr;
  uint32_t* table_mismatches = nullptr;
  uint32_t table_height = UINT32_MAX;
  uint32_t table_n = 0;
  uint32_t table_cap_n = 0;
  uint32_t bhashes_cap = 0;
  bool subgroup_coop_verified = false;
  unsigned workgroup;
  unsigned prehash_workgroup;
  std::mutex mutex;

  // Background prebuild of the next height's table was tried three ways and measured
  // slower than the synchronous rebuild it replaces on Xe2/GuC (B580, 2026-06): a second
  // queue is timesliced against mining whether paced, flat-out or priority_low (mining
  // drops to 50%/15%/starved-builder respectively), and fusing builder workgroups into the
  // mix kernel halves its occupancy through shared register allocation (~2x the slices'
  // GPU cost). The prehash's GPU-seconds cannot co-execute with mining on this hardware,
  // so the cheapest place to pay them is one concentrated rebuild per height.

  explicit AutolykosState(const std::string& dev_str)
    : device(get_dev(dev_str)),
      queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      workgroup(autolykos_workgroup(device)),
      prehash_workgroup(autolykos_prehash_workgroup(device))
  {
    if constexpr (!mom_sycl_portable_opencl) {
      if (!mom_has_usm_shared(device)) {
        throw std::string("autolykos2 SYCL device does not support shared allocations");
      }
    }

    set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", autolykos_compile_options(device));
    bundle = std::make_unique<MomKernelBundle>(
      mom_get_exec_bundle(queue.get_context())
    );
  }

  ~AutolykosState() { sycl_cleanup_noexcept("autolykos2", [&] { release(); }); }

  static unsigned autolykos_workgroup(const sycl::device& dev) {
    // CUDA/HIP: 128 is a reliable default for both the scalar diagnostic and the eight-lane
    // subgroup path. The latter is insensitive between 64 and 128 on current NVIDIA hardware.
    const unsigned preferred =
      (mom_is_cuda(dev) || mom_is_hip(dev)) ? 128 :
#if defined(_WIN32)
      // Windows Arc Level Zero is consistently faster at local size 32 for the split table path.
      32;
#else
      64;
#endif
    const unsigned fallback = sycl_default_workgroup(dev, {32, 64, 128, 256}, preferred);
    return env_workgroup("MOM_AUTOLYKOS2_WORKGROUP", fallback);
  }

  static unsigned autolykos_prehash_workgroup(const sycl::device& dev) {
    const unsigned fallback = sycl_default_workgroup(dev, {32, 64, 128, 256}, 64);
    return env_workgroup("MOM_AUTOLYKOS2_PREHASH_WORKGROUP", fallback);
  }

  // Accept an env override only if it is one of the supported workgroup sizes.
  static unsigned env_workgroup(const char* name, const unsigned fallback) {
    uint32_t parsed = 0;
    if (env_u32(name, parsed) && (parsed == 32 || parsed == 64 || parsed == 128 || parsed == 256)) {
      return parsed;
    }
    return fallback;
  }

  static const char* autolykos_compile_options(const sycl::device& dev) {
    const char* const value = std::getenv("MOM_AUTOLYKOS2_COMPILE_OPTIONS");
    if (value) return value;
    // SYCL_PROGRAM_COMPILE_OPTIONS is process-global; pearlhash's ESIMD (VC-backend) image rejects "-O3",
    // and it's a measured no-op for this kernel anyway (36.44 vs 36.41 MH/s on B580). Override via
    // MOM_AUTOLYKOS2_COMPILE_OPTIONS if needed.
    (void)dev;
    return "";
  }

  // Unlike the kawpow DAG chunking this defaults on: at live N the monolithic build is one
  // multi-second GPU job, the long-running-job shape that tripped xe/GuC scheduler cleanup
  // crashes in June 2026. ~3 s chunks cost nothing on the in-order queue. 0 = single kernel.
  static uint32_t autolykos_table_chunk_rows() {
    constexpr uint32_t fallback = 32U * 1024U * 1024U;
    uint32_t parsed = 0;
    return env_u32("MOM_AUTOLYKOS2_TABLE_CHUNK", parsed) ? parsed : fallback;
  }

  // 0 disables the post-build row check, 1 (default) cross-checks a strided sample of rows
  // against autolykos_prehash_digest_dev, 2 recomputes every row (slow, for debugging).
  static unsigned autolykos_table_verify_mode() {
    uint32_t parsed = 0;
    return env_u32("MOM_AUTOLYKOS2_VERIFY_TABLE", parsed) && parsed <= 2 ? parsed : 1;
  }

  static bool autolykos_split_table() {
    const char* const value = std::getenv("MOM_AUTOLYKOS2_SPLIT");
    return !(value && value[0] == '0');
  }

  // CUDA/HIP path: eight lanes own one 256-bit accumulator. This lowers the scalar kernel's register
  // footprint and turns every random table row into one contiguous 32-byte subgroup transaction.
  // The old HIP-only variable remains an alias for existing diagnostic scripts.
  static bool autolykos_subgroup_coop() {
    const char* value = std::getenv("MOM_AUTOLYKOS2_SUBGROUP_COOP");
    if (!value) value = std::getenv("MOM_AUTOLYKOS2_HIP_COOP");
    return !value || (std::strcmp(value, "0") && std::strcmp(value, "false"));
  }

  static bool autolykos_subgroup_coop_verify() {
    const char* value = std::getenv("MOM_AUTOLYKOS2_SUBGROUP_COOP_VERIFY");
    if (!value) value = std::getenv("MOM_AUTOLYKOS2_HIP_COOP_VERIFY");
    return value && (!std::strcmp(value, "1") || !std::strcmp(value, "true"));
  }

  void release() {
    queue.wait_and_throw();
    free_ptr(table);
    free_ptr(bhashes);
    if constexpr (mom_sycl_portable_opencl) {
      portable_results.release();
      portable_result_cap = 0;
    } else {
      free_ptr(result);
    }
    free_ptr(table_mismatches);
    table_height = UINT32_MAX;
    table_n = 0;
    table_cap_n = 0;
    bhashes_cap = 0;
  }

  void free_ptr(auto*& ptr) {
    if (ptr) sycl::free(ptr, queue);
    ptr = nullptr;
  }

  void ensure_result() {
    if constexpr (!mom_sycl_portable_opencl) {
      if (result) return;
      result = sycl::malloc_shared<AutolykosResult>(1, queue);
      if (!result) throw std::string("Can't allocate autolykos2 SYCL result buffer");
    }
  }

  void ensure_portable_results(const uint32_t count) {
    if constexpr (!mom_sycl_portable_opencl) return;
    if (portable_result_cap >= count) return;
    queue.wait_and_throw();
    portable_results.allocate(count);
    portable_result_cap = count;
  }

  void ensure_bhashes(const uint32_t intensity) {
    if (bhashes && bhashes_cap >= intensity) return;
    queue.wait_and_throw();
    free_ptr(bhashes);
    bhashes_cap = intensity;
    bhashes = sycl::malloc_device<uint32_t>(static_cast<uint64_t>(intensity) * 8U, queue);
    if (!bhashes) throw std::string("Can't allocate autolykos2 intermediate buffer");
  }

  bool should_use_table(const bool is_test) const {
    if constexpr (mom_sycl_portable_opencl) {
      // Some conforming ICDs cap individual buffers at 2 GiB. Recompute rows instead of requiring
      // the ~6.9-GiB fast table in the no-USM profile.
      (void)is_test;
      return false;
    } else {
      if (is_test || device.is_cpu()) return false;
      const char* const value = std::getenv("MOM_AUTOLYKOS2_TABLE");
      return !(value && value[0] == '0');
    }
  }

  // Recompute a sample of the freshly built table with the reference digest and throw on
  // any difference: the fast build kernel folds the constant pad, and a silently wrong
  // table would invalidate every share until the next block.
  void verify_table(const uint32_t height, const uint32_t n_len) {
    const unsigned mode = autolykos_table_verify_mode();
    if (!mode) return;

    if (!table_mismatches) {
      table_mismatches = sycl::malloc_shared<uint32_t>(1, queue);
      if (!table_mismatches) throw std::string("Can't allocate autolykos2 verify counter");
    }
    *table_mismatches = 0;

    const uint32_t stride = mode >= 2 ? 1 : 65521; // prime; ~3.3k of 216M rows in default mode
    const uint32_t count = (n_len + stride - 1) / stride;
    const uint32_t* const d_table = table;
    uint32_t* const d_mismatches = table_mismatches;
    const uint32_t local = prehash_workgroup;

    sycl_wait_and_throw(queue.submit([&](sycl::handler& h) {
      mom_use_bundle(h, *bundle);
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(round_up(count, local)), sycl::range<1>(local)),
        [=](sycl::nd_item<1> item) {
          const uint32_t gid = item.get_global_id(0);
          if (gid >= count) return;
          const uint32_t row = gid * stride;
          if (row >= n_len) return;

          uint8_t digest[32];
          autolykos_prehash_digest_dev(row, height, digest);
          uint32_t expected[8];
          digest_to_limbs_dev(digest, expected);

          const uint32_t* const stored = d_table + static_cast<uint64_t>(row) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
          bool differs = false;
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) differs |= stored[i] != expected[i];
          if (differs) {
            sycl::atomic_ref<
              uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
              sycl::access::address_space::global_space
            >(*d_mismatches).fetch_add(1);
          }
        }
      );
    }), device);

    if (*table_mismatches) {
      throw std::string("Autolykos2 table verification failed: ") +
            std::to_string(*table_mismatches) + " of " + std::to_string(count) + " sampled rows differ";
    }
  }

  void ensure_table(const uint32_t height, const uint32_t n_len, const bool is_test, const bool should_log) {
    if (!should_use_table(is_test)) {
      queue.wait_and_throw();
      free_ptr(table);
      table_height = UINT32_MAX;
      table_n = 0;
      table_cap_n = 0;
      return;
    }

    if (table && table_height == height && table_n == n_len) return;
    if (!mom_has_usm_device(device)) {
      throw std::string("autolykos2 SYCL GPU device does not support device allocations");
    }

    const uint64_t start_ms = now_ms();
    table_height = UINT32_MAX;
    if (!table || table_cap_n != n_len) {
      queue.wait_and_throw();
      free_ptr(table);
      table_cap_n = 0;
      table = sycl::malloc_device<uint32_t>(
        static_cast<uint64_t>(n_len) * (TABLE_ENTRY_BYTES / sizeof(uint32_t)), queue);
      if (!table) throw std::string("Can't allocate autolykos2 table");
      table_cap_n = n_len;
    }

    uint32_t* const d_table = table;
    sycl::queue& q = queue;
    auto& kb = *bundle;
    const uint32_t local = prehash_workgroup;

    const uint32_t chunk_rows = autolykos_table_chunk_rows();
    sycl::event build_event;
    for (uint32_t start_row = 0; start_row < n_len;) {
      const uint32_t current_rows = chunk_rows ? std::min(chunk_rows, n_len - start_row) : n_len;
      const uint32_t chunk_start = start_row;
      build_event = q.submit([&](sycl::handler& h) {
        mom_use_bundle(h, kb);
        h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(round_up(current_rows, local)), sycl::range<1>(local)),
          [=](sycl::nd_item<1> item) {
            const uint32_t local_row = item.get_global_id(0);
            if (local_row >= current_rows) return;
            autolykos_table_row_store_dev(d_table, chunk_start + local_row, height);
          }
        );
      });
      if (!chunk_rows) break;
      start_row += current_rows;
    }
    sycl_wait_and_throw(build_event, device);
    verify_table(height, n_len);

    table_height = height;
    table_n = n_len;
    if (should_log || std::getenv("MOM_LOOP_STATS")) {
      char elapsed[32];
      format_duration_ms(elapsed, sizeof(elapsed), now_ms() - start_ms);
      std::fprintf(stderr, "Autolykos2 table for height %u N %u calculated (%s)\n", height, n_len, elapsed);
    }
  }
};

using AutolykosStateMap = std::map<std::string, std::unique_ptr<AutolykosState>>;
