class State {
public:
  sycl::device device;
  sycl::queue queue;
  bool shared_io;
  bool shared_dag;
  UsmPtr<uint8_t> input;
  UsmPtr<uint8_t> target;
  UsmPtr<uint32_t> light_cache;
  UsmPtr<uint32_t> dag;
  UsmPtr<EtchashResult> result;
  uint32_t input_cap = 0;
  uint64_t light_cache_words = 0;
  uint64_t dag_words = 0;
  uint32_t epoch = UINT32_MAX;
  uint32_t seed_epoch = UINT32_MAX;
  // Avoid the seed-hash linear lookup on every batch.
  uint8_t cached_seed_hash[HASH_LEN]{};
  uint32_t cached_seed_epoch = UINT32_MAX;
  std::mutex mutex;
  int inline_pair_ok =
      -1; // -1 untested, 1 = inlined pair Keccak is bit-exact here (fast path), 0 = miscompiled (noinline)

  // Per-device gate for the etchash search Keccak. The fast pre-0.7.0 path inlines the Uint2 pair
  // permutation; IGC 2.36.x miscompiles that under the search kernel's register pressure
  // (intel-graphics-compiler#414 family). use_inline_pair() runs a one-time self-test that executes the REAL search kernel
  // both ways on a tiny synthetic DAG and only enables the inlined path if it is byte-for-byte equal to
  // the noinline reference (which ships correct on every device). DAG-gen always uses noinline.
  bool use_inline_pair() {
    if (inline_pair_ok < 0) {
      if (mom_env_flag("MOM_ETCHASH_FORCE_NOINLINE").value_or(false)) {
        inline_pair_ok = 0;
        std::fprintf(
            stderr, "Etchash: MOM_ETCHASH_FORCE_NOINLINE set -> using safe noinline search path\n");
      } else if (mom_env_flag("MOM_ETCHASH_FORCE_INLINE").value_or(false)) {
        inline_pair_ok = 1;
        std::fprintf(stderr,
                     "Etchash: MOM_ETCHASH_FORCE_INLINE set -> using fast inlined search path\n");
      } else {
        inline_pair_ok = probe_inline_pair() ? 1 : 0;
        if (!inline_pair_ok || mom_env_flag("MOM_ETCHASH_SELFTEST_LOG").value_or(false))
          std::fprintf(stderr,
                       "Etchash: inlined pair-Keccak self-test %s -> using %s search path\n",
                       inline_pair_ok ? "passed" : "FAILED (IGC miscompile)",
                       inline_pair_ok ? "fast inlined" : "safe noinline");
      }
    }
    return inline_pair_ok == 1;
  }

  // Compare real search kernels on a synthetic DAG; submitted faults must propagate.
  bool probe_inline_pair() {
    constexpr uint32_t PAGES = 4096; // tiny synthetic DAG: 4096 * 128 B = 512 KiB
    constexpr uint32_t WORDS = PAGES * ETHASH_MIX_WORDS;
    constexpr uint32_t NONCES = 256; // the miscompile is total+deterministic; 256 is ample
    uint32_t* d_dag_t = nullptr;
    uint8_t* d_in_t = nullptr;
    uint8_t* d_tgt_t = nullptr;
    EtchashResult* r_inl = nullptr;
    EtchashResult* r_ref = nullptr;
    // Upload sources must outlive the catch-path drain of accepted copies.
    std::vector<uint32_t> hdag;
    uint8_t hin[40], htgt[HASH_LEN];
    bool ok = false;
    bool submitted = false;
    const auto release = [&](const bool propagate_error) {
      void* const pointers[] = {d_dag_t, d_in_t, d_tgt_t, r_inl, r_ref};
      std::exception_ptr error;
      for (void* const pointer : pointers)
        if (pointer) {
          try {
            sycl::free(pointer, queue);
          } catch (...) {
            if (!error)
              error = std::current_exception();
          }
        }
      if (propagate_error && error)
        std::rethrow_exception(error);
    };
    try {
      d_dag_t = sycl::malloc_device<uint32_t>(WORDS, queue);
      d_in_t = sycl::malloc_device<uint8_t>(40, queue);
      d_tgt_t = sycl::malloc_device<uint8_t>(HASH_LEN, queue);
      r_inl = sycl::malloc_shared<EtchashResult>(1, queue);
      r_ref = sycl::malloc_shared<EtchashResult>(1, queue);
      if (!d_dag_t || !d_in_t || !d_tgt_t || !r_inl || !r_ref)
        throw std::string("Can't allocate Etchash inline-pair probe");

      hdag.resize(WORDS); // deterministic fill; content is irrelevant, only equality is
      uint32_t s = 0x9e3779b9u;
      for (auto& v : hdag) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v = s;
      }
      for (unsigned i = 0; i < 40; ++i)
        hin[i] = static_cast<uint8_t>(i * 7 + 1);
      std::memset(
          htgt, 0,
          HASH_LEN); // all-zero target: meets_target never fires, only is_test stores nonce 0
      submitted = true;
      queue.memcpy(d_dag_t, hdag.data(), WORDS * sizeof(uint32_t));
      queue.memcpy(d_in_t, hin, 40);
      queue.memcpy(d_tgt_t, htgt, HASH_LEN);
      queue.wait_and_throw();

      const FastModData m = make_fast_mod_data(PAGES);
      std::memset(r_inl, 0, sizeof(EtchashResult));
      sycl_wait_and_throw(submit_etchash_search_gpu<true>(queue, d_in_t, 0, d_dag_t, m, NONCES,
                                                          d_tgt_t, r_inl, true),
                          device);
      std::memset(r_ref, 0, sizeof(EtchashResult));
      sycl_wait_and_throw(submit_etchash_search_gpu<false>(queue, d_in_t, 0, d_dag_t, m, NONCES,
                                                           d_tgt_t, r_ref, true),
                          device);

      ok = r_inl->count == 1 && r_ref->count == 1 &&
           std::memcmp(r_inl->hash[0], r_ref->hash[0], HASH_LEN) == 0 &&
           std::memcmp(r_inl->mix_hash[0], r_ref->mix_hash[0], HASH_LEN) == 0;
    } catch (...) {
      if (submitted)
        sycl_cleanup_noexcept("etchash probe wait", [&] { queue.wait_and_throw(); });
      release(!submitted);
      // Allocation absence is optional; a submitted fault must not select another search kernel.
      if (submitted)
        throw;
      return false;
    }
    release(true);
    return ok;
  }

  // Bind allocation policy and compile options to one in-order queue.
  explicit State(const std::string& dev_str)
      : device(get_dev(dev_str)),
        queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
        shared_io(device.is_cpu() || !mom_has_usm_device(device)),
        shared_dag(device.is_cpu() || !mom_has_usm_device(device)) {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device))) {
      throw std::string("etchash SYCL device does not support required allocations");
    }
  }

  // Drain accepted work before releasing persistent storage.
  ~State() {
    sycl_cleanup_noexcept("etchash wait", [&] { queue.wait_and_throw(); });
    sycl_cleanup_noexcept("etchash release", [&] { release(); });
  }

  // DAG nodes per dispatch, or 0 for one dispatch; MOM_ETCHASH_DAG_CHUNK_NODES overrides.
  static uint32_t etchash_dag_chunk_nodes(const sycl::device& dev) {
    uint32_t parsed;
    if (mom_parse_env_u32("MOM_ETCHASH_DAG_CHUNK_NODES", parsed))
      return parsed;
    // Integrated GPUs also serve the desktop. Retire short DAG-generation command lists instead
    // of leaving one epoch-sized kernel in the driver long enough to trigger its hang watchdog.
    if (is_integrated_gpu(dev))
      return 1u << 12;
    return 0;
  }

  // Cache pool-seed resolution separately from the fork-adjusted DAG epoch. Pool heights are
  // unbounded, so reject epochs past the libethash size tables before the caller indexes them.
  void resolve_epochs(const uint32_t block_height, const uint8_t* const seed_hash,
                      uint32_t& epoch, uint32_t& seed_epoch) {
    if (!seed_hash || is_zero_hash(seed_hash)) {
      epoch = etchash_epoch(block_height);
      seed_epoch = etchash_seed_epoch(block_height);
    } else {
      if (cached_seed_epoch == UINT32_MAX ||
          std::memcmp(cached_seed_hash, seed_hash, HASH_LEN) != 0) {
        const uint32_t resolved_seed_epoch = seed_epoch_from_hash(seed_hash);
        std::memcpy(cached_seed_hash, seed_hash, HASH_LEN);
        cached_seed_epoch = resolved_seed_epoch;
      }
      seed_epoch = cached_seed_epoch;
      epoch = seed_epoch < ETCHASH_FORK_BLOCK / ETHASH_EPOCH ? seed_epoch : seed_epoch / 2;
    }
    if (epoch >= sizeof(dag_sizes) / sizeof(dag_sizes[0]))
      throw std::string("Bad etchash epoch");
  }

  // Free allocations and invalidate epoch/capacity keys.
  void release() {
    input.reset();
    target.reset();
    light_cache.reset();
    dag.reset();
    result.reset();
    input_cap = 0;
    light_cache_words = 0;
    dag_words = 0;
    epoch = UINT32_MAX;
    seed_epoch = UINT32_MAX;
  }

  // Grow job buffers after queue completion; failed allocation leaves capacity invalid.
  void ensure_input(const unsigned input_size) {
    if (input && input_size <= input_cap && result && target)
      return;
    queue.wait_and_throw();
    input.reset();
    target.reset();
    result.reset();
    input_cap = 0;
    constexpr const char* failure = "Can't allocate etchash SYCL input buffers";
    input = mom_usm_alloc<uint8_t>(queue, input_size, shared_io, failure);
    target = mom_usm_alloc<uint8_t>(queue, HASH_LEN, shared_io, failure);
    result = mom_usm_alloc<EtchashResult>(queue, 1, true, failure);
    input_cap = input_size;
  }

  // Validate dimensions and publish the rebuilt key only after all DAG chunks finish.
  void ensure_epoch(const uint32_t new_epoch, const uint32_t new_seed_epoch,
                    const uint64_t new_cache_bytes, const uint64_t new_dag_bytes,
                    const bool should_log) {
    // The entrypoint supplies consensus-table sizes for ordinary calls and only supplies the
    // bounded fixture sizes for the explicitly gated portable CPU test.
    if (new_epoch >= 2048 || !new_cache_bytes || !new_dag_bytes ||
        new_cache_bytes % (ETHASH_NODE_WORDS * sizeof(uint32_t)) != 0 ||
        new_dag_bytes % ETHASH_MIX_BYTES != 0) {
      throw std::string("Bad etchash epoch");
    }

    const uint64_t new_light_cache_words = new_cache_bytes / sizeof(uint32_t);
    const uint64_t new_dag_words = new_dag_bytes / sizeof(uint32_t);
    const uint64_t new_dag_nodes = new_dag_bytes / (ETHASH_NODE_WORDS * sizeof(uint32_t));
    if (!ethash_dag_words_addressable(new_dag_words))
      throw std::string("Etchash DAG exceeds 32-bit word addressing");
    if (epoch == new_epoch && seed_epoch == new_seed_epoch &&
        light_cache_words == new_light_cache_words && dag_words == new_dag_words && light_cache &&
        dag)
      return;

    std::vector<uint32_t> host_cache;
    const uint64_t start_ms = mom_now_ms();
    compute_light_cache(host_cache, new_cache_bytes, new_seed_epoch,
                        "Can't calculate etchash light cache");

    queue.wait_and_throw();
    // Invalidate the old key before an in-place rebuild can fail; it must not remain reusable.
    epoch = UINT32_MAX;
    seed_epoch = UINT32_MAX;
    constexpr const char* failure = "Can't allocate etchash DAG buffers";
    mom_resize_usm(queue, light_cache, light_cache_words, new_light_cache_words, shared_dag, failure);
    mom_resize_usm(queue, dag, dag_words, new_dag_words, shared_dag, failure);

    upload_ethash_light_cache(queue, device, light_cache, host_cache, shared_dag, "etchash cache upload");

    const uint32_t light_nodes = static_cast<uint32_t>(new_light_cache_words / ETHASH_NODE_WORDS);
    generate_ethash_dag<ETHASH_DATASET_PARENTS2>(
        queue, device, light_cache, light_nodes, dag, static_cast<uint32_t>(new_dag_nodes),
        ethash_dag_workgroup(device, "MOM_ETCHASH_DAG_WORKGROUP"), etchash_dag_chunk_nodes(device));

    epoch = new_epoch;
    seed_epoch = new_seed_epoch;
    if (should_log) {
      char elapsed[32];
      mom_format_duration_ms(elapsed, sizeof(elapsed), mom_now_ms() - start_ms);
      std::fprintf(stderr, "Etchash DAG for epoch %u seed epoch %u calculated (%s)\n", new_epoch,
                   new_seed_epoch, elapsed);
    }
  }
};

// Release device state before the compiler runtime unloads without escaping exceptions.
void etchash_cleanup_states() noexcept {
  clearPersistentDeviceStates<State>("etchash");
}
