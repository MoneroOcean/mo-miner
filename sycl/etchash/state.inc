class EtchashState {
public:
  sycl::device device;
  sycl::queue queue;
  std::unique_ptr<MomKernelBundle> bundle;
  bool shared_io;
  bool shared_dag;
  uint8_t* input = nullptr;
  uint8_t* target = nullptr;
  uint32_t* light_cache = nullptr;
  uint32_t* dag = nullptr;
  EtchashResult* result = nullptr;
  uint32_t input_cap = 0;
  uint64_t light_cache_words = 0;
  uint64_t dag_words = 0;
  uint32_t epoch = UINT32_MAX;
  uint32_t seed_epoch = UINT32_MAX;
  std::mutex mutex;
  int inline_pair_ok = -1;  // -1 untested, 1 = inlined pair Keccak is bit-exact here (fast path), 0 = miscompiled (noinline)

  // Per-device gate for the etchash search Keccak. The fast pre-0.7.0 path inlines the Uint2 pair
  // permutation; IGC 2.36.x miscompiles that under the search kernel's register pressure (see
  // ~/bug_report5). use_inline_pair() runs a one-time self-test that executes the REAL search kernel
  // both ways on a tiny synthetic DAG and only enables the inlined path if it is byte-for-byte equal to
  // the noinline reference (which ships correct on every device). DAG-gen always uses noinline.
  bool use_inline_pair() {
    if (inline_pair_ok < 0) {
      unsigned long forced = 0;
      if (mom_parse_env_ulong("MOM_ETCHASH_FORCE_NOINLINE", forced) && forced) {
        inline_pair_ok = 0;
        std::fprintf(stderr, "Etchash: MOM_ETCHASH_FORCE_NOINLINE set -> using safe noinline search path\n");
      } else if (mom_parse_env_ulong("MOM_ETCHASH_FORCE_INLINE", forced) && forced) {
        inline_pair_ok = 1;
        std::fprintf(stderr, "Etchash: MOM_ETCHASH_FORCE_INLINE set -> using fast inlined search path\n");
      } else {
        inline_pair_ok = probe_inline_pair() ? 1 : 0;
        if (!inline_pair_ok || std::getenv("MOM_ETCHASH_SELFTEST_LOG"))
          std::fprintf(stderr, "Etchash: inlined pair-Keccak self-test %s -> using %s search path\n",
            inline_pair_ok ? "passed" : "FAILED (IGC miscompile)", inline_pair_ok ? "fast inlined" : "safe noinline");
      }
    }
    return inline_pair_ok == 1;
  }

  bool probe_inline_pair() {
    constexpr uint32_t PAGES = 4096;                       // tiny synthetic DAG: 4096 * 128 B = 512 KiB
    constexpr uint32_t WORDS = PAGES * ETHASH_MIX_WORDS;
    constexpr uint32_t NONCES = 256;                       // the miscompile is total+deterministic; 256 is ample
    uint32_t* d_dag_t = nullptr; uint8_t* d_in_t = nullptr; uint8_t* d_tgt_t = nullptr;
    EtchashResult* r_inl = nullptr; EtchashResult* r_ref = nullptr;
    bool ok = false;
    try {
      d_dag_t = sycl::malloc_device<uint32_t>(WORDS, queue);
      d_in_t  = sycl::malloc_device<uint8_t>(40, queue);
      d_tgt_t = sycl::malloc_device<uint8_t>(HASH_LEN, queue);
      r_inl = sycl::malloc_shared<EtchashResult>(1, queue);
      r_ref = sycl::malloc_shared<EtchashResult>(1, queue);

      std::vector<uint32_t> hdag(WORDS);                   // deterministic fill; content is irrelevant, only equality is
      uint32_t s = 0x9e3779b9u;
      for (auto& v : hdag) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; v = s; }
      uint8_t hin[40];  for (unsigned i = 0; i < 40; ++i) hin[i] = static_cast<uint8_t>(i * 7 + 1);
      uint8_t htgt[HASH_LEN]; std::memset(htgt, 0, HASH_LEN);   // all-zero target: meets_target never fires, only is_test stores nonce 0
      queue.memcpy(d_dag_t, hdag.data(), WORDS * sizeof(uint32_t));
      queue.memcpy(d_in_t, hin, 40);
      queue.memcpy(d_tgt_t, htgt, HASH_LEN);
      queue.wait();

      const FastModData m = make_fast_mod_data(PAGES);
      std::memset(r_inl, 0, sizeof(EtchashResult));
      sycl_wait_and_throw(submit_etchash_search_gpu<true >(queue, *bundle, d_in_t, 0, d_dag_t, m, NONCES, d_tgt_t, r_inl, true), device);
      std::memset(r_ref, 0, sizeof(EtchashResult));
      sycl_wait_and_throw(submit_etchash_search_gpu<false>(queue, *bundle, d_in_t, 0, d_dag_t, m, NONCES, d_tgt_t, r_ref, true), device);

      ok = r_inl->count == 1 && r_ref->count == 1 &&
           std::memcmp(r_inl->output[0],   r_ref->output[0],   HASH_LEN) == 0 &&
           std::memcmp(r_inl->mix_hash[0], r_ref->mix_hash[0], HASH_LEN) == 0;
    } catch (...) {
      ok = false;  // any failure -> conservatively use the safe noinline path
    }
    if (d_dag_t) sycl::free(d_dag_t, queue);
    if (d_in_t)  sycl::free(d_in_t,  queue);
    if (d_tgt_t) sycl::free(d_tgt_t, queue);
    if (r_inl)   sycl::free(r_inl,   queue);
    if (r_ref)   sycl::free(r_ref,   queue);
    return ok;
  }

  explicit EtchashState(const std::string& dev_str)
    : device(get_dev(dev_str)),
      queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !mom_has_usm_device(device)),
      shared_dag(device.is_cpu() || !mom_has_usm_device(device))
  {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device))) {
      throw std::string("etchash SYCL device does not support required allocations");
    }

    set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", etchash_compile_options());
    bundle = std::make_unique<MomKernelBundle>(
      mom_get_exec_bundle(queue.get_context())
    );
  }

  ~EtchashState() { sycl_cleanup_noexcept("etchash", [&] { release(); }); }

  static unsigned etchash_dag_workgroup(const sycl::device& dev) {
    const unsigned fallback = sycl_default_workgroup(dev, {32, 64, 128, 256, 512}, dev.is_cpu() ? 128 : 64);
    unsigned long parsed = 0;
    if (!mom_parse_env_ulong("MOM_ETCHASH_DAG_WORKGROUP", parsed)) return fallback;
    switch (parsed) {
      case 32:
      case 64:
      case 128:
      case 256:
      case 512:
        return static_cast<unsigned>(parsed);
    }
    return fallback;
  }

  static uint32_t etchash_dag_chunk_nodes() {
    unsigned long parsed = 0;
    if (!mom_parse_env_ulong("MOM_ETCHASH_DAG_CHUNK_NODES", parsed) ||
        parsed > std::numeric_limits<uint32_t>::max()) return 0;
    return static_cast<uint32_t>(parsed);
  }

  static const char* etchash_compile_options() {
    const char* const value = std::getenv("MOM_ETCHASH_COMPILE_OPTIONS");
    // Default empty: "-O3" is process-global and PearlHash's ESIMD image rejects it
    // (and it is a no-op for etchash). Override via MOM_ETCHASH_COMPILE_OPTIONS if needed.
    return value ? value : "";
  }

  template<typename T>
  T* allocate(const size_t count, const bool shared) {
    return shared ? sycl::malloc_shared<T>(count, queue) : sycl::malloc_device<T>(count, queue);
  }

  void free_ptr(auto*& ptr) {
    if (ptr) sycl::free(ptr, queue);
    ptr = nullptr;
  }

  void release() {
    queue.wait_and_throw();
    free_ptr(input);
    free_ptr(target);
    free_ptr(light_cache);
    free_ptr(dag);
    free_ptr(result);
    input_cap = 0;
    light_cache_words = 0;
    dag_words = 0;
    epoch = UINT32_MAX;
    seed_epoch = UINT32_MAX;
  }

  void ensure_input(const unsigned input_size) {
    if (input_size <= input_cap && result && target) return;
    queue.wait_and_throw();
    free_ptr(input);
    free_ptr(target);
    free_ptr(result);
    input_cap = input_size;
    input = allocate<uint8_t>(input_cap, shared_io);
    target = allocate<uint8_t>(HASH_LEN, shared_io);
    result = sycl::malloc_shared<EtchashResult>(1, queue);
    if (!input || !target || !result) throw std::string("Can't allocate etchash SYCL input buffers");
  }

  void ensure_epoch(const uint32_t new_epoch, const uint32_t new_seed_epoch, const bool should_log) {
    if (epoch == new_epoch && seed_epoch == new_seed_epoch) return;
    if (new_epoch >= 2048 || !dag_sizes[new_epoch] || !cache_sizes[new_epoch]) {
      throw std::string("Bad etchash epoch");
    }

    const uint64_t new_light_cache_words = cache_sizes[new_epoch] / sizeof(uint32_t);
    const uint64_t new_dag_words = dag_sizes[new_epoch] / sizeof(uint32_t);
    const uint64_t new_dag_nodes = dag_sizes[new_epoch] / (ETHASH_NODE_WORDS * sizeof(uint32_t));

    std::vector<uint32_t> host_cache;
    const uint64_t start_ms = now_ms();
    compute_light_cache(host_cache, new_epoch, new_seed_epoch);

    queue.wait_and_throw();
    if (new_light_cache_words != light_cache_words) {
      free_ptr(light_cache);
      light_cache = allocate<uint32_t>(new_light_cache_words, shared_dag);
      light_cache_words = new_light_cache_words;
    }
    if (new_dag_words != dag_words) {
      free_ptr(dag);
      dag = allocate<uint32_t>(new_dag_words, shared_dag);
      dag_words = new_dag_words;
    }
    if (!light_cache || !dag) throw std::string("Can't allocate etchash DAG buffers");

    if (shared_dag) std::memcpy(light_cache, host_cache.data(), host_cache.size() * sizeof(uint32_t));
    else sycl_wait_and_throw(queue.memcpy(light_cache, host_cache.data(), host_cache.size() * sizeof(uint32_t)), device);

    const uint32_t light_nodes = static_cast<uint32_t>(new_light_cache_words / ETHASH_NODE_WORDS);
    const FastModData light_mod = make_fast_mod_data(light_nodes);
    const uint32_t dag_workgroup = etchash_dag_workgroup(device);
    const uint32_t total = static_cast<uint32_t>(new_dag_nodes);
    const uint32_t chunk_nodes = etchash_dag_chunk_nodes();
    uint32_t* const d_light = light_cache;
    uint32_t* const d_dag = dag;
    sycl::queue& q = queue;
    auto& kb = *bundle;

    sycl::event dag_event;
    for (uint32_t start_node = 0; start_node < total;) {
      const uint32_t current_nodes = chunk_nodes ? std::min(chunk_nodes, total - start_node) : total;
      const uint32_t chunk_start = start_node;
      dag_event = q.submit([&](sycl::handler& h) {
        mom_use_bundle(h, kb);
        h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(round_up(current_nodes, dag_workgroup)), sycl::range<1>(dag_workgroup)),
          [=](sycl::nd_item<1> item) {
            const uint32_t local_node = item.get_global_id(0);
            if (local_node >= current_nodes) return;
            const uint32_t node_index = chunk_start + local_node;

            uint32_t dag_node[ETHASH_NODE_WORDS];
            const uint32_t init = fast_mod_dev(node_index, light_mod);
#pragma unroll
            for (unsigned w = 0; w < ETHASH_NODE_WORDS; ++w) dag_node[w] = d_light[init * ETHASH_NODE_WORDS + w];
            dag_node[0] ^= node_index;
            keccak_512_words_dev(dag_node);

            for (uint32_t i = 0; i < ETHASH_DATASET_PARENTS2; ++i) {
              const uint32_t parent_index = fast_mod_dev(fnv_dev(node_index ^ i, dag_node[i & (ETHASH_NODE_WORDS - 1)]), light_mod);
              const uint32_t parent_base = parent_index * ETHASH_NODE_WORDS;
#pragma unroll
              for (unsigned w = 0; w < ETHASH_NODE_WORDS; ++w) {
                dag_node[w] = fnv_dev(dag_node[w], d_light[parent_base + w]);
              }
            }
            keccak_512_words_dev(dag_node);

            const uint32_t base = node_index * ETHASH_NODE_WORDS;
#pragma unroll
            for (unsigned w = 0; w < ETHASH_NODE_WORDS; ++w) d_dag[base + w] = dag_node[w];
          }
        );
      });
      if (!chunk_nodes) break;
      start_node += current_nodes;
    }
    sycl_wait_and_throw(dag_event, device);

    epoch = new_epoch;
    seed_epoch = new_seed_epoch;
    if (should_log) {
      char elapsed[32];
      format_duration_ms(elapsed, sizeof(elapsed), now_ms() - start_ms);
      std::fprintf(stderr, "Etchash DAG for epoch %u seed epoch %u calculated (%s)\n", new_epoch, new_seed_epoch, elapsed);
    }
  }

  static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()
    ).count();
  }
};

using EtchashStateMap = std::map<std::string, std::unique_ptr<EtchashState>>;

static EtchashStateMap& etchash_states() {
  static EtchashStateMap* const states = new EtchashStateMap;
  return *states;
}

static std::mutex& etchash_states_mutex() {
  static std::mutex* const mutex = new std::mutex;
  return *mutex;
}

static EtchashState& etchash_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(etchash_states_mutex());
  auto& state = etchash_states()[dev_str];
  if (!state) state = std::make_unique<EtchashState>(dev_str);
  return *state;
}

void etchash_cleanup_states() noexcept {
  try {
    std::lock_guard<std::mutex> lock(etchash_states_mutex());
    etchash_states().clear();
  } catch (...) {
    std::fprintf(stderr, "etchash: ordered SYCL cleanup failed\n");
  }
}

} // namespace mom_etchash

using namespace mom_etchash;
