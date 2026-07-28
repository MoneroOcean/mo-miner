class KawpowSgKernel;

static bool kawpow_default_subgroup_exchange(const sycl::device& device) {
  if (!device.is_gpu()) return false;
  if (!sycl_is_level_zero_gpu(device)) return true;
  const uint32_t build = mom_driver_build(device);
  // NEO 32224 aborts inside the subgroup-exchange ProgPoW kernel on Arc A-series, while the common
  // local-memory exchange is bit-exact and reached 21.8 MH/s on A770. Current NEO 37020 executes the
  // subgroup path correctly and it is 26% faster on B580, so keep that fast path outside the affected
  // driver family. MOM_KAWPOW_EXCHANGE=sg|slm remains an explicit diagnostic override.
  return build < 30000u || build >= 35000u;
}

class KawpowState {
public:
  sycl::device device;
  sycl::queue queue;
  std::unique_ptr<MomKernelBundle> bundle;
  bool shared_io;
  bool shared_dag;
  uint8_t* input = nullptr;
  uint32_t* light_cache = nullptr;
  uint32_t* dag = nullptr;
  KawpowResult* result = nullptr;
  uint32_t input_cap = 0;
  uint64_t light_cache_words = 0;
  uint64_t dag_words = 0;
  uint32_t epoch = UINT32_MAX;
  // Tracks the DAG byte size the current DAG was built with. KawPoW and FiroPow use different dataset
  // sizing (classic vs chfast), so the same epoch can map to different DAG sizes; the DAG is rebuilt
  // when this changes, not just when the epoch changes.
  uint64_t dag_bytes = 0;
  uint64_t period = UINT64_MAX;
  KawpowProgram program{};
  unsigned workgroup;
  bool default_subgroup_exchange;
  std::mutex mutex;

  // Active ProgPoW seal (set per-call from the variant, under `mutex`). On CUDA it is baked into the
  // JIT source (kawpow_emit_seal); on Intel/spir64 it rides in via SealParams. A device's KawpowState
  // is shared across the kawpow/firopow/evrprogpow entrypoints, so the seal can change between calls
  // and participates in the bundle cache key below (a seal change forces a rebuild).
  SealMode seal = SealMode::KAWPOW;
  uint32_t magic[15] = {};
  // Active inner-loop shape (set per-call from the variant, under `mutex`, like seal/magic). MeowPow
  // (16/6/9) differs from the 32/11/18 of the other variants; it drives make_program() and selects the
  // search-kernel template instantiation. Participates in the bundle cache key (a shape change rebuilds:
  // the CUDA JIT bakes a different program; the Intel kernel branches on it but the baked program
  // changes shape). regs<=KAWPOW_REGS, cnt_cache<=KAWPOW_CNT_CACHE, cnt_math<=KAWPOW_CNT_MATH.
  uint32_t regs = KAWPOW_REGS;
  uint32_t cnt_cache = KAWPOW_CNT_CACHE;
  uint32_t cnt_math = KAWPOW_CNT_MATH;

  // Per-period executable bundles (Intel: program/dag spec constants baked in; CUDA: source-JIT'd
  // kernel). The next period's bundle is built on a worker thread while the current one mines, so
  // the period switch no longer stalls hashing on the build.
  std::unique_ptr<MomKernelBundle> period_bundle;
  uint64_t period_bundle_period = UINT64_MAX;
  uint32_t period_bundle_epoch = UINT32_MAX;
  SealMode period_bundle_seal = SealMode::KAWPOW;
  uint32_t period_bundle_magic[15] = {};
  uint32_t period_bundle_regs = KAWPOW_REGS;
  uint32_t period_bundle_cnt_cache = KAWPOW_CNT_CACHE;
  uint32_t period_bundle_cnt_math = KAWPOW_CNT_MATH;
  std::unique_ptr<MomKernelBundle> next_bundle;
  uint64_t next_bundle_period = UINT64_MAX;
  uint32_t next_bundle_epoch = UINT32_MAX;
  SealMode next_bundle_seal = SealMode::KAWPOW;
  uint32_t next_bundle_magic[15] = {};
  uint32_t next_bundle_regs = KAWPOW_REGS;
  uint32_t next_bundle_cnt_cache = KAWPOW_CNT_CACHE;
  uint32_t next_bundle_cnt_math = KAWPOW_CNT_MATH;
  std::thread prefetch_thread;

  // CUDA kawpow path (decided once, then stable for this device's lifetime): the source-JIT folds
  // the per-period ProgPoW program for full speed, but needs a working runtime kernel_compiler
  // (libsycl-jit + a CUDA target). If the first build's JIT throws (e.g. an old driver or a host
  // missing libdevice), fall back permanently to the AOT spec-constant kernel -- correct, just
  // unfolded (~3x slower). -1 = undecided, 1 = JIT, 0 = spec-const. MOM_KAWPOW_JIT=0/1 forces a path.
  // Atomic because the prefetch thread reads/writes it while the launch path reads it.
  std::atomic<int> cuda_use_jit{-1};

  explicit KawpowState(const std::string& dev_str)
    : device(get_dev(dev_str)),
      queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !mom_has_usm_device(device)),
      shared_dag(device.is_cpu() || !mom_has_usm_device(device)),
      workgroup(kawpow_workgroup(device)),
      default_subgroup_exchange(kawpow_default_subgroup_exchange(device))
  {
    if (!mom_has_usm_shared(device)) {
      throw std::string("kawpow SYCL device does not support required allocations");
    }

    set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", kawpow_compile_options());
    bundle = std::make_unique<MomKernelBundle>(
      mom_get_exec_bundle(queue.get_context())
    );
    if (device.is_gpu() && sycl_is_level_zero_gpu(device) && !default_subgroup_exchange) {
      std::fprintf(stderr, "kawpow: Intel driver build %u uses stable local-memory exchange\n",
                   mom_driver_build(device));
    }
  }

  ~KawpowState() {
    if (prefetch_thread.joinable()) prefetch_thread.join();
    sycl_cleanup_noexcept("kawpow", [&] { release(); });
  }

  // Parse env var `name` as base-10 unsigned; returns false (leaving `out` untouched) on unset,
  // empty, malformed, or out-of-uint32-range input so callers fall back to their default.
  static bool parse_env_u32(const char* name, uint32_t& out) {
    const char* const value = std::getenv(name);
    if (!value || !*value) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno || end == value || *end || parsed > std::numeric_limits<uint32_t>::max()) return false;
    out = static_cast<uint32_t>(parsed);
    return true;
  }

  // Pick a work-group size: honor `name` if set to one of `allowed`, else `sycl_default_workgroup`.
  static unsigned select_workgroup(
    const char* name, const sycl::device& dev,
    const std::initializer_list<unsigned> allowed, const unsigned preferred
  ) {
    const unsigned fallback = sycl_default_workgroup(dev, allowed, preferred);
    uint32_t parsed;
    if (!parse_env_u32(name, parsed)) return fallback;
    for (const unsigned candidate : allowed) if (candidate == parsed) return parsed;
    return fallback;
  }

  static unsigned kawpow_workgroup(const sycl::device& dev) {
    // Windows CUDA, Intel, and HIP use 256 as before. On Linux CUDA, compact GPUs benefit from
    // local 256 (36-CU RTX 5060 Ti: +5--6%), while wider GPUs have enough resident blocks at local
    // 128 (60-CU L4 winner). Use the portable compute-unit count rather than a product-name list so
    // future devices choose a reasonable side of the same occupancy tradeoff.
#if defined(_WIN32)
    const unsigned gpu_default = 256;
#else
    const bool wide_cuda = mom_is_cuda(dev) &&
      dev.get_info<sycl::info::device::max_compute_units>() >= 48;
    const unsigned gpu_default = wide_cuda ? 128 : 256;
#endif
    return select_workgroup("MOM_KAWPOW_WORKGROUP", dev, {64, 128, 256, 512}, dev.is_cpu() ? 128 : gpu_default);
  }

  static unsigned kawpow_dag_workgroup(const sycl::device& dev) {
    return select_workgroup("MOM_KAWPOW_DAG_WORKGROUP", dev, {32, 64, 128, 256, 512}, dev.is_cpu() ? 128 : 64);
  }

  static uint32_t kawpow_dag_chunk_nodes() {
    uint32_t parsed = 0;  // 0 means "single full-DAG dispatch" (chunking disabled)
    parse_env_u32("MOM_KAWPOW_DAG_CHUNK_NODES", parsed);
    return parsed;
  }

  static const char* kawpow_compile_options() {
    // "-O3" is process-global; pearlhash's ESIMD image rejects it (and it's a no-op for kawpow), so
    // default to none. Override via MOM_KAWPOW_COMPILE_OPTIONS if needed.
    const char* const value = std::getenv("MOM_KAWPOW_COMPILE_OPTIONS");
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
    // Source-built bundles must die while DPC++'s ProgramManager is still alive. In particular, the
    // CUDA kernel_compiler registers dynamic device images whose bundle destructor calls
    // ProgramManager::removeImages(). Keep this explicit (rather than relying on member destruction)
    // so shutdown never leaves an outstanding prefetch/period image behind the queue allocations.
    next_bundle.reset();
    period_bundle.reset();
    free_ptr(input);
    free_ptr(light_cache);
    free_ptr(dag);
    free_ptr(result);
    input_cap = 0;
    light_cache_words = 0;
    dag_words = 0;
    epoch = UINT32_MAX;
    dag_bytes = 0;
    period = UINT64_MAX;
  }

  void ensure_input(const unsigned input_size) {
    if (input_size <= input_cap && result) return;
    queue.wait_and_throw();
    free_ptr(input);
    free_ptr(result);
    input_cap = input_size;
    input = allocate<uint8_t>(input_cap, shared_io);
    result = sycl::malloc_shared<KawpowResult>(1, queue);
    if (!input || !result) throw std::string("Can't allocate kawpow SYCL input buffers");
  }

  // new_epoch is the REAL epoch (the seed/period-bundle key); new_cache_bytes/new_dag_bytes are the
  // (possibly MeowPow-scaled) sizes; seed_epoch is the epoch the keccak seedhash derives from (== epoch
  // except MeowPow >= dag-change). The cache key includes dag_bytes so a size change forces a rebuild.
  void ensure_epoch(const uint32_t new_epoch, const uint64_t new_cache_bytes, const uint64_t new_dag_bytes,
                    const uint32_t seed_epoch, const bool should_log) {
    if (epoch == new_epoch && dag_bytes == new_dag_bytes) return;
    if (new_epoch >= 2048 || !new_cache_bytes || !new_dag_bytes ||
        new_cache_bytes % (NODE_WORDS * sizeof(uint32_t)) != 0 ||
        new_dag_bytes % (NODE_WORDS * sizeof(uint32_t)) != 0) {
      throw std::string("Bad kawpow epoch");
    }

    const uint64_t new_light_cache_words = new_cache_bytes / sizeof(uint32_t);
    const uint64_t new_dag_words = new_dag_bytes / sizeof(uint32_t);

    std::vector<uint32_t> host_cache;
    const uint64_t start_ms = now_ms();
    compute_light_cache(host_cache, new_cache_bytes, seed_epoch);

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
    if (!light_cache || !dag) throw std::string("Can't allocate kawpow DAG buffers");

    if (shared_dag) std::memcpy(light_cache, host_cache.data(), host_cache.size() * sizeof(uint32_t));
    else sycl_wait_and_throw(queue.memcpy(light_cache, host_cache.data(), host_cache.size() * sizeof(uint32_t)), device);

    const uint32_t light_nodes = static_cast<uint32_t>(new_light_cache_words / NODE_WORDS);
    const FastModData light_mod = make_fast_mod_data(light_nodes);

    const uint32_t dag_workgroup = kawpow_dag_workgroup(device);
    const uint32_t total = static_cast<uint32_t>(new_dag_words / NODE_WORDS);  // DAG node count
    const uint32_t chunk_nodes = kawpow_dag_chunk_nodes();
    uint32_t* const d_light = light_cache;
    uint32_t* const d_dag = dag;
    sycl::queue& q = queue;
    auto& kb = *bundle;
    (void)kb;

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

            uint32_t dag_node[NODE_WORDS];
            const uint32_t init = fast_mod_dev(node_index, light_mod);
#pragma unroll
            for (unsigned w = 0; w < NODE_WORDS; ++w) dag_node[w] = d_light[init * NODE_WORDS + w];
            dag_node[0] ^= node_index;
            keccak_512_words_dev(dag_node);

            for (uint32_t i = 0; i < KAWPOW_DATASET_PARENTS; ++i) {
              const uint32_t parent_index = fast_mod_dev(fnv_dev(node_index ^ i, dag_node[i & (NODE_WORDS - 1)]), light_mod);
              const uint32_t parent_base = parent_index * NODE_WORDS;
#pragma unroll
              for (unsigned w = 0; w < NODE_WORDS; ++w) {
                dag_node[w] = fnv_dev(dag_node[w], d_light[parent_base + w]);
              }
            }
            keccak_512_words_dev(dag_node);

            const uint32_t base = node_index * NODE_WORDS;
#pragma unroll
            for (unsigned w = 0; w < NODE_WORDS; ++w) d_dag[base + w] = dag_node[w];
          }
        );
      });
      if (!chunk_nodes) break;
      start_node += current_nodes;
    }
    sycl_wait_and_throw(dag_event, device);

    epoch = new_epoch;
    dag_bytes = new_dag_bytes;
    if (should_log) {
      char elapsed[32];
      format_duration_ms(elapsed, sizeof(elapsed), now_ms() - start_ms);
      std::fprintf(stderr, "KawPow DAG for epoch %u calculated (%s)\n", new_epoch, elapsed);
    }
  }

  void ensure_period(const uint64_t new_period) {
    // The program shape (regs/cnt_*) can change with the algo even at an unchanged period, so rebuild
    // when either changes. period_shape tracks the shape the cached program was generated with.
    if (period == new_period && program_shape_regs == regs && program_shape_cnt_cache == cnt_cache &&
        program_shape_cnt_math == cnt_math) return;
    program = make_program(new_period, regs, cnt_cache, cnt_math);
    period = new_period;
    program_shape_regs = regs;
    program_shape_cnt_cache = cnt_cache;
    program_shape_cnt_math = cnt_math;
  }
  uint32_t program_shape_regs = KAWPOW_REGS;
  uint32_t program_shape_cnt_cache = KAWPOW_CNT_CACHE;
  uint32_t program_shape_cnt_math = KAWPOW_CNT_MATH;

  // build_seal / build_magic / build_regs+cnt are passed by value (not read from the members) so the
  // prefetch thread, which calls this off the state mutex, never races kawpow_core writing state.*. Only
  // the source-JIT path bakes the seal/shape; the Intel spec-const path gets seal+shape via SealParams at
  // launch (and the per-shape make_program already bakes the right program into the spec constant).
  std::unique_ptr<MomKernelBundle>
  build_period_bundle(const uint64_t new_period, const FastModData dag_mod,
                      const SealMode build_seal, const uint32_t build_magic[15],
                      const uint32_t build_regs, const uint32_t build_cnt_cache,
                      const uint32_t build_cnt_math) {
#if defined(MOM_SYCL_ADAPTIVECPP) || defined(MOM_SYCL_PORTABLE_OPENCL)
    (void)new_period; (void)dag_mod; (void)build_seal; (void)build_magic;
    (void)build_regs; (void)build_cnt_cache; (void)build_cnt_math;
    return std::make_unique<MomKernelBundle>();
#else
#if defined(MOM_SYCL_HAS_CUDA)
    // CUDA: the source-JIT folds the per-period program to straight-line const ops (full speed); the
    // AOT spec-constant kernel below is correct but ~3x slower because spec constants do NOT fold on
    // the nvptx backend. Default to JIT; fall back to spec-const if the runtime kernel_compiler is
    // unavailable (see cuda_use_jit). MOM_KAWPOW_JIT=0/1 forces a path (1 = JIT, no fallback).
    if (mom_is_cuda(device)) {
      if (cuda_use_jit.load() < 0) {
        const char* const force = std::getenv("MOM_KAWPOW_JIT");
        if (force) cuda_use_jit.store(std::atoi(force) != 0 ? 1 : 0);
      }
      if (cuda_use_jit.load() != 0) {  // 1 (committed JIT) or -1 (undecided -> try JIT)
        try {
          // Compile the search kernel from SYCL source with the period program baked in as
          // a const struct (folds the random-math interpreter). Device body is sycl/kawpow/device.inc.
          // The per-algo keccak seal (mode + magic) is also baked in (kawpow_emit_seal) so firopow /
          // evrprogpow compute the correct seal on CUDA; emitted after kawpow_device_src() (defines
          // SealMode) and before the WRAPPER (reads JIT_SEAL/JIT_MAGIC).
          namespace syclex = sycl::ext::oneapi::experimental;
          const std::string kernel_name = kawpow_jit_kernel_name(
            new_period, dag_mod, build_seal, build_magic,
            build_regs, build_cnt_cache, build_cnt_math);
          const std::string src = std::string(KAWPOW_JIT_PRELUDE) + kawpow_device_src() +
            kawpow_emit_baked(make_program(new_period, build_regs, build_cnt_cache, build_cnt_math), dag_mod) +
            kawpow_emit_seal(build_seal, build_magic) +
            kawpow_emit_shape(build_regs, build_cnt_cache, build_cnt_math) +
            kawpow_named_jit_wrapper(kernel_name);
          auto kb_src = syclex::create_kernel_bundle_from_source(
            queue.get_context(), syclex::source_language::sycl, src);
          const char* const jit_opts = std::getenv("MOM_KAWPOW_JIT_OPTS");
          // Build for THIS device explicitly: without the device list the kernel_compiler targets a
          // generic nvptx default (sm_75 -> CUDA_ERROR_NO_BINARY on newer GPUs / needs sm_75 libdevice).
          // Passing the device makes it compile for the device's actual arch (e.g. sm_89 on an L4).
          const std::vector<sycl::device> jit_devs{device};
          auto exe = (jit_opts && *jit_opts)
            ? syclex::build(kb_src, jit_devs, syclex::properties{syclex::build_options{std::string(jit_opts)}})
            : syclex::build(kb_src, jit_devs);
          cuda_use_jit.store(1);
          return std::make_unique<sycl::kernel_bundle<sycl::bundle_state::executable>>(std::move(exe));
        } catch (const std::exception& e) {
          // Committed to JIT (user forced it, or an earlier period already JIT'd): a failure now is a
          // real error. Undecided (first build probing): fall back permanently to spec-const.
          if (cuda_use_jit.load() == 1) throw;
          fprintf(stderr, "kawpow: CUDA source-JIT unavailable (%s); using the slower AOT "
                  "spec-constant kernel\n", e.what());
          cuda_use_jit.store(0);
        }
      }
      // cuda_use_jit == 0: fall through to the spec-constant build below.
    }
#endif
    // Intel/spir64: SYCL-2020 specialization constants fold the per-period program at JIT time.
    auto input = sycl::get_kernel_bundle<sycl::bundle_state::input>(
      queue.get_context(), {device}, {sycl::get_kernel_id<KawpowSgKernel>()}
    );
    input.set_specialization_constant<kawpow_program_id>(
      make_program(new_period, build_regs, build_cnt_cache, build_cnt_math));
    input.set_specialization_constant<kawpow_dag_mod_id>(dag_mod);
    return std::make_unique<sycl::kernel_bundle<sycl::bundle_state::executable>>(
      sycl::build(input)
    );
#endif
  }

  // The seal AND the inner-loop shape participate in the cache key: a device's KawpowState is shared
  // across the kawpow/firopow/evrprogpow/meowpow entrypoints, so an algo switch can change the seal or
  // shape even when period/epoch are unchanged -- it must force a rebuild. `b*` are the current call's.
  static bool bundle_key_eq(const SealMode a, const uint32_t am[15], const uint32_t ar,
                            const uint32_t ac, const uint32_t amth,
                            const SealMode b, const uint32_t bm[15], const uint32_t br,
                            const uint32_t bc, const uint32_t bmth) {
    if (a != b || ar != br || ac != bc || amth != bmth) return false;
    for (unsigned i = 0; i < 15; ++i) if (am[i] != bm[i]) return false;
    return true;
  }

  void ensure_period_bundle(const uint64_t new_period, const uint32_t new_epoch, const FastModData dag_mod) {
    if (period_bundle && period_bundle_period == new_period && period_bundle_epoch == new_epoch &&
        bundle_key_eq(period_bundle_seal, period_bundle_magic, period_bundle_regs,
                      period_bundle_cnt_cache, period_bundle_cnt_math,
                      seal, magic, regs, cnt_cache, cnt_math)) return;

    if (prefetch_thread.joinable()) prefetch_thread.join();
    if (next_bundle && next_bundle_period == new_period && next_bundle_epoch == new_epoch &&
        bundle_key_eq(next_bundle_seal, next_bundle_magic, next_bundle_regs,
                      next_bundle_cnt_cache, next_bundle_cnt_math,
                      seal, magic, regs, cnt_cache, cnt_math)) {
      period_bundle = std::move(next_bundle);
    } else {
      period_bundle = build_period_bundle(new_period, dag_mod, seal, magic, regs, cnt_cache, cnt_math);
    }
    period_bundle_period = new_period;
    period_bundle_epoch = new_epoch;
    period_bundle_seal = seal;
    for (unsigned i = 0; i < 15; ++i) period_bundle_magic[i] = magic[i];
    period_bundle_regs = regs;
    period_bundle_cnt_cache = cnt_cache;
    period_bundle_cnt_math = cnt_math;

    next_bundle.reset();
    next_bundle_period = new_period + 1;
    next_bundle_epoch = new_epoch;
    next_bundle_seal = seal;
    for (unsigned i = 0; i < 15; ++i) next_bundle_magic[i] = magic[i];
    next_bundle_regs = regs;
    next_bundle_cnt_cache = cnt_cache;
    next_bundle_cnt_math = cnt_math;
    // Capture seal/magic/shape by value so the prefetch build is race-free w.r.t. a concurrent algo
    // switch writing state.* on the next call (FiroPow rebuilds every block, so this runs).
    SealMode pf_seal = seal;
    std::array<uint32_t, 15> pf_magic;
    for (unsigned i = 0; i < 15; ++i) pf_magic[i] = magic[i];
    const uint32_t pf_regs = regs, pf_cnt_cache = cnt_cache, pf_cnt_math = cnt_math;
    prefetch_thread = std::thread([this, next_period = new_period + 1, dag_mod, pf_seal, pf_magic,
                                   pf_regs, pf_cnt_cache, pf_cnt_math] {
      try {
        next_bundle = build_period_bundle(next_period, dag_mod, pf_seal, pf_magic.data(),
                                          pf_regs, pf_cnt_cache, pf_cnt_math);
      } catch (...) {
        next_bundle.reset();
      }
    });
  }

  static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()
    ).count();
  }
};

using KawpowStateMap = std::map<std::string, std::unique_ptr<KawpowState>>;

// Heap-own the registry and lock so they have no competing C++ static destructors. Node's explicit
