unsigned zelhash_slot_capacity(const sycl::device& device) {
  const bool gpu_compact = mom_is_cuda(device) || mom_is_hip(device);
  // Preserve the statistically selected fast capacity when it fits. On smaller or conservatively
  // reported devices, derive a reduced aligned stride from both total memory and the largest legal
  // allocation instead of recognizing individual products or failing at the first arena allocation.
  uint64_t capacity = MAX_SLOTS;
  const uint64_t global_mem = device.get_info<sycl::info::device::global_mem_size>();
  const uint64_t max_alloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
  const uint64_t reserve = std::min<uint64_t>(1ull << 30, global_mem / 8u);
  const uint64_t arena_budget = global_mem > reserve ? global_mem - reserve : global_mem;

  uint64_t words_per_slot = 0;
  for (unsigned level = 0; level < NSTORED_LEVELS; ++level) {
    const uint64_t level_words = level_u32(level, gpu_compact);
    words_per_slot += level_words;
    if (max_alloc)
      capacity = std::min(capacity, max_alloc / (NBUCKETS * level_words * sizeof(uint32_t)));
  }
  if (level_u32(0, gpu_compact) == 4u) {
    ++words_per_slot; // separate level-0 recovery index
    if (max_alloc)
      capacity = std::min(capacity, max_alloc / (NBUCKETS * sizeof(uint32_t)));
  }
  if (arena_budget)
    capacity = std::min(capacity, arena_budget / (NBUCKETS * words_per_slot * sizeof(uint32_t)));

  constexpr unsigned SLOT_ALIGNMENT = 16u;
  const unsigned aligned = round_down(static_cast<unsigned>(
    std::min<uint64_t>(capacity, std::numeric_limits<unsigned>::max())), SLOT_ALIGNMENT);
  const unsigned mean = ceil_div_u64(NUM_ENTRIES, NBUCKETS);
  const unsigned minimum = round_up(mean + ceil_sqrt(mean), SLOT_ALIGNMENT);
  if (aligned < minimum)
    throw std::string("zelhash device has insufficient allocation capacity");

  // Keep the public tuning override bounded by this device-derived capacity. This lets additional
  // architectures sweep a safe stride with one binary while automatic mode remains portable.
  if (const char* value = std::getenv("MOM_ZELHASH_SLOTS"); value && *value) {
    char* end = nullptr;
    const unsigned long requested = std::strtoul(value, &end, 10);
    if (!end || *end || requested < minimum || requested > aligned ||
        requested % SLOT_ALIGNMENT != 0u)
      throw std::string("MOM_ZELHASH_SLOTS must be a supported 16-slot-aligned capacity");
    return static_cast<unsigned>(requested);
  }
  return aligned;
}

// Per-device state: queue + bundle + USM arenas (modeled on FishState). The Wagner arenas
// are allocated lazily on the first mining solve to prove headroom; the M1 test path only needs the
// small base-state + rows buffers.
class ZelHashState {
public:
  sycl::device device; sycl::queue queue; std::unique_ptr<MomKernelBundle> bundle;
  bool shared_io, gpu_compact;
  unsigned slot_capacity;
  // M1 I/O
  uint64_t* base_h = nullptr;     // 8 words
  uint8_t*  base_pending = nullptr;// 12 bytes
  uint8_t*  rows = nullptr;       // EQUIHASH_TEST_ROWS * HASH_LENGTH
  // Four compact bucket levels plus counters and tiny recovery lists. The final round emits only
  // zero-field pairs; their parent trees are reconstructed after that filter.
  uint32_t* level[NSTORED_LEVELS] = {}; // level[L]: materialized bucketed collision records
  uint32_t* l0_index       = nullptr;  // CUDA compact level-0 leaf mapping (cold recovery stream)
  uint32_t* nslots         = nullptr;  // NSTORED_LEVELS * NBUCKETS atomic counters
  EqCandidateRef* ref      = nullptr;  // zero-node references awaiting compact recovery
  uint32_t* ref_count      = nullptr;  // shared: zero-node reference counter
  EqCandidate* cand        = nullptr;  // final candidate list (recovered leaf indices)
  uint32_t* cand_count     = nullptr;  // shared: atomic candidate counter
#if defined(MOM_EQ_FORWARD_MAP)
  uint32_t* invert_target  = nullptr;  // cold leaf-position inversion scratch
  uint32_t* invert_next    = nullptr;
  uint32_t* invert_head    = nullptr;
#endif
  static constexpr uint32_t CAND_CAP = 4096;
  bool arenas_built = false; std::mutex mutex;

  explicit ZelHashState(const std::string& dev_str)
    : device(get_dev(dev_str)), queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !mom_has_usm_device(device)),
      gpu_compact(mom_is_cuda(device) || mom_is_hip(device)),
      slot_capacity(zelhash_slot_capacity(device)) {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device)))
      throw std::string("zelhash SYCL device does not support required allocations");
    bundle = std::make_unique<MomKernelBundle>(mom_get_exec_bundle(queue.get_context()));
  }
  ~ZelHashState() { sycl_cleanup_noexcept("zelhash", [&] { queue.wait_and_throw(); free_all(); }); }
  template <typename T> T* alloc(size_t n) {
    return shared_io ? sycl::malloc_shared<T>(n, queue) : sycl::malloc_device<T>(n, queue);
  }
  void free_ptr(auto*& p) { if (p) sycl::free(p, queue); p = nullptr; }
  void free_all() {
    free_ptr(base_h); free_ptr(base_pending); free_ptr(rows);
    for (unsigned L = 0; L < NSTORED_LEVELS; ++L) free_ptr(level[L]);
    free_ptr(l0_index); free_ptr(nslots); free_ptr(ref); free_ptr(ref_count); free_ptr(cand); free_ptr(cand_count);
#if defined(MOM_EQ_FORWARD_MAP)
    free_ptr(invert_target); free_ptr(invert_next); free_ptr(invert_head);
#endif
    arenas_built = false;
  }
  void ensure_io() {
    if (base_h && base_pending && rows) return;
    free_ptr(base_h); free_ptr(base_pending); free_ptr(rows);
    // Shared (host-accessible) -- these are tiny and both the host (base-state write, M1 row readback)
    // and the device touch them, so keep them shared even on discrete GPUs (cf. FishState::result).
    base_h       = sycl::malloc_shared<uint64_t>(8, queue);
    base_pending = sycl::malloc_shared<uint8_t>(12, queue);
    rows         = sycl::malloc_shared<uint8_t>((size_t)EQUIHASH_TEST_ROWS * HASH_LENGTH, queue);
    if (!base_h || !base_pending || !rows) throw std::string("Can't allocate zelhash I/O buffers");
  }
  // Allocate the four materialized Wagner levels, per-level counters, and bounded recovery output.
  void ensure_arenas(bool log) {
    if (arenas_built) return;
    const uint64_t t0 = now_ms();
    free_all_arenas();
    const size_t slots_per_level = (size_t)NBUCKETS * slot_capacity;
    size_t total = 0;
    for (unsigned L = 0; L < NSTORED_LEVELS; ++L) {
      const size_t lvl_u32 = slots_per_level * level_u32(L, gpu_compact);
      level[L] = alloc<uint32_t>(lvl_u32); total += lvl_u32 * 4;
      if (!level[L]) throw std::string("Can't allocate zelhash level arena");
    }
    if (level_u32(0, gpu_compact) == 4u) {
      l0_index = alloc<uint32_t>(slots_per_level);
      total += slots_per_level * sizeof(uint32_t);
      if (!l0_index) throw std::string("Can't allocate zelhash level-0 index arena");
    }
    nslots     = alloc<uint32_t>((size_t)NSTORED_LEVELS * NBUCKETS);
    total += (size_t)NSTORED_LEVELS * NBUCKETS * 4;
    ref        = alloc<EqCandidateRef>(CAND_CAP);
    ref_count  = sycl::malloc_shared<uint32_t>(1, queue);
    cand       = alloc<EqCandidate>(CAND_CAP);
    cand_count = sycl::malloc_shared<uint32_t>(1, queue);     // host reads the candidate count
#if defined(MOM_EQ_FORWARD_MAP)
    if (gpu_compact) {
      constexpr size_t targets = (size_t)CAND_CAP * (1u << K);
      invert_target = alloc<uint32_t>(targets);
      invert_next = alloc<uint32_t>(targets);
      invert_head = alloc<uint32_t>(L0_INVERT_HEADS);
      total += (targets * 2u + L0_INVERT_HEADS) * sizeof(uint32_t);
    }
#endif
    if (!nslots || !ref || !ref_count || !cand || !cand_count)
      throw std::string("Can't allocate zelhash counters/candidates");
#if defined(MOM_EQ_FORWARD_MAP)
    if (gpu_compact && (!invert_target || !invert_next || !invert_head))
      throw std::string("Can't allocate zelhash recovery scratch");
#endif
    // Touch the counter array (lazily mapped USM device allocs surface OOM here, not mid-kernel).
    sycl_wait_and_throw(queue.memset(nslots, 0,
      (size_t)NSTORED_LEVELS * NBUCKETS * sizeof(uint32_t)), device);
    arenas_built = true;
    if (log) { std::fprintf(stderr,
      "zelhash Wagner arenas (~%.1f GiB, %u slots/bucket) allocated (%llu ms)\n",
      (double)total / (1024.0 * 1024.0 * 1024.0), slot_capacity,
      (unsigned long long)(now_ms() - t0)); std::fflush(stderr); }
  }
  void free_all_arenas() {
    for (unsigned L = 0; L < NSTORED_LEVELS; ++L) free_ptr(level[L]);
    free_ptr(l0_index); free_ptr(nslots); free_ptr(ref); free_ptr(ref_count); free_ptr(cand); free_ptr(cand_count);
#if defined(MOM_EQ_FORWARD_MAP)
    free_ptr(invert_target); free_ptr(invert_next); free_ptr(invert_head);
#endif
  }
  static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }
};

using ZelHashStateMap = std::map<std::string, std::unique_ptr<ZelHashState>>;

static ZelHashStateMap& zelhash_states() {
  static ZelHashStateMap* const states = new ZelHashStateMap;
  return *states;
}

static std::mutex& zelhash_states_mutex() {
  static std::mutex* const mutex = new std::mutex;
  return *mutex;
}

static ZelHashState& zelhash_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(zelhash_states_mutex());
  auto& state = zelhash_states()[dev_str];
  if (!state) state = std::make_unique<ZelHashState>(dev_str);
  return *state;
}

void zelhash_cleanup_states() noexcept {
  try {
    std::lock_guard<std::mutex> lock(zelhash_states_mutex());
    zelhash_states().clear();
  } catch (...) {
    std::fprintf(stderr, "zelhash: ordered SYCL cleanup failed\n");
  }
}
