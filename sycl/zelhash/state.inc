// Bound the aligned bucket capacity by total memory and single-allocation limits. Return 0 below
// MIN_SLOTS so discovery skips the device without throwing; MOM_ZELHASH_SLOTS applies only to the
// workload process (zelhash_configured_slots).
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

  const unsigned aligned = round_down(
      static_cast<unsigned>(std::min<uint64_t>(capacity, std::numeric_limits<unsigned>::max())),
      SLOT_ALIGNMENT);
  return aligned < MIN_SLOTS ? 0u : aligned;
}

// Resolve the workload's slot stride, rejecting an unusable device or MOM_ZELHASH_SLOTS value.
static unsigned zelhash_configured_slots(const sycl::device& device) {
  const unsigned capacity = zelhash_slot_capacity(device);
  if (!capacity)
    throw std::string("zelhash device has insufficient allocation capacity");

  // Keep the public tuning override bounded by this device-derived capacity. This lets additional
  // architectures sweep a safe stride with one binary while automatic mode remains portable.
  if (const char* value = std::getenv("MOM_ZELHASH_SLOTS"); value && *value) {
    unsigned long requested = 0;
    if (!mom_parse_env_ulong("MOM_ZELHASH_SLOTS", requested) ||
        requested < MIN_SLOTS || requested > capacity ||
        requested % SLOT_ALIGNMENT != 0u)
      throw std::string("MOM_ZELHASH_SLOTS must be a supported 16-slot-aligned capacity");
    return static_cast<unsigned>(requested);
  }
  return capacity;
}

// Per-device state: queue + USM arenas. The Wagner arenas are allocated lazily on the first solve;
// the generation test path only needs the small base-state and rows buffers.
class State {
public:
  sycl::device device;
  sycl::queue queue;
  bool shared_io, gpu_compact;
  unsigned slot_capacity;
  // Generation-test I/O
  UsmPtr<uint64_t> base_h;      // 8 words
  UsmPtr<uint8_t> base_pending; // 12 bytes
  UsmPtr<uint8_t> rows;         // EQUIHASH_TEST_ROWS * HASH_LENGTH
  // Four compact bucket levels plus counters and tiny recovery lists. The final round emits only
  // zero-field pairs; their parent trees are reconstructed after that filter.
  UsmPtr<uint32_t> level[NSTORED_LEVELS]; // level[L]: materialized bucketed collision records
  UsmPtr<uint32_t> l0_index;              // CUDA compact level-0 leaf mapping (cold recovery stream)
  UsmPtr<uint32_t> nslots;                // NSTORED_LEVELS * NBUCKETS atomic counters
  UsmPtr<EqCandidateRef> ref;             // zero-node references awaiting compact recovery
  UsmPtr<uint32_t> ref_count;             // shared: zero-node reference counter
  UsmPtr<EqCandidate> cand;               // final candidate list (recovered leaf indices)
  UsmPtr<uint32_t> cand_count;            // shared: atomic candidate counter
#if defined(MOM_EQ_FORWARD_MAP)
  UsmPtr<uint32_t> invert_target; // cold leaf-position inversion scratch
  UsmPtr<uint32_t> invert_next;
  UsmPtr<uint32_t> invert_head;
#endif
  static constexpr uint32_t CAND_CAP = 4096;
  bool arenas_built = false;
  std::mutex mutex;

  // Select backend record layout and capacity before creating persistent ordered solver state.
  explicit State(const std::string& dev_str)
      : device(get_dev(dev_str)),
        queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
        shared_io(device.is_cpu() || !mom_has_usm_device(device)),
        gpu_compact(mom_is_cuda(device) || mom_is_hip(device)),
        slot_capacity(zelhash_configured_slots(device)) {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device)))
      throw std::string("zelhash SYCL device does not support required allocations");
  }
  // Drain submitted work before the USM members free their allocations.
  ~State() {
    sycl_cleanup_noexcept("zelhash wait", [&] { queue.wait_and_throw(); });
  }
  // Allocate I/O-placed USM (shared on CPUs and devices without device USM).
  template <typename T> UsmPtr<T> alloc(const size_t n, const char* const failure) {
    return mom_usm_alloc<T>(queue, n, shared_io, failure);
  }
  // Clear layout-specific arenas and recovery scratch so failed construction can be retried cleanly.
  void free_all_arenas() {
    for (auto& arena : level)
      arena.reset();
    l0_index.reset();
    nslots.reset();
    ref.reset();
    ref_count.reset();
    cand.reset();
    cand_count.reset();
#if defined(MOM_EQ_FORWARD_MAP)
    invert_target.reset();
    invert_next.reset();
    invert_head.reset();
#endif
    arenas_built = false;
  }
  // Lazily allocate the tiny header/test buffers; a failure leaves none of them allocated.
  void ensure_io() {
    if (base_h && base_pending && rows)
      return;
    // Both the host (base-state write, test-row readback) and the device touch these, so keep them
    // shared even on discrete GPUs.
    constexpr const char* failure = "Can't allocate zelhash I/O buffers";
    auto new_base_h = mom_usm_alloc<uint64_t>(queue, 8, true, failure);
    auto new_base_pending = mom_usm_alloc<uint8_t>(queue, 12, true, failure);
    auto new_rows =
        mom_usm_alloc<uint8_t>(queue, (size_t)EQUIHASH_TEST_ROWS * HASH_LENGTH, true, failure);
    base_h = std::move(new_base_h);
    base_pending = std::move(new_base_pending);
    rows = std::move(new_rows);
  }
  // Allocate the capacity-selected Wagner levels, per-level counters and bounded recovery output on
  // the first solve; any failure releases every arena so a later solve retries from scratch.
  void ensure_arenas(bool log) {
    if (arenas_built)
      return;
    const uint64_t t0 = mom_now_ms();
    free_all_arenas();
    try {
      const size_t slots_per_level = (size_t)NBUCKETS * slot_capacity;
      size_t total = 0;
      for (unsigned L = 0; L < NSTORED_LEVELS; ++L) {
        const size_t lvl_u32 = slots_per_level * level_u32(L, gpu_compact);
        level[L] = alloc<uint32_t>(lvl_u32, "Can't allocate zelhash level arena");
        total += lvl_u32 * 4;
      }
      if (level_u32(0, gpu_compact) == 4u) {
        l0_index = alloc<uint32_t>(slots_per_level, "Can't allocate zelhash level-0 index arena");
        total += slots_per_level * sizeof(uint32_t);
      }
      constexpr const char* counters = "Can't allocate zelhash counters/candidates";
      nslots = alloc<uint32_t>((size_t)NSTORED_LEVELS * NBUCKETS, counters);
      total += (size_t)NSTORED_LEVELS * NBUCKETS * 4;
      ref = alloc<EqCandidateRef>(CAND_CAP, counters);
      ref_count = mom_usm_alloc<uint32_t>(queue, 1, true, counters);
      cand = alloc<EqCandidate>(CAND_CAP, counters);
      cand_count = mom_usm_alloc<uint32_t>(queue, 1, true, counters); // host reads the count
#if defined(MOM_EQ_FORWARD_MAP)
      if (gpu_compact) {
        constexpr size_t targets = (size_t)CAND_CAP * (1u << K);
        constexpr const char* scratch = "Can't allocate zelhash recovery scratch";
        invert_target = alloc<uint32_t>(targets, scratch);
        invert_next = alloc<uint32_t>(targets, scratch);
        invert_head = alloc<uint32_t>(L0_INVERT_HEADS, scratch);
        total += (targets * 2u + L0_INVERT_HEADS) * sizeof(uint32_t);
      }
#endif
      // Touch the counter array (lazily mapped USM device allocs surface OOM here, not mid-kernel).
      sycl_wait_and_throw(
          queue.memset(nslots, 0, (size_t)NSTORED_LEVELS * NBUCKETS * sizeof(uint32_t)), device);
      arenas_built = true;
      if (log) {
        std::fprintf(stderr,
                     "zelhash Wagner arenas (~%.1f GiB, %u slots/bucket) allocated (%llu ms)\n",
                     (double)total / (1024.0 * 1024.0 * 1024.0), slot_capacity,
                     (unsigned long long)(mom_now_ms() - t0));
        std::fflush(stderr);
      }
    } catch (...) {
      free_all_arenas();
      throw;
    }
  }
};

// Keep runtime-owned state under explicit shutdown control instead of static destructor order.
static DeviceStateRegistry<State>& registry() {
  static auto* const registry = new DeviceStateRegistry<State>;
  return *registry;
}

static State& state_for(const std::string& dev_str) {
  return registry().get(dev_str, [&] { return std::make_unique<State>(dev_str); });
}

// Drain persistent device states during ordered shutdown without propagating cleanup errors.
void zelhash_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "zelhash: ordered SYCL cleanup failed\n");
  }
}
