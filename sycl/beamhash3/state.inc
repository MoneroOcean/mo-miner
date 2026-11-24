struct BeamLevels {
  const uint32_t* tree[NUM_ROUNDS + 1]; // tree[L] (L=1..5): 2 u32/slot = parent mixed[L-1] indices
};
// Expand retained parent trees in canonical leader-first order; bounded stack/output failures reject the root.
inline unsigned beam_walk_leaves(const BeamLevels& lv, uint32_t final_slot, uint32_t* leaves) {
  struct Frame {
    unsigned level;
    uint32_t slot;
  };
  Frame stack[NUM_ROUNDS + 2];
  int sp = 0;
  stack[sp++] = {NUM_ROUNDS, final_slot};
  unsigned n = 0;
  while (sp > 0) {
    Frame f = stack[--sp];
    if (f.level == 0) {
      if (n >= NUM_LEAVES)
        return 0;
      // Level-0 gen writes leaf `index` at slot `index`, so the stored leaf index == the slot itself.
      // (mixed0 is therefore not persisted; recovery returns the slot directly.)
      leaves[n++] = f.slot;
      continue;
    }
    const uint32_t* tl = lv.tree[f.level] + (size_t)f.slot * 2;
    if (sp + 2 > (int)(NUM_ROUNDS + 2))
      return 0;
    // push follower first so the leader (lower index) is processed first -> leaves come out in the
    // canonical lowest-index-first tree order (matches getIndicesFromMinimal's field order).
    stack[sp++] = {f.level - 1, tl[1]};
    stack[sp++] = {f.level - 1, tl[0]};
  }
  return n;
}
class BeamFinalKernel;
// Recover only zero-XOR roots with distinct leaves; candidate writes remain capacity bounded.
static sycl::event submit_final(sycl::queue& q, const uint32_t* d_mixed5, uint32_t in_count,
                                BeamLevels lv, BeamCandidate* d_cand,
                                uint32_t* d_cand_count, uint32_t cand_cap) {
  constexpr unsigned IN_ST = mixed_stride_u32(NUM_ROUNDS); // level-5 stride (1 word + 0 idx) = 2
  constexpr unsigned WG = 128;
  const size_t global = ((size_t)in_count + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<BeamFinalKernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const uint32_t s = (uint32_t)it.get_global_id(0);
          if (s >= in_count)
            return;
          const uint32_t* rec = d_mixed5 + (size_t)s * IN_ST;
          const uint32_t w0 = rec[0]; // low 32 bits of the 24-bit merged field
          if ((w0 & ((1u << COLLISION_BIT_SIZE) - 1u)) != 0)
            return; // must be zero (full XOR collapse)
          uint32_t leaves[NUM_LEAVES];
          const unsigned n = beam_walk_leaves(lv, s, leaves);
          if (n != NUM_LEAVES)
            return;
          if (!mom_equihash::pow::leaves_distinct(leaves, n))
            return;
          const uint32_t idx = dev_atomic_u32(d_cand_count[0]).fetch_add(1u);
          if (idx >= cand_cap)
            return;
          for (unsigned i = 0; i < NUM_LEAVES; ++i)
            d_cand[idx].leaves[i] = leaves[i];
        });
  });
}

// ===========================================================================================
// Per-device state: queue + USM arenas.
// ===========================================================================================
class State {
public:
  sycl::device device;
  sycl::queue queue;
  bool shared_io;
  // Generation-test I/O
  UsmPtr<uint64_t> prePow; // 4 u64 (the SipHash key / IndividualWork)
  UsmPtr<uint8_t> rows;     // BEAMHASH3_TEST_ROWS * ROW_BYTES (raw 448-bit leaf rows)
  // Compact-capable GPU backends use two 64-byte arenas through level 3. Level 4 reuses compact[0]
  // as packed BeamLevel4Row storage while compact[1] retains the recovery trees;
  // MOM_BEAMHASH3_COMPACT=0 selects the fused fallback on non-CUDA backends.
  UsmPtr<uint32_t> final_mixed;         // fused layout: MIXED_CAP * mixed_stride_u32(5)
  UsmPtr<uint32_t> linear_mixed;        // CUDA layout: MIXED_CAP * MAX_MIXED_STRIDE
  UsmPtr<uint32_t> bucket[2];           // fused parity arenas, or CUDA's reusable bucket[0]
  UsmPtr<uint32_t> bucket_ns[2];        // counters paired with the allocated bucket arenas
  UsmPtr<BeamCompactRow> compact[2];    // packed two-arena Beam pipeline
  UsmPtr<uint32_t> compact_ns[2];
  UsmPtr<uint32_t> mixed_count;         // shared: per-round output counter (NUM_ROUNDS+1 entries)
  UsmPtr<uint32_t> tree[NUM_ROUNDS + 1]; // tree[L] (L=1..5): NUM_ENTRIES * 2
  UsmPtr<BeamCandidate> cand;
  UsmPtr<uint32_t> cand_count;
  static constexpr uint32_t CAND_CAP = 8192;
  bool arenas_built = false;
  std::mutex mutex;

  // Create the per-device ordered pipeline and require host-visible I/O plus GPU device allocations.
  explicit State(const std::string& dev_str)
      : device(get_dev(dev_str)),
        queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
        shared_io(device.is_cpu() || !mom_has_usm_device(device)) {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device)))
      throw std::string("beamhash3 SYCL device does not support required allocations");
  }
  ~State() {
    // Miner workers exit immediately after a benchmark sample. Keep process teardown noexcept like
    // every other persistent SYCL state: live kernel errors have already propagated through the
    // explicit waits, while a late CUDA-runtime-unloading result must not abort a completed worker.
    sycl_cleanup_noexcept("beamhash3 wait", [&] { queue.wait_and_throw(); });
  }
  // Allocate I/O-placed USM (shared on CPUs and devices without device USM).
  template <typename T> UsmPtr<T> alloc(const size_t n, const char* const failure) {
    return mom_usm_alloc<T>(queue, n, shared_io, failure);
  }
  // Allocate the tiny shared test/header buffers lazily; a failure leaves neither allocated.
  void ensure_io() {
    if (prePow && rows)
      return;
    constexpr const char* failure = "Can't allocate beamhash3 I/O buffers";
    auto new_pre_pow = mom_usm_alloc<uint64_t>(queue, 4, true, failure);
    auto new_rows =
        mom_usm_alloc<uint8_t>(queue, (size_t)BEAMHASH3_TEST_ROWS * ROW_BYTES, true, failure);
    prePow = std::move(new_pre_pow);
    rows = std::move(new_rows);
  }
  // Allocate only the selected compact, CUDA, or fused layout; partial construction releases all arenas.
  void ensure_arenas(bool log) {
    if (arenas_built)
      return;
    const uint64_t t0 = mom_now_ms();
    free_all_arenas();
    try {
      size_t total = 0;
      const bool cuda_layout = mom_is_cuda(device);
      const bool compact_layout = beam_compact_enabled(device);
      if (compact_layout) {
        for (unsigned parity = 0; parity < 2; ++parity) {
          constexpr const char* failure = "Can't allocate beamhash3 compact arenas";
          compact[parity] = alloc<BeamCompactRow>(COMPACT_MAX_ROWS, failure);
          total += COMPACT_MAX_ROWS * sizeof(BeamCompactRow);
          compact_ns[parity] = alloc<uint32_t>(COMPACT_MAX_NBUCKETS, failure);
          total += (size_t)COMPACT_MAX_NBUCKETS * sizeof(uint32_t);
        }
      } else if (cuda_layout) {
        const size_t n = (size_t)MIXED_CAP * MAX_MIXED_STRIDE;
        linear_mixed = alloc<uint32_t>(n, "Can't allocate beamhash3 CUDA linear arena");
        total += n * 4;
      } else {
        // Fused round 5 emits only one u64 work word per survivor.
        const size_t n = (size_t)MIXED_CAP * mixed_stride_u32(NUM_ROUNDS);
        final_mixed = alloc<uint32_t>(n, "Can't allocate beamhash3 final arena");
        total += n * 4;
      }
      if (!compact_layout) {
        for (unsigned L = 1; L <= NUM_ROUNDS; ++L) {
          const size_t tn = (size_t)MIXED_CAP * 2;
          tree[L] = alloc<uint32_t>(tn, "Can't allocate beamhash3 tree log");
          total += tn * 4;
        }
        if (cuda_layout) {
          const size_t n = (size_t)NBUCKETS * NSLOTS * MAX_BUCKET_STRIDE_ALL;
          constexpr const char* failure = "Can't allocate beamhash3 CUDA bucket arena";
          bucket[0] = alloc<uint32_t>(n, failure);
          total += n * 4;
          bucket_ns[0] = alloc<uint32_t>(NBUCKETS, failure);
          total += (size_t)NBUCKETS * 4;
        } else {
          const unsigned bucket_stride[2] = {MAX_BUCKET_STRIDE_EVEN, MAX_BUCKET_STRIDE_ODD};
          for (unsigned parity = 0; parity < 2; ++parity) {
            const size_t n = (size_t)NBUCKETS * NSLOTS * bucket_stride[parity];
            constexpr const char* failure = "Can't allocate beamhash3 fused bucket arena";
            bucket[parity] = alloc<uint32_t>(n, failure);
            total += n * 4;
            bucket_ns[parity] = alloc<uint32_t>(NBUCKETS, failure);
            total += (size_t)NBUCKETS * 4;
          }
        }
        mixed_count = mom_usm_alloc<uint32_t>(queue, NUM_ROUNDS + 1, true,
                                              "Can't allocate beamhash3 counters/candidates");
      }
      cand = alloc<BeamCandidate>(CAND_CAP, "Can't allocate beamhash3 counters/candidates");
      cand_count =
          mom_usm_alloc<uint32_t>(queue, 1, true, "Can't allocate beamhash3 counters/candidates");
      arenas_built = true;
      if (log) {
        std::fprintf(stderr, "beamhash3 Wagner arenas (~%.1f GiB) allocated (%llu ms)\n",
                     (double)total / (1024.0 * 1024.0 * 1024.0),
                     (unsigned long long)(mom_now_ms() - t0));
        std::fflush(stderr);
      }
    } catch (...) {
      free_all_arenas();
      throw;
    }
  }
  // Clear all layout-specific allocations and the built flag so a later solve can recreate state.
  void free_all_arenas() {
    final_mixed.reset();
    linear_mixed.reset();
    for (unsigned parity = 0; parity < 2; ++parity) {
      compact[parity].reset();
      compact_ns[parity].reset();
      bucket[parity].reset();
      bucket_ns[parity].reset();
    }
    for (auto& level : tree)
      level.reset();
    mixed_count.reset();
    cand.reset();
    cand_count.reset();
    arenas_built = false;
  }
};

// Release persistent solver states without allowing teardown failures to escape shutdown.
void beamhash3_cleanup_states() noexcept {
  clearPersistentDeviceStates<State>("beamhash3");
}
