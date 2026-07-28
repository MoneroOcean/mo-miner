struct BeamLevels {
  const uint32_t* tree[NUM_ROUNDS + 1];   // tree[L] (L=1..5): 2 u32/slot = parent mixed[L-1] indices
};
inline unsigned beam_walk_leaves(const BeamLevels& lv, uint32_t final_slot, uint32_t* leaves) {
  struct Frame { unsigned level; uint32_t slot; };
  Frame stack[NUM_ROUNDS + 2];
  int sp = 0;
  stack[sp++] = { NUM_ROUNDS, final_slot };
  unsigned n = 0;
  while (sp > 0) {
    Frame f = stack[--sp];
    if (f.level == 0) {
      if (n >= NUM_LEAVES) return 0;
      // Level-0 gen writes leaf `index` at slot `index`, so the stored leaf index == the slot itself.
      // (mixed0 is therefore not persisted; recovery returns the slot directly.)
      leaves[n++] = f.slot;
      continue;
    }
    const uint32_t* tl = lv.tree[f.level] + (size_t)f.slot * 2;
    if (sp + 2 > (int)(NUM_ROUNDS + 2)) return 0;
    // push follower first so the leader (lower index) is processed first -> leaves come out in the
    // canonical lowest-index-first tree order (matches getIndicesFromMinimal's field order).
    stack[sp++] = { f.level - 1, tl[1] };
    stack[sp++] = { f.level - 1, tl[0] };
  }
  return n;
}
inline bool beam_leaves_distinct(const uint32_t* leaves, unsigned n) {
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j = i + 1; j < n; ++j)
      if (leaves[i] == leaves[j]) return false;
  return true;
}

class BeamFinalKernel;
static sycl::event submit_final(sycl::queue& q, MomKernelBundle& kb, const uint32_t* d_mixed5,
                                uint32_t in_count, BeamLevels lv, BeamCandidate* d_cand,
                                uint32_t* d_cand_count, uint32_t cand_cap) {
  constexpr unsigned IN_ST = mixed_stride_u32(NUM_ROUNDS);   // level-5 stride (1 word + 0 idx) = 2
  constexpr unsigned WG = 128;
  const size_t global = ((size_t)in_count + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    h.parallel_for<BeamFinalKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
      const uint32_t s = (uint32_t)it.get_global_id(0);
      if (s >= in_count) return;
      const uint32_t* rec = d_mixed5 + (size_t)s * IN_ST;
      const uint32_t w0 = rec[0];   // low 32 bits of the 24-bit merged field
      if ((w0 & ((1u << COLLISION_BIT_SIZE) - 1u)) != 0) return;   // must be zero (full XOR collapse)
      uint32_t leaves[NUM_LEAVES];
      const unsigned n = beam_walk_leaves(lv, s, leaves);
      if (n != NUM_LEAVES) return;
      if (!beam_leaves_distinct(leaves, n)) return;
      const uint32_t idx = dev_atomic_u32(d_cand_count[0]).fetch_add(1u);
      if (idx >= cand_cap) return;
      for (unsigned i = 0; i < NUM_LEAVES; ++i) d_cand[idx].leaves[i] = leaves[i];
    });
  });
}

// ===========================================================================================
// Per-device state: queue + bundle + USM arenas.
// ===========================================================================================
class BeamState {
public:
  sycl::device device; sycl::queue queue; std::unique_ptr<MomKernelBundle> bundle; bool shared_io;
  // M1 I/O
  uint64_t* prePow = nullptr;   // 4 u64 (the SipHash key / IndividualWork)
  uint8_t*  rows = nullptr;     // BEAMHASH3_TEST_ROWS * ROW_BYTES (raw 448-bit leaf rows)
  // CUDA/HIP use two 64-byte compact arenas through level 3. Level 4 reuses compact[0] as packed
  // BeamLevel4Row storage while compact[1] retains the recovery trees. Intel retains the fused parity
  // arenas; MOM_BEAMHASH3_COMPACT=0 selects CUDA's linear or HIP's fused fallback.
  uint32_t* final_mixed = nullptr;  // fused layout: MIXED_CAP * mixed_stride_u32(5)
  uint32_t* linear_mixed = nullptr; // CUDA layout: MIXED_CAP * MAX_MIXED_STRIDE
  uint32_t* bucket[2] = {};         // fused parity arenas, or CUDA's reusable bucket[0]
  uint32_t* bucket_ns[2] = {};      // counters paired with the allocated bucket arenas
  BeamCompactRow* compact[2] = {};  // default CUDA/HIP packed two-arena Beam pipeline
  uint32_t* compact_ns[2] = {};
  uint32_t* mixed_count           = nullptr; // shared: per-round output counter (NUM_ROUNDS+1 entries)
  uint32_t* tree[NUM_ROUNDS + 1]  = {};   // tree[L] (L=1..5): NUM_ENTRIES * 2
  BeamCandidate* cand             = nullptr;
  uint32_t* cand_count            = nullptr;
  static constexpr uint32_t CAND_CAP = 8192;
  bool arenas_built = false; std::mutex mutex;

  explicit BeamState(const std::string& dev_str)
    : device(get_dev(dev_str)), queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !mom_has_usm_device(device)) {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device)))
      throw std::string("beamhash3 SYCL device does not support required allocations");
    bundle = std::make_unique<MomKernelBundle>(mom_get_exec_bundle(queue.get_context()));
  }
  ~BeamState() {
    // Miner workers exit immediately after a benchmark sample. Keep process teardown noexcept like
    // every other persistent SYCL state: live kernel errors have already propagated through the
    // explicit waits, while a late CUDA-runtime-unloading result must not abort a completed worker.
    sycl_cleanup_noexcept("beamhash3", [&] { queue.wait_and_throw(); free_all(); });
  }
  template <typename T> T* alloc(size_t n) {
    return shared_io ? sycl::malloc_shared<T>(n, queue) : sycl::malloc_device<T>(n, queue);
  }
  void free_ptr(auto*& p) { if (p) sycl::free(p, queue); p = nullptr; }
  void free_all() {
    free_ptr(prePow); free_ptr(rows);
    free_all_arenas();
  }
  void ensure_io() {
    if (prePow && rows) return;
    free_ptr(prePow); free_ptr(rows);
    prePow = sycl::malloc_shared<uint64_t>(4, queue);
    rows   = sycl::malloc_shared<uint8_t>((size_t)BEAMHASH3_TEST_ROWS * ROW_BYTES, queue);
    if (!prePow || !rows) throw std::string("Can't allocate beamhash3 I/O buffers");
  }
  void ensure_arenas(bool log) {
    if (arenas_built) return;
    const uint64_t t0 = now_ms();
    free_all_arenas();
    size_t total = 0;
    const bool cuda_layout = mom_is_cuda(device);
    const bool compact_layout = beam_compact_enabled(device);
    if (compact_layout) {
      const size_t n = (size_t)COMPACT_NBUCKETS * COMPACT_NSLOTS;
      for (unsigned parity = 0; parity < 2; ++parity) {
        compact[parity] = alloc<BeamCompactRow>(n); total += n * sizeof(BeamCompactRow);
        compact_ns[parity] = alloc<uint32_t>(COMPACT_NBUCKETS);
        total += (size_t)COMPACT_NBUCKETS * sizeof(uint32_t);
        if (!compact[parity] || !compact_ns[parity])
          throw std::string("Can't allocate beamhash3 compact arenas");
      }
    } else if (cuda_layout) {
      const size_t n = (size_t)MIXED_CAP * MAX_MIXED_STRIDE;
      linear_mixed = alloc<uint32_t>(n); total += n * 4;
      if (!linear_mixed) throw std::string("Can't allocate beamhash3 CUDA linear arena");
    } else {
      // Fused round 5 emits only one u64 work word per survivor.
      const size_t n = (size_t)MIXED_CAP * mixed_stride_u32(NUM_ROUNDS);
      final_mixed = alloc<uint32_t>(n); total += n * 4;
      if (!final_mixed) throw std::string("Can't allocate beamhash3 final arena");
    }
    if (!compact_layout) {
      for (unsigned L = 1; L <= NUM_ROUNDS; ++L) {
        const size_t tn = (size_t)MIXED_CAP * 2;
        tree[L] = alloc<uint32_t>(tn); total += tn * 4;
        if (!tree[L]) throw std::string("Can't allocate beamhash3 tree log");
      }
      if (cuda_layout) {
        const size_t n = (size_t)NBUCKETS * NSLOTS * MAX_BUCKET_STRIDE_ALL;
        bucket[0] = alloc<uint32_t>(n); total += n * 4;
        bucket_ns[0] = alloc<uint32_t>(NBUCKETS); total += (size_t)NBUCKETS * 4;
        if (!bucket[0] || !bucket_ns[0])
          throw std::string("Can't allocate beamhash3 CUDA bucket arena");
      } else {
        const unsigned bucket_stride[2] = { MAX_BUCKET_STRIDE_EVEN, MAX_BUCKET_STRIDE_ODD };
        for (unsigned parity = 0; parity < 2; ++parity) {
          const size_t n = (size_t)NBUCKETS * NSLOTS * bucket_stride[parity];
          bucket[parity] = alloc<uint32_t>(n); total += n * 4;
          bucket_ns[parity] = alloc<uint32_t>(NBUCKETS); total += (size_t)NBUCKETS * 4;
          if (!bucket[parity] || !bucket_ns[parity])
            throw std::string("Can't allocate beamhash3 fused bucket arena");
        }
      }
      mixed_count = sycl::malloc_shared<uint32_t>(NUM_ROUNDS + 1, queue);
    }
    cand        = alloc<BeamCandidate>(CAND_CAP);
    cand_count  = sycl::malloc_shared<uint32_t>(1, queue);
    if ((!compact_layout && !mixed_count) || !cand || !cand_count)
      throw std::string("Can't allocate beamhash3 counters/candidates");
    arenas_built = true;
    if (log) { std::fprintf(stderr, "beamhash3 Wagner arenas (~%.1f GiB) allocated (%llu ms)\n",
      (double)total / (1024.0 * 1024.0 * 1024.0), (unsigned long long)(now_ms() - t0)); std::fflush(stderr); }
  }
  void free_all_arenas() {
    free_ptr(final_mixed); free_ptr(linear_mixed);
    for (unsigned parity = 0; parity < 2; ++parity) {
      free_ptr(compact[parity]);
      free_ptr(compact_ns[parity]);
    }
    for (unsigned L = 0; L <= NUM_ROUNDS; ++L) free_ptr(tree[L]);
    for (unsigned parity = 0; parity < 2; ++parity) {
      free_ptr(bucket[parity]);
      free_ptr(bucket_ns[parity]);
    }
    free_ptr(mixed_count); free_ptr(cand); free_ptr(cand_count);
    arenas_built = false;
  }
  static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }
};

using BeamStateMap = std::map<std::string, std::unique_ptr<BeamState>>;

static BeamStateMap& beam_states() {
  static BeamStateMap* const states = new BeamStateMap;
  return *states;
}

static std::mutex& beam_states_mutex() {
  static std::mutex* const mutex = new std::mutex;
  return *mutex;
}

static BeamState& beam_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(beam_states_mutex());
  auto& state = beam_states()[dev_str];
  if (!state) state = std::make_unique<BeamState>(dev_str);
  return *state;
}

void beamhash3_cleanup_states() noexcept {
  try {
    std::lock_guard<std::mutex> lock(beam_states_mutex());
    beam_states().clear();
  } catch (...) {
    std::fprintf(stderr, "beamhash3: ordered SYCL cleanup failed\n");
  }
}
