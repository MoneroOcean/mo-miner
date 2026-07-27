// BeamHash III Wagner solver kernels + per-device state. Included into beamhash3.cpp inside
// namespace mom_beamhash3 (M2/M3).
//
// Pipeline (5 rounds, k=5):
//   gen/r1   : regenerate each of the 2^25 leaves, applyMix(448), scatter into bucketed[0].
//   round r  : collide bucketed[r-1] per bucket in SLM (low RESTBITS), XOR-merge, record the two parent
//              source IDs in tree[r], then -- for r<5 -- immediately apply the next round's mix and
//              scatter into the alternate bucket arena. Round 5 alone appends its compact final word
//              to a linear buffer for candidate recovery.
//   CUDA/HIP : alternate packed 64-byte rows through level 3, then retain that recovery arena while
//              level 4 emits thin 16-byte work+parent rows; MOM_BEAMHASH3_COMPACT=0 retains the older
//              CUDA-linear/HIP-fused fallback for diagnostics.
//   final    : a level-5 survivor whose merged 24-bit workBits == 0 (and 32 distinct leaves) is a
//              solution; walk tree[5..1] -> 32 leaf indices.
//
// remLen / mergeLen / padNum schedule, computed from the BeamHash III verify3 flow:
//   r1: remLen 448 (w7) padNum 1 -> mergeLen 424 (w7)   treeLen 1->2
//   r2: remLen 424 (w7) padNum 2 -> mergeLen 400 (w7)   treeLen 2->4
//   r3: remLen 400 (w7) padNum 4 -> mergeLen 376 (w6)   treeLen 4->8
//   r4: remLen 376 (w6) padNum 6 -> mergeLen 288 (w5)   treeLen 8->16
//   r5: remLen 288 (w5) padNum 9 -> mergeLen  24 (w1)   treeLen 16->32

// ---- Per-round compile-time schedule -------------------------------------------------------------
// Indexed by round r in 1..5.
constexpr unsigned ROUND_REMLEN[6]   = { 0, 448, 424, 400, 376, 288 };   // pre-round work-bit length
constexpr unsigned ROUND_MERGELEN[6] = { 0, 424, 400, 376, 288, 24  };   // post-merge work-bit length
constexpr unsigned ROUND_PADNUM[6]   = { 0, 1, 2, 4, 6, 9 };             // applyMix index count
// LEVEL L holds the input to round L+1. work words stored / indices stored (= padNum of round L+1).
constexpr unsigned LEVEL_WORDS[6] = { 7, 7, 7, 6, 5, 1 };  // ceil(remLen_of_next_round / 64), level5=1
// Indices stored per slot: min(treeSize=2^L, MAX_PAD=9). To reconstruct round R+1's first padNum (<=9)
// indices, every node must carry its OWN first 9 (a merge takes leader.prefix ++ follower.prefix capped
// at the merged tree's stored count) -- NOT merely padNum of the next round (that under-stores the
// leader and corrupts the round-5 mix; verified offline vs the JS oracle's full-tree applyMix).
constexpr unsigned LEVEL_NIDX[6]  = { 1, 2, 4, 8, 9, 0 };  // min(2^L, 9); level5 = final (no mix)
constexpr unsigned ceil_words(unsigned bits) { return (bits + 63) / 64; }

// Bucket records are consumed only after applyMix for the following round. BeamHash III deliberately
// discards the high 64 work bits after the round-4 mix, and after the round-5 mix only the low 48-bit
// collision key remains live. Do not carry those dead words through the random bucket arenas. Level 4
// also needs only its lead index (for canonical parent order); recovery uses the separately logged
// source ID. Linear CUDA handoff records retain LEVEL_WORDS/LEVEL_NIDX until their mix has run.
constexpr unsigned BUCKET_WORDS[6]     = { 7, 7, 7, 5, 1, 1 };
constexpr unsigned BUCKET_NIDX[6]      = { 1, 2, 4, 8, 1, 0 };
constexpr unsigned BUCKET_WORK_BITS[6] = { 448, 424, 400, 312, 48, 24 };

// ---- Bucketing: collide low 24 bits. RESTBITS in SLM, high BUCKBITS pick the global bucket. ----
constexpr unsigned RESTBITS  = 10;
constexpr unsigned BUCKBITS  = COLLISION_BIT_SIZE - RESTBITS;     // 14
constexpr unsigned NBUCKETS  = 1u << BUCKBITS;                    // 16384
constexpr unsigned NRESTBINS = 1u << RESTBITS;                    // 1024
// mean occupancy 2^25/16384 = 2048; measured peak ~2231 across rounds (≈ +4σ, σ=√2048≈45). 2368 keeps
// ~7.5σ over the mean (and +137 over the observed worst bucket) while trimming the bucket arena to ~3.0
// GiB. A slot overflow only drops that one element (collide clamps cnt<=NSLOTS); 7.5σ makes it astronomical.
constexpr unsigned NSLOTS    = 2368;

using dev_atomic_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device, sycl::access::address_space::global_space>;
using slm_atomic_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::work_group, sycl::access::address_space::local_space>;

// A "mixed" linear slot stores: LEVEL_WORDS[L] u64 workBits then LEVEL_NIDX[L] u32 indices. We store
// the u64 work words as 2 u32 each (lo, hi). Bucket slots use the post-mix compact shapes above; level
// 4 carries one explicit source word because its 16 dead work bits plus seven index-padding bits are
// three bits short of the 26-bit recovery source ID.
constexpr unsigned mixed_stride_u32(unsigned L) { return 2 * LEVEL_WORDS[L] + LEVEL_NIDX[L]; }
// Most bucket records do not need a trailing source word. A level-0 source is already its sole leaf
// prefix; at levels 1..3 the <=26-bit linear source ID fits in dead work bits followed (only where
// needed) by the unused high seven bits of stored 25-bit prefix indices. Level 4 is the compact-record
// exception described above. merge_workbits() masks work padding and prefix copies mask to 25 bits,
// so source metadata cannot affect PoW.
constexpr unsigned bucket_source_u32(unsigned L) { return L == 4 ? 1u : 0u; }
constexpr unsigned bucket_stride_u32(unsigned L) {
  return 2 * BUCKET_WORDS[L] + BUCKET_NIDX[L] + bucket_source_u32(L);
}
constexpr unsigned SOURCE_BITS = 26;
constexpr uint32_t SOURCE_MASK = (1u << SOURCE_BITS) - 1u;
constexpr uint32_t INDEX_MASK = (1u << INDEX_BITS) - 1u;
constexpr uint32_t low_mask_u32(unsigned bits) {
  return bits >= 32 ? 0xffffffffu : ((1u << bits) - 1u);
}
constexpr unsigned bucket_source_capacity(unsigned L) {
  if (L == 0) return INDEX_BITS;
  if (L == 4) return 32;
  const unsigned work_spare = BUCKET_WORDS[L] * 64 - BUCKET_WORK_BITS[L];
  return work_spare + BUCKET_NIDX[L] * (32 - INDEX_BITS);
}
static_assert(NUM_ENTRIES + NUM_ENTRIES / 32 < (1ull << SOURCE_BITS),
              "Beam source IDs must fit the packed bucket field");
static_assert(bucket_source_capacity(1) >= SOURCE_BITS && bucket_source_capacity(2) >= SOURCE_BITS &&
              bucket_source_capacity(3) >= SOURCE_BITS && bucket_source_capacity(4) >= SOURCE_BITS,
              "Every fused Beam level must carry its source ID losslessly");

template <unsigned L>
inline void bucket_store_source(uint32_t* rec, uint32_t source) {
  source &= SOURCE_MASK;
  if constexpr (L == 0) {
    // The generated level-0 record has prefix[0] == leaf slot == source.
    return;
  } else if constexpr (L == 4) {
    rec[2 * BUCKET_WORDS[L] + BUCKET_NIDX[L]] = source;
  } else {
    constexpr unsigned work_spare = BUCKET_WORDS[L] * 64 - BUCKET_WORK_BITS[L];
    constexpr unsigned work_take = work_spare < SOURCE_BITS ? work_spare : SOURCE_BITS;
    unsigned consumed = 0;
    if constexpr (work_take > 0) {
      constexpr unsigned last = BUCKET_WORDS[L] - 1;
      uint64_t word = (uint64_t)rec[2 * last] | ((uint64_t)rec[2 * last + 1] << 32);
      // Levels 1/2 arrive already masked; level 3 intentionally drops bits 312..375 only after its
      // mix. Clear the newly dead high bits before reusing them for the recovery source ID.
      word &= ~0ull >> work_spare;
      word |= (uint64_t)(source & low_mask_u32(work_take)) << (64 - work_spare);
      rec[2 * last] = (uint32_t)word;
      rec[2 * last + 1] = (uint32_t)(word >> 32);
      source >>= work_take;
      consumed = work_take;
    }
    constexpr unsigned prefix_base = 2 * BUCKET_WORDS[L];
#pragma unroll
    for (int i = (int)BUCKET_NIDX[L] - 1; i >= 0 && consumed < SOURCE_BITS; --i) {
      const unsigned take = std::min(32u - INDEX_BITS, SOURCE_BITS - consumed);
      rec[prefix_base + (unsigned)i] |= (source & low_mask_u32(take)) << INDEX_BITS;
      source >>= take;
      consumed += take;
    }
  }
}

template <unsigned L>
inline uint32_t bucket_load_source(const uint32_t* rec) {
  if constexpr (L == 0) {
    return rec[2 * BUCKET_WORDS[L]] & INDEX_MASK;
  } else if constexpr (L == 4) {
    return rec[2 * BUCKET_WORDS[L] + BUCKET_NIDX[L]] & SOURCE_MASK;
  } else {
    constexpr unsigned work_spare = BUCKET_WORDS[L] * 64 - BUCKET_WORK_BITS[L];
    constexpr unsigned work_take = work_spare < SOURCE_BITS ? work_spare : SOURCE_BITS;
    uint32_t source = 0;
    unsigned consumed = 0;
    if constexpr (work_take > 0) {
      constexpr unsigned last = BUCKET_WORDS[L] - 1;
      const uint64_t word = (uint64_t)rec[2 * last] | ((uint64_t)rec[2 * last + 1] << 32);
      source = (uint32_t)(word >> (64 - work_spare)) & low_mask_u32(work_take);
      consumed = work_take;
    }
    constexpr unsigned prefix_base = 2 * BUCKET_WORDS[L];
#pragma unroll
    for (int i = (int)BUCKET_NIDX[L] - 1; i >= 0 && consumed < SOURCE_BITS; --i) {
      const unsigned take = std::min(32u - INDEX_BITS, SOURCE_BITS - consumed);
      const uint32_t chunk = (rec[prefix_base + (unsigned)i] >> INDEX_BITS) & low_mask_u32(take);
      source |= chunk << consumed;
      consumed += take;
    }
    return source;
  }
}

// The fused pipeline alternates bucket arenas by level parity. Each arena only needs the widest record
// of the levels it can hold: even levels 0/2/4, odd levels 1/3. Round 5's output is only the compact
// level-5 word, so no wide linear scratch survives between rounds.
constexpr unsigned max_bucket_stride_parity(unsigned parity) {
  unsigned m = 0;
  for (unsigned L = parity; L < NUM_ROUNDS; L += 2) m = std::max(m, bucket_stride_u32(L));
  return m;
}
constexpr unsigned MAX_BUCKET_STRIDE_EVEN = max_bucket_stride_parity(0);
constexpr unsigned MAX_BUCKET_STRIDE_ODD  = max_bucket_stride_parity(1);
constexpr unsigned max_bucket_stride_all() {
  return std::max(MAX_BUCKET_STRIDE_EVEN, MAX_BUCKET_STRIDE_ODD);
}
constexpr unsigned max_mixed_stride() {
  unsigned m = 0;
  for (unsigned L = 1; L <= NUM_ROUNDS; ++L) m = std::max(m, mixed_stride_u32(L));
  return m;
}
constexpr unsigned MAX_BUCKET_STRIDE_ALL = max_bucket_stride_all();
constexpr unsigned MAX_MIXED_STRIDE = max_mixed_stride();

// A round can emit slightly MORE survivors than its input (a rest-bin with c slots yields ~c*(c-1)/2
// pairs, so the expected survivor count ~ input but the tail fluctuates above 2^25). Over-provision the
// parent-log source-ID namespace and compact final arena so no survivor (and therefore no solution) is
// dropped to the cap. Level 0 is the exact 2^25 leaf-index namespace and is never materialized.
// Measured per-round survivor counts hug 2^25 to within +0.006% (e.g. 33,556,370 vs 33,554,432). The
// over-provision absorbs the input-dependent tail; 1/32 (~+1.05M slots) is ~170x the observed excess.
constexpr uint64_t MIXED_CAP = (uint64_t)(NUM_ENTRIES + NUM_ENTRIES / 32);  // 2^25 * 1.03125

// ---- Compact two-arena solver -------------------------------------------------------------------
// BeamHash III was designed so the shrinking work bits and growing 25-bit index tree can share one
// 512-bit row. Levels 3 and 4 deliberately discard work bits that cannot affect a later collision;
// their complete 8/16-leaf trees then fit beside the remaining 312/48 work bits. Keeping that packed
// representation through every round removes both the wide linear CUDA handoff and the five separate
// recovery logs. This is the layout used by Beam's original BeamHashIII OpenCL solver, expressed here
// without OpenCL vector aliases so the same code can be evaluated by every SYCL backend.
struct BeamCandidate { uint32_t leaves[NUM_LEAVES]; };
struct alignas(16) BeamCompactRow { uint64_t word[8]; };
struct alignas(16) BeamLevel4Row {
  uint64_t work;
  uint32_t parent[2];
};
static_assert(sizeof(BeamCompactRow) == 64, "A compact Beam row must remain exactly 64 bytes");
static_assert(sizeof(BeamLevel4Row) == 16, "A thin Beam level-4 row must remain exactly 16 bytes");

constexpr unsigned COMPACT_BUCKET_BITS = 13;
constexpr unsigned COMPACT_NBUCKETS = 1u << COMPACT_BUCKET_BITS;  // low bits select physical storage
constexpr unsigned COMPACT_SUBGROUP_BITS = 1;    // bit 13 selects one of two collision workgroups
constexpr unsigned COMPACT_SUBGROUPS = 1u << COMPACT_SUBGROUP_BITS;
constexpr unsigned COMPACT_RESTBITS = COLLISION_BIT_SIZE - COMPACT_BUCKET_BITS - COMPACT_SUBGROUP_BITS;
constexpr unsigned COMPACT_NRESTBINS = 1u << COMPACT_RESTBITS;
static_assert(COMPACT_RESTBITS == 10, "Compact Beam SLM geometry must retain 1024 collision chains");
// Each physical bucket averages 4096 rows. Two collision workgroups split it into ~2048-row halves;
// the small margins below cover the measured input-dependent occupancy tail without wasting VRAM.
constexpr unsigned COMPACT_NSLOTS = 4480;
constexpr unsigned COMPACT_LOCAL_CAP = 2560;      // ~2048 rows per collision-workgroup half

inline bool beam_compact_enabled(const sycl::device& device) {
  if (!mom_is_cuda(device) && !mom_is_hip(device)) return false;
  const char* value = std::getenv("MOM_BEAMHASH3_COMPACT");
  return !value || (std::strcmp(value, "0") && std::strcmp(value, "false") &&
                    std::strcmp(value, "off"));
}

constexpr unsigned compact_work_bits(unsigned L) { return BUCKET_WORK_BITS[L]; }
constexpr unsigned compact_tree_len(unsigned L) { return 1u << L; }

inline uint32_t compact_get_index(const BeamCompactRow& row, unsigned work_bits, unsigned index) {
  const unsigned bit = work_bits + index * INDEX_BITS;
  const unsigned wi = bit >> 6;
  const unsigned off = bit & 63;
  uint64_t value = row.word[wi] >> off;
  if (off > 64 - INDEX_BITS && wi + 1 < 8) value |= row.word[wi + 1] << (64 - off);
  return (uint32_t)value & INDEX_MASK;
}

inline void compact_put_index(BeamCompactRow& row, unsigned work_bits, unsigned index, uint32_t value) {
  const unsigned bit = work_bits + index * INDEX_BITS;
  const unsigned wi = bit >> 6;
  const unsigned off = bit & 63;
  const uint64_t v = (uint64_t)(value & INDEX_MASK);
  row.word[wi] |= v << off;
  if (off > 64 - INDEX_BITS && wi + 1 < 8) row.word[wi + 1] |= v >> (64 - off);
}

template <unsigned L>
inline BeamCompactRow compact_make_row(const uint64_t work[WORK_WORDS], const uint32_t* tree) {
  BeamCompactRow row{};
  constexpr unsigned live_bits = compact_work_bits(L);
  constexpr unsigned full_words = live_bits / 64;
  constexpr unsigned tail_bits = live_bits & 63;
  for (unsigned i = 0; i < full_words; ++i) row.word[i] = work[i];
  if constexpr (tail_bits != 0) row.word[full_words] = work[full_words] & (~0ull >> (64 - tail_bits));
  for (unsigned i = 0; i < compact_tree_len(L); ++i) compact_put_index(row, live_bits, i, tree[i]);
  return row;
}

template <unsigned L>
inline void compact_load_work(const BeamCompactRow& row, uint64_t work[WORK_WORDS]) {
  constexpr unsigned live_bits = compact_work_bits(L);
  constexpr unsigned full_words = live_bits / 64;
  constexpr unsigned tail_bits = live_bits & 63;
  for (unsigned i = 0; i < WORK_WORDS; ++i) work[i] = 0;
  for (unsigned i = 0; i < full_words; ++i) work[i] = row.word[i];
  if constexpr (tail_bits != 0) work[full_words] = row.word[full_words] & (~0ull >> (64 - tail_bits));
}

// Level 3 ends with a 200-bit tree starting at bit 312. Round 4 needs the first nine indices for
// applyMix, then stores all 16 parent indices starting at bit 48 in the level-4 row. Express those two
// operations as fixed bitstream splices instead of decoding 16 u32s into private memory and packing
// them again. On CUDA that temporary array caused a register-pressure cliff in round 4.
inline void compact_l3_tree_stream(const BeamCompactRow& row, uint64_t tree[4]) {
  tree[0] = (row.word[4] >> 56) | (row.word[5] << 8);
  tree[1] = (row.word[5] >> 56) | (row.word[6] << 8);
  tree[2] = (row.word[6] >> 56) | (row.word[7] << 8);
  tree[3] = row.word[7] >> 56;  // only the low eight bits are live
}

inline void compact_round4_mix(uint64_t work[WORK_WORDS], const uint64_t leader[4],
                               const uint64_t follower[4]) {
  // applyMix(remLen=288, padNum=9): append leader's eight indices and the low 24 live bits of the
  // follower's first index to the 512-bit temporary value, then perform the standard rotated sum.
  uint64_t t[8];
  for (unsigned i = 0; i < WORK_WORDS; ++i) t[i] = work[i];
  t[4] |= leader[0] << 32;
  t[5] = (leader[0] >> 32) | (leader[1] << 32);
  t[6] = (leader[1] >> 32) | (leader[2] << 32);
  t[7] = (leader[2] >> 32) | (leader[3] << 32) | ((follower[0] & 0xffffffull) << 40);
  const uint64_t result = rotl64(t[0], 29) + rotl64(t[1], 58) + rotl64(t[2], 23) +
                          rotl64(t[3], 52) + rotl64(t[4], 17) + rotl64(t[5], 46) +
                          rotl64(t[6], 11) + rotl64(t[7], 40);
  work[0] = rotl64(result, 24);
}

#if !defined(MOM_SYCL_PORTABLE_OPENCL)
class BeamCompactSeedKernel;
static sycl::event submit_compact_seed(sycl::queue& q, MomKernelBundle& kb, const uint64_t* d_prePow,
                                       BeamCompactRow* d_out, uint32_t* d_out_ns, unsigned wg) {
  const size_t global = ((size_t)NUM_ENTRIES + wg - 1) / wg * wg;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    constexpr uint32_t SENTINEL = 0xffffffffu;
    constexpr unsigned ROW_U32 = sizeof(BeamCompactRow) / sizeof(uint32_t);
    sycl::local_accessor<uint32_t, 1> stage(sycl::range<1>((size_t)wg * ROW_U32), h);
    sycl::local_accessor<uint32_t, 1> destination(sycl::range<1>(wg), h);
    h.parallel_for<BeamCompactSeedKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)), [=](sycl::nd_item<1> it) {
      const uint32_t index = (uint32_t)it.get_global_id(0);
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);
      if (index < NUM_ENTRIES) {
        uint64_t prePow[4];
        for (unsigned i = 0; i < 4; ++i) prePow[i] = d_prePow[i];
        uint64_t work[WORK_WORDS];
        make_leaf(prePow, index, work);
        uint32_t tree[1] = { index };
        apply_mix(work, tree, 1, ROUND_REMLEN[1]);
        const BeamCompactRow row = compact_make_row<0>(work, tree);
        const uint32_t bucket = (uint32_t)work[0] & (COMPACT_NBUCKETS - 1u);
        const uint32_t slot = dev_atomic_u32(d_out_ns[bucket]).fetch_add(1u);
        if (slot < COMPACT_NSLOTS) {
          const uint32_t* src = reinterpret_cast<const uint32_t*>(&row);
          for (unsigned k = 0; k < ROW_U32; ++k) stage[(size_t)lid * ROW_U32 + k] = src[k];
          destination[lid] = bucket * COMPACT_NSLOTS + slot;
        } else {
          destination[lid] = SENTINEL;
        }
      } else {
        destination[lid] = SENTINEL;
      }
      sycl::group_barrier(it.get_group());

      // Direct per-lane 64-byte random stores make seed twice as slow as its SipHash work. Stage one
      // workgroup, then let consecutive subgroup lanes flush each row as a coalesced burst.
      auto sg = it.get_sub_group();
      const unsigned sg_size = (unsigned)sg.get_max_local_range()[0];
      const unsigned lane = (unsigned)sg.get_local_id()[0];
      const unsigned subgroups = (lsz + sg_size - 1) / sg_size;
      const unsigned subgroup = lid / sg_size;
      uint32_t* output = reinterpret_cast<uint32_t*>(d_out);
      for (unsigned r = subgroup; r < lsz; r += subgroups) {
        const uint32_t dst = destination[r];
        if (dst == SENTINEL) continue;
        for (unsigned k = lane; k < ROW_U32; k += sg_size)
          output[(size_t)dst * ROW_U32 + k] = stage[(size_t)r * ROW_U32 + k];
      }
    });
  });
}

template <unsigned R> class BeamCompactRoundKernel;
template <unsigned R>
static sycl::event submit_compact_round(sycl::queue& q, MomKernelBundle& kb,
                                        const BeamCompactRow* d_in, const uint32_t* d_in_ns,
                                        BeamCompactRow* d_out, uint32_t* d_out_ns,
                                        BeamCandidate* d_cand, uint32_t* d_cand_count,
                                        uint32_t cand_cap, unsigned wg) {
  static_assert(R >= 1 && R <= NUM_ROUNDS);
  constexpr unsigned L = R - 1;
  constexpr unsigned IN_TREE = compact_tree_len(L);
  constexpr unsigned OUT_TREE = compact_tree_len(R);
  constexpr unsigned mergeLen = ROUND_MERGELEN[R];
  const size_t global = (size_t)COMPACT_NBUCKETS * COMPACT_SUBGROUPS * wg;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    sycl::local_accessor<uint32_t, 1> bin_head(sycl::range<1>(COMPACT_NRESTBINS), h);
    sycl::local_accessor<uint32_t, 1> bin_next(sycl::range<1>(COMPACT_LOCAL_CAP), h);
    sycl::local_accessor<uint32_t, 1> slot_index(sycl::range<1>(COMPACT_LOCAL_CAP), h);
    sycl::local_accessor<uint32_t, 1> local_count(sycl::range<1>(1), h);
    h.parallel_for<BeamCompactRoundKernel<R>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)), [=](sycl::nd_item<1> it) {
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);
      const uint32_t group = (uint32_t)it.get_group(0);
      const uint32_t bucket = group / COMPACT_SUBGROUPS;
      const uint32_t half = group & (COMPACT_SUBGROUPS - 1u);
      if (lid == 0) local_count[0] = 0;
      for (unsigned b = lid; b < COMPACT_NRESTBINS; b += lsz) bin_head[b] = 0;
      sycl::group_barrier(it.get_group());

      uint32_t count = d_in_ns[bucket];
      if (count > COMPACT_NSLOTS) count = COMPACT_NSLOTS;
      const BeamCompactRow* input = d_in + (size_t)bucket * COMPACT_NSLOTS;
      // Two workgroups make one sequential pass each over a physical bucket and retain their own
      // one-bit half. This trades one coalesced key read per row for 2x fewer random output regions,
      // while keeping the collision work itself at the original 2^14-way parallelism.
      for (unsigned s = lid; s < count; s += lsz) {
        const uint32_t collision = (uint32_t)input[s].word[0] & ((1u << COLLISION_BIT_SIZE) - 1u);
        if (((collision >> COMPACT_BUCKET_BITS) & (COMPACT_SUBGROUPS - 1u)) != half) continue;
        const uint32_t pos = slm_atomic_u32(local_count[0]).fetch_add(1u);
        if (pos >= COMPACT_LOCAL_CAP) continue;
        const uint32_t rest = collision >> (COMPACT_BUCKET_BITS + COMPACT_SUBGROUP_BITS);
        const uint32_t prev = slm_atomic_u32(bin_head[rest]).exchange(pos + 1u);
        slot_index[pos] = s;
        bin_next[pos] = prev;
      }
      sycl::group_barrier(it.get_group());

      const uint32_t used = std::min<uint32_t>(local_count[0], COMPACT_LOCAL_CAP);
      for (unsigned pos = lid; pos < used; pos += lsz) {
        const uint32_t a_slot = slot_index[pos];
        BeamCompactRow arow{};
        bool have_a = false;
        uint64_t aw[WORK_WORDS];
        uint32_t a_lead = 0;
        for (uint32_t link = bin_next[pos]; link != 0; link = bin_next[link - 1u]) {
          const uint32_t b_slot = slot_index[link - 1u];
          if (!have_a) {
            arow = input[a_slot];
            compact_load_work<L>(arow, aw);
            a_lead = compact_get_index(arow, compact_work_bits(L), 0);
            have_a = true;
          }
          const BeamCompactRow brow = input[b_slot];
          if constexpr (R == NUM_ROUNDS) {
            // The physical bucket, half and SLM rest bin already matched low 24 bits. Beam's
            // final round needs all 48 remaining bits equal.
            if (((arow.word[0] ^ brow.word[0]) & 0x0000ffffffffffffull) != 0) continue;
          }
          uint64_t bw[WORK_WORDS];
          compact_load_work<L>(brow, bw);
          const uint32_t b_lead = compact_get_index(brow, compact_work_bits(L), 0);
          const bool a_first = a_lead < b_lead;
          const BeamCompactRow& leader = a_first ? arow : brow;
          const BeamCompactRow& follower = a_first ? brow : arow;
          if constexpr (R == NUM_ROUNDS) {
            uint32_t tree[OUT_TREE];
            for (unsigned i = 0; i < IN_TREE; ++i)
              tree[i] = compact_get_index(leader, compact_work_bits(L), i);
            for (unsigned i = 0; i < IN_TREE; ++i)
              tree[IN_TREE + i] = compact_get_index(follower, compact_work_bits(L), i);
            bool distinct = true;
            for (unsigned i = 0; i < OUT_TREE && distinct; ++i)
              for (unsigned j = i + 1; j < OUT_TREE; ++j)
                if (tree[i] == tree[j]) { distinct = false; break; }
            if (!distinct) continue;
            const uint32_t ci = dev_atomic_u32(d_cand_count[0]).fetch_add(1u);
            if (ci < cand_cap)
              for (unsigned i = 0; i < OUT_TREE; ++i) d_cand[ci].leaves[i] = tree[i];
          } else {
            uint64_t merged[WORK_WORDS];
            merge_workbits(aw, bw, mergeLen, merged);
            if constexpr (R == 4) {
              uint64_t leader_tree[4], follower_tree[4];
              compact_l3_tree_stream(leader, leader_tree);
              compact_l3_tree_stream(follower, follower_tree);
              compact_round4_mix(merged, leader_tree, follower_tree);
              const uint32_t out_bucket = (uint32_t)merged[0] & (COMPACT_NBUCKETS - 1u);
              const uint32_t out_slot = dev_atomic_u32(d_out_ns[out_bucket]).fetch_add(1u);
              if (out_slot < COMPACT_NSLOTS) {
                const uint32_t a_source = bucket * COMPACT_NSLOTS + a_slot;
                const uint32_t b_source = bucket * COMPACT_NSLOTS + b_slot;
                BeamLevel4Row* level4 = reinterpret_cast<BeamLevel4Row*>(d_out);
                level4[(size_t)out_bucket * COMPACT_NSLOTS + out_slot] = {
                  merged[0] & 0x0000ffffffffffffull,
                  { a_first ? a_source : b_source, a_first ? b_source : a_source }
                };
              }
            } else {
              uint32_t tree[OUT_TREE];
              for (unsigned i = 0; i < IN_TREE; ++i)
                tree[i] = compact_get_index(leader, compact_work_bits(L), i);
              for (unsigned i = 0; i < IN_TREE; ++i)
                tree[IN_TREE + i] = compact_get_index(follower, compact_work_bits(L), i);
              apply_mix(merged, tree, OUT_TREE, ROUND_REMLEN[R + 1]);
              const BeamCompactRow outrow = compact_make_row<R>(merged, tree);
              const uint32_t out_bucket = (uint32_t)merged[0] & (COMPACT_NBUCKETS - 1u);
              const uint32_t out_slot = dev_atomic_u32(d_out_ns[out_bucket]).fetch_add(1u);
              if (out_slot < COMPACT_NSLOTS)
                d_out[(size_t)out_bucket * COMPACT_NSLOTS + out_slot] = outrow;
            }
          }
        }
      }
    });
  });
}

class BeamCompactFinalThinKernel;
static sycl::event submit_compact_final_thin(
    sycl::queue& q, MomKernelBundle& kb, const BeamLevel4Row* d_in,
    const uint32_t* d_in_ns, const BeamCompactRow* d_level3,
    BeamCandidate* d_cand, uint32_t* d_cand_count, uint32_t cand_cap, unsigned wg) {
  const size_t global = (size_t)COMPACT_NBUCKETS * COMPACT_SUBGROUPS * wg;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    sycl::local_accessor<uint32_t, 1> bin_head(sycl::range<1>(COMPACT_NRESTBINS), h);
    sycl::local_accessor<uint32_t, 1> bin_next(sycl::range<1>(COMPACT_LOCAL_CAP), h);
    sycl::local_accessor<uint32_t, 1> slot_index(sycl::range<1>(COMPACT_LOCAL_CAP), h);
    sycl::local_accessor<uint32_t, 1> local_count(sycl::range<1>(1), h);
    h.parallel_for<BeamCompactFinalThinKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)), [=](sycl::nd_item<1> it) {
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);
      const uint32_t group = (uint32_t)it.get_group(0);
      const uint32_t bucket = group / COMPACT_SUBGROUPS;
      const uint32_t half = group & (COMPACT_SUBGROUPS - 1u);
      if (lid == 0) local_count[0] = 0;
      for (unsigned b = lid; b < COMPACT_NRESTBINS; b += lsz) bin_head[b] = 0;
      sycl::group_barrier(it.get_group());

      uint32_t count = d_in_ns[bucket];
      if (count > COMPACT_NSLOTS) count = COMPACT_NSLOTS;
      const BeamLevel4Row* input = d_in + (size_t)bucket * COMPACT_NSLOTS;
      for (unsigned s = lid; s < count; s += lsz) {
        const uint32_t collision = (uint32_t)input[s].work & ((1u << COLLISION_BIT_SIZE) - 1u);
        if (((collision >> COMPACT_BUCKET_BITS) & (COMPACT_SUBGROUPS - 1u)) != half) continue;
        const uint32_t pos = slm_atomic_u32(local_count[0]).fetch_add(1u);
        if (pos >= COMPACT_LOCAL_CAP) continue;
        const uint32_t rest = collision >> (COMPACT_BUCKET_BITS + COMPACT_SUBGROUP_BITS);
        const uint32_t prev = slm_atomic_u32(bin_head[rest]).exchange(pos + 1u);
        slot_index[pos] = s;
        bin_next[pos] = prev;
      }
      sycl::group_barrier(it.get_group());

      const uint32_t used = std::min<uint32_t>(local_count[0], COMPACT_LOCAL_CAP);
      for (unsigned pos = lid; pos < used; pos += lsz) {
        const BeamLevel4Row arow = input[slot_index[pos]];
        for (uint32_t link = bin_next[pos]; link != 0; link = bin_next[link - 1u]) {
          const BeamLevel4Row brow = input[slot_index[link - 1u]];
          if (((arow.work ^ brow.work) & 0x0000ffffffffffffull) != 0) continue;

          const BeamCompactRow a_leader = d_level3[arow.parent[0]];
          const BeamCompactRow b_leader = d_level3[brow.parent[0]];
          const bool a_first =
            compact_get_index(a_leader, compact_work_bits(3), 0) <
            compact_get_index(b_leader, compact_work_bits(3), 0);
          const uint32_t source[4] = {
            a_first ? arow.parent[0] : brow.parent[0],
            a_first ? arow.parent[1] : brow.parent[1],
            a_first ? brow.parent[0] : arow.parent[0],
            a_first ? brow.parent[1] : arow.parent[1],
          };
          uint32_t tree[NUM_LEAVES];
          for (unsigned part = 0; part < 4; ++part) {
            const BeamCompactRow row = d_level3[source[part]];
            for (unsigned i = 0; i < 8; ++i)
              tree[part * 8 + i] = compact_get_index(row, compact_work_bits(3), i);
          }
          bool distinct = true;
          for (unsigned i = 0; i < NUM_LEAVES && distinct; ++i)
            for (unsigned j = i + 1; j < NUM_LEAVES; ++j)
              if (tree[i] == tree[j]) { distinct = false; break; }
          if (!distinct) continue;
          const uint32_t ci = dev_atomic_u32(d_cand_count[0]).fetch_add(1u);
          if (ci < cand_cap)
            for (unsigned i = 0; i < NUM_LEAVES; ++i) d_cand[ci].leaves[i] = tree[i];
        }
      }
    });
  });
}
#endif

// ---- gen is FUSED into round-1 mix-scatter (see submit_mix_scatter, R==1): the leaf at slot==index is
// regenerated in-kernel via make_leaf, so no separate gen kernel and no level-0 DRAM arena. This elides
// a full level-0 write (~2 GiB) plus the round-1 read of it; scatter is DRAM-write bound so the extra
// siphash compute is free. Recovery still maps leaf index == level-0 slot directly. ----

// ---- Round r mix-scatter: read mixed[L] (L=r-1), applyMix(remLen), bucket by hi BUCKBITS of the low-24
// collision field, write a bucketed slot carrying the packed source linear index. ----
//
// SCATTER COALESCING (the dominant ~66%-of-solve cost). Naively, one lane writes its whole OUT_ST-u32
// (64..84 byte) record to a near-random global bucket slot; a warp's 32 lanes then hit 32 independent
// bucket regions, so each of the OUT_ST store instructions fans out to ~32 distinct cache lines -- a
// ~OUT_ST-x write-transaction amplification, and the kernel is DRAM-write-BW bound (WG sweep is flat ->
// not occupancy/atomic bound). An SLM-radix bucket grouping does NOT help: with 16384 buckets a WG of
// 256 survivors spreads ~1-per-bucket (run length ~1), so there is no same-bucket adjacency to batch.
// Instead we TRANSPOSE the wide-record COPY: each lane stages its computed record into SLM and reserves
// its global slot (atomic, unchanged), then the warp cooperatively copies one record at a time with
// consecutive lanes writing consecutive u32 of that record. Each ~84-byte record is then ONE coalesced
// transaction (~1 line) instead of ~OUT_ST scattered ones -> ~OUT_ST-x fewer write transactions. The
// bucket arena layout, bucket_ns semantics, NSLOTS and the collide kernel are all untouched; only the
// intra-bucket slot ASSIGNMENT order can change, which collide (rest-bin linked lists over ALL slots,
// order-independent) tolerates. Drops past NSLOTS / past in_count are honored via a sentinel.
template <unsigned R, bool DEVICE_COUNT = false> class BeamMixScatterKernel;
template <unsigned R, bool DEVICE_COUNT = false>
static sycl::event submit_mix_scatter(sycl::queue& q, MomKernelBundle& kb, const uint32_t* d_mixed,
                                      uint32_t in_count, uint32_t* d_bucket, uint32_t* d_bucket_ns,
                                      unsigned WG, const uint64_t* d_prePow = nullptr,
                                      const uint32_t* d_in_count = nullptr) {
  constexpr unsigned L = R - 1;
  constexpr unsigned IN_ST  = mixed_stride_u32(L);
  constexpr unsigned OUT_ST = bucket_stride_u32(L);
  // slot (<=NSLOTS<2^12) packed in low 12 bits, bucket (<2^14) above; SENTINEL = dropped/out-of-range.
  constexpr uint32_t SLOT_BITS = 12;
  constexpr uint32_t SENTINEL  = 0xffffffffu;
  static_assert(NSLOTS < (1u << SLOT_BITS), "NSLOTS must fit SLOT_BITS for the packed dst");
  static_assert(NBUCKETS <= (1u << (32 - SLOT_BITS)), "bucket must fit the packed dst");
  const size_t global = ((size_t)in_count + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    // Per-WG staging: each lane's OUT_ST-u32 record + its packed (bucket,slot) destination.
    sycl::local_accessor<uint32_t, 1> stage(sycl::range<1>((size_t)WG * OUT_ST), h);
    sycl::local_accessor<uint32_t, 1> dst(sycl::range<1>(WG), h);
    h.parallel_for<BeamMixScatterKernel<R, DEVICE_COUNT>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
      const uint32_t s   = (uint32_t)it.get_global_id(0);
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);
      uint32_t* st = &stage[(size_t)lid * OUT_ST];
      // A queued CUDA handoff dispatches the allocation bound and consumes the preceding collide
      // kernel's device-written count. The in-order queue is the memory dependency; avoiding a host
      // read/wait between rounds costs only one uniform, cache-hot counter load per work-item.
      uint32_t active_count = in_count;
      if constexpr (DEVICE_COUNT)
        active_count = std::min<uint32_t>(d_in_count[0], (uint32_t)MIXED_CAP);

      if (s < active_count) {
        uint64_t w[WORK_WORDS];
        for (unsigned i = 0; i < WORK_WORDS; ++i) w[i] = 0;
        uint32_t tree[LEVEL_NIDX[L] > 0 ? LEVEL_NIDX[L] : 1];
        if constexpr (R == 1) {
          // GEN FUSION: round 1's input is the leaf list, where slot == leaf index. Regenerate the
          // leaf in-kernel (make_leaf = LEVEL_WORDS[0] siphash24 calls) instead of reading a level-0
          // arena. This elides the separate gen kernel AND the level-0 DRAM write+read (~2 GiB each
          // way). Scatter is DRAM-write bound, so trading siphash compute for that read is free.
          uint64_t prePow[4];
          for (unsigned i = 0; i < 4; ++i) prePow[i] = d_prePow[i];
          make_leaf(prePow, s, w);
          tree[0] = s;   // the single leaf index (LEVEL_NIDX[0] == 1)
        } else {
          const uint32_t* rec = d_mixed + (size_t)s * IN_ST;
          for (unsigned i = 0; i < LEVEL_WORDS[L]; ++i)
            w[i] = (uint64_t)rec[2 * i] | ((uint64_t)rec[2 * i + 1] << 32);
          for (unsigned i = 0; i < LEVEL_NIDX[L]; ++i) tree[i] = rec[2 * LEVEL_WORDS[L] + i];
        }

        apply_mix(w, tree, LEVEL_NIDX[L], ROUND_REMLEN[R]);
        const uint32_t coll = collision_bits(w);
        const uint32_t bucket = coll >> RESTBITS;                 // hi BUCKBITS
        const uint32_t slot = dev_atomic_u32(d_bucket_ns[bucket]).fetch_add(1u);
        if (slot >= NSLOTS) {
          dst[lid] = SENTINEL;
        } else {
          // Stage the OUT_ST-u32 record into SLM (same byte layout the old direct write produced).
          for (unsigned i = 0; i < BUCKET_WORDS[L]; ++i) {
            st[2 * i]     = (uint32_t)(w[i] & 0xffffffffull);
            st[2 * i + 1] = (uint32_t)(w[i] >> 32);
          }
          for (unsigned i = 0; i < BUCKET_NIDX[L]; ++i)
            st[2 * BUCKET_WORDS[L] + i] = tree[i];
          bucket_store_source<L>(st, s);
          dst[lid] = (bucket << SLOT_BITS) | slot;
        }
      } else {
        dst[lid] = SENTINEL;
      }
      sycl::group_barrier(it.get_group());

      // Warp-cooperative coalesced flush: each sub-group copies one whole record at a time with
      // consecutive lanes writing consecutive u32 -> one coalesced burst per ~84-byte record.
      auto sg = it.get_sub_group();
      const unsigned sg_sz   = (unsigned)sg.get_max_local_range()[0];
      const unsigned lane     = (unsigned)sg.get_local_id()[0];
      const unsigned wg_warps = (lsz + sg_sz - 1) / sg_sz;
      const unsigned warp_id  = lid / sg_sz;
      for (unsigned r = warp_id; r < lsz; r += wg_warps) {
        const uint32_t packed = dst[r];
        if (packed == SENTINEL) continue;
        const uint32_t bucket = packed >> SLOT_BITS;
        const uint32_t slot   = packed & ((1u << SLOT_BITS) - 1u);
        uint32_t* o = d_bucket + ((size_t)bucket * NSLOTS + slot) * OUT_ST;
        const uint32_t* src = &stage[(size_t)r * OUT_ST];
        for (unsigned k = lane; k < OUT_ST; k += sg_sz) o[k] = src[k];
      }
    });
  });
}

// ---- Round r collide/advance: one WG per bucket. Read bucketed[L], collide low RESTBITS in SLM,
// XOR-merge (mergeLen), build the next index prefix, and log the parents. For rounds 1..4, applyMix for
// the following round immediately and scatter the result into the alternate bucket arena. Only round 5
// writes a compact linear level-5 record. This removes four wide linear writes, four wide linear reads,
// and four standalone scatter launches from the steady solve.
template <unsigned R, bool FUSED, bool PREDECESSOR_ONLY> class BeamCollideKernel;
template <unsigned R, bool FUSED, bool PREDECESSOR_ONLY>
static sycl::event submit_collide(sycl::queue& q, MomKernelBundle& kb, const uint32_t* d_bucket,
                                  const uint32_t* d_bucket_ns, uint32_t* d_final_out,
                                  uint32_t* d_out_count, uint32_t* d_tree, uint32_t* d_next_bucket,
                                  uint32_t* d_next_bucket_ns, unsigned wg) {
  constexpr unsigned L = R - 1;
  constexpr unsigned IN_ST   = bucket_stride_u32(L);
  constexpr unsigned OUT_L   = R;           // output level
  constexpr unsigned FINAL_ST = mixed_stride_u32(OUT_L);
  constexpr unsigned NEXT_ST  = bucket_stride_u32(OUT_L);
  constexpr unsigned mergeLen = ROUND_MERGELEN[R];
  const size_t global = (size_t)NBUCKETS * wg;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    sycl::local_accessor<uint32_t, 1> bin_head(sycl::range<1>(NRESTBINS), h);
    sycl::local_accessor<uint32_t, 1> bin_next(sycl::range<1>(NSLOTS), h);
    // Stage every slot's 24-bit collision key in SLM. The wide bucket records are read once here (the
    // inverse of the scatter-transpose); the rest-bin build and -- crucially -- the per-link compare in
    // the collide pass then hit SLM instead of re-striding the wide (60..84B) record in DRAM. Most link
    // traversals fail the full-24-bit match, so this removes the dominant scattered b[0],b[1] DRAM reads.
    sycl::local_accessor<uint32_t, 1> coll_key(sycl::range<1>(NSLOTS), h);
    h.parallel_for<BeamCollideKernel<R, FUSED, PREDECESSOR_ONLY>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)), [=](sycl::nd_item<1> it) {
      const uint32_t bucket = (uint32_t)it.get_group(0);
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);

      uint32_t cnt = d_bucket_ns[bucket];
      if (cnt > NSLOTS) cnt = NSLOTS;
      for (unsigned r = lid; r < NRESTBINS; r += lsz) bin_head[r] = 0;

      const uint32_t* in_base = d_bucket + (size_t)bucket * NSLOTS * IN_ST;
      constexpr uint32_t COLL_MASK = (1u << COLLISION_BIT_SIZE) - 1u;

      // Stage 24-bit collision keys into SLM (one strided DRAM read per slot, vs. two later).
      for (unsigned s = lid; s < cnt; s += lsz)
        coll_key[s] = in_base[(size_t)s * IN_ST] & COLL_MASK;
      sycl::group_barrier(it.get_group());

      // Build the rest-bin linked lists (key = low RESTBITS of the collision field), from SLM keys.
      for (unsigned s = lid; s < cnt; s += lsz) {
        const uint32_t rb = coll_key[s] & (NRESTBINS - 1u);
        bin_next[s] = slm_atomic_u32(bin_head[rb]).exchange(s + 1u);
      }
      sycl::group_barrier(it.get_group());

      for (unsigned s = lid; s < cnt; s += lsz) {
        const uint32_t a_coll = coll_key[s];
        const uint32_t rb = a_coll & (NRESTBINS - 1u);

        const uint32_t* a = nullptr;   // a's wide record is read lazily, only when a partner matches.
        uint64_t aw[WORK_WORDS];

        // Level Zero benefits from following only this slot's predecessor chain: atomic insertion
        // order defines which endpoint emits each unordered pair, so the numeric t>=s test and the
        // remainder of the chain are unnecessary. On HIP this was neutral/slightly slower (DPC++
        // 14.68 vs 14.70--14.71 H/s). The analogous predecessor walk is also ~5% slower on NVIDIA
        // warps in Equihash, so CUDA conservatively retains its already-fast common-head kernel.
        // Keep both as compile-time variants selected once from the host device.
        uint32_t link = PREDECESSOR_ONLY ? bin_next[s] : bin_head[rb];
        for (; link != 0; link = bin_next[link - 1u]) {
          const uint32_t t = link - 1u;
          if constexpr (!PREDECESSOR_ONLY) if (t >= s) continue;
          if (coll_key[t] != a_coll) continue;                   // full 24-bit collision field must match (SLM)

          if (!a) {   // lazy: only now do we touch the wide records from DRAM.
            a = in_base + (size_t)s * IN_ST;
            for (unsigned i = 0; i < WORK_WORDS; ++i) aw[i] = 0;
            for (unsigned i = 0; i < BUCKET_WORDS[L]; ++i)
              aw[i] = (uint64_t)a[2 * i] | ((uint64_t)a[2 * i + 1] << 32);
          }
          const uint32_t* b = in_base + (size_t)t * IN_ST;
          uint64_t bw[WORK_WORDS]; for (unsigned i = 0; i < WORK_WORDS; ++i) bw[i] = 0;
          for (unsigned i = 0; i < BUCKET_WORDS[L]; ++i)
            bw[i] = (uint64_t)b[2 * i] | ((uint64_t)b[2 * i + 1] << 32);

          // The two parents' first-NIDX[L] index lists + lead index decide the merge order
          // (lowest-index-first). Leader = the node whose lead index (tree[0]) is smaller.
          const uint32_t a_lead = a[2 * BUCKET_WORDS[L] + 0] & INDEX_MASK;
          const uint32_t b_lead = b[2 * BUCKET_WORDS[L] + 0] & INDEX_MASK;
          // distinct-index quick reject would require a full pairwise check; defer to recovery (final).

          uint64_t mw[WORK_WORDS];
          merge_workbits(aw, bw, mergeLen, mw);

          const uint32_t oslot = dev_atomic_u32(d_out_count[0]).fetch_add(1u);
          if (oslot >= MIXED_CAP) continue;

          // Build the merged first-NIDX[OUT_L] index tree from the two parents' stored prefixes in
          // lowest-index-first order. We have up to NIDX[L] per parent; the merged tree's first
          // NIDX[OUT_L] (<= 2*NIDX[L]) come from the leader's stored list then the follower's.
          uint32_t prefix[LEVEL_NIDX[OUT_L] > 0 ? LEVEL_NIDX[OUT_L] : 1];
          if (LEVEL_NIDX[OUT_L] > 0) {
            const uint32_t* lead = (a_lead < b_lead) ? a : b;
            const uint32_t* foll = (a_lead < b_lead) ? b : a;
            const unsigned base = 2 * BUCKET_WORDS[L];
            unsigned o_i = 0;
            for (unsigned i = 0; i < BUCKET_NIDX[L] && o_i < LEVEL_NIDX[OUT_L]; ++i)
              prefix[o_i++] = lead[base + i] & INDEX_MASK;
            for (unsigned i = 0; i < BUCKET_NIDX[L] && o_i < LEVEL_NIDX[OUT_L]; ++i)
              prefix[o_i++] = foll[base + i] & INDEX_MASK;
          }

          // A fused bucket record's source is this round's linear oslot, so the existing recovery
          // walk through tree[5..1] remains unchanged.
          const uint32_t a_src = bucket_load_source<L>(a);
          const uint32_t b_src = bucket_load_source<L>(b);
          uint32_t* tl = d_tree + (size_t)oslot * 2;
          tl[0] = (a_lead < b_lead) ? a_src : b_src;
          tl[1] = (a_lead < b_lead) ? b_src : a_src;

          if constexpr (FUSED && R < NUM_ROUNDS) {
            // Exactly the transform formerly performed by submit_mix_scatter<R+1>.
            apply_mix(mw, prefix, LEVEL_NIDX[OUT_L], ROUND_REMLEN[R + 1]);
            const uint32_t next_coll = collision_bits(mw);
            const uint32_t next_bucket = next_coll >> RESTBITS;
            const uint32_t next_slot = dev_atomic_u32(d_next_bucket_ns[next_bucket]).fetch_add(1u);
            if (next_slot >= NSLOTS) continue;
            uint32_t* o = d_next_bucket + ((size_t)next_bucket * NSLOTS + next_slot) * NEXT_ST;
            for (unsigned i = 0; i < BUCKET_WORDS[OUT_L]; ++i) {
              o[2 * i]     = (uint32_t)(mw[i] & 0xffffffffull);
              o[2 * i + 1] = (uint32_t)(mw[i] >> 32);
            }
            for (unsigned i = 0; i < BUCKET_NIDX[OUT_L]; ++i)
              o[2 * BUCKET_WORDS[OUT_L] + i] = prefix[i];
            bucket_store_source<OUT_L>(o, oslot);
          } else {
            uint32_t* o = d_final_out + (size_t)oslot * FINAL_ST;
            for (unsigned i = 0; i < LEVEL_WORDS[OUT_L]; ++i) {
              o[2 * i]     = (uint32_t)(mw[i] & 0xffffffffull);
              o[2 * i + 1] = (uint32_t)(mw[i] >> 32);
            }
            // The non-fused CUDA path feeds this linear record into the existing coalesced scatter.
            // Round 5 has no prefix, so the same loop compiles away for the fused final output.
            for (unsigned i = 0; i < LEVEL_NIDX[OUT_L]; ++i)
              o[2 * LEVEL_WORDS[OUT_L] + i] = prefix[i];
          }
        }
      }
    });
  });
}

// ---- Final: a level-5 survivor whose merged 24-bit workBits == 0 is a candidate. Walk tree[5..1] to
// its 32 leaf source indices (in mixed[0]), read each leaf's stored leaf index. ----
// Walk the tree from a level-OUT survivor down to its leaf indices. tree[L] maps a mixed[L] slot to its
// two parent mixed[L-1] slots. mixed[0] slot stores the leaf index at offset 2*LEVEL_WORDS[0].
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

// ---- M1 gen-validation: dump the first BEAMHASH3_TEST_ROWS raw leaf rows (7 LE u64 words) on-device. ----
class BeamGenTestKernel;
static sycl::event submit_gen_test(sycl::queue& q, MomKernelBundle& kb, const uint64_t* d_prePow,
                                   uint8_t* d_rows) {
  constexpr unsigned WG = 64;
  const size_t global = (BEAMHASH3_TEST_ROWS + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    h.parallel_for<BeamGenTestKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
      const uint32_t e = (uint32_t)it.get_global_id(0);
      if (e >= BEAMHASH3_TEST_ROWS) return;
      uint64_t prePow[4];
      for (unsigned i = 0; i < 4; ++i) prePow[i] = d_prePow[i];
      uint64_t w[WORK_WORDS];
      make_leaf(prePow, e, w);
      uint8_t* row = d_rows + (size_t)e * ROW_BYTES;
      for (unsigned i = 0; i < WORK_WORDS; ++i)
        for (unsigned b = 0; b < 8; ++b)
          row[i * 8 + b] = (uint8_t)((w[i] >> (8 * b)) & 0xff);
    });
  });
}
