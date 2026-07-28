// BeamHash III Wagner solver kernels and per-device state. Included into beamhash3.cpp inside
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

unsigned beamhash3_workgroup(const sycl::device& device) {
  if (beam_compact_enabled(device)) {
    const unsigned preferred = mom_is_hip(device) ? 512 : 256;
    unsigned result = sycl_default_workgroup(device, {128, 256, 512}, preferred);
    unsigned long parsed = 0;
    if (mom_parse_env_ulong("MOM_BEAMHASH3_COMPACT_WG", parsed) && parsed >= 16)
      result = static_cast<unsigned>(std::min<unsigned long>(parsed, 512));
    return result;
  }

  unsigned preferred = sycl_is_level_zero_gpu(device) ? 1024 : 640;
#if !defined(MOM_SYCL_ADAPTIVECPP)
  if (mom_is_cuda(device)) preferred = 768;
#endif
  unsigned result = sycl_default_workgroup(
    device, {128, 256, 384, 512, 640, 768, 1024}, preferred);
  unsigned long parsed = 0;
  if (mom_parse_env_ulong("MOM_BEAMHASH3_WORKGROUP", parsed) && parsed >= 16)
    result = static_cast<unsigned>(std::min<unsigned long>(parsed, 1024));
  return result;
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
