// BeamHash III Wagner solver kernels + per-device state. Included into beamhash3.cpp inside
// namespace mom_beamhash3 (M2/M3).
//
// Pipeline (5 rounds, k=5):
//   gen      : 2^25 leaves -> linear list mixed[0] (workBits 7 words + leaf index).
//   round r  : mix-scatter mixed[r-1] -> bucketed[r-1] (applyMix(remLen), bucket by hi BUCKBITS of the
//              low-24 collision field, carry the source linear index); collide per bucket in SLM (low
//              RESTBITS), XOR-merge -> append to mixed[r] (workBits + first NIDX[r] indices), record the
//              two parent linear indices in tree[r].
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
// the u64 work words as 2 u32 each (lo, hi). Stride in u32 = 2*WORDS + NIDX.
constexpr unsigned mixed_stride_u32(unsigned L) { return 2 * LEVEL_WORDS[L] + LEVEL_NIDX[L]; }
// A "bucketed" slot stores the SAME plus a trailing u32 src = source linear index in mixed[L].
constexpr unsigned bucket_stride_u32(unsigned L) { return mixed_stride_u32(L) + 1; }

// Widest strides (for the ping-pong scratch + the reused bucket arena).
constexpr unsigned max_mixed_stride() {
  unsigned m = 0; for (unsigned L = 1; L <= NUM_ROUNDS; ++L) m = std::max(m, mixed_stride_u32(L)); return m;
}
constexpr unsigned max_bucket_stride() {
  unsigned m = 0; for (unsigned L = 0; L < NUM_ROUNDS; ++L) m = std::max(m, bucket_stride_u32(L)); return m;
}
constexpr unsigned MAX_MIXED_STRIDE  = max_mixed_stride();
constexpr unsigned MAX_BUCKET_STRIDE = max_bucket_stride();

// A round can emit slightly MORE survivors than its input (a rest-bin with c slots yields ~c*(c-1)/2
// pairs, so the expected survivor count ~ input but the tail fluctuates above 2^25). Over-provision the
// linear mixed/scratch arenas by ~12% so no survivor (and so no solution) is dropped to the cap. mixed0
// holds exactly 2^25 leaves; the round outputs go to the wider scratch buffers.
// Measured per-round survivor counts hug 2^25 to within +0.006% (e.g. 33,556,370 vs 33,554,432). The
// over-provision absorbs the input-dependent tail; 1/32 (~+1.05M slots) is ~170x the observed excess.
constexpr uint64_t MIXED_CAP = (uint64_t)(NUM_ENTRIES + NUM_ENTRIES / 32);  // 2^25 * 1.03125

// ---- gen is FUSED into round-1 mix-scatter (see submit_mix_scatter, R==1): the leaf at slot==index is
// regenerated in-kernel via make_leaf, so no separate gen kernel and no level-0 DRAM arena. This elides
// a full level-0 write (~2 GiB) plus the round-1 read of it; scatter is DRAM-write bound so the extra
// siphash compute is free. Recovery still maps leaf index == level-0 slot directly. ----

// ---- Round r mix-scatter: read mixed[L] (L=r-1), applyMix(remLen), bucket by hi BUCKBITS of the low-24
// collision field, write a bucketed slot carrying the source linear index. ----
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
template <unsigned R> class BeamMixScatterKernel;
template <unsigned R>
static sycl::event submit_mix_scatter(sycl::queue& q, MOM_BUNDLE_T& kb, const uint32_t* d_mixed,
                                      uint32_t in_count, uint32_t* d_bucket, uint32_t* d_bucket_ns,
                                      unsigned WG, const uint64_t* d_prePow = nullptr) {
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
    MOM_USE_BUNDLE(h, kb);
    // Per-WG staging: each lane's OUT_ST-u32 record + its packed (bucket,slot) destination.
    sycl::local_accessor<uint32_t, 1> stage(sycl::range<1>((size_t)WG * OUT_ST), h);
    sycl::local_accessor<uint32_t, 1> dst(sycl::range<1>(WG), h);
    h.parallel_for<BeamMixScatterKernel<R>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
      const uint32_t s   = (uint32_t)it.get_global_id(0);
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);
      uint32_t* st = &stage[(size_t)lid * OUT_ST];

      if (s < in_count) {
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
          for (unsigned i = 0; i < LEVEL_WORDS[L]; ++i) {
            st[2 * i]     = (uint32_t)(w[i] & 0xffffffffull);
            st[2 * i + 1] = (uint32_t)(w[i] >> 32);
          }
          for (unsigned i = 0; i < LEVEL_NIDX[L]; ++i) st[2 * LEVEL_WORDS[L] + i] = tree[i];
          st[OUT_ST - 1] = s;   // source linear index in mixed[L]
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

// ---- Round r collide: one WG per bucket. Read bucketed[L], collide low RESTBITS in SLM, XOR-merge
// (mergeLen), build the next index tree (first NIDX[L+1] indices), append to mixed[L+1], log parents. ----
template <unsigned R> class BeamCollideKernel;
template <unsigned R>
static sycl::event submit_collide(sycl::queue& q, MOM_BUNDLE_T& kb, const uint32_t* d_bucket,
                                  const uint32_t* d_bucket_ns, uint32_t* d_mixed_out,
                                  uint32_t* d_out_count, uint32_t* d_tree, unsigned wg) {
  constexpr unsigned L = R - 1;
  constexpr unsigned IN_ST   = bucket_stride_u32(L);
  constexpr unsigned OUT_L   = R;           // output level
  constexpr unsigned OUT_ST  = mixed_stride_u32(OUT_L);
  constexpr unsigned mergeLen = ROUND_MERGELEN[R];
  const size_t global = (size_t)NBUCKETS * wg;
  return q.submit([&](sycl::handler& h) {
    MOM_USE_BUNDLE(h, kb);
    sycl::local_accessor<uint32_t, 1> bin_head(sycl::range<1>(NRESTBINS), h);
    sycl::local_accessor<uint32_t, 1> bin_next(sycl::range<1>(NSLOTS), h);
    // Stage every slot's 24-bit collision key in SLM. The wide bucket records are read once here (the
    // inverse of the scatter-transpose); the rest-bin build and -- crucially -- the per-link compare in
    // the collide pass then hit SLM instead of re-striding the wide (60..84B) record in DRAM. Most link
    // traversals fail the full-24-bit match, so this removes the dominant scattered b[0],b[1] DRAM reads.
    sycl::local_accessor<uint32_t, 1> coll_key(sycl::range<1>(NSLOTS), h);
    h.parallel_for<BeamCollideKernel<R>>(
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

        for (uint32_t link = bin_head[rb]; link != 0; link = bin_next[link - 1u]) {
          const uint32_t t = link - 1u;
          if (t >= s) continue;                                  // emit each unordered pair once
          if (coll_key[t] != a_coll) continue;                   // full 24-bit collision field must match (SLM)

          if (!a) {   // lazy: only now do we touch the wide records from DRAM.
            a = in_base + (size_t)s * IN_ST;
            for (unsigned i = 0; i < WORK_WORDS; ++i) aw[i] = 0;
            for (unsigned i = 0; i < LEVEL_WORDS[L]; ++i)
              aw[i] = (uint64_t)a[2 * i] | ((uint64_t)a[2 * i + 1] << 32);
          }
          const uint32_t* b = in_base + (size_t)t * IN_ST;
          uint64_t bw[WORK_WORDS]; for (unsigned i = 0; i < WORK_WORDS; ++i) bw[i] = 0;
          for (unsigned i = 0; i < LEVEL_WORDS[L]; ++i)
            bw[i] = (uint64_t)b[2 * i] | ((uint64_t)b[2 * i + 1] << 32);

          // The two parents' first-NIDX[L] index lists + lead index decide the merge order
          // (lowest-index-first). Leader = the node whose lead index (tree[0]) is smaller.
          const uint32_t a_lead = a[2 * LEVEL_WORDS[L] + 0];
          const uint32_t b_lead = b[2 * LEVEL_WORDS[L] + 0];
          // distinct-index quick reject would require a full pairwise check; defer to recovery (final).

          uint64_t mw[WORK_WORDS];
          merge_workbits(aw, bw, mergeLen, mw);

          const uint32_t oslot = dev_atomic_u32(d_out_count[0]).fetch_add(1u);
          if (oslot >= MIXED_CAP) continue;
          uint32_t* o = d_mixed_out + (size_t)oslot * OUT_ST;
          for (unsigned i = 0; i < LEVEL_WORDS[OUT_L]; ++i) {
            o[2 * i]     = (uint32_t)(mw[i] & 0xffffffffull);
            o[2 * i + 1] = (uint32_t)(mw[i] >> 32);
          }
          // Build the merged first-NIDX[OUT_L] index tree from the two parents' stored prefixes in
          // lowest-index-first order. We have up to NIDX[L] per parent; the merged tree's first
          // NIDX[OUT_L] (<= 2*NIDX[L]) come from the leader's stored list then the follower's.
          if (LEVEL_NIDX[OUT_L] > 0) {
            const uint32_t* lead = (a_lead < b_lead) ? a : b;
            const uint32_t* foll = (a_lead < b_lead) ? b : a;
            const unsigned base = 2 * LEVEL_WORDS[L];
            unsigned o_i = 0;
            for (unsigned i = 0; i < LEVEL_NIDX[L] && o_i < LEVEL_NIDX[OUT_L]; ++i)
              o[2 * LEVEL_WORDS[OUT_L] + (o_i++)] = lead[base + i];
            for (unsigned i = 0; i < LEVEL_NIDX[L] && o_i < LEVEL_NIDX[OUT_L]; ++i)
              o[2 * LEVEL_WORDS[OUT_L] + (o_i++)] = foll[base + i];
          }
          // Tree log: the two parent SOURCE linear indices (in mixed[L]) for recovery.
          const uint32_t a_src = a[IN_ST - 1], b_src = b[IN_ST - 1];
          uint32_t* tl = d_tree + (size_t)oslot * 2;
          tl[0] = (a_lead < b_lead) ? a_src : b_src;   // leader first (lowest-index-first order)
          tl[1] = (a_lead < b_lead) ? b_src : a_src;
        }
      }
    });
  });
}

// ---- Final: a level-5 survivor whose merged 24-bit workBits == 0 is a candidate. Walk tree[5..1] to
// its 32 leaf source indices (in mixed[0]), read each leaf's stored leaf index. ----
struct BeamCandidate { uint32_t leaves[NUM_LEAVES]; };

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
static sycl::event submit_final(sycl::queue& q, MOM_BUNDLE_T& kb, const uint32_t* d_mixed5,
                                uint32_t in_count, BeamLevels lv, BeamCandidate* d_cand,
                                uint32_t* d_cand_count, uint32_t cand_cap) {
  constexpr unsigned IN_ST = mixed_stride_u32(NUM_ROUNDS);   // level-5 stride (1 word + 0 idx) = 2
  constexpr unsigned WG = 128;
  const size_t global = ((size_t)in_count + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    MOM_USE_BUNDLE(h, kb);
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
  sycl::device device; sycl::queue queue; std::unique_ptr<MOM_BUNDLE_T> bundle; bool shared_io;
  // M1 I/O
  uint64_t* prePow = nullptr;   // 4 u64 (the SipHash key / IndividualWork)
  uint8_t*  rows = nullptr;     // BEAMHASH3_TEST_ROWS * ROW_BYTES (raw 448-bit leaf rows)
  // Wagner arenas. ALL levels (0..5) are transient and share ONE scratch buffer (leaf index == level-0
  // slot, so recovery needs no persistent mixed[0]; collide never reads its input mixed buffer, so each
  // level may overwrite the previous in place). One bucket arena (widest bucket stride) reused per round.
  uint32_t* scratch[1] = {};      // shared mixed[0..5]: MIXED_CAP * MAX_MIXED_STRIDE
  uint32_t* bucket   = nullptr;   // NBUCKETS*NSLOTS * MAX_BUCKET_STRIDE (reused each round)
  uint32_t* bucket_ns             = nullptr; // NBUCKETS counters (reused per round)
  uint32_t* mixed_count           = nullptr; // shared: per-round output counter (NUM_ROUNDS+1 entries)
  uint32_t* tree[NUM_ROUNDS + 1]  = {};   // tree[L] (L=1..5): NUM_ENTRIES * 2
  BeamCandidate* cand             = nullptr;
  uint32_t* cand_count            = nullptr;
  // mixed[L] resolved to a concrete buffer per round (entrypoint sets these to mixed0/scratch[]).
  static constexpr uint32_t CAND_CAP = 8192;
  bool arenas_built = false; std::mutex mutex;

  explicit BeamState(const std::string& dev_str)
    : device(get_dev(dev_str)), queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !device.has(sycl::aspect::usm_device_allocations)) {
    if (!device.has(sycl::aspect::usm_shared_allocations) ||
        (!device.is_cpu() && !device.has(sycl::aspect::usm_device_allocations)))
      throw std::string("beamhash3 SYCL device does not support required allocations");
    bundle = std::make_unique<MOM_BUNDLE_T>(MOM_GET_EXEC_BUNDLE(queue.get_context()));
  }
  ~BeamState() { queue.wait_and_throw(); free_all(); }
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
    { // Single shared transient scratch for levels 0..5. Level 0 needs NUM_ENTRIES*mixed_stride(0)
      // u32; later levels need MIXED_CAP*mixed_stride(L) -- MIXED_CAP*MAX_MIXED_STRIDE covers both.
      // Recovery never reads it (leaf index == level-0 slot); collide does not read its input mixed
      // buffer, so each level overwrites the previous in place -- no ping-pong / no persistent mixed0.
      const size_t n = (size_t)MIXED_CAP * MAX_MIXED_STRIDE;
      scratch[0] = alloc<uint32_t>(n); total += n * 4;
      if (!scratch[0]) throw std::string("Can't allocate beamhash3 scratch arena");
    }
    for (unsigned L = 1; L <= NUM_ROUNDS; ++L) {
      const size_t tn = (size_t)MIXED_CAP * 2;
      tree[L] = alloc<uint32_t>(tn); total += tn * 4;
      if (!tree[L]) throw std::string("Can't allocate beamhash3 tree log");
    }
    { const size_t n = (size_t)NBUCKETS * NSLOTS * MAX_BUCKET_STRIDE;
      bucket = alloc<uint32_t>(n); total += n * 4;
      if (!bucket) throw std::string("Can't allocate beamhash3 bucket arena"); }
    bucket_ns   = alloc<uint32_t>(NBUCKETS); total += (size_t)NBUCKETS * 4;
    mixed_count = sycl::malloc_shared<uint32_t>(NUM_ROUNDS + 1, queue);
    cand        = alloc<BeamCandidate>(CAND_CAP);
    cand_count  = sycl::malloc_shared<uint32_t>(1, queue);
    if (!bucket_ns || !mixed_count || !cand || !cand_count)
      throw std::string("Can't allocate beamhash3 counters/candidates");
    arenas_built = true;
    if (log) { std::fprintf(stderr, "beamhash3 Wagner arenas (~%.1f GiB) allocated (%llu ms)\n",
      (double)total / (1024.0 * 1024.0 * 1024.0), (unsigned long long)(now_ms() - t0)); std::fflush(stderr); }
  }
  void free_all_arenas() {
    free_ptr(scratch[0]);
    for (unsigned L = 0; L <= NUM_ROUNDS; ++L) free_ptr(tree[L]);
    free_ptr(bucket); free_ptr(bucket_ns); free_ptr(mixed_count); free_ptr(cand); free_ptr(cand_count);
    arenas_built = false;
  }
  static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }
};

static BeamState& beam_state(const std::string& dev_str) {
  static std::mutex m; static std::map<std::string, std::unique_ptr<BeamState>> states;
  std::lock_guard<std::mutex> lock(m);
  auto& s = states[dev_str]; if (!s) s = std::make_unique<BeamState>(dev_str); return *s;
}

// ---- M1 gen-validation: dump the first BEAMHASH3_TEST_ROWS raw leaf rows (7 LE u64 words) on-device. ----
class BeamGenTestKernel;
static sycl::event submit_gen_test(sycl::queue& q, MOM_BUNDLE_T& kb, const uint64_t* d_prePow,
                                   uint8_t* d_rows) {
  constexpr unsigned WG = 64;
  const size_t global = (BEAMHASH3_TEST_ROWS + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    MOM_USE_BUNDLE(h, kb);
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
