#if !defined(MOM_SYCL_PORTABLE_OPENCL)
#if defined(MOM_BEAMHASH3_ESIMD)
namespace beamhash3_esimd {
namespace esimd = sycl::ext::intel::esimd;
using U32 = esimd::simd<uint32_t, 16>;
using Mask = esimd::simd_mask<16>;

struct U64Pair {
  U32 lo;
  U32 hi;
};

SYCL_ESIMD_FUNCTION inline U32 mask_to_u32(Mask mask) {
  U32 result(0u);
  result.merge(U32(1u), mask);
  return result;
}

SYCL_ESIMD_FUNCTION inline U64Pair add(U64Pair a, U64Pair b) {
  const U32 lo = a.lo + b.lo;
  const U32 carry = mask_to_u32(lo < a.lo);
  return {lo, a.hi + b.hi + carry};
}

SYCL_ESIMD_FUNCTION inline U64Pair bit_xor(U64Pair a, U64Pair b) {
  return {a.lo ^ b.lo, a.hi ^ b.hi};
}

template <unsigned Bits>
SYCL_ESIMD_FUNCTION inline U64Pair rotl(U64Pair value) {
  static_assert(Bits > 0 && Bits < 64);
  if constexpr (Bits < 32) {
    return {(value.lo << Bits) | (value.hi >> (32 - Bits)),
            (value.hi << Bits) | (value.lo >> (32 - Bits))};
  } else if constexpr (Bits == 32) {
    return {value.hi, value.lo};
  } else {
    constexpr unsigned Shift = Bits - 32;
    return {(value.hi << Shift) | (value.lo >> (32 - Shift)),
            (value.lo << Shift) | (value.hi >> (32 - Shift))};
  }
}

template <unsigned Offset>
SYCL_ESIMD_FUNCTION inline U64Pair nonce(U32 index) {
  const U32 base_lo = index << 3;
  const U32 base_hi = index >> 29;
  const U32 nonce_lo = base_lo + U32(Offset);
  const U32 carry = mask_to_u32(nonce_lo < base_lo);
  return {nonce_lo, base_hi + carry};
}

SYCL_ESIMD_FUNCTION inline void sip_round(U64Pair& v0, U64Pair& v1, U64Pair& v2,
                                          U64Pair& v3) {
  v0 = add(v0, v1);
  v2 = add(v2, v3);
  v1 = rotl<13>(v1);
  v3 = rotl<16>(v3);
  v1 = bit_xor(v1, v0);
  v3 = bit_xor(v3, v2);
  v0 = rotl<32>(v0);
  v2 = add(v2, v1);
  v0 = add(v0, v3);
  v1 = rotl<17>(v1);
  v3 = rotl<21>(v3);
  v1 = bit_xor(v1, v2);
  v3 = bit_xor(v3, v0);
  v2 = rotl<32>(v2);
}

SYCL_ESIMD_FUNCTION inline U64Pair siphash24(U64Pair s0, U64Pair s1, U64Pair s2,
                                             U64Pair s3, U64Pair msg) {
  U64Pair v0 = s0, v1 = s1, v2 = s2, v3 = s3;
  v3 = bit_xor(v3, msg);
  sip_round(v0, v1, v2, v3);
  sip_round(v0, v1, v2, v3);
  v0 = bit_xor(v0, msg);
  v2.lo ^= 0xffu;
  sip_round(v0, v1, v2, v3);
  sip_round(v0, v1, v2, v3);
  sip_round(v0, v1, v2, v3);
  sip_round(v0, v1, v2, v3);
  return bit_xor(bit_xor(v0, v1), bit_xor(v2, v3));
}

SYCL_ESIMD_FUNCTION inline void store_rows(uint32_t* output, U32 buckets, U32 slots,
                                            U32 indices, U64Pair w0, U64Pair w1,
                                            U64Pair w2, U64Pair w3, U64Pair w4,
                                            U64Pair w5, U64Pair w6) {
  constexpr unsigned ROW_U32 = sizeof(BeamCompactRow) / sizeof(uint32_t);
#pragma unroll
  for (unsigned lane = 0; lane < 16; ++lane) {
    const uint32_t slot = slots[lane];
    if (slot >= COMPACT_EARLY_NSLOTS)
      continue;
    const uint32_t dst = buckets[lane] * COMPACT_EARLY_NSLOTS + slot;
    U32 row(0u);
    row[0] = w0.lo[lane];
    row[1] = w0.hi[lane];
    row[2] = w1.lo[lane];
    row[3] = w1.hi[lane];
    row[4] = w2.lo[lane];
    row[5] = w2.hi[lane];
    row[6] = w3.lo[lane];
    row[7] = w3.hi[lane];
    row[8] = w4.lo[lane];
    row[9] = w4.hi[lane];
    row[10] = w5.lo[lane];
    row[11] = w5.hi[lane];
    row[12] = w6.lo[lane];
    row[13] = w6.hi[lane];
    row[14] = indices[lane];
    row[15] = 0u;
    esimd::block_store<uint32_t, ROW_U32>(output + (size_t)dst * ROW_U32, row);
  }
}
} // namespace beamhash3_esimd

class BeamCompactSeedEsimdKernel;
static sycl::event submit_compact_seed_esimd(sycl::queue& q, const uint64_t* d_prePow,
                                             BeamCompactRow* d_out, uint32_t* d_out_ns) {
  static_assert(NUM_ENTRIES % 16 == 0, "ESIMD seed requires 16 leaves per work-item");
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<BeamCompactSeedEsimdKernel>(
        sycl::range<1>((size_t)NUM_ENTRIES / 16),
        [=](sycl::id<1> id) SYCL_ESIMD_KERNEL MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const uint32_t base = (uint32_t)id[0] * 16u;
          using namespace beamhash3_esimd;
          const auto key = esimd::block_load<uint32_t, 8>(
              reinterpret_cast<const uint32_t*>(d_prePow));
          const U64Pair s0{U32(key[0]), U32(key[1])};
          const U64Pair s1{U32(key[2]), U32(key[3])};
          const U64Pair s2{U32(key[4]), U32(key[5])};
          const U64Pair s3{U32(key[6]), U32(key[7])};
          const U32 indices(base, 1u);
          const U64Pair w0 = siphash24(s0, s1, s2, s3, nonce<0>(indices));
          U64Pair sum = rotl<29>(w0);
          const U64Pair w1 = siphash24(s0, s1, s2, s3, nonce<1>(indices));
          sum = add(sum, rotl<58>(w1));
          const U64Pair w2 = siphash24(s0, s1, s2, s3, nonce<2>(indices));
          sum = add(sum, rotl<23>(w2));
          const U64Pair w3 = siphash24(s0, s1, s2, s3, nonce<3>(indices));
          sum = add(sum, rotl<52>(w3));
          const U64Pair w4 = siphash24(s0, s1, s2, s3, nonce<4>(indices));
          sum = add(sum, rotl<17>(w4));
          const U64Pair w5 = siphash24(s0, s1, s2, s3, nonce<5>(indices));
          sum = add(sum, rotl<46>(w5));
          const U64Pair w6 = siphash24(s0, s1, s2, s3, nonce<6>(indices));
          sum = add(sum, rotl<11>(w6));
          sum = add(sum, rotl<40>(U64Pair{indices, U32(0u)}));
          const U64Pair mixed = rotl<24>(sum);
          const U32 buckets = mixed.lo & U32(compact_nbuckets(0) - 1u);
          const U32 byte_offsets = buckets << 2;
          const U32 slots = esimd::atomic_update<esimd::atomic_op::add>(
              d_out_ns, byte_offsets, U32(1u), esimd::simd_mask<16>(1));
          store_rows(reinterpret_cast<uint32_t*>(d_out), buckets, slots, indices, mixed, w1, w2,
                     w3, w4, w5, w6);
        });
  });
}
#endif

class BeamCompactSeedKernel;
static sycl::event submit_compact_seed(sycl::queue& q, const uint64_t* d_prePow,
                                       BeamCompactRow* d_out,
                                       uint32_t* d_out_ns, unsigned wg) {
  const size_t global = ((size_t)NUM_ENTRIES + wg - 1) / wg * wg;
  return q.submit([&](sycl::handler& h) {
    constexpr uint32_t SENTINEL = 0xffffffffu;
    constexpr unsigned ROW_U32 = sizeof(BeamCompactRow) / sizeof(uint32_t);
    sycl::local_accessor<uint32_t, 1> stage(sycl::range<1>((size_t)wg * ROW_U32), h);
    sycl::local_accessor<uint32_t, 1> destination(sycl::range<1>(wg), h);
    // Xe2 otherwise emits this seed as SIMD32. SIMD16 cuts its generated instruction count from
    // 1,096 to 651 and raised B580 solve-loop throughput from 85 to 86-87 iterations per 10 seconds.
    // The macro leaves NVPTX, AdaptiveCpp/HIP and portable OpenCL at their native subgroup width.
    h.parallel_for<BeamCompactSeedKernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) MOM_REQD_SG_16 MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const uint32_t index = (uint32_t)it.get_global_id(0);
          const unsigned lid = (unsigned)it.get_local_id(0);
          const unsigned lsz = (unsigned)it.get_local_range(0);
          if (index < NUM_ENTRIES) {
#if defined(MOM_BEAMHASH3_INTEL_PAIR32)
            // Stream named leaf pairs to SLM. An indexed seven-word aggregate made IGC allocate a
            // 896-byte private buffer and cost about 1% of B580 throughput.
            const BeamU64Pair s0 = beam_pair_from_u64(d_prePow[0]);
            const BeamU64Pair s1 = beam_pair_from_u64(d_prePow[1]);
            const BeamU64Pair s2 = beam_pair_from_u64(d_prePow[2]);
            const BeamU64Pair s3 = beam_pair_from_u64(d_prePow[3]);
            BeamU64Pair w0 = siphash24_pair(s0, s1, s2, s3, beam_pair_nonce(index, 0));
            BeamU64Pair sum = beam_pair_rotl<29>(w0);
            const BeamU64Pair w1 = siphash24_pair(s0, s1, s2, s3, beam_pair_nonce(index, 1));
            sum = beam_pair_add(sum, beam_pair_rotl<58>(w1));
            stage[(size_t)lid * ROW_U32 + 2] = w1.lo;
            stage[(size_t)lid * ROW_U32 + 3] = w1.hi;
            const BeamU64Pair w2 = siphash24_pair(s0, s1, s2, s3, beam_pair_nonce(index, 2));
            sum = beam_pair_add(sum, beam_pair_rotl<23>(w2));
            stage[(size_t)lid * ROW_U32 + 4] = w2.lo;
            stage[(size_t)lid * ROW_U32 + 5] = w2.hi;
            const BeamU64Pair w3 = siphash24_pair(s0, s1, s2, s3, beam_pair_nonce(index, 3));
            sum = beam_pair_add(sum, beam_pair_rotl<52>(w3));
            stage[(size_t)lid * ROW_U32 + 6] = w3.lo;
            stage[(size_t)lid * ROW_U32 + 7] = w3.hi;
            const BeamU64Pair w4 = siphash24_pair(s0, s1, s2, s3, beam_pair_nonce(index, 4));
            sum = beam_pair_add(sum, beam_pair_rotl<17>(w4));
            stage[(size_t)lid * ROW_U32 + 8] = w4.lo;
            stage[(size_t)lid * ROW_U32 + 9] = w4.hi;
            const BeamU64Pair w5 = siphash24_pair(s0, s1, s2, s3, beam_pair_nonce(index, 5));
            sum = beam_pair_add(sum, beam_pair_rotl<46>(w5));
            stage[(size_t)lid * ROW_U32 + 10] = w5.lo;
            stage[(size_t)lid * ROW_U32 + 11] = w5.hi;
            const BeamU64Pair w6 = siphash24_pair(s0, s1, s2, s3, beam_pair_nonce(index, 6));
            sum = beam_pair_add(sum, beam_pair_rotl<11>(w6));
            stage[(size_t)lid * ROW_U32 + 12] = w6.lo;
            stage[(size_t)lid * ROW_U32 + 13] = w6.hi;
            sum = beam_pair_add(sum, beam_pair_rotl<40>(BeamU64Pair{index, 0}));
            w0 = beam_pair_rotl<24>(sum);
            stage[(size_t)lid * ROW_U32] = w0.lo;
            stage[(size_t)lid * ROW_U32 + 1] = w0.hi;
            stage[(size_t)lid * ROW_U32 + 14] = index;
            stage[(size_t)lid * ROW_U32 + 15] = 0;
            const uint32_t bucket = w0.lo & (compact_nbuckets(0) - 1u);
#else
            uint64_t prePow[4];
            for (unsigned i = 0; i < 4; ++i)
              prePow[i] = d_prePow[i];
            uint64_t work[WORK_WORDS];
            make_leaf(prePow, index, work);
            uint32_t tree[1] = {index};
            apply_mix(work, tree, 1, ROUND_REMLEN[1]);
            const BeamCompactRow row = compact_make_row<0>(work, tree);
            const uint32_t bucket = (uint32_t)work[0] & (compact_nbuckets(0) - 1u);
#endif
            const uint32_t slot = dev_atomic_u32(d_out_ns[bucket]).fetch_add(1u);
            if (slot < compact_nslots(0)) {
#if !defined(MOM_BEAMHASH3_INTEL_PAIR32)
              const uint32_t* src = reinterpret_cast<const uint32_t*>(&row);
              for (unsigned k = 0; k < ROW_U32; ++k)
                stage[(size_t)lid * ROW_U32 + k] = src[k];
#endif
              destination[lid] = bucket * compact_nslots(0) + slot;
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
          const unsigned sg_size = (unsigned)sg.get_local_range()[0];
          const unsigned lane = (unsigned)sg.get_local_id()[0];
          const unsigned subgroups = (unsigned)sg.get_group_range()[0];
          const unsigned subgroup = (unsigned)sg.get_group_id()[0];
          uint32_t* output = reinterpret_cast<uint32_t*>(d_out);
          for (unsigned r = subgroup; r < lsz; r += subgroups) {
            const uint32_t dst = destination[r];
            if (dst == SENTINEL)
              continue;
            for (unsigned k = lane; k < ROW_U32; k += sg_size)
              output[(size_t)dst * ROW_U32 + k] = stage[(size_t)r * ROW_U32 + k];
          }
        });
  });
}

template <unsigned R, bool PairPartitions = false> class BeamCompactRoundKernel;
template <unsigned R, bool PairPartitions = false>
static sycl::event submit_compact_round(sycl::queue& q, const BeamCompactRow* d_in,
                                        const uint32_t* d_in_ns,
                                        BeamCompactRow* d_out, uint32_t* d_out_ns,
                                        BeamCandidate* d_cand, uint32_t* d_cand_count,
                                        uint32_t cand_cap, unsigned wg) {
  static_assert(R >= 1 && R <= NUM_ROUNDS);
  static_assert(!PairPartitions || R <= 4);
  constexpr unsigned L = R - 1;
  constexpr unsigned IN_TREE = compact_tree_len(L);
  constexpr unsigned OUT_TREE = compact_tree_len(R);
  constexpr unsigned mergeLen = ROUND_MERGELEN[R];
  constexpr unsigned IN_BUCKET_BITS = compact_bucket_bits(L);
  constexpr unsigned IN_NBUCKETS = compact_nbuckets(L);
  constexpr unsigned IN_SUBGROUP_BITS = compact_subgroup_bits(L);
  constexpr unsigned IN_SUBGROUPS = compact_subgroups(L);
  constexpr unsigned MERGED_SUBGROUP_BITS = PairPartitions ? 1u : 0u;
  constexpr unsigned GROUP_SUBGROUP_BITS = IN_SUBGROUP_BITS - MERGED_SUBGROUP_BITS;
  constexpr unsigned GROUP_SUBGROUPS = 1u << GROUP_SUBGROUP_BITS;
  constexpr unsigned BIN_COUNT = PairPartitions ? COMPACT_PAIRED_NRESTBINS : COMPACT_NRESTBINS;
  constexpr unsigned LOCAL_CAP = PairPartitions ? COMPACT_PAIRED_LOCAL_CAP : COMPACT_LOCAL_CAP;
  constexpr unsigned IN_NSLOTS = compact_nslots(L);
  using local_index_t = std::conditional_t<PairPartitions, uint16_t, uint32_t>;
  if constexpr (PairPartitions)
    static_assert(IN_NSLOTS <= UINT16_MAX && COMPACT_PAIRED_LOCAL_CAP <= UINT16_MAX,
                  "Compact paired slot indices must fit uint16_t");
  constexpr unsigned OUT_NBUCKETS = compact_nbuckets(R);
  constexpr unsigned OUT_NSLOTS = compact_nslots(R);
  const size_t global = (size_t)IN_NBUCKETS * GROUP_SUBGROUPS * wg;
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> bin_head(sycl::range<1>(BIN_COUNT), h);
    sycl::local_accessor<local_index_t, 1> bin_next(sycl::range<1>(LOCAL_CAP), h);
    sycl::local_accessor<local_index_t, 1> slot_index(sycl::range<1>(LOCAL_CAP), h);
    sycl::local_accessor<uint32_t, 1> local_count(sycl::range<1>(1), h);
    h.parallel_for<BeamCompactRoundKernel<R, PairPartitions>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) MOM_REQD_SG_16 MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned lid = (unsigned)it.get_local_id(0);
          const unsigned lsz = (unsigned)it.get_local_range(0);
          const uint32_t group = (uint32_t)it.get_group(0);
          const uint32_t bucket = group / GROUP_SUBGROUPS;
          const uint32_t subgroup = group & (GROUP_SUBGROUPS - 1u);
          if (lid == 0)
            local_count[0] = 0;
          for (unsigned b = lid; b < BIN_COUNT; b += lsz)
            bin_head[b] = 0;
          sycl::group_barrier(it.get_group());

          uint32_t count = d_in_ns[bucket];
          if (count > IN_NSLOTS)
            count = IN_NSLOTS;
          const BeamCompactRow* input = d_in + (size_t)bucket * IN_NSLOTS;
          for (unsigned s = lid; s < count; s += lsz) {
            const uint32_t collision =
                (uint32_t)input[s].word[0] & ((1u << COLLISION_BIT_SIZE) - 1u);
            const uint32_t input_subgroup =
                (collision >> IN_BUCKET_BITS) & (IN_SUBGROUPS - 1u);
            if ((input_subgroup >> MERGED_SUBGROUP_BITS) != subgroup)
              continue;
            const uint32_t pos = slm_atomic_u32(local_count[0]).fetch_add(1u);
            if (pos >= LOCAL_CAP)
              continue;
            const uint32_t original_rest = collision >> (IN_BUCKET_BITS + IN_SUBGROUP_BITS);
            const uint32_t rest =
                (original_rest << MERGED_SUBGROUP_BITS) |
                (input_subgroup & ((1u << MERGED_SUBGROUP_BITS) - 1u));
            const uint32_t prev = slm_atomic_u32(bin_head[rest]).exchange(pos + 1u);
            slot_index[pos] = static_cast<local_index_t>(s);
            bin_next[pos] = static_cast<local_index_t>(prev);
          }
          sycl::group_barrier(it.get_group());

          const uint32_t used = std::min<uint32_t>(local_count[0], LOCAL_CAP);
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
                if (((arow.word[0] ^ brow.word[0]) & 0x0000ffffffffffffull) != 0)
                  continue;
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
                    if (tree[i] == tree[j]) {
                      distinct = false;
                      break;
                    }
                if (!distinct)
                  continue;
                const uint32_t ci = dev_atomic_u32(d_cand_count[0]).fetch_add(1u);
                if (ci < cand_cap)
                  for (unsigned i = 0; i < OUT_TREE; ++i)
                    d_cand[ci].leaves[i] = tree[i];
              } else {
                uint64_t merged[WORK_WORDS];
                merge_workbits(aw, bw, mergeLen, merged);
                if constexpr (R == 4) {
                  uint64_t leader_tree[4], follower_tree[4];
                  compact_l3_tree_stream(leader, leader_tree);
                  compact_l3_tree_stream(follower, follower_tree);
                  compact_round4_mix(merged, leader_tree, follower_tree);
                  const uint32_t out_bucket = (uint32_t)merged[0] & (OUT_NBUCKETS - 1u);
                  const uint32_t out_slot = dev_atomic_u32(d_out_ns[out_bucket]).fetch_add(1u);
                  if (out_slot < OUT_NSLOTS) {
                    const uint32_t a_source = bucket * IN_NSLOTS + a_slot;
                    const uint32_t b_source = bucket * IN_NSLOTS + b_slot;
                    BeamLevel4Row* level4 = reinterpret_cast<BeamLevel4Row*>(d_out);
                    level4[(size_t)out_bucket * OUT_NSLOTS + out_slot] = {
                        merged[0] & 0x0000ffffffffffffull,
                        {a_first ? a_source : b_source, a_first ? b_source : a_source}};
                  }
                } else {
                  uint32_t tree[OUT_TREE];
                  for (unsigned i = 0; i < IN_TREE; ++i)
                    tree[i] = compact_get_index(leader, compact_work_bits(L), i);
                  for (unsigned i = 0; i < IN_TREE; ++i)
                    tree[IN_TREE + i] = compact_get_index(follower, compact_work_bits(L), i);
                  apply_mix(merged, tree, OUT_TREE, ROUND_REMLEN[R + 1]);
                  const BeamCompactRow outrow = compact_make_row<R>(merged, tree);
                  const uint32_t out_bucket = (uint32_t)merged[0] & (OUT_NBUCKETS - 1u);
                  const uint32_t out_slot = dev_atomic_u32(d_out_ns[out_bucket]).fetch_add(1u);
                  if (out_slot < OUT_NSLOTS)
                    d_out[(size_t)out_bucket * OUT_NSLOTS + out_slot] = outrow;
                }
              }
            }
          }
        });
  });
}

class BeamCompactFinalThinKernel;
static sycl::event submit_compact_final_thin(sycl::queue& q, const BeamLevel4Row* d_in,
                                             const uint32_t* d_in_ns,
                                             const BeamCompactRow* d_level3, BeamCandidate* d_cand,
                                             uint32_t* d_cand_count, uint32_t cand_cap,
                                             unsigned wg) {
  constexpr unsigned L = 4;
  constexpr unsigned BUCKET_BITS = compact_bucket_bits(L);
  constexpr unsigned NBUCKETS = compact_nbuckets(L);
  constexpr unsigned SUBGROUP_BITS = compact_subgroup_bits(L);
  constexpr unsigned SUBGROUPS = compact_subgroups(L);
  constexpr unsigned NSLOTS = compact_nslots(L);
  const size_t global = (size_t)NBUCKETS * SUBGROUPS * wg;
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> bin_head(sycl::range<1>(COMPACT_NRESTBINS), h);
    sycl::local_accessor<uint32_t, 1> bin_next(sycl::range<1>(COMPACT_LOCAL_CAP), h);
    sycl::local_accessor<uint32_t, 1> slot_index(sycl::range<1>(COMPACT_LOCAL_CAP), h);
    sycl::local_accessor<uint32_t, 1> local_count(sycl::range<1>(1), h);
    h.parallel_for<BeamCompactFinalThinKernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned lid = (unsigned)it.get_local_id(0);
          const unsigned lsz = (unsigned)it.get_local_range(0);
          const uint32_t group = (uint32_t)it.get_group(0);
          const uint32_t bucket = group / SUBGROUPS;
          const uint32_t half = group & (SUBGROUPS - 1u);
          if (lid == 0)
            local_count[0] = 0;
          for (unsigned b = lid; b < COMPACT_NRESTBINS; b += lsz)
            bin_head[b] = 0;
          sycl::group_barrier(it.get_group());

          uint32_t count = d_in_ns[bucket];
          if (count > NSLOTS)
            count = NSLOTS;
          const BeamLevel4Row* input = d_in + (size_t)bucket * NSLOTS;
          for (unsigned s = lid; s < count; s += lsz) {
            const uint32_t collision = (uint32_t)input[s].work & ((1u << COLLISION_BIT_SIZE) - 1u);
            if (((collision >> BUCKET_BITS) & (SUBGROUPS - 1u)) != half)
              continue;
            const uint32_t pos = slm_atomic_u32(local_count[0]).fetch_add(1u);
            if (pos >= COMPACT_LOCAL_CAP)
              continue;
            const uint32_t rest = collision >> (BUCKET_BITS + SUBGROUP_BITS);
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
              if (((arow.work ^ brow.work) & 0x0000ffffffffffffull) != 0)
                continue;

              const BeamCompactRow a_leader = d_level3[arow.parent[0]];
              const BeamCompactRow b_leader = d_level3[brow.parent[0]];
              const bool a_first = compact_get_index(a_leader, compact_work_bits(3), 0) <
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
                  if (tree[i] == tree[j]) {
                    distinct = false;
                    break;
                  }
              if (!distinct)
                continue;
              const uint32_t ci = dev_atomic_u32(d_cand_count[0]).fetch_add(1u);
              if (ci < cand_cap)
                for (unsigned i = 0; i < NUM_LEAVES; ++i)
                  d_cand[ci].leaves[i] = tree[i];
            }
          }
        });
  });
}
#endif
