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
