struct EqLevels {
  const uint32_t* level[NSTORED_LEVELS];
  const uint32_t* l0_index;
};

// Reconstruct each final pair's level-3 nodes by rescanning only their recorded parent buckets.
// A parent bucket contains about 8192 rows and about as many collision pairs; candidate recovery is
// rare (~44 zero nodes before DistinctIndices) and therefore much cheaper than carrying a 26-bit
// parent-pair identifier through all ~67 million survivors in every round.
template <bool GPU_COMPACT>
inline uint32_t recovery_follower(const uint32_t* g, unsigned level, unsigned k) {
  if (level == 0) {
#if defined(MOM_SYCL_HAS_CUDA)
    if constexpr (GPU_COMPACT) return g[k] & 0x1ffffffu;
#endif
    switch (k) {
      case 0: return ((g[0] >> 25) & 0x7fu) | ((g[1] & 0x3ffffu) << 7);
      case 1: return (g[1] >> 18) | ((g[2] & 0x7ffu) << 14);
      case 2: return (g[2] >> 11) | ((g[3] & 0xfu) << 21);
      default: return (g[3] >> 4) & 0x01ffffffu;
    }
  }
  return g[k + 1u];
}

template <bool GPU_COMPACT> class ZelHashRecoverKernel;
template <bool GPU_COMPACT = false>
static sycl::event submit_recover(
  sycl::queue& q, MomKernelBundle& kb, EqLevels lv, const uint32_t* d_nslots,
  const EqCandidateRef* d_ref, uint32_t ref_count, EqCandidate* d_cand,
  uint32_t* d_cand_count, uint32_t cand_cap, unsigned wg, unsigned slot_capacity,
  const uint32_t* d_ref_count = nullptr
) {
  if (!ref_count) return sycl::event{};
  const size_t global = (size_t)ref_count * wg;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    sycl::local_accessor<uint32_t, 1> bin_head(sycl::range<1>(NRESTBINS), h);
    sycl::local_accessor<uint16_t, 1> bin_next(sycl::range<1>(MAX_SLOTS), h);
    sycl::local_accessor<uint32_t, 1> node_bucket(sycl::range<1>(1u << K), h);
    sycl::local_accessor<uint32_t, 1> node_slot(sycl::range<1>(1u << K), h);
    sycl::local_accessor<uint32_t, 1> next_bucket(sycl::range<1>(1u << K), h);
    sycl::local_accessor<uint32_t, 1> next_slot(sycl::range<1>(1u << K), h);
    sycl::local_accessor<uint32_t, 1> target(sycl::range<1>(K), h);
    sycl::local_accessor<uint32_t, 1> found(sycl::range<1>(1), h);
    sycl::local_accessor<uint32_t, 1> valid(sycl::range<1>(1), h);
    h.parallel_for<ZelHashRecoverKernel<GPU_COMPACT>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
      [=](sycl::nd_item<1> it) {
      const unsigned candidate = (unsigned)it.get_group(0);
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);
      // The normal in-order chain launches the bounded recovery grid before the host reads the
      // final-round count. This uniform early exit removes an entire Windows event-wait quantum.
      if (d_ref_count && candidate >= *d_ref_count) return;
      if (lid == 0) {
        node_bucket[0] = node_bucket[1] = d_ref[candidate].bucket;
        node_slot[0] = d_ref[candidate].slot_a;
        node_slot[1] = d_ref[candidate].slot_b;
        valid[0] = 1;
      }
      sycl::group_barrier(it.get_group());

      // The fused final round already supplied the two level-3 children of the discarded level-4
      // zero node. Expand those two subtrees through levels 3..1 to recover all sixteen leaves.
      unsigned node_count = 2;
      for (unsigned level = K - 1u; level != 0; --level) {
        for (unsigned ni = 0; ni < node_count; ++ni) {
          if (lid == 0) {
            found[0] = 0xffffffffu;
            if (valid[0]) {
              const uint32_t cb = node_bucket[ni], cs = node_slot[ni];
              size_t child_pos = (size_t)cb * slot_capacity + cs;
#if defined(MOM_SYCL_HAS_CUDA)
              if constexpr (GPU_COMPACT) child_pos = cuda_record_pos(cb, cs);
#endif
              const uint32_t* child = lv.level[level]
                + child_pos * level_u32(level, GPU_COMPACT);
              next_bucket[2u * ni] = next_bucket[2u * ni + 1u] = child[0] >> RESTBITS;
              target[0] = field_from_parts(cb, stored_rest(child[0]));
              for (unsigned k = 1; k < level_fields(level); ++k) target[k] = child[k];
            }
          }
          for (unsigned r = lid; r < NRESTBINS; r += lsz) bin_head[r] = 0;
          sycl::group_barrier(it.get_group());
          if (!valid[0]) continue;

          const uint32_t pb = next_bucket[2u * ni];
          const unsigned parent_level = level - 1u;
          uint32_t cnt = d_nslots[(size_t)parent_level * NBUCKETS + pb];
          if (cnt > slot_capacity) cnt = slot_capacity;
          const unsigned parent_u32 = level_u32(parent_level, GPU_COMPACT);
          for (unsigned s = lid; s < cnt; s += lsz) {
            size_t record_pos = (size_t)pb * slot_capacity + s;
#if defined(MOM_SYCL_HAS_CUDA)
            if constexpr (GPU_COMPACT)
              record_pos = cuda_record_pos(pb, s);
#endif
            const uint32_t* record = lv.level[parent_level] + record_pos * parent_u32;
            uint32_t rb;
#if defined(MOM_SYCL_HAS_CUDA)
            if constexpr (GPU_COMPACT)
              rb = parent_level == 0u ? ((record[0] >> 25) | ((record[1] >> 25) << 7))
                                      : stored_rest(record[0]);
            else
#endif
              rb = parent_level == 0u ? field_rest(record[0]) : stored_rest(record[0]);
            const uint32_t previous = slm_atomic_u32(bin_head[rb]).exchange(s + 1u);
            bin_next[s] = (uint16_t)previous;
          }
          sycl::group_barrier(it.get_group());

          for (unsigned s = lid; s < cnt; s += lsz) {
            size_t a_pos = (size_t)pb * slot_capacity + s;
#if defined(MOM_SYCL_HAS_CUDA)
            if constexpr (GPU_COMPACT)
              a_pos = cuda_record_pos(pb, s);
#endif
            const uint32_t* a = lv.level[parent_level] + a_pos * parent_u32;
            for (uint32_t link = bin_next[s]; link != 0; link = bin_next[link - 1u]) {
              const unsigned t = link - 1u;
              size_t b_pos = (size_t)pb * slot_capacity + t;
#if defined(MOM_SYCL_HAS_CUDA)
              if constexpr (GPU_COMPACT)
                b_pos = cuda_record_pos(pb, t);
#endif
              const uint32_t* b = lv.level[parent_level] + b_pos * parent_u32;
              bool match = true;
              for (unsigned k = 0; k < level_fields(level); ++k) {
                if ((recovery_follower<GPU_COMPACT>(a, parent_level, k) ^
                     recovery_follower<GPU_COMPACT>(b, parent_level, k)) != target[k]) {
                  match = false; break;
                }
              }
              if (match) slm_atomic_u32(found[0]).exchange((s << 16) | t);
            }
          }
          sycl::group_barrier(it.get_group());
          if (lid == 0) {
            if (found[0] == 0xffffffffu) valid[0] = 0;
            else {
              next_slot[2u * ni] = found[0] >> 16;
              next_slot[2u * ni + 1u] = found[0] & 0xffffu;
            }
          }
          sycl::group_barrier(it.get_group());
        }
        node_count *= 2u;
        for (unsigned i = lid; i < node_count; i += lsz) {
          node_bucket[i] = next_bucket[i]; node_slot[i] = next_slot[i];
        }
        sycl::group_barrier(it.get_group());
      }

      constexpr unsigned PROOF = 1u << K;
      if (lid < PROOF && valid[0]) {
        const size_t leaf_pos = (size_t)node_bucket[lid] * slot_capacity + node_slot[lid];
#if defined(MOM_EQ_FORWARD_MAP)
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
        if constexpr (GPU_COMPACT) {
          // Generation stores a coalesced leaf->position map. Keep the physical position as
          // the recovery target; a single linear inversion pass resolves all final leaves together.
          next_slot[lid] = (node_bucket[lid] << 14) | node_slot[lid];
#else
          next_slot[lid] = lv.l0_index[leaf_pos];
#endif
        }
        else
#endif
        {
          const uint32_t* leaf = lv.level[0] + leaf_pos * level_u32(0, GPU_COMPACT);
          next_slot[lid] = dense_l0_load_index(leaf);
        }
      }
      sycl::group_barrier(it.get_group());
      if (lid == 0 && valid[0]) {
        bool distinct = true;
        for (unsigned i = 0; i < PROOF; ++i)
          for (unsigned j = i + 1; j < PROOF; ++j)
            if (next_slot[i] == next_slot[j]) distinct = false;
        if (distinct) {
          const uint32_t idx = dev_atomic_u32(d_cand_count[0]).fetch_add(1u);
          if (idx < cand_cap)
            for (unsigned i = 0; i < PROOF; ++i) d_cand[idx].leaves[i] = next_slot[i];
        }
      }
    });
  });
}
