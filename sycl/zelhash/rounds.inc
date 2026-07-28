struct EqCandidateRef { uint32_t bucket, slot_a, slot_b; };
struct EqCandidate { uint32_t leaves[1u << K]; };
template <unsigned R, bool GPU_COMPACT> class ZelHashRoundKernel;
template <unsigned R, bool GPU_COMPACT> class ZelHashSplitRoundKernel;
class ZelHashCudaTiledRound0Kernel;
template <unsigned R> class ZelHashCudaTiledLaterRoundKernel;
template <unsigned R, bool GPU_COMPACT = false>
static sycl::event submit_round(
  sycl::queue& q, MomKernelBundle& kb, const uint32_t* d_in, uint32_t* d_out,
  const uint32_t* d_in_nslots, uint32_t* d_out_nslots, unsigned wg, unsigned slot_capacity,
  EqCandidateRef* d_ref = nullptr, uint32_t* d_ref_count = nullptr, uint32_t cand_cap = 0
) {
  // Progressive per-level widths (compile-time): round R reads level R (IN_FIELDS field words at stride
  // IN_U32) and writes level R+1 (OUT_FIELDS = IN_FIELDS-1 field words at stride OUT_U32). Fewer words
  // written per survivor == less scattered global write traffic (the bandwidth bottleneck).
  constexpr unsigned IN_FIELDS  = level_fields(R);       // 5,4,3,2 for R=0..3
  constexpr unsigned IN_U32     = level_u32(R, GPU_COMPACT);
  constexpr unsigned OUT_FIELDS = level_fields(R + 1);   // 4,3,2,1
  constexpr unsigned OUT_U32    = level_u32(R + 1, GPU_COMPACT);
#include "round_cuda.inc"
  if constexpr (ROUND_DIRECT_CHAINS) {
    constexpr unsigned SPLITS = R < 2u ? 2u : 1u;
    constexpr unsigned SPLIT_BITS = R < 2u ? 1u : 0u;
    // Each early split averages at most MAX_SLOTS/2 rows. This leaves headroom at the capped input
    // size while keeping round 0 under Xe's 64-KiB SLM limit; the normal cap remains later.
    constexpr unsigned LOCAL_SLOTS = R < 2u ? 2432u : MAX_SLOTS;
    constexpr unsigned LOCAL_RESTBINS = NRESTBINS / SPLITS;
    const unsigned split_wg = R < 2u ? 512u : wg;
    const size_t global = (size_t)NBUCKETS * SPLITS * split_wg;
    return q.submit([&](sycl::handler& h) {
      mom_use_bundle(h, kb);
      sycl::local_accessor<uint32_t, 1> heads(sycl::range<1>(LOCAL_RESTBINS), h);
      sycl::local_accessor<uint32_t, 1> fields(sycl::range<1>(LOCAL_SLOTS * (IN_FIELDS - 1u)), h);
      sycl::local_accessor<uint16_t, 1> next(sycl::range<1>(LOCAL_SLOTS), h);
      sycl::local_accessor<uint16_t, 1> original(sycl::range<1>(LOCAL_SLOTS), h);
      sycl::local_accessor<uint32_t, 1> local_count(sycl::range<1>(1), h);
      h.parallel_for<ZelHashSplitRoundKernel<R, GPU_COMPACT>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(split_wg)),
        [=](sycl::nd_item<1> it) {
          const unsigned group = (unsigned)it.get_group(0);
          const uint32_t bucket = group / SPLITS;
          const uint32_t split = group & (SPLITS - 1u);
          const unsigned lid = (unsigned)it.get_local_id(0);
          const unsigned lsz = (unsigned)it.get_local_range(0);
          uint32_t* const head = heads.template get_multi_ptr<sycl::access::decorated::no>().get();
          uint32_t* const staged = fields.template get_multi_ptr<sycl::access::decorated::no>().get();
          uint16_t* const chain = next.template get_multi_ptr<sycl::access::decorated::no>().get();
          uint16_t* const input_slot = original.template get_multi_ptr<sycl::access::decorated::no>().get();
          uint32_t* const count_ptr = local_count.template get_multi_ptr<sycl::access::decorated::no>().get();

          uint32_t cnt = d_in_nslots[bucket];
          if (cnt > slot_capacity) cnt = slot_capacity;
          for (unsigned bin = lid; bin < LOCAL_RESTBINS; bin += lsz) head[bin] = 0;
          if (lid == 0) count_ptr[0] = 0;
          sycl::group_barrier(it.get_group());

          const uint32_t* in_base = d_in + (size_t)bucket * slot_capacity * IN_U32;
          for (unsigned s = lid; s < cnt; s += lsz) {
            const uint32_t* g = in_base + (size_t)s * IN_U32;
            const uint32_t rest = collision_rest<R, GPU_COMPACT>(g);
            if ((rest & (SPLITS - 1u)) != split) continue;
            const uint32_t pos = slm_atomic_u32(count_ptr[0]).fetch_add(1u);
            if (pos >= LOCAL_SLOTS) continue;
            input_slot[pos] = (uint16_t)s;
            for (unsigned k = 0; k < IN_FIELDS - 1u; ++k)
              staged[(size_t)k * LOCAL_SLOTS + pos] = load_follower<R, GPU_COMPACT>(g, k);
            const uint32_t previous = slm_atomic_u32(head[rest >> SPLIT_BITS]).exchange(pos + 1u);
            chain[pos] = (uint16_t)previous;
          }
          sycl::group_barrier(it.get_group());

          const uint32_t compact_count = sycl::min(count_ptr[0], (uint32_t)LOCAL_SLOTS);
          for (unsigned a = lid; a < compact_count; a += lsz) {
            for (uint32_t link = chain[a]; link != 0; link = chain[link - 1u]) {
              const unsigned b = link - 1u;
              const uint32_t new_active = staged[a] ^ staged[b];
              if constexpr (R == K - 1u) {
                if (new_active != 0u) continue;
                const uint32_t idx = dev_atomic_u32(d_ref_count[0]).fetch_add(1u);
                if (idx >= cand_cap) continue;
                d_ref[idx] = {bucket, input_slot[a], input_slot[b]};
              } else {
                const uint32_t out_bucket = field_bucket(new_active);
                const uint32_t oslot = dev_atomic_u32(d_out_nslots[out_bucket]).fetch_add(1u);
                if (oslot >= slot_capacity) continue;
                uint32_t* o = d_out + ((size_t)out_bucket * slot_capacity + oslot) * OUT_U32;
                const uint32_t first = field_rest(new_active) | (bucket << RESTBITS);
                if constexpr (R == 0u) {
                  store_intel_u32x4(o, first,
                    staged[(size_t)LOCAL_SLOTS + a] ^ staged[(size_t)LOCAL_SLOTS + b],
                    staged[2u * (size_t)LOCAL_SLOTS + a] ^ staged[2u * (size_t)LOCAL_SLOTS + b],
                    staged[3u * (size_t)LOCAL_SLOTS + a] ^ staged[3u * (size_t)LOCAL_SLOTS + b]);
                } else {
                  o[0] = first;
                  for (unsigned k = 1; k < OUT_FIELDS; ++k) {
                    const uint32_t* f = staged + (size_t)k * LOCAL_SLOTS;
                    o[k] = f[a] ^ f[b];
                  }
                }
              }
            }
          }
        });
    });
  }
  if constexpr (!ROUND_DIRECT_CHAINS) {
    const size_t global = (size_t)NBUCKETS * wg;
    return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    // One reusable SLM arena implements the fast block-local sequence used by mature Equihash solvers:
    // build exact-bin chains, overwrite the now-dead heads with a compact pair list, retain a bounded
    // number of pair IDs in each lane's registers, then overwrite the pair list with staged record
    // fields. Keeping links behind the maximum pair list makes these lifetimes non-overlapping; the
    // selected Intel path uses at most 29 KiB, CUDA/HIP about 53 KiB, and AdaptiveCpp/CUDA's
    // small-bucket fallback about 16 KiB. This avoids a random global load for every collision partner.
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
    constexpr unsigned PAIRS_PER_LANE = 3u;
#elif !defined(MOM_SYCL_HAS_CUDA) && !defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
    // WG1024 x 5 covers every measured Intel pair (a sixth slot did not change survivor counts).
    // The former WG512 x 9 cap silently discarded 150 pairs in the block-400000 exact profile.
    constexpr unsigned PAIRS_PER_LANE = 5u;
#else
    constexpr unsigned PAIRS_PER_LANE = 9u;
#endif
    constexpr unsigned PAIR_CAP = PAIRS_PER_LANE * ROUND_WG;
    constexpr unsigned LINK_U16_OFF = 2u * PAIR_CAP;
    constexpr unsigned COUNTER_U16_OFF = LINK_U16_OFF + MAX_SLOTS;
    constexpr unsigned SCRATCH_U32 = (COUNTER_U16_OFF + 3u) / 2u;
    sycl::local_accessor<uint32_t, 1> scratch(sycl::range<1>(SCRATCH_U32), h);
    constexpr unsigned FOLLOWERS = IN_FIELDS - 1u;
    h.parallel_for<ZelHashRoundKernel<R, GPU_COMPACT>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
      [=](sycl::nd_item<1> it) {
      const uint32_t bucket = (uint32_t)it.get_group(0);
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);
      uint32_t* const scratch32 = scratch.template get_multi_ptr<sycl::access::decorated::no>().get();
      uint16_t* const scratch16 = reinterpret_cast<uint16_t*>(scratch32);
      uint16_t* const bin_next = scratch16 + LINK_U16_OFF;
      constexpr unsigned COUNTER_U32 = COUNTER_U16_OFF / 2u;

      uint32_t cnt = d_in_nslots[bucket];
      if (cnt > slot_capacity) cnt = slot_capacity;
      for (unsigned r = lid; r < NRESTBINS; r += lsz) scratch32[r] = 0;
      if (lid == 0) scratch32[COUNTER_U32] = 0;
      sycl::group_barrier(it.get_group());

      const uint32_t* in_base = d_in + (size_t)bucket * slot_capacity * IN_U32;

      // Build the rest-bin linked lists. At levels >=1, w[0] also carries the parent bucket above
      // RESTBITS; masking it still selects the active field's exact low-bit bin.
      for (unsigned s = lid; s < cnt; s += lsz) {
        const uint32_t* g = in_base + (size_t)s * IN_U32;
        const uint32_t rb = collision_rest<R, GPU_COMPACT>(g);
        const uint32_t previous = slm_atomic_u32(scratch32[rb]).exchange(s + 1u);
        bin_next[s] = (uint16_t)previous;
      }
      sycl::group_barrier(it.get_group());

      // The predecessor chain contains exactly the records inserted before s, so every unordered pair
      // is emitted once without rereading the heads. Pair-list writes may now safely overwrite heads.
      for (unsigned s = lid; s < cnt; s += lsz) {
        uint32_t link = bin_next[s];                                // 1 + predecessor slot
        for (; link != 0; link = bin_next[link - 1u]) {
          const uint32_t t = link - 1u;
          const uint32_t pos = slm_atomic_u32(scratch32[COUNTER_U32]).fetch_add(1u);
          if (pos < PAIR_CAP) scratch32[pos] = (s << 16) | t;
        }
      }
      sycl::group_barrier(it.get_group());

      const uint32_t pair_count = sycl::min(scratch32[COUNTER_U32], (uint32_t)PAIR_CAP);
      uint32_t pair_code[PAIRS_PER_LANE];
      bool pair_valid[PAIRS_PER_LANE];
      for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
        const unsigned p = lid + j * lsz;
        pair_valid[j] = p < pair_count;
        pair_code[j] = pair_valid[j] ? scratch32[p] : 0u;
      }
      sycl::group_barrier(it.get_group());

      // A complete 32-bit field occupies at most 34 KiB on the selected paths, so stage each follower
      // in one pass. This is fewer barriers than splitting 64-bit slices into slot halves on portable 64-KiB
      // implementations, while pair IDs remain in registers and random global partner loads disappear.
      uint32_t xf[PAIRS_PER_LANE][FOLLOWERS];
      for (unsigned k = 0; k < FOLLOWERS; ++k) {
        for (unsigned s = lid; s < cnt; s += lsz) {
          const uint32_t* g = in_base + (size_t)s * IN_U32;
          scratch32[s] = load_follower<R, GPU_COMPACT>(g, k);
        }
        sycl::group_barrier(it.get_group());
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          if (!pair_valid[j]) continue;
          const unsigned s = pair_code[j] >> 16;
          const unsigned t = pair_code[j] & 0xffffu;
          xf[j][k] = scratch32[s] ^ scratch32[t];
        }
        sycl::group_barrier(it.get_group());
      }

      for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
        if (!pair_valid[j]) continue;
        // Slots always keep the ACTIVE collision field in w[0] and the remaining (higher) fields in
        // w[1..]. The active field XORs to 0; shift the rest down by one so w[0] becomes the NEXT
        // collision field at level R+1. (Position-relative, independent of which absolute field R is.)
        const uint32_t new_active = xf[j][0];                     // new active field (full 25-bit)
        if constexpr (R == K - 1u) {
          if (new_active != 0u) continue;
          const uint32_t idx = dev_atomic_u32(d_ref_count[0]).fetch_add(1u);
          if (idx >= cand_cap) continue;
          const uint32_t code = pair_code[j];
          d_ref[idx] = {bucket, code >> 16, code & 0xffffu};
          continue;
        }
        const uint32_t out_bucket = field_bucket(new_active);
        const uint32_t oslot = dev_atomic_u32(d_out_nslots[out_bucket]).fetch_add(1u);
        if (oslot >= slot_capacity) continue;
        // Single compact stream: w0 = active_low | (parent_bucket<<RESTBITS), followers in w1...
        uint32_t* o = d_out + ((size_t)out_bucket * slot_capacity + oslot) * OUT_U32;
        const uint32_t first = field_rest(new_active) | (bucket << RESTBITS);
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
        if constexpr (GPU_COMPACT && R == 0) {
          store_aligned_u32x4(o, first, xf[j][1], xf[j][2], xf[j][3]);
        } else if constexpr (GPU_COMPACT && R == 1) {
          store_aligned_u32x4(o, first, xf[j][1], xf[j][2], 0u);
        } else if constexpr (GPU_COMPACT && R == 2) {
          store_aligned_u32x4(o, first, xf[j][1], 0u, 0u);
        } else
#endif
        {
          o[0] = first;
          for (unsigned k = 1; k < OUT_FIELDS; ++k) o[k] = xf[j][k];
        }
      }
      });
    });
  }
}
