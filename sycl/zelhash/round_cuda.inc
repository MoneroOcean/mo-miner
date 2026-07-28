#if defined(MOM_SYCL_HAS_CUDA)
  if constexpr (GPU_COMPACT && R == 0u) {
    constexpr unsigned PAIRS_PER_LANE = 9u;
    constexpr unsigned PAIR_CAP = PAIRS_PER_LANE * ROUND_WG;
    constexpr unsigned LINK_U16_OFF = 2u * PAIR_CAP;
    constexpr unsigned COUNTER_U16_OFF = LINK_U16_OFF + MAX_SLOTS;
    constexpr unsigned COUNTER_U32 = COUNTER_U16_OFF / 2u;
    constexpr unsigned STAGE_U32 = 2u * MAX_SLOTS;
    const size_t global = (size_t)NBUCKETS * wg;
    return q.submit([&](sycl::handler& h) {
      mom_use_bundle(h, kb);
      // The chain/pair-list phase needs ~52 KiB. Once pair codes are in registers, reuse the same
      // arena as two 32-KiB field planes, matching the GPU's 64-KiB fast shared-memory shape.
      sycl::local_accessor<uint32_t, 1> scratch(sycl::range<1>(STAGE_U32), h);
      h.parallel_for<ZelHashCudaTiledRound0Kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const uint32_t bucket = (uint32_t)it.get_group(0);
        const unsigned lid = (unsigned)it.get_local_id(0);
        uint32_t* const scratch32 = scratch.template get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* const chain = reinterpret_cast<uint16_t*>(scratch32) + LINK_U16_OFF;
        const uint32_t cnt = sycl::min(d_in_nslots[bucket], (uint32_t)slot_capacity);
        // Exactly eight possible rows per lane. Retain the aligned records in registers through the
        // chain build; later half-staging deliberately ends each pair of field lifetimes.
        uint32_t f1[8], f2[8], f3[8], f4[8];
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          NativeU32x4 v = {0u, 0u, 0u, 0u};
          if (s < cnt)
            v = *reinterpret_cast<const NativeU32x4*>(d_in + cuda_record_pos(bucket, s) * IN_U32);
          f1[t] = v[0]; f2[t] = v[1]; f3[t] = v[2]; f4[t] = v[3];
        }

        for (unsigned r = lid; r < NRESTBINS; r += ROUND_WG) scratch32[r] = 0;
        if (lid == 0) scratch32[COUNTER_U32] = 0;
        sycl::group_barrier(it.get_group());

#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s >= cnt) continue;
          const uint32_t rb = (f1[t] >> 25) | ((f2[t] >> 25) << 7);
          chain[s] = (uint16_t)slm_atomic_u32(scratch32[rb]).exchange(s + 1u);
        }
        sycl::group_barrier(it.get_group());

        // Count pairs per lane, then reserve each lane's contiguous segment with one warp scan. This
        // removes the shared atomic (and its NVPTX vote/shuffle aggregation) from every chain link.
        uint32_t local_pairs = 0;
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s >= cnt) continue;
          for (uint32_t link = chain[s]; link != 0u; link = chain[link - 1u]) ++local_pairs;
        }
        const sycl::sub_group sg = it.get_sub_group();
        const unsigned lane = (unsigned)sg.get_local_linear_id();
        const unsigned warp = lid >> 5;
        const uint32_t lane_prefix = sycl::exclusive_scan_over_group(sg, local_pairs, sycl::plus<uint32_t>());
        const uint32_t warp_total = sycl::reduce_over_group(sg, local_pairs, sycl::plus<uint32_t>());
        if (lane == 0u) scratch32[warp] = warp_total;
        sycl::group_barrier(it.get_group());
        if (warp == 0u) {
          const uint32_t v = scratch32[lane];
          const uint32_t p = sycl::exclusive_scan_over_group(sg, v, sycl::plus<uint32_t>());
          scratch32[lane] = p;
          if (lane == 31u) scratch32[32] = p + v;
        }
        sycl::group_barrier(it.get_group());
        const uint32_t pair_base = scratch32[warp] + lane_prefix;
        const uint32_t pair_count = sycl::min(scratch32[32], (uint32_t)PAIR_CAP);
        sycl::group_barrier(it.get_group());
        uint32_t local_pos = 0;
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s >= cnt) continue;
          for (uint32_t link = chain[s]; link != 0u; link = chain[link - 1u]) {
            const uint32_t pos = pair_base + local_pos++;
            if (pos < PAIR_CAP) scratch32[pos] = (s << 16) | (link - 1u);
          }
        }
        sycl::group_barrier(it.get_group());

        uint32_t pair_code[PAIRS_PER_LANE];
        bool pair_valid[PAIRS_PER_LANE];
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          const unsigned p = lid + j * ROUND_WG;
          pair_valid[j] = p < pair_count;
          pair_code[j] = pair_valid[j] ? scratch32[p] : 0u;
        }
        sycl::group_barrier(it.get_group());

        // First half: stage f1/f2 from the retained records, then keep only their pair XORs.
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s < cnt) {
            scratch32[s] = f1[t] & 0x1ffffffu;
            scratch32[MAX_SLOTS + s] = f2[t] & 0x1ffffffu;
          }
        }
        sycl::group_barrier(it.get_group());
        uint32_t x0[PAIRS_PER_LANE], x1[PAIRS_PER_LANE];
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          if (!pair_valid[j]) { x0[j] = x1[j] = 0u; continue; }
          const unsigned a = pair_code[j] >> 16, b = pair_code[j] & 0xffffu;
          x0[j] = scratch32[a] ^ scratch32[b];
          x1[j] = scratch32[MAX_SLOTS + a] ^ scratch32[MAX_SLOTS + b];
        }
        sycl::group_barrier(it.get_group());

        // Second half overwrites the arena after every lane consumed the first. Pair outputs can now
        // be emitted as one aligned vector; the 32 input-record registers are no longer live.
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s < cnt) {
            scratch32[s] = f3[t] & 0x1ffffffu;
            scratch32[MAX_SLOTS + s] = f4[t] & 0x1ffffffu;
          }
        }
        sycl::group_barrier(it.get_group());
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          if (!pair_valid[j]) continue;
          const unsigned a = pair_code[j] >> 16, b = pair_code[j] & 0xffffu;
          const uint32_t new_active = x0[j];
          const uint32_t out_bucket = field_bucket(new_active);
          const uint32_t oslot = dev_atomic_u32(d_out_nslots[out_bucket]).fetch_add(1u);
          if (oslot >= slot_capacity) continue;
          uint32_t* o = d_out + cuda_record_pos(out_bucket, oslot) * OUT_U32;
          store_aligned_u32x4(o, field_rest(new_active) | (bucket << RESTBITS), x1[j],
            scratch32[a] ^ scratch32[b],
            scratch32[MAX_SLOTS + a] ^ scratch32[MAX_SLOTS + b]);
        }
      });
    });
  }
  if constexpr (GPU_COMPACT && R >= 1u) {
    constexpr unsigned PAIRS_PER_LANE = 9u;
    constexpr unsigned PAIR_CAP = PAIRS_PER_LANE * ROUND_WG;
    constexpr unsigned LINK_U16_OFF = 2u * PAIR_CAP;
    constexpr unsigned STAGE_U32 = 2u * MAX_SLOTS;
    const size_t global = (size_t)NBUCKETS * wg;
    return q.submit([&](sycl::handler& h) {
      mom_use_bundle(h, kb);
      sycl::local_accessor<uint32_t, 1> scratch(sycl::range<1>(STAGE_U32), h);
      h.parallel_for<ZelHashCudaTiledLaterRoundKernel<R>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const uint32_t bucket = (uint32_t)it.get_group(0);
        const unsigned lid = (unsigned)it.get_local_id(0);
        uint32_t* const scratch32 = scratch.template get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* const chain = reinterpret_cast<uint16_t*>(scratch32) + LINK_U16_OFF;
        const uint32_t cnt = sycl::min(d_in_nslots[bucket], (uint32_t)slot_capacity);
        uint32_t w0[8], w1[8], w2[8], w3[8];
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          NativeU32x4 v = {0u, 0u, 0u, 0u};
          if (s < cnt) {
            const uint32_t* p = d_in + cuda_record_pos(bucket, s) * IN_U32;
            if constexpr (R == 3u) {
              v[0] = p[0];
              v[1] = p[1];
            } else {
              v = *reinterpret_cast<const NativeU32x4*>(p);
            }
          }
          w0[t] = v[0]; w1[t] = v[1]; w2[t] = v[2]; w3[t] = v[3];
        }
        for (unsigned r = lid; r < NRESTBINS; r += ROUND_WG) scratch32[r] = 0;
        sycl::group_barrier(it.get_group());
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s < cnt)
            chain[s] = (uint16_t)slm_atomic_u32(scratch32[stored_rest(w0[t])]).exchange(s + 1u);
        }
        sycl::group_barrier(it.get_group());

        uint32_t local_pairs = 0;
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s >= cnt) continue;
          for (uint32_t link = chain[s]; link != 0u; link = chain[link - 1u]) ++local_pairs;
        }
        const sycl::sub_group sg = it.get_sub_group();
        const unsigned lane = (unsigned)sg.get_local_linear_id();
        const unsigned warp = lid >> 5;
        const uint32_t lane_prefix = sycl::exclusive_scan_over_group(sg, local_pairs, sycl::plus<uint32_t>());
        const uint32_t warp_total = sycl::reduce_over_group(sg, local_pairs, sycl::plus<uint32_t>());
        if (lane == 0u) scratch32[warp] = warp_total;
        sycl::group_barrier(it.get_group());
        if (warp == 0u) {
          const uint32_t v = scratch32[lane];
          const uint32_t p = sycl::exclusive_scan_over_group(sg, v, sycl::plus<uint32_t>());
          scratch32[lane] = p;
          if (lane == 31u) scratch32[32] = p + v;
        }
        sycl::group_barrier(it.get_group());
        const uint32_t pair_base = scratch32[warp] + lane_prefix;
        const uint32_t pair_count = sycl::min(scratch32[32], (uint32_t)PAIR_CAP);
        sycl::group_barrier(it.get_group());
        uint32_t local_pos = 0;
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s >= cnt) continue;
          for (uint32_t link = chain[s]; link != 0u; link = chain[link - 1u]) {
            const uint32_t pos = pair_base + local_pos++;
            if (pos < PAIR_CAP) scratch32[pos] = (s << 16) | (link - 1u);
          }
        }
        sycl::group_barrier(it.get_group());

        uint32_t pair_code[PAIRS_PER_LANE];
        bool pair_valid[PAIRS_PER_LANE];
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          const unsigned p = lid + j * ROUND_WG;
          pair_valid[j] = p < pair_count;
          pair_code[j] = pair_valid[j] ? scratch32[p] : 0u;
        }
        sycl::group_barrier(it.get_group());

#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s < cnt) {
            scratch32[s] = w1[t];
            if constexpr (R <= 2u) scratch32[MAX_SLOTS + s] = w2[t];
          }
        }
        sycl::group_barrier(it.get_group());
        uint32_t x0[PAIRS_PER_LANE], x1[PAIRS_PER_LANE];
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          if (!pair_valid[j]) { x0[j] = x1[j] = 0u; continue; }
          const unsigned a = pair_code[j] >> 16, b = pair_code[j] & 0xffffu;
          x0[j] = scratch32[a] ^ scratch32[b];
          if constexpr (R <= 2u) x1[j] = scratch32[MAX_SLOTS + a] ^ scratch32[MAX_SLOTS + b];
          else x1[j] = 0u;
        }

        if constexpr (R == 1u) {
          sycl::group_barrier(it.get_group());
#pragma unroll
          for (unsigned t = 0; t < 8u; ++t) {
            const unsigned s = lid + t * ROUND_WG;
            if (s < cnt) scratch32[s] = w3[t];
          }
          sycl::group_barrier(it.get_group());
        }
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          if (!pair_valid[j]) continue;
          const unsigned a = pair_code[j] >> 16, b = pair_code[j] & 0xffffu;
          const uint32_t new_active = x0[j];
          if constexpr (R == K - 1u) {
            if (new_active != 0u) continue;
            const uint32_t idx = dev_atomic_u32(d_ref_count[0]).fetch_add(1u);
            if (idx < cand_cap) d_ref[idx] = {bucket, a, b};
          } else {
            const uint32_t out_bucket = field_bucket(new_active);
            const uint32_t oslot = dev_atomic_u32(d_out_nslots[out_bucket]).fetch_add(1u);
            if (oslot >= slot_capacity) continue;
            uint32_t* o = d_out + cuda_record_pos(out_bucket, oslot) * OUT_U32;
            if constexpr (R == 1u)
              store_aligned_u32x4(o, field_rest(new_active) | (bucket << RESTBITS), x1[j],
                                  scratch32[a] ^ scratch32[b], 0u);
            else {
              o[0] = field_rest(new_active) | (bucket << RESTBITS);
              o[1] = x1[j];
            }
          }
        }
      });
    });
  }
#endif
