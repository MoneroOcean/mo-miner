template <bool INLINE_PAIR> class EtchashSearchKernel;  // distinct kernel name per instantiation

// GPU etchash search kernel, instantiated for the inlined (fast) and noinline (always-correct) pair
// Keccak. Identical to the 0.7.0 kernel except the two per-nonce Keccaks select their permutation by
// INLINE_PAIR. The caller picks the instantiation per device from EtchashState::use_inline_pair().
template <bool INLINE_PAIR>
static sycl::event submit_etchash_search_gpu(
  sycl::queue& q, MomKernelBundle& kb,
  const uint8_t* const d_input, const uint64_t start_nonce,
  const uint32_t* const __restrict__ d_dag, const FastModData dag_mod,
  const uint32_t intensity, const uint8_t* const d_target,
  EtchashResult* const d_result, const bool is_test
) {
  constexpr unsigned GROUP4_WORKGROUP = 128;
  constexpr unsigned GROUP4_SEED_WORDS = GROUP4_WORKGROUP * 16;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    sycl::local_accessor<uint32_t, 1> seeds(sycl::range<1>(GROUP4_SEED_WORDS), h);
    h.parallel_for<EtchashSearchKernel<INLINE_PAIR>>(
      sycl::nd_range<1>(
        sycl::range<1>(round_up(intensity, GROUP4_WORKGROUP)),
        sycl::range<1>(GROUP4_WORKGROUP)
      ),
      [=](sycl::nd_item<1> item) MOM_REQD_SG_16 {
        const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));
        const uint32_t group_hash_base = static_cast<uint32_t>(item.get_group(0)) * GROUP4_WORKGROUP;
        const uint32_t own_hash = group_hash_base + lid;

        if (own_hash < intensity) {
          uint32_t seed[16];
          keccak_512_header_nonce_pair_dev<INLINE_PAIR>(d_input, start_nonce + own_hash, seed);
#pragma unroll
          for (unsigned w = 0; w < 16; ++w) seeds[lid * 16 + w] = seed[w];
        }
        item.barrier(sycl::access::fence_space::local_space);

        const auto sg = item.get_sub_group();
        const uint32_t sg_lane = static_cast<uint32_t>(sg.get_local_id()[0]);
        const uint32_t lane4 = lid & 3U;
        const uint32_t group4_base = lid & ~3U;
        const uint32_t sg_group4_base = sg_lane & ~3U;

        for (unsigned owner = 0; owner < 4; ++owner) {
          const uint32_t owner_lid = group4_base + owner;
          const uint32_t hash_index = group_hash_base + owner_lid;
          if (hash_index >= intensity) continue;

          const uint32_t seed_base = owner_lid * 16;
          const uint32_t seed0 = seeds[seed_base];
          uint32_t mix[8];
#pragma unroll
          for (unsigned w = 0; w < 8; ++w) mix[w] = seeds[seed_base + ((lane4 * 8 + w) & 15U)];

          for (unsigned i = 0; i < ETHASH_ACCESSES; ++i) {
            const uint32_t selected_lane = (i & (ETHASH_MIX_WORDS - 1)) >> 3;
            const uint32_t selected_word = i & 7U;
            const uint32_t page_candidate = lane4 == selected_lane
              ? fast_mod_dev(fnv_dev(seed0 ^ i, mix[selected_word]), dag_mod)
              : 0;
            const uint32_t page = mom_select_from_group(sg, page_candidate, sg_group4_base + selected_lane);
            const uint32_t base = page * ETHASH_MIX_WORDS + lane4 * 8;
            const auto dag_words = *reinterpret_cast<const sycl::vec<uint64_t, 4>*>(d_dag + base);
#pragma unroll
            for (unsigned w = 0; w < 4; ++w) {
              const uint64_t dag_pair = dag_words[w];
              mix[w * 2] = fnv_dev(mix[w * 2], static_cast<uint32_t>(dag_pair));
              mix[w * 2 + 1] = fnv_dev(mix[w * 2 + 1], static_cast<uint32_t>(dag_pair >> 32));
            }
          }

          const uint32_t compressed0 = fnv_fold4_dev(mix);
          const uint32_t compressed1 = fnv_fold4_dev(mix + 4);

          uint32_t compressed_mix[8];
#pragma unroll
          for (unsigned lane = 0; lane < 4; ++lane) {
            compressed_mix[lane * 2] = mom_select_from_group(sg, compressed0, sg_group4_base + lane);
            compressed_mix[lane * 2 + 1] = mom_select_from_group(sg, compressed1, sg_group4_base + lane);
          }

          if (lane4 == owner) {
            uint32_t seed[16];
#pragma unroll
            for (unsigned w = 0; w < 16; ++w) seed[w] = seeds[seed_base + w];

            uint8_t final_hash[32];
            keccak_256_seed_mix_pair_dev<INLINE_PAIR>(seed, compressed_mix, final_hash);
            if ((is_test && hash_index == 0) || meets_target_dev(final_hash, d_target)) {
              store_etchash_result(d_result, start_nonce + hash_index, final_hash, compressed_mix);
            }
          }
        }
      }
    );
  });
}
