template <bool FULL> class FishKernel;

// Search kernel. FULL=true: gather 128B items from the prebuilt 4.6 GB DAG (fast, mining/bench).
// FULL=false: recompute each item from the 72 MB light cache (lazy; validation / low-mem only).
// d_input is the header; the 8-byte nonce goes at nonce_off (LE for the offline @32 test, BE @0 for
// live Iron Fish). target_be selects BE compare (Iron Fish) vs LE (offline test).
template <bool FULL>
static sycl::event submit_fishhash_search(
  sycl::queue& q, MomKernelBundle& kb, const uint8_t* d_input, unsigned input_size, uint64_t start_nonce,
  const uint64_t* __restrict d_cache, const uint64_t* __restrict d_dag, uint32_t intensity,
  const uint8_t* d_target, FishResult* d_result, bool is_test,
  unsigned nonce_off, bool nonce_be, bool target_be
) {
  constexpr unsigned WG = 128;  // Latency-bound gathers benefit from more in-flight subgroups.
  const size_t global = (static_cast<size_t>(intensity) + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    h.parallel_for<FishKernel<FULL>>(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
      const uint32_t gid = (uint32_t)it.get_global_id(0);
      if (gid >= intensity) return;
      const uint64_t nonce = start_nonce + gid;

      uint8_t header[180];
      const unsigned hlen = input_size <= 180 ? input_size : 180;
      for (unsigned i = 0; i < hlen; ++i) header[i] = d_input[i];
      if (nonce_be) for (unsigned i = 0; i < 8; ++i) header[nonce_off + i] = (uint8_t)(nonce >> (8*(7-i)));  // BE (Iron Fish @0)
      else          for (unsigned i = 0; i < 8; ++i) header[nonce_off + i] = (uint8_t)(nonce >> (8*i));        // LE (offline test @32)

      uint8_t seed[64]; blake3_dev(seed, 64, header, hlen);

      // mix(1024b) = {seed, seed} as 32 u32
      uint32_t mix[32];
      for (unsigned i = 0; i < 16; ++i) mix[i] = load32_le_dev(seed + 4*i);
      for (unsigned i = 0; i < 16; ++i) mix[16 + i] = mix[i];

      const uint32_t M = (uint32_t)FULL_DATASET_NUM_ITEMS;
      for (int i = 0; i < NUM_DATASET_ACCESSES; ++i) {
        const uint32_t p0 = mix[0] % M, p1 = mix[4] % M, p2 = mix[8] % M;
        // Fold the fnv/mul-add per 16-u64 item element as it streams in, instead of materializing
        // f0/f1/f2[16] + f1w/f2w[32] all at once. Each output (mix[2j],mix[2j+1]) depends only on
        // f0[j]/f1[j]/f2[j] and its own old mix slot, so only 3 u64 stay live per element -- a much
        // smaller register working set that lets more warps reside per SM (latency hiding on the
        // random gather). Bit-identical to the original array form.
        uint64_t lazy0[16], lazy1[16], lazy2[16];
        if constexpr (!FULL) {
          fish_dataset_item(d_cache, p0, lazy0);
          fish_dataset_item(d_cache, p1, lazy1);
          fish_dataset_item(d_cache, p2, lazy2);
        }
#if defined(__NVPTX__)
        // NVIDIA (nvptx) only: for the FULL DAG path, issue all 48 independent 64-bit item loads up
        // front in a tight batch (no arithmetic interleaved) so the LSU keeps many gather misses in
        // flight per warp before any is consumed -- memory-level parallelism on the random 4.6 GB
        // gather, the structure's latency limiter. +19% on L4. Off the FULL path this block is
        // skipped (lazy items already live). The 48-u64 batch buffer spills registers on Intel
        // (spir64) and collapses throughput ~13x, so spir64 takes the interleaved #else path below.
        uint64_t b0[16], b1[16], b2[16];
        if constexpr (FULL) {
          const uint64_t* __restrict q0 = d_dag + (size_t)p0 * 16;
          const uint64_t* __restrict q1 = d_dag + (size_t)p1 * 16;
          const uint64_t* __restrict q2 = d_dag + (size_t)p2 * 16;
          for (unsigned j = 0; j < 16; ++j) b0[j] = fish_dag_ld(q0 + j);
          for (unsigned j = 0; j < 16; ++j) b1[j] = fish_dag_ld(q1 + j);
          for (unsigned j = 0; j < 16; ++j) b2[j] = fish_dag_ld(q2 + j);
        }
#endif
        for (unsigned j = 0; j < 16; ++j) {
          uint64_t f0j, f1j, f2j;
          if constexpr (FULL) {
#if defined(__NVPTX__)
            f0j = b0[j];
            f1j = b1[j];
            f2j = b2[j];
#else
            // Intel (spir64): read each item element directly inside the fold so only 3 u64 stay
            // live per iteration -- smaller register footprint => more resident warps.
            f0j = fish_dag_ld(d_dag + (size_t)p0 * 16 + j);
            f1j = fish_dag_ld(d_dag + (size_t)p1 * 16 + j);
            f2j = fish_dag_ld(d_dag + (size_t)p2 * 16 + j);
#endif
          } else {
            f0j = lazy0[j]; f1j = lazy1[j]; f2j = lazy2[j];
          }
          const uint32_t blo = fnv1_dev(mix[2*j],   (uint32_t)f1j);
          const uint32_t bhi = fnv1_dev(mix[2*j+1], (uint32_t)(f1j >> 32));
          const uint32_t clo = mix[2*j]   ^ (uint32_t)f2j;
          const uint32_t chi = mix[2*j+1] ^ (uint32_t)(f2j >> 32);
          const uint64_t b = (uint64_t)blo | ((uint64_t)bhi << 32);
          const uint64_t c = (uint64_t)clo | ((uint64_t)chi << 32);
          const uint64_t v = f0j * b + c;
          mix[2*j] = (uint32_t)v; mix[2*j+1] = (uint32_t)(v >> 32);
        }
      }
      uint8_t mix_hash[32];
      for (unsigned i = 0; i < 32; i += 4) {
        uint32_t h = fnv1_dev(fnv1_dev(fnv1_dev(mix[i], mix[i+1]), mix[i+2]), mix[i+3]);
        store32_le_dev(mix_hash + i, h);
      }
      uint8_t final_data[96];
      for (unsigned i = 0; i < 64; ++i) final_data[i] = seed[i];
      for (unsigned i = 0; i < 32; ++i) final_data[64 + i] = mix_hash[i];
      uint8_t final_hash[32]; blake3_dev(final_hash, 32, final_data, 96);

      const bool hit = target_be ? meets_target_be_dev(final_hash, d_target) : meets_target_le_dev(final_hash, d_target);
      if ((is_test && gid == 0) || hit)
        store_fish_result(d_result, nonce, final_hash);
    });
  });
}

// ---- KarlsenHashV2 (FishHashPlus): SAME 4.6 GB DAG, different index derivation + BLAKE3 wrapping.
// powHash = BLAKE3_32(prePow(32)||ts(8 LE)||32 zeros||nonce(8 LE), 80B); seed = powHash(32)||zeros(32);
// mix = seed||seed; 32 rounds (mixGroup-fold index); collapse 32->8 u32; final = BLAKE3_32(mix_hash).
// Input = 80-byte Kaspa-style blob, 8-byte nonce at offset 72.
template <bool FULL> class KarlsenKernel;
template <bool FULL>
static sycl::event submit_karlsenhashv2_search(
  sycl::queue& q, MomKernelBundle& kb, const uint8_t* d_input, unsigned input_size, uint64_t start_nonce,
  const uint64_t* __restrict d_cache, const uint64_t* __restrict d_dag, uint32_t intensity,
  const uint8_t* d_target, FishResult* d_result, bool is_test
) {
  constexpr unsigned WG = 128;
  const size_t global = (static_cast<size_t>(intensity) + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    h.parallel_for<KarlsenKernel<FULL>>(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
      const uint32_t gid = (uint32_t)it.get_global_id(0);
      if (gid >= intensity) return;
      const uint64_t nonce = start_nonce + gid;

      uint8_t pre[80];
      for (unsigned i = 0; i < 80; ++i) pre[i] = (i < input_size) ? d_input[i] : 0;
      for (unsigned i = 0; i < 8; ++i) pre[72 + i] = (uint8_t)(nonce >> (8*i));  // nonce@72 LE

      uint8_t powh[32]; blake3_dev(powh, 32, pre, 80);

      uint32_t mix[32];                                   // seed = powHash(32) || zeros(32); mix = seed||seed
      for (unsigned i = 0; i < 8; ++i) mix[i] = load32_le_dev(powh + 4*i);
      for (unsigned i = 8; i < 16; ++i) mix[i] = 0;
      for (unsigned i = 0; i < 16; ++i) mix[16 + i] = mix[i];

      const uint32_t M = (uint32_t)FULL_DATASET_NUM_ITEMS;
      for (uint32_t rnd = 0; rnd < (uint32_t)NUM_DATASET_ACCESSES; ++rnd) {
        uint32_t mg[8];
        for (unsigned c = 0; c < 8; ++c) mg[c] = mix[4*c] ^ mix[4*c+1] ^ mix[4*c+2] ^ mix[4*c+3];
        const uint32_t p0 = (mg[0] ^ mg[3] ^ mg[6]) % M;
        const uint32_t p1 = (mg[1] ^ mg[4] ^ mg[7]) % M;
        const uint32_t p2 = (mg[2] ^ mg[5] ^ rnd) % M;
        uint64_t lazy0[16], lazy1[16], lazy2[16];
        if constexpr (!FULL) {
          fish_dataset_item(d_cache, p0, lazy0);
          fish_dataset_item(d_cache, p1, lazy1);
          fish_dataset_item(d_cache, p2, lazy2);
        }
#if defined(__NVPTX__)
        uint64_t f0[16], f1[16], f2[16];
        if constexpr (FULL) {
          // NVIDIA (nvptx) only: batch the 48 independent gather loads split per item so the LSU
          // keeps many random 4.6 GB DAG misses in flight per warp before consumption -- same
          // latency-hiding win as FishHash. Spills registers on Intel (spir64), which takes the
          // streaming path below instead.
          const uint64_t* __restrict q0 = d_dag + (size_t)p0 * 16;
          const uint64_t* __restrict q1 = d_dag + (size_t)p1 * 16;
          const uint64_t* __restrict q2 = d_dag + (size_t)p2 * 16;
          fish_dag_prefetch(q0);
          fish_dag_prefetch(q1);
          fish_dag_prefetch(q2);
          for (unsigned k = 0; k < 16; ++k) f0[k] = fish_dag_ld(q0 + k);
          for (unsigned k = 0; k < 16; ++k) f1[k] = fish_dag_ld(q1 + k);
          for (unsigned k = 0; k < 16; ++k) f2[k] = fish_dag_ld(q2 + k);
        }
#endif
        for (unsigned j = 0; j < 16; ++j) {
          uint64_t f0j, f1j, f2j;
          if constexpr (FULL) {
#if defined(__NVPTX__)
            f0j = f0[j]; f1j = f1[j]; f2j = f2[j];
#else
            // Keep only one element from each random DAG record live while folding it. The former
            // f0/f1/f2 + f1w/f2w arrays consumed hundreds of bytes of private storage per work-item
            // and could spill catastrophically on otherwise portable SPIR-V targets.
            f0j = fish_dag_ld(d_dag + (size_t)p0 * 16 + j);
            f1j = fish_dag_ld(d_dag + (size_t)p1 * 16 + j);
            f2j = fish_dag_ld(d_dag + (size_t)p2 * 16 + j);
#endif
          } else {
            f0j = lazy0[j]; f1j = lazy1[j]; f2j = lazy2[j];
          }
          const uint32_t blo = fnv1_dev(mix[2*j], (uint32_t)f1j);
          const uint32_t bhi = fnv1_dev(mix[2*j+1], (uint32_t)(f1j >> 32));
          const uint32_t clo = mix[2*j] ^ (uint32_t)f2j;
          const uint32_t chi = mix[2*j+1] ^ (uint32_t)(f2j >> 32);
          const uint64_t b = (uint64_t)blo | ((uint64_t)bhi << 32);
          const uint64_t c = (uint64_t)clo | ((uint64_t)chi << 32);
          const uint64_t v = f0j * b + c;
          mix[2*j] = (uint32_t)v; mix[2*j+1] = (uint32_t)(v >> 32);
        }
      }
      uint8_t mix_hash[32];
      for (unsigned i = 0; i < 32; i += 4) {
        uint32_t hh = fnv1_dev(fnv1_dev(fnv1_dev(mix[i], mix[i+1]), mix[i+2]), mix[i+3]);
        store32_le_dev(mix_hash + i, hh);
      }
      uint8_t final_hash[32]; blake3_dev(final_hash, 32, mix_hash, 32);

      if ((is_test && gid == 0) || meets_target_le_dev(final_hash, d_target))
        store_fish_result(d_result, nonce, final_hash);
    });
  });
}

// ---- Cooperative full-DAG search ---------------------------------------------------------------
// A scalar work-item owns all 16 u64 mix elements in the portable path above. On a random 4.6-GiB
// DAG that creates a long 48-load dependency window and a large register footprint. The production
// path instead maps one hash to a logical 16-lane team: each lane owns one u64 mix element and the
// three 128-byte DAG records are loaded as three coalesced 16-lane transactions. CUDA and RDNA run
// two teams per native 32-lane warp/wave; Intel requests a native 16-lane subgroup. Only the three
// data-dependent record indices cross lanes, through native shuffle/select operations.
constexpr unsigned FISH_TEAM = 16;
constexpr unsigned FISH_STAGE_BYTES = 180;  // header first, then reused as seed||collapsed mix
