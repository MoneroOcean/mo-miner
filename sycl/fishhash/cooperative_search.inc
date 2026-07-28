class FishCoopKernel;
static sycl::event submit_fishhash_search_coop(
  sycl::queue& q, MomKernelBundle& kb, const uint8_t* d_input, unsigned input_size, uint64_t start_nonce,
  const uint64_t* __restrict d_dag, uint32_t intensity, const uint8_t* d_target,
  FishResult* d_result, bool is_test, unsigned nonce_off, bool nonce_be, bool target_be,
  unsigned workgroup
) {
  const size_t work_items = static_cast<size_t>(intensity) * FISH_TEAM;
  const size_t global = (work_items + workgroup - 1) / workgroup * workgroup;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    sycl::local_accessor<uint8_t, 1> stage(
      sycl::range<1>((workgroup / FISH_TEAM) * FISH_STAGE_BYTES), h);
    h.parallel_for<FishCoopKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(workgroup)),
      [=](sycl::nd_item<1> it) MOM_REQD_SG_16 {
      const unsigned local_id = static_cast<unsigned>(it.get_local_id(0));
      const unsigned team_local = local_id / FISH_TEAM;
      const auto sg = it.get_sub_group();
      const unsigned sg_lane = static_cast<unsigned>(sg.get_local_linear_id());
      const unsigned team_lane = sg_lane & (FISH_TEAM - 1u);
      const unsigned team_base = sg_lane - team_lane;
      const uint32_t gid = static_cast<uint32_t>(it.get_global_id(0) / FISH_TEAM);
      const bool active = gid < intensity;
      const uint64_t nonce = start_nonce + gid;
      uint8_t* const work = &stage[team_local * FISH_STAGE_BYTES];

      const unsigned hlen = input_size <= FISH_STAGE_BYTES ? input_size : FISH_STAGE_BYTES;
      for (unsigned i = team_lane; i < hlen; i += FISH_TEAM) work[i] = d_input[i];
      if (team_lane == 0) {
        if (nonce_be) {
          for (unsigned i = 0; i < 8; ++i) work[nonce_off + i] = static_cast<uint8_t>(nonce >> (8 * (7 - i)));
        } else {
          for (unsigned i = 0; i < 8; ++i) work[nonce_off + i] = static_cast<uint8_t>(nonce >> (8 * i));
        }
      }
      sycl::group_barrier(it.get_group());

      // The leader performs the serial BLAKE3 prefix; SLM then distributes its eight u64 words to
      // the 16 lanes (the second half repeats the seed, matching mix={seed,seed}).
      if (team_lane == 0) {
        uint8_t seed[64];
        blake3_dev(seed, 64, work, hlen);
        for (unsigned i = 0; i < 64; ++i) work[i] = seed[i];
      }
      sycl::group_barrier(it.get_group());
      const uint64_t seed_word = load64_le_dev(work + 8 * (team_lane & 7u));
      uint64_t mix = seed_word;

      constexpr uint32_t M = static_cast<uint32_t>(FULL_DATASET_NUM_ITEMS);
      for (unsigned round = 0; round < NUM_DATASET_ACCESSES; ++round) {
        const uint32_t own_low = static_cast<uint32_t>(mix);
        const uint32_t p0 = mom_select_from_group(sg, own_low, team_base + 0u) % M;
        const uint32_t p1 = mom_select_from_group(sg, own_low, team_base + 2u) % M;
        const uint32_t p2 = mom_select_from_group(sg, own_low, team_base + 4u) % M;
        const uint64_t f0 = fish_dag_ld(d_dag + static_cast<size_t>(p0) * FISH_TEAM + team_lane);
        const uint64_t f1 = fish_dag_ld(d_dag + static_cast<size_t>(p1) * FISH_TEAM + team_lane);
        const uint64_t f2 = fish_dag_ld(d_dag + static_cast<size_t>(p2) * FISH_TEAM + team_lane);
        const uint32_t blo = fnv1_dev(static_cast<uint32_t>(mix), static_cast<uint32_t>(f1));
        const uint32_t bhi = fnv1_dev(static_cast<uint32_t>(mix >> 32), static_cast<uint32_t>(f1 >> 32));
        const uint64_t b = static_cast<uint64_t>(blo) | (static_cast<uint64_t>(bhi) << 32);
        mix = f0 * b + (mix ^ f2);
      }

      // Each even lane already owns the first u64 in its adjacent pair, so only the second needs a
      // shuffle. The leader reuses the staged seed as final_data[0..63] and appends the collapsed
      // 32-byte mix at [64..95].
      // Group collectives must be encountered by every lane in converged control flow. Fetch the
      // neighbour unconditionally, then let only the even lanes publish each four-u32 fold.
      const uint64_t m1 = mom_select_from_group(sg, mix, team_base + (team_lane | 1u));
      if ((team_lane & 1u) == 0) {
        const uint32_t folded = fnv1_dev(fnv1_dev(fnv1_dev(
          static_cast<uint32_t>(mix), static_cast<uint32_t>(mix >> 32)),
          static_cast<uint32_t>(m1)), static_cast<uint32_t>(m1 >> 32));
        store32_le_dev(work + 64 + 2 * team_lane, folded);
      }
      sycl::group_barrier(it.get_group());

      if (team_lane == 0 && active) {
        uint8_t final_hash[32];
        blake3_dev(final_hash, 32, work, 96);
        const bool hit = target_be ? meets_target_be_dev(final_hash, d_target)
                                   : meets_target_le_dev(final_hash, d_target);
        if ((is_test && gid == 0) || hit) store_fish_result(d_result, nonce, final_hash);
      }
    });
  });
}

class KarlsenCoopKernel;
static sycl::event submit_karlsenhashv2_search_coop(
  sycl::queue& q, MomKernelBundle& kb, const uint8_t* d_input, unsigned input_size, uint64_t start_nonce,
  const uint64_t* __restrict d_dag, uint32_t intensity, const uint8_t* d_target,
  FishResult* d_result, bool is_test, unsigned workgroup
) {
  const size_t work_items = static_cast<size_t>(intensity) * FISH_TEAM;
  const size_t global = (work_items + workgroup - 1) / workgroup * workgroup;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    sycl::local_accessor<uint8_t, 1> stage(
      sycl::range<1>((workgroup / FISH_TEAM) * FISH_STAGE_BYTES), h);
    h.parallel_for<KarlsenCoopKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(workgroup)),
      [=](sycl::nd_item<1> it) MOM_REQD_SG_16 {
      const unsigned local_id = static_cast<unsigned>(it.get_local_id(0));
      const unsigned team_local = local_id / FISH_TEAM;
      const auto sg = it.get_sub_group();
      const unsigned sg_lane = static_cast<unsigned>(sg.get_local_linear_id());
      const unsigned team_lane = sg_lane & (FISH_TEAM - 1u);
      const unsigned team_base = sg_lane - team_lane;
      const uint32_t gid = static_cast<uint32_t>(it.get_global_id(0) / FISH_TEAM);
      const bool active = gid < intensity;
      const uint64_t nonce = start_nonce + gid;
      uint8_t* const work = &stage[team_local * FISH_STAGE_BYTES];

      for (unsigned i = team_lane; i < 80; i += FISH_TEAM)
        work[i] = i < input_size ? d_input[i] : 0;
      if (team_lane == 0)
        for (unsigned i = 0; i < 8; ++i) work[72 + i] = static_cast<uint8_t>(nonce >> (8 * i));
      sycl::group_barrier(it.get_group());
      if (team_lane == 0) {
        uint8_t pow_hash[32];
        blake3_dev(pow_hash, 32, work, 80);
        for (unsigned i = 0; i < 32; ++i) work[i] = pow_hash[i];
        for (unsigned i = 32; i < 64; ++i) work[i] = 0;
      }
      sycl::group_barrier(it.get_group());
      uint64_t mix = load64_le_dev(work + 8 * (team_lane & 7u));

      constexpr uint32_t M = static_cast<uint32_t>(FULL_DATASET_NUM_ITEMS);
      for (uint32_t round = 0; round < static_cast<uint32_t>(NUM_DATASET_ACCESSES); ++round) {
        // Fold the two u64 lanes that form each four-u32 mix group with one shuffle, then select the
        // even lane representing each required group. This is exactly mg[0]^mg[3]^mg[6],
        // mg[1]^mg[4]^mg[7], and mg[2]^mg[5], but uses 9 rather than 12 u32 subgroup exchanges.
        const uint32_t lane_mix = static_cast<uint32_t>(mix) ^ static_cast<uint32_t>(mix >> 32);
        const uint32_t pair_mix = lane_mix ^ mom_select_from_group(
          sg, lane_mix, team_base + (team_lane ^ 1u));
        const uint32_t p0 = (mom_select_from_group(sg, pair_mix, team_base + 0u) ^
                             mom_select_from_group(sg, pair_mix, team_base + 6u) ^
                             mom_select_from_group(sg, pair_mix, team_base + 12u)) % M;
        const uint32_t p1 = (mom_select_from_group(sg, pair_mix, team_base + 2u) ^
                             mom_select_from_group(sg, pair_mix, team_base + 8u) ^
                             mom_select_from_group(sg, pair_mix, team_base + 14u)) % M;
        const uint32_t p2 = (mom_select_from_group(sg, pair_mix, team_base + 4u) ^
                             mom_select_from_group(sg, pair_mix, team_base + 10u) ^ round) % M;
        const uint64_t f0 = fish_dag_ld(d_dag + static_cast<size_t>(p0) * FISH_TEAM + team_lane);
        const uint64_t f1 = fish_dag_ld(d_dag + static_cast<size_t>(p1) * FISH_TEAM + team_lane);
        const uint64_t f2 = fish_dag_ld(d_dag + static_cast<size_t>(p2) * FISH_TEAM + team_lane);
        const uint32_t blo = fnv1_dev(static_cast<uint32_t>(mix), static_cast<uint32_t>(f1));
        const uint32_t bhi = fnv1_dev(static_cast<uint32_t>(mix >> 32), static_cast<uint32_t>(f1 >> 32));
        const uint64_t b = static_cast<uint64_t>(blo) | (static_cast<uint64_t>(bhi) << 32);
        mix = f0 * b + (mix ^ f2);
      }

      const uint64_t m1 = mom_select_from_group(sg, mix, team_base + (team_lane | 1u));
      if ((team_lane & 1u) == 0) {
        const uint32_t folded = fnv1_dev(fnv1_dev(fnv1_dev(
          static_cast<uint32_t>(mix), static_cast<uint32_t>(mix >> 32)),
          static_cast<uint32_t>(m1)), static_cast<uint32_t>(m1 >> 32));
        store32_le_dev(work + 2 * team_lane, folded);
      }
      sycl::group_barrier(it.get_group());
      if (team_lane == 0 && active) {
        uint8_t final_hash[32];
        blake3_dev(final_hash, 32, work, 32);
        if ((is_test && gid == 0) || meets_target_le_dev(final_hash, d_target))
          store_fish_result(d_result, nonce, final_hash);
      }
    });
  });
}

// ---- GPU DAG generation: compute all 37.7M dataset items into the 4.6 GB buffer (once per device) ----
