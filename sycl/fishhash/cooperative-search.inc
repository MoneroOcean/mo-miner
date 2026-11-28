// The cooperative kernels hash on words. The header blocks before the nonce's block are the same
// for every nonce of a job, so the host hashes them once and a team's leader compresses only that
// last block: one compression per seed instead of three (180-byte FishHash header) or two (Karlsen).
struct FishSeedJob {
  uint32_t cv[8];      // chaining value before the nonce's block
  uint32_t m[16];      // that block as little-endian words, its nonce words zero
  uint32_t len;        // bytes in that block
  uint32_t flags;      // its BLAKE3 flags
  uint32_t nonce_word; // index of the first nonce word in m
  uint32_t target_top; // most significant 32 bits of the share target
};
constexpr uint32_t B3_CHUNK_START = 1, B3_CHUNK_END = 2, B3_ROOT = 8;
constexpr unsigned FISH_STAGE_WORDS = FISH_STAGE_BYTES / sizeof(uint32_t);

inline uint32_t fish_bswap32(const uint32_t v) {
  return (v >> 24) | ((v >> 8) & 0x0000FF00u) | ((v << 8) & 0x00FF0000u) | (v << 24);
}

// `header` holds the `size` hashed bytes with the eight nonce bytes at `nonce_off`; `target` is
// compared big-endian for the 180-byte FishHash header and little-endian otherwise. The nonce must
// sit at `nonce_word` of the last block: the kernels patch that fixed position.
static FishSeedJob fish_seed_job(const uint8_t* header, unsigned size, unsigned nonce_off,
                                 unsigned nonce_word, const uint8_t* target, bool bigEndian) {
  FishSeedJob job{};
  for (unsigned i = 0; i < 8; ++i)
    job.cv[i] = B3_IV[i];
  unsigned pos = 0;
  for (; size - pos > 64; pos += 64)
    b3_compress_inplace(job.cv, header + pos, 64, 0, pos ? 0 : B3_CHUNK_START);
  if (nonce_off != pos + nonce_word * 4 || nonce_off + 8 > size)
    throw std::string("fishhash nonce is not at its fixed position in the last header block");
  uint8_t block[64] = {};
  std::memcpy(block, header + pos, size - pos);
  std::memset(block + nonce_word * 4, 0, 8);
  for (unsigned i = 0; i < 16; ++i)
    job.m[i] = load32_le_dev(block + 4 * i);
  job.len = size - pos;
  job.flags = (pos ? 0 : B3_CHUNK_START) | B3_CHUNK_END | B3_ROOT;
  job.nonce_word = nonce_word;
  job.target_top = bigEndian ? fish_bswap32(load32_le_dev(target)) : load32_le_dev(target + 28);
  return job;
}

template <unsigned Team> class FishCoopKernel;
// A team owns one 128-byte DAG item. Padded teams participate in every collective.
template <unsigned Team>
static sycl::event submit_fishhash_search_coop(sycl::queue& q, const FishSeedJob& seed,
                                               uint64_t start_nonce,
                                               const uint64_t* __restrict d_dag, uint32_t intensity,
                                               const uint8_t* d_target, FishResult* d_result,
                                               bool is_test, bool bigEndian, unsigned workgroup) {
  // The kernel writes the nonce at these words: 44 bytes into the 180-byte header's last block
  // (big-endian), 32 bytes into the 40-byte header's only block.
  if (seed.nonce_word != (bigEndian ? 11u : 8u))
    throw std::string("fishhash cooperative search got an unexpected nonce position");
  const size_t global = mom_round_up_size(static_cast<uint64_t>(intensity) * Team, workgroup);
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> stage(
        sycl::range<1>((workgroup / Team) * FISH_STAGE_WORDS), h);
    h.parallel_for<FishCoopKernel<Team>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(workgroup)),
        [=](sycl::nd_item<1> it) MOM_REQD_SG(Team) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned local_id = static_cast<unsigned>(it.get_local_id(0));
          const unsigned team_local = local_id / Team;
          const auto sg = it.get_sub_group();
          const unsigned sg_lane = static_cast<unsigned>(sg.get_local_linear_id());
          const unsigned team_lane = sg_lane & (Team - 1u);
          const unsigned team_base = sg_lane - team_lane;
          const uint32_t gid = static_cast<uint32_t>(it.get_global_id(0) / Team);
          const bool active = gid < intensity;
          const uint64_t nonce = start_nonce + gid;
          // Words 0..15 hold the team's seed, words 16..23 its collapsed mix.
          uint32_t* const work = &stage[team_local * FISH_STAGE_WORDS];

          // The leader stages the BLAKE3 seed once for the whole team.
          if (team_lane == 0) {
            uint32_t m[16];
#pragma unroll
            for (unsigned i = 0; i < 16; ++i)
              m[i] = seed.m[i];
            const uint32_t lo = static_cast<uint32_t>(nonce);
            const uint32_t hi = static_cast<uint32_t>(nonce >> 32);
            if (bigEndian) {
              m[11] = fish_bswap32(hi);
              m[12] = fish_bswap32(lo);
            } else {
              m[8] = lo;
              m[9] = hi;
            }
            uint32_t out[16];
            b3_compress_words(seed.cv, m, seed.len, seed.flags, out);
#pragma unroll
            for (unsigned i = 0; i < 16; ++i)
              work[i] = out[i];
          }
          sycl::group_barrier(it.get_group());
          static_assert(Team == 8 || Team == 16);
          constexpr unsigned words = 16 / Team;
          using Chunk = sycl::vec<uint64_t, words>;
          Chunk mix;
#pragma unroll
          for (unsigned w = 0; w < words; ++w) {
            const unsigned word = (team_lane * words + w) & 7u;
            mix[w] = work[2 * word] | (static_cast<uint64_t>(work[2 * word + 1]) << 32);
          }

          constexpr uint32_t M = static_cast<uint32_t>(FULL_DATASET_NUM_ITEMS);
          for (unsigned round = 0; round < NUM_DATASET_ACCESSES; ++round) {
            const auto fetch = [&](unsigned word) -> Chunk {
              const uint32_t page = mom_select_from_group(
                  sg, static_cast<uint32_t>(mix[0]), team_base + word / words) % M;
              const uint64_t* addr = d_dag + static_cast<size_t>(page) * 16 + team_lane * words;
              if constexpr (words == 1)
                return Chunk(fish_dag_ld(addr));
              else
                return Chunk(addr[0], addr[1]);
            };
            const Chunk f0 = fetch(0), f1 = fetch(2), f2 = fetch(4);
#pragma unroll
            for (unsigned w = 0; w < words; ++w) {
              const uint64_t x = mix[w], y = f1[w];
              const uint32_t lo = fnv1_dev(static_cast<uint32_t>(x), static_cast<uint32_t>(y));
              const uint32_t hi = fnv1_dev(static_cast<uint32_t>(x >> 32),
                                           static_cast<uint32_t>(y >> 32));
              mix[w] = f0[w] * (lo | (static_cast<uint64_t>(hi) << 32)) + (x ^ f2[w]);
            }
          }

          // Each digest word folds two adjacent u64s. SIMD8 already holds both in one lane.
          const uint64_t m0 = mix[0];
          uint64_t m1;
          if constexpr (words == 1)
            m1 = mom_select_from_group(sg, m0, team_base + (team_lane | 1u));
          else
            m1 = mix[1];
          if ((team_lane * words & 1u) == 0) {
            const uint32_t folded = fnv1_dev(
                fnv1_dev(fnv1_dev(static_cast<uint32_t>(m0), static_cast<uint32_t>(m0 >> 32)),
                         static_cast<uint32_t>(m1)),
                static_cast<uint32_t>(m1 >> 32));
            work[16 + team_lane * words / 2] = folded;
          }
          sycl::group_barrier(it.get_group());

          // final = BLAKE3(seed || collapsed mix): a full first block, then 32 bytes.
          if (team_lane == 0 && active) {
            uint32_t cv[8], m[16], out[16];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
              cv[i] = B3_IV[i];
#pragma unroll
            for (unsigned i = 0; i < 16; ++i)
              m[i] = work[i];
            b3_compress_words(cv, m, 64, B3_CHUNK_START, out);
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
              cv[i] = out[i];
              m[i] = work[16 + i];
              m[8 + i] = 0;
            }
            b3_compress_words(cv, m, 32, B3_CHUNK_END | B3_ROOT, out);
            // Only a hash whose top word passes needs its bytes and the full comparison.
            const uint32_t top = bigEndian ? fish_bswap32(out[0]) : out[7];
            if ((is_test && gid == 0) || top <= seed.target_top) {
              uint8_t final_hash[32];
#pragma unroll
              for (unsigned i = 0; i < 8; ++i)
                store32_le_dev(final_hash + 4 * i, out[i]);
              const bool hit = bigEndian ? mom_meets_target_be(final_hash, d_target)
                                         : mom_meets_target_le(final_hash, d_target);
              if ((is_test && gid == 0) || hit)
                mom_store_pow_hit(d_result, nonce, final_hash);
            }
          }
        });
  });
}

class KarlsenCoopKernel;
class KarlsenCoopLargeGrfKernel;
// Share Karlsen dataset gathers across each complete team without crossing nonce boundaries.
static sycl::event submit_karlsenhashv2_search_coop(
    sycl::queue& q, const FishSeedJob& seed, uint64_t start_nonce,
    const uint64_t* __restrict d_dag, uint32_t intensity, const uint8_t* d_target,
    FishResult* d_result, bool is_test, unsigned workgroup,
    [[maybe_unused]] bool large_grf = false) {
  // The kernel writes the nonce eight bytes into the 80-byte header's second block.
  if (seed.nonce_word != 2u)
    throw std::string("karlsenhashv2 cooperative search got an unexpected nonce position");
  const size_t global = mom_round_up_size(static_cast<uint64_t>(intensity) * FISH_TEAM, workgroup);
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> stage(
        sycl::range<1>((workgroup / FISH_TEAM) * FISH_STAGE_WORDS), h);
    const sycl::nd_range<1> range{sycl::range<1>(global), sycl::range<1>(workgroup)};
    const auto search = [=](sycl::nd_item<1> it) MOM_REQD_SG_16 MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned local_id = static_cast<unsigned>(it.get_local_id(0));
          const unsigned team_local = local_id / FISH_TEAM;
          const auto sg = it.get_sub_group();
          const unsigned sg_lane = static_cast<unsigned>(sg.get_local_linear_id());
          const unsigned team_lane = sg_lane & (FISH_TEAM - 1u);
          const unsigned team_base = sg_lane - team_lane;
          const uint32_t gid = static_cast<uint32_t>(it.get_global_id(0) / FISH_TEAM);
          const bool active = gid < intensity;
          const uint64_t nonce = start_nonce + gid;
          // Words 0..7 hold the team's 32-byte pow hash, later its collapsed mix.
          uint32_t* const work = &stage[team_local * FISH_STAGE_WORDS];

          if (team_lane == 0) {
            uint32_t m[16];
#pragma unroll
            for (unsigned i = 0; i < 16; ++i)
              m[i] = seed.m[i];
            m[2] = static_cast<uint32_t>(nonce);
            m[3] = static_cast<uint32_t>(nonce >> 32);
            uint32_t out[16];
            b3_compress_words(seed.cv, m, seed.len, seed.flags, out);
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
              work[i] = out[i];
          }
          sycl::group_barrier(it.get_group());
          // The 64-byte seed is the pow hash followed by zeros, repeated over the sixteen lanes.
          const unsigned word = team_lane & 7u;
          uint64_t mix = 0;
          if (word < 4)
            mix = work[2 * word] | (static_cast<uint64_t>(work[2 * word + 1]) << 32);

          constexpr uint32_t M = static_cast<uint32_t>(FULL_DATASET_NUM_ITEMS);
          for (uint32_t round = 0; round < static_cast<uint32_t>(NUM_DATASET_ACCESSES); ++round) {
            // Fold the two u64 lanes that form each four-u32 mix group with one shuffle, then select the
            // even lane representing each required group. This is exactly mg[0]^mg[3]^mg[6],
            // mg[1]^mg[4]^mg[7], and mg[2]^mg[5], but uses 9 rather than 12 u32 subgroup exchanges.
            const uint32_t lane_mix = static_cast<uint32_t>(mix) ^ static_cast<uint32_t>(mix >> 32);
            const uint32_t pair_mix =
                lane_mix ^ mom_select_from_group(sg, lane_mix, team_base + (team_lane ^ 1u));
            const uint32_t p0 = (mom_select_from_group(sg, pair_mix, team_base + 0u) ^
                                 mom_select_from_group(sg, pair_mix, team_base + 6u) ^
                                 mom_select_from_group(sg, pair_mix, team_base + 12u)) %
                                M;
            const uint32_t p1 = (mom_select_from_group(sg, pair_mix, team_base + 2u) ^
                                 mom_select_from_group(sg, pair_mix, team_base + 8u) ^
                                 mom_select_from_group(sg, pair_mix, team_base + 14u)) %
                                M;
            const uint32_t p2 = (mom_select_from_group(sg, pair_mix, team_base + 4u) ^
                                 mom_select_from_group(sg, pair_mix, team_base + 10u) ^ round) %
                                M;
            const uint64_t f0 =
                fish_dag_ld(d_dag + static_cast<size_t>(p0) * FISH_TEAM + team_lane);
            const uint64_t f1 =
                fish_dag_ld(d_dag + static_cast<size_t>(p1) * FISH_TEAM + team_lane);
            const uint64_t f2 =
                fish_dag_ld(d_dag + static_cast<size_t>(p2) * FISH_TEAM + team_lane);
            const uint32_t blo = fnv1_dev(static_cast<uint32_t>(mix), static_cast<uint32_t>(f1));
            const uint32_t bhi =
                fnv1_dev(static_cast<uint32_t>(mix >> 32), static_cast<uint32_t>(f1 >> 32));
            const uint64_t b = static_cast<uint64_t>(blo) | (static_cast<uint64_t>(bhi) << 32);
            mix = f0 * b + (mix ^ f2);
          }

          const uint64_t m1 = mom_select_from_group(sg, mix, team_base + (team_lane | 1u));
          if ((team_lane & 1u) == 0) {
            const uint32_t folded = fnv1_dev(
                fnv1_dev(fnv1_dev(static_cast<uint32_t>(mix), static_cast<uint32_t>(mix >> 32)),
                         static_cast<uint32_t>(m1)),
                static_cast<uint32_t>(m1 >> 32));
            work[team_lane >> 1] = folded;
          }
          sycl::group_barrier(it.get_group());
          // final = BLAKE3(collapsed mix), one 32-byte block.
          if (team_lane == 0 && active) {
            uint32_t cv[8], m[16], out[16];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
              cv[i] = B3_IV[i];
              m[i] = work[i];
              m[8 + i] = 0;
            }
            b3_compress_words(cv, m, 32, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT, out);
            // Only a hash whose top word passes needs its bytes and the full comparison.
            if ((is_test && gid == 0) || out[7] <= seed.target_top) {
              uint8_t final_hash[32];
#pragma unroll
              for (unsigned i = 0; i < 8; ++i)
                store32_le_dev(final_hash + 4 * i, out[i]);
              if ((is_test && gid == 0) || mom_meets_target_le(final_hash, d_target))
                mom_store_pow_hit(d_result, nonce, final_hash);
            }
          }
        };
#if defined(MOM_FISHHASH_INTEL_LARGE_GRF)
    // The 256-entry register file is what lets BMG-G21 run this kernel in 64-thread groups (see
    // State::cooperative_workgroup).
    if (large_grf)
      return h.parallel_for<KarlsenCoopLargeGrfKernel>(
          range,
          sycl::ext::oneapi::experimental::properties{
              sycl::ext::intel::experimental::grf_size<256>},
          search);
#endif
    h.parallel_for<KarlsenCoopKernel>(range, search);
  });
}

// ---- GPU DAG generation: compute all 37.7M dataset items into the 4.6 GB buffer (once per device) ----
