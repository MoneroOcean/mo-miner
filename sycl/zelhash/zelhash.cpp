// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
//
// ZelHash (Equihash 125,4) GPU solver -- Wagner bucket-collision (Tromp/djezo lineage).
//
// Pipeline:
//   gen    : 2^26 entries from a personalized "ZelProof" blake2b over the 140-byte header + the
//            ZelHash "twist" (16-aligned block prefix-sum of stock hashes), masked, split into 4
//            sub-elements, ExpandArray(25) -> 25-bit collision fields.
//   round  : bucket by high BUCKBITS, collide low RESTBITS, XOR pairs -> next round.
//   recover: walk tree -> 16 leaf indices -> CompressArray(26) -> 52-byte solution.
//
// Mining runs the complete solve path: gen-fill, four collision rounds, fused final filtering, recovery, and target
// filtering. The default is_test path still uses the cheaper gen-kernel cross-check; set
// MOM_ZELHASH_SOLVE to run the full block-400000 proof recovery vector.
//
// All vendors run the same complete proof pipeline and exact vectors. Compare matching binaries,
// configurations, and steady samples before attributing a throughput difference to the backend.

#include <sycl/sycl.hpp>

#if !defined(MOM_SYCL_HAS_CUDA) && !defined(MOM_SYCL_HAS_HIP) && \
    !defined(__NVPTX__) && !defined(__AMDGCN__) && \
    !defined(MOM_SYCL_ADAPTIVECPP) && !defined(MOM_SYCL_PORTABLE_OPENCL) && \
    __has_include(<sycl/ext/intel/experimental/grf_size_properties.hpp>)
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#define MOM_ZELHASH_INTEL_LARGE_GRF 1
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "../blake2b.h"
#include "../equihash-pow.h"
#include "../lib-internal.h"
#include "../../native/consts.h"

namespace mom_zelhash {

#include "layout.inc"

#include "blake2b.inc"
// ===========================================================================================
#include "generation.inc"
#include "rounds.inc"

#include "recovery.inc"
#if defined(MOM_EQ_FORWARD_MAP)
constexpr uint32_t L0_INVERT_HEADS = NBUCKETS;
#endif
#include "l0-invert.inc"
// ===========================================================================================
#include "state.inc"

#include "proof.inc"

} // namespace mom_zelhash

using namespace mom_zelhash;

// Solve one 140-byte Flux header, or return reference generation rows in test mode.
// Mining filters complete canonical proofs by target; benchmarks count all recovered proofs.
int zelhash(const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size,
            uint8_t* const solution_out, uint64_t* const /*pnonce*/, const uint8_t* const target,
            const unsigned intensity, const bool is_test, const bool is_benchmark,
            const std::string& dev_str) {
  const bool filter_target = !is_test && !is_benchmark;
  if (!input || !solution_out || (filter_target && !target) ||
      input_size != HEADER_LEN || intensity != 1) {
    throw std::string(
      "zelhash requires input, output, a mining target, a 140-byte header, and intensity 1");
  }
  State& state = persistentDeviceState<State>(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_io();

  // Host: personalized "ZelProof" base state over all 140 header bytes (base_h/base_pending are shared).
  const BaseState bs = make_base_state(input);
  std::memcpy(state.base_h, bs.h, 8 * sizeof(uint64_t));
  std::memcpy(state.base_pending, bs.pending, 12);

  // is_test path: the default runs the gen-kernel cross-check (gen-rows dump). Setting
  // MOM_ZELHASH_SOLVE switches it to the full Wagner solve so the standalone checker can assert the
  // block-400000 solution is found. (Mining, !is_test, always runs the solver.)
  const bool solve_in_test = is_test && mom_env_flag("MOM_ZELHASH_SOLVE").value_or(false);

  if (is_test && !solve_in_test) {
    // Generation check: run the naive gen kernel, then the optimized sub-group-scan kernel, and confirm they match
    // on-device before handing the rows back. The standalone checker diffs them vs the JS oracle.
    const size_t rows_bytes = (size_t)EQUIHASH_TEST_ROWS * HASH_LENGTH;
    std::memset(state.rows, 0, rows_bytes);
    sycl_wait_and_throw(submit_gen_test<true>(state.queue, state.base_h, state.base_pending,
                                              state.rows),
                        state.device);
    std::vector<uint8_t> naive(state.rows.get(), state.rows.get() + rows_bytes);

    // CPU devices are the standards-portability oracle. Some OpenCL CPU implementations under
    // full-system emulation miscompile group scans even though the scalar kernel remains exact;
    // real-GPU lanes below still cross-check the optimized mining scan against these same rows.
    if (state.device.is_cpu()) {
      std::memcpy(solution_out, naive.data(), rows_bytes);
      return 1;
    }

    std::memset(state.rows, 0, rows_bytes);
    sycl_wait_and_throw(submit_gen_test<false>(state.queue, state.base_h, state.base_pending,
                                               state.rows),
                        state.device);
    std::vector<uint8_t> scan(state.rows.get(), state.rows.get() + rows_bytes);

    const bool match = (naive == scan);
    if (!match) {
      // Report the first differing entry so the checker / log pinpoints the scan regression.
      for (size_t e = 0; e < EQUIHASH_TEST_ROWS; ++e)
        if (std::memcmp(naive.data() + e * HASH_LENGTH, scan.data() + e * HASH_LENGTH,
                        HASH_LENGTH)) {
          std::fprintf(stderr, "zelhash: optimized scan != naive at entry %zu\n", e);
          break;
        }
    }
    // Hand back the NAIVE rows (the reference-of-record); the checker also asserts naive==scan above.
    std::memcpy(solution_out, naive.data(), (size_t)EQUIHASH_TEST_ROWS * HASH_LENGTH);
    return match ? 1 : 0;
  }

  // ---- Full Wagner solve over the gen output for this exact 140-byte header. ----
  // Profiling enables verbose counters and per-stage waits; keep both off during ordinary mining
  // and throughput measurements. Full-solve tests retain their diagnostics.
  static const bool prof = mom_env_flag("MOM_ZELHASH_PROF").value_or(false);
  const bool log = is_test || prof;
  state.ensure_arenas(log);
  std::memset(solution_out, 0, SMALL_BLOB_SOL_LEN);
  const uint64_t t_start = mom_now_ms();
  sycl::queue& q = state.queue;

  // A fixed-stage CUDA chain avoids redundant event waits: it is essential on Windows and gives a
  // smaller repeatable gain on Linux with both supported CUDA compilers. Linux HIP instead loses
  // about 1%, and generic OpenCL retains its established waits. Keep a diagnostic opt-out.
  static const bool cuda_async_enabled = mom_env_flag("MOM_ZELHASH_CUDA_ASYNC").value_or(true);
#if defined(_WIN32)
  // Windows event completion has high host/driver overhead on the measured Level Zero and HIP
  // backends. CUDA gains over 60% with DPC++ and AdaptiveCpp after exact proofs on both.
  const bool async_chain =
      !prof && (sycl_is_level_zero_gpu(state.device) || mom_is_hip(state.device) ||
                (mom_is_cuda(state.device) && cuda_async_enabled));
#else
  const bool async_chain = !prof && (sycl_is_level_zero_gpu(state.device) ||
                                     (mom_is_cuda(state.device) && cuda_async_enabled));
#endif
  const bool wait_stages = !async_chain;

  // The queue is in-order and no round needs a host-sized handoff. In normal operation enqueue the
  // reset, generation, four collision rounds, and bounded recovery as one chain. The completed
  // recovery is the single synchronization point before its tiny shared counter is read.
  // Profiling deliberately waits per stage.
  *state.ref_count = 0;
  *state.cand_count = 0;
  auto reset = q.memset(state.nslots, 0, (size_t)NSTORED_LEVELS * NBUCKETS * sizeof(uint32_t));
  if (wait_stages)
    sycl_wait_and_throw(reset, state.device);

  // The block-local pair list is distributed at a compile-time fixed count per lane. The work-group
  // size tracks the selected bucket geometry and amortizes each super-bucket across roughly eight
  // rows per lane.
  constexpr unsigned wg = ROUND_WG;

  // Level 0: gen-fill 2^26 entries into round-0 buckets (field-0 bucketing).
  const uint64_t t_gen0 = mom_now_ms();
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
  auto gen = state.gpu_compact
                 ? submit_gen_fill<true>(q, state.base_h, state.base_pending, state.level[0],
                                         state.l0_index,
                                         state.nslots + 0u * NBUCKETS, state.slot_capacity)
                 : submit_gen_fill<false>(q, state.base_h, state.base_pending, state.level[0],
                                          nullptr, state.nslots + 0u * NBUCKETS,
                                          state.slot_capacity);
#else
  auto gen =
      submit_gen_fill<false>(q, state.base_h, state.base_pending, state.level[0], nullptr,
                             state.nslots + 0u * NBUCKETS, state.slot_capacity);
#endif
  if (wait_stages)
    sycl_wait_and_throw(gen, state.device);
  // Rounds 0..3: collide field R over (bucket, RESTBITS), XOR -> level R+1.
  const uint64_t t_r0 = mom_now_ms();
  uint64_t t_round[4] = {0, 0, 0, 0};
#if defined(MOM_SYCL_HAS_CUDA)
  // The combined Intel+CUDA build must retain both layouts and select by the actual queue backend.
  [[maybe_unused]] const auto round_events = state.gpu_compact
      ? run_collision_rounds<true>(state, wg, wait_stages, prof, t_round)
      : run_collision_rounds<false>(state, wg, wait_stages, prof, t_round);
#elif defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_ADAPTIVECPP)
  [[maybe_unused]] const auto round_events =
      run_collision_rounds<true>(state, wg, wait_stages, prof, t_round);
#else
  // oneAPI Intel-only artifact: do not instantiate the bad-on-Xe 16-byte writer at all.
  [[maybe_unused]] const auto round_events =
      run_collision_rounds<false>(state, wg, wait_stages, prof, t_round);
#endif
  const uint64_t t_rounds_end = mom_now_ms();

  // The fused round-3 kernel already retained only final-field-zero pairs.
  const uint64_t t_final = mom_now_ms();

  EqLevels lv;
  for (unsigned L = 0; L < NSTORED_LEVELS; ++L)
    lv.level[L] = state.level[L];
  lv.l0_index = state.l0_index;
  auto submit_recovery = [&](uint32_t count, const uint32_t* device_count) {
#if defined(MOM_SYCL_HAS_CUDA)
    if (state.gpu_compact)
      return submit_recover<true>(q, lv, state.nslots, state.ref, count, state.cand,
                                  state.cand_count, State::CAND_CAP, wg,
                                  state.slot_capacity, device_count);
    return submit_recover<false>(q, lv, state.nslots, state.ref, count, state.cand,
                                 state.cand_count, State::CAND_CAP, wg,
                                 state.slot_capacity, device_count);
#elif defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_ADAPTIVECPP)
    return submit_recover<true>(q, lv, state.nslots, state.ref, count, state.cand,
                                state.cand_count, State::CAND_CAP, wg,
                                state.slot_capacity, device_count);
#else
    return submit_recover<false>(q, lv, state.nslots, state.ref, count, state.cand,
                                 state.cand_count, State::CAND_CAP, wg,
                                 state.slot_capacity, device_count);
#endif
  };

  if (async_chain) {
    // Queue the maximum bounded grid behind round 3 and let each work-group consult the device
    // counter. This retains one final completion point instead of a host wait before recovery.
    sycl_wait_and_throw(submit_recovery(State::CAND_CAP, state.ref_count), state.device);
  } else {
    const uint32_t ref_n = std::min<uint32_t>(*state.ref_count, State::CAND_CAP);
    if (ref_n != 0)
      sycl_wait_and_throw(submit_recovery(ref_n, nullptr), state.device);
  }
#if defined(MOM_EQ_FORWARD_MAP)
  if (state.gpu_compact) {
    const uint32_t recovered = std::min<uint32_t>(*state.cand_count, State::CAND_CAP);
    if (recovered != 0) {
      sycl_wait_and_throw(submit_l0_invert(q, state.l0_index, state.cand, recovered * (1u << K),
                                           state.invert_target, state.invert_next, state.invert_head),
                          state.device);
    }
  }
#endif
  const uint64_t t_rounds = mom_now_ms();
  if (prof) {
    std::fprintf(stderr,
                 "[eq-prof] gen-fill %llums | r0 %llu r1 %llu r2 %llu r3 %llu (rounds %llums) | "
                 "recovery %llums\n",
                 (unsigned long long)(t_r0 - t_gen0), (unsigned long long)(t_round[0] - t_r0),
                 (unsigned long long)(t_round[1] - t_round[0]),
                 (unsigned long long)(t_round[2] - t_round[1]),
                 (unsigned long long)(t_round[3] - t_round[2]),
                 (unsigned long long)(t_rounds_end - t_r0),
                 (unsigned long long)(t_rounds - t_final));
    std::fflush(stderr);
  }

  // ---- Pull only the recovered candidate list to the host. ----
  const uint64_t t_recov0 = mom_now_ms();
  uint32_t cand_count = *state.cand_count;
  const uint32_t cand_n = std::min<uint32_t>(cand_count, State::CAND_CAP);

  if (log) {
    // Survivor counters are diagnostic-only; normal mining needs just the candidate readback below.
    std::vector<uint32_t> ns_host((size_t)NSTORED_LEVELS * NBUCKETS);
    sycl_wait_and_throw(q.memcpy(ns_host.data(), state.nslots, ns_host.size() * sizeof(uint32_t)),
                        state.device);
    for (unsigned L = 0; L < NSTORED_LEVELS; ++L) {
      uint64_t sum = 0, overflow = 0;
      for (unsigned bkt = 0; bkt < NBUCKETS; ++bkt) {
        const uint32_t c = ns_host[(size_t)L * NBUCKETS + bkt];
        sum += std::min<uint32_t>(c, state.slot_capacity);
        if (c > state.slot_capacity)
          overflow += c - state.slot_capacity;
      }
      std::fprintf(stderr, "zelhash level %u survivors=%llu dropped=%llu (%.4f%%)\n", L,
                   (unsigned long long)sum, (unsigned long long)overflow,
                   100.0 * (double)overflow / (double)(sum + overflow ? sum + overflow : 1));
    }
    std::fprintf(stderr, "zelhash zero-nodes=%u candidates=%u (solve %llums)\n", *state.ref_count,
                 cand_count, (unsigned long long)(t_rounds - t_gen0));
    std::fflush(stderr);
  }

  int n_solutions = 0;
  size_t distinct_count = 0; // distinct valid Equihash proofs found this solve (pre-target-filter)
  if (cand_n) {
    // Device recovery already produced the 16 leaf indices, so copy back only this tiny list.
    std::vector<EqCandidate> cand_h(cand_n);
    sycl_wait_and_throw(q.memcpy(cand_h.data(), state.cand, cand_n * sizeof(EqCandidate)),
                        state.device);

    // De-dup distinct solutions (the same proof can surface via several candidates).
    std::vector<std::array<uint8_t, 52>> seen;
    for (uint32_t ci = 0; ci < cand_n; ++ci) {
      // Each candidate is one recovered final-zero pair; DistinctIndices requires unique leaves.
      std::vector<uint32_t> leaves(cand_h[ci].leaves, cand_h[ci].leaves + (1u << K));
      if (!mom_equihash::pow::leaves_distinct(leaves.data(), 1u << K))
        continue;

      // Canonical IndicesBefore ordering, then compress to 52 bytes.
      mom_equihash::pow::canonicalize_leaves<K>(leaves.data());
      std::array<uint8_t, 52> sol{};
      compress_indices(leaves, sol.data());
      seen.push_back(sol);
    }

    // Candidate collection order is nondeterministic. Sorting before filtering keeps the
    // offline proof buffer deterministic while counting each distinct solution once.
    std::sort(seen.begin(), seen.end());
    seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
    distinct_count = seen.size();
    // Test mode emits all proofs that fit. Benchmarks count distinct proofs without hashing
    // their share targets or writing a payload.
    if (is_test || filter_target) {
      n_solutions = static_cast<int>(mom_equihash::pow::pack_solutions(
          seen, 52, SMALL_BLOB_SOL_LEN, false, solution_out, [&](const auto& proof) -> const uint8_t* {
        return !filter_target ||
            mom_equihash::pow::meets_target(input, HEADER_LEN, proof.data(), proof.size(), target)
            ? proof.data() : nullptr;
      }));
    }
    if (log) {
      std::fprintf(stderr, "zelhash %s solutions=%d (of %zu distinct):\n",
                   is_test ? "distinct" : "target-passing", n_solutions, seen.size());
      mom_equihash::pow::log_solutions(solution_out, 52);
    }
  }

  // Count distinct proofs before the target filter so a zero-target benchmark still measures Sol/s.
  static const bool perf_log = mom_env_flag("MOM_ZELHASH_PERF").value_or(false);
  if (perf_log && !is_test) {
    state.solutionRate.record(distinct_count, [&](double solutionsPerSecond, double solvesPerSecond) {
      std::fprintf(stderr, "[zelhash] %.2f Sol/s, %.2f solve/s (%s)\n", solutionsPerSecond,
                   solvesPerSecond, dev_str.c_str());
      std::fflush(stderr);
    });
  }

  if (prof) {
    std::fprintf(stderr, "[eq-prof] host-recovery %llums (cand=%u) | total %llums\n",
                 (unsigned long long)(mom_now_ms() - t_recov0), cand_count,
                 (unsigned long long)(mom_now_ms() - t_start));
    std::fflush(stderr);
  }
  if (log) {
    std::fprintf(stderr, "zelhash solve done (%llu ms, %d target-passing, %zu distinct)\n",
                 (unsigned long long)(mom_now_ms() - t_start), n_solutions,
                 distinct_count);
    std::fflush(stderr);
  }

  // Return semantics depend on the caller:
  //  * is_test SOLVE path (the offline vector): the core's is_test handler dumps the whole
  //    out-of-band buffer when this returns 1, so report 1-on-any-solution (the payload count is in
  //    solution_out[0]).
  //  * bench and mining: report the raw solver throughput (distinct proofs found this solve,
  //    target-independent) so the core's m_hash_count accounting yields Sol/s. Mining submits the
  //    target-passing payload counted in solution_out[0].
  if (is_test)
    return n_solutions > 0 ? 1 : 0;
  return (int)distinct_count;
}
