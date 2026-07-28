int zelhash(
  const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size,
  uint8_t* const solution_out, uint64_t* const /*pnonce*/, const uint8_t* const target,
  const unsigned /*intensity*/, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < HEADER_LEN) throw std::string("Bad zelhash input length");
  ZelHashState& state = zelhash_state(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_io();

  // Host: personalized "ZelProof" base state over all 140 header bytes (base_h/base_pending are shared).
  const BaseState bs = make_base_state(input);
  std::memcpy(state.base_h, bs.h, 8 * sizeof(uint64_t));
  std::memcpy(state.base_pending, bs.pending, 12);

  // is_test path: the default runs the gen-kernel cross-check (gen-rows dump). Setting
  // MOM_ZELHASH_SOLVE switches it to the full Wagner solve so the standalone checker can assert the
  // block-400000 solution is found. (Mining, !is_test, always runs the solver.)
  const bool solve_in_test = is_test && std::getenv("MOM_ZELHASH_SOLVE") != nullptr;

  if (is_test && !solve_in_test) {
    // M1: run the naive gen kernel, then the optimized sub-group-scan kernel, and confirm they match
    // on-device before handing the rows back. The standalone checker diffs them vs the JS oracle.
    const size_t rows_bytes = (size_t)EQUIHASH_TEST_ROWS * HASH_LENGTH;
    std::memset(state.rows, 0, rows_bytes);
    sycl_wait_and_throw(submit_gen_test<true>(state.queue, *state.bundle, state.base_h, state.base_pending, state.rows), state.device);
    std::vector<uint8_t> naive(state.rows, state.rows + rows_bytes);

    std::memset(state.rows, 0, rows_bytes);
    sycl_wait_and_throw(submit_gen_test<false>(state.queue, *state.bundle, state.base_h, state.base_pending, state.rows), state.device);
    std::vector<uint8_t> scan(state.rows, state.rows + rows_bytes);

    const bool match = (naive == scan);
    if (!match) {
      // Report the first differing entry so the checker / log pinpoints the scan regression.
      for (size_t e = 0; e < EQUIHASH_TEST_ROWS; ++e)
        if (std::memcmp(naive.data() + e * HASH_LENGTH, scan.data() + e * HASH_LENGTH, HASH_LENGTH)) {
          std::fprintf(stderr, "zelhash M1: optimized scan != naive at entry %zu\n", e);
          break;
        }
    }
    // Hand back the NAIVE rows (the reference-of-record); the checker also asserts naive==scan above.
    std::memcpy(solution_out, naive.data(), (size_t)EQUIHASH_TEST_ROWS * HASH_LENGTH);
    (void)is_benchmark;
    return match ? 1 : 0;
  }

  // ---- M2/M3 full Wagner solve over the gen output for this exact 140-byte header. -----------------
  state.ensure_arenas(!is_benchmark);
  std::memset(solution_out, 0, SMALL_BLOB_SOL_LEN);
  const bool log = !is_benchmark;
  // Per-phase profiler: MOM_ZELHASH_PROF prints a gen-fill / per-round / device- and host-recovery
  // ms breakdown even under bench (where `log` is off). It inserts an extra barrier + per-round host
  // waits, so it is a SEPARATE gate from MOM_ZELHASH_PERF (the Sol/s logger) -- enabling it perturbs
  // throughput slightly, so leave it off when measuring Sol/s.
  static const bool prof = std::getenv("MOM_ZELHASH_PROF") != nullptr;
  const uint64_t t_start = ZelHashState::now_ms();
  sycl::queue& q = state.queue;

  // A fixed-stage CUDA chain avoids redundant event waits: it is essential on Windows and gives a
  // smaller repeatable gain on Linux with both supported CUDA compilers. Linux HIP instead loses
  // about 1%, and generic OpenCL retains its established waits. Keep a diagnostic opt-out.
  static const bool cuda_async_enabled = [] {
    const char* value = std::getenv("MOM_ZELHASH_CUDA_ASYNC");
    return !value || (std::strcmp(value, "0") && std::strcmp(value, "false"));
  }();
#if defined(_WIN32)
  // Windows event completion has high host/driver overhead on the measured Level Zero and HIP
  // backends. CUDA gains over 60% with DPC++ and AdaptiveCpp after exact proofs on both.
  const bool async_chain = !prof &&
    (sycl_is_level_zero_gpu(state.device) || mom_is_hip(state.device) ||
     (mom_is_cuda(state.device) && cuda_async_enabled));
#else
  const bool async_chain = !prof &&
    (sycl_is_level_zero_gpu(state.device) || (mom_is_cuda(state.device) && cuda_async_enabled));
#endif
  const bool wait_stages = !async_chain;

  // The queue is in-order and no round needs a host-sized handoff. In normal operation enqueue the
  // reset, generation, four collision rounds, and bounded recovery as one chain. The completed
  // recovery is the single synchronization point before its tiny shared counter is read.
  // Profiling deliberately waits per stage.
  *state.ref_count = 0;
  *state.cand_count = 0;
  auto reset = q.memset(state.nslots, 0,
    (size_t)NSTORED_LEVELS * NBUCKETS * sizeof(uint32_t));
  if (wait_stages) sycl_wait_and_throw(reset, state.device);

  // The block-local pair list is distributed at a compile-time fixed count per lane. The work-group
  // size tracks the selected bucket geometry and amortizes each super-bucket across roughly eight
  // rows per lane.
  constexpr unsigned wg = ROUND_WG;

  // Level 0: gen-fill 2^26 entries into round-0 buckets (field-0 bucketing).
  const uint64_t t_gen0 = ZelHashState::now_ms();
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
  auto gen = state.gpu_compact
    ? submit_gen_fill<true>(q, *state.bundle, state.base_h, state.base_pending,
        state.level[0], state.l0_index, state.nslots + 0u * NBUCKETS, state.slot_capacity)
    : submit_gen_fill<false>(q, *state.bundle, state.base_h, state.base_pending,
        state.level[0], nullptr, state.nslots + 0u * NBUCKETS, state.slot_capacity);
#else
  auto gen = submit_gen_fill<false>(q, *state.bundle, state.base_h, state.base_pending,
    state.level[0], nullptr, state.nslots + 0u * NBUCKETS, state.slot_capacity);
#endif
  if (wait_stages) sycl_wait_and_throw(gen, state.device);
  // Rounds 0..3: collide field R over (bucket, RESTBITS), XOR -> level R+1.
  const uint64_t t_r0 = ZelHashState::now_ms();
  uint64_t t_round[4] = {0,0,0,0};
#if defined(MOM_SYCL_HAS_CUDA)
  // The combined Intel+CUDA build must retain both layouts and select by the actual queue backend.
  if (state.gpu_compact) run_collision_rounds<true>(state, wg, wait_stages, prof, t_round);
  else                  run_collision_rounds<false>(state, wg, wait_stages, prof, t_round);
#elif defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_ADAPTIVECPP)
  run_collision_rounds<true>(state, wg, wait_stages, prof, t_round);
#else
  // oneAPI Intel-only artifact: do not instantiate the bad-on-Xe 16-byte writer at all.
  run_collision_rounds<false>(state, wg, wait_stages, prof, t_round);
#endif
  const uint64_t t_rounds_end = ZelHashState::now_ms();

  // The fused round-3 kernel already retained only final-field-zero pairs.
  const uint64_t t_final = ZelHashState::now_ms();

  EqLevels lv;
  for (unsigned L = 0; L < NSTORED_LEVELS; ++L) lv.level[L] = state.level[L];
  lv.l0_index = state.l0_index;
  auto submit_recovery = [&](uint32_t count, const uint32_t* device_count) {
#if defined(MOM_SYCL_HAS_CUDA)
    if (state.gpu_compact)
      return submit_recover<true>(q, *state.bundle, lv, state.nslots, state.ref, count,
        state.cand, state.cand_count, ZelHashState::CAND_CAP, wg, state.slot_capacity, device_count);
    return submit_recover<false>(q, *state.bundle, lv, state.nslots, state.ref, count,
      state.cand, state.cand_count, ZelHashState::CAND_CAP, wg, state.slot_capacity, device_count);
#elif defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_ADAPTIVECPP)
    return submit_recover<true>(q, *state.bundle, lv, state.nslots, state.ref, count,
      state.cand, state.cand_count, ZelHashState::CAND_CAP, wg, state.slot_capacity, device_count);
#else
    return submit_recover<false>(q, *state.bundle, lv, state.nslots, state.ref, count,
      state.cand, state.cand_count, ZelHashState::CAND_CAP, wg, state.slot_capacity, device_count);
#endif
  };

  if (async_chain) {
    // Queue the maximum bounded grid behind round 3 and let each work-group consult the device
    // counter. This retains one final completion point instead of a host wait before recovery.
    sycl_wait_and_throw(submit_recovery(ZelHashState::CAND_CAP, state.ref_count), state.device);
  } else {
    const uint32_t ref_n = std::min<uint32_t>(*state.ref_count, ZelHashState::CAND_CAP);
    if (ref_n != 0)
      sycl_wait_and_throw(submit_recovery(ref_n, nullptr), state.device);
  }
#if defined(MOM_EQ_FORWARD_MAP)
  if (state.gpu_compact) {
    const uint32_t recovered = std::min<uint32_t>(*state.cand_count, ZelHashState::CAND_CAP);
    if (recovered != 0) {
      sycl_wait_and_throw(submit_l0_invert(q, *state.bundle, state.l0_index, state.cand,
        recovered * (1u << K), state.invert_target, state.invert_next, state.invert_head),
        state.device);
    }
  }
#endif
  const uint64_t t_rounds = ZelHashState::now_ms();
  if (prof) {
    std::fprintf(stderr,
      "[eq-prof] gen-fill %llums | r0 %llu r1 %llu r2 %llu r3 %llu (rounds %llums) | recovery %llums\n",
      (unsigned long long)(t_r0 - t_gen0),
      (unsigned long long)(t_round[0] - t_r0), (unsigned long long)(t_round[1] - t_round[0]),
      (unsigned long long)(t_round[2] - t_round[1]), (unsigned long long)(t_round[3] - t_round[2]),
      (unsigned long long)(t_rounds_end - t_r0), (unsigned long long)(t_rounds - t_final));
    std::fflush(stderr);
  }

  // ---- Pull only the recovered candidate list to the host. ----
  const uint64_t t_recov0 = ZelHashState::now_ms();
  uint32_t cand_count = *state.cand_count;
  const uint32_t cand_n = std::min<uint32_t>(cand_count, ZelHashState::CAND_CAP);

  if (log) {
    // Per-level survivor counts are diagnostic-only. Do not add a 640 KiB device-to-host transfer and
    // queue synchronization to every benchmark/mining solve merely to populate the verbose report.
    std::vector<uint32_t> ns_host((size_t)NSTORED_LEVELS * NBUCKETS);
    sycl_wait_and_throw(q.memcpy(ns_host.data(), state.nslots, ns_host.size() * sizeof(uint32_t)), state.device);
    for (unsigned L = 0; L < NSTORED_LEVELS; ++L) {
      uint64_t sum = 0, overflow = 0;
      for (unsigned bkt = 0; bkt < NBUCKETS; ++bkt) {
        const uint32_t c = ns_host[(size_t)L * NBUCKETS + bkt];
        sum += std::min<uint32_t>(c, state.slot_capacity);
        if (c > state.slot_capacity) overflow += c - state.slot_capacity;
      }
      std::fprintf(stderr, "zelhash level %u survivors=%llu dropped=%llu (%.4f%%)\n",
        L, (unsigned long long)sum, (unsigned long long)overflow,
        100.0 * (double)overflow / (double)(sum + overflow ? sum + overflow : 1));
    }
    std::fprintf(stderr, "zelhash zero-nodes=%u candidates=%u (solve %llums)\n",
      *state.ref_count, cand_count,
      (unsigned long long)(t_rounds - t_gen0));
    std::fflush(stderr);
  }

  int n_solutions = 0;
  size_t distinct_count = 0;   // distinct valid Equihash proofs found this solve (pre-target-filter)
  if (cand_n) {
    // Device recovery already produced the 16 leaf indices, so copy back only this tiny list.
    std::vector<EqCandidate> cand_h(cand_n);
    sycl_wait_and_throw(q.memcpy(cand_h.data(), state.cand, cand_n * sizeof(EqCandidate)), state.device);

    // De-dup distinct solutions (the same proof can surface via several candidates).
    std::vector<std::array<uint8_t, 52>> seen;
    for (uint32_t ci = 0; ci < cand_n; ++ci) {
      // Each candidate is one recovered final-zero pair with 16 distinct leaf indices.
      std::vector<uint32_t> leaves(cand_h[ci].leaves, cand_h[ci].leaves + (1u << K));
      if (leaves.size() != (size_t)(1u << K)) continue;   // PROOFSIZE = 16

      // DistinctIndices: all 16 leaves must be unique.
      std::vector<uint32_t> sorted = leaves;
      std::sort(sorted.begin(), sorted.end());
      bool distinct = true;
      for (size_t i = 1; i < sorted.size(); ++i) if (sorted[i] == sorted[i - 1]) { distinct = false; break; }
      if (!distinct) continue;

      // Canonical IndicesBefore ordering, then compress to 52 bytes.
      order_indices(leaves);
      std::array<uint8_t, 52> sol{};
      compress_indices(leaves, sol.data());
      bool dup = false;
      for (const auto& s : seen) if (s == sol) { dup = true; break; }
      if (dup) continue;
      seen.push_back(sol);
    }

    distinct_count = seen.size();
    // M5 mining path: each distinct solution is a valid Equihash proof, but only those whose Flux
    // block hash dSHA256(header || 0x34 || solution) meets the 256-bit PoW target are submittable.
    // The is_test SOLVE path (keystone / offline vector) keeps ALL distinct solutions so the checker
    // can assert the known block-400000 solution is present regardless of the (all-0xFF) target.
    std::vector<std::array<uint8_t, 52>> emit;
    emit.reserve(seen.size());
    for (const auto& sol : seen) {
      if (is_test) { emit.push_back(sol); continue; }   // SOLVE-test: emit every distinct solution
      uint8_t blockhash[32];
      flux_block_hash(input, sol.data(), blockhash);
      if (flux_meets_target(blockhash, target)) emit.push_back(sol);
    }

    // Sort the kept solutions lexicographically so the out-of-band buffer is deterministic regardless
    // of the non-deterministic candidate-collection order (atomic fetch_add). This makes the offline
    // vector's full-buffer comparison stable; the pool is indifferent to ordering.
    std::sort(emit.begin(), emit.end());

    // Hand the kept solutions back: solution_out is the 5120-byte small-blob buffer. Layout:
    // [0] = solution count (u8, capped), then count * 52 bytes.
    const unsigned cap = (SMALL_BLOB_SOL_LEN - 1) / 52;   // up to 98 solutions
    n_solutions = (int)std::min<size_t>(emit.size(), cap);
    solution_out[0] = (uint8_t)n_solutions;
    for (int i = 0; i < n_solutions; ++i)
      std::memcpy(solution_out + 1 + (size_t)i * 52, emit[i].data(), 52);
    if (log) {
      std::fprintf(stderr, "zelhash %s solutions=%d (of %zu distinct):\n",
        is_test ? "distinct" : "target-passing", n_solutions, seen.size());
      for (int i = 0; i < n_solutions; ++i) {
        std::fprintf(stderr, "  ");
        for (int b = 0; b < 52; ++b) std::fprintf(stderr, "%02x", emit[i][b]);
        std::fprintf(stderr, "\n");
      }
      std::fflush(stderr);
    }
  } else {
    solution_out[0] = 0;
  }

  // MOM_ZELHASH_PERF: Sol/s logger (mirrors MOM_FISHHASH_PERF). Accumulates the number of distinct
  // valid proofs found (not nonces, and BEFORE the target filter so bench/zero-target still measures
  // throughput) over a >=2s window; the solver averages ~1.88 solutions per nonce.
  static const bool perf_log = std::getenv("MOM_ZELHASH_PERF") != nullptr;
  if (perf_log && !is_test) {
    static uint64_t acc_sols = 0; static uint64_t acc_solves = 0;
    static auto t0 = std::chrono::steady_clock::now();
    acc_sols += (uint64_t)distinct_count;
    ++acc_solves;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec >= 2.0) {
      std::fprintf(stderr, "[zelhash] %.2f Sol/s, %.2f solve/s (%s)\n",
        (double)acc_sols / sec, (double)acc_solves / sec, dev_str.c_str());
      std::fflush(stderr);
      acc_sols = 0; acc_solves = 0; t0 = std::chrono::steady_clock::now();
    }
  }

  if (prof) {
    std::fprintf(stderr, "[eq-prof] host-recovery %llums (cand=%u) | total %llums\n",
      (unsigned long long)(ZelHashState::now_ms() - t_recov0), cand_count,
      (unsigned long long)(ZelHashState::now_ms() - t_start));
    std::fflush(stderr);
  }
  if (log) { std::fprintf(stderr, "zelhash solve done (%llu ms, %d target-passing, %zu distinct)\n",
    (unsigned long long)(ZelHashState::now_ms() - t_start), n_solutions, distinct_count); std::fflush(stderr); }

  // Return semantics depend on the caller:
  //  * is_test SOLVE path (the keystone / offline vector): the core's is_test handler dumps the whole
  //    out-of-band buffer when this returns 1, so report 1-on-any-solution (the payload count is in
  //    solution_out[0]).
  //  * bench: report the raw solver throughput (distinct proofs found this solve, target-independent)
  //    so the core's m_hash_count accounting yields Sol/s.
  //  * mining: report the target-passing payload count so the core drives send_result.
  if (is_test) return n_solutions > 0 ? 1 : 0;
  return is_benchmark ? (int)distinct_count : n_solutions;
}
