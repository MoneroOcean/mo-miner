} // namespace mom_beamhash3

using namespace mom_beamhash3;

// BeamHash III entrypoint. ABI matches zelhash (c29-like out-of-band solution count).
//   input  : prework(32) || nonce(8) || extranonce(4) = 44-byte blob.
//   is_test (default): M1 gen-validation -- dumps the first BEAMHASH3_TEST_ROWS raw 448-bit leaf rows
//            (7 LE u64 words each) into solution_out for the JS-oracle diff. MOM_BEAMHASH3_SOLVE switches
//            it to the full Wagner solve and dumps the found 104-byte solution(s).
//   mining (!is_test): the full Wagner solve; writes [count:u8][count*104-byte solution] to solution_out.
int beamhash3(
  const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size,
  uint8_t* const solution_out, uint64_t* const /*pnonce*/, const uint8_t* const target,
  const unsigned /*intensity*/, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < BLOB_BYTES) throw std::string("Bad beamhash3 input length");
  BeamState& state = beam_state(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_io();

  // Host: IndividualWork (the SipHash key) = BLAKE2b(prework||nonce||extranonce, Beam-PoW personal).
  uint64_t prePow[4];
  individual_work(input, prePow);
  std::memcpy(state.prePow, prePow, 4 * sizeof(uint64_t));

  const bool solve_in_test = is_test && std::getenv("MOM_BEAMHASH3_SOLVE") != nullptr;

  if (is_test && !solve_in_test) {
    // M1: dump the first BEAMHASH3_TEST_ROWS raw 448-bit leaf rows.
    const size_t rows_bytes = (size_t)BEAMHASH3_TEST_ROWS * ROW_BYTES;
    std::memset(state.rows, 0, rows_bytes);
    sycl_wait_and_throw(submit_gen_test(state.queue, *state.bundle, state.prePow, state.rows), state.device);
    // Zero the WHOLE out-of-band buffer (not just rows_bytes) so the test-result hex dump -- which the
    // core emits over the full SMALL_BLOB_SOL_LEN -- is deterministic for the offline gen vector; the
    // 3584-byte gen dump leaves a 1536-byte tail that alloc_mem (_mm_malloc) does not zero otherwise.
    std::memset(solution_out, 0, SMALL_BLOB_SOL_LEN);
    std::memcpy(solution_out, state.rows, rows_bytes);
    (void)is_benchmark;
    return 1;
  }

  // ---- M2/M3 full Wagner solve. -------------------------------------------------------------------
  state.ensure_arenas(!is_benchmark);
  std::memset(solution_out, 0, SMALL_BLOB_SOL_LEN);
  const bool log = !is_benchmark;
  static const bool prof = std::getenv("MOM_BEAMHASH3_PROF") != nullptr;
  const uint64_t t_start = BeamState::now_ms();
  sycl::queue& q = state.queue;
  const bool cuda_layout = mom_is_cuda(state.device);
  const bool level_zero_layout = sycl_is_level_zero_gpu(state.device);
  // The fused non-CUDA solve has no host-sized handoff between rounds. Its queue is in-order, so in
  // the hot path submit the whole chain and poll only its tail instead of synchronizing the host after
  // every memset/collide. CUDA normally reads each output count to size the next coalesced scatter;
  // the Windows CUDA path over-dispatches by the 3.125%-padded allocation bound and lets that scatter
  // consume the preceding kernel's device-written count, removing the same host completion points.
  // This closes a large Windows driver-wait gap; Linux keeps its count-sized scatter because the
  // production AdaptiveCpp path gains less than 1%. Profiling stays synchronous.
  const bool async_fused = !cuda_layout && !prof;
  static const bool cuda_async_enabled = [] {
    const char* value = std::getenv("MOM_BEAMHASH3_CUDA_ASYNC");
#if defined(_WIN32)
    return !value || (std::strcmp(value, "0") && std::strcmp(value, "false"));
#else
    return value && std::strcmp(value, "0") && std::strcmp(value, "false");
#endif
  }();
  const bool async_cuda = cuda_layout && !prof && cuda_async_enabled;
  const bool async_pipeline = async_fused || async_cuda;
  sycl::event fused_tail;
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  const bool compact_layout = beam_compact_enabled(state.device);
  if (compact_layout) {
    // Packed Beam rows alternate directly between two arenas through level 3. Level 4 reuses arena
    // zero as 16-byte work+parent records while arena one retains the level-3 recovery trees; this
    // removes the last round's otherwise-dead 48-byte tree traffic. The in-order queue makes every
    // clear/round dependency explicit without a host synchronization point.
    *state.cand_count = 0;
    // The collision-list SLM footprint, rather than threads, caps resident workgroups. Once level 4
    // became a thin record, WG256's faster final traversal slightly outweighs WG128's seed advantage
    // on Blackwell (32.92 vs 32.86 Sol/s, matched three-sample medians). HIP retains WG512.
    const unsigned compact_wg = beamhash3_workgroup(state.device);
    static unsigned compact_profile_call = 0;
    // AdaptiveCpp JITs each template specialization during the first solve. Profile the following
    // steady solve so compilation is never mistaken for GPU execution time.
    const bool compact_profile = prof && compact_profile_call++ == 1;
    uint64_t compact_stage_start = compact_profile ? BeamState::now_ms() : 0;
    auto compact_stage_done = [&](const char* name, sycl::event event) {
      if (!compact_profile) return;
      sycl_wait_and_throw(event, state.device);
      const uint64_t now = BeamState::now_ms();
      std::fprintf(stderr, "beamhash3 compact %s: %llu ms\n", name,
                   (unsigned long long)(now - compact_stage_start));
      std::fflush(stderr);
      compact_stage_start = now;
    };
    q.memset(state.compact_ns[0], 0, (size_t)COMPACT_NBUCKETS * sizeof(uint32_t));
    auto compact_event = submit_compact_seed(q, *state.bundle, state.prePow, state.compact[0],
                                             state.compact_ns[0], compact_wg);
    compact_stage_done("seed", compact_event);
    q.memset(state.compact_ns[1], 0, (size_t)COMPACT_NBUCKETS * sizeof(uint32_t));
    compact_event = submit_compact_round<1>(q, *state.bundle, state.compact[0], state.compact_ns[0],
                                            state.compact[1], state.compact_ns[1], state.cand,
                                            state.cand_count, BeamState::CAND_CAP, compact_wg);
    compact_stage_done("round 1", compact_event);
    q.memset(state.compact_ns[0], 0, (size_t)COMPACT_NBUCKETS * sizeof(uint32_t));
    compact_event = submit_compact_round<2>(q, *state.bundle, state.compact[1], state.compact_ns[1],
                                            state.compact[0], state.compact_ns[0], state.cand,
                                            state.cand_count, BeamState::CAND_CAP, compact_wg);
    compact_stage_done("round 2", compact_event);
    q.memset(state.compact_ns[1], 0, (size_t)COMPACT_NBUCKETS * sizeof(uint32_t));
    compact_event = submit_compact_round<3>(q, *state.bundle, state.compact[0], state.compact_ns[0],
                                            state.compact[1], state.compact_ns[1], state.cand,
                                            state.cand_count, BeamState::CAND_CAP, compact_wg);
    compact_stage_done("round 3", compact_event);
    q.memset(state.compact_ns[0], 0, (size_t)COMPACT_NBUCKETS * sizeof(uint32_t));
    compact_event = submit_compact_round<4>(q, *state.bundle, state.compact[1], state.compact_ns[1],
                                            state.compact[0], state.compact_ns[0], state.cand,
                                            state.cand_count, BeamState::CAND_CAP, compact_wg);
    compact_stage_done("round 4", compact_event);
    auto compact_tail = submit_compact_final_thin(
      q, *state.bundle, reinterpret_cast<const BeamLevel4Row*>(state.compact[0]),
      state.compact_ns[0], state.compact[1], state.cand, state.cand_count,
      BeamState::CAND_CAP, compact_wg);
    compact_stage_done("round 5", compact_tail);
    sycl_wait_and_throw(compact_tail, state.device);
    if (log) { std::fprintf(stderr, "beamhash3 compact solve candidates=%u (%llu ms)\n",
      std::min<uint32_t>(*state.cand_count, BeamState::CAND_CAP),
      (unsigned long long)(BeamState::now_ms() - t_start)); std::fflush(stderr); }
  } else
#endif
  {

  // With SLM-staged collision keys, a larger WG hides the wide-record latency. Packed/fused whole-
  // solve A/Bs on Arc B580 repeated at 11.94 H/s for WG1024 versus 11.39--11.40 for WG640 (+4.8%).
  // On RTX 5060 Ti, DPC++ CUDA's 150-solve stage mean improves 172.7 -> 161.8 ms at WG768, while
  // AdaptiveCpp peaks at WG640 (151.6 ms); HIP remains flat around its WG640 default.
  const unsigned wg = beamhash3_workgroup(state.device);
  unsigned scatter_wg = 128;
  { unsigned long parsed = 0; if (mom_parse_env_ulong("MOM_BEAMHASH3_SCATTER_WG", parsed) && parsed >= 16)
      scatter_wg = (unsigned)std::min<unsigned long>(parsed, 1024); }

  for (unsigned i = 0; i <= NUM_ROUNDS; ++i) state.mixed_count[i] = 0;
  *state.cand_count = 0;
  // GEN FUSION: there is no separate gen kernel / level-0 arena. Round-1 scatter regenerates each leaf
  // in-kernel (slot == leaf index), so round 1's "input count" is simply all 2^25 leaves.
  state.mixed_count[0] = (uint32_t)NUM_ENTRIES;

  // Level 0 is never materialized: round-1 scatter regenerates each leaf at slot==index. Thereafter,
  // collision R immediately applies mix R+1 and advances into the opposite parity bucket arena. This
  // preserves the linear oslot as the tree-log source ID while eliminating levels 1..4's linear arena.
  const uint64_t t_gen0 = prof ? BeamState::now_ms() : 0;
  auto gen_clear = q.memset(state.bucket_ns[0], 0, (size_t)NBUCKETS * sizeof(uint32_t));
  if (!async_pipeline) sycl_wait_and_throw(gen_clear, state.device);
  auto gen_scatter = submit_mix_scatter<1>(q, *state.bundle, nullptr, (uint32_t)NUM_ENTRIES,
                                           state.bucket[0], state.bucket_ns[0], scatter_wg,
                                           state.prePow);
  if (!async_pipeline) sycl_wait_and_throw(gen_scatter, state.device);
  if (prof) { std::fprintf(stderr, "beamhash3 gen/mix/scatter r1: %llu ms\n",
    (unsigned long long)(BeamState::now_ms() - t_gen0)); std::fflush(stderr); }

  // Rounds 1..4 collide and advance directly into the other bucket arena. Round 5 writes only the
  // compact final merged word. Each output counter remains a linear source-ID namespace for tree logs.
  auto do_round = [&](auto Rconst) {
    constexpr unsigned R = decltype(Rconst)::value;
    constexpr unsigned L = R - 1;
    constexpr unsigned CUR = L & 1u;
    constexpr unsigned NEXT = R & 1u;
    uint32_t in_count = 0;
    if (!async_pipeline)
      in_count = std::min<uint32_t>(state.mixed_count[L], (uint32_t)MIXED_CAP);
    if constexpr (R < NUM_ROUNDS) if (!cuda_layout) {
      auto clear = q.memset(state.bucket_ns[NEXT], 0, (size_t)NBUCKETS * sizeof(uint32_t));
      if (!async_fused) sycl_wait_and_throw(clear, state.device);
    }
    const uint64_t t_co = prof ? BeamState::now_ms() : 0;
    uint64_t t_collide_done = 0;
    if (cuda_layout) {
      // NVIDIA keeps a linear/coalesced handoff: the direct fused random wide writes regress by
      // 8--33% there. One reusable max-stride bucket and one reusable linear arena still fit in
      // 6.76 GiB, while preserving the packed source-ID representation and fused leaf generation.
      auto collide = submit_collide<R, false, false>(
        q, *state.bundle, state.bucket[0], state.bucket_ns[0], state.linear_mixed,
        state.mixed_count + R, state.tree[R], nullptr, nullptr, wg);
      if (async_cuda) fused_tail = collide;
      else sycl_wait_and_throw(collide, state.device);
      t_collide_done = prof ? BeamState::now_ms() : 0;
      if constexpr (R < NUM_ROUNDS) {
        auto clear = q.memset(state.bucket_ns[0], 0, (size_t)NBUCKETS * sizeof(uint32_t));
        if (async_cuda) fused_tail = clear;
        else sycl_wait_and_throw(clear, state.device);
        const uint32_t out_count = async_cuda ? (uint32_t)MIXED_CAP
          : std::min<uint32_t>(state.mixed_count[R], (uint32_t)MIXED_CAP);
        auto scatter = async_cuda ? submit_mix_scatter<R + 1, true>(
          q, *state.bundle, state.linear_mixed, out_count, state.bucket[0], state.bucket_ns[0],
          scatter_wg, state.prePow, state.mixed_count + R)
          : submit_mix_scatter<R + 1, false>(
          q, *state.bundle, state.linear_mixed, out_count, state.bucket[0], state.bucket_ns[0],
          scatter_wg, state.prePow, nullptr);
        if (async_cuda) fused_tail = scatter;
        else sycl_wait_and_throw(scatter, state.device);
      }
    } else {
      if (level_zero_layout) {
        auto collide = submit_collide<R, true, true>(
          q, *state.bundle, state.bucket[CUR], state.bucket_ns[CUR], state.final_mixed,
          state.mixed_count + R, state.tree[R],
          R < NUM_ROUNDS ? state.bucket[NEXT] : nullptr,
          R < NUM_ROUNDS ? state.bucket_ns[NEXT] : nullptr, wg);
        if (async_fused) fused_tail = collide;
        else sycl_wait_and_throw(collide, state.device);
      } else {
        auto collide = submit_collide<R, true, false>(
          q, *state.bundle, state.bucket[CUR], state.bucket_ns[CUR], state.final_mixed,
          state.mixed_count + R, state.tree[R],
          R < NUM_ROUNDS ? state.bucket[NEXT] : nullptr,
          R < NUM_ROUNDS ? state.bucket_ns[NEXT] : nullptr, wg);
        if (async_fused) fused_tail = collide;
        else sycl_wait_and_throw(collide, state.device);
      }
    }
    if (!async_pipeline && (prof || log)) {
      const uint32_t out = std::min<uint32_t>(state.mixed_count[R], (uint32_t)MIXED_CAP);
      if (prof) {
        const uint64_t t_end = BeamState::now_ms();
        // Peak input-bucket demand (bucket_ns keeps counting past NSLOTS) -> NSLOTS sizing headroom.
        uint32_t maxns = 0;
        { std::vector<uint32_t> ns(NBUCKETS);
          q.memcpy(ns.data(), state.bucket_ns[cuda_layout ? 0 : CUR],
                   (size_t)NBUCKETS * sizeof(uint32_t)).wait();
          for (uint32_t v : ns) if (v > maxns) maxns = v; }
        if (cuda_layout) {
          std::fprintf(stderr,
            "beamhash3 round %u: in=%u out=%u  collide=%llu ms scatter=%llu ms  maxbucket=%u/%u%s\n",
            R, in_count, out, (unsigned long long)(t_collide_done - t_co),
            (unsigned long long)(t_end - t_collide_done), maxns, (unsigned)NSLOTS,
            state.mixed_count[R] > MIXED_CAP ? " (OVERFLOW)" : "");
        } else {
          std::fprintf(stderr,
            "beamhash3 round %u: in=%u out=%u  collide/advance=%llu ms  maxbucket=%u/%u%s\n",
            R, in_count, out, (unsigned long long)(t_end - t_co), maxns, (unsigned)NSLOTS,
            state.mixed_count[R] > MIXED_CAP ? " (OVERFLOW)" : "");
        }
      } else {
        std::fprintf(stderr, "beamhash3 round %u: in=%u out=%u%s\n", R, in_count, out,
          state.mixed_count[R] > MIXED_CAP ? " (OVERFLOW)" : "");
      }
      std::fflush(stderr);
    }
  };
  do_round(std::integral_constant<unsigned, 1>{});
  do_round(std::integral_constant<unsigned, 2>{});
  do_round(std::integral_constant<unsigned, 3>{});
  do_round(std::integral_constant<unsigned, 4>{});
  do_round(std::integral_constant<unsigned, 5>{});
  // mixed_count[5] is shared USM; complete the queued chain before the host sizes the final launch.
  if (async_pipeline) sycl_wait_and_throw(fused_tail, state.device);
  if (async_pipeline && log) {
    for (unsigned R = 1; R <= NUM_ROUNDS; ++R) {
      const uint32_t in = std::min<uint32_t>(state.mixed_count[R - 1], (uint32_t)MIXED_CAP);
      const uint32_t out = std::min<uint32_t>(state.mixed_count[R], (uint32_t)MIXED_CAP);
      std::fprintf(stderr, "beamhash3 round %u: in=%u out=%u%s\n", R, in, out,
        state.mixed_count[R] > MIXED_CAP ? " (OVERFLOW)" : "");
    }
    std::fflush(stderr);
  }

  // Final: level-5 survivors with merged 24-bit == 0 -> walk tree -> 32 distinct leaf indices.
  // (Leaf index == level-0 slot; recovery needs only the tree[] logs.)
  BeamLevels lv;
  for (unsigned L = 0; L <= NUM_ROUNDS; ++L) lv.tree[L] = state.tree[L];
  const uint32_t fin_count = std::min<uint32_t>(state.mixed_count[NUM_ROUNDS], (uint32_t)MIXED_CAP);
  sycl_wait_and_throw(submit_final(q, *state.bundle,
                                   cuda_layout ? state.linear_mixed : state.final_mixed, fin_count, lv,
                                   state.cand, state.cand_count, BeamState::CAND_CAP), state.device);
  }

  const uint32_t cand_count = std::min<uint32_t>(*state.cand_count, BeamState::CAND_CAP);
  if (log) { std::fprintf(stderr, "beamhash3 candidates=%u (%llu ms)\n", cand_count,
    (unsigned long long)(BeamState::now_ms() - t_start)); std::fflush(stderr); }

  // ---- Host recovery: each candidate's 32 leaf indices -> CompressArray(25) -> 100 bytes; append the
  // 4-byte extranonce (the blob's last 4 bytes) to form the 104-byte solution. De-dup distinct. ----
  int n_solutions = 0;
  size_t distinct_count = 0;
  if (cand_count) {
    std::vector<BeamCandidate> cand_h(cand_count);
    sycl_wait_and_throw(q.memcpy(cand_h.data(), state.cand, cand_count * sizeof(BeamCandidate)), state.device);

    const uint8_t* extranonce = input + PREWORK_BYTES + NONCE_BYTES;   // 4 bytes

    std::vector<std::array<uint8_t, 104>> seen;
    for (uint32_t ci = 0; ci < cand_count; ++ci) {
      const uint32_t* leaves = cand_h[ci].leaves;
      // distinct check (defensive; the device already filtered).
      bool distinct = true;
      for (unsigned i = 0; i < NUM_LEAVES && distinct; ++i)
        for (unsigned j = i + 1; j < NUM_LEAVES; ++j)
          if (leaves[i] == leaves[j]) { distinct = false; break; }
      if (!distinct) continue;
      std::array<uint8_t, 104> sol{};
      uint8_t minimal[100];
      compress_indices(leaves, minimal);
      std::memcpy(sol.data(), minimal, 100);
      std::memcpy(sol.data() + 100, extranonce, 4);
      bool dup = false;
      for (const auto& s : seen) if (s == sol) { dup = true; break; }
      if (dup) continue;
      seen.push_back(sol);
    }
    distinct_count = seen.size();
    std::sort(seen.begin(), seen.end());

    // M5 mining path: every distinct solution is a valid BeamHash III proof, but only those whose
    // SHA-256 meets the packed Beam network difficulty are submittable. The is_test SOLVE path keeps
    // ALL distinct solutions (so the offline checker can assert the known keystone is present
    // regardless of the difficulty), and benching reports the raw distinct count as the Sol/s unit.
    // The packed int32 difficulty is carried in the low 4 bytes of the 32-byte big-endian target.
    const uint32_t packed = target ? (((uint32_t)target[28] << 24) | ((uint32_t)target[29] << 16) |
                                      ((uint32_t)target[30] << 8) | (uint32_t)target[31]) : 0;
    // MOM_BEAMHASH3_FORCE_SUBMIT: bypass the difficulty filter and emit the single best (lowest-hash)
    // distinct solution. A debug aid to exercise the live JSON-RPC `solution` submit ABI end-to-end on
    // a hard network difficulty (the pool will low-diff-reject it, which still proves the wire path).
    static const bool force_submit = std::getenv("MOM_BEAMHASH3_FORCE_SUBMIT") != nullptr;
    std::vector<std::array<uint8_t, 104>> emit;
    if (force_submit && !is_test && !is_benchmark && !seen.empty()) {
      size_t best = 0; uint8_t best_h[32]; sha256(seen[0].data(), 104, best_h);
      for (size_t i = 1; i < seen.size(); ++i) {
        uint8_t h[32]; sha256(seen[i].data(), 104, h);
        if (std::memcmp(h, best_h, 32) < 0) { std::memcpy(best_h, h, 32); best = i; }
      }
      emit.push_back(seen[best]);
    } else for (const auto& sol : seen) {
      if (is_test || is_benchmark || packed == 0) { emit.push_back(sol); continue; }
      uint8_t digest[32];
      sha256(sol.data(), 104, digest);
      if (beam_target_reached(digest, packed)) emit.push_back(sol);
    }

    // Hand back: [count:u8][count * 104-byte solution].
    const unsigned cap = (SMALL_BLOB_SOL_LEN - 1) / 104;
    n_solutions = (int)std::min<size_t>(emit.size(), cap);
    solution_out[0] = (uint8_t)n_solutions;
    for (int i = 0; i < n_solutions; ++i)
      std::memcpy(solution_out + 1 + (size_t)i * 104, emit[i].data(), 104);
    if (log) {
      std::fprintf(stderr, "beamhash3 emitted solutions=%d (of %zu distinct):\n", n_solutions, distinct_count);
      for (int i = 0; i < n_solutions; ++i) {
        std::fprintf(stderr, "  ");
        for (int b = 0; b < 104; ++b) std::fprintf(stderr, "%02x", emit[i][b]);
        std::fprintf(stderr, "\n");
      }
      std::fflush(stderr);
    }
  } else {
    solution_out[0] = 0;
  }

  if (log) { std::fprintf(stderr, "beamhash3 solve done (%llu ms, %d solutions, %zu distinct)\n",
    (unsigned long long)(BeamState::now_ms() - t_start), n_solutions, distinct_count); std::fflush(stderr); }

  if (is_test) return n_solutions > 0 ? 1 : 0;
  return is_benchmark ? (int)distinct_count : n_solutions;
}
