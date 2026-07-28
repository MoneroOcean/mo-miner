struct LoopStats {
  uint64_t window_start = 0, dispatch = 0, msg = 0, post = 0, iters = 0, jobs = 0,
           max_msg = 0, max_post = 0, dispatch_end = 0;
};

static uint64_t loop_now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()
  ).count();
}

// log a one-off stall line when a single loop phase blocks the compute loop for >100ms
static void loop_stall(const char* phase, const uint64_t us) {
  if (us > 100000)
    fprintf(stderr, "LOOPSTALL t=%llu %s=%.1fms\n",
      static_cast<unsigned long long>(time(nullptr)), phase, us / 1e3);
}

static void loop_stats_report(LoopStats& ls, const uint64_t now) {
  const uint64_t wall = now - ls.window_start;
  fprintf(stderr,
    "LOOPSTAT t=%llu wall=%.1fs dispatch=%.1f%% msg=%.3fs post=%.3fs iters=%llu jobs=%llu"
    " max_msg=%.1fms max_post=%.1fms\n",
    static_cast<unsigned long long>(time(nullptr)),
    wall / 1e6, 100.0 * ls.dispatch / wall, ls.msg / 1e6, ls.post / 1e6,
    static_cast<unsigned long long>(ls.iters), static_cast<unsigned long long>(ls.jobs),
    ls.max_msg / 1e3, ls.max_post / 1e3);
  fflush(stderr);
  ls = LoopStats{};
  ls.window_start = now;
}

void Core::Execute() {
  debug_startup("Core::Execute entered");
  const bool loop_stats = std::getenv("MOM_LOOP_STATS") != nullptr;
  LoopStats ls;
  bool runtime_initialized = false;
  auto init_runtime = [&]() {
    if (runtime_initialized) return;
    runtime_initialized = true;
    debug_startup("runtime init start");

    argon2_select_impl();
    debug_startup((std::string("argon2 impl ") + argon2_get_impl_name()).c_str());

#if !defined(_WIN32)
    if (ci.arch() == xmrig::ICpuInfo::ARCH_ZEN)
      xmrig::RxFix::setupMainLoopExceptionFrame();
    if (ci.has(xmrig::ICpuInfo::FLAG_SSE41)) rx_blake2b_compress = rx_blake2b_compress_sse41;
    if (ci.hasAVX2())                        rx_blake2b          = blake2b_avx2;
#endif

    randomx_set_scratchpad_prefetch_mode(0);
#if defined(_WIN32)
    randomx_set_huge_pages_jit(false);
    randomx_set_optimized_dataset_init(0);
#else
    randomx_set_huge_pages_jit(true);
    randomx_set_optimized_dataset_init(1);
#endif
    debug_startup("runtime init done");
  };

  while (true) {
    const uint64_t t_loop_start = loop_stats ? loop_now_us() : 0;
    if (loop_stats) {
      if (ls.dispatch_end) {
        const uint64_t post_us = t_loop_start - ls.dispatch_end;
        ls.post += post_us;
        if (post_us > ls.max_post) ls.max_post = post_us;
        loop_stall("post", post_us);
        ls.dispatch_end = 0;
      }
      if (!ls.window_start) ls.window_start = t_loop_start;
      else if (t_loop_start - ls.window_start > 10*1000*1000) loop_stats_report(ls, t_loop_start);
    }

    std::deque<Message> messages;
    fromNode.readAll(messages);
    for (const auto& message : messages) {
      try {
        debug_startup(("message " + message.name).c_str());
        if (message.name == "job" || message.name == "bench" || message.name == "test")
          init_runtime();
        if (loop_stats && (message.name == "job" || message.name == "bench")) ++ls.jobs;
        if (!process_message(message.name, message.values)) return;
      } catch(const std::string& err) {
        send_error(std::string("Message processing exception: ") + err);
      }
    }
    if (loop_stats && !messages.empty()) {
      const uint64_t msg_us = loop_now_us() - t_loop_start;
      ls.msg += msg_us;
      if (msg_us > ls.max_msg) ls.max_msg = msg_us;
      loop_stall("msg", msg_us);
    }

    // we skip first hash function run using m_hash_count check to exclude GPU compile time
    // that effectively skips it in test mode too
    static unsigned hashrate_check_counter = HASHRATE_COUNTER_INTERVAL;
    if (m_dev == DEV::RX_CPU) m_mutex_hashrate.lock();
    const uint64_t hash_count = m_hash_count;   // 64-bit: pearlhash's MAC count per attempt is 2^46, which truncated to 0 mod 2^32
    if (m_dev == DEV::RX_CPU) m_mutex_hashrate.unlock();
    // Most kernels dispatch fast enough that checking wall time every ten loop iterations avoids
    // pointless clock reads. A production PearlHash attempt lasts several seconds, however: after the
    // gfx12 WMMA optimization ten attempts can land just below 60 s, miss the reporting threshold,
    // and make a benchmark wait a further ten attempts. Check PearlHash after every completed attempt;
    // one host clock read per multi-second GPU dispatch is negligible and makes the 60 s window exact.
    if (hash_count && (m_dev == DEV::PEARLHASH_GPU || --hashrate_check_counter == 0)) {
      hashrate_check_counter = HASHRATE_COUNTER_INTERVAL;
      const uint64_t new_timestamp = std::chrono::time_point_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now()
      ).time_since_epoch().count();
      if (!m_timestamp || new_timestamp - m_timestamp > 60*1000) {
        if (m_timestamp) send_msg("hashrate", "hashrate", std::to_string(
          static_cast<float>(hash_count) / (new_timestamp - m_timestamp) * 1000.0f
        ));
        m_timestamp = new_timestamp;
        if (m_dev == DEV::RX_CPU) m_mutex_hashrate.lock();
        m_hash_count = 0;
        if (m_dev == DEV::RX_CPU) m_mutex_hashrate.unlock();
      }
    }

    if (!m_fn.any) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    { // m_fn.any is non-null here (early-continue above), so compute a batch
      init_runtime();
      // Only one device runs per dispatch, so a single solution-count + nonce/seed pair is shared
      // across all the GPU cases. dev_nonce doubles as the kernel's in/out found-nonce slot (and as
      // pearlhash's search seed); dev_sols is the per-call solution count (or C29's -1 EOL / 0 pending).
      int dev_sols = 0;
      uint64_t dev_nonce = 0;
      const bool is_test = !m_nonce32 && !m_nonce64;
      const uint64_t t_dispatch = loop_stats ? loop_now_us() : 0;
      try {
        switch (m_dev) {
          case DEV::CPU:
            m_fn.cpu(m_input, m_input_len, m_output, m_ctx, m_height);
            break;
          case DEV::GPU:
            m_fn.gpu_cn(
              m_input, m_input_len, m_output, m_spads, m_batch, m_dev_str, m_backend);
            break;
          case DEV::C29_GPU:
            dev_nonce = m_nonce_bytes == 4 ? bswap_32(*get_nonce32()) : bswap_64(*get_nonce64());
            dev_sols = m_fn.gpu_c29(
              m_job_ref, m_c29_proof_size, m_input, m_input_len, m_output,
              static_cast<uint32_t*>(m_spads), &dev_nonce, m_dev_str
            );
            break;
          case DEV::KAWPOW_GPU:
            std::memcpy(&dev_nonce, m_input + m_nonce_offset, sizeof(dev_nonce));
            dev_sols = m_fn.gpu_kawpow(
              m_job_ref, m_height, m_input, m_input_len, m_output,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target,
              m_batch, is_test, m_is_bench, m_dev_str
            );
            break;
          case DEV::ETCHASH_GPU:
            std::memcpy(&dev_nonce, m_input + m_nonce_offset, sizeof(dev_nonce));
            dev_sols = m_fn.gpu_etchash(
              m_job_ref, m_height, m_input, m_input_len, m_output,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target_bin, m_seed,
              m_batch, is_test, m_is_bench, m_dev_str
            );
            break;
          case DEV::AUTOLYKOS2_GPU:
            std::memcpy(&dev_nonce, m_input + m_nonce_offset, sizeof(dev_nonce));
            dev_sols = m_fn.gpu_autolykos2(
              m_job_ref, m_height, m_input, m_input_len, m_output,
              &dev_nonce, m_target_bin,
              m_batch, is_test, m_is_bench, m_dev_str
            );
            break;
          case DEV::PEARLHASH_GPU:
            dev_nonce = m_nonce64;   // the search seed is internal (not embedded in the blob)
            dev_sols = m_fn.gpu_pearlhash(
              m_job_ref, m_height, m_input, m_input_len, m_output,
              &dev_nonce, m_target_bin,
              m_batch, is_test, m_is_bench, m_dev_str, m_backend,
              m_pearlhash_n, m_pearlhash_k, m_pearlhash_rank
            );
            break;
          case DEV::FISHHASH_GPU:
            std::memcpy(&dev_nonce, m_input + m_nonce_offset, sizeof(dev_nonce));  // nonce at offset 32
            dev_sols = m_fn.gpu_fishhash(
              m_job_ref, m_height, m_input, m_input_len, m_output,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target_bin, m_seed,
              m_batch, is_test, m_is_bench, m_dev_str
            );
            break;
          case DEV::KARLSENHASHV2_GPU:
            std::memcpy(&dev_nonce, m_input + m_nonce_offset, sizeof(dev_nonce));  // nonce at offset 72
            dev_sols = m_fn.gpu_karlsenhashv2(
              m_job_ref, m_height, m_input, m_input_len, m_output,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target_bin, m_seed,
              m_batch, is_test, m_is_bench, m_dev_str
            );
            break;
          case DEV::ZELHASH_GPU:
            // c29-like: the 32-byte nonce lives in the header (offset 108); the solver writes the
            // 52-byte solution (or, in is_test, the gen-kernel rows) out-of-band into m_spads.
            std::memcpy(&dev_nonce, m_input + m_nonce_offset, sizeof(dev_nonce));  // low 8 bytes @108
            dev_sols = m_fn.gpu_zelhash(
              m_job_ref, m_height, m_input, m_input_len,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target_bin,
              m_batch, is_test, m_is_bench, m_dev_str
            );
            break;
          case DEV::BEAMHASH3_GPU:
            // c29-like: input is the prework||nonce||extranonce blob; the 8-byte Beam nonce is at
            // offset 32. The solver writes the 104-byte solution(s) out-of-band into m_spads (or, in
            // is_test, the gen-validation rows).
            std::memcpy(&dev_nonce, m_input + m_nonce_offset, sizeof(dev_nonce));  // 8-byte nonce @32
            dev_sols = m_fn.gpu_beamhash3(
              m_job_ref, m_height, m_input, m_input_len,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target_bin,
              m_batch, is_test, m_is_bench, m_dev_str
            );
            break;
          case DEV::RX_CPU: throw "Internal error: Unreachable code executed";
        }
      } catch(const std::string& err) {
        send_error(std::string("Compute function exception: ") + err);
        set_fn(nullptr);
        continue;
      } catch(const std::exception& e) {
        // Surface sycl::exception (and any std::exception) detail instead of a bare message.
        send_error(std::string("Compute function exception: ") + e.what());
        set_fn(nullptr);
        continue;
      } catch(...) {
        send_error("Compute function exception");
        set_fn(nullptr);
        continue;
      }
      if (loop_stats) {
        ls.dispatch_end = loop_now_us();
        ls.dispatch += ls.dispatch_end - t_dispatch;
        ++ls.iters;
      }

      if (is_test) {
        m_input_len = 0; // do not produce any more test jobs for async GPU code like in c29
        if (m_dev == DEV::KAWPOW_GPU || m_dev == DEV::ETCHASH_GPU) {
          if (dev_sols == 1) {
            char hash[HASH_LEN*2+1], mix[HASH_LEN*2+1];
            send_msg("test", "result", std::string(hash_bin2hex(hash, 0)) + " " +
                                      hash_bin2hex(static_cast<uint8_t*>(m_spads), mix));
          } else {
            send_error(std::string("No ") + (m_dev == DEV::KAWPOW_GPU ? "kawpow" : "etchash") + " test result");
          }
          set_fn(nullptr);
          continue;
        }
        if (m_dev == DEV::AUTOLYKOS2_GPU || m_dev == DEV::FISHHASH_GPU || m_dev == DEV::KARLSENHASHV2_GPU) {
          if (dev_sols == 1) {
            char hash[HASH_LEN*2+1];
            send_msg("test", "result", hash_bin2hex(hash, 0));
          } else {
            send_error("No " + m_algo_str + " test result");
          }
          set_fn(nullptr);
          continue;
        }
        if (m_dev == DEV::PEARLHASH_GPU) {
          if (dev_sols == 1) send_msg("test", "result", "ok");
          else send_error("No pearlhash test result");
          set_fn(nullptr);
          continue;
        }
        if (m_dev == DEV::ZELHASH_GPU || m_dev == DEV::BEAMHASH3_GPU) {
          // M1 gen-kernel validation (or the SOLVE path): the kernel dumped the rows / solution(s) into
          // m_spads; emit the whole buffer as hex so the standalone checker can diff it against the oracle.
          if (dev_sols == 1) {
            const uint8_t* const rows = static_cast<const uint8_t*>(m_spads);
            std::string hex; hex.reserve(SMALL_BLOB_SOL_LEN * 2);
            for (unsigned i = 0; i != SMALL_BLOB_SOL_LEN; ++i) {
              hex += hf_bin2hex(rows[i] >> 4);
              hex += hf_bin2hex(rows[i] & 0xF);
            }
            send_msg("test", "result", hex);
          } else {
            send_error(m_dev == DEV::BEAMHASH3_GPU ? "No beamhash3 test result" : "No zelhash test result");
          }
          set_fn(nullptr);
          continue;
        }
        if (m_dev == DEV::C29_GPU && dev_sols == 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
        if (m_dev == DEV::C29_GPU && dev_sols == -1) {
          send_msg("test", "result", "EOL");
          set_fn(nullptr);
          continue;
        }
        if (m_dev != DEV::C29_GPU || dev_sols == 1) {
          std::string result_hash_str;
          for (unsigned i = 0; i != m_batch; ++ i) {
            if (i) result_hash_str += " ";
            char hash[HASH_LEN*2+1];
            result_hash_str += hash_bin2hex(hash, i);
          }
          send_msg("test", "result", result_hash_str);
          if (m_dev != DEV::C29_GPU) set_fn(nullptr); // no async solutions anymore
        }
        continue;
      }

      // PearlHash's work unit is GEMM MACs (m*n*k) so its rate matches the "TH/s" GEMM bench. Equihash's
      // unit is a Wagner SOLUTION (the solver finds ~1.88 distinct proofs per nonce); the solver returns
      // that distinct count in dev_sols when benching, so the rate reads out as Sol/s. Others = 1/batch.
      m_hash_count += (m_dev == DEV::PEARLHASH_GPU)
        ? pearlhash_attempt_hashes(m_batch, m_pearlhash_n, m_pearlhash_k)
                    : (m_dev == DEV::ZELHASH_GPU || m_dev == DEV::BEAMHASH3_GPU) ? static_cast<uint64_t>(dev_sols < 0 ? 0 : dev_sols)
                    : m_batch;
      if (m_dev == DEV::KAWPOW_GPU || m_dev == DEV::ETCHASH_GPU) {
        const uint64_t prev_nonce = m_nonce64;
        if (dev_sols == 1 && m_target)
          send_result(dev_nonce, 8, m_output, nullptr, 32, nullptr, static_cast<uint8_t*>(m_spads));

        std::memcpy(m_input + m_nonce_offset, &m_nonce64, sizeof(m_nonce64));
        m_nonce64 += m_nonce_step;
        if (m_target && nonce_overflowed(prev_nonce, m_nonce64, m_nicehash_mask)) {
          send_error("Nonce overflow");
          set_fn(nullptr);
        }
        continue;
      }
      // Autolykos2, FishHash, and KarlsenHashV2 share the mine-result handling: 8-byte nonce,
      // single 32-byte hash, no mix, nonce embedded in the header at m_nonce_offset.
      if (m_dev == DEV::AUTOLYKOS2_GPU || m_dev == DEV::FISHHASH_GPU || m_dev == DEV::KARLSENHASHV2_GPU) {
        const uint64_t prev_nonce = m_nonce64;
        if (dev_sols == 1 && m_target)
          send_result(dev_nonce, 8, m_output);

        // live Iron Fish (fishhash, 180-byte header) reads the nonce big-endian; embed BE there, LE otherwise
        const uint64_t embed = (m_dev == DEV::FISHHASH_GPU && m_input_len >= 140) ? bswap_64(m_nonce64) : m_nonce64;
        std::memcpy(m_input + m_nonce_offset, &embed, sizeof(embed));
        m_nonce64 += m_nonce_step;
        if (m_target && nonce_overflowed(prev_nonce, m_nonce64, m_nicehash_mask)) {
          set_fn(nullptr);
          send_last_nonce(prev_nonce, m_nonce_bytes, m_pool_id);
        }
        continue;
      }
      if (m_dev == DEV::ZELHASH_GPU) {
        // M5 mining path: the solver writes the target-passing solutions out-of-band into m_spads as
        // [count:u8][count * 52-byte compressed solution]. For each one, emit a result carrying the
        // 53-byte compactSize-prefixed solution (0x34 || 52 B = 106 hex) the Flux/ZIP-301 submit needs.
        const uint64_t prev_nonce = m_nonce64;
        if (dev_sols > 0 && m_target) {
          const uint8_t* const sols = static_cast<const uint8_t*>(m_spads);
          const unsigned count = sols[0];
          for (unsigned i = 0; i < count; ++i) {
            uint8_t submit[1 + 52];
            submit[0] = 0x34;   // compactSize(52)
            std::memcpy(submit + 1, sols + 1 + static_cast<size_t>(i) * 52, 52);
            send_result(dev_nonce, m_nonce_bytes, m_output, nullptr, 32, nullptr, nullptr, submit, sizeof(submit));
          }
        }

        // Advance the low 8 bytes of the 32-byte header nonce (offset 108) as the search counter.
        std::memcpy(m_input + m_nonce_offset, &m_nonce64, sizeof(m_nonce64));
        m_nonce64 += m_nonce_step;
        if (m_target && nonce_overflowed(prev_nonce, m_nonce64, m_nicehash_mask)) {
          set_fn(nullptr);
          send_last_nonce(prev_nonce, m_nonce_bytes, m_pool_id);
        }
        continue;
      }
      if (m_dev == DEV::BEAMHASH3_GPU) {
        // M5 mining path: the solver already SHA-256-filtered its found proofs against the packed Beam
        // difficulty (carried in m_target_bin), and wrote the passing ones out-of-band into m_spads as
        // [count:u8][count * 104-byte solution]. For each, emit a result carrying the 8-byte Beam nonce
        // (offset 32) and the 104-byte solution (208 hex) the Beam JSON-RPC `solution` submit needs.
        const uint64_t prev_nonce = m_nonce64;
        if (dev_sols > 0 && m_target) {
          const uint8_t* const sols = static_cast<const uint8_t*>(m_spads);
          const unsigned count = sols[0];
          for (unsigned i = 0; i < count; ++i)
            send_result(dev_nonce, m_nonce_bytes, m_output, nullptr, 32, nullptr, nullptr,
                        sols + 1 + static_cast<size_t>(i) * 104, 104);
        }

        // Advance the 8-byte Beam nonce (offset 32) as the search counter. The nonce is written to the
        // blob BIG-endian (bswap), matching set_job's convention -- so the pool nonceprefix occupies the
        // LEADING physical bytes = the HIGH bytes of m_nonce64, which m_nicehash_mask fixes (standard
        // nonce-at-32 nicehash). The low bytes are the free search counter, advanced normally.
        const uint64_t embed = bswap_64(m_nonce64);
        std::memcpy(m_input + m_nonce_offset, &embed, sizeof(embed));
        m_nonce64 += m_nonce_step;
        if (m_target && nonce_overflowed(prev_nonce, m_nonce64, m_nicehash_mask)) {
          set_fn(nullptr);
          send_last_nonce(prev_nonce, m_nonce_bytes, m_pool_id);
        }
        continue;
      }
      if (m_dev == DEV::PEARLHASH_GPU) {
        const uint64_t prev_nonce = m_nonce64;
        // Emit at most once per (job_id, header) pair since the PlainProof rebuild is heavy and the
        // pool credits one share per job. PearlHashpool varies job_id (reuses header); HeroMiners the reverse.
        if (dev_sols == 1 && m_target) {
          std::string pearlhash_key = m_job_id;
          pearlhash_key.append(reinterpret_cast<const char*>(m_input), m_input_len);
          if (pearlhash_key != m_pearlhash_proof_job) {
            m_pearlhash_proof_job = pearlhash_key;
            MessageValues values;
            values["nonce"]       = nonce_to_hex(dev_nonce, 8); // winning seed, for logging/traceability
            values["plain_proof"] = pearlhash_proof();     // builds the proof for the captured tile (lazy)
            values["pool_id"]     = m_pool_id;
            values["worker_id"]   = m_worker_id;
            values["job_id"]      = m_job_id;
            send_msg("result", values);
          }
        }
        m_nonce64 += m_nonce_step;   // advance to the next seed (the blob is not touched)
        if (m_target && nonce_overflowed(prev_nonce, m_nonce64, m_nicehash_mask)) {
          set_fn(nullptr);
          send_last_nonce(prev_nonce, 8, m_pool_id);   // pearlhash seed is 64-bit
        }
        continue;
      }
      if (m_nonce_bytes == 4) {
        const uint32_t prev_nonce = m_nonce32;

        if (m_dev != DEV::C29_GPU) {
          for (unsigned i = 0; i != m_batch; ++i) {
            uint32_t* const pnonce = get_nonce32(i);
            if (m_target && *get_result(i) < m_target)
              send_result(bswap_32(*pnonce), 4, m_output + HASH_LEN * i);
            *pnonce = m_nonce32;
            m_nonce32 += m_nonce_step;
          }
        } else {
          if (dev_sols == 1 && m_target && *get_result() < m_target)
            send_result(dev_nonce, 4, m_output, static_cast<uint32_t*>(m_spads), m_c29_proof_size);
          *get_nonce32() = bswap_32(m_nonce32);
          m_nonce32 += m_nonce_step;
        }

        // fail if the nonce wrapped, or the nicehash-protected high bits changed
        if (m_target && nonce_overflowed(prev_nonce, m_nonce32, static_cast<uint32_t>(m_nicehash_mask))) {
          send_error("Nonce overflow");
          set_fn(nullptr);
          continue;
        }
      } else {
        const uint64_t prev_nonce = m_nonce64;

        if (m_dev != DEV::C29_GPU) {
          for (unsigned i = 0; i != m_batch; ++i) {
            uint64_t* const pnonce = get_nonce64(i);
            if (m_target && *get_result(i) < m_target)
              send_result(bswap_64(*pnonce), 8, m_output + HASH_LEN * i);
            *pnonce = m_nonce64;
            m_nonce64 += m_nonce_step;
          }
        } else {
          if (dev_sols == 1 && m_target && *get_result() < m_target)
            send_result(dev_nonce, 8, m_output, static_cast<uint32_t*>(m_spads), m_c29_proof_size);
          *get_nonce64() = bswap_64(m_nonce64);
          m_nonce64 += m_nonce_step;
        }

        // fail if the nonce wrapped, or the nicehash-protected high bits changed
        if (m_target && nonce_overflowed(prev_nonce, m_nonce64, m_nicehash_mask)) {
          send_error("Nonce overflow");
          set_fn(nullptr);
          continue;
        }
      }
    }
  }
}

AsyncWorker* create_worker(
  napi_env env, napi_value data, napi_value complete, napi_value error_callback, napi_value options
) {
  return new Core(env, data, complete, error_callback, options);
}
