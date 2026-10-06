struct LoopStats {
  uint64_t window_start = 0, dispatch = 0, msg = 0, post = 0, iters = 0, jobs = 0,
           max_msg = 0, max_post = 0, dispatch_end = 0;
  double cpu_start = -1.0;
};

static uint64_t loop_now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()
  ).count();
}

static double loop_cpu_seconds() {
#if defined(_WIN32)
  // MSVC clock() is elapsed wall time. Sum every thread's kernel and user time so the worker's
  // aggregate process load remains comparable with POSIX clock(), including values above 100%.
  FILETIME creation_time{}, exit_time{}, kernel_time{}, user_time{};
  if (!GetProcessTimes(GetCurrentProcess(), &creation_time, &exit_time, &kernel_time, &user_time))
    return -1.0;
  ULARGE_INTEGER kernel_ticks{}, user_ticks{};
  kernel_ticks.LowPart = kernel_time.dwLowDateTime;
  kernel_ticks.HighPart = kernel_time.dwHighDateTime;
  user_ticks.LowPart = user_time.dwLowDateTime;
  user_ticks.HighPart = user_time.dwHighDateTime;
  return static_cast<double>(kernel_ticks.QuadPart + user_ticks.QuadPart) / 10000000.0;
#else
  const std::clock_t value = std::clock();
  if (value == static_cast<std::clock_t>(-1))
    return -1.0;
  return static_cast<double>(value) / static_cast<double>(CLOCKS_PER_SEC);
#endif
}

// log a one-off stall line when a single loop phase blocks the compute loop for >100ms
static void loop_stall(const char* phase, const uint64_t us) {
  if (us > 100000)
    fprintf(stderr, "LOOPSTALL t=%llu %s=%.1fms\n",
      static_cast<unsigned long long>(time(nullptr)), phase, us / 1e3);
}

static void loop_stats_report(LoopStats& ls, const uint64_t now) {
  const uint64_t wall = now - ls.window_start;
  const double cpu_now = loop_cpu_seconds();
  const bool valid_cpu = ls.cpu_start >= 0.0 && cpu_now >= ls.cpu_start;
  const double cpu_percent = valid_cpu
      ? 100.0 * (cpu_now - ls.cpu_start) / (wall / 1.0e6)
      : -1.0;
  fprintf(stderr,
    "LOOPSTAT t=%llu wall=%.1fs cpu=%.1f%% dispatch=%.1f%% msg=%.3fs post=%.3fs"
    " iters=%llu jobs=%llu"
    " max_msg=%.1fms max_post=%.1fms\n",
    static_cast<unsigned long long>(time(nullptr)),
    wall / 1e6, cpu_percent, 100.0 * ls.dispatch / wall, ls.msg / 1e6, ls.post / 1e6,
    static_cast<unsigned long long>(ls.iters), static_cast<unsigned long long>(ls.jobs),
    ls.max_msg / 1e3, ls.max_post / 1e3);
  fflush(stderr);
  ls = LoopStats{};
  ls.window_start = now;
  ls.cpu_start = cpu_now;
}

void Core::Execute() try {
  debug_async_worker("Core::Execute entered");
  const bool loop_stats = std::getenv("MOM_LOOP_STATS") != nullptr;
  LoopStats ls;
  bool runtime_initialized = false;
  auto init_runtime = [&]() {
    if (runtime_initialized) return;
    runtime_initialized = true;
    debug_async_worker("runtime init start");

    argon2_select_impl();
    debug_async_worker((std::string("argon2 impl ") + argon2_get_impl_name()).c_str());

#if !defined(_WIN32)
    const xmrig::ICpuInfo* const cpu = xmrig::Cpu::info();
    if (cpu->arch() == xmrig::ICpuInfo::ARCH_ZEN)
      xmrig::RxFix::setupMainLoopExceptionFrame();
    if (cpu->has(xmrig::ICpuInfo::FLAG_SSE41)) {
      rx_blake2b_compress = rx_blake2b_compress_sse41;
    }
    if (cpu->hasAVX2()) {
      rx_blake2b = blake2b_avx2;
    }
#endif

    randomx_set_scratchpad_prefetch_mode(0);
#if defined(_WIN32)
    randomx_set_huge_pages_jit(false);
    randomx_set_optimized_dataset_init(0);
#else
    randomx_set_huge_pages_jit(true);
    randomx_set_optimized_dataset_init(1);
#endif
    debug_async_worker("runtime init done");
  };

  unsigned hashrate_check_counter = HASHRATE_COUNTER_INTERVAL;
  while (true) {
    const uint64_t t_loop_start = loop_stats ? loop_now_us() : 0;
    if (loop_stats) {
      if (ls.dispatch_end) {
        const uint64_t post_us = t_loop_start - ls.dispatch_end;
        ls.post += post_us;
        if (post_us > ls.max_post) {
          ls.max_post = post_us;
        }
        loop_stall("post", post_us);
        ls.dispatch_end = 0;
      }
      if (!ls.window_start) {
        ls.window_start = t_loop_start;
        ls.cpu_start = loop_cpu_seconds();
      } else if (t_loop_start - ls.window_start > 10*1000*1000) {
        loop_stats_report(ls, t_loop_start);
      }
    }

    const std::deque<Message> messages = fromNode.drain();
    for (const auto& message : messages) {
      try {
        debug_async_worker(("message " + message.name).c_str());
        if (message.name == "job" || message.name == "bench" || message.name == "test")
          init_runtime();
        if (loop_stats && (message.name == "job" || message.name == "bench")) {
          ++ls.jobs;
        }
        if (!process_message(message.name, message.values)) return;
      } catch (const std::string& err) {
        send_error(std::string("Message processing exception: ") + err);
      }
    }
    if (loop_stats && !messages.empty()) {
      const uint64_t msg_us = loop_now_us() - t_loop_start;
      ls.msg += msg_us;
      if (msg_us > ls.max_msg) {
        ls.max_msg = msg_us;
      }
      loop_stall("msg", msg_us);
    }

    // we skip first hash function run using m_hash_count check to exclude GPU compile time
    // that effectively skips it in test mode too
    const uint64_t hash_count = m_hash_count.load(std::memory_order_relaxed);
    // GPU batches can last seconds: check the 10 s benchmark / 60 s mining clock after each.
    // Keep ten-batch clock sampling for short CPU dispatches.
    const bool is_gpu = m_dev != DEV::CPU && m_dev != DEV::RX_CPU;
    if (mom::hashrate_check_due(hash_count, is_gpu, hashrate_check_counter,
                                HASHRATE_COUNTER_INTERVAL)) {
      const uint64_t new_timestamp = std::chrono::time_point_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now()
      ).time_since_epoch().count();
      const uint64_t interval_ms = m_job_mode == JobMode::bench ? 10*1000 : 60*1000;
      if (!m_timestamp || new_timestamp - m_timestamp > interval_ms) {
        const uint64_t completed = m_hash_count.exchange(0, std::memory_order_relaxed);
        if (m_timestamp) {
          send_msg("hashrate", "hashrate", std::to_string(
            static_cast<float>(completed) / (new_timestamp - m_timestamp) * 1000.0f
          ));
        }
        m_timestamp = new_timestamp;
      }
    }

    if (!m_has_fn) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    { // A compute function is active here (early-continue above), so compute a batch.
      init_runtime();
      // Only one device runs per dispatch, so a single solution-count + nonce/seed pair is shared
      // across all the GPU cases. dev_nonce doubles as the kernel's in/out found-nonce slot (and as
      // pearlhash's search seed); dev_sols is the per-call solution count (or C29's -1 EOL / 0 pending).
      int dev_sols = 0;
      uint64_t dev_nonce = 0;
      const bool is_test = m_job_mode == JobMode::test;
      const bool is_benchmark = m_job_mode == JobMode::bench;
      const unsigned dispatch_job_ref = m_job_ref.load(std::memory_order_relaxed);
      const uint64_t t_dispatch = loop_stats ? loop_now_us() : 0;
      try {
        switch (m_dev) {
          case DEV::CPU:
            m_fn.cpu(m_input, m_input_len, m_output, m_ctx, m_height);
            break;
          case DEV::GPU:
            m_fn.gpu_cn(m_input, m_input_len, m_output, m_batch, m_dev_str, m_backend);
            break;
          case DEV::C29_GPU:
            // Test vectors solve the supplied blob directly; their reported nonce is irrelevant and
            // may not use the mining layout. Keep it deterministic without reading outside the blob.
            if (is_test) {
              dev_nonce = 0;
            } else if (m_nonce_bytes == 4) {
              dev_nonce = mom::job_boundary::load_nonce<uint32_t>(nonce_address(), true);
            } else {
              dev_nonce = mom::job_boundary::load_nonce<uint64_t>(nonce_address(), true);
            }
            dev_sols = m_fn.gpu_c29(
              dispatch_job_ref, m_c29_proof_size, m_input, m_input_len, m_output,
              static_cast<uint32_t*>(m_spads), &dev_nonce, m_dev_str
            );
            break;
          case DEV::C30_GPU:
            dev_nonce = m_nonce64;
            dev_sols = m_fn.gpu_c29(
              dispatch_job_ref, m_c29_proof_size, m_input, m_input_len, m_output,
              static_cast<uint32_t*>(m_spads), &dev_nonce, m_dev_str
            );
            break;
          case DEV::KAWPOW_GPU:
            dev_nonce = mom::job_boundary::load_nonce<uint64_t>(nonce_address(), false);
            dev_sols = m_fn.gpu_kawpow(
              dispatch_job_ref, m_height, m_input, m_input_len, m_output,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target,
              m_batch, is_test, is_benchmark, m_dev_str
            );
            break;
          case DEV::ETCHASH_GPU:
            dev_nonce = mom::job_boundary::load_nonce<uint64_t>(nonce_address(), false);
            dev_sols = m_fn.gpu_etchash(
              dispatch_job_ref, m_height, m_input, m_input_len, m_output,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target_bin, m_seed,
              m_batch, is_test, is_benchmark, m_dev_str
            );
            break;
          case DEV::AUTOLYKOS2_GPU:
            dev_nonce = mom::job_boundary::load_nonce<uint64_t>(nonce_address(), false);
            dev_sols = m_fn.gpu_autolykos2(
              dispatch_job_ref, m_height, m_input, m_input_len, m_output,
              &dev_nonce, m_target_bin,
              m_batch, is_test, is_benchmark, m_dev_str
            );
            break;
          case DEV::PEARLHASH_GPU:
            dev_nonce = m_nonce64;   // the search seed is internal (not embedded in the blob)
            dev_sols = m_fn.gpu_pearlhash(
              dispatch_job_ref, m_height, m_input, m_input_len, m_output,
              &dev_nonce, m_target_bin,
              m_batch, is_test, is_benchmark, m_dev_str, m_backend,
              m_pearlhash_n, m_pearlhash_k, m_pearlhash_rank, m_pearlhash_cert_version
            );
            break;
          case DEV::FISHHASH_GPU:
            dev_nonce = mom::job_boundary::load_nonce<uint64_t>(
              nonce_address(), mom::job_boundary::nonce_big_endian(m_algo_str, m_input_len));
            dev_sols = m_fn.gpu_fishhash(
              dispatch_job_ref, m_height, m_input, m_input_len, m_output,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target_bin, m_seed,
              m_batch, is_test, is_benchmark, m_dev_str
            );
            break;
          case DEV::KARLSENHASHV2_GPU:
            dev_nonce = mom::job_boundary::load_nonce<uint64_t>(nonce_address(), false);
            dev_sols = m_fn.gpu_karlsenhashv2(
              dispatch_job_ref, m_height, m_input, m_input_len, m_output,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target_bin, m_seed,
              m_batch, is_test, is_benchmark, m_dev_str
            );
            break;
          case DEV::MISC_GPU:
            dev_nonce = mom::job_boundary::load_nonce<uint64_t>(
              nonce_address(), mom::job_boundary::nonce_big_endian(m_algo_str, m_input_len));
            dev_sols = m_fn.gpu_misc(
              dispatch_job_ref, m_height, m_input, m_input_len, m_output,
              nullptr, &dev_nonce, m_target_bin, m_seed,
              m_batch, is_test, is_benchmark, m_dev_str
            );
            break;
          case DEV::VERTHASH_GPU: {
            dev_nonce = mom::job_boundary::load_nonce<uint32_t>(nonce_address(), false);
            dev_sols = m_fn.gpu_verthash(
              dispatch_job_ref, m_height, m_input, m_input_len, m_output,
              nullptr, &dev_nonce, m_target_bin, m_seed,
              m_batch, is_test, is_benchmark, m_dev_str
            );
            break;
          }
          case DEV::ZELHASH_GPU:
            // The 32-byte nonce starts at offset 108; the counter follows any pool prefix. Solvers
            // write compact proofs (or test rows) out-of-band into m_spads.
            dev_nonce = mom::job_boundary::load_nonce<uint64_t>(nonce_address(), false);
            dev_sols = m_fn.gpu_zelhash(
              dispatch_job_ref, m_height, m_input, m_input_len,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target_bin,
              m_batch, is_test, is_benchmark, m_dev_str
            );
            break;
          case DEV::BEAMHASH3_GPU:
            // c29-like: input is the prework||nonce||extranonce blob; the 8-byte Beam nonce is at
            // offset 32. The solver writes the 104-byte solution(s) out-of-band into m_spads (or, in
            // is_test, the gen-validation rows).
            // Keep the numeric value matching a native little-endian read: Beam submissions reverse
            // this display value back to the physical nonce bytes sent in the solved header.
            dev_nonce = mom::job_boundary::load_nonce<uint64_t>(nonce_address(), false);
            dev_sols = m_fn.gpu_beamhash3(
              dispatch_job_ref, m_height, m_input, m_input_len,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target_bin,
              m_batch, is_test, is_benchmark, m_dev_str
            );
            break;
          case DEV::RX_CPU:
            // RandomX work runs only in the per-job thread pool.
            break;
        }
      } catch (const std::string& err) {
        send_error(std::string("Compute function exception: ") + err);
        clear_fn();
        continue;
      } catch (const std::exception& e) {
        // Surface sycl::exception (and any std::exception) detail instead of a bare message.
        send_error(std::string("Compute function exception: ") + e.what());
        clear_fn();
        continue;
      } catch (...) {
        send_error("Compute function exception");
        clear_fn();
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
            char hash[HASH_LEN*2+1];
            std::string result = hash_bin2hex(hash, 0);
            if (m_algo_str != "octopus") {
              char mix[HASH_LEN*2+1];
              result += " ";
              result += hash_bin2hex(static_cast<uint8_t*>(m_spads), mix);
            }
            send_msg("test", "result", result);
          } else {
            send_error("No " + m_algo_str + " test result");
          }
          clear_fn();
          continue;
        }
        // MISC GPU test APIs return one hash; intensity controls tested lanes, not output hashes.
        if (m_dev == DEV::AUTOLYKOS2_GPU || m_dev == DEV::FISHHASH_GPU ||
            m_dev == DEV::KARLSENHASHV2_GPU || m_dev == DEV::VERTHASH_GPU ||
            m_dev == DEV::MISC_GPU) {
          if (dev_sols == 1) {
            char hash[HASH_LEN*2+1];
            send_msg("test", "result", hash_bin2hex(hash, 0));
          } else {
            send_error("No " + m_algo_str + " test result");
          }
          clear_fn();
          continue;
        }
        if (m_dev == DEV::PEARLHASH_GPU) {
          if (dev_sols == 1) {
            send_msg("test", "result", "ok");
          } else {
            send_error("No pearlhash test result");
          }
          clear_fn();
          continue;
        }
        if (m_dev == DEV::ZELHASH_GPU || m_dev == DEV::BEAMHASH3_GPU) {
          // M1 gen-kernel validation (or the SOLVE path): the kernel dumped the rows / solution(s) into
          // m_spads; emit the whole buffer as hex so the standalone checker can diff it against the oracle.
          if (dev_sols == 1) {
            const uint8_t* const rows = static_cast<const uint8_t*>(m_spads);
            std::string hex;
            hex.reserve(SMALL_BLOB_SOL_LEN * 2);
            for (unsigned i = 0; i != SMALL_BLOB_SOL_LEN; ++i) {
              hex += hf_bin2hex(rows[i] >> 4);
              hex += hf_bin2hex(rows[i] & 0xF);
            }
            send_msg("test", "result", hex);
          } else {
            send_error("No " + m_algo_str + " test result");
          }
          clear_fn();
          continue;
        }
        if (m_dev == DEV::C29_GPU && dev_sols == 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
        if (m_dev == DEV::C29_GPU && dev_sols == -1) {
          send_msg("test", "result", "EOL");
          clear_fn();
          continue;
        }
        if (m_dev != DEV::C29_GPU || dev_sols == 1) {
          std::string result_hash_str;
          for (unsigned i = 0; i != m_batch; ++i) {
            if (i) {
              result_hash_str += " ";
            }
            char hash[HASH_LEN*2+1];
            result_hash_str += hash_bin2hex(hash, i);
          }
          send_msg("test", "result", result_hash_str);
          if (m_dev != DEV::C29_GPU) {
            clear_fn(); // no async solutions anymore
          }
        }
        continue;
      }

      // C29 counts new graphs, not proof recovery. PearlHash counts rank-128-equivalent GEMM MACs.
      // Equihash 192,7 counts input headers like miniZ; the other
      // Equihash solvers count returned proofs. All remaining GPU paths count the requested batch.
      uint64_t completed = m_batch;
      if (m_dev == DEV::C29_GPU) {
        completed = mom::job_boundary::c29_completed_graphs(dev_sols);
      } else if (m_dev == DEV::PEARLHASH_GPU) {
        completed = pearlhash_attempt_hashes(
          m_batch, m_pearlhash_n, m_pearlhash_k, m_pearlhash_rank
        );
      } else if ((m_dev == DEV::ZELHASH_GPU && m_algo_str != "equihash192_7") ||
                 m_dev == DEV::BEAMHASH3_GPU) {
        completed = static_cast<uint64_t>(dev_sols < 0 ? 0 : dev_sols);
      }
      m_hash_count.fetch_add(completed, std::memory_order_relaxed);
      if (m_dev == DEV::KAWPOW_GPU || m_dev == DEV::ETCHASH_GPU) {
        if (dev_sols == 1 && m_target) {
          if (m_algo_str == "octopus") {
            send_result(dev_nonce, 8, m_output);
          } else {
            send_result(
              dev_nonce, 8, m_output, nullptr, 32, nullptr, static_cast<uint8_t*>(m_spads)
            );
          }
        }

        if (m_nonce_exhausted) {
          send_error("Nonce overflow");
          clear_fn();
          continue;
        }
        const uint64_t first_nonce = m_nonce64;
        std::memcpy(m_input + m_nonce_offset, &m_nonce64, sizeof(m_nonce64));
        m_nonce64 = next_nonce_after_batch(
          first_nonce, m_batch, 1, m_nonce_step, m_nicehash_mask);
        continue;
      }
      // These single-blob algorithms share an 8-byte nonce, one 32-byte hash, and no mix hash.
      if (m_dev == DEV::AUTOLYKOS2_GPU || m_dev == DEV::FISHHASH_GPU ||
          m_dev == DEV::KARLSENHASHV2_GPU || m_dev == DEV::MISC_GPU) {
        if (dev_sols == 1 && m_target)
          send_result(dev_nonce, 8, m_output);

        if (m_nonce_exhausted) {
          clear_fn();
          send_last_nonce(m_nonce64, m_nonce_bytes, m_pool_id, m_job_id, m_job_token);
          continue;
        }
        const uint64_t first_nonce = m_nonce64;
        mom::job_boundary::store_nonce(
          nonce_address(), m_nonce64,
          mom::job_boundary::nonce_big_endian(m_algo_str, m_input_len));
        m_nonce64 = next_nonce_after_batch(
          first_nonce, m_batch, 1, m_nonce_step, m_nicehash_mask);
        continue;
      }
      if (m_dev == DEV::VERTHASH_GPU) {
        if (dev_sols == 1 && m_target) {
          send_result(dev_nonce, 4, m_output);
        }

        if (m_nonce_exhausted) {
          clear_fn();
          send_last_nonce(m_nonce32, 4, m_pool_id, m_job_id, m_job_token);
          continue;
        }
        const uint32_t first_nonce = m_nonce32;
        std::memcpy(m_input + m_nonce_offset, &m_nonce32, sizeof(m_nonce32));
        m_nonce32 = next_nonce_after_batch(
          first_nonce, m_batch, 1, m_nonce_step,
          static_cast<uint32_t>(m_nicehash_mask));
        continue;
      }
      if (m_dev == DEV::ZELHASH_GPU) {
        // M5 mining path: the solver writes the target-passing solutions out-of-band into m_spads as
        // [count:u8][compressed solutions]. Prefix each proof with its CompactSize for ZIP-301.
        if (dev_sols > 0 && m_target) {
          const unsigned solution_size = m_algo_str == "equihash192_7" ? 400 :
                                         m_algo_str == "zhash" ? 100 : 52;
          send_equihash_results(dev_nonce, solution_size, true);
        }

        // Advance the low 8 bytes of the 32-byte header nonce (offset 108) as the search counter.
        if (m_nonce_exhausted) {
          clear_fn();
          send_last_nonce(m_nonce64, m_nonce_bytes, m_pool_id, m_job_id, m_job_token);
          continue;
        }
        const uint64_t first_nonce = m_nonce64;
        std::memcpy(m_input + m_nonce_offset, &m_nonce64, sizeof(m_nonce64));
        m_nonce64 = next_nonce_after_batch(
          first_nonce, m_batch, 1, m_nonce_step, m_nicehash_mask);
        continue;
      }
      if (m_dev == DEV::BEAMHASH3_GPU) {
        // M5 mining path: the solver already SHA-256-filtered its found proofs against the packed Beam
        // difficulty (carried in m_target_bin), and wrote the passing ones out-of-band into m_spads as
        // [count:u8][count * 104-byte solution]. For each, emit a result carrying the 8-byte Beam nonce
        // (offset 32) and the 104-byte solution (208 hex) the Beam JSON-RPC `solution` submit needs.
        if (dev_sols > 0 && m_target) {
          send_equihash_results(dev_nonce, 104, false);
        }

        // The pool prefix occupies the leading physical bytes, so Beam stores its counter big-endian.
        if (m_nonce_exhausted) {
          clear_fn();
          send_last_nonce(m_nonce64, m_nonce_bytes, m_pool_id, m_job_id, m_job_token);
          continue;
        }
        const uint64_t first_nonce = m_nonce64;
        mom::job_boundary::store_nonce(nonce_address(), m_nonce64, true);
        m_nonce64 = next_nonce_after_batch(
          first_nonce, m_batch, 1, m_nonce_step, m_nicehash_mask);
        continue;
      }
      if (m_dev == DEV::PEARLHASH_GPU) {
        const uint64_t prev_nonce = m_nonce64;
        // Distinct winning seeds may be credited under the same pool job/header.
        if (dev_sols == 1 && m_target) {
          MessageValues values;
          values["nonce"]       = nonce_to_hex(dev_nonce, 8); // winning seed, for logging/traceability
          uint8_t jackpot[HASH_LEN];
          uint32_t adjustment_factor = 0;
          const char* const plain_proof = pearlhash_claim(jackpot, &adjustment_factor);
          if (!plain_proof) {
            send_error("PearlHash proof claim unavailable");
          } else {
            values["plain_proof"] = plain_proof; // builds the proof for the captured tile (lazy)
            char jackpot_hex[HASH_LEN * 2 + 1];
            values["jackpot"] = hash_bin2hex(jackpot, jackpot_hex, 0);
            values["adjustment_factor"] = std::to_string(adjustment_factor);
            values["pool_id"]     = m_pool_id;
            values["worker_id"]   = m_worker_id;
            values["job_id"]      = m_job_id;
            values["job_token"]   = m_job_token;
            send_msg("result", values);
          }
        }
        // PearlHash kernels consume only the low 32 bits of the external seed. Advance in the
        // negotiated worker partition and stop before the next value would wrap/repeat there.
        std::uint32_t next_seed = 0;
        const std::uint64_t pearl_seed_step =
          static_cast<std::uint64_t>(m_thread_num) * m_pearlhash_seed_stride;
        if (!mom::job_boundary::pearl_seed_next(
              static_cast<std::uint32_t>(prev_nonce), pearl_seed_step, next_seed)) {
          clear_fn();
          if (m_target) {
            send_last_nonce(prev_nonce, 8, m_pool_id, m_job_id, m_job_token);
          }
        } else {
          m_nonce64 = next_seed;
        }
        continue;
      }
      if (m_dev == DEV::C30_GPU) {
        const uint64_t prev_nonce = m_nonce64;
        if (dev_sols == 1 && m_target && std::memcmp(m_output, m_target_bin, HASH_LEN) <= 0)
          send_result(dev_nonce, 8, m_output, static_cast<uint32_t*>(m_spads), 42);
        m_nonce64 += m_nonce_step;
        if (m_target && mom::job_boundary::nonce_overflowed(
            prev_nonce, m_nonce64, m_nicehash_mask)) {
          clear_fn();
          send_last_nonce(prev_nonce, 8, m_pool_id, m_job_id, m_job_token);
        }
        continue;
      }
      if (m_nonce_bytes == 4) {
        if (m_dev != DEV::C29_GPU) {
          for (unsigned i = 0; i != m_batch; ++i) {
            const uint32_t nonce = mom::job_boundary::load_nonce<uint32_t>(
              nonce_address(i), false);
            if (m_target && result_word(i) < m_target)
              send_result(bswap_32(nonce), 4, m_output + HASH_LEN * i);
          }
        } else {
          if (dev_sols == 1 && m_target && result_word() < m_target)
            send_result(dev_nonce, 4, m_output, static_cast<uint32_t*>(m_spads), m_c29_proof_size);
        }

        if (m_nonce_exhausted) {
          send_error("Nonce overflow");
          clear_fn();
          continue;
        }
        const uint32_t first_nonce = m_nonce32;
        for (unsigned i = 0; i != m_batch; ++i)
          mom::job_boundary::store_nonce(
            nonce_address(i), first_nonce + i * m_nonce_step, true);
        m_nonce32 = next_nonce_after_batch(
          first_nonce, m_batch, m_nonce_step,
          static_cast<uint64_t>(m_batch) * m_nonce_step,
          static_cast<uint32_t>(m_nicehash_mask));
      } else {
        if (m_dev != DEV::C29_GPU) {
          for (unsigned i = 0; i != m_batch; ++i) {
            const uint64_t nonce = mom::job_boundary::load_nonce<uint64_t>(
              nonce_address(i), false);
            if (m_target && result_word(i) < m_target)
              send_result(bswap_64(nonce), 8, m_output + HASH_LEN * i);
          }
        } else {
          if (dev_sols == 1 && m_target && result_word() < m_target)
            send_result(dev_nonce, 8, m_output, static_cast<uint32_t*>(m_spads), m_c29_proof_size);
        }

        if (m_nonce_exhausted) {
          send_error("Nonce overflow");
          clear_fn();
          continue;
        }
        const uint64_t first_nonce = m_nonce64;
        for (unsigned i = 0; i != m_batch; ++i)
          mom::job_boundary::store_nonce(
            nonce_address(i), first_nonce + i * m_nonce_step, true);
        m_nonce64 = next_nonce_after_batch(
          first_nonce, m_batch, m_nonce_step,
          static_cast<uint64_t>(m_batch) * m_nonce_step,
          m_nicehash_mask);
      }
    }
  }
} catch (...) {
  // The normal close message releases these resources in process_message(). If the compute loop
  // unwinds first, release them here while the Core object is still alive.
  free_memory();
  throw;
}

std::unique_ptr<AsyncWorker> create_worker(
  napi_env env, napi_value data, napi_value complete, napi_value error_callback
) {
  return std::make_unique<Core>(env, data, complete, error_callback);
}
