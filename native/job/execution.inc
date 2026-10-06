void ghostrider(
  const uint8_t* input, const size_t input_size, uint8_t* const output,
  cryptonight_ctx** const ctx, const uint64_t height
) {
  xmrig::ghostrider::hash_octa(input, input_size, output, ctx, nullptr);
}

static void init_rx_dataset_thread(
  randomx_dataset* const dataset, randomx_cache* const cache,
  const unsigned long start, const unsigned long count
) {
  if (xmrig::Cpu::info()->hasAVX2() && (count % 5)) {
    randomx_init_dataset(dataset, cache, start, count - (count % 5));
    randomx_init_dataset(dataset, cache, start + count - 5, 5);
  } else randomx_init_dataset(dataset, cache, start, count);
}

static randomx_flags get_rx_vm_flags(
  const std::string& algo, const bool is_rx_jit, const randomx_dataset* const m_rx_dataset,
  const xmrig::VirtualMemory* const m_rx_dataset_mem
) {
  const xmrig::ICpuInfo* const cpu = xmrig::Cpu::info();
  unsigned rx_flags = RANDOMX_FLAG_DEFAULT;
  if (algo != "panthera" && m_rx_dataset_mem->isHugePages()) {
    rx_flags |= RANDOMX_FLAG_LARGE_PAGES;
  }
  if (cpu->hasAES()) {
    rx_flags |= RANDOMX_FLAG_HARD_AES;
  }
  if (m_rx_dataset) {
    rx_flags |= RANDOMX_FLAG_FULL_MEM;
  }
  if (is_rx_jit) {
    rx_flags |= RANDOMX_FLAG_JIT;
  }
  const auto assembly = cpu->assembly();
  if (assembly == xmrig::Assembly::RYZEN || assembly == xmrig::Assembly::BULLDOZER)
    rx_flags |= RANDOMX_FLAG_AMD;
  return static_cast<randomx_flags>(rx_flags);
}

void Core::set_job(
  const JobMode mode, const bool is_no_same_input, const MessageValues& v,
  std::function<void(void)> fn_extra_setup
) {
  if (!v.contains("dev"))      throw std::string("Missing dev job key");
  if (!v.contains("algo"))     throw std::string("Missing algo job key");
  if (!v.contains("blob_hex")) throw std::string("Missing blob_hex job key");
  const bool is_set_nonce = mode != JobMode::test;

  const auto parse_uint = [](
    const std::string& value, const unsigned base, const uint64_t max, const char* const key
  ) {
    uint64_t parsed = 0;
    if (!mom::job_boundary::parse_unsigned(value, parsed, base, base == 16) || parsed > max) {
      throw std::string("Invalid unsigned job key: ") + key;
    }
    return parsed;
  };

  // Optional job keys fall back to a default when absent; nonce and mask are hexadecimal.
  const auto opt_str = [&](const char* k) { return v.contains(k) ? v.at(k) : std::string(); };
  const auto opt_uint = [&](const char* k, unsigned def) {
    return v.contains(k)
      ? static_cast<unsigned>(parse_uint(v.at(k), 10, std::numeric_limits<unsigned>::max(), k))
      : def;
  };
  const auto opt_u64_hex = [&](const char* k) {
    return v.contains(k) ? parse_uint(v.at(k), 16, std::numeric_limits<uint64_t>::max(), k) : 0ull;
  };
  const std::string new_dev_str        = v.at("dev"),
                    new_algo_str       = v.at("algo"),
                    new_input_hex      = v.at("blob_hex"),
                    new_seed_hex       = opt_str("seed_hex"),
                    new_backend        = opt_str("backend").empty() ? "auto" : opt_str("backend");
  const unsigned    new_height         = opt_uint("height", 0),
                    new_thread_id      = opt_uint("thread_id", 0),
                    new_thread_num     = opt_uint("thread_num", 1),
                    new_nonce_bytes    = opt_uint("noncebytes", 4),
                    new_nonce_offset   = opt_uint("nonceoffset", 39),
                    new_c29_proof_size = opt_uint("proofsize", 32),
                    new_pearlhash_n        = opt_uint("pearlhash_n", 131072),
                    new_pearlhash_k        = opt_uint("pearlhash_k", 4096),
                    new_pearlhash_rank     = opt_uint("pearlhash_rank", 256),
                    new_pearlhash_cert_version = opt_uint("pearlhash_cert_version", 3);
  uint64_t          new_nonce          = opt_u64_hex("nonce");
  const uint64_t    new_nicehash_mask  = opt_u64_hex("nicehash_mask");

  // GPU launch depth is an explicit job field. CPU keeps cpu*B because B selects a
  // compiled multi-way CPU hash function rather than an algorithm tuning profile.
  std::string_view dev_name_view = new_dev_str;
  std::string_view dev_batch_view;
  const bool has_dev_batch = new_dev_str.find('*') != std::string::npos;
  if (has_dev_batch && !mom::job_boundary::split_once(
      new_dev_str, '*', dev_name_view, dev_batch_view)) {
    throw std::string("Invalid dev specification");
  }
  if (dev_name_view.empty()) {
    throw std::string("Invalid dev specification");
  }
  const std::string new_dev_name(dev_name_view);
  if (new_dev_name.starts_with("gpu") && has_dev_batch) {
    throw std::string("GPU intensity must use the explicit job intensity field");
  }
  const unsigned new_batch = new_dev_name.starts_with("gpu")
    ? opt_uint("intensity", 1)
    : (has_dev_batch
        ? static_cast<unsigned>(parse_uint(
            std::string(dev_batch_view), 10, std::numeric_limits<unsigned>::max(), "dev batch"))
        : 1);
  const DEV new_dev =
    new_dev_name == "cpu" ? (rx_cpu_name2config.contains(new_algo_str) ? DEV::RX_CPU : DEV::CPU) :
    new_algo_str == "kawpow" ? DEV::KAWPOW_GPU :
    new_algo_str == "firopow" ? DEV::KAWPOW_GPU :
    new_algo_str == "evrprogpow" ? DEV::KAWPOW_GPU :
    new_algo_str == "meowpow" ? DEV::KAWPOW_GPU :
    new_algo_str == "etchash" ? DEV::ETCHASH_GPU :
    new_algo_str == "octopus" ? DEV::ETCHASH_GPU :
    new_algo_str == "autolykos2" ? DEV::AUTOLYKOS2_GPU :
    new_algo_str == "pearlhash" ? DEV::PEARLHASH_GPU :
    new_algo_str == "fishhash" ? DEV::FISHHASH_GPU :
    new_algo_str == "karlsenhashv2" ? DEV::KARLSENHASHV2_GPU :
    new_algo_str == "hoohash" ? DEV::MISC_GPU :
    new_algo_str == "walahash" ? DEV::MISC_GPU :
    new_algo_str == "xelishashv3" ? DEV::MISC_GPU :
    new_algo_str == "nexapow" ? DEV::MISC_GPU :
    new_algo_str == "verthash" ? DEV::VERTHASH_GPU :
    new_algo_str == "equihash192_7" ? DEV::ZELHASH_GPU :
    new_algo_str == "zelhash" ? DEV::ZELHASH_GPU :
    new_algo_str == "zhash" ? DEV::ZELHASH_GPU :
    new_algo_str == "beamhash3" ? DEV::BEAMHASH3_GPU :
    new_algo_str == "c29" ? DEV::C29_GPU :
    new_algo_str == "c30" ? DEV::C30_GPU : DEV::GPU;
  std::uint32_t new_pearlhash_seed_base = 0;
  std::uint32_t new_pearlhash_seed_stride = 1;
  std::uint32_t new_pearlhash_seed_start = 0;
  if (new_dev == DEV::PEARLHASH_GPU) {
    if (v.contains("nonce_stride")) {
      const uint64_t stride = parse_uint(
        v.at("nonce_stride"), 10, std::numeric_limits<std::uint32_t>::max(), "nonce_stride");
      if (!stride || (stride & (stride - 1)) != 0)
        throw std::string("PearlHash nonce stride must be a positive power of two");
      new_pearlhash_seed_stride = static_cast<std::uint32_t>(stride);
    }
    if (new_nonce > std::numeric_limits<std::uint32_t>::max())
      throw std::string("PearlHash seed slot exceeds uint32_t");
    new_pearlhash_seed_base = static_cast<std::uint32_t>(new_nonce);
    if (!mom::job_boundary::pearl_seed_start(
          new_pearlhash_seed_base, new_thread_id, new_thread_num,
          new_pearlhash_seed_stride, new_pearlhash_seed_start)) {
      throw std::string("PearlHash seed partition overflows uint32_t");
    }
  }
  // NiceHash reserves physical nonce byte 3; BE generic counters therefore advance in increments of 256.
  const unsigned nonce_unit = new_nonce_bytes == 4 && new_nicehash_mask == 0xff &&
    (new_dev == DEV::CPU || new_dev == DEV::RX_CPU || new_dev == DEV::GPU) ? 256 : 1;

  if (new_batch == 0) throw std::string("Batch size must be positive");
  if (new_thread_num == 0 || new_thread_id >= new_thread_num)
    throw std::string("Invalid worker thread index/count");
  if (new_batch > std::numeric_limits<unsigned>::max() / new_thread_num / nonce_unit)
    throw std::string("Worker nonce step is too large");
  if ((new_dev == DEV::C29_GPU || new_dev == DEV::C30_GPU) && new_batch != 1)
    throw std::string("Cycle solvers require batch size 1");
  if ((new_dev == DEV::ZELHASH_GPU || new_dev == DEV::BEAMHASH3_GPU) && new_batch != 1)
    throw std::string("Single-header solvers require intensity 1");
  if (new_dev == DEV::C29_GPU && new_c29_proof_size != 32 && new_c29_proof_size != 42)
    throw std::string("c29 requires proof size 32 or 42");
  if (new_dev == DEV::C30_GPU && new_c29_proof_size != 42)
    throw std::string("c30 requires proof size 42");
  if (new_backend != "auto" && new_backend != "sycl" &&
      new_backend != "sycl-opencl" && new_backend != "sycl-l0" &&
      new_backend != "sycl-native" &&
      new_backend != "native")
    throw std::string("Invalid GPU backend");
  if (new_dev == DEV::PEARLHASH_GPU) {
    if (new_pearlhash_cert_version != 3)
      throw std::string("Unsupported pearlhash certificate version");
    const uint64_t pearlhash_m = mode == JobMode::test
      ? (new_backend == "native" ? 2048 : 256)
      : (new_batch >= 128 ? new_batch : 131072);
    const uint64_t pearlhash_n = mode == JobMode::test ? pearlhash_m : new_pearlhash_n;
    // Align effective matrices before allocation so transcript chunks, kernel indices, tile
    // searches, 32-depth matrix loads, transcript boundaries, and PlainProof accesses stay bounded.
    if (!mom::job_boundary::valid_pearlhash_shape(
          pearlhash_m, pearlhash_n, new_pearlhash_k, new_pearlhash_rank))
      throw std::string("Invalid pearlhash M/N/K/rank profile");
    if (new_pearlhash_rank < 128)
      throw std::string("PearlHash certificate V3 requires rank >= 128");
  }
  if (new_nonce_bytes != 4 && new_nonce_bytes != 8)
    throw std::string("Only support 4 or 8 bytes long nonces");
  if (new_dev == DEV::PEARLHASH_GPU && new_nonce_bytes != 8)
    throw std::string("pearlhash requires an 8-byte external seed");
  if (new_nonce_bytes == 4 &&
      (new_nonce > std::numeric_limits<uint32_t>::max() ||
       new_nicehash_mask > std::numeric_limits<uint32_t>::max()))
    throw std::string("Nonce and mask must fit the configured 4-byte width");
  if (new_dev == DEV::RX_CPU && new_nonce_bytes != 4)
    throw std::string("RandomX requires a 4-byte nonce");
  if (is_nonce_at_32_gpu_dev(new_dev) && new_dev != DEV::FISHHASH_GPU &&
      (new_nonce_bytes != 8 || new_nonce_offset != 32))
    throw std::string(new_algo_str + " requires an 8-byte nonce at offset 32");
  if (new_dev == DEV::MISC_GPU && new_algo_str != "nexapow" && (new_nonce_bytes != 8 ||
      new_nonce_offset != (new_algo_str == "xelishashv3" ? 40 : 72)))
    throw std::string(new_algo_str + " requires an 8-byte nonce at offset " +
      (new_algo_str == "xelishashv3" ? "40" : "72"));
  if (new_dev == DEV::VERTHASH_GPU && (new_nonce_bytes != 4 || new_nonce_offset != 76))
    throw std::string("verthash requires a 4-byte nonce at offset 76");
  // Zcash-family headers carry a 32-byte nonce at offset 108. Advance an 8-byte counter after any
  // fixed pool prefix while keeping it entirely inside that nonce.
  if (new_dev == DEV::ZELHASH_GPU &&
      (new_nonce_bytes != 8 || new_nonce_offset < 108 || new_nonce_offset > 132))
    throw std::string(new_algo_str + " requires an 8-byte counter inside its 32-byte nonce");
  // BeamHash III: the blob is prework(32)||nonce(8)||extranonce(4); the 8-byte Beam nonce is at offset 32.
  if (new_dev == DEV::BEAMHASH3_GPU && (new_nonce_bytes != 8 || new_nonce_offset != 32))
    throw std::string("beamhash3 requires an 8-byte nonce at offset 32");

  FN new_fn{};
  uint8_t new_seed[MAX_BLOB_LEN]{};
  unsigned new_seed_len = HASH_LEN;
  const RandomX_ConfigurationBase* new_rx_config;
  switch (new_dev) {
    case DEV::CPU: {
      const auto new_algo = algo_lookup(cpu_name2algo, new_algo_str);
      if (new_algo == xmrig::Algorithm::GHOSTRIDER_RTM) {
        if (new_batch != 8) throw std::string("Bad CPU batch");
        new_fn.cpu = ghostrider;
      } else {
        if (new_batch > MAX_CN_CPU_WAYS) throw std::string("Bad CPU batch");
        new_fn.cpu = xmrig::CnHash::fn(
          new_algo,
          cpu_params2variant[new_batch - 1][xmrig::Cpu::info()->hasAES() ? 0 : 1],
          xmrig::Assembly::AUTO
        );
        if (!new_fn.cpu) {
          throw std::string("Unsupported CPU algorithm/batch combination");
        }
      }
      break;
    }

    case DEV::RX_CPU: {
      if (new_seed_hex.empty()) throw std::string("No seed_hex job key");
      if ((new_seed_hex.size() & 1) || new_seed_hex.size() > MAX_BLOB_LEN * 2)
        throw std::string("Bad seed length");
      new_seed_len = new_seed_hex.size() / 2;
      if (!hex2bin(new_seed_hex.c_str(), new_seed_len, new_seed)) throw std::string("Bad seed hex");
      new_rx_config = algo_lookup(rx_cpu_name2config, new_algo_str);
      // All RandomX work is done in m_rx_threads, not the main dispatch loop.
      break;
    }

    case DEV::GPU:
      new_fn.gpu_cn = algo_lookup(gpu_cn_algo2fn, new_algo_str);
      break;

    case DEV::C29_GPU:
    case DEV::C30_GPU:
      new_fn.gpu_c29 = algo_lookup(gpu_c29_algo2fn, new_algo_str);
      break;

    case DEV::KAWPOW_GPU:
      new_fn.gpu_kawpow = algo_lookup(gpu_kawpow_algo2fn, new_algo_str);
      break;

    case DEV::ETCHASH_GPU:
      if (!new_seed_hex.empty()) {
        if (new_seed_hex.size() != HASH_LEN * 2) throw std::string("Bad seed length");
        if (!hex2bin(new_seed_hex.c_str(), HASH_LEN, new_seed)) throw std::string("Bad seed hex");
      }
      new_fn.gpu_etchash = algo_lookup(gpu_etchash_algo2fn, new_algo_str);
      break;

    case DEV::AUTOLYKOS2_GPU:
      new_fn.gpu_autolykos2 = algo_lookup(gpu_autolykos2_algo2fn, new_algo_str);
      break;

    case DEV::PEARLHASH_GPU:
      new_fn.gpu_pearlhash = algo_lookup(gpu_pearlhash_algo2fn, new_algo_str);
      break;

    case DEV::FISHHASH_GPU:
      new_fn.gpu_fishhash = algo_lookup(gpu_fishhash_algo2fn, new_algo_str);
      break;

    case DEV::KARLSENHASHV2_GPU:
      new_fn.gpu_karlsenhashv2 = algo_lookup(gpu_karlsenhashv2_algo2fn, new_algo_str);
      break;

    case DEV::MISC_GPU:
      new_fn.gpu_misc = algo_lookup(gpu_misc_algo2fn, new_algo_str);
      break;

    case DEV::VERTHASH_GPU:
      new_fn.gpu_verthash = algo_lookup(gpu_verthash_algo2fn, new_algo_str);
      break;

    case DEV::ZELHASH_GPU:
      new_fn.gpu_zelhash = algo_lookup(gpu_zelhash_algo2fn, new_algo_str);
      break;

    case DEV::BEAMHASH3_GPU:
      new_fn.gpu_beamhash3 = algo_lookup(gpu_beamhash3_algo2fn, new_algo_str);
      break;
  }

  uint8_t new_input[MAX_BLOB_LEN];
  const std::size_t new_input_size = new_input_hex.size();
  if ((new_input_size & 1) || new_input_size / 2 > MAX_BLOB_LEN)
    throw std::string("Bad input length");
  const unsigned new_input_len = static_cast<unsigned>(new_input_size / 2);
  if (!hex2bin(new_input_hex.c_str(), new_input_len, new_input))
    throw std::string("Bad input hex");
  if (new_dev == DEV::C30_GPU) {
    const bool canonical_search = new_input_len == 32 && new_nonce_bytes == 8 &&
      new_nonce_offset == 0;
    const bool test_vector = mode == JobMode::test && new_input_len == 40 &&
      new_nonce_bytes == 8 && new_nonce_offset == 32;
    // Mining advances an external nonce over the canonical 32-byte header; accepting the
    // embedded 40-byte form outside test mode would repeat one fixed nonce indefinitely.
    if (!canonical_search && !test_vector)
      throw std::string(
        "c30 requires a canonical 32-byte search header with an 8-byte nonce at offset 0; "
        "only test mode accepts the 40-byte header-plus-nonce vector at offset 32");
  }
  // Apply the common fixed/minimum input bounds first. Algorithm-specific layout checks below
  // refine variable forms such as NexaPow's 40- and 48-byte work blobs.
  const unsigned min_input_len = new_dev == DEV::PEARLHASH_GPU ? 76
    : new_dev == DEV::KARLSENHASHV2_GPU ? 80
    : new_dev == DEV::MISC_GPU
      ? (new_algo_str == "xelishashv3" ? 112 : new_algo_str == "nexapow" ? 40 : 80)
    : new_dev == DEV::VERTHASH_GPU ? 80
    : new_dev == DEV::ZELHASH_GPU ? 140 // full Zcash/Flux header
    : new_dev == DEV::BEAMHASH3_GPU ? 44 // prework(32)||nonce(8)||extranonce(4)
    : is_nonce_at_32_gpu_dev(new_dev) ? 40 : 0;
  const bool exact_input_len = new_algo_str == "xelishashv3" ||
    new_dev == DEV::PEARLHASH_GPU || new_dev == DEV::MISC_GPU ||
    new_dev == DEV::VERTHASH_GPU || new_dev == DEV::ZELHASH_GPU;
  bool bad_input_len;
  if (new_algo_str == "nexapow") {
    bad_input_len = new_nonce_bytes != 8 ||
      !mom::job_boundary::valid_nexapow_layout(new_input_len, new_nonce_offset);
  } else if (new_dev == DEV::BEAMHASH3_GPU) {
    bad_input_len = !mom::job_boundary::valid_beamhash3_input(new_input_len);
  } else {
    bad_input_len = exact_input_len ? new_input_len != min_input_len
                                    : new_input_len < min_input_len;
  }
  if (bad_input_len)
    throw std::string("Bad " + new_algo_str + " input length");
  if (new_dev == DEV::GPU && !mom::job_boundary::valid_cn_gpu_input(new_input_len))
    throw std::string("cn/gpu input must be 1 to 135 bytes");
  // FishHash keeps its short offline vector at offset 32; FIP-9 Iron Fish headers end with
  // an 8-byte big-endian nonce at offset 172.
  if (new_dev == DEV::FISHHASH_GPU &&
      (new_nonce_bytes != 8 ||
       !mom::job_boundary::valid_fishhash_layout(new_input_len, new_nonce_offset)))
    throw std::string(
      "fishhash requires a 40-byte header with nonce at offset 32 or a 180-byte header with "
      "nonce at offset 172");
  // KarlsenHashV2 uses an 8-byte nonce at offset 72 of its 80-byte Kaspa blob.
  if (new_dev == DEV::KARLSENHASHV2_GPU &&
      (new_nonce_bytes != 8 ||
       !mom::job_boundary::valid_karlsenhashv2_layout(new_input_len, new_nonce_offset)))
    throw std::string("karlsenhashv2 requires an 80-byte header with nonce at offset 72");
  // Test-only CPU/CN jobs hash arbitrary byte strings, and C29 solves the supplied header; neither
  // touches an embedded nonce. C30 and PearlHash keep counters outside the input blob. Every other
  // mining or GPU-dispatch path reads the embedded nonce and requires it inside the validated input.
  const bool external_nonce = new_dev == DEV::C30_GPU || new_dev == DEV::PEARLHASH_GPU;
  const bool test_skips_nonce = new_dev == DEV::CPU || new_dev == DEV::RX_CPU ||
    new_dev == DEV::GPU || new_dev == DEV::C29_GPU;
  const bool validates_embedded_nonce = !external_nonce &&
    (is_set_nonce || !test_skips_nonce || v.contains("nonceoffset"));
  if (validates_embedded_nonce &&
      (new_nonce_offset > new_input_len || new_nonce_bytes > new_input_len - new_nonce_offset))
    throw std::string("Nonce falls outside input blob");
  if (nonce_unit == 256 && is_set_nonce) {
    const uint32_t blob_nonce = mom::job_boundary::load_nonce<uint32_t>(
      new_input + new_nonce_offset, true);
    new_nonce = (new_nonce & ~uint64_t{0xff}) | (blob_nonce & 0xff);
  }
  if (new_dev == DEV::RX_CPU && is_set_nonce) {
    uint32_t first_initial_nonce = 0;
    uint32_t last_initial_nonce = 0;
    if (!mom::job_boundary::nonce_with_offset<uint32_t>(
          static_cast<uint32_t>(new_nonce),
          static_cast<uint64_t>(new_thread_id) * new_batch, nonce_unit, 0, first_initial_nonce) ||
        !mom::job_boundary::nonce_with_offset<uint32_t>(
          first_initial_nonce, new_batch - 1, nonce_unit, 0, last_initial_nonce))
      throw std::string("Initial RandomX worker nonce overflows");
  }

  const unsigned new_mem_size = algo2mem.at(new_algo_str);
  const bool is_algo_changed = m_algo_str != new_algo_str;
  const uint64_t new_nonce_prefix = new_nonce & new_nicehash_mask;
  const auto clear_mining_state = [&]() {
    // Bench/test nonces must never be reported as resumable work for the previous pool job.
    m_target = 0;
    std::memset(m_target_bin, 0, HASH_LEN);
    m_pool_id.clear();
    m_worker_id.clear();
    m_job_id.clear();
    m_job_token.clear();
    m_header_hash.clear();
  };
  const bool same_job_shape =
    is_no_same_input && m_job_mode == mode &&
    m_batch == new_batch && m_mem_size == new_mem_size &&
    m_seed_hex == new_seed_hex && m_algo_str == new_algo_str &&
    m_dev == new_dev && m_dev_str == new_dev_name &&
    m_height == new_height && m_nonce_bytes == new_nonce_bytes &&
    m_nonce_offset == new_nonce_offset && m_c29_proof_size == new_c29_proof_size &&
    m_input_len == new_input_len && m_nicehash_mask == new_nicehash_mask &&
    m_thread_id == new_thread_id && m_thread_num == new_thread_num &&
    m_pearlhash_seed_start == new_pearlhash_seed_start &&
    m_pearlhash_seed_stride == new_pearlhash_seed_stride &&
    m_backend == new_backend && m_pearlhash_n == new_pearlhash_n &&
    m_pearlhash_k == new_pearlhash_k && m_pearlhash_rank == new_pearlhash_rank &&
    m_pearlhash_cert_version == new_pearlhash_cert_version;
  const bool same_compute_input = same_job_shape &&
    (new_dev == DEV::RX_CPU || m_has_fn) && new_input_hex == m_input_hex &&
    m_nonce_prefix == new_nonce_prefix;
  const bool same_hashrate_domain =
    same_job_shape && mode != JobMode::test && m_has_fn;
  if (m_dev == DEV::RX_CPU) {
    stop_rx_threads();
  }
  if (same_compute_input && new_dev != DEV::RX_CPU) {
    // Pools can retarget c29 by sending the same work with a new target/job id.
    // Keep the existing nonce/search state and only refresh submit metadata.
    m_job_mode = mode;
    if (mode != JobMode::mine)
      clear_mining_state();
    fn_extra_setup();
    return;
  }

  // A failed replacement must leave the loop idle instead of dispatching the previous function
  // against released or partially allocated state. A changed header still invalidates old work,
  // but it must not erase a continuous rate window for the same compute configuration.
  clear_fn(!same_hashrate_domain);
  next_job_ref();
  if (m_batch != new_batch || m_mem_size != new_mem_size || m_seed_hex != new_seed_hex ||
      is_algo_changed || m_dev != new_dev) {
    if (m_dev == DEV::RX_CPU && new_dev == DEV::RX_CPU && m_seed_hex != new_seed_hex)
      destroy_rx_vms();
    // free previous memory
    const bool is_device_owned_memory_change =
      is_small_blob_gpu_dev(m_dev) || is_small_blob_gpu_dev(new_dev) ||
      m_dev == DEV::GPU || new_dev == DEV::GPU;
    free_memory(
      m_batch != new_batch || is_device_owned_memory_change,
      m_mem_size != new_mem_size,
      (m_dev != DEV::RX_CPU && new_dev == DEV::RX_CPU) || is_device_owned_memory_change,
      m_dev == DEV::RX_CPU && (new_dev != DEV::RX_CPU || is_algo_changed)
    );

    try {
      if (!is_small_blob_gpu_dev(new_dev) && new_dev != DEV::GPU && m_lpads == nullptr)
        m_lpads = alloc_huge_mem(allocation_size(new_batch, new_mem_size));

      if (new_dev == DEV::RX_CPU) {
        randomx_apply_config(*new_rx_config);
        // setup rx cache and dataset
        if (m_rx_cache_mem == nullptr)
          m_rx_cache_mem = alloc_huge_mem(RANDOMX_CACHE_MAX_SIZE);
        if (m_rx_dataset_mem == nullptr)
          m_rx_dataset_mem = alloc_huge_mem(RANDOMX_DATASET_MAX_SIZE);
        if (m_rx_cache == nullptr) {
          const bool try_jit_cache = new_algo_str != "panthera" && m_is_rx_jit;
          m_rx_cache = randomx_create_cache(
            try_jit_cache ? RANDOMX_FLAG_JIT : RANDOMX_FLAG_DEFAULT,
            m_rx_cache_mem->raw());
          if (m_rx_cache == nullptr && try_jit_cache) {
            m_is_rx_jit = false;
            m_rx_cache = randomx_create_cache(RANDOMX_FLAG_DEFAULT, m_rx_cache_mem->raw());
          }
          if (m_rx_cache == nullptr) throw std::string("Can't create RandomX cache");
        }
        if (m_rx_dataset == nullptr) {
          m_rx_dataset = randomx_create_dataset(m_rx_dataset_mem->raw());
          if (m_rx_dataset == nullptr) throw std::string("Can't create RandomX dataset");
        }
        // recompute cache, dataset for new seed
        if (m_seed_hex != new_seed_hex || is_algo_changed) {
          randomx_init_cache(m_rx_cache, new_seed, new_seed_len);
          // init dataset in parallel threads
          const unsigned long rx_dataset_item_count = randomx_dataset_item_count();
          const unsigned thread_count = std::thread::hardware_concurrency();
          if (new_algo_str == "panthera") {
            randomx_init_dataset(m_rx_dataset, m_rx_cache, 0, rx_dataset_item_count);
          } else if (thread_count > 1) {
            std::vector<std::thread> threads;
            threads.reserve(thread_count);
            try {
              for (unsigned i = 0; i < thread_count; ++i) {
                const auto a = static_cast<unsigned long>(
                  (static_cast<uint64_t>(rx_dataset_item_count) * i) / thread_count);
                const auto b = static_cast<unsigned long>(
                  (static_cast<uint64_t>(rx_dataset_item_count) * (i + 1)) / thread_count);
                threads.emplace_back(init_rx_dataset_thread, m_rx_dataset, m_rx_cache, a, b - a);
              }
            } catch (...) {
              join_threads(threads);
              throw;
            }
            join_threads(threads);
          } else {
            init_rx_dataset_thread(m_rx_dataset, m_rx_cache, 0, rx_dataset_item_count);
          }
        }

        // recreate vms
        if (m_vms.empty()) {
          std::vector<std::unique_ptr<randomx_vm, RandomXVmDeleter>> new_vms;
          new_vms.reserve(new_batch);
          for (unsigned i = 0; i != new_batch; ++i) {
            std::unique_ptr<randomx_vm, RandomXVmDeleter> vm(randomx_create_vm(
              get_rx_vm_flags(new_algo_str, m_is_rx_jit, m_rx_dataset, m_rx_dataset_mem), nullptr, m_rx_dataset,
              m_lpads->scratchpad() + i * new_mem_size, 0
            ));
            if (!vm) {
              throw std::string("Can't create RandomX VM");
            }
            new_vms.emplace_back(std::move(vm));
          }
          m_vms = std::move(new_vms);
        }
      } else if (new_dev == DEV::GPU) {
        if (m_input == nullptr)
          m_input = static_cast<uint8_t*>(alloc_mem(allocation_size(new_batch, MAX_BLOB_LEN)));
        if (m_output == nullptr)
          m_output = static_cast<uint8_t*>(alloc_mem(allocation_size(new_batch, HASH_LEN)));
      } else if (is_small_blob_gpu_dev(new_dev)) {
        if (m_input == nullptr) {
          m_input = static_cast<uint8_t*>(alloc_mem(MAX_BLOB_LEN));
        }
        if (m_output == nullptr) {
          m_output = static_cast<uint8_t*>(alloc_mem(HASH_LEN));
        }
        // Equihash adapters return compact proofs out-of-band; test paths dump validation rows here.
        if (m_spads == nullptr) {
          m_spads = alloc_mem(EQUIHASH_SOL_BUFFER_LEN);
        }
      } else { // setup cn/c29 stuff
        if (m_input == nullptr)
          m_input = static_cast<uint8_t*>(alloc_mem(allocation_size(new_batch, MAX_BLOB_LEN)));
        if (m_output == nullptr)
          m_output = static_cast<uint8_t*>(alloc_mem(allocation_size(new_batch, HASH_LEN)));
        if (m_spads == nullptr) {
          m_spads = alloc_mem(allocation_size(new_batch, SPAD_LEN));
        }
        if (m_ctx == nullptr) {
          m_ctx = new cryptonight_ctx*[new_batch]{};
          xmrig::CnCtx::create(m_ctx, m_lpads->scratchpad(), new_mem_size, new_batch);
        }
      }
      m_batch    = new_batch;
      m_mem_size = new_mem_size;
      m_seed_hex = new_seed_hex;
      m_algo_str = new_algo_str;
    } catch (...) {
      // Use the new batch count for cleanup and force allocation and seed initialization on retry.
      m_batch = new_batch;
      free_memory();
      m_algo_str.clear();
      throw;
    }
  }

  m_input_hex      = new_input_hex;
  m_dev            = new_dev;
  m_dev_str        = new_dev_name;
  m_height         = new_height;
  m_nonce_bytes    = new_nonce_bytes;
  m_nonce_offset   = new_nonce_offset;
  m_c29_proof_size = new_c29_proof_size;
  m_input_len      = new_input_len;
  m_nicehash_mask  = new_nicehash_mask;
  m_nonce_prefix   = new_nonce_prefix;
  m_thread_id      = new_thread_id;
  m_thread_num     = new_thread_num;
  m_pearlhash_seed_start = new_pearlhash_seed_start;
  m_pearlhash_seed_stride = new_pearlhash_seed_stride;
  m_backend        = new_backend;
  m_pearlhash_n    = new_pearlhash_n;
  m_pearlhash_k    = new_pearlhash_k;
  m_pearlhash_rank = new_pearlhash_rank;
  m_pearlhash_cert_version = new_pearlhash_cert_version;
  m_job_mode       = mode;
  std::memcpy(m_seed, new_seed, HASH_LEN);
  if (mode != JobMode::mine)
    clear_mining_state();
  fn_extra_setup();
  m_nonce_exhausted = false;

  if (new_dev == DEV::RX_CPU) {
    m_nonce32 = 0;
    m_nonce64 = 0;
    std::array<uint8_t, MAX_BLOB_LEN> job_input{};
    std::memcpy(job_input.data(), new_input, m_input_len);
    if (!xmrig::Cpu::info()->hasAES()) {
      SelectSoftAESImpl(m_batch);
    }
    m_rx_threads.reserve(m_batch);

    const unsigned job_ref = m_job_ref.load(std::memory_order_relaxed);
    const bool is_rx_v2 = new_algo_str == "rx/2";
    const xmrig::Algorithm rx_algo(cpu_name2algo.at(new_algo_str));
    const unsigned input_len = m_input_len;
    const unsigned nonce_offset = m_nonce_offset;
    const uint32_t nicehash_mask = static_cast<uint32_t>(m_nicehash_mask);
    const uint32_t nonce_step = static_cast<uint32_t>(
      static_cast<uint64_t>(new_thread_num) * m_batch * nonce_unit);
    const uint64_t target = m_target;
    const std::string pool_id = m_pool_id;
    const std::string job_id = m_job_id;
    const std::string job_token = m_job_token;
    const bool is_mining = mode == JobMode::mine;
    const bool is_test = mode == JobMode::test;
    try {
      for (unsigned batch_id = 0; batch_id != m_batch; ++batch_id) {
        randomx_vm* const vm = m_vms[batch_id].get();
        const auto initial_nonce = static_cast<uint32_t>(
          new_nonce + (static_cast<uint64_t>(new_thread_id) * m_batch + batch_id) * nonce_unit);
        m_rx_threads.emplace_back(
          [this, job_input, job_ref, is_rx_v2, rx_algo, input_len, nonce_offset,
           nicehash_mask, nonce_step, target, pool_id, job_id, job_token, is_mining, is_test, vm,
           initial_nonce]() {
            try {
              alignas(16) uint8_t input[MAX_BLOB_LEN];
              alignas(16) uint8_t output[HASH_LEN];
              alignas(16) uint8_t raw_hash[HASH_LEN];

              if (is_test) {
                randomx_calculate_hash(vm, job_input.data(), input_len, output, rx_algo);
                if (is_rx_v2) {
                  std::memcpy(raw_hash, output, HASH_LEN);
                  randomx_calculate_commitment(
                    job_input.data(), input_len, raw_hash, output);
                }
                char hash[HASH_LEN * 2 + 1];
                send_msg("test", "result", hash_bin2hex(output, hash));
                return;
              }

              alignas(16) uint8_t prev_input[MAX_BLOB_LEN];
              alignas(16) uint64_t temp_hash[8];
              std::memcpy(input, job_input.data(), input_len);
              uint32_t nonce = initial_nonce;
              if (nicehash_mask) {
                nonce |= mom::job_boundary::load_nonce<uint32_t>(
                  job_input.data() + nonce_offset, true) & nicehash_mask;
              }
              mom::job_boundary::store_nonce(input + nonce_offset, nonce, true);
              if (is_rx_v2) {
                std::memcpy(prev_input, input, input_len);
              }
              randomx_calculate_hash_first(vm, temp_hash, input, input_len, rx_algo);

              unsigned pending_hashes = 0;
              bool nonce_exhausted = false;
              while (job_ref == m_job_ref.load(std::memory_order_relaxed)) {
                const uint32_t next_nonce = nonce + nonce_step;
                nonce_exhausted = mom::job_boundary::nonce_overflowed(
                  nonce, next_nonce, nicehash_mask);
                if (!nonce_exhausted) {
                  mom::job_boundary::store_nonce(input + nonce_offset, next_nonce, true);
                }
                randomx_calculate_hash_next(vm, temp_hash, input, input_len, output, rx_algo);

                const uint8_t* commitment = nullptr;
                if (is_rx_v2) {
                  std::memcpy(raw_hash, output, HASH_LEN);
                  randomx_calculate_commitment(prev_input, input_len, raw_hash, output);
                  std::memcpy(prev_input, input, input_len);
                  commitment = raw_hash;
                }

                if (++pending_hashes == HASHRATE_COUNTER_INTERVAL) {
                  m_hash_count.fetch_add(pending_hashes, std::memory_order_relaxed);
                  pending_hashes = 0;
                }
                if (target && result_word(output, 0) < target) {
                  send_result(nonce, 4, output, nullptr, 32, commitment);
                }
                if (nonce_exhausted) {
                  send_error("Nonce overflow");
                  break;
                }
                nonce = next_nonce;
              }
              if (pending_hashes) {
                m_hash_count.fetch_add(pending_hashes, std::memory_order_relaxed);
              }
              if (is_mining && !nonce_exhausted) {
                send_last_nonce(nonce, 4, pool_id, job_id, job_token);
              }
            } catch (const std::string& err) {
              send_error(std::string("Compute function thread exception: ") + err);
            } catch (...) {
              send_error("Compute function thread exception");
            }
          }
        );
      }
    } catch (...) {
      stop_rx_threads();
      throw;
    }
  } else {
    if (external_nonce) {
      // C30 and PearlHash keep their 64-bit counters outside the input blob.
      memcpy(m_input, new_input, m_input_len);
      m_nonce_step = new_thread_num;
      if (new_dev == DEV::PEARLHASH_GPU) {
        m_nonce64 = is_set_nonce ? new_pearlhash_seed_start : 0;
      } else {
        m_nonce64 = is_set_nonce ? new_nonce + new_thread_id : 0;
        if (m_target && is_set_nonce && !mom::job_boundary::nonce_with_offset<uint64_t>(
            new_nonce, new_thread_id, 1, new_nicehash_mask, m_nonce64)) {
          throw std::string("Initial worker nonce overflows");
        }
      }
      m_nonce32 = 0;
      set_fn(new_fn);
      return;
    }
    if (new_dev == DEV::VERTHASH_GPU) {
      memcpy(m_input, new_input, m_input_len);
      m_nonce_step = new_thread_num * m_batch;
      m_nonce32 = 0;
      if (is_set_nonce) {
        const uint32_t mask = static_cast<uint32_t>(new_nicehash_mask);
        uint32_t current_nonce = static_cast<uint32_t>(new_nonce) + new_thread_id * m_batch;
        uint32_t last_nonce = current_nonce + m_batch - 1;
        if (m_target && (!mom::job_boundary::nonce_with_offset<uint32_t>(
              static_cast<uint32_t>(new_nonce), new_thread_id, m_batch, mask,
              current_nonce) ||
            !mom::job_boundary::nonce_with_offset<uint32_t>(
              current_nonce, m_batch - 1, 1, mask, last_nonce))) {
          throw std::string("Initial worker nonce range overflows");
        }
        std::memcpy(m_input + m_nonce_offset, &current_nonce, sizeof(current_nonce));

        m_nonce32 = next_nonce_after_batch(
          current_nonce, m_batch, 1, m_nonce_step, mask);
      }
      m_nonce64 = 0;
      set_fn(new_fn);
      return;
    }
    // Single-blob GPU pow with an embedded nonce: kawpow/etchash/autolykos2 (nonce at offset 32) and
    // KarlsenHashV2 (8-byte nonce at offset 72). One header blob (not a per-batch buffer); the kernel
    // searches start_nonce..start_nonce+batch. m_nonce_offset (not a hardcoded 32) places the nonce.
    if (is_nonce_at_32_gpu_dev(new_dev) || new_dev == DEV::KARLSENHASHV2_GPU ||
        new_dev == DEV::MISC_GPU || is_equihash_gpu_dev(new_dev)) {
      memcpy(m_input, new_input, m_input_len);
      m_nonce_step = new_thread_num * m_batch;
      // Mine mode embeds the starting nonce and points m_nonce64 at the next one. Live Iron Fish
      // (fishhash, 180-byte header) reads the 8-byte nonce big-endian, so embed it byte-swapped there;
      // XelisHashV3 and NexaPow also use BE; the remaining embedded-nonce layouts use LE.
      m_nonce64 = 0;
      if (is_set_nonce) {
        uint64_t current_nonce = new_nonce + static_cast<uint64_t>(new_thread_id) * m_batch;
        uint64_t last_nonce = current_nonce + m_batch - 1;
        if (m_target && (!mom::job_boundary::nonce_with_offset<uint64_t>(
              new_nonce, new_thread_id, m_batch, new_nicehash_mask, current_nonce) ||
            !mom::job_boundary::nonce_with_offset<uint64_t>(
              current_nonce, m_batch - 1, 1, new_nicehash_mask, last_nonce))) {
          throw std::string("Initial worker nonce range overflows");
        }
        mom::job_boundary::store_nonce(
          nonce_address(), current_nonce,
          mom::job_boundary::nonce_big_endian(new_algo_str, m_input_len));

        m_nonce64 = next_nonce_after_batch(
          current_nonce, m_batch, 1, m_nonce_step, new_nicehash_mask);
      }
      m_nonce32 = 0;
      set_fn(new_fn);
      return;
    }

    m_nonce_step = static_cast<unsigned>(
      static_cast<uint64_t>(new_thread_num) * nonce_unit);
    for (unsigned i = 0; i != m_batch; ++i)
      memcpy(m_input + m_input_len*i, new_input, m_input_len);
    if (m_nonce_bytes == 4) {
      m_nonce32 = 0;
      m_nonce64 = 0;
      if (is_set_nonce) {
        const uint32_t mask = static_cast<uint32_t>(m_nicehash_mask);
        m_nonce32 = static_cast<uint32_t>(new_nonce) +
          static_cast<uint32_t>(new_thread_id * nonce_unit);
        if (m_target && !mom::job_boundary::nonce_with_offset<uint32_t>(
            static_cast<uint32_t>(new_nonce), new_thread_id, nonce_unit, 0, m_nonce32)) {
          throw std::string("Initial worker nonce overflows");
        }
        if (m_nicehash_mask) {
          m_nonce32 |= mom::job_boundary::load_nonce<uint32_t>(
            nonce_address(new_input), true) & mask;
        }
        const uint32_t first_nonce = m_nonce32;
        uint32_t last_nonce = first_nonce + (m_batch - 1) * m_nonce_step;
        if (m_target && !mom::job_boundary::nonce_with_offset<uint32_t>(
            first_nonce, m_batch - 1, m_nonce_step, mask, last_nonce)) {
          throw std::string("Initial worker nonce range overflows");
        }
        for (unsigned i = 0; i != m_batch; ++i)
          mom::job_boundary::store_nonce(
            nonce_address(i), first_nonce + i * m_nonce_step, true);
        m_nonce32 = next_nonce_after_batch(
          first_nonce, m_batch, m_nonce_step,
          static_cast<uint64_t>(m_batch) * m_nonce_step, mask);
      }
    } else {
      m_nonce32 = 0;
      m_nonce64 = 0;
      if (is_set_nonce) {
        m_nonce64 = new_nonce + new_thread_id;
        if (m_target && !mom::job_boundary::nonce_with_offset<uint64_t>(
            new_nonce, new_thread_id, 1, 0, m_nonce64)) {
          throw std::string("Initial worker nonce overflows");
        }
        if (m_nicehash_mask) {
          m_nonce64 |= mom::job_boundary::load_nonce<uint64_t>(
            nonce_address(new_input), true) & m_nicehash_mask;
        }
        const uint64_t first_nonce = m_nonce64;
        uint64_t last_nonce = first_nonce + (m_batch - 1) * m_nonce_step;
        if (m_target && !mom::job_boundary::nonce_with_offset<uint64_t>(
            first_nonce, m_batch - 1, m_nonce_step, m_nicehash_mask, last_nonce)) {
          throw std::string("Initial worker nonce range overflows");
        }
        for (unsigned i = 0; i != m_batch; ++i)
          mom::job_boundary::store_nonce(
            nonce_address(i), first_nonce + i * m_nonce_step, true);
        m_nonce64 = next_nonce_after_batch(
          first_nonce, m_batch, m_nonce_step,
          static_cast<uint64_t>(m_batch) * m_nonce_step, m_nicehash_mask);
      }
    }
    set_fn(new_fn);
  }
}
