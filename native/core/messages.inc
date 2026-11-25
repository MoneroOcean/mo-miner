void (*rx_blake2b_compress)(blake2b_state* S, const uint8_t * block) = rx_blake2b_compress_integer;
int (*rx_blake2b)(void* out, size_t outlen, const void* in, size_t inlen) = rx_blake2b_default;

struct MsrValue {
  uint64_t value, mask;
  MsrValue() {}
  MsrValue(const uint64_t value, const uint64_t mask = ~0) : value(value), mask(mask) {}
};
typedef std::map<uint32_t, MsrValue> MsrItems;
static const MsrItems msr_mod_zen4_zen5 = {
  { 0xC0011020, MsrValue(0x0004400000000000ULL) },
  { 0xC0011021, MsrValue(0x0004000000000040ULL, ~0x20ULL) },
  { 0xC0011022, MsrValue(0x8680000401570000ULL) },
  { 0xC001102b, MsrValue(0x2040cc10ULL) }
};
static const std::map<xmrig::ICpuInfo::MsrMod, MsrItems> msr_mods = {
  { xmrig::ICpuInfo::MSR_MOD_RYZEN_17H, MsrItems {
    { 0xC0011020, MsrValue(0ULL) },
    { 0xC0011021, MsrValue(0x40ULL, ~0x20ULL) },
    { 0xC0011022, MsrValue(0x1510000ULL) },
    { 0xC001102b, MsrValue(0x2000cc16ULL) }
  }}, { xmrig::ICpuInfo::MSR_MOD_RYZEN_19H, MsrItems {
    { 0xC0011020, MsrValue(0x0004480000000000ULL) },
    { 0xC0011021, MsrValue(0x001c000200000040ULL, ~0x20ULL) },
    { 0xC0011022, MsrValue(0xc000000401570000ULL) },
    { 0xC001102b, MsrValue(0x2000cc10ULL) }
  }},
  { xmrig::ICpuInfo::MSR_MOD_RYZEN_19H_ZEN4, msr_mod_zen4_zen5 },
  { xmrig::ICpuInfo::MSR_MOD_RYZEN_1AH_ZEN5, msr_mod_zen4_zen5 },
  { xmrig::ICpuInfo::MSR_MOD_INTEL, MsrItems {
    { 0x1a4, MsrValue(0xf) }
  }}
};

// Lower-case hex of `size` bytes, as the controller expects for hashes, proofs and test rows.
std::string Core::bin2hex(const uint8_t* const data, const size_t size) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string hex(size * 2, '0');
  for (size_t i = 0; i < size; ++i) {
    hex[i * 2] = digits[data[i] >> 4];
    hex[i * 2 + 1] = digits[data[i] & 0xF];
  }
  return hex;
}

void Core::send_msg(const std::string& key, const MessageValues& values) {
  debug_async_worker("Core::send_msg ", key);
  sendToNode(Message(key, values));
  debug_async_worker("Core::send_msg done ", key);
}

void Core::send_msg(const std::string& topic, const std::string& key, const std::string& value) {
  MessageValues values;
  if (!key.empty()) {
    values[key] = value;
  }
  send_msg(topic, values);
}

void Core::send_error(const std::string& str) {
  send_msg("error", "message", str);
}

// Format a nonce as fixed-width little-endian-display hex: 8 digits for 4-byte nonces, else 16.
static std::string nonce_to_hex(const uint64_t nonce, const unsigned noncebytes) {
  char hex[sizeof(uint64_t) * 2 + 1];
  if (noncebytes == 4) {
    snprintf(hex, sizeof(hex), "%08x", static_cast<uint32_t>(nonce));
  } else {
    snprintf(hex, sizeof(hex), "%016" PRIx64, nonce);
  }
  return hex;
}

// Send one share: nonce (plus hash, CompactSize-prefixed Equihash solution, RandomX commitment,
// mix hash or cycle edges as present) with the current job's submit ids.
void Core::send_result(
  const uint64_t nonce, const unsigned noncebytes, const uint8_t* const output,
  const uint32_t* const edges, const unsigned cuckaroo_proof_size,
  const uint8_t* const commitment, const uint8_t* const mix_hash,
  const uint8_t* const solution, const unsigned solution_len
) {
  MessageValues values;
  values["nonce"] = nonce_to_hex(nonce, noncebytes);

  if (output) {
    values["hash"] = bin2hex(output, HASH_LEN);
  }

  // ZIP-301 shares carry a compactSize-prefixed Equihash proof instead of a 32-byte hash.
  if (solution && solution_len) {
    values["solution"] = bin2hex(solution, solution_len);
  }

  if (commitment) {
    values["commitment"] = bin2hex(commitment, HASH_LEN);
  }

  if (mix_hash) {
    values["mix_hash"] = bin2hex(mix_hash, HASH_LEN);
  }

  if (edges) {
    std::string edges_hex;
    edges_hex.reserve(cuckaroo_proof_size * 8); // 8 hex chars per edge
    char buf[9]; // enough for "%08x" + null
    for (unsigned i = 0; i < cuckaroo_proof_size; ++i) {
      snprintf(buf, sizeof(buf), "%08x", edges[i]);
      edges_hex.append(buf);
    }
    values["edges"] = edges_hex;
  }

  values["pool_id"]   = m_pool_id;
  values["worker_id"] = m_worker_id;
  values["job_id"]    = m_job_id;
  if (!m_job_token.empty()) {
    values["job_token"] = m_job_token;
  }
  if (!m_header_hash.empty()) {
    values["header_hash"] = m_header_hash;
  }
  send_msg("result", values);
}

// Report every hit of a single-blob GPU dispatch (benchmarks report none). Each slot holds a
// distinct nonce of the batch, so no nonce is sent twice. Returns false after reporting an
// impossible hit count; the caller then stops the function instead of trusting its buffers.
bool Core::send_gpu_hits(
  const int hits, const uint64_t* const nonces, const unsigned noncebytes,
  const uint8_t* const mix_hash
) {
  const bool valid = mom::job_boundary::for_each_gpu_hit(hits, GPU_MAX_HITS, [&](unsigned i) {
    if (m_target)
      send_result(nonces[i], noncebytes, m_output + i * HASH_LEN, nullptr, 32, nullptr,
                  mix_hash ? mix_hash + i * HASH_LEN : nullptr);
  });
  if (!valid)
    send_error("Invalid " + m_algo_str + " GPU hit count " + std::to_string(hits));
  return valid;
}

// Send each proof an Equihash solver left in m_spads as [count:u8][count * solution_size bytes].
// ZIP-301 pools take each proof with its CompactSize length prefix (1 byte below 253, else 0xfd
// plus a little-endian uint16); Beam takes the bare solution. A count beyond the buffer is an
// invalid solver result and is reported instead.
void Core::send_equihash_results(
  const uint64_t nonce, const unsigned solution_size, const bool compact_size_prefix
) {
  const uint8_t* const solutions = static_cast<const uint8_t*>(m_spads);
  const unsigned count = solutions[0];
  uint8_t submit[3 + 400];
  if (!solution_size || solution_size > sizeof(submit) - 3) {
    send_error("Invalid " + m_algo_str + " solution size");
    return;
  }
  const unsigned capacity = (EQUIHASH_SOL_BUFFER_LEN - 1) / solution_size;
  if (count > capacity) {
    send_error("Invalid " + m_algo_str + " solution count");
    return;
  }

  const unsigned prefix_size = compact_size_prefix
      ? mom::job_boundary::write_compact_size16(static_cast<uint16_t>(solution_size), submit) : 0;
  for (unsigned i = 0; i < count; ++i) {
    const uint8_t* proof = solutions + 1 + static_cast<size_t>(i) * solution_size;
    if (compact_size_prefix) {
      std::memcpy(submit + prefix_size, proof, solution_size);
      proof = submit;
    }
    send_result(nonce, m_nonce_bytes, nullptr, nullptr, 32, nullptr, nullptr,
                proof, prefix_size + solution_size);
  }
}

// Report how far a pool job got so the controller can resume it later; bench/test jobs have no
// pool ids and report nothing.
void Core::send_last_nonce(
  const uint64_t nonce, const unsigned noncebytes, const std::string& pool_id,
  const std::string& job_id, const std::string& job_token
) {
  if (pool_id.empty() || job_token.empty()) {
    return;
  }
  MessageValues result;
  result["nonce"]   = nonce_to_hex(nonce, noncebytes);
  result["pool_id"] = pool_id;
  result["job_id"]  = job_id;
  result["job_token"] = job_token;
  send_msg("last_nonce", result);
}

void Core::stop_rx_threads() noexcept {
  if (m_rx_threads.empty()) {
    return;
  }
  next_job_ref();
  join_threads(m_rx_threads);
  m_rx_threads.clear();
}

// Release worker buffers selectively for a job switch (all of them by default): a batch change
// frees the per-batch input/output/scratch buffers, a memory-size change the large pages, and the
// CN/RX flags their hash contexts and RandomX cache/dataset. RandomX threads and VMs stop first
// because they reference all of these.
void Core::free_memory(
  const bool is_batch_changed,
  const bool is_mem_size_changed,
  const bool is_free_cn,
  const bool is_free_rx
) {
  if (is_batch_changed || is_mem_size_changed || is_free_rx) {
    stop_rx_threads();
    m_vms.clear();
  }
  if (is_batch_changed || is_mem_size_changed) {
    if (m_lpads) {
      delete m_lpads;
      m_lpads = nullptr;
    }
  }
  if (is_batch_changed) {
    if (m_input) {
      _mm_free(m_input);
      m_input = nullptr;
    }
    if (m_output) {
      _mm_free(m_output);
      m_output = nullptr;
    }
  }
  if (is_batch_changed || is_mem_size_changed || is_free_cn) {
    if (m_ctx) {
      xmrig::CnCtx::release(m_ctx, m_batch);
      delete [] m_ctx;
      m_ctx = nullptr;
    }
  }
  if (is_batch_changed || is_free_cn) {
    if (m_spads) {
      _mm_free(m_spads);
      m_spads = nullptr;
    }
  }
  if (is_free_rx) {
    if (m_rx_dataset) {
      randomx_release_dataset(m_rx_dataset);
      m_rx_dataset = nullptr;
    }
    if (m_rx_cache) {
      randomx_release_cache(m_rx_cache);
      m_rx_cache = nullptr;
    }
    if (m_rx_dataset_mem) {
      delete m_rx_dataset_mem;
      m_rx_dataset_mem = nullptr;
    }
    if (m_rx_cache_mem) {
      delete m_rx_cache_mem;
      m_rx_cache_mem = nullptr;
    }
  }
}

void Core::set_fn(const FN& fn) {
  m_fn         = fn;
  m_has_fn     = true;
}

void Core::clear_fn(const bool reset_hashrate) {
  m_fn         = {};
  m_has_fn     = false;
  if (reset_hashrate) {
    m_timestamp = 0;
    m_hash_count.store(0, std::memory_order_relaxed);
  }
}
