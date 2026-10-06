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

static inline unsigned char hf_hex2bin(const char c, bool& err) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 0xA;
  if (c >= 'A' && c <= 'F') return c - 'A' + 0xA;
  err = true;
  return 0;
}

bool Core::hex2bin(const char* in, unsigned int len, unsigned char* out) {
  bool error = false;
  for (unsigned int i = 0; i < len; ++i, ++out, in += 2) {
    *out = (hf_hex2bin(*in, error) << 4) | hf_hex2bin(*(in + 1), error);
    if (error) return false;
  }
  return true;
}

static inline char hf_bin2hex(const unsigned n) {
  return n < 10 ? '0' + n : 'a' + (n - 10);
}

char* Core::hash_bin2hex(const uint8_t* const output, char* hash, const unsigned batch) const {
  char* hash0 = hash;
  for (unsigned i = 0, offset = batch * HASH_LEN; i != HASH_LEN; ++i, ++offset) {
    *hash++ = hf_bin2hex(output[offset] >> 4);
    *hash++ = hf_bin2hex(output[offset] & 0xF);
  }
  *hash = 0;
  return hash0;
}

char* Core::hash_bin2hex(char* const hash, const unsigned batch) const {
  return hash_bin2hex(m_output, hash, batch);
}

void Core::send_msg(const std::string key, const MessageValues& values) {
  debug_async_worker(("Core::send_msg " + key).c_str());
  sendToNode(Message(key, values));
  debug_async_worker(("Core::send_msg done " + key).c_str());
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

void Core::send_result(
  const uint64_t nonce, const unsigned noncebytes, const uint8_t* const output,
  const uint32_t* const edges, const unsigned c29_proof_size,
  const uint8_t* const commitment, const uint8_t* const mix_hash,
  const uint8_t* const solution, const unsigned solution_len
) {
  MessageValues values;
  values["nonce"] = nonce_to_hex(nonce, noncebytes);

  if (output) {
    char hash_hex[HASH_LEN * 2 + 1];
    values["hash"] = hash_bin2hex(output, hash_hex);
  }

  // ZIP-301 shares carry a compactSize-prefixed Equihash proof instead of a 32-byte hash.
  if (solution && solution_len) {
    std::string solution_hex;
    solution_hex.reserve(solution_len * 2);
    for (unsigned i = 0; i < solution_len; ++i) {
      solution_hex += hf_bin2hex(solution[i] >> 4);
      solution_hex += hf_bin2hex(solution[i] & 0xF);
    }
    values["solution"] = solution_hex;
  }

  if (commitment) {
    char commitment_hex[HASH_LEN * 2 + 1];
    values["commitment"] = hash_bin2hex(commitment, commitment_hex);
  }

  if (mix_hash) {
    char mix_hash_hex[HASH_LEN * 2 + 1];
    values["mix_hash"] = hash_bin2hex(mix_hash, mix_hash_hex);
  }

  if (edges) {
    std::string edges_hex;
    edges_hex.reserve(c29_proof_size * 8); // 8 hex chars per edge
    char buf[9]; // enough for "%08x" + null
    for (unsigned i = 0; i < c29_proof_size; ++i) {
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

void Core::send_equihash_results(
  const uint64_t nonce, const unsigned solution_size, const bool compact_size_prefix
) {
  const uint8_t* const solutions = static_cast<const uint8_t*>(m_spads);
  const unsigned count = solutions[0];
  const unsigned capacity = (EQUIHASH_SOL_BUFFER_LEN - 1) / solution_size;
  if (count > capacity) {
    send_error("Invalid " + m_algo_str + " solution count");
    return;
  }

  const unsigned prefix_size = compact_size_prefix ? (solution_size < 253 ? 1 : 3) : 0;
  for (unsigned i = 0; i < count; ++i) {
    const uint8_t* const solution = solutions + 1 + static_cast<size_t>(i) * solution_size;
    if (!compact_size_prefix) {
      send_result(nonce, m_nonce_bytes, nullptr, nullptr, 32, nullptr, nullptr,
                  solution, solution_size);
      continue;
    }

    uint8_t submit[3 + 400];
    if (prefix_size == 1) {
      submit[0] = static_cast<uint8_t>(solution_size);
    } else {
      submit[0] = 253;
      submit[1] = static_cast<uint8_t>(solution_size);
      submit[2] = static_cast<uint8_t>(solution_size >> 8);
    }
    std::memcpy(submit + prefix_size, solution, solution_size);
    send_result(nonce, m_nonce_bytes, nullptr, nullptr, 32, nullptr, nullptr,
                submit, solution_size + prefix_size);
  }
}

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

static uint64_t parse_padded_le_hex(const std::string& hex, const unsigned bytes) {
  bool error = false;
  uint64_t value = 0;
  for (unsigned i = 0; i < bytes; ++i) {
    const unsigned pos = i * 2;
    const unsigned char byte =
      (hf_hex2bin(pos < hex.size() ? hex[pos] : '0', error) << 4) |
       hf_hex2bin(pos + 1 < hex.size() ? hex[pos + 1] : '0', error);
    value |= static_cast<uint64_t>(byte) << (i * 8);
  }
  if (error) throw std::string("Bad target hex");
  return value;
}

static uint64_t parse_high_target_word(const std::string& target) {
  if (target.empty() || target.size() > HASH_LEN * 2)
    throw std::string("Bad target hex");
  bool error = false;
  for (const char c : target) {
    hf_hex2bin(c, error);
  }
  if (error) throw std::string("Bad target hex");
  std::string target64 = target.substr(0, sizeof(uint64_t) * 2);
  target64.resize(sizeof(uint64_t) * 2, '0');
  char* end = nullptr;
  const uint64_t value = strtoull(target64.c_str(), &end, 16);
  if (end == target64.c_str() || *end || value == 0) throw std::string("Bad target hex");
  return value;
}

static uint64_t parse_target_hex(const std::string& target, const bool is_kawpow_target) {
  if (is_kawpow_target || target.size() > sizeof(uint64_t) * 2) return parse_high_target_word(target);

  const uint64_t value = parse_padded_le_hex(target, target.size() <= sizeof(uint32_t) * 2 ?
                                                    sizeof(uint32_t) : sizeof(uint64_t));
  if (value == 0) throw std::string("Bad target hex");
  if (target.size() > sizeof(uint32_t) * 2) return value;
  return 0xFFFFFFFFFFFFFFFFULL / (0xFFFFFFFFULL / value);
}

static std::string strip_hex_prefix(const std::string& value) {
  return value.starts_with("0x") || value.starts_with("0X") ? value.substr(2) : value;
}

static void parse_big_target_hex(const std::string& target, uint8_t* const out) {
  std::string hex = strip_hex_prefix(target);
  if (hex.empty() || hex.size() > HASH_LEN * 2 || (hex.size() & 1))
    throw std::string("Bad target hex");
  hex.insert(0, HASH_LEN * 2 - hex.size(), '0');
  bool error = false;
  uint8_t any = 0;
  for (unsigned i = 0; i < HASH_LEN; ++i) {
    out[i] = (hf_hex2bin(hex[i * 2], error) << 4) | hf_hex2bin(hex[i * 2 + 1], error);
    if (error) throw std::string("Bad target hex");
    any |= out[i];
  }
  if (!any) throw std::string("Bad target hex"); // reject all-zero target
}

static void free_mem(void* const mem) { _mm_free(mem); }

void Core::stop_rx_threads() noexcept {
  if (m_rx_threads.empty()) {
    return;
  }
  next_job_ref();
  join_threads(m_rx_threads);
  m_rx_threads.clear();
}

void Core::destroy_rx_vms() noexcept {
  m_vms.clear();
}

void Core::free_memory(
  const bool is_batch_changed,
  const bool is_mem_size_changed,
  const bool is_free_cn,
  const bool is_free_rx
) {
  if (is_batch_changed || is_mem_size_changed || is_free_rx) {
    stop_rx_threads();
    destroy_rx_vms();
  }
  if (is_batch_changed || is_mem_size_changed) {
    if (m_lpads) {
      delete m_lpads;
      m_lpads = nullptr;
    }
  }
  if (is_batch_changed) {
    if (m_input) {
      free_mem(m_input);
      m_input = nullptr;
    }
    if (m_output) {
      free_mem(m_output);
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
      free_mem(m_spads);
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
