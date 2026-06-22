// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#include "core.h"
#include "../sycl/lib.h"   // pearl_proof()

#include "3rdparty/fmt/core.h"
#include "backend/cpu/Cpu.h"
#include "crypto/cn/CnCtx.h"
#include "crypto/randomx/blake2/blake2.h"
#include "crypto/randomx/blake2/avx2/blake2b.h"
#include "crypto/rx/RxFix.h"
#include "hw/msr/Msr.h"
#include "3rdparty/argon2.h"
#include "base/tools/bswap_64.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <inttypes.h>
#include <thread>
#include <cstring>

static const xmrig::ICpuInfo& cpu_info() { return *xmrig::Cpu::info(); }
#define ci cpu_info()
void (*rx_blake2b_compress)(blake2b_state* S, const uint8_t * block) = rx_blake2b_compress_integer;
int (*rx_blake2b)(void* out, size_t outlen, const void* in, size_t inlen) = rx_blake2b_default;

static void debug_startup(const char* message) {
  if (!std::getenv("MOM_DEBUG_STARTUP")) return;
  fprintf(stderr, "MOM_DEBUG_STARTUP %s\n", message);
  fflush(stderr);
}

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

std::vector<std::string> Core::tokenize(const std::string& str, const char delim) {
  std::vector<std::string> out;
  size_t start;
  size_t end = 0;
  while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
    end = str.find(delim, start);
    out.push_back(str.substr(start, end - start));
  }
  return out;
}

static inline char hf_bin2hex(const unsigned n) {
  return n < 10 ? '0' + n : 'a' + (n - 10);
}

char* Core::hash_bin2hex(const uint8_t* const output, char* hash, const unsigned batch) const {
  char* hash0 = hash;
  for (unsigned i = 0, offset = batch * HASH_LEN; i != HASH_LEN; ++ i, ++ offset) {
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
  static SimpleMutex mutex_message;
  SimpleLock lock(mutex_message);
  debug_startup(("Core::send_msg " + key).c_str());
  sendToNode(Message(key, values));
  debug_startup(("Core::send_msg done " + key).c_str());
}

void Core::send_msg(const std::string& topic, const std::string& key, const std::string& value) {
  MessageValues values;
  if (!key.empty()) values[key] = value;
  send_msg(topic, values);
}

void Core::send_error(const std::string& str) {
  send_msg("error", "message", str);
}

// Format a nonce as fixed-width little-endian-display hex: 8 digits for 4-byte nonces, else 16.
static std::string nonce_to_hex(const uint64_t nonce, const unsigned noncebytes) {
  char hex[sizeof(uint64_t) * 2 + 1];
  if (noncebytes == 4) snprintf(hex, sizeof(hex), "%08x", static_cast<uint32_t>(nonce));
  else                 snprintf(hex, sizeof(hex), "%016" PRIx64, nonce);
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

  char hash_hex[HASH_LEN * 2 + 1];
  values["hash"] = hash_bin2hex(output, hash_hex);

  // Out-of-band proof bytes for algos whose share isn't a 32-byte hash (equihash125_4: the Flux/ZIP-301
  // stratum submit carries the full compactSize-prefixed solution -- 0x34 || 52-byte compressed proof =
  // 53 bytes / 106 hex). Serialized like edges: caller passes the already-prefixed byte run.
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
  if (!m_header_hash.empty()) values["header_hash"] = m_header_hash;
  send_msg("result", values);
}

void Core::send_last_nonce(const uint64_t nonce, const unsigned noncebytes, const std::string& pool_id) {
  MessageValues result;
  result["nonce"]   = nonce_to_hex(nonce, noncebytes);
  result["pool_id"] = pool_id;
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
  std::string target64 = target.substr(0, sizeof(uint64_t) * 2);
  target64.resize(sizeof(uint64_t) * 2, '0');
  bool error = false;
  for (const char c : target64) hf_hex2bin(c, error);
  if (error) throw std::string("Bad target hex");
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

static bool nonce_overflowed(const uint64_t previous, const uint64_t next, const uint64_t mask) {
  return mask ? (previous & mask) != (next & mask) : previous > next;
}

static void free_mem(void* const mem) { _mm_free(mem); }

void Core::free_memory(
  const bool is_batch_changed,
  const bool is_mem_size_changed,
  const bool is_free_cn,
  const bool is_free_rx
) {
  // m_thread_pool need to be deleted first if anything rx related is deleted
  if (is_batch_changed || is_free_rx) {
    // ++ m_job_ref is to stop rx threads if any
    if (m_thread_pool) { ++ m_job_ref; delete m_thread_pool; m_thread_pool = nullptr; }
    if (m_vm) {
      for (unsigned i = 0; i != m_batch; ++ i) randomx_destroy_vm(m_vm[i]);
      delete [] m_vm; m_vm = nullptr;
    }
  }
  if (is_batch_changed || is_mem_size_changed) {
    if (m_lpads) { delete m_lpads; m_lpads = nullptr; }
  }
  if (is_batch_changed) {
    if (m_input)  { free_mem(m_input);  m_input  = nullptr; }
    if (m_output) { free_mem(m_output); m_output = nullptr; }
  }
  if (is_batch_changed || is_mem_size_changed || is_free_cn) {
    if (m_ctx) { xmrig::CnCtx::release(m_ctx, m_batch); delete [] m_ctx; m_ctx = nullptr; }
  }
  if (is_batch_changed || is_free_cn) {
    if (m_spads) { free_mem(m_spads); m_spads = nullptr; }
  }
  if (is_free_rx) {
    if (m_rx_dataset)     { randomx_release_dataset(m_rx_dataset); m_rx_dataset = nullptr; }
    if (m_rx_cache)       { randomx_release_cache(m_rx_cache); m_rx_cache = nullptr; }
    if (m_rx_dataset_mem) { delete m_rx_dataset_mem; m_rx_dataset_mem = nullptr; }
    if (m_rx_cache_mem)   { delete m_rx_cache_mem; m_rx_cache_mem = nullptr; }
  }
}

void Core::set_fn(cn_any_hash_fun fn) {
  m_fn.any     = fn;
  m_timestamp  = 0;
  m_hash_count = 0;
}

bool Core::process_message(const std::string& type, const MessageValues& v) {
  if (type == "job") {
    m_is_bench = false;
    if (!v.contains("target"))    throw std::string("Missing target job key");
    if (!v.contains("pool_id"))   throw std::string("Missing pool_id job key");
    if (!v.contains("worker_id")) throw std::string("Missing worker_id job key");
    if (!v.contains("job_id"))    throw std::string("Missing job_id job key");
    const std::string new_target_str = strip_hex_prefix(v.at("target"));

    const std::string algo = v.contains("algo") ? v.at("algo") : "";
    const bool is_kawpow_target = algo == "kawpow";
    // etchash/autolykos2/pearl/fishhash use a full 32-byte target instead of a single 64-bit word
    // (fishhash: live Iron Fish sends a 256-bit big-endian target -> m_target_bin -> meets_target_be_dev)
    const bool is_big_target = algo == "etchash" || algo == "autolykos2" || algo == "pearl" || algo == "fishhash" || algo == "equihash125_4" || algo == "beamhash3" ||
      // kHeavyHash family (Kaspa/Karlsen/Pyrin) compares the 32-byte hash against a full 256-bit BE
      // boundary via m_target_bin; without this the bin stays zero and live mining finds no shares.
      algo == "kheavyhash" || algo == "karlsenhashv2" || algo == "pyrinhashv2";
    const uint64_t new_target = is_big_target ? 1 : parse_target_hex(new_target_str, is_kawpow_target);
    uint8_t new_target_bin[HASH_LEN]{};
    if (is_big_target) parse_big_target_hex(new_target_str, new_target_bin);
    // The kHeavyHash family kernel (meets_target_le_dev) compares the little-endian-stored 32-byte hash
    // against a little-endian target array (index 31 = MSB), but the share target hex is big-endian.
    // Reverse the bytes so target[31] is the MSB, matching the hash's LE byte layout. (is_test passes a
    // zero target, which is symmetric, so the offline vector path is unaffected.)
    if (is_big_target && (algo == "kheavyhash" || algo == "karlsenhashv2" || algo == "pyrinhashv2"))
      for (unsigned i = 0; i < HASH_LEN / 2; ++i) {
        const uint8_t t = new_target_bin[i];
        new_target_bin[i] = new_target_bin[HASH_LEN - 1 - i];
        new_target_bin[HASH_LEN - 1 - i] = t;
      }

    const uint64_t prev_last_nonce = last_nonce();
    const std::string prev_pool_id = m_pool_id;
    set_job(true, true, v, [&]() {
      m_target    = new_target;
      if (is_big_target) std::memcpy(m_target_bin, new_target_bin, HASH_LEN);
      m_pool_id   = v.at("pool_id");
      m_worker_id = v.at("worker_id");
      m_job_id    = v.at("job_id");
      m_header_hash = v.contains("header_hash") ? v.at("header_hash") : "";
    });
    if (prev_last_nonce) send_last_nonce(prev_last_nonce, m_nonce_bytes, prev_pool_id);

  } else if (type == "bench") {
    debug_startup("process bench start");
    m_is_bench = true;
    set_job(true, true, v);
    debug_startup("process bench done");
    m_target = 0;

  } else if (type == "test") {
    debug_startup("process test start");
    m_is_bench = false;
    set_job(false, false, v);
    debug_startup("process test done");
    m_nonce32 = 0;
    m_nonce64 = 0;
    m_target  = 0;

  } else if (type == "pause") {
    ++ m_job_ref; // to stop rx threads if any
    set_fn(nullptr);

  } else if (type == "read_msr" || type == "write_msr") {
     // in case of unknown MSR mod support stop here
     const auto pi = msr_mods.find(ci.msrMod());
     if (pi == msr_mods.end()) throw std::string("Unsupported CPU");

     // build default_msr_items from input message params
     const MsrItems& msr_items = pi->second;
     MsrItems default_msr_items;
     for (const auto& vi : v) {
       const std::string& key = vi.first;
       if (key.starts_with("msr:0x")) {
         const std::string& value = vi.second;
         const uint32_t reg = strtoul(key.substr(6).c_str(), NULL, 16);
         auto parts = tokenize(value, ',');
         if (parts.size() != 2 || !parts[0].starts_with("0x") || !parts[1].starts_with("0x"))
           throw std::string("Wrong value,mask MSR value: " + value);
         default_msr_items[reg] = MsrValue(
           strtoul(parts[0].substr(2).c_str(), NULL, 16),
           strtoul(parts[1].substr(2).c_str(), NULL, 16)
         );
       }
     }

     auto msr = xmrig::Msr::get();

     if (type == "read_msr") { // read missing default_msr_items and return them all as result
       MessageValues result;
       for (const auto& i : msr_items) {
         const uint32_t reg = i.first;
         MsrValue value;
         const auto pi = default_msr_items.find(reg);
         if (pi == default_msr_items.end()) {
           const auto msr_item = msr->read(reg);
           if (!msr_item.isValid()) throw fmt::format("Can't read MSR register {:#x}", reg);
           value = MsrValue(msr_item.value(), msr_item.mask());
         } else value = pi->second;
         result[fmt::format("msr:{:#x}", reg)] =
           fmt::format("{:#x},{:#x}", value.value, value.mask);
       }
       send_msg("read_msr", result);

     } else { // write msr
       // set the MSR mod if the active algo needs it, otherwise reset MSRs to default values
       const std::string algo = v.contains("algo") ? v.at("algo") : "";
       const bool is_set_msr =
         algo.starts_with("rx/") || algo == "ghostrider" || algo == "cn-heavy/xhv";

       for (const auto& i : msr_items) {
         const uint32_t reg = i.first;
         xmrig::MsrItem msr_item;
         if (!is_set_msr) {
           const auto pi = default_msr_items.find(reg);
           if (pi == default_msr_items.end()) continue; // this item was not recorded before
           msr_item = xmrig::MsrItem(reg, pi->second.value, pi->second.mask);
         } else msr_item = xmrig::MsrItem(reg, i.second.value, i.second.mask);
         msr->write([&msr, msr_item, reg](const int32_t cpu) {
           if (!msr->write(msr_item, cpu))
             throw fmt::format("Can't write MSR register {:#x}", reg);
           return true;
         });
       }
     }

     return true;

  } else if (type == "close") {
    const uint64_t prev_last_nonce = last_nonce();
    if (prev_last_nonce) send_last_nonce(prev_last_nonce, m_nonce_bytes, m_pool_id);
    free_memory();
    return false; // stop processing messages

  } else if (type == "algo_params") {
    get_algo_params(v);
  }

  return true; // continue processing messages
}

// MOM_LOOP_STATS=1 prints, every ~10s on stderr, how the compute loop wall time splits
// between GPU/CPU dispatch calls and the host work around them (plus one-off stall lines).
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
    const uint64_t hash_count = m_hash_count;   // 64-bit: pearl's MAC count per attempt is 2^46, which truncated to 0 mod 2^32
    if (m_dev == DEV::RX_CPU) m_mutex_hashrate.unlock();
    if (hash_count && --hashrate_check_counter == 0) {
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
      // pearl's search seed); dev_sols is the per-call solution count (or C29's -1 EOL / 0 pending).
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
            m_fn.gpu_cn(m_input, m_input_len, m_output, m_spads, m_batch, m_dev_str);
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
          case DEV::PEARL_GPU:
            dev_nonce = m_nonce64;   // the search seed is internal (not embedded in the blob)
            dev_sols = m_fn.gpu_pearl(
              m_job_ref, m_height, m_input, m_input_len, m_output,
              &dev_nonce, m_target_bin,
              m_batch, is_test, m_is_bench, m_dev_str
            );
            break;
          case DEV::KHEAVYHASH_GPU:
            std::memcpy(&dev_nonce, m_input + m_nonce_offset, sizeof(dev_nonce));  // nonce at offset 72
            dev_sols = m_fn.gpu_kheavyhash(
              m_job_ref, m_height, m_input, m_input_len, m_output,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target_bin, m_seed,
              m_batch, is_test, m_is_bench, m_dev_str
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
          case DEV::PYRINHASHV2_GPU:
            std::memcpy(&dev_nonce, m_input + m_nonce_offset, sizeof(dev_nonce));  // nonce at offset 72
            dev_sols = m_fn.gpu_pyrinhashv2(
              m_job_ref, m_height, m_input, m_input_len, m_output,
              static_cast<uint8_t*>(m_spads), &dev_nonce, m_target_bin, m_seed,
              m_batch, is_test, m_is_bench, m_dev_str
            );
            break;
          case DEV::EQUIHASH125_4_GPU:
            // c29-like: the 32-byte nonce lives in the header (offset 108); the solver writes the
            // 52-byte solution (or, in is_test, the gen-kernel rows) out-of-band into m_spads.
            std::memcpy(&dev_nonce, m_input + m_nonce_offset, sizeof(dev_nonce));  // low 8 bytes @108
            dev_sols = m_fn.gpu_equihash125_4(
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
        if (m_dev == DEV::AUTOLYKOS2_GPU || m_dev == DEV::KHEAVYHASH_GPU || m_dev == DEV::FISHHASH_GPU || m_dev == DEV::KARLSENHASHV2_GPU || m_dev == DEV::PYRINHASHV2_GPU) {
          if (dev_sols == 1) {
            char hash[HASH_LEN*2+1];
            send_msg("test", "result", hash_bin2hex(hash, 0));
          } else {
            send_error(m_dev == DEV::KHEAVYHASH_GPU ? "No kheavyhash test result" : "No autolykos2 test result");
          }
          set_fn(nullptr);
          continue;
        }
        if (m_dev == DEV::PEARL_GPU) {
          if (dev_sols == 1) send_msg("test", "result", "ok");
          else send_error("No pearl test result");
          set_fn(nullptr);
          continue;
        }
        if (m_dev == DEV::EQUIHASH125_4_GPU || m_dev == DEV::BEAMHASH3_GPU) {
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
            send_error(m_dev == DEV::BEAMHASH3_GPU ? "No beamhash3 test result" : "No equihash125_4 test result");
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

      // Pearl's work unit is GEMM MACs (m*n*k) so its rate matches the "TH/s" GEMM bench. Equihash's
      // unit is a Wagner SOLUTION (the solver finds ~1.88 distinct proofs per nonce); the solver returns
      // that distinct count in dev_sols when benching, so the rate reads out as Sol/s. Others = 1/batch.
      m_hash_count += (m_dev == DEV::PEARL_GPU) ? pearl_attempt_hashes(m_batch)
                    : (m_dev == DEV::EQUIHASH125_4_GPU || m_dev == DEV::BEAMHASH3_GPU) ? static_cast<uint64_t>(dev_sols < 0 ? 0 : dev_sols)
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
      // autolykos2 and kHeavyHash share the mine-result handling: 8-byte nonce, single 32-byte hash
      // (no mix), nonce embedded in the header at m_nonce_offset (32 for autolykos2, 72 for kHeavyHash).
      if (m_dev == DEV::AUTOLYKOS2_GPU || m_dev == DEV::KHEAVYHASH_GPU || m_dev == DEV::FISHHASH_GPU || m_dev == DEV::KARLSENHASHV2_GPU || m_dev == DEV::PYRINHASHV2_GPU) {
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
      if (m_dev == DEV::EQUIHASH125_4_GPU) {
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
      if (m_dev == DEV::PEARL_GPU) {
        const uint64_t prev_nonce = m_nonce64;
        // Emit at most once per (job_id, header) pair since the PlainProof rebuild is heavy and the
        // pool credits one share per job. Pearlpool varies job_id (reuses header); HeroMiners the reverse.
        if (dev_sols == 1 && m_target) {
          std::string pearl_key = m_job_id;
          pearl_key.append(reinterpret_cast<const char*>(m_input), m_input_len);
          if (pearl_key != m_pearl_proof_job) {
            m_pearl_proof_job = pearl_key;
            MessageValues values;
            values["nonce"]       = nonce_to_hex(dev_nonce, 8); // winning seed, for logging/traceability
            values["plain_proof"] = pearl_proof();     // builds the proof for the captured tile (lazy)
            values["pool_id"]     = m_pool_id;
            values["worker_id"]   = m_worker_id;
            values["job_id"]      = m_job_id;
            send_msg("result", values);
          }
        }
        m_nonce64 += m_nonce_step;   // advance to the next seed (the blob is not touched)
        if (m_target && nonce_overflowed(prev_nonce, m_nonce64, m_nicehash_mask)) {
          set_fn(nullptr);
          send_last_nonce(prev_nonce, 8, m_pool_id);   // pearl seed is 64-bit
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

NAPI_MODULE(NODE_GYP_MODULE_NAME, AsyncWorkerWrapper::Init)
