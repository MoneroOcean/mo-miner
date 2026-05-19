// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#include "mominer-core.h"

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
#include <inttypes.h>
#include <thread>

static const xmrig::ICpuInfo& cpu_info() { return *xmrig::Cpu::info(); }
#define ci cpu_info()
void (*rx_blake2b_compress)(blake2b_state* S, const uint8_t * block) = rx_blake2b_compress_integer;
int (*rx_blake2b)(void* out, size_t outlen, const void* in, size_t inlen) = rx_blake2b_default;

static void debug_startup(const char* message) {
  if (!std::getenv("MOMINER_DEBUG_STARTUP")) return;
  fprintf(stderr, "MOMINER_DEBUG_STARTUP %s\n", message);
  fflush(stderr);
}

struct MsrValue {
  uint64_t value, mask;
  MsrValue() {};
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
  if (c >= '0' && c <= '9')      return c - '0';
  else if (c >= 'a' && c <= 'f') return c - 'a' + 0xA;
  else if (c >= 'A' && c <= 'F') return c - 'A' + 0xA;
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
  if (n < 10) return '0' + n;
  return 'a' + (n - 10);
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

void Core::send_result(
  const uint64_t nonce, const unsigned noncebytes, const uint8_t* const output,
  const uint32_t* const edges, const unsigned c29_proof_size,
  const uint8_t* const commitment
) {
  MessageValues values;

  // nonce hex
  char nonce_hex[sizeof(uint64_t) * 2 + 1];
  if (noncebytes == 4) {
    snprintf(nonce_hex, sizeof(nonce_hex), "%08x", static_cast<uint32_t>(nonce));
  } else {
    snprintf(nonce_hex, sizeof(nonce_hex), "%016" PRIx64, nonce);
  }
  values["nonce"] = nonce_hex;

  // hash hex
  char hash_hex[HASH_LEN * 2 + 1];
  values["hash"] = hash_bin2hex(output, hash_hex);

  if (commitment) {
    char commitment_hex[HASH_LEN * 2 + 1];
    values["commitment"] = hash_bin2hex(commitment, commitment_hex);
  }

  // edges hex
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
  send_msg("result", values);
}

void Core::send_last_nonce(const uint64_t nonce, const unsigned noncebytes, const std::string& pool_id) {
  MessageValues result;
  char nonce_hex[sizeof(uint64_t)*2+1];
  if (noncebytes == 4) {
    snprintf(nonce_hex, noncebytes*2+1, "%08x", static_cast<uint32_t>(nonce));
  } else {
    snprintf(nonce_hex, noncebytes*2+1, "%016" PRIx64, nonce);
  }
  result["nonce"]   = nonce_hex;
  result["pool_id"] = pool_id;
  send_msg("last_nonce", result);
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
    if (!v.contains("target"))    throw std::string("Missing target job key");
    if (!v.contains("pool_id"))   throw std::string("Missing pool_id job key");
    if (!v.contains("worker_id")) throw std::string("Missing worker_id job key");
    if (!v.contains("job_id"))    throw std::string("Missing job_id job key");
    const std::string new_target_str = v.at("target");

    uint64_t new_target;
    if (new_target_str.size() <= sizeof(uint32_t)*2) {
      uint32_t tmp = 0;
      char str[sizeof(uint32_t)*2 + 1] = "00000000";
      memcpy(str, new_target_str.c_str(), new_target_str.size());
      if ( !hex2bin(str, sizeof(uint32_t), reinterpret_cast<unsigned char*>(&tmp)) ||
           tmp == 0 ) throw std::string("Bad target hex");
      new_target = 0xFFFFFFFFFFFFFFFFULL / (0xFFFFFFFFULL / static_cast<uint64_t>(tmp));

    } else if (new_target_str.size() <= sizeof(uint64_t)*2) {
       uint64_t tmp = 0;
       char str[sizeof(uint64_t)*2 + 1] = "0000000000000000";
       memcpy(str, new_target_str.c_str(), new_target_str.size());
       if ( !hex2bin(str, sizeof(uint64_t), reinterpret_cast<unsigned char*>(&tmp)) ||
            tmp == 0 ) throw std::string("Bad target hex");
       new_target = tmp;

    } else throw std::string("Bad target hex");

    const uint64_t last_nonce = m_nonce_bytes == 4 ? m_nonce32 : m_nonce64;
    const std::string prev_pool_id = m_pool_id;
    set_job(true, true, v, [&]() {
      m_target    = new_target;
      m_pool_id   = v.at("pool_id");
      m_worker_id = v.at("worker_id");
      m_job_id    = v.at("job_id");
    });
    if (last_nonce) send_last_nonce(last_nonce, m_nonce_bytes, prev_pool_id);

  } else if (type == "bench") {
    debug_startup("process bench start");
    set_job(true, false, v);
    debug_startup("process bench done");
    m_target = 0;

  } else if (type == "test") {
    debug_startup("process test start");
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
       // if algo is set and needs MSR mod then set it
       // otherwise reset MSR to default values
       bool is_set_msr = false;
       { const auto pi = v.find("algo");
         if (pi != v.end()) {
           const auto& algo = pi->second;
           is_set_msr = algo.starts_with("rx/") || algo == "ghostrider" || algo == "cn-heavy/xhv";
         }
       }

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
    const uint64_t last_nonce = m_nonce_bytes == 4 ? m_nonce32 : m_nonce64;
    if (last_nonce) send_last_nonce(last_nonce, m_nonce_bytes, m_pool_id);
    free_memory();
    return false; // stop processing messages

  } else if (type == "algo_params") {
    get_algo_params(v);
  }

  return true; // continue processing messages
}

void Core::Execute() {
  debug_startup("Core::Execute entered");
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
#endif

#if !defined(_WIN32)
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
    std::deque<Message> messages;
    fromNode.readAll(messages);
    for (const auto& message : messages) {
      try {
        debug_startup(("message " + message.name).c_str());
        if (message.name == "job" || message.name == "bench" || message.name == "test")
          init_runtime();
        if (!process_message(message.name, message.values)) return;
      } catch(const std::string& err) {
        send_error(std::string("Message processing exception: ") + err);
      }
    }

    // we skip first hash function run using m_hash_count check to exclude GPU compile time
    // that effectively skips it in test mode too
    static unsigned hashrate_check_counter = HASHRATE_COUNTER_INTERVAL;
    if (m_dev == DEV::RX_CPU) m_mutex_hashrate.lock();
    const unsigned hash_count = m_hash_count;
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

    if (m_fn.any) {
      init_runtime();
      int c29_sols;
      uint64_t c29_nonce;
      try {
        switch (m_dev) {
          case DEV::CPU:
            m_fn.cpu(m_input, m_input_len, m_output, m_ctx, m_height);
            break;
          case DEV::GPU:
	    m_fn.gpu_cn(m_input, m_input_len, m_output, m_spads, m_batch, m_dev_str);
	    break;
          case DEV::C29_GPU:
	    c29_nonce = m_nonce_bytes == 4 ? bswap_32(*get_nonce32()) : bswap_64(*get_nonce64());
            c29_sols = m_fn.gpu_c29(
	      m_job_ref, m_c29_proof_size, m_input, m_input_len, m_output,
              static_cast<uint32_t*>(m_spads), &c29_nonce, m_dev_str
	    );
            break;

          case DEV::RX_CPU: throw "Internal error: Unreachable code executed";
        }
      } catch(const std::string& err) {
        send_error(std::string("Compute function exception: ") + err);
        set_fn(nullptr);
        continue;
      } catch(...) {
        send_error("Compute function exception");
        set_fn(nullptr);
        continue;
      }

      if (!m_nonce32 && !m_nonce64) { // test job
	m_input_len = 0; // do not produce any more test jobs for async GPU code like in c29
        if (m_dev == DEV::C29_GPU && c29_sols == 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
        if (m_dev == DEV::C29_GPU && c29_sols == -1) {
          send_msg("test", "result", "EOL");
	  set_fn(nullptr);
	  continue;
	}
	if (m_dev != DEV::C29_GPU || c29_sols == 1) {
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

      m_hash_count += m_batch; // here we do not need mutex since there are no threads
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
          if (c29_sols == 1 && m_target && *get_result() < m_target)
            send_result(c29_nonce, 4, m_output, static_cast<uint32_t*>(m_spads), m_c29_proof_size);
          *get_nonce32() = bswap_32(m_nonce32);
          m_nonce32 += m_nonce_step;
        }

	// check that current nonce is greater than previous one and nince hash protected nonce part is not changed
        if (m_target && ( m_nicehash_mask ? (prev_nonce & static_cast<uint32_t>(m_nicehash_mask)) != (m_nonce32 & static_cast<uint32_t>(m_nicehash_mask)) :
                          prev_nonce > m_nonce32 )
        ) {
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
          if (c29_sols == 1 && m_target && *get_result() < m_target)
            send_result(c29_nonce, 8, m_output, static_cast<uint32_t*>(m_spads), m_c29_proof_size);
          *get_nonce64() = bswap_64(m_nonce64);
          m_nonce64 += m_nonce_step;
        }

	// check that current nonce is greater than previous one and nince hash protected nonce part is not changed
        if (m_target && ( m_nicehash_mask ? (prev_nonce & m_nicehash_mask) != (m_nonce64 & m_nicehash_mask) :
                          prev_nonce > m_nonce64 )
        ) {
          send_error("Nonce overflow");
          set_fn(nullptr);
          continue;
        }
     }

    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

AsyncWorker* create_worker(
  napi_env env, napi_value data, napi_value complete, napi_value error_callback, napi_value options
) {
  return new Core(env, data, complete, error_callback, options);
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, AsyncWorkerWrapper::Init)
