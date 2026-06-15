// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#pragma once

#include "async-worker.h"
#include "ctpl-stl.h" // used for randomx threads
#include "crypto/common/VirtualMemory.h"
#include "crypto/cn/CnHash.h"
#include "crypto/randomx/randomx.h"
#include "consts.h"

typedef void (*cn_any_hash_fun)();
typedef void (*gpu_cn_hash_fun)(
  const uint8_t* input, unsigned input_size, uint8_t* output,
  void* Spads, unsigned batch, const std::string& dev_str
);
typedef int (*gpu_c29_hash_fun)(
  unsigned job_ref, unsigned c29_proof_size,
  const uint8_t* input, unsigned input_size, uint8_t* output,
  uint32_t* output_edges, uint64_t* pnonce, const std::string& dev_str
);
typedef int (*gpu_kawpow_hash_fun)(
  unsigned job_ref, uint32_t height,
  const uint8_t* input, unsigned input_size, uint8_t* output,
  uint8_t* mix_hash, uint64_t* pnonce, uint64_t target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);
typedef int (*gpu_etchash_hash_fun)(
  unsigned job_ref, uint32_t height,
  const uint8_t* input, unsigned input_size, uint8_t* output,
  uint8_t* mix_hash, uint64_t* pnonce, const uint8_t* target, const uint8_t* seed_hash,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);
typedef int (*gpu_autolykos2_hash_fun)(
  unsigned job_ref, uint32_t height,
  const uint8_t* input, unsigned input_size, uint8_t* output,
  uint64_t* pnonce, const uint8_t* target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);
// pearl: same ABI as autolykos2, but pnonce carries the search SEED (not a blob nonce) and on a
// hit the winning seed+tile produce a PlainProof retrieved out-of-band via pearl_proof().
typedef int (*gpu_pearl_hash_fun)(
  unsigned job_ref, uint32_t height,
  const uint8_t* input, unsigned input_size, uint8_t* output,
  uint64_t* pseed, const uint8_t* target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);
static_assert(
  sizeof(cn_any_hash_fun) == sizeof(xmrig::cn_hash_fun) &&
  sizeof(cn_any_hash_fun) == sizeof(gpu_cn_hash_fun) &&
  sizeof(cn_any_hash_fun) == sizeof(gpu_c29_hash_fun) &&
  sizeof(cn_any_hash_fun) == sizeof(gpu_kawpow_hash_fun) &&
  sizeof(cn_any_hash_fun) == sizeof(gpu_etchash_hash_fun) &&
  sizeof(cn_any_hash_fun) == sizeof(gpu_autolykos2_hash_fun) &&
  sizeof(cn_any_hash_fun) == sizeof(gpu_pearl_hash_fun),
  "Compute function pointers differ in size!"
);
union FN {
  cn_any_hash_fun    any;
  xmrig::cn_hash_fun cpu;
  gpu_cn_hash_fun    gpu_cn;
  gpu_c29_hash_fun   gpu_c29;
  gpu_kawpow_hash_fun gpu_kawpow;
  gpu_etchash_hash_fun gpu_etchash;
  gpu_autolykos2_hash_fun gpu_autolykos2;
  gpu_pearl_hash_fun gpu_pearl;
};
enum DEV { CPU, RX_CPU, GPU, C29_GPU, KAWPOW_GPU, ETCHASH_GPU, AUTOLYKOS2_GPU, PEARL_GPU };

inline bool is_nonce_at_32_gpu_dev(const DEV dev) {
  return dev == DEV::KAWPOW_GPU || dev == DEV::ETCHASH_GPU || dev == DEV::AUTOLYKOS2_GPU;
}
// GPU pow devices that allocate a single small input blob + 32-byte output (not a per-batch buffer).
inline bool is_small_blob_gpu_dev(const DEV dev) {
  return is_nonce_at_32_gpu_dev(dev) || dev == DEV::PEARL_GPU;
}

class Core: public AsyncWorker {
  const unsigned HASHRATE_COUNTER_INTERVAL = 10; // iterations to skip to update/check hashrate
  FN m_fn;
  DEV m_dev;
  xmrig::VirtualMemory *m_lpads, *m_rx_cache_mem, *m_rx_dataset_mem;
  void* m_spads;
  struct cryptonight_ctx** m_ctx;
  uint8_t *m_input, *m_output;
  uint8_t m_target_bin[HASH_LEN]{}, m_seed[HASH_LEN]{};
  unsigned m_job_ref, m_height, m_batch, m_mem_size, m_input_len, m_nonce_step,
           m_nonce_bytes, m_nonce_offset, m_c29_proof_size;
  uint32_t m_nonce32; // next nonce that will be used in an input
  uint64_t m_nonce64, m_nicehash_mask, m_target, m_timestamp, m_hash_count;
  std::string m_algo_str, m_dev_str, m_seed_hex, m_input_hex, m_pool_id, m_worker_id, m_job_id, m_header_hash;
  std::string m_pearl_proof_job;   // job_id of the last pearl share emitted (one share built per pool job)
  bool m_is_rx_jit, m_is_bench;
  randomx_cache*   m_rx_cache;
  randomx_dataset* m_rx_dataset;
  ctpl::thread_pool* m_thread_pool;
  randomx_vm** m_vm;
  SimpleMutex m_mutex_hashrate;

  inline uint32_t* get_nonce32(uint8_t* const input, const unsigned batch) {
    return reinterpret_cast<uint32_t*>(input + (batch * m_input_len) + m_nonce_offset);
  }
  inline uint32_t* get_nonce32(const unsigned batch = 0) {
    return get_nonce32(m_input, batch);
  }
  inline uint64_t* get_nonce64(uint8_t* const input, const unsigned batch) {
    return reinterpret_cast<uint64_t*>(input + (batch * m_input_len) + m_nonce_offset);
  }
  inline uint64_t* get_nonce64(const unsigned batch = 0) {
    return get_nonce64(m_input, batch);
  }
  // last nonce reached on the current device; pearl keeps its 64-bit search seed in m_nonce64
  inline uint64_t last_nonce() const {
    return (m_nonce_bytes == 4 && m_dev != DEV::PEARL_GPU) ? m_nonce32 : m_nonce64;
  }
  // points at the most-significant uint64_t of the 32-byte hash (little-endian top word)
  inline const uint64_t* get_result(const uint8_t* const output, const unsigned batch) const {
    return reinterpret_cast<const uint64_t*>(output + (batch * HASH_LEN) + HASH_LEN - sizeof(uint64_t));
  }
  inline const uint64_t* get_result(const unsigned batch = 0) const {
    return get_result(m_output, batch);
  }

  char* hash_bin2hex(const uint8_t* const output, char* hash, const unsigned batch = 0) const;
  char* hash_bin2hex(char* const hash, const unsigned batch) const;
  void send_msg(const std::string key, const MessageValues& values);
  void send_msg(
    const std::string& topic, const std::string& key = std::string(),
    const std::string& value = std::string()
  );
  void send_error(const std::string& str);
  void send_result(
    uint64_t nonce, unsigned noncebytes, const uint8_t* output,
    const uint32_t* edges = nullptr, unsigned c29_proof_size = 32,
    const uint8_t* commitment = nullptr, const uint8_t* mix_hash = nullptr
  );
  void send_last_nonce(uint64_t nonce, unsigned noncebytes, const std::string& pool_id);
  void free_memory(
    const bool is_batch_changed    = true,
    const bool is_mem_size_changed = true,
    const bool is_free_cn          = true,
    const bool is_free_rx          = true
  );
  void set_fn(cn_any_hash_fun fn);
  void set_job(
    const bool is_set_nonce, const bool is_no_same_input, const MessageValues& v,
    std::function<void(void)> fn_extra_setup = [](){}
  );
  void get_algo_params(const MessageValues& v);
  bool process_message(const std::string& type, const MessageValues& v);

  static bool hex2bin(const char* in, unsigned int len, unsigned char* out);
  static std::vector<std::string> tokenize(const std::string& str, const char delim);

  public:

  Core(
    napi_env env, napi_value data, napi_value complete,
    napi_value error_callback, napi_value options
  ) : AsyncWorker(env, data, complete, error_callback),
      m_dev(CPU), m_lpads(nullptr), m_rx_cache_mem(nullptr), m_rx_dataset_mem(nullptr),
      m_spads(nullptr), m_ctx(nullptr), m_input(nullptr), m_output(nullptr),
      m_job_ref(0), m_height(0), m_batch(0), m_mem_size(0), m_input_len(0),
      m_nonce_step(1), m_nonce_bytes(4), m_nonce_offset(39), m_c29_proof_size(32),
      m_nonce32(0), m_nonce64(0), m_nicehash_mask(0), m_target(0), m_timestamp(0),
      m_hash_count(0), m_is_rx_jit(true), m_is_bench(false), m_rx_cache(nullptr), m_rx_dataset(nullptr),
      m_thread_pool(nullptr), m_vm(nullptr)
  {
    m_fn.any = nullptr;
  }

  void Execute() override;
};
