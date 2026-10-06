// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include "async-worker.h"
#include "job-boundary.h"
#include "crypto/common/VirtualMemory.h"
#include "crypto/cn/CnHash.h"
#include "crypto/randomx/randomx.h"
#include "consts.h"

#include <memory>
#include <thread>
#include <vector>

typedef void (*gpu_cn_hash_fun)(
  const uint8_t* input, unsigned input_size, uint8_t* output,
  unsigned batch, const std::string& dev_str, const std::string& backend
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
// PearlHash extends the Autolykos2 call shape with backend, matrix, and certificate controls.
// pseed carries the search seed; a hit exposes its PlainProof and trusted-share claim together.
typedef int (*gpu_pearlhash_hash_fun)(
  unsigned job_ref, uint32_t height,
  const uint8_t* input, unsigned input_size, uint8_t* output,
  uint64_t* pseed, const uint8_t* target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str,
  const std::string& backend, unsigned n, unsigned k, unsigned rank, unsigned cert_version
);
// FishHash variants share the etchash call shape; each implementation owns its
// target byte order and header layout. seed_hash is unused.
typedef gpu_etchash_hash_fun gpu_fishhash_hash_fun;
typedef gpu_etchash_hash_fun gpu_karlsenhashv2_hash_fun;
typedef gpu_etchash_hash_fun gpu_verthash_hash_fun;
// Equihash 125,4/144,5/192,7 solver. The 32-byte nonce lives at header offset 108; 52/100/400-byte
// proofs are returned out-of-band. pnonce carries its current 8-byte counter.
typedef int (*gpu_zelhash_hash_fun)(
  unsigned job_ref, uint32_t height,
  const uint8_t* input, unsigned input_size, uint8_t* solution_out,
  uint64_t* pnonce, const uint8_t* target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);
// BeamHash III (Beam): Wagner k=5 bucket-collision solver. Same c29-like ABI as zelhash -- the
// input is the prework(32)||nonce(8)||extranonce(4) blob; the solver returns a solution COUNT and writes
// the 104-byte solution(s) out-of-band into solution_out. is_test runs the M1 gen-validation path.
typedef gpu_zelhash_hash_fun gpu_beamhash3_hash_fun;
struct FN {
  xmrig::cn_hash_fun cpu{};
  gpu_cn_hash_fun gpu_cn{};
  gpu_c29_hash_fun gpu_c29{};
  gpu_kawpow_hash_fun gpu_kawpow{};
  gpu_etchash_hash_fun gpu_etchash{};
  gpu_autolykos2_hash_fun gpu_autolykos2{};
  gpu_pearlhash_hash_fun gpu_pearlhash{};
  gpu_fishhash_hash_fun gpu_fishhash{};
  gpu_karlsenhashv2_hash_fun gpu_karlsenhashv2{};
  gpu_etchash_hash_fun gpu_misc{};
  gpu_verthash_hash_fun gpu_verthash{};
  gpu_zelhash_hash_fun gpu_zelhash{};
  gpu_beamhash3_hash_fun gpu_beamhash3{};
};
enum DEV {
  CPU,
  RX_CPU,
  GPU,
  C29_GPU,
  C30_GPU,
  KAWPOW_GPU,
  ETCHASH_GPU,
  AUTOLYKOS2_GPU,
  PEARLHASH_GPU,
  FISHHASH_GPU,
  KARLSENHASHV2_GPU,
  MISC_GPU,
  VERTHASH_GPU,
  ZELHASH_GPU,
  BEAMHASH3_GPU,
};

enum class JobMode {
  mine,
  bench,
  test,
};

struct RandomXVmDeleter {
  void operator()(randomx_vm* const vm) const noexcept {
    if (vm) {
      randomx_destroy_vm(vm);
    }
  }
};

inline bool is_nonce_at_32_gpu_dev(const DEV dev) {
  return dev == DEV::KAWPOW_GPU || dev == DEV::ETCHASH_GPU || dev == DEV::AUTOLYKOS2_GPU || dev == DEV::FISHHASH_GPU;
}
// Equihash solvers return proofs out-of-band instead of running a conventional hash loop.
inline bool is_equihash_gpu_dev(const DEV dev) {
  return dev == DEV::ZELHASH_GPU || dev == DEV::BEAMHASH3_GPU;
}
// GPU pow devices that allocate a single small input blob + small output (not a per-batch buffer).
// KarlsenHashV2 is small-blob (80-byte header) but its nonce is at offset 72, not 32.
// Equihash carries a 32-byte nonce at offset 108 and an out-of-band proof buffer.
inline bool is_small_blob_gpu_dev(const DEV dev) {
  return is_nonce_at_32_gpu_dev(dev) || dev == DEV::PEARLHASH_GPU || dev == DEV::KARLSENHASHV2_GPU ||
    dev == DEV::MISC_GPU || dev == DEV::VERTHASH_GPU || dev == DEV::C30_GPU ||
    is_equihash_gpu_dev(dev);
}

class Core: public AsyncWorker {
  const unsigned HASHRATE_COUNTER_INTERVAL = 10; // iterations to skip to update/check hashrate
  inline static std::atomic<unsigned> s_job_ref_source{0};
  FN m_fn;
  DEV m_dev;
  xmrig::VirtualMemory *m_lpads, *m_rx_cache_mem, *m_rx_dataset_mem;
  void* m_spads;
  struct cryptonight_ctx** m_ctx;
  uint8_t *m_input, *m_output;
  uint8_t m_target_bin[HASH_LEN]{}, m_seed[HASH_LEN]{};
  std::atomic<unsigned> m_job_ref;
  unsigned m_height, m_batch, m_mem_size, m_input_len, m_nonce_step,
           m_nonce_bytes, m_nonce_offset, m_thread_id, m_thread_num,
           m_pearlhash_seed_stride, m_c29_proof_size,
           m_pearlhash_n, m_pearlhash_k, m_pearlhash_rank, m_pearlhash_cert_version;
  // The next batch start, or the final nonce of the prepared batch once exhausted.
  uint32_t m_nonce32;
  uint32_t m_pearlhash_seed_start;
  uint64_t m_nonce64, m_nicehash_mask, m_nonce_prefix, m_target, m_timestamp;
  std::atomic<uint64_t> m_hash_count;
  std::string m_algo_str, m_dev_str, m_seed_hex, m_input_hex, m_pool_id, m_worker_id, m_job_id,
              m_job_token, m_header_hash, m_backend;
  bool m_has_fn, m_is_rx_jit, m_nonce_exhausted;
  JobMode m_job_mode;
  randomx_cache*   m_rx_cache;
  randomx_dataset* m_rx_dataset;
  std::vector<std::thread> m_rx_threads;
  std::vector<std::unique_ptr<randomx_vm, RandomXVmDeleter>> m_vms;
  static void join_threads(std::vector<std::thread>& threads) noexcept {
    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }
  inline uint8_t* nonce_address(uint8_t* const input, const unsigned batch = 0) const {
    return input + (batch * m_input_len) + m_nonce_offset;
  }
  inline uint8_t* nonce_address(const unsigned batch = 0) const {
    return nonce_address(m_input, batch);
  }
  // last nonce reached on the current device; pearlhash keeps its 64-bit search seed in m_nonce64
  inline uint64_t last_nonce() const {
    if (m_dev == DEV::RX_CPU) {
      return 0;
    }
    return (m_nonce_bytes == 4 && m_dev != DEV::PEARLHASH_GPU) ? m_nonce32 : m_nonce64;
  }
  // Read the most-significant uint64_t of the 32-byte hash as a little-endian word.
  inline uint64_t result_word(const uint8_t* const output, const unsigned batch) const {
    return mom::job_boundary::load_nonce<uint64_t>(
      output + (batch * HASH_LEN) + HASH_LEN - sizeof(uint64_t), false);
  }
  inline uint64_t result_word(const unsigned batch = 0) const {
    return result_word(m_output, batch);
  }
  template <typename UInt>
  inline UInt next_nonce_after_batch(
    UInt first, uint64_t count, uint64_t lane_stride, uint64_t batch_stride,
    UInt protected_mask
  ) {
    if (!m_target)
      return static_cast<UInt>(first + static_cast<UInt>(batch_stride));
    UInt last = first;
    UInt next = first;
    if (!mom::job_boundary::next_nonce_batch(
        first, count, lane_stride, batch_stride, protected_mask, last, next)) {
      m_nonce_exhausted = true;
      return last;
    }
    return next;
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
    const uint8_t* commitment = nullptr, const uint8_t* mix_hash = nullptr,
    const uint8_t* solution = nullptr, unsigned solution_len = 0
  );
  void send_last_nonce(
    uint64_t nonce, unsigned noncebytes, const std::string& pool_id, const std::string& job_id,
    const std::string& job_token
  );
  void send_equihash_results(uint64_t nonce, unsigned solution_size, bool compact_size_prefix);
  void stop_rx_threads() noexcept;
  void destroy_rx_vms() noexcept;
  void free_memory(
    const bool is_batch_changed    = true,
    const bool is_mem_size_changed = true,
    const bool is_free_cn          = true,
    const bool is_free_rx          = true
  );
  void set_fn(const FN& fn);
  void clear_fn(bool reset_hashrate = true);
  void set_job(
    JobMode mode, const bool is_no_same_input, const MessageValues& v,
    std::function<void(void)> fn_extra_setup = [](){}
  );
  void get_algo_params(const MessageValues& v);
  bool process_message(const std::string& type, const MessageValues& v);

  static bool hex2bin(const char* in, unsigned int len, unsigned char* out);
  inline void next_job_ref() {
    m_job_ref.store(s_job_ref_source.fetch_add(1, std::memory_order_relaxed),
                    std::memory_order_relaxed);
  }
  public:

  Core(napi_env env, napi_value data, napi_value complete, napi_value error_callback)
    : AsyncWorker(env, data, complete, error_callback),
      m_dev(CPU), m_lpads(nullptr), m_rx_cache_mem(nullptr), m_rx_dataset_mem(nullptr),
      m_spads(nullptr), m_ctx(nullptr), m_input(nullptr), m_output(nullptr),
      m_job_ref(s_job_ref_source.fetch_add(1, std::memory_order_relaxed)),
      m_height(0), m_batch(0), m_mem_size(0), m_input_len(0),
      m_nonce_step(1), m_nonce_bytes(4), m_nonce_offset(39), m_thread_id(0),
      m_thread_num(1), m_pearlhash_seed_stride(1),
      m_c29_proof_size(32),
      m_pearlhash_n(131072), m_pearlhash_k(4096), m_pearlhash_rank(256),
      m_pearlhash_cert_version(3),
      m_nonce32(0), m_pearlhash_seed_start(0), m_nonce64(0), m_nicehash_mask(0),
      m_nonce_prefix(0),
      m_target(0), m_timestamp(0),
      m_hash_count(0), m_has_fn(false), m_is_rx_jit(true), m_nonce_exhausted(false),
      m_job_mode(JobMode::mine),
      m_rx_cache(nullptr), m_rx_dataset(nullptr)
  {}

  ~Core() override {
    // The worker's close handler still needs the derived resources and job metadata.
    stop();
  }

  void Execute() override;
};
