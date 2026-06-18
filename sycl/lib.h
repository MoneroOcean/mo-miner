// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>

// Windows needs explicit DLL visibility; everything else exports by default.
#if defined(_WIN32) && defined(MOM_SYCL_BUILD)
#define MOM_SYCL_API __declspec(dllexport)
#elif defined(_WIN32)
#define MOM_SYCL_API __declspec(dllimport)
#else
#define MOM_SYCL_API
#endif

MOM_SYCL_API std::map<std::string, std::string> algo_params(
  unsigned max_cpu_batch, unsigned cpu_sockets, unsigned cpu_threads, unsigned cpu_l3cache,
  const std::map<std::string, unsigned>& algo2mem,
  const std::set<std::string>& cpu_algos,
  const std::set<std::string>& gpu_cn_algos,
  const std::set<std::string>& gpu_c29_algos,
  const std::set<std::string>& gpu_kawpow_algos,
  const std::set<std::string>& gpu_etchash_algos,
  const std::set<std::string>& gpu_autolykos2_algos,
  const std::set<std::string>& gpu_pearl_algos
);

MOM_SYCL_API void cn_gpu(
  const uint8_t* inputs, unsigned input_size, uint8_t* output,
  void* Spads, unsigned batch, const std::string& dev_str
);

MOM_SYCL_API int c29(
  unsigned job_id, unsigned c29_proof_size,
  const uint8_t* inputs, unsigned input_size, uint8_t* output,
  uint32_t* output_edges, uint64_t* pnonce, const std::string& dev_str
);

MOM_SYCL_API int kawpow(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
  uint8_t* mix_hash, uint64_t* pnonce, uint64_t target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);

// FiroPow / EvrProgPow: ProgPoW-0.9.4 variants of KawPoW. Same gpu_kawpow_hash_fun ABI; each bakes
// its own epoch/period divisors + keccak seal (FiroPow: padding-constant seal; EvrProgPow: KawPoW
// seal with the EVRMORE-PROGPOW magic).
MOM_SYCL_API int firopow(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
  uint8_t* mix_hash, uint64_t* pnonce, uint64_t target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);

MOM_SYCL_API int evrprogpow(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
  uint8_t* mix_hash, uint64_t* pnonce, uint64_t target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);

MOM_SYCL_API int etchash(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
  uint8_t* mix_hash, uint64_t* pnonce, const uint8_t* target, const uint8_t* seed_hash,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);

MOM_SYCL_API int autolykos2(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
  uint64_t* pnonce, const uint8_t* target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);

// pearl: input is the 76-byte incomplete header; pseed is the search seed (in/out, set to the
// winning seed on a hit); intensity is the square matrix edge (m=n). On a hit returns 1 and the
// pool-ready base64 PlainProof is available from pearl_proof() (thread-local to this call).
MOM_SYCL_API int pearl(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
  uint64_t* pseed, const uint8_t* target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);
MOM_SYCL_API const char* pearl_proof();
// GEMM MACs per pearl attempt (m*n*k, m=n) -- the work unit the pearl "TH/s" hashrate is quoted in,
// so the core counts this rather than the seed/intensity batch. Mirrors pearl()'s intensity clamp.
MOM_SYCL_API uint64_t pearl_attempt_hashes(unsigned intensity);
