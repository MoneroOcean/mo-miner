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
  const std::set<std::string>& gpu_pearlhash_algos,
  const std::set<std::string>& gpu_fishhash_algos,
  const std::set<std::string>& gpu_karlsenhashv2_algos,
  const std::set<std::string>& gpu_zelhash_algos,
  const std::set<std::string>& gpu_beamhash3_algos
);

MOM_SYCL_API void cn_gpu(
  const uint8_t* inputs, unsigned input_size, uint8_t* output,
  void* Spads, unsigned batch, const std::string& dev_str, const std::string& backend
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

// MeowPow (Meowcoin): KawPoW variant with the classic-Ethereum DAG sizing but a shorter ProgPoW inner
// loop (REGS 16, CNT_CACHE 6, CNT_MATH 9), period 6, and the "MEOWCOINMEOWPOW" keccak seal magic.
MOM_SYCL_API int meowpow(
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

// FishHash (Iron Fish / Karlsen): ASIC-resistant memory-hard PoW (Ethash-derived + BLAKE3). Same
// etchash ABI (32-byte LE target; 8-byte nonce at offset 32; seed_hash unused). Fixed 4.6 GB DAG.
MOM_SYCL_API int fishhash(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
  uint8_t* mix_hash, uint64_t* pnonce, const uint8_t* target, const uint8_t* seed_hash,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);

// KarlsenHashV2 (Karlsen KLS): FishHashPlus -- the FishHash 4.6 GB DAG with a folded index derivation
// and plain-BLAKE3 wrapping. Same etchash ABI; 80-byte Kaspa blob with the 8-byte nonce at offset 72.
MOM_SYCL_API int karlsenhashv2(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
  uint8_t* mix_hash, uint64_t* pnonce, const uint8_t* target, const uint8_t* seed_hash,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);

// Equihash 125,4 (ZelHash / Flux): Wagner bucket-collision solver (Tromp/djezo lineage). C29-like
// ABI -- the 32-byte nonce lives in the 140-byte header (offset 108); the solver returns a solution
// COUNT and writes the 52-byte compressed solution(s) out-of-band into solution_out. 256-bit big
// target. is_test runs the M1 gen-kernel validation path (dumps the first entries' expanded rows).
MOM_SYCL_API int zelhash(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* solution_out,
  uint64_t* pnonce, const uint8_t* target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);

// BeamHash III (Beam): Wagner k=5 bucket-collision solver. Same c29-like ABI as zelhash. Input
// is the prework(32)||nonce(8)||extranonce(4) blob; the solver writes the 104-byte solution(s)
// out-of-band into solution_out and returns the count. is_test runs the M1 gen-validation path.
MOM_SYCL_API int beamhash3(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* solution_out,
  uint64_t* pnonce, const uint8_t* target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);

// pearlhash: input is the 76-byte incomplete header; pseed is the search seed (in/out, set to the
// winning seed on a hit); intensity carries M and the explicit shape fields may select rectangular
// M/N/K/rank profiles. On a hit returns 1 and the pool-ready base64 PlainProof is available from
// pearlhash_proof().
MOM_SYCL_API int pearlhash(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
  uint64_t* pseed, const uint8_t* target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str,
  const std::string& backend, unsigned n, unsigned k, unsigned rank
);
MOM_SYCL_API const char* pearlhash_proof();
// GEMM MACs per pearlhash attempt (m*n*k) -- the work unit the pearlhash "TH/s" hashrate is quoted in, so
// the core counts this rather than the seed/intensity batch. Mirrors pearlhash()'s shape selection.
MOM_SYCL_API uint64_t pearlhash_attempt_hashes(unsigned intensity, unsigned n, unsigned k);

// Release process-scoped SYCL state while the Node environment and compiler runtime are still
// alive. The addon registers this as an environment cleanup hook; it is intentionally separate from
// C++ static destruction because runtime-compiled device images have their own module destructors.
MOM_SYCL_API void sycl_cleanup() noexcept;
