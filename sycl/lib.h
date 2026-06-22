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
  const std::set<std::string>& gpu_pearl_algos,
  const std::set<std::string>& gpu_kheavyhash_algos,
  const std::set<std::string>& gpu_fishhash_algos,
  const std::set<std::string>& gpu_karlsenhashv2_algos,
  const std::set<std::string>& gpu_pyrinhashv2_algos,
  const std::set<std::string>& gpu_equihash125_4_algos,
  const std::set<std::string>& gpu_beamhash3_algos
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

// kHeavyHash (Kaspa): compute-bound 64x64-matrix + double-Keccak PoW. Shares the etchash ABI
// (32-byte LE target; seed_hash unused, no DAG/epoch). The per-job matrix is host-generated from
// the 80-byte header's first 32 bytes (pre_pow_hash); nonce is an 8-byte LE u64 at offset 72.
MOM_SYCL_API int kheavyhash(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
  uint8_t* mix_hash, uint64_t* pnonce, const uint8_t* target, const uint8_t* seed_hash,
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

// PyrinHashV2 (Pyrin PYI): kHeavyHash-family (no DAG) -- 64x64 matrix matvec, but plain-BLAKE3 powHash/
// final + V2 nibble-XOR reduction. Same etchash ABI; 80-byte Kaspa blob, 8-byte nonce at offset 72.
MOM_SYCL_API int pyrinhashv2(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
  uint8_t* mix_hash, uint64_t* pnonce, const uint8_t* target, const uint8_t* seed_hash,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);

// Equihash 125,4 (ZelHash / Flux): Wagner bucket-collision solver (Tromp/djezo lineage). C29-like
// ABI -- the 32-byte nonce lives in the 140-byte header (offset 108); the solver returns a solution
// COUNT and writes the 52-byte compressed solution(s) out-of-band into solution_out. 256-bit big
// target. is_test runs the M1 gen-kernel validation path (dumps the first entries' expanded rows).
MOM_SYCL_API int equihash125_4(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* solution_out,
  uint64_t* pnonce, const uint8_t* target,
  unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str
);

// BeamHash III (Beam): Wagner k=5 bucket-collision solver. Same c29-like ABI as equihash125_4. Input
// is the prework(32)||nonce(8)||extranonce(4) blob; the solver writes the 104-byte solution(s)
// out-of-band into solution_out and returns the count. is_test runs the M1 gen-validation path.
MOM_SYCL_API int beamhash3(
  unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* solution_out,
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
