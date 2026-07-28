// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>
//
// PearlHashHash (PRL) NoisyGEMM proof-of-useful-work GPU search kernel (SYCL / Level-Zero).
// Per seed: counter-RNG A/B -> keyed-BLAKE3 commitment roots -> sparse low-rank noise -> noised
// int8 A'*B' XMX GEMM tiles -> per-16x16-tile XOR/rotl13 transcript -> keyed-BLAKE3 jackpot vs
// target. On a win the host builds the pool PlainProof (Merkle openings) directly -- see the
// PEARLHASH_STANDALONE block. Validated against the real verify_plain_proof_v2 (pearl-research-labs).
//
// Throughput notes (Arc B580, k=1024, rank=64, m=n=16384): the search GEMM sustains ~32-34
// TH/s (DPAS MAC/s); the full per-seed attempt ~28-30 TH/s, the rest being the mandatory
// BLAKE3 commitment over A/Bt. Key optimizations: A/B are never materialized (regenerated from
// the counter-RNG inside the consumers), the commitment Merkle tree is reduced in parallel,
// the search uses a PEARLHASH_HR x PEARLHASH_NTILE register tile with software-pipelined B loads.

#include <sycl/sycl.hpp>
#ifndef PEARLHASH_STANDALONE
#include "lib-internal.h"
#endif
#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__) && \
    !defined(MOM_PEARLHASH_ESIMD_TU)
#include <cuda.h>
#include <nvrtc.h>
#endif
#if defined(MOM_SYCL_HAS_HIP)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#endif
#ifdef PEARLHASH_ESIMD
#include <sycl/ext/intel/esimd.hpp>   // experimental register-resident DPAS search path
#endif
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <string>
#include <vector>
#include <set>
#include <array>
#include <map>
#include <mutex>
#include <memory>
#include <thread>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#if defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_HAS_CUDA)
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

// ---- One-shot BLAKE3 (keyed + unkeyed) for host and SYCL device code ----
// Used by the PearlHash kernel for matrix-commitment roots, commitment hashes, the noise PRNG and the
// PoW jackpot. Ported from the BLAKE3 reference impl; validated against the python verifier.
// Device-friendly: no std::, no heap, no recursion; cv_stack is a fixed array.
namespace pearlhash_b3 {

constexpr uint32_t B3_CHUNK_START = 1u << 0;
constexpr uint32_t B3_CHUNK_END = 1u << 1;
constexpr uint32_t B3_PARENT = 1u << 2;
constexpr uint32_t B3_ROOT = 1u << 3;
constexpr uint32_t B3_KEYED = 1u << 4;
constexpr uint32_t B3_CHUNK_LEN = 1024;

inline void iv_words(uint32_t o[8]) {
  o[0] = 0x6a09e667; o[1] = 0xbb67ae85; o[2] = 0x3c6ef372; o[3] = 0xa54ff53a;
  o[4] = 0x510e527f; o[5] = 0x9b05688c; o[6] = 0x1f83d9ab; o[7] = 0x5be0cd19;
}
inline uint32_t b3_rotr(uint32_t w, int c) { return (w >> c) | (w << (32 - c)); }
inline uint32_t le32(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24); }

inline void b3_g(uint32_t* s, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
  s[a] = s[a] + s[b] + mx; s[d] = b3_rotr(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];      s[b] = b3_rotr(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + my; s[d] = b3_rotr(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];      s[b] = b3_rotr(s[b] ^ s[c], 7);
}
inline void b3_round(uint32_t* s, const uint32_t* m) {
  b3_g(s, 0, 4, 8, 12, m[0], m[1]);  b3_g(s, 1, 5, 9, 13, m[2], m[3]);
  b3_g(s, 2, 6, 10, 14, m[4], m[5]); b3_g(s, 3, 7, 11, 15, m[6], m[7]);
  b3_g(s, 0, 5, 10, 15, m[8], m[9]); b3_g(s, 1, 6, 11, 12, m[10], m[11]);
  b3_g(s, 2, 7, 8, 13, m[12], m[13]); b3_g(s, 3, 4, 9, 14, m[14], m[15]);
}

// compress -> full 16-word state.
inline void b3_compress(const uint32_t cv[8], const uint32_t block[16],
                        uint64_t counter, uint32_t block_len, uint32_t flags, uint32_t out[16]) {
  static const int PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  uint32_t s[16];
  for (int i = 0; i < 8; i++) s[i] = cv[i];
  s[8] = 0x6a09e667; s[9] = 0xbb67ae85; s[10] = 0x3c6ef372; s[11] = 0xa54ff53a;
  s[12] = (uint32_t)counter; s[13] = (uint32_t)(counter >> 32); s[14] = block_len; s[15] = flags;
  uint32_t m[16];
  for (int i = 0; i < 16; i++) m[i] = block[i];
  for (int r = 0; r < 7; r++) {
    b3_round(s, m);
    if (r < 6) { uint32_t p[16]; for (int i = 0; i < 16; i++) p[i] = m[PERM[i]]; for (int i = 0; i < 16; i++) m[i] = p[i]; }
  }
  for (int i = 0; i < 8; i++) { out[i] = s[i] ^ s[i + 8]; out[i + 8] = s[i + 8] ^ cv[i]; }
}

inline void words_from_le(const uint8_t* b, uint32_t len, uint32_t out[16]) {
  for (int i = 0; i < 16; i++) {
    uint32_t w = 0;
    for (int j = 0; j < 4; j++) { uint32_t idx = i * 4 + j; if (idx < len) w |= (uint32_t)b[idx] << (8 * j); }
    out[i] = w;
  }
}

// A finalized node: its input CV + last block, enough to recompute root or chaining value.
struct Output {
  uint32_t input_cv[8];
  uint32_t block_words[16];
  uint64_t counter;
  uint32_t block_len;
  uint32_t flags;
};
inline void output_cv(const Output& o, uint32_t cv[8]) {
  uint32_t t[16]; b3_compress(o.input_cv, o.block_words, o.counter, o.block_len, o.flags, t);
  for (int i = 0; i < 8; i++) cv[i] = t[i];
}
inline void output_root(const Output& o, uint8_t out[32]) {
  uint32_t t[16]; b3_compress(o.input_cv, o.block_words, 0, o.block_len, o.flags | B3_ROOT, t);
  for (int i = 0; i < 8; i++) { out[i * 4] = t[i]; out[i * 4 + 1] = t[i] >> 8; out[i * 4 + 2] = t[i] >> 16; out[i * 4 + 3] = t[i] >> 24; }
}
inline void parent_output(const uint32_t left[8], const uint32_t right[8], const uint32_t key[8], uint32_t flags, Output& o) {
  for (int i = 0; i < 8; i++) { o.input_cv[i] = key[i]; o.block_words[i] = left[i]; o.block_words[i + 8] = right[i]; }
  o.counter = 0; o.block_len = 64; o.flags = B3_PARENT | flags;
}

// One-shot keyed (key32 != nullptr) or unkeyed BLAKE3 -> out[32].
inline void b3(const uint8_t* input, uint32_t len, const uint8_t* key32, uint8_t out[32]) {
  uint32_t key[8], base_flags = 0;
  if (key32) { for (int i = 0; i < 8; i++) key[i] = le32(key32 + i*4); base_flags = B3_KEYED; }
  else iv_words(key);

  uint32_t cv_stack[54][8];
  int stack_len = 0;
  uint64_t chunk_counter = 0;
  uint32_t off = 0;

  // chunk state
  uint32_t cv[8];
  for (int i = 0; i < 8; i++) cv[i] = key[i];
  uint8_t block[64];
  uint32_t block_len = 0, blocks_compressed = 0;

  auto start_flag = [&]() -> uint32_t { return blocks_compressed == 0 ? B3_CHUNK_START : 0u; };

  // produce the final Output of the current chunk
  auto chunk_output = [&](Output& o) {
    uint32_t w[16]; words_from_le(block, block_len, w);
    for (int i = 0; i < 8; i++) o.input_cv[i] = cv[i];
    for (int i = 0; i < 16; i++) o.block_words[i] = w[i];
    o.counter = chunk_counter; o.block_len = block_len; o.flags = base_flags | start_flag() | B3_CHUNK_END;
  };
  auto reset_chunk = [&](uint64_t counter) {
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    block_len = 0; blocks_compressed = 0; chunk_counter = counter;
  };
  auto add_cv = [&](uint32_t new_cv[8], uint64_t total_chunks) {
    while ((total_chunks & 1) == 0) {
      Output po; parent_output(cv_stack[stack_len - 1], new_cv, key, base_flags, po);
      stack_len--; uint32_t merged[8]; output_cv(po, merged);
      for (int i = 0; i < 8; i++) new_cv[i] = merged[i];
      total_chunks >>= 1;
    }
    for (int i = 0; i < 8; i++) cv_stack[stack_len][i] = new_cv[i];
    stack_len++;
  };

  while (off < len) {
    // Finalize a full chunk before adding more input (keeps its 16th block as CHUNK_END).
    if (64u * blocks_compressed + block_len == B3_CHUNK_LEN) {
      Output o; chunk_output(o);
      uint32_t ccv[8]; output_cv(o, ccv);
      uint64_t total = chunk_counter + 1;
      add_cv(ccv, total);
      reset_chunk(total);
    }
    uint32_t want_chunk = B3_CHUNK_LEN - (64u * blocks_compressed + block_len);
    uint32_t avail = len - off;
    uint32_t take = want_chunk < avail ? want_chunk : avail;
    uint32_t t = take;
    while (t > 0) {
      if (block_len == 64) {  // compress a non-final block (more data follows in this chunk)
        uint32_t w[16]; words_from_le(block, 64, w);
        uint32_t cout[16]; b3_compress(cv, w, chunk_counter, 64, base_flags | start_flag(), cout);
        for (int i = 0; i < 8; i++) cv[i] = cout[i];
        blocks_compressed++; block_len = 0;
      }
      uint32_t want = 64 - block_len;
      uint32_t n = want < t ? want : t;
      for (uint32_t i = 0; i < n; i++) block[block_len + i] = input[off + i];
      block_len += n; off += n; t -= n;
    }
  }

  // finalize
  Output o; chunk_output(o);
  int rem = stack_len;
  while (rem > 0) {
    rem--;
    uint32_t ocv[8]; output_cv(o, ocv);
    parent_output(cv_stack[rem], ocv, key, base_flags, o);
  }
  output_root(o, out);
}

// ---- Merkle-style tree primitives (for parallel keyed-BLAKE3 of large buffers) ----
inline void load_key(const uint8_t* key32, uint32_t key[8], uint32_t& base_flags) {
  if (key32) { for (int i = 0; i < 8; i++) key[i] = le32(key32 + i*4); base_flags = B3_KEYED; }
  else { iv_words(key); base_flags = 0; }
}
inline void cv_to_bytes(const uint32_t cv[8], uint8_t out[32]) {
  for (int i = 0; i < 8; i++) { out[i*4]=cv[i]; out[i*4+1]=cv[i]>>8; out[i*4+2]=cv[i]>>16; out[i*4+3]=cv[i]>>24; }
}
// Non-root chaining value of one chunk (<=1024 bytes) at chunkIndex.
inline void chunk_cv(const uint8_t* data, uint32_t len, uint64_t chunkIndex, const uint8_t* key32, uint8_t out[32]) {
  uint32_t key[8], base; load_key(key32, key, base);
  uint32_t cv[8]; for (int i = 0; i < 8; i++) cv[i] = key[i];
  uint32_t blocks = (len + 63) / 64; if (blocks == 0) blocks = 1;
  for (uint32_t bi = 0; bi < blocks; bi++) {
    uint32_t bstart = bi * 64; uint32_t blen = (len > bstart) ? (len - bstart < 64 ? len - bstart : 64) : 0;
    uint32_t flags = base; if (bi == 0) flags |= B3_CHUNK_START; if (bi == blocks - 1) flags |= B3_CHUNK_END;
    uint32_t w[16]; words_from_le(data + bstart, blen, w);
    uint32_t t[16]; b3_compress(cv, w, chunkIndex, blen, flags, t);
    for (int i = 0; i < 8; i++) cv[i] = t[i];
  }
  cv_to_bytes(cv, out);
}
// Parent CV of two child CVs; isRoot applies the ROOT flag (top of tree).
inline void parent_cv(const uint8_t l[32], const uint8_t r[32], const uint8_t* key32, bool isRoot, uint8_t out[32]) {
  uint32_t key[8], base; load_key(key32, key, base);
  uint32_t block[16];
  for (int i = 0; i < 8; i++) { block[i] = le32(l + i*4); block[8+i] = le32(r + i*4); }
  uint32_t t[16]; b3_compress(key, block, 0, 64, base | B3_PARENT | (isRoot ? B3_ROOT : 0), t);
  cv_to_bytes(t, out);
}
// Merge numChunks contiguous CVs (cvs[i*32]) into the BLAKE3 root, applying ROOT at the top.
// Iterative largest-power-of-two-subtree split (matches pearl-blake3 merkle.rs / blake3.js).
inline void merge_root(uint8_t* cvs, int numChunks, const uint8_t* key32, uint8_t out[32]) {
  if (numChunks == 1) { for (int i = 0; i < 32; i++) out[i] = cvs[i]; return; }
  // explicit stack of half-open ranges to process post-order
  struct Fr { int lo, hi; bool done; };
  Fr stk[64]; int sp = 0;
  uint8_t res[64][32]; int rsp = 0;  // results stack of CVs
  stk[sp++] = {0, numChunks, false};
  while (sp > 0) {
    Fr& f = stk[sp - 1];
    if (f.hi - f.lo == 1) { for (int i = 0; i < 32; i++) res[rsp][i] = cvs[f.lo * 32 + i]; rsp++; sp--; continue; }
    if (!f.done) {
      f.done = true;
      int sz = f.hi - f.lo, p = 1; while (p * 2 < sz) p *= 2;
      int mid = f.lo + p;
      stk[sp++] = {mid, f.hi, false};  // right pushed first -> processed/popped after left
      stk[sp++] = {f.lo, mid, false};
    } else {
      // children results are top two of res stack: left then right (left pushed last -> on top)
      bool isRoot = (f.lo == 0 && f.hi == numChunks);
      uint8_t left[32], right[32];
      for (int i = 0; i < 32; i++) { left[i] = res[rsp - 2][i]; right[i] = res[rsp - 1][i]; }
      rsp -= 2;
      parent_cv(left, right, key32, isRoot, res[rsp]); rsp++;
      sp--;
    }
  }
  for (int i = 0; i < 32; i++) out[i] = res[0][i];
}

}  // namespace pearlhash_b3

namespace mom_pearlhash {

using pearlhash_b3::b3;

// --- IGC multi-accumulator codegen workaround -------------------------------
// Holding several joint_matrix accumulator fragments live in a C-array and
// iterating over that array with a plain `for` / `#pragma unroll` miscompiles on
// the Intel oneAPI 2026.0 stack for Battlemage/Xe2 (Arc B580): the program JITs
// cleanly but aborts at runtime with UR_RESULT_ERROR_DEVICE_LOST. The defect is
//   https://github.com/intel/llvm/issues/21409
// ("the pragma unroll is not doing what is expected" over an array of matrix
// fragments). It is gated by the host icpx -O2 SPIR-V shape (not the IGC JIT
// level), is non-monotonic in fragment count (NTILE 2/3 crash, 1/4/6/8 ran), and
// still reproduces on nightly-2026-06-12. A standalone reproducer + analysis
// lives in ~/bug_report3. The fix from #21409 is to replace the loop with a
// C++17 fold-expression manual unroll, which generates correct code at every
// NTILE; MU<N>(f) below is exactly that and is used for every loop that touches
// the accumulator-fragment array. Do NOT turn these back into a `for` loop.
template <class T, T... I, class F>
static inline void mu_impl(std::integer_sequence<T, I...>, F&& f) { (f(std::integral_constant<T, I>{}), ...); }
template <int N, class F>
static inline void MU(F&& f) { mu_impl(std::make_integer_sequence<int, N>{}, static_cast<F&&>(f)); }

// counter-based RNG (lowbias32) -- the deterministic A/B matrix generator
static inline int8_t gv(uint32_t seed, uint32_t idx) {
  uint32_t x = seed + idx * 0x9e3779b9u;
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return (int8_t)((x & 127) - 64);
}
static inline uint32_t mulhi(uint32_t a, uint32_t b) { return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32); }
static inline uint32_t rotl(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
static inline uint32_t rd_u32le(const uint8_t* p) { return pearlhash_b3::le32(p); }
// Sparse-noise index pair for one k-column: first index `f` in [0,rank), second `se != f` (a
// permutation step within the rank). Same mapping for both the A and B noise tables.
static inline void perm_pair(uint32_t u, int rank, int32_t& f, int32_t& se) {
  uint32_t a = u & (rank - 1); f = (int32_t)a; se = (int32_t)(a ^ (1 + mulhi(rank - 1, u)));
}
static inline void mk_seed(uint8_t out[32], char c0) { const char* s = c0 == 'A' ? "A_tensor" : "B_tensor"; for (int i = 0; i < 32; i++) out[i] = i < 8 ? (uint8_t)s[i] : 0; }
static inline void dev_randHash(int index, const uint8_t* seed32, const uint8_t* key32, int prepend, uint8_t out[32]) {
  uint8_t mb[64]{};   // (1+index) LE at word `prepend`, seed32 at bytes 32..63, rest zero
  uint32_t v = (uint32_t)(1 + index);
  mb[prepend*4]=(uint8_t)v; mb[prepend*4+1]=(uint8_t)(v>>8); mb[prepend*4+2]=(uint8_t)(v>>16); mb[prepend*4+3]=(uint8_t)(v>>24);
  for (int i = 0; i < 32; i++) mb[32+i] = seed32[i];
  b3(mb, 64, key32, out);
}

struct Result { int found; uint32_t seed; uint32_t row; uint32_t col; uint32_t chk; };

struct Buffers {
  int8_t *EAL, *EBR, *EBRt, *Ap, *Bp;       // A'/B' (noised, search inputs); A/B regenerated from RNG
  int8_t *ApWmma, *BpWmma;
  int32_t *EARp1, *EARp2, *EBLq1, *EBLq2;
  uint8_t *cA, *cB, *key, *target, *CVA, *CVB;
  uint32_t* transcript;
  Result* result;
};

template <bool DirectLayout>
static void k_noise_ebr(sycl::queue& q, const Buffers& B, int n, int rank, int blocks) {
  q.parallel_for(sycl::range<1>(blocks), [=](sycl::id<1> id) {
    const int i = (int)id[0];
    uint8_t sd[32]; mk_seed(sd, 'B');
    uint8_t h[32]; dev_randHash(i, sd, B.cB, 0, h);
    const int total = n * rank;
    for (int j = 0; j < 32; ++j) {
      const int o = i * 32 + j;
      if (o >= total) break;
      const int8_t value = (int8_t)((h[j] & 63) - 32);
      if constexpr (DirectLayout) B.EBR[(o % rank) * n + o / rank] = value;
      else B.EBRt[o] = value;
    }
  });
}

// Parallel BLAKE3 Merkle reduction of `nc` contiguous chunk CVs in `cvs` (uint8 buffer sized
// >= 2*nc*32). For power-of-two nc the largest-power-of-two split degenerates to a balanced
// binary tree, so a pairwise level reduction is bit-identical to merge_root(). Each level reads
// the previous level's CV region and writes a fresh, non-overlapping region (no in-place race);
// the final parent carries the ROOT flag. Returns the CV-unit offset of the root. Falls back to
// nothing here — caller must guarantee nc is a power of two for this path.
static int reduce_tree_pow2(sycl::queue& q, uint8_t* cvs, const uint8_t* key, int nc) {
  int rd = 0, wr = nc, cnt = nc;
  while (cnt > 1) {
    int half = cnt / 2; bool isRoot = (half == 1); int rd0 = rd, wr0 = wr;
    q.parallel_for(sycl::range<1>(half), [=](sycl::id<1> id) { int i = (int)id[0];
      pearlhash_b3::parent_cv(cvs + (size_t)(rd0 + 2 * i) * 32, cvs + (size_t)(rd0 + 2 * i + 1) * 32, key, isRoot, cvs + (size_t)(wr0 + i) * 32); });
    rd = wr; wr += half; cnt = half;
  }
  return rd;
}
static void k_roots(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k) {
  auto B = bb;
  const int lenA = m * k, lenB = n * k; const uint32_t tot = (uint32_t)(m * k);
  const int ncA = (lenA + 1023) / 1024, ncB = (lenB + 1023) / 1024;
  // A bytes are regenerated on the fly (A[idx]=gv(seed,idx)) -- no A buffer materialized.
  q.parallel_for(sycl::range<1>(ncA), [=](sycl::id<1> id) { int ci = (int)id[0]; int off = ci * 1024; int len = lenA - off < 1024 ? lenA - off : 1024;
    uint8_t buf[1024];
    for (int t = 0; t < len; t++) buf[t] = (uint8_t)gv(seed, (uint32_t)(off + t));
    pearlhash_b3::chunk_cv(buf, (uint32_t)len, (uint64_t)ci, B.key, B.CVA + ci * 32); });
  // B' root is over Bt (n x k); each 1024-byte chunk of the Bt byte-stream is regenerated straight
  // from B's RNG (B[i]=gv(seed,tot+i)) so neither B nor a transposed Bt buffer exists. Bt byte
  // b -> B[(b%k)*n + b/k].
  q.parallel_for(sycl::range<1>(ncB), [=](sycl::id<1> id) { int ci = (int)id[0]; int off = ci * 1024; int len = lenB - off < 1024 ? lenB - off : 1024;
    uint8_t buf[1024];
    int j = off / k, r = off % k;        // Bt byte (off+t) -> column j, depth r of B; advance incrementally
    for (int t = 0; t < len; t++) { buf[t] = (uint8_t)gv(seed, tot + (uint32_t)(r * n + j)); if (++r == k) { r = 0; j++; } }
    pearlhash_b3::chunk_cv(buf, (uint32_t)len, (uint64_t)ci, B.key, B.CVB + ci * 32); });
  // Both branches finish the same commitment chain (cB = B3(key||rootB), cA = B3(cB||rootA)); they
  // differ only in how each tree's root is obtained (parallel pairwise reduce vs serial merge_root).
  const bool pow2 = ((ncA & (ncA - 1)) == 0) && ((ncB & (ncB - 1)) == 0);
  if (pow2) {
    int offA = reduce_tree_pow2(q, B.CVA, B.key, ncA);
    int offB = reduce_tree_pow2(q, B.CVB, B.key, ncB);
    q.single_task([=]() {
      uint8_t buf[64];
      for (int i = 0; i < 32; i++) { buf[i] = B.key[i]; buf[32 + i] = B.CVB[(size_t)offB * 32 + i]; } b3(buf, 64, nullptr, B.cB);
      for (int i = 0; i < 32; i++) { buf[i] = B.cB[i]; buf[32 + i] = B.CVA[(size_t)offA * 32 + i]; } b3(buf, 64, nullptr, B.cA);
    });
  } else {
    q.single_task([=]() {
      uint8_t rootA[32], rootB[32], buf[64];
      pearlhash_b3::merge_root(B.CVA, ncA, B.key, rootA);
      pearlhash_b3::merge_root(B.CVB, ncB, B.key, rootB);
      for (int i = 0; i < 32; i++) { buf[i] = B.key[i]; buf[32 + i] = rootB[i]; } b3(buf, 64, nullptr, B.cB);
      for (int i = 0; i < 32; i++) { buf[i] = B.cB[i]; buf[32 + i] = rootA[i]; } b3(buf, 64, nullptr, B.cA);
    });
  }
}

static void k_noise(sycl::queue& q, const Buffers& bb, int m, int n, int k, int rank) {
  auto B = bb;
  const int dAL = (m * rank + 31) / 32, dBR = (n * rank + 31) / 32, dP = (k * 4 + 31) / 32;
  q.parallel_for(sycl::range<1>(dAL), [=](sycl::id<1> id) { int i = (int)id[0]; uint8_t sd[32]; mk_seed(sd, 'A'); uint8_t h[32]; dev_randHash(i, sd, B.cA, 0, h);
    int total = m * rank; for (int j = 0; j < 32; j++) { int o = i * 32 + j; if (o < total) B.EAL[o] = (int8_t)((h[j] & 63) - 32); } });
  const bool cpu = q.get_device().is_cpu();
  if (cpu) {
    // The Windows OpenCL CPU runtime faults in the otherwise-trivial transpose kernel. Its small
    // compatibility workload writes the final layout directly; GPUs retain coalesced generation
    // plus transpose, which is substantially faster for production matrix sizes.
    k_noise_ebr<true>(q, B, n, rank, dBR);
  } else {
    k_noise_ebr<false>(q, B, n, rank, dBR);
  }
  q.parallel_for(sycl::range<1>(dP), [=](sycl::id<1> id) { int i = (int)id[0]; uint8_t sd[32]; mk_seed(sd, 'A'); uint8_t h[32]; dev_randHash(i, sd, B.cA, 1, h);
    for (int kk = 0; kk < 8; kk++) { int idx = i * 8 + kk; if (idx >= k) break; perm_pair(rd_u32le(h + kk*4), rank, B.EARp1[idx], B.EARp2[idx]); } });
  q.parallel_for(sycl::range<1>(dP), [=](sycl::id<1> id) { int i = (int)id[0]; uint8_t sd[32]; mk_seed(sd, 'B'); uint8_t h[32]; dev_randHash(i, sd, B.cB, 1, h);
    for (int kk = 0; kk < 8; kk++) { int idx = i * 8 + kk; if (idx >= k) break; perm_pair(rd_u32le(h + kk*4), rank, B.EBLq1[idx], B.EBLq2[idx]); } });
  if (!cpu) {
    q.parallel_for(sycl::range<1>((size_t)rank * n), [=](sycl::id<1> id) {
      const int idx = (int)id[0], r = idx / n, c = idx % n;
      B.EBR[idx] = B.EBRt[c * rank + r];
    });
  }
}

static sycl::event compute_ab(sycl::queue& q, const Buffers& bb, uint32_t seed,
           int m, int n, int k, int rank,
           bool portable) {
  auto B = bb; const uint32_t tot = (uint32_t)(m * k);
  // Explicit capture list (not [=]): capturing the extra `portable` bool under [=] makes the host
  // (icx) and device (clang) compilers disagree on the lambda layout in the dual-compiler combined
  // build ("Unexpected kernel lambda size"). Listing the captures explicitly keeps them in sync.
  int8_t* const eal = B.EAL;
  int8_t* const ap = B.Ap;
  int32_t* const earp1 = B.EARp1;
  int32_t* const earp2 = B.EARp2;
  q.parallel_for(sycl::range<1>((size_t)m * k), [eal, ap, earp1, earp2, seed, k, rank, portable](sycl::id<1> id) { int idx = (int)id[0], i = idx / k, c = idx % k;
    int acc = (int)eal[i * rank + earp1[c]] - (int)eal[i * rank + earp2[c]];
    int8_t v = (int8_t)(gv(seed, (uint32_t)idx) + acc);
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
    (void)portable;
    ap[idx] = v;   // row-major A' for the NVIDIA mma.sync search_cuda (nvptx device pass)
#else
    // The spir64 image carries BOTH searches; the host sets `portable` to match the one it will launch:
    //  - portable (the dp4a int8 GEMM search(), used by every OpenCL device -- AMD, the CPU device, and
    //    Intel-OpenCL): ROW-MAJOR A'[i*k+c], so each output row reads a contiguous int8 run over k that
    //    packs into the 4-wide dp4a operand.
    //  - else (ESIMD search_esimd): TILE-MAJOR -- each 8x32 dpas-A fragment is one contiguous 256-byte
    //    block_load (the fast Intel path on Level-Zero).
    if (portable) ap[idx] = v;
    else ap[((size_t)((i >> 3) * (k / 32) + (c >> 5)) * 256) + (i & 7) * 32 + (c & 31)] = v;
#endif
  });
  // B' layout matches the search variant that will run (explicit capture, same reason as above):
  //  - nvptx (__NVPTX__): ROW-MAJOR B'[r,j] for the joint_matrix B operand, else COLUMN-major for the
  //    default mma.sync B operand (coalesced 32-bit loads).
  //  - portable (dp4a search()): COLUMN-major B'[r,j] at j*k+r, so each output column reads a
  //    contiguous int8 run over k -- the dp4a counterpart to the row-major A'.
  //  - else (Intel XMX ESIMD): TILE-MAJOR VNNI, each 16-wide N-tile's k*16 block contiguous (stride 64),
  //    Bp[ (j/16)*k*16 + (r/4)*64 + (j%16)*4 + (r%4) ] = B'[r,j], coalesced sub-group read.
  int8_t* const ebr = B.EBR;
  int8_t* const bp = B.Bp;
  int32_t* const eblq1 = B.EBLq1;
  int32_t* const eblq2 = B.EBLq2;
  if (portable) {
    // B' is generated row-major (adjacent j values read adjacent EBR bytes) but the portable/WMMA
    // search consumes it column-major (adjacent r values). A direct bp[j*k+r] store makes adjacent
    // work-items write k bytes apart and reduced gfx12 throughput to ~1.5 GB/s. Transpose a 32x32
    // tile through 1056 bytes of local memory: both global reads and writes are contiguous.
    return q.submit([&](sycl::handler& h) {
      sycl::local_accessor<int8_t, 2> tile(sycl::range<2>(32, 33), h);
      const size_t groups_r = (static_cast<size_t>(k) + 31) / 32;
      const size_t groups_j = (static_cast<size_t>(n) + 31) / 32;
      h.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(groups_r * 8, groups_j * 32),
                          sycl::range<2>(8, 32)),
        [ebr, bp, eblq1, eblq2, seed, tot, k, n, tile](sycl::nd_item<2> item) {
          const int lr = static_cast<int>(item.get_local_id(0));
          const int lj = static_cast<int>(item.get_local_id(1));
          const int r0 = static_cast<int>(item.get_group(0)) * 32;
          const int j0 = static_cast<int>(item.get_group(1)) * 32;
#pragma unroll
          for (int d = 0; d < 32; d += 8) {
            const int r = r0 + lr + d, j = j0 + lj;
            if (r < k && j < n) {
              const int acc = static_cast<int>(ebr[eblq1[r] * n + j]) -
                              static_cast<int>(ebr[eblq2[r] * n + j]);
              tile[lr + d][lj] =
                static_cast<int8_t>(gv(seed, tot + static_cast<uint32_t>(r * n + j)) + acc);
            }
          }
          item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
          for (int d = 0; d < 32; d += 8) {
            const int j = j0 + lr + d, r = r0 + lj;
            if (j < n && r < k) bp[static_cast<size_t>(j) * k + r] = tile[lj][lr + d];
          }
        });
    });
  }
  return q.parallel_for(sycl::range<1>((size_t)k * n), [ebr, bp, eblq1, eblq2, seed, tot, k, n, portable](sycl::id<1> id) { int idx = (int)id[0], r = idx / n, j = idx % n;
    int acc = (int)ebr[eblq1[r] * n + j] - (int)ebr[eblq2[r] * n + j];
    int8_t v = (int8_t)(gv(seed, tot + (uint32_t)idx) + acc);
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
    (void)portable;
#endif
#if defined(__NVPTX__)
    bp[(size_t)j * k + r] = v;                 // col-major B'[r,j] for mma.sync B operand (coalesced loads)
#else
    (void)n;
    if (portable) bp[(size_t)j * k + r] = v;   // col-major B'[r,j] for the dp4a search()
    else bp[(size_t)(j / 16) * k * 16 + (size_t)(r / 4) * 64 + (size_t)(j % 16) * 4 + (r % 4)] = v;
#endif
  });
}

#if defined(MOM_SYCL_HAS_HIP)
struct PearlHashPrepEvents { sycl::event a, b; };

static PearlHashPrepEvents compute_ab_amd_wmma(
    sycl::queue& a_queue, sycl::queue& b_queue, const Buffers& b,
    uint32_t seed, int m, int n, int k, int rank) {
  const uint32_t tot = static_cast<uint32_t>(m * k);
  sycl::event a_event = a_queue.parallel_for(
    sycl::range<1>(static_cast<size_t>(m) * k / 8U),
    [eal=b.EAL, ap=b.ApWmma, earp1=b.EARp1, earp2=b.EARp2,
     seed, k, rank](sycl::id<1> id) {
      const size_t packed_chunk = id[0];
      const int lane = static_cast<int>(packed_chunk & 31U);
      const size_t fragment = packed_chunk >> 5;
      const size_t row_block = fragment / static_cast<size_t>(k / 16);
      const int k_block = static_cast<int>(
        fragment - row_block * static_cast<size_t>(k / 16));
      const size_t row = row_block * 16U + static_cast<size_t>(lane & 15);
      const int col = k_block * 16 + (lane >> 4) * 8;
      uint64_t packed = 0;
#pragma unroll
      for (int byte = 0; byte < 8; ++byte) {
        const int c = col + byte;
        const int acc = static_cast<int>(eal[row * rank + earp1[c]]) -
                        static_cast<int>(eal[row * rank + earp2[c]]);
        const int8_t value = static_cast<int8_t>(
          gv(seed, static_cast<uint32_t>(row * k + c)) + acc);
        packed |= static_cast<uint64_t>(static_cast<uint8_t>(value)) << (byte * 8);
      }
      reinterpret_cast<uint64_t*>(ap)[packed_chunk] = packed;
    });
  // Generate B in row-major 32x32 tiles (coalesced EBR reads), transpose in local memory, and
  // write the exact packed fragment bytes consumed by gfx12 WMMA. This avoids materializing and
  // rereading the 64 MiB portable B matrix on every nonce.
  sycl::event b_event = b_queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<int8_t, 2> tile(sycl::range<2>(32, 33), h);
    const size_t groups_r = (static_cast<size_t>(k) + 31) / 32;
    const size_t groups_j = (static_cast<size_t>(n) + 31) / 32;
    h.parallel_for(
      sycl::nd_range<2>(sycl::range<2>(groups_r * 8, groups_j * 32),
                        sycl::range<2>(8, 32)),
      [ebr=b.EBR, bp=b.BpWmma, eblq1=b.EBLq1, eblq2=b.EBLq2,
       seed, tot, k, n, tile](sycl::nd_item<2> item) {
        const int lr = static_cast<int>(item.get_local_id(0));
        const int lj = static_cast<int>(item.get_local_id(1));
        const int r0 = static_cast<int>(item.get_group(0)) * 32;
        const int j0 = static_cast<int>(item.get_group(1)) * 32;
#pragma unroll
        for (int d = 0; d < 32; d += 8) {
          const int r = r0 + lr + d, j = j0 + lj;
          if (r < k && j < n) {
            const int acc = static_cast<int>(ebr[eblq1[r] * n + j]) -
                            static_cast<int>(ebr[eblq2[r] * n + j]);
            tile[lr + d][lj] = static_cast<int8_t>(
              gv(seed, tot + static_cast<uint32_t>(r * n + j)) + acc);
          }
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int d = 0; d < 32; d += 8) {
          const int j = j0 + lr + d, r = r0 + lj;
          if (j < n && r < k) {
            const size_t fragment =
              (static_cast<size_t>(j / 16) * (k / 16) + r / 16) * 256U;
            const int lane = (j & 15) + ((r & 8) ? 16 : 0);
            bp[fragment + static_cast<size_t>(lane) * 8U + (r & 7)] =
              tile[lj][lr + d];
          }
        }
      });
  });
  return {a_event, b_event};
}
#endif

// The Intel Windows OpenCL CPU compiler faults in compute_ab's indirect-index kernels even though
// the same SPIR-V is correct on Linux and Intel GPUs. CPU allocations are shared USM, so prepare
// the two modest test/fallback matrices directly on the host and still exercise the expensive,
// algorithm-defining GEMM/search as SYCL. GPUs never take this path.
static void compute_ab_cpu(const Buffers& B, uint32_t seed, int m, int n, int k, int rank) {
  const uint32_t tot = (uint32_t)(m * k);
  for (int idx = 0; idx < m * k; ++idx) {
    const int i = idx / k, c = idx % k;
    const int acc = (int)B.EAL[i * rank + B.EARp1[c]] -
                    (int)B.EAL[i * rank + B.EARp2[c]];
    B.Ap[idx] = (int8_t)(gv(seed, (uint32_t)idx) + acc);
  }
  for (int idx = 0; idx < k * n; ++idx) {
    const int r = idx / n, j = idx % n;
    const int acc = (int)B.EBR[B.EBLq1[r] * n + j] -
                    (int)B.EBR[B.EBLq2[r] * n + j];
    B.Bp[(size_t)j * k + r] = (int8_t)(gv(seed, tot + (uint32_t)idx) + acc);
  }
}

// The portable ER x EC micro-tile of output cells each work-item owns raises the dp4a MAC/byte so
// each A'/B' int32 word fetched from L2 feeds ER*EC / (ER+EC) dot products instead of one). A 16x16 hash
// tile is computed by one work-group of 256/(ER*EC) work-items. Tunable for the GPU's register file.
constexpr int pearlhash_portable_er = 4;
constexpr int pearlhash_portable_ec = 4;
// Finalize one tile: serialize its 16-word transcript little-endian, keyed-BLAKE3 it into the
// jackpot, and test jackpot <= target as a little-endian u256. Shared by both search paths; the
// winner-claim atomic differs and stays in the caller (joint_matrix uses sycl::atomic_ref, ESIMD
// esimd::atomic_update). Scalar-only, so it compiles in both the SYCL and ESIMD kernel contexts.
static inline bool tile_wins(const uint32_t tr[16], const uint8_t* key, const uint8_t* target) {
  uint8_t tb[64];
  for (int w = 0; w < 16; w++) { tb[w*4]=tr[w]&0xff; tb[w*4+1]=(tr[w]>>8)&0xff; tb[w*4+2]=(tr[w]>>16)&0xff; tb[w*4+3]=(tr[w]>>24)&0xff; }
  uint8_t jp[32]; b3(tb, 64, key, jp);
  for (int i = 31; i >= 0; i--) { if (jp[i] < target[i]) return true; if (jp[i] > target[i]) return false; }
  return true;   // jackpot == target: still within bound
}
// Position-weighted mix of a tile's (row,col) + its 16-word transcript. When MOM_PEARLHASH_CHK is set every
// search path atomically sums tile_mix over ALL tiles into result->chk -- a whole-search checksum used to
// cross-validate the portable search() against search_esimd/search_cuda: identical A'/B' must give an
// identical chk. The sum is order-independent (any tile schedule) and the row/col weighting makes it
// sensitive to a tile mis-mapping (e.g. an A'/B' transpose). Scalar-only, so it works in every kernel.
[[maybe_unused]] static inline uint32_t tile_mix(uint32_t row, uint32_t col, const uint32_t tr[16]) {
  uint32_t h = row * 2654435761u + col * 2246822519u + 0x9e3779b9u;
  for (int w = 0; w < 16; w++) h = (h ^ tr[w]) * 16777619u + (uint32_t)w;
  return h;
}

// gfx12 exposes wave32 int8 WMMA as an AMDGPU compiler builtin before either SYCL implementation
// exposes it through joint_matrix. DPC++'s AMD device pass accepts that builtin directly inside a
// normal SYCL kernel, so keep this as a backend extension in SYCL source instead of compiling a
// second HIP kernel at runtime. The two shipped shapes are template specializations: K and Rank
// remain compile-time constants and the WMMA loop has the same code shape as the former native path.
#if defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
using PearlHashAmdV2i = int __attribute__((ext_vector_type(2)));
using PearlHashAmdV8i = int __attribute__((ext_vector_type(8)));
template <int K, int Rank, int ER, int EC> class PearlHashAmdWmmaKernel;
template <int K, int Rank, int ER, int EC> class PearlHashAmdHashKernel;

template <int K, int Rank, int ER = 4, int EC = 2>
static void search_amd_wmma_t(sycl::queue& q, const Buffers& bb,
                              uint32_t seed, int m, int n, bool dbg) {
  static thread_local unsigned kernel_stats = 0;
  const bool stats = std::getenv("MOM_PEARLHASH_STATS") && kernel_stats++ < 3;
  const auto wmma_start = std::chrono::steady_clock::now();
  constexpr int wave_size = 32;
  int threads = 256;
  if (const char* value = std::getenv("MOM_PEARLHASH_AMD_WMMA_THREADS")) {
    const int parsed = std::atoi(value);
    if (parsed == 32 || parsed == 64 || parsed == 128 || parsed == 256) threads = parsed;
  }
  const int blocks_m = m / (16 * ER);
  const int blocks_n = n / (16 * EC);
  const uint64_t waves = static_cast<uint64_t>(blocks_m) * blocks_n;
  const size_t global = static_cast<size_t>(waves * wave_size);
  const int8_t* const ap = bb.Ap;
  const int8_t* const bp = bb.Bp;
  Result* const result = bb.result;
  const uint8_t* const cA = bb.cA;
  const uint8_t* const target = bb.target;
  uint32_t* const transcript = bb.transcript;

  // Traverse square groups of waves before advancing through the complete output. This retains the
  // corresponding A'/B' bands in the GPU caches instead of streaming all of B' again for every A'
  // band. Small correctness-test shapes fall back to the simple row-major mapping.
  int cache_block = 32;
  if (const char* value = std::getenv("MOM_PEARLHASH_AMD_WMMA_CACHE_BLOCK"))
    cache_block = std::atoi(value);
  const bool blocked = cache_block > 0 && blocks_m % cache_block == 0 &&
                       blocks_n % cache_block == 0;

  q.parallel_for<PearlHashAmdWmmaKernel<K, Rank, ER, EC>>(
    sycl::nd_range<1>{global, threads},
    [ap, bp, result, cA, target, transcript, blocks_n, cache_block, blocked, seed, dbg](sycl::nd_item<1> item)
      [[sycl::reqd_sub_group_size(32)]] {
#if defined(__AMDGCN__)
    const int lane = static_cast<int>(item.get_local_linear_id() & 31U);
    const uint64_t wave = item.get_global_linear_id() / wave_size;
    int row_group, col_group;
    if (blocked) {
      const uint64_t block_area = static_cast<uint64_t>(cache_block) * cache_block;
      const uint64_t block = wave / block_area;
      const uint64_t inner = wave - block * block_area;
      const int blocks_w = blocks_n / cache_block;
      row_group = static_cast<int>(block / blocks_w) * cache_block +
                  static_cast<int>(inner / cache_block);
      col_group = static_cast<int>(block - (block / blocks_w) * blocks_w) * cache_block +
                  static_cast<int>(inner % cache_block);
    } else {
      row_group = static_cast<int>(wave / blocks_n);
      col_group = static_cast<int>(wave - static_cast<uint64_t>(row_group) * blocks_n);
    }
    const int tile_row = row_group * ER;
    const int tile_col = col_group * EC;
    const int row_or_col = lane & 15;
    const int k_half = (lane & 16) ? 8 : 0;
    auto a_words = sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(
      reinterpret_cast<const int32_t*>(ap));
    auto b_words = sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(
      reinterpret_cast<const int32_t*>(bp));
    PearlHashAmdV8i acc[ER][EC] = {};
    uint32_t tr_lane[ER][EC] = {};

#pragma unroll 2
    for (int kk = 0; kk < K; kk += 16) {
      PearlHashAmdV2i a[ER], b[EC];
#pragma unroll
      for (int i = 0; i < ER; ++i) {
        const uint64_t word =
          (static_cast<uint64_t>((tile_row + i) * 16 + row_or_col) * K + kk + k_half) / 4U;
        a[i] = PearlHashAmdV2i{a_words[word], a_words[word + 1]};
      }
#pragma unroll
      for (int j = 0; j < EC; ++j) {
        const uint64_t word =
          (static_cast<uint64_t>((tile_col + j) * 16 + row_or_col) * K + kk + k_half) / 4U;
        b[j] = PearlHashAmdV2i{b_words[word], b_words[word + 1]};
      }
#pragma unroll
      for (int i = 0; i < ER; ++i) {
#pragma unroll
        for (int j = 0; j < EC; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            true, a[i], true, b[j], acc[i][j], false);
      }

      if ((kk + 16) % Rank == 0) {
        const int rc = kk / Rank;
#pragma unroll
        for (int i = 0; i < ER; ++i) {
#pragma unroll
          for (int j = 0; j < EC; ++j) {
            uint32_t x = 0;
#pragma unroll
            for (int e = 0; e < 8; ++e) x ^= static_cast<uint32_t>(acc[i][j][e]);
#pragma unroll
            for (int delta = 1; delta < wave_size; delta <<= 1) {
              x ^= __builtin_amdgcn_ds_bpermute(static_cast<uint32_t>(lane ^ delta) * 4U, x);
            }
            if (lane == rc) tr_lane[i][j] = x;
          }
        }
      }
    }

    // Each transcript word lives in the lane matching its rank-chunk number. Gather those words
    // directly from the wave and finalize the tile here. This removes the former 4 GiB full-search
    // transcript plus its separate scan kernel while keeping every architecture-facing operation in
    // SYCL (the only backend extension above is the WMMA instruction itself).
    const auto sg = item.get_sub_group();
#pragma unroll
    for (int i = 0; i < ER; ++i) {
#pragma unroll
      for (int j = 0; j < EC; ++j) {
        uint32_t tr[16];
#pragma unroll
        for (int word = 0; word < 16; ++word)
          tr[word] = sycl::select_from_group(sg, tr_lane[i][j], static_cast<uint32_t>(word));
        if (lane == 0) {
          const uint32_t row = static_cast<uint32_t>(tile_row + i) * 16U;
          const uint32_t col = static_cast<uint32_t>(tile_col + j) * 16U;
          if (transcript) {
            const uint64_t tile = static_cast<uint64_t>(row / 16U) *
                                  static_cast<uint64_t>(blocks_n * EC) + col / 16U;
#pragma unroll
            for (int word = 0; word < 16; ++word)
              transcript[tile * 16U + static_cast<uint32_t>(word)] = tr[word];
          } else {
            if (dbg) {
              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                               sycl::access::address_space::global_space> checksum(result->chk);
              checksum.fetch_add(tile_mix(row, col, tr));
            }
            if (tile_wins(tr, cA, target)) {
              sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                               sycl::access::address_space::global_space> found(result->found);
              if (found.exchange(1) == 0) {
                result->seed = seed; result->row = row; result->col = col;
              }
            }
          }
        }
      }
    }
#else
    (void)item; (void)ap; (void)bp; (void)result; (void)cA; (void)target;
    (void)transcript; (void)blocks_n; (void)cache_block; (void)blocked; (void)seed; (void)dbg;
#endif
  });
  if (stats) {
    q.wait_and_throw();
    const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - wmma_start).count();
    std::fprintf(stderr, "PEARLHASH_SYCL_STATS wmma_only=%.3fms\n", ms);
  }

  if (transcript) {
    const auto hash_start = std::chrono::steady_clock::now();
    const uint64_t tile_columns = static_cast<uint64_t>(n / 16);
    const uint64_t tiles = static_cast<uint64_t>(m / 16) * tile_columns;
    q.parallel_for<PearlHashAmdHashKernel<K, Rank, ER, EC>>(
      sycl::range<1>(static_cast<size_t>(tiles)),
      [transcript, result, cA, target, tile_columns, seed, dbg](sycl::id<1> id) {
        const uint64_t tile = id[0];
        uint32_t tr[16];
#pragma unroll
        for (int word = 0; word < 16; ++word)
          tr[word] = transcript[tile * 16U + static_cast<uint32_t>(word)];
        const uint32_t row = static_cast<uint32_t>(tile / tile_columns) * 16U;
        const uint32_t col = static_cast<uint32_t>(tile - (tile / tile_columns) * tile_columns) * 16U;
        if (dbg) {
          sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                           sycl::access::address_space::global_space> checksum(result->chk);
          checksum.fetch_add(tile_mix(row, col, tr));
        }
        if (tile_wins(tr, cA, target)) {
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                           sycl::access::address_space::global_space> found(result->found);
          if (found.exchange(1) == 0) {
            result->seed = seed; result->row = row; result->col = col;
          }
        }
      });
    if (stats) {
      q.wait_and_throw();
      const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - hash_start).count();
      std::fprintf(stderr, "PEARLHASH_SYCL_STATS hash_only=%.3fms\n", ms);
    }
  }
}
#endif

// Portable int8 GEMM search -- the matrix-hardware-free path that EVERY OpenCL device takes: AMD GPUs
// (no ESIMD, no usable joint_matrix), the SYCL CPU device (cpu1), and Intel OpenCL. Each
// work-group computes one 16x16 hash tile; its work-items split the tile's 256 cells into ER x EC
// micro-tiles (one work-item per micro-tile, ER*EC int32 accumulators in registers). A'/B' are
// laid out by compute_ab(portable=true) -- A' row-major, B' column-major -- so each output cell reads a
// contiguous int8 run over k that the compiler packs (4 int8 at a time, loaded as one int32) into the
// 4-wide integer dot product: dp4a on Intel/AMD GPUs, plain scalar int MACs on the CPU -- bit-identical
// either way, since int8*int8->int32 is exact everywhere. There is NO SLM staging and NO per-rank
// barrier: A'/B' reuse is left to the L2 cache, and each work-item folds its own cells into a private
// 16-word partial transcript during the k-loop, so the ONLY work-group synchronisation is a single
// XOR-reduction tree at the end (a plain barrier tree -- not a sub-group op, so no reqd_sub_group_size
// is needed and the kernel runs on AMD and the CPU where the sub-group size is not 16). The result
// (found/seed/row/col) and transcript are bit-identical to search_esimd / search_cuda for the same A'/B'
// (cross-checked via MOM_PEARLHASH_CHK), so this path finds the same tiles. Requires k % rank == 0 and
// k/rank <= 16 (true for every shipped shape: network 4096/256 and pearlpool 1024/64 both give 16).
// Compiled in every build that carries attempt() -- including the Windows/dpcpp Intel build where it
// shares the TU with search_esimd -- and EXCLUDED only from the separate spir64-only ESIMD TU
// (pearlhash_esimd.cpp, which sets MOM_PEARLHASH_ESIMD_TU and re-includes this file just for search_esimd).
#if !defined(MOM_PEARLHASH_ESIMD_TU)
// Intel's Windows CPU OpenCL JIT corrupts its host heap after the work-group search below (the
// failure is an access violation/STATUS_HEAP_CORRUPTION after the kernel has otherwise completed).
// Keep the CPU fallback inside SYCL, but give it a deliberately simple single-task implementation:
// no local accessor, work-group barriers, or shared-USM atomics. CPU mining is only a compatibility
// fallback, so one serial standards-only kernel is preferable to relying on that broken JIT path.
// GPU devices never call this function and retain the measured parallel implementation unchanged.
static void search_cpu(
    sycl::queue& q, const Buffers& bb, uint32_t seed,
    int m, int n, int k, int rank, bool dbg) {
  auto B = bb;
  q.single_task([B, seed, m, n, k, rank, dbg]() {
    constexpr int HT = 16;
    const int nb = k / rank;
    bool found = B.result->found != 0;
    for (int row_base = 0; row_base < m; row_base += HT) {
      for (int col_base = 0; col_base < n; col_base += HT) {
        int32_t acc[HT][HT];
        for (int i = 0; i < HT; ++i)
          for (int j = 0; j < HT; ++j)
            acc[i][j] = 0;
        uint32_t transcript[16]{};
        for (int boundary = 0; boundary < nb; ++boundary) {
          const int begin = boundary * rank;
          const int end = begin + rank;
          for (int depth = begin; depth < end; ++depth) {
            for (int i = 0; i < HT; ++i) {
              const int a = static_cast<int>(
                B.Ap[static_cast<size_t>(row_base + i) * k + depth]);
              for (int j = 0; j < HT; ++j) {
                const int b = static_cast<int>(
                  B.Bp[static_cast<size_t>(col_base + j) * k + depth]);
                acc[i][j] += a * b;
              }
            }
          }
          uint32_t mixed = 0;
          for (int i = 0; i < HT; ++i)
            for (int j = 0; j < HT; ++j)
              mixed ^= static_cast<uint32_t>(acc[i][j]);
          transcript[boundary] = mixed;
        }
        if (dbg)
          B.result->chk += tile_mix(
            static_cast<uint32_t>(row_base), static_cast<uint32_t>(col_base), transcript);
        if (!found && tile_wins(transcript, B.cA, B.target)) {
          B.result->found = 1;
          B.result->seed = seed;
          B.result->row = static_cast<uint32_t>(row_base);
          B.result->col = static_cast<uint32_t>(col_base);
          found = true;
          if (!dbg) return;
        }
      }
    }
  });
}

template <int ER, int EC>
static void search_t(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg) {
  constexpr int HT = 16;
  constexpr int WG = (HT * HT) / (ER * EC);                 // work-items per tile
  [[maybe_unused]] constexpr int BW = HT / EC;              // micro-tile blocks per row (unused on the empty nvptx pass)
  static_assert(HT % ER == 0 && HT % EC == 0, "pearlhash portable: ER/EC must divide 16");
  static_assert((WG & (WG - 1)) == 0, "pearlhash portable: work-group size must be a power of two");
  auto B = bb;
  const int tilesH = m / HT, tilesW = n / HT, k4 = k / 4, nb = k / rank;
  const size_t ntiles = (size_t)tilesH * tilesW;
  // The production A'/B' matrices are 512 MiB each. Plain row-major tile order re-streams all of B'
  // for each A' band; on HIP, visit square tile blocks instead so both slices remain in L2/Infinity
  // Cache. This is only a group-id remap: arithmetic, allocations, and every non-HIP path are unchanged.
  int cache_block = 0;
#if defined(MOM_SYCL_HAS_HIP)
  #if defined(MOM_SYCL_ADAPTIVECPP)
  const bool is_hip = q.get_device().get_backend() == sycl::backend::hip;
  #else
  const bool is_hip = q.get_device().get_backend() == sycl::backend::ext_oneapi_hip;
  #endif
  if (is_hip) {
    cache_block = 64;
    if (const char* value = std::getenv("MOM_PEARLHASH_AMD_DP4A_CACHE_BLOCK"))
      cache_block = std::atoi(value);
  }
#endif
  const bool blocked = cache_block > 1 && tilesH % cache_block == 0 &&
                       tilesW % cache_block == 0;
  const int blocksW = blocked ? tilesW / cache_block : 0;
  // A full 131072-square 2x2 search contains exactly 2^32 work-items. AdaptiveCpp's HIP launcher
  // cannot represent that as one ND-range (hipErrorInvalidValue), even though the group count is
  // legal. Submit bounded contiguous pieces on the in-order queue and add their group base in the
  // kernel. This also keeps custom larger shapes away from the same 32-bit launcher edge.
  constexpr size_t max_groups_per_launch = size_t{1} << 22;
  for (size_t group_base = 0; group_base < ntiles; group_base += max_groups_per_launch) {
    const size_t group_count = std::min(max_groups_per_launch, ntiles - group_base);
    q.submit([&](sycl::handler& h) {
      sycl::local_accessor<uint32_t, 1> red(WG * 16, h);   // batched per-boundary XOR reduction scratch
      // Explicit capture list (not [=]): keeps the kernel-lambda layout identical across the host, spir64
      // and nvptx passes (the nvptx body is #ifdef'd out below), so the dual-compiler combined build's
      // "Unexpected kernel lambda size" static assert stays satisfied.
      h.parallel_for(sycl::nd_range<1>(group_count * WG, WG),
                   [B, seed, red, tilesW, k4, rank, nb, dbg, cache_block, blocked, blocksW,
                    group_base](sycl::nd_item<1> it) {
      const int g = static_cast<int>(group_base + it.get_group(0));
      int R, C;
      if (blocked) {
        const int area = cache_block * cache_block, block = g / area, inner = g % area;
        R = (block / blocksW) * cache_block + inner / cache_block;
        C = (block % blocksW) * cache_block + inner % cache_block;
      } else { R = g / tilesW; C = g % tilesW; }
      const int lid = (int)it.get_local_linear_id();
      const int r0 = (lid / BW) * ER, c0 = (lid % BW) * EC;     // this work-item's micro-tile origin
      const int rowBase = R * HT, colBase = C * HT;
      // A'/B' read as int32 words (4 packed int8) straight from global -- the L2 absorbs the per-tile
      // reuse (each 128KB tile slice stays resident), which is why no SLM staging is needed.
      auto A32 = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                   reinterpret_cast<const int32_t*>(B.Ap));
      auto B32 = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                   reinterpret_cast<const int32_t*>(B.Bp));
      int32_t acc[ER][EC]; for (int i = 0; i < ER; i++) for (int j = 0; j < EC; j++) acc[i][j] = 0;
      uint32_t pX[16]; for (int w = 0; w < 16; w++) pX[w] = 0;   // this work-item's per-boundary partial XOR
      for (int rc = 0; rc < nb; rc++) {
        const int c4lo = rc * (rank / 4), c4hi = c4lo + (rank / 4);
        for (int c4 = c4lo; c4 < c4hi; c4++) {        // each step = 4 columns packed into one int32 word
          int32_t bw[EC]; for (int j = 0; j < EC; j++) bw[j] = B32[(size_t)(colBase + c0 + j) * k4 + c4];
          for (int i = 0; i < ER; i++) {
            const int32_t aw = A32[(size_t)(rowBase + r0 + i) * k4 + c4];
            const int a0 = (int8_t)aw, a1 = (int8_t)(aw >> 8), a2 = (int8_t)(aw >> 16), a3 = (int8_t)(aw >> 24);
            for (int j = 0; j < EC; j++) {           // dp4a: 4 int8 MACs (one ALU op on Intel/AMD GPUs)
              const int32_t b = bw[j];
              acc[i][j] += a0*(int8_t)b + a1*(int8_t)(b >> 8) + a2*(int8_t)(b >> 16) + a3*(int8_t)(b >> 24);
            }
          }
        }
        uint32_t lx = 0; for (int i = 0; i < ER; i++) for (int j = 0; j < EC; j++) lx ^= (uint32_t)acc[i][j];
        pX[rc] = lx;   // cumulative tile XOR at this rank boundary (rc < nb <= 16)
      }
      // Single batched work-group reduction: XOR every work-item's 16-word partial transcript together.
      auto grp = it.get_group();
      for (int w = 0; w < 16; w++) red[lid * 16 + w] = pX[w];
      sycl::group_barrier(grp);
      for (int half = WG >> 1; half > 0; half >>= 1) {
        if (lid < half) for (int w = 0; w < 16; w++) red[lid * 16 + w] ^= red[(lid + half) * 16 + w];
        sycl::group_barrier(grp);
      }
      if (lid == 0) {
        // nb <= 16 => each rank boundary owns a distinct transcript word, so tr[w] == part[w] (the
        // rotl-fold in search_esimd only matters when nb > 16, which no shipped shape uses).
        uint32_t t16[16]; for (int w = 0; w < 16; w++) t16[w] = red[w];
        if (dbg) {
          sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> c(B.result->chk);
          c.fetch_add(tile_mix((uint32_t)rowBase, (uint32_t)colBase, t16));
        }
        if (tile_wins(t16, B.cA, B.target)) {
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> a(B.result->found);
          if (a.exchange(1) == 0) { B.result->seed = seed; B.result->row = (uint32_t)rowBase; B.result->col = (uint32_t)colBase; }
        }
      }
      });
    });
  }
}
// Run the portable search at the measured B580 OpenCL 4x4 micro-tile. Retuning changes the typed
// constants above rather than adding a preprocessor variant. SLM staging and software-pipelined prefetch were both measured slower on
// the B580 (the L2 already absorbs the A'/B' reuse; IGC schedules the loads better than a manual pipeline).
static void search(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg) {
#if defined(MOM_SYCL_HAS_HIP)
  // gfx12 normally takes the SYCL WMMA specialization below. Keep its measured-best 2x2 dp4a
  // fallback for unsupported shapes/compilers (RX 9060 XT: 750 vs 496 GH/s for 2x2 versus the
  // Intel-tuned 4x4). Non-HIP devices retain their own 4x4 default.
  bool is_hip = false;
  #if defined(MOM_SYCL_ADAPTIVECPP)
  is_hip = q.get_device().get_backend() == sycl::backend::hip;
  #else
  is_hip = q.get_device().get_backend() == sycl::backend::ext_oneapi_hip;
  #endif
  if (is_hip) {
    // Keep several fully typed dp4a micro-tiles in the architecture-neutral image. This is a host
    // dispatch, so each hot kernel contains no runtime tile branch. MOM_PEARLHASH_AMD_DP4A_TILE is a
    // tuning/validation override; 2x2 remains the measured default until the final AMD SSCP sweep.
    const char* const tile = std::getenv("MOM_PEARLHASH_AMD_DP4A_TILE");
    if (tile && !std::strcmp(tile, "1x1")) return search_t<1, 1>(q, bb, seed, m, n, k, rank, dbg);
    if (tile && !std::strcmp(tile, "2x4")) return search_t<2, 4>(q, bb, seed, m, n, k, rank, dbg);
    if (tile && !std::strcmp(tile, "4x2")) return search_t<4, 2>(q, bb, seed, m, n, k, rank, dbg);
    if (tile && !std::strcmp(tile, "4x4")) return search_t<4, 4>(q, bb, seed, m, n, k, rank, dbg);
    if (tile && !std::strcmp(tile, "8x2")) return search_t<8, 2>(q, bb, seed, m, n, k, rank, dbg);
    return search_t<2, 2>(q, bb, seed, m, n, k, rank, dbg);
  }
#endif
  search_t<pearlhash_portable_er, pearlhash_portable_ec>(q, bb, seed, m, n, k, rank, dbg);
}
#endif  // !MOM_PEARLHASH_ESIMD_TU

// NVIDIA pearlhash throughput lives entirely in the int8 Tensor Cores, reached here through the
// mma.sync path below. It is built with the intel/llvm DPC++ CUDA backend (nvptx64) -- the SAME
// DPC++ as the Intel build -- so every SYCL kernel stays unified across both GPU vendors, and the
// runtime kernel_compiler JIT that kawpow's per-period specialization relies on is available too.
#if defined(MOM_SYCL_HAS_CUDA)
// ---- NVIDIA tensor-core int8 mma.sync search (DPC++ CUDA backend) ----
// Built with intel/llvm DPC++ for nvptx64 (-fsycl-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_89),
// where joint_matrix maps to the L4's int8 Tensor Cores (wmma.mma.sync IMMA). NVIDIA constraints
// differ from Intel XMX: fixed WMMA shape m16n16k16 (one 16x16 hash tile == one accumulator), only
// row_major/col_major layouts (NO ext_intel_packed VNNI -- compute_ab writes B' row-major here), and
// a 32-lane warp per matrix op. One sub-group (warp) owns one 16x16 hash tile; the cumulative tile is
// XOR-folded into the transcript every `rank` columns via joint_matrix_apply (per-lane, in-register)
// + a sub-group XOR reduce -- no SLM readback (the bottleneck that capped Intel joint_matrix). The
// XOR of all 256 cells is order-independent, so the NVIDIA fragment distribution needs no remap; the
// jackpot is bit-identical to the reference given identical A'/B'.
// Per-warp register tile: each warp computes an ER x EC grid of 16x16 hash tiles, reusing each A'
// row-fragment across EC columns and each B' col-fragment across ER rows (raises MAC/byte so the
// mma issue rate, not the loads, bounds throughput -- the naive 1-tile/warp kernel was load-bound at
// ~6 TH/s). The block swizzle groups warps into BLK x BLK super-blocks so each block's A'/B' slice stays
// L2-resident (the m=n=131072 shape is 512MB/operand -> DRAM-bound without it). Tunable at build.
// 4x4 register tile gives the best MAC/byte reuse without accumulator spill. The block floor is
// generic rather than L4-specific; runtime cache sizing may select a larger value below.
constexpr int pearlhash_cuda_er = 4;
constexpr int pearlhash_cuda_ec = 4;
constexpr int pearlhash_cuda_block_floor = 16;
// One warp computes an ER x EC grid of 16x16 hash tiles using the int8 Tensor-Core MMA
// mma.sync.aligned.m16n8k32 (a 16x16 tile = two n=8 halves). The accumulator stays in registers and
// the inner-hash reads those registers DIRECTLY -- unlike joint_matrix_apply, a plain register read
// does not spill the accumulators / pessimize the GEMM (that capped the joint_matrix path at ~20).
// This is the NVIDIA analog of the Intel ESIMD dpas path. A' is row-major, B' is COLUMN-major (so
// each thread's 4 contiguous int8 of a fragment is one aligned 32-bit load). The super-block
// swizzle keeps the A'/B' slice L2-resident at the 131072 shape.
//
// mma.m16n8k32.s8 register fragment layout (PTX ISA): lane -> gid=lane/4, tid=lane%4.
//   A(16x32,row): a[0]=A[gid][tid*4..],   a[1]=A[gid+8][tid*4..],   a[2]=A[gid][16+tid*4..],   a[3]=A[gid+8][16+tid*4..]
//   B(32x8 ,col): b[0]=B[tid*4..][gid],   b[1]=B[16+tid*4..][gid]
//   C(16x8 ,row): c[0]=C[gid][tid*2], c[1]=C[gid][tid*2+1], c[2]=C[gid+8][tid*2], c[3]=C[gid+8][tid*2+1]
static inline void mma_m16n8k32_s8(int32_t c[4], const uint32_t a[4], const uint32_t b[2]) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  (void)c; (void)a; (void)b;   // host pass: never executed
#endif
}
static inline uint32_t ld_u32(const int8_t* __restrict__ p) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  uint32_t value;
  // PearlHash's A/B matrices are immutable for the whole search. Route fragment reads through CUDA's
  // non-coherent read path and prefetch one nearby 128-byte sector into L2; the blocked traversal
  // consumes that sector in adjacent warps. This is the NVIDIA counterpart of Intel's ESIMD LSC
  // cache hints, and avoids changing the portable/ESIMD/HIP kernels.
  asm volatile("ld.global.nc.L2::128B.u32 %0, [%1];\n" : "=r"(value) : "l"(p));
  return value;
#else
  return *reinterpret_cast<const uint32_t*>(p);
#endif
}
// cp.async helpers (PEARLHASH_CU_PIPE pipeline): async global->shared 16-byte copies that overlap with
// mma, plus the generic->shared address conversion cp.async needs. Device-only (inline PTX).
static inline uint32_t cu_smem_addr(const void* p) {
  uint32_t a = 0;
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
#else
  (void)p;
#endif
  return a;
}
static inline void cu_cp_async16(uint32_t smem, const int8_t* g) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(g));
#else
  (void)smem; (void)g;
#endif
}
static inline void cu_cp_commit() {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile("cp.async.commit_group;\n");
#endif
}
template <int N> static inline void cu_cp_wait() {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
static inline void cu_ldmatrix_x4(uint32_t out[4], uint32_t smem) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]) : "r"(smem));
#else
  (void)out; (void)smem;
#endif
}
static inline void cu_ldmatrix_x2(uint32_t out[2], uint32_t smem) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
    : "=r"(out[0]), "=r"(out[1]) : "r"(smem));
#else
  (void)out; (void)smem;
#endif
}
// CUTE Swizzle<3,4,3>: XOR logical byte-address bits 7..9 into bits 4..6. The low four
// bits stay unchanged, so every cp.async destination remains naturally 16-byte aligned.
static inline int cu_swizzle_128(int off) { return off ^ ((off & 0x380) >> 3); }

// DEFAULT NVIDIA path: one warp per ER x EC grid of 16x16 hash tiles, mma.sync m16n8k32 int8, operands
// from global (L2-resident via the BLK super-block swizzle), accumulators resident in registers
// (pearlhash's inner-hash needs CUMULATIVE C, so tiles cannot be evicted like a normal GEMM), inner-hash
// folds them per rank via direct register reads. ~34 TH/s on an L4 -- correct and the best working
// config; occupancy-bound by the resident-accumulator register tile.
static void search_cuda_direct(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank) {
  constexpr int SG = 32, ER = pearlhash_cuda_er, EC = pearlhash_cuda_ec;
  auto B = bb;
  const int tilesH = m / (16 * ER), tilesW = n / (16 * EC);
  // BLK super-block size: pick the largest power-of-two whose A'/B' operand slice fits the usable L2
  // budget. On <=32 MiB caches reserve half for the concurrent accumulator/transcript traffic; using
  // all 32 MiB regresses RTX 5060 Ti by ~3%. Larger caches can retain the full slice (L4's 48 MiB L2
  // still selects 64, +15% vs the old fixed 16). Small-L2 cards stay at the typed block floor.
  // Slice ~= BLK*16*k*(ER+EC) int8 bytes. MOM_PEARLHASH_CU_BLK overrides. BLK must evenly divide both
  // tile dimensions or the super-block swizzle disables (falls back to row-major).
  int BLK = pearlhash_cuda_block_floor;
  if (const char* e = std::getenv("MOM_PEARLHASH_CU_BLK")) {
    BLK = std::atoi(e); if (BLK < 1) BLK = pearlhash_cuda_block_floor;
  } else {
    const size_t l2 = q.get_device().get_info<sycl::info::device::global_mem_cache_size>();
    const size_t cache_budget = l2 <= (size_t(32) << 20) ? l2 / 2 : l2;
    const size_t per_blk = (size_t)16 * k * (ER + EC);   // operand bytes per unit of BLK
    for (int cand = 128; cand > pearlhash_cuda_block_floor; cand >>= 1)
      if ((size_t)cand * per_blk <= cache_budget && (tilesW % cand) == 0 && (tilesH % cand) == 0) { BLK = cand; break; }
  }
  const size_t nWarp = (size_t)tilesH * tilesW;
  const bool blocked = BLK > 1 && (tilesW % BLK) == 0 && (tilesH % BLK) == 0;
  const int blocksW = blocked ? tilesW / BLK : 0;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> trbuf(sycl::range<1>(ER * EC * 16), h);  // per-warp transcripts in SLM
    h.parallel_for(sycl::nd_range<1>(nWarp * SG, SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
      auto sg = it.get_sub_group();
      const int lane = (int)sg.get_local_linear_id(), gid = lane >> 2, tid = lane & 3;
      const int warp = (int)it.get_group(0);
      int Rg, Cg;
      if (blocked) { const int blk = warp / (BLK * BLK), intra = warp % (BLK * BLK);
        Rg = (blk / blocksW) * BLK + (intra / BLK); Cg = (blk % blocksW) * BLK + (intra % BLK); }
      else { Rg = warp / tilesW; Cg = warp % tilesW; }
      const int rowBase = Rg * 16 * ER, colBase = Cg * 16 * EC;
      const int8_t* __restrict__ Ap = B.Ap; const int8_t* __restrict__ Bp = B.Bp;   // A' row-major (i*k+c), B' col-major (j*k+c)
      int32_t acc[ER * EC * 2][4];
#pragma unroll
      for (int i = 0; i < ER * EC * 2; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0; }
      if (lane == 0)
#pragma unroll
        for (int i = 0; i < ER * EC * 16; i++) trbuf[i] = 0;
      int rc = 0;
      for (int p = 0; p < k; p += 32) {
        uint32_t ra[ER][4];
#pragma unroll
        for (int r = 0; r < ER; r++) {
          const size_t r0 = (size_t)(rowBase + r * 16 + gid) * k + p, r8 = (size_t)(rowBase + r * 16 + gid + 8) * k + p;
          ra[r][0] = ld_u32(Ap + r0 + tid * 4);      ra[r][1] = ld_u32(Ap + r8 + tid * 4);
          ra[r][2] = ld_u32(Ap + r0 + 16 + tid * 4); ra[r][3] = ld_u32(Ap + r8 + 16 + tid * 4);
        }
#pragma unroll
        for (int c = 0; c < EC; c++)
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            const size_t cb = (size_t)(colBase + c * 16 + hh * 8 + gid) * k + p;   // col-major B' column
            uint32_t rb[2] = { ld_u32(Bp + cb + tid * 4), ld_u32(Bp + cb + 16 + tid * 4) };
#pragma unroll
            for (int r = 0; r < ER; r++) mma_m16n8k32_s8(acc[(r * EC + c) * 2 + hh], ra[r], rb);
          }
        if ((p + 32) % rank == 0) {     // rank-chunk boundary: XOR-fold cumulative tiles (register read)
#pragma unroll
          for (int t = 0; t < ER * EC; t++) {
            uint32_t part = (uint32_t)acc[t * 2][0] ^ (uint32_t)acc[t * 2][1] ^ (uint32_t)acc[t * 2][2] ^ (uint32_t)acc[t * 2][3]
                          ^ (uint32_t)acc[t * 2 + 1][0] ^ (uint32_t)acc[t * 2 + 1][1] ^ (uint32_t)acc[t * 2 + 1][2] ^ (uint32_t)acc[t * 2 + 1][3];
            part = sycl::reduce_over_group(sg, part, sycl::bit_xor<uint32_t>{});
            if (lane == 0) trbuf[t * 16 + rc % 16] = rotl(trbuf[t * 16 + rc % 16], 13) ^ part;
          }
          rc++;
        }
      }
      if (lane == 0) {
        for (int r = 0; r < ER; r++) for (int c = 0; c < EC; c++) {
          const int t = r * EC + c;
          uint32_t full[16];
#pragma unroll
          for (int s = 0; s < 16; s++) full[s] = trbuf[t * 16 + s];
          if (!tile_wins(full, B.cA, B.target)) continue;
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> at(B.result->found);
          if (at.exchange(1) == 0) { B.result->seed = seed; B.result->row = (uint32_t)(rowBase + r * 16); B.result->col = (uint32_t)(colBase + c * 16); }
        }
      }
    });
  });
}
// Blackwell path: SMEM-staged + cp.async double-buffered mma.sync pipeline.
// Eight warps cover a 128x256 output tile and stage 128 K values at a time in 96 KiB of swizzled
// local memory. Double-buffered cp.async overlaps the next global-memory slice with the current
// mma.sync work, while ldmatrix feeds each warp's 64x64 accumulator tile. The 16 transcript words
// are split across lane pairs (eight registers per lane), avoiding a second local-memory surface.
// This deliberately trades occupancy for operand reuse and is faster only on Blackwell.
constexpr int pearlhash_cuda_warp_rows = 2;
constexpr int pearlhash_cuda_warp_columns = 4;
constexpr int pearlhash_cuda_tile_rows = 4;
constexpr int pearlhash_cuda_tile_columns = 4;
// K staged per SMEM chunk; multiple of 16 for cp.async and 32 for the matrix K dimension.
constexpr int pearlhash_cuda_block_k = 128;
static void search_cuda_pipe(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank) {
  constexpr int WM = pearlhash_cuda_warp_rows, WN = pearlhash_cuda_warp_columns,
                TM = pearlhash_cuda_tile_rows, TN = pearlhash_cuda_tile_columns,
                BK = pearlhash_cuda_block_k;
  constexpr int NW = WM * WN, TBSZ = NW * 32, BM = WM * TM * 16, BN = WN * TN * 16;
  constexpr int AB = BM * BK, BB = BN * BK, A16 = AB / 16, B16 = BB / 16;
  auto B = bb;
  const int mB = m / BM, nB = n / BN, nChunks = k / BK;
  const size_t nBlocks = (size_t)mB * nB;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int8_t, 1> As(sycl::range<1>(2 * AB), h);                 // double-buffered A' [2][BM][BK]
    sycl::local_accessor<int8_t, 1> Bs(sycl::range<1>(2 * BB), h);                 // double-buffered B' [2][BN][BK] (col-major B')
    h.parallel_for(sycl::nd_range<1>(nBlocks * TBSZ, TBSZ), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
      auto grp = it.get_group();
      const int tid = (int)it.get_local_linear_id();
      const int warp = tid >> 5, lane = tid & 31;
      const int wM = warp / WN, wN = warp % WN;
      // CUDA's dim3(mBlocks,nBlocks) traversal varies x/row fastest, preserving the twice-larger
      // B slice while successive row blocks stream their smaller A slices.
      const int block = (int)it.get_group(0), bR = block % mB, bC = block / mB;
      const int rowBlock = bR * BM, colBlock = bC * BN;
      const int8_t* __restrict__ Ap = B.Ap; const int8_t* __restrict__ Bp = B.Bp;
      int8_t* Asp = As.get_multi_ptr<sycl::access::decorated::no>().get();
      int8_t* Bsp = Bs.get_multi_ptr<sycl::access::decorated::no>().get();
      uint32_t tr[8] = {};
      int32_t acc[TM * TN * 2][4];
#pragma unroll
      for (int i = 0; i < TM * TN * 2; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0; }
      const int warpRow = wM * TM * 16, warpCol = wN * TN * 16;
      auto load_chunk = [&](int buf, int p) {     // async-copy k-chunk at offset p into SMEM buffer buf
        int8_t* aDst = Asp + buf * AB; int8_t* bDst = Bsp + buf * BB;
        for (int ci = tid; ci < A16; ci += TBSZ) { const int mm = (ci * 16) / BK, kk = (ci * 16) % BK;
          cu_cp_async16(cu_smem_addr(aDst + cu_swizzle_128(mm * BK + kk)), Ap + (size_t)(rowBlock + mm) * k + p + kk); }
        for (int ci = tid; ci < B16; ci += TBSZ) { const int nn = (ci * 16) / BK, kk = (ci * 16) % BK;
          cu_cp_async16(cu_smem_addr(bDst + cu_swizzle_128(nn * BK + kk)), Bp + (size_t)(colBlock + nn) * k + p + kk); }
        cu_cp_commit();
      };
      load_chunk(0, 0);     // prologue
      int rc = 0;
      for (int c = 0; c < nChunks; c++) {
        const int buf = c & 1;
        if (c + 1 < nChunks) load_chunk((c + 1) & 1, (c + 1) * BK);   // prefetch next chunk (overlaps the mma below)
        cu_cp_wait<1>();              // current chunk's copy done (keep the prefetch in flight)
        sycl::group_barrier(grp);     // make the cp.async writes visible to all warps
        const int8_t* Ab = Asp + buf * AB; const int8_t* Bb = Bsp + buf * BB;
#pragma unroll
        for (int ks = 0; ks < BK; ks += 32) {
          uint32_t ra[TM][4], rb[TN][2][2];
#pragma unroll
          for (int mt = 0; mt < TM; mt++) {
            const int logical = (warpRow + mt * 16 + (lane >> 1)) * BK + ks + (lane & 1) * 16;
            cu_ldmatrix_x4(ra[mt], cu_smem_addr(Ab + cu_swizzle_128(logical)));
          }
#pragma unroll
          for (int nt = 0; nt < TN; nt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
              const int logical = (warpCol + nt * 16 + hh * 8 + ((lane >> 1) & 7)) * BK + ks + (lane & 1) * 16;
              cu_ldmatrix_x2(rb[nt][hh], cu_smem_addr(Bb + cu_swizzle_128(logical)));
            }
#pragma unroll
          for (int nt = 0; nt < TN; nt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++)
#pragma unroll
              for (int mt = 0; mt < TM; mt++)
                mma_m16n8k32_s8(acc[(mt * TN + nt) * 2 + hh], ra[mt], rb[nt][hh]);
          if (((c * BK + ks) + 32) % rank == 0) {     // cumulative k hit a rank boundary -> fold
#pragma unroll
            for (int t = 0; t < TM * TN; t++) {
              uint32_t part = (uint32_t)acc[t*2][0] ^ (uint32_t)acc[t*2][1] ^ (uint32_t)acc[t*2][2] ^ (uint32_t)acc[t*2][3]
                            ^ (uint32_t)acc[t*2+1][0] ^ (uint32_t)acc[t*2+1][1] ^ (uint32_t)acc[t*2+1][2] ^ (uint32_t)acc[t*2+1][3];
              part = sycl::reduce_over_group(it.get_sub_group(), part, sycl::bit_xor<uint32_t>{});
              if (lane == t * 2 + (rc & 1)) tr[rc >> 1] = part;
            }
            rc++;
          }
        }
        sycl::group_barrier(grp);   // mma done reading this buffer before it is reused two chunks later
      }
      uint32_t other[8];
#pragma unroll
      for (int i = 0; i < 8; i++) other[i] = sycl::select_from_group(it.get_sub_group(), tr[i], lane ^ 1);
      if ((lane & 1) == 0) {
        const int t = lane >> 1, mt = t / TN, nt = t % TN;
        uint32_t full[16];
#pragma unroll
        for (int i = 0; i < 8; i++) { full[i * 2] = tr[i]; full[i * 2 + 1] = other[i]; }
        if (tile_wins(full, B.cA, B.target)) {
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> at(B.result->found);
          if (at.exchange(1) == 0) { B.result->seed = seed; B.result->row = (uint32_t)(rowBlock + warpRow + mt * 16); B.result->col = (uint32_t)(colBlock + warpCol + nt * 16); }
        }
      }
    });
  });
}
static void search_cuda(sycl::queue& q, const Buffers& b, uint32_t seed, int m, int n, int k, int rank) {
  const std::string name = q.get_device().get_info<sycl::info::device::name>();
  // RTX 50 / Blackwell benefits from operand sharing across eight warps. Ada and older retain the
  // direct L2 path, which is faster there. The distributed transcript maps exactly 16 rank chunks.
  const bool blackwell = name.find("RTX 50") != std::string::npos || name.find("Blackwell") != std::string::npos;
  if (blackwell && k == 4096 && rank == 256 && m % 128 == 0 && n % 256 == 0)
    search_cuda_pipe(q, b, seed, m, n, k, rank);
  else
    search_cuda_direct(q, b, seed, m, n, k, rank);
}
#endif  // MOM_SYCL_HAS_CUDA

#ifdef PEARLHASH_ESIMD
// ---- experimental ESIMD register-resident DPAS search (alternative to search()) ----
// One ESIMD work-item owns an ER x EC grid of 16x16 hash tiles in its own GRF
// (ESIMD sub-group size is 1 -- no lane cooperation). The DPAS accumulators stay in registers and
// the per-rank inner-hash XORs them in-GRF, avoiding the joint_matrix_store->SLM->barrier readback
// that caps search() at ~10 TH/s. xmx::dpas(C,B,A): SystolicDepth=8, RepeatCount=8, int8*int8->i32,
// K=32/call; A=8x32 (simd int8 256), B=32x16 VNNI (simd int8 512, == compute_ab's Bp layout),
// C=8x16 (simd int32 128). B' loads straight from Bp; A' is 8 strided row loads from row-major Ap.
// ER=EC=2 (a 2x2 grid of 16x16 tiles => 8 int32 accumulators) is the B580 sweet spot: it maximizes
// operand reuse (16 MAC/byte) while staying within the GRF -- 9+ accumulators spill and collapse
// throughput (ER=4 or EC=4 measured 5-11 TH/s vs 34 at 2x2). Override at build time if retuning.
// ER=EC=2 gives eight accumulators and is the measured GRF sweet spot without spilling.
constexpr int pearlhash_esimd_columns = 2;
constexpr int pearlhash_esimd_rows = 2;
// Cache-blocked work-item traversal (tile swizzle): instead of row-major wi->(Rg,Cg) -- which
// re-streams ALL of B' for every row-band and goes DRAM-bound once A'/B' exceed L2 (the m=n=131072
// HeroMiners shape: 512MB each, ~2TB of redundant DRAM reads -> 12.6 TH/s) -- iterate a BLK x BLK
// square of work-items per "super-block" so its A'/B' slice (~16MB at BLK=64) stays resident in L2
// and is reused across the block's work-items. Cuts DRAM traffic ~32x -> compute-bound again. BLK=0
// disables (plain row-major). Requires tilesH%BLK==0 && tilesW%BLK==0, else falls back to row-major.
constexpr int pearlhash_esimd_block = 64;
namespace esimd_ns = sycl::ext::intel::esimd;
namespace xmx_ns   = sycl::ext::intel::esimd::xmx;

// In-register XOR tree reduction of simd<uint32_t,N> -> uint32_t (N a power of two).
// esimd::reduce<bit_xor> is unimplemented on this stack, so fold by halving select<>s in GRF.
template <int N>
static inline uint32_t esimd_xor_reduce(esimd_ns::simd<uint32_t, N> v) {
  if constexpr (N == 1) return v[0];
  else {
    constexpr int H = N / 2;
    esimd_ns::simd<uint32_t, H> lo = v.template select<H, 1>(0);
    esimd_ns::simd<uint32_t, H> hi = v.template select<H, 1>(H);
    return esimd_xor_reduce<H>(lo ^ hi);
  }
}

// External in the standalone spir64-only ESIMD TU (pearlhash_esimd.cpp) so the main pearlhash.o can call it;
// static in the single-TU Intel/Windows build where it lives in this same object.
#ifdef MOM_PEARLHASH_ESIMD_TU
void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg) {
#else
static void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg) {
#endif
  constexpr int ER = pearlhash_esimd_rows, EC = pearlhash_esimd_columns, HT = 16;
  auto B = bb;
  const int tilesH = m / (HT * ER), tilesW = n / (HT * EC);
  const size_t nWI = (size_t)tilesH * tilesW;
  constexpr int BLK = pearlhash_esimd_block;
  // L2 cache-blocking: only swizzle into BLK x BLK super-blocks when the grid divides evenly (see
  // the cache-blocking note above); otherwise fall back to plain row-major traversal.
  const bool blocked = BLK > 1 && (tilesW % BLK) == 0 && (tilesH % BLK) == 0;
  const int blocksW = blocked ? tilesW / BLK : 0;
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(nWI), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
      using esimd_ns::simd;
      const int wi = (int)id[0];
      int Rg, Cg;
      if (blocked) {                              // swizzle into BLK x BLK super-blocks for L2 reuse
        const int blk = wi / (BLK * BLK), intra = wi % (BLK * BLK);
        Rg = (blk / blocksW) * BLK + (intra / BLK);
        Cg = (blk % blocksW) * BLK + (intra % BLK);
      } else { Rg = wi / tilesW; Cg = wi % tilesW; }
      const int rowBase = Rg * HT * ER, colBase = Cg * EC;
      const int8_t* Ap = B.Ap;
      const int8_t* Bp = B.Bp;
      simd<int32_t, 128> acc0[ER * EC], acc1[ER * EC];
      MU<ER * EC>([&](auto t) { acc0[t] = 0; acc1[t] = 0; });
      uint32_t tr[ER * EC][16] = {};   // per-tile transcript, XOR/rotl-folded at each rank boundary
      int rc = 0;
      // Prefetch the next K-chunk's A' into cache each iteration: cheap (no GRF, unlike a
      // double-buffered load which spills) and addresses load LATENCY. A' is the latency-critical
      // operand (2*ER scattered 256B fragments); a distance-32 A-only prefetch lifts search
      // 31.5->34.2 TH/s, while prefetching the contiguous reused B' measured no gain. ESIMD prefetch
      // is documented DG2/PVC but works on B580/Xe2; max 256B/call, needs an explicit cache hint.
      constexpr auto PFH = esimd_ns::properties{esimd_ns::cache_hint_L1<esimd_ns::cache_hint::cached>,
                                                esimd_ns::cache_hint_L2<esimd_ns::cache_hint::cached>,
                                                esimd_ns::alignment<256>};
      for (int p = 0; p < k; p += 32) {
        // A' tile-major (compute_ab): the 8x32 fragment for band b, k-chunk p is one 256B block;
        // B' is one contiguous 512B VNNI block reused across all ER row-bands.
        simd<int8_t, 256> a0[ER], a1[ER];
        simd<int8_t, 512> bf[EC];
        MU<ER>([&](auto r) { const int band0 = (rowBase + (int)r * HT) >> 3;
          a0[r] = esimd_ns::block_load<int8_t, 256>(Ap + ((size_t)(band0 * (k / 32) + (p >> 5)) * 256));
          a1[r] = esimd_ns::block_load<int8_t, 256>(Ap + ((size_t)((band0 + 1) * (k / 32) + (p >> 5)) * 256)); });
        MU<EC>([&](auto c) { bf[c] = esimd_ns::block_load<int8_t, 512>(Bp + (size_t)(colBase + (int)c) * k * 16 + (size_t)(p / 4) * 64); });
        if (p + 32 < k) MU<ER>([&](auto r) { const int band0 = (rowBase + (int)r * HT) >> 3;
          esimd_ns::prefetch<int8_t, 256>(Ap, ((size_t)(band0 * (k / 32) + ((p + 32) >> 5)) * 256), PFH);
          esimd_ns::prefetch<int8_t, 256>(Ap, ((size_t)((band0 + 1) * (k / 32) + ((p + 32) >> 5)) * 256), PFH); });
        MU<EC>([&](auto c) { MU<ER>([&](auto r) { const int idx = (int)r * EC + (int)c;
          acc0[idx] = xmx_ns::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(acc0[idx], bf[c], a0[r]);
          acc1[idx] = xmx_ns::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(acc1[idx], bf[c], a1[r]); }); });
        if ((p + 32) % rank == 0) {   // rank-chunk boundary: XOR-fold the cumulative tile into the transcript
          MU<ER * EC>([&](auto t) {
            simd<uint32_t, 128> x = acc0[t].template bit_cast_view<uint32_t>() ^ acc1[t].template bit_cast_view<uint32_t>();
            tr[t][rc % 16] = rotl(tr[t][rc % 16], 13) ^ esimd_xor_reduce<128>(x);
          });
          rc++;
        }
      }
      for (int r = 0; r < ER; r++) for (int c = 0; c < EC; c++) {
        if (dbg) {   // whole-search checksum for cross-validation (see tile_mix); ESIMD atomic add
          const uint32_t mx = tile_mix((uint32_t)(rowBase + r * HT), (uint32_t)((colBase + c) * HT), tr[r * EC + c]);
          esimd_ns::atomic_update<esimd_ns::atomic_op::add>(
              &B.result->chk, esimd_ns::simd<uint32_t, 1>(0), esimd_ns::simd<uint32_t, 1>(mx), esimd_ns::simd_mask<1>(1));
        }
        if (!tile_wins(tr[r * EC + c], B.cA, B.target)) continue;
        // ESIMD has no sycl::atomic_ref; use esimd::atomic_update<xchg> to pick one winner
        esimd_ns::simd<int, 1> old = esimd_ns::atomic_update<esimd_ns::atomic_op::xchg>(
            &B.result->found, esimd_ns::simd<uint32_t, 1>(0), esimd_ns::simd<int, 1>(1), esimd_ns::simd_mask<1>(1));
        if (old[0] == 0) { B.result->seed = seed; B.result->row = (uint32_t)(rowBase + r * HT); B.result->col = (uint32_t)((colBase + c) * HT); }
      }
    });
  });
}
#endif  // PEARLHASH_ESIMD


// In the combined build the ESIMD search lives in the separate spir64-only TU (pearlhash_esimd.cpp);
// declare it here so attempt() can call it. (When PEARLHASH_ESIMD is set this is the same TU, defined above.)
#if defined(MOM_PEARLHASH_HAS_ESIMD) && !defined(PEARLHASH_ESIMD)
void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg);
#endif

#if defined(MOM_SYCL_HAS_HIP)
// HIP itself is already required by AdaptiveCpp's HIP backend, but HIPRTC is optional: loading it
// dynamically lets a release worker retain the fully functional SYCL path when the source compiler
// is absent or broken. Generated code is cached locally and is never part of a release archive.
struct PearlHashHiprtcApi {
#if defined(_WIN32)
  HMODULE library = nullptr;
#else
  void* library = nullptr;
#endif
  decltype(&hiprtcCreateProgram) create_program = nullptr;
  decltype(&hiprtcCompileProgram) compile_program = nullptr;
  decltype(&hiprtcGetProgramLogSize) get_log_size = nullptr;
  decltype(&hiprtcGetProgramLog) get_log = nullptr;
  decltype(&hiprtcGetCodeSize) get_code_size = nullptr;
  decltype(&hiprtcGetCode) get_code = nullptr;
  decltype(&hiprtcDestroyProgram) destroy_program = nullptr;
  decltype(&hiprtcGetErrorString) error_string = nullptr;
  std::string error;

  template <typename T>
  bool symbol(T& output, const char* name) {
#if defined(_WIN32)
    output = reinterpret_cast<T>(GetProcAddress(library, name));
#else
    output = reinterpret_cast<T>(dlsym(library, name));
#endif
    if (output) return true;
    error = std::string("missing HIPRTC symbol ") + name;
    return false;
  }

  PearlHashHiprtcApi() {
#if defined(_WIN32)
    const char* names[] = {"hiprtc.dll", "hiprtc0701.dll", "hiprtc0700.dll",
                           "hiprtc0604.dll", "hiprtc0603.dll", "hiprtc0507.dll"};
    for (const char* name : names)
      if ((library = LoadLibraryA(name))) break;
#else
    const char* names[] = {"libhiprtc.so", "libhiprtc.so.7", "libhiprtc.so.6"};
    for (const char* name : names)
      if ((library = dlopen(name, RTLD_NOW | RTLD_LOCAL))) break;
#endif
    if (!library) {
      error = "HIPRTC library is not installed";
      return;
    }
    (void)(symbol(create_program, "hiprtcCreateProgram") &&
           symbol(compile_program, "hiprtcCompileProgram") &&
           symbol(get_log_size, "hiprtcGetProgramLogSize") &&
           symbol(get_log, "hiprtcGetProgramLog") &&
           symbol(get_code_size, "hiprtcGetCodeSize") &&
           symbol(get_code, "hiprtcGetCode") &&
           symbol(destroy_program, "hiprtcDestroyProgram") &&
           symbol(error_string, "hiprtcGetErrorString"));
  }

  bool available() const {
    return library && create_program && compile_program && get_log_size && get_log &&
           get_code_size && get_code && destroy_program && error_string;
  }
  static PearlHashHiprtcApi& instance() {
    static PearlHashHiprtcApi api;
    return api;
  }
};
#endif

#if defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_HAS_CUDA)
static uint64_t pearlhash_jit_hash(const std::string& value) {
  uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char byte : value) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

static std::filesystem::path pearlhash_jit_cache_dir() {
  if (const char* configured = std::getenv("MOM_JIT_CACHE_DIR"); configured && *configured)
    return configured;
#if defined(_WIN32)
  if (const char* base = std::getenv("LOCALAPPDATA"); base && *base)
    return std::filesystem::path(base) / "mom" / "jit";
#else
  if (const char* base = std::getenv("XDG_CACHE_HOME"); base && *base)
    return std::filesystem::path(base) / "mom" / "jit";
  if (const char* home = std::getenv("HOME"); home && *home)
    return std::filesystem::path(home) / ".cache" / "mom" / "jit";
#endif
  return {};
}

static std::vector<char> pearlhash_read_cache(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return {};
  const std::streamsize size = stream.tellg();
  if (size <= 0 || size > 64 * 1024 * 1024) return {};
  std::vector<char> code(static_cast<size_t>(size));
  stream.seekg(0);
  if (!stream.read(code.data(), size)) return {};
  return code;
}

static void pearlhash_write_cache(const std::filesystem::path& path,
                              const std::vector<char>& code) noexcept {
  try {
    if (path.empty() || code.empty()) return;
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    {
      std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
      if (!stream.write(code.data(), static_cast<std::streamsize>(code.size()))) return;
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
      std::filesystem::remove(path, error);
      error.clear();
      std::filesystem::rename(temporary, path, error);
      if (error) std::filesystem::remove(temporary, error);
    }
  } catch (...) {
    // A read-only home/cache directory is not a mining failure; the in-process module still works.
  }
}
#endif

// PearlHash's shared SYCL CUDA kernel already uses cp.async, ldmatrix, and mma.sync, but the SYCL
// execution model cannot express CUTE's staged CTA mainloop without manually duplicating a large
// part of CUTE's layout machinery. That path plateaus near 55.5 TH/s on RTX 5060 Ti; compiling the
// architecture-neutral source below with NVRTC reaches 71.5 TH/s at the same 150 W. The override is
// deliberately optional: no device binary is shipped, and any missing compiler/header/runtime or
// unsupported GPU falls back to the normal selectable SYCL implementation.
#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__) && \
    !defined(MOM_PEARLHASH_ESIMD_TU)
struct PearlHashCudaDriverApi {
#if defined(_WIN32)
  HMODULE library = nullptr;
#else
  void* library = nullptr;
#endif
  decltype(&cuInit) init = nullptr;
  decltype(&cuDeviceGetAttribute) device_attribute = nullptr;
  decltype(&cuDevicePrimaryCtxRetain) retain_primary = nullptr;
  decltype(&cuDevicePrimaryCtxRelease) release_primary = nullptr;
  decltype(&cuCtxSetCurrent) set_current = nullptr;
  decltype(&cuCtxSynchronize) synchronize = nullptr;
  decltype(&cuModuleLoadData) module_load = nullptr;
  decltype(&cuModuleUnload) module_unload = nullptr;
  decltype(&cuModuleGetFunction) module_function = nullptr;
  decltype(&cuFuncSetAttribute) function_attribute = nullptr;
  decltype(&cuLaunchKernel) launch = nullptr;
  decltype(&cuEventCreate) event_create = nullptr;
  decltype(&cuEventDestroy) event_destroy = nullptr;
  decltype(&cuEventRecord) event_record = nullptr;
  decltype(&cuEventQuery) event_query = nullptr;
  decltype(&cuEventElapsedTime) event_elapsed = nullptr;
  decltype(&cuGetErrorString) error_string = nullptr;
  std::string error;

  template <typename T>
  bool symbol(T& output, const char* name) {
#if defined(_WIN32)
    output = reinterpret_cast<T>(GetProcAddress(library, name));
#else
    output = reinterpret_cast<T>(dlsym(library, name));
#endif
    if (output) return true;
    error = std::string("missing CUDA driver symbol ") + name;
    return false;
  }

  PearlHashCudaDriverApi() {
#if defined(_WIN32)
    library = LoadLibraryA("nvcuda.dll");
#else
    library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
    if (!library) {
      error = "the CUDA driver library is not installed";
      return;
    }
    (void)(symbol(init, "cuInit") &&
           symbol(device_attribute, "cuDeviceGetAttribute") &&
           symbol(retain_primary, "cuDevicePrimaryCtxRetain") &&
           symbol(release_primary, "cuDevicePrimaryCtxRelease") &&
           symbol(set_current, "cuCtxSetCurrent") &&
           symbol(synchronize, "cuCtxSynchronize") &&
           symbol(module_load, "cuModuleLoadData") &&
           symbol(module_unload, "cuModuleUnload") &&
           symbol(module_function, "cuModuleGetFunction") &&
           symbol(function_attribute, "cuFuncSetAttribute") &&
           symbol(launch, "cuLaunchKernel") &&
           symbol(event_create, "cuEventCreate") &&
           symbol(event_destroy, "cuEventDestroy") &&
           symbol(event_record, "cuEventRecord") &&
           symbol(event_query, "cuEventQuery") &&
           symbol(event_elapsed, "cuEventElapsedTime") &&
           symbol(error_string, "cuGetErrorString"));
  }

  bool available() const {
    return library && init && device_attribute && retain_primary && release_primary &&
           set_current && synchronize && module_load && module_unload && module_function &&
           function_attribute && launch && event_create && event_destroy && event_record &&
           event_query && event_elapsed && error_string;
  }
  static PearlHashCudaDriverApi& instance() {
    static PearlHashCudaDriverApi api;
    return api;
  }
};

struct PearlHashNvrtcApi {
#if defined(_WIN32)
  HMODULE library = nullptr;
#else
  void* library = nullptr;
#endif
  decltype(&nvrtcCreateProgram) create_program = nullptr;
  decltype(&nvrtcCompileProgram) compile_program = nullptr;
  decltype(&nvrtcGetProgramLogSize) get_log_size = nullptr;
  decltype(&nvrtcGetProgramLog) get_log = nullptr;
  decltype(&nvrtcGetPTXSize) get_ptx_size = nullptr;
  decltype(&nvrtcGetPTX) get_ptx = nullptr;
  decltype(&nvrtcGetCUBINSize) get_cubin_size = nullptr;
  decltype(&nvrtcGetCUBIN) get_cubin = nullptr;
  decltype(&nvrtcGetNumSupportedArchs) get_num_supported_archs = nullptr;
  decltype(&nvrtcGetSupportedArchs) get_supported_archs = nullptr;
  decltype(&nvrtcDestroyProgram) destroy_program = nullptr;
  decltype(&nvrtcGetErrorString) error_string = nullptr;
  std::string error;

  template <typename T>
  bool symbol(T& output, const char* name) {
#if defined(_WIN32)
    output = reinterpret_cast<T>(GetProcAddress(library, name));
#else
    output = reinterpret_cast<T>(dlsym(library, name));
#endif
    if (output) return true;
    error = std::string("missing NVRTC symbol ") + name;
    return false;
  }

  PearlHashNvrtcApi() {
#if defined(_WIN32)
    const char* names[] = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll",
                           "nvrtc64_121_0.dll", "nvrtc64_122_0.dll"};
    for (const char* name : names)
      if ((library = LoadLibraryA(name))) break;
#else
    const char* names[] = {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"};
    for (const char* name : names)
      if ((library = dlopen(name, RTLD_NOW | RTLD_LOCAL))) break;
#endif
    if (!library) {
      error = "NVRTC is not installed";
      return;
    }
    (void)(symbol(create_program, "nvrtcCreateProgram") &&
           symbol(compile_program, "nvrtcCompileProgram") &&
           symbol(get_log_size, "nvrtcGetProgramLogSize") &&
           symbol(get_log, "nvrtcGetProgramLog") &&
           symbol(get_ptx_size, "nvrtcGetPTXSize") &&
           symbol(get_ptx, "nvrtcGetPTX") &&
           symbol(get_cubin_size, "nvrtcGetCUBINSize") &&
           symbol(get_cubin, "nvrtcGetCUBIN") &&
           symbol(get_num_supported_archs, "nvrtcGetNumSupportedArchs") &&
           symbol(get_supported_archs, "nvrtcGetSupportedArchs") &&
           symbol(destroy_program, "nvrtcDestroyProgram") &&
           symbol(error_string, "nvrtcGetErrorString"));
  }

  bool available() const {
    return library && create_program && compile_program && get_log_size && get_log &&
           get_ptx_size && get_ptx && get_cubin_size && get_cubin &&
           get_num_supported_archs && get_supported_archs && destroy_program && error_string;
  }
  static PearlHashNvrtcApi& instance() {
    static PearlHashNvrtcApi api;
    return api;
  }
};

static std::filesystem::path pearlhash_cuda_header_root(
    const char* configured, const std::vector<std::filesystem::path>& candidates,
    const std::filesystem::path& required) {
  if (configured && *configured) {
    const std::filesystem::path path(configured);
    if (std::filesystem::exists(path / required)) return path;
  }
  for (const auto& path : candidates)
    if (std::filesystem::exists(path / required)) return path;
  return {};
}

struct PearlHashCudaSearch {
  CUmodule module = nullptr;
  CUfunction gemm_kernel = nullptr;
  CUfunction hash_kernel = nullptr;
  CUevent started = nullptr;
  CUevent gemm_done = nullptr;
  CUevent done = nullptr;
  CUcontext context = nullptr;
  CUdevice device = 0;
  bool retained_primary = false;
  bool checked = false;
  bool enabled = false;
  unsigned stats_printed = 0;

  static bool is_cuda(const sycl::device& device) {
  #if defined(MOM_SYCL_ADAPTIVECPP)
    return device.get_backend() == sycl::backend::cuda;
  #else
    return device.get_backend() == sycl::backend::ext_oneapi_cuda;
  #endif
  }
  static void cuda_check(PearlHashCudaDriverApi& api, CUresult status,
                         const char* operation) {
    if (status == CUDA_SUCCESS) return;
    const char* detail = nullptr;
    if (api.error_string) (void)api.error_string(status, &detail);
    throw std::string(operation) + ": " + (detail ? detail : "CUDA driver error");
  }
  static std::string rtc_error(PearlHashNvrtcApi& api, nvrtcResult status,
                               nvrtcProgram program, const char* operation) {
    size_t size = 0;
    std::string log;
    if (program && api.get_log_size(program, &size) == NVRTC_SUCCESS && size) {
      log.resize(size);
      (void)api.get_log(program, log.data());
    }
    return std::string(operation) + ": " + api.error_string(status) +
           (log.empty() ? "" : "\n" + log);
  }
  static CUdevice native_device(const sycl::device& device) {
  #if defined(MOM_SYCL_ADAPTIVECPP)
    (void)device;
    return 0;
  #else
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(device);
  #endif
  }
  static bool supports(sycl::queue& q, std::string* const reason = nullptr) {
    const auto unavailable = [&](const std::string& message) {
      if (reason) *reason = message;
      return false;
    };
    if (!is_cuda(q.get_device()))
      return unavailable("the selected SYCL device is not CUDA");
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
    if (!api.available()) return unavailable(api.error);
    if (api.init(0) != CUDA_SUCCESS) return unavailable("cuInit failed");
    int major = 0;
    if (api.device_attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             native_device(q.get_device())) != CUDA_SUCCESS)
      return unavailable("the CUDA compute capability could not be read");
    if (major < 8) return unavailable("the CUTE int8 mainloop requires compute capability 8.0+");
    return true;
  }

  void release() noexcept {
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
    if (api.available() && context) (void)api.set_current(context);
    if (done && api.event_destroy) (void)api.event_destroy(done);
    if (gemm_done && api.event_destroy) (void)api.event_destroy(gemm_done);
    if (started && api.event_destroy) (void)api.event_destroy(started);
    if (module && api.module_unload) (void)api.module_unload(module);
    if (retained_primary && api.release_primary) (void)api.release_primary(device);
    done = nullptr;
    gemm_done = nullptr;
    started = nullptr;
    module = nullptr;
    gemm_kernel = nullptr;
    hash_kernel = nullptr;
    context = nullptr;
    retained_primary = false;
    enabled = false;
  }

  bool ensure(sycl::queue& q, const Buffers& b, int m, int n, int k, int rank) {
    if (!supports(q) || m % 128 || n % 256 || k != 4096 || rank != 256)
      return false;
    // The two-stage native path requires a complete transcript surface. Never launch the CUTE
    // producer when an oversized or failed USM allocation left it null; the shared SYCL path can
    // still execute without that surface.
    if (!b.transcript) return false;
    if (checked) return enabled;
    checked = true;
    try {
      PearlHashCudaDriverApi& driver = PearlHashCudaDriverApi::instance();
      PearlHashNvrtcApi& rtc = PearlHashNvrtcApi::instance();
      if (!rtc.available()) throw rtc.error;

      device = native_device(q.get_device());
      cuda_check(driver, driver.retain_primary(&context, device),
                 "cuDevicePrimaryCtxRetain(PearlHash)");
      retained_primary = true;
      cuda_check(driver, driver.set_current(context), "cuCtxSetCurrent(PearlHash)");

      std::vector<std::filesystem::path> cuda_candidates;
      if (const char* path = std::getenv("CUDA_PATH"); path && *path)
        cuda_candidates.emplace_back(path);
      if (const char* path = std::getenv("CUDA_HOME"); path && *path)
        cuda_candidates.emplace_back(path);
#if defined(_WIN32)
      cuda_candidates.emplace_back("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0");
      cuda_candidates.emplace_back("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6");
#else
      cuda_candidates.emplace_back("/usr/local/cuda");
      cuda_candidates.emplace_back("/opt/nvidia-cuda-ubuntu/usr");
#endif
      const std::filesystem::path cuda_include = pearlhash_cuda_header_root(
        nullptr, [&] {
          std::vector<std::filesystem::path> paths;
          for (const auto& root : cuda_candidates) paths.push_back(root / "include");
          return paths;
        }(), "cuda_runtime.h");

      std::vector<std::filesystem::path> cutlass_candidates;
#if defined(_WIN32)
      if (const char* program_data = std::getenv("ProgramData"); program_data && *program_data)
        cutlass_candidates.emplace_back(
          std::filesystem::path(program_data) / "mom" / "cutlass" / "include");
#else
      cutlass_candidates.emplace_back("/opt/mom/cutlass/include");
      cutlass_candidates.emplace_back("/usr/local/include/cutlass");
#endif
      const std::filesystem::path cutlass_include = pearlhash_cuda_header_root(
        std::getenv("MOM_CUTLASS_INCLUDE_DIR"), cutlass_candidates, "cute/tensor.hpp");

      std::vector<std::filesystem::path> cccl_candidates;
      if (!cuda_include.empty()) {
        cccl_candidates.push_back(cuda_include);
        cccl_candidates.push_back(cuda_include / "cccl");
      }
      const std::filesystem::path cccl_include = pearlhash_cuda_header_root(
        std::getenv("MOM_CCCL_INCLUDE_DIR"), cccl_candidates, "cuda/std/cstdint");
      if (cuda_include.empty()) throw std::string("CUDA runtime headers are not installed");
      if (cutlass_include.empty()) throw std::string("CUTLASS headers are not installed");
      if (cccl_include.empty()) throw std::string("CCCL headers are not installed");

      int capability_major = 0, capability_minor = 0;
      cuda_check(driver, driver.device_attribute(
        &capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
        "cuDeviceGetAttribute(PearlHash compute capability major)");
      cuda_check(driver, driver.device_attribute(
        &capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
        "cuDeviceGetAttribute(PearlHash compute capability minor)");
      const int device_arch = capability_major * 10 + capability_minor;
      int arch_count = 0;
      nvrtcResult arch_status = rtc.get_num_supported_archs(&arch_count);
      if (arch_status != NVRTC_SUCCESS || arch_count <= 0)
        throw rtc_error(rtc, arch_status, nullptr, "nvrtcGetNumSupportedArchs(PearlHash)");
      std::vector<int> supported_archs(static_cast<size_t>(arch_count));
      arch_status = rtc.get_supported_archs(supported_archs.data());
      if (arch_status != NVRTC_SUCCESS)
        throw rtc_error(rtc, arch_status, nullptr, "nvrtcGetSupportedArchs(PearlHash)");
      int target_arch = 0;
      for (const int arch : supported_archs)
        if (arch <= device_arch && arch > target_arch) target_arch = arch;
      if (target_arch < 80)
        throw std::string("NVRTC does not support PearlHash's required compute capability 8.0+");
      // A cubin is valid only for the exact architecture it names. When NVRTC knows the selected
      // GPU, emit final machine code locally and avoid a second driver-JIT pass; otherwise retain
      // forward-compatible PTX for newer GPUs (for example CUDA 12.6 on a Blackwell card).
      const bool emit_cubin = target_arch == device_arch;

      const std::string source(
#include "cuda/pearlhash_kernel.inc"
      );
      std::vector<std::string> option_storage{
        "--std=c++17",
        "--gpu-architecture=" + std::string(emit_cubin ? "sm_" : "compute_") +
          std::to_string(target_arch),
        "--include-path=" + cutlass_include.generic_string(),
        "--include-path=" + cuda_include.generic_string(),
        "--include-path=" + cccl_include.generic_string(),
        "--device-as-default-execution-space",
        "--use_fast_math"
      };
      std::vector<const char*> options;
      options.reserve(option_storage.size());
      for (const auto& option : option_storage) options.push_back(option.c_str());

      std::ostringstream key_stream;
      key_stream << source;
      for (const auto& option : option_storage) key_stream << '\0' << option;
      key_stream << "\0pearlhash-cuda-cache-v1";
      std::ostringstream filename;
      filename << "pearlhash-cuda-" << (emit_cubin ? "sm" : "compute") << target_arch << "-"
               << std::hex << std::setw(16) << std::setfill('0')
               << pearlhash_jit_hash(key_stream.str()) << (emit_cubin ? ".cubin" : ".ptx");
      const std::filesystem::path cache_dir = pearlhash_jit_cache_dir();
      const std::filesystem::path cache_path =
        cache_dir.empty() ? std::filesystem::path{} : cache_dir / filename.str();
      std::vector<char> code = cache_path.empty()
        ? std::vector<char>{} : pearlhash_read_cache(cache_path);
      if (!code.empty() && driver.module_load(&module, code.data()) != CUDA_SUCCESS) {
        std::error_code error;
        std::filesystem::remove(cache_path, error);
        code.clear();
      }
      if (code.empty()) {
        nvrtcProgram program = nullptr;
        nvrtcResult status = rtc.create_program(
          &program, source.c_str(), "pearlhash-cute.cu", 0, nullptr, nullptr);
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, program, "nvrtcCreateProgram(PearlHash)");
        status = rtc.compile_program(program, static_cast<int>(options.size()), options.data());
        if (status != NVRTC_SUCCESS) {
          const std::string message =
            rtc_error(rtc, status, program, "nvrtcCompileProgram(PearlHash)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        size_t code_size = 0;
        status = emit_cubin
          ? rtc.get_cubin_size(program, &code_size)
          : rtc.get_ptx_size(program, &code_size);
        if (status != NVRTC_SUCCESS) {
          const std::string message =
            rtc_error(rtc, status, program,
                      emit_cubin ? "nvrtcGetCUBINSize(PearlHash)" : "nvrtcGetPTXSize(PearlHash)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        code.resize(code_size);
        status = emit_cubin
          ? rtc.get_cubin(program, code.data())
          : rtc.get_ptx(program, code.data());
        if (status != NVRTC_SUCCESS) {
          const std::string message =
            rtc_error(rtc, status, program,
                      emit_cubin ? "nvrtcGetCUBIN(PearlHash)" : "nvrtcGetPTX(PearlHash)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        (void)rtc.destroy_program(&program);
        cuda_check(driver, driver.module_load(&module, code.data()),
                   "cuModuleLoadData(PearlHash)");
        pearlhash_write_cache(cache_path, code);
      }
      cuda_check(driver, driver.module_function(&gemm_kernel, module, "pearlhash_cuda_gemm"),
                 "cuModuleGetFunction(PearlHash GEMM)");
      cuda_check(driver, driver.module_function(&hash_kernel, module, "pearlhash_cuda_hash"),
                 "cuModuleGetFunction(PearlHash hash)");
      constexpr int shared_bytes = 98304;
      cuda_check(driver, driver.function_attribute(
        gemm_kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes),
        "cuFuncSetAttribute(PearlHash shared memory)");
      cuda_check(driver, driver.function_attribute(
        gemm_kernel, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100),
        "cuFuncSetAttribute(PearlHash carveout)");
      cuda_check(driver, driver.event_create(&started, CU_EVENT_DEFAULT),
                 "cuEventCreate(PearlHash start)");
      cuda_check(driver, driver.event_create(&gemm_done, CU_EVENT_DEFAULT),
                 "cuEventCreate(PearlHash GEMM done)");
      cuda_check(driver, driver.event_create(&done, CU_EVENT_DEFAULT),
                 "cuEventCreate(PearlHash done)");
      enabled = true;
    } catch (const std::string& error) {
      release();
      std::fprintf(stderr, "PearlHash CUDA source-JIT unavailable, using the best SYCL kernel: %s\n",
                   error.c_str());
    } catch (const std::exception& error) {
      release();
      std::fprintf(stderr, "PearlHash CUDA source-JIT unavailable, using the best SYCL kernel: %s\n",
                   error.what());
    }
    return enabled;
  }

  void fail(const std::string& error) noexcept {
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
    if (api.available() && context) {
      (void)api.set_current(context);
      (void)api.synchronize();
    }
    release();
    checked = true;
    std::fprintf(stderr, "PearlHash CUDA source-JIT failed, using the best SYCL kernel: %s\n",
                 error.c_str());
  }

  void launch(const Buffers& b, uint32_t seed, int m, int n, bool debug) {
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
    cuda_check(api, api.set_current(context), "cuCtxSetCurrent(PearlHash launch)");
    void* a = b.Ap;
    void* bp = b.Bp;
    void* transcript = b.transcript;
    void* key = b.cA;
    void* target = b.target;
    void* result = b.result;
    int k = 4096;
    int debug_int = debug ? 1 : 0;
    void* gemm_args[] = {&a, &bp, &m, &n, &k, &transcript};
    void* hash_args[] = {&transcript, &key, &target, &result, &seed,
                         &m, &n, &debug_int};
    constexpr unsigned threads = 256;
    constexpr unsigned shared_bytes = 98304;
    cuda_check(api, api.event_record(started, nullptr),
               "cuEventRecord(PearlHash start)");
    cuda_check(api, api.launch(gemm_kernel, static_cast<unsigned>(m / 128),
                               static_cast<unsigned>(n / 256), 1,
                               threads, 1, 1, shared_bytes, nullptr,
                               gemm_args, nullptr),
               "cuLaunchKernel(PearlHash GEMM)");
    cuda_check(api, api.event_record(gemm_done, nullptr),
               "cuEventRecord(PearlHash GEMM done)");
    const uint64_t tiles = static_cast<uint64_t>(m / 16) *
                           static_cast<uint64_t>(n / 16);
    cuda_check(api, api.launch(hash_kernel,
                               static_cast<unsigned>((tiles + threads - 1) / threads),
                               1, 1, threads, 1, 1, 0, nullptr,
                               hash_args, nullptr),
               "cuLaunchKernel(PearlHash hash)");
    cuda_check(api, api.event_record(done, nullptr), "cuEventRecord(PearlHash done)");
  }

  void wait(double learned_us) {
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
#if !defined(_WIN32)
    const auto poll = learned_us > 20000.0 ? std::chrono::microseconds(500)
                                           : std::chrono::microseconds(100);
#else
    (void)learned_us;
#endif
    for (;;) {
      const CUresult status = api.event_query(done);
      if (status == CUDA_SUCCESS) {
        if (std::getenv("MOM_PEARLHASH_STATS") && stats_printed++ < 4) {
          float gemm_ms = 0.0f, hash_ms = 0.0f;
          cuda_check(api, api.event_elapsed(&gemm_ms, started, gemm_done),
                     "cuEventElapsedTime(PearlHash CUDA GEMM)");
          cuda_check(api, api.event_elapsed(&hash_ms, gemm_done, done),
                     "cuEventElapsedTime(PearlHash CUDA hash)");
          std::fprintf(stderr, "PEARLHASH_CUDA_STATS gemm=%.3fms hash=%.3fms\n",
                       gemm_ms, hash_ms);
        }
        return;
      }
      if (status != CUDA_ERROR_NOT_READY)
        cuda_check(api, status, "cuEventQuery(PearlHash CUDA)");
#if defined(_WIN32)
      mom_sycl_poll_pause();
#else
      std::this_thread::sleep_for(poll);
#endif
    }
  }
  ~PearlHashCudaSearch() { release(); }
};
#else
struct PearlHashCudaSearch {
  static bool is_cuda(const sycl::device&) { return false; }
  static bool supports(sycl::queue&, std::string* const reason = nullptr) {
    if (reason) *reason = "CUDA source-JIT support is not part of this worker";
    return false;
  }
  bool ensure(sycl::queue&, const Buffers&, int, int, int, int) { return false; }
  void launch(const Buffers&, uint32_t, int, int, bool) {}
  void wait(double) {}
  void release() noexcept {}
  void fail(const std::string&) noexcept {}
};
#endif

struct PearlHashHipSearch {
#if defined(MOM_SYCL_HAS_HIP)
  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
  hipFunction_t hash_kernel = nullptr;
  hipEvent_t started = nullptr;
  hipEvent_t wmma_done = nullptr;
  hipEvent_t done = nullptr;
  uint32_t* transcript = nullptr;
  size_t transcript_bytes = 0;
  bool checked = false;
  bool enabled = false;
  unsigned stats_printed = 0;
  int tile_er = 4;
  int tile_ec = 4;
  int k_unroll = 2;
  int compiled_k = 0;
  int compiled_rank = 0;
  unsigned wmma_threads = 256;

  static void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess) throw std::string(operation) + ": " + hipGetErrorString(status);
  }
  static std::string rtc_error(PearlHashHiprtcApi& api, hiprtcResult status,
                               hiprtcProgram program, const char* operation) {
    size_t size = 0;
    std::string log;
    if (program && api.get_log_size(program, &size) == HIPRTC_SUCCESS && size) {
      log.resize(size);
      (void)api.get_log(program, log.data());
    }
    return std::string(operation) + ": " + api.error_string(status) +
           (log.empty() ? "" : "\n" + log);
  }
  static bool is_hip(const sycl::device& device) {
  #if defined(MOM_SYCL_ADAPTIVECPP)
    return device.get_backend() == sycl::backend::hip;
  #else
    return device.get_backend() == sycl::backend::ext_oneapi_hip;
  #endif
  }
  static bool supports(sycl::queue& q, std::string* const reason = nullptr) {
    const auto unavailable = [&](const std::string& message) {
      if (reason) *reason = message;
      return false;
    };
    if (!is_hip(q.get_device())) return unavailable("the selected SYCL device is not HIP");
    int device = 0;
    hipDeviceProp_t properties{};
    const hipError_t device_status = hipGetDevice(&device);
    if (device_status != hipSuccess)
      return unavailable(std::string("hipGetDevice: ") + hipGetErrorString(device_status));
    const hipError_t properties_status = hipGetDeviceProperties(&properties, device);
    if (properties_status != hipSuccess)
      return unavailable(std::string("hipGetDeviceProperties: ") +
                         hipGetErrorString(properties_status));
    if (!std::strncmp(properties.gcnArchName, "gfx1200", 7) ||
        !std::strncmp(properties.gcnArchName, "gfx1201", 7)) return true;
    return unavailable(std::string("gfx12 WMMA is unavailable on ") +
                       (properties.gcnArchName[0] ? properties.gcnArchName : "an unknown architecture"));
  }
  void release() noexcept {
    if (done) (void)hipEventDestroy(done);
    if (wmma_done) (void)hipEventDestroy(wmma_done);
    if (started) (void)hipEventDestroy(started);
    if (transcript) (void)hipFree(transcript);
    if (module) (void)hipModuleUnload(module);
    done = nullptr;
    wmma_done = nullptr;
    started = nullptr;
    transcript = nullptr;
    transcript_bytes = 0;
    module = nullptr;
    kernel = nullptr;
    hash_kernel = nullptr;
    compiled_k = 0;
    compiled_rank = 0;
    enabled = false;
  }
  bool ensure(sycl::queue& q, int m, int n, int k, int rank) {
    if (!supports(q)) return false;
    // The fixed wave tile and cumulative transcript require these divisibilities. All shipped
    // PearlHash shapes satisfy them; retaining the check keeps custom research shapes safe.
    const int rows_per_wave = 16 * tile_er, cols_per_wave = 16 * tile_ec;
    const uint64_t waves = m > 0 && n > 0 ? (uint64_t)(m / rows_per_wave) * (uint64_t)(n / cols_per_wave) : 0;
    if (m % rows_per_wave || n % cols_per_wave || k % 16 || rank % 16 || k % rank ||
        k / rank != 16 || waves % (wmma_threads / 32)) {
      return false;
    }
    if (checked && (compiled_k != k || compiled_rank != rank)) {
      return false;
    }
    if (!checked) {
      try {
        checked = true;
        int device = 0;
        hipDeviceProp_t properties{};
        hip_check(hipGetDevice(&device), "hipGetDevice(PearlHash)");
        hip_check(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties(PearlHash)");
        std::string arch = properties.gcnArchName;
        if (const size_t colon = arch.find(':'); colon != std::string::npos) arch.resize(colon);
        if (arch != "gfx1200" && arch != "gfx1201") {
          return false;
        }

        const std::string source(
#include "hip/pearlhash_kernel.inc"
        );
        const std::string arch_option = "--offload-arch=" + arch;
        const std::string er_option = "-DPEARLHASH_ER=" + std::to_string(tile_er);
        const std::string ec_option = "-DPEARLHASH_EC=" + std::to_string(tile_ec);
        const std::string k_option = "-DPEARLHASH_K=" + std::to_string(k);
        const std::string rank_option = "-DPEARLHASH_RANK=" + std::to_string(rank);
        const std::string unroll_option = "-DPEARLHASH_K_UNROLL=" + std::to_string(k_unroll);
        const char* options[] = {"-O3", "--std=c++17", arch_option.c_str(), er_option.c_str(),
                                 ec_option.c_str(), k_option.c_str(), rank_option.c_str(),
                                 unroll_option.c_str()};
        std::ostringstream key_stream;
        key_stream << source << '\0' << arch;
        for (const char* option : options) key_stream << '\0' << option;
        key_stream << "\0pearlhash-hip-cache-v1";
        std::ostringstream filename;
        filename << "pearlhash-" << arch << '-' << std::hex << std::setw(16)
                 << std::setfill('0') << pearlhash_jit_hash(key_stream.str()) << ".hsaco";
        const std::filesystem::path cache_dir = pearlhash_jit_cache_dir();
        const std::filesystem::path cache_path =
          cache_dir.empty() ? std::filesystem::path{} : cache_dir / filename.str();
        std::vector<char> code = cache_path.empty()
          ? std::vector<char>{} : pearlhash_read_cache(cache_path);
        if (!code.empty() && hipModuleLoadData(&module, code.data()) != hipSuccess) {
          std::error_code error;
          std::filesystem::remove(cache_path, error);
          code.clear();
        }
        if (code.empty()) {
          PearlHashHiprtcApi& rtc = PearlHashHiprtcApi::instance();
          if (!rtc.available()) throw rtc.error;
          hiprtcProgram program = nullptr;
          hiprtcResult status = rtc.create_program(
            &program, source.c_str(), "pearlhash-wmma.hip", 0, nullptr, nullptr);
          if (status != HIPRTC_SUCCESS)
            throw rtc_error(rtc, status, program, "hiprtcCreateProgram(PearlHash)");
          status = rtc.compile_program(program, 8, options);
          if (status != HIPRTC_SUCCESS) {
            const std::string message =
              rtc_error(rtc, status, program, "hiprtcCompileProgram(PearlHash)");
            (void)rtc.destroy_program(&program);
            throw message;
          }
          size_t code_size = 0;
          status = rtc.get_code_size(program, &code_size);
          if (status != HIPRTC_SUCCESS) {
            const std::string message =
              rtc_error(rtc, status, program, "hiprtcGetCodeSize(PearlHash)");
            (void)rtc.destroy_program(&program);
            throw message;
          }
          code.resize(code_size);
          status = rtc.get_code(program, code.data());
          if (status != HIPRTC_SUCCESS) {
            const std::string message =
              rtc_error(rtc, status, program, "hiprtcGetCode(PearlHash)");
            (void)rtc.destroy_program(&program);
            throw message;
          }
          (void)rtc.destroy_program(&program);
          hip_check(hipModuleLoadData(&module, code.data()), "hipModuleLoadData(PearlHash)");
          pearlhash_write_cache(cache_path, code);
        }
        hip_check(hipModuleGetFunction(&kernel, module, "pearlhash_wmma_transcript"), "hipModuleGetFunction(PearlHash WMMA)");
        hip_check(hipModuleGetFunction(&hash_kernel, module, "pearlhash_hash_search"), "hipModuleGetFunction(PearlHash hash)");
        hip_check(hipEventCreate(&started), "hipEventCreate(PearlHash start)");
        hip_check(hipEventCreate(&wmma_done), "hipEventCreate(PearlHash WMMA done)");
        hip_check(hipEventCreate(&done), "hipEventCreate(PearlHash done)");
        compiled_k = k;
        compiled_rank = rank;
        enabled = true;
      } catch (const std::string& error) {
        release();
        std::fprintf(stderr, "PearlHash gfx12 WMMA unavailable, using portable dp4a: %s\n", error.c_str());
        return false;
      }
    }
    if (!enabled) return false;
    const size_t bytes = (size_t)(m / 16) * (size_t)(n / 16) * 16u * sizeof(uint32_t);
    if (bytes != transcript_bytes) {
      try {
        if (transcript) hip_check(hipFree(transcript), "hipFree(PearlHash transcript)");
        transcript = nullptr;
        transcript_bytes = 0;
        hip_check(hipMalloc(reinterpret_cast<void**>(&transcript), bytes), "hipMalloc(PearlHash transcript)");
        transcript_bytes = bytes;
      } catch (const std::string& error) {
        release();
        std::fprintf(stderr, "PearlHash gfx12 transcript allocation unavailable, using portable dp4a: %s\n", error.c_str());
        return false;
      }
    }
    return true;
  }
  void fail(const std::string& error) noexcept {
    (void)hipDeviceSynchronize();
    release();
    checked = true;
    std::fprintf(stderr, "PearlHash HIP source-JIT failed, using the best SYCL kernel: %s\n",
                 error.c_str());
  }
  void launch(const Buffers& b, uint32_t seed, int m, int n, int k, int rank, bool debug) {
    const unsigned threads = wmma_threads;
    const unsigned waves_per_block = threads / 32;
    const uint64_t waves = (uint64_t)(m / (16 * tile_er)) * (uint64_t)(n / (16 * tile_ec));
    const unsigned blocks = (unsigned)((waves + waves_per_block - 1) / waves_per_block);
    void* A = b.ApWmma;
    void* B = b.BpWmma;
    void* transcript_arg = transcript;
    void* key = b.cA;
    void* target = b.target;
    void* result = b.result;
    int debug_int = debug ? 1 : 0;
    void* wmma_args[] = {&A, &B, &transcript_arg, &m, &n};
    hip_check(hipEventRecord(started, nullptr), "hipEventRecord(PearlHash start)");
    hip_check(hipModuleLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, wmma_args, nullptr),
              "hipModuleLaunchKernel(PearlHash WMMA)");
    hip_check(hipEventRecord(wmma_done, nullptr), "hipEventRecord(PearlHash WMMA done)");
    const uint64_t tiles = (uint64_t)(m / 16) * (uint64_t)(n / 16);
    constexpr unsigned hash_threads = 256;
    const unsigned hash_blocks = (unsigned)((tiles + hash_threads - 1) / hash_threads);
    void* hash_args[] = {&transcript_arg, &key, &target, &result, &seed, &m, &n, &debug_int};
    hip_check(hipModuleLaunchKernel(hash_kernel, hash_blocks, 1, 1, hash_threads, 1, 1, 0, nullptr,
                                    hash_args, nullptr),
              "hipModuleLaunchKernel(PearlHash hash)");
    hip_check(hipEventRecord(done, nullptr), "hipEventRecord(PearlHash)");
  }
  // hipEventSynchronize can spin a complete CPU core for the many-second production GEMM. Querying
  // at a sub-millisecond cadence is effectively zero-cost for hashrate and keeps the miner idle while
  // the GPU works. The caller first sleeps through 90% of its learned attempt duration.
  void wait(double learned_us) {
#if !defined(_WIN32)
    const auto poll = learned_us > 20000.0 ? std::chrono::microseconds(500)
                                           : std::chrono::microseconds(100);
#else
    (void)learned_us;
#endif
    for (;;) {
      const hipError_t status = hipEventQuery(done);
      if (status == hipSuccess) {
        if (std::getenv("MOM_PEARLHASH_STATS") && stats_printed++ < 4) {
          float wmma_ms = 0.0f, hash_ms = 0.0f;
          hip_check(hipEventElapsedTime(&wmma_ms, started, wmma_done), "hipEventElapsedTime(PearlHash WMMA)");
          hip_check(hipEventElapsedTime(&hash_ms, wmma_done, done), "hipEventElapsedTime(PearlHash hash)");
          std::fprintf(stderr, "PEARLHASH_HIP_STATS wmma=%.3fms hash=%.3fms\n", wmma_ms, hash_ms);
        }
        return;
      }
      if (status != hipErrorNotReady) hip_check(status, "hipEventQuery(PearlHash)");
#if defined(_WIN32)
      mom_sycl_poll_pause();
#else
      std::this_thread::sleep_for(poll);
#endif
    }
  }
  ~PearlHashHipSearch() { release(); }
#else
  static bool supports(sycl::queue&, std::string* const reason = nullptr) {
    if (reason) *reason = "HIP source-JIT support is not part of this worker";
    return false;
  }
  bool ensure(sycl::queue&, int, int, int, int) { return false; }
  void wait(double) {}
  void release() noexcept {}
  void fail(const std::string&) noexcept {}
#endif
};


struct PearlHashAmdWmmaSearch {
#if defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
  static bool disabled() {
    const char* value = std::getenv("MOM_PEARLHASH_AMD_WMMA");
    return value && (!std::strcmp(value, "0") || !std::strcmp(value, "false"));
  }

  static bool is_gfx12() {
    int device = 0;
    hipDeviceProp_t properties{};
    const hipError_t current = hipGetDevice(&device);
    const hipError_t info = current == hipSuccess
      ? hipGetDeviceProperties(&properties, device) : current;
    if (std::getenv("MOM_PEARLHASH_AMD_WMMA_LOG"))
      std::fprintf(stderr, "pearlhash: HIP device=%d current=%d properties=%d arch=%s\n",
                   device, static_cast<int>(current), static_cast<int>(info), properties.gcnArchName);
    return current == hipSuccess && info == hipSuccess &&
           !std::strncmp(properties.gcnArchName, "gfx12", 5);
  }

  bool ensure(sycl::queue& q, int m, int n, int k, int rank) {
    const bool is_hip = q.get_device().get_backend() == sycl::backend::ext_oneapi_hip;
    const bool off = disabled();
    const bool gfx12 = is_gfx12();
    if (std::getenv("MOM_PEARLHASH_AMD_WMMA_LOG"))
      std::fprintf(stderr, "pearlhash: DPC++ WMMA hip=%d disabled=%d gfx12=%d shape=%dx%d/%d/%d\n",
                   is_hip, off, gfx12, m, n, k, rank);
    if (!is_hip || off || !gfx12) return false;
    if ((k != 4096 || rank != 256) &&
        (k != 2048 || rank != 128) &&
        (k != 1024 || rank != 64)) return false;
    if (m % 64 || n % 32) return false;
    const uint64_t waves = static_cast<uint64_t>(m / 64) * static_cast<uint64_t>(n / 32);
    if (waves % 8) return false;  // eight wave32 sub-groups per 256-thread work-group
    return true;
  }

  void launch(sycl::queue& q, const Buffers& b, uint32_t seed, int m, int n,
              int k, int rank, bool debug) {
    if (k == 4096 && rank == 256)
      search_amd_wmma_t<4096, 256>(q, b, seed, m, n, debug);
    else if (k == 2048 && rank == 128)
      search_amd_wmma_t<2048, 128, 4, 4>(q, b, seed, m, n, debug);
    else
      search_amd_wmma_t<1024, 64>(q, b, seed, m, n, debug);
  }

  void release(sycl::queue&) noexcept {}
#else
  bool ensure(sycl::queue&, int, int, int, int) { return false; }
  void launch(sycl::queue&, const Buffers&, uint32_t, int, int, int, int, bool) {}
  void release(sycl::queue&) noexcept {}
#endif
};

enum class PearlHashSearchBackend { sycl, amd_sycl_wmma, hip_jit, cuda_jit };

static PearlHashSearchBackend attempt(
    sycl::queue& q, const Buffers& b, uint32_t seed, int m, int n, int k, int rank,
    const std::string& backend = "auto", PearlHashHipSearch* hip = nullptr,
    PearlHashCudaSearch* cuda = nullptr, PearlHashAmdWmmaSearch* amd_wmma = nullptr,
    sycl::queue* hip_prep_queue = nullptr, bool validate_native = false) {
  static thread_local unsigned stats_attempts = 0;
  const bool stats = std::getenv("MOM_PEARLHASH_STATS") && stats_attempts++ < 3;
  auto stats_last = std::chrono::steady_clock::now();
  auto mark = [&](const char* stage) {
    if (!stats) return;
    q.wait_and_throw();
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - stats_last).count();
    std::fprintf(stderr, "PEARLHASH_SYCL_STATS %s=%.3fms\n", stage, ms);
    stats_last = now;
  };
  k_roots(q, b, seed, m, n, k);          // commitment roots cA/cB (A/Bt regenerated from RNG)
  mark("roots");
  k_noise(q, b, m, n, k, rank);          // sparse low-rank noise E_AL/E_AR, E_BL/E_BR
  mark("noise");

  // Pick the search kernel by the running device's backend BEFORE laying out A'/B': compute_ab needs to
  // know whether to write the portable row-major A' / column-major B' (search()) or the ESIMD tile-major
  // layout. CUDA -> search_cuda (mma.sync, nvptx). Any OpenCL device (AMD GPU, the CPU device, and
  // Intel OpenCL) -> the portable dp4a search() -- it is the single matrix-hardware-free path, so
  // the same image runs on AMD and the CPU and Intel OpenCL behaves like another OpenCL GPU. (Set
  // MOM_PEARLHASH_ESIMD to opt an Intel-OpenCL *GPU* back into the faster ESIMD path.) Level-Zero / default
  // Intel -> ESIMD. search_cuda/search_esimd/search() are each compiled only in the builds that can run
  // them, so the references below are guarded to match.
#if defined(MOM_SYCL_HAS_CUDA)
  #if defined(MOM_SYCL_ADAPTIVECPP)
  const bool is_cuda = q.get_device().get_backend() == sycl::backend::cuda;
  #else
  const bool is_cuda = q.get_device().get_backend() == sycl::backend::ext_oneapi_cuda;
  #endif
#else
  const bool is_cuda = false;
#endif
  const bool dbg = validate_native || std::getenv("MOM_PEARLHASH_CHK") != nullptr;
  const bool tuned_sycl = backend == "auto" || backend == "sycl-native" ||
                          backend == "native";
  // The portable search() is compiled into every build that has attempt() (i.e. not the ESIMD-only TU),
  // so it is always callable here -- including the Windows/dpcpp Intel build, which has search_esimd in
  // the SAME TU. Route any OpenCL device (AMD GPU, the CPU device, Intel OpenCL) to it; an
  // Intel-OpenCL GPU can opt back into ESIMD with MOM_PEARLHASH_ESIMD. CUDA -> search_cuda; Level-Zero -> ESIMD.
  bool use_portable = !tuned_sycl;
#if defined(PEARLHASH_ESIMD) || defined(MOM_PEARLHASH_HAS_ESIMD)
#if defined(MOM_SYCL_ADAPTIVECPP)
  if (tuned_sycl && !is_cuda && q.get_device().get_backend() == sycl::backend::ocl) {
#else
  if (tuned_sycl && !is_cuda && q.get_device().get_backend() == sycl::backend::opencl) {
#endif
    const bool is_gpu = q.get_device().is_gpu();
    use_portable = !(is_gpu && std::getenv("MOM_PEARLHASH_ESIMD"));
  }
#else
  use_portable = !tuned_sycl || !is_cuda;
#endif

#if defined(MOM_SYCL_HAS_HIP)
  if ((backend == "auto" || backend == "native") && hip && hip->ensure(q, m, n, k, rank)) {
    try {
      q.wait_and_throw(); // both independent prep streams consume the completed noise tables
      sycl::queue& prep = hip_prep_queue ? *hip_prep_queue : q;
      PearlHashPrepEvents ready = compute_ab_amd_wmma(q, prep, b, seed, m, n, k, rank);
      while (ready.a.get_info<sycl::info::event::command_execution_status>() !=
               sycl::info::event_command_status::complete ||
             ready.b.get_info<sycl::info::event::command_execution_status>() !=
               sycl::info::event_command_status::complete)
#if defined(_WIN32)
        mom_sycl_poll_pause();
#else
        std::this_thread::sleep_for(std::chrono::microseconds(100));
#endif
      ready.a.wait_and_throw();
      ready.b.wait_and_throw();
      mark("ab");
      hip->launch(b, seed, m, n, k, rank, dbg);
      return PearlHashSearchBackend::hip_jit;
    } catch (const std::string& error) {
      hip->fail(error);
      b.result->found = 0;
      b.result->chk = 0;
    } catch (const std::exception& error) {
      hip->fail(error.what());
      b.result->found = 0;
      b.result->chk = 0;
    }
  }
  // DPC++ HIP can lower its gfx12 builtin directly. Shipped AdaptiveCpp workers omit this
  // experimental branch and fall through to portable SYCL.
  if (tuned_sycl && amd_wmma && amd_wmma->ensure(q, m, n, k, rank)) {
    compute_ab(q, b, seed, m, n, k, rank, /*portable=*/true);
    mark("ab");
    amd_wmma->launch(q, b, seed, m, n, k, rank, dbg);
    return PearlHashSearchBackend::amd_sycl_wmma;
  }
#endif
#if defined(MOM_SYCL_HAS_CUDA)
  if ((backend == "auto" || backend == "native") && is_cuda && cuda &&
      cuda->ensure(q, b, m, n, k, rank)) {
    try {
      // CUTE consumes the common row-major A / column-major B representation. The preparation
      // remains the shared SYCL kernel; only the matrix mainloop and transcript hash are native.
      compute_ab(q, b, seed, m, n, k, rank, /*portable=*/true);
      q.wait_and_throw();
      mark("ab");
      cuda->launch(b, seed, m, n, dbg);
      return PearlHashSearchBackend::cuda_jit;
    } catch (const std::string& error) {
      cuda->fail(error);
      b.result->found = 0;
      b.result->chk = 0;
    } catch (const std::exception& error) {
      cuda->fail(error.what());
      b.result->found = 0;
      b.result->chk = 0;
    }
  }
#endif
  if (q.get_device().is_cpu()) {
    q.wait_and_throw();
    compute_ab_cpu(b, seed, m, n, k, rank);
  } else {
    compute_ab(q, b, seed, m, n, k, rank, /*portable=*/use_portable);
  }
  mark("ab");
#if defined(MOM_SYCL_HAS_CUDA)
  if (is_cuda && tuned_sycl) {
    search_cuda(q, b, seed, m, n, k, rank);
    mark("search_cuda");
    return PearlHashSearchBackend::sycl;
  }
#endif
#ifndef MOM_PEARLHASH_ESIMD_TU   // search() is not compiled into the ESIMD-only TU (which only exports search_esimd)
  if (use_portable) {
    if (q.get_device().is_cpu())
      search_cpu(q, b, seed, m, n, k, rank, dbg);
    else
      search(q, b, seed, m, n, k, rank, dbg);
    mark("search_portable");
    return PearlHashSearchBackend::sycl;
  }   // portable dp4a int8 GEMM (AMD / CPU / Intel-OpenCL)
#endif
#if defined(PEARLHASH_ESIMD) || defined(MOM_PEARLHASH_HAS_ESIMD)
  search_esimd(q, b, seed, m, n, k, rank, dbg); // ESIMD register-resident DPAS path (Intel GPUs, Level-Zero, ~53 TH/s)
  mark("search_esimd");
#else
  search(q, b, seed, m, n, k, rank, dbg);       // unreached (use_portable already handled non-CUDA above)
  mark("search_portable");
#endif
  return PearlHashSearchBackend::sycl;
}

}  // namespace mom_pearlhash

// The standalone spir64-only ESIMD TU (pearlhash_esimd.cpp) only needs the device kernels inside the
// namespace above; the host-side entry point + PlainProof builder below belong to the main TU only.
#ifndef MOM_PEARLHASH_ESIMD_TU
using namespace mom_pearlhash;

// 52-byte NoisyGEMM config block hashed with the 76-byte header into the per-job key (k, rank, and
// the two sparsity bytes at [9]/[15]); bytes 0..3 = k LE, 4..5 = rank LE.
static void config_bytes(uint8_t cfg[52], int k, int rank) {
  memset(cfg, 0, 52);
  cfg[0]=k&0xff; cfg[1]=(k>>8)&0xff; cfg[2]=(k>>16)&0xff; cfg[3]=(k>>24)&0xff; cfg[4]=rank&0xff; cfg[5]=(rank>>8)&0xff; cfg[9]=15; cfg[15]=15;
}
// Per-job key = unkeyed BLAKE3 of the 76-byte header concatenated with the 52-byte config block.
static void derive_key(const uint8_t* header76, int k, int rank, uint8_t key32[32]) {
  uint8_t cfg[52], kb[128]; config_bytes(cfg, k, rank);
  memcpy(kb, header76, 76); memcpy(kb + 76, cfg, 52);
  pearlhash_b3::b3(kb, 128, nullptr, key32);
}

// ---- PlainProof construction on the host (the pool submission for a winning tile) ----
// On a winning tile the server builds the pool submission itself: it regenerates the A / Bt byte
// streams from the counter-RNG, builds the keyed-BLAKE3 Merkle tree, opens the 16 revealed rows
// (A) and cols (Bt), bincode-serializes and base64-encodes. Mirrors pearl-blake3 merkle.rs and is
// byte-identical to the old JS path (validated against verify_plain_proof_v2). Runs once per found
// share (rare), so plain host code -- no GPU. CHUNK=1024; leaves use non-root chunk_cv, the root
// uses merge_root (ROOT at top); for power-of-two leaf counts the pairwise odd-carry layer tree
// (siblings) is identical to merge_root's tree.
using CV = std::array<uint8_t, 32>;
// Regenerate leaf `leaf` (1024 bytes, zero-padded) of A ('A') or Bt ('B') from the seed RNG.
static int gen_leaf(char which, uint32_t seed, int leaf, int m, int n, int k, uint8_t out[1024]) {
  const uint32_t tot = (uint32_t)(m * k);
  const int total = (which == 'A' ? m : n) * k, off = leaf * 1024;
  int len = total - off < 1024 ? total - off : 1024;
  for (int t = 0; t < len; t++) {
    int b = off + t;
    out[t] = which == 'A' ? (uint8_t)gv(seed, (uint32_t)b)
                          : (uint8_t)gv(seed, tot + (uint32_t)((b % k) * n + (b / k)));
  }
  for (int t = len; t < 1024; t++) out[t] = 0;
  return len;
}
static std::vector<std::vector<CV>> build_layers(char which, uint32_t seed, int numLeaves, int m, int n, int k, const uint8_t* key) {
  std::vector<CV> l0(numLeaves); uint8_t chunk[1024];
  for (int i = 0; i < numLeaves; i++) { int len = gen_leaf(which, seed, i, m, n, k, chunk); pearlhash_b3::chunk_cv(chunk, (uint32_t)len, (uint64_t)i, key, l0[i].data()); }
  std::vector<std::vector<CV>> layers; layers.push_back(std::move(l0));
  while (layers.back().size() > 2) {
    auto& prev = layers.back(); std::vector<CV> next;
    for (size_t i = 0; i < prev.size(); i += 2) {
      if (i + 1 < prev.size()) { CV p; pearlhash_b3::parent_cv(prev[i].data(), prev[i + 1].data(), key, false, p.data()); next.push_back(p); }
      else next.push_back(prev[i]);
    }
    layers.push_back(std::move(next));
  }
  return layers;
}
static void put_u64le(std::vector<uint8_t>& v, uint64_t n) { for (int i = 0; i < 8; i++) v.push_back((uint8_t)(n >> (8 * i))); }
static void put_bytes(std::vector<uint8_t>& v, const uint8_t* p, int n) { v.insert(v.end(), p, p + n); }
// Append a serialized MatrixMerkleProof for the 16 contiguous `rows` of matrix `which` (cols wide).
static void ser_matrix_proof(std::vector<uint8_t>& out, char which, uint32_t seed, int m, int n, int k,
                             const uint8_t* key, int cols, const std::vector<int>& rows) {
  const int numLeaves = ((which == 'A' ? m : n) * k) / 1024;
  auto layers = build_layers(which, seed, numLeaves, m, n, k, key);
  uint8_t root[32]; { std::vector<CV> tmp = layers[0]; pearlhash_b3::merge_root((uint8_t*)tmp.data(), numLeaves, key, root); }
  std::set<int> leafset;
  for (int row : rows) { int first = (int)((long)row * cols / 1024), last = (int)(((long)(row + 1) * cols - 1) / 1024); for (int i = first; i <= last; i++) leafset.insert(i); }
  std::vector<int> leaves(leafset.begin(), leafset.end());
  // MerkleProof: leaf_data (Vec<[u8;1024]>), leaf_indices, total_leaves, root, siblings
  put_u64le(out, leaves.size());
  uint8_t chunk[1024];
  for (int li : leaves) { gen_leaf(which, seed, li, m, n, k, chunk); put_u64le(out, 1024); put_bytes(out, chunk, 1024); }
  put_u64le(out, leaves.size()); for (int li : leaves) put_u64le(out, (uint64_t)li);
  put_u64le(out, (uint64_t)numLeaves);
  put_bytes(out, root, 32);
  // siblings: walk up, emit each node whose sibling is not itself in the active set
  std::set<int> cur(leafset); int levelLen = numLeaves, level = 0; std::vector<CV> sib;
  while (levelLen > 1 && !cur.empty()) {
    auto& nodes = layers[level];
    for (int i : cur) { if (i % 2 == 1) { if (!cur.count(i - 1)) sib.push_back(nodes[i - 1]); }
                        else if (!cur.count(i + 1) && i + 1 < levelLen) sib.push_back(nodes[i + 1]); }
    std::set<int> nx; for (int i : cur) nx.insert(i >> 1); cur.swap(nx);
    levelLen = (levelLen + 1) / 2; level++;
  }
  put_u64le(out, sib.size()); for (auto& s : sib) put_bytes(out, s.data(), 32);
  // MatrixMerkleProof tail: row_indices
  put_u64le(out, rows.size()); for (int r : rows) put_u64le(out, (uint64_t)r);
}
static std::string base64(const std::vector<uint8_t>& d) {
  static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o; for (size_t i = 0; i < d.size(); i += 3) {
    uint32_t x = d[i] << 16 | (i + 1 < d.size() ? d[i + 1] << 8 : 0) | (i + 2 < d.size() ? d[i + 2] : 0);
    o += T[(x >> 18) & 63]; o += T[(x >> 12) & 63];
    o += (i + 1 < d.size()) ? T[(x >> 6) & 63] : '='; o += (i + 2 < d.size()) ? T[x & 63] : '=';
  }
  return o;
}
// Full PlainProof (base64) for a winning tile, from the seed + job key (no noised matrices needed).
static std::string build_plain_proof(uint32_t seed, int m, int n, int k, int rank, const uint8_t* key, int row, int col) {
  std::vector<int> aRows, btRows; for (int r = 0; r < 16; r++) { aRows.push_back(row + r); btRows.push_back(col + r); }
  std::vector<uint8_t> buf;
  put_u64le(buf, (uint64_t)m); put_u64le(buf, (uint64_t)n); put_u64le(buf, (uint64_t)k); put_u64le(buf, (uint64_t)rank);
  ser_matrix_proof(buf, 'A', seed, m, n, k, key, k, aRows);
  ser_matrix_proof(buf, 'B', seed, m, n, k, key, k, btRows);
  buf.push_back(0x00);   // moe: Option::None
  return base64(buf);
}
template <typename T>
static T* alloc_buffer(sycl::queue& q, size_t count) {
  return q.get_device().is_cpu() ? sycl::malloc_shared<T>(count, q)
                                 : sycl::malloc_device<T>(count, q);
}
static Buffers alloc_buffers(sycl::queue& q, int m, int n, int k, int rank,
                             bool hip_jit = false, bool cuda_jit = false) {
  Buffers b{};
  b.EAL=alloc_buffer<int8_t>(q,m*rank); b.EBR=alloc_buffer<int8_t>(q,rank*n);
  b.EBRt=alloc_buffer<int8_t>(q,n*rank);
  b.Ap=alloc_buffer<int8_t>(q,m*k); b.Bp=alloc_buffer<int8_t>(q,k*n);
#if defined(MOM_SYCL_HAS_HIP)
  // HIPRTC WMMA consumes an architecture-native fragment layout. Keep it separate from the
  // portable SYCL layout so a failed native launch can immediately fall back without regenerating.
  if (hip_jit) {
    b.ApWmma=alloc_buffer<int8_t>(q,m*k);
    b.BpWmma=alloc_buffer<int8_t>(q,k*n);
  }
#endif
  b.EARp1=alloc_buffer<int32_t>(q,k); b.EARp2=alloc_buffer<int32_t>(q,k);
  b.EBLq1=alloc_buffer<int32_t>(q,k); b.EBLq2=alloc_buffer<int32_t>(q,k);
  b.cA=alloc_buffer<uint8_t>(q,32); b.cB=alloc_buffer<uint8_t>(q,32);
  b.key=alloc_buffer<uint8_t>(q,32); b.target=alloc_buffer<uint8_t>(q,32);
  b.CVA=alloc_buffer<uint8_t>(q,(2*((m*k+1023)/1024)+2)*32);
  b.CVB=alloc_buffer<uint8_t>(q,(2*((n*k+1023)/1024)+2)*32);
  // Separating jackpot BLAKE3 from the matrix loop restores occupancy. Portable paths retain a
  // conservative 256 MiB cap; the explicitly selected CUDA source-JIT path can use a larger
  // transcript to amortize the common seed/root/noise preparation without affecting other workers.
  const uint64_t transcript_bytes = static_cast<uint64_t>(m / 16) * (n / 16) * 16U * sizeof(uint32_t);
  const uint64_t transcript_limit = cuda_jit ? 4ULL * 1024U * 1024U * 1024U
                                             : 256ULL * 1024U * 1024U;
  if (transcript_bytes <= transcript_limit)
    b.transcript=alloc_buffer<uint32_t>(q, static_cast<size_t>(transcript_bytes / sizeof(uint32_t)));
  b.result=sycl::malloc_shared<Result>(1,q); b.result->found=0;
  return b;
}

#ifndef PEARLHASH_STANDALONE
// ---- native mom entry point (DEV::PEARLHASH_GPU) ----
namespace {
// PlainProof carries M/N/K/rank, so the job can select a device-efficient profile without changing
// proof semantics. mom.js obtains defaults from GPU-COMPILERS.md and passes explicit config values.
// One persistent in-order queue + device buffer set per GPU; the seed search reuses them.
struct PearlHashState {
  sycl::queue queue;
  sycl::queue prep_queue;
  PearlHashHipSearch hip;
  PearlHashCudaSearch cuda;
  PearlHashAmdWmmaSearch amd_wmma;
  int m = 0, n = 0, k = 0, rank = 0;
  bool hip_jit_buffers = false;
  bool cuda_jit_buffers = false;
  Buffers b{};
  uint8_t key[32] = {0}, header[76] = {0};
  bool have_header = false;
  bool native_notice = false;
  double wait_ema_us = 0.0; // EMA of recent attempt wall times (us); paces the pre-wait sleep below
  std::mutex mutex;
  explicit PearlHashState(const std::string& dev_str)
    : queue(get_dev(dev_str), sycl::property::queue::in_order{}),
      prep_queue(queue.get_context(), queue.get_device(),
                 sycl::property::queue::in_order{}) {}
  void free_buffers() {
    // Free every non-null member independently (sycl::free is no-op'd by the per-element `if (p)`),
    // so a partial-allocation failure where Ap is null but earlier members (EAL/EBR/EBRt/...) are
    // not still releases them instead of leaking. No early return on !b.Ap.
    queue.wait();
    prep_queue.wait();
    hip.release();
    cuda.release();
    for (void* p : {(void*)b.EAL,(void*)b.EBR,(void*)b.EBRt,(void*)b.Ap,(void*)b.Bp,
                    (void*)b.ApWmma,(void*)b.BpWmma,
                    (void*)b.EARp1,(void*)b.EARp2,(void*)b.EBLq1,(void*)b.EBLq2,
                    (void*)b.cA,(void*)b.cB,(void*)b.key,(void*)b.target,
                    (void*)b.CVA,(void*)b.CVB,(void*)b.transcript,(void*)b.result})
      if (p) sycl::free(p, queue);
    b = Buffers{};
  }
  void ensure(int m_, int n_, int k_, int rank_, bool hip_jit_, bool cuda_jit_) {
    if (b.Ap && m == m_ && n == n_ && k == k_ && rank == rank_ &&
        hip_jit_buffers == hip_jit_ && cuda_jit_buffers == cuda_jit_) return;
    free_buffers();
    m = m_; n = n_; k = k_; rank = rank_;
    hip_jit_buffers = hip_jit_;
    cuda_jit_buffers = cuda_jit_;
    b = alloc_buffers(queue, m, n, k, rank, hip_jit_buffers, cuda_jit_buffers);
    have_header = false;
  }
  ~PearlHashState() { sycl_cleanup_noexcept("pearlhash", [&] { free_buffers(); }); }
};
using PearlHashStateMap = std::map<std::string, std::unique_ptr<PearlHashState>>;
PearlHashStateMap& pearlhash_states() {
  static PearlHashStateMap* const states = new PearlHashStateMap;
  return *states;
}
std::mutex& pearlhash_states_mutex() {
  static std::mutex* const mutex = new std::mutex;
  return *mutex;
}
PearlHashState& pearlhash_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(pearlhash_states_mutex());
  auto& st = pearlhash_states()[dev_str];
  if (!st) st = std::make_unique<PearlHashState>(dev_str);
  return *st;
}
thread_local std::string g_pearlhash_proof;   // last built proof, fetched right after pearlhash() returns 1
// Winning tile captured by pearlhash() on a find; consumed by pearlhash_proof() to build the proof lazily.
thread_local struct PearlHashFound { uint32_t seed; int row, col, m, n, k, rank; uint8_t key[32]; bool valid; } g_pf{};
} // namespace

void pearlhash_cleanup_states() noexcept {
  try {
    std::lock_guard<std::mutex> lock(pearlhash_states_mutex());
    pearlhash_states().clear();
  } catch (...) {
    std::fprintf(stderr, "pearlhash: ordered SYCL cleanup failed\n");
  }
}

// One seeded NoisyGEMM attempt per call (seed = *pseed). Returns 1 on a winning tile, writes the
// winning seed back to *pseed, and stashes the base64 PlainProof for pearlhash_proof(). intensity = M.
int pearlhash(
  const unsigned job_ref, const uint32_t, const uint8_t* const input, const unsigned input_size,
  uint8_t* const, uint64_t* const pseed, const uint8_t* const target,
  const unsigned intensity, const bool is_test, const bool, const std::string& dev_str,
  const std::string& backend, const unsigned requested_n, const unsigned requested_k,
  const unsigned requested_rank
) {
  if (input_size < 76) throw std::string("Bad pearlhash input length (need 76-byte header)");
  const int k = static_cast<int>(requested_k);
  const int rank = static_cast<int>(requested_rank);
  // M arrives as the worker's explicit primary tuning value; N can be selected independently. Native HIP
  // tests use the smallest square that activates the 32x32 cache-block traversal, so the
  // native-vs-SYCL checksum comparison covers production tile mapping without making CPU tests
  // expensive. Other implementations retain the compact 256x256 vector.
  const int test_edge = backend == "native" ? 2048 : 256;
  int m = is_test ? test_edge : ((int)intensity >= 128 ? (int)intensity : 131072);
  const int n = is_test ? m : static_cast<int>(requested_n);

  PearlHashState& st = pearlhash_state(dev_str);
  std::lock_guard<std::mutex> lock(st.mutex);
  const bool native_requested = backend == "auto" || backend == "native";
  std::string hip_reason, cuda_reason;
  const bool hip_jit = native_requested &&
                       PearlHashHipSearch::supports(st.queue, &hip_reason);
  const bool cuda_jit = native_requested &&
                        PearlHashCudaSearch::supports(st.queue, &cuda_reason);
  if (native_requested && !hip_jit && !cuda_jit && !st.native_notice) {
    st.native_notice = true;
    std::fprintf(stderr, "PearlHash native backend unavailable, using SYCL: %s\n",
                 (PearlHashCudaSearch::is_cuda(st.queue.get_device()) ? cuda_reason : hip_reason).c_str());
  }
  st.ensure(m, n, k, rank, hip_jit, cuda_jit);
  sycl::queue& q = st.queue;
  Buffers& b = st.b;

  if (!st.have_header || std::memcmp(st.header, input, 76) != 0) {  // (re)derive key on header change
    std::memcpy(st.header, input, 76);
    derive_key(input, k, rank, st.key); q.memcpy(b.key, st.key, 32); st.have_header = true;
  }
  (void)job_ref;
  uint8_t tgtLE[32];
  if (is_test) std::memset(tgtLE, 0xff, 32);                         // test: the first tile wins
  else for (int i = 0; i < 32; i++) tgtLE[i] = target[31 - i];       // core gives BE; kernel wants LE
  q.memcpy(b.target, tgtLE, 32);
  b.result->found = 0;
  b.result->chk = 0;   // whole-search checksum (filled only when MOM_PEARLHASH_CHK is set; see tile_mix)
  q.wait();

  const uint32_t attempt_seed = static_cast<uint32_t>(*pseed);
  PearlHashSearchBackend search_backend =
    attempt(q, b, attempt_seed, m, n, k, rank, backend, &st.hip, &st.cuda, &st.amd_wmma,
            &st.prep_queue, is_test);
  // Start after submission: a first call can include synchronous device-image/JIT setup, and feeding
  // that one-off cost into the wait EMA makes later completed GPU attempts sleep needlessly.
  const auto attempt_start = std::chrono::steady_clock::now();
  // The in-order queue's wait busy-spins a host core on both backends (CUDA's native sync and the
  // Intel GPU wait), and pearlhash waits once per attempt -- a long, very stable GEMM. So sleep through
  // most of the attempt (an EMA of recent attempt wall times) before waiting, leaving only a short
  // spinning tail; the result is read from shared USM afterwards so completion stays exact. The 90%
  // sleep stays under the attempt time, adding no latency. Mirrors the cn/gpu pre-read sleep.
  if (st.wait_ema_us > 2000.0) {
    const double elapsed_us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - attempt_start).count());
    const double sleep_us = st.wait_ema_us * 0.90 - elapsed_us;
    if (sleep_us > 0.0)
      std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(sleep_us)));
  }
  if (search_backend == PearlHashSearchBackend::hip_jit ||
      search_backend == PearlHashSearchBackend::cuda_jit) {
    try {
      if (search_backend == PearlHashSearchBackend::hip_jit)
        st.hip.wait(st.wait_ema_us);
      else
        st.cuda.wait(st.wait_ema_us);
    } catch (const std::string& error) {
      if (search_backend == PearlHashSearchBackend::hip_jit) st.hip.fail(error);
      else st.cuda.fail(error);
      b.result->found = 0;
      b.result->chk = 0;
      search_backend = attempt(q, b, attempt_seed, m, n, k, rank, "sycl-native",
                               nullptr, nullptr, &st.amd_wmma, nullptr, is_test);
      q.wait_and_throw();
    }
  } else q.wait_and_throw();
#if defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_HAS_CUDA)
  if (is_test && search_backend != PearlHashSearchBackend::sycl) {
    const uint32_t native_checksum = b.result->chk;
    b.result->found = 0;
    b.result->chk = 0;
    compute_ab(q, b, attempt_seed, m, n, k, rank, /*portable=*/true);
    search(q, b, attempt_seed, m, n, k, rank, /*dbg=*/true);
    q.wait();
    if (b.result->chk != native_checksum) {
      std::ostringstream message;
      message << "PearlHash native/SYCL checksum mismatch (native=0x" << std::hex
              << native_checksum << ", SYCL=0x" << b.result->chk << ')';
      throw message.str();
    }
  }
#endif
  {
    const double us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - attempt_start).count());
    st.wait_ema_us = st.wait_ema_us == 0.0 ? us : st.wait_ema_us * 0.8 + us * 0.2;
  }
  if (std::getenv("MOM_PEARLHASH_CHK"))   // cross-validation: same A'/B' -> identical chk on every search path
    std::fprintf(stderr, "PEARLHASH_CHK dev=%s seed=%u chk=%08x\n", dev_str.c_str(), (uint32_t)*pseed, b.result->chk);
  if (!b.result->found) return 0;
  *pseed = b.result->seed;
  if (is_test) return 1;                                             // test: core only needs "ok", no proof
  // Capture the winning tile; DEFER the heavy host PlainProof rebuild (regenerate A/Bt + the BLAKE3
  // Merkle tree, ~tens of ms at m=n=16384) to pearlhash_proof(), which core calls ONCE per pool job_id.
  // Building it here on every find would starve the DPAS search: at a low pool difficulty a tile is
  // found on most attempts, yet the pool credits only one share per job. Deferring keeps the search
  // running flat out (~6x faster live).
  g_pf.seed = b.result->seed; g_pf.row = (int)b.result->row; g_pf.col = (int)b.result->col;
  g_pf.m = m; g_pf.n = n; g_pf.k = k; g_pf.rank = rank; std::memcpy(g_pf.key, st.key, 32); g_pf.valid = true;
  return 1;
}
// Builds the PlainProof for the last winning tile captured by pearlhash() (lazy: called once per job_id
// by core, off the search hot path). Returns "" if no tile has been found yet.
const char* pearlhash_proof() {
  if (g_pf.valid)
    g_pearlhash_proof = build_plain_proof(g_pf.seed, g_pf.m, g_pf.n, g_pf.k, g_pf.rank, g_pf.key, g_pf.row, g_pf.col);
  return g_pearlhash_proof.c_str();
}
// GEMM MACs per attempt = m*n*k, mirroring pearlhash()'s shape selection. This is the work unit
// the pearlhash hashrate is reported in, so the core's display matches the "TH/s" GEMM benchmark.
uint64_t pearlhash_attempt_hashes(unsigned intensity, unsigned n, unsigned k) {
  const uint64_t m = intensity >= 128 ? intensity : 131072;
  return m * static_cast<uint64_t>(n) * static_cast<uint64_t>(k);
}
#endif  // !PEARLHASH_STANDALONE

#ifdef PEARLHASH_STANDALONE
#include <cstdio>
#include <chrono>
#include <poll.h>
#include <unistd.h>
static void parse_hex(const char* hex, uint8_t* out, int nbytes) {
  for (int i = 0; i < nbytes; i++) { unsigned x; sscanf(hex + 2 * i, "%2x", &x); out[i] = (uint8_t)x; }
}

// Persistent search server: stdin lines `job <headerHex(152)> <targetHexLE(64)>` set the
// current work; the GPU searches incrementing seeds continuously and prints
// `found <seed> <plainProofBase64>` on hits. (Standalone test/bench harness only.)
static int run_server(int k, int rank, int m, int n) {
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  fprintf(stderr, "pearlhash-server dev: %s  m=%d n=%d k=%d rank=%d\n", q.get_device().get_info<sycl::info::device::name>().c_str(), m, n, k, rank);
  Buffers b = alloc_buffers(q, m, n, k, rank);

  bool haveJob=false; uint32_t seed=1; std::string inbuf; uint8_t curKey[32]={0};
  char rd[8192];
  auto poll_stdin=[&](int timeout_ms){
    pollfd pfd{0, POLLIN, 0};
    if (poll(&pfd,1,timeout_ms)>0 && (pfd.revents&POLLIN)) { int nr=read(0,rd,sizeof(rd)); if(nr<=0) return false; inbuf.append(rd,nr); }
    size_t nl;
    while ((nl=inbuf.find('\n'))!=std::string::npos) {
      std::string line=inbuf.substr(0,nl); inbuf.erase(0,nl+1);
      if (line.rfind("job ",0)==0) {
        char hh[160], tt[80]; if (sscanf(line.c_str(),"job %159s %79s",hh,tt)==2) {
          uint8_t header[76], tgt[32];
          parse_hex(hh,header,76); derive_key(header,k,rank,curKey); parse_hex(tt,tgt,32);
          q.memcpy(b.key,curKey,32); q.memcpy(b.target,tgt,32); q.wait();
          haveJob=true; seed=1; b.result->found=0;
        }
      } else if (line=="quit") return false;
    }
    return true;
  };
  if (!poll_stdin(-1)) return 0;  // block for first job
  while (true) {
    if (!poll_stdin(0)) break;
    if (!haveJob) { if(!poll_stdin(100)) break; continue; }
    attempt(q, b, seed, m, n, k, rank); q.wait();
    if (b.result->found) {     // build the pool-ready PlainProof here; JS only relays it
      std::string proof = build_plain_proof(b.result->seed, m, n, k, rank, curKey, (int)b.result->row, (int)b.result->col);
      printf("found %u %s\n", b.result->seed, proof.c_str()); fflush(stdout); b.result->found=0;
    }
    seed++;
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 6 && std::string(argv[1]) == "server")
    return run_server(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (std::string(argv[1]) == "proof") {   // proof <hdr> k rank m n seed row col -> base64 (host, no GPU)
    int k = atoi(argv[3]), rank = atoi(argv[4]), m = atoi(argv[5]), n = atoi(argv[6]);
    uint8_t header[76], key[32];
    parse_hex(argv[2], header, 76); derive_key(header, k, rank, key);
    std::printf("%s\n", build_plain_proof((uint32_t)strtoul(argv[7], nullptr, 10), m, n, k, rank, key, atoi(argv[8]), atoi(argv[9])).c_str());
    return 0;
  }
  const char* headerHex = argv[1];
  int k = atoi(argv[2]), rank = atoi(argv[3]), m = atoi(argv[4]), n = atoi(argv[5]);
  uint32_t seedBase = (uint32_t)strtoul(argv[6], nullptr, 10);
  int count = atoi(argv[7]);
  const char* tgtHexLE = argv[8];

  uint8_t header[76], key[32], target[32];
  parse_hex(headerHex, header, 76); derive_key(header, k, rank, key);
  parse_hex(tgtHexLE, target, 32);

  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  std::printf("dev: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  Buffers b = alloc_buffers(q, m, n, k, rank);
  q.memcpy(b.key, key, 32); q.memcpy(b.target, target, 32); q.wait();

  attempt(q, b, seedBase, m, n, k, rank); q.wait();           // warm up JIT
  bool warmHit = b.result->found; b.result->found = 0;

  auto t0 = std::chrono::steady_clock::now();
  int found = 0, done = 0; uint32_t startSeed = warmHit ? seedBase : seedBase + 1;
  for (int i = 0; i < count; i++, done++) { uint32_t seed = startSeed + i; attempt(q, b, seed, m, n, k, rank); q.wait();
    if (b.result->found) { std::printf("FOUND seed=%u row=%u col=%u (attempt %d)\n", b.result->seed, b.result->row, b.result->col, i); found = 1; break; } }
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  double aps = done / (secs > 0 ? secs : 1);
  double macPerAttempt = (double)m*n*k + (double)m*k + (double)k*n;
  std::printf("%s  %.3fs  %.1f attempts/s  ~%.4g MAC/s  (%.4g TH/s-equiv)\n",
              found ? "OK" : "no-hit", secs, aps, aps * macPerAttempt, aps * macPerAttempt / 1e12);
  return 0;
}
#endif
#endif  // !MOM_PEARLHASH_ESIMD_TU
