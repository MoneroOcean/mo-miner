// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>
//
// Pearl (PRL) NoisyGEMM proof-of-useful-work GPU search kernel (SYCL / Level-Zero).
// Per seed: counter-RNG A/B -> keyed-BLAKE3 commitment roots -> sparse low-rank noise -> noised
// int8 A'*B' XMX GEMM tiles -> per-16x16-tile XOR/rotl13 transcript -> keyed-BLAKE3 jackpot vs
// target. On a win the host builds the pool PlainProof (Merkle openings) directly -- see the
// PEARL_STANDALONE block. Validated against the real verify_plain_proof_v2 (pearl-research-labs).
//
// Throughput notes (Arc B580, k=1024, rank=64, m=n=16384): the search GEMM sustains ~32-34
// TH/s (DPAS MAC/s); the full per-seed attempt ~28-30 TH/s, the rest being the mandatory
// BLAKE3 commitment over A/Bt. Key optimizations: A/B are never materialized (regenerated from
// the counter-RNG inside the consumers), the commitment Merkle tree is reduced in parallel,
// the search uses a PEARL_HR x PEARL_NTILE register tile with software-pipelined B loads.

#include <sycl/sycl.hpp>
#ifdef PEARL_ESIMD
#include <sycl/ext/intel/esimd.hpp>   // experimental register-resident DPAS search path
#endif
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

// ---- One-shot BLAKE3 (keyed + unkeyed) for host and SYCL device code ----
// Used by the Pearl kernel for matrix-commitment roots, commitment hashes, the noise PRNG and the
// PoW jackpot. Ported from the BLAKE3 reference impl; validated against the python verifier.
// Device-friendly: no std::, no heap, no recursion; cv_stack is a fixed array.
namespace pearl_b3 {

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

}  // namespace pearl_b3

namespace mom_pearl {

using pearl_b3::b3;

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
static inline uint32_t rd_u32le(const uint8_t* p) { return pearl_b3::le32(p); }
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
  int32_t *EARp1, *EARp2, *EBLq1, *EBLq2;
  uint8_t *cA, *cB, *key, *target, *CVA, *CVB;
  Result* result;
};

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
      pearl_b3::parent_cv(cvs + (size_t)(rd0 + 2 * i) * 32, cvs + (size_t)(rd0 + 2 * i + 1) * 32, key, isRoot, cvs + (size_t)(wr0 + i) * 32); });
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
    pearl_b3::chunk_cv(buf, (uint32_t)len, (uint64_t)ci, B.key, B.CVA + ci * 32); });
  // B' root is over Bt (n x k); each 1024-byte chunk of the Bt byte-stream is regenerated straight
  // from B's RNG (B[i]=gv(seed,tot+i)) so neither B nor a transposed Bt buffer exists. Bt byte
  // b -> B[(b%k)*n + b/k].
  q.parallel_for(sycl::range<1>(ncB), [=](sycl::id<1> id) { int ci = (int)id[0]; int off = ci * 1024; int len = lenB - off < 1024 ? lenB - off : 1024;
    uint8_t buf[1024];
    int j = off / k, r = off % k;        // Bt byte (off+t) -> column j, depth r of B; advance incrementally
    for (int t = 0; t < len; t++) { buf[t] = (uint8_t)gv(seed, tot + (uint32_t)(r * n + j)); if (++r == k) { r = 0; j++; } }
    pearl_b3::chunk_cv(buf, (uint32_t)len, (uint64_t)ci, B.key, B.CVB + ci * 32); });
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
      pearl_b3::merge_root(B.CVA, ncA, B.key, rootA);
      pearl_b3::merge_root(B.CVB, ncB, B.key, rootB);
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
  q.parallel_for(sycl::range<1>(dBR), [=](sycl::id<1> id) { int i = (int)id[0]; uint8_t sd[32]; mk_seed(sd, 'B'); uint8_t h[32]; dev_randHash(i, sd, B.cB, 0, h);
    int total = n * rank; for (int j = 0; j < 32; j++) { int o = i * 32 + j; if (o < total) B.EBRt[o] = (int8_t)((h[j] & 63) - 32); } });
  q.parallel_for(sycl::range<1>(dP), [=](sycl::id<1> id) { int i = (int)id[0]; uint8_t sd[32]; mk_seed(sd, 'A'); uint8_t h[32]; dev_randHash(i, sd, B.cA, 1, h);
    for (int kk = 0; kk < 8; kk++) { int idx = i * 8 + kk; if (idx >= k) break; perm_pair(rd_u32le(h + kk*4), rank, B.EARp1[idx], B.EARp2[idx]); } });
  q.parallel_for(sycl::range<1>(dP), [=](sycl::id<1> id) { int i = (int)id[0]; uint8_t sd[32]; mk_seed(sd, 'B'); uint8_t h[32]; dev_randHash(i, sd, B.cB, 1, h);
    for (int kk = 0; kk < 8; kk++) { int idx = i * 8 + kk; if (idx >= k) break; perm_pair(rd_u32le(h + kk*4), rank, B.EBLq1[idx], B.EBLq2[idx]); } });
  q.parallel_for(sycl::range<1>((size_t)rank * n), [=](sycl::id<1> id) { int idx = (int)id[0], r = idx / n, c = idx % n; B.EBR[idx] = B.EBRt[c * rank + r]; });
}

static void compute_ab(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool portable) {
  auto B = bb; const uint32_t tot = (uint32_t)(m * k);
  // Explicit capture list (not [=]): capturing the extra `portable` bool under [=] makes the host
  // (icx) and device (clang) compilers disagree on the lambda layout in the dual-compiler combined
  // build ("Unexpected kernel lambda size"). Listing the captures explicitly keeps them in sync.
  q.parallel_for(sycl::range<1>((size_t)m * k), [B, seed, k, rank, portable](sycl::id<1> id) { int idx = (int)id[0], i = idx / k, c = idx % k;
    int acc = (int)B.EAL[i * rank + B.EARp1[c]] - (int)B.EAL[i * rank + B.EARp2[c]];
    int8_t v = (int8_t)(gv(seed, (uint32_t)idx) + acc);
#ifdef __NVPTX__
    (void)portable;
    B.Ap[idx] = v;   // row-major A' for the NVIDIA mma.sync search_cuda (nvptx device pass)
#else
    // The spir64 image carries BOTH searches; the host sets `portable` to match the one it will launch:
    //  - portable (the dp4a int8 GEMM search(), used by every OpenCL device -- AMD, the CPU device, and
    //    Intel-OpenCL): ROW-MAJOR A'[i*k+c], so each output row reads a contiguous int8 run over k that
    //    packs into the 4-wide dp4a operand.
    //  - else (ESIMD search_esimd): TILE-MAJOR -- each 8x32 dpas-A fragment is one contiguous 256-byte
    //    block_load (the fast Intel path on Level-Zero).
    if (portable) B.Ap[idx] = v;
    else B.Ap[((size_t)((i >> 3) * (k / 32) + (c >> 5)) * 256) + (i & 7) * 32 + (c & 31)] = v;
#endif
  });
  // B' layout matches the search variant that will run (explicit capture, same reason as above):
  //  - nvptx (__NVPTX__): ROW-MAJOR B'[r,j] for the joint_matrix B operand, else COLUMN-major for the
  //    default mma.sync B operand (coalesced 32-bit loads).
  //  - portable (dp4a search()): COLUMN-major B'[r,j] at j*k+r, so each output column reads a
  //    contiguous int8 run over k -- the dp4a counterpart to the row-major A'.
  //  - else (Intel XMX ESIMD): TILE-MAJOR VNNI, each 16-wide N-tile's k*16 block contiguous (stride 64),
  //    Bp[ (j/16)*k*16 + (r/4)*64 + (j%16)*4 + (r%4) ] = B'[r,j], coalesced sub-group read.
  q.parallel_for(sycl::range<1>((size_t)k * n), [B, seed, tot, k, n, portable](sycl::id<1> id) { int idx = (int)id[0], r = idx / n, j = idx % n;
    int acc = (int)B.EBR[B.EBLq1[r] * n + j] - (int)B.EBR[B.EBLq2[r] * n + j];
    int8_t v = (int8_t)(gv(seed, tot + (uint32_t)idx) + acc);
#ifdef __NVPTX__
    (void)portable;
#endif
#if defined(__NVPTX__) && defined(PEARL_CU_JM)
    B.Bp[(size_t)r * n + j] = v;                 // row-major B'[r,j] for joint_matrix use::b
#elif defined(__NVPTX__)
    B.Bp[(size_t)j * k + r] = v;                 // col-major B'[r,j] for mma.sync B operand (coalesced loads)
#else
    (void)n;
    if (portable) B.Bp[(size_t)j * k + r] = v;   // col-major B'[r,j] for the dp4a search()
    else B.Bp[(size_t)(j / 16) * k * 16 + (size_t)(r / 4) * 64 + (size_t)(j % 16) * 4 + (r % 4)] = v;
#endif
  });
}

// PEARL_ER x PEARL_EC: the micro-tile of output cells each work-item owns (raises the dp4a MAC/byte so
// each A'/B' int32 word fetched from L2 feeds ER*EC / (ER+EC) dot products instead of one). A 16x16 hash
// tile is computed by one work-group of 256/(ER*EC) work-items. Tunable for the GPU's register file.
#ifndef PEARL_ER
#define PEARL_ER 4
#endif
#ifndef PEARL_EC
#define PEARL_EC 4
#endif
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
// Position-weighted mix of a tile's (row,col) + its 16-word transcript. When MOM_PEARL_CHK is set every
// search path atomically sums tile_mix over ALL tiles into result->chk -- a whole-search checksum used to
// cross-validate the portable search() against search_esimd/search_cuda: identical A'/B' must give an
// identical chk. The sum is order-independent (any tile schedule) and the row/col weighting makes it
// sensitive to a tile mis-mapping (e.g. an A'/B' transpose). Scalar-only, so it works in every kernel.
[[maybe_unused]] static inline uint32_t tile_mix(uint32_t row, uint32_t col, const uint32_t tr[16]) {
  uint32_t h = row * 2654435761u + col * 2246822519u + 0x9e3779b9u;
  for (int w = 0; w < 16; w++) h = (h ^ tr[w]) * 16777619u + (uint32_t)w;
  return h;
}
// Portable int8 GEMM search -- the matrix-hardware-free path that EVERY OpenCL device takes: AMD GPUs
// (no ESIMD, no usable joint_matrix), the SYCL CPU device (cpu1), and Intel-OpenCL (gpu1o). Each
// work-group computes one 16x16 hash tile; its work-items split the tile's 256 cells into PEARL_ER x
// PEARL_EC micro-tiles (one work-item per micro-tile, ER*EC int32 accumulators in registers). A'/B' are
// laid out by compute_ab(portable=true) -- A' row-major, B' column-major -- so each output cell reads a
// contiguous int8 run over k that the compiler packs (4 int8 at a time, loaded as one int32) into the
// 4-wide integer dot product: dp4a on Intel/AMD GPUs, plain scalar int MACs on the CPU -- bit-identical
// either way, since int8*int8->int32 is exact everywhere. There is NO SLM staging and NO per-rank
// barrier: A'/B' reuse is left to the L2 cache, and each work-item folds its own cells into a private
// 16-word partial transcript during the k-loop, so the ONLY work-group synchronisation is a single
// XOR-reduction tree at the end (a plain barrier tree -- not a sub-group op, so no reqd_sub_group_size
// is needed and the kernel runs on AMD and the CPU where the sub-group size is not 16). The result
// (found/seed/row/col) and transcript are bit-identical to search_esimd / search_cuda for the same A'/B'
// (cross-checked via MOM_PEARL_CHK), so this path finds the same tiles. Requires k % rank == 0 and
// k/rank <= 16 (true for every shipped shape: network 4096/256 and pearlpool 1024/64 both give 16).
// Compiled in every build that carries attempt() -- including the Windows/dpcpp Intel build where it
// shares the TU with search_esimd -- and EXCLUDED only from the separate spir64-only ESIMD TU
// (pearl_esimd.cpp, which sets MOM_PEARL_ESIMD_TU and re-includes this file just for search_esimd).
#if !defined(MOM_PEARL_ESIMD_TU)
template <int ER, int EC>
static void search_t(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg) {
  constexpr int HT = 16;
  constexpr int WG = (HT * HT) / (ER * EC);                 // work-items per tile
  [[maybe_unused]] constexpr int BW = HT / EC;              // micro-tile blocks per row (unused on the empty nvptx pass)
  static_assert(HT % ER == 0 && HT % EC == 0, "pearl portable: ER/EC must divide 16");
  static_assert((WG & (WG - 1)) == 0, "pearl portable: work-group size must be a power of two");
  auto B = bb;
  const int tilesW = n / HT, k4 = k / 4, nb = k / rank;   // k4 = int32 columns; nb = rank chunks (<= 16)
  const size_t ntiles = (size_t)(m / HT) * tilesW;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> red(WG * 16, h);   // batched per-boundary XOR reduction scratch
    // Explicit capture list (not [=]): keeps the kernel-lambda layout identical across the host, spir64
    // and nvptx passes (the nvptx body is #ifdef'd out below), so the dual-compiler combined build's
    // "Unexpected kernel lambda size" static assert stays satisfied.
    h.parallel_for(sycl::nd_range<1>(ntiles * WG, WG),
                   [B, seed, red, tilesW, k4, rank, nb, dbg](sycl::nd_item<1> it) {
#ifndef __NVPTX__
      const int g = (int)it.get_group(0), R = g / tilesW, C = g % tilesW;
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
#else
      (void)it; (void)B; (void)seed; (void)red; (void)tilesW; (void)k4; (void)rank; (void)nb; (void)dbg;
#endif
    });
  });
}
// Run the portable search at the configured micro-tile. PEARL_ER/PEARL_EC default to the B580 OpenCL
// sweet spot (4x4, measured); override at build time (-DPEARL_ER=.. -DPEARL_EC=..) to retune for another
// GPU's register file. SLM staging and software-pipelined prefetch were both tried and measured slower on
// the B580 (the L2 already absorbs the A'/B' reuse; IGC schedules the loads better than a manual pipeline).
static void search(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg) {
  search_t<PEARL_ER, PEARL_EC>(q, bb, seed, m, n, k, rank, dbg);
}
#endif  // !MOM_PEARL_ESIMD_TU

// NVIDIA pearl throughput lives entirely in the int8 Tensor Cores, reached here through the
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
// ~6 TH/s). PEARL_CU_BLK swizzles warps into BLK x BLK super-blocks so each block's A'/B' slice stays
// L2-resident (the m=n=131072 shape is 512MB/operand -> DRAM-bound without it). Tunable at build.
#ifndef PEARL_CU_ER
#define PEARL_CU_ER 4    // 4x4 register tile = best MAC/byte reuse without accumulator spill (generic, not L4-tuned)
#endif
#ifndef PEARL_CU_EC
#define PEARL_CU_EC 4
#endif
#ifndef PEARL_CU_BLK
#define PEARL_CU_BLK 16  // super-block swizzle for L2 reuse at the big network shape
#endif
#if defined(PEARL_CU_JM)
// joint_matrix variant (kept for comparison; the SYCL accumulator read caps it at ~20 TH/s -- the
// default below uses native mma.sync inline PTX instead). Build with -DPEARL_CU_JM + B' row-major.
static void search_cuda(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank) {
  using namespace sycl::ext::oneapi::experimental::matrix;
  constexpr int T = 16, SG = 32, ER = PEARL_CU_ER, EC = PEARL_CU_EC, BLK = PEARL_CU_BLK;
  auto B = bb;
  const int tilesH = m / (T * ER), tilesW = n / (T * EC);
  const size_t nWarp = (size_t)tilesH * tilesW;
  const bool blocked = BLK > 1 && (tilesW % BLK) == 0 && (tilesH % BLK) == 0;
  const int blocksW = blocked ? tilesW / BLK : 0;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> trbuf(sycl::range<1>(ER * EC * 16), h);  // per-warp transcripts in SLM
    h.parallel_for(sycl::nd_range<1>(nWarp * SG, SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
      auto sg = it.get_sub_group();
      const int lane = (int)sg.get_local_linear_id();
      const int warp = (int)it.get_group(0);
      int Rg, Cg;
      if (blocked) { const int blk = warp / (BLK * BLK), intra = warp % (BLK * BLK);
        Rg = (blk / blocksW) * BLK + (intra / BLK); Cg = (blk % blocksW) * BLK + (intra % BLK); }
      else { Rg = warp / tilesW; Cg = warp % tilesW; }
      const int rowBase = Rg * T * ER, colBase = Cg * T * EC;
      auto Ap = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(B.Ap);
      auto Bp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(B.Bp);
      joint_matrix<sycl::sub_group, int8_t, use::a, T, T, layout::row_major> a[ER];
      joint_matrix<sycl::sub_group, int8_t, use::b, T, T, layout::row_major> b[EC];
      joint_matrix<sycl::sub_group, int32_t, use::accumulator, T, T> acc[ER * EC];
#pragma unroll
      for (int i = 0; i < ER * EC; i++) joint_matrix_fill(sg, acc[i], 0);
      // Transcripts live in SLM, not per-lane registers: holding ER*EC*16 u32/lane spilled the
      // accumulators out of tensor-core registers and collapsed the GEMM (79 -> 17 TH/s). Only lane 0
      // touches trbuf (the fold reduces across the warp first), so no SLM race / barrier is needed.
      if (lane == 0)
#pragma unroll
        for (int i = 0; i < ER * EC * 16; i++) trbuf[i] = 0;
      int rc = 0;
      for (int p = 0; p < k; p += T) {
#pragma unroll
        for (int r = 0; r < ER; r++) joint_matrix_load(sg, a[r], Ap + (size_t)(rowBase + r * T) * k + p, k);
#pragma unroll
        for (int c = 0; c < EC; c++) joint_matrix_load(sg, b[c], Bp + (size_t)p * n + colBase + c * T, n);
#pragma unroll
        for (int r = 0; r < ER; r++)
#pragma unroll
          for (int c = 0; c < EC; c++) joint_matrix_mad(sg, acc[r * EC + c], a[r], b[c], acc[r * EC + c]);
        if ((p + T) % rank == 0) {     // rank-chunk boundary: XOR-fold each cumulative tile into SLM
#ifndef PEARL_CU_NODRAIN
#pragma unroll
          for (int t = 0; t < ER * EC; t++) {
            uint32_t part = 0;
            joint_matrix_apply(sg, acc[t], [&](int32_t& v) { part ^= (uint32_t)v; });
            part = sycl::reduce_over_group(sg, part, sycl::bit_xor<uint32_t>{});
            if (lane == 0) trbuf[t * 16 + rc % 16] = rotl(trbuf[t * 16 + rc % 16], 13) ^ part;
          }
#endif
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
          if (at.exchange(1) == 0) { B.result->seed = seed; B.result->row = (uint32_t)(rowBase + r * T); B.result->col = (uint32_t)(colBase + c * T); }
        }
      }
    });
  });
}
#else   // default NVIDIA path: native mma.sync IMMA via inline PTX
// One warp computes an ER x EC grid of 16x16 hash tiles using the int8 Tensor-Core MMA
// mma.sync.aligned.m16n8k32 (a 16x16 tile = two n=8 halves). The accumulator stays in registers and
// the inner-hash reads those registers DIRECTLY -- unlike joint_matrix_apply, a plain register read
// does not spill the accumulators / pessimize the GEMM (that capped the joint_matrix path at ~20).
// This is the NVIDIA analog of the Intel ESIMD dpas path. A' is row-major, B' is COLUMN-major (so
// each thread's 4 contiguous int8 of a fragment is one aligned 32-bit load). PEARL_CU_BLK super-block
// swizzle keeps the A'/B' slice L2-resident at the 131072 shape.
//
// mma.m16n8k32.s8 register fragment layout (PTX ISA): lane -> gid=lane/4, tid=lane%4.
//   A(16x32,row): a[0]=A[gid][tid*4..],   a[1]=A[gid+8][tid*4..],   a[2]=A[gid][16+tid*4..],   a[3]=A[gid+8][16+tid*4..]
//   B(32x8 ,col): b[0]=B[tid*4..][gid],   b[1]=B[16+tid*4..][gid]
//   C(16x8 ,row): c[0]=C[gid][tid*2], c[1]=C[gid][tid*2+1], c[2]=C[gid+8][tid*2], c[3]=C[gid+8][tid*2+1]
static inline void mma_m16n8k32_s8(int32_t c[4], const uint32_t a[4], const uint32_t b[2]) {
#ifdef __NVPTX__
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  (void)c; (void)a; (void)b;   // host pass: never executed
#endif
}
static inline uint32_t ld_u32(const int8_t* __restrict__ p) { return *reinterpret_cast<const uint32_t*>(p); }
// cp.async helpers (PEARL_CU_PIPE pipeline): async global->shared 16-byte copies that overlap with
// mma, plus the generic->shared address conversion cp.async needs. Device-only (inline PTX).
static inline uint32_t cu_smem_addr(const void* p) {
  uint32_t a = 0;
#ifdef __NVPTX__
  asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
#else
  (void)p;
#endif
  return a;
}
static inline void cu_cp_async16(uint32_t smem, const int8_t* g) {
#ifdef __NVPTX__
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(g));
#else
  (void)smem; (void)g;
#endif
}
static inline void cu_cp_commit() {
#ifdef __NVPTX__
  asm volatile("cp.async.commit_group;\n");
#endif
}
template <int N> static inline void cu_cp_wait() {
#ifdef __NVPTX__
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
#if !defined(PEARL_CU_PIPE)
// DEFAULT NVIDIA path: one warp per ER x EC grid of 16x16 hash tiles, mma.sync m16n8k32 int8, operands
// from global (L2-resident via the BLK super-block swizzle), accumulators resident in registers
// (pearl's inner-hash needs CUMULATIVE C, so tiles cannot be evicted like a normal GEMM), inner-hash
// folds them per rank via direct register reads. ~24 TH/s on an L4 -- correct and the best working
// config; occupancy-bound by the resident-accumulator register tile. (PEARL_CU_PIPE selects the
// experimental SMEM/cp.async pipeline.)
static void search_cuda(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank) {
  constexpr int SG = 32, ER = PEARL_CU_ER, EC = PEARL_CU_EC;
  auto B = bb;
  const int tilesH = m / (16 * ER), tilesW = n / (16 * EC);
  // BLK super-block size: pick the largest power-of-two whose A'/B' operand slice stays L2-resident,
  // derived from the device's actual L2 cache size -- so the L2-reuse win generalizes across NVIDIA
  // cards (L4's 48MB L2 -> 64, +15% vs the old fixed 16; small-L2 cards stay at the PEARL_CU_BLK floor
  // and can't thrash/regress). Slice ~= BLK*16*k*(ER+EC) int8 bytes. MOM_PEARL_CU_BLK overrides. Must
  // evenly divide both tile dims or super-block swizzle disables (falls back to row-major).
  int BLK = PEARL_CU_BLK;
  if (const char* e = std::getenv("MOM_PEARL_CU_BLK")) {
    BLK = std::atoi(e); if (BLK < 1) BLK = PEARL_CU_BLK;
  } else {
    const size_t l2 = q.get_device().get_info<sycl::info::device::global_mem_cache_size>();
    const size_t per_blk = (size_t)16 * k * (ER + EC);   // operand bytes per unit of BLK
    for (int cand = 128; cand > PEARL_CU_BLK; cand >>= 1)
      if ((size_t)cand * per_blk <= l2 && (tilesW % cand) == 0 && (tilesH % cand) == 0) { BLK = cand; break; }
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
#ifndef PEARL_CU_NODRAIN
#pragma unroll
          for (int t = 0; t < ER * EC; t++) {
            uint32_t part = (uint32_t)acc[t * 2][0] ^ (uint32_t)acc[t * 2][1] ^ (uint32_t)acc[t * 2][2] ^ (uint32_t)acc[t * 2][3]
                          ^ (uint32_t)acc[t * 2 + 1][0] ^ (uint32_t)acc[t * 2 + 1][1] ^ (uint32_t)acc[t * 2 + 1][2] ^ (uint32_t)acc[t * 2 + 1][3];
#ifndef PEARL_CU_NOREDUCE
            part = sycl::reduce_over_group(sg, part, sycl::bit_xor<uint32_t>{});
#endif
            if (lane == 0) trbuf[t * 16 + rc % 16] = rotl(trbuf[t * 16 + rc % 16], 13) ^ part;
          }
#endif
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
#else   // PEARL_CU_PIPE: SMEM-staged + cp.async double-buffered mma.sync pipeline
// ncu of the default kernel: 78% memory throughput, 16.5% occupancy (208 regs/thread) -> memory-bound
// + occupancy-starved. This pipeline fixes both: a block of WM x WN warps stages each k-chunk of
// A'(BM x BK) and B'(BN x BK) in SMEM and REUSES it across all warps (cuts global/L2 traffic), with
// cp.async DOUBLE-BUFFERING so the next chunk's async copy overlaps the current chunk's mma (no barrier
// stall -- the Phase-A barrier-only version regressed to 9), and a small per-warp register tile (TM x
// TN) -> high occupancy. Accumulators are cumulative (resident); inner-hash folds them per rank.
#ifndef PEARL_CU_WM
#define PEARL_CU_WM 4    // warps per block along M
#endif
#ifndef PEARL_CU_WN
#define PEARL_CU_WN 2    // warps per block along N
#endif
#ifndef PEARL_CU_TM
#define PEARL_CU_TM 2    // 16x16 hash tiles per warp along M
#endif
#ifndef PEARL_CU_TN
#define PEARL_CU_TN 2    // 16x16 hash tiles per warp along N
#endif
#ifndef PEARL_CU_BK
#define PEARL_CU_BK 64   // K staged per SMEM chunk (multiple of 16 for cp.async, of 32 for the mma K)
#endif
static void search_cuda(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank) {
  constexpr int WM = PEARL_CU_WM, WN = PEARL_CU_WN, TM = PEARL_CU_TM, TN = PEARL_CU_TN, BK = PEARL_CU_BK;
  constexpr int NW = WM * WN, TBSZ = NW * 32, BM = WM * TM * 16, BN = WN * TN * 16;
  constexpr int AB = BM * BK, BB = BN * BK, A16 = AB / 16, B16 = BB / 16;
  auto B = bb;
  const int mB = m / BM, nB = n / BN, nChunks = k / BK;
  const size_t nBlocks = (size_t)mB * nB;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int8_t, 1> As(sycl::range<1>(2 * AB), h);                 // double-buffered A' [2][BM][BK]
    sycl::local_accessor<int8_t, 1> Bs(sycl::range<1>(2 * BB), h);                 // double-buffered B' [2][BN][BK] (col-major B')
    sycl::local_accessor<uint32_t, 1> trbuf(sycl::range<1>(NW * TM * TN * 16), h); // per-warp transcripts
    h.parallel_for(sycl::nd_range<1>(nBlocks * TBSZ, TBSZ), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
      auto grp = it.get_group();
      const int tid = (int)it.get_local_linear_id();
      const int warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tln = lane & 3;
      const int wM = warp / WN, wN = warp % WN;
      const int block = (int)it.get_group(0), bR = block / nB, bC = block % nB;
      const int rowBlock = bR * BM, colBlock = bC * BN;
      const int8_t* __restrict__ Ap = B.Ap; const int8_t* __restrict__ Bp = B.Bp;
      int8_t* Asp = As.get_multi_ptr<sycl::access::decorated::no>().get();
      int8_t* Bsp = Bs.get_multi_ptr<sycl::access::decorated::no>().get();
      uint32_t* tr = trbuf.get_multi_ptr<sycl::access::decorated::no>().get() + warp * TM * TN * 16;
      int32_t acc[TM * TN * 2][4];
#pragma unroll
      for (int i = 0; i < TM * TN * 2; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0; }
      if (lane == 0)
#pragma unroll
        for (int i = 0; i < TM * TN * 16; i++) tr[i] = 0;
      const int warpRow = wM * TM * 16, warpCol = wN * TN * 16;
      auto load_chunk = [&](int buf, int p) {     // async-copy k-chunk at offset p into SMEM buffer buf
        int8_t* aDst = Asp + buf * AB; int8_t* bDst = Bsp + buf * BB;
        for (int ci = tid; ci < A16; ci += TBSZ) { const int mm = (ci * 16) / BK, kk = (ci * 16) % BK;
          cu_cp_async16(cu_smem_addr(aDst + mm * BK + kk), Ap + (size_t)(rowBlock + mm) * k + p + kk); }
        for (int ci = tid; ci < B16; ci += TBSZ) { const int nn = (ci * 16) / BK, kk = (ci * 16) % BK;
          cu_cp_async16(cu_smem_addr(bDst + nn * BK + kk), Bp + (size_t)(colBlock + nn) * k + p + kk); }
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
          uint32_t ra[TM][4];
#pragma unroll
          for (int mt = 0; mt < TM; mt++) {
            const int mr = warpRow + mt * 16 + gid, o = ks + tln * 4;
            ra[mt][0] = ld_u32(Ab + mr * BK + o);        ra[mt][1] = ld_u32(Ab + (mr + 8) * BK + o);
            ra[mt][2] = ld_u32(Ab + mr * BK + o + 16);   ra[mt][3] = ld_u32(Ab + (mr + 8) * BK + o + 16);
          }
#pragma unroll
          for (int nt = 0; nt < TN; nt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
              const int nc = warpCol + nt * 16 + hh * 8 + gid, o = ks + tln * 4;
              uint32_t rb[2] = { ld_u32(Bb + nc * BK + o), ld_u32(Bb + nc * BK + o + 16) };
#pragma unroll
              for (int mt = 0; mt < TM; mt++) mma_m16n8k32_s8(acc[(mt * TN + nt) * 2 + hh], ra[mt], rb);
            }
          if (((c * BK + ks) + 32) % rank == 0) {     // cumulative k hit a rank boundary -> fold
#ifndef PEARL_CU_NODRAIN
#pragma unroll
            for (int t = 0; t < TM * TN; t++) {
              uint32_t part = (uint32_t)acc[t*2][0] ^ (uint32_t)acc[t*2][1] ^ (uint32_t)acc[t*2][2] ^ (uint32_t)acc[t*2][3]
                            ^ (uint32_t)acc[t*2+1][0] ^ (uint32_t)acc[t*2+1][1] ^ (uint32_t)acc[t*2+1][2] ^ (uint32_t)acc[t*2+1][3];
              part = sycl::reduce_over_group(it.get_sub_group(), part, sycl::bit_xor<uint32_t>{});
              if (lane == 0) tr[t * 16 + rc % 16] = rotl(tr[t * 16 + rc % 16], 13) ^ part;
            }
#endif
            rc++;
          }
        }
        sycl::group_barrier(grp);   // mma done reading this buffer before it is reused two chunks later
      }
      if (lane == 0) {
#pragma unroll
        for (int mt = 0; mt < TM; mt++)
#pragma unroll
          for (int nt = 0; nt < TN; nt++) {
            const int t = mt * TN + nt;
            uint32_t full[16];
#pragma unroll
            for (int s = 0; s < 16; s++) full[s] = tr[t * 16 + s];
            if (!tile_wins(full, B.cA, B.target)) continue;
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> at(B.result->found);
            if (at.exchange(1) == 0) { B.result->seed = seed; B.result->row = (uint32_t)(rowBlock + warpRow + mt * 16); B.result->col = (uint32_t)(colBlock + warpCol + nt * 16); }
          }
      }
    });
  });
}
#endif  // !PEARL_CU_PIPE
#endif  // PEARL_CU_JM
#endif  // MOM_SYCL_HAS_CUDA

#ifdef PEARL_ESIMD
// ---- experimental ESIMD register-resident DPAS search (alternative to search()) ----
// One ESIMD work-item owns a PEARL_E_HR x PEARL_E_NTILE grid of 16x16 hash tiles in its own GRF
// (ESIMD sub-group size is 1 -- no lane cooperation). The DPAS accumulators stay in registers and
// the per-rank inner-hash XORs them in-GRF, avoiding the joint_matrix_store->SLM->barrier readback
// that caps search() at ~10 TH/s. xmx::dpas(C,B,A): SystolicDepth=8, RepeatCount=8, int8*int8->i32,
// K=32/call; A=8x32 (simd int8 256), B=32x16 VNNI (simd int8 512, == compute_ab's Bp layout),
// C=8x16 (simd int32 128). B' loads straight from Bp; A' is 8 strided row loads from row-major Ap.
// ER=EC=2 (a 2x2 grid of 16x16 tiles => 8 int32 accumulators) is the B580 sweet spot: it maximizes
// operand reuse (16 MAC/byte) while staying within the GRF -- 9+ accumulators spill and collapse
// throughput (ER=4 or EC=4 measured 5-11 TH/s vs 34 at 2x2). Override at build time if retuning.
#ifndef PEARL_E_NTILE
#define PEARL_E_NTILE 2   // ESIMD column-tiles per work-item (EC); same GRF sweet spot as search()
#endif
#ifndef PEARL_E_HR
#define PEARL_E_HR 2      // ESIMD row-bands per work-item (ER); ER=EC=2 => 8 accumulators, no spill
#endif
// Cache-blocked work-item traversal (tile swizzle): instead of row-major wi->(Rg,Cg) -- which
// re-streams ALL of B' for every row-band and goes DRAM-bound once A'/B' exceed L2 (the m=n=131072
// HeroMiners shape: 512MB each, ~2TB of redundant DRAM reads -> 12.6 TH/s) -- iterate a BLK x BLK
// square of work-items per "super-block" so its A'/B' slice (~16MB at BLK=64) stays resident in L2
// and is reused across the block's work-items. Cuts DRAM traffic ~32x -> compute-bound again. BLK=0
// disables (plain row-major). Requires tilesH%BLK==0 && tilesW%BLK==0, else falls back to row-major.
#ifndef PEARL_E_BLK
#define PEARL_E_BLK 64
#endif
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

// External in the standalone spir64-only ESIMD TU (pearl_esimd.cpp) so the main pearl.o can call it;
// static in the single-TU Intel/Windows build where it lives in this same object.
#ifdef MOM_PEARL_ESIMD_TU
void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg) {
#else
static void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg) {
#endif
  constexpr int ER = PEARL_E_HR, EC = PEARL_E_NTILE, HT = 16;
  auto B = bb;
  const int tilesH = m / (HT * ER), tilesW = n / (HT * EC);
  const size_t nWI = (size_t)tilesH * tilesW;
  constexpr int BLK = PEARL_E_BLK;
  // L2 cache-blocking: only swizzle into BLK x BLK super-blocks when the grid divides evenly (see
  // the PEARL_E_BLK note above); otherwise fall back to plain row-major traversal.
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
#endif  // PEARL_ESIMD


// In the combined build the ESIMD search lives in the separate spir64-only TU (pearl_esimd.cpp);
// declare it here so attempt() can call it. (When PEARL_ESIMD is set this is the same TU, defined above.)
#if defined(MOM_PEARL_HAS_ESIMD) && !defined(PEARL_ESIMD)
void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg);
#endif

static void attempt(sycl::queue& q, const Buffers& b, uint32_t seed, int m, int n, int k, int rank) {
  k_roots(q, b, seed, m, n, k);          // commitment roots cA/cB (A/Bt regenerated from RNG)
  k_noise(q, b, m, n, k, rank);          // sparse low-rank noise E_AL/E_AR, E_BL/E_BR

  // Pick the search kernel by the running device's backend BEFORE laying out A'/B': compute_ab needs to
  // know whether to write the portable row-major A' / column-major B' (search()) or the ESIMD tile-major
  // layout. CUDA -> search_cuda (mma.sync, nvptx). Any OpenCL device (AMD GPU, the CPU device, and
  // Intel-OpenCL gpu1o) -> the portable dp4a search() -- it is the single matrix-hardware-free path, so
  // the same image runs on AMD and the CPU and gpu1o behaves like a non-Intel GPU would. (Set
  // MOM_PEARL_ESIMD to opt an Intel-OpenCL *GPU* back into the faster ESIMD path.) Level-Zero / default
  // Intel -> ESIMD. search_cuda/search_esimd/search() are each compiled only in the builds that can run
  // them, so the references below are guarded to match.
#if defined(MOM_SYCL_HAS_CUDA)
  const bool is_cuda = q.get_device().get_backend() == sycl::backend::ext_oneapi_cuda;
#else
  const bool is_cuda = false;
#endif
  const bool dbg = std::getenv("MOM_PEARL_CHK") != nullptr;   // whole-search checksum cross-validation
  // The portable search() is compiled into every build that has attempt() (i.e. not the ESIMD-only TU),
  // so it is always callable here -- including the Windows/dpcpp Intel build, which has search_esimd in
  // the SAME TU. Route any OpenCL device (AMD GPU, the CPU device, Intel-OpenCL gpu1o) to it; an
  // Intel-OpenCL GPU can opt back into ESIMD with MOM_PEARL_ESIMD. CUDA -> search_cuda; Level-Zero -> ESIMD.
  bool use_portable = false;
#if defined(PEARL_ESIMD) || defined(MOM_PEARL_HAS_ESIMD)
  if (!is_cuda && q.get_device().get_backend() == sycl::backend::opencl) {
    const bool is_gpu = q.get_device().is_gpu();
    use_portable = !(is_gpu && std::getenv("MOM_PEARL_ESIMD"));
  }
#else
  use_portable = !is_cuda;   // dpcpp-cuda build: no ESIMD, so any non-CUDA device takes the portable path
#endif

  compute_ab(q, b, seed, m, n, k, rank, /*portable=*/use_portable);

#if defined(MOM_SYCL_HAS_CUDA)
  if (is_cuda) { search_cuda(q, b, seed, m, n, k, rank); return; }
#endif
#ifndef MOM_PEARL_ESIMD_TU   // search() is not compiled into the ESIMD-only TU (which only exports search_esimd)
  if (use_portable) { search(q, b, seed, m, n, k, rank, dbg); return; }   // portable dp4a int8 GEMM (AMD / CPU / Intel-OpenCL)
#endif
#if defined(PEARL_ESIMD) || defined(MOM_PEARL_HAS_ESIMD)
  search_esimd(q, b, seed, m, n, k, rank, dbg); // ESIMD register-resident DPAS path (Intel GPUs, Level-Zero, ~53 TH/s)
#else
  search(q, b, seed, m, n, k, rank, dbg);       // unreached (use_portable already handled non-CUDA above)
#endif
}

}  // namespace mom_pearl

// The standalone spir64-only ESIMD TU (pearl_esimd.cpp) only needs the device kernels inside the
// namespace above; the host-side entry point + PlainProof builder below belong to the main TU only.
#ifndef MOM_PEARL_ESIMD_TU
using namespace mom_pearl;

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
  pearl_b3::b3(kb, 128, nullptr, key32);
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
  for (int i = 0; i < numLeaves; i++) { int len = gen_leaf(which, seed, i, m, n, k, chunk); pearl_b3::chunk_cv(chunk, (uint32_t)len, (uint64_t)i, key, l0[i].data()); }
  std::vector<std::vector<CV>> layers; layers.push_back(std::move(l0));
  while (layers.back().size() > 2) {
    auto& prev = layers.back(); std::vector<CV> next;
    for (size_t i = 0; i < prev.size(); i += 2) {
      if (i + 1 < prev.size()) { CV p; pearl_b3::parent_cv(prev[i].data(), prev[i + 1].data(), key, false, p.data()); next.push_back(p); }
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
  uint8_t root[32]; { std::vector<CV> tmp = layers[0]; pearl_b3::merge_root((uint8_t*)tmp.data(), numLeaves, key, root); }
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
static Buffers alloc_buffers(sycl::queue& q, int m, int n, int k, int rank) {
  Buffers b{};
  b.EAL=sycl::malloc_device<int8_t>(m*rank,q); b.EBR=sycl::malloc_device<int8_t>(rank*n,q); b.EBRt=sycl::malloc_device<int8_t>(n*rank,q);
  b.Ap=sycl::malloc_device<int8_t>(m*k,q); b.Bp=sycl::malloc_device<int8_t>(k*n,q);
  b.EARp1=sycl::malloc_device<int32_t>(k,q); b.EARp2=sycl::malloc_device<int32_t>(k,q);
  b.EBLq1=sycl::malloc_device<int32_t>(k,q); b.EBLq2=sycl::malloc_device<int32_t>(k,q);
  b.cA=sycl::malloc_device<uint8_t>(32,q); b.cB=sycl::malloc_device<uint8_t>(32,q);
  b.key=sycl::malloc_device<uint8_t>(32,q); b.target=sycl::malloc_device<uint8_t>(32,q);
  b.CVA=sycl::malloc_device<uint8_t>((2*((m*k+1023)/1024)+2)*32,q); b.CVB=sycl::malloc_device<uint8_t>((2*((n*k+1023)/1024)+2)*32,q);
  b.result=sycl::malloc_shared<Result>(1,q); b.result->found=0;
  return b;
}

#ifndef PEARL_STANDALONE
// ---- native mom entry point (DEV::PEARL_GPU) ----
#include "lib-internal.h"   // get_dev() + MOM_SYCL_API
namespace {
// Network-standard NoisyGEMM shape (what HeroMiners/LuckyPool and the reference miner use): k=4096,
// noise_rank=256, m=n=131072 (m=n set from intensity in pearl()). pearlpool.cloud's 1024/64/16384 is
// the low-mem exception; override via MOM_PEARL_K / MOM_PEARL_RANK (and a smaller dev batch) for it.
static int env_int(const char* name, int dflt) { const char* e = std::getenv(name); int v = e && *e ? atoi(e) : 0; return v ? v : dflt; }
static int pearl_k()    { return env_int("MOM_PEARL_K", 4096); }
static int pearl_rank() { return env_int("MOM_PEARL_RANK", 256); }
// One persistent in-order queue + device buffer set per GPU; the seed search reuses them.
struct PearlState {
  sycl::queue queue;
  int m = 0, n = 0, k = 0, rank = 0;
  Buffers b{};
  uint8_t key[32] = {0}, header[76] = {0};
  bool have_header = false;
  double wait_ema_us = 0.0; // EMA of recent attempt wall times (us); paces the pre-wait sleep below
  std::mutex mutex;
  explicit PearlState(const std::string& dev_str)
    : queue(get_dev(dev_str), sycl::property::queue::in_order{}) {}
  void free_buffers() {
    // Free every non-null member independently (sycl::free is no-op'd by the per-element `if (p)`),
    // so a partial-allocation failure where Ap is null but earlier members (EAL/EBR/EBRt/...) are
    // not still releases them instead of leaking. No early return on !b.Ap.
    queue.wait();
    for (void* p : {(void*)b.EAL,(void*)b.EBR,(void*)b.EBRt,(void*)b.Ap,(void*)b.Bp,
                    (void*)b.EARp1,(void*)b.EARp2,(void*)b.EBLq1,(void*)b.EBLq2,
                    (void*)b.cA,(void*)b.cB,(void*)b.key,(void*)b.target,
                    (void*)b.CVA,(void*)b.CVB,(void*)b.result}) if (p) sycl::free(p, queue);
    b = Buffers{};
  }
  void ensure(int m_, int n_, int k_, int rank_) {
    if (b.Ap && m == m_ && n == n_ && k == k_ && rank == rank_) return;
    free_buffers();
    m = m_; n = n_; k = k_; rank = rank_;
    b = alloc_buffers(queue, m, n, k, rank);
    have_header = false;
  }
  ~PearlState() { free_buffers(); }
};
std::mutex g_states_mutex;
std::map<std::string, std::unique_ptr<PearlState>> g_states;
PearlState& pearl_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(g_states_mutex);
  auto& st = g_states[dev_str];
  if (!st) st = std::make_unique<PearlState>(dev_str);
  return *st;
}
thread_local std::string g_pearl_proof;   // last built proof, fetched right after pearl() returns 1
// Winning tile captured by pearl() on a find; consumed by pearl_proof() to build the proof lazily.
thread_local struct PearlFound { uint32_t seed; int row, col, m, n, k, rank; uint8_t key[32]; bool valid; } g_pf{};
} // namespace

// One seeded NoisyGEMM attempt per call (seed = *pseed). Returns 1 on a winning tile, writes the
// winning seed back to *pseed, and stashes the base64 PlainProof for pearl_proof(). intensity = m=n.
int pearl(
  const unsigned job_ref, const uint32_t, const uint8_t* const input, const unsigned input_size,
  uint8_t* const, uint64_t* const pseed, const uint8_t* const target,
  const unsigned intensity, const bool is_test, const bool, const std::string& dev_str
) {
  if (input_size < 76) throw std::string("Bad pearl input length (need 76-byte header)");
  const int k = pearl_k(), rank = pearl_rank();
  // intensity = m = n (the square NoisyGEMM edge), taken from the dev batch "gpuN*<intensity>".
  // A bare "gpuN" (no batch -> intensity 1) means "use the network-standard m=n" (131072); an
  // explicit batch >=128 is honoured (e.g. gpuN*16384 for low-mem cards / pearlpool.cloud).
  int m = is_test ? 256 : ((int)intensity >= 128 ? (int)intensity : 131072);
  const int n = m;

  PearlState& st = pearl_state(dev_str);
  std::lock_guard<std::mutex> lock(st.mutex);
  st.ensure(m, n, k, rank);
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
  b.result->chk = 0;   // whole-search checksum (filled only when MOM_PEARL_CHK is set; see tile_mix)
  q.wait();

  attempt(q, b, (uint32_t)*pseed, m, n, k, rank);
  // The in-order queue's wait busy-spins a host core on both backends (CUDA's native sync and the
  // Intel GPU wait), and pearl waits once per attempt -- a long, very stable GEMM. So sleep through
  // most of the attempt (an EMA of recent attempt wall times) before waiting, leaving only a short
  // spinning tail; the result is read from shared USM afterwards so completion stays exact. The 90%
  // sleep stays under the attempt time, adding no latency. Mirrors the cn/gpu pre-read sleep.
  const auto attempt_start = std::chrono::steady_clock::now();
  if (st.wait_ema_us > 2000.0)
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(st.wait_ema_us * 0.90)));
  q.wait();
  {
    const double us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - attempt_start).count());
    st.wait_ema_us = st.wait_ema_us == 0.0 ? us : st.wait_ema_us * 0.8 + us * 0.2;
  }
  if (std::getenv("MOM_PEARL_CHK"))   // cross-validation: same A'/B' -> identical chk on every search path
    std::fprintf(stderr, "PEARL_CHK dev=%s seed=%u chk=%08x\n", dev_str.c_str(), (uint32_t)*pseed, b.result->chk);
  if (!b.result->found) return 0;
  *pseed = b.result->seed;
  if (is_test) return 1;                                             // test: core only needs "ok", no proof
  // Capture the winning tile; DEFER the heavy host PlainProof rebuild (regenerate A/Bt + the BLAKE3
  // Merkle tree, ~tens of ms at m=n=16384) to pearl_proof(), which core calls ONCE per pool job_id.
  // Building it here on every find would starve the DPAS search: at a low pool difficulty a tile is
  // found on most attempts, yet the pool credits only one share per job. Deferring keeps the search
  // running flat out (~6x faster live).
  g_pf.seed = b.result->seed; g_pf.row = (int)b.result->row; g_pf.col = (int)b.result->col;
  g_pf.m = m; g_pf.n = n; g_pf.k = k; g_pf.rank = rank; std::memcpy(g_pf.key, st.key, 32); g_pf.valid = true;
  return 1;
}
// Builds the PlainProof for the last winning tile captured by pearl() (lazy: called once per job_id
// by core, off the search hot path). Returns "" if no tile has been found yet.
const char* pearl_proof() {
  if (g_pf.valid)
    g_pearl_proof = build_plain_proof(g_pf.seed, g_pf.m, g_pf.n, g_pf.k, g_pf.rank, g_pf.key, g_pf.row, g_pf.col);
  return g_pearl_proof.c_str();
}
// GEMM MACs per attempt = m*n*k (m=n), mirroring pearl()'s intensity clamp. This is the work unit
// the pearl hashrate is reported in, so the core's display matches the "TH/s" GEMM benchmark.
uint64_t pearl_attempt_hashes(unsigned intensity) {
  const uint64_t m = intensity >= 128 ? intensity : 131072;
  return m * m * (uint64_t)pearl_k();
}
#endif  // !PEARL_STANDALONE

#ifdef PEARL_STANDALONE
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
  fprintf(stderr, "pearl-server dev: %s  m=%d n=%d k=%d rank=%d\n", q.get_device().get_info<sycl::info::device::name>().c_str(), m, n, k, rank);
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
#endif  // !MOM_PEARL_ESIMD_TU
