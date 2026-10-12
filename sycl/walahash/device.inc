// WalaHash consensus math and packed hash primitives; included inside mom_walahash.
#include "../blake3-device.inc"

struct Result {
  uint32_t count;
  uint64_t nonce;
  uint8_t hash[32];
};
template <bool, bool> class SearchKernel;
#ifdef WALAHASH_ESIMD
class PrehashKernel;
class FinalKernel;
#endif
#ifdef MOM_SYCL_HAS_HIP
class AmdPrehashKernel;
class AmdFinalKernel;
#endif
#ifdef WALAHASH_ESIMD
namespace esimd = sycl::ext::intel::esimd;
#endif

inline uint64_t load64le(const uint8_t* p) {
  uint64_t v = 0;
  for (unsigned i = 0; i < 8; ++i)
    v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}
inline uint64_t rotl64(uint64_t v, unsigned n) {
  return (v << n) | (v >> (64 - n));
}
// Accumulate four packed int8 products; unsupported backends retain the scalar-equivalent path.
template <bool Native> inline int32_t dot4(uint32_t a, uint32_t b, int32_t c) {
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  // The signed API carries unsigned accumulator bits; keep DP4A's modulo-2^32 sum.
  const uint32_t accumulator = static_cast<uint32_t>(c);
  return sycl::bit_cast<int32_t>(mom::nvidia::dot4_u8<Native>(a, b, accumulator));
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) && \
    (!defined(MOM_SYCL_ADAPTIVECPP) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610))
  if constexpr (Native) {
    uint32_t result;
    const uint32_t accumulator = static_cast<uint32_t>(c);
    asm volatile("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(accumulator));
    return static_cast<int32_t>(result);
  }
#elif !defined(MOM_SYCL_ADAPTIVECPP) && !defined(WALAHASH_ESIMD)
  if constexpr (Native)
    return sycl::ext::oneapi::dot_acc(a, b, c);
#endif
  return c + static_cast<uint8_t>(a) * static_cast<uint8_t>(b) +
         static_cast<uint8_t>(a >> 8) * static_cast<uint8_t>(b >> 8) +
         static_cast<uint8_t>(a >> 16) * static_cast<uint8_t>(b >> 16) +
         static_cast<uint8_t>(a >> 24) * static_cast<uint8_t>(b >> 24);
}

// Check the generated matrix with the consensus floating-elimination tolerance.
static unsigned matrix_rank(const uint8_t matrix[4096]) {
  double a[4096];
  for (unsigned i = 0; i < 4096; ++i)
    a[i] = matrix[i];
  bool selected[64]{};
  unsigned rank = 0;
  for (unsigned col = 0; col < 64; ++col) {
    unsigned row = 0;
    while (row < 64 && (selected[row] || sycl::fabs(a[row * 64 + col]) <= 1e-9))
      ++row;
    if (row == 64)
      continue;
    ++rank;
    selected[row] = true;
    for (unsigned j = col + 1; j < 64; ++j)
      a[row * 64 + j] /= a[row * 64 + col];
    for (unsigned i = 0; i < 64; ++i)
      if (i != row && sycl::fabs(a[i * 64 + col]) > 1e-9)
        for (unsigned j = col + 1; j < 64; ++j)
          a[i * 64 + j] -= a[row * 64 + j] * a[i * 64 + col];
  }
  return rank;
}

// Advance the header-seeded generator until it produces a full-rank 64x64 nibble matrix.
static void make_matrix(const uint8_t hash[32], uint8_t matrix[4096]) {
  uint64_t s0 = load64le(hash), s1 = load64le(hash + 8), s2 = load64le(hash + 16),
           s3 = load64le(hash + 24);
  do
    for (unsigned i = 0; i < 4096; i += 16) {
      const uint64_t v = s0 + rotl64(s0 + s3, 23), t = s1 << 17;
      s2 ^= s0;
      s3 ^= s1;
      s1 ^= s2;
      s0 ^= s3;
      s2 ^= t;
      s3 = rotl64(s3, 45);
      for (unsigned j = 0; j < 16; ++j)
        matrix[i + j] = static_cast<uint8_t>((v >> (4 * j)) & 15);
    }
  while (matrix_rank(matrix) != 64);
}

// Permute the SHA3 state; the 23-round specialization leaves the final round to packed code.
template <unsigned FullRounds = 24> inline void sha3f(uint64_t st[25]) {
  static_assert(FullRounds == 23 || FullRounds == 24);
  constexpr uint64_t rc[24] = {0x1,
                               0x8082,
                               0x800000000000808a,
                               0x8000000080008000,
                               0x808b,
                               0x80000001,
                               0x8000000080008081,
                               0x8000000000008009,
                               0x8a,
                               0x88,
                               0x80008009,
                               0x8000000a,
                               0x8000808b,
                               0x800000000000008b,
                               0x8000000000008089,
                               0x8000000000008003,
                               0x8000000000008002,
                               0x8000000000000080,
                               0x800a,
                               0x800000008000000a,
                               0x8000000080008081,
                               0x8000000000008080,
                               0x80000001,
                               0x8000000080008008};
  for (unsigned r = 0; r < FullRounds; ++r) {
    uint64_t b[5];
    b[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
    b[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
    b[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
    b[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
    b[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];
    for (unsigned i = 0; i < 5; ++i) {
      const uint64_t t = b[(i + 4) % 5] ^ rotl64(b[(i + 1) % 5], 1);
      st[i] ^= t;
      st[i + 5] ^= t;
      st[i + 10] ^= t;
      st[i + 15] ^= t;
      st[i + 20] ^= t;
    }
    const uint64_t t = st[1];
    st[1] = rotl64(st[6], 44);
    st[6] = rotl64(st[9], 20);
    st[9] = rotl64(st[22], 61);
    st[22] = rotl64(st[14], 39);
    st[14] = rotl64(st[20], 18);
    st[20] = rotl64(st[2], 62);
    st[2] = rotl64(st[12], 43);
    st[12] = rotl64(st[13], 25);
    st[13] = rotl64(st[19], 8);
    st[19] = rotl64(st[23], 56);
    st[23] = rotl64(st[15], 41);
    st[15] = rotl64(st[4], 27);
    st[4] = rotl64(st[24], 14);
    st[24] = rotl64(st[21], 2);
    st[21] = rotl64(st[8], 55);
    st[8] = rotl64(st[16], 45);
    st[16] = rotl64(st[5], 36);
    st[5] = rotl64(st[3], 28);
    st[3] = rotl64(st[18], 21);
    st[18] = rotl64(st[17], 15);
    st[17] = rotl64(st[11], 10);
    st[11] = rotl64(st[7], 6);
    st[7] = rotl64(st[10], 3);
    st[10] = rotl64(t, 1);
    for (unsigned j = 0; j < 25; j += 5) {
      b[0] = st[j];
      b[1] = st[j + 1];
      st[j] ^= (~st[j + 1]) & st[j + 2];
      st[j + 1] ^= (~st[j + 2]) & st[j + 3];
      st[j + 2] ^= (~st[j + 3]) & st[j + 4];
      st[j + 3] ^= (~st[j + 4]) & b[0];
      st[j + 4] ^= (~b[0]) & b[1];
    }
    st[0] ^= rc[r];
  }
  if constexpr (FullRounds == 23) {
    // SHA3-256 exposes only the first four lanes. Compute just those lanes in the final Keccak
    // round instead of transforming the other 21 values that the WalaHash matrix never reads.
    const uint64_t c0 = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
    const uint64_t c1 = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
    const uint64_t c2 = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
    const uint64_t c3 = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
    const uint64_t c4 = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];
    const uint64_t x0 = st[0] ^ c4 ^ rotl64(c1, 1);
    const uint64_t x1 = rotl64(st[6] ^ c0 ^ rotl64(c2, 1), 44);
    const uint64_t x2 = rotl64(st[12] ^ c1 ^ rotl64(c3, 1), 43);
    const uint64_t x3 = rotl64(st[18] ^ c2 ^ rotl64(c4, 1), 21);
    const uint64_t x4 = rotl64(st[24] ^ c3 ^ rotl64(c0, 1), 14);
    st[0] = x0 ^ (~x1 & x2) ^ rc[23];
    st[1] = x1 ^ (~x2 & x3);
    st[2] = x2 ^ (~x3 & x4);
    st[3] = x3 ^ (~x4 & x0);
  }
}
// Hash a fixed 32-byte input with SHA3-256 domain separation.
inline void sha3_256(uint8_t out[32], const uint8_t in[32]) {
  uint64_t st[25]{};
  for (unsigned i = 0; i < 4; ++i)
    st[i] = load64le(in + 8 * i);
  st[4] ^= 0x06;
  st[16] ^= 0x8000000000000000ULL;
  sha3f(st);
  for (unsigned i = 0; i < 32; ++i)
    out[i] = static_cast<uint8_t>(st[i / 8] >> (8 * (i % 8)));
}

#if defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)
// Apply SHA3-256 to eight packed words for the CUDA fixed-hash path.
inline void sha3_words64(uint32_t io[8]) {
  uint64_t st[25]{};
  for (unsigned i = 0; i < 4; ++i)
    st[i] = static_cast<uint64_t>(io[2 * i]) | (static_cast<uint64_t>(io[2 * i + 1]) << 32);
  st[4] = 0x06;
  st[16] = 0x8000000000000000ULL;
  sha3f<23>(st);
  for (unsigned i = 0; i < 4; ++i) {
    io[2 * i] = static_cast<uint32_t>(st[i]);
    io[2 * i + 1] = static_cast<uint32_t>(st[i] >> 32);
  }
}
#endif

#ifdef MOM_SYCL_HAS_HIP
#include "amd_wmma.inc"
#endif

#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA) || defined(MOM_SYCL_HAS_HIP)
#if defined(__clang__) || defined(__GNUC__)
#define MOM_WALAHASH_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define MOM_WALAHASH_ALWAYS_INLINE inline
#endif
// Apply a compile-time BLAKE3 message schedule so device code avoids dynamic permutation.
template <unsigned R>
MOM_WALAHASH_ALWAYS_INLINE void b3_fixed_round(uint32_t st[16], const uint32_t m[16]) {
  constexpr const uint8_t* sc = B3_MSG[R];
  b3_g(st, 0, 4, 8, 12, m[sc[0]], m[sc[1]]);
  b3_g(st, 1, 5, 9, 13, m[sc[2]], m[sc[3]]);
  b3_g(st, 2, 6, 10, 14, m[sc[4]], m[sc[5]]);
  b3_g(st, 3, 7, 11, 15, m[sc[6]], m[sc[7]]);
  b3_g(st, 0, 5, 10, 15, m[sc[8]], m[sc[9]]);
  b3_g(st, 1, 6, 11, 12, m[sc[10]], m[sc[11]]);
  b3_g(st, 2, 7, 8, 13, m[sc[12]], m[sc[13]]);
  b3_g(st, 3, 4, 9, 14, m[sc[14]], m[sc[15]]);
}
// Compress one fixed-size BLAKE3 block with explicit counter and flags.
MOM_WALAHASH_ALWAYS_INLINE void b3_fixed(uint32_t out[8], const uint32_t cv[8],
                                         const uint32_t m[16], uint32_t len, uint32_t flags) {
  uint32_t st[16] = {cv[0],    cv[1],    cv[2],    cv[3],    cv[4], cv[5], cv[6], cv[7],
                     B3_IV[0], B3_IV[1], B3_IV[2], B3_IV[3], 0,     0,     len,   flags};
  b3_fixed_round<0>(st, m);
  b3_fixed_round<1>(st, m);
  b3_fixed_round<2>(st, m);
  b3_fixed_round<3>(st, m);
  b3_fixed_round<4>(st, m);
  b3_fixed_round<5>(st, m);
  b3_fixed_round<6>(st, m);
  for (unsigned i = 0; i < 8; ++i)
    out[i] = st[i] ^ st[i + 8];
}
struct Blake3Cv {
  uint32_t words[8];
};
// Cache the invariant first 64 bytes of an 80-byte work header.
MOM_WALAHASH_ALWAYS_INLINE void b3_first_block_cv(Blake3Cv& cv, const uint8_t* input) {
  uint32_t m[16];
  for (unsigned i = 0; i < 16; ++i)
    m[i] = load32_le_dev(input + 4 * i);
  b3_fixed(cv.words, B3_IV, m, 64, 1);
}
// Hash the nonce-bearing header tail using the cached first-block chaining value.
MOM_WALAHASH_ALWAYS_INLINE void b3_80_from_cv(uint32_t out[8], const Blake3Cv& cv,
                                              const uint8_t* tail, uint64_t nonce) {
  uint32_t m[16]{};
  m[0] = load32_le_dev(tail);
  m[1] = load32_le_dev(tail + 4);
  m[2] = nonce;
  m[3] = nonce >> 32;
  b3_fixed(out, cv.words, m, 16, 10);
}
// Hash a fixed 32-byte value with the BLAKE3 root flags.
MOM_WALAHASH_ALWAYS_INLINE void b3_32(uint32_t out[8], const uint32_t input[8]) {
  uint32_t m[16]{};
  for (unsigned i = 0; i < 8; ++i)
    m[i] = input[i];
  b3_fixed(out, B3_IV, m, 32, 11);
}
#undef MOM_WALAHASH_ALWAYS_INLINE
struct KeccakWord {
  uint32_t lo, hi;
};
inline KeccakWord operator^(KeccakWord a, KeccakWord b) {
  return {a.lo ^ b.lo, a.hi ^ b.hi};
}
inline KeccakWord operator&(KeccakWord a, KeccakWord b) {
  return {a.lo & b.lo, a.hi & b.hi};
}
inline KeccakWord operator|(KeccakWord a, KeccakWord b) {
  return {a.lo | b.lo, a.hi | b.hi};
}
inline KeccakWord operator~(KeccakWord a) {
  return {~a.lo, ~a.hi};
}
#ifdef WALAHASH_ESIMD
// Implement Keccak chi with ESIMD lane logic for the packed state.
inline KeccakWord keccak_chi(KeccakWord a, KeccakWord b, KeccakWord c) {
  return a ^ (~b & c);
}
#endif
inline KeccakWord& operator^=(KeccakWord& a, KeccakWord b) {
  a.lo ^= b.lo;
  a.hi ^= b.hi;
  return a;
}
// Rotate bit-interleaved even/odd halves; odd shifts also swap the halves.
template <unsigned N> inline KeccakWord keccak_rotl(KeccakWord a) {
#ifdef WALAHASH_ESIMD
  if constexpr ((N & 1) == 0)
    return {sycl::rotate(a.lo, N / 2u), sycl::rotate(a.hi, N / 2u)};
  else
    return {sycl::rotate(a.hi, (N + 1) / 2u), sycl::rotate(a.lo, N / 2u)};
#else
  if constexpr ((N & 1) == 0)
    return {mo_rotate(a.lo, static_cast<uint32_t>(N / 2u)),
            mo_rotate(a.hi, static_cast<uint32_t>(N / 2u))};
  else
    return {mo_rotate(a.hi, static_cast<uint32_t>((N + 1) / 2u)),
            mo_rotate(a.lo, static_cast<uint32_t>(N / 2u))};
#endif
}
// Gather alternating bits into one 32-bit half of a bit-interleaved Keccak lane.
inline uint32_t keccak_compact(uint32_t x) {
  x &= 0x55555555;
  x = (x | (x >> 1)) & 0x33333333;
  x = (x | (x >> 2)) & 0x0f0f0f0f;
  x = (x | (x >> 4)) & 0x00ff00ff;
  return (x | (x >> 8)) & 0xffff;
}
// Scatter a packed half back into alternating bits of a Keccak lane.
inline uint32_t keccak_expand(uint32_t x) {
  x &= 0xffff;
  x = (x | (x << 8)) & 0x00ff00ff;
  x = (x | (x << 4)) & 0x0f0f0f0f;
  x = (x | (x << 2)) & 0x33333333;
  return (x | (x << 1)) & 0x55555555;
}
// Convert a little-endian 64-bit lane into even/odd-bit halves for packed permutation.
inline KeccakWord keccak_pack(uint32_t lo, uint32_t hi) {
  return {keccak_compact(lo) | (keccak_compact(hi) << 16),
          keccak_compact(lo >> 1) | (keccak_compact(hi >> 1) << 16)};
}
// Recover the ordinary 64-bit lane words from their bit-interleaved halves.
inline void keccak_unpack(KeccakWord a, uint32_t& lo, uint32_t& hi) {
  lo = keccak_expand(a.lo) | keccak_expand(a.hi) << 1;
  hi = keccak_expand(a.lo >> 16) | keccak_expand(a.hi >> 16) << 1;
}
struct KeccakState {
  KeccakWord a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18,
      a19, a20, a21, a22, a23, a24;
};
constexpr uint32_t KECCAK_RC_LO[24] = {1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0,
                                       1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0};
constexpr uint32_t KECCAK_RC_HI[24] = {
    0,          0x89,       0x8000008b, 0x80008080, 0x8b,       0x8000,     0x80008088, 0x80000082,
    0xb,        0xa,        0x8082,     0x8003,     0x808b,     0x8000000b, 0x8000008a, 0x80000081,
    0x80000081, 0x80000008, 0x83,       0x80008003, 0x80008088, 0x80000088, 0x8000,     0x80008082};
// Apply one Keccak-f round to the bit-interleaved state.
inline void keccak_round(KeccakState& a, KeccakWord rc) {
  const KeccakWord c0 = a.a0 ^ a.a5 ^ a.a10 ^ a.a15 ^ a.a20,
                   c1 = a.a1 ^ a.a6 ^ a.a11 ^ a.a16 ^ a.a21,
                   c2 = a.a2 ^ a.a7 ^ a.a12 ^ a.a17 ^ a.a22,
                   c3 = a.a3 ^ a.a8 ^ a.a13 ^ a.a18 ^ a.a23,
                   c4 = a.a4 ^ a.a9 ^ a.a14 ^ a.a19 ^ a.a24;
  const KeccakWord d0 = c4 ^ keccak_rotl<1>(c1), d1 = c0 ^ keccak_rotl<1>(c2),
                   d2 = c1 ^ keccak_rotl<1>(c3), d3 = c2 ^ keccak_rotl<1>(c4),
                   d4 = c3 ^ keccak_rotl<1>(c0);
  a.a0 ^= d0;
  a.a5 ^= d0;
  a.a10 ^= d0;
  a.a15 ^= d0;
  a.a20 ^= d0;
  a.a1 ^= d1;
  a.a6 ^= d1;
  a.a11 ^= d1;
  a.a16 ^= d1;
  a.a21 ^= d1;
  a.a2 ^= d2;
  a.a7 ^= d2;
  a.a12 ^= d2;
  a.a17 ^= d2;
  a.a22 ^= d2;
  a.a3 ^= d3;
  a.a8 ^= d3;
  a.a13 ^= d3;
  a.a18 ^= d3;
  a.a23 ^= d3;
  a.a4 ^= d4;
  a.a9 ^= d4;
  a.a14 ^= d4;
  a.a19 ^= d4;
  a.a24 ^= d4;
  const KeccakWord t = a.a1;
  a.a1 = keccak_rotl<44>(a.a6);
  a.a6 = keccak_rotl<20>(a.a9);
  a.a9 = keccak_rotl<61>(a.a22);
  a.a22 = keccak_rotl<39>(a.a14);
  a.a14 = keccak_rotl<18>(a.a20);
  a.a20 = keccak_rotl<62>(a.a2);
  a.a2 = keccak_rotl<43>(a.a12);
  a.a12 = keccak_rotl<25>(a.a13);
  a.a13 = keccak_rotl<8>(a.a19);
  a.a19 = keccak_rotl<56>(a.a23);
  a.a23 = keccak_rotl<41>(a.a15);
  a.a15 = keccak_rotl<27>(a.a4);
  a.a4 = keccak_rotl<14>(a.a24);
  a.a24 = keccak_rotl<2>(a.a21);
  a.a21 = keccak_rotl<55>(a.a8);
  a.a8 = keccak_rotl<45>(a.a16);
  a.a16 = keccak_rotl<36>(a.a5);
  a.a5 = keccak_rotl<28>(a.a3);
  a.a3 = keccak_rotl<21>(a.a18);
  a.a18 = keccak_rotl<15>(a.a17);
  a.a17 = keccak_rotl<10>(a.a11);
  a.a11 = keccak_rotl<6>(a.a7);
  a.a7 = keccak_rotl<3>(a.a10);
  a.a10 = keccak_rotl<1>(t);
  const KeccakWord b0 = a.a0, b1 = a.a1, b5 = a.a5, b6 = a.a6, b10 = a.a10, b11 = a.a11,
                   b15 = a.a15, b16 = a.a16, b20 = a.a20, b21 = a.a21;
#ifdef WALAHASH_ESIMD
  a.a0 = keccak_chi(a.a0, a.a1, a.a2);
  a.a1 = keccak_chi(a.a1, a.a2, a.a3);
  a.a2 = keccak_chi(a.a2, a.a3, a.a4);
  a.a3 = keccak_chi(a.a3, a.a4, b0);
  a.a4 = keccak_chi(a.a4, b0, b1);
  a.a5 = keccak_chi(a.a5, a.a6, a.a7);
  a.a6 = keccak_chi(a.a6, a.a7, a.a8);
  a.a7 = keccak_chi(a.a7, a.a8, a.a9);
  a.a8 = keccak_chi(a.a8, a.a9, b5);
  a.a9 = keccak_chi(a.a9, b5, b6);
  a.a10 = keccak_chi(a.a10, a.a11, a.a12);
  a.a11 = keccak_chi(a.a11, a.a12, a.a13);
  a.a12 = keccak_chi(a.a12, a.a13, a.a14);
  a.a13 = keccak_chi(a.a13, a.a14, b10);
  a.a14 = keccak_chi(a.a14, b10, b11);
  a.a15 = keccak_chi(a.a15, a.a16, a.a17);
  a.a16 = keccak_chi(a.a16, a.a17, a.a18);
  a.a17 = keccak_chi(a.a17, a.a18, a.a19);
  a.a18 = keccak_chi(a.a18, a.a19, b15);
  a.a19 = keccak_chi(a.a19, b15, b16);
  a.a20 = keccak_chi(a.a20, a.a21, a.a22);
  a.a21 = keccak_chi(a.a21, a.a22, a.a23);
  a.a22 = keccak_chi(a.a22, a.a23, a.a24);
  a.a23 = keccak_chi(a.a23, a.a24, b20);
  a.a24 = keccak_chi(a.a24, b20, b21);
#else
  // Keep lanes 1, 2, 8, 12, 17, and 20 complemented between rounds. The transformed chi step is
  // algebraically identical to Keccak but needs only five explicit complements per round; theta and
  // rho/pi carry the fixed mask into the input pattern used below.
  a.a0 = a.a0 ^ (a.a1 | a.a2);
  a.a1 = a.a1 ^ (~a.a2 | a.a3);
  a.a2 = a.a2 ^ (a.a3 & a.a4);
  a.a3 = a.a3 ^ (a.a4 | b0);
  a.a4 = a.a4 ^ (b0 & b1);
  a.a5 = a.a5 ^ (a.a6 | a.a7);
  a.a6 = a.a6 ^ (a.a7 & a.a8);
  a.a7 = a.a7 ^ (a.a8 | ~a.a9);
  a.a8 = a.a8 ^ (a.a9 | b5);
  a.a9 = a.a9 ^ (b5 & b6);
  a.a10 = a.a10 ^ (a.a11 | a.a12);
  a.a11 = a.a11 ^ (a.a12 & a.a13);
  a.a12 = a.a12 ^ (~a.a13 & a.a14);
  a.a13 = ~a.a13 ^ (a.a14 | b10);
  a.a14 = a.a14 ^ (b10 & b11);
  a.a15 = a.a15 ^ (a.a16 & a.a17);
  a.a16 = a.a16 ^ (a.a17 | a.a18);
  a.a17 = a.a17 ^ (~a.a18 | a.a19);
  a.a18 = ~a.a18 ^ (a.a19 & b15);
  a.a19 = a.a19 ^ (b15 | b16);
  a.a20 = a.a20 ^ (~a.a21 & a.a22);
  a.a21 = ~a.a21 ^ (a.a22 | a.a23);
  a.a22 = a.a22 ^ (a.a23 & a.a24);
  a.a23 = a.a23 ^ (a.a24 | b20);
  a.a24 = a.a24 ^ (b20 & b21);
#endif
  a.a0 ^= rc;
}
#ifndef WALAHASH_ESIMD
// Build the padded single-block SHA3 state from a fixed 32-byte input.
inline void sha3_words_init(KeccakState& a, const uint32_t io[8]) {
  constexpr KeccakWord ones{~0U, ~0U};
  a.a0 = keccak_pack(io[0], io[1]);
  a.a1 = ~keccak_pack(io[2], io[3]);
  a.a2 = ~keccak_pack(io[4], io[5]);
  a.a3 = keccak_pack(io[6], io[7]);
  a.a4 = {2, 1};
  a.a8 = ones;
  a.a12 = ones;
  a.a16 = {0, 0x80000000};
  a.a17 = ones;
  a.a20 = ones;
}
// Complete the final permutation round and extract the first eight digest words.
inline void sha3_words_finish(KeccakState& a, uint32_t io[8]) {
  // SHA3-256 exposes only the first four lanes. Complete round 24 for that output row instead of
  // transforming the other 21 lanes that WalaHash immediately discards.
  const KeccakWord c0 = a.a0 ^ a.a5 ^ a.a10 ^ a.a15 ^ a.a20,
                   c1 = a.a1 ^ a.a6 ^ a.a11 ^ a.a16 ^ a.a21,
                   c2 = a.a2 ^ a.a7 ^ a.a12 ^ a.a17 ^ a.a22,
                   c3 = a.a3 ^ a.a8 ^ a.a13 ^ a.a18 ^ a.a23,
                   c4 = a.a4 ^ a.a9 ^ a.a14 ^ a.a19 ^ a.a24;
  const KeccakWord x0 = a.a0 ^ c4 ^ keccak_rotl<1>(c1),
                   x1 = keccak_rotl<44>(a.a6 ^ c0 ^ keccak_rotl<1>(c2)),
                   x2 = keccak_rotl<43>(a.a12 ^ c1 ^ keccak_rotl<1>(c3)),
                   x3 = keccak_rotl<21>(a.a18 ^ c2 ^ keccak_rotl<1>(c4)),
                   x4 = keccak_rotl<14>(a.a24 ^ c3 ^ keccak_rotl<1>(c0));
  a.a0 = x0 ^ (x1 | x2) ^ KeccakWord { KECCAK_RC_LO[23], KECCAK_RC_HI[23] };
  a.a1 = ~(x1 ^ (~x2 | x3));
  a.a2 = ~(x2 ^ (x3 & x4));
  a.a3 = x3 ^ (x4 | x0);
  keccak_unpack(a.a0, io[0], io[1]);
  keccak_unpack(a.a1, io[2], io[3]);
  keccak_unpack(a.a2, io[4], io[5]);
  keccak_unpack(a.a3, io[6], io[7]);
}
#endif
// Hash fixed-size packed input, choosing the ESIMD or scalar permutation at compile time.
inline void sha3_words(uint32_t io[8]) {
  KeccakState a{};
#ifdef WALAHASH_ESIMD
  a.a0 = keccak_pack(io[0], io[1]);
  a.a1 = keccak_pack(io[2], io[3]);
  a.a2 = keccak_pack(io[4], io[5]);
  a.a3 = keccak_pack(io[6], io[7]);
  a.a4 = {2, 1};
  a.a16 = {0, 0x80000000};
#pragma unroll 1
  for (unsigned r = 0; r < 24; ++r)
    keccak_round(a, {KECCAK_RC_LO[r], KECCAK_RC_HI[r]});
  keccak_unpack(a.a0, io[0], io[1]);
  keccak_unpack(a.a1, io[2], io[3]);
  keccak_unpack(a.a2, io[4], io[5]);
  keccak_unpack(a.a3, io[6], io[7]);
#else
  sha3_words_init(a, io);
  // Keep the first round outside the runtime-indexed loop so the compiler can fold the mostly-zero
  // initial SHA3 state. The remaining rounds stay rolled to avoid inflating per-thread registers.
  keccak_round(a, {KECCAK_RC_LO[0], KECCAK_RC_HI[0]});
#pragma unroll 1
  for (unsigned r = 1; r < 23; ++r)
    keccak_round(a, {KECCAK_RC_LO[r], KECCAK_RC_HI[r]});
  sha3_words_finish(a, io);
#endif
}

#endif
