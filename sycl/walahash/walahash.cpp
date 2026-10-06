// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
// WagLayla WalaHash: BLAKE3 -> SHA3-256 -> HeavyHash matrix -> BLAKE3.
//
// Production paths use Intel ESIMD/XMX, NVIDIA NVPTX DP4A with fixed-hash fusion at workgroup 128,
// and gfx12 AMD WMMA with split32 fixed hashes at workgroup 256. Replicated A/B measurements on an
// RTX 5060 Ti are 888.61 MH/s for fixed-fused DP4A versus 625.08 generic fused and 221--229 staged;
// an RX 9060 XT reaches 679.50 MH/s on Linux and 667.68 MH/s on Windows with packed-output split32
// WMMA, versus 448.37 generic and 497.35 partial64.
// The fixed-size BLAKE3 specialization captures the nonce-invariant first-block chaining value;
// split32 Keccak's lane-complement form reduces dependency-heavy chi operations on gfx12. Xe2
// instead reaches 628.30 MH/s with standard chi and sycl::rotate, versus 539.92 for lane-complement
// chi and 547.17 when the standard form uses the shared rotate wrapper.
// Explicit NVPTX DP4A and gfx12 WMMA match their compiler/hardware matrix instructions. Devices
// without a retained native path use the exact portable SYCL implementation.

#include <sycl/sycl.hpp>

#if defined(MOM_SYCL_HAS_HIP)
#include <hip/hip_runtime_api.h>
#include "../hiprtc-api.h"
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "../lib-internal.h"
#include "../nvidia-features.h"
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
#include "../nvidia-dot.h"
#endif
#include "../../native/job-boundary.h"
#if defined(WALAHASH_ESIMD)
#include "../intel-dpas.h"
#elif !defined(MOM_SYCL_ADAPTIVECPP)
#include <sycl/ext/oneapi/dot_product.hpp>
#endif
#include "../../native/consts.h"

namespace mom_walahash {

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

#if defined(MOM_SYCL_HAS_CUDA)
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

#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_HAS_HIP)
#if defined(__clang__) || defined(__GNUC__)
#define MOM_WALAHASH_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define MOM_WALAHASH_ALWAYS_INLINE inline
#endif
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
MOM_WALAHASH_ALWAYS_INLINE void b3_first_block_cv(Blake3Cv& cv, const uint8_t* input) {
  uint32_t m[16];
  for (unsigned i = 0; i < 16; ++i)
    m[i] = load32_le_dev(input + 4 * i);
  b3_fixed(cv.words, B3_IV, m, 64, 1);
}
MOM_WALAHASH_ALWAYS_INLINE void b3_80_from_cv(uint32_t out[8], const Blake3Cv& cv,
                                              const uint8_t* tail, uint64_t nonce) {
  uint32_t m[16]{};
  m[0] = load32_le_dev(tail);
  m[1] = load32_le_dev(tail + 4);
  m[2] = nonce;
  m[3] = nonce >> 32;
  b3_fixed(out, cv.words, m, 16, 10);
}
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
inline KeccakWord keccak_chi(KeccakWord a, KeccakWord b, KeccakWord c) {
  return a ^ (~b & c);
}
#endif
inline KeccakWord& operator^=(KeccakWord& a, KeccakWord b) {
  a.lo ^= b.lo;
  a.hi ^= b.hi;
  return a;
}
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
inline uint32_t keccak_compact(uint32_t x) {
  x &= 0x55555555;
  x = (x | (x >> 1)) & 0x33333333;
  x = (x | (x >> 2)) & 0x0f0f0f0f;
  x = (x | (x >> 4)) & 0x00ff00ff;
  return (x | (x >> 8)) & 0xffff;
}
inline uint32_t keccak_expand(uint32_t x) {
  x &= 0xffff;
  x = (x | (x << 8)) & 0x00ff00ff;
  x = (x | (x << 4)) & 0x0f0f0f0f;
  x = (x | (x << 2)) & 0x33333333;
  return (x | (x << 1)) & 0x55555555;
}
inline KeccakWord keccak_pack(uint32_t lo, uint32_t hi) {
  return {keccak_compact(lo) | (keccak_compact(hi) << 16),
          keccak_compact(lo >> 1) | (keccak_compact(hi >> 1) << 16)};
}
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

class State {
public:
  sycl::device device;
  sycl::queue queue;
  uint8_t* matrix = nullptr;
  uint8_t* input = nullptr;
  uint8_t* target = nullptr;
  uint8_t matrix_seed[32]{};
  Result* result = nullptr;
  bool matrix_ready = false;
#ifdef MOM_SYCL_HAS_CUDA
  const int cuda_sm = mom::nvidia::compute_capability(device);
#endif
  std::mutex mutex;
#ifdef MOM_SYCL_HAS_HIP
  WalaHashAmdWmma amd_wmma;
#endif
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_HAS_HIP)
  Blake3Cv prehash_cv{};
#endif
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_HIP)
  uint8_t *hashes = nullptr, *mixed = nullptr;
  unsigned capacity = 0;
#if defined(WALAHASH_ESIMD)
  uint8_t* vectors = nullptr;
#endif
#endif
  explicit State(const std::string& dev)
      : device(get_dev(dev)), queue(device, sycl::property::queue::in_order{}) {
    if (!mom_has_usm_device(device))
      throw std::string("walahash requires SYCL device USM");
    try {
      matrix = sycl::malloc_device<uint8_t>(4096, queue);
      input = sycl::malloc_device<uint8_t>(72, queue);
      target = sycl::malloc_device<uint8_t>(32, queue);
      result = sycl::malloc_device<Result>(1, queue);
      if (!matrix || !input || !target || !result)
        throw std::string("Can't allocate walahash buffers");
    } catch (...) {
      release();
      throw;
    }
  }
  ~State() {
    release();
  }
  template <typename T> void free_ptr(T*& pointer) noexcept {
    if (pointer)
      try {
        sycl::free(pointer, queue);
      } catch (...) {
      }
    pointer = nullptr;
  }
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_HIP)
  void free_accelerated() noexcept {
#if defined(WALAHASH_ESIMD)
    free_ptr(vectors);
#endif
    free_ptr(mixed);
    free_ptr(hashes);
    capacity = 0;
  }
#endif
  void release() noexcept {
    sycl_cleanup_noexcept("walahash wait", [&] {
      queue.wait_and_throw();
    });
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_HIP)
    free_accelerated();
#endif
    free_ptr(result);
    free_ptr(target);
    free_ptr(input);
    free_ptr(matrix);
  }
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_HIP)
  void ensure(unsigned count) {
    if (count <= capacity)
      return;
    constexpr unsigned alignment =
#if defined(MOM_SYCL_HAS_HIP)
        64;
#else
        16;
#endif
    const unsigned next_capacity = (count + alignment - 1) & ~(alignment - 1);
    queue.wait_and_throw();
    free_accelerated();
    try {
#if defined(WALAHASH_ESIMD)
      vectors = sycl::malloc_device<uint8_t>(static_cast<size_t>(next_capacity) * 64, queue);
#endif
      hashes = sycl::malloc_device<uint8_t>(static_cast<size_t>(next_capacity) * 32, queue);
      mixed = sycl::malloc_device<uint8_t>(static_cast<size_t>(next_capacity) * 32, queue);
#if defined(WALAHASH_ESIMD)
      if (!vectors || !hashes || !mixed)
#else
      if (!hashes || !mixed)
#endif
        throw std::string("Can't allocate accelerated walahash buffers");
      capacity = next_capacity;
    } catch (...) {
      free_accelerated();
      throw;
    }
  }
#endif
};
static DeviceStateRegistry<State>& registry() {
  static auto* const value = new DeviceStateRegistry<State>;
  return *value;
}
static State& state_for(const std::string& dev) {
  return registry().get(dev, [&] { return std::make_unique<State>(dev); });
}

void walahash_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "walahash: ordered SYCL cleanup failed\n");
  }
}

template <bool Native, bool FixedHashes>
static sycl::event search(State& s, uint64_t first, unsigned count, bool is_test,
                          const unsigned work_group) {
  const size_t global = (static_cast<size_t>(count) + work_group - 1) / work_group * work_group;
  const uint8_t* matrix = s.matrix;
  const uint8_t* input = s.input;
#ifdef MOM_SYCL_HAS_CUDA
  const Blake3Cv prehash_cv = s.prehash_cv;
#endif
  const uint8_t* target = s.target;
  Result* result = s.result;
  return s.queue.submit([&](sycl::handler& h) {
    h.parallel_for<SearchKernel<Native, FixedHashes>>(
        sycl::nd_range<1>(global, work_group),
#ifdef MOM_SYCL_HAS_CUDA
        [=](sycl::nd_item<1> it) [[sycl::work_group_size_hint(128)]] MOM_SYCL_KERNEL_ARGS_RESTRICT {
#else
        [=](sycl::nd_item<1> it) MOM_SYCL_KERNEL_ARGS_RESTRICT {
#endif
          const unsigned gid = it.get_global_id(0);
          if (gid >= count)
            return;
          const uint64_t nonce = first + gid;
          uint32_t pow_words[8];
          uint8_t* pow = reinterpret_cast<uint8_t*>(pow_words);
#ifdef MOM_SYCL_HAS_CUDA
          if constexpr (FixedHashes) {
            b3_80_from_cv(pow_words, prehash_cv, input + 64, nonce);
            sha3_words64(pow_words);
          } else
#endif
          {
            uint8_t pre[80];
            for (unsigned i = 0; i < 72; ++i)
              pre[i] = input[i];
            for (unsigned i = 0; i < 8; ++i)
              pre[72 + i] = nonce >> (8 * i);
            blake3_dev(pow, 32, pre, 80);
            sha3_256(pow, pow);
          }
          uint32_t vec[16];
#pragma unroll
          for (unsigned k = 0; k < 16; ++k) {
            const unsigned j = 4 * k;
            vec[k] = static_cast<uint32_t>(pow[j >> 1] >> 4) |
                     static_cast<uint32_t>(pow[j >> 1] & 15) << 8 |
                     static_cast<uint32_t>(pow[(j + 2) >> 1] >> 4) << 16 |
                     static_cast<uint32_t>(pow[(j + 2) >> 1] & 15) << 24;
          }
          auto m4 = sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(
              reinterpret_cast<const uint32_t*>(matrix));
          uint32_t mixed_words[8];
          uint8_t* mixed = reinterpret_cast<uint8_t*>(mixed_words);
          for (unsigned i = 0; i < 64; i += 4) {
            int32_t acc[4]{};
#pragma unroll
            for (unsigned k = 0; k < 16; ++k) {
              const uint32_t b = vec[k];
#pragma unroll
              for (unsigned r = 0; r < 4; ++r)
                acc[r] = dot4<Native>(m4[(i + r) * 16 + k], b, acc[r]);
            }
            uint8_t nibble[4];
            for (unsigned r = 0; r < 4; ++r)
              nibble[r] = (acc[r] & 15) ^ ((acc[r] >> 4) & 15) ^ ((acc[r] >> 8) & 15);
            mixed[i / 2] = pow[i / 2] ^ static_cast<uint8_t>(nibble[0] << 4 | nibble[1]);
            mixed[i / 2 + 1] = pow[i / 2 + 1] ^ static_cast<uint8_t>(nibble[2] << 4 | nibble[3]);
          }
#ifdef MOM_SYCL_HAS_CUDA
          if constexpr (FixedHashes)
            b3_32(pow_words, mixed_words);
          else
#endif
            blake3_dev(pow, 32, mixed, 32);
          bool hit = true;
          for (int i = 31; i >= 0; --i)
            if (pow[i] != target[i]) {
              hit = pow[i] < target[i];
              break;
            }
          if ((is_test && gid == 0) || hit) {
            using A =
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>;
            if (A(result->count).fetch_add(1) == 0) {
              result->nonce = nonce;
              for (unsigned i = 0; i < 32; ++i)
                result->hash[i] = pow[i];
            }
          }
        });
  });
}

#ifdef WALAHASH_ESIMD
static bool use_esimd(const sycl::device& d) {
  const char* configured = std::getenv("MOM_WALAHASH_SYCL_NATIVE");
  return (!configured || std::strcmp(configured, "0")) && d.is_gpu() &&
         d.has(sycl::aspect::ext_intel_matrix) && d.get_backend() != sycl::backend::opencl;
}
static sycl::event prehash(State& s, uint64_t first, unsigned count) {
  const uint8_t* input = s.input;
  const Blake3Cv prehash_cv = s.prehash_cv;
  uint8_t* hashes = s.hashes;
  uint8_t* vectors = s.vectors;
  return s.queue.submit([&](sycl::handler& h) {
    h.parallel_for<PrehashKernel>(
        sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const unsigned gid = id[0], lane = gid & 15, tile = gid >> 4;
      uint32_t pow[8];
      b3_80_from_cv(pow, prehash_cv, input + 64, first + gid);
      sha3_words(pow);
      auto hash32 = reinterpret_cast<uint32_t*>(hashes);
      for (unsigned i = 0; i < 8; ++i)
        hash32[(size_t)gid * 8 + i] = pow[i];
      for (unsigned k = 0; k < 64; ++k) {
        const uint8_t byte = pow[k >> 3] >> (8 * ((k >> 1) & 3)),
                      v = (k & 1) ? byte & 15 : byte >> 4;
        vectors[(size_t)tile * 1024 + (k >> 2) * 64 + lane * 4 + (k & 3)] = v;
      }
    });
  });
}
template <unsigned Width> static sycl::event multiply_esimd(State& s, unsigned count) {
  const int8_t* matrix = reinterpret_cast<int8_t*>(s.matrix);
  const int8_t* vectors = reinterpret_cast<int8_t*>(s.vectors);
  const uint8_t* hashes = s.hashes;
  uint8_t* mixed = s.mixed;
  const unsigned stride = s.capacity;
  return s.queue.submit([&](sycl::handler& h) {
    h.parallel_for(
        sycl::range<1>((count + 15) / 16),
        [=](sycl::id<1> id) SYCL_ESIMD_KERNEL MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const size_t tile = id[0];
      for (unsigned band = 0; band < 8; ++band) {
        esimd::simd<int32_t, 128> acc = 0;
        for (unsigned chunk = 0; chunk < 2; ++chunk) {
          auto a = esimd::block_load<int8_t, 256>(matrix + (band * 2 + chunk) * 256);
          auto b = esimd::block_load<int8_t, 512>(vectors + tile * 1024 + chunk * 512);
          acc = mom::intel_matrix::dpas_8x16x32<Width>(acc, b, a);
        }
        esimd::simd<uint32_t, 128> nibble = (acc & 15) ^ ((acc >> 4) & 15) ^ ((acc >> 8) & 15);
        const esimd::simd<uint32_t, 16> hash_offsets(0, 32);
        for (unsigned pair = 0; pair < 4; ++pair) {
          const unsigned byte = band * 4 + pair;
          const auto hi = nibble.template select<16, 1>(pair * 32),
                     lo = nibble.template select<16, 1>(pair * 32 + 16);
          const esimd::simd<uint8_t, 16> packed = esimd::convert<uint8_t>((hi << 4) | lo);
          const auto original =
              esimd::gather<uint8_t, 16>(hashes + tile * 512 + byte, hash_offsets);
          esimd::block_store<uint8_t, 16>(mixed + (size_t)byte * stride + tile * 16,
                                          original ^ packed);
        }
      }
    });
  });
}
static sycl::event finalize(State& s, uint64_t first, unsigned count, bool is_test) {
  const uint8_t* mixed = s.mixed;
  const unsigned stride = s.capacity;
  const uint8_t* target = s.target;
  Result* result = s.result;
  return s.queue.submit([&](sycl::handler& h) {
    h.parallel_for<FinalKernel>(
        sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const unsigned gid = id[0];
      uint32_t block[8], pow[8];
      for (unsigned i = 0; i < 32; ++i)
        reinterpret_cast<uint8_t*>(block)[i] = mixed[(size_t)i * stride + gid];
      b3_32(pow, block);
      bool hit = true;
      for (int i = 31; i >= 0; --i) {
        const uint8_t byte = pow[i >> 2] >> (8 * (i & 3));
        if (byte != target[i]) {
          hit = byte < target[i];
          break;
        }
      }
      if ((is_test && gid == 0) || hit) {
        using A =
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>;
        if (A(result->count).fetch_add(1) == 0) {
          result->nonce = first + gid;
          for (unsigned i = 0; i < 32; ++i)
            result->hash[i] = pow[i >> 2] >> (8 * (i & 3));
        }
      }
    });
  });
}
#endif

#ifdef MOM_SYCL_HAS_HIP
static sycl::event prehash_amd(State& s, uint64_t first, unsigned count) {
  const uint8_t* input = s.input;
  const Blake3Cv prehash_cv = s.prehash_cv;
  uint8_t* hashes = s.hashes;
  return s.queue.submit([&](sycl::handler& h) {
    h.parallel_for<AmdPrehashKernel>(
        sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const unsigned gid = id[0];
      uint32_t pow[8];
      b3_80_from_cv(pow, prehash_cv, input + 64, first + gid);
      sha3_words(pow);
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(pow);
      for (unsigned i = 0; i < 32; ++i)
        hashes[(size_t)gid * 32 + i] = bytes[i];
    });
  });
}

static sycl::event finalize_amd(State& s, uint64_t first, unsigned count, bool is_test) {
  const uint8_t* mixed = s.mixed;
  const uint8_t* target = s.target;
  Result* result = s.result;
  return s.queue.submit([&](sycl::handler& h) {
    h.parallel_for<AmdFinalKernel>(
        sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const unsigned gid = id[0];
      uint32_t pow_words[8];
      uint8_t* pow = reinterpret_cast<uint8_t*>(pow_words);
      uint32_t block_words[8];
      uint8_t* block = reinterpret_cast<uint8_t*>(block_words);
      for (unsigned i = 0; i < 32; ++i)
        block[i] = mixed[(size_t)gid * 32 + i];
      b3_32(pow_words, block_words);
      bool hit = true;
      for (int i = 31; i >= 0; --i)
        if (pow[i] != target[i]) {
          hit = pow[i] < target[i];
          break;
        }
      if ((is_test && gid == 0) || hit) {
        using A =
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>;
        if (A(result->count).fetch_add(1) == 0) {
          result->nonce = first + gid;
          for (unsigned i = 0; i < 32; ++i)
            result->hash[i] = pow[i];
        }
      }
    });
  });
}
#endif

} // namespace mom_walahash

using namespace mom_walahash;
int walahash(unsigned, uint32_t, const uint8_t* input, unsigned input_size, uint8_t* output,
             uint8_t*, uint64_t* pnonce, const uint8_t* target, const uint8_t*, unsigned intensity,
             bool is_test, bool, const std::string& dev) {
  if (!input || !output || !pnonce || !target || input_size != 80 || !intensity)
    throw std::string("walahash requires an 80-byte header and nonzero intensity");
  if (!mom::job_boundary::padded_u32_range_fits(intensity, 256))
    throw std::string("WalaHash intensity is too large");
  State& state = state_for(dev);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.queue.memcpy(state.input, input, 72);
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_HAS_HIP)
  b3_first_block_cv(state.prehash_cv, input);
#endif
  state.queue.memcpy(state.target, target, 32);
  const char* configured = std::getenv("MOM_WALAHASH_SYCL_NATIVE");
  bool native = !configured || std::strcmp(configured, "0");
#ifdef MOM_SYCL_HAS_CUDA
  // Native=true selects DP4A even in the non-fixed-hash kernel. Gate both dispatches together.
  if (mom_is_cuda(state.device) && !mom::nvidia::has_dp4a(state.cuda_sm))
    native = false;
#endif
#ifdef WALAHASH_ESIMD
  const bool intel_fast = use_esimd(state.device);
  if (intel_fast)
    state.ensure(intensity);
#endif
  unsigned fused_work_group =
      sycl_default_workgroup(state.device, {32, 64, 128, 256}, 256);
#ifdef MOM_SYCL_HAS_CUDA
  const char* cuda_required = std::getenv("MOM_WALAHASH_CUDA_REQUIRE");
  const bool cuda_require = cuda_required && *cuda_required && std::strcmp(cuda_required, "0");
  const bool cuda_device =
      state.device.is_gpu() && state.device.get_backend() == sycl::backend::ext_oneapi_cuda;
  const bool cuda_fast = native && cuda_device && sycl_workgroup_fits(state.device, 128);
  if (cuda_require && !cuda_fast)
    throw std::string("WalaHash CUDA DP4A/fixed-hash path required but unavailable");
#endif
#ifdef MOM_SYCL_HAS_HIP
  const bool amd_fast =
      native && sycl_workgroup_fits(state.device, 256) && state.amd_wmma.ensure(state.queue);
  const char* amd_required = std::getenv("MOM_WALAHASH_AMD_WMMA_REQUIRE");
  if (amd_required && *amd_required && std::strcmp(amd_required, "0") && !amd_fast)
    throw std::string("WalaHash gfx12 WMMA required but unavailable");
  if (amd_fast)
    state.ensure(intensity);
#endif
  if (!state.matrix_ready || std::memcmp(state.matrix_seed, input, 32)) {
    uint8_t matrix[4096];
    make_matrix(input, matrix);
#ifdef WALAHASH_ESIMD
    if (intel_fast) {
      uint8_t tiled[4096];
      for (unsigned r = 0; r < 64; ++r)
        for (unsigned k = 0; k < 64; ++k)
          tiled[((r >> 3) * 2 + (k >> 5)) * 256 + (r & 7) * 32 + (k & 31)] = matrix[r * 64 + k];
      sycl_wait_and_throw(state.queue.memcpy(state.matrix, tiled, sizeof(tiled)), state.device);
    } else
#endif
      sycl_wait_and_throw(state.queue.memcpy(state.matrix, matrix, sizeof(matrix)), state.device);
    std::memcpy(state.matrix_seed, input, 32);
    state.matrix_ready = true;
  }
  state.queue.memset(state.result, 0, sizeof(Result));
  sycl::event search_event;
  bool dispatched = false;
#ifdef WALAHASH_ESIMD
  if (intel_fast) {
    const unsigned width = mom::intel_matrix::dpas_width(state.device);
    prehash(state, *pnonce, intensity);
    if (width == 8)
      multiply_esimd<8>(state, intensity);
    else
      multiply_esimd<16>(state, intensity);
    search_event = finalize(state, *pnonce, intensity, is_test);
    dispatched = true;
  }
#endif
#ifdef MOM_SYCL_HAS_CUDA
  if (!dispatched && cuda_fast) {
    search_event = search<true, true>(state, *pnonce, intensity, is_test, 128);
    dispatched = true;
  }
#endif
#ifdef MOM_SYCL_HAS_HIP
  if (!dispatched && amd_fast) {
    // WMMA consumes complete 64-sample tiles. Small correctness jobs still finalize only their real
    // item count, but prehash and mix a padded tile just as the original serial path did.
    const unsigned matrix_count = state.capacity;
    prehash_amd(state, *pnonce, matrix_count);
    walahash_amd_mix(
        state.queue, state.amd_wmma, reinterpret_cast<int8_t*>(state.matrix), state.hashes,
        state.mixed, matrix_count);
    search_event = finalize_amd(state, *pnonce, intensity, is_test);
    dispatched = true;
  }
#endif
  if (!dispatched && native) {
    search_event = search<true, false>(state, *pnonce, intensity, is_test, fused_work_group);
    dispatched = true;
  }
  if (!dispatched)
    search_event = search<false, false>(state, *pnonce, intensity, is_test, fused_work_group);
  // Some runtimes busy-spin while submitting a host copy behind live kernels. Complete the GPU
  // search with the shared low-CPU wait before enqueuing the already-ready result copy.
  sycl_wait_and_throw(search_event, state.device);
  Result result;
  sycl_wait_and_throw(state.queue.memcpy(&result, state.result, sizeof(result)), state.device);
  if (!result.count)
    return 0;
  *pnonce = result.nonce;
  std::memcpy(output, result.hash, 32);
  return 1;
}
