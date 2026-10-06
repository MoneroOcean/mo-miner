// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
// XelisHashV3 GPU search. Algorithm follows the MIT-licensed XELIS and TNN implementations.
//
// Performance/portability notes:
// - Stage 3 accounts for roughly 97% of measured B580 time. Each nonce follows a serial chain of
//   dependent random scratchpad reads and branch-heavy integer operations, so ordinary loop unrolling
//   cannot expose much memory-level parallelism.
// - Intel Level Zero uses a capability-gated ESIMD kernel with explicit scalar gathers and exact
//   replacements for operations unavailable in ESIMD. CPU, older iGPU, OpenCL, NVIDIA, AMD, and unsupported
//   Intel paths retain the ordinary portable SYCL kernel.
// - FP64 only accelerates exact integer division. Devices without it use integer quotient
//   estimates in the same kernel; no floating-point emulation or consensus changes are needed.
// - Cache-only hints, next-read prefetch, block load/store, subgroup-width changes, and a copied
//   NVIDIA-style software pipeline were neutral or slower in matched tests. That narrows the useful
//   direction toward overlapping independent nonces; it does not establish a hardware ceiling.
// - Replacing B580's 64-bit division helpers with exact reciprocal-based 32-bit operations removed
//   those helpers but raised Stage 3 from 5,796 to 6,579 instructions and left 2.16 KH/s unchanged.
// - Complete each per-job setup transfer before Stage 1. Windows Level Zero also needs explicit
//   ESIMD stage barriers; relying only on its in-order queue produced noncanonical results.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

#ifdef MOM_XELISHASHV3_HOST_TEST
#include <cmath>
namespace sycl {
inline int clz(uint64_t x) {
  return x ? __builtin_clzll(x) : 64;
}
inline uint64_t mul_hi(uint64_t a, uint64_t b) {
  return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) >> 64);
}
inline float fma(float a, float b, float c) {
  return std::fma(a, b, c);
}
inline double fma(double a, double b, double c) {
  return std::fma(a, b, c);
}
inline float rsqrt(float x) {
  return 1.0f / std::sqrt(x);
}
} // namespace sycl
inline uint64_t mo_rotate(uint64_t x, uint64_t n) {
  const uint64_t shift = n & 63;
  return (x << shift) | (x >> ((64 - shift) & 63));
}
#else
#include <sycl/sycl.hpp>
#if defined(PEARLHASH_ESIMD) && !defined(MOM_SYCL_ADAPTIVECPP)
// The build sets PEARLHASH_ESIMD only for its spir64-only oneAPI worker. Reuse that compilation
// capability marker here so combined spir64/NVPTX and portable OpenCL workers keep ordinary SYCL.
#define MOM_XELISHASHV3_ESIMD
#include <sycl/ext/intel/esimd.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>

#include "../lib-internal.h"
#endif

namespace mom_xelishashv3 {

#include "../blake3-device.inc"

constexpr size_t WORDS = 531 * 128, HALF = WORDS / 2, BYTES = WORDS * 8;
struct Result {
  uint32_t count;
  uint64_t nonce;
  uint8_t hash[32];
};
template <bool> class Stage1Kernel;
template <bool, bool> class Stage3Kernel;
#ifdef MOM_XELISHASHV3_ESIMD
template <bool, bool> class Stage3EsimdKernel;
namespace esimd = sycl::ext::intel::esimd;
#endif
template <bool> class Stage4Kernel;

#ifndef MOM_XELISHASHV3_HOST_TEST
inline bool xelishashv3_supported(const sycl::device& dev) {
  return mom_has_usm_device(dev);
}
inline uint64_t xelishashv3_scratch_capacity(const sycl::device& dev) {
  const uint64_t global_mem = dev.get_info<sycl::info::device::global_mem_size>();
  const uint64_t global_limit = (global_mem / 5) * 2 + (global_mem % 5) * 2 / 5;
  return std::min<uint64_t>(dev.get_info<sycl::info::device::max_mem_alloc_size>(), global_limit) /
         BYTES;
}
#endif

inline uint64_t rotl64(uint64_t x, uint32_t n) {
  return mo_rotate(x, static_cast<uint64_t>(n));
}
inline uint64_t rotr64(uint64_t x, uint32_t n) {
  return mo_rotate(x, static_cast<uint64_t>(-n));
}
inline uint64_t mul_wide32(uint32_t a, uint32_t b) {
#ifdef MOM_XELISHASHV3_HOST_TEST
  return static_cast<uint64_t>(a) * b;
#else
  return mo_mul_wide_u32(a, b);
#endif
}
inline uint64_t mul_hi64_portable(uint64_t a, uint64_t b) {
  const uint32_t a0 = static_cast<uint32_t>(a), a1 = a >> 32,
                 b0 = static_cast<uint32_t>(b), b1 = b >> 32;
  const uint64_t w0 = mul_wide32(a0, b0), t = mul_wide32(a1, b0) + (w0 >> 32),
                 w1 = (t & 0xffffffffULL) + mul_wide32(a0, b1);
  return mul_wide32(a1, b1) + (t >> 32) + (w1 >> 32);
}
template <bool Esimd = false> inline uint64_t mul_hi64(uint64_t a, uint64_t b) {
#if defined(MOM_SYCL_ADAPTIVECPP) || defined(MOM_SYCL_PORTABLE_OPENCL)
  (void)Esimd;
  return mul_hi64_portable(a, b);
#elif defined(MOM_XELISHASHV3_ESIMD)
  if constexpr (Esimd)
    return mul_hi64_portable(a, b);
  else
    return sycl::mul_hi(a, b);
#else
  return sycl::mul_hi(a, b);
#endif
}
template <bool Esimd = false> inline uint32_t clz64(uint64_t x) {
#ifdef MOM_XELISHASHV3_ESIMD
  if constexpr (Esimd) {
    if (!x)
      return 64;
    uint32_t n = 0;
    if (!(x >> 32)) {
      n += 32;
      x <<= 32;
    }
    if (!(x >> 48)) {
      n += 16;
      x <<= 16;
    }
    if (!(x >> 56)) {
      n += 8;
      x <<= 8;
    }
    if (!(x >> 60)) {
      n += 4;
      x <<= 4;
    }
    if (!(x >> 62)) {
      n += 2;
      x <<= 2;
    }
    return n + !(x >> 63);
  } else
    return sycl::clz(x);
#else
  (void)Esimd;
  return sycl::clz(x);
#endif
}
template <bool Esimd = false, typename T> inline T math_fma(T a, T b, T c) {
#ifdef MOM_XELISHASHV3_ESIMD
  if constexpr (Esimd) {
    const esimd::simd<T, 1> av = a, bv = b, cv = c;
    return (av * bv + cv)[0];
  } else
    return sycl::fma(a, b, c);
#else
  (void)Esimd;
  return sycl::fma(a, b, c);
#endif
}
template <bool Esimd = false> inline float math_rsqrt(float x) {
#ifdef MOM_XELISHASHV3_ESIMD
  if constexpr (Esimd)
    return esimd::rsqrt(x);
  else
    return sycl::rsqrt(x);
#else
  (void)Esimd;
  return sycl::rsqrt(x);
#endif
}
inline uint64_t murmur(uint64_t x) {
  x ^= x >> 55;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 32;
  x *= 0xc4ceb9fe1a85ec53ULL;
  return x ^ (x >> 15);
}
template <bool Esimd = false> inline uint64_t map_index(uint64_t x) {
  x ^= x >> 33;
  return mul_hi64<Esimd>(x * 0xff51afd7ed558ccdULL, HALF);
}
template <bool Esimd = false> inline uint64_t isqrt64(uint64_t x) {
  if (x < 2)
    return x;
  const uint64_t original = x;
  const uint32_t shift = clz64<Esimd>(x) & ~1u;
  x <<= shift;
  const uint32_t hi = x >> 32,
                 seed = static_cast<uint32_t>(math_fma<Esimd>(
                     1.407374884e14f, math_rsqrt<Esimd>(static_cast<float>(hi)), -438.0f));
  uint32_t root = static_cast<uint32_t>((static_cast<uint64_t>(seed) * hi) >> 32) * 2;
  const uint64_t rem = x - static_cast<uint64_t>(root) * root;
  root += static_cast<uint32_t>(
      (static_cast<uint64_t>(static_cast<uint32_t>(rem >> 32) + 1) * seed) >> 32);
  root >>= shift / 2;
  uint64_t square = static_cast<uint64_t>(root) * root;
  root -= square > original;
  square = static_cast<uint64_t>(root) * root;
  root += root != std::numeric_limits<uint32_t>::max() && square + 2ULL * root + 1 <= original;
  return root;
}

// Exact portable fallback for the 128/64 operations in the consensus function. The optimized
// operation selector keeps it off all cheap branches; no compiler-specific 128-bit device type.
template <bool Esimd = false>
inline void divmod64(uint64_t a, uint64_t d, uint64_t& quotient, uint64_t& remainder) {
  if (a < d) {
    quotient = 0;
    remainder = a;
    return;
  }
  if (d <= 0xffffffffULL) {
    const uint32_t dv = d, ahi = a >> 32, qhi = ahi / dv, rhi = ahi - qhi * dv;
    const uint64_t mid = static_cast<uint64_t>(rhi) << 32 | static_cast<uint32_t>(a);
    const uint32_t qlo = mid / dv;
    quotient = static_cast<uint64_t>(qhi) << 32 | qlo;
    remainder = mid - static_cast<uint64_t>(qlo) * dv;
    return;
  }
  const int shift = clz64<Esimd>(d);
  const uint32_t v = (d << shift) >> 32;
  const uint64_t top = shift ? a >> (32 - shift) : a >> 32;
  uint32_t estimate;
  if (v == 0xffffffffu)
    estimate = static_cast<uint32_t>(top >> 32);
  else
    estimate = static_cast<uint32_t>(top / (static_cast<uint64_t>(v) + 1));
  uint64_t rem = a - static_cast<uint64_t>(estimate) * d;
  if (rem > a) {
    --estimate;
    rem += d;
  } else if (rem >= d) {
    ++estimate;
    rem -= d;
  }
  // Dividing by v + 1 can leave the normalized quotient estimate two low.
  if (rem >= d) {
    ++estimate;
    rem -= d;
  }
  quotient = estimate;
  remainder = rem;
}
template <bool Esimd = false, bool Fp64 = true>
inline uint32_t div96(uint64_t r, uint32_t x, uint64_t d, double reciprocal, uint64_t& out) {
  uint64_t q;
  if constexpr (Fp64)
    q = static_cast<uint64_t>((static_cast<double>(r) * 4294967296.0 + x) * reciprocal);
  else
    // d is normalized and r < d: its high limb gives a quotient estimate at most two high.
    q = r / (d >> 32);
  if (q > 0xffffffffULL)
    q = 0xffffffffULL;
  uint64_t rem = (r << 32) | x, product = q * d;
  int64_t hi =
      static_cast<int64_t>(r >> 32) - static_cast<int64_t>(mul_hi64<Esimd>(q, d)) - (rem < product);
  rem -= product;
  const uint64_t over = hi < 0;
  q -= over;
  rem += d & -over;
  hi += over & (rem < d);
  if constexpr (!Fp64) {
    const uint64_t over_again = hi < 0;
    q -= over_again;
    rem += d & -over_again;
    hi += over_again & (rem < d);
  }
  const uint64_t under = (hi > 0) | (rem >= d);
  q += under;
  rem -= d & -under;
  out = rem;
  return static_cast<uint32_t>(q);
}
template <bool Esimd = false, bool Fp64 = true>
inline void divmod128_fast(uint64_t hi, uint64_t lo, uint64_t d, uint64_t& quotient,
                           uint64_t& remainder) {
  d += d == 0;
  uint64_t ignored, r;
  divmod64<Esimd>(hi, d, ignored, r);
  const int shift = clz64<Esimd>(d);
  const uint64_t normalized = d << shift;
  double reciprocal = 0;
  if constexpr (Fp64) {
    reciprocal = 1.0f / static_cast<float>(normalized);
    reciprocal *= math_fma<Esimd>(-static_cast<double>(normalized), reciprocal, 2.0);
  }
  const uint64_t nhi = shift ? (r << shift) | (lo >> (64 - shift)) : r, nlo = lo << shift;
  uint64_t rem;
  const uint32_t qhi = div96<Esimd, Fp64>(nhi, nlo >> 32, normalized, reciprocal, rem),
                 qlo = div96<Esimd, Fp64>(rem, nlo, normalized, reciprocal, rem);
  quotient = static_cast<uint64_t>(qhi) << 32 | qlo;
  remainder = rem >> shift;
}
template <bool Esimd = false>
inline uint64_t mod64_newton(uint64_t a, uint64_t d, const double* lut) {
  const int shift = clz64<Esimd>(d);
  const uint32_t v = (d << shift) >> 32, slot = (v >> 22) - 512;
  const uint64_t top = shift ? a >> (32 - shift) : a >> 32;
  double reciprocal = lut[slot];
  reciprocal *=
      math_fma<Esimd>(-static_cast<double>(static_cast<uint64_t>(v) + 1), reciprocal, 2.0);
  reciprocal *=
      math_fma<Esimd>(-static_cast<double>(static_cast<uint64_t>(v) + 1), reciprocal, 2.0);
  const uint32_t q = static_cast<uint32_t>(static_cast<double>(top) * reciprocal);
  uint64_t rem = a - static_cast<uint64_t>(q) * d;
  if (rem > a)
    rem += d;
  // The reciprocal estimate can be several units low.
  while (rem >= d)
    rem -= d;
  return rem;
}
template <bool Esimd = false, bool Fp64 = true>
inline void divmod128_newton(uint64_t hi, uint64_t lo, uint64_t d, const double* lut,
                             uint64_t& quotient, uint64_t& remainder) {
  if constexpr (!Fp64) {
    divmod128_fast<Esimd, false>(hi, lo, d, quotient, remainder);
  } else {
    d += d == 0;
    uint64_t ignored, r;
    if (d <= 0xffffffffULL)
      divmod64<Esimd>(hi, d, ignored, r);
    else
      r = mod64_newton<Esimd>(hi, d, lut);
    const int shift = clz64<Esimd>(d);
    const uint64_t normalized = d << shift;
    double reciprocal = 1.0f / static_cast<float>(normalized);
    reciprocal *= math_fma<Esimd>(-static_cast<double>(normalized), reciprocal, 2.0);
    const uint64_t nhi = shift ? (r << shift) | (lo >> (64 - shift)) : r, nlo = lo << shift;
    uint64_t rem;
    const uint32_t qhi = div96<Esimd>(nhi, nlo >> 32, normalized, reciprocal, rem),
                   qlo = div96<Esimd>(rem, nlo, normalized, reciprocal, rem);
    quotient = static_cast<uint64_t>(qhi) << 32 | qlo;
    remainder = rem >> shift;
  }
}
// Exact Knuth 3/2-limb division for selectors 11 and 13, without a device u128 type.
template <bool Esimd = false, bool Fp64 = true>
inline void divmod128_by128(uint64_t hi, uint64_t lo, uint64_t divisor_hi, uint64_t divisor_lo,
                            const double* lut, uint64_t& quotient, uint64_t& remainder) {
  if (!divisor_hi) {
    divmod128_newton<Esimd, Fp64>(hi, lo, divisor_lo, lut, quotient, remainder);
    return;
  }
  if (hi < divisor_hi || (hi == divisor_hi && lo < divisor_lo)) {
    quotient = 0;
    remainder = lo;
    return;
  }
  const uint32_t shift = clz64<Esimd>(divisor_hi);
  uint64_t v1 = divisor_hi, v0 = divisor_lo, u2 = 0, u1 = hi, u0 = lo;
  if (shift) {
    v1 = divisor_hi << shift | divisor_lo >> (64 - shift);
    v0 = divisor_lo << shift;
    u2 = hi >> (64 - shift);
    u1 = hi << shift | lo >> (64 - shift);
    u0 = lo << shift;
  }
  uint64_t q, r;
  divmod128_fast<Esimd, Fp64>(u2, u1, v1, q, r);
  for (unsigned correction = 0; correction < 2; ++correction) {
    const uint64_t product_lo = q * v0, product_hi = mul_hi64<Esimd>(q, v0);
    if (product_hi < r || (product_hi == r && product_lo <= u0))
      break;
    --q;
    const uint64_t previous = r;
    r += v1;
    if (r < previous)
      break;
  }
  quotient = q;
  remainder = lo - q * divisor_lo;
}
template <bool Esimd = false>
inline uint64_t barrett_reduce(uint64_t lo, uint64_t hi, uint64_t mod, uint64_t mu_lo,
                               uint64_t mu_hi) {
  const uint64_t p0h = mul_hi64<Esimd>(lo, mu_lo), p1l = lo * mu_hi,
                 p1h = mul_hi64<Esimd>(lo, mu_hi), p2l = hi * mu_lo,
                 p2h = mul_hi64<Esimd>(hi, mu_lo), p3l = hi * mu_hi;
  uint64_t mid = p0h + p1l, c1 = mid < p0h;
  const uint64_t before = mid;
  mid += p2l;
  const uint64_t q = p3l + p1h + p2h + c1 + (mid < before), qm_lo = q * mod,
                 qm_hi = mul_hi64<Esimd>(q, mod);
  uint64_t r = lo - qm_lo;
  const uint64_t rh = hi - qm_hi - (lo < qm_lo);
  if (rh || r >= mod)
    r -= mod;
  return r;
}
template <bool Esimd = false, bool Fp64 = true>
inline uint64_t modpow(uint64_t base, uint64_t exp, uint64_t mod) {
  mod += mod == 0;
  if (mod == 1)
    return 0;
  base %= mod;
  if (base < 2)
    return base;
  if (!exp)
    return 1;
  uint64_t mu_hi, top_remainder, mu_lo, ignored;
  divmod64<Esimd>(~0ULL, mod, mu_hi, top_remainder);
  divmod128_fast<Esimd, Fp64>(top_remainder, ~0ULL, mod, mu_lo, ignored);
  uint64_t out = 1;
  while (exp) {
    if (exp & 1)
      out = barrett_reduce<Esimd>(out * base, mul_hi64<Esimd>(out, base), mod, mu_lo, mu_hi);
    exp >>= 1;
    if (exp)
      base = barrett_reduce<Esimd>(base * base, mul_hi64<Esimd>(base, base), mod, mu_lo, mu_hi);
  }
  return out;
}

inline uint8_t gf2(uint8_t x) {
  return static_cast<uint8_t>((x << 1) ^ ((x >> 7) * 0x1b));
}
inline uint8_t gf_mul(uint8_t a, uint8_t b) {
  uint8_t r = 0;
  for (unsigned i = 0; i < 8; ++i) {
    r ^= static_cast<uint8_t>(-static_cast<int>(b & 1) & a);
    b >>= 1;
    a = gf2(a);
  }
  return r;
}
template <bool Esimd = false> inline uint8_t aes_sbox(uint8_t x) {
  if (!x)
    return 0x63;
  uint8_t x2 = gf_mul(x, x), x4 = gf_mul(x2, x2), x8 = gf_mul(x4, x4), x16 = gf_mul(x8, x8),
          x32 = gf_mul(x16, x16), x64 = gf_mul(x32, x32),
          y = gf_mul(gf_mul(gf_mul(gf_mul(gf_mul(gf_mul(x64, x32), x16), x8), x4), x2),
                     gf_mul(x64, x64));
#ifdef MOM_XELISHASHV3_ESIMD
  if constexpr (Esimd) {
    // Keep the affine transform explicitly 32-bit. IGC otherwise recognizes each byte expression
    // as an i8 rotate, but vISA's rol instruction does not accept UB operands. Replicating the byte
    // makes a 32-bit rotate equivalent in every byte and prevents that invalid narrowing.
    const esimd::simd<uint32_t, 1> y4 = static_cast<uint32_t>(y) * 0x01010101u;
    const auto r1 = (y4 << 1) | (y4 >> 31), r2 = (y4 << 2) | (y4 >> 30),
               r3 = (y4 << 3) | (y4 >> 29), r4 = (y4 << 4) | (y4 >> 28);
    return static_cast<uint8_t>((y4 ^ r1 ^ r2 ^ r3 ^ r4 ^ 0x63636363u)[0]);
  }
#endif
  return static_cast<uint8_t>(y | y << 8) ^ static_cast<uint8_t>((y << 1) | (y >> 7)) ^
         static_cast<uint8_t>((y << 2) | (y >> 6)) ^ static_cast<uint8_t>((y << 3) | (y >> 5)) ^
         static_cast<uint8_t>((y << 4) | (y >> 4)) ^ 0x63;
}
template <bool Esimd = false> inline void aes_round(uint64_t& lo, uint64_t& hi) {
  constexpr uint8_t key[16] = {'x', 'e', 'l', 'i', 's', 'h', 'a', 's',
                               'h', '-', 'p', 'o', 'w', '-', 'v', '3'};
  uint8_t in[16], s[16];
  for (unsigned i = 0; i < 8; ++i) {
    in[i] = lo >> (8 * i);
    in[i + 8] = hi >> (8 * i);
  }
  for (unsigned i = 0; i < 16; ++i)
    s[i] = aes_sbox<Esimd>(in[i]);
  lo = hi = 0;
  for (unsigned col = 0; col < 4; ++col) {
    const unsigned i = 4 * col;
    const uint8_t a = s[i], b = s[(i + 5) & 15], c = s[(i + 10) & 15], d = s[(i + 15) & 15];
    const uint8_t o[4] = {static_cast<uint8_t>(gf2(a) ^ (gf2(b) ^ b) ^ c ^ d ^ key[i]),
                          static_cast<uint8_t>(a ^ gf2(b) ^ (gf2(c) ^ c) ^ d ^ key[i + 1]),
                          static_cast<uint8_t>(a ^ b ^ gf2(c) ^ (gf2(d) ^ d) ^ key[i + 2]),
                          static_cast<uint8_t>((gf2(a) ^ a) ^ b ^ c ^ gf2(d) ^ key[i + 3])};
    for (unsigned j = 0; j < 4; ++j)
      if (i + j < 8)
        lo |= static_cast<uint64_t>(o[j]) << (8 * (i + j));
      else
        hi |= static_cast<uint64_t>(o[j]) << (8 * (i + j - 8));
  }
}

template <bool Esimd = false, bool Fp64 = true>
inline uint64_t operation(uint32_t op, uint64_t a, uint64_t b, uint64_t c, uint32_t r,
                          uint64_t result, uint32_t i, uint32_t j, const double* lut) {
  // These are the 16 consensus operation selectors. The masked groups keep the common arithmetic
  // cases branch-light while the two full-width division selectors stay on their exact slow path.
  if (op == 11 || op == 13) {
    const uint64_t t0 = rotl64(result, r);
    if (op == 11) {
      const uint64_t divisor_lo = a | 2;
      const bool divisor_greater = b < t0 || (b == t0 && c < divisor_lo);
      if (divisor_greater)
        return c;
      uint64_t quotient, remainder;
      divmod128_by128<Esimd, Fp64>(b, c, t0, divisor_lo, lut, quotient, remainder);
      return remainder;
    }
    const uint64_t divisor_lo = c | 8;
    const bool numerator_greater = t0 > a || (t0 == a && b > divisor_lo);
    if (!numerator_greater)
      return a ^ b;
    uint64_t quotient, remainder;
    divmod128_by128<Esimd, Fp64>(t0, b, a, divisor_lo, lut, quotient, remainder);
    return quotient;
  }
  if (op >= 3 && (op <= 9 || op >= 14)) {
    const uint64_t m3 = -static_cast<uint64_t>(op == 3), m4 = -static_cast<uint64_t>(op == 4),
                   m5 = -static_cast<uint64_t>(op == 5), m6 = -static_cast<uint64_t>(op == 6),
                   m7 = -static_cast<uint64_t>(op == 7), m8 = -static_cast<uint64_t>(op == 8),
                   m9 = -static_cast<uint64_t>(op == 9), m14 = -static_cast<uint64_t>(op == 14),
                   m15 = -static_cast<uint64_t>(op == 15), ab = a * b, ac = a * c, bc = b * c,
                   cx = (ac & (m3 | m8)) | (ab & m4) | (c & m5) | (a & m6) | (bc & m7),
                   cy = (bc & m3) | (b & m5) | (c & m6) | (a & m7) | (b & m8),
                   cz = (ac & m4) | (a & m5) | (b & m6), mhi = m14 | m15,
                   hp = (a & m14) | (c & m15), hq = (c & m14) | (b & m15),
                   hn = (bc & m14) | ((ab + c * rotr64(result, r)) & m15);
    return (cx + cy - cz) | (ab * c & m9) | ((mul_hi64<Esimd>(hp, hq) + hn) & mhi);
  }
  const uint64_t mur = murmur(c ^ result ^ i ^ j) | 1, m0 = -static_cast<uint64_t>(op == 0),
                 m1 = -static_cast<uint64_t>(op == 1), m2 = -static_cast<uint64_t>(op == 2),
                 m10 = -static_cast<uint64_t>(op == 10), m12 = -static_cast<uint64_t>(op == 12),
                 pa = m0 | m10 | m12, pb = m1, pc = m2,
                 sq0 = isqrt64<Esimd>(((b + j) & pa) | ((b | 2) & pb) | ((a + i) & pc)),
                 sq1 = isqrt64<Esimd>(((a + j) & pb) | ((c + j) & pc) | (4 & ~(pb | pc))),
                 hi = ((a + i) & m0) | (a & m10) | (c & m12) | ~pa,
                 lo = (sq0 & m0) | (b & m10) | (a & m12),
                 divisor = (mur & m0) | ((c | 1) & m10) | ((b | 4) & m12) | ~pa,
                 num = ((c + i) & m1) | ~pb, den0 = (sq0 & m1) | ~pb, den = den0 + (den0 == 0);
  uint64_t quotient, remainder;
  divmod128_newton<Esimd, Fp64>(hi & pa, (lo & pa) | (num & pb) | (1 & ~(pa | pb)),
                              (divisor & pa) | (den & pb) | (1 & ~(pa | pb)), lut, quotient,
                              remainder);
  const uint64_t ra = (remainder & (m0 | m10)) | (quotient & m12),
                 rb = rotl64(remainder, i + j) * sq1 & m1,
                 rc = (sq0 * sq1) ^ (b + i + j);
  return (ra & pa) | (rb & pb) | (rc & pc);
}

#ifndef MOM_XELISHASHV3_HOST_TEST
template <bool Native> inline uint64_t scratch_load(const uint64_t* p) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  if constexpr (Native) {
    uint64_t value;
    asm("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(p));
    return value;
  }
#else
  (void)Native;
#endif
  return *p;
}
template <bool Native> inline void scratch_store(uint64_t* p, uint64_t value) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  if constexpr (Native) {
    asm volatile("st.global.wt.u64 [%0], %1;" ::"l"(p), "l"(value));
    return;
  }
#else
  (void)Native;
#endif
  *p = value;
}

template <bool Esimd> inline uint64_t stage3_load(const uint64_t* p) {
#ifdef MOM_XELISHASHV3_ESIMD
  if constexpr (Esimd) {
    constexpr auto properties =
        esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>,
                          esimd::cache_hint_L2<esimd::cache_hint::cached>, esimd::alignment<8>};
    return esimd::gather<uint64_t, 1, 1>(p, esimd::simd<uint64_t, 1>(0), properties)[0];
  } else
    return *p;
#else
  (void)Esimd;
  return *p;
#endif
}
template <bool Esimd> inline void stage3_store(uint64_t* p, uint64_t value) {
#ifdef MOM_XELISHASHV3_ESIMD
  if constexpr (Esimd) {
    constexpr auto properties =
        esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>,
                          esimd::cache_hint_L2<esimd::cache_hint::write_back>, esimd::alignment<8>};
    esimd::scatter<uint64_t, 1, 1>(p, esimd::simd<uint64_t, 1>(0), esimd::simd<uint64_t, 1>(value),
                                   properties);
    return;
  } else
    *p = value;
#else
  (void)Esimd;
  *p = value;
#endif
}
template <bool Esimd> inline void stage3_order_writes() {
#if defined(MOM_XELISHASHV3_ESIMD) && defined(_WIN32)
  if constexpr (Esimd) {
    // Windows Level Zero has returned stale L2 data after the dependent scatters on some boots.
    // Keep the narrow group scope, but clean dirty lines before this work-item reads them again.
    esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::clean,
                 esimd::fence_scope::group>();
  }
#else
  (void)Esimd;
#endif
}

inline void chacha_block(uint32_t out[16], const uint8_t key[32], const uint8_t nonce[12],
                         uint32_t counter) {
  uint32_t x[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};
  for (unsigned i = 0; i < 8; ++i)
    x[4 + i] = load32_le_dev(key + 4 * i);
  x[12] = counter;
  for (unsigned i = 0; i < 3; ++i)
    x[13 + i] = load32_le_dev(nonce + 4 * i);
  uint32_t v[16];
  for (unsigned i = 0; i < 16; ++i)
    v[i] = x[i];
#define QR(a, b, c, d)                                                                             \
  v[a] += v[b];                                                                                    \
  v[d] = mo_rotate(v[d] ^ v[a], 16u);                                                              \
  v[c] += v[d];                                                                                    \
  v[b] = mo_rotate(v[b] ^ v[c], 12u);                                                              \
  v[a] += v[b];                                                                                    \
  v[d] = mo_rotate(v[d] ^ v[a], 8u);                                                               \
  v[c] += v[d];                                                                                    \
  v[b] = mo_rotate(v[b] ^ v[c], 7u)
  for (unsigned round = 0; round < 4; ++round) {
    QR(0, 4, 8, 12);
    QR(1, 5, 9, 13);
    QR(2, 6, 10, 14);
    QR(3, 7, 11, 15);
    QR(0, 5, 10, 15);
    QR(1, 6, 11, 12);
    QR(2, 7, 8, 13);
    QR(3, 4, 9, 14);
  }
#undef QR
  for (unsigned i = 0; i < 16; ++i)
    out[i] = v[i] + x[i];
}
// Stage 1 derives one nonce's scratchpad with BLAKE3-keyed ChaCha blocks.
inline void stage1(const uint8_t source[112], uint64_t* scratch, uint64_t nonce) {
  uint8_t input[112], buffer[64], key[32], chain_nonce[12];
  for (unsigned i = 0; i < 112; ++i)
    input[i] = source[i];
  for (unsigned i = 0; i < 8; ++i)
    input[40 + i] = nonce >> (8 * (7 - i));
  blake3_dev(buffer, 32, input, 112);
  for (unsigned i = 0; i < 12; ++i)
    chain_nonce[i] = buffer[i];
  uint32_t* dst = reinterpret_cast<uint32_t*>(scratch);
  constexpr unsigned blocks = (BYTES / 4) / 64;
  for (unsigned chunk = 0; chunk < 4; ++chunk) {
    if (chunk)
      for (unsigned i = 0; i < 32; ++i)
        buffer[i] = key[i];
    for (unsigned i = 0; i < 32; ++i)
      buffer[32 + i] = chunk * 32 + i < 112 ? input[chunk * 32 + i] : 0;
    blake3_dev(key, 32, buffer, 64);
    for (unsigned block = 0; block < blocks; ++block) {
      uint32_t words[16];
      chacha_block(words, key, chain_nonce, block);
      for (unsigned i = 0; i < 16; ++i)
        *dst++ = words[i];
      if (block + 1 == blocks)
        for (unsigned i = 0; i < 12; ++i)
          chain_nonce[i] = static_cast<uint8_t>(words[13 + i / 4] >> (8 * (i & 3)));
    }
  }
}
template <bool Native, bool Esimd = false, bool Fp64 = true>
inline void stage3(uint64_t* scratch, const double* lut) {
  // Stage 3 is the consensus-critical dependent walk. NVPTX overlaps one iteration's reads/writes;
  // ESIMD and portable SYCL share the straight-line ordering because that pipeline did not help B580.
  uint64_t* a_mem = scratch;
  uint64_t* b_mem = scratch + HALF;
  uint64_t addr_a = stage3_load<Esimd>(b_mem + HALF - 1),
           addr_b = stage3_load<Esimd>(a_mem + HALF - 1) >> 32;
  uint32_t r = 0;
  for (uint32_t i = 0; i < 2; ++i) {
    uint64_t ma = stage3_load<Esimd>(a_mem + map_index<Esimd>(addr_a)),
             mb = stage3_load<Esimd>(b_mem + map_index<Esimd>(ma ^ addr_b));
    aes_round<Esimd>(mb, ma);
    uint64_t result = ~(mb ^ ma);
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
    uint64_t read_a = map_index<Esimd>(result), next_a = scratch_load<Native>(a_mem + read_a),
             write_a = HALF + 1, write_b = HALF + 1, pending_t = 0;
    uint32_t pending_ij = 0;
    bool pending = false;
    for (uint32_t j = 0; j < HALF; ++j) {
      const uint64_t a = pending && write_a == read_a ? pending_t : next_a,
                     baddr = map_index<Esimd>(a ^ ~rotr64(result, r));
      uint64_t b = scratch_load<Native>(b_mem + baddr), old_a = 0, old_b = 0, pending_b = 0,
               c = scratch_load<Native>(scratch + r);
      if (pending) {
        old_a = scratch_load<Native>(a_mem + write_a);
        old_b = scratch_load<Native>(b_mem + write_b);
        pending_b = old_b ^ old_a ^ rotr64(pending_t, pending_ij);
        if (write_a == r)
          c = pending_t;
        else if (write_b + HALF == r)
          c = pending_b;
      }
      r = r < WORDS - 1 ? r + 1 : 0;
      const uint32_t op = rotl64(result, static_cast<uint32_t>(c)) & 15;
      if (pending) {
        if (write_b == baddr)
          b = pending_b;
        scratch_store<Native>(b_mem + write_b, pending_b);
        scratch_store<Native>(a_mem + write_a, pending_t);
      }
      const uint64_t v = operation<Esimd, Fp64>(op, a, b, c, r, result, i, j, lut);
      const uint64_t seed = v ^ result;
      result = rotl64(seed, r);
      const uint64_t idx = map_index<Esimd>(seed);
      read_a = map_index<Esimd>(result);
      next_a = scratch_load<Native>(a_mem + read_a);
      pending_t = (murmur(v) & (1ULL << 58) ? scratch_load<Native>(b_mem + idx)
                                            : scratch_load<Native>(a_mem + idx)) ^
                  result;
      write_a = map_index<Esimd>(pending_t ^ result ^ 0x9e3779b97f4a7c15ULL);
      write_b = map_index<Esimd>(write_a ^ ~result ^ 0xd2b74407b1ce6e93ULL);
      pending_ij = i + j;
      pending = true;
    }
    if (pending) {
      const uint64_t old_a = scratch_load<Native>(a_mem + write_a),
                     old_b = scratch_load<Native>(b_mem + write_b);
      scratch_store<Native>(b_mem + write_b, old_b ^ old_a ^ rotr64(pending_t, pending_ij));
      scratch_store<Native>(a_mem + write_a, pending_t);
    }
#else
    for (uint32_t j = 0; j < HALF; ++j) {
      const uint64_t a = stage3_load<Esimd>(a_mem + map_index<Esimd>(result)),
                     b = stage3_load<Esimd>(b_mem + map_index<Esimd>(a ^ ~rotr64(result, r))),
                     c = stage3_load<Esimd>(scratch + r);
      r = r < WORDS - 1 ? r + 1 : 0;
      const uint64_t v = operation<Esimd, Fp64>(rotl64(result, static_cast<uint32_t>(c)) & 15,
                                               a, b, c, r, result, i, j, lut),
                     seed = v ^ result;
      result = rotl64(seed, r);
      const uint64_t idx = map_index<Esimd>(seed),
                     t = (murmur(v) & (1ULL << 58) ? stage3_load<Esimd>(b_mem + idx)
                                                   : stage3_load<Esimd>(a_mem + idx)) ^
                         result,
                     ia = map_index<Esimd>(t ^ result ^ 0x9e3779b97f4a7c15ULL),
                     ib = map_index<Esimd>(ia ^ ~result ^ 0xd2b74407b1ce6e93ULL),
                     old = stage3_load<Esimd>(a_mem + ia), old_b = stage3_load<Esimd>(b_mem + ib);
      stage3_store<Esimd>(a_mem + ia, t);
      stage3_store<Esimd>(b_mem + ib, old_b ^ old ^ rotr64(t, i + j));
      // Only this work-item consumes the dependent stores; avoid a system-wide commit fence.
      stage3_order_writes<Esimd>();
    }
#endif
    addr_a = modpow<Esimd, Fp64>(addr_a, addr_b, result);
    addr_b = isqrt64<Esimd>(result) * (r + 1) * isqrt64<Esimd>(addr_a);
  }
}

inline void b3_compress_words(uint32_t cv[8], const uint32_t* m, uint64_t counter, uint32_t flags) {
  uint32_t st[16] = {cv[0],
                     cv[1],
                     cv[2],
                     cv[3],
                     cv[4],
                     cv[5],
                     cv[6],
                     cv[7],
                     B3_IV[0],
                     B3_IV[1],
                     B3_IV[2],
                     B3_IV[3],
                     static_cast<uint32_t>(counter),
                     static_cast<uint32_t>(counter >> 32),
                     64,
                     flags};
  for (unsigned round = 0; round < 7; ++round)
    b3_round(st, m, round);
  for (unsigned i = 0; i < 8; ++i)
    cv[i] = st[i] ^ st[i + 8];
}
inline void b3_parent(const uint32_t left[8], const uint32_t right[8], uint32_t out[8], bool root) {
  uint32_t m[16], cv[8];
  for (unsigned i = 0; i < 8; ++i) {
    cv[i] = B3_IV[i];
    m[i] = left[i];
    m[i + 8] = right[i];
  }
  b3_compress_words(cv, m, 0, 4u | (root ? 8u : 0u));
  for (unsigned i = 0; i < 8; ++i)
    out[i] = cv[i];
}
// Stage 4 reduces the scratchpad through the algorithm's BLAKE3 tree and emits the final hash.
inline void stage4(const uint64_t* scratch, uint8_t hash[32]) {
  const uint32_t* words = reinterpret_cast<const uint32_t*>(scratch);
  uint32_t stack[10][8];
  unsigned n = 0;
  for (uint64_t chunk = 0; chunk < 531; ++chunk) {
    uint32_t cv[8];
    for (unsigned i = 0; i < 8; ++i)
      cv[i] = B3_IV[i];
    for (unsigned block = 0; block < 16; ++block)
      b3_compress_words(cv, words + chunk * 256 + block * 16, chunk,
                        (block == 0 ? 1u : 0u) | (block == 15 ? 2u : 0u));
    for (unsigned i = 0; i < 8; ++i)
      stack[n][i] = cv[i];
    ++n;
    for (uint64_t merges = chunk + 1; !(merges & 1) && n > 1; merges >>= 1) {
      uint32_t parent[8];
      b3_parent(stack[n - 2], stack[n - 1], parent, false);
      --n;
      for (unsigned i = 0; i < 8; ++i)
        stack[n - 1][i] = parent[i];
    }
  }
  while (n > 2) {
    uint32_t parent[8];
    b3_parent(stack[n - 2], stack[n - 1], parent, false);
    --n;
    for (unsigned i = 0; i < 8; ++i)
      stack[n - 1][i] = parent[i];
  }
  uint32_t root[8];
  b3_parent(stack[0], stack[1], root, true);
  for (unsigned i = 0; i < 8; ++i)
    store32_le_dev(hash + 4 * i, root[i]);
}

class State {
public:
  sycl::device device;
  sycl::queue queue;
  uint8_t *input = nullptr, *target = nullptr;
  uint64_t* scratch = nullptr;
  double* lut = nullptr;
  Result* result = nullptr;
  unsigned capacity = 0;
  std::mutex mutex;
  const bool profile;
  unsigned profile_reports = 0;
  explicit State(const std::string& dev)
      : device(get_dev(dev)),
        queue(device,
              sycl::async_handler([](sycl::exception_list errors) {
                for (const std::exception_ptr& error : errors)
                  std::rethrow_exception(error);
              }),
              sycl::property_list{sycl::property::queue::in_order{}}),
        profile([] {
          const char* configured = std::getenv("MOM_XELISHASHV3_PROFILE");
          return configured && *configured && std::strcmp(configured, "0") != 0;
        }()) {
    try {
      if (!xelishashv3_supported(device))
        throw std::string("xelishashv3 requires SYCL device USM");
      input = sycl::malloc_device<uint8_t>(112, queue);
      target = sycl::malloc_device<uint8_t>(32, queue);
      result = sycl::malloc_device<Result>(1, queue);
      if (!input || !target || !result)
        throw std::string("Can't allocate xelishashv3 buffers");
      if (device.has(sycl::aspect::fp64)) {
        lut = sycl::malloc_device<double>(512, queue);
        if (!lut)
          throw std::string("Can't allocate xelishashv3 reciprocal table");
        double host[512];
        for (unsigned i = 0; i < 512; ++i)
          host[i] = 1.0 / ((512.5 + i) * (1u << 22));
        sycl_wait_and_throw(queue.memcpy(lut, host, sizeof(host)), device);
      }
    } catch (...) {
      sycl_cleanup_noexcept("xelishashv3 constructor", [&] {
        cleanup();
      });
      throw;
    }
  }
  ~State() {
    sycl_cleanup_noexcept("xelishashv3", [&] {
      cleanup();
    });
  }
  void free_all() {
    if (lut) {
      sycl::free(lut, queue);
      lut = nullptr;
    }
    if (scratch) {
      sycl::free(scratch, queue);
      scratch = nullptr;
    }
    if (result) {
      sycl::free(result, queue);
      result = nullptr;
    }
    if (target) {
      sycl::free(target, queue);
      target = nullptr;
    }
    if (input) {
      sycl::free(input, queue);
      input = nullptr;
    }
    capacity = 0;
  }
  void cleanup() {
    try {
      queue.wait_and_throw();
    } catch (...) {
      free_all();
      throw;
    }
    free_all();
  }
  void ensure(unsigned count) {
    if (static_cast<uint64_t>(count) > xelishashv3_scratch_capacity(device))
      throw std::string("xelishashv3 intensity exceeds device memory limit");
    const size_t max_count = std::numeric_limits<size_t>::max() / WORDS;
    if (static_cast<uint64_t>(count) > max_count)
      throw std::string("xelishashv3 intensity is too large");
    const size_t count_size = static_cast<size_t>(count);
    if (count <= capacity)
      return;
    queue.wait_and_throw();
    uint64_t* replacement = sycl::malloc_device<uint64_t>(count_size * WORDS, queue);
    if (!replacement)
      throw std::string("Can't allocate xelishashv3 scratchpads");
    uint64_t* old_scratch = scratch;
    scratch = replacement;
    capacity = count;
    if (old_scratch)
      sycl::free(old_scratch, queue);
  }
};
static DeviceStateRegistry<State>& registry() {
  static auto* const value = new DeviceStateRegistry<State>;
  return *value;
}
static State& state_for(const std::string& dev) {
  return registry().get(dev, [&] { return std::make_unique<State>(dev); });
}
void xelishashv3_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "xelishashv3: ordered SYCL cleanup failed\n");
  }
}

template <bool Native>
static void search(State& s, uint64_t first, unsigned count, bool test, bool use_esimd = false) {
  const uint8_t* input = s.input;
  uint64_t* scratch = s.scratch;
  const uint8_t* target = s.target;
  Result* result = s.result;
  const double* lut = s.lut;
  using Clock = std::chrono::steady_clock;
  const auto total_start = s.profile ? Clock::now() : Clock::time_point{};
  // Profiling needs per-stage barriers for meaningful timings. Portable OpenCL also completes one
  // kernel at a time because some runtimes reject the full queued chain during kernel setup.
  // Windows Level Zero requires the same barriers around ESIMD despite the queue being in order.
  const auto run_stage = [&](const char* stage, bool required, const auto& submit) {
#ifdef MOM_SYCL_PORTABLE_OPENCL
    required = true;
#endif
#ifdef _WIN32
    required = required || use_esimd;
#endif
    try {
      const sycl::event event = submit();
      if (required || s.profile)
        sycl_wait_and_throw(event, s.device);
    } catch (...) {
      std::fprintf(stderr, "XelisHashV3 %s failed\n", stage);
      throw;
    }
  };
  const auto stage1_start = s.profile ? Clock::now() : Clock::time_point{};
  run_stage("stage 1", false, [&] {
    return s.queue.submit([&](sycl::handler& h) {
      h.parallel_for<Stage1Kernel<Native>>(
          sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        stage1(input, scratch + id[0] * WORDS, first + id[0]);
      });
    });
  });
  const auto stage1_end = s.profile ? Clock::now() : Clock::time_point{};
  const auto stage3_start = s.profile ? Clock::now() : Clock::time_point{};
  // Specialize at dispatch, so the no-FP64 device kernel contains no double arithmetic.
  const auto submit_stage3 = [&]<bool Fp64>() {
#ifdef MOM_XELISHASHV3_ESIMD
    if constexpr (Native) {
      if (use_esimd) {
        run_stage("stage 3", false, [&] {
          return s.queue.submit([&](sycl::handler& h) {
            h.parallel_for<Stage3EsimdKernel<Native, Fp64>>(
                sycl::range<1>(count),
                [=](sycl::id<1> id) SYCL_ESIMD_KERNEL MOM_SYCL_KERNEL_ARGS_RESTRICT {
                  stage3<Native, true, Fp64>(scratch + id[0] * WORDS, lut);
                });
          });
        });
        return;
      }
    }
#endif
    run_stage("stage 3", false, [&] {
      return s.queue.submit([&](sycl::handler& h) {
        h.parallel_for<Stage3Kernel<Native, Fp64>>(
            sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          stage3<Native, false, Fp64>(scratch + id[0] * WORDS, lut);
        });
      });
    });
  };
  if (lut)
    submit_stage3.template operator()<true>();
  else
    submit_stage3.template operator()<false>();
  const auto stage3_end = s.profile ? Clock::now() : Clock::time_point{};
  const auto stage4_start = s.profile ? Clock::now() : Clock::time_point{};
  run_stage("stage 4", true, [&] {
    return s.queue.submit([&](sycl::handler& h) {
      h.parallel_for<Stage4Kernel<Native>>(
          sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        uint8_t hash[32];
        stage4(scratch + id[0] * WORDS, hash);
        bool hit = true;
        for (unsigned i = 0; i < 32; ++i)
          if (hash[i] != target[i]) {
            hit = hash[i] < target[i];
            break;
          }
        if (test && id[0] == 0) {
          result->count = 1;
          result->nonce = first;
          for (unsigned i = 0; i < 32; ++i)
            result->hash[i] = hash[i];
        } else if (!test && hit) {
          using Atomic =
              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                               sycl::access::address_space::global_space>;
          if (Atomic(result->count).fetch_add(1) == 0) {
            result->nonce = first + id[0];
            for (unsigned i = 0; i < 32; ++i)
              result->hash[i] = hash[i];
          }
        }
      });
    });
  });
  const auto stage4_end = s.profile ? Clock::now() : Clock::time_point{};
  const auto total_end = s.profile ? Clock::now() : Clock::time_point{};
  if (s.profile && s.profile_reports++ < 8) {
    const double wall_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start).count();
    const double stage1_ms =
        std::chrono::duration<double, std::milli>(stage1_end - stage1_start).count();
    const double stage3_ms =
        std::chrono::duration<double, std::milli>(stage3_end - stage3_start).count();
    const double stage4_ms =
        std::chrono::duration<double, std::milli>(stage4_end - stage4_start).count();
    std::fprintf(
        stderr,
        "XelisHashV3 profile count=%u native=%u esimd=%u wall_ms=%.3f stage_ms=%.3f,%.3f,%.3f\n",
        s.profile_reports, Native ? 1u : 0u, use_esimd ? 1u : 0u, wall_ms, stage1_ms, stage3_ms,
        stage4_ms);
  }
}

} // namespace mom_xelishashv3

using namespace mom_xelishashv3;
int xelishashv3(unsigned, uint32_t, const uint8_t* input, unsigned input_size, uint8_t* output,
                uint8_t*, uint64_t* nonce, const uint8_t* target, const uint8_t*,
                unsigned intensity, bool is_test, bool, const std::string& dev) try {
  if (!input || !output || !nonce || !target || input_size != 112 || !intensity)
    throw std::string(
      "xelishashv3 requires input, output, nonce, target, a 112-byte header, and nonzero intensity");
  State& s = state_for(dev);
  std::lock_guard<std::mutex> lock(s.mutex);
  s.ensure(intensity);
  sycl_wait_and_throw(s.queue.memcpy(s.input, input, 112), s.device);
  sycl_wait_and_throw(s.queue.memcpy(s.target, target, 32), s.device);
  sycl_wait_and_throw(s.queue.memset(s.result, 0, sizeof(Result)), s.device);
  const char* configured = std::getenv("MOM_XELISHASHV3_SYCL_NATIVE");
  const bool native = !configured || std::strcmp(configured, "0");
#ifdef MOM_XELISHASHV3_ESIMD
  // The cached ESIMD gathers require newer Intel hardware; XeLP lacks those LSC instructions.
  // The matrix capability keeps these older devices on the ordinary kernel without a SKU list.
  const bool use_esimd = native && sycl_is_level_zero_gpu(s.device) &&
                         s.device.has(sycl::aspect::ext_intel_matrix);
#else
  const bool use_esimd = false;
#endif
  if (native)
    search<true>(s, *nonce, intensity, is_test, use_esimd);
  else
    search<false>(s, *nonce, intensity, is_test);
  Result found;
  sycl_wait_and_throw(s.queue.memcpy(&found, s.result, sizeof(found)), s.device);
  if (!found.count)
    return 0;
  *nonce = found.nonce;
  std::memcpy(output, found.hash, 32);
  return 1;
} catch (const sycl::exception& error) {
  std::fprintf(stderr, "XelisHashV3 SYCL error: %s\n", error.what());
  throw;
}
#else
} // namespace mom_xelishashv3
#endif
