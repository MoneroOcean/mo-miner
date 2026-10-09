// Consensus arithmetic shared with the host-only operation fixture.
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
// Require device USM for the per-nonce scratch and work buffers.
inline bool xelishashv3_supported(const sycl::device& dev) {
  return mom_has_usm_device(dev);
}
// Bound nonce capacity by both per-allocation and total device scratch-memory limits.
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
// Form an exact 32x32 product; the host fixture and device path keep identical unsigned semantics.
inline uint64_t mul_wide32(uint32_t a, uint32_t b) {
#ifdef MOM_XELISHASHV3_HOST_TEST
  return static_cast<uint64_t>(a) * b;
#else
  return mo_mul_wide_u32(a, b);
#endif
}
// Assemble the high half of a 64x64 product from 32-bit limbs without device uint128 support.
inline uint64_t mul_hi64_portable(uint64_t a, uint64_t b) {
  const uint32_t a0 = static_cast<uint32_t>(a), a1 = a >> 32,
                 b0 = static_cast<uint32_t>(b), b1 = b >> 32;
  const uint64_t w0 = mul_wide32(a0, b0), t = mul_wide32(a1, b0) + (w0 >> 32),
                 w1 = (t & 0xffffffffULL) + mul_wide32(a0, b1);
  return mul_wide32(a1, b1) + (t >> 32) + (w1 >> 32);
}
// Select exact high-product arithmetic compatible with ordinary SYCL and ESIMD.
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
// Count leading zeros with the zero-input convention required by divisor normalization.
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
// Express multiply-add in the form accepted by ordinary SYCL and explicit SIMD code generation.
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
// Select the backend reciprocal-square-root operation used by integer-square-root estimation.
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
// Scramble a scratch-derived word before indexing and operation selection.
inline uint64_t murmur(uint64_t x) {
  x ^= x >> 55;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 32;
  x *= 0xc4ceb9fe1a85ec53ULL;
  return x ^ (x >> 15);
}
// Map a mixed 64-bit word into the fixed scratch range without division.
template <bool Esimd = false> inline uint64_t map_index(uint64_t x) {
  x ^= x >> 33;
  return mul_hi64<Esimd>(x * 0xff51afd7ed558ccdULL, HALF);
}
// Estimate the integer square root with floating arithmetic, then correct to the exact floor.
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
// Divide (r<<32)|x by normalized d with r<d, correcting the estimated 32-bit quotient.
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
// Divide a two-limb numerator by a 64-bit divisor and return the low quotient limb plus remainder.
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
// Reduce by a nonzero wide divisor using the normalized reciprocal LUT and exact correction.
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
// Use the reciprocal LUT for the wide-divisor remainder; devices without fp64 use limb division.
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
// Reduce a two-limb product using a precomputed Barrett reciprocal and exact final subtraction.
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
// Evaluate modular exponentiation with overflow-safe products and the consensus zero-modulus rule.
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

// Double an AES byte in GF(2^8), reducing by the Rijndael polynomial.
inline uint8_t gf2(uint8_t x) {
  return static_cast<uint8_t>((x << 1) ^ ((x >> 7) * 0x1b));
}
// Multiply AES bytes in GF(2^8) for the scalar S-box path.
inline uint8_t gf_mul(uint8_t a, uint8_t b) {
  uint8_t r = 0;
  for (unsigned i = 0; i < 8; ++i) {
    r ^= static_cast<uint8_t>(-static_cast<int>(b & 1) & a);
    b >>= 1;
    a = gf2(a);
  }
  return r;
}
// Compute the AES substitution byte; ESIMD uses its compatible algebraic implementation.
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
// Apply one AES round to two 64-bit state halves using the consensus byte layout.
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

// Evaluate the selected consensus arithmetic/AES operation with identical backend overflow semantics.
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
