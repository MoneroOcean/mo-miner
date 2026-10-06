// Portable NexaPow/Echelon consensus kernel.  The arithmetic is deliberately standards-only:
// 32-bit limbs, 64-bit products, and Jacobian secp256k1 points.  It is a correctness baseline for
// every SYCL backend; vendor-specific field-multiply tuning can be added without changing this API.

#ifndef MOM_NEXAPOW_CONSTANT
#define MOM_NEXAPOW_CONSTANT static constexpr
#endif

#ifndef MOM_NEXAPOW_SHA_ONLY
struct U256 {
  uint32_t w[8];
};

static constexpr U256 kFieldPrime{{
    0xfffffc2fu,
    0xfffffffeu,
    0xffffffffu,
    0xffffffffu,
    0xffffffffu,
    0xffffffffu,
    0xffffffffu,
    0xffffffffu,
}};
MOM_NEXAPOW_CONSTANT U256 kGroupOrder{{
    0xd0364141u,
    0xbfd25e8cu,
    0xaf48a03bu,
    0xbaaedce6u,
    0xfffffffeu,
    0xffffffffu,
    0xffffffffu,
    0xffffffffu,
}};
static constexpr U256 kGeneratorX{{
    0x16f81798u,
    0x59f2815bu,
    0x2dce28d9u,
    0x029bfcdbu,
    0xce870b07u,
    0x55a06295u,
    0xf9dcbbacu,
    0x79be667eu,
}};
static constexpr U256 kGeneratorY{{
    0xfb10d4b8u,
    0x9c47d08fu,
    0xa6855419u,
    0xfd17b448u,
    0x0e1108a8u,
    0x5da4fbfcu,
    0x26a3c465u,
    0x483ada77u,
}};

#include "generator_lut.inc"

inline U256 np_zero() {
  return U256{};
}
inline U256 np_one() {
  U256 result{};
  result.w[0] = 1;
  return result;
}

inline int np_cmp(const U256& a, const U256& b) {
  for (int i = 7; i >= 0; --i) {
    if (a.w[i] != b.w[i])
      return a.w[i] < b.w[i] ? -1 : 1;
  }
  return 0;
}

inline bool np_zero_p(const U256& a) {
  uint32_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= a.w[i];
  return value == 0;
}

inline U256 np_from_be(const uint8_t* bytes) {
  U256 result{};
  for (unsigned i = 0; i < 8; ++i) {
    const unsigned j = 31u - 4u * i;
    result.w[i] = static_cast<uint32_t>(bytes[j]) | (static_cast<uint32_t>(bytes[j - 1]) << 8) |
                  (static_cast<uint32_t>(bytes[j - 2]) << 16) |
                  (static_cast<uint32_t>(bytes[j - 3]) << 24);
  }
  return result;
}

inline void np_to_be(const U256& value, uint8_t* bytes) {
  for (unsigned i = 0; i < 8; ++i) {
    const unsigned j = 31u - 4u * i;
    bytes[j] = static_cast<uint8_t>(value.w[i]);
    bytes[j - 1] = static_cast<uint8_t>(value.w[i] >> 8);
    bytes[j - 2] = static_cast<uint8_t>(value.w[i] >> 16);
    bytes[j - 3] = static_cast<uint8_t>(value.w[i] >> 24);
  }
}

inline uint32_t np_add_raw(const U256& a, const U256& b, U256& out) {
  uint64_t carry = 0;
  for (unsigned i = 0; i < 8; ++i) {
    const uint64_t sum = static_cast<uint64_t>(a.w[i]) + b.w[i] + carry;
    out.w[i] = static_cast<uint32_t>(sum);
    carry = sum >> 32;
  }
  return static_cast<uint32_t>(carry);
}

inline uint32_t np_sub_raw(const U256& a, const U256& b, U256& out) {
  uint32_t borrow = 0;
  for (unsigned i = 0; i < 8; ++i) {
    const uint32_t ai = a.w[i];
    const uint64_t subtrahend = static_cast<uint64_t>(b.w[i]) + borrow;
    const uint64_t difference = static_cast<uint64_t>(ai) - subtrahend;
    out.w[i] = static_cast<uint32_t>(difference);
    borrow = ai < subtrahend;
  }
  return borrow;
}

inline U256 np_add_mod(const U256& a, const U256& b, const U256& modulus) {
  U256 result{};
  if (np_add_raw(a, b, result) || np_cmp(result, modulus) >= 0)
    np_sub_raw(result, modulus, result);
  return result;
}

inline U256 np_sub_mod(const U256& a, const U256& b, const U256& modulus) {
  U256 result{};
  if (np_cmp(a, b) >= 0) {
    np_sub_raw(a, b, result);
  } else {
    U256 difference{};
    np_sub_raw(b, a, difference);
    np_sub_raw(modulus, difference, result);
  }
  return result;
}

MOM_NEXAPOW_CONSTANT U256 kGroupR2{{
    0x67d7d140u,
    0x896cf214u,
    0x0e7cf878u,
    0x741496c2u,
    0x5bcd07c6u,
    0xe697f5e4u,
    0x81c69bc5u,
    0x9d671cd5u,
}};

// Coarsely integrated operand-scanning Montgomery multiplication for the fixed
// secp256k1 group order.  The ten normalized 32-bit limbs avoid a device 128-bit type.
inline U256 np_group_montgomery_mul(const U256& a, const U256& b) {
  constexpr uint32_t n0_inverse = 0x5588b13fu;
  uint64_t t[10]{};
  for (unsigned i = 0; i < 8; ++i) {
    uint64_t carry = 0;
    for (unsigned j = 0; j < 8; ++j) {
      const uint64_t sum = t[j] + static_cast<uint64_t>(a.w[i]) * b.w[j] + carry;
      t[j] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    uint64_t sum = t[8] + carry;
    t[8] = static_cast<uint32_t>(sum);
    t[9] += sum >> 32;

    const uint32_t multiplier = static_cast<uint32_t>(t[0]) * n0_inverse;
    carry = 0;
    for (unsigned j = 0; j < 8; ++j) {
      sum = t[j] + static_cast<uint64_t>(multiplier) * kGroupOrder.w[j] + carry;
      t[j] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    sum = t[8] + carry;
    t[8] = static_cast<uint32_t>(sum);
    t[9] += sum >> 32;
    for (unsigned j = 0; j < 9; ++j)
      t[j] = t[j + 1];
    t[9] = 0;
  }
  U256 result{};
  for (unsigned i = 0; i < 8; ++i)
    result.w[i] = static_cast<uint32_t>(t[i]);
  if (t[8] || np_cmp(result, kGroupOrder) >= 0)
    np_sub_raw(result, kGroupOrder, result);
  return result;
}

inline U256 np_mul_mod(const U256& a, const U256& b, const U256& modulus) {
  (void)modulus;
  return np_group_montgomery_mul(np_group_montgomery_mul(a, b), kGroupR2);
}

inline U256 np_reduce_mod(const U256& value, const U256& modulus) {
  U256 result = value;
  if (np_cmp(result, modulus) >= 0)
    np_sub_raw(result, modulus, result);
  return result;
}

inline void np_comba_add_product(uint64_t& low, uint64_t& high, const uint32_t a,
                                 const uint32_t b) {
  const uint64_t product = static_cast<uint64_t>(a) * b;
  const uint64_t old_low = low;
  low += product;
  high += static_cast<uint64_t>(low < old_low);
}

inline uint64_t np_comba_carry(const uint64_t low, const uint64_t high) {
  return (low >> 32) | (high << 32);
}

// A fixed 8x8 Comba multiply.  The pair of accumulators is a small software 128-bit value: low
// holds the low 64 bits and high counts carries out of it.  This is device-portable and avoids
// both variable limb walks and a dependency on a vendor-specific 128-bit integer type.
inline void np_comba_mul(const U256& a, const U256& b, uint32_t product[16]) {
  uint64_t carry = 0;
  for (unsigned column = 0; column < 16; ++column) {
    uint64_t low = carry, high = 0;
    for (unsigned row = 0; row < 8; ++row) {
      const int other = static_cast<int>(column) - static_cast<int>(row);
      if (other >= 0 && other < 8)
        np_comba_add_product(low, high, a.w[row], b.w[other]);
    }
    product[column] = static_cast<uint32_t>(low);
    carry = np_comba_carry(low, high);
  }
}

inline void np_comba_square(const U256& a, uint32_t product[16]) {
  uint64_t carry = 0;
  for (unsigned column = 0; column < 16; ++column) {
    uint64_t low = carry, high = 0;
    for (unsigned row = 0; row < 8; ++row) {
      const int other = static_cast<int>(column) - static_cast<int>(row);
      if (other >= static_cast<int>(row) && other < 8) {
        np_comba_add_product(low, high, a.w[row], a.w[other]);
        if (other != static_cast<int>(row))
          np_comba_add_product(low, high, a.w[row], a.w[other]);
      }
    }
    product[column] = static_cast<uint32_t>(low);
    carry = np_comba_carry(low, high);
  }
}

// secp256k1's pseudo-Mersenne prime makes the field reduction substantially smaller than a
// generic 512/256 division: 2^256 == 2^32 + 977 (mod p).  Two fixed folds cover the carry that
// the first fold can create above bit 256; the final fixed subtraction count is sufficient for a
// product of two canonical field elements.
inline U256 np_field_reduce(const uint32_t product[16]) {
  uint64_t limbs[10]{};
  for (unsigned i = 0; i < 8; ++i) {
    limbs[i] += product[i] + static_cast<uint64_t>(product[i + 8]) * 977u;
    limbs[i + 1] += product[i + 8];
  }
  auto normalize = [&] {
    for (unsigned i = 0; i < 9; ++i) {
      limbs[i + 1] += limbs[i] >> 32;
      limbs[i] = static_cast<uint32_t>(limbs[i]);
    }
  };
  normalize();
  for (unsigned fold = 0; fold < 2; ++fold) {
    const uint64_t high0 = limbs[8], high1 = limbs[9];
    limbs[8] = limbs[9] = 0;
    limbs[0] += high0 * 977u;
    limbs[1] += high0 + high1 * 977u;
    limbs[2] += high1;
    normalize();
  }
  U256 result{};
  for (unsigned i = 0; i < 8; ++i)
    result.w[i] = static_cast<uint32_t>(limbs[i]);
  if (np_cmp(result, kFieldPrime) >= 0)
    np_sub_raw(result, kFieldPrime, result);
  return result;
}

inline U256 np_field_mul(const U256& a, const U256& b) {
  uint32_t product[16];
  np_comba_mul(a, b, product);
  return np_field_reduce(product);
}

inline U256 np_field_square(const U256& a) {
  uint32_t product[16];
  np_comba_square(a, product);
  return np_field_reduce(product);
}
inline U256 np_field_add(const U256& a, const U256& b) {
  return np_add_mod(a, b, kFieldPrime);
}
inline U256 np_field_sub(const U256& a, const U256& b) {
  return np_sub_mod(a, b, kFieldPrime);
}

struct Point {
  U256 x, y, z;
  bool infinity;
};

inline Point np_infinity() {
  return {np_zero(), np_zero(), np_zero(), true};
}

inline Point np_double(const Point& point) {
  if (point.infinity || np_zero_p(point.y))
    return np_infinity();
  const U256 yy = np_field_square(point.y);
  U256 s = np_field_mul(point.x, yy);
  s = np_field_add(s, s);
  s = np_field_add(s, s); // S = 4*X*Y^2
  U256 m = np_field_square(point.x);
  m = np_field_add(m, np_field_add(m, m)); // M = 3*X^2
  U256 x = np_field_sub(np_field_sub(np_field_square(m), s), s);
  U256 yyyy = np_field_square(yy);
  U256 eight_yyyy = np_field_add(yyyy, yyyy);
  eight_yyyy = np_field_add(eight_yyyy, eight_yyyy);
  eight_yyyy = np_field_add(eight_yyyy, eight_yyyy);
  const U256 y = np_field_sub(np_field_mul(m, np_field_sub(s, x)), eight_yyyy);
  const U256 z = np_field_add(np_field_mul(point.y, point.z), np_field_mul(point.y, point.z));
  return {x, y, z, false};
}

// Add an affine point to a Jacobian point.  Scalar multiplication always adds the fixed generator,
// which keeps this path smaller than a general Jacobian/Jacobian addition.
inline Point np_add_affine(const Point& point, const U256& ax, const U256& ay) {
  if (point.infinity)
    return {ax, ay, np_one(), false};
  const U256 z2 = np_field_square(point.z);
  const U256 u2 = np_field_mul(ax, z2);
  const U256 s2 = np_field_mul(ay, np_field_mul(point.z, z2));
  const U256 h = np_field_sub(u2, point.x);
  const U256 r = np_field_sub(s2, point.y);
  if (np_zero_p(h))
    return np_zero_p(r) ? np_double(point) : np_infinity();
  const U256 hh = np_field_square(h);
  const U256 hhh = np_field_mul(h, hh);
  const U256 two_x_hh = np_field_add(np_field_mul(point.x, hh), np_field_mul(point.x, hh));
  U256 x = np_field_sub(np_field_sub(np_field_square(r), hhh), two_x_hh);
  const U256 y = np_field_sub(np_field_mul(r, np_field_sub(np_field_mul(point.x, hh), x)),
                              np_field_mul(point.y, hhh));
  const U256 z = np_field_mul(point.z, h);
  return {x, y, z, false};
}

inline Point np_scalar_mul(const U256& scalar) {
  Point result = np_infinity();
  for (int bit = 255; bit >= 0; --bit) {
    result = np_double(result);
    if ((scalar.w[static_cast<unsigned>(bit) >> 5] >> (bit & 31)) & 1u)
      result = np_add_affine(result, kGeneratorX, kGeneratorY);
  }
  return result;
}

inline unsigned np_window_value(const U256& scalar, const unsigned low, const unsigned width) {
  const unsigned word = low >> 5, shift = low & 31u;
  uint64_t value = scalar.w[word] >> shift;
  if (shift && word != 7)
    value |= static_cast<uint64_t>(scalar.w[word + 1]) << (32u - shift);
  return static_cast<unsigned>(value) & (width == 32 ? ~0u : (1u << width) - 1u);
}

// Fixed-base eight-bit multiplication using pre-shifted affine points.  The reference
// double-and-add above remains available as the differential oracle for tests.
inline Point np_generator_mul(const U256& scalar) {
  Point result = np_infinity();
  for (unsigned window = 0; window != 32; ++window) {
    const unsigned digit = np_window_value(scalar, window * 8u, 8u);
    if (digit)
      result = np_add_affine(result, kGeneratorWindow8[window][digit - 1u][0],
                             kGeneratorWindow8[window][digit - 1u][1]);
  }
  return result;
}

struct FieldPowerChain {
  U256 x2, x22, x223;
};

inline FieldPowerChain np_field_power_chain(const U256& value) {
  const auto square_n = [](U256 x, const unsigned count) {
    for (unsigned i = 0; i < count; ++i)
      x = np_field_square(x);
    return x;
  };
  const U256 x2 = np_field_mul(np_field_square(value), value);
  const U256 x3 = np_field_mul(np_field_square(x2), value);
  const U256 x6 = np_field_mul(square_n(x3, 3), x3);
  const U256 x9 = np_field_mul(square_n(x6, 3), x3);
  const U256 x11 = np_field_mul(square_n(x9, 2), x2);
  const U256 x22 = np_field_mul(square_n(x11, 11), x11);
  const U256 x44 = np_field_mul(square_n(x22, 22), x22);
  const U256 x88 = np_field_mul(square_n(x44, 44), x44);
  const U256 x176 = np_field_mul(square_n(x88, 88), x88);
  const U256 x220 = np_field_mul(square_n(x176, 44), x44);
  const U256 x223 = np_field_mul(square_n(x220, 3), x3);
  return {x2, x22, x223};
}

inline U256 np_field_inverse(const U256& value) {
  const auto chain = np_field_power_chain(value);
  U256 high = chain.x223;
  for (unsigned i = 0; i < 33; ++i)
    high = np_field_square(high);
  U256 low = np_one();
  constexpr uint32_t low_exponent = 0xfffffc2du; // 2^32 - 979
  for (unsigned bit = 32; bit-- != 0;) {
    low = np_field_square(low);
    if ((low_exponent >> bit) & 1u)
      low = np_field_mul(low, value);
  }
  return np_field_mul(high, low); // 2^256 - 2^32 - 979 == p - 2
}

inline Point np_affine(const Point& point) {
  if (point.infinity)
    return point;
  const U256 inverse = np_field_inverse(point.z);
  const U256 inverse2 = np_field_square(inverse);
  const U256 inverse3 = np_field_mul(inverse2, inverse);
  return {np_field_mul(point.x, inverse2), np_field_mul(point.y, inverse3), np_one(), false};
}

inline bool np_quadratic_residue_power(const U256& value) {
  // Since p == 3 (mod 4), value is a residue iff value^((p + 1) / 4) squares back.
  const auto chain = np_field_power_chain(value);
  U256 root = chain.x223;
  for (unsigned i = 0; i < 23; ++i)
    root = np_field_square(root);
  root = np_field_mul(root, chain.x22);
  for (unsigned i = 0; i < 6; ++i)
    root = np_field_square(root);
  root = np_field_mul(root, chain.x2);
  root = np_field_square(np_field_square(root));
  return np_cmp(np_field_square(root), value) == 0;
}

inline void np_shift_right(U256& value) {
  uint32_t carry = 0;
  for (int i = 7; i >= 0; --i) {
    const uint32_t next = value.w[i] & 1u;
    value.w[i] = (value.w[i] >> 1u) | (carry << 31u);
    carry = next;
  }
}

inline bool np_quadratic_residue(const U256& value) {
  if (np_zero_p(value))
    return true;
  U256 a = value, n = kFieldPrime;
  bool negative = false;
  while (!np_zero_p(a)) {
    while (!(a.w[0] & 1u)) {
      np_shift_right(a);
      const uint32_t mod8 = n.w[0] & 7u;
      if (mod8 == 3u || mod8 == 5u)
        negative = !negative;
    }
    if (np_cmp(a, n) < 0) {
      const U256 swap = a;
      a = n;
      n = swap;
      if ((a.w[0] & 3u) == 3u && (n.w[0] & 3u) == 3u)
        negative = !negative;
    }
    np_sub_raw(a, n, a);
  }
  return n.w[0] == 1u && !negative;
}
#endif

MOM_NEXAPOW_CONSTANT uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffaU, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u,
};

inline uint32_t np_rotr(const uint32_t x, const unsigned n) {
  return (x >> n) | (x << (32u - n));
}

#ifndef MOM_NEXAPOW_SHA_ATTR
#define MOM_NEXAPOW_SHA_ATTR inline
#endif
MOM_NEXAPOW_SHA_ATTR void np_sha256_words(uint32_t state[8], const uint32_t block[16]) {
  uint32_t w[16];
#if defined(__SYCL_DEVICE_ONLY__)
#pragma unroll
#endif
  for (unsigned i = 0; i < 16; ++i)
    w[i] = block[i];
  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
#if defined(__SYCL_DEVICE_ONLY__)
#pragma unroll
#endif
  for (unsigned i = 0; i < 64; ++i) {
    const unsigned j = i & 15u;
    if (i >= 16) {
      const uint32_t s0 =
          np_rotr(w[(j + 1u) & 15u], 7) ^ np_rotr(w[(j + 1u) & 15u], 18) ^ (w[(j + 1u) & 15u] >> 3);
      const uint32_t s1 = np_rotr(w[(j + 14u) & 15u], 17) ^ np_rotr(w[(j + 14u) & 15u], 19) ^
                          (w[(j + 14u) & 15u] >> 10);
      w[j] += s0 + w[(j + 9u) & 15u] + s1;
    }
    const uint32_t s1 = np_rotr(e, 6) ^ np_rotr(e, 11) ^ np_rotr(e, 25);
    const uint32_t t1 = h + s1 + (g ^ (e & (f ^ g))) + kSha256K[i] + w[j];
    const uint32_t s0 = np_rotr(a, 2) ^ np_rotr(a, 13) ^ np_rotr(a, 22);
    const uint32_t t2 = s0 + ((a & b) | (c & (a | b)));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

inline void np_sha256_block(uint32_t state[8], const uint8_t block[64]) {
  uint32_t words[16];
  for (unsigned i = 0; i < 16; ++i) {
    words[i] = (static_cast<uint32_t>(block[4 * i]) << 24) |
               (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
               (static_cast<uint32_t>(block[4 * i + 2]) << 8) | block[4 * i + 3];
  }
  np_sha256_words(state, words);
}

inline void np_sha256_init(uint32_t state[8]) {
  constexpr uint32_t initial[8] = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
  };
  for (unsigned i = 0; i < 8; ++i)
    state[i] = initial[i];
}

inline void np_sha256_output(const uint32_t state[8], uint8_t output[32]) {
  for (unsigned i = 0; i < 8; ++i) {
    output[4 * i] = static_cast<uint8_t>(state[i] >> 24);
    output[4 * i + 1] = static_cast<uint8_t>(state[i] >> 16);
    output[4 * i + 2] = static_cast<uint8_t>(state[i] >> 8);
    output[4 * i + 3] = static_cast<uint8_t>(state[i]);
  }
}

// All Echelon SHA inputs have a known length.  Keeping the length and block count in the type
// removes runtime padding decisions and lets device compilers size the local buffer exactly.
template <unsigned Length, unsigned Blocks>
inline void np_sha256_fixed(const uint8_t* input, uint8_t output[32]) {
  static_assert(Blocks * 64u >= Length + 9u, "SHA-256 block count is too small");
  uint8_t padded[Blocks * 64u]{};
  for (unsigned i = 0; i < Length; ++i)
    padded[i] = input[i];
  padded[Length] = 0x80;
  const uint64_t bits = static_cast<uint64_t>(Length) * 8u;
  for (unsigned i = 0; i < 8; ++i)
    padded[Blocks * 64u - 1u - i] = static_cast<uint8_t>(bits >> (8u * i));
  uint32_t state[8];
  np_sha256_init(state);
  for (unsigned offset = 0; offset < Blocks * 64u; offset += 64)
    np_sha256_block(state, padded + offset);
  np_sha256_output(state, output);
}

inline void np_sha256d_49(const uint8_t input[49], uint8_t output[32]) {
  uint8_t first[32];
  np_sha256_fixed<49, 1>(input, first);
  np_sha256_fixed<32, 1>(first, output);
}

inline void np_sha256d_41(const uint8_t input[41], uint8_t output[32]) {
  uint8_t first[32];
  np_sha256_fixed<41, 1>(input, first);
  np_sha256_fixed<32, 1>(first, output);
}

inline void np_sha256d_45(const uint8_t input[45], uint8_t output[32]) {
  uint8_t first[32];
  np_sha256_fixed<45, 1>(input, first);
  np_sha256_fixed<32, 1>(first, output);
}

#ifndef MOM_NEXAPOW_SHA_ONLY
MOM_NEXAPOW_SHA_ATTR void np_hmac_states(const uint32_t key[8], uint32_t inner[8],
                                         uint32_t outer[8]) {
  uint32_t block[16];
  for (unsigned i = 0; i < 16; ++i)
    block[i] = (i < 8 ? key[i] : 0) ^ 0x36363636u;
  np_sha256_init(inner);
  np_sha256_words(inner, block);
  for (unsigned i = 0; i < 16; ++i)
    block[i] ^= 0x6a6a6a6au;
  np_sha256_init(outer);
  np_sha256_words(outer, block);
}

MOM_NEXAPOW_SHA_ATTR void np_hmac_outer(const uint32_t digest[8], const uint32_t outer[8],
                                        uint32_t output[8]) {
  uint32_t block[16]{};
  for (unsigned i = 0; i < 8; ++i) {
    block[i] = digest[i];
    output[i] = outer[i];
  }
  block[8] = 0x80000000u;
  block[15] = 96u * 8u;
  np_sha256_words(output, block);
}

MOM_NEXAPOW_SHA_ATTR void np_hmac32(const uint32_t inner[8], const uint32_t outer[8],
                                    const uint32_t data[8], uint32_t output[8]) {
  uint32_t block[16]{}, digest[8];
  for (unsigned i = 0; i < 8; ++i) {
    block[i] = data[i];
    digest[i] = inner[i];
  }
  block[8] = 0x80000000u;
  block[15] = 96u * 8u;
  np_sha256_words(digest, block);
  np_hmac_outer(digest, outer, output);
}

MOM_NEXAPOW_SHA_ATTR void np_hmac_update113(const uint32_t inner[8], const uint32_t outer[8],
                                            const uint32_t v[8], const uint8_t marker,
                                            const uint32_t key[8], const uint32_t message[8],
                                            uint32_t output[8]) {
  static constexpr uint32_t algorithm[4] = {
      0x5363686eu,
      0x6f72722bu,
      0x53484132u,
      0x35362020u,
  };
  uint32_t block[16], digest[8];
  for (unsigned i = 0; i < 8; ++i) {
    block[i] = v[i];
    digest[i] = inner[i];
  }
  block[8] = (static_cast<uint32_t>(marker) << 24) | (key[0] >> 8);
  for (unsigned i = 9; i < 16; ++i)
    block[i] = (key[i - 9] << 24) | (key[i - 8] >> 8);
  np_sha256_words(digest, block);
  block[0] = (key[7] << 24) | (message[0] >> 8);
  for (unsigned i = 1; i < 8; ++i)
    block[i] = (message[i - 1] << 24) | (message[i] >> 8);
  block[8] = (message[7] << 24) | (algorithm[0] >> 8);
  for (unsigned i = 9; i < 12; ++i)
    block[i] = (algorithm[i - 9] << 24) | (algorithm[i - 8] >> 8);
  block[12] = (algorithm[3] << 24) | 0x00800000u;
  block[13] = block[14] = 0;
  block[15] = 177u * 8u;
  np_sha256_words(digest, block);
  np_hmac_outer(digest, outer, output);
}

#ifndef MOM_NEXAPOW_RFC6979_ATTR
#define MOM_NEXAPOW_RFC6979_ATTR inline
#endif
MOM_NEXAPOW_RFC6979_ATTR U256 np_rfc6979(const uint8_t message[32], const U256& key,
                                         const unsigned counter) {
  static constexpr uint32_t initial_inner[8] = {
      0xf454deadu, 0x9725214fu, 0x90daf2a0u, 0xdf1228eau,
      0x64e5750fu, 0xa3924181u, 0x824a932bu, 0xf8e04e32u,
  };
  static constexpr uint32_t initial_outer[8] = {
      0xd385480fu, 0x7abb6477u, 0x37c9c538u, 0x5dd82467u,
      0x8e043a72u, 0x753434b0u, 0xdeb82818u, 0x361d45a6u,
  };
  const U256 reduced_message = np_reduce_mod(np_from_be(message), kGroupOrder);
  uint32_t secret[8], reduced[8], hmac_key[8], v[8], inner[8], outer[8];
  for (unsigned i = 0; i < 8; ++i) {
    secret[i] = key.w[7u - i];
    reduced[i] = reduced_message.w[7u - i];
    v[i] = 0x01010101u;
    inner[i] = initial_inner[i];
    outer[i] = initial_outer[i];
  }
  np_hmac_update113(inner, outer, v, 0, secret, reduced, hmac_key);
  np_hmac_states(hmac_key, inner, outer);
  np_hmac32(inner, outer, v, v);
  np_hmac_update113(inner, outer, v, 1, secret, reduced, hmac_key);
  np_hmac_states(hmac_key, inner, outer);
  np_hmac32(inner, outer, v, v);
  for (unsigned i = 0; i <= counter; ++i)
    np_hmac32(inner, outer, v, v);
  U256 result{};
  for (unsigned i = 0; i < 8; ++i)
    result.w[i] = v[7u - i];
  return result;
}

inline bool np_challenge(const U256& r, const Point& public_key, const uint8_t message[32],
                         U256& out) {
  uint8_t input[97], compressed[33];
  compressed[0] = static_cast<uint8_t>(2u + (public_key.y.w[0] & 1u));
  np_to_be(public_key.x, compressed + 1);
  np_to_be(r, input);
  for (unsigned i = 0; i < sizeof(compressed); ++i)
    input[32 + i] = compressed[i];
  for (unsigned i = 0; i < 32; ++i)
    input[65 + i] = message[i];
  uint8_t digest[32];
  np_sha256_fixed<97, 2>(input, digest);
  out = np_from_be(digest);
  return !np_zero_p(out) && np_cmp(out, kGroupOrder) < 0;
}

inline bool np_sign(const uint8_t h1[32], const U256& private_key, uint8_t signature[64]) {
  if (np_zero_p(private_key) || np_cmp(private_key, kGroupOrder) >= 0)
    return false;
  const Point public_key = np_affine(np_generator_mul(private_key));
  for (unsigned counter = 0; counter != 256; ++counter) {
    U256 k = np_rfc6979(h1, private_key, counter);
    if (np_zero_p(k) || np_cmp(k, kGroupOrder) >= 0)
      continue;
    const Point nonce_point = np_affine(np_generator_mul(k));
    if (!np_quadratic_residue(nonce_point.y))
      k = np_sub_mod(np_zero(), k, kGroupOrder);
    U256 challenge{};
    if (!np_challenge(nonce_point.x, public_key, h1, challenge))
      return false;
    const U256 s = np_add_mod(k, np_mul_mod(challenge, private_key, kGroupOrder), kGroupOrder);
    np_to_be(nonce_point.x, signature);
    np_to_be(s, signature + 32);
    return true;
  }
  return false;
}

inline void np_nonce_bytes(const uint8_t header[32], const uint8_t extranonce[8],
                           const uint64_t miner_nonce, uint8_t serialized[49],
                           const unsigned extra_bytes = 8u) {
  for (unsigned i = 0; i < 32; ++i)
    serialized[i] = header[31u - i];
  serialized[32] = static_cast<uint8_t>(8u + extra_bytes);
  for (unsigned i = 0; i < extra_bytes; ++i)
    serialized[33 + i] = extranonce[i];
  for (unsigned i = 0; i < 8; ++i)
    serialized[33 + extra_bytes + i] = static_cast<uint8_t>(miner_nonce >> (8u * (7u - i)));
}

inline bool np_hash_one(const uint8_t header[32], const uint8_t extranonce[8],
                        const uint64_t miner_nonce, uint8_t final_hash[32],
                        const unsigned extra_bytes = 8u) {
  if (extra_bytes != 0u && extra_bytes != 4u && extra_bytes != 8u)
    return false;
  uint8_t serialized[49], mining_hash[32], h1[32], signature[64], internal_final[32];
  np_nonce_bytes(header, extranonce, miner_nonce, serialized, extra_bytes);
  if (extra_bytes == 0u)
    np_sha256d_41(serialized, mining_hash);
  else if (extra_bytes == 4u)
    np_sha256d_45(serialized, mining_hash);
  else
    np_sha256d_49(serialized, mining_hash);
  np_sha256_fixed<32, 1>(mining_hash, h1);
  if (!np_sign(h1, np_from_be(mining_hash), signature))
    return false;
  np_sha256_fixed<64, 2>(signature, internal_final);
  for (unsigned i = 0; i < 32; ++i)
    final_hash[i] = internal_final[31u - i];
  return true;
}
#endif
