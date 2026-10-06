#include "field_10x26_ref.inc"
#include "field_8x32_portable.inc"
#ifdef MOM_NEXAPOW_SYCL_NATIVE_FIELD
#include "field_4x64_ptx.inc"
#include "field_4x64_hybrid_ptx.inc"
#endif

namespace sycl_pipeline {

#if defined(__clang__) || defined(__GNUC__)
#define MOM_NEXAPOW_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define MOM_NEXAPOW_ALWAYS_INLINE inline
#endif
#if defined(MOM_NEXAPOW_SYCL_NATIVE_FIELD) && !defined(MOM_NEXAPOW_FORCE_PORTABLE) &&              \
    defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#define MOM_NEXAPOW_USE_NATIVE_FIELD
#endif
#if defined(MOM_NEXAPOW_USE_NATIVE_FIELD) || defined(MOM_NEXAPOW_PORTABLE_FIELD32)
#define MOM_NEXAPOW_FIELD_INLINE MOM_NEXAPOW_ALWAYS_INLINE
#else
#define MOM_NEXAPOW_FIELD_INLINE inline
#endif

constexpr uint32_t kWorkGroup = 128u;
constexpr uint32_t kPublicCurveWorkGroup = 64u;
constexpr uint32_t kNonceCurveWorkGroup = 128u;
constexpr uint32_t kCompactPoints = 32u * 255u;

struct TableLayout {
  uint32_t wide_bits;
  uint32_t wide_rows;
  uint32_t narrow_bits;
  uint32_t total_rows;
};
static_assert(std::is_trivially_copyable<TableLayout>::value);

constexpr TableLayout kSmallTable{22u, 4u, 21u, 12u};
constexpr TableLayout kLargeTable{24u, 3u, 23u, 11u};

constexpr uint32_t table_entries(const uint32_t bits) {
  return (1u << bits) - 1u;
}
constexpr uint32_t table_points(const TableLayout layout) {
  return layout.wide_rows * table_entries(layout.wide_bits) +
         (layout.total_rows - layout.wide_rows) * table_entries(layout.narrow_bits);
}
static_assert(table_points(kSmallTable) == 33554420u);
static_assert(table_points(kLargeTable) == 117440501u);

#if defined(MOM_NEXAPOW_PORTABLE_FIELD32)
struct Field {
  uint32_t n[8];
};
#else
struct Field {
  uint32_t n[10];
};
#endif
struct alignas(16) AffineTablePoint {
  U256 x, y;
};
struct CurvePoint {
  Field x, y, zz, zzz;
  bool infinity;
};
struct PointPlanes {
  Field *x = nullptr, *y = nullptr, *z = nullptr;
};
struct SearchResult {
  uint32_t count, pad;
  uint64_t nonce;
};
struct HeaderContext {
  uint32_t words[10], state[8], word10_prefix, extra_bytes;
};

static_assert(sizeof(AffineTablePoint) == 64);
static_assert(alignof(AffineTablePoint) == 16);
static_assert(sizeof(SearchResult) == 16);

#ifdef MOM_NEXAPOW_USE_NATIVE_FIELD
MOM_NEXAPOW_FIELD_INLINE Field field_from_u256(const U256& value) {
  Field result{};
  for (uint32_t i = 0; i != 8u; ++i)
    result.n[i] = value.w[i];
  return result;
}
MOM_NEXAPOW_FIELD_INLINE void field_normalize(Field&) {
}
MOM_NEXAPOW_FIELD_INLINE U256 field_to_u256(const Field& value) {
  U256 result{};
  for (uint32_t i = 0; i != 8u; ++i)
    result.w[i] = value.n[i];
  return result;
}
MOM_NEXAPOW_FIELD_INLINE Field field_zero() {
  return {};
}
MOM_NEXAPOW_FIELD_INLINE Field field_one() {
  Field result{};
  result.n[0] = 1u;
  return result;
}
MOM_NEXAPOW_FIELD_INLINE Field field_mul(const Field& a, const Field& b) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  Field result{};
  mom_nexa_field32_hybrid_ptx::mul(result.n, a.n, b.n);
  return result;
#else
  return field_from_u256(np_field_mul(field_to_u256(a), field_to_u256(b)));
#endif
}
MOM_NEXAPOW_FIELD_INLINE Field field_square(const Field& value) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  Field result{};
  mom_nexa_field32_hybrid_ptx::square(result.n, value.n);
  return result;
#else
  return field_from_u256(np_field_square(field_to_u256(value)));
#endif
}
MOM_NEXAPOW_FIELD_INLINE void field_normalize_weak(Field&) {
}
MOM_NEXAPOW_FIELD_INLINE Field field_sub(const Field& a, const Field& b) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  Field result{};
  mom_nexa_field32_hybrid_ptx::sub(result.n, a.n, b.n);
  return result;
#else
  return field_from_u256(np_field_sub(field_to_u256(a), field_to_u256(b)));
#endif
}
MOM_NEXAPOW_FIELD_INLINE Field field_inverse(const Field& value) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  uint64_t limbs[4];
  for (uint32_t i = 0; i != 4u; ++i)
    limbs[i] = value.n[2u * i] | (static_cast<uint64_t>(value.n[2u * i + 1u]) << 32u);
  mom_nexa_field64_ptx::inverse(limbs);
  Field result{};
  for (uint32_t i = 0; i != 4u; ++i) {
    result.n[2u * i] = static_cast<uint32_t>(limbs[i]);
    result.n[2u * i + 1u] = static_cast<uint32_t>(limbs[i] >> 32u);
  }
  return result;
#else
  return field_from_u256(np_field_inverse(field_to_u256(value)));
#endif
}
#elif defined(MOM_NEXAPOW_PORTABLE_FIELD32)
MOM_NEXAPOW_FIELD_INLINE Field field_from_u256(const U256& value) {
  Field result{};
  for (uint32_t i = 0; i != 8u; ++i)
    result.n[i] = value.w[i];
  return result;
}
MOM_NEXAPOW_FIELD_INLINE void field_normalize(Field&) {
}
MOM_NEXAPOW_FIELD_INLINE U256 field_to_u256(const Field& value) {
  U256 result{};
  for (uint32_t i = 0; i != 8u; ++i)
    result.w[i] = value.n[i];
  return result;
}
MOM_NEXAPOW_FIELD_INLINE Field field_zero() {
  return {};
}
MOM_NEXAPOW_FIELD_INLINE Field field_one() {
  Field result{};
  result.n[0] = 1u;
  return result;
}
MOM_NEXAPOW_FIELD_INLINE Field field_mul(const Field& a, const Field& b) {
  Field result{};
  mom_nexa_field32_portable::mul(result.n, a.n, b.n);
  return result;
}
MOM_NEXAPOW_FIELD_INLINE Field field_square(const Field& value) {
  Field result{};
  mom_nexa_field32_portable::square(result.n, value.n);
  return result;
}
MOM_NEXAPOW_FIELD_INLINE void field_normalize_weak(Field&) {
}
MOM_NEXAPOW_FIELD_INLINE Field field_sub(const Field& a, const Field& b) {
  Field result{};
  mom_nexa_field32_portable::sub(result.n, a.n, b.n);
  return result;
}
MOM_NEXAPOW_FIELD_INLINE Field field_inverse(const Field& value) {
  return field_from_u256(np_field_inverse(field_to_u256(value)));
}
#else
MOM_NEXAPOW_FIELD_INLINE Field field_from_u256(const U256& value) {
  Field result{};
  for (uint32_t i = 0; i != 10u; ++i)
    result.n[i] = np_window_value(value, 26u * i, i == 9u ? 22u : 26u);
  return result;
}

MOM_NEXAPOW_FIELD_INLINE void field_normalize(Field& value) {
  uint32_t t0 = value.n[0], t1 = value.n[1], t2 = value.n[2], t3 = value.n[3], t4 = value.n[4],
           t5 = value.n[5], t6 = value.n[6], t7 = value.n[7], t8 = value.n[8], t9 = value.n[9], m,
           x = t9 >> 22u;
  t9 &= 0x03fffffu;
  t0 += x * 0x3d1u;
  t1 += x << 6u;
  t1 += t0 >> 26u;
  t0 &= 0x3ffffffu;
  t2 += t1 >> 26u;
  t1 &= 0x3ffffffu;
  t3 += t2 >> 26u;
  t2 &= 0x3ffffffu;
  m = t2;
  t4 += t3 >> 26u;
  t3 &= 0x3ffffffu;
  m &= t3;
  t5 += t4 >> 26u;
  t4 &= 0x3ffffffu;
  m &= t4;
  t6 += t5 >> 26u;
  t5 &= 0x3ffffffu;
  m &= t5;
  t7 += t6 >> 26u;
  t6 &= 0x3ffffffu;
  m &= t6;
  t8 += t7 >> 26u;
  t7 &= 0x3ffffffu;
  m &= t7;
  t9 += t8 >> 26u;
  t8 &= 0x3ffffffu;
  m &= t8;
  x = (t9 >> 22u) | ((t9 == 0x03fffffu) & (m == 0x3ffffffu) &
                     ((t1 + 0x40u + ((t0 + 0x3d1u) >> 26u)) > 0x3ffffffu));
  t0 += x * 0x3d1u;
  t1 += x << 6u;
  t1 += t0 >> 26u;
  t0 &= 0x3ffffffu;
  t2 += t1 >> 26u;
  t1 &= 0x3ffffffu;
  t3 += t2 >> 26u;
  t2 &= 0x3ffffffu;
  t4 += t3 >> 26u;
  t3 &= 0x3ffffffu;
  t5 += t4 >> 26u;
  t4 &= 0x3ffffffu;
  t6 += t5 >> 26u;
  t5 &= 0x3ffffffu;
  t7 += t6 >> 26u;
  t6 &= 0x3ffffffu;
  t8 += t7 >> 26u;
  t7 &= 0x3ffffffu;
  t9 = (t9 + (t8 >> 26u)) & 0x03fffffu;
  t8 &= 0x3ffffffu;
  const uint32_t normalized[] = {t0, t1, t2, t3, t4, t5, t6, t7, t8, t9};
  for (uint32_t i = 0; i != 10u; ++i)
    value.n[i] = normalized[i];
}

MOM_NEXAPOW_FIELD_INLINE U256 field_to_u256(Field value) {
  field_normalize(value);
  U256 result{};
  for (uint32_t i = 0; i != 10u; ++i) {
    const uint32_t low = 26u * i, word = low >> 5u, shift = low & 31u;
    result.w[word] |= value.n[i] << shift;
    if (shift && word != 7u)
      result.w[word + 1u] |= value.n[i] >> (32u - shift);
  }
  return result;
}

MOM_NEXAPOW_FIELD_INLINE Field field_zero() {
  return {};
}
MOM_NEXAPOW_FIELD_INLINE Field field_one() {
  Field result{};
  result.n[0] = 1u;
  return result;
}
MOM_NEXAPOW_FIELD_INLINE Field field_mul(const Field& a, const Field& b) {
  Field result{};
  mom_nexa_field10_ref::mul(result.n, a.n, b.n);
  return result;
}
MOM_NEXAPOW_FIELD_INLINE Field field_square(const Field& value) {
  Field result{};
  mom_nexa_field10_ref::square(result.n, value.n);
  return result;
}
MOM_NEXAPOW_FIELD_INLINE void field_normalize_weak(Field& value) {
  uint32_t carry = value.n[9] >> 22u;
  value.n[9] &= 0x03fffffu;
  value.n[0] += carry * 0x3d1u;
  value.n[1] += carry << 6u;
  for (uint32_t i = 0; i != 9u; ++i) {
    value.n[i + 1u] += value.n[i] >> 26u;
    value.n[i] &= 0x3ffffffu;
  }
}
MOM_NEXAPOW_FIELD_INLINE Field field_sub(const Field& a, const Field& b) {
  constexpr uint32_t prime[10] = {
      0x3fffc2fu, 0x3ffffbfu, 0x3ffffffu, 0x3ffffffu, 0x3ffffffu,
      0x3ffffffu, 0x3ffffffu, 0x3ffffffu, 0x3ffffffu, 0x03fffffu,
  };
  Field result{};
  for (uint32_t i = 0; i != 10u; ++i)
    result.n[i] = 4u * prime[i] - b.n[i] + a.n[i];
  return result;
}
MOM_NEXAPOW_FIELD_INLINE Field field_inverse(const Field& value) {
  return field_from_u256(np_field_inverse(field_to_u256(value)));
}
#endif

MOM_NEXAPOW_FIELD_INLINE Field field_mul_sub_mul(const Field& a, const Field& b, const Field& c,
                                                 const Field& d) {
#if defined(MOM_NEXAPOW_USE_NATIVE_FIELD) || defined(MOM_NEXAPOW_PORTABLE_FIELD32)
  Field result{};
#ifdef MOM_NEXAPOW_USE_NATIVE_FIELD
  mom_nexa_field32_hybrid_ptx::mul_sub_mul(result.n, a.n, b.n, c.n, d.n);
#else
  mom_nexa_field32_portable::mul_sub_mul(result.n, a.n, b.n, c.n, d.n);
#endif
  return result;
#else
  return field_sub(field_mul(a, b), field_mul(c, d));
#endif
}

MOM_NEXAPOW_FIELD_INLINE bool field_quadratic_residue(const Field& value) {
#ifdef MOM_NEXAPOW_USE_NATIVE_FIELD
  uint64_t limbs[4];
  for (uint32_t i = 0; i != 4u; ++i)
    limbs[i] = value.n[2u * i] | (static_cast<uint64_t>(value.n[2u * i + 1u]) << 32u);
  return mom_nexa_field64_ptx::is_quadratic_residue(limbs);
#else
  return np_quadratic_residue(field_to_u256(value));
#endif
}

template <bool Add> MOM_NEXAPOW_ALWAYS_INLINE U256 group_op(const U256& a, const U256& b) {
#ifdef MOM_NEXAPOW_USE_NATIVE_FIELD
  constexpr uint64_t order[4] = {
      0xbfd25e8cd0364141ULL,
      0xbaaedce6af48a03bULL,
      0xfffffffffffffffeULL,
      0xffffffffffffffffULL,
  };
  uint64_t lhs[4], rhs[4], output[4];
  for (uint32_t i = 0; i != 4u; ++i) {
    lhs[i] = a.w[2u * i] | (static_cast<uint64_t>(a.w[2u * i + 1u]) << 32u);
    rhs[i] = b.w[2u * i] | (static_cast<uint64_t>(b.w[2u * i + 1u]) << 32u);
  }
  if constexpr (Add)
    mom_nexa_field64_ptx::add_mod(output, lhs, rhs, order);
  else
    mom_nexa_field64_ptx::sub_mod(output, lhs, rhs, order);
  U256 result{};
  for (uint32_t i = 0; i != 4u; ++i) {
    result.w[2u * i] = static_cast<uint32_t>(output[i]);
    result.w[2u * i + 1u] = static_cast<uint32_t>(output[i] >> 32u);
  }
  return result;
#else
  return Add ? np_add_mod(a, b, kGroupOrder) : np_sub_mod(a, b, kGroupOrder);
#endif
}

template <uint32_t HighLimbs, uint32_t OutputLimbs>
MOM_NEXAPOW_ALWAYS_INLINE void group_fold_complement(uint32_t* output, const uint32_t* low,
                                                     const uint32_t* high) {
  constexpr uint32_t complement[5] = {
      0x2fc9bebfu, 0x402da173u, 0x50b75fc4u, 0x45512319u, 1u,
  };
  uint64_t carry = 0u;
#if defined(__SYCL_DEVICE_ONLY__)
#pragma unroll
#endif
  for (uint32_t column = 0; column != OutputLimbs; ++column) {
    uint64_t lower = carry, upper = 0u;
    if (column < 8u) {
      const uint64_t before = lower;
      lower += low[column];
      upper += lower < before;
    }
#if defined(__SYCL_DEVICE_ONLY__)
#pragma unroll
#endif
    for (uint32_t limb = 0; limb != HighLimbs; ++limb) {
      const int factor = static_cast<int>(column) - static_cast<int>(limb);
      if (factor >= 0 && factor < 5)
        np_comba_add_product(lower, upper, high[limb], complement[factor]);
    }
    output[column] = static_cast<uint32_t>(lower);
    carry = np_comba_carry(lower, upper);
  }
}

MOM_NEXAPOW_ALWAYS_INLINE U256 group_mul(const U256& a, const U256& b) {
  uint32_t product[16], work[13], high[5];
#ifdef MOM_NEXAPOW_USE_NATIVE_FIELD
  mom_nexa_field32_hybrid_ptx::detail::mul_256_comba32(a.w, b.w, product);
#else
  np_comba_mul(a, b, product);
#endif
  group_fold_complement<8, 13>(work, product, product + 8);
  for (uint32_t i = 0; i != 5u; ++i)
    high[i] = work[8u + i];
  group_fold_complement<5, 9>(work, work, high);
  uint32_t overflow = work[8];
  group_fold_complement<1, 9>(work, work, &overflow);
  overflow = work[8];
  group_fold_complement<1, 9>(work, work, &overflow);
  U256 result{}, reduced{};
  for (uint32_t i = 0; i != 8u; ++i)
    result.w[i] = work[i];
  if (!np_sub_raw(result, kGroupOrder, reduced))
    result = reduced;
  return result;
}

template <unsigned ExtraBytes>
inline HeaderContext make_header_context_mode(const uint8_t header[32], const uint8_t extranonce[8]) {
  static_assert(ExtraBytes == 0u || ExtraBytes == 4u || ExtraBytes == 8u);
  HeaderContext output{};
  output.extra_bytes = ExtraBytes;
  (void)extranonce;
  for (uint32_t word = 0; word != 8u; ++word) {
    const uint32_t byte = 31u - 4u * word;
    output.words[word] = (static_cast<uint32_t>(header[byte]) << 24u) |
                         (static_cast<uint32_t>(header[byte - 1u]) << 16u) |
                         (static_cast<uint32_t>(header[byte - 2u]) << 8u) | header[byte - 3u];
  }
  if constexpr (ExtraBytes == 4u) {
    output.words[8] = 0x0c000000u | (static_cast<uint32_t>(extranonce[0]) << 16u) |
                      (static_cast<uint32_t>(extranonce[1]) << 8u) | extranonce[2];
    output.words[9] = static_cast<uint32_t>(extranonce[3]) << 24u;
  } else if constexpr (ExtraBytes == 8u) {
    output.words[8] = 0x10000000u | (static_cast<uint32_t>(extranonce[0]) << 16u) |
                      (static_cast<uint32_t>(extranonce[1]) << 8u) | extranonce[2];
    output.words[9] = (static_cast<uint32_t>(extranonce[3]) << 24u) |
                      (static_cast<uint32_t>(extranonce[4]) << 16u) |
                      (static_cast<uint32_t>(extranonce[5]) << 8u) | extranonce[6];
    output.word10_prefix = static_cast<uint32_t>(extranonce[7]) << 24u;
  }
  uint32_t a = 0x6a09e667u, b = 0xbb67ae85u, c = 0x3c6ef372u, d = 0xa54ff53au;
  uint32_t e = 0x510e527fu, f = 0x9b05688cu, g = 0x1f83d9abu, h = 0x5be0cd19u;
  constexpr uint32_t precomputed_rounds = ExtraBytes == 0u ? 8u : ExtraBytes == 4u ? 9u : 10u;
  for (uint32_t i = 0; i != precomputed_rounds; ++i) {
    const uint32_t s1 = np_rotr(e, 6) ^ np_rotr(e, 11) ^ np_rotr(e, 25);
    const uint32_t t1 = h + s1 + (g ^ (e & (f ^ g))) + kSha256K[i] + output.words[i];
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
  const uint32_t state[] = {a, b, c, d, e, f, g, h};
  for (uint32_t i = 0; i != 8u; ++i)
    output.state[i] = state[i];
  return output;
}

inline HeaderContext make_header_context(const uint8_t header[32], const uint8_t extranonce[8],
                                         const unsigned extra_bytes) {
  if (extra_bytes == 0u)
    return make_header_context_mode<0u>(header, extranonce);
  if (extra_bytes == 4u)
    return make_header_context_mode<4u>(header, extranonce);
  return make_header_context_mode<8u>(header, extranonce);
}

template <unsigned ExtraBytes>
MOM_NEXAPOW_ALWAYS_INLINE void fast_header_sha256(const HeaderContext& context,
                                                  const uint64_t nonce, uint32_t output[8]) {
  uint32_t w[16] = {
      context.words[0],
      context.words[1],
      context.words[2],
      context.words[3],
      context.words[4],
      context.words[5],
      context.words[6],
      context.words[7],
      0u,
      0u,
      0u,
      0u,
      0u,
      0u,
  };
  if constexpr (ExtraBytes == 0u) {
    w[8] = 0x08000000u | (static_cast<uint32_t>(nonce >> 40u) & 0x00ffffffu);
    w[9] = static_cast<uint32_t>(nonce >> 8u);
    w[10] = (static_cast<uint32_t>(nonce) << 24u) | 0x00800000u;
    w[15] = 328u;
  } else if constexpr (ExtraBytes == 4u) {
    w[8] = context.words[8];
    w[9] = context.words[9] | (static_cast<uint32_t>(nonce >> 40u) & 0x00ffffffu);
    w[10] = static_cast<uint32_t>(nonce >> 8u);
    w[11] = (static_cast<uint32_t>(nonce) << 24u) | 0x00800000u;
    w[15] = 360u;
  } else {
    w[8] = context.words[8];
    w[9] = context.words[9];
    w[10] = context.word10_prefix | (static_cast<uint32_t>(nonce >> 40u) & 0x00ffffffu);
    w[11] = static_cast<uint32_t>(nonce >> 8u);
    w[12] = (static_cast<uint32_t>(nonce) << 24u) | 0x00800000u;
    w[15] = 392u;
  }
  uint32_t a = context.state[0], b = context.state[1], c = context.state[2], d = context.state[3];
  uint32_t e = context.state[4], f = context.state[5], g = context.state[6], h = context.state[7];
  constexpr uint32_t first_round = ExtraBytes == 0u ? 8u : ExtraBytes == 4u ? 9u : 10u;
#if defined(__SYCL_DEVICE_ONLY__)
#pragma unroll
#endif
  for (uint32_t i = first_round; i != 64u; ++i) {
    const uint32_t j = i & 15u;
    if (i >= 16u) {
      const uint32_t s0 = np_rotr(w[(j + 1u) & 15u], 7) ^ np_rotr(w[(j + 1u) & 15u], 18) ^
                          (w[(j + 1u) & 15u] >> 3u);
      const uint32_t s1 = np_rotr(w[(j + 14u) & 15u], 17) ^ np_rotr(w[(j + 14u) & 15u], 19) ^
                          (w[(j + 14u) & 15u] >> 10u);
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
  constexpr uint32_t initial[8] = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
  };
  const uint32_t state[] = {a, b, c, d, e, f, g, h};
  for (uint32_t i = 0; i != 8u; ++i)
    output[i] = initial[i] + state[i];
}

MOM_NEXAPOW_ALWAYS_INLINE void fast_header_sha256(const HeaderContext& context,
                                                  const uint64_t nonce, uint32_t output[8]) {
  if (context.extra_bytes == 0u)
    fast_header_sha256<0u>(context, nonce, output);
  else if (context.extra_bytes == 4u)
    fast_header_sha256<4u>(context, nonce, output);
  else
    fast_header_sha256<8u>(context, nonce, output);
}

MOM_NEXAPOW_ALWAYS_INLINE __attribute__((flatten)) void fast_sha256_32(const uint32_t input[8],
                                                                       uint32_t output[8]) {
  uint32_t block[16]{};
  for (uint32_t i = 0; i != 8u; ++i)
    block[i] = input[i];
  block[8] = 0x80000000u;
  block[15] = 256u;
  np_sha256_init(output);
  np_sha256_words(output, block);
}

MOM_NEXAPOW_ALWAYS_INLINE U256 fast_rfc6979(const uint32_t message[8], const U256& key) {
  constexpr uint32_t initial_inner[8] = {
      0xf454deadu, 0x9725214fu, 0x90daf2a0u, 0xdf1228eau,
      0x64e5750fu, 0xa3924181u, 0x824a932bu, 0xf8e04e32u,
  };
  constexpr uint32_t initial_outer[8] = {
      0xd385480fu, 0x7abb6477u, 0x37c9c538u, 0x5dd82467u,
      0x8e043a72u, 0x753434b0u, 0xdeb82818u, 0x361d45a6u,
  };
  U256 message_value{};
  for (uint32_t i = 0; i != 8u; ++i)
    message_value.w[i] = message[7u - i];
  const U256 reduced_message = np_reduce_mod(message_value, kGroupOrder);
  uint32_t secret[8], reduced[8], hmac_key[8], v[8], inner[8], outer[8];
  for (uint32_t i = 0; i != 8u; ++i) {
    secret[i] = key.w[7u - i];
    reduced[i] = reduced_message.w[7u - i];
    v[i] = 0x01010101u;
    inner[i] = initial_inner[i];
    outer[i] = initial_outer[i];
  }
  np_hmac_update113(inner, outer, v, 0u, secret, reduced, hmac_key);
  np_hmac_states(hmac_key, inner, outer);
  np_hmac32(inner, outer, v, v);
  np_hmac_update113(inner, outer, v, 1u, secret, reduced, hmac_key);
  np_hmac_states(hmac_key, inner, outer);
  np_hmac32(inner, outer, v, v);
  np_hmac32(inner, outer, v, v);
  U256 result{};
  for (uint32_t i = 0; i != 8u; ++i)
    result.w[i] = v[7u - i];
  return result;
}

MOM_NEXAPOW_ALWAYS_INLINE __attribute__((flatten)) bool
fast_challenge(const U256& nonce_x, const U256& public_x, const uint32_t public_y_lsb,
               const U256& h1, U256& challenge) {
  uint32_t block[16]{}, state[8], message[8];
  for (uint32_t i = 0; i != 8u; ++i)
    block[i] = nonce_x.w[7u - i];
  block[8] = ((2u + public_y_lsb) << 24u) | (public_x.w[7] >> 8u);
  for (uint32_t i = 1u; i != 8u; ++i)
    block[8u + i] = (public_x.w[8u - i] << 24u) | (public_x.w[7u - i] >> 8u);
  np_sha256_init(state);
  np_sha256_words(state, block);
  for (uint32_t i = 0; i != 8u; ++i)
    message[i] = h1.w[7u - i];
  block[0] = ((public_x.w[0] & 0xffu) << 24u) | (message[0] >> 8u);
  for (uint32_t i = 1u; i != 8u; ++i)
    block[i] = (message[i - 1u] << 24u) | (message[i] >> 8u);
  block[8] = (message[7] << 24u) | 0x00800000u;
  for (uint32_t i = 9u; i != 15u; ++i)
    block[i] = 0u;
  block[15] = 776u;
  np_sha256_words(state, block);
  for (uint32_t i = 0; i != 8u; ++i)
    challenge.w[i] = state[7u - i];
  return !np_zero_p(challenge) && np_cmp(challenge, kGroupOrder) < 0;
}

inline uint32_t reverse_bytes(const uint32_t value) {
  return (value >> 24u) | ((value >> 8u) & 0x0000ff00u) | ((value << 8u) & 0x00ff0000u) |
         (value << 24u);
}

MOM_NEXAPOW_ALWAYS_INLINE __attribute__((flatten)) bool
signature_meets_target(const U256& r, const U256& s, const uint8_t target[32]) {
  uint32_t block[16], state[8];
  for (uint32_t i = 0; i != 8u; ++i) {
    block[i] = r.w[7u - i];
    block[8u + i] = s.w[7u - i];
  }
  np_sha256_init(state);
  np_sha256_words(state, block);
  for (uint32_t i = 0; i != 16u; ++i)
    block[i] = 0u;
  block[0] = 0x80000000u;
  block[15] = 512u;
  np_sha256_words(state, block);
  for (uint32_t i = 0; i != 8u; ++i) {
    const uint32_t value = reverse_bytes(state[7u - i]);
    const uint32_t limit = (static_cast<uint32_t>(target[4u * i]) << 24u) |
                           (static_cast<uint32_t>(target[4u * i + 1u]) << 16u) |
                           (static_cast<uint32_t>(target[4u * i + 2u]) << 8u) | target[4u * i + 3u];
    if (value != limit)
      return value < limit;
  }
  return true;
}

MOM_NEXAPOW_ALWAYS_INLINE void finish_signature(const uint32_t ordinal, const uint64_t first_nonce,
                                                const U256& public_x, const uint32_t public_y_lsb,
                                                const U256& nonce_x, const Field& nonce_y,
                                                const Field& nonce_z, const U256& h1,
                                                const U256& private_key, const U256& nonce_key,
                                                const uint8_t target[32], SearchResult* result) {
  U256 challenge{};
  if (!fast_challenge(nonce_x, public_x, public_y_lsb, h1, challenge))
    return;
  const U256 product = group_mul(challenge, private_key);
  const U256 positive = group_op<true>(nonce_key, product);
  const U256 negative = group_op<false>(product, nonce_key);
  const bool positive_candidate = signature_meets_target(nonce_x, positive, target);
  const bool negative_candidate = signature_meets_target(nonce_x, negative, target);
  if (!positive_candidate && !negative_candidate)
    return;
  const bool residue = field_quadratic_residue(field_mul(nonce_y, nonce_z));
  if (residue ? !positive_candidate : !negative_candidate)
    return;
  using Atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>;
  uint32_t expected = 0;
  if (Atomic(result->count).compare_exchange_strong(expected, 1u))
    result->nonce = first_nonce + ordinal;
}

constexpr uint32_t table_bits(const TableLayout layout, const uint32_t row) {
  return row < layout.wide_rows ? layout.wide_bits : layout.narrow_bits;
}
constexpr uint32_t table_start(const TableLayout layout, const uint32_t row) {
  return row < layout.wide_rows ? row * layout.wide_bits
                                : layout.wide_rows * layout.wide_bits +
                                      (row - layout.wide_rows) * layout.narrow_bits;
}
constexpr uint32_t table_offset(const TableLayout layout, const uint32_t row) {
  const uint32_t wide_entries = table_entries(layout.wide_bits);
  const uint32_t narrow_entries = table_entries(layout.narrow_bits);
  return row < layout.wide_rows ? row * wide_entries
                                : layout.wide_rows * wide_entries +
                                      (row - layout.wide_rows) * narrow_entries;
}
constexpr bool table_layout_is_complete(const TableLayout layout) {
  const uint32_t last = layout.total_rows - 1u;
  return table_start(layout, last) + table_bits(layout, last) == 256u &&
         table_offset(layout, last) + table_entries(table_bits(layout, last)) ==
             table_points(layout);
}
static_assert(table_layout_is_complete(kSmallTable));
static_assert(table_layout_is_complete(kLargeTable));
MOM_NEXAPOW_ALWAYS_INLINE AffineTablePoint load_table_point(const AffineTablePoint* table,
                                                            const uint64_t index) {
#ifdef MOM_NEXAPOW_USE_NATIVE_FIELD
  const uint32_t* input = reinterpret_cast<const uint32_t*>(table + index);
  AffineTablePoint result{};
#define MOM_NEXAPOW_LOAD4(offset, a, b, c, d)                                                      \
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"                                          \
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)                                                \
               : "l"(input + offset))
  MOM_NEXAPOW_LOAD4(0, result.x.w[0], result.x.w[1], result.x.w[2], result.x.w[3]);
  MOM_NEXAPOW_LOAD4(4, result.x.w[4], result.x.w[5], result.x.w[6], result.x.w[7]);
  MOM_NEXAPOW_LOAD4(8, result.y.w[0], result.y.w[1], result.y.w[2], result.y.w[3]);
  MOM_NEXAPOW_LOAD4(12, result.y.w[4], result.y.w[5], result.y.w[6], result.y.w[7]);
#undef MOM_NEXAPOW_LOAD4
  return result;
#else
  const auto* input = reinterpret_cast<const sycl::uint4*>(table + index);
  const sycl::uint4 x0 = input[0], x1 = input[1], y0 = input[2], y1 = input[3];
  return {{{x0.x(), x0.y(), x0.z(), x0.w(), x1.x(), x1.y(), x1.z(), x1.w()}},
          {{y0.x(), y0.y(), y0.z(), y0.w(), y1.x(), y1.y(), y1.z(), y1.w()}}};
#endif
}

inline CurvePoint infinity() {
  return {field_zero(), field_zero(), field_zero(), field_zero(), true};
}

MOM_NEXAPOW_ALWAYS_INLINE CurvePoint add_affine_pair(const U256& x1, const U256& y1, const U256& x2,
                                                     const U256& y2) {
  const Field ax1 = field_from_u256(x1), ay1 = field_from_u256(y1);
  const Field h = field_sub(field_from_u256(x2), ax1);
  const Field r = field_sub(field_from_u256(y2), ay1);
  const Field hh = field_square(h), hhh = field_mul(h, hh);
  const Field xhh = field_mul(ax1, hh);
  Field x = field_sub(field_sub(field_sub(field_square(r), hhh), xhh), xhh);
  field_normalize_weak(x);
  Field y = field_mul_sub_mul(r, field_sub(xhh, x), ay1, hhh);
  field_normalize_weak(y);
  return {x, y, hh, hhh, false};
}

MOM_NEXAPOW_ALWAYS_INLINE CurvePoint add_affine(const CurvePoint& point, const U256& x,
                                                const U256& y) {
  const Field affine_x = field_from_u256(x), affine_y = field_from_u256(y);
  if (__builtin_expect(point.infinity, 0))
    return {affine_x, affine_y, field_one(), field_one(), false};
  const Field h = field_sub(field_mul(affine_x, point.zz), point.x);
  const Field r = field_sub(field_mul(affine_y, point.zzz), point.y);
  const Field hh = field_square(h), hhh = field_mul(h, hh);
  const Field xhh = field_mul(point.x, hh);
  Field output_x = field_sub(field_sub(field_sub(field_square(r), hhh), xhh), xhh);
  field_normalize_weak(output_x);
  Field output_y = field_mul_sub_mul(r, field_sub(xhh, output_x), point.y, hhh);
  field_normalize_weak(output_y);
  return {output_x, output_y, field_mul(point.zz, hh), field_mul(point.zzz, hhh), false};
}

inline AffineTablePoint affine(const CurvePoint& point) {
  const Field inverse = field_inverse(field_mul(point.zz, point.zzz));
  return {field_to_u256(field_mul(point.x, field_mul(point.zzz, inverse))),
          field_to_u256(field_mul(point.y, field_mul(point.zz, inverse)))};
}

inline uint32_t window_value(const U256& scalar, const uint32_t low, const uint32_t width) {
  const uint32_t limb = low >> 5u, shift = low & 31u;
  uint64_t value = static_cast<uint64_t>(scalar.w[limb]) >> shift;
  if (shift && limb != 7u)
    value |= static_cast<uint64_t>(scalar.w[limb + 1u]) << (32u - shift);
  return static_cast<uint32_t>(value) & ((1u << width) - 1u);
}

MOM_NEXAPOW_ALWAYS_INLINE CurvePoint multiply(const U256& scalar, const AffineTablePoint* table,
                                              const TableLayout layout) {
  const uint32_t wide_entries = table_entries(layout.wide_bits);
  const uint32_t first_digit =
      window_value(scalar, table_start(layout, 0u), table_bits(layout, 0u));
  const uint32_t second_digit =
      window_value(scalar, table_start(layout, 1u), table_bits(layout, 1u));
  CurvePoint result;
  if (__builtin_expect(first_digit && second_digit, 1)) {
    const AffineTablePoint first = load_table_point(table, first_digit - 1u);
    const AffineTablePoint second = load_table_point(table, wide_entries + second_digit - 1u);
    result = add_affine_pair(first.x, first.y, second.x, second.y);
  } else {
    result = infinity();
    if (first_digit) {
      const AffineTablePoint point = load_table_point(table, first_digit - 1u);
      result = add_affine(result, point.x, point.y);
    }
    if (second_digit) {
      const AffineTablePoint point = load_table_point(table, wide_entries + second_digit - 1u);
      result = add_affine(result, point.x, point.y);
    }
  }
#if defined(__SYCL_DEVICE_ONLY__)
#pragma unroll 1
#endif
  for (uint32_t row = 2u; row != layout.total_rows; ++row) {
    const uint32_t digit =
        window_value(scalar, table_start(layout, row), table_bits(layout, row));
    if (__builtin_expect(digit != 0u, 1)) {
      const AffineTablePoint point =
          load_table_point(table, table_offset(layout, row) + digit - 1u);
      result = add_affine(result, point.x, point.y);
    }
  }
  return result;
}

MOM_NEXAPOW_ALWAYS_INLINE Field store_curve(const CurvePoint& point, const uint32_t index,
                                            const PointPlanes& output) {
  if (point.infinity) {
    output.x[index] = output.y[index] = output.z[index] = field_zero();
    return field_zero();
  }
  const Field z = field_mul(point.zz, point.zzz);
  output.x[index] = field_mul(point.x, point.zzz);
  output.y[index] = field_mul(point.y, point.zz);
  output.z[index] = z;
  return z;
}

inline void tree_up(const sycl::nd_item<1>& item, const sycl::local_accessor<Field, 1>& tree) {
  const uint32_t lane = item.get_local_linear_id();
  for (uint32_t nodes = kWorkGroup / 2u; nodes; nodes >>= 1u) {
    if (lane < nodes) {
      const uint32_t node = nodes + lane;
      tree[node] = field_mul(tree[2u * node], tree[2u * node + 1u]);
    }
#ifdef MOM_NEXAPOW_USE_NATIVE_FIELD
    if (nodes >= 64u)
      item.barrier(sycl::access::fence_space::local_space);
    else
      sycl::group_barrier(item.get_sub_group());
#else
    item.barrier(sycl::access::fence_space::local_space);
#endif
  }
}

inline void tree_root_barrier(const sycl::nd_item<1>& item) {
#ifdef MOM_NEXAPOW_USE_NATIVE_FIELD
  sycl::group_barrier(item.get_sub_group());
#else
  item.barrier(sycl::access::fence_space::local_space);
#endif
}

inline void tree_down(const sycl::nd_item<1>& item, const sycl::local_accessor<Field, 1>& tree) {
  const uint32_t lane = item.get_local_linear_id();
  for (uint32_t nodes = 1u; nodes != kWorkGroup; nodes <<= 1u) {
    if (lane < nodes) {
      const uint32_t node = nodes + lane;
      const Field inverse = tree[node], left = tree[2u * node], right = tree[2u * node + 1u];
      tree[2u * node] = field_mul(inverse, right);
      tree[2u * node + 1u] = field_mul(inverse, left);
    }
#ifdef MOM_NEXAPOW_USE_NATIVE_FIELD
    if (nodes >= 32u)
      item.barrier(sycl::access::fence_space::local_space);
    else
      sycl::group_barrier(item.get_sub_group());
#else
    item.barrier(sycl::access::fence_space::local_space);
#endif
  }
}
#undef MOM_NEXAPOW_USE_NATIVE_FIELD
#undef MOM_NEXAPOW_FIELD_INLINE
#undef MOM_NEXAPOW_ALWAYS_INLINE
class ShiftedTableKernel;
class LargeTableKernel;
class TableCheckKernel;
class HashKernel;
class InitialHashKernel;
class Rfc6979Kernel;
class PublicCurveKernel;
class NonceCurveKernel;
class NonceCurveProductsKernel;
class ProductsKernel;
class BlockInverseKernel;
class FinishKernel;

} // namespace sycl_pipeline

class NexaPowSyclSearch {
  using Affine = sycl_pipeline::AffineTablePoint;
  using Field = sycl_pipeline::Field;
  using Planes = sycl_pipeline::PointPlanes;
  using SearchResult = sycl_pipeline::SearchResult;

  sycl_pipeline::TableLayout layout_ = sycl_pipeline::kSmallTable;
  Affine* table_ = nullptr;
  U256 *private_keys_ = nullptr, *nonce_keys_ = nullptr, *h1_ = nullptr;
  uint32_t* valid_ = nullptr;
  Planes public_{}, nonce_{};
  Field *products_ = nullptr, *inverses_ = nullptr;
  uint8_t* target_ = nullptr;
  SearchResult* result_ = nullptr;
  uint32_t capacity_ = 0;
  bool checked_ = false, enabled_ = false;
  uint32_t searches_ = 0;
  std::string error_;

  template <typename T> static void release_one(T*& pointer, sycl::queue& queue,
                                               const bool checked = false) {
    if (!pointer)
      return;
    // A throwing free may already have retired it; never retry the same pointer.
    T* allocation = pointer;
    pointer = nullptr;
    try {
      sycl::free(allocation, queue);
    } catch (...) {
      if (checked)
        throw;
    }
  }

  void release_buffers(sycl::queue& queue) noexcept {
    release_one(result_, queue);
    release_one(target_, queue);
    release_one(inverses_, queue);
    release_one(products_, queue);
    release_one(nonce_.z, queue);
    release_one(nonce_.y, queue);
    release_one(nonce_.x, queue);
    release_one(public_.z, queue);
    release_one(public_.y, queue);
    release_one(public_.x, queue);
    release_one(valid_, queue);
    release_one(h1_, queue);
    release_one(nonce_keys_, queue);
    release_one(private_keys_, queue);
    capacity_ = 0;
  }

  void allocate_buffers(sycl::queue& queue, const uint32_t count) {
    if (count <= capacity_ && result_)
      return;
    release_buffers(queue);
    const size_t blocks =
        (static_cast<size_t>(count) + sycl_pipeline::kWorkGroup - 1u) / sycl_pipeline::kWorkGroup;
    try {
      private_keys_ = sycl::malloc_device<U256>(count, queue);
      nonce_keys_ = sycl::malloc_device<U256>(count, queue);
      h1_ = sycl::malloc_device<U256>(count, queue);
      valid_ = sycl::malloc_device<uint32_t>(count, queue);
      public_.x = sycl::malloc_device<Field>(count, queue);
      public_.y = sycl::malloc_device<Field>(count, queue);
      public_.z = sycl::malloc_device<Field>(count, queue);
      nonce_.x = sycl::malloc_device<Field>(count, queue);
      nonce_.y = sycl::malloc_device<Field>(count, queue);
      nonce_.z = sycl::malloc_device<Field>(count, queue);
      products_ = sycl::malloc_device<Field>(blocks, queue);
      inverses_ = sycl::malloc_device<Field>(blocks, queue);
      target_ = sycl::malloc_device<uint8_t>(32, queue);
      result_ = sycl::malloc_device<SearchResult>(1, queue);
      if (!private_keys_ || !nonce_keys_ || !h1_ || !valid_ || !public_.x || !public_.y ||
          !public_.z || !nonce_.x || !nonce_.y || !nonce_.z || !products_ || !inverses_ ||
          !target_ || !result_)
        throw std::string("NexaPoW staged SYCL buffer allocation failed");
    } catch (...) {
      release_buffers(queue);
      throw;
    }
    capacity_ = count;
  }

  bool setup_tables(sycl::queue& queue, const sycl_pipeline::TableLayout layout,
                    std::string& failure) {
    Affine* compact = nullptr;
    Affine* shifted = nullptr;
    try {
      compact = sycl::malloc_device<Affine>(sycl_pipeline::kCompactPoints, queue);
      shifted = sycl::malloc_device<Affine>(7u * sycl_pipeline::kCompactPoints, queue);
      if (!compact || !shifted)
        throw std::string("NexaPoW staged SYCL seed-table allocation failed");
      table_ = sycl::malloc_device<Affine>(sycl_pipeline::table_points(layout), queue);
      if (!table_)
        throw std::string("NexaPoW staged SYCL table allocation failed");
    } catch (...) {
      // Allocation-only absence is recoverable: no table commands have been submitted yet.
      release_one(compact, queue);
      release_one(shifted, queue);
      release_one(table_, queue);
      failure = "NexaPoW staged SYCL table allocation failed";
      return false;
    }
    // The readback destination must outlive the catch-path queue drain.
    Affine check{};
    try {
      queue.memcpy(compact, kGeneratorWindow8, sizeof(kGeneratorWindow8));
      queue.submit([&](sycl::handler& handler) {
        handler.parallel_for<sycl_pipeline::ShiftedTableKernel>(
            sycl::range<1>(7u * sycl_pipeline::kCompactPoints),
            [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
              const uint32_t ordinal = id[0], bank = ordinal / sycl_pipeline::kCompactPoints;
              const uint32_t within = ordinal % sycl_pipeline::kCompactPoints;
              const uint32_t row = within / 255u, digit = within % 255u + 1u;
              const uint32_t shift = bank + 1u, low = (digit << shift) & 255u;
              const uint32_t high = digit >> (8u - shift);
              sycl_pipeline::CurvePoint sum = sycl_pipeline::infinity();
              if (low)
                sum = sycl_pipeline::add_affine(sum, compact[row * 255u + low - 1u].x,
                                                compact[row * 255u + low - 1u].y);
              if (high && row + 1u != 32u)
                sum = sycl_pipeline::add_affine(sum, compact[(row + 1u) * 255u + high - 1u].x,
                                                compact[(row + 1u) * 255u + high - 1u].y);
              shifted[ordinal] = high && row + 1u == 32u ? Affine{} : sycl_pipeline::affine(sum);
            });
      });
      for (uint32_t window = 0; window != layout.total_rows; ++window) {
        const uint32_t entries = sycl_pipeline::table_entries(
            sycl_pipeline::table_bits(layout, window));
        Affine* output = table_ + sycl_pipeline::table_offset(layout, window);
        queue.submit([&](sycl::handler& handler) {
          handler.parallel_for<sycl_pipeline::LargeTableKernel>(
              sycl::range<1>(entries), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
                const uint32_t ordinal = id[0];
                const uint32_t digit = ordinal + 1u;
                const uint32_t start = sycl_pipeline::table_start(layout, window);
                const uint32_t shift = start & 7u, row = start / 8u;
                const Affine* input =
                    shift ? shifted + (shift - 1u) * sycl_pipeline::kCompactPoints : compact;
                sycl_pipeline::CurvePoint sum = sycl_pipeline::infinity();
                for (uint32_t part = 0; part != 3u; ++part) {
                  const uint32_t part_digit = (digit >> (8u * part)) & 255u;
                  if (part_digit)
                    sum = sycl_pipeline::add_affine(sum,
                                                    input[(row + part) * 255u + part_digit - 1u].x,
                                                    input[(row + part) * 255u + part_digit - 1u].y);
                }
                output[ordinal] = sycl_pipeline::affine(sum);
              });
        });
      }
      const U256 check_scalar{{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                               0xffffffffu, 0xffffffffu, 0xffffffffu}};
      const Affine* table = table_;
      const sycl::event table_check = queue.submit([&](sycl::handler& handler) {
        handler.single_task<sycl_pipeline::TableCheckKernel>([=]() MOM_SYCL_KERNEL_ARGS_RESTRICT {
          shifted[0] = sycl_pipeline::affine(sycl_pipeline::multiply(check_scalar, table, layout));
        });
      });
      // Do not submit the host copy behind the long table build: some runtimes busy-spin during
      // that submission. Finish the self-check with the shared low-CPU wait first.
      sycl_wait_and_throw(table_check, queue.get_device());
      sycl_wait_and_throw(queue.memcpy(&check, shifted, sizeof(check)), queue.get_device());
      const Point expected = np_affine(np_scalar_mul(check_scalar));
      release_one(compact, queue, true);
      release_one(shifted, queue, true);
      if (np_cmp(check.x, expected.x) || np_cmp(check.y, expected.y)) {
        // Completed, wrong arithmetic can try the smaller layout; a failed dispatch cannot.
        release_one(table_, queue, true);
        failure = "NexaPoW staged SYCL table self-check failed";
        return false;
      }
      return true;
    } catch (...) {
      // Earlier table commands can still own these pointers after a later submission fails.
      sycl_cleanup_noexcept("nexapow staged setup wait", [&] { queue.wait_and_throw(); });
      release_one(compact, queue);
      release_one(shifted, queue);
      release_one(table_, queue);
      throw;
    }
  }

public:
  const char* table_name() const {
    return layout_.wide_bits == sycl_pipeline::kLargeTable.wide_bits ? "large" : "small";
  }

  uint32_t table_points() const {
    return sycl_pipeline::table_points(layout_);
  }

  uint64_t table_bytes() const {
    return static_cast<uint64_t>(table_points()) * sizeof(Affine);
  }

  bool ensure(sycl::queue& queue, const sycl::device& device, std::string* reason = nullptr) {
    if (checked_) {
      if (reason)
        *reason = error_;
      return enabled_;
    }
    const uint64_t memory = device.get_info<sycl::info::device::global_mem_size>();
    if (memory < (uint64_t{5} << 30)) {
      checked_ = true;
      error_ = "the staged SYCL table requires at least 5 GiB of device memory";
      if (reason)
        *reason = error_;
      return false;
    }

#if defined(_WIN32) && defined(MOM_SYCL_HAS_HIP)
    // Windows HIP retains the measured faster small layout; see the file header.
    const sycl_pipeline::TableLayout preferred = sycl_pipeline::kSmallTable;
#else
    const sycl_pipeline::TableLayout preferred =
        memory >= (uint64_t{10} << 30) ? sycl_pipeline::kLargeTable : sycl_pipeline::kSmallTable;
#endif
    const auto try_setup = [&](const sycl_pipeline::TableLayout layout, std::string& failure) {
      layout_ = layout;
      if (setup_tables(queue, layout, failure)) {
        enabled_ = true;
        error_.clear();
        return true;
      }
      return false;
    };

    std::string first_failure;
    if (!try_setup(preferred, first_failure) &&
        preferred.wide_bits == sycl_pipeline::kLargeTable.wide_bits) {
      const std::string large_failure = first_failure;
      std::string small_failure;
      if (!try_setup(sycl_pipeline::kSmallTable, small_failure))
        error_ = "large table setup failed: " + large_failure + "; small table setup failed: " +
                 small_failure;
    } else if (!first_failure.empty()) {
      error_ = first_failure;
    }
    // Latch only a completed setup decision, never a submitted-work exception.
    checked_ = true;
    if (reason)
      *reason = error_;
    return enabled_;
  }

  void release(sycl::queue& queue) noexcept {
    release_buffers(queue);
    release_one(table_, queue);
    enabled_ = false;
  }

  bool search(sycl::queue& queue, const uint8_t header[32], const uint8_t extranonce[8],
              const uint64_t first_nonce, const uint8_t target[32], const uint32_t count,
              const unsigned extra_bytes, uint64_t& found_nonce) {
    using namespace sycl_pipeline;
    const bool stats = std::getenv("MOM_NEXAPOW_STATS") && searches_++ == 1u;
    const char* split_hash_value = std::getenv("MOM_NEXAPOW_SPLIT_HASH");
    const bool split_hash = split_hash_value ? std::strcmp(split_hash_value, "0")
                                             : !sycl_is_level_zero_gpu(queue.get_device());
    const char* split_curve_value = std::getenv("MOM_NEXAPOW_SPLIT_CURVE_PRODUCTS");
    const bool split_curve = !split_curve_value || std::strcmp(split_curve_value, "0");
    auto last = std::chrono::steady_clock::now();
    const auto mark = [&](const char* stage) {
      if (!stats)
        return;
      queue.wait_and_throw();
      const auto now = std::chrono::steady_clock::now();
      std::fprintf(stderr, "NEXAPOW_SYCL_STATS %s=%.3fms\n", stage,
                   std::chrono::duration<double, std::milli>(now - last).count());
      last = now;
    };
    allocate_buffers(queue, count);
    SearchResult found{};
    MomSyclHostTransferGuard host_transfers(queue, "nexapow staged host transfers");
    queue.memcpy(target_, target, 32);
    queue.memset(result_, 0, sizeof(SearchResult));
    U256* private_keys = private_keys_;
    U256* nonce_keys = nonce_keys_;
    U256* h1 = h1_;
    uint32_t* valid = valid_;
    const Affine* table = table_;
    const sycl_pipeline::TableLayout layout = layout_;
    const HeaderContext context = make_header_context(header, extranonce, extra_bytes);
    const size_t global = (static_cast<size_t>(count) + kWorkGroup - 1u) / kWorkGroup * kWorkGroup;
    if (split_hash) {
      queue.submit([&](sycl::handler& handler) {
        handler.parallel_for<InitialHashKernel>(
            sycl::nd_range<1>(global, kWorkGroup),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_work_group_size(kWorkGroup)]] MOM_SYCL_KERNEL_ARGS_RESTRICT {
              const uint32_t ordinal = item.get_global_linear_id();
              if (ordinal >= count)
                return;
              uint32_t first_words[8], mining_words[8], h1_words[8];
              fast_header_sha256(context, first_nonce + ordinal, first_words);
              fast_sha256_32(first_words, mining_words);
              fast_sha256_32(mining_words, h1_words);
              U256 private_key{}, h1_value{};
              for (uint32_t i = 0; i != 8u; ++i) {
                private_key.w[i] = mining_words[7u - i];
                h1_value.w[i] = h1_words[7u - i];
              }
              private_keys[ordinal] = private_key;
              h1[ordinal] = h1_value;
              valid[ordinal] = !np_zero_p(private_key) && np_cmp(private_key, kGroupOrder) < 0;
            });
      });
      queue.submit([&](sycl::handler& handler) {
        handler.parallel_for<Rfc6979Kernel>(
            sycl::nd_range<1>(global, kWorkGroup),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_work_group_size(kWorkGroup)]] MOM_SYCL_KERNEL_ARGS_RESTRICT {
              const uint32_t ordinal = item.get_global_linear_id();
              if (ordinal >= count || !valid[ordinal])
                return;
              uint32_t h1_words[8];
              for (uint32_t i = 0; i != 8u; ++i)
                h1_words[i] = h1[ordinal].w[7u - i];
              const U256 nonce_key = fast_rfc6979(h1_words, private_keys[ordinal]);
              nonce_keys[ordinal] = nonce_key;
              valid[ordinal] = !np_zero_p(nonce_key) && np_cmp(nonce_key, kGroupOrder) < 0;
            });
      });
    } else
      queue.submit([&](sycl::handler& handler) {
        handler.parallel_for<HashKernel>(
            sycl::nd_range<1>(global, kWorkGroup),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_work_group_size(kWorkGroup)]] MOM_SYCL_KERNEL_ARGS_RESTRICT {
              const uint32_t ordinal = item.get_global_linear_id();
              if (ordinal >= count)
                return;
              uint32_t first_words[8], mining_words[8], h1_words[8];
              fast_header_sha256(context, first_nonce + ordinal, first_words);
              fast_sha256_32(first_words, mining_words);
              fast_sha256_32(mining_words, h1_words);
              U256 private_key{}, h1_value{};
              for (uint32_t i = 0; i != 8u; ++i) {
                private_key.w[i] = mining_words[7u - i];
                h1_value.w[i] = h1_words[7u - i];
              }
              const U256 nonce_key = fast_rfc6979(h1_words, private_key);
              private_keys[ordinal] = private_key;
              nonce_keys[ordinal] = nonce_key;
              h1[ordinal] = h1_value;
              valid[ordinal] = !np_zero_p(private_key) && np_cmp(private_key, kGroupOrder) < 0 &&
                               !np_zero_p(nonce_key) && np_cmp(nonce_key, kGroupOrder) < 0;
            });
      });
    mark("hash");
    const Planes public_points = public_, nonce_points = nonce_;
    Field* products = products_;
    Field* inverses = inverses_;
    const uint8_t* device_target = target_;
    SearchResult* result = result_;
    queue.submit([&](sycl::handler& handler) {
      handler.parallel_for<PublicCurveKernel>(
          sycl::nd_range<1>(global, kPublicCurveWorkGroup),
          [=](sycl::nd_item<1> item)
              [[sycl::reqd_work_group_size(kPublicCurveWorkGroup)]]
              MOM_SYCL_KERNEL_ARGS_RESTRICT {
            const uint32_t ordinal = item.get_global_linear_id();
            if (ordinal < count)
              store_curve(multiply(private_keys[ordinal], table, layout), ordinal, public_points);
          });
    });
    mark("public");
    if (split_curve) {
      queue.submit([&](sycl::handler& handler) {
        handler.parallel_for<NonceCurveKernel>(
            sycl::nd_range<1>(global, kNonceCurveWorkGroup),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_work_group_size(kNonceCurveWorkGroup)]]
                MOM_SYCL_KERNEL_ARGS_RESTRICT {
              const uint32_t ordinal = item.get_global_linear_id();
              if (ordinal < count && valid[ordinal])
                store_curve(multiply(nonce_keys[ordinal], table, layout), ordinal, nonce_points);
            });
      });
      mark("nonce");
      queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<Field, 1> tree(sycl::range<1>(2u * kWorkGroup), handler);
        handler.parallel_for<ProductsKernel>(
            sycl::nd_range<1>(global, kWorkGroup),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_work_group_size(kWorkGroup)]] MOM_SYCL_KERNEL_ARGS_RESTRICT {
              const uint32_t ordinal = item.get_global_linear_id(),
                             lane = item.get_local_linear_id();
              Field value = field_one();
              if (ordinal < count && valid[ordinal])
                value = field_mul(public_points.z[ordinal], nonce_points.z[ordinal]);
              tree[kWorkGroup + lane] = value;
              item.barrier(sycl::access::fence_space::local_space);
              tree_up(item, tree);
              if (!lane)
                products[item.get_group_linear_id()] = tree[1];
            });
      });
      mark("products");
    } else {
      queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<Field, 1> tree(sycl::range<1>(2u * kWorkGroup), handler);
        handler.parallel_for<NonceCurveProductsKernel>(
            sycl::nd_range<1>(global, kWorkGroup),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_work_group_size(kWorkGroup)]] MOM_SYCL_KERNEL_ARGS_RESTRICT {
              const uint32_t ordinal = item.get_global_linear_id(),
                             lane = item.get_local_linear_id();
              const bool active = ordinal < count && valid[ordinal];
              const CurvePoint point =
                  active ? multiply(nonce_keys[ordinal], table, layout) : infinity();
              const Field nonce_z =
                  active ? store_curve(point, ordinal, nonce_points) : field_zero();
              Field value = field_one();
              if (active)
                value = field_mul(public_points.z[ordinal], nonce_z);
              tree[kWorkGroup + lane] = value;
              item.barrier(sycl::access::fence_space::local_space);
              tree_up(item, tree);
              if (!lane)
                products[item.get_group_linear_id()] = tree[1];
            });
      });
      mark("nonce-products");
    }
    const uint32_t blocks = static_cast<uint32_t>(global / kWorkGroup);
    const size_t inverse_global =
        (static_cast<size_t>(blocks) + kWorkGroup - 1u) / kWorkGroup * kWorkGroup;
    queue.submit([&](sycl::handler& handler) {
      sycl::local_accessor<Field, 1> tree(sycl::range<1>(2u * kWorkGroup), handler);
      handler.parallel_for<BlockInverseKernel>(
          sycl::nd_range<1>(inverse_global, kWorkGroup),
          [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
            const uint32_t ordinal = item.get_global_linear_id(), lane = item.get_local_linear_id();
            tree[kWorkGroup + lane] = ordinal < blocks ? products[ordinal] : field_one();
            item.barrier(sycl::access::fence_space::local_space);
            tree_up(item, tree);
            if (!lane)
              tree[1] = field_inverse(tree[1]);
            tree_root_barrier(item);
            tree_down(item, tree);
            if (ordinal < blocks)
              inverses[ordinal] = tree[kWorkGroup + lane];
          });
    });
    mark("inverse");
    const sycl::event finish = queue.submit([&](sycl::handler& handler) {
      sycl::local_accessor<Field, 1> tree(sycl::range<1>(2u * kWorkGroup), handler);
      handler.parallel_for<FinishKernel>(
          sycl::nd_range<1>(global, kWorkGroup),
          [=](sycl::nd_item<1> item)
              [[sycl::reqd_work_group_size(kWorkGroup)]] MOM_SYCL_KERNEL_ARGS_RESTRICT {
            const uint32_t ordinal = item.get_global_linear_id(), lane = item.get_local_linear_id();
            Field value = field_one();
            const bool active = ordinal < count && valid[ordinal];
            if (active)
              value = field_mul(public_points.z[ordinal], nonce_points.z[ordinal]);
            tree[kWorkGroup + lane] = value;
            item.barrier(sycl::access::fence_space::local_space);
            tree_up(item, tree);
            if (!lane)
              tree[1] = inverses[item.get_group_linear_id()];
            tree_root_barrier(item);
            tree_down(item, tree);
            if (!active)
              return;
            const Field product_inverse = tree[kWorkGroup + lane];
            const Field public_inverse = field_mul(nonce_points.z[ordinal], product_inverse);
            const Field nonce_inverse = field_mul(public_points.z[ordinal], product_inverse);
            const U256 public_x =
                field_to_u256(field_mul(public_points.x[ordinal], public_inverse));
            const U256 public_y =
                field_to_u256(field_mul(public_points.y[ordinal], public_inverse));
            const U256 nonce_x = field_to_u256(field_mul(nonce_points.x[ordinal], nonce_inverse));
            finish_signature(ordinal, first_nonce, public_x, public_y.w[0] & 1u, nonce_x,
                             nonce_points.y[ordinal], nonce_points.z[ordinal], h1[ordinal],
                             private_keys[ordinal], nonce_keys[ordinal], device_target, result);
          });
    });
    // Do not submit the host copy behind the search: some runtimes busy-spin during that
    // submission. Complete the final kernel with the shared low-CPU wait first.
    sycl_wait_and_throw(finish, queue.get_device());
    mark("finish");
    sycl_wait_and_throw(queue.memcpy(&found, result_, sizeof(found)), queue.get_device());
    if (!found.count)
      return false;
    found_nonce = found.nonce;
    return true;
  }
};
