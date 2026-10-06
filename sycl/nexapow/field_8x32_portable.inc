// Standards-only secp256k1 field arithmetic for GPU compilers. Inputs and outputs are canonical
// little-endian 8x32 limbs; fixed expansion keeps limb temporaries in registers.
#include <cstdint>

namespace mom_nexa_field32_portable {

#if defined(__clang__) || defined(__GNUC__)
#define MOM_NEXAPOW_PORTABLE_INLINE inline __attribute__((always_inline))
#else
#define MOM_NEXAPOW_PORTABLE_INLINE inline
#endif

struct Accumulator {
  uint32_t low = 0, middle = 0, high = 0, low_carry = 0;
};

// Keep low and middle carry chains independent; normalize them once per emitted column.
MOM_NEXAPOW_PORTABLE_INLINE void add_value(Accumulator& acc, const uint32_t low,
                                           const uint32_t middle, const uint32_t high) {
  const uint32_t old_low = acc.low;
  acc.low += low;
  acc.low_carry += acc.low < old_low;
  const uint32_t old_middle = acc.middle;
  acc.middle += middle;
  acc.high += high + (acc.middle < old_middle);
}

MOM_NEXAPOW_PORTABLE_INLINE void add_product(Accumulator& acc, const uint32_t a, const uint32_t b) {
#if defined(__SYCL_DEVICE_ONLY__)
  add_value(acc, a * b, sycl::mul_hi(a, b), 0u);
#else
  const uint64_t product = static_cast<uint64_t>(a) * b;
  add_value(acc, static_cast<uint32_t>(product), static_cast<uint32_t>(product >> 32u), 0u);
#endif
}

MOM_NEXAPOW_PORTABLE_INLINE void add_doubled_product(Accumulator& acc, const uint32_t a,
                                                     const uint32_t b) {
#if defined(__SYCL_DEVICE_ONLY__)
  const uint32_t low = a * b, high = sycl::mul_hi(a, b);
#else
  const uint64_t product = static_cast<uint64_t>(a) * b;
  const uint32_t low = static_cast<uint32_t>(product), high = static_cast<uint32_t>(product >> 32u);
#endif
  add_value(acc, low << 1u, (high << 1u) | (low >> 31u), high >> 31u);
}

MOM_NEXAPOW_PORTABLE_INLINE void output_limb(Accumulator& acc, uint32_t& output) {
  output = acc.low;
  const uint32_t old_middle = acc.middle;
  acc.low = acc.middle + acc.low_carry;
  acc.middle = acc.high + (acc.low < old_middle);
  acc.high = 0;
  acc.low_carry = 0;
}

MOM_NEXAPOW_PORTABLE_INLINE void multiply_wide(const uint32_t a[8], const uint32_t b[8],
                                               uint32_t output[16]) {
  Accumulator acc{};
#define MOM_NEXAPOW_ACC(i, j) add_product(acc, a[i], b[j])
#define MOM_NEXAPOW_OUT(i) output_limb(acc, output[i])
  MOM_NEXAPOW_ACC(0, 0);
  MOM_NEXAPOW_OUT(0);
  MOM_NEXAPOW_ACC(0, 1);
  MOM_NEXAPOW_ACC(1, 0);
  MOM_NEXAPOW_OUT(1);
  MOM_NEXAPOW_ACC(0, 2);
  MOM_NEXAPOW_ACC(1, 1);
  MOM_NEXAPOW_ACC(2, 0);
  MOM_NEXAPOW_OUT(2);
  MOM_NEXAPOW_ACC(0, 3);
  MOM_NEXAPOW_ACC(1, 2);
  MOM_NEXAPOW_ACC(2, 1);
  MOM_NEXAPOW_ACC(3, 0);
  MOM_NEXAPOW_OUT(3);
  MOM_NEXAPOW_ACC(0, 4);
  MOM_NEXAPOW_ACC(1, 3);
  MOM_NEXAPOW_ACC(2, 2);
  MOM_NEXAPOW_ACC(3, 1);
  MOM_NEXAPOW_ACC(4, 0);
  MOM_NEXAPOW_OUT(4);
  MOM_NEXAPOW_ACC(0, 5);
  MOM_NEXAPOW_ACC(1, 4);
  MOM_NEXAPOW_ACC(2, 3);
  MOM_NEXAPOW_ACC(3, 2);
  MOM_NEXAPOW_ACC(4, 1);
  MOM_NEXAPOW_ACC(5, 0);
  MOM_NEXAPOW_OUT(5);
  MOM_NEXAPOW_ACC(0, 6);
  MOM_NEXAPOW_ACC(1, 5);
  MOM_NEXAPOW_ACC(2, 4);
  MOM_NEXAPOW_ACC(3, 3);
  MOM_NEXAPOW_ACC(4, 2);
  MOM_NEXAPOW_ACC(5, 1);
  MOM_NEXAPOW_ACC(6, 0);
  MOM_NEXAPOW_OUT(6);
  MOM_NEXAPOW_ACC(0, 7);
  MOM_NEXAPOW_ACC(1, 6);
  MOM_NEXAPOW_ACC(2, 5);
  MOM_NEXAPOW_ACC(3, 4);
  MOM_NEXAPOW_ACC(4, 3);
  MOM_NEXAPOW_ACC(5, 2);
  MOM_NEXAPOW_ACC(6, 1);
  MOM_NEXAPOW_ACC(7, 0);
  MOM_NEXAPOW_OUT(7);
  MOM_NEXAPOW_ACC(1, 7);
  MOM_NEXAPOW_ACC(2, 6);
  MOM_NEXAPOW_ACC(3, 5);
  MOM_NEXAPOW_ACC(4, 4);
  MOM_NEXAPOW_ACC(5, 3);
  MOM_NEXAPOW_ACC(6, 2);
  MOM_NEXAPOW_ACC(7, 1);
  MOM_NEXAPOW_OUT(8);
  MOM_NEXAPOW_ACC(2, 7);
  MOM_NEXAPOW_ACC(3, 6);
  MOM_NEXAPOW_ACC(4, 5);
  MOM_NEXAPOW_ACC(5, 4);
  MOM_NEXAPOW_ACC(6, 3);
  MOM_NEXAPOW_ACC(7, 2);
  MOM_NEXAPOW_OUT(9);
  MOM_NEXAPOW_ACC(3, 7);
  MOM_NEXAPOW_ACC(4, 6);
  MOM_NEXAPOW_ACC(5, 5);
  MOM_NEXAPOW_ACC(6, 4);
  MOM_NEXAPOW_ACC(7, 3);
  MOM_NEXAPOW_OUT(10);
  MOM_NEXAPOW_ACC(4, 7);
  MOM_NEXAPOW_ACC(5, 6);
  MOM_NEXAPOW_ACC(6, 5);
  MOM_NEXAPOW_ACC(7, 4);
  MOM_NEXAPOW_OUT(11);
  MOM_NEXAPOW_ACC(5, 7);
  MOM_NEXAPOW_ACC(6, 6);
  MOM_NEXAPOW_ACC(7, 5);
  MOM_NEXAPOW_OUT(12);
  MOM_NEXAPOW_ACC(6, 7);
  MOM_NEXAPOW_ACC(7, 6);
  MOM_NEXAPOW_OUT(13);
  MOM_NEXAPOW_ACC(7, 7);
  MOM_NEXAPOW_OUT(14);
  output[15] = acc.low;
#undef MOM_NEXAPOW_OUT
#undef MOM_NEXAPOW_ACC
}

MOM_NEXAPOW_PORTABLE_INLINE void subtract_product_limb(uint32_t& output, const uint32_t product,
                                                       uint32_t& borrow) {
  const uint32_t difference = output - product;
  const uint32_t product_borrow = output < product;
  output = difference - borrow;
  borrow = product_borrow | (difference < borrow);
}

MOM_NEXAPOW_PORTABLE_INLINE uint32_t subtract_product_wide(uint32_t output[16], const uint32_t a[8],
                                                          const uint32_t b[8]) {
  Accumulator acc{};
  uint32_t borrow = 0;
#define MOM_NEXAPOW_ACC(i, j) add_product(acc, a[i], b[j])
#define MOM_NEXAPOW_SUB_OUT(i)                                                                    \
  do {                                                                                             \
    uint32_t product_limb;                                                                         \
    output_limb(acc, product_limb);                                                                \
    subtract_product_limb(output[i], product_limb, borrow);                                        \
  } while (0)
  MOM_NEXAPOW_ACC(0, 0);
  MOM_NEXAPOW_SUB_OUT(0);
  MOM_NEXAPOW_ACC(0, 1);
  MOM_NEXAPOW_ACC(1, 0);
  MOM_NEXAPOW_SUB_OUT(1);
  MOM_NEXAPOW_ACC(0, 2);
  MOM_NEXAPOW_ACC(1, 1);
  MOM_NEXAPOW_ACC(2, 0);
  MOM_NEXAPOW_SUB_OUT(2);
  MOM_NEXAPOW_ACC(0, 3);
  MOM_NEXAPOW_ACC(1, 2);
  MOM_NEXAPOW_ACC(2, 1);
  MOM_NEXAPOW_ACC(3, 0);
  MOM_NEXAPOW_SUB_OUT(3);
  MOM_NEXAPOW_ACC(0, 4);
  MOM_NEXAPOW_ACC(1, 3);
  MOM_NEXAPOW_ACC(2, 2);
  MOM_NEXAPOW_ACC(3, 1);
  MOM_NEXAPOW_ACC(4, 0);
  MOM_NEXAPOW_SUB_OUT(4);
  MOM_NEXAPOW_ACC(0, 5);
  MOM_NEXAPOW_ACC(1, 4);
  MOM_NEXAPOW_ACC(2, 3);
  MOM_NEXAPOW_ACC(3, 2);
  MOM_NEXAPOW_ACC(4, 1);
  MOM_NEXAPOW_ACC(5, 0);
  MOM_NEXAPOW_SUB_OUT(5);
  MOM_NEXAPOW_ACC(0, 6);
  MOM_NEXAPOW_ACC(1, 5);
  MOM_NEXAPOW_ACC(2, 4);
  MOM_NEXAPOW_ACC(3, 3);
  MOM_NEXAPOW_ACC(4, 2);
  MOM_NEXAPOW_ACC(5, 1);
  MOM_NEXAPOW_ACC(6, 0);
  MOM_NEXAPOW_SUB_OUT(6);
  MOM_NEXAPOW_ACC(0, 7);
  MOM_NEXAPOW_ACC(1, 6);
  MOM_NEXAPOW_ACC(2, 5);
  MOM_NEXAPOW_ACC(3, 4);
  MOM_NEXAPOW_ACC(4, 3);
  MOM_NEXAPOW_ACC(5, 2);
  MOM_NEXAPOW_ACC(6, 1);
  MOM_NEXAPOW_ACC(7, 0);
  MOM_NEXAPOW_SUB_OUT(7);
  MOM_NEXAPOW_ACC(1, 7);
  MOM_NEXAPOW_ACC(2, 6);
  MOM_NEXAPOW_ACC(3, 5);
  MOM_NEXAPOW_ACC(4, 4);
  MOM_NEXAPOW_ACC(5, 3);
  MOM_NEXAPOW_ACC(6, 2);
  MOM_NEXAPOW_ACC(7, 1);
  MOM_NEXAPOW_SUB_OUT(8);
  MOM_NEXAPOW_ACC(2, 7);
  MOM_NEXAPOW_ACC(3, 6);
  MOM_NEXAPOW_ACC(4, 5);
  MOM_NEXAPOW_ACC(5, 4);
  MOM_NEXAPOW_ACC(6, 3);
  MOM_NEXAPOW_ACC(7, 2);
  MOM_NEXAPOW_SUB_OUT(9);
  MOM_NEXAPOW_ACC(3, 7);
  MOM_NEXAPOW_ACC(4, 6);
  MOM_NEXAPOW_ACC(5, 5);
  MOM_NEXAPOW_ACC(6, 4);
  MOM_NEXAPOW_ACC(7, 3);
  MOM_NEXAPOW_SUB_OUT(10);
  MOM_NEXAPOW_ACC(4, 7);
  MOM_NEXAPOW_ACC(5, 6);
  MOM_NEXAPOW_ACC(6, 5);
  MOM_NEXAPOW_ACC(7, 4);
  MOM_NEXAPOW_SUB_OUT(11);
  MOM_NEXAPOW_ACC(5, 7);
  MOM_NEXAPOW_ACC(6, 6);
  MOM_NEXAPOW_ACC(7, 5);
  MOM_NEXAPOW_SUB_OUT(12);
  MOM_NEXAPOW_ACC(6, 7);
  MOM_NEXAPOW_ACC(7, 6);
  MOM_NEXAPOW_SUB_OUT(13);
  MOM_NEXAPOW_ACC(7, 7);
  MOM_NEXAPOW_SUB_OUT(14);
  subtract_product_limb(output[15], acc.low, borrow);
#undef MOM_NEXAPOW_SUB_OUT
#undef MOM_NEXAPOW_ACC
  return 0u - borrow;
}

// Explicit 32-bit reduction carries avoid widened serial chains.
MOM_NEXAPOW_PORTABLE_INLINE uint32_t reduce_limb(
    const uint32_t input_low, const uint32_t input_high,
    const uint32_t shifted_high, const uint32_t carry_in, uint32_t& out) {
#if defined(__SYCL_DEVICE_ONLY__)
  const uint32_t product_low = input_high * 977u;
  const uint32_t product_high = sycl::mul_hi(input_high, 977u);
#else
  const uint64_t product = static_cast<uint64_t>(input_high) * 977u;
  const uint32_t product_low = static_cast<uint32_t>(product);
  const uint32_t product_high = static_cast<uint32_t>(product >> 32u);
#endif
  uint32_t low = product_low;
  uint32_t high = product_high;
  uint32_t old_low = low;
  low += input_low;
  high += (low < old_low);
  old_low = low;
  low += shifted_high;
  high += (low < old_low);
  old_low = low;
  low += carry_in;
  high += (low < old_low);
  out = low;
  return high;
}

MOM_NEXAPOW_PORTABLE_INLINE void reduce(const uint32_t input[16], uint32_t output[8]) {
  uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
  uint32_t carry = reduce_limb(input[0], input[8], 0u, 0u, r0);
  carry = reduce_limb(input[1], input[9], input[8], carry, r1);
  carry = reduce_limb(input[2], input[10], input[9], carry, r2);
  carry = reduce_limb(input[3], input[11], input[10], carry, r3);
  carry = reduce_limb(input[4], input[12], input[11], carry, r4);
  carry = reduce_limb(input[5], input[13], input[12], carry, r5);
  carry = reduce_limb(input[6], input[14], input[13], carry, r6);
  carry = reduce_limb(input[7], input[15], input[14], carry, r7);
  uint64_t high = static_cast<uint64_t>(input[15]) + carry;
  uint64_t sum;
  // The initial high may reach 2^32 + 977, so retain its top bit through the two affected columns.
#define MOM_NEXAPOW_FOLD_HIGH()                                                                              \
  do {                                                                                                       \
    const uint32_t high_low = static_cast<uint32_t>(high);                                                   \
    const uint32_t high_top = static_cast<uint32_t>(high >> 32u);                                            \
    uint32_t fold_carry = reduce_limb(r0, high_low, 0u, 0u, r0);                                             \
    fold_carry += high_top * 977u;                                                                           \
    fold_carry = reduce_limb(r1, 0u, high_low, fold_carry, r1);                                              \
    fold_carry = reduce_limb(r2, 0u, high_top, fold_carry, r2);                                              \
    fold_carry = reduce_limb(r3, 0u, 0u, fold_carry, r3);                                                    \
    fold_carry = reduce_limb(r4, 0u, 0u, fold_carry, r4);                                                    \
    fold_carry = reduce_limb(r5, 0u, 0u, fold_carry, r5);                                                    \
    fold_carry = reduce_limb(r6, 0u, 0u, fold_carry, r6);                                                    \
    fold_carry = reduce_limb(r7, 0u, 0u, fold_carry, r7);                                                    \
    high = fold_carry;                                                                                       \
  } while (0)
  MOM_NEXAPOW_FOLD_HIGH();
  // A zero high means the remaining folds are identities.
  if (high) {
    MOM_NEXAPOW_FOLD_HIGH();
    if (high)
      MOM_NEXAPOW_FOLD_HIGH();
  }
#undef MOM_NEXAPOW_FOLD_HIGH
  if (r7 == 0xffffffffu && r6 == 0xffffffffu && r5 == 0xffffffffu && r4 == 0xffffffffu &&
      r3 == 0xffffffffu && r2 == 0xffffffffu &&
      (r1 == 0xffffffffu || (r1 == 0xfffffffeu && r0 >= 0xfffffc2fu))) {
    sum = static_cast<uint64_t>(r0) + 977u;
    r0 = static_cast<uint32_t>(sum);
    sum = (sum >> 32u) + r1 + 1u;
    r1 = static_cast<uint32_t>(sum);
    sum = (sum >> 32u) + r2;
    r2 = static_cast<uint32_t>(sum);
    sum = (sum >> 32u) + r3;
    r3 = static_cast<uint32_t>(sum);
    sum = (sum >> 32u) + r4;
    r4 = static_cast<uint32_t>(sum);
    sum = (sum >> 32u) + r5;
    r5 = static_cast<uint32_t>(sum);
    sum = (sum >> 32u) + r6;
    r6 = static_cast<uint32_t>(sum);
    r7 += static_cast<uint32_t>(sum >> 32u);
  }
  output[0] = r0;
  output[1] = r1;
  output[2] = r2;
  output[3] = r3;
  output[4] = r4;
  output[5] = r5;
  output[6] = r6;
  output[7] = r7;
}

MOM_NEXAPOW_PORTABLE_INLINE void mul(uint32_t output[8], const uint32_t a[8], const uint32_t b[8]) {
  uint32_t wide[16];
  multiply_wide(a, b, wide);
  reduce(wide, output);
}

MOM_NEXAPOW_PORTABLE_INLINE void square(uint32_t output[8], const uint32_t value[8]) {
  Accumulator acc{};
  uint32_t wide[16];
#define MOM_NEXAPOW_DIAG(i) add_product(acc, value[i], value[i])
#define MOM_NEXAPOW_DOUBLE(i, j) add_doubled_product(acc, value[i], value[j])
#define MOM_NEXAPOW_SQUARE_OUT(i) output_limb(acc, wide[i])
  MOM_NEXAPOW_DIAG(0);
  MOM_NEXAPOW_SQUARE_OUT(0);
  MOM_NEXAPOW_DOUBLE(0, 1);
  MOM_NEXAPOW_SQUARE_OUT(1);
  MOM_NEXAPOW_DOUBLE(0, 2);
  MOM_NEXAPOW_DIAG(1);
  MOM_NEXAPOW_SQUARE_OUT(2);
  MOM_NEXAPOW_DOUBLE(0, 3);
  MOM_NEXAPOW_DOUBLE(1, 2);
  MOM_NEXAPOW_SQUARE_OUT(3);
  MOM_NEXAPOW_DOUBLE(0, 4);
  MOM_NEXAPOW_DOUBLE(1, 3);
  MOM_NEXAPOW_DIAG(2);
  MOM_NEXAPOW_SQUARE_OUT(4);
  MOM_NEXAPOW_DOUBLE(0, 5);
  MOM_NEXAPOW_DOUBLE(1, 4);
  MOM_NEXAPOW_DOUBLE(2, 3);
  MOM_NEXAPOW_SQUARE_OUT(5);
  MOM_NEXAPOW_DOUBLE(0, 6);
  MOM_NEXAPOW_DOUBLE(1, 5);
  MOM_NEXAPOW_DOUBLE(2, 4);
  MOM_NEXAPOW_DIAG(3);
  MOM_NEXAPOW_SQUARE_OUT(6);
  MOM_NEXAPOW_DOUBLE(0, 7);
  MOM_NEXAPOW_DOUBLE(1, 6);
  MOM_NEXAPOW_DOUBLE(2, 5);
  MOM_NEXAPOW_DOUBLE(3, 4);
  MOM_NEXAPOW_SQUARE_OUT(7);
  MOM_NEXAPOW_DOUBLE(1, 7);
  MOM_NEXAPOW_DOUBLE(2, 6);
  MOM_NEXAPOW_DOUBLE(3, 5);
  MOM_NEXAPOW_DIAG(4);
  MOM_NEXAPOW_SQUARE_OUT(8);
  MOM_NEXAPOW_DOUBLE(2, 7);
  MOM_NEXAPOW_DOUBLE(3, 6);
  MOM_NEXAPOW_DOUBLE(4, 5);
  MOM_NEXAPOW_SQUARE_OUT(9);
  MOM_NEXAPOW_DOUBLE(3, 7);
  MOM_NEXAPOW_DOUBLE(4, 6);
  MOM_NEXAPOW_DIAG(5);
  MOM_NEXAPOW_SQUARE_OUT(10);
  MOM_NEXAPOW_DOUBLE(4, 7);
  MOM_NEXAPOW_DOUBLE(5, 6);
  MOM_NEXAPOW_SQUARE_OUT(11);
  MOM_NEXAPOW_DOUBLE(5, 7);
  MOM_NEXAPOW_DIAG(6);
  MOM_NEXAPOW_SQUARE_OUT(12);
  MOM_NEXAPOW_DOUBLE(6, 7);
  MOM_NEXAPOW_SQUARE_OUT(13);
  MOM_NEXAPOW_DIAG(7);
  MOM_NEXAPOW_SQUARE_OUT(14);
  wide[15] = acc.low;
#undef MOM_NEXAPOW_SQUARE_OUT
#undef MOM_NEXAPOW_DOUBLE
#undef MOM_NEXAPOW_DIAG
  reduce(wide, output);
}

MOM_NEXAPOW_PORTABLE_INLINE void sub(uint32_t output[8], const uint32_t a[8], const uint32_t b[8]) {
  // Use 32-bit borrow steps; read both operands before storing an aliased output.
  uint32_t borrow = 0;
#if defined(__clang__) || defined(__SYCL_DEVICE_ONLY__)
#pragma unroll
#endif
  for (uint32_t i = 0; i != 8u; ++i) {
    uint32_t limb = a[i];
    subtract_product_limb(limb, b[i], borrow);
    output[i] = limb;
  }
  const uint32_t mask = 0u - borrow;
  borrow = 0;
  subtract_product_limb(output[0], 977u & mask, borrow);
  subtract_product_limb(output[1], 1u & mask, borrow);
#if defined(__clang__) || defined(__SYCL_DEVICE_ONLY__)
#pragma unroll
#endif
  for (uint32_t i = 2; i != 8u; ++i) {
    subtract_product_limb(output[i], 0u, borrow);
  }
}

MOM_NEXAPOW_PORTABLE_INLINE void mul_sub_mul(uint32_t output[8], const uint32_t a[8],
                                             const uint32_t b[8], const uint32_t c[8],
                                             const uint32_t d[8]) {
  uint32_t wide[16];
  multiply_wide(a, b, wide);
  const uint32_t correction_mask = subtract_product_wide(wide, c, d);
  reduce(wide, output);
  // A wrapped 512-bit underflow adds 2^512 mod p = K^2, so subtract it after reduction.
  const uint32_t correction[8] = {0x000e90a1u & correction_mask, 0x000007a2u & correction_mask,
                                  0x00000001u & correction_mask, 0u, 0u, 0u, 0u, 0u};
  sub(output, output, correction);
}

#undef MOM_NEXAPOW_PORTABLE_INLINE
} // namespace mom_nexa_field32_portable
