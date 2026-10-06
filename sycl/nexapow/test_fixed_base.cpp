// Host-only differential test.  Compile with:
// c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic test_fixed_base.cpp -o /tmp/nexapow-fixed-base && /tmp/nexapow-fixed-base

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace mom_nexapow {
#include "device.inc"
#include "field_8x32_portable.inc"

inline bool same_point(const Point& left, const Point& right) {
  if (left.infinity || right.infinity)
    return left.infinity == right.infinity;
  const U256 left_z2 = np_field_square(left.z);
  const U256 right_z2 = np_field_square(right.z);
  const U256 left_x = np_field_mul(left.x, right_z2);
  const U256 right_x = np_field_mul(right.x, left_z2);
  if (np_cmp(left_x, right_x) != 0)
    return false;
  const U256 left_z3 = np_field_mul(left_z2, left.z);
  const U256 right_z3 = np_field_mul(right_z2, right.z);
  return np_cmp(np_field_mul(left.y, right_z3), np_field_mul(right.y, left_z3)) == 0;
}

inline bool sign_reference(const uint8_t h1[32], const U256& private_key, uint8_t signature[64]) {
  if (np_zero_p(private_key) || np_cmp(private_key, kGroupOrder) >= 0)
    return false;
  const Point public_key = np_affine(np_scalar_mul(private_key));
  for (unsigned counter = 0; counter != 256; ++counter) {
    U256 k = np_rfc6979(h1, private_key, counter);
    if (np_zero_p(k) || np_cmp(k, kGroupOrder) >= 0)
      continue;
    const Point nonce_point = np_affine(np_scalar_mul(k));
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

inline U256 scalar_from_words(const uint32_t words[8]) {
  U256 result{};
  for (unsigned i = 0; i < 8; ++i)
    result.w[i] = words[i];
  return result;
}

inline uint64_t next_random(uint64_t& state) {
  state ^= state << 7;
  state ^= state >> 9;
  state ^= state << 8;
  return state;
}

int check_scalar(const U256& scalar, const unsigned ordinal) {
  if (!same_point(np_scalar_mul(scalar), np_generator_mul(scalar))) {
    std::fprintf(stderr, "fixed-base mismatch for scalar %u\n", ordinal);
    return 1;
  }
  return 0;
}

inline int check_reduction(const std::array<uint32_t, 16>& input, const U256& expected) {
  U256 portable{};
  const U256 reference = np_field_reduce(input.data());
  mom_nexa_field32_portable::reduce(input.data(), portable.w);
  if (std::memcmp(portable.w, expected.w, sizeof(portable.w)) ||
      std::memcmp(reference.w, expected.w, sizeof(reference.w)) ||
      np_cmp(portable, kFieldPrime) >= 0 || np_cmp(reference, kFieldPrime) >= 0) {
    std::fputs("portable reduction regression failure\n", stderr);
    return 1;
  }

  std::array<uint32_t, 16> alias_low = input;
  std::array<uint32_t, 16> alias_high = input;
  mom_nexa_field32_portable::reduce(alias_low.data(), alias_low.data());
  mom_nexa_field32_portable::reduce(alias_high.data(), alias_high.data() + 8);
  if (std::memcmp(alias_low.data(), expected.w, sizeof(expected.w)) ||
      std::memcmp(alias_high.data() + 8, expected.w, sizeof(expected.w))) {
    std::fputs("portable reduction regression failure\n", stderr);
    return 1;
  }
  return 0;
}

inline int check_field_arithmetic(const U256& a, const U256& b, const U256& c,
                                  const U256& d) {
  const U256 expected_sub = np_field_sub(a, b);
  const U256 expected_mul = np_field_mul(a, b);
  const U256 expected_square = np_field_square(a);
  const U256 expected_fused = np_field_sub(np_field_mul(a, b), np_field_mul(c, d));

  U256 sub{};
  mom_nexa_field32_portable::sub(sub.w, a.w, b.w);
  if (std::memcmp(sub.w, expected_sub.w, sizeof(sub.w)) || np_cmp(sub, kFieldPrime) >= 0) {
    std::fputs("portable sub mismatch\n", stderr);
    return 1;
  }
  U256 mul{};
  mom_nexa_field32_portable::mul(mul.w, a.w, b.w);
  if (std::memcmp(mul.w, expected_mul.w, sizeof(mul.w)) || np_cmp(mul, kFieldPrime) >= 0) {
    std::fputs("portable mul mismatch\n", stderr);
    return 1;
  }
  for (unsigned alias = 0; alias < 2; ++alias) {
    std::array<U256, 2> operands{a, b};
    mom_nexa_field32_portable::sub(operands[alias].w, operands[0].w, operands[1].w);
    if (std::memcmp(operands[alias].w, expected_sub.w, sizeof(operands[alias].w)) ||
        np_cmp(operands[alias], kFieldPrime) >= 0) {
      std::fprintf(stderr, "portable sub output alias mismatch at %u\n", alias);
      return 1;
    }
    operands = {a, b};
    mom_nexa_field32_portable::mul(operands[alias].w, operands[0].w, operands[1].w);
    if (std::memcmp(operands[alias].w, expected_mul.w, sizeof(operands[alias].w)) ||
        np_cmp(operands[alias], kFieldPrime) >= 0) {
      std::fprintf(stderr, "portable mul output alias mismatch at %u\n", alias);
      return 1;
    }
  }
  {
    U256 output = a;
    mom_nexa_field32_portable::sub(output.w, output.w, output.w);
    const U256 zero{};
    if (std::memcmp(output.w, zero.w, sizeof(output.w))) {
      std::fputs("portable sub same-storage mismatch\n", stderr);
      return 1;
    }
  }
  {
    U256 output = a;
    mom_nexa_field32_portable::mul(output.w, output.w, output.w);
    if (std::memcmp(output.w, expected_square.w, sizeof(output.w)) ||
        np_cmp(output, kFieldPrime) >= 0) {
      std::fputs("portable mul same-storage mismatch\n", stderr);
      return 1;
    }
  }

  U256 square{};
  mom_nexa_field32_portable::square(square.w, a.w);
  if (std::memcmp(square.w, expected_square.w, sizeof(square.w)) ||
      np_cmp(square, kFieldPrime) >= 0) {
    std::fputs("portable square mismatch\n", stderr);
    return 1;
  }
  {
    U256 output = a;
    mom_nexa_field32_portable::square(output.w, output.w);
    if (std::memcmp(output.w, expected_square.w, sizeof(output.w)) ||
        np_cmp(output, kFieldPrime) >= 0) {
      std::fputs("portable square in-place mismatch\n", stderr);
      return 1;
    }
  }

  U256 fused{};
  mom_nexa_field32_portable::mul_sub_mul(fused.w, a.w, b.w, c.w, d.w);
  if (std::memcmp(fused.w, expected_fused.w, sizeof(fused.w)) ||
      np_cmp(fused, kFieldPrime) >= 0) {
    std::fputs("portable fused mismatch\n", stderr);
    return 1;
  }
  for (unsigned alias = 0; alias < 4; ++alias) {
    std::array<U256, 4> operands{a, b, c, d};
    mom_nexa_field32_portable::mul_sub_mul(operands[alias].w, operands[0].w, operands[1].w,
                                           operands[2].w, operands[3].w);
    if (std::memcmp(operands[alias].w, expected_fused.w, sizeof(operands[alias].w)) ||
        np_cmp(operands[alias], kFieldPrime) >= 0) {
      std::fprintf(stderr, "portable fused output alias mismatch at %u\n", alias);
      return 1;
    }
  }
  return 0;
}

} // namespace mom_nexapow

int main() {
  using namespace mom_nexapow;
  unsigned ordinal = 0;
  {
    const std::array<uint32_t, 16> input{};
    const U256 expected{};
    if (check_reduction(input, expected))
      return 1;
  }
  {
    std::array<uint32_t, 16> input{};
    U256 expected{};
    for (unsigned i = 0; i < 8; ++i)
      input[i] = kFieldPrime.w[i];
    if (check_reduction(input, expected))
      return 1;
  }
  {
    std::array<uint32_t, 16> input{};
    U256 expected{};
    input[8] = 1;
    expected.w[0] = 977u;
    expected.w[1] = 1u;
    if (check_reduction(input, expected))
      return 1;
  }
  // For B=2^32 and C=B+977, (2^512-1) mod p = C^2-1.
  {
    std::array<uint32_t, 16> input{};
    U256 expected{};
    input.fill(UINT32_MAX);
    expected.w[0] = 977u * 977u - 1u;
    expected.w[1] = 2u * 977u;
    expected.w[2] = 1u;
    if (check_reduction(input, expected))
      return 1;
  }
  // The first correction wraps to 2^64-1, so the final correction must carry into word 2.
  {
    std::array<uint32_t, 16> input{};
    U256 expected{};
    input.fill(UINT32_MAX);
    input[0] = static_cast<uint32_t>(0u - (977u * 977u - 977u + 1u));
    input[1] = static_cast<uint32_t>(0u - 2u * 977u);
    expected.w[0] = 976u;
    expected.w[1] = 1u;
    expected.w[2] = 1u;
    if (check_reduction(input, expected))
      return 1;
  }
  U256 window_scalar{{0x01234567u, 0x89abcdefu, 0xfedcba98u, 0x76543210u, 0x0f1e2d3cu, 0x4b5a6978u,
                      0x8796a5b4u, 0xc3d2e1f0u}};
  for (unsigned low = 0; low != 256; ++low) {
    for (unsigned width = 1; width <= 32 && low + width <= 256; ++width) {
      unsigned expected = 0;
      for (unsigned bit = 0; bit != width; ++bit)
        expected |= ((window_scalar.w[(low + bit) >> 5] >> ((low + bit) & 31u)) & 1u) << bit;
      if (np_window_value(window_scalar, low, width) != expected) {
        std::fprintf(stderr, "window extraction mismatch at %u/%u\n", low, width);
        return 1;
      }
    }
  }
  const uint32_t edge_words[][8] = {
      {0, 0, 0, 0, 0, 0, 0, 0},
      {1, 0, 0, 0, 0, 0, 0, 0},
      {2, 0, 0, 0, 0, 0, 0, 0},
      {3, 0, 0, 0, 0, 0, 0, 0},
      {15, 0, 0, 0, 0, 0, 0, 0},
      {16, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0x80000000u},
      {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
       0x7fffffffu},
      {0xd0364140u, 0xbfd25e8cu, 0xaf48a03bu, 0xbaaedce6u, 0xfffffffeu, 0xffffffffu, 0xffffffffu,
       0xffffffffu},
      {0xd0364141u, 0xbfd25e8cu, 0xaf48a03bu, 0xbaaedce6u, 0xfffffffeu, 0xffffffffu, 0xffffffffu,
       0xffffffffu},
  };
  for (const auto& words : edge_words)
    if (check_scalar(scalar_from_words(words), ordinal++))
      return 1;

  uint32_t zero[8]{}, one[8]{1}, negative_one[8];
  mom_nexa_field32_portable::mul_sub_mul(negative_one, zero, zero, one, one);
  const U256 expected_negative_one = np_field_sub(np_zero(), np_one());
  if (std::memcmp(negative_one, expected_negative_one.w, sizeof(negative_one))) {
    std::fputs("portable 8x32 fused subtraction underflow mismatch\n", stderr);
    return 1;
  }
  for (unsigned i = 1; i <= 64; ++i) {
    U256 scalar{};
    scalar.w[0] = i;
    if (check_scalar(scalar, ordinal++))
      return 1;
  }

  const U256 directed_zero = np_zero();
  const U256 directed_one = np_one();
  U256 directed_word_max{};
  directed_word_max.w[0] = UINT32_MAX;
  const U256 directed_field_minus_one = np_field_sub(directed_zero, directed_one);
  const U256 directed[] = {directed_zero, directed_one, directed_word_max,
                           directed_field_minus_one};
  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      if (check_field_arithmetic(directed[i], directed[j], directed[j], directed[i]))
        return 1;
    }
  }

  uint64_t random_state = 0x4f1bbcdc3a7e91d5ULL;
  for (unsigned sample = 0; sample < 1024; ++sample) {
    U256 value{}, other{}, third{}, fourth{};
    for (uint32_t& word : value.w)
      word = static_cast<uint32_t>(next_random(random_state));
    for (uint32_t& word : other.w)
      word = static_cast<uint32_t>(next_random(random_state));
    for (uint32_t& word : third.w)
      word = static_cast<uint32_t>(next_random(random_state));
    for (uint32_t& word : fourth.w)
      word = static_cast<uint32_t>(next_random(random_state));
    std::array<uint32_t, 16> reduction_input{};
    for (unsigned i = 0; i < 8; ++i) {
      reduction_input[i] = value.w[i];
      reduction_input[8u + i] = other.w[i];
    }
    const U256 reduction_expected = np_field_reduce(reduction_input.data());
    if (check_reduction(reduction_input, reduction_expected))
      return 1;
    value = np_reduce_mod(value, kFieldPrime);
    other = np_reduce_mod(other, kFieldPrime);
    third = np_reduce_mod(third, kFieldPrime);
    fourth = np_reduce_mod(fourth, kFieldPrime);
    if (check_field_arithmetic(value, other, third, fourth))
      return 1;
    if (np_quadratic_residue(value) != np_quadratic_residue_power(value)) {
      std::fprintf(stderr, "binary Jacobi mismatch for field value %u\n", sample);
      return 1;
    }
  }
  for (unsigned sample = 0; sample < 24; ++sample) {
    U256 scalar{};
    for (uint32_t& word : scalar.w)
      word = static_cast<uint32_t>(next_random(random_state));
    scalar = np_reduce_mod(scalar, kGroupOrder);
    if (np_zero_p(scalar))
      scalar = np_one();
    if (check_scalar(scalar, ordinal++))
      return 1;
  }

  const uint8_t messages[][32] = {
      {0},
      {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa,
       0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba,
       0xdc, 0xfe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef},
  };
  const uint8_t private_bytes[][32] = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
      {0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33,
       0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
       0xff, 0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80},
  };
  for (unsigned i = 0; i < 2; ++i) {
    const U256 private_key = np_from_be(private_bytes[i]);
    uint8_t optimized[64]{}, reference[64]{};
    if (np_sign(messages[i], private_key, optimized) !=
            sign_reference(messages[i], private_key, reference) ||
        std::memcmp(optimized, reference, sizeof(optimized)) != 0) {
      std::fprintf(stderr, "RFC6979 signature mismatch for vector %u\n", i);
      return 1;
    }
  }
  std::puts("nexapow fixed-base differential tests passed");
  return 0;
}
