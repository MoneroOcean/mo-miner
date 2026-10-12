// Execute the actual cooperative Horner/FNV section against a scalar integer oracle.
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace sycl {
inline float fma(float a, float b, float c) { return std::fma(a, b, c); }
inline uint32_t mul_hi(uint32_t a, uint32_t b) {
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
}
}
#ifdef MOM_SYCL_PORTABLE_OPENCL
constexpr bool mom_sycl_portable_opencl = true;
#else
constexpr bool mom_sycl_portable_opencl = false;
#endif
#include "field.inc"

static void check(const uint32_t* coefficients, const uint32_t* points,
                  uint32_t lane, uint32_t offset) {
  constexpr uint32_t WARP = 32;
#include "polynomial.inc"
#ifdef MOM_SYCL_ADAPTIVECPP_CUDA
  static_assert(POLYNOMIAL_STEP == 8);
#else
  static_assert(POLYNOMIAL_STEP == 16);
#endif
  static_assert(POLYNOMIAL_BATCH == 16); // The separate batched kernel retains its original shape.
  uint64_t expected_hash = 0;
  for (uint32_t point = 0; point < POLYNOMIAL_LANES; ++point) {
    uint64_t expected = 0;
    for (uint32_t coefficient = POLYNOMIAL_SIZE; coefficient-- != 0;)
      expected = (expected * points[lane + point * WARP] + coefficients[offset + coefficient]) % MOD;
    assert(polynomial[point] == expected);
    expected_hash = expected_hash * 0x01000193ULL ^ expected;
  }
  assert(polynomial_hash == expected_hash);
}

int main() {
  std::array<uint32_t, POLYNOMIAL_SIZE * 4> coefficients{};
  std::array<uint32_t, POLYNOMIAL_LANES * 32> points{};
  uint32_t random = 0x7359a12c;
  const auto next = [&]() {
    random = random * 1664525U + 1013904223U;
    return random % MOD;
  };
  for (unsigned pattern = 0; pattern < 4; ++pattern) {
    for (auto& value : coefficients)
      value = pattern == 0 ? 0 : pattern == 1 ? 1 : pattern == 2 ? MOD - 1 : next();
    for (auto& point : points) point = next();
    // Include field endpoints in every lane as well as independent random points.
    for (unsigned lane = 0; lane < 32; ++lane) {
      points[lane] = 0;
      points[lane + 32] = 1;
      points[lane + 64] = MOD - 1;
    }
    for (unsigned warp = 0; warp < 4; ++warp)
      for (unsigned lane = 0; lane < 32; ++lane)
        check(coefficients.data(), points.data(), lane, warp * POLYNOMIAL_SIZE);
  }
  std::cout << "Octopus polynomial: 512 lanes / 16384 points and ordered FNV passed\n";
}
