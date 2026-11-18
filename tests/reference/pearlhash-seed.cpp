// CPU-only PearlHash certificate V3 seed-derivation reference.
// Expected values are pinned by pearl-research-labs/pearl seed.rs at commit
// b713614301c42bfc3ea7c35e92917b7f4b161d8b.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#include "../../sycl/pearlhash/blake3.inc"
#include "../../sycl/pearlhash/claim.h"
#include "../../sycl/pearlhash/seed.inc"

namespace {

using Hash = std::array<std::uint8_t, 32>;

constexpr Hash repeated(const std::uint8_t value) {
  Hash result{};
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = value;
  return result;
}

constexpr Hash SALTED_B = {
  0x60, 0xed, 0x9b, 0x73, 0xc5, 0xa9, 0x59, 0x9b, 0x20, 0x0b, 0x6c, 0xd5, 0x63, 0xe7, 0xf0, 0xd5,
  0xd9, 0xa6, 0x7d, 0x24, 0x02, 0xd8, 0x5f, 0xd4, 0xef, 0x96, 0x6c, 0x58, 0x00, 0x80, 0xd0, 0xe5,
};
constexpr Hash SALTED_A = {
  0x30, 0x17, 0x84, 0x16, 0x80, 0x05, 0xec, 0x83, 0x3a, 0xb0, 0xaa, 0x60, 0x00, 0x6f, 0x7f, 0xe7,
  0xfa, 0xaa, 0x95, 0x30, 0x7d, 0x8c, 0x1f, 0xc6, 0x81, 0x9b, 0x2f, 0xfd, 0xd7, 0x17, 0xec, 0xcf,
};

void check(const bool condition) {
  if (!condition)
    std::abort();
}

void expect_hash(const std::uint8_t actual[32], const std::uint8_t expected[32]) {
  check(std::memcmp(actual, expected, 32) == 0);
}

void derive(const std::uint32_t m, const std::uint32_t n, std::uint8_t c_b[32],
            std::uint8_t c_a[32]) {
  const Hash key = repeated(0x11);
  const Hash raw_a = repeated(0xaa);
  const Hash raw_b = repeated(0xbb);
  mom_pearlhash::pearlhash_commitment_seeds(
      key.data(), raw_a.data(), raw_b.data(), m, n, c_b, c_a);
}

void test_salts() {
  constexpr char context_a[] = "pearl/cert-v3/noise-seed/A";
  constexpr char context_b[] = "pearl/cert-v3/noise-seed/B";
  std::uint8_t salt_a[32], salt_b[32];
  pearlhash_b3::b3(
      reinterpret_cast<const std::uint8_t*>(context_a), sizeof(context_a) - 1, nullptr, salt_a);
  pearlhash_b3::b3(
      reinterpret_cast<const std::uint8_t*>(context_b), sizeof(context_b) - 1, nullptr, salt_b);
  expect_hash(salt_a, mom_pearlhash::PEARLHASH_SEED_SALT_A);
  expect_hash(salt_b, mom_pearlhash::PEARLHASH_SEED_SALT_B);
}

void test_v3_vector() {
  std::uint8_t salted_b[32], salted_a[32];
  derive(192, 320, salted_b, salted_a);
  expect_hash(salted_b, SALTED_B.data());
  expect_hash(salted_a, SALTED_A.data());
}

void test_dimension_sensitivity() {
  std::uint8_t base_b[32], base_a[32];
  std::uint8_t changed_m_b[32], changed_m_a[32];
  std::uint8_t changed_n_b[32], changed_n_a[32];
  derive(192, 320, base_b, base_a);
  derive(193, 320, changed_m_b, changed_m_a);
  derive(192, 321, changed_n_b, changed_n_a);
  check(std::memcmp(base_b, changed_m_b, sizeof(base_b)) == 0);
  check(std::memcmp(base_a, changed_m_a, sizeof(base_a)) != 0);
  check(std::memcmp(base_b, changed_n_b, sizeof(base_b)) != 0);
  check(std::memcmp(base_a, changed_n_a, sizeof(base_a)) != 0);
}

} // namespace

int main() {
  uint32_t factor = 0;
  check(mom_pearlhash::pearlhash_adjustment_factor(16, 16, 4096, 256, &factor));
  check(factor == 524288);
  check(mom_pearlhash::pearlhash_adjustment_factor(16, 16, 8192, 128, &factor));
  check(factor == 2097152);
  check(mom_pearlhash::pearlhash_adjustment_factor(16, 16, 2048, 256, &factor));
  check(factor == 262144);
  check(mom_pearlhash::pearlhash_adjustment_factor(4, 8, 2048, 128, &factor));
  check(factor == 65536);
  check(!mom_pearlhash::pearlhash_adjustment_factor(0, 16, 4096, 256, &factor));
  check(!mom_pearlhash::pearlhash_adjustment_factor(UINT32_MAX, UINT32_MAX, 4096, 256, &factor));
  test_salts();
  test_v3_vector();
  test_dimension_sensitivity();
  return 0;
}
