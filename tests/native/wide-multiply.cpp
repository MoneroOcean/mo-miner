// Differential test of the extracted production limb helpers, without a SYCL SDK/GPU.
#include <array>
#include <cassert>
#include <cstdint>
#include <random>

// Only the non-portable mom_mul_hi_u64 branch reaches this stand-in for the SYCL builtin.
namespace sycl {
inline uint64_t mul_hi(uint64_t a, uint64_t b) {
  return uint64_t((static_cast<unsigned __int128>(a) * b) >> 64);
}
} // namespace sycl
#include "wide-multiply.inc"

int main() {
  const auto check = [](uint64_t a, uint64_t b) {
    assert(mom_mul_wide_u32(uint32_t(a), uint32_t(b)) == uint64_t(uint32_t(a)) * uint32_t(b));
    const uint64_t high = uint64_t((static_cast<unsigned __int128>(a) * b) >> 64);
    assert(mul_hi64_portable(a, b) == high);
    assert(mom_mul_hi_u64(a, b) == high);
  };
  constexpr std::array<uint64_t, 12> edges{
    0, 1, 0xffff, 0x10000, 0xffffffff, 0x100000000,
    0xffffffff00000000, 0x8000000000000000, UINT64_MAX - 1, UINT64_MAX,
    33984, 0x55555555aaaaaaaa,
  };
  for (auto a : edges) for (auto b : edges) check(a, b);
  std::mt19937_64 random(0x58454c4953);
  for (unsigned i = 0; i < 100000; ++i) check(random(), random());
}
