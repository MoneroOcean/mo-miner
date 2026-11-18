// Differential test of the extracted production limb helpers, without a SYCL SDK/GPU.
#include <array>
#include <cassert>
#include <cstdint>
#include <random>
#include "wide-multiply.inc"

int main() {
  const auto check = [](uint64_t a, uint64_t b) {
    assert(mo_mul_wide_u32(uint32_t(a), uint32_t(b)) == uint64_t(uint32_t(a)) * uint32_t(b));
    assert(mul_hi64_portable(a, b) == uint64_t((static_cast<unsigned __int128>(a) * b) >> 64));
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
