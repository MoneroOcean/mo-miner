#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

// Only the integer SYCL intrinsics/types used by crypto.inc need a CPU counterpart here.
namespace sycl {
using uint4 = std::array<uint32_t, 4>;
inline uint64_t rotate(const uint64_t value, const uint64_t amount) {
  return std::rotl(value, static_cast<int>(amount & 63U));
}
inline uint64_t bitselect(const uint64_t a, const uint64_t b, const uint64_t c) {
  return (a & ~c) | (b & c);
}
} // namespace sycl

inline constexpr unsigned cn_gpu_lanes_per_hash = 16;
template <typename T> inline T mo_rotate(const T value, const T amount) {
  return std::rotl(value, static_cast<int>(amount & (sizeof(T) * 8 - 1)));
}
template <typename T> inline T mo_bitselect(const T a, const T b, const T c) {
  return (a & ~c) | (b & c);
}

#include "../../sycl/cn_gpu/crypto.inc"

int main() {
  // UINT_MAX-1 is the largest valid padded batch, so UINT_MAX-2 is its last hash index.
  constexpr unsigned last_hash = std::numeric_limits<unsigned>::max() - 2U;
  static_assert(cn_gpu_scratchpad_word_offset(8191) == 4294443008ULL);
  static_assert(cn_gpu_scratchpad_word_offset(8192) == 4294967296ULL);
  static_assert(cn_gpu_scratchpad_word_offset(8193) == 4295491584ULL);
  static_assert(cn_gpu_scratchpad_word_offset(last_hash) == 2251799812112384ULL);
  std::printf("CN/GPU scratchpad word offsets: 8191=%zu 8192=%zu 8193=%zu %u=%zu\n",
              cn_gpu_scratchpad_word_offset(8191), cn_gpu_scratchpad_word_offset(8192),
              cn_gpu_scratchpad_word_offset(8193), last_hash,
              cn_gpu_scratchpad_word_offset(last_hash));

  uint64_t seed = 0x9e3779b97f4a7c15ULL;
  for (unsigned sample = 0; sample < 64; ++sample) {
    std::array<uint64_t, 25> state{};
    if (sample) {
      for (auto& lane : state) {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        lane = seed * 0x2545f4914f6cdd1dULL;
      }
    }
    auto pairs = state, builtins = state, intrinsics = state, selected = state;
    keccak32(pairs.data());
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
    keccak64<false>(builtins.data());
    keccak64<true>(intrinsics.data());
#else
    keccak<true>(builtins.data());
    keccak<true>(intrinsics.data());
#endif
    keccak(selected.data());
    if (pairs != builtins || pairs != intrinsics || pairs != selected ||
        (!sample && pairs[0] != 0xf1258f7940e1dde7ULL)) {
      std::fprintf(stderr, "CN/GPU Keccak policy mismatch, sample=%u\n", sample);
      std::abort();
    }
  }
  std::puts("CN/GPU Keccak policies match on zero and 63 varied states");
}
