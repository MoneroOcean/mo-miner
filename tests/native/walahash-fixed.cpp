// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include "../../sycl/nvidia-dot.h"
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

// Compile the production integer helpers without a SYCL runtime. Define the AdaptiveCpp
// feature after nvidia-dot.h so its host scalar fallback remains independently testable.
namespace sycl {
using std::bit_cast;
using std::fabs;
inline uint32_t rotate(uint32_t value, uint32_t count) {
  return std::rotl(value, static_cast<int>(count));
}
}
inline uint32_t mo_rotate(uint32_t value, uint32_t count) {
  return sycl::rotate(value, count);
}
#define MOM_SYCL_ADAPTIVECPP
#if defined(TEST_WALAHASH_DPCPP)
#define MOM_SYCL_HAS_CUDA
#else
#define MOM_SYCL_ADAPTIVECPP_CUDA
#endif
#include "../../sycl/walahash/device.inc"

static uint32_t random_state = 0x91a70u;
static uint32_t next() {
  random_state ^= random_state << 13;
  random_state ^= random_state >> 17;
  random_state ^= random_state << 5;
  return random_state;
}

template <bool Native>
static void mix(uint8_t out[32], const uint8_t pow[32], const uint8_t matrix[4096]) {
  uint32_t vec[16];
  for (unsigned k = 0; k < 16; ++k) {
    const unsigned j = 4 * k;
    vec[k] = static_cast<uint32_t>(pow[j >> 1] >> 4) |
             static_cast<uint32_t>(pow[j >> 1] & 15) << 8 |
             static_cast<uint32_t>(pow[(j + 2) >> 1] >> 4) << 16 |
             static_cast<uint32_t>(pow[(j + 2) >> 1] & 15) << 24;
  }
  for (unsigned i = 0; i < 64; ++i) {
    int32_t accumulator = 0;
    for (unsigned k = 0; k < 16; ++k)
      accumulator = dot4<Native>(load32_le_dev(matrix + i * 64 + k * 4), vec[k], accumulator);
    const uint8_t nibble = (accumulator & 15) ^ ((accumulator >> 4) & 15) ^
                           ((accumulator >> 8) & 15);
    if ((i & 1) == 0)
      out[i / 2] = pow[i / 2] ^ static_cast<uint8_t>(nibble << 4);
    else
      out[i / 2] ^= nibble;
  }
}

static void check(const uint8_t input[80], uint64_t nonce, const uint8_t matrix[4096]) {
  uint8_t header[80];
  std::memcpy(header, input, sizeof(header));
  for (unsigned i = 0; i < 8; ++i)
    header[72 + i] = static_cast<uint8_t>(nonce >> (8 * i));
  uint8_t generic[32], fixed[32];
  uint32_t words[8];
  Blake3Cv cv;
  b3_first_block_cv(cv, input);
  blake3_dev(generic, 32, header, sizeof(header));
  b3_80_from_cv(words, cv, input + 64, nonce);
  for (unsigned i = 0; i < 8; ++i)
    store32_le_dev(fixed + 4 * i, words[i]);
  assert(std::memcmp(generic, fixed, 32) == 0);
  sha3_256(generic, generic);
  sha3_words64(words);
  for (unsigned i = 0; i < 8; ++i)
    store32_le_dev(fixed + 4 * i, words[i]);
  assert(std::memcmp(generic, fixed, 32) == 0);
  uint8_t generic_mix[32], fixed_mix[32];
  mix<false>(generic_mix, generic, matrix);
  mix<true>(fixed_mix, fixed, matrix);
  assert(std::memcmp(generic_mix, fixed_mix, 32) == 0);
  blake3_dev(generic, 32, generic_mix, 32);
  for (unsigned i = 0; i < 8; ++i)
    words[i] = load32_le_dev(fixed_mix + 4 * i);
  b3_32(words, words);
  for (unsigned i = 0; i < 8; ++i)
    store32_le_dev(fixed + 4 * i, words[i]);
  assert(std::memcmp(generic, fixed, 32) == 0);
}

int main() {
  constexpr uint64_t nonces[] = {
    0, 1, 0xffffffffu, 0x100000000ULL, 0x8000000000000000ULL,
    std::numeric_limits<uint64_t>::max()
  };
  unsigned cases = 0;
  for (unsigned sample = 0; sample < 128; ++sample) {
    uint8_t input[80], matrix[4096];
    for (auto& byte : input)
      byte = sample < 2 ? static_cast<uint8_t>(sample * 255) : static_cast<uint8_t>(next());
    for (auto& byte : matrix)
      byte = sample < 2 ? static_cast<uint8_t>(sample * 15) : static_cast<uint8_t>(next() & 15);
    for (const uint64_t nonce : nonces) {
      check(input, nonce, matrix);
      ++cases;
    }
    check(input, (static_cast<uint64_t>(next()) << 32) | next(), matrix);
    ++cases;
  }
  std::printf("WalaHash fixed/generic CPU equivalence passed: %u cases, %u assertions\n",
              cases, cases * 4);
}
