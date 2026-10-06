// CPU-only scalar reference for the bounded Verthash test fixture.
//
// This intentionally does not include the SYCL implementation: it derives the expected result
// from the same header/seed/search operation order with a compact deterministic dataset. The
// production implementation must still exercise its SeedKernel and SearchKernel against this
// result.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct Uint2 {
  std::uint32_t x, y;
};

constexpr std::uint32_t TEST_MDIV = 4093;
constexpr std::size_t TEST_DATA_ELEMENTS = static_cast<std::size_t>(TEST_MDIV) * 2 + 2;

std::uint32_t test_data_word(const std::uint64_t index, const std::uint64_t seed) {
  std::uint64_t value = index + seed;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return static_cast<std::uint32_t>(value ^ (value >> 32));
}

Uint2 test_data_item(const std::size_t index) {
  return {test_data_word(index, 0x243f6a8885a308d3ULL),
          test_data_word(index, 0x13198a2e03707344ULL)};
}

std::uint64_t rotl64(const std::uint64_t value, const unsigned shift) {
  return (value << shift) | (value >> (64 - shift));
}

void keccakf1600(std::uint64_t state[25]) {
  static constexpr std::uint64_t rc[24] = {
      0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
      0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
      0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
      0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
      0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
      0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
  static constexpr unsigned rotations[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                                             27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};
  static constexpr unsigned positions[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                                             15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};
  for (unsigned round = 0; round < 24; ++round) {
    std::uint64_t column[5];
    for (unsigned i = 0; i < 5; ++i)
      column[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
    for (unsigned i = 0; i < 5; ++i) {
      const std::uint64_t value = column[(i + 4) % 5] ^ rotl64(column[(i + 1) % 5], 1);
      for (unsigned j = 0; j < 25; j += 5)
        state[j + i] ^= value;
    }
    std::uint64_t value = state[1];
    for (unsigned i = 0; i < 24; ++i) {
      const unsigned position = positions[i];
      column[0] = state[position];
      state[position] = rotl64(value, rotations[i]);
      value = column[0];
    }
    for (unsigned j = 0; j < 25; j += 5) {
      for (unsigned i = 0; i < 5; ++i)
        column[i] = state[j + i];
      for (unsigned i = 0; i < 5; ++i)
        state[j + i] ^= (~column[(i + 1) % 5]) & column[(i + 2) % 5];
    }
    state[0] ^= rc[round];
  }
}

std::uint32_t fnv1a(const std::uint32_t a, const std::uint32_t b) {
  return (a ^ b) * 0x01000193u;
}

std::uint32_t rotate32(const std::uint32_t value, const unsigned shift) {
  return (value << shift) | (value >> (32 - shift));
}

void make_state(const std::uint32_t header[20], const unsigned first, std::uint64_t state[25]) {
  for (unsigned i = 0; i < 25; ++i)
    state[i] = 0;
  const std::uint32_t word0 = (header[0] & 0xffffff00u) | ((header[0] + first) & 0xffu);
  state[0] = static_cast<std::uint64_t>(word0) | (static_cast<std::uint64_t>(header[1]) << 32);
  for (unsigned i = 1; i < 9; ++i)
    state[i] = static_cast<std::uint64_t>(header[i * 2]) |
               (static_cast<std::uint64_t>(header[i * 2 + 1]) << 32);
  keccakf1600(state);
}

std::array<std::uint8_t, 32> hash_one(const std::array<std::uint8_t, 80>& bytes,
                                      const std::uint32_t nonce) {
  std::uint32_t header[20];
  for (unsigned i = 0; i < 20; ++i)
    header[i] = static_cast<std::uint32_t>(bytes[i * 4]) |
                (static_cast<std::uint32_t>(bytes[i * 4 + 1]) << 8) |
                (static_cast<std::uint32_t>(bytes[i * 4 + 2]) << 16) |
                (static_cast<std::uint32_t>(bytes[i * 4 + 3]) << 24);

  std::uint64_t states[8][25];
  for (unsigned i = 0; i < 8; ++i)
    make_state(header, i + 1, states[i]);

  std::uint64_t seed_state[25]{};
  for (unsigned i = 0; i < 9; ++i)
    seed_state[i] = static_cast<std::uint64_t>(header[i * 2]) |
                    (static_cast<std::uint64_t>(header[i * 2 + 1]) << 32);
  seed_state[9] = static_cast<std::uint64_t>(header[18]) | (static_cast<std::uint64_t>(nonce) << 32);
  seed_state[10] ^= 0x06u;
  seed_state[16] ^= 0x8000000000000000ULL;
  keccakf1600(seed_state);

  Uint2 values[4];
  for (unsigned lane = 0; lane < 4; ++lane)
    values[lane] = {static_cast<std::uint32_t>(seed_state[lane]),
                    static_cast<std::uint32_t>(seed_state[lane] >> 32)};

  std::uint32_t combined[128]{};
  for (unsigned lane = 0; lane < 4; ++lane) {
    for (unsigned pass = 0; pass < 2; ++pass) {
      std::uint64_t state[25];
      std::memcpy(state, states[lane * 2 + pass], sizeof(state));
      state[0] ^= static_cast<std::uint64_t>(header[18]) | (static_cast<std::uint64_t>(nonce) << 32);
      state[1] ^= 0x06u;
      state[8] ^= 0x8000000000000000ULL;
      keccakf1600(state);
      for (unsigned i = 0; i < 8; ++i) {
        const unsigned at = (lane * 16 + pass * 8 + i) * 2;
        combined[at] = static_cast<std::uint32_t>(state[i]);
        combined[at + 1] = static_cast<std::uint32_t>(state[i] >> 32);
      }
    }
  }

  std::uint32_t accumulator[4] = {0x811c9dc5u, 0x811c9dc5u, 0x811c9dc5u, 0x811c9dc5u};
  for (unsigned i = 0; i < 4096; ++i) {
    Uint2 items[4];
    for (unsigned lane = 0; lane < 4; ++lane) {
      const std::uint32_t word = combined[i & 127u];
      const unsigned shift = i >> 7;
      const std::uint32_t seek = shift ? rotate32(word, shift) : word;
      const std::size_t index = (fnv1a(seek, accumulator[lane]) % TEST_MDIV) * 2 + lane;
      items[lane] = test_data_item(index);
      values[lane].x = fnv1a(values[lane].x, items[lane].x);
      values[lane].y = fnv1a(values[lane].y, items[lane].y);
    }
    for (unsigned lane = 0; lane < 4; ++lane)
      for (unsigned other = 0; other < 4; ++other) {
        accumulator[lane] = fnv1a(accumulator[lane], items[other].x);
        accumulator[lane] = fnv1a(accumulator[lane], items[other].y);
      }
  }

  std::array<std::uint8_t, 32> result{};
  for (unsigned lane = 0; lane < 4; ++lane) {
    const std::uint32_t words[2] = {values[lane].x, values[lane].y};
    for (unsigned word = 0; word < 2; ++word)
      for (unsigned byte = 0; byte < 4; ++byte)
        result[lane * 8 + word * 4 + byte] = static_cast<std::uint8_t>(words[word] >> (byte * 8));
  }
  return result;
}

void check(const bool condition) {
  if (!condition)
    std::abort();
}

} // namespace

int main() {
  std::array<std::uint8_t, 80> header{};
  for (unsigned i = 0; i < 76; ++i)
    header[i] = static_cast<std::uint8_t>(i);
  const auto result = hash_one(header, 0);
  constexpr std::array<std::uint8_t, 32> expected = {
      0x24, 0xe6, 0xb8, 0x92, 0xaa, 0xf2, 0x51, 0x1e, 0x23, 0x2a, 0x75,
      0x52, 0xc2, 0x52, 0x8a, 0xf1, 0xb3, 0xfb, 0x7c, 0xdc, 0xae, 0x8a,
      0x5a, 0x55, 0x2a, 0x8b, 0x26, 0xc2, 0x40, 0xfe, 0x20, 0xd6};
  check(result == expected);
  std::puts("verthash-small-reference:passed");
}
