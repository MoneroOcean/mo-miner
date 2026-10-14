// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "../../sycl/pearlhash/blake3.inc"

int main() {
  uint32_t state = 0x713acf09u;
  for (unsigned sample = 0; sample < 512; ++sample) {
    uint32_t words[16];
    uint8_t key[32], input[64], expected[32], actual[32];
    for (auto& word : words) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      word = sample == 0 ? 0u : sample == 1 ? UINT32_MAX : state;
    }
    for (unsigned i = 0; i < sizeof(input); ++i)
      input[i] = static_cast<uint8_t>(words[i / 4] >> (8 * (i % 4)));
    for (unsigned i = 0; i < sizeof(key); ++i)
      key[i] = static_cast<uint8_t>(state + i * 37u + sample);
    pearlhash_b3::b3(input, sizeof(input), key, expected);
    pearlhash_b3::b3_keyed_64_words(words, key, actual);
    if (std::memcmp(expected, actual, sizeof(actual)) != 0) {
      std::fprintf(stderr, "PearlHash fixed-block jackpot mismatch at sample %u\n", sample);
      return 1;
    }
  }
  std::puts("PearlHash fixed-block jackpot matches general keyed BLAKE3 (512 fixtures)");
}
