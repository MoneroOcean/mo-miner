// The split search's register-only index hash must equal the byte-wise reference for every
// message, nonce and table row. Compiles the two device routines for the host.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Host stand-ins for the worker definitions the two includes rely on.
constexpr bool mom_sycl_portable_opencl = false;
constexpr unsigned HASH_LEN = 32;
__extension__ typedef unsigned __int128 mom_u128;
inline uint64_t mom_mul_hi_u64(const uint64_t a, const uint64_t b) {
  return static_cast<uint64_t>((static_cast<mom_u128>(a) * b) >> 64);
}

#include "blake2b.inc"
#include "prehash.inc"

int main() {
  uint64_t seed = 0x9E3779B97F4A7C15ULL;
  const auto next = [&] { // splitmix64
    uint64_t z = (seed += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  };
  // Small, power-of-two and odd row counts: the row index is a 64-bit remainder by this count.
  for (const uint32_t n_len : {1U, 2U, 255U, 256U, 65537U, 1U << 20, 104729U}) {
    std::vector<uint32_t> table(static_cast<size_t>(n_len) * 8U);
    for (uint32_t& word : table)
      word = static_cast<uint32_t>(next());
    const uint64_t divisor = divM64(n_len);
    for (unsigned round = 0; round < 2000; ++round) {
      uint8_t message[32];
      for (uint8_t& byte : message)
        byte = static_cast<uint8_t>(next());
      // Nonces with every byte position exercised, including the all-ones and zero edges.
      const uint64_t nonce = round == 0 ? 0 : round == 1 ? UINT64_MAX : next();
      uint8_t reference[32];
      autolykos_index_hash_dev(message, nonce, table.data(), 0, n_len, divisor, reference);
      uint64_t message_words[4];
      std::memcpy(message_words, message, sizeof(message_words));
      uint32_t words[8];
      autolykos_index_words_dev(message_words, nonce, table.data(), n_len, divisor, words);
      for (unsigned i = 0; i < 8; ++i) {
        if (words[i] != load32_be_dev(reference + i * 4)) {
          std::fprintf(stderr, "n_len=%u round=%u word %u differs\n", n_len, round, i);
          return 1;
        }
      }
    }
  }
  return 0;
}
