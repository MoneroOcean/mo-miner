#include <cstdint>
#include <cstdio>
#include <stdexcept>

#define __device__
#define __forceinline__ inline

// PTX indexes the ternary truth table with (a,b,c) as its high-to-low input bits.
static uint32_t fixture_lop3(uint32_t a, uint32_t b, uint32_t c, unsigned truth) {
  uint32_t result = 0;
  for (unsigned bit = 0; bit < 32; ++bit) {
    const unsigned index = (((a >> bit) & 1u) << 2) |
                           (((b >> bit) & 1u) << 1) | ((c >> bit) & 1u);
    result |= ((truth >> index) & 1u) << bit;
  }
  return result;
}
#include "xor3.inc"

static uint32_t reduce(const uint32_t* values) {
  const auto accumulator = [values](unsigned index) { return values[index]; };
  constexpr unsigned lo = 0, hi = 4;
#include "reduction.inc"
  return part;
}

static uint32_t random_word(uint32_t& seed) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}

int main() {
  uint32_t seed = 0x19570123;
  for (unsigned trial = 0; trial < 10000; ++trial) {
    uint32_t values[8], expected = 0;
    for (auto& word : values) {
      word = random_word(seed);
      expected ^= word;
    }
    if (reduce(values) != expected)
      return std::fputs("FAIL CUDA checkpoint XOR differs from the canonical reduction\n", stderr), 1;
  }
  std::puts("PASS actual-source CUDA checkpoint XOR truth table and operand mapping");
}
