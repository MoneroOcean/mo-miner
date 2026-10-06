// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include "../../sycl/nvidia-dot.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>

static uint32_t oracle(uint32_t a, uint32_t b, uint32_t accumulator, bool is_signed) {
  int64_t sum = accumulator;
  for (unsigned lane = 0; lane < 4; ++lane) {
    int64_t x = (a >> (8 * lane)) & 255;
    int64_t y = (b >> (8 * lane)) & 255;
    if (is_signed) {
      if (x >= 128) x -= 256;
      if (y >= 128) y -= 256;
    }
    sum += x * y;
  }
  return static_cast<uint32_t>(sum);
}

static unsigned assertions = 0;

static void check(uint32_t a, uint32_t b, uint32_t accumulator) {
  const uint32_t signed_result = oracle(a, b, accumulator, true);
  const uint32_t unsigned_result = oracle(a, b, accumulator, false);
  assert(mom::nvidia::dot4_s8<false>(a, b, accumulator) == signed_result);
  assert(mom::nvidia::dot4_s8<true>(a, b, accumulator) == signed_result);
  assert(mom::nvidia::dot4_u8<false>(a, b, accumulator) == unsigned_result);
  assert(mom::nvidia::dot4_u8<true>(a, b, accumulator) == unsigned_result);
  assertions += 4;
}

int main() {
  constexpr uint32_t accumulators[] = {
    0, 1, 0x7fffffffu, 0x80000000u, 0xfffffffeu, std::numeric_limits<uint32_t>::max()
  };
  // Every byte pair in every lane covers sign extension and pack order independently.
  for (unsigned lane = 0; lane < 4; ++lane)
    for (uint32_t a = 0; a < 256; ++a)
      for (uint32_t b = 0; b < 256; ++b)
        for (const uint32_t accumulator : accumulators)
          check(a << (8 * lane), b << (8 * lane), accumulator);
  constexpr uint32_t words[] = {
    0, 0xffffffffu, 0x80808080u, 0x7f7f7f7fu, 0x807f01ffu, 0xff017f80u
  };
  for (const uint32_t a : words)
    for (const uint32_t b : words)
      for (const uint32_t accumulator : accumulators)
        check(a, b, accumulator);
  uint32_t state = 0x93610a7u;
  auto next = [&] {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };
  for (unsigned i = 0; i < 20000; ++i) {
    const uint32_t a = next(), b = next(), accumulator = next();
    check(a, b, accumulator);
  }
  assert(mom::nvidia::dot4_s8<false>(0xffffffffu, 0x01010101u, 0) == 0xfffffffcu);
  assert(mom::nvidia::dot4_u8<false>(0xffffffffu, 0x01010101u, 0) == 1020);
  assertions += 2;
  std::printf("NVIDIA packed-dot scalar tests passed: %u assertions\n", assertions);
}
