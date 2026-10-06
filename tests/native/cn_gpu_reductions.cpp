#include "../../sycl/cn_gpu/reductions.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace {

struct State {
  std::array<int32_t, 64> words;
  std::array<float, 64> weights;
};

using Order = std::array<unsigned, 16>;

void require(const bool condition, const char* const label) {
  if (!condition) {
    std::fprintf(stderr, "CN/GPU reduction regression: %s\n", label);
    std::abort();
  }
}

State reference(const State& input) {
  State first = input;
  for (unsigned lane = 0; lane < 16; ++lane) {
    const unsigned b = 16 * (lane / 4) + lane % 4;
    first.words[lane] = input.words[b] ^ input.words[b + 4] ^
                        input.words[b + 8] ^ input.words[b + 12];
    first.weights[lane] = (input.weights[b] + input.weights[b + 4]) +
                          (input.weights[b + 8] + input.weights[b + 12]);
  }
  State second = first;
  for (unsigned lane = 0; lane < 16; ++lane) {
    const unsigned b = 16 * (lane / 4) + lane % 4;
    const float xf = std::fabs((first.weights[b] + first.weights[b + 4]) +
                               (first.weights[b + 8] + first.weights[b + 12]));
    second.words[lane] = first.words[lane] ^ first.words[lane + 4] ^
                         first.words[lane + 8] ^ first.words[lane + 12] ^
                         static_cast<int32_t>(xf * 16777216.0f);
    second.weights[lane] = xf * 0.015625f;
  }
  return second;
}

State staged(State state, const Order& first_order, const Order& second_order) {
  std::array<int32_t, 64> seed;
  seed.fill(std::bit_cast<int32_t>(0xa5a5a5a5U));
  for (const unsigned lane : first_order) {
    seed[lane] = cn_gpu_reduce_int(state.words.data(), lane);
    seed[16 + lane] = std::bit_cast<int32_t>(cn_gpu_reduce_float(state.weights.data(), lane));
  }
  for (const unsigned lane : second_order) {
    const float xf = std::fabs(cn_gpu_reduce_staged_float(state.weights.data(), seed.data(), lane));
    state.words[lane] = cn_gpu_reduce_staged_int(state.words.data(), seed.data(), lane) ^
                        static_cast<int32_t>(xf * 16777216.0f);
    state.weights[lane] = xf * 0.015625f;
  }
  for (unsigned index = 32; index < seed.size(); ++index)
    require(seed[index] == std::bit_cast<int32_t>(0xa5a5a5a5U), "staging stays in available seed space");
  return state;
}

State old_in_place(State state, const Order& order) {
  for (const unsigned lane : order) {
    state.words[lane] = cn_gpu_reduce_int(state.words.data(), lane);
    state.weights[lane] = cn_gpu_reduce_float(state.weights.data(), lane);
  }
  for (const unsigned lane : order) {
    const float xf = std::fabs(cn_gpu_reduce_float(state.weights.data(), lane));
    state.words[lane] ^= state.words[lane + 4] ^ state.words[lane + 8] ^ state.words[lane + 12] ^
                         static_cast<int32_t>(xf * 16777216.0f);
    state.weights[lane] = xf * 0.015625f;
  }
  return state;
}

bool equal(const State& a, const State& b) {
  if (a.words != b.words)
    return false;
  for (unsigned index = 0; index < a.weights.size(); ++index)
    if (std::bit_cast<uint32_t>(a.weights[index]) != std::bit_cast<uint32_t>(b.weights[index]))
      return false;
  return true;
}

} // namespace

int main() {
  std::array<Order, 5> orders;
  for (unsigned lane = 0; lane < 16; ++lane) {
    orders[0][lane] = lane;
    orders[1][lane] = 15 - lane;
    orders[2][lane] = (lane + 8) % 16; // Second subgroup finishes before the first starts.
    orders[3][lane] = (lane % 2) * 8 + lane / 2;
    orders[4][lane] = (lane * 5) % 16;
  }
  std::mt19937 random(0x434e4750U);
  constexpr unsigned cases = 1000;
  bool reproduced_old_overwrite = false;
  for (unsigned n = 0; n < cases; ++n) {
    State input;
    for (unsigned index = 0; index < input.words.size(); ++index) {
      input.words[index] = std::bit_cast<int32_t>(static_cast<uint32_t>(random()));
      // Match the recurrence's signed [2,4) bit shaping. Sums exercise rounding while xf's
      // scaled float-to-int conversion stays inside the signed 32-bit range.
      const uint32_t bits = (static_cast<uint32_t>(random()) & 0x807fffffU) | 0x40000000U;
      input.weights[index] = std::bit_cast<float>(bits);
    }
    const State expected = reference(input);
    reproduced_old_overwrite |= !equal(old_in_place(input, orders[2]), expected);
    for (const Order& first : orders)
      for (const Order& second : orders)
        require(equal(staged(input, first, second), expected), "schedule-independent snapshot result");
  }
  require(reproduced_old_overwrite, "fixture exposes the old in-place reduction overwrite");
  std::printf("CN/GPU reduction host test passed: %u inputs, 25 execution orders each\n", cases);
}
