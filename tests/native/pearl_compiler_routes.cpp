// Exercise actual extracted producer/capability code under each compiler's preprocessor profile.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "packet.inc"
static int8_t gv(uint32_t seed, uint32_t index, bool) {return static_cast<int8_t>((seed + index) % 64);}
template <bool PackedCuda>
static void produce(size_t index, int k, int rank, int8_t *ap, const int8_t *eal,
                    const int32_t *earp1, const int32_t *earp2, bool portable) {
  const std::array<size_t, 1> id{index}; const uint32_t seed = 7; const bool positive_inputs = true;
#include "producer.inc"
}
namespace mom::nvidia {
static int compute_capability(int) {return 120;}
static bool has_int8_async_matrix(int) {return true;}
}
struct State {
  struct Queue {int get_device() const {return 0;}} queue;
#include "tensor.inc"
};
int main(int argc, char **argv) {
  if (argc != 2) {return 2;}
  const bool packed = !std::strcmp(argv[1], "packed"), portable = std::strcmp(argv[1], "tiled") != 0;
  constexpr int m = 64, k = 64, rank = 128;
  std::array<int8_t, m * k> actual{}, expected{}; std::array<int8_t, m * rank> eal{};
  std::array<int32_t, k> first{}, second{};
  for (size_t i = 0; i < eal.size(); ++i) {eal[i] = static_cast<int8_t>(int(i % 7) - 3);}
  for (int i = 0; i < k; ++i) {first[i] = i; second[i] = i + 1;}
  for (size_t i = 0; i < actual.size(); ++i) {
    if (packed) {produce<true>(i, k, rank, actual.data(), eal.data(), first.data(), second.data(), portable);}
    else {produce<false>(i, k, rank, actual.data(), eal.data(), first.data(), second.data(), portable);}
    const size_t row = i / k, depth = i % k;
    size_t offset = i;
    if (packed) {offset = nvidia_vector128_offset<true>(row, depth, k);}
#if !defined(__NVPTX__) || (defined(MOM_SYCL_ADAPTIVECPP) && !defined(__CUDA_ARCH__))
    else if (!portable) {offset = ((row / 8) * (k / 32) + depth / 32) * 256 + (row % 8) * 32 + depth % 32;}
#endif
    expected[offset] = static_cast<int8_t>(gv(7, static_cast<uint32_t>(i), true) +
        eal[row * rank + first[depth]] - eal[row * rank + second[depth]]);
  }
  if (actual != expected) {std::fputs("A producer layout mismatch\n", stderr); return 1;}
  constexpr bool tensor = MOM_TEST_CUDA_TENSOR;
  if (State{}.cuda_tensor != tensor) {std::fputs("Uncompiled CUDA tensor route admitted\n", stderr); return 1;}
  std::puts("Producer layout and compiler route passed");
}
