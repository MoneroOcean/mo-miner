#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include "../../xmrig/3rdparty/libethash/data_sizes.h"

static bool admits(uint64_t new_dag_words) {
  try {
#include "etchash_dag_limit.inc"
    return true;
  } catch (const std::string& error) {
    assert(error == "Etchash DAG exceeds 32-bit word addressing");
    return false;
  }
}

int main() {
  constexpr uint64_t words = uint64_t{1} << 32;
  assert(admits(words - 1));
  assert(admits(words));
  assert(!admits(words + 1));
  assert(admits(0));
  assert(!admits(UINT64_MAX));
  constexpr unsigned epochs = sizeof(dag_sizes) / sizeof(dag_sizes[0]);
  static_assert(epochs == 2048);
  static_assert(sizeof(cache_sizes) / sizeof(cache_sizes[0]) == epochs);
  unsigned first = epochs;
  for (unsigned epoch = 0; epoch < epochs; ++epoch) {
    assert(dag_sizes[epoch] % 128 == 0);
    const bool accepted = admits(dag_sizes[epoch] / sizeof(uint32_t));
    assert(accepted == (epoch < 1921));
    if (!accepted && first == epochs) first = epoch;
  }
  assert(first == 1921);
  assert(dag_sizes[first] == UINT64_C(17188256896));
  std::printf("Etchash word limit: 5 boundaries + %u immutable epochs; "
              "first rejected=%u, DAG=%llu bytes\n", epochs, first,
              static_cast<unsigned long long>(dag_sizes[first]));
}
