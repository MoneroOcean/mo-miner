#include <cassert>
#include <cstdint>
#include <cstdio>
#include "../../xmrig/3rdparty/libethash/data_sizes.h"

// Production ethash_dag_words_addressable from sycl/ethash-dag.inc, shared by Etchash and KawPow.
#include "ethash_dag_words.inc"

int main() {
  constexpr uint64_t words = uint64_t{1} << 32;
  assert(ethash_dag_words_addressable(words - 1));
  assert(ethash_dag_words_addressable(words));
  assert(!ethash_dag_words_addressable(words + 1));
  assert(ethash_dag_words_addressable(0));
  assert(!ethash_dag_words_addressable(UINT64_MAX));
  constexpr unsigned epochs = sizeof(dag_sizes) / sizeof(dag_sizes[0]);
  static_assert(epochs == 2048);
  static_assert(sizeof(cache_sizes) / sizeof(cache_sizes[0]) == epochs);
  unsigned first = epochs;
  for (unsigned epoch = 0; epoch < epochs; ++epoch) {
    assert(dag_sizes[epoch] % 128 == 0);
    const bool accepted = ethash_dag_words_addressable(dag_sizes[epoch] / sizeof(uint32_t));
    assert(accepted == (epoch < 1921));
    if (!accepted && first == epochs) first = epoch;
  }
  assert(first == 1921);
  assert(dag_sizes[first] == UINT64_C(17188256896));
  std::printf("Ethash DAG word limit: 5 boundaries + %u immutable epochs; "
              "first rejected=%u, DAG=%llu bytes\n", epochs, first,
              static_cast<unsigned long long>(dag_sizes[first]));
}
