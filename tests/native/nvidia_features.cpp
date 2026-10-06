// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include "../../sycl/nvidia-features.h"

#include <cassert>
#include <climits>

int main() {
  using namespace mom::nvidia;
  constexpr int devices[] = {0, 30, 35, 50, 52, 60, 61, 62, 70, 72, 75, 80, 86, 89, 90, 100, 120};
  for (const int sm : devices) {
    assert(has_dp4a(sm) == (sm >= 61));
    assert(has_int8_async_matrix(sm) == (sm >= 80));
    if (sm)
      assert(compute_capability(sm / 10, sm % 10) == sm);
  }
  // P100 has FP64 and V100 has Tensor Cores, but neither fact implies Ampere's int8/async ISA.
  assert(!has_dp4a(60));
  assert(has_dp4a(70));
  assert(!has_int8_async_matrix(70));
  assert(!has_int8_async_matrix(75));
  assert(has_int8_async_matrix(80));
  assert(has_int8_async_matrix(120));
  assert(!has_dp4a(-1));
  assert(!has_int8_async_matrix(-1));
  assert(compute_capability(0, 0) == 0);
  assert(compute_capability(-1, 0) == 0);
  assert(compute_capability(7, -1) == 0);
  assert(compute_capability(7, 10) == 0);
  assert(compute_capability(INT_MAX, 0) == 0);
  assert(compute_capability(INT_MAX / 10, 7) == INT_MAX);
  assert(compute_capability(INT_MAX / 10, 8) == 0);
  assert(compute_capability((INT_MAX - 9) / 10, 9) > 0);
}
