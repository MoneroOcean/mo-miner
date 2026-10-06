// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include "../../sycl/octopus/validation.h"

#include <cassert>

int main() {
  using mom::octopus::test_path_failure;
  using mom::octopus::TestPathFailure;
  // Older/unknown AMD and the generic CUDA worker still validate the ordinary full-DAG kernel.
  assert(test_path_failure(true, true, true, false, false) == TestPathFailure::none);
  assert(test_path_failure(true, true, false, true, false) == TestPathFailure::none);
  // Compilation, allocation or search failure on supported hardware must not pass by fallback.
  assert(test_path_failure(true, true, true, true, false) == TestPathFailure::native);
  assert(test_path_failure(true, true, true, true, true) == TestPathFailure::none);
  // Light-cache correctness is not evidence that a required full-DAG path executed.
  assert(test_path_failure(true, false, true, false, false) == TestPathFailure::full_dag);
  assert(test_path_failure(true, false, true, true, true) == TestPathFailure::full_dag);
  assert(test_path_failure(false, false, false, false, false) == TestPathFailure::none);
}
