// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#include "../../native/hashrate-sampling.h"

#include <cstdio>
#include <cstdlib>

namespace {

void expect(const char* const label, const bool condition) {
  if (!condition) {
    std::fprintf(stderr, "%s: condition failed\n", label);
    std::abort();
  }
}

void test_gpu_dispatches_are_sampling_points() {
  unsigned remaining = 10;
  for (unsigned dispatch = 0; dispatch < 3; ++dispatch) {
    expect("GPU dispatch is due", mom::hashrate_check_due(1, true, remaining, 10));
    expect("GPU dispatch resets counter", remaining == 10);
  }
}

void test_zero_hash_dispatch_is_inert() {
  unsigned remaining = 7;
  expect("zero CPU dispatch is not due", !mom::hashrate_check_due(0, false, remaining, 10));
  expect("zero CPU dispatch preserves counter", remaining == 7);

  remaining = 3;
  expect("zero GPU dispatch is not due", !mom::hashrate_check_due(0, true, remaining, 10));
  expect("zero GPU dispatch preserves counter", remaining == 3);
}

void test_cpu_dispatch_interval_is_unchanged() {
  unsigned remaining = 10;
  for (unsigned dispatch = 1; dispatch < 10; ++dispatch) {
    expect("CPU dispatch waits for interval",
           !mom::hashrate_check_due(1, false, remaining, 10));
    expect("CPU dispatch decrements counter", remaining == 10 - dispatch);
  }
  expect("tenth CPU dispatch is due", mom::hashrate_check_due(1, false, remaining, 10));
  expect("tenth CPU dispatch resets counter", remaining == 10);

  for (unsigned dispatch = 1; dispatch < 10; ++dispatch)
    expect("second CPU interval waits", !mom::hashrate_check_due(1, false, remaining, 10));
  expect("second CPU interval is due", mom::hashrate_check_due(1, false, remaining, 10));
  expect("second CPU interval resets counter", remaining == 10);
}

void test_cpu_gpu_transitions_reset_sampling_state() {
  unsigned remaining = 10;
  for (unsigned dispatch = 0; dispatch < 3; ++dispatch)
    expect("CPU pre-transition waits", !mom::hashrate_check_due(1, false, remaining, 10));
  expect("CPU pre-transition decrements", remaining == 7);

  expect("GPU transition is due", mom::hashrate_check_due(1, true, remaining, 10));
  expect("GPU transition resets counter", remaining == 10);
  expect("zero dispatch after GPU is inert", !mom::hashrate_check_due(0, false, remaining, 10));
  expect("zero dispatch after GPU preserves counter", remaining == 10);

  for (unsigned dispatch = 1; dispatch < 10; ++dispatch)
    expect("CPU post-transition waits", !mom::hashrate_check_due(1, false, remaining, 10));
  expect("CPU post-transition is due", mom::hashrate_check_due(1, false, remaining, 10));
  expect("CPU post-transition resets counter", remaining == 10);
}

} // namespace

int main() {
  test_gpu_dispatches_are_sampling_points();
  test_zero_hash_dispatch_is_inert();
  test_cpu_dispatch_interval_is_unchanged();
  test_cpu_gpu_transitions_reset_sampling_state();
  std::puts("Hashrate sampling host test passed");
  return 0;
}
