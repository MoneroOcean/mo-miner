// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#include "../../native/cpu-scheduling.h"

#include <cassert>
#include <string>
#include <vector>

namespace {

constexpr unsigned KiB = 1024;
constexpr unsigned MiB = 1024 * KiB;

struct SchedulingCase {
  const char* algo;
  unsigned max_cpu_batch;
  unsigned socket_count;
  unsigned thread_count;
  unsigned l3cache;
  unsigned batch_mem;
  std::vector<unsigned> expected_threads;
  const char* expected_devices;
};

void test_scheduling_cases() {
  const SchedulingCase cases[] = {
    {"rx/0", 1, 2, 8, 8 * MiB, 2 * MiB, {2, 2}, "cpu*2^2"},
    {"rx/arq", 5, 1, 32, MiB, 2 * MiB, {1}, "cpu"},
    {"ghostrider", 5, 2, 2, 32 * MiB, 2 * MiB, {8, 8}, "cpu*8^2"},
    {"argon2/chukwa", 5, 1, 2, 8 * MiB, MiB, {1, 1}, "cpu^2"},
    {"cn/0", 5, 1, 3, 10 * MiB, 2 * MiB, {2, 2, 1}, "cpu*2^2,cpu"},
  };

  for (const auto& test : cases) {
    const auto threads = mom::cpu::cpu_thread_batches(
      test.algo, test.max_cpu_batch, test.socket_count, test.thread_count, test.l3cache,
      test.batch_mem);
    assert(std::vector<unsigned>(threads.begin(), threads.end()) == test.expected_threads);
    std::string devices;
    mom::cpu::append_grouped_cpu_devs(devices, threads);
    assert(devices == test.expected_devices);
  }
}

void test_result_separator() {
  std::string devices;
  mom::cpu::add_result_dev(devices, "cpu");
  mom::cpu::add_result_dev(devices, "cpu*2");
  assert(devices == "cpu,cpu*2");
}

} // namespace

int main() {
  test_scheduling_cases();
  test_result_separator();
  return 0;
}
