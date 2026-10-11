#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static unsigned copies, waits, reports;
static std::size_t bytes;
static int report(FILE*, const char*, ...) { ++reports; return 0; }
static int flush(FILE*) { return 0; }
struct Queue {
  int memcpy(void* dst, const void* src, std::size_t count) {
    ++copies; bytes += count; std::memcpy(dst, src, count); return 0;
  }
};
static void sycl_wait_and_throw(int, int) { ++waits; }
// Mock geometry tests the exact diagnostic copy expression, not a production arena layout.
constexpr unsigned NSTORED_LEVELS = 2, NBUCKETS = 3;
struct State {
  std::vector<uint32_t> counters = std::vector<uint32_t>(NSTORED_LEVELS * NBUCKETS, 1);
  uint32_t* nslots = counters.data();
  uint32_t count = 1, *ref_count = &count;
  unsigned slot_capacity = 2;
  int device = 0;
  bool arena_log = false;
  void ensure_arenas(bool log) { arena_log = log; }
};
struct BeamState { static unsigned now_ms() { return 1; } };
static bool zel(bool is_test, bool is_benchmark, bool expected) {
  State state; Queue q;
  const unsigned cand_count = 1;
  const uint64_t t_rounds = 1, t_gen0 = 0;
#include "zel-setup.inc"
#include "zel-diagnostic.inc"
  return state.arena_log == expected && copies == unsigned(expected) &&
    waits == unsigned(expected) && bytes == (expected ? sizeof(uint32_t) * NSTORED_LEVELS * NBUCKETS : 0) &&
    reports == (expected ? NSTORED_LEVELS + 1 : 0);
}
static bool beam(bool is_test, bool is_benchmark, bool expected) {
  State state;
  const uint64_t t_start = 0;
  const int n_solutions = 1;
  const std::size_t distinct_count = 1;
#include "beam-setup.inc"
#include "beam-diagnostic.inc"
  return state.arena_log == expected && reports == unsigned(expected) && copies == 0 && waits == 0;
}
int main(int argc, char** argv) {
  if (argc != 4 || (std::strcmp(argv[3], "zel") != 0 && std::strcmp(argv[3], "beam") != 0)) return 2;
  const bool is_test = std::atoi(argv[1]) == 1, is_benchmark = std::atoi(argv[1]) == 2;
  const bool expected = is_test || std::atoi(argv[2]) != 0;
  if (std::strcmp(argv[3], "zel") == 0 && !zel(is_test, is_benchmark, expected)) {
    std::fprintf(stderr, "FAIL Zel diagnostic mode=%s prof=%s copies=%u waits=%u\n", argv[1], argv[2], copies, waits);
    return 17;
  }
  if (std::strcmp(argv[3], "beam") == 0 && !beam(is_test, is_benchmark, expected)) {
    std::fprintf(stderr, "FAIL Beam diagnostic mode=%s prof=%s reports=%u\n", argv[1], argv[2], reports);
    return 18;
  }
  std::puts("PASS actual-source solver diagnostic gates");
}
