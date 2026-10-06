#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define MOM_SYCL_KERNEL_ARGS_RESTRICT
struct Trace {
  int setup = 0, large = 0, small = 0, submits = 0, copies = 0, waits = 0;
  int searches = 0, monolithic = 0, query = 0, readbacks = 0;
  int pending = 0, cleanup_waits = 0, free_before_drain = 0, free_faults = 0;
} counts;
static std::string fault;
static int error_kind = 0, allocation_step = 0;
static bool first_only = false, numerical_mismatch = false;
static bool found_candidate = true, invalid_candidate = false;
static bool cleanup_wait_error = false;
static bool retire_then_throw = false;
static void* failed_retirement = nullptr;
static size_t expected_uncertain_allocations = 0;
static std::exception_ptr injected;
static std::unordered_set<void*> allocations;
static std::unordered_map<void*, std::function<void()>> mock_owners;
static std::unordered_map<void*, size_t> mock_sizes;
static std::unordered_map<void*, unsigned> release_attempts;
static uint64_t device_memory = uint64_t{16} << 30;
static void* readback_destination = nullptr;

static void require(bool value, const char* reason) {
  if (!value)
    throw std::logic_error(reason);
}
namespace sycl {
struct exception : std::runtime_error {
  using std::runtime_error::runtime_error;
};
}
static void inject(const char* stage) {
  if (fault != stage)
    return;
  try {
    if (error_kind == 0)
      throw sycl::exception("submitted fixture fault");
    if (error_kind == 1)
      throw std::string("submitted fixture fault");
    if (error_kind == 2)
      throw std::runtime_error("submitted fixture fault");
    throw 37;
  } catch (...) {
    injected = std::current_exception();
    throw;
  }
}
struct U256 { uint32_t w[8]{}; };
struct Point { U256 x{}, y{}; };
static void complete_readback() {
  if (!readback_destination)
    return;
  std::memset(readback_destination, 0, sizeof(Point));
  if (numerical_mismatch && (!first_only || counts.setup == 1))
    static_cast<Point*>(readback_destination)->x.w[0] = 1;
  readback_destination = nullptr;
}
namespace sycl_pipeline {
constexpr uint32_t kWorkGroup = 128, kCompactPoints = 4;
struct TableLayout { uint32_t wide_bits, wide_rows, narrow_bits, total_rows; };
constexpr TableLayout kSmallTable{22, 4, 21, 12}, kLargeTable{24, 3, 23, 11};
// Only geometry/math is mocked. No virtual GiB allocation or device operation occurs.
constexpr uint32_t table_points(TableLayout v) { return v.wide_bits == 24 ? 3 : 2; }
constexpr uint32_t table_entries(uint32_t) { return 1; }
constexpr uint32_t table_bits(TableLayout, uint32_t) { return 1; }
constexpr uint32_t table_start(TableLayout, uint32_t) { return 0; }
constexpr uint32_t table_offset(TableLayout, uint32_t) { return 0; }
using Field = U256;
using AffineTablePoint = Point;
using CurvePoint = Point;
struct PointPlanes { Field *x = nullptr, *y = nullptr, *z = nullptr; };
struct SearchResult {
  uint32_t count = 0, pad = 0;
  uint64_t nonce = 0;
};
inline CurvePoint infinity() { return {}; }
inline CurvePoint add_affine(CurvePoint a, U256, U256) { return a; }
inline Point affine(Point a) { return a; }
inline Point multiply(U256, const Point*, TableLayout) { return {}; }
class ShiftedTableKernel;
class LargeTableKernel;
class TableCheckKernel;
}
static Point kGeneratorWindow8[sycl_pipeline::kCompactPoints]{};
inline Point np_affine(Point p) { return p; }
inline Point np_scalar_mul(U256) { return {}; }
inline int np_cmp(U256 a, U256 b) { return std::memcmp(&a, &b, sizeof(a)); }
class SearchKernel;
namespace sycl {
namespace info { namespace device { struct global_mem_size {}; } }
struct device {
  template <typename T> uint64_t get_info() const {
    ++counts.query;
    inject("memory-query");
    return device_memory;
  }
};
template <int N> struct range {
  size_t size;
  range(size_t n) : size(n) {}
};
template <int N> struct id {
  size_t value;
  size_t operator[](int) const { return value; }
};
struct handler {
  template <typename K, typename F> void parallel_for(range<1>, F f) {
    if constexpr (std::is_same_v<K, SearchKernel>)
      f(id<1>{0});
  }
  template <typename K, typename F> void single_task(F) {}
};
struct event { bool copy; };
struct queue {
  device get_device() const { return {}; }
  event memcpy(void* dest, const void* source, size_t bytes) {
    ++counts.copies;
    if (bytes == sizeof(Point)) {
      ++counts.readbacks;
      inject("readback");
      require(!readback_destination, "previous host readback was not completed");
      readback_destination = dest;
    } else {
      inject("copy");
      std::memcpy(dest, source, bytes);
    }
    ++counts.pending;
    return {true};
  }
  event memset(void* dest, int value, size_t bytes) {
    std::memset(dest, value, bytes);
    return {true};
  }
  template <typename F> event submit(F f) {
    ++counts.submits;
    inject("dispatch");
    if (counts.submits == 2)
      inject("table-row-dispatch");
    if (counts.submits == 13)
      inject("selfcheck-dispatch");
    handler h;
    f(h);
    ++counts.pending;
    return {false};
  }
  void wait_and_throw() {
    ++counts.cleanup_waits;
    complete_readback();
    counts.pending = 0;
    if (cleanup_wait_error)
      throw std::runtime_error("cleanup wait fixture error");
  }
};
template <typename T> T* malloc_device(size_t count, queue&) {
  int step = count == sycl_pipeline::kCompactPoints ? 1 :
      count == 7 * sycl_pipeline::kCompactPoints ? 2 : 3;
  if (step == 1)
    ++counts.setup;
  if (step == 3)
    count == 3 ? ++counts.large : ++counts.small;
  if (fault == "allocation" && step == allocation_step &&
      (!first_only || counts.setup == 1)) {
    if (error_kind == -1)
      return nullptr;
    inject("allocation");
  }
  auto* result = new T[count]{};
  allocations.insert(result);
  mock_owners[result] = [=] { delete[] result; };
  mock_sizes[result] = count;
  return result;
}
template <typename T> void free(T* value, queue&) {
  ++release_attempts[value];
  if (counts.pending)
    ++counts.free_before_drain;
  const size_t size = mock_sizes.at(value);
  const char* cleanup_stage = size == sycl_pipeline::kCompactPoints ? "free-compact" :
      size == 7 * sycl_pipeline::kCompactPoints ? "free-shifted" : "free-table";
  if (!counts.free_faults && fault == cleanup_stage) {
    ++counts.free_faults;
    failed_retirement = value;
    if (retire_then_throw) {
      require(allocations.erase(value) == 1, "retirement error released a foreign pointer");
      mock_owners.erase(value);
      mock_sizes.erase(value);
      delete[] value;
    }
    inject(cleanup_stage);
  }
  require(allocations.erase(value) == 1, "allocation released twice or foreign allocation");
  mock_owners.erase(value);
  mock_sizes.erase(value);
  delete[] value;
}
enum class memory_order { relaxed };
enum class memory_scope { device };
namespace access { enum class address_space { global_space }; }
template <typename T, memory_order, memory_scope, access::address_space> struct atomic_ref {
  T& value;
  explicit atomic_ref(T& v) : value(v) {}
  T fetch_add(T n) {
    const T old = value;
    value += n;
    return old;
  }
};
}
static void sycl_wait_and_throw(sycl::event event, const sycl::device&) {
  if (!counts.monolithic)
    ++counts.waits;
  inject(event.copy ? "readback-wait" : "kernel-wait");
  complete_readback();
  counts.pending = 0;
}
template <typename F> void sycl_cleanup_noexcept(const char*, F f) noexcept {
  try {
    f();
  } catch (...) {
  }
}

#include "search-prefix.inc"
  bool search(sycl::queue&, const uint8_t[32], const uint8_t[8], uint64_t first,
              const uint8_t[32], uint32_t count, unsigned, uint64_t& found) {
    ++counts.searches;
    inject("search");
    found = first + (invalid_candidate ? count : 1);
    return found_candidate;
  }
};

#include "caller-support.inc"
struct State {
  sycl::device device;
  sycl::queue queue;
  NexaPowSyclSearch portable;
  bool staged_logged = false, staged_failure_logged = false;
  std::mutex mutex;
  uint8_t header_storage[32]{}, extra_storage[8]{}, target_storage[32]{};
  Result result_storage{};
  uint8_t *header = header_storage, *extranonce = extra_storage, *target = target_storage;
  Result* result = &result_storage;
  void init_portable() { ++counts.monolithic; }
  ~State() { portable.release(queue); }
};
static std::unique_ptr<State> state;
static State& state_for(const std::string&) { return *state; }
static bool env_enabled(const char*) { return false; }
static const char* staged_field_name(const sycl::device&) { return "host-fixture"; }
static void test_recorded_vector(State&, const uint8_t*, uint8_t*, uint64_t*) {
  throw std::logic_error("unexpected recorded-vector path");
}
static constexpr uint8_t kRecordedHeader[32]{}, kRecordedExtranonce[8]{};
static constexpr uint8_t kRecordedMinerNonceBytes[8]{}, kRecordedHash[32]{};
static constexpr uint64_t kRecordedMinerNonce = 0;
static bool np_hash_one(const uint8_t*, const uint8_t*, uint64_t, uint8_t* output, unsigned) {
  std::memset(output, 0x11, 32);
  return true;
}
namespace mom { namespace job_boundary {
inline bool padded_u32_range_fits(unsigned n, unsigned g) { return n <= UINT32_MAX - g + 1; }
} }
#include "caller.inc"

static void reset() {
  state.reset();
  require(allocations.empty(), "previous case leaked allocations");
  require(!readback_destination, "previous case left a host readback pending");
  counts = {};
  fault.clear();
  injected = {};
  error_kind = allocation_step = 0;
  first_only = numerical_mismatch = invalid_candidate = false;
  cleanup_wait_error = false;
  retire_then_throw = false;
  failed_retirement = nullptr;
  expected_uncertain_allocations = 0;
  release_attempts.clear();
  found_candidate = true;
  device_memory = uint64_t{16} << 30;
  state = std::make_unique<State>();
}
struct Attempt {
  uint8_t input[48]{}, output[32], target[32];
  uint64_t nonce = 19;
  int result = -1;
  std::exception_ptr error;
  Attempt() {
    std::memset(output, 0xcc, sizeof(output));
    std::memset(target, 0xff, sizeof(target));
  }
  void run() {
    try {
      result = nexapow(0, 0, input, 44, output, nullptr, &nonce, target, nullptr, 2,
                       false, false, "same-device-epoch");
    } catch (...) {
      error = std::current_exception();
    }
  }
  void unchanged() const {
    require(nonce == 19, "failed attempt committed nonce");
    for (uint8_t v : output)
      require(v == 0xcc, "failed attempt committed output");
  }
};
static int passed = 0, failed = 0;
static void test(const std::string& name, const std::function<void()>& body) {
  reset();
  try {
    body();
    state.reset();
    require(allocations.size() == expected_uncertain_allocations,
            "case retained unexpected allocations or hid uncertain retirement");
    std::printf("ok %d - %s\n", passed + failed + 1, name.c_str());
    ++passed;
  } catch (const std::exception& error) {
    std::printf("not ok %d - %s\n  # %s\n", passed + failed + 1, name.c_str(), error.what());
    ++failed;
  } catch (...) {
    std::printf("not ok %d - %s\n  # unexpected exception type\n",
                passed + failed + 1, name.c_str());
    ++failed;
  }
  // Only reclaim test-owned mock memory after a recorded negative; never native/GPU state.
  state.reset();
  for (const auto& owner : mock_owners)
    owner.second();
  mock_owners.clear();
  mock_sizes.clear();
  allocations.clear();
}
int main() {
  std::puts("TAP version 13");
  for (unsigned gib : {4u, 5u, 10u}) {
    test("memory threshold " + std::to_string(gib) + " GiB", [=] {
      device_memory = uint64_t{gib} << 30;
      Attempt a;
      a.run();
      require(!a.error && a.result == 1, "completed availability path failed");
      require(counts.setup == (gib < 5 ? 0 : 1), "wrong setup admission threshold");
      require(counts.monolithic == (gib < 5 ? 1 : 0), "wrong availability fallback");
      require(counts.large == (gib >= 10 ? 1 : 0), "wrong layout admission");
      require(counts.small == (gib == 5 ? 1 : 0), "wrong small layout admission");
    });
  }
  for (int step : {1, 2, 3}) {
    for (int kind : {-1, 0, 1, 2, 3}) {
      for (bool only : {true, false}) {
        test("allocation step=" + std::to_string(step) + " kind=" + std::to_string(kind) +
             (only ? " first-only" : " both-layouts"), [=] {
          fault = "allocation";
          allocation_step = step;
          error_kind = kind;
          first_only = only;
          Attempt a;
          a.run();
          require(!a.error && a.result == 1, "allocation absence did not retain fallback");
          require(counts.setup == 2, "allocation absence did not try smaller layout");
          require(counts.monolithic == (only ? 0 : 1), "wrong allocation fallback route");
          require(counts.searches == (only ? 1 : 0), "wrong staged allocation decision");
        });
      }
    }
  }
  for (bool only : {true, false}) {
    test(only ? "completed numeric mismatch selects small" :
                "completed numeric mismatch selects monolithic", [=] {
      numerical_mismatch = true;
      first_only = only;
      Attempt a;
      a.run();
      require(!a.error && a.result == 1, "completed numerical mismatch was fatal");
      require(counts.setup == 2 && counts.readbacks == 2 && counts.waits == 4,
              "numeric fallback occurred before completed self-check");
      require(counts.monolithic == (only ? 0 : 1), "wrong completed numeric fallback");
    });
  }
  for (const char* stage : {"copy", "dispatch", "table-row-dispatch", "selfcheck-dispatch",
                            "kernel-wait", "readback", "readback-wait"}) {
    for (int kind : {0, 1, 2, 3}) {
      for (bool cleanup_error : {false, true}) {
        test(std::string(stage) + " error=" + std::to_string(kind) +
             (cleanup_error ? " cleanup-wait-error abort/retry" : " drain/abort/retry"), [=] {
          fault = stage;
          error_kind = kind;
          cleanup_wait_error = cleanup_error;
          Attempt a;
          a.run();
          require(a.error && a.error == injected, "submitted exception was swallowed/replaced");
          a.unchanged();
          require(counts.setup == 1 && counts.small == 0, "submitted fault tried smaller table");
          require(counts.searches == 0 && counts.monolithic == 0, "submitted fault searched/fell back");
          require(allocations.empty(), "submitted fault retained table allocations");
          require(counts.cleanup_waits == 1 && counts.free_before_drain == 0,
                  "submitted catch freed pointers before its best-effort drain");
          require(!readback_destination, "submitted catch left its host readback pending");
          fault.clear();
          Attempt retry;
          retry.run();
          require(!retry.error && retry.result == 1, "explicit same-epoch retry failed");
          require(counts.setup == 2 && counts.large == 2 && counts.small == 0,
                  "checked latch skipped setup after incomplete decision");
          require(counts.monolithic == 0 && counts.searches == 1, "retry silently fell back");
        });
      }
    }
  }
  for (const char* stage : {"free-compact", "free-shifted", "free-table"}) {
    for (int kind : {0, 1, 2, 3}) {
      for (bool retired : {false, true}) {
        test(std::string(stage) + " completed-cleanup error=" + std::to_string(kind) +
             (retired ? " retired-then-threw" : " threw-before-retirement"), [=] {
          fault = stage;
          error_kind = kind;
          retire_then_throw = retired;
          expected_uncertain_allocations = retired ? 0 : 1;
          numerical_mismatch = fault == "free-table";
          Attempt a;
          a.run();
          require(a.error && a.error == injected, "completed cleanup error was hidden/replaced");
          a.unchanged();
          require(counts.setup == 1 && counts.small == 0 && counts.monolithic == 0 &&
                  counts.searches == 0, "cleanup failure enabled smaller/monolithic/search fallback");
          require(failed_retirement && release_attempts.at(failed_retirement) == 1,
                  "uncertain or retired allocation was released more than once");
          require(allocations.size() == expected_uncertain_allocations &&
                  counts.cleanup_waits == 1 &&
                  counts.free_before_drain == 0, "cleanup failure did not drain/reclaim safely");
        });
      }
    }
  }
  for (int kind : {0, 1, 2, 3}) {
    test("search error=" + std::to_string(kind) + " propagates without commit", [=] {
      fault = "search";
      error_kind = kind;
      Attempt a;
      a.run();
      require(a.error && a.error == injected, "search exception was swallowed/replaced");
      a.unchanged();
      require(counts.setup == 1 && counts.searches == 1 && counts.monolithic == 0,
              "failed search disabled/retried fallback");
    });
    test("memory query error=" + std::to_string(kind) + " does not latch", [=] {
      fault = "memory-query";
      error_kind = kind;
      Attempt a;
      a.run();
      require(a.error && a.error == injected && counts.setup == 0, "memory-query fault changed");
      a.unchanged();
      fault.clear();
      Attempt retry;
      retry.run();
      require(!retry.error && retry.result == 1 && counts.query == 2 && counts.setup == 1,
              "failed memory query latched unavailable");
    });
  }
  test("completed search with no candidate is not fallback", [] {
    found_candidate = false;
    Attempt a;
    a.run();
    require(!a.error && a.result == 0 && counts.monolithic == 0, "no candidate triggered fallback");
    a.unchanged();
  });
  test("invalid candidate aborts before nonce/output commit", [] {
    invalid_candidate = true;
    Attempt a;
    a.run();
    require(a.error && counts.monolithic == 0, "invalid candidate was committed or retried");
    a.unchanged();
  });
  state.reset();
  std::printf("1..%d\n# tests %d\n# pass %d\n# fail %d\n# skipped 0\n",
              passed + failed, passed + failed, passed, failed);
  return failed ? 1 : 0;
}
