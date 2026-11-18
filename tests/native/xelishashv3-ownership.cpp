#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

struct Ledger {
  std::set<void*> live;
  std::map<void*, unsigned> frees;
  unsigned allocations = 0, free_calls = 0, waits = 0, suppressed = 0;
  bool invalid_free = false;
} ledger;
static bool fail_alloc, throw_free, throw_wait;
static std::uint64_t scratch_limit = 100;

static void require(bool ok, const char* message) {
  if (!ok)
    throw std::logic_error(message);
}
namespace sycl {
struct device {};
struct queue {
  void wait_and_throw() const {
    ++ledger.waits;
    if (throw_wait)
      throw std::runtime_error("original wait failure");
  }
};

template <typename T> T* malloc_device(std::size_t count, const queue&) {
  if (fail_alloc)
    return nullptr;
  T* pointer = new T[count]{};
  ledger.live.insert(pointer);
  ++ledger.allocations;
  return pointer;
}

template <typename T> T* malloc_shared(std::size_t count, const queue& q) {
  return malloc_device<T>(count, q);
}

template <typename T> void free(T* pointer, const queue&) {
  if (!pointer)
    return;
  void* raw = pointer;
  if (!ledger.live.erase(raw))
    ledger.invalid_free = true;
  ++ledger.free_calls;
  ++ledger.frees[raw];
  delete[] pointer;
  if (throw_free)
    throw std::runtime_error("USM free failure");
}
} // namespace sycl
template <typename Fn> void sycl_cleanup_noexcept(const char*, Fn&& fn) noexcept {
  try {
    fn();
  } catch (...) {
    ++ledger.suppressed;
  }
}
#include "usm.inc"
struct Result {};
constexpr std::size_t WORDS = 2;
static std::uint64_t xelishashv3_scratch_capacity(const sycl::device&) {
  return scratch_limit;
}
class State {
public:
  sycl::device device;
  sycl::queue queue;
  UsmPtr<std::uint8_t> input, target;
  UsmPtr<std::uint64_t> scratch;
  UsmPtr<double> lut;
  UsmPtr<Result> result;
  unsigned capacity = 0;

#include "ownership.inc"
};
static void reset() {
  require(ledger.live.empty(), "fixture leaked an allocation");
  ledger = {};
  fail_alloc = throw_free = throw_wait = false;
  scratch_limit = 100;
}
static void allocate_all(State& state) {
  state.input = mom_usm_alloc<std::uint8_t>(state.queue, 1, false, "input");
  state.target = mom_usm_alloc<std::uint8_t>(state.queue, 1, false, "target");
  state.scratch = mom_usm_alloc<std::uint64_t>(state.queue, WORDS, false, "scratch");
  state.lut = mom_usm_alloc<double>(state.queue, 1, false, "lut");
  state.result = mom_usm_alloc<Result>(state.queue, 1, false, "result");
  state.capacity = 1;
}
static void cleanup_cases() {
  reset();
  { State state; allocate_all(state); }
  require(ledger.free_calls == 5 && ledger.waits == 1 && ledger.live.empty() &&
              !ledger.invalid_free,
          "ordinary teardown leaked or repeated a resource");
  reset();
  { State state; allocate_all(state); throw_free = true; }
  require(ledger.free_calls == 5 && ledger.suppressed == 5 && ledger.live.empty() &&
              !ledger.invalid_free,
          "throwing frees did not detach every resource");
  reset();
  { State state; allocate_all(state); throw_free = true; throw_wait = true; }
  require(ledger.free_calls == 5 && ledger.suppressed == 6 && ledger.live.empty(),
          "a failed teardown wait skipped a free");
}
static void grow_cases() {
  reset();
  { State state; state.scratch = mom_usm_alloc<std::uint64_t>(state.queue, WORDS, false, "old");
    state.capacity = 1; auto* const old = state.scratch.get(); fail_alloc = true;
    bool caught = false;
    try { state.ensure(2); } catch (const std::string&) { caught = true; }
    require(caught && state.scratch.get() == old && state.capacity == 1 && ledger.free_calls == 0,
            "failed scratch growth discarded the old allocation");
    fail_alloc = false; }
  reset();
  { State state; state.scratch = mom_usm_alloc<std::uint64_t>(state.queue, WORDS, false, "old");
    state.capacity = 1; auto* const old = state.scratch.get(); throw_free = true;
    state.ensure(2);
    auto* const replacement = state.scratch.get();
    require(replacement != old && state.capacity == 2 && ledger.free_calls == 1 &&
                ledger.frees[old] == 1 && !ledger.live.count(old) && ledger.live.count(replacement),
            "scratch growth did not commit the replacement exactly once"); }
  require(ledger.free_calls == 2 && ledger.live.empty() && !ledger.invalid_free,
          "replacement teardown leaked or repeated a free");
}
int main() {
  try {
    cleanup_cases();
    grow_cases();
    std::puts("PASS XelisHashV3 ownership cleanup and growth");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
