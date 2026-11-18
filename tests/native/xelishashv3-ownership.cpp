#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

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
struct Result {};
constexpr std::size_t WORDS = 2;
static std::uint64_t xelishashv3_scratch_capacity(const sycl::device&) {
  return scratch_limit;
}
class State {
public:
  sycl::device device;
  sycl::queue queue;
  std::uint8_t *input = nullptr, *target = nullptr;
  std::uint64_t* scratch = nullptr;
  double* lut = nullptr;
  Result* result = nullptr;
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
  state.input = sycl::malloc_device<std::uint8_t>(1, state.queue);
  state.target = sycl::malloc_device<std::uint8_t>(1, state.queue);
  state.scratch = sycl::malloc_device<std::uint64_t>(WORDS, state.queue);
  state.lut = sycl::malloc_device<double>(1, state.queue);
  state.result = sycl::malloc_device<Result>(1, state.queue);
  state.capacity = 1;
  require(state.input && state.target && state.scratch && state.lut && state.result,
          "fixture setup allocation failed");
}
static void cleanup_cases() {
  reset();
  { State state; allocate_all(state); state.cleanup(); state.cleanup();
    require(ledger.free_calls == 5 && ledger.live.empty() && !ledger.invalid_free,
            "ordinary cleanup leaked or repeated a resource"); }
  reset();
  { State state; allocate_all(state); throw_free = true; state.cleanup(); state.cleanup();
    require(ledger.free_calls == 5 && ledger.live.empty() && !ledger.invalid_free,
            "throwing frees did not detach every resource"); }
  reset();
  { State state; allocate_all(state); throw_free = true; throw_wait = true;
    bool caught = false;
    try { state.cleanup(); } catch (const std::runtime_error& error) {
      caught = std::string(error.what()) == "original wait failure";
    }
    const unsigned frees = ledger.free_calls;
    try { state.cleanup(); } catch (const std::runtime_error& error) {
      caught = caught && std::string(error.what()) == "original wait failure";
    }
    require(caught && frees == 5 && ledger.free_calls == frees && ledger.live.empty(),
            "cleanup replaced the wait failure or retried frees"); }
}
static void grow_cases() {
  reset();
  { State state; state.scratch = sycl::malloc_device<std::uint64_t>(WORDS, state.queue);
    state.capacity = 1; auto* const old = state.scratch; fail_alloc = true;
    bool caught = false;
    try { state.ensure(2); } catch (const std::string&) { caught = true; }
    require(caught && state.scratch == old && state.capacity == 1 && ledger.free_calls == 0,
            "failed scratch growth discarded the old allocation");
    state.cleanup(); }
  reset();
  { State state; state.scratch = sycl::malloc_device<std::uint64_t>(WORDS, state.queue);
    state.capacity = 1; auto* const old = state.scratch; throw_free = true;
    state.ensure(2);
    auto* const replacement = state.scratch;
    require(replacement != old && state.capacity == 2 && ledger.free_calls == 1 &&
                ledger.frees[old] == 1 && !ledger.live.count(old) && ledger.live.count(replacement),
            "scratch growth did not commit the replacement exactly once");
    state.cleanup(); state.cleanup();
    require(ledger.free_calls == 2 && ledger.live.empty() && !ledger.invalid_free,
            "replacement cleanup leaked or repeated a free"); }
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
