#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

struct Counters {
  int allocations = 0, frees = 0, builds = 0, waits = 0, points = 0;
  int light = 0, full = 0, errors = 0, clears = 0, calls = 0, commits = 0;
  int queries = 0, drains = 0, free_before_drain = 0;
} counts;
static int mode, error_kind;
static const void* exception_address;
static uint32_t candidate_storage, cache_storage;
static bool scalar;
static std::string last_error;
static bool pending_build;

static void require(bool condition, const char* message) {
  if (!condition)
    throw std::logic_error(message);
}

static void inject() {
  if (error_kind == 1) {
    try {
      throw std::string("DAG fault");
    } catch (const std::string& error) {
      exception_address = &error;
      throw;
    }
  }
  if (error_kind == 2) {
    try {
      throw 53;
    } catch (const int& error) {
      exception_address = &error;
      throw;
    }
  }
  try {
    throw std::runtime_error("DAG fault");
  } catch (const std::exception& error) {
    exception_address = &error;
    throw;
  }
}

namespace sycl {
struct event { int dispatch = 0; };
struct device {};
struct queue {
  event memcpy(void*, const void*, size_t) { return {}; }
  void wait_and_throw() {
    ++counts.drains;
    pending_build = false;
    if (mode == 11)
      throw std::runtime_error("cleanup wait fault");
  }
};
static void free(void* pointer, const queue&) {
  require(pointer == &candidate_storage, "freed another resource");
  if (pending_build)
    ++counts.free_before_drain;
  ++counts.frees;
}
}

#include "cleanup.inc"

constexpr uint32_t HASH_LEN = 32, CACHE_NODE_WORDS = 16, EPOCH_LENGTH = 1U << 19, MAX_RESULTS = 15;
#include "result.inc"
struct FastModData {};
static FastModData make_fast_mod_data(uint64_t) { return {}; }
static uint64_t dataset_size(uint32_t) { return ((2ULL << 18) + 3) * 64; }
static uint64_t load64(const uint8_t* input) {
  uint64_t value;
  std::memcpy(&value, input, sizeof(value));
  return value;
}
static sycl::event build_dag(sycl::queue&, uint32_t* cache, uint32_t nodes,
                              uint32_t* dag, uint32_t offset, uint32_t count) {
  require(cache == &cache_storage && nodes == 1 && dag == &candidate_storage,
          "DAG inputs changed");
  const int dispatch = ++counts.builds;
  require(offset == static_cast<uint32_t>(dispatch - 1) * (1U << 18),
          "DAG dispatch offset changed");
  require(count == (dispatch == 3 ? 3u : 1U << 18), "DAG dispatch length changed");
  if ((mode == 6 && dispatch == 1) || (mode == 7 && dispatch == 2))
    inject();
  pending_build = true;
  return {dispatch};
}
static void sycl_wait_and_throw(sycl::event event, sycl::device) {
  if (!event.dispatch)
    return;
  ++counts.waits;
  if (((mode == 8 || mode == 11) && event.dispatch == 1) ||
      (mode == 9 && event.dispatch == 2))
    inject();
  pending_build = false;
}

class State {
public:
  sycl::device device;
  sycl::queue queue;
  bool shared_io = true, target_ready = false, points_ready = false;
  bool full_dag = false, dag_attempted = false;
  uint32_t epoch = 0, cache_nodes = 1, dag_nodes = 0;
  uint8_t header_storage[HASH_LEN]{}, target_storage[HASH_LEN]{};
  uint8_t* header = header_storage;
  uint8_t* target = target_storage;
  uint32_t* cache = &cache_storage;
  uint32_t* dag = nullptr;
  uint32_t* points = &cache_storage;
  Result result_storage{37, {77}, {}};
  Result* result = &result_storage;
  std::array<uint8_t, HASH_LEN> target_copy{};
  std::mutex mutex;

  template <typename T> T* allocate(size_t) {
    ++counts.allocations;
    if (mode >= 3 && mode <= 5)
      inject();
    return mode == 2 ? nullptr : reinterpret_cast<T*>(&candidate_storage);
  }
  bool wants_full_dag() const {
    ++counts.queries;
    if (mode == 10)
      inject();
    return mode != 1;
  }
  uint64_t now_ms() const { return 100; }
  void ensure_cache(uint32_t) { cache_nodes = 1; }
  bool ensure_points(const uint8_t*) {
    ++counts.points;
    return true;
  }
#include "state.inc"
};

static State* active_state;
static State& state_for(const std::string&) { return *active_state; }
template <bool Full, typename... Args> static sycl::event search(Args&&...) {
  if (Full)
    ++counts.full;
  else
    ++counts.light;
  active_state->result->count = 1;
  active_state->result->nonce[0] = 99;
  std::memset(active_state->result->hash[0], 17, HASH_LEN);
  return {};
}
template <bool Full, typename... Args> static sycl::event search_scalar(Args&&... args) {
  return search<Full>(args...);
}
template <bool Full, typename... Args> static sycl::event search_batched(Args&&... args) {
  return search<Full>(args...);
}
#include "validation.inc"
#include "caller.inc"

static uint8_t input[HASH_LEN + sizeof(uint64_t)]{}, target[HASH_LEN]{};
static int call(uint8_t* output, uint64_t* nonce) {
  ++counts.calls;
  return octopus(0, 0, input, sizeof(input), output, nullptr, nonce, target, nullptr,
                 8, false, true, "fixture");
}
static void send_error(const std::string& error) {
  ++counts.errors;
  last_error = error;
}
static bool has_fn = true;
static void clear_fn() {
  ++counts.clears;
  has_fn = false;
}
static void compute_owner(uint8_t* output, uint64_t* nonce) {
  for (int tick = 0; tick != 2; ++tick) {
    if (!has_fn)
      continue;
    try {
      call(output, nonce);
#include "compute-catch.inc"
    ++counts.commits;
  }
}

static void run() {
  const bool fault = mode >= 6;
  State state;
  active_state = &state;
  uint8_t output[HASH_LEN];
  std::memset(output, 23, sizeof(output));
  uint64_t nonce = 71;
  int returned = -1;
  bool caught = false, same_exception = false;
  try {
    returned = call(output, &nonce);
  } catch (const std::string& error) {
    caught = true;
    same_exception = error_kind == 1 && &error == exception_address;
  } catch (const std::exception& error) {
    caught = true;
    same_exception = error_kind == 0 && &error == exception_address;
  } catch (const int& error) {
    caught = true;
    same_exception = error_kind == 2 && &error == exception_address;
  }
  require(caught == fault, "DAG fault was swallowed before the real search caller");
  if (fault) {
    require(same_exception, "DAG fault changed exception type/identity");
    require(returned == -1 && nonce == 71 && output[0] == 23 && state.result->count == 37,
            "failed DAG initialization committed/reset output");
    require(counts.points == 0 && counts.light == 0 && counts.full == 0,
            "failed DAG initialization continued preparing/submitting search");
    const int failed_allocations = mode == 10 ? 0 : 1;
    require(counts.frees == failed_allocations && !state.dag && !state.full_dag && !state.dag_attempted,
            "failed DAG candidate cleanup/retry state is stale");
    require(counts.free_before_drain == 0 && !pending_build,
            "failed DAG candidate was freed before accepted work completed");
    const int fault_mode = mode;
    mode = 0;
    counts.builds = 0;
    counts.waits = 0;
    require(call(output, &nonce) == 1 && state.full_dag && state.dag_attempted &&
            counts.allocations == failed_allocations + 1 && counts.queries == 2 &&
            counts.builds == 3 && counts.waits == 3 &&
            counts.light == 0 && counts.full == 1, "same-epoch retry skipped required DAG build");
    state.free_ptr(state.dag);
    require(counts.frees == failed_allocations + 1,
            "successful retry DAG was not released exactly once");

    State worker_state;
    active_state = &worker_state;
    counts = {};
    mode = fault_mode;
    has_fn = true;
    compute_owner(output, &nonce);
    require(counts.calls == 1 && counts.errors == 1 && counts.clears == 1 &&
            counts.commits == 0 && !has_fn && counts.light == 0 && counts.full == 0 &&
            counts.frees == failed_allocations && counts.free_before_drain == 0 && !pending_build,
            "real compute catch did not report/clear the failed dispatch");
    require(last_error == (error_kind == 2 ? "Compute function exception" :
            "Compute function exception: DAG fault"), "real compute error detail changed");
  } else {
    const bool full = mode == 0;
    require(returned == 1 && nonce == 99 && output[0] == 17 && state.full_dag == full,
            "successful/unsupported DAG search changed");
    require(counts.light == (full ? 0 : 1) && counts.full == (full ? 1 : 0) &&
            counts.builds == (full ? 3 : 0) && counts.waits == (full ? 3 : 0),
            "pre-submit absence or successful full-DAG route changed");
    const int allocations = counts.allocations;
    require(call(output, &nonce) == 1 && counts.allocations == allocations && counts.queries == 1,
            "completed/unavailable DAG is no longer cached");
    state.free_ptr(state.dag);
    state.free_ptr(state.dag);
    require(counts.frees == (full ? 1 : 0), "DAG ownership was released twice");
  }
  require(!pending_build && counts.free_before_drain == 0, "DAG work outlived its allocation");
}

int main(int argc, char** argv) {
  if (argc != 4)
    return 2;
  mode = std::atoi(argv[1]);
  error_kind = std::atoi(argv[2]);
  scalar = std::atoi(argv[3]) != 0;
  unsetenv("MOM_OCTOPUS_TEST_FULL_DAG");
  unsetenv("MOM_OCTOPUS_TEST_NATIVE");
  if (scalar)
    setenv("MOM_OCTOPUS_SCALAR", "1", 1);
  else
    unsetenv("MOM_OCTOPUS_SCALAR");
  try {
    run();
    std::puts("PASS actual-source Octopus DAG fault boundary");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
