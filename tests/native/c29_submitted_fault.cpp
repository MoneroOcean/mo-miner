#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

static int mode, proof_size, owner;
static int calls, seeds, trims, waits, copies, thread_attempts, threads, errors, clears;
static const void* original_exception;
static bool m_has_fn = true;
static std::string last_error;
static void require(bool ok, const char* message) {
  if (!ok)
    throw std::logic_error(message);
}
namespace sycl {
struct exception : std::runtime_error { using std::runtime_error::runtime_error; };
struct device {};
struct event {};
struct uint2 {
  uint32_t a = 0, b = 0;
  uint2() = default;
  uint2(uint32_t x, uint32_t y) : a(x), b(y) {}
  uint32_t x() const { return a; }
  uint32_t y() const { return b; }
};
struct ulong4 { template <typename... T> explicit ulong4(T...) {} };
enum { read_only, write_only, read_write, no_init };
namespace access {
enum class fence_space { local_space };
enum class address_space { local_space };
}
enum class memory_order { relaxed };
enum class memory_scope { work_group };
template <int D> struct range { explicit range(size_t) {} };
template <int D> struct nd_range { nd_range(range<D>, range<D>) {} };
template <int D> struct nd_item {
  uint32_t get_global_id(int) const { return 0; }
  uint32_t get_local_id(int) const { return 0; }
  void barrier(access::fence_space) const {}
};
struct handler {
  template <typename A, typename T> void fill(A, T) {}
  template <typename R, typename F> void parallel_for(R, F) {}
};
template <typename T, int D> struct buffer {
  std::vector<T> data;
  explicit buffer(range<D>) : data(64) {}
  buffer(T* input, range<D>) : data(input, input + proof_size) {}
};
template <typename T, int D> struct accessor {
  buffer<T, D>& value;
  accessor(buffer<T, D>& b, handler&, int, int = 0) : value(b) {}
  T& operator[](size_t i) const { return value.data[i]; }
};
template <typename T, int D> accessor(buffer<T, D>&, handler&, int) -> accessor<T, D>;
template <typename T, int D> accessor(buffer<T, D>&, handler&, int, int) -> accessor<T, D>;
template <typename T, int D> struct host_accessor {
  buffer<T, D>& value;
  host_accessor(buffer<T, D>& b, int) : value(b) {}
  T* get_pointer() const { return value.data.data(); }
};
template <typename T, int D> host_accessor(buffer<T, D>&, int) -> host_accessor<T, D>;
template <typename T, int D> struct local_accessor {
  mutable T data[64]{};
  local_accessor(range<D>, handler&) {}
  T& operator[](size_t i) const { return data[i]; }
};
template <typename T, memory_order, memory_scope, access::address_space> struct atomic_ref {
  T& value;
  explicit atomic_ref(T& v) : value(v) {}
  T fetch_min(T other) {
    const T before = value;
    value = std::min(value, other);
    return before;
  }
};
struct queue {
  device get_device() const;
  template <typename F> event submit(F f) {
    handler h;
    f(h);
    return {};
  }
};
template <typename T> T min(T a, T b) { return std::min(a, b); }
}
static void inject() {
  try {
    throw sycl::exception("C29 submitted graph fault");
  } catch (const sycl::exception& error) {
    original_exception = &error;
    throw;
  }
}
sycl::device sycl::queue::get_device() const {
  if (mode == 1)
    inject();
  return {};
}
constexpr uint32_t EDGE_BITS = 29, NUM_EDGES = 1u << EDGE_BITS, EDGE_BLOCK_SIZE = 64, EDGE_BLOCK_MASK = 63;
constexpr uint64_t EDGE_MASK = NUM_EDGES - 1;
constexpr uint32_t MAX_TRIMMED_EDGE_COUNT = 64;
#define MOM_SYCL_KERNEL_ARGS_RESTRICT
static void siphash_fill_block(uint64_t&, uint64_t&, uint64_t&, uint64_t&, uint64_t,
                                uint64_t* out) { std::fill_n(out, EDGE_BLOCK_SIZE, 0); }
struct C29Buffers {};
struct C29Profile {
  C29Profile(unsigned, uint64_t, unsigned, sycl::device) {}
  void mark(const char*) {}
  void print_events() {}
  void finish(uint32_t) {}
};
#include "result-state.inc"
#include "cycle.inc"
#include "job-boundary.inc"
struct C29State {
  sycl::queue queue;
  C29Buffers buffers;
  std::shared_ptr<C29ResultState> result_state = std::make_shared<C29ResultState>();
};
static C29State* active;
static C29State& c29_state(const std::string& dev) {
  require(dev == "fixture", "device binding changed");
  return *active;
}
#include "activate.inc"
static void set_sycl_env(const char*, const char*) {}
static int c29_portable_test(const uint8_t*, unsigned, uint8_t*, const std::string&) {
  throw std::logic_error("hardware/portable guard unexpectedly selected");
}
static void rx_blake2b(void* output, size_t n, const void*, size_t) { std::memset(output, 31, n); }
struct Atomic {
  uint32_t fetch_min(uint32_t n) { return n; }
};
template <typename A> static Atomic c29_global_atomic(const A&, size_t) { return {}; }
static void sycl_wait_and_throw(sycl::event, sycl::device) {
  ++waits;
  if (mode == 4)
    inject();
}
static void c29_read_buffer(sycl::queue&, sycl::buffer<uint32_t, 1>&, uint32_t* out, unsigned) {
  ++copies;
  if (mode == 5)
    inject();
  *out = static_cast<uint32_t>(proof_size);
}
static void c29_read_buffer(sycl::queue&, sycl::buffer<sycl::uint2, 1>&,
                            sycl::uint2* out, unsigned n) {
  ++copies;
  if (mode == 6)
    inject();
  for (uint32_t i = 0; i != n / 2; ++i) {
    out[i * 2] = {i, i};
    out[i * 2 + 1] = {(i + 1) % (n / 2), i};
  }
}
// Deterministic host-thread boundary; the actual moved completion guard and cycle body are retained.
struct Task { virtual ~Task() = default; virtual void run() = 0; };
template <typename F> struct TypedTask : Task {
  F fn;
  explicit TypedTask(F&& f) : fn(std::move(f)) {}
  void run() override { fn(); }
};
static std::vector<std::unique_ptr<Task>> tasks;
struct TestThread {
  std::unique_ptr<Task> task;
  template <typename F> explicit TestThread(F&& f)
      : task(std::make_unique<TypedTask<F>>(std::move(f))) {
    ++thread_attempts;
    if (mode == 7)
      throw std::runtime_error("thread unavailable");
  }
  void detach() {
    ++threads;
    tasks.push_back(std::move(task));
  }
};
#include "search.inc"
static int call(unsigned job, unsigned proof, const uint8_t* input, unsigned n, uint8_t* output,
                 uint32_t* edges, uint64_t* nonce, const std::string& dev) {
  ++calls;
  return c29(job, proof, input, n, output, edges, nonce, dev);
}
static void send_error(const std::string& value) {
  ++errors;
  last_error = value;
}
static void clear_fn() {
  ++clears;
  m_has_fn = false;
}
enum class DEV { C29_GPU, PEARLHASH_GPU, ZELHASH_GPU, BEAMHASH3_GPU };
static uint64_t pearlhash_attempt_hashes(unsigned, unsigned, unsigned, unsigned) { return 0; }
static uint8_t input[4] = {0, 0, 0, 71};
static uint8_t* nonce_address() { return input; }
static std::atomic<uint64_t> m_hash_count{0};
static void compute_owner(uint8_t* output, uint32_t* edges) {
  const DEV m_dev = DEV::C29_GPU;
  const bool is_test = false;
  const unsigned m_nonce_bytes = 4, dispatch_job_ref = 9, m_c29_proof_size = proof_size;
  const auto m_input = input;
  const unsigned m_input_len = sizeof(input), m_batch = 1;
  const auto m_output = output;
  void* m_spads = edges;
  const std::string m_dev_str = "fixture", m_algo_str = "c29";
  const unsigned m_pearlhash_n = 1, m_pearlhash_k = 1, m_pearlhash_rank = 1;
  struct { decltype(&call) gpu_c29 = call; } m_fn;
  for (int tick = 0; tick != 2; ++tick) {
#include "active-guard.inc"
    int dev_sols = 0;
    uint64_t dev_nonce = 0;
    try {
      switch (m_dev) {
#include "dispatch.inc"
        default: throw std::logic_error("unexpected device");
      }
#include "compute-catch.inc"
#include "accounting.inc"
  }
}
static void run() {
  C29State state;
  active = &state;
  uint8_t output[32];
  std::memset(output, 23, sizeof(output));
  uint32_t edges[42]{};
  uint64_t nonce = 71;
  const bool fault = mode >= 1 && mode <= 6;
  if (owner) {
    compute_owner(output, edges);
    require(calls == (fault ? 1 : 2), "compute continued dispatching a faulted graph");
    require(errors == (fault ? 1 : 0) && clears == (fault ? 1 : 0) && m_has_fn != fault,
            "real Core catch did not report and clear the submitted fault");
    require(m_hash_count == (fault ? 0 : 2), "failed graph committed hash accounting");
    if (fault)
      require(last_error == "Compute function exception: C29 submitted graph fault",
              "compute error lost the original SYCL detail");
  } else {
    int result = -9;
    bool caught = false;
    try {
      result = call(9, proof_size, input, sizeof(input), output, edges, &nonce, "fixture");
    } catch (const sycl::exception& error) {
      caught = true;
      require(&error == original_exception, "submitted exception identity changed");
    }
    require(caught == fault, "submitted graph fault was swallowed by full search/entry");
    require(result == (fault ? -9 : 0), "graph entry return changed");
  }
  require(nonce == 71 && output[0] == 23, "failed/pending graph committed nonce/output");
  require(thread_attempts == (fault ? 0 : calls), "fault started a CPU cycle thread");
  require(threads == (mode == 7 || fault ? 0 : calls), "thread-start boundary changed");
  for (auto& task : tasks)
    task->run();
  tasks.clear();
  require(state.result_state->running_searches.empty(), "completion guard left a stale running graph");
  require(state.result_state->solutions.size() == (mode == 7 || fault ? 0u : static_cast<size_t>(calls)),
          "valid cycle/legacy thread-start result changed");
  for (const auto& solution : state.result_state->solutions) {
    require(solution.edges.size() == static_cast<unsigned>(proof_size) &&
            solution.job_ref == 9 && solution.nonce == 71, "valid graph metadata changed");
    for (const auto word : solution.seed)
      require(word == 0x1f1f1f1f1f1f1f1fULL, "valid graph seed changed");
  }
}
int main(int argc, char** argv) {
  if (argc != 4)
    return 2;
  mode = std::atoi(argv[1]);
  proof_size = std::atoi(argv[2]);
  owner = std::atoi(argv[3]);
  unsetenv("MOM_C29_TEST_EDGE");
  try {
    run();
    std::puts("PASS actual-source C29 submitted fault boundary");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
