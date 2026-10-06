#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static int kind, fault_wait, layout, owner;
static int waits, kernels, copies, cycles, calls, errors, clears, results;
static int copy_fault, copy_error_kind, cleanup_waits;
static bool cleanup_error;
static bool m_has_fn = true;
static std::string last_error;
static std::exception_ptr original_error;
static void require(bool ok, const char* detail) {
  if (!ok)
    throw std::logic_error(detail);
}
namespace sycl {
struct exception : std::runtime_error { using std::runtime_error::runtime_error; };
using exception_list = std::vector<std::exception_ptr>;
using async_handler = std::function<void(exception_list)>;
namespace info::device { struct vendor {}; }
namespace property::queue { struct in_order {}; }
struct property_list { explicit property_list(property::queue::in_order) {} };
struct device {
  template <typename T> std::string get_info() const { return "Intel"; }
};
struct uint2 {
  std::uint32_t a = 0, b = 0;
  uint2() = default;
  uint2(std::uint32_t x, std::uint32_t y) : a(x), b(y) {}
  std::uint32_t x() const { return a; }
  std::uint32_t y() const { return b; }
};
struct queue;
struct PendingCopy {
  void* destination;
  const void* source;
  std::size_t bytes;
};
static std::vector<PendingCopy> pending_copies;
struct event {
  queue* value = nullptr;
  void wait_and_throw() const;
};
struct queue {
  async_handler handler;
  queue(device, async_handler h, property_list) : handler(std::move(h)) {}
  event memcpy(void* out, const void* in, std::size_t n) {
    ++copies;
    if (copy_fault == copies)
      std::rethrow_exception(original_error);
    pending_copies.push_back({out, in, n});
    return {this};
  }
  template <typename T> event fill(T* out, T value, std::size_t n) {
    std::fill_n(out, n, value);
    return {this};
  }
  void complete() {
    for (const PendingCopy& copy : pending_copies)
      std::memcpy(copy.destination, copy.source, copy.bytes);
    pending_copies.clear();
  }
  void wait_and_throw() {
    ++cleanup_waits;
    complete();
    if (cleanup_error)
      throw std::string("secondary C30 cleanup fault");
  }
};
void event::wait_and_throw() const {
  ++waits;
  value->complete();
  exception_list pending;
  if (kind && waits == fault_wait) {
    if (kind == 3)
      original_error = std::make_exception_ptr(std::logic_error("C30 non-SYCL completion fault"));
    else
      original_error = std::make_exception_ptr(exception("C30 submitted completion fault"));
    pending.push_back(original_error);
    if (kind == 2)
      pending.push_back(std::make_exception_ptr(exception("C30 second completion fault")));
  }
  value->handler(pending);
}
}
static sycl::device get_dev(const std::string& dev) {
  require(dev == "fixture", "device binding changed");
  return {};
}
static void sycl_wait_and_throw(sycl::event event, sycl::device) { event.wait_and_throw(); }
#include "cleanup.inc"
static bool mom_parse_env_ulong(const char*, unsigned long&) { return false; }
struct C30DeviceKey { std::uint64_t k0, k1, k2, k3; };
struct C30PackedEdge {};
constexpr unsigned C30_BUCKET_COUNT = 2;
#include "c30-types.inc"
namespace mom::c30::host {
struct SipKey { std::uint64_t k0, k1, k2, k3; };
struct Endpoints { std::uint32_t u = 0, v = 0; };
static SipKey graph_key(const std::array<std::uint8_t, 32>&, std::uint64_t nonce) {
  return {nonce, 2, 3, 4};
}
static std::optional<std::array<Endpoints, kProofSize>>
find_cycle(const std::vector<Endpoints>&) {
  ++cycles;
  return std::array<Endpoints, kProofSize>{};
}
static std::array<std::uint8_t, 32> solution_hash(const std::array<std::uint32_t, kProofSize>&) {
  std::array<std::uint8_t, 32> hash{};
  hash.fill(42);
  return hash;
}
static bool verify(SipKey, const std::array<std::uint32_t, kProofSize>& edges) {
  for (unsigned i = 0; i != kProofSize; ++i)
    if (edges[i] != i)
      return false;
  return true;
}
}
namespace mom::c30 {
constexpr unsigned kDefaultTrimRounds = 384, kDefaultSparseTrimStart = 64;
constexpr unsigned kSeedBucketStride = 1, kTrimBucketStride = 1, kLowMemoryTrimBucketStride = 1;
constexpr unsigned kMaxHostEdges = 64;
struct DeviceMemory {
  std::array<std::array<sycl::uint2, kProofSize>, 2> edge_storage{};
  std::array<std::array<C30PackedEdge, kProofSize>, 2> packed_storage{};
  std::array<std::array<std::uint32_t, C30_BUCKET_COUNT>, 2> count_storage{};
  std::array<std::uint32_t, kProofSize> u{}, v{}, recovered_storage{};
  std::uint32_t cursor_storage = 0, overflow_storage = 0;
  std::array<sycl::uint2*, 2> edges{edge_storage[0].data(), edge_storage[1].data()};
  std::array<C30PackedEdge*, 2> packed_edges{packed_storage[0].data(), packed_storage[1].data()};
  std::array<std::uint32_t*, 2> bucket_counts{count_storage[0].data(), count_storage[1].data()};
  std::uint32_t* cursors = &cursor_storage;
  std::uint32_t* overflow = &overflow_storage;
  std::uint32_t* target_u = u.data();
  std::uint32_t* target_v = v.data();
  std::uint32_t* recovered = recovered_storage.data();
  bool low_memory = layout != 0;
  explicit DeviceMemory(sycl::queue&) {}
  void allocate_edges(unsigned, std::uint64_t) {}
};
static std::uint32_t count_edges(const std::vector<std::uint32_t>& values) {
  std::uint32_t sum = 0;
  for (const auto value : values)
    sum += value;
  return sum;
}
static sycl::event c30_seed_fixed(sycl::queue& q, C30DeviceKey, std::uint32_t* counts,
                                  unsigned, sycl::uint2*, std::uint32_t*) {
  ++kernels;
  counts[0] = kProofSize;
  return {&q};
}
static sycl::event c30_seed_packed(sycl::queue& q, C30DeviceKey, std::uint32_t* counts,
                                   unsigned, C30PackedEdge*, std::uint32_t*) {
  ++kernels;
  counts[0] = kProofSize;
  return {&q};
}
template <typename T>
static sycl::event trim(sycl::queue& q, T*, std::uint32_t*, unsigned,
                        std::uint32_t* out_counts, unsigned, T*, std::uint32_t*) {
  ++kernels;
  out_counts[0] = kProofSize;
  return {&q};
}
#define c30_trim_fixed trim
#define c30_trim_sparse_fixed trim
#define c30_trim_packed trim
#define c30_trim_sparse_packed trim
template <typename T>
static sycl::event compact(sycl::queue& q, T*, std::uint32_t*, unsigned,
                           sycl::uint2*, std::uint32_t* cursor) {
  ++kernels;
  *cursor = kProofSize;
  return {&q};
}
#define c30_compact compact
#define c30_compact_packed compact
static sycl::event c30_recover(sycl::queue& q, C30DeviceKey, std::uint32_t*,
                               std::uint32_t*, std::uint32_t* out) {
  ++kernels;
  for (unsigned i = 0; i != kProofSize; ++i)
    out[i] = i;
  return {&q};
}
#define c30_recover32 c30_recover
#include "solver.inc"
}
struct C30State {
  mom::c30::Solver solver;
  std::mutex mutex;
  C30State() : solver("fixture") {}
};
static C30State* active;
static C30State& c30_state(const std::string& dev) {
  require(dev == "fixture", "entry device binding changed");
  return *active;
}
static int c30_portable_test(const mom::c30::Job&, std::uint8_t*, const std::string&) {
  throw std::logic_error("hardware opt-in unexpectedly selected");
}
#include "entry.inc"
#include "job-boundary.inc"
static int call(unsigned job, unsigned proof, const std::uint8_t* input, unsigned n,
                 std::uint8_t* output, std::uint32_t* edges, std::uint64_t* nonce,
                 const std::string& dev) {
  ++calls;
  return c30(job, proof, input, n, output, edges, nonce, dev);
}
static void send_error(const std::string& message) {
  ++errors;
  last_error = message;
}
static void clear_fn() {
  ++clears;
  m_has_fn = false;
}
static void send_last_nonce(std::uint64_t, unsigned, int, int, int) {}
static void send_result(std::uint64_t nonce, unsigned, const std::uint8_t* hash,
                         const std::uint32_t* edges, unsigned proof) {
  require(nonce >= 71 && hash[0] == 42 && edges[41] == 41 && proof == 42,
          "successful Core result changed");
  ++results;
}
enum class DEV { C29_GPU, C30_GPU, PEARLHASH_GPU, ZELHASH_GPU, BEAMHASH3_GPU };
static std::uint64_t pearlhash_attempt_hashes(unsigned, unsigned, unsigned, unsigned) { return 0; }
static std::atomic<std::uint64_t> m_hash_count{0};
static std::uint64_t m_nonce64 = 71;
static void compute_owner(std::uint8_t* output, std::uint32_t* edges) {
  const DEV m_dev = DEV::C30_GPU;
  const unsigned dispatch_job_ref = 9, m_c29_proof_size = 42, m_batch = 1;
  std::uint8_t input[32]{};
  const auto m_input = input;
  const unsigned m_input_len = sizeof(input);
  const auto m_output = output;
  void* m_spads = edges;
  const std::string m_dev_str = "fixture", m_algo_str = "c30";
  const unsigned m_pearlhash_n = 1, m_pearlhash_k = 1, m_pearlhash_rank = 1;
  const std::uint64_t m_target = 1, m_nonce_step = 1, m_nicehash_mask = 0;
  struct { decltype(&call) gpu_c29 = call; } m_fn;
  const int m_pool_id = 1, m_job_id = 2, m_job_token = 3;
  std::uint8_t target[32];
  std::memset(target, 255, sizeof(target));
  const auto m_target_bin = target;
  constexpr unsigned HASH_LEN = 32;
  for (unsigned tick = 0; tick != 2; ++tick) {
#include "active-guard.inc"
    int dev_sols = 0;
    std::uint64_t dev_nonce = 0;
    try {
      switch (m_dev) {
#include "dispatch.inc"
        default: throw std::logic_error("unexpected Core route");
      }
#include "compute-catch.inc"
#include "accounting.inc"
#include "nonce-commit.inc"
  }
}
static void run() {
  C30State state;
  active = &state;
  std::uint8_t input[32]{}, output[32];
  std::memset(output, 23, sizeof(output));
  std::uint32_t edges[42];
  std::fill_n(edges, 42, 99);
  std::uint64_t nonce = 71;
  if (owner) {
    compute_owner(output, edges);
    require(errors == (kind ? 1 : 0) && clears == (kind ? 1 : 0) && m_has_fn != (kind != 0),
            "Core did not clear and report the completion error");
    require(calls == (kind ? 1 : 2), "Core dispatched another graph after the fault");
    require(m_hash_count == (kind ? 0 : 2) && results == (kind ? 0 : 2) &&
            m_nonce64 == (kind ? 71 : 73), "fault committed Core hashes/results/nonce");
    if (kind)
      require(last_error == (kind == 3
          ? "Compute function exception: C30 non-SYCL completion fault"
          : "Compute function exception: C30 submitted completion fault"), "Core error detail changed");
  } else {
    bool caught = false;
    int result = -9;
    try {
      result = call(9, 42, input, sizeof(input), output, edges, &nonce, "fixture");
    } catch (const std::exception&) {
      caught = true;
      require(std::current_exception() == original_error, "completion exception identity changed");
    }
    require(caught == (kind != 0) && result == (kind ? -9 : 1), "completion error was swallowed");
  }
  if (kind) {
    require(waits == fault_wait, "solver passed a faulted wait into another stage");
    require(output[0] == 23 && edges[0] == 99 && nonce == 71, "fault committed entry output/nonce");
    const int kernels_before[] = {0, 1, 385, 385, 386, 386, 386, 387, 387};
    const int copies_before[] = {0, 0, 0, 2, 2, 3, 4, 6, 7};
    require(kernels == kernels_before[fault_wait] && copies == copies_before[fault_wait],
            "solver continued submitting kernels/readbacks after failed completion");
    require(cycles == (fault_wait >= 7 ? 1 : 0), "fault advanced CPU cycle search");
  } else {
    require(output[0] == 42 && edges[41] == 41 && cycles == calls, "successful graph changed");
  }
  // Only an explicit later job reactivates the owner; reuse is not claimed as real GPU recovery.
  kind = 0;
  m_has_fn = true;
  const int before = calls;
  if (owner)
    compute_owner(output, edges);
  else
    require(call(10, 42, input, sizeof(input), output, edges, &nonce, "fixture") == 1,
            "later explicit graph failed");
  require(calls == before + (owner ? 2 : 1) && output[0] == 42 && edges[41] == 41,
          "later explicit job did not resume the normal success path");
}
static void run_copy_fault() {
  C30State state;
  active = &state;
  std::uint8_t input[32]{}, output[32];
  std::memset(output, 23, sizeof(output));
  std::uint32_t edges[42];
  std::fill_n(edges, 42, 99);
  std::uint64_t nonce = 71;
  try {
    if (copy_error_kind == 0)
      throw std::runtime_error("C30 copy submission fault");
    if (copy_error_kind == 1)
      throw std::string("C30 copy submission fault");
    throw 37;
  } catch (...) {
    original_error = std::current_exception();
  }
  std::exception_ptr caught;
  try {
    call(9, 42, input, sizeof(input), output, edges, &nonce, "fixture");
  } catch (...) {
    caught = std::current_exception();
  }
  require(caught == original_error, "copy submission changed the original exception");
  require(sycl::pending_copies.empty(), "copy submission unwound host locals before drain");
  require(cleanup_waits == 1, "copy failure did not retire exactly once before solver destruction");
  require(copies == copy_fault && calls == 1, "copy submission retried another graph or transfer");
  require(cycles == (copy_fault == 6 ? 1 : 0), "copy failure advanced the cycle search");
  require(output[0] == 23 && edges[0] == 99 && nonce == 71, "copy failure committed output/nonce");
}
int main(int argc, char** argv) {
  if (argc != 5 && argc != 6)
    return 2;
  const bool copy_case = argc == 6 && std::strcmp(argv[1], "copies") == 0;
  if (copy_case) {
    copy_fault = std::atoi(argv[2]);
    copy_error_kind = std::atoi(argv[3]);
    cleanup_error = std::atoi(argv[4]) != 0;
    layout = std::atoi(argv[5]);
  } else {
    kind = std::atoi(argv[1]);
    fault_wait = std::atoi(argv[2]);
    layout = std::atoi(argv[3]);
    owner = std::atoi(argv[4]);
  }
  unsetenv("MOM_C30_TEST_EDGE");
  unsetenv("MOM_C30_PROFILE");
  try {
    if (copy_case)
      run_copy_fault();
    else
      run();
    std::puts("PASS actual-source C30 asynchronous completion boundary");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
