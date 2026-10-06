#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char* message) {
  if (!condition)
    throw std::logic_error(message);
}

struct PendingCopy {
  void* destination;
  const void* source;
  size_t bytes;
};
static std::vector<PendingCopy> pending_copies;
static std::exception_ptr injected;
static unsigned copies, submits, queries, waits, completed;
#if !defined(MOM_MATRIX_FIXTURE) && !defined(MOM_NESTED_FIXTURE)
static unsigned premature_retirements, retirements;
#endif
#if defined(MOM_MATRIX_FIXTURE)
static unsigned preparations;
#endif
#if defined(MOM_ENTRY_FIXTURE) && MOM_ENTRY_OCTOPUS
static unsigned preparations;
#endif
static int stage, kind;
static bool cleanup_error;

namespace sycl {
struct exception : std::runtime_error { using std::runtime_error::runtime_error; };
struct event {};
struct queue {
  int get_device() const { return 0; }
  event memcpy(void* destination, const void* source, size_t bytes) {
    ++copies;
#if !defined(MOM_MATRIX_FIXTURE) && !defined(MOM_NESTED_FIXTURE)
    if (stage == 4 && copies == 2)
      std::rethrow_exception(injected);
#endif
#if defined(MOM_MATRIX_FIXTURE)
    if (stage == 2 && copies == (MOM_MATRIX_HOO ? 2u : 1u))
      std::rethrow_exception(injected);
#endif
#if defined(MOM_NESTED_FIXTURE)
    if (stage && copies == MOM_NESTED_FAIL_COPY)
      std::rethrow_exception(injected);
#endif
    pending_copies.push_back({destination, source, bytes});
    return {};
  }
  void query() {
    ++queries;
    if (stage == 3)
      std::rethrow_exception(injected);
  }
  void submit() {
    ++submits;
    if ((stage == 1 && submits == 1) || (stage == 2 && submits == 2))
      std::rethrow_exception(injected);
  }
  event memset(void* destination, int value, size_t bytes) {
    submit();
    std::memset(destination, value, bytes);
    return {};
  }
  void wait_and_throw() {
    ++waits;
    for (const PendingCopy& copy : pending_copies) {
      std::memcpy(copy.destination, copy.source, copy.bytes);
      ++completed;
    }
    pending_copies.clear();
    if (cleanup_error && std::uncaught_exceptions())
      throw std::string("secondary cleanup fault");
  }
};
}

#include "cleanup.inc"

#if !defined(MOM_MATRIX_FIXTURE) && !defined(MOM_NESTED_FIXTURE)
struct HostOwner {
  std::unique_ptr<uint8_t[]> storage = std::make_unique<uint8_t[]>(64);
  HostOwner() { std::memset(storage.get(), 0xa5, 64); }
  ~HostOwner() {
    const auto begin = reinterpret_cast<uintptr_t>(storage.get());
    for (const PendingCopy& copy : pending_copies)
      if (const auto source = reinterpret_cast<uintptr_t>(copy.source);
          source >= begin && source < begin + 64)
        ++premature_retirements;
    ++retirements;
  }
};

struct BufferedInput {
  void write(sycl::queue& queue, const uint8_t* source, size_t bytes) {
    queue.memcpy(storage, source, bytes);
  }
  uint8_t storage[64]{};
};
struct Result { uint8_t contents[8]; };
using EtchashResult = Result;
using KawpowResult = Result;
using FishResult = Result;
static constexpr unsigned HASH_LEN = 32;
struct State {
  sycl::queue queue;
  bool shared_io = false;
  BufferedInput buffered_inputs;
  uint8_t input[64]{}, target[32]{};
  uint8_t* header = input;
  uint8_t extranonce[8]{};
  Result result_storage;
  Result* result = &result_storage;
#if defined(MOM_ENTRY_FIXTURE) && MOM_ENTRY_OCTOPUS
  bool points_ready = true, target_ready = true;
  std::array<uint8_t, HASH_LEN> points_header{}, target_copy{};
  bool ensure_points(const uint8_t next_header[HASH_LEN]) {
#include "points-cache-hit.inc"
    ++preparations;
#include "points-cache-commit.inc"
    return true;
  }
#endif
};

#if defined(MOM_ENTRY_FIXTURE)
#if !MOM_ENTRY_OCTOPUS
#include "search-result.inc"
#endif
static void dispatch_entry(State& state, const uint8_t* input, const uint8_t* target) {
  sycl::queue& queue = state.queue;
#if !MOM_ENTRY_OCTOPUS
  (void) input;
  uint8_t* target_ = state.target;
  SearchResult result_storage{};
  SearchResult* result_ = &result_storage;
#endif
#include "entry.inc"
  // Kernel math is unchanged; exercise synchronous faults following the real accepted uploads.
  queue.query();
  queue.submit();
  queue.submit();
  queue.wait_and_throw();
#if MOM_ENTRY_OCTOPUS
#include "entry-catch.inc"
#endif
}
#else

template <bool Portable>
static void dispatch(State& state, const uint8_t* input, const uint8_t* target) {
  [[maybe_unused]] constexpr bool mom_sycl_portable_opencl = Portable;
  [[maybe_unused]] sycl::queue& q = state.queue;
  [[maybe_unused]] const uint8_t* const inputs = input;
  [[maybe_unused]] uint8_t* const d_inputs = state.input;
  [[maybe_unused]] const size_t input_bytes = 40;
  [[maybe_unused]] const unsigned input_size = 40;
  [[maybe_unused]] constexpr bool is_test = false;
  (void) target;
#if MOM_HOST_TRANSFER_WITH_GUARD
#include "upload.after.inc"
#else
#include "upload.before.inc"
#endif
  // Device kernels are mocked; their first and later synchronous submission failures must
  // unwind through the actual entry upload scope, not a duplicate cleanup implementation.
  q.query();
  q.submit();
  q.submit();
  q.wait_and_throw();
}
#endif
#elif defined(MOM_MATRIX_FIXTURE)
struct FloatPair { float hi, lo; };
struct MatrixState {
  sycl::queue queue;
  int device = 0;
  bool matrix_ready = true;
  uint8_t matrix_seed[32]{};
#if MOM_MATRIX_HOO
  double canonical_matrix[4096]{};
  double matrix[4096]{};
  FloatPair normal[4096 * 16]{};
#else
  uint8_t matrix[4096]{};
#endif
};
template <typename T>
static void make_matrix(const uint8_t* input, T* matrix) {
  ++preparations;
  for (unsigned i = 0; i < 4096; ++i)
    matrix[i] = static_cast<T>(input[0]);
  if (stage == 1)
    std::rethrow_exception(injected);
}
static MatrixState* active_matrix;
static void sycl_wait_and_throw(sycl::event, int) {
  active_matrix->queue.wait_and_throw();
  if (stage == 3)
    std::rethrow_exception(injected);
}
static void update_matrix(MatrixState& state, const uint8_t* input) {
#include "matrix.inc"
}
#else
enum class SessionStatus { complete, device_error };
struct RunReport {
  SessionStatus status = SessionStatus::complete;
  std::string error;
  uint32_t zero_root_count = 3;
  std::array<uint32_t, 5> flat_counts{};
};
static unsigned converted;
struct Lifecycle {
  void device_error() { ++converted; }
};
static sycl::queue* active_nested_queue;
static void sycl_wait_and_throw(sycl::event, int) {
  active_nested_queue->wait_and_throw();
}
struct NestedSession {
  sycl::queue queue_;
  Lifecycle lifecycle_;
  std::string last_error_;
  struct Spec { static constexpr unsigned proof_indices = 2; };
  static constexpr unsigned fast_candidate_roots = 2;
  struct Options { bool collect_round_counts = true; } options_;
  uint32_t device_counts[6]{}, device_leaves[6]{};
  uint8_t device_valid[3]{};
  uint32_t* flat_counts_ = device_counts;
  uint32_t* result_counts_storage_ = device_counts;
  uint32_t* recovered_leaves_ = device_leaves;
  uint8_t* recovered_valid_ = device_valid;
  std::array<uint32_t, fast_candidate_roots * Spec::proof_indices> fast_leaves{};
  std::array<uint8_t, fast_candidate_roots> fast_valid{};
  RunReport run() {
    RunReport report;
    try {
#include "nested.inc"
      return report;
#include "converted-catch.inc"
#include "error-report.inc"
};
#endif

static std::exception_ptr make_error() {
  try {
    if (kind == 0)
      throw std::runtime_error("original host-transfer fault");
    if (kind == 1)
      throw std::string("original host-transfer fault");
    throw 37;
  } catch (...) {
    return std::current_exception();
  }
}

#if defined(MOM_MATRIX_FIXTURE)
int main(int argc, char** argv) {
  if (argc != 4)
    return 2;
  stage = std::atoi(argv[1]);
  kind = std::atoi(argv[2]);
  cleanup_error = std::atoi(argv[3]) != 0;
  injected = make_error();
  auto owner = std::make_unique<MatrixState>();
  MatrixState& state = *owner;
  active_matrix = &state;
  uint8_t old_header[32]{}, changed_header[32];
  std::memset(changed_header, 7, sizeof(changed_header));
  std::exception_ptr caught;
  try {
    try {
      update_matrix(state, changed_header);
    } catch (...) {
      caught = std::current_exception();
    }
    require(stage ? caught == injected : !caught, "matrix error was swallowed/replaced");
    require(pending_copies.empty(), "matrix failure left a local-owner upload pending");
    if (stage) {
      require(!state.matrix_ready, "failed changed header kept the previous matrix ready");
      require(std::memcmp(state.matrix_seed, old_header, 32) == 0,
              "failed changed header committed its seed");
    } else {
      require(state.matrix_ready && std::memcmp(state.matrix_seed, changed_header, 32) == 0,
              "completed changed header did not commit readiness and seed");
      require(waits == 1, "completed matrix added a queue wait");
    }
    const unsigned before_preparations = preparations, before_copies = copies;
    stage = 0;
    update_matrix(state, old_header);
    require(preparations == before_preparations + 1 &&
            copies == before_copies + (MOM_MATRIX_HOO ? 2u : 1u),
            "old-header retry reused a matrix invalidated by the failed changed header");
    require(state.matrix_ready && std::memcmp(state.matrix_seed, old_header, 32) == 0 &&
            state.matrix[0] == 0, "old-header retry did not regenerate and commit its matrix");
    const unsigned cached_preparations = preparations, cached_copies = copies, cached_waits = waits;
    update_matrix(state, old_header);
    require(preparations == cached_preparations && copies == cached_copies && waits == cached_waits,
            "successful matrix cache hit added preparation, copies or waits");
    std::puts("PASS actual-source matrix readiness and retry");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
#elif defined(MOM_NESTED_FIXTURE)
int main(int argc, char** argv) {
  if (argc != 4)
    return 2;
  stage = std::atoi(argv[1]);
  kind = std::atoi(argv[2]);
  cleanup_error = std::atoi(argv[3]) != 0;
  injected = make_error();
  NestedSession state;
  active_nested_queue = &state.queue_;
  RunReport report;
  std::exception_ptr caught;
  try {
    try {
      report = state.run();
    } catch (...) {
      caught = std::current_exception();
    }
    require(pending_copies.empty(), "nested readback left its try-local destination pending");
    if (stage && kind == 0) {
      require(!caught && converted == 1 && report.status == SessionStatus::device_error &&
              report.error == "original host-transfer fault",
              "actual session catch changed the converted primary error");
    } else {
      require(stage ? caught == injected : !caught, "nested primary exception was swallowed/replaced");
      require(converted == 0, "session converted a non-standard exception or healthy result");
    }
    require(copies == (stage ? MOM_NESTED_FAIL_COPY : MOM_NESTED_COPIES),
            "nested fault retried a readback or fallback");
    require(waits == 1, "nested unwind skipped retirement or success added a queue wait");
    std::puts("PASS actual-source nested readback and converted status");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
#elif defined(MOM_ENTRY_FIXTURE)
int main(int argc, char** argv) {
  if (argc != 5)
    return 2;
  stage = std::atoi(argv[1]);
  kind = std::atoi(argv[2]);
  cleanup_error = std::atoi(argv[3]) != 0;
  const bool shared = std::atoi(argv[4]) != 0;
  injected = make_error();
  State state;
  state.shared_io = shared;
  std::exception_ptr caught;
  try {
    {
      HostOwner input, target;
      try {
        dispatch_entry(state, input.storage.get(), target.storage.get());
      } catch (...) {
        caught = std::current_exception();
      }
    }
    require(stage ? caught == injected : !caught, "entry primary exception was swallowed/replaced");
    require(!premature_retirements, "accepted entry transfer outlived its host owner");
    require(pending_copies.empty(), "entry returned with an accepted transfer pending");
    require(waits == 1, "entry success added a queue wait or unwind skipped retirement");
#if MOM_ENTRY_OCTOPUS
    require(copies == (shared ? 0u : 2u) && completed == (shared ? 0u : stage == 4 ? 1u : 2u),
            "Octopus retried or skipped its entry uploads");
    require(queries == (stage == 4 ? 0u : 1u) &&
            submits == (stage == 4 || stage == 3 ? 0u : stage == 1 ? 1u : 2u),
            "Octopus retried a query, kernel or fallback");
    if (stage)
      require(!state.points_ready && !state.target_ready,
              "Octopus fault kept header or target cache ready");
    else
      require(state.points_ready && state.target_ready, "Octopus success did not commit caches");
    const unsigned before_copies = copies, before_preparations = preparations;
    stage = 0;
    cleanup_error = false;
    {
      HostOwner input, target;
      dispatch_entry(state, input.storage.get(), target.storage.get());
      require(copies == before_copies + (caught && !shared ? 2u : 0u) &&
              preparations == before_preparations + (caught ? 1u : 0u),
              "Octopus same-header retry skipped invalidated uploads or point preparation");
      require(state.points_ready && state.target_ready && state.header[0] == 0xa5 &&
              state.target[0] == 0xa5, "Octopus retry did not commit the original header and target");
      const unsigned cached_copies = copies, cached_preparations = preparations;
      dispatch_entry(state, input.storage.get(), target.storage.get());
      require(copies == cached_copies && preparations == cached_preparations && waits == 3,
              "Octopus healthy cache hit added uploads, preparation or queue waits");
    }
#else
    require(copies == 1 && completed == 1 && queries == (stage == 1 ? 0u : 1u) &&
            submits == (stage == 1 || stage == 3 ? 1u : stage == 2 ? 2u : 3u),
            "Nexa staged entry retried or skipped a copy, query or submission");
#endif
    std::puts("PASS actual-source entry transfers and cache readiness");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
#else
int main(int argc, char** argv) {
  if (argc != 5)
    return 2;
  stage = std::atoi(argv[1]);
  kind = std::atoi(argv[2]);
  cleanup_error = std::atoi(argv[3]) != 0;
  const int mode = std::atoi(argv[4]);
  injected = make_error();
  State state;
  state.shared_io = mode == 1;
  std::exception_ptr caught;
  try {
    {
      HostOwner input, target;
      try {
        if (mode == 2)
          dispatch<true>(state, input.storage.get(), target.storage.get());
        else
          dispatch<false>(state, input.storage.get(), target.storage.get());
      } catch (...) {
        caught = std::current_exception();
      }
    }
    require(stage ? caught == injected : !caught, "original exception was swallowed/replaced");
    require(submits == (stage == 3 || stage == 4 ? 0u : stage == 1 ? 1u : 2u),
            "fault retried a kernel or selected fallback");
    require(queries == (stage == 4 ? 0u : 1u), "dispatch retried after query failure");
    require(copies == (mode == 1 ? 0u : stage == 4 ? 2u : MOM_HOST_TRANSFER_UPLOADS),
            "host uploads were retried or skipped");
    require(!premature_retirements, "accepted transfer outlived its host owner");
    require(pending_copies.empty(), "dispatch returned with an accepted transfer pending");
    require(waits == 1, "success added a queue wait or unwind skipped retirement");
    require(completed == copies - (stage == 4 ? 1u : 0u) && retirements == 2,
            "host owners were not retired after copies");
    std::puts("PASS actual-source host-transfer lifetime");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
#endif
