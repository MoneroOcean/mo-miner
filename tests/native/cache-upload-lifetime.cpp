#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MOM_CACHE_GUARDED
#error "MOM_CACHE_GUARDED must select the production guard or its negative control"
#endif
#ifndef MOM_CACHE_HAS_SHARED
#define MOM_CACHE_HAS_SHARED 1
#endif
#ifndef MOM_CACHE_WALA_TILED
#define MOM_CACHE_WALA_TILED 0
#endif

static bool fail_submit = false;
static bool cleanup_error = false;
static bool source_retired = false;
static bool invalid_lifetime = false;
static bool pending = false;
static unsigned copies = 0;
static unsigned drains = 0;
static void* pending_destination = nullptr;
static const void* pending_source = nullptr;
static std::size_t pending_bytes = 0;
static std::exception_ptr primary;

namespace sycl {
struct device {};
struct event {};

struct queue {
  event memcpy(void* const destination, const void* const source, const std::size_t bytes) {
    ++copies;
    pending = true;
    pending_destination = destination;
    pending_source = source;
    pending_bytes = bytes;
    // Model the backend boundary under investigation: the transfer is accepted, then submission
    // reports an error before returning its event to the caller.
    if (fail_submit)
      std::rethrow_exception(primary);
    return {};
  }

  void wait_and_throw() {
    ++drains;
    if (pending) {
      if (source_retired) {
        // Do not perform an undefined read from a source whose local owner has already died.
        invalid_lifetime = true;
      } else {
        std::memcpy(pending_destination, pending_source, pending_bytes);
      }
      pending = false;
      pending_destination = nullptr;
      pending_source = nullptr;
      pending_bytes = 0;
    }
    // Cleanup faults must never replace the original transfer exception.
    if (cleanup_error && std::uncaught_exceptions())
      throw std::runtime_error("secondary cache-upload cleanup fault");
  }
};
} // namespace sycl

static sycl::queue* active_queue = nullptr;

#include "mom_env_flag.inc"  // the production on/off switch parser (sycl/lib-internal.h)
#include "cleanup.inc"

static void require(const bool condition, const char* const message) {
  if (!condition)
    throw std::logic_error(message);
}

static std::exception_ptr make_primary(const int kind) {
  try {
    if (kind == 0)
      throw std::runtime_error("original cache-upload fault");
    if (kind == 1)
      throw std::string("original cache-upload fault");
    throw 37;
  } catch (...) {
    return std::current_exception();
  }
}

static std::array<std::uint8_t, 4096> wala_matrix;
static std::array<std::uint64_t, 32> fishhash_cache;
static std::array<std::uint32_t, 32> octopus_points;

struct WalaState {
  sycl::queue& queue;
  sycl::device& device;
  std::uint8_t* matrix;
};

[[maybe_unused]] static void make_matrix(const std::uint8_t* const input,
                                          std::uint8_t* const matrix) {
  std::fill_n(matrix, 4096, input[0]);
}

static void sycl_wait_and_throw(const sycl::event, const sycl::device&) {
  require(active_queue != nullptr, "cache-upload wait had no active queue");
  active_queue->wait_and_throw();
}

// The registration test supplies this include from the real guard-plus-copy source span. The
// surrounding declarations deliberately keep every source owner local to this invocation.
static void run_scope(sycl::queue& queue, const bool shared) {
  sycl::device device;
  active_queue = &queue;

  [[maybe_unused]] bool shared_io = shared;
  [[maybe_unused]] bool cpu_data = shared;
  [[maybe_unused]] bool points_ready = true;
  [[maybe_unused]] const bool intel_fast = MOM_CACHE_WALA_TILED != 0;
  [[maybe_unused]] std::uint8_t input[80]{};
  [[maybe_unused]] std::uint8_t matrix[4096]{};
  [[maybe_unused]] std::vector<std::uint32_t> host_cache(16, 0x11223344u);
  [[maybe_unused]] std::vector<std::uint64_t> hcache(16, 0x1122334455667788ULL);
  [[maybe_unused]] std::vector<std::uint64_t> host(16, 0x8899aabbccddeeffULL);
  [[maybe_unused]] const std::size_t data_bytes = host.size() * sizeof(std::uint64_t);
  [[maybe_unused]] void* data = fishhash_cache.data();
  [[maybe_unused]] std::array<std::uint32_t, 16> host_points{};
  [[maybe_unused]] const std::size_t bytes = host_cache.size() * sizeof(std::uint32_t);
  [[maybe_unused]] const char* scope = "ethash cache upload";
  [[maybe_unused]] void* cache = fishhash_cache.data();
  [[maybe_unused]] std::uint32_t* points = octopus_points.data();
  [[maybe_unused]] WalaState state{queue, device, wala_matrix.data()};

  // Route setup is supplied by compiler-policy.js; only the following source-derived transfer
  // span is included here. The route setup aliases the destination expected by that span.
#include "cache_scope.inc"
}

struct CaseResult {
  bool caught = false;
  std::exception_ptr exception;
  bool pending_after = false;
  bool invalid = false;
  unsigned case_copies = 0;
  unsigned case_drains = 0;
};

static CaseResult run_case(const bool shared, const int kind, const bool secondary) {
  fail_submit = !shared && kind >= 0;
  cleanup_error = secondary;
  source_retired = false;
  invalid_lifetime = false;
  pending = false;
  pending_destination = nullptr;
  pending_source = nullptr;
  pending_bytes = 0;
  copies = 0;
  drains = 0;
  primary = make_primary(kind);

  sycl::queue queue;
  std::exception_ptr caught;
  try {
    run_scope(queue, shared);
  } catch (...) {
    caught = std::current_exception();
  }

  // All source owners in the extracted production span have now left scope. A negative control
  // still has an accepted copy; drain it only to make the lifetime failure deterministic.
  source_retired = true;
  if (pending)
    queue.wait_and_throw();

  return {caught != nullptr, caught, pending, invalid_lifetime, copies, drains};
}

static void check_case(const CaseResult& result, const bool expected_fault, const bool shared,
                       const bool expected_hazard, const int expected_kind) {
  require(result.caught == expected_fault, "cache-upload primary exception was swallowed");
  require(expected_fault ? result.exception == primary : !result.exception,
          "cache-upload exception identity was lost");
  if (expected_fault) {
    try {
      std::rethrow_exception(result.exception);
    } catch (const std::runtime_error& error) {
      require(expected_kind == 0 && std::string(error.what()) == "original cache-upload fault",
              "cache-upload std::exception was replaced");
    } catch (const std::string& error) {
      require(expected_kind == 1 && error == "original cache-upload fault",
              "cache-upload std::string exception was replaced");
    } catch (const int error) {
      require(expected_kind == 2 && error == 37, "cache-upload unknown exception was replaced");
    } catch (...) {
      require(false, "cache-upload exception type changed");
    }
  }
  require(!result.pending_after, "cache-upload left an accepted copy pending");
  require(result.invalid == expected_hazard, "cache-upload lifetime hazard expectation changed");
  const unsigned expected_copies = shared ? 0u : 1u;
  require(result.case_copies == expected_copies, "cache-upload copy count changed");
  if (shared || !expected_fault)
    require(result.case_drains == (shared ? 0u : 1u),
            "healthy cache-upload path added or skipped a queue drain");
  else
    require(result.case_drains >= 1u && result.case_drains <= 2u,
            "cache-upload fault skipped or excessively repeated cleanup");
}

int main() {
  try {
    // A successful device copy must be retired by the production wait exactly once. The guard is
    // unwind-only and must not add a second healthy drain.
    check_case(run_case(false, -1, false), false, false, false, -1);

    for (const int kind : {0, 1, 2}) {
      for (const bool secondary : {false, true}) {
        const CaseResult result = run_case(false, kind, secondary);
        const bool hazard = MOM_CACHE_GUARDED == 0;
        check_case(result, true, false, hazard, kind);
        if (MOM_CACHE_GUARDED != 0)
          require(result.caught && result.invalid == false,
                  "guarded cache-upload fault changed the original exception boundary");
      }
    }

#if MOM_CACHE_HAS_SHARED
    // Shared-USM branches perform a synchronous host copy and therefore need no queue drain.
    check_case(run_case(true, -1, false), false, true, false, -1);
#endif

    if (MOM_CACHE_GUARDED == 0) {
      std::fputs("FAIL accepted cache upload outlived its local source\n", stderr);
      return 1;
    }
    std::puts("PASS actual-source cache upload lifetime and exception preservation");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
