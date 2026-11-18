// Host-only source-derived check for the NexaPoW monolithic submission boundary.
// The compiler-policy test supplies the production Result and entry scope below; this file only
// provides a tiny in-order queue that records command order and retires accepted host transfers.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

static void require(const bool condition, const char* const message) {
  if (!condition)
    throw std::logic_error(message);
}

enum EventKind { clear_event, search_event, copy_event };

enum Fault { healthy, submit_fault, search_wait_fault, readback_wait_fault };

struct PendingCopy {
  void* destination;
  const void* source;
  std::size_t bytes;
};

struct Control {
  Fault fault = healthy;
  std::string order;
  unsigned copies = 0;
  unsigned submits = 0;
  unsigned waits = 0;
  unsigned premature_retirements = 0;
};

static Control control;
static std::vector<PendingCopy> pending_copies;
static std::exception_ptr primary_error;

static void drain_copies() {
  for (const PendingCopy& copy : pending_copies)
    std::memcpy(copy.destination, copy.source, copy.bytes);
  pending_copies.clear();
}

static void event_wait(const int kind) {
  ++control.waits;
  control.order.push_back('W');
  if ((kind == search_event && control.fault == search_wait_fault) ||
      (kind == copy_event && control.fault == readback_wait_fault))
    std::rethrow_exception(primary_error);
  drain_copies();
}

namespace sycl {

struct device {};

template <int> struct range {
  explicit range(const std::size_t) {}
};

template <int> struct id {
  std::size_t value;
  std::size_t operator[](const int) const { return value; }
};

struct event {
  int kind;
  void wait_and_throw() const { event_wait(kind); }
};

struct handler {
  template <typename Kernel, typename Function>
  void parallel_for(const range<1>, Function function) {
    function(id<1>{0});
  }
};

enum class memory_order { relaxed };
enum class memory_scope { device };
namespace access {
enum class address_space { global_space };
}

template <typename T, memory_order, memory_scope, access::address_space> struct atomic_ref {
  T& value;
  explicit atomic_ref(T& value) : value(value) {}
  T fetch_add(const T amount) {
    const T old = value;
    value += amount;
    return old;
  }
};

struct queue {
  device get_device() const { return {}; }

  event memcpy(void* const destination, const void* const source, const std::size_t bytes) {
    ++control.copies;
    control.order.push_back('C');
    pending_copies.push_back({destination, source, bytes});
    return {copy_event};
  }

  event memset(void* const destination, const int value, const std::size_t bytes) {
    control.order.push_back('M');
    std::memset(destination, value, bytes);
    return {clear_event};
  }

  template <typename Function> event submit(Function function) {
    ++control.submits;
    control.order.push_back('S');
    if (control.fault == submit_fault)
      std::rethrow_exception(primary_error);
    handler h;
    function(h);
    return {search_event};
  }

  void wait_and_throw() {
    ++control.waits;
    control.order.push_back('W');
    drain_copies();
  }
};

} // namespace sycl

#include "cleanup.inc"

static void sycl_wait_and_throw(const sycl::event event, const sycl::device&) {
  event.wait_and_throw();
}

#include "result.inc"

class SearchKernel;

#ifndef MOM_SYCL_KERNEL_ARGS_RESTRICT
#define MOM_SYCL_KERNEL_ARGS_RESTRICT
#endif

static bool np_hash_one(const uint8_t*, const uint8_t*, const uint64_t, uint8_t hash[32],
                        const unsigned) {
  for (unsigned i = 0; i < 32; ++i)
    hash[i] = static_cast<uint8_t>(0x5a + i);
  return true;
}

static bool np_meets_target(const uint8_t*, const uint8_t*) {
  return true;
}

struct State {
  sycl::device device;
  sycl::queue queue;
  uint8_t header[32]{};
  uint8_t extranonce[8]{};
  uint8_t target[32]{};
  Result result_storage{};
  Result* result = &result_storage;

  void init_portable() {}
};

static int run_entry(State& state, const uint8_t* const input, uint8_t* const output,
                     uint64_t* const pnonce, const uint8_t* const target) {
  const uint64_t first = *pnonce;
  constexpr unsigned count = 1;
  constexpr unsigned extra_bytes = 8;
  constexpr bool is_test = false;
#include "entry.inc"
}

template <std::size_t Size> struct HostOwner {
  std::array<uint8_t, Size> bytes{};

  ~HostOwner() {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(bytes.data());
    const uintptr_t end = begin + Size;
    for (const PendingCopy& copy : pending_copies) {
      const uintptr_t source = reinterpret_cast<uintptr_t>(copy.source);
      if (source >= begin && source < end)
        ++control.premature_retirements;
    }
  }
};

static std::exception_ptr make_primary_error() {
  try {
    throw std::runtime_error("NexaPoW submission fault");
  } catch (...) {
    return std::current_exception();
  }
}

static void reset(const Fault fault) {
  control = {};
  control.fault = fault;
  pending_copies.clear();
  primary_error = make_primary_error();
}

static void run_case(const Fault fault, const char* const expected_order,
                     const unsigned expected_waits) {
  reset(fault);
  State state;
  uint8_t output[32]{};
  uint64_t nonce = 7;
  std::exception_ptr caught;
  int result = -1;
  {
    HostOwner<40> input;
    HostOwner<32> target;
    std::fill(input.bytes.begin(), input.bytes.end(), 0x11);
    std::fill(target.bytes.begin(), target.bytes.end(), 0xff);
    try {
      result = run_entry(state, input.bytes.data(), output, &nonce, target.bytes.data());
    } catch (...) {
      caught = std::current_exception();
    }
    require(pending_copies.empty(), "host-transfer guard left an accepted copy pending");
  }
  require(control.premature_retirements == 0,
          "host-transfer guard retired a caller-owned buffer too late");
  require(control.order == expected_order, "NexaPoW command order changed");
  require(control.waits == expected_waits, "NexaPoW added or skipped a required wait");
  require((fault == healthy) == !caught, "NexaPoW fault was swallowed or invented");
  if (fault == healthy) {
    require(result == 1 && nonce == 7, "healthy NexaPoW result changed");
    for (unsigned i = 0; i < 32; ++i)
      require(output[i] == static_cast<uint8_t>(0x5a + i),
              "healthy NexaPoW readback changed");
    require(control.copies == 4 && control.submits == 1,
            "healthy NexaPoW submission/readback count changed");
  } else {
    require(caught == primary_error, "NexaPoW replaced the original submission fault");
    require(control.submits == 1, "NexaPoW retried a failed submission");
  }
}

int main() {
  try {
    // Three uploads, memset, search, search completion, result readback, readback completion.
    run_case(healthy, "CCCMSWCW", 2);
    // A failed search submission is retired by the production host-transfer guard only.
    run_case(submit_fault, "CCCMSW", 1);
    // A post-submit device fault is preserved while the guard drains accepted uploads.
    run_case(search_wait_fault, "CCCMSWW", 2);
    // The final result-copy wait remains part of the production readback contract.
    run_case(readback_wait_fault, "CCCMSWCWW", 3);
    std::puts("PASS actual-source NexaPoW monolithic submission order");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
