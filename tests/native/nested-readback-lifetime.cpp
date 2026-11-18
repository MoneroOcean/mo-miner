// Host-only fixture for source-extracted nested SYCL readbacks.
// The compiler-policy test writes entry.inc from production code and builds this file twice:
// once with the local MomSyclHostTransferGuard and once without it.  MOM_NESTED_ROUTE=1 selects
// HooHash's Result owner; MOM_NESTED_ROUTE=2 selects BeamHash3's diagnostic vector owner.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

using std::size_t;
using std::uint32_t;
using std::uint8_t;

#ifndef MOM_NESTED_ROUTE
#error "MOM_NESTED_ROUTE must select the extracted HooHash or BeamHash3 readback"
#endif

struct Control {
  unsigned fault = 0; // 1: accepted submission throws; 2: completion throws.
  bool secondary = false;
  bool triggered = false;
  bool owner_alive = true;
  bool premature = false;
  unsigned copies = 0;
  unsigned event_waits = 0;
  unsigned drains = 0;
  void* pending_destination = nullptr;
  const void* pending_source = nullptr;
  std::size_t pending_bytes = 0;
  std::exception_ptr primary;

  void retire() {
    if (!pending_destination)
      return;
    if (!owner_alive)
      premature = true;
    std::memcpy(pending_destination, pending_source, pending_bytes);
    pending_destination = nullptr;
    pending_source = nullptr;
    pending_bytes = 0;
  }
};
static Control control;

namespace sycl {
struct device {
  bool is_gpu() const {
    return false;
  }
};

struct event {
  void wait_and_throw() const {
    ++control.event_waits;
    if (control.fault == 2 && !control.triggered) {
      control.triggered = true;
      std::rethrow_exception(control.primary);
    }
    control.retire();
  }
};

struct queue {
  event memcpy(void* const destination, const void* const source, const std::size_t bytes) {
    if (control.pending_destination)
      std::exit(91);
    ++control.copies;
    control.pending_destination = destination;
    control.pending_source = source;
    control.pending_bytes = bytes;
    if (control.fault == 1)
      std::rethrow_exception(control.primary);
    return {};
  }

  void wait_and_throw() {
    ++control.drains;
    control.retire();
    if (control.secondary && std::uncaught_exceptions())
      throw std::runtime_error("secondary cleanup fault");
  }
};
} // namespace sycl

// The production helper waits the accepted event and is intentionally reduced to the same
// throwing boundary here; the production guard itself is source-extracted into cleanup.inc.
inline void sycl_wait_and_throw(sycl::event event, const sycl::device&) {
  event.wait_and_throw();
}

#include "cleanup.inc"

// HooHash's source Result is deliberately given a destructor in this fixture so a missing local
// guard is detected by ownership state, not by stack poisoning or sanitizer heuristics.
struct Result {
  std::uint32_t offset = 0;
  std::uint8_t hash[32]{};

  ~Result() {
    if (control.pending_destination == &offset)
      control.owner_alive = false;
  }
};

// The test replaces only BeamHash3's std::vector type with this tracked owner.
template <typename T> class TrackedVector {
  std::unique_ptr<T[]> storage_;
  std::size_t size_;

public:
  explicit TrackedVector(const std::size_t size)
      : storage_(std::make_unique<T[]>(size)), size_(size) {}
  ~TrackedVector() {
    if (control.pending_destination == storage_.get())
      control.owner_alive = false;
  }
  T* data() {
    return storage_.get();
  }
  const T* data() const {
    return storage_.get();
  }
  std::size_t size() const {
    return size_;
  }
};

struct HooState {
  sycl::queue queue;
  sycl::device device;
  std::uint32_t result_storage = 0;
  std::uint32_t* result = &result_storage;
};

constexpr unsigned NBUCKETS = 8;
constexpr unsigned CUR = 0;
struct BeamState {
  sycl::device device;
  std::uint32_t bucket_storage[2][NBUCKETS]{};
  std::uint32_t* bucket_ns[2] = {bucket_storage[0], bucket_storage[1]};
};

static void owner_scope() {
#if MOM_NESTED_ROUTE == 1
  HooState state;
#elif MOM_NESTED_ROUTE == 2
  sycl::queue q;
  BeamState state;
  const bool cuda_layout = false;
#else
#error "unknown MOM_NESTED_ROUTE"
#endif

#include "entry.inc"
}

static std::exception_ptr make_primary() {
  try {
    throw std::runtime_error("original nested readback fault");
  } catch (...) {
    return std::current_exception();
  }
}

static void require(const bool condition, const char* const message) {
  if (!condition)
    throw std::logic_error(message);
}

int main(int argc, char** argv) {
  if (argc != 3)
    return 2;
  control.fault = static_cast<unsigned>(std::atoi(argv[1]));
  control.secondary = std::atoi(argv[2]) != 0;
  control.primary = make_primary();

  std::exception_ptr caught;
  try {
    owner_scope();
  } catch (...) {
    caught = std::current_exception();
  }

  try {
    require(control.copies == 1, "nested readback retried or skipped its accepted copy");
    require(control.event_waits == (control.fault == 1 ? 0u : 1u),
            "nested readback did not respect its submission/completion boundary");
    require(control.fault ? caught == control.primary : !caught,
            "nested readback changed or swallowed the original exception");
    require(!control.premature, "nested readback retired after its local owner was destroyed");
    require(!control.pending_destination, "nested readback left an accepted copy pending");
    require(control.drains == (control.fault ? 1u : 0u),
            "healthy readback added a queue drain or fault skipped cleanup retirement");
    std::printf("PASS nested readback route=%d fault=%d secondary=%d waits=%u drains=%u\n",
                MOM_NESTED_ROUTE, control.fault, control.secondary, control.event_waits,
                control.drains);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
