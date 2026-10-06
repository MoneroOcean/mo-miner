#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct Control {
  bool fault = false, secondary = false, triggered = false;
  unsigned copies = 0, queries = 0, waits = 0, drains = 0;
  unsigned fault_query = ROUTE == 3 ? 2u : 1u;
  void* destination = nullptr;
  const void* source = nullptr;
  std::size_t bytes = 0;
  std::exception_ptr primary;

  void flush() {
    if (destination) {
      std::memcpy(destination, source, bytes);
      destination = nullptr;
    }
    if (secondary && triggered) throw std::runtime_error("secondary drain fault");
  }
};
static Control control;

namespace sycl {
namespace info {
enum class event_command_status { complete };
namespace event {
struct command_execution_status {};
}
}

struct device {
  bool is_gpu() const {
    return false;
  }
};

struct event {
  template <class T> info::event_command_status get_info() const {
    return info::event_command_status::complete;
  }
  void wait_and_throw() const {
    ++control.waits;
    // Monolithic Nexa already caches its device; fault its actual wait, not an invented query.
    if (ROUTE == 4 && control.fault && !control.triggered) {
      std::printf("WAIT order=copy-first copies=%u\n", control.copies);
      std::fflush(stdout);
      control.triggered = true;
      std::rethrow_exception(control.primary);
    }
    control.flush();
  }
};

struct queue {
  event memcpy(void* destination, const void* source, std::size_t bytes) {
    if (control.destination) std::exit(91);
    ++control.copies;
    control.destination = destination;
    control.source = source;
    control.bytes = bytes;
    return {};
  }
  device get_device() const {
    if (++control.queries == control.fault_query && control.fault && ROUTE != 4) {
      std::printf("QUERY order=%s copies=%u\n", control.destination ? "copy-first" : "query-first", control.copies);
      std::fflush(stdout);
      control.triggered = true;
      std::rethrow_exception(control.primary);
    }
    return {};
  }
  void wait_and_throw() {
    ++control.drains;
    control.flush();
  }
};
}

constexpr bool mom_sycl_portable_opencl = false;
bool mom_is_opencl(const sycl::device&) {
  return false;
}
#include "wait.inc"
#include "cleanup.inc"
#include "search-result.inc"
#include "monolithic-result.inc"
struct BucketArenaOverflow : std::runtime_error {
  explicit BucketArenaOverflow(const std::string& text) : std::runtime_error(text) {}
};
struct ActiveRound0 {
  static constexpr std::uint32_t bucket_slot_capacity = 1;
};
using ActiveRound1 = ActiveRound0;
using ActiveRound2 = ActiveRound0;
using ActiveRound3 = ActiveRound0;
using ActiveRound4 = ActiveRound0;
using ActiveRound5 = ActiveRound0;
using ActiveRound6 = ActiveRound0;
constexpr std::size_t count_stride = 4;

static void owner_scope() {
  sycl::queue queue;
  sycl::queue& queue_ = queue;
  SearchResult device_result{1, 0, 7};
  SearchResult* result_ = &device_result;
  // Monolithic Nexa uses a different result ABI and passes its cached device to the wait.
  struct {
    sycl::queue& queue;
    sycl::device device;
    Result result_storage{};
    Result* result = &result_storage;
  } state{queue, {}};
  std::uint32_t device_status[2]{ROUTE == 3 ? 1u : 0u, 0};
  std::uint32_t* overflow_ = device_status;
  std::uint32_t device_counts[count_stride]{1, 1, 1, 1};
  std::array<std::uint32_t*, 7> counts_{};
  counts_.fill(device_counts);
  // The production kernels are already complete at these readbacks. Only their unrelated
  // work is omitted; declarations, guards, copy expressions and overflow branch are actual.
#include "entry.inc"
}

int main(int argc, char** argv) {
  if (argc != 4) return 90;
  control.fault = std::atoi(argv[1]);
  const int kind = std::atoi(argv[2]);
  control.secondary = std::atoi(argv[3]);
  try {
    if (kind == 0) throw std::runtime_error("primary query fault");
    if (kind == 1) throw std::string("primary query fault");
    throw 73;
  } catch (...) {
    control.primary = std::current_exception();
  }
  bool original = false, overflow = false;
  try {
    owner_scope();
  } catch (const BucketArenaOverflow&) {
    overflow = true;
  } catch (...) {
    original = std::current_exception() == control.primary;
  }
  // A guard-free baseline may leave a copy accepted. ASan observes the genuine stack/heap
  // owner ending before this drain; no proxy owner or source-spelling assertion is involved.
  if (control.destination) {
    try {
      control.flush();
    } catch (...) {}
  }
  const bool okay = control.fault ? control.triggered && original : (ROUTE != 3 || overflow);
  std::printf("%s original=%d copies=%u waits=%u drains=%u pending=%d\n", okay ? "PASS" : "FAIL",
              original, control.copies, control.waits, control.drains, control.destination != nullptr);
  return okay && !control.destination ? 0 : 17;
}
