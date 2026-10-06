#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

enum class Stage {
  success, submit, native_queue, interop_wait, prepare, prepare_wait, make_a,
  gemm, finish, completion, compiler_exception, queue_allocation, buffer_allocation,
  unavailable, hip_status, changed_points, non_hip
};

struct Counters {
  int interop = 0;
  int api = 0;
  int allocations = 0;
  int prepare = 0;
  int make_a = 0;
  int gemm = 0;
  int finish = 0;
  int completion = 0;
} counts;

static Stage failure_stage;
static int error_kind;
static const void* thrown_address;
static bool reached_return;
static bool checked_at_destroy;
static int device_at_destroy = -1;
static constexpr int selected_device = 7;

static void require(bool condition, const char* message) {
  if (!condition)
    throw std::logic_error(message);
}

static void inject(Stage stage) {
  if (stage == Stage::success || failure_stage != stage)
    return;
  // Check the original thrown object, not a reconstructed message or exception.
  if (error_kind == 1) {
    try {
      throw std::string("interop fault");
    } catch (const std::string& error) {
      thrown_address = &error;
      throw;
    }
  }
  if (error_kind == 2) {
    try {
      throw 37;
    } catch (const int& error) {
      thrown_address = &error;
      throw;
    }
  }
  try {
    throw std::runtime_error("interop fault");
  } catch (const std::exception& error) {
    thrown_address = &error;
    throw;
  }
}

using hipDevice_t = int;
using hipStream_t = void*;
using hipError_t = int;
constexpr hipError_t hipSuccess = 0;
constexpr hipError_t hipErrorUnknown = 999;
static hipError_t hipStreamGetDevice(hipStream_t, hipDevice_t* device) {
  ++counts.api;
  if (failure_stage == Stage::hip_status) {
    *device = 987654;
    return error_kind;
  }
  *device = selected_device;
  return hipSuccess;
}
static const char* hipGetErrorString(hipError_t status) {
  switch (status) {
    case 1: return "hipErrorInvalidValue";
    case 3: return "hipErrorNotInitialized";
    case 4: return "hipErrorDeinitialized";
    case 201: return "hipErrorInvalidContext";
    case 400: return "hipErrorInvalidHandle";
    case 709: return "hipErrorContextIsDestroyed";
    default: return "hipErrorUnknown";
  }
}
static hipError_t hipSetDevice(hipDevice_t) {
  return hipSuccess;
}

namespace sycl {
enum class backend {hip};
struct device { bool hip = true; };
struct context {};
namespace property {
namespace queue {
struct in_order {};
}
}
struct property_list {
  explicit property_list(property::queue::in_order) {}
};
struct interop_handle {
  template <backend>
  hipStream_t get_native_queue() {
    inject(Stage::native_queue);
    return reinterpret_cast<void*>(17);
  }
};
struct event {
  Stage completion_stage = Stage::success;
  void wait_and_throw() const {
    if (completion_stage == Stage::completion)
      ++counts.completion;
    inject(completion_stage);
  }
};
struct queue {
  bool hip = true;
  queue() = default;
  queue(context, device, property_list) {
    if (failure_stage == Stage::queue_allocation)
      throw std::bad_alloc();
  }
  template <typename Callback>
  event AdaptiveCpp_enqueue_custom_operation(Callback callback) {
    ++counts.interop;
    inject(Stage::submit);
    if (!hip)
      throw std::runtime_error("not a HIP queue");
    interop_handle handle;
    callback(handle);
    return {Stage::interop_wait};
  }
  context get_context() const { return {}; }
  device get_device() const { return {hip}; }
  void wait_and_throw() {}
};
template <typename T>
T* malloc_device(size_t, queue&) {
  if (failure_stage == Stage::buffer_allocation)
    throw std::bad_alloc();
  ++counts.allocations;
  static T storage{};
  return &storage;
}
template <typename T>
T* aligned_alloc_device(size_t, size_t count, queue& queue) {
  return malloc_device<T>(count, queue);
}
template <typename T>
void free(T*, queue&) {}
}

[[maybe_unused]] static bool mom_is_hip(sycl::device device) {
  return device.hip;
}

#include "identity.inc"

constexpr unsigned NativeSlots = 2;
constexpr unsigned NativeProducts = 6;
constexpr unsigned NativeM = 64;
constexpr unsigned NativeN = 1024;
constexpr unsigned NativeK = 1024;
constexpr unsigned NativeGroups = 2;
constexpr unsigned NativeSubgroup = 32;
using NativeProduct = int32_t;
struct FastModData {};
struct Result {};

struct OctopusAmdWmma {
  hipDevice_t device = -1;
  void* kernel = nullptr;
  unsigned rows = 0;
  bool checked = false;
#include "wmma-init.inc"
    inject(Stage::compiler_exception);
    if (failure_stage == Stage::unavailable)
      return false;
    kernel = &counts;
    return true;
  }
  ~OctopusAmdWmma() {
    checked_at_destroy = checked;
    device_at_destroy = device;
  }
};

#include "gemm-context.inc"
static void sycl_wait_and_throw(const sycl::event event, sycl::device) {
  event.wait_and_throw();
}
static sycl::event native_make_b(sycl::queue&, const uint32_t*, int8_t*) {
  ++counts.prepare;
  inject(Stage::prepare);
  return {Stage::prepare_wait};
}
static void native_make_a(sycl::queue&, const uint8_t*, uint64_t, int8_t*) {
  ++counts.make_a;
  inject(Stage::make_a);
}
static void native_gemm(sycl::queue&, NativeGemmContext&, const int8_t*, const int8_t*,
                        NativeProduct*) {
  ++counts.gemm;
  inject(Stage::gemm);
}
static sycl::event native_finish(sycl::queue&, const uint8_t*, const uint32_t*, FastModData,
                                 const NativeProduct*, uint64_t, unsigned, const uint8_t*,
                                 Result*, bool) {
  ++counts.finish;
  inject(Stage::finish);
  return {Stage::completion};
}

#include "octopus.inc"

int main(int argc, char** argv) {
  if (argc != 4)
    return 2;
  const int target = std::atoi(argv[1]);
  failure_stage = static_cast<Stage>(std::atoi(argv[2]));
  error_kind = std::atoi(argv[3]);
  sycl::queue queue;
  queue.hip = failure_stage != Stage::non_hip;
  hipDevice_t device = -1;
  hipStream_t stream = nullptr;
  bool returned = false;
  bool caught = false;
  bool same_exception = false;
  bool status_exception = false;
  try {
    if (target < 2) {
      if (target == 0)
        mom::hip_queue_identity(queue, device, stream);
      else
        mom::hip_queue_device(queue, device);
      returned = true;
    } else {
      OctopusSyclNativeSearch search;
      uint8_t header[32]{};
      uint32_t points[1024]{};
      Result result;
      returned = search.search(queue, header, nullptr, {}, points, 42, 5, 130, nullptr, &result, false);
      if (returned && (failure_stage == Stage::success || failure_stage == Stage::changed_points)) {
        const uint64_t version = failure_stage == Stage::changed_points ? 43 : 42;
        returned = search.search(queue, header, nullptr, {}, points, version, 5, 130, nullptr,
                                  &result, false);
      }
    }
    reached_return = true;
  } catch (const std::string& error) {
    caught = true;
    same_exception = error_kind == 1 && &error == thrown_address;
    status_exception = failure_stage == Stage::hip_status &&
        error == std::string("hipStreamGetDevice: ") + hipGetErrorString(error_kind);
  } catch (const std::exception& error) {
    caught = true;
    same_exception = error_kind == 0 && &error == thrown_address;
  } catch (const int& error) {
    caught = true;
    same_exception = error_kind == 2 && &error == thrown_address;
  }
  try {
    const bool allocation = failure_stage == Stage::queue_allocation ||
        failure_stage == Stage::buffer_allocation;
    const bool unavailable = failure_stage == Stage::unavailable || failure_stage == Stage::non_hip;
    const bool success = failure_stage == Stage::success || failure_stage == Stage::changed_points;
    const bool fault = !allocation && !unavailable && !success;
    require(caught == fault, "submitted/interop fault was converted to unavailable");
    if (fault) {
      require((failure_stage == Stage::hip_status ? status_exception : same_exception) &&
                  !reached_return,
              "fault identity/type/diagnostic changed or fallback resumed");
      if (target < 2)
        require(device == -1 && stream == nullptr, "failed identity committed partial outputs");
      if (target == 2 && (failure_stage == Stage::submit || failure_stage == Stage::native_queue ||
                         failure_stage == Stage::interop_wait || failure_stage == Stage::hip_status)) {
        require(!checked_at_destroy, "identity fault latched WMMA as checked");
        require(device_at_destroy == -1, "identity fault leaked a device into cleanup");
      }
    } else {
      require(reached_return && returned == success, "success/unavailability behavior changed");
    }
    if (target < 2) {
      require(counts.interop == 1, "identity query retried");
      if (success) {
        require(device == selected_device, "selected HIP device changed");
        if (target == 0)
          require(stream == reinterpret_cast<void*>(17), "selected HIP stream changed");
      }
    } else if (success) {
      require(counts.interop == 1 && counts.api == 1, "cached ensure repeated HIP interop");
      require(counts.allocations == 5, "cached pipeline buffers were reallocated");
      require(counts.prepare == (failure_stage == Stage::changed_points ? 2 : 1),
              "point-version preparation cache changed");
      require(counts.make_a == 6 && counts.gemm == 6 && counts.finish == 6 &&
                  counts.completion == 4,
              "successful two-slot pipeline changed");
    } else {
      require(counts.interop == (failure_stage == Stage::non_hip ? 0 : 1),
              "non-HIP fallback queried HIP or a failed attempt retried");
      if (allocation || unavailable)
        require(counts.prepare == 0 && counts.make_a == 0 && counts.finish == 0,
                "startup unavailability submitted search work");
    }
    std::puts("PASS actual-source HIP interop/Octopus fault boundary");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
