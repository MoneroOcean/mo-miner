#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>

#include "../lib-internal.h"

namespace mom_nexapow {

#define MOM_NEXAPOW_SHA_ONLY
#include "device.inc"
#undef MOM_NEXAPOW_SHA_ONLY

class TestShaKernel;

void nexapow_test_sha256d_49(sycl::queue& queue, const std::uint8_t input[49],
                             std::uint8_t output[32]) {
  std::uint8_t* device_input = nullptr;
  std::uint8_t* device_output = nullptr;
  auto release = [&](std::uint8_t*& pointer) {
    auto* allocation = pointer;
    pointer = nullptr; // A throwing free may already have retired it; never retry that pointer.
    if (allocation)
      sycl::free(allocation, queue);
  };
  try {
    device_input = sycl::malloc_device<std::uint8_t>(49, queue);
    device_output = sycl::malloc_device<std::uint8_t>(32, queue);
    if (!device_input || !device_output)
      throw std::runtime_error("NexaPoW test probe allocation failed");
    MomSyclHostTransferGuard host_transfers(queue, "nexapow test probe transfers");
    queue.memcpy(device_input, input, 49);
    queue.submit([&](sycl::handler& handler) {
      handler.single_task<TestShaKernel>([=] {
        np_sha256d_49(device_input, device_output);
      });
    });
    sycl_wait_and_throw(queue.memcpy(output, device_output, 32), queue.get_device());
    release(device_input);
    release(device_output);
  } catch (...) {
    sycl_cleanup_noexcept("nexapow test probe input free", [&] { release(device_input); });
    sycl_cleanup_noexcept("nexapow test probe output free", [&] { release(device_output); });
    throw;
  }
}

} // namespace mom_nexapow
