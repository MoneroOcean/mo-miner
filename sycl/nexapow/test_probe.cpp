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
  std::uint8_t* device_input = sycl::malloc_device<std::uint8_t>(49, queue);
  std::uint8_t* device_output = sycl::malloc_device<std::uint8_t>(32, queue);
  if (!device_input || !device_output) {
    if (device_input)
      sycl::free(device_input, queue);
    if (device_output)
      sycl::free(device_output, queue);
    throw std::runtime_error("NexaPoW test probe allocation failed");
  }
  try {
    queue.memcpy(device_input, input, 49);
    queue.submit([&](sycl::handler& handler) {
      handler.single_task<TestShaKernel>([=] {
        np_sha256d_49(device_input, device_output);
      });
    });
    sycl_wait_and_throw(queue.memcpy(output, device_output, 32), queue.get_device());
  } catch (...) {
    sycl::free(device_input, queue);
    sycl::free(device_output, queue);
    throw;
  }
  sycl::free(device_input, queue);
  sycl::free(device_output, queue);
}

} // namespace mom_nexapow
