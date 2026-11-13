// Build-only device IR fixture. Compile the maintained primitive without changing/duplicating it.
// CUDA attributes mark this test wrapper; production algorithm sources remain SYCL.
#include <cstdint>
#pragma clang force_cuda_host_device begin
namespace device_hash {
#include "../../sycl/blake3-device.inc"
}
#pragma clang force_cuda_host_device end

extern "C" __attribute__((global)) void mom_device_blake3(std::uint8_t *out,
    const std::uint8_t *input, unsigned length) {
  device_hash::blake3_dev(out, 32, input, length);
}
