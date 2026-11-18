// Compile-only SSCP input: the maintained SYCL primitive follows the real generic compiler path.
#include <SYCL/sycl.hpp>
#include <cstdint>
namespace device_hash {
#include "../../sycl/blake3-device.inc"
}

void submit_native_primitive(sycl::queue &queue, std::uint8_t *out, const std::uint8_t *input,
                             unsigned length, float *mathOut, const float *mathIn) {
  queue.parallel_for<class NativeBlake3>(sycl::range<1>(4), [=](sycl::id<1> index) {
    device_hash::blake3_dev(out + index[0] * 32, 32, input, length);
  });
  // Exercise real device-library math and the strict/fast-math policies, not just integer IR.
  queue.parallel_for<class NativeMath>(sycl::range<1>(4), [=](sycl::id<1> index) {
    const float value = mathIn[index[0]];
    mathOut[index[0]] = sycl::sqrt(value) + sycl::sin(value) / value;
  });
}
