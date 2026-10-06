// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <limits>

namespace mom::nvidia {

// Zero means unknown: never infer an optional instruction from a failed device query.
constexpr int compute_capability(const int major, const int minor) {
  return major > 0 && minor >= 0 && minor <= 9 &&
                 major <= (std::numeric_limits<int>::max() - minor) / 10
             ? major * 10 + minor
             : 0;
}

constexpr bool has_dp4a(const int sm) {
  return sm >= 61;
}

// Pearl/Octopus need both m16n8k32 int8 MMA and cp.async, not merely any Tensor Core.
constexpr bool has_int8_async_matrix(const int sm) {
  return sm >= 80;
}

} // namespace mom::nvidia

#if defined(MOM_SYCL_HAS_CUDA)
#include <sycl/sycl.hpp>
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(MOM_SYCL_ADAPTIVECPP)
#include "cuda-api.h"
#endif

namespace mom::nvidia {

inline int compute_capability(const sycl::device& device) noexcept {
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(MOM_SYCL_ADAPTIVECPP)
  try {
    if (device.get_backend() != sycl::backend::ext_oneapi_cuda)
      return 0;
    CudaDriverApi& driver = CudaDriverApi::instance();
    if (!driver.init || !driver.device_attribute || driver.init(0) != CUDA_SUCCESS)
      return 0;
    // Query the selected SYCL device, not ordinal zero or a launcher-supplied capability hint.
    const CUdevice native = sycl::get_native<sycl::backend::ext_oneapi_cuda>(device);
    int major = 0, minor = 0;
    if (driver.device_attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, native) !=
            CUDA_SUCCESS ||
        driver.device_attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, native) !=
            CUDA_SUCCESS)
      return 0;
    return compute_capability(major, minor);
  } catch (...) {
    return 0;
  }
#else
  (void)device;
  return 0;
#endif
}

} // namespace mom::nvidia
#endif
