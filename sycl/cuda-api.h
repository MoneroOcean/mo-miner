// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#if (defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)) && \
    !defined(__SYCL_DEVICE_ONLY__)

#include <cuda.h>
#if defined(MOM_EMBEDDED_CUDA_IMAGES)
#include "cuda-images.h"
#else
#include "nvrtc-api.h"
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "../runtime/system-driver.h"
#else
#include <dlfcn.h>
#endif

#include <string>

namespace mom {

// Process-lifetime driver entrypoints; optional acceleration must check availability before use.
class CudaDriverApi {
#if defined(_WIN32)
  HMODULE library_ = nullptr;
#else
  void* library_ = nullptr;
#endif

  template <typename T>
  bool load_symbol(T& output, const char* const name) {
#if defined(_WIN32)
    output = reinterpret_cast<T>(GetProcAddress(library_, name));
#else
    output = reinterpret_cast<T>(dlsym(library_, name));
#endif
    if (output)
      return true;
    error = std::string("missing CUDA driver symbol ") + name;
    return false;
  }

  // Load required calls first, then optional extended calls without losing basic driver support.
  CudaDriverApi() {
#if defined(_WIN32)
    // Reuse the native loader: a system driver must never come from CWD or PATH.
    library_ = runtime::systemDriverLibrary<runtime::SystemDriver::cuda>();
#else
    library_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
    if (!library_) {
      error = "the CUDA driver library is not installed";
      return;
    }
    const bool basic =
      load_symbol(init, "cuInit") &&
      load_symbol(device_attribute, "cuDeviceGetAttribute") &&
      load_symbol(retain_primary, "cuDevicePrimaryCtxRetain") &&
      load_symbol(release_primary, "cuDevicePrimaryCtxRelease") &&
      load_symbol(set_current, "cuCtxSetCurrent") &&
      load_symbol(module_load, "cuModuleLoadData") &&
      load_symbol(module_unload, "cuModuleUnload") &&
      load_symbol(module_function, "cuModuleGetFunction") &&
      load_symbol(launch, "cuLaunchKernel") &&
      load_symbol(event_create, "cuEventCreate") &&
      load_symbol(event_destroy, "cuEventDestroy") &&
      load_symbol(event_record, "cuEventRecord") &&
      load_symbol(event_query, "cuEventQuery") &&
      load_symbol(error_string, "cuGetErrorString");
    if (basic) {
      (void)(load_symbol(synchronize, "cuCtxSynchronize") &&
             load_symbol(function_attribute, "cuFuncSetAttribute") &&
             load_symbol(event_elapsed, "cuEventElapsedTime"));
    }
  }

public:
  CudaDriverApi(const CudaDriverApi&) = delete;
  CudaDriverApi& operator=(const CudaDriverApi&) = delete;

  decltype(&cuInit) init = nullptr;
  decltype(&cuDeviceGetAttribute) device_attribute = nullptr;
  decltype(&cuDevicePrimaryCtxRetain) retain_primary = nullptr;
  decltype(&cuDevicePrimaryCtxRelease) release_primary = nullptr;
  decltype(&cuCtxSetCurrent) set_current = nullptr;
  decltype(&cuCtxSynchronize) synchronize = nullptr;
  decltype(&cuModuleLoadData) module_load = nullptr;
  decltype(&cuModuleUnload) module_unload = nullptr;
  decltype(&cuModuleGetFunction) module_function = nullptr;
  decltype(&cuFuncSetAttribute) function_attribute = nullptr;
  decltype(&cuLaunchKernel) launch = nullptr;
  decltype(&cuEventCreate) event_create = nullptr;
  decltype(&cuEventDestroy) event_destroy = nullptr;
  decltype(&cuEventRecord) event_record = nullptr;
  decltype(&cuEventQuery) event_query = nullptr;
  decltype(&cuEventElapsedTime) event_elapsed = nullptr;
  decltype(&cuGetErrorString) error_string = nullptr;
  std::string error;

  bool basic_available() const {
    return library_ && init && device_attribute && retain_primary && release_primary &&
           set_current && module_load && module_unload && module_function && launch &&
           event_create && event_destroy && event_record && event_query && error_string;
  }

  bool extended_available() const {
    return basic_available() && synchronize && function_attribute && event_elapsed;
  }

  static CudaDriverApi& instance() {
    // The process owns the compiler/driver libraries. Do not unload them underneath live SYCL state.
    static CudaDriverApi api;
    return api;
  }
};

// Preserve the failing native operation in a recoverable host exception.
inline void cuda_check(CudaDriverApi& api, const CUresult status,
                       const char* const operation) {
  if (status == CUDA_SUCCESS)
    return;
  const char* detail = nullptr;
  if (api.error_string)
    (void)api.error_string(status, &detail);
  throw std::string(operation) + ": " + (detail ? detail : "CUDA driver error");
}

#if defined(MOM_EMBEDDED_CUDA_IMAGES)
// Retry only image/target compatibility failures. Resource, context, argument, and driver-JIT
// failures are operational errors and must not be hidden by trying a different PTX image.
inline bool cuda_image_retryable(const CUresult status) {
  return status == CUDA_ERROR_INVALID_IMAGE || status == CUDA_ERROR_NO_BINARY_FOR_GPU ||
         status == CUDA_ERROR_INVALID_PTX || status == CUDA_ERROR_UNSUPPORTED_PTX_VERSION;
}

template <typename Loader>
CUresult cuda_load_image_candidates(const cuda_images::Candidates& candidates, CUmodule* module,
                                    Loader&& loader) {
  if (!module)
    return CUDA_ERROR_INVALID_VALUE;
  *module = nullptr;
  const cuda_images::Blob* const images[] = {&candidates.primary, &candidates.compatibility};
  CUresult last_status = CUDA_ERROR_INVALID_IMAGE;
  bool attempted = false;
  for (const cuda_images::Blob* const image : images) {
    if (!image->data || !image->size)
      continue;
    attempted = true;
    last_status = loader(module, image->data);
    if (last_status == CUDA_SUCCESS)
      return last_status;
    *module = nullptr;
    if (!cuda_image_retryable(last_status))
      return last_status;
  }
  return attempted ? last_status : CUDA_ERROR_INVALID_IMAGE;
}
#endif

// Manually launched work is invisible to SYCL queue drains. Retire it on an error before callers
// can reuse/free its buffers; the cleanup status must not replace the original driver failure.
// Native callers require synchronize before submitting work. Successful calls never synchronize.
inline void cuda_check_submitted(CudaDriverApi& api, const CUresult status,
                                 const char* const operation) {
  if (status != CUDA_SUCCESS)
    (void)api.synchronize();
  cuda_check(api, status, operation);
}

} // namespace mom

#endif
