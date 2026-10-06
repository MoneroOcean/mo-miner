// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include "amd-features.h"

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <sycl/sycl.hpp>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <exception>
#include <string>

namespace mom {

inline bool hip_queue_identity(sycl::queue& queue, hipDevice_t& device, hipStream_t& stream,
                               std::string* const reason = nullptr) {
  hipError_t status = hipErrorUnknown;
  try {
    sycl::event event = queue.AdaptiveCpp_enqueue_custom_operation(
        [&](sycl::interop_handle& handle) {
          stream = handle.get_native_queue<sycl::backend::hip>();
          status = hipStreamGetDevice(stream, &device);
        });
    event.wait_and_throw();
  } catch (const std::exception& error) {
    if (reason)
      *reason = std::string("HIP queue interop: ") + error.what();
    return false;
  } catch (...) {
    if (reason)
      *reason = "HIP queue interop failed";
    return false;
  }
  if (status == hipSuccess)
    return true;
  if (reason)
    *reason = std::string("hipStreamGetDevice: ") + hipGetErrorString(status);
  return false;
}

inline bool hip_queue_device(sycl::queue& queue, hipDevice_t& device) {
  hipStream_t stream = nullptr;
  return hip_queue_identity(queue, device, stream);
}

// HIPRTC is optional at runtime. Keeping its entry points behind one dynamic loader lets each
// source-JIT algorithm retain its portable SYCL fallback when the compiler library is unavailable.
class HiprtcApi {
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
    error = std::string("missing HIPRTC symbol ") + name;
    return false;
  }

  HiprtcApi() {
#if defined(_WIN32)
    const char* const names[] = {"hiprtc.dll",     "hiprtc0701.dll", "hiprtc0700.dll",
                                 "hiprtc0604.dll", "hiprtc0603.dll", "hiprtc0507.dll"};
    for (const char* const name : names)
      if ((library_ = LoadLibraryA(name)))
        break;
#else
    const char* const names[] = {"libhiprtc.so", "libhiprtc.so.7", "libhiprtc.so.6"};
    for (const char* const name : names)
      if ((library_ = dlopen(name, RTLD_NOW | RTLD_LOCAL)))
        break;
#endif
    if (!library_) {
      error = "HIPRTC library is not installed";
      return;
    }
    (void)(load_symbol(create_program, "hiprtcCreateProgram") &&
           load_symbol(compile_program, "hiprtcCompileProgram") &&
           load_symbol(get_log_size, "hiprtcGetProgramLogSize") &&
           load_symbol(get_log, "hiprtcGetProgramLog") &&
           load_symbol(get_code_size, "hiprtcGetCodeSize") &&
           load_symbol(get_code, "hiprtcGetCode") &&
           load_symbol(destroy_program, "hiprtcDestroyProgram") &&
           load_symbol(error_string, "hiprtcGetErrorString"));
  }

public:
  HiprtcApi(const HiprtcApi&) = delete;
  HiprtcApi& operator=(const HiprtcApi&) = delete;

  decltype(&hiprtcCreateProgram) create_program = nullptr;
  decltype(&hiprtcCompileProgram) compile_program = nullptr;
  decltype(&hiprtcGetProgramLogSize) get_log_size = nullptr;
  decltype(&hiprtcGetProgramLog) get_log = nullptr;
  decltype(&hiprtcGetCodeSize) get_code_size = nullptr;
  decltype(&hiprtcGetCode) get_code = nullptr;
  decltype(&hiprtcDestroyProgram) destroy_program = nullptr;
  decltype(&hiprtcGetErrorString) error_string = nullptr;
  std::string error;

  bool available() const {
    return library_ && create_program && compile_program && get_log_size && get_log &&
           get_code_size && get_code && destroy_program && error_string;
  }

  static HiprtcApi& instance() {
    static HiprtcApi api;
    return api;
  }
};

} // namespace mom
