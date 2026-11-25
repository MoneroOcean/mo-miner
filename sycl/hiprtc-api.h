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

#include <string>

namespace mom {

// Resolve the queue's actual native device/stream and propagate any submitted interop failure.
inline void hip_queue_identity(sycl::queue& queue, hipDevice_t& device, hipStream_t& stream) {
  hipError_t status = hipErrorUnknown;
  hipDevice_t candidate_device = -1;
  hipStream_t candidate_stream = nullptr;
  // Submitted interop and native identity faults are not optional-backend unavailability.
  sycl::event event = queue.AdaptiveCpp_enqueue_custom_operation(
      [&](sycl::interop_handle& handle) {
        candidate_stream = handle.get_native_queue<sycl::backend::hip>();
        status = hipStreamGetDevice(candidate_stream, &candidate_device);
      });
  event.wait_and_throw();
  if (status != hipSuccess)
    throw std::string("hipStreamGetDevice: ") + hipGetErrorString(status);
  device = candidate_device;
  stream = candidate_stream;
}

inline void hip_queue_device(sycl::queue& queue, hipDevice_t& device) {
  hipStream_t stream = nullptr;
  hip_queue_identity(queue, device, stream);
}

// Query the actual HIP architecture after the caller has established that the queue is HIP.
// Queue-interop errors intentionally propagate to mining callers; optional discovery callers may
// catch them and retain the ordinary profile. Keep this query shared with the worker route so the
// native/default capability boundary cannot drift.
inline bool hip_gfx12_int8_wmma_supported(sycl::queue& queue,
                                          std::string* const reason = nullptr) {
  hipDevice_t device = -1;
  hipStream_t stream = nullptr;
  hipDeviceProp_t properties{};
  hip_queue_identity(queue, device, stream);
  const hipError_t properties_status = hipGetDeviceProperties(&properties, device);
  if (properties_status != hipSuccess) {
    if (reason)
      *reason = std::string("hipGetDeviceProperties: ") +
                hipGetErrorString(properties_status);
    return false;
  }
  if (mom::amd::has_gfx12_int8_wmma(properties.gcnArchName))
    return true;
  if (reason)
    *reason = std::string("gfx12 WMMA is unavailable on ") +
              (properties.gcnArchName[0] ? properties.gcnArchName : "an unknown architecture");
  return false;
}

// HIPRTC is optional in Engineering Mode; one compiler contract preserves each algorithm's
// portable fallback. Native builds bind the same entry points statically in every host TU.
class HiprtcApi {
  bool loaded_ = false;
#if !defined(MOM_STATIC_HIPRTC)
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
#endif

  // Bind once per process; incomplete Engineering installations stay unavailable to all callers.
  HiprtcApi() {
#if defined(MOM_STATIC_HIPRTC)
    // Native releases bind the same compiler contract to archives, including HIPRTC's built-in
    // header and static COMGR. Never search for or extract an application library in this mode.
    create_program = &hiprtcCreateProgram;
    compile_program = &hiprtcCompileProgram;
    get_log_size = &hiprtcGetProgramLogSize;
    get_log = &hiprtcGetProgramLog;
    get_code_size = &hiprtcGetCodeSize;
    get_code = &hiprtcGetCode;
    destroy_program = &hiprtcDestroyProgram;
    error_string = &hiprtcGetErrorString;
    loaded_ = true;
#else
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
    loaded_ = true;
    (void)(load_symbol(create_program, "hiprtcCreateProgram") &&
           load_symbol(compile_program, "hiprtcCompileProgram") &&
           load_symbol(get_log_size, "hiprtcGetProgramLogSize") &&
           load_symbol(get_log, "hiprtcGetProgramLog") &&
           load_symbol(get_code_size, "hiprtcGetCodeSize") &&
           load_symbol(get_code, "hiprtcGetCode") &&
           load_symbol(destroy_program, "hiprtcDestroyProgram") &&
           load_symbol(error_string, "hiprtcGetErrorString"));
#endif
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
    return loaded_ && create_program && compile_program && get_log_size && get_log &&
           get_code_size && get_code && destroy_program && error_string;
  }

  static HiprtcApi& instance() {
    static HiprtcApi api;
    return api;
  }
};

// Own only the temporary compiler program; release it before loading the resulting module.
class HiprtcProgram {
  HiprtcApi& api_;
  hiprtcProgram program_ = nullptr;

public:
  // Construct after available() validates the compiler entry points.
  explicit HiprtcProgram(HiprtcApi& api) noexcept : api_(api) {}
  HiprtcProgram(const HiprtcProgram&) = delete;
  HiprtcProgram& operator=(const HiprtcProgram&) = delete;

  hiprtcProgram* address() noexcept { return &program_; }
  hiprtcProgram get() const noexcept { return program_; }

  // Relinquish ownership before the runtime call so error cleanup cannot destroy a handle twice.
  void reset() noexcept {
    if (!program_)
      return;
    hiprtcProgram program = program_;
    program_ = nullptr;
    (void)api_.destroy_program(&program);
  }

  ~HiprtcProgram() {
    reset();
  }
};

} // namespace mom
