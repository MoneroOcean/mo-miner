// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__)

#include <cuda.h>
#include <nvrtc.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <string>

namespace mom {

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

  CudaDriverApi() {
#if defined(_WIN32)
    library_ = LoadLibraryA("nvcuda.dll");
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

class NvrtcApi {
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
    error = std::string("missing NVRTC symbol ") + name;
    return false;
  }

  NvrtcApi() {
#if defined(_WIN32)
    const char* const names[] = {
      "nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_121_0.dll", "nvrtc64_122_0.dll"
    };
    for (const char* const name : names) {
      if ((library_ = LoadLibraryA(name)))
        break;
    }
#else
    const char* const names[] = {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"};
    for (const char* const name : names) {
      if ((library_ = dlopen(name, RTLD_NOW | RTLD_LOCAL)))
        break;
    }
#endif
    if (!library_) {
      error = "NVRTC is not installed";
      return;
    }
    const bool ptx =
      load_symbol(create_program, "nvrtcCreateProgram") &&
      load_symbol(compile_program, "nvrtcCompileProgram") &&
      load_symbol(get_log_size, "nvrtcGetProgramLogSize") &&
      load_symbol(get_log, "nvrtcGetProgramLog") &&
      load_symbol(get_ptx_size, "nvrtcGetPTXSize") &&
      load_symbol(get_ptx, "nvrtcGetPTX") &&
      load_symbol(destroy_program, "nvrtcDestroyProgram") &&
      load_symbol(error_string, "nvrtcGetErrorString");
    if (ptx) {
      (void)(load_symbol(get_cubin_size, "nvrtcGetCUBINSize") &&
             load_symbol(get_cubin, "nvrtcGetCUBIN") &&
             load_symbol(get_num_supported_archs, "nvrtcGetNumSupportedArchs") &&
             load_symbol(get_supported_archs, "nvrtcGetSupportedArchs"));
    }
  }

public:
  NvrtcApi(const NvrtcApi&) = delete;
  NvrtcApi& operator=(const NvrtcApi&) = delete;

  decltype(&nvrtcCreateProgram) create_program = nullptr;
  decltype(&nvrtcCompileProgram) compile_program = nullptr;
  decltype(&nvrtcGetProgramLogSize) get_log_size = nullptr;
  decltype(&nvrtcGetProgramLog) get_log = nullptr;
  decltype(&nvrtcGetPTXSize) get_ptx_size = nullptr;
  decltype(&nvrtcGetPTX) get_ptx = nullptr;
  decltype(&nvrtcGetCUBINSize) get_cubin_size = nullptr;
  decltype(&nvrtcGetCUBIN) get_cubin = nullptr;
  decltype(&nvrtcGetNumSupportedArchs) get_num_supported_archs = nullptr;
  decltype(&nvrtcGetSupportedArchs) get_supported_archs = nullptr;
  decltype(&nvrtcDestroyProgram) destroy_program = nullptr;
  decltype(&nvrtcGetErrorString) error_string = nullptr;
  std::string error;

  bool ptx_available() const {
    return library_ && create_program && compile_program && get_log_size && get_log &&
           get_ptx_size && get_ptx && destroy_program && error_string;
  }

  bool cubin_available() const {
    return ptx_available() && get_cubin_size && get_cubin && get_num_supported_archs &&
           get_supported_archs;
  }

  static NvrtcApi& instance() {
    // NVRTC stays loaded for the same process lifetime as cached native modules.
    static NvrtcApi api;
    return api;
  }
};

// Own only the temporary compiler program; release it before loading the resulting module.
class NvrtcProgram {
  NvrtcApi& api_;
  nvrtcProgram program_ = nullptr;

public:
  // Construct after ptx_available()/cubin_available() validates the compiler entry points.
  explicit NvrtcProgram(NvrtcApi& api) noexcept : api_(api) {}
  NvrtcProgram(const NvrtcProgram&) = delete;
  NvrtcProgram& operator=(const NvrtcProgram&) = delete;

  nvrtcProgram* address() noexcept { return &program_; }
  nvrtcProgram get() const noexcept { return program_; }

  void reset() noexcept {
    if (!program_)
      return;
    nvrtcProgram program = program_;
    program_ = nullptr;
    (void)api_.destroy_program(&program);
  }

  ~NvrtcProgram() {
    reset();
  }
};

inline void cuda_check(CudaDriverApi& api, const CUresult status,
                       const char* const operation) {
  if (status == CUDA_SUCCESS)
    return;
  const char* detail = nullptr;
  if (api.error_string)
    (void)api.error_string(status, &detail);
  throw std::string(operation) + ": " + (detail ? detail : "CUDA driver error");
}

inline std::string rtc_error(NvrtcApi& api, const nvrtcResult status, nvrtcProgram program,
                             const char* const operation) {
  size_t size = 0;
  std::string log;
  if (program && api.get_log_size(program, &size) == NVRTC_SUCCESS && size) {
    log.resize(size);
    (void)api.get_log(program, log.data());
  }
  return std::string(operation) + ": " + api.error_string(status) +
         (log.empty() ? "" : "\n" + log);
}

} // namespace mom

#endif
