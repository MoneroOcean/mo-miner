// Native Windows CUDA/NVML uses the installed drivers with the official SDK's exact types.
#pragma once
#ifndef _WIN32
#error The Windows CUDA thunk bridge is Windows-only; Linux keeps its existing forwarding path.
#endif
#include "system-driver.h"
#define NVML_STATIC_IMPORT
#include <cuda.h>
#include <nvml.h>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace mom::runtime {
template <class Signature> struct WindowsCudaCall;

// Like WindowsHipCall, a generated import pointer retains the compiler's ABI without parsing
// an AST or restating parameter lists. Missing drivers/exports never claim successful compute.
template <class Result, class... Args> struct WindowsCudaCall<Result (*)(Args...)> {
  template <const char* Name> static Result call(Args... args) {
    constexpr auto driver = std::is_same_v<Result, CUresult> ? SystemDriver::cuda : SystemDriver::nvml;
    static const auto function = systemDriverSymbol<driver, Result (*)(Args...)>(Name);
    if (function) return function(args...);
    if constexpr (std::is_same_v<Result, CUresult>) {
      if constexpr (std::string_view(Name) == "cuGetErrorName" || std::string_view(Name) == "cuGetErrorString") {
        static_assert(sizeof...(Args) == 2);
        auto output = std::get<1>(std::forward_as_tuple(args...));
        if (output) *output = "CUDA driver unavailable";
      }
      return std::string_view(Name) == "cuInit" ? CUDA_ERROR_NO_DEVICE : CUDA_ERROR_NOT_INITIALIZED;
    } else if constexpr (std::is_same_v<Result, nvmlReturn_t>) return NVML_ERROR_UNINITIALIZED;
    else if constexpr (std::is_same_v<Result, const char*>) return "NVIDIA management driver unavailable";
    else static_assert(std::is_same_v<Result, CUresult>, "Review the missing-driver fallback for this NVIDIA result type");
  }
};
} // namespace mom::runtime
