// Native Windows binds HIP to the installed AMD display driver, never a bundled runtime.
#pragma once
#ifndef _WIN32
#error The HIP system-driver bridge is Windows-only; Linux HIP remains statically linked.
#endif
#include "system-driver.h"
#include <hip/hip_runtime_api.h>
#include <type_traits>

namespace mom::runtime {
template <class Signature> struct WindowsHipCall;

// Generated __imp_ bindings retain the official SDK signature. Nothing loads during static
// initialization: the first actual HIP call resolves only System32, then caches success/failure
// for process lifetime. Absent drivers/exports must never fabricate successful GPU operations.
template <class Result, class... Args> struct WindowsHipCall<Result (*)(Args...)> {
  template <const char* Name> static Result call(Args... args) {
    static const auto function = systemDriverSymbol<SystemDriver::hip, Result (*)(Args...)>(Name);
    if (function) return function(args...);
    if constexpr (std::is_same_v<Result, hipError_t>) return hipErrorNoDevice;
    else if constexpr (std::is_same_v<Result, const char*>) return "AMD HIP system driver is unavailable";
    else static_assert(std::is_same_v<Result, hipError_t>, "Review the missing-driver fallback for this HIP result type");
  }
};
} // namespace mom::runtime
