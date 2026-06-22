// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <string>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "lib.h"

// SYCL builds, all DPC++: oneAPI icx for Intel GPU / Windows (default), the intel/llvm
// nightly clang CUDA backend for NVIDIA (-Dmom_sycl_impl=dpcpp-cuda), and the combined
// build that AOTs both spir64 + nvptx in one mom.node (scripts/build-combined.dockerfile).
// They share almost all code; the NVIDIA-specific spots are gated three ways:
//   * device code           -> per device-compilation pass via the compiler's __NVPTX__
//   * host code that must be COMPILED for CUDA capability -> MOM_SYCL_HAS_CUDA (set by
//     binding.gyp for the dpcpp-cuda and dpcpp-combined modes)
//   * host runtime decisions -> mom_is_cuda(device) below (the actual device backend)

// The cooperative ProgPoW / Ethash / cn-gpu kernels run on 16-wide sub-groups on Intel
// GPUs, requested via reqd_sub_group_size(16). NVIDIA warps are fixed at 32 lanes (no
// 16-wide sub-group), so the nvptx device pass drops the attribute and lets the native
// 32-wide warp stand; those kernels address each cooperative team relative to its base
// lane within the sub-group, correct at both 16 and 32 lanes. Gating on the per-pass
// __NVPTX__ (not a build macro) lets the combined build emit sg16 in its spir64 image and
// no requirement in its nvptx image, so each device loads the image that fits it.
#if defined(__NVPTX__)
  #define MOM_REQD_SG_16
#else
  #define MOM_REQD_SG_16 [[sycl::reqd_sub_group_size(16)]]
#endif

// Bind the context's executable kernel bundle to each handler as a build-cache hint.
#define MOM_BUNDLE_T sycl::kernel_bundle<sycl::bundle_state::executable>
#define MOM_GET_EXEC_BUNDLE(ctx) \
  sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx)
#define MOM_USE_BUNDLE(handler, kb) (handler).use_kernel_bundle(kb)

// Thin wrappers kept for call-site readability; both DPC++ builds use the SYCL builtins.
template <typename T> inline T mo_rotate(const T x, const T n) {
  return sycl::rotate(x, n);
}
template <typename T> inline T mo_bitselect(const T a, const T b, const T c) {
  return sycl::bitselect(a, b, c);
}
// offset is in units of N elements (matching the SYCL vec load/store contract).
template <typename VecT, typename T>
inline void mo_vec_load(VecT& v, const size_t offset, const T* const p) {
  v.load(offset, p);
}
template <typename VecT, typename T>
inline void mo_vec_store(const VecT& v, const size_t offset, T* const p) {
  v.store(offset, p);
}

inline void set_sycl_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

// Parse a base-10 unsigned long, requiring the variable to be present, non-empty, and fully numeric.
inline bool mom_parse_env_ulong(const char* const name, unsigned long& out) {
  const char* const value = std::getenv(name);
  if (!value || !*value) return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno || end == value || *end) return false;
  out = parsed;
  return true;
}

inline bool sycl_is_level_zero_gpu(const sycl::device& device) {
  return
    device.is_gpu() &&
    device.get_platform().get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos;
}

// Runtime test for the DPC++ CUDA backend. The enum exists in every DPC++ sycl header, so this
// compiles in the Intel-only build too (where it simply never matches). Use this for host-side
// per-device decisions in the combined build, instead of the old build-wide MOM_SYCL_CUDA macro.
inline bool mom_is_cuda(const sycl::device& device) {
  return device.get_backend() == sycl::backend::ext_oneapi_cuda;
}

inline unsigned sycl_default_workgroup(
  const sycl::device& device, const std::initializer_list<unsigned> allowed, const unsigned preferred
) {
  const size_t reported_max = device.get_info<sycl::info::device::max_work_group_size>();
  // Clamp the reported limit into [1, UINT_MAX] before comparing against unsigned candidates.
  const unsigned max_workgroup = std::max<unsigned>(1u, static_cast<unsigned>(
    std::min<size_t>(reported_max, std::numeric_limits<unsigned>::max())));
  unsigned selected = 0;
  for (const unsigned candidate : allowed) {
    if (candidate <= preferred && candidate <= max_workgroup) selected = std::max(selected, candidate);
  }
  return selected ? selected : *std::min_element(allowed.begin(), allowed.end());
}

// Branch-free modulo by a runtime divisor via multiply-shift (Granlund-Montgomery).
// Shared by the kawpow/etchash/autolykos2 kernels. Layout must stay byte-compatible
// with the FastModData mirror in kawpow_jit.inc.
struct FastModData { uint32_t reciprocal, increment, shift, divisor; };

inline uint32_t clz32_host(const uint32_t value) {
#if defined(_MSC_VER)
  unsigned long index;
  _BitScanReverse(&index, value);
  return 31U - static_cast<uint32_t>(index);
#else
  return static_cast<uint32_t>(__builtin_clz(value));
#endif
}

inline FastModData make_fast_mod_data(const uint32_t divisor) {
  FastModData data{};  // increment defaults to 0
  data.divisor = divisor;
  if ((divisor & (divisor - 1U)) == 0) {  // power of two: exact shift, reciprocal 1
    data.reciprocal = 1;
    data.shift = 31U - clz32_host(divisor);
  } else {
    data.shift = 63U - clz32_host(divisor);
    const uint64_t n = 1ULL << data.shift;
    const uint64_t q = n / divisor;
    const uint64_t r = n - q * divisor;
    // Round the reciprocal up unless the remainder lets us round down with increment=1.
    if (r * 2 < divisor) {
      data.reciprocal = static_cast<uint32_t>(q);
      data.increment = 1;
    } else {
      data.reciprocal = static_cast<uint32_t>(q + 1);
    }
  }
  return data;
}

inline uint32_t fast_mod_dev(const uint32_t a, const FastModData d) {
  const uint64_t t = a;
  const uint32_t q = static_cast<uint32_t>(((t + d.increment) * d.reciprocal) >> d.shift);
  return a - q * d.divisor;
}

inline void sycl_wait_and_throw(sycl::event event, const sycl::device& device) {
  // Several GPU backends busy-spin a host core inside native event waits. Polling the event status
  // with a short sleep keeps GPU mining from pinning one CPU thread while preserving exact completion.
  // CPU devices keep the native wait because their "kernel" work is host work and should not be hidden.
  const bool poll_wait = device.is_gpu();
  if (!poll_wait) {
    event.wait_and_throw();
    return;
  }
  while (event.get_info<sycl::info::event::command_execution_status>() !=
         sycl::info::event_command_status::complete) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  event.wait_and_throw();
}

sycl::device get_dev(const std::string& dev_str);
