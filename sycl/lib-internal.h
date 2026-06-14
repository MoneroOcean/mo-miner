// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
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

// SYCL backend detection. The NVIDIA build compiles these sources with
// AdaptiveCpp (acpp), which predefines __ACPP__ (older names: __HIPSYCL__ /
// __OPENSYCL__). The Intel GPU / Windows builds use oneAPI DPC++, which defines
// none of them. All NVIDIA-specific specialization is gated on MOMINER_ACPP so
// the DPC++ build compiles the identical text with this macro undefined.
#if defined(__ACPP__) || defined(__HIPSYCL__) || defined(__OPENSYCL__)
  #define MOMINER_ACPP 1
#endif

// The cooperative ProgPoW / Ethash / cn-gpu kernels run on 16-wide sub-groups on
// Intel GPUs, requested via reqd_sub_group_size(16). NVIDIA warps are fixed at 32
// lanes and AdaptiveCpp does not honor a 16-wide request, so on the NVIDIA build
// we drop the attribute and let the native 32-wide sub-group stand; those kernels
// address each cooperative team relative to its base lane within the sub-group,
// which is correct at both 16 and 32 lanes.
#if defined(MOMINER_ACPP)
  #define MOMINER_REQD_SG_16
#else
  #define MOMINER_REQD_SG_16 [[sycl::reqd_sub_group_size(16)]]
#endif

// AdaptiveCpp does not provide the SYCL-2020 kernel_bundle API. The Intel build
// fetches the context's executable bundle and binds it to each handler purely as
// a build-cache hint; on AdaptiveCpp the generic-SSCP JIT compiles kernels
// directly, so these become no-ops (the bundle handle is a dummy int).
#if defined(MOMINER_ACPP)
  #define MOMINER_BUNDLE_T int
  #define MOMINER_GET_EXEC_BUNDLE(ctx) 0
  #define MOMINER_USE_BUNDLE(handler, kb) ((void)(kb))
#else
  #define MOMINER_BUNDLE_T sycl::kernel_bundle<sycl::bundle_state::executable>
  #define MOMINER_GET_EXEC_BUNDLE(ctx) \
    sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx)
  #define MOMINER_USE_BUNDLE(handler, kb) (handler).use_kernel_bundle(kb)
#endif

// sycl::rotate (left rotate) and sycl::bitselect are SYCL builtins that
// AdaptiveCpp does not expose. Forward to the builtins on DPC++ (unchanged Intel
// codegen) and provide portable bit-twiddle equivalents on AdaptiveCpp.
template <typename T> inline T mo_rotate(const T x, const T n) {
#if defined(MOMINER_ACPP)
  constexpr unsigned bits = sizeof(T) * 8u;
  const T s = n & static_cast<T>(bits - 1u);
  return s == 0 ? x : static_cast<T>((x << s) | (x >> (static_cast<T>(bits) - s)));
#else
  return sycl::rotate(x, n);
#endif
}
template <typename T> inline T mo_bitselect(const T a, const T b, const T c) {
#if defined(MOMINER_ACPP)
  return (a & ~c) | (b & c);
#else
  return sycl::bitselect(a, b, c);
#endif
}

// sycl::vec::load/store accept a raw pointer on DPC++ but require a multi_ptr on
// AdaptiveCpp; fall back to element-wise copy there. offset is in units of N
// elements (matching the SYCL load/store contract).
template <typename VecT, typename T>
inline void mo_vec_load(VecT& v, const size_t offset, const T* const p) {
#if defined(MOMINER_ACPP)
  constexpr unsigned N = sizeof(VecT) / sizeof(T);
  const T* const base = p + offset * N;
  for (unsigned i = 0; i < N; ++i) v[i] = base[i];
#else
  v.load(offset, p);
#endif
}
template <typename VecT, typename T>
inline void mo_vec_store(const VecT& v, const size_t offset, T* const p) {
#if defined(MOMINER_ACPP)
  constexpr unsigned N = sizeof(VecT) / sizeof(T);
  T* const base = p + offset * N;
  for (unsigned i = 0; i < N; ++i) base[i] = v[i];
#else
  v.store(offset, p);
#endif
}

inline void set_sycl_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

inline bool sycl_is_level_zero_gpu(const sycl::device& device) {
  return
    device.is_gpu() &&
    device.get_platform().get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos;
}

inline unsigned sycl_default_workgroup(
  const sycl::device& device, const std::initializer_list<unsigned> allowed, const unsigned preferred
) {
  const size_t reported_max = device.get_info<sycl::info::device::max_work_group_size>();
  const unsigned max_workgroup = reported_max > std::numeric_limits<unsigned>::max()
    ? std::numeric_limits<unsigned>::max()
    : std::max(1u, static_cast<unsigned>(reported_max));
  unsigned selected = 0;
  for (const unsigned candidate : allowed) {
    if (candidate <= preferred && candidate <= max_workgroup) selected = std::max(selected, candidate);
  }
  return selected ? selected : *std::min_element(allowed.begin(), allowed.end());
}

// Branch-free modulo by a runtime divisor via multiply-shift (Granlund-Montgomery).
// Shared by the kawpow/etchash/autolykos2 kernels.
struct FastModData {
  uint32_t reciprocal;
  uint32_t increment;
  uint32_t shift;
  uint32_t divisor;
};

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
  FastModData data{};
  data.divisor = divisor;
  if ((divisor & (divisor - 1U)) == 0) {
    data.reciprocal = 1;
    data.increment = 0;
    data.shift = 31U - clz32_host(divisor);
  } else {
    data.shift = 63U - clz32_host(divisor);
    const uint64_t n = 1ULL << data.shift;
    const uint64_t q = n / divisor;
    const uint64_t r = n - q * divisor;
    if (r * 2 < divisor) {
      data.reciprocal = static_cast<uint32_t>(q);
      data.increment = 1;
    } else {
      data.reciprocal = static_cast<uint32_t>(q + 1);
      data.increment = 0;
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
  if (!sycl_is_level_zero_gpu(device)) {
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
