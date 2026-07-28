// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "lib.h"

// Keep the build-system define at this boundary. Algorithm code should prefer this compile-time
// policy value and `if constexpr` over scattering preprocessor branches through kernels.
#if defined(MOM_SYCL_PORTABLE_OPENCL)
inline constexpr bool mom_sycl_portable_opencl = true;
#else
inline constexpr bool mom_sycl_portable_opencl = false;
#endif

#if defined(MOM_SYCL_ADAPTIVECPP)
inline constexpr bool mom_sycl_adaptivecpp = true;
#else
inline constexpr bool mom_sycl_adaptivecpp = false;
#endif

// Device storage used by the standards-only OpenCL profile. Native compiler artifacts retain USM
// pointers and their measured hot paths; the portable artifact specializes the same kernel source
// with buffer accessors because OpenCL implementations are not required to expose Intel's USM
// extension. Algorithms opt in one allocation at a time by asking for a device_view() inside the
// existing queue submission. This keeps allocation policy at compile time and adds no native hot-
// kernel branch.
template <typename T>
class MomBufferAllocation {
  std::unique_ptr<sycl::buffer<T, 1>> buffer_;
  size_t count_ = 0;

public:
  MomBufferAllocation() = default;
  MomBufferAllocation(const MomBufferAllocation&) = delete;
  MomBufferAllocation& operator=(const MomBufferAllocation&) = delete;

  void allocate(const size_t count) {
    if (buffer_ && count_ >= count) return;
    buffer_.reset();
    buffer_ = std::make_unique<sycl::buffer<T, 1>>(sycl::range<1>{count});
    count_ = count;
  }

  void release() { buffer_.reset(); count_ = 0; }

  template <sycl::access_mode Mode>
  auto device_view(sycl::handler& handler) {
    return sycl::accessor<T, 1, Mode, sycl::target::device>{*buffer_, handler};
  }

  sycl::event write(sycl::queue& queue, const T* const source, const size_t count) {
    return queue.submit([&](sycl::handler& handler) {
      auto view = sycl::accessor<T, 1, sycl::access_mode::write, sycl::target::device>{
        *buffer_, handler, sycl::range<1>{count}};
      handler.copy(source, view);
    });
  }

  sycl::event fill(sycl::queue& queue, const T& value) {
    return queue.submit([&](sycl::handler& handler) {
      auto view = device_view<sycl::access_mode::write>(handler);
      handler.fill(view, value);
    });
  }

  void read(T* const destination, const size_t count) {
    sycl::host_accessor view{*buffer_, sycl::read_only};
    std::copy_n(view.begin(), count, destination);
  }
};

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
#if defined(__NVPTX__) || defined(__AMDGCN__) || defined(MOM_SYCL_ADAPTIVECPP) || \
    defined(MOM_SYCL_PORTABLE_OPENCL)
  #define MOM_REQD_SG_16
#else
  #define MOM_REQD_SG_16 [[sycl::reqd_sub_group_size(16)]]
#endif

// Bind the context's executable kernel bundle to each handler as a build-cache hint.
#if defined(MOM_SYCL_ADAPTIVECPP) || defined(MOM_SYCL_PORTABLE_OPENCL)
struct MomKernelBundle {};
inline MomKernelBundle mom_get_exec_bundle(const sycl::context&) { return {}; }
inline void mom_use_bundle(sycl::handler&, MomKernelBundle&) {}
#else
using MomKernelBundle = sycl::kernel_bundle<sycl::bundle_state::executable>;
inline MomKernelBundle mom_get_exec_bundle(const sycl::context& context) {
  return sycl::get_kernel_bundle<sycl::bundle_state::executable>(context);
}
inline void mom_use_bundle(sycl::handler& handler, MomKernelBundle& bundle) {
  handler.use_kernel_bundle(bundle);
}
#endif

// Thin wrappers kept for call-site readability. Clang's rotate builtins are available in every
// compiler used here and lower to the native funnel-shift/rotate instruction. The portable profile
// deliberately keeps its volatile expression so the SPIR-V translator cannot recreate llvm.fshl.
template <typename T> inline T mo_rotate(const T x, const T n) {
  if constexpr (mom_sycl_portable_opencl) {
    constexpr T bits = sizeof(T) * 8;
    const T shift = n & (bits - 1);
    // A volatile intermediate prevents LLVM from recreating llvm.fshl, which is not part of the
    // portable OpenCL SPIR-V environment and is left unresolved by some DPC++ translator paths.
    volatile T left = static_cast<T>(x << shift);
    return shift ? static_cast<T>(left | (x >> (bits - shift))) : x;
  } else if constexpr (sizeof(T) == sizeof(uint32_t))
    return static_cast<T>(__builtin_rotateleft32(static_cast<uint32_t>(x),
                                                 static_cast<uint32_t>(n)));
  else
    return static_cast<T>(__builtin_rotateleft64(static_cast<uint64_t>(x),
                                                 static_cast<uint64_t>(n)));
}
template <typename T> inline T mo_bitselect(const T a, const T b, const T c) {
  return (a & ~c) | (b & c);
}
// offset is in units of N elements (matching the SYCL vec load/store contract).
template <typename VecT, typename T>
inline void mo_vec_load(VecT& v, const size_t offset, const T* const p) {
  if constexpr (mom_sycl_adaptivecpp)
    for (size_t i = 0; i < v.size(); ++i) v[i] = p[offset * v.size() + i];
  else
    v.load(offset, p);
}
template <typename VecT, typename T>
inline void mo_vec_store(const VecT& v, const size_t offset, T* const p) {
  if constexpr (mom_sycl_adaptivecpp)
    for (size_t i = 0; i < v.size(); ++i) p[offset * v.size() + i] = v[i];
  else
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

// Intel exposes versions such as 1.6.32224 (Level Zero) or 24.52.032224 (OpenCL). Return the
// largest numeric component, which is their common NEO build number; this also stays harmless for
// other vendors whose display-version fields are much smaller.
inline uint32_t mom_driver_build(const sycl::device& device) {
  const std::string version = device.get_info<sycl::info::device::driver_version>();
  uint32_t component = 0, largest = 0;
  for (const char c : version) {
    if (c >= '0' && c <= '9') {
      component = component > (UINT32_MAX - 9u) / 10u ? UINT32_MAX
                                                       : component * 10u + static_cast<uint32_t>(c - '0');
    } else {
      largest = std::max(largest, component);
      component = 0;
    }
  }
  return std::max(largest, component);
}

// Runtime test for the DPC++ CUDA backend. The enum exists in every DPC++ sycl header, so this
// compiles in the Intel-only build too (where it simply never matches). Use this for host-side
// per-device decisions in the combined build, instead of the old build-wide MOM_SYCL_CUDA macro.
inline bool mom_is_cuda(const sycl::device& device) {
#if defined(MOM_SYCL_ADAPTIVECPP)
  return device.get_backend() == sycl::backend::cuda;
#else
  return device.get_backend() == sycl::backend::ext_oneapi_cuda;
#endif
}

inline bool mom_is_hip(const sycl::device& device) {
#if defined(MOM_SYCL_ADAPTIVECPP)
  return device.get_backend() == sycl::backend::hip;
#elif defined(MOM_SYCL_HAS_HIP)
  return device.get_backend() == sycl::backend::ext_oneapi_hip;
#else
  (void)device;
  return false;
#endif
}

// The current Unified Runtime HIP adapter on Windows implements HIP device/managed allocations but
// does not advertise the corresponding SYCL USM aspects. Trust the backend capability there; Linux
// HIP already reports these aspects normally. Actual allocation failures still propagate from SYCL.
inline bool mom_has_usm_device(const sycl::device& device) {
  return mom_is_hip(device) || device.has(sycl::aspect::usm_device_allocations);
}

inline bool mom_has_usm_shared(const sycl::device& device) {
  return mom_is_hip(device) || device.has(sycl::aspect::usm_shared_allocations);
}

// The DPC++ AMD libclc built by --hip currently omits the SPIR-V subgroup shuffle entry points.
// gfx1200 has native wave32 DS permutes, so use them directly instead of leaving unresolved
// __spirv_GroupNonUniformShuffle{,Down} calls at device link. The lane arguments used by the miner
// are subgroup-relative and DPC++ maps one subgroup to one AMD wave.
inline uint32_t mom_select_from_group(const sycl::sub_group& group, uint32_t value, uint32_t lane) {
#if defined(__AMDGCN__) && !defined(MOM_SYCL_ADAPTIVECPP)
  (void)group;
  return __builtin_amdgcn_ds_bpermute(lane * 4U, value);
#else
  return sycl::select_from_group(group, value, lane);
#endif
}

inline uint64_t mom_select_from_group(const sycl::sub_group& group, uint64_t value, uint32_t lane) {
  const uint32_t lo = mom_select_from_group(group, static_cast<uint32_t>(value), lane);
  const uint32_t hi = mom_select_from_group(group, static_cast<uint32_t>(value >> 32), lane);
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

inline uint32_t mom_shift_group_left(const sycl::sub_group& group, uint32_t value, uint32_t delta) {
#if defined(__AMDGCN__) && !defined(MOM_SYCL_ADAPTIVECPP)
  return mom_select_from_group(group, value,
    static_cast<uint32_t>(group.get_local_linear_id()) + delta);
#else
  return sycl::shift_group_left(group, value, delta);
#endif
}

inline sycl::uint4 mom_select_from_group(const sycl::sub_group& group, const sycl::uint4 value,
                                         uint32_t lane) {
  return sycl::uint4(mom_select_from_group(group, value.x(), lane),
                     mom_select_from_group(group, value.y(), lane),
                     mom_select_from_group(group, value.z(), lane),
                     mom_select_from_group(group, value.w(), lane));
}

inline sycl::uint4 mom_shift_group_left(const sycl::sub_group& group, const sycl::uint4 value,
                                        uint32_t delta) {
  return sycl::uint4(mom_shift_group_left(group, value.x(), delta),
                     mom_shift_group_left(group, value.y(), delta),
                     mom_shift_group_left(group, value.z(), delta),
                     mom_shift_group_left(group, value.w(), delta));
}

inline bool mom_is_opencl(const sycl::device& device) {
#if defined(MOM_SYCL_ADAPTIVECPP)
  return device.get_backend() == sycl::backend::ocl;
#else
  return device.get_backend() == sycl::backend::opencl;
#endif
}

inline uint64_t mo_mul_hi_u64(const uint64_t a, const uint64_t b) {
#if defined(MOM_SYCL_ADAPTIVECPP)
    const uint64_t a0 = static_cast<uint32_t>(a), a1 = a >> 32;
    const uint64_t b0 = static_cast<uint32_t>(b), b1 = b >> 32;
    const uint64_t p0 = a0 * b0, p1 = a0 * b1, p2 = a1 * b0, p3 = a1 * b1;
    const uint64_t carry = (p0 >> 32) + static_cast<uint32_t>(p1) + static_cast<uint32_t>(p2);
    return p3 + (p1 >> 32) + (p2 >> 32) + (carry >> 32);
#else
  return sycl::mul_hi(a, b);
#endif
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
// with the FastModData mirror in kawpow/jit.inc.
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

#if defined(_WIN32)
void mom_sycl_poll_pause();
#endif

inline void sycl_wait_and_throw(sycl::event event, const sycl::device& device) {
  if constexpr (mom_sycl_portable_opencl) {
    // The OpenCL specification permits implementations to publish coarse event status updates.
    // Rusticl can leave a completed command reported as submitted until a blocking wait flushes the
    // queue, so status polling would wait forever with an idle GPU. Use the standardized wait here;
    // native CUDA/HIP/Level-Zero artifacts keep the low-CPU polling path measured below.
    (void)device;
    event.wait_and_throw();
    return;
  }
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
#if defined(_WIN32)
    // std::this_thread::sleep_for(100us) can round up to Windows' default 15.6-ms timer quantum.
    // That fixed delay dominated short GPU dispatches even though the device event had completed.
    // A high-resolution waitable timer retains the low-CPU polling design without a busy-spin or
    // process-wide timeBeginPeriod() side effect.
    mom_sycl_poll_pause();
#else
    std::this_thread::sleep_for(std::chrono::microseconds(100));
#endif
  }
  event.wait_and_throw();
}

inline void sycl_log_cleanup_exception(const char* const scope, const char* const message) noexcept {
  if (!std::getenv("MOM_SYCL_CLEANUP_DEBUG")) return;
  std::fprintf(stderr, "MOM_SYCL_CLEANUP_DEBUG %s ignored cleanup exception: %s\n",
               scope, message ? message : "unknown");
  std::fflush(stderr);
}

template <typename Fn>
inline void sycl_cleanup_noexcept(const char* const scope, Fn&& fn) noexcept {
  try {
    fn();
  } catch (const sycl::exception& e) {
    sycl_log_cleanup_exception(scope, e.what());
  } catch (const std::exception& e) {
    sycl_log_cleanup_exception(scope, e.what());
  } catch (...) {
    sycl_log_cleanup_exception(scope, "non-standard exception");
  }
}

sycl::device get_dev(const std::string& dev_str);
