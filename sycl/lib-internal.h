// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "device-state.h"
#include "lib.h"
#include "workgroup-limits.h"

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

inline constexpr uint64_t KiB = 1024ULL;
inline constexpr uint64_t MiB = 1024ULL * KiB;
inline constexpr uint64_t GiB = 1024ULL * MiB;

// This is an unchecked promise that every accessor and USM kernel argument addresses
// non-overlapping storage for the kernel lifetime. Apply it only after tracing the complete
// allocation and call-site provenance.
#if __has_cpp_attribute(intel::kernel_args_restrict)
#define MOM_SYCL_KERNEL_ARGS_RESTRICT [[intel::kernel_args_restrict]]
#else
#define MOM_SYCL_KERNEL_ARGS_RESTRICT
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
    if (!count)
      throw std::string("SYCL buffer allocation size must be nonzero");
    if (buffer_ && count_ >= count)
      return;
    buffer_.reset();
    buffer_ = std::make_unique<sycl::buffer<T, 1>>(sycl::range<1>{count});
    count_ = count;
  }

  void release() {
    buffer_.reset();
    count_ = 0;
  }

  template <sycl::access_mode Mode>
  auto device_view(sycl::handler& handler) {
    if (!buffer_)
      throw std::string("SYCL buffer is not allocated");
    return sycl::accessor<T, 1, Mode, sycl::target::device>{*buffer_, handler};
  }

  sycl::event write(sycl::queue& queue, const T* const source, const size_t count) {
    if (!buffer_ || count > count_ || (!source && count))
      throw std::string("Invalid SYCL buffer write");
    return queue.submit([&](sycl::handler& handler) {
      auto view = sycl::accessor<T, 1, sycl::access_mode::write, sycl::target::device>{
        *buffer_, handler, sycl::range<1>{count}};
      handler.copy(source, view);
    });
  }

  sycl::event fill(sycl::queue& queue, const T& value) {
    if (!buffer_)
      throw std::string("SYCL buffer is not allocated");
    return queue.submit([&](sycl::handler& handler) {
      auto view = device_view<sycl::access_mode::write>(handler);
      handler.fill(view, value);
    });
  }

  void read(T* const destination, const size_t count) {
    if (!buffer_ || count > count_ || (!destination && count))
      throw std::string("Invalid SYCL buffer read");
    sycl::host_accessor view{*buffer_, sycl::read_only};
    std::copy_n(view.begin(), count, destination);
  }
};

// DPC++ and AdaptiveCpp share these helpers. Keep device-compilation guards (such as
// __NVPTX__), host build capabilities (MOM_SYCL_HAS_CUDA / MOM_SYCL_HAS_HIP), and
// runtime device checks (mom_is_cuda / mom_is_hip) distinct: a multi-backend build's
// capabilities do not identify the device selected for a particular job.

// The cooperative ProgPoW / Ethash / cn-gpu kernels run on 16-wide sub-groups on Intel
// GPUs, requested via reqd_sub_group_size(16). NVIDIA warps are fixed at 32 lanes (no
// 16-wide sub-group), so the nvptx device pass drops the attribute and lets the native
// 32-wide warp stand; those kernels address each cooperative team relative to its base
// lane within the sub-group, correct at both 16 and 32 lanes. Gating on the per-pass
// __NVPTX__ (not a build macro) lets the combined build emit sg16 in its spir64 image and
// no requirement in its nvptx image, so each device loads the image that fits it.
#if defined(__NVPTX__) || defined(MOM_SYCL_ADAPTIVECPP) || \
    defined(MOM_SYCL_PORTABLE_OPENCL)
  #define MOM_REQD_SG_16
#else
  #define MOM_REQD_SG_16 [[sycl::reqd_sub_group_size(16)]]
#endif

#if defined(MOM_SYCL_ADAPTIVECPP) || defined(MOM_SYCL_PORTABLE_OPENCL)
struct MomExecutableKernelBundle {};
#else
using MomExecutableKernelBundle = sycl::kernel_bundle<sycl::bundle_state::executable>;
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
    for (size_t i = 0; i < v.size(); ++i) {
      v[i] = p[offset * v.size() + i];
    }
  else
    v.load(offset, p);
}
template <typename VecT, typename T>
inline void mo_vec_store(const VecT& v, const size_t offset, T* const p) {
  if constexpr (mom_sycl_adaptivecpp)
    for (size_t i = 0; i < v.size(); ++i) {
      p[offset * v.size() + i] = v[i];
    }
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
  if (!value || !*value)
    return false;
  unsigned long parsed = 0;
  for (const char* cursor = value; *cursor; ++cursor) {
    if (*cursor < '0' || *cursor > '9')
      return false;
    const unsigned long digit = static_cast<unsigned long>(*cursor - '0');
    if (parsed > (std::numeric_limits<unsigned long>::max() - digit) / 10)
      return false;
    parsed = parsed * 10 + digit;
  }
  out = parsed;
  return true;
}

// Optional Intel geometry for host-side performance policy, not a requested subgroup size.
// Unavailable/failed queries retain each caller's existing unknown-device default.
inline unsigned mom_intel_eu_simd_width(const sycl::device& device) {
#if !defined(MOM_SYCL_ADAPTIVECPP) && !defined(MOM_SYCL_PORTABLE_OPENCL)
  try {
    if (device.is_gpu() &&
        device.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos &&
        device.has(sycl::aspect::ext_intel_gpu_eu_simd_width))
      return device.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
  } catch (const sycl::exception&) {
    // Optional performance information must not make an otherwise usable device fail.
  }
#else
  (void)device;
#endif
  return 0;
}

inline bool sycl_is_level_zero_gpu(const sycl::device& device) {
  return
    device.is_gpu() &&
    device.get_platform().get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos;
}

inline bool is_integrated_gpu(const sycl::device& device) {
  if (!device.is_gpu())
    return false;
  // SYCL 2020 deprecated this query in favor of USM aspects, but those describe allocation
  // support rather than whether GPU memory is physically shared with the host. Device discovery
  // needs the latter distinction and every supported runtime still implements this SYCL 1.2.1
  // property. Keep the unavoidable warning local instead of hiding unrelated deprecations.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
  const bool integrated = device.get_info<sycl::info::device::host_unified_memory>();
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
  return integrated;
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

inline uint32_t mom_select_from_group(const sycl::sub_group& group, uint32_t value, uint32_t lane) {
  return sycl::select_from_group(group, value, lane);
}

inline uint64_t mom_select_from_group(const sycl::sub_group& group, uint64_t value, uint32_t lane) {
  const uint32_t lo = mom_select_from_group(group, static_cast<uint32_t>(value), lane);
  const uint32_t hi = mom_select_from_group(group, static_cast<uint32_t>(value >> 32), lane);
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

inline uint32_t mom_shift_group_left(const sycl::sub_group& group, uint32_t value, uint32_t delta) {
  return sycl::shift_group_left(group, value, delta);
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

inline uint64_t mo_mul_wide_u32(const uint32_t a, const uint32_t b) {
#if defined(MOM_SYCL_PORTABLE_OPENCL)
  const uint32_t a0 = static_cast<uint16_t>(a), a1 = a >> 16,
                 b0 = static_cast<uint16_t>(b), b1 = b >> 16;
  const uint64_t p0 = a0 * b0, p1 = a0 * b1, p2 = a1 * b0, p3 = a1 * b1;
  return p0 + ((p1 + p2) << 16) + (p3 << 32);
#else
  return static_cast<uint64_t>(a) * b;
#endif
}

inline uint64_t mo_mul_hi_u64(const uint64_t a, const uint64_t b) {
#if defined(MOM_SYCL_ADAPTIVECPP) || defined(MOM_SYCL_PORTABLE_OPENCL)
  const uint32_t a0 = static_cast<uint32_t>(a), a1 = a >> 32,
                 b0 = static_cast<uint32_t>(b), b1 = b >> 32;
  const uint64_t p0 = mo_mul_wide_u32(a0, b0), p1 = mo_mul_wide_u32(a0, b1),
                 p2 = mo_mul_wide_u32(a1, b0), p3 = mo_mul_wide_u32(a1, b1);
  const uint64_t carry = (p0 >> 32) + static_cast<uint32_t>(p1) + static_cast<uint32_t>(p2);
  return p3 + (p1 >> 32) + (p2 >> 32) + (carry >> 32);
#else
  return sycl::mul_hi(a, b);
#endif
}

inline bool sycl_local_memory_fits(const sycl::device& device, const size_t bytes) {
  return bytes <= device.get_info<sycl::info::device::local_mem_size>();
}

inline bool sycl_workgroup_fits(const sycl::device& device, const unsigned workgroup,
                                const size_t fixed_local_bytes = 0,
                                const size_t local_bytes_per_item = 0,
                                const size_t local_bytes_per_team = 0,
                                const unsigned team_size = 1) {
  return mom_workgroup_fits(
      workgroup, device.get_info<sycl::info::device::max_work_group_size>(),
      device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0],
      device.get_info<sycl::info::device::local_mem_size>(), fixed_local_bytes,
      local_bytes_per_item, local_bytes_per_team, team_size);
}

// The DPC++ SPIR image requests subgroup 16. CUDA and AdaptiveCpp deliberately omit that attribute
// and use their native subgroup, while the standards-only OpenCL image uses local-memory fallbacks.
inline bool sycl_reqd_subgroup_16_supported(const sycl::device& device) {
  if constexpr (mom_sycl_adaptivecpp || mom_sycl_portable_opencl)
    return true;
  if (mom_is_cuda(device) || mom_is_hip(device))
    return true;
  const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
  return std::find(sizes.begin(), sizes.end(), 16u) != sizes.end();
}

inline unsigned sycl_default_workgroup(
  const sycl::device& device, const std::initializer_list<unsigned> allowed,
  const unsigned preferred, const size_t fixed_local_bytes = 0,
  const size_t local_bytes_per_item = 0, const size_t local_bytes_per_team = 0,
  const unsigned team_size = 1
) {
  unsigned selected = 0;
  for (const unsigned candidate : allowed) {
    if (candidate <= preferred && sycl_workgroup_fits(
          device, candidate, fixed_local_bytes, local_bytes_per_item,
          local_bytes_per_team, team_size)) {
      selected = std::max(selected, candidate);
    }
  }
  if (selected)
    return selected;

  unsigned smallest = 0;
  for (const unsigned candidate : allowed) {
    if (sycl_workgroup_fits(device, candidate, fixed_local_bytes, local_bytes_per_item,
                           local_bytes_per_team, team_size) &&
        (!smallest || candidate < smallest))
      smallest = candidate;
  }
  if (!smallest)
    throw std::string("No valid SYCL workgroup size");
  return smallest;
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
  if (!divisor)
    throw std::string("Fast modulus divisor must be positive");
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

inline size_t round_up_size(const uint64_t value, const uint64_t step) {
  const uint64_t maximum = std::numeric_limits<size_t>::max();
  if (!step || step > maximum || value > maximum - (step - 1))
    throw std::string("SYCL range is too large");
  return static_cast<size_t>(((value + step - 1) / step) * step);
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
  try {
    // The OpenCL specification permits implementations to publish coarse event status updates.
    // Rusticl can leave a completed command reported as submitted until a blocking wait flushes the
    // queue, so status polling would wait forever with an idle GPU. Restrict the standardized wait
    // to actual OpenCL devices: this portable artifact can also run on Level Zero, whose native wait
    // busy-spins a host core and should use the low-CPU polling path below.
    // Several GPU backends busy-spin a host core inside native event waits. Polling the event status
    // with a short sleep keeps GPU mining from pinning one CPU thread while preserving exact completion.
    // CPU devices keep the native wait because their "kernel" work is host work and should not be hidden.
    const bool poll_wait = (!mom_sycl_portable_opencl || !mom_is_opencl(device)) &&
                           device.is_gpu();
    if (poll_wait) {
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
    }
  } catch (...) {
    // A failed device/backend/status query can leave accepted work using caller-owned host buffers.
    try {
      event.wait_and_throw();
    } catch (...) {
    }
    throw;
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
  } catch (const std::exception& e) {
    sycl_log_cleanup_exception(scope, e.what());
  } catch (...) {
    sycl_log_cleanup_exception(scope, "non-standard exception");
  }
}

// Declare after the host buffers it protects and before their first asynchronous transfer. On
// unwind, retire accepted commands before those buffers die; success keeps its existing waits.
class MomSyclHostTransferGuard {
  sycl::queue& queue_;
  const char* scope_;
  int exceptions_;

public:
  MomSyclHostTransferGuard(sycl::queue& queue, const char* scope) noexcept
      : queue_(queue), scope_(scope), exceptions_(std::uncaught_exceptions()) {}
  MomSyclHostTransferGuard(const MomSyclHostTransferGuard&) = delete;
  MomSyclHostTransferGuard& operator=(const MomSyclHostTransferGuard&) = delete;

  ~MomSyclHostTransferGuard() noexcept {
    if (std::uncaught_exceptions() > exceptions_)
      sycl_cleanup_noexcept(scope_, [&] { queue_.wait_and_throw(); });
  }
};

sycl::device get_dev(const std::string& dev_str);
