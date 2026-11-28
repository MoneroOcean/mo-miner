// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>

#include "../native/job-boundary.h"

namespace pow_intensity_detail {

inline constexpr uint64_t kMiB = 1ULL << 20;
inline constexpr uint64_t kGiB = 1ULL << 30;

} // namespace pow_intensity_detail

// Round down to a whole multiple of step (step > 0) so intensities stay workgroup-aligned.
inline unsigned round_down_to_multiple(const unsigned value, const unsigned step) {
  return value - value % step;
}

struct PowDeviceProfile {
  unsigned compute_units;
  uint64_t global_mem;
  uint64_t max_alloc;
};

struct pearlhash_profile_t {
  unsigned m, n, k, rank;
};

inline constexpr uint64_t pearlhash_result_allocation_bytes = 192;
inline constexpr uint64_t pearlhash_runtime_reserve_bytes =
    512 * pow_intensity_detail::kMiB;

struct PearlHashAllocationCounts {
  size_t m_rank, n_rank, m_k, n_k, cva_bytes, cvb_bytes;
  uint64_t max_alloc, total_bytes;
};

// Compute the exact allocation footprint for a legal shape; leave no partial counts on failure.
inline bool pearlhash_allocation_counts(const pearlhash_profile_t& shape,
                                        PearlHashAllocationCounts& out) {
  out = {};
  if (!mom::job_boundary::valid_pearlhash_shape(shape.m, shape.n, shape.k, shape.rank))
    return false;
  // Legal dimensions bound each product below 2^40 and the entire required footprint below
  // 4 TiB. Widen before multiplying; the actual allocator shares these counts.
  const uint64_t m_rank = static_cast<uint64_t>(shape.m) * shape.rank;
  const uint64_t n_rank = static_cast<uint64_t>(shape.n) * shape.rank;
  const uint64_t m_k = static_cast<uint64_t>(shape.m) * shape.k;
  const uint64_t n_k = static_cast<uint64_t>(shape.n) * shape.k;
  const uint64_t cva_bytes = (2 * (m_k / 1024 + (m_k % 1024 != 0)) + 2) * 32;
  const uint64_t cvb_bytes = (2 * (n_k / 1024 + (n_k % 1024 != 0)) + 2) * 32;
  const uint64_t noise_bytes = static_cast<uint64_t>(shape.k) * sizeof(int32_t);
  const uint64_t largest = std::max({m_rank, n_rank, m_k, n_k, cva_bytes, cvb_bytes,
                                    noise_bytes, pearlhash_result_allocation_bytes});
  if (largest > std::numeric_limits<size_t>::max())
    return false;
  const uint64_t total = m_rank + 2 * n_rank + m_k + n_k + cva_bytes + cvb_bytes +
      4 * noise_bytes + 4 * 32 + pearlhash_result_allocation_bytes;
  out = {static_cast<size_t>(m_rank), static_cast<size_t>(n_rank),
         static_cast<size_t>(m_k), static_cast<size_t>(n_k),
         static_cast<size_t>(cva_bytes), static_cast<size_t>(cvb_bytes), largest, total};
  return true;
}

// Admit required buffers only when both per-allocation and total memory leave runtime headroom.
inline bool pearlhash_required_memory_fits(const PowDeviceProfile& device,
                                           const pearlhash_profile_t& shape) {
  PearlHashAllocationCounts counts{};
  return device.max_alloc && device.global_mem >= pearlhash_runtime_reserve_bytes &&
      pearlhash_allocation_counts(shape, counts) && counts.max_alloc <= device.max_alloc &&
      counts.total_bytes <= device.global_mem - pearlhash_runtime_reserve_bytes;
}

// Select measured Intel PearlHash shapes by SIMD width and available memory.
inline pearlhash_profile_t pearlhash_intel_capability_profile(
    const PowDeviceProfile& device, const unsigned simd_width, const bool level_zero_gpu,
    const bool discrete_gpu, const unsigned execution_units = 0) {
  const pearlhash_profile_t ordinary = {131072, 131072, 2048, 128};
  if (!level_zero_gpu || !discrete_gpu ||
      device.global_mem < 8 * pow_intensity_detail::kGiB)
    return ordinary;
  if (simd_width == 16) {
    const pearlhash_profile_t previous = {131072, 131072, 4096, 128};
    if (execution_units == 160) {
      const pearlhash_profile_t larger = {262144, 131072, 4096, 128};
      if (pearlhash_required_memory_fits(device, larger))
        return larger;
    }
    return previous;
  }
  // Keep power-of-two dimensions while amortizing preparation over the available execution units.
  if (simd_width == 8)
    return {131072u * (execution_units >= 512 ? 4u : execution_units >= 256 ? 2u : 1u),
            131072, 2048, 128};
  return ordinary;
}

// Amortize dispatch over the largest measured complete-SM-wave batch, bounded
// by signed indexing and disjoint raw streams. Keep the smaller batch when
// memory requires it; unknown capacity or missing native features fail closed.
inline pearlhash_profile_t pearlhash_nvidia_compact_profile_for(
    const PowDeviceProfile& device, const bool native_compact) {
  if (!native_compact || !device.compute_units)
    return {};
  // The measured 36-SM class benefits from sixteen full waves on the wider
  // grid. Other SM counts retain their existing batch until measured; exact
  // memory admission also keeps smaller-allocation devices on that path.
  if (device.compute_units == 36) {
    constexpr uint64_t fast_n = 32768;
    const uint64_t fast_quantum = uint64_t(192) * device.compute_units /
        std::gcd(device.compute_units, unsigned(fast_n / 256));
    const pearlhash_profile_t fast = {unsigned(16 * fast_quantum), unsigned(fast_n), 65536, 128};
    if (pearlhash_required_memory_fits(device, fast))
      return fast;
  }
  constexpr uint64_t n = 15360, k = 65536, minimum_m = 16000;
  // Use the compact kernel's logical CTA rows, not its native warp-tile height.
  const uint64_t quantum = uint64_t(192) * device.compute_units /
      std::gcd(device.compute_units, unsigned(n / 256));
  const uint64_t minimum = ((minimum_m + quantum - 1) / quantum) * quantum;
  const uint64_t bound = std::min(uint64_t(std::numeric_limits<int32_t>::max()) / k,
                                (uint64_t(1) << 32) / k - n);
  const uint64_t preferred = bound / quantum * quantum;
  for (const uint64_t m : {preferred, minimum}) {
    if (m < minimum || m > bound)
      continue;
    const pearlhash_profile_t profile = {unsigned(m), unsigned(n), unsigned(k), 128};
    if (pearlhash_required_memory_fits(device, profile))
      return profile;
  }
  return {};
}

// Prefer measured native AMD profiles in throughput order, but admit every required allocation
// and runtime headroom first. Unknown capability or capacity retains the portable profile.
inline pearlhash_profile_t pearlhash_amd_native_profile_for(const PowDeviceProfile& device,
                                                            const bool native_int8) {
  if (native_int8) {
    // m is a whole number of the native kernel's 208-row lane-proof groups (kAmdLaneRows).
    for (const auto& profile : {pearlhash_profile_t{524160, 524288, 2048, 128},
                                pearlhash_profile_t{262080, 262144, 2048, 128}})
      if (pearlhash_required_memory_fits(device, profile))
        return profile;
  }
  return {131072, 131072, 2048, 128};
}

// Admit an optional fast-path buffer without stealing the required buffers' reserved capacity.
inline bool pearlhash_optional_allocation_fits(const PowDeviceProfile& device,
                                               const PearlHashAllocationCounts& required,
                                               const uint64_t optional_bytes) {
  return device.max_alloc && optional_bytes <= device.max_alloc &&
      device.global_mem >= pearlhash_runtime_reserve_bytes &&
      required.total_bytes <= device.global_mem - pearlhash_runtime_reserve_bytes &&
      optional_bytes <=
          device.global_mem - pearlhash_runtime_reserve_bytes - required.total_bytes;
}

// Keep the preferred complete configuration, fall back to the compact one, or reject known limits.
inline pearlhash_profile_t pearlhash_capacity_profile(const PowDeviceProfile& device,
                                                     const pearlhash_profile_t& preferred) {
  // Unknown limits retain the existing profile for ordinary discovery/admission to reject;
  // they must never be treated as evidence that the allocations fit.
  if (!device.global_mem || !device.max_alloc)
    return preferred;
  if (pearlhash_required_memory_fits(device, preferred))
    return preferred;
  const pearlhash_profile_t compact = {65536, 65536, 4096, 256};
  return pearlhash_required_memory_fits(device, compact) ? compact : pearlhash_profile_t{};
}

struct PowIntensityScale {
  unsigned fallback_workgroup;
  unsigned base_work_items;
  unsigned compute_unit_divisor;
};

struct PowIntensityHeuristic {
  PowIntensityScale compact;
  PowIntensityScale balanced;
  PowIntensityScale wide;
  unsigned max_work_items_per_gib = 0;
};

inline constexpr PowIntensityHeuristic progpow_intensity_heuristic = {
    {256, 16384, 48}, {256, 32768, 48}, {256, 32768, 36}, 4};

// Classify dispatch capacity from reported resource dimensions; compute_units must be nonzero.
inline unsigned pow_device_score(const PowDeviceProfile& profile) {
  // Favor deeper in-flight batches only on GPUs with enough parallelism and memory headroom.
  const uint64_t mem_per_cu = profile.global_mem / profile.compute_units;
  unsigned score = 0;
  if (profile.compute_units >= 128)
    score += 2;
  else if (profile.compute_units >= 64)
    score += 1;
  if (profile.global_mem >= 8ULL * pow_intensity_detail::kGiB)
    score += 2;
  else if (profile.global_mem >= 6ULL * pow_intensity_detail::kGiB)
    score += 1;
  if (profile.max_alloc >= 2ULL * pow_intensity_detail::kGiB)
    score += 1;
  if (mem_per_cu >= 48ULL * pow_intensity_detail::kMiB)
    score += 1;
  return score;
}

// Pick the heuristic tier from the device score: wide (>= 5), balanced (>= 3), else compact.
inline PowIntensityScale select_pow_intensity_scale(const PowDeviceProfile& profile,
                                                    const PowIntensityHeuristic& heuristic) {
  const unsigned score = pow_device_score(profile);
  if (score >= 5)
    return heuristic.wide;
  if (score >= 3)
    return heuristic.balanced;
  return heuristic.compact;
}

// Scale an aligned batch while capping memory depth; local and profile.compute_units must be nonzero.
inline unsigned pow_intensity(const PowDeviceProfile& profile,
                              const PowIntensityHeuristic& heuristic, const unsigned local,
                              const unsigned explicit_max_work_items_per_gib = 0) {
  const PowIntensityScale scale = select_pow_intensity_scale(profile, heuristic);
  uint64_t intensity64 =
      static_cast<uint64_t>(local) * scale.base_work_items * profile.compute_units;
  intensity64 /= scale.compute_unit_divisor;
  const unsigned max_work_items_per_gib =
      explicit_max_work_items_per_gib ? explicit_max_work_items_per_gib
                                      : heuristic.max_work_items_per_gib;
  if (max_work_items_per_gib) {
    // Bound dispatch depth by memory while preserving at least one GiB of reported capacity.
    const uint64_t memory_depth_cap =
        std::max<uint64_t>(1, profile.global_mem / pow_intensity_detail::kGiB) *
        max_work_items_per_gib * pow_intensity_detail::kMiB;
    intensity64 = std::min(intensity64, memory_depth_cap);
  }
  intensity64 = std::min<uint64_t>(intensity64, std::numeric_limits<unsigned>::max());
  const unsigned intensity = round_down_to_multiple(static_cast<unsigned>(intensity64), local);
  return std::max(intensity, local * 4096u);
}
