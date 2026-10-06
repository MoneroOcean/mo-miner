// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

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

inline constexpr uint64_t pearlhash_result_allocation_bytes = 64;
inline constexpr uint64_t pearlhash_runtime_reserve_bytes =
    512 * pow_intensity_detail::kMiB;

struct PearlHashAllocationCounts {
  size_t m_rank, n_rank, m_k, n_k, cva_bytes, cvb_bytes;
  uint64_t max_alloc, total_bytes;
};

inline bool pearlhash_allocation_counts(const pearlhash_profile_t& shape,
                                        PearlHashAllocationCounts& out,
                                        const bool hip_jit = false) {
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
      4 * noise_bytes + 4 * 32 + pearlhash_result_allocation_bytes +
      (hip_jit ? m_k + n_k : 0);
  out = {static_cast<size_t>(m_rank), static_cast<size_t>(n_rank),
         static_cast<size_t>(m_k), static_cast<size_t>(n_k),
         static_cast<size_t>(cva_bytes), static_cast<size_t>(cvb_bytes), largest, total};
  return true;
}

inline bool pearlhash_required_memory_fits(const PowDeviceProfile& device,
                                           const pearlhash_profile_t& shape) {
  PearlHashAllocationCounts counts{};
  return device.max_alloc && device.global_mem >= pearlhash_runtime_reserve_bytes &&
      pearlhash_allocation_counts(shape, counts) && counts.max_alloc <= device.max_alloc &&
      counts.total_bytes <= device.global_mem - pearlhash_runtime_reserve_bytes;
}

inline bool pearlhash_optional_allocation_fits(const PowDeviceProfile& device,
                                               const PearlHashAllocationCounts& required,
                                               const uint64_t optional_bytes) {
  return device.max_alloc && optional_bytes <= device.max_alloc &&
      device.global_mem >= pearlhash_runtime_reserve_bytes &&
      required.total_bytes <= device.global_mem - pearlhash_runtime_reserve_bytes &&
      optional_bytes <=
          device.global_mem - pearlhash_runtime_reserve_bytes - required.total_bytes;
}

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

inline PowIntensityScale select_pow_intensity_scale(const PowDeviceProfile& profile,
                                                    const PowIntensityHeuristic& heuristic) {
  const unsigned score = pow_device_score(profile);
  if (score >= 5)
    return heuristic.wide;
  if (score >= 3)
    return heuristic.balanced;
  return heuristic.compact;
}

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
