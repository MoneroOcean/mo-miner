// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

inline constexpr unsigned cn_gpu_lanes_per_hash = 16;

// Keep a required OpenCL transport first; unknown/mixed widths retain the portable L0 path.
inline constexpr const char* cn_gpu_intel_backend(const bool needs_opencl,
                                                const bool all_native_width8) {
  return needs_opencl ? "sycl-opencl" : all_native_width8 ? "sycl-native" : "sycl-l0";
}

// Zero means invalid: rounding UINT_MAX to an even batch would wrap to zero.
inline constexpr unsigned cn_gpu_padded_batch(const unsigned batch) {
  return batch && batch != std::numeric_limits<unsigned>::max() ? (batch + 1U) & ~1U : 0U;
}

inline constexpr unsigned cn_gpu_intel_level_zero_batch(
    const unsigned compute_units, const bool native_width8, const uint64_t global_mem,
    const uint64_t max_alloc, const unsigned requested_batch = 0) {
  constexpr uint64_t reserve = 512ULL << 20;
  constexpr uint64_t scratch_bytes = 2ULL << 20;
  constexpr uint64_t bytes_per_hash = scratch_bytes + 200 + 135 + 32;
  if (!compute_units || global_mem <= reserve || !max_alloc)
    return 0;

  // Budget every padded hash, not just its scratchpad; reported free memory is not reliable here.
  const uint64_t capacity = std::min(
      std::min((global_mem - reserve) / bytes_per_hash, max_alloc / scratch_bytes),
      static_cast<uint64_t>(std::numeric_limits<unsigned>::max() - 1U));
  const unsigned even_capacity = static_cast<unsigned>(capacity) & ~1U;
  const uint64_t default_batch = std::max<uint64_t>(
      8, (static_cast<uint64_t>(compute_units) * (native_width8 ? 8U : 3U)) & ~7ULL);
  return static_cast<unsigned>(std::min<uint64_t>(
      requested_batch ? requested_batch : default_batch, even_capacity));
}

inline constexpr unsigned cn_gpu_recurrence_group_size(const bool native_width8) {
  return cn_gpu_lanes_per_hash * (native_width8 ? 1U : 2U);
}

// The validated, even batch needs two AES rows unless it divides into four-row groups.
inline constexpr unsigned cn_gpu_aes_group_rows(const unsigned padded_batch) {
  return padded_batch % 4U == 0 ? 4U : 2U;
}
