// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <cstdint>

namespace mom {

// A nonempty GPU dispatch is a useful sampling point every time; CPU/RX_CPU work checks on the
// existing interval so CPU batching remains unchanged. A zero-hash dispatch is never a sample.
inline bool hashrate_check_due(const std::uint64_t completed_hashes, const bool is_gpu,
                               unsigned& remaining, const unsigned cpu_interval) noexcept {
  if (completed_hashes == 0)
    return false;
  if (is_gpu || remaining <= 1) {
    remaining = cpu_interval;
    return true;
  }
  --remaining;
  return false;
}

} // namespace mom
