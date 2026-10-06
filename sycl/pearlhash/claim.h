// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <cstdint>
#include <limits>

namespace mom_pearlhash {

inline bool pearlhash_adjustment_factor(const uint64_t selected_rows,
                                        const uint64_t selected_columns, const uint64_t k,
                                        const uint64_t rank, uint32_t* const output) {
  if (!output || !selected_rows || !selected_columns || !rank)
    return false;
  const uint64_t dot = k - (k % rank);
  const uint64_t steps = dot / rank;
  const uint64_t limit = std::numeric_limits<uint32_t>::max();
  uint64_t factor = selected_rows;
  const uint64_t factors[] = {selected_columns, steps, 128};
  for (const uint64_t value : factors) {
    if (!value || factor > limit / value)
      return false;
    factor *= value;
  }
  *output = static_cast<uint32_t>(factor);
  return true;
}

} // namespace mom_pearlhash
