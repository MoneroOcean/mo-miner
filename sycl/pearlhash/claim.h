// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <cstdint>
#include <limits>

#include "proof-axis.h"

namespace mom_pearlhash {

// Compute the rank-128-equivalent work factor for opened rows/columns, rejecting zero or uint32 overflow.
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

// Native selection owns the real public opening pattern. Scale an unmodified
// BE base target only here; a final jackpot target is merely reversed to LE.
// Use byte multiplication with a 64-bit carry so any uint32 factor is safe;
// saturate once after multiplication, never try to undo a clamped JS target.
// Commit from a temporary so failure leaves output intact and aliases are safe.
inline bool pearlhash_search_target_le(
    const uint8_t* const final_be, const uint8_t* const base_be,
    const uint64_t rows, const uint64_t columns, const uint64_t k, const uint64_t rank,
    uint8_t* const output_le) {
  if (!output_le || (!base_be && !final_be))
    return false;
  uint8_t bytes[32];
  if (!base_be) {
    for (unsigned i = 0; i < 32; ++i)
      bytes[i] = final_be[31 - i];
  } else {
    uint32_t factor = 0;
    if (!pearlhash_adjustment_factor(rows, columns, k, rank, &factor))
      return false;
    uint64_t carry = 0;
    for (unsigned i = 0; i < 32; ++i) {
      const uint64_t product = uint64_t(base_be[31 - i]) * factor + carry;
      bytes[i] = static_cast<uint8_t>(product);
      carry = product >> 8;
    }
    if (carry)
      for (auto& byte : bytes) byte = 0xff;
  }
  for (unsigned i = 0; i < 32; ++i)
    output_le[i] = bytes[i];
  return true;
}

} // namespace mom_pearlhash
