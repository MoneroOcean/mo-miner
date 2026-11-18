// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <cstdint>

// Inclusive 256-bit share-target tests for host verification and device kernels: a hash meets the
// target when hash <= target, with both operands in the same byte order.

// Compare 32-byte big-endian values (byte 0 is the most significant).
inline bool mom_meets_target_be(const uint8_t* const hash, const uint8_t* const target) {
  for (unsigned i = 0; i < 32; ++i) {
    if (hash[i] != target[i])
      return hash[i] < target[i];
  }
  return true;
}

// Compare 32-byte little-endian values (byte 31 is the most significant).
inline bool mom_meets_target_le(const uint8_t* const hash, const uint8_t* const target) {
  for (int i = 31; i >= 0; --i) {
    if (hash[i] != target[i])
      return hash[i] < target[i];
  }
  return true;
}
