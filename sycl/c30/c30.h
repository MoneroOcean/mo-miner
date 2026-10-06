// Cortex c30 SYCL miner.
//
// This header contains only protocol types shared by the solver and host verification. SYCL state
// remains private to c30.cpp and does not change the existing C29 ABI.
#pragma once

#include <array>
#include <cstdint>

namespace mom::c30 {

inline constexpr unsigned kEdgeBits = 30;
inline constexpr unsigned kProofSize = 42;

struct Job {
  // Cortex's graph key is Blake2b(header || nonce_le).  The Cortex vector uses a 32-byte seal hash.
  std::array<std::uint8_t, 32> header{};
  std::uint64_t nonce = 0;
};

struct Solution {
  std::uint64_t nonce = 0;
  std::array<std::uint32_t, kProofSize> edges{};
  std::array<std::uint8_t, 32> hash{};
};

} // namespace mom::c30
