// Host-side Cortex c30 primitives shared by the SYCL pipeline and CPU validation.
#pragma once

#include "c30.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace mom::c30::host {

struct SipKey {
  std::uint64_t k0;
  std::uint64_t k1;
  std::uint64_t k2;
  std::uint64_t k3;
};

struct Endpoints {
  std::uint32_t u;
  std::uint32_t v;
};

SipKey graph_key(const std::array<std::uint8_t, 32>& header, std::uint64_t nonce);
std::uint64_t siphash48(const SipKey& key, std::uint64_t nonce);
Endpoints edge(const SipKey& key, std::uint32_t nonce);

bool verify(const SipKey& key, const std::array<std::uint32_t, kProofSize>& edges);
std::array<std::uint8_t, 32> solution_hash(const std::array<std::uint32_t, kProofSize>& edges);

// Finds one simple 42-cycle in a trimmed endpoint graph. Nonces are recovered separately because
// retaining endpoint pairs avoids repeating the stateful SipHash chain during every trim round.
std::optional<std::array<Endpoints, kProofSize>> find_cycle(
    const std::vector<Endpoints>& trimmed_edges);

// Host-only convenience path used by the small-vector tests.
std::optional<std::array<std::uint32_t, kProofSize>> find_cycle(
    const SipKey& key, const std::vector<std::uint32_t>& trimmed_edges);

} // namespace mom::c30::host
