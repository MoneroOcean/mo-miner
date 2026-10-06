// CPU-only test for the c30 Cortex primitives.
#include "c30_host.h"

#include "crypto/randomx/blake2/blake2.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

void (*rx_blake2b_compress)(blake2b_state*, const std::uint8_t*) = rx_blake2b_compress_integer;
int (*rx_blake2b)(void*, std::size_t, const void*, std::size_t) = rx_blake2b_default;

int main() {
  using namespace mom::c30;
  const std::array<std::uint8_t, 32> header = {0x8b, 0xbb, 0x88, 0x97, 0xa7, 0x96, 0x76, 0x34,
                                               0xe1, 0x5b, 0xae, 0x66, 0x2e, 0xe2, 0x3e, 0x16,
                                               0xe8, 0xc8, 0x56, 0x69, 0xf3, 0xca, 0x0a, 0x9e,
                                               0x65, 0x84, 0xf8, 0xf4, 0xaa, 0x41, 0xf2, 0x20};
  constexpr std::uint64_t nonce = 0x1e000000d90820d4ULL;
  const auto key = host::graph_key(header, nonce);
  constexpr std::array<std::uint8_t, 32> expected_key = {
      0x00, 0x14, 0xb9, 0x44, 0x7c, 0x5d, 0x27, 0x3a, 0x93, 0xd8, 0x03,
      0x90, 0x95, 0xe1, 0xbe, 0x26, 0xee, 0x91, 0x68, 0x85, 0x78, 0x6a,
      0xeb, 0x54, 0x9a, 0x7f, 0x9a, 0x94, 0xaf, 0x12, 0x13, 0x65};
  const std::uint64_t key_words[4] = {key.k0, key.k1, key.k2, key.k3};
  for (unsigned i = 0; i < 4; ++i)
    for (unsigned byte = 0; byte < 8; ++byte)
      if (reinterpret_cast<const std::uint8_t*>(&key_words[i])[byte] != expected_key[i * 8 + byte])
        return std::fprintf(stderr, "c30 graph key mismatch\n");

  constexpr std::array<std::uint32_t, kProofSize> solution = {
      20556923,  49328084,  66386344,  128738901, 132651276,  149372831,  181150355,
      187948573, 212976704, 230374362, 238671778, 262353442,  322221503,  373413741,
      375602693, 477881229, 482331279, 513034595, 547999576,  554241182,  559490951,
      565679447, 614213668, 760389481, 821628732, 858636160,  863402811,  866207996,
      875109483, 881247921, 884674238, 896486016, 909490614,  914020222,  915733019,
      926780695, 959049446, 979837634, 989118169, 1000739073, 1028175605, 1045923045};
  if (!host::verify(key, solution)) {
    std::fprintf(stderr, "c30 graph proof mismatch\n");
    return 1;
  }
  std::vector<host::Endpoints> endpoints;
  endpoints.reserve(solution.size());
  for (const std::uint32_t edge : solution)
    endpoints.push_back(host::edge(key, edge));
  if (!host::find_cycle(endpoints)) {
    std::fprintf(stderr, "c30 endpoint cycle recovery mismatch\n");
    return 1;
  }
  std::vector<host::Endpoints> mixed{{1, 1}, {2, 1}, {2, 2}, {1, 2}};
  for (std::uint32_t i = 0; i < kProofSize / 2; ++i) {
    // Deliberately overlap u/v numeric values: equal numbers on opposite bipartite sides are not
    // path intersections and used to make the host forest discard valid edges.
    mixed.push_back({100 + i, 100 + i});
    mixed.push_back({100 + (i + 1) % (kProofSize / 2), 100 + i});
  }
  if (!host::find_cycle(mixed)) {
    std::fprintf(stderr, "c30 mixed-cycle recovery mismatch\n");
    return 1;
  }
  constexpr std::array<std::uint8_t, 32> expected_hash = {
      0x20, 0x28, 0xe2, 0x2a, 0xa1, 0x2a, 0xb3, 0xb0, 0x7e, 0xc6, 0xf0,
      0xc5, 0x74, 0xb2, 0x82, 0x98, 0x56, 0x5d, 0x47, 0xd8, 0xb0, 0x26,
      0x25, 0xfb, 0xfb, 0x85, 0xb3, 0x7d, 0xa0, 0xe9, 0xb4, 0xe4};
  if (host::solution_hash(solution) != expected_hash) {
    std::fprintf(stderr, "c30 solution hash mismatch\n");
    return 1;
  }
  std::puts("Cortex c30 host test: PASS");
  return 0;
}
