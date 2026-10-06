// Host-side Cortex c30 graph, proof, and solution-hash implementation.
#include "c30_host.h"

#include "crypto/randomx/blake2/blake2.h"
#include "base/crypto/keccak.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace mom::c30::host {
namespace {

constexpr std::uint32_t kEdgeMask = (1u << kEdgeBits) - 1u;
constexpr std::uint32_t kEdgeBlockMask = 63u;

std::uint64_t load_le64(const std::uint8_t* p) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= static_cast<std::uint64_t>(p[i]) << (i * 8);
  return value;
}

std::uint64_t rotate_left(const std::uint64_t value, const unsigned bits) {
  return (value << bits) | (value >> (64 - bits));
}

void sip_round(std::uint64_t& v0, std::uint64_t& v1, std::uint64_t& v2, std::uint64_t& v3) {
  v0 += v1;
  v2 += v3;
  v1 = rotate_left(v1, 13);
  v3 = rotate_left(v3, 16);
  v1 ^= v0;
  v3 ^= v2;
  v0 = rotate_left(v0, 32);
  v2 += v1;
  v0 += v3;
  v1 = rotate_left(v1, 17);
  v3 = rotate_left(v3, 21);
  v1 ^= v2;
  v3 ^= v0;
  v2 = rotate_left(v2, 32);
}

using Graph = std::unordered_map<std::uint32_t, std::uint32_t>;

std::vector<std::uint32_t> create_path(const Graph& graph_u, const Graph& graph_v,
                                       const bool starts_in_u, const std::uint32_t start) {
  std::vector<std::uint32_t> path{start};
  path.reserve(64);
  bool in_u = starts_in_u;
  const Graph* graph = in_u ? &graph_u : &graph_v;
  auto link = graph->find(start);
  while (link != graph->end() && path.size() < 8192) {
    path.push_back(link->second);
    in_u = !in_u;
    graph = in_u ? &graph_u : &graph_v;
    link = graph->find(path.back());
  }
  return path;
}

void reverse_path(Graph& graph_u, Graph& graph_v, const std::vector<std::uint32_t>& path,
                  const bool starts_in_u) {
  Graph* graphs[2] = {&graph_u, &graph_v};
  for (std::size_t i = path.size() - 1; i > 0; --i) {
    const unsigned remove = starts_in_u ? ((i - 1) & 1u) : !((i - 1) & 1u);
    graphs[remove]->erase(path[i - 1]);
    (*graphs[1u - remove])[path[i]] = path[i - 1];
  }
}

std::optional<std::array<Endpoints, kProofSize>>
find_cycle_graph(const std::vector<Endpoints>& edges) {
  if (edges.size() > (1u << 22))
    return std::nullopt;
  Graph graph_u, graph_v;
  graph_u.reserve(edges.size());
  graph_v.reserve(edges.size());

  for (const Endpoints edge : edges) {
    const auto u_link = graph_u.find(edge.u);
    if (u_link != graph_u.end() && u_link->second == edge.v)
      continue;
    const auto v_link = graph_v.find(edge.v);
    if (v_link != graph_v.end() && v_link->second == edge.u)
      continue;
    const auto path_u = create_path(graph_u, graph_v, true, edge.u);
    const auto path_v = create_path(graph_u, graph_v, false, edge.v);
    std::size_t join_u = path_u.size(), join_v = path_v.size();
    for (std::size_t i = 0; i < path_u.size() && join_u == path_u.size(); ++i) {
      // The endpoint number spaces overlap, but u and v remain distinct graph partitions. Since
      // these paths start on opposite sides, a real intersection has opposite index parity.
      for (std::size_t j = 1u - (i & 1u); j < path_v.size(); j += 2u)
        if (path_v[j] == path_u[i]) {
          join_u = i;
          join_v = j;
          break;
        }
    }
    if (join_u != path_u.size()) {
      if (1 + join_u + join_v != kProofSize)
        continue;
      std::array<Endpoints, kProofSize> cycle{};
      std::size_t output = 0;
      cycle[output++] = edge;
      for (std::size_t i = 0; i < join_u; ++i)
        cycle[output++] =
            i & 1u ? Endpoints{path_u[i + 1], path_u[i]} : Endpoints{path_u[i], path_u[i + 1]};
      for (std::size_t i = 0; i < join_v; ++i)
        cycle[output++] =
            i & 1u ? Endpoints{path_v[i], path_v[i + 1]} : Endpoints{path_v[i + 1], path_v[i]};
      return cycle;
    }
    if (path_u.size() > path_v.size()) {
      reverse_path(graph_u, graph_v, path_v, false);
      graph_v[edge.v] = edge.u;
    } else {
      reverse_path(graph_u, graph_v, path_u, true);
      graph_u[edge.u] = edge.v;
    }
  }
  return std::nullopt;
}

} // namespace

SipKey graph_key(const std::array<std::uint8_t, 32>& header, const std::uint64_t nonce) {
  std::array<std::uint8_t, 40> header_nonce{};
  std::copy(header.begin(), header.end(), header_nonce.begin());
  for (unsigned i = 0; i < sizeof(nonce); ++i)
    header_nonce[32 + i] = static_cast<std::uint8_t>(nonce >> (i * 8));

  std::array<std::uint8_t, 32> digest{};
  if (rx_blake2b_default(digest.data(), digest.size(), header_nonce.data(), header_nonce.size()) !=
      0)
    return {};
  return {load_le64(digest.data()), load_le64(digest.data() + 8), load_le64(digest.data() + 16),
          load_le64(digest.data() + 24)};
}

std::uint64_t siphash48(const SipKey& key, const std::uint64_t nonce) {
  std::uint64_t v0 = key.k0;
  std::uint64_t v1 = key.k1;
  std::uint64_t v2 = key.k2;
  std::uint64_t v3 = key.k3;
  v3 ^= nonce;
  for (unsigned i = 0; i < 4; ++i)
    sip_round(v0, v1, v2, v3);
  v0 ^= nonce;
  v2 ^= 0xff;
  for (unsigned i = 0; i < 8; ++i)
    sip_round(v0, v1, v2, v3);
  return (v0 ^ v1) ^ (v2 ^ v3);
}

Endpoints edge(const SipKey& key, const std::uint32_t nonce) {
  // Cortex's SipHash block is stateful: one state walks all 64 nonces, and every value except the
  // last is XORed with that block's final value.  It is not equivalent to two independent hashes.
  const std::uint32_t target = nonce & kEdgeBlockMask;
  const std::uint64_t first = static_cast<std::uint64_t>(nonce & ~kEdgeBlockMask);
  std::uint64_t v0 = key.k0;
  std::uint64_t v1 = key.k1;
  std::uint64_t v2 = key.k2;
  std::uint64_t v3 = key.k3;
  std::uint64_t selected = 0;
  std::uint64_t last = 0;
  for (std::uint32_t i = 0; i < 64; ++i) {
    const std::uint64_t value_nonce = first + i;
    v3 ^= value_nonce;
    for (unsigned round = 0; round < 4; ++round)
      sip_round(v0, v1, v2, v3);
    v0 ^= value_nonce;
    v2 ^= 0xff;
    for (unsigned round = 0; round < 8; ++round)
      sip_round(v0, v1, v2, v3);
    const std::uint64_t value = (v0 ^ v1) ^ (v2 ^ v3);
    if (i == target)
      selected = value;
    if (i == kEdgeBlockMask)
      last = value;
  }
  const std::uint64_t value = target == kEdgeBlockMask ? last : selected ^ last;
  return {static_cast<std::uint32_t>(value) & kEdgeMask,
          static_cast<std::uint32_t>(value >> 32) & kEdgeMask};
}

bool verify(const SipKey& key, const std::array<std::uint32_t, kProofSize>& edges) {
  std::array<std::uint32_t, 2 * kProofSize> endpoints{};
  std::uint32_t xor_u = 0;
  std::uint32_t xor_v = 0;

  for (unsigned i = 0; i < kProofSize; ++i) {
    if (edges[i] > kEdgeMask || (i && edges[i] <= edges[i - 1]))
      return false;
    const Endpoints pair = edge(key, edges[i]);
    endpoints[2 * i] = pair.u;
    endpoints[2 * i + 1] = pair.v;
    xor_u ^= pair.u;
    xor_v ^= pair.v;
  }
  if (xor_u || xor_v)
    return false;

  unsigned length = 0;
  unsigned current = 0;
  do {
    unsigned match = current;
    for (unsigned i = current; (i = (i + 2) % (2 * kProofSize)) != current;) {
      if (endpoints[i] == endpoints[current]) {
        if (match != current)
          return false;
        match = i;
      }
    }
    if (match == current)
      return false;
    current = match ^ 1;
    ++length;
  } while (current != 0);
  return length == kProofSize;
}

std::array<std::uint8_t, 32> solution_hash(const std::array<std::uint32_t, kProofSize>& edges) {
  std::array<std::uint8_t, kProofSize * sizeof(std::uint32_t)> bytes{};
  for (unsigned i = 0; i < kProofSize; ++i) {
    for (unsigned byte = 0; byte < sizeof(std::uint32_t); ++byte)
      bytes[4 * i + byte] = static_cast<std::uint8_t>(edges[i] >> (8 * (3 - byte)));
  }
  std::array<std::uint8_t, 32> digest{};
  xmrig::keccak(bytes.data(), bytes.size(), digest.data(), digest.size());
  return digest;
}

std::optional<std::array<Endpoints, kProofSize>>
find_cycle(const std::vector<Endpoints>& trimmed_edges) {
  return find_cycle_graph(trimmed_edges);
}

std::optional<std::array<std::uint32_t, kProofSize>>
find_cycle(const SipKey& key, const std::vector<std::uint32_t>& trimmed_edges) {
  if (trimmed_edges.size() > (1u << 22))
    return std::nullopt;
  std::vector<Endpoints> edges;
  edges.reserve(trimmed_edges.size());
  for (const std::uint32_t nonce : trimmed_edges)
    edges.push_back(edge(key, nonce));
  const auto cycle = find_cycle_graph(edges);
  if (!cycle)
    return std::nullopt;
  std::array<std::uint32_t, kProofSize> candidate{};
  std::vector<bool> used(edges.size());
  for (unsigned i = 0; i < kProofSize; ++i) {
    const auto match = std::find_if(edges.begin(), edges.end(), [&](const Endpoints& edge) {
      const std::size_t index = &edge - edges.data();
      return !used[index] && edge.u == (*cycle)[i].u && edge.v == (*cycle)[i].v;
    });
    if (match == edges.end())
      return std::nullopt;
    const std::size_t index = static_cast<std::size_t>(match - edges.begin());
    used[index] = true;
    candidate[i] = trimmed_edges[index];
  }
  std::sort(candidate.begin(), candidate.end());
  return verify(key, candidate) ? std::optional{candidate} : std::nullopt;
}

} // namespace mom::c30::host
