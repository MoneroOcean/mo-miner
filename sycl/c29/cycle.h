// Copyright GNU GPLv3 (c) 2025-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mom_c29 {

constexpr uint32_t kNoParent = std::numeric_limits<uint32_t>::max();

constexpr uint32_t kEdgeBits = 29, kNumEdges = (1u << kEdgeBits);
// Recovery uses kNumEdges as its unresolved marker; every legal nonce sorts below it in signed
// and unsigned min implementations. The parent-forest sentinel above has a separate domain.
static_assert(kNumEdges < static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));

constexpr bool is_valid_nonce(const uint32_t nonce) {
  return nonce < kNumEdges;
}

struct DenseEdge {
  uint32_t first;
  uint32_t second;
};

// Dense side-specific IDs avoid repeated hash lookups during parent traversal; CPU cycle search
// overlaps GPU graphs, so host savings need not raise hashrate.
inline void create_path(const std::vector<uint32_t>& parent, const uint32_t start,
                        std::vector<uint32_t>& path, const uint32_t max_path_length = 8192) {
  path.clear();
  path.push_back(start);
  while (path.size() < max_path_length) {
    const uint32_t next = parent[path.back()];
    if (next == kNoParent)
      break;
    path.push_back(next);
  }
}

inline void reverse_path(std::vector<uint32_t>& parent, const std::vector<uint32_t>& path) {
  for (int32_t i = static_cast<int32_t>(path.size()) - 2; i >= 0; --i) {
    parent[path[static_cast<size_t>(i)]] = kNoParent;
    parent[path[static_cast<size_t>(i + 1)]] = path[static_cast<size_t>(i)];
  }
}

// Find cycles with a dense parent forest while retaining baseline insertion and output order.
template <typename Edge>
std::vector<std::vector<Edge>> find_cycles(const std::vector<Edge>& trimmed_edges,
                                           const uint32_t target_cycle_length) {
  const uint32_t edge_count = static_cast<uint32_t>(trimmed_edges.size());
  std::vector<DenseEdge> dense_edges;
  dense_edges.reserve(edge_count);
  std::vector<uint32_t> raw_id_by_dense;
  raw_id_by_dense.reserve(static_cast<size_t>(edge_count) * 2);

  // The dictionary is needed only while assigning each side's raw IDs a dense identity.
  {
    std::unordered_map<uint64_t, uint32_t> vertex_dictionary;
    vertex_dictionary.reserve(static_cast<size_t>(edge_count) * 2);
    const auto intern = [&](const uint32_t side, const uint32_t raw_id) {
      const uint64_t key = (uint64_t{side} << 32) | raw_id;
      const auto found = vertex_dictionary.find(key);
      if (found != vertex_dictionary.end())
        return found->second;
      const uint32_t dense = static_cast<uint32_t>(raw_id_by_dense.size());
      vertex_dictionary.emplace(key, dense);
      raw_id_by_dense.push_back(raw_id);
      return dense;
    };

    for (const Edge& edge : trimmed_edges)
      dense_edges.push_back({intern(0, edge.x()), intern(1, edge.y())});
  }

  std::vector<uint32_t> parent(raw_id_by_dense.size(), kNoParent);
  std::vector<std::vector<Edge>> solutions;
  std::vector<uint32_t> path_from_first;
  std::vector<uint32_t> path_from_second;
  path_from_first.reserve(64);
  path_from_second.reserve(64);

  for (uint32_t edge_idx = 0; edge_idx < edge_count; ++edge_idx) {
    const uint32_t node_first = dense_edges[edge_idx].first;
    const uint32_t node_second = dense_edges[edge_idx].second;
    if (parent[node_first] == node_second || parent[node_second] == node_first)
      continue;

    create_path(parent, node_first, path_from_first);
    create_path(parent, node_second, path_from_second);
    int64_t join_first = -1;
    int64_t join_second = -1;
    if (path_from_first.size() < 8192 && path_from_second.size() < 8192) {
      int64_t first = static_cast<int64_t>(path_from_first.size()) - 1;
      int64_t second = static_cast<int64_t>(path_from_second.size()) - 1;
      while (first >= 0 && second >= 0 &&
             path_from_first[static_cast<size_t>(first)] ==
                 path_from_second[static_cast<size_t>(second)]) {
        join_first = first;
        join_second = second;
        --first;
        --second;
      }
    } else {
      // Capped paths retain the baseline's first-match traversal order exactly.
      for (uint32_t first = 0; first < path_from_first.size() && join_first < 0; ++first) {
        for (uint32_t second = 0; second < path_from_second.size(); ++second) {
          if ((first & 1) != (second & 1) &&
              path_from_first[first] == path_from_second[second]) {
            join_first = first;
            join_second = second;
            break;
          }
        }
      }
    }

    if (join_first >= 0) {
      const int64_t cycle_length = 1 + join_first + join_second;
      if (cycle_length == target_cycle_length) {
        std::vector<Edge> cycle_edges;
        cycle_edges.reserve(target_cycle_length);
        cycle_edges.push_back({raw_id_by_dense[node_first], raw_id_by_dense[node_second]});
        const auto append_path = [&](const std::vector<uint32_t>& path,
                                     const int64_t path_edge_count,
                                     const bool starts_in_first) {
          bool current_in_first = starts_in_first;
          for (int64_t i = 0; i < path_edge_count; ++i) {
            const uint32_t from = path[static_cast<size_t>(i)];
            const uint32_t to = path[static_cast<size_t>(i + 1)];
            if (current_in_first)
              cycle_edges.push_back({raw_id_by_dense[from], raw_id_by_dense[to]});
            else
              cycle_edges.push_back({raw_id_by_dense[to], raw_id_by_dense[from]});
            current_in_first = !current_in_first;
          }
        };
        append_path(path_from_first, join_first, true);
        append_path(path_from_second, join_second, false);
        solutions.push_back(std::move(cycle_edges));
      }
      // Keep the same spanning forest when a connected edge has another length.
      continue;
    }
    if (path_from_first.size() > path_from_second.size()) {
      reverse_path(parent, path_from_second);
      parent[node_second] = node_first;
    } else {
      reverse_path(parent, path_from_first);
      parent[node_first] = node_second;
    }
  }
  return solutions;
}

} // namespace mom_c29
