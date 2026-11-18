#include "../../sycl/c29/cycle.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

namespace {

struct Edge {
  uint32_t u;
  uint32_t v;

  uint32_t x() const {
    return u;
  }

  uint32_t y() const {
    return v;
  }
};

std::set<uint64_t> canonical_edge_set(const std::vector<Edge>& edges) {
  std::set<uint64_t> result;
  for (const Edge& edge : edges)
    result.insert(static_cast<uint64_t>(edge.u) << 32 | edge.v);
  return result;
}

void assert_ordered_cycle(const std::vector<Edge>& actual,
                          const std::vector<Edge>& expected) {
  assert(actual.size() == expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    assert(actual[index].u == expected[index].u);
    assert(actual[index].v == expected[index].v);
  }
}

std::vector<Edge> make_ring(const uint32_t half, const uint32_t offset = 0) {
  std::vector<Edge> edges;
  edges.reserve(static_cast<std::size_t>(half) * 2);
  for (uint32_t index = 0; index < half; ++index) {
    edges.push_back({offset + index, offset + index});
    edges.push_back({offset + ((index + 1) % half), offset + index});
  }
  return edges;
}

void assert_single_ring(const std::vector<Edge>& edges, const uint32_t target,
                        const std::set<uint64_t>& expected) {
  const auto cycles = mom_c29::find_cycles(edges, target);
  assert(cycles.size() == 1);
  assert(cycles.front().size() == target);
  assert(canonical_edge_set(cycles.front()) == expected);
}

void assert_valid_ring_cycles(const std::vector<std::vector<Edge>>& cycles,
                              const uint32_t target,
                              const std::set<uint64_t>& expected) {
  assert(!cycles.empty());
  for (const auto& cycle : cycles) {
    assert(cycle.size() == target);
    assert(canonical_edge_set(cycle) == expected);
  }
}

void test_ring_length(const uint32_t half) {
  const uint32_t target = half * 2;
  const std::vector<Edge> canonical = make_ring(half);
  const std::set<uint64_t> expected = canonical_edge_set(canonical);
  assert(canonical.size() == target);

  assert_single_ring(canonical, target, expected);

  std::vector<Edge> reversed = canonical;
  std::reverse(reversed.begin(), reversed.end());
  assert_single_ring(reversed, target, expected);

  std::mt19937 rng(0xc29d2026U);
  for (uint32_t shuffle_index = 0; shuffle_index < 16; ++shuffle_index) {
    std::vector<Edge> shuffled = canonical;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    assert_single_ring(shuffled, target, expected);
  }

  const std::vector<Edge> second_ring = make_ring(half, 1000);
  std::vector<Edge> disconnected = canonical;
  disconnected.insert(disconnected.end(), second_ring.begin(), second_ring.end());
  const auto disconnected_cycles = mom_c29::find_cycles(disconnected, target);
  assert(disconnected_cycles.size() == 2);
  std::set<std::set<uint64_t>> actual_sets;
  for (const auto& cycle : disconnected_cycles) {
    assert(cycle.size() == target);
    actual_sets.insert(canonical_edge_set(cycle));
  }
  assert(actual_sets.size() == 2);
  assert(actual_sets.count(expected) == 1);
  assert(actual_sets.count(canonical_edge_set(second_ring)) == 1);

  const auto wrong_target_cycles = mom_c29::find_cycles(canonical, target - 2);
  assert(wrong_target_cycles.empty());

  std::vector<Edge> duplicates = canonical;
  duplicates.insert(duplicates.end(), canonical.begin(), canonical.end());
  // A closing edge is not inserted into the forest, so repeating it may emit the cycle again.
  const auto duplicate_cycles = mom_c29::find_cycles(duplicates, target);
  assert_valid_ring_cycles(duplicate_cycles, target, expected);
}

} // namespace

int main() {
  // Equal numeric IDs on opposite bipartite sides must not be treated as a path intersection.
  const std::vector<Edge> edges = {
    {7, 6}, {4, 5}, {5, 7}, {4, 4}, {5, 5}, {6, 6}, {4, 6}, {5, 4},
  };
  const auto cycles = mom_c29::find_cycles(edges, 4);
  assert(cycles.size() == 1);
  assert(cycles.front().size() == 4);
  const std::set<uint64_t> expected = {
    (uint64_t{4} << 32) | 4,
    (uint64_t{4} << 32) | 5,
    (uint64_t{5} << 32) | 4,
    (uint64_t{5} << 32) | 5,
  };
  assert(canonical_edge_set(cycles.front()) == expected);
  const std::vector<Edge> expected_order = {
    {5, 4}, {4, 4}, {4, 5}, {5, 5},
  };
  assert_ordered_cycle(cycles.front(), expected_order);

  std::vector<Edge> repeated_edges = edges;
  repeated_edges.push_back(edges.back());
  const auto repeated_cycles = mom_c29::find_cycles(repeated_edges, 4);
  assert(repeated_cycles.size() == 2);
  for (const auto& cycle : repeated_cycles)
    assert_ordered_cycle(cycle, expected_order);

  // Closing a non-target cycle must not replace an edge in the spanning forest and hide a later
  // target-length cycle in the same component.
  const std::vector<Edge> mixed_cycles = {
    {1, 1}, {6, 3}, {1, 3}, {2, 8}, {6, 8}, {2, 1}, {7, 8}, {7, 3},
  };
  const auto target_cycles = mom_c29::find_cycles(mixed_cycles, 4);
  assert(target_cycles.size() == 1);
  const std::set<uint64_t> expected_target = {
    (uint64_t{6} << 32) | 3,
    (uint64_t{6} << 32) | 8,
    (uint64_t{7} << 32) | 3,
    (uint64_t{7} << 32) | 8,
  };
  assert(canonical_edge_set(target_cycles.front()) == expected_target);
  const std::vector<Edge> expected_target_order = {
    {7, 3}, {7, 8}, {6, 8}, {6, 3},
  };
  assert_ordered_cycle(target_cycles.front(), expected_target_order);

  std::vector<Edge> repeated_mixed_cycles = mixed_cycles;
  repeated_mixed_cycles.push_back(mixed_cycles.back());
  const auto repeated_target_cycles = mom_c29::find_cycles(repeated_mixed_cycles, 4);
  assert(repeated_target_cycles.size() == 2);
  for (const auto& cycle : repeated_target_cycles)
    assert_ordered_cycle(cycle, expected_target_order);

  test_ring_length(16);
  test_ring_length(21);
  return 0;
}
