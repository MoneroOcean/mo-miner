// Copyright GNU GPLv3 (c) 2026 MoneroOcean
// Host-only regression for cached PearlHash Merkle leaves.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../../sycl/pearlhash/blake3.inc"
#include "../../sycl/pearlhash/claim.h"

namespace mom_pearlhash {

static size_t checked_elements(const int rows, const int columns, const char*) {
  if (rows <= 0 || columns <= 0)
    throw std::string("invalid dimensions");
  const size_t row_count = static_cast<size_t>(rows);
  const size_t column_count = static_cast<size_t>(columns);
  if (row_count > std::numeric_limits<size_t>::max() / column_count)
    throw std::string("invalid dimensions");
  return row_count * column_count;
}

static size_t checked_chunk_count(const size_t bytes, const char*) {
  const size_t chunks = bytes / 1024 + (bytes % 1024 != 0);
  if (chunks > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::string("invalid chunk count");
  return chunks;
}

static inline int8_t gv(uint32_t seed, uint32_t idx) {
  uint32_t x = seed + idx * 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return static_cast<int8_t>((x & 127) - 64);
}

#include "../../sycl/pearlhash/host.inc"

static std::vector<CV> leaf_cvs(char which, uint32_t seed, int m, int n, int k,
                                const uint8_t* key) {
  const size_t total = checked_elements(which == 'A' ? m : n, k, "fixture matrix");
  const size_t count = checked_chunk_count(total, "fixture matrix");
  std::vector<CV> result(count);
  uint8_t chunk[1024];
  for (size_t i = 0; i < count; ++i) {
    const size_t len = gen_leaf(which, seed, i, m, n, k, chunk);
    pearlhash_b3::chunk_cv(chunk, static_cast<uint32_t>(len), static_cast<uint64_t>(i), key,
                           result[i].data());
  }
  return result;
}

struct Profile {
  int m, n, k, rank, row, col;
};

} // namespace mom_pearlhash

int main() {
  using namespace mom_pearlhash;
  constexpr uint32_t seed = 0x12345678u;
  uint8_t header[76];
  uint8_t key[32];
  for (unsigned i = 0; i < sizeof(header); ++i)
    header[i] = static_cast<uint8_t>(0x31u + i * 17u);
  derive_key(header, 1024, 128, key);

  // Retained top layers must match the separate canonical reduction, including odd carries.
  for (const char which : {'A', 'B'}) {
    for (const int count : {1, 2, 3, 5, 6, 7, 9, 16, 17, 31, 32, 33, 48, 65, 127, 128, 129}) {
      std::vector<CV> leaves = leaf_cvs(which, seed, count, count, 1024, key);
      const auto layers = build_layers(which, seed, count, count, count, 1024, key, &leaves);
      std::vector<CV> top = layers.back();
      uint8_t expected[32], retained[32];
      pearlhash_b3::merge_root(reinterpret_cast<uint8_t*>(leaves.data()), count, key, expected);
      pearlhash_b3::merge_root(reinterpret_cast<uint8_t*>(top.data()),
                              static_cast<int>(top.size()), key, retained);
      std::vector<uint8_t> encoded;
      ser_matrix_proof(encoded, which, seed, count, count, 1024, key, 1024, {0}, &leaves);
      // One opened 1024-byte leaf: five u64 fields precede the serialized root.
      constexpr size_t root_offset = 1024 + 5 * sizeof(uint64_t);
      if (std::memcmp(expected, retained, 32) != 0 || encoded.size() < root_offset + 32 ||
          std::memcmp(expected, encoded.data() + root_offset, 32) != 0) {
        std::fprintf(stderr, "PearlHash retained/serialized root mismatch for %d leaves\n", count);
        return 1;
      }
    }
  }

  for (const Profile profile : {Profile{32, 32, 1024, 128, 0, 8},
                                Profile{24, 24, 1024, 128, 0, 8},
                                Profile{32, 24, 4160, 128, 0, 8}}) {
    const std::vector<CV> cached_a = leaf_cvs('A', seed, profile.m, profile.n, profile.k, key);
    const std::vector<CV> cached_b = leaf_cvs('B', seed, profile.m, profile.n, profile.k, key);
    uint32_t regenerated_factor = 0, cached_factor = 0;
    const std::string regenerated = build_plain_proof(
        seed, profile.m, profile.n, profile.k, profile.rank, key, profile.row, profile.col,
        &regenerated_factor);
    const std::string cached = build_plain_proof(
        seed, profile.m, profile.n, profile.k, profile.rank, key, profile.row, profile.col,
        &cached_factor, &cached_a, &cached_b);
    if (regenerated != cached || regenerated_factor != cached_factor) {
      std::fprintf(stderr, "PearlHash cached PlainProof mismatch for %zu leaves\n", cached_a.size());
      return 1;
    }
  }

  const Profile profile{32, 32, 1024, 128, 0, 8};
  const std::vector<CV> cached_a = leaf_cvs('A', seed, profile.m, profile.n, profile.k, key);
  const std::vector<CV> cached_b = leaf_cvs('B', seed, profile.m, profile.n, profile.k, key);
  std::vector<CV> wrong = cached_a;
  wrong.pop_back();
  uint32_t factor = 0;
  try {
    (void)build_plain_proof(seed, profile.m, profile.n, profile.k, profile.rank, key,
                             profile.row, profile.col, &factor, &wrong, &cached_b);
    std::fprintf(stderr, "PearlHash accepted a cached leaf count mismatch\n");
    return 1;
  } catch (const std::string&) {
  }

  std::puts("PearlHash cached PlainProof matches regenerated proof (power-of-two/non-power-of-two)");
  return 0;
}
