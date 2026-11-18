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
#include "../../sycl/pearlhash/seed.inc"

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

static inline int8_t gv(uint32_t seed, uint32_t idx, bool positive_inputs = false,
                        bool seed_prefix_inputs = false) {
  // Only the compact raw solver uses this legal stream. Legacy native inputs
  // remain the hashed 63/64 stream, rather than inheriting a new seed contract.
  if (positive_inputs && seed_prefix_inputs)
    return static_cast<int8_t>(idx < 32 ? 63 + ((seed >> idx) & 1u) : 64);
  uint32_t x = seed + idx * 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return positive_inputs ? static_cast<int8_t>(63 + (x & 1u))
                         : static_cast<int8_t>((x & 127u) - 64);
}

#include "../../sycl/pearlhash/host.inc"

static std::vector<CV> leaf_cvs(char which, uint32_t seed, int m, int n, int k,
                                const uint8_t* key, bool positive_inputs = false) {
  const size_t total = checked_elements(which == 'A' ? m : n, k, "fixture matrix");
  const size_t count = checked_chunk_count(total, "fixture matrix");
  std::vector<CV> result(count);
  uint8_t chunk[1024];
  for (size_t i = 0; i < count; ++i) {
    const size_t len = gen_leaf(which, seed, i, m, n, k, chunk, positive_inputs);
    pearlhash_b3::chunk_cv(chunk, static_cast<uint32_t>(len), static_cast<uint64_t>(i), key,
                           result[i].data());
  }
  return result;
}

struct Profile {
  int m, n, k, rank, row, col;
};

// Retaining B must leave A free to advance, preserve seed zero, and keep legacy proof bytes intact.
static bool fixed_b_host_contract() {
  constexpr int m = 64, n = 64, k = 8192, rank = 128;
  constexpr uint32_t a0 = 0x10203040u, a1 = 0x50607080u;
  constexpr uint32_t b0 = 0u, b1 = 0xfedcba98u;
  uint8_t header[76]{}, key[32]{};
  for (unsigned i = 0; i < sizeof(header); ++i)
    header[i] = static_cast<uint8_t>(0x21u + i * 29u);
  derive_key_pair_dist8(header, k, rank, key);
  const auto a0_leaves = leaf_cvs('A', a0, m, n, k, key, true);
  const auto a1_leaves = leaf_cvs('A', a1, m, n, k, key, true);
  const auto b0_leaves = leaf_cvs('B', b0, m, n, k, key, true);
  const auto b1_leaves = leaf_cvs('B', b1, m, n, k, key, true);
  const auto root = [&](const std::vector<CV>& input) {
    std::vector<CV> copy = input;
    CV result{};
    pearlhash_b3::merge_root(reinterpret_cast<uint8_t*>(copy.data()),
                              static_cast<int>(copy.size()), key, result.data());
    return result;
  };
  const CV raw_a0 = root(a0_leaves), raw_a1 = root(a1_leaves);
  const CV raw_b0 = root(b0_leaves), raw_b1 = root(b1_leaves);
  std::array<uint8_t, 32> cb0{}, cb1{}, ca0{}, ca1{};
  pearlhash_commitment_seed_b(key, raw_b0.data(), n, cb0.data());
  pearlhash_commitment_seed_b(key, raw_b1.data(), n, cb1.data());
  pearlhash_commitment_seed_a(cb0.data(), raw_a0.data(), m, ca0.data());
  pearlhash_commitment_seed_a(cb0.data(), raw_a1.data(), m, ca1.data());
  if (raw_a0 == raw_a1 || raw_b0 == raw_b1 || cb0 == cb1 || ca0 == ca1)
    return false;
  uint32_t f0 = 0, f1 = 0, f2 = 0, f3 = 0;
  const auto uncached = build_pair_plain_proof(a0, m, n, k, rank, key, 0, 0, &f0,
                                                nullptr, nullptr, true, &b0);
  const auto cached = build_pair_plain_proof(a0, m, n, k, rank, key, 0, 0, &f1,
                                              &a0_leaves, &b0_leaves, true, &b0);
  const auto advancing = build_pair_plain_proof(a1, m, n, k, rank, key, 0, 0, &f2,
                                                 &a1_leaves, &b0_leaves, true, &b0);
  const auto changed_b = build_pair_plain_proof(a0, m, n, k, rank, key, 0, 0, &f3,
                                                 &a0_leaves, &b1_leaves, true, &b1);
  if (uncached.empty() || uncached != cached || advancing.empty() || advancing == uncached ||
      changed_b.empty() || changed_b == uncached || f0 != f1 || f0 != f2 || f0 != f3)
    return false;
  uint32_t null_factor = 0, explicit_factor = 0;
  const auto null_b = build_pair_plain_proof(a0, m, n, k, rank, key, 0, 0, &null_factor,
                                              nullptr, nullptr, true, nullptr);
  const auto explicit_a = build_pair_plain_proof(a0, m, n, k, rank, key, 0, 0,
                                                  &explicit_factor, nullptr, nullptr, true, &a0);
  uint32_t reference_null_factor = 0, reference_explicit_factor = 0;
  const auto reference_null = build_pair_plain_proof(a0, m, n, k, rank, key, 0, 0,
                                                      &reference_null_factor, nullptr, nullptr,
                                                      false, nullptr);
  const auto reference_explicit = build_pair_plain_proof(a0, m, n, k, rank, key, 0, 0,
                                                          &reference_explicit_factor, nullptr,
                                                          nullptr, false, &a0);
  return !null_b.empty() && null_b == explicit_a && null_factor == explicit_factor &&
      !reference_null.empty() && reference_null == reference_explicit &&
      reference_null_factor == reference_explicit_factor;
}

} // namespace mom_pearlhash

int main() {
  using namespace mom_pearlhash;
  if (!fixed_b_host_contract()) {
    std::fprintf(stderr, "PearlHash fixed-B host contract failed\n");
    return 18;
  }
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
    if (profile.m == 32 && profile.n == 32 && profile.k == 1024 && profile.rank == 128) {
      // Fixed baseline fixture: the default-false proof bytes must remain identical after adding
      // the optional positive family. The digest is BLAKE3 over the complete base64 proof string.
      static constexpr std::array<uint8_t, 32> expected_digest = {
          0xec, 0x1f, 0x2e, 0xa2, 0x91, 0x76, 0x44, 0xb1, 0x91, 0xfe, 0x0a,
          0x32, 0xa5, 0x7b, 0x63, 0xb0, 0xec, 0x46, 0x97, 0xbc, 0x4f, 0x5a,
          0x09, 0xfc, 0x60, 0x35, 0xeb, 0x6b, 0x4b, 0xbe, 0xed, 0x74};
      uint8_t digest[32];
      pearlhash_b3::b3(reinterpret_cast<const uint8_t*>(regenerated.data()), regenerated.size(),
                       nullptr, digest);
      if (regenerated.size() != 45080 || std::memcmp(digest, expected_digest.data(), 32) != 0) {
        std::fprintf(stderr, "PearlHash default-false proof bytes changed\n");
        return 7;
      }
    }
  }

  // Positive mapping is constrained to the signed-safe byte family, while false remains the
  // reference stream. The fixed vectors cover both 63 and 64 without hardware assumptions.
  std::set<int> positive_values;
  for (const uint32_t fixed_seed : {0u, 1u, 0x12345678u, 0xffffffffu}) {
    for (uint32_t idx = 0; idx < 4096; ++idx) {
      const int reference = static_cast<int>(gv(fixed_seed, idx, false));
      const int positive = static_cast<int>(gv(fixed_seed, idx, true));
      if (reference != static_cast<int>(gv(fixed_seed, idx)) || (positive != 63 && positive != 64)) {
        std::fprintf(stderr, "PearlHash gv mapping changed or escaped [63,64]\n");
        return 8;
      }
      positive_values.insert(positive);
    }
  }
  if (positive_values != std::set<int>({63, 64})) {
    std::fprintf(stderr, "PearlHash positive gv did not exercise both values\n");
    return 9;
  }
  for (const int value : {63, 64}) {
    for (int first = -32; first <= 31; ++first) {
      for (int second = -32; second <= 31; ++second) {
        const int noised = value + first - second;
        if (noised < 0 || noised > 127 || static_cast<int>(static_cast<int8_t>(noised)) != noised) {
          std::fprintf(stderr, "PearlHash positive noise escaped signed-safe [0,127]\n");
          return 10;
        }
      }
    }
  }

  // Cached leaves and regeneration must carry the same explicit positive mode and retain the
  // existing adjustment factor. A reference-mode proof with the same key must differ.
  const Profile positive_profile{32, 32, 1024, 128, 0, 8};
  const std::vector<CV> positive_a = leaf_cvs('A', seed, positive_profile.m, positive_profile.n,
                                              positive_profile.k, key, true);
  const std::vector<CV> positive_b = leaf_cvs('B', seed, positive_profile.m, positive_profile.n,
                                              positive_profile.k, key, true);
  uint32_t positive_regenerated_factor = 0, positive_cached_factor = 0;
  const std::string positive_regenerated = build_plain_proof(
      seed, positive_profile.m, positive_profile.n, positive_profile.k, positive_profile.rank, key,
      positive_profile.row, positive_profile.col,
      &positive_regenerated_factor, nullptr, nullptr, 16, 16, true);
  const std::string positive_cached = build_plain_proof(
      seed, positive_profile.m, positive_profile.n, positive_profile.k, positive_profile.rank, key,
      positive_profile.row, positive_profile.col,
      &positive_cached_factor, &positive_a, &positive_b, 16, 16, true);
  uint32_t positive_reference_factor = 0;
  const std::string reference = build_plain_proof(
      seed, positive_profile.m, positive_profile.n, positive_profile.k, positive_profile.rank, key,
      positive_profile.row, positive_profile.col, &positive_reference_factor,
      nullptr, nullptr, 16, 16, false);
  if (positive_regenerated.empty() || positive_regenerated != positive_cached ||
      positive_regenerated_factor != positive_cached_factor || positive_regenerated_factor != 262144 ||
      positive_regenerated == reference) {
    std::fprintf(stderr, "PearlHash positive cached/regenerated mode mismatch\n");
    return 11;
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

  // Pattern counts bind the job key before commitments/noise. The native HIP rectangle keeps
  // the same 256 selected cells, but must not reuse the square proof's key or serialization.
  for (const int rows : {16, 8, 4}) {
    const int cols = 256 / rows;
    uint8_t config[52], expected_config[52]{}, rectangle_key[32], expected_key[32], input[128];
    expected_config[1] = 8; // K2048, little endian
    expected_config[4] = 128;
    expected_config[9] = static_cast<uint8_t>(rows - 1);
    expected_config[15] = static_cast<uint8_t>(cols - 1);
    config_bytes(config, 2048, 128, rows, cols);
    if (std::memcmp(config, expected_config, 52)) return 2;
    derive_key(header, 2048, 128, rectangle_key, rows, cols);
    std::memcpy(input, header, 76);
    std::memcpy(input + 76, expected_config, 52);
    pearlhash_b3::b3(input, sizeof(input), nullptr, expected_key);
    if (std::memcmp(rectangle_key, expected_key, 32)) return 3;
    const auto a = leaf_cvs('A', seed, 32, 128, 2048, rectangle_key);
    const auto b = leaf_cvs('B', seed, 32, 128, 2048, rectangle_key);
    uint32_t f1 = 0, f2 = 0;
    const auto regenerated = build_plain_proof(seed, 32, 128, 2048, 128, rectangle_key,
                                                rows, cols, &f1, nullptr, nullptr, rows, cols);
    const auto cached = build_plain_proof(seed, 32, 128, 2048, 128, rectangle_key,
                                           rows, cols, &f2, &a, &b, rows, cols);
    if (regenerated.empty() || regenerated != cached || f1 != 524288 || f2 != f1) return 4;
    if (rows == 16) {
      uint8_t default_key[32];
      derive_key(header, 2048, 128, default_key);
      if (std::memcmp(default_key, rectangle_key, 32)) return 5;
    }
  }

  // CUDA's paired rows/columns change the public pattern, not its 256-cell
  // area. Bind that exact config into the key and retain one Merkle serializer.
  uint8_t pair_key[32], contiguous_key[32], pair_config[52], expected_pair_config[52]{};
  expected_pair_config[1] = 0x20; // K8192, little endian
  expected_pair_config[4] = 128;
  for (const int offset : {8, 14}) {
    expected_pair_config[offset + 1] = 1;
    expected_pair_config[offset + 2] = 3;
    expected_pair_config[offset + 3] = 7;
  }
  config_bytes_pattern(pair_config, 8192, 128, kPairDist8PatternBytes, kPairDist8PatternBytes);
  derive_key_pair_dist8(header, 8192, 128, pair_key);
  derive_key(header, 8192, 128, contiguous_key);
  uint8_t pair_input[128], expected_pair_key[32];
  std::memcpy(pair_input, header, 76);
  std::memcpy(pair_input + 76, expected_pair_config, 52);
  pearlhash_b3::b3(pair_input, sizeof(pair_input), nullptr, expected_pair_key);
  if (std::memcmp(pair_config, expected_pair_config, 52) ||
      std::memcmp(pair_key, expected_pair_key, 32) || !std::memcmp(pair_key, contiguous_key, 32))
    return 12;
  constexpr std::array<int, 16> pair_offsets = {
      0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57};
  std::array<unsigned, 64> selected{};
  for (const int base : {0, 2, 4, 6}) {
    std::vector<int> rows, cols;
    if (!pair_dist8_lists(base, base, 64, 64, rows, cols) || rows != cols || rows.size() != 16)
      return 13;
    for (size_t i = 0; i < pair_offsets.size(); ++i) {
      if (rows[i] != base + pair_offsets[i]) return 14;
      ++selected[static_cast<size_t>(rows[i])];
    }
  }
  if (std::any_of(selected.begin(), selected.end(), [](unsigned count) { return count != 1; }))
    return 15;
  for (const bool positive : {false, true}) {
    const auto a = leaf_cvs('A', seed, 64, 64, 8192, pair_key, positive);
    const auto b = leaf_cvs('B', seed, 64, 64, 8192, pair_key, positive);
    for (const int base : {0, 2, 6}) {
      uint32_t f1 = 0, f2 = 0;
      const auto regenerated = build_pair_plain_proof(
          seed, 64, 64, 8192, 128, pair_key, base, base, &f1, nullptr, nullptr, positive);
      const auto cached = build_pair_plain_proof(
          seed, 64, 64, 8192, 128, pair_key, base, base, &f2, &a, &b, positive);
      if (regenerated.empty() || regenerated != cached || f1 != 2097152 || f1 != f2)
        return 16;
    }
  }
  for (const int invalid : {-1, 1, 3, 5, 7, 8, 64, std::numeric_limits<int>::max()}) {
    if (!build_pair_plain_proof(seed, 64, 64, 8192, 128, pair_key, invalid, 0, &factor).empty() ||
        !build_pair_plain_proof(seed, 64, 64, 8192, 128, pair_key, 0, invalid, &factor).empty())
      return 17;
  }

  // Reject unsupported rectangles and coordinates before generating or indexing any leaves.
  for (const std::array<int, 4> invalid : {
           std::array<int, 4>{0, 16, 0, 0}, {16, 0, 0, 0}, {32, 8, 0, 0},
           {8, 32, -1, 0}, {8, 32, 0, -1}, {8, 32, 25, 0}, {8, 32, 0, 33},
           {16, 16, 17, 0}, {16, 16, 0, 49}, {4, 64, -1, 0}, {4, 64, 0, -1},
           {4, 64, 29, 0}, {4, 64, 0, 1}, {4, 64, 0, std::numeric_limits<int>::max()}}) {
    if (!build_plain_proof(seed, 32, 64, 2048, 128, key, invalid[2], invalid[3],
                           &factor, nullptr, nullptr, invalid[0], invalid[1]).empty()) {
      std::fprintf(stderr, "PearlHash accepted invalid proof geometry or coordinates\n");
      return 6;
    }
  }

  std::puts("PearlHash cached PlainProof matches regenerated proof (power-of-two/non-power-of-two)");
  return 0;
}
