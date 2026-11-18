// Copyright GNU GPLv3 (c) 2026 MoneroOcean
// CPU-only regression for the pair rank-128 dense GEMM/hash coordinate ABI.
//
// The runner generates pair_dense_mapping_generated.h from the current kernel
// source. It contains the actual producer store, hash-side inverse, and
// checkpoint recurrence, rather than hand-written copies. The fixture independently checks the
// producer/index bijection across both axes, every slot, and the terminal B
// warp tile.

#include "pair_dense_mapping_generated.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#ifndef PAIR_MAPPING_SOURCE_LABEL
#define PAIR_MAPPING_SOURCE_LABEL "unknown"
#endif

namespace {

bool require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL source=" << PAIR_MAPPING_SOURCE_LABEL << ": " << message << '\n';
    return false;
  }
  return true;
}

using Word = std::uint32_t;

static Word reference_checkpoint(const std::array<Word, 128>& values) {
  // Every accumulator cell contributes once; XOR is independent of its physical ordering.
  Word lane_part = 0;
  for (const Word value : values)
    lane_part ^= value;
  return lane_part;
}

static std::array<Word, 128> checkpoint_values(int family, int lane,
                                                int checkpoint) {
  std::array<Word, 128> values{};
  for (int index = 0; index < 128; ++index) {
    Word x = static_cast<Word>(0x9e3779b9u * static_cast<Word>(index + 1)) ^
             static_cast<Word>(0x85ebca6bu * static_cast<Word>(lane + 3)) ^
             static_cast<Word>(0xc2b2ae35u * static_cast<Word>(checkpoint + 5));
    if (family == 1)
      x ^= 0x80000000u;
    if (family == 2)
      x = static_cast<Word>((x << 1) ^ 0x80000001u);
    if (family == 3) {
      static constexpr Word boundary[8] = {
          0xffffffffu, 0x80000000u, 0x7fffffffu, 0u,
          0xaaaaaaaau, 0x55555555u, 0x00000001u, 0xfffffffeu};
      x = boundary[index & 7] ^ static_cast<Word>(
          static_cast<Word>(lane) * 0x9e3779b9u +
          static_cast<Word>(checkpoint) * 0x7f4a7c15u +
          static_cast<Word>(index));
    }
    values[index] = x;
  }
  return values;
}

static bool check_source_checkpoint() {
  for (int family = 0; family < 4; ++family) {
    for (int checkpoint = 0; checkpoint < 16; ++checkpoint) {
      for (int lane = 0; lane < 32; ++lane) {
        const auto values = checkpoint_values(family, lane, checkpoint);
        std::array<std::uint8_t, 128> touched{};
        const Word actual = pair_mapping_actual::actual_checkpoint(values, touched);
        const Word expected = reference_checkpoint(values);
        if (!require(actual == expected,
                     "source checkpoint XOR differs from independent oracle"))
          return false;
        for (int index = 0; index < 128; ++index) {
          if (!require(touched[index] == 1,
                       "source checkpoint did not touch each of 128 accumulator cells"))
            return false;
        }
      }
    }
  }
  return true;
}

bool run() {
  // Four CTAs in M and two in N exercise multiple warp blocks on both axes;
  // the last warp tile is the terminal B-side tile at column n - 64.
  constexpr uint32_t m = 512;
  constexpr uint32_t n = 512;
  constexpr uint64_t warp_tiles_m = m / 64u;
  constexpr uint64_t warp_tiles_n = n / 64u;
  constexpr uint64_t tile_count = (uint64_t(m) / 16u) * (uint64_t(n) / 16u);
  constexpr uint64_t cta_m = m / 128u;
  constexpr uint64_t cta_n = n / 256u;
  std::vector<uint8_t> seen(tile_count, 0);
  uint64_t records = 0;

  for (uint32_t block_m = 0; block_m < cta_m; ++block_m) {
    for (uint32_t block_n = 0; block_n < cta_n; ++block_n) {
      for (uint32_t warp_m = 0; warp_m < 2; ++warp_m) {
        for (uint32_t warp_n = 0; warp_n < 4; ++warp_n) {
          for (uint32_t slot = 0; slot < 16; ++slot) {
            // This is one of the producer lanes satisfying (lane & 4) == 0;
            // the source's slot expression must recover this exact slot.
            const uint32_t lane = (slot / 4u) * 8u + (slot % 4u);
            const uint64_t tile = pair_mapping_actual::actual_dense_store_index(
                block_m, warp_m, block_n, warp_n, n, lane);
            if (!require(tile < tile_count, "producer index escaped transcript bounds"))
              return false;
            if (!require(!seen[tile], "producer emitted a duplicate dense tile index"))
              return false;
            seen[tile] = 1;
            ++records;

            const auto actual = pair_mapping_actual::actual_hash_inverse(tile, n, 0);
            // Derive the expected coordinate from the physical producer, not
            // from its encoded index: even a bijective wrong permutation must fail.
            const auto expected = pair_mapping_actual::Point{
                (uint64_t(block_m) * 2u + warp_m) * 64u + (slot / 4u) * 2u,
                (uint64_t(block_n) * 4u + warp_n) * 64u + (slot % 4u) * 2u};
            if (!require(actual.row == expected.row && actual.col == expected.col,
                         "hash inverse disagrees with dense producer coordinate"))
              return false;
            if (!require(actual.row + 57u < m && actual.col + 57u < n,
                         "pair proof span escaped matrix bounds"))
              return false;
          }
        }
      }
    }
  }

  if (!require(records == tile_count, "producer did not cover every dense transcript tile"))
    return false;
  for (uint64_t tile = 0; tile < tile_count; ++tile) {
    if (!require(seen[tile] != 0, "dense transcript tile was never produced"))
      return false;
  }

  // Explicit terminal-B assertion: the final warp tile's slot 0 must start
  // at (m-64,n-64), not at a row-major 16x16 interpretation of the index.
  const uint64_t terminal_tile = (warp_tiles_m * warp_tiles_n - 1u) * 16u;
  const auto terminal = pair_mapping_actual::actual_hash_inverse(terminal_tile, n, 0);
  if (!require(terminal.row == m - 64u && terminal.col == n - 64u,
               "terminal B coordinate was not recovered exactly"))
    return false;

  // The last full-worker slice has a nonzero row offset and touches the final
  // byte of the 4-GiB B matrix. Keep all byte arithmetic explicitly 64-bit.
  const uint64_t last = pair_mapping_actual::actual_dense_store_index(
      63, 1, 2047, 3, 524288, 27);
  const auto full_end = pair_mapping_actual::actual_hash_inverse(last, 524288, 122880);
  if (!require(full_end.row + 57u == 131071u && full_end.col + 57u == 524287u &&
                   (full_end.col + 57u) * 8192u + 8191u == 0xffffffffull,
               "last full-worker slice or 4-GiB B endpoint was decoded incorrectly"))
    return false;

  if (!check_source_checkpoint())
    return false;

  std::cout << "PASS pair_dense_mapping source=" << PAIR_MAPPING_SOURCE_LABEL
            << " cta_grid=" << cta_m << "x" << cta_n
            << " warp_tiles=" << warp_tiles_m << "x" << warp_tiles_n
            << " slots=16 records=" << records
            << " terminal=" << terminal.row << "," << terminal.col
            << " checkpoint=4x16x32x128 wrap=PASS\n";
  return true;
}

}  // namespace

int main() {
  return run() ? 0 : 1;
}
