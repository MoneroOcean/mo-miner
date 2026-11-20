// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <array>
#include <cstdint>

namespace mom_pearlhash {

// One axis of a verifier-canonical opening pattern (pearl_verify pattern_from_list): up to three
// (stride, length) dimensions, innermost first; each stride is a multiple of the inner period and
// unused outer dimensions repeat that period with length 1. Device kernels map tiles through it.
struct ProofAxis {
  uint32_t stride[3], length[3];

  // Contiguous, or `length0` indices `stride0` apart repeated `length1` times `stride1` apart.
  static constexpr ProofAxis make(uint32_t stride0, uint32_t length0, uint32_t stride1 = 0,
                                  uint32_t length1 = 1) {
    const uint32_t inner = stride0 * length0, outer = length1 > 1 ? stride1 : inner;
    return {{stride0, outer, outer * length1}, {length0, length1, 1}};
  }
  constexpr uint32_t count() const { return length[0] * length[1] * length[2]; }
  constexpr uint32_t period(int d) const { return stride[d] * length[d]; }
  // Offset of the i-th opened index from the origin; the lists are sorted and distinct.
  constexpr uint32_t component(uint32_t i) const {
    return i % length[0] * stride[0] + i / length[0] % length[1] * stride[1] +
           i / (length[0] * length[1]) * stride[2];
  }
  constexpr uint32_t span() const { return component(count() - 1) + 1; }
  // The verifier's offset_is_valid: below every stride within that dimension's period.
  constexpr bool canonical(uint32_t origin) const {
    for (int d = 2; d >= 0; --d) {
      origin %= period(d);
      if (origin >= stride[d])
        return false;
    }
    return true;
  }
  // The t-th canonical origin in increasing order. For these tiling patterns the origins'
  // openings partition every full period, so tiles(extent) covers each index exactly once.
  constexpr uint32_t origin(uint32_t t) const {
    uint32_t value = 0, unit = 1;
    for (int d = 0; d < 3; ++d) {
      const uint32_t radix = stride[d] / unit;
      value += t % radix * unit;
      t /= radix;
      unit = period(d);
    }
    return value + t * unit;
  }
  constexpr uint32_t origins_per_period() const {
    return stride[0] * (stride[1] / period(0)) * (stride[2] / period(1));
  }
  // Openings that fit inside full periods of an axis of `extent` indices.
  constexpr uint32_t tiles(uint32_t extent) const {
    return extent / period(2) * origins_per_period();
  }
  constexpr bool valid(int origin, int extent) const {
    return count() > 0 && origin >= 0 && extent > 0 && canonical(static_cast<uint32_t>(origin)) &&
           uint64_t(origin) + span() <= uint64_t(extent);
  }
  // Config-block encoding: (stride / inner period - 1, length - 1) per dimension.
  constexpr std::array<uint8_t, 6> bytes() const {
    std::array<uint8_t, 6> out{};
    uint32_t inner = 1;
    for (int d = 0; d < 3; ++d) {
      out[2 * d] = static_cast<uint8_t>(stride[d] / inner - 1);
      out[2 * d + 1] = static_cast<uint8_t>(length[d] - 1);
      inner = period(d);
    }
    return out;
  }
  constexpr bool operator==(const ProofAxis& other) const {
    for (int d = 0; d < 3; ++d)
      if (stride[d] != other.stride[d] || length[d] != other.length[d])
        return false;
    return true;
  }
};
// C++17 std::array comparison is not constexpr; pattern byte checks run at compile time.
constexpr bool same_pattern(const std::array<uint8_t, 6>& a, const std::array<uint8_t, 6>& b) {
  for (int i = 0; i < 6; ++i)
    if (a[i] != b[i])
      return false;
  return true;
}
// Native CUDA rank-128 pairs: the verifier's canonical 8-by-8-by-2 pattern on both axes.
inline constexpr ProofAxis kPairDist8Axis = ProofAxis::make(1, 2, 8, 8);
// CUDA compact 24x8 openings: rows in adjacent pairs 16 apart, columns pairs 8 apart, 64 between.
inline constexpr ProofAxis kCompactRows = ProofAxis::make(1, 2, 16, 12);
inline constexpr ProofAxis kCompactCols = ProofAxis::make(8, 2, 64, 4);
// gfx12 wave64 WMMA (13x4 16x16 tiles per wave): one lane's accumulators are rows {4q+e+16i} and
// columns {c+16j}, so each lane opens its own 52x4 proof and folds it without cross-lane reduction.
inline constexpr ProofAxis kAmdLaneRows = ProofAxis::make(1, 4, 16, 13);
inline constexpr ProofAxis kAmdLaneCols = ProofAxis::make(16, 4);
static_assert(kPairDist8Axis.span() == 58 && kPairDist8Axis.origin(5) == 66 &&
              kPairDist8Axis.canonical(6) && !kPairDist8Axis.canonical(8) &&
              same_pattern(kPairDist8Axis.bytes(), std::array<uint8_t, 6>{0, 1, 3, 7, 0, 0}));
static_assert(kAmdLaneRows.count() == 52 && kAmdLaneRows.origin(5) == 212 &&
              kAmdLaneRows.tiles(524160) == 10080 && kAmdLaneCols.origin(17) == 65 &&
              kAmdLaneCols.tiles(524288) == 131072 &&
              same_pattern(kAmdLaneRows.bytes(), std::array<uint8_t, 6>{0, 3, 3, 12, 0, 0}) &&
              same_pattern(kAmdLaneCols.bytes(), std::array<uint8_t, 6>{15, 3, 0, 0, 0, 0}));
// Contiguous 16x16 openings: the portable and legacy native proof geometry.
inline constexpr ProofAxis kSquareAxis = ProofAxis::make(1, 16);

} // namespace mom_pearlhash
