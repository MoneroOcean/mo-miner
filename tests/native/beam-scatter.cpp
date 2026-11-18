#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

// Supply subgroup membership explicitly, without a SYCL/device/runtime dependency.
struct Group {
  unsigned id, groups, lane, actual;
  auto get_group_id() const { return std::array<unsigned, 1>{id}; }
  auto get_group_range() const { return std::array<unsigned, 1>{groups}; }
  auto get_local_id() const { return std::array<unsigned, 1>{lane}; }
  auto get_local_range() const { return std::array<unsigned, 1>{actual}; }
};
struct Item {
  Group group;
  auto get_sub_group() const { return group; }
};
struct alignas(16) BeamCompactRow { uint32_t value[16]; };
constexpr uint32_t SENTINEL = 0xffffffffu, SLOT_BITS = 12, NSLOTS = 2368;
constexpr unsigned ROW_U32 = sizeof(BeamCompactRow) / sizeof(uint32_t);

static void compact_flush(Item it, unsigned lsz, uint32_t* stage,
                          uint32_t* destination, BeamCompactRow* d_out) {
#include "beam_compact_flush.inc"
}
static void generic_flush(Item it, unsigned lsz, unsigned OUT_ST, uint32_t* stage,
                          uint32_t* dst, uint32_t* d_bucket) {
#include "beam_generic_flush.inc"
}

static uint32_t rng = 0x76fedaccu;
static unsigned next() {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return rng;
}
static std::vector<Group> topology(unsigned wg, unsigned width, unsigned arrangement) {
  const unsigned groups = (wg + width - 1) / width;
  std::vector<unsigned> order(wg);
  for (unsigned i = 0; i < wg; ++i) order[i] = i;
  if (arrangement == 1) {
    unsigned out = 0;
    for (unsigned lane = 0; lane < width; ++lane)
      for (unsigned group = 0; group < groups; ++group)
        if (group * width + lane < wg) order[out++] = group * width + lane;
    std::vector<unsigned> inverse(wg);
    for (unsigned i = 0; i < wg; ++i) inverse[order[i]] = i;
    order = inverse;
  } else if (arrangement == 2) {
    for (unsigned n = wg; n > 1; --n) std::swap(order[n - 1], order[next() % n]);
  }
  std::vector<Group> mapping(wg);
  std::vector<unsigned> seen(wg);
  for (unsigned group = 0; group < groups; ++group) {
    const unsigned actual = std::min(width, wg - group * width);
    for (unsigned lane = 0; lane < actual; ++lane) {
      const unsigned physical = order[group * width + lane];
      assert(physical < wg && !seen[physical]++);
      mapping[physical] = {group, groups, lane, actual};
    }
  }
  for (unsigned count : seen) assert(count == 1);
  return mapping;
}

static void check(bool compact, const std::vector<Group>& mapping, unsigned words,
                  bool drops, unsigned contiguous_width) {
  const unsigned wg = mapping.size();
  std::vector<uint32_t> stage(wg * words), dst(wg);
  for (unsigned row = 0; row < wg; ++row) {
    dst[row] = drops && row % 3 == 0 ? SENTINEL : row;
    for (unsigned word = 0; word < words; ++word)
      stage[row * words + word] = row * words + word + 1;
  }
  std::vector<unsigned> counts(wg * words), writer(wg * words, UINT32_MAX);
  std::vector<BeamCompactRow> compact_out(wg);
  std::vector<uint32_t> generic_out(wg * words);
  for (unsigned lid = 0; lid < wg; ++lid) {
    std::fill(generic_out.begin(), generic_out.end(), 0);
    for (auto& row : compact_out)
      for (auto& value : row.value) value = 0;
    if (compact)
      compact_flush({mapping[lid]}, wg, stage.data(), dst.data(), compact_out.data());
    else
      generic_flush({mapping[lid]}, wg, words, stage.data(), dst.data(), generic_out.data());
    const uint32_t* values = compact ? reinterpret_cast<const uint32_t*>(compact_out.data())
                                    : generic_out.data();
    // Isolated per-item output records every writer after the production WG barrier.
    for (unsigned cell = 0; cell < wg * words; ++cell) {
      if (!values[cell]) continue;
      assert(values[cell] == stage[cell]);
      ++counts[cell];
      writer[cell] = std::min(writer[cell], lid);
    }
  }
  for (unsigned row = 0; row < wg; ++row) {
    for (unsigned word = 0; word < words; ++word) {
      const unsigned cell = row * words + word;
      const bool dropped = dst[row] == SENTINEL;
      assert(counts[cell] == (dropped ? 0u : 1u));
      if (!dropped && contiguous_width) {
        const unsigned groups = wg / contiguous_width;
        assert(writer[cell] == (row % groups) * contiguous_width + word % contiguous_width);
      }
    }
  }
}

int main() {
  unsigned cases = 0, contiguous_full = 0;
  for (unsigned wg : {16u, 24u, 31u, 32u, 48u, 64u, 85u, 96u, 128u, 256u}) {
    for (unsigned width : {4u, 8u, 16u, 32u, 64u}) {
      for (unsigned arrangement : {0u, 1u, 2u}) {
        const unsigned ordinary = arrangement == 0 && wg % width == 0 ? width : 0;
        for (bool drops : {false, true}) {
          for (unsigned words : {4u, 15u, 16u, 18u}) {
            const auto mapping = topology(wg, width, arrangement);
            for (bool compact : {false, true}) {
              if (compact && words != ROW_U32) continue;
              check(compact, mapping, words, drops, ordinary);
              ++cases;
              if (ordinary) ++contiguous_full;
            }
          }
        }
      }
    }
  }
  std::printf("Beam scatter: %u source-extracted ownership cases, %u full-SG writer checks; "
              "complete values copied exactly once\n", cases, contiguous_full);
}
