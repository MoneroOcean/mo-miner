// CPU-only oracle for the producer-packed rank-128 CUDA matrix layout.
// The CUTE map below is reconstructed independently from lane/register ownership.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "offset.inc"

namespace {

constexpr std::uint64_t kTileBytes = 64 * 32;

[[noreturn]] static void fail(const char* message) {
  throw std::runtime_error(message);
}

static void require(bool condition, const char* message) {
  if (!condition)
    fail(message);
}

struct CuteMap {
  std::array<std::uint16_t, kTileBytes> a{};
  std::array<std::uint16_t, kTileBytes> b{};
};

static std::uint16_t cute_source_offset(bool is_a, int lane, int reg) {
  const int source_base = (lane / 4) * 32 + (lane % 4) * 4;
  const int i0 = reg % 4;
  const int i1 = (reg / 4) % 2;
  const int packet_group = reg / 16;
  const int source = is_a
      ? source_base + i0 + i1 * 256 + ((reg / 8) % 2) * 16 + packet_group * 512
      : source_base + i0 + i1 * 16 + (reg / 8) * 256;
  require(source >= 0 && source < static_cast<int>(kTileBytes),
          "independent CUTE source map escaped its tile");
  return static_cast<std::uint16_t>(source);
}

static std::uint16_t cute_packet_offset(int lane, int reg) {
  const int packed = (reg / 16) * 512 + lane * 16 + reg % 16;
  require(packed >= 0 && packed < static_cast<int>(kTileBytes),
          "independent CUTE packet map escaped its tile");
  return static_cast<std::uint16_t>(packed);
}

static CuteMap make_cute_map() {
  CuteMap map;
  std::array<bool, kTileBytes> source_a{}, source_b{}, packed{};
  for (int lane = 0; lane < 32; ++lane) {
    for (int reg = 0; reg < 64; ++reg) {
      const std::uint16_t packed_offset = cute_packet_offset(lane, reg);
      const std::uint16_t source_offset_a = cute_source_offset(true, lane, reg);
      const std::uint16_t source_offset_b = cute_source_offset(false, lane, reg);
      require(!source_a[source_offset_a] && !source_b[source_offset_b] &&
                  !packed[packed_offset],
              "independent CUTE map aliases a source or packet byte");
      source_a[source_offset_a] = source_b[source_offset_b] = packed[packed_offset] = true;
      map.a[source_offset_a] = map.b[source_offset_b] = packed_offset;
    }
  }
  for (std::size_t i = 0; i < kTileBytes; ++i)
    require(source_a[i] && source_b[i] && packed[i], "independent CUTE map has a hole");
  return map;
}

template <bool IsA>
static std::uint64_t actual_offset(std::uint64_t row, std::uint64_t depth, int k) {
  require(row <= std::numeric_limits<std::size_t>::max() &&
              depth <= std::numeric_limits<std::size_t>::max(),
          "coordinate does not fit size_t");
  return static_cast<std::uint64_t>(nvidia_vector128_offset<IsA>(
      static_cast<std::size_t>(row), static_cast<std::size_t>(depth), k));
}

template <bool IsA>
static void check_tile_bijection() {
  std::array<bool, kTileBytes> seen{};
  for (std::uint64_t row = 0; row < 64; ++row) {
    for (std::uint64_t depth = 0; depth < 32; ++depth) {
      const std::uint64_t offset = actual_offset<IsA>(row, depth, 32);
      require(offset < kTileBytes && !seen[offset], "producer tile map aliases a byte");
      seen[offset] = true;
    }
  }
  for (bool byte : seen)
    require(byte, "producer tile map leaves a byte uninitialized");
}

template <bool IsA>
static std::uint64_t cute_expected_offset(const CuteMap& map, std::uint64_t row,
                                         std::uint64_t depth, int k) {
  const std::uint64_t tile = (row / 64) * static_cast<std::uint64_t>(k / 32) + depth / 32;
  const std::uint64_t source = (row % 64) * 32 + depth % 32;
  const std::uint16_t local = IsA ? map.a[source] : map.b[source];
  return tile * kTileBytes + local;
}

template <bool IsA>
static std::uint64_t canonical_value(std::uint64_t row, std::uint64_t depth) {
  return (IsA ? UINT64_C(0xA000000000000000) : UINT64_C(0xB000000000000000)) |
         ((row & UINT64_C(0x0000FFFFFFFFFFFF)) << 13) | (depth & UINT64_C(0x1FFF));
}

template <bool IsA>
static void check_scatter_gather(const CuteMap& map) {
  constexpr int k = 8192;
  constexpr std::uint64_t rows_per_block = 64;
  constexpr std::uint64_t block_bytes = rows_per_block * k;
  constexpr std::array<std::uint64_t, 4> block_bases = {
      0, UINT64_C(262144) - 64, UINT64_C(524288) - 64, UINT64_C(1048576) - 64};
  std::uint64_t checked = 0;
  for (const std::uint64_t block_base : block_bases) {
    const std::uint64_t packed_base = (block_base / 64) * (k / 32) * kTileBytes;
    std::vector<std::uint64_t> actual(block_bytes), expected(block_bytes);
    std::vector<std::uint8_t> actual_seen(block_bytes), expected_seen(block_bytes);
    for (std::uint64_t local_row = 0; local_row < rows_per_block; ++local_row) {
      for (std::uint64_t depth = 0; depth < static_cast<std::uint64_t>(k); ++depth) {
        const std::uint64_t row = block_base + local_row;
        const std::uint64_t packed = actual_offset<IsA>(row, depth, k);
        const std::uint64_t cute = cute_expected_offset<IsA>(map, row, depth, k);
        require(packed == cute && packed >= packed_base && packed < packed_base + block_bytes,
                "producer scatter disagrees with the independent CUTE map");
        const std::size_t actual_index = static_cast<std::size_t>(packed - packed_base);
        const std::size_t expected_index = static_cast<std::size_t>(cute - packed_base);
        require(!actual_seen[actual_index] && !expected_seen[expected_index],
                "producer scatter aliases a packed value slot");
        actual_seen[actual_index] = expected_seen[expected_index] = 1;
        const std::uint64_t value = canonical_value<IsA>(row, depth);
        actual[actual_index] = value;
        expected[expected_index] = value;
        ++checked;
      }
    }
    require(actual == expected, "producer scatter differs from CUTE expected values");
    for (std::size_t i = 0; i < block_bytes; ++i)
      require(actual_seen[i] && expected_seen[i], "producer scatter left a slot empty");
    for (std::uint64_t local_row = 0; local_row < rows_per_block; ++local_row)
      for (std::uint64_t depth = 0; depth < static_cast<std::uint64_t>(k); ++depth)
        require(actual[static_cast<std::size_t>(actual_offset<IsA>(block_base + local_row, depth, k) -
                                               packed_base)] == canonical_value<IsA>(
                                                   block_base + local_row, depth),
                "producer gather changed a logical value");
  }
  std::printf("SCATTER_GATHER is_a=%d values=%llu PASS\n", IsA ? 1 : 0,
              static_cast<unsigned long long>(checked));
}

template <bool IsA>
static void check_shape_and_boundaries() {
  constexpr int k = 2048;
  constexpr std::uint64_t rows = 524288;
  constexpr std::array<std::uint64_t, 8> sample_rows = {0, 1, 7, 63, 64, 127, rows - 2, rows - 1};
  constexpr std::array<std::uint64_t, 8> sample_depths = {0, 1, 3, 4, 31, 32, 1023, 2047};
  for (const std::uint64_t row : sample_rows)
    for (const std::uint64_t depth : sample_depths)
      require(actual_offset<IsA>(row, depth, k) < rows * static_cast<std::uint64_t>(k),
              "full measured shape escaped its operand");

  constexpr std::uint64_t two_gib = UINT64_C(1) << 31;
  constexpr std::uint64_t four_gib = UINT64_C(1) << 32;
  constexpr std::uint64_t row_two = two_gib / k;
  constexpr std::uint64_t row_four = four_gib / k;
  require(actual_offset<IsA>(row_two, 0, k) == two_gib,
          "exact 2 GiB producer offset changed");
  require(actual_offset<IsA>(row_four, 0, k) == four_gib,
          "exact 4 GiB producer offset changed");
  for (const std::uint64_t row : {row_four - 64, row_four - 1, row_four,
                                  row_four + 1, row_four + 63}) {
    for (const std::uint64_t depth : {0ULL, 1ULL, 3ULL, 4ULL, 31ULL, 32ULL, 2047ULL}) {
      const std::uint64_t offset = actual_offset<IsA>(row, depth, k);
      require(offset >= four_gib - 64 * k && offset < four_gib + 64 * k,
              "4 GiB boundary offset escaped its neighboring rows");
    }
  }
  constexpr int wide_k = 8192;
  constexpr std::uint64_t wide_rows = UINT64_C(1048576);
  constexpr std::uint64_t wide_bytes = wide_rows * static_cast<std::uint64_t>(wide_k);
  // Check the offset helper's numeric range, not admission of a mining profile at this size.
  for (const std::uint64_t boundary : {UINT64_C(1) << 31, UINT64_C(1) << 32,
                                       UINT64_C(1) << 33}) {
    const std::uint64_t row = boundary / wide_k;
    require(actual_offset<IsA>(row, 0, wide_k) == boundary &&
                actual_offset<IsA>(row - 1, wide_k - 1, wide_k) == boundary - 1,
            "8 GiB packed operand crossed a boundary incorrectly");
    if (row < wide_rows)
      require(actual_offset<IsA>(row, wide_k - 1, wide_k) < wide_bytes,
              "8 GiB packed operand escaped its extent");
  }
  std::printf("BOUNDARIES is_a=%d 2G=0x%llx 4G=0x%llx PASS\n", IsA ? 1 : 0,
              static_cast<unsigned long long>(two_gib), static_cast<unsigned long long>(four_gib));
}

static void check_32bit_negative_control() {
  constexpr std::uint64_t four_gib = UINT64_C(1) << 32;
  bool rejected = false;
  try {
    if (four_gib > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error("offset does not fit uint32_t");
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  require(rejected && static_cast<std::uint32_t>(four_gib) == 0,
          "32-bit truncation control did not reject 4 GiB");
  std::puts("CAST32_NEGATIVE_CONTROL rejected=PASS");
}

// Match the packed mainloop's old packet-index order without ever forming its
// final one-past numeric prefetch as a pointer. Cover the terminal 4-GiB block.
static void check_rolling_prefetch_sequence() {
  constexpr std::uint64_t packets = 256;
  constexpr std::uint64_t packet_end = packets * kTileBytes;
  constexpr std::uint64_t b_bytes = (UINT64_C(524288) / 64 - 1) * packet_end;
  std::uint64_t odd_bytes = kTileBytes;
  std::uint64_t last_byte = 0;
  unsigned loads = 1; // The prime load consumes packet zero.
  for (std::uint64_t tile = 0; tile < 64; ++tile) {
    for (std::uint64_t pair = 0; pair < 2; ++pair) {
      const std::uint64_t odd = tile * 4 + pair * 2 + 1;
      require(odd_bytes == odd * kTileBytes && odd_bytes < packet_end,
              "rolling odd prefetch changed its packet or escaped the block");
      last_byte = b_bytes + odd_bytes + kTileBytes - 1;
      ++loads;
      const std::uint64_t even_bytes = odd_bytes + kTileBytes;
      require((even_bytes < packet_end) == (odd + 1 < packets),
              "rolling even guard changed the terminal prefetch");
      if (even_bytes < packet_end) {
        require(even_bytes == (odd + 1) * kTileBytes, "rolling even packet changed");
        last_byte = b_bytes + even_bytes + kTileBytes - 1;
        ++loads;
      }
      odd_bytes += 2 * kTileBytes;
    }
  }
  require(loads == packets && odd_bytes == packet_end + kTileBytes,
          "rolling prefetch count or final numeric recurrence changed");
  require(last_byte == (UINT64_C(1) << 32) - 1, "rolling B endpoint is not 4 GiB - 1");
  std::puts("ROLLING_PREFETCH_SEQUENCE packets=256 B_end=0xffffffff PASS");
}

} // namespace

int main() {
  try {
    const CuteMap map = make_cute_map();
    check_tile_bijection<true>();
    check_tile_bijection<false>();
    check_shape_and_boundaries<true>();
    check_shape_and_boundaries<false>();
    check_scatter_gather<true>(map);
    check_scatter_gather<false>(map);
    check_32bit_negative_control();
    check_rolling_prefetch_sequence();
    std::puts("PRODUCER_OFFSET_TEST PASS");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "PRODUCER_OFFSET_TEST FAIL: %s\n", error.what());
    return 1;
  }
}
