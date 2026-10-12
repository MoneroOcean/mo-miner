// Independent CPU arithmetic oracle; opaque WMMA fragments need no guessed lane/cell mapping.
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

static uint32_t rotl13(uint32_t value) { return (value << 13) | (value >> 19); }

static void check_alignment() {
  for (uintptr_t residue = 0; residue < 32; residue += 4) {
    const uintptr_t raw = 4096 + residue;
    const uintptr_t aligned = (raw + 31) & ~uintptr_t{31};
    assert(aligned % 32 == 0 && aligned >= raw);
    assert(aligned - raw <= 28);
    assert(aligned + 512 * sizeof(uint32_t) <= raw + 520 * sizeof(uint32_t));
    assert((aligned + 128 * sizeof(uint32_t)) % 32 == 0);
  }
}

static void check_cache_remap(uint32_t rows, uint32_t cols, uint32_t cache_block) {
  const bool blocked = cache_block > 1 &&
      uint64_t(cache_block) * cache_block <= std::numeric_limits<uint32_t>::max() &&
      rows % cache_block == 0 && cols % cache_block == 0;
  const uint64_t count = uint64_t(rows) * cols;
  const bool exhaustive = count <= (1U << 22);
  std::vector<bool> seen(exhaustive ? static_cast<size_t>(count) : 0);
  const auto check = [&](uint32_t g) {
    uint32_t row = g / cols, col = g % cols;
    if (blocked) {
      const uint32_t area = cache_block * cache_block, block = g / area, inner = g % area;
      row = (block / (cols / cache_block)) * cache_block + inner / cache_block;
      col = (block % (cols / cache_block)) * cache_block + inner % cache_block;
      const uint64_t inverse =
          (uint64_t(row / cache_block) * (cols / cache_block) + col / cache_block) * area +
          (row % cache_block) * cache_block + col % cache_block;
      assert(inverse == g);
    }
    assert(row < rows && col < cols);
    if (exhaustive) {
      const size_t index = static_cast<size_t>(row) * cols + col;
      assert(!seen[index]);
      seen[index] = true;
    }
  };
  if (exhaustive) {
    for (uint32_t g = 0; g < count; ++g) check(g);
    for (bool present : seen) assert(present);
  } else {
    // Full production shapes are bounded samples plus the analytic inverse, not a huge CPU sweep.
    for (uint32_t g : {0U, 1U, cols - 1, cols, static_cast<uint32_t>(count / 2),
                       static_cast<uint32_t>(count - 1)}) check(g);
  }
}

static void check_matrix(int k, int rank, bool extreme, int permutation = 1) {
  std::array<uint32_t, 256> cumulative{}, reference{};
  std::array<uint32_t, 16> transcript{}, expected{};
  for (int boundary = 0; boundary < k / rank; ++boundary) {
    uint32_t folded = 0, reference_folded = 0;
    std::array<bool, 256> seen{};
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        int32_t exact = 0;
        for (int depth = boundary * rank; depth < (boundary + 1) * rank; ++depth) {
          const int a = extreme ? -128 : ((row * 67 + depth * 29 + (depth >> 3)) & 255) - 128;
          const int b = extreme ? -128 : ((col * 43 + depth * 19 + 11) & 255) - 128;
          exact += a * b;
        }
        reference[row * 16 + col] += static_cast<uint32_t>(exact);
        reference_folded ^= reference[row * 16 + col];
      }
    }
    for (int lane = 0; lane < 32; ++lane) {
      for (int cell = 0; cell < 8; ++cell) {
        // Test multiple stable bijections, not an undocumented WMMA fragment layout.
        const int slot = lane * 8 + cell;
        const int index = (slot * permutation + 19) & 255;
        const int row = index / 16, col = index % 16;
        assert(!seen[index]);
        seen[index] = true;
        float partial = 0;
        int32_t exact = 0;
        for (int depth = boundary * rank; depth < (boundary + 1) * rank; ++depth) {
          const int a = extreme ? -128 : ((row * 67 + depth * 29 + (depth >> 3)) & 255) - 128;
          const int b = extreme ? -128 : ((col * 43 + depth * 19 + 11) & 255) - 128;
          const int product = a * b;
          assert(static_cast<float>(product) == product);
          partial += static_cast<float>(product);
          exact += product;
        }
        assert(partial == exact && std::abs(exact) <= rank * 16384);
        cumulative[slot] += static_cast<uint32_t>(static_cast<int32_t>(partial));
        assert(cumulative[slot] == reference[index]);
        folded ^= cumulative[slot];
      }
    }
    for (bool present : seen) assert(present);
    const int slot = boundary % 16;
    transcript[slot] = rotl13(transcript[slot]) ^ folded;
    expected[slot] = rotl13(expected[slot]) ^ reference_folded;
  }
  assert(transcript == expected);
}

int main() {
  check_alignment();
  for (uint32_t block : {0U, 1U, 3U, 8U, 16U, UINT32_MAX}) check_cache_remap(8, 8, block);
  for (uint32_t block : {0U, 1U, 64U, 128U}) check_cache_remap(128, 512, block);
  check_cache_remap(1024, 4096, 64); // Pilot tile grid, exhaustive within a fixed 4-Mtile bound.
  check_cache_remap(8192, 32768, 64); // Full profile tile grid, sampled/inverse-checked.
  for (int rank : {128, 256, 512, 1024}) {
    for (int permutation : {1, 73, 255})
      check_matrix(16 * rank, rank, false, permutation);
  }
  // Two packed int8 words per lane stage all 256 canonical half operands exactly once.
  std::array<bool, 256> staged{};
  for (int lane = 0; lane < 32; ++lane) {
    for (int part = 0; part < 2; ++part) {
      const int word = lane + 32 * part;
      const int row_or_col = word / 4, depth4 = word % 4;
      for (int byte = 0; byte < 4; ++byte) {
        const int index = row_or_col * 16 + depth4 * 4 + byte;
        assert(!staged[index]);
        staged[index] = true;
      }
    }
  }
  for (bool present : staged) assert(present);
  check_matrix(16384, 1024, true); // Exactly 2^24 remains representable at each rank boundary.
  check_matrix(2112, 128, false); // Legal K%rank tail is excluded, as in the verifier.
  check_matrix(8256, 512, false);
  assert(static_cast<uint32_t>(static_cast<int32_t>(-1.0f)) == UINT32_MAX);
  assert(static_cast<uint32_t>(static_cast<int32_t>(-16646144.0f)) == uint32_t(-16646144));
  uint32_t cumulative = 0;
  int64_t exact = 0;
  for (int step = 0; step < 1024; ++step) {
    const float partial = step % 3 ? 16777216.0f : -16646144.0f;
    cumulative += static_cast<uint32_t>(static_cast<int32_t>(partial));
    exact += static_cast<int32_t>(partial);
    assert(cumulative == static_cast<uint32_t>(exact));
  }
  assert(16777216.0f + 1.0f == 16777216.0f); // Why cumulative C must remain integer.
}
