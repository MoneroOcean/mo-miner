// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

// CPU-only oracle for the CTA coordinate equations in sycl/pearlhash/cuda-kernel.inc.
// It intentionally does not copy CUTE layouts or test the separate 64-bit B-offset path.

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

struct CtaCoordinate {
  int row;
  int column;
};

struct FixtureBlockIdx {
  int x;
  int y;
};

// The compiler-policy test writes swizzle.inc from cuda-kernel.inc between the block_m declaration
// and the cta_coord statement. That keeps this fixture tied to the actual source math while leaving
// CUTE types, matrix strides, and the separate 64-bit B-offset path out of the CPU oracle.
static CtaCoordinate source_swizzle(int block_x, int block_y, int m, int n, int k) {
  const FixtureBlockIdx blockIdx{block_x, block_y};
#include "swizzle.inc"
  return {block_m, block_n};
}

static void check_profile(const char* name, int m, int n, int k, bool expect_swizzle) {
  assert(m > 0 && n > 0 && m % 128 == 0 && n % 256 == 0);
  const int grid_m = m / 128;
  const int grid_n = n / 256;
  const size_t count = static_cast<size_t>(grid_m) * static_cast<size_t>(grid_n);
  std::vector<uint8_t> seen(count, 0);

  bool moved = false;
  const std::array<std::pair<int, int>, 4> corners = {{{0, 0},
                                                        {0, grid_n - 1},
                                                        {grid_m - 1, 0},
                                                        {grid_m - 1, grid_n - 1}}};
  for (const auto& [block_x, block_y] : corners) {
    const CtaCoordinate first = source_swizzle(block_x, block_y, m, n, k);
    const CtaCoordinate second = source_swizzle(block_x, block_y, m, n, k);
    assert(first.row == second.row && first.column == second.column);
    assert(first.row >= 0 && first.row < grid_m);
    assert(first.column >= 0 && first.column < grid_n);
  }

  for (int block_y = 0; block_y < grid_n; ++block_y) {
    for (int block_x = 0; block_x < grid_m; ++block_x) {
      const CtaCoordinate mapped = source_swizzle(block_x, block_y, m, n, k);
      assert(mapped.row >= 0 && mapped.row < grid_m);
      assert(mapped.column >= 0 && mapped.column < grid_n);
      assert(static_cast<int64_t>(mapped.row) * 128 < m);
      assert(static_cast<int64_t>(mapped.column) * 256 < n);
      if (mapped.row != block_x || mapped.column != block_y)
        moved = true;
      const size_t index = static_cast<size_t>(mapped.row) * static_cast<size_t>(grid_n) +
          static_cast<size_t>(mapped.column);
      assert(index < seen.size() && !seen[index]);
      seen[index] = 1;
      if (!expect_swizzle)
        assert(mapped.row == block_x && mapped.column == block_y);
    }
  }
  for (const uint8_t present : seen)
    assert(present);
  if (expect_swizzle)
    assert(moved);
  std::printf("PASS %s grid=%dx%d branch=%s ctas=%zu\n", name, grid_m, grid_n,
              expect_swizzle ? "swizzled" : "fallback", count);
}

int main() {
  // The two measured NVIDIA profiles: K8192 selects n=2, while K4096 selects n=16.
  check_profile("K8192/n2 full", 131072, 524288, 8192, true);
  check_profile("K4096/n16 full", 65536, 65536, 4096, true);

  // Non-divisible grid dimensions must retain the source's row-major fallback.
  check_profile("K8192 fallback grid_n", 2048, 3 * 256, 8192, false);
  check_profile("K4096 fallback grid_m", 15 * 128, 16 * 256, 4096, false);

  std::puts("PearlHash CUDA swizzle coordinate checks passed");
}
