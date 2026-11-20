// CPU model of the fused HIP kernel's lane-local transcripts. The driver extracts the production
// checkpoint fold, lane LDS row, opening coordinates and tail guard from hip-kernel.inc. This
// oracle recomputes every published opening the way pearl_verify does -- XOR of the opened cells
// at each rank checkpoint, rotl13-folded into 16 slots -- from the host's own opening axes.
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "proof-axis.h"

#ifndef PEARLHASH_ER
#define PEARLHASH_ER 13
#endif
#ifndef PEARLHASH_EC
#define PEARLHASH_EC 4
#endif
#ifndef PEARLHASH_K
#define PEARLHASH_K 2048
#endif
#ifndef PEARLHASH_RANK
#define PEARLHASH_RANK 128
#endif

using mom_pearlhash::kAmdLaneCols;
using mom_pearlhash::kAmdLaneRows;
using v8i = std::array<int32_t, 4>;
static_assert(kAmdLaneRows.length[1] == PEARLHASH_ER && kAmdLaneCols.length[0] == PEARLHASH_EC,
              "kernel tiles must match the host's lane openings");

[[maybe_unused]] static uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

static uint32_t next_random(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

int main() {
  try {
    // Two column groups; row group 1 is a tail with only 8 of 13 tiles. The slice starts at a
    // whole row group, as hip-jit.inc launches it.
    constexpr int group_rows = PEARLHASH_ER * 16, group_cols = PEARLHASH_EC * 16;
    const int m = group_rows + 8 * 16, n = 2 * group_cols, row_offset = 3 * group_rows;
    constexpr int checkpoints = PEARLHASH_K / PEARLHASH_RANK;
    // Hardware wave64 C layout, stated independently of the kernel: lane quarters 0..3 hold
    // rows 0-3, 8-11, 4-7, 12-15 of each 16x16 tile; lane % 16 is the column.
    constexpr int quarter_row[4] = {0, 8, 4, 12};
    // cells[t][row][col]: the slice's cumulative C at checkpoint t.
    std::vector<std::vector<uint32_t>> cells(checkpoints, std::vector<uint32_t>(size_t(m) * n));
    const int waves = 2 * 2; // (row group, column group) per wave; one 256-thread block
    std::vector<uint32_t> lane_transcript(256 * 17, 0xdeada55au);
    std::vector<std::array<std::array<v8i, PEARLHASH_EC>, PEARLHASH_ER>> lanes(256);
    for (auto& lane : lanes)
      for (auto& row : lane)
        for (auto& tile : row) tile = {};
    uint32_t state = 0x9e3779b9u;
    for (int kk = 0; kk < PEARLHASH_K; kk += 32) {
      for (int wave = 0; wave < waves; ++wave) {
        const int tile_row = (wave / 2) * PEARLHASH_ER, tile_col = (wave % 2) * PEARLHASH_EC;
        for (int lane = 0; lane < 64; ++lane) {
          const unsigned thread = unsigned(wave * 64 + lane);
          auto& acc = lanes[thread];
          for (int i = 0; i < PEARLHASH_ER; ++i)
            for (int j = 0; j < PEARLHASH_EC; ++j)
              for (int e = 0; e < 4; ++e) {
                acc[i][j][e] = int32_t(uint32_t(acc[i][j][e]) + next_random(state));
                const int row = (tile_row + i) * 16 + quarter_row[lane >> 4] + e;
                const int col = (tile_col + j) * 16 + (lane & 15);
                if ((kk + 32) % PEARLHASH_RANK == 0 && row < m && col < n)
                  cells[kk / PEARLHASH_RANK][size_t(row) * n + col] = uint32_t(acc[i][j][e]);
              }
          struct { unsigned x; } threadIdx{thread};
#include "lane-row.inc"
#include "checkpoint-fold.inc"
        }
      }
    }

    std::vector<unsigned> covered(size_t(m) * n);
    unsigned openings = 0;
    for (int wave = 0; wave < waves; ++wave) {
      const int tile_row = (wave / 2) * PEARLHASH_ER, tile_col = (wave % 2) * PEARLHASH_EC;
      const bool active_wave = true;
      for (int lane = 0; lane < 64; ++lane) {
        struct { unsigned x; } threadIdx{unsigned(wave * 64 + lane)};
#include "lane-row.inc"
#include "coordinates.inc"
        const bool publishes = (
#include "tail-guard.inc"
        );
        const bool complete = (wave / 2) == 0;
        require(publishes == complete, "tail group published, or a complete group did not");
        if (!publishes) continue;
        require(kAmdLaneRows.valid(int(proof_row), row_offset + m) &&
                    kAmdLaneCols.valid(int(proof_col), n),
                "published opening is not canonical inside the matrix");
        std::array<uint32_t, 16> expected{};
        for (int t = 0; t < checkpoints; ++t) {
          uint32_t tile = 0;
          for (uint32_t r = 0; r < kAmdLaneRows.count(); ++r)
            for (uint32_t c = 0; c < kAmdLaneCols.count(); ++c) {
              const int row = int(proof_row) - row_offset + int(kAmdLaneRows.component(r));
              const int col = int(proof_col) + int(kAmdLaneCols.component(c));
              tile ^= cells[t][size_t(row) * n + col];
              if (t == 0) ++covered[size_t(row) * n + col];
            }
          expected[t % 16] = rotl32(expected[t % 16], 13) ^ tile;
        }
        for (int w = 0; w < 16; ++w)
          require(transcript[w] == expected[w], "lane transcript differs from the opened cells");
        ++openings;
      }
    }
    for (int row = 0; row < group_rows; ++row)
      for (int col = 0; col < n; ++col)
        require(covered[size_t(row) * n + col] == 1, "openings do not cover complete groups once");
    std::printf("CPU lane transcript oracle passed: K=%d rank=%d checkpoints=%d openings=%u\n",
                PEARLHASH_K, PEARLHASH_RANK, checkpoints, openings);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
