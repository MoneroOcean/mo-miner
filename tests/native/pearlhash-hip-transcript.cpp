// CPU-only model of the final fused raw source.  The driver supplies reduction, LDS-store,
// and one-lane coordinate snippets extracted from the frozen production overlay.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#ifndef PEARLHASH_ER
#define PEARLHASH_ER 12
#endif
#ifndef PEARLHASH_EC
#define PEARLHASH_EC 4
#endif
#ifndef PEARLHASH_K
#define PEARLHASH_K 4096
#endif
#ifndef PEARLHASH_RANK
#define PEARLHASH_RANK 256
#endif
#ifndef PEARLHASH_DIRECT_SLOT
#if PEARLHASH_ER == 13
#define PEARLHASH_DIRECT_SLOT 1
#else
#define PEARLHASH_DIRECT_SLOT 0
#endif
#endif

// Exercise the actual JIT contract, not a copied assertion that could keep
// passing after the production guard was accidentally removed.
#include "kernel-guard.inc"
#include "proofs-per-wave.inc"
#include "shared-extent.inc"

constexpr uint32_t expected_proofs_per_wave = (uint32_t)PEARLHASH_ER * 4u;
static_assert(PEARLHASH_ER != 13 || expected_proofs_per_wave == 52u,
              "ER13 must expose exactly 52 proofs per wave");
static_assert(proofs_per_wave == expected_proofs_per_wave,
              "production proof count differs from the independent ER*4 oracle");
static_assert(production_fused_transcript_words ==
                  size_t(4) * expected_proofs_per_wave * 16u,
              "production shared transcript extent differs from the proof count");

struct LaneValues {
  std::array<uint32_t, 64> v{};
  LaneValues(uint32_t value = 0) { v.fill(value); }
  LaneValues& operator^=(const LaneValues& other) {
    for (unsigned lane = 0; lane < v.size(); ++lane) v[lane] ^= other.v[lane];
    return *this;
  }
};

template <int Mask>
static LaneValues pearlhash_xor_dpp(const LaneValues& x) {
  static_assert(Mask == 1 || Mask == 2 || Mask == 4 || Mask == 8,
                "checkpoint DPP mask must be one row XOR bit");
  LaneValues out;
  for (unsigned lane = 0; lane < out.v.size(); ++lane)
    out.v[lane] = x.v[lane] ^ x.v[lane ^ Mask];
  return out;
}

using v8i = std::array<LaneValues, 4>;
using State = std::array<std::array<std::array<uint32_t, PEARLHASH_EC / 4>, PEARLHASH_ER>, 64>;
using Accum = std::array<std::array<std::array<v8i, PEARLHASH_EC>, PEARLHASH_ER>, 4>;
using CtaLds = std::array<uint32_t, production_fused_transcript_words>;
using WriteCounts = std::array<unsigned, production_fused_transcript_words>;

struct LdsWriter {
  CtaLds& values;
  WriteCounts& writes;
  uint32_t& operator[](std::size_t index) {
    if (index >= values.size()) throw std::runtime_error("fused LDS index escaped allocation");
    ++writes[index];
    return values[index];
  }
};

static uint32_t next_random(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static uint64_t rotl32_calls = 0;

[[maybe_unused]] static uint32_t rotl32(uint32_t x, int n) {
  ++rotl32_calls;
  return (x << n) | (x >> (32 - n));
}

[[maybe_unused]] static uint32_t rotl13(uint32_t x) { return (x << 13) | (x >> 19); }

static CtaLds execute_all_stores(const std::vector<Accum>& accum_history,
                                 std::array<State, 4>& states, int checkpoints,
                                 const std::array<bool, 4>& active,
                                 WriteCounts& write_counts) {
  constexpr uint32_t lds_sentinel = 0xdeada55au;
  CtaLds fused_transcript;
  fused_transcript.fill(lds_sentinel);
  for (int checkpoint = 0; checkpoint < checkpoints; ++checkpoint) {
    const int rc = checkpoint;
    const int kk = (checkpoint + 1) * PEARLHASH_RANK - 32;
    (void)kk;
    for (int wave = 0; wave < 4; ++wave) {
      const uint32_t wave_in_block = static_cast<uint32_t>(wave);
      const bool active_wave = active[wave];
      (void)wave_in_block;
      const auto& acc = accum_history[checkpoint][wave];
      for (int lane = 0; lane < 64; ++lane) {
        const uint32_t proof_quarter =
            ((uint32_t)(lane >> 4) & 1u) * 2u + (uint32_t)(lane >> 5);
        LdsWriter fused_transcript_alias{fused_transcript, write_counts};
        (void)proof_quarter;
        (void)fused_transcript_alias;
#define fused_transcript fused_transcript_alias
#define uint32_t LaneValues
#define reduced reduced_vectors
#include "reduction.inc"
#undef reduced
#undef uint32_t
        std::array<std::array<uint32_t, PEARLHASH_EC / 4>, PEARLHASH_ER> reduced{};
        for (int i = 0; i < PEARLHASH_ER; ++i)
          for (int p = 0; p < PEARLHASH_EC / 4; ++p)
            reduced[i][p] = reduced_vectors[i][p].v[lane];
        auto& tr = states[wave][lane];
        (void)tr;
#include "checkpoint-owner.inc"
#undef fused_transcript
      }
    }
  }
  for (int wave = 0; wave < 4; ++wave) {
    const uint32_t wave_in_block = static_cast<uint32_t>(wave);
    const bool active_wave = active[wave];
    (void)wave_in_block;
    (void)active_wave;
    for (int lane = 0; lane < 64; ++lane) {
      const uint32_t proof_quarter =
          ((uint32_t)(lane >> 4) & 1u) * 2u + (uint32_t)(lane >> 5);
      (void)proof_quarter;
      LdsWriter fused_transcript_alias{fused_transcript, write_counts};
      (void)fused_transcript_alias;
#define fused_transcript fused_transcript_alias
      auto& tr = states[wave][lane];
      (void)tr;
#include "final-store.inc"
#undef fused_transcript
    }
  }
  return fused_transcript;
}

struct HashCoordinate {
  uint32_t local_proof;
  uint32_t row;
  uint32_t col;
  bool in_bounds;
};

static HashCoordinate hash_coordinate(int lane_in, int tile_row, int tile_col, int m, int n,
                                      int row_offset, bool active_wave) {
  const uint32_t lane = static_cast<uint32_t>(lane_in);
  const bool hash_lane_active =
#include "hash-guard.inc"
      ;
  if (!hash_lane_active) return {UINT32_MAX, 0, 0, false};
#include "hash-coordinates.inc"
  bool in_bounds = false;
#include "tail-guard.inc"
  return {local_proof, proof_row, proof_col, in_bounds};
}

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

static void check_reduction_and_stores(int checkpoints) {
#if PEARLHASH_DIRECT_SLOT
  require(checkpoints == 16, "direct-slot fixture must model exactly sixteen checkpoints");
#endif
  Accum accum{};
  std::vector<Accum> accum_history;
  std::array<State, 4> expected_states{};
  uint32_t rng = 0x6d2b79f5u;

  for (int checkpoint = 0; checkpoint < checkpoints; ++checkpoint) {
    const int slot = checkpoint & 15;
    for (int wave = 0; wave < 4; ++wave) {
      for (int i = 0; i < PEARLHASH_ER; ++i)
        for (int j = 0; j < PEARLHASH_EC; ++j)
          for (int e = 0; e < 4; ++e)
            for (auto& value : accum[wave][i][j][e].v)
              value += next_random(rng);

      for (int quarter = 0; quarter < 4; ++quarter) {
        const int lane = quarter * 16 + slot;
        const int row_group = (quarter & 1) * 2 + (quarter >> 1);
        for (int i = 0; i < PEARLHASH_ER; ++i)
          for (int p = 0; p < PEARLHASH_EC / 4; ++p) {
            uint32_t x = 0;
            for (int j = 4 * p; j < 4 * p + 4; ++j)
              for (int e = 0; e < 4; ++e)
                for (int source_lane = quarter * 16; source_lane < (quarter + 1) * 16;
                     ++source_lane)
                  x ^= accum[wave][i][j][e].v[source_lane];
            const int proof = i * 4 + row_group;
#if PEARLHASH_DIRECT_SLOT
            expected_states[wave][lane][i][p] = x;
#else
            expected_states[wave][lane][i][p] =
                rotl13(expected_states[wave][lane][i][p]) ^ x;
#endif
            (void)proof;
          }
      }

    }
    accum_history.push_back(accum);
  }

  for (const auto& active : {std::array<bool, 4>{true, false, false, false},
                             std::array<bool, 4>{true, true, true, false},
                             std::array<bool, 4>{true, true, true, true}}) {
    std::array<State, 4> states{};
    WriteCounts write_counts{};
    rotl32_calls = 0;
    const CtaLds lds = execute_all_stores(accum_history, states, checkpoints, active, write_counts);
    unsigned active_waves = 0;
    for (bool wave_active : active) active_waves += wave_active ? 1u : 0u;
    const uint64_t expected_rotl32_calls = PEARLHASH_DIRECT_SLOT ? 0u :
        uint64_t(checkpoints) * uint64_t(active_waves) * uint64_t(PEARLHASH_ER) *
        uint64_t(PEARLHASH_EC / 4) * 4u;
    require(rotl32_calls == expected_rotl32_calls,
            "checkpoint recurrence call count changed for the selected profile");
#if !PEARLHASH_DIRECT_SLOT
    for (int wave = 0; wave < 4; ++wave)
      for (int lane = 0; lane < 64; ++lane)
        for (int i = 0; i < PEARLHASH_ER; ++i)
          for (int p = 0; p < PEARLHASH_EC / 4; ++p)
            if (active[wave])
              require(states[wave][lane][i][p] == expected_states[wave][lane][i][p],
                      "reduction differs from independent quarter-wave oracle");
#endif
    constexpr uint32_t lds_sentinel = 0xdeada55au;
    for (int wave = 0; wave < 4; ++wave)
      for (int proof = 0; proof < PEARLHASH_ER * 4; ++proof)
        for (int word = 0; word < 16; ++word) {
          const uint32_t actual = lds[(wave * PEARLHASH_ER * 4 + proof) * 16 + word];
          const unsigned writes = write_counts[(wave * PEARLHASH_ER * 4 + proof) * 16 + word];
          if (!active[wave]) {
            require(actual == lds_sentinel, "inactive wave escaped the guarded LDS store");
            require(writes == 0, "inactive wave wrote a fused LDS slot");
            continue;
          }
          require(writes == 1, "active fused LDS slot was not written exactly once");
          const int i = proof / 4;
          const int row_group = proof & 3;
          const int quarter = (row_group & 1) * 2 + (row_group >> 1);
          const int lane = quarter * 16 + word;
#if PEARLHASH_DIRECT_SLOT
          const uint32_t expected = expected_states[wave][lane][i][0];
#else
          const uint32_t expected = states[wave][lane][i][0];
#endif
          require(actual == expected,
                  "fused LDS proof index differs from the source mapping");
        }
  }
}

static void check_coordinates() {
  constexpr int tile_er = PEARLHASH_ER;
  const auto check_shape = [&](int m, int n, int row_offset) {
    const int groups_m = (m + tile_er * 16 - 1) / (tile_er * 16);
    const int groups_n = (n + 63) / 64;
    int accepted = 0;
    for (int row_group = 0; row_group < groups_m; ++row_group)
      for (int col_group = 0; col_group < groups_n; ++col_group)
        for (int lane = 0; lane < 64; ++lane) {
          const HashCoordinate coordinate =
              hash_coordinate(lane, row_group * tile_er, col_group * 4, m, n, row_offset, true);
          const bool expected_active = static_cast<uint32_t>(lane) < expected_proofs_per_wave;
          if (!expected_active) {
            require(coordinate.local_proof == UINT32_MAX && !coordinate.in_bounds,
                    "hash lane guard admitted an inactive proof lane");
            continue;
          }
          const uint32_t expected_i = static_cast<uint32_t>(lane % tile_er);
          const uint32_t expected_q = static_cast<uint32_t>(lane / tile_er);
          const uint32_t expected_local = expected_i * 4u + expected_q;
          const uint32_t expected_row = static_cast<uint32_t>(row_offset) +
              static_cast<uint32_t>(row_group * tile_er + static_cast<int>(expected_i)) * 16u +
              expected_q * 4u;
          const uint32_t expected_col = static_cast<uint32_t>(col_group * 4) * 16u;
          require(coordinate.local_proof == expected_local &&
                      coordinate.row == expected_row && coordinate.col == expected_col,
                  "fused hash coordinate differs from independent tile oracle");
          const bool expected_in_bounds = expected_row < static_cast<uint32_t>(row_offset + m) &&
                                          expected_col < static_cast<uint32_t>(n);
          require(coordinate.in_bounds == expected_in_bounds,
                  "fused hash tail guard differs from independent bounds oracle");
          accepted += coordinate.in_bounds ? 1 : 0;
        }
    const int expected_accepted = (m / 4) * (n / 64);
    require(accepted == expected_accepted,
            "active one-lane hash mask or tail coordinate coverage changed");
  };
  for (const int m : {192, 176, 128, 64, 16})
    for (const int n : {64, 128})
      for (const int row_offset : {0, 1536, 3072})
        check_shape(m, n, row_offset);
}

static void check_inactive_hash() {
  for (int lane = 0; lane < 64; ++lane) {
    const HashCoordinate coordinate = hash_coordinate(lane, 0, 0, 128, 128, 0, false);
    require(coordinate.local_proof == UINT32_MAX && !coordinate.in_bounds,
            "inactive wave reached the fused hash reader");
  }
}

int main() try {
#if PEARLHASH_DIRECT_SLOT
  check_reduction_and_stores(16);
#else
  check_reduction_and_stores(16);
  check_reduction_and_stores(32);
  check_reduction_and_stores(64);
#endif
  check_coordinates();
  check_inactive_hash();
  std::printf("CPU fused oracle passed: ER=%d K=%d/R=%d, checkpoint/store parity, tails, inactive waves and coordinates\n",
              PEARLHASH_ER, PEARLHASH_K, PEARLHASH_RANK);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
