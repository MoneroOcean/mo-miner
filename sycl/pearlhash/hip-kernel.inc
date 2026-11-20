// Runtime-compiled gfx12 wave64 matrix/transcript loop; producers and proofs remain shared SYCL.
R"===HIP(

typedef unsigned char      uint8_t;
typedef signed char        int8_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
#define PEARLHASH_WAVE64
#define PEARLHASH_PACK_K32_W64
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(4)));

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
#ifndef PEARLHASH_K_UNROLL
#define PEARLHASH_K_UNROLL 1
#endif

// Mirrors the host Result layout in matrix.inc: up to four distinct winning tiles per attempt.
constexpr int kPearlHashWinners = 4;
struct PearlHashWinner {
  uint32_t row;
  uint32_t col;
  uint8_t jackpot[32];
};
struct PearlHashResult {
  int found;
  uint32_t seed;
  uint32_t chk;
  PearlHashWinner winners[kPearlHashWinners];
};

#define DEV __device__ __forceinline__

DEV uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
DEV uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

// BLAKE3 quarter-round on state words a,b,c,d with message words mx,my (32-bit wraparound).
DEV void b3_g(uint32_t* s, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
  s[a] = s[a] + s[b] + mx;
  s[d] = rotr32(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];
  s[b] = rotr32(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + my;
  s[d] = rotr32(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];
  s[b] = rotr32(s[b] ^ s[c], 7);
}

// One BLAKE3 round; the caller permutes m between rounds.
DEV void b3_round(uint32_t* s, const uint32_t* m) {
  b3_g(s, 0, 4, 8, 12, m[0], m[1]);
  b3_g(s, 1, 5, 9, 13, m[2], m[3]);
  b3_g(s, 2, 6, 10, 14, m[4], m[5]);
  b3_g(s, 3, 7, 11, 15, m[6], m[7]);
  b3_g(s, 0, 5, 10, 15, m[8], m[9]);
  b3_g(s, 1, 6, 11, 12, m[10], m[11]);
  b3_g(s, 2, 7, 8, 13, m[12], m[13]);
  b3_g(s, 3, 4, 9, 14, m[14], m[15]);
}

// PearlHash's jackpot input is exactly one 64-byte block. Specializing the general BLAKE3 routine to
// that fixed keyed/root case saves the chunk-state machinery in every 16x16 output tile.
DEV void b3_keyed_64(const uint32_t block[16], const uint8_t* key, uint32_t out[8]) {
  const int perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  uint32_t cv[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    cv[i] = (uint32_t)key[i * 4] | ((uint32_t)key[i * 4 + 1] << 8) |
            ((uint32_t)key[i * 4 + 2] << 16) | ((uint32_t)key[i * 4 + 3] << 24);
  uint32_t s[16] = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
                    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                    0u, 0u, 64u, 27u}; // CHUNK_START | CHUNK_END | ROOT | KEYED_HASH
  uint32_t m[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    m[i] = block[i];
  }
  #pragma unroll
  for (int r = 0; r < 7; ++r) {
    b3_round(s, m);
    if (r != 6) {
      uint32_t p[16];
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        p[i] = m[perm[i]];
      }
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        m[i] = p[i];
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = s[i] ^ s[i + 8];
  }
}

// Keyed-BLAKE3 one 16-word transcript and test the little-endian jackpot <= target inclusively.
DEV bool tile_wins(const uint32_t tr[16], const uint8_t* key, const uint8_t* target,
                   uint32_t jackpot[8]) {
  b3_keyed_64(tr, key, jackpot);
  for (int i = 31; i >= 0; --i) {
    const uint8_t byte = (uint8_t)(jackpot[i >> 2] >> ((i & 3) * 8));
    if (byte < target[i]) return true;
    if (byte > target[i]) return false;
  }
  return true;
}


// Debug-only position-salted transcript checksum (MOM_PEARLHASH_CHK); not part of consensus.
DEV uint32_t tile_mix(uint32_t row, uint32_t col, const uint32_t tr[16]) {
  uint32_t h = row * 2654435761u + col * 2246822519u + 0x9e3779b9u;
  #pragma unroll
  for (int w = 0; w < 16; ++w) {
    h = (h ^ tr[w]) * 16777619u + (uint32_t)w;
  }
  return h;
}

// Map a wave index to its (ER*16)x(EC*16) output block. Grids that are multiples of the cache
// block walk rows inside column bands for L2 reuse; others fall back to row-major order.
DEV void wave_groups(uint32_t wave, int m, int n, int* row_group, int* col_group) {
  const int blocks_m = m / (16 * PEARLHASH_ER) + (m % (16 * PEARLHASH_ER) != 0);
  const int blocks_n = n / (16 * PEARLHASH_EC) + (n % (16 * PEARLHASH_EC) != 0);
  // Target a 24-MiB A band (192*K bytes per row group), with bounded B reuse/launch sizes.
  constexpr int cache_rows = 131072 / PEARLHASH_K < 16 ? 16 :
                             131072 / PEARLHASH_K > 64 ? 64 : 131072 / PEARLHASH_K;
  constexpr int cache_columns = 8;
  if (blocks_m % cache_rows == 0 && blocks_n % cache_columns == 0) {
    const int block_area = cache_rows * cache_columns;
    const uint32_t block = wave / (uint32_t)block_area;
    const uint32_t inner = wave - block * (uint32_t)block_area;
    const int blocks_w = blocks_n / cache_columns;
    *row_group = (int)(block / (uint32_t)blocks_w) * cache_rows +
                 (int)(inner / cache_columns);
    *col_group = (int)(block % (uint32_t)blocks_w) * cache_columns +
                 (int)(inner % cache_columns);
  } else {
    *row_group = (int)(wave / (uint32_t)blocks_n);
    *col_group = (int)(wave - (uint32_t)*row_group * (uint32_t)blocks_n);
  }
}

#if defined(__gfx1200__) || defined(__gfx1201__)
#if __has_builtin(__builtin_amdgcn_make_buffer_rsrc) && \
    __has_builtin(__builtin_amdgcn_raw_buffer_load_b64)
#define PEARLHASH_RAW_BUFFER 1
// A wave-uniform, bounded group keeps 64-bit allocation addresses out of per-lane load offsets.
// gfx12 needs the same format/OOB descriptor flags used by ROCm's composable-kernel implementation;
// a zero flags word can compile successfully but return incorrect matrix data.
template <int Tiles>
DEV __amdgpu_buffer_rsrc_t pearlhash_input_resource(const int8_t* base, int tile, int available = Tiles) {
  constexpr uint64_t tile_bytes = (uint64_t)PEARLHASH_K * 16u;
  static_assert(tile_bytes * Tiles <= 0x7fffffffu, "PearlHash buffer offsets exceed signed range");
  const int first = __builtin_amdgcn_readfirstlane(tile);
  const int count = __builtin_amdgcn_readfirstlane(
      available < 0 ? 0 : (available < Tiles ? available : Tiles));
  return __builtin_amdgcn_make_buffer_rsrc(
      (void*)(base + (uint64_t)first * tile_bytes), (short)0, tile_bytes * (uint64_t)count, 0x31004000u);
}
#else
#define PEARLHASH_RAW_BUFFER 0
#endif
// One wave64 per (ER*16)x(EC*16) block of K-packed A/B: accumulate int8 WMMA tiles, fold each
// rank checkpoint into the lane's own 52x4 proof transcript, then hash it in-kernel and publish
// the winners. Every lane owns one proof, so no cross-lane reduction or barrier is needed.
extern "C" __global__ __launch_bounds__(256)
void pearlhash_wmma_hash_fused(const int8_t* A, const int8_t* B,
                               const uint8_t* key, const uint8_t* target,
                               PearlHashResult* result, uint32_t seed, int m, int n,
                               int row_offset, int debug_checksum) {
  const int lane = (int)(threadIdx.x & 63);
  static_assert(PEARLHASH_K > 0 && PEARLHASH_RANK > 0,
                "PearlHash dimensions must be positive");
  static_assert(PEARLHASH_ER == 13 && PEARLHASH_EC == 4,
                "the host opens 52x4 lane proofs over 13x4 tiles (kAmdLaneRows/kAmdLaneCols)");
  const uint32_t wave = (uint32_t)(blockIdx.x * (blockDim.x / 64) + threadIdx.x / 64);
  const uint32_t groups_m = m / (16 * PEARLHASH_ER) + (m % (16 * PEARLHASH_ER) != 0);
  const uint32_t groups_n = n / (16 * PEARLHASH_EC) + (n % (16 * PEARLHASH_EC) != 0);
  const bool active_wave = wave < (uint64_t)groups_m * groups_n;
  int row_group, col_group;
  wave_groups(active_wave ? wave : 0, m, n, &row_group, &col_group);
  const int tile_row = active_wave ? row_group * PEARLHASH_ER : 0;
  const int tile_col = active_wave ? col_group * PEARLHASH_EC : 0;
  // Lane-private transcripts; the 17-word stride spreads one slot's stores across LDS banks.
  __shared__ uint32_t lane_transcript[256 * 17];
  uint32_t* const transcript = &lane_transcript[threadIdx.x * 17u];
  // A lane holds rows 4q..4q+3 of every A tile (WMMA wave64 C layout) and column lane%16 of
  // every B tile: exactly its 52x4 opening, whose origin is the first of each.
  const uint32_t proof_quarter =
      ((uint32_t)(lane >> 4) & 1u) * 2u + (uint32_t)(lane >> 5);
#if PEARLHASH_RAW_BUFFER
  // Tail accumulators have no proof owner; reuse a valid tile rather than issue any OOB load.
  // These values are wave-uniform. Make that explicit so raw descriptor/scalar offsets do not
  // become per-load waterfall loops after LLVM's conservative work-item divergence analysis.
  const int a_tiles = __builtin_amdgcn_readfirstlane(m / 16 - tile_row);
  const int b_tiles = __builtin_amdgcn_readfirstlane(n / 16 - tile_col);
  const auto a_resource = pearlhash_input_resource<PEARLHASH_ER>(A, tile_row, a_tiles);
  const auto b_resource = pearlhash_input_resource<PEARLHASH_EC>(B, tile_col, b_tiles);
#endif
  v8i acc[PEARLHASH_ER][PEARLHASH_EC] = {};

  static_assert(PEARLHASH_K % 32 == 0 && PEARLHASH_RANK % 32 == 0,
                "paired WMMA loads require complete 32-byte rank groups");
  #pragma unroll PEARLHASH_K_UNROLL
  for (int kk = 0; kk < PEARLHASH_K; kk += 32) {
    v2i a[PEARLHASH_ER], b[PEARLHASH_EC];
    #pragma unroll
    for (int i = 0; i < PEARLHASH_ER; ++i) {
#if PEARLHASH_RAW_BUFFER
      a[i] = __builtin_amdgcn_raw_buffer_load_b64(
          a_resource, (kk / 32) * 512 + lane * 8,
          (i < a_tiles ? i : 0) * (PEARLHASH_K * 16), 0);
#else
      const uint64_t fragment =
          ((uint64_t)(tile_row + i) * (PEARLHASH_K / 32) + kk / 32) * 512u;
      a[i] = tile_row + i < m / 16
          ? *reinterpret_cast<const v2i*>(A + fragment + (uint32_t)lane * 8u) : v2i{};
#endif
    }
    #pragma unroll
    for (int j = 0; j < PEARLHASH_EC; ++j) {
#if PEARLHASH_RAW_BUFFER
      b[j] = __builtin_amdgcn_raw_buffer_load_b64(
          b_resource, (kk / 32) * 512 + lane * 8,
          (j < b_tiles ? j : 0) * (PEARLHASH_K * 16), 0);
#else
      const uint64_t fragment =
          ((uint64_t)(tile_col + j) * (PEARLHASH_K / 32) + kk / 32) * 512u;
      b[j] = tile_col + j < n / 16
          ? *reinterpret_cast<const v2i*>(B + fragment + (uint32_t)lane * 8u) : v2i{};
#endif
    }
    #pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
      __builtin_amdgcn_sched_barrier(0);
      #pragma unroll
      for (int i = 0; i < PEARLHASH_ER; ++i) {
        #pragma unroll
        for (int j = 0; j < PEARLHASH_EC; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12(
              true, a[i][phase], true, b[j][phase], acc[i][j], false);
      }
    }

    if ((kk + 32) % PEARLHASH_RANK == 0) {
      // RDNA4 issues this VALU work in the WMMA pipe, so keep it minimal: XOR the lane's 208
      // accumulators as thirteen independent row chains, not one serial chain, then tree them.
      uint32_t part[PEARLHASH_ER];
      #pragma unroll
      for (int i = 0; i < PEARLHASH_ER; ++i) {
        uint32_t p0 = (uint32_t)acc[i][0][0] ^ (uint32_t)acc[i][0][1] ^ (uint32_t)acc[i][0][2];
        uint32_t p1 = (uint32_t)acc[i][0][3] ^ (uint32_t)acc[i][1][0] ^ (uint32_t)acc[i][1][1];
        p0 = p0 ^ (uint32_t)acc[i][1][2] ^ (uint32_t)acc[i][1][3];
        p1 = p1 ^ (uint32_t)acc[i][2][0] ^ (uint32_t)acc[i][2][1];
        p0 = p0 ^ (uint32_t)acc[i][2][2] ^ (uint32_t)acc[i][2][3];
        p1 = p1 ^ (uint32_t)acc[i][3][0] ^ (uint32_t)acc[i][3][1];
        part[i] = p0 ^ p1 ^ (uint32_t)acc[i][3][2] ^ (uint32_t)acc[i][3][3];
      }
      #pragma unroll
      for (int step = 1; step < PEARLHASH_ER; step *= 2) {
        #pragma unroll
        for (int i = 0; i + step < PEARLHASH_ER; i += 2 * step)
          part[i] ^= part[i + step];
      }
      // The verifier's slots start at zero, so a first visit is just the XOR.
      const int checkpoint = kk / PEARLHASH_RANK;
      uint32_t& slot = transcript[checkpoint & 15];
      slot = checkpoint < 16 ? part[0] : rotl32(slot, 13) ^ part[0];
    }
  }

  // Tail groups reuse clamped tiles (see a_tiles) and own no complete opening.
  const uint32_t proof_row = (uint32_t)row_offset + (uint32_t)tile_row * 16u + proof_quarter * 4u;
  const uint32_t proof_col = (uint32_t)tile_col * 16u + (uint32_t)(lane & 15);
  if (active_wave && tile_row + PEARLHASH_ER <= m / 16 && tile_col + PEARLHASH_EC <= n / 16) {
    uint32_t tr_hash[16];
    #pragma unroll
    for (int w = 0; w < 16; ++w)
      tr_hash[w] = transcript[w];
    if (debug_checksum)
      atomicAdd(&result->chk, tile_mix(proof_row, proof_col, tr_hash));
    uint32_t jackpot[8];
    if (tile_wins(tr_hash, key, target, jackpot)) {
      const int slot = atomicAdd(&result->found, 1);
      if (slot < kPearlHashWinners) {
        PearlHashWinner& winner = result->winners[slot];
        result->seed = seed;
        winner.row = proof_row;
        winner.col = proof_col;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
          winner.jackpot[w * 4] = (uint8_t)jackpot[w];
          winner.jackpot[w * 4 + 1] = (uint8_t)(jackpot[w] >> 8);
          winner.jackpot[w * 4 + 2] = (uint8_t)(jackpot[w] >> 16);
          winner.jackpot[w * 4 + 3] = (uint8_t)(jackpot[w] >> 24);
        }
      }
    }
  }
}
#endif


)===HIP"
