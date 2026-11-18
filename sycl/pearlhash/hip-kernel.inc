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

#if PEARLHASH_ER == 13
#define PEARLHASH_DIRECT_SLOT 1
#else
#define PEARLHASH_DIRECT_SLOT 0
#endif

struct PearlHashResult {
  int found;
  uint32_t seed;
  uint32_t row;
  uint32_t col;
  uint32_t chk;
  uint8_t jackpot[32];
};

#define DEV __device__ __forceinline__

DEV uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
DEV uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

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

#if defined(__gfx1200__) || defined(__gfx1201__)
// Fuse x ^= DPP(x) into one VALU instruction. Literal controls retain the
// existing full row/bank masks and bound_ctrl=false semantics.
template <int Mask>
DEV uint32_t pearlhash_xor_dpp(uint32_t x) {
  static_assert(Mask == 1 || Mask == 2 || Mask == 4 || Mask == 8,
                "checkpoint DPP mask must be one row XOR bit");
  uint32_t y;
  asm volatile("v_xor_b32_dpp %0, %1, %1 row_xmask:%2 row_mask:0xf bank_mask:0xf bound_ctrl:0"
               : "=v"(y) : "v"(x), "n"(Mask));
  return y;
}
#endif

DEV uint32_t tile_mix(uint32_t row, uint32_t col, const uint32_t tr[16]) {
  uint32_t h = row * 2654435761u + col * 2246822519u + 0x9e3779b9u;
  #pragma unroll
  for (int w = 0; w < 16; ++w) {
    h = (h ^ tr[w]) * 16777619u + (uint32_t)w;
  }
  return h;
}

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

// A wave64 quarter owns four rows. Fold four adjacent B tiles into 4x64 proofs,
// keeping all lanes useful and BLAKE3 outside the matrix kernel's register budget.
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
extern "C" __global__ __launch_bounds__(256)
void pearlhash_wmma_hash_fused(const int8_t* A, const int8_t* B,
                               const uint8_t* key, const uint8_t* target,
                               PearlHashResult* result, uint32_t seed, int m, int n,
                               int row_offset, int debug_checksum) {
  const int lane = (int)(threadIdx.x & 63);
  static_assert(PEARLHASH_K > 0 && PEARLHASH_RANK > 0,
                "PearlHash dimensions must be positive");
  constexpr bool direct_slots = PEARLHASH_ER == 13;
  static_assert(PEARLHASH_EC == 4 &&
                (PEARLHASH_ER == 12 || (direct_slots &&
                 PEARLHASH_K % PEARLHASH_RANK == 0 &&
                 PEARLHASH_K / PEARLHASH_RANK == 16)),
                "ER13 requires exactly sixteen checkpoint slots");
  const uint32_t wave = (uint32_t)(blockIdx.x * (blockDim.x / 64) + threadIdx.x / 64);
  const uint32_t wave_in_block = (uint32_t)(threadIdx.x / 64);
  const uint32_t groups_m = m / (16 * PEARLHASH_ER) + (m % (16 * PEARLHASH_ER) != 0);
  const uint32_t groups_n = n / (16 * PEARLHASH_EC) + (n % (16 * PEARLHASH_EC) != 0);
  const bool active_wave = wave < (uint64_t)groups_m * groups_n;
  int row_group, col_group;
  wave_groups(active_wave ? wave : 0, m, n, &row_group, &col_group);
  const int tile_row = active_wave ? row_group * PEARLHASH_ER : 0;
  const int tile_col = active_wave ? col_group * PEARLHASH_EC : 0;
  constexpr uint32_t proofs_per_wave = (uint32_t)PEARLHASH_ER * 4u;
  __shared__ uint32_t fused_transcript[4 * PEARLHASH_ER * 4 * 16];
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
  static_assert(PEARLHASH_EC % 4 == 0, "contiguous 4x64 proofs require four B tiles");
  // Each 16-lane quarter-wave owns one four-row proof across four adjacent B tiles.
  // All four quarter-waves participate; no lane is discarded and no cross-half permute is needed.
#if !PEARLHASH_DIRECT_SLOT
  uint32_t tr[PEARLHASH_ER][PEARLHASH_EC / 4] = {};
#endif

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
      const int rc = kk / PEARLHASH_RANK;
      uint32_t reduced[PEARLHASH_ER][PEARLHASH_EC / 4] = {};
      // Four adjacent B tiles make one contiguous 64-column proof with four rows.
      // DPP is deliberately confined to each 16-lane quarter-wave.
      #pragma unroll
      for (int i = 0; i < PEARLHASH_ER; ++i) {
        #pragma unroll
        for (int p = 0; p < PEARLHASH_EC / 4; ++p) {
          uint32_t x = 0;
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int e = 0; e < 4; ++e)
              x ^= (uint32_t)acc[i][4 * p + j][e];
          }
          x = pearlhash_xor_dpp<1>(x);
          x = pearlhash_xor_dpp<2>(x);
          x = pearlhash_xor_dpp<4>(x);
          x = pearlhash_xor_dpp<8>(x);
          reduced[i][p] = x;
        }
      }

      const int slot = rc & 15;
      // With sixteen checkpoints each slot is visited once. Write its final value
      // directly to existing LDS, freeing registers for a taller matrix tile.
      if (active_wave && ((lane & 15) == slot)) {
        #pragma unroll
        for (int i = 0; i < PEARLHASH_ER; ++i) {
          #pragma unroll
          for (int p = 0; p < PEARLHASH_EC / 4; ++p) {
#if PEARLHASH_DIRECT_SLOT
            const uint32_t local_proof = (uint32_t)i * 4u + proof_quarter;
            fused_transcript[wave_in_block * proofs_per_wave * 16u +
                             local_proof * 16u + (uint32_t)slot] = reduced[i][p];
#else
            tr[i][p] = rotl32(tr[i][p], 13) ^ reduced[i][p];
#endif
          }
        }
      }
    }
  }

  // Wrapped checkpoint slots retain their per-lane recurrence. The q16 route
  // already filled LDS; inactive waves still reach the common CTA barrier.
#if !PEARLHASH_DIRECT_SLOT
  #pragma unroll
  for (int i = 0; i < PEARLHASH_ER; ++i) {
    if (active_wave) {
      const uint32_t proof = (uint32_t)i * 4u + proof_quarter;
      fused_transcript[(wave_in_block * proofs_per_wave + proof) * 16u +
                       (uint32_t)(lane & 15)] = tr[i][0];
    }
  }
#endif
  __syncthreads();

  // One lane handles each 4x64 proof after the barrier. This retains the
  // keyed-64 BLAKE3 implementation while eliminating global transcript traffic
  // and the separate hash launch.
  if (active_wave && lane < proofs_per_wave) {
    const uint32_t hash_i = (uint32_t)lane % (uint32_t)PEARLHASH_ER;
    const uint32_t hash_q = (uint32_t)lane / (uint32_t)PEARLHASH_ER;
    const uint32_t local_proof = hash_i * 4u + hash_q;
    const uint32_t proof_row = (uint32_t)row_offset +
                               (uint32_t)(tile_row + (int)hash_i) * 16u +
                               hash_q * 4u;
    const uint32_t proof_col = (uint32_t)tile_col * 16u;
    if (proof_row < (uint32_t)row_offset + (uint32_t)m &&
        proof_col < (uint32_t)n) {
      uint32_t tr_hash[16];
      #pragma unroll
      for (int w = 0; w < 16; ++w)
        tr_hash[w] = fused_transcript[wave_in_block * proofs_per_wave * 16u +
                                      local_proof * 16u + (uint32_t)w];
      if (debug_checksum)
        atomicAdd(&result->chk, tile_mix(proof_row, proof_col, tr_hash));
      uint32_t jackpot[8];
      if (tile_wins(tr_hash, key, target, jackpot) &&
          atomicExch(&result->found, 1) == 0) {
        result->seed = seed;
        result->row = proof_row;
        result->col = proof_col;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
          result->jackpot[w * 4] = (uint8_t)jackpot[w];
          result->jackpot[w * 4 + 1] = (uint8_t)(jackpot[w] >> 8);
          result->jackpot[w * 4 + 2] = (uint8_t)(jackpot[w] >> 16);
          result->jackpot[w * 4 + 3] = (uint8_t)(jackpot[w] >> 24);
        }
      }
    }
  }
}
#endif


)===HIP"
