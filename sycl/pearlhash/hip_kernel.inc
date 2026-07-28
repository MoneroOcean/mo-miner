// gfx12 wave32 WMMA search for PearlHash. This is compiled at runtime by HIPRTC so the same host-side
// SYCL implementation can use AMD matrix hardware with either DPC++ or AdaptiveCpp, on Linux or
// Windows. The producer kernels and proof code remain shared SYCL; only the GEMM/search hot loop is
// native because neither compiler currently exposes gfx12 int8 WMMA through portable SYCL matrix APIs.
R"===HIP(
typedef unsigned char      uint8_t;
typedef signed char        int8_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

#ifndef PEARLHASH_ER
#define PEARLHASH_ER 4
#endif
#ifndef PEARLHASH_EC
#define PEARLHASH_EC 2
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
#ifndef PEARLHASH_CACHE_BLOCK
#define PEARLHASH_CACHE_BLOCK 32
#endif

struct PearlHashResult {
  int found;
  uint32_t seed;
  uint32_t row;
  uint32_t col;
  uint32_t chk;
};

#define DEV __device__ __forceinline__

DEV uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

DEV void b3_g(uint32_t* s, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
  s[a] = s[a] + s[b] + mx; s[d] = rotr32(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];      s[b] = rotr32(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + my; s[d] = rotr32(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];      s[b] = rotr32(s[b] ^ s[c], 7);
}

DEV void b3_round(uint32_t* s, const uint32_t* m) {
  b3_g(s, 0, 4, 8, 12, m[0], m[1]);  b3_g(s, 1, 5, 9, 13, m[2], m[3]);
  b3_g(s, 2, 6, 10, 14, m[4], m[5]); b3_g(s, 3, 7, 11, 15, m[6], m[7]);
  b3_g(s, 0, 5, 10, 15, m[8], m[9]); b3_g(s, 1, 6, 11, 12, m[10], m[11]);
  b3_g(s, 2, 7, 8, 13, m[12], m[13]); b3_g(s, 3, 4, 9, 14, m[14], m[15]);
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
  for (int i = 0; i < 16; ++i) m[i] = block[i];
  #pragma unroll
  for (int r = 0; r < 7; ++r) {
    b3_round(s, m);
    if (r != 6) {
      uint32_t p[16];
      #pragma unroll
      for (int i = 0; i < 16; ++i) p[i] = m[perm[i]];
      #pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = p[i];
    }
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = s[i] ^ s[i + 8];
}

DEV bool tile_wins(const uint32_t tr[16], const uint8_t* key, const uint8_t* target) {
  uint32_t jackpot[8];
  b3_keyed_64(tr, key, jackpot);
  for (int i = 31; i >= 0; --i) {
    const uint8_t byte = (uint8_t)(jackpot[i >> 2] >> ((i & 3) * 8));
    if (byte < target[i]) return true;
    if (byte > target[i]) return false;
  }
  return true;
}

DEV uint32_t tile_mix(uint32_t row, uint32_t col, const uint32_t tr[16]) {
  uint32_t h = row * 2654435761u + col * 2246822519u + 0x9e3779b9u;
  #pragma unroll
  for (int w = 0; w < 16; ++w) h = (h ^ tr[w]) * 16777619u + (uint32_t)w;
  return h;
}

DEV void wave_groups(uint32_t wave, int m, int n, int* row_group, int* col_group) {
  const int blocks_m = m / (16 * PEARLHASH_ER);
  const int blocks_n = n / (16 * PEARLHASH_EC);
  if (PEARLHASH_CACHE_BLOCK > 1 && blocks_m % PEARLHASH_CACHE_BLOCK == 0 &&
      blocks_n % PEARLHASH_CACHE_BLOCK == 0) {
    const int block_area = PEARLHASH_CACHE_BLOCK * PEARLHASH_CACHE_BLOCK;
    const uint32_t block = wave / (uint32_t)block_area;
    const uint32_t inner = wave - block * (uint32_t)block_area;
    const int blocks_w = blocks_n / PEARLHASH_CACHE_BLOCK;
    *row_group = (int)(block / (uint32_t)blocks_w) * PEARLHASH_CACHE_BLOCK +
                 (int)(inner / PEARLHASH_CACHE_BLOCK);
    *col_group = (int)(block % (uint32_t)blocks_w) * PEARLHASH_CACHE_BLOCK +
                 (int)(inner % PEARLHASH_CACHE_BLOCK);
  } else {
    *row_group = (int)(wave / (uint32_t)blocks_n);
    *col_group = (int)(wave - (uint32_t)*row_group * (uint32_t)blocks_n);
  }
}

// One wave computes a compile-time rectangular group of adjacent 16x16 output tiles. gfx12
// distributes each WMMA accumulator as
// eight rows in each half-wave; XORing its eight elements and then the wave reconstructs the exact
// cumulative tile transcript used by the portable SYCL and Intel/NVIDIA matrix paths. Store that
// transcript for a separate hash kernel: mixing BLAKE3's large scalar register set into this WMMA
// kernel cuts matrix occupancy, while the extra sequential transcript write+read is small beside
// the GEMM. Keep the architecture-specific builtin behind an explicit source guard: HIPRTC always
// targets the installed GPU, and unsupported architectures simply expose no native entry point so
// the host selects the common SYCL implementation.
#if defined(__gfx1200__) || defined(__gfx1201__)
extern "C" __global__ __launch_bounds__(256)
void pearlhash_wmma_transcript(const int8_t* A, const int8_t* B, uint32_t* transcript,
                           int m, int n) {
  const int lane = (int)(threadIdx.x & 31);
  const int wave = (int)(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
  int row_group, col_group;
  wave_groups((uint32_t)wave, m, n, &row_group, &col_group);
  const int tile_row = row_group * PEARLHASH_ER;
  const int tile_col = col_group * PEARLHASH_EC;
  v8i acc[PEARLHASH_ER][PEARLHASH_EC] = {};
  #pragma unroll PEARLHASH_K_UNROLL
  for (int kk = 0; kk < PEARLHASH_K; kk += 16) {
    v2i a[PEARLHASH_ER], b[PEARLHASH_EC];
    #pragma unroll
    for (int i = 0; i < PEARLHASH_ER; ++i) {
      const uint32_t fragment =
        ((uint32_t)(tile_row + i) * (PEARLHASH_K / 16) + kk / 16) * 256u;
      a[i] = *reinterpret_cast<const v2i*>(A + fragment + (uint32_t)lane * 8u);
    }
    #pragma unroll
    for (int j = 0; j < PEARLHASH_EC; ++j) {
      const uint32_t fragment =
        ((uint32_t)(tile_col + j) * (PEARLHASH_K / 16) + kk / 16) * 256u;
      b[j] = *reinterpret_cast<const v2i*>(B + fragment + (uint32_t)lane * 8u);
    }
    #pragma unroll
    for (int i = 0; i < PEARLHASH_ER; ++i) {
      #pragma unroll
      for (int j = 0; j < PEARLHASH_EC; ++j)
        acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
          true, a[i], true, b[j], acc[i][j], false);
    }

    if ((kk + 16) % PEARLHASH_RANK == 0) {
      const int rc = kk / PEARLHASH_RANK;
      #pragma unroll
      for (int i = 0; i < PEARLHASH_ER; ++i) {
        #pragma unroll
        for (int j = 0; j < PEARLHASH_EC; ++j) {
          uint32_t x = 0;
          #pragma unroll
          for (int e = 0; e < 8; ++e) x ^= (uint32_t)acc[i][j][e];
          #pragma unroll
          for (int d = 1; d < 32; d <<= 1) x ^= __shfl_xor(x, d, 32);
          if (lane == rc) {
            // Wave-major output keeps the 8 transcripts produced by one wave contiguous. The hash
            // kernel below applies the same wave-to-matrix mapping when it records row and column.
            const uint32_t tile = (uint32_t)wave * (PEARLHASH_ER * PEARLHASH_EC) + i * PEARLHASH_EC + j;
            transcript[tile * 16u + (uint32_t)rc] = x;
          }
        }
      }
    }
  }
}
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void pearlhash_hash_search(const uint32_t* transcript, const uint8_t* key, const uint8_t* target,
                       PearlHashResult* result, uint32_t seed, int m, int n, int debug_checksum) {
  const uint32_t tile = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t tile_count = ((uint32_t)m / 16u) * ((uint32_t)n / 16u);
  if (tile >= tile_count) return;
  uint32_t tr[16];
  #pragma unroll
  for (int w = 0; w < 16; ++w) tr[w] = transcript[tile * 16u + (uint32_t)w];
  const uint32_t wave = tile / (PEARLHASH_ER * PEARLHASH_EC);
  const int local_tile = (int)(tile % (PEARLHASH_ER * PEARLHASH_EC));
  int row_group, col_group;
  wave_groups(wave, m, n, &row_group, &col_group);
  const uint32_t row = (uint32_t)(row_group * PEARLHASH_ER + local_tile / PEARLHASH_EC) * 16u;
  const uint32_t col = (uint32_t)(col_group * PEARLHASH_EC + local_tile % PEARLHASH_EC) * 16u;
  if (debug_checksum) atomicAdd(&result->chk, tile_mix(row, col, tr));
  if (tile_wins(tr, key, target) && atomicExch(&result->found, 1) == 0) {
    result->seed = seed;
    result->row = row;
    result->col = col;
  }
}
)===HIP"
