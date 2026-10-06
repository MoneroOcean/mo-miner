// NVIDIA's public SYCL matrix interface does not expose the staged CUTE mainloop needed by
// PearlHash. At M131072/N524288/K8192/rank128 on RTX 5060 Ti, the retained Linux measurements are
// 56.39 TH/s for SYCL-native and 80.24 TH/s for this mainloop at 150 W (2026-09-26).
// Compile it with NVRTC for the newest target supported by both the installed compiler and selected
// GPU, then cache the resulting PTX or cubin only on that miner. Releases never contain a
// precompiled device image. The normal SYCL kernel remains the portable and user-selectable
// fallback.
R"===CUDA(
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime.h>
#include <cuda/std/cstdint>

using namespace cute;
using cuda::std::int8_t;
using cuda::std::int64_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;

using CtaTile = Shape<_128, _256, _128>;
using TiledMma = decltype(make_tiled_mma(
  SM80_16x8x32_S32S8S8S32_TN{},
  Layout<Shape<_1, _1, _1>>{},
  Tile<_64, _64, _32>{}));

using SmemLayoutA = decltype(composition(
  Swizzle<3, 4, 3>{},
  Layout<Shape<Shape<_16, _8>, Shape<_128, _1>, Shape<_1, _2>>,
         Stride<Stride<_128, _2048>, Stride<_1, _0>, Stride<_0, _16384>>>{}));
using SmemLayoutB = decltype(composition(
  Swizzle<3, 4, 3>{},
  Layout<Shape<Shape<_16, _16>, Shape<_128, _1>, Shape<_1, _2>>,
         Stride<Stride<_128, _2048>, Stride<_1, _0>, Stride<_0, _32768>>>{}));
using WarpSmemLayoutA = decltype(composition(
  Swizzle<3, 4, 3>{},
  Layout<Shape<Shape<_16, _4>, Shape<_128, _1>, Shape<_1, _2>>,
         Stride<Stride<_128, _2048>, Stride<_1, _0>, Stride<_0, _16384>>>{}));
using WarpSmemLayoutB = decltype(composition(
  Swizzle<3, 4, 3>{},
  Layout<Shape<Shape<_16, _4>, Shape<_128, _1>, Shape<_1, _2>>,
         Stride<Stride<_128, _2048>, Stride<_1, _0>, Stride<_0, _32768>>>{}));

using GmemCopy = decltype(make_tiled_copy(
  Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cutlass::uint128_t>, int8_t>{},
  Layout<Shape<_32, _8>, Stride<_8, _1>>{},
  Layout<Shape<_1, _16>>{}));
using SmemCopyA = Copy_Atom<SM75_U32x4_LDSM_N, int8_t>;
using SmemCopyB = Copy_Atom<SM75_U32x2_LDSM_N, int8_t>;

struct SharedStorage {
  alignas(1024) cutlass::Array<int8_t, cosize_v<SmemLayoutA>> a;
  alignas(1024) cutlass::Array<int8_t, cosize_v<SmemLayoutB>> b;
};

struct PearlHashResult {
  int found;
  uint32_t seed;
  uint32_t row;
  uint32_t col;
  uint32_t chk;
  unsigned char jackpot[32];
};

__device__ __forceinline__ uint32_t load32(const unsigned char* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
         (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}
__device__ __forceinline__ void b3_g(
    uint32_t* s, int a, int b, int c, int d, uint32_t x, uint32_t y) {
  s[a] += s[b] + x;
  s[d] = rotr32(s[d] ^ s[a], 16);
  s[c] += s[d];
  s[b] = rotr32(s[b] ^ s[c], 12);
  s[a] += s[b] + y;
  s[d] = rotr32(s[d] ^ s[a], 8);
  s[c] += s[d];
  s[b] = rotr32(s[b] ^ s[c], 7);
}
__device__ __forceinline__ void b3_round(uint32_t* s, const uint32_t* m) {
  b3_g(s,0,4,8,12,m[0],m[1]);
  b3_g(s,1,5,9,13,m[2],m[3]);
  b3_g(s,2,6,10,14,m[4],m[5]);
  b3_g(s,3,7,11,15,m[6],m[7]);
  b3_g(s,0,5,10,15,m[8],m[9]);
  b3_g(s,1,6,11,12,m[10],m[11]);
  b3_g(s,2,7,8,13,m[12],m[13]);
  b3_g(s,3,4,9,14,m[14],m[15]);
}
__device__ __forceinline__ uint32_t tile_mix(
    uint32_t row, uint32_t col, const uint32_t* tr) {
  uint32_t h = row * 2654435761u + col * 2246822519u + 0x9e3779b9u;
#pragma unroll
  for (int w = 0; w < 16; ++w) {
    h = (h ^ tr[w]) * 16777619u + uint32_t(w);
  }
  return h;
}

extern "C" __global__ __launch_bounds__(256)
void pearlhash_cuda_hash(const uint32_t* transcript, const unsigned char* key,
                     const unsigned char* target, PearlHashResult* result,
                     uint32_t seed, int m, int n, int row_offset, int debug_checksum) {
  const uint32_t tile = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t tile_columns = uint32_t(n) / 16u;
  const uint32_t tile_count = (uint32_t(m) / 16u) * tile_columns;
  if (tile >= tile_count) return;
  uint32_t transcript_words[16], message[16], state[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    transcript_words[i] = transcript[uint64_t(tile) * 16u + uint32_t(i)];
    message[i] = transcript_words[i];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    state[i] = load32(key + i * 4);
  }
  state[8] = 0x6a09e667u;
  state[9] = 0xbb67ae85u;
  state[10] = 0x3c6ef372u;
  state[11] = 0xa54ff53au;
  state[12] = 0;
  state[13] = 0;
  state[14] = 64;
  state[15] = 27; // KEYED_HASH | CHUNK_START | CHUNK_END | ROOT
  constexpr int permutation[16] = {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8};
#pragma unroll
  for (int round = 0; round < 7; ++round) {
    b3_round(state, message);
    if (round != 6) {
      uint32_t permuted[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        permuted[i] = message[permutation[i]];
      }
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        message[i] = permuted[i];
      }
    }
  }
  bool wins = true;
#pragma unroll
  for (int i = 7; i >= 0; --i) {
    const uint32_t jackpot = state[i] ^ state[i + 8];
    const uint32_t bound = load32(target + i * 4);
    if (jackpot != bound) {
      wins = jackpot < bound;
      break;
    }
  }
  const uint32_t row = uint32_t(row_offset) + (tile / tile_columns) * 16u;
  const uint32_t col = (tile % tile_columns) * 16u;
  if (debug_checksum) {
    atomicAdd(&result->chk, tile_mix(row, col, transcript_words));
  }
  if (wins && atomicExch(&result->found, 1) == 0) {
    result->seed = seed;
    result->row = row;
    result->col = col;
    for (int i = 0; i < 8; ++i) {
      const uint32_t word = state[i] ^ state[i + 8];
      result->jackpot[i * 4] = static_cast<unsigned char>(word);
      result->jackpot[i * 4 + 1] = static_cast<unsigned char>(word >> 8);
      result->jackpot[i * 4 + 2] = static_cast<unsigned char>(word >> 16);
      result->jackpot[i * 4 + 3] = static_cast<unsigned char>(word >> 24);
    }
  }
}

extern "C" __global__ __launch_bounds__(256)
void pearlhash_cuda_gemm(const int8_t* __restrict__ ap, const int8_t* __restrict__ bp,
                     int m, int n, int k, uint32_t* transcript_out) {
  auto shape_mnk = make_shape(m, n, k);
  // The rank-128 reference profile has a 4-GiB B matrix; CUTE's offset arithmetic follows
  // the stride type, so an int stride would wrap above 2 GiB despite the 64-bit pointer.
  const auto input_stride = make_stride(static_cast<int64_t>(k), _1{});
  Tensor mA = make_tensor(make_gmem_ptr(ap), select<0, 2>(shape_mnk), input_stride);
  Tensor mB = make_tensor(make_gmem_ptr(bp), select<1, 2>(shape_mnk), input_stride);
  static_assert(sizeof(decltype(mB.layout()(make_coord(0, 0)))) >= sizeof(int64_t),
                "PearlHash global matrix offsets must be 64-bit");
  // Nearby CTAs reuse A/B through L2. Rank-128 groups also leave room for the neighboring
  // group, which can begin while the previous group's final CTAs are still active.
  int block_m = int(blockIdx.x);
  int block_n = int(blockIdx.y);
  constexpr int swizzle_m = 16;
  const int swizzle_n = k == 8192 ? 4 : 16;
  const int grid_m = m / 128;
  const int grid_n = n / 256;
  if (grid_m % swizzle_m == 0 && grid_n % swizzle_n == 0) {
    const int linear = int(blockIdx.y) * grid_m + int(blockIdx.x);
    const int supertile = linear / (swizzle_m * swizzle_n);
    const int local = linear % (swizzle_m * swizzle_n);
    const int supertiles_n = grid_n / swizzle_n;
    block_m = (supertile / supertiles_n) * swizzle_m + local / swizzle_n;
    block_n = (supertile % supertiles_n) * swizzle_n + local % swizzle_n;
  }
  auto cta_coord = make_coord(block_m, block_n, _);
  Tensor gA = local_tile(mA, CtaTile{}, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, CtaTile{}, cta_coord, Step<X, _1, _1>{});

  extern __shared__ char storage_raw[];
  SharedStorage& storage = *reinterpret_cast<SharedStorage*>(storage_raw);
  Tensor sA = make_tensor(make_smem_ptr(storage.a.data()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(storage.b.data()), SmemLayoutB{});
  GmemCopy gmem_copy;
  auto copy_a = gmem_copy.get_slice(threadIdx.x);
  auto tAgA = copy_a.partition_S(gA);
  auto tAsA = copy_a.partition_D(sA);
  auto copy_b = gmem_copy.get_slice(threadIdx.x);
  auto tBgB = copy_b.partition_S(gB);
  auto tBsB = copy_b.partition_D(sB);

  constexpr int stages = 2;
  int tiles_left = size<3>(tAgA);
  int next_tile = 0;
  copy(gmem_copy, tAgA(_, _, _, next_tile), tAsA(_, _, _, 0));
  copy(gmem_copy, tBgB(_, _, _, next_tile), tBsB(_, _, _, 0));
  cp_async_fence();
  --tiles_left;
  if (tiles_left > 0) {
    ++next_tile;
  }

  const int warp = int(threadIdx.x) >> 5;
  const int lane = int(threadIdx.x) & 31;
  const int warp_m = warp >> 2;
  const int warp_n = warp & 3;
  TiledMma mma;
  auto thread_mma = mma.get_slice(lane);
  Tensor identity = make_identity_tensor(make_shape(64, 64));
  Tensor output_tile = local_tile(
    identity, Shape<_64, _64, _128>{}, make_coord(0, 0, _), Step<_1, _1, X>{});
  Tensor thread_output = thread_mma.partition_C(output_tile);
  Tensor warp_a = make_tensor(
    make_smem_ptr(storage.a.data() + warp_m * 64 * 128), WarpSmemLayoutA{});
  Tensor warp_b = make_tensor(
    make_smem_ptr(storage.b.data() + warp_n * 64 * 128), WarpSmemLayoutB{});
  Tensor fragment_a = thread_mma.partition_fragment_A(warp_a(_, _, 0));
  Tensor fragment_b = thread_mma.partition_fragment_B(warp_b(_, _, 0));
  Tensor accumulator = thread_mma.make_fragment_C(thread_output);
  clear(accumulator);
  uint32_t transcript[8] = {};
  int rank_chunk = 0;

  auto smem_copy_a = make_tiled_copy_A(SmemCopyA{}, mma);
  auto smem_thread_a = smem_copy_a.get_slice(threadIdx.x);
  Tensor thread_smem_a = smem_thread_a.partition_S(warp_a);
  Tensor thread_reg_a = smem_thread_a.retile_D(fragment_a);
  auto smem_copy_b = make_tiled_copy_B(SmemCopyB{}, mma);
  auto smem_thread_b = smem_copy_b.get_slice(threadIdx.x);
  Tensor thread_smem_b = smem_thread_b.partition_S(warp_b);
  Tensor thread_reg_b = smem_thread_b.retile_D(fragment_b);

  int read_stage = 0;
  int write_stage = 1;
  auto stage_a = thread_smem_a(_, _, _, read_stage);
  auto stage_b = thread_smem_b(_, _, _, read_stage);
  constexpr int k_blocks = size<2>(fragment_a);
  cp_async_wait<0>();
  __syncthreads();
  copy(SmemCopyA{}, stage_a(_, _, Int<0>{}), thread_reg_a(_, _, Int<0>{}));
  copy(SmemCopyB{}, stage_b(_, _, Int<0>{}), thread_reg_b(_, _, Int<0>{}));

  while (tiles_left > -(stages - 1)) {
    CUTE_UNROLL
    for (int kb = 0; kb < k_blocks; ++kb) {
      if (kb == k_blocks - 1) {
        stage_a = thread_smem_a(_, _, _, read_stage);
        stage_b = thread_smem_b(_, _, _, read_stage);
        cp_async_wait<0>();
        __syncthreads();
      }
      const int next_kb = (kb + 1) % k_blocks;
      copy(SmemCopyA{}, stage_a(_, _, next_kb), thread_reg_a(_, _, next_kb));
      copy(SmemCopyB{}, stage_b(_, _, next_kb), thread_reg_b(_, _, next_kb));
      if (kb == 0) {
        copy(gmem_copy, tAgA(_, _, _, next_tile), tAsA(_, _, _, write_stage));
        copy(gmem_copy, tBgB(_, _, _, next_tile), tBsB(_, _, _, write_stage));
        cp_async_fence();
        --tiles_left;
        if (tiles_left > 0) {
          ++next_tile;
        }
        write_stage = read_stage;
        read_stage ^= 1;
      }
      gemm(mma, fragment_a(_, _, kb), fragment_b(_, _, kb), accumulator);
    }
    ++rank_chunk;
    constexpr int stages_per_rank = PEARLHASH_RANK / 128;
    if (rank_chunk % stages_per_rank == 0) {
      const int transcript_word = (rank_chunk / stages_per_rank - 1) & 15;
      uint32_t lane_part = 0;
      CUTE_UNROLL
      for (int tile = 0; tile < 16; ++tile) {
        const int tile_r = tile >> 2;
        const int tile_c = tile & 3;
        const int lo = ((tile_c * 2) * 4 + tile_r) * 4;
        const int hi = ((tile_c * 2 + 1) * 4 + tile_r) * 4;
        uint32_t part = uint32_t(accumulator(lo)) ^ uint32_t(accumulator(lo + 1)) ^
                        uint32_t(accumulator(lo + 2)) ^ uint32_t(accumulator(lo + 3)) ^
                        uint32_t(accumulator(hi)) ^ uint32_t(accumulator(hi + 1)) ^
                        uint32_t(accumulator(hi + 2)) ^ uint32_t(accumulator(hi + 3));
        // Every lane contributes to this tile. The native path already requires SM 8.0+,
        // where one warp reduction replaces the five shuffle/XOR steps.
        part = __reduce_xor_sync(0xffffffffu, part);
        if ((lane >> 1) == tile)
          lane_part = part;
      }
      // A lane pair owns one tile. Rotate its eight words instead of dynamically indexing them:
      // NVRTC can keep this fixed-index queue in registers. Both supported profiles complete whole
      // 16-word cycles, so the queue finishes in canonical transcript order.
      if ((lane & 1) == (transcript_word & 1)) {
        const uint32_t previous = transcript[0];
        CUTE_UNROLL
        for (int word = 0; word < 7; ++word)
          transcript[word] = transcript[word + 1];
        transcript[7] = ((previous << 13) | (previous >> 19)) ^ lane_part;
      }
    }
  }

  const int local_tile = lane >> 1;
  const int tile_r = local_tile >> 2;
  const int tile_c = local_tile & 3;
  const uint64_t global_tile =
    (uint64_t(block_m) * 8u + uint64_t(warp_m * 4 + tile_r)) * uint64_t(n / 16) +
    uint64_t(block_n) * 16u + uint64_t(warp_n * 4 + tile_c);
  CUTE_UNROLL
  for (int word = 0; word < 8; ++word) {
    const uint32_t partner = __shfl_xor_sync(0xffffffffu, transcript[word], 1);
    if ((lane & 1) == 0) {
      transcript_out[global_tile * 16u + uint64_t(word * 2)] = transcript[word];
      transcript_out[global_tile * 16u + uint64_t(word * 2 + 1)] = partner;
    }
  }
}
)===CUDA"
