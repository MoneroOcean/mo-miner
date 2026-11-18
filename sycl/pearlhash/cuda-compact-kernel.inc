// Native 64x96 fragments expose logical 24x8 public openings. Keep the host
// compact policy and launch geometry synchronized with this source-JIT body.
R"MOM_PEARL(// Compact rank-128 PearlHash solver: reconstruct operand fragments directly
// from rank noise, then hash the canonical 24x8 proof transcript. Requires
// native byte-transposing LDSM; host admission selects the sm120a feature target.
#include <cuda_runtime.h>
#include <cuda/std/cstdint>
#include <cute/arch/copy_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

using namespace cute;
using cuda::std::int8_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;

namespace dense_warp96 {

constexpr int kCtaM = 192;
constexpr int kCtaN = 256;
constexpr int kWarpM = 64;
constexpr int kWarpN = 96;
constexpr int kKTile = 32;
constexpr int kRank = 128;
constexpr int kAccumulatorsPerLane = 192;
constexpr int kTranscriptWords = 16;

using TiledMma = decltype(make_tiled_mma(
    SM80_16x8x32_S32U8U8S32_TN{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<_64, _96, _32>{}));

__device__ __forceinline__ uint32_t rotl13(uint32_t x) {
  // Rotate the previous FIFO contribution before folding the next checkpoint;
  // the 13-bit rotation and 16-word transcript are consensus invariants.
  return (x << 13) | (x >> 19);
}

__device__ __forceinline__ void swizzle_cta(int& block_m, int& block_n,
                                             int m, int n, int k) {
  constexpr int swizzle_m = 16;
  const int swizzle_n = k == 8192 ? 2 : 16;
  const int grid_m = m / kCtaM;
  const int grid_n = n / kCtaN;
  if (grid_m % swizzle_m == 0 && grid_n % swizzle_n == 0) {
    const int linear = int(blockIdx.y) * grid_m + int(blockIdx.x);
    const int supertile = linear / (swizzle_m * swizzle_n);
    const int local = linear % (swizzle_m * swizzle_n);
    const int supertiles_n = grid_n / swizzle_n;
    block_m = (supertile / supertiles_n) * swizzle_m + local / swizzle_n;
    block_n = (supertile % supertiles_n) * swizzle_n + local % swizzle_n;
  }
}

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

__device__ __forceinline__ void b3_g(uint32_t* s, int a, int b, int c, int d,
                                     uint32_t x, uint32_t y) {
  s[a] += s[b] + x;
  s[d] = rotr32(s[d] ^ s[a], 16);
  s[c] += s[d];
  s[b] = rotr32(s[b] ^ s[c], 12);
  s[a] += s[b] + y;
  s[d] = rotr32(s[d] ^ s[a], 8);
  s[c] += s[d];
  s[b] = rotr32(s[b] ^ s[c], 7);
}

__device__ __forceinline__ void b3_round(uint32_t* s, const uint32_t* msg) {
  b3_g(s, 0, 4, 8, 12, msg[0], msg[1]);
  b3_g(s, 1, 5, 9, 13, msg[2], msg[3]);
  b3_g(s, 2, 6, 10, 14, msg[4], msg[5]);
  b3_g(s, 3, 7, 11, 15, msg[6], msg[7]);
  b3_g(s, 0, 5, 10, 15, msg[8], msg[9]);
  b3_g(s, 1, 6, 11, 12, msg[10], msg[11]);
  b3_g(s, 2, 7, 8, 13, msg[12], msg[13]);
  b3_g(s, 3, 4, 9, 14, msg[14], msg[15]);
}

__device__ __forceinline__ uint32_t tile_mix(uint32_t row, uint32_t col,
                                              const uint32_t* transcript) {
  uint32_t h = row * 2654435761u + col * 2246822519u + 0x9e3779b9u;
#pragma unroll
  for (int word = 0; word < kTranscriptWords; ++word)
    h = (h ^ transcript[word]) * 16777619u + uint32_t(word);
  return h;
}

// Hash one lane's canonical transcript at the caller's explicit 24x8 proof
// origin. These records must not be decoded as ordinary 16x16 matrix tiles.
__device__ __forceinline__ void hash_local(
    const uint32_t (&transcript_words)[kTranscriptWords],
    const unsigned char* key, const unsigned char* target,
    PearlHashResult* result, uint32_t seed, uint32_t row, uint32_t col,
    int debug_checksum) {
  uint32_t message[kTranscriptWords];
  uint32_t state[16];
#pragma unroll
  for (int word = 0; word < kTranscriptWords; ++word) {
    message[word] = transcript_words[word];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i)
    state[i] = load32(key + i * 4);
  state[8] = 0x6a09e667u;
  state[9] = 0xbb67ae85u;
  state[10] = 0x3c6ef372u;
  state[11] = 0xa54ff53au;
  state[12] = 0;
  state[13] = 0;
  state[14] = 64;
  state[15] = 27;
  constexpr int permutation[16] =
      {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
#pragma unroll
  for (int round = 0; round < 7; ++round) {
    b3_round(state, message);
    if (round != 6) {
      uint32_t permuted[16];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        permuted[i] = message[permutation[i]];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        message[i] = permuted[i];
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

  if (debug_checksum)
    atomicAdd(&result->chk, tile_mix(row, col, transcript_words));
  if (wins && atomicExch(&result->found, 1) == 0) {
    result->seed = seed;
    result->row = row;
    result->col = col;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t word = state[i] ^ state[i + 8];
      result->jackpot[i * 4] = static_cast<unsigned char>(word);
      result->jackpot[i * 4 + 1] = static_cast<unsigned char>(word >> 8);
      result->jackpot[i * 4 + 2] = static_cast<unsigned char>(word >> 16);
      result->jackpot[i * 4 + 3] = static_cast<unsigned char>(word >> 24);
    }
  }
}

}  // namespace dense_warp96



#if defined(__CUDA_ARCH__) && !defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
#error "reconstruct requires native SM100A LDSM8 support"
#endif

using namespace cute;
using cuda::std::int8_t;
using cuda::std::int32_t;
using cuda::std::uint8_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;

namespace reconstruct96 {

constexpr int kCtaM = 192;
constexpr int kCtaN = 256;
constexpr int kRank = 128;
constexpr int kK32 = 32;
constexpr int kPitchA = 208;
constexpr int kPitchB = 256;
constexpr int kQueueWords = 16;
constexpr int kAccumulatorsPerLane = 192;
constexpr uint32_t kByteBias = 0x40404040u;
constexpr int kSharedBytes = kRank * kPitchA + kRank * kPitchB + 2 * 192 * 32 + kQueueWords * 256 * 4;

static_assert(kRank == 128, "packed selector ABI requires rank 128");
static_assert(kSharedBytes == 88064, "direct reconstruct shared storage changed");

struct PearlRawLDSM {
  using SRegisters = cute::uint128_t[1];
  using DRegisters = uint32_t[4];

  __device__ __forceinline__ static void copy(
      cute::uint128_t const& source, uint32_t& d0, uint32_t& d1,
      uint32_t& d2, uint32_t& d3) {
    const uint32_t pointer = cute::cast_smem_ptr_to_uint(&source);
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 "
                 "{%0, %1, %2, %3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(pointer));
  }
};

struct ReconstructSharedStorage {
  alignas(16) uint8_t eal_rank[kRank * kPitchA];
  alignas(16) uint8_t ebr_rank[kRank * kPitchB];
  alignas(16) uint4 a_stage[2][2][6][32];
  // Private B fragments require no shared stage.
  alignas(16) volatile uint32_t fifo[kQueueWords][256];
};

static_assert(sizeof(ReconstructSharedStorage) == kSharedBytes,
              "reconstruct storage accounting changed");

__device__ __forceinline__ uint4 rank_pair_packet(
    const uint8_t* first, const uint8_t* second) {
  uint4 lhs;
  uint4 rhs;
  PearlRawLDSM::copy(*reinterpret_cast<const cute::uint128_t*>(first), lhs.x,
                    lhs.y, lhs.z, lhs.w);
  PearlRawLDSM::copy(*reinterpret_cast<const cute::uint128_t*>(second), rhs.x,
                    rhs.y, rhs.z, rhs.w);
  return make_uint4(lhs.x + kByteBias - rhs.x,
                    lhs.y + kByteBias - rhs.y,
                    lhs.z + kByteBias - rhs.z,
                    lhs.w + kByteBias - rhs.w);
}

__device__ __forceinline__ void apply_a_prefix(uint4& packet, uint32_t seed,
                                                int kfuture, int lane) {
  if (lane >= 4)
    return;
  uint32_t low = 0;
  uint32_t high = 0;
#pragma unroll
  for (int byte = 0; byte < 4; ++byte) {
    low |= (((seed >> unsigned(kfuture + lane * 4 + byte)) & 1u) ^ 1u)
            << (8 * byte);
    high |= (((seed >> unsigned(kfuture + 16 + lane * 4 + byte)) & 1u) ^ 1u)
             << (8 * byte);
  }
  packet.x -= low;
  packet.z -= high;
}

// Each lane supplies its validated selector to the same public LDSM
// collective. Keep the pair raw until it is consumed at the next K32.
struct RawBPair { uint4 first; uint4 second; };

// Enumerate real tail blocks instead of disabling cache-local ordering for
// the whole grid. Complete groups are byte-for-byte the old fast mapping.
__device__ __forceinline__ void swizzle_with_tails(
    int& block_m, int& block_n, int m, int n, int k) {
  const int gm = m / kCtaM, gn = n / kCtaN;
  constexpr int sm = 16;
  const int sn = k == 8192 ? 2 : 16;
  if (gm % sm == 0 && gn % sn == 0) {
    dense_warp96::swizzle_cta(block_m, block_n, m, n, k);
    return;
  }
  const uint64_t linear = uint64_t(block_n) * uint64_t(gm) + uint64_t(block_m);
  const uint64_t row_group = linear / (uint64_t(sm) * uint64_t(gn));
  const int row_start = int(row_group) * sm;
  const int height = gm - row_start < sm ? gm - row_start : sm;
  const uint64_t row_local = linear - row_group * uint64_t(sm) * uint64_t(gn);
  const uint64_t col_group = row_local / (uint64_t(height) * uint64_t(sn));
  const int col_start = int(col_group) * sn;
  const int width = gn - col_start < sn ? gn - col_start : sn;
  const uint64_t local = row_local - col_group * uint64_t(height) * uint64_t(sn);
  block_m = row_start + int(local / uint64_t(width));
  block_n = col_start + int(local % uint64_t(width));
}
// Produce each native register fragment once per CTA. Whole-warp predicates
// preserve the public LDSM collective; consumers load the exact saved words,
// not a canonical byte tile requiring a second transpose.
template <bool Initial>
__device__ __forceinline__ void produce_fragment_stage(
    ReconstructSharedStorage& storage, int slot, int warp, int lane,
    uint32_t packed, uint64_t global_row_base, uint32_t seed) {
  const int owner=warp>>2,group=warp&3;
  const int first=int(packed&255u),second=int((packed>>8)&255u);
  const int row=owner*96+group*16;
  uint4 packet=rank_pair_packet(storage.eal_rank+uint64_t(first)*kPitchA+row,
                                storage.eal_rank+uint64_t(second)*kPitchA+row);
  if constexpr(Initial) {
    if(global_row_base==0&&warp==0)apply_a_prefix(packet,seed,0,lane);
  }
  storage.a_stage[slot][owner][group][lane]=make_uint4(packet.x,packet.z,packet.y,packet.w);
  // Six A packets belong to this four-warp M group. Its first two warps
  // also produce packets4 and5; neither named barrier crosses M groups.
  if(group<2) {
    const int extra=owner*96+64+group*16;
    const uint4 more=rank_pair_packet(
      storage.eal_rank+uint64_t(first)*kPitchA+extra,
      storage.eal_rank+uint64_t(second)*kPitchA+extra);
    storage.a_stage[slot][owner][4+group][lane]=make_uint4(more.x,more.z,more.y,more.w);
  }
}

// Private B uses the exact canonical shared-producer byte order, without
// a shared stage. Every call is a whole-warp LDSM collective.
__device__ __forceinline__ uint4 private_b_packet(
    ReconstructSharedStorage& storage,int owner,int group,uint32_t packed) {
  const int first=int((packed>>16)&255u),second=int(packed>>24);
  // Each 16-column slab has 128 contiguous aligned rank packets.
  const int row=owner*8192+group*2048;
  const uint4 p=rank_pair_packet(storage.ebr_rank+uint64_t(first)*16u+row,
                                storage.ebr_rank+uint64_t(second)*16u+row);
  return p;
}

// Each warp's lane and logical ownership are immutable. Cache only these
// byte bases; slot/group offsets remain compile-time fixed in the hot loop.
__device__ __forceinline__ uint4 load_stage_packet(uint32_t address) {
  uint4 out;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
    : "=r"(out.x),"=r"(out.y),"=r"(out.z),"=r"(out.w):"r"(address));
  return out;
}
// Pure scalar extraction. Upper output bytes select the explicit zero
// source; there is no sign extension, memory effect, or ordering dependency.
template<int Byte>
__device__ __forceinline__ uint32_t selector_byte(uint32_t packed) {
  static_assert(Byte>=0&&Byte<4,"selector byte outside packed word");
  {
    uint32_t value;
    asm("prmt.b32 %0,%1,0,%2;":"=r"(value):"r"(packed),"n"(0x4440|Byte));
    return value;
  }
}

// Load both already-addressed B noise rows as whole-warp collectives.
// The caller retains these rank-row bases across the four native-A packets.
__device__ __forceinline__ RawBPair load_private_b_rows(uint32_t first,uint32_t second) {
  RawBPair p;
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];"
    :"=r"(p.first.x),"=r"(p.first.y),"=r"(p.first.z),"=r"(p.first.w):"r"(first));
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];"
    :"=r"(p.second.x),"=r"(p.second.y),"=r"(p.second.z),"=r"(p.second.w):"r"(second));
  return p;
}
__device__ __forceinline__ uint4 finish_private_b(const RawBPair& p) {
  return make_uint4(p.first.x+kByteBias-p.second.x,
                   p.first.y+kByteBias-p.second.y,
                   p.first.z+kByteBias-p.second.z,
                   p.first.w+kByteBias-p.second.w);
}


// Same public LDSM operands/extraction as the original A producer. Only
// the subtract/store is deferred until after independent current-row MMA.
template<bool Extra>
__device__ __forceinline__ RawBPair load_pending_a(
    ReconstructSharedStorage& storage,int owner,int group,uint32_t packed) {
  const int first=int(packed&255u),second=int((packed>>8)&255u);
  const int row=owner*96+(group+(Extra?4:0))*16;
  RawBPair p;
  PearlRawLDSM::copy(*reinterpret_cast<const cute::uint128_t*>(
    storage.eal_rank+uint64_t(first)*kPitchA+row),p.first.x,p.first.y,p.first.z,p.first.w);
  PearlRawLDSM::copy(*reinterpret_cast<const cute::uint128_t*>(
    storage.eal_rank+uint64_t(second)*kPitchA+row),p.second.x,p.second.y,p.second.z,p.second.w);
  return p;
}
template<bool Extra>
__device__ __forceinline__ void publish_pending_a(
    ReconstructSharedStorage& storage,int slot,int owner,int group,int lane,
    const RawBPair& p) {
  // Logical A is streamed as native B in x,z,y,w word order.
  storage.a_stage[slot][owner][group+(Extra?4:0)][lane]=make_uint4(
    p.first.x+kByteBias-p.second.x,p.first.z+kByteBias-p.second.z,
    p.first.y+kByteBias-p.second.y,p.first.w+kByteBias-p.second.w);
}

// Logical B forms native A; stream transposed logical A as native B.
template<int Pair,class FragmentA,class Accumulator>
__device__ __forceinline__ void mma_column_pair(
    const uint4& bv,const FragmentA& fragment_a,Accumulator& accumulator) {
  auto a=recast<uint32_t>(fragment_a);auto c=recast<uint32_t>(accumulator);
  static_assert(decltype(size<0>(c))::value==4&&decltype(size<1>(c))::value==4&&
                decltype(size<2>(c))::value==12,"transposed C192 layout");
  // Group independent cells by B operand for native operand-cache reuse.
  // The final pair keeps its separate last-use/refill order below.
#pragma unroll
  for(int half=0;half<2;++half) {
#pragma unroll
    for(int atom=0;atom<4;++atom) {
      const int m=(Pair&1)?3-atom:atom;
      const int n=Pair*2+half;
      SM80_16x8x32_S32U8U8S32_TN::fma(
        c(0,m,n),c(1,m,n),c(2,m,n),c(3,m,n),
        a(4*m),a(4*m+1),a(4*m+2),a(4*m+3),
        half?bv.z:bv.x,half?bv.w:bv.y,
        c(0,m,n),c(1,m,n),c(2,m,n),c(3,m,n));
    }
  }
}

// Retire both final consumers before each native-A packet is refilled.
// Only one raw pair crosses the independent previous-column MMA work.
template<class FragmentA,class Accumulator>
__device__ __forceinline__ void finish_pair_and_load_next_a(
    const uint4& bv,FragmentA& fragment_a,Accumulator& accumulator,
    ReconstructSharedStorage& storage,int owner,uint32_t following,bool future) {
  auto a=recast<uint32_t>(fragment_a);auto c=recast<uint32_t>(accumulator);
  RawBPair pending{};
  uint32_t first_row=0,second_row=0;
  if(future) {
    // All four private native-A packets use the same two rank-row bases.
    const uint32_t base=cute::cast_smem_ptr_to_uint(&storage.ebr_rank[owner*8192]);
    first_row=base+selector_byte<2>(following)*16u;
    second_row=base+selector_byte<3>(following)*16u;
    pending=load_private_b_rows(first_row,second_row);
  }
#pragma unroll
  for(int group=0;group<4;++group) {
#pragma unroll
    for(int half=0;half<2;++half) {
      constexpr int Pair=5;const int n=Pair*2+half;
      SM80_16x8x32_S32U8U8S32_TN::fma(
        c(0,group,n),c(1,group,n),c(2,group,n),c(3,group,n),
        a(4*group),a(4*group+1),a(4*group+2),a(4*group+3),
        half?bv.z:bv.x,half?bv.w:bv.y,
        c(0,group,n),c(1,group,n),c(2,group,n),c(3,group,n));
    }
    if(future) {
      const uint4 p=finish_private_b(pending);
      a(group*4)=p.x;a(group*4+1)=p.y;a(group*4+2)=p.z;a(group*4+3)=p.w;
      if(group<3)pending=load_private_b_rows(first_row+2048u*(group+1),second_row+2048u*(group+1));
    }
  }
}

// Each source word is four rank bytes in one128-byte row. All source
// addresses are naturally aligned; shared destinations retain the old byte
// transpose. Per-byte modulo256 bias equals signedint8+32 for every input.
template<int Rows,int Pitch>
__device__ __forceinline__ void upload_noise_words(uint8_t* output,const int8_t* input,
                                                 uint64_t global_base,int thread) {
  static_assert(Rows%8==0&&Pitch>=Rows,"complete rank-word upload");
  for(int linear=thread;linear<Rows*128/4;linear+=256) {
    const int group=linear/32;
    const int row=(group/4)*4+(linear%32)/8,rank=(group%4)*32+(linear%8)*4;
    // Cold bijection matches the public24x8 logical opening pattern.
    const int physical_row=Rows==192?16*((row%96)/8)+8*(row/96)+row%8:
                                      64*((row%64)/16)+16*(row/64)+row%16;
    const int8_t* address=input+(global_base+uint64_t(physical_row))*128+uint64_t(rank);
    uint32_t raw;
    asm volatile("ld.global.u32 %0, [%1];":"=r"(raw):"l"(address));
#pragma unroll
    for(int byte=0;byte<4;++byte)
      if constexpr(Rows==256)
        output[((row/16)*128+(rank+byte))*16+row%16]=static_cast<uint8_t>((raw>>(8*byte))+32u);
      else
        output[(rank+byte)*Pitch+row]=static_cast<uint8_t>((raw>>(8*byte))+32u);
  }
}

template<int FixedK>
__device__ __forceinline__ void compact_gemm_impl(
                  ReconstructSharedStorage& storage,const int8_t* __restrict__ eal,
                  const int8_t* __restrict__ ebr,
                  const uint32_t* __restrict__ packed_indices,
                  const int32_t* __restrict__ a_second_compat_unused,
                  const int32_t* __restrict__ b_first_compat_unused,
                  const int32_t* __restrict__ b_second_compat_unused,
                  uint32_t seed, uint32_t b_seed, int m, int n, int input_k,
                  uint32_t* __restrict__ transcript_out,
                  const unsigned char* __restrict__ key,
                  const unsigned char* __restrict__ target,
                  dense_warp96::PearlHashResult* __restrict__ result,
                  int row_offset, int debug_checksum) {
  const int k = FixedK == 0 ? input_k : FixedK;
  (void)a_second_compat_unused;
  (void)b_first_compat_unused;
  (void)b_second_compat_unused;
  (void)b_seed;
  if (m <= 0 || n <= 0 || k <= 0 || (m % kCtaM) != 0 ||
      (n % kCtaN) != 0 || (k % kRank) != 0)
    return;

  // Optional transcript output is for correctness checks; normal search is fused.
  if (!key || !target || !result || row_offset < 0 ||
      uint64_t(row_offset) + uint64_t(m) > 0xffffffffu)
    return;

  int block_m = int(blockIdx.x);
  int block_n = int(blockIdx.y);
  swizzle_with_tails(block_m, block_n, m, n, k);

  const int thread = int(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int warp_m = warp >> 2;
  const int warp_n = warp & 3;
  const uint64_t global_row_base = uint64_t(block_m) * kCtaM;
  const uint64_t global_col_base = uint64_t(block_n) * kCtaN;

  upload_noise_words<kCtaM,kPitchA>(storage.eal_rank,eal,global_row_base,thread);
  upload_noise_words<kCtaN,kPitchB>(storage.ebr_rank,ebr,global_col_base,thread);
  __syncthreads();

  using TiledMma = dense_warp96::TiledMma;
  TiledMma mma;
  auto thread_mma = mma.get_slice(lane);
  auto static_a32 = make_tensor(
      make_smem_ptr(reinterpret_cast<int8_t*>(0)),
      Layout<Shape<_64, _32>, Stride<_32, _1>>{});
  auto static_b32 = make_tensor(
      make_smem_ptr(reinterpret_cast<int8_t*>(0)),
      Layout<Shape<_96, _32>, Stride<_32, _1>>{});
  auto fragment_a = thread_mma.partition_fragment_A(static_a32);
  auto fragment_b = thread_mma.partition_fragment_B(static_b32);

  auto identity = make_identity_tensor(make_shape(_64{}, _96{}));
  auto output_tile = local_tile(
      identity, Shape<_64, _96, _32>{}, make_coord(0, 0, _),
      Step<_1, _1, X>{});
  auto thread_output = thread_mma.partition_C(output_tile);
  auto accumulator = thread_mma.make_fragment_C(thread_output);
  static_assert(decltype(size(accumulator))::value == kAccumulatorsPerLane,
                "64x96 C fragment must remain 192 values per lane");
  clear(accumulator);

  // Each thread owns one FIFO column; packet stages never alias it.
#pragma unroll
  for(int word=0;word<kQueueWords;++word)storage.fifo[word][thread]=0;
  const uint32_t initial_packed=packed_indices[lane];
  uint32_t following_word=packed_indices[32+lane];
  produce_fragment_stage<true>(storage,0,warp,lane,initial_packed,global_row_base,seed);
  // Initial A is published once to the whole CTA; raw tables are immutable.
  __syncthreads();
  constexpr int k32_per_checkpoint = kRank / kK32;
  static_assert(k32_per_checkpoint == 4,
                "explicit checkpoint loop requires four K32 slices");
  const uint32_t a_stage_base=cute::cast_smem_ptr_to_uint(&storage.a_stage[0][warp_m][0][lane]);
  // B is immutable input, privately reconstructed for this consumer warp.
  auto initial_b=recast<uint32_t>(fragment_a);
#pragma unroll
  for(int group=0;group<4;++group) {
    const uint4 p=private_b_packet(storage,warp_n,group,initial_packed);
    initial_b(group*4)=p.x;initial_b(group*4+1)=p.y;
    initial_b(group*4+2)=p.z;initial_b(group*4+3)=p.w;
  }
  const int checkpoint_count = k / kRank;
#pragma unroll (FixedK == 65536 ? 4 : 1)
  for (int checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
    // The private FIFO value is unchanged by matrix work; load it at use.
#pragma unroll
    for (int step = 0; step < k32_per_checkpoint; ++step) {
      const int k32 = checkpoint * k32_per_checkpoint + step;
      const int slot=k32&1;
      const bool future=step+1<k32_per_checkpoint||checkpoint+1<checkpoint_count;
      const uint32_t following=future?following_word:0;
      const bool prefetch=step+2<k32_per_checkpoint||checkpoint+1<checkpoint_count;
      if(prefetch)following_word=packed_indices[(k32+2)*kK32+lane];
      RawBPair pending_a{};
      // Next B is reconstructed at each final-row last use.
      uint4 row_a=load_stage_packet(a_stage_base+uint32_t(slot)*6144u);
      {
        const uint4 av=row_a;
        row_a=load_stage_packet(a_stage_base+uint32_t(slot)*6144u+512u);
        if(future)pending_a=load_pending_a<false>(storage,warp_m,warp_n,following);
        mma_column_pair<0>(av,fragment_a,accumulator);
        if(future)publish_pending_a<false>(storage,slot^1,warp_m,warp_n,lane,pending_a);
      }
      {
        const uint4 av=row_a;
        row_a=load_stage_packet(a_stage_base+uint32_t(slot)*6144u+1024u);
        if(future&&warp_n<2)pending_a=load_pending_a<true>(storage,warp_m,warp_n,following);
        mma_column_pair<1>(av,fragment_a,accumulator);
        if(future&&warp_n<2)publish_pending_a<true>(storage,slot^1,warp_m,warp_n,lane,pending_a);
      }
      {
        const uint4 av=row_a;
        row_a=load_stage_packet(a_stage_base+uint32_t(slot)*6144u+1536u);
        // B group2 is refilled only after its final-row consumers.
        mma_column_pair<2>(av,fragment_a,accumulator);
      }
      {
        const uint4 av=row_a;
        row_a=load_stage_packet(a_stage_base+uint32_t(slot)*6144u+2048u);
        // B group3 is refilled only after its final-row consumers.
        mma_column_pair<3>(av,fragment_a,accumulator);
      }
      // All four B groups are still the current generation here.
      {
        const uint4 av=row_a;
        row_a=load_stage_packet(a_stage_base+uint32_t(slot)*6144u+2560u);
        mma_column_pair<4>(av,fragment_a,accumulator);
      }
      // Every current A read in this M group precedes this barrier; publish
      // next A and free the old slot. B/FIFO have no inter-warp write sharing.
      asm volatile("bar.sync %0, 128;"::"r"(warp_m+1):"memory");
      finish_pair_and_load_next_a(row_a,fragment_a,accumulator,storage,warp_n,following,future);
    }

    {
      const uint32_t previous=storage.fifo[checkpoint&15][thread];
      uint32_t lane_part = 0;
#pragma unroll
      for (int i = 0; i < kAccumulatorsPerLane; ++i)
        lane_part ^= uint32_t(accumulator(i));
      storage.fifo[checkpoint&15][thread]=dense_warp96::rotl13(previous)^lane_part;
    }
  }

  // Accumulators are dead: materialize the canonical FIFO only for hashing.
  uint32_t transcript[kQueueWords];
#pragma unroll
  for(int word=0;word<kQueueWords;++word)
    transcript[word]=storage.fifo[(checkpoint_count+word)&15][thread];
  const uint64_t global_warp_row = uint64_t(block_m) * 2u +
                                   uint64_t(warp_m);
  const uint64_t global_warp_col = uint64_t(block_n) * 4u +
                                   uint64_t(warp_n);
  const uint64_t dense_index =
      (global_warp_row * (uint64_t(n) / 64u) + global_warp_col) * 32u +
      uint64_t(lane);
  // Keep optional byte-exact output for qualification; normal fusion avoids it.
  if (transcript_out != nullptr) {
#pragma unroll
    for (int word = 0; word < kQueueWords; ++word)
      transcript_out[dense_index * kQueueWords + uint64_t(word)] = transcript[word];
  }
  const uint32_t row = uint32_t(uint64_t(row_offset) + uint64_t(block_m)*192u + uint64_t(warp_m)*8u + uint64_t(2*(lane&3)));
  const uint32_t col = uint32_t(uint64_t(block_n)*256u + uint64_t(warp_n)*16u + uint64_t(lane>>2));
  dense_warp96::hash_local(transcript, key, target, result, seed, row, col, debug_checksum);
}

extern "C" __global__ __launch_bounds__(256)
void compact_gemm(const int8_t* __restrict__ eal,
                  const int8_t* __restrict__ ebr,
                  const uint32_t* __restrict__ packed_indices,
                  const int32_t* __restrict__ a_second_compat_unused,
                  const int32_t* __restrict__ b_first_compat_unused,
                  const int32_t* __restrict__ b_second_compat_unused,
                  uint32_t seed, uint32_t b_seed, int m, int n, int k,
                  uint32_t* __restrict__ transcript_out,
                  const unsigned char* __restrict__ key,
                  const unsigned char* __restrict__ target,
                  dense_warp96::PearlHashResult* __restrict__ result,
                  int row_offset, int debug_checksum) {
  // Public entry-scope optimization hint, compiled directly by NVRTC13.3.
  asm volatile(".pragma \"mma_throughput\";");
  __shared__ ReconstructSharedStorage storage;
  if (k == 65536)
    compact_gemm_impl<65536>(storage,eal,ebr,packed_indices,a_second_compat_unused,b_first_compat_unused,b_second_compat_unused,
                      seed,b_seed,m,n,k,transcript_out,key,target,result,row_offset,debug_checksum);
  else
    compact_gemm_impl<0>(storage,eal,ebr,packed_indices,a_second_compat_unused,b_first_compat_unused,b_second_compat_unused,
                      seed,b_seed,m,n,k,transcript_out,key,target,result,row_offset,debug_checksum);
}

}  // namespace reconstruct96





// The same source-JIT module owns selector preparation and fused GEMM.
#include <cuda_runtime.h>
#include <cuda/std/cstdint>
using cuda::std::uint32_t;

struct BankGroup {
  unsigned char ranks[128],banks[8],maximum,at_maximum;
};
// Score replacing a known-present rank without changing counts. Duplicate
// ranks alter bank pressure only when the first or last use changes.
__device__ __forceinline__ unsigned swap_max(const BankGroup& g,unsigned outgoing,unsigned incoming) {
  if(outgoing==incoming)return g.maximum;
  const unsigned a=outgoing&7,b=incoming&7;
  const unsigned remove=g.ranks[outgoing]==1,add=g.ranks[incoming]==0;
  if(a==b)return max(unsigned(g.maximum)-(remove>add&&g.banks[a]==g.maximum&&g.at_maximum==1),unsigned(g.banks[a])+add-remove);
  return max(unsigned(g.maximum)-(remove&&g.banks[a]==g.maximum&&g.at_maximum==1),unsigned(g.banks[b])+add);
}
__device__ __forceinline__ void refresh(BankGroup& g) {
  unsigned m=0,n=0;
#pragma unroll
  for(unsigned b=0;b<8;++b)m=max(m,unsigned(g.banks[b]));
#pragma unroll
  for(unsigned b=0;b<8;++b)n+=g.banks[b]==m;
  g.maximum=m;g.at_maximum=n;
}
__device__ __forceinline__ void exchange(BankGroup& g,unsigned outgoing,unsigned incoming) {
  if(--g.ranks[outgoing]==0)--g.banks[outgoing&7];
  if(g.ranks[incoming]++==0)++g.banks[incoming&7];
  refresh(g);
}
// One CTA owns an entire checkpoint. Every barrier is reached by all128
// threads; only the tuple/group updates are conditional, never the barriers.
__device__ __forceinline__ uint32_t swap_checkpoint(
    uint32_t (&words)[128],BankGroup (&groups)[16][4],unsigned thread) {
  const unsigned lane=thread&31,warp=thread>>5;
  auto* bytes=reinterpret_cast<unsigned char*>(groups);
  for(unsigned i=thread;i<sizeof(groups);i+=128)bytes[i]=0;
  __syncthreads();
  if(thread<64) {
    const unsigned group=thread/4,stream=thread&3;
    auto& g=groups[group][stream];
#pragma unroll
    for(unsigned i=0;i<8;++i) {
      const unsigned rank=(words[group*8+i]>>(8*stream))&127;
      if(g.ranks[rank]++==0)++g.banks[rank&7];
    }
    refresh(g);
  }
  __syncthreads();
  __shared__ unsigned keys[4];
  __shared__ uint32_t partner_words[4];
#pragma unroll 1
  for(unsigned i=0;i<120;++i) {
    const uint32_t outgoing=words[i],word=words[thread];
    const unsigned x=i/8,y=thread/8;
    int delta=0;
    if(y>x) {
#pragma unroll
      for(unsigned stream=0;stream<4;++stream) {
        const unsigned a=(outgoing>>(8*stream))&127,b=(word>>(8*stream))&127;
        const auto& gx=groups[x][stream];const auto& gy=groups[y][stream];
        delta+=(stream<2?5:8)*(int(swap_max(gx,a,b))+int(swap_max(gy,b,a))-gx.maximum-gy.maximum);
      }
    }
    // 128 encodes the global partner index, preserving lowest-index ties.
    const unsigned local=__reduce_min_sync(0xffffffffu,unsigned(delta+256)*128+thread);
    const uint32_t selected=__shfl_sync(0xffffffffu,word,local&31);
    if(lane==0){keys[warp]=local;partner_words[warp]=selected;}
    __syncthreads();
    const unsigned key=min(min(keys[0],keys[1]),min(keys[2],keys[3]));
    if((key>>7)<256) {
      const unsigned partner=key&127,other=partner/8;
      const uint32_t incoming=partner_words[partner>>5];
      if(thread<4) {
        const unsigned a=(outgoing>>(8*thread))&127,b=(incoming>>(8*thread))&127;
        exchange(groups[x][thread],a,b);exchange(groups[other][thread],b,a);
      }
      if(thread==i)words[i]=incoming;
      if(thread==partner)words[partner]=outgoing;
    }
    // Also protects keys/partner_words when no swap was accepted. Omitting
    // this for non-improving iterations races the next warp reduction.
    __syncthreads();
  }
  return words[thread];
}

// One checkpoint per CTA: warp0 partitions selectors, then four warps refine
// their K32 blocks. Preserve checkpoint0: the seed-bearing A prefix is injected
// at its canonical K positions. Only later constant-raw checkpoints may reorder.
extern "C" __global__ __launch_bounds__(128)
void selector_prepare(const uint32_t* input,uint32_t* output,unsigned k) {
  if(blockDim.x!=128||k<128||(k%128)||blockIdx.y||blockIdx.z)return;
  const unsigned thread=threadIdx.x,lane=thread&31,warp=thread>>5;
  const unsigned base=blockIdx.x*128;if(base>=k)return;
  const uint32_t original=input[base+thread];
  __shared__ unsigned invalid;
  if(thread==0)invalid=0;
  __syncthreads();
  if((original&0x80808080u)!=0)atomicOr(&invalid,1u);
  __syncthreads();if(invalid)return;
  if(base==0){output[thread]=original;return;}
  __shared__ uint32_t canonical[128],ordered[128];
  __shared__ unsigned char counts[16][4][8],sizes[16];
  __shared__ BankGroup groups[16][4];
  canonical[thread]=original;
  auto* bytes=reinterpret_cast<unsigned char*>(counts);
  for(unsigned i=thread;i<sizeof(counts);i+=128)bytes[i]=0;
  // Partition runs only on warp0; refinement starts after the CTA fence
  // below. Reuse its rank counters as seen bits, then swap_word clears them.
  auto* seen_bytes=reinterpret_cast<unsigned char*>(groups);
  for(unsigned i=thread;i<sizeof(groups);i+=128)seen_bytes[i]=0;
  if(thread<16)sizes[thread]=0;
  __syncthreads();
  if(warp==0) {
#pragma unroll 1
    for(unsigned i=0;i<128;++i) {
      const uint32_t word=canonical[i];unsigned key=0xffffffffu;
      if(lane<16&&sizes[lane]<8) {
        unsigned score=0;
#pragma unroll
        for(unsigned s=0;s<4;++s) {
          const unsigned rank=(word>>(8*s))&127;
          // Match refinement's distinct-rank bank objective: a duplicate
          // contributes no new address to this bin's working set.
          score+=(s<2?5:8)*(groups[lane][s].ranks[rank]?0:2*counts[lane][s][rank&7]+1);
        }
        key=(score*9+sizes[lane])*16+lane;
      }
      const unsigned winner=__reduce_min_sync(0xffffffffu,key)&15u;
      if(lane==0)ordered[winner*8+sizes[winner]]=word;
      if(lane<4) {
        const unsigned rank=(word>>(8*lane))&127;
        if(!groups[winner][lane].ranks[rank]) {
          groups[winner][lane].ranks[rank]=1;
          ++counts[winner][lane][rank&7];
        }
      }
      __syncwarp();
      if(lane==0)++sizes[winner];
      __syncwarp();
    }
  }
  __syncthreads();
  const uint32_t word=swap_checkpoint(ordered,groups,thread);
  output[base+thread]=word;
}



)MOM_PEARL"
