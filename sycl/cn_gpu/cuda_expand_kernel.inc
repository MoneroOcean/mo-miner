// DPC++ and AdaptiveCpp both leave cn/gpu below 80% of the Windows NVIDIA reference; profiling
// isolates the loss to scratchpad expansion: several materially different SYCL forms (32/64-bit
// Keccak, register-only state, coalesced stores, and explicit work-group bounds) all converge near
// 2.1-2.9 KH/s on Windows while the same DPC++ PTX reaches 3.4 KH/s on Linux. NVIDIA's runtime
// compiler can instead optimize this one integer kernel for the installed GPU and bypass that
// Windows SYCL code-generation ceiling. Releases ship only this architecture-neutral CUDA source;
// NVRTC compiles and caches forward-compatible PTX locally, the installed driver lowers it for the
// actual GPU, and any unavailable/unsupported compiler or launch automatically falls back to the
// complete SYCL pipeline.
R"===CUDA(
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

__device__ __forceinline__ uint64_t rotl64(const uint64_t value, const unsigned shift) {
  return (value << shift) | (value >> (64U - shift));
}

__device__ __forceinline__ uint64_t bitselect64(
    const uint64_t a, const uint64_t b, const uint64_t c) {
  return (a & ~c) | (b & c);
}

__device__ __forceinline__ void cn_keccak(uint64_t* const s) {
  const uint32_t rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
  };
  const uint32_t piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
  };
  const uint64_t rndc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
  };

  #pragma unroll 1
  for (unsigned round = 0; round < 24; ++round) {
    uint64_t bc[5];
    bc[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20] ^
            rotl64(s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22], 1);
    bc[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21] ^
            rotl64(s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23], 1);
    bc[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22] ^
            rotl64(s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24], 1);
    bc[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23] ^
            rotl64(s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20], 1);
    bc[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24] ^
            rotl64(s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21], 1);
    s[0] ^= bc[4]; s[5] ^= bc[4]; s[10] ^= bc[4]; s[15] ^= bc[4]; s[20] ^= bc[4];
    s[1] ^= bc[0]; s[6] ^= bc[0]; s[11] ^= bc[0]; s[16] ^= bc[0]; s[21] ^= bc[0];
    s[2] ^= bc[1]; s[7] ^= bc[1]; s[12] ^= bc[1]; s[17] ^= bc[1]; s[22] ^= bc[1];
    s[3] ^= bc[2]; s[8] ^= bc[2]; s[13] ^= bc[2]; s[18] ^= bc[2]; s[23] ^= bc[2];
    s[4] ^= bc[3]; s[9] ^= bc[3]; s[14] ^= bc[3]; s[19] ^= bc[3]; s[24] ^= bc[3];

    uint64_t t = s[1];
    #pragma unroll
    for (unsigned i = 0; i < 24; ++i) {
      const uint64_t next = s[piln[i]];
      s[piln[i]] = rotl64(t, rotc[i]);
      t = next;
    }

    #pragma unroll
    for (unsigned i = 0; i < 25; i += 5) {
      const uint64_t first = s[i];
      const uint64_t second = s[i + 1];
      s[i] = bitselect64(s[i] ^ s[i + 2], s[i], s[i + 1]);
      s[i + 1] = bitselect64(s[i + 1] ^ s[i + 3], s[i + 1], s[i + 2]);
      s[i + 2] = bitselect64(s[i + 2] ^ s[i + 4], s[i + 2], s[i + 3]);
      s[i + 3] = bitselect64(s[i + 3] ^ first, s[i + 3], s[i + 4]);
      s[i + 4] = bitselect64(s[i + 4] ^ second, s[i + 4], first);
    }
    s[0] ^= rndc[round];
  }
}

extern "C" __global__ __launch_bounds__(32, 1)
void cn_gpu_expand_cuda(
    const uint64_t* const states, uint64_t* const scratchpads, const uint32_t batch) {
  const uint64_t thread = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = uint64_t(batch) * 4096ULL;
  if (thread >= total) return;

  const uint32_t stripe = uint32_t(thread) & 4095U;
  const uint32_t hash_index = uint32_t(thread >> 12);
  const uint64_t* const input = states + uint64_t(hash_index) * 25ULL;
  uint64_t* output = scratchpads + uint64_t(hash_index) * 262144ULL +
                     uint64_t(stripe) * 64ULL;
  uint64_t hash[25];
  hash[0] = input[0] ^ stripe;
  #pragma unroll
  for (unsigned i = 1; i < 25; ++i) hash[i] = input[i];

  #pragma unroll
  for (unsigned part = 0; part < 3; ++part) {
    cn_keccak(hash);
    const unsigned words = part == 0 ? 20U : 22U;
    #pragma unroll
    for (unsigned i = 0; i < words; ++i) output[i] = hash[i];
    output += words;
  }
}
)===CUDA"
