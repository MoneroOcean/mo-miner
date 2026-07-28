namespace mom_autolykos2 {

#if defined(MOM_SYCL_PORTABLE_OPENCL)
#define AUTOLYKOS_DEVICE_INLINE inline __attribute__((always_inline))
#else
#define AUTOLYKOS_DEVICE_INLINE inline
#endif

constexpr uint32_t CONST_MES_SIZE       = 8192;
constexpr uint32_t K_LEN                = 32;
constexpr uint32_t INIT_N_LEN           = 0x04000000U;
constexpr uint32_t MAX_N_LEN            = 0x7FC9FF98U;
constexpr uint32_t INCREASE_START       = 600U * 1024U;
constexpr uint32_t INCREASE_END         = 4198400U;
constexpr uint32_t INCREASE_PERIOD      = 50U * 1024U;
constexpr uint32_t TABLE_ENTRY_BYTES    = 32;
constexpr uint32_t MAX_AUTOLYKOS_OUTPUTS = 15;

constexpr uint64_t BLAKE2B_IV[8] = {
  0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
  0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
  0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
  0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

constexpr uint8_t BLAKE2B_SIGMA[12][16] = {
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },
  {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 },
  { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },
  { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9 },
  {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11 },
  {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10 },
  { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5 },
  {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 }
};

template <bool Portable = mom_sycl_portable_opencl>
inline uint64_t blake2b_iv(const unsigned index) {
  if constexpr (Portable) {
    switch (index) {
      case 0: return 0x6A09E667F3BCC908ULL;
      case 1: return 0xBB67AE8584CAA73BULL;
      case 2: return 0x3C6EF372FE94F82BULL;
      case 3: return 0xA54FF53A5F1D36F1ULL;
      case 4: return 0x510E527FADE682D1ULL;
      case 5: return 0x9B05688C2B3E6C1FULL;
      case 6: return 0x1F83D9ABFB41BD6BULL;
      default: return 0x5BE0CD19137E2179ULL;
    }
  } else {
    return BLAKE2B_IV[index];
  }
}

AUTOLYKOS_DEVICE_INLINE uint8_t blake2b_sigma(const unsigned round, const unsigned index) {
  // Rounds 1 and 11 share this permutation. Keeping it as the initialized/default value avoids an
  // llvm-spirv switch phi with two identical incoming edges, which Mesa's SPIR-V validator rejects.
  uint64_t packed = 0x357b20c16df984aeULL;
  switch (round) {
    case 0:  packed = 0xfedcba9876543210ULL; break;
    case 2:  packed = 0x491763eadf250c8bULL; break;
    case 3:  packed = 0x8f04a562ebcd1397ULL; break;
    case 4:  packed = 0xd386cb1efa427509ULL; break;
    case 5:  packed = 0x91ef57d438b0a6c2ULL; break;
    case 6:  packed = 0xb8293670a4def15cULL; break;
    case 7:  packed = 0xa2684f05931ce7bdULL; break;
    case 8:  packed = 0x5a417d2c803b9ef6ULL; break;
    case 9:  packed = 0x0dc3e9bf5167482aULL; break;
    case 10: packed = 0xfedcba9876543210ULL; break;
    default: break;
  }
  return static_cast<uint8_t>((packed >> (index * 4U)) & 0xfU);
}

struct AutolykosJobData {
  uint8_t message[HASH_LEN];
  uint8_t target[HASH_LEN];
};

struct AutolykosResult {
  uint32_t count;
  uint64_t nonce[MAX_AUTOLYKOS_OUTPUTS];
  uint8_t output[MAX_AUTOLYKOS_OUTPUTS][HASH_LEN];
};

struct AutolykosPortableResult {
  uint32_t found;
  uint64_t nonce;
  uint8_t output[HASH_LEN];
};

inline uint32_t round_up(const uint32_t value, const uint32_t step) {
  return ((value + step - 1) / step) * step;
}

// Fast unsigned remainder by a runtime divisor. The K_LEN=32 dataset reads do `idx % n_len` per nonce
// and n_len is a per-height runtime value. Keeping this divide-free helps both NVIDIA's nvptx image
// and Intel Level Zero on Arc. M = floor(2^64 / n_len) + 1 is precomputed once on the host (mo_modM),
// then x % n_len == hi64( (M*x mod 2^64) * n_len ). Bit-exact for x, n_len in [1, 2^32).
inline uint64_t mo_modM(const uint32_t d) { return d ? (0xFFFFFFFFFFFFFFFFULL / d) + 1ULL : 0ULL; }
AUTOLYKOS_DEVICE_INLINE uint32_t mo_mod_u32(const uint32_t x, const uint32_t d, const uint64_t M) {
  return static_cast<uint32_t>(mo_mul_hi_u64(M * static_cast<uint64_t>(x), static_cast<uint64_t>(d)));
}

inline uint64_t mo_divM64(const uint32_t d) {
  if (!d) return 0;
  const uint64_t q = 0xFFFFFFFFFFFFFFFFULL / d;
  return (0xFFFFFFFFFFFFFFFFULL % d) == static_cast<uint64_t>(d - 1U) ? q + 1ULL : q;
}

// Exact Barrett reduction for h3 = uint64 % n_len. For x < 2^64 and d < 2^32,
// q = high64(x * floor(2^64/d)) is at most one low, so one subtraction after
// x - q*d produces the exact remainder.
AUTOLYKOS_DEVICE_INLINE uint32_t mo_mod_u64(const uint64_t x, const uint32_t d, const uint64_t M) {
  if (d <= 1) return 0;
  const uint64_t q = mo_mul_hi_u64(x, M);
  uint64_t r = x - q * static_cast<uint64_t>(d);
  if (r >= d) r -= d;
  return static_cast<uint32_t>(r);
}

AUTOLYKOS_DEVICE_INLINE uint64_t bswap64_dev(const uint64_t value) {
  if constexpr (mom_sycl_portable_opencl) {
    // Keep this as ordinary shifts for OpenCL SPIR-V consumers. LLVM otherwise recognizes the
    // expression as llvm.bswap, which llvm-spirv can leave as an unresolved implementation helper.
    volatile uint64_t source = value;
    uint64_t result = 0;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      const volatile uint64_t byte = (source >> (i * 8U)) & 0xffU;
      result |= byte << ((7U - i) * 8U);
    }
    return result;
  } else {
    return ((value & 0x00000000000000FFULL) << 56) |
           ((value & 0x000000000000FF00ULL) << 40) |
           ((value & 0x0000000000FF0000ULL) << 24) |
           ((value & 0x00000000FF000000ULL) << 8) |
           ((value & 0x000000FF00000000ULL) >> 8) |
           ((value & 0x0000FF0000000000ULL) >> 24) |
           ((value & 0x00FF000000000000ULL) >> 40) |
           ((value & 0xFF00000000000000ULL) >> 56);
  }
}

AUTOLYKOS_DEVICE_INLINE uint32_t bswap32_dev(const uint32_t value) {
  volatile uint32_t source = value;
  uint32_t result = 0;
#pragma unroll
  for (unsigned i = 0; i < 4; ++i) {
    const volatile uint32_t byte = (source >> (i * 8U)) & 0xffU;
    result |= byte << ((3U - i) * 8U);
  }
  return result;
}

AUTOLYKOS_DEVICE_INLINE uint64_t rotr64_dev(const uint64_t value, const unsigned shift) {
  if constexpr (mom_sycl_portable_opencl) {
    volatile uint64_t right = value >> shift;
    volatile uint64_t left = value << (64U - shift);
    return right | left;
  } else {
    return (value >> shift) | (value << (64U - shift));
  }
}

AUTOLYKOS_DEVICE_INLINE uint32_t load32_be_dev(const uint8_t* const input) {
  if constexpr (mom_sycl_portable_opencl) {
    const volatile uint8_t* const bytes = input;
    uint32_t value = static_cast<uint32_t>(bytes[0]) << 24;
    value |= static_cast<uint32_t>(bytes[1]) << 16;
    value |= static_cast<uint32_t>(bytes[2]) << 8;
    value |= static_cast<uint32_t>(bytes[3]);
    return value;
  } else {
    return (static_cast<uint32_t>(input[0]) << 24) |
           (static_cast<uint32_t>(input[1]) << 16) |
           (static_cast<uint32_t>(input[2]) << 8) |
            static_cast<uint32_t>(input[3]);
  }
}

AUTOLYKOS_DEVICE_INLINE uint64_t load64_be_dev(const uint8_t* const input) {
  return (static_cast<uint64_t>(load32_be_dev(input)) << 32) |
          static_cast<uint64_t>(load32_be_dev(input + 4));
}

AUTOLYKOS_DEVICE_INLINE uint64_t load64_le_dev(const uint8_t* const input) {
  return static_cast<uint64_t>(input[0]) |
         (static_cast<uint64_t>(input[1]) << 8) |
         (static_cast<uint64_t>(input[2]) << 16) |
         (static_cast<uint64_t>(input[3]) << 24) |
         (static_cast<uint64_t>(input[4]) << 32) |
         (static_cast<uint64_t>(input[5]) << 40) |
         (static_cast<uint64_t>(input[6]) << 48) |
         (static_cast<uint64_t>(input[7]) << 56);
}

AUTOLYKOS_DEVICE_INLINE void store32_be_dev(uint8_t* const output, const uint32_t value) {
  if constexpr (mom_sycl_portable_opencl) {
    volatile uint8_t* const bytes = output;
    bytes[0] = static_cast<uint8_t>(value >> 24);
    bytes[1] = static_cast<uint8_t>(value >> 16);
    bytes[2] = static_cast<uint8_t>(value >> 8);
    bytes[3] = static_cast<uint8_t>(value);
  } else {
    output[0] = static_cast<uint8_t>(value >> 24);
    output[1] = static_cast<uint8_t>(value >> 16);
    output[2] = static_cast<uint8_t>(value >> 8);
    output[3] = static_cast<uint8_t>(value);
  }
}

AUTOLYKOS_DEVICE_INLINE uint8_t word_byte_be_dev(const uint32_t word, const unsigned byte_index) {
  return static_cast<uint8_t>(word >> ((3U - byte_index) * 8U));
}

AUTOLYKOS_DEVICE_INLINE uint64_t pack_be32_pair_as_blake_word(const uint32_t a, const uint32_t b) {
  if constexpr (mom_sycl_portable_opencl) {
    volatile uint32_t av = a, bv = b;
    uint64_t result = static_cast<uint64_t>((av >> 24) & 0xffU);
    result |= static_cast<uint64_t>((av >> 16) & 0xffU) << 8;
    result |= static_cast<uint64_t>((av >> 8) & 0xffU) << 16;
    result |= static_cast<uint64_t>(av & 0xffU) << 24;
    result |= static_cast<uint64_t>((bv >> 24) & 0xffU) << 32;
    result |= static_cast<uint64_t>((bv >> 16) & 0xffU) << 40;
    result |= static_cast<uint64_t>((bv >> 8) & 0xffU) << 48;
    result |= static_cast<uint64_t>(bv & 0xffU) << 56;
    return result;
  } else {
    return (static_cast<uint64_t>((a >> 24) & 0xffU) << 0) |
           (static_cast<uint64_t>((a >> 16) & 0xffU) << 8) |
           (static_cast<uint64_t>((a >> 8) & 0xffU) << 16) |
           (static_cast<uint64_t>(a & 0xffU) << 24) |
           (static_cast<uint64_t>((b >> 24) & 0xffU) << 32) |
           (static_cast<uint64_t>((b >> 16) & 0xffU) << 40) |
           (static_cast<uint64_t>((b >> 8) & 0xffU) << 48) |
           (static_cast<uint64_t>(b & 0xffU) << 56);
  }
}

AUTOLYKOS_DEVICE_INLINE void blake2b_g(
  uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d,
  const uint64_t x, const uint64_t y
) {
  a = a + b + x;
  d = rotr64_dev(d ^ a, 32);
  c += d;
  b = rotr64_dev(b ^ c, 24);
  a = a + b + y;
  d = rotr64_dev(d ^ a, 16);
  c += d;
  b = rotr64_dev(b ^ c, 63);
}

AUTOLYKOS_DEVICE_INLINE void blake2b_compress_dev(uint64_t h[8], const uint64_t m[16], const uint64_t t, const bool last) {
  uint64_t v[16];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) v[i] = h[i];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    if constexpr (mom_sycl_portable_opencl) v[i + 8] = blake2b_iv(i);
    else v[i + 8] = BLAKE2B_IV[i];
  }
  v[12] ^= t;
  if (last) v[14] = ~v[14];

  // Native builds unroll this loop and fold the constant table. The generic OpenCL profile keeps
  // the compact switch-based form to avoid multi-minute JIT compilation on CPU implementations.
#if defined(MOM_SYCL_PORTABLE_OPENCL)
#pragma unroll 1
#else
#pragma unroll
#endif
  for (unsigned r = 0; r < 12; ++r) {
    if constexpr (mom_sycl_portable_opencl) {
      blake2b_g(v[0], v[4], v[ 8], v[12], m[blake2b_sigma(r,  0)], m[blake2b_sigma(r,  1)]);
      blake2b_g(v[1], v[5], v[ 9], v[13], m[blake2b_sigma(r,  2)], m[blake2b_sigma(r,  3)]);
      blake2b_g(v[2], v[6], v[10], v[14], m[blake2b_sigma(r,  4)], m[blake2b_sigma(r,  5)]);
      blake2b_g(v[3], v[7], v[11], v[15], m[blake2b_sigma(r,  6)], m[blake2b_sigma(r,  7)]);
      blake2b_g(v[0], v[5], v[10], v[15], m[blake2b_sigma(r,  8)], m[blake2b_sigma(r,  9)]);
      blake2b_g(v[1], v[6], v[11], v[12], m[blake2b_sigma(r, 10)], m[blake2b_sigma(r, 11)]);
      blake2b_g(v[2], v[7], v[ 8], v[13], m[blake2b_sigma(r, 12)], m[blake2b_sigma(r, 13)]);
      blake2b_g(v[3], v[4], v[ 9], v[14], m[blake2b_sigma(r, 14)], m[blake2b_sigma(r, 15)]);
    } else {
      // Preserve the proven native source shape. DPC++ HIP miscompiled the equivalent helper-lambda
      // indexing form on gfx1200 and produced a deterministic wrong Autolykos hash.
      blake2b_g(v[0], v[4], v[ 8], v[12], m[BLAKE2B_SIGMA[r][ 0]], m[BLAKE2B_SIGMA[r][ 1]]);
      blake2b_g(v[1], v[5], v[ 9], v[13], m[BLAKE2B_SIGMA[r][ 2]], m[BLAKE2B_SIGMA[r][ 3]]);
      blake2b_g(v[2], v[6], v[10], v[14], m[BLAKE2B_SIGMA[r][ 4]], m[BLAKE2B_SIGMA[r][ 5]]);
      blake2b_g(v[3], v[7], v[11], v[15], m[BLAKE2B_SIGMA[r][ 6]], m[BLAKE2B_SIGMA[r][ 7]]);
      blake2b_g(v[0], v[5], v[10], v[15], m[BLAKE2B_SIGMA[r][ 8]], m[BLAKE2B_SIGMA[r][ 9]]);
      blake2b_g(v[1], v[6], v[11], v[12], m[BLAKE2B_SIGMA[r][10]], m[BLAKE2B_SIGMA[r][11]]);
      blake2b_g(v[2], v[7], v[ 8], v[13], m[BLAKE2B_SIGMA[r][12]], m[BLAKE2B_SIGMA[r][13]]);
      blake2b_g(v[3], v[4], v[ 9], v[14], m[BLAKE2B_SIGMA[r][14]], m[BLAKE2B_SIGMA[r][15]]);
    }
  }

#pragma unroll
  for (unsigned i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
}

AUTOLYKOS_DEVICE_INLINE void blake2b256_init_dev(uint64_t h[8]) {
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    if constexpr (mom_sycl_portable_opencl) h[i] = blake2b_iv(i);
    else h[i] = BLAKE2B_IV[i];
  }
  h[0] ^= 0x01010020ULL;
}

AUTOLYKOS_DEVICE_INLINE void blake2b256_output_dev(const uint64_t h[8], uint8_t out[32]) {
#pragma unroll
  for (unsigned i = 0; i < 32; ++i) out[i] = static_cast<uint8_t>(h[i >> 3] >> ((i & 7U) * 8U));
}

AUTOLYKOS_DEVICE_INLINE void blake2b256_oneblock_dev(const uint8_t* const input, const uint32_t len, uint8_t out[32]) {
  uint64_t h[8];
  uint64_t m[16];
  blake2b256_init_dev(h);
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) m[i] = 0;
  for (uint32_t i = 0; i < len; ++i) {
    m[i >> 3] |= static_cast<uint64_t>(input[i]) << ((i & 7U) * 8U);
  }
  blake2b_compress_dev(h, m, len, true);
  blake2b256_output_dev(h, out);
}

AUTOLYKOS_DEVICE_INLINE void blake2b256_71_dev(const uint8_t input[71], uint8_t out[32]) {
  uint64_t h[8];
  uint64_t m[16];
  blake2b256_init_dev(h);
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) m[i] = 0;
#pragma unroll
  for (unsigned i = 0; i < 71; ++i) {
    m[i >> 3] |= static_cast<uint64_t>(input[i]) << ((i & 7U) * 8U);
  }
  blake2b_compress_dev(h, m, 71, true);
  blake2b256_output_dev(h, out);
}

AUTOLYKOS_DEVICE_INLINE void blake2b256_message_nonce_dev(const uint8_t message[32], const uint64_t nonce, uint8_t out[32]) {
  uint64_t h[8];
  uint64_t m[16];
  blake2b256_init_dev(h);
  m[0] = load64_le_dev(message);
  m[1] = load64_le_dev(message + 8);
  m[2] = load64_le_dev(message + 16);
  m[3] = load64_le_dev(message + 24);
  m[4] = bswap64_dev(nonce);
#pragma unroll
  for (unsigned i = 5; i < 16; ++i) m[i] = 0;
  blake2b_compress_dev(h, m, 40, true);
  blake2b256_output_dev(h, out);
}

AUTOLYKOS_DEVICE_INLINE void blake2b256_sum_dev(const uint32_t sum[8], uint8_t out[32]) {
  uint64_t h[8];
  uint64_t m[16];
  blake2b256_init_dev(h);
  m[0] = pack_be32_pair_as_blake_word(sum[0], sum[1]);
  m[1] = pack_be32_pair_as_blake_word(sum[2], sum[3]);
  m[2] = pack_be32_pair_as_blake_word(sum[4], sum[5]);
  m[3] = pack_be32_pair_as_blake_word(sum[6], sum[7]);
#pragma unroll
  for (unsigned i = 4; i < 16; ++i) m[i] = 0;
  blake2b_compress_dev(h, m, 32, true);
  blake2b256_output_dev(h, out);
}

AUTOLYKOS_DEVICE_INLINE void autolykos_prehash_digest_dev(const uint32_t index, const uint32_t height, uint8_t digest[32]) {
  uint64_t h[8];
  uint64_t m[16];
  blake2b256_init_dev(h);

  uint64_t ctr = 0;
  m[0] = pack_be32_pair_as_blake_word(index, height);
#pragma unroll
  for (unsigned i = 1; i < 16; ++i) m[i] = bswap64_dev(ctr++);

  uint64_t total = 0;
  for (unsigned block = 0; block < 64; ++block) {
    total += 128;
    blake2b_compress_dev(h, m, total, false);
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) m[i] = bswap64_dev(ctr++);
  }

  m[0] = bswap64_dev(CONST_MES_SIZE / sizeof(uint64_t) - 1);
#pragma unroll
  for (unsigned i = 1; i < 16; ++i) m[i] = 0;
  total += 8;
  blake2b_compress_dev(h, m, total, true);
  blake2b256_output_dev(h, digest);
}

// Fast table-row prehash. Two differences from autolykos_prehash_digest_dev above (which is
// kept unchanged as the no-table mining path and the verification reference):
// 1. Explicit 32-bit pair arithmetic — Xe2 has no native 64-bit integer ALU and the uint64_t
//    lowering leaves throughput on the table (same finding as the kawpow keccak, which uses
//    Uint2 pairs). rotr64 by 32 becomes a half swap, 24/16/63 become shift pairs.
// 2. The 8192-byte Autolykos2 pad is the 64-bit big-endian integers 0..1023, so every pad
//    message word is bswap64(item) with item < 1024: zero low half, two-shift high half.
//    Message providers feed these without materializing a 16-word block.
