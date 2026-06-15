// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>

#include "lib-internal.h"
#include "../native/consts.h"

namespace mom_autolykos2 {

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

struct AutolykosJobData {
  uint8_t message[HASH_LEN];
  uint8_t target[HASH_LEN];
};

struct AutolykosResult {
  uint32_t count;
  uint64_t nonce[MAX_AUTOLYKOS_OUTPUTS];
  uint8_t output[MAX_AUTOLYKOS_OUTPUTS][HASH_LEN];
};

inline uint32_t round_up(const uint32_t value, const uint32_t step) {
  return ((value + step - 1) / step) * step;
}

// Fast unsigned remainder by a runtime divisor (Lemire's "faster remainder"). The K_LEN=32 dataset
// reads do `idx % n_len` per nonce and n_len is a per-height runtime value. On the nvptx device pass
// (__NVPTX__) NVIDIA has no integer-divide unit, so `%` emulates as a slow multi-op sequence:
// M = floor(2^64 / n_len) + 1 is precomputed once on the host (mo_modM), then
// x % n_len == hi64( (M*x mod 2^64) * n_len ). Bit-exact for x, n_len in [1, 2^32). On Intel (which
// has a divide unit) keep the plain modulo. Gating on the per-pass __NVPTX__ (not a build macro)
// lets the combined build pick the right path in each device image.
inline uint64_t mo_modM(const uint32_t d) { return d ? (0xFFFFFFFFFFFFFFFFULL / d) + 1ULL : 0ULL; }
inline uint32_t mo_mod_u32(const uint32_t x, const uint32_t d, const uint64_t M) {
#if defined(__NVPTX__)
  return static_cast<uint32_t>(sycl::mul_hi(M * static_cast<uint64_t>(x), static_cast<uint64_t>(d)));
#else
  (void)M; return x % d;
#endif
}

inline uint64_t bswap64_dev(const uint64_t value) {
  return ((value & 0x00000000000000FFULL) << 56) |
         ((value & 0x000000000000FF00ULL) << 40) |
         ((value & 0x0000000000FF0000ULL) << 24) |
         ((value & 0x00000000FF000000ULL) << 8) |
         ((value & 0x000000FF00000000ULL) >> 8) |
         ((value & 0x0000FF0000000000ULL) >> 24) |
         ((value & 0x00FF000000000000ULL) >> 40) |
         ((value & 0xFF00000000000000ULL) >> 56);
}

inline uint64_t rotr64_dev(const uint64_t value, const unsigned shift) {
  return (value >> shift) | (value << (64U - shift));
}

inline uint32_t load32_be_dev(const uint8_t* const input) {
  return (static_cast<uint32_t>(input[0]) << 24) |
         (static_cast<uint32_t>(input[1]) << 16) |
         (static_cast<uint32_t>(input[2]) << 8) |
          static_cast<uint32_t>(input[3]);
}

inline uint64_t load64_be_dev(const uint8_t* const input) {
  return (static_cast<uint64_t>(load32_be_dev(input)) << 32) |
          static_cast<uint64_t>(load32_be_dev(input + 4));
}

inline uint64_t load64_le_dev(const uint8_t* const input) {
  return static_cast<uint64_t>(input[0]) |
         (static_cast<uint64_t>(input[1]) << 8) |
         (static_cast<uint64_t>(input[2]) << 16) |
         (static_cast<uint64_t>(input[3]) << 24) |
         (static_cast<uint64_t>(input[4]) << 32) |
         (static_cast<uint64_t>(input[5]) << 40) |
         (static_cast<uint64_t>(input[6]) << 48) |
         (static_cast<uint64_t>(input[7]) << 56);
}

inline void store32_be_dev(uint8_t* const output, const uint32_t value) {
  output[0] = static_cast<uint8_t>(value >> 24);
  output[1] = static_cast<uint8_t>(value >> 16);
  output[2] = static_cast<uint8_t>(value >> 8);
  output[3] = static_cast<uint8_t>(value);
}

inline uint8_t word_byte_be_dev(const uint32_t word, const unsigned byte_index) {
  return static_cast<uint8_t>(word >> ((3U - byte_index) * 8U));
}

inline uint64_t pack_be32_pair_as_blake_word(const uint32_t a, const uint32_t b) {
  return (static_cast<uint64_t>((a >> 24) & 0xffU) << 0) |
         (static_cast<uint64_t>((a >> 16) & 0xffU) << 8) |
         (static_cast<uint64_t>((a >> 8) & 0xffU) << 16) |
         (static_cast<uint64_t>(a & 0xffU) << 24) |
         (static_cast<uint64_t>((b >> 24) & 0xffU) << 32) |
         (static_cast<uint64_t>((b >> 16) & 0xffU) << 40) |
         (static_cast<uint64_t>((b >> 8) & 0xffU) << 48) |
         (static_cast<uint64_t>(b & 0xffU) << 56);
}

inline void blake2b_g(
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

inline void blake2b_compress_dev(uint64_t h[8], const uint64_t m[16], const uint64_t t, const bool last) {
  uint64_t v[16];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) v[i] = h[i];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) v[i + 8] = BLAKE2B_IV[i];
  v[12] ^= t;
  if (last) v[14] = ~v[14];

  // Unroll so BLAKE2B_SIGMA[r][i] becomes a compile-time constant and m[16] stays
  // in registers instead of spilling to local memory.
#pragma unroll
  for (unsigned r = 0; r < 12; ++r) {
    blake2b_g(v[0], v[4], v[ 8], v[12], m[BLAKE2B_SIGMA[r][ 0]], m[BLAKE2B_SIGMA[r][ 1]]);
    blake2b_g(v[1], v[5], v[ 9], v[13], m[BLAKE2B_SIGMA[r][ 2]], m[BLAKE2B_SIGMA[r][ 3]]);
    blake2b_g(v[2], v[6], v[10], v[14], m[BLAKE2B_SIGMA[r][ 4]], m[BLAKE2B_SIGMA[r][ 5]]);
    blake2b_g(v[3], v[7], v[11], v[15], m[BLAKE2B_SIGMA[r][ 6]], m[BLAKE2B_SIGMA[r][ 7]]);
    blake2b_g(v[0], v[5], v[10], v[15], m[BLAKE2B_SIGMA[r][ 8]], m[BLAKE2B_SIGMA[r][ 9]]);
    blake2b_g(v[1], v[6], v[11], v[12], m[BLAKE2B_SIGMA[r][10]], m[BLAKE2B_SIGMA[r][11]]);
    blake2b_g(v[2], v[7], v[ 8], v[13], m[BLAKE2B_SIGMA[r][12]], m[BLAKE2B_SIGMA[r][13]]);
    blake2b_g(v[3], v[4], v[ 9], v[14], m[BLAKE2B_SIGMA[r][14]], m[BLAKE2B_SIGMA[r][15]]);
  }

#pragma unroll
  for (unsigned i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
}

inline void blake2b256_init_dev(uint64_t h[8]) {
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) h[i] = BLAKE2B_IV[i];
  h[0] ^= 0x01010020ULL;
}

inline void blake2b256_output_dev(const uint64_t h[8], uint8_t out[32]) {
#pragma unroll
  for (unsigned i = 0; i < 32; ++i) out[i] = static_cast<uint8_t>(h[i >> 3] >> ((i & 7U) * 8U));
}

inline void blake2b256_oneblock_dev(const uint8_t* const input, const uint32_t len, uint8_t out[32]) {
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

inline void blake2b256_71_dev(const uint8_t input[71], uint8_t out[32]) {
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

inline void blake2b256_message_nonce_dev(const uint8_t message[32], const uint64_t nonce, uint8_t out[32]) {
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

inline void blake2b256_sum_dev(const uint32_t sum[8], uint8_t out[32]) {
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

inline void autolykos_prehash_digest_dev(const uint32_t index, const uint32_t height, uint8_t digest[32]) {
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
struct B2bWord {
  uint32_t lo, hi;
};

inline uint32_t pad_message_hi_dev(const uint32_t item) {
  return ((item & 0xFFU) << 24) | ((item >> 8) << 16);
}

struct PrehashBlock0Message { // word 0 carries index|height, words 1..15 are pad items 0..14
  uint32_t index_be, height_be;
  inline B2bWord word(const unsigned i) const {
    return i == 0 ? B2bWord{index_be, height_be} : B2bWord{0, pad_message_hi_dev(i - 1)};
  }
};

struct PrehashPadMessage { // blocks 1..63: word i is pad item 16*block - 1 + i
  uint32_t base;
  inline B2bWord word(const unsigned i) const {
    return B2bWord{0, pad_message_hi_dev(base + i)};
  }
};

struct PrehashFinalMessage { // word 0 is pad item 1023, the rest is padding zeros
  inline B2bWord word(const unsigned i) const {
    return i == 0 ? B2bWord{0, pad_message_hi_dev(CONST_MES_SIZE / sizeof(uint64_t) - 1)}
                  : B2bWord{0, 0};
  }
};

inline void blake2b_g_pair(
  B2bWord& a, B2bWord& b, B2bWord& c, B2bWord& d, const B2bWord x, const B2bWord y
) {
  uint32_t lo = a.lo + b.lo;
  uint32_t carry = lo < b.lo ? 1U : 0U;
  a.lo = lo + x.lo;
  carry += a.lo < x.lo ? 1U : 0U;
  a.hi = a.hi + b.hi + x.hi + carry;
  { // d = rotr64(d ^ a, 32): swap halves
    const uint32_t t = d.lo ^ a.lo;
    d.lo = d.hi ^ a.hi;
    d.hi = t;
  }
  lo = c.lo + d.lo;
  c.hi = c.hi + d.hi + (lo < d.lo ? 1U : 0U);
  c.lo = lo;
  { // b = rotr64(b ^ c, 24)
    const uint32_t bl = b.lo ^ c.lo, bh = b.hi ^ c.hi;
    b.lo = (bl >> 24) | (bh << 8);
    b.hi = (bh >> 24) | (bl << 8);
  }
  lo = a.lo + b.lo;
  carry = lo < b.lo ? 1U : 0U;
  a.lo = lo + y.lo;
  carry += a.lo < y.lo ? 1U : 0U;
  a.hi = a.hi + b.hi + y.hi + carry;
  { // d = rotr64(d ^ a, 16)
    const uint32_t dl = d.lo ^ a.lo, dh = d.hi ^ a.hi;
    d.lo = (dl >> 16) | (dh << 16);
    d.hi = (dh >> 16) | (dl << 16);
  }
  lo = c.lo + d.lo;
  c.hi = c.hi + d.hi + (lo < d.lo ? 1U : 0U);
  c.lo = lo;
  { // b = rotr64(b ^ c, 63) == rotl64(b ^ c, 1)
    const uint32_t bl = b.lo ^ c.lo, bh = b.hi ^ c.hi;
    b.lo = (bl << 1) | (bh >> 31);
    b.hi = (bh << 1) | (bl >> 31);
  }
}

template <typename Message>
inline void blake2b_compress_pair_dev(B2bWord h[8], const Message& msg, const uint32_t t, const bool last) {
  B2bWord v[16];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) v[i] = h[i];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    v[i + 8] = B2bWord{static_cast<uint32_t>(BLAKE2B_IV[i]), static_cast<uint32_t>(BLAKE2B_IV[i] >> 32)};
  }
  v[12].lo ^= t; // the prehash message is 8200 bytes, the counter never reaches the high half
  if (last) {
    v[14].lo = ~v[14].lo;
    v[14].hi = ~v[14].hi;
  }

  for (unsigned r = 0; r < 12; ++r) {
    blake2b_g_pair(v[0], v[4], v[ 8], v[12], msg.word(BLAKE2B_SIGMA[r][ 0]), msg.word(BLAKE2B_SIGMA[r][ 1]));
    blake2b_g_pair(v[1], v[5], v[ 9], v[13], msg.word(BLAKE2B_SIGMA[r][ 2]), msg.word(BLAKE2B_SIGMA[r][ 3]));
    blake2b_g_pair(v[2], v[6], v[10], v[14], msg.word(BLAKE2B_SIGMA[r][ 4]), msg.word(BLAKE2B_SIGMA[r][ 5]));
    blake2b_g_pair(v[3], v[7], v[11], v[15], msg.word(BLAKE2B_SIGMA[r][ 6]), msg.word(BLAKE2B_SIGMA[r][ 7]));
    blake2b_g_pair(v[0], v[5], v[10], v[15], msg.word(BLAKE2B_SIGMA[r][ 8]), msg.word(BLAKE2B_SIGMA[r][ 9]));
    blake2b_g_pair(v[1], v[6], v[11], v[12], msg.word(BLAKE2B_SIGMA[r][10]), msg.word(BLAKE2B_SIGMA[r][11]));
    blake2b_g_pair(v[2], v[7], v[ 8], v[13], msg.word(BLAKE2B_SIGMA[r][12]), msg.word(BLAKE2B_SIGMA[r][13]));
    blake2b_g_pair(v[3], v[4], v[ 9], v[14], msg.word(BLAKE2B_SIGMA[r][14]), msg.word(BLAKE2B_SIGMA[r][15]));
  }

#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    h[i].lo ^= v[i].lo ^ v[i + 8].lo;
    h[i].hi ^= v[i].hi ^ v[i + 8].hi;
  }
}

inline void autolykos_prehash_digest_fast_dev(const uint32_t index, const uint32_t height, uint8_t digest[32]) {
  B2bWord h[8];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    h[i] = B2bWord{static_cast<uint32_t>(BLAKE2B_IV[i]), static_cast<uint32_t>(BLAKE2B_IV[i] >> 32)};
  }
  h[0].lo ^= 0x01010020U;

  const uint32_t index_be = (index >> 24) | ((index >> 8) & 0xFF00U) |
                            ((index << 8) & 0xFF0000U) | (index << 24);
  const uint32_t height_be = (height >> 24) | ((height >> 8) & 0xFF00U) |
                             ((height << 8) & 0xFF0000U) | (height << 24);
  blake2b_compress_pair_dev(h, PrehashBlock0Message{index_be, height_be}, 128, false);
  for (uint32_t block = 1; block < 64; ++block) {
    blake2b_compress_pair_dev(h, PrehashPadMessage{16U * block - 1U}, (block + 1U) * 128U, false);
  }
  blake2b_compress_pair_dev(h, PrehashFinalMessage{}, CONST_MES_SIZE + 8U, true);

#pragma unroll
  for (unsigned i = 0; i < 32; ++i) {
    const B2bWord word = h[i >> 3];
    const unsigned byte = i & 7U;
    digest[i] = static_cast<uint8_t>((byte < 4 ? word.lo >> (byte * 8U) : word.hi >> ((byte - 4U) * 8U)));
  }
}

// A table row is the digest packed big-endian into 8 words, with its top byte (digest[0])
// forced to zero — load32_be of word 0 reads digest[0..3], so just mask that byte off.
inline void digest_to_limbs_dev(const uint8_t digest[32], uint32_t limbs[8]) {
  limbs[0] = load32_be_dev(digest) & 0x00FFFFFFU;
#pragma unroll
  for (unsigned i = 1; i < 8; ++i) limbs[i] = load32_be_dev(digest + i * 4);
}

inline void autolykos_table_row_store_dev(uint32_t* const table, const uint32_t gid, const uint32_t height) {
  uint8_t digest[32];
  autolykos_prehash_digest_fast_dev(gid, height, digest);
  digest_to_limbs_dev(digest, table + static_cast<uint64_t>(gid) * (TABLE_ENTRY_BYTES / sizeof(uint32_t)));
}

inline void table_item_limbs_dev(
  const uint32_t* const table, const uint32_t index, const uint32_t height, uint32_t limbs[8]
) {
  if (table) {
    const uint32_t* const item = table + static_cast<uint64_t>(index) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) limbs[i] = item[i];
    return;
  }

  uint8_t digest[32];
  autolykos_prehash_digest_dev(index, height, digest);
  digest_to_limbs_dev(digest, limbs);
}

// 256-bit big-endian add: sum[0..7] += add[0..7], propagating carry from least- (i=7) to
// most-significant limb. The Autolykos2 sum wraps mod 2^256, so the final carry is dropped.
inline void add_limbs_dev(uint32_t sum[8], const uint32_t* const add) {
  uint64_t carry = 0;
  for (int i = 7; i >= 0; --i) {
    const uint64_t v = static_cast<uint64_t>(sum[i]) + add[i] + carry;
    sum[i] = static_cast<uint32_t>(v);
    carry = v >> 32;
  }
}

inline bool meets_target_dev(const uint8_t output[32], const uint8_t target[32]) {
  for (unsigned i = 0; i < 32; ++i) {
    if (output[i] == target[i]) continue;
    return output[i] < target[i];
  }
  return true;
}

inline void store_autolykos_result(
  AutolykosResult* const result, const uint64_t nonce, const uint8_t output[32]
) {
  using atomic_ref = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space
  >;
  const uint32_t index = atomic_ref(result->count).fetch_add(1);
  if (index >= MAX_AUTOLYKOS_OUTPUTS) return;

  result->nonce[index] = nonce;
#pragma unroll
  for (unsigned i = 0; i < 32; ++i) result->output[index][i] = output[i];
}

// Autolykos2 per-nonce prehash: blake2b(message||nonce) selects table row h3, then
// index_hash = blake2b(row[1..31] || message || nonce) seeds the K_LEN table reads.
// Row bytes come from the prebuilt table when present, else from the on-the-fly digest.
inline void autolykos_index_hash_dev(
  const uint8_t message[32], const uint64_t nonce, const uint32_t* const table,
  const uint32_t height, const uint32_t n_len, uint8_t index_hash[32]
) {
  uint8_t nonce_be[8];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) nonce_be[i] = static_cast<uint8_t>(nonce >> ((7U - i) * 8U));

  uint8_t h1_input[40];
#pragma unroll
  for (unsigned i = 0; i < 32; ++i) h1_input[i] = message[i];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) h1_input[32 + i] = nonce_be[i];

  uint8_t h1[32];
  blake2b256_oneblock_dev(h1_input, sizeof(h1_input), h1);
  const uint32_t h3 = static_cast<uint32_t>(load64_be_dev(h1 + 24) % n_len);

  uint8_t seed[71]; // row[1..31] (31) || message (32) || nonce (8)
  if (table) {
    const uint32_t* const h3_item = table + static_cast<uint64_t>(h3) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
#pragma unroll
    for (unsigned i = 0; i < 31; ++i) {
      const unsigned pos = i + 1U;
      seed[i] = word_byte_be_dev(h3_item[pos >> 2], pos & 3U);
    }
  } else {
    uint8_t h3_digest[32];
    autolykos_prehash_digest_dev(h3, height, h3_digest);
#pragma unroll
    for (unsigned i = 0; i < 31; ++i) seed[i] = h3_digest[i + 1];
  }
#pragma unroll
  for (unsigned i = 0; i < 32; ++i) seed[31 + i] = message[i];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) seed[63 + i] = nonce_be[i];

  blake2b256_oneblock_dev(seed, sizeof(seed), index_hash);
}

inline uint32_t calc_n(const uint32_t height) {
  if (height < INCREASE_START) return INIT_N_LEN;
  if (height >= INCREASE_END) return MAX_N_LEN;

  uint32_t n = INIT_N_LEN;
  const uint32_t iters = (height - INCREASE_START) / INCREASE_PERIOD + 1;
  for (uint32_t i = 0; i < iters; ++i) n = n / 100U * 105U;
  return n;
}

static uint64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now().time_since_epoch()
  ).count();
}

static void format_duration_ms(char* out, size_t out_size, uint64_t ms) {
  if (ms < 1000) {
    std::snprintf(out, out_size, "%" PRIu64 "ms", ms);
  } else if (ms < 60000) {
    std::snprintf(out, out_size, "%.2f s", static_cast<double>(ms) / 1000.0);
  } else {
    std::snprintf(out, out_size, "%.2f min", static_cast<double>(ms) / 60000.0);
  }
}

// Parse env var `name` as a base-10 u32. Writes *out and returns true only when the
// variable is set to a clean, fully-consumed, in-range integer; otherwise returns false
// (unset, empty, trailing junk, overflow) so the caller can keep its fallback.
static bool env_u32(const char* name, uint32_t& out) {
  const char* const value = std::getenv(name);
  if (!value || !*value) return false;

  char* end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno || end == value || *end || parsed > std::numeric_limits<uint32_t>::max()) return false;
  out = static_cast<uint32_t>(parsed);
  return true;
}

struct AutolykosState {
  sycl::device device;
  sycl::queue queue;
  std::unique_ptr<MOM_BUNDLE_T> bundle;
  uint32_t* table = nullptr;
  uint32_t* bhashes = nullptr;
  AutolykosResult* result = nullptr;
  uint32_t* table_mismatches = nullptr;
  uint32_t table_height = UINT32_MAX;
  uint32_t table_n = 0;
  uint32_t table_cap_n = 0;
  uint32_t bhashes_cap = 0;
  unsigned workgroup;
  unsigned prehash_workgroup;
  std::mutex mutex;

  // Background prebuild of the next height's table was tried three ways and measured
  // slower than the synchronous rebuild it replaces on Xe2/GuC (B580, 2026-06): a second
  // queue is timesliced against mining whether paced, flat-out or priority_low (mining
  // drops to 50%/15%/starved-builder respectively), and fusing builder workgroups into the
  // mix kernel halves its occupancy through shared register allocation (~2x the slices'
  // GPU cost). The prehash's GPU-seconds cannot co-execute with mining on this hardware,
  // so the cheapest place to pay them is one concentrated rebuild per height.

  explicit AutolykosState(const std::string& dev_str)
    : device(get_dev(dev_str)),
      queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      workgroup(autolykos_workgroup(device)),
      prehash_workgroup(autolykos_prehash_workgroup(device))
  {
    if (!device.has(sycl::aspect::usm_shared_allocations)) {
      throw std::string("autolykos2 SYCL device does not support shared allocations");
    }

    set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", autolykos_compile_options(device));
    bundle = std::make_unique<MOM_BUNDLE_T>(
      MOM_GET_EXEC_BUNDLE(queue.get_context())
    );
  }

  ~AutolykosState() { release(); }

  static unsigned autolykos_workgroup(const sycl::device& dev) {
    // NVIDIA: 128 gives the best occupancy for the unrolled per-thread lookup (measured ~68 vs
    // ~64 MH/s at 64 on an L4); the cooperative local==64 path is intentionally not used there.
    const unsigned fallback = sycl_default_workgroup(dev, {32, 64, 128, 256}, mom_is_cuda(dev) ? 128 : 64);
    return env_workgroup("MOM_AUTOLYKOS2_WORKGROUP", fallback);
  }

  static unsigned autolykos_prehash_workgroup(const sycl::device& dev) {
    const unsigned fallback = sycl_default_workgroup(dev, {32, 64, 128, 256}, 64);
    return env_workgroup("MOM_AUTOLYKOS2_PREHASH_WORKGROUP", fallback);
  }

  // Accept an env override only if it is one of the supported workgroup sizes.
  static unsigned env_workgroup(const char* name, const unsigned fallback) {
    uint32_t parsed = 0;
    if (env_u32(name, parsed) && (parsed == 32 || parsed == 64 || parsed == 128 || parsed == 256)) {
      return parsed;
    }
    return fallback;
  }

  static const char* autolykos_compile_options(const sycl::device& dev) {
    const char* const value = std::getenv("MOM_AUTOLYKOS2_COMPILE_OPTIONS");
    if (value) return value;
    // SYCL_PROGRAM_COMPILE_OPTIONS is process-global; pearl's ESIMD (VC-backend) image rejects "-O3",
    // and it's a measured no-op for this kernel anyway (36.44 vs 36.41 MH/s on B580). Override via
    // MOM_AUTOLYKOS2_COMPILE_OPTIONS if needed.
    (void)dev;
    return "";
  }

  // Unlike the kawpow DAG chunking this defaults on: at live N the monolithic build is one
  // multi-second GPU job, the long-running-job shape that tripped xe/GuC scheduler cleanup
  // crashes in June 2026. ~3 s chunks cost nothing on the in-order queue. 0 = single kernel.
  static uint32_t autolykos_table_chunk_rows() {
    constexpr uint32_t fallback = 32U * 1024U * 1024U;
    uint32_t parsed = 0;
    return env_u32("MOM_AUTOLYKOS2_TABLE_CHUNK", parsed) ? parsed : fallback;
  }

  // 0 disables the post-build row check, 1 (default) cross-checks a strided sample of rows
  // against autolykos_prehash_digest_dev, 2 recomputes every row (slow, for debugging).
  static unsigned autolykos_table_verify_mode() {
    uint32_t parsed = 0;
    return env_u32("MOM_AUTOLYKOS2_VERIFY_TABLE", parsed) && parsed <= 2 ? parsed : 1;
  }

  void release() {
    queue.wait_and_throw();
    free_ptr(table);
    free_ptr(bhashes);
    free_ptr(result);
    free_ptr(table_mismatches);
    table_height = UINT32_MAX;
    table_n = 0;
    table_cap_n = 0;
    bhashes_cap = 0;
  }

  void free_ptr(auto*& ptr) {
    if (ptr) sycl::free(ptr, queue);
    ptr = nullptr;
  }

  void ensure_result() {
    if (result) return;
    result = sycl::malloc_shared<AutolykosResult>(1, queue);
    if (!result) throw std::string("Can't allocate autolykos2 SYCL result buffer");
  }

  void ensure_bhashes(const uint32_t intensity) {
    if (bhashes && bhashes_cap >= intensity) return;
    queue.wait_and_throw();
    free_ptr(bhashes);
    bhashes_cap = intensity;
    bhashes = sycl::malloc_device<uint32_t>(static_cast<uint64_t>(intensity) * 8U, queue);
    if (!bhashes) throw std::string("Can't allocate autolykos2 intermediate buffer");
  }

  bool should_use_table(const bool is_test) const {
    if (is_test || device.is_cpu()) return false;
    const char* const value = std::getenv("MOM_AUTOLYKOS2_TABLE");
    return !(value && value[0] == '0');
  }

  // Recompute a sample of the freshly built table with the reference digest and throw on
  // any difference: the fast build kernel folds the constant pad, and a silently wrong
  // table would invalidate every share until the next block.
  void verify_table(const uint32_t height, const uint32_t n_len) {
    const unsigned mode = autolykos_table_verify_mode();
    if (!mode) return;

    if (!table_mismatches) {
      table_mismatches = sycl::malloc_shared<uint32_t>(1, queue);
      if (!table_mismatches) throw std::string("Can't allocate autolykos2 verify counter");
    }
    *table_mismatches = 0;

    const uint32_t stride = mode >= 2 ? 1 : 65521; // prime; ~3.3k of 216M rows in default mode
    const uint32_t count = (n_len + stride - 1) / stride;
    const uint32_t* const d_table = table;
    uint32_t* const d_mismatches = table_mismatches;
    const uint32_t local = prehash_workgroup;

    sycl_wait_and_throw(queue.submit([&](sycl::handler& h) {
      MOM_USE_BUNDLE(h, *bundle);
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(round_up(count, local)), sycl::range<1>(local)),
        [=](sycl::nd_item<1> item) {
          const uint32_t gid = item.get_global_id(0);
          if (gid >= count) return;
          const uint32_t row = gid * stride;
          if (row >= n_len) return;

          uint8_t digest[32];
          autolykos_prehash_digest_dev(row, height, digest);
          uint32_t expected[8];
          digest_to_limbs_dev(digest, expected);

          const uint32_t* const stored = d_table + static_cast<uint64_t>(row) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
          bool differs = false;
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) differs |= stored[i] != expected[i];
          if (differs) {
            sycl::atomic_ref<
              uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
              sycl::access::address_space::global_space
            >(*d_mismatches).fetch_add(1);
          }
        }
      );
    }), device);

    if (*table_mismatches) {
      throw std::string("Autolykos2 table verification failed: ") +
            std::to_string(*table_mismatches) + " of " + std::to_string(count) + " sampled rows differ";
    }
  }

  void ensure_table(const uint32_t height, const uint32_t n_len, const bool is_test, const bool should_log) {
    if (!should_use_table(is_test)) {
      queue.wait_and_throw();
      free_ptr(table);
      table_height = UINT32_MAX;
      table_n = 0;
      table_cap_n = 0;
      return;
    }

    if (table && table_height == height && table_n == n_len) return;
    if (!device.has(sycl::aspect::usm_device_allocations)) {
      throw std::string("autolykos2 SYCL GPU device does not support device allocations");
    }

    const uint64_t start_ms = now_ms();
    table_height = UINT32_MAX;
    if (!table || table_cap_n != n_len) {
      queue.wait_and_throw();
      free_ptr(table);
      table_cap_n = 0;
      table = sycl::malloc_device<uint32_t>(
        static_cast<uint64_t>(n_len) * (TABLE_ENTRY_BYTES / sizeof(uint32_t)), queue);
      if (!table) throw std::string("Can't allocate autolykos2 table");
      table_cap_n = n_len;
    }

    uint32_t* const d_table = table;
    sycl::queue& q = queue;
    auto& kb = *bundle;
    const uint32_t local = prehash_workgroup;

    const uint32_t chunk_rows = autolykos_table_chunk_rows();
    sycl::event build_event;
    for (uint32_t start_row = 0; start_row < n_len;) {
      const uint32_t current_rows = chunk_rows ? std::min(chunk_rows, n_len - start_row) : n_len;
      const uint32_t chunk_start = start_row;
      build_event = q.submit([&](sycl::handler& h) {
        MOM_USE_BUNDLE(h, kb);
        h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(round_up(current_rows, local)), sycl::range<1>(local)),
          [=](sycl::nd_item<1> item) {
            const uint32_t local_row = item.get_global_id(0);
            if (local_row >= current_rows) return;
            autolykos_table_row_store_dev(d_table, chunk_start + local_row, height);
          }
        );
      });
      if (!chunk_rows) break;
      start_row += current_rows;
    }
    sycl_wait_and_throw(build_event, device);
    verify_table(height, n_len);

    table_height = height;
    table_n = n_len;
    if (should_log || std::getenv("MOM_LOOP_STATS")) {
      char elapsed[32];
      format_duration_ms(elapsed, sizeof(elapsed), now_ms() - start_ms);
      std::fprintf(stderr, "Autolykos2 table for height %u N %u calculated (%s)\n", height, n_len, elapsed);
    }
  }
};

static AutolykosState& autolykos_state(const std::string& dev_str) {
  static std::mutex states_mutex;
  static std::map<std::string, std::unique_ptr<AutolykosState>> states;

  std::lock_guard<std::mutex> lock(states_mutex);
  auto& state = states[dev_str];
  if (!state) state = std::make_unique<AutolykosState>(dev_str);
  return *state;
}

// Return the last winning share found by a mining kernel (0 = none). The kernels may stash up
// to MAX_AUTOLYKOS_OUTPUTS hits but the API reports a single nonce/hash, so pick the last slot.
static int take_autolykos_result(const AutolykosResult* result, uint8_t* output, uint64_t* pnonce) {
  const uint32_t count = result->count;
  if (count == 0) return 0;
  const uint32_t index = std::min(count, MAX_AUTOLYKOS_OUTPUTS) - 1;
  *pnonce = result->nonce[index];
  std::memcpy(output, result->output[index], HASH_LEN);
  return 1;
}

} // namespace mom_autolykos2

using namespace mom_autolykos2;

int autolykos2(
  const unsigned, const uint32_t height, const uint8_t* const input, const unsigned input_size,
  uint8_t* const output, uint64_t* const pnonce, const uint8_t* const target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < 40) throw std::string("Bad autolykos2 input length");

  AutolykosState& state = autolykos_state(dev_str);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  state.ensure_result();
  const uint32_t n_len = calc_n(height);
  const uint64_t n_len_M = mo_modM(n_len);   // Lemire magic for the hot `idx % n_len` (see mo_mod_u32)
  state.ensure_table(height, n_len, is_test, !is_benchmark);

  AutolykosJobData job{};
  std::memcpy(job.message, input, HASH_LEN);
  std::memcpy(job.target, target, HASH_LEN);

  uint64_t start_nonce = 0;
  std::memcpy(&start_nonce, input + HASH_LEN, sizeof(start_nonce));
  const uint32_t effective_intensity = is_test ? 1 : intensity;
  const uint32_t global_size = round_up(std::max(effective_intensity, state.workgroup), state.workgroup);
  std::memset(state.result, 0, sizeof(AutolykosResult));

  sycl::queue& q = state.queue;
  auto& kb = *state.bundle;
  const uint32_t* const __restrict__ d_table = state.table;
  AutolykosResult* const d_result = state.result;
  const uint32_t local_size = state.workgroup;

  if (d_table && !is_test) {
    state.ensure_bhashes(effective_intensity);
    uint32_t* const d_bhashes = state.bhashes;

    q.submit([&](sycl::handler& h) {
      MOM_USE_BUNDLE(h, kb);
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const uint32_t gid = item.get_global_id(0);
          if (gid >= effective_intensity) return;

          uint8_t index_hash[32];
          autolykos_index_hash_dev(job.message, start_nonce + gid, d_table, height, n_len, index_hash);

          uint32_t* const out = d_bhashes + static_cast<uint64_t>(gid) * 8U;
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) out[i] = load32_be_dev(index_hash + i * 4);
        }
      );
    });

    sycl_wait_and_throw(q.submit([&](sycl::handler& h) {
      MOM_USE_BUNDLE(h, kb);
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const uint32_t gid = item.get_global_id(0);
          if (gid >= effective_intensity) return;

          const uint32_t* const hash_words = d_bhashes + static_cast<uint64_t>(gid) * 8U;
          uint32_t words[9];
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) words[i] = hash_words[i];
          words[8] = words[0];

          uint32_t sum[8] = {};
          // Unroll the K_LEN table-row reads so the independent random loads issue
          // in parallel (memory-level parallelism) and hide DRAM latency instead
          // of stalling one-at-a-time (the read loop is the autolykos2 bottleneck).
#pragma unroll
          for (unsigned k = 0; k < K_LEN; ++k) {
            const unsigned word_index = k >> 2;
            uint32_t idx;
            switch (k & 3U) {
              case 0: idx = words[word_index]; break;
              case 1: idx = (words[word_index] << 8) | (words[word_index + 1] >> 24); break;
              case 2: idx = (words[word_index] << 16) | (words[word_index + 1] >> 16); break;
              default: idx = (words[word_index] << 24) | (words[word_index + 1] >> 8); break;
            }
            idx = mo_mod_u32(idx, n_len, n_len_M);
            add_limbs_dev(sum, d_table + static_cast<uint64_t>(idx) * (TABLE_ENTRY_BYTES / sizeof(uint32_t)));
          }

          uint8_t final_input[32];
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) store32_be_dev(final_input + i * 4, sum[i]);

          uint8_t final_hash[32];
          blake2b256_oneblock_dev(final_input, sizeof(final_input), final_hash);
          if (meets_target_dev(final_hash, job.target)) {
            store_autolykos_result(d_result, start_nonce + gid, final_hash);
          }
        }
      );
    }), state.device);

    return take_autolykos_result(state.result, output, pnonce);
  }

  sycl_wait_and_throw(q.submit([&](sycl::handler& h) {
    MOM_USE_BUNDLE(h, kb);
    sycl::local_accessor<uint32_t, 1> shared_index(sycl::range<1>(64), h);
    sycl::local_accessor<uint32_t, 1> shared_data(sycl::range<1>(512), h);
    h.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
      [=](sycl::nd_item<1> item) {
        const uint32_t gid = item.get_global_id(0);
        const uint32_t local_id = item.get_local_id(0);
        const bool active = gid < effective_intensity;

        const uint64_t nonce = start_nonce + gid;
        uint32_t indices[K_LEN];
#pragma unroll
        for (unsigned i = 0; i < K_LEN; ++i) indices[i] = 0;

        if (active) {
          uint8_t index_hash[32];
          autolykos_index_hash_dev(job.message, nonce, d_table, height, n_len, index_hash);

#pragma unroll
          for (unsigned k = 0; k < K_LEN; ++k) {
            indices[k] = mo_mod_u32(
              ((static_cast<uint32_t>(index_hash[(k + 0) & 31U]) << 24) |
               (static_cast<uint32_t>(index_hash[(k + 1) & 31U]) << 16) |
               (static_cast<uint32_t>(index_hash[(k + 2) & 31U]) << 8) |
                static_cast<uint32_t>(index_hash[(k + 3) & 31U])), n_len, n_len_M);
          }
        }

        uint32_t sum[8] = {};

        if (d_table && local_size == 64) {
          const uint32_t thread_id = local_id & 7U;
          const uint32_t hash_id = local_id >> 3;
          for (unsigned k = 0; k < K_LEN; ++k) {
            shared_index[local_id] = indices[k];
            item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
            for (unsigned group = 0; group < 8; ++group) {
              const uint32_t source_lane = hash_id + group * 8U;
              const uint32_t* const table_item =
                d_table + static_cast<uint64_t>(shared_index[source_lane]) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
              shared_data[(group * 64U) + (hash_id * 8U) + thread_id] = table_item[thread_id];
            }
            item.barrier(sycl::access::fence_space::local_space);

            uint64_t carry = 0;
            for (int i = 7; i >= 0; --i) {
              const uint64_t v = static_cast<uint64_t>(sum[i]) + shared_data[local_id * 8U + i] + carry;
              sum[i] = static_cast<uint32_t>(v);
              carry = v >> 32;
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
        } else {
          if (!active) return;
          for (unsigned k = 0; k < K_LEN; ++k) {
            uint32_t limbs[8];
            table_item_limbs_dev(d_table, indices[k], height, limbs);
            add_limbs_dev(sum, limbs);
          }
        }

        if (!active) return;

        uint8_t final_input[32];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) store32_be_dev(final_input + i * 4, sum[i]);

        uint8_t final_hash[32];
        blake2b256_oneblock_dev(final_input, sizeof(final_input), final_hash);
        if (is_test || meets_target_dev(final_hash, job.target)) {
          store_autolykos_result(d_result, nonce, final_hash);
        }
      }
    );
  }), state.device);

  return take_autolykos_result(state.result, output, pnonce);
}
