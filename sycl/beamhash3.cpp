// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
//
// BeamHash III (Beam) GPU solver -- Wagner bucket-collision (Equihash-family, k=5).
//
// NOT stock Equihash: reuses the equihash125_4.cpp Wagner infra (bucket-sort, slot-shrink, Cantor tree
// recovery, 25-bit CompressArray pack), but the row generation and the per-round mixing are different:
//   gen      : 2^25 leaves; each leaf's 448 work bits = 7 u64 = SipHash-2-4(key=IndividualWork(4 u64),
//              msg=(index<<3)+i) for i=0..6.  IndividualWork = BLAKE2b(prework||nonce||extranonce,
//              personal "Beam-PoW"+le32(448)+le32(5)).
//   applyMix : a non-linear mix BEFORE every round -- serialize (workBits | indexTree-pad) to 512 bits,
//              fold the 8 u64 words with rotl by (29*(i+1))&63 and modular add, rotl<<24, write into the
//              low 64 work bits.  Couples the index tree back into the work bits each round (the ASIC
//              resistance mechanism). NO Equihash analog.
//   round    : collide low 24 work bits XOR=0 (rounds 1-4); round 5 collides low 48 bits. After collision
//              merge = (a.workBits ^ b.workBits) >> 24, masked to remLen.
//   recover  : walk the per-level tree -> 32 leaf indices -> CompressArray(25) -> 100-byte minimal, then
//              the 104-byte solution (low 100 bytes minimal + top 4 bytes extranonce).
//
// Bit-exactly mirrors the BeamHash III reference algorithm. The is_test path validates gen+mix on-device.
//
// Mining runs the complete BeamHash III solve path. The default is_test path keeps the cheap gen+mix
// oracle validation; set MOM_BEAMHASH3_SOLVE for the M4 keystone full-solve vector.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "lib-internal.h"
#include "../native/consts.h"

namespace mom_beamhash3 {

// ---- BeamHash III parameters (beamHashIII.h) ----
constexpr unsigned WORK_BIT_SIZE      = 448;
constexpr unsigned COLLISION_BIT_SIZE = 24;
constexpr unsigned NUM_ROUNDS         = 5;             // Wagner k=5 -> 2^5 = 32 leaves
constexpr unsigned NUM_LEAVES         = 1u << NUM_ROUNDS; // 32
constexpr unsigned INDEX_BITS         = COLLISION_BIT_SIZE + 1; // 25-bit indices (index space 2^25)
constexpr uint32_t INDEX_SPACE_BITS   = 25;
constexpr uint64_t NUM_ENTRIES        = 1ull << INDEX_SPACE_BITS;  // 2^25 leaves

constexpr unsigned WORK_WORDS = 7;   // 448 bits = 7 u64 words (w[0] = least-significant)

// blob layout (host-side, passed in `input`): prework(32) || nonce(8) || extranonce(4) = 44 bytes.
constexpr unsigned PREWORK_BYTES    = 32;
constexpr unsigned NONCE_BYTES      = 8;
constexpr unsigned EXTRANONCE_BYTES = 4;
constexpr unsigned BLOB_BYTES       = PREWORK_BYTES + NONCE_BYTES + EXTRANONCE_BYTES; // 44

// M1 gen-validation dump: the first BEAMHASH3_TEST_ROWS leaves' raw 448-bit rows, 7 LE u64 words each.
constexpr unsigned BEAMHASH3_TEST_ROWS = 64;
constexpr unsigned ROW_BYTES           = WORK_WORDS * 8; // 56
static_assert(BEAMHASH3_TEST_ROWS * ROW_BYTES <= SMALL_BLOB_SOL_LEN, "M1 dump must fit the small-blob buffer");

// ===========================================================================================
// BLAKE2b-512 with personalization (host-side IndividualWork only). Self-contained; matches the JS
// oracle's blake2bFull exactly (out_len=32, personal "Beam-PoW"+le32(448)+le32(5)).
// ===========================================================================================
static constexpr uint64_t B2B_IV[8] = {
  0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
  0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull,
};
static constexpr uint8_t B2B_SIGMA[12][16] = {
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
  {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
  { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
  { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
  { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9},
  {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
  {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
  { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5},
  {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0},
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
};
inline uint64_t b2b_rotr64(uint64_t x, unsigned n) { return (x >> n) | (x << (64 - n)); }
inline uint64_t b2b_load64(const uint8_t* p) {
  return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
         ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) | ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}
inline void b2b_g(uint64_t* v, unsigned a, unsigned b, unsigned c, unsigned d, uint64_t x, uint64_t y) {
  v[a] = v[a] + v[b] + x; v[d] = b2b_rotr64(v[d] ^ v[a], 32);
  v[c] = v[c] + v[d];     v[b] = b2b_rotr64(v[b] ^ v[c], 24);
  v[a] = v[a] + v[b] + y; v[d] = b2b_rotr64(v[d] ^ v[a], 16);
  v[c] = v[c] + v[d];     v[b] = b2b_rotr64(v[b] ^ v[c], 63);
}
inline void b2b_compress(uint64_t h[8], const uint64_t m[16], uint64_t t0, uint64_t t1, bool last) {
  uint64_t v[16];
  for (unsigned i = 0; i < 8; ++i) { v[i] = h[i]; v[i + 8] = B2B_IV[i]; }
  v[12] ^= t0; v[13] ^= t1;
  if (last) v[14] ^= 0xffffffffffffffffull;
  for (unsigned r = 0; r < 12; ++r) {
    const uint8_t* s = B2B_SIGMA[r];
    b2b_g(v, 0, 4, 8,  12, m[s[0]],  m[s[1]]);
    b2b_g(v, 1, 5, 9,  13, m[s[2]],  m[s[3]]);
    b2b_g(v, 2, 6, 10, 14, m[s[4]],  m[s[5]]);
    b2b_g(v, 3, 7, 11, 15, m[s[6]],  m[s[7]]);
    b2b_g(v, 0, 5, 10, 15, m[s[8]],  m[s[9]]);
    b2b_g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
    b2b_g(v, 2, 7, 8,  13, m[s[12]], m[s[13]]);
    b2b_g(v, 3, 4, 9,  14, m[s[14]], m[s[15]]);
  }
  for (unsigned i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
}

// IndividualWork = BLAKE2b32( prework(32) || nonce(8) || extranonce(4) ) with the Beam-PoW personal.
// Returns the 32-byte digest as 4 little-endian u64 (the SipHash key). Input is exactly 44 bytes (one
// block, 44 pending), so a single final compress with t0=44.
static void individual_work(const uint8_t blob[BLOB_BYTES], uint64_t prePow[4]) {
  uint8_t param[64] = {0};
  param[0] = 32;            // out_len
  param[2] = 1;             // fanout
  param[3] = 1;             // depth
  const uint8_t personal[16] = {
    'B','e','a','m','-','P','o','W',
    (uint8_t)(WORK_BIT_SIZE & 0xFF), (uint8_t)((WORK_BIT_SIZE >> 8) & 0xFF),
    (uint8_t)((WORK_BIT_SIZE >> 16) & 0xFF), (uint8_t)((WORK_BIT_SIZE >> 24) & 0xFF),
    (uint8_t)(NUM_ROUNDS & 0xFF), (uint8_t)((NUM_ROUNDS >> 8) & 0xFF),
    (uint8_t)((NUM_ROUNDS >> 16) & 0xFF), (uint8_t)((NUM_ROUNDS >> 24) & 0xFF),
  };
  std::memcpy(param + 48, personal, 16);

  uint64_t h[8];
  for (unsigned i = 0; i < 8; ++i) h[i] = B2B_IV[i] ^ b2b_load64(param + i * 8);

  uint8_t blk[128] = {0};
  std::memcpy(blk, blob, BLOB_BYTES);   // 44 bytes, rest zero-padded
  uint64_t m[16];
  for (unsigned i = 0; i < 16; ++i) m[i] = b2b_load64(blk + i * 8);
  b2b_compress(h, m, BLOB_BYTES, 0, true);   // t0 = 44 (total absorbed), single + final block

  // 32-byte digest = h[0..3] as little-endian u64 words.
  for (unsigned i = 0; i < 4; ++i) prePow[i] = h[i];
}

// ===========================================================================================
// SipHash-2-4 (device + host). prePow[0..3] are v0..v3 directly (already xored into the magic by the
// blake2b output, per beam). nonce = the 64-bit counter (index<<3)+i. Mirrors the JS oracle exactly.
// ===========================================================================================
inline uint64_t rotl64(uint64_t x, unsigned b) { return (x << b) | (x >> (64 - b)); }

inline uint64_t siphash24(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3, uint64_t nonce) {
  uint64_t v0 = s0, v1 = s1, v2 = s2, v3 = s3;
  auto sip_round = [&]() {
    v0 += v1; v2 += v3;
    v1 = rotl64(v1, 13);
    v3 = rotl64(v3, 16);
    v1 ^= v0; v3 ^= v2;
    v0 = rotl64(v0, 32);
    v2 += v1; v0 += v3;
    v1 = rotl64(v1, 17);
    v3 = rotl64(v3, 21);
    v1 ^= v2; v3 ^= v0;
    v2 = rotl64(v2, 32);
  };
  v3 ^= nonce;
  sip_round(); sip_round();
  v0 ^= nonce;
  v2 ^= 0xffull;
  sip_round(); sip_round(); sip_round(); sip_round();
  return v0 ^ v1 ^ v2 ^ v3;
}

// ===========================================================================================
// StepElem on the device: a row carries WORK_WORDS u64 work bits (w[0]=LSB) + an index tree.
// The index tree at level L holds 2^L indices (lowest-index-first within each merge). For applyMix we
// only ever need the FIRST padNum (<=9) indices of the tree.
// ===========================================================================================

// makeLeaf: workBits word i = siphash24(prePow, (index<<3)+i). indexTree=[index].
inline void make_leaf(const uint64_t prePow[4], uint32_t index, uint64_t w[WORK_WORDS]) {
  const uint64_t base = ((uint64_t)index << 3);
  for (unsigned i = 0; i < WORK_WORDS; ++i)
    w[i] = siphash24(prePow[0], prePow[1], prePow[2], prePow[3], base + i);
}

// 448-bit right shift by COLLISION_BIT_SIZE (24) across WORK_WORDS u64 words, then mask to remLen bits.
inline void merge_workbits(const uint64_t a[WORK_WORDS], const uint64_t b[WORK_WORDS],
                           unsigned remLen, uint64_t out[WORK_WORDS]) {
  uint64_t x[WORK_WORDS];
  for (unsigned i = 0; i < WORK_WORDS; ++i) x[i] = a[i] ^ b[i];
  // shift right by 24 bits (logical, 448-bit).
  constexpr unsigned SH = COLLISION_BIT_SIZE;  // 24
  for (unsigned i = 0; i < WORK_WORDS; ++i) {
    uint64_t lo = x[i] >> SH;
    uint64_t hi = (i + 1 < WORK_WORDS) ? (x[i + 1] << (64 - SH)) : 0ull;
    out[i] = lo | hi;
  }
  // mask to low remLen bits.
  for (unsigned i = 0; i < WORK_WORDS; ++i) {
    const unsigned bit0 = i * 64;
    if (bit0 >= remLen) { out[i] = 0; continue; }
    const unsigned rem = remLen - bit0;
    if (rem < 64) out[i] &= ((rem == 0) ? 0ull : ((uint64_t)~0ull >> (64 - rem)));
  }
}

// applyMix: in-place modify w[] using the first `padNum` indices of the index tree.
//   tempBits(512) = workBits (8 u64 words, words 7 = 0).
//   padNum = min( ((512-remLen)+24)/25 , treeLen )
//   for i in 0..padNum-1: tempBits |= indexTree[i] << (remLen + i*25)   (512-bit shift)
//   result = sum_{i=0..7} rotl64(word_i, (29*(i+1))&63); result = rotl64(result, 24)
//   w[0] = result (low 64 work bits replaced).
inline void apply_mix(uint64_t w[WORK_WORDS], const uint32_t* tree, unsigned treeLen, unsigned remLen) {
  uint64_t t[8];
  for (unsigned i = 0; i < WORK_WORDS; ++i) t[i] = w[i];
  t[7] = 0;

  unsigned padNum = ((512u - remLen) + COLLISION_BIT_SIZE) / (COLLISION_BIT_SIZE + 1); // /25
  if (padNum > treeLen) padNum = treeLen;
  for (unsigned i = 0; i < padNum; ++i) {
    const unsigned bitpos = remLen + i * (COLLISION_BIT_SIZE + 1);   // remLen + i*25
    const uint64_t val = (uint64_t)tree[i];
    // OR `val << bitpos` into the 512-bit t[] (val is <=25 bits, spans at most 2 words).
    const unsigned word = bitpos >> 6;
    const unsigned off  = bitpos & 63;
    if (word < 8) t[word] |= (val << off);
    if (off != 0 && word + 1 < 8) t[word + 1] |= (val >> (64 - off));
  }

  uint64_t result = 0;
  for (unsigned i = 0; i < 8; ++i)
    result += rotl64(t[i], (29u * (i + 1)) & 0x3f);
  result = rotl64(result, 24);
  w[0] = result;
}

inline uint32_t collision_bits(const uint64_t w[WORK_WORDS]) {
  return (uint32_t)(w[0] & ((1u << COLLISION_BIT_SIZE) - 1u));
}

// ===========================================================================================
// CompressArray(25): pack 32 indices (25 bits each) into 100 bytes (matches getIndicesFromMinimal's
// inverse). The JS reference builds a bitset<800> LSB-first: field i occupies bits [i*25, i*25+25).
// So byte b's bit k (LSB) = stream bit (b*8+k). We emit 100 bytes little-endian-bit-packed.
// ===========================================================================================
static void compress_indices(const uint32_t idx[NUM_LEAVES], uint8_t out[100]) {
  // Build the 800-bit little-endian bit stream: out[byte] bit-k = field(byte*8+k).
  for (unsigned i = 0; i < 100; ++i) out[i] = 0;
  for (unsigned f = 0; f < NUM_LEAVES; ++f) {
    const uint32_t v = idx[f] & ((1u << INDEX_BITS) - 1u);
    const unsigned base = f * INDEX_BITS;   // bit position in the stream
    for (unsigned b = 0; b < INDEX_BITS; ++b) {
      if (v & (1u << b)) {
        const unsigned pos = base + b;
        out[pos >> 3] |= (uint8_t)(1u << (pos & 7));
      }
    }
  }
}

// ===========================================================================================
// Host SHA-256 (FIPS 180-4) + Beam Difficulty::IsTargetReached. Beam's PoW share check is
// SHA-256 of the 104-byte solution (ECC::Hash::Value, a 32-byte uintBig read BIG-endian = MSB
// first), tested against the 32-bit packed network difficulty (s_MantissaBits=24):
//   order    = packed >> 24
//   mantissa = (1<<24) | (packed & 0xFFFFFF)              // a 25-bit normalized value
//   a        = hashValue(256-bit) * mantissa              // up to a 281-bit product
//   reached  <=> a fits in (256 + 24 - order) bits, i.e. a < 2^(280-order)
// (beam core/difficulty.cpp IsTargetReached: a.get_ConstSlice().IsWithinOrder(nBits+s_MantissaBits-order).)
// Self-contained; runs only for the handful of candidate solutions per solve.
// ===========================================================================================
static constexpr uint32_t SHA256_K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};
inline uint32_t sha256_rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

static void sha256(const uint8_t* msg, size_t len, uint8_t out[32]) {
  uint32_t h[8] = { 0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                    0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u };
  const size_t total = ((len + 8) / 64 + 1) * 64;
  std::vector<uint8_t> buf(total, 0);
  std::memcpy(buf.data(), msg, len);
  buf[len] = 0x80;
  const uint64_t bits = (uint64_t)len * 8;
  for (unsigned i = 0; i < 8; ++i) buf[total - 1 - i] = (uint8_t)(bits >> (8 * i));
  for (size_t off = 0; off < total; off += 64) {
    uint32_t w[64];
    for (unsigned i = 0; i < 16; ++i)
      w[i] = ((uint32_t)buf[off + 4*i] << 24) | ((uint32_t)buf[off + 4*i + 1] << 16) |
             ((uint32_t)buf[off + 4*i + 2] << 8) | (uint32_t)buf[off + 4*i + 3];
    for (unsigned i = 16; i < 64; ++i) {
      const uint32_t s0 = sha256_rotr(w[i-15], 7) ^ sha256_rotr(w[i-15], 18) ^ (w[i-15] >> 3);
      const uint32_t s1 = sha256_rotr(w[i-2], 17) ^ sha256_rotr(w[i-2], 19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (unsigned i = 0; i < 64; ++i) {
      const uint32_t S1 = sha256_rotr(e,6) ^ sha256_rotr(e,11) ^ sha256_rotr(e,25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + S1 + ch + SHA256_K[i] + w[i];
      const uint32_t S0 = sha256_rotr(a,2) ^ sha256_rotr(a,13) ^ sha256_rotr(a,22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + maj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
  }
  for (unsigned i = 0; i < 8; ++i) {
    out[4*i]   = (uint8_t)(h[i] >> 24); out[4*i+1] = (uint8_t)(h[i] >> 16);
    out[4*i+2] = (uint8_t)(h[i] >> 8);  out[4*i+3] = (uint8_t)(h[i]);
  }
}

// Beam Difficulty::IsTargetReached: hash(32B, big-endian) * mantissa < 2^(280-order).
// hash_be[0] is the most-significant byte. The product (256 + 25 bits) is held in 16-bit limbs.
static bool beam_target_reached(const uint8_t hash_be[32], uint32_t packed) {
  constexpr uint32_t MANTISSA_BITS = 24;
  constexpr uint32_t MAX_ORDER     = 256u - MANTISSA_BITS - 1u; // s_MaxOrder
  constexpr uint32_t INF           = (MAX_ORDER + 1u) << MANTISSA_BITS;
  if (packed > INF) return false;   // invalid difficulty

  const uint32_t order    = packed >> MANTISSA_BITS;
  const uint64_t mantissa = (uint64_t)((1u << MANTISSA_BITS) | (packed & ((1u << MANTISSA_BITS) - 1u)));

  // a = hash (256-bit, big-endian) * mantissa, in 16 limbs of 16 bits (LSB-first), product fits 18 limbs.
  uint16_t hw[16];
  for (unsigned i = 0; i < 16; ++i)   // limb 0 = least-significant 16 bits = hash_be's last 2 bytes
    hw[i] = (uint16_t)(((uint32_t)hash_be[31 - 2*i - 1] << 8) | hash_be[31 - 2*i]);
  // Schoolbook: prod += hw[i]*mantissa << (16*i). mantissa is 25-bit -> per-limb partial is 41-bit.
  uint16_t prod[18] = {0};
  for (unsigned i = 0; i < 16; ++i) {
    uint64_t partial = (uint64_t)hw[i] * mantissa;   // up to 41 bits
    uint64_t cc = 0;
    for (unsigned j = 0; j < 4 && (i + j) < 18; ++j) {
      uint64_t sum = (uint64_t)prod[i + j] + (uint16_t)(partial & 0xffff) + cc;
      prod[i + j] = (uint16_t)(sum & 0xffff);
      cc = sum >> 16;
      partial >>= 16;
    }
    // propagate any remaining carry
    unsigned k = i + 4;
    while (cc && k < 18) {
      uint64_t sum = (uint64_t)prod[k] + cc;
      prod[k] = (uint16_t)(sum & 0xffff);
      cc = sum >> 16;
      ++k;
    }
  }

  // reached <=> product < 2^(280 - order). Find the index of the highest set bit.
  const uint32_t threshold = 280u - order;   // order <= MAX_ORDER=231 -> threshold in [49,280]
  int highbit = -1;
  for (int limb = 17; limb >= 0 && highbit < 0; --limb) {
    if (prod[limb] == 0) continue;
    for (int b = 15; b >= 0; --b)
      if (prod[limb] & (1u << b)) { highbit = limb * 16 + b; break; }
  }
  // value < 2^threshold  <=>  highest set bit index < threshold (or value == 0).
  return (highbit < 0) || ((uint32_t)highbit < threshold);
}

// (solver kernels + state appended below in M2/M3.)

#include "beamhash3_solver.inc"

} // namespace mom_beamhash3

using namespace mom_beamhash3;

// BeamHash III entrypoint. ABI matches equihash125_4 (c29-like out-of-band solution count).
//   input  : prework(32) || nonce(8) || extranonce(4) = 44-byte blob.
//   is_test (default): M1 gen-validation -- dumps the first BEAMHASH3_TEST_ROWS raw 448-bit leaf rows
//            (7 LE u64 words each) into solution_out for the JS-oracle diff. MOM_BEAMHASH3_SOLVE switches
//            it to the full Wagner solve and dumps the found 104-byte solution(s).
//   mining (!is_test): the full Wagner solve; writes [count:u8][count*104-byte solution] to solution_out.
int beamhash3(
  const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size,
  uint8_t* const solution_out, uint64_t* const /*pnonce*/, const uint8_t* const target,
  const unsigned /*intensity*/, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < BLOB_BYTES) throw std::string("Bad beamhash3 input length");
  BeamState& state = beam_state(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_io();

  // Host: IndividualWork (the SipHash key) = BLAKE2b(prework||nonce||extranonce, Beam-PoW personal).
  uint64_t prePow[4];
  individual_work(input, prePow);
  std::memcpy(state.prePow, prePow, 4 * sizeof(uint64_t));

  const bool solve_in_test = is_test && std::getenv("MOM_BEAMHASH3_SOLVE") != nullptr;

  if (is_test && !solve_in_test) {
    // M1: dump the first BEAMHASH3_TEST_ROWS raw 448-bit leaf rows.
    const size_t rows_bytes = (size_t)BEAMHASH3_TEST_ROWS * ROW_BYTES;
    std::memset(state.rows, 0, rows_bytes);
    sycl_wait_and_throw(submit_gen_test(state.queue, *state.bundle, state.prePow, state.rows), state.device);
    // Zero the WHOLE out-of-band buffer (not just rows_bytes) so the test-result hex dump -- which the
    // core emits over the full SMALL_BLOB_SOL_LEN -- is deterministic for the offline gen vector; the
    // 3584-byte gen dump leaves a 1536-byte tail that alloc_mem (_mm_malloc) does not zero otherwise.
    std::memset(solution_out, 0, SMALL_BLOB_SOL_LEN);
    std::memcpy(solution_out, state.rows, rows_bytes);
    (void)is_benchmark;
    return 1;
  }

  // ---- M2/M3 full Wagner solve. -------------------------------------------------------------------
  state.ensure_arenas(!is_benchmark);
  std::memset(solution_out, 0, SMALL_BLOB_SOL_LEN);
  const bool log = !is_benchmark;
  static const bool prof = std::getenv("MOM_BEAMHASH3_PROF") != nullptr;
  const uint64_t t_start = BeamState::now_ms();
  sycl::queue& q = state.queue;

  // Collide WG=640: with the SLM-staged collision keys the kernel is no longer DRAM-bound on the
  // collision field, so a larger WG (more slots resident per bucket SLM, better latency hiding) wins.
  // Pre-gen-fusion L4 sweep: 128=276 256=290 512=257 1024=272 -> 512. After gen fusion (round-1 scatter
  // regenerates leaves) the whole-solve sweep shifted: 384=262 512=232 640=228 768=228 ms -> 640 optimal.
  unsigned wg = 640;
  { unsigned long parsed = 0; if (mom_parse_env_ulong("MOM_BEAMHASH3_WORKGROUP", parsed) && parsed >= 16)
      wg = (unsigned)std::min<unsigned long>(parsed, 1024); }
  unsigned scatter_wg = 128;
  { unsigned long parsed = 0; if (mom_parse_env_ulong("MOM_BEAMHASH3_SCATTER_WG", parsed) && parsed >= 16)
      scatter_wg = (unsigned)std::min<unsigned long>(parsed, 1024); }

  for (unsigned i = 0; i <= NUM_ROUNDS; ++i) state.mixed_count[i] = 0;
  *state.cand_count = 0;
  // GEN FUSION: there is no separate gen kernel / level-0 arena. Round-1 scatter regenerates each leaf
  // in-kernel (slot == leaf index), so round 1's "input count" is simply all 2^25 leaves.
  state.mixed_count[0] = (uint32_t)NUM_ENTRIES;

  // ALL levels (1..5) are transient and share ONE scratch buffer. Level-0 (the leaves) is never
  // materialized -- round-1 scatter regenerates the leaf at slot==index. Recovery returns the slot
  // directly (leaf index == slot). Within a round, scatter fully drains the input into the bucket
  // arena (group_barrier between kernels), then collide reads ONLY the bucket arena and overwrites
  // scratch with level R. So a single scratch suffices for the whole pipeline.
  const uint64_t t_gen0 = prof ? BeamState::now_ms() : 0;
  if (prof) { std::fprintf(stderr, "beamhash3 gen: fused into round-1 scatter (%llu ms)\n",
    (unsigned long long)(BeamState::now_ms() - t_gen0)); std::fflush(stderr); }

  auto level_buf = [&](unsigned /*L*/) -> uint32_t* { return state.scratch[0]; };

  // Rounds 1..5: mix-scatter then collide.
  auto do_round = [&](auto Rconst) {
    constexpr unsigned R = decltype(Rconst)::value;
    constexpr unsigned L = R - 1;
    const uint32_t in_count = std::min<uint32_t>(state.mixed_count[L], (uint32_t)MIXED_CAP);
    sycl_wait_and_throw(q.memset(state.bucket_ns, 0, (size_t)NBUCKETS * sizeof(uint32_t)), state.device);
    const uint64_t t_sc = prof ? BeamState::now_ms() : 0;
    sycl_wait_and_throw(submit_mix_scatter<R>(q, *state.bundle, level_buf(L), in_count,
                                              state.bucket, state.bucket_ns, scatter_wg, state.prePow), state.device);
    const uint64_t t_co = prof ? BeamState::now_ms() : 0;
    sycl_wait_and_throw(submit_collide<R>(q, *state.bundle, state.bucket, state.bucket_ns,
                                          level_buf(R), state.mixed_count + R, state.tree[R], wg), state.device);
    if (prof || log) {
      const uint32_t out = std::min<uint32_t>(state.mixed_count[R], (uint32_t)MIXED_CAP);
      if (prof) {
        const uint64_t t_end = BeamState::now_ms();
        // Peak bucket demand this round (bucket_ns keeps counting past NSLOTS) -> NSLOTS sizing headroom.
        uint32_t maxns = 0;
        { std::vector<uint32_t> ns(NBUCKETS);
          q.memcpy(ns.data(), state.bucket_ns, (size_t)NBUCKETS * sizeof(uint32_t)).wait();
          for (uint32_t v : ns) if (v > maxns) maxns = v; }
        std::fprintf(stderr, "beamhash3 round %u: in=%u out=%u  scatter=%llu ms collide=%llu ms  maxbucket=%u/%u%s\n",
          R, in_count, out, (unsigned long long)(t_co - t_sc), (unsigned long long)(t_end - t_co),
          maxns, (unsigned)NSLOTS,
          state.mixed_count[R] > MIXED_CAP ? " (OVERFLOW)" : "");
      } else {
        std::fprintf(stderr, "beamhash3 round %u: in=%u out=%u%s\n", R, in_count, out,
          state.mixed_count[R] > MIXED_CAP ? " (OVERFLOW)" : "");
      }
      std::fflush(stderr);
    }
  };
  do_round(std::integral_constant<unsigned, 1>{});
  do_round(std::integral_constant<unsigned, 2>{});
  do_round(std::integral_constant<unsigned, 3>{});
  do_round(std::integral_constant<unsigned, 4>{});
  do_round(std::integral_constant<unsigned, 5>{});

  // Final: level-5 survivors with merged 24-bit == 0 -> walk tree -> 32 distinct leaf indices.
  // (Leaf index == level-0 slot; recovery needs only the tree[] logs.)
  BeamLevels lv;
  for (unsigned L = 0; L <= NUM_ROUNDS; ++L) lv.tree[L] = state.tree[L];
  const uint32_t fin_count = std::min<uint32_t>(state.mixed_count[NUM_ROUNDS], (uint32_t)MIXED_CAP);
  sycl_wait_and_throw(submit_final(q, *state.bundle, level_buf(NUM_ROUNDS), fin_count, lv,
                                   state.cand, state.cand_count, BeamState::CAND_CAP), state.device);

  const uint32_t cand_count = std::min<uint32_t>(*state.cand_count, BeamState::CAND_CAP);
  if (log) { std::fprintf(stderr, "beamhash3 candidates=%u (%llu ms)\n", cand_count,
    (unsigned long long)(BeamState::now_ms() - t_start)); std::fflush(stderr); }

  // ---- Host recovery: each candidate's 32 leaf indices -> CompressArray(25) -> 100 bytes; append the
  // 4-byte extranonce (the blob's last 4 bytes) to form the 104-byte solution. De-dup distinct. ----
  int n_solutions = 0;
  size_t distinct_count = 0;
  if (cand_count) {
    std::vector<BeamCandidate> cand_h(cand_count);
    sycl_wait_and_throw(q.memcpy(cand_h.data(), state.cand, cand_count * sizeof(BeamCandidate)), state.device);

    const uint8_t* extranonce = input + PREWORK_BYTES + NONCE_BYTES;   // 4 bytes

    std::vector<std::array<uint8_t, 104>> seen;
    for (uint32_t ci = 0; ci < cand_count; ++ci) {
      const uint32_t* leaves = cand_h[ci].leaves;
      // distinct check (defensive; the device already filtered).
      bool distinct = true;
      for (unsigned i = 0; i < NUM_LEAVES && distinct; ++i)
        for (unsigned j = i + 1; j < NUM_LEAVES; ++j)
          if (leaves[i] == leaves[j]) { distinct = false; break; }
      if (!distinct) continue;
      std::array<uint8_t, 104> sol{};
      uint8_t minimal[100];
      compress_indices(leaves, minimal);
      std::memcpy(sol.data(), minimal, 100);
      std::memcpy(sol.data() + 100, extranonce, 4);
      bool dup = false;
      for (const auto& s : seen) if (s == sol) { dup = true; break; }
      if (dup) continue;
      seen.push_back(sol);
    }
    distinct_count = seen.size();
    std::sort(seen.begin(), seen.end());

    // M5 mining path: every distinct solution is a valid BeamHash III proof, but only those whose
    // SHA-256 meets the packed Beam network difficulty are submittable. The is_test SOLVE path keeps
    // ALL distinct solutions (so the offline checker can assert the known keystone is present
    // regardless of the difficulty), and benching reports the raw distinct count as the Sol/s unit.
    // The packed int32 difficulty is carried in the low 4 bytes of the 32-byte big-endian target.
    const uint32_t packed = target ? (((uint32_t)target[28] << 24) | ((uint32_t)target[29] << 16) |
                                      ((uint32_t)target[30] << 8) | (uint32_t)target[31]) : 0;
    // MOM_BEAMHASH3_FORCE_SUBMIT: bypass the difficulty filter and emit the single best (lowest-hash)
    // distinct solution. A debug aid to exercise the live JSON-RPC `solution` submit ABI end-to-end on
    // a hard network difficulty (the pool will low-diff-reject it, which still proves the wire path).
    static const bool force_submit = std::getenv("MOM_BEAMHASH3_FORCE_SUBMIT") != nullptr;
    std::vector<std::array<uint8_t, 104>> emit;
    if (force_submit && !is_test && !is_benchmark && !seen.empty()) {
      size_t best = 0; uint8_t best_h[32]; sha256(seen[0].data(), 104, best_h);
      for (size_t i = 1; i < seen.size(); ++i) {
        uint8_t h[32]; sha256(seen[i].data(), 104, h);
        if (std::memcmp(h, best_h, 32) < 0) { std::memcpy(best_h, h, 32); best = i; }
      }
      emit.push_back(seen[best]);
    } else for (const auto& sol : seen) {
      if (is_test || is_benchmark || packed == 0) { emit.push_back(sol); continue; }
      uint8_t digest[32];
      sha256(sol.data(), 104, digest);
      if (beam_target_reached(digest, packed)) emit.push_back(sol);
    }

    // Hand back: [count:u8][count * 104-byte solution].
    const unsigned cap = (SMALL_BLOB_SOL_LEN - 1) / 104;
    n_solutions = (int)std::min<size_t>(emit.size(), cap);
    solution_out[0] = (uint8_t)n_solutions;
    for (int i = 0; i < n_solutions; ++i)
      std::memcpy(solution_out + 1 + (size_t)i * 104, emit[i].data(), 104);
    if (log) {
      std::fprintf(stderr, "beamhash3 emitted solutions=%d (of %zu distinct):\n", n_solutions, distinct_count);
      for (int i = 0; i < n_solutions; ++i) {
        std::fprintf(stderr, "  ");
        for (int b = 0; b < 104; ++b) std::fprintf(stderr, "%02x", emit[i][b]);
        std::fprintf(stderr, "\n");
      }
      std::fflush(stderr);
    }
  } else {
    solution_out[0] = 0;
  }

  if (log) { std::fprintf(stderr, "beamhash3 solve done (%llu ms, %d solutions, %zu distinct)\n",
    (unsigned long long)(BeamState::now_ms() - t_start), n_solutions, distinct_count); std::fflush(stderr); }

  if (is_test) return n_solutions > 0 ? 1 : 0;
  return is_benchmark ? (int)distinct_count : n_solutions;
}
