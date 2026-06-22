// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
//
// Equihash 125,4 (ZelHash / Flux) GPU solver -- Wagner bucket-collision (Tromp/djezo lineage).
//
// Pipeline:
//   gen    : 2^26 entries from a personalized "ZelProof" blake2b over the 140-byte header + the
//            ZelHash "twist" (16-aligned block prefix-sum of stock hashes), masked, split into 4
//            sub-elements, ExpandArray(25) -> 25-bit collision fields.
//   round  : (M2+) bucket by high BUCKBITS, collide low RESTBITS, XOR pairs -> next round.
//   recover: (M3+) walk tree -> 16 leaf indices -> CompressArray(26) -> 52-byte solution.
//
// Mining runs the complete solve path: gen-fill, four collision rounds, final scan, recovery, and target
// filtering. The default is_test path still uses the cheaper gen-kernel cross-check; set
// MOM_EQUIHASH_SOLVE to run the full block-400000 proof recovery vector.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "lib-internal.h"
#include "../native/consts.h"

namespace mom_equihash125_4 {

// ---- 125,4 parameters (fluxd equihash.{h,cpp}) ----
constexpr unsigned N = 125, K = 4;
constexpr unsigned COLLISION_BIT_LENGTH  = N / (K + 1);              // 25
constexpr unsigned COLLISION_BYTE_LENGTH = (COLLISION_BIT_LENGTH + 7) >> 3; // 4
constexpr unsigned HASH_LENGTH           = (K + 1) * COLLISION_BYTE_LENGTH; // 20  (== EQUIHASH_ROW_LEN)
constexpr unsigned INDICES_PER_HASH      = 512 / N;                 // 4
constexpr unsigned HASH_OUTPUT           = INDICES_PER_HASH * ((N + 7) >> 3); // 64
constexpr unsigned SUB_ELEMENT_BYTES     = (N + 7) >> 3;            // 16
constexpr uint64_t HEADER_LEN            = 140;
[[maybe_unused]] constexpr uint64_t NUM_ENTRIES = 1ull << (COLLISION_BIT_LENGTH + 1); // 2^26 (M2 rounds)
static_assert(HASH_LENGTH == EQUIHASH_ROW_LEN, "row length must match the host-side dump buffer width");

// ---- Wagner solver memory plan (RESTBITS=10 -> 32768 buckets). 2^26 entries over 32768 buckets =
// mean occupancy 2048; NSLOTS over-provisions toward ~0% drop. There are K+1=5 "levels": level 0 is
// the gen output bucketed on field-0, levels 1..3 are the round-0..2 survivors bucketed on field 1..3,
// and level 4 holds the round-3 survivors (final-field XOR candidates). Each round r reads level r,
// collides field r's RESTBITS, and writes level r+1.
constexpr unsigned RESTBITS  = 10;
constexpr unsigned BUCKBITS  = COLLISION_BIT_LENGTH - RESTBITS;     // 15
constexpr unsigned NBUCKETS  = 1u << BUCKBITS;                      // 32768
constexpr unsigned NRESTBINS = 1u << RESTBITS;                      // 1024 sub-bins per bucket (SLM)
constexpr unsigned NSLOTS    = 2112;                               // mean+~1.4σ (σ≈45); SLM-occupancy
                                                                   // sweet spot: rounds 94->73ms vs 2304,
                                                                   // ~0% drop. 2080/2048 regress (overflow
                                                                   // path serializes gen-fill: 25->139ms).
constexpr unsigned NLEVELS   = K + 1;                              // 5 bucket levels (0..4)

// PROGRESSIVE per-level slot shrink (Tromp-style, deep levels only). At level L the slot stores the
// ACTIVE collision field in w[0] (high BUCKBITS pick the bucket, low RESTBITS the SLM rest-bin) plus
// the carried higher fields in w[1..]. Each round collides the active field (XORs it to 0) and shifts
// the rest down, so level L+1 carries ONE FEWER field than level L:
//   level 0 : fields 0,1,2,3,4 (active=0) + w[5] eh_index  (level_fields=5)
//   level 1 : carried fields 1,2,3,4                       (level_fields=4)
//   level 2 : carried fields 2,3,4                         (level_fields=3)
//   level 3 : carried fields 3,4                           (level_fields=2)
//   level 4 : final field 4 (must be 0)                    (level_fields=1)
// eh_index (the level-0 leaf index) lives only at level 0 (w[5]); recovery walks the per-level tree
// logs to the level-0 leaves and reads it there. Tree log: for each survivor WRITTEN to level L+1 we
// record its two PARENT slot indices within the parent bucket as a Cantor-packed pair plus the parent
// bucket id, so recovery can walk L+1 -> L.
//
// LEVEL_FIELDS[L] = number of carried collision fields at level L (NLEVELS - L: 5,4,3,2,1).
// LEVEL_U32_TBL[L] = the STORED slot width in u32. The ~93%-of-solve cost is the scattered global slot
// WRITES + per-survivor atomic bucket fetch_adds, and on B580 GDDR6 those are sensitive to the slot
// STRIDE, not just byte volume. The HARD constraint from a per-level B580 sweep: a 16-byte / 4-word
// output slot is PATHOLOGICAL for a heavy writer round (level-1 at width 4 blew r0 124->220 ms -- 2
// records share a 32-byte line, so the atomic-driven scatter false-shares). So we never put a heavy
// writer on a 16-byte stride. The active-field-low PACK (see LEVEL_U32_TBL below) shrinks every level
// by one word WITHOUT losing field bits, landing L1 at 20 B (not the 16 B that false-shares); L2 stays
// at 5 words to keep its writer off 16 B. Net on B580: rounds 70 -> 56 ms, ~10.8 -> ~12.9 solve/s, and
// the footprint drops ~7.0 -> ~5.9 GiB.
constexpr unsigned level_fields(unsigned L) { return NLEVELS - L; }            // 5,4,3,2,1
// CO-LOCATED tree log: the per-survivor tree record (parent_bucket + Cantor pair, TREE_U32=2 words) is
// stored INLINE at the TAIL of the survivor's level slot instead of in a separate ~2.25 GiB arena.
// Attribution (eq-prof, MOM_EQ_SKIP=1) showed the SEPARATE tree-arena scatter was ~83% of round time
// (rounds 420 -> 70 ms when skipped): two independent multi-GiB write streams thrash the GDDR6
// write-combine buffers / TLB far worse than their byte volume implies. Folding the tree into the slot
// makes ONE coalesced write stream per survivor and lands every slot on its own line. The per-level slot
// widths and their PACKED/unpacked layout are documented at LEVEL_U32_TBL below (active-field-low pack).
// (Prior split-arena per-level sweep notes for the record-only widths are in git history at b2ec078; the
// pre-pack co-located {6,6,5,5,3} 7.03-GiB layout is at HEAD before this change.)
constexpr unsigned TREE_U32 = 2;                                              // [parent_bucket, cantor]
// ACTIVE-FIELD-LOW packing (Tromp redundancy elision). At level L>=1 the slot's ACTIVE collision field
// (w[0], collided by the NEXT round) has its high BUCKBITS EQUAL to the bucket the slot physically lives
// in -- so the high bits are redundant and need not be stored. We pack the active field's low RESTBITS
// (10 bits) together with the tree's parent_bucket (15 bits, == BUCKBITS) into a SINGLE w[0] word:
//   w0 = active_low(10) | (parent_bucket << RESTBITS)        [25 bits used]
//   w1..w[F-1] = the F-1 FOLLOWER fields (full 25 bits)      [F == level_fields(L)]
//   w[F] = Cantor(parent slot pair)
// -> level width = F+1 (one word less than the unpacked F-fields + 2-word tree = F+2). The active field
// XORs to 0 in its round and is then dropped; within a bucket all slots share the high BUCKBITS, so the
// low-RESTBITS compare IS the full 25-bit collision test (rest-bin == active_low). Recovery reads pbucket
// from w0>>RESTBITS and Cantor from w[F].
//   level 0 : UNPACKED 5 fields + eh_index (w[5])              = 6 words (24 B), NO tree (leaves)
//   level 1 : PACKED   [aLow|pbkt], f2,f3,f4, cantor          = 5 words (20 B)  <- heavy r0 writer 24->20
//   level 2 : UNPACKED 3 fields + [pbucket, cantor]           = 5 words (20 B)  <- 16B(4w packed) FALSE-SHARES
//   level 3 : PACKED   [aLow|pbkt], f4, cantor                = 3 words (12 B)
//   level 4 : PACKED   [aLow|pbkt], cantor                    = 2 words (8 B)
// Net on B580: a stride probe showed r0 22->19, r1 19->17 ms when L1 shrinks 24->20 B (the active-field-
// low pack realises that without losing field bits). L2 stays UNPACKED at 5 words: its packed 4-word/16-B
// stride re-triggers the documented heavy-writer false-share pathology. Total (6+5+5+3+2)=21 words/slot
// (was 25) -> ~5.9 GiB.
constexpr bool     LEVEL_PACKED_TBL[NLEVELS] = { false, true, false, true, true };
constexpr unsigned LEVEL_U32_TBL[NLEVELS]    = { 6, 5, 5, 3, 2 };             // co-located slot width/level
constexpr unsigned level_u32(unsigned L)    { return LEVEL_U32_TBL[L]; }
constexpr bool     level_packed(unsigned L) { return LEVEL_PACKED_TBL[L]; }
// Inline tree offset within an UNPACKED level-L slot (L>=1): the LAST TREE_U32 words. For PACKED levels
// the parent_bucket lives in w0's high bits and the Cantor in w[level_fields(L)] (see read_tree below).
constexpr unsigned tree_off(unsigned L)     { return level_u32(L) - TREE_U32; }
// Read the inline tree log (parent_bucket + Cantor pair) from a level-L slot pointer (L>=1), handling
// both packed and unpacked layouts uniformly.
inline void read_tree(const uint32_t* slotp, unsigned L, uint32_t& pbucket, uint32_t& cantor) {
  if (level_packed(L)) { pbucket = slotp[0] >> RESTBITS; cantor = slotp[level_fields(L)]; }
  else                 { const uint32_t* tl = slotp + tree_off(L); pbucket = tl[0]; cantor = tl[1]; }
}

// ===========================================================================================
// BLAKE2b-512 with personalization (RFC 7693 + BLAKE2 param block). Self-contained so the personal
// param block matches the Equihash 125,4 reference exactly. Used host-side (base midstate) and device-side
// (per-index finalize). All scalar / IGC-safe.
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
inline uint32_t b2b_load32le(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

inline void b2b_g(uint64_t* v, unsigned a, unsigned b, unsigned c, unsigned d, uint64_t x, uint64_t y) {
  v[a] = v[a] + v[b] + x; v[d] = b2b_rotr64(v[d] ^ v[a], 32);
  v[c] = v[c] + v[d];     v[b] = b2b_rotr64(v[b] ^ v[c], 24);
  v[a] = v[a] + v[b] + y; v[d] = b2b_rotr64(v[d] ^ v[a], 16);
  v[c] = v[c] + v[d];     v[b] = b2b_rotr64(v[b] ^ v[c], 63);
}

// Compress a 128-byte block (16 u64 m[]) into the 8-word chain h[] with counter t0/t1 and final flag.
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

// The personalized blake2b base state after absorbing the full 140-byte header: 8 chain words +
// the 12 trailing pending bytes (header[128..139]) that have not yet been compressed. t0 is fixed at
// 128 (exactly one full block was compressed). Per-index work appends htole32(g) -> 16 pending bytes.
struct BaseState {
  uint64_t h[8];
  uint8_t  pending[12];   // header bytes 128..139
};

// Build the personalized base state on the host from the 140-byte header.
static BaseState make_base_state(const uint8_t* header140) {
  // Personalization param block (64 bytes): out_len=64, fanout=1, depth=1, personal[48..63].
  // personal = "ZelProof" || htole32(125) || htole32(4).
  uint8_t param[64] = {0};
  param[0] = HASH_OUTPUT;  // 64
  param[2] = 1;            // fanout
  param[3] = 1;            // depth
  const uint8_t personal[16] = {
    'Z','e','l','P','r','o','o','f',
    (uint8_t)(N & 0xFF), (uint8_t)((N >> 8) & 0xFF), (uint8_t)((N >> 16) & 0xFF), (uint8_t)((N >> 24) & 0xFF),
    (uint8_t)(K & 0xFF), (uint8_t)((K >> 8) & 0xFF), (uint8_t)((K >> 16) & 0xFF), (uint8_t)((K >> 24) & 0xFF),
  };
  std::memcpy(param + 48, personal, 16);

  BaseState st;
  for (unsigned i = 0; i < 8; ++i) st.h[i] = B2B_IV[i] ^ b2b_load64(param + i * 8);

  // Absorb 140 header bytes: one full 128-byte block compressed (t0=128), 12 bytes left pending.
  uint64_t m[16];
  for (unsigned i = 0; i < 16; ++i) m[i] = b2b_load64(header140 + i * 8);
  b2b_compress(st.h, m, 128, 0, false);
  for (unsigned i = 0; i < 12; ++i) st.pending[i] = header140[128 + i];
  return st;
}

// ---- Device-side stock hash: blake2b( base || htole32(g) ), 64 bytes -> 16 LE u32 lanes (acc[16]).
// base->pending(12) + 4 nonce-index bytes = 16 pending bytes, then finalize (t0 = 128 + 16 = 144).
inline void stock_hash_words(const uint64_t base_h[8], const uint8_t base_pending[12], uint32_t g,
                             uint32_t out[16]) {
  uint8_t blk[128];
  for (unsigned i = 0; i < 12; ++i) blk[i] = base_pending[i];
  blk[12] = (uint8_t)(g       & 0xFF);
  blk[13] = (uint8_t)((g >> 8) & 0xFF);
  blk[14] = (uint8_t)((g >> 16) & 0xFF);
  blk[15] = (uint8_t)((g >> 24) & 0xFF);
  for (unsigned i = 16; i < 128; ++i) blk[i] = 0;

  uint64_t h[8];
  for (unsigned i = 0; i < 8; ++i) h[i] = base_h[i];
  uint64_t m[16];
  for (unsigned i = 0; i < 16; ++i) m[i] = b2b_load64(blk + i * 8);
  b2b_compress(h, m, 144, 0, true);   // t0 = 128 (base) + 16 (pending) = 144; last block

  // Emit the 64-byte digest as 16 little-endian u32 words.
  for (unsigned w = 0; w < 8; ++w) {
    out[2 * w]     = (uint32_t)(h[w] & 0xffffffffull);
    out[2 * w + 1] = (uint32_t)(h[w] >> 32);
  }
}

// ---- 32-bit-pair (Uint2) BLAKE2b for the device gen-fill hot path. Xe (B580/Xe2) has no native 64-bit
// integer ALU: the scalar uint64_t stock_hash_words above lowers every add/rotr/xor to emulated 64-bit
// ops. Splitting each 64-bit word into a {lo, hi} 32-bit pair lets the EU run native 32-bit ALU (rotr by
// 32 -> half-swap, by 24/16/63 -> shift pairs, add -> add-with-carry). Same technique as autolykos2's
// prehash (~1.7x there). Bit-identical to stock_hash_words; the gen output is the 16 LE u32 lanes
// out[2w]=h[w].lo, out[2w+1]=h[w].hi -- so the pair form maps to the output with NO recombination.
struct B2bPair { uint32_t lo, hi; };

inline void b2b_g_pair(B2bPair& a, B2bPair& b, B2bPair& c, B2bPair& d, B2bPair x, B2bPair y) {
  uint32_t lo = a.lo + b.lo;
  uint32_t carry = lo < b.lo ? 1u : 0u;
  a.lo = lo + x.lo; carry += a.lo < x.lo ? 1u : 0u;
  a.hi = a.hi + b.hi + x.hi + carry;
  { const uint32_t t = d.lo ^ a.lo; d.lo = d.hi ^ a.hi; d.hi = t; }      // rotr64(.,32): swap halves
  lo = c.lo + d.lo; c.hi = c.hi + d.hi + (lo < d.lo ? 1u : 0u); c.lo = lo;
  { const uint32_t bl = b.lo ^ c.lo, bh = b.hi ^ c.hi;                   // rotr64(.,24)
    b.lo = (bl >> 24) | (bh << 8); b.hi = (bh >> 24) | (bl << 8); }
  lo = a.lo + b.lo; carry = lo < b.lo ? 1u : 0u;
  a.lo = lo + y.lo; carry += a.lo < y.lo ? 1u : 0u;
  a.hi = a.hi + b.hi + y.hi + carry;
  { const uint32_t dl = d.lo ^ a.lo, dh = d.hi ^ a.hi;                   // rotr64(.,16)
    d.lo = (dl >> 16) | (dh << 16); d.hi = (dh >> 16) | (dl << 16); }
  lo = c.lo + d.lo; c.hi = c.hi + d.hi + (lo < d.lo ? 1u : 0u); c.lo = lo;
  { const uint32_t bl = b.lo ^ c.lo, bh = b.hi ^ c.hi;                   // rotr64(.,63) == rotl64(.,1)
    b.lo = (bl << 1) | (bh >> 31); b.hi = (bh << 1) | (bl >> 31); }
}

// Pair-form stock hash. The message block is base_pending[0..11] || htole32(g) || zeros, so only the
// first two message words are non-zero: m0={pending[0..3], pending[4..7]}, m1={pending[8..11], g}.
// Pass those two precomputed (loop-invariant across the 16-block) as m0/m1.
inline void stock_hash_words_pair(const B2bPair base_h[8], B2bPair m0, B2bPair m1, uint32_t g,
                                  uint32_t out[16]) {
  m1.hi = g;   // bytes 12..15 of the block (htole32(g)); pending fills lo + m0
  B2bPair v[16];
  for (unsigned i = 0; i < 8; ++i) v[i] = base_h[i];
  for (unsigned i = 0; i < 8; ++i)
    v[i + 8] = B2bPair{ (uint32_t)(B2B_IV[i] & 0xffffffffull), (uint32_t)(B2B_IV[i] >> 32) };
  v[12].lo ^= 144u;                        // t0 = 144; t1 = 0
  v[14].lo = ~v[14].lo; v[14].hi = ~v[14].hi;   // last block
  const B2bPair zero{0u, 0u};
  // Message words: index 0 -> m0, index 1 -> m1, all others 0.
  auto mw = [&](unsigned i) -> B2bPair { return i == 0 ? m0 : (i == 1 ? m1 : zero); };
  for (unsigned r = 0; r < 12; ++r) {
    const uint8_t* s = B2B_SIGMA[r];
    b2b_g_pair(v[0], v[4], v[ 8], v[12], mw(s[0]),  mw(s[1]));
    b2b_g_pair(v[1], v[5], v[ 9], v[13], mw(s[2]),  mw(s[3]));
    b2b_g_pair(v[2], v[6], v[10], v[14], mw(s[4]),  mw(s[5]));
    b2b_g_pair(v[3], v[7], v[11], v[15], mw(s[6]),  mw(s[7]));
    b2b_g_pair(v[0], v[5], v[10], v[15], mw(s[8]),  mw(s[9]));
    b2b_g_pair(v[1], v[6], v[11], v[12], mw(s[10]), mw(s[11]));
    b2b_g_pair(v[2], v[7], v[ 8], v[13], mw(s[12]), mw(s[13]));
    b2b_g_pair(v[3], v[4], v[ 9], v[14], mw(s[14]), mw(s[15]));
  }
  for (unsigned w = 0; w < 8; ++w) {
    out[2 * w]     = base_h[w].lo ^ v[w].lo ^ v[w + 8].lo;
    out[2 * w + 1] = base_h[w].hi ^ v[w].hi ^ v[w + 8].hi;
  }
}

// ---- ZelHash twist (NAIVE): for entry g sum stock hashes over [g & ~0xF .. g] lane-wise (u32 wrap),
// then mask raw bytes 15/31/47/63 with &0xF8. Produces the 64-byte (16 u32) generator output.
inline void zelhash_twist_naive(const uint64_t base_h[8], const uint8_t base_pending[12], uint32_t g,
                                uint32_t acc[16]) {
  const uint32_t start = g & 0xFFFFFFF0u;
  for (unsigned i = 0; i < 16; ++i) acc[i] = 0;
  for (uint32_t g2 = start; g2 <= g; ++g2) {
    uint32_t tmp[16];
    stock_hash_words(base_h, base_pending, g2, tmp);
    for (unsigned i = 0; i < 16; ++i) acc[i] += tmp[i];   // lane-wise u32 add, wraparound
  }
  // Mask bytes 15,31,47,63 (clear low 3 bits): each is the high byte of acc lane 3,7,11,15.
  acc[3]  &= 0xF8FFFFFFu;
  acc[7]  &= 0xF8FFFFFFu;
  acc[11] &= 0xF8FFFFFFu;
  acc[15] &= 0xF8FFFFFFu;
}

// ---- ExpandArray(in16, bitLen=25, bytePad=0) -> 20 bytes (5 fields of 25 bits in 4-byte BE slots).
// Bit-exact port of Zcash util.cpp ExpandArray. Used on the (g%4) sub-element.
inline void expand_array_25(const uint8_t in[SUB_ELEMENT_BYTES], uint8_t out[HASH_LENGTH]) {
  constexpr unsigned BIT_LEN  = COLLISION_BIT_LENGTH;     // 25
  constexpr unsigned OUT_WIDTH = (BIT_LEN + 7) >> 3;      // 4
  constexpr uint32_t BIT_MASK = (1u << BIT_LEN) - 1u;     // 0x1FFFFFF
  uint32_t acc_value = 0;
  unsigned acc_bits = 0, j = 0;
  for (unsigned i = 0; i < SUB_ELEMENT_BYTES; ++i) {
    acc_value = (acc_value << 8) | in[i];
    acc_bits += 8;
    if (acc_bits >= BIT_LEN) {
      acc_bits -= BIT_LEN;
      for (unsigned x = 0; x < OUT_WIDTH; ++x) {
        const unsigned shift = acc_bits + 8u * (OUT_WIDTH - x - 1);
        out[j + x] = (uint8_t)((acc_value >> shift) & ((BIT_MASK >> (8u * (OUT_WIDTH - x - 1))) & 0xFFu));
      }
      j += OUT_WIDTH;
    }
  }
}

// ---- The expanded 20-byte collision row for entry index e (== ref indexHashRow(e)). ----
// generateZelHash(e/4) -> 64B; take the (e%4) 16-byte sub-element; ExpandArray(25) -> 20 bytes.
inline void entry_row_naive(const uint64_t base_h[8], const uint8_t base_pending[12], uint32_t e,
                            uint8_t row[HASH_LENGTH]) {
  uint32_t acc[16];
  zelhash_twist_naive(base_h, base_pending, e / INDICES_PER_HASH, acc);
  // Reinterpret acc as 64 bytes (LE u32 words), slice the (e%4)-th 16-byte sub-element.
  uint8_t sub[SUB_ELEMENT_BYTES];
  const unsigned base_word = (e % INDICES_PER_HASH) * (SUB_ELEMENT_BYTES / 4);  // *4 words
  for (unsigned w = 0; w < SUB_ELEMENT_BYTES / 4; ++w) {
    const uint32_t v = acc[base_word + w];
    sub[4 * w]     = (uint8_t)(v & 0xFF);
    sub[4 * w + 1] = (uint8_t)((v >> 8) & 0xFF);
    sub[4 * w + 2] = (uint8_t)((v >> 16) & 0xFF);
    sub[4 * w + 3] = (uint8_t)((v >> 24) & 0xFF);
  }
  expand_array_25(sub, row);
}

// ---- Decode the 20-byte expanded row (5 big-endian 25-bit fields in 4-byte slots) into 5 u32s.
// This matches the JS reference's xorRows()/hasCollision() field granularity exactly: field f lives
// in row bytes [4f, 4f+4) big-endian, value in [0, 2^25). Collisions zero a whole field.
inline void row_to_fields(const uint8_t row[HASH_LENGTH], uint32_t f[NLEVELS]) {
  for (unsigned k = 0; k < NLEVELS; ++k) {
    const unsigned o = k * COLLISION_BYTE_LENGTH;
    f[k] = ((uint32_t)row[o] << 24) | ((uint32_t)row[o + 1] << 16) |
           ((uint32_t)row[o + 2] << 8) | (uint32_t)row[o + 3];
  }
}

// ===========================================================================================
// M1 gen-kernel validation: compute the first EQUIHASH_TEST_ROWS entries' 20-byte expanded rows.
// NAIVE_TWIST=true uses the per-entry naive twist; false uses the sub-group inclusive-scan over the
// 16-aligned block (one sub-group per block prefix-sums the 16 stock hashes). Both must produce
// identical rows; the entrypoint runs both so hash-vector tests cover both implementations.
// ===========================================================================================
template <bool NAIVE_TWIST> class EquihashGenTestKernel;

template <bool NAIVE_TWIST>
static sycl::event submit_gen_test(
  sycl::queue& q, MOM_BUNDLE_T& kb, const uint64_t* d_base_h, const uint8_t* d_base_pending,
  uint8_t* d_rows /* EQUIHASH_TEST_ROWS * HASH_LENGTH */
) {
  if constexpr (NAIVE_TWIST) {
    constexpr unsigned WG = 64;
    const size_t global = (EQUIHASH_TEST_ROWS + WG - 1) / WG * WG;
    return q.submit([&](sycl::handler& h) {
      MOM_USE_BUNDLE(h, kb);
      h.parallel_for<EquihashGenTestKernel<true>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const uint32_t e = (uint32_t)it.get_global_id(0);
        if (e >= EQUIHASH_TEST_ROWS) return;
        uint64_t base_h[8];
        for (unsigned i = 0; i < 8; ++i) base_h[i] = d_base_h[i];
        uint8_t pending[12];
        for (unsigned i = 0; i < 12; ++i) pending[i] = d_base_pending[i];
        uint8_t row[HASH_LENGTH];
        entry_row_naive(base_h, pending, e, row);
        for (unsigned i = 0; i < HASH_LENGTH; ++i) d_rows[(size_t)e * HASH_LENGTH + i] = row[i];
      });
    });
  } else {
    // Optimized: one sub-group of 16 lanes per 16-aligned block. Each lane g2 computes its own stock
    // hash; inclusive_scan_over_group (lane-wise u32 add) gives the twist prefix-sum for every entry
    // in the block in one pass. EQUIHASH_TEST_ROWS entries span EQUIHASH_TEST_ROWS/4 hash indices, but
    // each hash index e/4 belongs to a 16-block; we run one work-group of 16 per block touched.
    // The largest hash index we need is (EQUIHASH_TEST_ROWS-1)/4; round up to whole 16-blocks.
    constexpr unsigned MAX_HASH_IDX = (EQUIHASH_TEST_ROWS - 1) / INDICES_PER_HASH; // inclusive
    constexpr unsigned NUM_BLOCKS   = (MAX_HASH_IDX / 16) + 1;
    constexpr unsigned SG = 16;
    const size_t global = (size_t)NUM_BLOCKS * SG;
    return q.submit([&](sycl::handler& h) {
      MOM_USE_BUNDLE(h, kb);
      h.parallel_for<EquihashGenTestKernel<false>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(SG)),
        [=](sycl::nd_item<1> it) MOM_REQD_SG_16 {
        const sycl::sub_group sg = it.get_sub_group();
        const unsigned lane = sg.get_local_linear_id();          // 0..15 within the block
        const uint32_t block = (uint32_t)it.get_group(0);        // which 16-aligned hash-index block
        const uint32_t hash_idx = block * 16u + lane;            // this lane's hash index g

        uint64_t base_h[8];
        for (unsigned i = 0; i < 8; ++i) base_h[i] = d_base_h[i];
        uint8_t pending[12];
        for (unsigned i = 0; i < 12; ++i) pending[i] = d_base_pending[i];

        // This lane's own stock hash (16 u32 lanes), then an inclusive prefix-sum across the 16 lanes.
        uint32_t mine[16];
        stock_hash_words(base_h, pending, hash_idx, mine);
        uint32_t acc[16];
        for (unsigned i = 0; i < 16; ++i)
          acc[i] = sycl::inclusive_scan_over_group(sg, mine[i], sycl::plus<uint32_t>());

        // acc now holds the twist accumulator for hash index `hash_idx` (sum over its 16-block prefix
        // up to and including this lane). Mask bytes 15/31/47/63, then emit the 4 sub-element rows for
        // the (up to) 4 entry indices e = hash_idx*4 + s that fall inside EQUIHASH_TEST_ROWS.
        acc[3]  &= 0xF8FFFFFFu; acc[7]  &= 0xF8FFFFFFu;
        acc[11] &= 0xF8FFFFFFu; acc[15] &= 0xF8FFFFFFu;
        for (unsigned s = 0; s < INDICES_PER_HASH; ++s) {
          const uint32_t e = hash_idx * INDICES_PER_HASH + s;
          if (e >= EQUIHASH_TEST_ROWS) continue;
          uint8_t sub[SUB_ELEMENT_BYTES];
          const unsigned base_word = s * (SUB_ELEMENT_BYTES / 4);
          for (unsigned w = 0; w < SUB_ELEMENT_BYTES / 4; ++w) {
            const uint32_t v = acc[base_word + w];
            sub[4 * w]     = (uint8_t)(v & 0xFF);
            sub[4 * w + 1] = (uint8_t)((v >> 8) & 0xFF);
            sub[4 * w + 2] = (uint8_t)((v >> 16) & 0xFF);
            sub[4 * w + 3] = (uint8_t)((v >> 24) & 0xFF);
          }
          uint8_t row[HASH_LENGTH];
          expand_array_25(sub, row);
          for (unsigned i = 0; i < HASH_LENGTH; ++i) d_rows[(size_t)e * HASH_LENGTH + i] = row[i];
        }
      });
    });
  }
}

// ===========================================================================================
// M2/M3 solver kernels: gen-fill (round-0 bucketing) + collision rounds + candidate collection.
// ===========================================================================================
//
// A "level L" slot lives at  level[L] + ((bucket * NSLOTS + slot) * level_u32(L))  (width 6,5,5,3,2 u32).
// Its tree log is CO-LOCATED in the slot (read via read_tree): for UNPACKED levels [parent_bucket,
// Cantor(a,b)] sits at the slot tail (tree_off(L)); for PACKED levels (L1/L3/L4) parent_bucket folds into
// w0's high bits and Cantor is at w[level_fields(L)]. a>b are the two parent slot indices within the
// parent bucket at level L-1. Level 0 has no tree (leaves carry the eh_index in w[5] directly).
// nslots[L*NBUCKETS + bucket] counts occupancy.

using dev_atomic_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device, sycl::access::address_space::global_space>;
using slm_atomic_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::work_group, sycl::access::address_space::local_space>;

inline uint32_t cantor_pack(uint32_t a, uint32_t b) {   // a>b assumed by caller
  return a * (a + 1u) / 2u + b;
}
inline void cantor_unpack(uint32_t c, uint32_t& a, uint32_t& b) {
  // Inverse of a*(a+1)/2 + b with a>=b: a = floor((sqrt(8c+1)-1)/2).
  uint32_t a_ = (uint32_t)((sycl::sqrt(8.0 * (double)c + 1.0) - 1.0) / 2.0);
  while ((a_ + 1u) * (a_ + 2u) / 2u <= c) ++a_;          // correct float rounding
  while (a_ != 0 && a_ * (a_ + 1u) / 2u > c) --a_;
  a = a_; b = c - a_ * (a_ + 1u) / 2u;
}

// ---- Gen-fill: produce all 2^26 entries and scatter each into its round-0 bucket. One sub-group of
// 16 lanes per 16-aligned hash-index block (the validated M1 scan), then each lane emits its 4
// sub-element entries. Bucket = high BUCKBITS of field-0; slot record carries fields 0..4 + eh_index.
class EquihashGenFillKernel;
static sycl::event submit_gen_fill(
  sycl::queue& q, MOM_BUNDLE_T& kb, const uint64_t* d_base_h, const uint8_t* d_base_pending,
  uint32_t* d_level0, uint32_t* d_nslots
) {
  constexpr uint32_t NUM_HASHES = (uint32_t)(NUM_ENTRIES / INDICES_PER_HASH);   // 2^24
  constexpr uint32_t NUM_BLOCKS = NUM_HASHES / 16u;
  constexpr unsigned SG = 16;
  const size_t global = (size_t)NUM_BLOCKS * SG;
  return q.submit([&](sycl::handler& h) {
    MOM_USE_BUNDLE(h, kb);
    h.parallel_for<EquihashGenFillKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(SG)),
      [=](sycl::nd_item<1> it) MOM_REQD_SG_16 {
      const sycl::sub_group sg = it.get_sub_group();
      const unsigned lane = sg.get_local_linear_id();
      const uint32_t block = (uint32_t)it.get_group(0);
      const uint32_t hash_idx = block * 16u + lane;

      // Pair-form base state + the two non-zero message words (block = pending[0..11] || htole32(g)).
      // These are loop-invariant across the 16-block; only g (m1.hi) varies per lane.
      B2bPair base_hp[8];
      for (unsigned i = 0; i < 8; ++i)
        base_hp[i] = B2bPair{ (uint32_t)(d_base_h[i] & 0xffffffffull), (uint32_t)(d_base_h[i] >> 32) };
      uint8_t pending[12];
      for (unsigned i = 0; i < 12; ++i) pending[i] = d_base_pending[i];
      const B2bPair m0{ b2b_load32le(pending + 0), b2b_load32le(pending + 4) };
      const B2bPair m1{ b2b_load32le(pending + 8), 0u };   // m1.hi = g, set inside stock_hash_words_pair

      uint32_t mine[16];
      stock_hash_words_pair(base_hp, m0, m1, hash_idx, mine);
      uint32_t acc[16];
      for (unsigned i = 0; i < 16; ++i)
        acc[i] = sycl::inclusive_scan_over_group(sg, mine[i], sycl::plus<uint32_t>());
      acc[3] &= 0xF8FFFFFFu; acc[7] &= 0xF8FFFFFFu; acc[11] &= 0xF8FFFFFFu; acc[15] &= 0xF8FFFFFFu;

      for (unsigned s = 0; s < INDICES_PER_HASH; ++s) {
        const uint32_t e = hash_idx * INDICES_PER_HASH + s;
        uint8_t sub[SUB_ELEMENT_BYTES];
        const unsigned base_word = s * (SUB_ELEMENT_BYTES / 4);
        for (unsigned w = 0; w < SUB_ELEMENT_BYTES / 4; ++w) {
          const uint32_t v = acc[base_word + w];
          sub[4 * w]     = (uint8_t)(v & 0xFF);
          sub[4 * w + 1] = (uint8_t)((v >> 8) & 0xFF);
          sub[4 * w + 2] = (uint8_t)((v >> 16) & 0xFF);
          sub[4 * w + 3] = (uint8_t)((v >> 24) & 0xFF);
        }
        uint8_t row[HASH_LENGTH];
        expand_array_25(sub, row);
        uint32_t f[NLEVELS];
        row_to_fields(row, f);

        const uint32_t bucket = f[0] >> RESTBITS;                 // high BUCKBITS of field-0
        const uint32_t slot = dev_atomic_u32(d_nslots[bucket]).fetch_add(1u);
        if (slot >= NSLOTS) continue;                             // drop on overflow
        uint32_t* rec = d_level0 + ((size_t)bucket * NSLOTS + slot) * level_u32(0);  // 6-word level-0 slot
        rec[0] = f[0]; rec[1] = f[1]; rec[2] = f[2]; rec[3] = f[3]; rec[4] = f[4];
        rec[5] = e;   // eh_index (leaf), at w[level_fields(0)] == w[5]
      }
    });
  });
}

// ---- Collision round R: read level R, collide field R across (bucket, RESTBITS) pairs, XOR the
// carried fields -> write level R+1. One work-group per bucket; SLM rest-bin table indexes by the low
// RESTBITS of field R. For each colliding pair we XOR fields R+1..4, write a survivor at level R+1 (its
// bucket = high BUCKBITS of the new active field R+1), and record the Cantor-packed parent slot pair.
template <unsigned R> class EquihashRoundKernel;
template <unsigned R>
static sycl::event submit_round(
  sycl::queue& q, MOM_BUNDLE_T& kb, const uint32_t* d_in, uint32_t* d_out,
  const uint32_t* d_in_nslots, uint32_t* d_out_nslots, unsigned wg
) {
  // Progressive per-level widths (compile-time): round R reads level R (IN_FIELDS field words at stride
  // IN_U32) and writes level R+1 (OUT_FIELDS = IN_FIELDS-1 field words at stride OUT_U32). Fewer words
  // written per survivor == less scattered global write traffic (the bandwidth bottleneck).
  constexpr unsigned IN_FIELDS  = level_fields(R);       // 5,4,3,2 for R=0..3
  constexpr unsigned IN_U32     = level_u32(R);          // input slot stride (6,6,5,5)
  constexpr unsigned OUT_FIELDS = level_fields(R + 1);   // 4,3,2,1
  constexpr unsigned OUT_U32    = level_u32(R + 1);      // output slot stride (6,5,5,3)
  constexpr unsigned OUT_TREE_OFF = tree_off(R + 1);     // inline tree offset in the output slot
  const size_t global = (size_t)NBUCKETS * wg;
  return q.submit([&](sycl::handler& h) {
    MOM_USE_BUNDLE(h, kb);
    // SLM rest-bin table: bin_head[r] = (1 + slot index of the most-recent slot in rest-bin r), 0=empty;
    // bin_next[slot] threads a singly-linked list so every earlier slot in the bin is a collision pair.
    sycl::local_accessor<uint32_t, 1> bin_head(sycl::range<1>(NRESTBINS), h);
    sycl::local_accessor<uint32_t, 1> bin_next(sycl::range<1>(NSLOTS), h);
    // Cache each slot's IN_FIELDS collision fields (w[0..IN_FIELDS-1]) in SLM. The collision walk then
    // reads BOTH the active-field compare and the XOR operands from SLM instead of re-fetching scattered
    // global records -- a record is loaded once and reused across all its pairs (mean ~2), roughly
    // halving the round's global read traffic. With the progressive shrink later rounds cache fewer
    // words/slot, freeing SLM. (eh_index is excluded -- unused in rounds.)
    constexpr unsigned SLM_REC = IN_FIELDS;
    sycl::local_accessor<uint32_t, 1> slm_rec(sycl::range<1>((size_t)NSLOTS * SLM_REC), h);
    h.parallel_for<EquihashRoundKernel<R>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
      [=](sycl::nd_item<1> it) {
      const uint32_t bucket = (uint32_t)it.get_group(0);
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);

      uint32_t cnt = d_in_nslots[bucket];
      if (cnt > NSLOTS) cnt = NSLOTS;
      for (unsigned r = lid; r < NRESTBINS; r += lsz) bin_head[r] = 0;
      sycl::group_barrier(it.get_group());

      const uint32_t* in_base = d_in + (size_t)bucket * NSLOTS * IN_U32;

      // Build the rest-bin linked lists AND load each slot's IN_FIELDS fields into SLM (one per thread).
      // For a PACKED input level, w[0] holds (active_low | pbucket<<RESTBITS); we cache only active_low
      // (the low RESTBITS suffice: within a bucket the high BUCKBITS are all == bucket, so the rest-bin
      // and the collision compare are exactly the low-RESTBITS test). The followers (w[1..]) are full.
      constexpr bool IN_PACKED = level_packed(R);
      for (unsigned s = lid; s < cnt; s += lsz) {
        const uint32_t* g = in_base + (size_t)s * IN_U32;
        uint32_t* l = &slm_rec[(size_t)s * SLM_REC];
        l[0] = IN_PACKED ? (g[0] & (NRESTBINS - 1u)) : g[0];        // active (low RESTBITS if packed)
        for (unsigned k = 1; k < IN_FIELDS; ++k) l[k] = g[k];       // carried followers (full 25-bit)
        const uint32_t rb = l[0] & (NRESTBINS - 1u);
        bin_next[s] = slm_atomic_u32(bin_head[rb]).exchange(s + 1u);
      }
      sycl::group_barrier(it.get_group());

      // For each slot, walk the chain of earlier same-bin slots and emit one survivor per pair.
      for (unsigned s = lid; s < cnt; s += lsz) {
        const uint32_t* a = &slm_rec[(size_t)s * SLM_REC];
        const uint32_t af0 = a[0];
        const uint32_t rb = af0 & (NRESTBINS - 1u);
        uint32_t link = bin_head[rb];                               // 1 + head slot
        // Skip forward to entries strictly before s (the head list is newest-first; entries after s in
        // emission would double-count). We emit a pair (s, t) only when t < s.
        for (; link != 0; link = bin_next[link - 1u]) {
          const uint32_t t = link - 1u;
          if (t >= s) continue;                                     // emit each unordered pair once
          const uint32_t* b = &slm_rec[(size_t)t * SLM_REC];
          if (b[0] != af0) continue;                                // full active field (25-bit) must match
          // Slots always keep the ACTIVE collision field in w[0] and the remaining (higher) fields in
          // w[1..]. The active field XORs to 0; shift the rest down by one so w[0] becomes the NEXT
          // collision field at level R+1. (Position-relative, independent of which absolute field R is.)
          uint32_t xf[IN_FIELDS];
          for (unsigned k = 0; k < IN_FIELDS; ++k) xf[k] = a[k] ^ b[k];   // xf[0] == 0 by construction
          const uint32_t new_active = xf[1];                      // new active field (full 25-bit)
          const uint32_t out_bucket = new_active >> RESTBITS;
          const uint32_t oslot = dev_atomic_u32(d_out_nslots[out_bucket]).fetch_add(1u);
          if (oslot >= NSLOTS) continue;
          // Single co-located write stream so one slot == one cache-line touch. PACKED output (L1/L3/L4):
          // w0 = active_low | (parent_bucket<<RESTBITS); followers in w1..; Cantor in w[OUT_FIELDS].
          // UNPACKED output (L2): followers at the head + [parent_bucket, Cantor] tree log at the tail.
          uint32_t* o = d_out + ((size_t)out_bucket * NSLOTS + oslot) * OUT_U32;
          const uint32_t hi = s > t ? s : t, lo = s > t ? t : s;
          const uint32_t cantor = cantor_pack(hi, lo);
          constexpr bool OUT_PACKED = level_packed(R + 1);
          if constexpr (OUT_PACKED) {
            // active field's high BUCKBITS == out_bucket (redundant); store only its low RESTBITS, folded
            // with the parent_bucket. Followers (xf[2..]) keep full precision.
            o[0] = (new_active & (NRESTBINS - 1u)) | (bucket << RESTBITS);
            for (unsigned k = 1; k < OUT_FIELDS; ++k) o[k] = xf[k + 1];   // followers xf[2..]
            o[OUT_FIELDS] = cantor;
          } else {
            for (unsigned k = 0; k < OUT_FIELDS; ++k) o[k] = xf[k + 1];   // shift carried fields down
            o[OUT_TREE_OFF + 0] = bucket;
            o[OUT_TREE_OFF + 1] = cantor;
          }
        }
      }
    });
  });
}

// ---- Device tree walk: expand the node at (level, bucket, slot) into its leaf eh_indices. Iterative
// (no recursion on GPU): a fixed stack of (level,bucket,slot) frames, max depth = K+1. Leaves are read
// from level-0 slot w[5]. Returns the number of leaves written (== 2^level), or 0 on overflow.
struct EqLevels {
  const uint32_t* level[NLEVELS];            // level[L] slot storage; tree log inline at tail (tree_off)
};
inline unsigned walk_leaves(const EqLevels& lv, unsigned level, uint32_t bucket, uint32_t slot,
                            uint32_t* leaves, unsigned cap) {
  // Explicit DFS stack of frames. Depth is tiny (<=K+1) and total leaves <= 2^K = 16.
  struct Frame { unsigned level; uint32_t bucket, slot; };
  Frame stack[NLEVELS + 1];
  int sp = 0;
  stack[sp++] = {level, bucket, slot};
  unsigned n = 0;
  while (sp > 0) {
    Frame f = stack[--sp];
    if (f.level == 0) {
      if (n >= cap) return 0;
      leaves[n++] = lv.level[0][((size_t)f.bucket * NSLOTS + f.slot) * level_u32(0) + level_fields(0)];
      continue;
    }
    // Inline tree log: tail [pbucket,cantor] for unpacked levels, or w0-high/w[fields] for packed.
    const uint32_t* slotp = lv.level[f.level] + ((size_t)f.bucket * NSLOTS + f.slot) * level_u32(f.level);
    uint32_t pb, cantor; read_tree(slotp, f.level, pb, cantor);
    uint32_t a, b; cantor_unpack(cantor, a, b);
    if (sp + 2 > (int)(NLEVELS + 1)) return 0;
    stack[sp++] = {f.level - 1, pb, a};
    stack[sp++] = {f.level - 1, pb, b};
  }
  return n;
}
// 16 leaves distinct? (small N, O(n^2) pairwise.)
inline bool leaves_distinct(const uint32_t* leaves, unsigned n) {
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j = i + 1; j < n; ++j)
      if (leaves[i] == leaves[j]) return false;
  return true;
}

// ---- Final candidate scan: K=4 rounds collided fields 0..3, so each level-4 node already spans 16
// leaves with fields 0..3 == 0. It is a genuine solution iff its remaining field 4 (w[0]) is also 0.
// (~2^26 level-4 nodes, ~2 with field4==0 -> the expected ~1.88 solutions.) We additionally require its
// 16 leaves to be distinct (walk the tree on-device) to drop Wagner's index-degenerate zero-nodes.
// The level-4 node (16-leaf subtree) that solves. The device final kernel already walks the tree to
// the 16 leaf eh_indices (for the distinctness check), so it STORES them here directly -- the host then
// only copies back this tiny candidate list (a few KiB) instead of the whole ~4.2 GiB level0+tree
// arenas, which was the dominant per-solve cost (~1.5 s of D->H memcpy).
struct EqCandidate { uint32_t leaves[1u << K]; };
class EquihashFinalKernel;
static sycl::event submit_final(
  sycl::queue& q, MOM_BUNDLE_T& kb, const uint32_t* d_in, const uint32_t* d_in_nslots,
  EqLevels lv, EqCandidate* d_cand, uint32_t* d_cand_count, uint32_t cand_cap, unsigned wg
) {
  const size_t global = (size_t)NBUCKETS * wg;
  return q.submit([&](sycl::handler& h) {
    MOM_USE_BUNDLE(h, kb);
    h.parallel_for<EquihashFinalKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
      [=](sycl::nd_item<1> it) {
      const uint32_t bucket = (uint32_t)it.get_group(0);
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);
      uint32_t cnt = d_in_nslots[bucket];
      if (cnt > NSLOTS) cnt = NSLOTS;
      const uint32_t* in_base = d_in + (size_t)bucket * NSLOTS * level_u32(K);
      constexpr unsigned PROOF = 1u << K;   // 16
      // Level 4 is PACKED: w0 = field4_low | (parent_bucket<<RESTBITS). The full final field 4 ==
      // (bucket<<RESTBITS) | field4_low (its high BUCKBITS == this slot's bucket). A solution needs
      // field4 == 0, i.e. THIS bucket == 0 AND field4_low == 0.
      if (bucket != 0) return;              // field 4 high bits (== bucket) must be zero
      for (unsigned s = lid; s < cnt; s += lsz) {
        const uint32_t* a = in_base + (size_t)s * level_u32(K);
        if ((a[0] & (NRESTBINS - 1u)) != 0) continue;  // field 4 low RESTBITS must be zero too
        uint32_t leaves[PROOF];
        unsigned n = walk_leaves(lv, K, bucket, s, leaves, PROOF);
        if (n != PROOF) continue;
        if (!leaves_distinct(leaves, PROOF)) continue;
        const uint32_t idx = dev_atomic_u32(d_cand_count[0]).fetch_add(1u);
        if (idx >= cand_cap) continue;
        for (unsigned i = 0; i < PROOF; ++i) d_cand[idx].leaves[i] = leaves[i];
      }
    });
  });
}

// ===========================================================================================
// Per-device state: queue + bundle + USM arenas (modeled on FishState). The ~5.9 GiB Wagner arenas
// are allocated lazily on the first mining solve to prove headroom; the M1 test path only needs the
// small base-state + rows buffers.
// ===========================================================================================
class EquihashState {
public:
  sycl::device device; sycl::queue queue; std::unique_ptr<MOM_BUNDLE_T> bundle; bool shared_io;
  // M1 I/O
  uint64_t* base_h = nullptr;     // 8 words
  uint8_t*  base_pending = nullptr;// 12 bytes
  uint8_t*  rows = nullptr;       // EQUIHASH_TEST_ROWS * HASH_LENGTH
  // Wagner arenas (M2/M3). Five bucket levels (0..4) with the per-survivor tree log CO-LOCATED inline in
  // each slot (no separate tree arena), per-level nslots. Each slot at level L = level_u32(L) u32
  // (6,5,5,3,2 words incl. inline tree, active-field-low packed at L1/L3/L4); each level = NBUCKETS*NSLOTS
  // slots. Total (6+5+5+3+2)*NBUCKETS*NSLOTS*4 ~= 5.82 GiB (was 7.03 GiB at the unpacked {6,6,5,5,3}).
  // Fits a >=8 GiB card with extra headroom. cand_* holds final candidates.
  uint32_t* level[NLEVELS] = {};       // level[L]: NBUCKETS*NSLOTS*level_u32(L) u32 (tree log inline at tail)
  uint32_t* nslots         = nullptr;  // NLEVELS * NBUCKETS atomic counters
  EqCandidate* cand        = nullptr;  // final candidate list (parent bucket + cantor at level 4)
  uint32_t* cand_count     = nullptr;  // shared: atomic candidate counter
  static constexpr uint32_t CAND_CAP = 4096;
  bool arenas_built = false; std::mutex mutex;

  explicit EquihashState(const std::string& dev_str)
    : device(get_dev(dev_str)), queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !device.has(sycl::aspect::usm_device_allocations)) {
    if (!device.has(sycl::aspect::usm_shared_allocations) ||
        (!device.is_cpu() && !device.has(sycl::aspect::usm_device_allocations)))
      throw std::string("equihash125_4 SYCL device does not support required allocations");
    bundle = std::make_unique<MOM_BUNDLE_T>(MOM_GET_EXEC_BUNDLE(queue.get_context()));
  }
  ~EquihashState() { queue.wait_and_throw(); free_all(); }
  template <typename T> T* alloc(size_t n) {
    return shared_io ? sycl::malloc_shared<T>(n, queue) : sycl::malloc_device<T>(n, queue);
  }
  void free_ptr(auto*& p) { if (p) sycl::free(p, queue); p = nullptr; }
  void free_all() {
    free_ptr(base_h); free_ptr(base_pending); free_ptr(rows);
    for (unsigned L = 0; L < NLEVELS; ++L) free_ptr(level[L]);
    free_ptr(nslots); free_ptr(cand); free_ptr(cand_count);
    arenas_built = false;
  }
  void ensure_io() {
    if (base_h && base_pending && rows) return;
    free_ptr(base_h); free_ptr(base_pending); free_ptr(rows);
    // Shared (host-accessible) -- these are tiny and both the host (base-state write, M1 row readback)
    // and the device touch them, so keep them shared even on discrete GPUs (cf. FishState::result).
    base_h       = sycl::malloc_shared<uint64_t>(8, queue);
    base_pending = sycl::malloc_shared<uint8_t>(12, queue);
    rows         = sycl::malloc_shared<uint8_t>((size_t)EQUIHASH_TEST_ROWS * HASH_LENGTH, queue);
    if (!base_h || !base_pending || !rows) throw std::string("Can't allocate equihash125_4 I/O buffers");
  }
  // Allocate the Wagner bucket arenas: 5 levels of slot storage (tree log inline) + per-level counters.
  void ensure_arenas(bool log) {
    if (arenas_built) return;
    const uint64_t t0 = now_ms();
    free_all_arenas();
    const size_t slots_per_level = (size_t)NBUCKETS * NSLOTS;
    size_t total = 0;
    for (unsigned L = 0; L < NLEVELS; ++L) {
      const size_t lvl_u32 = slots_per_level * level_u32(L);   // co-located width: 6,5,5,3,2 words/slot
      level[L] = alloc<uint32_t>(lvl_u32); total += lvl_u32 * 4;
      if (!level[L]) throw std::string("Can't allocate equihash125_4 level arena");
    }
    nslots     = alloc<uint32_t>((size_t)NLEVELS * NBUCKETS); total += (size_t)NLEVELS * NBUCKETS * 4;
    cand       = alloc<EqCandidate>(CAND_CAP);
    cand_count = sycl::malloc_shared<uint32_t>(1, queue);     // host reads the candidate count
    if (!nslots || !cand || !cand_count)
      throw std::string("Can't allocate equihash125_4 counters/candidates");
    // Touch the counter array (lazily mapped USM device allocs surface OOM here, not mid-kernel).
    sycl_wait_and_throw(queue.memset(nslots, 0, (size_t)NLEVELS * NBUCKETS * sizeof(uint32_t)), device);
    arenas_built = true;
    if (log) { std::fprintf(stderr, "equihash125_4 Wagner arenas (~%.1f GiB) allocated (%llu ms)\n",
      (double)total / (1024.0 * 1024.0 * 1024.0), (unsigned long long)(now_ms() - t0)); std::fflush(stderr); }
  }
  void free_all_arenas() {
    for (unsigned L = 0; L < NLEVELS; ++L) free_ptr(level[L]);
    free_ptr(nslots); free_ptr(cand); free_ptr(cand_count);
  }
  static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }
};

static EquihashState& equihash_state(const std::string& dev_str) {
  static std::mutex m; static std::map<std::string, std::unique_ptr<EquihashState>> states;
  std::lock_guard<std::mutex> lock(m);
  auto& s = states[dev_str]; if (!s) s = std::make_unique<EquihashState>(dev_str); return *s;
}

// ===========================================================================================
// Host-side recovery: each final candidate already carries its 16 leaf eh_indices (the device final
// kernel walked the tree on-device for the distinctness check and stored them), so the host only
// canonicalises (orderindices: the Zcash IndicesBefore subtree ordering), rejects DistinctIndices
// violations, and CompressArray(26)s the 16 indices into the 52-byte solution. No arena copyback.
// ===========================================================================================

// orderindices: the Zcash GetIndices canonical ordering. The recovered 2^K leaves come out in tree
// order; IsValidSolution's IndicesBefore rule requires, at every internal node, that the subtree whose
// minimum (== first) index is smaller comes first. Re-derive that ordering bottom-up by swapping each
// sibling pair (and the blocks they head) so the smaller-leading half precedes. Matches fluxd's
// "if (indices[0] < indicesRight[0]) ... else swap".
static void order_indices(std::vector<uint32_t>& idx) {
  const size_t n = idx.size();
  for (size_t span = 1; span < n; span *= 2) {
    for (size_t i = 0; i + 2 * span <= n; i += 2 * span) {
      if (idx[i + span] < idx[i]) {  // right half leads with a smaller index -> it must come first
        for (size_t k = 0; k < span; ++k) std::swap(idx[i + k], idx[i + span + k]);
      }
    }
  }
}

// CompressArray(26,0): pack 16 indices (26 bits each, big-endian) into 52 bytes. Exact inverse of the
// reference expandArray used by getIndicesFromMinimal.
static void compress_indices(const std::vector<uint32_t>& idx, uint8_t out[52]) {
  constexpr unsigned BIT_LEN = COLLISION_BIT_LENGTH + 1;   // 26
  constexpr unsigned IN_WIDTH = ((BIT_LEN + 7) >> 3);      // 4 (each index expanded as 4 BE bytes)
  // Build the expanded byte stream (16 * 4 = 64 bytes, big-endian 26-bit fields) then compress.
  std::vector<uint8_t> in(idx.size() * IN_WIDTH, 0);
  for (size_t i = 0; i < idx.size(); ++i) {
    in[i * IN_WIDTH + 0] = (uint8_t)((idx[i] >> 24) & 0xFF);
    in[i * IN_WIDTH + 1] = (uint8_t)((idx[i] >> 16) & 0xFF);
    in[i * IN_WIDTH + 2] = (uint8_t)((idx[i] >> 8) & 0xFF);
    in[i * IN_WIDTH + 3] = (uint8_t)(idx[i] & 0xFF);
  }
  const size_t out_len = idx.size() * BIT_LEN / 8;        // 52
  uint32_t acc_value = 0; unsigned acc_bits = 0; size_t j = 0;
  const uint32_t bit_mask = ((uint32_t)1 << BIT_LEN) - 1u;
  for (size_t i = 0; i < in.size(); ++i) {
    // CompressArray (Zcash util.cpp): read bit_len bits from each IN_WIDTH input element, low byte_pad=0.
    if ((i % IN_WIDTH) == 0) {
      // begin a new input element: take its BIT_LEN low bits
      uint32_t v = ((uint32_t)in[i] << 24) | ((uint32_t)in[i + 1] << 16) |
                   ((uint32_t)in[i + 2] << 8) | (uint32_t)in[i + 3];
      v &= bit_mask;
      acc_value = (acc_value << BIT_LEN) | v;
      acc_bits += BIT_LEN;
      while (acc_bits >= 8 && j < out_len) {
        acc_bits -= 8;
        out[j++] = (uint8_t)((acc_value >> acc_bits) & 0xFF);
      }
    }
  }
}

// ===========================================================================================
// Host-side SHA-256 (FIPS 180-4) + the Flux block-hash/PoW target test. The PoW hash is
// dSHA256(header(140) || compactSize(0x34) || solution(52)) = a 193-byte preimage. The 32-byte
// double-SHA output is read LITTLE-ENDIAN as a 256-bit integer and compared <= the 256-bit target.
// (fluxd pow.cpp: UintToArith256(hash) <= bnTarget; mom stores the target big-endian in m_target_bin,
// so the LE-read hash <= BE-stored target is a plain big-endian byte compare of the reversed hash.)
// Self-contained like the BLAKE2b above -- this rare host path runs only for the ~1.88 candidates/solve.
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
  // Pad: msg || 0x80 || 0x00.. || 64-bit big-endian bit length, to a multiple of 64.
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

// dSHA256 of (header(140) || 0x34 || solution(52)) -> the 32-byte internal (LE) block hash.
static void flux_block_hash(const uint8_t* header140, const uint8_t solution[52], uint8_t out[32]) {
  uint8_t preimage[HEADER_LEN + 1 + 52];
  std::memcpy(preimage, header140, HEADER_LEN);
  preimage[HEADER_LEN] = 0x34;                     // compactSize(52)
  std::memcpy(preimage + HEADER_LEN + 1, solution, 52);
  uint8_t once[32];
  sha256(preimage, sizeof(preimage), once);
  sha256(once, 32, out);
}

// PoW target test: hash read little-endian <= target (m_target_bin is a 256-bit BIG-endian array).
// Equivalent big-endian byte compare of the reversed hash vs the target (cf. fishhash meets_target_be).
static bool flux_meets_target(const uint8_t hash_le[32], const uint8_t* target_be) {
  for (int i = 0; i < 32; ++i) {
    const uint8_t hb = hash_le[31 - i];             // most-significant byte of the LE-read hash first
    if (hb != target_be[i]) return hb < target_be[i];
  }
  return true;   // exactly equal still meets target
}

} // namespace mom_equihash125_4

using namespace mom_equihash125_4;

int equihash125_4(
  const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size,
  uint8_t* const solution_out, uint64_t* const /*pnonce*/, const uint8_t* const target,
  const unsigned /*intensity*/, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < HEADER_LEN) throw std::string("Bad equihash125_4 input length");
  EquihashState& state = equihash_state(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_io();

  // Host: personalized "ZelProof" base state over all 140 header bytes (base_h/base_pending are shared).
  const BaseState bs = make_base_state(input);
  std::memcpy(state.base_h, bs.h, 8 * sizeof(uint64_t));
  std::memcpy(state.base_pending, bs.pending, 12);

  // is_test path: the default runs the gen-kernel cross-check (gen-rows dump). Setting
  // MOM_EQUIHASH_SOLVE switches it to the full Wagner solve so the standalone checker can assert the
  // block-400000 solution is found. (Mining, !is_test, always runs the solver.)
  const bool solve_in_test = is_test && std::getenv("MOM_EQUIHASH_SOLVE") != nullptr;

  if (is_test && !solve_in_test) {
    // M1: run the naive gen kernel, then the optimized sub-group-scan kernel, and confirm they match
    // on-device before handing the rows back. The standalone checker diffs them vs the JS oracle.
    const size_t rows_bytes = (size_t)EQUIHASH_TEST_ROWS * HASH_LENGTH;
    std::memset(state.rows, 0, rows_bytes);
    sycl_wait_and_throw(submit_gen_test<true>(state.queue, *state.bundle, state.base_h, state.base_pending, state.rows), state.device);
    std::vector<uint8_t> naive(state.rows, state.rows + rows_bytes);

    std::memset(state.rows, 0, rows_bytes);
    sycl_wait_and_throw(submit_gen_test<false>(state.queue, *state.bundle, state.base_h, state.base_pending, state.rows), state.device);
    std::vector<uint8_t> scan(state.rows, state.rows + rows_bytes);

    const bool match = (naive == scan);
    if (!match) {
      // Report the first differing entry so the checker / log pinpoints the scan regression.
      for (size_t e = 0; e < EQUIHASH_TEST_ROWS; ++e)
        if (std::memcmp(naive.data() + e * HASH_LENGTH, scan.data() + e * HASH_LENGTH, HASH_LENGTH)) {
          std::fprintf(stderr, "equihash125_4 M1: optimized scan != naive at entry %zu\n", e);
          break;
        }
    }
    // Hand back the NAIVE rows (the reference-of-record); the checker also asserts naive==scan above.
    std::memcpy(solution_out, naive.data(), (size_t)EQUIHASH_TEST_ROWS * HASH_LENGTH);
    (void)is_benchmark;
    return match ? 1 : 0;
  }

  // ---- M2/M3 full Wagner solve over the gen output for this exact 140-byte header. -----------------
  state.ensure_arenas(!is_benchmark);
  std::memset(solution_out, 0, SMALL_BLOB_SOL_LEN);
  const bool log = !is_benchmark;
  // Per-phase profiler: MOM_EQUIHASH_PROF prints a gen-fill / per-round / final-scan / host-recovery
  // ms breakdown even under bench (where `log` is off). It inserts an extra barrier + per-round host
  // waits, so it is a SEPARATE gate from MOM_EQUIHASH_PERF (the Sol/s logger) -- enabling it perturbs
  // throughput slightly, so leave it off when measuring Sol/s.
  static const bool prof = std::getenv("MOM_EQUIHASH_PROF") != nullptr;
  const uint64_t t_start = EquihashState::now_ms();
  sycl::queue& q = state.queue;

  // Reset all per-level counters + the candidate counter for this nonce.
  sycl_wait_and_throw(q.memset(state.nslots, 0, (size_t)NLEVELS * NBUCKETS * sizeof(uint32_t)), state.device);
  *state.cand_count = 0;

  // Workgroup size for the round kernels (one WG per bucket, striping its slots). A bucket holds ~2048
  // slots, so a larger WG cuts the per-thread serial slot count. With the SLM record cache the round is
  // ~431ms at WG=512 on the B580 (vs ~487ms at 768 -- 512 balances threads against the 59KB SLM/WG best).
  // Override with MOM_EQUIHASH_WORKGROUP.
  unsigned wg = 512;
  { unsigned long parsed = 0; if (mom_parse_env_ulong("MOM_EQUIHASH_WORKGROUP", parsed) && parsed >= 16)
      wg = (unsigned)std::min<unsigned long>(parsed, 1024); }

  // Level 0: gen-fill 2^26 entries into round-0 buckets (field-0 bucketing).
  const uint64_t t_gen0 = EquihashState::now_ms();
  sycl_wait_and_throw(submit_gen_fill(q, *state.bundle, state.base_h, state.base_pending,
                                      state.level[0], state.nslots + 0u * NBUCKETS), state.device);

  if (prof) sycl_wait_and_throw(q.ext_oneapi_submit_barrier(), state.device);
  // Rounds 0..3: collide field R over (bucket, RESTBITS), XOR -> level R+1.
  const uint64_t t_r0 = EquihashState::now_ms();
  uint64_t t_round[4] = {0,0,0,0};
  sycl_wait_and_throw(submit_round<0>(q, *state.bundle, state.level[0], state.level[1],
    state.nslots + 0u * NBUCKETS, state.nslots + 1u * NBUCKETS, wg), state.device);
  if (prof) t_round[0] = EquihashState::now_ms();
  sycl_wait_and_throw(submit_round<1>(q, *state.bundle, state.level[1], state.level[2],
    state.nslots + 1u * NBUCKETS, state.nslots + 2u * NBUCKETS, wg), state.device);
  if (prof) t_round[1] = EquihashState::now_ms();
  sycl_wait_and_throw(submit_round<2>(q, *state.bundle, state.level[2], state.level[3],
    state.nslots + 2u * NBUCKETS, state.nslots + 3u * NBUCKETS, wg), state.device);
  if (prof) t_round[2] = EquihashState::now_ms();
  sycl_wait_and_throw(submit_round<3>(q, *state.bundle, state.level[3], state.level[4],
    state.nslots + 3u * NBUCKETS, state.nslots + 4u * NBUCKETS, wg), state.device);
  if (prof) t_round[3] = EquihashState::now_ms();
  const uint64_t t_rounds_end = EquihashState::now_ms();

  // Final: each level-4 node with field4==0 AND 16 distinct leaves is a solution candidate.
  EqLevels lv;
  for (unsigned L = 0; L < NLEVELS; ++L) lv.level[L] = state.level[L];
  sycl_wait_and_throw(submit_final(q, *state.bundle, state.level[4], state.nslots + 4u * NBUCKETS,
                                   lv, state.cand, state.cand_count, EquihashState::CAND_CAP, wg), state.device);
  const uint64_t t_rounds = EquihashState::now_ms();
  if (prof) {
    std::fprintf(stderr,
      "[eq-prof] gen-fill %llums | r0 %llu r1 %llu r2 %llu r3 %llu (rounds %llums) | final-scan %llums\n",
      (unsigned long long)(t_r0 - t_gen0),
      (unsigned long long)(t_round[0] - t_r0), (unsigned long long)(t_round[1] - t_round[0]),
      (unsigned long long)(t_round[2] - t_round[1]), (unsigned long long)(t_round[3] - t_round[2]),
      (unsigned long long)(t_rounds_end - t_r0), (unsigned long long)(t_rounds - t_rounds_end));
    std::fflush(stderr);
  }

  // ---- Pull the level/tree arenas + candidate list to host for recovery. ----
  const uint64_t t_recov0 = EquihashState::now_ms();
  uint32_t cand_count = *state.cand_count;
  const uint32_t cand_n = std::min<uint32_t>(cand_count, EquihashState::CAND_CAP);

  // Per-level survivor counts (sum of nslots) for the M2/drop-rate report.
  std::vector<uint32_t> ns_host((size_t)NLEVELS * NBUCKETS);
  sycl_wait_and_throw(q.memcpy(ns_host.data(), state.nslots, ns_host.size() * sizeof(uint32_t)), state.device);
  if (log) {
    for (unsigned L = 0; L < NLEVELS; ++L) {
      uint64_t sum = 0, overflow = 0;
      for (unsigned bkt = 0; bkt < NBUCKETS; ++bkt) {
        const uint32_t c = ns_host[(size_t)L * NBUCKETS + bkt];
        sum += std::min<uint32_t>(c, NSLOTS);
        if (c > NSLOTS) overflow += c - NSLOTS;
      }
      std::fprintf(stderr, "equihash125_4 level %u survivors=%llu dropped=%llu (%.4f%%)\n",
        L, (unsigned long long)sum, (unsigned long long)overflow,
        100.0 * (double)overflow / (double)(sum + overflow ? sum + overflow : 1));
    }
    std::fprintf(stderr, "equihash125_4 candidates=%u (gen %llums, rounds %llums)\n", cand_count,
      (unsigned long long)(t_r0 - t_gen0), (unsigned long long)(t_rounds - t_r0));
    std::fflush(stderr);
  }

  int n_solutions = 0;
  size_t distinct_count = 0;   // distinct valid Equihash proofs found this solve (pre-target-filter)
  if (cand_n) {
    // The device final kernel already walked each candidate's tree to its 16 leaf eh_indices, so we
    // copy back ONLY the tiny candidate list (a few KiB) -- no whole-arena D->H copyback.
    std::vector<EqCandidate> cand_h(cand_n);
    sycl_wait_and_throw(q.memcpy(cand_h.data(), state.cand, cand_n * sizeof(EqCandidate)), state.device);

    // De-dup distinct solutions (the same proof can surface via several candidates).
    std::vector<std::array<uint8_t, 52>> seen;
    for (uint32_t ci = 0; ci < cand_n; ++ci) {
      // Each candidate is one level-4 node (16-leaf subtree) whose 125-bit XOR is zero; its 16 leaf
      // eh_indices were collected on-device.
      std::vector<uint32_t> leaves(cand_h[ci].leaves, cand_h[ci].leaves + (1u << K));
      if (leaves.size() != (size_t)(1u << K)) continue;   // PROOFSIZE = 16

      // DistinctIndices: all 16 leaves must be unique.
      std::vector<uint32_t> sorted = leaves;
      std::sort(sorted.begin(), sorted.end());
      bool distinct = true;
      for (size_t i = 1; i < sorted.size(); ++i) if (sorted[i] == sorted[i - 1]) { distinct = false; break; }
      if (!distinct) continue;

      // Canonical IndicesBefore ordering, then compress to 52 bytes.
      order_indices(leaves);
      std::array<uint8_t, 52> sol{};
      compress_indices(leaves, sol.data());
      bool dup = false;
      for (const auto& s : seen) if (s == sol) { dup = true; break; }
      if (dup) continue;
      seen.push_back(sol);
    }

    distinct_count = seen.size();
    // M5 mining path: each distinct solution is a valid Equihash proof, but only those whose Flux
    // block hash dSHA256(header || 0x34 || solution) meets the 256-bit PoW target are submittable.
    // The is_test SOLVE path (keystone / offline vector) keeps ALL distinct solutions so the checker
    // can assert the known block-400000 solution is present regardless of the (all-0xFF) target.
    std::vector<std::array<uint8_t, 52>> emit;
    emit.reserve(seen.size());
    for (const auto& sol : seen) {
      if (is_test) { emit.push_back(sol); continue; }   // SOLVE-test: emit every distinct solution
      uint8_t blockhash[32];
      flux_block_hash(input, sol.data(), blockhash);
      if (flux_meets_target(blockhash, target)) emit.push_back(sol);
    }

    // Sort the kept solutions lexicographically so the out-of-band buffer is deterministic regardless
    // of the non-deterministic candidate-collection order (atomic fetch_add). This makes the offline
    // vector's full-buffer comparison stable; the pool is indifferent to ordering.
    std::sort(emit.begin(), emit.end());

    // Hand the kept solutions back: solution_out is the 5120-byte small-blob buffer. Layout:
    // [0] = solution count (u8, capped), then count * 52 bytes.
    const unsigned cap = (SMALL_BLOB_SOL_LEN - 1) / 52;   // up to 98 solutions
    n_solutions = (int)std::min<size_t>(emit.size(), cap);
    solution_out[0] = (uint8_t)n_solutions;
    for (int i = 0; i < n_solutions; ++i)
      std::memcpy(solution_out + 1 + (size_t)i * 52, emit[i].data(), 52);
    if (log) {
      std::fprintf(stderr, "equihash125_4 %s solutions=%d (of %zu distinct):\n",
        is_test ? "distinct" : "target-passing", n_solutions, seen.size());
      for (int i = 0; i < n_solutions; ++i) {
        std::fprintf(stderr, "  ");
        for (int b = 0; b < 52; ++b) std::fprintf(stderr, "%02x", emit[i][b]);
        std::fprintf(stderr, "\n");
      }
      std::fflush(stderr);
    }
  } else {
    solution_out[0] = 0;
  }

  // MOM_EQUIHASH_PERF: Sol/s logger (mirrors MOM_FISHHASH_PERF). Accumulates the number of distinct
  // valid proofs found (not nonces, and BEFORE the target filter so bench/zero-target still measures
  // throughput) over a >=2s window; the solver averages ~1.88 solutions per nonce.
  static const bool perf_log = std::getenv("MOM_EQUIHASH_PERF") != nullptr;
  if (perf_log && !is_test) {
    static uint64_t acc_sols = 0; static uint64_t acc_solves = 0;
    static auto t0 = std::chrono::steady_clock::now();
    acc_sols += (uint64_t)distinct_count;
    ++acc_solves;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec >= 2.0) {
      std::fprintf(stderr, "[equihash125_4] %.2f Sol/s, %.2f solve/s (%s)\n",
        (double)acc_sols / sec, (double)acc_solves / sec, dev_str.c_str());
      std::fflush(stderr);
      acc_sols = 0; acc_solves = 0; t0 = std::chrono::steady_clock::now();
    }
  }

  if (prof) {
    std::fprintf(stderr, "[eq-prof] host-recovery %llums (cand=%u) | total %llums\n",
      (unsigned long long)(EquihashState::now_ms() - t_recov0), cand_count,
      (unsigned long long)(EquihashState::now_ms() - t_start));
    std::fflush(stderr);
  }
  if (log) { std::fprintf(stderr, "equihash125_4 solve done (%llu ms, %d target-passing, %zu distinct)\n",
    (unsigned long long)(EquihashState::now_ms() - t_start), n_solutions, distinct_count); std::fflush(stderr); }

  // Return semantics depend on the caller:
  //  * is_test SOLVE path (the keystone / offline vector): the core's is_test handler dumps the whole
  //    out-of-band buffer when this returns 1, so report 1-on-any-solution (the payload count is in
  //    solution_out[0]).
  //  * bench: report the raw solver throughput (distinct proofs found this solve, target-independent)
  //    so the core's m_hash_count accounting yields Sol/s.
  //  * mining: report the target-passing payload count so the core drives send_result.
  if (is_test) return n_solutions > 0 ? 1 : 0;
  return is_benchmark ? (int)distinct_count : n_solutions;
}
