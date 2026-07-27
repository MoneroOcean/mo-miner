// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
//
// ZelHash (Equihash 125,4) GPU solver -- Wagner bucket-collision (Tromp/djezo lineage).
//
// Pipeline:
//   gen    : 2^26 entries from a personalized "ZelProof" blake2b over the 140-byte header + the
//            ZelHash "twist" (16-aligned block prefix-sum of stock hashes), masked, split into 4
//            sub-elements, ExpandArray(25) -> 25-bit collision fields.
//   round  : (M2+) bucket by high BUCKBITS, collide low RESTBITS, XOR pairs -> next round.
//   recover: (M3+) walk tree -> 16 leaf indices -> CompressArray(26) -> 52-byte solution.
//
// Mining runs the complete solve path: gen-fill, four collision rounds, fused final filtering, recovery, and target
// filtering. The default is_test path still uses the cheaper gen-kernel cross-check; set
// MOM_ZELHASH_SOLVE to run the full block-400000 proof recovery vector.

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

namespace mom_zelhash {

// CUDA needs aligned stores on both compilers; HIP is faster with the denser scalar record stream.
#if defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)
#define MOM_EQ_ALIGNED_GPU_RECORDS 1
#endif
// DPC++ lowers a coalesced leaf->position map much better than four random reverse-map stores.
// The rare recovery pass resolves final leaves with one linear inversion scan.
#if defined(MOM_SYCL_HAS_CUDA)
#define MOM_EQ_FORWARD_MAP 1
#endif

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

constexpr unsigned ceil_div_u64(uint64_t value, unsigned divisor) {
  return static_cast<unsigned>((value + divisor - 1u) / divisor);
}
constexpr unsigned ceil_sqrt(unsigned value) {
  unsigned root = 0;
  while (static_cast<uint64_t>(root) * root < value) ++root;
  return root;
}
constexpr unsigned round_up(unsigned value, unsigned alignment) {
  return (value + alignment - 1u) / alignment * alignment;
}
constexpr unsigned round_down(unsigned value, unsigned alignment) {
  return value / alignment * alignment;
}
constexpr unsigned bucket_capacity(unsigned buckets, unsigned deviations, unsigned alignment,
                                   bool round_capacity_down = false) {
  // Bucket occupancy is binomial. Size the arena from its mean plus a backend-appropriate number
  // of standard deviations, then align the physical stride for vector memory transactions. This
  // derives the old hand-tuned capacities while scaling automatically if the bucket geometry changes.
  const unsigned mean = ceil_div_u64(NUM_ENTRIES, buckets);
  const unsigned limit = mean + deviations * ceil_sqrt(mean);
  return round_capacity_down ? round_down(limit, alignment) : round_up(limit, alignment);
}

// ---- Wagner solver memory plan. Intel oneAPI gains from 16384 super-buckets and 1024-lane groups:
// its ~29-KiB scratch arena amortizes each bucket with fewer register-held pairs per lane. CUDA/HIP
// instead lose materially with that geometry and retain 8192 buckets, 1024 lanes, and a ~53-KiB
// arena. AdaptiveCpp CUDA cannot opt a Windows kernel into >48 KiB dynamic shared memory, so its
// lower-priority fallback uses 32768 smaller buckets and the same common collision code with a
// ~16-KiB scratch arena.
// Four materialized levels hold the gen output and round-0..2 survivors. Round 3 filters its final
// field in-place and emits only zero pairs, so the otherwise 0.27--0.53-GiB level-4 stream is absent.
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
constexpr unsigned RESTBITS  = 10;
constexpr unsigned BUCKBITS  = COLLISION_BIT_LENGTH - RESTBITS;     // 15
constexpr unsigned NBUCKETS  = 1u << BUCKBITS;                      // 32768
constexpr unsigned NRESTBINS = 1u << RESTBITS;                      // 1024 exact bins per bucket
constexpr unsigned MAX_SLOTS = bucket_capacity(NBUCKETS, 1u, 64u);
constexpr unsigned ROUND_WG   = 1024;
#elif !defined(MOM_SYCL_HAS_CUDA) && !defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
constexpr unsigned RESTBITS  = 11;
constexpr unsigned BUCKBITS  = COLLISION_BIT_LENGTH - RESTBITS;     // 14
constexpr unsigned NBUCKETS  = 1u << BUCKBITS;                      // 16384
constexpr unsigned NRESTBINS = 1u << RESTBITS;                      // 2048 exact bins per super-bucket
constexpr unsigned MAX_SLOTS = bucket_capacity(NBUCKETS, 6u, 16u);
constexpr unsigned ROUND_WG  = 1024;
#elif defined(MOM_SYCL_HAS_CUDA)
// Eight fixed records per 1024-lane work-group make the collision input fully unrollable. A mildly
// larger, non-power-of-two bucket count keeps the same total capacity while reducing the per-bucket
// cap from 8704 to exactly 8192. The quotient is the exact-bin id and the remainder selects a bucket.
constexpr unsigned RESTBITS  = 12;                                  // packed metadata width
constexpr unsigned NBUCKETS  = 8759;
constexpr unsigned NRESTBINS = 3831;                                // ceil(2^25 / 8759)
constexpr unsigned MAX_SLOTS = bucket_capacity(NBUCKETS, 6u, 16u);
constexpr unsigned ROUND_WG  = 1024;
#else
constexpr unsigned RESTBITS  = 12;
constexpr unsigned BUCKBITS  = COLLISION_BIT_LENGTH - RESTBITS;     // 13
constexpr unsigned NBUCKETS  = 1u << BUCKBITS;                      // 8192
constexpr unsigned NRESTBINS = 1u << RESTBITS;                      // 4096 exact bins per super-bucket
#if defined(MOM_SYCL_HAS_HIP)
// Three standard deviations preserve virtually every generated row while avoiding the large,
// power-of-two-related strides that concentrate progressive records in too few cache sets.
constexpr unsigned MAX_SLOTS = bucket_capacity(NBUCKETS, 3u, 16u, true);
#else
constexpr unsigned MAX_SLOTS = bucket_capacity(NBUCKETS, 5u, 64u);
#endif
constexpr unsigned ROUND_WG  = 1024;
#endif
#if !defined(MOM_SYCL_HAS_CUDA) && !defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
// Intel Xe benefits from walking exact-bin chains directly in SLM. Splitting the two early rounds
// keeps their larger follower records below the device's local-memory limit; later rounds fit whole.
constexpr bool ROUND_DIRECT_CHAINS = true;
#else
constexpr bool ROUND_DIRECT_CHAINS = false;
#endif
constexpr unsigned NLEVELS   = K + 1;                              // five progressive field counts
constexpr unsigned NSTORED_LEVELS = K;                             // final zero pairs replace level 4
static_assert(MAX_SLOTS < (1u << 16));
inline uint32_t stored_rest(uint32_t word) { return word & ((1u << RESTBITS) - 1u); }
#if defined(MOM_SYCL_HAS_CUDA)
inline uint32_t field_rest(uint32_t field) { return (field & 0x1ffffffu) / NBUCKETS; }
inline uint32_t field_bucket(uint32_t field) { return (field & 0x1ffffffu) % NBUCKETS; }
inline uint32_t field_from_parts(uint32_t bucket, uint32_t rest) { return rest * NBUCKETS + bucket; }
#else
inline uint32_t field_rest(uint32_t field) { return field & (NRESTBINS - 1u); }
inline uint32_t field_bucket(uint32_t field) { return field >> RESTBITS; }
inline uint32_t field_from_parts(uint32_t bucket, uint32_t rest) { return (bucket << RESTBITS) | rest; }
#endif
template <unsigned R, bool GPU_COMPACT> inline uint32_t collision_rest(const uint32_t* g) {
#if defined(MOM_SYCL_HAS_CUDA)
  if constexpr (GPU_COMPACT && R == 0u)
    return (g[0] >> 25) | ((g[1] >> 25) << 7);
#endif
  return R == 0u ? field_rest(g[0]) : stored_rest(g[0]);
}

// Progressive records carry one fewer 25-bit collision field after each round. The common level 0
// densely packs five fields plus its 26-bit leaf index into five words. CUDA stores the five fields in
// one aligned vector and keeps the recovery-only index in a separate stream. At L>=1 the active field's
// high bits are the physical bucket and need not be stored; w0 combines active_low with the previous
// (parent) bucket, followed by the remaining full-width fields. Parent pair IDs are reconstructed only
// for the handful of final zero nodes, avoiding both the old multi-GiB tree stream and 26 tree bits in
// every survivor.
// Intel and HIP use the natural {5,4,3,2,1} shape. Intel's heavy level-1 scatter must be emitted as
// one explicit 16-byte vector store: four scalar stores at the same stride are pathological on Xe.
// CUDA keeps aligned 16-byte records through level 2 and the natural 8-byte level 3 used by the
// final collision round. Physical 16-slot tiling below retains aligned transactions without the
// cache-alias penalty previously seen with dense late records in bucket-major order.
constexpr unsigned level_fields(unsigned L) { return NLEVELS - L; }            // 5,4,3,2,1
#if !defined(MOM_SYCL_HAS_CUDA) && !defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
constexpr unsigned LEVEL_U32_TBL[NLEVELS]    = { 5, 4, 3, 2, 1 };
#else
constexpr unsigned LEVEL_U32_TBL[NLEVELS]    = { 5, 5, 3, 2, 1 };
#endif
// The combined DPC++ binary carries both compile-time shapes and selects by the queue's actual backend.
constexpr unsigned level_u32(unsigned L, bool gpu_compact = false) {
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
#if defined(MOM_SYCL_HAS_CUDA)
  constexpr unsigned CUDA_U32_TBL[NLEVELS] = { 4, 4, 4, 2, 2 };
#else
  constexpr unsigned CUDA_U32_TBL[NLEVELS] = { 4, 4, 4, 4, 2 };
#endif
  if (gpu_compact) return CUDA_U32_TBL[L];
#endif
  return gpu_compact && L == 1 ? 4u : LEVEL_U32_TBL[L];
}
#if defined(MOM_SYCL_HAS_CUDA)
// CUDA record levels use 16-slot tiles rather than storing one complete bucket at a time:
//
//   [slot / 16][bucket][slot % 16]
//
// Adjacent lanes still issue two coalesced 16-record segments, while different warps spread their
// random bucket stores across fewer cache sectors. Logical bucket/slot IDs remain unchanged, so the
// common collision and recovery algorithms only need this physical-address translation.
inline size_t cuda_record_pos(uint32_t bucket, uint32_t slot) {
  return (((size_t)(slot >> 4u) * NBUCKETS + bucket) << 4u) + (slot & 15u);
}
#endif
// Common dense level-0 leaf: five 25-bit fields + one 26-bit Equihash index = 151 bits in five words.
// f0 stays in w0[24:0], so bucket/rest-bin extraction remains a single masked load. A 20-byte stride is
// also the measured Intel/HIP-safe scatter shape (unlike CUDA's separate aligned hash/index streams).
inline void dense_l0_store(uint32_t* o, const uint32_t* f, uint32_t eh_index) {
  o[0] = f[0] | ((f[1] & 0x7fu) << 25);
  o[1] = (f[1] >> 7) | ((f[2] & 0x3fffu) << 18);
  o[2] = (f[2] >> 14) | ((f[3] & 0x1fffffu) << 11);
  o[3] = (f[3] >> 21) | ((f[4] & 0x01ffffffu) << 4) | ((eh_index & 0x7u) << 29);
  o[4] = eh_index >> 3;
}
template <unsigned R, bool GPU_COMPACT>
inline uint32_t load_follower(const uint32_t* g, unsigned k) {
  if constexpr (R == 0) {
#if defined(MOM_SYCL_HAS_CUDA)
    if constexpr (GPU_COMPACT) return g[k] & 0x1ffffffu;
#endif
    switch (k) {
      case 0: return ((g[0] >> 25) & 0x7fu) | ((g[1] & 0x3ffffu) << 7);
      case 1: return (g[1] >> 18) | ((g[2] & 0x7ffu) << 14);
      case 2: return (g[2] >> 11) | ((g[3] & 0xfu) << 21);
      default: return (g[3] >> 4) & 0x01ffffffu;
    }
  } else {
    return g[k + 1u];
  }
}
inline uint32_t dense_l0_load_index(const uint32_t* g) {
  return (g[3] >> 29) | ((g[4] & 0x007fffffu) << 3);
}
#if !defined(MOM_SYCL_HAS_CUDA) && !defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
using IntelU32x4 = uint32_t __attribute__((ext_vector_type(4)));
static_assert(sizeof(IntelU32x4) == 16);
inline void store_intel_u32x4(uint32_t* p, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  *reinterpret_cast<IntelU32x4*>(p) = IntelU32x4{x, y, z, w};
}
#else
inline void store_intel_u32x4(uint32_t* p, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  p[0] = x; p[1] = y; p[2] = z; p[3] = w;
}
#endif
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
using NativeU32x4 = uint32_t __attribute__((ext_vector_type(4)));
static_assert(sizeof(NativeU32x4) == 16);
inline void store_aligned_u32x4(uint32_t* p, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  // Both CUDA toolchains scalarize sycl::vec::store here. Preserve the 16-byte alignment in IR so
  // the two traffic-heavy CUDA rounds lower to one vector global store per survivor.
  *reinterpret_cast<NativeU32x4*>(p) = NativeU32x4{x, y, z, w};
}
inline void dense_l0_store_compact(uint32_t* o, const uint32_t* f) {
  // CUDA keeps the 125 hash bits in one aligned vector and writes the leaf mapping to a separate
  // stream. Generation scatters and round-0 reads are 16 rather than 20 bytes per row.
#if defined(MOM_SYCL_HAS_CUDA)
  // The physical bucket already carries most of f0. Keep only its exact-bin quotient in spare high
  // bits and store f1..f4 directly, so collision rounds avoid unpacking cross-word 25-bit fields.
  const uint32_t rest = field_rest(f[0]);
  store_aligned_u32x4(o, f[1] | ((rest & 0x7fu) << 25),
                         f[2] | ((rest >> 7) << 25), f[3], f[4]);
#else
  store_aligned_u32x4(o,
    f[0] | ((f[1] & 0x7fu) << 25),
    (f[1] >> 7) | ((f[2] & 0x3fffu) << 18),
    (f[2] >> 14) | ((f[3] & 0x1fffffu) << 11),
    (f[3] >> 21) | ((f[4] & 0x01ffffffu) << 4));
#endif
}
#endif

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
#if defined(__NVPTX__)
  uint32_t al = a.lo, ah = a.hi, bl = b.lo, bh = b.hi;
  uint32_t cl = c.lo, ch = c.hi, dl = d.lo, dh = d.hi;
  // Keep the complete 64-bit G primitive in registers. PTX carry chains avoid the compare/select
  // sequences generated for portable uint32_t carry detection, and funnel shifts map each cross-half
  // rotate to one instruction per half.
  asm volatile(
    "{ .reg .u32 t0, t1;\n"
    "add.cc.u32 %0, %0, %2; addc.u32 %1, %1, %3;\n"
    "add.cc.u32 %0, %0, %8; addc.u32 %1, %1, %9;\n"
    "xor.b32 t0, %6, %0; xor.b32 t1, %7, %1; mov.b32 %6, t1; mov.b32 %7, t0;\n"
    "add.cc.u32 %4, %4, %6; addc.u32 %5, %5, %7;\n"
    "xor.b32 t0, %2, %4; xor.b32 t1, %3, %5;\n"
    "shf.r.wrap.b32 %2, t0, t1, 24; shf.r.wrap.b32 %3, t1, t0, 24;\n"
    "add.cc.u32 %0, %0, %2; addc.u32 %1, %1, %3;\n"
    "add.cc.u32 %0, %0, %10; addc.u32 %1, %1, %11;\n"
    "xor.b32 t0, %6, %0; xor.b32 t1, %7, %1;\n"
    "shf.r.wrap.b32 %6, t0, t1, 16; shf.r.wrap.b32 %7, t1, t0, 16;\n"
    "add.cc.u32 %4, %4, %6; addc.u32 %5, %5, %7;\n"
    "xor.b32 t0, %2, %4; xor.b32 t1, %3, %5;\n"
    "shf.l.wrap.b32 %2, t1, t0, 1; shf.l.wrap.b32 %3, t0, t1, 1;\n"
    "}\n"
    : "+&r"(al), "+&r"(ah), "+&r"(bl), "+&r"(bh),
      "+&r"(cl), "+&r"(ch), "+&r"(dl), "+&r"(dh)
    : "r"(x.lo), "r"(x.hi), "r"(y.lo), "r"(y.hi));
  a = B2bPair{al, ah}; b = B2bPair{bl, bh};
  c = B2bPair{cl, ch}; d = B2bPair{dl, dh};
#else
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
#endif
}

template <uint8_t I>
inline B2bPair b2b_message_pair(const B2bPair m0, const B2bPair m1) {
  if constexpr (I == 0) return m0;
  if constexpr (I == 1) return m1;
  return B2bPair{0u, 0u};
}

// Make both the round and every sigma/message index a template constant. This removes the tiny
// indexed loop from the 2^24-invocation generator while keeping b2b_g_pair and all arithmetic common
// across backends. It cuts Arc B580 gen-fill from roughly 20 to 12--13 ms.
template <unsigned R>
inline void b2b_rounds_pair(B2bPair (&v)[16], const B2bPair m0, const B2bPair m1) {
  static_assert(R < 12);
  b2b_g_pair(v[0], v[4], v[ 8], v[12], b2b_message_pair<B2B_SIGMA[R][0]>(m0, m1),
                                              b2b_message_pair<B2B_SIGMA[R][1]>(m0, m1));
  b2b_g_pair(v[1], v[5], v[ 9], v[13], b2b_message_pair<B2B_SIGMA[R][2]>(m0, m1),
                                              b2b_message_pair<B2B_SIGMA[R][3]>(m0, m1));
  b2b_g_pair(v[2], v[6], v[10], v[14], b2b_message_pair<B2B_SIGMA[R][4]>(m0, m1),
                                              b2b_message_pair<B2B_SIGMA[R][5]>(m0, m1));
  b2b_g_pair(v[3], v[7], v[11], v[15], b2b_message_pair<B2B_SIGMA[R][6]>(m0, m1),
                                              b2b_message_pair<B2B_SIGMA[R][7]>(m0, m1));
  b2b_g_pair(v[0], v[5], v[10], v[15], b2b_message_pair<B2B_SIGMA[R][8]>(m0, m1),
                                              b2b_message_pair<B2B_SIGMA[R][9]>(m0, m1));
  b2b_g_pair(v[1], v[6], v[11], v[12], b2b_message_pair<B2B_SIGMA[R][10]>(m0, m1),
                                              b2b_message_pair<B2B_SIGMA[R][11]>(m0, m1));
  b2b_g_pair(v[2], v[7], v[ 8], v[13], b2b_message_pair<B2B_SIGMA[R][12]>(m0, m1),
                                              b2b_message_pair<B2B_SIGMA[R][13]>(m0, m1));
  b2b_g_pair(v[3], v[4], v[ 9], v[14], b2b_message_pair<B2B_SIGMA[R][14]>(m0, m1),
                                              b2b_message_pair<B2B_SIGMA[R][15]>(m0, m1));
  if constexpr (R + 1 < 12) b2b_rounds_pair<R + 1>(v, m0, m1);
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
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  // AdaptiveCpp/CUDA's generic optimizer expands the template form into a larger register-live
  // region: RTX 5060 Ti fell 3.9% (26.62 -> 25.57 Sol/s), while its indexed form remained fastest.
  // Intel gains about 12% from compile-time expansion, DPC++/CUDA gains 2.2%, and HIP is neutral, so
  // retain only this measured compiler/backend exception rather than forfeiting the portable win.
  const B2bPair zero{0u, 0u};
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
#else
  b2b_rounds_pair<0>(v, m0, m1);
#endif
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

// The solver needs the five integer fields, not the intermediate 16-byte sub-element and 20-byte
// ExpandArray row. Extract the same consecutive 25-bit big-endian bit groups directly from four
// little-endian accumulator words. Besides removing two short-lived private arrays, this exposes a
// fixed shift/or network to the device compiler instead of a byte loop with a running bit counter.
inline void words_to_fields_25(const uint32_t* w, uint32_t f[NLEVELS]) {
#if defined(__NVPTX__)
  const uint32_t q0 = __builtin_bswap32(w[0]);
  const uint32_t q1 = __builtin_bswap32(w[1]);
  const uint32_t q2 = __builtin_bswap32(w[2]);
  const uint32_t q3 = __builtin_bswap32(w[3]);
  f[0] = q0 >> 7;
  f[1] = ((q0 & 0x7fu) << 18) | (q1 >> 14);
  f[2] = ((q1 & 0x3fffu) << 11) | (q2 >> 21);
  f[3] = ((q2 & 0x1fffffu) << 4) | (q3 >> 28);
  f[4] = (q3 >> 3) & 0x1ffffffu;
#else
  const auto b = [=](unsigned i) { return (w[i >> 2] >> (8u * (i & 3u))) & 0xffu; };
  f[0] = (b(0) << 17) | (b(1) << 9)  | (b(2) << 1)  | (b(3) >> 7);
  f[1] = ((b(3) & 0x7fu) << 18) | (b(4) << 10) | (b(5) << 2) | (b(6) >> 6);
  f[2] = ((b(6) & 0x3fu) << 19) | (b(7) << 11) | (b(8) << 3) | (b(9) >> 5);
  f[3] = ((b(9) & 0x1fu) << 20) | (b(10) << 12) | (b(11) << 4) | (b(12) >> 4);
  f[4] = ((b(12) & 0x0fu) << 21) | (b(13) << 13) | (b(14) << 5) | (b(15) >> 3);
#endif
}

// ===========================================================================================
// M1 gen-kernel validation: compute the first EQUIHASH_TEST_ROWS entries' 20-byte expanded rows.
// NAIVE_TWIST=true uses the per-entry naive twist; false uses the sub-group inclusive-scan over the
// 16-aligned block (one sub-group per block prefix-sums the 16 stock hashes). Both must produce
// identical rows; the entrypoint runs both so hash-vector tests cover both implementations.
// ===========================================================================================
template <bool NAIVE_TWIST> class ZelHashGenTestKernel;

template <bool NAIVE_TWIST>
static sycl::event submit_gen_test(
  sycl::queue& q, MomKernelBundle& kb, const uint64_t* d_base_h, const uint8_t* d_base_pending,
  uint8_t* d_rows /* EQUIHASH_TEST_ROWS * HASH_LENGTH */
) {
  if constexpr (NAIVE_TWIST) {
    constexpr unsigned WG = 64;
    const size_t global = (EQUIHASH_TEST_ROWS + WG - 1) / WG * WG;
    return q.submit([&](sycl::handler& h) {
      mom_use_bundle(h, kb);
      h.parallel_for<ZelHashGenTestKernel<true>>(
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
      mom_use_bundle(h, kb);
      h.parallel_for<ZelHashGenTestKernel<false>>(
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
// A level-L slot lives at level[L] + ((bucket * slot_capacity + slot) * level_u32(L, gpu_compact)).
// At L>=1, w0 carries active_low plus the parent bucket. Parent slots are reconstructed from that
// bucket after the final filter. Level 0 has no parent metadata; its dense record carries the leaf index.
// nslots[L*NBUCKETS + bucket] counts occupancy.

using dev_atomic_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device, sycl::access::address_space::global_space>;
using slm_atomic_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::work_group, sycl::access::address_space::local_space>;

inline uint32_t zelhash_scan16(const sycl::sub_group& sg, uint32_t x) {
#if defined(__NVPTX__)
  // SYCL's generic segmented scan expands into a much larger NVPTX shuffle network. ZelHash always
  // resets at a 16-hash boundary, so use PTX's native 16-lane clamp and let one warp carry two
  // independent prefixes. This is the CUDA analogue of Intel's native 16-wide sub-group scan.
  const unsigned lane = sg.get_local_linear_id() & 15u;
  uint32_t prior;
  asm volatile("shfl.sync.up.b32 %0, %1, 1, 0x1000, 0xffffffff;" : "=r"(prior) : "r"(x));
  if (lane >= 1u) x += prior;
  asm volatile("shfl.sync.up.b32 %0, %1, 2, 0x1000, 0xffffffff;" : "=r"(prior) : "r"(x));
  if (lane >= 2u) x += prior;
  asm volatile("shfl.sync.up.b32 %0, %1, 4, 0x1000, 0xffffffff;" : "=r"(prior) : "r"(x));
  if (lane >= 4u) x += prior;
  asm volatile("shfl.sync.up.b32 %0, %1, 8, 0x1000, 0xffffffff;" : "=r"(prior) : "r"(x));
  if (lane >= 8u) x += prior;
  return x;
#elif defined(MOM_SYCL_HAS_HIP)
  // AMD wavefronts are wider than each independent 16-hash ZelHash prefix. Use four native
  // subgroup permutes with an explicit 16-lane boundary so a 256-thread work-group can keep whole
  // wavefronts occupied instead of launching one half-wave work-group per hash block.
  const unsigned lane = (unsigned)sg.get_local_linear_id();
  const unsigned team_lane = lane & 15u;
  for (unsigned delta = 1u; delta <= 8u; delta <<= 1u) {
    const unsigned source = team_lane >= delta ? lane - delta : lane;
    const uint32_t prior = mom_select_from_group(sg, x, source);
    if (team_lane >= delta) x += prior;
  }
  return x;
#else
  return sycl::inclusive_scan_over_group(sg, x, sycl::plus<uint32_t>());
#endif
}

#if defined(__NVPTX__)
inline void zelhash_scan16_words(const sycl::sub_group& sg, const uint32_t mine[16], uint32_t acc[16]) {
  const unsigned lane = sg.get_local_linear_id() & 15u;
  uint32_t prior[16];
#pragma unroll
  for (unsigned i = 0; i < 16; ++i)
    asm volatile("shfl.sync.up.b32 %0, %1, 1, 0x1000, 0xffffffff;" : "=r"(prior[i]) : "r"(mine[i]));
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) acc[i] = mine[i] + (lane >= 1u ? prior[i] : 0u);
#pragma unroll
  for (unsigned i = 0; i < 16; ++i)
    asm volatile("shfl.sync.up.b32 %0, %1, 2, 0x1000, 0xffffffff;" : "=r"(prior[i]) : "r"(acc[i]));
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) if (lane >= 2u) acc[i] += prior[i];
#pragma unroll
  for (unsigned i = 0; i < 16; ++i)
    asm volatile("shfl.sync.up.b32 %0, %1, 4, 0x1000, 0xffffffff;" : "=r"(prior[i]) : "r"(acc[i]));
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) if (lane >= 4u) acc[i] += prior[i];
#pragma unroll
  for (unsigned i = 0; i < 16; ++i)
    asm volatile("shfl.sync.up.b32 %0, %1, 8, 0x1000, 0xffffffff;" : "=r"(prior[i]) : "r"(acc[i]));
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) if (lane >= 8u) acc[i] += prior[i];
}
#endif

// ---- Gen-fill: produce all 2^26 entries and scatter each into its round-0 bucket. One sub-group of
// 16 lanes per 16-aligned hash-index block (the validated M1 scan), then each lane emits its 4
// sub-element entries. field_bucket() maps field-0 to the physical bucket; the recovery index is
// packed into each common record or written to CUDA's separate cold stream.
template <bool GPU_COMPACT> class ZelHashGenFillKernelT;
template <bool GPU_COMPACT = false>
static sycl::event submit_gen_fill(
  sycl::queue& q, MomKernelBundle& kb, const uint64_t* d_base_h, const uint8_t* d_base_pending,
  uint32_t* d_level0, uint32_t* d_l0_index, uint32_t* d_nslots, unsigned slot_capacity
) {
  constexpr uint32_t NUM_HASHES = (uint32_t)(NUM_ENTRIES / INDICES_PER_HASH);   // 2^24
#if defined(MOM_SYCL_HAS_HIP)
  constexpr unsigned WG = 256u;
#elif defined(MOM_SYCL_HAS_CUDA)
  // The CUDA-selected template uses full blocks for occupancy; Intel launches the false template
  // with its measured 16-wide work-group. AdaptiveCpp/CUDA retains 16 because its SSCP path cannot
  // carry the inline PTX specialization above.
  constexpr unsigned WG = GPU_COMPACT ? 256u : 16u;
#else
  constexpr unsigned WG = 16u;
#endif
  const size_t global = NUM_HASHES;
  // The 76-byte Blake base is identical for every work-item. Capture it as plain kernel arguments
  // so all SYCL backends can use their constant/uniform path instead of issuing the same USM loads
  // from every lane. This is a small, measured cross-vendor win and keeps the generator common.
  BaseState base_capture{};
  std::memcpy(base_capture.h, d_base_h, sizeof(base_capture.h));
  std::memcpy(base_capture.pending, d_base_pending, sizeof(base_capture.pending));
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    h.parallel_for<ZelHashGenFillKernelT<GPU_COMPACT>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
      [=](sycl::nd_item<1> it) MOM_REQD_SG_16 {
      const sycl::sub_group sg = it.get_sub_group();
      const uint32_t hash_idx = (uint32_t)it.get_global_linear_id();

      // Pair-form base state + the two non-zero message words (block = pending[0..11] || htole32(g)).
      // These are loop-invariant across the 16-block; only g (m1.hi) varies per lane.
      B2bPair base_hp[8];
      for (unsigned i = 0; i < 8; ++i)
        base_hp[i] = B2bPair{ (uint32_t)(base_capture.h[i] & 0xffffffffull),
                              (uint32_t)(base_capture.h[i] >> 32) };
      const B2bPair m0{ b2b_load32le(base_capture.pending + 0),
                        b2b_load32le(base_capture.pending + 4) };
      const B2bPair m1{ b2b_load32le(base_capture.pending + 8), 0u };
      // m1.hi = g is set inside stock_hash_words_pair.

      uint32_t mine[16];
      stock_hash_words_pair(base_hp, m0, m1, hash_idx, mine);
      uint32_t acc[16];
#if defined(__NVPTX__)
      zelhash_scan16_words(sg, mine, acc);
#else
      for (unsigned i = 0; i < 16; ++i)
        acc[i] = zelhash_scan16(sg, mine[i]);
#endif
      acc[3] &= 0xF8FFFFFFu; acc[7] &= 0xF8FFFFFFu; acc[11] &= 0xF8FFFFFFu; acc[15] &= 0xF8FFFFFFu;

#if defined(MOM_EQ_FORWARD_MAP)
      uint32_t l0_forward[INDICES_PER_HASH] = {
        0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu
      };
#endif
      for (unsigned s = 0; s < INDICES_PER_HASH; ++s) {
        const uint32_t e = hash_idx * INDICES_PER_HASH + s;
        const unsigned base_word = s * (SUB_ELEMENT_BYTES / 4);
        uint32_t f[NLEVELS];
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
        // AdaptiveCpp lowers this byte expansion faster than its fixed shift network. DPC++/NVPTX
        // instead uses the direct four-word slicer below; all later solver stages remain common.
        uint8_t sub[SUB_ELEMENT_BYTES];
        for (unsigned w = 0; w < SUB_ELEMENT_BYTES / 4; ++w) {
          const uint32_t v = acc[base_word + w];
          sub[4 * w]     = (uint8_t)(v & 0xFF);
          sub[4 * w + 1] = (uint8_t)((v >> 8) & 0xFF);
          sub[4 * w + 2] = (uint8_t)((v >> 16) & 0xFF);
          sub[4 * w + 3] = (uint8_t)((v >> 24) & 0xFF);
        }
        uint8_t row[HASH_LENGTH];
        expand_array_25(sub, row);
        row_to_fields(row, f);
#else
        words_to_fields_25(acc + base_word, f);
#endif

        const uint32_t bucket = field_bucket(f[0]);
        const uint32_t slot = dev_atomic_u32(d_nslots[bucket]).fetch_add(1u);
#if defined(MOM_EQ_FORWARD_MAP)
        if constexpr (GPU_COMPACT)
          l0_forward[s] = slot < slot_capacity ? (bucket << 14) | slot : 0xffffffffu;
#endif
        if (slot >= slot_capacity) continue;                         // drop on overflow
        size_t pos = (size_t)bucket * slot_capacity + slot;
#if defined(MOM_SYCL_HAS_CUDA)
        if constexpr (GPU_COMPACT) pos = cuda_record_pos(bucket, slot);
#endif
        uint32_t* rec = d_level0 + pos * level_u32(0, GPU_COMPACT);
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
        if constexpr (GPU_COMPACT) {
          dense_l0_store_compact(rec, f);
#if !defined(MOM_EQ_FORWARD_MAP)
          d_l0_index[pos] = e;
#endif
        } else
#endif
        {
          dense_l0_store(rec, f, e);
        }
      }
#if defined(MOM_EQ_FORWARD_MAP)
      if constexpr (GPU_COMPACT)
        store_aligned_u32x4(d_l0_index + (size_t)hash_idx * INDICES_PER_HASH,
          l0_forward[0], l0_forward[1], l0_forward[2], l0_forward[3]);
#endif
    });
  });
}

// ---- Collision round R: read level R, collide field R across (bucket, RESTBITS) pairs, XOR the
// carried fields -> write level R+1. One work-group per bucket; SLM rest-bin table indexes by the low
// RESTBITS of field R. For each colliding pair we XOR fields R+1..4, write a survivor at level R+1,
// map its new active field through field_bucket(), and retain its parent bucket for recovery.
// The final round does not materialize level 4: only pairs whose last field XORs to zero can be
// solutions, so it records their two level-3 nodes directly for the rare recovery pass.
struct EqCandidateRef { uint32_t bucket, slot_a, slot_b; };
struct EqCandidate { uint32_t leaves[1u << K]; };
template <unsigned R, bool GPU_COMPACT> class ZelHashRoundKernel;
template <unsigned R, bool GPU_COMPACT> class ZelHashSplitRoundKernel;
class ZelHashCudaTiledRound0Kernel;
template <unsigned R> class ZelHashCudaTiledLaterRoundKernel;
template <unsigned R, bool GPU_COMPACT = false>
static sycl::event submit_round(
  sycl::queue& q, MomKernelBundle& kb, const uint32_t* d_in, uint32_t* d_out,
  const uint32_t* d_in_nslots, uint32_t* d_out_nslots, unsigned wg, unsigned slot_capacity,
  EqCandidateRef* d_ref = nullptr, uint32_t* d_ref_count = nullptr, uint32_t cand_cap = 0
) {
  // Progressive per-level widths (compile-time): round R reads level R (IN_FIELDS field words at stride
  // IN_U32) and writes level R+1 (OUT_FIELDS = IN_FIELDS-1 field words at stride OUT_U32). Fewer words
  // written per survivor == less scattered global write traffic (the bandwidth bottleneck).
  constexpr unsigned IN_FIELDS  = level_fields(R);       // 5,4,3,2 for R=0..3
  constexpr unsigned IN_U32     = level_u32(R, GPU_COMPACT);
  constexpr unsigned OUT_FIELDS = level_fields(R + 1);   // 4,3,2,1
  constexpr unsigned OUT_U32    = level_u32(R + 1, GPU_COMPACT);
#if defined(MOM_SYCL_HAS_CUDA)
  if constexpr (GPU_COMPACT && R == 0u) {
    constexpr unsigned PAIRS_PER_LANE = 9u;
    constexpr unsigned PAIR_CAP = PAIRS_PER_LANE * ROUND_WG;
    constexpr unsigned LINK_U16_OFF = 2u * PAIR_CAP;
    constexpr unsigned COUNTER_U16_OFF = LINK_U16_OFF + MAX_SLOTS;
    constexpr unsigned COUNTER_U32 = COUNTER_U16_OFF / 2u;
    constexpr unsigned STAGE_U32 = 2u * MAX_SLOTS;
    const size_t global = (size_t)NBUCKETS * wg;
    return q.submit([&](sycl::handler& h) {
      mom_use_bundle(h, kb);
      // The chain/pair-list phase needs ~52 KiB. Once pair codes are in registers, reuse the same
      // arena as two 32-KiB field planes, matching the GPU's 64-KiB fast shared-memory shape.
      sycl::local_accessor<uint32_t, 1> scratch(sycl::range<1>(STAGE_U32), h);
      h.parallel_for<ZelHashCudaTiledRound0Kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const uint32_t bucket = (uint32_t)it.get_group(0);
        const unsigned lid = (unsigned)it.get_local_id(0);
        uint32_t* const scratch32 = scratch.template get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* const chain = reinterpret_cast<uint16_t*>(scratch32) + LINK_U16_OFF;
        const uint32_t cnt = sycl::min(d_in_nslots[bucket], (uint32_t)slot_capacity);
        // Exactly eight possible rows per lane. Retain the aligned records in registers through the
        // chain build; later half-staging deliberately ends each pair of field lifetimes.
        uint32_t f1[8], f2[8], f3[8], f4[8];
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          NativeU32x4 v = {0u, 0u, 0u, 0u};
          if (s < cnt)
            v = *reinterpret_cast<const NativeU32x4*>(d_in + cuda_record_pos(bucket, s) * IN_U32);
          f1[t] = v[0]; f2[t] = v[1]; f3[t] = v[2]; f4[t] = v[3];
        }

        for (unsigned r = lid; r < NRESTBINS; r += ROUND_WG) scratch32[r] = 0;
        if (lid == 0) scratch32[COUNTER_U32] = 0;
        sycl::group_barrier(it.get_group());

#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s >= cnt) continue;
          const uint32_t rb = (f1[t] >> 25) | ((f2[t] >> 25) << 7);
          chain[s] = (uint16_t)slm_atomic_u32(scratch32[rb]).exchange(s + 1u);
        }
        sycl::group_barrier(it.get_group());

        // Count pairs per lane, then reserve each lane's contiguous segment with one warp scan. This
        // removes the shared atomic (and its NVPTX vote/shuffle aggregation) from every chain link.
        uint32_t local_pairs = 0;
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s >= cnt) continue;
          for (uint32_t link = chain[s]; link != 0u; link = chain[link - 1u]) ++local_pairs;
        }
        const sycl::sub_group sg = it.get_sub_group();
        const unsigned lane = (unsigned)sg.get_local_linear_id();
        const unsigned warp = lid >> 5;
        const uint32_t lane_prefix = sycl::exclusive_scan_over_group(sg, local_pairs, sycl::plus<uint32_t>());
        const uint32_t warp_total = sycl::reduce_over_group(sg, local_pairs, sycl::plus<uint32_t>());
        if (lane == 0u) scratch32[warp] = warp_total;
        sycl::group_barrier(it.get_group());
        if (warp == 0u) {
          const uint32_t v = scratch32[lane];
          const uint32_t p = sycl::exclusive_scan_over_group(sg, v, sycl::plus<uint32_t>());
          scratch32[lane] = p;
          if (lane == 31u) scratch32[32] = p + v;
        }
        sycl::group_barrier(it.get_group());
        const uint32_t pair_base = scratch32[warp] + lane_prefix;
        const uint32_t pair_count = sycl::min(scratch32[32], (uint32_t)PAIR_CAP);
        sycl::group_barrier(it.get_group());
        uint32_t local_pos = 0;
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s >= cnt) continue;
          for (uint32_t link = chain[s]; link != 0u; link = chain[link - 1u]) {
            const uint32_t pos = pair_base + local_pos++;
            if (pos < PAIR_CAP) scratch32[pos] = (s << 16) | (link - 1u);
          }
        }
        sycl::group_barrier(it.get_group());

        uint32_t pair_code[PAIRS_PER_LANE];
        bool pair_valid[PAIRS_PER_LANE];
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          const unsigned p = lid + j * ROUND_WG;
          pair_valid[j] = p < pair_count;
          pair_code[j] = pair_valid[j] ? scratch32[p] : 0u;
        }
        sycl::group_barrier(it.get_group());

        // First half: stage f1/f2 from the retained records, then keep only their pair XORs.
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s < cnt) {
            scratch32[s] = f1[t] & 0x1ffffffu;
            scratch32[MAX_SLOTS + s] = f2[t] & 0x1ffffffu;
          }
        }
        sycl::group_barrier(it.get_group());
        uint32_t x0[PAIRS_PER_LANE], x1[PAIRS_PER_LANE];
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          if (!pair_valid[j]) { x0[j] = x1[j] = 0u; continue; }
          const unsigned a = pair_code[j] >> 16, b = pair_code[j] & 0xffffu;
          x0[j] = scratch32[a] ^ scratch32[b];
          x1[j] = scratch32[MAX_SLOTS + a] ^ scratch32[MAX_SLOTS + b];
        }
        sycl::group_barrier(it.get_group());

        // Second half overwrites the arena after every lane consumed the first. Pair outputs can now
        // be emitted as one aligned vector; the 32 input-record registers are no longer live.
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s < cnt) {
            scratch32[s] = f3[t] & 0x1ffffffu;
            scratch32[MAX_SLOTS + s] = f4[t] & 0x1ffffffu;
          }
        }
        sycl::group_barrier(it.get_group());
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          if (!pair_valid[j]) continue;
          const unsigned a = pair_code[j] >> 16, b = pair_code[j] & 0xffffu;
          const uint32_t new_active = x0[j];
          const uint32_t out_bucket = field_bucket(new_active);
          const uint32_t oslot = dev_atomic_u32(d_out_nslots[out_bucket]).fetch_add(1u);
          if (oslot >= slot_capacity) continue;
          uint32_t* o = d_out + cuda_record_pos(out_bucket, oslot) * OUT_U32;
          store_aligned_u32x4(o, field_rest(new_active) | (bucket << RESTBITS), x1[j],
            scratch32[a] ^ scratch32[b],
            scratch32[MAX_SLOTS + a] ^ scratch32[MAX_SLOTS + b]);
        }
      });
    });
  }
  if constexpr (GPU_COMPACT && R >= 1u) {
    constexpr unsigned PAIRS_PER_LANE = 9u;
    constexpr unsigned PAIR_CAP = PAIRS_PER_LANE * ROUND_WG;
    constexpr unsigned LINK_U16_OFF = 2u * PAIR_CAP;
    constexpr unsigned STAGE_U32 = 2u * MAX_SLOTS;
    const size_t global = (size_t)NBUCKETS * wg;
    return q.submit([&](sycl::handler& h) {
      mom_use_bundle(h, kb);
      sycl::local_accessor<uint32_t, 1> scratch(sycl::range<1>(STAGE_U32), h);
      h.parallel_for<ZelHashCudaTiledLaterRoundKernel<R>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const uint32_t bucket = (uint32_t)it.get_group(0);
        const unsigned lid = (unsigned)it.get_local_id(0);
        uint32_t* const scratch32 = scratch.template get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* const chain = reinterpret_cast<uint16_t*>(scratch32) + LINK_U16_OFF;
        const uint32_t cnt = sycl::min(d_in_nslots[bucket], (uint32_t)slot_capacity);
        uint32_t w0[8], w1[8], w2[8], w3[8];
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          NativeU32x4 v = {0u, 0u, 0u, 0u};
          if (s < cnt) {
            const uint32_t* p = d_in + cuda_record_pos(bucket, s) * IN_U32;
            if constexpr (R == 3u) {
              v[0] = p[0];
              v[1] = p[1];
            } else {
              v = *reinterpret_cast<const NativeU32x4*>(p);
            }
          }
          w0[t] = v[0]; w1[t] = v[1]; w2[t] = v[2]; w3[t] = v[3];
        }
        for (unsigned r = lid; r < NRESTBINS; r += ROUND_WG) scratch32[r] = 0;
        sycl::group_barrier(it.get_group());
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s < cnt)
            chain[s] = (uint16_t)slm_atomic_u32(scratch32[stored_rest(w0[t])]).exchange(s + 1u);
        }
        sycl::group_barrier(it.get_group());

        uint32_t local_pairs = 0;
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s >= cnt) continue;
          for (uint32_t link = chain[s]; link != 0u; link = chain[link - 1u]) ++local_pairs;
        }
        const sycl::sub_group sg = it.get_sub_group();
        const unsigned lane = (unsigned)sg.get_local_linear_id();
        const unsigned warp = lid >> 5;
        const uint32_t lane_prefix = sycl::exclusive_scan_over_group(sg, local_pairs, sycl::plus<uint32_t>());
        const uint32_t warp_total = sycl::reduce_over_group(sg, local_pairs, sycl::plus<uint32_t>());
        if (lane == 0u) scratch32[warp] = warp_total;
        sycl::group_barrier(it.get_group());
        if (warp == 0u) {
          const uint32_t v = scratch32[lane];
          const uint32_t p = sycl::exclusive_scan_over_group(sg, v, sycl::plus<uint32_t>());
          scratch32[lane] = p;
          if (lane == 31u) scratch32[32] = p + v;
        }
        sycl::group_barrier(it.get_group());
        const uint32_t pair_base = scratch32[warp] + lane_prefix;
        const uint32_t pair_count = sycl::min(scratch32[32], (uint32_t)PAIR_CAP);
        sycl::group_barrier(it.get_group());
        uint32_t local_pos = 0;
#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s >= cnt) continue;
          for (uint32_t link = chain[s]; link != 0u; link = chain[link - 1u]) {
            const uint32_t pos = pair_base + local_pos++;
            if (pos < PAIR_CAP) scratch32[pos] = (s << 16) | (link - 1u);
          }
        }
        sycl::group_barrier(it.get_group());

        uint32_t pair_code[PAIRS_PER_LANE];
        bool pair_valid[PAIRS_PER_LANE];
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          const unsigned p = lid + j * ROUND_WG;
          pair_valid[j] = p < pair_count;
          pair_code[j] = pair_valid[j] ? scratch32[p] : 0u;
        }
        sycl::group_barrier(it.get_group());

#pragma unroll
        for (unsigned t = 0; t < 8u; ++t) {
          const unsigned s = lid + t * ROUND_WG;
          if (s < cnt) {
            scratch32[s] = w1[t];
            if constexpr (R <= 2u) scratch32[MAX_SLOTS + s] = w2[t];
          }
        }
        sycl::group_barrier(it.get_group());
        uint32_t x0[PAIRS_PER_LANE], x1[PAIRS_PER_LANE];
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          if (!pair_valid[j]) { x0[j] = x1[j] = 0u; continue; }
          const unsigned a = pair_code[j] >> 16, b = pair_code[j] & 0xffffu;
          x0[j] = scratch32[a] ^ scratch32[b];
          if constexpr (R <= 2u) x1[j] = scratch32[MAX_SLOTS + a] ^ scratch32[MAX_SLOTS + b];
          else x1[j] = 0u;
        }

        if constexpr (R == 1u) {
          sycl::group_barrier(it.get_group());
#pragma unroll
          for (unsigned t = 0; t < 8u; ++t) {
            const unsigned s = lid + t * ROUND_WG;
            if (s < cnt) scratch32[s] = w3[t];
          }
          sycl::group_barrier(it.get_group());
        }
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          if (!pair_valid[j]) continue;
          const unsigned a = pair_code[j] >> 16, b = pair_code[j] & 0xffffu;
          const uint32_t new_active = x0[j];
          if constexpr (R == K - 1u) {
            if (new_active != 0u) continue;
            const uint32_t idx = dev_atomic_u32(d_ref_count[0]).fetch_add(1u);
            if (idx < cand_cap) d_ref[idx] = {bucket, a, b};
          } else {
            const uint32_t out_bucket = field_bucket(new_active);
            const uint32_t oslot = dev_atomic_u32(d_out_nslots[out_bucket]).fetch_add(1u);
            if (oslot >= slot_capacity) continue;
            uint32_t* o = d_out + cuda_record_pos(out_bucket, oslot) * OUT_U32;
            if constexpr (R == 1u)
              store_aligned_u32x4(o, field_rest(new_active) | (bucket << RESTBITS), x1[j],
                                  scratch32[a] ^ scratch32[b], 0u);
            else {
              o[0] = field_rest(new_active) | (bucket << RESTBITS);
              o[1] = x1[j];
            }
          }
        }
      });
    });
  }
#endif
  if constexpr (ROUND_DIRECT_CHAINS) {
    constexpr unsigned SPLITS = R < 2u ? 2u : 1u;
    constexpr unsigned SPLIT_BITS = R < 2u ? 1u : 0u;
    // Each early split averages at most MAX_SLOTS/2 rows. This leaves headroom at the capped input
    // size while keeping round 0 under Xe's 64-KiB SLM limit; the normal cap remains later.
    constexpr unsigned LOCAL_SLOTS = R < 2u ? 2432u : MAX_SLOTS;
    constexpr unsigned LOCAL_RESTBINS = NRESTBINS / SPLITS;
    const unsigned split_wg = R < 2u ? 512u : wg;
    const size_t global = (size_t)NBUCKETS * SPLITS * split_wg;
    return q.submit([&](sycl::handler& h) {
      mom_use_bundle(h, kb);
      sycl::local_accessor<uint32_t, 1> heads(sycl::range<1>(LOCAL_RESTBINS), h);
      sycl::local_accessor<uint32_t, 1> fields(sycl::range<1>(LOCAL_SLOTS * (IN_FIELDS - 1u)), h);
      sycl::local_accessor<uint16_t, 1> next(sycl::range<1>(LOCAL_SLOTS), h);
      sycl::local_accessor<uint16_t, 1> original(sycl::range<1>(LOCAL_SLOTS), h);
      sycl::local_accessor<uint32_t, 1> local_count(sycl::range<1>(1), h);
      h.parallel_for<ZelHashSplitRoundKernel<R, GPU_COMPACT>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(split_wg)),
        [=](sycl::nd_item<1> it) {
          const unsigned group = (unsigned)it.get_group(0);
          const uint32_t bucket = group / SPLITS;
          const uint32_t split = group & (SPLITS - 1u);
          const unsigned lid = (unsigned)it.get_local_id(0);
          const unsigned lsz = (unsigned)it.get_local_range(0);
          uint32_t* const head = heads.template get_multi_ptr<sycl::access::decorated::no>().get();
          uint32_t* const staged = fields.template get_multi_ptr<sycl::access::decorated::no>().get();
          uint16_t* const chain = next.template get_multi_ptr<sycl::access::decorated::no>().get();
          uint16_t* const input_slot = original.template get_multi_ptr<sycl::access::decorated::no>().get();
          uint32_t* const count_ptr = local_count.template get_multi_ptr<sycl::access::decorated::no>().get();

          uint32_t cnt = d_in_nslots[bucket];
          if (cnt > slot_capacity) cnt = slot_capacity;
          for (unsigned bin = lid; bin < LOCAL_RESTBINS; bin += lsz) head[bin] = 0;
          if (lid == 0) count_ptr[0] = 0;
          sycl::group_barrier(it.get_group());

          const uint32_t* in_base = d_in + (size_t)bucket * slot_capacity * IN_U32;
          for (unsigned s = lid; s < cnt; s += lsz) {
            const uint32_t* g = in_base + (size_t)s * IN_U32;
            const uint32_t rest = collision_rest<R, GPU_COMPACT>(g);
            if ((rest & (SPLITS - 1u)) != split) continue;
            const uint32_t pos = slm_atomic_u32(count_ptr[0]).fetch_add(1u);
            if (pos >= LOCAL_SLOTS) continue;
            input_slot[pos] = (uint16_t)s;
            for (unsigned k = 0; k < IN_FIELDS - 1u; ++k)
              staged[(size_t)k * LOCAL_SLOTS + pos] = load_follower<R, GPU_COMPACT>(g, k);
            const uint32_t previous = slm_atomic_u32(head[rest >> SPLIT_BITS]).exchange(pos + 1u);
            chain[pos] = (uint16_t)previous;
          }
          sycl::group_barrier(it.get_group());

          const uint32_t compact_count = sycl::min(count_ptr[0], (uint32_t)LOCAL_SLOTS);
          for (unsigned a = lid; a < compact_count; a += lsz) {
            for (uint32_t link = chain[a]; link != 0; link = chain[link - 1u]) {
              const unsigned b = link - 1u;
              const uint32_t new_active = staged[a] ^ staged[b];
              if constexpr (R == K - 1u) {
                if (new_active != 0u) continue;
                const uint32_t idx = dev_atomic_u32(d_ref_count[0]).fetch_add(1u);
                if (idx >= cand_cap) continue;
                d_ref[idx] = {bucket, input_slot[a], input_slot[b]};
              } else {
                const uint32_t out_bucket = field_bucket(new_active);
                const uint32_t oslot = dev_atomic_u32(d_out_nslots[out_bucket]).fetch_add(1u);
                if (oslot >= slot_capacity) continue;
                uint32_t* o = d_out + ((size_t)out_bucket * slot_capacity + oslot) * OUT_U32;
                const uint32_t first = field_rest(new_active) | (bucket << RESTBITS);
                if constexpr (R == 0u) {
                  store_intel_u32x4(o, first,
                    staged[(size_t)LOCAL_SLOTS + a] ^ staged[(size_t)LOCAL_SLOTS + b],
                    staged[2u * (size_t)LOCAL_SLOTS + a] ^ staged[2u * (size_t)LOCAL_SLOTS + b],
                    staged[3u * (size_t)LOCAL_SLOTS + a] ^ staged[3u * (size_t)LOCAL_SLOTS + b]);
                } else {
                  o[0] = first;
                  for (unsigned k = 1; k < OUT_FIELDS; ++k) {
                    const uint32_t* f = staged + (size_t)k * LOCAL_SLOTS;
                    o[k] = f[a] ^ f[b];
                  }
                }
              }
            }
          }
        });
    });
  }
  if constexpr (!ROUND_DIRECT_CHAINS) {
    const size_t global = (size_t)NBUCKETS * wg;
    return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    // One reusable SLM arena implements the fast block-local sequence used by mature Equihash solvers:
    // build exact-bin chains, overwrite the now-dead heads with a compact pair list, retain a bounded
    // number of pair IDs in each lane's registers, then overwrite the pair list with staged record
    // fields. Keeping links behind the maximum pair list makes these lifetimes non-overlapping; the
    // selected Intel path uses at most 29 KiB, CUDA/HIP about 53 KiB, and AdaptiveCpp/CUDA's
    // small-bucket fallback about 16 KiB. This avoids a random global load for every collision partner.
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
    constexpr unsigned PAIRS_PER_LANE = 3u;
#elif !defined(MOM_SYCL_HAS_CUDA) && !defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
    // WG1024 x 5 covers every measured Intel pair (a sixth slot did not change survivor counts).
    // The former WG512 x 9 cap silently discarded 150 pairs in the block-400000 exact profile.
    constexpr unsigned PAIRS_PER_LANE = 5u;
#else
    constexpr unsigned PAIRS_PER_LANE = 9u;
#endif
    constexpr unsigned PAIR_CAP = PAIRS_PER_LANE * ROUND_WG;
    constexpr unsigned LINK_U16_OFF = 2u * PAIR_CAP;
    constexpr unsigned COUNTER_U16_OFF = LINK_U16_OFF + MAX_SLOTS;
    constexpr unsigned SCRATCH_U32 = (COUNTER_U16_OFF + 3u) / 2u;
    sycl::local_accessor<uint32_t, 1> scratch(sycl::range<1>(SCRATCH_U32), h);
    constexpr unsigned FOLLOWERS = IN_FIELDS - 1u;
    h.parallel_for<ZelHashRoundKernel<R, GPU_COMPACT>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
      [=](sycl::nd_item<1> it) {
      const uint32_t bucket = (uint32_t)it.get_group(0);
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);
      uint32_t* const scratch32 = scratch.template get_multi_ptr<sycl::access::decorated::no>().get();
      uint16_t* const scratch16 = reinterpret_cast<uint16_t*>(scratch32);
      uint16_t* const bin_next = scratch16 + LINK_U16_OFF;
      constexpr unsigned COUNTER_U32 = COUNTER_U16_OFF / 2u;

      uint32_t cnt = d_in_nslots[bucket];
      if (cnt > slot_capacity) cnt = slot_capacity;
      for (unsigned r = lid; r < NRESTBINS; r += lsz) scratch32[r] = 0;
      if (lid == 0) scratch32[COUNTER_U32] = 0;
      sycl::group_barrier(it.get_group());

      const uint32_t* in_base = d_in + (size_t)bucket * slot_capacity * IN_U32;

      // Build the rest-bin linked lists. At levels >=1, w[0] also carries the parent bucket above
      // RESTBITS; masking it still selects the active field's exact low-bit bin.
      for (unsigned s = lid; s < cnt; s += lsz) {
        const uint32_t* g = in_base + (size_t)s * IN_U32;
        const uint32_t rb = collision_rest<R, GPU_COMPACT>(g);
        const uint32_t previous = slm_atomic_u32(scratch32[rb]).exchange(s + 1u);
        bin_next[s] = (uint16_t)previous;
      }
      sycl::group_barrier(it.get_group());

      // The predecessor chain contains exactly the records inserted before s, so every unordered pair
      // is emitted once without rereading the heads. Pair-list writes may now safely overwrite heads.
      for (unsigned s = lid; s < cnt; s += lsz) {
        uint32_t link = bin_next[s];                                // 1 + predecessor slot
        for (; link != 0; link = bin_next[link - 1u]) {
          const uint32_t t = link - 1u;
          const uint32_t pos = slm_atomic_u32(scratch32[COUNTER_U32]).fetch_add(1u);
          if (pos < PAIR_CAP) scratch32[pos] = (s << 16) | t;
        }
      }
      sycl::group_barrier(it.get_group());

      const uint32_t pair_count = sycl::min(scratch32[COUNTER_U32], (uint32_t)PAIR_CAP);
      uint32_t pair_code[PAIRS_PER_LANE];
      bool pair_valid[PAIRS_PER_LANE];
      for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
        const unsigned p = lid + j * lsz;
        pair_valid[j] = p < pair_count;
        pair_code[j] = pair_valid[j] ? scratch32[p] : 0u;
      }
      sycl::group_barrier(it.get_group());

      // A complete 32-bit field occupies at most 34 KiB on the selected paths, so stage each follower
      // in one pass. This is fewer barriers than splitting 64-bit slices into slot halves on portable 64-KiB
      // implementations, while pair IDs remain in registers and random global partner loads disappear.
      uint32_t xf[PAIRS_PER_LANE][FOLLOWERS];
      for (unsigned k = 0; k < FOLLOWERS; ++k) {
        for (unsigned s = lid; s < cnt; s += lsz) {
          const uint32_t* g = in_base + (size_t)s * IN_U32;
          scratch32[s] = load_follower<R, GPU_COMPACT>(g, k);
        }
        sycl::group_barrier(it.get_group());
        for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
          if (!pair_valid[j]) continue;
          const unsigned s = pair_code[j] >> 16;
          const unsigned t = pair_code[j] & 0xffffu;
          xf[j][k] = scratch32[s] ^ scratch32[t];
        }
        sycl::group_barrier(it.get_group());
      }

      for (unsigned j = 0; j < PAIRS_PER_LANE; ++j) {
        if (!pair_valid[j]) continue;
        // Slots always keep the ACTIVE collision field in w[0] and the remaining (higher) fields in
        // w[1..]. The active field XORs to 0; shift the rest down by one so w[0] becomes the NEXT
        // collision field at level R+1. (Position-relative, independent of which absolute field R is.)
        const uint32_t new_active = xf[j][0];                     // new active field (full 25-bit)
        if constexpr (R == K - 1u) {
          if (new_active != 0u) continue;
          const uint32_t idx = dev_atomic_u32(d_ref_count[0]).fetch_add(1u);
          if (idx >= cand_cap) continue;
          const uint32_t code = pair_code[j];
          d_ref[idx] = {bucket, code >> 16, code & 0xffffu};
          continue;
        }
        const uint32_t out_bucket = field_bucket(new_active);
        const uint32_t oslot = dev_atomic_u32(d_out_nslots[out_bucket]).fetch_add(1u);
        if (oslot >= slot_capacity) continue;
        // Single compact stream: w0 = active_low | (parent_bucket<<RESTBITS), followers in w1...
        uint32_t* o = d_out + ((size_t)out_bucket * slot_capacity + oslot) * OUT_U32;
        const uint32_t first = field_rest(new_active) | (bucket << RESTBITS);
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
        if constexpr (GPU_COMPACT && R == 0) {
          store_aligned_u32x4(o, first, xf[j][1], xf[j][2], xf[j][3]);
        } else if constexpr (GPU_COMPACT && R == 1) {
          store_aligned_u32x4(o, first, xf[j][1], xf[j][2], 0u);
        } else if constexpr (GPU_COMPACT && R == 2) {
          store_aligned_u32x4(o, first, xf[j][1], 0u, 0u);
        } else
#endif
        {
          o[0] = first;
          for (unsigned k = 1; k < OUT_FIELDS; ++k) o[k] = xf[j][k];
        }
      }
      });
    });
  }
}

struct EqLevels {
  const uint32_t* level[NSTORED_LEVELS];
  const uint32_t* l0_index;
};

// Reconstruct each final pair's level-3 nodes by rescanning only their recorded parent buckets.
// A parent bucket contains about 8192 rows and about as many collision pairs; candidate recovery is
// rare (~44 zero nodes before DistinctIndices) and therefore much cheaper than carrying a 26-bit
// parent-pair identifier through all ~67 million survivors in every round.
template <bool GPU_COMPACT>
inline uint32_t recovery_follower(const uint32_t* g, unsigned level, unsigned k) {
  if (level == 0) {
#if defined(MOM_SYCL_HAS_CUDA)
    if constexpr (GPU_COMPACT) return g[k] & 0x1ffffffu;
#endif
    switch (k) {
      case 0: return ((g[0] >> 25) & 0x7fu) | ((g[1] & 0x3ffffu) << 7);
      case 1: return (g[1] >> 18) | ((g[2] & 0x7ffu) << 14);
      case 2: return (g[2] >> 11) | ((g[3] & 0xfu) << 21);
      default: return (g[3] >> 4) & 0x01ffffffu;
    }
  }
  return g[k + 1u];
}

template <bool GPU_COMPACT> class ZelHashRecoverKernel;
template <bool GPU_COMPACT = false>
static sycl::event submit_recover(
  sycl::queue& q, MomKernelBundle& kb, EqLevels lv, const uint32_t* d_nslots,
  const EqCandidateRef* d_ref, uint32_t ref_count, EqCandidate* d_cand,
  uint32_t* d_cand_count, uint32_t cand_cap, unsigned wg, unsigned slot_capacity,
  const uint32_t* d_ref_count = nullptr
) {
  if (!ref_count) return sycl::event{};
  const size_t global = (size_t)ref_count * wg;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    sycl::local_accessor<uint32_t, 1> bin_head(sycl::range<1>(NRESTBINS), h);
    sycl::local_accessor<uint16_t, 1> bin_next(sycl::range<1>(MAX_SLOTS), h);
    sycl::local_accessor<uint32_t, 1> node_bucket(sycl::range<1>(1u << K), h);
    sycl::local_accessor<uint32_t, 1> node_slot(sycl::range<1>(1u << K), h);
    sycl::local_accessor<uint32_t, 1> next_bucket(sycl::range<1>(1u << K), h);
    sycl::local_accessor<uint32_t, 1> next_slot(sycl::range<1>(1u << K), h);
    sycl::local_accessor<uint32_t, 1> target(sycl::range<1>(K), h);
    sycl::local_accessor<uint32_t, 1> found(sycl::range<1>(1), h);
    sycl::local_accessor<uint32_t, 1> valid(sycl::range<1>(1), h);
    h.parallel_for<ZelHashRecoverKernel<GPU_COMPACT>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
      [=](sycl::nd_item<1> it) {
      const unsigned candidate = (unsigned)it.get_group(0);
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);
      // The normal in-order chain launches the bounded recovery grid before the host reads the
      // final-round count. This uniform early exit removes an entire Windows event-wait quantum.
      if (d_ref_count && candidate >= *d_ref_count) return;
      if (lid == 0) {
        node_bucket[0] = node_bucket[1] = d_ref[candidate].bucket;
        node_slot[0] = d_ref[candidate].slot_a;
        node_slot[1] = d_ref[candidate].slot_b;
        valid[0] = 1;
      }
      sycl::group_barrier(it.get_group());

      // The fused final round already supplied the two level-3 children of the discarded level-4
      // zero node. Expand those two subtrees through levels 3..1 to recover all sixteen leaves.
      unsigned node_count = 2;
      for (unsigned level = K - 1u; level != 0; --level) {
        for (unsigned ni = 0; ni < node_count; ++ni) {
          if (lid == 0) {
            found[0] = 0xffffffffu;
            if (valid[0]) {
              const uint32_t cb = node_bucket[ni], cs = node_slot[ni];
              size_t child_pos = (size_t)cb * slot_capacity + cs;
#if defined(MOM_SYCL_HAS_CUDA)
              if constexpr (GPU_COMPACT) child_pos = cuda_record_pos(cb, cs);
#endif
              const uint32_t* child = lv.level[level]
                + child_pos * level_u32(level, GPU_COMPACT);
              next_bucket[2u * ni] = next_bucket[2u * ni + 1u] = child[0] >> RESTBITS;
              target[0] = field_from_parts(cb, stored_rest(child[0]));
              for (unsigned k = 1; k < level_fields(level); ++k) target[k] = child[k];
            }
          }
          for (unsigned r = lid; r < NRESTBINS; r += lsz) bin_head[r] = 0;
          sycl::group_barrier(it.get_group());
          if (!valid[0]) continue;

          const uint32_t pb = next_bucket[2u * ni];
          const unsigned parent_level = level - 1u;
          uint32_t cnt = d_nslots[(size_t)parent_level * NBUCKETS + pb];
          if (cnt > slot_capacity) cnt = slot_capacity;
          const unsigned parent_u32 = level_u32(parent_level, GPU_COMPACT);
          for (unsigned s = lid; s < cnt; s += lsz) {
            size_t record_pos = (size_t)pb * slot_capacity + s;
#if defined(MOM_SYCL_HAS_CUDA)
            if constexpr (GPU_COMPACT)
              record_pos = cuda_record_pos(pb, s);
#endif
            const uint32_t* record = lv.level[parent_level] + record_pos * parent_u32;
            uint32_t rb;
#if defined(MOM_SYCL_HAS_CUDA)
            if constexpr (GPU_COMPACT)
              rb = parent_level == 0u ? ((record[0] >> 25) | ((record[1] >> 25) << 7))
                                      : stored_rest(record[0]);
            else
#endif
              rb = parent_level == 0u ? field_rest(record[0]) : stored_rest(record[0]);
            const uint32_t previous = slm_atomic_u32(bin_head[rb]).exchange(s + 1u);
            bin_next[s] = (uint16_t)previous;
          }
          sycl::group_barrier(it.get_group());

          for (unsigned s = lid; s < cnt; s += lsz) {
            size_t a_pos = (size_t)pb * slot_capacity + s;
#if defined(MOM_SYCL_HAS_CUDA)
            if constexpr (GPU_COMPACT)
              a_pos = cuda_record_pos(pb, s);
#endif
            const uint32_t* a = lv.level[parent_level] + a_pos * parent_u32;
            for (uint32_t link = bin_next[s]; link != 0; link = bin_next[link - 1u]) {
              const unsigned t = link - 1u;
              size_t b_pos = (size_t)pb * slot_capacity + t;
#if defined(MOM_SYCL_HAS_CUDA)
              if constexpr (GPU_COMPACT)
                b_pos = cuda_record_pos(pb, t);
#endif
              const uint32_t* b = lv.level[parent_level] + b_pos * parent_u32;
              bool match = true;
              for (unsigned k = 0; k < level_fields(level); ++k) {
                if ((recovery_follower<GPU_COMPACT>(a, parent_level, k) ^
                     recovery_follower<GPU_COMPACT>(b, parent_level, k)) != target[k]) {
                  match = false; break;
                }
              }
              if (match) slm_atomic_u32(found[0]).exchange((s << 16) | t);
            }
          }
          sycl::group_barrier(it.get_group());
          if (lid == 0) {
            if (found[0] == 0xffffffffu) valid[0] = 0;
            else {
              next_slot[2u * ni] = found[0] >> 16;
              next_slot[2u * ni + 1u] = found[0] & 0xffffu;
            }
          }
          sycl::group_barrier(it.get_group());
        }
        node_count *= 2u;
        for (unsigned i = lid; i < node_count; i += lsz) {
          node_bucket[i] = next_bucket[i]; node_slot[i] = next_slot[i];
        }
        sycl::group_barrier(it.get_group());
      }

      constexpr unsigned PROOF = 1u << K;
      if (lid < PROOF && valid[0]) {
        const size_t leaf_pos = (size_t)node_bucket[lid] * slot_capacity + node_slot[lid];
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
        if constexpr (GPU_COMPACT) {
#if defined(MOM_EQ_FORWARD_MAP)
          // Generation stores a coalesced leaf->position map. Keep the physical position as
          // the recovery target; a single linear inversion pass resolves all final leaves together.
          next_slot[lid] = (node_bucket[lid] << 14) | node_slot[lid];
#else
          next_slot[lid] = lv.l0_index[leaf_pos];
#endif
        }
        else
#endif
        {
          const uint32_t* leaf = lv.level[0] + leaf_pos * level_u32(0, GPU_COMPACT);
          next_slot[lid] = dense_l0_load_index(leaf);
        }
      }
      sycl::group_barrier(it.get_group());
      if (lid == 0 && valid[0]) {
        bool distinct = true;
        for (unsigned i = 0; i < PROOF; ++i)
          for (unsigned j = i + 1; j < PROOF; ++j)
            if (next_slot[i] == next_slot[j]) distinct = false;
        if (distinct) {
          const uint32_t idx = dev_atomic_u32(d_cand_count[0]).fetch_add(1u);
          if (idx < cand_cap)
            for (unsigned i = 0; i < PROOF; ++i) d_cand[idx].leaves[i] = next_slot[i];
        }
      }
    });
  });
}

#if defined(MOM_EQ_FORWARD_MAP)
constexpr uint32_t L0_INVERT_HEADS = NBUCKETS;
inline uint32_t l0_invert_hash(uint32_t x) {
  return x >> 14;
}

class ZelHashL0InvertSetupKernel;
class ZelHashL0InvertKernel;
static sycl::event submit_l0_invert(
  sycl::queue& q, MomKernelBundle& kb, const uint32_t* d_forward, EqCandidate* d_cand,
  uint32_t target_count, uint32_t* d_target, uint32_t* d_next, uint32_t* d_head
) {
  constexpr unsigned WG = 256;
  q.memset(d_head, 0xff, (size_t)L0_INVERT_HEADS * sizeof(uint32_t));
  uint32_t* const output = reinterpret_cast<uint32_t*>(d_cand);
  q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    h.parallel_for<ZelHashL0InvertSetupKernel>(
      sycl::nd_range<1>(sycl::range<1>(((size_t)target_count + WG - 1u) / WG * WG),
                        sycl::range<1>(WG)),
      [=](sycl::nd_item<1> it) {
        const uint32_t i = (uint32_t)it.get_global_id(0);
        if (i >= target_count) return;
        const uint32_t key = output[i];
        d_target[i] = key;
        output[i] = 0xffffffffu;
        d_next[i] = dev_atomic_u32(d_head[l0_invert_hash(key)]).exchange(i);
      });
  });
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    h.parallel_for<ZelHashL0InvertKernel>(
      sycl::nd_range<1>(sycl::range<1>(NUM_ENTRIES), sycl::range<1>(WG)),
      [=](sycl::nd_item<1> it) {
        const uint32_t leaf = (uint32_t)it.get_global_id(0);
        const uint32_t key = d_forward[leaf];
        if (key == 0xffffffffu) return;
        for (uint32_t node = d_head[l0_invert_hash(key)];
             node != 0xffffffffu; node = d_next[node]) {
          if (d_target[node] == key) output[node] = leaf;
        }
      });
  });
}
#endif

// ===========================================================================================
// Per-device state: queue + bundle + USM arenas (modeled on FishState). The Wagner arenas
// are allocated lazily on the first mining solve to prove headroom; the M1 test path only needs the
// small base-state + rows buffers.
// ===========================================================================================
static unsigned device_slot_capacity(const sycl::device& device, bool gpu_compact) {
  // Preserve the statistically selected fast capacity when it fits. On smaller or conservatively
  // reported devices, derive a reduced aligned stride from both total memory and the largest legal
  // allocation instead of recognizing individual products or failing at the first arena allocation.
  uint64_t capacity = MAX_SLOTS;
  const uint64_t global_mem = device.get_info<sycl::info::device::global_mem_size>();
  const uint64_t max_alloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
  const uint64_t reserve = std::min<uint64_t>(1ull << 30, global_mem / 8u);
  const uint64_t arena_budget = global_mem > reserve ? global_mem - reserve : global_mem;

  uint64_t words_per_slot = 0;
  for (unsigned level = 0; level < NSTORED_LEVELS; ++level) {
    const uint64_t level_words = level_u32(level, gpu_compact);
    words_per_slot += level_words;
    if (max_alloc)
      capacity = std::min(capacity, max_alloc / (NBUCKETS * level_words * sizeof(uint32_t)));
  }
  if (level_u32(0, gpu_compact) == 4u) {
    ++words_per_slot; // separate level-0 recovery index
    if (max_alloc)
      capacity = std::min(capacity, max_alloc / (NBUCKETS * sizeof(uint32_t)));
  }
  if (arena_budget)
    capacity = std::min(capacity, arena_budget / (NBUCKETS * words_per_slot * sizeof(uint32_t)));

  constexpr unsigned SLOT_ALIGNMENT = 16u;
  const unsigned aligned = round_down(static_cast<unsigned>(
    std::min<uint64_t>(capacity, std::numeric_limits<unsigned>::max())), SLOT_ALIGNMENT);
  const unsigned mean = ceil_div_u64(NUM_ENTRIES, NBUCKETS);
  const unsigned minimum = round_up(mean + ceil_sqrt(mean), SLOT_ALIGNMENT);
  if (aligned < minimum)
    throw std::string("zelhash device has insufficient allocation capacity");

  // Developer-only sweep hook: production uses the device-derived value above. Keeping the
  // override bounded and aligned lets additional architectures validate the heuristic with one
  // binary instead of rebuilding every candidate stride.
  if (const char* value = std::getenv("MOM_ZELHASH_SLOTS"); value && *value) {
    char* end = nullptr;
    const unsigned long requested = std::strtoul(value, &end, 10);
    if (!end || *end || requested < minimum || requested > aligned ||
        requested % SLOT_ALIGNMENT != 0u)
      throw std::string("MOM_ZELHASH_SLOTS must be a supported 16-slot-aligned capacity");
    return static_cast<unsigned>(requested);
  }
  return aligned;
}

class ZelHashState {
public:
  sycl::device device; sycl::queue queue; std::unique_ptr<MomKernelBundle> bundle;
  bool shared_io, gpu_compact;
  unsigned slot_capacity;
  // M1 I/O
  uint64_t* base_h = nullptr;     // 8 words
  uint8_t*  base_pending = nullptr;// 12 bytes
  uint8_t*  rows = nullptr;       // EQUIHASH_TEST_ROWS * HASH_LENGTH
  // Four compact bucket levels plus counters and tiny recovery lists. The final round emits only
  // zero-field pairs; their parent trees are reconstructed after that filter.
  uint32_t* level[NSTORED_LEVELS] = {}; // level[L]: materialized bucketed collision records
  uint32_t* l0_index       = nullptr;  // CUDA compact level-0 leaf mapping (cold recovery stream)
  uint32_t* nslots         = nullptr;  // NSTORED_LEVELS * NBUCKETS atomic counters
  EqCandidateRef* ref      = nullptr;  // zero-node references awaiting compact recovery
  uint32_t* ref_count      = nullptr;  // shared: zero-node reference counter
  EqCandidate* cand        = nullptr;  // final candidate list (recovered leaf indices)
  uint32_t* cand_count     = nullptr;  // shared: atomic candidate counter
#if defined(MOM_EQ_FORWARD_MAP)
  uint32_t* invert_target  = nullptr;  // cold leaf-position inversion scratch
  uint32_t* invert_next    = nullptr;
  uint32_t* invert_head    = nullptr;
#endif
  static constexpr uint32_t CAND_CAP = 4096;
  bool arenas_built = false; std::mutex mutex;

  explicit ZelHashState(const std::string& dev_str)
    : device(get_dev(dev_str)), queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !mom_has_usm_device(device)),
      gpu_compact(mom_is_cuda(device) || mom_is_hip(device)),
      slot_capacity(device_slot_capacity(device, gpu_compact)) {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device)))
      throw std::string("zelhash SYCL device does not support required allocations");
    bundle = std::make_unique<MomKernelBundle>(mom_get_exec_bundle(queue.get_context()));
  }
  ~ZelHashState() { sycl_cleanup_noexcept("zelhash", [&] { queue.wait_and_throw(); free_all(); }); }
  template <typename T> T* alloc(size_t n) {
    return shared_io ? sycl::malloc_shared<T>(n, queue) : sycl::malloc_device<T>(n, queue);
  }
  void free_ptr(auto*& p) { if (p) sycl::free(p, queue); p = nullptr; }
  void free_all() {
    free_ptr(base_h); free_ptr(base_pending); free_ptr(rows);
    for (unsigned L = 0; L < NSTORED_LEVELS; ++L) free_ptr(level[L]);
    free_ptr(l0_index); free_ptr(nslots); free_ptr(ref); free_ptr(ref_count); free_ptr(cand); free_ptr(cand_count);
#if defined(MOM_EQ_FORWARD_MAP)
    free_ptr(invert_target); free_ptr(invert_next); free_ptr(invert_head);
#endif
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
    if (!base_h || !base_pending || !rows) throw std::string("Can't allocate zelhash I/O buffers");
  }
  // Allocate the four materialized Wagner levels, per-level counters, and bounded recovery output.
  void ensure_arenas(bool log) {
    if (arenas_built) return;
    const uint64_t t0 = now_ms();
    free_all_arenas();
    const size_t slots_per_level = (size_t)NBUCKETS * slot_capacity;
    size_t total = 0;
    for (unsigned L = 0; L < NSTORED_LEVELS; ++L) {
      const size_t lvl_u32 = slots_per_level * level_u32(L, gpu_compact);
      level[L] = alloc<uint32_t>(lvl_u32); total += lvl_u32 * 4;
      if (!level[L]) throw std::string("Can't allocate zelhash level arena");
    }
    if (level_u32(0, gpu_compact) == 4u) {
      l0_index = alloc<uint32_t>(slots_per_level);
      total += slots_per_level * sizeof(uint32_t);
      if (!l0_index) throw std::string("Can't allocate zelhash level-0 index arena");
    }
    nslots     = alloc<uint32_t>((size_t)NSTORED_LEVELS * NBUCKETS);
    total += (size_t)NSTORED_LEVELS * NBUCKETS * 4;
    ref        = alloc<EqCandidateRef>(CAND_CAP);
    ref_count  = sycl::malloc_shared<uint32_t>(1, queue);
    cand       = alloc<EqCandidate>(CAND_CAP);
    cand_count = sycl::malloc_shared<uint32_t>(1, queue);     // host reads the candidate count
#if defined(MOM_EQ_FORWARD_MAP)
    if (gpu_compact) {
      constexpr size_t targets = (size_t)CAND_CAP * (1u << K);
      invert_target = alloc<uint32_t>(targets);
      invert_next = alloc<uint32_t>(targets);
      invert_head = alloc<uint32_t>(L0_INVERT_HEADS);
      total += (targets * 2u + L0_INVERT_HEADS) * sizeof(uint32_t);
    }
#endif
    if (!nslots || !ref || !ref_count || !cand || !cand_count)
      throw std::string("Can't allocate zelhash counters/candidates");
#if defined(MOM_EQ_FORWARD_MAP)
    if (gpu_compact && (!invert_target || !invert_next || !invert_head))
      throw std::string("Can't allocate zelhash recovery scratch");
#endif
    // Touch the counter array (lazily mapped USM device allocs surface OOM here, not mid-kernel).
    sycl_wait_and_throw(queue.memset(nslots, 0,
      (size_t)NSTORED_LEVELS * NBUCKETS * sizeof(uint32_t)), device);
    arenas_built = true;
    if (log) { std::fprintf(stderr,
      "zelhash Wagner arenas (~%.1f GiB, %u slots/bucket) allocated (%llu ms)\n",
      (double)total / (1024.0 * 1024.0 * 1024.0), slot_capacity,
      (unsigned long long)(now_ms() - t0)); std::fflush(stderr); }
  }
  void free_all_arenas() {
    for (unsigned L = 0; L < NSTORED_LEVELS; ++L) free_ptr(level[L]);
    free_ptr(l0_index); free_ptr(nslots); free_ptr(ref); free_ptr(ref_count); free_ptr(cand); free_ptr(cand_count);
#if defined(MOM_EQ_FORWARD_MAP)
    free_ptr(invert_target); free_ptr(invert_next); free_ptr(invert_head);
#endif
  }
  static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }
};

using ZelHashStateMap = std::map<std::string, std::unique_ptr<ZelHashState>>;

static ZelHashStateMap& zelhash_states() {
  static ZelHashStateMap* const states = new ZelHashStateMap;
  return *states;
}

static std::mutex& zelhash_states_mutex() {
  static std::mutex* const mutex = new std::mutex;
  return *mutex;
}

static ZelHashState& zelhash_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(zelhash_states_mutex());
  auto& state = zelhash_states()[dev_str];
  if (!state) state = std::make_unique<ZelHashState>(dev_str);
  return *state;
}

void zelhash_cleanup_states() noexcept {
  try {
    std::lock_guard<std::mutex> lock(zelhash_states_mutex());
    zelhash_states().clear();
  } catch (...) {
    std::fprintf(stderr, "zelhash: ordered SYCL cleanup failed\n");
  }
}

template <bool GPU_COMPACT>
static void run_collision_rounds(ZelHashState& state, unsigned wg, bool wait_stages,
                                 bool prof, uint64_t t_round[4]) {
  sycl::queue& q = state.queue;
  auto r0 = submit_round<0, GPU_COMPACT>(q, *state.bundle, state.level[0], state.level[1],
    state.nslots + 0u * NBUCKETS, state.nslots + 1u * NBUCKETS, wg, state.slot_capacity);
  if (wait_stages) sycl_wait_and_throw(r0, state.device);
  if (prof) t_round[0] = ZelHashState::now_ms();
  auto r1 = submit_round<1, GPU_COMPACT>(q, *state.bundle, state.level[1], state.level[2],
    state.nslots + 1u * NBUCKETS, state.nslots + 2u * NBUCKETS, wg, state.slot_capacity);
  if (wait_stages) sycl_wait_and_throw(r1, state.device);
  if (prof) t_round[1] = ZelHashState::now_ms();
  auto r2 = submit_round<2, GPU_COMPACT>(q, *state.bundle, state.level[2], state.level[3],
    state.nslots + 2u * NBUCKETS, state.nslots + 3u * NBUCKETS, wg, state.slot_capacity);
  if (wait_stages) sycl_wait_and_throw(r2, state.device);
  if (prof) t_round[2] = ZelHashState::now_ms();
  auto r3 = submit_round<3, GPU_COMPACT>(q, *state.bundle, state.level[3], nullptr,
    state.nslots + 3u * NBUCKETS, nullptr, wg, state.slot_capacity,
    state.ref, state.ref_count, ZelHashState::CAND_CAP);
  if (wait_stages) sycl_wait_and_throw(r3, state.device);
  if (prof) t_round[3] = ZelHashState::now_ms();
}

// ===========================================================================================
// Host-side finishing: the compact device recovery already returned each candidate's 16 leaf indices,
// so the host only canonicalises Zcash IndicesBefore order and CompressArray(26)s the proof. No arena
// copyback is required.
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

} // namespace mom_zelhash

using namespace mom_zelhash;

int zelhash(
  const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size,
  uint8_t* const solution_out, uint64_t* const /*pnonce*/, const uint8_t* const target,
  const unsigned /*intensity*/, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < HEADER_LEN) throw std::string("Bad zelhash input length");
  ZelHashState& state = zelhash_state(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_io();

  // Host: personalized "ZelProof" base state over all 140 header bytes (base_h/base_pending are shared).
  const BaseState bs = make_base_state(input);
  std::memcpy(state.base_h, bs.h, 8 * sizeof(uint64_t));
  std::memcpy(state.base_pending, bs.pending, 12);

  // is_test path: the default runs the gen-kernel cross-check (gen-rows dump). Setting
  // MOM_ZELHASH_SOLVE switches it to the full Wagner solve so the standalone checker can assert the
  // block-400000 solution is found. (Mining, !is_test, always runs the solver.)
  const bool solve_in_test = is_test && std::getenv("MOM_ZELHASH_SOLVE") != nullptr;

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
          std::fprintf(stderr, "zelhash M1: optimized scan != naive at entry %zu\n", e);
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
  // Per-phase profiler: MOM_ZELHASH_PROF prints a gen-fill / per-round / device- and host-recovery
  // ms breakdown even under bench (where `log` is off). It inserts an extra barrier + per-round host
  // waits, so it is a SEPARATE gate from MOM_ZELHASH_PERF (the Sol/s logger) -- enabling it perturbs
  // throughput slightly, so leave it off when measuring Sol/s.
  static const bool prof = std::getenv("MOM_ZELHASH_PROF") != nullptr;
  const uint64_t t_start = ZelHashState::now_ms();
  sycl::queue& q = state.queue;

  // A fixed-stage CUDA chain avoids redundant event waits: it is essential on Windows and gives a
  // smaller repeatable gain on Linux with both supported CUDA compilers. Linux HIP instead loses
  // about 1%, and generic OpenCL retains its established waits. Keep a diagnostic opt-out.
  static const bool cuda_async_enabled = [] {
    const char* value = std::getenv("MOM_ZELHASH_CUDA_ASYNC");
    return !value || (std::strcmp(value, "0") && std::strcmp(value, "false"));
  }();
#if defined(_WIN32)
  // Windows event completion has high host/driver overhead on the measured Level Zero and HIP
  // backends. CUDA gains over 60% with DPC++ and AdaptiveCpp after exact proofs on both.
  const bool async_chain = !prof &&
    (sycl_is_level_zero_gpu(state.device) || mom_is_hip(state.device) ||
     (mom_is_cuda(state.device) && cuda_async_enabled));
#else
  const bool async_chain = !prof &&
    (sycl_is_level_zero_gpu(state.device) || (mom_is_cuda(state.device) && cuda_async_enabled));
#endif
  const bool wait_stages = !async_chain;

  // The queue is in-order and no round needs a host-sized handoff. In normal operation enqueue the
  // reset, generation, four collision rounds, and bounded recovery as one chain. The completed
  // recovery is the single synchronization point before its tiny shared counter is read.
  // Profiling deliberately waits per stage.
  *state.ref_count = 0;
  *state.cand_count = 0;
  auto reset = q.memset(state.nslots, 0,
    (size_t)NSTORED_LEVELS * NBUCKETS * sizeof(uint32_t));
  if (wait_stages) sycl_wait_and_throw(reset, state.device);

  // The block-local pair list is distributed at a compile-time fixed count per lane. The work-group
  // size tracks the selected bucket geometry and amortizes each super-bucket across roughly eight
  // rows per lane.
  constexpr unsigned wg = ROUND_WG;

  // Level 0: gen-fill 2^26 entries into round-0 buckets (field-0 bucketing).
  const uint64_t t_gen0 = ZelHashState::now_ms();
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
  auto gen = state.gpu_compact
    ? submit_gen_fill<true>(q, *state.bundle, state.base_h, state.base_pending,
        state.level[0], state.l0_index, state.nslots + 0u * NBUCKETS, state.slot_capacity)
    : submit_gen_fill<false>(q, *state.bundle, state.base_h, state.base_pending,
        state.level[0], nullptr, state.nslots + 0u * NBUCKETS, state.slot_capacity);
#else
  auto gen = submit_gen_fill<false>(q, *state.bundle, state.base_h, state.base_pending,
    state.level[0], nullptr, state.nslots + 0u * NBUCKETS, state.slot_capacity);
#endif
  if (wait_stages) sycl_wait_and_throw(gen, state.device);
  // Rounds 0..3: collide field R over (bucket, RESTBITS), XOR -> level R+1.
  const uint64_t t_r0 = ZelHashState::now_ms();
  uint64_t t_round[4] = {0,0,0,0};
#if defined(MOM_SYCL_HAS_CUDA)
  // The combined Intel+CUDA build must retain both layouts and select by the actual queue backend.
  if (state.gpu_compact) run_collision_rounds<true>(state, wg, wait_stages, prof, t_round);
  else                  run_collision_rounds<false>(state, wg, wait_stages, prof, t_round);
#elif defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_ADAPTIVECPP)
  run_collision_rounds<true>(state, wg, wait_stages, prof, t_round);
#else
  // oneAPI Intel-only artifact: do not instantiate the bad-on-Xe 16-byte writer at all.
  run_collision_rounds<false>(state, wg, wait_stages, prof, t_round);
#endif
  const uint64_t t_rounds_end = ZelHashState::now_ms();

  // The fused round-3 kernel already retained only final-field-zero pairs.
  const uint64_t t_final = ZelHashState::now_ms();

  EqLevels lv;
  for (unsigned L = 0; L < NSTORED_LEVELS; ++L) lv.level[L] = state.level[L];
  lv.l0_index = state.l0_index;
  auto submit_recovery = [&](uint32_t count, const uint32_t* device_count) {
#if defined(MOM_SYCL_HAS_CUDA)
    if (state.gpu_compact)
      return submit_recover<true>(q, *state.bundle, lv, state.nslots, state.ref, count,
        state.cand, state.cand_count, ZelHashState::CAND_CAP, wg, state.slot_capacity, device_count);
    return submit_recover<false>(q, *state.bundle, lv, state.nslots, state.ref, count,
      state.cand, state.cand_count, ZelHashState::CAND_CAP, wg, state.slot_capacity, device_count);
#elif defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_ADAPTIVECPP)
    return submit_recover<true>(q, *state.bundle, lv, state.nslots, state.ref, count,
      state.cand, state.cand_count, ZelHashState::CAND_CAP, wg, state.slot_capacity, device_count);
#else
    return submit_recover<false>(q, *state.bundle, lv, state.nslots, state.ref, count,
      state.cand, state.cand_count, ZelHashState::CAND_CAP, wg, state.slot_capacity, device_count);
#endif
  };

  if (async_chain) {
    // Queue the maximum bounded grid behind round 3 and let each work-group consult the device
    // counter. This retains one final completion point instead of a host wait before recovery.
    sycl_wait_and_throw(submit_recovery(ZelHashState::CAND_CAP, state.ref_count), state.device);
  } else {
    const uint32_t ref_n = std::min<uint32_t>(*state.ref_count, ZelHashState::CAND_CAP);
    if (ref_n != 0)
      sycl_wait_and_throw(submit_recovery(ref_n, nullptr), state.device);
  }
#if defined(MOM_EQ_FORWARD_MAP)
  if (state.gpu_compact) {
    const uint32_t recovered = std::min<uint32_t>(*state.cand_count, ZelHashState::CAND_CAP);
    if (recovered != 0) {
      sycl_wait_and_throw(submit_l0_invert(q, *state.bundle, state.l0_index, state.cand,
        recovered * (1u << K), state.invert_target, state.invert_next, state.invert_head),
        state.device);
    }
  }
#endif
  const uint64_t t_rounds = ZelHashState::now_ms();
  if (prof) {
    std::fprintf(stderr,
      "[eq-prof] gen-fill %llums | r0 %llu r1 %llu r2 %llu r3 %llu (rounds %llums) | recovery %llums\n",
      (unsigned long long)(t_r0 - t_gen0),
      (unsigned long long)(t_round[0] - t_r0), (unsigned long long)(t_round[1] - t_round[0]),
      (unsigned long long)(t_round[2] - t_round[1]), (unsigned long long)(t_round[3] - t_round[2]),
      (unsigned long long)(t_rounds_end - t_r0), (unsigned long long)(t_rounds - t_final));
    std::fflush(stderr);
  }

  // ---- Pull only the recovered candidate list to the host. ----
  const uint64_t t_recov0 = ZelHashState::now_ms();
  uint32_t cand_count = *state.cand_count;
  const uint32_t cand_n = std::min<uint32_t>(cand_count, ZelHashState::CAND_CAP);

  if (log) {
    // Per-level survivor counts are diagnostic-only. Do not add a 640 KiB device-to-host transfer and
    // queue synchronization to every benchmark/mining solve merely to populate the verbose report.
    std::vector<uint32_t> ns_host((size_t)NSTORED_LEVELS * NBUCKETS);
    sycl_wait_and_throw(q.memcpy(ns_host.data(), state.nslots, ns_host.size() * sizeof(uint32_t)), state.device);
    for (unsigned L = 0; L < NSTORED_LEVELS; ++L) {
      uint64_t sum = 0, overflow = 0;
      for (unsigned bkt = 0; bkt < NBUCKETS; ++bkt) {
        const uint32_t c = ns_host[(size_t)L * NBUCKETS + bkt];
        sum += std::min<uint32_t>(c, state.slot_capacity);
        if (c > state.slot_capacity) overflow += c - state.slot_capacity;
      }
      std::fprintf(stderr, "zelhash level %u survivors=%llu dropped=%llu (%.4f%%)\n",
        L, (unsigned long long)sum, (unsigned long long)overflow,
        100.0 * (double)overflow / (double)(sum + overflow ? sum + overflow : 1));
    }
    std::fprintf(stderr, "zelhash zero-nodes=%u candidates=%u (solve %llums)\n",
      *state.ref_count, cand_count,
      (unsigned long long)(t_rounds - t_gen0));
    std::fflush(stderr);
  }

  int n_solutions = 0;
  size_t distinct_count = 0;   // distinct valid Equihash proofs found this solve (pre-target-filter)
  if (cand_n) {
    // Device recovery already produced the 16 leaf indices, so copy back only this tiny list.
    std::vector<EqCandidate> cand_h(cand_n);
    sycl_wait_and_throw(q.memcpy(cand_h.data(), state.cand, cand_n * sizeof(EqCandidate)), state.device);

    // De-dup distinct solutions (the same proof can surface via several candidates).
    std::vector<std::array<uint8_t, 52>> seen;
    for (uint32_t ci = 0; ci < cand_n; ++ci) {
      // Each candidate is one recovered final-zero pair with 16 distinct leaf indices.
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
      std::fprintf(stderr, "zelhash %s solutions=%d (of %zu distinct):\n",
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

  // MOM_ZELHASH_PERF: Sol/s logger (mirrors MOM_FISHHASH_PERF). Accumulates the number of distinct
  // valid proofs found (not nonces, and BEFORE the target filter so bench/zero-target still measures
  // throughput) over a >=2s window; the solver averages ~1.88 solutions per nonce.
  static const bool perf_log = std::getenv("MOM_ZELHASH_PERF") != nullptr;
  if (perf_log && !is_test) {
    static uint64_t acc_sols = 0; static uint64_t acc_solves = 0;
    static auto t0 = std::chrono::steady_clock::now();
    acc_sols += (uint64_t)distinct_count;
    ++acc_solves;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec >= 2.0) {
      std::fprintf(stderr, "[zelhash] %.2f Sol/s, %.2f solve/s (%s)\n",
        (double)acc_sols / sec, (double)acc_solves / sec, dev_str.c_str());
      std::fflush(stderr);
      acc_sols = 0; acc_solves = 0; t0 = std::chrono::steady_clock::now();
    }
  }

  if (prof) {
    std::fprintf(stderr, "[eq-prof] host-recovery %llums (cand=%u) | total %llums\n",
      (unsigned long long)(ZelHashState::now_ms() - t_recov0), cand_count,
      (unsigned long long)(ZelHashState::now_ms() - t_start));
    std::fflush(stderr);
  }
  if (log) { std::fprintf(stderr, "zelhash solve done (%llu ms, %d target-passing, %zu distinct)\n",
    (unsigned long long)(ZelHashState::now_ms() - t_start), n_solutions, distinct_count); std::fflush(stderr); }

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
