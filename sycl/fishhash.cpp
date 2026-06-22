// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
//
// FishHash (Iron Fish IRON / Karlsen KLS) GPU search kernel. ASIC-resistant, memory-hard, Ethash-derived.
// Per hash: blake3(header) -> 64B seed -> 32 dataset accesses (3x 128B fetches, mix=f0*f1+f2 u64) ->
// collapse -> blake3(seed||mix_hash) -> 32B. DAG is FIXED (not epoch-based): 1.18M x 64B light cache ->
// 37.7M x 128B (4.6 GB) dataset, both from a fixed seed. Ported bit-exact from github.com/iron-fish/
// fish-hash (cpp/FishHash.cpp + 3rdParty/{blake3,keccak}); validated offline (light_cache[0], blake3
// seed, final hash). This first version uses LAZY lookup (computes dataset items from the 72 MB light
// cache on the fly) -- correct but slow; the full-DAG fast path is built when intensity warrants it.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
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

namespace mom_fishhash {

constexpr uint32_t MAX_OUTPUTS              = 15;
constexpr uint32_t FNV_PRIME               = 0x01000193u;
constexpr int      LIGHT_CACHE_NUM_ITEMS   = 1179641;     // x 64B  = 72 MiB
constexpr int      FULL_DATASET_NUM_ITEMS  = 37748717;    // x 128B = 4.6 GiB
constexpr int      FULL_DATASET_ITEM_PARENTS = 512;
constexpr int      NUM_DATASET_ACCESSES    = 32;
constexpr int      LIGHT_CACHE_ROUNDS      = 3;
// fixed FishHash seed (FishHash.cpp)
static constexpr uint8_t FISH_SEED[32] = {
  0xeb,0x01,0x63,0xae,0xf2,0xab,0x1c,0x5a, 0x66,0x31,0x0c,0x1c,0x14,0xd6,0x0f,0x42,
  0x55,0xa9,0xb3,0x9b,0x0e,0xdf,0x26,0x53, 0x98,0x44,0xf1,0x17,0xad,0x67,0x21,0x19 };

struct FishResult {
  uint32_t count;
  uint64_t nonce[MAX_OUTPUTS];
  uint8_t output[MAX_OUTPUTS][HASH_LEN];
};

// ---- LE load/store ----
inline uint32_t load32_le_dev(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline void store32_le_dev(uint8_t* p, uint32_t v) { p[0]=v; p[1]=v>>8; p[2]=v>>16; p[3]=v>>24; }
inline uint64_t load64_le_dev(const uint8_t* p) {
  return (uint64_t)load32_le_dev(p) | ((uint64_t)load32_le_dev(p + 4) << 32);
}

// ---- Keccak-f[1600] scalar (copied from etchash; IGC-safe) ----
inline uint64_t rotl64_dev(uint64_t v, unsigned s) { return (v << s) | (v >> (64 - s)); }
inline void keccakf1600_dev(uint64_t st[25]) {
  static constexpr uint64_t rndc[24] = {
    0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
    0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
    0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
    0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
    0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
    0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL };
  static constexpr unsigned rotc[24] = {1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
  static constexpr unsigned piln[24] = {10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
  for (unsigned r = 0; r < 24; ++r) {
    uint64_t bc[5];
    for (unsigned i = 0; i < 5; ++i) bc[i] = st[i]^st[i+5]^st[i+10]^st[i+15]^st[i+20];
    for (unsigned i = 0; i < 5; ++i) { const uint64_t t = bc[(i+4)%5]^rotl64_dev(bc[(i+1)%5],1);
      for (unsigned j = 0; j < 25; j += 5) st[j+i] ^= t; }
    uint64_t t = st[1];
    for (unsigned i = 0; i < 24; ++i) { const unsigned j = piln[i]; bc[0]=st[j]; st[j]=rotl64_dev(t,rotc[i]); t=bc[0]; }
    for (unsigned j = 0; j < 25; j += 5) { for (unsigned i=0;i<5;++i) bc[i]=st[j+i];
      for (unsigned i=0;i<5;++i) st[j+i] ^= (~bc[(i+1)%5]) & bc[(i+2)%5]; }
    st[0] ^= rndc[r];
  }
}
// Keccak-512 (original Keccak, 0x01 pad; rate 72B/9 words). in_len <= 72. Output 8 u64 (64 bytes).
inline void keccak512_dev(uint64_t out[8], const uint8_t* in, unsigned in_len) {
  uint8_t buf[72];
  for (unsigned i = 0; i < 72; ++i) buf[i] = (i < in_len) ? in[i] : 0;
  buf[in_len] = 0x01;
  buf[71] |= 0x80;
  uint64_t st[25];
  for (unsigned i = 0; i < 9; ++i) st[i] = load64_le_dev(buf + i*8);
  for (unsigned i = 9; i < 25; ++i) st[i] = 0;
  keccakf1600_dev(st);
  for (unsigned i = 0; i < 8; ++i) out[i] = st[i];
}
inline void keccak512_words_dev(uint64_t io[8]) {  // keccak512 of a 64-byte (8 u64) buffer in place
  uint8_t buf[64];
  for (unsigned i = 0; i < 8; ++i) for (unsigned b = 0; b < 8; ++b) buf[i*8+b] = (uint8_t)(io[i] >> (8*b));
  keccak512_dev(io, buf, 64);
}

// ---- BLAKE3 (single chunk, input <= 1024 bytes; ported from blake3_portable.c) ----
static constexpr uint32_t B3_IV[8] = {0x6A09E667u,0xBB67AE85u,0x3C6EF372u,0xA54FF53Au,0x510E527Fu,0x9B05688Cu,0x1F83D9ABu,0x5BE0CD19u};
static constexpr uint8_t B3_MSG[7][16] = {
  {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
  {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},{10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
  {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},{9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
  {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13} };
inline uint32_t rotr32_dev(uint32_t w, uint32_t c) { return (w >> c) | (w << (32 - c)); }
inline void b3_g(uint32_t* s, unsigned a, unsigned b, unsigned c, unsigned d, uint32_t x, uint32_t y) {
  s[a]=s[a]+s[b]+x; s[d]=rotr32_dev(s[d]^s[a],16); s[c]=s[c]+s[d]; s[b]=rotr32_dev(s[b]^s[c],12);
  s[a]=s[a]+s[b]+y; s[d]=rotr32_dev(s[d]^s[a],8);  s[c]=s[c]+s[d]; s[b]=rotr32_dev(s[b]^s[c],7);
}
inline void b3_round(uint32_t st[16], const uint32_t* m, unsigned r) {
  const uint8_t* sc = B3_MSG[r];
  b3_g(st,0,4,8,12, m[sc[0]], m[sc[1]]); b3_g(st,1,5,9,13, m[sc[2]], m[sc[3]]);
  b3_g(st,2,6,10,14,m[sc[4]], m[sc[5]]); b3_g(st,3,7,11,15,m[sc[6]], m[sc[7]]);
  b3_g(st,0,5,10,15,m[sc[8]], m[sc[9]]); b3_g(st,1,6,11,12,m[sc[10]],m[sc[11]]);
  b3_g(st,2,7,8,13, m[sc[12]],m[sc[13]]);b3_g(st,3,4,9,14, m[sc[14]],m[sc[15]]);
}
// compress -> 64-byte xof output. cv[8], block[64], block_len, counter, flags.
inline void b3_compress_xof(const uint32_t cv[8], const uint8_t block[64], uint8_t block_len,
                            uint64_t counter, uint8_t flags, uint8_t out[64]) {
  uint32_t m[16]; for (unsigned i = 0; i < 16; ++i) m[i] = load32_le_dev(block + 4*i);
  uint32_t st[16] = { cv[0],cv[1],cv[2],cv[3],cv[4],cv[5],cv[6],cv[7],
                      B3_IV[0],B3_IV[1],B3_IV[2],B3_IV[3],
                      (uint32_t)counter,(uint32_t)(counter>>32),(uint32_t)block_len,(uint32_t)flags };
  for (unsigned r = 0; r < 7; ++r) b3_round(st, m, r);
  for (unsigned i = 0; i < 8; ++i) { store32_le_dev(out+i*4, st[i]^st[i+8]); store32_le_dev(out+(i+8)*4, st[i+8]^cv[i]); }
}
inline void b3_compress_inplace(uint32_t cv[8], const uint8_t block[64], uint8_t block_len,
                                uint64_t counter, uint8_t flags) {
  uint8_t out[64]; b3_compress_xof(cv, block, block_len, counter, flags, out);
  for (unsigned i = 0; i < 8; ++i) cv[i] = load32_le_dev(out + i*4);
}
// single-chunk blake3, out_len <= 64. flags: CHUNK_START=1, CHUNK_END=2, ROOT=8.
inline void blake3_dev(uint8_t* out, unsigned out_len, const uint8_t* in, unsigned in_len) {
  uint32_t cv[8]; for (unsigned i = 0; i < 8; ++i) cv[i] = B3_IV[i];
  unsigned pos = 0; bool first = true;
  // full blocks except the last
  while (in_len - pos > 64) {
    uint8_t blk[64]; for (unsigned i = 0; i < 64; ++i) blk[i] = in[pos+i];
    b3_compress_inplace(cv, blk, 64, 0, first ? 1u : 0u);
    pos += 64; first = false;
  }
  // last block (1..64 bytes; or 0 if in_len==0 -> one empty block)
  const unsigned last_len = in_len - pos;
  uint8_t blk[64]; for (unsigned i = 0; i < 64; ++i) blk[i] = (i < last_len) ? in[pos+i] : 0;
  const uint8_t flags = (uint8_t)((first ? 1u : 0u) | 2u | 8u);  // [CHUNK_START] | CHUNK_END | ROOT
  uint8_t xof[64]; b3_compress_xof(cv, blk, (uint8_t)last_len, 0, flags, xof);
  for (unsigned i = 0; i < out_len; ++i) out[i] = xof[i];
}

inline uint32_t fnv1_dev(uint32_t u, uint32_t v) { return (u * FNV_PRIME) ^ v; }

// ---- dataset item (calculate_dataset_item_1024): 128 bytes = 16 u64 = 32 u32 ----
// item_state for one 64-byte half. cache: light cache (LIGHT_CACHE_NUM_ITEMS x 8 u64).
inline void fish_item_half(const uint64_t* __restrict cache, uint32_t index, uint64_t mix[8]) {
  for (unsigned i = 0; i < 8; ++i) mix[i] = cache[(index % LIGHT_CACHE_NUM_ITEMS) * 8 + i];
  // mix.word32s[0] ^= index  (low 32 bits of mix[0])
  mix[0] ^= (uint64_t)index;
  keccak512_words_dev(mix);
  for (uint32_t round = 0; round < (uint32_t)FULL_DATASET_ITEM_PARENTS; ++round) {
    // t = fnv1(index ^ round, mix.word32s[round % 16]);  word32s[w] = (w even) low : high of mix[w/2]
    const unsigned w = round % 16;
    const uint32_t mw = (w & 1) ? (uint32_t)(mix[w/2] >> 32) : (uint32_t)mix[w/2];
    const uint32_t t = fnv1_dev(index ^ round, mw);
    const uint32_t parent = t % (uint32_t)LIGHT_CACHE_NUM_ITEMS;
    // mix = fnv1(mix, cache[parent]) over 16 u32
    const uint64_t* pc = cache + (size_t)parent * 8;
    for (unsigned k = 0; k < 8; ++k) {
      const uint32_t lo = fnv1_dev((uint32_t)mix[k], (uint32_t)pc[k]);
      const uint32_t hi = fnv1_dev((uint32_t)(mix[k] >> 32), (uint32_t)(pc[k] >> 32));
      mix[k] = (uint64_t)lo | ((uint64_t)hi << 32);
    }
  }
  keccak512_words_dev(mix);
}
// item -> 16 u64 (128 bytes)
inline void fish_dataset_item(const uint64_t* __restrict cache, uint32_t index, uint64_t out[16]) {
  fish_item_half(cache, index * 2,     out);       // out[0..7]
  fish_item_half(cache, index * 2 + 1, out + 8);   // out[8..15]
}

inline bool meets_target_le_dev(const uint8_t out[32], const uint8_t* target) {
  for (int i = 31; i >= 0; --i) { if (out[i] == target[i]) continue; return out[i] < target[i]; }
  return true;
}
// Big-endian compare (Iron Fish: hashValue and set_target are both 256-bit BE, byte 0 = MSB).
inline bool meets_target_be_dev(const uint8_t out[32], const uint8_t* target) {
  for (int i = 0; i < 32; ++i) { if (out[i] == target[i]) continue; return out[i] < target[i]; }
  return true;
}
inline void store_fish_result(FishResult* r, uint64_t nonce, const uint8_t out[32]) {
  using ar = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;
  const uint32_t idx = ar(r->count).fetch_add(1);
  if (idx >= MAX_OUTPUTS) return;
  r->nonce[idx] = nonce;
  for (unsigned i = 0; i < 32; ++i) r->output[idx][i] = out[i];
}

// ---- Read-only-cache (LDG) DAG load + software prefetch (NVIDIA only) ----
// The FULL DAG-gather is random-access-LATENCY bound: 3 independent 128B item reads per access at
// data-dependent indices, and the 32 accesses are serial. d_dag is const __restrict, which on the
// CUDA backend *should* let the load go through the read-only data cache (ld.global.nc / LDG), but
// the SYCL->NVPTX lowering does not reliably emit .nc for this u64 gather. Force it: ld.global.nc
// keeps the line in the read-only/L1 texture cache (the DAG is read-only for the entire search),
// which on the LSU side widens the per-SM in-flight-miss window without contending for the regular
// global-load cache other warps use. The same 64 bits at the same address are returned -> identical
// value, identical hash. Off the nvptx device pass this is the plain dereference (unchanged
// Intel/CPU codegen, and unchanged for the lazy FULL=false path which never calls it).
inline uint64_t fish_dag_ld(const uint64_t* __restrict p) {
#if defined(__NVPTX__)
  uint64_t v;
  asm("ld.global.nc.u64 %0, [%1];" : "=l"(v) : "l"(p));
  return v;
#else
  return *p;
#endif
}
// Non-semantic hint: pull a DAG item's 128B line toward L1 as soon as its index is known, so the
// LSU overlaps that miss with the serial per-element fold below (and the prior access's tail).
// A prefetch never changes a loaded value, so the computed hash is unaffected; no-op off nvptx.
inline void fish_dag_prefetch(const uint64_t* __restrict p) {
#if defined(__NVPTX__)
  asm volatile("prefetch.global.L1 [%0];" :: "l"(p));
#else
  (void)p;
#endif
}

template <bool FULL> class FishKernel;

// Search kernel. FULL=true: gather 128B items from the prebuilt 4.6 GB DAG (fast, mining/bench).
// FULL=false: recompute each item from the 72 MB light cache (lazy; validation / low-mem only).
// d_input is the header; the 8-byte nonce goes at nonce_off (LE for the offline @32 test, BE @0 for
// live Iron Fish). target_be selects BE compare (Iron Fish) vs LE (offline test).
template <bool FULL>
static sycl::event submit_fishhash_search(
  sycl::queue& q, MOM_BUNDLE_T& kb, const uint8_t* d_input, unsigned input_size, uint64_t start_nonce,
  const uint64_t* __restrict d_cache, const uint64_t* __restrict d_dag, uint32_t intensity,
  const uint8_t* d_target, FishResult* d_result, bool is_test,
  unsigned nonce_off, bool nonce_be, bool target_be
) {
  constexpr unsigned WG = 128;  // search: latency-bound gather -> more in-flight subgroups (tuning vs SOTA)
  const size_t global = (static_cast<size_t>(intensity) + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    MOM_USE_BUNDLE(h, kb);
    h.parallel_for<FishKernel<FULL>>(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
      const uint32_t gid = (uint32_t)it.get_global_id(0);
      if (gid >= intensity) return;
      const uint64_t nonce = start_nonce + gid;

      uint8_t header[180];
      const unsigned hlen = input_size <= 180 ? input_size : 180;
      for (unsigned i = 0; i < hlen; ++i) header[i] = d_input[i];
      if (nonce_be) for (unsigned i = 0; i < 8; ++i) header[nonce_off + i] = (uint8_t)(nonce >> (8*(7-i)));  // BE (Iron Fish @0)
      else          for (unsigned i = 0; i < 8; ++i) header[nonce_off + i] = (uint8_t)(nonce >> (8*i));        // LE (offline test @32)

      uint8_t seed[64]; blake3_dev(seed, 64, header, hlen);

      // mix(1024b) = {seed, seed} as 32 u32
      uint32_t mix[32];
      for (unsigned i = 0; i < 16; ++i) mix[i] = load32_le_dev(seed + 4*i);
      for (unsigned i = 0; i < 16; ++i) mix[16 + i] = mix[i];

      const uint32_t M = (uint32_t)FULL_DATASET_NUM_ITEMS;
      for (int i = 0; i < NUM_DATASET_ACCESSES; ++i) {
        const uint32_t p0 = mix[0] % M, p1 = mix[4] % M, p2 = mix[8] % M;
        // Fold the fnv/mul-add per 16-u64 item element as it streams in, instead of materializing
        // f0/f1/f2[16] + f1w/f2w[32] all at once. Each output (mix[2j],mix[2j+1]) depends only on
        // f0[j]/f1[j]/f2[j] and its own old mix slot, so only 3 u64 stay live per element -- a much
        // smaller register working set that lets more warps reside per SM (latency hiding on the
        // random gather). Bit-identical to the original array form.
        uint64_t lazy0[16], lazy1[16], lazy2[16];
        if constexpr (!FULL) {
          fish_dataset_item(d_cache, p0, lazy0);
          fish_dataset_item(d_cache, p1, lazy1);
          fish_dataset_item(d_cache, p2, lazy2);
        }
#if defined(__NVPTX__)
        // NVIDIA (nvptx) only: for the FULL DAG path, issue all 48 independent 64-bit item loads up
        // front in a tight batch (no arithmetic interleaved) so the LSU keeps many gather misses in
        // flight per warp before any is consumed -- memory-level parallelism on the random 4.6 GB
        // gather, the structure's latency limiter. +19% on L4. Off the FULL path this block is
        // skipped (lazy items already live). The 48-u64 batch buffer spills registers on Intel
        // (spir64) and collapses throughput ~13x, so spir64 takes the interleaved #else path below.
        uint64_t b0[16], b1[16], b2[16];
        if constexpr (FULL) {
          const uint64_t* __restrict q0 = d_dag + (size_t)p0 * 16;
          const uint64_t* __restrict q1 = d_dag + (size_t)p1 * 16;
          const uint64_t* __restrict q2 = d_dag + (size_t)p2 * 16;
          for (unsigned j = 0; j < 16; ++j) b0[j] = fish_dag_ld(q0 + j);
          for (unsigned j = 0; j < 16; ++j) b1[j] = fish_dag_ld(q1 + j);
          for (unsigned j = 0; j < 16; ++j) b2[j] = fish_dag_ld(q2 + j);
        }
#endif
        for (unsigned j = 0; j < 16; ++j) {
          uint64_t f0j, f1j, f2j;
          if constexpr (FULL) {
#if defined(__NVPTX__)
            f0j = b0[j];
            f1j = b1[j];
            f2j = b2[j];
#else
            // Intel (spir64): read each item element directly inside the fold so only 3 u64 stay
            // live per iteration -- smaller register footprint => more resident warps.
            f0j = fish_dag_ld(d_dag + (size_t)p0 * 16 + j);
            f1j = fish_dag_ld(d_dag + (size_t)p1 * 16 + j);
            f2j = fish_dag_ld(d_dag + (size_t)p2 * 16 + j);
#endif
          } else {
            f0j = lazy0[j]; f1j = lazy1[j]; f2j = lazy2[j];
          }
          const uint32_t blo = fnv1_dev(mix[2*j],   (uint32_t)f1j);
          const uint32_t bhi = fnv1_dev(mix[2*j+1], (uint32_t)(f1j >> 32));
          const uint32_t clo = mix[2*j]   ^ (uint32_t)f2j;
          const uint32_t chi = mix[2*j+1] ^ (uint32_t)(f2j >> 32);
          const uint64_t b = (uint64_t)blo | ((uint64_t)bhi << 32);
          const uint64_t c = (uint64_t)clo | ((uint64_t)chi << 32);
          const uint64_t v = f0j * b + c;
          mix[2*j] = (uint32_t)v; mix[2*j+1] = (uint32_t)(v >> 32);
        }
      }
      uint8_t mix_hash[32];
      for (unsigned i = 0; i < 32; i += 4) {
        uint32_t h = fnv1_dev(fnv1_dev(fnv1_dev(mix[i], mix[i+1]), mix[i+2]), mix[i+3]);
        store32_le_dev(mix_hash + i, h);
      }
      uint8_t final_data[96];
      for (unsigned i = 0; i < 64; ++i) final_data[i] = seed[i];
      for (unsigned i = 0; i < 32; ++i) final_data[64 + i] = mix_hash[i];
      uint8_t final_hash[32]; blake3_dev(final_hash, 32, final_data, 96);

      const bool hit = target_be ? meets_target_be_dev(final_hash, d_target) : meets_target_le_dev(final_hash, d_target);
      if ((is_test && gid == 0) || hit)
        store_fish_result(d_result, nonce, final_hash);
    });
  });
}

// ---- KarlsenHashV2 (FishHashPlus): SAME 4.6 GB DAG, different index derivation + BLAKE3 wrapping.
// powHash = BLAKE3_32(prePow(32)||ts(8 LE)||32 zeros||nonce(8 LE), 80B); seed = powHash(32)||zeros(32);
// mix = seed||seed; 32 rounds (mixGroup-fold index); collapse 32->8 u32; final = BLAKE3_32(mix_hash).
// Input = 80-byte Kaspa-style blob, 8-byte nonce at offset 72 (same blob shape as kHeavyHash).
template <bool FULL> class KarlsenKernel;
template <bool FULL>
static sycl::event submit_karlsenhashv2_search(
  sycl::queue& q, MOM_BUNDLE_T& kb, const uint8_t* d_input, unsigned input_size, uint64_t start_nonce,
  const uint64_t* __restrict d_cache, const uint64_t* __restrict d_dag, uint32_t intensity,
  const uint8_t* d_target, FishResult* d_result, bool is_test
) {
  constexpr unsigned WG = 128;
  const size_t global = (static_cast<size_t>(intensity) + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    MOM_USE_BUNDLE(h, kb);
    h.parallel_for<KarlsenKernel<FULL>>(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
      const uint32_t gid = (uint32_t)it.get_global_id(0);
      if (gid >= intensity) return;
      const uint64_t nonce = start_nonce + gid;

      uint8_t pre[80];
      for (unsigned i = 0; i < 80; ++i) pre[i] = (i < input_size) ? d_input[i] : 0;
      for (unsigned i = 0; i < 8; ++i) pre[72 + i] = (uint8_t)(nonce >> (8*i));  // nonce@72 LE

      uint8_t powh[32]; blake3_dev(powh, 32, pre, 80);

      uint32_t mix[32];                                   // seed = powHash(32) || zeros(32); mix = seed||seed
      for (unsigned i = 0; i < 8; ++i) mix[i] = load32_le_dev(powh + 4*i);
      for (unsigned i = 8; i < 16; ++i) mix[i] = 0;
      for (unsigned i = 0; i < 16; ++i) mix[16 + i] = mix[i];

      const uint32_t M = (uint32_t)FULL_DATASET_NUM_ITEMS;
      for (uint32_t rnd = 0; rnd < (uint32_t)NUM_DATASET_ACCESSES; ++rnd) {
        uint32_t mg[8];
        for (unsigned c = 0; c < 8; ++c) mg[c] = mix[4*c] ^ mix[4*c+1] ^ mix[4*c+2] ^ mix[4*c+3];
        const uint32_t p0 = (mg[0] ^ mg[3] ^ mg[6]) % M;
        const uint32_t p1 = (mg[1] ^ mg[4] ^ mg[7]) % M;
        const uint32_t p2 = (mg[2] ^ mg[5] ^ rnd) % M;
        uint64_t f0[16], f1[16], f2[16];
        if constexpr (FULL) {
#if defined(__NVPTX__)
          // NVIDIA (nvptx) only: batch the 48 independent gather loads split per item so the LSU
          // keeps many random 4.6 GB DAG misses in flight per warp before consumption -- same
          // latency-hiding win as FishHash. Spills registers on Intel (spir64), which takes the
          // interleaved #else load below instead.
          const uint64_t* __restrict q0 = d_dag + (size_t)p0 * 16;
          const uint64_t* __restrict q1 = d_dag + (size_t)p1 * 16;
          const uint64_t* __restrict q2 = d_dag + (size_t)p2 * 16;
          fish_dag_prefetch(q0);
          fish_dag_prefetch(q1);
          fish_dag_prefetch(q2);
          for (unsigned k = 0; k < 16; ++k) f0[k] = fish_dag_ld(q0 + k);
          for (unsigned k = 0; k < 16; ++k) f1[k] = fish_dag_ld(q1 + k);
          for (unsigned k = 0; k < 16; ++k) f2[k] = fish_dag_ld(q2 + k);
#else
          // Intel (spir64): original interleaved load -- one element of each item per iteration.
          for (unsigned k = 0; k < 16; ++k) {
            f0[k] = fish_dag_ld(d_dag + (size_t)p0 * 16 + k);
            f1[k] = fish_dag_ld(d_dag + (size_t)p1 * 16 + k);
            f2[k] = fish_dag_ld(d_dag + (size_t)p2 * 16 + k);
          }
#endif
        } else {
          fish_dataset_item(d_cache, p0, f0); fish_dataset_item(d_cache, p1, f1); fish_dataset_item(d_cache, p2, f2);
        }
        uint32_t f1w[32], f2w[32];
        for (unsigned j = 0; j < 16; ++j) { f1w[2*j]=(uint32_t)f1[j]; f1w[2*j+1]=(uint32_t)(f1[j]>>32);
                                            f2w[2*j]=(uint32_t)f2[j]; f2w[2*j+1]=(uint32_t)(f2[j]>>32); }
        for (unsigned j = 0; j < 32; ++j) { f1w[j] = fnv1_dev(mix[j], f1w[j]); f2w[j] = mix[j] ^ f2w[j]; }
        for (unsigned j = 0; j < 16; ++j) {
          const uint64_t a = f0[j];
          const uint64_t b = (uint64_t)f1w[2*j] | ((uint64_t)f1w[2*j+1] << 32);
          const uint64_t c = (uint64_t)f2w[2*j] | ((uint64_t)f2w[2*j+1] << 32);
          const uint64_t v = a * b + c;
          mix[2*j] = (uint32_t)v; mix[2*j+1] = (uint32_t)(v >> 32);
        }
      }
      uint8_t mix_hash[32];
      for (unsigned i = 0; i < 32; i += 4) {
        uint32_t hh = fnv1_dev(fnv1_dev(fnv1_dev(mix[i], mix[i+1]), mix[i+2]), mix[i+3]);
        store32_le_dev(mix_hash + i, hh);
      }
      uint8_t final_hash[32]; blake3_dev(final_hash, 32, mix_hash, 32);

      if ((is_test && gid == 0) || meets_target_le_dev(final_hash, d_target))
        store_fish_result(d_result, nonce, final_hash);
    });
  });
}

// ---- GPU DAG generation: compute all 37.7M dataset items into the 4.6 GB buffer (once per device) ----
class FishDagKernel;
static sycl::event submit_fishhash_dag_gen(
  sycl::queue& q, MOM_BUNDLE_T& kb, const uint64_t* __restrict d_cache, uint64_t* __restrict d_dag,
  uint32_t start, uint32_t count
) {
  constexpr unsigned WG = 64;
  const size_t global = (static_cast<size_t>(count) + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    MOM_USE_BUNDLE(h, kb);
    h.parallel_for<FishDagKernel>(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
      const uint32_t i = (uint32_t)it.get_global_id(0);
      if (i >= count) return;
      const uint32_t idx = start + i;
      uint64_t item[16];
      fish_dataset_item(d_cache, idx, item);
      for (unsigned k = 0; k < 16; ++k) d_dag[(size_t)idx * 16 + k] = item[k];
    });
  });
}

// ---- HOST: build the 72 MB light cache (keccak512 chain + 3 rounds), == ethash light cache ----
static void keccak512_host(uint64_t out[8], const uint8_t* in, unsigned in_len) { keccak512_dev(out, in, in_len); }
static void build_light_cache_host(std::vector<uint64_t>& cache /* N*8 */) {
  cache.resize((size_t)LIGHT_CACHE_NUM_ITEMS * 8);
  keccak512_host(&cache[0], FISH_SEED, 32);
  for (int i = 1; i < LIGHT_CACHE_NUM_ITEMS; ++i) {
    uint8_t buf[64]; for (unsigned k = 0; k < 8; ++k) for (unsigned b = 0; b < 8; ++b) buf[k*8+b] = (uint8_t)(cache[(size_t)(i-1)*8+k] >> (8*b));
    keccak512_host(&cache[(size_t)i*8], buf, 64);
  }
  for (int q = 0; q < LIGHT_CACHE_ROUNDS; ++q) {
    for (int i = 0; i < LIGHT_CACHE_NUM_ITEMS; ++i) {
      const uint32_t t = (uint32_t)cache[(size_t)i*8];                 // word32s[0] = low 32 of word64s[0]
      const uint32_t v = t % (uint32_t)LIGHT_CACHE_NUM_ITEMS;
      const uint32_t w = (uint32_t)(LIGHT_CACHE_NUM_ITEMS + (i - 1)) % (uint32_t)LIGHT_CACHE_NUM_ITEMS;
      uint8_t x[64];
      for (unsigned k = 0; k < 8; ++k) { const uint64_t xr = cache[(size_t)v*8+k] ^ cache[(size_t)w*8+k];
        for (unsigned b = 0; b < 8; ++b) x[k*8+b] = (uint8_t)(xr >> (8*b)); }
      keccak512_host(&cache[(size_t)i*8], x, 64);
    }
  }
}

class FishState {
public:
  sycl::device device; sycl::queue queue; std::unique_ptr<MOM_BUNDLE_T> bundle; bool shared_io;
  uint8_t* input = nullptr; uint8_t* target = nullptr; uint64_t* cache = nullptr; uint64_t* dag = nullptr; FishResult* result = nullptr;
  unsigned input_cap = 0; bool cache_built = false; bool dag_built = false; std::mutex mutex;
  explicit FishState(const std::string& dev_str)
    : device(get_dev(dev_str)), queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !device.has(sycl::aspect::usm_device_allocations)) {
    if (!device.has(sycl::aspect::usm_shared_allocations) || (!device.is_cpu() && !device.has(sycl::aspect::usm_device_allocations)))
      throw std::string("fishhash SYCL device does not support required allocations");
    bundle = std::make_unique<MOM_BUNDLE_T>(MOM_GET_EXEC_BUNDLE(queue.get_context()));
  }
  ~FishState() { queue.wait_and_throw(); free_all(); }
  template <typename T> T* alloc(size_t n) { return shared_io ? sycl::malloc_shared<T>(n, queue) : sycl::malloc_device<T>(n, queue); }
  void free_ptr(auto*& p) { if (p) sycl::free(p, queue); p = nullptr; }
  void free_all() { free_ptr(input); free_ptr(target); free_ptr(cache); free_ptr(dag); free_ptr(result); }
  void ensure_io(unsigned in_sz) {
    if (input && in_sz <= input_cap && target && result) return;
    queue.wait_and_throw(); free_ptr(input); free_ptr(target); free_ptr(result);
    input_cap = in_sz < 180 ? 180 : in_sz;
    input = alloc<uint8_t>(input_cap); target = alloc<uint8_t>(HASH_LEN); result = sycl::malloc_shared<FishResult>(1, queue);
    if (!input || !target || !result) throw std::string("Can't allocate fishhash buffers");
  }
  void ensure_cache(bool log) {
    if (cache_built) return;
    const uint64_t t0 = now_ms();
    std::vector<uint64_t> hcache; build_light_cache_host(hcache);
    free_ptr(cache);  // release a partial buffer from a previously-failed attempt before re-allocating
    cache = alloc<uint64_t>(hcache.size());
    if (!cache) throw std::string("Can't allocate fishhash light cache");
    if (shared_io) std::memcpy(cache, hcache.data(), hcache.size() * sizeof(uint64_t));
    else sycl_wait_and_throw(queue.memcpy(cache, hcache.data(), hcache.size() * sizeof(uint64_t)), device);
    cache_built = true;
    if (log) std::fprintf(stderr, "FishHash light cache (72 MiB) built (%llu ms)\n", (unsigned long long)(now_ms() - t0));
  }
  void ensure_dag(bool log) {
    if (dag_built) return;
    ensure_cache(log);
    const uint64_t t0 = now_ms();
    free_ptr(dag);  // release a partial buffer from a previously-failed attempt before re-allocating
    dag = alloc<uint64_t>((size_t)FULL_DATASET_NUM_ITEMS * 16);  // 4.6 GiB
    if (!dag) throw std::string("Can't allocate fishhash DAG (4.6 GiB)");
    const uint32_t CHUNK = 1u << 21;  // 2M items/launch to avoid the GPU watchdog
    for (uint32_t s = 0; s < (uint32_t)FULL_DATASET_NUM_ITEMS; s += CHUNK) {
      const uint32_t c = std::min(CHUNK, (uint32_t)FULL_DATASET_NUM_ITEMS - s);
      sycl_wait_and_throw(submit_fishhash_dag_gen(queue, *bundle, cache, dag, s, c), device);
    }
    dag_built = true;
    if (log) std::fprintf(stderr, "FishHash DAG (4.6 GiB) built (%llu ms)\n", (unsigned long long)(now_ms() - t0));
  }
  static uint64_t now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
};

static FishState& fishhash_state(const std::string& dev_str) {
  static std::mutex m; static std::map<std::string, std::unique_ptr<FishState>> states;
  std::lock_guard<std::mutex> lock(m);
  auto& s = states[dev_str]; if (!s) s = std::make_unique<FishState>(dev_str); return *s;
}

} // namespace mom_fishhash

using namespace mom_fishhash;

int fishhash(
  const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size, uint8_t* const output,
  uint8_t* const mix_hash, uint64_t* const pnonce, const uint8_t* const target, const uint8_t* const /*seed_hash*/,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < 40) throw std::string("Bad fishhash input length");
  FishState& state = fishhash_state(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_io(input_size);
  if (is_test) state.ensure_cache(!is_benchmark);   // lazy light-cache path (no 4.6 GiB alloc)
  else state.ensure_dag(!is_benchmark);             // mining/bench: build the full DAG once + gather

  // Iron Fish (180B header) uses nonce@0 big-endian + BE target; the offline vector uses nonce@32 LE.
  const bool ironfish = input_size >= 140;
  const unsigned nonce_off = ironfish ? 0u : 32u;
  const bool nonce_be = ironfish, target_be = ironfish;
  uint64_t start_nonce = 0;
  if (nonce_be) { for (unsigned i = 0; i < 8; ++i) start_nonce = (start_nonce << 8) | input[nonce_off + i]; }
  else { std::memcpy(&start_nonce, input + nonce_off, sizeof(start_nonce)); }
  if (state.shared_io) { std::memcpy(state.input, input, input_size); std::memcpy(state.target, target, HASH_LEN); }
  else { state.queue.memcpy(state.input, input, input_size); state.queue.memcpy(state.target, target, HASH_LEN); }
  std::memset(state.result, 0, sizeof(FishResult));

  sycl_wait_and_throw(
    is_test
      ? submit_fishhash_search<false>(state.queue, *state.bundle, state.input, input_size, start_nonce,
          state.cache, nullptr, intensity, state.target, state.result, is_test, nonce_off, nonce_be, target_be)
      : submit_fishhash_search<true>(state.queue, *state.bundle, state.input, input_size, start_nonce,
          nullptr, state.dag, intensity, state.target, state.result, is_test, nonce_off, nonce_be, target_be),
    state.device);

  static const bool perf_log = std::getenv("MOM_FISHHASH_PERF") != nullptr;
  if (perf_log && !is_test) {
    static uint64_t acc = 0; static auto t0 = std::chrono::steady_clock::now();
    acc += intensity;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec >= 2.0) { std::fprintf(stderr, "[fishhash] %.2f MH/s (%s)\n", static_cast<double>(acc) / sec / 1e6, dev_str.c_str()); acc = 0; t0 = std::chrono::steady_clock::now(); }
  }

  const uint32_t count = state.result->count;
  if (count == 0) return 0;
  const uint32_t idx = std::min(count, MAX_OUTPUTS) - 1;
  *pnonce = state.result->nonce[idx];
  std::memcpy(output, state.result->output[idx], HASH_LEN);
  if (mix_hash) std::memset(mix_hash, 0, HASH_LEN);
  return 1;
}

int karlsenhashv2(
  const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size, uint8_t* const output,
  uint8_t* const mix_hash, uint64_t* const pnonce, const uint8_t* const target, const uint8_t* const /*seed_hash*/,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < 80) throw std::string("Bad karlsenhashv2 input length");
  FishState& state = fishhash_state(dev_str);  // shares the FishHash 4.6 GB DAG (identical seed/params)
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_io(input_size);
  if (is_test) state.ensure_cache(!is_benchmark);
  else state.ensure_dag(!is_benchmark);

  uint64_t start_nonce = 0; std::memcpy(&start_nonce, input + 72, sizeof(start_nonce));  // nonce@72 LE
  if (state.shared_io) { std::memcpy(state.input, input, input_size); std::memcpy(state.target, target, HASH_LEN); }
  else { state.queue.memcpy(state.input, input, input_size); state.queue.memcpy(state.target, target, HASH_LEN); }
  std::memset(state.result, 0, sizeof(FishResult));

  sycl_wait_and_throw(
    is_test
      ? submit_karlsenhashv2_search<false>(state.queue, *state.bundle, state.input, input_size, start_nonce,
          state.cache, nullptr, intensity, state.target, state.result, is_test)
      : submit_karlsenhashv2_search<true>(state.queue, *state.bundle, state.input, input_size, start_nonce,
          nullptr, state.dag, intensity, state.target, state.result, is_test),
    state.device);

  static const bool perf_log = std::getenv("MOM_KARLSEN_PERF") != nullptr;
  if (perf_log && !is_test) {
    static uint64_t acc = 0; static auto t0 = std::chrono::steady_clock::now();
    acc += intensity;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec >= 2.0) { std::fprintf(stderr, "[karlsenhashv2] %.2f MH/s (%s)\n", static_cast<double>(acc) / sec / 1e6, dev_str.c_str()); acc = 0; t0 = std::chrono::steady_clock::now(); }
  }

  const uint32_t count = state.result->count;
  if (count == 0) return 0;
  const uint32_t idx = std::min(count, MAX_OUTPUTS) - 1;
  *pnonce = state.result->nonce[idx];
  std::memcpy(output, state.result->output[idx], HASH_LEN);
  if (mix_hash) std::memset(mix_hash, 0, HASH_LEN);
  return 1;
}
