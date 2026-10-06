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
#include "../blake3-device.inc"
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

inline uint32_t fnv1_dev(uint32_t u, uint32_t v) {
  return (u * FNV_PRIME) ^ v;
}

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
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
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
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile("prefetch.global.L1 [%0];" :: "l"(p));
#else
  (void)p;
#endif
}
