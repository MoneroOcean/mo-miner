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
