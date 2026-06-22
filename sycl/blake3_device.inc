// BLAKE3 single-chunk device hash (shared by fishhash.cpp + kheavyhash.cpp). Extracted from fishhash.cpp.
inline uint32_t load32_le_dev(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline void store32_le_dev(uint8_t* p, uint32_t v) { p[0]=v; p[1]=v>>8; p[2]=v>>16; p[3]=v>>24; }
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
