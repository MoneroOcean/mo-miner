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
