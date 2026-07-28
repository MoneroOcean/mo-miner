inline Uint2 make_u2(const uint32_t x, const uint32_t y) {
  return Uint2{x, y};
}

inline Uint2 operator^(const Uint2 a, const Uint2 b) {
  return make_u2(a.x ^ b.x, a.y ^ b.y);
}

inline Uint2 operator&(const Uint2 a, const Uint2 b) {
  return make_u2(a.x & b.x, a.y & b.y);
}

inline Uint2 operator~(const Uint2 a) {
  return make_u2(~a.x, ~a.y);
}

inline Uint2& operator^=(Uint2& a, const Uint2 b) {
  a.x ^= b.x;
  a.y ^= b.y;
  return a;
}

inline Uint2 rotl64_pair_dev(const Uint2 value, const unsigned shift) {
  const unsigned s = shift & 63U;
  if (s == 0) return value;
  if (s < 32) {
    return make_u2(
      (value.x << s) | (value.y >> (32U - s)),
      (value.y << s) | (value.x >> (32U - s))
    );
  }
  if (s == 32) return make_u2(value.y, value.x);
  return make_u2(
    (value.y << (s - 32U)) | (value.x >> (64U - s)),
    (value.x << (s - 32U)) | (value.y >> (64U - s))
  );
}

inline void keccakf1600_pair_chi_dev(Uint2 st[25], const unsigned row, const Uint2 t[25]) {
  st[row + 0] = t[row + 0] ^ ((~t[row + 1]) & t[row + 2]);
  st[row + 1] = t[row + 1] ^ ((~t[row + 2]) & t[row + 3]);
  st[row + 2] = t[row + 2] ^ ((~t[row + 3]) & t[row + 4]);
  st[row + 3] = t[row + 3] ^ ((~t[row + 4]) & t[row + 0]);
  st[row + 4] = t[row + 4] ^ ((~t[row + 0]) & t[row + 1]);
}

inline void keccakf1600_pair_round_dev(Uint2 st[25], const unsigned round) {
  static constexpr Uint2 rndc[24] = {
    {0x00000001, 0x00000000}, {0x00008082, 0x00000000},
    {0x0000808a, 0x80000000}, {0x80008000, 0x80000000},
    {0x0000808b, 0x00000000}, {0x80000001, 0x00000000},
    {0x80008081, 0x80000000}, {0x00008009, 0x80000000},
    {0x0000008a, 0x00000000}, {0x00000088, 0x00000000},
    {0x80008009, 0x00000000}, {0x8000000a, 0x00000000},
    {0x8000808b, 0x00000000}, {0x0000008b, 0x80000000},
    {0x00008089, 0x80000000}, {0x00008003, 0x80000000},
    {0x00008002, 0x80000000}, {0x00000080, 0x80000000},
    {0x0000800a, 0x00000000}, {0x8000000a, 0x80000000},
    {0x80008081, 0x80000000}, {0x00008080, 0x80000000},
    {0x80000001, 0x00000000}, {0x80008008, 0x80000000}
  };

  Uint2 t[25];
  Uint2 u;

  t[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
  t[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
  t[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
  t[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
  t[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

  u = t[4] ^ rotl64_pair_dev(t[1], 1);
  st[0] ^= u; st[5] ^= u; st[10] ^= u; st[15] ^= u; st[20] ^= u;
  u = t[0] ^ rotl64_pair_dev(t[2], 1);
  st[1] ^= u; st[6] ^= u; st[11] ^= u; st[16] ^= u; st[21] ^= u;
  u = t[1] ^ rotl64_pair_dev(t[3], 1);
  st[2] ^= u; st[7] ^= u; st[12] ^= u; st[17] ^= u; st[22] ^= u;
  u = t[2] ^ rotl64_pair_dev(t[4], 1);
  st[3] ^= u; st[8] ^= u; st[13] ^= u; st[18] ^= u; st[23] ^= u;
  u = t[3] ^ rotl64_pair_dev(t[0], 1);
  st[4] ^= u; st[9] ^= u; st[14] ^= u; st[19] ^= u; st[24] ^= u;

  t[0] = st[0];
  t[10] = rotl64_pair_dev(st[1], 1);
  t[20] = rotl64_pair_dev(st[2], 62);
  t[5] = rotl64_pair_dev(st[3], 28);
  t[15] = rotl64_pair_dev(st[4], 27);

  t[16] = rotl64_pair_dev(st[5], 36);
  t[1] = rotl64_pair_dev(st[6], 44);
  t[11] = rotl64_pair_dev(st[7], 6);
  t[21] = rotl64_pair_dev(st[8], 55);
  t[6] = rotl64_pair_dev(st[9], 20);

  t[7] = rotl64_pair_dev(st[10], 3);
  t[17] = rotl64_pair_dev(st[11], 10);
  t[2] = rotl64_pair_dev(st[12], 43);
  t[12] = rotl64_pair_dev(st[13], 25);
  t[22] = rotl64_pair_dev(st[14], 39);

  t[23] = rotl64_pair_dev(st[15], 41);
  t[8] = rotl64_pair_dev(st[16], 45);
  t[18] = rotl64_pair_dev(st[17], 15);
  t[3] = rotl64_pair_dev(st[18], 21);
  t[13] = rotl64_pair_dev(st[19], 8);

  t[14] = rotl64_pair_dev(st[20], 18);
  t[24] = rotl64_pair_dev(st[21], 2);
  t[9] = rotl64_pair_dev(st[22], 61);
  t[19] = rotl64_pair_dev(st[23], 56);
  t[4] = rotl64_pair_dev(st[24], 14);

  keccakf1600_pair_chi_dev(st, 0, t);
  st[0] ^= rndc[round];
  keccakf1600_pair_chi_dev(st, 5, t);
  keccakf1600_pair_chi_dev(st, 10, t);
  keccakf1600_pair_chi_dev(st, 15, t);
  keccakf1600_pair_chi_dev(st, 20, t);
}

#if defined(__NVPTX__)
// NVIDIA (nvptx device image): the 32-bit-pair permutation is correct and tuned for warps; unchanged from 0.7.0.
inline void keccakf1600_pair_dev(Uint2 st[25]) {
  for (unsigned round = 0; round < 24; ++round) keccakf1600_pair_round_dev(st, round);
}
#else
// SPIR-V device image (Intel IGC / AMD / CPU). IGC 2.36.x silently miscompiles the Uint2 32-bit-pair
// Keccak-f[1600] when it is INLINED into the high-register-pressure DAG kernel on Xe2/BMG (Arc B580):
// the pair state held live across the parent-mix loop comes out 100% wrong. It is bit-exact in
// isolation, so forcing the permutation OUT OF LINE (noinline) dodges the register-allocation
// miscompile while keeping the fast pair path, bit-exact on every IGC. (KawPoW uses this pair Keccak
// only in DAG generation; the per-nonce seal uses keccak_f800, unaffected.) Clean 2.28.4-good ->
// 2.36.3-broken regression; see ~/bug_report5 (sibling of intel-graphics-compiler#414).
inline __attribute__((noinline)) void keccakf1600_pair_dev(Uint2 st[25]) {
  for (unsigned round = 0; round < 24; ++round) keccakf1600_pair_round_dev(st, round);
}
#endif

inline void keccak_512_words_dev(uint32_t words[NODE_WORDS]) {
  Uint2 st[25]{};
  for (unsigned i = 0; i < 8; ++i) st[i] = make_u2(words[i * 2], words[i * 2 + 1]);
  st[8] = make_u2(0x00000001, 0x80000000);
  keccakf1600_pair_dev(st);
  for (unsigned i = 0; i < 8; ++i) {
    words[i * 2] = st[i].x;
    words[i * 2 + 1] = st[i].y;
  }
}

inline void keccak_f800_round_dev(uint32_t st[25], const unsigned round) {
  static constexpr uint32_t rndc[22] = {
    0x00000001, 0x00008082, 0x0000808a, 0x80008000, 0x0000808b, 0x80000001,
    0x80008081, 0x00008009, 0x0000008a, 0x00000088, 0x80008009, 0x8000000a,
    0x8000808b, 0x0000008b, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800a, 0x8000000a, 0x80008081, 0x00008080
  };
  static constexpr uint32_t rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
  };
  static constexpr uint32_t piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
  };

  uint32_t bc[5];
  for (unsigned i = 0; i < 5; ++i) {
    bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
  }
  for (unsigned i = 0; i < 5; ++i) {
    const uint32_t t = bc[(i + 4) % 5] ^ rotl32_dev(bc[(i + 1) % 5], 1);
    for (unsigned j = 0; j < 25; j += 5) st[j + i] ^= t;
  }

  uint32_t t = st[1];
  for (unsigned i = 0; i < 24; ++i) {
    const uint32_t j = piln[i];
    bc[0] = st[j];
    st[j] = rotl32_dev(t, rotc[i]);
    t = bc[0];
  }

  for (unsigned j = 0; j < 25; j += 5) {
    for (unsigned i = 0; i < 5; ++i) bc[i] = st[j + i];
    for (unsigned i = 0; i < 5; ++i) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
  }
  st[0] ^= rndc[round];
}

inline void keccak_f800_dev(uint32_t st[25]) {
  for (unsigned round = 0; round < 22; ++round) keccak_f800_round_dev(st, round);
}

// REGS is the active per-lane register count (KawPoW 32, meowpow 16); defaults to KAWPOW_REGS so the
// existing variants compile unchanged. `mix` is sized at the 32-reg max by the caller; only the first
// REGS entries are seeded/used.
template<unsigned REGS = KAWPOW_REGS>
MO_KAW_AINLINE void fill_mix_dev(const uint32_t seed0, const uint32_t seed1, const uint32_t lane, uint32_t* mix) {
  uint32_t fnv_hash = FNV_OFFSET_BASIS;
  Kiss99 st{};
  st.z = fnv1a(fnv_hash, seed0);
  st.w = fnv1a(fnv_hash, seed1);
  st.jsr = fnv1a(fnv_hash, lane);
  st.jcong = fnv1a(fnv_hash, lane);

  for (unsigned i = 0; i < REGS; ++i) mix[i] = kiss99(st);
}
