
// ---- BeamHash III parameters (beamHashIII.h) ----
constexpr unsigned WORK_BIT_SIZE = 448;
constexpr unsigned COLLISION_BIT_SIZE = 24;
constexpr unsigned NUM_ROUNDS = 5;                      // Wagner k=5 -> 2^5 = 32 leaves
constexpr unsigned NUM_LEAVES = 1u << NUM_ROUNDS;       // 32
constexpr unsigned INDEX_BITS = COLLISION_BIT_SIZE + 1; // 25-bit indices (index space 2^25)
constexpr uint32_t INDEX_SPACE_BITS = 25;
constexpr uint64_t NUM_ENTRIES = 1ull << INDEX_SPACE_BITS; // 2^25 leaves

constexpr unsigned WORK_WORDS = 7; // 448 bits = 7 u64 words (w[0] = least-significant)

// blob layout (host-side, passed in `input`): prework(32) || nonce(8) || extranonce(4) = 44 bytes.
constexpr unsigned PREWORK_BYTES = 32;
constexpr unsigned NONCE_BYTES = 8;
constexpr unsigned EXTRANONCE_BYTES = 4;
constexpr unsigned BLOB_BYTES = PREWORK_BYTES + NONCE_BYTES + EXTRANONCE_BYTES; // 44

// Generation-validation dump: the first BEAMHASH3_TEST_ROWS leaves' raw 448-bit rows, 7 LE u64 words each.
constexpr unsigned BEAMHASH3_TEST_ROWS = 64;
constexpr unsigned ROW_BYTES = WORK_WORDS * 8; // 56
static_assert(BEAMHASH3_TEST_ROWS * ROW_BYTES <= SMALL_BLOB_SOL_LEN,
              "generation dump must fit the small-blob buffer");

// ===========================================================================================
// SipHash-2-4 (device + host). prePow[0..3] are v0..v3 directly (already xored into the magic by the
// blake2b output, per beam). nonce = the 64-bit counter (index<<3)+i. Mirrors the JS oracle exactly.
// ===========================================================================================
inline uint64_t rotl64(uint64_t x, unsigned b) {
  return mom_rotate(x, static_cast<uint64_t>(b));
}

// Generate one Beam SipHash value with two message rounds and four finalization rounds.
inline uint64_t siphash24(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3, uint64_t nonce) {
  uint64_t v0 = s0, v1 = s1, v2 = s2, v3 = s3;
  auto sip_round = [&]() {
    v0 += v1;
    v2 += v3;
    v1 = rotl64(v1, 13);
    v3 = rotl64(v3, 16);
    v1 ^= v0;
    v3 ^= v2;
    v0 = rotl64(v0, 32);
    v2 += v1;
    v0 += v3;
    v1 = rotl64(v1, 17);
    v3 = rotl64(v3, 21);
    v1 ^= v2;
    v3 ^= v0;
    v2 = rotl64(v2, 32);
  };
  v3 ^= nonce;
  sip_round();
  sip_round();
  v0 ^= nonce;
  v2 ^= 0xffull;
  sip_round();
  sip_round();
  sip_round();
  sip_round();
  return v0 ^ v1 ^ v2 ^ v3;
}

#if defined(MOM_BEAMHASH3_INTEL_PAIR32)
// Xe2 lowers SipHash's repeated uint64 additions through partial-int64 sequences. Keeping each word
// as two uint32 halves avoids that compiler bottleneck while retaining the portable uint64 path.
struct BeamU64Pair {
  uint32_t lo;
  uint32_t hi;
};

inline BeamU64Pair beam_pair_from_u64(uint64_t value) {
  return {static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 32)};
}

inline uint64_t beam_pair_to_u64(BeamU64Pair value) {
  return static_cast<uint64_t>(value.lo) | (static_cast<uint64_t>(value.hi) << 32);
}

// Carry split-word addition exactly for the portable paired-integer hashing path.
inline BeamU64Pair beam_pair_add(BeamU64Pair a, BeamU64Pair b) {
  const uint32_t lo = a.lo + b.lo;
  return {lo, a.hi + b.hi + static_cast<uint32_t>(lo < a.lo)};
}

inline BeamU64Pair beam_pair_xor(BeamU64Pair a, BeamU64Pair b) {
  return {a.lo ^ b.lo, a.hi ^ b.hi};
}

template <unsigned Bits>
// Rotate the paired words at compile-time constants without requiring native 64-bit shifts.
inline BeamU64Pair beam_pair_rotl(BeamU64Pair value) {
  static_assert(Bits > 0 && Bits < 64);
  if constexpr (Bits < 32) {
    return {(value.lo << Bits) | (value.hi >> (32 - Bits)),
            (value.hi << Bits) | (value.lo >> (32 - Bits))};
  } else if constexpr (Bits == 32) {
    return {value.hi, value.lo};
  } else {
    constexpr unsigned Shift = Bits - 32;
    return {(value.hi << Shift) | (value.lo >> (32 - Shift)),
            (value.lo << Shift) | (value.hi >> (32 - Shift))};
  }
}

// Keep paired-integer leaf hashing bit-equivalent to the native SipHash implementation.
inline BeamU64Pair siphash24_pair(BeamU64Pair s0, BeamU64Pair s1, BeamU64Pair s2,
                                  BeamU64Pair s3, BeamU64Pair nonce) {
  BeamU64Pair v0 = s0;
  BeamU64Pair v1 = s1;
  BeamU64Pair v2 = s2;
  BeamU64Pair v3 = s3;
  auto sip_round = [&]() {
    v0 = beam_pair_add(v0, v1);
    v2 = beam_pair_add(v2, v3);
    v1 = beam_pair_rotl<13>(v1);
    v3 = beam_pair_rotl<16>(v3);
    v1 = beam_pair_xor(v1, v0);
    v3 = beam_pair_xor(v3, v2);
    v0 = beam_pair_rotl<32>(v0);
    v2 = beam_pair_add(v2, v1);
    v0 = beam_pair_add(v0, v3);
    v1 = beam_pair_rotl<17>(v1);
    v3 = beam_pair_rotl<21>(v3);
    v1 = beam_pair_xor(v1, v2);
    v3 = beam_pair_xor(v3, v0);
    v2 = beam_pair_rotl<32>(v2);
  };
  v3 = beam_pair_xor(v3, nonce);
  sip_round();
  sip_round();
  v0 = beam_pair_xor(v0, nonce);
  v2.lo ^= 0xffu;
  sip_round();
  sip_round();
  sip_round();
  sip_round();
  return beam_pair_xor(beam_pair_xor(v0, v1), beam_pair_xor(v2, v3));
}

// Intel seed path: form each nonce pair inline so leaf words can be written directly to SLM.
inline BeamU64Pair beam_pair_nonce(uint32_t index, uint32_t offset) {
  const uint32_t base_lo = index << 3;
  const uint32_t base_hi = index >> 29;
  const uint32_t nonce_lo = base_lo + offset;
  return {nonce_lo, base_hi + static_cast<uint32_t>(nonce_lo < base_lo)};
}
#endif

// ===========================================================================================
// StepElem on the device: a row carries WORK_WORDS u64 work bits (w[0]=LSB) + an index tree.
// The index tree at level L holds 2^L indices (lowest-index-first within each merge). For applyMix we
// only ever need the FIRST padNum (<=9) indices of the tree.
// ===========================================================================================

// makeLeaf: workBits word i = siphash24(prePow, (index<<3)+i). indexTree=[index].
// Build the 448-bit Beam leaf work from its SipHash index-derived nonce sequence.
inline void make_leaf(const uint64_t prePow[4], uint32_t index, uint64_t w[WORK_WORDS]) {
#if defined(MOM_BEAMHASH3_INTEL_PAIR32)
  const BeamU64Pair s0 = beam_pair_from_u64(prePow[0]);
  const BeamU64Pair s1 = beam_pair_from_u64(prePow[1]);
  const BeamU64Pair s2 = beam_pair_from_u64(prePow[2]);
  const BeamU64Pair s3 = beam_pair_from_u64(prePow[3]);
  const uint32_t base_lo = index << 3;
  const uint32_t base_hi = index >> 29;
  for (unsigned i = 0; i < WORK_WORDS; ++i) {
    const uint32_t nonce_lo = base_lo + i;
    const BeamU64Pair nonce = {nonce_lo, base_hi + static_cast<uint32_t>(nonce_lo < base_lo)};
    w[i] = beam_pair_to_u64(siphash24_pair(s0, s1, s2, s3, nonce));
  }
#else
  const uint64_t base = ((uint64_t)index << 3);
  for (unsigned i = 0; i < WORK_WORDS; ++i)
    w[i] = siphash24(prePow[0], prePow[1], prePow[2], prePow[3], base + i);
#endif
}

// 448-bit right shift by COLLISION_BIT_SIZE (24) across WORK_WORDS u64 words, then mask to remLen bits.
// XOR child work and discard the collision prefix before the next mixing stage.
inline void merge_workbits(const uint64_t a[WORK_WORDS], const uint64_t b[WORK_WORDS],
                           unsigned remLen, uint64_t out[WORK_WORDS]) {
  uint64_t x[WORK_WORDS];
  for (unsigned i = 0; i < WORK_WORDS; ++i)
    x[i] = a[i] ^ b[i];
  // shift right by 24 bits (logical, 448-bit).
  constexpr unsigned SH = COLLISION_BIT_SIZE; // 24
  for (unsigned i = 0; i < WORK_WORDS; ++i) {
    uint64_t lo = x[i] >> SH;
    uint64_t hi = (i + 1 < WORK_WORDS) ? (x[i + 1] << (64 - SH)) : 0ull;
    out[i] = lo | hi;
  }
  // mask to low remLen bits.
  for (unsigned i = 0; i < WORK_WORDS; ++i) {
    const unsigned bit0 = i * 64;
    if (bit0 >= remLen) {
      out[i] = 0;
      continue;
    }
    const unsigned rem = remLen - bit0;
    if (rem < 64)
      out[i] &= ((rem == 0) ? 0ull : ((uint64_t)~0ull >> (64 - rem)));
  }
}

// applyMix: in-place modify w[] using the first `padNum` indices of the index tree.
//   tempBits(512) = workBits (8 u64 words, words 7 = 0).
//   padNum = min( ((512-remLen)+24)/25 , treeLen )
//   for i in 0..padNum-1: tempBits |= indexTree[i] << (remLen + i*25)   (512-bit shift)
//   result = sum_{i=0..7} rotl64(word_i, (29*(i+1))&63); result = rotl64(result, 24)
//   w[0] = result (low 64 work bits replaced).
inline void apply_mix(uint64_t w[WORK_WORDS], const uint32_t* tree, unsigned treeLen,
                      unsigned remLen) {
  uint64_t t[8];
  for (unsigned i = 0; i < WORK_WORDS; ++i)
    t[i] = w[i];
  t[7] = 0;

  unsigned padNum = ((512u - remLen) + COLLISION_BIT_SIZE) / (COLLISION_BIT_SIZE + 1); // /25
  if (padNum > treeLen)
    padNum = treeLen;
  for (unsigned i = 0; i < padNum; ++i) {
    const unsigned bitpos = remLen + i * (COLLISION_BIT_SIZE + 1); // remLen + i*25
    const uint64_t val = (uint64_t)tree[i];
    // OR `val << bitpos` into the 512-bit t[] (val is <=25 bits, spans at most 2 words).
    const unsigned word = bitpos >> 6;
    const unsigned off = bitpos & 63;
    if (word < 8)
      t[word] |= (val << off);
    if (off != 0 && word + 1 < 8)
      t[word + 1] |= (val >> (64 - off));
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
