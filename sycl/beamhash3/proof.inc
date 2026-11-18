// IndividualWork = BLAKE2b-256(prework || nonce || extranonce) personalised "Beam-PoW" ||
// le32(448) || le32(5). The 44-byte blob is one final block; the digest's four little-endian words
// are the SipHash key.
static void individual_work(const uint8_t blob[BLOB_BYTES], uint64_t prePow[4]) {
  uint8_t personal[16];
  mom_blake2b::equihash_personal("Beam-PoW", WORK_BIT_SIZE, NUM_ROUNDS, personal);
  uint64_t h[8];
  mom_blake2b::init(h, 32, personal);
  uint8_t block[128] = {0};
  std::memcpy(block, blob, BLOB_BYTES);
  mom_blake2b::compress_bytes(h, block, BLOB_BYTES, true);
  for (unsigned i = 0; i < 4; ++i)
    prePow[i] = h[i];
}

// CompressArray(25): pack 32 canonical 25-bit leaf indices into Beam's 100-byte solution payload,
// the inverse of getIndicesFromMinimal. Field i occupies stream bits [i*25, i*25+25), LSB-first, so
// byte b's bit k is stream bit b*8+k.
static void compress_indices(const uint32_t idx[NUM_LEAVES], uint8_t out[100]) {
  // Build the 800-bit little-endian bit stream: out[byte] bit-k = field(byte*8+k).
  for (unsigned i = 0; i < 100; ++i)
    out[i] = 0;
  for (unsigned f = 0; f < NUM_LEAVES; ++f) {
    const uint32_t v = idx[f] & ((1u << INDEX_BITS) - 1u);
    const unsigned base = f * INDEX_BITS; // bit position in the stream
    for (unsigned b = 0; b < INDEX_BITS; ++b) {
      if (v & (1u << b)) {
        const unsigned pos = base + b;
        out[pos >> 3] |= (uint8_t)(1u << (pos & 7));
      }
    }
  }
}

// Beam Difficulty::IsTargetReached. Beam's PoW share check is SHA-256 (equihash-pow.h) of the
// 104-byte solution (ECC::Hash::Value, a 32-byte uintBig read big-endian), tested against the
// 32-bit packed network difficulty (s_MantissaBits=24):
//   order    = packed >> 24
//   mantissa = (1<<24) | (packed & 0xFFFFFF)              // a 25-bit normalized value
//   a        = hashValue(256-bit) * mantissa              // up to a 281-bit product
//   reached  <=> a fits in (256 + 24 - order) bits, i.e. a < 2^(280-order)
// (beam core/difficulty.cpp IsTargetReached: a.get_ConstSlice().IsWithinOrder(nBits+s_MantissaBits-order).)
// hash_be[0] is the most-significant byte; the product (256 + 25 bits) is held in 16-bit limbs.
static bool beam_target_reached(const uint8_t hash_be[32], uint32_t packed) {
  constexpr uint32_t MANTISSA_BITS = 24;
  constexpr uint32_t MAX_ORDER = 256u - MANTISSA_BITS - 1u; // s_MaxOrder
  constexpr uint32_t INF = (MAX_ORDER + 1u) << MANTISSA_BITS;
  if (packed > INF)
    return false; // invalid difficulty

  const uint32_t order = packed >> MANTISSA_BITS;
  const uint64_t mantissa =
      (uint64_t)((1u << MANTISSA_BITS) | (packed & ((1u << MANTISSA_BITS) - 1u)));

  // a = hash (256-bit, big-endian) * mantissa, in 16 limbs of 16 bits (LSB-first), product fits 18 limbs.
  uint16_t hw[16];
  for (unsigned i = 0; i < 16; ++i) // limb 0 = least-significant 16 bits = hash_be's last 2 bytes
    hw[i] = (uint16_t)(((uint32_t)hash_be[31 - 2 * i - 1] << 8) | hash_be[31 - 2 * i]);
  // Schoolbook: prod += hw[i]*mantissa << (16*i). mantissa is 25-bit -> per-limb partial is 41-bit.
  uint16_t prod[18] = {0};
  for (unsigned i = 0; i < 16; ++i) {
    uint64_t partial = (uint64_t)hw[i] * mantissa; // up to 41 bits
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
  const uint32_t threshold = 280u - order; // order <= MAX_ORDER=231 -> threshold in [49,280]
  int highbit = -1;
  for (int limb = 17; limb >= 0 && highbit < 0; --limb) {
    if (prod[limb] == 0)
      continue;
    for (int b = 15; b >= 0; --b)
      if (prod[limb] & (1u << b)) {
        highbit = limb * 16 + b;
        break;
      }
  }
  // value < 2^threshold  <=>  highest set bit index < threshold (or value == 0).
  return (highbit < 0) || ((uint32_t)highbit < threshold);
}

