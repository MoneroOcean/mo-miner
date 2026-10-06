// Certificate V3 uses salted noise-seed derivation while keeping serialized Merkle roots raw.
namespace mom_pearlhash {

static constexpr uint8_t PEARLHASH_SEED_SALT_A[32] = {
  0x82, 0x49, 0x40, 0x6c, 0xa0, 0xed, 0x15, 0x16, 0x96, 0x16, 0xf6, 0x92, 0xfc, 0xf0, 0x76, 0xf8,
  0x92, 0xdb, 0xdb, 0x2a, 0x70, 0x23, 0xb8, 0x52, 0xf0, 0xd4, 0x77, 0x19, 0xc3, 0x90, 0x01, 0x7b,
};
static constexpr uint8_t PEARLHASH_SEED_SALT_B[32] = {
  0x11, 0x30, 0x06, 0x32, 0xec, 0x63, 0x01, 0xca, 0x2b, 0xe2, 0xaf, 0x71, 0x8b, 0x3f, 0x4d, 0x4f,
  0x1a, 0xe9, 0xc6, 0x39, 0x88, 0xe8, 0xcc, 0x04, 0x48, 0x44, 0x30, 0x1d, 0x71, 0xb8, 0x9a, 0xa9,
};

static inline void pearlhash_bind_root(const uint8_t raw_root[32], const uint32_t dimension,
                                        const uint8_t salt[32], uint8_t bound_root[32]) {
  uint8_t message[64]{};
  for (int i = 0; i < 32; ++i)
    message[i] = raw_root[i];
  message[32] = static_cast<uint8_t>(dimension);
  message[33] = static_cast<uint8_t>(dimension >> 8);
  message[34] = static_cast<uint8_t>(dimension >> 16);
  message[35] = static_cast<uint8_t>(dimension >> 24);
  pearlhash_b3::b3(message, 64, salt, bound_root);
}

// V3 binds each raw root to its matrix dimension with a domain-separated key, then computes
// cB = BLAKE3(key || bound_rootB) and cA = BLAKE3(cB || bound_rootA). The raw roots remain
// available to the host PlainProof builder and are never replaced in the proof.
static inline void pearlhash_commitment_seeds(
    const uint8_t key[32], const uint8_t raw_root_a[32], const uint8_t raw_root_b[32],
    const uint32_t m, const uint32_t n, uint8_t c_b[32], uint8_t c_a[32]) {
  uint8_t root_a[32], root_b[32], input[64];
  pearlhash_bind_root(raw_root_a, m, PEARLHASH_SEED_SALT_A, root_a);
  pearlhash_bind_root(raw_root_b, n, PEARLHASH_SEED_SALT_B, root_b);
  for (int i = 0; i < 32; ++i) {
    input[i] = key[i];
    input[32 + i] = root_b[i];
  }
  pearlhash_b3::b3(input, 64, nullptr, c_b);
  for (int i = 0; i < 32; ++i) {
    input[i] = c_b[i];
    input[32 + i] = root_a[i];
  }
  pearlhash_b3::b3(input, 64, nullptr, c_a);
}

} // namespace mom_pearlhash
