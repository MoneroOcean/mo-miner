namespace mom_etchash {

constexpr uint32_t ETCHASH_FORK_BLOCK      = 11700000;
constexpr uint32_t ETHASH_EPOCH            = 30000;
constexpr uint32_t ETCHASH_EPOCH           = 60000;
constexpr uint32_t ETHASH_NODE_WORDS       = 16;
constexpr uint32_t ETHASH_MIX_WORDS        = ETHASH_MIX_BYTES / sizeof(uint32_t);
constexpr uint32_t ETHASH_DATASET_PARENTS2 = 256;
constexpr uint32_t FNV_PRIME               = 0x01000193U;
constexpr uint32_t MAX_ETCHASH_OUTPUTS     = 15;

// FastModData / make_fast_mod_data / fast_mod_dev are shared from lib-internal.h.

struct EtchashResult {
  uint32_t count;
  uint64_t nonce[MAX_ETCHASH_OUTPUTS];
  uint8_t output[MAX_ETCHASH_OUTPUTS][HASH_LEN];
  uint8_t mix_hash[MAX_ETCHASH_OUTPUTS][HASH_LEN];
};

struct Uint2 {
  uint32_t x;
  uint32_t y;
};

inline uint32_t round_up(const uint32_t value, const uint32_t step) {
  return ((value + step - 1) / step) * step;
}

inline uint32_t fnv_dev(const uint32_t x, const uint32_t y) {
  return x * FNV_PRIME ^ y;
}

// Compress 4 consecutive mix words into one via the ethash FNV chain.
inline uint32_t fnv_fold4_dev(const uint32_t* const v) {
  return fnv_dev(fnv_dev(fnv_dev(v[0], v[1]), v[2]), v[3]);
}

inline uint32_t load32_le_dev(const uint8_t* const input) {
  return static_cast<uint32_t>(input[0]) |
         (static_cast<uint32_t>(input[1]) << 8) |
         (static_cast<uint32_t>(input[2]) << 16) |
         (static_cast<uint32_t>(input[3]) << 24);
}

inline uint64_t load64_le_dev(const uint8_t* const input) {
  return static_cast<uint64_t>(load32_le_dev(input)) |
         (static_cast<uint64_t>(load32_le_dev(input + 4)) << 32);
}

inline void store32_le_dev(uint8_t* const output, const uint32_t value) {
  output[0] = static_cast<uint8_t>(value);
  output[1] = static_cast<uint8_t>(value >> 8);
  output[2] = static_cast<uint8_t>(value >> 16);
  output[3] = static_cast<uint8_t>(value >> 24);
}

inline uint64_t rotl64_dev(const uint64_t value, const unsigned shift) {
  return (value << shift) | (value >> (64U - shift));
}

inline void keccakf1600_u64_round_dev(uint64_t st[25], const unsigned round) {
  static constexpr uint64_t rndc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
  };
  static constexpr unsigned rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
  };
  static constexpr unsigned piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
  };

  uint64_t bc[5];
  for (unsigned i = 0; i < 5; ++i) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
  for (unsigned i = 0; i < 5; ++i) {
    const uint64_t t = bc[(i + 4) % 5] ^ rotl64_dev(bc[(i + 1) % 5], 1);
    for (unsigned j = 0; j < 25; j += 5) st[j + i] ^= t;
  }

  uint64_t t = st[1];
  for (unsigned i = 0; i < 24; ++i) {
    const unsigned j = piln[i];
    bc[0] = st[j];
    st[j] = rotl64_dev(t, rotc[i]);
    t = bc[0];
  }

  for (unsigned j = 0; j < 25; j += 5) {
    for (unsigned i = 0; i < 5; ++i) bc[i] = st[j + i];
    for (unsigned i = 0; i < 5; ++i) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
  }

  st[0] ^= rndc[round];
}

inline void keccakf1600_u64_dev(uint64_t st[25]) {
  for (unsigned round = 0; round < 24; ++round) keccakf1600_u64_round_dev(st, round);
}

inline void store64_words_le_dev(uint32_t* const output, const unsigned lane, const uint64_t value) {
  output[lane * 2] = static_cast<uint32_t>(value);
  output[lane * 2 + 1] = static_cast<uint32_t>(value >> 32);
}

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

// Scatter a 64-bit lane held as a Uint2 into two little-endian 32-bit words.
inline void store_pair_words_dev(uint32_t* const out, const unsigned lane, const Uint2 value) {
  out[lane * 2] = value.x;
  out[lane * 2 + 1] = value.y;
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
// SPIR-V device image (Intel IGC / AMD / CPU). IGC 2.36.x SILENTLY MISCOMPILES the Uint2 32-bit-pair
// Keccak-f[1600] when it is INLINED into the high-register-pressure DAG/search kernels on Xe2/BMG
// (Arc B580): the pair state held live across the loop comes out 100% wrong. The permutation is
// bit-exact in isolation, so forcing it OUT OF LINE (noinline) dodges the register-allocation
// miscompile while keeping the fast 32-bit pair path -- ~13% faster than a scalar u64 fallback and
// bit-exact on every IGC. Clean IGC 2.28.4-good -> 2.36.3-broken regression; see ~/bug_report5
// (sibling of intel-graphics-compiler#414). This out-of-line function is the always-correct reference
// path (used by DAG-gen and by the noinline search instantiation, byte-identical to 0.7.0 codegen).
inline __attribute__((noinline)) void keccakf1600_pair_dev(Uint2 st[25]) {
  for (unsigned round = 0; round < 24; ++round) keccakf1600_pair_round_dev(st, round);
}
#endif

// Selects the pair permutation for a search-kernel instantiation. INLINE_PAIR=true expands the 24
// rounds IN PLACE (the fast pre-0.7.0 path, miscompiled on the affected IGC); false calls the single
// out-of-line keccakf1600_pair_dev reference. Inlining the rounds directly -- rather than via a second
// identical always_inline function -- keeps the noinline path's codegen byte-identical to 0.7.0 (an
// identical-bodied always_inline twin made IGC miscompile the *noinline* path even on a good IGC).
template <bool INLINE_PAIR>
inline void keccakf1600_pair_sel_dev(Uint2 st[25]) {
  if constexpr (INLINE_PAIR) { for (unsigned round = 0; round < 24; ++round) keccakf1600_pair_round_dev(st, round); }
  else                       { keccakf1600_pair_dev(st); }
}

inline void keccak_512_words_dev(uint32_t words[ETHASH_NODE_WORDS]) {
  Uint2 st[25]{};
  for (unsigned i = 0; i < 8; ++i) st[i] = make_u2(words[i * 2], words[i * 2 + 1]);
  st[8] = make_u2(0x00000001, 0x80000000);
  keccakf1600_pair_dev(st);
  for (unsigned i = 0; i < 8; ++i) store_pair_words_dev(words, i, st[i]);
}

inline void keccak_512_header_nonce_dev(const uint8_t* const header_hash, const uint64_t nonce, uint32_t out[16]) {
  uint64_t st[25]{};
  for (unsigned i = 0; i < 4; ++i) st[i] = load64_le_dev(header_hash + i * 8);
  st[4] = nonce;
  st[5] = 0x0000000000000001ULL;
  st[8] = 0x8000000000000000ULL;
  keccakf1600_u64_dev(st);
  for (unsigned i = 0; i < 8; ++i) store64_words_le_dev(out, i, st[i]);
}

template <bool INLINE_PAIR>
inline void keccak_512_header_nonce_pair_dev(const uint8_t* const header_hash, const uint64_t nonce, uint32_t out[16]) {
  Uint2 st[25]{};
  for (unsigned i = 0; i < 4; ++i) {
    st[i] = make_u2(load32_le_dev(header_hash + i * 8), load32_le_dev(header_hash + i * 8 + 4));
  }
  st[4] = make_u2(static_cast<uint32_t>(nonce), static_cast<uint32_t>(nonce >> 32));
  st[5] = make_u2(0x00000001, 0x00000000);
  st[8] = make_u2(0x00000000, 0x80000000);
  keccakf1600_pair_sel_dev<INLINE_PAIR>(st);
  for (unsigned i = 0; i < 8; ++i) store_pair_words_dev(out, i, st[i]);
}

inline void keccak_256_seed_mix_dev(const uint32_t seed[16], const uint32_t mix[8], uint8_t out[32]) {
  uint64_t st[25]{};
  for (unsigned i = 0; i < 8; ++i) st[i] = static_cast<uint64_t>(seed[i * 2]) |
                                            (static_cast<uint64_t>(seed[i * 2 + 1]) << 32);
  for (unsigned i = 0; i < 4; ++i) st[8 + i] = static_cast<uint64_t>(mix[i * 2]) |
                                                (static_cast<uint64_t>(mix[i * 2 + 1]) << 32);
  st[12] = 0x0000000000000001ULL;
  st[16] = 0x8000000000000000ULL;
  keccakf1600_u64_dev(st);
  for (unsigned i = 0; i < 4; ++i) {
    store32_le_dev(out + i * 8, static_cast<uint32_t>(st[i]));
    store32_le_dev(out + i * 8 + 4, static_cast<uint32_t>(st[i] >> 32));
  }
}

template <bool INLINE_PAIR>
inline void keccak_256_seed_mix_pair_dev(const uint32_t seed[16], const uint32_t mix[8], uint8_t out[32]) {
  Uint2 st[25]{};
  for (unsigned i = 0; i < 8; ++i) st[i] = make_u2(seed[i * 2], seed[i * 2 + 1]);
  for (unsigned i = 0; i < 4; ++i) st[8 + i] = make_u2(mix[i * 2], mix[i * 2 + 1]);
  st[12] = make_u2(0x00000001, 0x00000000);
  st[16] = make_u2(0x00000000, 0x80000000);
  keccakf1600_pair_sel_dev<INLINE_PAIR>(st);
  for (unsigned i = 0; i < 4; ++i) {
    store32_le_dev(out + i * 8, st[i].x);
    store32_le_dev(out + i * 8 + 4, st[i].y);
  }
}

inline bool meets_target_dev(const uint8_t output[32], const uint8_t* const target) {
  for (unsigned i = 0; i < 32; ++i) {
    if (output[i] == target[i]) continue;
    return output[i] < target[i];
  }
  return true;
}

inline void store_etchash_result(
  EtchashResult* const result, const uint64_t nonce, const uint8_t output[32],
  const uint32_t mix_hash_words[8]
) {
  using atomic_ref = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space
  >;
  const uint32_t index = atomic_ref(result->count).fetch_add(1);
  if (index >= MAX_ETCHASH_OUTPUTS) return;

  result->nonce[index] = nonce;
  for (unsigned i = 0; i < 32; ++i) result->output[index][i] = output[i];
  for (unsigned i = 0; i < 8; ++i) store32_le_dev(result->mix_hash[index] + i * 4, mix_hash_words[i]);
}

inline uint32_t etchash_epoch(const uint32_t height) {
  // ETC doubled the epoch length at the ECIP-1099 fork: 30000 blocks before, 60000 after.
  // The two divisors differ (ETHASH_EPOCH vs ETCHASH_EPOCH) despite their similar names.
  return height < ETCHASH_FORK_BLOCK ? height / ETHASH_EPOCH : height / ETCHASH_EPOCH;
}

inline uint32_t etchash_seed_epoch(const uint32_t height) {
  return height / ETHASH_EPOCH;
}

static void format_duration_ms(char* out, size_t out_size, uint64_t ms) {
  if (ms < 1000) {
    std::snprintf(out, out_size, "%" PRIu64 "ms", ms);
  } else if (ms < 60000) {
    std::snprintf(out, out_size, "%.2f s", static_cast<double>(ms) / 1000.0);
  } else {
    std::snprintf(out, out_size, "%.2f min", static_cast<double>(ms) / 60000.0);
  }
}

void compute_light_cache(std::vector<uint32_t>& cache, const uint32_t epoch, const uint32_t seed_epoch) {
  const uint64_t cache_bytes = cache_sizes[epoch];
  cache.assign(cache_bytes / sizeof(uint32_t), 0);

  const ethash_h256_t seed = ethash_get_seedhash(seed_epoch);
  if (!ethash_compute_cache_nodes(cache.data(), cache_bytes, &seed)) {
    throw std::string("Can't calculate etchash light cache");
  }
}

bool is_zero_hash(const uint8_t* const hash) {
  for (unsigned i = 0; i < HASH_LEN; ++i) if (hash[i]) return false;
  return true;
}

uint32_t seed_epoch_from_hash(const uint8_t* const seed_hash) {
  constexpr uint32_t max_seed_epoch = 4096;
  for (uint32_t seed_epoch = 0; seed_epoch != max_seed_epoch; ++seed_epoch) {
    const ethash_h256_t current = ethash_get_seedhash(seed_epoch);
    if (std::memcmp(current.b, seed_hash, HASH_LEN) == 0) return seed_epoch;
  }
  throw std::string("Unknown etchash seed hash");
}

void resolve_epochs(
  const uint32_t block_height,
  const uint8_t* const seed_hash,
  uint32_t& epoch,
  uint32_t& seed_epoch
) {
  if (seed_hash && !is_zero_hash(seed_hash)) {
    seed_epoch = seed_epoch_from_hash(seed_hash);
    epoch = seed_epoch < ETCHASH_FORK_BLOCK / ETHASH_EPOCH ? seed_epoch : seed_epoch / 2;
    return;
  }
  epoch = etchash_epoch(block_height);
  seed_epoch = etchash_seed_epoch(block_height);
}
