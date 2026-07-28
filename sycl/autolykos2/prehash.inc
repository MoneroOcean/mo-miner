struct B2bWord {
  uint32_t lo, hi;
};

inline uint32_t pad_message_hi_dev(const uint32_t item) {
  return ((item & 0xFFU) << 24) | ((item >> 8) << 16);
}

struct PrehashBlock0Message { // word 0 carries index|height, words 1..15 are pad items 0..14
  uint32_t index_be, height_be;
  inline B2bWord word(const unsigned i) const {
    return i == 0 ? B2bWord{index_be, height_be} : B2bWord{0, pad_message_hi_dev(i - 1)};
  }
};

struct PrehashPadMessage { // blocks 1..63: word i is pad item 16*block - 1 + i
  uint32_t base;
  inline B2bWord word(const unsigned i) const {
    return B2bWord{0, pad_message_hi_dev(base + i)};
  }
};

struct PrehashFinalMessage { // word 0 is pad item 1023, the rest is padding zeros
  inline B2bWord word(const unsigned i) const {
    return i == 0 ? B2bWord{0, pad_message_hi_dev(CONST_MES_SIZE / sizeof(uint64_t) - 1)}
                  : B2bWord{0, 0};
  }
};

inline void blake2b_g_pair(
  B2bWord& a, B2bWord& b, B2bWord& c, B2bWord& d, const B2bWord x, const B2bWord y
) {
  uint32_t lo = a.lo + b.lo;
  uint32_t carry = lo < b.lo ? 1U : 0U;
  a.lo = lo + x.lo;
  carry += a.lo < x.lo ? 1U : 0U;
  a.hi = a.hi + b.hi + x.hi + carry;
  { // d = rotr64(d ^ a, 32): swap halves
    const uint32_t t = d.lo ^ a.lo;
    d.lo = d.hi ^ a.hi;
    d.hi = t;
  }
  lo = c.lo + d.lo;
  c.hi = c.hi + d.hi + (lo < d.lo ? 1U : 0U);
  c.lo = lo;
  { // b = rotr64(b ^ c, 24)
    const uint32_t bl = b.lo ^ c.lo, bh = b.hi ^ c.hi;
    b.lo = (bl >> 24) | (bh << 8);
    b.hi = (bh >> 24) | (bl << 8);
  }
  lo = a.lo + b.lo;
  carry = lo < b.lo ? 1U : 0U;
  a.lo = lo + y.lo;
  carry += a.lo < y.lo ? 1U : 0U;
  a.hi = a.hi + b.hi + y.hi + carry;
  { // d = rotr64(d ^ a, 16)
    const uint32_t dl = d.lo ^ a.lo, dh = d.hi ^ a.hi;
    d.lo = (dl >> 16) | (dh << 16);
    d.hi = (dh >> 16) | (dl << 16);
  }
  lo = c.lo + d.lo;
  c.hi = c.hi + d.hi + (lo < d.lo ? 1U : 0U);
  c.lo = lo;
  { // b = rotr64(b ^ c, 63) == rotl64(b ^ c, 1)
    const uint32_t bl = b.lo ^ c.lo, bh = b.hi ^ c.hi;
    b.lo = (bl << 1) | (bh >> 31);
    b.hi = (bh << 1) | (bl >> 31);
  }
}

template <typename Message>
inline void blake2b_compress_pair_dev(B2bWord h[8], const Message& msg, const uint32_t t, const bool last) {
  B2bWord v[16];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) v[i] = h[i];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    const uint64_t iv = mom_sycl_portable_opencl ? blake2b_iv(i) : BLAKE2B_IV[i];
    v[i + 8] = B2bWord{static_cast<uint32_t>(iv), static_cast<uint32_t>(iv >> 32)};
  }
  v[12].lo ^= t; // the prehash message is 8200 bytes, the counter never reaches the high half
  if (last) {
    v[14].lo = ~v[14].lo;
    v[14].hi = ~v[14].hi;
  }

#if defined(MOM_SYCL_PORTABLE_OPENCL)
#pragma unroll 1
#else
#pragma unroll
#endif
  for (unsigned r = 0; r < 12; ++r) {
    if constexpr (mom_sycl_portable_opencl) {
      blake2b_g_pair(v[0], v[4], v[ 8], v[12], msg.word(blake2b_sigma(r,  0)), msg.word(blake2b_sigma(r,  1)));
      blake2b_g_pair(v[1], v[5], v[ 9], v[13], msg.word(blake2b_sigma(r,  2)), msg.word(blake2b_sigma(r,  3)));
      blake2b_g_pair(v[2], v[6], v[10], v[14], msg.word(blake2b_sigma(r,  4)), msg.word(blake2b_sigma(r,  5)));
      blake2b_g_pair(v[3], v[7], v[11], v[15], msg.word(blake2b_sigma(r,  6)), msg.word(blake2b_sigma(r,  7)));
      blake2b_g_pair(v[0], v[5], v[10], v[15], msg.word(blake2b_sigma(r,  8)), msg.word(blake2b_sigma(r,  9)));
      blake2b_g_pair(v[1], v[6], v[11], v[12], msg.word(blake2b_sigma(r, 10)), msg.word(blake2b_sigma(r, 11)));
      blake2b_g_pair(v[2], v[7], v[ 8], v[13], msg.word(blake2b_sigma(r, 12)), msg.word(blake2b_sigma(r, 13)));
      blake2b_g_pair(v[3], v[4], v[ 9], v[14], msg.word(blake2b_sigma(r, 14)), msg.word(blake2b_sigma(r, 15)));
    } else {
      blake2b_g_pair(v[0], v[4], v[ 8], v[12], msg.word(BLAKE2B_SIGMA[r][ 0]), msg.word(BLAKE2B_SIGMA[r][ 1]));
      blake2b_g_pair(v[1], v[5], v[ 9], v[13], msg.word(BLAKE2B_SIGMA[r][ 2]), msg.word(BLAKE2B_SIGMA[r][ 3]));
      blake2b_g_pair(v[2], v[6], v[10], v[14], msg.word(BLAKE2B_SIGMA[r][ 4]), msg.word(BLAKE2B_SIGMA[r][ 5]));
      blake2b_g_pair(v[3], v[7], v[11], v[15], msg.word(BLAKE2B_SIGMA[r][ 6]), msg.word(BLAKE2B_SIGMA[r][ 7]));
      blake2b_g_pair(v[0], v[5], v[10], v[15], msg.word(BLAKE2B_SIGMA[r][ 8]), msg.word(BLAKE2B_SIGMA[r][ 9]));
      blake2b_g_pair(v[1], v[6], v[11], v[12], msg.word(BLAKE2B_SIGMA[r][10]), msg.word(BLAKE2B_SIGMA[r][11]));
      blake2b_g_pair(v[2], v[7], v[ 8], v[13], msg.word(BLAKE2B_SIGMA[r][12]), msg.word(BLAKE2B_SIGMA[r][13]));
      blake2b_g_pair(v[3], v[4], v[ 9], v[14], msg.word(BLAKE2B_SIGMA[r][14]), msg.word(BLAKE2B_SIGMA[r][15]));
    }
  }

#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    h[i].lo ^= v[i].lo ^ v[i + 8].lo;
    h[i].hi ^= v[i].hi ^ v[i + 8].hi;
  }
}

inline void autolykos_prehash_digest_fast_dev(const uint32_t index, const uint32_t height, uint8_t digest[32]) {
  B2bWord h[8];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    const uint64_t iv = mom_sycl_portable_opencl ? blake2b_iv(i) : BLAKE2B_IV[i];
    h[i] = B2bWord{static_cast<uint32_t>(iv), static_cast<uint32_t>(iv >> 32)};
  }
  h[0].lo ^= 0x01010020U;

  const uint32_t index_be = mom_sycl_portable_opencl ? bswap32_dev(index)
    : (index >> 24) | ((index >> 8) & 0xFF00U) | ((index << 8) & 0xFF0000U) | (index << 24);
  const uint32_t height_be = mom_sycl_portable_opencl ? bswap32_dev(height)
    : (height >> 24) | ((height >> 8) & 0xFF00U) | ((height << 8) & 0xFF0000U) | (height << 24);
  blake2b_compress_pair_dev(h, PrehashBlock0Message{index_be, height_be}, 128, false);
  for (uint32_t block = 1; block < 64; ++block) {
    blake2b_compress_pair_dev(h, PrehashPadMessage{16U * block - 1U}, (block + 1U) * 128U, false);
  }
  blake2b_compress_pair_dev(h, PrehashFinalMessage{}, CONST_MES_SIZE + 8U, true);

#pragma unroll
  for (unsigned i = 0; i < 32; ++i) {
    const B2bWord word = h[i >> 3];
    const unsigned byte = i & 7U;
    digest[i] = static_cast<uint8_t>((byte < 4 ? word.lo >> (byte * 8U) : word.hi >> ((byte - 4U) * 8U)));
  }
}

// A table row is the digest packed big-endian into 8 words, with its top byte (digest[0])
// forced to zero — load32_be of word 0 reads digest[0..3], so just mask that byte off.
AUTOLYKOS_DEVICE_INLINE void digest_to_limbs_dev(const uint8_t digest[32], uint32_t limbs[8]) {
  limbs[0] = load32_be_dev(digest) & 0x00FFFFFFU;
#pragma unroll
  for (unsigned i = 1; i < 8; ++i) limbs[i] = load32_be_dev(digest + i * 4);
}

inline void autolykos_table_row_store_dev(uint32_t* const table, const uint32_t gid, const uint32_t height) {
  uint8_t digest[32];
  autolykos_prehash_digest_fast_dev(gid, height, digest);
  digest_to_limbs_dev(digest, table + static_cast<uint64_t>(gid) * (TABLE_ENTRY_BYTES / sizeof(uint32_t)));
}

AUTOLYKOS_DEVICE_INLINE void table_item_limbs_dev(
  const uint32_t* const table, const uint32_t index, const uint32_t height, uint32_t limbs[8]
) {
  if (table) {
    const uint32_t* const item = table + static_cast<uint64_t>(index) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) limbs[i] = item[i];
    return;
  }

  uint8_t digest[32];
  autolykos_prehash_digest_dev(index, height, digest);
  digest_to_limbs_dev(digest, limbs);
}

// 256-bit big-endian add: sum[0..7] += add[0..7], propagating carry from least- (i=7) to
// most-significant limb. The Autolykos2 sum wraps mod 2^256, so the final carry is dropped.
AUTOLYKOS_DEVICE_INLINE void add_limbs_dev(uint32_t sum[8], const uint32_t* const add) {
  uint64_t carry = 0;
  for (int i = 7; i >= 0; --i) {
    const uint64_t v = static_cast<uint64_t>(sum[i]) + add[i] + carry;
    sum[i] = static_cast<uint32_t>(v);
    carry = v >> 32;
  }
}

AUTOLYKOS_DEVICE_INLINE uint32_t autolykos_index_from_words_dev(
  const uint32_t word0, const uint32_t word1, const unsigned byte_offset,
  const uint32_t n_len, const uint64_t n_len_M
) {
  uint32_t idx;
  switch (byte_offset) {
    case 0: idx = word0; break;
    case 1:
      if constexpr (mom_sycl_portable_opencl) { volatile uint32_t a = word0 << 8, b = word1 >> 24; idx = a | b; }
      else idx = (word0 << 8) | (word1 >> 24);
      break;
    case 2:
      if constexpr (mom_sycl_portable_opencl) { volatile uint32_t a = word0 << 16, b = word1 >> 16; idx = a | b; }
      else idx = (word0 << 16) | (word1 >> 16);
      break;
    default:
      if constexpr (mom_sycl_portable_opencl) { volatile uint32_t a = word0 << 24, b = word1 >> 8; idx = a | b; }
      else idx = (word0 << 24) | (word1 >> 8);
      break;
  }
  return mo_mod_u32(idx, n_len, n_len_M);
}

AUTOLYKOS_DEVICE_INLINE bool meets_target_dev(const uint8_t output[32], const uint8_t target[32]) {
  for (unsigned i = 0; i < 32; ++i) {
    if (output[i] == target[i]) continue;
    return output[i] < target[i];
  }
  return true;
}

inline void store_autolykos_result(
  AutolykosResult* const result, const uint64_t nonce, const uint8_t output[32]
) {
  using atomic_ref = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space
  >;
  const uint32_t index = atomic_ref(result->count).fetch_add(1);
  if (index >= MAX_AUTOLYKOS_OUTPUTS) return;

  result->nonce[index] = nonce;
#pragma unroll
  for (unsigned i = 0; i < 32; ++i) result->output[index][i] = output[i];
}

// Autolykos2 per-nonce prehash: blake2b(message||nonce) selects table row h3, then
// index_hash = blake2b(row[1..31] || message || nonce) seeds the K_LEN table reads.
// Row bytes come from the prebuilt table when present, else from the on-the-fly digest.
AUTOLYKOS_DEVICE_INLINE void autolykos_index_hash_dev(
  const uint8_t message[32], const uint64_t nonce, const uint32_t* const table,
  const uint32_t height, const uint32_t n_len, const uint64_t n_len_divM, uint8_t index_hash[32]
) {
  uint8_t nonce_be[8];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) nonce_be[i] = static_cast<uint8_t>(nonce >> ((7U - i) * 8U));

  uint8_t h1_input[40];
#pragma unroll
  for (unsigned i = 0; i < 32; ++i) h1_input[i] = message[i];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) h1_input[32 + i] = nonce_be[i];

  uint8_t h1[32];
  blake2b256_oneblock_dev(h1_input, sizeof(h1_input), h1);
  const uint32_t h3 = mo_mod_u64(load64_be_dev(h1 + 24), n_len, n_len_divM);

  uint8_t seed[71]; // row[1..31] (31) || message (32) || nonce (8)
  if (table) {
    const uint32_t* const h3_item = table + static_cast<uint64_t>(h3) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
#pragma unroll
    for (unsigned i = 0; i < 31; ++i) {
      const unsigned pos = i + 1U;
      seed[i] = word_byte_be_dev(h3_item[pos >> 2], pos & 3U);
    }
  } else {
    uint8_t h3_digest[32];
    autolykos_prehash_digest_dev(h3, height, h3_digest);
#pragma unroll
    for (unsigned i = 0; i < 31; ++i) seed[i] = h3_digest[i + 1];
  }
#pragma unroll
  for (unsigned i = 0; i < 32; ++i) seed[31 + i] = message[i];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) seed[63 + i] = nonce_be[i];

  blake2b256_oneblock_dev(seed, sizeof(seed), index_hash);
}

#if defined(MOM_SYCL_PORTABLE_OPENCL)
AUTOLYKOS_DEVICE_INLINE void autolykos_index_hash_portable_dev(
  const uint8_t message[32], const uint64_t nonce, const uint32_t height,
  const uint32_t n_len, const uint64_t n_len_divM, uint8_t index_hash[32]
) {
  uint8_t nonce_be[8];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) nonce_be[i] = static_cast<uint8_t>(nonce >> ((7U - i) * 8U));

  uint8_t h1_input[40];
#pragma unroll
  for (unsigned i = 0; i < 32; ++i) h1_input[i] = message[i];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) h1_input[32 + i] = nonce_be[i];

  uint8_t h1[32];
  blake2b256_oneblock_dev(h1_input, sizeof(h1_input), h1);
  const uint32_t h3 = mo_mod_u64(load64_be_dev(h1 + 24), n_len, n_len_divM);
  uint8_t h3_digest[32];
  autolykos_prehash_digest_dev(h3, height, h3_digest);

  uint8_t seed[71];
#pragma unroll
  for (unsigned i = 0; i < 31; ++i) seed[i] = h3_digest[i + 1];
#pragma unroll
  for (unsigned i = 0; i < 32; ++i) seed[31 + i] = message[i];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) seed[63 + i] = nonce_be[i];
  blake2b256_oneblock_dev(seed, sizeof(seed), index_hash);
}
#endif
