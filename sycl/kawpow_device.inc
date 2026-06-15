// Device-side KawPow code, shared verbatim by the AOT build (this file is #included by
// kawpow.cpp) and by the DPC++ CUDA runtime-source JIT path (read at runtime and appended
// after a prelude that defines the constants/structs/FastModData). Keep it self-contained:
// it may rely only on constants/structs/FastModData/fast_mod_dev being already defined.

inline uint32_t round_up(const uint32_t value, const uint32_t step) {
  return ((value + step - 1) / step) * step;
}

// Force-inline the program-application chain so a baked `const KawpowProgram` const-propagates
// through it on the runtime JIT (the interpreter switches then fold to single ops).
#if defined(__clang__)
#define MO_KAW_AINLINE inline __attribute__((always_inline))
#else
#define MO_KAW_AINLINE inline
#endif

MO_KAW_AINLINE uint32_t fnv1a(uint32_t& h, const uint32_t d) {
  h = (h ^ d) * FNV_PRIME;
  return h;
}

inline uint32_t kiss99(Kiss99& st) {
  st.z = 36969 * (st.z & 65535) + (st.z >> 16);
  st.w = 18000 * (st.w & 65535) + (st.w >> 16);
  const uint32_t mwc = (st.z << 16) + st.w;
  st.jsr ^= st.jsr << 17;
  st.jsr ^= st.jsr >> 13;
  st.jsr ^= st.jsr << 5;
  st.jcong = 69069 * st.jcong + 1234567;
  return (mwc ^ st.jcong) + st.jsr;
}

inline uint32_t merge_shift(const uint32_t selector) {
  return ((selector >> 16) % 31) + 1;
}

MO_KAW_AINLINE uint32_t rotl32_dev(const uint32_t value, const uint32_t shift) {
  const uint32_t s = shift & 31U;
  return (value << s) | (value >> ((0U - s) & 31U));
}

MO_KAW_AINLINE uint32_t rotr32_dev(const uint32_t value, const uint32_t shift) {
  const uint32_t s = shift & 31U;
  return (value >> s) | (value << ((0U - s) & 31U));
}

MO_KAW_AINLINE uint32_t fnv_dev(const uint32_t x, const uint32_t y) {
  return x * FNV_PRIME ^ y;
}

MO_KAW_AINLINE uint32_t mul_hi_dev(const uint32_t a, const uint32_t b) {
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
}

MO_KAW_AINLINE uint32_t random_math_dev(
  const uint32_t a, const uint32_t b, const uint32_t selector
) {
  switch (selector) {
    case 0: return a + b;
    case 1: return a * b;
    case 2: return mul_hi_dev(a, b);
    case 3: return sycl::min(a, b);
    case 4: return rotl32_dev(a, b);
    case 5: return rotr32_dev(a, b);
    case 6: return a & b;
    case 7: return a | b;
    case 8: return a ^ b;
    case 9: return sycl::clz(a) + sycl::clz(b);
    default: return sycl::popcount(a) + sycl::popcount(b);
  }
}

MO_KAW_AINLINE void random_merge_dev(
  uint32_t& a, const uint32_t b, const uint32_t mode, const uint32_t shift
) {
  switch (mode) {
    case 0: a = (a * 33) + b; break;
    case 1: a = (a ^ b) * 33; break;
    case 2: a = rotl32_dev(a, shift) ^ b; break;
    default: a = rotr32_dev(a, shift) ^ b; break;
  }
}

template<unsigned I, typename LocalDag>
MO_KAW_AINLINE void apply_progpow_ops_dev(
  uint32_t mix[KAWPOW_REGS], const LocalDag& c_dag, const KawpowProgram& program
) {
  if constexpr (I < KAWPOW_CNT_MATH) {
    if constexpr (I < KAWPOW_CNT_CACHE) {
      const KawpowCacheOp op = program.cache[I];
      const uint32_t data = c_dag[mix[op.src] & (KAWPOW_CACHE_WORDS - 1)];
      random_merge_dev(mix[op.dst], data, op.merge_mode, op.merge_shift);
    }

    const KawpowMathOp op = program.math[I];
    const uint32_t data = random_math_dev(mix[op.src1], mix[op.src2], op.math_selector);
    random_merge_dev(mix[op.dst], data, op.merge_mode, op.merge_shift);

    apply_progpow_ops_dev<I + 1>(mix, c_dag, program);
  }
}

template<unsigned I>
MO_KAW_AINLINE void apply_progpow_data_dev(
  uint32_t mix[KAWPOW_REGS], const DagLoad data_dag, const KawpowProgram& program
) {
  if constexpr (I < KAWPOW_DATA_LOADS) {
    const KawpowDataOp op = program.data[I];
    random_merge_dev(mix[op.dst], data_dag.s[I], op.merge_mode, op.merge_shift);
    apply_progpow_data_dev<I + 1>(mix, data_dag, program);
  }
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

inline void keccakf1600_pair_dev(Uint2 st[25]) {
  for (unsigned round = 0; round < 24; ++round) keccakf1600_pair_round_dev(st, round);
}

inline void keccak_512_words_dev(uint32_t words[NODE_WORDS]) {
  Uint2 st[25]{};
  for (unsigned i = 0; i < 8; ++i) {
    st[i] = make_u2(words[i * 2], words[i * 2 + 1]);
  }
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

MO_KAW_AINLINE void fill_mix_dev(const uint32_t seed0, const uint32_t seed1, const uint32_t lane, uint32_t mix[KAWPOW_REGS]) {
  uint32_t fnv_hash = FNV_OFFSET_BASIS;
  Kiss99 st{};
  st.z = fnv1a(fnv_hash, seed0);
  st.w = fnv1a(fnv_hash, seed1);
  st.jsr = fnv1a(fnv_hash, lane);
  st.jcong = fnv1a(fnv_hash, lane);

  for (unsigned i = 0; i < KAWPOW_REGS; ++i) mix[i] = kiss99(st);
}

inline uint32_t bswap32_dev(const uint32_t value) {
  return ((value & 0x000000FFU) << 24) |
         ((value & 0x0000FF00U) << 8) |
         ((value & 0x00FF0000U) >> 8) |
         ((value & 0xFF000000U) >> 24);
}

// Cross-lane exchange policies for the search loop. The GPU instantiation maps
// the 16 ProgPoW lanes onto one 16-wide subgroup and exchanges values with
// shuffles (no barriers); the SLM instantiation keeps the original local-memory
// exchange and runs on any device (CPU SYCL device in particular).

struct KawpowSgExchange {
  sycl::sub_group sg;
  // Base sub-group lane of this work-item's 16-lane ProgPoW group. On NVIDIA the
  // native sub-group is a 32-wide warp holding two ProgPoW groups, so shuffles
  // must stay within the group (base 0 or 16). On a 16-wide Intel sub-group this
  // is always 0, reproducing the original lane indices.
  uint32_t base;
  inline uint32_t lane(const uint32_t l) const { return base + l; }

  inline void seed(uint32_t& s0, uint32_t& s1, const uint32_t state2[8], const uint32_t h0) const {
    s0 = sycl::select_from_group(sg, state2[0], lane(h0));
    s1 = sycl::select_from_group(sg, state2[1], lane(h0));
  }

  inline uint32_t offset(const uint32_t mix0, const uint32_t src_lane) const {
    return sycl::select_from_group(sg, mix0, lane(src_lane));
  }

  inline void digest(uint32_t digest_out[8], const uint32_t lane_hash, const bool keep) const {
    // digest_temp[j] = fnv1a(fnv1a(basis, lane_hash[j]), lane_hash[j + 8]);
    // lanes 0..7 each fold their pair, then the eight values are distributed.
    // shift_group_left by 8 stays within each 16-lane group on a 32-wide warp
    // (groups are 16-aligned), so only the distribution select needs rebasing.
    const uint32_t other = sycl::shift_group_left(sg, lane_hash, 8);
    uint32_t fold = FNV_OFFSET_BASIS;
    fnv1a(fold, lane_hash);
    fnv1a(fold, other);
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      const uint32_t value = sycl::select_from_group(sg, fold, lane(i));
      if (keep) digest_out[i] = value;
    }
  }
};

struct KawpowSlmExchange {
  sycl::nd_item<1> item;
  uint32_t* share;    // this lane group's 16 share slots
  uint32_t* offsets;  // this lane group's offset slot
  uint32_t lane_id;
  bool cpu_offset_barrier;

  inline void seed(uint32_t& s0, uint32_t& s1, const uint32_t state2[8], const uint32_t h0) const {
    if (lane_id == h0) {
      share[0] = state2[0];
      share[1] = state2[1];
    }
    item.barrier(sycl::access::fence_space::local_space);
    s0 = share[0];
    s1 = share[1];
  }

  inline uint32_t offset(const uint32_t mix0, const uint32_t src_lane) const {
    if (lane_id == src_lane) *offsets = mix0;
    item.barrier(sycl::access::fence_space::local_space);
    const uint32_t selected = *offsets;
    if (cpu_offset_barrier) item.barrier(sycl::access::fence_space::local_space);
    return selected;
  }

  inline void digest(uint32_t digest_out[8], const uint32_t lane_hash, const bool keep) const {
    share[lane_id] = lane_hash;
    item.barrier(sycl::access::fence_space::local_space);

    uint32_t digest_temp[8];
    for (unsigned i = 0; i < 8; ++i) digest_temp[i] = FNV_OFFSET_BASIS;
    for (unsigned i = 0; i < KAWPOW_LANES; ++i) {
      digest_temp[i % 8] = fnv1a(digest_temp[i % 8], share[i]);
    }
    if (keep) {
      for (unsigned i = 0; i < 8; ++i) digest_out[i] = digest_temp[i];
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
};

// One full ProgPoW search round for this work-item's lane group: 16 hashes are
// computed cooperatively; the work-item keeps the digest of its own nonce.
template <typename Exchange, typename LocalDag>
MO_KAW_AINLINE void kawpow_search_dev(
  const Exchange& ex, const uint32_t lane_id, const uint32_t state2[8],
  const LocalDag& c_dag, const DagLoad* const __restrict__ dag, const KawpowProgram& program,
  const FastModData dag_mod, uint32_t digest[8]
) {
  for (uint32_t h0 = 0; h0 < KAWPOW_LANES; ++h0) {
    uint32_t mix[KAWPOW_REGS];
    uint32_t s0, s1;
    ex.seed(s0, s1, state2, h0);
    fill_mix_dev(s0, s1, lane_id, mix);

    for (uint32_t loop = 0; loop < KAWPOW_CNT_DAG; ++loop) {
      const uint32_t selected_offset = ex.offset(mix[0], loop & (KAWPOW_LANES - 1));
      uint32_t offset = fast_mod_dev(selected_offset, dag_mod);
      offset = offset * KAWPOW_LANES + ((lane_id ^ loop) & (KAWPOW_LANES - 1));
      const DagLoad data_dag = dag[offset];

      apply_progpow_ops_dev<0>(mix, c_dag, program);
      apply_progpow_data_dev<0>(mix, data_dag, program);
    }

    uint32_t lane_hash = FNV_OFFSET_BASIS;
    for (unsigned i = 0; i < KAWPOW_REGS; ++i) lane_hash = fnv1a(lane_hash, mix[i]);

    ex.digest(digest, lane_hash, h0 == lane_id);
  }
}

// Initial keccak-f800 absorb of header+nonce shared by both kernel variants.
inline void kawpow_initial_dev(
  const uint8_t* const input, const uint32_t nonce_low, const uint32_t nonce_high, uint32_t state2[8]
) {
  uint32_t st[25];
  const uint32_t* const job_words = reinterpret_cast<const uint32_t*>(input);
  for (unsigned i = 0; i < 8; ++i) st[i] = job_words[i];
  st[8] = nonce_low;
  st[9] = nonce_high;
  for (unsigned i = 10; i < 25; ++i) st[i] = RAVENCOIN_KAWPOW[i - 10];
  keccak_f800_dev(st);
  for (unsigned i = 0; i < 8; ++i) state2[i] = st[i];
}

inline void kawpow_final_dev(const uint32_t state2[8], const uint32_t digest[8], uint32_t final_state[25]) {
  for (unsigned i = 0; i < 8; ++i) final_state[i] = state2[i];
  for (unsigned i = 0; i < 8; ++i) final_state[i + 8] = digest[i];
  for (unsigned i = 16; i < 25; ++i) final_state[i] = RAVENCOIN_KAWPOW[i - 16];
  keccak_f800_dev(final_state);
}

inline bool kawpow_meets_target_words(const uint32_t output[8], const uint64_t target) {
  const uint32_t hash_hi = bswap32_dev(output[0]);
  const uint32_t hash_lo = bswap32_dev(output[1]);
  const uint32_t target_hi = static_cast<uint32_t>(target >> 32);
  const uint32_t target_lo = static_cast<uint32_t>(target);
  return hash_hi < target_hi || (hash_hi == target_hi && hash_lo <= target_lo);
}

inline void store_kawpow_result(
  KawpowResult* const result, const uint64_t nonce, const uint32_t output[8],
  const uint32_t mix_hash[8]
) {
  using atomic_ref = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space
  >;
  const uint32_t index = atomic_ref(result->count).fetch_add(1);
  if (index >= MAX_KAWPOW_OUTPUTS) return;

  result->nonce[index] = nonce;
  for (unsigned i = 0; i < 8; ++i) {
    result->output[index][i] = output[i];
    result->mix_hash[index][i] = mix_hash[i];
  }
}
