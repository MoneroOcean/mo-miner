// Device-side KawPow code, shared verbatim by the AOT build (this file is #included by
// kawpow.cpp) and by the DPC++ CUDA runtime-source JIT path (read at runtime and appended
// after a prelude that defines the constants/structs/FastModData). Keep it self-contained:
// it may rely only on constants/structs/FastModData/fast_mod_dev being already defined.

inline uint32_t round_up(const uint32_t value, const uint32_t step) {
  return ((value + step - 1) / step) * step;
}

// Force-inline the program-application chain ONLY on the runtime-JIT path, where a baked
// `const KawpowProgram` must const-propagate through it (the interpreter switches then fold to
// single ops). On the AOT build (Intel icx / NVIDIA AOT) plain `inline` is used so the compiler
// chooses inlining itself — forcing always_inline there spills registers and cost ~16% on Intel.
// MO_KAW_JIT_FOLD is defined only by the JIT prelude (sycl/kawpow/jit.inc).
#if defined(MO_KAW_JIT_FOLD) && defined(__clang__)
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

// CNT_CACHE/CNT_MATH are template non-type params (default to the KawPoW 11/18) so meowpow can run the
// shorter 6/9 sequence with the SAME KawpowProgram layout (arrays still sized at the 18/11 max; only
// the active prefix is generated/applied). KawPoW/FiroPow/EvrProgPow keep the defaults -> bit-identical.
template<unsigned I, typename LocalDag,
         unsigned CNT_CACHE = KAWPOW_CNT_CACHE, unsigned CNT_MATH = KAWPOW_CNT_MATH>
MO_KAW_AINLINE void apply_progpow_ops_dev(
  uint32_t* mix, const LocalDag& c_dag, const KawpowProgram& program
) {
  if constexpr (I < CNT_MATH) {
    if constexpr (I < CNT_CACHE) {
      const KawpowCacheOp op = program.cache[I];
      const uint32_t data = c_dag[mix[op.src] & (KAWPOW_CACHE_WORDS - 1)];
      random_merge_dev(mix[op.dst], data, op.merge_mode, op.merge_shift);
    }

    const KawpowMathOp op = program.math[I];
    const uint32_t data = random_math_dev(mix[op.src1], mix[op.src2], op.math_selector);
    random_merge_dev(mix[op.dst], data, op.merge_mode, op.merge_shift);

    apply_progpow_ops_dev<I + 1, LocalDag, CNT_CACHE, CNT_MATH>(mix, c_dag, program);
  }
}

template<unsigned I>
MO_KAW_AINLINE void apply_progpow_data_dev(
  uint32_t* mix, const DagLoad data_dag, const KawpowProgram& program
) {
  if constexpr (I < KAWPOW_DATA_LOADS) {
    const KawpowDataOp op = program.data[I];
    random_merge_dev(mix[op.dst], data_dag.s[I], op.merge_mode, op.merge_shift);
    apply_progpow_data_dev<I + 1>(mix, data_dag, program);
  }
}

#include "keccak.inc"
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
    s0 = mom_select_from_group(sg, state2[0], lane(h0));
    s1 = mom_select_from_group(sg, state2[1], lane(h0));
  }

  inline uint32_t offset(const uint32_t mix0, const uint32_t src_lane) const {
    return mom_select_from_group(sg, mix0, lane(src_lane));
  }

  inline void digest(uint32_t digest_out[8], const uint32_t lane_hash, const bool keep) const {
    // Each lane j folds fnv1a(fnv1a(basis, lane_hash[j]), lane_hash[j + 8]); lanes
    // 0..7 hold the eight results, which are then distributed to every lane.
    // shift_group_left by 8 stays within each 16-lane group on a 32-wide warp
    // (groups are 16-aligned), so only the distribution select needs rebasing.
    const uint32_t other = mom_shift_group_left(sg, lane_hash, 8);
    uint32_t fold = FNV_OFFSET_BASIS;
    fnv1a(fold, lane_hash);
    fnv1a(fold, other);
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      const uint32_t value = mom_select_from_group(sg, fold, lane(i));
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
// REGS/CNT_CACHE/CNT_MATH are template non-type params (default to the KawPoW 32/11/18); meowpow
// instantiates 16/6/9. mix[] is always sized at the 32-reg max so the KawpowProgram layout is shared.
template <typename Exchange, typename LocalDag,
          unsigned REGS = KAWPOW_REGS, unsigned CNT_CACHE = KAWPOW_CNT_CACHE,
          unsigned CNT_MATH = KAWPOW_CNT_MATH>
MO_KAW_AINLINE void kawpow_search_dev(
  const Exchange& ex, const uint32_t lane_id, const uint32_t state2[8],
  const LocalDag& c_dag, const DagLoad* const __restrict__ dag, const KawpowProgram& program,
  const FastModData dag_mod, uint32_t digest[8]
) {
  for (uint32_t h0 = 0; h0 < KAWPOW_LANES; ++h0) {
    uint32_t mix[KAWPOW_REGS];
    uint32_t s0, s1;
    ex.seed(s0, s1, state2, h0);
    fill_mix_dev<REGS>(s0, s1, lane_id, mix);

    for (uint32_t loop = 0; loop < KAWPOW_CNT_DAG; ++loop) {
      const uint32_t selected_offset = ex.offset(mix[0], loop & (KAWPOW_LANES - 1));
      uint32_t offset = fast_mod_dev(selected_offset, dag_mod);
      offset = offset * KAWPOW_LANES + ((lane_id ^ loop) & (KAWPOW_LANES - 1));
      const DagLoad data_dag = dag[offset];

      apply_progpow_ops_dev<0, LocalDag, CNT_CACHE, CNT_MATH>(mix, c_dag, program);
      apply_progpow_data_dev<0>(mix, data_dag, program);
    }

    uint32_t lane_hash = FNV_OFFSET_BASIS;
    for (unsigned i = 0; i < REGS; ++i) lane_hash = fnv1a(lane_hash, mix[i]);

    ex.digest(digest, lane_hash, h0 == lane_id);
  }
}

// ProgPoW seal variant selector. KawPoW and EvrProgPow share the same keccak-f800 seal structure
// (15 magic words absorbed into the high state words), differing only in the magic data; FiroPow uses
// a structurally different seal with NO magic array -- just keccak padding constants. The seal is
// selected per-algo via the SealMode passed to the kernel; the program/DAG machinery is identical for
// all three (only EPOCH_LENGTH and PERIOD_LENGTH change, and those live host-side).
enum class SealMode : uint32_t { KAWPOW = 0, FIRO = 1 };

// Initial keccak-f800 absorb of header+nonce. KAWPOW/EVR: st[10..24] = the 15-word magic array.
// FIRO: st[10]=0x00000001, st[18]=0x80008081, all other high words zero (keccak padding, no magic).
// `magic` is read only in KAWPOW mode (FIRO ignores it).
inline void kawpow_initial_dev(
  const uint8_t* const input, const uint32_t nonce_low, const uint32_t nonce_high,
  const SealMode seal, const uint32_t* const magic, uint32_t state2[8]
) {
  uint32_t st[25];
  const uint32_t* const job_words = reinterpret_cast<const uint32_t*>(input);
  for (unsigned i = 0; i < 8; ++i) st[i] = job_words[i];
  st[8] = nonce_low;
  st[9] = nonce_high;
  if (seal == SealMode::FIRO) {
    for (unsigned i = 10; i < 25; ++i) st[i] = 0;
    st[10] = 0x00000001U;
    st[18] = 0x80008081U;
  } else {
    for (unsigned i = 10; i < 25; ++i) st[i] = magic[i - 10];
  }
  keccak_f800_dev(st);
  for (unsigned i = 0; i < 8; ++i) state2[i] = st[i];
}

// Final keccak-f800 absorb of seed-hash (state2) + digest (mix). KAWPOW/EVR: st[16..24] = magic[0..8].
// FIRO: st[17]=0x00000001, st[24]=0x80008081, all other high words (st[16], st[18..23]) zero.
inline void kawpow_final_dev(
  const uint32_t state2[8], const uint32_t digest[8],
  const SealMode seal, const uint32_t* const magic, uint32_t final_state[25]
) {
  for (unsigned i = 0; i < 8; ++i) {
    final_state[i] = state2[i];
    final_state[i + 8] = digest[i];
  }
  if (seal == SealMode::FIRO) {
    for (unsigned i = 16; i < 25; ++i) final_state[i] = 0;
    final_state[17] = 0x00000001U;
    final_state[24] = 0x80008081U;
  } else {
    for (unsigned i = 16; i < 25; ++i) final_state[i] = magic[i - 16];
  }
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
