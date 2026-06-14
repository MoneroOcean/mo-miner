// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

// SYCL KawPow implementation based on XMRig's KawPow reference and OpenCL
// runner structure.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "lib-internal.h"
#include "../native/consts.h"
#include "../xmrig/3rdparty/libethash/ethash.h"
#include "../xmrig/3rdparty/libethash/data_sizes.h"

namespace mominer_kawpow {

constexpr uint32_t KAWPOW_EPOCH_LENGTH  = 7500;
constexpr uint32_t KAWPOW_PERIOD_LENGTH = 3;
constexpr uint32_t KAWPOW_LANES         = 16;
constexpr uint32_t KAWPOW_REGS          = 32;
constexpr uint32_t KAWPOW_DAG_LOADS     = 4;
constexpr uint32_t KAWPOW_CACHE_WORDS   = 4096;
constexpr uint32_t KAWPOW_CNT_DAG       = 64;
constexpr uint32_t KAWPOW_CNT_CACHE     = 11;
constexpr uint32_t KAWPOW_CNT_MATH      = 18;
constexpr uint32_t KAWPOW_DATA_LOADS    = 256 / (sizeof(uint32_t) * KAWPOW_LANES);
constexpr uint32_t KAWPOW_DATASET_PARENTS = 512;
constexpr uint32_t NODE_WORDS           = 16;
constexpr uint32_t FNV_PRIME            = 0x01000193U;
constexpr uint32_t FNV_OFFSET_BASIS     = 0x811c9dc5U;
constexpr uint32_t MAX_KAWPOW_OUTPUTS   = 15;

constexpr uint32_t RAVENCOIN_KAWPOW[15] = {
  0x00000072, 0x00000041, 0x00000056, 0x00000045, 0x0000004E,
  0x00000043, 0x0000004F, 0x00000049, 0x0000004E, 0x0000004B,
  0x00000041, 0x00000057, 0x00000050, 0x0000004F, 0x00000057
};

// Used by MOMINER_LOOP_STATS to break a dispatch call into host-side phases.
static uint64_t kawpow_now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()
  ).count();
}

static bool kawpow_loop_stats() {
  static const bool enabled = std::getenv("MOMINER_LOOP_STATS") != nullptr;
  return enabled;
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

struct KawpowCacheOp {
  uint32_t src;
  uint32_t dst;
  uint32_t merge_mode;
  uint32_t merge_shift;
};

struct KawpowMathOp {
  uint32_t src1;
  uint32_t src2;
  uint32_t dst;
  uint32_t math_selector;
  uint32_t merge_mode;
  uint32_t merge_shift;
};

struct KawpowDataOp {
  uint32_t dst;
  uint32_t merge_mode;
  uint32_t merge_shift;
};

struct KawpowProgram {
  KawpowCacheOp cache[KAWPOW_CNT_CACHE];
  KawpowMathOp math[KAWPOW_CNT_MATH];
  KawpowDataOp data[KAWPOW_DATA_LOADS];
};

#if !defined(MOMINER_ACPP)
constexpr sycl::specialization_id<KawpowProgram> kawpow_program_id;
constexpr sycl::specialization_id<FastModData> kawpow_dag_mod_id;
constexpr sycl::specialization_id<bool> kawpow_cpu_offset_barrier_id;
#endif

struct KawpowResult {
  uint32_t count;
  uint64_t nonce[MAX_KAWPOW_OUTPUTS];
  uint32_t output[MAX_KAWPOW_OUTPUTS][8];
  uint32_t mix_hash[MAX_KAWPOW_OUTPUTS][8];
};

struct Uint2 {
  uint32_t x;
  uint32_t y;
};

struct DagLoad {
  uint32_t s[KAWPOW_DAG_LOADS];
};

struct Kiss99 {
  uint32_t z;
  uint32_t w;
  uint32_t jsr;
  uint32_t jcong;
};

inline uint32_t round_up(const uint32_t value, const uint32_t step) {
  return ((value + step - 1) / step) * step;
}

inline uint32_t fnv1a(uint32_t& h, const uint32_t d) {
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

KawpowProgram make_program(const uint64_t period) {
  KawpowProgram program{};
  uint32_t dst_seq[KAWPOW_REGS];
  uint32_t src_seq[KAWPOW_REGS];

  for (uint32_t i = 0; i < KAWPOW_REGS; ++i) {
    dst_seq[i] = i;
    src_seq[i] = i;
  }

  uint32_t fnv_hash = FNV_OFFSET_BASIS;
  Kiss99 rnd{
    fnv1a(fnv_hash, static_cast<uint32_t>(period)),
    fnv1a(fnv_hash, static_cast<uint32_t>(period >> 32)),
    fnv1a(fnv_hash, static_cast<uint32_t>(period)),
    fnv1a(fnv_hash, static_cast<uint32_t>(period >> 32))
  };

  for (uint32_t i = KAWPOW_REGS; i > 1; --i) {
    std::swap(dst_seq[i - 1], dst_seq[kiss99(rnd) % i]);
    std::swap(src_seq[i - 1], src_seq[kiss99(rnd) % i]);
  }

  uint32_t dst_counter = 0;
  uint32_t src_counter = 0;
  uint32_t cache_counter = 0;
  uint32_t math_counter = 0;

  for (uint32_t i = 0; i < KAWPOW_CNT_CACHE || i < KAWPOW_CNT_MATH; ++i) {
    if (i < KAWPOW_CNT_CACHE) {
      program.cache[cache_counter++] = {
        src_seq[(src_counter++) % KAWPOW_REGS],
        dst_seq[(dst_counter++) % KAWPOW_REGS],
        0,
        0
      };
      const uint32_t selector = kiss99(rnd);
      auto& op = program.cache[cache_counter - 1];
      op.merge_mode = selector % 4;
      op.merge_shift = merge_shift(selector);
    }

    if (i < KAWPOW_CNT_MATH) {
      const uint32_t src_rnd = kiss99(rnd) % (KAWPOW_REGS * (KAWPOW_REGS - 1));
      const uint32_t src1 = src_rnd % KAWPOW_REGS;
      uint32_t src2 = src_rnd / KAWPOW_REGS;
      if (src2 >= src1) ++src2;

      const uint32_t math_selector = kiss99(rnd);
      const uint32_t merge_selector = kiss99(rnd);
      program.math[math_counter++] = {
        src1,
        src2,
        dst_seq[(dst_counter++) % KAWPOW_REGS],
        math_selector % 11,
        merge_selector % 4,
        merge_shift(merge_selector)
      };
    }
  }

  {
    const uint32_t selector = kiss99(rnd);
    program.data[0] = { 0, selector % 4, merge_shift(selector) };
  }
  for (uint32_t i = 1; i < KAWPOW_DATA_LOADS; ++i) {
    const uint32_t selector = kiss99(rnd);
    program.data[i] = {
      dst_seq[(dst_counter++) % KAWPOW_REGS],
      selector % 4,
      merge_shift(selector)
    };
  }

  return program;
}

void compute_light_cache(std::vector<uint32_t>& cache, const uint32_t epoch) {
  const uint64_t cache_bytes = cache_sizes[epoch];
  cache.assign(cache_bytes / sizeof(uint32_t), 0);

  const ethash_h256_t seed = ethash_get_seedhash(epoch);
  if (!ethash_compute_cache_nodes(cache.data(), cache_bytes, &seed)) {
    throw std::string("Can't calculate kawpow light cache");
  }
}

inline uint32_t rotl32_dev(const uint32_t value, const uint32_t shift) {
  const uint32_t s = shift & 31U;
  return (value << s) | (value >> ((0U - s) & 31U));
}

inline uint32_t rotr32_dev(const uint32_t value, const uint32_t shift) {
  const uint32_t s = shift & 31U;
  return (value >> s) | (value << ((0U - s) & 31U));
}

inline uint32_t fnv_dev(const uint32_t x, const uint32_t y) {
  return x * FNV_PRIME ^ y;
}

inline uint32_t mul_hi_dev(const uint32_t a, const uint32_t b) {
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
}

inline uint32_t random_math_dev(
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

inline void random_merge_dev(
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
inline void apply_progpow_ops_dev(
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
inline void apply_progpow_data_dev(
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

inline void fill_mix_dev(const uint32_t seed0, const uint32_t seed1, const uint32_t lane, uint32_t mix[KAWPOW_REGS]) {
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
inline void kawpow_search_dev(
  const Exchange& ex, const uint32_t lane_id, const uint32_t state2[8],
  const LocalDag& c_dag, const DagLoad* const dag, const KawpowProgram& program,
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

bool kawpow_meets_target_host(const uint32_t output[8], const uint64_t target) {
  return kawpow_meets_target_words(output, target);
}

// Named so the per-period specialized bundle can be built ahead of time.
class KawpowSgKernel;

class KawpowState {
public:
  sycl::device device;
  sycl::queue queue;
  std::unique_ptr<MOMINER_BUNDLE_T> bundle;
  bool shared_io;
  bool shared_dag;
  uint8_t* input = nullptr;
  uint32_t* light_cache = nullptr;
  uint32_t* dag = nullptr;
  KawpowResult* result = nullptr;
  uint32_t input_cap = 0;
  uint64_t light_cache_words = 0;
  uint64_t dag_words = 0;
  uint32_t epoch = UINT32_MAX;
  uint64_t period = UINT64_MAX;
  KawpowProgram program{};
  unsigned workgroup;
  std::mutex mutex;

#if !defined(MOMINER_ACPP)
  // Per-period executable bundles with the program/dag spec constants baked
  // in. The next period's bundle is built on a worker thread while the
  // current one mines, so the period switch no longer stalls hashing on JIT.
  std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> period_bundle;
  uint64_t period_bundle_period = UINT64_MAX;
  uint32_t period_bundle_epoch = UINT32_MAX;
  std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> next_bundle;
  uint64_t next_bundle_period = UINT64_MAX;
  uint32_t next_bundle_epoch = UINT32_MAX;
  std::thread prefetch_thread;
#endif

  explicit KawpowState(const std::string& dev_str)
    : device(get_dev(dev_str)),
      queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !device.has(sycl::aspect::usm_device_allocations)),
      shared_dag(device.is_cpu() || !device.has(sycl::aspect::usm_device_allocations)),
      workgroup(kawpow_workgroup(device))
  {
    if (!device.has(sycl::aspect::usm_shared_allocations) ||
        (!device.is_cpu() && !device.has(sycl::aspect::usm_device_allocations))) {
      throw std::string("kawpow SYCL device does not support required allocations");
    }

    set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", kawpow_compile_options(device));
    bundle = std::make_unique<MOMINER_BUNDLE_T>(
      MOMINER_GET_EXEC_BUNDLE(queue.get_context())
    );
  }

  ~KawpowState() {
#if !defined(MOMINER_ACPP)
    if (prefetch_thread.joinable()) prefetch_thread.join();
#endif
    release();
  }

  static unsigned kawpow_workgroup(const sycl::device& dev) {
#if defined(MOMINER_ACPP)
    // NVIDIA: a 64-thread block (two ProgPoW 16-lane groups + headroom) keeps the
    // register-heavy unrolled body resident and clocks high on the power-capped L4
    // (sweep: 64=14.9 > 128=14.5 > 256=13.7 MH/s). Intel keeps its 256 default.
    const unsigned fallback = sycl_default_workgroup(dev, {64, 128, 256, 512}, dev.is_cpu() ? 128 : 64);
#else
    const unsigned fallback = sycl_default_workgroup(dev, {64, 128, 256, 512}, dev.is_cpu() ? 128 : 256);
#endif
    const char* const value = std::getenv("MOMINER_KAWPOW_WORKGROUP");
    if (!value || !*value) return fallback;

    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno || end == value || *end) return fallback;

    switch (parsed) {
      case 64:
      case 128:
      case 256:
      case 512:
        return static_cast<unsigned>(parsed);
    }
    return fallback;
  }

  static unsigned kawpow_dag_workgroup(const sycl::device& dev) {
    const unsigned fallback = sycl_default_workgroup(dev, {32, 64, 128, 256, 512}, dev.is_cpu() ? 128 : 64);
    const char* const value = std::getenv("MOMINER_KAWPOW_DAG_WORKGROUP");
    if (!value || !*value) return fallback;

    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno || end == value || *end) return fallback;

    switch (parsed) {
      case 32:
      case 64:
      case 128:
      case 256:
      case 512:
        return static_cast<unsigned>(parsed);
    }
    return fallback;
  }

  static uint32_t kawpow_dag_chunk_nodes() {
    const char* const value = std::getenv("MOMINER_KAWPOW_DAG_CHUNK_NODES");
    if (!value || !*value) return 0;

    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno || end == value || *end || parsed > std::numeric_limits<uint32_t>::max()) return 0;
    return static_cast<uint32_t>(parsed);
  }

  static const char* kawpow_compile_options(const sycl::device& dev) {
    const char* const value = std::getenv("MOMINER_KAWPOW_COMPILE_OPTIONS");
    if (value) return value;
    return dev.is_gpu() ? "-O3" : "";
  }

  template<typename T>
  T* allocate(const size_t count, const bool shared) {
    return shared ? sycl::malloc_shared<T>(count, queue) : sycl::malloc_device<T>(count, queue);
  }

  void free_ptr(auto*& ptr) {
    if (ptr) sycl::free(ptr, queue);
    ptr = nullptr;
  }

  void release() {
    queue.wait_and_throw();
    free_ptr(input);
    free_ptr(light_cache);
    free_ptr(dag);
    free_ptr(result);
    input_cap = 0;
    light_cache_words = 0;
    dag_words = 0;
    epoch = UINT32_MAX;
    period = UINT64_MAX;
  }

  void ensure_input(const unsigned input_size) {
    if (input_size <= input_cap && result) return;
    queue.wait_and_throw();
    free_ptr(input);
    free_ptr(result);
    input_cap = input_size;
    input = allocate<uint8_t>(input_cap, shared_io);
    result = sycl::malloc_shared<KawpowResult>(1, queue);
    if (!input || !result) throw std::string("Can't allocate kawpow SYCL input buffers");
  }

  void ensure_epoch(const uint32_t new_epoch, const bool should_log) {
    if (epoch == new_epoch) return;
    if (new_epoch >= 2048 || !dag_sizes[new_epoch] || !cache_sizes[new_epoch]) {
      throw std::string("Bad kawpow epoch");
    }

    const uint64_t new_light_cache_words = cache_sizes[new_epoch] / sizeof(uint32_t);
    const uint64_t new_dag_words = dag_sizes[new_epoch] / sizeof(uint32_t);
    const uint64_t new_dag_nodes = dag_sizes[new_epoch] / (NODE_WORDS * sizeof(uint32_t));

    std::vector<uint32_t> host_cache;
    const uint64_t start_ms = now_ms();
    compute_light_cache(host_cache, new_epoch);

    queue.wait_and_throw();
    if (new_light_cache_words != light_cache_words) {
      free_ptr(light_cache);
      light_cache = allocate<uint32_t>(new_light_cache_words, shared_dag);
      light_cache_words = new_light_cache_words;
    }
    if (new_dag_words != dag_words) {
      free_ptr(dag);
      dag = allocate<uint32_t>(new_dag_words, shared_dag);
      dag_words = new_dag_words;
    }
    if (!light_cache || !dag) throw std::string("Can't allocate kawpow DAG buffers");

    if (shared_dag) std::memcpy(light_cache, host_cache.data(), host_cache.size() * sizeof(uint32_t));
    else sycl_wait_and_throw(queue.memcpy(light_cache, host_cache.data(), host_cache.size() * sizeof(uint32_t)), device);

    const uint32_t light_nodes = static_cast<uint32_t>(new_light_cache_words / NODE_WORDS);
    const FastModData light_mod = make_fast_mod_data(light_nodes);

    const uint32_t dag_workgroup = kawpow_dag_workgroup(device);
    const uint32_t total = static_cast<uint32_t>(new_dag_nodes);
    const uint32_t chunk_nodes = kawpow_dag_chunk_nodes();
    uint32_t* const d_light = light_cache;
    uint32_t* const d_dag = dag;
    sycl::queue& q = queue;
    auto& kb = *bundle;
    (void)kb;

    sycl::event dag_event;
    for (uint32_t start_node = 0; start_node < total;) {
      const uint32_t current_nodes = chunk_nodes ? std::min(chunk_nodes, total - start_node) : total;
      const uint32_t chunk_start = start_node;
      dag_event = q.submit([&](sycl::handler& h) {
        MOMINER_USE_BUNDLE(h, kb);
        h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(round_up(current_nodes, dag_workgroup)), sycl::range<1>(dag_workgroup)),
          [=](sycl::nd_item<1> item) {
            const uint32_t local_node = item.get_global_id(0);
            if (local_node >= current_nodes) return;
            const uint32_t node_index = chunk_start + local_node;

            uint32_t dag_node[NODE_WORDS];
            const uint32_t init = fast_mod_dev(node_index, light_mod);
#pragma unroll
            for (unsigned w = 0; w < NODE_WORDS; ++w) dag_node[w] = d_light[init * NODE_WORDS + w];
            dag_node[0] ^= node_index;
            keccak_512_words_dev(dag_node);

            for (uint32_t i = 0; i < KAWPOW_DATASET_PARENTS; ++i) {
              const uint32_t parent_index = fast_mod_dev(fnv_dev(node_index ^ i, dag_node[i & (NODE_WORDS - 1)]), light_mod);
              const uint32_t parent_base = parent_index * NODE_WORDS;
#pragma unroll
              for (unsigned w = 0; w < NODE_WORDS; ++w) {
                dag_node[w] = fnv_dev(dag_node[w], d_light[parent_base + w]);
              }
            }
            keccak_512_words_dev(dag_node);

            const uint32_t base = node_index * NODE_WORDS;
#pragma unroll
            for (unsigned w = 0; w < NODE_WORDS; ++w) d_dag[base + w] = dag_node[w];
          }
        );
      });
      if (!chunk_nodes) break;
      start_node += current_nodes;
    }
    sycl_wait_and_throw(dag_event, device);

    epoch = new_epoch;
    if (should_log) {
      char elapsed[32];
      format_duration_ms(elapsed, sizeof(elapsed), now_ms() - start_ms);
      std::fprintf(stderr, "KawPow DAG for epoch %u calculated (%s)\n", new_epoch, elapsed);
    }
  }

  void ensure_period(const uint64_t new_period) {
    if (period == new_period) return;
    program = make_program(new_period);
    period = new_period;
  }

#if !defined(MOMINER_ACPP)
  std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>>
  build_period_bundle(const uint64_t new_period, const FastModData dag_mod) {
    auto input = sycl::get_kernel_bundle<sycl::bundle_state::input>(
      queue.get_context(), {device}, {sycl::get_kernel_id<KawpowSgKernel>()}
    );
    input.set_specialization_constant<kawpow_program_id>(make_program(new_period));
    input.set_specialization_constant<kawpow_dag_mod_id>(dag_mod);
    return std::make_unique<sycl::kernel_bundle<sycl::bundle_state::executable>>(
      sycl::build(input)
    );
  }

  void ensure_period_bundle(const uint64_t new_period, const uint32_t new_epoch, const FastModData dag_mod) {
    if (period_bundle && period_bundle_period == new_period && period_bundle_epoch == new_epoch) return;

    if (prefetch_thread.joinable()) prefetch_thread.join();
    if (next_bundle && next_bundle_period == new_period && next_bundle_epoch == new_epoch) {
      period_bundle = std::move(next_bundle);
    } else {
      period_bundle = build_period_bundle(new_period, dag_mod);
    }
    period_bundle_period = new_period;
    period_bundle_epoch = new_epoch;

    next_bundle.reset();
    next_bundle_period = new_period + 1;
    next_bundle_epoch = new_epoch;
    prefetch_thread = std::thread([this, next_period = new_period + 1, dag_mod] {
      try {
        next_bundle = build_period_bundle(next_period, dag_mod);
      } catch (...) {
        next_bundle.reset();
      }
    });
  }

#endif

  static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()
    ).count();
  }
};

static KawpowState& kawpow_state(const std::string& dev_str) {
  static std::mutex states_mutex;
  static std::map<std::string, std::unique_ptr<KawpowState>> states;

  std::lock_guard<std::mutex> lock(states_mutex);
  auto& state = states[dev_str];
  if (!state) state = std::make_unique<KawpowState>(dev_str);
  return *state;
}

} // namespace mominer_kawpow

using namespace mominer_kawpow;

int kawpow(
  const unsigned, const uint32_t block_height, const uint8_t* const input, const unsigned input_size, uint8_t* const output,
  uint8_t* const mix_hash, uint64_t* const pnonce, const uint64_t target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < 40) throw std::string("Bad kawpow input length");

  const bool loop_stats = kawpow_loop_stats();
  const uint64_t t_enter = loop_stats ? kawpow_now_us() : 0;

  KawpowState& state = kawpow_state(dev_str);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  state.ensure_input(input_size);

  const uint32_t epoch = block_height / KAWPOW_EPOCH_LENGTH;
  const uint64_t period = block_height / KAWPOW_PERIOD_LENGTH;
  const uint64_t t_state = loop_stats ? kawpow_now_us() : 0;
  state.ensure_epoch(epoch, !is_benchmark);
  const uint64_t t_epoch = loop_stats ? kawpow_now_us() : 0;
  state.ensure_period(period);

  uint64_t start_nonce = 0;
  std::memcpy(&start_nonce, input + 32, sizeof(start_nonce));
  const uint32_t global_size = round_up(std::max(intensity, state.workgroup), state.workgroup);
  const uint32_t dag_elements = static_cast<uint32_t>(dag_sizes[epoch] / 256);
  const FastModData dag_mod = make_fast_mod_data(dag_elements);

  if (state.shared_io) std::memcpy(state.input, input, input_size);
  else state.queue.memcpy(state.input, input, input_size);
  std::memset(state.result, 0, sizeof(KawpowResult));

  sycl::queue& q = state.queue;
  const unsigned local_size = state.workgroup;
  uint8_t* const d_input = state.input;
  const DagLoad* const d_dag_load = reinterpret_cast<const DagLoad*>(state.dag);
  KawpowResult* const d_result = state.result;

  // GPU devices use the subgroup (warp-shuffle) exchange; the CPU SYCL device
  // (is_gpu()==false) and an explicit MOMINER_KAWPOW_EXCHANGE=slm use the
  // barrier/local-memory exchange. On AdaptiveCpp the subgroup kernel captures
  // the ProgPoW program by value instead of via SYCL-2020 specialization
  // constants / kernel bundles (which AdaptiveCpp does not provide).
  const char* const exchange_env = std::getenv("MOMINER_KAWPOW_EXCHANGE");
  const bool use_sg = state.device.is_gpu() && !(exchange_env && std::strcmp(exchange_env, "slm") == 0);

  if (use_sg) {
#if !defined(MOMINER_ACPP)
    const uint64_t t_io = loop_stats ? kawpow_now_us() : 0;
    state.ensure_period_bundle(period, epoch, dag_mod);
    const uint64_t t_bundle = loop_stats ? kawpow_now_us() : 0;
    const sycl::event search_event = q.submit([&](sycl::handler& h) {
      const auto c_dag = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(KAWPOW_CACHE_WORDS), h);
      h.use_kernel_bundle(*state.period_bundle);
      h.parallel_for<KawpowSgKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item, sycl::kernel_handler kh) MOMINER_REQD_SG_16 {
          const KawpowProgram program = kh.get_specialization_constant<kawpow_program_id>();
          const FastModData dag_mod = kh.get_specialization_constant<kawpow_dag_mod_id>();
          const uint32_t lid = item.get_local_id(0);
          const uint32_t gid = item.get_global_id(0);
          const bool active = gid < intensity;
          const uint64_t full_nonce = start_nonce + gid;

          for (uint32_t word = lid * KAWPOW_DAG_LOADS; word < KAWPOW_CACHE_WORDS;
               word += item.get_local_range(0) * KAWPOW_DAG_LOADS) {
            const DagLoad load = d_dag_load[word / KAWPOW_DAG_LOADS];
#pragma unroll
            for (unsigned i = 0; i < KAWPOW_DAG_LOADS; ++i) c_dag[word + i] = load.s[i];
          }
          item.barrier(sycl::access::fence_space::local_space);

          const auto sg = item.get_sub_group();
          const uint32_t lane_id = static_cast<uint32_t>(sg.get_local_id()[0]) & (KAWPOW_LANES - 1);

          uint32_t state2[8];
          kawpow_initial_dev(d_input, static_cast<uint32_t>(full_nonce),
                             static_cast<uint32_t>(full_nonce >> 32), state2);

          uint32_t digest[8];
          const KawpowSgExchange ex{sg,
            static_cast<uint32_t>(sg.get_local_id()[0]) & ~(KAWPOW_LANES - 1u)};
          kawpow_search_dev(ex, lane_id, state2, c_dag, d_dag_load, program, dag_mod, digest);

          uint32_t final_state[25];
          kawpow_final_dev(state2, digest, final_state);
          if ((is_test && gid == 0) || (active && target && kawpow_meets_target_words(final_state, target))) {
            store_kawpow_result(d_result, full_nonce, final_state, digest);
          }
        }
      );
    });
    if (loop_stats) {
      const uint64_t t_submit = kawpow_now_us();
      const uint64_t pre_us = t_submit - t_enter;
      if (pre_us > 50000) fprintf(stderr,
        "KAWSTALL t=%llu pre=%.1fms (state=%.1f epoch=%.1f period+io=%.1f bundle=%.1f submit=%.1f)\n",
        static_cast<unsigned long long>(time(nullptr)),
        pre_us / 1e3, (t_state - t_enter) / 1e3, (t_epoch - t_state) / 1e3,
        (t_io - t_epoch) / 1e3, (t_bundle - t_io) / 1e3, (t_submit - t_bundle) / 1e3);
    }
    sycl_wait_and_throw(search_event, state.device);
#else
    // AdaptiveCpp subgroup path: per-warp shuffles, no work-group barriers (unlike
    // the SLM exchange). The ProgPoW program + DAG modulus are captured by value
    // (small structs passed as kernel arguments); on a 32-wide NVIDIA warp two
    // independent 16-lane ProgPoW groups run via the half-warp base in the exchange.
    const sycl::event search_event = q.submit([&](sycl::handler& h) {
      const auto c_dag = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(KAWPOW_CACHE_WORDS), h);
      // JIT-specialize the per-period ProgPoW program + DAG modulus so the
      // random-math interpreter folds to constant ops (AdaptiveCpp recompiles
      // and caches per unique value), instead of running a switch per math op.
      sycl::specialized<KawpowProgram> sg_program = state.program;
      sycl::specialized<FastModData> sg_dag_mod = dag_mod;
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) MOMINER_REQD_SG_16 {
          const KawpowProgram program = sg_program;
          const FastModData dag_mod = sg_dag_mod;
          const uint32_t lid = item.get_local_id(0);
          const uint32_t gid = item.get_global_id(0);
          const bool active = gid < intensity;
          const uint64_t full_nonce = start_nonce + gid;

          for (uint32_t word = lid * KAWPOW_DAG_LOADS; word < KAWPOW_CACHE_WORDS;
               word += item.get_local_range(0) * KAWPOW_DAG_LOADS) {
            const DagLoad load = d_dag_load[word / KAWPOW_DAG_LOADS];
#pragma unroll
            for (unsigned i = 0; i < KAWPOW_DAG_LOADS; ++i) c_dag[word + i] = load.s[i];
          }
          item.barrier(sycl::access::fence_space::local_space);

          const auto sg = item.get_sub_group();
          const uint32_t lane_id = static_cast<uint32_t>(sg.get_local_id()[0]) & (KAWPOW_LANES - 1);

          uint32_t state2[8];
          kawpow_initial_dev(d_input, static_cast<uint32_t>(full_nonce),
                             static_cast<uint32_t>(full_nonce >> 32), state2);

          uint32_t digest[8];
          const KawpowSgExchange ex{sg,
            static_cast<uint32_t>(sg.get_local_id()[0]) & ~(KAWPOW_LANES - 1u)};
          kawpow_search_dev(ex, lane_id, state2, c_dag, d_dag_load, program, dag_mod, digest);

          uint32_t final_state[25];
          kawpow_final_dev(state2, digest, final_state);
          if ((is_test && gid == 0) || (active && target && kawpow_meets_target_words(final_state, target))) {
            store_kawpow_result(d_result, full_nonce, final_state, digest);
          }
        }
      );
    });
    sycl_wait_and_throw(search_event, state.device);
#endif
  } else {
    sycl_wait_and_throw(q.submit([&](sycl::handler& h) {
      const auto share = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(local_size), h);
      const auto offsets = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(local_size / KAWPOW_LANES), h);
      const auto c_dag = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(KAWPOW_CACHE_WORDS), h);
#if defined(MOMINER_ACPP)
      // AdaptiveCpp: capture the program/dag/barrier by value (no spec constants).
      const KawpowProgram slm_program = state.program;
      const FastModData slm_dag_mod = dag_mod;
      const bool slm_cpu_barrier = state.device.is_cpu();
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const KawpowProgram program = slm_program;
          const FastModData dag_mod = slm_dag_mod;
          const bool cpu_offset_barrier = slm_cpu_barrier;
#else
      h.set_specialization_constant<kawpow_program_id>(state.program);
      h.set_specialization_constant<kawpow_dag_mod_id>(dag_mod);
      h.set_specialization_constant<kawpow_cpu_offset_barrier_id>(state.device.is_cpu());
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item, sycl::kernel_handler kh) {
          const KawpowProgram program = kh.get_specialization_constant<kawpow_program_id>();
          const FastModData dag_mod = kh.get_specialization_constant<kawpow_dag_mod_id>();
          const bool cpu_offset_barrier = kh.get_specialization_constant<kawpow_cpu_offset_barrier_id>();
#endif
          const uint32_t lid = item.get_local_id(0);
          const uint32_t gid = item.get_global_id(0);
          const bool active = gid < intensity;

          const uint32_t lane_id = lid & (KAWPOW_LANES - 1);
          const uint32_t group_id = lid / KAWPOW_LANES;
          const uint64_t full_nonce = start_nonce + gid;

          for (uint32_t word = lid * KAWPOW_DAG_LOADS; word < KAWPOW_CACHE_WORDS;
               word += item.get_local_range(0) * KAWPOW_DAG_LOADS) {
            const DagLoad load = d_dag_load[word / KAWPOW_DAG_LOADS];
#pragma unroll
            for (unsigned i = 0; i < KAWPOW_DAG_LOADS; ++i) c_dag[word + i] = load.s[i];
          }
          item.barrier(sycl::access::fence_space::local_space);

          uint32_t state2[8];
          kawpow_initial_dev(d_input, static_cast<uint32_t>(full_nonce),
                             static_cast<uint32_t>(full_nonce >> 32), state2);

          uint32_t digest[8];
          const KawpowSlmExchange ex{
            item, &share[group_id * KAWPOW_LANES], &offsets[group_id], lane_id, cpu_offset_barrier
          };
          kawpow_search_dev(ex, lane_id, state2, c_dag, d_dag_load, program, dag_mod, digest);

          uint32_t final_state[25];
          kawpow_final_dev(state2, digest, final_state);
          if ((is_test && gid == 0) || (active && target && kawpow_meets_target_words(final_state, target))) {
            store_kawpow_result(d_result, full_nonce, final_state, digest);
          }
        }
      );
    }), state.device);
  }

  const uint32_t result_count = std::min(state.result->count, MAX_KAWPOW_OUTPUTS);
  for (uint32_t index = 0; index < result_count; ++index) {
    if (!is_test && !kawpow_meets_target_host(state.result->output[index], target)) continue;

    const uint64_t found_nonce = state.result->nonce[index];
    std::memcpy(output, state.result->output[index], HASH_LEN);
    std::memcpy(mix_hash, state.result->mix_hash[index], HASH_LEN);
    *pnonce = found_nonce;
    return 1;
  }

  return 0;
}
