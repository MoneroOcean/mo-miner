// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
// Portable SYCL implementation of Conflux's Octopus proof of work.
//
// The input is the 32-byte Keccak-256 problem hash followed by an 8-byte LE nonce,
// matching the etchash-style GPU ABI.  The consensus details intentionally mirror
// tests/reference/octopus_conflux.cpp. Portable tests use the light cache; explicit native-vector
// coverage requires the full DAG and rejects native fallback only on capability-positive devices.
//
// Performance/portability notes:
// - Matrix execution is capability-gated: Intel uses XMX/ESIMD, NVIDIA uses inline tensor-core
//   operations inside a SYCL kernel, and gfx12 AMD uses source-JIT WMMA. DPC++'s NVIDIA matrix API
//   lacks the efficient m16n8k32 shape, so that path uses inline PTX while retaining SYCL ownership
//   of memory, launches, and fallback. Unsupported devices retain the complete portable SYCL path.
// - NVIDIA fuses six field products before its four-lane DAG finish. Aligned 128-bit evict-first
//   loads fit its one-pass random DAG access pattern without retaining those pages in L1. The
//   remaining gap is not a proven hardware limit.
// - gfx12 AMD likewise fuses the six WMMA products into one normalized matrix. This reduces the two
//   pipeline slots' product storage from 192 MiB to 32 MiB and removes five intermediate writes and
//   rereads per value; on the RX 9060 XT it raised 29.08 MH/s to 31.22 MH/s versus 38.72 MH/s from
//   the local same-card reference. The remaining random-DAG gap is not a proven hardware limit.
// - B580 uses XMX for field products, then fuses Keccak and the random DAG mix in ESIMD. Four
//   independent hashes expose dependent page reads without intermediate seed/mix buffers, and
//   the field-specific modulus fold halves matrix-result traffic. Batches of 65536 groups amortize
//   submission work; two slots use 1.25 GiB, with allocation failure retaining the portable fallback.
//   The Intel 32x16 XMX tile reuses B across four row fragments; 256 GRFs prevent measured 128-GRF
//   spills. Bounded next-K input pipelining raised the matched Linux median from 30.42 to 31.27 MH/s;
//   the remaining best-peer gap is not a proven ceiling.
//   Width-8 devices use two XMX row fragments and two hashes per finish work-item. The smaller
//   finish state improved matched A770 throughput by 1.2%; its remaining gap is not a proven ceiling.

#include <sycl/sycl.hpp>
#if defined(OCTOPUS_ESIMD)
#include "../intel-dpas.h"
#if __has_include(<sycl/ext/intel/experimental/grf_size_properties.hpp>)
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#define MOM_OCTOPUS_INTEL_LARGE_GRF 1
#endif
#endif
#if defined(MOM_SYCL_HAS_HIP)
#include <hip/hip_runtime_api.h>
#include "../hiprtc-api.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../lib-internal.h"
#include "../nvidia-features.h"
#include "validation.h"
#include "../../native/consts.h"
#include "../../xmrig/3rdparty/libethash/ethash.h"

namespace mom_octopus {

static std::atomic<uint64_t> next_points_version{1};
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
static bool sycl_native_requested() {
  const char* configured = std::getenv("MOM_OCTOPUS_SYCL_NATIVE");
  return configured && *configured && std::strcmp(configured, "0") != 0;
}

static bool native_matrix_supported(sycl::queue& queue) {
#if defined(OCTOPUS_ESIMD)
  return sycl_is_level_zero_gpu(queue.get_device()) &&
         queue.get_device().has(sycl::aspect::ext_intel_matrix);
#elif defined(MOM_SYCL_HAS_CUDA)
  return mom::nvidia::has_int8_async_matrix(
      mom::nvidia::compute_capability(queue.get_device()));
#elif defined(MOM_SYCL_HAS_HIP)
  hipDevice_t device = -1;
  hipDeviceProp_t properties{};
  if (!mom_is_hip(queue.get_device()))
    return false;
  mom::hip_queue_device(queue, device);
  return hipGetDeviceProperties(&properties, device) == hipSuccess &&
         mom::amd::has_gfx12_int8_wmma(properties.gcnArchName);
#else
  (void)queue;
  return false;
#endif
}
#endif

constexpr uint32_t MOD = 1032193;
constexpr uint32_t CACHE_NODE_WORDS = 16;
constexpr uint32_t DATASET_PARENTS = 256;
constexpr uint32_t PAGE_WORDS = 64;
constexpr uint32_t POLYNOMIAL_SIZE = 1024;
constexpr uint32_t POLYNOMIAL_LANES = 32;
constexpr uint32_t POLYNOMIAL_BATCH = 16;
constexpr uint32_t COOPERATIVE_WARPS = 4;
constexpr uint32_t ACCESSES = 32;
constexpr uint32_t FNV_PRIME = 0x01000193U;
constexpr uint64_t EPOCH_LENGTH = 1ULL << 19;
constexpr unsigned MAX_RESULTS = 15;

struct Result {
  uint32_t count;
  uint64_t nonce[MAX_RESULTS];
  uint8_t hash[MAX_RESULTS][HASH_LEN];
};

inline uint32_t load32(const uint8_t* const p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline uint64_t load64(const uint8_t* const p) {
  return static_cast<uint64_t>(load32(p)) | (static_cast<uint64_t>(load32(p + 4)) << 32);
}

inline void store32(uint8_t* const p, const uint32_t x) {
  p[0] = static_cast<uint8_t>(x);
  p[1] = static_cast<uint8_t>(x >> 8);
  p[2] = static_cast<uint8_t>(x >> 16);
  p[3] = static_cast<uint8_t>(x >> 24);
}

inline uint64_t rotl64(const uint64_t x, const unsigned n) {
  return mo_rotate(x, static_cast<uint64_t>(n));
}

inline uint32_t fnv32(const uint32_t x, const uint32_t y) {
  return x * FNV_PRIME ^ y;
}
inline uint64_t fnv64(const uint64_t x, const uint64_t y) {
  return x * static_cast<uint64_t>(FNV_PRIME) ^ y;
}

// Older Windows DPC++ needs explicit NVPTX unrolling to keep Keccak state out of local memory.
// Restrict it to that device target so portable SPIR-V does not gain vendor loop-control metadata.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#define MOM_OCTOPUS_NVPTX_UNROLL _Pragma("unroll")
#else
#define MOM_OCTOPUS_NVPTX_UNROLL
#endif

inline void keccak_round(uint64_t state[25], const unsigned round) {
  static constexpr uint64_t round_constants[24] = {
      0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
      0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
      0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
      0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
      0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
      0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
  };
  static constexpr unsigned rotations[24] = {
      1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
  };
  static constexpr unsigned permutation[24] = {
      10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
  };

  uint64_t column[5];
  MOM_OCTOPUS_NVPTX_UNROLL
  for (unsigned x = 0; x < 5; ++x)
    column[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
  MOM_OCTOPUS_NVPTX_UNROLL
  for (unsigned x = 0; x < 5; ++x) {
    const uint64_t d = column[(x + 4) % 5] ^ rotl64(column[(x + 1) % 5], 1);
    MOM_OCTOPUS_NVPTX_UNROLL
    for (unsigned y = 0; y < 25; y += 5)
      state[x + y] ^= d;
  }

  uint64_t t = state[1];
  MOM_OCTOPUS_NVPTX_UNROLL
  for (unsigned i = 0; i < 24; ++i) {
    const unsigned index = permutation[i];
    const uint64_t previous = state[index];
    state[index] = rotl64(t, rotations[i]);
    t = previous;
  }
  MOM_OCTOPUS_NVPTX_UNROLL
  for (unsigned y = 0; y < 25; y += 5) {
    MOM_OCTOPUS_NVPTX_UNROLL
    for (unsigned x = 0; x < 5; ++x)
      column[x] = state[y + x];
    MOM_OCTOPUS_NVPTX_UNROLL
    for (unsigned x = 0; x < 5; ++x)
      state[y + x] = column[x] ^ ((~column[(x + 1) % 5]) & column[(x + 2) % 5]);
  }
  state[0] ^= round_constants[round];
}

#undef MOM_OCTOPUS_NVPTX_UNROLL

inline void keccak(uint64_t state[25]) {
  for (unsigned round = 0; round < 24; ++round)
    keccak_round(state, round);
}

inline void keccak512_words(uint32_t words[CACHE_NODE_WORDS]) {
  uint64_t state[25]{};
  for (unsigned i = 0; i < 8; ++i)
    state[i] =
        static_cast<uint64_t>(words[i * 2]) | (static_cast<uint64_t>(words[i * 2 + 1]) << 32);
  state[8] = 0x8000000000000001ULL; // Keccak padding for a 64-byte input.
  keccak(state);
  for (unsigned i = 0; i < 8; ++i) {
    words[i * 2] = static_cast<uint32_t>(state[i]);
    words[i * 2 + 1] = static_cast<uint32_t>(state[i] >> 32);
  }
}

inline void keccak512_header_seed(const uint8_t header[HASH_LEN], const uint64_t value,
                                  uint32_t out[CACHE_NODE_WORDS]) {
  uint64_t state[25]{};
  for (unsigned i = 0; i < 4; ++i)
    state[i] = load64(header + i * 8);
  state[4] = value;
  state[5] = 1; // Keccak padding starts immediately after the 40-byte input.
  state[8] = 0x8000000000000000ULL;
  keccak(state);
  for (unsigned i = 0; i < 8; ++i) {
    out[i * 2] = static_cast<uint32_t>(state[i]);
    out[i * 2 + 1] = static_cast<uint32_t>(state[i] >> 32);
  }
}

inline void keccak256_seed_mix(const uint32_t seed[CACHE_NODE_WORDS], const uint32_t mix[8],
                               uint8_t out[HASH_LEN]) {
  uint64_t state[25]{};
  for (unsigned i = 0; i < 8; ++i)
    state[i] = static_cast<uint64_t>(seed[i * 2]) | (static_cast<uint64_t>(seed[i * 2 + 1]) << 32);
  for (unsigned i = 0; i < 4; ++i)
    state[8 + i] =
        static_cast<uint64_t>(mix[i * 2]) | (static_cast<uint64_t>(mix[i * 2 + 1]) << 32);
  state[12] = 1; // Keccak padding after the 96-byte input.
  state[16] = 0x8000000000000000ULL;
  keccak(state);
  for (unsigned i = 0; i < 4; ++i) {
    store32(out + i * 8, static_cast<uint32_t>(state[i]));
    store32(out + i * 8 + 4, static_cast<uint32_t>(state[i] >> 32));
  }
}

struct SipHash {
  uint64_t v0, v1, v2, v3;

  void round() {
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
  }

  void hash24(const uint64_t nonce) {
    v3 ^= nonce;
    round();
    round();
    v0 ^= nonce;
    v2 ^= 0xFF;
    round();
    round();
    round();
    round();
  }

  uint64_t value() const {
    return v0 ^ v1 ^ v2 ^ v3;
  }
};

inline uint32_t gcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    const uint32_t remainder = a % b;
    a = b;
    b = remainder;
  }
  return a;
}

inline uint32_t pow_mod(uint32_t base, uint32_t exponent) {
  uint32_t result = 1;
  while (exponent != 0) {
    if (exponent & 1U)
      result = static_cast<uint32_t>((static_cast<uint64_t>(result) * base) % MOD);
    base = static_cast<uint32_t>((static_cast<uint64_t>(base) * base) % MOD);
    exponent >>= 1;
  }
  return result;
}

inline uint32_t remap(const uint64_t value) {
  uint32_t exponent = static_cast<uint32_t>(value % (MOD - 2)) + 1;
  uint32_t divisor;
  while ((divisor = gcd(exponent, MOD - 1)) != 1)
    exponent /= divisor;
  return pow_mod(11, exponent);
}

inline uint32_t mod_field(const uint64_t value) {
  // All inputs are below 2*MOD^2+MOD < 2^41.  floor(2^41/MOD) gives a
  // quotient at most one low, replacing 64-bit division with one multiply.
  constexpr uint32_t RECIPROCAL = 2130438;
  const uint32_t product_high =
#if defined(MOM_SYCL_ADAPTIVECPP) || defined(MOM_SYCL_PORTABLE_OPENCL)
      // Some OpenCL CPU JITs miscompile the SPIR-V mul_hi builtin. The explicit widening form has
      // identical integer semantics and leaves native DPC++ GPU artifacts on their measured path.
      static_cast<uint32_t>((static_cast<uint64_t>(static_cast<uint32_t>(value)) * RECIPROCAL) >>
                            32);
#else
      sycl::mul_hi(static_cast<uint32_t>(value), RECIPROCAL);
#endif
  const uint32_t quotient = (product_high + static_cast<uint32_t>(value >> 32) * RECIPROCAL) >> 9;
  const uint32_t remainder = static_cast<uint32_t>(value - static_cast<uint64_t>(quotient) * MOD);
  return remainder >= MOD ? remainder - MOD : remainder;
}

inline uint32_t mad_field(const uint32_t a, const uint32_t b, const uint32_t c) {
  if constexpr (mom_sycl_portable_opencl) {
    // OpenCL does not require enough precision from the floating approximation below to keep its
    // quotient within one correction. Reuse the exact reciprocal reduction in the compatibility
    // artifact; native GPU builds retain their measured FMA path.
    return mod_field(static_cast<uint64_t>(a) * b + c);
  }
  constexpr float INVERSE = 1.0f / MOD;
  const uint32_t quotient = static_cast<uint32_t>(
      sycl::fma(static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)) * INVERSE);
  int32_t remainder = static_cast<int32_t>(a * b + c - quotient * MOD);
  if (remainder < 0)
    remainder += MOD;
  if (remainder >= static_cast<int32_t>(MOD))
    remainder -= MOD;
  return static_cast<uint32_t>(remainder);
}

inline bool meets_target(const uint8_t hash[HASH_LEN], const uint8_t* const target) {
  for (unsigned i = 0; i < HASH_LEN; ++i) {
    if (hash[i] != target[i])
      return hash[i] < target[i];
  }
  return true;
}

inline void save_result(Result* const result, const uint64_t nonce, const uint8_t hash[HASH_LEN]) {
  using Atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>;
  const uint32_t index = Atomic(result->count).fetch_add(1);
  if (index >= MAX_RESULTS)
    return;
  result->nonce[index] = nonce;
  for (unsigned i = 0; i < HASH_LEN; ++i)
    result->hash[index][i] = hash[i];
}

inline void dataset_item(const uint32_t* const cache, const FastModData cache_mod,
                         const uint32_t node_index, uint32_t item[CACHE_NODE_WORDS]) {
  const uint32_t cache_index = fast_mod_dev(node_index, cache_mod);
  for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
    item[word] = cache[cache_index * CACHE_NODE_WORDS + word];
  item[0] ^= node_index;
  keccak512_words(item);
  for (uint32_t i = 0; i < DATASET_PARENTS; ++i) {
    const uint32_t parent = fast_mod_dev(fnv32(node_index ^ i, item[i & 15]), cache_mod);
    for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
      item[word] = fnv32(item[word], cache[parent * CACHE_NODE_WORDS + word]);
  }
  keccak512_words(item);
}

inline void make_polynomial_coefficients(const uint8_t* const header, const uint64_t nonce,
                                         const uint32_t lane, const uint32_t offset,
                                         sycl::local_accessor<uint32_t, 1> coefficients) {
  const uint64_t v0 = load64(header), v1 = load64(header + 8);
  const uint64_t v2 = load64(header + 16), v3 = load64(header + 24);
  SipHash sip{v0, v1, v2, v3};
  sip.hash24((nonce / POLYNOMIAL_LANES) * POLYNOMIAL_LANES + lane);
  for (uint32_t row = 0; row < POLYNOMIAL_SIZE / POLYNOMIAL_LANES; ++row) {
    sip.round();
    coefficients[offset + row * POLYNOMIAL_LANES + lane] = static_cast<uint32_t>(sip.value()) % MOD;
  }
}

inline void make_polynomial_lane(const uint64_t nonce, const uint32_t lane,
                                 sycl::local_accessor<uint32_t, 1> coefficients,
                                 sycl::local_accessor<uint32_t, 1> polynomial,
                                 sycl::local_accessor<uint32_t, 1> parameters) {
  const uint32_t a = parameters[0];
  const uint32_t b = parameters[1];
  const uint32_t c = parameters[2];
  const uint32_t w = parameters[3];
  const uint32_t w2 = parameters[4];
  const uint32_t remainder = static_cast<uint32_t>(nonce % POLYNOMIAL_LANES);
  const uint32_t exponent = remainder + lane * POLYNOMIAL_LANES;
  const uint32_t wpow = pow_mod(w, exponent);
  const uint32_t w2pow = pow_mod(w2, exponent);
  const uint32_t x =
      mod_field(static_cast<uint64_t>(a) * w2pow + static_cast<uint64_t>(b) * wpow + c);
  uint32_t value = 0;
  for (uint32_t i = POLYNOMIAL_SIZE; i-- != 0;)
    value = mad_field(value, x, coefficients[i]);
  polynomial[lane] = value;
}

template <bool FULL_DAG>
inline void octopus_hash_scalar(
    sycl::nd_item<1> item, const uint8_t* const header, const uint64_t nonce,
    const uint32_t* const cache, const FastModData cache_mod, const uint32_t* const dag,
    const FastModData page_mod, const bool active, const bool is_test, const uint8_t* const target,
    Result* const result, sycl::local_accessor<uint32_t, 1> coefficients,
    sycl::local_accessor<uint32_t, 1> polynomial, sycl::local_accessor<uint32_t, 1> parameters,
    sycl::local_accessor<uint32_t, 1> seed, sycl::local_accessor<uint32_t, 1> mix,
    sycl::local_accessor<uint32_t, 1> dataset_words, sycl::local_accessor<uint32_t, 1> page,
    sycl::local_accessor<uint32_t, 1> compressed) {
  const uint32_t lane = static_cast<uint32_t>(item.get_local_id(0));

  make_polynomial_coefficients(header, nonce, lane, 0, coefficients);
  sycl::group_barrier(item.get_group());

  if (lane == 0) {
    const uint64_t v0 = load64(header), v1 = load64(header + 8);
    const uint64_t v2 = load64(header + 16), v3 = load64(header + 24);
    const uint32_t a = remap(v0), b = remap(v1);
    uint64_t c_input = v2;
    uint32_t c;
    do {
      c = remap(c_input++);
    } while ((static_cast<uint64_t>(b) * b) % MOD == (static_cast<uint64_t>(4) * a * c) % MOD);
    const uint32_t w = remap(v3);
    parameters[0] = a;
    parameters[1] = b;
    parameters[2] = c;
    parameters[3] = w;
    parameters[4] = mod_field(static_cast<uint64_t>(w) * w);
  }
  sycl::group_barrier(item.get_group());

  make_polynomial_lane(nonce, lane, coefficients, polynomial, parameters);
  sycl::group_barrier(item.get_group());
  if (lane == 0) {
    uint64_t polynomial_hash = 0;
    for (uint32_t i = 0; i < POLYNOMIAL_LANES; ++i)
      polynomial_hash = fnv64(polynomial_hash, polynomial[i]);
    uint32_t local_seed[CACHE_NODE_WORDS];
    keccak512_header_seed(header, polynomial_hash, local_seed);
    for (unsigned i = 0; i < CACHE_NODE_WORDS; ++i)
      seed[i] = local_seed[i];
  }
  sycl::group_barrier(item.get_group());

  for (unsigned word = lane; word < PAGE_WORDS; word += POLYNOMIAL_LANES)
    mix[word] = seed[word & (CACHE_NODE_WORDS - 1)];
  sycl::group_barrier(item.get_group());

  for (uint32_t access = 0; access < ACCESSES; ++access) {
    if (lane == 0) {
      page[0] = fast_mod_dev(fnv32(seed[0] ^ access ^ polynomial[access], mix[access]), page_mod);
    }
    sycl::group_barrier(item.get_group());

    if constexpr (!FULL_DAG) {
      if (lane == 0) {
        for (unsigned node = 0; node < 4; ++node) {
          uint32_t item_words[CACHE_NODE_WORDS];
          dataset_item(cache, cache_mod, page[0] * 4 + node, item_words);
          for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
            dataset_words[node * CACHE_NODE_WORDS + word] = item_words[word];
        }
      }
      sycl::group_barrier(item.get_group());
    }

    for (unsigned word = lane; word < PAGE_WORDS; word += POLYNOMIAL_LANES) {
      const size_t base =
          (static_cast<size_t>(page[0]) * 4 + word / CACHE_NODE_WORDS) * CACHE_NODE_WORDS;
      const uint32_t value = FULL_DAG ? dag[base + word % CACHE_NODE_WORDS] : dataset_words[word];
      mix[word] = fnv32(mix[word], value);
    }
    // The next page uses mix[access + 1], so all 64 words must be visible before it is selected.
    sycl::group_barrier(item.get_group());
  }

  if (lane < 8) {
    const unsigned left_offset = lane * 4;
    const unsigned right_offset = (8 + lane) * 4;
    uint32_t left = mix[left_offset], right = mix[right_offset];
    for (unsigned j = 1; j < 4; ++j) {
      left = fnv32(left, mix[left_offset + j]);
      right = fnv32(right, mix[right_offset + j]);
    }
    compressed[lane] = fnv32(left, right);
  }
  sycl::group_barrier(item.get_group());

  if (lane == 0 && active) {
    uint32_t local_seed[CACHE_NODE_WORDS];
    uint32_t local_compressed[8];
    for (unsigned i = 0; i < CACHE_NODE_WORDS; ++i)
      local_seed[i] = seed[i];
    for (unsigned i = 0; i < 8; ++i)
      local_compressed[i] = compressed[i];
    uint8_t output[HASH_LEN];
    keccak256_seed_mix(local_seed, local_compressed, output);
    if (is_test || meets_target(output, target))
      save_result(result, nonce, output);
  }
}

template <bool FULL_DAG>
inline void
octopus_hash_batched(sycl::nd_item<1> item, const uint8_t* const header, const uint64_t start,
                     const uint32_t* const cache, const FastModData cache_mod,
                     const uint32_t* const dag, const FastModData page_mod, const uint32_t count,
                     const uint8_t* const target, Result* const result, const bool is_test,
                     sycl::local_accessor<uint32_t, 1> coefficients, const uint32_t* const points) {
  constexpr uint32_t SUBGROUP = POLYNOMIAL_LANES;
  constexpr uint32_t SUBGROUPS = 4;
  constexpr uint32_t WORKGROUP = SUBGROUP * SUBGROUPS;
  const uint32_t local = static_cast<uint32_t>(item.get_local_id(0));
  const uint32_t subgroup = local / SUBGROUP;
  const uint32_t lane = local % SUBGROUP;
  const uint64_t last_nonce = start + count - 1;
  const uint64_t aligned_start = start & ~(static_cast<uint64_t>(SUBGROUP) - 1);
  const uint64_t group_start = aligned_start + static_cast<uint64_t>(item.get_group(0)) * WORKGROUP;
  const uint64_t block_start = group_start + static_cast<uint64_t>(subgroup) * SUBGROUP;
  const uint64_t nonce = block_start + lane;
  const bool block_active = block_start <= last_nonce && block_start + SUBGROUP - 1 >= start;
  const bool active = nonce >= start && nonce <= last_nonce;

  // A subgroup owns one aligned 32-nonce block.  Coefficients are generated once
  // per active block; every active lane then owns exactly one nonce and evaluates
  // all 32 points for it.  Thus count remains a nonce count, not 32x work items;
  // only the aligned padding lanes in the first/last block are inactive.
  if (block_active)
    make_polynomial_coefficients(header, block_start, lane, subgroup * POLYNOMIAL_SIZE,
                                 coefficients);
  sycl::group_barrier(item.get_group());

  if (!active)
    return;

  uint32_t polynomial[POLYNOMIAL_LANES];
  const uint32_t coefficient_offset = subgroup * POLYNOMIAL_SIZE;
  const uint32_t point_offset = lane;
  // Sixteen accumulators keep adjacent point evaluations together without hiding
  // the exact Horner order behind a vendor-specific vector intrinsic.
  for (uint32_t first = 0; first < POLYNOMIAL_LANES; first += POLYNOMIAL_BATCH) {
    uint32_t value[POLYNOMIAL_BATCH] = {};
    uint32_t x[POLYNOMIAL_BATCH];
    for (unsigned packed = 0; packed < POLYNOMIAL_BATCH; ++packed)
      x[packed] = points[point_offset + (first + packed) * SUBGROUP];
    for (uint32_t coefficient = POLYNOMIAL_SIZE; coefficient-- != 0;) {
      const uint32_t term = coefficients[coefficient_offset + coefficient];
      for (unsigned packed = 0; packed < POLYNOMIAL_BATCH; ++packed)
        value[packed] = mad_field(value[packed], x[packed], term);
    }
    for (unsigned packed = 0; packed < POLYNOMIAL_BATCH; ++packed)
      polynomial[first + packed] = value[packed];
  }

  uint64_t polynomial_hash = 0;
  for (uint32_t i = 0; i < POLYNOMIAL_LANES; ++i)
    polynomial_hash = fnv64(polynomial_hash, polynomial[i]);
  uint32_t seed[CACHE_NODE_WORDS];
  keccak512_header_seed(header, polynomial_hash, seed);
  uint32_t mix[PAGE_WORDS];
  for (unsigned word = 0; word < PAGE_WORDS; ++word)
    mix[word] = seed[word & (CACHE_NODE_WORDS - 1)];

  for (uint32_t access = 0; access < ACCESSES; ++access) {
    const uint32_t page =
        fast_mod_dev(fnv32(seed[0] ^ access ^ polynomial[access], mix[access]), page_mod);
    for (unsigned node = 0; node < 4; ++node) {
      uint32_t item_words[CACHE_NODE_WORDS];
      if constexpr (FULL_DAG) {
        const size_t base = (static_cast<size_t>(page) * 4 + node) * CACHE_NODE_WORDS;
        for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
          item_words[word] = dag[base + word];
      } else {
        dataset_item(cache, cache_mod, page * 4 + node, item_words);
      }
      for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
        mix[node * CACHE_NODE_WORDS + word] =
            fnv32(mix[node * CACHE_NODE_WORDS + word], item_words[word]);
    }
  }

  uint32_t compressed[8];
  for (unsigned lane8 = 0; lane8 < 8; ++lane8) {
    const unsigned left_offset = lane8 * 4;
    const unsigned right_offset = (8 + lane8) * 4;
    uint32_t left = mix[left_offset], right = mix[right_offset];
    for (unsigned j = 1; j < 4; ++j) {
      left = fnv32(left, mix[left_offset + j]);
      right = fnv32(right, mix[right_offset + j]);
    }
    compressed[lane8] = fnv32(left, right);
  }
  uint8_t output[HASH_LEN];
  keccak256_seed_mix(seed, compressed, output);
  if (is_test ? nonce == start : meets_target(output, target))
    save_result(result, nonce, output);
}

inline uint32_t fnv_reduce4(const uint32_t value[4]) {
  return fnv32(fnv32(fnv32(value[0], value[1]), value[2]), value[3]);
}

// Mirrors open-cfxmine's CUDA layout: each native warp evaluates 32 nonces, then
// its two 16-lane teams chase four hashes at once.  Every DAG page is consequently
// read as sixteen contiguous uint4 values instead of 64 scalar private loads.
inline void octopus_hash_cooperative(sycl::nd_item<1> item, const uint8_t* const header,
                                     const uint64_t start, const uint32_t* const dag,
                                     const FastModData page_mod, const uint32_t count,
                                     const uint8_t* const target, Result* const result,
                                     const bool is_test,
                                     sycl::local_accessor<uint32_t, 1> coefficients,
                                     const uint32_t* const points) {
  constexpr uint32_t WARP = 32;
  const uint32_t local = static_cast<uint32_t>(item.get_local_id(0));
  const uint32_t warp = local / WARP;
  const uint32_t lane = local % WARP;
  const uint64_t aligned_start = start & ~(static_cast<uint64_t>(WARP) - 1);
  const uint64_t group_start =
      aligned_start + static_cast<uint64_t>(item.get_group(0)) * item.get_local_range(0);
  const uint64_t block_start = group_start + static_cast<uint64_t>(warp) * WARP;
  const uint64_t nonce = block_start + lane;
  const uint64_t last_nonce = start + count - 1;
  const bool active = nonce >= start && nonce <= last_nonce;
  const uint32_t offset = warp * POLYNOMIAL_SIZE;

  make_polynomial_coefficients(header, block_start, lane, offset, coefficients);
  sycl::group_barrier(item.get_group());
  uint32_t polynomial[POLYNOMIAL_LANES];
  uint64_t polynomial_hash = 0;
  for (uint32_t first = 0; first < POLYNOMIAL_LANES; first += POLYNOMIAL_BATCH) {
    uint32_t value[POLYNOMIAL_BATCH]{};
    uint32_t x[POLYNOMIAL_BATCH];
    for (unsigned packed = 0; packed < POLYNOMIAL_BATCH; ++packed)
      x[packed] = points[lane + (first + packed) * WARP];
    for (uint32_t coefficient = POLYNOMIAL_SIZE; coefficient-- != 0;) {
      const uint32_t term = coefficients[offset + coefficient];
      for (unsigned packed = 0; packed < POLYNOMIAL_BATCH; ++packed)
        value[packed] = mad_field(value[packed], x[packed], term);
    }
    for (unsigned packed = 0; packed < POLYNOMIAL_BATCH; ++packed) {
      polynomial[first + packed] = value[packed];
      polynomial_hash = fnv64(polynomial_hash, value[packed]);
    }
  }
  uint32_t seed[CACHE_NODE_WORDS];
  keccak512_header_seed(header, polynomial_hash, seed);
  sycl::group_barrier(item.get_group());
  for (uint32_t access = 0; access < ACCESSES; ++access)
    coefficients[offset + access * WARP + lane] = polynomial[access];
  sycl::group_barrier(item.get_group());

  const sycl::sub_group sg = item.get_sub_group();
  const uint32_t team_base = lane & ~15U;
  const uint32_t team_lane = lane & 15U;
  const uint32_t mix_part = team_lane & 3U;
  uint32_t final_mix[8]{};
  for (uint32_t first = 0; first < 16; first += 4) {
    uint32_t mix[4][4], initial[4];
    for (uint32_t parallel = 0; parallel < 4; ++parallel) {
      const uint32_t source = team_base + first + parallel;
      initial[parallel] = mom_select_from_group(sg, seed[0], source);
      for (uint32_t part = 0; part < 4; ++part)
        for (uint32_t word = 0; word < 4; ++word) {
          const uint32_t selected = mom_select_from_group(sg, seed[part * 4 + word], source);
          if (mix_part == part)
            mix[parallel][word] = selected;
        }
    }
    for (uint32_t access = 0; access < ACCESSES; access += 4) {
      const uint32_t address_lane = team_base + access / 4;
      for (uint32_t step = 0; step < 4; ++step) {
        for (uint32_t parallel = 0; parallel < 4; ++parallel) {
          const uint32_t hash_lane = team_base + first + parallel;
          const uint32_t polynomial_value =
              coefficients[offset + (access + step) * WARP + hash_lane];
          uint32_t page = fast_mod_dev(
              fnv32(initial[parallel] ^ (access + step) ^ polynomial_value, mix[parallel][step]),
              page_mod);
          page = mom_select_from_group(sg, page, address_lane);
          const size_t dag_offset = static_cast<size_t>(page) * PAGE_WORDS + team_lane * 4;
          for (uint32_t word = 0; word < 4; ++word)
            mix[parallel][word] = fnv32(mix[parallel][word], dag[dag_offset + word]);
        }
      }
    }
    for (uint32_t parallel = 0; parallel < 4; ++parallel) {
      const uint32_t reduced = fnv_reduce4(mix[parallel]);
      const uint32_t hash_lane = team_base + first + parallel;
      for (uint32_t word = 0; word < 8; ++word) {
        const uint32_t compressed = fnv32(mom_select_from_group(sg, reduced, team_base + word),
                                          mom_select_from_group(sg, reduced, team_base + 8 + word));
        if (lane == hash_lane)
          final_mix[word] = compressed;
      }
    }
  }
  uint8_t output[HASH_LEN];
  keccak256_seed_mix(seed, final_mix, output);
  if (active && (is_test ? nonce == start : meets_target(output, target)))
    save_result(result, nonce, output);
}

class OctopusDagKernel;
template <bool FULL_DAG> class OctopusScalarSearchKernel;
template <bool FULL_DAG> class OctopusBatchedSearchKernel;
class OctopusCooperativeSearchKernel;

static sycl::event build_dag(sycl::queue& queue, const uint32_t* const cache,
                             const uint32_t cache_nodes,
                             uint32_t* const dag, const uint32_t start, const uint32_t count) {
  constexpr unsigned WORKGROUP = 128;
  const FastModData cache_mod = make_fast_mod_data(cache_nodes);
  const size_t global = (static_cast<size_t>(count) + WORKGROUP - 1) / WORKGROUP * WORKGROUP;
  return queue.submit([&](sycl::handler& handler) {
    handler.parallel_for<OctopusDagKernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WORKGROUP)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const uint32_t local = static_cast<uint32_t>(item.get_global_id(0));
          if (local >= count)
            return;
          const uint32_t index = start + local;
          uint32_t node[CACHE_NODE_WORDS];
          const uint32_t cache_index = fast_mod_dev(index, cache_mod);
          for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
            node[word] = cache[cache_index * CACHE_NODE_WORDS + word];
          node[0] ^= index;
          keccak512_words(node);
          for (uint32_t i = 0; i < DATASET_PARENTS; ++i) {
            const uint32_t parent = fast_mod_dev(fnv32(index ^ i, node[i & 15]), cache_mod);
            for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
              node[word] = fnv32(node[word], cache[parent * CACHE_NODE_WORDS + word]);
          }
          keccak512_words(node);
          for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
            dag[static_cast<size_t>(index) * CACHE_NODE_WORDS + word] = node[word];
        });
  });
}

template <bool FULL_DAG>
static sycl::event
search_scalar(sycl::queue& queue, const uint8_t* const header, const uint32_t* const cache,
              const FastModData cache_mod, const uint32_t* const dag, const FastModData page_mod,
              const uint64_t start, const uint32_t count, const uint8_t* const target,
              Result* const result, const bool is_test) {
  const size_t local = POLYNOMIAL_LANES;
  const size_t global = static_cast<size_t>(count) * local;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<uint32_t, 1> coefficients(sycl::range<1>(POLYNOMIAL_SIZE), handler);
    sycl::local_accessor<uint32_t, 1> polynomial(sycl::range<1>(POLYNOMIAL_LANES), handler);
    sycl::local_accessor<uint32_t, 1> parameters(sycl::range<1>(5), handler);
    sycl::local_accessor<uint32_t, 1> seed(sycl::range<1>(CACHE_NODE_WORDS), handler);
    sycl::local_accessor<uint32_t, 1> mix(sycl::range<1>(PAGE_WORDS), handler);
    sycl::local_accessor<uint32_t, 1> dataset_words(sycl::range<1>(PAGE_WORDS), handler);
    sycl::local_accessor<uint32_t, 1> page(sycl::range<1>(1), handler);
    sycl::local_accessor<uint32_t, 1> compressed(sycl::range<1>(8), handler);
    handler.parallel_for<OctopusScalarSearchKernel<FULL_DAG>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const uint32_t index = static_cast<uint32_t>(item.get_group(0));
          octopus_hash_scalar<FULL_DAG>(
              item, header, start + index, cache, cache_mod, dag, page_mod,
              index < count && (!is_test || index == 0), is_test, target, result, coefficients,
              polynomial, parameters, seed, mix, dataset_words, page, compressed);
        });
  });
}

template <bool FULL_DAG>
static sycl::event
search_batched(sycl::queue& queue, const uint8_t* const header, const uint32_t* const cache,
               const FastModData cache_mod, const uint32_t* const dag, const FastModData page_mod,
               const uint32_t* const points, const uint64_t start, const uint32_t count,
               const uint8_t* const target, Result* const result, const bool is_test) {
  constexpr size_t local = POLYNOMIAL_LANES * 4;
  const uint64_t aligned_start = start & ~(static_cast<uint64_t>(POLYNOMIAL_LANES) - 1);
  const uint64_t covered = (start - aligned_start) + count;
  const size_t groups = static_cast<size_t>((covered + local - 1) / local);
  const size_t global = groups * local;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<uint32_t, 1> coefficients(sycl::range<1>(POLYNOMIAL_SIZE * 4), handler);
    handler.parallel_for<OctopusBatchedSearchKernel<FULL_DAG>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          octopus_hash_batched<FULL_DAG>(item, header, start, cache, cache_mod, dag, page_mod,
                                         count, target, result, is_test, coefficients, points);
        });
  });
}

#if !defined(MOM_SYCL_PORTABLE_OPENCL)
static sycl::event search_cooperative(sycl::queue& queue, const uint8_t* const header,
                                      const uint32_t* const dag,
                                      const FastModData page_mod, const uint32_t* const points,
                                      const uint64_t start, const uint32_t count,
                                      const uint8_t* const target, Result* const result,
                                      const bool is_test) {
  constexpr size_t local = POLYNOMIAL_LANES * COOPERATIVE_WARPS;
  const uint64_t aligned_start = start & ~(static_cast<uint64_t>(POLYNOMIAL_LANES) - 1);
  const uint64_t covered = (start - aligned_start) + count;
  const size_t global = static_cast<size_t>((covered + local - 1) / local) * local;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<uint32_t, 1> coefficients(
        sycl::range<1>(POLYNOMIAL_SIZE * COOPERATIVE_WARPS), handler);
    handler.parallel_for<OctopusCooperativeSearchKernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] MOM_SYCL_KERNEL_ARGS_RESTRICT {
          octopus_hash_cooperative(item, header, start, dag, page_mod, count, target, result,
                                   is_test, coefficients, points);
        });
  });
}
#endif

#if defined(MOM_OCTOPUS_HAS_SYCL_NATIVE)
#if defined(MOM_SYCL_HAS_HIP)
#include "amd_wmma.inc"
#endif
#include "nvidia_tensor.inc"
#endif

inline bool is_prime(const uint64_t value) {
  if (value < 2 || (value % 2) == 0)
    return value == 2;
  for (uint64_t divisor = 3; divisor <= value / divisor; divisor += 2)
    if (value % divisor == 0)
      return false;
  return true;
}

inline uint64_t cache_size(const uint32_t epoch) {
  uint64_t size = (1ULL << 24) + (1ULL << 16) * epoch - 64;
  while (!is_prime(size / 64))
    size -= 128;
  return size;
}

inline uint64_t dataset_size(const uint32_t epoch) {
  uint64_t size = (1ULL << 32) + (1ULL << 24) * epoch - 256;
  while (!is_prime(size / 256))
    size -= 512;
  return size;
}

class State {
public:
  sycl::device device;
  sycl::queue queue;
  const bool shared_io;
  uint8_t* header = nullptr;
  uint8_t* target = nullptr;
  uint32_t* cache = nullptr;
  uint32_t* dag = nullptr;
  uint32_t* points = nullptr;
  Result* result = nullptr;
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
  OctopusSyclNativeSearch sycl_native;
#endif
  uint32_t epoch = std::numeric_limits<uint32_t>::max();
  uint32_t cache_nodes = 0;
  uint32_t dag_nodes = 0;
  std::array<uint8_t, HASH_LEN> points_header{};
  std::array<uint8_t, HASH_LEN> target_copy{};
  uint64_t points_version = 0;
  bool points_ready = false;
  bool target_ready = false;
  bool full_dag = false;
  bool dag_attempted = false;
  std::mutex mutex;

  explicit State(const std::string& dev)
      : device(get_dev(dev)), queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
        shared_io(device.is_cpu() || !mom_has_usm_device(device)) {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device)))
      throw std::string("octopus requires SYCL device and shared USM");
    try {
      header = allocate<uint8_t>(HASH_LEN);
      target = allocate<uint8_t>(HASH_LEN);
      points = allocate<uint32_t>(POLYNOMIAL_SIZE);
      result = sycl::malloc_shared<Result>(1, queue);
      if (!header || !target || !points || !result)
        throw std::string("Can't allocate octopus I/O buffers");
    } catch (...) {
      sycl_cleanup_noexcept("octopus init", [&] {
        release();
      });
      throw;
    }
  }

  ~State() {
    sycl_cleanup_noexcept("octopus", [&] {
      release();
    });
  }

  template <typename T> T* allocate(const size_t count) {
    return shared_io ? sycl::malloc_shared<T>(count, queue) : sycl::malloc_device<T>(count, queue);
  }

  template <typename T> void free_ptr(T*& pointer) noexcept {
    if (pointer)
      try {
        sycl::free(pointer, queue);
      } catch (...) {
      }
    pointer = nullptr;
  }

  void release() noexcept {
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
    sycl_native.release();
#endif
    try {
      queue.wait_and_throw();
    } catch (...) {
    }
    free_ptr(dag);
    free_ptr(cache);
    free_ptr(points);
    free_ptr(target);
    free_ptr(header);
    free_ptr(result);
  }

  bool ensure_points(const uint8_t next_header[HASH_LEN]) {
    if (points_ready && std::memcmp(points_header.data(), next_header, HASH_LEN) == 0)
      return false;
    const uint64_t v0 = load64(next_header), v1 = load64(next_header + 8);
    const uint64_t v2 = load64(next_header + 16), v3 = load64(next_header + 24);
    const uint32_t a = remap(v0), b = remap(v1);
    uint64_t c_input = v2;
    uint32_t c;
    do {
      c = remap(c_input++);
    } while ((static_cast<uint64_t>(b) * b) % MOD == (static_cast<uint64_t>(4) * a * c) % MOD);
    const uint32_t w = remap(v3), w2 = mod_field(static_cast<uint64_t>(w) * w);
    uint32_t wpow = 1, w2pow = 1;
    std::array<uint32_t, POLYNOMIAL_SIZE> host_points;
    for (uint32_t exponent = 0; exponent < POLYNOMIAL_SIZE; ++exponent) {
      host_points[exponent] =
          mod_field(static_cast<uint64_t>(a) * w2pow + static_cast<uint64_t>(b) * wpow + c);
      wpow = mod_field(static_cast<uint64_t>(wpow) * w);
      w2pow = mod_field(static_cast<uint64_t>(w2pow) * w2);
    }
    // A failed copy must not leave the previous header marked ready.
    points_ready = false;
    if (shared_io)
      std::memcpy(points, host_points.data(), sizeof(host_points));
    else
      sycl_wait_and_throw(queue.memcpy(points, host_points.data(), sizeof(host_points)), device);
    std::memcpy(points_header.data(), next_header, HASH_LEN);
    points_ready = true;
    points_version = next_points_version.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  bool wants_full_dag() const {
    if (device.is_cpu() || shared_io || std::getenv("MOM_OCTOPUS_LIGHT"))
      return false;
    const uint64_t bytes = dataset_size(epoch);
    const uint64_t reserve = 512ULL << 20;
    const uint64_t memory = device.get_info<sycl::info::device::global_mem_size>();
    const uint64_t max_alloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
    return max_alloc >= bytes && memory >= bytes + cache_size(epoch) + reserve;
  }

  void ensure_cache(const uint32_t next_epoch) {
    const uint64_t bytes = cache_size(next_epoch);
    const uint32_t nodes = static_cast<uint32_t>(bytes / (CACHE_NODE_WORDS * sizeof(uint32_t)));
    if (cache && cache_nodes == nodes)
      return;
    std::vector<uint32_t> host_cache(static_cast<size_t>(bytes / sizeof(uint32_t)));
    const ethash_h256_t seed = ethash_get_seedhash(next_epoch);
    if (!ethash_compute_cache_nodes(host_cache.data(), bytes, &seed))
      throw std::string("Can't calculate octopus light cache");
    queue.wait_and_throw();
    free_ptr(cache);
    cache = allocate<uint32_t>(host_cache.size());
    if (!cache)
      throw std::string("Can't allocate octopus light cache");
    if (shared_io)
      std::memcpy(cache, host_cache.data(), bytes);
    else
      sycl_wait_and_throw(queue.memcpy(cache, host_cache.data(), bytes), device);
    cache_nodes = nodes;
  }

  void ensure_dag(const bool log) {
    if (full_dag || dag_attempted)
      return;
    const bool available = wants_full_dag();
    dag_attempted = true;
    if (!available)
      return;
    const uint64_t bytes = dataset_size(epoch);
    const uint32_t nodes = static_cast<uint32_t>(bytes / (CACHE_NODE_WORDS * sizeof(uint32_t)));
    uint32_t* candidate = nullptr;
    try {
      candidate = allocate<uint32_t>(static_cast<size_t>(bytes / sizeof(uint32_t)));
    } catch (...) {
      candidate = nullptr;
    }
    if (!candidate)
      return; // The light path remains correct on small or fragmented devices.

    const uint64_t start = now_ms();
    const uint32_t chunk =
        1U << 18; // Keep individual dispatches below Windows WDDM watchdog limits.
    try {
      for (uint32_t offset = 0; offset < nodes; offset += chunk) {
        const uint32_t count = std::min(chunk, nodes - offset);
        sycl_wait_and_throw(build_dag(queue, cache, cache_nodes, candidate, offset, count), device);
      }
    } catch (...) {
      sycl_cleanup_noexcept("octopus DAG wait", [&] { queue.wait_and_throw(); });
      free_ptr(candidate);
      // A submitted DAG fault must not become cached light-path availability.
      dag_attempted = false;
      throw;
    }
    free_ptr(dag);
    dag = candidate;
    dag_nodes = nodes;
    full_dag = true;
    if (log)
      std::fprintf(stderr, "Octopus DAG epoch %u built (%.2f s)\n", epoch,
                   static_cast<double>(now_ms() - start) / 1000.0);
  }

  void ensure_epoch(const uint32_t next_epoch, const bool log, const bool test) {
    if (epoch != next_epoch) {
      // Invalidate the old epoch before an in-place rebuild can fail; it must not remain reusable.
      epoch = std::numeric_limits<uint32_t>::max();
      free_ptr(dag);
      free_ptr(cache);
      full_dag = false;
      dag_attempted = false;
      dag_nodes = 0;
      cache_nodes = 0;
      ensure_cache(next_epoch);
      epoch = next_epoch;
    }
    if (!test || std::getenv("MOM_OCTOPUS_TEST_FULL_DAG"))
      ensure_dag(log);
  }
};

static DeviceStateRegistry<State>& registry() {
  static auto* const value = new DeviceStateRegistry<State>;
  return *value;
}

static State& state_for(const std::string& dev) {
  return registry().get(dev, [&] { return std::make_unique<State>(dev); });
}

void octopus_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "octopus: ordered SYCL cleanup failed\n");
  }
}

} // namespace mom_octopus

using namespace mom_octopus;

int octopus(unsigned, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
            uint8_t* mix_hash, uint64_t* nonce, const uint8_t* target, const uint8_t*,
            unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str) {
  if (!input || !output || !nonce || input_size < HASH_LEN + sizeof(uint64_t) || !intensity)
    throw std::string("octopus requires a 32-byte problem hash, nonce, and nonzero intensity");
  if (!is_test && !target)
    throw std::string("octopus requires a target outside test mode");
  const uint64_t start_nonce = load64(input + HASH_LEN);
  if (static_cast<uint64_t>(intensity) - 1 > std::numeric_limits<uint64_t>::max() - start_nonce)
    throw std::string("octopus nonce range exceeds uint64 max");
  State& state = state_for(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_epoch(height / EPOCH_LENGTH, !is_benchmark, is_test);
  const bool header_changed = state.ensure_points(input);
  const bool target_changed =
      target && (!state.target_ready || std::memcmp(state.target_copy.data(), target, HASH_LEN));
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
  sycl::event input_copy_event;
  bool has_input_copy_event = false;
#endif
  try {
    MomSyclHostTransferGuard host_transfers(state.queue, "octopus host transfers");
    if (state.shared_io) {
      if (header_changed)
        std::memcpy(state.header, input, HASH_LEN);
      if (target_changed)
        std::memcpy(state.target, target, HASH_LEN);
    } else {
      if (header_changed) {
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
        input_copy_event = state.queue.memcpy(state.header, input, HASH_LEN);
        has_input_copy_event = true;
#else
        state.queue.memcpy(state.header, input, HASH_LEN);
#endif
      }
      if (target_changed) {
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
        input_copy_event = state.queue.memcpy(state.target, target, HASH_LEN);
        has_input_copy_event = true;
#else
        state.queue.memcpy(state.target, target, HASH_LEN);
#endif
      }
    }
    if (target_changed) {
      std::memcpy(state.target_copy.data(), target, HASH_LEN);
      state.target_ready = true;
    }
    std::memset(state.result, 0, sizeof(*state.result));
    const FastModData cache_mod = make_fast_mod_data(state.cache_nodes);
    const FastModData page_mod = make_fast_mod_data(dataset_size(state.epoch) / 256);
    // MOM_OCTOPUS_SCALAR keeps the original one-nonce/full-DAG kernel available
    // as a correctness oracle while the default path uses the cooperative kernel.
    const bool scalar = std::getenv("MOM_OCTOPUS_SCALAR") != nullptr;
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
    const auto subgroup_sizes = state.device.get_info<sycl::info::device::sub_group_sizes>();
    // Older HIP GPUs can use wave64; their fallback must not launch a wave32-only kernel.
    const bool subgroup_32 = mom_is_cuda(state.device) ||
        std::find(subgroup_sizes.begin(), subgroup_sizes.end(), 32u) != subgroup_sizes.end();
#endif
    bool searched = false;
    const bool prove_native = is_test && std::getenv("MOM_OCTOPUS_TEST_NATIVE");
    bool native_supported = false;
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
    // Qualify hardware before a failed native launch can invalidate queue/interop queries. JIT
    // availability is deliberately not part of this predicate: supported-device failures must fail.
    if (prove_native)
      native_supported = native_matrix_supported(state.queue);
    const bool sycl_native_device =
#ifdef OCTOPUS_ESIMD
        sycl_is_level_zero_gpu(state.device) && state.device.has(sycl::aspect::ext_intel_matrix);
#else
        mom_is_cuda(state.device) || mom_is_hip(state.device);
#endif
    if (sycl_native_requested() && state.full_dag && sycl_native_device && !scalar) {
      if (has_input_copy_event)
        sycl_wait_and_throw(input_copy_event, state.device);
      searched = state.sycl_native.search(state.queue, state.header, state.dag, page_mod,
                                          state.points, state.points_version, start_nonce, intensity,
                                          state.target, state.result, is_test);
      if (!searched)
        std::memset(state.result, 0, sizeof(*state.result));
    }
#endif
    const auto validation_failure = mom::octopus::test_path_failure(
        is_test && std::getenv("MOM_OCTOPUS_TEST_FULL_DAG"), state.full_dag,
        prove_native, native_supported, searched);
    if (validation_failure == mom::octopus::TestPathFailure::full_dag)
      throw std::string("Octopus full-DAG validation could not run");
    if (validation_failure == mom::octopus::TestPathFailure::native)
      throw std::string("Octopus SYCL-native full-DAG validation could not run");
    if (!searched && state.full_dag) {
      if (scalar) {
        sycl_wait_and_throw(search_scalar<true>(state.queue, state.header, state.cache, cache_mod,
                                                state.dag, page_mod, start_nonce, intensity,
                                                state.target, state.result, is_test),
                            state.device);
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
      } else if (!std::getenv("MOM_OCTOPUS_BATCHED") && subgroup_32) {
        sycl_wait_and_throw(search_cooperative(state.queue, state.header, state.dag, page_mod,
                                               state.points, start_nonce, intensity, state.target,
                                               state.result, is_test),
                            state.device);
#endif
      } else {
        sycl_wait_and_throw(search_batched<true>(state.queue, state.header, state.cache, cache_mod,
                                                 state.dag, page_mod, state.points, start_nonce,
                                                 intensity, state.target, state.result, is_test),
                            state.device);
      }
    } else if (!searched) {
      if (scalar) {
        sycl_wait_and_throw(search_scalar<false>(state.queue, state.header, state.cache, cache_mod,
                                                 nullptr, page_mod, start_nonce, intensity,
                                                 state.target, state.result, is_test),
                            state.device);
      } else {
        sycl_wait_and_throw(search_batched<false>(state.queue, state.header, state.cache, cache_mod,
                                                  nullptr, page_mod, state.points, start_nonce,
                                                  intensity, state.target, state.result, is_test),
                            state.device);
      }
    }
    if (!state.result->count)
      return 0;
    const uint32_t index = std::min(state.result->count, MAX_RESULTS) - 1;
    *nonce = state.result->nonce[index];
    std::memcpy(output, state.result->hash[index], HASH_LEN);
    if (mix_hash)
      std::memset(mix_hash, 0, HASH_LEN);
    return 1;
  } catch (...) {
    // A failed upload must not make a same-header retry skip its header or target copy.
    state.points_ready = state.target_ready = false;
    throw;
  }
}
