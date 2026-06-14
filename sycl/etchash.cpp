// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "lib-internal.h"
#include "../native/consts.h"
#include "../xmrig/3rdparty/libethash/ethash.h"
#include "../xmrig/3rdparty/libethash/data_sizes.h"

namespace mominer_etchash {

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

inline void keccak_512_words_dev(uint32_t words[ETHASH_NODE_WORDS]) {
  Uint2 st[25]{};
  for (unsigned i = 0; i < 8; ++i) st[i] = make_u2(words[i * 2], words[i * 2 + 1]);
  st[8] = make_u2(0x00000001, 0x80000000);
  keccakf1600_pair_dev(st);
  for (unsigned i = 0; i < 8; ++i) {
    words[i * 2] = st[i].x;
    words[i * 2 + 1] = st[i].y;
  }
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

inline void keccak_512_header_nonce_pair_dev(const uint8_t* const header_hash, const uint64_t nonce, uint32_t out[16]) {
  Uint2 st[25]{};
  for (unsigned i = 0; i < 4; ++i) {
    st[i] = make_u2(load32_le_dev(header_hash + i * 8), load32_le_dev(header_hash + i * 8 + 4));
  }
  st[4] = make_u2(static_cast<uint32_t>(nonce), static_cast<uint32_t>(nonce >> 32));
  st[5] = make_u2(0x00000001, 0x00000000);
  st[8] = make_u2(0x00000000, 0x80000000);
  keccakf1600_pair_dev(st);
  for (unsigned i = 0; i < 8; ++i) {
    out[i * 2] = st[i].x;
    out[i * 2 + 1] = st[i].y;
  }
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

inline void keccak_256_seed_mix_pair_dev(const uint32_t seed[16], const uint32_t mix[8], uint8_t out[32]) {
  Uint2 st[25]{};
  for (unsigned i = 0; i < 8; ++i) st[i] = make_u2(seed[i * 2], seed[i * 2 + 1]);
  for (unsigned i = 0; i < 4; ++i) st[8 + i] = make_u2(mix[i * 2], mix[i * 2 + 1]);
  st[12] = make_u2(0x00000001, 0x00000000);
  st[16] = make_u2(0x00000000, 0x80000000);
  keccakf1600_pair_dev(st);
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

class EtchashState {
public:
  sycl::device device;
  sycl::queue queue;
  std::unique_ptr<MOMINER_BUNDLE_T> bundle;
  bool shared_io;
  bool shared_dag;
  uint8_t* input = nullptr;
  uint8_t* target = nullptr;
  uint32_t* light_cache = nullptr;
  uint32_t* dag = nullptr;
  EtchashResult* result = nullptr;
  uint32_t input_cap = 0;
  uint64_t light_cache_words = 0;
  uint64_t dag_words = 0;
  uint32_t epoch = UINT32_MAX;
  uint32_t seed_epoch = UINT32_MAX;
  std::mutex mutex;

  explicit EtchashState(const std::string& dev_str)
    : device(get_dev(dev_str)),
      queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !device.has(sycl::aspect::usm_device_allocations)),
      shared_dag(device.is_cpu() || !device.has(sycl::aspect::usm_device_allocations))
  {
    if (!device.has(sycl::aspect::usm_shared_allocations) ||
        (!device.is_cpu() && !device.has(sycl::aspect::usm_device_allocations))) {
      throw std::string("etchash SYCL device does not support required allocations");
    }

    set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", etchash_compile_options(device));
    bundle = std::make_unique<MOMINER_BUNDLE_T>(
      MOMINER_GET_EXEC_BUNDLE(queue.get_context())
    );
  }

  ~EtchashState() { release(); }

  static unsigned etchash_dag_workgroup(const sycl::device& dev) {
    const unsigned fallback = sycl_default_workgroup(dev, {32, 64, 128, 256, 512}, dev.is_cpu() ? 128 : 64);
    const char* const value = std::getenv("MOMINER_ETCHASH_DAG_WORKGROUP");
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

  static uint32_t etchash_dag_chunk_nodes() {
    const char* const value = std::getenv("MOMINER_ETCHASH_DAG_CHUNK_NODES");
    if (!value || !*value) return 0;

    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno || end == value || *end || parsed > std::numeric_limits<uint32_t>::max()) return 0;
    return static_cast<uint32_t>(parsed);
  }

  static const char* etchash_compile_options(const sycl::device& dev) {
    const char* const value = std::getenv("MOMINER_ETCHASH_COMPILE_OPTIONS");
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
    free_ptr(target);
    free_ptr(light_cache);
    free_ptr(dag);
    free_ptr(result);
    input_cap = 0;
    light_cache_words = 0;
    dag_words = 0;
    epoch = UINT32_MAX;
    seed_epoch = UINT32_MAX;
  }

  void ensure_input(const unsigned input_size) {
    if (input_size <= input_cap && result && target) return;
    queue.wait_and_throw();
    free_ptr(input);
    free_ptr(target);
    free_ptr(result);
    input_cap = input_size;
    input = allocate<uint8_t>(input_cap, shared_io);
    target = allocate<uint8_t>(HASH_LEN, shared_io);
    result = sycl::malloc_shared<EtchashResult>(1, queue);
    if (!input || !target || !result) throw std::string("Can't allocate etchash SYCL input buffers");
  }

  void ensure_epoch(const uint32_t new_epoch, const uint32_t new_seed_epoch, const bool should_log) {
    if (epoch == new_epoch && seed_epoch == new_seed_epoch) return;
    if (new_epoch >= 2048 || !dag_sizes[new_epoch] || !cache_sizes[new_epoch]) {
      throw std::string("Bad etchash epoch");
    }

    const uint64_t new_light_cache_words = cache_sizes[new_epoch] / sizeof(uint32_t);
    const uint64_t new_dag_words = dag_sizes[new_epoch] / sizeof(uint32_t);
    const uint64_t new_dag_nodes = dag_sizes[new_epoch] / (ETHASH_NODE_WORDS * sizeof(uint32_t));

    std::vector<uint32_t> host_cache;
    const uint64_t start_ms = now_ms();
    compute_light_cache(host_cache, new_epoch, new_seed_epoch);

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
    if (!light_cache || !dag) throw std::string("Can't allocate etchash DAG buffers");

    if (shared_dag) std::memcpy(light_cache, host_cache.data(), host_cache.size() * sizeof(uint32_t));
    else sycl_wait_and_throw(queue.memcpy(light_cache, host_cache.data(), host_cache.size() * sizeof(uint32_t)), device);

    const uint32_t light_nodes = static_cast<uint32_t>(new_light_cache_words / ETHASH_NODE_WORDS);
    const FastModData light_mod = make_fast_mod_data(light_nodes);
    const uint32_t dag_workgroup = etchash_dag_workgroup(device);
    const uint32_t total = static_cast<uint32_t>(new_dag_nodes);
    const uint32_t chunk_nodes = etchash_dag_chunk_nodes();
    uint32_t* const d_light = light_cache;
    uint32_t* const d_dag = dag;
    sycl::queue& q = queue;
    auto& kb = *bundle;

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

            uint32_t dag_node[ETHASH_NODE_WORDS];
            const uint32_t init = fast_mod_dev(node_index, light_mod);
#pragma unroll
            for (unsigned w = 0; w < ETHASH_NODE_WORDS; ++w) dag_node[w] = d_light[init * ETHASH_NODE_WORDS + w];
            dag_node[0] ^= node_index;
            keccak_512_words_dev(dag_node);

            for (uint32_t i = 0; i < ETHASH_DATASET_PARENTS2; ++i) {
              const uint32_t parent_index = fast_mod_dev(fnv_dev(node_index ^ i, dag_node[i & (ETHASH_NODE_WORDS - 1)]), light_mod);
              const uint32_t parent_base = parent_index * ETHASH_NODE_WORDS;
#pragma unroll
              for (unsigned w = 0; w < ETHASH_NODE_WORDS; ++w) {
                dag_node[w] = fnv_dev(dag_node[w], d_light[parent_base + w]);
              }
            }
            keccak_512_words_dev(dag_node);

            const uint32_t base = node_index * ETHASH_NODE_WORDS;
#pragma unroll
            for (unsigned w = 0; w < ETHASH_NODE_WORDS; ++w) d_dag[base + w] = dag_node[w];
          }
        );
      });
      if (!chunk_nodes) break;
      start_node += current_nodes;
    }
    sycl_wait_and_throw(dag_event, device);

    epoch = new_epoch;
    seed_epoch = new_seed_epoch;
    if (should_log) {
      char elapsed[32];
      format_duration_ms(elapsed, sizeof(elapsed), now_ms() - start_ms);
      std::fprintf(stderr, "Etchash DAG for epoch %u seed epoch %u calculated (%s)\n", new_epoch, new_seed_epoch, elapsed);
    }
  }

  static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()
    ).count();
  }
};

static EtchashState& etchash_state(const std::string& dev_str) {
  static std::mutex states_mutex;
  static std::map<std::string, std::unique_ptr<EtchashState>> states;

  std::lock_guard<std::mutex> lock(states_mutex);
  auto& state = states[dev_str];
  if (!state) state = std::make_unique<EtchashState>(dev_str);
  return *state;
}

} // namespace mominer_etchash

using namespace mominer_etchash;

int etchash(
  const unsigned, const uint32_t block_height, const uint8_t* const input, const unsigned input_size, uint8_t* const output,
  uint8_t* const mix_hash, uint64_t* const pnonce, const uint8_t* const target, const uint8_t* const seed_hash,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < 40) throw std::string("Bad etchash input length");

  EtchashState& state = etchash_state(dev_str);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  state.ensure_input(input_size);
  uint32_t epoch = 0;
  uint32_t seed_epoch = 0;
  resolve_epochs(block_height, seed_hash, epoch, seed_epoch);
  state.ensure_epoch(epoch, seed_epoch, !is_benchmark);

  uint64_t start_nonce = 0;
  std::memcpy(&start_nonce, input + 32, sizeof(start_nonce));
  const uint32_t dag_pages = static_cast<uint32_t>(dag_sizes[epoch] / ETHASH_MIX_BYTES);
  const FastModData dag_mod = make_fast_mod_data(dag_pages);

  if (state.shared_io) {
    std::memcpy(state.input, input, input_size);
    std::memcpy(state.target, target, HASH_LEN);
  } else {
    state.queue.memcpy(state.input, input, input_size);
    state.queue.memcpy(state.target, target, HASH_LEN);
  }
  std::memset(state.result, 0, sizeof(EtchashResult));

  sycl::queue& q = state.queue;
  auto& kb = *state.bundle;
  const uint8_t* const d_input = state.input;
  const uint8_t* const d_target = state.target;
  const uint32_t* const d_dag = state.dag;
  EtchashResult* const d_result = state.result;
  constexpr unsigned ETCHASH_LANES = ETHASH_MIX_WORDS;
  constexpr unsigned SEED_OFFSET = 0;
  constexpr unsigned PAGE_OFFSET = SEED_OFFSET + 16;
  constexpr unsigned MIX_OFFSET = PAGE_OFFSET + 2;
  constexpr unsigned CMIX_OFFSET = MIX_OFFSET + ETHASH_MIX_WORDS;
  constexpr unsigned SCRATCH_WORDS = CMIX_OFFSET + 8;

  if (state.device.is_gpu()) {
    constexpr unsigned GROUP4_WORKGROUP = 128;
    constexpr unsigned GROUP4_SEED_WORDS = GROUP4_WORKGROUP * 16;

    sycl_wait_and_throw(q.submit([&](sycl::handler& h) {
      MOMINER_USE_BUNDLE(h, kb);
      sycl::local_accessor<uint32_t, 1> seeds(sycl::range<1>(GROUP4_SEED_WORDS), h);
      h.parallel_for(
        sycl::nd_range<1>(
          sycl::range<1>(round_up(intensity, GROUP4_WORKGROUP)),
          sycl::range<1>(GROUP4_WORKGROUP)
        ),
        [=](sycl::nd_item<1> item) MOMINER_REQD_SG_16 {
          const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));
          const uint32_t group_hash_base = static_cast<uint32_t>(item.get_group(0)) * GROUP4_WORKGROUP;
          const uint32_t own_hash = group_hash_base + lid;

          if (own_hash < intensity) {
            uint32_t seed[16];
            keccak_512_header_nonce_pair_dev(d_input, start_nonce + own_hash, seed);
#pragma unroll
            for (unsigned w = 0; w < 16; ++w) seeds[lid * 16 + w] = seed[w];
          }
          item.barrier(sycl::access::fence_space::local_space);

          const auto sg = item.get_sub_group();
          const uint32_t sg_lane = static_cast<uint32_t>(sg.get_local_id()[0]);
          const uint32_t lane4 = lid & 3U;
          const uint32_t group4_base = lid & ~3U;
          const uint32_t sg_group4_base = sg_lane & ~3U;

          for (unsigned owner = 0; owner < 4; ++owner) {
            const uint32_t owner_lid = group4_base + owner;
            const uint32_t hash_index = group_hash_base + owner_lid;
            if (hash_index >= intensity) continue;

            const uint32_t seed_base = owner_lid * 16;
            const uint32_t seed0 = seeds[seed_base];
            uint32_t mix[8];
#pragma unroll
            for (unsigned w = 0; w < 8; ++w) mix[w] = seeds[seed_base + ((lane4 * 8 + w) & 15U)];

            for (unsigned i = 0; i < ETHASH_ACCESSES; ++i) {
              const uint32_t selected_lane = (i & (ETHASH_MIX_WORDS - 1)) >> 3;
              const uint32_t selected_word = i & 7U;
              const uint32_t page_candidate = lane4 == selected_lane
                ? fast_mod_dev(fnv_dev(seed0 ^ i, mix[selected_word]), dag_mod)
                : 0;
              const uint32_t page = sycl::select_from_group(sg, page_candidate, sg_group4_base + selected_lane);
              const uint32_t base = page * ETHASH_MIX_WORDS + lane4 * 8;
              const auto dag_words = *reinterpret_cast<const sycl::vec<uint64_t, 4>*>(d_dag + base);
#pragma unroll
              for (unsigned w = 0; w < 4; ++w) {
                const uint64_t dag_pair = dag_words[w];
                mix[w * 2] = fnv_dev(mix[w * 2], static_cast<uint32_t>(dag_pair));
                mix[w * 2 + 1] = fnv_dev(mix[w * 2 + 1], static_cast<uint32_t>(dag_pair >> 32));
              }
            }

            uint32_t compressed0 = mix[0];
            compressed0 = fnv_dev(compressed0, mix[1]);
            compressed0 = fnv_dev(compressed0, mix[2]);
            compressed0 = fnv_dev(compressed0, mix[3]);
            uint32_t compressed1 = mix[4];
            compressed1 = fnv_dev(compressed1, mix[5]);
            compressed1 = fnv_dev(compressed1, mix[6]);
            compressed1 = fnv_dev(compressed1, mix[7]);

            uint32_t compressed_mix[8];
#pragma unroll
            for (unsigned lane = 0; lane < 4; ++lane) {
              compressed_mix[lane * 2] = sycl::select_from_group(sg, compressed0, sg_group4_base + lane);
              compressed_mix[lane * 2 + 1] = sycl::select_from_group(sg, compressed1, sg_group4_base + lane);
            }

            if (lane4 == owner) {
              uint32_t seed[16];
#pragma unroll
              for (unsigned w = 0; w < 16; ++w) seed[w] = seeds[seed_base + w];

              uint8_t final_hash[32];
              keccak_256_seed_mix_pair_dev(seed, compressed_mix, final_hash);
              if ((is_test && hash_index == 0) || meets_target_dev(final_hash, d_target)) {
                store_etchash_result(d_result, start_nonce + hash_index, final_hash, compressed_mix);
              }
            }
          }
        }
      );
    }), state.device);
  } else {
    sycl_wait_and_throw(q.submit([&](sycl::handler& h) {
      MOMINER_USE_BUNDLE(h, kb);
      sycl::local_accessor<uint32_t, 1> scratch(sycl::range<1>(SCRATCH_WORDS), h);
      h.parallel_for(
        sycl::nd_range<1>(
          sycl::range<1>(static_cast<size_t>(intensity) * ETCHASH_LANES),
          sycl::range<1>(ETCHASH_LANES)
        ),
        [=](sycl::nd_item<1> item) {
          const uint32_t gid = static_cast<uint32_t>(item.get_group(0));
          const uint32_t lane = static_cast<uint32_t>(item.get_local_id(0));

          const uint64_t nonce = start_nonce + gid;
          if (lane == 0) {
            uint32_t seed[16];
            keccak_512_header_nonce_dev(d_input, nonce, seed);
#pragma unroll
            for (unsigned w = 0; w < 16; ++w) scratch[SEED_OFFSET + w] = seed[w];
          }
          item.barrier(sycl::access::fence_space::local_space);

          const uint32_t seed0 = scratch[SEED_OFFSET];
          uint32_t mix_word = scratch[SEED_OFFSET + (lane & (ETHASH_NODE_WORDS - 1))];

          for (unsigned i = 0; i < ETHASH_ACCESSES; ++i) {
            if (lane == (i & (ETHASH_MIX_WORDS - 1))) {
              scratch[PAGE_OFFSET + (i & 1)] = fast_mod_dev(fnv_dev(seed0 ^ i, mix_word), dag_mod);
            }
            item.barrier(sycl::access::fence_space::local_space);
            const uint32_t page = scratch[PAGE_OFFSET + (i & 1)];
            const uint32_t base = page * ETHASH_MIX_WORDS;
            mix_word = fnv_dev(mix_word, d_dag[base + lane]);
          }

          scratch[MIX_OFFSET + lane] = mix_word;
          item.barrier(sycl::access::fence_space::local_space);

          if (lane < 8) {
            const uint32_t w = lane * 4;
            uint32_t reduction = scratch[MIX_OFFSET + w + 0];
            reduction = fnv_dev(reduction, scratch[MIX_OFFSET + w + 1]);
            reduction = fnv_dev(reduction, scratch[MIX_OFFSET + w + 2]);
            reduction = fnv_dev(reduction, scratch[MIX_OFFSET + w + 3]);
            scratch[CMIX_OFFSET + lane] = reduction;
          }
          item.barrier(sycl::access::fence_space::local_space);

          if (lane != 0) return;

          uint32_t seed[16];
          uint32_t compressed_mix[8];
#pragma unroll
          for (unsigned w = 0; w < 16; ++w) seed[w] = scratch[SEED_OFFSET + w];
#pragma unroll
          for (unsigned w = 0; w < 8; ++w) compressed_mix[w] = scratch[CMIX_OFFSET + w];

          uint8_t final_hash[32];
          keccak_256_seed_mix_dev(seed, compressed_mix, final_hash);
          if ((is_test && gid == 0) || meets_target_dev(final_hash, d_target)) {
            store_etchash_result(d_result, nonce, final_hash, compressed_mix);
          }
        }
      );
    }), state.device);
  }

  const uint32_t count = state.result->count;
  if (count == 0) return 0;
  const uint32_t index = std::min(count, MAX_ETCHASH_OUTPUTS) - 1;
  *pnonce = state.result->nonce[index];
  std::memcpy(output, state.result->output[index], HASH_LEN);
  std::memcpy(mix_hash, state.result->mix_hash[index], HASH_LEN);
  return 1;
}
