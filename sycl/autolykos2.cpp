// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>

#include "lib-internal.h"
#include "../native/consts.h"

namespace mominer_autolykos2 {

constexpr uint32_t CONST_MES_SIZE       = 8192;
constexpr uint32_t K_LEN                = 32;
constexpr uint32_t INIT_N_LEN           = 0x04000000U;
constexpr uint32_t MAX_N_LEN            = 0x7FC9FF98U;
constexpr uint32_t INCREASE_START       = 600U * 1024U;
constexpr uint32_t INCREASE_END         = 4198400U;
constexpr uint32_t INCREASE_PERIOD      = 50U * 1024U;
constexpr uint32_t TABLE_ENTRY_BYTES    = 32;
constexpr uint32_t MAX_AUTOLYKOS_OUTPUTS = 15;

constexpr uint64_t BLAKE2B_IV[8] = {
  0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
  0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
  0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
  0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

constexpr uint8_t BLAKE2B_SIGMA[12][16] = {
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },
  {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 },
  { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },
  { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9 },
  {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11 },
  {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10 },
  { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5 },
  {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 }
};

struct AutolykosJobData {
  uint8_t message[HASH_LEN];
  uint8_t target[HASH_LEN];
};

struct AutolykosResult {
  uint32_t count;
  uint64_t nonce[MAX_AUTOLYKOS_OUTPUTS];
  uint8_t output[MAX_AUTOLYKOS_OUTPUTS][HASH_LEN];
};

inline uint32_t round_up(const uint32_t value, const uint32_t step) {
  return ((value + step - 1) / step) * step;
}

inline uint64_t bswap64_dev(const uint64_t value) {
  return ((value & 0x00000000000000FFULL) << 56) |
         ((value & 0x000000000000FF00ULL) << 40) |
         ((value & 0x0000000000FF0000ULL) << 24) |
         ((value & 0x00000000FF000000ULL) << 8) |
         ((value & 0x000000FF00000000ULL) >> 8) |
         ((value & 0x0000FF0000000000ULL) >> 24) |
         ((value & 0x00FF000000000000ULL) >> 40) |
         ((value & 0xFF00000000000000ULL) >> 56);
}

inline uint64_t rotr64_dev(const uint64_t value, const unsigned shift) {
  return (value >> shift) | (value << (64U - shift));
}

inline uint32_t load32_be_dev(const uint8_t* const input) {
  return (static_cast<uint32_t>(input[0]) << 24) |
         (static_cast<uint32_t>(input[1]) << 16) |
         (static_cast<uint32_t>(input[2]) << 8) |
          static_cast<uint32_t>(input[3]);
}

inline uint64_t load64_be_dev(const uint8_t* const input) {
  return (static_cast<uint64_t>(load32_be_dev(input)) << 32) |
          static_cast<uint64_t>(load32_be_dev(input + 4));
}

inline uint64_t load64_le_dev(const uint8_t* const input) {
  return static_cast<uint64_t>(input[0]) |
         (static_cast<uint64_t>(input[1]) << 8) |
         (static_cast<uint64_t>(input[2]) << 16) |
         (static_cast<uint64_t>(input[3]) << 24) |
         (static_cast<uint64_t>(input[4]) << 32) |
         (static_cast<uint64_t>(input[5]) << 40) |
         (static_cast<uint64_t>(input[6]) << 48) |
         (static_cast<uint64_t>(input[7]) << 56);
}

inline void store32_be_dev(uint8_t* const output, const uint32_t value) {
  output[0] = static_cast<uint8_t>(value >> 24);
  output[1] = static_cast<uint8_t>(value >> 16);
  output[2] = static_cast<uint8_t>(value >> 8);
  output[3] = static_cast<uint8_t>(value);
}

inline uint8_t word_byte_be_dev(const uint32_t word, const unsigned byte_index) {
  return static_cast<uint8_t>(word >> ((3U - byte_index) * 8U));
}

inline uint64_t pack_be32_pair_as_blake_word(const uint32_t a, const uint32_t b) {
  return (static_cast<uint64_t>((a >> 24) & 0xffU) << 0) |
         (static_cast<uint64_t>((a >> 16) & 0xffU) << 8) |
         (static_cast<uint64_t>((a >> 8) & 0xffU) << 16) |
         (static_cast<uint64_t>(a & 0xffU) << 24) |
         (static_cast<uint64_t>((b >> 24) & 0xffU) << 32) |
         (static_cast<uint64_t>((b >> 16) & 0xffU) << 40) |
         (static_cast<uint64_t>((b >> 8) & 0xffU) << 48) |
         (static_cast<uint64_t>(b & 0xffU) << 56);
}

inline void blake2b_g(
  uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d,
  const uint64_t x, const uint64_t y
) {
  a = a + b + x;
  d = rotr64_dev(d ^ a, 32);
  c += d;
  b = rotr64_dev(b ^ c, 24);
  a = a + b + y;
  d = rotr64_dev(d ^ a, 16);
  c += d;
  b = rotr64_dev(b ^ c, 63);
}

inline void blake2b_compress_dev(uint64_t h[8], const uint64_t m[16], const uint64_t t, const bool last) {
  uint64_t v[16];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) v[i] = h[i];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) v[i + 8] = BLAKE2B_IV[i];
  v[12] ^= t;
  if (last) v[14] = ~v[14];

  for (unsigned r = 0; r < 12; ++r) {
    blake2b_g(v[0], v[4], v[ 8], v[12], m[BLAKE2B_SIGMA[r][ 0]], m[BLAKE2B_SIGMA[r][ 1]]);
    blake2b_g(v[1], v[5], v[ 9], v[13], m[BLAKE2B_SIGMA[r][ 2]], m[BLAKE2B_SIGMA[r][ 3]]);
    blake2b_g(v[2], v[6], v[10], v[14], m[BLAKE2B_SIGMA[r][ 4]], m[BLAKE2B_SIGMA[r][ 5]]);
    blake2b_g(v[3], v[7], v[11], v[15], m[BLAKE2B_SIGMA[r][ 6]], m[BLAKE2B_SIGMA[r][ 7]]);
    blake2b_g(v[0], v[5], v[10], v[15], m[BLAKE2B_SIGMA[r][ 8]], m[BLAKE2B_SIGMA[r][ 9]]);
    blake2b_g(v[1], v[6], v[11], v[12], m[BLAKE2B_SIGMA[r][10]], m[BLAKE2B_SIGMA[r][11]]);
    blake2b_g(v[2], v[7], v[ 8], v[13], m[BLAKE2B_SIGMA[r][12]], m[BLAKE2B_SIGMA[r][13]]);
    blake2b_g(v[3], v[4], v[ 9], v[14], m[BLAKE2B_SIGMA[r][14]], m[BLAKE2B_SIGMA[r][15]]);
  }

#pragma unroll
  for (unsigned i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
}

inline void blake2b256_init_dev(uint64_t h[8]) {
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) h[i] = BLAKE2B_IV[i];
  h[0] ^= 0x01010020ULL;
}

inline void blake2b256_output_dev(const uint64_t h[8], uint8_t out[32]) {
#pragma unroll
  for (unsigned i = 0; i < 32; ++i) out[i] = static_cast<uint8_t>(h[i >> 3] >> ((i & 7U) * 8U));
}

inline void blake2b256_oneblock_dev(const uint8_t* const input, const uint32_t len, uint8_t out[32]) {
  uint64_t h[8];
  uint64_t m[16];
  blake2b256_init_dev(h);
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) m[i] = 0;
  for (uint32_t i = 0; i < len; ++i) {
    m[i >> 3] |= static_cast<uint64_t>(input[i]) << ((i & 7U) * 8U);
  }
  blake2b_compress_dev(h, m, len, true);
  blake2b256_output_dev(h, out);
}

inline void blake2b256_71_dev(const uint8_t input[71], uint8_t out[32]) {
  uint64_t h[8];
  uint64_t m[16];
  blake2b256_init_dev(h);
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) m[i] = 0;
#pragma unroll
  for (unsigned i = 0; i < 71; ++i) {
    m[i >> 3] |= static_cast<uint64_t>(input[i]) << ((i & 7U) * 8U);
  }
  blake2b_compress_dev(h, m, 71, true);
  blake2b256_output_dev(h, out);
}

inline void blake2b256_message_nonce_dev(const uint8_t message[32], const uint64_t nonce, uint8_t out[32]) {
  uint64_t h[8];
  uint64_t m[16];
  blake2b256_init_dev(h);
  m[0] = load64_le_dev(message);
  m[1] = load64_le_dev(message + 8);
  m[2] = load64_le_dev(message + 16);
  m[3] = load64_le_dev(message + 24);
  m[4] = bswap64_dev(nonce);
#pragma unroll
  for (unsigned i = 5; i < 16; ++i) m[i] = 0;
  blake2b_compress_dev(h, m, 40, true);
  blake2b256_output_dev(h, out);
}

inline void blake2b256_sum_dev(const uint32_t sum[8], uint8_t out[32]) {
  uint64_t h[8];
  uint64_t m[16];
  blake2b256_init_dev(h);
  m[0] = pack_be32_pair_as_blake_word(sum[0], sum[1]);
  m[1] = pack_be32_pair_as_blake_word(sum[2], sum[3]);
  m[2] = pack_be32_pair_as_blake_word(sum[4], sum[5]);
  m[3] = pack_be32_pair_as_blake_word(sum[6], sum[7]);
#pragma unroll
  for (unsigned i = 4; i < 16; ++i) m[i] = 0;
  blake2b_compress_dev(h, m, 32, true);
  blake2b256_output_dev(h, out);
}

inline void autolykos_prehash_digest_dev(const uint32_t index, const uint32_t height, uint8_t digest[32]) {
  uint64_t h[8];
  uint64_t m[16];
  blake2b256_init_dev(h);

  uint64_t ctr = 0;
  m[0] = pack_be32_pair_as_blake_word(index, height);
#pragma unroll
  for (unsigned i = 1; i < 16; ++i) m[i] = bswap64_dev(ctr++);

  uint64_t total = 0;
  for (unsigned block = 0; block < 64; ++block) {
    total += 128;
    blake2b_compress_dev(h, m, total, false);
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) m[i] = bswap64_dev(ctr++);
  }

  m[0] = bswap64_dev(CONST_MES_SIZE / sizeof(uint64_t) - 1);
#pragma unroll
  for (unsigned i = 1; i < 16; ++i) m[i] = 0;
  total += 8;
  blake2b_compress_dev(h, m, total, true);
  blake2b256_output_dev(h, digest);
}

inline void table_item_limbs_dev(
  const uint32_t* const table, const uint32_t index, const uint32_t height, uint32_t limbs[8]
) {
  if (table) {
    const uint32_t* const item = table + static_cast<uint64_t>(index) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) limbs[i] = item[i];
    return;
  }

  uint8_t digest[32];
  uint8_t item[32];
  autolykos_prehash_digest_dev(index, height, digest);
  item[0] = 0;
#pragma unroll
  for (unsigned i = 1; i < 32; ++i) item[i] = digest[i];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) limbs[i] = load32_be_dev(item + i * 4);
}

inline bool meets_target_dev(const uint8_t output[32], const uint8_t target[32]) {
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

inline uint32_t calc_n(const uint32_t height) {
  if (height < INCREASE_START) return INIT_N_LEN;
  if (height >= INCREASE_END) return MAX_N_LEN;

  uint32_t n = INIT_N_LEN;
  const uint32_t iters = (height - INCREASE_START) / INCREASE_PERIOD + 1;
  for (uint32_t i = 0; i < iters; ++i) n = n / 100U * 105U;
  return n;
}

static uint64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now().time_since_epoch()
  ).count();
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

struct AutolykosState {
  sycl::device device;
  sycl::queue queue;
  std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> bundle;
  uint32_t* table = nullptr;
  uint32_t* bhashes = nullptr;
  AutolykosResult* result = nullptr;
  uint32_t table_height = UINT32_MAX;
  uint32_t table_n = 0;
  uint32_t bhashes_cap = 0;
  unsigned workgroup;
  unsigned prehash_workgroup;
  std::mutex mutex;

  explicit AutolykosState(const std::string& dev_str)
    : device(get_dev(dev_str)),
      queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      workgroup(autolykos_workgroup(device)),
      prehash_workgroup(autolykos_prehash_workgroup(device))
  {
    if (!device.has(sycl::aspect::usm_shared_allocations)) {
      throw std::string("autolykos2 SYCL device does not support shared allocations");
    }

    set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", autolykos_compile_options(device));
    bundle = std::make_unique<sycl::kernel_bundle<sycl::bundle_state::executable>>(
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(queue.get_context())
    );
  }

  ~AutolykosState() { release(); }

  static unsigned autolykos_workgroup(const sycl::device& dev) {
    const unsigned fallback = sycl_default_workgroup(dev, {32, 64, 128, 256}, 64);
    const char* const value = std::getenv("MOMINER_AUTOLYKOS2_WORKGROUP");
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
        return static_cast<unsigned>(parsed);
    }
    return fallback;
  }

  static unsigned autolykos_prehash_workgroup(const sycl::device& dev) {
    const unsigned fallback = sycl_default_workgroup(dev, {32, 64, 128, 256}, 64);
    const char* const value = std::getenv("MOMINER_AUTOLYKOS2_PREHASH_WORKGROUP");
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
        return static_cast<unsigned>(parsed);
    }
    return fallback;
  }

  static const char* autolykos_compile_options(const sycl::device& dev) {
    const char* const value = std::getenv("MOMINER_AUTOLYKOS2_COMPILE_OPTIONS");
    if (value) return value;
    return dev.is_gpu() ? "-O3" : "";
  }

  void release() {
    queue.wait_and_throw();
    free_ptr(table);
    free_ptr(bhashes);
    free_ptr(result);
    table_height = UINT32_MAX;
    table_n = 0;
    bhashes_cap = 0;
  }

  void free_ptr(auto*& ptr) {
    if (ptr) sycl::free(ptr, queue);
    ptr = nullptr;
  }

  void ensure_result() {
    if (result) return;
    result = sycl::malloc_shared<AutolykosResult>(1, queue);
    if (!result) throw std::string("Can't allocate autolykos2 SYCL result buffer");
  }

  void ensure_bhashes(const uint32_t intensity) {
    if (bhashes && bhashes_cap >= intensity) return;
    queue.wait_and_throw();
    free_ptr(bhashes);
    bhashes_cap = intensity;
    bhashes = sycl::malloc_device<uint32_t>(static_cast<uint64_t>(intensity) * 8U, queue);
    if (!bhashes) throw std::string("Can't allocate autolykos2 intermediate buffer");
  }

  bool should_use_table(const bool is_test) const {
    if (is_test || device.is_cpu()) return false;
    const char* const value = std::getenv("MOMINER_AUTOLYKOS2_TABLE");
    return !(value && value[0] == '0');
  }

  void ensure_table(const uint32_t height, const uint32_t n_len, const bool is_test, const bool should_log) {
    if (!should_use_table(is_test)) {
      queue.wait_and_throw();
      free_ptr(table);
      table_height = UINT32_MAX;
      table_n = 0;
      return;
    }

    if (table && table_height == height && table_n == n_len) return;
    if (!device.has(sycl::aspect::usm_device_allocations)) {
      throw std::string("autolykos2 SYCL GPU device does not support device allocations");
    }

    queue.wait_and_throw();
    free_ptr(table);
    const uint64_t table_words = static_cast<uint64_t>(n_len) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
    table = sycl::malloc_device<uint32_t>(table_words, queue);
    if (!table) throw std::string("Can't allocate autolykos2 table");

    const uint64_t start_ms = now_ms();
    uint32_t* const d_table = table;
    sycl::queue& q = queue;
    auto& kb = *bundle;
    const uint32_t local = prehash_workgroup;

    sycl_wait_and_throw(q.submit([&](sycl::handler& h) {
      h.use_kernel_bundle(kb);
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(round_up(n_len, local)), sycl::range<1>(local)),
        [=](sycl::nd_item<1> item) {
          const uint32_t gid = item.get_global_id(0);
          if (gid >= n_len) return;
          uint8_t digest[32];
          autolykos_prehash_digest_dev(gid, height, digest);
          uint8_t table_item[32];
          table_item[0] = 0;
#pragma unroll
          for (unsigned i = 1; i < 32; ++i) table_item[i] = digest[i];
          uint32_t* const out = d_table + static_cast<uint64_t>(gid) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) out[i] = load32_be_dev(table_item + i * 4);
        }
      );
    }), device);

    table_height = height;
    table_n = n_len;
    if (should_log) {
      char elapsed[32];
      format_duration_ms(elapsed, sizeof(elapsed), now_ms() - start_ms);
      std::fprintf(stderr, "Autolykos2 table for height %u N %u calculated (%s)\n", height, n_len, elapsed);
    }
  }
};

static AutolykosState& autolykos_state(const std::string& dev_str) {
  static std::mutex states_mutex;
  static std::map<std::string, std::unique_ptr<AutolykosState>> states;

  std::lock_guard<std::mutex> lock(states_mutex);
  auto& state = states[dev_str];
  if (!state) state = std::make_unique<AutolykosState>(dev_str);
  return *state;
}

} // namespace mominer_autolykos2

using namespace mominer_autolykos2;

int autolykos2(
  const unsigned, const uint32_t height, const uint8_t* const input, const unsigned input_size,
  uint8_t* const output, uint64_t* const pnonce, const uint8_t* const target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < 40) throw std::string("Bad autolykos2 input length");

  AutolykosState& state = autolykos_state(dev_str);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  state.ensure_result();
  const uint32_t n_len = calc_n(height);
  state.ensure_table(height, n_len, is_test, !is_benchmark);

  AutolykosJobData job{};
  std::memcpy(job.message, input, HASH_LEN);
  std::memcpy(job.target, target, HASH_LEN);

  uint64_t start_nonce = 0;
  std::memcpy(&start_nonce, input + HASH_LEN, sizeof(start_nonce));
  const uint32_t effective_intensity = is_test ? 1 : intensity;
  const uint32_t global_size = round_up(std::max(effective_intensity, state.workgroup), state.workgroup);
  std::memset(state.result, 0, sizeof(AutolykosResult));

  sycl::queue& q = state.queue;
  auto& kb = *state.bundle;
  const uint32_t* const d_table = state.table;
  AutolykosResult* const d_result = state.result;
  const uint32_t local_size = state.workgroup;

  if (d_table && !is_test) {
    state.ensure_bhashes(effective_intensity);
    uint32_t* const d_bhashes = state.bhashes;

    q.submit([&](sycl::handler& h) {
      h.use_kernel_bundle(kb);
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const uint32_t gid = item.get_global_id(0);
          if (gid >= effective_intensity) return;

          const uint64_t nonce = start_nonce + gid;
          uint8_t nonce_be[8];
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) nonce_be[i] = static_cast<uint8_t>(nonce >> ((7U - i) * 8U));

          uint8_t h1_input[40];
#pragma unroll
          for (unsigned i = 0; i < 32; ++i) h1_input[i] = job.message[i];
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) h1_input[32 + i] = nonce_be[i];

          uint8_t h1[32];
          blake2b256_oneblock_dev(h1_input, sizeof(h1_input), h1);
          const uint32_t h3 = static_cast<uint32_t>(load64_be_dev(h1 + 24) % n_len);

          const uint32_t* const h3_item = d_table + static_cast<uint64_t>(h3) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
          uint8_t seed[71];
#pragma unroll
          for (unsigned i = 0; i < 31; ++i) {
            const unsigned pos = i + 1U;
            seed[i] = word_byte_be_dev(h3_item[pos >> 2], pos & 3U);
          }
#pragma unroll
          for (unsigned i = 0; i < 32; ++i) seed[31 + i] = job.message[i];
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) seed[63 + i] = nonce_be[i];

          uint8_t index_hash[32];
          blake2b256_oneblock_dev(seed, sizeof(seed), index_hash);

          uint32_t* const out = d_bhashes + static_cast<uint64_t>(gid) * 8U;
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) out[i] = load32_be_dev(index_hash + i * 4);
        }
      );
    });

    sycl_wait_and_throw(q.submit([&](sycl::handler& h) {
      h.use_kernel_bundle(kb);
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const uint32_t gid = item.get_global_id(0);
          if (gid >= effective_intensity) return;

          const uint32_t* const hash_words = d_bhashes + static_cast<uint64_t>(gid) * 8U;
          uint32_t words[9];
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) words[i] = hash_words[i];
          words[8] = words[0];

          uint32_t sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          for (unsigned k = 0; k < K_LEN; ++k) {
            const unsigned word_index = k >> 2;
            uint32_t idx;
            switch (k & 3U) {
              case 0: idx = words[word_index]; break;
              case 1: idx = (words[word_index] << 8) | (words[word_index + 1] >> 24); break;
              case 2: idx = (words[word_index] << 16) | (words[word_index + 1] >> 16); break;
              default: idx = (words[word_index] << 24) | (words[word_index + 1] >> 8); break;
            }
            idx %= n_len;

            const uint32_t* const table_item =
              d_table + static_cast<uint64_t>(idx) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
            uint64_t carry = 0;
            for (int i = 7; i >= 0; --i) {
              const uint64_t v = static_cast<uint64_t>(sum[i]) + table_item[i] + carry;
              sum[i] = static_cast<uint32_t>(v);
              carry = v >> 32;
            }
          }

          uint8_t final_input[32];
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) store32_be_dev(final_input + i * 4, sum[i]);

          uint8_t final_hash[32];
          blake2b256_oneblock_dev(final_input, sizeof(final_input), final_hash);
          if (meets_target_dev(final_hash, job.target)) {
            store_autolykos_result(d_result, start_nonce + gid, final_hash);
          }
        }
      );
    }), state.device);

    const uint32_t count = state.result->count;
    if (count == 0) return 0;
    const uint32_t index = std::min(count, MAX_AUTOLYKOS_OUTPUTS) - 1;
    *pnonce = state.result->nonce[index];
    std::memcpy(output, state.result->output[index], HASH_LEN);
    return 1;
  }

  sycl_wait_and_throw(q.submit([&](sycl::handler& h) {
    h.use_kernel_bundle(kb);
    sycl::local_accessor<uint32_t, 1> shared_index(sycl::range<1>(64), h);
    sycl::local_accessor<uint32_t, 1> shared_data(sycl::range<1>(512), h);
    h.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
      [=](sycl::nd_item<1> item) {
        const uint32_t gid = item.get_global_id(0);
        const uint32_t local_id = item.get_local_id(0);
        const bool active = gid < effective_intensity;

        const uint64_t nonce = start_nonce + gid;
        uint32_t indices[K_LEN];
#pragma unroll
        for (unsigned i = 0; i < K_LEN; ++i) indices[i] = 0;

        if (active) {
          uint8_t nonce_be[8];
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) nonce_be[i] = static_cast<uint8_t>(nonce >> ((7U - i) * 8U));

          uint8_t h1_input[40];
#pragma unroll
          for (unsigned i = 0; i < 32; ++i) h1_input[i] = job.message[i];
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) h1_input[32 + i] = nonce_be[i];

          uint8_t h1[32];
          blake2b256_oneblock_dev(h1_input, sizeof(h1_input), h1);
          const uint32_t h3 = static_cast<uint32_t>(load64_be_dev(h1 + 24) % n_len);

          uint8_t seed[71];
          if (d_table) {
            const uint32_t* const h3_item = d_table + static_cast<uint64_t>(h3) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
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
          for (unsigned i = 0; i < 32; ++i) seed[31 + i] = job.message[i];
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) seed[63 + i] = nonce_be[i];

          uint8_t index_hash[32];
          blake2b256_oneblock_dev(seed, sizeof(seed), index_hash);

#pragma unroll
          for (unsigned k = 0; k < K_LEN; ++k) {
            indices[k] =
              ((static_cast<uint32_t>(index_hash[(k + 0) & 31U]) << 24) |
               (static_cast<uint32_t>(index_hash[(k + 1) & 31U]) << 16) |
               (static_cast<uint32_t>(index_hash[(k + 2) & 31U]) << 8) |
                static_cast<uint32_t>(index_hash[(k + 3) & 31U])) % n_len;
          }
        }

        uint32_t sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        if (d_table && local_size == 64) {
          const uint32_t thread_id = local_id & 7U;
          const uint32_t hash_id = local_id >> 3;
          for (unsigned k = 0; k < K_LEN; ++k) {
            shared_index[local_id] = indices[k];
            item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
            for (unsigned group = 0; group < 8; ++group) {
              const uint32_t source_lane = hash_id + group * 8U;
              const uint32_t* const table_item =
                d_table + static_cast<uint64_t>(shared_index[source_lane]) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
              shared_data[(group * 64U) + (hash_id * 8U) + thread_id] = table_item[thread_id];
            }
            item.barrier(sycl::access::fence_space::local_space);

            uint64_t carry = 0;
            for (int i = 7; i >= 0; --i) {
              const uint64_t v = static_cast<uint64_t>(sum[i]) + shared_data[local_id * 8U + i] + carry;
              sum[i] = static_cast<uint32_t>(v);
              carry = v >> 32;
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
        } else {
          if (!active) return;
          for (unsigned k = 0; k < K_LEN; ++k) {
            uint32_t limbs[8];
            table_item_limbs_dev(d_table, indices[k], height, limbs);
            uint64_t carry = 0;
            for (int i = 7; i >= 0; --i) {
              const uint64_t v = static_cast<uint64_t>(sum[i]) + limbs[i] + carry;
              sum[i] = static_cast<uint32_t>(v);
              carry = v >> 32;
            }
          }
        }

        if (!active) return;

        uint8_t final_input[32];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) store32_be_dev(final_input + i * 4, sum[i]);

        uint8_t final_hash[32];
        blake2b256_oneblock_dev(final_input, sizeof(final_input), final_hash);
        if (is_test || meets_target_dev(final_hash, job.target)) {
          store_autolykos_result(d_result, nonce, final_hash);
        }
      }
    );
  }), state.device);

  const uint32_t count = state.result->count;
  if (count == 0) return 0;
  const uint32_t index = std::min(count, MAX_AUTOLYKOS_OUTPUTS) - 1;
  *pnonce = state.result->nonce[index];
  std::memcpy(output, state.result->output[index], HASH_LEN);
  return 1;
}
