// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
// Verthash GPU search, ported from the GPLv2+ reference miner (github.com/CryptoGraphics/VerthashMiner).
//
// Performance/portability notes:
// - Each nonce performs 4096 dependent dataset rounds through four cooperating lanes. This makes
//   random-memory latency and cross-lane mapping more important than arithmetic throughput.
// - B580 throughput reaches its practical batch knee near 524288 nonces. Its production kernel is
//   spill-free SIMD32/128-GRF; forced SIMD16 spills, and tested workgroup/unroll changes were neutral.
// - A one-work-item split layout was materially slower, so the four-lane portable mapping is retained.
//   These measurements localize the current code-generation gap but do not prove a hardware ceiling.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <vector>

#include "../lib-internal.h"
#include "../../native/consts.h"
#include "data_file.hpp"

namespace mom_verthash {

struct Uint2 {
  uint32_t x, y;
};
struct Result {
  uint32_t count, nonce, hash[8];
};
constexpr size_t DATA_BYTES = data_file::expected_bytes;
static_assert(sizeof(Uint2) == 8 && DATA_BYTES % sizeof(Uint2) == 0);
constexpr uint32_t MDIV = 80216063;
// The bounded fixture is deliberately separate from the official dataset. It is selected only for
// an explicit CPU test call, while production, benchmark, and GPU test calls keep MDIV and the
// verified 1.20-GiB file below.
constexpr uint32_t TEST_MDIV = 4093;
constexpr size_t TEST_DATA_BYTES = (static_cast<size_t>(TEST_MDIV) * 2 + 2) * sizeof(Uint2);
constexpr size_t TEST_DATA_ELEMENTS = TEST_DATA_BYTES / sizeof(Uint2);
constexpr unsigned WG = 64;

inline uint32_t test_data_word(uint64_t index, uint64_t seed) {
  uint64_t value = index + seed;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return static_cast<uint32_t>(value ^ (value >> 32));
}

inline Uint2 test_data_item(const size_t index) {
  return {test_data_word(index, 0x243f6a8885a308d3ULL),
          test_data_word(index, 0x13198a2e03707344ULL)};
}

inline uint64_t rotl64(uint64_t value, unsigned shift) {
  return (value << shift) | (value >> (64 - shift));
}

inline void keccakf1600(uint64_t state[25]) {
  static constexpr uint64_t rc[24] = {
      0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
      0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
      0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
      0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
      0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
      0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
  static constexpr unsigned rotations[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                                             27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};
  static constexpr unsigned positions[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                                             15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};
  for (unsigned round = 0; round < 24; ++round) {
    uint64_t column[5];
    for (unsigned i = 0; i < 5; ++i)
      column[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
    for (unsigned i = 0; i < 5; ++i) {
      const uint64_t value = column[(i + 4) % 5] ^ rotl64(column[(i + 1) % 5], 1);
      for (unsigned j = 0; j < 25; j += 5)
        state[j + i] ^= value;
    }
    uint64_t value = state[1];
    for (unsigned i = 0; i < 24; ++i) {
      const unsigned position = positions[i];
      column[0] = state[position];
      state[position] = rotl64(value, rotations[i]);
      value = column[0];
    }
    for (unsigned j = 0; j < 25; j += 5) {
      for (unsigned i = 0; i < 5; ++i)
        column[i] = state[j + i];
      for (unsigned i = 0; i < 5; ++i)
        state[j + i] ^= (~column[(i + 1) % 5]) & column[(i + 2) % 5];
    }
    state[0] ^= rc[round];
  }
}

inline void make_state(const uint32_t* header, unsigned first, uint64_t state[25]) {
  for (unsigned i = 0; i < 25; ++i)
    state[i] = 0;
  const uint32_t word0 = (header[0] & 0xffffff00u) | ((header[0] + first) & 0xffu);
  state[0] = static_cast<uint64_t>(word0) | (static_cast<uint64_t>(header[1]) << 32);
  for (unsigned i = 1; i < 9; ++i)
    state[i] =
        static_cast<uint64_t>(header[i * 2]) | (static_cast<uint64_t>(header[i * 2 + 1]) << 32);
  keccakf1600(state);
}

inline uint32_t fnv1a(uint32_t a, uint32_t b) {
  return (a ^ b) * 0x01000193u;
}

template <bool Compact> class SeedKernel;
template <bool Compact> class SearchKernel;

class State {
public:
  sycl::device device;
  sycl::queue queue;
  bool compact_test_data = false;
  Uint2* data = nullptr;
  uint32_t* header = nullptr;
  uint64_t* states = nullptr;
  Uint2* hashes = nullptr;
  uint32_t* target = nullptr;
  Result* result = nullptr;
  unsigned hash_capacity = 0;
  std::mutex mutex;

  explicit State(const std::string& dev_str, const bool use_compact_test_data)
      : device(get_dev(dev_str)),
        queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
        compact_test_data(use_compact_test_data && device.is_cpu()) {
    if (!mom_has_usm_device(device) || !mom_has_usm_shared(device))
      throw std::string("verthash requires SYCL device and shared USM");
    try {
      load_data();
      header = sycl::malloc_device<uint32_t>(20, queue);
      states = sycl::malloc_device<uint64_t>(8 * 25, queue);
      target = sycl::malloc_device<uint32_t>(8, queue);
      result = sycl::malloc_shared<Result>(1, queue);
      if (!header || !states || !target || !result)
        throw std::string("Can't allocate verthash buffers");
    } catch (...) {
      release();
      throw;
    }
  }
  ~State() {
    release();
  }
  void release() noexcept {
    sycl_cleanup_noexcept("verthash wait", [&] {
      queue.wait_and_throw();
    });
    free_all();
  }
  void free_ptr(auto*& p) noexcept {
    if (p)
      try {
        sycl::free(p, queue);
      } catch (...) {
      }
    p = nullptr;
  }
  void free_all() noexcept {
    free_ptr(result);
    free_ptr(target);
    free_ptr(hashes);
    free_ptr(states);
    free_ptr(header);
    free_ptr(data);
  }
  void ensure_hashes(unsigned count) {
    if (hashes && count <= hash_capacity)
      return;
    queue.wait_and_throw();
    free_ptr(hashes);
    hash_capacity = 0;
    Uint2* const next = sycl::malloc_device<Uint2>(static_cast<size_t>(count) * 4, queue);
    if (!next)
      throw std::string("Can't allocate verthash hash buffer");
    hashes = next;
    hash_capacity = count;
  }

private:
  void load_data() {
    std::vector<Uint2> host;
    if (compact_test_data) {
      host.resize(TEST_DATA_ELEMENTS);
      for (size_t i = 0; i < host.size(); ++i)
        host[i] = test_data_item(i);
    } else {
      const std::filesystem::path selected = data_file::selection();
      std::error_code error;
      const bool exists = std::filesystem::exists(selected, error) && !error;
      if (error)
        throw std::string("Cannot access Verthash data file: ") + selected.string();
      if (!data_file::has_expected_size(selected)) {
        if (exists)
          throw std::string("Invalid Verthash data file; remove it to regenerate: ") +
                selected.string();
        std::fprintf(stderr,
                     "Verthash data is missing; generating 1.20 GiB at %s. "
                     "This one-time step usually takes a few minutes.\n",
                     selected.string().c_str());
        data_file::generate(selected);
        std::fprintf(stderr, "Verthash data generated and verified.\n");
      }

      {
        std::ifstream file(selected, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != static_cast<std::streamoff>(DATA_BYTES))
          throw std::string("Invalid Verthash data file: ") + selected.string();
        file.seekg(0);
        host.resize(DATA_BYTES / sizeof(Uint2));
        if (!file.read(reinterpret_cast<char*>(host.data()), DATA_BYTES))
          throw std::string("Cannot read Verthash data file: ") + selected.string();
        const int trailing = file.peek();
        if (trailing != std::char_traits<char>::eof() || !file.eof())
          throw std::string("Invalid Verthash data file: ") + selected.string();
      }
      if (data_file::sha256(reinterpret_cast<const std::uint8_t*>(host.data()), DATA_BYTES) !=
          data_file::expected_sha256)
        throw std::string("Invalid Verthash data file; remove it to regenerate: ") +
              selected.string();
    }
    const size_t data_bytes = host.size() * sizeof(Uint2);
    // Intel's Windows CPU OpenCL runtime can fault while copying the 1.20 GiB dataset into device
    // USM. Shared USM is native host memory on a CPU and avoids that runtime path; GPU builds retain
    // device USM and its measured random-read performance.
    const bool cpu_data = device.is_cpu();
    data = cpu_data ? sycl::malloc_shared<Uint2>(host.size(), queue)
                    : sycl::malloc_device<Uint2>(host.size(), queue);
    if (!data)
      throw compact_test_data ? std::string("Can't allocate bounded Verthash test dataset")
                              : std::string("Can't allocate verthash dataset (1.20 GiB)");
    if (cpu_data)
      std::memcpy(data, host.data(), data_bytes);
    else
      sycl_wait_and_throw(queue.memcpy(data, host.data(), data_bytes), device);
  }
};

static DeviceStateRegistry<State>& registry() {
  static auto* const value = new DeviceStateRegistry<State>;
  return *value;
}

static State& get_state(const std::string& dev_str, const bool is_test) {
  const char* const configured = std::getenv("MOM_SYCL_PORTABLE_TEST");
  const bool requested = is_test && configured && std::strcmp(configured, "1") == 0;
  // Resolve the device before choosing the registry key. A GPU test with an inherited switch
  // must share the normal official state rather than allocating a second full dataset.
  const bool compact = requested && get_dev(dev_str).is_cpu();
  const std::string key = dev_str + (compact ? "\x1f" "compact-test" : "\x1f" "full");
  return registry().get(key, [&] { return std::make_unique<State>(dev_str, compact); });
}

void verthash_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "verthash: ordered SYCL cleanup failed\n");
  }
}

template <bool Compact>
static sycl::event submit_search(State& s, uint32_t start_nonce, unsigned count, bool is_test) {
  uint32_t* const header = s.header;
  uint64_t* const states = s.states;
  Uint2* const data = s.data;
  Uint2* const hashes = s.hashes;
  uint32_t* const target = s.target;
  Result* const result = s.result;
  constexpr uint32_t data_mdiv = Compact ? TEST_MDIV : MDIV;
  s.queue.submit([&](sycl::handler& h) {
    h.parallel_for<SeedKernel<Compact>>(
        sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const size_t hash = id[0];
      const uint32_t nonce = start_nonce + static_cast<uint32_t>(hash);
      uint64_t st[25]{};
      for (unsigned i = 0; i < 9; ++i)
        st[i] =
            static_cast<uint64_t>(header[i * 2]) | (static_cast<uint64_t>(header[i * 2 + 1]) << 32);
      st[9] = static_cast<uint64_t>(header[18]) | (static_cast<uint64_t>(nonce) << 32);
      st[10] ^= 0x06u;
      st[16] ^= 0x8000000000000000ULL;
      keccakf1600(st);
      for (unsigned i = 0; i < 4; ++i)
        hashes[hash * 4 + i] = {static_cast<uint32_t>(st[i]), static_cast<uint32_t>(st[i] >> 32)};
    });
  });
  return s.queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> sha3(WG / 4 * 128, h);
#if defined(MOM_SYCL_PORTABLE_OPENCL)
    // Standards-only OpenCL lacks the required subgroup shuffle extension.
    // Exchange each hash's four lanes through local memory instead.
    sycl::local_accessor<uint32_t, 1> exchange(WG * 2, h);
#endif
    h.parallel_for<SearchKernel<Compact>>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(count) * 4), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const size_t gid = it.get_global_id(0);
          const unsigned lid = static_cast<unsigned>(it.get_local_id(0)), lane = lid & 3u;
#if defined(MOM_SYCL_PORTABLE_OPENCL)
          const unsigned group_base = lid & ~3u;
#else
          const sycl::sub_group subgroup = it.get_sub_group();
          const unsigned subgroup_base = subgroup.get_local_linear_id() & ~3u;
#endif
          const size_t hash = gid >> 2;
          const uint32_t nonce = start_nonce + static_cast<uint32_t>(hash);
          uint32_t* combined = &sha3[(lid >> 2) * 128];

          for (unsigned pass = 0; pass < 2; ++pass) {
            uint64_t st[25];
            const uint64_t* base = states + (lane * 2 + pass) * 25;
            for (unsigned i = 0; i < 25; ++i)
              st[i] = base[i];
            st[0] ^= static_cast<uint64_t>(header[18]) | (static_cast<uint64_t>(nonce) << 32);
            st[1] ^= 0x06u;
            st[8] ^= 0x8000000000000000ULL;
            keccakf1600(st);
            for (unsigned i = 0; i < 8; ++i) {
              const unsigned at = (lane * 16 + pass * 8 + i) * 2;
              combined[at] = static_cast<uint32_t>(st[i]);
              combined[at + 1] = static_cast<uint32_t>(st[i] >> 32);
            }
          }
          sycl::group_barrier(it.get_group());

          Uint2 value = hashes[hash * 4 + lane];
          uint32_t accumulator = 0x811c9dc5u;
          for (unsigned i = 0; i < 4096; ++i) {
            const uint32_t word = combined[i & 127u];
            const unsigned shift = i >> 7;
            const uint32_t seek = shift ? mo_rotate(word, shift) : word;
            const Uint2 item = data[((fnv1a(seek, accumulator) % data_mdiv) << 1) + lane];
            value.x = fnv1a(value.x, item.x);
            value.y = fnv1a(value.y, item.y);
#if defined(MOM_SYCL_PORTABLE_OPENCL)
            exchange[lid * 2] = item.x;
            exchange[lid * 2 + 1] = item.y;
            it.barrier(sycl::access::fence_space::local_space);
            for (unsigned l = 0; l < 4; ++l) {
              accumulator = fnv1a(accumulator, exchange[(group_base + l) * 2]);
              accumulator = fnv1a(accumulator, exchange[(group_base + l) * 2 + 1]);
            }
            it.barrier(sycl::access::fence_space::local_space);
#else
            for (unsigned l = 0; l < 4; ++l) {
              accumulator =
                  fnv1a(accumulator, sycl::select_from_group(subgroup, item.x, subgroup_base + l));
              accumulator =
                  fnv1a(accumulator, sycl::select_from_group(subgroup, item.y, subgroup_base + l));
            }
#endif
          }
          uint32_t words[8];
#if defined(MOM_SYCL_PORTABLE_OPENCL)
          exchange[lid * 2] = value.x;
          exchange[lid * 2 + 1] = value.y;
          it.barrier(sycl::access::fence_space::local_space);
          for (unsigned l = 0; l < 4; ++l) {
            words[l * 2] = exchange[(group_base + l) * 2];
            words[l * 2 + 1] = exchange[(group_base + l) * 2 + 1];
          }
          it.barrier(sycl::access::fence_space::local_space);
#else
          for (unsigned l = 0; l < 4; ++l) {
            words[l * 2] = sycl::select_from_group(subgroup, value.x, subgroup_base + l);
            words[l * 2 + 1] = sycl::select_from_group(subgroup, value.y, subgroup_base + l);
          }
#endif
          if (lane != 3)
            return;
          bool hit = true;
          for (int i = 7; i >= 0; --i)
            if (words[i] != target[i]) {
              hit = words[i] < target[i];
              break;
            }
          if ((is_test && hash == 0) || hit) {
            using Atomic =
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>;
            if (Atomic(result->count).fetch_add(1) == 0) {
              result->nonce = nonce;
              for (unsigned i = 0; i < 8; ++i)
                result->hash[i] = words[i];
            }
          }
        });
  });
}

} // namespace mom_verthash

using namespace mom_verthash;
int verthash(unsigned, uint32_t, const uint8_t* input, unsigned input_size, uint8_t* output,
             uint8_t*, uint64_t* pnonce, const uint8_t* target, const uint8_t*, unsigned intensity,
             bool is_test, bool is_benchmark, const std::string& dev_str) {
  if (!input || !output || !pnonce || !target)
    throw std::string("verthash received incomplete job buffers");
  if (input_size != 80)
    throw std::string("verthash requires an 80-byte header");
  if (intensity == 0 || intensity % 16)
    throw std::string("verthash intensity must be positive and divisible by 16");
  // Bounded data is a correctness fixture only; benchmarks and all mining use the official state.
  State& state = get_state(dev_str, is_test && !is_benchmark);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_hashes(intensity);
  uint32_t header[20];
  std::memcpy(header, input, sizeof(header));
  uint64_t states[8][25];
  for (unsigned i = 0; i < 8; ++i)
    make_state(header, i + 1, states[i]);
  uint32_t start_nonce = static_cast<uint32_t>(*pnonce);
  MomSyclHostTransferGuard host_transfers(state.queue, "verthash host transfers");
  state.queue.memcpy(state.header, header, sizeof(header));
  state.queue.memcpy(state.states, states, sizeof(states));
  state.queue.memcpy(state.target, target, HASH_LEN);
  std::memset(state.result, 0, sizeof(Result));
  const sycl::event search = state.compact_test_data
                                 ? submit_search<true>(state, start_nonce, intensity, is_test)
                                 : submit_search<false>(state, start_nonce, intensity, is_test);
  sycl_wait_and_throw(search, state.device);
  if (!state.result->count)
    return 0;
  *pnonce = state.result->nonce;
  std::memcpy(output, state.result->hash, HASH_LEN);
  return 1;
}
