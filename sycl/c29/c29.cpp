// Copyright GNU GPLv3 (c) 2025-2026 MoneroOcean <support@moneroocean.stream>

// SYCL c29 miner prototype based on Grin GPU Miner (https://github.com/swap-dev/SwapReferenceMiner)
// OpenCL mining code by Jiri Photon Vadura and John Tromp
//
// Performance/portability notes:
// - Intel seed/trim profiles are dominated by bucket scatter atomics and repeated graph trimming.
// - Intel uses 42 trim pairs and at most a 4 MiB handoff to the dense host cycle finder.
//   Shallower trimming can overflow that capacity and lose proofs; 42 pairs is not a fundamental
//   minimum. Portable builds retain 80 pairs and a 1 MiB handoff.
// - The B580 comparison uses a local NVIDIA peer because no controlled Intel reference miner is
//   available. That identifies optimization headroom, not an established hardware ceiling.
//   The lolMiner/RTX 5060 Ti peer target is 7.40 g/s, reproduced in three local runs at 150 W.
// - Keep the complete portable SYCL path available; vendor-specific changes must be capability-gated.
#include <sycl/sycl.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "state.inc"

namespace {

int c29_portable_test(const uint8_t* input, unsigned input_size, uint8_t* output,
                      const std::string& dev_str);

} // namespace

#include "search.inc"

namespace {

int c29_portable_test(const uint8_t* const input, const unsigned input_size,
                      uint8_t* const output, const std::string& dev_str) {
  // CPU-compatible bounded guard; full C29 vectors retain end-to-end solver coverage on discrete
  // GPUs.
  sycl::queue queue(get_dev(dev_str), sycl::property::queue::in_order{});
  union {
    uint8_t blake_output[32];
    uint64_t k[4];
  };
  rx_blake2b(blake_output, 32, input, input_size);
  const uint64_t seed_k0 = k[0], seed_k1 = k[1], seed_k2 = k[2], seed_k3 = k[3];

  sycl::uint2* const endpoints = sycl::malloc_shared<sycl::uint2>(4, queue);
  if (!endpoints)
    throw std::bad_alloc();
  try {
    const sycl::event event = queue.submit([&](sycl::handler& handler) {
      handler.parallel_for(sycl::range<1>{4}, [=](sycl::id<1> item) {
        const uint32_t edge = item[0] == 3 ? 63u : item[0] == 2 ? 62u : item[0];
        uint64_t v0 = seed_k0, v1 = seed_k1, v2 = seed_k2, v3 = seed_k3;
        uint64_t hash_block[EDGE_BLOCK_SIZE];
        siphash_fill_block(v0, v1, v2, v3, 0, hash_block);
        const uint64_t hash_last = hash_block[EDGE_BLOCK_MASK];
        const uint64_t hash_lookup =
            edge == EDGE_BLOCK_MASK ? hash_last : hash_block[edge] ^ hash_last;
        endpoints[item] = sycl::uint2{static_cast<uint32_t>(hash_lookup & EDGE_MASK),
                                      static_cast<uint32_t>((hash_lookup >> 32) & EDGE_MASK)};
      });
    });
    sycl_wait_and_throw(event, queue.get_device());
    for (unsigned i = 0; i < 4; ++i) {
      const uint32_t pair[2] = {endpoints[i].x(), endpoints[i].y()};
      std::memcpy(output + i * sizeof(pair), pair, sizeof(pair));
    }
  } catch (...) {
    sycl::free(endpoints, queue);
    throw;
  }
  sycl::free(endpoints, queue);
  return 1;
}

} // namespace
