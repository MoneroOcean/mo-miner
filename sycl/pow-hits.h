// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include "../native/consts.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>

// Target hits of one single-blob search (sycl/lib.h): up to GPU_MAX_HITS nonces with their 32-byte
// hashes, stored as Word[32 / sizeof(Word)]. `count` keeps advancing past the capacity, so the host
// must clamp it; kernel completion publishes the slots.
template <typename Word = uint8_t>
struct PowHits {
  static constexpr unsigned hash_words = HASH_LEN / sizeof(Word);
  uint32_t count;
  uint64_t nonce[GPU_MAX_HITS];
  Word hash[GPU_MAX_HITS][hash_words];
};

// Claim the next hit slot with one device-scope atomic and store the nonce and hash there. Returns
// the claimed slot so callers can add per-hit data, or GPU_MAX_HITS when every slot is taken.
template <typename Word>
inline uint32_t mom_store_pow_hit(PowHits<Word>* const hits, const uint64_t nonce,
                                  const Word* const hash) {
  using atomic_ref = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>;
  const uint32_t slot = atomic_ref(hits->count).fetch_add(1);
  if (slot >= GPU_MAX_HITS)
    return GPU_MAX_HITS;
  hits->nonce[slot] = nonce;
  for (unsigned i = 0; i < PowHits<Word>::hash_words; ++i)
    hits->hash[slot][i] = hash[i];
  return slot;
}

// Copy completed hits into the single-blob result ABI: hit i goes to nonces[i] and
// output[i * HASH_LEN]. Returns the number of hits copied.
template <typename Word>
inline int mom_take_pow_hits(const PowHits<Word>& hits, uint8_t* const output,
                             uint64_t* const nonces) {
  const uint32_t count = std::min<uint32_t>(hits.count, GPU_MAX_HITS);
  for (uint32_t i = 0; i < count; ++i) {
    nonces[i] = hits.nonce[i];
    std::memcpy(output + i * HASH_LEN, hits.hash[i], HASH_LEN);
  }
  return static_cast<int>(count);
}
