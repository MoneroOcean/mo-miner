// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include "../xmrig/3rdparty/libethash/ethash.h"
#include "../xmrig/3rdparty/libethash/endian.h"
#include "base/crypto/sha3.h"

#include <cstring>

namespace {

constexpr unsigned ETHASH_NODE_WORDS = 16;

union EthashNode {
  uint8_t bytes[ETHASH_NODE_WORDS * sizeof(uint32_t)];
  uint32_t words[ETHASH_NODE_WORDS];
};

static_assert(sizeof(EthashNode) == ETHASH_HASH_BYTES);

void keccak256(void* out, const void* in, const unsigned bytes) {
  sha3_HashBuffer(256, SHA3_FLAGS_KECCAK, in, bytes, out, 32);
}

void keccak512(void* out, const void* in, const unsigned bytes) {
  sha3_HashBuffer(512, SHA3_FLAGS_KECCAK, in, bytes, out, 64);
}

}

extern "C" ethash_h256_t ethash_get_seedhash(uint64_t epoch) {
  ethash_h256_t result{};
  for (uint32_t i = 0; i < epoch; ++i) keccak256(&result, &result, 32);
  return result;
}

extern "C" bool ethash_compute_cache_nodes(
  void* nodes_ptr,
  uint64_t cache_size,
  const ethash_h256_t* seed
) {
  if (cache_size % sizeof(EthashNode) != 0) return false;
  const uint32_t num_nodes = static_cast<uint32_t>(cache_size / sizeof(EthashNode));
  auto* nodes = static_cast<EthashNode*>(nodes_ptr);

  keccak512(nodes[0].bytes, seed, 32);
  for (uint32_t i = 1; i != num_nodes; ++i) {
    keccak512(nodes[i].bytes, nodes[i - 1].bytes, 64);
  }

  for (uint32_t j = 0; j != ETHASH_CACHE_ROUNDS; ++j) {
    for (uint32_t i = 0; i != num_nodes; ++i) {
      const uint32_t idx = nodes[i].words[0] % num_nodes;
      EthashNode data = nodes[(num_nodes - 1 + i) % num_nodes];
      for (uint32_t w = 0; w != ETHASH_NODE_WORDS; ++w) {
        data.words[w] ^= nodes[idx].words[w];
      }
      keccak512(nodes[i].bytes, data.bytes, sizeof(data));
    }
  }

  fix_endian_arr32(nodes[0].words, num_nodes * ETHASH_NODE_WORDS);
  return true;
}
