#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include "../../native/consts.h"
#include "../../xmrig/3rdparty/libethash/ethash.h"
#include "../../xmrig/3rdparty/libethash/data_sizes.h"

// Compile-only stand-ins for the SYCL hit store, which the epoch helpers never touch.
template <typename Word = uint8_t> struct PowHits {
  uint32_t count;
};
template <typename Word> uint32_t mom_store_pow_hit(PowHits<Word>*, uint64_t, const Word*) {
  return GPU_MAX_HITS;
}

// Deterministic seed chain: each epoch's seed hash encodes that epoch, so tests name pool seeds.
ethash_h256_t ethash_get_seedhash(const uint64_t epoch) {
  ethash_h256_t seed{};
  seed.b[0] = static_cast<uint8_t>(epoch);
  seed.b[1] = static_cast<uint8_t>(epoch >> 8);
  seed.b[31] = 0xa5;
  return seed;
}

// The whole production device file: epoch constants, height helpers, is_zero_hash and the
// bounded pool-seed search.
#include "../../sycl/etchash/device.inc"

struct EpochResolver {
  uint8_t cached_seed_hash[HASH_LEN]{};
  uint32_t cached_seed_epoch = UINT32_MAX;
#include "etchash_resolve_epochs.inc"
};

static bool resolves(const uint32_t height, const uint8_t* const seed, uint32_t& epoch,
                     uint32_t& seed_epoch) {
  EpochResolver resolver;
  try {
    resolver.resolve_epochs(height, seed, epoch, seed_epoch);
    return true;
  } catch (const std::string& error) {
    assert(error == "Bad etchash epoch" || error == "Unknown etchash seed hash");
    return false;
  }
}

int main() {
  const uint8_t zero_seed[HASH_LEN] = {};
  uint32_t epoch = 0, seed_epoch = 0;

  assert(resolves(0, nullptr, epoch, seed_epoch) && epoch == 0 && seed_epoch == 0);
  assert(resolves(11699999, zero_seed, epoch, seed_epoch) && epoch == 389 && seed_epoch == 389);
  assert(resolves(11700000, zero_seed, epoch, seed_epoch) && epoch == 195 && seed_epoch == 390);
  // The last height whose doubled epoch still has a libethash size-table entry.
  assert(resolves(122879999, zero_seed, epoch, seed_epoch) && epoch == 2047);
  // Pool-controlled heights past the tables must be rejected before cache_sizes/dag_sizes indexing.
  assert(!resolves(122880000, nullptr, epoch, seed_epoch));
  assert(!resolves(UINT32_MAX, zero_seed, epoch, seed_epoch));

  // A pool seed overrides the height and keeps pre-fork epochs below the fork's seed epoch.
  const ethash_h256_t pre_fork = ethash_get_seedhash(389);
  assert(resolves(UINT32_MAX, pre_fork.b, epoch, seed_epoch) && epoch == 389 && seed_epoch == 389);
  const ethash_h256_t last = ethash_get_seedhash(4095);
  assert(resolves(0, last.b, epoch, seed_epoch) && epoch == 2047 && seed_epoch == 4095);
  // The seed chain search is bounded; an unknown seed is rejected rather than guessed.
  ethash_h256_t unknown = ethash_get_seedhash(4096);
  assert(!resolves(0, unknown.b, epoch, seed_epoch));
  unknown.b[31] = 0;
  assert(!resolves(0, unknown.b, epoch, seed_epoch));
  std::printf("Etchash epoch bounds: height and seed resolution stay within %zu table entries\n",
              sizeof(dag_sizes) / sizeof(dag_sizes[0]));
}
