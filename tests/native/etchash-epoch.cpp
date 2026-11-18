#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include "../../native/consts.h"
#include "../../xmrig/3rdparty/libethash/data_sizes.h"

// Production epoch constants, height helpers, and is_zero_hash from sycl/etchash/device.inc.
#include "etchash_epoch_helpers.inc"

static uint32_t seed_lookup_result = 0;
static uint32_t seed_epoch_from_hash(const uint8_t*) {
  return seed_lookup_result;
}

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
    assert(error == "Bad etchash epoch");
    return false;
  }
}

int main() {
  const uint8_t zero_seed[HASH_LEN] = {};
  uint8_t pool_seed[HASH_LEN] = {};
  pool_seed[0] = 1;
  uint32_t epoch = 0, seed_epoch = 0;

  assert(resolves(0, nullptr, epoch, seed_epoch) && epoch == 0 && seed_epoch == 0);
  assert(resolves(11699999, zero_seed, epoch, seed_epoch) && epoch == 389 && seed_epoch == 389);
  assert(resolves(11700000, zero_seed, epoch, seed_epoch) && epoch == 195 && seed_epoch == 390);
  // The last height whose doubled epoch still has a libethash size-table entry.
  assert(resolves(122879999, zero_seed, epoch, seed_epoch) && epoch == 2047);
  // Pool-controlled heights past the tables must be rejected before cache_sizes/dag_sizes indexing.
  assert(!resolves(122880000, nullptr, epoch, seed_epoch));
  assert(!resolves(UINT32_MAX, zero_seed, epoch, seed_epoch));

  seed_lookup_result = 389;
  assert(resolves(UINT32_MAX, pool_seed, epoch, seed_epoch) && epoch == 389 && seed_epoch == 389);
  seed_lookup_result = 4095;
  assert(resolves(0, pool_seed, epoch, seed_epoch) && epoch == 2047 && seed_epoch == 4095);
  std::printf("Etchash epoch bounds: height and seed resolution stay within %zu table entries\n",
              sizeof(dag_sizes) / sizeof(dag_sizes[0]));
}
