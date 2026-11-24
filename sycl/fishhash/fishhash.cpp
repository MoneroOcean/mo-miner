// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
//
// Iron Fish FishHash and KarlsenHashV2 share a fixed DAG. FishHash starts from a 64-byte BLAKE3 seed
// and fetches three DAG rows per mix round. Each 64-bit lane updates with
// f0 * fnv_pair(mix, f1) + (mix ^ f2), where fnv_pair applies FNV-1 independently to each 32-bit half.
// The 32-byte digest is BLAKE3(seed || collapsed_mix). The 72 MiB light cache and 4.5 GiB DAG
// both derive from the same fixed seed. Ported bit-exact from github.com/iron-fish/
// fish-hash (cpp/FishHash.cpp + 3rdParty/{blake3,keccak}); validated offline (light_cache[0], blake3
// seed, final hash). Mining and benchmarking always use the full DAG; the lazy 72 MiB light-cache
// path is only for the default offline vector. Production benchmarks use Iron Fish's 180-byte,
// nonce-at-172 big-endian header (FIP-9 swaps the old randomness/graffiti positions). Older
// 76-byte synthetic benchmark rates are not comparable;
// the supported short offline vector is 40 bytes with a little-endian nonce at byte 32.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "../lib-internal.h"
#include "../pow-hits.h"
#include "../pow-target.h"
#include "../../native/job-boundary.h"
#include "../../native/consts.h"

namespace mom_fishhash {

#include "device.inc"
#include "search.inc"
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
#include "cooperative-search.inc"
#endif
#include "dag.inc"

#include "state.inc"

} // namespace mom_fishhash

using namespace mom_fishhash;

// Both algorithms use the same fixed DAG and host transfer lifetime.
template <bool Karlsen>
static int searchFixedDag(const uint8_t* input, unsigned input_size, uint8_t* output,
                          uint64_t* pnonce, const uint8_t* target, unsigned intensity,
                          bool is_test, bool is_benchmark, const std::string& dev_str) {
  constexpr const char* algorithm = Karlsen ? "karlsenhashv2" : "fishhash";
  if (!input || !output || !pnonce || !target || !intensity)
    throw std::string(algorithm) + " requires input, output, nonce, target, and nonzero intensity";
  // The scalar kernel has the widest per-hash padding; this also covers the cooperative path.
  if (!mom::job_boundary::padded_u32_range_fits(intensity, 128))
    throw std::string(algorithm) + " intensity is too large";
  const std::size_t nonce_off = Karlsen ? 72 : mom::job_boundary::fishhash_nonce_offset(input_size);
  if constexpr (Karlsen) {
    if (!mom::job_boundary::valid_karlsenhashv2_layout(input_size, nonce_off))
      throw std::string("karlsenhashv2 requires an 80-byte header with nonce at offset 72");
  } else if (!mom::job_boundary::valid_fishhash_layout(input_size, nonce_off)) {
    throw std::string(
        "fishhash requires a 40-byte header with nonce at offset 32 or a 180-byte header with "
        "nonce at offset 172");
  }

  // Only the 180-byte Iron Fish FIP-9 header encodes the nonce in big-endian order.
  const bool bigEndian = !Karlsen && input_size == 180;
  uint64_t start_nonce = 0;
  if (bigEndian) {
    for (unsigned i = 0; i < 8; ++i)
      start_nonce = (start_nonce << 8) | input[nonce_off + i];
  } else {
    std::memcpy(&start_nonce, input + nonce_off, sizeof(start_nonce));
  }

  State& state = persistentDeviceState<State>(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  const bool lightTest = state.prepareJob(input_size, is_test, is_benchmark);
  // Device copies borrow the caller's job bytes until the queue finishes using them.
  MomSyclHostTransferGuard host_transfers(state.queue, Karlsen ? "karlsenhashv2 host transfers" : "fishhash host transfers");
  if (state.shared_io) {
    std::memcpy(state.input, input, input_size);
    std::memcpy(state.target, target, HASH_LEN);
  } else {
    state.queue.memcpy(state.input, input, input_size);
    state.queue.memcpy(state.target, target, HASH_LEN);
  }
  const auto search = [&]([[maybe_unused]] bool cooperative, unsigned count, bool test) {
    if constexpr (Karlsen) {
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
      if (cooperative)
        return submit_karlsenhashv2_search_coop(state.queue, state.input, input_size, start_nonce,
                                               state.dag, count, state.target, state.result,
                                               test, state.coop_workgroup);
#endif
      if (lightTest)
        return submit_karlsenhashv2_search<false>(state.queue, state.input, input_size, start_nonce,
                                                 state.cache, nullptr, count, state.target, state.result, test);
      return submit_karlsenhashv2_search<true>(state.queue, state.input, input_size, start_nonce,
                                              nullptr, state.dag, count, state.target, state.result, test);
    } else {
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
      if (cooperative)
        return submit_fishhash_search_coop(state.queue, state.input, input_size, start_nonce, state.dag,
                                          count, state.target, state.result, test, nonce_off, bigEndian,
                                          state.coop_workgroup);
#endif
      if (lightTest)
        return submit_fishhash_search<false>(state.queue, state.input, input_size, start_nonce,
                                            state.cache, nullptr, count, state.target, state.result,
                                            test, nonce_off, bigEndian);
      return submit_fishhash_search<true>(state.queue, state.input, input_size, start_nonce,
                                          nullptr, state.dag, count, state.target, state.result,
                                          test, nonce_off, bigEndian);
    }
  };
  int& choice = Karlsen ? state.karlsenhashv2Cooperative : state.fishhashCooperative;
  const bool cooperative = !lightTest && selectCooperative(state, choice, algorithm, !Karlsen, search);
  std::memset(state.result, 0, sizeof(*state.result));
  sycl_wait_and_throw(search(cooperative, intensity, is_test), state.device);

  static const bool perf_log = std::getenv(Karlsen ? "MOM_KARLSEN_PERF" : "MOM_FISHHASH_PERF") != nullptr;
  if (perf_log && !is_test) {
    auto& window = Karlsen ? state.karlsenhashv2Rate : state.fishhashRate;
    window.record(intensity, [&](double hashesPerSecond, double) {
      std::fprintf(stderr, "[%s] %.2f MH/s (%s)\n", algorithm, hashesPerSecond / 1e6, dev_str.c_str());
    });
  }
  return mom_take_pow_hits(*state.result, output, pnonce);
}

int fishhash(const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size,
             uint8_t* const output, uint8_t* const /*mix_hash*/, uint64_t* const pnonce,
             const uint8_t* const target, const uint8_t* const /*seed_hash*/,
             const unsigned intensity, const bool is_test, const bool is_benchmark,
             const std::string& dev_str) {
  return searchFixedDag<false>(input, input_size, output, pnonce, target, intensity, is_test, is_benchmark, dev_str);
}

int karlsenhashv2(const unsigned, const uint32_t, const uint8_t* const input,
                  const unsigned input_size, uint8_t* const output, uint8_t* const /*mix_hash*/,
                  uint64_t* const pnonce, const uint8_t* const target,
                  const uint8_t* const /*seed_hash*/, const unsigned intensity, const bool is_test,
                  const bool is_benchmark, const std::string& dev_str) {
  return searchFixedDag<true>(input, input_size, output, pnonce, target, intensity, is_test, is_benchmark, dev_str);
}
