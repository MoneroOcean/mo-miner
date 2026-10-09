// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
//
// FishHash (Iron Fish IRON / Karlsen KLS) GPU search kernel. ASIC-resistant, memory-hard, Ethash-derived.
// Per hash: blake3(header) -> 64B seed -> 32 dataset accesses (3x 128B fetches, mix=f0*f1+f2 u64) ->
// collapse -> blake3(seed||mix_hash) -> 32B. DAG is FIXED (not epoch-based): 1.18M x 64B light cache ->
// 37.7M x 128B (4.6 GB) dataset, both from a fixed seed. Ported bit-exact from github.com/iron-fish/
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
#include "../../native/job-boundary.h"
#include "../../native/consts.h"

namespace mom_fishhash {

#include "device.inc"
#include "search.inc"
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
#include "cooperative_search.inc"
#endif
#include "dag.inc"

#include "state.inc"

} // namespace mom_fishhash

using namespace mom_fishhash;

// Validate header/nonce layout and search the fixed DAG; short tests may derive light rows.
int fishhash(const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size,
             uint8_t* const output, uint8_t* const mix_hash, uint64_t* const pnonce,
             const uint8_t* const target, const uint8_t* const /*seed_hash*/,
             const unsigned intensity, const bool is_test, const bool is_benchmark,
             const std::string& dev_str) {
  if (!input || !output || !pnonce || !target || !intensity)
    throw std::string("fishhash requires input, output, nonce, target, and nonzero intensity");
  // The scalar kernel has the widest per-hash padding; this also covers the cooperative path.
  if (!mom::job_boundary::padded_u32_range_fits(intensity, 128))
    throw std::string("fishhash intensity is too large");
  const std::size_t nonce_off = mom::job_boundary::fishhash_nonce_offset(input_size);
  if (!mom::job_boundary::valid_fishhash_layout(input_size, nonce_off))
    throw std::string(
        "fishhash requires a 40-byte header with nonce at offset 32 or a 180-byte header with "
        "nonce at offset 172");
  FishState& state = fishhash_state(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_io(input_size);
  // MOM_FISHHASH_FULL_TEST makes the normal one-nonce vector build/use the full DAG, so the
  // production cooperative kernel can be checked bit-exactly without slowing the default light-cache
  // vector. The cooperative search is the GPU default; MOM_FISHHASH_COOP=0 retains the scalar path
  // for diagnosis.
  const bool full_test = is_test && std::getenv("MOM_FISHHASH_FULL_TEST") != nullptr;
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  const char* const coop_env = std::getenv("MOM_FISHHASH_COOP");
  const bool cooperative = state.device.is_gpu() && !(coop_env && coop_env[0] == '0');
#endif
  if (is_test && !full_test)
    state.ensure_cache(!is_benchmark); // lazy path: no 4.6-GiB allocation
  else
    state.ensure_dag(!is_benchmark); // mining/bench or explicit full test

  // FIP-9 Iron Fish headers end with an 8-byte BE nonce; the short offline vector uses nonce@32 LE.
  const bool ironfish = input_size == 180;
  const bool nonce_be = ironfish, target_be = ironfish;
  uint64_t start_nonce = 0;
  if (nonce_be) {
    for (unsigned i = 0; i < 8; ++i)
      start_nonce = (start_nonce << 8) | input[nonce_off + i];
  } else {
    std::memcpy(&start_nonce, input + nonce_off, sizeof(start_nonce));
  }
  MomSyclHostTransferGuard host_transfers(state.queue, "fishhash host transfers");
  if (state.shared_io) {
    std::memcpy(state.input, input, input_size);
    std::memcpy(state.target, target, HASH_LEN);
  } else {
    state.queue.memcpy(state.input, input, input_size);
    state.queue.memcpy(state.target, target, HASH_LEN);
  }
  std::memset(state.result, 0, sizeof(FishResult));

  sycl::event search;
  if (is_test && !full_test) {
    search = submit_fishhash_search<false>(
        state.queue, state.input, input_size, start_nonce, state.cache, nullptr, intensity,
        state.target, state.result, is_test, nonce_off, nonce_be, target_be);
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  } else if (cooperative) {
    search = submit_fishhash_search_coop(
        state.queue, state.input, input_size, start_nonce, state.dag, intensity, state.target,
        state.result, is_test, nonce_off, nonce_be, target_be, state.coop_workgroup);
#endif
  } else {
    search = submit_fishhash_search<true>(state.queue, state.input, input_size, start_nonce, nullptr,
                                          state.dag, intensity, state.target, state.result, is_test,
                                          nonce_off, nonce_be, target_be);
  }
  sycl_wait_and_throw(search, state.device);

  static const bool perf_log = std::getenv("MOM_FISHHASH_PERF") != nullptr;
  if (perf_log && !is_test) {
    static uint64_t acc = 0;
    static auto t0 = std::chrono::steady_clock::now();
    acc += intensity;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec >= 2.0) {
      std::fprintf(stderr, "[fishhash] %.2f MH/s (%s)\n", static_cast<double>(acc) / sec / 1e6,
                   dev_str.c_str());
      acc = 0;
      t0 = std::chrono::steady_clock::now();
    }
  }

  const uint32_t count = state.result->count;
  if (count == 0)
    return 0;
  const uint32_t idx = std::min(count, MAX_OUTPUTS) - 1;
  *pnonce = state.result->nonce[idx];
  std::memcpy(output, state.result->output[idx], HASH_LEN);
  if (mix_hash)
    std::memset(mix_hash, 0, HASH_LEN);
  return 1;
}

// Reuse the fixed FishHash dataset with Karlsen hashing and target byte order.
int karlsenhashv2(const unsigned, const uint32_t, const uint8_t* const input,
                  const unsigned input_size, uint8_t* const output, uint8_t* const mix_hash,
                  uint64_t* const pnonce, const uint8_t* const target,
                  const uint8_t* const /*seed_hash*/, const unsigned intensity, const bool is_test,
                  const bool is_benchmark, const std::string& dev_str) {
  if (!input || !output || !pnonce || !target || !intensity)
    throw std::string(
      "karlsenhashv2 requires input, output, nonce, target, and nonzero intensity");
  if (!mom::job_boundary::padded_u32_range_fits(intensity, 128))
    throw std::string("karlsenhashv2 intensity is too large");
  if (!mom::job_boundary::valid_karlsenhashv2_layout(input_size, 72))
    throw std::string("karlsenhashv2 requires an 80-byte header with nonce at offset 72");
  FishState& state =
      fishhash_state(dev_str); // shares the FishHash 4.6 GB DAG (identical seed/params)
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_io(input_size);
  const bool full_test = is_test && std::getenv("MOM_FISHHASH_FULL_TEST") != nullptr;
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  const char* const coop_env = std::getenv("MOM_FISHHASH_COOP");
#endif
  if (is_test && !full_test)
    state.ensure_cache(!is_benchmark);
  else
    state.ensure_dag(!is_benchmark);

  uint64_t start_nonce = 0;
  std::memcpy(&start_nonce, input + 72, sizeof(start_nonce)); // nonce@72 LE
  MomSyclHostTransferGuard host_transfers(state.queue, "karlsenhashv2 host transfers");
  if (state.shared_io) {
    std::memcpy(state.input, input, input_size);
    std::memcpy(state.target, target, HASH_LEN);
  } else {
    state.queue.memcpy(state.input, input, input_size);
    state.queue.memcpy(state.target, target, HASH_LEN);
  }
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  const bool cooperative =
      state.device.is_gpu() &&
      (coop_env ? coop_env[0] != '0'
                : (!is_test && select_karlsen_cooperative(state, input_size, start_nonce)));
#endif
  std::memset(state.result, 0, sizeof(FishResult));

  sycl::event search;
  if (is_test && !full_test) {
    search = submit_karlsenhashv2_search<false>(state.queue, state.input, input_size, start_nonce,
                                                state.cache, nullptr, intensity, state.target,
                                                state.result, is_test);
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  } else if (cooperative) {
    search = submit_karlsenhashv2_search_coop(state.queue, state.input, input_size, start_nonce,
                                              state.dag, intensity, state.target, state.result,
                                              is_test, state.coop_workgroup);
#endif
  } else {
    search = submit_karlsenhashv2_search<true>(state.queue, state.input, input_size, start_nonce,
                                               nullptr, state.dag, intensity, state.target,
                                               state.result, is_test);
  }
  sycl_wait_and_throw(search, state.device);

  static const bool perf_log = std::getenv("MOM_KARLSEN_PERF") != nullptr;
  if (perf_log && !is_test) {
    static uint64_t acc = 0;
    static auto t0 = std::chrono::steady_clock::now();
    acc += intensity;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec >= 2.0) {
      std::fprintf(stderr, "[karlsenhashv2] %.2f MH/s (%s)\n", static_cast<double>(acc) / sec / 1e6,
                   dev_str.c_str());
      acc = 0;
      t0 = std::chrono::steady_clock::now();
    }
  }

  const uint32_t count = state.result->count;
  if (count == 0)
    return 0;
  const uint32_t idx = std::min(count, MAX_OUTPUTS) - 1;
  *pnonce = state.result->nonce[idx];
  std::memcpy(output, state.result->output[idx], HASH_LEN);
  if (mix_hash)
    std::memset(mix_hash, 0, HASH_LEN);
  return 1;
}
