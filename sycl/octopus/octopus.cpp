// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
// Portable SYCL implementation of Conflux's Octopus proof of work.
//
// The input is the 32-byte Keccak-256 problem hash followed by an 8-byte LE nonce,
// matching the etchash-style GPU ABI.  The consensus details intentionally mirror
// tests/reference/octopus_conflux.cpp. Portable tests use the light cache; explicit native-vector
// coverage requires the full DAG and rejects native fallback only on capability-positive devices.
//
// Performance/portability notes:
// - Matrix execution is capability-gated: Intel uses XMX/ESIMD, NVIDIA uses inline tensor-core
//   operations inside a SYCL kernel, and gfx12 AMD uses source-JIT WMMA. DPC++'s NVIDIA matrix API
//   lacks the efficient m16n8k32 shape, so that path uses inline PTX while retaining SYCL ownership
//   of memory, launches, and fallback. Unsupported devices retain the complete portable SYCL path.
// - NVIDIA fuses six field products before its four-lane DAG finish. Aligned 128-bit evict-first
//   loads fit its one-pass random DAG access pattern without retaining those pages in L1. The
//   remaining gap is not a proven hardware limit.
// - gfx12 AMD likewise fuses the six WMMA products into one normalized matrix. This reduces the two
//   pipeline slots' product storage from 192 MiB to 32 MiB and removes five intermediate writes and
//   rereads per value; on the RX 9060 XT it raised 29.08 MH/s to 31.22 MH/s versus 38.72 MH/s from
//   the local same-card reference. The remaining random-DAG gap is not a proven hardware limit.
// - B580 uses XMX for field products, then fuses Keccak and the random DAG mix in ESIMD. Four
//   independent hashes expose dependent page reads without intermediate seed/mix buffers, and
//   the field-specific modulus fold halves matrix-result traffic. Batches of 65536 groups amortize
//   submission work; two slots use 1.25 GiB, with allocation failure retaining the portable fallback.
//   The Intel 32x16 XMX tile reuses B across four row fragments; 256 GRFs prevent measured 128-GRF
//   spills. Bounded next-K input pipelining raised the matched Linux median from 30.42 to 31.27 MH/s;
//   the remaining best-peer gap is not a proven ceiling.
//   Width-8 devices use two XMX row fragments and two hashes per finish work-item. The smaller
//   finish state improved matched A770 throughput by 1.2%; its remaining gap is not a proven ceiling.

#include <sycl/sycl.hpp>
#if defined(OCTOPUS_ESIMD)
#include "../intel-dpas.h"
#if __has_include(<sycl/ext/intel/experimental/grf_size_properties.hpp>)
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#define MOM_OCTOPUS_INTEL_LARGE_GRF 1
#endif
#endif
#if defined(MOM_SYCL_HAS_HIP)
#include <hip/hip_runtime_api.h>
#include "../hiprtc-api.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../lib-internal.h"
#include "../nvidia-features.h"
#include "validation.h"
#include "../../native/consts.h"
#include "../../xmrig/3rdparty/libethash/ethash.h"

namespace mom_octopus {

#include "device.inc"

#include "search.inc"

#include "state.inc"

void octopus_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "octopus: ordered SYCL cleanup failed\n");
  }
}

} // namespace mom_octopus

using namespace mom_octopus;

// Validate work, serialize per-device resources, and dispatch one batch; only a found share updates outputs.
int octopus(unsigned, uint32_t height, const uint8_t* input, unsigned input_size, uint8_t* output,
            uint8_t* mix_hash, uint64_t* nonce, const uint8_t* target, const uint8_t*,
            unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev_str) {
  if (!input || !output || !nonce || input_size < HASH_LEN + sizeof(uint64_t) || !intensity)
    throw std::string("octopus requires a 32-byte problem hash, nonce, and nonzero intensity");
  if (!is_test && !target)
    throw std::string("octopus requires a target outside test mode");
  const uint64_t start_nonce = load64(input + HASH_LEN);
  if (static_cast<uint64_t>(intensity) - 1 > std::numeric_limits<uint64_t>::max() - start_nonce)
    throw std::string("octopus nonce range exceeds uint64 max");
  State& state = state_for(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_epoch(height / EPOCH_LENGTH, !is_benchmark, is_test);
  const bool header_changed = state.ensure_points(input);
  const bool target_changed =
      target && (!state.target_ready || std::memcmp(state.target_copy.data(), target, HASH_LEN));
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
  sycl::event input_copy_event;
  bool has_input_copy_event = false;
#endif
  try {
    MomSyclHostTransferGuard host_transfers(state.queue, "octopus host transfers");
    if (state.shared_io) {
      if (header_changed)
        std::memcpy(state.header, input, HASH_LEN);
      if (target_changed)
        std::memcpy(state.target, target, HASH_LEN);
    } else {
      if (header_changed) {
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
        input_copy_event = state.queue.memcpy(state.header, input, HASH_LEN);
        has_input_copy_event = true;
#else
        state.queue.memcpy(state.header, input, HASH_LEN);
#endif
      }
      if (target_changed) {
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
        input_copy_event = state.queue.memcpy(state.target, target, HASH_LEN);
        has_input_copy_event = true;
#else
        state.queue.memcpy(state.target, target, HASH_LEN);
#endif
      }
    }
    if (target_changed) {
      std::memcpy(state.target_copy.data(), target, HASH_LEN);
      state.target_ready = true;
    }
    std::memset(state.result, 0, sizeof(*state.result));
    const FastModData cache_mod = make_fast_mod_data(state.cache_nodes);
    const FastModData page_mod = make_fast_mod_data(dataset_size(state.epoch) / 256);
    // MOM_OCTOPUS_SCALAR keeps the original one-nonce/full-DAG kernel available
    // as a correctness oracle while the default path uses the cooperative kernel.
    const bool scalar = std::getenv("MOM_OCTOPUS_SCALAR") != nullptr;
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
    const auto subgroup_sizes = state.device.get_info<sycl::info::device::sub_group_sizes>();
    // Older HIP GPUs can use wave64; their fallback must not launch a wave32-only kernel.
    const bool subgroup_32 = mom_is_cuda(state.device) ||
        std::find(subgroup_sizes.begin(), subgroup_sizes.end(), 32u) != subgroup_sizes.end();
#endif
    bool searched = false;
    const bool prove_native = is_test && std::getenv("MOM_OCTOPUS_TEST_NATIVE");
    bool native_supported = false;
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
    // Qualify hardware before a failed native launch can invalidate queue/interop queries. JIT
    // availability is deliberately not part of this predicate: supported-device failures must fail.
    if (prove_native)
      native_supported = native_matrix_supported(state.queue);
    const bool sycl_native_device =
#ifdef OCTOPUS_ESIMD
        sycl_is_level_zero_gpu(state.device) && state.device.has(sycl::aspect::ext_intel_matrix);
#else
        mom_is_cuda(state.device) || mom_is_hip(state.device);
#endif
    if (sycl_native_requested() && state.full_dag && sycl_native_device && !scalar) {
      if (has_input_copy_event)
        sycl_wait_and_throw(input_copy_event, state.device);
      searched = state.sycl_native.search(state.queue, state.header, state.dag, page_mod,
                                          state.points, state.points_version, start_nonce, intensity,
                                          state.target, state.result, is_test);
      if (!searched)
        std::memset(state.result, 0, sizeof(*state.result));
    }
#endif
    const auto validation_failure = mom::octopus::test_path_failure(
        is_test && std::getenv("MOM_OCTOPUS_TEST_FULL_DAG"), state.full_dag,
        prove_native, native_supported, searched);
    if (validation_failure == mom::octopus::TestPathFailure::full_dag)
      throw std::string("Octopus full-DAG validation could not run");
    if (validation_failure == mom::octopus::TestPathFailure::native)
      throw std::string("Octopus SYCL-native full-DAG validation could not run");
    if (!searched && state.full_dag) {
      if (scalar) {
        sycl_wait_and_throw(search_scalar<true>(state.queue, state.header, state.cache, cache_mod,
                                                state.dag, page_mod, start_nonce, intensity,
                                                state.target, state.result, is_test),
                            state.device);
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
      } else if (!std::getenv("MOM_OCTOPUS_BATCHED") && subgroup_32) {
        sycl_wait_and_throw(search_cooperative(state.queue, state.header, state.dag, page_mod,
                                               state.points, start_nonce, intensity, state.target,
                                               state.result, is_test),
                            state.device);
#endif
      } else {
        sycl_wait_and_throw(search_batched<true>(state.queue, state.header, state.cache, cache_mod,
                                                 state.dag, page_mod, state.points, start_nonce,
                                                 intensity, state.target, state.result, is_test),
                            state.device);
      }
    } else if (!searched) {
      if (scalar) {
        sycl_wait_and_throw(search_scalar<false>(state.queue, state.header, state.cache, cache_mod,
                                                 nullptr, page_mod, start_nonce, intensity,
                                                 state.target, state.result, is_test),
                            state.device);
      } else {
        sycl_wait_and_throw(search_batched<false>(state.queue, state.header, state.cache, cache_mod,
                                                  nullptr, page_mod, state.points, start_nonce,
                                                  intensity, state.target, state.result, is_test),
                            state.device);
      }
    }
    if (!state.result->count)
      return 0;
    const uint32_t index = std::min(state.result->count, MAX_RESULTS) - 1;
    *nonce = state.result->nonce[index];
    std::memcpy(output, state.result->hash[index], HASH_LEN);
    if (mix_hash)
      std::memset(mix_hash, 0, HASH_LEN);
    return 1;
  } catch (...) {
    // A failed upload must not make a same-header retry skip its header or target copy.
    state.points_ready = state.target_ready = false;
    throw;
  }
}
