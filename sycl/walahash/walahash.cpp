// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
// WagLayla WalaHash: BLAKE3 -> SHA3-256 -> HeavyHash matrix -> BLAKE3.
//
// Production paths use Intel ESIMD/XMX, NVIDIA NVPTX DP4A with fixed-hash fusion at workgroup 128,
// and gfx12 AMD WMMA with split32 fixed hashes at workgroup 256. Replicated A/B measurements on an
// RTX 5060 Ti are 888.61 MH/s for fixed-fused DP4A versus 625.08 generic fused and 221--229 staged;
// an RX 9060 XT reaches 679.50 MH/s on Linux and 667.68 MH/s on Windows with packed-output split32
// WMMA, versus 448.37 generic and 497.35 partial64.
// The fixed-size BLAKE3 specialization captures the nonce-invariant first-block chaining value;
// split32 Keccak's lane-complement form reduces dependency-heavy chi operations on gfx12. Xe2
// instead reaches 628.30 MH/s with standard chi and sycl::rotate, versus 539.92 for lane-complement
// chi and 547.17 when the standard form uses the shared rotate wrapper.
// Explicit NVPTX DP4A and gfx12 WMMA match their compiler/hardware matrix instructions. Devices
// without a retained native path use the exact portable SYCL implementation.

#include <sycl/sycl.hpp>

#if defined(MOM_SYCL_HAS_HIP)
#include <hip/hip_runtime_api.h>
#include "../hiprtc-api.h"
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "../lib-internal.h"
#include "../nvidia-features.h"
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
#include "../nvidia-dot.h"
#endif
#include "../../native/job-boundary.h"
#if defined(WALAHASH_ESIMD)
#include "../intel-dpas.h"
#elif !defined(MOM_SYCL_ADAPTIVECPP)
#include <sycl/ext/oneapi/dot_product.hpp>
#endif
#include "../../native/consts.h"

namespace mom_walahash {

#include "device.inc"

#include "state.inc"

void walahash_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "walahash: ordered SYCL cleanup failed\n");
  }
}

#include "search.inc"

} // namespace mom_walahash

using namespace mom_walahash;
// Validate the fixed header, serialize device buffers, and search one nonce batch on the selected backend.
int walahash(unsigned, uint32_t, const uint8_t* input, unsigned input_size, uint8_t* output,
             uint8_t*, uint64_t* pnonce, const uint8_t* target, const uint8_t*, unsigned intensity,
             bool is_test, bool, const std::string& dev) {
  if (!input || !output || !pnonce || !target || input_size != 80 || !intensity)
    throw std::string("walahash requires an 80-byte header and nonzero intensity");
  if (!mom::job_boundary::padded_u32_range_fits(intensity, 256))
    throw std::string("WalaHash intensity is too large");
  State& state = state_for(dev);
  std::lock_guard<std::mutex> lock(state.mutex);
  MomSyclHostTransferGuard host_transfers(state.queue, "walahash host transfers");
  state.queue.memcpy(state.input, input, 72);
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA) || defined(MOM_SYCL_HAS_HIP)
  b3_first_block_cv(state.prehash_cv, input);
#endif
  state.queue.memcpy(state.target, target, 32);
  const char* configured = std::getenv("MOM_WALAHASH_SYCL_NATIVE");
  bool native = !configured || std::strcmp(configured, "0");
#ifdef MOM_SYCL_HAS_CUDA
  // Native=true selects DP4A even in the non-fixed-hash kernel. Gate both dispatches together.
  if (mom_is_cuda(state.device) && !mom::nvidia::has_dp4a(state.cuda_sm))
    native = false;
#endif
#ifdef WALAHASH_ESIMD
  const bool intel_fast = use_esimd(state.device);
  if (intel_fast)
    state.ensure(intensity);
#endif
  unsigned fused_work_group =
      sycl_default_workgroup(state.device, {32, 64, 128, 256}, 256);
#if defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  // Fixed hashes need only integer math; AdaptiveCpp independently gates DP4A at JIT target reflection.
  const char* cuda_required = std::getenv("MOM_WALAHASH_CUDA_REQUIRE");
  const bool cuda_require = cuda_required && *cuda_required && std::strcmp(cuda_required, "0");
  const bool cuda_device = state.device.is_gpu() && mom_is_cuda(state.device);
  const bool cuda_fast = native && cuda_device && sycl_workgroup_fits(state.device, 128);
  if (cuda_require && !cuda_fast)
    throw std::string("WalaHash CUDA fixed-hash path required but unavailable");
#endif
#ifdef MOM_SYCL_HAS_HIP
  const bool amd_fast =
      native && sycl_workgroup_fits(state.device, 256) && state.amd_wmma.ensure(state.queue);
  const char* amd_required = std::getenv("MOM_WALAHASH_AMD_WMMA_REQUIRE");
  if (amd_required && *amd_required && std::strcmp(amd_required, "0") && !amd_fast)
    throw std::string("WalaHash gfx12 WMMA required but unavailable");
  if (amd_fast)
    state.ensure(intensity);
#endif
  if (!state.matrix_ready || std::memcmp(state.matrix_seed, input, 32)) {
    state.matrix_ready = false;
    uint8_t matrix[4096];
    make_matrix(input, matrix);
#ifdef WALAHASH_ESIMD
    if (intel_fast) {
      uint8_t tiled[4096];
      for (unsigned r = 0; r < 64; ++r)
        for (unsigned k = 0; k < 64; ++k)
          tiled[((r >> 3) * 2 + (k >> 5)) * 256 + (r & 7) * 32 + (k & 31)] = matrix[r * 64 + k];
      sycl_wait_and_throw(state.queue.memcpy(state.matrix, tiled, sizeof(tiled)), state.device);
    } else
#endif
      sycl_wait_and_throw(state.queue.memcpy(state.matrix, matrix, sizeof(matrix)), state.device);
    std::memcpy(state.matrix_seed, input, 32);
    state.matrix_ready = true;
  }
  state.queue.memset(state.result, 0, sizeof(Result));
  sycl::event search_event;
  bool dispatched = false;
#ifdef WALAHASH_ESIMD
  if (intel_fast) {
    const unsigned width = mom::intel_matrix::dpas_width(state.device);
    prehash(state, *pnonce, intensity);
    if (width == 8)
      multiply_esimd<8>(state, intensity);
    else
      multiply_esimd<16>(state, intensity);
    search_event = finalize(state, *pnonce, intensity, is_test);
    dispatched = true;
  }
#endif
#if defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  if (!dispatched && cuda_fast) {
    search_event = search<true, true>(state, *pnonce, intensity, is_test, 128);
    dispatched = true;
  }
#endif
#ifdef MOM_SYCL_HAS_HIP
  if (!dispatched && amd_fast) {
    // WMMA consumes complete 64-sample tiles. Small correctness jobs still finalize only their real
    // item count, but prehash and mix a padded tile just as the original serial path did.
    const unsigned matrix_count = state.capacity;
    prehash_amd(state, *pnonce, matrix_count);
    walahash_amd_mix(
        state.queue, state.amd_wmma, reinterpret_cast<int8_t*>(state.matrix), state.hashes,
        state.mixed, matrix_count);
    search_event = finalize_amd(state, *pnonce, intensity, is_test);
    dispatched = true;
  }
#endif
  if (!dispatched && native) {
    search_event = search<true, false>(state, *pnonce, intensity, is_test, fused_work_group);
    dispatched = true;
  }
  if (!dispatched)
    search_event = search<false, false>(state, *pnonce, intensity, is_test, fused_work_group);
  // Some runtimes busy-spin while submitting a host copy behind live kernels. Complete the GPU
  // search with the shared low-CPU wait before enqueuing the already-ready result copy.
  sycl_wait_and_throw(search_event, state.device);
  Result result;
  sycl_wait_and_throw(state.queue.memcpy(&result, state.result, sizeof(result)), state.device);
  if (!result.count)
    return 0;
  *pnonce = result.nonce;
  std::memcpy(output, result.hash, 32);
  return 1;
}
