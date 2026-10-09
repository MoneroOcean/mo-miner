// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
// XelisHashV3 GPU search. Algorithm follows the MIT-licensed XELIS and TNN implementations.
//
// Performance/portability notes:
// - Stage 3 accounts for roughly 97% of measured B580 time. Each nonce follows a serial chain of
//   dependent random scratchpad reads and branch-heavy integer operations, so ordinary loop unrolling
//   cannot expose much memory-level parallelism.
// - Intel Level Zero uses a capability-gated ESIMD kernel with explicit scalar gathers and exact
//   replacements for operations unavailable in ESIMD. CPU, older iGPU, OpenCL, NVIDIA, AMD, and unsupported
//   Intel paths retain the ordinary portable SYCL kernel.
// - FP64 only accelerates exact integer division. Devices without it use integer quotient
//   estimates in the same kernel; no floating-point emulation or consensus changes are needed.
// - Cache-only hints, next-read prefetch, block load/store, subgroup-width changes, and a copied
//   NVIDIA-style software pipeline were neutral or slower in matched tests. That narrows the useful
//   direction toward overlapping independent nonces; it does not establish a hardware ceiling.
// - Replacing B580's 64-bit division helpers with exact reciprocal-based 32-bit operations removed
//   those helpers but raised Stage 3 from 5,796 to 6,579 instructions and left 2.16 KH/s unchanged.
// - Complete each per-job setup transfer before Stage 1. Windows Level Zero also needs explicit
//   ESIMD stage barriers; relying only on its in-order queue produced noncanonical results.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

#ifdef MOM_XELISHASHV3_HOST_TEST
#include <cmath>
namespace sycl {
inline int clz(uint64_t x) {
  return x ? __builtin_clzll(x) : 64;
}
inline uint64_t mul_hi(uint64_t a, uint64_t b) {
  return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) >> 64);
}
inline float fma(float a, float b, float c) {
  return std::fma(a, b, c);
}
inline double fma(double a, double b, double c) {
  return std::fma(a, b, c);
}
inline float rsqrt(float x) {
  return 1.0f / std::sqrt(x);
}
} // namespace sycl
inline uint64_t mo_rotate(uint64_t x, uint64_t n) {
  const uint64_t shift = n & 63;
  return (x << shift) | (x >> ((64 - shift) & 63));
}
#else
#include <sycl/sycl.hpp>
#if defined(PEARLHASH_ESIMD) && !defined(MOM_SYCL_ADAPTIVECPP)
// The build sets PEARLHASH_ESIMD only for its spir64-only oneAPI worker. Reuse that compilation
// capability marker here so combined spir64/NVPTX and portable OpenCL workers keep ordinary SYCL.
#define MOM_XELISHASHV3_ESIMD
#include <sycl/ext/intel/esimd.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>

#include "../lib-internal.h"
#endif

namespace mom_xelishashv3 {

#include "device.inc"

#ifndef MOM_XELISHASHV3_HOST_TEST
#include "stage.inc"

#include "state.inc"

void xelishashv3_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "xelishashv3: ordered SYCL cleanup failed\n");
  }
}

#include "search.inc"

} // namespace mom_xelishashv3

using namespace mom_xelishashv3;
// Validate the fixed header, resize per-device scratch, and search a serialized nonce batch.
int xelishashv3(unsigned, uint32_t, const uint8_t* input, unsigned input_size, uint8_t* output,
                uint8_t*, uint64_t* nonce, const uint8_t* target, const uint8_t*,
                unsigned intensity, bool is_test, bool, const std::string& dev) try {
  if (!input || !output || !nonce || !target || input_size != 112 || !intensity)
    throw std::string(
      "xelishashv3 requires input, output, nonce, target, a 112-byte header, and nonzero intensity");
  State& s = state_for(dev);
  std::lock_guard<std::mutex> lock(s.mutex);
  s.ensure(intensity);
  sycl_wait_and_throw(s.queue.memcpy(s.input, input, 112), s.device);
  sycl_wait_and_throw(s.queue.memcpy(s.target, target, 32), s.device);
  sycl_wait_and_throw(s.queue.memset(s.result, 0, sizeof(Result)), s.device);
  const char* configured = std::getenv("MOM_XELISHASHV3_SYCL_NATIVE");
  const bool native = !configured || std::strcmp(configured, "0");
#ifdef MOM_XELISHASHV3_ESIMD
  // The cached ESIMD gathers require newer Intel hardware; XeLP lacks those LSC instructions.
  // The matrix capability keeps these older devices on the ordinary kernel without a SKU list.
  const bool use_esimd = native && sycl_is_level_zero_gpu(s.device) &&
                         s.device.has(sycl::aspect::ext_intel_matrix);
#else
  const bool use_esimd = false;
#endif
  if (native)
    search<true>(s, *nonce, intensity, is_test, use_esimd);
  else
    search<false>(s, *nonce, intensity, is_test);
  Result found;
  sycl_wait_and_throw(s.queue.memcpy(&found, s.result, sizeof(found)), s.device);
  if (!found.count)
    return 0;
  *nonce = found.nonce;
  std::memcpy(output, found.hash, 32);
  return 1;
} catch (const sycl::exception& error) {
  std::fprintf(stderr, "XelisHashV3 SYCL error: %s\n", error.what());
  throw;
}
#else
} // namespace mom_xelishashv3
#endif
