// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>
//
// PearlHash (PRL) NoisyGEMM proof-of-useful-work GPU search kernel (SYCL / Level-Zero).
// Per seed: counter-RNG A/B -> keyed-BLAKE3 commitment roots -> sparse low-rank noise -> noised
// int8 A'*B' XMX GEMM tiles -> per-16x16-tile XOR/rotl13 transcript -> keyed-BLAKE3 jackpot vs
// target. Current certificate V3 binds each raw Merkle root to its matrix dimension before the
// shared salted-noise chain and requires rank >= 128; older certificates are intentionally rejected.
// On a win the host builds the pool PlainProof (with raw Merkle openings) directly.
//
// A/B are never materialized (they are regenerated from the counter-RNG inside each consumer),
// the commitment Merkle tree is reduced in parallel, and the search uses a PEARLHASH_HR x
// PEARLHASH_NTILE register tile with software-pipelined B loads.

#include <sycl/sycl.hpp>
#include "../lib-internal.h"
#ifndef MOM_PEARLHASH_ESIMD_TU
// The standalone Intel ESIMD image has no CUDA host dispatch or toolkit include path.
#include "../nvidia-features.h"
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
#include "../nvidia-dot.h"
#endif
#endif
#include "../../native/job-boundary.h"
#include "../pow-intensity.h"
#include "esimd_route.h"
#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__) && !defined(MOM_PEARLHASH_ESIMD_TU)
#include "../cuda-api.h"
#endif
#if defined(MOM_SYCL_HAS_HIP)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime_api.h>
#include "../hiprtc-api.h"
#endif
#ifdef PEARLHASH_ESIMD
#include "../intel-dpas.h"
#if __has_include(<sycl/ext/intel/experimental/grf_size_properties.hpp>) && \
    __has_include(<sycl/ext/oneapi/experimental/enqueue_functions.hpp>)
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#define MOM_PEARLHASH_INTEL_LARGE_GRF 1
#endif
#endif
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_HAS_CUDA)
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif
#if !defined(__SYCL_DEVICE_ONLY__)
#include "../jit-cache.h"
#endif

#include "blake3.inc"
#include "seed.inc"

namespace mom_pearlhash {

#include "matrix.inc"

#include "sycl_search.inc"

// NVIDIA pearlhash throughput lives entirely in the int8 Tensor Cores, reached here through the
// mma.sync path below. It is built with the intel/llvm DPC++ CUDA backend (nvptx64) -- the SAME
// DPC++ as the Intel build -- so every SYCL kernel stays unified across both GPU vendors, and the
// runtime kernel_compiler JIT that kawpow's per-period specialization relies on is available too.
#include "cuda_sycl_search.inc"

#include "esimd_search.inc"
#include "dpasw_search.inc"

#ifdef MOM_PEARLHASH_ESIMD_TU
} // namespace mom_pearlhash
#else
// PearlHash's shared SYCL CUDA kernel already uses cp.async, ldmatrix, and mma.sync, but the SYCL
// execution model cannot express CUTE's staged CTA mainloop without manually duplicating a large
// part of CUTE's layout machinery. That path plateaus near 55.5 TH/s on RTX 5060 Ti; compiling the
// architecture-neutral source below with NVRTC reaches 71.5 TH/s at the same 150 W. The override is
// deliberately optional: no device binary is shipped, and any missing compiler/header/runtime or
// unsupported GPU falls back to the normal selectable SYCL implementation.
#include "cuda_jit.inc"

#include "hip_jit.inc"

#include "dispatch.inc"

} // namespace mom_pearlhash

// The separate ESIMD translation unit exports kernels only; host state and entrypoints live here.
#include "claim.h"
using namespace mom_pearlhash;

#include "host.inc"
#include "state.inc"

// Release device owners in runtime-safe order, retaining only the documented Windows CPU ICD exception.
void pearlhash_cleanup_states() noexcept {
  try {
#if defined(_WIN32)
    // Intel's standalone Windows CPU OpenCL ICD can fault after a successful PearlHash result while
    // DPC++ destroys this algorithm's shared-USM allocation/queue graph. The state registry itself
    // is deliberately heap-owned and has no process-exit destructor, so retain only CPU states and
    // let process teardown reclaim them. Discrete GPU states still require ordered cleanup (notably
    // AdaptiveCpp/CUDA, which otherwise reports cudaErrorCudartUnloading) and are erased normally.
    pearlhash_registry().clear([](const auto& entry) {
      return !entry.second->queue.get_device().is_cpu();
    });
#else
    pearlhash_registry().clear();
#endif
  } catch (...) {
    std::fprintf(stderr, "pearlhash: ordered SYCL cleanup failed\n");
  }
}

// One seeded NoisyGEMM attempt per call (seed = *pseed). Returns 1 on a winning tile, writes the
// winning seed back to *pseed, and captures its tile for lazy pearlhash_claim(). intensity = M.
int pearlhash(const unsigned job_ref, const uint32_t, const uint8_t* const input,
              const unsigned input_size, uint8_t* const, uint64_t* const pseed,
              const uint8_t* const target, const unsigned intensity, const bool is_test, const bool,
              const std::string& dev_str, const std::string& backend, const unsigned requested_n,
              const unsigned requested_k, const unsigned requested_rank,
              const unsigned cert_version) {
  if (!input || !pseed || input_size < 76)
    throw std::string("Bad pearlhash input length (need 76-byte header)");
  if (!is_test && !target)
    throw std::string("PearlHash target is required outside test mode");
  constexpr unsigned max_dimension =
      static_cast<unsigned>(std::numeric_limits<int>::max());
  if (intensity > max_dimension || requested_n > max_dimension || requested_k > max_dimension ||
      requested_rank > max_dimension)
    throw std::string("PearlHash dimensions exceed 32-bit kernel indexing");
  const int k = static_cast<int>(requested_k);
  const int rank = static_cast<int>(requested_rank);
  if (cert_version != 3)
    throw std::string("Unsupported pearlhash certificate version");
  if (rank < 128)
    throw std::string("PearlHash certificate V3 requires rank >= 128");
  // M arrives as the worker's explicit primary tuning value; N can be selected independently. Native HIP
  // tests use the smallest square that activates the 32x32 cache-block traversal, so the
  // native-vs-SYCL checksum comparison covers production tile mapping without making CPU tests
  // expensive. Other implementations retain the compact 256x256 vector.
  const int test_edge = backend == "native" ? 2048 : 256;
  const int m = is_test ? test_edge
                        : (intensity >= 128 ? static_cast<int>(intensity) : 131072);
  const int n = is_test ? m : static_cast<int>(requested_n);
  if (!mom::job_boundary::valid_pearlhash_shape(
          static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k),
          static_cast<uint64_t>(rank)))
    throw std::string("Invalid pearlhash M/N/K/rank profile");
  (void)pearlhash_attempt_hashes(intensity, requested_n, requested_k, requested_rank);

  g_pf.valid = false;
  g_pf.claim_ready = false;
  PearlHashState& st = pearlhash_state(dev_str);
  std::lock_guard<std::mutex> lock(st.mutex);
  const bool native_requested = backend == "auto" || backend == "native";
  std::string hip_reason, cuda_reason;
  const bool hip_device = native_requested && PearlHashHipSearch::supports(st.queue, &hip_reason);
  const bool cuda_device = native_requested && PearlHashCudaSearch::supports(st.queue, &cuda_reason);
  const bool hip_jit = hip_device && PearlHashHipSearch::supports_shape(m, n, k, rank);
  const bool cuda_jit = cuda_device && PearlHashCudaSearch::supports_shape(m, n, k, rank);
  if (hip_device && !hip_jit)
    hip_reason = "the selected PearlHash profile requires the shared SYCL path";
  if (cuda_device && !cuda_jit)
    cuda_reason = "the selected PearlHash profile requires the shared SYCL path";
  if (native_requested && !hip_jit && !cuda_jit && !st.native_notice) {
    st.native_notice = true;
    std::fprintf(
        stderr, "PearlHash native backend unavailable, using SYCL: %s\n",
        (PearlHashCudaSearch::is_cuda(st.queue.get_device()) ? cuda_reason : hip_reason).c_str());
  }
  st.ensure(m, n, k, rank, hip_jit, cuda_jit);
  sycl::queue& q = st.queue;
  Buffers& b = st.b;

  uint8_t tgtLE[32];
  const uint32_t attempt_seed = static_cast<uint32_t>(*pseed);
  PearlHashSearchBackend search_backend;
  std::chrono::steady_clock::time_point attempt_start;
  try {
    if (!st.have_header ||
        std::memcmp(st.header, input, 76) != 0) { // (re)derive key on header change
      st.have_header = false;
      std::memcpy(st.header, input, 76);
      derive_key(input, k, rank, st.key);
      q.memcpy(b.key, st.key, 32);
    }
    (void)job_ref;
    if (is_test)
      std::memset(tgtLE, 0xff, 32); // test: the first tile wins
    else
      for (int i = 0; i < 32; i++)
        tgtLE[i] = target[31 - i]; // core gives BE; kernel wants LE
    q.memcpy(b.target, tgtLE, 32);
    b.result->found = 0;
    b.result->chk =
        0; // whole-search checksum (filled only when MOM_PEARLHASH_CHK is set; see tile_mix)

    search_backend =
        attempt(q, b, attempt_seed, m, n, k, rank, backend, &st.hip, &st.cuda,
                &st.prep_queue, is_test, st.cuda_tensor, st.cuda_f16);
    // Start after submission: a first call can include synchronous device-image/JIT setup, and feeding
    // that one-off cost into the wait EMA makes later completed GPU attempts sleep needlessly.
    attempt_start = std::chrono::steady_clock::now();
    // Intel SYCL waits must drive queue progress before updating the attempt-time EMA;
    // sleeping first can feed the delay back into subsequent attempts. Keep other backends' pacing.
    const bool intel_sycl_gpu = search_backend == PearlHashSearchBackend::sycl &&
        q.get_device().is_gpu() &&
        q.get_device().get_info<sycl::info::device::vendor_id>() == 0x8086;
    if (st.wait_ema_us > 2000.0 && !intel_sycl_gpu) {
      const double elapsed_us =
          static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - attempt_start)
                                  .count());
      const double sleep_us = st.wait_ema_us * 0.90 - elapsed_us;
      if (sleep_us > 0.0)
        std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(sleep_us)));
    }
    if (search_backend == PearlHashSearchBackend::hip_jit ||
        search_backend == PearlHashSearchBackend::cuda_jit) {
      if (search_backend == PearlHashSearchBackend::hip_jit)
        st.hip.wait(st.wait_ema_us);
      else
        st.cuda.wait(st.wait_ema_us);
    } else {
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER)
      // OpenCL keeps its native wait/progress semantics; other Intel GPU queues poll completion
      // before sleeping, without reintroducing EMA pre-wait delays.
      if (intel_sycl_gpu && !mom_is_opencl(q.get_device()))
        sycl_wait_and_throw(q.ext_oneapi_submit_barrier(), q.get_device());
      else
#endif
        q.wait_and_throw();
    }
    st.have_header = true;
  } catch (...) {
    // Keep the staged target alive until any accepted copy has retired.
    sycl_cleanup_noexcept("pearlhash target upload", [&] { q.wait_and_throw(); });
    throw;
  }
  if (is_test && q.get_device().is_gpu()) {
    // Search backends share GPU preparation, so agreement between them cannot validate the
    // commitment roots or V3 noise seeds. Recompute these independently on the host in tests.
    const std::size_t chunks_a = (static_cast<std::size_t>(m) * k + 1023) / 1024;
    const std::size_t chunks_b = (static_cast<std::size_t>(n) * k + 1023) / 1024;
    std::vector<uint8_t> cvs_a(chunks_a * 32), cvs_b(chunks_b * 32);
    std::array<uint8_t, 32> expected_a{}, expected_b{}, observed_a{}, observed_b{};
    Buffers reference{};
    reference.key = st.key;
    reference.CVA = cvs_a.data();
    reference.CVB = cvs_b.data();
    reference.cA = expected_a.data();
    reference.cB = expected_b.data();
    prepare_roots_cpu(reference, attempt_seed, m, n, k);
    try {
      q.memcpy(observed_a.data(), b.cA, observed_a.size());
      q.memcpy(observed_b.data(), b.cB, observed_b.size());
      q.wait_and_throw();
    } catch (...) {
      sycl_cleanup_noexcept("pearlhash commitment readback", [&] { q.wait_and_throw(); });
      throw;
    }
    if (observed_a != expected_a || observed_b != expected_b)
      throw std::string("PearlHash GPU/host commitment seed mismatch");
    const char* const search_name = search_backend == PearlHashSearchBackend::cuda_jit
      ? "cuda-jit" : search_backend == PearlHashSearchBackend::hip_jit ? "hip-jit" : "sycl";
    std::fprintf(stderr, "PEARLHASH_TEST search=%s cert_version=%u host_seeds_match=true\n",
                 search_name, cert_version);
    // Automatic capability fallback may resolve to the reference, not a tuned search.
    if (backend == "sycl")
      std::fprintf(stderr, "PEARLHASH_TEST reference_search=true\n");
  }
  // search() retains the scalar MAC expression, independent of the optional NVIDIA packed-dot path.
  // Explicit "sycl" already uses this reference; tuned searches must compare against it.
  // Keep this outside vendor guards so Intel ESIMD receives the same correctness check.
  if (is_test && q.get_device().is_gpu() && backend != "sycl") {
    const uint32_t accelerated_checksum = b.result->chk;
    b.result->found = 0;
    b.result->chk = 0;
    compute_ab(q, b, attempt_seed, m, n, k, rank, /*portable=*/true);
    search(q, b, attempt_seed, m, n, k, rank, /*dbg=*/true);
    q.wait_and_throw();
    if (b.result->chk != accelerated_checksum) {
      std::ostringstream message;
      message << "PearlHash accelerated/SYCL checksum mismatch (accelerated=0x" << std::hex
              << accelerated_checksum << ", SYCL=0x" << b.result->chk << ')';
      throw message.str();
    }
    std::fprintf(stderr, "PEARLHASH_TEST search_checksum_match=true\n");
  }
  {
    const double us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                              std::chrono::steady_clock::now() - attempt_start)
                                              .count());
    st.wait_ema_us = st.wait_ema_us == 0.0 ? us : st.wait_ema_us * 0.8 + us * 0.2;
  }
  if (std::getenv(
          "MOM_PEARLHASH_CHK")) // cross-validation: same A'/B' -> identical chk on every search path
    std::fprintf(stderr, "PEARLHASH_CHK dev=%s seed=%u chk=%08x\n", dev_str.c_str(),
                 (uint32_t)*pseed, b.result->chk);
  if (!b.result->found)
    return 0;
  *pseed = b.result->seed;
  if (is_test)
    return 1; // test: core only needs "ok", no proof
  // Capture the winning tile; rebuild its expensive PlainProof only when the core requests a claim.
  // Distinct winning seeds under one job are separate shares; retries reuse the captured claim.
  g_pf.seed = b.result->seed;
  g_pf.row = (int)b.result->row;
  g_pf.col = (int)b.result->col;
  g_pf.m = m;
  g_pf.n = n;
  g_pf.k = k;
  g_pf.rank = rank;
  std::memcpy(g_pf.key, st.key, 32);
  std::memcpy(g_pf.jackpot, b.result->jackpot, 32);
  g_pf.claim_ready = false;
  g_pf.valid = true;
  return 1;
}
// Builds and caches the PlainProof plus its matching trusted-share metadata for the last winning tile.
// Repeated calls for the captured tile return the same claim.
const char* pearlhash_claim(uint8_t jackpot[32], uint32_t* adjustment_factor) {
  if (!g_pf.valid || !jackpot || !adjustment_factor)
    return nullptr;
  if (!g_pf.claim_ready) {
    g_pearlhash_proof = build_plain_proof(g_pf.seed, g_pf.m, g_pf.n, g_pf.k, g_pf.rank, g_pf.key,
                                          g_pf.row, g_pf.col, &g_pf.adjustment_factor);
    if (g_pearlhash_proof.empty())
      return nullptr;
    g_pf.claim_ready = true;
  }
  std::memcpy(jackpot, g_pf.jackpot, 32);
  *adjustment_factor = g_pf.adjustment_factor;
  return g_pearlhash_proof.c_str();
}
// Rank-128-equivalent GEMM MACs per attempt. V3's target is normalized to rank 128,
// so this is also the stable accounting unit advertised to switching pools.
uint64_t pearlhash_attempt_hashes(unsigned intensity, unsigned n, unsigned k, unsigned rank) {
  const uint64_t m = intensity >= 128 ? intensity : 131072;
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  if (!mom::job_boundary::valid_pearlhash_shape(m, n, k, rank))
    throw std::string("Invalid PearlHash work factor");
  if (n && m > maximum / n)
    throw std::string("PearlHash work count overflows");
  const uint64_t mn = m * n;
  const uint64_t effective_k = static_cast<uint64_t>(k / rank) * 128;
  if (effective_k && mn > maximum / effective_k)
    throw std::string("PearlHash work count overflows");
  return mn * effective_k;
}

#endif
