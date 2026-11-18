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
#include "esimd-route.h"
#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__) && !defined(MOM_PEARLHASH_ESIMD_TU)
#include "../cuda-api.h"
#if defined(MOM_EMBEDDED_NVRTC_HEADERS) && !defined(MOM_EMBEDDED_CUDA_IMAGES)
#include "../../native/jit-headers.h"
#include "nvrtc-headers.inc"
#endif
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
#if !defined(MOM_EMBEDDED_CUDA_IMAGES) || defined(MOM_SYCL_HAS_HIP)
#include <filesystem>
#endif
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
#if !defined(__SYCL_DEVICE_ONLY__) && \
    (!defined(MOM_EMBEDDED_CUDA_IMAGES) || defined(MOM_SYCL_HAS_HIP))
#include "../jit-cache.h"
#endif

#include "../blake3-device.inc"
#include "blake3.inc"
#include "seed.inc"

namespace mom_pearlhash {

#include "matrix.inc"

#include "sycl-search.inc"

// NVIDIA pearlhash throughput lives entirely in the int8 Tensor Cores, reached here through the
// mma.sync path below. It is built with the intel/llvm DPC++ CUDA backend (nvptx64) -- the SAME
// DPC++ as the Intel build -- so every SYCL kernel stays unified across both GPU vendors, and the
// runtime kernel_compiler JIT that kawpow's per-period specialization relies on is available too.
#include "cuda-sycl-search.inc"

#include "esimd-search.inc"
#include "dpasw-search.inc"

#ifdef MOM_PEARLHASH_ESIMD_TU
} // namespace mom_pearlhash
#else
// Optional source-JIT kernels express native layouts through CuTe while SYCL owns preparation
// and device state. Missing tools or unsupported hardware retain the shared SYCL path; an error
// after native submission must propagate instead of retrying on a potentially faulted queue.
namespace compact_private {
static bool selected(int m,int n,int k,int rank);
}
#include "cuda-jit.inc"

#include "hip-jit.inc"

#include "dispatch.inc"

} // namespace mom_pearlhash

namespace mom_pearlhash {
// Keep unqualified Windows/compiler profiles on their existing discovery shape.
// Optional dependency/property failures must not hide an otherwise usable GPU.
bool compact_available(const sycl::device& device) noexcept {
#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__) && !defined(_WIN32) && \
    (!defined(MOM_SYCL_ADAPTIVECPP) || defined(MOM_EMBEDDED_NVRTC_HEADERS) || \
     defined(MOM_EMBEDDED_CUDA_IMAGES))
  try {
    return PearlHashCudaSearch::compact_available(device);
  } catch (...) {
    return false;
  }
#else
  (void)device;
  return false;
#endif
}
} // namespace mom_pearlhash

// The separate ESIMD translation unit exports kernels only; host state and entrypoints live here.
#include "claim.h"
using namespace mom_pearlhash;

#include "host.inc"
#include "compact-policy.inc"
namespace mom_pearlhash::compact_private {
static bool selected(int m,int n,int k,int rank) {
  // Shape selects the candidate; CUDA hardware/compiler/resource admission is
  // still mandatory before its different proof pattern can become effective.
  return CompactProfile::supports(m,n,k,rank);
}
}
#include "state.inc"

namespace {
// Route admission owns backend/capability/shape checks; SYCL vendor IDs are not consistently PCI
// IDs (AdaptiveCpp HIP reports 1022). Keep the older F16/F32 route on reference bytes:
// positive cumulative dots can exceed its exact-integer range, even while fitting signed int32.
static bool pearlhash_positive_inputs(const sycl::device& device, const bool native_int8_route,
                                      const int k) {
  constexpr uint64_t max_positive_operand = 127;
  const bool int32_dot_bound = k > 0 && static_cast<uint64_t>(k) * max_positive_operand *
      max_positive_operand <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  return device.is_gpu() && native_int8_route && int32_dot_bound;
}
} // namespace

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
// winning seed back to *pseed, and captures its immutable claim. intensity = M.
int pearlhash(const unsigned job_ref, const uint32_t, const uint8_t* const input,
              const unsigned input_size, uint8_t* const, uint64_t* const pseed,
              const uint8_t* const target, const unsigned intensity, const bool is_test, const bool,
              const std::string& dev_str, const std::string& backend, const unsigned requested_n,
              const unsigned requested_k, const unsigned requested_rank,
              const unsigned cert_version, const uint8_t* const base_target) {
  // Even a rejected next call must not leave the previous seed's claim available.
  g_pf.valid = false;
  g_pf.claim_ready = false;
  g_pf.leaves_a.clear();
  g_pf.leaves_b.clear();
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
  // tests use a compact square whose first row slice activates the native cache traversal, so the
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

  // Device state is acquired only after all input/profile validation. Once the state is valid, clear
  // the prior fixed-B snapshot before any route, allocation, producer, or proof operation can throw.
  State& st = state_for(dev_str);
  std::lock_guard<std::mutex> lock(st.mutex);
  const bool prior_fixed_b_valid = st.fixed_b_valid;
  const uint32_t prior_fixed_b_seed = st.fixed_b_seed;
  st.fixed_b_valid = false;

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
  // Preflight only the already-admitted native route. This makes a failed JIT fall back with
  // reference bytes, while the mode remains a scalar in Buffers for every producer/consumer.
  bool native_int8_route = false;
  bool cuda_native_int8_route = false;
#if defined(MOM_SYCL_HAS_HIP)
  if (hip_jit)
    native_int8_route = st.hip.ensure(q, b, m, n, k, rank);
#endif
#if defined(MOM_SYCL_HAS_CUDA)
  if (!native_int8_route && cuda_jit)
    cuda_native_int8_route = st.cuda.ensure(q, b, m, n, k, rank);
#endif
  native_int8_route = native_int8_route || cuda_native_int8_route;
  // The pair kernel is the rank-128 native CUDA candidate. A JIT decline leaves
  // this false before roots/noise/key derivation, so the existing contiguous
  // path remains the complete fallback with its original protocol bytes.
  const bool compact_mode = cuda_native_int8_route && st.cuda.compact;
  // A declined compiler must leave the portable key, target and proof geometry
  // intact. Never carry the compact pattern across an unavailable JIT route.
  b.proof_rows = compact_mode ? CompactProfile::rows : hip_jit ? 4 : 16;
  b.proof_cols = compact_mode ? CompactProfile::cols : hip_jit ? 64 : 16;
  const bool pair_dist8 = cuda_native_int8_route && rank == 128 && (k == 8192 || compact_mode);
  if (st.have_header && st.header_pair_dist8 != pair_dist8)
    st.have_header = false;
  bool reuse_fixed_b = false;
#if defined(MOM_SYCL_HAS_CUDA)
  const bool tuned_sycl = backend == "auto" || backend == "sycl-native" || backend == "native";
  const bool cuda_tensor_route = tuned_sycl && st.cuda_tensor &&
      PearlHashCudaSearch::is_cuda(q.get_device()) && cuda_sycl_search_supports_shape(m, n);
#else
  const bool cuda_tensor_route = false;
#endif
  b.positive_inputs = pearlhash_positive_inputs(
      q.get_device(), native_int8_route || cuda_tensor_route, k);
  // Keep the compact seed contract out of AMD, legacy CUDA, and fallback jobs.
  b.seed_prefix_inputs = compact_mode && b.positive_inputs;

  uint8_t tgtLE[32];
  const uint32_t attempt_seed = static_cast<uint32_t>(*pseed);
  PearlHashSearchBackend search_backend;
  std::chrono::steady_clock::time_point attempt_start;
  try {
    // The CUDA rank-128 U8 MMA is valid only for the exact positive producer contract.
    // Keep this inside the existing attempt try/catch so cleanup and fallback lifecycle remain
    // unchanged if a future route/capability refactor ever violates the invariant.
    if (pair_dist8 && !b.positive_inputs)
      throw std::string("PearlHash CUDA pair requires positive inputs in [0,127]");
    if (!st.have_header ||
        std::memcmp(st.header, input, 76) != 0) { // (re)derive key on header change
      st.have_header = false;
      std::memcpy(st.header, input, 76);
      if (compact_mode)
        CompactProfile::derive_key(input,st.key);
      else if (pair_dist8)
        derive_key_pair_dist8(input, k, rank, st.key);
      else
        derive_key(input, k, rank, st.key, b.proof_rows, b.proof_cols);
      st.header_pair_dist8 = pair_dist8;
      q.memcpy(b.key, st.key, 32);
    }
    reuse_fixed_b = pair_dist8 && prior_fixed_b_valid && st.have_header &&
        st.header_pair_dist8 && std::memcmp(st.header, input, 76) == 0;
    (void)job_ref;
    if (is_test)
      std::memset(tgtLE, 0xff, 32); // test: all tiles are admissible; scheduling selects the winner
    else if (!pearlhash_search_target_le(
                 target, base_target, b.proof_rows, b.proof_cols, k, rank, tgtLE))
      throw std::string("Invalid PearlHash search target for the selected proof pattern");
    q.memcpy(b.target, tgtLE, 32);
    b.result->found = 0;
    b.result->chk =
        0; // whole-search checksum (filled only when MOM_PEARLHASH_CHK is set; see tile_mix)
    // The producer needs the selected B seed before it submits roots/noise/packing.  A cache hit
    // keeps the first completed pair attempt's seed; a miss uses this attempt's seed and commits
    // it only after the full completion/validation path below succeeds.
    b.fixed_b_seed = reuse_fixed_b ? prior_fixed_b_seed : attempt_seed;

    sycl::event compact_roots_ready;
    if (compact_mode && reuse_fixed_b) {
      std::array<uint8_t,32> key{};
      std::copy(st.key,st.key+32,key.begin());
      st.compact_roots.update(key,m,n,k,rank,attempt_seed);
      pearlhash_commitment_seed_a(st.compact_b.data(),st.compact_roots.root.data(),
                                 static_cast<uint32_t>(m),st.compact_a.data());
      // Sources are state-owned until both queues retire, including exceptional unwind.
      q.memcpy(b.CVA,st.compact_roots.first_leaf.data(),32);
      compact_roots_ready=q.memcpy(b.cA,st.compact_a.data(),32);
    }
    search_backend =
        attempt(q, b, attempt_seed, m, n, k, rank, backend, &st.hip, &st.cuda,
        hip_jit ? &st.prep_queue : nullptr, is_test, st.cuda_tensor, st.cuda_f16, pair_dist8,
        reuse_fixed_b, compact_mode && reuse_fixed_b ? &compact_roots_ready : nullptr);
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
    // Retire both producer queues before retrying with their shared buffers. Attempt each drain
    // independently so a main-queue fault cannot leave B preparation running after this unwind.
    sycl_cleanup_noexcept("pearlhash target upload", [&] { q.wait_and_throw(); });
    sycl_cleanup_noexcept("pearlhash preparation", [&] { st.prep_queue.wait_and_throw(); });
    throw;
  }
  const auto commit_fixed_b = [&] {
    if (pair_dist8) {
      st.fixed_b_seed = reuse_fixed_b ? prior_fixed_b_seed : attempt_seed;
      st.fixed_b_valid = true;
    }
  };
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
    reference.positive_inputs = b.positive_inputs;
    reference.seed_prefix_inputs = b.seed_prefix_inputs;
    const uint32_t* const reference_b_seed = reuse_fixed_b ? &prior_fixed_b_seed : nullptr;
    prepare_roots_cpu(reference, attempt_seed, m, n, k, reference_b_seed);
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
    const Result accelerated = *b.result;
    const bool winner_in_bounds = pair_dist8
        ? pair_dist8_offset_valid(static_cast<int>(accelerated.row), m) &&
              pair_dist8_offset_valid(static_cast<int>(accelerated.col), n)
        : accelerated.row % b.proof_rows == 0 && accelerated.col % b.proof_cols == 0 &&
              accelerated.row <= static_cast<unsigned>(m - b.proof_rows) &&
              accelerated.col <= static_cast<unsigned>(n - b.proof_cols);
    if (!accelerated.found || accelerated.seed != attempt_seed || !winner_in_bounds)
      throw std::string("PearlHash accelerated winner is missing or out of bounds");
    b.result->found = 0;
    b.result->chk = 0;
    compute_ab(q, b, attempt_seed, m, n, k, rank, /*portable=*/true);
    // The all-winning test target makes the selected tile schedule-dependent. Check that
    // exact tile on the host before the portable search overwrites the accelerated result;
    // transcript checksums alone cannot detect a broken jackpot hash or byte ordering.
    std::vector<int> winner_rows, winner_cols;
    if (pair_dist8) {
      if (!pair_dist8_lists(static_cast<int>(accelerated.row), static_cast<int>(accelerated.col),
                            m, n, winner_rows, winner_cols))
        throw std::string("PearlHash pair-dist8 winner pattern is invalid");
    } else {
      for (int r = 0; r < b.proof_rows; ++r)
        winner_rows.push_back(static_cast<int>(accelerated.row) + r);
      for (int c = 0; c < b.proof_cols; ++c)
        winner_cols.push_back(static_cast<int>(accelerated.col) + c);
    }
    std::vector<int8_t> winner_a(static_cast<size_t>(winner_rows.size()) * k),
                        winner_b(static_cast<size_t>(winner_cols.size()) * k);
    std::array<uint8_t, 32> winner_key{};
    try {
      for (size_t i = 0; i < winner_rows.size(); ++i)
        q.memcpy(winner_a.data() + i * static_cast<size_t>(k),
                 b.Ap + static_cast<size_t>(winner_rows[i]) * k, static_cast<size_t>(k));
      for (size_t i = 0; i < winner_cols.size(); ++i)
        q.memcpy(winner_b.data() + i * static_cast<size_t>(k),
                 b.Bp + static_cast<size_t>(winner_cols[i]) * k, static_cast<size_t>(k));
      q.memcpy(winner_key.data(), b.cA, winner_key.size());
      q.wait_and_throw();
    } catch (...) {
      sycl_cleanup_noexcept("pearlhash test winner readback", [&] { q.wait_and_throw(); });
      throw;
    }
    Result expected{};
    Buffers winner{};
    winner.proof_rows = static_cast<int>(winner_rows.size());
    winner.proof_cols = static_cast<int>(winner_cols.size());
    // Gathered winner buffers form a contiguous proof rectangle for the scalar
    // host oracle; explicit row/column selection has already happened above.
    winner.Ap = winner_a.data();
    winner.Bp = winner_b.data();
    winner.cA = winner_key.data();
    winner.target = tgtLE;
    winner.result = &expected;
    search_cpu(q, winner, attempt_seed, b.proof_rows, b.proof_cols, k, rank, false);
    if (!expected.found || std::memcmp(expected.jackpot, accelerated.jackpot, 32) != 0)
      throw std::string("PearlHash accelerated/host jackpot mismatch");
    std::fprintf(stderr, "PEARLHASH_TEST jackpot_match=true\n");
    // The portable search carries the same logical 16x16 tile order as the
    // native route.  When pair_dist8 is set it maps each logical tile to the
    // selected 64x64-periodic rows/columns, so this is a full-work checksum,
    // not merely a winner-tile oracle.
    search(q, b, attempt_seed, m, n, k, rank, /*dbg=*/true, pair_dist8);
    q.wait_and_throw();
    if (b.result->chk != accelerated.chk) {
      std::ostringstream message;
      message << "PearlHash accelerated/SYCL checksum mismatch (accelerated=0x" << std::hex
              << accelerated.chk << ", SYCL=0x" << b.result->chk << ')';
      throw message.str();
    }
    std::fprintf(stderr, "PEARLHASH_TEST search_checksum_match=true\n");
    if (pair_dist8) {
      // The scalar checksum path above uses ordinary row/column B' and therefore overwrites the
      // native packed buffer. Restore that exact layout before committing the reusable B snapshot.
      compute_ab<false, true>(q, b, attempt_seed, m, n, k, rank, /*portable=*/true,
                              /*force_b_pack=*/true);
      q.wait_and_throw();
    }
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
  if (compact_mode && !reuse_fixed_b) {
    // Cold host-tree work must not contaminate the GPU wait EMA. Capture
    // both immutable trees once; validity is committed only after the claim.
    std::vector<CV> a(checked_chunk_count(checked_elements(m,k,"compact A cache"),"compact A cache"));
    std::vector<CV> bt(checked_chunk_count(checked_elements(n,k,"compact B cache"),"compact B cache"));
    std::array<uint8_t,32> key{};
    std::copy(st.key,st.key+32,key.begin());
    try {
      q.memcpy(a.data(),b.CVA,a.size()*sizeof(CV));
      q.memcpy(bt.data(),b.CVB,bt.size()*sizeof(CV));
      q.memcpy(st.compact_b.data(),b.cB,32);
      q.wait_and_throw();
    } catch (...) {
      // Retire queued destinations before either local vector is destroyed.
      sycl_cleanup_noexcept("compact root capture",[&]{q.wait_and_throw();});
      throw;
    }
    st.compact_roots.capture(key,m,n,k,rank,attempt_seed,a,bt);
  }
  if (!b.result->found) {
    commit_fixed_b();
    return 0;
  }
  if (is_test) {
    *pseed = b.result->seed;
    commit_fixed_b();
    return 1; // test: core only needs "ok", no proof
  }
  // Compact trees are mutable state. Serialize under the existing state
  // mutex; only immutable bytes may survive unlock or be reused on retries.
  // Other routes keep their original lazy leaf snapshot.
  g_pf.seed = b.result->seed;
  g_pf.b_seed = pair_dist8 ? (reuse_fixed_b ? prior_fixed_b_seed : attempt_seed) : attempt_seed;
  g_pf.row = (int)b.result->row;
  g_pf.col = (int)b.result->col;
  g_pf.m = m;
  g_pf.n = n;
  g_pf.k = k;
  g_pf.rank = rank;
  g_pf.proof_rows = b.proof_rows;
  g_pf.proof_cols = b.proof_cols;
  g_pf.pair_dist8 = pair_dist8;
  g_pf.positive_inputs = b.positive_inputs;
  std::memcpy(g_pf.key, st.key, 32);
  std::memcpy(g_pf.jackpot, b.result->jackpot, 32);
  if (compact_mode) {
    const auto proof_start = std::chrono::steady_clock::now();
    std::array<uint8_t,32> key{};
    std::copy(st.key,st.key+32,key.begin());
    g_pearlhash_proof=st.compact_roots.proof(key,m,n,k,rank,attempt_seed,
                                           g_pf.row,g_pf.col,&g_pf.adjustment_factor);
    if (g_pearlhash_proof.empty())
      throw std::string("Invalid compact PearlHash winning proof");
    g_pf.claim_ready = true;
    if (std::getenv("MOM_PEARLHASH_STATS"))
      std::fprintf(stderr,"PEARLHASH_SYCL_STATS proof=%.3fms\n",
                   std::chrono::duration<double,std::milli>(
                       std::chrono::steady_clock::now()-proof_start).count());
  } else if (q.get_device().is_gpu()) {
    // Snapshot this winning seed before the next attempt can replace its device CVs. Reusing
    // those leaves avoids regenerating and hashing both full matrices on the host per share.
    g_pf.leaves_a.resize(checked_chunk_count(checked_elements(m, k, "proof A"), "proof A"));
    g_pf.leaves_b.resize(checked_chunk_count(checked_elements(n, k, "proof B"), "proof B"));
    try {
      q.memcpy(g_pf.leaves_a.data(), b.CVA, g_pf.leaves_a.size() * sizeof(CV));
      q.memcpy(g_pf.leaves_b.data(), b.CVB, g_pf.leaves_b.size() * sizeof(CV));
      q.wait_and_throw();
    } catch (...) {
      sycl_cleanup_noexcept("pearlhash proof leaf readback", [&] { q.wait_and_throw(); });
      throw;
    }
  }
  commit_fixed_b();
  *pseed = b.result->seed;
  g_pf.valid = true;
  return 1;
}
// Builds and caches the PlainProof plus its matching trusted-share metadata for the last winning tile.
// Repeated calls for the captured tile return the same claim.
const char* pearlhash_claim(uint8_t jackpot[32], uint32_t* adjustment_factor) {
  if (!g_pf.valid || !jackpot || !adjustment_factor)
    return nullptr;
  if (!g_pf.claim_ready) {
    // Compact claims must already be immutable; never use unlocked trees
    // or silently regenerate a missing compact snapshot from newer state.
    if (g_pf.proof_rows == CompactProfile::rows)
      return nullptr;
    const auto proof_start = std::chrono::steady_clock::now();
    if (g_pf.pair_dist8)
      g_pearlhash_proof = build_pair_plain_proof(
          g_pf.seed, g_pf.m, g_pf.n, g_pf.k, g_pf.rank, g_pf.key, g_pf.row, g_pf.col,
          &g_pf.adjustment_factor, g_pf.leaves_a.empty() ? nullptr : &g_pf.leaves_a,
          g_pf.leaves_b.empty() ? nullptr : &g_pf.leaves_b, g_pf.positive_inputs, &g_pf.b_seed);
    else
      g_pearlhash_proof = build_plain_proof(
          g_pf.seed, g_pf.m, g_pf.n, g_pf.k, g_pf.rank, g_pf.key, g_pf.row, g_pf.col,
          &g_pf.adjustment_factor, g_pf.leaves_a.empty() ? nullptr : &g_pf.leaves_a,
          g_pf.leaves_b.empty() ? nullptr : &g_pf.leaves_b, g_pf.proof_rows, g_pf.proof_cols,
          g_pf.positive_inputs);
    if (g_pearlhash_proof.empty())
      return nullptr;
    g_pf.leaves_a.clear();
    g_pf.leaves_b.clear();
    g_pf.claim_ready = true;
    if (std::getenv("MOM_PEARLHASH_STATS"))
      std::fprintf(stderr, "PEARLHASH_SYCL_STATS proof=%.3fms\n",
                   std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - proof_start).count());
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
