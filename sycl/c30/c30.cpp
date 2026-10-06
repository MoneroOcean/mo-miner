// Cortex c30 GPU miner.
//
// The existing C29 entry point is intentionally not reused here: its ABI assumes a pre-derived
// graph key and its endpoint buffers would make C30 exceed a safe 16-GiB memory budget.
//
// Memory/performance notes:
// - Twelve-GiB devices use the packed 48-bit graph representation. Larger GPUs may use the wider
//   layout, but comparisons must force the same layout before attributing a speed difference.
// - The packed path is bit-exact and includes the portable device-SipHash guard; it is not a reduced
//   correctness mode. Current B580 results exceed the local NVIDIA/AMD packed-layout peers.
// - Published NVIDIA/AMD C30 figures from different GPU generations remain context only. On the local
//   RTX 5060 Ti, lolMiner rejects the GPU, GMiner 3.44 has no compatible Blackwell kernel image on
//   Linux or Windows, and the unchanged official Cortex PoolMiner crashes during CUDA Cuckoo
//   initialization before producing a rate.
// - Each long kernel chain completes through the shared low-CPU wait before a result transfer; the
//   in-order queue still carries dependencies within seed, trim, compact, and recovery chains.
// - Cycle search and final proof validation intentionally run on the host after GPU graph trimming,
//   so C30 process CPU usage is not a reliable signal of runtime busy-waiting.
#include "c30.h"
#include "c30_host.h"

#include "../lib-internal.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "c30_kernels.inc"

namespace mom::c30 {
namespace {

constexpr std::uint64_t kEdgeCount = 1ull << kEdgeBits;
constexpr std::uint32_t kBucketBaseStride = 1u << (kEdgeBits - C30_BUCKET_BITS);
constexpr std::uint32_t kSeedBucketStride = kBucketBaseStride * 65u / 64u;
constexpr std::uint32_t kTrimBucketStride = kBucketBaseStride * 11u / 16u;
constexpr std::uint32_t kLowMemoryTrimBucketStride = kBucketBaseStride * 21u / 32u;
constexpr std::uint64_t kSeedCapacity =
    static_cast<std::uint64_t>(C30_BUCKET_COUNT) * kSeedBucketStride;
constexpr std::uint64_t kTrimCapacity =
    static_cast<std::uint64_t>(C30_BUCKET_COUNT) * kTrimBucketStride;
constexpr std::uint64_t kLowMemoryTrimCapacity =
    static_cast<std::uint64_t>(C30_BUCKET_COUNT) * kLowMemoryTrimBucketStride;
constexpr std::uint64_t kMaxEdgeArrayBytes = kSeedCapacity * sizeof(sycl::uint2);
constexpr std::uint64_t kSafetyReserve = 512ull * 1024ull * 1024ull;
constexpr std::uint64_t kCounterBytes = 2ull * C30_BUCKET_COUNT * sizeof(std::uint32_t);
constexpr std::uint64_t kRequiredDeviceBytes =
    (kSeedCapacity + kTrimCapacity) * sizeof(sycl::uint2) + kCounterBytes;
constexpr std::uint64_t kLowMemoryRequiredDeviceBytes =
    (kSeedCapacity + kLowMemoryTrimCapacity) * sizeof(C30PackedEdge) + kCounterBytes;
constexpr std::uint64_t kLowMemoryReserve = 128ull * 1024ull * 1024ull;
constexpr std::uint32_t kMaxHostEdges = 1u << 22;
constexpr unsigned kDefaultTrimRounds = 384;
constexpr unsigned kDefaultSparseTrimStart = 64;

static_assert(sizeof(std::size_t) >= 8, "c30 requires a 64-bit host");
static_assert(kEdgeCount == (1ull << 30), "Cortex c30 must use edge_bits=30");
static_assert(kProofSize == 42, "Cortex c30 must use proof_size=42");

std::uint32_t count_edges(const std::vector<std::uint32_t>& counts) {
  std::uint64_t total = 0;
  for (const std::uint32_t count : counts)
    total += count;
  if (total > kEdgeCount)
    throw std::runtime_error("c30 trim count overflow");
  return static_cast<std::uint32_t>(total);
}

struct DeviceMemory {
  sycl::queue& queue;
  std::array<sycl::uint2*, 2> edges{nullptr, nullptr};
  std::array<std::uint64_t, 2> edge_capacity{0, 0};
  std::array<C30PackedEdge*, 2> packed_edges{nullptr, nullptr};
  std::array<std::uint64_t, 2> packed_capacity{0, 0};
  std::array<std::uint32_t*, 2> bucket_counts{nullptr, nullptr};
  std::uint32_t* cursors = nullptr;
  std::uint32_t* overflow = nullptr;
  std::uint32_t* target_u = nullptr;
  std::uint32_t* target_v = nullptr;
  std::uint32_t* recovered = nullptr;
  bool low_memory = false;

  explicit DeviceMemory(sycl::queue& q) : queue(q) {
    const sycl::device device = queue.get_device();
    const std::uint64_t global_bytes = device.get_info<sycl::info::device::global_mem_size>();
    const std::uint64_t max_alloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
    if (!mom_has_usm_device(device))
      throw std::runtime_error("c30 requires device USM");
    const char* const force_low_memory = std::getenv("MOM_C30_FORCE_LOW_MEMORY");
    low_memory = (force_low_memory && force_low_memory[0] != '0') ||
                 max_alloc < kMaxEdgeArrayBytes ||
                 global_bytes < kRequiredDeviceBytes + kSafetyReserve ||
                 mom_intel_eu_simd_width(device) == 8;
    if (low_memory && global_bytes < kLowMemoryRequiredDeviceBytes + kLowMemoryReserve)
      throw std::runtime_error("c30 needs 10.2-GiB low-memory graph storage plus reserve");

    try {
      if (low_memory) {
        allocate_packed(0, kSeedCapacity);
        allocate_packed(1, kLowMemoryTrimCapacity);
      } else {
        allocate_edges(0, kSeedCapacity);
        allocate_edges(1, kTrimCapacity);
      }
      for (auto& count_array : bucket_counts)
        count_array = sycl::malloc_device<std::uint32_t>(C30_BUCKET_COUNT, queue);
      cursors = sycl::malloc_device<std::uint32_t>(C30_BUCKET_COUNT, queue);
      overflow = sycl::malloc_device<std::uint32_t>(1, queue);
      target_u = sycl::malloc_device<std::uint32_t>(kProofSize, queue);
      target_v = sycl::malloc_device<std::uint32_t>(kProofSize, queue);
      recovered = sycl::malloc_device<std::uint32_t>(kProofSize, queue);
      if (!bucket_counts[0] || !bucket_counts[1] || !cursors || !overflow || !target_u ||
          !target_v || !recovered)
        throw std::bad_alloc();
    } catch (...) {
      release();
      throw;
    }
  }

  ~DeviceMemory() {
    release();
  }

  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory&) = delete;

  void allocate_edges(const unsigned index, const std::uint64_t count) {
    if (edge_capacity[index] >= count)
      return;
    if (edges[index])
      sycl::free(edges[index], queue);
    edges[index] = nullptr;
    edge_capacity[index] = 0;
    try {
      edges[index] = sycl::malloc_device<sycl::uint2>(count, queue);
      if (!edges[index])
        throw std::bad_alloc();
      edge_capacity[index] = count;
    } catch (...) {
      if (edges[index])
        sycl::free(edges[index], queue);
      edges[index] = nullptr;
      throw;
    }
  }

  void allocate_packed(const unsigned index, const std::uint64_t count) {
    if (packed_capacity[index] >= count)
      return;
    if (packed_edges[index])
      sycl::free(packed_edges[index], queue);
    packed_edges[index] = nullptr;
    packed_capacity[index] = 0;
    packed_edges[index] = sycl::malloc_device<C30PackedEdge>(count, queue);
    if (!packed_edges[index])
      throw std::bad_alloc();
    packed_capacity[index] = count;
  }

  void release_packed(const unsigned index) noexcept {
    if (packed_edges[index])
      try {
        sycl::free(packed_edges[index], queue);
      } catch (...) {
      }
    packed_edges[index] = nullptr;
    packed_capacity[index] = 0;
  }

  void release() noexcept {
    for (unsigned i = 0; i < edges.size(); ++i) {
      if (edges[i])
        try {
          sycl::free(edges[i], queue);
        } catch (...) {
        }
      edges[i] = nullptr;
      edge_capacity[i] = 0;
    }
    for (unsigned i = 0; i < packed_edges.size(); ++i)
      release_packed(i);
    for (auto& count_array : bucket_counts) {
      if (count_array)
        try {
          sycl::free(count_array, queue);
        } catch (...) {
        }
      count_array = nullptr;
    }
    if (cursors) {
      try {
        sycl::free(cursors, queue);
      } catch (...) {
      }
      cursors = nullptr;
    }
    if (overflow)
      try {
        sycl::free(overflow, queue);
      } catch (...) {
      }
    overflow = nullptr;
    if (target_u)
      try {
        sycl::free(target_u, queue);
      } catch (...) {
      }
    if (target_v)
      try {
        sycl::free(target_v, queue);
      } catch (...) {
      }
    if (recovered)
      try {
        sycl::free(recovered, queue);
      } catch (...) {
      }
    target_u = nullptr;
    target_v = nullptr;
    recovered = nullptr;
  }
};

} // namespace

struct Solver {
  sycl::device device;
  sycl::queue queue;
  DeviceMemory memory;

  explicit Solver(const std::string& dev_str)
      : device(get_dev(dev_str)),
        queue(device, sycl::async_handler([](sycl::exception_list errors) {
                for (const std::exception_ptr& error : errors) {
                  try {
                    std::rethrow_exception(error);
                  } catch (const sycl::exception& exception) {
                    std::fprintf(stderr, "c30 asynchronous SYCL error: %s\n", exception.what());
                  }
                }
              }),
              sycl::property_list{sycl::property::queue::in_order{}}),
        memory(queue) {
  }

  ~Solver() {
    sycl_cleanup_noexcept("c30", [&] {
      queue.wait_and_throw();
    });
  }

  std::optional<Solution> solve(const Job& job) {
    using Clock = std::chrono::steady_clock;
    const bool profile = std::getenv("MOM_C30_PROFILE") != nullptr;
    auto stage = Clock::now();
    const auto report = [&](const char* label) {
      if (profile)
        std::fprintf(stderr, "c30 profile %-18s %.3f s\n", label,
                     std::chrono::duration<double>(Clock::now() - stage).count());
      stage = Clock::now();
    };
    report("allocation");
    const host::SipKey host_key = host::graph_key(job.header, job.nonce);
    const C30DeviceKey key{host_key.k0, host_key.k1, host_key.k2, host_key.k3};
    unsigned trim_rounds = kDefaultTrimRounds;
    unsigned long requested = 0;
    if (mom_parse_env_ulong("MOM_C30_TRIM_ROUNDS", requested)) {
      if (requested >= 64 && requested <= 512)
        trim_rounds = static_cast<unsigned>(requested);
    }
    unsigned sparse_trim_start = kDefaultSparseTrimStart;
    if (mom_parse_env_ulong("MOM_C30_SPARSE_TRIM_START", requested) &&
        requested <= std::numeric_limits<unsigned>::max()) {
      sparse_trim_start =
          requested <= trim_rounds ? static_cast<unsigned>(requested)
                                   : std::numeric_limits<unsigned>::max();
    }
    std::vector<std::uint32_t> counts(C30_BUCKET_COUNT);
    queue.fill(memory.bucket_counts[0], 0u, C30_BUCKET_COUNT);
    queue.fill(memory.overflow, 0u, 1);
    const bool pair32 =
        device.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos;
    const sycl::event seed = memory.low_memory
        ? c30_seed_packed(queue, key, memory.bucket_counts[0], kSeedBucketStride,
                          memory.packed_edges[0], memory.overflow)
        : c30_seed_fixed(queue, key, memory.bucket_counts[0], kSeedBucketStride,
                         memory.edges[0], memory.overflow);
    sycl_wait_and_throw(seed, device);
    report("initial seed");

    unsigned input_index = 0;
    unsigned output_index = 1;
    sycl::event trim_event;
    std::array<std::uint32_t, 2> fixed_strides = {kSeedBucketStride, kTrimBucketStride};
    if (memory.low_memory)
      fixed_strides = {kLowMemoryTrimBucketStride, kLowMemoryTrimBucketStride};
    for (unsigned round = 0; round < trim_rounds; ++round) {
      queue.fill(memory.bucket_counts[output_index], 0u, C30_BUCKET_COUNT);
      const bool sparse = round >= sparse_trim_start;
      if (memory.low_memory) {
        const std::uint32_t input_stride = round ? kLowMemoryTrimBucketStride : kSeedBucketStride;
        if (sparse)
          trim_event = c30_trim_sparse_packed(
              queue, memory.packed_edges[input_index], memory.bucket_counts[input_index],
              input_stride, memory.bucket_counts[output_index], kLowMemoryTrimBucketStride,
              memory.packed_edges[output_index], memory.overflow);
        else
          trim_event = c30_trim_packed(
              queue, memory.packed_edges[input_index], memory.bucket_counts[input_index],
              input_stride, memory.bucket_counts[output_index], kLowMemoryTrimBucketStride,
              memory.packed_edges[output_index], memory.overflow);
      } else if (sparse) {
        trim_event = c30_trim_sparse_fixed(
            queue, memory.edges[input_index], memory.bucket_counts[input_index],
            fixed_strides[input_index], memory.bucket_counts[output_index],
            fixed_strides[output_index], memory.edges[output_index], memory.overflow);
      } else {
        trim_event = c30_trim_fixed(
            queue, memory.edges[input_index], memory.bucket_counts[input_index],
            fixed_strides[input_index], memory.bucket_counts[output_index],
            fixed_strides[output_index], memory.edges[output_index], memory.overflow);
      }
      std::swap(input_index, output_index);
    }
    // Finish the trim chain before submitting D2H copies; otherwise some runtimes busy-spin while
    // the copy submission waits for hundreds of queued kernels.
    sycl_wait_and_throw(trim_event, device);
    std::uint32_t overflow = 0;
    queue.memcpy(counts.data(), memory.bucket_counts[input_index],
                 C30_BUCKET_COUNT * sizeof(std::uint32_t));
    sycl_wait_and_throw(queue.memcpy(&overflow, memory.overflow, sizeof(overflow)), device);
    if (overflow)
      throw std::runtime_error("c30 fixed bucket overflow");
    const std::uint32_t survivor_count = count_edges(counts);
    if (survivor_count == 0)
      return std::nullopt;

    queue.fill(memory.cursors, 0u, 1);
    sycl::event compact_event;
    if (memory.low_memory) {
      memory.allocate_edges(0, survivor_count);
      compact_event = c30_compact_packed(
          queue, memory.packed_edges[input_index], memory.bucket_counts[input_index],
          kLowMemoryTrimBucketStride, memory.edges[0], memory.cursors);
    } else {
      compact_event = c30_compact(
          queue, memory.edges[input_index], memory.bucket_counts[input_index],
          fixed_strides[input_index], memory.edges[output_index], memory.cursors);
    }
    sycl_wait_and_throw(compact_event, device);
    std::uint32_t compacted = 0;
    sycl_wait_and_throw(queue.memcpy(&compacted, memory.cursors, sizeof(compacted)), device);
    if (compacted != survivor_count)
      throw std::runtime_error("c30 compact count mismatch");
    input_index = memory.low_memory ? 0 : output_index;
    report("trim rounds");
    if (profile)
      std::fprintf(stderr, "c30 profile rounds             %u\n", trim_rounds);
    if (profile)
      std::fprintf(stderr, "c30 profile survivors          %u\n", survivor_count);

    if (survivor_count > kMaxHostEdges)
      throw std::runtime_error("c30 trimmed graph is too large for the host cycle search");
    std::vector<sycl::uint2> trimmed_edges(survivor_count);
    sycl_wait_and_throw(
        queue.memcpy(trimmed_edges.data(), memory.edges[input_index],
                     static_cast<std::size_t>(survivor_count) * sizeof(sycl::uint2)),
        device);

    std::vector<host::Endpoints> trimmed(survivor_count);
    for (std::size_t i = 0; i < trimmed.size(); ++i)
      trimmed[i] = {trimmed_edges[i].x(), trimmed_edges[i].y()};
    const auto cycle = host::find_cycle(trimmed);
    report("copy/cycle search");
    if (!cycle)
      return std::nullopt;

    std::array<std::uint32_t, kProofSize> target_u{}, target_v{};
    for (unsigned i = 0; i < kProofSize; ++i) {
      target_u[i] = (*cycle)[i].u;
      target_v[i] = (*cycle)[i].v;
    }
    queue.memcpy(memory.target_u, target_u.data(), sizeof(target_u));
    queue.memcpy(memory.target_v, target_v.data(), sizeof(target_v));
    queue.fill(memory.recovered, std::numeric_limits<std::uint32_t>::max(), kProofSize);
    const sycl::event recover_event = pair32
        ? c30_recover32(queue, key, memory.target_u, memory.target_v, memory.recovered)
        : c30_recover(queue, key, memory.target_u, memory.target_v, memory.recovered);
    sycl_wait_and_throw(recover_event, device);

    Solution solution;
    solution.nonce = job.nonce;
    sycl_wait_and_throw(
        queue.memcpy(solution.edges.data(), memory.recovered, sizeof(solution.edges)), device);
    report("nonce recovery");
    if (std::find(solution.edges.begin(), solution.edges.end(),
                  std::numeric_limits<std::uint32_t>::max()) != solution.edges.end())
      throw std::runtime_error("c30 nonce recovery failed");
    std::sort(solution.edges.begin(), solution.edges.end());
    solution.hash = host::solution_hash(solution.edges);
    if (!host::verify(host_key, solution.edges))
      throw std::runtime_error("c30 host proof revalidation failed");
    return solution;
  }
};

} // namespace mom::c30

namespace {

struct C30State {
  mom::c30::Solver solver;
  std::mutex mutex;
  explicit C30State(const std::string& dev_str) : solver(dev_str) {
  }
};

DeviceStateRegistry<C30State>& c30_registry() {
  static auto* const registry = new DeviceStateRegistry<C30State>;
  return *registry;
}

C30State& c30_state(const std::string& dev_str) {
  return c30_registry().get(dev_str, [&] { return std::make_unique<C30State>(dev_str); });
}

int c30_portable_test(const mom::c30::Job& job, std::uint8_t* output, const std::string& dev_str) {
  sycl::queue queue(get_dev(dev_str), sycl::property::queue::in_order{});
  const mom::c30::host::SipKey host_key = mom::c30::host::graph_key(job.header, job.nonce);
  const C30DeviceKey key{host_key.k0, host_key.k1, host_key.k2, host_key.k3};
  sycl::uint2* endpoints = sycl::malloc_shared<sycl::uint2>(4, queue);
  if (!endpoints)
    throw std::bad_alloc();
  try {
    const sycl::event event = queue.submit([&](sycl::handler& handler) {
      handler.parallel_for(sycl::range<1>{4}, [=](sycl::id<1> item) {
        const std::uint32_t edge = item[0] == 3 ? 63u : item[0] == 2 ? 62u : item[0];
        const std::uint64_t hash = c30_edge_hash(key, edge);
        endpoints[item] = sycl::uint2{c30_endpoint(hash, true), c30_other_endpoint(hash, true)};
      });
    });
    sycl_wait_and_throw(event, queue.get_device());
    for (unsigned i = 0; i < 4; ++i) {
      const std::uint32_t pair[2] = {endpoints[i].x(), endpoints[i].y()};
      std::memcpy(output + i * sizeof(pair), pair, sizeof(pair));
    }
  } catch (...) {
    sycl::free(endpoints, queue);
    throw;
  }
  sycl::free(endpoints, queue);
  return 1;
}

} // namespace

void c30_cleanup_states() noexcept {
  try {
    c30_registry().clear();
  } catch (...) {
    std::fprintf(stderr, "c30: ordered SYCL cleanup failed\n");
  }
}

int c30(unsigned, const unsigned proof_size, const std::uint8_t* input, const unsigned input_size,
        std::uint8_t* output, std::uint32_t* output_edges, std::uint64_t* nonce,
        const std::string& dev_str) {
  if (!input || !output || !output_edges || !nonce)
    throw std::string("c30 requires input, output, edge, and nonce pointers");
  if (input_size != 32 && input_size != 40)
    throw std::string("c30 requires a 32-byte search header or 40-byte test vector; got ") +
        std::to_string(input_size) + " bytes";
  if (proof_size != mom::c30::kProofSize)
    throw std::string("c30 requires proof size 42; got ") + std::to_string(proof_size);
  // Search uses the canonical 32-byte header with an external nonce. The 40-byte header-plus-nonce
  // form remains for test vectors, whose embedded nonce is extracted below.
  mom::c30::Job job;
  std::copy_n(input, job.header.size(), job.header.begin());
  if (input_size == 40) {
    job.nonce = 0;
    for (unsigned i = 0; i < 8; ++i)
      job.nonce |= static_cast<std::uint64_t>(input[32 + i]) << (8 * i);
  } else
    job.nonce = *nonce;
  if (std::getenv("MOM_C30_TEST_EDGE"))
    return c30_portable_test(job, output, dev_str);
  C30State& state = c30_state(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  const std::optional<mom::c30::Solution> solution = state.solver.solve(job);
  if (!solution)
    return 0;
  *nonce = solution->nonce;
  std::copy(solution->hash.begin(), solution->hash.end(), output);
  std::copy(solution->edges.begin(), solution->edges.end(), output_edges);
  return 1;
}
