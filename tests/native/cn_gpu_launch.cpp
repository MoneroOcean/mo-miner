#include "../../sycl/cn_gpu/launch.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

void require(const bool condition, const char* const label) {
  if (!condition) {
    std::fprintf(stderr, "CN/GPU launch regression: %s\n", label);
    std::abort();
  }
}

void check_batch(const unsigned batch) {
  const unsigned padded = cn_gpu_padded_batch(batch);
  require(padded >= batch && padded - batch <= 1, "at most one padding hash");
  require(padded % 2 == 0, "even scratchpad allocation");
  const unsigned aes_rows = cn_gpu_aes_group_rows(padded);
  require(aes_rows == 2 || aes_rows == 4, "supported AES row count");
  require(padded % aes_rows == 0, "AES launch divides without extra allocation");

  for (const bool width8 : {false, true}) {
    const unsigned group_size = cn_gpu_recurrence_group_size(width8);
    const unsigned hashes_per_group = group_size / cn_gpu_lanes_per_hash;
    require(group_size == (width8 ? 16U : 32U), "recurrence group width");
    const uint64_t items = static_cast<uint64_t>(padded) * cn_gpu_lanes_per_hash;
    require(items % group_size == 0, "recurrence launch divides exactly");
    const uint64_t groups = items / group_size;
    require((groups - 1) * hashes_per_group + hashes_per_group - 1 == padded - 1,
            "last group stays inside allocated hashes");

    // Exhaust small launches; integer-limit cases above need no huge allocation or GPU.
    if (padded > 2048)
      continue;
    std::vector<unsigned> lanes(padded * cn_gpu_lanes_per_hash);
    for (uint64_t group = 0; group < groups; ++group) {
      for (unsigned lid = 0; lid < group_size; ++lid) {
        const unsigned sub = lid / cn_gpu_lanes_per_hash;
        const unsigned lane = lid % cn_gpu_lanes_per_hash;
        const uint64_t hash = group * hashes_per_group + sub;
        require(hash < padded, "no unallocated padding hash");
        require(sub * cn_gpu_lanes_per_hash + lane < group_size,
                "local-memory slice stays inside workgroup");
        ++lanes[hash * cn_gpu_lanes_per_hash + lane];
      }
    }
    for (const unsigned count : lanes)
      require(count == 1, "every hash lane executes exactly once");
  }
}

void check_intel_batch(const unsigned compute_units, const bool native_width8,
                       const uint64_t global_mem, const uint64_t max_alloc,
                       const unsigned requested, const unsigned expected,
                       const char* const label) {
  constexpr uint64_t reserve = 512ULL << 20;
  constexpr uint64_t scratch_bytes = 2ULL << 20;
  constexpr uint64_t bytes_per_hash = scratch_bytes + 200 + 135 + 32;
  const unsigned batch = cn_gpu_intel_level_zero_batch(
      compute_units, native_width8, global_mem, max_alloc, requested);
  require(batch == expected, label);
  if (!batch)
    return;
  const unsigned padded = cn_gpu_padded_batch(batch);
  require(padded != 0, "Intel batch padding does not overflow");
  require(static_cast<uint64_t>(padded) * scratch_bytes <= max_alloc,
          "padded Intel scratch fits the largest allocation");
  require(global_mem > reserve &&
              static_cast<uint64_t>(padded) * bytes_per_hash <= global_mem - reserve,
          "padded Intel buffers fit after the memory reserve");
}

} // namespace

int main() {
  constexpr unsigned max_batch = std::numeric_limits<unsigned>::max();
  constexpr uint64_t GiB = 1ULL << 30;
  constexpr uint64_t reserve = 512ULL << 20;
  constexpr uint64_t scratch_bytes = 2ULL << 20;
  constexpr uint64_t bytes_per_hash = scratch_bytes + 200 + 135 + 32;
  constexpr uint64_t max_memory = std::numeric_limits<uint64_t>::max();
  static_assert(cn_gpu_lanes_per_hash == 16);
  static_assert(cn_gpu_padded_batch(1) == 2);
  static_assert(cn_gpu_padded_batch(8) == 8);
  require(std::strcmp(cn_gpu_intel_backend(false, true), "sycl-native") == 0,
          "known native width8 uses its optimized recurrence worker");
  require(std::strcmp(cn_gpu_intel_backend(false, false), "sycl-l0") == 0,
          "unknown or mixed widths retain portable Level Zero");
  for (const bool width8 : {false, true})
    require(std::strcmp(cn_gpu_intel_backend(true, width8), "sycl-opencl") == 0,
            "required OpenCL takes priority over width tuning");
  require(cn_gpu_padded_batch(0) == 0, "reject empty batch");
  require(cn_gpu_padded_batch(max_batch) == 0, "reject overflowing padding");
  for (unsigned batch = 1; batch <= 65; ++batch)
    check_batch(batch);
  for (const unsigned batch : std::array<unsigned, 7>{
           1535, 1536, 1537, 2047, 2048, max_batch - 2, max_batch - 1})
    check_batch(batch);

  check_intel_batch(512, true, 16 * GiB, 8 * GiB, 0, 4096, "native width8 uses eight hashes/EU");
  check_intel_batch(512, false, 16 * GiB, 8 * GiB, 0, 1536, "unknown width retains three hashes/EU");
  check_intel_batch(256, false, 16 * GiB, 8 * GiB, 0, 768, "wider Intel retains three hashes/EU");
  check_intel_batch(1, true, 16 * GiB, 8 * GiB, 0, 8, "small device retains minimum batch");
  check_intel_batch(0, true, 16 * GiB, 8 * GiB, 4096, 0, "reject unknown compute-unit count");
  check_intel_batch(512, true, 0, 8 * GiB, 0, 0, "reject unknown total memory");
  check_intel_batch(512, true, reserve, 8 * GiB, 0, 0, "preserve the complete memory reserve");
  check_intel_batch(512, true, 16 * GiB, 0, 0, 0, "reject unknown allocation limit");
  check_intel_batch(512, true, GiB, 8 * GiB, 0, 254, "account for state and maximum input/output");
  check_intel_batch(512, true, 16 * GiB, GiB, 0, 512, "clamp scratch to maximum allocation");
  check_intel_batch(512, true, reserve + bytes_per_hash, 8 * GiB, 0, 0,
                    "one hash of capacity cannot fit its even padding");
  check_intel_batch(512, true, 16 * GiB, scratch_bytes, 0, 0,
                    "one scratch allocation cannot fit even padding");
  for (const bool needs_opencl : {false, true}) {
    for (const bool all_width8 : {false, true}) {
      const char* const backend = cn_gpu_intel_backend(needs_opencl, all_width8);
      const bool native_width8 = std::strcmp(backend, "sycl-native") == 0;
      check_intel_batch(512, native_width8, 16 * GiB, 8 * GiB, 0,
                        !needs_opencl && all_width8 ? 4096 : 1536,
                        "mixed, unknown, or required-OpenCL transport stays shallow");
    }
  }
  for (const bool width8 : {false, true}) {
    check_intel_batch(max_batch, width8, max_memory, max_memory, 0, max_batch - 1,
                      "compute-unit multiplication never wraps");
    check_intel_batch(512, width8, max_memory, max_memory, max_batch, max_batch - 1,
                      "explicit unsigned limit leaves room for padding");
  }
  for (const unsigned requested : {0U, 1U, 23U, 24U, 25U, max_batch}) {
    check_intel_batch(512, true, reserve + 25 * bytes_per_hash, 25 * scratch_bytes,
                      requested, requested && requested < 24 ? requested : 24,
                      "odd override is bounded by padded capacity");
  }

  std::puts("CN/GPU launch host test passed");
}
