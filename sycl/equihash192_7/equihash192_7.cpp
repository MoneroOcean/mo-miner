// Equihash 192,7 (Zclassic) complete SYCL solver.
//
// Performance/portability notes:
// - The Wagner arenas need about 6 GiB before runtime reserve, so production requires an 8-GiB-class
//   device. The same exact solver and Zclassic proof vector are used on Intel, NVIDIA, and AMD.
// - Round-0 partitioning and workgroup 1024 improved the measured Intel/NVIDIA paths; later collision
//   rounds remain the dominant cross-vendor gap. Scaled results from other AMD GPUs are not a hardware
//   limit or a same-GPU target.
// - Level Zero splits the first middle collision round into eight disjoint bin ranges. This cut that
//   stage by about 14%. Splitting the final two rounds into two ranges saves another 0.9 ms per
//   solve and raises robust steady B580 throughput from 9.65 to 9.73 I/s.
// - gfx12 HIP keeps the middle-round record aligned: it is about 20% faster on RX 9060 XT, while the
//   split record remains about 3% faster on B580 and preserves the NVIDIA/OpenCL paths.
// - gfx12 and oneAPI store the six-field round-2 record in its natural 24 bytes. Removing 8 bytes of
//   padding raised RX 9060 XT throughput by about 2.5% and B580 from 9.73 to 10.56 I/s.
// - oneAPI also stores the final two-field record in its natural 12 bytes, reducing that collision
//   stage by about 15% and raising measured B580 throughput from 10.56 to 10.75 I/s.
// - Storing oneAPI's seven-field round-1 record in its natural 28 bytes reduces the first collision
//   stage enough to raise replicated B580 throughput from 10.75 to 10.83 I/s.
// - Storing oneAPI's level-0 record in its natural 28 bytes reduces generation time and raises
//   replicated B580 throughput from 10.83 to 10.98 I/s.
// - Level Zero prefers 13 bucket bits and falls back to the 12-bit layout on bounded overflow.
//   Halving each bucket's capacity reduces generation, first-round, and final-round work enough to
//   raise replicated B580 throughput from 10.98 to 11.53 I/s.
// - Four partitions match the lower-density 13-bit first-collision buckets better than eight,
//   reducing that stage by about 5% and raising replicated B580 throughput to 11.64 I/s. The
//   overflow fallback keeps eight partitions for its denser 12-bit buckets.
// - A staged Level Zero arena keeps 13-bit buckets for generation, the first collision, and the
//   final collision while using 12-bit buckets through the middle rounds. This cuts the first
//   middle collision from about 16.50 to 14.14 ms, improves the following rounds, and raises the
//   replicated B580 result from 11.64 to 12.36 I/s. Bounded overflow retries the uniform 13-bit
//   arena and then the 12-bit production arena, so the faster layout does not discard valid work.
// - Splitting the staged arena's first two 12-bit collision rounds into four disjoint bin ranges
//   cuts them from about 13.45/11.79 to 12.20/11.20 ms and raises replicated B580 throughput from
//   12.36 to 12.63 I/s.
// - Widening the staged arena's outer levels and first two collision levels to 14 bucket bits,
//   while using a 512-thread root workgroup, lowers the first middle collision from about 13.63 to
//   11.55 ms. The following transition costs more, but confirmed B580 throughput rises to 12.86 I/s.
// - Giving that sparse 14-to-12-bit transition two partitions instead of four cuts it from about
//   13.66 to 12.11 ms and raises confirmed B580 throughput to 13.12 I/s.
// - gfx12 splits the first collision round into eight disjoint bin ranges instead of two, cutting
//   that stage by about 9% and improving measured RX 9060 XT throughput by about 1.5%.
// - gfx12 splits the final three collision rounds across 8/4/4 disjoint bin ranges. This preserves
//   every collision pair while reducing per-workgroup bin work, improving RX 9060 XT by about 6%.
// - Splitting the two remaining gfx12 middle rounds into eight ranges cuts those stages by about
//   18% and 13%, raising measured RX 9060 XT throughput from 13.77 to 14.40 I/s.
// - The gfx12 two-field round record uses its natural 12-byte packed size instead of 16-byte
//   alignment. This halves the final collision stage and raises throughput from 14.40 to 15.59 I/s.
// - HIP caches selected collision records when every active/retry layout fits local memory,
//   raising RX 9060 XT throughput to 16.13 I/s. Smaller devices retain the prior global-load path.
// - Sixteen extra HIP slots per 12-bit bucket break the old 4-KiB-multiple record strides. This
//   reduced several collision stages sharply and raised RX 9060 XT from a fresh 16.10 to 20.00 I/s
//   against a 20.2-I/s local same-card reference.
//   The address-layout effect is measured, not a claim about undocumented cache indexing. Matching
//   retry/hybrid capacities retain the same parent-reference widths and cost only 9.75 MiB more;
//   allocation/local-memory checks and complete bounded retries remain in place.
// - oneAPI uses the same sixteen-slot stride padding. Matched B580 Linux replays improved from 12.97 to
//   15.64 I/s; the address-layout effect is measured, not a hardware-ceiling claim. The shared arena
//   capacities preserve parent-reference widths and bounded retries; Intel physical arena storage grows
//   by 9 MiB.
// - Non-final zero remainders are pruned like other optimized Equihash solvers because duplicate
//   provenance otherwise floods the late arenas; every emitted proof is independently verified.
// - The final compute event completes through the shared low-CPU wait before result-copy submission;
//   some runtimes otherwise busy-spin while enqueuing that copy behind live kernels.

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#include "../equihash_pow.hpp"
#include "../lib-internal.h"
#include "../../native/consts.h"
#include "equihash192_7_direct_session.hpp"

namespace mom_equihash192_7 {

using Solver = direct::Session<>;

class GenerationTestKernel;

static int run_generation_test(const std::string& device_name, const std::uint8_t* input,
                               std::uint8_t* output) {
  using Spec = direct::Spec;
  std::array<std::uint8_t, Spec::header_length> header;
  std::array<std::uint8_t, Spec::hash_length> digest{};
  std::memcpy(header.data(), input, header.size());
  sycl::queue queue{get_dev(device_name)};
  {
    sycl::buffer bytes(digest.data(), sycl::range<1>(digest.size()));
    queue.submit([&](sycl::handler& handler) {
      sycl::accessor result(bytes, handler, sycl::write_only, sycl::no_init);
      handler.single_task<GenerationTestKernel>([=] {
        std::uint8_t value[Spec::hash_length];
        mom_equihash::hash_index<Spec>(header.data(), 0, value);
        for (unsigned i = 0; i < Spec::hash_length; ++i)
          result[i] = value[i];
      });
    });
  }
  std::memset(output, 0, SMALL_BLOB_SOL_LEN);
  std::memcpy(output, digest.data(), digest.size());
  return 1;
}

struct MiningState {
  std::mutex mutex;
  Solver solver;
  explicit MiningState(const std::string& device_name) : solver(get_dev(device_name)) {
  }
};

static DeviceStateRegistry<MiningState>& registry() {
  static auto* const value = new DeviceStateRegistry<MiningState>;
  return *value;
}

static MiningState& state_for(const std::string& device_name) {
  return registry().get(device_name, [&] { return std::make_unique<MiningState>(device_name); });
}

void cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "equihash192_7: ordered SYCL cleanup failed\n");
  }
}

} // namespace mom_equihash192_7

int equihash192_7(unsigned, std::uint32_t, const std::uint8_t* input, const unsigned input_size,
                  std::uint8_t* solution_out, std::uint64_t*, const std::uint8_t* target,
                  const unsigned intensity, const bool is_test, const bool is_benchmark,
                  const std::string& device_name) {
  using Spec = mom_equihash192_7::direct::Spec;
  if (!input || !solution_out || (!is_test && !is_benchmark && !target) ||
      input_size != Spec::header_length || intensity != 1)
    throw std::string(
      "equihash192_7 requires input, output, a mining target, a 140-byte header, and intensity 1");
  if (is_test && std::getenv("MOM_EQUIHASH192_7_SOLVE") == nullptr)
    return mom_equihash192_7::run_generation_test(device_name, input, solution_out);
  auto& state = mom_equihash192_7::state_for(device_name);
  std::lock_guard<std::mutex> lock(state.mutex);
  std::uint8_t header[Spec::header_length];
  std::memcpy(header, input, sizeof(header));
  auto solutions = state.solver.run(header);
  if (is_test)
    std::sort(solutions.begin(), solutions.end(), [](const auto& left, const auto& right) {
      return left.encoded < right.encoded;
    });
  if (is_test)
    std::memset(solution_out, 0, SMALL_BLOB_SOL_LEN);
  const std::size_t capacity =
      std::min<std::size_t>(255, (EQUIHASH_SOL_BUFFER_LEN - 1) / Spec::solution_length);
  unsigned count = 0;
  for (const auto& solution : solutions) {
    if (count == capacity || (is_test && count == 1))
      break;
    if (!is_test && !is_benchmark &&
        !mom_equihash::pow::meets_target(header, sizeof(header), solution.encoded.data(),
                                         solution.encoded.size(), target))
      continue;
    std::memcpy(solution_out + 1 + count * solution.encoded.size(), solution.encoded.data(),
                solution.encoded.size());
    ++count;
  }
  solution_out[0] = static_cast<std::uint8_t>(count);
  return static_cast<int>(count);
}
