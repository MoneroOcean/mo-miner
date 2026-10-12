// ZHash Equihash(144,5) SYCL solver and miner adapter.
//
// Performance/portability notes:
// - Generation and the complete Wagner solve are checked against the strict BTCGPU vector; reported
//   speed counts every recovered solution on every vendor.
// - Proof counts vary by input, so performance comparisons use longer sample sets with outliers
//   trimmed; a short-window median can understate the same steady solver.
// - The same RX 9060 XT runs lolMiner at 83.1 Sol/s on Linux and 83.0 Sol/s on Windows, so the
//   remaining matched-card gap is real but is not evidence of an OS or hardware limit.
// - Non-final zero remainders are pruned like other optimized Equihash solvers because duplicate
//   provenance otherwise floods the late arenas; every emitted proof is independently verified.
// - The compact layout avoids an Intel local-memory occupancy cliff. A bounded overflow frees that
//   arena, retries the same header with the safe layout, then restores the compact path; only one
//   arena is resident at a time, preserving support for 4 GiB devices.
// - Intel's dedicated worker uses 13 bucket bits throughout, with 4480 slots per initial bucket and
//   5632 per later bucket. This removes the early-round bottleneck without overflowing ordinary
//   jobs, cuts the steady B580 solve from about 36.6 to 33.0 ms, and raises Linux throughput from
//   52.26 to 57.94 Sol/s. Keeping its first collision data in the split record head cuts that round
//   further and reaches 59.20 Sol/s. Adding a B580 AOT image to the Linux dpcpp worker raises the
//   measured rate from 59.20 to 60.77 Sol/s. The oneAPI, AdaptiveCpp, and OpenCL workers retain
//   their established layouts; the combined dpcpp fallback images share the vector-checked B13
//   layout because compile-time layout selection applies to the whole translation unit.
//   A bounded 24-hour tuning pass did not prove a B580 hardware/compiler ceiling, so the best-peer
//   gap remains open rather than being attributed to hardware.
// - HIP divides the first four collision rounds into 8/4/4/4 disjoint bin ranges and caches each
//   selected record when that layout fits the device's reported local memory. This preserves every
//   collision and the same overflow retry while raising RX 9060 XT Linux throughput from 42.01 to
//   55.84 Sol/s. CUDA measures the uncached partitioned schedule against one workgroup per bucket;
//   other backends retain the latter. Keeping the last
//   round's parent references bucket-relative reduces its HIP record from 16 to 12 bytes and raises
//   the same card to 57.89 Sol/s. Round three also omits trailing field bits that cannot exclude a
//   valid proof; the final full-proof rehash rejects the extra candidates. Its 12-byte HIP record
//   raises the same card to 61.11 Sol/s. The remaining reference gap is not a proven hardware cap.
// - HIP's fast arena now has 8720 slots per bucket instead of 8704: bucket starts for the 16-byte
//   heads advance by 256 bytes modulo 4 KiB instead of repeating the same offset. This small
//   spacing change reduced RX 9060 XT generation/first-collision times from about 7.7/10.2 to
//   6.1/6.5 ms. It adds 5 MiB across the retained arenas; bounded writes and the 9216-slot safe
//   retry are unchanged. The timing supports an address-layout bottleneck, not a hardware ceiling.
// - The final compute event completes through the shared low-CPU wait before result-copy submission;
//   some runtimes otherwise busy-spin while enqueuing that copy behind live kernels.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

#include "../equihash_pow.hpp"
#include "../lib-internal.h"
#include "../../native/consts.h"
#include "equihash_sycl.hpp"
#include "zhash_session.hpp"

namespace mom_zhash {

class ZHashGenerationTestKernel;
using FastSession = Session<FastZHashArenaLayout>;

// Return a bounded device-generated reference row without allocating the full solver arenas.
static int run_generation_test(const std::string& device_name, const std::uint8_t* input,
                               std::uint8_t* output) {
  std::array<std::uint8_t, ZHashSpec::header_length> header;
  std::array<std::uint8_t, EQUIHASH_ROW_LEN> row_bytes{};
  std::memcpy(header.data(), input, header.size());
  sycl::queue queue{get_dev(device_name)};
  {
    sycl::buffer<std::uint8_t> bytes(row_bytes.data(), sycl::range<1>(row_bytes.size()));
    queue.submit([&](sycl::handler& handler) {
      sycl::accessor result(bytes, handler, sycl::write_only, sycl::no_init);
      handler.single_task<ZHashGenerationTestKernel>([=] {
        mom_equihash::Row<ZHashSpec> row{};
        mom_equihash::row_from_index<ZHashSpec>(header.data(), 0, row);
        for (unsigned field = 0; field < ZHashSpec::rounds; ++field)
          for (unsigned byte = 0; byte < 3; ++byte)
            result[field * 3 + byte] = static_cast<std::uint8_t>(row.fields[field] >> (8 * byte));
        result[18] = result[19] = 0;
      });
    });
  }
  std::memset(output, 0, SMALL_BLOB_SOL_LEN);
  std::memcpy(output, row_bytes.data(), row_bytes.size());
  return 1;
}

class MiningState {
public:
  sycl::queue queue;
  const SessionOptions session_options;
  std::unique_ptr<FastSession> fast_session;
  std::mutex mutex;
  const bool profile;
  unsigned profile_reports = 0;

  static SessionOptions options() {
    SessionOptions value;
    value.collect_stage_times = std::getenv("MOM_ZHASH_PROFILE") != nullptr;
    return value;
  }

  explicit MiningState(const std::string& device_name)
      : queue(get_dev(device_name),
              sycl::property_list{sycl::property::queue::in_order{}}),
        session_options(options()),
        profile(session_options.collect_stage_times) {
    create_fast_session();
  }

  // Retry an overflowing compact solve with the safe layout; only one large arena remains resident.
  RunReport run(const std::uint8_t (&header)[ZHashSpec::header_length],
                const bool force_safe_retry) {
    if (!fast_session)
      create_fast_session();
    RunReport report;
#ifdef MOM_SYCL_ADAPTIVECPP_CUDA
    if (cuda_schedule_ < 0 && !profile && mom_is_cuda(queue.get_device()))
      report = select_cuda_schedule(header);
    else
#endif
      report = fast_session->run(header);
    if (profile && profile_reports < 8)
      std::fprintf(stderr, "ZHash fast overflow=%u,%u,%u,%u,%u bytes=%llu\n",
                   report.overflow_counts[0], report.overflow_counts[1], report.overflow_counts[2],
                   report.overflow_counts[3], report.overflow_counts[4],
                   static_cast<unsigned long long>(report.device_bytes));
    if (report.status == SessionStatus::device_error)
      return report;
    if (report.status != SessionStatus::capacity_overflow && !force_safe_retry)
      return report;

    RunReport fast_report;
    const bool compare_retries = force_safe_retry &&
                                 report.status != SessionStatus::capacity_overflow;
    if (compare_retries)
      fast_report = report;
    fast_session.reset();
    {
      Session<> safe_session(queue, session_options);
      report = safe_session.run(header);
    }
    if (compare_retries && report.status != SessionStatus::device_error &&
        report.status != SessionStatus::capacity_overflow && !same_solutions(fast_report, report)) {
      report.status = SessionStatus::device_error;
      report.error = "ZHash compact and safe layouts disagree";
    }
    if (report.status != SessionStatus::capacity_overflow &&
        report.status != SessionStatus::device_error)
      create_fast_session();
    return report;
  }

private:
  // Session deduplicates canonical proofs, but parallel schedules need not return them in order.
  static bool same_solutions(const RunReport& left, const RunReport& right) {
    return left.status == right.status && left.solutions.size() == right.solutions.size() &&
           std::all_of(left.solutions.begin(), left.solutions.end(), [&](const Solution& solution) {
             return std::any_of(right.solutions.begin(), right.solutions.end(),
                                [&](const Solution& other) { return solution.encoded == other.encoded; });
           });
  }

#ifdef MOM_SYCL_ADAPTIVECPP_CUDA
  int cuda_schedule_ = -1;

  // Warm both existing schedules, then compare ABBA on one header without extra allocations.
  // Return its first result only; probes must agree on every proof and never hide a device fault.
  RunReport select_cuda_schedule(const std::uint8_t (&header)[ZHashSpec::header_length]) {
    RunReport first;
    double seconds[2]{};
    bool valid_timings = true;
    const int schedules[]{0, 1, 0, 1, 1, 0};
    for (unsigned step = 0; step < 6; ++step) {
      const int schedule = schedules[step];
      fast_session->set_cuda_partition(schedule != 0);
      const auto begin = std::chrono::steady_clock::now();
      RunReport report = fast_session->run(header);
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      if (report.status == SessionStatus::device_error)
        return report;
      if (report.status == SessionStatus::capacity_overflow) {
        // Leave bounded recovery to the normal safe-layout retry, with the legacy schedule.
        cuda_schedule_ = 0;
        fast_session->set_cuda_partition(false);
        return report;
      }
      if (step == 0) {
        first = report;
      } else if (!same_solutions(first, report)) {
        report.status = SessionStatus::device_error;
        report.error = "ZHash CUDA scheduling proof mismatch";
        return report;
      }
      if (step >= 2) {
        valid_timings = valid_timings && std::isfinite(elapsed) && elapsed > 0;
        seconds[schedule] += elapsed;
      }
    }
    valid_timings = valid_timings && std::isfinite(seconds[0]) && std::isfinite(seconds[1]);
    cuda_schedule_ = valid_timings && seconds[1] <= seconds[0] / 1.05 ? 1 : 0;
    fast_session->set_cuda_partition(cuda_schedule_ != 0);
    if (valid_timings)
      std::fprintf(stderr, "ZHash: selected %s scheduling (flat %.2f, partitioned %.2f I/s)\n",
                   cuda_schedule_ ? "partitioned" : "flat", 2 / seconds[0], 2 / seconds[1]);
    return first;
  }
#endif

  // Restore the measured fast schedule after the temporary, always-conservative safe retry.
  void create_fast_session() {
    fast_session = std::make_unique<FastSession>(queue, session_options);
#ifdef MOM_SYCL_ADAPTIVECPP_CUDA
    fast_session->set_cuda_partition(cuda_schedule_ == 1);
#endif
  }
};

// Keep persistent sessions under explicit shutdown control to avoid late static SYCL teardown.
static DeviceStateRegistry<MiningState>& registry() {
  static auto* const value = new DeviceStateRegistry<MiningState>;
  return *value;
}

static MiningState& state_for(const std::string& device_name) {
  return registry().get(device_name, [&] {
    return std::make_unique<MiningState>(device_name);
  });
}

// Release cached mining sessions at ordered shutdown without propagating cleanup failures.
void zhash_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "zhash: ordered SYCL cleanup failed\n");
  }
}

} // namespace mom_zhash

// Adapt canonical solver results to the count-prefixed miner ABI, filtering targets only in mining mode.
int zhash(unsigned, std::uint32_t, const std::uint8_t* input, const unsigned input_size,
          std::uint8_t* solution_out, std::uint64_t*, const std::uint8_t* target,
          const unsigned intensity, const bool is_test, const bool is_benchmark,
          const std::string& device_name) {
  if (!input || !solution_out || (!is_test && !is_benchmark && !target) ||
      input_size != mom_zhash::ZHashSpec::header_length || intensity != 1)
    throw std::string(
      "zhash requires input, output, a mining target, a 140-byte header, and intensity 1");
  if (is_test && std::getenv("MOM_ZHASH_SOLVE") == nullptr)
    return mom_zhash::run_generation_test(device_name, input, solution_out);
  auto& state = mom_zhash::state_for(device_name);
  std::lock_guard<std::mutex> lock(state.mutex);
  std::uint8_t header[mom_zhash::ZHashSpec::header_length];
  std::memcpy(header, input, sizeof(header));
  std::chrono::steady_clock::time_point run_start;
  if (state.profile)
    run_start = std::chrono::steady_clock::now();
  auto report = state.run(
      header, is_test && std::getenv("MOM_ZHASH_RETRY_TEST") != nullptr);
  if (state.profile && state.profile_reports++ < 8) {
    const auto run_end = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
    double staged_ms = 0;
    for (const double elapsed : report.stage_ms)
      staged_ms += elapsed;
    std::fprintf(stderr,
                 "ZHash profile wall_ms=%.3f staged_ms=%.3f stage_ms="
                 "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f roots=%u solutions=%zu\n",
                 wall_ms, staged_ms, report.stage_ms[0], report.stage_ms[1], report.stage_ms[2],
                 report.stage_ms[3], report.stage_ms[4], report.stage_ms[5], report.zero_root_count,
                 report.solutions.size());
  }
  if (report.status == mom_zhash::SessionStatus::capacity_overflow ||
      report.status == mom_zhash::SessionStatus::device_error)
    throw report.error.empty() ? std::string("zhash solver failed") : report.error;

  if (is_test)
    std::sort(report.solutions.begin(), report.solutions.end(),
              [](const auto& left, const auto& right) {
                return left.encoded < right.encoded;
              });
  if (is_test)
    std::memset(solution_out, 0, SMALL_BLOB_SOL_LEN);
  const std::size_t capacity = std::min<std::size_t>(
      255, (EQUIHASH_SOL_BUFFER_LEN - 1) / mom_zhash::ZHashSpec::solution_length);
  unsigned count = 0;
  for (const auto& solution : report.solutions) {
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
