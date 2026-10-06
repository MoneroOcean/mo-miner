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
//   55.84 Sol/s; other backends retain the simpler one-workgroup-per-bucket path. Keeping the last
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

  RunReport run(const std::uint8_t (&header)[ZHashSpec::header_length],
                const bool force_safe_retry) {
    if (!fast_session)
      create_fast_session();
    RunReport report = fast_session->run(header);
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
    if (compare_retries) {
      bool matches = fast_report.status == report.status &&
                     fast_report.solutions.size() == report.solutions.size();
      for (const Solution& fast_solution : fast_report.solutions) {
        matches = matches && std::any_of(
            report.solutions.begin(), report.solutions.end(), [&](const Solution& safe_solution) {
              return fast_solution.encoded == safe_solution.encoded;
            });
      }
      if (!matches) {
        report.status = SessionStatus::device_error;
        report.error = "ZHash compact and safe layouts disagree";
      }
    }
    if (report.status != SessionStatus::capacity_overflow &&
        report.status != SessionStatus::device_error)
      create_fast_session();
    return report;
  }

private:
  void create_fast_session() {
    fast_session = std::make_unique<FastSession>(queue, session_options);
  }
};

static DeviceStateRegistry<MiningState>& registry() {
  static auto* const value = new DeviceStateRegistry<MiningState>;
  return *value;
}

static MiningState& state_for(const std::string& device_name) {
  return registry().get(device_name, [&] {
    return std::make_unique<MiningState>(device_name);
  });
}

void zhash_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "zhash: ordered SYCL cleanup failed\n");
  }
}

} // namespace mom_zhash

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
