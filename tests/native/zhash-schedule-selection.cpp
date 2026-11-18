#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct ZHashSpec {
  static constexpr unsigned header_length = 1, proof_indices = 2, solution_length = 2;
};
#include "reports.inc"
static void require(bool condition, const char* message) {
  if (!condition) throw std::logic_error(message);
}
namespace fixture_clock {
static unsigned reads;
static std::array<double, 6> elapsed;
struct steady_clock {
  static std::chrono::duration<double> now() {
    const unsigned read = reads++;
    require(read < 12, "calibration repeated its timing loop");
    return std::chrono::duration<double>(read % 2 ? elapsed[read / 2] : 0);
  }
};
using std::chrono::duration;
}
struct Control {
  std::vector<int> calls;
  unsigned fast_created = 0, active_fast = 0, active_safe = 0, fast_calls = 0;
  int fault_call = -1, throw_call = -1, mismatch_call = -1, mismatch_kind = 0;
  SessionStatus fault = SessionStatus::device_error, safe_status = SessionStatus::solved;
  bool safe_mismatch = false, safe_throw = false, empty = false;
  RunReport result(bool partition, bool safe) {
    calls.push_back(safe ? 2 : partition ? 1 : 0);
    const unsigned call = safe ? 0 : ++fast_calls;
    if ((!safe && static_cast<int>(call) == throw_call) || (safe && safe_throw))
      throw std::runtime_error(safe ? "safe runtime fault" : "fast runtime fault");
    RunReport report;
    report.status = empty ? SessionStatus::no_solution : SessionStatus::solved;
    report.device_bytes = 1000 + call;
    report.root_count = call;
    report.flat_counts.fill(call);
    report.stage_ms.fill(call);
    if (!empty) {
      Solution a{}, b{};
      a.encoded = {1, 2}; b.encoded = {3, 4};
      report.solutions = partition ? std::vector<Solution>{b, a} : std::vector<Solution>{a, b};
    }
    if ((!safe && static_cast<int>(call) == mismatch_call) || (safe && safe_mismatch)) {
      if (mismatch_kind == 0) report.solutions.back().encoded[1] ^= 1;
      else if (mismatch_kind == 1) report.solutions.pop_back();
      else report.status = SessionStatus::no_solution;
    }
    if (safe || static_cast<int>(call) == fault_call) {
      report.status = safe ? safe_status : fault;
      if (report.status == SessionStatus::device_error || report.status == SessionStatus::capacity_overflow)
        report.error = safe ? "safe reported fault" : "fast reported fault";
    }
    return report;
  }
};
struct Device { bool cuda; };
struct Queue {
  Control* control;
  bool cuda;
  Device get_device() const { return {cuda}; }
};
[[maybe_unused]] static bool mom_is_cuda(Device device) { return device.cuda; }
struct SessionOptions {};
struct FastSession {
  Control& control;
  bool cuda_partition_ = false;
  FastSession(Queue queue, const SessionOptions&) : control(*queue.control) {
    require(!control.active_fast && !control.active_safe, "overlapping arenas");
    ++control.fast_created; ++control.active_fast;
  }
  ~FastSession() { --control.active_fast; }
#ifdef MOM_SYCL_ADAPTIVECPP_CUDA
#include "set_cuda_partition.inc"
#endif
  RunReport run(const std::uint8_t (&header)[ZHashSpec::header_length]) {
    require(header[0] == 0x3a, "calibration changed its header");
    return control.result(cuda_partition_, false);
  }
};
template <typename = void> struct Session {
  Control& control;
  Session(Queue queue, const SessionOptions&) : control(*queue.control) {
    require(!control.active_fast && !control.active_safe, "safe allocation before fast release");
    ++control.active_safe;
  }
  ~Session() { --control.active_safe; }
  RunReport run(const std::uint8_t (&header)[ZHashSpec::header_length]) {
    require(header[0] == 0x3a, "retry changed its header");
    return control.result(false, true);
  }
};
struct MiningState {
  Control control;
  Queue queue;
  SessionOptions session_options;
  std::unique_ptr<FastSession> fast_session;
  bool profile;
  unsigned profile_reports = 0;
  int cuda_schedule_ = -1;
  MiningState(bool profiling = false, bool cuda = true) : queue{&control, cuda}, profile(profiling) {
    fixture_clock::reads = 0;
    fixture_clock::elapsed = {1, .9, 1, .9, .9, 1};
    create_fast_session();
  }
#include "same_solutions.inc"
#include "run.inc"
#ifdef MOM_SYCL_ADAPTIVECPP_CUDA
#include "select_cuda_schedule.inc"
#endif
#include "create_fast_session.inc"
};
static const std::uint8_t header[]{0x3a};
static void expect_calls(const MiningState& state, std::initializer_list<int> expected) {
  require(state.control.calls == std::vector<int>(expected), "unexpected scheduling/retry order");
}
static void expect_runtime_fault(MiningState& state, bool safe = false) {
  try { state.run(header, safe); }
  catch (const std::runtime_error& error) {
    require(std::string(error.what()) == (safe ? "safe runtime fault" : "fast runtime fault"),
            "runtime fault replaced");
    return;
  }
  throw std::logic_error("runtime fault masked");
}
static void checks() {
  MiningState base;
  const auto first = base.run(header, false);
  require(first.root_count == 1 && first.device_bytes == 1001 && first.flat_counts[0] == 1 &&
          first.stage_ms[0] == 1 && first.solutions.front().encoded[0] == 1,
          "selector did not return the original flat report");
#ifdef MOM_SYCL_ADAPTIVECPP_CUDA
  expect_calls(base, {0, 1, 0, 1, 1, 0});
  require(base.cuda_schedule_ == 1 && base.fast_session->cuda_partition_, "material gain ignored");
  require(base.control.fast_created == 1, "calibration allocated another arena");
  base.run(header, false);
  expect_calls(base, {0, 1, 0, 1, 1, 0, 1});
  for (double flat : {1.0, 1.0499, 1.05}) {
    MiningState threshold;
    fixture_clock::elapsed = {99, .01, flat, 1, 1, flat};
    threshold.run(header, false);
    require(threshold.cuda_schedule_ == (flat == 1.05 ? 1 : 0), "five-percent boundary or warmup bias");
  }
  for (int step = 2; step <= 6; ++step) for (int kind = 0; kind < 3; ++kind) {
    MiningState mismatch;
    mismatch.control.mismatch_call = step; mismatch.control.mismatch_kind = kind;
    const auto report = mismatch.run(header, false);
    require(report.status == SessionStatus::device_error &&
            report.error == "ZHash CUDA scheduling proof mismatch", "full proof/status mismatch ignored");
    require(mismatch.cuda_schedule_ == -1 && mismatch.control.fast_calls == static_cast<unsigned>(step),
            "proof mismatch cached or continued");
  }
  for (unsigned step = 2; step < 6; ++step)
    for (double invalid : {0., -1., std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
      MiningState timing;
      fixture_clock::elapsed[step] = invalid;
      timing.run(header, false);
      require(timing.cuda_schedule_ == 0 && !timing.fast_session->cuda_partition_, "invalid timing selected partition");
    }
  MiningState sum_overflow;
  fixture_clock::elapsed.fill(std::numeric_limits<double>::max());
  sum_overflow.run(header, false);
  require(sum_overflow.cuda_schedule_ == 0, "nonfinite timing sum selected partition");
  MiningState empty;
  empty.control.empty = true;
  require(empty.run(header, false).status == SessionStatus::no_solution && empty.cuda_schedule_ == 1,
          "matching empty proof sets rejected");
  for (int step = 1; step <= 6; ++step) {
    MiningState device;
    device.control.fault_call = step;
    const auto report = device.run(header, true);
    require(report.status == SessionStatus::device_error && report.error == "fast reported fault",
            "device fault masked by safe retry");
    require(device.cuda_schedule_ == -1 && device.control.fast_calls == static_cast<unsigned>(step),
            "device fault cached or continued");
    MiningState overflow;
    overflow.control.fault_call = step; overflow.control.fault = SessionStatus::capacity_overflow;
    require(overflow.run(header, false).status == SessionStatus::solved, "bounded overflow did not retry");
    require(overflow.cuda_schedule_ == 0 && !overflow.fast_session->cuda_partition_ &&
            overflow.control.fast_calls == static_cast<unsigned>(step) && overflow.control.calls.back() == 2,
            "overflow did not stop calibration with conservative retry");
    overflow.run(header, false);
    require(overflow.control.calls.back() == 0, "overflow fallback choice not cached");
    MiningState thrown;
    thrown.control.throw_call = step;
    expect_runtime_fault(thrown);
    require(thrown.cuda_schedule_ == -1 && thrown.control.fast_calls == static_cast<unsigned>(step),
            "runtime fault cached or retried");
  }
#else
  expect_calls(base, {0});
#endif
  for (bool profile : {false, true}) {
    MiningState bypass(profile, profile);
    bypass.run(header, false);
    expect_calls(bypass, {0});
    require(bypass.cuda_schedule_ == -1, "profile/non-CUDA calibrated");
  }
  for (int choice : {0, 1}) {
    MiningState later;
    later.cuda_schedule_ = choice; later.fast_session.reset(); later.create_fast_session();
#ifdef MOM_SYCL_ADAPTIVECPP_CUDA
    const int effective_choice = choice;
    require(later.fast_session->cuda_partition_ == (choice == 1), "recreated session lost preference");
#else
    const int effective_choice = 0;
#endif
    later.control.fault_call = 1; later.control.fault = SessionStatus::capacity_overflow;
    later.run(header, false);
    expect_calls(later, {effective_choice, 2});
    require(later.fast_session != nullptr, "successful safe retry failed to restore fast session");
    later.run(header, false);
    require(later.control.calls.back() == effective_choice, "safe retry lost cached schedule");
    MiningState device;
    device.cuda_schedule_ = choice; device.fast_session.reset(); device.create_fast_session();
    device.control.fault_call = 1;
    const auto device_report = device.run(header, true);
    require(device_report.status == SessionStatus::device_error && device_report.error == "fast reported fault" &&
            device.cuda_schedule_ == choice, "cached device fault changed preference or was masked");
    expect_calls(device, {effective_choice});
    MiningState runtime;
    runtime.cuda_schedule_ = choice; runtime.fast_session.reset(); runtime.create_fast_session();
    runtime.control.throw_call = 1;
    expect_runtime_fault(runtime);
    expect_calls(runtime, {effective_choice});
    for (auto status : {SessionStatus::device_error, SessionStatus::capacity_overflow}) {
      MiningState failure;
      failure.cuda_schedule_ = choice; failure.fast_session.reset(); failure.create_fast_session();
      failure.control.safe_status = status;
      const auto report = failure.run(header, true);
      require(report.status == status && report.error == "safe reported fault", "safe fault overwritten");
      expect_calls(failure, {effective_choice, 2});
      require(!failure.fast_session, "safe failure restored a fast session");
    }
  }
  MiningState safe_thrown(true);
  safe_thrown.control.safe_throw = true;
  expect_runtime_fault(safe_thrown, true);
  for (bool mismatch : {false, true}) {
    MiningState forced(true);
    forced.control.safe_mismatch = mismatch;
    const auto report = forced.run(header, true);
    require(report.status == (mismatch ? SessionStatus::device_error : SessionStatus::solved),
            "forced safe comparison failed");
    if (mismatch) require(report.error == "ZHash compact and safe layouts disagree", "wrong proof mismatch error");
    expect_calls(forced, {0, 2});
  }
  std::puts("PASS actual-source ZHash scheduling selection");
}
int main() {
  try { checks(); }
  catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
