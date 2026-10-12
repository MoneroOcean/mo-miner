#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fixture_clock {
static double seconds;
struct steady_clock {
  static std::chrono::duration<double> now() { return std::chrono::duration<double>(seconds); }
};
using std::chrono::duration;
}
struct ProductionArena {};
struct RetryArena {};
struct HybridArena {};
struct Spec { static constexpr unsigned header_length = 1; };
struct Solution { std::array<unsigned char, 2> encoded; };
struct BucketArenaOverflow : std::runtime_error {
  BucketArenaOverflow() : std::runtime_error("bucket overflow") {}
};
struct Device { bool cuda = true; };
struct Queue { Device device; Device get_device() const { return device; } };
[[maybe_unused]] static bool mom_is_cuda(Device device) { return device.cuda; }
static void require(bool value, const char* message) {
  if (!value) throw std::logic_error(message);
}

template <typename Arena = ProductionArena> struct Session {
  Queue queue_;
  bool level_zero_ = false;
  int cuda_arena_ = -1;
  std::vector<int> calls;
  double elapsed[3]{1, .9, 1};
  int fault_call = -1, second_fault_call = -1, fault_kind = 0, proof_call = -1;
  bool count_mismatch = false, empty_proofs = false;
  template <typename Layout, bool CacheInput>
  std::vector<Solution> run_layout(const std::uint8_t (&)[Spec::header_length]) {
    static_assert(!CacheInput, "fixture exercises the CUDA uncached path");
    const int layout = std::is_same_v<Layout, RetryArena> ? 1 :
                       std::is_same_v<Layout, HybridArena> ? 2 : 0;
    calls.push_back(layout);
    fixture_clock::seconds += elapsed[layout];
    if (static_cast<int>(calls.size()) == fault_call ||
        static_cast<int>(calls.size()) == second_fault_call) {
      if (fault_kind == 0) throw BucketArenaOverflow();
      throw std::runtime_error(fault_kind == 1 ? "GPU runtime fault" : "root capacity fault");
    }
    if (empty_proofs) return {};
    std::vector<Solution> result{{{{1, 2}}}, {{{3, 4}}}};
    if (static_cast<int>(calls.size()) == proof_call) {
      if (count_mismatch) result.pop_back();
      else result[1].encoded[1] ^= 1;
    }
    return result;
  }
#include "run_with_cache.inc"
#ifdef MOM_SYCL_ADAPTIVECPP_CUDA
#include "select_cuda_arena.inc"
#endif
};

static const std::uint8_t header[1]{};
static void clear_environment() {
  for (const auto* name : {"MOM_EQUIHASH192_7_HYBRID_ARENA", "MOM_EQUIHASH192_7_RETRY_ARENA",
                           "MOM_EQUIHASH192_7_PROFILE"}) unsetenv(name);
}
static void expect_calls(const Session<>& session, std::initializer_list<int> expected) {
  require(session.calls == std::vector<int>(expected), "layout dispatch order");
}
[[maybe_unused]] static void expect_error(Session<>& session, const char* expected) {
  try { session.run_with_cache<false>(header); }
  catch (const std::runtime_error& error) {
    require(std::string(error.what()) == expected, "fault was replaced");
    return;
  }
  throw std::logic_error("expected propagated error");
}

static void run_checks() {
  clear_environment();
  Session<> plain;
  plain.run_with_cache<false>(header);
#ifdef MOM_SYCL_ADAPTIVECPP_CUDA
  expect_calls(plain, {0, 1, 0, 1, 1, 0});
  require(plain.cuda_arena_ == 1, "material gain selects retry");
  plain.run_with_cache<false>(header);
  expect_calls(plain, {0, 1, 0, 1, 1, 0, 1});
  Session<> empty;
  empty.empty_proofs = true;
  require(empty.run_with_cache<false>(header).empty(), "empty proof set was changed");
  expect_calls(empty, {0, 1, 0, 1, 1, 0});
  require(empty.cuda_arena_ == 1, "empty proof set prevented calibration");
  for (double timing : {1.0, .99}) {
    Session<> noise;
    noise.elapsed[1] = timing;
    noise.run_with_cache<false>(header);
    require(noise.cuda_arena_ == 0, "noise must retain production");
    noise.run_with_cache<false>(header);
    expect_calls(noise, {0, 1, 0, 1, 1, 0, 0});
  }
  for (int step = 2; step <= 6; ++step) {
    for (bool count : {false, true}) {
      Session<> mismatch;
      mismatch.proof_call = step;
      mismatch.count_mismatch = count;
      expect_error(mismatch, "Equihash(192,7) CUDA arena proof mismatch");
      require(mismatch.cuda_arena_ == -1, "failed proof comparison cached a choice");
      require(mismatch.calls.size() == static_cast<unsigned>(step), "mismatching proof was retried");
    }
  }
  for (int step = 1; step <= 6; ++step) {
    Session<> overflow;
    overflow.fault_call = step;
    overflow.run_with_cache<false>(header);
    require(overflow.calls.size() == static_cast<unsigned>(step + 1), "overflow continued calibration");
    require(overflow.calls.back() == 1 - overflow.calls[step - 1], "calibration fallback arena");
    require(overflow.cuda_arena_ == overflow.calls.back(), "fallback choice not cached");
    overflow.run_with_cache<false>(header);
    require(overflow.calls.back() == overflow.cuda_arena_, "fallback choice not reused");
  }
  Session<> both_overflow;
  both_overflow.fault_call = 1;
  both_overflow.second_fault_call = 2;
  expect_error(both_overflow, "bucket overflow");
  require(both_overflow.cuda_arena_ == -1, "both overflowing arenas cached a choice");
  expect_calls(both_overflow, {0, 1});
  for (int choice : {0, 1}) {
    Session<> later;
    later.cuda_arena_ = choice;
    later.fault_call = 1;
    later.run_with_cache<false>(header);
    expect_calls(later, {choice, 1 - choice});
    Session<> exhausted;
    exhausted.cuda_arena_ = choice;
    exhausted.fault_call = 1;
    exhausted.second_fault_call = 2;
    expect_error(exhausted, "bucket overflow");
    expect_calls(exhausted, {choice, 1 - choice});
    require(exhausted.cuda_arena_ == choice, "failed fallback changed cached choice");
    for (int kind : {1, 2}) {
      Session<> fault;
      fault.cuda_arena_ = choice;
      fault.fault_call = 1;
      fault.fault_kind = kind;
      expect_error(fault, kind == 1 ? "GPU runtime fault" : "root capacity fault");
      require(fault.calls.size() == 1, "non-bucket fault retried");
    }
  }
  for (int kind : {1, 2}) {
    Session<> fault;
    fault.fault_call = 2;
    fault.fault_kind = kind;
    expect_error(fault, kind == 1 ? "GPU runtime fault" : "root capacity fault");
    require(fault.calls.size() == 2 && fault.cuda_arena_ == -1, "calibration fault masked");
  }
#else
  expect_calls(plain, {0});
#endif
  Session<> non_cuda;
  non_cuda.queue_.device.cuda = false;
  non_cuda.run_with_cache<false>(header);
  expect_calls(non_cuda, {0});
  for (const auto* name : {"MOM_EQUIHASH192_7_PROFILE", "MOM_EQUIHASH192_7_RETRY_ARENA",
                           "MOM_EQUIHASH192_7_HYBRID_ARENA"}) {
    setenv(name, "0", 1); // These diagnostic flags are intentionally presence-sensitive.
    Session<> diagnostic;
    diagnostic.run_with_cache<false>(header);
    const int expected = std::string(name).find("HYBRID") != std::string::npos ? 2 :
                         std::string(name).find("RETRY") != std::string::npos ? 1 : 0;
    expect_calls(diagnostic, {expected});
    require(diagnostic.cuda_arena_ == -1, "diagnostic changed automatic choice");
    clear_environment();
  }
  setenv("MOM_EQUIHASH192_7_HYBRID_ARENA", "1", 1);
  setenv("MOM_EQUIHASH192_7_RETRY_ARENA", "1", 1);
  Session<> priority;
  priority.run_with_cache<false>(header);
  expect_calls(priority, {2});
  clear_environment();
  Session<RetryArena> retry_session;
  retry_session.run_with_cache<false>(header);
  require(retry_session.calls == std::vector<int>{1}, "non-production session calibrated");
  std::puts("PASS actual-source Equihash192 arena selection");
}

int main() {
  try { run_checks(); }
  catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
