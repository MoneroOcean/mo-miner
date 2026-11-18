// Host-only coverage for PearlHash capacity admission and shared matrix allocation.
#include "pow-intensity.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <string>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace sycl {

enum class errc { memory_allocation, feature_not_supported, invalid, runtime };

class error_category final : public std::error_category {
 public:
  const char* name() const noexcept override { return "fixture-sycl"; }
  std::string message(const int value) const override {
    return "fixture error " + std::to_string(value);
  }
};

inline const std::error_category& sycl_category() noexcept {
  static const error_category category;
  return category;
}

inline std::error_code make_error_code(const errc error) noexcept {
  return {static_cast<int>(error), sycl_category()};
}

struct exception : std::exception {
  explicit exception(const errc error = errc::memory_allocation) : error_(make_error_code(error)) {}
  const std::error_code& code() const noexcept { return error_; }
  const char* what() const noexcept override { return "fixture SYCL allocation failure"; }

 private:
  std::error_code error_;
};

} // namespace sycl

namespace std {
template <> struct is_error_code_enum<sycl::errc> : true_type {};
} // namespace std

namespace fixture_allocator {

enum class Action {
  success,
  null_result,
  memory_allocation,
  feature_not_supported,
  invalid,
  runtime,
  unrelated_error,
};

inline Action plan[2] = {Action::success, Action::success};
inline unsigned calls = 0;
inline std::vector<void*> freed;
alignas(std::max_align_t) inline std::array<std::array<std::byte, 64>, 2> tokens{};

inline void reset(const Action first, const Action second) {
  plan[0] = first;
  plan[1] = second;
  calls = 0;
  freed.clear();
}

template <typename T> T* allocate() {
  const unsigned call = calls++;
  const Action action = call < 2 ? plan[call] : Action::success;
  if (action == Action::null_result)
    return nullptr;
  if (action == Action::memory_allocation)
    throw sycl::exception{sycl::errc::memory_allocation};
  if (action == Action::feature_not_supported)
    throw sycl::exception{sycl::errc::feature_not_supported};
  if (action == Action::invalid)
    throw sycl::exception{sycl::errc::invalid};
  if (action == Action::runtime)
    throw sycl::exception{sycl::errc::runtime};
  if (action == Action::unrelated_error)
    throw std::runtime_error("fixture allocator failure");
  return reinterpret_cast<T*>(tokens[call].data());
}

} // namespace fixture_allocator

namespace sycl {

struct device {
  bool is_cpu() const { return false; }
};

struct queue {
  device get_device() const { return {}; }
};

template <typename T> T* malloc_shared(const std::size_t, const queue&) {
  return fixture_allocator::allocate<T>();
}

template <typename T> T* malloc_device(const std::size_t, const queue&) {
  return fixture_allocator::allocate<T>();
}

inline void free(void* const pointer, const queue&) noexcept {
  fixture_allocator::freed.push_back(pointer);
}

} // namespace sycl

struct Result {};

struct Buffers {
  int proof_rows = 16, proof_cols = 16;
  int8_t *EAL = nullptr, *EBR = nullptr, *EBRt = nullptr, *Ap = nullptr, *Bp = nullptr;
  int32_t *EARp1 = nullptr, *EARp2 = nullptr, *EBLq1 = nullptr, *EBLq2 = nullptr;
  uint8_t *cA = nullptr, *cB = nullptr, *key = nullptr, *target = nullptr, *CVA = nullptr,
          *CVB = nullptr;
  uint32_t* transcript = nullptr;
  int transcript_rows = 0;
  Result* result = nullptr;
};

#include "pearlhash_allocation_policy.inc"
#include "pearlhash_native_allocation.inc"

static bool hip_layout_admitted(const Buffers& b, const int) {
#include "hip-layout-admission.inc"
}

namespace {

void require(const bool condition, const char* const message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

void require_freed_exact(const std::initializer_list<void*> expected, const char* const message) {
  require(fixture_allocator::freed.size() == expected.size(), message);
  for (void* const pointer : expected) {
    size_t occurrences = 0;
    for (void* const freed : fixture_allocator::freed)
      occurrences += freed == pointer;
    require(occurrences == 1, message);
  }
  for (void* const freed : fixture_allocator::freed) {
    bool expected_pointer = false;
    for (void* const pointer : expected)
      expected_pointer = expected_pointer || freed == pointer;
    require(expected_pointer, message);
  }
}

void expect_optional_transcript(const fixture_allocator::Action action, const bool expected_success,
                                const bool expected_sycl_propagated, const bool expected_unrelated) {
  fixture_allocator::reset(action, fixture_allocator::Action::success);
  sycl::queue queue;
  Buffers b{};
  b.proof_rows = 4;
  b.proof_cols = 64;
  alignas(std::max_align_t) std::array<std::byte, 1> portable_a_token{}, portable_b_token{};
  b.Ap = reinterpret_cast<int8_t*>(portable_a_token.data());
  b.Bp = reinterpret_cast<int8_t*>(portable_b_token.data());
  int8_t* const portable_a = b.Ap;
  int8_t* const portable_b = b.Bp;
  bool sycl_propagated = false, unrelated = false;
  try {
    b.transcript = alloc_optional<uint32_t>(queue, 1);
  } catch (const sycl::exception&) {
    sycl_propagated = true;
  } catch (const std::runtime_error&) {
    unrelated = true;
  } catch (...) {
    require(false, "optional transcript allocation changed the propagated exception type");
  }
  require(sycl_propagated == expected_sycl_propagated,
          "optional transcript SYCL error-code policy changed");
  require(unrelated == expected_unrelated, "optional transcript exception policy changed");
  require((b.transcript != nullptr) == expected_success,
          "optional transcript allocation result changed");
  require(b.Ap == portable_a && b.Bp == portable_b,
          "optional transcript allocation changed shared matrices");
  require(b.proof_rows == 4 && b.proof_cols == 64,
          "optional transcript allocation changed proof geometry");
  require(fixture_allocator::calls == 1, "optional transcript allocator call count changed");
  if (expected_success) {
    require(b.transcript == reinterpret_cast<uint32_t*>(fixture_allocator::tokens[0].data()),
            "optional transcript returned the wrong pointer");
  }
  // This is the same owner cleanup used by the state object. It must free the shared matrices
  // exactly once whether the optional allocation declined, succeeded, or propagated a fault.
  free_allocations(queue, b);
  if (expected_success)
    require_freed_exact({portable_a_token.data(), portable_b_token.data(),
                         fixture_allocator::tokens[0].data()},
                        "shared matrices/transcript were not freed exactly once");
  else
    require_freed_exact({portable_a_token.data(), portable_b_token.data()},
                        "failed optional allocation changed shared cleanup");
  require(b.Ap == nullptr && b.Bp == nullptr && b.transcript == nullptr,
          "shared cleanup left an allocation pointer live");
}

} // namespace

int main() {
  constexpr uint64_t reserve = pearlhash_runtime_reserve_bytes;
  const pearlhash_profile_t shape = {131072, 131072, 2048, 128};
  PearlHashAllocationCounts portable{};
  require(pearlhash_allocation_counts(shape, portable), "ordinary shape was rejected");

  const uint64_t required_limit = portable.total_bytes + reserve;
  require(pearlhash_required_layout_admitted({1, required_limit, portable.max_alloc}, portable),
          "exact required capacity was rejected");
  require(!pearlhash_required_layout_admitted(
              {1, required_limit - 1, portable.max_alloc}, portable),
          "oversize required layout was admitted");
  require(!pearlhash_required_layout_admitted(
              {1, required_limit, portable.max_alloc - 1}, portable),
          "per-allocation limit was ignored");

  // Unknown limits do not waive a different limit that the runtime did report.
  require(!pearlhash_required_layout_admitted({1, 0, portable.max_alloc - 1}, portable),
          "known per-allocation limit was ignored when global memory was unknown");
  require(!pearlhash_required_layout_admitted({1, required_limit - 1, 0}, portable),
          "known global limit was ignored when max allocation was unknown");
  require(pearlhash_required_layout_admitted({1, 0, portable.max_alloc}, portable),
          "unknown global capacity stopped an otherwise admitted layout");
  require(pearlhash_required_layout_admitted({1, required_limit, 0}, portable),
          "unknown per-allocation capacity stopped an otherwise admitted layout");
  require(pearlhash_required_layout_admitted({1, 0, 0}, portable),
          "both unknown capacity limits changed existing behavior");

  Buffers layout{};
  require(!hip_layout_admitted(layout, 1024),
          "HIP launch guard admitted a layout with no native matrices");
  int8_t native_a = 0, native_b = 0;
  uint32_t transcript = 0;
  layout.Ap = &native_a;
  layout.transcript = &transcript;
  layout.transcript_rows = 512;
  require(!hip_layout_admitted(layout, 1024),
          "HIP launch guard admitted a layout with no native B matrix");
  layout.Bp = &native_b;
  layout.Ap = nullptr;
  require(!hip_layout_admitted(layout, 1024),
          "HIP launch guard admitted a layout with no native A matrix");
  layout.Ap = &native_a;
  layout.transcript = nullptr;
  layout.transcript_rows = 0;
  require(hip_layout_admitted(layout, 1024),
          "HIP launch guard retained an unnecessary transcript requirement");

  expect_optional_transcript(fixture_allocator::Action::null_result, false, false, false);
  expect_optional_transcript(fixture_allocator::Action::memory_allocation, false, false, false);
  expect_optional_transcript(fixture_allocator::Action::feature_not_supported, false, false, false);
  expect_optional_transcript(fixture_allocator::Action::success, true, false, false);
  expect_optional_transcript(fixture_allocator::Action::invalid, false, true, false);
  expect_optional_transcript(fixture_allocator::Action::runtime, false, true, false);
  expect_optional_transcript(fixture_allocator::Action::unrelated_error, false, false, true);

  std::puts("PearlHash allocation admission and native fallback passed");
}
