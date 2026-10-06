// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../../native/consts.h"

static unsigned assertions = 0;
static void require(bool value, const char *label) {
  ++assertions;
  if (value) return;
  std::fprintf(stderr, "ZelHash state regression: %s\n", label);
  std::abort();
}
static std::map<void *, size_t> live;
static unsigned allocations = 0, fail_allocation = 0, waits = 0, memsets = 0;
static bool throw_allocation = false, fail_wait = false, fail_memset = false;
namespace sycl {
namespace info::device {
struct global_mem_size {};
struct max_mem_alloc_size {};
}
namespace property::queue {
struct in_order {};
}
struct property_list {
  template <class T> property_list(T) {}
};
struct device {
  uint64_t memory = 16ull << 30, max_allocation = 4ull << 30;
  bool compact = false;
  bool is_cpu() const { return false; }
  template <class Info> uint64_t get_info() const {
    if constexpr (std::is_same_v<Info, info::device::global_mem_size>)
      return memory;
    else
      return max_allocation;
  }
};
struct event {};
struct queue {
  device d;
  queue(device selected, property_list) : d(selected) {}
  void wait_and_throw() {
    ++waits;
    if (fail_wait) throw std::runtime_error("wait failure");
  }
  event memset(void *, int, size_t) {
    ++memsets;
    if (fail_memset) throw std::runtime_error("memset failure");
    return {};
  }
};
template <class T> T *malloc_shared(size_t count, queue &) {
  ++allocations;
  if (allocations == fail_allocation) {
    if (throw_allocation) throw std::runtime_error("allocation failure");
    return nullptr;
  }
  // The boundary records production sizes but allocates only bookkeeping bytes, never GPU arenas.
  void *result = std::malloc(16);
  require(result != nullptr, "fixture allocator");
  live.emplace(result, count * sizeof(T));
  return static_cast<T *>(result);
}
template <class T> T *malloc_device(size_t count, queue &q) { return malloc_shared<T>(count, q); }
template <class T> void free(T *pointer, queue &) {
  require(live.erase(pointer) == 1, "each allocation released exactly once");
  std::free(pointer);
}
}
static sycl::device configured;
static sycl::device get_dev(const std::string &) { return configured; }
static bool mom_is_cuda(const sycl::device &d) { return d.compact; }
static bool mom_is_hip(const sycl::device &) { return false; }
static bool mom_has_usm_device(const sycl::device &) { return true; }
static bool mom_has_usm_shared(const sycl::device &) { return true; }
template <class F> static void sycl_cleanup_noexcept(const char *, F action) noexcept {
  try {
    action();
  } catch (...) {
  }
}
static void sycl_wait_and_throw(sycl::event, const sycl::device &) {
  if (fail_memset) throw std::runtime_error("event failure");
}
#include "ulong_parser.inc"
#include "zelhash_layout_math.inc"
struct EqCandidateRef {
  uint32_t bucket, slot_a, slot_b;
};
struct EqCandidate {
  uint32_t leaves[1u << K];
};
#if defined(MOM_EQ_FORWARD_MAP)
constexpr uint32_t L0_INVERT_HEADS = NBUCKETS;
#endif
#include "zelhash_state.inc"
} // namespace mom_zelhash
using namespace mom_zelhash;

static void reset_failure(unsigned fail = 0, bool throws = false) {
  allocations = 0;
  fail_allocation = fail;
  throw_allocation = throws;
  fail_memset = false;
  fail_wait = false;
}
static bool entry_rejects([[maybe_unused]] const uint8_t *input,
                          [[maybe_unused]] uint8_t *solution_out,
                          [[maybe_unused]] const uint8_t *target, unsigned input_size,
                          [[maybe_unused]] unsigned intensity, [[maybe_unused]] bool is_test,
                          [[maybe_unused]] bool is_benchmark) {
  try {
    {
#include "zelhash_entry_guard.inc"
    }
    return false;
  } catch (const std::string &) {
    return true;
  }
}
static void test_entry() {
  uint8_t bytes[140] = {};
  for (bool test : {false, true})
    for (bool bench : {false, true}) {
      require(!entry_rejects(bytes, bytes, bytes, 140, 1, test, bench), "valid entry accepted");
      require(entry_rejects(nullptr, bytes, bytes, 140, 1, test, bench), "null input rejected");
      require(entry_rejects(bytes, nullptr, bytes, 140, 1, test, bench), "null output rejected");
      require(entry_rejects(bytes, bytes, nullptr, 140, 1, test, bench) == (!test && !bench),
              "target contract");
      for (unsigned size : {0u, 139u, 141u, UINT32_MAX})
        require(entry_rejects(bytes, bytes, bytes, size, 1, test, bench), "exact header length");
      for (unsigned intensity : {0u, 2u, UINT32_MAX})
        require(entry_rejects(bytes, bytes, bytes, 140, intensity, test, bench),
                "single-solve intensity");
    }
}
static void test_capacity() {
  unsetenv("MOM_ZELHASH_SLOTS");
  const unsigned mean = ceil_div_u64(NUM_ENTRIES, NBUCKETS);
  const unsigned minimum = round_up(mean + ceil_sqrt(mean), 16);
  sycl::device d;
  require(zelhash_slot_capacity(d) == MAX_SLOTS, "large-device capacity");
  const uint64_t bytes_per_slot = uint64_t(NBUCKETS) * level_u32(0, false) * 4;
  d.max_allocation = bytes_per_slot * minimum;
  require(zelhash_slot_capacity(d) == minimum, "exact allocation minimum");
  d.max_allocation = bytes_per_slot * minimum - 1;
  bool rejected = false;
  try {
    (void)zelhash_slot_capacity(d);
  } catch (const std::string &) {
    rejected = true;
  }
  require(rejected, "below allocation minimum rejected");
  d = {};
  for (const char *value : {"junk", "1x", "0", "-1", "999999999999999999999999999999999999", "1"}) {
    setenv("MOM_ZELHASH_SLOTS", value, 1);
    rejected = false;
    try {
      (void)zelhash_slot_capacity(d);
    } catch (const std::string &) {
      rejected = true;
    }
    require(rejected && allocations == 0 && live.empty(),
            "malformed or out-of-range capacity rejected without allocation");
  }
  const std::string valid = std::to_string(minimum);
  setenv("MOM_ZELHASH_SLOTS", valid.c_str(), 1);
  require(zelhash_slot_capacity(d) == minimum, "minimum override");
  const std::string whitespace = " " + valid;
  setenv("MOM_ZELHASH_SLOTS", whitespace.c_str(), 1);
  rejected = false;
  try {
    (void)zelhash_slot_capacity(d);
  } catch (const std::string &) {
    rejected = true;
  }
  require(rejected && allocations == 0 && live.empty(),
          "leading whitespace rejected without allocation");
  unsetenv("MOM_ZELHASH_SLOTS");
}
static void test_lifetime() {
  for (bool compact : {false, true}) {
    configured = {};
    configured.compact = compact;
    unsigned arena_attempts = 0;
    {
      reset_failure();
      ZelHashState state("fixture");
      state.ensure_io();
      require(live.size() == 3, "three actual I/O allocations");
      allocations = 0;
      state.ensure_arenas(false);
      arena_attempts = allocations;
      require(state.arenas_built, "arenas marked complete after retirement");
      const unsigned old_allocations = allocations;
      state.ensure_arenas(false);
      require(allocations == old_allocations, "successful state reused");
    }
    require(live.empty(), "normal destructor releases all");
    for (bool throws : {false, true})
      for (unsigned fail = 1; fail <= 3; ++fail) {
        reset_failure(fail, throws);
        ZelHashState state("fixture");
        bool rejected = false;
        try {
          state.ensure_io();
        } catch (...) {
          rejected = true;
        }
        require(rejected && live.empty(), "partial I/O is failure-atomic");
        require(!state.base_h && !state.base_pending && !state.rows, "I/O pointers reset");
        reset_failure();
        state.ensure_io();
        require(live.size() == 3, "I/O retry succeeds");
      }
    require(live.empty(), "I/O retry teardown");
    for (bool throws : {false, true})
      for (unsigned fail = 1; fail <= arena_attempts; ++fail) {
        reset_failure();
        ZelHashState state("fixture");
        state.ensure_io();
        reset_failure(fail, throws);
        bool rejected = false;
        try {
          state.ensure_arenas(false);
        } catch (...) {
          rejected = true;
        }
        require(rejected && live.size() == 3 && !state.arenas_built,
                "partial arenas failure-atomic; I/O preserved");
        for (auto *pointer : state.level)
          require(!pointer, "level pointer reset");
        require(!state.l0_index && !state.nslots && !state.ref && !state.ref_count && !state.cand &&
                    !state.cand_count,
                "arena pointers reset");
        reset_failure();
        state.ensure_arenas(false);
        require(state.arenas_built, "arena retry succeeds");
      }
    require(live.empty(), "arena retry teardown");
    {
      reset_failure();
      ZelHashState state("fixture");
      state.ensure_io();
      fail_memset = true;
      bool rejected = false;
      try {
        state.ensure_arenas(false);
      } catch (...) {
        rejected = true;
      }
      require(rejected && live.size() == 3 && !state.arenas_built,
              "retirement failure releases arenas");
      reset_failure();
      state.ensure_arenas(false);
    }
    require(live.empty(), "retirement retry teardown");
    {
      reset_failure();
      ZelHashState state("fixture");
      state.ensure_io();
      state.ensure_arenas(false);
      fail_wait = true;
    }
    require(live.empty(), "destructor releases even when wait throws");
    reset_failure();
  }
}
int main() {
  test_entry();
  test_capacity();
  test_lifetime();
  std::printf("ZelHash source-bound CPU entry/capacity/lifetime passed: %u assertions\n",
              assertions);
}
