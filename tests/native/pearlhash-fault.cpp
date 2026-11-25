#include <chrono>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "mom_env_flag.inc"  // the production on/off switch parser (sycl/lib-internal.h)
#include "claim.h"
using mom_pearlhash::pearlhash_search_target_le;
using mom_pearlhash::ProofAxis;
using mom_pearlhash::kSquareAxis;

enum class Stage {
  success, unavailable, queue, preparation, event_a, event_b, launch, wait, query_a, query_b,
  roots, noise, key_copy, target_copy, readback_a, readback_b
};

struct Counters {
  int roots = 0;
  int noise = 0;
  int preparation = 0;
  int root_dependencies = 0;
  int preparation_dependencies = 0;
  int event_wait_a = 0;
  int event_wait_b = 0;
  int queue_wait = 0;
  int launch = 0;
  int wait = 0;
  int fallback = 0;
  int fail = 0;
  int copies = 0;
  int retired_copies = 0;
  int derived_keys = 0;
} counts;

static Stage failure_stage;
static bool string_error;
static bool cleanup_error;
static const void* thrown_address;
static constexpr uint32_t original_seed = 19;
static uint32_t expected_attempt_seed = original_seed;

static void require(bool condition, const char* message) {
  if (!condition)
    throw std::logic_error(message);
}

static void inject(Stage stage) {
  if (failure_stage != stage)
    return;
  if (string_error) {
    try {
      throw std::string("submitted fault");
    } catch (const std::string& error) {
      if (!thrown_address)
        thrown_address = &error;
      throw;
    }
  }
  try {
    throw std::runtime_error("submitted fault");
  } catch (const std::exception& error) {
    if (!thrown_address)
      thrown_address = &error;
    throw;
  }
}

namespace sycl {
enum class backend {ext_oneapi_cuda, hip};
namespace info {
namespace device {
struct vendor_id {};
}
namespace event {
struct command_execution_status {};
}
enum class event_command_status {submitted, running, complete};
}
struct queue;
enum class event_kind {generic, roots, a_noise, b_noise, a_preparation, b_preparation};
struct event {
  bool first;
  event_kind kind;
  const queue* owner;
  event(bool first_value = false, event_kind kind_value = event_kind::generic,
        const queue* owner_value = nullptr)
      : first(first_value), kind(kind_value), owner(owner_value) {}
  template <typename T>
  info::event_command_status get_info() {
    inject(first ? Stage::query_a : Stage::query_b);
    return info::event_command_status::complete;
  }
  void wait_and_throw() {
    if (first)
      ++counts.event_wait_a;
    else
      ++counts.event_wait_b;
    inject(first ? Stage::event_a : Stage::event_b);
  }
};
struct device {
  backend selected;
  bool gpu = false;
  int queue_id = 0;
  backend get_backend() const { return selected; }
  bool is_cpu() const { return false; }
  bool is_gpu() const { return gpu; }
  template <typename T> unsigned get_info() const { return 0; }
};
struct PendingCopy {
  void* destination;
  const void* source;
  size_t size;
};
struct queue {
  device selected;
  std::vector<PendingCopy> pending;
  void* key_address = nullptr;
  void* target_address = nullptr;
  const void* root_a_address = nullptr;
  device get_device() const { return selected; }
  void memcpy(void* destination, const void* source, size_t size) {
    const int copy = counts.copies++;
    require(copy < 4, "unexpected queued copy");
    inject(destination == key_address ? Stage::key_copy :
           destination == target_address ? Stage::target_copy :
           source == root_a_address ? Stage::readback_a : Stage::readback_b);
    pending.push_back({destination, source, size});
  }
  void wait_and_throw() {
    ++counts.queue_wait;
    for (const auto& copy : pending) {
      std::memcpy(copy.destination, copy.source, copy.size);
      ++counts.retired_copies;
    }
    pending.clear();
    if (cleanup_error && thrown_address)
      throw std::logic_error("cleanup fault");
    inject(Stage::queue);
  }
};
}
static bool mom_is_hip(const sycl::device& device) {
  return device.selected == sycl::backend::hip;
}
static bool mom_is_cuda(const sycl::device& device) {
  return device.selected == sycl::backend::ext_oneapi_cuda;
}

// Model the shared helper's GPU query/completion boundary; its polling/unwind policy has its own
// actual-source tests. This fixture keeps device.is_gpu() false outside this dispatch boundary.
static void sycl_wait_and_throw(sycl::event event, const sycl::device& device) {
  require(event.owner && event.owner->get_device().queue_id == device.queue_id &&
              ((event.first && (event.kind == sycl::event_kind::a_noise ||
                                event.kind == sycl::event_kind::a_preparation)) ||
               (!event.first && (event.kind == sycl::event_kind::b_noise ||
                                 event.kind == sycl::event_kind::b_preparation))),
          "HIP A/B completion event was waited on the wrong queue or stream side");
  (void)event.get_info<sycl::info::event::command_execution_status>();
  event.wait_and_throw();
}

constexpr int kPearlhashWinners = 4;
struct Winner {
  unsigned row = 16;
  unsigned col = 32;
  uint8_t jackpot[32]{};
};
struct Result {
  int found = 41;
  int chk = 73;
  uint32_t seed = 777;
  Winner winners[kPearlhashWinners];
};
struct Buffers {
  Result* result;
  uint8_t* key = nullptr;
  uint8_t* target = nullptr;
  uint8_t* CVA = nullptr;
  uint8_t* CVB = nullptr;
  uint8_t* cA = nullptr;
  uint8_t* cB = nullptr;
  ProofAxis proof_rows = kSquareAxis, proof_cols = kSquareAxis;
  bool positive_inputs = false;
  bool seed_prefix_inputs = false;
  bool fixed_b_reuse = false;
  uint32_t fixed_b_seed = 0;
};
using CV = std::array<uint8_t, 32>;
struct PearlHashClaim {
  int row, col;
  uint8_t jackpot[32];
  uint32_t adjustment_factor;
  std::string proof;
};
// Starts with a published claim so every path must visibly invalidate or replace it.
struct Proof {
  uint32_t seed;
  uint32_t b_seed;
  int m, n, k, rank;
  ProofAxis proof_rows, proof_cols;
  uint8_t key[32];
  std::vector<CV> leaves_a, leaves_b;
  bool positive_inputs = false;
  int count = 1;
  PearlHashClaim claims[kPearlhashWinners];
} g_pf;

struct PearlHashHipSearch {
  bool ensure(sycl::queue& q, const Buffers&, int, int, int, int) {
    return q.get_device().selected == sycl::backend::hip && failure_stage != Stage::unavailable;
  }
  void launch(const Buffers&, uint32_t seed, int, int, int, int, bool) {
    require(seed == expected_attempt_seed, "launch changed the attempt seed");
    ++counts.launch;
    inject(Stage::launch);
  }
  void wait(double) {
    ++counts.wait;
    inject(Stage::wait);
  }
  void fail(const std::string&) { ++counts.fail; }
};
struct PearlHashCudaSearch {
  bool ensure(sycl::queue&, const Buffers&, int, int, int, int) {
    return failure_stage != Stage::unavailable;
  }
  void launch(const Buffers&, uint32_t seed, int, int, int, bool) {
    require(seed == expected_attempt_seed, "launch changed the attempt seed");
    ++counts.launch;
    inject(Stage::launch);
  }
  void wait(double) {
    ++counts.wait;
    inject(Stage::wait);
  }
  void fail(const std::string&) { ++counts.fail; }
};
struct PearlHashEventPair {
  sycl::event a{true};
  sycl::event b{false};
};

static sycl::event k_roots(sycl::queue& q, const Buffers&, uint32_t seed, int, int, int) {
  require(seed == expected_attempt_seed, "roots changed the attempt seed");
  ++counts.roots;
  inject(Stage::roots);
  return sycl::event{true, sycl::event_kind::roots, &q};
}
static PearlHashEventPair k_noise(sycl::queue& a, sycl::queue& b, const Buffers&, int, int,
                                  int, int, const sycl::event& roots) {
  require(roots.kind == sycl::event_kind::roots && roots.owner == &a,
          "noise lost the commitment-root dependency");
  ++counts.root_dependencies;
  if (a.selected.selected != sycl::backend::hip)
    require(&a == &b, "non-HIP noise escaped the main queue");
  else
    require(&a != &b, "HIP A/B noise producers must use distinct queues");
  ++counts.noise;
  inject(Stage::noise);
  return {sycl::event{true, sycl::event_kind::a_noise, &a},
          sycl::event{false, sycl::event_kind::b_noise, &b}};
}
static void fixture_compute_ab(sycl::queue&, const Buffers&, uint32_t seed, int, int, int, int,
                               bool) {
  require(seed == expected_attempt_seed, "preparation changed the attempt seed");
  ++counts.preparation;
  inject(Stage::preparation);
}
template <bool Dpasw = false, bool PackedCuda = false>
static sycl::event compute_ab(sycl::queue& q, const Buffers& b, uint32_t seed, int m, int n, int k,
                              int rank, bool portable, bool force_b_pack = false) {
  (void)Dpasw;
  (void)PackedCuda;
  (void)force_b_pack;
  fixture_compute_ab(q, b, seed, m, n, k, rank, portable);
  return {};
}
static PearlHashEventPair compute_ab_amd_wmma(
    sycl::queue& q, sycl::queue& prep, const Buffers& b, uint32_t seed, int m, int n, int k,
    int rank, const sycl::event& a_noise, const sycl::event& b_noise) {
  require(&q != &prep && a_noise.kind == sycl::event_kind::a_noise &&
              a_noise.owner == &q && b_noise.kind == sycl::event_kind::b_noise &&
              b_noise.owner == &prep,
          "HIP preparation lost split noise queue/event ownership");
  ++counts.preparation_dependencies;
  fixture_compute_ab(q, b, seed, m, n, k, rank, true);
  return {sycl::event{true, sycl::event_kind::a_preparation, &q},
          sycl::event{false, sycl::event_kind::b_preparation, &prep}};
}
static void compute_ab_cpu(const Buffers&, uint32_t, int, int, int, int) {
}
static bool cuda_sycl_search_supports_shape(int, int) {
  return false;
}
static size_t checked_elements(int n, int k, const char*) {
  return static_cast<size_t>(n) * static_cast<size_t>(k);
}
static size_t checked_chunk_count(size_t bytes, const char*) {
  return bytes / 1024 + (bytes % 1024 != 0);
}
static void search(sycl::queue&, const Buffers&, uint32_t seed, int, int, int, int, bool,
                   bool = false) {
  require(seed == expected_attempt_seed, "fallback changed the attempt seed");
  ++counts.fallback;
}
static void search_cpu(
    sycl::queue& q, const Buffers& b, uint32_t seed, int m, int n, int k, int rank, bool debug) {
  search(q, b, seed, m, n, k, rank, debug);
}
static void search_cuda(
    sycl::queue& q, const Buffers& b, uint32_t seed, int m, int n, int k, int rank, bool debug) {
  search(q, b, seed, m, n, k, rank, debug);
}

#include "dispatch.inc"

struct State {
  PearlHashHipSearch hip;
  PearlHashCudaSearch cuda;
  sycl::queue prep_queue;
  double wait_ema_us = 0.0;
  bool cuda_tensor = false;
  bool cuda_f16 = false;
  uint8_t key[32]{};
  uint8_t header[76]{};
  bool have_header = false;
  bool header_pair_dist8 = false;
  bool fixed_b_valid = false;
  uint32_t fixed_b_seed = 0;
};

static void derive_key(const uint8_t*, int, int, uint8_t* key, const ProofAxis&,
                       const ProofAxis&) {
  ++counts.derived_keys;
  std::memset(key, 7, 32);
}
static void prepare_roots_cpu(const Buffers& b, uint32_t, int, int, int,
                              const uint32_t* = nullptr) {
  std::memset(b.cA, 3, 32);
  std::memset(b.cB, 4, 32);
}
template <typename Fn> static void sycl_cleanup_noexcept(const char*, Fn&& fn) noexcept {
  try {
    fn();
  } catch (...) {
  }
}

static int run_copies(sycl::queue& q, State& st, Buffers& b, uint64_t* pseed, bool is_test) {
#include "invalidate.inc"
#include "state-invalidation.inc"
  constexpr bool pair_dist8 = false;
  constexpr int m = 2048, n = 2048, k = 2048, rank = 128;
  constexpr bool hip_jit = false;
  constexpr unsigned job_ref = 1, cert_version = 3;
  const std::string backend = "native";
  const uint8_t input[76] = {1};
  uint8_t target[32];
  const uint8_t* const base_target = nullptr;
  bool reuse_fixed_b = false;
  for (unsigned i = 0; i < 32; ++i)
    target[i] = static_cast<uint8_t>(i);
#include "upload.inc"
#include "readback.inc"
#include "commit.inc"
}

static int check_copies(bool readback) {
  sycl::queue q{{sycl::backend::ext_oneapi_cuda, readback}, {}};
  State st{{}, {}, {q.get_device(), {}}};
  Result result;
  g_pf.leaves_a.emplace_back();
  g_pf.leaves_b.emplace_back();
  std::array<uint8_t, 32> key{}, target{}, root_a{}, root_b{};
  root_a.fill(3);
  root_b.fill(4);
  Buffers b{&result, key.data(), target.data(), nullptr, nullptr, root_a.data(), root_b.data()};
  q.key_address = key.data();
  q.target_address = target.data();
  q.root_a_address = root_a.data();
  st.have_header = true;
  st.fixed_b_valid = true;
  st.fixed_b_seed = 0xfeed1234u;
  uint64_t seed = original_seed;
  bool caught = false;
  bool same_exception = false;
  int completed = -1;
  try {
    completed = run_copies(q, st, b, &seed, readback);
  } catch (const std::string& error) {
    caught = true;
    same_exception = string_error && &error == thrown_address;
  } catch (const std::exception& error) {
    caught = true;
    same_exception = !string_error && &error == thrown_address;
  }
  const bool fault = failure_stage != Stage::success;
  require(caught == fault, "queued-copy fault was swallowed or healthy path failed");
  require(q.pending.empty(), "queued copy outlived its stack owner");
  require(!st.fixed_b_valid, "entry invalidation retained a stale fixed-B snapshot");
  if (counts.retired_copies >= 2) {
    for (unsigned i = 0; i < 32; ++i) {
      require(key[i] == 7 && target[i] == (readback ? 255 : 31 - i),
              "deferred upload changed the key or target byte order");
    }
  }
  require(counts.fallback == 0 && counts.fail == 0, "queued-copy fault retried another search");
  require(seed == original_seed && g_pf.count == 0 && g_pf.leaves_a.empty() &&
              g_pf.leaves_b.empty(),
          "queued-copy path committed a seed or proof");
  if (fault) {
    require(same_exception, "queue drain replaced the original exception");
    require(completed == -1, "queued-copy fault reached the result commit");
    const bool uploaded = failure_stage == Stage::readback_a || failure_stage == Stage::readback_b;
    require(st.have_header == uploaded, "failed new-header upload was cached as ready");
    const bool waited = uploaded || failure_stage == Stage::queue || failure_stage == Stage::wait;
    require(counts.queue_wait == 1 + (waited ? 1 : 0) + (uploaded ? 0 : 1),
            "fault cleanup did not retire the owning queues");
    if (!uploaded) {
      if (failure_stage != Stage::wait)
        require(counts.launch == 0 && counts.wait == 0, "upload fault reached native search");
      counts = {};
      failure_stage = Stage::success;
      cleanup_error = false;
      (void)run_copies(q, st, b, &seed, false);
      require(counts.derived_keys == 1 && st.have_header, "same header did not retry its failed upload");
    }
  } else {
    require(completed == 0 && st.have_header, "healthy upload behavior changed");
    require(counts.derived_keys == 1 && counts.queue_wait == (readback ? 2 : 1),
            "healthy path added a drain or lost its header");
    require(counts.retired_copies == (readback ? 4 : 2), "healthy deferred copies did not complete");
    counts = {};
    (void)run_copies(q, st, b, &seed, false);
    require(counts.derived_keys == 0 && counts.copies == 1 && counts.queue_wait == 1,
            "healthy cached header was uploaded again or gained a drain");
  }
  require(q.pending.empty(), "retry left a queued copy");
  std::puts("PASS actual-source Pearl queued-copy lifetime");
  return 0;
}

static int run(bool dispatch, bool hip, sycl::queue& q, Buffers& b, uint64_t* pseed) {
#include "invalidate.inc"
  constexpr bool pair_dist8 = false;
  State st{{}, {}, {q.get_device(), {}}};
#include "state-invalidation.inc"
  bool reuse_fixed_b = pair_dist8 && prior_fixed_b_valid;
  q.selected.queue_id = 1;
  st.prep_queue.selected.queue_id = 2;
  const uint32_t attempt_seed = static_cast<uint32_t>(*pseed);
  constexpr int m = 2048, n = 2048, k = 2048, rank = 128;
  constexpr bool is_test = false;
  PearlHashSearchBackend search_backend = dispatch
      ? attempt(q, b, attempt_seed, m, n, k, rank, "native", &st.hip,
                hip ? nullptr : &st.cuda, &st.prep_queue)
      : hip ? PearlHashSearchBackend::hip_jit : PearlHashSearchBackend::cuda_jit;
#include "wait.inc"
#include "commit-fixed-b.inc"
#include "commit.inc"
}

struct PairObservation {
  bool reuse;
  uint32_t producer_b_seed;
  uint32_t committed_b_seed;
  bool cache_valid;
  uint32_t attempt_seed;
};

// This is deliberately assembled from the production state/header/seed/dispatch/commit blocks
// by compiler-policy.js. It is a state-transition fixture, not an independent fixed-B model.
static PairObservation run_pair_attempt(sycl::queue& q, State& st, Buffers& b, uint64_t* pseed,
                                        bool pair_dist8, const uint8_t* input) {
#include "invalidate.inc"
#include "state-invalidation.inc"
  constexpr int m = 2048, n = 2048, k = 8192, rank = 128;
  const std::string backend = "native";
  b.positive_inputs = true;
  q.key_address = b.key;
#include "pair-header.inc"
  const uint32_t attempt_seed = static_cast<uint32_t>(*pseed);
  expected_attempt_seed = attempt_seed;
#include "fixed-b-seed.inc"
  PearlHashSearchBackend search_backend;
  try {
    search_backend = attempt(q, b, attempt_seed, m, n, k, rank, backend, &st.hip, &st.cuda,
                              nullptr, false, true, false, pair_dist8, reuse_fixed_b);
#include "wait.inc"
  }
#include "pair-catch.inc"
#include "commit-fixed-b.inc"
  b.result->found = 0; // Exercise the actual no-winner commit path used by a completed attempt.
  commit_fixed_b();
  return {b.fixed_b_reuse, b.fixed_b_seed, st.fixed_b_seed, st.fixed_b_valid, attempt_seed};
}

static int check_pair_lifecycle() {
  sycl::queue q{{sycl::backend::ext_oneapi_cuda, true}, {}};
  State st{{}, {}, {q.get_device(), {}}};
  Result result;
  result.found = 0;
  std::array<uint8_t, 32> key{}, target{};
  Buffers b{&result, key.data(), target.data()};
  uint8_t first_header[76] = {1};
  uint8_t changed_header[76] = {2};
  failure_stage = Stage::success;
  string_error = false;
  cleanup_error = false;
  thrown_address = nullptr;

  uint64_t first_seed = original_seed;
  const PairObservation first = run_pair_attempt(q, st, b, &first_seed, true, first_header);
  require(!first.reuse && first.producer_b_seed == original_seed && first.cache_valid,
          "first pair attempt did not commit its B seed");
  require(first.committed_b_seed == original_seed && first.attempt_seed == original_seed,
          "first pair commit changed the producer seed");

  uint64_t second_seed = original_seed + 1;
  const PairObservation second = run_pair_attempt(q, st, b, &second_seed, true, first_header);
  require(second.reuse && second.producer_b_seed == original_seed && second.cache_valid,
          "same-header pair attempt did not reuse the committed B seed");
  require(second.attempt_seed != first.attempt_seed,
          "same-header pair lifecycle did not advance the A attempt seed");
  require(second.committed_b_seed == original_seed,
          "pair cache hit replaced the committed B seed");

  // A header changes the keyed B roots/noise and must force a fresh B miss. This catches reuse
  // based only on the newly copied header rather than the header that was cached with fixed B.
  uint64_t changed_seed = original_seed + 2;
  const PairObservation changed =
      run_pair_attempt(q, st, b, &changed_seed, true, changed_header);
  require(!changed.reuse && changed.producer_b_seed == changed_seed && changed.cache_valid,
          "header change reused fixed B from the previous header");

  // Switching away from the pair route invalidates the retained pair state; returning to pair
  // mode must be a miss even when the header bytes are unchanged.
  uint64_t ordinary_seed = original_seed + 3;
  const PairObservation ordinary =
      run_pair_attempt(q, st, b, &ordinary_seed, false, changed_header);
  require(!ordinary.reuse && !ordinary.cache_valid,
          "ordinary route retained pair fixed-B state");
  uint64_t returned_seed = original_seed + 4;
  const PairObservation returned =
      run_pair_attempt(q, st, b, &returned_seed, true, changed_header);
  require(!returned.reuse && returned.producer_b_seed == returned_seed && returned.cache_valid,
          "pair route after ordinary mode did not rebuild B");

  // A producer fault is before the commit block and must leave the cache invalid. The production
  // catch block is included above, so queued ownership is checked in the same fixture.
  failure_stage = Stage::preparation;
  uint64_t failed_seed = original_seed + 5;
  bool caught = false;
  try {
    (void)run_pair_attempt(q, st, b, &failed_seed, true, changed_header);
  } catch (const std::exception&) {
    caught = true;
  } catch (const std::string&) {
    caught = true;
  }
  require(caught && !st.fixed_b_valid && q.pending.empty(),
          "pair producer fault published or retained fixed B");
  failure_stage = Stage::success;
  expected_attempt_seed = original_seed;
  std::puts("PASS actual-source Pearl pair fixed-B lifecycle");
  return 0;
}

// Exercise the real native winner-capture block with deferred copies, including failed readbacks.
static int check_native_capture(bool hip) {
  sycl::queue q{{hip ? sycl::backend::hip : sycl::backend::ext_oneapi_cuda, true}, {}};
  const size_t chunks = checked_chunk_count(checked_elements(2048, 2048, "fixture"), "fixture");
  std::vector<CV> a(chunks), b(chunks);
  for (size_t i = 0; i < chunks; ++i)
    for (size_t j = 0; j < 32; ++j) {
      a[i][j] = static_cast<uint8_t>(i * 13 + j);
      b[i][j] = static_cast<uint8_t>(i * 37 + j + 1);
    }
  Result result;
  Buffers buffers{&result};
  buffers.CVA = reinterpret_cast<uint8_t*>(a.data());
  buffers.CVB = reinterpret_cast<uint8_t*>(b.data());
  q.root_a_address = buffers.CVA;
  g_pf.leaves_a.emplace_back();
  g_pf.leaves_b.emplace_back();
  uint64_t seed = original_seed;
  bool caught = false, same_exception = false;
  try {
    // found=41 keeps every winner slot; the commit reports how many it captured.
    require(run(false, hip, q, buffers, &seed) == kPearlhashWinners, "native winner was lost");
  } catch (const std::string& error) {
    caught = true;
    same_exception = string_error && &error == thrown_address;
  } catch (const std::exception& error) {
    caught = true;
    same_exception = !string_error && &error == thrown_address;
  }
  const bool fault = failure_stage != Stage::success;
  require(caught == fault, "native capture fault was swallowed or success failed");
  require(q.pending.empty(), "native capture left a queued copy targeting proof storage");
  if (fault) {
    require(same_exception, "native capture changed the original exception");
    require(seed == original_seed && g_pf.count == 0,
            "failed native capture committed a seed or exposed a stale claim");
  } else {
    require(seed == result.seed && g_pf.count > 0 && g_pf.claims[0].proof.empty() &&
                g_pf.leaves_a == a && g_pf.leaves_b == b,
            "native capture did not retain the winning seed's exact leaves");
    require(counts.copies == 2 && counts.retired_copies == 2,
            "native capture failed to finish both readbacks");
  }
  std::puts("PASS actual-source Pearl native proof capture");
  return 0;
}

// A selected pair key cannot fall through to the ordinary contiguous search if
// a later admission check declines. Exercise the actual dispatch, not a model.
static int check_pair_decline() {
#include "invalidate.inc"
  sycl::queue q{{sycl::backend::ext_oneapi_cuda}, {}};
  Result result;
  Buffers b{&result};
  PearlHashCudaSearch cuda;
  failure_stage = Stage::unavailable;
  bool caught = false;
  try {
    (void)attempt(q, b, original_seed, 2048, 2048, 8192, 128, "native",
                  nullptr, &cuda, nullptr, false, false, false, true, false);
  } catch (const std::string& error) {
    caught = error == "PearlHash pair-dist8 CUDA route became unavailable after admission";
  }
  require(caught && counts.fallback == 0 && counts.launch == 0 && counts.fail == 0,
          "admitted pair route silently changed its proof pattern or retried");
  require(g_pf.count == 0 && g_pf.leaves_a.empty() && g_pf.leaves_b.empty(),
          "declined pair route exposed a stale claim");
  std::puts("PASS actual-source Pearl pair decline fails closed");
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 5)
    return 2;
  const bool dispatch = std::atoi(argv[1]) != 0;
  const bool hip = std::atoi(argv[2]) != 0;
  failure_stage = static_cast<Stage>(std::atoi(argv[3]));
  string_error = std::atoi(argv[4]) != 0;
  if (std::string(argv[1]) == "pair-lifecycle") {
    try {
      return check_pair_lifecycle();
    } catch (const std::exception& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
  if (std::string(argv[1]) == "pair") {
    try {
      return check_pair_decline();
    } catch (const std::exception& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
  if (std::string(argv[1]) == "capture") {
    string_error = std::atoi(argv[4]) % 2 != 0;
    cleanup_error = std::atoi(argv[4]) >= 2;
    try {
      return check_native_capture(hip);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
  if (std::string(argv[1]) == "copies") {
    string_error = std::atoi(argv[4]) % 2 != 0;
    cleanup_error = std::atoi(argv[4]) >= 2;
    try {
      return check_copies(std::atoi(argv[2]) != 0);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
#if defined(_WIN32)
  _putenv_s("MOM_PEARLHASH_STATS", "");
  _putenv_s("MOM_PEARLHASH_CHK", "");
#else
  unsetenv("MOM_PEARLHASH_STATS");
  unsetenv("MOM_PEARLHASH_CHK");
#endif
  sycl::queue q{{hip ? sycl::backend::hip : sycl::backend::ext_oneapi_cuda}, {}};
  Result result;
  g_pf.leaves_a.emplace_back();
  g_pf.leaves_b.emplace_back();
  Buffers b{&result};
  uint64_t seed = original_seed;
  bool caught = false;
  bool same_exception = false;
  int completed = -1;
  try {
    completed = run(dispatch, hip, q, b, &seed);
  } catch (const std::string& error) {
    caught = true;
    same_exception = string_error && &error == thrown_address;
  } catch (const std::exception& error) {
    caught = true;
    same_exception = !string_error && &error == thrown_address;
  }
  try {
    const bool fault = failure_stage != Stage::success && failure_stage != Stage::unavailable;
    require(caught == fault, "submitted fault was swallowed or a successful path failed");
    require(counts.fail == 0, "submitted fault disabled the native backend and retried");
    if (fault) {
      require(same_exception, "submitted fault changed exception identity or type");
      require(counts.fallback == 0, "submitted fault retried the generic search");
      require(result.found == 41 && result.chk == 73 && result.seed == 777,
              "submitted fault reset the result");
      require(seed == original_seed, "submitted fault committed a winning seed");
      require(g_pf.count == 0 && g_pf.leaves_a.empty() && g_pf.leaves_b.empty(),
              "submitted fault published a proof");
      require(completed == -1, "submitted fault reached the result commit");
    } else {
      require(completed == kPearlhashWinners && seed == 777 && g_pf.count == kPearlhashWinners && g_pf.claims[0].proof.empty() &&
                  g_pf.leaves_a.empty() && g_pf.leaves_b.empty(),
              "successful path failed to capture the winning result");
      const bool unavailable = failure_stage == Stage::unavailable;
      require(counts.fallback == (unavailable ? 1 : 0), "unavailability fallback changed");
      require(counts.launch == (dispatch && !unavailable ? 1 : 0), "native launch count changed");
      require(counts.wait == (unavailable ? 0 : 1), "native wait count changed");
    }
    require(counts.roots == (dispatch ? 1 : 0), "attempt was resubmitted");
    require(counts.noise == (dispatch ? 1 : 0), "noise preparation was repeated");
    if (dispatch && hip) {
      if (failure_stage != Stage::roots) {
        require(counts.root_dependencies == 1,
                "HIP noise did not retain the commitment-root dependency");
        require(counts.noise == 1, "HIP split noise was not submitted exactly once");
      }
      if (failure_stage != Stage::roots && failure_stage != Stage::noise &&
          failure_stage != Stage::unavailable)
        require(counts.preparation_dependencies == 1,
                "HIP preparation did not retain both noise event dependencies");

      int expected_a = -1, expected_b = -1;
      switch (failure_stage) {
      case Stage::success:
      case Stage::unavailable:
      case Stage::event_b:
      case Stage::launch:
      case Stage::wait:
        expected_a = expected_b = 1;
        break;
      case Stage::event_a:
      case Stage::query_b:
        expected_a = 1;
        expected_b = 0;
        break;
      case Stage::query_a:
      case Stage::preparation:
        expected_a = expected_b = 0;
        break;
      default:
        break;
      }
      if (expected_a >= 0) {
        require(counts.event_wait_a == expected_a && counts.event_wait_b == expected_b,
                "HIP A/B completion events were not retired independently");
      }
    }
    std::puts("PASS actual-source Pearl native fault boundary");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
