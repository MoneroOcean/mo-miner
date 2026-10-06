#include <chrono>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

enum class Stage {
  success, unavailable, queue, preparation, event_a, event_b, launch, wait, query_a, query_b,
  roots, noise, key_copy, target_copy, readback_a, readback_b
};

struct Counters {
  int roots = 0;
  int noise = 0;
  int preparation = 0;
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
enum class event_command_status {complete};
}
struct device {
  backend selected;
  bool gpu = false;
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
struct event {
  bool first;
  template <typename T>
  info::event_command_status get_info() {
    inject(first ? Stage::query_a : Stage::query_b);
    return info::event_command_status::complete;
  }
  void wait_and_throw() {
    inject(first ? Stage::event_a : Stage::event_b);
  }
};
}

struct Result {
  int found = 41;
  int chk = 73;
  uint32_t seed = 777;
  unsigned row = 16;
  unsigned col = 32;
  uint8_t jackpot[32]{};
};
struct Buffers {
  Result* result;
  uint8_t* key = nullptr;
  uint8_t* target = nullptr;
  uint8_t* CVA = nullptr;
  uint8_t* CVB = nullptr;
  uint8_t* cA = nullptr;
  uint8_t* cB = nullptr;
};
struct Proof {
  uint32_t seed;
  int row, col, m, n, k, rank;
  uint8_t key[32], jackpot[32];
  bool claim_ready = true;
  bool valid = true;
} g_pf;

struct PearlHashHipSearch {
  bool ensure(sycl::queue& q, int, int, int, int) {
    return q.get_device().selected == sycl::backend::hip && failure_stage != Stage::unavailable;
  }
  void launch(const Buffers&, uint32_t seed, int, int, int, int, bool) {
    require(seed == original_seed, "launch changed the attempt seed");
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
    require(seed == original_seed, "launch changed the attempt seed");
    ++counts.launch;
    inject(Stage::launch);
  }
  void wait(double) {
    ++counts.wait;
    inject(Stage::wait);
  }
  void fail(const std::string&) { ++counts.fail; }
};
struct PearlHashPrepEvents {
  sycl::event a{true};
  sycl::event b{false};
};

static void k_roots(sycl::queue&, const Buffers&, uint32_t seed, int, int, int) {
  require(seed == original_seed, "roots changed the attempt seed");
  ++counts.roots;
  inject(Stage::roots);
}
static void k_noise(sycl::queue&, const Buffers&, int, int, int, int) {
  ++counts.noise;
  inject(Stage::noise);
}
static void compute_ab(sycl::queue&, const Buffers&, uint32_t seed, int, int, int, int, bool) {
  require(seed == original_seed, "preparation changed the attempt seed");
  ++counts.preparation;
  inject(Stage::preparation);
}
static PearlHashPrepEvents compute_ab_amd_wmma(
    sycl::queue& q, sycl::queue&, const Buffers& b, uint32_t seed, int m, int n, int k, int rank) {
  compute_ab(q, b, seed, m, n, k, rank, true);
  return {};
}
static void compute_ab_cpu(const Buffers&, uint32_t, int, int, int, int) {
}
static bool cuda_sycl_search_supports_shape(int, int) {
  return false;
}
static size_t checked_elements(int n, int k, const char*) {
  return static_cast<size_t>(n) * static_cast<size_t>(k);
}
static void search(sycl::queue&, const Buffers&, uint32_t seed, int, int, int, int, bool) {
  require(seed == original_seed, "fallback changed the attempt seed");
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
  uint8_t key[32]{};
  uint8_t header[76]{};
  bool have_header = false;
};

static void derive_key(const uint8_t*, int, int, uint8_t* key) {
  ++counts.derived_keys;
  std::memset(key, 7, 32);
}
static void prepare_roots_cpu(const Buffers& b, uint32_t, int, int, int) {
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
  constexpr int m = 2048, n = 2048, k = 2048, rank = 128;
  constexpr unsigned job_ref = 1, cert_version = 3;
  const std::string backend = "native";
  const uint8_t input[76] = {1};
  uint8_t target[32];
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
  std::array<uint8_t, 32> key{}, target{}, root_a{}, root_b{};
  root_a.fill(3);
  root_b.fill(4);
  Buffers b{&result, key.data(), target.data(), nullptr, nullptr, root_a.data(), root_b.data()};
  q.key_address = key.data();
  q.target_address = target.data();
  q.root_a_address = root_a.data();
  st.have_header = true;
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
  if (counts.retired_copies >= 2) {
    for (unsigned i = 0; i < 32; ++i) {
      require(key[i] == 7 && target[i] == (readback ? 255 : 31 - i),
              "deferred upload changed the key or target byte order");
    }
  }
  require(counts.fallback == 0 && counts.fail == 0, "queued-copy fault retried another search");
  require(seed == original_seed && !g_pf.valid && !g_pf.claim_ready,
          "queued-copy path committed a seed or proof");
  if (fault) {
    require(same_exception, "queue drain replaced the original exception");
    require(completed == -1, "queued-copy fault reached the result commit");
    const bool uploaded = failure_stage == Stage::readback_a || failure_stage == Stage::readback_b;
    require(st.have_header == uploaded, "failed new-header upload was cached as ready");
    const bool waited = uploaded || failure_stage == Stage::queue || failure_stage == Stage::wait;
    require(counts.queue_wait == 1 + (waited ? 1 : 0), "fault was not drained exactly once");
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
  State st{{}, {}, {q.get_device(), {}}};
  const uint32_t attempt_seed = static_cast<uint32_t>(*pseed);
  constexpr int m = 2048, n = 2048, k = 2048, rank = 128;
  constexpr bool is_test = false;
  PearlHashSearchBackend search_backend = dispatch
      ? attempt(q, b, attempt_seed, m, n, k, rank, "native", hip ? &st.hip : nullptr,
                hip ? nullptr : &st.cuda, &st.prep_queue)
      : hip ? PearlHashSearchBackend::hip_jit : PearlHashSearchBackend::cuda_jit;
#include "wait.inc"
#include "commit.inc"
}

int main(int argc, char** argv) {
  if (argc != 5)
    return 2;
  const bool dispatch = std::atoi(argv[1]) != 0;
  const bool hip = std::atoi(argv[2]) != 0;
  failure_stage = static_cast<Stage>(std::atoi(argv[3]));
  string_error = std::atoi(argv[4]) != 0;
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
      require(!g_pf.valid && !g_pf.claim_ready, "submitted fault published a proof");
      require(completed == -1, "submitted fault reached the result commit");
    } else {
      require(completed == 1 && seed == 777 && g_pf.valid && !g_pf.claim_ready,
              "successful path failed to capture the winning result");
      const bool unavailable = failure_stage == Stage::unavailable;
      require(counts.fallback == (unavailable ? 1 : 0), "unavailability fallback changed");
      require(counts.launch == (dispatch && !unavailable ? 1 : 0), "native launch count changed");
      require(counts.wait == (unavailable ? 0 : 1), "native wait count changed");
    }
    require(counts.roots == (dispatch ? 1 : 0), "attempt was resubmitted");
    require(counts.noise == (dispatch ? 1 : 0), "noise preparation was repeated");
    std::puts("PASS actual-source Pearl native fault boundary");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
