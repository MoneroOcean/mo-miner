#include "../lib-internal.h"
#include "../../native/consts.h"
#include "crypto/randomx/blake2/blake2.h"
#include "cycle.h"

#if defined(_WIN32)
#include <windows.h>
#endif

// Cuckaroo Cycle algorithm constants and fixed kernel sizing.
const constexpr uint64_t DUCK_SIZE_A = 129, DUCK_SIZE_B = 83, INDEX_SIZE = 4096,
                         MAX_INITIAL_INDEX_SIZE = 65536;

const constexpr uint64_t BUFFER_SIZE_A1 = DUCK_SIZE_A * 1024 * (4096 - 128) * 2,
                         BUFFER_SIZE_A2 = DUCK_SIZE_A * 1024 * 256 * 2,
                         BUFFER_SIZE_B = DUCK_SIZE_B * 1024 * 4096 * 2;

const constexpr uint32_t COMPUTE_THREADS = 1024, PORTABLE_TRIMMING_ROUNDS = 80,
                         INTEL_TRIMMING_ROUNDS = 42, EDGE_BLOCK_SIZE = 64,
                         EDGE_BITS = mom_c29::kEdgeBits,
                         BUCKET_OFFSET = 255, BUCKET_STEP = 32;
#ifdef MOM_C29_INTEL
const constexpr uint32_t MAX_TRIMMED_EDGE_COUNT = 512 * COMPUTE_THREADS, DEFAULT_SEED_BLOCKS = 4,
                         DEFAULT_SEED_LOCAL_SIZE = 256;
#else
const constexpr uint32_t MAX_TRIMMED_EDGE_COUNT = 128 * COMPUTE_THREADS, DEFAULT_SEED_BLOCKS = 16,
                         DEFAULT_SEED_LOCAL_SIZE = 128;
#endif

// Trimming uses fixed-capacity GPU buckets and a bounded host handoff. An overfull bucket or final
// handoff discards excess edges, which can make this nonce miss a cycle but cannot fabricate one:
// recovery accepts only edges regenerated from the nonce. Raising these limits is therefore a
// measured memory/performance trade-off, not a correctness fallback to add to the mining hot path.
const constexpr uint32_t EDGE_BLOCK_MASK = EDGE_BLOCK_SIZE - 1, NUM_EDGES = mom_c29::kNumEdges,
                         EDGE_MASK = NUM_EDGES - 1;

static uint32_t c29_env_u32(const char* name, const uint32_t default_value) {
  const char* const value = std::getenv(name);
  if (!value || !value[0]) return default_value;

  const unsigned long parsed = std::strtoul(value, nullptr, 10);
  return parsed == 0 ? default_value : static_cast<uint32_t>(parsed);
}

static bool c29_profile_enabled() { static const bool v = [](){ const char* e = std::getenv("MOM_C29_PROFILE"); return e && e[0] && e[0] != '0'; }(); return v; }

static uint32_t c29_profile_limit() { static const uint32_t v = [](){ const char* e = std::getenv("MOM_C29_PROFILE_LIMIT"); if (!e || !e[0]) return 1u; const unsigned long p = std::strtoul(e, nullptr, 10); return p == 0 ? UINT32_MAX : static_cast<uint32_t>(p); }(); return v; }

uint32_t c29_seed_local_size() {
  static const uint32_t v = []() {
    const uint32_t r = c29_env_u32("MOM_C29_SEED_LOCAL_SIZE", DEFAULT_SEED_LOCAL_SIZE);
    return (r == 64u || r == 128u || r == 256u) ? r : DEFAULT_SEED_LOCAL_SIZE;
  }();
  return v;
}

uint32_t c29_seed_blocks_per_item() {
  static const uint32_t v = []() {
    const uint32_t r = c29_env_u32("MOM_C29_SEED_BLOCKS", DEFAULT_SEED_BLOCKS);
    return (r == 4u || r == 8u || r == 16u || r == 32u) ? r : DEFAULT_SEED_BLOCKS;
  }();
  return v;
}

static uint32_t c29_trim_rounds(const sycl::device& device) {
#ifdef MOM_C29_INTEL
  // The combined DPC++ artifact also contains a CUDA image; keep its established trim depth.
  return mom_is_cuda(device) ? PORTABLE_TRIMMING_ROUNDS : INTEL_TRIMMING_ROUNDS;
#else
  (void)device;
  return PORTABLE_TRIMMING_ROUNDS;
#endif
}

static bool c29_direct_seed_enabled() { static const bool v = [](){ const char* e = std::getenv("MOM_C29_DIRECT_SEED"); return !(e && e[0] == '0'); }(); return v; }

static uint64_t c29_now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()
  ).count();
}

struct C29ProfileEvent {
  const char* name;
  sycl::event event;
};

struct C29Profile {
  bool enabled;
  uint64_t start_us;
  uint64_t last_us;
  std::vector<C29ProfileEvent> events;
  C29Profile(const unsigned job_ref, const uint64_t nonce, const unsigned proof_size,
             const sycl::device& device)
      : enabled(false), start_us(c29_now_us()), last_us(start_us) {
    static std::atomic<uint32_t> profiled_graphs{0};
    enabled = c29_profile_enabled() &&
              profiled_graphs.fetch_add(1, std::memory_order_relaxed) < c29_profile_limit();
    if (!enabled)
      return;

    std::printf(
        "[c29-profile] graph start job=%u nonce=%llu proof=%u device=\"%s\" platform=\"%s\" "
        "index_size=%llu trim_rounds=%u compute_threads=%u seed_local=%u seed_blocks=%u "
        "direct_seed=%u\n",
        job_ref, static_cast<unsigned long long>(nonce), proof_size,
        device.get_info<sycl::info::device::name>().c_str(),
        device.get_platform().get_info<sycl::info::platform::name>().c_str(),
        static_cast<unsigned long long>(INDEX_SIZE), c29_trim_rounds(device), COMPUTE_THREADS,
        c29_seed_local_size(), c29_seed_blocks_per_item(), c29_direct_seed_enabled() ? 1u : 0u);
  }

  void mark(const char* name) {
    if (!enabled) return;

    const uint64_t now_us = c29_now_us();
    std::printf("[c29-profile] host %-28s %.3f ms\n", name, (now_us - last_us) / 1000.0);
    last_us = now_us;
  }

  void add_event(const char* name, const sycl::event& event) { if (enabled) events.push_back({name, event}); }

  void print_events() {
    if (!enabled) return;

    double total_ms = 0.0;

    for (const C29ProfileEvent& entry : events) {
      try {
        const uint64_t start = entry.event.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t end   = entry.event.get_profiling_info<sycl::info::event_profiling::command_end>();
        const double ms = static_cast<double>(end - start) / 1000000.0;
        total_ms += ms;
        std::printf("[c29-profile] device %-26s %.3f ms\n", entry.name, ms);
      } catch (const sycl::exception& e) {
        std::printf("[c29-profile] device %-26s unavailable: %s\n", entry.name, e.what());
      }
    }
    std::printf("[c29-profile] device total                    %.3f ms\n", total_ms);
  }

  void finish(const uint32_t trimmed_edges) {
    if (!enabled) return;

    const uint64_t now_us = c29_now_us();
    std::printf("[c29-profile] graph end trimmed=%u host_total=%.3f ms\n", trimmed_edges, (now_us - start_us) / 1000.0);
  }
};

struct C29Buffers {
  sycl::buffer<uint32_t, 1> buffer_a1{sycl::range<1>{BUFFER_SIZE_A1}};
  sycl::buffer<uint32_t, 1> buffer_a2{sycl::range<1>{BUFFER_SIZE_A2}};
  sycl::buffer<uint32_t, 1> buffer_b{sycl::range<1>{BUFFER_SIZE_B}};
  sycl::buffer<uint32_t, 1> buffer_i1{sycl::range<1>{MAX_INITIAL_INDEX_SIZE}};
  sycl::buffer<uint32_t, 1> buffer_i2{sycl::range<1>{MAX_INITIAL_INDEX_SIZE}};
  sycl::buffer<uint32_t, 1> buffer_trimmed_edge_count{sycl::range<1>{1}};
  sycl::buffer<sycl::uint2, 1> buffer_trimmed_edges_u2{sycl::range<1>{MAX_TRIMMED_EDGE_COUNT}};
};

using C29GlobalAtomicRef =
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>;

template <typename Accessor>
inline C29GlobalAtomicRef c29_global_atomic(const Accessor& accessor, const size_t index) {
  auto pointer = accessor.template get_multi_ptr<sycl::access::decorated::yes>();
  return C29GlobalAtomicRef{pointer[index]};
}

struct C29QueuedSolution {
  std::vector<sycl::uint2> edges;
  uint64_t seed[4];
  unsigned job_ref;
  uint64_t nonce;
};

struct C29ResultState {
  std::mutex mutex;
  std::vector<C29QueuedSolution> solutions;
  std::map<unsigned, uint32_t> running_searches;
  unsigned active_job_ref = 0;
};

static sycl::async_handler c29_exception_handler() {
  return [](sycl::exception_list exceptions) {
    for (const std::exception_ptr& exception : exceptions) {
      try {
        std::rethrow_exception(exception);
      } catch (const sycl::exception& error) {
        std::printf("Caught asynchronous SYCL exception:\n%s\n", error.what());
        throw;
      }
    }
  };
}

struct C29State {
  sycl::device device;
  sycl::queue queue;
  C29Buffers buffers;
  std::shared_ptr<C29ResultState> result_state;

  explicit C29State(const std::string& dev_str)
      : device(get_dev(dev_str)),
        queue(c29_profile_enabled()
                  ? sycl::queue{device, c29_exception_handler(),
                                sycl::property_list{sycl::property::queue::in_order{},
                                                    sycl::property::queue::enable_profiling{}}}
                  : sycl::queue{device, c29_exception_handler(),
                                sycl::property_list{sycl::property::queue::in_order{}}}) {
    result_state = std::make_shared<C29ResultState>();
  }

  ~C29State() {
    sycl_cleanup_noexcept("c29", [&] {
      queue.wait_and_throw();
    });
  }
};

static DeviceStateRegistry<C29State>& c29_registry() {
  static auto* const registry = new DeviceStateRegistry<C29State>;
  return *registry;
}

static C29State& c29_state(const std::string& dev_str) {
  return c29_registry().get(dev_str, [&] { return std::make_unique<C29State>(dev_str); });
}

void c29_cleanup_states() noexcept {
  try {
    c29_registry().clear();
  } catch (...) {
    std::fprintf(stderr, "c29: ordered SYCL cleanup failed\n");
  }
}

template <typename T>
static void c29_read_buffer(sycl::queue& queue, sycl::buffer<T, 1>& buffer,
                            T* const dest, const size_t count) {
  if (count == 0) return;

  const sycl::event event = queue.submit([&](sycl::handler& handler) {
    sycl::accessor accessor{
      buffer, handler, sycl::range<1>{count}, sycl::id<1>{0}, sycl::read_only
    };
    handler.copy(accessor, dest);
  });
  sycl_wait_and_throw(event, queue.get_device());
}

static void c29_activate_job(const std::shared_ptr<C29ResultState>& result_state,
                             const unsigned job_ref) {
  std::lock_guard<std::mutex> lock(result_state->mutex);
  if (result_state->active_job_ref == job_ref)
    return;

  result_state->active_job_ref = job_ref;
  // Keep detached searches isolated to the device's current job.
  for (auto it = result_state->solutions.begin(); it != result_state->solutions.end();) {
    if (it->job_ref != job_ref)
      it = result_state->solutions.erase(it);
    else
      ++it;
  }
}
