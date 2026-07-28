#include "../lib-internal.h"
#include "../../native/consts.h"
#include "crypto/randomx/blake2/blake2.h"

#if defined(_WIN32)
#include <windows.h>
#endif

// Cuckaroo Cycle algorithm constants and fixed kernel sizing.
const constexpr uint64_t DUCK_SIZE_A = 129, DUCK_SIZE_B = 83, INDEX_SIZE = 4096;

const constexpr uint64_t BUFFER_SIZE_A1 = DUCK_SIZE_A * 1024 * (4096 - 128) * 2,
                         BUFFER_SIZE_A2 = DUCK_SIZE_A * 1024 * 256 * 2,
                         BUFFER_SIZE_B  = DUCK_SIZE_B * 1024 * 4096 * 2;

const constexpr uint32_t COMPUTE_THREADS = 1024, TRIMMING_ROUNDS = 80,
                         EDGE_BLOCK_SIZE = 64, EDGE_BITS = 29,
                         BUCKET_MASK_4K = 4095, BUCKET_OFFSET = 255,
                         BUCKET_STEP = 32, EDGE_COUNTER_WORDS = 8192,
                         DEFAULT_SEED_BLOCKS = 16;

const constexpr uint32_t MAX_TRIMMED_EDGE_COUNT = 128 * COMPUTE_THREADS,
                         EDGE_BLOCK_MASK = EDGE_BLOCK_SIZE - 1,
                         NUM_EDGES = (1u << EDGE_BITS),
                         EDGE_MASK = NUM_EDGES - 1;

static uint32_t c29_env_u32(const char* name, const uint32_t default_value) {
  const char* const value = std::getenv(name);
  if (!value || !value[0]) return default_value;

  const unsigned long parsed = std::strtoul(value, nullptr, 10);
  return parsed == 0 ? default_value : static_cast<uint32_t>(parsed);
}

static bool c29_profile_enabled() { static const bool v = [](){ const char* e = std::getenv("MOM_C29_PROFILE"); return e && e[0] && e[0] != '0'; }(); return v; }

static uint32_t c29_profile_limit() { static const uint32_t v = [](){ const char* e = std::getenv("MOM_C29_PROFILE_LIMIT"); if (!e || !e[0]) return 1u; const unsigned long p = std::strtoul(e, nullptr, 10); return p == 0 ? UINT32_MAX : static_cast<uint32_t>(p); }(); return v; }

uint32_t c29_seed_local_size() { static const uint32_t v = [](){ const uint32_t r = c29_env_u32("MOM_C29_SEED_LOCAL_SIZE", 128u); return (r == 64u || r == 128u || r == 256u) ? r : 128u; }(); return v; }

uint32_t c29_seed_blocks_per_item() { static const uint32_t v = [](){ const uint32_t r = c29_env_u32("MOM_C29_SEED_BLOCKS", DEFAULT_SEED_BLOCKS); return (r == 4u || r == 8u || r == 16u || r == 32u) ? r : DEFAULT_SEED_BLOCKS; }(); return v; }

static bool c29_direct_seed_enabled() { static const bool v = [](){ const char* e = std::getenv("MOM_C29_DIRECT_SEED"); return !(e && e[0] == '0'); }(); return v; }

static uint64_t c29_now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()
  ).count();
}

struct C29ProfileEvent { const char* name; sycl::event event; };

struct C29Profile {
  bool enabled;
  uint64_t start_us;
  uint64_t last_us;
  std::vector<C29ProfileEvent> events;
  C29Profile(const unsigned job_ref, const uint64_t nonce, const unsigned proof_size,
             const sycl::queue& queue, const std::string& dev_str)
      : enabled(false), start_us(c29_now_us()), last_us(start_us) {
    static std::atomic<uint32_t> profiled_graphs{0};
    enabled = c29_profile_enabled() && profiled_graphs.fetch_add(1, std::memory_order_relaxed) < c29_profile_limit();
    if (!enabled) return;

    const sycl::device dev = queue.get_device();
    std::printf("[c29-profile] graph start job=%u nonce=%llu proof=%u dev=%s device=\"%s\" platform=\"%s\" "
                "index_size=%llu trim_rounds=%u compute_threads=%u seed_local=%u seed_blocks=%u direct_seed=%u\n",
                job_ref, static_cast<unsigned long long>(nonce), proof_size, dev_str.c_str(),
                dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_platform().get_info<sycl::info::platform::name>().c_str(),
                static_cast<unsigned long long>(INDEX_SIZE), TRIMMING_ROUNDS, COMPUTE_THREADS,
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
  sycl::buffer<uint32_t, 1> buffer_i1{sycl::range<1>{INDEX_SIZE}};
  sycl::buffer<uint32_t, 1> buffer_i2{sycl::range<1>{INDEX_SIZE}};
  sycl::buffer<uint32_t, 1> buffer_trimmed_edge_count{sycl::range<1>{1}};
  sycl::buffer<sycl::uint2, 1> buffer_trimmed_edges_u2{sycl::range<1>{MAX_TRIMMED_EDGE_COUNT}};
};

using C29GlobalAtomicRef = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
  sycl::memory_scope::device, sycl::access::address_space::global_space>;

template <typename Accessor>
inline C29GlobalAtomicRef c29_global_atomic(const Accessor& accessor, const size_t index) {
  auto pointer = accessor.template get_multi_ptr<sycl::access::decorated::yes>();
  return C29GlobalAtomicRef{pointer[index]};
}

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
  MomKernelBundle bundle;
  C29Buffers buffers;

  explicit C29State(const std::string& dev_str)
      : device(get_dev(dev_str)),
        queue(c29_profile_enabled()
          ? sycl::queue{device, c29_exception_handler(),
                        sycl::property_list{sycl::property::queue::in_order{},
                                            sycl::property::queue::enable_profiling{}}}
          : sycl::queue{device, c29_exception_handler(),
                        sycl::property_list{sycl::property::queue::in_order{}}}),
        bundle(mom_get_exec_bundle(queue.get_context())) {}

  ~C29State() { sycl_cleanup_noexcept("c29", [&] { queue.wait_and_throw(); }); }
};

using C29StateMap = std::map<std::string, std::unique_ptr<C29State>>;

static C29StateMap& c29_states() {
  static C29StateMap* const states = new C29StateMap;
  return *states;
}

static std::mutex& c29_states_mutex() {
  static std::mutex* const mutex = new std::mutex;
  return *mutex;
}

static C29State& c29_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(c29_states_mutex());
  auto& state = c29_states()[dev_str];
  if (!state) state = std::make_unique<C29State>(dev_str);
  return *state;
}

void c29_cleanup_states() noexcept {
  try {
    std::lock_guard<std::mutex> lock(c29_states_mutex());
    c29_states().clear();
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

// Global solution management
class C29SolutionMutex {
  std::atomic_flag m_flag = ATOMIC_FLAG_INIT;

public:
  void lock() {
    while (m_flag.test_and_set(std::memory_order_acquire)) {
#if defined(_WIN32)
      Sleep(0);
#else
      std::this_thread::yield();
#endif
    }
  }

  void unlock() { m_flag.clear(std::memory_order_release); }
};

class C29SolutionLock {
  C29SolutionMutex& m_mutex;

public:
  explicit C29SolutionLock(C29SolutionMutex& mutex) : m_mutex(mutex) { m_mutex.lock(); }
  ~C29SolutionLock() { m_mutex.unlock(); }

  C29SolutionLock(const C29SolutionLock&) = delete;
  C29SolutionLock& operator=(const C29SolutionLock&) = delete;
};

static std::list<std::vector<sycl::uint2>> global_solutions;
static std::list<std::vector<uint64_t>> global_seeds;
static std::list<unsigned> global_job_refs;
static std::list<uint64_t> global_nonces;
static C29SolutionMutex global_solutions_mutex;
static std::atomic<uint32_t> running_search_threads{0};

// Walk the alternating bipartite graph from start_node, following each node's stored
// successor until a dead end (or max_path_length) is reached.
static std::vector<uint32_t> create_path(const std::unordered_map<uint32_t, uint32_t>& graph_u,
    const std::unordered_map<uint32_t, uint32_t>& graph_v, const bool start_in_u,
    const uint32_t start_node, const uint32_t max_path_length = 8192) {
  std::vector<uint32_t> path;
  path.reserve(64);
  path.push_back(start_node);

  const std::unordered_map<uint32_t, uint32_t>* current_graph = start_in_u ? &graph_u : &graph_v;
  bool current_in_u = start_in_u;
  auto it = current_graph->find(start_node);

  while (it != current_graph->end() && path.size() < max_path_length) {
    const uint32_t next_node = it->second;
    path.push_back(next_node);
    current_in_u = !current_in_u;
    current_graph = current_in_u ? &graph_u : &graph_v;
    it = current_graph->find(next_node);
  }

  return path;
}

// Reverse the direction of every edge along path, flipping which side (u/v) each link is stored on.
static void reverse_path(std::unordered_map<uint32_t, uint32_t>& graph_u, std::unordered_map<uint32_t, uint32_t>& graph_v,
    const std::vector<uint32_t>& path, const bool starts_in_u) {
  std::unordered_map<uint32_t, uint32_t>* graphs[2] = {&graph_u, &graph_v};

  for (int32_t i = static_cast<int32_t>(path.size()) - 2; i >= 0; i--) {
    const uint32_t node_a = path[i], node_b = path[i + 1];
    const uint32_t idx_remove = (starts_in_u ? (i & 1) : !(i & 1)) ? 1 : 0;
    const uint32_t idx_add    = 1 - idx_remove;
    graphs[idx_remove]->erase(node_a);
    (*graphs[idx_add])[node_b] = node_a;
  }
}

// Main function to find cycles in trimmed graph - returns all valid target_cycle_length-cycles
static std::list<std::vector<sycl::uint2>> find_cycles(const std::vector<sycl::uint2>& trimmed_edges,
                                                       const uint32_t target_cycle_length) {
  const uint32_t edge_count = trimmed_edges.size();

  std::unordered_map<uint32_t, uint32_t> graph_u, graph_v;
  graph_u.reserve(edge_count);
  graph_v.reserve(edge_count);
  std::list<std::vector<sycl::uint2>> solutions;

  for (uint32_t edge_idx = 0; edge_idx < edge_count; edge_idx++) {
    const uint32_t node_u = trimmed_edges[edge_idx].x(), node_v = trimmed_edges[edge_idx].y();

    // Skip edges already present (find result reused to avoid a second lookup).
    auto it_u = graph_u.find(node_u);
    if (it_u != graph_u.end() && it_u->second == node_v) continue;

    auto it_v = graph_v.find(node_v);
    if (it_v != graph_v.end() && it_v->second == node_u) continue;

    // Walk from both endpoints; a shared node means the new edge closes a cycle.
    const std::vector<uint32_t> path_from_u = create_path(graph_u, graph_v, true, node_u);
    const std::vector<uint32_t> path_from_v = create_path(graph_u, graph_v, false, node_v);

    // Find the first intersection with a simple O(n*m) scan.
    int64_t join_a = -1, join_b = -1;
    for (uint32_t i = 0; i < path_from_u.size(); i++) {
      const auto it = std::find(path_from_v.begin(), path_from_v.end(), path_from_u[i]);
      if (it != path_from_v.end()) {
        join_a = i;
        join_b = it - path_from_v.begin();
        break;
      }
    }

    const int64_t cycle_length = join_a != -1 ? 1 + join_a + join_b : 0;

    if (cycle_length == target_cycle_length) {
      // Closing edge plus the two paths back to the join point form the cycle.
      std::vector<sycl::uint2> cycle_edges;
      cycle_edges.reserve(target_cycle_length);
      cycle_edges.push_back({node_u, node_v});
      for (int64_t i = 0; i < join_a; i++)
        cycle_edges.push_back({path_from_u[i + 1], path_from_u[i]});
      for (int64_t i = 0; i < join_b; i++)
        cycle_edges.push_back({path_from_v[i + 1], path_from_v[i]});

      solutions.push_back(std::move(cycle_edges));
    } else {
      // No cycle yet: splice the new edge in by reversing the shorter path (cheaper).
      if (path_from_u.size() > path_from_v.size()) {
        reverse_path(graph_u, graph_v, path_from_v, false);
        graph_v[node_v] = node_u;
      } else {
        reverse_path(graph_u, graph_v, path_from_u, true);
        graph_u[node_u] = node_v;
      }
    }
  }

  return solutions;
}
