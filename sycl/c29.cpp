// Copyright GNU GPLv3 (c) 2025-2025 MoneroOcean <support@moneroocean.stream>

// SYCL c29 miner prototype based on Grin GPU Miner (https://github.com/swap-dev/SwapReferenceMiner)
// OpenCL mining code by Jiri Photon Vadura and John Tromp
#include <sycl/sycl.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <thread>
#include <unordered_map>
#include <vector>

#include "lib-internal.h"
#include "../native/consts.h"
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
                         DEFAULT_SEED_BLOCKS = 32;

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

static uint32_t c29_seed_local_size() { static const uint32_t v = [](){ const uint32_t r = c29_env_u32("MOM_C29_SEED_LOCAL_SIZE", 128u); return (r == 64u || r == 128u || r == 256u) ? r : 128u; }(); return v; }

static uint32_t c29_seed_blocks_per_item() { static const uint32_t v = [](){ const uint32_t r = c29_env_u32("MOM_C29_SEED_BLOCKS", DEFAULT_SEED_BLOCKS); return (r == 4u || r == 8u || r == 16u || r == 32u) ? r : DEFAULT_SEED_BLOCKS; }(); return v; }

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

static C29Buffers& get_c29_buffers() { static C29Buffers buffers; return buffers; }

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

    // Find the first intersection (simple O(n*m) scan, as in the reference miner).
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

// SipHash cryptographic round function for edge generation
#define SIPHASH_ROUND_LAMBDA_MACRO\
  auto siphash_round = [](uint64_t& v0, uint64_t& v1, uint64_t& v2, uint64_t& v3) {\
    v0 += v1; v2 += v3; v1 = mo_rotate(v1, static_cast<uint64_t>(13));\
    v3 = mo_rotate(v3, static_cast<uint64_t>(16)); v1 ^= v0; v3 ^= v2;\
    v0 = mo_rotate(v0, static_cast<uint64_t>(32)); v2 += v1; v0 += v3;\
    v1 = mo_rotate(v1, static_cast<uint64_t>(17)); v3 = mo_rotate(v3, static_cast<uint64_t>(21));\
    v1 ^= v2; v3 ^= v0; v2 = mo_rotate(v2, static_cast<uint64_t>(32));\
  };
#define SIPHASH_FILL_BLOCK(V0, V1, V2, V3, BASE, OUT, I)\
  for (uint32_t I = 0; I < EDGE_BLOCK_SIZE; I++) {\
    V3 ^= BASE + I; siphash_round(V0, V1, V2, V3); siphash_round(V0, V1, V2, V3);\
    V0 ^= BASE + I; V2 ^= 0xff;\
    siphash_round(V0, V1, V2, V3); siphash_round(V0, V1, V2, V3);\
    siphash_round(V0, V1, V2, V3); siphash_round(V0, V1, V2, V3);\
    OUT[I] = (V0 ^ V1) ^ (V2 ^ V3);\
  }
#define C29_WRITE8(DEST, INDEX, STORE, BUCKET, BASE)\
  do {\
    (DEST)[(INDEX)] = sycl::ulong4(STORE[(BUCKET)][(BASE)], STORE[(BUCKET)][(BASE) + 1], STORE[(BUCKET)][(BASE) + 2], STORE[(BUCKET)][(BASE) + 3]);\
    (DEST)[(INDEX) + 1] = sycl::ulong4(STORE[(BUCKET)][(BASE) + 4], STORE[(BUCKET)][(BASE) + 5], STORE[(BUCKET)][(BASE) + 6], STORE[(BUCKET)][(BASE) + 7]);\
  } while (false)

// Worker function that performs GPU trimming and cycle finding in separate thread
static void start_new_c29_solution_search(const uint64_t seed_k0, const uint64_t seed_k1,
                                          const uint64_t seed_k2, const uint64_t seed_k3,
                                          const unsigned job_ref, const uint64_t nonce,
                                          const unsigned c29_proof_size,
                                          sycl::queue& compute_queue) {
  try {
    static auto kernel_bundle = MOM_GET_EXEC_BUNDLE(compute_queue.get_context());
    C29Profile profile(job_ref, nonce, c29_proof_size, compute_queue, compute_queue.get_device().get_info<sycl::info::device::name>());
    C29Buffers& c29_buffers = get_c29_buffers();

    // Reuse the persistent GPU buffers across graphs, reinterpreting each as uint2 (edge pairs)
    // and ulong4 (8-word block writes) for the different kernel access patterns.
    auto buffer_a1_u2  = c29_buffers.buffer_a1.reinterpret<sycl::uint2>(sycl::range(BUFFER_SIZE_A1 / 2));
    auto buffer_a1_ul4 = c29_buffers.buffer_a1.reinterpret<sycl::ulong4>(sycl::range(BUFFER_SIZE_A1 / 8));
    auto buffer_a2_u2  = c29_buffers.buffer_a2.reinterpret<sycl::uint2>(sycl::range(BUFFER_SIZE_A2 / 2));
    auto buffer_a2_ul4 = c29_buffers.buffer_a2.reinterpret<sycl::ulong4>(sycl::range(BUFFER_SIZE_A2 / 8));
    auto buffer_b_u2   = c29_buffers.buffer_b.reinterpret<sycl::uint2>(sycl::range(BUFFER_SIZE_B / 2));
    auto buffer_b_ul4  = c29_buffers.buffer_b.reinterpret<sycl::ulong4>(sycl::range(BUFFER_SIZE_B / 8));
    sycl::buffer<uint32_t, 1>& buffer_i1 = c29_buffers.buffer_i1;
    sycl::buffer<uint32_t, 1>& buffer_i2 = c29_buffers.buffer_i2;
    sycl::buffer<uint32_t, 1>& buffer_trimmed_edge_count = c29_buffers.buffer_trimmed_edge_count;
    sycl::buffer<sycl::uint2, 1>& buffer_trimmed_edges_u2 = c29_buffers.buffer_trimmed_edges_u2;

    profile.mark("buffers ready");

    auto zero_buffer = [&](const char* name, sycl::buffer<uint32_t, 1>& buffer) {
      const sycl::event event = compute_queue.submit([&](sycl::handler& handler) {
        sycl::accessor accessor{buffer, handler, sycl::write_only, sycl::no_init};
        MOM_USE_BUNDLE(handler, kernel_bundle);
        handler.fill(accessor, static_cast<uint32_t>(0));
      });
      profile.add_event(name, event);
    };

    zero_buffer("clear_i1_start", buffer_i1);
    zero_buffer("clear_i2_start", buffer_i2);

    const uint32_t duck_edges_a = static_cast<uint32_t>(DUCK_SIZE_A) * 1024;
    const uint32_t duck_edges_b = static_cast<uint32_t>(DUCK_SIZE_B) * 1024;

    using local_atomic_ref  = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>;
    using global_atomic_ref = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;

    // FluffySeed2A: Generate initial edges using SipHash cryptographic function
    const uint32_t seed_local_size = c29_seed_local_size();
    const uint32_t seed_blocks_per_item = c29_seed_blocks_per_item();
    const uint32_t seed_work_items = NUM_EDGES / (EDGE_BLOCK_SIZE * seed_blocks_per_item);
    const bool direct_seed = c29_direct_seed_enabled();

    if (direct_seed) {
      const sycl::event direct_seed_event = compute_queue.submit([&](sycl::handler& handler) {
        sycl::accessor acc_buffer_a1{buffer_a1_u2, handler, sycl::write_only, sycl::no_init};
        sycl::accessor acc_buffer_a2{buffer_a2_u2, handler, sycl::write_only, sycl::no_init};
        sycl::accessor acc_index_2{buffer_i2, handler, sycl::read_write};
        MOM_USE_BUNDLE(handler, kernel_bundle);
        handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(seed_work_items), sycl::range<1>(seed_local_size)),
                            [=](sycl::nd_item<1> item) {
          const uint32_t global_id = item.get_global_id(0);

          SIPHASH_ROUND_LAMBDA_MACRO;

          for (uint32_t block_offset = 0; block_offset < seed_blocks_per_item * EDGE_BLOCK_SIZE; block_offset += EDGE_BLOCK_SIZE) {
            const uint64_t base_nonce = global_id * (seed_blocks_per_item * EDGE_BLOCK_SIZE) + block_offset;
            uint64_t sip_v0 = seed_k0, sip_v1 = seed_k1, sip_v2 = seed_k2, sip_v3 = seed_k3;
            uint64_t hash_block[EDGE_BLOCK_SIZE];

            #pragma vector always
            SIPHASH_FILL_BLOCK(sip_v0, sip_v1, sip_v2, sip_v3, base_nonce, hash_block, nonce_offset);

            const uint64_t hash_last = hash_block[EDGE_BLOCK_MASK];

            for (uint32_t hash_index = 0; hash_index < EDGE_BLOCK_SIZE; hash_index++) {
              const uint64_t hash_lookup = hash_index == EDGE_BLOCK_MASK ? hash_last : hash_block[hash_index] ^ hash_last;
              const uint32_t edge_u      = static_cast<uint32_t>(hash_lookup & EDGE_MASK);
              const uint32_t edge_v      = static_cast<uint32_t>((hash_lookup >> 32) & EDGE_MASK);
              if (!(edge_u || edge_v)) continue;

              const uint32_t bucket_id = ((edge_u & 63u) << 6) | ((edge_u >> 6) & 63u);
              const uint32_t bucket_index = sycl::min(global_atomic_ref(acc_index_2[bucket_id]).fetch_add(1), duck_edges_a - 1);

              if (bucket_id < 62 * 64)
                acc_buffer_a1[bucket_id * duck_edges_a + bucket_index] = sycl::uint2(edge_u, edge_v);
              else
                acc_buffer_a2[(bucket_id - 62 * 64) * duck_edges_a + bucket_index] = sycl::uint2(edge_u, edge_v);
            }
          }
        });
      });
      profile.add_event("seed2a_direct", direct_seed_event);
    } else {
      const sycl::event seed_event = compute_queue.submit([&](sycl::handler& handler) {
        sycl::accessor acc_buffer_b{buffer_b_ul4, handler, sycl::write_only, sycl::no_init};
        sycl::accessor acc_buffer_a1{buffer_a1_ul4, handler, sycl::write_only, sycl::no_init};
        sycl::accessor acc_index_1{buffer_i1, handler, sycl::read_write};

        // Local memory for temporary edge storage and bucket counters
        sycl::local_accessor<uint64_t, 2> temp_storage{sycl::range<2>(64, 16), handler};
        sycl::local_accessor<uint32_t, 1> bucket_counters{sycl::range<1>(64), handler};
        MOM_USE_BUNDLE(handler, kernel_bundle);
        handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(seed_work_items), sycl::range<1>(seed_local_size)),
                            [=](sycl::nd_item<1> item) {
          const uint32_t global_id = item.get_global_id(0);
          const uint32_t local_id  = item.get_local_id(0);

          if (local_id < 64) bucket_counters[local_id] = 0;

          item.barrier(sycl::access::fence_space::local_space);

          SIPHASH_ROUND_LAMBDA_MACRO;

          // Process nonces in blocks to generate graph edges efficiently
          for (uint32_t block_offset = 0; block_offset < seed_blocks_per_item * EDGE_BLOCK_SIZE; block_offset += EDGE_BLOCK_SIZE) {
            const uint64_t base_nonce = global_id * (seed_blocks_per_item * EDGE_BLOCK_SIZE) + block_offset;
            uint64_t sip_v0 = seed_k0, sip_v1 = seed_k1, sip_v2 = seed_k2, sip_v3 = seed_k3;
            uint64_t hash_block[EDGE_BLOCK_SIZE];

            #pragma vector always
            SIPHASH_FILL_BLOCK(sip_v0, sip_v1, sip_v2, sip_v3, base_nonce, hash_block, nonce_offset);

            const uint64_t hash_last = hash_block[EDGE_BLOCK_MASK];

            // Extract graph edges from hash values and distribute to buckets
            for (uint32_t hash_index = 0; hash_index < EDGE_BLOCK_SIZE; hash_index++) {
              const uint64_t hash_lookup = hash_index == EDGE_BLOCK_MASK ? hash_last : hash_block[hash_index] ^ hash_last;
              const uint32_t edge_u      = static_cast<uint32_t>(hash_lookup & EDGE_MASK);
              const uint32_t edge_v      = static_cast<uint32_t>((hash_lookup >> 32) & EDGE_MASK);
              const uint32_t bucket_id   = edge_u & 63;
              const uint32_t counter       = local_atomic_ref(bucket_counters[bucket_id]).fetch_add(1);
              const uint32_t counter_local = counter % 16;
              temp_storage[bucket_id][counter_local] = edge_u | (static_cast<uint64_t>(edge_v) << 32);

              item.barrier(sycl::access::fence_space::local_space);

              // Write accumulated edges when buffer reaches threshold
              if ((counter > 0) && (counter_local == 0 || counter_local == 8)) {
                const uint32_t write_count = sycl::min(global_atomic_ref(acc_index_1[bucket_id]).fetch_add(8), duck_edges_a * 64 - 8);
                const uint32_t write_index = ((bucket_id < 32 ? bucket_id : bucket_id - 32) * duck_edges_a * 64 + write_count) >> 2;
                auto* const dest_buffer    = bucket_id < 32 ? &acc_buffer_b : &acc_buffer_a1;
                C29_WRITE8((*dest_buffer), write_index, temp_storage, bucket_id, 8 - counter_local);

                // Clear written entries for reuse
                for (uint32_t clear_index = 0; clear_index < 8; clear_index++) temp_storage[bucket_id][8 - counter_local + clear_index] = 0;
              }
            }
          }

          item.barrier(sycl::access::fence_space::local_space);

          // Final flush of remaining edges in local buffers
          if (local_id < 64) {
            const uint32_t final_counter     = bucket_counters[local_id];
            const uint32_t final_counter_base = (final_counter % 16) >= 8 ? 8 : 0;
            const uint32_t final_write_count  = sycl::min(global_atomic_ref(acc_index_1[local_id]).fetch_add(8), duck_edges_a * 64 - 8);
            const uint32_t final_write_index  = ((local_id < 32 ? local_id : local_id - 32) * duck_edges_a * 64 + final_write_count) >> 2;
            auto* const final_dest_buffer     = local_id < 32 ? &acc_buffer_b : &acc_buffer_a1;
            C29_WRITE8((*final_dest_buffer), final_write_index, temp_storage, local_id, final_counter_base);
          }
        });
      });

      profile.add_event("seed2a", seed_event);

      // FluffySeed2B: Redistribute edges to smaller buckets for better memory access patterns
      auto redistribute_edges = [](sycl::queue& queue, sycl::buffer<sycl::uint2, 1>& source_buffer,
                                   sycl::buffer<sycl::ulong4, 1>& dest_buffer_1, sycl::buffer<sycl::ulong4, 1>& dest_buffer_2,
                                   sycl::buffer<uint32_t, 1>& source_indexes, sycl::buffer<uint32_t, 1>& dest_indexes,
                                   const uint32_t start_block, const uint32_t duck_edges_a, auto& kernel_bundle) {
        return queue.submit([&](sycl::handler& handler) {
          sycl::accessor acc_source{source_buffer, handler, sycl::read_only};
          sycl::accessor acc_dest_1{dest_buffer_1, handler, sycl::write_only, sycl::no_init};
          sycl::accessor acc_dest_2{dest_buffer_2, handler, sycl::write_only, sycl::no_init};
          sycl::accessor acc_source_indexes{source_indexes, handler, sycl::read_only};
          sycl::accessor acc_dest_indexes{dest_indexes, handler, sycl::read_write};
          const constexpr uint32_t BUCKET_GRANULARITY = 32;
          sycl::local_accessor<uint64_t, 2> temp_storage{sycl::range<2>(64, 16), handler};
          sycl::local_accessor<uint32_t, 1> bucket_counters{sycl::range<1>(64), handler};
          MOM_USE_BUNDLE(handler, kernel_bundle);
          handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(1024 * 128), sycl::range<1>(128)),
                              [=](sycl::nd_item<1> item) {
            const uint32_t local_id      = item.get_local_id(0);
            const uint32_t work_group_id = item.get_group(0);
            if (local_id < 64) bucket_counters[local_id] = 0;

            item.barrier(sycl::access::fence_space::local_space);

            const uint32_t current_bucket         = work_group_id / BUCKET_GRANULARITY;
            const uint32_t micro_block_number     = work_group_id % BUCKET_GRANULARITY;
            const uint32_t bucket_edge_count      = sycl::min(acc_source_indexes[current_bucket + start_block], duck_edges_a * 64);
            const uint32_t micro_block_edge_count = duck_edges_a * 64 / BUCKET_GRANULARITY;
            const uint32_t processing_loops       = micro_block_edge_count / 128;
            const bool     use_dest_buffer_2      = start_block == 32 && current_bucket >= 30;
            const uint32_t memory_offset          = use_dest_buffer_2 ? 0 : start_block * duck_edges_a * 64;
            const uint32_t bucket_offset          = use_dest_buffer_2 ? 30 : 0;
            auto* const    destination_buffer     = use_dest_buffer_2 ? &acc_dest_2 : &acc_dest_1;

            for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
              const uint32_t edge_index = micro_block_number * micro_block_edge_count + 128 * loop_index + local_id;
              const sycl::uint2 current_edge = edge_index < bucket_edge_count ? acc_source[(current_bucket * duck_edges_a * 64) + edge_index]
                                                                               : sycl::uint2(0, 0);
              const bool skip_edge      = (edge_index >= bucket_edge_count) || (current_edge.x() == 0 && current_edge.y() == 0);
              const uint32_t bucket_id  = (current_edge.x() >> 6) & (64 - 1);
              uint32_t edge_counter = 0, local_counter = 0;

              if (!skip_edge) {
                edge_counter  = local_atomic_ref(bucket_counters[bucket_id]).fetch_add(1);
                local_counter = edge_counter % 16;
                temp_storage[bucket_id][local_counter] = current_edge.x() | (static_cast<uint64_t>(current_edge.y()) << 32);
              }

              item.barrier(sycl::access::fence_space::local_space);

              if ((edge_counter > 0) && (local_counter == 0 || local_counter == 8)) {
                const uint32_t write_count = sycl::min(global_atomic_ref(acc_dest_indexes[start_block * 64 + current_bucket * 64 + bucket_id]).fetch_add(8), duck_edges_a - 8);
                const uint32_t write_index = (memory_offset + (((current_bucket - bucket_offset) * 64 + bucket_id) * duck_edges_a + write_count)) >> 2;
                C29_WRITE8((*destination_buffer), write_index, temp_storage, bucket_id, 8 - local_counter);
                for (uint32_t clear_index = 0; clear_index < 8; clear_index++) temp_storage[bucket_id][8 - local_counter + clear_index] = 0;
              }
            }

            item.barrier(sycl::access::fence_space::local_space);

            // Final flush for remaining edges
            if (local_id < 64) {
              const uint32_t final_counter      = bucket_counters[local_id];
              const uint32_t final_counter_base = (final_counter % 16) >= 8 ? 8 : 0;
              const uint32_t final_write_count  = sycl::min(global_atomic_ref(acc_dest_indexes[start_block * 64 + current_bucket * 64 + local_id]).fetch_add(8), duck_edges_a - 8);
              const uint32_t final_write_index  = (memory_offset + (((current_bucket - bucket_offset) * 64 + local_id) * duck_edges_a + final_write_count)) >> 2;
              C29_WRITE8((*destination_buffer), final_write_index, temp_storage, local_id, final_counter_base);
            }
          });
        });
      };

      profile.add_event("redistribute_32",
        redistribute_edges(compute_queue, buffer_a1_u2, buffer_a1_ul4, buffer_a2_ul4, buffer_i1, buffer_i2, 32, duck_edges_a, kernel_bundle));
      profile.add_event("redistribute_0",
        redistribute_edges(compute_queue, buffer_b_u2, buffer_a1_ul4, buffer_a2_ul4, buffer_i1, buffer_i2, 0, duck_edges_a, kernel_bundle));
      zero_buffer("clear_i1_seed", buffer_i1);
    }

    // Helper functions for 2-bit edge counters used in trimming phases
    auto increment_2bit_counter = [](const uint32_t bucket_id, sycl::local_accessor<uint32_t, 1> counters) -> void {
      const uint32_t word_index = bucket_id >> 5;
      const uint8_t bit_index   = bucket_id & 0x1F;
      const uint32_t bit_mask   = 1 << bit_index;
      const uint32_t old_value  = local_atomic_ref(counters[word_index]).fetch_or(bit_mask) & bit_mask;
      if (old_value > 0) local_atomic_ref(counters[word_index + 4096]).fetch_or(bit_mask);
    };

    auto read_2bit_counter = [](const uint32_t bucket_id, const sycl::local_accessor<uint32_t, 1>& counters) -> bool {
      const uint32_t word_index = bucket_id >> 5;
      const uint8_t bit_index   = bucket_id & 0x1F;
      const uint32_t bit_mask   = 1 << bit_index;
      return (counters[word_index + 4096] & bit_mask) > 0;
    };

    // FluffyRound1: First trimming round using 2-bit counters to remove low-degree edges
    const sycl::event round1_event = compute_queue.submit([&](sycl::handler& handler) {
      sycl::accessor acc_a1{buffer_a1_u2, handler, sycl::read_only};
      sycl::accessor acc_a2{buffer_a2_u2, handler, sycl::read_only};
      sycl::accessor acc_b{buffer_b_u2, handler, sycl::write_only, sycl::no_init};
      sycl::accessor acc_i2{buffer_i2, handler, sycl::read_write};
      sycl::accessor acc_i1{buffer_i1, handler, sycl::read_write};
      sycl::local_accessor<uint32_t, 1> edge_counters{sycl::range<1>(EDGE_COUNTER_WORDS), handler};
      MOM_USE_BUNDLE(handler, kernel_bundle);
      handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(4096 * COMPUTE_THREADS), sycl::range<1>(COMPUTE_THREADS)),
                          [=](sycl::nd_item<1> item) {
        const uint32_t local_id      = item.get_local_id(0);
        const uint32_t work_group_id = item.get_group(0);
        const bool is_source_1    = work_group_id < (62 * 64);
        auto* const source_buffer = is_source_1 ? &acc_a1 : &acc_a2;
        const uint32_t read_group = is_source_1 ? work_group_id : work_group_id - (62 * 64);

        for (uint32_t counter_index = local_id; counter_index < EDGE_COUNTER_WORDS; counter_index += COMPUTE_THREADS)
          edge_counters[counter_index] = 0;

        item.barrier(sycl::access::fence_space::local_space);

        const uint32_t edges_in_bucket = sycl::min(acc_i2[work_group_id], duck_edges_a);
        const uint32_t processing_loops = (edges_in_bucket + COMPUTE_THREADS - 1) / COMPUTE_THREADS;

        // First pass: count edge occurrences to identify degree-2+ edges
        for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
          const uint32_t edge_local_index = loop_index * COMPUTE_THREADS + local_id;
          if (edge_local_index < edges_in_bucket) {
            const sycl::uint2 current_edge = (*source_buffer)[duck_edges_a * read_group + edge_local_index];
            if (current_edge.x() || current_edge.y())
              increment_2bit_counter((current_edge.x() & EDGE_MASK) >> 12, edge_counters);
          }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Second pass: write edges that appear multiple times (degree >= 2)
        for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
          const uint32_t edge_local_index = loop_index * COMPUTE_THREADS + local_id;
          if (edge_local_index < edges_in_bucket) {
            const sycl::uint2 current_edge = (*source_buffer)[duck_edges_a * read_group + edge_local_index];
            if ((current_edge.x() || current_edge.y()) && read_2bit_counter((current_edge.x() & EDGE_MASK) >> 12, edge_counters)) {
              const uint32_t bucket_id = current_edge.y() & BUCKET_MASK_4K;
              const uint32_t bucket_index = sycl::min(global_atomic_ref(acc_i1[bucket_id]).fetch_add(1), duck_edges_b - 1);
              acc_b[bucket_id * duck_edges_b + bucket_index] = sycl::uint2(current_edge.y(), current_edge.x()); // Swap edge direction
            }
          }
        }

        if (local_id == 0) acc_i2[work_group_id] = 0;
      });
    });

    profile.add_event("round1", round1_event);

    // FluffyRoundNO1: First trimming round with memory offset optimization
    const sycl::event round_no1_event = compute_queue.submit([&](sycl::handler& handler) {
      sycl::accessor acc_b{buffer_b_u2, handler, sycl::read_only};
      sycl::accessor acc_a1{buffer_a1_u2, handler, sycl::write_only, sycl::no_init};
      sycl::accessor acc_i1{buffer_i1, handler, sycl::read_write};
      sycl::accessor acc_i2{buffer_i2, handler, sycl::read_write};
      sycl::local_accessor<uint32_t, 1> edge_counters{sycl::range<1>(EDGE_COUNTER_WORDS), handler};
      MOM_USE_BUNDLE(handler, kernel_bundle);
      handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(4096 * COMPUTE_THREADS), sycl::range<1>(COMPUTE_THREADS)),
                          [=](sycl::nd_item<1> item) {
        const uint32_t local_id      = item.get_local_id(0);
        const uint32_t work_group_id = item.get_group(0);

        for (uint32_t counter_index = local_id; counter_index < EDGE_COUNTER_WORDS; counter_index += COMPUTE_THREADS)
          edge_counters[counter_index] = 0;

        item.barrier(sycl::access::fence_space::local_space);

        const uint32_t edges_in_bucket = sycl::min(acc_i1[work_group_id], duck_edges_b);
        const uint32_t processing_loops = (edges_in_bucket + COMPUTE_THREADS - 1) / COMPUTE_THREADS;

        // First pass: count edge occurrences
        for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
          const uint32_t edge_local_index = loop_index * COMPUTE_THREADS + local_id;
          if (edge_local_index < edges_in_bucket) {
            const sycl::uint2 current_edge = acc_b[(duck_edges_b * work_group_id) + edge_local_index];
            if (current_edge.x() || current_edge.y())
              increment_2bit_counter((current_edge.x() & EDGE_MASK) >> 12, edge_counters);
          }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Second pass: write filtered edges with memory offset for better cache locality
        for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
          const uint32_t edge_local_index = loop_index * COMPUTE_THREADS + local_id;
          if (edge_local_index < edges_in_bucket) {
            const sycl::uint2 current_edge = acc_b[(duck_edges_b * work_group_id) + edge_local_index];
            if ((current_edge.x() || current_edge.y()) && read_2bit_counter((current_edge.x() & EDGE_MASK) >> 12, edge_counters)) {
              const uint32_t bucket_id = current_edge.y() & BUCKET_MASK_4K;
              const uint32_t bucket_index = sycl::min(global_atomic_ref(acc_i2[bucket_id]).fetch_add(1),
                                                      duck_edges_b - 1 - ((bucket_id & BUCKET_OFFSET) * BUCKET_STEP));
              acc_a1[((bucket_id & BUCKET_OFFSET) * BUCKET_STEP) + (bucket_id * duck_edges_b) + bucket_index] =
                sycl::uint2(current_edge.y(), current_edge.x());
            }
          }
        }

        if (local_id == 0) acc_i1[work_group_id] = 0;
      });
    });

    profile.add_event("round_no1", round_no1_event);

    // FluffyRoundNON: Subsequent trimming rounds with memory offset optimization
    auto trim_round_with_offset = [&](sycl::queue& queue, sycl::buffer<sycl::uint2, 1>& source_buffer,
                                      sycl::buffer<sycl::uint2, 1>& dest_buffer, sycl::buffer<uint32_t, 1>& source_indexes,
                                      sycl::buffer<uint32_t, 1>& dest_indexes) {
      return queue.submit([&](sycl::handler& handler) {
        sycl::accessor acc_source{source_buffer, handler, sycl::read_only};
        sycl::accessor acc_dest{dest_buffer, handler, sycl::write_only, sycl::no_init};
        sycl::accessor acc_source_indexes{source_indexes, handler, sycl::read_write};
        sycl::accessor acc_dest_indexes{dest_indexes, handler, sycl::read_write};
        sycl::local_accessor<uint32_t, 1> edge_counters{sycl::range<1>(EDGE_COUNTER_WORDS), handler};
        MOM_USE_BUNDLE(handler, kernel_bundle);
        handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(4096 * COMPUTE_THREADS), sycl::range<1>(COMPUTE_THREADS)),
                            [=](sycl::nd_item<1> item) {
          const uint32_t local_id      = item.get_local_id(0);
          const uint32_t work_group_id = item.get_group(0);

          for (uint32_t counter_index = local_id; counter_index < EDGE_COUNTER_WORDS; counter_index += COMPUTE_THREADS)
            edge_counters[counter_index] = 0;

          item.barrier(sycl::access::fence_space::local_space);

          // Clamp the read count to the per-bucket written region: the writer (skewed kernel above)
          // caps each bucket index at duck_edges_b - 1 - ((bucket & BUCKET_OFFSET) * BUCKET_STEP), and
          // the reads here add the same skew, so without this the top work-groups read past the highest
          // written index (OOB device read / stale cross-bucket data). Mirrors the write clamp exactly.
          const uint32_t edges_in_bucket = sycl::min(acc_source_indexes[work_group_id],
                                                     duck_edges_b - ((work_group_id & BUCKET_OFFSET) * BUCKET_STEP));
          const uint32_t processing_loops = (edges_in_bucket + COMPUTE_THREADS - 1) / COMPUTE_THREADS;

          // First pass: count edge occurrences
          for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
            const uint32_t edge_local_index = loop_index * COMPUTE_THREADS + local_id;
            if (edge_local_index < edges_in_bucket) {
              const sycl::uint2 current_edge = acc_source[((work_group_id & BUCKET_OFFSET) * BUCKET_STEP) +
                                                          (duck_edges_b * work_group_id) + edge_local_index];
              if (current_edge.x() || current_edge.y())
                increment_2bit_counter((current_edge.x() & EDGE_MASK) >> 12, edge_counters);
            }
          }

          item.barrier(sycl::access::fence_space::local_space);

          // Second pass: write filtered edges
          for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
            const uint32_t edge_local_index = loop_index * COMPUTE_THREADS + local_id;
            if (edge_local_index < edges_in_bucket) {
              const sycl::uint2 current_edge = acc_source[((work_group_id & BUCKET_OFFSET) * BUCKET_STEP) +
                                                          (duck_edges_b * work_group_id) + edge_local_index];
              if ((current_edge.x() || current_edge.y()) && read_2bit_counter((current_edge.x() & EDGE_MASK) >> 12, edge_counters)) {
                const uint32_t bucket_id = current_edge.y() & BUCKET_MASK_4K;
                const uint32_t bucket_index = sycl::min(global_atomic_ref(acc_dest_indexes[bucket_id]).fetch_add(1),
                                                        duck_edges_b - 1 - ((bucket_id & BUCKET_OFFSET) * BUCKET_STEP));
                acc_dest[((bucket_id & BUCKET_OFFSET) * BUCKET_STEP) + (bucket_id * duck_edges_b) + bucket_index] =
                  sycl::uint2(current_edge.y(), current_edge.x());
              }
            }
          }

          if (local_id == 0) acc_source_indexes[work_group_id] = 0;
        });
      });
    };

    // Main trimming loop: iteratively reduce edge count by removing low-degree edges
    profile.add_event("trim_a_to_b_initial",
      trim_round_with_offset(compute_queue, buffer_a1_u2, buffer_b_u2, buffer_i2, buffer_i1));

    for (uint32_t round_index = 0; round_index < TRIMMING_ROUNDS; round_index++) {
      profile.add_event("trim_b_to_a",
        trim_round_with_offset(compute_queue, buffer_b_u2, buffer_a1_u2, buffer_i1, buffer_i2));
      profile.add_event("trim_a_to_b",
        trim_round_with_offset(compute_queue, buffer_a1_u2, buffer_b_u2, buffer_i2, buffer_i1));
    }

    // FluffyTailO: Collect final edges into contiguous output buffer
    uint32_t trimmed_edge_count = 0;
    zero_buffer("clear_trimmed_count", buffer_trimmed_edge_count);

    const sycl::event tail_event = compute_queue.submit([&](sycl::handler& handler) {
      sycl::accessor acc_b{buffer_b_u2, handler, sycl::read_only};
      sycl::accessor acc_edges{buffer_trimmed_edges_u2, handler, sycl::write_only, sycl::no_init};
      sycl::accessor acc_i1{buffer_i1, handler, sycl::read_write};
      sycl::accessor acc_trimmed_edge_count{buffer_trimmed_edge_count, handler, sycl::read_write};
      sycl::local_accessor<uint32_t, 1> output_index{sycl::range<1>(1), handler};
      MOM_USE_BUNDLE(handler, kernel_bundle);
      handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(4096 * COMPUTE_THREADS), sycl::range<1>(COMPUTE_THREADS)),
                          [=](sycl::nd_item<1> item) {
        const uint32_t local_id      = item.get_local_id(0);
        const uint32_t work_group_id = item.get_group(0);
        const uint32_t edges_to_copy = acc_i1[work_group_id];

        // Thread 0 reserves space in output buffer
        if (local_id == 0) output_index[0] = global_atomic_ref(acc_trimmed_edge_count[0]).fetch_add(edges_to_copy);

        item.barrier(sycl::access::fence_space::local_space);

        // Copy edges to contiguous output buffer
        for (uint32_t edge_index = local_id; edge_index < edges_to_copy; edge_index += COMPUTE_THREADS) {
          const uint32_t index = output_index[0] + edge_index;
          if (index < MAX_TRIMMED_EDGE_COUNT)
            acc_edges[index] = acc_b[((work_group_id & BUCKET_OFFSET) * BUCKET_STEP) + work_group_id * duck_edges_b + edge_index];
        }

        if (local_id == 0) acc_i1[work_group_id] = 0;
      });
    });

    profile.add_event("tail", tail_event);

    // Intel OpenCL may busy-spin a host core inside the blocking D2H read if the copy is enqueued
    // before the preceding trim kernels finish. Wait for the final trim event with our poll/sleep
    // helper first; then the small readbacks below only move already-ready data.
    sycl_wait_and_throw(tail_event, compute_queue.get_device());

    // Read final trimmed edges from GPU memory
    c29_read_buffer(compute_queue, buffer_trimmed_edge_count, &trimmed_edge_count, 1);
    trimmed_edge_count = sycl::min(trimmed_edge_count, MAX_TRIMMED_EDGE_COUNT);

    profile.mark("read trimmed count");
    profile.print_events();

    std::vector<sycl::uint2> trimmed_edges(trimmed_edge_count);

    c29_read_buffer(compute_queue, buffer_trimmed_edges_u2, trimmed_edges.data(), trimmed_edge_count);

    profile.mark("read trimmed edges");
    profile.finish(trimmed_edge_count);

    // Cycle search runs on a detached thread so the GPU queue is freed for the next graph.
    running_search_threads.fetch_add(1);
    std::thread([=]() {
      const std::list<std::vector<sycl::uint2>> solutions = find_cycles(trimmed_edges, c29_proof_size);

      if (!solutions.empty()) {
        C29SolutionLock lock(global_solutions_mutex);
        for (const auto& solution : solutions) {
          global_solutions.push_back(solution);
          global_seeds.push_back(std::vector<uint64_t>{seed_k0, seed_k1, seed_k2, seed_k3});
          global_job_refs.push_back(job_ref);
          global_nonces.push_back(nonce);
        }
      }

      running_search_threads.fetch_sub(1);
    }).detach();
  } catch (const sycl::exception& e) {
    printf("Error in solution search worker: %s\n", e.what());
  }
}

int c29(const unsigned job_ref, const unsigned c29_proof_size,
        const uint8_t* const input, const unsigned input_size,
        uint8_t* const output, uint32_t* const output_edges,
        uint64_t* const pnonce, const std::string& dev_str) {
  try {
    const sycl::async_handler exception_handler = [] (sycl::exception_list exceptions) {
      for (std::exception_ptr const& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch(sycl::exception const& e) {
          printf("Caught asynchronous SYCL exception:\n%s\n", e.what());
          throw;
        }
      }
    };

    static sycl::device compute_device = get_dev(dev_str);

    // SYCL_PROGRAM_COMPILE_OPTIONS is process-global; pearl's ESIMD (VC-backend) image rejects "-O3"
    // ("invalid api option"), which is a no-op for c29 anyway, so leave it empty to stay compatible.
    [[maybe_unused]] static const bool sycl_compile_env_set =
      (set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", ""), true);

    static auto compute_queue = c29_profile_enabled()
      ? sycl::queue{compute_device, exception_handler,
                    sycl::property_list{sycl::property::queue::in_order{},
                                        sycl::property::queue::enable_profiling{}}}
      : sycl::queue{compute_device, exception_handler,
                    sycl::property_list{sycl::property::queue::in_order{}}};
    static auto kernel_bundle = MOM_GET_EXEC_BUNDLE(compute_queue.get_context());

    // Pop the oldest queued solution that belongs to the current job (older jobs are discarded).
    bool has_solution = false;
    std::vector<sycl::uint2> solution;
    std::vector<uint64_t> solution_seed;
    uint64_t solution_nonce = 0;

    { C29SolutionLock lock(global_solutions_mutex);
      while (!global_solutions.empty()) {
        solution            = global_solutions.front();
        solution_seed       = global_seeds.front();
        const unsigned ref  = global_job_refs.front();
        solution_nonce      = global_nonces.front();
        global_solutions.pop_front();
        global_seeds.pop_front();
        global_job_refs.pop_front();
        global_nonces.pop_front();

        if (ref == job_ref) {
          has_solution = true;
          break;
        }
      }
    }

    if (has_solution) {
      // Convert cycle edges to 64-bit format for recovery kernel
      std::vector<uint64_t> recovery_edges;
      recovery_edges.reserve(c29_proof_size);
      for (const auto& edge : solution) recovery_edges.push_back(static_cast<uint64_t>(edge.y()) << 32 | edge.x());

      const uint64_t k0 = solution_seed[0], k1 = solution_seed[1], k2 = solution_seed[2], k3 = solution_seed[3];

      sycl::buffer<uint64_t, 1> buffer_edges{recovery_edges.data(), sycl::range<1>{c29_proof_size}};
      sycl::buffer<uint32_t, 1> buffer_nonces{sycl::range<1>{c29_proof_size}};

      // FluffyRecovery kernel - find nonces that generate solution edges
      compute_queue.submit([&](sycl::handler& handler) {
        sycl::accessor acc_edges{buffer_edges, handler, sycl::read_only};
        sycl::accessor acc_nonces{buffer_nonces, handler, sycl::write_only, sycl::no_init};
        sycl::local_accessor<uint32_t, 1> local_nonces{sycl::range<1>{c29_proof_size}, handler};
        MOM_USE_BUNDLE(handler, kernel_bundle);
        handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(2048 * 256), sycl::range<1>(256)),
                            [=](sycl::nd_item<1> item) {
          const uint32_t gid = item.get_global_id(0);
          const uint32_t lid = item.get_local_id(0);

          if (lid < c29_proof_size) local_nonces[lid] = 0;

          item.barrier(sycl::access::fence_space::local_space);

          SIPHASH_ROUND_LAMBDA_MACRO

          for (uint32_t block = 0; block < 1024; block += EDGE_BLOCK_SIZE) {
            const uint64_t base_nonce = gid * 1024 + block;
            uint64_t v0 = k0, v1 = k1, v2 = k2, v3 = k3;
            uint64_t sip_block[EDGE_BLOCK_SIZE];

            SIPHASH_FILL_BLOCK(v0, v1, v2, v3, base_nonce, sip_block, b);
            const uint64_t last = sip_block[EDGE_BLOCK_MASK];

            // Check each generated edge against target edges
            for (int32_t s = EDGE_BLOCK_MASK; s >= 0; s--) {
              const uint64_t lookup = s == EDGE_BLOCK_MASK ? last : sip_block[s] ^ last;
              const uint64_t u      = lookup & EDGE_MASK;
              const uint64_t v      = (lookup >> 32) & EDGE_MASK;
              const uint64_t edge_a = u | (v << 32);
              const uint64_t edge_b = v | (u << 32);

              // Match against solution edges (both orientations)
              for (uint32_t idx = 0; idx < c29_proof_size; idx++) {
                if (acc_edges[idx] == edge_a || acc_edges[idx] == edge_b)
                  local_nonces[idx] = base_nonce + s;
              }
            }
          }

          item.barrier(sycl::access::fence_space::local_space);

          if (lid < c29_proof_size && local_nonces[lid] > 0) acc_nonces[lid] = local_nonces[lid];
        });
      });

      std::vector<uint32_t> nonces(c29_proof_size);

      { sycl::host_accessor acc{buffer_nonces, sycl::read_only};
        std::memcpy(nonces.data(), acc.get_pointer(), c29_proof_size * sizeof(uint32_t));
      }

      // Cuckaroo29 requires the recovered edge nonces in ascending order.
      std::sort(nonces.begin(), nonces.end());

      // Pack the sorted nonces little-endian, EDGE_BITS bits each, into the proof bitstream.
      const uint32_t packed_len = (c29_proof_size * EDGE_BITS + 7) / 8;
      std::vector<uint8_t> packed(packed_len, 0);
      uint32_t bit_pos = 0;

      for (const uint32_t nonce : nonces) {
        for (uint32_t bit = 0; bit < EDGE_BITS; bit++, bit_pos++)
          if (nonce & (1u << bit)) packed[bit_pos / 8] |= (1u << (bit_pos % 8));
      }

      rx_blake2b(output, 32, packed.data(), packed_len);

      // Reverse the 32-byte hash in place to match the pool's expected byte order.
      for (int i = 0; i < 16; i++) std::swap(output[i], output[31 - i]);

      *pnonce = solution_nonce;
      std::memcpy(output_edges, nonces.data(), c29_proof_size * sizeof(uint32_t));

      return 1; // one solution produced
    } else if (input_size) { // start a new asynchronous solution search
      union { uint8_t blake_output[32]; uint64_t k[4]; };
      rx_blake2b(blake_output, 32, input, input_size);
      start_new_c29_solution_search(k[0], k[1], k[2], k[3], job_ref, *pnonce, c29_proof_size, compute_queue);

      return 0; // no immediate results
    } else { // poll: -1 once every worker has drained, otherwise 0 (still searching)
      return running_search_threads.load() == 0 ? -1 : 0;
    }
  } catch (const sycl::exception& e) {
    printf("Caught synchronous SYCL exception:\n%s\n", e.what());
    throw;
  }
}
