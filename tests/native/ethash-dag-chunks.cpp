// Chunked Ethash DAG generation must retire each dispatch before recording the next, so desktop and
// WDDM drivers never hold one watchdog-length batch; the KawPow/Etchash chunk policies must honor an
// explicit override before their device defaults. Production sources are included unchanged.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sycl {
template <int D> struct range {
  explicit range(const size_t size) : size(size) {}
  size_t size;
};
template <int D> struct nd_range {
  nd_range(const range<D> global, const range<D> local) : global(global), local(local) {}
  range<D> global, local;
};
template <int D> struct nd_item {
  size_t id;
  size_t get_global_id(int) const { return id; }
};
struct event {};
struct device {
  bool integrated = false, cuda = false, hip = false;
  bool is_cpu() const { return false; }
};
struct handler {
  template <typename Kernel> void parallel_for(const nd_range<1> range, Kernel kernel) {
    assert(range.global.size % range.local.size == 0);
    for (size_t id = 0; id < range.global.size; ++id)
      kernel(nd_item<1>{id});
  }
};
} // namespace sycl

// Command trace: S = submitted dispatch, W = retired dispatch.
static std::string trace;
static unsigned fail_wait = 0;
struct Queue {
  sycl::event memcpy(void* destination, const void* source, size_t bytes) {
    std::memcpy(destination, source, bytes);
    return {};
  }
  template <typename Submit> sycl::event submit(Submit submit) {
    trace += 'S';
    sycl::handler handler;
    submit(handler);
    return {};
  }
};
namespace sycl {
using queue = Queue;
}

// Upload lifetime is exercised separately with the production guard and fault injection.
struct MomSyclHostTransferGuard {
  MomSyclHostTransferGuard(Queue&, const char*) {}
};

#define MOM_SYCL_KERNEL_ARGS_RESTRICT
struct FastModData {
  uint32_t divisor;
};
static FastModData make_fast_mod_data(const uint32_t divisor) { return {divisor}; }
static uint32_t fast_mod_dev(const uint32_t value, const FastModData data) {
  return value % data.divisor;
}
#include "round-up-size.inc"
static void sycl_wait_and_throw(sycl::event, const sycl::device&) {
  trace += 'W';
  if (fail_wait && std::count(trace.begin(), trace.end(), 'W') == fail_wait)
    throw std::runtime_error("submitted DAG fault");
}
static unsigned sycl_default_workgroup(const sycl::device&, std::initializer_list<unsigned>,
                                       const unsigned preferred) {
  return preferred;
}
static bool sycl_workgroup_fits(const sycl::device&, const unsigned size) { return size <= 256; }
#include "unsigned-env.inc"
struct ethash_h256_t {
  uint8_t bytes[32];
};
static ethash_h256_t ethash_get_seedhash(uint32_t) { return {}; }
static bool ethash_compute_cache_nodes(void*, uint64_t, const ethash_h256_t*) { return true; }

namespace fixture {
// Deterministic stand-ins: the property is dispatch order and node coverage, not hash values.
static uint32_t fnv_dev(const uint32_t a, const uint32_t b) { return (a * 0x01000193u) ^ b; }
static void keccak_512_words_dev(uint32_t* const node) {
  for (unsigned word = 0; word < 16; ++word)
    node[word] = node[word] * 0x9e3779b1u + word + 1;
}
#include "sycl/ethash-dag.inc"
} // namespace fixture

inline bool is_integrated_gpu(const sycl::device& device) { return device.integrated; }
inline bool mom_is_cuda(const sycl::device& device) { return device.cuda; }
inline bool mom_is_hip(const sycl::device& device) { return device.hip; }
#if defined(MOM_FIXTURE_WINDOWS)
#define _WIN32 1
#endif
struct KawPowPolicy {
#include "kawpow_dag_chunk.inc"
};
struct EtchashPolicy {
#include "etchash_dag_chunk.inc"
};

static std::vector<uint32_t> generate(const uint32_t total, const uint32_t chunk) {
  std::vector<uint32_t> light(7 * fixture::ETHASH_DAG_NODE_WORDS);
  for (size_t i = 0; i < light.size(); ++i)
    light[i] = static_cast<uint32_t>(i * 2654435761u);
  std::vector<uint32_t> dag(total * fixture::ETHASH_DAG_NODE_WORDS, 0xa5a5a5a5u);
  Queue queue;
  trace.clear();
  fixture::generate_ethash_dag<8>(queue, sycl::device{}, light.data(), 7, dag.data(), total, 4,
                                  chunk);
  return dag;
}

static std::vector<uint32_t> generate_slice(const uint32_t total, const uint32_t start,
                                            const uint32_t count) {
  std::vector<uint32_t> light(7 * fixture::ETHASH_DAG_NODE_WORDS);
  for (size_t i = 0; i < light.size(); ++i)
    light[i] = static_cast<uint32_t>(i * 2654435761u);
  std::vector<uint32_t> dag(total * fixture::ETHASH_DAG_NODE_WORDS, 0xa5a5a5a5u);
  Queue queue;
  trace.clear();
  const sycl::event event = fixture::submit_ethash_dag_chunk<8, size_t>(
      queue, light.data(), make_fast_mod_data(7), dag.data(), start, count, 4);
  sycl_wait_and_throw(event, sycl::device{});
  return dag;
}

static void set(const char* const name, const char* const value) {
  if (value)
    setenv(name, value, 1);
  else
    unsetenv(name);
}

int main() {
  const size_t maximum_range = std::numeric_limits<size_t>::max();
  assert(mom_round_up_size(maximum_range, maximum_range) == maximum_range);
  assert(mom_round_up_size(maximum_range - 1, 2) == maximum_range - 1);
  assert(mom_round_up_size(maximum_range - 2, 2) == maximum_range - 1);
  for (uint64_t step : {0, 2}) {
    try {
      mom_round_up_size(maximum_range, step);
      assert(false);
    } catch (const std::string& error) {
      assert(error == "SYCL range is too large");
    }
  }

  const std::vector<uint32_t> whole = generate(10, 0);
  assert(trace == "SW");
  assert(generate(10, 3) == whole && trace == "SWSWSWSW");
  assert(generate(10, 10) == whole && trace == "SW");
  assert(generate(10, 64) == whole && trace == "SW");
  assert(generate(0, 0).empty() && trace.empty());
  assert(generate(0, 3).empty() && trace.empty());

  // Octopus uses the same node transform but needs size_t output offsets for its largest DAGs.
  const std::vector<uint32_t> slice = generate_slice(10, 3, 4);
  assert(trace == "SW");
  for (uint32_t node = 0; node < 10; ++node) {
    for (unsigned word = 0; word < fixture::ETHASH_DAG_NODE_WORDS; ++word) {
      const size_t index = static_cast<size_t>(node) * fixture::ETHASH_DAG_NODE_WORDS + word;
      assert(slice[index] == ((node >= 3 && node < 7) ? whole[index] : 0xa5a5a5a5u));
    }
  }

  // A submitted chunk fault propagates and no later chunk is recorded.
  fail_wait = 2;
  try {
    generate(10, 3);
    assert(false);
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()) == "submitted DAG fault");
  }
  assert(trace == "SWSW");
  fail_wait = 0;

  const sycl::device discrete{}, integrated{true, false, false};
  const sycl::device cuda{false, true, false}, hip{false, false, true};
#if defined(MOM_FIXTURE_WINDOWS)
  constexpr uint32_t wddm = 1u << 18;
#else
  constexpr uint32_t wddm = 0;
#endif
  for (const char* const name : {"MOM_KAWPOW_DAG_CHUNK_NODES", "MOM_ETCHASH_DAG_CHUNK_NODES"})
    set(name, nullptr);
  assert(KawPowPolicy::kawpow_dag_chunk_nodes(discrete) == 0);
  assert(KawPowPolicy::kawpow_dag_chunk_nodes(integrated) == 1u << 12);
  assert(KawPowPolicy::kawpow_dag_chunk_nodes(cuda) == wddm);
  assert(KawPowPolicy::kawpow_dag_chunk_nodes(hip) == wddm);
  assert(EtchashPolicy::etchash_dag_chunk_nodes(discrete) == 0);
  assert(EtchashPolicy::etchash_dag_chunk_nodes(integrated) == 1u << 12);
  assert(EtchashPolicy::etchash_dag_chunk_nodes(cuda) == 0);

  // Explicit overrides win over every device default, including a single full dispatch.
  set("MOM_KAWPOW_DAG_CHUNK_NODES", "0");
  set("MOM_ETCHASH_DAG_CHUNK_NODES", "0");
  assert(KawPowPolicy::kawpow_dag_chunk_nodes(integrated) == 0);
  assert(EtchashPolicy::etchash_dag_chunk_nodes(integrated) == 0);
  set("MOM_KAWPOW_DAG_CHUNK_NODES", "77");
  set("MOM_ETCHASH_DAG_CHUNK_NODES", "77");
  assert(KawPowPolicy::kawpow_dag_chunk_nodes(cuda) == 77);
  assert(EtchashPolicy::etchash_dag_chunk_nodes(discrete) == 77);

  // Malformed or 32-bit-overflowing overrides keep the device default.
  for (const char* const value : {"4294967296", "12x", ""}) {
    set("MOM_KAWPOW_DAG_CHUNK_NODES", value);
    set("MOM_ETCHASH_DAG_CHUNK_NODES", value);
    assert(KawPowPolicy::kawpow_dag_chunk_nodes(integrated) == 1u << 12);
    assert(EtchashPolicy::etchash_dag_chunk_nodes(integrated) == 1u << 12);
  }
  std::puts("PASS Ethash DAG chunk retirement and override policy");
}
