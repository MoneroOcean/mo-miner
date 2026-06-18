// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

// SYCL KawPow implementation based on XMRig's KawPow reference and OpenCL
// runner structure.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "lib-internal.h"
#include "../native/consts.h"
#include "../xmrig/3rdparty/libethash/ethash.h"
#include "../xmrig/3rdparty/libethash/data_sizes.h"

namespace mom_kawpow {

constexpr uint32_t KAWPOW_EPOCH_LENGTH  = 7500;
constexpr uint32_t KAWPOW_PERIOD_LENGTH = 3;
// FiroPow: DAG epoch every 1300 blocks, ProgPoW program changes every block (period == height).
constexpr uint32_t FIROPOW_EPOCH_LENGTH  = 1300;
constexpr uint32_t FIROPOW_PERIOD_LENGTH = 1;
// EvrProgPow: same seal structure as KawPoW (different magic), DAG epoch every 12000 blocks.
constexpr uint32_t EVRPROGPOW_EPOCH_LENGTH  = 12000;
constexpr uint32_t EVRPROGPOW_PERIOD_LENGTH = 3;
constexpr uint32_t KAWPOW_LANES         = 16;
constexpr uint32_t KAWPOW_REGS          = 32;
constexpr uint32_t KAWPOW_DAG_LOADS     = 4;
constexpr uint32_t KAWPOW_CACHE_WORDS   = 4096;
constexpr uint32_t KAWPOW_CNT_DAG       = 64;
constexpr uint32_t KAWPOW_CNT_CACHE     = 11;
constexpr uint32_t KAWPOW_CNT_MATH      = 18;
constexpr uint32_t KAWPOW_DATA_LOADS    = 256 / (sizeof(uint32_t) * KAWPOW_LANES);
constexpr uint32_t KAWPOW_DATASET_PARENTS = 512;
constexpr uint32_t NODE_WORDS           = 16;
constexpr uint32_t FNV_PRIME            = 0x01000193U;
constexpr uint32_t FNV_OFFSET_BASIS     = 0x811c9dc5U;
constexpr uint32_t MAX_KAWPOW_OUTPUTS   = 15;

constexpr uint32_t RAVENCOIN_KAWPOW[15] = {
  0x00000072, 0x00000041, 0x00000056, 0x00000045, 0x0000004E,
  0x00000043, 0x0000004F, 0x00000049, 0x0000004E, 0x0000004B,
  0x00000041, 0x00000057, 0x00000050, 0x0000004F, 0x00000057
};

// EvrProgPow seal magic: ASCII "EVRMORE-PROGPOW" (hyphen 0x2D at index 7). Same 15-word placement
// (st[10..24] initial / st[16..24] final) and seal structure as KawPoW -- only the data differs.
constexpr uint32_t EVRMORE_EVRPROGPOW[15] = {
  0x00000045, 0x00000056, 0x00000052, 0x0000004D, 0x0000004F,  // E V R M O
  0x00000052, 0x00000045, 0x0000002D, 0x00000050, 0x00000052,  // R E - P R
  0x0000004F, 0x00000047, 0x00000050, 0x0000004F, 0x00000057   // O G P O W
};

// FiroPow uses the chfast/ethash (EIP-1057) dataset sizing, NOT the classic Ethereum sizing baked
// into data_sizes.h (which KawPoW/Ravencoin uses). The light CACHE sizing is identical between the
// two (so cache_sizes[epoch] + ethash_get_seedhash + ethash_compute_cache_nodes are reused), but the
// full DATASET is larger: init 1.5 GiB (vs 1 GiB), growth 8 MiB/epoch, num_items = largest prime
// <= init/128 + epoch*growth/128, then *128 bytes. The DAG-gen kernel and search are otherwise
// identical -- only the node count (and thus the search modulus) changes.
constexpr int FIRO_FULL_DATASET_INIT_ITEMS = ((1 << 30) + (1 << 29)) / 128;  // 12582912 (1.5 GiB)
constexpr int FIRO_FULL_DATASET_GROWTH_ITEMS = (1 << 23) / 128;              // 65536 per epoch
// EvrProgPow uses the SAME chfast/EIP-1057 prime sizing but a 3 GiB init (confirmed against the
// EvrmoreOrg/cpp-evrprogpow reference: full_dataset_init=(1<<30)*3, growth=1<<23, num_items=
// largest_prime(init/128 + epoch*growth/128)*128; epoch 0 = 25165813 items = 3.0000 GiB).
constexpr int EVR_FULL_DATASET_INIT_ITEMS = ((1u << 30) * 3) / 128;          // 25165824 (3 GiB)
constexpr int EVR_FULL_DATASET_GROWTH_ITEMS = (1 << 23) / 128;               // 65536 per epoch

static bool firo_is_odd_prime(const int number) {
  for (int64_t d = 3; d * d <= static_cast<int64_t>(number); d += 2)
    if (number % d == 0) return false;
  return true;
}

static int firo_find_largest_prime(int n) {
  if (n < 2) return 0;
  if (n == 2) return 2;
  if (n % 2 == 0) --n;
  while (!firo_is_odd_prime(n)) n -= 2;
  return n;
}

// chfast/EIP-1057 full-dataset byte size for an epoch: largest_prime(init + epoch*growth) * 128.
// Shared by FiroPow (1.5 GiB init) and EvrProgPow (3 GiB init); a multiple of 128 (so also of 64).
static uint64_t chfast_dag_bytes(const uint32_t epoch, const int init_items, const int growth_items) {
  const int upper = init_items + static_cast<int>(epoch) * growth_items;
  return static_cast<uint64_t>(firo_find_largest_prime(upper)) * 128ull;
}

// Used by MOM_LOOP_STATS to break a dispatch call into host-side phases.
static uint64_t kawpow_now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()
  ).count();
}

static bool kawpow_loop_stats() {
  static const bool enabled = std::getenv("MOM_LOOP_STATS") != nullptr;
  return enabled;
}

static void format_duration_ms(char* out, size_t out_size, uint64_t ms) {
  if (ms < 1000) {
    std::snprintf(out, out_size, "%" PRIu64 "ms", ms);
  } else if (ms < 60000) {
    std::snprintf(out, out_size, "%.2f s", static_cast<double>(ms) / 1000.0);
  } else {
    std::snprintf(out, out_size, "%.2f min", static_cast<double>(ms) / 60000.0);
  }
}

struct KawpowCacheOp {
  uint32_t src;
  uint32_t dst;
  uint32_t merge_mode;
  uint32_t merge_shift;
};

struct KawpowMathOp {
  uint32_t src1;
  uint32_t src2;
  uint32_t dst;
  uint32_t math_selector;
  uint32_t merge_mode;
  uint32_t merge_shift;
};

struct KawpowDataOp {
  uint32_t dst;
  uint32_t merge_mode;
  uint32_t merge_shift;
};

struct KawpowProgram {
  KawpowCacheOp cache[KAWPOW_CNT_CACHE];
  KawpowMathOp math[KAWPOW_CNT_MATH];
  KawpowDataOp data[KAWPOW_DATA_LOADS];
};

// Per-period spec constants for the Intel spec-const path (the CUDA build folds the program via the
// runtime source JIT instead, but these stay declared for the shared SLM/CPU code below).
constexpr sycl::specialization_id<KawpowProgram> kawpow_program_id;
constexpr sycl::specialization_id<FastModData> kawpow_dag_mod_id;
constexpr sycl::specialization_id<bool> kawpow_cpu_offset_barrier_id;

struct KawpowResult {
  uint32_t count;
  uint64_t nonce[MAX_KAWPOW_OUTPUTS];
  uint32_t output[MAX_KAWPOW_OUTPUTS][8];
  uint32_t mix_hash[MAX_KAWPOW_OUTPUTS][8];
};

struct Uint2 {
  uint32_t x;
  uint32_t y;
};

// alignas(16): the per-loop dag read (kawpow_device.inc) is the hottest global op (64/nonce); 16-byte
// alignment lets it lower to one 128-bit vector load (ld.global.v4.u32 / block_load) instead of 4
// scalar loads. The DAG base is 256B-aligned USM and offsets step whole DagLoads, so it is already
// 16B-aligned at runtime -- only the declared alignment was blocking the wide load. Size stays 16
// bytes, so this stays byte-compatible with the kawpow_jit.inc mirror (which gets the same alignas).
struct alignas(16) DagLoad {
  uint32_t s[KAWPOW_DAG_LOADS];
};

struct Kiss99 {
  uint32_t z;
  uint32_t w;
  uint32_t jsr;
  uint32_t jcong;
};

#include "kawpow_device.inc"
#include "kawpow_jit.inc"

// Per-algo ProgPoW variant: epoch/period divisors (host-side DAG/program selection) plus the keccak
// seal mode + magic words baked into the kernel. KawPoW and EvrProgPow share SealMode::KAWPOW (magic
// differs); FiroPow uses SealMode::FIRO (no magic). SealMode is defined in kawpow_device.inc.
struct KawpowVariant {
  uint32_t epoch_length;
  uint32_t period_length;
  SealMode seal;
  const uint32_t* magic;  // 15 words; ignored for SealMode::FIRO
  // Dataset sizing: chfast_init_items == 0 means classic Ethereum sizing (data_sizes.h table, KawPoW);
  // otherwise chfast/EIP-1057 sizing largest_prime(init + epoch*growth)*128 -- FiroPow 1.5 GiB init,
  // EvrProgPow 3 GiB init. Light cache sizing is identical across all three variants.
  int chfast_init_items;
  int chfast_growth_items;
};

constexpr KawpowVariant KAWPOW_VARIANT{
  KAWPOW_EPOCH_LENGTH, KAWPOW_PERIOD_LENGTH, SealMode::KAWPOW, RAVENCOIN_KAWPOW, 0, 0};
constexpr KawpowVariant FIROPOW_VARIANT{
  FIROPOW_EPOCH_LENGTH, FIROPOW_PERIOD_LENGTH, SealMode::FIRO, RAVENCOIN_KAWPOW,
  FIRO_FULL_DATASET_INIT_ITEMS, FIRO_FULL_DATASET_GROWTH_ITEMS};
constexpr KawpowVariant EVRPROGPOW_VARIANT{
  EVRPROGPOW_EPOCH_LENGTH, EVRPROGPOW_PERIOD_LENGTH, SealMode::KAWPOW, EVRMORE_EVRPROGPOW,
  EVR_FULL_DATASET_INIT_ITEMS, EVR_FULL_DATASET_GROWTH_ITEMS};

KawpowProgram make_program(const uint64_t period) {
  KawpowProgram program{};
  uint32_t dst_seq[KAWPOW_REGS];
  uint32_t src_seq[KAWPOW_REGS];

  for (uint32_t i = 0; i < KAWPOW_REGS; ++i) {
    dst_seq[i] = i;
    src_seq[i] = i;
  }

  uint32_t fnv_hash = FNV_OFFSET_BASIS;
  Kiss99 rnd{
    fnv1a(fnv_hash, static_cast<uint32_t>(period)),
    fnv1a(fnv_hash, static_cast<uint32_t>(period >> 32)),
    fnv1a(fnv_hash, static_cast<uint32_t>(period)),
    fnv1a(fnv_hash, static_cast<uint32_t>(period >> 32))
  };

  for (uint32_t i = KAWPOW_REGS; i > 1; --i) {
    std::swap(dst_seq[i - 1], dst_seq[kiss99(rnd) % i]);
    std::swap(src_seq[i - 1], src_seq[kiss99(rnd) % i]);
  }

  uint32_t dst_counter = 0;
  uint32_t src_counter = 0;
  uint32_t cache_counter = 0;
  uint32_t math_counter = 0;

  for (uint32_t i = 0; i < KAWPOW_CNT_CACHE || i < KAWPOW_CNT_MATH; ++i) {
    if (i < KAWPOW_CNT_CACHE) {
      // Order matters: the src/dst sequence reads must precede the kiss99() selector draw to match
      // the reference ProgPoW program (the test vector guards this).
      const uint32_t src = src_seq[(src_counter++) % KAWPOW_REGS];
      const uint32_t dst = dst_seq[(dst_counter++) % KAWPOW_REGS];
      const uint32_t selector = kiss99(rnd);
      program.cache[cache_counter++] = {src, dst, selector % 4, merge_shift(selector)};
    }

    if (i < KAWPOW_CNT_MATH) {
      const uint32_t src_rnd = kiss99(rnd) % (KAWPOW_REGS * (KAWPOW_REGS - 1));
      const uint32_t src1 = src_rnd % KAWPOW_REGS;
      uint32_t src2 = src_rnd / KAWPOW_REGS;
      if (src2 >= src1) ++src2;

      const uint32_t math_selector = kiss99(rnd);
      const uint32_t merge_selector = kiss99(rnd);
      program.math[math_counter++] = {
        src1,
        src2,
        dst_seq[(dst_counter++) % KAWPOW_REGS],
        math_selector % 11,
        merge_selector % 4,
        merge_shift(merge_selector)
      };
    }
  }

  for (uint32_t i = 0; i < KAWPOW_DATA_LOADS; ++i) {
    // The first data load always targets register 0; the rest pull from the shuffled dst sequence.
    const uint32_t dst = i == 0 ? 0 : dst_seq[(dst_counter++) % KAWPOW_REGS];
    const uint32_t selector = kiss99(rnd);
    program.data[i] = {dst, selector % 4, merge_shift(selector)};
  }

  return program;
}

void compute_light_cache(std::vector<uint32_t>& cache, const uint32_t epoch) {
  const uint64_t cache_bytes = cache_sizes[epoch];
  cache.assign(cache_bytes / sizeof(uint32_t), 0);

  const ethash_h256_t seed = ethash_get_seedhash(epoch);
  if (!ethash_compute_cache_nodes(cache.data(), cache_bytes, &seed)) {
    throw std::string("Can't calculate kawpow light cache");
  }
}

// Named so the per-period specialized bundle can be built ahead of time.
class KawpowSgKernel;

class KawpowState {
public:
  sycl::device device;
  sycl::queue queue;
  std::unique_ptr<MOM_BUNDLE_T> bundle;
  bool shared_io;
  bool shared_dag;
  uint8_t* input = nullptr;
  uint32_t* light_cache = nullptr;
  uint32_t* dag = nullptr;
  KawpowResult* result = nullptr;
  uint32_t input_cap = 0;
  uint64_t light_cache_words = 0;
  uint64_t dag_words = 0;
  uint32_t epoch = UINT32_MAX;
  // Tracks the DAG byte size the current DAG was built with. KawPoW and FiroPow use different dataset
  // sizing (classic vs chfast), so the same epoch can map to different DAG sizes; the DAG is rebuilt
  // when this changes, not just when the epoch changes.
  uint64_t dag_bytes = 0;
  uint64_t period = UINT64_MAX;
  KawpowProgram program{};
  unsigned workgroup;
  std::mutex mutex;

  // Active ProgPoW seal (set per-call from the variant, under `mutex`). On CUDA it is baked into the
  // JIT source (kawpow_emit_seal); on Intel/spir64 it rides in via SealParams. A device's KawpowState
  // is shared across the kawpow/firopow/evrprogpow entrypoints, so the seal can change between calls
  // and participates in the bundle cache key below (a seal change forces a rebuild).
  SealMode seal = SealMode::KAWPOW;
  uint32_t magic[15] = {};

  // Per-period executable bundles (Intel: program/dag spec constants baked in; CUDA: source-JIT'd
  // kernel). The next period's bundle is built on a worker thread while the current one mines, so
  // the period switch no longer stalls hashing on the build.
  std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> period_bundle;
  uint64_t period_bundle_period = UINT64_MAX;
  uint32_t period_bundle_epoch = UINT32_MAX;
  SealMode period_bundle_seal = SealMode::KAWPOW;
  uint32_t period_bundle_magic[15] = {};
  std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> next_bundle;
  uint64_t next_bundle_period = UINT64_MAX;
  uint32_t next_bundle_epoch = UINT32_MAX;
  SealMode next_bundle_seal = SealMode::KAWPOW;
  uint32_t next_bundle_magic[15] = {};
  std::thread prefetch_thread;

  // CUDA kawpow path (decided once, then stable for this device's lifetime): the source-JIT folds
  // the per-period ProgPoW program for full speed, but needs a working runtime kernel_compiler
  // (libsycl-jit + a CUDA target). If the first build's JIT throws (e.g. an old driver or a host
  // missing libdevice), fall back permanently to the AOT spec-constant kernel -- correct, just
  // unfolded (~3x slower). -1 = undecided, 1 = JIT, 0 = spec-const. MOM_KAWPOW_JIT=0/1 forces a path.
  // Atomic because the prefetch thread reads/writes it while the launch path reads it.
  std::atomic<int> cuda_use_jit{-1};

  explicit KawpowState(const std::string& dev_str)
    : device(get_dev(dev_str)),
      queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !device.has(sycl::aspect::usm_device_allocations)),
      shared_dag(device.is_cpu() || !device.has(sycl::aspect::usm_device_allocations)),
      workgroup(kawpow_workgroup(device))
  {
    if (!device.has(sycl::aspect::usm_shared_allocations) ||
        (!device.is_cpu() && !device.has(sycl::aspect::usm_device_allocations))) {
      throw std::string("kawpow SYCL device does not support required allocations");
    }

    set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", kawpow_compile_options());
    bundle = std::make_unique<MOM_BUNDLE_T>(
      MOM_GET_EXEC_BUNDLE(queue.get_context())
    );
  }

  ~KawpowState() {
    if (prefetch_thread.joinable()) prefetch_thread.join();
    release();
  }

  // Parse env var `name` as base-10 unsigned; returns false (leaving `out` untouched) on unset,
  // empty, malformed, or out-of-uint32-range input so callers fall back to their default.
  static bool parse_env_u32(const char* name, uint32_t& out) {
    const char* const value = std::getenv(name);
    if (!value || !*value) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno || end == value || *end || parsed > std::numeric_limits<uint32_t>::max()) return false;
    out = static_cast<uint32_t>(parsed);
    return true;
  }

  // Pick a work-group size: honor `name` if set to one of `allowed`, else `sycl_default_workgroup`.
  static unsigned select_workgroup(
    const char* name, const sycl::device& dev,
    const std::initializer_list<unsigned> allowed, const unsigned preferred
  ) {
    const unsigned fallback = sycl_default_workgroup(dev, allowed, preferred);
    uint32_t parsed;
    if (!parse_env_u32(name, parsed)) return fallback;
    for (const unsigned candidate : allowed) if (candidate == parsed) return parsed;
    return fallback;
  }

  static unsigned kawpow_workgroup(const sycl::device& dev) {
    // GPU default is a 256-thread block: the heavy search kernel (80 regs on the CUDA source-baked
    // build) wants more resident warps to hide DAG-read latency (CUDA sweep: 256=13.2 > 128 > 64).
    return select_workgroup("MOM_KAWPOW_WORKGROUP", dev, {64, 128, 256, 512}, dev.is_cpu() ? 128 : 256);
  }

  static unsigned kawpow_dag_workgroup(const sycl::device& dev) {
    return select_workgroup("MOM_KAWPOW_DAG_WORKGROUP", dev, {32, 64, 128, 256, 512}, dev.is_cpu() ? 128 : 64);
  }

  static uint32_t kawpow_dag_chunk_nodes() {
    uint32_t parsed = 0;  // 0 means "single full-DAG dispatch" (chunking disabled)
    parse_env_u32("MOM_KAWPOW_DAG_CHUNK_NODES", parsed);
    return parsed;
  }

  static const char* kawpow_compile_options() {
    // "-O3" is process-global; pearl's ESIMD image rejects it (and it's a no-op for kawpow), so
    // default to none. Override via MOM_KAWPOW_COMPILE_OPTIONS if needed.
    const char* const value = std::getenv("MOM_KAWPOW_COMPILE_OPTIONS");
    return value ? value : "";
  }

  template<typename T>
  T* allocate(const size_t count, const bool shared) {
    return shared ? sycl::malloc_shared<T>(count, queue) : sycl::malloc_device<T>(count, queue);
  }

  void free_ptr(auto*& ptr) {
    if (ptr) sycl::free(ptr, queue);
    ptr = nullptr;
  }

  void release() {
    queue.wait_and_throw();
    free_ptr(input);
    free_ptr(light_cache);
    free_ptr(dag);
    free_ptr(result);
    input_cap = 0;
    light_cache_words = 0;
    dag_words = 0;
    epoch = UINT32_MAX;
    dag_bytes = 0;
    period = UINT64_MAX;
  }

  void ensure_input(const unsigned input_size) {
    if (input_size <= input_cap && result) return;
    queue.wait_and_throw();
    free_ptr(input);
    free_ptr(result);
    input_cap = input_size;
    input = allocate<uint8_t>(input_cap, shared_io);
    result = sycl::malloc_shared<KawpowResult>(1, queue);
    if (!input || !result) throw std::string("Can't allocate kawpow SYCL input buffers");
  }

  void ensure_epoch(const uint32_t new_epoch, const uint64_t new_dag_bytes, const bool should_log) {
    if (epoch == new_epoch && dag_bytes == new_dag_bytes) return;
    if (new_epoch >= 2048 || !new_dag_bytes || !cache_sizes[new_epoch]) {
      throw std::string("Bad kawpow epoch");
    }

    const uint64_t new_light_cache_words = cache_sizes[new_epoch] / sizeof(uint32_t);
    const uint64_t new_dag_words = new_dag_bytes / sizeof(uint32_t);

    std::vector<uint32_t> host_cache;
    const uint64_t start_ms = now_ms();
    compute_light_cache(host_cache, new_epoch);

    queue.wait_and_throw();
    if (new_light_cache_words != light_cache_words) {
      free_ptr(light_cache);
      light_cache = allocate<uint32_t>(new_light_cache_words, shared_dag);
      light_cache_words = new_light_cache_words;
    }
    if (new_dag_words != dag_words) {
      free_ptr(dag);
      dag = allocate<uint32_t>(new_dag_words, shared_dag);
      dag_words = new_dag_words;
    }
    if (!light_cache || !dag) throw std::string("Can't allocate kawpow DAG buffers");

    if (shared_dag) std::memcpy(light_cache, host_cache.data(), host_cache.size() * sizeof(uint32_t));
    else sycl_wait_and_throw(queue.memcpy(light_cache, host_cache.data(), host_cache.size() * sizeof(uint32_t)), device);

    const uint32_t light_nodes = static_cast<uint32_t>(new_light_cache_words / NODE_WORDS);
    const FastModData light_mod = make_fast_mod_data(light_nodes);

    const uint32_t dag_workgroup = kawpow_dag_workgroup(device);
    const uint32_t total = static_cast<uint32_t>(new_dag_words / NODE_WORDS);  // DAG node count
    const uint32_t chunk_nodes = kawpow_dag_chunk_nodes();
    uint32_t* const d_light = light_cache;
    uint32_t* const d_dag = dag;
    sycl::queue& q = queue;
    auto& kb = *bundle;
    (void)kb;

    sycl::event dag_event;
    for (uint32_t start_node = 0; start_node < total;) {
      const uint32_t current_nodes = chunk_nodes ? std::min(chunk_nodes, total - start_node) : total;
      const uint32_t chunk_start = start_node;
      dag_event = q.submit([&](sycl::handler& h) {
        MOM_USE_BUNDLE(h, kb);
        h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(round_up(current_nodes, dag_workgroup)), sycl::range<1>(dag_workgroup)),
          [=](sycl::nd_item<1> item) {
            const uint32_t local_node = item.get_global_id(0);
            if (local_node >= current_nodes) return;
            const uint32_t node_index = chunk_start + local_node;

            uint32_t dag_node[NODE_WORDS];
            const uint32_t init = fast_mod_dev(node_index, light_mod);
#pragma unroll
            for (unsigned w = 0; w < NODE_WORDS; ++w) dag_node[w] = d_light[init * NODE_WORDS + w];
            dag_node[0] ^= node_index;
            keccak_512_words_dev(dag_node);

            for (uint32_t i = 0; i < KAWPOW_DATASET_PARENTS; ++i) {
              const uint32_t parent_index = fast_mod_dev(fnv_dev(node_index ^ i, dag_node[i & (NODE_WORDS - 1)]), light_mod);
              const uint32_t parent_base = parent_index * NODE_WORDS;
#pragma unroll
              for (unsigned w = 0; w < NODE_WORDS; ++w) {
                dag_node[w] = fnv_dev(dag_node[w], d_light[parent_base + w]);
              }
            }
            keccak_512_words_dev(dag_node);

            const uint32_t base = node_index * NODE_WORDS;
#pragma unroll
            for (unsigned w = 0; w < NODE_WORDS; ++w) d_dag[base + w] = dag_node[w];
          }
        );
      });
      if (!chunk_nodes) break;
      start_node += current_nodes;
    }
    sycl_wait_and_throw(dag_event, device);

    epoch = new_epoch;
    dag_bytes = new_dag_bytes;
    if (should_log) {
      char elapsed[32];
      format_duration_ms(elapsed, sizeof(elapsed), now_ms() - start_ms);
      std::fprintf(stderr, "KawPow DAG for epoch %u calculated (%s)\n", new_epoch, elapsed);
    }
  }

  void ensure_period(const uint64_t new_period) {
    if (period == new_period) return;
    program = make_program(new_period);
    period = new_period;
  }

  // build_seal / build_magic are passed by value (not read from the members) so the prefetch thread,
  // which calls this off the state mutex, never races kawpow_core writing state.seal/state.magic. Only
  // the CUDA JIT path bakes the seal; the Intel spec-const path gets it via SealParams at launch.
  std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>>
  build_period_bundle(const uint64_t new_period, const FastModData dag_mod,
                      const SealMode build_seal, const uint32_t build_magic[15]) {
#if defined(MOM_SYCL_HAS_CUDA)
    // CUDA: the source-JIT folds the per-period program to straight-line const ops (full speed); the
    // AOT spec-constant kernel below is correct but ~3x slower because spec constants do NOT fold on
    // the nvptx backend. Default to JIT; fall back to spec-const if the runtime kernel_compiler is
    // unavailable (see cuda_use_jit). MOM_KAWPOW_JIT=0/1 forces a path (1 = JIT, no fallback).
    if (mom_is_cuda(device)) {
      if (cuda_use_jit.load() < 0) {
        const char* const force = std::getenv("MOM_KAWPOW_JIT");
        if (force) cuda_use_jit.store(std::atoi(force) != 0 ? 1 : 0);
      }
      if (cuda_use_jit.load() != 0) {  // 1 (committed JIT) or -1 (undecided -> try JIT)
        try {
          // CUDA JIT: compile the search kernel from SYCL source with the period program baked in as
          // a const struct (folds the random-math interpreter). Device body is sycl/kawpow_device.inc.
          // The per-algo keccak seal (mode + magic) is also baked in (kawpow_emit_seal) so firopow /
          // evrprogpow compute the correct seal on CUDA; emitted after kawpow_device_src() (defines
          // SealMode) and before the WRAPPER (reads JIT_SEAL/JIT_MAGIC).
          namespace syclex = sycl::ext::oneapi::experimental;
          const std::string src = std::string(KAWPOW_JIT_PRELUDE) + kawpow_device_src() +
            kawpow_emit_baked(make_program(new_period), dag_mod) +
            kawpow_emit_seal(build_seal, build_magic) + KAWPOW_JIT_WRAPPER;
          auto kb_src = syclex::create_kernel_bundle_from_source(
            queue.get_context(), syclex::source_language::sycl, src);
          const char* const jit_opts = std::getenv("MOM_KAWPOW_JIT_OPTS");
          // Build for THIS device explicitly: without the device list the kernel_compiler targets a
          // generic nvptx default (sm_75 -> CUDA_ERROR_NO_BINARY on newer GPUs / needs sm_75 libdevice).
          // Passing the device makes it compile for the device's actual arch (e.g. sm_89 on an L4).
          const std::vector<sycl::device> jit_devs{device};
          auto exe = (jit_opts && *jit_opts)
            ? syclex::build(kb_src, jit_devs, syclex::properties{syclex::build_options{std::string(jit_opts)}})
            : syclex::build(kb_src, jit_devs);
          cuda_use_jit.store(1);
          return std::make_unique<sycl::kernel_bundle<sycl::bundle_state::executable>>(std::move(exe));
        } catch (const std::exception& e) {
          // Committed to JIT (user forced it, or an earlier period already JIT'd): a failure now is a
          // real error. Undecided (first build probing): fall back permanently to spec-const.
          if (cuda_use_jit.load() == 1) throw;
          fprintf(stderr, "kawpow: CUDA source-JIT unavailable (%s); using the slower AOT "
                  "spec-constant kernel\n", e.what());
          cuda_use_jit.store(0);
        }
      }
      // cuda_use_jit == 0: fall through to the spec-constant build below.
    }
#endif
    // Intel/spir64: SYCL-2020 specialization constants fold the per-period program at JIT time.
    auto input = sycl::get_kernel_bundle<sycl::bundle_state::input>(
      queue.get_context(), {device}, {sycl::get_kernel_id<KawpowSgKernel>()}
    );
    input.set_specialization_constant<kawpow_program_id>(make_program(new_period));
    input.set_specialization_constant<kawpow_dag_mod_id>(dag_mod);
    return std::make_unique<sycl::kernel_bundle<sycl::bundle_state::executable>>(
      sycl::build(input)
    );
  }

  // The seal participates in the cache key: a device's KawpowState is shared across the
  // kawpow/firopow/evrprogpow entrypoints, so an algo switch can change the seal even when the
  // period/epoch are unchanged -- it must force a rebuild. `s`/`m` are the current call's seal.
  static bool seal_eq(const SealMode a, const uint32_t am[15], const SealMode b, const uint32_t bm[15]) {
    if (a != b) return false;
    for (unsigned i = 0; i < 15; ++i) if (am[i] != bm[i]) return false;
    return true;
  }

  void ensure_period_bundle(const uint64_t new_period, const uint32_t new_epoch, const FastModData dag_mod) {
    if (period_bundle && period_bundle_period == new_period && period_bundle_epoch == new_epoch &&
        seal_eq(period_bundle_seal, period_bundle_magic, seal, magic)) return;

    if (prefetch_thread.joinable()) prefetch_thread.join();
    if (next_bundle && next_bundle_period == new_period && next_bundle_epoch == new_epoch &&
        seal_eq(next_bundle_seal, next_bundle_magic, seal, magic)) {
      period_bundle = std::move(next_bundle);
    } else {
      period_bundle = build_period_bundle(new_period, dag_mod, seal, magic);
    }
    period_bundle_period = new_period;
    period_bundle_epoch = new_epoch;
    period_bundle_seal = seal;
    for (unsigned i = 0; i < 15; ++i) period_bundle_magic[i] = magic[i];

    next_bundle.reset();
    next_bundle_period = new_period + 1;
    next_bundle_epoch = new_epoch;
    next_bundle_seal = seal;
    for (unsigned i = 0; i < 15; ++i) next_bundle_magic[i] = magic[i];
    // Capture seal/magic by value so the prefetch build is race-free w.r.t. a concurrent algo switch
    // writing state.seal/state.magic on the next call (FiroPow rebuilds every block, so this runs).
    SealMode pf_seal = seal;
    std::array<uint32_t, 15> pf_magic;
    for (unsigned i = 0; i < 15; ++i) pf_magic[i] = magic[i];
    prefetch_thread = std::thread([this, next_period = new_period + 1, dag_mod, pf_seal, pf_magic] {
      try {
        next_bundle = build_period_bundle(next_period, dag_mod, pf_seal, pf_magic.data());
      } catch (...) {
        next_bundle.reset();
      }
    });
  }

  static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()
    ).count();
  }
};

static KawpowState& kawpow_state(const std::string& dev_str) {
  static std::mutex states_mutex;
  static std::map<std::string, std::unique_ptr<KawpowState>> states;

  std::lock_guard<std::mutex> lock(states_mutex);
  auto& state = states[dev_str];
  if (!state) state = std::make_unique<KawpowState>(dev_str);
  return *state;
}

} // namespace mom_kawpow

using namespace mom_kawpow;

// Per-work-item value-captured copy of the seal mode + 15 magic words (the device lambda cannot
// dereference the host `KawpowVariant::magic` constexpr pointer). FIRO ignores the magic words.
struct SealParams {
  SealMode seal;
  uint32_t magic[15];
};

static int kawpow_core(
  const KawpowVariant& variant,
  const uint32_t block_height, const uint8_t* const input, const unsigned input_size, uint8_t* const output,
  uint8_t* const mix_hash, uint64_t* const pnonce, const uint64_t target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < 40) throw std::string("Bad kawpow input length");

  const bool loop_stats = kawpow_loop_stats();
  const uint64_t t_enter = loop_stats ? kawpow_now_us() : 0;

  SealParams seal_params{variant.seal, {}};
  for (unsigned i = 0; i < 15; ++i) seal_params.magic[i] = variant.magic[i];

  KawpowState& state = kawpow_state(dev_str);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  // Publish this call's seal into the state (under the mutex) so build_period_bundle bakes it into the
  // CUDA JIT source and the bundle cache key reflects it. A device shared across algos can switch seal.
  state.seal = seal_params.seal;
  for (unsigned i = 0; i < 15; ++i) state.magic[i] = seal_params.magic[i];
  state.ensure_input(input_size);

  const uint32_t epoch = block_height / variant.epoch_length;
  const uint64_t period = block_height / variant.period_length;
  // FiroPow's full dataset uses chfast/EIP-1057 sizing (1.5 GiB init); KawPoW/EvrProgPow use the
  // classic Ethereum dataset table. The cache sizing is shared (cache_sizes[]).
  if (epoch >= 2048) throw std::string("Bad kawpow epoch");
  const uint64_t dag_bytes = variant.chfast_init_items
    ? chfast_dag_bytes(epoch, variant.chfast_init_items, variant.chfast_growth_items)
    : dag_sizes[epoch];
  const uint64_t t_state = loop_stats ? kawpow_now_us() : 0;
  state.ensure_epoch(epoch, dag_bytes, !is_benchmark);
  const uint64_t t_epoch = loop_stats ? kawpow_now_us() : 0;
  state.ensure_period(period);

  uint64_t start_nonce = 0;
  std::memcpy(&start_nonce, input + 32, sizeof(start_nonce));
  const uint32_t global_size = round_up(std::max(intensity, state.workgroup), state.workgroup);
  const uint32_t dag_elements = static_cast<uint32_t>(dag_bytes / 256);
  const FastModData dag_mod = make_fast_mod_data(dag_elements);

  if (state.shared_io) std::memcpy(state.input, input, input_size);
  else state.queue.memcpy(state.input, input, input_size);
  std::memset(state.result, 0, sizeof(KawpowResult));

  sycl::queue& q = state.queue;
  const unsigned local_size = state.workgroup;
  uint8_t* const d_input = state.input;
  const DagLoad* const d_dag_load = reinterpret_cast<const DagLoad*>(state.dag);
  KawpowResult* const d_result = state.result;

  // GPU devices use the subgroup (warp-shuffle) exchange; the CPU SYCL device
  // (is_gpu()==false) and an explicit MOM_KAWPOW_EXCHANGE=slm use the
  // barrier/local-memory exchange.
  const char* const exchange_env = std::getenv("MOM_KAWPOW_EXCHANGE");
  const bool use_sg = state.device.is_gpu() && !(exchange_env && std::strcmp(exchange_env, "slm") == 0);

  if (use_sg) {
    // Sub-group (warp-shuffle) exchange path. KawpowSgExchange is sub-group-size-agnostic (its
    // `& ~(LANES-1)` base runs two 16-lane ProgPoW groups inside a 32-wide warp), so the same kernel
    // runs at sub-group=16 on Intel and natively at warp=32 on CUDA (reqd_sub_group_size dropped here).
    // The per-period ProgPoW program is folded to constants: Intel via SYCL-2020 specialization
    // constants + kernel_bundle; CUDA via a runtime source-compiled kernel (kawpow_jit.inc).
    const uint64_t t_io = loop_stats ? kawpow_now_us() : 0;
    state.ensure_period_bundle(period, epoch, dag_mod);
    const uint64_t t_bundle = loop_stats ? kawpow_now_us() : 0;
    sycl::event search_event;
#if defined(MOM_SYCL_HAS_CUDA)
    // Must match build_period_bundle's choice: launch the JIT kernel iff that path was selected.
    if (mom_is_cuda(state.device) && state.cuda_use_jit.load() == 1) {
    // Launch the source-JIT'd kernel: program/dag-mod are baked into the bundle, so the only kernel
    // args are the per-call buffers/scalars; work-group local memory comes from work_group_static.
    sycl::kernel jit_kernel = state.period_bundle->ext_oneapi_get_kernel("kawpow_search_jit");
    search_event = q.submit([&](sycl::handler& h) {
      h.set_args(d_input, d_dag_load, d_result,
                 static_cast<uint32_t>(intensity), start_nonce, target, static_cast<int>(is_test));
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)), jit_kernel);
    });
    } else
#endif
    {
    search_event = q.submit([&](sycl::handler& h) {
      const auto c_dag = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(KAWPOW_CACHE_WORDS), h);
      h.use_kernel_bundle(*state.period_bundle);
      h.parallel_for<KawpowSgKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item, sycl::kernel_handler kh) MOM_REQD_SG_16 {
          const KawpowProgram program = kh.get_specialization_constant<kawpow_program_id>();
          const FastModData dag_mod = kh.get_specialization_constant<kawpow_dag_mod_id>();
          const uint32_t lid = item.get_local_id(0);
          const uint32_t gid = item.get_global_id(0);
          const bool active = gid < intensity;
          const uint64_t full_nonce = start_nonce + gid;

          for (uint32_t word = lid * KAWPOW_DAG_LOADS; word < KAWPOW_CACHE_WORDS;
               word += item.get_local_range(0) * KAWPOW_DAG_LOADS) {
            const DagLoad load = d_dag_load[word / KAWPOW_DAG_LOADS];
#pragma unroll
            for (unsigned i = 0; i < KAWPOW_DAG_LOADS; ++i) c_dag[word + i] = load.s[i];
          }
          item.barrier(sycl::access::fence_space::local_space);

          const auto sg = item.get_sub_group();
          const uint32_t lane_id = static_cast<uint32_t>(sg.get_local_id()[0]) & (KAWPOW_LANES - 1);

          uint32_t state2[8];
          kawpow_initial_dev(d_input, static_cast<uint32_t>(full_nonce),
                             static_cast<uint32_t>(full_nonce >> 32),
                             seal_params.seal, seal_params.magic, state2);

          uint32_t digest[8];
          const KawpowSgExchange ex{sg,
            static_cast<uint32_t>(sg.get_local_id()[0]) & ~(KAWPOW_LANES - 1u)};
          kawpow_search_dev(ex, lane_id, state2, c_dag, d_dag_load, program, dag_mod, digest);

          uint32_t final_state[25];
          kawpow_final_dev(state2, digest, seal_params.seal, seal_params.magic, final_state);
          if ((is_test && gid == 0) || (active && target && kawpow_meets_target_words(final_state, target))) {
            store_kawpow_result(d_result, full_nonce, final_state, digest);
          }
        }
      );
    });
    }
    if (loop_stats) {
      const uint64_t t_submit = kawpow_now_us();
      const uint64_t pre_us = t_submit - t_enter;
      if (pre_us > 50000) fprintf(stderr,
        "KAWSTALL t=%llu pre=%.1fms (state=%.1f epoch=%.1f period+io=%.1f bundle=%.1f submit=%.1f)\n",
        static_cast<unsigned long long>(time(nullptr)),
        pre_us / 1e3, (t_state - t_enter) / 1e3, (t_epoch - t_state) / 1e3,
        (t_io - t_epoch) / 1e3, (t_bundle - t_io) / 1e3, (t_submit - t_bundle) / 1e3);
    }
    sycl_wait_and_throw(search_event, state.device);
  } else {
    sycl_wait_and_throw(q.submit([&](sycl::handler& h) {
      const auto share = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(local_size), h);
      const auto offsets = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(local_size / KAWPOW_LANES), h);
      const auto c_dag = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(KAWPOW_CACHE_WORDS), h);
      h.set_specialization_constant<kawpow_program_id>(state.program);
      h.set_specialization_constant<kawpow_dag_mod_id>(dag_mod);
      h.set_specialization_constant<kawpow_cpu_offset_barrier_id>(state.device.is_cpu());
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item, sycl::kernel_handler kh) {
          const KawpowProgram program = kh.get_specialization_constant<kawpow_program_id>();
          const FastModData dag_mod = kh.get_specialization_constant<kawpow_dag_mod_id>();
          const bool cpu_offset_barrier = kh.get_specialization_constant<kawpow_cpu_offset_barrier_id>();
          const uint32_t lid = item.get_local_id(0);
          const uint32_t gid = item.get_global_id(0);
          const bool active = gid < intensity;

          const uint32_t lane_id = lid & (KAWPOW_LANES - 1);
          const uint32_t group_id = lid / KAWPOW_LANES;
          const uint64_t full_nonce = start_nonce + gid;

          for (uint32_t word = lid * KAWPOW_DAG_LOADS; word < KAWPOW_CACHE_WORDS;
               word += item.get_local_range(0) * KAWPOW_DAG_LOADS) {
            const DagLoad load = d_dag_load[word / KAWPOW_DAG_LOADS];
#pragma unroll
            for (unsigned i = 0; i < KAWPOW_DAG_LOADS; ++i) c_dag[word + i] = load.s[i];
          }
          item.barrier(sycl::access::fence_space::local_space);

          uint32_t state2[8];
          kawpow_initial_dev(d_input, static_cast<uint32_t>(full_nonce),
                             static_cast<uint32_t>(full_nonce >> 32),
                             seal_params.seal, seal_params.magic, state2);

          uint32_t digest[8];
          const KawpowSlmExchange ex{
            item, &share[group_id * KAWPOW_LANES], &offsets[group_id], lane_id, cpu_offset_barrier
          };
          kawpow_search_dev(ex, lane_id, state2, c_dag, d_dag_load, program, dag_mod, digest);

          uint32_t final_state[25];
          kawpow_final_dev(state2, digest, seal_params.seal, seal_params.magic, final_state);
          if ((is_test && gid == 0) || (active && target && kawpow_meets_target_words(final_state, target))) {
            store_kawpow_result(d_result, full_nonce, final_state, digest);
          }
        }
      );
    }), state.device);
  }

  const uint32_t result_count = std::min(state.result->count, MAX_KAWPOW_OUTPUTS);
  for (uint32_t index = 0; index < result_count; ++index) {
    if (!is_test && !kawpow_meets_target_words(state.result->output[index], target)) continue;

    std::memcpy(output, state.result->output[index], HASH_LEN);
    std::memcpy(mix_hash, state.result->mix_hash[index], HASH_LEN);
    *pnonce = state.result->nonce[index];
    return 1;
  }

  return 0;
}

// ABI entrypoints (gpu_kawpow_hash_fun). The fn-pointer signature does not carry the algo name, so
// each ProgPoW variant is a DISTINCT exported function that bakes its own epoch/period/seal/magic.
int kawpow(
  const unsigned job_id, const uint32_t block_height, const uint8_t* const input, const unsigned input_size,
  uint8_t* const output, uint8_t* const mix_hash, uint64_t* const pnonce, const uint64_t target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  return kawpow_core(KAWPOW_VARIANT, block_height, input, input_size, output, mix_hash, pnonce,
                     target, intensity, is_test, is_benchmark, dev_str);
}

int firopow(
  const unsigned job_id, const uint32_t block_height, const uint8_t* const input, const unsigned input_size,
  uint8_t* const output, uint8_t* const mix_hash, uint64_t* const pnonce, const uint64_t target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  return kawpow_core(FIROPOW_VARIANT, block_height, input, input_size, output, mix_hash, pnonce,
                     target, intensity, is_test, is_benchmark, dev_str);
}

int evrprogpow(
  const unsigned job_id, const uint32_t block_height, const uint8_t* const input, const unsigned input_size,
  uint8_t* const output, uint8_t* const mix_hash, uint64_t* const pnonce, const uint64_t target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  return kawpow_core(EVRPROGPOW_VARIANT, block_height, input, input_size, output, mix_hash, pnonce,
                     target, intensity, is_test, is_benchmark, dev_str);
}
