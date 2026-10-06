#ifndef MOM_PEARLHASH_ESIMD_TU
#include "claim.h"

using namespace mom_pearlhash;

// 52-byte NoisyGEMM config block hashed with the 76-byte header into the per-job key (k, rank, and
// the two sparsity bytes at [9]/[15]); bytes 0..3 = k LE, 4..5 = rank LE.
static void config_bytes(uint8_t cfg[52], int k, int rank) {
  memset(cfg, 0, 52);
  cfg[0] = k & 0xff;
  cfg[1] = (k >> 8) & 0xff;
  cfg[2] = (k >> 16) & 0xff;
  cfg[3] = (k >> 24) & 0xff;
  cfg[4] = rank & 0xff;
  cfg[5] = (rank >> 8) & 0xff;
  cfg[9] = 15;
  cfg[15] = 15;
}
// Per-job key = unkeyed BLAKE3 of the 76-byte header concatenated with the 52-byte config block.
static void derive_key(const uint8_t* header76, int k, int rank, uint8_t key32[32]) {
  uint8_t cfg[52], kb[128];
  config_bytes(cfg, k, rank);
  memcpy(kb, header76, 76);
  memcpy(kb + 76, cfg, 52);
  pearlhash_b3::b3(kb, 128, nullptr, key32);
}

// ---- PlainProof construction on the host (the pool submission for a winning tile) ----
// On a winning tile the server builds the pool submission itself: it regenerates the A / Bt byte
// streams from the counter-RNG, builds the keyed-BLAKE3 Merkle tree, opens the 16 revealed rows
// (A) and cols (Bt), bincode-serializes and base64-encodes. Mirrors pearl-blake3 merkle.rs and is
// byte-identical to the pool's raw-root proof format. Certificate V3 changes only the noise-seed
// commitment chain, not these serialized roots. Runs once per found
// share (rare), so plain host code -- no GPU. CHUNK=1024; leaves use non-root chunk_cv, the root
// uses merge_root (ROOT at top); for power-of-two leaf counts the pairwise odd-carry layer tree
// (siblings) is identical to merge_root's tree.
using CV = std::array<uint8_t, 32>;
// Regenerate leaf `leaf` (1024 bytes, zero-padded) of A ('A') or Bt ('B') from the seed RNG.
static size_t gen_leaf(char which, uint32_t seed, size_t leaf, int m, int n, int k,
                       uint8_t out[1024]) {
  const size_t total = checked_elements(which == 'A' ? m : n, k, "PlainProof matrix");
  const size_t num_leaves = checked_chunk_count(total, "PlainProof matrix");
  if (leaf >= num_leaves)
    throw std::string("Invalid pearlhash PlainProof leaf");
  const size_t offset = leaf * 1024;
  const size_t len = std::min(size_t{1024}, total - offset);
  const uint32_t tot = static_cast<uint32_t>(checked_elements(m, k, "M*K PlainProof"));
  for (size_t t = 0; t < len; t++) {
    const size_t byte = offset + t;
    if (which == 'A') {
      out[t] = static_cast<uint8_t>(gv(seed, static_cast<uint32_t>(byte)));
    } else {
      const size_t source_offset =
          (byte % static_cast<size_t>(k)) * static_cast<size_t>(n) + byte / k;
      out[t] = static_cast<uint8_t>(gv(seed, tot + static_cast<uint32_t>(source_offset)));
    }
  }
  for (size_t t = len; t < 1024; t++)
    out[t] = 0;
  return len;
}
static std::vector<std::vector<CV>> build_layers(char which, uint32_t seed, int numLeaves, int m,
                                                 int n, int k, const uint8_t* key) {
  std::vector<CV> l0(numLeaves);
  uint8_t chunk[1024];
  for (int i = 0; i < numLeaves; i++) {
    int len = gen_leaf(which, seed, i, m, n, k, chunk);
    pearlhash_b3::chunk_cv(chunk, (uint32_t)len, (uint64_t)i, key, l0[i].data());
  }
  std::vector<std::vector<CV>> layers;
  layers.push_back(std::move(l0));
  while (layers.back().size() > 2) {
    auto& prev = layers.back();
    std::vector<CV> next;
    for (size_t i = 0; i < prev.size(); i += 2) {
      if (i + 1 < prev.size()) {
        CV p;
        pearlhash_b3::parent_cv(prev[i].data(), prev[i + 1].data(), key, false, p.data());
        next.push_back(p);
      } else
        next.push_back(prev[i]);
    }
    layers.push_back(std::move(next));
  }
  return layers;
}
static void put_u64le(std::vector<uint8_t>& v, uint64_t n) {
  for (int i = 0; i < 8; i++)
    v.push_back((uint8_t)(n >> (8 * i)));
}
static void put_bytes(std::vector<uint8_t>& v, const uint8_t* p, size_t n) {
  v.insert(v.end(), p, p + n);
}
// Append a serialized MatrixMerkleProof for the 16 contiguous `rows` of matrix `which` (cols wide).
static void ser_matrix_proof(std::vector<uint8_t>& out, char which, uint32_t seed, int m, int n,
                             int k, const uint8_t* key, int cols, const std::vector<int>& rows) {
  const size_t total_bytes = checked_elements(which == 'A' ? m : n, k, "PlainProof matrix");
  const size_t num_leaves_size = checked_chunk_count(total_bytes, "PlainProof matrix");
  const int numLeaves = static_cast<int>(num_leaves_size);
  auto layers = build_layers(which, seed, numLeaves, m, n, k, key);
  uint8_t root[32];
  {
    std::vector<CV> tmp = layers[0];
    pearlhash_b3::merge_root((uint8_t*)tmp.data(), numLeaves, key, root);
  }
  std::set<int> leafset;
  for (int row : rows) {
    const size_t row_start = static_cast<size_t>(row) * static_cast<size_t>(cols);
    const size_t row_end = (static_cast<size_t>(row) + 1) * static_cast<size_t>(cols);
    const size_t first = row_start / 1024;
    const size_t last = (row_end - 1) / 1024;
    if (last >= num_leaves_size || last > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::string("Invalid pearlhash PlainProof row leaf range");
    for (size_t i = first; i <= last; i++)
      leafset.insert(static_cast<int>(i));
  }
  std::vector<int> leaves(leafset.begin(), leafset.end());
  // MerkleProof: leaf_data (Vec<[u8;1024]>), leaf_indices, total_leaves, root, siblings
  put_u64le(out, leaves.size());
  uint8_t chunk[1024];
  for (int li : leaves) {
    gen_leaf(which, seed, static_cast<size_t>(li), m, n, k, chunk);
    put_u64le(out, 1024);
    put_bytes(out, chunk, 1024);
  }
  put_u64le(out, leaves.size());
  for (int li : leaves)
    put_u64le(out, (uint64_t)li);
  put_u64le(out, (uint64_t)numLeaves);
  put_bytes(out, root, 32);
  // siblings: walk up, emit each node whose sibling is not itself in the active set
  std::set<int> cur(leafset);
  size_t levelLen = num_leaves_size;
  size_t level = 0;
  std::vector<CV> sib;
  while (levelLen > 1 && !cur.empty()) {
    auto& nodes = layers[level];
    for (int i : cur) {
      if (i % 2 == 1) {
        if (!cur.count(i - 1))
          sib.push_back(nodes[i - 1]);
      } else if (!cur.count(i + 1) && static_cast<size_t>(i + 1) < levelLen)
        sib.push_back(nodes[i + 1]);
    }
    std::set<int> nx;
    for (int i : cur)
      nx.insert(i >> 1);
    cur.swap(nx);
    levelLen = levelLen / 2 + levelLen % 2;
    level++;
  }
  put_u64le(out, sib.size());
  for (auto& s : sib)
    put_bytes(out, s.data(), 32);
  // MatrixMerkleProof tail: row_indices
  put_u64le(out, rows.size());
  for (int r : rows)
    put_u64le(out, (uint64_t)r);
}
static std::string base64(const std::vector<uint8_t>& d) {
  static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o;
  for (size_t i = 0; i < d.size(); i += 3) {
    uint32_t x =
        d[i] << 16 | (i + 1 < d.size() ? d[i + 1] << 8 : 0) | (i + 2 < d.size() ? d[i + 2] : 0);
    o += T[(x >> 18) & 63];
    o += T[(x >> 12) & 63];
    o += (i + 1 < d.size()) ? T[(x >> 6) & 63] : '=';
    o += (i + 2 < d.size()) ? T[x & 63] : '=';
  }
  return o;
}
// Full PlainProof (base64) for a winning tile, from the seed + job key (no noised matrices needed).
static std::string build_plain_proof(uint32_t seed, int m, int n, int k, int rank,
                                     const uint8_t* key, int row, int col,
                                     uint32_t* adjustment_factor) {
  std::vector<int> aRows, btRows;
  for (int r = 0; r < 16; r++) {
    aRows.push_back(row + r);
    btRows.push_back(col + r);
  }
  if (!pearlhash_adjustment_factor(aRows.size(), btRows.size(), k, rank, adjustment_factor))
    return {};
  std::vector<uint8_t> buf;
  put_u64le(buf, (uint64_t)m);
  put_u64le(buf, (uint64_t)n);
  put_u64le(buf, (uint64_t)k);
  put_u64le(buf, (uint64_t)rank);
  ser_matrix_proof(buf, 'A', seed, m, n, k, key, k, aRows);
  ser_matrix_proof(buf, 'B', seed, m, n, k, key, k, btRows);
  buf.push_back(0x00); // moe: Option::None
  return base64(buf);
}
template <typename T> static T* alloc_buffer(sycl::queue& q, size_t count) {
  return q.get_device().is_cpu() ? sycl::malloc_shared<T>(count, q)
                                 : sycl::malloc_device<T>(count, q);
}

template <typename T>
static T* alloc_required(sycl::queue& q, const size_t count, const char* const name) {
  T* const result = alloc_buffer<T>(q, count);
  if (!result)
    throw std::string("Can't allocate pearlhash ") + name;
  return result;
}

static void free_allocations(sycl::queue& q, Buffers& b) noexcept {
  for (void* pointer :
       {static_cast<void*>(b.EAL), static_cast<void*>(b.EBR), static_cast<void*>(b.EBRt),
        static_cast<void*>(b.Ap), static_cast<void*>(b.Bp), static_cast<void*>(b.ApWmma),
        static_cast<void*>(b.BpWmma), static_cast<void*>(b.EARp1), static_cast<void*>(b.EARp2),
        static_cast<void*>(b.EBLq1), static_cast<void*>(b.EBLq2), static_cast<void*>(b.cA),
        static_cast<void*>(b.cB), static_cast<void*>(b.key), static_cast<void*>(b.target),
        static_cast<void*>(b.CVA), static_cast<void*>(b.CVB),
        static_cast<void*>(b.transcript), static_cast<void*>(b.result)}) {
    if (pointer)
      try {
        sycl::free(pointer, q);
      } catch (...) {
      }
  }
  b = Buffers{};
}

static Buffers alloc_buffers(sycl::queue& q, int m, int n, int k, int rank, bool hip_jit = false,
                             bool cuda_jit = false) {
  static_assert(sizeof(Result) <= pearlhash_result_allocation_bytes);
  Buffers b{};
  PearlHashAllocationCounts counts{};
  if (!pearlhash_allocation_counts({static_cast<unsigned>(m), static_cast<unsigned>(n),
                                   static_cast<unsigned>(k), static_cast<unsigned>(rank)},
                                  counts, hip_jit))
    throw std::string("Invalid pearlhash allocation dimensions");
  try {
    b.EAL = alloc_required<int8_t>(q, counts.m_rank, "EAL");
    b.EBR = alloc_required<int8_t>(q, counts.n_rank, "EBR");
    b.EBRt = alloc_required<int8_t>(q, counts.n_rank, "EBRt");
    b.Ap = alloc_required<int8_t>(q, counts.m_k, "Ap");
    b.Bp = alloc_required<int8_t>(q, counts.n_k, "Bp");
#if defined(MOM_SYCL_HAS_HIP)
    // HIPRTC WMMA consumes an architecture-native fragment layout; keep its matrices separate
    // from the portable SYCL layout used when source-JIT initialization is unavailable.
    if (hip_jit) {
      b.ApWmma = alloc_required<int8_t>(q, counts.m_k, "WMMA Ap");
      b.BpWmma = alloc_required<int8_t>(q, counts.n_k, "WMMA Bp");
    }
#endif
    b.EARp1 = alloc_required<int32_t>(q, static_cast<size_t>(k), "EARp1");
    b.EARp2 = alloc_required<int32_t>(q, static_cast<size_t>(k), "EARp2");
    b.EBLq1 = alloc_required<int32_t>(q, static_cast<size_t>(k), "EBLq1");
    b.EBLq2 = alloc_required<int32_t>(q, static_cast<size_t>(k), "EBLq2");
    b.cA = alloc_required<uint8_t>(q, 32, "cA");
    b.cB = alloc_required<uint8_t>(q, 32, "cB");
    b.key = alloc_required<uint8_t>(q, 32, "key");
    b.target = alloc_required<uint8_t>(q, 32, "target");
    b.CVA = alloc_required<uint8_t>(q, counts.cva_bytes, "CVA");
    b.CVB = alloc_required<uint8_t>(q, counts.cvb_bytes, "CVB");
    // Keep BLAKE3 separate from the matrix loop to preserve occupancy. CUDA processes row slices:
    // the full reference profile would otherwise need a 16-GiB transcript in addition to A'/B'.
    // Other backends still require a complete transcript and keep their existing 256-MiB cap.
    const int transcript_rows = cuda_jit
        ? std::min(m, static_cast<int>(((1ULL << 32) / static_cast<unsigned>(n)) / 128) * 128)
        : m;
    const uint64_t transcript_bytes =
        static_cast<uint64_t>(transcript_rows / 16) * (n / 16) * 16U * sizeof(uint32_t);
    const uint64_t transcript_limit =
        cuda_jit ? 1ULL << 30 : 256ULL * 1024U * 1024U;
    const sycl::device device = q.get_device();
    const PowDeviceProfile capacity = {
        1, device.get_info<sycl::info::device::global_mem_size>(),
        device.get_info<sycl::info::device::max_mem_alloc_size>()};
    const bool transcript_fits = !cuda_jit ||
        pearlhash_optional_allocation_fits(capacity, counts, transcript_bytes);
    if (transcript_bytes > 0 && transcript_bytes <= transcript_limit && transcript_fits) {
      b.transcript =
          alloc_buffer<uint32_t>(q, static_cast<size_t>(transcript_bytes / sizeof(uint32_t)));
      if (b.transcript)
        b.transcript_rows = transcript_rows;
    }
    b.result = sycl::malloc_shared<Result>(1, q);
    if (!b.result)
      throw std::string("Can't allocate pearlhash result");
    b.result->found = 0;
    return b;
  } catch (...) {
    free_allocations(q, b);
    throw;
  }
}

// ---- native mom entry point (DEV::PEARLHASH_GPU) ----
namespace {
// PlainProof carries M/N/K/rank, so the job can select a device-efficient profile without changing
// proof semantics. mom.js obtains defaults from GPU-CONFIG.md and passes explicit config values.
// One persistent in-order queue + device buffer set per GPU; the seed search reuses them.
struct PearlHashState {
  sycl::queue queue;
  sycl::queue prep_queue;
#ifdef MOM_SYCL_HAS_CUDA
  const bool cuda_tensor = mom::nvidia::has_int8_async_matrix(
      mom::nvidia::compute_capability(queue.get_device()));
#else
  static constexpr bool cuda_tensor = false;
#endif
  PearlHashHipSearch hip;
  PearlHashCudaSearch cuda;
  int m = 0, n = 0, k = 0, rank = 0;
  bool hip_jit_buffers = false;
  bool cuda_jit_buffers = false;
  Buffers b{};
  uint8_t key[32] = {0}, header[76] = {0};
  bool have_header = false;
  bool native_notice = false;
  double wait_ema_us = 0.0; // EMA of recent attempt wall times (us); paces the pre-wait sleep below
  std::mutex mutex;
  explicit PearlHashState(const std::string& dev_str)
      : queue(get_dev(dev_str), sycl::property::queue::in_order{}),
        prep_queue(queue.get_context(), queue.get_device(), sycl::property::queue::in_order{}) {
  }
  void release() noexcept {
    sycl_cleanup_noexcept("pearlhash queue", [&] {
      queue.wait_and_throw();
    });
    sycl_cleanup_noexcept("pearlhash prep queue", [&] {
      prep_queue.wait_and_throw();
    });
    hip.release();
    cuda.release();
    free_allocations(queue, b);
    m = n = k = rank = 0;
    hip_jit_buffers = false;
    cuda_jit_buffers = false;
    have_header = false;
  }
  void ensure(int m_, int n_, int k_, int rank_, bool hip_jit_, bool cuda_jit_) {
    if (b.Ap && m == m_ && n == n_ && k == k_ && rank == rank_ && hip_jit_buffers == hip_jit_ &&
        cuda_jit_buffers == cuda_jit_)
      return;
    release();
    Buffers next = alloc_buffers(queue, m_, n_, k_, rank_, hip_jit_, cuda_jit_);
    b = next;
    m = m_;
    n = n_;
    k = k_;
    rank = rank_;
    hip_jit_buffers = hip_jit_;
    cuda_jit_buffers = cuda_jit_;
    have_header = false;
  }
  ~PearlHashState() {
    release();
  }
};
DeviceStateRegistry<PearlHashState>& pearlhash_registry() {
  static auto* const registry = new DeviceStateRegistry<PearlHashState>;
  return *registry;
}
PearlHashState& pearlhash_state(const std::string& dev_str) {
  return pearlhash_registry().get(dev_str, [&] {
    return std::make_unique<PearlHashState>(dev_str);
  });
}
thread_local std::string
    g_pearlhash_proof; // last built proof, fetched right after pearlhash() returns 1
// Winning tile captured by pearlhash() on a find; consumed by pearlhash_claim() to build the proof lazily.
thread_local struct PearlHashFound {
  uint32_t seed;
  int row, col, m, n, k, rank;
  uint8_t key[32];
  uint8_t jackpot[32];
  uint32_t adjustment_factor;
  bool claim_ready;
  bool valid;
} g_pf{};
} // namespace

void pearlhash_cleanup_states() noexcept {
  try {
#if defined(_WIN32)
    // Intel's standalone Windows CPU OpenCL ICD can fault after a successful PearlHash result while
    // DPC++ destroys this algorithm's shared-USM allocation/queue graph. The state registry itself
    // is deliberately heap-owned and has no process-exit destructor, so retain only CPU states and
    // let process teardown reclaim them. Discrete GPU states still require ordered cleanup (notably
    // AdaptiveCpp/CUDA, which otherwise reports cudaErrorCudartUnloading) and are erased normally.
    pearlhash_registry().clear([](const auto& entry) {
      return !entry.second->queue.get_device().is_cpu();
    });
#else
    pearlhash_registry().clear();
#endif
  } catch (...) {
    std::fprintf(stderr, "pearlhash: ordered SYCL cleanup failed\n");
  }
}

// One seeded NoisyGEMM attempt per call (seed = *pseed). Returns 1 on a winning tile, writes the
// winning seed back to *pseed, and stashes the base64 PlainProof for pearlhash_claim(). intensity = M.
int pearlhash(const unsigned job_ref, const uint32_t, const uint8_t* const input,
              const unsigned input_size, uint8_t* const, uint64_t* const pseed,
              const uint8_t* const target, const unsigned intensity, const bool is_test, const bool,
              const std::string& dev_str, const std::string& backend, const unsigned requested_n,
              const unsigned requested_k, const unsigned requested_rank,
              const unsigned cert_version) {
  if (!input || !pseed || input_size < 76)
    throw std::string("Bad pearlhash input length (need 76-byte header)");
  if (!is_test && !target)
    throw std::string("PearlHash target is required outside test mode");
  constexpr unsigned max_dimension =
      static_cast<unsigned>(std::numeric_limits<int>::max());
  if (intensity > max_dimension || requested_n > max_dimension || requested_k > max_dimension ||
      requested_rank > max_dimension)
    throw std::string("PearlHash dimensions exceed 32-bit kernel indexing");
  const int k = static_cast<int>(requested_k);
  const int rank = static_cast<int>(requested_rank);
  if (cert_version != 3)
    throw std::string("Unsupported pearlhash certificate version");
  if (rank < 128)
    throw std::string("PearlHash certificate V3 requires rank >= 128");
  // M arrives as the worker's explicit primary tuning value; N can be selected independently. Native HIP
  // tests use the smallest square that activates the 32x32 cache-block traversal, so the
  // native-vs-SYCL checksum comparison covers production tile mapping without making CPU tests
  // expensive. Other implementations retain the compact 256x256 vector.
  const int test_edge = backend == "native" ? 2048 : 256;
  const int m = is_test ? test_edge
                        : (intensity >= 128 ? static_cast<int>(intensity) : 131072);
  const int n = is_test ? m : static_cast<int>(requested_n);
  if (!mom::job_boundary::valid_pearlhash_shape(
          static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k),
          static_cast<uint64_t>(rank)))
    throw std::string("Invalid pearlhash M/N/K/rank profile");
  (void)pearlhash_attempt_hashes(intensity, requested_n, requested_k, requested_rank);

  g_pf.valid = false;
  g_pf.claim_ready = false;
  PearlHashState& st = pearlhash_state(dev_str);
  std::lock_guard<std::mutex> lock(st.mutex);
  const bool native_requested = backend == "auto" || backend == "native";
  std::string hip_reason, cuda_reason;
  const bool hip_device = native_requested && PearlHashHipSearch::supports(st.queue, &hip_reason);
  const bool cuda_device = native_requested && PearlHashCudaSearch::supports(st.queue, &cuda_reason);
  const bool hip_jit = hip_device && PearlHashHipSearch::supports_shape(m, n, k, rank);
  const bool cuda_jit = cuda_device && PearlHashCudaSearch::supports_shape(m, n, k, rank);
  if (hip_device && !hip_jit)
    hip_reason = "the selected PearlHash profile requires the shared SYCL path";
  if (cuda_device && !cuda_jit)
    cuda_reason = "the selected PearlHash profile requires the shared SYCL path";
  if (native_requested && !hip_jit && !cuda_jit && !st.native_notice) {
    st.native_notice = true;
    std::fprintf(
        stderr, "PearlHash native backend unavailable, using SYCL: %s\n",
        (PearlHashCudaSearch::is_cuda(st.queue.get_device()) ? cuda_reason : hip_reason).c_str());
  }
  st.ensure(m, n, k, rank, hip_jit, cuda_jit);
  sycl::queue& q = st.queue;
  Buffers& b = st.b;

  uint8_t tgtLE[32];
  const uint32_t attempt_seed = static_cast<uint32_t>(*pseed);
  PearlHashSearchBackend search_backend;
  std::chrono::steady_clock::time_point attempt_start;
  try {
    if (!st.have_header ||
        std::memcmp(st.header, input, 76) != 0) { // (re)derive key on header change
      st.have_header = false;
      std::memcpy(st.header, input, 76);
      derive_key(input, k, rank, st.key);
      q.memcpy(b.key, st.key, 32);
    }
    (void)job_ref;
    if (is_test)
      std::memset(tgtLE, 0xff, 32); // test: the first tile wins
    else
      for (int i = 0; i < 32; i++)
        tgtLE[i] = target[31 - i]; // core gives BE; kernel wants LE
    q.memcpy(b.target, tgtLE, 32);
    b.result->found = 0;
    b.result->chk =
        0; // whole-search checksum (filled only when MOM_PEARLHASH_CHK is set; see tile_mix)

    search_backend =
        attempt(q, b, attempt_seed, m, n, k, rank, backend, &st.hip, &st.cuda,
                &st.prep_queue, is_test, st.cuda_tensor);
    // Start after submission: a first call can include synchronous device-image/JIT setup, and feeding
    // that one-off cost into the wait EMA makes later completed GPU attempts sleep needlessly.
    attempt_start = std::chrono::steady_clock::now();
    // Intel SYCL waits must drive queue progress before updating the attempt-time EMA;
    // sleeping first can feed the delay back into subsequent attempts. Keep other backends' pacing.
    const bool intel_sycl_gpu = search_backend == PearlHashSearchBackend::sycl &&
        q.get_device().is_gpu() &&
        q.get_device().get_info<sycl::info::device::vendor_id>() == 0x8086;
    if (st.wait_ema_us > 2000.0 && !intel_sycl_gpu) {
      const double elapsed_us =
          static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - attempt_start)
                                  .count());
      const double sleep_us = st.wait_ema_us * 0.90 - elapsed_us;
      if (sleep_us > 0.0)
        std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(sleep_us)));
    }
    if (search_backend == PearlHashSearchBackend::hip_jit ||
        search_backend == PearlHashSearchBackend::cuda_jit) {
      if (search_backend == PearlHashSearchBackend::hip_jit)
        st.hip.wait(st.wait_ema_us);
      else
        st.cuda.wait(st.wait_ema_us);
    } else {
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER)
      // OpenCL keeps its native wait/progress semantics; other Intel GPU queues poll completion
      // before sleeping, without reintroducing EMA pre-wait delays.
      if (intel_sycl_gpu && !mom_is_opencl(q.get_device()))
        sycl_wait_and_throw(q.ext_oneapi_submit_barrier(), q.get_device());
      else
#endif
        q.wait_and_throw();
    }
    st.have_header = true;
  } catch (...) {
    // Keep the staged target alive until any accepted copy has retired.
    sycl_cleanup_noexcept("pearlhash target upload", [&] { q.wait_and_throw(); });
    throw;
  }
  if (is_test && q.get_device().is_gpu()) {
    // Search backends share GPU preparation, so agreement between them cannot validate the
    // commitment roots or V3 noise seeds. Recompute these independently on the host in tests.
    const std::size_t chunks_a = (static_cast<std::size_t>(m) * k + 1023) / 1024;
    const std::size_t chunks_b = (static_cast<std::size_t>(n) * k + 1023) / 1024;
    std::vector<uint8_t> cvs_a(chunks_a * 32), cvs_b(chunks_b * 32);
    std::array<uint8_t, 32> expected_a{}, expected_b{}, observed_a{}, observed_b{};
    Buffers reference{};
    reference.key = st.key;
    reference.CVA = cvs_a.data();
    reference.CVB = cvs_b.data();
    reference.cA = expected_a.data();
    reference.cB = expected_b.data();
    prepare_roots_cpu(reference, attempt_seed, m, n, k);
    try {
      q.memcpy(observed_a.data(), b.cA, observed_a.size());
      q.memcpy(observed_b.data(), b.cB, observed_b.size());
      q.wait_and_throw();
    } catch (...) {
      sycl_cleanup_noexcept("pearlhash commitment readback", [&] { q.wait_and_throw(); });
      throw;
    }
    if (observed_a != expected_a || observed_b != expected_b)
      throw std::string("PearlHash GPU/host commitment seed mismatch");
    const char* const search_name = search_backend == PearlHashSearchBackend::cuda_jit
      ? "cuda-jit" : search_backend == PearlHashSearchBackend::hip_jit ? "hip-jit" : "sycl";
    std::fprintf(stderr, "PEARLHASH_TEST search=%s cert_version=%u host_seeds_match=true\n",
                 search_name, cert_version);
    // Automatic capability fallback may resolve to the reference, not a tuned search.
    if (backend == "sycl")
      std::fprintf(stderr, "PEARLHASH_TEST reference_search=true\n");
  }
  // search() retains the scalar MAC expression, independent of the optional NVIDIA packed-dot path.
  // Explicit "sycl" already uses this reference; tuned searches must compare against it.
  // Keep this outside vendor guards so Intel ESIMD receives the same correctness check.
  if (is_test && q.get_device().is_gpu() && backend != "sycl") {
    const uint32_t accelerated_checksum = b.result->chk;
    b.result->found = 0;
    b.result->chk = 0;
    compute_ab(q, b, attempt_seed, m, n, k, rank, /*portable=*/true);
    search(q, b, attempt_seed, m, n, k, rank, /*dbg=*/true);
    q.wait_and_throw();
    if (b.result->chk != accelerated_checksum) {
      std::ostringstream message;
      message << "PearlHash accelerated/SYCL checksum mismatch (accelerated=0x" << std::hex
              << accelerated_checksum << ", SYCL=0x" << b.result->chk << ')';
      throw message.str();
    }
    std::fprintf(stderr, "PEARLHASH_TEST search_checksum_match=true\n");
  }
  {
    const double us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                              std::chrono::steady_clock::now() - attempt_start)
                                              .count());
    st.wait_ema_us = st.wait_ema_us == 0.0 ? us : st.wait_ema_us * 0.8 + us * 0.2;
  }
  if (std::getenv(
          "MOM_PEARLHASH_CHK")) // cross-validation: same A'/B' -> identical chk on every search path
    std::fprintf(stderr, "PEARLHASH_CHK dev=%s seed=%u chk=%08x\n", dev_str.c_str(),
                 (uint32_t)*pseed, b.result->chk);
  if (!b.result->found)
    return 0;
  *pseed = b.result->seed;
  if (is_test)
    return 1; // test: core only needs "ok", no proof
  // Capture the winning tile; rebuild its expensive PlainProof only when the core requests a claim.
  // Distinct winning seeds under one job are separate shares; retries reuse the captured claim.
  g_pf.seed = b.result->seed;
  g_pf.row = (int)b.result->row;
  g_pf.col = (int)b.result->col;
  g_pf.m = m;
  g_pf.n = n;
  g_pf.k = k;
  g_pf.rank = rank;
  std::memcpy(g_pf.key, st.key, 32);
  std::memcpy(g_pf.jackpot, b.result->jackpot, 32);
  g_pf.claim_ready = false;
  g_pf.valid = true;
  return 1;
}
// Builds and caches the PlainProof plus its matching trusted-share metadata for the last winning tile.
// Repeated calls for the captured tile return the same claim.
const char* pearlhash_claim(uint8_t jackpot[32], uint32_t* adjustment_factor) {
  if (!g_pf.valid || !jackpot || !adjustment_factor)
    return nullptr;
  if (!g_pf.claim_ready) {
    g_pearlhash_proof = build_plain_proof(g_pf.seed, g_pf.m, g_pf.n, g_pf.k, g_pf.rank, g_pf.key,
                                          g_pf.row, g_pf.col, &g_pf.adjustment_factor);
    if (g_pearlhash_proof.empty())
      return nullptr;
    g_pf.claim_ready = true;
  }
  std::memcpy(jackpot, g_pf.jackpot, 32);
  *adjustment_factor = g_pf.adjustment_factor;
  return g_pearlhash_proof.c_str();
}
// Rank-128-equivalent GEMM MACs per attempt. V3's target is normalized to rank 128,
// so this is also the stable accounting unit advertised to switching pools.
uint64_t pearlhash_attempt_hashes(unsigned intensity, unsigned n, unsigned k, unsigned rank) {
  const uint64_t m = intensity >= 128 ? intensity : 131072;
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  if (!mom::job_boundary::valid_pearlhash_shape(m, n, k, rank))
    throw std::string("Invalid PearlHash work factor");
  if (n && m > maximum / n)
    throw std::string("PearlHash work count overflows");
  const uint64_t mn = m * n;
  const uint64_t effective_k = static_cast<uint64_t>(k / rank) * 128;
  if (effective_k && mn > maximum / effective_k)
    throw std::string("PearlHash work count overflows");
  return mn * effective_k;
}

#endif // !MOM_PEARLHASH_ESIMD_TU
