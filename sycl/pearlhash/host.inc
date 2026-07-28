#ifndef MOM_PEARLHASH_ESIMD_TU
using namespace mom_pearlhash;

// 52-byte NoisyGEMM config block hashed with the 76-byte header into the per-job key (k, rank, and
// the two sparsity bytes at [9]/[15]); bytes 0..3 = k LE, 4..5 = rank LE.
static void config_bytes(uint8_t cfg[52], int k, int rank) {
  memset(cfg, 0, 52);
  cfg[0]=k&0xff; cfg[1]=(k>>8)&0xff; cfg[2]=(k>>16)&0xff; cfg[3]=(k>>24)&0xff; cfg[4]=rank&0xff; cfg[5]=(rank>>8)&0xff; cfg[9]=15; cfg[15]=15;
}
// Per-job key = unkeyed BLAKE3 of the 76-byte header concatenated with the 52-byte config block.
static void derive_key(const uint8_t* header76, int k, int rank, uint8_t key32[32]) {
  uint8_t cfg[52], kb[128]; config_bytes(cfg, k, rank);
  memcpy(kb, header76, 76); memcpy(kb + 76, cfg, 52);
  pearlhash_b3::b3(kb, 128, nullptr, key32);
}

// ---- PlainProof construction on the host (the pool submission for a winning tile) ----
// On a winning tile the server builds the pool submission itself: it regenerates the A / Bt byte
// streams from the counter-RNG, builds the keyed-BLAKE3 Merkle tree, opens the 16 revealed rows
// (A) and cols (Bt), bincode-serializes and base64-encodes. Mirrors pearl-blake3 merkle.rs and is
// byte-identical to the old JS path (validated against verify_plain_proof_v2). Runs once per found
// share (rare), so plain host code -- no GPU. CHUNK=1024; leaves use non-root chunk_cv, the root
// uses merge_root (ROOT at top); for power-of-two leaf counts the pairwise odd-carry layer tree
// (siblings) is identical to merge_root's tree.
using CV = std::array<uint8_t, 32>;
// Regenerate leaf `leaf` (1024 bytes, zero-padded) of A ('A') or Bt ('B') from the seed RNG.
static int gen_leaf(char which, uint32_t seed, int leaf, int m, int n, int k, uint8_t out[1024]) {
  const uint32_t tot = (uint32_t)(m * k);
  const int total = (which == 'A' ? m : n) * k, off = leaf * 1024;
  int len = total - off < 1024 ? total - off : 1024;
  for (int t = 0; t < len; t++) {
    int b = off + t;
    out[t] = which == 'A' ? (uint8_t)gv(seed, (uint32_t)b)
                          : (uint8_t)gv(seed, tot + (uint32_t)((b % k) * n + (b / k)));
  }
  for (int t = len; t < 1024; t++) out[t] = 0;
  return len;
}
static std::vector<std::vector<CV>> build_layers(char which, uint32_t seed, int numLeaves, int m, int n, int k, const uint8_t* key) {
  std::vector<CV> l0(numLeaves); uint8_t chunk[1024];
  for (int i = 0; i < numLeaves; i++) { int len = gen_leaf(which, seed, i, m, n, k, chunk); pearlhash_b3::chunk_cv(chunk, (uint32_t)len, (uint64_t)i, key, l0[i].data()); }
  std::vector<std::vector<CV>> layers; layers.push_back(std::move(l0));
  while (layers.back().size() > 2) {
    auto& prev = layers.back(); std::vector<CV> next;
    for (size_t i = 0; i < prev.size(); i += 2) {
      if (i + 1 < prev.size()) { CV p; pearlhash_b3::parent_cv(prev[i].data(), prev[i + 1].data(), key, false, p.data()); next.push_back(p); }
      else next.push_back(prev[i]);
    }
    layers.push_back(std::move(next));
  }
  return layers;
}
static void put_u64le(std::vector<uint8_t>& v, uint64_t n) { for (int i = 0; i < 8; i++) v.push_back((uint8_t)(n >> (8 * i))); }
static void put_bytes(std::vector<uint8_t>& v, const uint8_t* p, int n) { v.insert(v.end(), p, p + n); }
// Append a serialized MatrixMerkleProof for the 16 contiguous `rows` of matrix `which` (cols wide).
static void ser_matrix_proof(std::vector<uint8_t>& out, char which, uint32_t seed, int m, int n, int k,
                             const uint8_t* key, int cols, const std::vector<int>& rows) {
  const int numLeaves = ((which == 'A' ? m : n) * k) / 1024;
  auto layers = build_layers(which, seed, numLeaves, m, n, k, key);
  uint8_t root[32]; { std::vector<CV> tmp = layers[0]; pearlhash_b3::merge_root((uint8_t*)tmp.data(), numLeaves, key, root); }
  std::set<int> leafset;
  for (int row : rows) { int first = (int)((long)row * cols / 1024), last = (int)(((long)(row + 1) * cols - 1) / 1024); for (int i = first; i <= last; i++) leafset.insert(i); }
  std::vector<int> leaves(leafset.begin(), leafset.end());
  // MerkleProof: leaf_data (Vec<[u8;1024]>), leaf_indices, total_leaves, root, siblings
  put_u64le(out, leaves.size());
  uint8_t chunk[1024];
  for (int li : leaves) { gen_leaf(which, seed, li, m, n, k, chunk); put_u64le(out, 1024); put_bytes(out, chunk, 1024); }
  put_u64le(out, leaves.size()); for (int li : leaves) put_u64le(out, (uint64_t)li);
  put_u64le(out, (uint64_t)numLeaves);
  put_bytes(out, root, 32);
  // siblings: walk up, emit each node whose sibling is not itself in the active set
  std::set<int> cur(leafset); int levelLen = numLeaves, level = 0; std::vector<CV> sib;
  while (levelLen > 1 && !cur.empty()) {
    auto& nodes = layers[level];
    for (int i : cur) { if (i % 2 == 1) { if (!cur.count(i - 1)) sib.push_back(nodes[i - 1]); }
                        else if (!cur.count(i + 1) && i + 1 < levelLen) sib.push_back(nodes[i + 1]); }
    std::set<int> nx; for (int i : cur) nx.insert(i >> 1); cur.swap(nx);
    levelLen = (levelLen + 1) / 2; level++;
  }
  put_u64le(out, sib.size()); for (auto& s : sib) put_bytes(out, s.data(), 32);
  // MatrixMerkleProof tail: row_indices
  put_u64le(out, rows.size()); for (int r : rows) put_u64le(out, (uint64_t)r);
}
static std::string base64(const std::vector<uint8_t>& d) {
  static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o; for (size_t i = 0; i < d.size(); i += 3) {
    uint32_t x = d[i] << 16 | (i + 1 < d.size() ? d[i + 1] << 8 : 0) | (i + 2 < d.size() ? d[i + 2] : 0);
    o += T[(x >> 18) & 63]; o += T[(x >> 12) & 63];
    o += (i + 1 < d.size()) ? T[(x >> 6) & 63] : '='; o += (i + 2 < d.size()) ? T[x & 63] : '=';
  }
  return o;
}
// Full PlainProof (base64) for a winning tile, from the seed + job key (no noised matrices needed).
static std::string build_plain_proof(uint32_t seed, int m, int n, int k, int rank, const uint8_t* key, int row, int col) {
  std::vector<int> aRows, btRows; for (int r = 0; r < 16; r++) { aRows.push_back(row + r); btRows.push_back(col + r); }
  std::vector<uint8_t> buf;
  put_u64le(buf, (uint64_t)m); put_u64le(buf, (uint64_t)n); put_u64le(buf, (uint64_t)k); put_u64le(buf, (uint64_t)rank);
  ser_matrix_proof(buf, 'A', seed, m, n, k, key, k, aRows);
  ser_matrix_proof(buf, 'B', seed, m, n, k, key, k, btRows);
  buf.push_back(0x00);   // moe: Option::None
  return base64(buf);
}
template <typename T>
static T* alloc_buffer(sycl::queue& q, size_t count) {
  return q.get_device().is_cpu() ? sycl::malloc_shared<T>(count, q)
                                 : sycl::malloc_device<T>(count, q);
}
static Buffers alloc_buffers(sycl::queue& q, int m, int n, int k, int rank,
                             bool hip_jit = false, bool cuda_jit = false) {
  Buffers b{};
  b.EAL=alloc_buffer<int8_t>(q,m*rank); b.EBR=alloc_buffer<int8_t>(q,rank*n);
  b.EBRt=alloc_buffer<int8_t>(q,n*rank);
  b.Ap=alloc_buffer<int8_t>(q,m*k); b.Bp=alloc_buffer<int8_t>(q,k*n);
#if defined(MOM_SYCL_HAS_HIP)
  // HIPRTC WMMA consumes an architecture-native fragment layout. Keep it separate from the
  // portable SYCL layout so a failed native launch can immediately fall back without regenerating.
  if (hip_jit) {
    b.ApWmma=alloc_buffer<int8_t>(q,m*k);
    b.BpWmma=alloc_buffer<int8_t>(q,k*n);
  }
#endif
  b.EARp1=alloc_buffer<int32_t>(q,k); b.EARp2=alloc_buffer<int32_t>(q,k);
  b.EBLq1=alloc_buffer<int32_t>(q,k); b.EBLq2=alloc_buffer<int32_t>(q,k);
  b.cA=alloc_buffer<uint8_t>(q,32); b.cB=alloc_buffer<uint8_t>(q,32);
  b.key=alloc_buffer<uint8_t>(q,32); b.target=alloc_buffer<uint8_t>(q,32);
  b.CVA=alloc_buffer<uint8_t>(q,(2*((m*k+1023)/1024)+2)*32);
  b.CVB=alloc_buffer<uint8_t>(q,(2*((n*k+1023)/1024)+2)*32);
  // Separating jackpot BLAKE3 from the matrix loop restores occupancy. Portable paths retain a
  // conservative 256 MiB cap; the explicitly selected CUDA source-JIT path can use a larger
  // transcript to amortize the common seed/root/noise preparation without affecting other workers.
  const uint64_t transcript_bytes = static_cast<uint64_t>(m / 16) * (n / 16) * 16U * sizeof(uint32_t);
  const uint64_t transcript_limit = cuda_jit ? 4ULL * 1024U * 1024U * 1024U
                                             : 256ULL * 1024U * 1024U;
  if (transcript_bytes <= transcript_limit)
    b.transcript=alloc_buffer<uint32_t>(q, static_cast<size_t>(transcript_bytes / sizeof(uint32_t)));
  b.result=sycl::malloc_shared<Result>(1,q); b.result->found=0;
  return b;
}

#ifndef PEARLHASH_STANDALONE
// ---- native mom entry point (DEV::PEARLHASH_GPU) ----
namespace {
// PlainProof carries M/N/K/rank, so the job can select a device-efficient profile without changing
// proof semantics. mom.js obtains defaults from GPU-COMPILERS.md and passes explicit config values.
// One persistent in-order queue + device buffer set per GPU; the seed search reuses them.
struct PearlHashState {
  sycl::queue queue;
  sycl::queue prep_queue;
  PearlHashHipSearch hip;
  PearlHashCudaSearch cuda;
  PearlHashAmdWmmaSearch amd_wmma;
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
      prep_queue(queue.get_context(), queue.get_device(),
                 sycl::property::queue::in_order{}) {}
  void free_buffers() {
    // Free every non-null member independently (sycl::free is no-op'd by the per-element `if (p)`),
    // so a partial-allocation failure where Ap is null but earlier members (EAL/EBR/EBRt/...) are
    // not still releases them instead of leaking. No early return on !b.Ap.
    queue.wait();
    prep_queue.wait();
    hip.release();
    cuda.release();
    for (void* p : {(void*)b.EAL,(void*)b.EBR,(void*)b.EBRt,(void*)b.Ap,(void*)b.Bp,
                    (void*)b.ApWmma,(void*)b.BpWmma,
                    (void*)b.EARp1,(void*)b.EARp2,(void*)b.EBLq1,(void*)b.EBLq2,
                    (void*)b.cA,(void*)b.cB,(void*)b.key,(void*)b.target,
                    (void*)b.CVA,(void*)b.CVB,(void*)b.transcript,(void*)b.result})
      if (p) sycl::free(p, queue);
    b = Buffers{};
  }
  void ensure(int m_, int n_, int k_, int rank_, bool hip_jit_, bool cuda_jit_) {
    if (b.Ap && m == m_ && n == n_ && k == k_ && rank == rank_ &&
        hip_jit_buffers == hip_jit_ && cuda_jit_buffers == cuda_jit_) return;
    free_buffers();
    m = m_; n = n_; k = k_; rank = rank_;
    hip_jit_buffers = hip_jit_;
    cuda_jit_buffers = cuda_jit_;
    b = alloc_buffers(queue, m, n, k, rank, hip_jit_buffers, cuda_jit_buffers);
    have_header = false;
  }
  ~PearlHashState() { sycl_cleanup_noexcept("pearlhash", [&] { free_buffers(); }); }
};
using PearlHashStateMap = std::map<std::string, std::unique_ptr<PearlHashState>>;
PearlHashStateMap& pearlhash_states() {
  static PearlHashStateMap* const states = new PearlHashStateMap;
  return *states;
}
std::mutex& pearlhash_states_mutex() {
  static std::mutex* const mutex = new std::mutex;
  return *mutex;
}
PearlHashState& pearlhash_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(pearlhash_states_mutex());
  auto& st = pearlhash_states()[dev_str];
  if (!st) st = std::make_unique<PearlHashState>(dev_str);
  return *st;
}
thread_local std::string g_pearlhash_proof;   // last built proof, fetched right after pearlhash() returns 1
// Winning tile captured by pearlhash() on a find; consumed by pearlhash_proof() to build the proof lazily.
thread_local struct PearlHashFound { uint32_t seed; int row, col, m, n, k, rank; uint8_t key[32]; bool valid; } g_pf{};
} // namespace

void pearlhash_cleanup_states() noexcept {
  try {
    std::lock_guard<std::mutex> lock(pearlhash_states_mutex());
#if defined(_WIN32)
    // Intel's standalone Windows CPU OpenCL ICD can fault after a successful PearlHash result while
    // DPC++ destroys this algorithm's shared-USM allocation/queue graph. The state registry itself
    // is deliberately heap-owned and has no process-exit destructor, so retain only CPU states and
    // let process teardown reclaim them. Discrete GPU states still require ordered cleanup (notably
    // AdaptiveCpp/CUDA, which otherwise reports cudaErrorCudartUnloading) and are erased normally.
    for (auto it = pearlhash_states().begin(); it != pearlhash_states().end();) {
      if (it->second->queue.get_device().is_cpu()) ++it;
      else it = pearlhash_states().erase(it);
    }
#else
    pearlhash_states().clear();
#endif
  } catch (...) {
    std::fprintf(stderr, "pearlhash: ordered SYCL cleanup failed\n");
  }
}

// One seeded NoisyGEMM attempt per call (seed = *pseed). Returns 1 on a winning tile, writes the
// winning seed back to *pseed, and stashes the base64 PlainProof for pearlhash_proof(). intensity = M.
int pearlhash(
  const unsigned job_ref, const uint32_t, const uint8_t* const input, const unsigned input_size,
  uint8_t* const, uint64_t* const pseed, const uint8_t* const target,
  const unsigned intensity, const bool is_test, const bool, const std::string& dev_str,
  const std::string& backend, const unsigned requested_n, const unsigned requested_k,
  const unsigned requested_rank
) {
  if (input_size < 76) throw std::string("Bad pearlhash input length (need 76-byte header)");
  const int k = static_cast<int>(requested_k);
  const int rank = static_cast<int>(requested_rank);
  // M arrives as the worker's explicit primary tuning value; N can be selected independently. Native HIP
  // tests use the smallest square that activates the 32x32 cache-block traversal, so the
  // native-vs-SYCL checksum comparison covers production tile mapping without making CPU tests
  // expensive. Other implementations retain the compact 256x256 vector.
  const int test_edge = backend == "native" ? 2048 : 256;
  int m = is_test ? test_edge : ((int)intensity >= 128 ? (int)intensity : 131072);
  const int n = is_test ? m : static_cast<int>(requested_n);

  PearlHashState& st = pearlhash_state(dev_str);
  std::lock_guard<std::mutex> lock(st.mutex);
  const bool native_requested = backend == "auto" || backend == "native";
  std::string hip_reason, cuda_reason;
  const bool hip_jit = native_requested &&
                       PearlHashHipSearch::supports(st.queue, &hip_reason);
  const bool cuda_jit = native_requested &&
                        PearlHashCudaSearch::supports(st.queue, &cuda_reason);
  if (native_requested && !hip_jit && !cuda_jit && !st.native_notice) {
    st.native_notice = true;
    std::fprintf(stderr, "PearlHash native backend unavailable, using SYCL: %s\n",
                 (PearlHashCudaSearch::is_cuda(st.queue.get_device()) ? cuda_reason : hip_reason).c_str());
  }
  st.ensure(m, n, k, rank, hip_jit, cuda_jit);
  sycl::queue& q = st.queue;
  Buffers& b = st.b;

  if (!st.have_header || std::memcmp(st.header, input, 76) != 0) {  // (re)derive key on header change
    std::memcpy(st.header, input, 76);
    derive_key(input, k, rank, st.key); q.memcpy(b.key, st.key, 32); st.have_header = true;
  }
  (void)job_ref;
  uint8_t tgtLE[32];
  if (is_test) std::memset(tgtLE, 0xff, 32);                         // test: the first tile wins
  else for (int i = 0; i < 32; i++) tgtLE[i] = target[31 - i];       // core gives BE; kernel wants LE
  q.memcpy(b.target, tgtLE, 32);
  b.result->found = 0;
  b.result->chk = 0;   // whole-search checksum (filled only when MOM_PEARLHASH_CHK is set; see tile_mix)
  q.wait();

  const uint32_t attempt_seed = static_cast<uint32_t>(*pseed);
  PearlHashSearchBackend search_backend =
    attempt(q, b, attempt_seed, m, n, k, rank, backend, &st.hip, &st.cuda, &st.amd_wmma,
            &st.prep_queue, is_test);
  // Start after submission: a first call can include synchronous device-image/JIT setup, and feeding
  // that one-off cost into the wait EMA makes later completed GPU attempts sleep needlessly.
  const auto attempt_start = std::chrono::steady_clock::now();
  // The in-order queue's wait busy-spins a host core on both backends (CUDA's native sync and the
  // Intel GPU wait), and pearlhash waits once per attempt -- a long, very stable GEMM. So sleep through
  // most of the attempt (an EMA of recent attempt wall times) before waiting, leaving only a short
  // spinning tail; the result is read from shared USM afterwards so completion stays exact. The 90%
  // sleep stays under the attempt time, adding no latency. Mirrors the cn/gpu pre-read sleep.
  if (st.wait_ema_us > 2000.0) {
    const double elapsed_us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - attempt_start).count());
    const double sleep_us = st.wait_ema_us * 0.90 - elapsed_us;
    if (sleep_us > 0.0)
      std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(sleep_us)));
  }
  if (search_backend == PearlHashSearchBackend::hip_jit ||
      search_backend == PearlHashSearchBackend::cuda_jit) {
    try {
      if (search_backend == PearlHashSearchBackend::hip_jit)
        st.hip.wait(st.wait_ema_us);
      else
        st.cuda.wait(st.wait_ema_us);
    } catch (const std::string& error) {
      if (search_backend == PearlHashSearchBackend::hip_jit) st.hip.fail(error);
      else st.cuda.fail(error);
      b.result->found = 0;
      b.result->chk = 0;
      search_backend = attempt(q, b, attempt_seed, m, n, k, rank, "sycl-native",
                               nullptr, nullptr, &st.amd_wmma, nullptr, is_test);
      q.wait_and_throw();
    }
  } else q.wait_and_throw();
#if defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_HAS_CUDA)
  if (is_test && search_backend != PearlHashSearchBackend::sycl) {
    const uint32_t native_checksum = b.result->chk;
    b.result->found = 0;
    b.result->chk = 0;
    compute_ab(q, b, attempt_seed, m, n, k, rank, /*portable=*/true);
    search(q, b, attempt_seed, m, n, k, rank, /*dbg=*/true);
    q.wait();
    if (b.result->chk != native_checksum) {
      std::ostringstream message;
      message << "PearlHash native/SYCL checksum mismatch (native=0x" << std::hex
              << native_checksum << ", SYCL=0x" << b.result->chk << ')';
      throw message.str();
    }
  }
#endif
  {
    const double us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - attempt_start).count());
    st.wait_ema_us = st.wait_ema_us == 0.0 ? us : st.wait_ema_us * 0.8 + us * 0.2;
  }
  if (std::getenv("MOM_PEARLHASH_CHK"))   // cross-validation: same A'/B' -> identical chk on every search path
    std::fprintf(stderr, "PEARLHASH_CHK dev=%s seed=%u chk=%08x\n", dev_str.c_str(), (uint32_t)*pseed, b.result->chk);
  if (!b.result->found) return 0;
  *pseed = b.result->seed;
  if (is_test) return 1;                                             // test: core only needs "ok", no proof
  // Capture the winning tile; DEFER the heavy host PlainProof rebuild (regenerate A/Bt + the BLAKE3
  // Merkle tree, ~tens of ms at m=n=16384) to pearlhash_proof(), which core calls ONCE per pool job_id.
  // Building it here on every find would starve the DPAS search: at a low pool difficulty a tile is
  // found on most attempts, yet the pool credits only one share per job. Deferring keeps the search
  // running flat out (~6x faster live).
  g_pf.seed = b.result->seed; g_pf.row = (int)b.result->row; g_pf.col = (int)b.result->col;
  g_pf.m = m; g_pf.n = n; g_pf.k = k; g_pf.rank = rank; std::memcpy(g_pf.key, st.key, 32); g_pf.valid = true;
  return 1;
}
// Builds the PlainProof for the last winning tile captured by pearlhash() (lazy: called once per job_id
// by core, off the search hot path). Returns "" if no tile has been found yet.
const char* pearlhash_proof() {
  if (g_pf.valid)
    g_pearlhash_proof = build_plain_proof(g_pf.seed, g_pf.m, g_pf.n, g_pf.k, g_pf.rank, g_pf.key, g_pf.row, g_pf.col);
  return g_pearlhash_proof.c_str();
}
// GEMM MACs per attempt = m*n*k, mirroring pearlhash()'s shape selection. This is the work unit
// the pearlhash hashrate is reported in, so the core's display matches the "TH/s" GEMM benchmark.
uint64_t pearlhash_attempt_hashes(unsigned intensity, unsigned n, unsigned k) {
  const uint64_t m = intensity >= 128 ? intensity : 131072;
  return m * static_cast<uint64_t>(n) * static_cast<uint64_t>(k);
}
#endif  // !PEARLHASH_STANDALONE

#ifdef PEARLHASH_STANDALONE
#include <cstdio>
#include <chrono>
#include <poll.h>
#include <unistd.h>
static void parse_hex(const char* hex, uint8_t* out, int nbytes) {
  for (int i = 0; i < nbytes; i++) { unsigned x; sscanf(hex + 2 * i, "%2x", &x); out[i] = (uint8_t)x; }
}

// Persistent search server: stdin lines `job <headerHex(152)> <targetHexLE(64)>` set the
// current work; the GPU searches incrementing seeds continuously and prints
// `found <seed> <plainProofBase64>` on hits. (Standalone test/bench harness only.)
static int run_server(int k, int rank, int m, int n) {
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  fprintf(stderr, "pearlhash-server dev: %s  m=%d n=%d k=%d rank=%d\n", q.get_device().get_info<sycl::info::device::name>().c_str(), m, n, k, rank);
  Buffers b = alloc_buffers(q, m, n, k, rank);

  bool haveJob=false; uint32_t seed=1; std::string inbuf; uint8_t curKey[32]={0};
  char rd[8192];
  auto poll_stdin=[&](int timeout_ms){
    pollfd pfd{0, POLLIN, 0};
    if (poll(&pfd,1,timeout_ms)>0 && (pfd.revents&POLLIN)) { int nr=read(0,rd,sizeof(rd)); if(nr<=0) return false; inbuf.append(rd,nr); }
    size_t nl;
    while ((nl=inbuf.find('\n'))!=std::string::npos) {
      std::string line=inbuf.substr(0,nl); inbuf.erase(0,nl+1);
      if (line.rfind("job ",0)==0) {
        char hh[160], tt[80]; if (sscanf(line.c_str(),"job %159s %79s",hh,tt)==2) {
          uint8_t header[76], tgt[32];
          parse_hex(hh,header,76); derive_key(header,k,rank,curKey); parse_hex(tt,tgt,32);
          q.memcpy(b.key,curKey,32); q.memcpy(b.target,tgt,32); q.wait();
          haveJob=true; seed=1; b.result->found=0;
        }
      } else if (line=="quit") return false;
    }
    return true;
  };
  if (!poll_stdin(-1)) return 0;  // block for first job
  while (true) {
    if (!poll_stdin(0)) break;
    if (!haveJob) { if(!poll_stdin(100)) break; continue; }
    attempt(q, b, seed, m, n, k, rank); q.wait();
    if (b.result->found) {     // build the pool-ready PlainProof here; JS only relays it
      std::string proof = build_plain_proof(b.result->seed, m, n, k, rank, curKey, (int)b.result->row, (int)b.result->col);
      printf("found %u %s\n", b.result->seed, proof.c_str()); fflush(stdout); b.result->found=0;
    }
    seed++;
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 6 && std::string(argv[1]) == "server")
    return run_server(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (std::string(argv[1]) == "proof") {   // proof <hdr> k rank m n seed row col -> base64 (host, no GPU)
    int k = atoi(argv[3]), rank = atoi(argv[4]), m = atoi(argv[5]), n = atoi(argv[6]);
    uint8_t header[76], key[32];
    parse_hex(argv[2], header, 76); derive_key(header, k, rank, key);
    std::printf("%s\n", build_plain_proof((uint32_t)strtoul(argv[7], nullptr, 10), m, n, k, rank, key, atoi(argv[8]), atoi(argv[9])).c_str());
    return 0;
  }
  const char* headerHex = argv[1];
  int k = atoi(argv[2]), rank = atoi(argv[3]), m = atoi(argv[4]), n = atoi(argv[5]);
  uint32_t seedBase = (uint32_t)strtoul(argv[6], nullptr, 10);
  int count = atoi(argv[7]);
  const char* tgtHexLE = argv[8];

  uint8_t header[76], key[32], target[32];
  parse_hex(headerHex, header, 76); derive_key(header, k, rank, key);
  parse_hex(tgtHexLE, target, 32);

  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  std::printf("dev: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  Buffers b = alloc_buffers(q, m, n, k, rank);
  q.memcpy(b.key, key, 32); q.memcpy(b.target, target, 32); q.wait();

  attempt(q, b, seedBase, m, n, k, rank); q.wait();           // warm up JIT
  bool warmHit = b.result->found; b.result->found = 0;

  auto t0 = std::chrono::steady_clock::now();
  int found = 0, done = 0; uint32_t startSeed = warmHit ? seedBase : seedBase + 1;
  for (int i = 0; i < count; i++, done++) { uint32_t seed = startSeed + i; attempt(q, b, seed, m, n, k, rank); q.wait();
    if (b.result->found) { std::printf("FOUND seed=%u row=%u col=%u (attempt %d)\n", b.result->seed, b.result->row, b.result->col, i); found = 1; break; } }
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  double aps = done / (secs > 0 ? secs : 1);
  double macPerAttempt = (double)m*n*k + (double)m*k + (double)k*n;
  std::printf("%s  %.3fs  %.1f attempts/s  ~%.4g MAC/s  (%.4g TH/s-equiv)\n",
              found ? "OK" : "no-hit", secs, aps, aps * macPerAttempt, aps * macPerAttempt / 1e12);
  return 0;
}
#endif
#endif  // !MOM_PEARLHASH_ESIMD_TU
