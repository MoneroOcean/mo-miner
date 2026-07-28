namespace mom_pearlhash {

using pearlhash_b3::b3;

// --- IGC multi-accumulator codegen workaround -------------------------------
// Holding several joint_matrix accumulator fragments live in a C-array and
// iterating over that array with a plain `for` / `#pragma unroll` miscompiles on
// the Intel oneAPI 2026.0 stack for Battlemage/Xe2 (Arc B580): the program JITs
// cleanly but aborts at runtime with UR_RESULT_ERROR_DEVICE_LOST. The defect is
//   https://github.com/intel/llvm/issues/21409
// ("the pragma unroll is not doing what is expected" over an array of matrix
// fragments). It is gated by the host icpx -O2 SPIR-V shape (not the IGC JIT
// level), is non-monotonic in fragment count (NTILE 2/3 crash, 1/4/6/8 ran), and
// still reproduces on nightly-2026-06-12. A standalone reproducer + analysis
// lives in ~/bug_report3. The fix from #21409 is to replace the loop with a
// C++17 fold-expression manual unroll, which generates correct code at every
// NTILE; MU<N>(f) below is exactly that and is used for every loop that touches
// the accumulator-fragment array. Do NOT turn these back into a `for` loop.
template <class T, T... I, class F>
static inline void mu_impl(std::integer_sequence<T, I...>, F&& f) { (f(std::integral_constant<T, I>{}), ...); }
template <int N, class F>
static inline void MU(F&& f) { mu_impl(std::make_integer_sequence<int, N>{}, static_cast<F&&>(f)); }

// counter-based RNG (lowbias32) -- the deterministic A/B matrix generator
static inline int8_t gv(uint32_t seed, uint32_t idx) {
  uint32_t x = seed + idx * 0x9e3779b9u;
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return (int8_t)((x & 127) - 64);
}
static inline uint32_t mulhi(uint32_t a, uint32_t b) { return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32); }
static inline uint32_t rotl(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
static inline uint32_t rd_u32le(const uint8_t* p) { return pearlhash_b3::le32(p); }
// Sparse-noise index pair for one k-column: first index `f` in [0,rank), second `se != f` (a
// permutation step within the rank). Same mapping for both the A and B noise tables.
static inline void perm_pair(uint32_t u, int rank, int32_t& f, int32_t& se) {
  uint32_t a = u & (rank - 1); f = (int32_t)a; se = (int32_t)(a ^ (1 + mulhi(rank - 1, u)));
}
static inline void mk_seed(uint8_t out[32], char c0) { const char* s = c0 == 'A' ? "A_tensor" : "B_tensor"; for (int i = 0; i < 32; i++) out[i] = i < 8 ? (uint8_t)s[i] : 0; }
static inline void dev_randHash(int index, const uint8_t* seed32, const uint8_t* key32, int prepend, uint8_t out[32]) {
  uint8_t mb[64]{};   // (1+index) LE at word `prepend`, seed32 at bytes 32..63, rest zero
  uint32_t v = (uint32_t)(1 + index);
  mb[prepend*4]=(uint8_t)v; mb[prepend*4+1]=(uint8_t)(v>>8); mb[prepend*4+2]=(uint8_t)(v>>16); mb[prepend*4+3]=(uint8_t)(v>>24);
  for (int i = 0; i < 32; i++) mb[32+i] = seed32[i];
  b3(mb, 64, key32, out);
}

struct Result { int found; uint32_t seed; uint32_t row; uint32_t col; uint32_t chk; };

struct Buffers {
  int8_t *EAL, *EBR, *EBRt, *Ap, *Bp;       // A'/B' (noised, search inputs); A/B regenerated from RNG
  int8_t *ApWmma, *BpWmma;
  int32_t *EARp1, *EARp2, *EBLq1, *EBLq2;
  uint8_t *cA, *cB, *key, *target, *CVA, *CVB;
  uint32_t* transcript;
  Result* result;
};

template <bool DirectLayout>
static void k_noise_ebr(sycl::queue& q, const Buffers& B, int n, int rank, int blocks) {
  q.parallel_for(sycl::range<1>(blocks), [=](sycl::id<1> id) {
    const int i = (int)id[0];
    uint8_t sd[32]; mk_seed(sd, 'B');
    uint8_t h[32]; dev_randHash(i, sd, B.cB, 0, h);
    const int total = n * rank;
    for (int j = 0; j < 32; ++j) {
      const int o = i * 32 + j;
      if (o >= total) break;
      const int8_t value = (int8_t)((h[j] & 63) - 32);
      if constexpr (DirectLayout) B.EBR[(o % rank) * n + o / rank] = value;
      else B.EBRt[o] = value;
    }
  });
}

// Parallel BLAKE3 Merkle reduction of `nc` contiguous chunk CVs in `cvs` (uint8 buffer sized
// >= 2*nc*32). For power-of-two nc the largest-power-of-two split degenerates to a balanced
// binary tree, so a pairwise level reduction is bit-identical to merge_root(). Each level reads
// the previous level's CV region and writes a fresh, non-overlapping region (no in-place race);
// the final parent carries the ROOT flag. Returns the CV-unit offset of the root. Falls back to
// nothing here — caller must guarantee nc is a power of two for this path.
static int reduce_tree_pow2(sycl::queue& q, uint8_t* cvs, const uint8_t* key, int nc) {
  int rd = 0, wr = nc, cnt = nc;
  while (cnt > 1) {
    int half = cnt / 2; bool isRoot = (half == 1); int rd0 = rd, wr0 = wr;
    q.parallel_for(sycl::range<1>(half), [=](sycl::id<1> id) { int i = (int)id[0];
      pearlhash_b3::parent_cv(cvs + (size_t)(rd0 + 2 * i) * 32, cvs + (size_t)(rd0 + 2 * i + 1) * 32, key, isRoot, cvs + (size_t)(wr0 + i) * 32); });
    rd = wr; wr += half; cnt = half;
  }
  return rd;
}
static void k_roots(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k) {
  auto B = bb;
  const int lenA = m * k, lenB = n * k; const uint32_t tot = (uint32_t)(m * k);
  const int ncA = (lenA + 1023) / 1024, ncB = (lenB + 1023) / 1024;
  // A bytes are regenerated on the fly (A[idx]=gv(seed,idx)) -- no A buffer materialized.
  q.parallel_for(sycl::range<1>(ncA), [=](sycl::id<1> id) { int ci = (int)id[0]; int off = ci * 1024; int len = lenA - off < 1024 ? lenA - off : 1024;
    uint8_t buf[1024];
    for (int t = 0; t < len; t++) buf[t] = (uint8_t)gv(seed, (uint32_t)(off + t));
    pearlhash_b3::chunk_cv(buf, (uint32_t)len, (uint64_t)ci, B.key, B.CVA + ci * 32); });
  // B' root is over Bt (n x k); each 1024-byte chunk of the Bt byte-stream is regenerated straight
  // from B's RNG (B[i]=gv(seed,tot+i)) so neither B nor a transposed Bt buffer exists. Bt byte
  // b -> B[(b%k)*n + b/k].
  q.parallel_for(sycl::range<1>(ncB), [=](sycl::id<1> id) { int ci = (int)id[0]; int off = ci * 1024; int len = lenB - off < 1024 ? lenB - off : 1024;
    uint8_t buf[1024];
    int j = off / k, r = off % k;        // Bt byte (off+t) -> column j, depth r of B; advance incrementally
    for (int t = 0; t < len; t++) { buf[t] = (uint8_t)gv(seed, tot + (uint32_t)(r * n + j)); if (++r == k) { r = 0; j++; } }
    pearlhash_b3::chunk_cv(buf, (uint32_t)len, (uint64_t)ci, B.key, B.CVB + ci * 32); });
  // Both branches finish the same commitment chain (cB = B3(key||rootB), cA = B3(cB||rootA)); they
  // differ only in how each tree's root is obtained (parallel pairwise reduce vs serial merge_root).
  const bool pow2 = ((ncA & (ncA - 1)) == 0) && ((ncB & (ncB - 1)) == 0);
  if (pow2) {
    int offA = reduce_tree_pow2(q, B.CVA, B.key, ncA);
    int offB = reduce_tree_pow2(q, B.CVB, B.key, ncB);
    q.single_task([=]() {
      uint8_t buf[64];
      for (int i = 0; i < 32; i++) { buf[i] = B.key[i]; buf[32 + i] = B.CVB[(size_t)offB * 32 + i]; } b3(buf, 64, nullptr, B.cB);
      for (int i = 0; i < 32; i++) { buf[i] = B.cB[i]; buf[32 + i] = B.CVA[(size_t)offA * 32 + i]; } b3(buf, 64, nullptr, B.cA);
    });
  } else {
    q.single_task([=]() {
      uint8_t rootA[32], rootB[32], buf[64];
      pearlhash_b3::merge_root(B.CVA, ncA, B.key, rootA);
      pearlhash_b3::merge_root(B.CVB, ncB, B.key, rootB);
      for (int i = 0; i < 32; i++) { buf[i] = B.key[i]; buf[32 + i] = rootB[i]; } b3(buf, 64, nullptr, B.cB);
      for (int i = 0; i < 32; i++) { buf[i] = B.cB[i]; buf[32 + i] = rootA[i]; } b3(buf, 64, nullptr, B.cA);
    });
  }
}

static void k_noise(sycl::queue& q, const Buffers& bb, int m, int n, int k, int rank) {
  auto B = bb;
  const int dAL = (m * rank + 31) / 32, dBR = (n * rank + 31) / 32, dP = (k * 4 + 31) / 32;
  q.parallel_for(sycl::range<1>(dAL), [=](sycl::id<1> id) { int i = (int)id[0]; uint8_t sd[32]; mk_seed(sd, 'A'); uint8_t h[32]; dev_randHash(i, sd, B.cA, 0, h);
    int total = m * rank; for (int j = 0; j < 32; j++) { int o = i * 32 + j; if (o < total) B.EAL[o] = (int8_t)((h[j] & 63) - 32); } });
  const bool cpu = q.get_device().is_cpu();
  if (cpu) {
    // The Windows OpenCL CPU runtime faults in the otherwise-trivial transpose kernel. Its small
    // compatibility workload writes the final layout directly; GPUs retain coalesced generation
    // plus transpose, which is substantially faster for production matrix sizes.
    k_noise_ebr<true>(q, B, n, rank, dBR);
  } else {
    k_noise_ebr<false>(q, B, n, rank, dBR);
  }
  q.parallel_for(sycl::range<1>(dP), [=](sycl::id<1> id) { int i = (int)id[0]; uint8_t sd[32]; mk_seed(sd, 'A'); uint8_t h[32]; dev_randHash(i, sd, B.cA, 1, h);
    for (int kk = 0; kk < 8; kk++) { int idx = i * 8 + kk; if (idx >= k) break; perm_pair(rd_u32le(h + kk*4), rank, B.EARp1[idx], B.EARp2[idx]); } });
  q.parallel_for(sycl::range<1>(dP), [=](sycl::id<1> id) { int i = (int)id[0]; uint8_t sd[32]; mk_seed(sd, 'B'); uint8_t h[32]; dev_randHash(i, sd, B.cB, 1, h);
    for (int kk = 0; kk < 8; kk++) { int idx = i * 8 + kk; if (idx >= k) break; perm_pair(rd_u32le(h + kk*4), rank, B.EBLq1[idx], B.EBLq2[idx]); } });
  if (!cpu) {
    q.parallel_for(sycl::range<1>((size_t)rank * n), [=](sycl::id<1> id) {
      const int idx = (int)id[0], r = idx / n, c = idx % n;
      B.EBR[idx] = B.EBRt[c * rank + r];
    });
  }
}

#if defined(_WIN32)
// Intel's standalone Windows CPU OpenCL runtime can access-violate while JITing/executing Pearl's
// otherwise portable commitment/noise kernels on older x86 CPUs. The final matrix preparation and
// search already use shared USM on the host for this device class; complete that compatibility path
// by producing the same roots and sparse-noise tables here. GPU paths and Linux CPU kernel coverage
// remain unchanged. CPU Pearl mining is not a performance target, while deterministic archive
// validation across supported Windows CPUs is valuable.
static void prepare_roots_cpu(
    const Buffers& B, uint32_t seed, int m, int n, int k) {
  const int lenA = m * k, lenB = n * k;
  const int ncA = (lenA + 1023) / 1024, ncB = (lenB + 1023) / 1024;
  const uint32_t tot = static_cast<uint32_t>(m * k);
  uint8_t chunk[1024];
  for (int ci = 0; ci < ncA; ++ci) {
    const int off = ci * 1024, len = std::min(1024, lenA - off);
    for (int t = 0; t < len; ++t)
      chunk[t] = static_cast<uint8_t>(gv(seed, static_cast<uint32_t>(off + t)));
    pearlhash_b3::chunk_cv(chunk, static_cast<uint32_t>(len), static_cast<uint64_t>(ci),
                          B.key, B.CVA + static_cast<size_t>(ci) * 32);
  }
  for (int ci = 0; ci < ncB; ++ci) {
    const int off = ci * 1024, len = std::min(1024, lenB - off);
    int j = off / k, r = off % k;
    for (int t = 0; t < len; ++t) {
      chunk[t] = static_cast<uint8_t>(gv(seed, tot + static_cast<uint32_t>(r * n + j)));
      if (++r == k) { r = 0; ++j; }
    }
    pearlhash_b3::chunk_cv(chunk, static_cast<uint32_t>(len), static_cast<uint64_t>(ci),
                          B.key, B.CVB + static_cast<size_t>(ci) * 32);
  }
  uint8_t rootA[32], rootB[32], input[64];
  pearlhash_b3::merge_root(B.CVA, ncA, B.key, rootA);
  pearlhash_b3::merge_root(B.CVB, ncB, B.key, rootB);
  for (int i = 0; i < 32; ++i) { input[i] = B.key[i]; input[32 + i] = rootB[i]; }
  b3(input, 64, nullptr, B.cB);
  for (int i = 0; i < 32; ++i) { input[i] = B.cB[i]; input[32 + i] = rootA[i]; }
  b3(input, 64, nullptr, B.cA);
}

static void prepare_noise_cpu(const Buffers& B, int m, int n, int k, int rank) {
  const int dAL = (m * rank + 31) / 32, dBR = (n * rank + 31) / 32;
  for (int i = 0; i < dAL; ++i) {
    uint8_t seed[32], hash[32]; mk_seed(seed, 'A'); dev_randHash(i, seed, B.cA, 0, hash);
    for (int j = 0; j < 32; ++j) {
      const int out = i * 32 + j;
      if (out < m * rank) B.EAL[out] = static_cast<int8_t>((hash[j] & 63) - 32);
    }
  }
  for (int i = 0; i < dBR; ++i) {
    uint8_t seed[32], hash[32]; mk_seed(seed, 'B'); dev_randHash(i, seed, B.cB, 0, hash);
    for (int j = 0; j < 32; ++j) {
      const int out = i * 32 + j;
      if (out < n * rank)
        B.EBR[static_cast<size_t>(out % rank) * n + out / rank] =
          static_cast<int8_t>((hash[j] & 63) - 32);
    }
  }
  const int blocks = (k + 7) / 8;
  for (int i = 0; i < blocks; ++i) {
    uint8_t seedA[32], seedB[32], hashA[32], hashB[32];
    mk_seed(seedA, 'A'); mk_seed(seedB, 'B');
    dev_randHash(i, seedA, B.cA, 1, hashA);
    dev_randHash(i, seedB, B.cB, 1, hashB);
    for (int lane = 0; lane < 8; ++lane) {
      const int index = i * 8 + lane;
      if (index >= k) break;
      perm_pair(rd_u32le(hashA + lane * 4), rank, B.EARp1[index], B.EARp2[index]);
      perm_pair(rd_u32le(hashB + lane * 4), rank, B.EBLq1[index], B.EBLq2[index]);
    }
  }
}
#endif

static sycl::event compute_ab(sycl::queue& q, const Buffers& bb, uint32_t seed,
           int m, int n, int k, int rank,
           bool portable) {
  auto B = bb; const uint32_t tot = (uint32_t)(m * k);
  // Explicit capture list (not [=]): capturing the extra `portable` bool under [=] makes the host
  // (icx) and device (clang) compilers disagree on the lambda layout in the dual-compiler combined
  // build ("Unexpected kernel lambda size"). Listing the captures explicitly keeps them in sync.
  int8_t* const eal = B.EAL;
  int8_t* const ap = B.Ap;
  int32_t* const earp1 = B.EARp1;
  int32_t* const earp2 = B.EARp2;
  q.parallel_for(sycl::range<1>((size_t)m * k), [eal, ap, earp1, earp2, seed, k, rank, portable](sycl::id<1> id) { int idx = (int)id[0], i = idx / k, c = idx % k;
    int acc = (int)eal[i * rank + earp1[c]] - (int)eal[i * rank + earp2[c]];
    int8_t v = (int8_t)(gv(seed, (uint32_t)idx) + acc);
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
    (void)portable;
    ap[idx] = v;   // row-major A' for the NVIDIA mma.sync search_cuda (nvptx device pass)
#else
    // The spir64 image carries BOTH searches; the host sets `portable` to match the one it will launch:
    //  - portable (the dp4a int8 GEMM search(), used by every OpenCL device -- AMD, the CPU device, and
    //    Intel-OpenCL): ROW-MAJOR A'[i*k+c], so each output row reads a contiguous int8 run over k that
    //    packs into the 4-wide dp4a operand.
    //  - else (ESIMD search_esimd): TILE-MAJOR -- each 8x32 dpas-A fragment is one contiguous 256-byte
    //    block_load (the fast Intel path on Level-Zero).
    if (portable) ap[idx] = v;
    else ap[((size_t)((i >> 3) * (k / 32) + (c >> 5)) * 256) + (i & 7) * 32 + (c & 31)] = v;
#endif
  });
  // B' layout matches the search variant that will run (explicit capture, same reason as above):
  //  - nvptx (__NVPTX__): ROW-MAJOR B'[r,j] for the joint_matrix B operand, else COLUMN-major for the
  //    default mma.sync B operand (coalesced 32-bit loads).
  //  - portable (dp4a search()): COLUMN-major B'[r,j] at j*k+r, so each output column reads a
  //    contiguous int8 run over k -- the dp4a counterpart to the row-major A'.
  //  - else (Intel XMX ESIMD): TILE-MAJOR VNNI, each 16-wide N-tile's k*16 block contiguous (stride 64),
  //    Bp[ (j/16)*k*16 + (r/4)*64 + (j%16)*4 + (r%4) ] = B'[r,j], coalesced sub-group read.
  int8_t* const ebr = B.EBR;
  int8_t* const bp = B.Bp;
  int32_t* const eblq1 = B.EBLq1;
  int32_t* const eblq2 = B.EBLq2;
  if (portable) {
    // B' is generated row-major (adjacent j values read adjacent EBR bytes) but the portable/WMMA
    // search consumes it column-major (adjacent r values). A direct bp[j*k+r] store makes adjacent
    // work-items write k bytes apart and reduced gfx12 throughput to ~1.5 GB/s. Transpose a 32x32
    // tile through 1056 bytes of local memory: both global reads and writes are contiguous.
    return q.submit([&](sycl::handler& h) {
      sycl::local_accessor<int8_t, 2> tile(sycl::range<2>(32, 33), h);
      const size_t groups_r = (static_cast<size_t>(k) + 31) / 32;
      const size_t groups_j = (static_cast<size_t>(n) + 31) / 32;
      h.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(groups_r * 8, groups_j * 32),
                          sycl::range<2>(8, 32)),
        [ebr, bp, eblq1, eblq2, seed, tot, k, n, tile](sycl::nd_item<2> item) {
          const int lr = static_cast<int>(item.get_local_id(0));
          const int lj = static_cast<int>(item.get_local_id(1));
          const int r0 = static_cast<int>(item.get_group(0)) * 32;
          const int j0 = static_cast<int>(item.get_group(1)) * 32;
#pragma unroll
          for (int d = 0; d < 32; d += 8) {
            const int r = r0 + lr + d, j = j0 + lj;
            if (r < k && j < n) {
              const int acc = static_cast<int>(ebr[eblq1[r] * n + j]) -
                              static_cast<int>(ebr[eblq2[r] * n + j]);
              tile[lr + d][lj] =
                static_cast<int8_t>(gv(seed, tot + static_cast<uint32_t>(r * n + j)) + acc);
            }
          }
          item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
          for (int d = 0; d < 32; d += 8) {
            const int j = j0 + lr + d, r = r0 + lj;
            if (j < n && r < k) bp[static_cast<size_t>(j) * k + r] = tile[lj][lr + d];
          }
        });
    });
  }
  return q.parallel_for(sycl::range<1>((size_t)k * n), [ebr, bp, eblq1, eblq2, seed, tot, k, n, portable](sycl::id<1> id) { int idx = (int)id[0], r = idx / n, j = idx % n;
    int acc = (int)ebr[eblq1[r] * n + j] - (int)ebr[eblq2[r] * n + j];
    int8_t v = (int8_t)(gv(seed, tot + (uint32_t)idx) + acc);
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
    (void)portable;
#endif
#if defined(__NVPTX__)
    bp[(size_t)j * k + r] = v;                 // col-major B'[r,j] for mma.sync B operand (coalesced loads)
#else
    (void)n;
    if (portable) bp[(size_t)j * k + r] = v;   // col-major B'[r,j] for the dp4a search()
    else bp[(size_t)(j / 16) * k * 16 + (size_t)(r / 4) * 64 + (size_t)(j % 16) * 4 + (r % 4)] = v;
#endif
  });
}

#if defined(MOM_SYCL_HAS_HIP)
struct PearlHashPrepEvents { sycl::event a, b; };

static PearlHashPrepEvents compute_ab_amd_wmma(
    sycl::queue& a_queue, sycl::queue& b_queue, const Buffers& b,
    uint32_t seed, int m, int n, int k, int rank) {
  const uint32_t tot = static_cast<uint32_t>(m * k);
  sycl::event a_event = a_queue.parallel_for(
    sycl::range<1>(static_cast<size_t>(m) * k / 8U),
    [eal=b.EAL, ap=b.ApWmma, earp1=b.EARp1, earp2=b.EARp2,
     seed, k, rank](sycl::id<1> id) {
      const size_t packed_chunk = id[0];
      const int lane = static_cast<int>(packed_chunk & 31U);
      const size_t fragment = packed_chunk >> 5;
      const size_t row_block = fragment / static_cast<size_t>(k / 16);
      const int k_block = static_cast<int>(
        fragment - row_block * static_cast<size_t>(k / 16));
      const size_t row = row_block * 16U + static_cast<size_t>(lane & 15);
      const int col = k_block * 16 + (lane >> 4) * 8;
      uint64_t packed = 0;
#pragma unroll
      for (int byte = 0; byte < 8; ++byte) {
        const int c = col + byte;
        const int acc = static_cast<int>(eal[row * rank + earp1[c]]) -
                        static_cast<int>(eal[row * rank + earp2[c]]);
        const int8_t value = static_cast<int8_t>(
          gv(seed, static_cast<uint32_t>(row * k + c)) + acc);
        packed |= static_cast<uint64_t>(static_cast<uint8_t>(value)) << (byte * 8);
      }
      reinterpret_cast<uint64_t*>(ap)[packed_chunk] = packed;
    });
  // Generate B in row-major 32x32 tiles (coalesced EBR reads), transpose in local memory, and
  // write the exact packed fragment bytes consumed by gfx12 WMMA. This avoids materializing and
  // rereading the 64 MiB portable B matrix on every nonce.
  sycl::event b_event = b_queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<int8_t, 2> tile(sycl::range<2>(32, 33), h);
    const size_t groups_r = (static_cast<size_t>(k) + 31) / 32;
    const size_t groups_j = (static_cast<size_t>(n) + 31) / 32;
    h.parallel_for(
      sycl::nd_range<2>(sycl::range<2>(groups_r * 8, groups_j * 32),
                        sycl::range<2>(8, 32)),
      [ebr=b.EBR, bp=b.BpWmma, eblq1=b.EBLq1, eblq2=b.EBLq2,
       seed, tot, k, n, tile](sycl::nd_item<2> item) {
        const int lr = static_cast<int>(item.get_local_id(0));
        const int lj = static_cast<int>(item.get_local_id(1));
        const int r0 = static_cast<int>(item.get_group(0)) * 32;
        const int j0 = static_cast<int>(item.get_group(1)) * 32;
#pragma unroll
        for (int d = 0; d < 32; d += 8) {
          const int r = r0 + lr + d, j = j0 + lj;
          if (r < k && j < n) {
            const int acc = static_cast<int>(ebr[eblq1[r] * n + j]) -
                            static_cast<int>(ebr[eblq2[r] * n + j]);
            tile[lr + d][lj] = static_cast<int8_t>(
              gv(seed, tot + static_cast<uint32_t>(r * n + j)) + acc);
          }
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int d = 0; d < 32; d += 8) {
          const int j = j0 + lr + d, r = r0 + lj;
          if (j < n && r < k) {
            const size_t fragment =
              (static_cast<size_t>(j / 16) * (k / 16) + r / 16) * 256U;
            const int lane = (j & 15) + ((r & 8) ? 16 : 0);
            bp[fragment + static_cast<size_t>(lane) * 8U + (r & 7)] =
              tile[lj][lr + d];
          }
        }
      });
  });
  return {a_event, b_event};
}
#endif

// The Intel Windows OpenCL CPU compiler faults in compute_ab's indirect-index kernels even though
// the same SPIR-V is correct on Linux and Intel GPUs. CPU allocations are shared USM, so prepare
// the two modest test/fallback matrices directly on the host and still exercise the expensive,
// algorithm-defining GEMM/search as SYCL. GPUs never take this path.
static void compute_ab_cpu(const Buffers& B, uint32_t seed, int m, int n, int k, int rank) {
  const uint32_t tot = (uint32_t)(m * k);
  for (int idx = 0; idx < m * k; ++idx) {
    const int i = idx / k, c = idx % k;
    const int acc = (int)B.EAL[i * rank + B.EARp1[c]] -
                    (int)B.EAL[i * rank + B.EARp2[c]];
    B.Ap[idx] = (int8_t)(gv(seed, (uint32_t)idx) + acc);
  }
  for (int idx = 0; idx < k * n; ++idx) {
    const int r = idx / n, j = idx % n;
    const int acc = (int)B.EBR[B.EBLq1[r] * n + j] -
                    (int)B.EBR[B.EBLq2[r] * n + j];
    B.Bp[(size_t)j * k + r] = (int8_t)(gv(seed, tot + (uint32_t)idx) + acc);
  }
}

// The portable ER x EC micro-tile of output cells each work-item owns raises the dp4a MAC/byte so
// each A'/B' int32 word fetched from L2 feeds ER*EC / (ER+EC) dot products instead of one). A 16x16 hash
// tile is computed by one work-group of 256/(ER*EC) work-items. Tunable for the GPU's register file.
constexpr int pearlhash_portable_er = 4;
constexpr int pearlhash_portable_ec = 4;
// Finalize one tile: serialize its 16-word transcript little-endian, keyed-BLAKE3 it into the
// jackpot, and test jackpot <= target as a little-endian u256. Shared by both search paths; the
// winner-claim atomic differs and stays in the caller (joint_matrix uses sycl::atomic_ref, ESIMD
// esimd::atomic_update). Scalar-only, so it compiles in both the SYCL and ESIMD kernel contexts.
static inline bool tile_wins(const uint32_t tr[16], const uint8_t* key, const uint8_t* target) {
  uint8_t tb[64];
  for (int w = 0; w < 16; w++) { tb[w*4]=tr[w]&0xff; tb[w*4+1]=(tr[w]>>8)&0xff; tb[w*4+2]=(tr[w]>>16)&0xff; tb[w*4+3]=(tr[w]>>24)&0xff; }
  uint8_t jp[32]; b3(tb, 64, key, jp);
  for (int i = 31; i >= 0; i--) { if (jp[i] < target[i]) return true; if (jp[i] > target[i]) return false; }
  return true;   // jackpot == target: still within bound
}
// Position-weighted mix of a tile's (row,col) + its 16-word transcript. When MOM_PEARLHASH_CHK is set every
// search path atomically sums tile_mix over ALL tiles into result->chk -- a whole-search checksum used to
// cross-validate the portable search() against search_esimd/search_cuda: identical A'/B' must give an
// identical chk. The sum is order-independent (any tile schedule) and the row/col weighting makes it
// sensitive to a tile mis-mapping (e.g. an A'/B' transpose). Scalar-only, so it works in every kernel.
[[maybe_unused]] static inline uint32_t tile_mix(uint32_t row, uint32_t col, const uint32_t tr[16]) {
  uint32_t h = row * 2654435761u + col * 2246822519u + 0x9e3779b9u;
  for (int w = 0; w < 16; w++) h = (h ^ tr[w]) * 16777619u + (uint32_t)w;
  return h;
}

// gfx12 exposes wave32 int8 WMMA as an AMDGPU compiler builtin before either SYCL implementation
// exposes it through joint_matrix. DPC++'s AMD device pass accepts that builtin directly inside a
// normal SYCL kernel, so keep this as a backend extension in SYCL source instead of compiling a
// second HIP kernel at runtime. The two shipped shapes are template specializations: K and Rank
// remain compile-time constants and the WMMA loop has the same code shape as the former native path.
