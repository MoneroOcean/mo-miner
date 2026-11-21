using pearlhash_b3::b3;

// Validate positive matrix dimensions and checked host-size multiplication before allocation.
static size_t checked_elements(const int rows, const int columns, const char* const name) {
  if (rows <= 0 || columns <= 0)
    throw std::string("Invalid pearlhash ") + name + " dimensions";
  const size_t row_count = static_cast<size_t>(rows);
  const size_t column_count = static_cast<size_t>(columns);
  if (row_count > std::numeric_limits<size_t>::max() / column_count)
    throw std::string("Invalid pearlhash ") + name + " dimensions";
  return row_count * column_count;
}

// Round a byte count to Merkle leaves while enforcing the tree's signed-index limit.
static size_t checked_chunk_count(const size_t bytes, const char* const name) {
  const size_t chunks = bytes / 1024 + (bytes % 1024 != 0);
  if (chunks > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::string("Invalid pearlhash ") + name + " chunk count";
  return chunks;
}

// --- IGC multi-accumulator codegen workaround -------------------------------
// Holding several joint_matrix accumulator fragments live in a C-array and
// iterating over that array with a plain `for` / `#pragma unroll` miscompiles on
// the Intel oneAPI 2026.0 stack for Battlemage/Xe2 (Arc B580): the program JITs
// cleanly but aborts at runtime with UR_RESULT_ERROR_DEVICE_LOST. The defect is
//   https://github.com/intel/llvm/issues/21409
// ("the pragma unroll is not doing what is expected" over an array of matrix
// fragments). It is gated by the host icpx -O2 SPIR-V shape (not the IGC JIT
// level), is non-monotonic in fragment count (NTILE 2/3 crash, 1/4/6/8 ran), and
// still reproduces on nightly-2026-06-12. The fix from #21409 is to replace the
// loop with a C++17 fold-expression manual unroll, which generates correct code at every
// NTILE; MU<N>(f) below is exactly that and is used for every loop that touches
// the accumulator-fragment array. Do NOT turn these back into a `for` loop.
template <class T, T... I, class F>
static inline void mu_impl(std::integer_sequence<T, I...>, F&& f) {
  (f(std::integral_constant<T, I>{}), ...);
}
template <int N, class F> static inline void MU(F&& f) {
  mu_impl(std::make_integer_sequence<int, N>{}, static_cast<F&&>(f));
}

// counter-based RNG (lowbias32) -- the deterministic A/B matrix generator
// Legal raw bytes 63/64 plus rank noise [-63,63] give nonnegative int8 operands [0,127].
// Native INT8 routes benefit from this data pattern without changing work or proof rules;
// false retains the reference stream for other routes. Roots and openings share this mode.
static inline int8_t gv(uint32_t seed, uint32_t idx, bool positive_inputs = false,
                        bool seed_prefix_inputs = false) {
  // Only the compact raw solver uses this legal stream. Legacy native inputs
  // remain the hashed 63/64 stream, rather than inheriting a new seed contract.
  if (positive_inputs && seed_prefix_inputs)
    return static_cast<int8_t>(idx < 32 ? 63 + ((seed >> idx) & 1u) : 64);
  uint32_t x = seed + idx * 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return positive_inputs ? static_cast<int8_t>(63 + (x & 1u))
                         : static_cast<int8_t>((x & 127u) - 64);
}
static inline uint32_t mulhi(uint32_t a, uint32_t b) {
  return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
}
static inline uint32_t rotl(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}
// Sparse-noise index pair for one k-column: first index `f` in [0,rank), second `se != f` (a
// permutation step within the rank). Same mapping for both the A and B noise tables.
static inline void perm_pair(uint32_t u, int rank, int32_t& f, int32_t& se) {
  uint32_t a = u & (rank - 1);
  f = (int32_t)a;
  se = (int32_t)(a ^ (1 + mulhi(rank - 1, u)));
}
// Construct the protocol's zero-padded A_tensor or B_tensor domain-separation seed.
static inline void mk_seed(uint8_t out[32], char c0) {
  const char* s = c0 == 'A' ? "A_tensor" : "B_tensor";
  for (int i = 0; i < 32; i++)
    out[i] = i < 8 ? (uint8_t)s[i] : 0;
}
// Derive one keyed noise block with the protocol counter position and seed layout.
static inline void dev_randHash(int index, const uint8_t* seed32, const uint8_t* key32, int prepend,
                                uint8_t out[32]) {
  uint8_t mb[64]{}; // (1+index) LE at word `prepend`, seed32 at bytes 32..63, rest zero
  uint32_t v = (uint32_t)(1 + index);
  mb[prepend * 4] = (uint8_t)v;
  mb[prepend * 4 + 1] = (uint8_t)(v >> 8);
  mb[prepend * 4 + 2] = (uint8_t)(v >> 16);
  mb[prepend * 4 + 3] = (uint8_t)(v >> 24);
  for (int i = 0; i < 32; i++)
    mb[32 + i] = seed32[i];
  b3(mb, 64, key32, out);
}

// Winning tiles kept per attempt. Each distinct tile is a separate pool share, so a search keeps
// counting after the first win; `found` may exceed the slots, and later winners are dropped.
// The JIT kernels mirror this layout exactly.
constexpr int kPearlhashWinners = 4;
struct Winner {
  uint32_t row;
  uint32_t col;
  uint8_t jackpot[32];
};
struct Result {
  int found;
  uint32_t seed;
  uint32_t chk;
  Winner winners[kPearlhashWinners];
};

struct Buffers {
  ProofAxis proof_rows = kSquareAxis, proof_cols = kSquareAxis;
  int8_t *EAL, *EBR, *EBRt, *Ap, *Bp; // A'/B' (noised, search inputs); A/B regenerated from RNG
  int32_t *EARp1, *EARp2, *EBLq1, *EBLq2;
  uint8_t *cA, *cB, *key, *target, *CVA, *CVB;
  uint32_t* transcript;
  int transcript_rows;
  Result* result;
  bool positive_inputs = false; // bound to this attempt and retained with winning proof leaves
  bool seed_prefix_inputs = false; // admitted compact route only; cleared on every attempt
  size_t compact_a_prefix_cvs = 0, compact_b_prefix_cvs = 0;
  size_t compact_a_root_offset = 0, compact_b_root_offset = 0;
  // Appended host-dispatch bit: only the admitted native CUDA pair route may retain B's
  // producer/commitment layout between attempts. All shared kernels leave it false.
  bool fixed_b_reuse = false;
  uint32_t fixed_b_seed = 0;
};

// The fields the commitment and noise producers read. Their kernels capture this, not the whole
// Buffers: every captured byte travels in each launch's argument payload, and the proof axes
// added to Buffers measurably slowed the A770's per-attempt kernels.
struct ProducerBuffers {
  int8_t *EAL, *EBR, *EBRt;
  int32_t *EARp1, *EARp2, *EBLq1, *EBLq2;
  uint8_t *cA, *cB, *key, *CVA, *CVB;
  bool positive_inputs, seed_prefix_inputs, fixed_b_reuse;
};
static ProducerBuffers producer_buffers(const Buffers& b) {
  return {b.EAL,   b.EBR, b.EBRt, b.EARp1, b.EARp2, b.EBLq1, b.EBLq2, b.cA,
          b.cB,    b.key, b.CVA,  b.CVB,   b.positive_inputs, b.seed_prefix_inputs,
          b.fixed_b_reuse};
}

// Generate B-side rank noise and write either the direct or transposed search layout.
template <bool DirectLayout>
static sycl::event k_noise_ebr(sycl::queue& q, const ProducerBuffers& B, int n, int rank, int blocks,
                               const sycl::event& roots) {
  return q.submit([&](sycl::handler& h) {
    h.depends_on(roots);
    h.parallel_for(sycl::range<1>(blocks), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
    const int i = (int)id[0];
    uint8_t sd[32];
    mk_seed(sd, 'B');
    uint8_t h[32];
    dev_randHash(i, sd, B.cB, 0, h);
    const int total = n * rank;
    for (int j = 0; j < 32; ++j) {
      const int o = i * 32 + j;
      if (o >= total)
        break;
      const int8_t value = (int8_t)((h[j] & 63) - 32);
      if constexpr (DirectLayout)
        B.EBR[(o % rank) * n + o / rank] = value;
      else
        B.EBRt[o] = value;
    }
    });
  });
}

// Keep the original leaves for cached PlainProof openings. Parent levels alternate between
// disjoint regions in the existing 2*nc-CV allocation; carrying an odd tail unchanged preserves
// BLAKE3's left-power-of-two tree. Requires nc > 0 and the state's in-order queue so each level
// finishes before the next reads it. Returns the CV-unit offset of the root.
static size_t reduce_tree(sycl::queue& q, uint8_t* cvs, const uint8_t* key, int nc,
                          bool retain_complete_levels = false,
                          size_t* retained_prefix_cvs = nullptr) {
  if (retain_complete_levels) {
    const size_t leaves = static_cast<size_t>(nc);
    if (nc <= 0 || leaves % 4096 != 0 ||
        leaves > (std::numeric_limits<size_t>::max() - 2) / 2)
      throw std::string("Invalid compact PearlHash Merkle leaf count");
    const size_t capacity = 2 * leaves + 2;
    size_t count = leaves, input = 0, cursor = leaves;
    // Preserve the complete, contiguous power-of-two prefix. The existing reducer handles the
    // residual top in free CV space once its odd count can no longer be halved exactly.
    while (count > 2 && count % 2 == 0) {
      const size_t parents = count / 2;
      const size_t input_offset = input, output_offset = cursor;
      q.parallel_for(sycl::range<1>(parents), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const size_t child = id[0] * 2;
        const uint8_t* left = cvs + (input_offset + child) * 32;
        uint8_t* out = cvs + (output_offset + id[0]) * 32;
        pearlhash_b3::parent_cv(left, left + 32, key, false, out);
      });
      input = cursor;
      cursor += parents;
      count = parents;
    }
    if (cursor > capacity || input > capacity || count > (capacity - input) / 2)
      throw std::string("Invalid compact PearlHash Merkle scratch bound");
    if (retained_prefix_cvs)
      *retained_prefix_cvs = cursor;
    return input + reduce_tree(q, cvs + input * 32, key, static_cast<int>(count));
  }
  const size_t first = static_cast<size_t>(nc);
  const size_t second = first + static_cast<size_t>(nc / 2 + nc % 2);
  size_t rd = 0, wr = first;
  int cnt = nc;
  while (cnt > 1) {
    const int parents = cnt / 2 + cnt % 2;
    const bool is_root = cnt == 2;
    const int count = cnt;
    const size_t rd0 = rd, wr0 = wr;
    q.parallel_for(sycl::range<1>(parents), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const size_t i = id[0], child = i * 2;
      const uint8_t* left = cvs + (rd0 + child) * 32;
      uint8_t* out = cvs + (wr0 + i) * 32;
      if (child + 1 < static_cast<size_t>(count))
        pearlhash_b3::parent_cv(left, left + 32, key, is_root, out);
      else
        for (int byte = 0; byte < 32; ++byte)
          out[byte] = left[byte];
    });
    rd = wr;
    wr = wr == first ? second : first;
    cnt = parents;
  }
  return rd;
}
// Commit to the attempt's raw A and Bt byte streams: hash every 1024-byte leaf in parallel,
// reduce both keyed Merkle trees, then derive cB/cA from the roots on the device. A fixed-B pair
// attempt keeps CVB/cB from its first attempt and recomputes only A. Returns the seed event.
[[maybe_unused]] static sycl::event k_roots(sycl::queue& q, Buffers& bb, uint32_t seed, int m,
                                            int n, int k) {
  const ProducerBuffers B = producer_buffers(bb);
  const bool reuse_b = B.fixed_b_reuse;
  const size_t lenA = checked_elements(m, k, "M*K roots");
  const size_t lenB = checked_elements(n, k, "N*K roots");
  const uint32_t tot = static_cast<uint32_t>(lenA);
  const size_t ncA = checked_chunk_count(lenA, "A root");
  const size_t ncB = checked_chunk_count(lenB, "B root");
  // A bytes are regenerated on the fly (A[idx]=gv(seed,idx)) -- no A buffer materialized.
  q.parallel_for(sycl::range<1>(ncA), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
    const size_t ci = id[0];
    const size_t off = ci * 1024;
    const size_t remaining = lenA - off;
    const size_t len = remaining < 1024 ? remaining : 1024;
    uint8_t buf[1024];
    for (size_t t = 0; t < len; t++)
      buf[t] = static_cast<uint8_t>(gv(seed, static_cast<uint32_t>(off + t),
                                        B.positive_inputs, B.seed_prefix_inputs));
    pearlhash_b3::chunk_cv(buf, static_cast<uint32_t>(len), static_cast<uint64_t>(ci), B.key,
                           B.CVA + ci * 32);
  });
  // B' root is over Bt (n x k); each 1024-byte chunk of the Bt byte-stream is regenerated straight
  // from B's RNG (B[i]=gv(seed,tot+i)) so neither B nor a transposed Bt buffer exists. Bt byte
  // b -> B[(b%k)*n + b/k]. A fixed-B pair attempt retains CVB/cB and skips this whole producer.
  if (!reuse_b) {
    q.parallel_for(sycl::range<1>(ncB), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const size_t ci = id[0];
      const size_t off = ci * 1024;
      const size_t remaining = lenB - off;
      const size_t len = remaining < 1024 ? remaining : 1024;
      uint8_t buf[1024];
      size_t j = off / static_cast<size_t>(k);
      size_t r = off % static_cast<size_t>(k); // Bt byte -> column j, depth r of B
      for (size_t t = 0; t < len; t++) {
        const size_t source_offset = r * static_cast<size_t>(n) + j;
        buf[t] = static_cast<uint8_t>(
            gv(seed, tot + static_cast<uint32_t>(source_offset),
               B.positive_inputs, B.seed_prefix_inputs));
        if (++r == static_cast<size_t>(k)) {
          r = 0;
          j++;
        }
      }
      pearlhash_b3::chunk_cv(buf, static_cast<uint32_t>(len), static_cast<uint64_t>(ci), B.key,
                             B.CVB + ci * 32);
    });
  }
  size_t prefix_a = 0, prefix_b = 0;
  const size_t offA = reduce_tree(q, B.CVA, B.key, static_cast<int>(ncA),
                                  B.seed_prefix_inputs, &prefix_a);
  const size_t offB = reuse_b ? 0 : reduce_tree(q, B.CVB, B.key, static_cast<int>(ncB),
                                                B.seed_prefix_inputs, &prefix_b);
  bb.compact_a_prefix_cvs = B.seed_prefix_inputs ? prefix_a : 0;
  bb.compact_b_prefix_cvs = B.seed_prefix_inputs && !reuse_b ? prefix_b : 0;
  bb.compact_a_root_offset = B.seed_prefix_inputs ? offA : 0;
  bb.compact_b_root_offset = B.seed_prefix_inputs && !reuse_b ? offB : 0;
  return q.single_task([=]() MOM_SYCL_KERNEL_ARGS_RESTRICT {
    uint8_t rootA[32], rootB[32];
    for (int i = 0; i < 32; ++i)
      rootA[i] = B.CVA[offA * 32 + i];
    if (reuse_b) {
      // cB remains the fixed seed's commitment; only cA depends on this attempt's A root.
      pearlhash_commitment_seed_a(B.cB, rootA, static_cast<uint32_t>(m), B.cA);
    } else {
      for (int i = 0; i < 32; ++i)
        rootB[i] = B.CVB[offB * 32 + i];
      pearlhash_commitment_seeds(B.key, rootA, rootB, static_cast<uint32_t>(m),
                                 static_cast<uint32_t>(n), B.cB, B.cA);
    }
  });
}

// Two independent producer streams return their final A/B events through one shared contract.
struct PearlHashEventPair {
  sycl::event a, b;
};

// Generate A and B noise on their owning queues. Passing the root event makes the split queue
// safe while retaining the same in-order single-queue behavior for portable backends.
[[maybe_unused]] static PearlHashEventPair k_noise(sycl::queue& a_queue, sycl::queue& b_queue,
                                                      const Buffers& bb, int m, int n,
                                                      int k, int rank, const sycl::event& roots) {
  const ProducerBuffers B = producer_buffers(bb);
  const bool reuse_b = B.fixed_b_reuse;
  const int dAL = (m * rank + 31) / 32, dBR = (n * rank + 31) / 32, dP = (k * 4 + 31) / 32;
  sycl::event a_noise = a_queue.submit([&](sycl::handler& h) {
    h.depends_on(roots);
    h.parallel_for(sycl::range<1>(dAL), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
    int i = (int)id[0];
    uint8_t sd[32];
    mk_seed(sd, 'A');
    uint8_t h[32];
    dev_randHash(i, sd, B.cA, 0, h);
    int total = m * rank;
    for (int j = 0; j < 32; j++) {
      int o = i * 32 + j;
      if (o < total)
        B.EAL[o] = (int8_t)((h[j] & 63) - 32);
    }
    });
  });
  const bool cpu = a_queue.get_device().is_cpu();
  sycl::event b_noise = a_noise;
  if (!reuse_b && cpu) {
    // The Windows OpenCL CPU runtime faults in the otherwise-trivial transpose kernel. Its small
    // compatibility workload writes the final layout directly; GPUs retain coalesced generation
    // plus transpose, which is substantially faster for production matrix sizes.
    b_noise = k_noise_ebr<true>(b_queue, B, n, rank, dBR, roots);
  } else if (!reuse_b) {
    b_noise = k_noise_ebr<false>(b_queue, B, n, rank, dBR, roots);
  }
  // The direct CPU layout was submitted above only to retain its compatibility path. GPU B noise
  // uses the transposed temporary followed by one device-side copy, both on b_queue.
  if (!reuse_b && !cpu) {
    b_noise = b_queue.submit([&](sycl::handler& h) {
      h.depends_on(b_noise);
      h.parallel_for(sycl::range<1>((size_t)rank * n), [=](sycl::id<1> id)
                                                           MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const int idx = (int)id[0], r = idx / n, c = idx % n;
        B.EBR[idx] = B.EBRt[c * rank + r];
      });
    });
  }
  sycl::event a_indices = a_queue.submit([&](sycl::handler& h) {
    h.depends_on(a_noise);
    h.parallel_for(sycl::range<1>(dP), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
    int i = (int)id[0];
    uint8_t sd[32];
    mk_seed(sd, 'A');
    uint8_t h[32];
    dev_randHash(i, sd, B.cA, 1, h);
    for (int kk = 0; kk < 8; kk++) {
      int idx = i * 8 + kk;
      if (idx >= k)
        break;
      perm_pair(load32_le_dev(h + kk * 4), rank, B.EARp1[idx], B.EARp2[idx]);
    }
    });
  });
  if (reuse_b)
    return {a_indices, a_indices};
  sycl::event b_indices = b_queue.submit([&](sycl::handler& h) {
    h.depends_on(b_noise);
    h.parallel_for(sycl::range<1>(dP), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      int i = (int)id[0];
      uint8_t sd[32];
      mk_seed(sd, 'B');
      uint8_t h[32];
      dev_randHash(i, sd, B.cB, 1, h);
      for (int kk = 0; kk < 8; kk++) {
        int idx = i * 8 + kk;
        if (idx >= k)
          break;
        perm_pair(load32_le_dev(h + kk * 4), rank, B.EBLq1[idx], B.EBLq2[idx]);
      }
    });
  });
  return {a_indices, b_indices};
}

// Host reference for GPU commitment-seed checks and the Windows CPU compatibility path.
static void prepare_roots_cpu(const Buffers& B, uint32_t seed, int m, int n, int k,
                              const uint32_t* b_seed_override = nullptr) {
  const size_t lenA = checked_elements(m, k, "M*K roots");
  const size_t lenB = checked_elements(n, k, "N*K roots");
  const size_t ncA = checked_chunk_count(lenA, "A root");
  const size_t ncB = checked_chunk_count(lenB, "B root");
  const uint32_t tot = static_cast<uint32_t>(lenA);
  const uint32_t b_seed = b_seed_override ? *b_seed_override : seed;
  uint8_t chunk[1024];
  for (size_t ci = 0; ci < ncA; ++ci) {
    const size_t off = ci * 1024;
    const size_t len = std::min(size_t{1024}, lenA - off);
    for (size_t t = 0; t < len; ++t)
      chunk[t] = static_cast<uint8_t>(gv(seed, static_cast<uint32_t>(off + t),
                                          B.positive_inputs, B.seed_prefix_inputs));
    pearlhash_b3::chunk_cv(chunk, static_cast<uint32_t>(len), static_cast<uint64_t>(ci), B.key,
                           B.CVA + ci * 32);
  }
  for (size_t ci = 0; ci < ncB; ++ci) {
    const size_t off = ci * 1024;
    const size_t len = std::min(size_t{1024}, lenB - off);
    size_t j = off / static_cast<size_t>(k), r = off % static_cast<size_t>(k);
    for (size_t t = 0; t < len; ++t) {
      const size_t source_offset = r * static_cast<size_t>(n) + j;
      chunk[t] = static_cast<uint8_t>(
          gv(b_seed, tot + static_cast<uint32_t>(source_offset),
             B.positive_inputs, B.seed_prefix_inputs));
      if (++r == static_cast<size_t>(k)) {
        r = 0;
        ++j;
      }
    }
    pearlhash_b3::chunk_cv(chunk, static_cast<uint32_t>(len), static_cast<uint64_t>(ci), B.key,
                           B.CVB + ci * 32);
  }
  uint8_t rootA[32], rootB[32];
  pearlhash_b3::merge_root(B.CVA, static_cast<int>(ncA), B.key, rootA);
  pearlhash_b3::merge_root(B.CVB, static_cast<int>(ncB), B.key, rootB);
  pearlhash_commitment_seeds(B.key, rootA, rootB, static_cast<uint32_t>(m),
                             static_cast<uint32_t>(n), B.cB, B.cA);
}

#if defined(_WIN32)
// Intel's Windows CPU OpenCL runtime can access-violate in commitment/noise kernels on older CPUs.
// Its shared-USM fallback uses host preparation; GPU mining keeps the device kernels unchanged.
static void prepare_noise_cpu(const Buffers& B, int m, int n, int k, int rank) {
  const int dAL = (m * rank + 31) / 32, dBR = (n * rank + 31) / 32;
  for (int i = 0; i < dAL; ++i) {
    uint8_t seed[32], hash[32];
    mk_seed(seed, 'A');
    dev_randHash(i, seed, B.cA, 0, hash);
    for (int j = 0; j < 32; ++j) {
      const int out = i * 32 + j;
      if (out < m * rank)
        B.EAL[out] = static_cast<int8_t>((hash[j] & 63) - 32);
    }
  }
  for (int i = 0; i < dBR; ++i) {
    uint8_t seed[32], hash[32];
    mk_seed(seed, 'B');
    dev_randHash(i, seed, B.cB, 0, hash);
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
    mk_seed(seedA, 'A');
    mk_seed(seedB, 'B');
    dev_randHash(i, seedA, B.cA, 1, hashA);
    dev_randHash(i, seedB, B.cB, 1, hashB);
    for (int lane = 0; lane < 8; ++lane) {
      const int index = i * 8 + lane;
      if (index >= k)
        break;
      perm_pair(load32_le_dev(hashA + lane * 4), rank, B.EARp1[index], B.EARp2[index]);
      perm_pair(load32_le_dev(hashB + lane * 4), rank, B.EBLq1[index], B.EBLq2[index]);
    }
  }
}
#endif

// Inverse of the CUTE SM80 K32 fragment map used by the unified rank-128 CUDA
// payload. Every logical 64x32 tile remains 2048 bytes and every destination
// expression is size_t so the full B operand may end exactly at 2^32 bytes.
template <bool IsA>
static inline size_t nvidia_vector128_offset(size_t row, size_t depth, int k) {
  const size_t lane = 4U * (row & 7U) + ((depth >> 2U) & 3U);
  const size_t word = IsA
      ? ((row & 63U) >> 4U) * 4U + ((depth >> 4U) & 1U) * 2U + ((row >> 3U) & 1U)
      : (((row & 63U) >> 3U) * 2U) + ((depth >> 4U) & 1U);
  return ((row / 64U) * static_cast<size_t>(k / 32) + depth / 32U) * 2048U +
         (word / 4U) * 512U + lane * 16U + (word % 4U) * 4U + (depth & 3U);
}

// Materialize the noised search operands A'[i,c] = A[i,c] + EAL[i,p1(c)] - EAL[i,p2(c)] (B' alike
// from EBR and the EBLq pairs) in the layout the selected search consumes: row-major for the
// portable dp4a search, the DPAS/DPASW packing on Intel, or the 128-bit packets of the CUDA
// source-JIT route. Values wrap to int8 exactly as the verifier computes them.
template <bool Dpasw = false, bool PackedCuda = false>
[[maybe_unused]] static sycl::event compute_ab(sycl::queue& q, const Buffers& bb, uint32_t seed,
                                                int m, int n, int k, int rank, bool portable,
                                                bool force_b_pack = false) {
  auto B = bb;
  const size_t a_elements = checked_elements(m, k, "M*K");
  const size_t b_elements = checked_elements(n, k, "N*K");
  const uint32_t tot = static_cast<uint32_t>(a_elements);
  const bool positive_inputs = B.positive_inputs;
  const uint32_t b_seed = B.fixed_b_reuse ? B.fixed_b_seed : seed;
  // Explicit capture list (not [=]): capturing the extra `portable` bool under [=] makes the host
  // (icx) and device (clang) compilers disagree on the lambda layout in the dual-compiler combined
  // build ("Unexpected kernel lambda size"). Listing the captures explicitly keeps them in sync.
  int8_t* const eal = B.EAL;
  int8_t* const ap = B.Ap;
  int32_t* const earp1 = B.EARp1;
  int32_t* const earp2 = B.EARp2;
  sycl::event a_event = q.parallel_for(sycl::range<1>(a_elements),
                                       [eal, ap, earp1, earp2, seed, k, rank,
                                               portable, positive_inputs](sycl::id<1> id)
      MOM_SYCL_KERNEL_ARGS_RESTRICT {
    const size_t idx = id[0], i = idx / static_cast<size_t>(k), c = idx % static_cast<size_t>(k);
    const int acc = static_cast<int>(eal[i * static_cast<size_t>(rank) + earp1[c]]) -
                    static_cast<int>(eal[i * static_cast<size_t>(rank) + earp2[c]]);
    const int8_t v = static_cast<int8_t>(
        gv(seed, static_cast<uint32_t>(idx), positive_inputs) + acc);
    // The source-JIT consumer requires the same packet on direct NVPTX and generic SSCP.
    // Layout is a consumer contract, not a preprocessor property of the producer's compiler.
    if constexpr (PackedCuda) {
      ap[nvidia_vector128_offset<true>(i, c, k)] = v;
      return;
    }
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
    (void)portable;
    ap[idx] = v;
#else
    // The spir64 image carries BOTH searches; the host sets `portable` to match the one it will launch:
    //  - portable (the dp4a int8 GEMM search(), used by every OpenCL device -- AMD, the CPU device, and
    //    Intel-OpenCL): ROW-MAJOR A'[i*k+c], so each output row reads a contiguous int8 run over k that
    //    packs into the 4-wide dp4a operand.
    //  - else (ESIMD search_esimd): TILE-MAJOR -- each 8x32 dpas-A fragment is one contiguous 256-byte
    //    block_load (the fast Intel path on Level-Zero).
    if (portable) {
      ap[idx] = v;
    } else {
      ap[((i >> 3) * static_cast<size_t>(k / 32) + (c >> 5)) * 256 + (i & 7) * 32 +
          (c & 31)] = v;
    }
#endif
  });
  // Native pair retries retain the packed B bytes.  The GPU test oracle temporarily regenerates
  // portable row-major B' for its scalar checksum; force_b_pack restores the packed layout before
  // the next native attempt while still using the retained B seed.
  if (B.fixed_b_reuse && PackedCuda && !force_b_pack)
    return a_event;
  // B' layout matches the search variant that will run (explicit capture, same reason as above):
  //  - native rank-128 CUDA: producer-packed K32 packets for direct vector loads;
  //    other nvptx paths retain COLUMN-major mma.sync B operands.
  //  - portable (dp4a search()): COLUMN-major B'[r,j] at j*k+r, so each output column reads a
  //    contiguous int8 run over k -- the dp4a counterpart to the row-major A'.
  //  - else (Intel XMX ESIMD): TILE-MAJOR VNNI, each 16-wide N-tile's k*16 block contiguous (stride 64),
  //    Bp[ (j/16)*k*16 + (r/4)*64 + (j%16)*4 + (r%4) ] = B'[r,j], coalesced sub-group read.
  int8_t* const ebr = B.EBR;
  int8_t* const bp = B.Bp;
  int32_t* const eblq1 = B.EBLq1;
  int32_t* const eblq2 = B.EBLq2;
  if constexpr (Dpasw) {
    // Generate one complete VNNI word per item: consecutive items write contiguous four-byte
    // words instead of scattering one byte across distant column banks. K is a multiple of 64.
    if (!portable) {
      return q.parallel_for(sycl::range<1>(b_elements / 4),
              [ebr, bp, eblq1, eblq2, b_seed, tot, k, n, positive_inputs](sycl::id<1> id)
              MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const size_t word = id[0];
        const size_t j = (word / (static_cast<size_t>(k) * 2)) * 8 + word % 8;
        const size_t r0 = ((word / 8) % (static_cast<size_t>(k) / 4)) * 4;
        sycl::vec<int8_t, 4> packed;
        MU<4>([&](auto part) {
          const size_t r = r0 + part;
          const int acc = static_cast<int>(ebr[static_cast<size_t>(eblq1[r]) * n + j]) -
                          static_cast<int>(ebr[static_cast<size_t>(eblq2[r]) * n + j]);
          packed[part] = static_cast<int8_t>(
              gv(b_seed, tot + static_cast<uint32_t>(r * n + j), positive_inputs) + acc);
        });
        packed.store(word,
            sycl::multi_ptr<int8_t, sycl::access::address_space::global_space>(bp));
      });
    }
  }
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
          sycl::nd_range<2>(sycl::range<2>(groups_r * 8, groups_j * 32), sycl::range<2>(8, 32)),
          [ebr, bp, eblq1, eblq2, b_seed, tot, k, n, positive_inputs, tile](sycl::nd_item<2> item)
              MOM_SYCL_KERNEL_ARGS_RESTRICT {
            const int lr = static_cast<int>(item.get_local_id(0));
            const int lj = static_cast<int>(item.get_local_id(1));
            const size_t r0 = item.get_group(0) * 32;
            const size_t j0 = item.get_group(1) * 32;
#pragma unroll
            for (int d = 0; d < 32; d += 8) {
              const size_t r = r0 + static_cast<size_t>(lr + d);
              const size_t j = j0 + static_cast<size_t>(lj);
              if (r < static_cast<size_t>(k) && j < static_cast<size_t>(n)) {
                const size_t first = static_cast<size_t>(eblq1[r]) * static_cast<size_t>(n) + j;
                const size_t second = static_cast<size_t>(eblq2[r]) * static_cast<size_t>(n) + j;
                const int acc = static_cast<int>(ebr[first]) - static_cast<int>(ebr[second]);
                tile[lr + d][lj] = static_cast<int8_t>(
                    gv(b_seed, tot + static_cast<uint32_t>(r * n + j), positive_inputs) + acc);
              }
            }
            sycl::group_barrier(item.get_group());
#pragma unroll
            for (int d = 0; d < 32; d += 8) {
              const size_t j = j0 + static_cast<size_t>(lr + d);
              const size_t r = r0 + static_cast<size_t>(lj);
              if (j < static_cast<size_t>(n) && r < static_cast<size_t>(k))
                bp[PackedCuda ? nvidia_vector128_offset<false>(j, r, k)
                              : j * static_cast<size_t>(k) + r] = tile[lj][lr + d];
            }
          });
    });
  }
  return q.parallel_for(sycl::range<1>(b_elements),
                        [ebr, bp, eblq1, eblq2, b_seed, tot, k, n, portable, positive_inputs]
                        (sycl::id<1> id)
      MOM_SYCL_KERNEL_ARGS_RESTRICT {
    const size_t idx = id[0], r = idx / static_cast<size_t>(n), j = idx % static_cast<size_t>(n);
    const size_t first = static_cast<size_t>(eblq1[r]) * static_cast<size_t>(n) + j;
    const size_t second = static_cast<size_t>(eblq2[r]) * static_cast<size_t>(n) + j;
    const int acc = static_cast<int>(ebr[first]) - static_cast<int>(ebr[second]);
    const int8_t v = static_cast<int8_t>(
        gv(b_seed, tot + static_cast<uint32_t>(idx), positive_inputs) + acc);
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
    (void)portable;
#endif
#if defined(__NVPTX__)
    bp[j * static_cast<size_t>(k) + r] = v; // col-major B'[r,j] for mma.sync B operand (coalesced loads)
#else
    (void)n;
    if (portable) {
      bp[j * static_cast<size_t>(k) + r] = v;   // col-major B'[r,j] for the dp4a search()
    } else {
      bp[(j / 16) * static_cast<size_t>(k) * 16 + (r / 4) * 64 + (j % 16) * 4 +
          (r % 4)] = v;
    }
#endif
  });
}

#if defined(MOM_SYCL_HAS_HIP)
// Paired K16 fragments give each wave64 lane two scalar WMMA operands in one 64-bit load.
// Row/depth are canonical matrix coordinates; both producers must use this same byte layout.
static inline size_t amd_wmma_offset(size_t row, size_t depth, int k) {
  const size_t fragment = (row / 16 * static_cast<size_t>(k / 32) + depth / 32) * 512U;
  const size_t lane = (row & 15) + (depth & 12) * 4U;
  return fragment + lane * 8U + (depth & 3) + ((depth & 16) >> 2);
}
// Select the packed layout only after native admission, reusing the ordinary A/B allocations.
// The caller joins both producer queues before search; portable fallback regenerates its layout.
static PearlHashEventPair compute_ab_amd_wmma(sycl::queue& a_queue, sycl::queue& b_queue,
                                               const Buffers& b, uint32_t seed, int m, int n, int k,
                                               int rank, const sycl::event& a_noise_ready,
                                               const sycl::event& b_noise_ready) {
  const uint32_t tot = static_cast<uint32_t>(checked_elements(m, k, "M*K"));
  const bool positive_inputs = b.positive_inputs;
  sycl::event a_event = a_queue.submit([&](sycl::handler& h) {
    h.depends_on(a_noise_ready);
    h.parallel_for(
      sycl::range<1>(static_cast<size_t>(m) * k / 4U),
      [eal = b.EAL, ap = b.Ap, earp1 = b.EARp1, earp2 = b.EARp2, seed, k,
       rank, positive_inputs](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const size_t word = id[0];
        const size_t row = word / static_cast<size_t>(k / 4);
        const int col = static_cast<int>(word % static_cast<size_t>(k / 4)) * 4;
        uint32_t packed = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
          const int c = col + byte;
          const int acc = static_cast<int>(eal[row * rank + earp1[c]]) -
                          static_cast<int>(eal[row * rank + earp2[c]]);
          const int8_t value = static_cast<int8_t>(
              gv(seed, static_cast<uint32_t>(row * k + c), positive_inputs) + acc);
          packed |= static_cast<uint32_t>(static_cast<uint8_t>(value)) << (byte * 8);
        }
        reinterpret_cast<uint32_t*>(ap)[amd_wmma_offset(row, col, k) / 4U] = packed;
      });
  });
  // Generate B in row-major 32x32 tiles (coalesced EBR reads), transpose in local memory, and
  // write the exact packed fragment bytes consumed by gfx12 WMMA. This avoids materializing and
  // rereading the 64 MiB portable B matrix on every nonce.
  sycl::event b_event = b_queue.submit([&](sycl::handler& h) {
    h.depends_on(b_noise_ready);
    sycl::local_accessor<int8_t, 2> tile(sycl::range<2>(32, 33), h);
    const size_t groups_r = (static_cast<size_t>(k) + 31) / 32;
    const size_t groups_j = (static_cast<size_t>(n) + 31) / 32;
    h.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(groups_r * 8, groups_j * 32), sycl::range<2>(8, 32)),
        [ebr = b.EBR, bp = b.Bp, eblq1 = b.EBLq1, eblq2 = b.EBLq2, seed, tot, k, n,
         positive_inputs, tile](sycl::nd_item<2> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const int lr = static_cast<int>(item.get_local_id(0));
          const int lj = static_cast<int>(item.get_local_id(1));
          const size_t r0 = item.get_group(0) * 32;
          const size_t j0 = item.get_group(1) * 32;
#pragma unroll
          for (int d = 0; d < 32; d += 8) {
            const size_t r = r0 + static_cast<size_t>(lr + d);
            const size_t j = j0 + static_cast<size_t>(lj);
            if (r < static_cast<size_t>(k) && j < static_cast<size_t>(n)) {
              const size_t first = static_cast<size_t>(eblq1[r]) * static_cast<size_t>(n) + j;
              const size_t second = static_cast<size_t>(eblq2[r]) * static_cast<size_t>(n) + j;
              const int acc = static_cast<int>(ebr[first]) - static_cast<int>(ebr[second]);
              tile[lr + d][lj] = static_cast<int8_t>(
                  gv(seed, tot + static_cast<uint32_t>(r * n + j), positive_inputs) + acc);
            }
          }
          item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
          for (int d = 0; d < 32; d += 8) {
            const size_t j = j0 + static_cast<size_t>(lr + d);
            const size_t r = r0 + static_cast<size_t>(lj);
            if (j < static_cast<size_t>(n) && r < static_cast<size_t>(k)) {
              bp[amd_wmma_offset(j, r, k)] = tile[lj][lr + d];
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
[[maybe_unused]] static void compute_ab_cpu(const Buffers& B, uint32_t seed, int m, int n, int k,
                                            int rank) {
  const uint32_t tot = static_cast<uint32_t>(checked_elements(m, k, "M*K"));
  for (int idx = 0; idx < m * k; ++idx) {
    const int i = idx / k, c = idx % k;
    const int acc = (int)B.EAL[i * rank + B.EARp1[c]] - (int)B.EAL[i * rank + B.EARp2[c]];
    B.Ap[idx] = (int8_t)(gv(seed, (uint32_t)idx, B.positive_inputs) + acc);
  }
  const size_t b_elements = checked_elements(n, k, "N*K");
  for (size_t idx = 0; idx < b_elements; ++idx) {
    const size_t r = idx / static_cast<size_t>(n), j = idx % static_cast<size_t>(n);
    const size_t first = static_cast<size_t>(B.EBLq1[r]) * static_cast<size_t>(n) + j;
    const size_t second = static_cast<size_t>(B.EBLq2[r]) * static_cast<size_t>(n) + j;
    const int acc = static_cast<int>(B.EBR[first]) - static_cast<int>(B.EBR[second]);
    B.Bp[j * static_cast<size_t>(k) + r] = static_cast<int8_t>(
        gv(seed, tot + static_cast<uint32_t>(idx), B.positive_inputs) + acc);
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
static inline bool tile_wins(const uint32_t tr[16], const uint8_t* key, const uint8_t* target,
                             uint8_t jackpot[32]) {
  pearlhash_b3::b3_keyed_64_words(tr, key, jackpot);
  for (int i = 31; i >= 0; i--) {
    if (jackpot[i] < target[i])
      return true;
    if (jackpot[i] > target[i])
      return false;
  }
  return true; // jackpot == target: still within bound
}
// Position-weighted mix of a tile's (row,col) + its 16-word transcript. When MOM_PEARLHASH_CHK is set every
// search path atomically sums tile_mix over ALL tiles into result->chk -- a whole-search checksum used to
// cross-validate the portable search() against search_esimd/search_cuda: identical A'/B' must give an
// identical chk. The sum is order-independent (any tile schedule) and the row/col weighting makes it
// sensitive to a tile mis-mapping (e.g. an A'/B' transpose). Scalar-only, so it works in every kernel.
[[maybe_unused]] static inline uint32_t tile_mix(uint32_t row, uint32_t col,
                                                 const uint32_t tr[16]) {
  uint32_t h = row * 2654435761u + col * 2246822519u + 0x9e3779b9u;
  for (int w = 0; w < 16; w++)
    h = (h ^ tr[w]) * 16777619u + (uint32_t)w;
  return h;
}

// gfx12 exposes wave32 int8 WMMA as an AMDGPU compiler builtin before either SYCL implementation
// exposes it through joint_matrix. DPC++'s AMD device pass accepts that builtin directly inside a
// normal SYCL kernel, so keep this as a backend extension in SYCL source instead of compiling a
// second HIP kernel at runtime. The two shipped shapes are template specializations: K and Rank
// remain compile-time constants and the WMMA loop has the same code shape as the former native path.
