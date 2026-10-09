// Scratch access and consensus stages; host-only operation fixtures exclude these kernels.
template <bool Native> inline uint64_t scratch_load(const uint64_t* p) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  if constexpr (Native) {
    uint64_t value;
    asm("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(p));
    return value;
  }
#else
  (void)Native;
#endif
  return *p;
}
// Store dependent scratch words with the cache policy paired with scratch_load.
template <bool Native> inline void scratch_store(uint64_t* p, uint64_t value) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  if constexpr (Native) {
    asm volatile("st.global.wt.u64 [%0], %1;" ::"l"(p), "l"(value));
    return;
  }
#else
  (void)Native;
#endif
  *p = value;
}

// Read one stage-3 scratch word through the backend-specific gather path.
template <bool Esimd> inline uint64_t stage3_load(const uint64_t* p) {
#ifdef MOM_XELISHASHV3_ESIMD
  if constexpr (Esimd) {
    constexpr auto properties =
        esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>,
                          esimd::cache_hint_L2<esimd::cache_hint::cached>, esimd::alignment<8>};
    return esimd::gather<uint64_t, 1, 1>(p, esimd::simd<uint64_t, 1>(0), properties)[0];
  } else
    return *p;
#else
  (void)Esimd;
  return *p;
#endif
}
// Write one stage-3 scratch word through the backend-specific scatter path.
template <bool Esimd> inline void stage3_store(uint64_t* p, uint64_t value) {
#ifdef MOM_XELISHASHV3_ESIMD
  if constexpr (Esimd) {
    constexpr auto properties =
        esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>,
                          esimd::cache_hint_L2<esimd::cache_hint::write_back>, esimd::alignment<8>};
    esimd::scatter<uint64_t, 1, 1>(p, esimd::simd<uint64_t, 1>(0), esimd::simd<uint64_t, 1>(value),
                                   properties);
    return;
  } else
    *p = value;
#else
  (void)Esimd;
  *p = value;
#endif
}
// Order scratch scatters before dependent gathers on the affected Windows ESIMD runtime.
template <bool Esimd> inline void stage3_order_writes() {
#if defined(MOM_XELISHASHV3_ESIMD) && defined(_WIN32)
  if constexpr (Esimd) {
    // Windows Level Zero has returned stale L2 data after the dependent scatters on some boots.
    // Keep the narrow group scope, but clean dirty lines before this work-item reads them again.
    esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::clean,
                 esimd::fence_scope::group>();
  }
#else
  (void)Esimd;
#endif
}

// Generate one ChaCha stream block from the stage key, nonce, counter, and round count.
inline void chacha_block(uint32_t out[16], const uint8_t key[32], const uint8_t nonce[12],
                         uint32_t counter) {
  uint32_t x[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};
  for (unsigned i = 0; i < 8; ++i)
    x[4 + i] = load32_le_dev(key + 4 * i);
  x[12] = counter;
  for (unsigned i = 0; i < 3; ++i)
    x[13 + i] = load32_le_dev(nonce + 4 * i);
  uint32_t v[16];
  for (unsigned i = 0; i < 16; ++i)
    v[i] = x[i];
#define QR(a, b, c, d)                                                                             \
  v[a] += v[b];                                                                                    \
  v[d] = mo_rotate(v[d] ^ v[a], 16u);                                                              \
  v[c] += v[d];                                                                                    \
  v[b] = mo_rotate(v[b] ^ v[c], 12u);                                                              \
  v[a] += v[b];                                                                                    \
  v[d] = mo_rotate(v[d] ^ v[a], 8u);                                                               \
  v[c] += v[d];                                                                                    \
  v[b] = mo_rotate(v[b] ^ v[c], 7u)
  for (unsigned round = 0; round < 4; ++round) {
    QR(0, 4, 8, 12);
    QR(1, 5, 9, 13);
    QR(2, 6, 10, 14);
    QR(3, 7, 11, 15);
    QR(0, 5, 10, 15);
    QR(1, 6, 11, 12);
    QR(2, 7, 8, 13);
    QR(3, 4, 9, 14);
  }
#undef QR
  for (unsigned i = 0; i < 16; ++i)
    out[i] = v[i] + x[i];
}
// Stage 1 derives one nonce's scratchpad with BLAKE3-keyed ChaCha blocks.
inline void stage1(const uint8_t source[112], uint64_t* scratch, uint64_t nonce) {
  uint8_t input[112], buffer[64], key[32], chain_nonce[12];
  for (unsigned i = 0; i < 112; ++i)
    input[i] = source[i];
  for (unsigned i = 0; i < 8; ++i)
    input[40 + i] = nonce >> (8 * (7 - i));
  blake3_dev(buffer, 32, input, 112);
  for (unsigned i = 0; i < 12; ++i)
    chain_nonce[i] = buffer[i];
  uint32_t* dst = reinterpret_cast<uint32_t*>(scratch);
  constexpr unsigned blocks = (BYTES / 4) / 64;
  for (unsigned chunk = 0; chunk < 4; ++chunk) {
    if (chunk)
      for (unsigned i = 0; i < 32; ++i)
        buffer[i] = key[i];
    for (unsigned i = 0; i < 32; ++i)
      buffer[32 + i] = chunk * 32 + i < 112 ? input[chunk * 32 + i] : 0;
    blake3_dev(key, 32, buffer, 64);
    for (unsigned block = 0; block < blocks; ++block) {
      uint32_t words[16];
      chacha_block(words, key, chain_nonce, block);
      for (unsigned i = 0; i < 16; ++i)
        *dst++ = words[i];
      if (block + 1 == blocks)
        for (unsigned i = 0; i < 12; ++i)
          chain_nonce[i] = static_cast<uint8_t>(words[13 + i / 4] >> (8 * (i & 3)));
    }
  }
}
// Run the data-dependent mixing loop in one work-item per nonce; scratch accesses remain sequential.
template <bool Native, bool Esimd = false, bool Fp64 = true>
inline void stage3(uint64_t* scratch, const double* lut) {
  // Stage 3 is the consensus-critical dependent walk. NVPTX overlaps one iteration's reads/writes;
  // ESIMD and portable SYCL share the straight-line ordering because that pipeline did not help B580.
  uint64_t* a_mem = scratch;
  uint64_t* b_mem = scratch + HALF;
  uint64_t addr_a = stage3_load<Esimd>(b_mem + HALF - 1),
           addr_b = stage3_load<Esimd>(a_mem + HALF - 1) >> 32;
  uint32_t r = 0;
  for (uint32_t i = 0; i < 2; ++i) {
    uint64_t ma = stage3_load<Esimd>(a_mem + map_index<Esimd>(addr_a)),
             mb = stage3_load<Esimd>(b_mem + map_index<Esimd>(ma ^ addr_b));
    aes_round<Esimd>(mb, ma);
    uint64_t result = ~(mb ^ ma);
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
    uint64_t read_a = map_index<Esimd>(result), next_a = scratch_load<Native>(a_mem + read_a),
             write_a = HALF + 1, write_b = HALF + 1, pending_t = 0;
    uint32_t pending_ij = 0;
    bool pending = false;
    for (uint32_t j = 0; j < HALF; ++j) {
      const uint64_t a = pending && write_a == read_a ? pending_t : next_a,
                     baddr = map_index<Esimd>(a ^ ~rotr64(result, r));
      uint64_t b = scratch_load<Native>(b_mem + baddr), old_a = 0, old_b = 0, pending_b = 0,
               c = scratch_load<Native>(scratch + r);
      if (pending) {
        old_a = scratch_load<Native>(a_mem + write_a);
        old_b = scratch_load<Native>(b_mem + write_b);
        pending_b = old_b ^ old_a ^ rotr64(pending_t, pending_ij);
        if (write_a == r)
          c = pending_t;
        else if (write_b + HALF == r)
          c = pending_b;
      }
      r = r < WORDS - 1 ? r + 1 : 0;
      const uint32_t op = rotl64(result, static_cast<uint32_t>(c)) & 15;
      if (pending) {
        if (write_b == baddr)
          b = pending_b;
        scratch_store<Native>(b_mem + write_b, pending_b);
        scratch_store<Native>(a_mem + write_a, pending_t);
      }
      const uint64_t v = operation<Esimd, Fp64>(op, a, b, c, r, result, i, j, lut);
      const uint64_t seed = v ^ result;
      result = rotl64(seed, r);
      const uint64_t idx = map_index<Esimd>(seed);
      read_a = map_index<Esimd>(result);
      next_a = scratch_load<Native>(a_mem + read_a);
      pending_t = (murmur(v) & (1ULL << 58) ? scratch_load<Native>(b_mem + idx)
                                            : scratch_load<Native>(a_mem + idx)) ^
                  result;
      write_a = map_index<Esimd>(pending_t ^ result ^ 0x9e3779b97f4a7c15ULL);
      write_b = map_index<Esimd>(write_a ^ ~result ^ 0xd2b74407b1ce6e93ULL);
      pending_ij = i + j;
      pending = true;
    }
    if (pending) {
      const uint64_t old_a = scratch_load<Native>(a_mem + write_a),
                     old_b = scratch_load<Native>(b_mem + write_b);
      scratch_store<Native>(b_mem + write_b, old_b ^ old_a ^ rotr64(pending_t, pending_ij));
      scratch_store<Native>(a_mem + write_a, pending_t);
    }
#else
    for (uint32_t j = 0; j < HALF; ++j) {
      const uint64_t a = stage3_load<Esimd>(a_mem + map_index<Esimd>(result)),
                     b = stage3_load<Esimd>(b_mem + map_index<Esimd>(a ^ ~rotr64(result, r))),
                     c = stage3_load<Esimd>(scratch + r);
      r = r < WORDS - 1 ? r + 1 : 0;
      const uint64_t v = operation<Esimd, Fp64>(rotl64(result, static_cast<uint32_t>(c)) & 15,
                                               a, b, c, r, result, i, j, lut),
                     seed = v ^ result;
      result = rotl64(seed, r);
      const uint64_t idx = map_index<Esimd>(seed),
                     t = (murmur(v) & (1ULL << 58) ? stage3_load<Esimd>(b_mem + idx)
                                                   : stage3_load<Esimd>(a_mem + idx)) ^
                         result,
                     ia = map_index<Esimd>(t ^ result ^ 0x9e3779b97f4a7c15ULL),
                     ib = map_index<Esimd>(ia ^ ~result ^ 0xd2b74407b1ce6e93ULL),
                     old = stage3_load<Esimd>(a_mem + ia), old_b = stage3_load<Esimd>(b_mem + ib);
      stage3_store<Esimd>(a_mem + ia, t);
      stage3_store<Esimd>(b_mem + ib, old_b ^ old ^ rotr64(t, i + j));
      // Only this work-item consumes the dependent stores; avoid a system-wide commit fence.
      stage3_order_writes<Esimd>();
    }
#endif
    addr_a = modpow<Esimd, Fp64>(addr_a, addr_b, result);
    addr_b = isqrt64<Esimd>(result) * (r + 1) * isqrt64<Esimd>(addr_a);
  }
}

// Compress one scratch block while retaining the BLAKE3 chaining-value word layout.
inline void b3_compress_words(uint32_t cv[8], const uint32_t* m, uint64_t counter, uint32_t flags) {
  uint32_t st[16] = {cv[0],
                     cv[1],
                     cv[2],
                     cv[3],
                     cv[4],
                     cv[5],
                     cv[6],
                     cv[7],
                     B3_IV[0],
                     B3_IV[1],
                     B3_IV[2],
                     B3_IV[3],
                     static_cast<uint32_t>(counter),
                     static_cast<uint32_t>(counter >> 32),
                     64,
                     flags};
  for (unsigned round = 0; round < 7; ++round)
    b3_round(st, m, round);
  for (unsigned i = 0; i < 8; ++i)
    cv[i] = st[i] ^ st[i + 8];
}
// Combine two BLAKE3 tree nodes; only the final parent sets ROOT.
inline void b3_parent(const uint32_t left[8], const uint32_t right[8], uint32_t out[8], bool root) {
  uint32_t m[16], cv[8];
  for (unsigned i = 0; i < 8; ++i) {
    cv[i] = B3_IV[i];
    m[i] = left[i];
    m[i + 8] = right[i];
  }
  b3_compress_words(cv, m, 0, 4u | (root ? 8u : 0u));
  for (unsigned i = 0; i < 8; ++i)
    out[i] = cv[i];
}
// Stage 4 reduces the scratchpad through the algorithm's BLAKE3 tree and emits the final hash.
inline void stage4(const uint64_t* scratch, uint8_t hash[32]) {
  const uint32_t* words = reinterpret_cast<const uint32_t*>(scratch);
  uint32_t stack[10][8];
  unsigned n = 0;
  for (uint64_t chunk = 0; chunk < 531; ++chunk) {
    uint32_t cv[8];
    for (unsigned i = 0; i < 8; ++i)
      cv[i] = B3_IV[i];
    for (unsigned block = 0; block < 16; ++block)
      b3_compress_words(cv, words + chunk * 256 + block * 16, chunk,
                        (block == 0 ? 1u : 0u) | (block == 15 ? 2u : 0u));
    for (unsigned i = 0; i < 8; ++i)
      stack[n][i] = cv[i];
    ++n;
    for (uint64_t merges = chunk + 1; !(merges & 1) && n > 1; merges >>= 1) {
      uint32_t parent[8];
      b3_parent(stack[n - 2], stack[n - 1], parent, false);
      --n;
      for (unsigned i = 0; i < 8; ++i)
        stack[n - 1][i] = parent[i];
    }
  }
  while (n > 2) {
    uint32_t parent[8];
    b3_parent(stack[n - 2], stack[n - 1], parent, false);
    --n;
    for (unsigned i = 0; i < 8; ++i)
      stack[n - 1][i] = parent[i];
  }
  uint32_t root[8];
  b3_parent(stack[0], stack[1], root, true);
  for (unsigned i = 0; i < 8; ++i)
    store32_le_dev(hash + 4 * i, root[i]);
}
