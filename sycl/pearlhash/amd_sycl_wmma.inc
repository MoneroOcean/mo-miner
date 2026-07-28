#if defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
using PearlHashAmdV2i = int __attribute__((ext_vector_type(2)));
using PearlHashAmdV8i = int __attribute__((ext_vector_type(8)));
template <int K, int Rank, int ER, int EC> class PearlHashAmdWmmaKernel;
template <int K, int Rank, int ER, int EC> class PearlHashAmdHashKernel;

template <int K, int Rank, int ER = 4, int EC = 2>
static void search_amd_wmma_t(sycl::queue& q, const Buffers& bb,
                              uint32_t seed, int m, int n, bool dbg) {
  static thread_local unsigned kernel_stats = 0;
  const bool stats = std::getenv("MOM_PEARLHASH_STATS") && kernel_stats++ < 3;
  const auto wmma_start = std::chrono::steady_clock::now();
  constexpr int wave_size = 32;
  int threads = 256;
  if (const char* value = std::getenv("MOM_PEARLHASH_AMD_WMMA_THREADS")) {
    const int parsed = std::atoi(value);
    if (parsed == 32 || parsed == 64 || parsed == 128 || parsed == 256) threads = parsed;
  }
  const int blocks_m = m / (16 * ER);
  const int blocks_n = n / (16 * EC);
  const uint64_t waves = static_cast<uint64_t>(blocks_m) * blocks_n;
  const size_t global = static_cast<size_t>(waves * wave_size);
  const int8_t* const ap = bb.Ap;
  const int8_t* const bp = bb.Bp;
  Result* const result = bb.result;
  const uint8_t* const cA = bb.cA;
  const uint8_t* const target = bb.target;
  uint32_t* const transcript = bb.transcript;

  // Traverse square groups of waves before advancing through the complete output. This retains the
  // corresponding A'/B' bands in the GPU caches instead of streaming all of B' again for every A'
  // band. Small correctness-test shapes fall back to the simple row-major mapping.
  int cache_block = 32;
  if (const char* value = std::getenv("MOM_PEARLHASH_AMD_WMMA_CACHE_BLOCK"))
    cache_block = std::atoi(value);
  const bool blocked = cache_block > 0 && blocks_m % cache_block == 0 &&
                       blocks_n % cache_block == 0;

  q.parallel_for<PearlHashAmdWmmaKernel<K, Rank, ER, EC>>(
    sycl::nd_range<1>{global, threads},
    [ap, bp, result, cA, target, transcript, blocks_n, cache_block, blocked, seed, dbg](sycl::nd_item<1> item)
      [[sycl::reqd_sub_group_size(32)]] {
#if defined(__AMDGCN__)
    const int lane = static_cast<int>(item.get_local_linear_id() & 31U);
    const uint64_t wave = item.get_global_linear_id() / wave_size;
    int row_group, col_group;
    if (blocked) {
      const uint64_t block_area = static_cast<uint64_t>(cache_block) * cache_block;
      const uint64_t block = wave / block_area;
      const uint64_t inner = wave - block * block_area;
      const int blocks_w = blocks_n / cache_block;
      row_group = static_cast<int>(block / blocks_w) * cache_block +
                  static_cast<int>(inner / cache_block);
      col_group = static_cast<int>(block - (block / blocks_w) * blocks_w) * cache_block +
                  static_cast<int>(inner % cache_block);
    } else {
      row_group = static_cast<int>(wave / blocks_n);
      col_group = static_cast<int>(wave - static_cast<uint64_t>(row_group) * blocks_n);
    }
    const int tile_row = row_group * ER;
    const int tile_col = col_group * EC;
    const int row_or_col = lane & 15;
    const int k_half = (lane & 16) ? 8 : 0;
    auto a_words = sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(
      reinterpret_cast<const int32_t*>(ap));
    auto b_words = sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(
      reinterpret_cast<const int32_t*>(bp));
    PearlHashAmdV8i acc[ER][EC] = {};
    uint32_t tr_lane[ER][EC] = {};

#pragma unroll 2
    for (int kk = 0; kk < K; kk += 16) {
      PearlHashAmdV2i a[ER], b[EC];
#pragma unroll
      for (int i = 0; i < ER; ++i) {
        const uint64_t word =
          (static_cast<uint64_t>((tile_row + i) * 16 + row_or_col) * K + kk + k_half) / 4U;
        a[i] = PearlHashAmdV2i{a_words[word], a_words[word + 1]};
      }
#pragma unroll
      for (int j = 0; j < EC; ++j) {
        const uint64_t word =
          (static_cast<uint64_t>((tile_col + j) * 16 + row_or_col) * K + kk + k_half) / 4U;
        b[j] = PearlHashAmdV2i{b_words[word], b_words[word + 1]};
      }
#pragma unroll
      for (int i = 0; i < ER; ++i) {
#pragma unroll
        for (int j = 0; j < EC; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            true, a[i], true, b[j], acc[i][j], false);
      }

      if ((kk + 16) % Rank == 0) {
        const int rc = kk / Rank;
#pragma unroll
        for (int i = 0; i < ER; ++i) {
#pragma unroll
          for (int j = 0; j < EC; ++j) {
            uint32_t x = 0;
#pragma unroll
            for (int e = 0; e < 8; ++e) x ^= static_cast<uint32_t>(acc[i][j][e]);
#pragma unroll
            for (int delta = 1; delta < wave_size; delta <<= 1) {
              x ^= __builtin_amdgcn_ds_bpermute(static_cast<uint32_t>(lane ^ delta) * 4U, x);
            }
            if (lane == rc) tr_lane[i][j] = x;
          }
        }
      }
    }

    // Each transcript word lives in the lane matching its rank-chunk number. Gather those words
    // directly from the wave and finalize the tile here. This removes the former 4 GiB full-search
    // transcript plus its separate scan kernel while keeping every architecture-facing operation in
    // SYCL (the only backend extension above is the WMMA instruction itself).
    const auto sg = item.get_sub_group();
#pragma unroll
    for (int i = 0; i < ER; ++i) {
#pragma unroll
      for (int j = 0; j < EC; ++j) {
        uint32_t tr[16];
#pragma unroll
        for (int word = 0; word < 16; ++word)
          tr[word] = sycl::select_from_group(sg, tr_lane[i][j], static_cast<uint32_t>(word));
        if (lane == 0) {
          const uint32_t row = static_cast<uint32_t>(tile_row + i) * 16U;
          const uint32_t col = static_cast<uint32_t>(tile_col + j) * 16U;
          if (transcript) {
            const uint64_t tile = static_cast<uint64_t>(row / 16U) *
                                  static_cast<uint64_t>(blocks_n * EC) + col / 16U;
#pragma unroll
            for (int word = 0; word < 16; ++word)
              transcript[tile * 16U + static_cast<uint32_t>(word)] = tr[word];
          } else {
            if (dbg) {
              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                               sycl::access::address_space::global_space> checksum(result->chk);
              checksum.fetch_add(tile_mix(row, col, tr));
            }
            if (tile_wins(tr, cA, target)) {
              sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                               sycl::access::address_space::global_space> found(result->found);
              if (found.exchange(1) == 0) {
                result->seed = seed; result->row = row; result->col = col;
              }
            }
          }
        }
      }
    }
#else
    (void)item; (void)ap; (void)bp; (void)result; (void)cA; (void)target;
    (void)transcript; (void)blocks_n; (void)cache_block; (void)blocked; (void)seed; (void)dbg;
#endif
  });
  if (stats) {
    q.wait_and_throw();
    const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - wmma_start).count();
    std::fprintf(stderr, "PEARLHASH_SYCL_STATS wmma_only=%.3fms\n", ms);
  }

  if (transcript) {
    const auto hash_start = std::chrono::steady_clock::now();
    const uint64_t tile_columns = static_cast<uint64_t>(n / 16);
    const uint64_t tiles = static_cast<uint64_t>(m / 16) * tile_columns;
    q.parallel_for<PearlHashAmdHashKernel<K, Rank, ER, EC>>(
      sycl::range<1>(static_cast<size_t>(tiles)),
      [transcript, result, cA, target, tile_columns, seed, dbg](sycl::id<1> id) {
        const uint64_t tile = id[0];
        uint32_t tr[16];
#pragma unroll
        for (int word = 0; word < 16; ++word)
          tr[word] = transcript[tile * 16U + static_cast<uint32_t>(word)];
        const uint32_t row = static_cast<uint32_t>(tile / tile_columns) * 16U;
        const uint32_t col = static_cast<uint32_t>(tile - (tile / tile_columns) * tile_columns) * 16U;
        if (dbg) {
          sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                           sycl::access::address_space::global_space> checksum(result->chk);
          checksum.fetch_add(tile_mix(row, col, tr));
        }
        if (tile_wins(tr, cA, target)) {
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                           sycl::access::address_space::global_space> found(result->found);
          if (found.exchange(1) == 0) {
            result->seed = seed; result->row = row; result->col = col;
          }
        }
      });
    if (stats) {
      q.wait_and_throw();
      const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - hash_start).count();
      std::fprintf(stderr, "PEARLHASH_SYCL_STATS hash_only=%.3fms\n", ms);
    }
  }
}
#endif
