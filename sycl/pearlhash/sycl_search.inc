// Portable int8 GEMM search -- the matrix-hardware-free path that EVERY OpenCL device takes: AMD GPUs
// (no ESIMD, no usable joint_matrix), the SYCL CPU device (cpu1), and Intel OpenCL. Each
// work-group computes one 16x16 hash tile; its work-items split the tile's 256 cells into ER x EC
// micro-tiles (one work-item per micro-tile, ER*EC int32 accumulators in registers). A'/B' are
// laid out by compute_ab(portable=true) -- A' row-major, B' column-major -- so each output cell reads a
// contiguous int8 run over k that the compiler packs (4 int8 at a time, loaded as one int32) into the
// 4-wide integer dot product: dp4a on Intel/AMD GPUs, plain scalar int MACs on the CPU -- bit-identical
// either way, since int8*int8->int32 is exact everywhere. There is NO SLM staging and NO per-rank
// barrier: A'/B' reuse is left to the L2 cache, and each work-item folds its own cells into a private
// 16-word partial transcript during the k-loop, so the ONLY work-group synchronisation is a single
// XOR-reduction tree at the end (a plain barrier tree -- not a sub-group op, so no reqd_sub_group_size
// is needed and the kernel runs on AMD and the CPU where the sub-group size is not 16). The result
// (found/seed/row/col) and transcript are bit-identical to search_esimd / search_cuda for the same A'/B'
// (cross-checked via MOM_PEARLHASH_CHK), so this path finds the same tiles. Requires k % rank == 0 and
// k/rank <= 16 (true for every shipped shape: network 4096/256 and pearlpool 1024/64 both give 16).
// Compiled in every build that carries attempt() -- including the Windows/dpcpp Intel build where it
// shares the TU with search_esimd -- and EXCLUDED only from the separate spir64-only ESIMD TU
// (esimd.cpp, which sets MOM_PEARLHASH_ESIMD_TU and re-includes this file just for search_esimd).
#if !defined(MOM_PEARLHASH_ESIMD_TU)
// Intel's Windows CPU OpenCL JIT corrupts its host heap after the work-group search below (the
// failure is an access violation/STATUS_HEAP_CORRUPTION after the kernel has otherwise completed).
// The failure is sensitive to the DPC++ build: even standards-only single-task/range variants can
// corrupt the Windows CPU runtime while passing with another build of the same compiler. CPU
// allocations are shared USM, so finish the compact compatibility search on the host after the
// SYCL root/noise stages. CPU mining is not a performance target; deterministic correctness across
// OpenCL implementations is more useful than another fragile JIT shape. GPU devices never call
// this function and retain the measured parallel implementation unchanged.
static void search_cpu(
    sycl::queue& q, const Buffers& bb, uint32_t seed,
    int m, int n, int k, int rank, bool dbg) {
  q.wait_and_throw();
  constexpr int HT = 16;
  const int nb = k / rank;
  for (int row_base = 0; row_base < m; row_base += HT) {
    for (int col_base = 0; col_base < n; col_base += HT) {
      int32_t acc[HT][HT]{};
      uint32_t transcript[16]{};
      for (int boundary = 0; boundary < nb; ++boundary) {
        const int end = (boundary + 1) * rank;
        for (int depth = boundary * rank; depth < end; ++depth) {
          for (int i = 0; i < HT; ++i) {
            const int a = static_cast<int>(
              bb.Ap[static_cast<size_t>(row_base + i) * k + depth]);
            for (int j = 0; j < HT; ++j) {
              const int b = static_cast<int>(
                bb.Bp[static_cast<size_t>(col_base + j) * k + depth]);
              acc[i][j] += a * b;
            }
          }
        }
        for (int i = 0; i < HT; ++i)
          for (int j = 0; j < HT; ++j)
            transcript[boundary] ^= static_cast<uint32_t>(acc[i][j]);
      }
      if (dbg)
        bb.result->chk += tile_mix(
          static_cast<uint32_t>(row_base), static_cast<uint32_t>(col_base), transcript);
      if (!bb.result->found && tile_wins(transcript, bb.cA, bb.target)) {
        bb.result->found = 1;
        bb.result->seed = seed;
        bb.result->row = static_cast<uint32_t>(row_base);
        bb.result->col = static_cast<uint32_t>(col_base);
        if (!dbg) return;
      }
    }
  }
}

template <int ER, int EC>
static void search_t(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg) {
  constexpr int HT = 16;
  constexpr int WG = (HT * HT) / (ER * EC);                 // work-items per tile
  [[maybe_unused]] constexpr int BW = HT / EC;              // micro-tile blocks per row (unused on the empty nvptx pass)
  static_assert(HT % ER == 0 && HT % EC == 0, "pearlhash portable: ER/EC must divide 16");
  static_assert((WG & (WG - 1)) == 0, "pearlhash portable: work-group size must be a power of two");
  auto B = bb;
  const int tilesH = m / HT, tilesW = n / HT, k4 = k / 4, nb = k / rank;
  const size_t ntiles = (size_t)tilesH * tilesW;
  // The production A'/B' matrices are 512 MiB each. Plain row-major tile order re-streams all of B'
  // for each A' band; on HIP, visit square tile blocks instead so both slices remain in L2/Infinity
  // Cache. This is only a group-id remap: arithmetic, allocations, and every non-HIP path are unchanged.
  int cache_block = 0;
#if defined(MOM_SYCL_HAS_HIP)
  #if defined(MOM_SYCL_ADAPTIVECPP)
  const bool is_hip = q.get_device().get_backend() == sycl::backend::hip;
  #else
  const bool is_hip = q.get_device().get_backend() == sycl::backend::ext_oneapi_hip;
  #endif
  if (is_hip) {
    cache_block = 64;
    if (const char* value = std::getenv("MOM_PEARLHASH_AMD_DP4A_CACHE_BLOCK"))
      cache_block = std::atoi(value);
  }
#endif
  const bool blocked = cache_block > 1 && tilesH % cache_block == 0 &&
                       tilesW % cache_block == 0;
  const int blocksW = blocked ? tilesW / cache_block : 0;
  // A full 131072-square 2x2 search contains exactly 2^32 work-items. AdaptiveCpp's HIP launcher
  // cannot represent that as one ND-range (hipErrorInvalidValue), even though the group count is
  // legal. Submit bounded contiguous pieces on the in-order queue and add their group base in the
  // kernel. This also keeps custom larger shapes away from the same 32-bit launcher edge.
  constexpr size_t max_groups_per_launch = size_t{1} << 22;
  for (size_t group_base = 0; group_base < ntiles; group_base += max_groups_per_launch) {
    const size_t group_count = std::min(max_groups_per_launch, ntiles - group_base);
    q.submit([&](sycl::handler& h) {
      sycl::local_accessor<uint32_t, 1> red(WG * 16, h);   // batched per-boundary XOR reduction scratch
      // Explicit capture list (not [=]): keeps the kernel-lambda layout identical across the host, spir64
      // and nvptx passes (the nvptx body is #ifdef'd out below), so the dual-compiler combined build's
      // "Unexpected kernel lambda size" static assert stays satisfied.
      h.parallel_for(sycl::nd_range<1>(group_count * WG, WG),
                   [B, seed, red, tilesW, k4, rank, nb, dbg, cache_block, blocked, blocksW,
                    group_base](sycl::nd_item<1> it) {
      const int g = static_cast<int>(group_base + it.get_group(0));
      int R, C;
      if (blocked) {
        const int area = cache_block * cache_block, block = g / area, inner = g % area;
        R = (block / blocksW) * cache_block + inner / cache_block;
        C = (block % blocksW) * cache_block + inner % cache_block;
      } else { R = g / tilesW; C = g % tilesW; }
      const int lid = (int)it.get_local_linear_id();
      const int r0 = (lid / BW) * ER, c0 = (lid % BW) * EC;     // this work-item's micro-tile origin
      const int rowBase = R * HT, colBase = C * HT;
      // A'/B' read as int32 words (4 packed int8) straight from global -- the L2 absorbs the per-tile
      // reuse (each 128KB tile slice stays resident), which is why no SLM staging is needed.
      auto A32 = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                   reinterpret_cast<const int32_t*>(B.Ap));
      auto B32 = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                   reinterpret_cast<const int32_t*>(B.Bp));
      int32_t acc[ER][EC]; for (int i = 0; i < ER; i++) for (int j = 0; j < EC; j++) acc[i][j] = 0;
      uint32_t pX[16]; for (int w = 0; w < 16; w++) pX[w] = 0;   // this work-item's per-boundary partial XOR
      for (int rc = 0; rc < nb; rc++) {
        const int c4lo = rc * (rank / 4), c4hi = c4lo + (rank / 4);
        for (int c4 = c4lo; c4 < c4hi; c4++) {        // each step = 4 columns packed into one int32 word
          int32_t bw[EC]; for (int j = 0; j < EC; j++) bw[j] = B32[(size_t)(colBase + c0 + j) * k4 + c4];
          for (int i = 0; i < ER; i++) {
            const int32_t aw = A32[(size_t)(rowBase + r0 + i) * k4 + c4];
            const int a0 = (int8_t)aw, a1 = (int8_t)(aw >> 8), a2 = (int8_t)(aw >> 16), a3 = (int8_t)(aw >> 24);
            for (int j = 0; j < EC; j++) {           // dp4a: 4 int8 MACs (one ALU op on Intel/AMD GPUs)
              const int32_t b = bw[j];
              acc[i][j] += a0*(int8_t)b + a1*(int8_t)(b >> 8) + a2*(int8_t)(b >> 16) + a3*(int8_t)(b >> 24);
            }
          }
        }
        uint32_t lx = 0; for (int i = 0; i < ER; i++) for (int j = 0; j < EC; j++) lx ^= (uint32_t)acc[i][j];
        pX[rc] = lx;   // cumulative tile XOR at this rank boundary (rc < nb <= 16)
      }
      // Single batched work-group reduction: XOR every work-item's 16-word partial transcript together.
      auto grp = it.get_group();
      for (int w = 0; w < 16; w++) red[lid * 16 + w] = pX[w];
      sycl::group_barrier(grp);
      for (int half = WG >> 1; half > 0; half >>= 1) {
        if (lid < half) for (int w = 0; w < 16; w++) red[lid * 16 + w] ^= red[(lid + half) * 16 + w];
        sycl::group_barrier(grp);
      }
      if (lid == 0) {
        // nb <= 16 => each rank boundary owns a distinct transcript word, so tr[w] == part[w] (the
        // rotl-fold in search_esimd only matters when nb > 16, which no shipped shape uses).
        uint32_t t16[16]; for (int w = 0; w < 16; w++) t16[w] = red[w];
        if (dbg) {
          sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> c(B.result->chk);
          c.fetch_add(tile_mix((uint32_t)rowBase, (uint32_t)colBase, t16));
        }
        if (tile_wins(t16, B.cA, B.target)) {
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> a(B.result->found);
          if (a.exchange(1) == 0) { B.result->seed = seed; B.result->row = (uint32_t)rowBase; B.result->col = (uint32_t)colBase; }
        }
      }
      });
    });
  }
}
// Run the portable search at the measured B580 OpenCL 4x4 micro-tile. Retuning changes the typed
// constants above rather than adding a preprocessor variant. SLM staging and software-pipelined prefetch were both measured slower on
// the B580 (the L2 already absorbs the A'/B' reuse; IGC schedules the loads better than a manual pipeline).
static void search(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg) {
#if defined(MOM_SYCL_HAS_HIP)
  // gfx12 normally takes the SYCL WMMA specialization below. Keep its measured-best 2x2 dp4a
  // fallback for unsupported shapes/compilers (RX 9060 XT: 750 vs 496 GH/s for 2x2 versus the
  // Intel-tuned 4x4). Non-HIP devices retain their own 4x4 default.
  bool is_hip = false;
  #if defined(MOM_SYCL_ADAPTIVECPP)
  is_hip = q.get_device().get_backend() == sycl::backend::hip;
  #else
  is_hip = q.get_device().get_backend() == sycl::backend::ext_oneapi_hip;
  #endif
  if (is_hip) {
    // Keep several fully typed dp4a micro-tiles in the architecture-neutral image. This is a host
    // dispatch, so each hot kernel contains no runtime tile branch. MOM_PEARLHASH_AMD_DP4A_TILE is a
    // tuning/validation override; 2x2 remains the measured default until the final AMD SSCP sweep.
    const char* const tile = std::getenv("MOM_PEARLHASH_AMD_DP4A_TILE");
    if (tile && !std::strcmp(tile, "1x1")) return search_t<1, 1>(q, bb, seed, m, n, k, rank, dbg);
    if (tile && !std::strcmp(tile, "2x4")) return search_t<2, 4>(q, bb, seed, m, n, k, rank, dbg);
    if (tile && !std::strcmp(tile, "4x2")) return search_t<4, 2>(q, bb, seed, m, n, k, rank, dbg);
    if (tile && !std::strcmp(tile, "4x4")) return search_t<4, 4>(q, bb, seed, m, n, k, rank, dbg);
    if (tile && !std::strcmp(tile, "8x2")) return search_t<8, 2>(q, bb, seed, m, n, k, rank, dbg);
    return search_t<2, 2>(q, bb, seed, m, n, k, rank, dbg);
  }
#endif
  search_t<pearlhash_portable_er, pearlhash_portable_ec>(q, bb, seed, m, n, k, rank, dbg);
}
#endif  // !MOM_PEARLHASH_ESIMD_TU
