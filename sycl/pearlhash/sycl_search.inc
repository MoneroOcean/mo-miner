// Portable int8 GEMM search -- the matrix-hardware-free path that EVERY OpenCL device takes: AMD GPUs
// (no ESIMD, no usable joint_matrix), the SYCL CPU device (cpu1), and Intel OpenCL. Each
// work-group computes one 16x16 hash tile; its work-items split the tile's 256 cells into ER x EC
// micro-tiles (one work-item per micro-tile, ER*EC wrapping uint32 accumulators in registers). A'/B' are
// laid out by compute_ab(portable=true) -- A' row-major, B' column-major -- so each output cell reads a
// contiguous int8 run over k that the compiler packs (4 int8 at a time, loaded as one int32) into the
// 4-wide integer dot product: dp4a on Intel/AMD GPUs, plain scalar int MACs on the CPU -- bit-identical
// either way, since int8*int8->int32 is exact everywhere. Ordinary DP4A has no SLM staging or per-rank
// barrier: A'/B' reuse is left to the L2 cache, and each work-item folds its own cells into a private
// 16-word partial transcript during the k-loop, so the ONLY work-group synchronisation is a single
// XOR-reduction tree at the end (a plain barrier tree -- not a sub-group op, so no reqd_sub_group_size
// is needed and the kernel runs on AMD and the CPU where the sub-group size is not 16). The result
// (found/seed/row/col) and transcript are bit-identical to search_esimd / search_cuda for the same A'/B'
// (cross-checked via MOM_PEARLHASH_CHK), so this path finds the same tiles. Each complete rank-boundary
// part folds into one of the 16 transcript words; the verifier likewise excludes K % rank trailing cells.
// The optional Volta FP16 specialization stages canonical K16 slices for WMMA before this same fold.
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
static void search_cpu(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k,
                       int rank, bool dbg) {
  q.wait_and_throw();
  constexpr int HT = 16;
  const int nb = k / rank;
  for (int row_base = 0; row_base < m; row_base += HT) {
    for (int col_base = 0; col_base < n; col_base += HT) {
      uint32_t acc[HT][HT]{};
      uint32_t transcript[16]{};
      for (int boundary = 0; boundary < nb; ++boundary) {
        const int end = (boundary + 1) * rank;
        for (int depth = boundary * rank; depth < end; ++depth) {
          for (int i = 0; i < HT; ++i) {
            const int a = static_cast<int>(bb.Ap[static_cast<size_t>(row_base + i) * k + depth]);
            for (int j = 0; j < HT; ++j) {
              const int b = static_cast<int>(bb.Bp[static_cast<size_t>(col_base + j) * k + depth]);
              acc[i][j] += static_cast<uint32_t>(a * b);
            }
          }
        }
        uint32_t part = 0;
        for (int i = 0; i < HT; ++i)
          for (int j = 0; j < HT; ++j)
            part ^= static_cast<uint32_t>(acc[i][j]);
        const int slot = boundary % 16;
        transcript[slot] = rotl(transcript[slot], 13) ^ part;
      }
      if (dbg)
        bb.result->chk +=
            tile_mix(static_cast<uint32_t>(row_base), static_cast<uint32_t>(col_base), transcript);
      uint8_t jackpot[32];
      if (!bb.result->found && tile_wins(transcript, bb.cA, bb.target, jackpot)) {
        bb.result->found = 1;
        bb.result->seed = seed;
        bb.result->row = static_cast<uint32_t>(row_base);
        bb.result->col = static_cast<uint32_t>(col_base);
        for (int i = 0; i < 32; ++i)
          bb.result->jackpot[i] = jackpot[i];
        if (!dbg)
          return;
      }
    }
  }
}

#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
// The SSCP reflection pass resolves this existing runtime field by its query suffix.
extern "C" bool __acpp_sscp_jit_reflect_knows_ptx_version();
extern "C" int __acpp_sscp_jit_reflect_ptx_version();
struct PearlPtxVersion {
  static bool is_known() { return __acpp_sscp_jit_reflect_knows_ptx_version(); }
  static int get() { return __acpp_sscp_jit_reflect_ptx_version(); }
};
// Target reflection is immutable across the warp/rank loop, including the rare DP4A fallback.
static inline bool pearlhash_uses_wmma_f16() {
  __acpp_if_target_sscp(
    namespace jit = sycl::AdaptiveCpp_jit;
    namespace query = jit::reflection_query;
    if (jit::knows<query::compiler_backend>() &&
        jit::reflect<query::compiler_backend>() == jit::compiler_backend::ptx &&
        jit::knows<query::target_vendor_id>() &&
        jit::reflect<query::target_vendor_id>() == jit::vendor_id::nvidia &&
        jit::knows<query::target_arch>() &&
        jit::reflect<query::target_arch>() >= 70 &&
        !mom::nvidia::has_int8_mma(jit::reflect<query::target_arch>()) &&
        jit::knows<PearlPtxVersion>() && jit::reflect<PearlPtxVersion>() >= 60)
      return true;
  )
  return false;
}

// All signed int8 inputs are exact half values; the first element occupies the low 16 bits.
static inline uint32_t pearlhash_f16x2_s8(int8_t low, int8_t high) {
  __acpp_if_target_sscp(
    if (pearlhash_uses_wmma_f16()) {
      // Avoid the generic half type's software IEEE conversion in this inner loop.
      uint32_t packed;
      asm volatile("%{ .reg .b16 lo,hi;\n"
                   "cvt.rn.f16.s32 lo,%1; cvt.rn.f16.s32 hi,%2;\n"
                   "mov.b32 %0,%{lo,hi%}; %}\n"
                   : "=r"(packed)
                   : "r"(static_cast<int>(low)), "r"(static_cast<int>(high)));
      return packed;
    }
  )
  const uint32_t lo = sycl::bit_cast<uint16_t>(static_cast<sycl::half>(static_cast<int>(low)));
  const uint32_t hi = sycl::bit_cast<uint16_t>(static_cast<sycl::half>(static_cast<int>(high)));
  return lo | (hi << 16);
}

// WMMA loads distribute opaque fragments from canonical shared A-row/B-column matrices.
static inline void pearlhash_wmma_m16n16k16_f16(
    float c[8], const uint32_t* a, const uint32_t* b) {
#define MOM_PEARLHASH_WMMA(SUFFIX) \
      asm volatile("%{ .reg .f32 f<8>; .reg .b32 a<8>, b<8>;\n" \
                   ".reg .u64 ap, bp; .reg .u32 as, bs;\n" \
                   "cvta.to.shared.u64 ap,%8; cvta.to.shared.u64 bp,%9;\n" \
                   "cvt.u32.u64 as,ap; cvt.u32.u64 bs,bp;\n" \
                   "wmma.load.a.sync" SUFFIX ".m16n16k16.row.shared.f16 " \
                   "%{a0,a1,a2,a3,a4,a5,a6,a7%}, [as], 16;\n" \
                   "wmma.load.b.sync" SUFFIX ".m16n16k16.col.shared.f16 " \
                   "%{b0,b1,b2,b3,b4,b5,b6,b7%}, [bs], 16;\n" \
                   "mov.b32 f0,%0; mov.b32 f1,%1; mov.b32 f2,%2; mov.b32 f3,%3;\n" \
                   "mov.b32 f4,%4; mov.b32 f5,%5; mov.b32 f6,%6; mov.b32 f7,%7;\n" \
                   "wmma.mma.sync" SUFFIX ".m16n16k16.row.col.f32.f32 " \
                   "%{f0,f1,f2,f3,f4,f5,f6,f7%}, %{a0,a1,a2,a3,a4,a5,a6,a7%}, " \
                   "%{b0,b1,b2,b3,b4,b5,b6,b7%}, %{f0,f1,f2,f3,f4,f5,f6,f7%};\n" \
                   "mov.b32 %0,f0; mov.b32 %1,f1; mov.b32 %2,f2; mov.b32 %3,f3;\n" \
                   "mov.b32 %4,f4; mov.b32 %5,f5; mov.b32 %6,f6; mov.b32 %7,f7; %}\n" \
                   : "+r"(bits[0]), "+r"(bits[1]), "+r"(bits[2]), "+r"(bits[3]), \
                     "+r"(bits[4]), "+r"(bits[5]), "+r"(bits[6]), "+r"(bits[7]) \
                   : "l"(a_address), "l"(b_address) \
                   : "memory");
  __acpp_if_target_sscp(
    if (pearlhash_uses_wmma_f16()) {
      // Integer constraints survive the SSCP host frontend. PTX mov.b32 only reinterprets
      // the same FP32 bits, avoiding either a host-only floating constraint or conversion.
      uint32_t bits[8];
      for (int i = 0; i < 8; ++i)
        bits[i] = sycl::bit_cast<uint32_t>(c[i]);
      const uint64_t a_address = reinterpret_cast<uintptr_t>(a);
      const uint64_t b_address = reinterpret_cast<uintptr_t>(b);
      // Convert generic pointers, not their low bits; the clobber orders hidden shared reads.
      // PTX before 6.3 implies alignment; later versions require the explicit modifier.
      if (sycl::AdaptiveCpp_jit::reflect<PearlPtxVersion>() >= 63) {
        MOM_PEARLHASH_WMMA(".aligned");
      } else {
        MOM_PEARLHASH_WMMA("");
      }
      for (int i = 0; i < 8; ++i)
        c[i] = sycl::bit_cast<float>(bits[i]);
    }
  )
#undef MOM_PEARLHASH_WMMA
}
// ACPP's dynamic shared base is only declared 4-byte aligned. Leave headroom and
// align the interior pointer explicitly; over-aligning this type can hide that gap.
struct PearlF16Scratch { uint32_t words[520]; };
static_assert(alignof(PearlF16Scratch) == alignof(uint32_t) && sizeof(PearlF16Scratch) == 2080,
              "WMMA scratch needs 32-byte headroom before its 512-word reduction");
template <int ER, int EC, bool Native = false, bool FP16Matrix = false>
#else
template <int ER, int EC>
#endif
// Submit the selected portable/native SYCL tile specialization without changing proof semantics.
static void search_t(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k,
                     int rank, bool dbg) {
  constexpr int HT = 16;
  constexpr int WG = (HT * HT) / (ER * EC); // work-items per tile
  [[maybe_unused]] constexpr int BW =
      HT / EC; // micro-tile blocks per row (unused on the empty nvptx pass)
  static_assert(HT % ER == 0 && HT % EC == 0, "pearlhash portable: ER/EC must divide 16");
  static_assert((WG & (WG - 1)) == 0, "pearlhash portable: work-group size must be a power of two");
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  static_assert(!FP16Matrix || (Native && ER * EC == 8),
                "pearlhash f16: one full warp owns one 16x16 tile");
#endif
  auto B = bb;
  const uint32_t tilesH = static_cast<uint32_t>(m / HT), tilesW = static_cast<uint32_t>(n / HT);
  const int k4 = k / 4, nb = k / rank;
  const size_t ntiles = static_cast<size_t>(tilesH) * tilesW;
  // The production A'/B' matrices are 512 MiB each. Plain row-major tile order re-streams all of B'
  // for each A' band; on HIP, visit square tile blocks instead so both slices remain in L2/Infinity
  // Cache. This is only a group-id remap: arithmetic and allocations are unchanged.
  int cache_block = 0;
#if defined(MOM_SYCL_HAS_HIP)
  const bool is_hip = mom_is_hip(q.get_device());
  if (is_hip) {
    cache_block = 64;
    if (std::getenv("MOM_PEARLHASH_AMD_DP4A_CACHE_BLOCK")) {
      unsigned long configured = 0;
      if (mom_parse_env_ulong("MOM_PEARLHASH_AMD_DP4A_CACHE_BLOCK", configured) &&
          configured <= static_cast<unsigned long>(std::numeric_limits<int>::max()))
        cache_block = static_cast<int>(configured);
      else
        cache_block = 0;
    }
  }
#endif
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  if constexpr (FP16Matrix) {
    // Reuse the native CUDA cache-order override without changing the default order.
    unsigned long configured = 0;
    if (mom_parse_env_ulong("MOM_PEARLHASH_CU_BLK", configured) &&
        configured <= static_cast<unsigned long>(std::numeric_limits<int>::max()))
      cache_block = static_cast<int>(configured);
  }
#endif
  const bool blocked = cache_block > 1 &&
                       static_cast<uint64_t>(cache_block) * cache_block <=
                         std::numeric_limits<uint32_t>::max() &&
                       tilesH % static_cast<uint32_t>(cache_block) == 0 &&
                       tilesW % static_cast<uint32_t>(cache_block) == 0;
  const uint32_t blocksW = blocked ? tilesW / static_cast<uint32_t>(cache_block) : 0;
  // A full 131072-square 2x2 search contains exactly 2^32 work-items. AdaptiveCpp's HIP launcher
  // cannot represent that as one ND-range (hipErrorInvalidValue), even though the group count is
  // legal. Submit bounded contiguous pieces on the in-order queue and add their group base in the
  // kernel. This also keeps custom larger shapes away from the same 32-bit launcher edge.
  constexpr size_t max_groups_per_launch = size_t{1} << 22;
  for (size_t group_base = 0; group_base < ntiles; group_base += max_groups_per_launch) {
    const size_t group_count = std::min(max_groups_per_launch, ntiles - group_base);
    q.submit([&](sycl::handler& h) {
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
      using Scratch = std::conditional_t<FP16Matrix, PearlF16Scratch, uint32_t>;
      sycl::local_accessor<Scratch, 1> red(FP16Matrix ? 1 : WG * 16, h);
#else
      sycl::local_accessor<uint32_t, 1> red(WG * 16, h);
#endif
      // Explicit capture list (not [=]): keeps the kernel-lambda layout identical across the host, spir64
      // and nvptx passes (the nvptx body is #ifdef'd out below), so the dual-compiler combined build's
      // "Unexpected kernel lambda size" static assert stays satisfied.
      h.parallel_for(
          sycl::nd_range<1>(group_count * WG, WG),
          [B, seed, red, tilesW, k4, rank, nb, dbg, cache_block, blocked, blocksW,
           group_base](sycl::nd_item<1> it) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const uint32_t g = static_cast<uint32_t>(group_base + it.get_group(0));
        uint32_t R, C;
        if (blocked) {
          const uint32_t block_size = static_cast<uint32_t>(cache_block);
          const uint32_t area = block_size * block_size, block = g / area, inner = g % area;
          R = (block / blocksW) * block_size + inner / block_size;
          C = (block % blocksW) * block_size + inner % block_size;
        } else {
          R = g / tilesW;
          C = g % tilesW;
        }
        const int lid = (int)it.get_local_linear_id();
        auto red_word = [&](int index) -> uint32_t& {
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
          if constexpr (FP16Matrix) {
            const uintptr_t raw = reinterpret_cast<uintptr_t>(&red[0].words[0]);
            return reinterpret_cast<uint32_t*>((raw + 31) & ~uintptr_t{31})[index];
          } else
#endif
            return red[index];
        };
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
        const bool f16_matrix = FP16Matrix && pearlhash_uses_wmma_f16();
        const int c4step = f16_matrix ? 4 : 1;
#else
        constexpr int c4step = 1;
#endif
        const int r0 = (lid / BW) * ER, c0 = (lid % BW) * EC; // this work-item's micro-tile origin
        const uint32_t rowBase = R * HT, colBase = C * HT;
        // A'/B' read as int32 words (4 packed int8) straight from global -- the L2 absorbs the per-tile
        // reuse (each 128KB tile slice stays resident), which is why no SLM staging is needed.
        auto A32 = sycl::address_space_cast<sycl::access::address_space::global_space,
                                            sycl::access::decorated::no>(
            reinterpret_cast<const int32_t*>(B.Ap));
        auto B32 = sycl::address_space_cast<sycl::access::address_space::global_space,
                                            sycl::access::decorated::no>(
            reinterpret_cast<const int32_t*>(B.Bp));
        uint32_t acc[ER][EC];
        for (int i = 0; i < ER; i++)
          for (int j = 0; j < EC; j++)
            acc[i][j] = 0;
        uint32_t pX[16];
        for (int w = 0; w < 16; w++)
          pX[w] = 0; // this work-item's partial 16-word transcript
        for (int rc = 0; rc < nb; rc++) {
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
          [[maybe_unused]] float rank_acc[8]{};
#endif
          const int c4lo = rc * (rank / 4), c4hi = c4lo + (rank / 4);
          for (int c4 = c4lo; c4 < c4hi; c4 += c4step) {
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
            bool f16_step = false;
            if constexpr (FP16Matrix)
              if (f16_matrix) {
                for (int part = 0; part < 2; ++part) {
                  const int word = lid + part * WG, row = word / 4, depth = word % 4;
                  const uint32_t aw = static_cast<uint32_t>(
                      A32[(rowBase + static_cast<size_t>(row)) * k4 + c4 + depth]);
                  const uint32_t bw = static_cast<uint32_t>(
                      B32[(colBase + static_cast<size_t>(row)) * k4 + c4 + depth]);
                  red_word(2 * word) =
                      pearlhash_f16x2_s8(static_cast<int8_t>(aw), static_cast<int8_t>(aw >> 8));
                  red_word(2 * word + 1) =
                      pearlhash_f16x2_s8(static_cast<int8_t>(aw >> 16), static_cast<int8_t>(aw >> 24));
                  red_word(128 + 2 * word) =
                      pearlhash_f16x2_s8(static_cast<int8_t>(bw), static_cast<int8_t>(bw >> 8));
                  red_word(128 + 2 * word + 1) =
                      pearlhash_f16x2_s8(static_cast<int8_t>(bw >> 16), static_cast<int8_t>(bw >> 24));
                }
                // This specialization is one full CUDA warp; order its shared writes explicitly.
                sycl::group_barrier(it.get_sub_group(), sycl::memory_scope::sub_group);
                pearlhash_wmma_m16n16k16_f16(rank_acc, &red_word(0), &red_word(128));
                // Consume before the next stage, and before reusing this object for reduction.
                sycl::group_barrier(it.get_sub_group(), sycl::memory_scope::sub_group);
                f16_step = true;
              }
            // Uniform unknown reflection retains exact 2x4 DP4A, not a trap or mixed C mapping.
            if (!f16_step)
#endif
            {
              int32_t bw[EC];
              for (int j = 0; j < EC; j++)
                bw[j] = B32[(colBase + static_cast<size_t>(c0 + j)) * static_cast<size_t>(k4) +
                            static_cast<size_t>(c4)];
              for (int i = 0; i < ER; i++) {
                const int32_t aw =
                    A32[(rowBase + static_cast<size_t>(r0 + i)) * static_cast<size_t>(k4) +
                        static_cast<size_t>(c4)];
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
                if constexpr (Native) {
                  for (int j = 0; j < EC; j++)
                    acc[i][j] = mom::nvidia::dot4_s8<true>(static_cast<uint32_t>(aw),
                                                         static_cast<uint32_t>(bw[j]), acc[i][j]);
                } else
#endif
                {
                  const int a0 = (int8_t)aw, a1 = (int8_t)(aw >> 8), a2 = (int8_t)(aw >> 16),
                            a3 = (int8_t)(aw >> 24);
                  for (int j = 0; j < EC; j++) { // dp4a: 4 int8 MACs (one ALU op on Intel/AMD GPUs)
                    const int32_t b = bw[j];
                    const int product =
                        a0 * static_cast<int8_t>(b) + a1 * static_cast<int8_t>(b >> 8) +
                        a2 * static_cast<int8_t>(b >> 16) + a3 * static_cast<int8_t>(b >> 24);
                    acc[i][j] += static_cast<uint32_t>(product);
                  }
                }
              }
            }
          }
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
          if constexpr (FP16Matrix)
            for (int cell = 0; cell < 8; ++cell)
              // rank<=1024 bounds every partial by 2^24. Convert signed first, then add
              // modulo 2^32; retaining cumulative C in FP32 would change the transcript.
              acc[cell / EC][cell % EC] +=
                  static_cast<uint32_t>(static_cast<int32_t>(rank_acc[cell]));
#endif
          uint32_t lx = 0;
          for (int i = 0; i < ER; i++)
            for (int j = 0; j < EC; j++)
              lx ^= (uint32_t)acc[i][j];
          const int slot = rc % 16;
          pX[slot] = rotl(pX[slot], 13) ^ lx; // cumulative tile XOR folded by transcript slot
        }
        // Single batched work-group reduction: XOR every work-item's 16-word partial transcript together.
        auto grp = it.get_group();
        for (int w = 0; w < 16; w++)
          red_word(lid * 16 + w) = pX[w];
        sycl::group_barrier(grp);
        for (int half = WG >> 1; half > 0; half >>= 1) {
          if (lid < half)
            for (int w = 0; w < 16; w++)
              red_word(lid * 16 + w) ^= red_word((lid + half) * 16 + w);
          sycl::group_barrier(grp);
        }
        if (lid == 0) {
          // The per-work-item folds are linear over XOR, so their reduction is the exact shared
          // rotl-fold of each cumulative tile part used by search_esimd and CUDA-SYCL.
          uint32_t t16[16];
          for (int w = 0; w < 16; w++)
            t16[w] = red_word(w);
          if (dbg) {
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                c(B.result->chk);
            c.fetch_add(tile_mix((uint32_t)rowBase, (uint32_t)colBase, t16));
          }
          uint8_t jackpot[32];
          if (tile_wins(t16, B.cA, B.target, jackpot)) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                a(B.result->found);
            if (a.exchange(1) == 0) {
              B.result->seed = seed;
              B.result->row = (uint32_t)rowBase;
              B.result->col = (uint32_t)colBase;
              for (int i = 0; i < 32; ++i)
                B.result->jackpot[i] = jackpot[i];
            }
          }
        }
      });
    });
  }
}
// Run the portable search at the measured B580 OpenCL 4x4 micro-tile. Retuning changes the typed
// constants above rather than adding a preprocessor variant. SLM staging and software-pipelined prefetch were both measured slower on
// the B580 (the L2 already absorbs the A'/B' reuse; IGC schedules the loads better than a manual pipeline).
static void search(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank,
                   bool dbg) {
#if defined(MOM_SYCL_HAS_HIP)
  // HIP keeps its measured-best 2x2 dp4a path (RX 9060 XT: 750 vs 496 GH/s for 2x2 versus the
  // Intel-tuned 4x4). Non-HIP devices retain their own 4x4 default.
  const bool is_hip = mom_is_hip(q.get_device());
  if (is_hip) {
    // Keep several fully typed dp4a micro-tiles in the architecture-neutral image. This is a host
    // dispatch, so each hot kernel contains no runtime tile branch. MOM_PEARLHASH_AMD_DP4A_TILE is a
    // tuning/validation override; 2x2 remains the measured default until the final AMD SSCP sweep.
    const char* const tile = std::getenv("MOM_PEARLHASH_AMD_DP4A_TILE");
    if (tile && !std::strcmp(tile, "1x1"))
      return search_t<1, 1>(q, bb, seed, m, n, k, rank, dbg);
    if (tile && !std::strcmp(tile, "2x4"))
      return search_t<2, 4>(q, bb, seed, m, n, k, rank, dbg);
    if (tile && !std::strcmp(tile, "4x2"))
      return search_t<4, 2>(q, bb, seed, m, n, k, rank, dbg);
    if (tile && !std::strcmp(tile, "4x4"))
      return search_t<4, 4>(q, bb, seed, m, n, k, rank, dbg);
    if (tile && !std::strcmp(tile, "8x2"))
      return search_t<8, 2>(q, bb, seed, m, n, k, rank, dbg);
    return search_t<2, 2>(q, bb, seed, m, n, k, rank, dbg);
  }
#endif
  search_t<pearlhash_portable_er, pearlhash_portable_ec>(q, bb, seed, m, n, k, rank, dbg);
}
#endif // !MOM_PEARLHASH_ESIMD_TU
