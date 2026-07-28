#if defined(MOM_SYCL_HAS_CUDA)
// ---- NVIDIA tensor-core int8 mma.sync search (DPC++ CUDA backend) ----
// Built with intel/llvm DPC++ for nvptx64 (-fsycl-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_89),
// where joint_matrix maps to the L4's int8 Tensor Cores (wmma.mma.sync IMMA). NVIDIA constraints
// differ from Intel XMX: fixed WMMA shape m16n16k16 (one 16x16 hash tile == one accumulator), only
// row_major/col_major layouts (NO ext_intel_packed VNNI -- compute_ab writes B' row-major here), and
// a 32-lane warp per matrix op. One sub-group (warp) owns one 16x16 hash tile; the cumulative tile is
// XOR-folded into the transcript every `rank` columns via joint_matrix_apply (per-lane, in-register)
// + a sub-group XOR reduce -- no SLM readback (the bottleneck that capped Intel joint_matrix). The
// XOR of all 256 cells is order-independent, so the NVIDIA fragment distribution needs no remap; the
// jackpot is bit-identical to the reference given identical A'/B'.
// Per-warp register tile: each warp computes an ER x EC grid of 16x16 hash tiles, reusing each A'
// row-fragment across EC columns and each B' col-fragment across ER rows (raises MAC/byte so the
// mma issue rate, not the loads, bounds throughput -- the naive 1-tile/warp kernel was load-bound at
// ~6 TH/s). The block swizzle groups warps into BLK x BLK super-blocks so each block's A'/B' slice stays
// L2-resident (the m=n=131072 shape is 512MB/operand -> DRAM-bound without it). Tunable at build.
// 4x4 register tile gives the best MAC/byte reuse without accumulator spill. The block floor is
// generic rather than L4-specific; runtime cache sizing may select a larger value below.
constexpr int pearlhash_cuda_er = 4;
constexpr int pearlhash_cuda_ec = 4;
constexpr int pearlhash_cuda_block_floor = 16;
// One warp computes an ER x EC grid of 16x16 hash tiles using the int8 Tensor-Core MMA
// mma.sync.aligned.m16n8k32 (a 16x16 tile = two n=8 halves). The accumulator stays in registers and
// the inner-hash reads those registers DIRECTLY -- unlike joint_matrix_apply, a plain register read
// does not spill the accumulators / pessimize the GEMM (that capped the joint_matrix path at ~20).
// This is the NVIDIA analog of the Intel ESIMD dpas path. A' is row-major, B' is COLUMN-major (so
// each thread's 4 contiguous int8 of a fragment is one aligned 32-bit load). The super-block
// swizzle keeps the A'/B' slice L2-resident at the 131072 shape.
//
// mma.m16n8k32.s8 register fragment layout (PTX ISA): lane -> gid=lane/4, tid=lane%4.
//   A(16x32,row): a[0]=A[gid][tid*4..],   a[1]=A[gid+8][tid*4..],   a[2]=A[gid][16+tid*4..],   a[3]=A[gid+8][16+tid*4..]
//   B(32x8 ,col): b[0]=B[tid*4..][gid],   b[1]=B[16+tid*4..][gid]
//   C(16x8 ,row): c[0]=C[gid][tid*2], c[1]=C[gid][tid*2+1], c[2]=C[gid+8][tid*2], c[3]=C[gid+8][tid*2+1]
static inline void mma_m16n8k32_s8(int32_t c[4], const uint32_t a[4], const uint32_t b[2]) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  (void)c; (void)a; (void)b;   // host pass: never executed
#endif
}
static inline uint32_t ld_u32(const int8_t* __restrict__ p) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  uint32_t value;
  // PearlHash's A/B matrices are immutable for the whole search. Route fragment reads through CUDA's
  // non-coherent read path and prefetch one nearby 128-byte sector into L2; the blocked traversal
  // consumes that sector in adjacent warps. This is the NVIDIA counterpart of Intel's ESIMD LSC
  // cache hints, and avoids changing the portable/ESIMD/HIP kernels.
  asm volatile("ld.global.nc.L2::128B.u32 %0, [%1];\n" : "=r"(value) : "l"(p));
  return value;
#else
  return *reinterpret_cast<const uint32_t*>(p);
#endif
}
// cp.async helpers (PEARLHASH_CU_PIPE pipeline): async global->shared 16-byte copies that overlap with
// mma, plus the generic->shared address conversion cp.async needs. Device-only (inline PTX).
static inline uint32_t cu_smem_addr(const void* p) {
  uint32_t a = 0;
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
#else
  (void)p;
#endif
  return a;
}
static inline void cu_cp_async16(uint32_t smem, const int8_t* g) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(g));
#else
  (void)smem; (void)g;
#endif
}
static inline void cu_cp_commit() {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile("cp.async.commit_group;\n");
#endif
}
template <int N> static inline void cu_cp_wait() {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
static inline void cu_ldmatrix_x4(uint32_t out[4], uint32_t smem) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]) : "r"(smem));
#else
  (void)out; (void)smem;
#endif
}
static inline void cu_ldmatrix_x2(uint32_t out[2], uint32_t smem) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
    : "=r"(out[0]), "=r"(out[1]) : "r"(smem));
#else
  (void)out; (void)smem;
#endif
}
// CUTE Swizzle<3,4,3>: XOR logical byte-address bits 7..9 into bits 4..6. The low four
// bits stay unchanged, so every cp.async destination remains naturally 16-byte aligned.
static inline int cu_swizzle_128(int off) { return off ^ ((off & 0x380) >> 3); }

// DEFAULT NVIDIA path: one warp per ER x EC grid of 16x16 hash tiles, mma.sync m16n8k32 int8, operands
// from global (L2-resident via the BLK super-block swizzle), accumulators resident in registers
// (pearlhash's inner-hash needs CUMULATIVE C, so tiles cannot be evicted like a normal GEMM), inner-hash
// folds them per rank via direct register reads. ~34 TH/s on an L4 -- correct and the best working
// config; occupancy-bound by the resident-accumulator register tile.
static void search_cuda_direct(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank) {
  constexpr int SG = 32, ER = pearlhash_cuda_er, EC = pearlhash_cuda_ec;
  auto B = bb;
  const int tilesH = m / (16 * ER), tilesW = n / (16 * EC);
  // BLK super-block size: pick the largest power-of-two whose A'/B' operand slice fits the usable L2
  // budget. On <=32 MiB caches reserve half for the concurrent accumulator/transcript traffic; using
  // all 32 MiB regresses RTX 5060 Ti by ~3%. Larger caches can retain the full slice (L4's 48 MiB L2
  // still selects 64, +15% vs the old fixed 16). Small-L2 cards stay at the typed block floor.
  // Slice ~= BLK*16*k*(ER+EC) int8 bytes. MOM_PEARLHASH_CU_BLK overrides. BLK must evenly divide both
  // tile dimensions or the super-block swizzle disables (falls back to row-major).
  int BLK = pearlhash_cuda_block_floor;
  if (const char* e = std::getenv("MOM_PEARLHASH_CU_BLK")) {
    BLK = std::atoi(e); if (BLK < 1) BLK = pearlhash_cuda_block_floor;
  } else {
    const size_t l2 = q.get_device().get_info<sycl::info::device::global_mem_cache_size>();
    const size_t cache_budget = l2 <= (size_t(32) << 20) ? l2 / 2 : l2;
    const size_t per_blk = (size_t)16 * k * (ER + EC);   // operand bytes per unit of BLK
    for (int cand = 128; cand > pearlhash_cuda_block_floor; cand >>= 1)
      if ((size_t)cand * per_blk <= cache_budget && (tilesW % cand) == 0 && (tilesH % cand) == 0) { BLK = cand; break; }
  }
  const size_t nWarp = (size_t)tilesH * tilesW;
  const bool blocked = BLK > 1 && (tilesW % BLK) == 0 && (tilesH % BLK) == 0;
  const int blocksW = blocked ? tilesW / BLK : 0;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> trbuf(sycl::range<1>(ER * EC * 16), h);  // per-warp transcripts in SLM
    h.parallel_for(sycl::nd_range<1>(nWarp * SG, SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
      auto sg = it.get_sub_group();
      const int lane = (int)sg.get_local_linear_id(), gid = lane >> 2, tid = lane & 3;
      const int warp = (int)it.get_group(0);
      int Rg, Cg;
      if (blocked) { const int blk = warp / (BLK * BLK), intra = warp % (BLK * BLK);
        Rg = (blk / blocksW) * BLK + (intra / BLK); Cg = (blk % blocksW) * BLK + (intra % BLK); }
      else { Rg = warp / tilesW; Cg = warp % tilesW; }
      const int rowBase = Rg * 16 * ER, colBase = Cg * 16 * EC;
      const int8_t* __restrict__ Ap = B.Ap; const int8_t* __restrict__ Bp = B.Bp;   // A' row-major (i*k+c), B' col-major (j*k+c)
      int32_t acc[ER * EC * 2][4];
#pragma unroll
      for (int i = 0; i < ER * EC * 2; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0; }
      if (lane == 0)
#pragma unroll
        for (int i = 0; i < ER * EC * 16; i++) trbuf[i] = 0;
      int rc = 0;
      for (int p = 0; p < k; p += 32) {
        uint32_t ra[ER][4];
#pragma unroll
        for (int r = 0; r < ER; r++) {
          const size_t r0 = (size_t)(rowBase + r * 16 + gid) * k + p, r8 = (size_t)(rowBase + r * 16 + gid + 8) * k + p;
          ra[r][0] = ld_u32(Ap + r0 + tid * 4);      ra[r][1] = ld_u32(Ap + r8 + tid * 4);
          ra[r][2] = ld_u32(Ap + r0 + 16 + tid * 4); ra[r][3] = ld_u32(Ap + r8 + 16 + tid * 4);
        }
#pragma unroll
        for (int c = 0; c < EC; c++)
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            const size_t cb = (size_t)(colBase + c * 16 + hh * 8 + gid) * k + p;   // col-major B' column
            uint32_t rb[2] = { ld_u32(Bp + cb + tid * 4), ld_u32(Bp + cb + 16 + tid * 4) };
#pragma unroll
            for (int r = 0; r < ER; r++) mma_m16n8k32_s8(acc[(r * EC + c) * 2 + hh], ra[r], rb);
          }
        if ((p + 32) % rank == 0) {     // rank-chunk boundary: XOR-fold cumulative tiles (register read)
#pragma unroll
          for (int t = 0; t < ER * EC; t++) {
            uint32_t part = (uint32_t)acc[t * 2][0] ^ (uint32_t)acc[t * 2][1] ^ (uint32_t)acc[t * 2][2] ^ (uint32_t)acc[t * 2][3]
                          ^ (uint32_t)acc[t * 2 + 1][0] ^ (uint32_t)acc[t * 2 + 1][1] ^ (uint32_t)acc[t * 2 + 1][2] ^ (uint32_t)acc[t * 2 + 1][3];
            part = sycl::reduce_over_group(sg, part, sycl::bit_xor<uint32_t>{});
            if (lane == 0) trbuf[t * 16 + rc % 16] = rotl(trbuf[t * 16 + rc % 16], 13) ^ part;
          }
          rc++;
        }
      }
      if (lane == 0) {
        for (int r = 0; r < ER; r++) for (int c = 0; c < EC; c++) {
          const int t = r * EC + c;
          uint32_t full[16];
#pragma unroll
          for (int s = 0; s < 16; s++) full[s] = trbuf[t * 16 + s];
          if (!tile_wins(full, B.cA, B.target)) continue;
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> at(B.result->found);
          if (at.exchange(1) == 0) { B.result->seed = seed; B.result->row = (uint32_t)(rowBase + r * 16); B.result->col = (uint32_t)(colBase + c * 16); }
        }
      }
    });
  });
}
// Blackwell path: SMEM-staged + cp.async double-buffered mma.sync pipeline.
// Eight warps cover a 128x256 output tile and stage 128 K values at a time in 96 KiB of swizzled
// local memory. Double-buffered cp.async overlaps the next global-memory slice with the current
// mma.sync work, while ldmatrix feeds each warp's 64x64 accumulator tile. The 16 transcript words
// are split across lane pairs (eight registers per lane), avoiding a second local-memory surface.
// This deliberately trades occupancy for operand reuse and is faster only on Blackwell.
constexpr int pearlhash_cuda_warp_rows = 2;
constexpr int pearlhash_cuda_warp_columns = 4;
constexpr int pearlhash_cuda_tile_rows = 4;
constexpr int pearlhash_cuda_tile_columns = 4;
// K staged per SMEM chunk; multiple of 16 for cp.async and 32 for the matrix K dimension.
constexpr int pearlhash_cuda_block_k = 128;
static void search_cuda_pipe(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank) {
  constexpr int WM = pearlhash_cuda_warp_rows, WN = pearlhash_cuda_warp_columns,
                TM = pearlhash_cuda_tile_rows, TN = pearlhash_cuda_tile_columns,
                BK = pearlhash_cuda_block_k;
  constexpr int NW = WM * WN, TBSZ = NW * 32, BM = WM * TM * 16, BN = WN * TN * 16;
  constexpr int AB = BM * BK, BB = BN * BK, A16 = AB / 16, B16 = BB / 16;
  auto B = bb;
  const int mB = m / BM, nB = n / BN, nChunks = k / BK;
  const size_t nBlocks = (size_t)mB * nB;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int8_t, 1> As(sycl::range<1>(2 * AB), h);                 // double-buffered A' [2][BM][BK]
    sycl::local_accessor<int8_t, 1> Bs(sycl::range<1>(2 * BB), h);                 // double-buffered B' [2][BN][BK] (col-major B')
    h.parallel_for(sycl::nd_range<1>(nBlocks * TBSZ, TBSZ), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
      auto grp = it.get_group();
      const int tid = (int)it.get_local_linear_id();
      const int warp = tid >> 5, lane = tid & 31;
      const int wM = warp / WN, wN = warp % WN;
      // CUDA's dim3(mBlocks,nBlocks) traversal varies x/row fastest, preserving the twice-larger
      // B slice while successive row blocks stream their smaller A slices.
      const int block = (int)it.get_group(0), bR = block % mB, bC = block / mB;
      const int rowBlock = bR * BM, colBlock = bC * BN;
      const int8_t* __restrict__ Ap = B.Ap; const int8_t* __restrict__ Bp = B.Bp;
      int8_t* Asp = As.get_multi_ptr<sycl::access::decorated::no>().get();
      int8_t* Bsp = Bs.get_multi_ptr<sycl::access::decorated::no>().get();
      uint32_t tr[8] = {};
      int32_t acc[TM * TN * 2][4];
#pragma unroll
      for (int i = 0; i < TM * TN * 2; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0; }
      const int warpRow = wM * TM * 16, warpCol = wN * TN * 16;
      auto load_chunk = [&](int buf, int p) {     // async-copy k-chunk at offset p into SMEM buffer buf
        int8_t* aDst = Asp + buf * AB; int8_t* bDst = Bsp + buf * BB;
        for (int ci = tid; ci < A16; ci += TBSZ) { const int mm = (ci * 16) / BK, kk = (ci * 16) % BK;
          cu_cp_async16(cu_smem_addr(aDst + cu_swizzle_128(mm * BK + kk)), Ap + (size_t)(rowBlock + mm) * k + p + kk); }
        for (int ci = tid; ci < B16; ci += TBSZ) { const int nn = (ci * 16) / BK, kk = (ci * 16) % BK;
          cu_cp_async16(cu_smem_addr(bDst + cu_swizzle_128(nn * BK + kk)), Bp + (size_t)(colBlock + nn) * k + p + kk); }
        cu_cp_commit();
      };
      load_chunk(0, 0);     // prologue
      int rc = 0;
      for (int c = 0; c < nChunks; c++) {
        const int buf = c & 1;
        if (c + 1 < nChunks) load_chunk((c + 1) & 1, (c + 1) * BK);   // prefetch next chunk (overlaps the mma below)
        cu_cp_wait<1>();              // current chunk's copy done (keep the prefetch in flight)
        sycl::group_barrier(grp);     // make the cp.async writes visible to all warps
        const int8_t* Ab = Asp + buf * AB; const int8_t* Bb = Bsp + buf * BB;
#pragma unroll
        for (int ks = 0; ks < BK; ks += 32) {
          uint32_t ra[TM][4], rb[TN][2][2];
#pragma unroll
          for (int mt = 0; mt < TM; mt++) {
            const int logical = (warpRow + mt * 16 + (lane >> 1)) * BK + ks + (lane & 1) * 16;
            cu_ldmatrix_x4(ra[mt], cu_smem_addr(Ab + cu_swizzle_128(logical)));
          }
#pragma unroll
          for (int nt = 0; nt < TN; nt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
              const int logical = (warpCol + nt * 16 + hh * 8 + ((lane >> 1) & 7)) * BK + ks + (lane & 1) * 16;
              cu_ldmatrix_x2(rb[nt][hh], cu_smem_addr(Bb + cu_swizzle_128(logical)));
            }
#pragma unroll
          for (int nt = 0; nt < TN; nt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++)
#pragma unroll
              for (int mt = 0; mt < TM; mt++)
                mma_m16n8k32_s8(acc[(mt * TN + nt) * 2 + hh], ra[mt], rb[nt][hh]);
          if (((c * BK + ks) + 32) % rank == 0) {     // cumulative k hit a rank boundary -> fold
#pragma unroll
            for (int t = 0; t < TM * TN; t++) {
              uint32_t part = (uint32_t)acc[t*2][0] ^ (uint32_t)acc[t*2][1] ^ (uint32_t)acc[t*2][2] ^ (uint32_t)acc[t*2][3]
                            ^ (uint32_t)acc[t*2+1][0] ^ (uint32_t)acc[t*2+1][1] ^ (uint32_t)acc[t*2+1][2] ^ (uint32_t)acc[t*2+1][3];
              part = sycl::reduce_over_group(it.get_sub_group(), part, sycl::bit_xor<uint32_t>{});
              if (lane == t * 2 + (rc & 1)) tr[rc >> 1] = part;
            }
            rc++;
          }
        }
        sycl::group_barrier(grp);   // mma done reading this buffer before it is reused two chunks later
      }
      uint32_t other[8];
#pragma unroll
      for (int i = 0; i < 8; i++) other[i] = sycl::select_from_group(it.get_sub_group(), tr[i], lane ^ 1);
      if ((lane & 1) == 0) {
        const int t = lane >> 1, mt = t / TN, nt = t % TN;
        uint32_t full[16];
#pragma unroll
        for (int i = 0; i < 8; i++) { full[i * 2] = tr[i]; full[i * 2 + 1] = other[i]; }
        if (tile_wins(full, B.cA, B.target)) {
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> at(B.result->found);
          if (at.exchange(1) == 0) { B.result->seed = seed; B.result->row = (uint32_t)(rowBlock + warpRow + mt * 16); B.result->col = (uint32_t)(colBlock + warpCol + nt * 16); }
        }
      }
    });
  });
}
static void search_cuda(sycl::queue& q, const Buffers& b, uint32_t seed, int m, int n, int k, int rank) {
  const std::string name = q.get_device().get_info<sycl::info::device::name>();
  // RTX 50 / Blackwell benefits from operand sharing across eight warps. Ada and older retain the
  // direct L2 path, which is faster there. The distributed transcript maps exactly 16 rank chunks.
  const bool blackwell = name.find("RTX 50") != std::string::npos || name.find("Blackwell") != std::string::npos;
  if (blackwell && k == 4096 && rank == 256 && m % 128 == 0 && n % 256 == 0)
    search_cuda_pipe(q, b, seed, m, n, k, rank);
  else
    search_cuda_direct(q, b, seed, m, n, k, rank);
}
#endif  // MOM_SYCL_HAS_CUDA
