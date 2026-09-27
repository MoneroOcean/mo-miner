#ifdef PEARLHASH_ESIMD
// ---- experimental ESIMD register-resident DPAS search (alternative to search()) ----
// One ESIMD work-item owns an ER x EC grid of 16x16 hash tiles in its own GRF
// (ESIMD sub-group size is 1 -- no lane cooperation). The DPAS accumulators stay in registers and
// the per-rank inner-hash XORs them in-GRF, avoiding the joint_matrix_store->SLM->barrier readback
// that caps search() at ~10 TH/s. xmx::dpas(C,B,A): SystolicDepth=8, RepeatCount=8, int8*int8->i32,
// K=32/call; A=8x32 (simd int8 256), B=32x16 VNNI (simd int8 512, == compute_ab's Bp layout),
// C=8x16 (simd int32 128). B' loads straight from Bp; A' is 8 strided row loads from row-major Ap.
// ER=EC=2 (a 2x2 grid of 16x16 tiles => 8 int32 accumulators) is the B580 sweet spot: it maximizes
// operand reuse (16 MAC/byte) while staying within the GRF -- 9+ accumulators spill and collapse
// throughput (ER=4 or EC=4 measured 5-11 TH/s vs 34 at 2x2). Override at build time if retuning.
// ER=EC=2 gives eight accumulators and is the measured GRF sweet spot without spilling.
constexpr int pearlhash_esimd_columns = 2;
constexpr int pearlhash_esimd_rows = 2;
// Cache-blocked work-item traversal (tile swizzle): instead of row-major wi->(Rg,Cg) -- which
// re-streams ALL of B' for every row-band and goes DRAM-bound once A'/B' exceed L2 (the m=n=131072
// HeroMiners shape: 512MB each, ~2TB of redundant DRAM reads -> 12.6 TH/s) -- iterate a BLK x BLK
// square of work-items per "super-block" so its A'/B' slice (~16MB at BLK=64) stays resident in L2
// and is reused across the block's work-items. Cuts DRAM traffic ~32x -> compute-bound again. BLK=0
// disables (plain row-major). Requires tilesH%BLK==0 && tilesW%BLK==0, else falls back to row-major.
constexpr int pearlhash_esimd_block = 64;
namespace esimd_ns = sycl::ext::intel::esimd;
namespace xmx_ns   = sycl::ext::intel::esimd::xmx;

// In-register XOR tree reduction of simd<uint32_t,N> -> uint32_t (N a power of two).
// esimd::reduce<bit_xor> is unimplemented on this stack, so fold by halving select<>s in GRF.
template <int N>
static inline uint32_t esimd_xor_reduce(esimd_ns::simd<uint32_t, N> v) {
  if constexpr (N == 1) return v[0];
  else {
    constexpr int H = N / 2;
    esimd_ns::simd<uint32_t, H> lo = v.template select<H, 1>(0);
    esimd_ns::simd<uint32_t, H> hi = v.template select<H, 1>(H);
    return esimd_xor_reduce<H>(lo ^ hi);
  }
}

// External in the standalone spir64-only ESIMD TU (esimd.cpp) so the main pearlhash.o can call it;
// static in the single-TU Intel/Windows build where it lives in this same object.
#ifdef MOM_PEARLHASH_ESIMD_TU
void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg) {
#else
static void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg) {
#endif
  constexpr int ER = pearlhash_esimd_rows, EC = pearlhash_esimd_columns, HT = 16;
  auto B = bb;
  const int tilesH = m / (HT * ER), tilesW = n / (HT * EC);
  const size_t nWI = (size_t)tilesH * tilesW;
  constexpr int BLK = pearlhash_esimd_block;
  // L2 cache-blocking: only swizzle into BLK x BLK super-blocks when the grid divides evenly (see
  // the cache-blocking note above); otherwise fall back to plain row-major traversal.
  const bool blocked = BLK > 1 && (tilesW % BLK) == 0 && (tilesH % BLK) == 0;
  const int blocksW = blocked ? tilesW / BLK : 0;
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(nWI), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
      using esimd_ns::simd;
      const int wi = (int)id[0];
      int Rg, Cg;
      if (blocked) {                              // swizzle into BLK x BLK super-blocks for L2 reuse
        const int blk = wi / (BLK * BLK), intra = wi % (BLK * BLK);
        Rg = (blk / blocksW) * BLK + (intra / BLK);
        Cg = (blk % blocksW) * BLK + (intra % BLK);
      } else { Rg = wi / tilesW; Cg = wi % tilesW; }
      const int rowBase = Rg * HT * ER, colBase = Cg * EC;
      const int8_t* Ap = B.Ap;
      const int8_t* Bp = B.Bp;
      simd<int32_t, 128> acc0[ER * EC], acc1[ER * EC];
      MU<ER * EC>([&](auto t) { acc0[t] = 0; acc1[t] = 0; });
      uint32_t tr[ER * EC][16] = {};   // per-tile transcript, XOR/rotl-folded at each rank boundary
      int rc = 0;
      // Prefetch the next K-chunk's A' into cache each iteration: cheap (no GRF, unlike a
      // double-buffered load which spills) and addresses load LATENCY. A' is the latency-critical
      // operand (2*ER scattered 256B fragments); a distance-32 A-only prefetch lifts search
      // 31.5->34.2 TH/s, while prefetching the contiguous reused B' measured no gain. ESIMD prefetch
      // is documented DG2/PVC but works on B580/Xe2; max 256B/call, needs an explicit cache hint.
      constexpr auto PFH = esimd_ns::properties{esimd_ns::cache_hint_L1<esimd_ns::cache_hint::cached>,
                                                esimd_ns::cache_hint_L2<esimd_ns::cache_hint::cached>,
                                                esimd_ns::alignment<256>};
      for (int p = 0; p < k; p += 32) {
        // A' tile-major (compute_ab): the 8x32 fragment for band b, k-chunk p is one 256B block;
        // B' is one contiguous 512B VNNI block reused across all ER row-bands.
        simd<int8_t, 256> a0[ER], a1[ER];
        simd<int8_t, 512> bf[EC];
        MU<ER>([&](auto r) { const int band0 = (rowBase + (int)r * HT) >> 3;
          a0[r] = esimd_ns::block_load<int8_t, 256>(Ap + ((size_t)(band0 * (k / 32) + (p >> 5)) * 256));
          a1[r] = esimd_ns::block_load<int8_t, 256>(Ap + ((size_t)((band0 + 1) * (k / 32) + (p >> 5)) * 256)); });
        MU<EC>([&](auto c) { bf[c] = esimd_ns::block_load<int8_t, 512>(Bp + (size_t)(colBase + (int)c) * k * 16 + (size_t)(p / 4) * 64); });
        if (p + 32 < k) MU<ER>([&](auto r) { const int band0 = (rowBase + (int)r * HT) >> 3;
          esimd_ns::prefetch<int8_t, 256>(Ap, ((size_t)(band0 * (k / 32) + ((p + 32) >> 5)) * 256), PFH);
          esimd_ns::prefetch<int8_t, 256>(Ap, ((size_t)((band0 + 1) * (k / 32) + ((p + 32) >> 5)) * 256), PFH); });
        MU<EC>([&](auto c) { MU<ER>([&](auto r) { const int idx = (int)r * EC + (int)c;
          acc0[idx] = xmx_ns::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(acc0[idx], bf[c], a0[r]);
          acc1[idx] = xmx_ns::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(acc1[idx], bf[c], a1[r]); }); });
        if ((p + 32) % rank == 0) {   // rank-chunk boundary: XOR-fold the cumulative tile into the transcript
          MU<ER * EC>([&](auto t) {
            simd<uint32_t, 128> x = acc0[t].template bit_cast_view<uint32_t>() ^ acc1[t].template bit_cast_view<uint32_t>();
            tr[t][rc % 16] = rotl(tr[t][rc % 16], 13) ^ esimd_xor_reduce<128>(x);
          });
          rc++;
        }
      }
      for (int r = 0; r < ER; r++) for (int c = 0; c < EC; c++) {
        if (dbg) {   // whole-search checksum for cross-validation (see tile_mix); ESIMD atomic add
          const uint32_t mx = tile_mix((uint32_t)(rowBase + r * HT), (uint32_t)((colBase + c) * HT), tr[r * EC + c]);
          esimd_ns::atomic_update<esimd_ns::atomic_op::add>(
              &B.result->chk, esimd_ns::simd<uint32_t, 1>(0), esimd_ns::simd<uint32_t, 1>(mx), esimd_ns::simd_mask<1>(1));
        }
        if (!tile_wins(tr[r * EC + c], B.cA, B.target)) continue;
        // ESIMD has no sycl::atomic_ref; use esimd::atomic_update<xchg> to pick one winner
        esimd_ns::simd<int, 1> old = esimd_ns::atomic_update<esimd_ns::atomic_op::xchg>(
            &B.result->found, esimd_ns::simd<uint32_t, 1>(0), esimd_ns::simd<int, 1>(1), esimd_ns::simd_mask<1>(1));
        if (old[0] == 0) { B.result->seed = seed; B.result->row = (uint32_t)(rowBase + r * HT); B.result->col = (uint32_t)((colBase + c) * HT); }
      }
    });
  });
}
// Alchemist uses SIMD8 DPAS operands: C=8x8, B=32x8 VNNI, A=8x32.
// Two vertically adjacent 16x16 hash tiles retain eight 64-element accumulators.
// DPASW pairs adjacent work-items, each supplying half of the shared A operand.
// B and the cumulative rank-boundary transcripts remain private to each work-item.
constexpr int pearlhash_alchemist_rows = 2;
constexpr int pearlhash_alchemist_columns = 1;
constexpr int pearlhash_alchemist_block_rows = 64;
constexpr int pearlhash_alchemist_block_columns = 64;

template<bool Wide, int AN>
static inline esimd_ns::simd<int32_t, 64> alchemist_mad(
    esimd_ns::simd<int32_t, 64> c, esimd_ns::simd<int8_t, 256> b,
    esimd_ns::simd<int8_t, AN> a) {
  if constexpr (Wide) return xmx_ns::dpasw<8, 8, int32_t, int8_t, int8_t>(c, b, a);
  else return xmx_ns::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(c, b, a);
}

template<int FixedRank, bool Wide>
static sycl::event search_esimd_alchemist_impl(
    sycl::queue& q, const Buffers& bb, uint32_t seed,
    int m, int n, int k, int rank, bool dbg) {
  constexpr int ER = pearlhash_alchemist_rows, EC = pearlhash_alchemist_columns, HT = 16;
  constexpr int BH = pearlhash_alchemist_block_rows, BW = pearlhash_alchemist_block_columns;
  constexpr int AN = Wide ? 128 : 256;
  auto B = bb;
  const int tilesH = m / (HT * ER), tilesW = n / (HT * EC);
  const size_t nWI = size_t(tilesH) * tilesW;
  const bool blocked = tilesW % BW == 0 && tilesH % BH == 0;
  const int blocksW = blocked ? tilesW / BW : 0;
  const size_t localSize = nWI % 32 == 0 ? 32 : (nWI % 2 == 0 ? 2 : 1);
  return q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(nWI, localSize), [=](sycl::nd_item<1> id) SYCL_ESIMD_KERNEL {
      using esimd_ns::simd;
      const int wi = int(id.get_global_id(0));
      // Dispatch ensures that each even/odd pair stays in the same A row band.
      const size_t aHalfOffset = Wide ? (id.get_local_id(0) & 1) * 128 : 0;
      int Rg, Cg;
      if (blocked) {
        const int blk = wi / (BH * BW), intra = wi % (BH * BW);
        Rg = (blk / blocksW) * BH + intra / BW;
        Cg = (blk % blocksW) * BW + intra % BW;
      } else { Rg = wi / tilesW; Cg = wi % tilesW; }
      const int rowBase = Rg * HT * ER, colBase = Cg * EC;
      const int8_t* Ap = B.Ap;
      const int8_t* Bp = B.Bp;
      simd<int32_t, 64> accTL[ER * EC], accTR[ER * EC];
      simd<int32_t, 64> accBL[ER * EC], accBR[ER * EC];
      MU<ER * EC>([&](auto t) { accTL[t] = 0; accTR[t] = 0; accBL[t] = 0; accBR[t] = 0; });
      uint32_t tr[ER * EC][16] = {};
      int rc = 0;
      // Eight K32 steps span the common rank256 interval. Keep a runtime-rank
      // instantiation for other valid jobs, including partial unroll groups.
#pragma unroll 8
      for (int p = 0; p < k; p += 32) {
        simd<int8_t, AN> a0[ER], a1[ER];
        MU<ER>([&](auto r) {
          const int band0 = (rowBase + int(r) * HT) >> 3;
          a0[r] = esimd_ns::block_load<int8_t, AN>(Ap + size_t(band0 * (k / 32) + (p >> 5)) * 256 + aHalfOffset);
          a1[r] = esimd_ns::block_load<int8_t, AN>(Ap + size_t((band0 + 1) * (k / 32) + (p >> 5)) * 256 + aHalfOffset);
        });
        MU<EC>([&](auto c) {
          // Issue both current B loads before consuming either. Do not retain
          // next-step operands: that increased pressure without improving timing.
          simd<int8_t, 256> earlyB[2];
          MU<2>([&](auto half) {
            earlyB[half] = esimd_ns::block_load<int8_t, 256>(
                Bp + size_t(2 * (colBase + int(c)) + int(half)) * k * 8 + size_t(p) * 8);
          });
          MU<2>([&](auto half) {
            simd<int8_t, 256> bf = earlyB[half];
            MU<ER>([&](auto r) {
              constexpr int idx = decltype(r)::value * EC + decltype(c)::value;
              if constexpr (decltype(half)::value == 0) {
                accTL[idx] = alchemist_mad<Wide>(accTL[idx], bf, a0[r]);
                accBL[idx] = alchemist_mad<Wide>(accBL[idx], bf, a1[r]);
              } else {
                accTR[idx] = alchemist_mad<Wide>(accTR[idx], bf, a0[r]);
                accBR[idx] = alchemist_mad<Wide>(accBR[idx], bf, a1[r]);
              }
            });
          });
        });
        if ((p + 32) % (FixedRank ? FixedRank : rank) == 0) {
          MU<ER * EC>([&](auto t) {
            simd<uint32_t, 64> x = accTL[t].template bit_cast_view<uint32_t>() ^
                accTR[t].template bit_cast_view<uint32_t>() ^ accBL[t].template bit_cast_view<uint32_t>() ^
                accBR[t].template bit_cast_view<uint32_t>();
            tr[t][rc % 16] = rotl(tr[t][rc % 16], 13) ^ esimd_xor_reduce<64>(x);
          });
          rc++;
        }
      }
      for (int r = 0; r < ER; r++) for (int c = 0; c < EC; c++) {
        if (dbg) {
          const uint32_t mx = tile_mix(uint32_t(rowBase + r * HT), uint32_t((colBase + c) * HT), tr[r * EC + c]);
          esimd_ns::atomic_update<esimd_ns::atomic_op::add>(
              &B.result->chk, simd<uint32_t, 1>(0), simd<uint32_t, 1>(mx), esimd_ns::simd_mask<1>(1));
        }
        if (!tile_wins(tr[r * EC + c], B.cA, B.target)) continue;
        simd<int, 1> old = esimd_ns::atomic_update<esimd_ns::atomic_op::xchg>(
            &B.result->found, simd<uint32_t, 1>(0), simd<int, 1>(1), esimd_ns::simd_mask<1>(1));
        if (old[0] == 0) {
          B.result->seed = seed;
          B.result->row = uint32_t(rowBase + r * HT);
          B.result->col = uint32_t((colBase + c) * HT);
        }
      }
    });
  });
}

// External linkage also supports the combined build's separate spir64 ESIMD TU.
sycl::event search_esimd_alchemist(sycl::queue& q, const Buffers& bb, uint32_t seed,
                                  int m, int n, int k, int rank, bool dbg) {
  // Odd-width grids cannot pair without crossing an A row band. Both kernels
  // consume the same SIMD8 B packing; ordinary DPAS loads the complete A tile.
  const int tilesW = n / (16 * pearlhash_alchemist_columns);
  if (tilesW % 2 == 0) {
    if (rank == 256) return search_esimd_alchemist_impl<256, true>(q, bb, seed, m, n, k, rank, dbg);
    return search_esimd_alchemist_impl<0, true>(q, bb, seed, m, n, k, rank, dbg);
  }
  if (rank == 256) return search_esimd_alchemist_impl<256, false>(q, bb, seed, m, n, k, rank, dbg);
  return search_esimd_alchemist_impl<0, false>(q, bb, seed, m, n, k, rank, dbg);
}
#endif  // PEARLHASH_ESIMD


// In the combined build the ESIMD search lives in the separate spir64-only TU (esimd.cpp);
// declare it here so attempt() can call it. (When PEARLHASH_ESIMD is set this is the same TU, defined above.)
#if defined(MOM_PEARLHASH_HAS_ESIMD) && !defined(PEARLHASH_ESIMD)
void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg);
sycl::event search_esimd_alchemist(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank, bool dbg);
#endif

