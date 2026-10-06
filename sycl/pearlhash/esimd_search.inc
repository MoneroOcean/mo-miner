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
// throughput; ER=4 or EC=4 measured far slower than 2x2. Retain the qualified 256-GRF request
// on eight-column DPAS devices; source vector sizes alone do not establish actual allocation.
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

// In-register XOR tree reduction of simd<uint32_t,N> -> uint32_t (N a power of two).
// esimd::reduce<bit_xor> is unimplemented on this stack, so fold by halving select<>s in GRF.
template <int N> static inline uint32_t esimd_xor_reduce(esimd_ns::simd<uint32_t, N> v) {
  if constexpr (N == 1)
    return v[0];
  else {
    constexpr int H = N / 2;
    esimd_ns::simd<uint32_t, H> lo = v.template select<H, 1>(0);
    esimd_ns::simd<uint32_t, H> hi = v.template select<H, 1>(H);
    return esimd_xor_reduce<H>(lo ^ hi);
  }
}

template <unsigned Width>
static void search_esimd_impl(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n,
                               int k, int rank, bool dbg) {
  static_assert(Width == 8 || Width == 16);
  constexpr int ER = pearlhash_esimd_rows, EC = pearlhash_esimd_columns, HT = 16;
  // Keep accumulators in native column banks: narrow DPAS must not gather/scatter an 8x16
  // accumulator on every call. Proof tiles and operand reuse remain 16x16 and ER x EC.
  constexpr int CB = HT / Width, NW = 8 * Width;
  auto B = bb;
  const int tilesH = m / (HT * ER), tilesW = n / (HT * EC);
  const size_t nWI = (size_t)tilesH * tilesW;
  constexpr int BLK = pearlhash_esimd_block;
  // L2 cache-blocking: only swizzle into BLK x BLK super-blocks when the grid divides evenly (see
  // the cache-blocking note above); otherwise fall back to plain row-major traversal.
  const bool blocked = BLK > 1 && (tilesW % BLK) == 0 && (tilesH % BLK) == 0;
  const int blocksW = blocked ? tilesW / BLK : 0;
  q.submit([&](sycl::handler& h) {
    const auto kernel = [=](sycl::id<1> id) SYCL_ESIMD_KERNEL MOM_SYCL_KERNEL_ARGS_RESTRICT {
      using esimd_ns::simd;
      const int wi = (int)id[0];
      int Rg, Cg;
      if (blocked) { // swizzle into BLK x BLK super-blocks for L2 reuse
        const int blk = wi / (BLK * BLK), intra = wi % (BLK * BLK);
        Rg = (blk / blocksW) * BLK + (intra / BLK);
        Cg = (blk % blocksW) * BLK + (intra % BLK);
      } else {
        Rg = wi / tilesW;
        Cg = wi % tilesW;
      }
      const int rowBase = Rg * HT * ER, colBase = Cg * EC;
      const int8_t* Ap = B.Ap;
      const int8_t* Bp = B.Bp;
      simd<int32_t, NW> acc0[ER * EC * CB], acc1[ER * EC * CB];
      MU<ER * EC * CB>([&](auto t) {
        acc0[t] = 0;
        acc1[t] = 0;
      });
      uint32_t tr[ER * EC][16] = {}; // per-tile transcript, XOR/rotl-folded at each rank boundary
      int rc = 0;
      // Prefetch the next K-chunk's A' into cache each iteration: cheap (no GRF, unlike a
      // double-buffered load which spills) and addresses load LATENCY. A' is the latency-critical
      // operand (2*ER scattered 256B fragments); a distance-32 A-only prefetch materially improved
      // search throughput, while prefetching the contiguous reused B' measured no gain. ESIMD
      // prefetch is documented for DG2/PVC but works on B580/Xe2; max 256B/call, with an explicit
      // cache hint.
      constexpr auto PFH = esimd_ns::properties{
          esimd_ns::cache_hint_L1<esimd_ns::cache_hint::cached>,
          esimd_ns::cache_hint_L2<esimd_ns::cache_hint::cached>, esimd_ns::alignment<256>};
      for (int p = 0; p < k; p += 32) {
        // A' tile-major (compute_ab): the 8x32 fragment for band b, k-chunk p is one 256B block;
        // B' is one contiguous 512B VNNI block reused across all ER row-bands.
        simd<int8_t, 256> a0[ER], a1[ER];
        simd<int8_t, 32 * Width> bf[EC * CB];
        MU<ER>([&](auto r) {
          const int band0 = (rowBase + (int)r * HT) >> 3;
          a0[r] =
              esimd_ns::block_load<int8_t, 256>(Ap + ((size_t)(band0 * (k / 32) + (p >> 5)) * 256));
          a1[r] = esimd_ns::block_load<int8_t, 256>(
              Ap + ((size_t)((band0 + 1) * (k / 32) + (p >> 5)) * 256));
        });
        MU<EC>([&](auto c) {
          auto packed = esimd_ns::block_load<int8_t, 512>(
              Bp + (size_t)(colBase + (int)c) * k * 16 + (size_t)(p / 4) * 64);
          if constexpr (Width == 16) {
            bf[c] = packed;
          } else {
            // Split VNNI columns once per K-chunk, then reuse across both A halves and ER rows.
            MU<CB>([&](auto column) {
              auto words = packed.template bit_cast_view<int32_t, 8, 16>()
                               .template select<8, 1, Width, 1>(0, (int)column * Width).read();
              bf[(int)c * CB + (int)column] = words.template bit_cast_view<int8_t>().read();
            });
          }
        });
        if (p + 32 < k)
          MU<ER>([&](auto r) {
            const int band0 = (rowBase + (int)r * HT) >> 3;
            esimd_ns::prefetch<int8_t, 256>(
                Ap, ((size_t)(band0 * (k / 32) + ((p + 32) >> 5)) * 256), PFH);
            esimd_ns::prefetch<int8_t, 256>(
                Ap, ((size_t)((band0 + 1) * (k / 32) + ((p + 32) >> 5)) * 256), PFH);
          });
        MU<EC>([&](auto c) {
          MU<ER>([&](auto r) {
            const int idx = (int)r * EC + (int)c;
            MU<CB>([&](auto column) {
              const int bank = idx * CB + (int)column;
              acc0[bank] = esimd_ns::xmx::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(
                  acc0[bank], bf[(int)c * CB + (int)column], a0[r]);
            });
            MU<CB>([&](auto column) {
              const int bank = idx * CB + (int)column;
              acc1[bank] = esimd_ns::xmx::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(
                  acc1[bank], bf[(int)c * CB + (int)column], a1[r]);
            });
          });
        });
        if ((p + 32) % rank ==
            0) { // rank-chunk boundary: XOR-fold the cumulative tile into the transcript
          MU<ER * EC>([&](auto t) {
            uint32_t folded = 0;
            MU<CB>([&](auto column) {
              const int bank = (int)t * CB + (int)column;
              simd<uint32_t, NW> x = acc0[bank].template bit_cast_view<uint32_t>() ^
                                      acc1[bank].template bit_cast_view<uint32_t>();
              folded ^= esimd_xor_reduce<NW>(x);
            });
            // XOR all native banks before the single per-tile transcript rotation.
            tr[t][rc % 16] = rotl(tr[t][rc % 16], 13) ^ folded;
          });
          rc++;
        }
      }
      for (int r = 0; r < ER; r++)
        for (int c = 0; c < EC; c++) {
          if (dbg) { // whole-search checksum for cross-validation (see tile_mix); ESIMD atomic add
            const uint32_t mx = tile_mix((uint32_t)(rowBase + r * HT),
                                         (uint32_t)((colBase + c) * HT), tr[r * EC + c]);
            esimd_ns::atomic_update<esimd_ns::atomic_op::add>(
                &B.result->chk, esimd_ns::simd<uint32_t, 1>(0), esimd_ns::simd<uint32_t, 1>(mx),
                esimd_ns::simd_mask<1>(1));
          }
          uint8_t jackpot[32];
          if (!tile_wins(tr[r * EC + c], B.cA, B.target, jackpot))
            continue;
          // ESIMD has no sycl::atomic_ref; use esimd::atomic_update<xchg> to pick one winner
          esimd_ns::simd<int, 1> old = esimd_ns::atomic_update<esimd_ns::atomic_op::xchg>(
              &B.result->found, esimd_ns::simd<uint32_t, 1>(0), esimd_ns::simd<int, 1>(1),
              esimd_ns::simd_mask<1>(1));
          if (old[0] == 0) {
            B.result->seed = seed;
            B.result->row = (uint32_t)(rowBase + r * HT);
            B.result->col = (uint32_t)((colBase + c) * HT);
            for (int i = 0; i < 32; ++i)
              B.result->jackpot[i] = jackpot[i];
          }
        }
    };
#if defined(MOM_PEARLHASH_INTEL_LARGE_GRF)
    if constexpr (Width == 8) {
      namespace oneapi_exp = sycl::ext::oneapi::experimental;
      oneapi_exp::parallel_for(
          h, oneapi_exp::launch_config{
                 sycl::range<1>(nWI),
                 oneapi_exp::properties{sycl::ext::intel::experimental::grf_size<256>}},
          kernel);
    } else
#endif
    {
      h.parallel_for(sycl::range<1>(nWI), kernel);
    }
  });
}

// External in the standalone spir64-only TU; static in the single-TU Intel/Windows build.
#ifndef MOM_PEARLHASH_ESIMD_TU
static
#endif
unsigned esimd_search_width(const sycl::device& device) noexcept {
  try {
    return mom::intel_matrix::dpas_width(device);
  } catch (...) {
    return 0;
  }
}

#ifndef MOM_PEARLHASH_ESIMD_TU
static
#endif
void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank,
                  bool dbg, const unsigned width) {
  if (width == 8u)
    search_esimd_impl<8>(q, bb, seed, m, n, k, rank, dbg);
  else if (width == 16u)
    search_esimd_impl<16>(q, bb, seed, m, n, k, rank, dbg);
  else
    throw std::string("PearlHash ESIMD search requires a qualified DPAS width");
}
#endif // PEARLHASH_ESIMD

// In the combined build the ESIMD search lives in the separate spir64-only TU (esimd.cpp);
// declare it here so attempt() can call it. (When PEARLHASH_ESIMD is set this is the same TU, defined above.)
#if defined(MOM_PEARLHASH_HAS_ESIMD) && !defined(PEARLHASH_ESIMD)
unsigned esimd_search_width(const sycl::device& device) noexcept;
void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank,
                  bool dbg, unsigned width);
#endif
