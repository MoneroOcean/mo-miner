// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

// Adapted from friskens' Alchemist PearlHash search: https://github.com/MoneroOcean/mo-miner/pull/3.
// The combined AOT profile keeps ordinary ESIMD: Xe2 cannot compile a DPASW entry even if unselected.
#if defined(PEARLHASH_ESIMD) && !defined(MOM_PEARLHASH_ESIMD_TU)
static bool dpasw_esimd_device(const sycl::device& device) noexcept {
#if defined(SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE)
  try {
    namespace exp = sycl::ext::oneapi::experimental;
    const auto arch = device.get_info<exp::info::device::architecture>();
    return arch == exp::architecture::intel_gpu_dg2_g10 ||
           arch == exp::architecture::intel_gpu_dg2_g11 ||
           arch == exp::architecture::intel_gpu_dg2_g12;
  } catch (...) {
    return false;
  }
#else
  (void)device;
  return false;
#endif
}

// DG2 SIMD8: C=8x8, B=32x8 VNNI, A=8x32; DPASW pairs share A.
constexpr int pearlhash_dpasw_rows = 2;
constexpr int pearlhash_dpasw_columns = 1;
constexpr int pearlhash_dpasw_block_rows = 64;
constexpr int pearlhash_dpasw_block_columns = 64;

template <int AN>
static inline esimd_ns::simd<int32_t, 64> dpasw_mad(
    esimd_ns::simd<int32_t, 64> c, esimd_ns::simd<int8_t, 256> b,
    esimd_ns::simd<int8_t, AN> a) {
  return esimd_ns::xmx::dpasw<8, 8, int32_t, int8_t, int8_t>(c, b, a);
}

template <int FixedRank>
static sycl::event search_esimd_dpasw_impl(
    sycl::queue& q, const Buffers& bb, uint32_t seed,
    int m, int n, int k, int rank, bool dbg) {
  constexpr int ER = pearlhash_dpasw_rows, EC = pearlhash_dpasw_columns, HT = 16;
  constexpr int BH = pearlhash_dpasw_block_rows, BW = pearlhash_dpasw_block_columns;
  constexpr int AN = 128;
  auto B = bb;
  const int tilesH = m / (HT * ER), tilesW = n / (HT * EC);
  const size_t nWI = size_t(tilesH) * tilesW;
  const bool blocked = tilesW % BW == 0 && tilesH % BH == 0;
  const int blocksW = blocked ? tilesW / BW : 0;
  // Validated N is a multiple of 32: every row has an even number of paired column tiles.
  const size_t localSize = nWI % 32 == 0 ? 32 : 2;
  return q.submit([&](sycl::handler& h) {
    const auto kernel = [=](sycl::nd_item<1> id) SYCL_ESIMD_KERNEL MOM_SYCL_KERNEL_ARGS_RESTRICT {
      using esimd_ns::simd;
      const int wi = int(id.get_global_id(0));
      // Consecutive local work-items share an A row band and supply different 128-byte halves.
      const size_t aHalfOffset = (id.get_local_id(0) & 1) * 128;
      int Rg, Cg;
      if (blocked) {
        const int blk = wi / (BH * BW), intra = wi % (BH * BW);
        Rg = (blk / blocksW) * BH + intra / BW;
        Cg = (blk % blocksW) * BW + intra % BW;
      } else {
        Rg = wi / tilesW;
        Cg = wi % tilesW;
      }
      const int rowBase = Rg * HT * ER, colBase = Cg * EC;
      const int8_t* Ap = B.Ap;
      const int8_t* Bp = B.Bp;
      simd<int32_t, 64> accTL[ER * EC], accTR[ER * EC];
      simd<int32_t, 64> accBL[ER * EC], accBR[ER * EC];
      MU<ER * EC>([&](auto t) {
        accTL[t] = 0;
        accTR[t] = 0;
        accBL[t] = 0;
        accBR[t] = 0;
      });
      uint32_t tr[ER * EC][16] = {};
      int rc = 0;
#pragma unroll 8
      for (int p = 0; p < k; p += 32) {
        simd<int8_t, AN> a0[ER], a1[ER];
        MU<ER>([&](auto r) {
          const int band0 = (rowBase + int(r) * HT) >> 3;
          a0[r] = esimd_ns::block_load<int8_t, AN>(Ap + size_t(band0 * (k / 32) + (p >> 5)) * 256 + aHalfOffset);
          a1[r] = esimd_ns::block_load<int8_t, AN>(Ap + size_t((band0 + 1) * (k / 32) + (p >> 5)) * 256 + aHalfOffset);
        });
        MU<EC>([&](auto c) {
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
                accTL[idx] = dpasw_mad(accTL[idx], bf, a0[r]);
                accBL[idx] = dpasw_mad(accBL[idx], bf, a1[r]);
              } else {
                accTR[idx] = dpasw_mad(accTR[idx], bf, a0[r]);
                accBR[idx] = dpasw_mad(accBR[idx], bf, a1[r]);
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
      for (int r = 0; r < ER; r++) {
        for (int c = 0; c < EC; c++) {
          if (dbg) {
            const uint32_t mx = tile_mix(uint32_t(rowBase + r * HT),
                                         uint32_t((colBase + c) * HT), tr[r * EC + c]);
            esimd_ns::atomic_update<esimd_ns::atomic_op::add>(
                &B.result->chk, simd<uint32_t, 1>(0), simd<uint32_t, 1>(mx), esimd_ns::simd_mask<1>(1));
          }
          uint8_t jackpot[32];
          if (!tile_wins(tr[r * EC + c], B.cA, B.target, jackpot))
            continue;
          simd<int, 1> old = esimd_ns::atomic_update<esimd_ns::atomic_op::xchg>(
              &B.result->found, simd<uint32_t, 1>(0), simd<int, 1>(1), esimd_ns::simd_mask<1>(1));
          if (old[0] == 0) {
            B.result->seed = seed;
            B.result->row = uint32_t(rowBase + r * HT);
            B.result->col = uint32_t((colBase + c) * HT);
            for (int i = 0; i < 32; ++i)
              B.result->jackpot[i] = jackpot[i];
          }
        }
      }
    };
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(nWI), sycl::range<1>(localSize)), kernel);
  });
}

static sycl::event search_esimd_dpasw(sycl::queue& q, const Buffers& bb, uint32_t seed,
                                      int m, int n, int k, int rank, bool dbg) {
  if (rank == 256)
    return search_esimd_dpasw_impl<256>(q, bb, seed, m, n, k, rank, dbg);
  return search_esimd_dpasw_impl<0>(q, bb, seed, m, n, k, rank, dbg);
}
#endif // oneAPI main-TU ESIMD only
