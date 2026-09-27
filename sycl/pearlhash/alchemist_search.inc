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
