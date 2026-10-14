#ifdef PEARLHASH_ESIMD
// ---- experimental ESIMD register-resident DPAS search (alternative to search()) ----
// One ESIMD work-item owns an ER x EC grid of 16x16 hash tiles in its own GRF
// (ESIMD sub-group size is 1 -- no lane cooperation). The DPAS accumulators stay in registers and
// the per-rank inner-hash XORs them in-GRF, avoiding the joint_matrix_store->SLM->barrier readback
// that caps search() at ~10 TH/s. xmx::dpas(C,B,A): SystolicDepth=8, RepeatCount=8, int8*int8->i32,
// K=32/call; A=8x32 (simd int8 256), B=32x16 VNNI (simd int8 512, == compute_ab's Bp layout),
// C=8x16 (simd int32 128). B' loads straight from Bp; A' is 8 strided row loads from row-major Ap.
// The ordinary route keeps a 2x2 tile. The experimental wide rank-128 route pairs a 2x4 tile
// and pipelined operands with 256 GRFs; larger tiles cannot be judged at the ordinary 128-GRF
// allocation. Source vector sizes alone do not establish actual allocation or absence of spills.
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

// Each lane hashes one proof tile. Keeping all tiles in vector lanes avoids repeating scalar
// BLAKE3 seven times per tile after the matrix work; this is exactly one keyed root block.
template <int Tiles>
SYCL_ESIMD_FUNCTION static inline void esimd_jackpots(
    const esimd_ns::simd<uint32_t, Tiles> (&transcript)[16], const uint8_t* key,
    esimd_ns::simd<uint32_t, Tiles> (&digest)[8]) {
  using Words = esimd_ns::simd<uint32_t, Tiles>;
  Words s[16], message[16];
  auto key_bytes = esimd_ns::block_load<uint8_t, 32>(key);
  esimd_ns::simd<uint32_t, 8> key_words = key_bytes.template bit_cast_view<uint32_t>().read();
  MU<8>([&](auto i) { s[i] = static_cast<uint32_t>(key_words[i]); });
  s[8] = 0x6a09e667u;
  s[9] = 0xbb67ae85u;
  s[10] = 0x3c6ef372u;
  s[11] = 0xa54ff53au;
  s[12] = 0;
  s[13] = 0;
  s[14] = 64;
  s[15] = pearlhash_b3::B3_KEYED | pearlhash_b3::B3_CHUNK_START |
          pearlhash_b3::B3_CHUNK_END | pearlhash_b3::B3_ROOT;
  MU<16>([&](auto i) { message[i] = transcript[i]; });
  const auto mix = [&](int a, int b, int c, int d, Words x, Words y) SYCL_ESIMD_FUNCTION {
    s[a] += s[b] + x;
    Words v = s[d] ^ s[a];
    s[d] = (v >> 16) | (v << 16);
    s[c] += s[d];
    v = s[b] ^ s[c];
    s[b] = (v >> 12) | (v << 20);
    s[a] += s[b] + y;
    v = s[d] ^ s[a];
    s[d] = (v >> 8) | (v << 24);
    s[c] += s[d];
    v = s[b] ^ s[c];
    s[b] = (v >> 7) | (v << 25);
  };
  MU<7>([&](auto round) {
    mix(0, 4, 8, 12, message[0], message[1]);
    mix(1, 5, 9, 13, message[2], message[3]);
    mix(2, 6, 10, 14, message[4], message[5]);
    mix(3, 7, 11, 15, message[6], message[7]);
    mix(0, 5, 10, 15, message[8], message[9]);
    mix(1, 6, 11, 12, message[10], message[11]);
    mix(2, 7, 8, 13, message[12], message[13]);
    mix(3, 4, 9, 14, message[14], message[15]);
    if constexpr (round < 6) {
      Words permuted[16];
      MU<16>([&](auto i) { permuted[i] = message[pearlhash_b3::B3_PERM[i]]; });
      MU<16>([&](auto i) { message[i] = permuted[i]; });
    }
  });
  MU<8>([&](auto i) { digest[i] = s[i] ^ s[i + 8]; });
}

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

// Submit the selected Intel DPAS tile variant with the common jackpot/target check.
template <unsigned Width, int FixedRank = 0, int Columns = pearlhash_esimd_columns, int FixedK = 0>
static void search_esimd_impl(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n,
                               int requested_k, int rank, bool dbg) {
  static_assert(Width == 8 || Width == 16);
  constexpr int ER = pearlhash_esimd_rows, EC = Columns, HT = 16;
  const int k = FixedK ? FixedK : requested_k;
  // Keep accumulators in native column banks: narrow DPAS must not gather/scatter an 8x16
  // accumulator on every call. Proof tiles and operand reuse remain 16x16 and ER x EC.
  constexpr int CB = HT / Width, NW = 8 * Width;
  auto B = bb;
  const int tilesH = m / (HT * ER), tilesW = n / (HT * EC);
  const size_t nWI = (size_t)tilesH * tilesW;
  constexpr int BLK = Width == 16 ? 32 : pearlhash_esimd_block;
  // L2 cache-blocking: only swizzle into BLK x BLK super-blocks when the grid divides evenly (see
  // the cache-blocking note above); otherwise fall back to plain row-major traversal.
  const bool blocked = BLK > 1 && (tilesW % BLK) == 0 && (tilesH % BLK) == 0;
  const int blocksW = blocked ? tilesW / BLK : 0;
  const bool striped = Width == 16 && EC == 4 && tilesH % 64 == 0 && tilesW % 2 == 0;
  q.submit([&](sycl::handler& h) {
    const auto work = [=](sycl::id<1> id) SYCL_ESIMD_FUNCTION MOM_SYCL_KERNEL_ARGS_RESTRICT {
      using esimd_ns::simd;
      const int wi = (int)id[0];
      int Rg, Cg;
      if (striped) {
        unsigned row, column;
        pearlhash_esimd_stripe_position(static_cast<unsigned>(wi),
                                       static_cast<unsigned>(tilesW), row, column);
        Rg = static_cast<int>(row);
        Cg = static_cast<int>(column);
      } else if (blocked) { // swizzle into BLK x BLK super-blocks for L2 reuse
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
      // One lane per tile from the rank fold through jackpot hashing, avoiding a final transpose.
      simd<uint32_t, ER * EC> tr[16];
      MU<16>([&](auto i) { tr[i] = 0; });
      // The generic path prefetches A' without retaining another operand bank in GRF.
      // The wide fixed-rank variant instead pipelines two complete operand banks below.
      // A' is the latency-critical
      // operand (2*ER scattered 256B fragments); a distance-32 A-only prefetch materially improved
      // search throughput, while prefetching the contiguous reused B' measured no gain. ESIMD
      // prefetch is documented for DG2/PVC but works on B580/Xe2; max 256B/call, with an explicit
      // cache hint.
      constexpr auto PFH = esimd_ns::properties{
          esimd_ns::cache_hint_L1<esimd_ns::cache_hint::cached>,
          esimd_ns::cache_hint_L2<esimd_ns::cache_hint::cached>, esimd_ns::alignment<256>};
      constexpr int OperandBanks = FixedRank == 128 ? 2 : 1;
      simd<int8_t, 256> a0[OperandBanks][ER], a1[OperandBanks][ER];
      simd<int8_t, 32 * Width> bf[OperandBanks][EC * CB];
      const auto load = [&](auto bank, const int p) SYCL_ESIMD_FUNCTION {
        // A' tile-major (compute_ab): the 8x32 fragment for band b, k-chunk p is one 256B block;
        // B' is one contiguous 512B VNNI block reused across all ER row-bands.
        MU<ER>([&](auto r) {
          const int band0 = (rowBase + (int)r * HT) >> 3;
          a0[bank][r] =
              esimd_ns::block_load<int8_t, 256>(Ap + ((size_t)(band0 * (k / 32) + (p >> 5)) * 256));
          a1[bank][r] = esimd_ns::block_load<int8_t, 256>(
              Ap + ((size_t)((band0 + 1) * (k / 32) + (p >> 5)) * 256));
        });
        MU<EC>([&](auto c) {
          auto packed = esimd_ns::block_load<int8_t, 512>(
              Bp + (size_t)(colBase + (int)c) * k * 16 + (size_t)(p / 4) * 64);
          if constexpr (Width == 16) {
            bf[bank][c] = packed;
          } else {
            // Split VNNI columns once per K-chunk, then reuse across both A halves and ER rows.
            MU<CB>([&](auto column) {
              auto words = packed.template bit_cast_view<int32_t, 8, 16>()
                               .template select<8, 1, Width, 1>(0, (int)column * Width).read();
              bf[bank][(int)c * CB + (int)column] = words.template bit_cast_view<int8_t>().read();
            });
          }
        });
        if (striped && p + 128 < k) {
          // Nearby rows share B; spread one-half-bank lookahead across those rows rather
          // than prefetching all four full column banks in every work-item.
          const int bank = (Rg >> 1) & 3;
          const size_t offset = (size_t)(colBase + bank) * k * 16 +
                                (size_t)((p >> 5) + 4) * 512 + (Rg & 1) * 256;
          esimd_ns::prefetch<int8_t, 256>(Bp, offset, PFH);
        }
        if (FixedRank == 0 && p + 32 < k)
          MU<ER>([&](auto r) {
            const int band0 = (rowBase + (int)r * HT) >> 3;
            esimd_ns::prefetch<int8_t, 256>(
                Ap, ((size_t)(band0 * (k / 32) + ((p + 32) >> 5)) * 256), PFH);
            esimd_ns::prefetch<int8_t, 256>(
                Ap, ((size_t)((band0 + 1) * (k / 32) + ((p + 32) >> 5)) * 256), PFH);
          });
      };
      const auto accumulate = [&](auto operand) SYCL_ESIMD_FUNCTION {
        MU<EC>([&](auto c) {
          MU<ER>([&](auto r) {
            const int idx = (int)r * EC + (int)c;
            MU<CB>([&](auto column) {
              const int bank = idx * CB + (int)column;
              acc0[bank] = esimd_ns::xmx::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(
                  acc0[bank], bf[operand][(int)c * CB + (int)column], a0[operand][r]);
            });
            MU<CB>([&](auto column) {
              const int bank = idx * CB + (int)column;
              acc1[bank] = esimd_ns::xmx::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(
                  acc1[bank], bf[operand][(int)c * CB + (int)column], a1[operand][r]);
            });
          });
        });
      };
      const auto fold = [&](const unsigned slot) SYCL_ESIMD_FUNCTION {
          simd<uint32_t, ER * EC> folded_tiles;
          MU<ER * EC>([&](auto t) {
            uint32_t folded = 0;
            MU<CB>([&](auto column) {
              const int bank = (int)t * CB + (int)column;
              simd<uint32_t, NW> x = acc0[bank].template bit_cast_view<uint32_t>() ^
                                      acc1[bank].template bit_cast_view<uint32_t>();
              folded ^= esimd_xor_reduce<NW>(x);
            });
            folded_tiles[t] = folded;
          });
          // XOR all native banks before rotating each tile's transcript once.
          tr[slot] = ((tr[slot] << 13) | (tr[slot] >> 19)) ^ folded_tiles;
      };
      if constexpr (FixedRank == 128) {
        // Fold only complete rank groups; the verifier excludes a partial trailing group.
        // Four explicit chunks expose DPAS scheduling without a boundary test per chunk.
        load(std::integral_constant<int, 0>{}, 0);
        for (int group = 0; group < k / 128; ++group) {
          MU<4>([&](auto chunk) {
            constexpr int current = decltype(chunk)::value & 1;
            const int p = group * 128 + (int)chunk * 32;
            if (p + 32 < (k / 128) * 128)
              load(std::integral_constant<int, 1 - current>{}, p + 32);
            accumulate(std::integral_constant<int, current>{});
          });
          fold((unsigned)group & 15u);
        }
      } else {
        unsigned rc = 0;
        for (int p = 0; p < k; p += 32) {
          load(std::integral_constant<int, 0>{}, p);
          accumulate(std::integral_constant<int, 0>{});
          if ((p + 32) % rank == 0)
            fold(rc++ & 15u);
        }
      }
      simd<uint32_t, ER * EC> jackpots[8];
      esimd_jackpots(tr, B.cA, jackpots);
      auto target_bytes = esimd_ns::block_load<uint8_t, 32>(B.target);
      simd<uint32_t, 8> target_words = target_bytes.template bit_cast_view<uint32_t>().read();
      esimd_ns::simd_mask<ER * EC> less = 0, equal = 1;
      // Native Intel words are little-endian, so compare the most significant word first.
      MU<8>([&](auto i) {
        constexpr int word = 7 - decltype(i)::value;
        const uint32_t bound = target_words[word];
        less |= equal & (jackpots[word] < bound);
        equal &= jackpots[word] == bound;
      });
      const auto wins = less | equal;
      for (int r = 0; r < ER; r++)
        for (int c = 0; c < EC; c++) {
          if (dbg) { // whole-search checksum for cross-validation (see tile_mix); ESIMD atomic add
            uint32_t tile_transcript[16];
            MU<16>([&](auto i) { tile_transcript[i] = tr[i][r * EC + c]; });
            const uint32_t mx = tile_mix((uint32_t)(rowBase + r * HT),
                                         (uint32_t)((colBase + c) * HT), tile_transcript);
            esimd_ns::atomic_update<esimd_ns::atomic_op::add>(
                &B.result->chk, esimd_ns::simd<uint32_t, 1>(0), esimd_ns::simd<uint32_t, 1>(mx),
                esimd_ns::simd_mask<1>(1));
          }
          if (!wins[r * EC + c])
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
              B.result->jackpot[i] = static_cast<uint8_t>(
                  jackpots[i / 4][r * EC + c] >> (8 * (i % 4)));
          }
        }
    };
    const auto kernel = [=](sycl::id<1> id) SYCL_ESIMD_KERNEL { work(id); };
    if constexpr (Width == 16 && FixedRank == 128) {
      const auto grouped = [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
        work(item.get_global_id());
      };
      const size_t local = nWI % 16 == 0 ? 16 : 1;
#if defined(MOM_PEARLHASH_INTEL_LARGE_GRF)
      namespace oneapi_exp = sycl::ext::oneapi::experimental;
      oneapi_exp::nd_launch(
          h, oneapi_exp::launch_config{
                 sycl::nd_range<1>{sycl::range<1>(nWI), sycl::range<1>(local)},
                 oneapi_exp::properties{sycl::ext::intel::experimental::grf_size<256>}},
          grouped);
#else
      h.parallel_for(sycl::nd_range<1>{sycl::range<1>(nWI), sycl::range<1>(local)}, grouped);
#endif
    } else
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
// Dispatch Intel ESIMD tile variants; combined builds export this symbol from the spir64-only TU.
void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank,
                  bool dbg, const unsigned width) {
  if (width == 8u)
    search_esimd_impl<8>(q, bb, seed, m, n, k, rank, dbg);
  else if (width == 16u) {
    // Specialize the common fold interval without changing other proof configurations.
    if (pearlhash_wide_pipeline_route(width, rank, n)) {
      if (k == 4096)
        search_esimd_impl<16, 128, 4, 4096>(q, bb, seed, m, n, k, rank, dbg);
      else
        search_esimd_impl<16, 128, 4>(q, bb, seed, m, n, k, rank, dbg);
    }
    else
      search_esimd_impl<16>(q, bb, seed, m, n, k, rank, dbg);
  }
  else
    throw std::string("PearlHash ESIMD search requires a qualified DPAS width");
}
#endif // PEARLHASH_ESIMD

// In the combined build the ESIMD search lives in the separate spir64-only TU (esimd.cpp);
// declare it here so attempt() can call it. (When PEARLHASH_ESIMD is set this is the same TU, defined above.)
#if defined(MOM_PEARLHASH_HAS_ESIMD) && !defined(PEARLHASH_ESIMD)
unsigned esimd_search_width(const sycl::device& device) noexcept;
// Dispatch Intel ESIMD tile variants; combined builds export this symbol from the spir64-only TU.
void search_esimd(sycl::queue& q, const Buffers& bb, uint32_t seed, int m, int n, int k, int rank,
                  bool dbg, unsigned width);
#endif
