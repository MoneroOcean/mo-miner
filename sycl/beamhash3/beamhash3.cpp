// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
//
// BeamHash III (Beam) GPU solver -- Wagner bucket-collision (Equihash-family, k=5).
//
// NOT stock Equihash: reuses the zelhash.cpp Wagner infra (bucket-sort, slot-shrink, Cantor tree
// recovery, 25-bit CompressArray pack), but the row generation and the per-round mixing are different:
//   gen      : 2^25 leaves; each leaf's 448 work bits = 7 u64 = SipHash-2-4(key=IndividualWork(4 u64),
//              msg=(index<<3)+i) for i=0..6.  IndividualWork = BLAKE2b(prework||nonce||extranonce,
//              personal "Beam-PoW"+le32(448)+le32(5)).
//   applyMix : a non-linear mix BEFORE every round -- serialize (workBits | indexTree-pad) to 512 bits,
//              fold the 8 u64 words with rotl by (29*(i+1))&63 and modular add, rotl<<24, write into the
//              low 64 work bits.  Couples the index tree back into the work bits each round (the ASIC
//              resistance mechanism). NO Equihash analog.
//   round    : collide low 24 work bits XOR=0 (rounds 1-4); round 5 collides low 48 bits. After collision
//              merge = (a.workBits ^ b.workBits) >> 24, masked to remLen.
//   recover  : walk the per-level tree -> 32 leaf indices -> CompressArray(25) -> 100-byte minimal, then
//              the 104-byte solution (low 100 bytes minimal + top 4 bytes extranonce).
//
// Bit-exactly mirrors the BeamHash III reference algorithm. The is_test path validates gen+mix on-device.
//
// Mining runs the complete BeamHash III solve path. The default is_test path keeps the cheap gen+mix
// oracle validation; set MOM_BEAMHASH3_SOLVE for the M4 keystone full-solve vector.
//
// Performance/portability notes:
// - Sol/s varies with the nonce window. Repeating a fresh process repeats its initial work;
//   compare matched steady windows, not just the first report. At 150 W on RTX 5060 Ti,
//   historical/current solvers both measured about 32.3 Sol/s steady; an old first report was 36.9.
// - Memory-limited devices use the compact two-arena solver. B580 uses an explicit SIMD16 ESIMD
//   seed, 768-lane ordinary collision workgroups, and paired first/fourth-round partitions; larger
//   devices may select the full layout, so cross-GPU rates are not automatically comparable.
// - The ESIMD seed cuts the measured B580 seed from about 33 to 15 ms and raises the reproducible
//   Linux median from 20.63 to 22.80 Sol/s. Other devices retain the portable seed.
// - Requesting SIMD16 for the compact collision rounds raised matched B580 steady solve-loop work
//   from 86 to 97 iterations per roughly ten seconds. HIP, NVPTX, and portable OpenCL retain their
//   native subgroup width through the capability-gated attribute macro.
// - Pairing two logical partitions while retaining the original collision keys in distinct SLM bins
//   raised matched B580 solve-loop throughput from 12.28 to 12.60 attempts/s and the reproducible
//   five-sample Linux median to 24.72 Sol/s.
// - Pairing middle rounds, concentrating early output buckets, direct source-index chains, and
//   four-way round-1 merging did not improve on 12.60 attempts/s: reduced input scans were offset
//   by slower collision traversal or less balanced workgroups.
// - In matched SIMD16 middle-round tests, 768-lane groups beat 128/256/512/1024. Halving local
//   index widths and narrowing the two barriers to local memory were throughput-neutral; theoretical
//   SLM occupancy alone does not explain the gap.
// - Four collision rounds still consume about 66 ms. Even a zero-cost seed would miss the 80% best-
//   peer target, so the remaining Intel gap requires collision redesign; no controlled same-B580
//   reference currently proves an attainable hardware limit.

#include <sycl/sycl.hpp>

#if defined(PEARLHASH_ESIMD) && !defined(MOM_SYCL_ADAPTIVECPP)
// The build defines PEARLHASH_ESIMD only for its oneAPI ESIMD-capable worker.
#define MOM_BEAMHASH3_ESIMD
#include <sycl/ext/intel/esimd.hpp>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

#include "../lib-internal.h"
#include "../../native/consts.h"

#include "common.inc"
#include "solver.inc"

#include "entry.inc"
