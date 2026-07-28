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

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "../lib-internal.h"
#include "../../native/consts.h"

#include "common.inc"
#include "solver.inc"

#include "entry.inc"
