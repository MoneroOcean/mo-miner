// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
//
// ZelHash (Equihash 125,4) GPU solver -- Wagner bucket-collision (Tromp/djezo lineage).
//
// Pipeline:
//   gen    : 2^26 entries from a personalized "ZelProof" blake2b over the 140-byte header + the
//            ZelHash "twist" (16-aligned block prefix-sum of stock hashes), masked, split into 4
//            sub-elements, ExpandArray(25) -> 25-bit collision fields.
//   round  : (M2+) bucket by high BUCKBITS, collide low RESTBITS, XOR pairs -> next round.
//   recover: (M3+) walk tree -> 16 leaf indices -> CompressArray(26) -> 52-byte solution.
//
// Mining runs the complete solve path: gen-fill, four collision rounds, fused final filtering, recovery, and target
// filtering. The default is_test path still uses the cheaper gen-kernel cross-check; set
// MOM_ZELHASH_SOLVE to run the full block-400000 proof recovery vector.
//
// Performance/portability note: all vendors run the same complete proof pipeline and exact vectors.
// Like-for-like stage timing shows no inherent Windows/AMD regression; stale mixed-source rates must
// not be used to infer an OS or hardware limitation.

#include <sycl/sycl.hpp>

#if !defined(MOM_SYCL_HAS_CUDA) && !defined(MOM_SYCL_HAS_HIP) && \
    !defined(__NVPTX__) && !defined(__AMDGCN__) && \
    !defined(MOM_SYCL_ADAPTIVECPP) && !defined(MOM_SYCL_PORTABLE_OPENCL) && \
    __has_include(<sycl/ext/intel/experimental/grf_size_properties.hpp>)
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#define MOM_ZELHASH_INTEL_LARGE_GRF 1
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "../lib-internal.h"
#include "../blake2b_pair.hpp"
#include "../../native/consts.h"

#include "layout.inc"

#include "blake2b.inc"
// ===========================================================================================
#include "generation.inc"
#include "rounds.inc"

#include "recovery.inc"
#if defined(MOM_EQ_FORWARD_MAP)
constexpr uint32_t L0_INVERT_HEADS = NBUCKETS;
#endif
#include "l0_invert.inc"
// ===========================================================================================
#include "state.inc"

#include "proof.inc"

#include "entry.inc"
