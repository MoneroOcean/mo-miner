// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
//
// FishHash (Iron Fish IRON / Karlsen KLS) GPU search kernel. ASIC-resistant, memory-hard, Ethash-derived.
// Per hash: blake3(header) -> 64B seed -> 32 dataset accesses (3x 128B fetches, mix=f0*f1+f2 u64) ->
// collapse -> blake3(seed||mix_hash) -> 32B. DAG is FIXED (not epoch-based): 1.18M x 64B light cache ->
// 37.7M x 128B (4.6 GB) dataset, both from a fixed seed. Ported bit-exact from github.com/iron-fish/
// fish-hash (cpp/FishHash.cpp + 3rdParty/{blake3,keccak}); validated offline (light_cache[0], blake3
// seed, final hash). Mining and benchmarking always use the full DAG; the lazy 72 MiB light-cache
// path is only for the default offline vector. Production benchmarks use Iron Fish's 180-byte,
// nonce-at-172 big-endian header (FIP-9 swaps the old randomness/graffiti positions). Older
// 76-byte synthetic benchmark rates are not comparable;
// the supported short offline vector is 40 bytes with a little-endian nonce at byte 32.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "../lib-internal.h"
#include "../../native/job-boundary.h"
#include "../../native/consts.h"

#include "device.inc"
#include "search.inc"
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
#include "cooperative_search.inc"
#endif
#include "dag.inc"

#include "state.inc"

#include "entry.inc"
