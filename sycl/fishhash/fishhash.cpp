// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
//
// FishHash (Iron Fish IRON / Karlsen KLS) GPU search kernel. ASIC-resistant, memory-hard, Ethash-derived.
// Per hash: blake3(header) -> 64B seed -> 32 dataset accesses (3x 128B fetches, mix=f0*f1+f2 u64) ->
// collapse -> blake3(seed||mix_hash) -> 32B. DAG is FIXED (not epoch-based): 1.18M x 64B light cache ->
// 37.7M x 128B (4.6 GB) dataset, both from a fixed seed. Ported bit-exact from github.com/iron-fish/
// fish-hash (cpp/FishHash.cpp + 3rdParty/{blake3,keccak}); validated offline (light_cache[0], blake3
// seed, final hash). This first version uses LAZY lookup (computes dataset items from the 72 MB light
// cache on the fly) -- correct but slow; the full-DAG fast path is built when intensity warrants it.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
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

#include "device.inc"
#include "search.inc"
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
#include "cooperative_search.inc"
#endif
#include "dag.inc"

#include "state.inc"

#include "entry.inc"
