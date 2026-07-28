// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>
//
// PearlHashHash (PRL) NoisyGEMM proof-of-useful-work GPU search kernel (SYCL / Level-Zero).
// Per seed: counter-RNG A/B -> keyed-BLAKE3 commitment roots -> sparse low-rank noise -> noised
// int8 A'*B' XMX GEMM tiles -> per-16x16-tile XOR/rotl13 transcript -> keyed-BLAKE3 jackpot vs
// target. On a win the host builds the pool PlainProof (Merkle openings) directly -- see the
// PEARLHASH_STANDALONE block. Validated against the real verify_plain_proof_v2 (pearl-research-labs).
//
// Throughput notes (Arc B580, k=1024, rank=64, m=n=16384): the search GEMM sustains ~32-34
// TH/s (DPAS MAC/s); the full per-seed attempt ~28-30 TH/s, the rest being the mandatory
// BLAKE3 commitment over A/Bt. Key optimizations: A/B are never materialized (regenerated from
// the counter-RNG inside the consumers), the commitment Merkle tree is reduced in parallel,
// the search uses a PEARLHASH_HR x PEARLHASH_NTILE register tile with software-pipelined B loads.

#include <sycl/sycl.hpp>
#ifndef PEARLHASH_STANDALONE
#include "../lib-internal.h"
#endif
#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__) && \
    !defined(MOM_PEARLHASH_ESIMD_TU)
#include <cuda.h>
#include <nvrtc.h>
#endif
#if defined(MOM_SYCL_HAS_HIP)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#endif
#ifdef PEARLHASH_ESIMD
#include <sycl/ext/intel/esimd.hpp>   // experimental register-resident DPAS search path
#endif
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <string>
#include <vector>
#include <set>
#include <array>
#include <map>
#include <mutex>
#include <memory>
#include <thread>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#if defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_HAS_CUDA)
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

#include "blake3.inc"

#include "matrix.inc"
#include "amd_sycl_wmma.inc"

#include "sycl_search.inc"

// NVIDIA pearlhash throughput lives entirely in the int8 Tensor Cores, reached here through the
// mma.sync path below. It is built with the intel/llvm DPC++ CUDA backend (nvptx64) -- the SAME
// DPC++ as the Intel build -- so every SYCL kernel stays unified across both GPU vendors, and the
// runtime kernel_compiler JIT that kawpow's per-period specialization relies on is available too.
#include "cuda_sycl_search.inc"

#include "esimd_search.inc"

#include "jit_cache.inc"

// PearlHash's shared SYCL CUDA kernel already uses cp.async, ldmatrix, and mma.sync, but the SYCL
// execution model cannot express CUTE's staged CTA mainloop without manually duplicating a large
// part of CUTE's layout machinery. That path plateaus near 55.5 TH/s on RTX 5060 Ti; compiling the
// architecture-neutral source below with NVRTC reaches 71.5 TH/s at the same 150 W. The override is
// deliberately optional: no device binary is shipped, and any missing compiler/header/runtime or
// unsupported GPU falls back to the normal selectable SYCL implementation.
#include "cuda_jit.inc"

#include "hip_jit.inc"
#include "amd_wmma_dispatch.inc"

#include "dispatch.inc"
#include "host.inc"
