// Copyright GNU GPLv3 (c) 2025-2025 MoneroOcean <support@moneroocean.stream>

// SYCL c29 miner prototype based on Grin GPU Miner (https://github.com/swap-dev/SwapReferenceMiner)
// OpenCL mining code by Jiri Photon Vadura and John Tromp
#include <sycl/sycl.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "state.inc"

#include "search.inc"
