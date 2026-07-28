// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

// SYCL KawPow implementation based on XMRig's KawPow reference and OpenCL
// runner structure.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "../lib-internal.h"
#include "../../native/consts.h"
#include "../../xmrig/3rdparty/libethash/ethash.h"
#include "../../xmrig/3rdparty/libethash/data_sizes.h"

#include "program.inc"
#include "state.inc"
#include "entry.inc"
