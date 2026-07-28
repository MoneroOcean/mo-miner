// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "../lib-internal.h"
#include "../../native/consts.h"
#include "../../xmrig/3rdparty/libethash/ethash.h"
#include "../../xmrig/3rdparty/libethash/data_sizes.h"

#include "device.inc"

#include "search.inc"

#include "state.inc"

#include "entry.inc"
