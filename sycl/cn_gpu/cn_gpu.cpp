// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

// SYCL cn/gpu implementation based on the public CryptoNight-GPU specification.
// OpenCL mining code by wolf9466, fireice_uk and psychocrypt
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__)
#include <filesystem>
#include <iomanip>
#include <sstream>
#include "../cuda-api.h"
#include "../jit-cache.h"
#endif

#include "../lib-internal.h"
#include "../../native/consts.h"

#include "crypto.inc"

#include "recurrence.inc"

#include "cuda_jit.inc"

#include "state.inc"
#include "entry.inc"
