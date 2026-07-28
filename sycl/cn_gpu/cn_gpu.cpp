// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

// SYCL cn/gpu implementation based on the public CryptoNight-GPU specification.
// OpenCL mining code by wolf9466, fireice_uk and psychocrypt
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__)
#include <cuda.h>
#include <nvrtc.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

#include "../lib-internal.h"
#include "../../native/consts.h"

#include "crypto.inc"

#include "recurrence.inc"

#include "cuda_jit.inc"

#include "state.inc"
#include "entry.inc"
