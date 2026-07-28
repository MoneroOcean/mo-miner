// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#include "core.h"
#include "../sycl/lib.h"   // pearlhash_proof()

#include "3rdparty/fmt/core.h"
#include "backend/cpu/Cpu.h"
#include "crypto/cn/CnCtx.h"
#include "crypto/randomx/blake2/blake2.h"
#include "crypto/randomx/blake2/avx2/blake2b.h"
#include "crypto/rx/RxFix.h"
#include "hw/msr/Msr.h"
#include "3rdparty/argon2.h"
#include "base/tools/bswap_64.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <inttypes.h>
#include <thread>
#include <cstring>

#include "core/messages.inc"

#include "core/commands.inc"
#include "core/execution.inc"
static void cleanup_sycl_runtime(void*) { sycl_cleanup(); }

static napi_value init_module(napi_env env, napi_value exports) {
  napi_value result = AsyncWorkerWrapper::Init(env, exports);
  // Run after every compute worker/TSFN has drained but before Node unloads this addon or the SYCL
  // runtime. DPC++ source-JIT images have module destructors of their own, while AdaptiveCpp CUDA
  // otherwise reaches cudaErrorCudartUnloading (error 4) from late queue/allocator destruction.
  // Releasing every algorithm registry here avoids depending on cross-DLL C++ static order.
  AsyncWorker::check(env, napi_add_env_cleanup_hook(env, cleanup_sycl_runtime, nullptr));
  return result;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init_module)
