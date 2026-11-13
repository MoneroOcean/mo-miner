#pragma once

#include "../native/async-worker.h"

namespace facebook::jsi { class Runtime; }
struct SHConsoleContext;

namespace mom::runtime {
// All JS handles and libuv callbacks belong to the runtime thread. Close/drain them before the
// language runtime is destroyed; freeing native handles alone would leave callbacks holding JS.
using WorkerFactory = std::unique_ptr<AsyncWorker> (*)(std::unique_ptr<WorkerEvents>);
SHConsoleContext *createLoop(facebook::jsi::Runtime &runtime, WorkerFactory workerFactory = nullptr);
bool runLoop(SHConsoleContext *context);
void closeLoop(SHConsoleContext *context);
void freeLoop(SHConsoleContext *context);
}
