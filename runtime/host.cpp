// No source/bytecode evaluation: all JavaScript (including compatibility modules) is AOT code.
#include "hermes/VM/static_h.h"
#include "hermes/hermes.h"
#include "event-loop.h"
#include "process.h"
#include "services.h"
#ifdef MOM_NATIVE_RUNTIME
#include "../sycl/lib.h"
#endif

extern "C" SHConsoleContext *init_console_bindings(
    SHRuntime *runtime, int argc, const char *const *argv) {
  auto &rt = *_sh_get_hermes_runtime(runtime);
#ifdef MOM_NATIVE_RUNTIME
  auto *context = mom::runtime::createLoop(rt, create_worker);
#else
  auto *context = mom::runtime::createLoop(rt);
#endif
  try {
    mom::runtime::installProcess(rt, argc, argv);
    mom::runtime::installServices(rt);
  } catch (...) {
    mom::runtime::closeLoop(context);
    mom::runtime::freeLoop(context);
    throw;
  }
  return context;
}

extern "C" bool run_event_loop(SHRuntime *, SHConsoleContext *context) {
  return mom::runtime::runLoop(context);
}

extern "C" void release_console_bindings(SHConsoleContext *context) {
  mom::runtime::closeLoop(context);
#ifdef MOM_NATIVE_RUNTIME
  // Match the development addon's cleanup hook: all Core threads and callbacks must drain
  // before retiring algorithm queues/USM, while the static compiler/driver is still alive.
  sycl_cleanup();
#endif
}
extern "C" void free_console_context(SHConsoleContext *context) {mom::runtime::freeLoop(context);}
