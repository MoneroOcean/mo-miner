// Compile the reviewed provider initializer with fake driver entrypoints, never a GPU driver.
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cstring>

#define ZE_ENTRYPOINTS(X) \
  X(zeMemAllocHost) X(zeMemAllocDevice) X(zeMemAllocShared) X(zeMemFree) \
  X(zeMemGetIpcHandle) X(zeMemPutIpcHandle) X(zeMemOpenIpcHandle) X(zeMemCloseIpcHandle) \
  X(zeContextMakeMemoryResident) X(zeContextEvictMemory) X(zeDeviceGetProperties) \
  X(zeMemFreeExt) X(zeMemGetAllocProperties)
static unsigned calls, opens, symbols, closes;
#define ENTRY(name) static int name() {++calls; return 0;}
ZE_ENTRYPOINTS(ENTRY)
#undef ENTRY
struct Ops {
#define ENTRY(name) int (*name)() = nullptr;
ZE_ENTRYPOINTS(ENTRY)
#undef ENTRY
} g_ze_ops;
[[maybe_unused]] static bool Init_ze_global_state_failed;
[[maybe_unused]] static void* ze_lib_handle;
[[maybe_unused]] static void* utils_open_library(const char*, int) {++opens; return nullptr;}
[[maybe_unused]] static void* utils_get_symbol_addr(void*, const char*, const char*) {++symbols; return nullptr;}
[[maybe_unused]] static void utils_close_library(void*) {++closes;}
#define LOG_FATAL(...) ((void)0)
#define LOG_INFO(...) ((void)0)
#define LOG_DEBUG(...) ((void)0)
#define UMF_UTIL_OPEN_LIBRARY_NO_LOAD 1
#include "bindings.inc"

int main() try {
  for (int attempt = 0; attempt < 3; ++attempt) {
    init_ze_global_state();
    if (opens || symbols || closes || ze_lib_handle || Init_ze_global_state_failed || calls)
      throw std::runtime_error("Static provider probed a shared loader or called the driver during binding");
#define ENTRY(name) if (g_ze_ops.name != &name) throw std::runtime_error("Incorrect static entry: " #name);
ZE_ENTRYPOINTS(ENTRY)
#undef ENTRY
  }
#define ENTRY(name) if (g_ze_ops.name() != 0) throw std::runtime_error("Entry call mismatch");
ZE_ENTRYPOINTS(ENTRY)
#undef ENTRY
  if (calls != 13) throw std::runtime_error("Static entry table lost a function");
  std::puts("Static UMF entry binding: no loader probes, 13 exact callbacks, repeat initialization passed");
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what()); return 1;
}
