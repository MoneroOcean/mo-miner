// Exercise the real loader without a GPU or a driver: an explicit known GPU candidate must
// reach driver loading directly, never first require an application libze_loader.so.
#include <ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
unsigned driverCalls = 0, loaderCalls = 0;
thread_local bool loadFailed = false;
}
// The real native guard forwards permitted driver loads to this spy, never to the OS.
extern "C" void* dlopen(const char* name, int) {
  if (name && std::strcmp(name, "libze_intel_gpu.so.1") == 0) ++driverCalls;
  if (name && std::strncmp(name, "libze_loader.so", 15) == 0) ++loaderCalls;
  loadFailed = true;
  return nullptr; // No third-party library or hardware is ever admitted by this fixture.
}
extern "C" char* dlerror() {
  if (!loadFailed) return nullptr;
  loadFailed = false;
  static char message[] = "fixture GPU driver unavailable";
  return message;
}
int main() {
  if (setenv("ZE_ENABLE_ALT_DRIVERS", "libze_intel_gpu.so.1", 1)) return 85;
  const auto result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  if (result != ZE_RESULT_ERROR_UNINITIALIZED || driverCalls != 1 || loaderCalls) {
    std::fprintf(stderr, "Level Zero static boundary: result=%u driver=%u loader=%u\n",
                 unsigned(result), driverCalls, loaderCalls);
    return 86;
  }
  std::puts("Full static Level Zero reaches its GPU driver without an application loader");
}
