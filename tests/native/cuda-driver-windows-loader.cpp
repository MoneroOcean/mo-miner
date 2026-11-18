// Windows CUDA driver lookup through the production CudaDriverApi, compiled on any host against a
// recording <windows.h> stub (-D_WIN32). nvcuda.dll must come only from the System32-restricted
// loader; a search-path LoadLibraryA would admit a planted driver from the CWD or PATH.
#include "../../sycl/cuda-api.h"

#include <cstdio>
#include <cstring>

static int system32_driver_loads = 0, other_loads = 0;

HMODULE LoadLibraryExA(const char* name, void* file, unsigned long flags) {
  if (std::strcmp(name, "nvcuda.dll") == 0 && !file && flags == LOAD_LIBRARY_SEARCH_SYSTEM32)
    ++system32_driver_loads;
  else
    ++other_loads;
  return reinterpret_cast<HMODULE>(1);
}

HMODULE LoadLibraryA(const char*) {
  ++other_loads;
  return reinterpret_cast<HMODULE>(1);
}

static void driver_symbol() {}

FARPROC GetProcAddress(HMODULE, const char*) {
  return &driver_symbol;
}

int main() {
  const auto& api = mom::CudaDriverApi::instance();
  if (system32_driver_loads != 1 || other_loads != 0 || !api.extended_available()) {
    std::fprintf(stderr, "system32=%d other=%d available=%d error=%s\n", system32_driver_loads,
                 other_loads, api.extended_available(), api.error.c_str());
    return 1;
  }
  std::puts("CUDA driver loaded from System32 only");
  return 0;
}
