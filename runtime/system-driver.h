// Fixed system-driver identities only. Never admit an application runtime or user-supplied path.
#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mom::runtime {
enum class SystemDriver { cuda, nvml
#ifdef _WIN32
  // Windows ships this API with the display driver. Linux's HIP runtime is an application
  // component and must stay static; do not add a corresponding Linux loader exception.
  , hip
#endif
};

// Keep modules alive through all queue/allocator/compiler shutdown. Failed lookups remain failed
// for this process; neither a stale pointer nor an unrelated globally exported shim is callable.
template <SystemDriver Driver> auto systemDriverLibrary() {
#ifdef _WIN32
  static const HMODULE library = LoadLibraryExA(Driver == SystemDriver::cuda ? "nvcuda.dll" :
                                               Driver == SystemDriver::nvml ? "nvml.dll" : "amdhip64_7.dll",
                                               nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
  static void* const library = dlopen(Driver == SystemDriver::cuda ? "libcuda.so.1" : "libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
  return library;
}

// Exact typed wrappers cache the result, so this lookup never enters a hot dispatch loop.
template <SystemDriver Driver, class Function> Function systemDriverSymbol(const char* name) {
  const auto library = systemDriverLibrary<Driver>();
#ifdef _WIN32
  return library ? reinterpret_cast<Function>(GetProcAddress(library, name)) : nullptr;
#else
  return library ? reinterpret_cast<Function>(dlsym(library, name)) : nullptr;
#endif
}
} // namespace mom::runtime
