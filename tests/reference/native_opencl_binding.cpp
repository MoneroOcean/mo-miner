// Exercise the ORIGINAL adapter storage/lifetime TU, with either linked or dynamic ICD mode.
// The native library guard deliberately denies application loaders; no GPU driver is loaded.
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <cstdio>
#include <cstring>
namespace ocl {
extern decltype(::clGetPlatformIDs)* clGetPlatformIDs_ptr;
bool loadOCLLibrary();
void unloadOCLLibrary();
}

int main(int argc, char** argv) {
  const bool required = argc == 2 && std::strcmp(argv[1], "--require-linked") == 0;
  // Windows can validate the actual SDK archive without initializing any installed driver.
  const bool bindingsOnly = argc == 2 && std::strcmp(argv[1], "--bindings-only") == 0;
  for (int attempt = 0; attempt < 3; ++attempt) {
    const bool loaded = ocl::loadOCLLibrary();
    const bool bound = ocl::clGetPlatformIDs_ptr == &::clGetPlatformIDs;
    if ((required || bindingsOnly) && (!loaded || !bound)) return 86;
#ifdef MOM_NATIVE_OPENCL_LINKED
    if (!loaded || !bound) return 1;
    if (!bindingsOnly) {
      cl_uint count = 123;
      if (ocl::clGetPlatformIDs_ptr(0, nullptr, &count) != CL_PLATFORM_NOT_FOUND_KHR || count) return 2;
    }
#else
    if (loaded || bound) return 3;
#endif
    ocl::unloadOCLLibrary();
  }
  std::puts("Original OpenCL storage/lifetime: linked ICD or guarded dynamic negative passed");
  return 0;
}
