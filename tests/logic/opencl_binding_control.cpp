// Exercise the maintained probe itself with a headerless ICD, not another binding check.
#define main opencl_binding_main
#include "../reference/native_opencl_binding.cpp"
#undef main

namespace { unsigned queries = 0; }
extern "C" int clGetPlatformIDs(cl_uint, cl_platform_id*, cl_uint* count) {
  ++queries; *count = 0; return CL_PLATFORM_NOT_FOUND_KHR;
}
namespace ocl {
decltype(::clGetPlatformIDs)* clGetPlatformIDs_ptr = &::clGetPlatformIDs;
bool loadOCLLibrary() { return true; }
void unloadOCLLibrary() {}
}
int main(int argc, char** argv) {
  const bool negative = argc == 2 && std::strcmp(argv[1], "negative") == 0;
  const bool platforms = argc == 2 && std::strcmp(argv[1], "platforms") == 0;
  if (negative) ocl::clGetPlatformIDs_ptr = nullptr;
  char mode[] = "--bindings-only";
  char* args[]{argv[0], mode};
  const int result = opencl_binding_main(platforms ? 1 : 2, args);
  return result == (negative ? 86 : 0) && queries == (platforms ? 3u : 0u) ? 0 : 1;
}
