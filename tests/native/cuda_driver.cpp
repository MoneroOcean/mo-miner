// CPU-only contract test for the exact driver boundary shipped in the reviewed ACPP patch.
// Types below model only the four stable CUDA ABIs; real SDK/runtime compilation is a separate
// opt-in proof. No GPU, driver, CUDA SDK, shared library or external endpoint is used here.
#include <cassert>
#include <cstring>
#include <string>
#include <iostream>

using CUmodule = void *;
using CUfunction = void *;
using CUstream = void *;
using CUjit_option = int;
using CUresult = int;
constexpr int CUDA_SUCCESS = 0, CUDA_ERROR_NOT_INITIALIZED = 3;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaEnableLegacyStream = 1;
enum cudaDriverEntryPointQueryResult {cudaDriverEntryPointSuccess, cudaDriverEntryPointSymbolNotFound};
using PFN_cuModuleLoadDataEx_v2010 = CUresult (*)(CUmodule *, const void *, unsigned, CUjit_option *, void **);
using PFN_cuModuleUnload_v2000 = CUresult (*)(CUmodule);
using PFN_cuModuleGetFunction_v2000 = CUresult (*)(CUfunction *, CUmodule, const char *);
using PFN_cuLaunchKernel_v4000 = CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                          unsigned, CUstream, void **, void **);
static std::string scenario;
static unsigned queries = 0, loads = 0, unloads = 0, functions = 0, launches = 0;
static int sentinel;
static CUmodule module = &sentinel;
static CUfunction function = &queries;
static CUresult outcome() {return scenario == "call-error" ? 719 : CUDA_SUCCESS;}
CUresult cuModuleLoadDataEx(CUmodule *out, const void *image, unsigned count, CUjit_option *options, void **values) {
  assert(std::strcmp(static_cast<const char *>(image), "ptx") == 0 && count == 0 && !options && !values);
  ++loads; *out = module; return outcome();
}
CUresult cuModuleUnload(CUmodule value) {assert(value == module); ++unloads; return outcome();}
CUresult cuModuleGetFunction(CUfunction *out, CUmodule value, const char *name) {
  assert(value == module && std::strcmp(name, "kernel") == 0); ++functions; *out = function; return outcome();
}
CUresult cuLaunchKernel(CUfunction value, unsigned x, unsigned y, unsigned z, unsigned bx, unsigned by, unsigned bz,
                       unsigned shared, CUstream stream, void **parameters, void **extra) {
  assert(value == function && x == 2 && y == 3 && z == 4 && bx == 5 && by == 6 && bz == 7 && shared == 128);
  assert(stream == &launches && parameters == nullptr && extra == nullptr); ++launches; return outcome();
}
cudaError_t cudaGetDriverEntryPointByVersion(const char *name, void **out, unsigned version,
                                            unsigned long long flags, cudaDriverEntryPointQueryResult *status) {
  assert(flags == cudaEnableLegacyStream); ++queries;
  if (scenario == "no-driver") {return 35;}
  *status = cudaDriverEntryPointSuccess;
  if (!std::strcmp(name, "cuModuleLoadDataEx")) {assert(version == 2010); *out = reinterpret_cast<void *>(&cuModuleLoadDataEx);}
  else if (!std::strcmp(name, "cuModuleUnload")) {assert(version == 2000); *out = reinterpret_cast<void *>(&cuModuleUnload);}
  else if (!std::strcmp(name, "cuModuleGetFunction")) {assert(version == 2000); *out = reinterpret_cast<void *>(&cuModuleGetFunction);}
  else {assert(!std::strcmp(name, "cuLaunchKernel") && version == 4000); *out = reinterpret_cast<void *>(&cuLaunchKernel);}
  if (scenario == "null") {*out = nullptr;}
  if (scenario == "query-error") {return 801;}
  if (scenario == "status-error") {*status = cudaDriverEntryPointSymbolNotFound;}
  return cudaSuccess;
}
#include "cuda_driver.hpp"

int main(int argc, char **argv) {
  assert(argc == 2); scenario = argv[1];
#ifdef ACPP_STATIC_RUNTIME
  const bool absent = scenario == "no-driver" || scenario == "null" || scenario == "query-error" || scenario == "status-error";
#else
  const bool absent = false;
#endif
  const int expected = absent ? CUDA_ERROR_NOT_INITIALIZED : outcome();
  for (unsigned i = 0; i < 20; ++i) {
    CUmodule actual_module = nullptr; CUfunction actual_function = nullptr;
    using namespace hipsycl::rt;
    assert(cuda_driver::cuModuleLoadDataEx(&actual_module, "ptx", 0U, static_cast<CUjit_option *>(nullptr), static_cast<void **>(nullptr)) == expected);
    assert(cuda_driver::cuModuleGetFunction(&actual_function, module, "kernel") == expected);
    assert(cuda_driver::cuLaunchKernel(function, 2U, 3U, 4U, 5U, 6U, 7U, 128U, &launches, nullptr, nullptr) == expected);
    assert(cuda_driver::cuModuleUnload(module) == expected);
    assert(absent ? !actual_module && !actual_function : actual_module == module && actual_function == function);
  }
  assert(loads == (absent ? 0U : 20U) && loads == functions && loads == launches && loads == unloads);
#ifdef ACPP_STATIC_RUNTIME
  assert(queries == 4); // No repeated lookup inside the steady dispatch path, even for failures.
#else
  assert(queries == 0); // Engineering/shared mode must not acquire the native lookup policy.
#endif
  std::cout << "driver boundary passed " << scenario << " queries=" << queries << " launches=" << launches << '\n';
}
