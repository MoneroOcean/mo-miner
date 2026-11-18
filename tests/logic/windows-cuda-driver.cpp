// Exercise the real bridge with test-only SDK/OS declarations: no NVIDIA SDK, driver or GPU.
#include "runtime/windows-cuda-driver.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int mode;
unsigned cudaLoads, nvmlLoads, lookups, initCalls;
CUresult fakeInit(unsigned flags) {if (flags) std::abort(); ++initCalls; return CUDA_SUCCESS;}
CUresult fakeAlloc(CUdeviceptr* output, std::size_t bytes) {if (bytes != 123) std::abort(); *output = 456; return CUDA_SUCCESS;}
CUresult fakeError(CUresult value, const char** output) {if (value != CUDA_ERROR_NO_DEVICE) std::abort(); *output = "fake CUDA error"; return CUDA_SUCCESS;}
nvmlReturn_t fakeNvmlInit() {return NVML_SUCCESS;}
const char* fakeNvmlError(nvmlReturn_t value) {if (value != NVML_ERROR_UNINITIALIZED) std::abort(); return "fake NVML error";}
nvmlReturn_t fakeFan(nvmlDevice_t device, unsigned* output) {if (device) std::abort(); *output = 41; return NVML_SUCCESS;}
inline constexpr char init[] = "cuInit", alloc[] = "cuMemAlloc_v2", error[] = "cuGetErrorString";
inline constexpr char nvmlInit[] = "nvmlInit_v2", nvmlError[] = "nvmlErrorString", fan[] = "nvmlDeviceGetFanSpeed";
template <class F> using Call = mom::runtime::WindowsCudaCall<F>;
}

HMODULE LoadLibraryExA(const char* name, void* file, unsigned flags) {
  if (file || flags != LOAD_LIBRARY_SEARCH_SYSTEM32) std::abort();
  if (!std::strcmp(name, "nvcuda.dll")) {++cudaLoads; return mode == 1 ? nullptr : (HMODULE)1;}
  if (!std::strcmp(name, "nvml.dll")) {++nvmlLoads; return mode == 1 ? nullptr : (HMODULE)2;}
  std::abort();
}
FARPROC GetProcAddress(HMODULE module, const char* name) {
  if (!module) std::abort();
  ++lookups;
  if (mode == 2 && (!std::strcmp(name, alloc) || !std::strcmp(name, error) || !std::strcmp(name, fan))) return nullptr;
  if (module == (HMODULE)1) {
    if (!std::strcmp(name, init)) return reinterpret_cast<FARPROC>(&fakeInit);
    if (!std::strcmp(name, alloc)) return reinterpret_cast<FARPROC>(&fakeAlloc);
    if (!std::strcmp(name, error)) return reinterpret_cast<FARPROC>(&fakeError);
  } else if (module == (HMODULE)2) {
    if (!std::strcmp(name, nvmlInit)) return reinterpret_cast<FARPROC>(&fakeNvmlInit);
    if (!std::strcmp(name, nvmlError)) return reinterpret_cast<FARPROC>(&fakeNvmlError);
    if (!std::strcmp(name, fan)) return reinterpret_cast<FARPROC>(&fakeFan);
  }
  std::abort();
}

int main(int argc, char** argv) {
  if (argc != 2 || (mode = std::atoi(argv[1])) < 0 || mode > 2) return 1;
  if (cudaLoads || nvmlLoads || lookups) return 2;
  const auto expectedInit = mode == 1 ? CUDA_ERROR_NO_DEVICE : CUDA_SUCCESS;
  if (Call<decltype(&::cuInit)>::call<init>(0) != expectedInit || Call<decltype(&::cuInit)>::call<init>(0) != expectedInit) return 3;
  CUdeviceptr address = 77;
  if (Call<decltype(&::cuMemAlloc_v2)>::call<alloc>(&address, 123) != (mode ? CUDA_ERROR_NOT_INITIALIZED : CUDA_SUCCESS) || address != (mode ? 77 : 456)) return 4;
  const char* description = nullptr;
  if (Call<decltype(&::cuGetErrorString)>::call<error>(CUDA_ERROR_NO_DEVICE, &description) != (mode ? CUDA_ERROR_NOT_INITIALIZED : CUDA_SUCCESS) || !description) return 5;
  if (Call<decltype(&::nvmlInit_v2)>::call<nvmlInit>() != (mode == 1 ? NVML_ERROR_UNINITIALIZED : NVML_SUCCESS)) return 6;
  if (!Call<decltype(&::nvmlErrorString)>::call<nvmlError>(NVML_ERROR_UNINITIALIZED)) return 7;
  unsigned speed = 99;
  if (Call<decltype(&::nvmlDeviceGetFanSpeed)>::call<fan>(nullptr, &speed) != (mode ? NVML_ERROR_UNINITIALIZED : NVML_SUCCESS) || speed != (mode ? 99U : 41U)) return 8;
  if (cudaLoads != 1 || nvmlLoads != 1 || initCalls != (mode == 1 ? 0U : 2U)) return 9;
  const auto previous = lookups;
  // Success and failure stay cached for process lifetime, even if later environment changes.
  const auto expected = mode ? CUDA_ERROR_NOT_INITIALIZED : CUDA_SUCCESS;
  mode = 0;
  if (Call<decltype(&::cuMemAlloc_v2)>::call<alloc>(&address, 123) != expected || lookups != previous || cudaLoads != 1 || nvmlLoads != 1) return 10;
  std::puts("Windows SDK-typed NVIDIA bridge: ABI, fallback and cached lifetime passed");
  return 0;
}
