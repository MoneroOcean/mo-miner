// Run generated declarations/forwarding against a fake loader, never a host driver or GPU.
#define dlopen mock_dlopen
#define dlsym mock_dlsym
#include "driver-forwarding.cpp"
#undef dlopen
#undef dlsym
#include <atomic>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <thread>

static int mode = 0;
static std::atomic<unsigned> cuda_loads{0}, nvml_loads{0}, lookups{0};
static bool recovered = false;
static CUresult fake_init(unsigned flags) {return flags == 7 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;}
static CUresult fake_alloc(CUdeviceptr* out, size_t size) {*out = size + 0x123456789ULL; return CUDA_SUCCESS;}
static CUresult fake_context(CUcontext* out) {*out = reinterpret_cast<CUcontext>(0x12345678); return CUDA_SUCCESS;}
static CUresult fake_error(CUresult, const char** out) {*out = "forwarded CUDA error"; return CUDA_SUCCESS;}
static nvmlReturn_t fake_nvml() {return NVML_SUCCESS;}
static const char* fake_nvml_error(nvmlReturn_t) {return "forwarded NVML error";}
static nvmlReturn_t fake_fan(nvmlDevice_t device, unsigned* speed) {
  if (device != reinterpret_cast<nvmlDevice_t>(0xabcdef)) return NVML_ERROR_UNINITIALIZED;
  *speed = 37; return NVML_SUCCESS;
}

extern "C" void* mock_dlopen(const char* name, int flags) noexcept {
  assert(flags == (RTLD_NOW | RTLD_LOCAL));
  if (std::strcmp(name, "libcuda.so.1") == 0) ++cuda_loads;
  else if (std::strcmp(name, "libnvidia-ml.so.1") == 0) ++nvml_loads;
  else {assert(false);}
  return mode == 1 ? nullptr : reinterpret_cast<void*>(1);
}
extern "C" void* mock_dlsym(void*, const char* name) noexcept {
  ++lookups;
  if (mode == 2 && !recovered && std::strcmp(name, "cuMemAlloc_v2") == 0) return nullptr;
  if (std::strcmp(name, "cuInit") == 0) return reinterpret_cast<void*>(&fake_init);
  if (std::strcmp(name, "cuMemAlloc_v2") == 0) return reinterpret_cast<void*>(&fake_alloc);
  if (std::strcmp(name, "cuCtxGetCurrent") == 0) return reinterpret_cast<void*>(&fake_context);
  if (std::strcmp(name, "cuGetErrorString") == 0) return reinterpret_cast<void*>(&fake_error);
  if (std::strcmp(name, "nvmlInit_v2") == 0) return reinterpret_cast<void*>(&fake_nvml);
  if (std::strcmp(name, "nvmlErrorString") == 0) return reinterpret_cast<void*>(&fake_nvml_error);
  if (std::strcmp(name, "nvmlDeviceGetFanSpeed") == 0) return reinterpret_cast<void*>(&fake_fan);
  assert(false); return nullptr;
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  mode = std::atoi(argv[1]);
  const bool available = mode != 1;
  CUcontext context = nullptr;
  // Distinct function signatures must share one process-lifetime module handle even on first
  // concurrent entry, while caching their own correctly typed symbol separately.
  std::thread first([&] {assert(cuInit(7) == (available ? CUDA_SUCCESS : CUDA_ERROR_NO_DEVICE));});
  std::thread second([&] {assert(cuCtxGetCurrent(&context) == (available ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED));});
  first.join(); second.join();
  assert(context == (available ? reinterpret_cast<CUcontext>(0x12345678) : nullptr));
  CUdeviceptr address = 99;
  assert(cuMemAlloc_v2(&address, 0x100000003ULL) == (available && mode != 2 ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED));
  assert(address == (available && mode != 2 ? 0x22345678cULL : 99));
  recovered = true; // A failed cached symbol is never retried behind a running queue's back.
  const auto before = lookups.load();
  assert(cuMemAlloc_v2(&address, 11) == (available && mode != 2 ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED));
  assert(lookups == before);
  const char* error = nullptr;
  assert(cuGetErrorString(CUDA_ERROR_NOT_INITIALIZED, &error) == (available ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED));
  assert(error && std::strstr(error, available ? "forwarded" : "unavailable"));
  assert(nvmlInit_v2() == (available ? NVML_SUCCESS : NVML_ERROR_UNINITIALIZED));
  assert(std::strstr(nvmlErrorString(NVML_ERROR_UNINITIALIZED), available ? "forwarded" : "unavailable"));
  unsigned speed = 91;
  assert(nvmlDeviceGetFanSpeed(reinterpret_cast<nvmlDevice_t>(0xabcdef), &speed) == (available ? NVML_SUCCESS : NVML_ERROR_UNINITIALIZED));
  assert(speed == (available ? 37 : 91));
  assert(cuda_loads == 1 && nvml_loads == 1);
  assert(available || lookups == 0);
  std::puts("SDK-typed optional CUDA/NVML forwarding: ABI, absent/partial drivers, concurrency and cached lifetime passed");
}
