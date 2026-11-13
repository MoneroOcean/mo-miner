// CPU-only final-loader spy: real metadata code and driver-symbol helper, no real driver loads.
#include "runtime/nvidia-metadata.h"
#include <nvml.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <climits>
#include <new>
#include <string>
#include <vector>

static std::string mode;
static unsigned loads = 0, lookups = 0, inits = 0, releases = 0, references = 3;
extern "C" nvmlReturn_t nvmlInit_v2() {
  ++inits; if (mode == "init-error") return NVML_ERROR_UNKNOWN;
  ++references; return NVML_SUCCESS;
}
extern "C" nvmlReturn_t nvmlShutdown() {
  ++releases; --references;
  return mode == "shutdown-error" ? NVML_ERROR_UNKNOWN : NVML_SUCCESS;
}
extern "C" nvmlReturn_t nvmlDeviceGetCount_v2(unsigned* count) {
  *count = mode == "zero" ? 0 : mode == "huge-count" ? UINT_MAX : 2;
  return mode == "count-error" ? NVML_ERROR_UNKNOWN : NVML_SUCCESS;
}
extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned index, nvmlDevice_t* device) {
  *device = mode == "null-handle" ? nullptr : reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(index + 1));
  return mode == "handle-error" ? NVML_ERROR_UNKNOWN : NVML_SUCCESS;
}
extern "C" nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) {
  if (mode == "throw") throw std::bad_alloc();
  if (mode == "unset") return NVML_SUCCESS;
  const auto index = reinterpret_cast<uintptr_t>(device) - 1;
  *major = index == 0 ? 12 : 7; *minor = 0;
  if (mode == "overflow") *major = INT_MAX;
  if (mode == "negative-major") *major = -1;
  if (mode == "zero-major") *major = 0;
  if (mode == "negative-minor") *minor = -1;
  if (mode == "bad-minor") *minor = 10;
  return mode == "second-error" && index == 1 ? NVML_ERROR_UNKNOWN : NVML_SUCCESS;
}
extern "C" void* dlopen(const char* name, int flags) noexcept {
  ++loads;
  if (std::strcmp(name, "libnvidia-ml.so.1") || flags != (RTLD_NOW | RTLD_LOCAL)) return nullptr;
  return mode == "missing-driver" ? nullptr : &loads;
}
extern "C" void* dlsym(void* library, const char* name) noexcept {
  ++lookups;
  if (library != &loads || (mode == "missing-symbol" && std::strcmp(name, "nvmlDeviceGetCudaComputeCapability") == 0)) return nullptr;
#define ENTRY(function) if (std::strcmp(name, #function) == 0) return reinterpret_cast<void*>(&function)
  ENTRY(nvmlInit_v2); ENTRY(nvmlShutdown); ENTRY(nvmlDeviceGetCount_v2);
  ENTRY(nvmlDeviceGetHandleByIndex_v2); ENTRY(nvmlDeviceGetCudaComputeCapability);
#undef ENTRY
  return nullptr;
}
int main(int argc, char** argv) {
  if (argc != 2) return 86;
  mode = argv[1];
  const bool valid = mode == "mixed" || mode == "twice";
  const unsigned calls = mode == "twice" ? 2 : 1;
  for (unsigned i = 0; i < calls; ++i) {
    bool thrown = false; std::vector<unsigned> result;
    try {result = mom::runtime::nvidiaComputeCapabilities();} catch (const std::bad_alloc&) {thrown = true;}
    if (thrown != (mode == "throw") || (valid ? result != std::vector<unsigned>{120, 70} : !result.empty())) return 86;
  }
  const bool initialized = mode != "stub" && mode != "missing-driver" && mode != "missing-symbol";
  const bool owned = initialized && mode != "init-error";
  if (inits != (initialized ? calls : 0) || releases != (owned ? calls : 0) || references != 3) return 86;
  if (loads != (mode == "stub" ? 0u : 1u) || lookups != (mode == "stub" || mode == "missing-driver" ? 0u : 5u)) return 86;
  std::puts("metadata/result/driver identity/refcounts/cleanup passed; no real GPU driver");
}
