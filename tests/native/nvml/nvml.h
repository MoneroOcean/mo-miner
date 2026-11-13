// Test-only SDK stand-in. Real integration must compile against the independently pinned header.
#pragma once
struct nvmlDevice_st;
using nvmlDevice_t = nvmlDevice_st*;
enum nvmlReturn_t {NVML_SUCCESS = 0, NVML_ERROR_UNKNOWN = 999};
extern "C" {
nvmlReturn_t nvmlInit_v2();
nvmlReturn_t nvmlShutdown();
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned*);
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned, nvmlDevice_t*);
nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t, int*, int*);
}
