#pragma once
#ifndef NVML_STATIC_IMPORT
#error "The bridge must use the documented NVML static import policy."
#endif
enum nvmlReturn_t {NVML_SUCCESS, NVML_ERROR_UNINITIALIZED};
using nvmlDevice_t = struct NvmlDevice*;
extern "C" {
nvmlReturn_t nvmlInit_v2();
const char* nvmlErrorString(nvmlReturn_t);
nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t, unsigned*);
}
