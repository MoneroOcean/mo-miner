#pragma once
#include <cstddef>
enum CUresult {CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_NO_DEVICE};
using CUdeviceptr = unsigned long long;
extern "C" {
CUresult cuInit(unsigned);
CUresult cuMemAlloc_v2(CUdeviceptr*, std::size_t);
CUresult cuGetErrorString(CUresult, const char**);
}
