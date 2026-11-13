// Header-only ABI fixture: compile/link generated bindings without an SDK, driver or CRT.
#pragma once
enum class HipStatus { missing };
enum class CudaStatus { missing };
struct DriverPayload { unsigned long long values[2]; };
extern "C" {
HipStatus hipGetDeviceCount(int*);
const char* hipGetErrorString(HipStatus);
HipStatus hipCopyPayload(const DriverPayload*, void (*)(int));
CudaStatus cuInit(unsigned);
const char* nvmlErrorString(int);
}
namespace mom::runtime {
template<class> struct WindowsHipCall;
template<class R, class... A> struct WindowsHipCall<R(*)(A...)> {
  template<const char*> static R call(A...) { return R{}; }
};
template<class> struct WindowsCudaCall;
template<class R, class... A> struct WindowsCudaCall<R(*)(A...)> {
  template<const char*> static R call(A...) { return R{}; }
};
}
