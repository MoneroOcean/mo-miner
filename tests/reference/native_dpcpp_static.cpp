// CPU-only dependency/lifecycle baseline; no queue or GPU kernel is dispatched.
#include "../../runtime/windows-stack.h"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <stdexcept>
#ifdef MOM_NATIVE_DPCPP_JIT
#include <detail/jit_compiler.hpp>
#include <cstdlib>
#include <cstring>

// Exercise the maintained runtime bridge, not only the compiler's C entrypoints. No platform,
// context, queue or device is needed to compile/own/retire these images and prefixed entries.
static void check_source_jit() {
  auto& compiler = sycl::detail::jit_compiler::get_instance();
  if (!compiler.isAvailable()) throw std::runtime_error("Static runtime source-JIT binding is unavailable");
  const char* selected = std::getenv("MOM_DPCPP_JIT_TARGET");
  const std::string target = selected ? selected : "spirv";
  if (target != "spirv" && target != "sm_61" && target != "sm_120") throw std::runtime_error("Unknown host-JIT target");
  const auto format = target == "spirv" ? jit_compiler::BinaryFormat::SPIRV : jit_compiler::BinaryFormat::PTX;
  const std::string source = R"SYCL(
#include <sycl/sycl.hpp>
namespace ex = sycl::ext::oneapi::experimental;
extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((ex::single_task_kernel))
void mom_host_jit_probe(int* output) {output[0] = 29;}
)SYCL";
  std::string first;
  for (int attempt = 0; attempt < 3; ++attempt) {
    std::string log;
    const auto compiled = compiler.compileSYCL("mom-native-host", source, {}, {"-std=c++17", "-O2"}, &log, format);
    struct Retire {
      sycl::detail::jit_compiler& compiler;
      sycl_device_binaries binaries;
      ~Retire() {if (binaries) compiler.destroyDeviceBinaries(binaries);}
    } retire{compiler, compiled.first};
    if (!compiled.first || compiled.first->NumDeviceBinaries != 1 || compiled.second != "mom-native-host$")
      throw std::runtime_error("Runtime lost JIT binary ownership/prefix");
    const auto& image = compiled.first->DeviceBinaries[0];
    const auto size = image.BinaryEnd - image.BinaryStart;
    if (size < 16 || size > (64 << 20)) throw std::runtime_error("Runtime returned an invalid JIT image");
    const std::string bytes(reinterpret_cast<const char*>(image.BinaryStart), size);
    if (!attempt) first = bytes;
    else if (bytes != first) throw std::runtime_error("Host-JIT cache/recovery changed the compiled image");
    if (target == "spirv") {
      uint32_t magic = 0; std::memcpy(&magic, image.BinaryStart, sizeof(magic));
      if (image.Format != SYCL_DEVICE_BINARY_TYPE_SPIRV || magic != 0x07230203)
        throw std::runtime_error("Runtime did not retain the SPIR-V image");
    } else if (bytes.find(".target " + target) == std::string::npos)
      throw std::runtime_error("Runtime selected the wrong PTX target");
    bool registered = false;
    std::string names;
    for (auto entry = image.EntriesBegin; entry != image.EntriesEnd; entry = entry->Increment()) {
      const std::string name = entry->GetName();
      names += name + ";";
      // DPC++ wraps an extern-C free function in its registered __sycl_kernel_ entry.
      registered |= name == "mom-native-host$__sycl_kernel_mom_host_jit_probe";
    }
    if (!registered) throw std::runtime_error("Runtime lost the prefixed free-function registration: " + names);
    if (!attempt) {
      bool rejected = false;
      try {
        const auto bad = compiler.compileSYCL("mom-native-invalid", "not C++;", {}, {"-std=c++17"}, &log, format);
        compiler.destroyDeviceBinaries(bad.first);
      } catch (const sycl::exception& error) {rejected = error.code() == sycl::errc::build;}
      if (!rejected) throw std::runtime_error("Host-JIT did not reject malformed source");
    }
  }
  std::printf("Static DPC++ host-JIT: %s binding/source/cache/recovery/registration/retirement passed\n", target.c_str());
}
#endif
#if defined(MOM_NATIVE_DPCPP_L0) || defined(MOM_NATIVE_DPCPP_CUDA) || defined(MOM_NATIVE_DPCPP_OPENCL)
#include <unified-runtime/ur_api.h>
#include <unified-runtime/ur_ddi.h>
#endif
#ifdef MOM_NATIVE_DPCPP_OPENCL
#include <CL/cl.h>
#include <CL/cl_ext.h>
namespace ur::opencl {ur_result_t urAdapterGetDdiTables(ur_dditable_t*);}
namespace ocl {
extern decltype(::clGetPlatformIDs)* clGetPlatformIDs_ptr;
bool loadOCLLibrary();
void unloadOCLLibrary();
}
#endif
#ifdef MOM_NATIVE_DPCPP_L0
namespace ur::level_zero {ur_result_t urAdapterGetDdiTables(ur_dditable_t*);}
namespace umf {ur_result_t getProviderNativeError(const char*, int32_t);}
extern "C" ur_result_t mom_ur_level_zero_v2_factory(ur_dditable_t*);
#endif
#ifdef MOM_NATIVE_DPCPP_CUDA
extern "C" ur_result_t mom_ur_cuda_factory(ur_dditable_t*);
#endif

int main() try {
#ifdef MOM_NATIVE_DPCPP_JIT_ONLY
  // Reuse the exact original runtime-bridge checks on Windows without enumerating drivers.
  // This build-only lane has no platform/device/queue dispatch or mining qualification.
  #ifndef MOM_NATIVE_DPCPP_JIT
    #error MOM_NATIVE_DPCPP_JIT_ONLY requires the real source-JIT bridge
  #endif
  check_source_jit();
  return 0;
#else
#ifdef MOM_NATIVE_DPCPP_OPENCL
  // Exercise the linked ORIGINAL ICD entrypoint and complete UR dispatch population. Merely
  // finding no devices would also pass with a missing adapter or a rejected libOpenCL dlopen.
  if (!ocl::loadOCLLibrary() || ocl::clGetPlatformIDs_ptr != &::clGetPlatformIDs)
    throw std::runtime_error("OpenCL adapter did not bind its linked ICD entrypoint");
  ocl::unloadOCLLibrary();
  if (!ocl::loadOCLLibrary() || ocl::clGetPlatformIDs_ptr != &::clGetPlatformIDs)
    throw std::runtime_error("Static OpenCL entrypoint lifetime changed");
  cl_uint platforms = 123;
  if (clGetPlatformIDs(0, nullptr, &platforms) != CL_PLATFORM_NOT_FOUND_KHR || platforms)
    throw std::runtime_error("Unexpected static OpenCL no-driver result");
  ur_dditable_t opencl{};
  uint32_t adapters = 123;
  if (ur::opencl::urAdapterGetDdiTables(&opencl) != UR_RESULT_SUCCESS || !opencl.Adapter.pfnGet ||
      !opencl.Platform.pfnGet || !opencl.EnqueueExp.pfnKernelLaunchWithArgsExp || urLoaderInit(0, nullptr) != UR_RESULT_SUCCESS ||
      opencl.Adapter.pfnGet(0, nullptr, &adapters) != UR_RESULT_ERROR_UNINITIALIZED || adapters)
    throw std::runtime_error("Static OpenCL adapter/ICD registration failed");
#endif
#ifdef MOM_NATIVE_DPCPP_JIT
  check_source_jit();
#endif
#ifdef MOM_NATIVE_DPCPP_CUDA
  ur_dditable_t cuda{};
  if (mom_ur_cuda_factory(nullptr) != UR_RESULT_ERROR_INVALID_NULL_POINTER ||
      mom_ur_cuda_factory(&cuda) != UR_RESULT_SUCCESS || !cuda.Adapter.pfnGet ||
      !cuda.Platform.pfnGet || !cuda.Device.pfnGet || !cuda.EnqueueExp.pfnKernelLaunchWithArgsExp ||
      urLoaderInit(0, nullptr) != UR_RESULT_SUCCESS)
    throw std::runtime_error("Static CUDA adapter/loader registration failed");
  uint32_t cuda_count = 0;
  if (urAdapterGet(0, nullptr, &cuda_count) != UR_RESULT_SUCCESS || cuda_count != 1)
    throw std::runtime_error("Static CUDA adapter was not registered exactly once");
#endif
#ifdef MOM_NATIVE_DPCPP_L0
  // A shared static image must retain this provider's real translator, not OpenCL's unused stub.
  if (umf::getProviderNativeError("Level Zero", UR_RESULT_ERROR_OUT_OF_HOST_MEMORY) != UR_RESULT_ERROR_OUT_OF_HOST_MEMORY ||
      umf::getProviderNativeError("OpenCL", UR_RESULT_ERROR_OUT_OF_HOST_MEMORY) != UR_RESULT_ERROR_UNKNOWN)
    throw std::runtime_error("Static adapters changed provider-specific error translation");
  // An absent-device check alone could pass with a broken static registry. Admit the complete
  // adapter table and loader first; neither action dispatches any kernel or needs a driver.
  ur_dditable_t table{};
  if (ur::level_zero::urAdapterGetDdiTables(&table) != UR_RESULT_SUCCESS || !table.Adapter.pfnGet ||
      urLoaderInit(0, nullptr) != UR_RESULT_SUCCESS)
    throw std::runtime_error("Static Level Zero adapter/loader registration failed");
  uint32_t count = 123;
  // Query the L0 table directly: a simultaneously registered empty CUDA adapter must not be
  // mistaken for an Intel driver, nor change this adapter's own absent-driver result.
  const auto result = table.Adapter.pfnGet(0, nullptr, &count);
  // This upstream adapter reports UNSUPPORTED_VERSION for its empty driver list, not an ABI
  // mismatch. Check both its explicit result and zero count rather than hiding init failures.
  if (count || result != UR_RESULT_ERROR_UNSUPPORTED_VERSION)
    throw std::runtime_error("Unexpected Level Zero no-driver result");
  // V2 shares only opaque handles/table ABI with V1, never its implementation symbols.
  ur_dditable_t v2{};
  if (mom_ur_level_zero_v2_factory(nullptr) != UR_RESULT_ERROR_INVALID_NULL_POINTER ||
      mom_ur_level_zero_v2_factory(&v2) != UR_RESULT_SUCCESS || !v2.Adapter.pfnGet ||
      !v2.Platform.pfnGet || !v2.EnqueueExp.pfnKernelLaunchWithArgsExp)
    throw std::runtime_error("Static Level Zero V2 dispatch table is incomplete");
  count = 123;
  if (v2.Adapter.pfnGet(0, nullptr, &count) != UR_RESULT_ERROR_UNSUPPORTED_VERSION || count)
    throw std::runtime_error("Unexpected Level Zero V2 no-driver result");
#endif
  for (int attempt = 0; attempt < 3; ++attempt) {
#ifdef MOM_NATIVE_DPCPP_CUDA
    // This original adapter exposes a CUDA platform even without devices/driver. Do not
    // mistake its empty platform for actual hardware, or silently admit a usable device.
    for (const auto& platform : sycl::platform::get_platforms()) {
      if (platform.get_backend() != sycl::backend::ext_oneapi_cuda || !platform.get_devices().empty())
        throw std::runtime_error("The absent-driver CUDA baseline unexpectedly found a device");
    }
#else
    if (!sycl::platform::get_platforms().empty())
      throw std::runtime_error("This baseline must have no GPU driver/device access");
#endif
    bool missing = false;
    try {const sycl::device device{sycl::gpu_selector_v}; (void)device;}
    catch (const sycl::exception&) {missing = true;}
    if (!missing) throw std::runtime_error("Unavailable device unexpectedly admitted");
    if (sycl::range<3>{2, 3, 5}.size() != 30 || sycl::sqrt(25.0f) != 5.0f)
      throw std::runtime_error("Maintained host SYCL API mismatch");
  }
  std::puts("Static DPC++ core: repeated absent-device/lifecycle and host API checks passed");
  return 0;
#endif
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what()); return 1;
}
