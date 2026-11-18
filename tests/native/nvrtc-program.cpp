#include <algorithm>
#include <array>
#include <string_view>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

static bool fail_allocation = false;
static unsigned allocation_faults = 0;
// Keep custom allocation hooks opaque to GCC's inlined new/delete mismatch analysis.
[[gnu::noinline]] void* operator new(std::size_t size) {
  if (fail_allocation) {
    fail_allocation = false;
    ++allocation_faults;
    throw std::bad_alloc();
  }
  if (void* memory = std::malloc(size ? size : 1))
    return memory;
  throw std::bad_alloc();
}
[[gnu::noinline]] void operator delete(void* memory) noexcept { std::free(memory); }
[[gnu::noinline]] void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

using nvrtcProgram = void*;
using nvrtcResult = int;
using CUresult = int;
using CUdevice = int;
using CUdevice_attribute = int;
using CUmodule = void*;
using CUfunction = void*;
using CUevent = void*;
using CUcontext = void*;
constexpr int NVRTC_SUCCESS = 0;
constexpr int CUDA_SUCCESS = 0;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 1;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 2;
constexpr int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 3;
constexpr int CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 4;
constexpr int CU_EVENT_DEFAULT = 0;
constexpr int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 5;
constexpr int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 6;
constexpr int CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 7;
constexpr int CU_DEVICE_ATTRIBUTE_WARP_SIZE = 8;
static int variant;
static int fault;
static std::string compiled_architecture;
static int handles[8];
static bool program_live = false;
struct Counts {
  unsigned creates = 0;
  unsigned destroys = 0;
  unsigned null_destroys = 0;
  unsigned compile = 0;
  unsigned log_size = 0;
  unsigned log = 0;
  unsigned code_size = 0;
  unsigned code = 0;
  unsigned cubin_size = 0;
  unsigned cubin = 0;
  unsigned ptx_size = 0;
  unsigned ptx = 0;
  unsigned module_load = 0;
  unsigned late_load = 0;
  unsigned release_with_program = 0;
} counts;
static nvrtcResult create_program(nvrtcProgram* output, const char*, const char*,
                                 int, const char* const*, const char* const*) {
  ++counts.creates;
  if (fault == 2)
    return 1;
  if (fault == 12) {
    *output = nullptr;
    return NVRTC_SUCCESS;
  }
  *output = &handles[0];
  program_live = true;
  return fault == 3 ? 1 : NVRTC_SUCCESS;
}
static nvrtcResult compile_program(nvrtcProgram program, int count, const char* const* options) {
  ++counts.compile;
  for (int i = 0; i < count; ++i)
    if (std::string(options[i]).find("--gpu-architecture=") == 0)
      compiled_architecture = options[i];
  return !program || fault == 4 || fault == 5 || fault == 6 || fault == 9
      ? 1 : NVRTC_SUCCESS;
}
static nvrtcResult get_log_size(nvrtcProgram program, std::size_t* size) {
  ++counts.log_size;
  if (!program || fault == 5)
    return 1;
  *size = 4096;
  if (fault == 9)
    fail_allocation = true;
  return NVRTC_SUCCESS;
}
static nvrtcResult get_log(nvrtcProgram, char* output) {
  ++counts.log;
  if (fault == 6)
    return 1;
  std::strcpy(output, "fake compiler diagnostic");
  return NVRTC_SUCCESS;
}
static nvrtcResult get_code_size(nvrtcProgram, std::size_t* size) {
  ++counts.code_size;
  if (fault == 7)
    return 1;
  *size = fault == 25 ? 0 : fault == 26 ? 128 * 1024 * 1024 + 1 : 4096;
  if (fault == 10)
    fail_allocation = true;
  return NVRTC_SUCCESS;
}
static nvrtcResult get_code(nvrtcProgram, char* output) {
  ++counts.code;
  if (fault == 8 || fault == 25 || fault == 26)
    return 1;
  std::memset(output, 0, 4096);
  return NVRTC_SUCCESS;
}
static nvrtcResult get_ptx_size(nvrtcProgram program, std::size_t* size) {
  ++counts.ptx_size;
  return get_code_size(program, size);
}
static nvrtcResult get_ptx(nvrtcProgram program, char* output) {
  ++counts.ptx;
  return get_code(program, output);
}
static nvrtcResult get_cubin_size(nvrtcProgram program, std::size_t* size) {
  ++counts.cubin_size;
  return get_code_size(program, size);
}
static nvrtcResult get_cubin(nvrtcProgram program, char* output) {
  ++counts.cubin;
  return get_code(program, output);
}
static nvrtcResult destroy_program(nvrtcProgram* program) {
  if (*program) {
    ++counts.destroys;
    program_live = false;
  } else {
    ++counts.null_destroys;
  }
  // Do not clear the API argument: the owner must clear its own record.
  return NVRTC_SUCCESS;
}
static const char* error_string(nvrtcResult) { return "fake compiler error"; }
namespace sycl {
enum class backend { ext_oneapi_cuda, cuda, opencl };
struct device {
  backend get_backend() const {
    return fault == 16 ? backend::opencl : backend::ext_oneapi_cuda;
  }
};
struct queue { device get_device() const { return {}; } };
template<backend> CUdevice get_native(const device&) { return 0; }
}
static bool mom_is_cuda(const sycl::device& device) {
  return device.get_backend() == sycl::backend::ext_oneapi_cuda;
}
namespace mom {
class CudaDriverApi {
public:
  std::string error = "driver unavailable";
  bool basic_available() const { return true; }
  bool extended_available() const { return true; }
  int (*init)(unsigned) = +[](unsigned) { return 0; };
  int (*device_attribute)(int*, int, int) = +[](int* output, int attribute, int) {
    *output = attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        ? (fault == 11 ? 7 : 12) : 0;
    return 0;
  };
  int (*retain_primary)(void**, int) = +[](void** context, int) {
    *context = &handles[1];
    return 0;
  };
  int (*set_current)(void*) = +[](void*) { return 0; };
  int (*synchronize)() = +[]() { return 0; };
  int (*release_primary)(int) = +[](int) {
    if (program_live)
      ++counts.release_with_program;
    return 0;
  };
  int (*module_load)(void**, const void*) = +[](void** module, const void*) {
    ++counts.module_load;
    if (program_live)
      ++counts.late_load;
    *module = &handles[2];
    return 0;
  };
  int (*module_unload)(void*) = +[](void*) { return 0; };
  int (*module_function)(void**, void*, const char*) = +[](void** function, void*, const char*) {
    *function = &handles[3];
    return 0;
  };
  int (*function_attribute)(void*, int, int) = +[](void*, int, int) { return 0; };
  int (*event_create)(void**, unsigned) = +[](void** event, unsigned) {
    *event = &handles[4];
    return 0;
  };
  int (*event_destroy)(void*) = +[](void*) { return 0; };
  int (*error_string)(int, const char**) = +[](int, const char** detail) {
    *detail = "fake driver error";
    return 0;
  };
  static CudaDriverApi& instance() {
    static CudaDriverApi api;
    return api;
  }
};
class NvrtcApi {
public:
  decltype(&::create_program) create_program = ::create_program;
  decltype(&::compile_program) compile_program = ::compile_program;
  decltype(&::get_log_size) get_log_size = ::get_log_size;
  decltype(&::get_log) get_log = ::get_log;
  decltype(&::get_ptx_size) get_ptx_size = ::get_ptx_size;
  decltype(&::get_ptx) get_ptx = ::get_ptx;
  decltype(&::get_cubin_size) get_cubin_size = ::get_cubin_size;
  decltype(&::get_cubin) get_cubin = ::get_cubin;
  decltype(&::destroy_program) destroy_program = ::destroy_program;
  decltype(&::error_string) error_string = ::error_string;
  std::string error = "NVRTC unavailable";
  int (*version)(int*, int*) = +[](int* major, int* minor) {
    *major = 13; *minor = 3; return NVRTC_SUCCESS;
  };
  bool ptx_available() const { return fault != 1; }
  bool cubin_available() const { return fault != 1 && variant != 5; }
  int (*get_num_supported_archs)(int*) = +[](int* count) {
    *count = fault == 22 ? 0 : fault == 23 ? 129 : 1;
    return fault == 21 ? 1 : 0;
  };
  int (*get_supported_archs)(int*) = +[](int* archs) {
    *archs = variant == 3 || variant == 4 ? 80 : 120;
    return fault == 24 ? 1 : 0;
  };
  static NvrtcApi& instance() {
    static NvrtcApi api;
    return api;
  }
};
namespace jit_cache {
std::uint64_t hash(std::string_view) { return 1; }
std::filesystem::path directory() {
  return fault == 15 ? std::filesystem::path("fixture-cache") : std::filesystem::path{};
}
std::vector<char> read(const std::filesystem::path&, size_t = size_t(-1)) { return {0}; }
void write(const std::filesystem::path&, const std::vector<char>&) {}
}
}
#include "api.inc"
#include "cn.inc"
using PearlHashCudaDriverApi = mom::CudaDriverApi;
using PearlHashNvrtcApi = mom::NvrtcApi;
using mom::cuda_check;
using mom::rtc_error;
static std::filesystem::path pearlhash_cuda_header_root(
    const char*, const std::vector<std::filesystem::path>&, const std::filesystem::path&) {
  return "fixture-headers";
}
alignas(16) static unsigned char fixture_ap[32];
alignas(16) static unsigned char fixture_bp[32];
struct Buffers {
  void* Ap = fixture_ap;
  void* Bp = fixture_bp;
  void* EAL = fixture_ap;
  void* EBRt = fixture_bp;
  void* transcript = &handles[5];
  int transcript_rows = 128;
};
// Keep legacy source-JIT lifetime separate from actual compact admission tests.
namespace compact_private {
static bool selected(int, int, int, int) { return false; }
}
#include "pearl.inc"

static void check(bool valid, const char* message) {
  if (!valid)
    throw std::runtime_error(message);
}
static void check_candidate_shape() {
  check(PearlHashCudaSearch::supports_shape(131072, 524288, 8192, 128),
        "rank-128 full shape was rejected");
  check(PearlHashCudaSearch::supports_shape(128, 256, 4096, 256),
        "rank-256 tile shape was rejected");
  check(!PearlHashCudaSearch::supports_shape(127, 256, 8192, 128),
        "unaligned M shape was admitted");
  check(!PearlHashCudaSearch::supports_shape(128, 255, 8192, 128),
        "unaligned N shape was admitted");
  check(!PearlHashCudaSearch::supports_shape(128, 256, 4096, 128),
        "rank/K mismatch was admitted");
}
int main(int argc, char** argv) {
  try {
    check(argc == 3, "expected variant and case");
    variant = std::atoi(argv[1]);
    fault = std::atoi(argv[2]);
    check_candidate_shape();
    check(variant >= 1 && variant <= 5, "invalid variant");
    check(fault >= 0 && fault <= 26, "invalid case");
    const bool cn_variant = variant == 1 || variant == 4 || variant == 5;
#if defined(MOM_FIXTURE_HAS_OWNER)
    static_assert(!std::is_copy_constructible_v<mom::NvrtcProgram>);
    static_assert(!std::is_copy_assignable_v<mom::NvrtcProgram>);
    if (fault == 13) {
      {
        mom::NvrtcProgram program(mom::NvrtcApi::instance());
        check(program.get() == nullptr, "empty program is not null");
        program.reset();
        program.reset();
      }
      check(counts.destroys == 0 && counts.null_destroys == 0, "empty program was destroyed");
      std::puts("PASS empty owner");
      return 0;
    }
    if (fault == 14) {
      {
        mom::NvrtcProgram program(mom::NvrtcApi::instance());
        check(create_program(program.address(), "source", "owner", 0, nullptr, nullptr) == 0,
              "owner creation failed");
        program.reset();
        check(program.get() == nullptr, "reset retained handle");
        program.reset();
      }
      check(counts.destroys == 1 && !program_live, "reset/destructor repeated destruction");
      std::puts("PASS explicit reset");
      return 0;
    }
#else
    check(fault != 13 && fault != 14, "owner cases require candidate");
#endif
    bool result = false;
    std::string reason;
    sycl::queue queue;
    if (cn_variant) {
      CnGpuCudaExpand value;
      result = value.ensure(queue, &reason);
    } else {
      PearlHashCudaSearch value;
      Buffers buffers;
      if (fault == 17)
        buffers.Ap = nullptr;
      else if (fault == 18)
        buffers.Bp = nullptr;
      else if (fault == 19)
        buffers.Ap = fixture_ap + 1;
      else if (fault == 20)
        buffers.Bp = fixture_bp + 1;
      result = value.ensure(queue, buffers, 128, 256, 8192, 128);
      if (fault == 0) {
        value.release();
        value.release();
        check(!value.enabled && !value.checked && !value.module && !value.context &&
                  !value.started && !value.done && !value.packed,
              "release did not clear candidate state");
      }
    }
    const bool invalid_packed_buffers = !cn_variant && fault >= 17 && fault <= 20;
    const bool creates = fault != 1 && fault != 11 && fault != 15 && fault != 16 &&
        !(fault >= 21 && fault <= 24) && !invalid_packed_buffers;
    const bool owns = creates && fault != 2 && fault != 12;
    check(counts.creates == (creates ? 1u : 0u), "create count changed");
    check(counts.destroys == (owns ? 1u : 0u), "temporary program not destroyed once");
    check(!program_live, "temporary program leaked");
    check(counts.null_destroys == 0, "null program passed to destroy");
    check(counts.late_load == 0, "module loaded before program destruction");
    check(counts.release_with_program == 0, "driver resources released before program");
    check(result == (fault == 0 || fault == 15), "fallback/success result changed");
    check(allocation_faults == (fault == 9 || fault == 10 ? 1u : 0u),
          "allocation injection did not execute");
    const unsigned failure_calls[] = {
        counts.compile, counts.log_size, counts.log, counts.code_size, counts.code};
    if (fault >= 4 && fault <= 8)
      check(failure_calls[fault - 4] == 1, "fault injection stage was not reached");
    if (counts.code_size)
      check((variant == 1 || variant == 2 ? counts.cubin_size : counts.ptx_size) == counts.code_size,
            "PTX/CUBIN size selection changed");
    if (counts.code)
      check((variant == 1 || variant == 2 ? counts.cubin : counts.ptx) == counts.code,
            "PTX/CUBIN retrieval selection changed");
    if (cn_variant && counts.compile)
      check(compiled_architecture == (variant == 1 ? "--gpu-architecture=sm_120" :
             "--gpu-architecture=compute_80"), "CN native/forward compiler target changed");
    if (fault == 25 || fault == 26)
      check(counts.code_size == 1 && counts.code == 0, "invalid output size reached retrieval");
    std::puts("PASS actual-source NVRTC lifetime");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
