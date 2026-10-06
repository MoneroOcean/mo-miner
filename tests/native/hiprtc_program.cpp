#include <algorithm>
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
#include <type_traits>
#include <vector>

static bool fail_allocation = false;
static unsigned allocation_faults = 0;
void* operator new(std::size_t size) {
  if (fail_allocation) {
    fail_allocation = false;
    ++allocation_faults;
    throw std::bad_alloc();
  }
  if (void* memory = std::malloc(size ? size : 1))
    return memory;
  throw std::bad_alloc();
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

using hiprtcProgram = void*;
using hiprtcResult = int;
using hipDevice_t = int;
using hipStream_t = void*;
using hipModule_t = void*;
using hipFunction_t = void*;
using hipEvent_t = void*;
using hipError_t = int;
constexpr int HIPRTC_SUCCESS = 0;
constexpr int hipSuccess = 0;
struct hipDeviceProp_t { char gcnArchName[64] = "gfx1201:feature"; };
static int fault = 0;
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
  unsigned module_load = 0;
  unsigned load_with_program = 0;
} counts;

static hiprtcResult create_program(hiprtcProgram* output, const char* source, const char* name,
                                  int, const char* const*, const char* const*) {
  ++counts.creates;
  (void)source;
  (void)name;
  if (fault == 2)
    return 1;
  if (fault == 12) {
    *output = nullptr;
    return HIPRTC_SUCCESS;
  }
  *output = &handles[0];
  program_live = true;
  return fault == 3 ? 1 : HIPRTC_SUCCESS;
}
static hiprtcResult compile_program(hiprtcProgram program, int count, const char* const* options) {
  ++counts.compile;
  (void)count;
  (void)options;
  return !program || fault == 4 || fault == 5 || fault == 6 || fault == 9
      ? 1 : HIPRTC_SUCCESS;
}
static hiprtcResult get_log_size(hiprtcProgram program, std::size_t* size) {
  ++counts.log_size;
  if (!program || fault == 5)
    return 1;
  *size = 4096;
  if (fault == 9)
    fail_allocation = true;
  return HIPRTC_SUCCESS;
}
static hiprtcResult get_log(hiprtcProgram, char* output) {
  ++counts.log;
  if (fault == 6)
    return 1;
  std::strcpy(output, "fake compiler diagnostic");
  return HIPRTC_SUCCESS;
}
static hiprtcResult get_code_size(hiprtcProgram, std::size_t* size) {
  ++counts.code_size;
  if (fault == 7)
    return 1;
  *size = 4096;
  if (fault == 10)
    fail_allocation = true;
  return HIPRTC_SUCCESS;
}
static hiprtcResult get_code(hiprtcProgram, char* output) {
  ++counts.code;
  if (fault == 8)
    return 1;
  std::memset(output, 0, 4096);
  return HIPRTC_SUCCESS;
}
static hiprtcResult destroy_program(hiprtcProgram* program) {
  if (*program) {
    ++counts.destroys;
    program_live = false;
  } else {
    ++counts.null_destroys;
  }
  // Deliberately do not clear the API out-parameter: the owner must clear its own record.
  return HIPRTC_SUCCESS;
}
static const char* error_string(hiprtcResult) { return "fake compiler error"; }
static hipError_t hipSetDevice(hipDevice_t) { return hipSuccess; }
static hipError_t hipGetDeviceProperties(hipDeviceProp_t* properties, hipDevice_t) {
  if (fault == 11)
    std::strcpy(properties->gcnArchName, "gfx1100");
  return hipSuccess;
}
static const char* hipGetErrorString(hipError_t) { return "fake HIP error"; }
static hipError_t hipModuleLoadData(hipModule_t* module, const void*) {
  ++counts.module_load;
  if (program_live)
    ++counts.load_with_program;
  *module = &handles[1];
  return hipSuccess;
}
static hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t, const char*) {
  *function = &handles[2];
  return hipSuccess;
}
static hipError_t hipEventCreate(hipEvent_t* event) {
  *event = &handles[3];
  return hipSuccess;
}
static hipError_t hipEventDestroy(hipEvent_t) { return hipSuccess; }
static hipError_t hipFree(void*) { return hipSuccess; }
static hipError_t hipModuleUnload(hipModule_t) { return hipSuccess; }
static hipError_t hipMalloc(void** memory, std::size_t) {
  *memory = &handles[4];
  return hipSuccess;
}
namespace sycl {
struct device {};
struct queue { device get_device() const { return {}; } };
}
static bool mom_is_hip(const sycl::device&) { return true; }
namespace mom {
namespace amd {
bool has_gfx12_int8_wmma(const char* arch) { return std::strncmp(arch, "gfx12", 5) == 0; }
}
void hip_queue_identity(sycl::queue&, hipDevice_t& device, hipStream_t& stream) {
  device = 0;
  stream = &handles[5];
}
void hip_queue_device(sycl::queue& queue, hipDevice_t& device) {
  hipStream_t stream = nullptr;
  hip_queue_identity(queue, device, stream);
}
class HiprtcApi {
public:
  decltype(&::create_program) create_program = ::create_program;
  decltype(&::compile_program) compile_program = ::compile_program;
  decltype(&::get_log_size) get_log_size = ::get_log_size;
  decltype(&::get_log) get_log = ::get_log;
  decltype(&::get_code_size) get_code_size = ::get_code_size;
  decltype(&::get_code) get_code = ::get_code;
  decltype(&::destroy_program) destroy_program = ::destroy_program;
  decltype(&::error_string) error_string = ::error_string;
  std::string error = "HIPRTC unavailable";
  bool available() const { return fault != 1; }
  static HiprtcApi& instance() {
    static HiprtcApi api;
    return api;
  }
};
namespace jit_cache {
std::uint64_t hash(const std::string&) { return 1; }
std::filesystem::path directory() { return {}; }
std::vector<char> read(const std::filesystem::path&) { return {}; }
void write(const std::filesystem::path&, const std::vector<char>&) {}
}
}

#include "owner.inc"
#include "octopus.inc"
#include "walahash.inc"
#include "pearl.inc"

static void check(bool valid, const char* message) {
  if (!valid)
    throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  try {
    check(argc == 3, "expected algorithm and case");
    const int algorithm = std::atoi(argv[1]);
    fault = std::atoi(argv[2]);
    check(algorithm >= 1 && algorithm <= 3, "invalid algorithm");
    check(fault >= 0 && fault <= 14, "invalid case");
#if defined(MOM_FIXTURE_HAS_OWNER)
    static_assert(!std::is_copy_constructible_v<mom::HiprtcProgram>);
    static_assert(!std::is_copy_assignable_v<mom::HiprtcProgram>);
    if (fault == 13) {
      {
        mom::HiprtcProgram program(mom::HiprtcApi::instance());
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
        mom::HiprtcProgram program(mom::HiprtcApi::instance());
        check(create_program(program.address(), "source", "owner", 0, nullptr, nullptr) == 0,
              "owner create failed");
        program.reset();
        check(program.get() == nullptr, "reset retained handle");
        program.reset();
      }
      check(counts.destroys == 1 && !program_live, "reset/destructor repeated destruction");
      std::puts("PASS explicit reset");
      return 0;
    }
#else
    check(fault < 13, "owner cases require candidate");
#endif
    bool result = false;
    bool allocation = false;
    sycl::queue queue;
    try {
      if (algorithm == 1) {
        OctopusAmdWmma value;
        result = value.ensure(queue, 64);
      } else if (algorithm == 2) {
        WalaHashAmdWmma value;
        result = value.ensure(queue);
      } else {
        PearlHashHipSearch value;
        result = value.ensure(queue, 256, 256, 2048, 128);
      }
    } catch (const std::bad_alloc&) {
      allocation = true;
    }
    const bool has_program = fault != 1 && fault != 2 && fault != 11 && fault != 12;
    check(counts.creates == (fault == 1 || fault == 11 ? 0u : 1u), "create count changed");
    check(counts.destroys == (has_program ? 1u : 0u), "temporary program not destroyed once");
    check(!program_live, "temporary program leaked");
    check(counts.null_destroys == 0, "null program passed to destroy");
    const unsigned failure_calls[] = {
        counts.compile, counts.log_size, counts.log, counts.code_size, counts.code};
    if (fault >= 4 && fault <= 8)
      check(failure_calls[fault - 4] == 1, "fault injection stage was not reached");
    check(counts.load_with_program == 0, "module loaded before compiler program destruction");
    check(result == (fault == 0), "fallback/success result changed");
    check(allocation == ((fault == 9 || fault == 10) && algorithm != 3),
          "allocation exception boundary changed");
    check(allocation_faults == (fault == 9 || fault == 10 ? 1u : 0u),
          "allocation injection did not execute");
    std::puts("PASS actual-source RTC lifetime");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
