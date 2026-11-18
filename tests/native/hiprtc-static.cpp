// Exercise the maintained compiler adapter without a HIP SDK, GPU, or shared library.
#include <cassert>
#include <cstddef>
#include <cstring>
#include <string>

using hiprtcProgram = void*;
using hiprtcResult = int;
unsigned library_queries = 0, creates = 0, destroys = 0;
[[maybe_unused]] constexpr int RTLD_NOW = 1, RTLD_LOCAL = 0;
void* dlopen(const char*, int) { ++library_queries; return nullptr; }
void* dlsym(void*, const char*) { assert(false); return nullptr; }
hiprtcResult hiprtcCreateProgram(hiprtcProgram* output, const char*, const char*, int,
                                const char* const*, const char* const*) {
  *output = &creates; ++creates; return 0;
}
hiprtcResult hiprtcCompileProgram(hiprtcProgram program, int count, const char* const*) {
  assert(program == &creates && count == 3); return 7;
}
hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram, std::size_t* size) { *size = 4; return 0; }
hiprtcResult hiprtcGetProgramLog(hiprtcProgram, char* out) { std::memcpy(out, "log", 4); return 0; }
hiprtcResult hiprtcGetCodeSize(hiprtcProgram, std::size_t* size) { *size = 5; return 0; }
hiprtcResult hiprtcGetCode(hiprtcProgram, char* out) { std::memcpy(out, "code", 5); return 0; }
hiprtcResult hiprtcDestroyProgram(hiprtcProgram* value) { assert(*value == &creates); ++destroys; *value = nullptr; return 0; }
const char* hiprtcGetErrorString(hiprtcResult value) { assert(value == 7); return "failure"; }

#include "hiprtc-contract.inc"

int main() {
  auto& api = mom::HiprtcApi::instance();
  assert(&api == &mom::HiprtcApi::instance());
#ifdef MOM_STATIC_HIPRTC
  assert(api.available() && api.error.empty() && library_queries == 0);
  assert(api.create_program == &hiprtcCreateProgram && api.compile_program == &hiprtcCompileProgram);
  assert(api.get_log_size == &hiprtcGetProgramLogSize && api.get_log == &hiprtcGetProgramLog);
  assert(api.get_code_size == &hiprtcGetCodeSize && api.get_code == &hiprtcGetCode);
  assert(api.destroy_program == &hiprtcDestroyProgram && api.error_string == &hiprtcGetErrorString);
  {
    mom::HiprtcProgram program(api);
    assert(api.create_program(program.address(), "source", "test.hip", 0, nullptr, nullptr) == 0);
    assert(api.compile_program(program.get(), 3, nullptr) == 7);
    std::size_t size = 0; char text[5]{};
    assert(api.get_log_size(program.get(), &size) == 0 && size == 4);
    assert(api.get_log(program.get(), text) == 0 && std::string(text) == "log");
    assert(api.get_code_size(program.get(), &size) == 0 && size == 5);
    assert(api.get_code(program.get(), text) == 0 && std::string(text) == "code");
    assert(std::string(api.error_string(7)) == "failure");
    program.reset(); program.reset();
  }
  assert(creates == 1 && destroys == 1 && library_queries == 0);
#else
  assert(!api.available() && api.error == "HIPRTC library is not installed");
  assert(library_queries == 3 && creates == 0 && destroys == 0);
#endif
}
