#include "runtime/windows-hip-driver.h"
#include <cstdlib>
#include <cstring>

namespace {
int loads = 0, lookups = 0, calls = 0;
bool available = true, exported = true;
hipError_t count(int* result) { ++calls; *result = 42; return static_cast<hipError_t>(17); }
const char* error(int) { ++calls; return "real driver error"; }
constexpr char countName[] = "hipGetDeviceCount", errorName[] = "hipGetErrorString";
using Count = mom::runtime::WindowsHipCall<hipError_t (*)(int*)>;
using Error = mom::runtime::WindowsHipCall<const char* (*)(int)>;
}

HMODULE LoadLibraryExA(const char* name, void* file, unsigned flags) {
  if (file || flags != LOAD_LIBRARY_SEARCH_SYSTEM32 ||
      (std::strcmp(name, "amdhip64_7.dll") && std::strcmp(name, "nvcuda.dll") && std::strcmp(name, "nvml.dll"))) std::abort();
  ++loads;
  return available ? reinterpret_cast<HMODULE>(1) : nullptr;
}
FARPROC GetProcAddress(HMODULE module, const char* name) {
  if (!module) std::abort();
  ++lookups;
  if (!exported) return nullptr;
  if (!std::strcmp(name, countName)) return reinterpret_cast<FARPROC>(&count);
  if (!std::strcmp(name, errorName)) return reinterpret_cast<FARPROC>(&error);
  std::abort();
}

int main(int argc, char** argv) {
#ifdef MOM_TEST_UNSUPPORTED_RESULT
  return mom::runtime::WindowsHipCall<int (*)(int)>::call<errorName>(0);
#else
  if (argc != 2 || loads || lookups || calls) return 1;
  const int mode = std::atoi(argv[1]);
  int value = -1;
  switch (mode) {
    case 0: return 0; // Including/binding the bridge must not load a driver at startup.
    case 1:
      available = false;
      return Count::call<countName>(&value) != hipErrorNoDevice || value != -1 || calls || loads != 1 || lookups;
    case 2:
      available = false;
      return std::strcmp(Error::call<errorName>(0), "AMD HIP system driver is unavailable") || calls || lookups;
    case 3:
      exported = false;
      return Count::call<countName>(&value) != hipErrorNoDevice || calls || lookups != 1;
    case 4: return Count::call<countName>(&value) != static_cast<hipError_t>(17) || value != 42 || calls != 1;
    case 5: return std::strcmp(Error::call<errorName>(0), "real driver error") || calls != 1;
    case 6:
      Count::call<countName>(&value); Error::call<errorName>(0);
      return loads != 1 || lookups != 2;
    case 7:
      Count::call<countName>(&value); Count::call<countName>(&value);
      return loads != 1 || lookups != 1 || calls != 2;
    case 8:
      available = false; Count::call<countName>(&value); available = true;
      return Count::call<countName>(&value) != hipErrorNoDevice || calls || loads != 1 || lookups;
    case 9:
      mom::runtime::systemDriverSymbol<mom::runtime::SystemDriver::cuda, hipError_t (*)(int*)>(countName);
      mom::runtime::systemDriverSymbol<mom::runtime::SystemDriver::nvml, const char* (*)(int)>(errorName);
      return loads != 2 || lookups != 2;
    default: return 2;
  }
#endif
}
