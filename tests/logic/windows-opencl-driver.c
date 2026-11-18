// The real native ICD path guard, compiled as both C and C++; no Windows SDK or GPU needed.
#include "runtime/windows-opencl-driver.h"
#include <stdlib.h>

static unsigned loads, directoryResult = 19;
static const char* expected;
static unsigned expectedFlags;
static int available = 1;
static const char* portableCpuOptIn;
unsigned GetEnvironmentVariableA(const char* name, char* buffer, unsigned size) {
  if (strcmp(name, "MOM_REQUIRE_PORTABLE_CPU_TESTS") || size != 2) abort();
  if (!portableCpuOptIn) return 0;
  const size_t length = strlen(portableCpuOptIn);
  if (length >= size) return (unsigned)length + 1;
  memcpy(buffer, portableCpuOptIn, length + 1);
  return (unsigned)length;
}
unsigned GetSystemDirectoryA(char* buffer, unsigned size) {
  if (size != MAX_PATH) abort();
  if (directoryResult == 19) memcpy(buffer, "C:\\Windows\\System32", 20);
  return directoryResult;
}
HMODULE LoadLibraryExA(const char* name, void* file, unsigned flags) {
  if (file || !expected || strcmp(name, expected) || flags != expectedFlags) abort();
  ++loads; return available ? (HMODULE)1 : NULL;
}
static int allow(const char* name, int absolute) {
  expected = name; expectedFlags = LOAD_LIBRARY_SEARCH_SYSTEM32 | (absolute ? LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR : 0);
  const unsigned before = loads;
  return mom_native_opencl_load_driver(name) != (HMODULE)1 || loads != before + 1;
}
int main(void) {
  if (loads) return 1;
  if (allow("IntelOpenCL64.dll", 0) || allow("nvopencl64.dll", 0) ||
      allow("C:\\Windows\\System32\\amdocl64.dll", 1) ||
      allow("c:\\wInDoWs\\sYsTeM32\\DriverStore\\FileRepository\\intel\\igdrcl64.dll", 1)) return 2;
  const char* bad[] = {NULL, "", ".", "..", "driver.dll.", "driver.dll ", "x:driver.dll", "C:driver.dll",
    "foo\\driver.dll", "\\\\server\\share\\driver.dll", "\\\\?\\C:\\Windows\\System32\\driver.dll",
    "C:\\Users\\sap\\driver.dll", "C:\\Windows\\System32-other\\driver.dll", "C:/Windows/System32/driver.dll",
    "C:\\Windows\\System32\\..\\driver.dll", "C:\\Windows\\System32\\.. \\driver.dll",
    "C:\\Windows\\System32\\.\\driver.dll", "C:\\Windows\\System32\\x.\\driver.dll",
    "C:\\Windows\\System32\\\\driver.dll", "C:\\Windows\\System32\\driver.dll:stream",
    "C:\\Windows\\System32\\driver.dll\\", "%SystemRoot%\\System32\\driver.dll", "bad\n.dll", "bad?.dll"};
  const unsigned before = loads;
  for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); ++i)
    if (mom_native_opencl_load_driver(bad[i]) || loads != before) return 3;
  const char* cpuDriver = "C:\\Program Files (x86)\\Common Files\\Intel\\Shared Libraries\\runtime\\intelocl64.dll";
  if (mom_native_opencl_load_driver(cpuDriver) || loads != before) return 8;
  portableCpuOptIn = "0";
  if (mom_native_opencl_load_driver(cpuDriver) || loads != before) return 9;
  portableCpuOptIn = "10";
  if (mom_native_opencl_load_driver(cpuDriver) || loads != before) return 10;
  portableCpuOptIn = "1";
  if (allow(cpuDriver, 1)) return 11;
  const unsigned afterCpu = loads;
  const char* badCpu[] = {
    "C:\\Program Files (x86)\\Common Files\\Intel\\Shared Libraries-other\\intelocl64.dll",
    "C:\\Program Files (x86)\\Common Files\\Intel\\Shared Libraries\\..\\intelocl64.dll",
    "C:\\Program Files (x86)\\Common Files\\Intel\\Shared Libraries\\runtime\\other.dll",
    "C:\\Users\\sap\\intelocl64.dll"};
  for (size_t i = 0; i < sizeof(badCpu) / sizeof(*badCpu); ++i)
    if (mom_native_opencl_load_driver(badCpu[i]) || loads != afterCpu) return 12;
  portableCpuOptIn = NULL;
  char longName[32769]; memset(longName, 'x', sizeof(longName)); longName[32768] = 0;
  if (mom_native_opencl_load_driver(longName) || loads != afterCpu) return 4;
  directoryResult = 0;
  if (mom_native_opencl_load_driver("C:\\Windows\\System32\\driver.dll") || loads != afterCpu) return 5;
  directoryResult = MAX_PATH;
  if (mom_native_opencl_load_driver("C:\\Windows\\System32\\driver.dll") || loads != afterCpu) return 6;
  available = 0; expected = "nvopencl64.dll"; expectedFlags = LOAD_LIBRARY_SEARCH_SYSTEM32;
  if (mom_native_opencl_load_driver(expected) || loads != afterCpu + 1) return 7;
  return 0;
}
