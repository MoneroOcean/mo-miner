// Production ICD discovery trusts OS-registered drivers; the test opt-in selects no path from env.
#pragma once
#ifndef _WIN32
#error The Windows ICD boundary is Windows-only; Linux uses its existing driver-load guard.
#endif
#include <windows.h>
#include <stddef.h>
#include <string.h>

/* Compare `length` bytes with ASCII case folding. */
static int mom_native_ascii_iequal(const char* a, const char* b, size_t length) {
  for (size_t i = 0; i < length; ++i) {
    const unsigned char x = (unsigned char)a[i], y = (unsigned char)b[i];
    if ((x >= 'A' && x <= 'Z' ? x + 32 : x) != (y >= 'A' && y <= 'Z' ? y + 32 : y)) return 0;
  }
  return 1;
}

static int mom_native_portable_cpu_tests_enabled(void) {
  char value[2] = {0, 0};
  return GetEnvironmentVariableA("MOM_REQUIRE_PORTABLE_CPU_TESTS", value, sizeof(value)) == 1 &&
      value[0] == '1' && value[1] == 0;
}

/* Is this the Intel CPU OpenCL runtime's ICD under its registered Program Files root? */
static int mom_native_portable_cpu_opencl_path(const char* name, size_t length) {
  static const char root[] = "C:\\Program Files (x86)\\Common Files\\Intel\\Shared Libraries\\";
  static const char filename[] = "intelocl64.dll";
  const size_t rootLength = sizeof(root) - 1, filenameLength = sizeof(filename) - 1;
  if (length < rootLength + filenameLength || !mom_native_ascii_iequal(name, root, rootLength))
    return 0;
  const char* leaf = strrchr(name, '\\');
  return leaf && (size_t)(name + length - (leaf + 1)) == filenameLength &&
      mom_native_ascii_iequal(leaf + 1, filename, filenameLength);
}

/* Preserve the registered spelling; reject paths Windows could normalize outside System32.
 * Relative names use System32-only lookup. Absolute driver-store paths may resolve their own
 * companion DLLs, but never the executable directory, current directory or PATH. The optional
 * Intel CPU OpenCL ICD is admitted only from the standalone runtime's registered install root. */
static void* mom_native_opencl_load_driver(const char* name) {
  if (!name || !*name) return NULL;
  size_t length = 0;
  for (; name[length]; ++length) {
    const unsigned char c = (unsigned char)name[length];
    if (length >= 32767 || c < 32 || c == 127 || c == '/' || c == '"' ||
        c == '?' || c == '*' || c == '<' || c == '>' || c == '|' || (c == ':' && length != 1)) return NULL;
  }
  const int absolute = strchr(name, '\\') != NULL;
  const int portableCpu = absolute && mom_native_portable_cpu_tests_enabled() &&
      mom_native_portable_cpu_opencl_path(name, length);
  size_t start = 0;
  if (absolute) {
    if (!((name[0] >= 'A' && name[0] <= 'Z') || (name[0] >= 'a' && name[0] <= 'z')) ||
        name[1] != ':' || name[2] != '\\') return NULL;
    char system[MAX_PATH];
    if (!portableCpu) {
      const unsigned count = GetSystemDirectoryA(system, MAX_PATH);
      if (!count || count >= MAX_PATH || length <= count || name[count] != '\\') return NULL;
      // ASCII folding suffices for ordinary Windows directory names; unusual case aliases
      // fail closed rather than permitting a different directory through normalization.
      if (!mom_native_ascii_iequal(name, system, count)) return NULL;
    }
    start = 3;
  } else if (strchr(name, ':')) return NULL;
  for (size_t i = start; i <= length; ++i) {
    if (i == length || name[i] == '\\') {
      // This also excludes dot segments with trailing spaces/dots and empty components.
      if (i == start || name[i - 1] == '.' || name[i - 1] == ' ') return NULL;
      start = i + 1;
    }
  }
  return LoadLibraryExA(name, NULL, LOAD_LIBRARY_SEARCH_SYSTEM32 |
      (absolute ? LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR : 0));
}
