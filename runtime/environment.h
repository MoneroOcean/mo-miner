// Keep the native controller's OS environment and statically linked C runtime coherent.
#pragma once
#include <uv.h>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#include <string>
#include <vector>
#endif

namespace mom::runtime {
// libuv updates Windows' OS block, but SYCL/compiler code reads the CRT's separate getenv
// table. Update both before worker initialization; preserve libuv's Unicode and empty-value
// semantics in the OS block used by JavaScript and child processes. Linux already shares it.
inline int setEnvironment(const char* name, const char* value) {
#ifdef _WIN32
  // Reject invalid names before the CRT's invalid-parameter handler can terminate the process.
  if (!*name || std::strchr(name, '=')) return UV_EINVAL;
  const auto nameSize = uv_wtf8_length_as_utf16(name);
  const auto valueSize = uv_wtf8_length_as_utf16(value ? value : "");
  if (nameSize < 1 || valueSize < 1) return UV_EINVAL;
  const auto wide = [](const char* text, size_t size) {
    std::vector<uint16_t> units(size);
    uv_wtf8_to_utf16(text, units.data(), size);
    return std::wstring(units.begin(), units.end() - 1);
  };
  const auto wideName = wide(name, static_cast<size_t>(nameSize));
  const auto wideValue = wide(value ? value : "", static_cast<size_t>(valueSize));
  if (_wputenv_s(wideName.c_str(), wideValue.c_str())) return UV_EIO;
#endif
  // The CRT treats an empty value as deletion. Reapply libuv even after CRT success so
  // process.env.NAME = "" remains an existing empty OS variable, distinct from deletion.
  return value ? uv_os_setenv(name, value) : uv_os_unsetenv(name);
}
} // namespace mom::runtime
