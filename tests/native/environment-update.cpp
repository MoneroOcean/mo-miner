// OS/CRT copies are intentionally disjoint, as in a statically linked Windows executable.
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <cstdio>
static std::map<std::string, std::string> osEnvironment, crtEnvironment;
static std::vector<std::string> calls;
static bool invalidEncoding = false, crtFailure = false, osFailure = false;
static std::string narrow(const wchar_t* text) {
  std::string result; while (*text) result.push_back(static_cast<char>(*text++)); return result;
}
int _wputenv_s(const wchar_t* name, const wchar_t* value) {
  calls.push_back("crt");
  if (crtFailure) return 12;
  if (*value) crtEnvironment[narrow(name)] = narrow(value); else crtEnvironment.erase(narrow(name));
  return 0;
}
int uv_os_setenv(const char* name, const char* value) {
  calls.push_back("set");
  if (!*name || std::strchr(name, '=')) return -22;
  if (osFailure) return -5;
  osEnvironment[name] = value; return 0;
}
int uv_os_unsetenv(const char* name) {
  calls.push_back("unset");
  if (!*name || std::strchr(name, '=')) return -22;
  if (osFailure) return -5;
  osEnvironment.erase(name); return 0;
}
long uv_wtf8_length_as_utf16(const char* text) {return invalidEncoding ? -1 : static_cast<long>(std::strlen(text) + 1);}
void uv_wtf8_to_utf16(const char* text, uint16_t* out, size_t length) {
  if (length != std::strlen(text) + 1) std::abort();
  for (size_t i = 0; i < length; ++i) out[i] = static_cast<unsigned char>(text[i]);
}
#include "runtime/environment.h"
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const int mode = std::atoi(argv[1]);
  const char* name = "MOM_SYCL_COMPILER", *value = "acpp-hip";
  if (mode == 1) value = "";
  if (mode == 2) {value = nullptr; osEnvironment[name] = "old"; crtEnvironment[name] = "old";}
  if (mode == 3) name = "";
  if (mode == 4) name = "BAD=NAME";
  if (mode == 5) invalidEncoding = true;
  if (mode == 6) crtFailure = true;
  if (mode == 7) osFailure = true;
  if (mode == 8) {osEnvironment[name] = "old"; crtEnvironment[name] = "old";}
  if (mode == 9) {name = "TRUNCATED\0IGNORED"; value = "kept\0ignored";}
#ifdef MOM_ENV_OS_ONLY_BASELINE
  const int status = value ? uv_os_setenv(name, value) : uv_os_unsetenv(name);
#else
  const int status = mom::runtime::setEnvironment(name, value);
#endif
#ifdef _WIN32
  if (mode >= 3 && mode <= 6) return status != (mode == 6 ? -5 : -22) || osEnvironment.size() || crtEnvironment.size() || (mode == 6 ? calls != std::vector<std::string>{"crt"} : !calls.empty());
  if (mode == 7) return status != -5 || osEnvironment.size() || crtEnvironment[name] != "acpp-hip";
  if (status || calls != std::vector<std::string>{"crt", value ? "set" : "unset"}) return 3;
  if (value && *value ? crtEnvironment[name] != value : crtEnvironment.count(name) != 0) return 4;
#else
  if (mode == 3 || mode == 4) return status != -22;
  if (mode == 7) return status != -5;
  if (status || calls != std::vector<std::string>{value ? "set" : "unset"}) return 5;
#endif
  if (value ? osEnvironment[name] != value : osEnvironment.count(name) != 0) return 6;
  return 0;
}
