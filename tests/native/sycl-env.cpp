// set_sycl_env runs on the compute thread while other threads read the environment. It must not
// call setenv again for an unchanged value; each glibc setenv allocates and swaps an entry.
#include <cassert>
#include <cstdlib>
#include <cstring>

static int setenv_calls = 0;
extern "C" int __real_setenv(const char* name, const char* value, int overwrite);
extern "C" int __wrap_setenv(const char* name, const char* value, int overwrite) {
  ++setenv_calls;
  return __real_setenv(name, value, overwrite);
}

#include "sycl_env.inc"

int main() {
  constexpr const char* name = "SYCL_PROGRAM_COMPILE_OPTIONS";
  unsetenv(name);
  set_sycl_env(name, "");
  assert(setenv_calls == 1 && std::strcmp(std::getenv(name), "") == 0);
  set_sycl_env(name, "");
  assert(setenv_calls == 1);
  set_sycl_env(name, "-O2");
  assert(setenv_calls == 2 && std::strcmp(std::getenv(name), "-O2") == 0);
  set_sycl_env(name, "-O2");
  assert(setenv_calls == 2);
  return 0;
}
