// Rejected application libraries must preserve the POSIX loader error contract, including
// one-shot consumption and thread isolation. No driver or other library is loaded here.
#include <dlfcn.h>
#include <cstring>
#include <thread>

int main() {
  (void)dlerror();
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (dlopen("mom-forbidden-application.so", RTLD_NOW)) return 85;
    const char* error = dlerror();
    if (!error || !std::strstr(error, "application library") || dlerror()) return 86;
  }
  if (dlopen("mom-forbidden-application.so", RTLD_NOW)) return 87;
  bool isolated = false;
  std::thread other([&] {isolated = dlerror() == nullptr;});
  other.join();
  if (!isolated || !dlerror() || dlerror()) return 88;
  void* self = dlopen(nullptr, RTLD_NOW);
  if (!self || dlerror()) return 89;
  return dlclose(self);
}
