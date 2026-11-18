// Model ROCr's exact static-libdrm self lookup without loading a driver or using any GPU.
#include <cassert>
#include <dlfcn.h>

extern "C" int amdgpu_device_get_fd(void*) { return 37; }
int main(int argc, char**) {
  assert(argc == 2);
  auto* entry = reinterpret_cast<int (*)(void*)>(dlsym(RTLD_DEFAULT, "amdgpu_device_get_fd"));
  if (!entry) return 3; // The old link policy loses this function despite its static definition.
  assert(entry(nullptr) == 37);
}
