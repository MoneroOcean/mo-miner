// No actual dynamic loader or GPU is involved: verify the linked native boundary's forwarding.
#include <cstring>

static unsigned calls = 0;
static const char* forwarded = nullptr;
static int forwarded_flags = 0;
static void* recordLoad(const char* name, int flags) {
  ++calls; forwarded = name; forwarded_flags = flags; return &calls;
}
extern "C" void* __real_dlopen(const char* name, int flags) {return recordLoad(name, flags);}
extern "C" char* __real_dlerror() {return nullptr;}
extern "C" void* __real_dlsym(void*, const char*) {return nullptr;}
extern "C" void* __real_dlvsym(void*, const char*, const char*) {return nullptr;}
extern "C" void* __wrap_dlopen(const char* name, int flags);
#ifdef MOM_TEST_KHRONOS_ICD
// --wrap also rewrites __real_dlopen references back to dlopen. Intercept that final edge;
// defining only __real_dlopen would accidentally invoke libc's actual loader in this fixture.
extern "C" void* dlopen(const char* name, int flags) {return recordLoad(name, flags);}
extern "C" char* dlerror() {return nullptr;}
extern "C" void* khrIcdOsLibraryLoad(const char* name);
#endif

int main(int argc, char** argv) {
  if (argc != 3) return 86;
  const char* name = std::strcmp(argv[1], "<self>") == 0 ? nullptr : argv[1];
  const bool allowed = std::strcmp(argv[2], "allowed") == 0;
#ifdef MOM_TEST_KHRONOS_ICD
  // Exercise the original linked ICD's loader path, but never call the actual OS loader.
  void* result = khrIcdOsLibraryLoad(name);
#else
  void* result = __wrap_dlopen(name, 0x1234);
#endif
  // These checks must survive Release/NDEBUG: the canonical static host uses that profile.
  if (calls != (allowed ? 1U : 0U) || result != (allowed ? &calls : nullptr)) return 86;
  if (allowed) {
    if (forwarded != name) return 86;
#ifndef MOM_TEST_KHRONOS_ICD
    if (forwarded_flags != 0x1234) return 86;
#endif
  }
}
