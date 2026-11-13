// Native compiler SDKs share LLVM's option/target registries and LLD's process-global context.
// Serialize whole compilation entries, not just one SDK's linker call. Recursion permits the
// shared SYCL full/partial/prepare/translate entries to call one another without deadlocking.
#include <mutex>

static std::recursive_mutex& compiler_gate() {
  static std::recursive_mutex gate;
  return gate;
}

extern "C" void mom_native_compiler_lock() { compiler_gate().lock(); }
extern "C" void mom_native_compiler_unlock() { compiler_gate().unlock(); }
