// Compile the actual constructor/binding with instrumented ABI callbacks, without a compiler SDK.
#include <array>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

static unsigned opens, lookups, unloads, calls;
namespace sycl::detail::ur {
void* loadOsLibrary(const std::string&) {++opens; return nullptr;}
void* getOsLibraryFuncAddress(void*, const char*) {++lookups; return nullptr;}
int unloadOsLibrary(void*) {++unloads; return 0;}
}
[[maybe_unused]] static void printPerformanceWarning(const std::string&) {}
namespace jit_compiler {
#define ENTRY(name) int name(int value) {++calls; return value + 1;}
ENTRY(addToJITConfiguration) ENTRY(resetJITConfiguration) ENTRY(materializeSpecConstants)
ENTRY(calculateHash) ENTRY(compileSYCL) ENTRY(destroyBinary)
#undef ENTRY
}
class Binding {
 public:
  using AddToConfigFuncT = int(*)(int); using ResetConfigFuncT = int(*)(int);
  using MaterializeSpecConstFuncT = int(*)(int); using CalculateHashFuncT = int(*)(int);
  using CompileSYCLFuncT = int(*)(int); using DestroyBinaryFuncT = int(*)(int);
  AddToConfigFuncT AddToConfigHandle = nullptr; ResetConfigFuncT ResetConfigHandle = nullptr;
  MaterializeSpecConstFuncT MaterializeSpecConstHandle = nullptr; CalculateHashFuncT CalculateHashHandle = nullptr;
  CompileSYCLFuncT CompileSYCLHandle = nullptr; DestroyBinaryFuncT DestroyBinaryHandle = nullptr;
  bool Available = false;
  std::function<void(void*)> CustomDeleterForLibHandle = [](void* ptr) {if (ptr) sycl::detail::ur::unloadOsLibrary(ptr);};
  std::unique_ptr<void, decltype(CustomDeleterForLibHandle)> LibraryHandle{nullptr, CustomDeleterForLibHandle};
  Binding() {
#include "bindings.inc"
  }
};
int main() try {
  for (int iteration = 0; iteration < 3; ++iteration) {
    Binding binding;
    if (!binding.Available || binding.LibraryHandle || opens || lookups || unloads)
      throw std::runtime_error("Static JIT binding attempted an application loader or was unavailable");
    const std::array<std::pair<int(*)(int), int(*)(int)>, 6> entries{{
      {binding.AddToConfigHandle, jit_compiler::addToJITConfiguration},
      {binding.ResetConfigHandle, jit_compiler::resetJITConfiguration},
      {binding.MaterializeSpecConstHandle, jit_compiler::materializeSpecConstants},
      {binding.CalculateHashHandle, jit_compiler::calculateHash},
      {binding.CompileSYCLHandle, jit_compiler::compileSYCL},
      {binding.DestroyBinaryHandle, jit_compiler::destroyBinary}}};
    for (const auto& entry : entries) {
      if (entry.first != entry.second || entry.first(17) != 18) throw std::runtime_error("Incorrect static JIT entrypoint");
    }
  }
  if (calls != 18 || unloads) throw std::runtime_error("Static JIT entrypoint/lifetime mismatch");
  std::puts("Static JIT binding: six exact callbacks, repeated lifecycle, no loader ownership");
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what()); return 1;
}
