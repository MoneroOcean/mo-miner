// Two compiler versions intentionally instantiate the same constructor with different layouts
// or semantics. Isolation must preserve each COMDAT, not accidentally share one implementation.
#include <cstdio>
namespace llvm {
template<unsigned N> struct State { State(); unsigned value; };
template<unsigned N> State<N>::State() : value(MOM_COMPILER_VALUE) {}
template struct State<0>;
#ifdef MOM_COMPILER_INCOMPLETE
int missing_compiler_entry();
#endif
}
#ifdef MOM_COMPILER_HOST
extern "C" int isolated_compiler();
int main() {
  llvm::State<0> host;
  if (host.value != MOM_COMPILER_VALUE || isolated_compiler() != 17) return 1;
  std::puts("Independent compiler constructors and original C bridge passed");
}
#else
extern "C" int isolated_compiler() {
#ifdef MOM_COMPILER_INCOMPLETE
  return llvm::missing_compiler_entry();
#else
  return llvm::State<0>{}.value;
#endif
}
#endif
