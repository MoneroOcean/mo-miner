// Legacy SDK constructors must run before main, just like modern language initializers.
#include <cstdio>
static int legacy = 0;
static void initialize_legacy() { legacy = 73; }
using Initializer = void (*)();
[[gnu::used, gnu::section(".ctors")]] static Initializer initialize = initialize_legacy;
int main() {
  if (legacy != 73) {std::fputs("Legacy static initializer was not called\n", stderr); return 1;}
  std::puts("Legacy static initializer passed");
}
