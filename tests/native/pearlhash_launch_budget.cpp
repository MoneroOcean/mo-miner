// Copyright GNU GPLv3 (c) 2026 MoneroOcean
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include "launch-budget.inc"

int main() {
  for (const int k : {2048, 4096, 4160, 8192, 65536}) {
    const size_t groups = pearlhash_portable_launch_groups(k, true);
    if (!groups || uint64_t(groups) * 256 * k > (uint64_t{1} << 32) ||
        pearlhash_portable_launch_groups(k, false) != (size_t{1} << 22))
      return 1;
    for (const size_t count : {size_t{1}, groups - 1, groups, groups + 1, size_t{1} << 26}) {
      size_t covered = 0;
      for (size_t base = 0; base < count; base += groups)
        covered += std::min(groups, count - base);
      if (covered != count)
        return 1;
    }
  }
  for (const int k : {0, -1}) {
    try {
      (void)pearlhash_portable_launch_groups(k, true);
      return 1;
    } catch (const std::string&) {}
  }
  std::puts("PearlHash portable launch work is bounded with complete tile coverage");
}
