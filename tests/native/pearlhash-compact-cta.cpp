// The NVIDIA compact profile must emit shapes the compact kernel admits (whole kCtaM x kCtaN CTAs)
// and whole waves of CTAs across the device's compute units. The CTA geometry comes from the
// kernel via -DCOMPACT_CTA_M/-DCOMPACT_CTA_N; prints how many unit counts received a profile.
#include "../../sycl/pow-intensity.h"

#include <cstdio>

int main() {
  unsigned checked = 0;
  for (unsigned units = 1; units <= 256; ++units) {
    const PowDeviceProfile device = {units, 64ULL << 30, 64ULL << 30};
    const pearlhash_profile_t profile = pearlhash_nvidia_compact_profile_for(device, true);
    // Unit counts whose wave quantum exceeds the 32-bit index bound fail closed to no profile.
    if (!profile.m && !profile.n)
      continue;
    const uint64_t ctas = uint64_t(profile.m / COMPACT_CTA_M) * (profile.n / COMPACT_CTA_N);
    if (profile.m % COMPACT_CTA_M || profile.n % COMPACT_CTA_N || ctas % units) {
      std::fprintf(stderr, "units=%u m=%u n=%u is not a whole CTA wave\n", units, profile.m, profile.n);
      return 1;
    }
    ++checked;
  }
  const pearlhash_profile_t absent = pearlhash_nvidia_compact_profile_for({36, 64ULL << 30, 64ULL << 30}, false);
  if (absent.m || absent.n) {
    std::fprintf(stderr, "compact profile without the native feature\n");
    return 1;
  }
  std::printf("compact CTA waves checked=%u\n", checked);
  return 0;
}
