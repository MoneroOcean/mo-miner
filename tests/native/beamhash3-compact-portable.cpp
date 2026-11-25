// Drive the extracted production beam_compact_enabled() across devices and overrides. The portable
// worker does not compile BeamHash3's compact kernels, so it must never select compact arenas.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace sycl {
struct device {
  bool cuda, hip, level_zero;
};
} // namespace sycl

inline bool mom_is_cuda(const sycl::device& device) { return device.cuda; }
inline bool mom_is_hip(const sycl::device& device) { return device.hip; }
inline bool sycl_is_level_zero_gpu(const sycl::device& device) { return device.level_zero; }

#include "mom_env_flag.inc"

#include "beam_compact_enabled.inc"

int main() {
  const sycl::device devices[] = {
    {false, false, false}, {true, false, false}, {false, true, false}, {false, false, true}};
  const char* const settings[] = {nullptr, "1", "on", "0"};
  unsigned enabled = 0;
  for (const char* setting : settings) {
    if (setting)
      setenv("MOM_BEAMHASH3_COMPACT", setting, 1);
    else
      unsetenv("MOM_BEAMHASH3_COMPACT");
    for (const auto& device : devices)
      enabled += beam_compact_enabled(device);
  }
  std::printf("beam compact enabled=%u\n", enabled);
}
