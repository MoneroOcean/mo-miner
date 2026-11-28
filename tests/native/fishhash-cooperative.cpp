#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include "../../sycl/workgroup-limits.h"

constexpr unsigned FISH_TEAM = 16, FISH_STAGE_BYTES = 96, HASH_LEN = 32;
constexpr bool mom_sycl_portable_opencl = false;
constexpr bool fishhash_large_grf = true;
struct Device {
  unsigned local_bytes = 8192;
  bool subgroup16 = true;
  bool gpu = true;
  bool subgroup8 = true;
  bool bmg_g21 = false;
  bool is_gpu() const { return gpu; }
};
namespace sycl { using device = Device; }
struct Result { unsigned count; uint8_t hash[1][HASH_LEN]; };
struct State { Device device; Result* result; unsigned fish_workgroup; unsigned karlsen_workgroup = 0; };
bool mom_is_cuda(Device) { return false; }
bool mom_is_hip(Device) { return false; }
bool mom_is_bmg_g21(Device dev) { return dev.bmg_g21; }
bool sycl_is_level_zero_gpu(Device) { return true; }
bool sycl_supports_subgroup_size(Device device, unsigned width) { return width == 8 && device.subgroup8; }
bool sycl_reqd_subgroup_16_supported(Device device) { return device.subgroup16; }
bool sycl_workgroup_fits(Device device, unsigned size, size_t fixed_bytes,
                        size_t item_bytes, size_t team_bytes, unsigned team_size) {
  return mom_workgroup_fits(size, 512, 512, device.local_bytes, fixed_bytes,
                           item_bytes, team_bytes, team_size);
}
void sycl_wait_and_throw(int, Device) {}

#include "selection.inc"

int main() {
  unsetenv("MOM_FISHHASH_WORKGROUP");
  assert(cooperative_workgroup(Device{}) == 128);
  Device bmg;
  bmg.bmg_g21 = true;
  assert(cooperative_workgroup(bmg, FISH_TEAM, true) == 64);
  // KarlsenHashV2 follows FishHash to 64 threads where the large register file is compiled in.
  assert(cooperative_workgroup(bmg) == 64);
  // Four sixteen-lane teams, or eight eight-lane teams, of FISH_STAGE_BYTES each must fit.
  assert(cooperative_workgroup(Device{384}) == 64);
  assert(cooperative_workgroup(Device{383}) == 0);
  assert(cooperative_workgroup(Device{767}, 8) == 0);
  assert(cooperative_workgroup(Device{768}, 8) == 64);
  assert(cooperative_workgroup(Device{8192, false}) == 0);
  assert(cooperative_workgroup(Device{8192, false}, 8) == 128);
  assert(cooperative_workgroup(Device{8192, true, true, false}, 8) == 0);
  assert(cooperative_workgroup(Device{8192, true, false}) == 0);
  setenv("MOM_FISHHASH_WORKGROUP", "512", 1);
  assert(cooperative_workgroup(Device{}) == 512);
  assert(cooperative_workgroup(Device{768}) == 128);
  unsetenv("MOM_FISHHASH_WORKGROUP");
  setenv("MOM_FISHHASH_COOP", "1", 1);
  State unsupported{{}, nullptr, 0};
  int unavailableChoice = -1;
  assert(!selectCooperative(unsupported, unavailableChoice, "fixture", true,
                            [](bool, unsigned, bool) { assert(false); return 0; }));
  // An explicit override skips the probe in both directions; "false" used to force it on.
  const auto unprobed = [](bool, unsigned, bool) { assert(false); return 0; };
  State forced{{}, nullptr, 128};
  int forcedChoice = -1;
  assert(selectCooperative(forced, forcedChoice, "fixture", true, unprobed));
  assert(!selectCooperative(forced, forcedChoice, "fixture", false, unprobed));
  setenv("MOM_FISHHASH_COOP", "false", 1);
  assert(!selectCooperative(forced, forcedChoice, "fixture", true, unprobed));
  unsetenv("MOM_FISHHASH_COOP");
  for (unsigned mode = 0; mode < 3; ++mode) {
    Result result{};
    State state{{}, &result, 128};
    int choice = -1;
    unsigned probes = 0, timed = 0;
    const auto search = [&](bool cooperative, unsigned count, bool test) {
      if (test) {
        assert(count == 1);
        ++probes;
        if (!cooperative || mode != 0) {
          result.count = 1;
          result.hash[0][0] = cooperative && mode == 1 ? 43 : 42;
        }
      } else {
        ++timed;
      }
      return 0;
    };
    const bool selected = selectCooperative(state, choice, "fixture", true, search);
    assert(probes == (mode == 0 ? 1 : 2));
    assert(timed == (mode == 2 ? 6 : 0));
    if (mode != 2)
      assert(!selected && choice == 0);
    assert(selectCooperative(state, choice, "fixture", true, search) == selected);
    assert(probes == (mode == 0 ? 1 : 2));
    assert(timed == (mode == 2 ? 6 : 0));
  }
  std::puts("PASS FishHash cooperative selection");
}
