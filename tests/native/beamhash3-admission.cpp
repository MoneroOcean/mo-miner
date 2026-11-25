#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include "../../sycl/workgroup-limits.h"

namespace sycl {
namespace info::device {
struct local_mem_size {};
struct max_work_group_size {};
template <int Dimensions> struct max_work_item_sizes {};
} // namespace info::device
struct device {
  std::size_t local_memory = 64 * 1024;
  bool cuda = false;
  template <class Info> auto get_info() const {
    if constexpr (std::is_same_v<Info, info::device::local_mem_size>)
      return local_memory;
    else if constexpr (std::is_same_v<Info, info::device::max_work_group_size>)
      return std::size_t{1024};
    else
      return std::array<std::size_t, 1>{1024};
  }
};
} // namespace sycl

static bool mom_is_cuda(const sycl::device& device) { return device.cuda; }
static bool mom_is_hip(const sycl::device&) { return false; }
static bool sycl_is_level_zero_gpu(const sycl::device&) { return false; }

// Production launch helpers from sycl/lib-internal.h.
#include "beam_workgroup_helpers.inc"

// Fixture geometry: the admission contract, not today's layout byte counts, is under test.
constexpr std::size_t GENERIC_COLLISION_LOCAL_BYTES = 20000;
constexpr std::size_t GENERIC_SCATTER_LOCAL_BYTES_PER_ITEM = 200;
constexpr std::size_t COMPACT_COLLISION_LOCAL_BYTES = 24000;
constexpr std::size_t COMPACT_SEED_LOCAL_BYTES_PER_ITEM = 100;

// Production beam_compact_enabled and beamhash3_workgroup from sycl/beamhash3/layout.inc.
#include "beamhash3_workgroup.inc"

// Discovery calls beamhash3_workgroup without a try block, so a short device must yield 0.
static unsigned workgroup(const std::size_t local_memory, const bool cuda) {
  sycl::device device;
  device.local_memory = local_memory;
  device.cuda = cuda;
  try {
    return beamhash3_workgroup(device);
  } catch (const std::string& error) {
    std::fprintf(stderr, "BeamHash3 admission threw: %s\n", error.c_str());
    std::abort();
  }
}

int main() {
  assert(workgroup(GENERIC_COLLISION_LOCAL_BYTES - 1, false) == 0);
  assert(workgroup(128 * GENERIC_SCATTER_LOCAL_BYTES_PER_ITEM - 1, false) == 0);
  assert(workgroup(128 * GENERIC_SCATTER_LOCAL_BYTES_PER_ITEM, false) == 128);
  assert(workgroup(640 * GENERIC_SCATTER_LOCAL_BYTES_PER_ITEM, false) == 640);
  assert(workgroup(COMPACT_COLLISION_LOCAL_BYTES - 1, true) == 0);
  assert(workgroup(COMPACT_COLLISION_LOCAL_BYTES, true) == 128);
  assert(workgroup(256 * COMPACT_SEED_LOCAL_BYTES_PER_ITEM, true) == 256);
  std::printf("BeamHash3 admission: short devices are skipped without throwing\n");
}
