// Device discovery must admit Equihash(192,7) and ZHash devices on exactly the local memory their
// sessions require, under every layout define (portable/CUDA, HIP, Intel).
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace sycl {
namespace info::device {
struct local_mem_size {};
} // namespace info::device
struct device {
  std::size_t local_memory = 0;
  template <class Info> std::size_t get_info() const {
    return local_memory;
  }
};
} // namespace sycl

// Production sycl_local_memory_fits from sycl/lib-internal.h.
#include "local_memory_fits.inc"
#include "../../sycl/zhash/session-layout.h"

namespace mom_equihash192_7 {
namespace direct {
using Spec = mom_equihash::Equihash192_7;
// Production BucketLayout and ProductionSlotsPerBucket from sycl/equihash192-7/layout.h.
#include "equihash192_layout.inc"
} // namespace direct
// Session<>::Layout is ProductionArena's level-0 layout, ArenaLayout<12, ProductionSlotsPerBucket>.
struct Solver {
  using Layout = direct::BucketLayout<12, direct::ProductionSlotsPerBucket>;
};
// Production equihash192_7_intensity from equihash192-7.cpp.
#include "equihash192_admission.inc"
} // namespace mom_equihash192_7

namespace mom_zhash {
// Production zhash_intensity from zhash.cpp.
#include "zhash_admission.inc"
} // namespace mom_zhash

static sycl::device with_local_memory(const std::size_t bytes) {
  sycl::device device;
  device.local_memory = bytes;
  return device;
}

int main() {
  using Layout = mom_equihash192_7::Solver::Layout;
  constexpr std::size_t equihash = Layout::local_memory_bytes;
  static_assert(equihash == Layout::local_bins * sizeof(std::uint32_t) +
                               Layout::bucket_slot_capacity * sizeof(std::uint16_t));
  assert(mom_equihash192_7::equihash192_7_intensity(with_local_memory(equihash)) == 1);
  assert(mom_equihash192_7::equihash192_7_intensity(with_local_memory(equihash - 1)) == 0);

  // The fast session and its safe-layout retry must both fit.
  constexpr std::size_t zhash = std::max(mom_zhash::FastZHashArenaLayout::local_memory_bytes,
                                         mom_zhash::DefaultZHashArenaLayout::local_memory_bytes);
  assert(mom_zhash::zhash_intensity(with_local_memory(zhash)) == 1);
  assert(mom_zhash::zhash_intensity(with_local_memory(zhash - 1)) == 0);
  std::printf("Equihash admission: 192,7 needs %zu B, ZHash needs %zu B\n", equihash, zhash);
}
