// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "zelhash_generator_guard.inc"

namespace sycl {
struct exception : std::runtime_error {
  using std::runtime_error::runtime_error;
};
namespace info::device {
struct vendor {};
} // namespace info::device
#if !defined(MOM_SYCL_ADAPTIVECPP) && !defined(MOM_SYCL_PORTABLE_OPENCL)
enum class aspect { ext_intel_gpu_eu_simd_width };
namespace ext::intel::info::device {
struct gpu_eu_simd_width {};
} // namespace ext::intel::info::device
#endif
struct device {
  bool gpu = true, advertised = true;
  std::string vendor_name = "Intel";
  unsigned width = 8, failure = 0, driver_build = 37020;
  bool subgroup_supported = true, level_zero = true;
  mutable unsigned queries = 0, backend_queries = 0, driver_queries = 0;
  bool is_gpu() const {
    ++queries;
    return gpu;
  }
  template <typename Info> auto get_info() const {
    ++queries;
    if constexpr (std::is_same_v<Info, info::device::vendor>) {
      if (failure == 1) throw exception("vendor unavailable");
      return vendor_name;
#if !defined(MOM_SYCL_ADAPTIVECPP) && !defined(MOM_SYCL_PORTABLE_OPENCL)
    } else {
      static_assert(std::is_same_v<Info, ext::intel::info::device::gpu_eu_simd_width>);
      if (failure == 3) throw exception("width unavailable");
      if (failure == 4) throw std::logic_error("unrelated error");
      return width;
#endif
    }
  }
#if !defined(MOM_SYCL_ADAPTIVECPP) && !defined(MOM_SYCL_PORTABLE_OPENCL)
  bool has(aspect) const {
    ++queries;
    if (failure == 2) throw exception("aspect unavailable");
    return advertised;
  }
#endif
};
struct event {
  bool large_grf, compact, forwarded;
};
struct queue {
  device d;
  const device &get_device() const { return d; }
};
} // namespace sycl

// Generated from production, including its actual build guards; excluded profiles have no tags.
#include "intel_geometry_helper.inc"

constexpr bool native_profile =
#if !defined(MOM_SYCL_ADAPTIVECPP) && !defined(MOM_SYCL_PORTABLE_OPENCL)
    true;
#else
    false;
#endif
[[maybe_unused]] static bool sycl_is_level_zero_gpu(const sycl::device &device) {
  ++device.backend_queries;
  return device.level_zero;
}
// The production wrapper remains intact; this boundary records which image it submits.
static unsigned zelhash_calls = 0;
static uint64_t zelhash_h[8];
static uint8_t zelhash_pending[12];
static uint32_t zelhash_level[1], zelhash_index[1], zelhash_slots[1];
template <bool GPU_COMPACT, bool LARGE_GRF = false>
static sycl::event submit_gen_fill_impl(sycl::queue &, const uint64_t *h, const uint8_t *pending,
                                        uint32_t *level, uint32_t *index, uint32_t *slots,
                                        unsigned capacity) {
  ++zelhash_calls;
  return {LARGE_GRF, GPU_COMPACT,
          h == zelhash_h && pending == zelhash_pending && level == zelhash_level &&
              index == zelhash_index && slots == zelhash_slots && capacity == 123};
}
#include "zelhash_generator_dispatch.inc"

unsigned assertions = 0;
void require(bool condition, const char *label) {
  ++assertions;
  if (condition) return;
  std::fprintf(stderr, "Intel geometry regression: %s\n", label);
  std::abort();
}

void test_zelhash_generator_policy() {
  const auto check = [](const sycl::device &device, bool eligible, bool compact = false) {
    sycl::queue q{device};
    const unsigned before = zelhash_calls;
    const auto event = compact
                           ? submit_gen_fill<true>(q, zelhash_h, zelhash_pending, zelhash_level,
                                                   zelhash_index, zelhash_slots, 123)
                           : submit_gen_fill<false>(q, zelhash_h, zelhash_pending, zelhash_level,
                                                    zelhash_index, zelhash_slots, 123);
#if defined(MOM_ZELHASH_INTEL_LARGE_GRF)
    require(event.large_grf == eligible, "Zel GenFill exact-width native choice");
#else
    (void)eligible;
    require(!event.large_grf, "Zel GenFill excluded profile retains default");
#endif
    require(zelhash_calls == before + 1, "Zel GenFill submits one image");
    require(event.compact == compact && event.forwarded, "Zel GenFill arguments unchanged");
  };
  sycl::device device;
  for (unsigned width : {8u, 16u, 0u}) {
    device = {};
    device.width = width;
    check(device, width == 8);
  }
  device = {};
  device.vendor_name = "Other";
  check(device, false);
  device = {};
  device.level_zero = false;
  check(device, false);
  device = {};
  device.gpu = false;
  check(device, false);
  device = {};
  device.advertised = false;
  check(device, false);
  for (unsigned failure : {1u, 2u, 3u}) {
    device = {};
    device.failure = failure;
    check(device, false);
  }
  device = {};
  check(device, true, true);
  device = {};
  device.failure = 4;
  bool unrelated_threw = false;
  try {
    check(device, false);
  } catch (const std::logic_error &) {
    unrelated_threw = true;
  }
  require(unrelated_threw ==
#if defined(MOM_ZELHASH_INTEL_LARGE_GRF)
              true,
#else
              false,
#endif
          "Zel GenFill unrelated query error not swallowed");
}

int main() {
  test_zelhash_generator_policy();
  sycl::device device;
  for (unsigned width : {0u, 4u, 8u, 16u, 32u}) {
    device.width = width;
    device.queries = 0;
    require(mom_intel_eu_simd_width(device) == (native_profile ? width : 0), "exact width");
    require(device.queries == (native_profile ? 4u : 0u), "profile query count");
  }
  device = {};
  for (unsigned missing = 1; missing <= 3; ++missing) {
    device.gpu = missing != 1;
    device.vendor_name = missing == 2 ? "Other" : "Intel";
    device.advertised = missing != 3;
    device.queries = 0;
    require(mom_intel_eu_simd_width(device) == 0, "ineligible geometry unknown");
    require(device.queries == (native_profile ? missing : 0u), "query short circuit");
  }
  device = {};
  for (unsigned failure : {1u, 2u, 3u}) {
    device.failure = failure;
    require(mom_intel_eu_simd_width(device) == 0, "optional query failure unknown");
  }
  device.failure = 4;
  bool unrelated_threw = false;
  try {
    (void)mom_intel_eu_simd_width(device);
  } catch (const std::logic_error &) {
    unrelated_threw = true;
  }
  require(unrelated_threw == native_profile, "unrelated error not swallowed");
  std::printf("Intel geometry CPU profile passed: %u assertions\n", assertions);
}
