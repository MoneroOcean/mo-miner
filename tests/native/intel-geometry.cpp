// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "zelhash_generator_guard.inc"

namespace sycl {
struct exception : std::runtime_error { using std::runtime_error::runtime_error; };
namespace info::device { struct vendor {}; }
#if !defined(MOM_SYCL_ADAPTIVECPP) && !defined(MOM_SYCL_PORTABLE_OPENCL)
enum class aspect { ext_intel_gpu_eu_simd_width };
namespace ext::intel::info::device { struct gpu_eu_simd_width {}; }
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
constexpr bool portable_profile =
#if defined(MOM_SYCL_PORTABLE_OPENCL)
    true;
#else
    false;
#endif

// Hardware predicates are fixture inputs; the decision body and optional query are production code.
static bool kawpow_subgroup_exchange_supported(const sycl::device& device) {
  return device.gpu && !portable_profile && device.subgroup_supported;
}
static bool sycl_is_level_zero_gpu(const sycl::device& device) {
  ++device.backend_queries;
  return device.level_zero;
}
static uint32_t mom_driver_build(const sycl::device& device) {
  ++device.driver_queries;
  return device.driver_build;
}
#include "kawpow_exchange_policy.inc"

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

// Small boundary parameters test the real selection/fit statements, not GPU allocation sizes.
constexpr std::uint64_t kMaxEdgeArrayBytes = 100, kRequiredDeviceBytes = 200, kSafetyReserve = 50;
constexpr std::uint64_t kLowMemoryRequiredDeviceBytes = 100, kLowMemoryReserve = 25;

// Permit the pre-width8 policy to compile so the regression negative fails on behavior.
bool packed([[maybe_unused]] const sycl::device& device, std::uint64_t global_bytes = 250,
            std::uint64_t max_alloc = 100, const char* force = nullptr) {
  if (force ? setenv("MOM_C30_FORCE_LOW_MEMORY", force, 1) : unsetenv("MOM_C30_FORCE_LOW_MEMORY"))
    throw std::runtime_error("fixture ENV setup failed");
  bool low_memory = false;
#include "c30_layout_policy.inc"
  return low_memory;
}

unsigned assertions = 0;
void require(bool condition, const char* label) {
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

void test_kawpow_policy() {
  sycl::device device;
  for (unsigned width : {8u, 0u, 4u, 16u, 32u}) {
    device = {};
    device.width = width;
    require(kawpow_default_subgroup_exchange(device) ==
            (!portable_profile && (!native_profile || width != 8)), "KawPow exact-width default");
    require(device.queries == (native_profile ? 4u : 0u), "KawPow profile query count");
  }
  for (unsigned build : {0u, 29999u, 30000u, 34999u, 35000u, 37020u}) {
    device = {};
    device.width = 16;
    device.driver_build = build;
    const bool allowed = build < 30000u || build >= 35000u;
    require(kawpow_default_subgroup_exchange(device) == (!portable_profile && allowed),
            "KawPow old driver boundaries");
    require(device.queries == (native_profile && allowed ? 4u : 0u),
            "KawPow affected driver skips optional query");
  }
  device = {};
  device.level_zero = false;
  device.failure = 4;
  require(kawpow_default_subgroup_exchange(device) == !portable_profile,
          "KawPow non-Level-Zero default unchanged");
  require(device.queries == 0 && device.driver_queries == 0, "KawPow other backend short circuit");
  for (unsigned missing : {1u, 2u}) {
    device = {};
    device.gpu = missing != 1;
    device.subgroup_supported = missing != 2;
    device.failure = 4;
    require(!kawpow_default_subgroup_exchange(device), "KawPow unsupported remains local memory");
    require(device.queries == 0 && device.backend_queries == 0 && device.driver_queries == 0,
            "KawPow eligibility short circuit");
  }
  for (unsigned missing : {1u, 2u}) {
    device = {};
    device.vendor_name = missing == 1 ? "Other" : "Intel";
    device.advertised = missing != 2;
    require(kawpow_default_subgroup_exchange(device) == !portable_profile,
            "KawPow non-Intel or unadvertised retains default");
  }
  for (unsigned failure : {1u, 2u, 3u}) {
    device = {};
    device.failure = failure;
    require(kawpow_default_subgroup_exchange(device) == !portable_profile,
            "KawPow optional query failure retains default");
  }
  device = {};
  device.driver_build = 32224;
  device.failure = 4;
  require(!kawpow_default_subgroup_exchange(device), "KawPow workaround precedes geometry");
  require(device.queries == 0, "KawPow workaround does not query geometry");
  device = {};
  device.failure = 4;
  bool unrelated_threw = false;
  try {
    (void)kawpow_default_subgroup_exchange(device);
  } catch (const std::logic_error&) {
    unrelated_threw = true;
  }
  require(unrelated_threw == native_profile, "KawPow unrelated error not swallowed");
}

int main() {
  test_zelhash_generator_policy();
  test_kawpow_policy();
  sycl::device device;
  for (unsigned width : {0u, 4u, 8u, 16u, 32u}) {
    device.width = width;
    device.queries = 0;
    require(mom_intel_eu_simd_width(device) == (native_profile ? width : 0), "exact width");
    require(device.queries == (native_profile ? 4u : 0u), "profile query count");
    require(packed(device) == (native_profile && width == 8), "packed only at width8");
  }
  device = {};
  for (unsigned missing = 1; missing <= 3; ++missing) {
    device.gpu = missing != 1;
    device.vendor_name = missing == 2 ? "Other" : "Intel";
    device.advertised = missing != 3;
    device.queries = 0;
    require(mom_intel_eu_simd_width(device) == 0, "ineligible geometry unknown");
    require(device.queries == (native_profile ? missing : 0u), "query short circuit");
    require(!packed(device), "ineligible retains full");
  }
  device = {};
  for (unsigned failure : {1u, 2u, 3u}) {
    device.failure = failure;
    require(mom_intel_eu_simd_width(device) == 0, "optional query failure unknown");
    require(!packed(device), "optional failure retains full");
  }
  device.failure = 4;
  bool unrelated_threw = false;
  try {
    (void)mom_intel_eu_simd_width(device);
  } catch (const std::logic_error&) {
    unrelated_threw = true;
  }
  require(unrelated_threw == native_profile, "unrelated error not swallowed");
  for (const char* force : {"", "1", "false", " 0"}) {
    device.queries = 0;
    require(packed(device, 250, 100, force), "force semantics");
    require(device.queries == 0, "forced packed skips optional query");
  }
  device = {};
  device.width = 16;
  for (const char* force : {"0", "00", "0false"})
    require(!packed(device, 250, 100, force), "leading zero does not force");
  device.width = 8;
  require(packed(device, 250, 100, "0") == native_profile, "zero does not force full");
  device.failure = 4;
  device.queries = 0;
  require(packed(device, 250, 99), "allocation threshold minus one");
  require(packed(device, 249), "full global threshold minus one");
  require(packed(device, 125, 99), "exact packed fit admitted");
  bool fit_threw = false;
  try {
    (void)packed(device, 124, 99);
  } catch (const std::runtime_error& error) {
    fit_threw = std::string(error.what()).find("low-memory") != std::string::npos;
  }
  require(fit_threw, "packed fit threshold minus one rejected");
  require(device.queries == 0, "memory policy skips optional queries");
  std::printf("Intel geometry CPU profile passed: %u assertions\n", assertions);
}
