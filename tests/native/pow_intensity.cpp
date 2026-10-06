#include "../../sycl/pow-intensity.h"

#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr uint64_t kMiB = 1ULL << 20;
constexpr uint64_t kGiB = 1ULL << 30;

unsigned assertions = 0;

void expect_equal(const char* const label, const uint64_t actual, const uint64_t expected) {
  ++assertions;
  if (actual != expected) {
    std::fprintf(stderr, "%s: got %llu, expected %llu\n", label,
                 static_cast<unsigned long long>(actual),
                 static_cast<unsigned long long>(expected));
    std::abort();
  }
}

void expect_true(const char* const label, const bool condition) {
  ++assertions;
  if (!condition) {
    std::fprintf(stderr, "%s: condition failed\n", label);
    std::abort();
  }
}

void expect_scale(const char* const label, const PowIntensityScale& actual,
                  const PowIntensityScale& expected) {
  expect_equal(label, actual.fallback_workgroup, expected.fallback_workgroup);
  expect_equal(label, actual.base_work_items, expected.base_work_items);
  expect_equal(label, actual.compute_unit_divisor, expected.compute_unit_divisor);
}

void expect_pearl_profile(const char* const label, const pearlhash_profile_t& actual,
                          const pearlhash_profile_t& expected) {
  expect_equal(label, actual.m, expected.m);
  expect_equal(label, actual.n, expected.n);
  expect_equal(label, actual.k, expected.k);
  expect_equal(label, actual.rank, expected.rank);
}

void test_pearlhash_capacity() {
  const pearlhash_profile_t wide = {131072, 524288, 8192, 128};
  const pearlhash_profile_t compact = {65536, 65536, 4096, 256};
  const pearlhash_profile_t ordinary = {131072, 131072, 2048, 128};
  constexpr uint64_t reserve = 512 * kMiB;
  PearlHashAllocationCounts wide_counts{}, compact_counts{}, ordinary_counts{}, hip_counts{};
  expect_true("Pearl wide legal", pearlhash_allocation_counts(wide, wide_counts));
  expect_equal("Pearl wide Bp spans full uint32 byte domain", wide_counts.n_k, 4 * kGiB);
  expect_equal("Pearl wide Ap", wide_counts.m_k, kGiB);
  expect_equal("Pearl wide EAL", wide_counts.m_rank, 16 * kMiB);
  expect_equal("Pearl wide EBR", wide_counts.n_rank, 64 * kMiB);
  expect_equal("Pearl wide CVA", wide_counts.cva_bytes, 64 * kMiB + 64);
  expect_equal("Pearl wide CVB", wide_counts.cvb_bytes, 256 * kMiB + 64);
  expect_equal("Pearl wide required maximum", wide_counts.max_alloc, 4 * kGiB);
  expect_equal("Pearl wide complete required budget", wide_counts.total_bytes, 5855379776ULL);
  expect_true("Pearl compact legal", pearlhash_allocation_counts(compact, compact_counts));
  expect_equal("Pearl compact required maximum", compact_counts.max_alloc, 256 * kMiB);
  expect_equal("Pearl compact complete required budget", compact_counts.total_bytes, 620822848ULL);
  expect_true("Pearl ordinary legal", pearlhash_allocation_counts(ordinary, ordinary_counts));
  expect_equal("Pearl ordinary complete required budget", ordinary_counts.total_bytes, 620790080ULL);
  expect_true("Pearl HIP duplicate layout legal",
               pearlhash_allocation_counts(ordinary, hip_counts, true));
  expect_equal("Pearl HIP unchanged maximum", hip_counts.max_alloc, ordinary_counts.max_alloc);
  expect_equal("Pearl HIP duplicate matrix bytes",
                hip_counts.total_bytes - ordinary_counts.total_bytes, 512 * kMiB);

  constexpr uint64_t transcript = 1 * kGiB;
  expect_true("Pearl optional transcript exact capacity",
      pearlhash_optional_allocation_fits(
          {80, compact_counts.total_bytes + reserve + transcript, transcript},
          compact_counts, transcript));
  expect_true("Pearl optional transcript preserves reserve",
      !pearlhash_optional_allocation_fits(
          {80, compact_counts.total_bytes + reserve + transcript - 1, transcript},
          compact_counts, transcript));
  expect_true("Pearl optional transcript respects maximum allocation",
      !pearlhash_optional_allocation_fits(
          {80, compact_counts.total_bytes + reserve + transcript, transcript - 1},
          compact_counts, transcript));
  expect_true("Pearl compact two GiB skips one GiB transcript",
      !pearlhash_optional_allocation_fits({80, 2 * kGiB, 512 * kMiB}, compact_counts,
                                          transcript));

  expect_pearl_profile("Pearl adequate modern memory unchanged",
      pearlhash_capacity_profile({48, 16 * kGiB, 8 * kGiB}, wide), wide);
  const struct {
    uint64_t max_alloc;
    pearlhash_profile_t expected;
  } allocation_cases[] = {
    {4 * kGiB - 1, compact}, {4 * kGiB, wide}, {4 * kGiB + 1, wide},
    {2 * kGiB, compact}, {256 * kMiB - 1, {}}, {256 * kMiB, compact},
    {256 * kMiB + 1, compact}
  };
  for (const auto& test : allocation_cases)
    expect_pearl_profile("Pearl max-single boundary",
        pearlhash_capacity_profile({80, 32 * kGiB, test.max_alloc}, wide), test.expected);

  const struct {
    uint64_t global_mem;
    pearlhash_profile_t expected;
  } total_cases[] = {
    {wide_counts.total_bytes + reserve - 1, compact},
    {wide_counts.total_bytes + reserve, wide},
    {wide_counts.total_bytes + reserve + 1, wide},
    {compact_counts.total_bytes + reserve - 1, {}},
    {compact_counts.total_bytes + reserve, compact},
    {compact_counts.total_bytes + reserve + 1, compact},
    {reserve - 1, {}}
  };
  for (const auto& test : total_cases)
    expect_pearl_profile("Pearl total and reserve boundary",
        pearlhash_capacity_profile({80, test.global_mem, 4 * kGiB}, wide), test.expected);

  for (const auto& device : {PowDeviceProfile{80, 0, 4 * kGiB},
                            PowDeviceProfile{80, 32 * kGiB, 0}, PowDeviceProfile{80, 0, 0}}) {
    expect_pearl_profile("Pearl unknown capacity preserves preferred",
                          pearlhash_capacity_profile(device, wide), wide);
    expect_true("Pearl unknown capacity never claims fit",
                 !pearlhash_required_memory_fits(device, wide));
  }
  expect_pearl_profile("Pearl UINT64 capacity does not overflow reserve arithmetic",
      pearlhash_capacity_profile({80, UINT64_MAX, UINT64_MAX}, wide), wide);
  expect_true("Pearl last aligned signed-index shape legal",
      pearlhash_allocation_counts({262112, 131072, 8192, 128}, ordinary_counts));
  expect_true("Pearl signed2GiB A-side boundary rejected",
      !pearlhash_allocation_counts({262144, 131072, 8192, 128}, ordinary_counts));
  expect_equal("Pearl invalid counts clear stale state", ordinary_counts.total_bytes, 0);
  expect_true("Pearl overflowing dimensions rejected before multiplication",
      !pearlhash_allocation_counts({UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX},
                                  ordinary_counts));
  expect_true("Pearl signed tile-index overflow rejected",
      !pearlhash_allocation_counts({131072, 4194304, 2048, 128}, ordinary_counts));
  expect_true("Pearl zero shape rejected", !pearlhash_allocation_counts({}, ordinary_counts));
}

} // namespace

int main() {
  const PowDeviceProfile wide_8_gib = {512, 8 * kGiB, 4 * kGiB};
  const PowDeviceProfile wide_12_gib = {512, 12 * kGiB, 4 * kGiB};
  const PowDeviceProfile wide_16_gib = {512, 16 * kGiB, 4 * kGiB};
  expect_scale("wide selection", select_pow_intensity_scale(wide_12_gib,
                                                            progpow_intensity_heuristic),
               progpow_intensity_heuristic.wide);

  // Synthetic high-EU profiles exercise the shared ProgPoW cap without naming a device.
  const unsigned cap = progpow_intensity_heuristic.max_work_items_per_gib;
  expect_equal("wide 8 GiB cap", pow_intensity(wide_8_gib, progpow_intensity_heuristic, 256),
               cap * 8 * kMiB);
  expect_equal("wide 12 GiB cap", pow_intensity(wide_12_gib, progpow_intensity_heuristic, 256),
               cap * 12 * kMiB);
  expect_equal("wide 16 GiB cap", pow_intensity(wide_16_gib, progpow_intensity_heuristic, 256),
               cap * 16 * kMiB);

  PowIntensityHeuristic uncapped_progpow = progpow_intensity_heuristic;
  uncapped_progpow.max_work_items_per_gib = 0;
  expect_equal("cap zero", pow_intensity(wide_12_gib, uncapped_progpow, 256), 119304448u);
  expect_equal("explicit cap 4", pow_intensity(wide_12_gib, uncapped_progpow, 256, 4),
               4 * 12 * kMiB);
  expect_equal("explicit cap 9", pow_intensity(wide_12_gib, uncapped_progpow, 256, 9),
               9 * 12 * kMiB);

  const PowDeviceProfile b580_like = {160, 12 * kGiB, 4 * kGiB};
  expect_equal("B580-like profile remains below cap",
               pow_intensity(b580_like, progpow_intensity_heuristic, 256),
               37282560u);

  const PowDeviceProfile compact_profile = {32, 4 * kGiB, 1 * kGiB};
  const PowDeviceProfile balanced_profile = {64, 6 * kGiB, 1 * kGiB};
  expect_scale("compact selection",
               select_pow_intensity_scale(compact_profile, progpow_intensity_heuristic),
               progpow_intensity_heuristic.compact);
  expect_scale("balanced selection",
               select_pow_intensity_scale(balanced_profile, progpow_intensity_heuristic),
               progpow_intensity_heuristic.balanced);

  const PowIntensityScale arithmetic_scale = {64, 3001, 3};
  const PowDeviceProfile arithmetic_profile = {5, 8 * kGiB, 1 * kGiB};
  const PowIntensityHeuristic arithmetic_heuristic = {arithmetic_scale, arithmetic_scale,
                                                      arithmetic_scale};
  const unsigned aligned = pow_intensity(arithmetic_profile, arithmetic_heuristic, 64);
  expect_equal("minimum and alignment", aligned, 320064u);
  expect_true("alignment", aligned % 64 == 0);

  const PowIntensityScale minimum_scale = {256, 1, 1};
  const PowDeviceProfile minimum_profile = {1, 1 * kGiB, 1 * kGiB};
  const PowIntensityHeuristic minimum_heuristic = {minimum_scale, minimum_scale, minimum_scale};
  expect_equal("minimum intensity", pow_intensity(minimum_profile, minimum_heuristic, 256),
               256u * 4096u);

  const PowIntensityScale clamp_scale = {1, 2, 1};
  const PowDeviceProfile large_valid_profile = {std::numeric_limits<unsigned>::max(), 8 * kGiB,
                                                4 * kGiB};
  const PowIntensityHeuristic clamp_heuristic = {clamp_scale, clamp_scale, clamp_scale};
  expect_equal("UINT32 clamp",
               pow_intensity(large_valid_profile, clamp_heuristic, 1),
               std::numeric_limits<unsigned>::max());

  const PowDeviceProfile sub_gib_profile = {512, 512 * kMiB, 4 * kGiB};
  expect_equal("cap minimum one GiB",
               pow_intensity(sub_gib_profile, progpow_intensity_heuristic, 256), 4 * kMiB);

  const unsigned before_pearl = assertions;
  test_pearlhash_capacity();
  std::printf("Pearl capacity host test passed (%u assertions)\n", assertions - before_pearl);
  std::printf("Pow intensity host test passed (%u assertions total)\n", assertions);
  return 0;
}
