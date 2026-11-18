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
  PearlHashAllocationCounts wide_counts{}, compact_counts{}, ordinary_counts{};
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

void test_pearlhash_amd_native_profile() {
  const pearlhash_profile_t ordinary = {131072, 131072, 2048, 128};
  const pearlhash_profile_t candidate = {262144, 262144, 2048, 128};
  const pearlhash_profile_t native = {524288, 524288, 2048, 128};
  PearlHashAllocationCounts candidate_counts{}, native_counts{};
  expect_true("Pearl AMD candidate shape legal",
              mom::job_boundary::valid_pearlhash_shape(
                  candidate.m, candidate.n, candidate.k, candidate.rank));
  expect_true("Pearl AMD native shape legal",
              mom::job_boundary::valid_pearlhash_shape(
                  native.m, native.n, native.k, native.rank));
  expect_true("Pearl AMD candidate allocation counts",
              pearlhash_allocation_counts(candidate, candidate_counts));
  expect_true("Pearl AMD native allocation counts",
              pearlhash_allocation_counts(native, native_counts));
  const uint64_t exact_global = candidate_counts.total_bytes + pearlhash_runtime_reserve_bytes;
  const uint64_t exact_alloc = candidate_counts.max_alloc;
  const uint64_t native_global = native_counts.total_bytes + pearlhash_runtime_reserve_bytes;
  const uint64_t native_alloc = native_counts.max_alloc;

  expect_pearl_profile("Pearl AMD native candidate exact fit",
      pearlhash_amd_native_profile_for({80, native_global, native_alloc}, true), native);
  expect_pearl_profile("Pearl AMD native global boundary selects next candidate",
      pearlhash_amd_native_profile_for({80, native_global - 1, native_alloc}, true), candidate);
  expect_pearl_profile("Pearl AMD native allocation boundary selects next candidate",
      pearlhash_amd_native_profile_for({80, native_global, native_alloc - 1}, true), candidate);
  expect_pearl_profile("Pearl AMD candidate exact fit",
      pearlhash_amd_native_profile_for({80, exact_global, exact_alloc}, true),
      candidate);
  expect_pearl_profile("Pearl AMD candidate global boundary",
      pearlhash_amd_native_profile_for({80, exact_global - 1, exact_alloc}, true), ordinary);
  expect_pearl_profile("Pearl AMD candidate allocation boundary",
      pearlhash_amd_native_profile_for({80, exact_global, exact_alloc - 1}, true), ordinary);
  expect_pearl_profile("Pearl AMD unadmitted route fallback",
      pearlhash_amd_native_profile_for({80, 16 * kGiB, 8 * kGiB}, false), ordinary);
  expect_pearl_profile("Pearl AMD unknown capacity fallback",
      pearlhash_amd_native_profile_for({80, 0, 0}, true), ordinary);
}

void test_pearlhash_intel_capability_profile() {
  const pearlhash_profile_t ordinary = {131072, 131072, 2048, 128};
  const pearlhash_profile_t wide = {131072, 131072, 4096, 128};
  const uint64_t eight_gib = 8 * kGiB;

  expect_pearl_profile("Pearl Intel width16 Level Zero profile",
                       pearlhash_intel_capability_profile(16, true, true, eight_gib), wide);
  expect_pearl_profile("Pearl Intel width16 larger memory profile",
                       pearlhash_intel_capability_profile(16, true, true, 12 * kGiB), wide);
  for (const unsigned width : {0u, 8u, 32u})
    expect_pearl_profile("Pearl Intel unsupported width fallback",
                         pearlhash_intel_capability_profile(width, true, true, eight_gib), ordinary);
  expect_pearl_profile("Pearl Intel width16 low memory fallback",
                       pearlhash_intel_capability_profile(16, true, true, eight_gib - 1), ordinary);
  expect_pearl_profile("Pearl Intel width16 non-Level-Zero fallback",
                       pearlhash_intel_capability_profile(16, false, true, eight_gib), ordinary);
  expect_pearl_profile("Pearl integrated width16 fallback",
                       pearlhash_intel_capability_profile(16, true, false, eight_gib), ordinary);
  expect_pearl_profile("Pearl Intel narrow larger device",
      pearlhash_intel_capability_profile(8, true, true, eight_gib, 512),
      {524288, 131072, 2048, 128});
  expect_pearl_profile("Pearl Intel narrow intermediate device",
      pearlhash_intel_capability_profile(8, true, true, eight_gib, 511),
      {262144, 131072, 2048, 128});
  expect_pearl_profile("Pearl Intel narrow smaller device",
      pearlhash_intel_capability_profile(8, true, true, eight_gib, 255), ordinary);
  expect_pearl_profile("Pearl Intel narrow low memory fallback",
      pearlhash_intel_capability_profile(8, true, true, eight_gib - 1, 512), ordinary);
  expect_pearl_profile("Pearl Intel narrow row count capped",
      pearlhash_intel_capability_profile(8, true, true, eight_gib,
          std::numeric_limits<unsigned>::max()), {524288, 131072, 2048, 128});
  expect_pearl_profile("Pearl Intel wide profile does not inherit narrow scaling",
      pearlhash_intel_capability_profile(16, true, true, eight_gib, 512), wide);
}

// Exercise the real policy across hardware sizes and exact allocation limits;
// no SDK or GPU is required to catch signed overflow or lost fallback paths.
void test_pearlhash_nvidia_compact_profile() {
  constexpr pearlhash_profile_t fast = {27648,32768,65536,128};
  constexpr pearlhash_profile_t large = {32256,15360,65536,128};
  constexpr pearlhash_profile_t small = {16128,15360,65536,128};
  const PowDeviceProfile roomy = {36,16*kGiB,8*kGiB};
  expect_pearl_profile("Pearl native larger preferred",
      pearlhash_nvidia_compact_profile_for(roomy,true),fast);
  for (unsigned units = 1; units <= 512; ++units) {
    const PowDeviceProfile d = {units,16*kGiB,8*kGiB};
    const auto p = pearlhash_nvidia_compact_profile_for(d,true);
    const uint64_t quantum = uint64_t(192)*units/std::gcd(units,60u);
    const uint64_t minimum = (16000+quantum-1)/quantum*quantum;
    const uint64_t bound = uint64_t(std::numeric_limits<int32_t>::max())/65536;
    if (minimum > bound) {
      expect_equal("Pearl impossible complete wave",p.m,0);
    } else {
      expect_pearl_profile("Pearl measured or largest complete wave",p, units == 36 ? fast :
          pearlhash_profile_t{unsigned(bound/quantum*quantum),15360,65536,128});
      expect_equal("Pearl complete native CTA rows",p.m%192,0);
      expect_equal("Pearl complete native CTA columns",p.n%256,0);
      expect_equal("Pearl no partial SM wave",(uint64_t(p.m)/192*(p.n/256))%units,0);
      expect_true("Pearl signed M*K",uint64_t(p.m)*p.k<=uint64_t(INT32_MAX));
      expect_true("Pearl disjoint raw streams",(uint64_t(p.m)+p.n)*p.k<=(uint64_t(1)<<32));
      expect_true("Pearl whole allocation admission",pearlhash_required_memory_fits(d,p));
    }
    expect_equal("Pearl no native admission",pearlhash_nvidia_compact_profile_for(d,false).m,0);
  }
  for (unsigned units : {0u,65535u,26843546u,std::numeric_limits<unsigned>::max()})
    expect_equal("Pearl unknown/extreme CU decline",
        pearlhash_nvidia_compact_profile_for({units,16*kGiB,8*kGiB},true).m,0);
  PearlHashAllocationCounts fast_counts{},big{},little{};
  expect_true("Pearl fast required counts",pearlhash_allocation_counts(fast,fast_counts));
  expect_equal("Pearl fast memory budget",fast_counts.total_bytes,4219863360ULL);
  expect_equal("Pearl fast largest allocation",fast_counts.max_alloc,2*kGiB);
  expect_pearl_profile("Pearl fast exact memory",
      pearlhash_nvidia_compact_profile_for(
          {36,fast_counts.total_bytes+pearlhash_runtime_reserve_bytes,fast_counts.max_alloc},true),fast);
  expect_pearl_profile("Pearl fast total-memory fallback",
      pearlhash_nvidia_compact_profile_for(
          {36,fast_counts.total_bytes+pearlhash_runtime_reserve_bytes-1,fast_counts.max_alloc},true),large);
  expect_pearl_profile("Pearl fast max-allocation fallback",
      pearlhash_nvidia_compact_profile_for({36,16*kGiB,fast_counts.max_alloc-1},true),large);
  expect_true("Pearl larger required counts",pearlhash_allocation_counts(large,big));
  expect_true("Pearl smaller required counts",pearlhash_allocation_counts(small,little));
  expect_pearl_profile("Pearl larger exact memory",
      pearlhash_nvidia_compact_profile_for(
          {36,big.total_bytes+pearlhash_runtime_reserve_bytes,big.max_alloc},true),large);
  expect_pearl_profile("Pearl total-memory fallback",
      pearlhash_nvidia_compact_profile_for(
          {36,big.total_bytes+pearlhash_runtime_reserve_bytes-1,big.max_alloc},true),small);
  expect_pearl_profile("Pearl max-allocation fallback",
      pearlhash_nvidia_compact_profile_for({36,16*kGiB,big.max_alloc-1},true),small);
  expect_pearl_profile("Pearl smaller exact memory",
      pearlhash_nvidia_compact_profile_for(
          {36,little.total_bytes+pearlhash_runtime_reserve_bytes,little.max_alloc},true),small);
  for (const auto d : {PowDeviceProfile{36,little.total_bytes+pearlhash_runtime_reserve_bytes-1,little.max_alloc},
       {36,16*kGiB,little.max_alloc-1},{36,0,8*kGiB},{36,16*kGiB,0},
       {36,pearlhash_runtime_reserve_bytes-1,8*kGiB}})
    expect_equal("Pearl insufficient/unknown capacity",
        pearlhash_nvidia_compact_profile_for(d,true).m,0);
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
  test_pearlhash_intel_capability_profile();
  test_pearlhash_capacity();
  test_pearlhash_amd_native_profile();
  test_pearlhash_nvidia_compact_profile();
  std::printf("Pearl capacity host test passed (%u assertions)\n", assertions - before_pearl);
  std::printf("Pow intensity host test passed (%u assertions total)\n", assertions);
  return 0;
}
