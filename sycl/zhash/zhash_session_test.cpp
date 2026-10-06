// Host-only lifecycle and footprint checks.  The tiny layout is injectable so this test never
// allocates a real ZHash arena and never needs a SYCL runtime or physical GPU.

#include <cstdint>
#include <iostream>

#include "zhash_session_layout.hpp"

namespace {

// Smallest test arena here that retains the production HIP parent-record widths.
using Tiny = mom_zhash::ZHashArenaLayout<512>;

int test_footprint() {
#ifdef MOM_ZHASH_INTEL_LATE_BUCKETS
  static_assert(mom_zhash::FastZHashArenaLayout::slot_count == 36700160);
  static_assert(mom_zhash::FastZHashArenaLayout::LaterLayout::slot_count == 46137344);
  static_assert(mom_zhash::FastZHashArenaLayout::device_bytes == 3690209532ull);
  static_assert(mom_zhash::FastZHashArenaLayout::local_memory_bytes == 19456);
  static_assert(mom_zhash::DefaultZHashArenaLayout::slot_count == 38273024);
  static_assert(mom_zhash::DefaultZHashArenaLayout::LaterLayout::slot_count == 46137344);
  static_assert(mom_zhash::DefaultZHashArenaLayout::device_bytes == 3753124092ull);
  static_assert(mom_zhash::DefaultZHashArenaLayout::local_memory_bytes == 19456);
#elif defined(MOM_SYCL_HAS_HIP)
  static_assert(mom_zhash::FastZHashArenaLayout::slot_count == 35717120);
  static_assert(mom_zhash::FastZHashArenaLayout::device_bytes == 2864898300ull);
  static_assert(mom_zhash::FastZHashArenaLayout::local_memory_bytes == 33824);
  static_assert(mom_zhash::DefaultZHashArenaLayout::slot_count == 37748736);
  static_assert(mom_zhash::DefaultZHashArenaLayout::device_bytes == 3027427580ull);
  static_assert(mom_zhash::DefaultZHashArenaLayout::local_memory_bytes == 34816);
#else
  static_assert(mom_zhash::FastZHashArenaLayout::slot_count == 35651584);
  static_assert(mom_zhash::FastZHashArenaLayout::device_bytes == 3144868092ull);
  static_assert(mom_zhash::FastZHashArenaLayout::local_memory_bytes == 33792);
  static_assert(mom_zhash::DefaultZHashArenaLayout::slot_count == 37748736);
  static_assert(mom_zhash::DefaultZHashArenaLayout::device_bytes == 3329417468ull);
  static_assert(mom_zhash::DefaultZHashArenaLayout::local_memory_bytes == 34816);
#endif
  static_assert(mom_zhash::DefaultZHashArenaLayout::device_bytes + 512ull * 1024 * 1024 <=
                4ull * 1024 * 1024 * 1024);
  static_assert(Tiny::slot_count == 2097152);
  static_assert(Tiny::level0_bytes == Tiny::slot_count * 20);
  static_assert(sizeof(mom_zhash::DefaultZHashArenaLayout::Round1Record) == 20);
  static_assert(Tiny::round1_bytes == Tiny::slot_count * 20);
  static_assert(Tiny::round2_bytes == Tiny::LaterLayout::slot_count * 16);
#if defined(MOM_SYCL_HAS_HIP)
  static_assert(Tiny::round3_bytes == Tiny::LaterLayout::slot_count * 12);
  static_assert(Tiny::round4_bytes == Tiny::LaterLayout::slot_count * 12);
#else
  static_assert(Tiny::round3_bytes == Tiny::LaterLayout::slot_count * 16);
  static_assert(Tiny::round4_bytes == Tiny::LaterLayout::slot_count * 16);
#endif
  constexpr std::size_t expected = Tiny::provenance_bytes + Tiny::bucket_counts_bytes +
                                   Tiny::candidate_root_bytes + Tiny::recovered_leaf_bytes +
                                   Tiny::recovered_valid_bytes + Tiny::verified_field_bytes +
                                   Tiny::counter_bytes;
  static_assert(Tiny::device_bytes == expected);
  if (!Tiny::is_valid() || Tiny::device_bytes != expected)
    return 1;
  return 0;
}

int test_lifecycle() {
  mom_zhash::ZHashSessionLifecycle lifecycle(Tiny::slot_count);
  if (!lifecycle.begin() || lifecycle.status() != mom_zhash::SessionStatus::running)
    return 1;
  if (!lifecycle.generated(0) || lifecycle.stage() != 1)
    return 2;
  for (unsigned round = 0; round < 4; ++round) {
    if (!lifecycle.flat_output(12u - round) || !lifecycle.scattered(0))
      return 3;
  }
  if (lifecycle.stage() != 5 || !lifecycle.flat_output(1) || !lifecycle.roots(1, 1))
    return 4;
  lifecycle.finish(true);
  if (lifecycle.status() != mom_zhash::SessionStatus::solved || lifecycle.root_count() != 1 ||
      lifecycle.zero_root_count() != 1)
    return 5;
  return 0;
}

int test_overflow_resets_attempt() {
  mom_zhash::ZHashSessionLifecycle lifecycle(4);
  if (!lifecycle.begin() || !lifecycle.generated(0))
    return 1;
  if (lifecycle.flat_output(5) || lifecycle.status() != mom_zhash::SessionStatus::capacity_overflow)
    return 2;
  if (!lifecycle.begin() || lifecycle.status() != mom_zhash::SessionStatus::running)
    return 3;
  if (lifecycle.generated(1) || lifecycle.status() != mom_zhash::SessionStatus::capacity_overflow)
    return 4;
  if (!lifecycle.begin() || !lifecycle.generated(0))
    return 5;
  if (!lifecycle.flat_output(1) || lifecycle.scattered(1) ||
      lifecycle.status() != mom_zhash::SessionStatus::capacity_overflow)
    return 6;
  return 0;
}

} // namespace

int main() {
  if (const int result = test_footprint())
    return 10 + result;
  if (const int result = test_lifecycle())
    return 20 + result;
  if (const int result = test_overflow_resets_attempt())
    return 30 + result;
  std::cout << "ZHash session layout/lifecycle: PASS\n";
  return 0;
}
