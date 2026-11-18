// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include "../../sycl/workgroup-limits.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>

namespace {

unsigned assertions = 0;

void require(const bool condition, const char* const label) {
  ++assertions;
  if (!condition) {
    std::fprintf(stderr, "Workgroup limits mismatch: %s\n", label);
    std::abort();
  }
}

} // namespace

int main() {
  constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
  require(mom_workgroup_fits(256, 256, 256, 16384, 16384), "exact fixed-memory fit");
  require(!mom_workgroup_fits(256, 255, 256, 16384, 16384), "workgroup one below");
  require(!mom_workgroup_fits(256, 256, 255, 16384, 16384), "one-dimensional limit one below");
  require(!mom_workgroup_fits(256, 256, 256, 16383, 16384), "local memory one below");
  require(!mom_workgroup_fits(0, 256, 256, 16384), "zero workgroup");
  require(!mom_workgroup_fits(1, 0, 256, 16384), "zero workgroup limit");
  require(!mom_workgroup_fits(1, 256, 0, 16384), "zero one-dimensional limit");
  require(!mom_workgroup_fits(1, 256, 256, 0, 1), "zero memory cannot hold fixed scratch");
  require(mom_workgroup_fits(1, 256, 256, 0), "zero-memory kernel needs no scratch");
  require(mom_workgroup_fits(256, 256, 256, 1024, 0, 4), "exact per-item fit");
  require(!mom_workgroup_fits(256, 256, 256, 1023, 0, 4), "per-item memory one below");
  require(mom_workgroup_fits(32, 32, 32, 264, 128, 4, 4, 16), "exact team scratch fit");
  require(!mom_workgroup_fits(32, 32, 32, 263, 128, 4, 4, 16), "team scratch one below");
  require(!mom_workgroup_fits(32, 32, 32, 264, 128, 4, 4, 0), "zero team size");
  require(!mom_workgroup_fits(31, 32, 32, 264, 128, 4, 4, 16), "incomplete team");
  require(!mom_workgroup_fits(16, 32, 32, 264, 128, 4, 4, 32), "team exceeds workgroup");
  require(mom_workgroup_fits(1, maximum, maximum, maximum, maximum), "maximum fixed fit");
  require(!mom_workgroup_fits(1, maximum, maximum, maximum, maximum, 1),
          "fixed plus per-item overflow cannot fit");
  require(!mom_workgroup_fits(2, maximum, maximum, maximum, 0, maximum),
          "per-item multiplication overflow cannot fit");
  require(!mom_workgroup_fits(32, maximum, maximum, maximum, 0, 0, maximum, 16),
          "per-team multiplication overflow cannot fit");

  const struct {
    unsigned workgroup;
    std::size_t bytes;
  } kawpow[] = {{64, 16656}, {128, 16928}, {256, 17472}, {512, 18560}};
  for (const auto& fixture : kawpow) {
    // The common ProgPoW search allocates uint32 cache[4096], share[WG], and offsets[WG/16].
    require(mom_workgroup_fits(fixture.workgroup, 1024, 1024, fixture.bytes,
                               4096 * sizeof(std::uint32_t), sizeof(std::uint32_t),
                               sizeof(std::uint32_t), 16), "exact ProgPoW scratch fit");
    require(!mom_workgroup_fits(fixture.workgroup, 1024, 1024, fixture.bytes - 1,
                                4096 * sizeof(std::uint32_t), sizeof(std::uint32_t),
                                sizeof(std::uint32_t), 16), "ProgPoW scratch one below");
    require(mom_workgroup_fits(fixture.workgroup, 1024, 1024, 65536,
                               4096 * sizeof(std::uint32_t), sizeof(std::uint32_t),
                               sizeof(std::uint32_t), 16), "ordinary ProgPoW sizes unchanged");
  }
  for (const unsigned subgroup : {16u, 32u, 64u})
    require(mom_subgroup_supports_team(subgroup, 16), "native subgroup holds complete teams");
  for (const unsigned subgroup : {0u, 1u, 8u, 24u})
    require(!mom_subgroup_supports_team(subgroup, 16), "native subgroup cannot split a team");
  require(!mom_subgroup_supports_team(32, 0), "unknown team size");
  std::printf("Workgroup limits host test passed (%u assertions)\n", assertions);
  return 0;
}
