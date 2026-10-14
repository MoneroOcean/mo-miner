// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include "../../sycl/pearlhash/esimd_route.h"

#include <cstdio>
#include <vector>

int main() {
  const struct {
    const char* name;
    bool tuned_sycl;
    bool supported_backend;
    bool discrete_gpu;
    bool intel_matrix;
    unsigned dpas_width;
    bool expected;
  } cases[] = {
    {"Intel Level Zero width8", true, true, true, true, 8u, true},
    {"Intel Level Zero width16", true, true, true, true, 16u, true},
    {"Intel OpenCL width8 opted in", true, true, true, true, 8u, true},
    {"Intel OpenCL width16 opted in", true, true, true, true, 16u, true},
    {"Intel OpenCL without opt-in", true, false, true, true, 8u, false},
    {"AMD OpenCL forced override", true, true, true, false, 16u, false},
    {"CPU forced override", true, true, false, true, 8u, false},
    {"integrated GPU forced override", true, true, false, true, 16u, false},
    {"missing Intel matrix capability", true, true, true, false, 8u, false},
    {"unknown or failed DPAS query", true, true, true, true, 0u, false},
    {"unsupported DPAS width4", true, true, true, true, 4u, false},
    {"unsupported DPAS width32", true, true, true, true, 32u, false},
    {"unsupported backend forced override", true, false, true, true, 16u, false},
    {"generic SYCL request width8", false, true, true, true, 8u, false},
    {"generic SYCL request width16", false, true, true, true, 16u, false}
  };
  for (const auto& fixture : cases) {
    const bool actual = pearlhash_esimd_route(
        fixture.tuned_sycl, fixture.supported_backend, fixture.discrete_gpu,
        fixture.intel_matrix, fixture.dpas_width);
    if (actual != fixture.expected) {
      std::fprintf(stderr, "PearlHash ESIMD routing mismatch: %s\n", fixture.name);
      return 1;
    }
  }
  const struct {
    const char* name;
    bool esimd_allowed;
    unsigned width;
    bool dg2;
    int rank;
    bool expected;
  } paired[] = {
    {"DG2 P128", true, 8u, true, 128, true},
    {"DG2 P256", true, 8u, true, 256, true},
    {"B580 width16 unchanged", true, 16u, false, 128, false},
    {"width16 even if mislabeled DG2", true, 16u, true, 256, false},
    {"unknown architecture", true, 8u, false, 128, false},
    {"failed width query", true, 0u, true, 128, false},
    {"portable backend", false, 8u, true, 128, false},
    {"unqualified OpenCL retains ordinary", false, 8u, true, 128, false},
    {"CPU or integrated GPU", false, 8u, false, 256, false},
    {"rank512 retains ordinary DPAS", true, 8u, true, 512, false},
    {"rank1024 retains ordinary DPAS", true, 8u, true, 1024, false},
    {"rank0", true, 8u, true, 0, false},
    {"negative rank", true, 8u, true, -1, false},
    {"malformed rank", true, 8u, true, 129, false}
  };
  for (const auto& fixture : paired) {
    if (pearlhash_dpasw_route(fixture.esimd_allowed, fixture.width, fixture.dg2,
                            fixture.rank) != fixture.expected) {
      std::fprintf(stderr, "PearlHash DPASW routing mismatch: %s\n", fixture.name);
      return 1;
    }
  }
  const struct {
    unsigned width;
    int rank;
    int columns;
    bool expected;
  } wide[] = {
    {16u, 128, 131072, true},
    {16u, 128, 192, true},
    {16u, 128, 160, false},
    {16u, 128, 0, false},
    {16u, 128, -64, false},
    {16u, 256, 131072, false},
    {16u, 512, 131072, false},
    {8u, 128, 131072, false},
    {0u, 128, 131072, false}
  };
  for (const auto& fixture : wide) {
    if (pearlhash_wide_pipeline_route(fixture.width, fixture.rank, fixture.columns) !=
        fixture.expected) {
      std::fprintf(stderr, "PearlHash wide pipeline routing mismatch: width=%u rank=%d n=%d\n",
                   fixture.width, fixture.rank, fixture.columns);
      return 1;
    }
  }
  for (const unsigned rows : {64u, 128u, 256u}) {
    for (const unsigned columns : {2u, 8u, 64u}) {
      std::vector<bool> seen(rows * columns);
      for (unsigned wi = 0; wi < rows * columns; ++wi) {
        unsigned row, column;
        pearlhash_esimd_stripe_position(wi, columns, row, column);
        if (row >= rows || column >= columns || seen[row * columns + column]) {
          std::fputs("PearlHash striped traversal is not a bijection\n", stderr);
          return 1;
        }
        seen[row * columns + column] = true;
      }
    }
  }
  std::puts("PearlHash ESIMD routing and striped traversal host tests passed");
  return 0;
}
