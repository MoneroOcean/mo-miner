// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include "../../sycl/pearlhash/esimd_route.h"

#include <cstdio>

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
  std::puts("PearlHash ESIMD routing host test passed (15 fixtures)");
  return 0;
}
