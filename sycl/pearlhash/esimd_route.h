// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

// An explicit backend request cannot supply missing hardware capabilities.
constexpr bool pearlhash_esimd_route(const bool tuned_sycl, const bool supported_backend,
                                    const bool discrete_gpu, const bool intel_matrix,
                                    const unsigned dpas_width) {
  return tuned_sycl && supported_backend && discrete_gpu && intel_matrix &&
         (dpas_width == 8u || dpas_width == 16u);
}

// Pair only the DG2 profiles qualified against a full independent proof.
constexpr bool pearlhash_dpasw_route(const bool esimd_allowed, const unsigned dpas_width,
                                    const bool dg2, const int rank) {
  return esimd_allowed && dpas_width == 8u && dg2 && (rank == 128 || rank == 256);
}
