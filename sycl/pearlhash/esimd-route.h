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

// Four column tiles must cover the full matrix; retain the ordinary route for partial groups.
constexpr bool pearlhash_wide_pipeline_route(const unsigned dpas_width, const int rank,
                                           const int columns) {
  return dpas_width == 16u && rank == 128 && columns > 0 && columns % 64 == 0;
}

// Traverse four 16-row strips and two column groups to retain B operands across nearby work.
inline void pearlhash_esimd_stripe_position(unsigned wi, unsigned tiles_w,
                                           unsigned& row, unsigned& column) {
  const unsigned block = wi / 32, inner = wi % 32;
  const unsigned blocks_w = tiles_w / 2;
  const unsigned stripe = block / (4 * blocks_w), within = block % (4 * blocks_w);
  row = (4 * stripe + within % 4) * 16 + inner / 2;
  column = (within / 4) * 2 + inner % 2;
}
