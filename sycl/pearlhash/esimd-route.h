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

// One work-group is a whole Xe core at 256 GRFs (32 threads): 8 row groups by 4 column groups, so
// its threads share every A band four ways and every B bank eight ways through L1.
inline constexpr unsigned pearlhash_esimd_group_rows = 8, pearlhash_esimd_group_columns = 4;

// Work-groups stacked per stripe. B streams past each stripe once, so taller stripes cut memory
// traffic until their A rows (32 per row group) outgrow about 8 MiB of the L2 cache.
constexpr unsigned pearlhash_esimd_strips(const unsigned k) {
  const unsigned strips = (8u << 20) / (k * 32u * pearlhash_esimd_group_rows);
  return strips ? strips : 1u;
}

// One block is one work-group; blocks walk down a stripe before stepping to the next columns.
inline void pearlhash_esimd_stripe_position(unsigned wi, unsigned tiles_w, unsigned strips,
                                           unsigned& row, unsigned& column) {
  constexpr unsigned rows = pearlhash_esimd_group_rows, columns = pearlhash_esimd_group_columns;
  const unsigned block = wi / (rows * columns), inner = wi % (rows * columns);
  const unsigned blocks_w = tiles_w / columns;
  const unsigned stripe = block / (strips * blocks_w), within = block % (strips * blocks_w);
  row = (strips * stripe + within % strips) * rows + inner / columns;
  column = (within / strips) * columns + inner % columns;
}
