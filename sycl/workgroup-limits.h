// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <cstddef>

constexpr bool mom_subgroup_supports_team(const std::size_t subgroup, const unsigned team_size) {
  return team_size && subgroup >= team_size && subgroup % team_size == 0u;
}

constexpr bool mom_workgroup_fits(
    const unsigned workgroup, const std::size_t max_workgroup,
    const std::size_t max_work_items, const std::size_t local_bytes,
    const std::size_t fixed_local_bytes = 0, const std::size_t local_bytes_per_item = 0,
    const std::size_t local_bytes_per_team = 0, const unsigned team_size = 1) {
  if (!workgroup || workgroup > max_workgroup || workgroup > max_work_items ||
      fixed_local_bytes > local_bytes || !team_size || workgroup % team_size != 0u)
    return false;
  std::size_t available = local_bytes - fixed_local_bytes;
  const unsigned teams = workgroup / team_size;
  // Check before multiplying so hostile/overflow-sized requirements cannot appear to fit.
  if (local_bytes_per_team > available / teams)
    return false;
  available -= local_bytes_per_team * teams;
  return local_bytes_per_item <= available / workgroup;
}
