// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

namespace mom::octopus {

enum class TestPathFailure { none, full_dag, native };

// A full-DAG correctness vector may use the portable kernel. Native-proof vectors must not
// conceal a failed optimized path on capability-positive hardware, but unsupported devices fall back.
constexpr TestPathFailure test_path_failure(const bool require_full_dag, const bool full_dag,
                                          const bool prove_native, const bool native_supported,
                                          const bool native_searched) {
  if (require_full_dag && !full_dag)
    return TestPathFailure::full_dag;
  if (prove_native && native_supported && !native_searched)
    return TestPathFailure::native;
  return TestPathFailure::none;
}

} // namespace mom::octopus
