#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/mom-intel-geometry.XXXXXX")
trap 'rm -rf -- "$out"' EXIT
cxx=${CXX:-c++}

# Test exact production statements without importing SYCL or duplicating the policy.
awk '
  /^inline unsigned mom_intel_eu_simd_width\(/ { copies++; copying = 1 }
  copying { print }
  copying && /^}/ { copying = 0; complete = 1 }
  END { if (copies != 1 || !complete) exit 1 }
' "$root/sycl/lib-internal.h" > "$out/intel_geometry_helper.inc"
test -s "$out/intel_geometry_helper.inc"

# Keep the actual profile guard and complete wrapper; the fixture mocks only submission.
mkdir -p "$out/sycl/ext/intel/experimental"
printf '%s\n' '// Test-only property-header presence; no SYCL API implementation.' > \
  "$out/sycl/ext/intel/experimental/grf_size_properties.hpp"
awk '
  /^#if !defined\(MOM_SYCL_HAS_CUDA\)/ { copies++; copying = 1 }
  copying { print }
  copying && /^#endif/ { copying = 0; complete = 1 }
  END { if (copies != 1 || !complete) exit 1 }
' "$root/sycl/zelhash/zelhash.cpp" > "$out/zelhash_generator_guard.inc"
awk '
  /^template <bool GPU_COMPACT = false>/ { prefix = $0 }
  /^static sycl::event submit_gen_fill\(/ { copies++; copying = 1; print prefix }
  copying { print }
  copying && /^}/ { copying = 0; complete = 1 }
  END { if (copies != 1 || !complete) exit 1 }
' "$root/sycl/zelhash/generation.inc" > "$out/zelhash_generator_dispatch.inc"
test -s "$out/zelhash_generator_guard.inc"
test -s "$out/zelhash_generator_dispatch.inc"

awk '
  /const char\* const force_low_memory = std::getenv/ { copies++; copying = 1 }
  copying { print }
  copying && /throw std::runtime_error/ { copying = 0; complete = 1 }
  END { if (copies != 1 || !complete) exit 1 }
' "$root/sycl/c30/c30.cpp" > "$out/c30_layout_policy.inc"
awk '
  /^static bool kawpow_default_subgroup_exchange\(/ { copies++; copying = 1 }
  copying { print }
  copying && /^}/ { copying = 0; complete = 1 }
  END { if (copies != 1 || !complete) exit 1 }
' "$root/sycl/kawpow/state.inc" > "$out/kawpow_exchange_policy.inc"
test -s "$out/intel_geometry_helper.inc"
test -s "$out/c30_layout_policy.inc"
test -s "$out/kawpow_exchange_policy.inc"

for profile in native acpp portable cuda hip acpp-cuda; do
  flags=()
  case "$profile" in
    acpp) flags=(-DMOM_SYCL_ADAPTIVECPP) ;;
    portable) flags=(-DMOM_SYCL_PORTABLE_OPENCL) ;;
    cuda) flags=(-DMOM_SYCL_HAS_CUDA) ;;
    hip) flags=(-DMOM_SYCL_HAS_HIP) ;;
    acpp-cuda) flags=(-DMOM_SYCL_ADAPTIVECPP -DMOM_SYCL_ADAPTIVECPP_CUDA) ;;
  esac
  "$cxx" -std=c++17 -O2 -Wall -Wextra -Werror -pedantic "${flags[@]}" -I "$out" \
    "$root/tests/native/intel_geometry.cpp" -o "$out/$profile"
  printf '%s: ' "$profile"
  "$out/$profile"
done
