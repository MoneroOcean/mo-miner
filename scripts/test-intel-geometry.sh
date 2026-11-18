#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/mom-intel-geometry.XXXXXX")
trap 'rm -rf -- "$out"' EXIT
cxx=${CXX:-c++}

# Print whole production declarations through the shared token-anchored extractor; a missing or
# ambiguous anchor fails the run.
extract_cpp() {
  local name=$1 file=$2
  shift 2
  node "$root/tests/common/cpp-source.js" "$root/$file" "$@" > "$out/$name"
}

extract_cpp intel_geometry_helper.inc sycl/lib-internal.h 'inline unsigned mom_intel_eu_simd_width('

# Keep the actual large-GRF build guard and complete wrapper; the fixture mocks only submission.
mkdir -p "$out/sycl/ext/intel/experimental"
printf '%s\n' '// Test-only property-header presence; no SYCL API implementation.' > \
  "$out/sycl/ext/intel/experimental/grf_size_properties.hpp"
awk '
  /^#if/ { group = ""; grouping = 1 }
  grouping { group = group $0 "\n" }
  /^#endif/ && grouping {
    grouping = 0
    if (group ~ /#define MOM_ZELHASH_INTEL_LARGE_GRF/) { printf "%s", group; found++ }
  }
  END { if (found != 1) exit 1 }
' "$root/sycl/zelhash/zelhash.cpp" > "$out/zelhash_generator_guard.inc"
extract_cpp zelhash_generator_dispatch.inc sycl/zelhash/generation.inc \
  'static sycl::event submit_gen_fill('

extract_cpp c30_device_memory.inc sycl/c30/c30.cpp 'struct DeviceMemory {'
printf ';\n' >> "$out/c30_device_memory.inc"
extract_cpp kawpow_exchange_policy.inc sycl/kawpow/state.inc \
  'static bool kawpow_default_subgroup_exchange('

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
    "$root/tests/native/intel-geometry.cpp" -o "$out/$profile"
  printf '%s: ' "$profile"
  "$out/$profile"
done
