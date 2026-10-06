#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/mom-zelhash-layout.XXXXXX")
trap 'rm -rf -- "$out"' EXIT
cxx=${CXX:-c++}

# Extract complete production arithmetic; vector stores are tested by full SYCL compilation.
awk '
  { print }
  /^inline uint32_t dense_l0_load_index\(/ { copies++; copying = 1 }
  copying && /^}/ { complete++; exit }
  END { if (copies != 1 || complete != 1) exit 1 }
' "$root/sycl/zelhash/layout.inc" > "$out/zelhash_layout_math.inc"
awk '
  /^static constexpr uint64_t B2B_IV/ { copies++; copying = 1 }
  copying { print }
  END { if (copies != 1) exit 1 }
' "$root/sycl/zelhash/layout.inc" > "$out/zelhash_blake_constants.inc"
awk '
  /^inline void dense_l0_store_compact\(/ { copies++; copying = 1 }
  copying { print }
  copying && /^}/ { complete++; copying = 0 }
  END { if (copies != 1 || complete != 1 || copying) exit 1 }
' "$root/sycl/zelhash/layout.inc" > "$out/zelhash_compact_store.inc"
awk '
  /^template <bool GPU_COMPACT>/ { prefix = $0 }
  /^inline uint32_t recovery_follower\(/ { copies++; copying = 1; print prefix }
  copying { print }
  copying && /^}/ { complete++; copying = 0 }
  END { if (copies != 1 || complete != 1 || copying) exit 1 }
' "$root/sycl/zelhash/recovery.inc" > "$out/zelhash_recovery_follower.inc"
awk '
  /if \(lid < PROOF && valid\[0\]\)/ { copies++; copying = 1 }
  copying && /sycl::group_barrier/ { complete++; copying = 0 }
  copying { print }
  END { if (copies != 1 || complete != 1 || copying) exit 1 }
' "$root/sycl/zelhash/recovery.inc" > "$out/zelhash_recovery_leaf.inc"
awk '
  /^unsigned zelhash_slot_capacity\(/ { copies++; copying = 1 }
  copying && /^static DeviceStateRegistry/ { complete++; copying = 0 }
  copying { print }
  END { if (copies != 1 || complete != 1 || copying) exit 1 }
' "$root/sycl/zelhash/state.inc" > "$out/zelhash_state.inc"
awk '
  /^  const bool filter_target/ { copies++; copying = 1 }
  copying && /^  ZelHashState& state/ { complete++; copying = 0 }
  copying { print }
  END { if (copies != 1 || complete != 1 || copying) exit 1 }
' "$root/sycl/zelhash/entry.inc" > "$out/zelhash_entry_guard.inc"
test "$(grep -cF 'if (!filter_target)' "$root/sycl/zelhash/entry.inc")" -eq 1
awk '
  /^inline bool mom_parse_env_ulong\(/ { copies++; copying = 1 }
  copying { print }
  copying && /^}/ { complete++; copying = 0 }
  END { if (copies != 1 || complete != 1 || copying) exit 1 }
' "$root/sycl/lib-internal.h" > "$out/ulong_parser.inc"

for profile in native portable cuda hip acpp acpp-cuda; do
  flags=()
  case "$profile" in
    portable) flags=(-DMOM_SYCL_PORTABLE_OPENCL) ;;
    cuda) flags=(-DMOM_SYCL_HAS_CUDA) ;;
    hip) flags=(-DMOM_SYCL_HAS_HIP) ;;
    acpp) flags=(-DMOM_SYCL_ADAPTIVECPP) ;;
    acpp-cuda) flags=(-DMOM_SYCL_ADAPTIVECPP -DMOM_SYCL_ADAPTIVECPP_CUDA) ;;
  esac
  "$cxx" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic "${flags[@]}" -I "$out" \
    "$root/tests/native/zelhash_layout.cpp" -o "$out/$profile"
  printf '%s: ' "$profile"
  "$out/$profile"
  "$cxx" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic "${flags[@]}" -I "$out" \
    "$root/tests/native/zelhash_state.cpp" -o "$out/$profile-state"
  printf '%s-state: ' "$profile"
  "$out/$profile-state"
done
