#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/mom-zelhash-layout.XXXXXX")
trap 'rm -rf -- "$out"' EXIT
cxx=${CXX:-c++}

# Print whole production declarations through the shared token-anchored extractor; a missing or
# ambiguous anchor fails the run. Indented anchors name nested members or statements exactly.
extract_cpp() {
  local name=$1 file=$2
  shift 2
  node "$root/tests/common/cpp-source.js" "$root/$file" "$@" > "$out/$name"
}

# Production geometry and scalar record arithmetic: layout.inc through its leaf decoder. The vector
# stores after it use clang-only types and are covered by full SYCL compilation instead.
node -e '
  const fs = require("node:fs"), {extractCppBlock} = require(process.argv[1]);
  const source = fs.readFileSync(process.argv[2], "utf8");
  const block = extractCppBlock(source, process.argv[3]);
  process.stdout.write(source.slice(0, source.indexOf(block) + block.length));
' "$root/tests/common/cpp-source.js" "$root/sycl/zelhash/layout.inc" \
  'inline uint32_t dense_l0_load_index(' > "$out/zelhash_layout_math.inc"
extract_cpp zelhash_compact_store.inc sycl/zelhash/layout.inc 'inline void dense_l0_store_compact('
extract_cpp zelhash_recovery_follower.inc sycl/zelhash/recovery.inc \
  'inline uint32_t recovery_follower('
extract_cpp zelhash_recovery_leaf.inc sycl/zelhash/recovery.inc \
  '          if (lid < PROOF && valid[0]) {'
extract_cpp zelhash_state.inc sycl/zelhash/state.inc 'unsigned zelhash_slot_capacity(' \
  'static unsigned zelhash_configured_slots(' 'class State {'
printf ';\n' >> "$out/zelhash_state.inc"
extract_cpp zelhash_entry_guard.inc sycl/zelhash/zelhash.cpp '  if (!input || !solution_out'
extract_cpp usm_ptr.inc sycl/lib-internal.h 'class UsmPtr {'
printf ';\n' >> "$out/usm_ptr.inc"
extract_cpp usm_alloc.inc sycl/lib-internal.h 'UsmPtr<T> mom_usm_alloc(' 'inline uint64_t mom_now_ms('
cat "$out/usm_alloc.inc" >> "$out/usm_ptr.inc"
extract_cpp ulong_parser.inc sycl/lib-internal.h 'inline bool mom_parse_env_ulong('

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
    "$root/tests/native/zelhash-layout.cpp" -o "$out/$profile"
  printf '%s: ' "$profile"
  "$out/$profile"
  "$cxx" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic "${flags[@]}" -I "$out" \
    "$root/tests/native/zelhash-state.cpp" -o "$out/$profile-state"
  printf '%s-state: ' "$profile"
  "$out/$profile-state"
done
