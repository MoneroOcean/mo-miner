#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/mom-host.XXXXXX")
trap 'rm -rf -- "$out"' EXIT
cxx=${CXX:-c++}
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic)

compiler_plan() {
  MOM_GPU_BACKEND=$1 MOM_LINUX_BUILD_COMPILER=$2 MOM_LINUX_BUILD_PLAN_ONLY=1 \
    bash "$root/scripts/multicompiler-entrypoint.sh"
}

assert_compiler_plan() {
  local actual
  actual="$(compiler_plan "$1" "$2")"
  test "$actual" = "$3"
}

assert_compiler_plan intel all $'workers=oneapi dpcpp-opencl\ndefault=oneapi'
assert_compiler_plan intel_igpu all $'workers=dpcpp-opencl\ndefault=dpcpp-opencl'
assert_compiler_plan INTEL_IGPU all $'workers=dpcpp-opencl\ndefault=dpcpp-opencl'
assert_compiler_plan intel_igpu dpcpp-opencl $'workers=dpcpp-opencl\ndefault=dpcpp-opencl'
assert_compiler_plan nvidia all $'workers=dpcpp dpcpp-opencl acpp-cuda\ndefault=dpcpp'
assert_compiler_plan amd all $'workers=dpcpp-opencl acpp-hip\ndefault=acpp-hip'
assert_compiler_plan opencl all $'workers=dpcpp-opencl\ndefault=dpcpp-opencl'
assert_compiler_plan all all \
  $'workers=oneapi dpcpp dpcpp-opencl acpp-cuda acpp-hip\ndefault=oneapi'
assert_compiler_plan nvidia acpp-cuda $'workers=acpp-cuda\ndefault=acpp-cuda'
assert_compiler_plan amd acpp-hip $'workers=acpp-hip\ndefault=acpp-hip'
if compiler_plan intel acpp-cuda >/dev/null 2>&1; then
  echo 'incompatible Linux compiler selector unexpectedly succeeded' >&2
  exit 1
fi
if compiler_plan intel_igpu oneapi >/dev/null 2>&1; then
  echo 'Intel integrated backend unexpectedly admitted the discrete worker' >&2
  exit 1
fi

build_run() {
  local name=$1
  shift
  "$cxx" "${flags[@]}" "$@" -o "$out/$name"
  "$out/$name"
}

# Print whole production declarations through the shared token-anchored extractor; a missing or
# ambiguous anchor fails the run. Indented anchors name nested members or statements exactly.
extract_cpp() {
  local name=$1 file=$2
  shift 2
  node "$root/tests/common/cpp-source.js" "$root/$file" "$@" > "$out/$name"
}

# Compile the production flush loops without importing SYCL or duplicating ownership.
extract_cpp beam_compact_flush.inc sycl/beamhash3/compact-solver.inc \
  '          for (unsigned r = subgroup; r < lsz; r += subgroups) {'
extract_cpp beam_generic_flush.inc sycl/beamhash3/generic-solver.inc \
  '            for (unsigned r = warp_id; r < lsz; r += wg_warps) {'
build_run beam-scatter -UNDEBUG -I "$out" "$root/tests/native/beam-scatter.cpp"
extract_cpp beam_workgroup_helpers.inc sycl/lib-internal.h 'inline bool mom_parse_env_ulong(' \
  'inline bool sycl_local_memory_fits(' 'inline bool sycl_workgroup_fits(' \
  'inline unsigned sycl_default_workgroup('
extract_cpp beamhash3_workgroup.inc sycl/beamhash3/layout.inc 'inline bool beam_compact_enabled(' \
  'unsigned beamhash3_workgroup('
build_run beamhash3-admission -UNDEBUG -I "$out" "$root/tests/native/beamhash3-admission.cpp"
extract_cpp sycl_env.inc sycl/lib-internal.h 'inline void mom_set_compile_options_env('
build_run sycl-env -UNDEBUG -pthread -I "$out" -I "$root/native" "$root/tests/native/sycl-env.cpp" \
  -Wl,--wrap=setenv

extract_cpp ethash_dag_words.inc sycl/ethash-dag.inc 'inline bool ethash_dag_words_addressable('
build_run ethash-dag-words -UNDEBUG -I "$out" "$root/tests/native/ethash-dag-words.cpp"
extract_cpp etchash_resolve_epochs.inc sycl/etchash/state.inc '  void resolve_epochs('
build_run etchash-epoch -UNDEBUG -I "$out" "$root/tests/native/etchash-epoch.cpp"

build_run cn-gpu-launch "$root/tests/native/cn-gpu-launch.cpp"
build_run cn-gpu-keccak -std=c++20 "$root/tests/native/cn-gpu-keccak.cpp"
build_run cn-gpu-keccak-cuda -std=c++20 -D__NVPTX__ "$root/tests/native/cn-gpu-keccak.cpp"
build_run cn-gpu-keccak-opencl -std=c++20 -DMOM_SYCL_PORTABLE_OPENCL \
  "$root/tests/native/cn-gpu-keccak.cpp"
build_run cn-gpu-reductions -std=c++20 "$root/tests/native/cn-gpu-reductions.cpp"
build_run jit-cache -pthread "$root/tests/native/jit-cache.cpp"
build_run device-state -pthread "$root/tests/native/device-state.cpp"
build_run pow-intensity "$root/tests/native/pow-intensity.cpp"
build_run c29-cycle "$root/tests/native/c29-cycle.cpp"
build_run c29-recovery-range "$root/tests/native/c29-recovery-range.cpp"
build_run job-boundary -I "$root/native" "$root/tests/native/job-boundary.cpp"
build_run hashrate-sampling "$root/tests/native/hashrate-sampling.cpp"
build_run walahash-fixed-acpp -std=c++20 -Wno-unknown-pragmas -Wno-unused-function \
  "$root/tests/native/walahash-fixed.cpp"
build_run walahash-fixed-dpcpp -std=c++20 -Wno-unknown-pragmas -Wno-unused-function \
  -DTEST_WALAHASH_DPCPP "$root/tests/native/walahash-fixed.cpp"

build_run progpow-epoch "$root/tests/native/progpow-epoch.cpp"
extract_cpp kawpow_bundle_key.inc sycl/kawpow/state.inc '  static bool bundle_key_eq(' \
  '  void ensure_period_bundle('
build_run kawpow-bundle-key -UNDEBUG -pthread -I "$out" "$root/tests/native/kawpow-bundle-key.cpp"

build_run pearlhash-seed "$root/tests/reference/pearlhash-seed.cpp"
build_run pearlhash-jackpot "$root/tests/native/pearlhash-jackpot.cpp"
build_run pearlhash-plain-proof "$root/tests/native/pearlhash-plain-proof.cpp"

bash "$root/scripts/test-intel-geometry.sh"
bash "$root/scripts/test-zelhash-layout.sh"

build_run zhash-core "$root/tests/native/zhash-core.cpp"
build_run zhash-core-hip -DMOM_SYCL_HAS_HIP "$root/tests/native/zhash-core.cpp"
build_run zhash-session "$root/tests/native/zhash-session.cpp"
build_run zhash-session-hip -DMOM_SYCL_HAS_HIP "$root/tests/native/zhash-session.cpp"
build_run zhash-session-intel -D__INTEL_LLVM_COMPILER -DMOM_ZHASH_INTEL_LATE_BUCKETS \
  "$root/tests/native/zhash-session.cpp"
extract_cpp local_memory_fits.inc sycl/lib-internal.h 'inline bool sycl_local_memory_fits('
extract_cpp equihash192-7-layout.inc sycl/equihash192-7/layout.h 'struct BucketLayout {'
printf ';\n' >> "$out/equihash192-7-layout.inc"
# The production slot count is a preprocessor-selected constant, so copy its whole conditional.
awk '
  /^#if/ { group = ""; grouping = 1 }
  grouping { group = group $0 "\n" }
  /^#endif/ && grouping {
    grouping = 0
    if (group ~ /constexpr unsigned ProductionSlotsPerBucket =/) { printf "%s", group; found++ }
  }
  END { if (found != 1) exit 1 }
' "$root/sycl/equihash192-7/layout.h" >> "$out/equihash192-7-layout.inc"
extract_cpp equihash192-7-admission.inc sycl/equihash192-7/equihash192-7.cpp \
  'unsigned equihash192_7_intensity('
extract_cpp zhash_admission.inc sycl/zhash/zhash.cpp 'unsigned zhash_intensity('
build_run equihash-admission -UNDEBUG -I "$out" "$root/tests/native/equihash-admission.cpp"
build_run equihash-admission-hip -UNDEBUG -DMOM_SYCL_HAS_HIP -I "$out" \
  "$root/tests/native/equihash-admission.cpp"
build_run equihash-admission-intel -UNDEBUG -D__INTEL_LLVM_COMPILER -DMOM_ZHASH_INTEL_LATE_BUCKETS \
  -I "$out" "$root/tests/native/equihash-admission.cpp"
build_run equihash-admission-intel-unbranded -UNDEBUG -DMOM_SYCL_INTEL_LAYOUT \
  -DMOM_ZHASH_INTEL_LATE_BUCKETS -I "$out" "$root/tests/native/equihash-admission.cpp"
build_run equihash192-7-core "$root/tests/native/equihash192-7-core.cpp"
build_run equihash192-7-core-intel -D__INTEL_LLVM_COMPILER \
  "$root/tests/native/equihash192-7-core.cpp"
build_run equihash192-7-core-intel-unbranded -DMOM_SYCL_INTEL_LAYOUT \
  "$root/tests/native/equihash192-7-core.cpp"
build_run zhash-session-intel-unbranded -DMOM_SYCL_INTEL_LAYOUT -DMOM_ZHASH_INTEL_LATE_BUCKETS \
  "$root/tests/native/zhash-session.cpp"
build_run equihash192-7-core-spir -D__SYCL_DEVICE_ONLY__ -D__SPIR__ \
  "$root/tests/native/equihash192-7-core.cpp"
build_run zhash-core-spir -D__SYCL_DEVICE_ONLY__ -D__SPIR__ \
  "$root/tests/native/zhash-core.cpp"
build_run c30-host -I "$root/xmrig" -I "$root/sycl/c30" \
  "$root/sycl/c30/host.cpp" \
  "$root/tests/native/c30-host.cpp" \
  "$root/xmrig/crypto/randomx/blake2/blake2b.c" \
  "$root/xmrig/base/crypto/keccak.cpp"
build_run nvidia-features "$root/tests/native/nvidia-features.cpp"
build_run nvidia-dot "$root/tests/native/nvidia-dot.cpp"
build_run amd-features "$root/tests/native/amd-features.cpp"
build_run workgroup-limits "$root/tests/native/workgroup-limits.cpp"
build_run pearlhash-esimd-route "$root/tests/native/pearlhash-esimd-route.cpp"
build_run pearlhash-f16-matrix "$root/tests/native/pearlhash-f16-matrix.cpp"
build_run octopus-validation "$root/tests/native/octopus-validation.cpp"
build_run nexapow-recorded "$root/tests/native/nexapow-recorded.cpp"
build_run nexapow-fixed-base "$root/tests/native/nexapow-fixed-base.cpp"
build_run cpu-scheduling -std=c++20 "$root/tests/native/cpu-scheduling.cpp"
build_run verthash-data -std=c++20 -pthread "$root/tests/native/verthash-data.cpp"

# Consensus references complement the production helper/session regressions above.
build_run nexapow-reference "$root/tests/reference/nexapow-echelon.cpp"
build_run octopus-reference "$root/tests/reference/octopus-conflux.cpp"
build_run c30-reference -I "$root/xmrig" "$root/tests/reference/c30-cortex.cpp" \
  "$root/xmrig/crypto/randomx/blake2/blake2b.c" "$root/xmrig/base/crypto/keccak.cpp"

# Preserve each former standalone target's compiler policy, including the operation
# differential build and its separate standards-only implementation variant.
flags=(-std=c++17 -O2 -Wall -Wextra -Werror)
build_run zhash-reference "$root/tests/reference/zhash-144-5.cpp"
build_run equihash192-7-reference "$root/tests/reference/equihash192-7-zclassic.cpp"
flags=(-std=c++20 -O2)
build_run xelishashv3-operations "$root/tests/logic/xelishashv3-operations.cpp"
build_run xelishashv3-operations-portable -DMOM_SYCL_PORTABLE_OPENCL \
  "$root/tests/logic/xelishashv3-operations.cpp"

echo 'Host correctness tests passed'
