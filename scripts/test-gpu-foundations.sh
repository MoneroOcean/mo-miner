#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/mom-gpu-foundations.XXXXXX")
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

build_run() {
  local name=$1
  shift
  "$cxx" "${flags[@]}" "$@" -o "$out/$name"
  "$out/$name"
}
# Compile the production flushes without importing SYCL or duplicating ownership.
awk '
  /^          auto sg = it.get_sub_group\(\);$/ { copies++; copying = 1 }
  copying && /^        \}\);$/ { copying = 0; complete++ }
  copying { print }
  END { if (copies != 1 || complete != 1 || copying) exit 1 }
' "$root/sycl/beamhash3/compact_solver.inc" > "$out/beam_compact_flush.inc"
awk '
  /^            auto sg = it.get_sub_group\(\);$/ { copies++; copying = 1 }
  copying && /^          \}$/ { copying = 0; complete++ }
  copying { print }
  END { if (copies != 1 || complete != 1 || copying) exit 1 }
' "$root/sycl/beamhash3/generic_solver.inc" > "$out/beam_generic_flush.inc"
test -s "$out/beam_compact_flush.inc"
test -s "$out/beam_generic_flush.inc"
build_run beam-scatter -UNDEBUG -I "$out" "$root/tests/native/beam_scatter.cpp"

awk '
  /^    if \(new_dag_words > / { copies++; copying = 1 }
  copying { print }
  copying && /throw std::string\("Etchash DAG exceeds 32-bit word addressing"\);$/ {
    copying = 0; complete++
  }
  END { if (copies != 1 || complete != 1 || copying) exit 1 }
' "$root/sycl/etchash/state.inc" > "$out/etchash_dag_limit.inc"
build_run etchash-word-limit -UNDEBUG -I "$out" "$root/tests/native/etchash_word_limit.cpp"

build_run cn-gpu-launch "$root/tests/native/cn_gpu_launch.cpp"
build_run cn-gpu-keccak -std=c++20 "$root/tests/native/cn_gpu_keccak.cpp"
build_run cn-gpu-keccak-cuda -std=c++20 -D__NVPTX__ "$root/tests/native/cn_gpu_keccak.cpp"
build_run cn-gpu-keccak-opencl -std=c++20 -DMOM_SYCL_PORTABLE_OPENCL \
  "$root/tests/native/cn_gpu_keccak.cpp"
build_run cn-gpu-reductions -std=c++20 "$root/tests/native/cn_gpu_reductions.cpp"
build_run jit-cache -pthread "$root/tests/native/jit_cache.cpp"
build_run device-state -pthread "$root/tests/native/device_state.cpp"
build_run pow-intensity "$root/tests/native/pow_intensity.cpp"
build_run c29-cycle "$root/tests/native/c29_cycle.cpp"
build_run c29-recovery-range "$root/tests/native/c29_recovery_range.cpp"
build_run job-boundary -I "$root/native" "$root/tests/native/job_boundary.cpp"
build_run hashrate-sampling "$root/tests/native/hashrate_sampling.cpp"

awk '
  /^    if \(new_dag_words > / { copies++; copying = 1 }
  copying { print }
  copying && /throw std::string\("KawPow DAG exceeds 32-bit word addressing"\);$/ {
    copying = 0; complete++
  }
  END { if (copies != 1 || complete != 1 || copying) exit 1 }
' "$root/sycl/kawpow/state.inc" > "$out/progpow_dag_limit.inc"
build_run progpow-epoch -I "$out" "$root/tests/native/progpow_epoch.cpp"

build_run pearlhash-seed "$root/tests/reference/pearlhash_seed.cpp"

bash "$root/scripts/test-intel-geometry.sh"
bash "$root/scripts/test-zelhash-layout.sh"

build_run zhash-core "$root/sycl/zhash/zhash_cpu_test.cpp"
build_run zhash-core-hip -DMOM_SYCL_HAS_HIP "$root/sycl/zhash/zhash_cpu_test.cpp"
build_run zhash-session "$root/sycl/zhash/zhash_session_test.cpp"
build_run zhash-session-hip -DMOM_SYCL_HAS_HIP "$root/sycl/zhash/zhash_session_test.cpp"
build_run zhash-session-intel -D__INTEL_LLVM_COMPILER -DMOM_ZHASH_INTEL_LATE_BUCKETS \
  "$root/sycl/zhash/zhash_session_test.cpp"
build_run equihash192-core "$root/sycl/equihash192_7/equihash192_7_cpu_test.cpp"
build_run equihash192-core-intel -D__INTEL_LLVM_COMPILER \
  "$root/sycl/equihash192_7/equihash192_7_cpu_test.cpp"
build_run c30-host -I "$root/xmrig" -I "$root/sycl/c30" \
  "$root/sycl/c30/c30_host.cpp" \
  "$root/sycl/c30/c30_host_test.cpp" \
  "$root/xmrig/crypto/randomx/blake2/blake2b.c" \
  "$root/xmrig/base/crypto/keccak.cpp"
build_run nvidia-features "$root/tests/native/nvidia_features.cpp"
build_run nvidia-dot "$root/tests/native/nvidia_dot.cpp"
build_run amd-features "$root/tests/native/amd_features.cpp"
build_run workgroup-limits "$root/tests/native/workgroup_limits.cpp"
build_run pearlhash-esimd-route "$root/tests/native/pearlhash_esimd_route.cpp"
build_run octopus-validation "$root/tests/native/octopus_validation.cpp"
build_run nexapow-recorded -DMOM_NEXAPOW_HOST_TEST "$root/sycl/nexapow/nexapow.cpp"
build_run nexapow-fixed-base "$root/sycl/nexapow/test_fixed_base.cpp"
build_run cpu-scheduling -std=c++20 "$root/tests/native/cpu_scheduling.cpp"
build_run verthash-data -std=c++20 -pthread "$root/tests/native/verthash_data.cpp"

echo 'GPU foundation host tests passed'
