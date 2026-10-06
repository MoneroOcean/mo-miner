#!/usr/bin/env bash
# Combined build orchestration (run inside scripts/build-combined.dockerfile, which
# has sourced oneAPI setvars and put the nightly DPC++ in /opt/dpcpp). Drives node-gyp with
# the dual-compiler wrapper (scripts/cxx-combined.sh): icx for host objects, nightly clang
# for the SYCL objects and the final -fsycl link.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DPCPP="${MOM_DPCPP_ROOT:-/opt/dpcpp}"
ICPX="${MOM_ICPX:-icpx}"
WRAP="$ROOT/scripts/cxx-combined.sh"
build_jobs="${MOM_BUILD_JOBS:-$(nproc)}"
build_log="$ROOT/build/combined-build-output.log"
verbose_build="${MOM_BUILD_VERBOSE:-0}"
MOM_BUILD_LOG="$build_log"
source "$ROOT/scripts/build-helpers.sh"

compiler_first_line() {
  "$@" --version 2>&1 | sed -n '1p'
}

# Dual-compiler builds link at the OBJECT level: icx host objects + clang -fsycl link. LTO is
# impossible across the two (icx LLVM bitcode != nightly clang bitcode, and the nightly tarball
# ships no LLVMgold.so/lld plugin for the link), so force LTO off. The icx CPU-codegen advantage
# is in instruction selection, not LTO (clang+LTO never closed the rx/0 gap), so this is cheap.
export MOM_LTO=0

# Build- and run-time both need the nightly libsycl ahead of oneAPI's same-soname lib.
export LD_LIBRARY_PATH="$DPCPP/lib:${LD_LIBRARY_PATH:-}"

# AOT device targets (Intel SPIR-V + NVIDIA PTX), consumed by both the sycl-TU compile and the link
# in cxx-combined.sh. ONE low NVIDIA arch (sm_80, Ampere) is the default, NOT a multi-arch list: the
# nightly clang's multi-nvptx-target fatbin is mis-selected at runtime (every algo CUDA_ERROR_NO_BINARY
# on an sm_89 device), whereas a single sm_80 image carries forward-compatible PTX that the driver
# JITs to the actual GPU at load -- so one sm_80 build runs natively on Ampere/Ada/Hopper (verified
# on an L4/sm_89: 7/7 algos, pearlhash 34.1 TH/s and autolykos2 76.8 MH/s, identical to a native sm_89
# build). pearlhash's int8 mma.m16n8k32 needs sm_80, so sm_80 is also the floor. Override to widen/narrow.
export MOM_COMBINED_TARGETS="${MOM_COMBINED_TARGETS:-spir64,spir64_gen,nvidia_gpu_sm_80}"
# Default the Intel device only when the selected target set contains an Intel AOT image. This keeps
# an explicit NVIDIA-only or portable-only MOM_COMBINED_TARGETS override valid without a second knob.
if [ -z "${MOM_INTEL_AOT_DEVICE:-}" ] && [[ ",$MOM_COMBINED_TARGETS," == *,spir64_gen,* ]]; then
  export MOM_INTEL_AOT_DEVICE=bmg-g21
fi
export MOM_DPCPP_IMPL="${MOM_DPCPP_IMPL:-dpcpp-combined}"

dpcpp_identity="$(compiler_first_line "$DPCPP/bin/clang++")"
icpx_identity="$(compiler_first_line "$ICPX")"

# Reconfigure (and wipe build/) only on a real change of mode/targets/node, NOT merely because a
# prior build failed -- so iterating on one source recompiles just that TU + relinks. The marker
# includes the target set so changing it forces a clean reconfigure. MOM_FORCE_REBUILD=1 forces it.
node_build_version="$(printf '%s\n' \
  "node=$(node -p process.version)" \
  "compiler=$MOM_DPCPP_IMPL" \
  "host=dual-icx-clang" \
  "targets=$MOM_COMBINED_TARGETS" \
  "intel_aot_device=${MOM_INTEL_AOT_DEVICE:-unset}" \
  "cpu=${MOM_CPU_MARCH:-unset}" \
  "portable=${MOM_PORTABLE_BUILD:-0}" \
  "dpcpp_root=$DPCPP" \
  "icpx=$ICPX" \
  "cmplr_root=${CMPLR_ROOT:-unset}" \
  "dpcpp_clang_id=$dpcpp_identity" \
  "icpx_id=$icpx_identity")"
if [ "${MOM_FORCE_REBUILD:-0}" = 1 ] \
   || [ "$(cat build/.node-version 2>/dev/null || true)" != "$node_build_version" ] \
   || [ binding.gyp -nt build/Makefile ] \
   || [ scripts/cpu-cflags.sh -nt build/Makefile ] \
   || [ scripts/cpu-optflags.sh -nt build/Makefile ] \
   || [ scripts/cpu-feature.sh -nt build/Makefile ] \
   || [ "$WRAP" -nt build/Makefile ] \
   || ! grep -q "/root/mom" build/Makefile 2>/dev/null; then
  rm -rf build
  mom_run_quiet "[combined] node-gyp configure (CXX=cxx-combined.sh, CC=icx, impl=$MOM_DPCPP_IMPL, targets=$MOM_COMBINED_TARGETS)" \
    env CC=icx CXX="$WRAP" LINK="$WRAP" node-gyp configure --nodedir=/usr/local -- -Dmom_sycl_impl="$MOM_DPCPP_IMPL"
  # Stamp the marker right after configure so a later failed build still skips the reconfigure.
  mkdir -p build && echo "$node_build_version" > build/.node-version
fi

export MOM_COMBINED_LOG="$ROOT/build/combined-routing.log"
: > "$MOM_COMBINED_LOG"
# node-gyp already relinks when an object or recorded command changes. A wrapper edit leaves the
# recorded command path unchanged, so invalidate only for that linker-only input. An unconditional
# relink takes several minutes because DPC++ must regenerate every CUDA image.
link_target="$ROOT/build/Release/obj.target/mom.node"
if [[ -s $link_target ]] && [[ $WRAP -nt $link_target ]]; then
  rm -f "$ROOT/build/Release/mom.node" "$link_target"
fi
mom_run_quiet "[combined] node-gyp build" \
  env JOBS="$build_jobs" CC=icx CXX="$WRAP" LINK="$WRAP" MAKEFLAGS="-s -j${build_jobs}" \
    node-gyp build --nodedir=/usr/local --jobs "$build_jobs" --silent

if [ "$verbose_build" = 1 ]; then
  echo "[combined] compiler routing (want: 7 SYCL->clang, rest HOST->icpx, 1 LINK->clang):"
  sort "$MOM_COMBINED_LOG" | uniq -c | sed 's/^/    /'
  echo "    --- SYCL TUs routed to clang: $(grep -c '^SYCL' "$MOM_COMBINED_LOG") ---"
fi

mkdir -p build && echo "$node_build_version" > build/.node-version

if [ "$verbose_build" = 1 ]; then
  echo "[combined] build OK:"
  ls -la build/Release/mom.node
  echo "[combined] mom.node SYCL/runtime linkage:"
  ldd build/Release/mom.node | grep -iE "sycl|svml|irc|imf|intlc|cuda|ze_|opencl" || true
fi
