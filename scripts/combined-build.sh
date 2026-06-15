#!/usr/bin/env bash
# Option C combined build orchestration (run inside scripts/build-combined.dockerfile, which
# has sourced oneAPI setvars and put the nightly DPC++ in /opt/dpcpp). Drives node-gyp with
# the dual-compiler wrapper (scripts/cxx-combined.sh): icx for host objects, nightly clang
# for the SYCL objects and the final -fsycl link.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DPCPP="${MOM_DPCPP_ROOT:-/opt/dpcpp}"
WRAP="$ROOT/scripts/cxx-combined.sh"
chmod +x "$WRAP"

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
# on an L4/sm_89: 7/7 algos, pearl 34.1 TH/s and autolykos2 76.8 MH/s, identical to a native sm_89
# build). pearl's int8 mma.m16n8k32 needs sm_80, so sm_80 is also the floor. Override to widen/narrow.
export MOM_COMBINED_TARGETS="${MOM_COMBINED_TARGETS:-spir64,nvidia_gpu_sm_80}"

# Reconfigure (and wipe build/) only on a real change of mode/targets/node, NOT merely because a
# prior build failed -- so iterating on one source recompiles just that TU + relinks. The marker
# includes the target set so changing it forces a clean reconfigure. MOM_FORCE_REBUILD=1 forces it.
node_build_version="$(node -p 'process.version'):combined:dual-icx-clang:${MOM_COMBINED_TARGETS}"
if [ "${MOM_FORCE_REBUILD:-0}" = 1 ] \
   || [ "$(cat build/.node-version 2>/dev/null || true)" != "$node_build_version" ] \
   || [ binding.gyp -nt build/Makefile ] \
   || ! grep -q "/root/mom" build/Makefile 2>/dev/null; then
  rm -rf build
  echo "[combined] node-gyp configure (CXX=cxx-combined.sh, CC=icx, impl=dpcpp-combined, targets=$MOM_COMBINED_TARGETS)"
  CC=icx CXX="$WRAP" LINK="$WRAP" node-gyp configure --nodedir=/usr/local -- -Dmom_sycl_impl=dpcpp-combined
  # Stamp the marker right after configure so a later failed build still skips the reconfigure.
  mkdir -p build && echo "$node_build_version" > build/.node-version
fi

echo "[combined] node-gyp build"
export MOM_COMBINED_LOG="$ROOT/build/combined-routing.log"
: > "$MOM_COMBINED_LOG"
# Force a relink each run (cheap; keeps compiled objects) so wrapper/link tweaks take effect
# without a full reconfigure+recompile.
rm -f build/Release/mom.node build/Release/obj.target/mom.node
JOBS="$(nproc)" CC=icx CXX="$WRAP" LINK="$WRAP" MAKEFLAGS=-s node-gyp build --nodedir=/usr/local --silent

echo "[combined] compiler routing (want: 7 SYCL->clang, rest HOST->icpx, 1 LINK->clang):"
sort "$MOM_COMBINED_LOG" | uniq -c | sed 's/^/    /'
echo "    --- SYCL TUs routed to clang: $(grep -c '^SYCL' "$MOM_COMBINED_LOG") ---"

mkdir -p build && echo "$node_build_version" > build/.node-version

echo "[combined] build OK:"
ls -la build/Release/mom.node
echo "[combined] mom.node SYCL/runtime linkage:"
ldd build/Release/mom.node | grep -iE "sycl|svml|irc|imf|intlc|cuda|ze_|opencl" || true
