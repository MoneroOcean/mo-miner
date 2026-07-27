#!/usr/bin/env bash
set -euo pipefail

host_root=$PWD
build_jobs="${MOM_BUILD_JOBS:-$(nproc)}"
compiler="$host_root/scripts/cxx-adaptivecpp.sh"
host_cc=${MOM_ADAPTIVE_HOST_CC:-clang-21}
backend=${MOM_ADAPTIVE_BACKEND:-hip}
case "$backend" in
  hip|cuda) ;;
  *) echo "Unsupported MOM_ADAPTIVE_BACKEND=$backend" >&2; exit 2 ;;
esac
chmod +x "$compiler"
build_dir=${MOM_ADAPTIVE_BUILD_DIR:-build-adaptive-$backend}
MOM_BUILD_LOG="/tmp/mom-adaptive-build/build-output.log"
source "$host_root/scripts/build-helpers.sh"
# Include host CPU policy so portable packages cannot inherit cached -march=native objects.
marker="$(node -p process.version):adaptivecpp-$backend:${ACPP_TARGETS:-}:cpu=${MOM_CPU_MARCH:-unset}:portable=${MOM_PORTABLE_BUILD:-0}"
if [ "$(cat "$build_dir/.node-version" 2>/dev/null || true)" != "$marker" ] ||
   [ ! -s "$build_dir/Release/mom.node" ] ||
   find binding.gyp native sycl xmrig scripts/cpu-cflags.sh scripts/cxx-adaptivecpp.sh -type f \
     -newer "$build_dir/Release/mom.node" -print -quit | grep -q .; then
  rm -rf /tmp/mom-adaptive-build "$build_dir"
  mkdir -p /tmp/mom-adaptive-build
  cp -a . /tmp/mom-adaptive-build/source
  cd /tmp/mom-adaptive-build/source
  rm -rf build build-amd build-adaptive-hip build-adaptive-cuda
  mom_run_quiet "[adaptivecpp-$backend] node-gyp configure" env \
    CC="$host_cc" CXX="$compiler" LINK="$compiler" node-gyp configure --nodedir=/usr/local \
    -- -Dmom_sycl_impl="adaptivecpp-$backend"
  mom_run_quiet "[adaptivecpp-$backend] node-gyp build" env JOBS="$build_jobs" \
    CC="$host_cc" CXX="$compiler" LINK="$compiler" \
    node-gyp build --nodedir=/usr/local --jobs "$build_jobs"
  mkdir -p "$host_root/$build_dir/Release"
  cp build/Release/mom.node "$host_root/$build_dir/Release/mom.node"
  printf '%s\n' "$marker" >"$host_root/$build_dir/.node-version"
  chown -R --reference="$host_root" "$host_root/$build_dir"
  cd "$host_root"
  rm -rf /tmp/mom-adaptive-build
fi
export MOM_NATIVE_PATH="$host_root/$build_dir/Release/mom.node"
exec "$@"
