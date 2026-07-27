#!/usr/bin/env bash
set -euo pipefail

backend=${MOM_GPU_BACKEND:-intel}
jobs=${MOM_BUILD_JOBS:-$(nproc)}
opencl_device_type=${MOM_OPENCL_DEVICE_TYPE:-gpu}
case "$opencl_device_type" in
  gpu|cpu) ;;
  *) echo "MOM_OPENCL_DEVICE_TYPE must be gpu or cpu" >&2; exit 2 ;;
esac
if [ "$backend" = intel ]; then
  export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS="${UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS:-1}"
fi
if [ -n "${MOM_GPU_INDEX:-}" ]; then
  case "$MOM_GPU_INDEX" in *[!0-9]*) echo "MOM_GPU_INDEX must be a non-negative integer" >&2; exit 2 ;; esac
  case "$backend" in
    # Device names are sorted by hardware name before the addon applies MOM_GPU_INDEX.
    intel) export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}" ;;
    nvidia) export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-cuda:$MOM_GPU_INDEX}" ;;
    amd) export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$MOM_GPU_INDEX}" ;;
    opencl) export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-opencl:$opencl_device_type}" ;;
  esac
fi
if [ "$backend" = intel ]; then
  export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
fi
if [ "$backend" = nvidia ]; then
  export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-cuda:*}"
elif [ "$backend" = amd ]; then
  export ACPP_VISIBILITY_MASK="${ACPP_VISIBILITY_MASK:-hip}"
fi
artifact_dir=build-compilers/Release
mkdir -p "$artifact_dir"
# Remove the pre-policy flat names left by older development images. Compiler workers and their
# runtimes now always live in isolated key/ directories on both operating systems.
rm -f "$artifact_dir"/mom-*.node
# DPC++ HIP is retained only as an explicitly requested reproducer toolchain; it is no longer a
# release-policy worker. Remove an artifact left by an older checkout before publishing build/lin.
rm -rf "$artifact_dir/dpcpp-hip"
set +u
# The disk-efficient multicompiler image inherits the combined oneAPI image, whose Docker build has
# already sourced setvars. Force a clean process-local refresh instead of treating that inherited
# marker as an error; the previous Ubuntu scratch final did not carry the marker.
. /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u

# node-gyp unconditionally uses a top-level build/ directory. Keep it as a scratch workspace while
# compiling, then restore the persistent platform tree and publish Linux output under build/lin.
# This prevents a Windows PE addon returned by win/run.sh from ever colliding with a Linux ELF addon.
platforms_hold=build-platforms-hold
if [ -e "$platforms_hold" ]; then
  echo "$platforms_hold exists from an interrupted build; refusing to overwrite it" >&2
  exit 1
fi
if [ -d build ]; then mv build "$platforms_hold"; fi
restore_platform_tree() {
  rm -rf build
  if [ -d "$platforms_hold" ]; then mv "$platforms_hold" build; else mkdir -p build; fi
}
trap restore_platform_tree EXIT
# The NVIDIA and generic OpenCL workers are built by the open-source DPC++ tree, not oneAPI. Keep
# its compiler/runtime first; NVIDIA also needs CUDA tools for KawPow's runtime SYCL-source JIT.
# Without this, libsycl-jit resolves its resource directory as /lib/clang/... and silently falls
# back to the roughly 3x slower AOT ProgPoW kernel. Packaged workers carry the same files beside the
# addon; this path setup is specifically for the consolidated development container used by r.sh.
if [ "$backend" = nvidia ] || [ "$backend" = opencl ]; then
  export PATH="/opt/dpcpp/bin:$PATH"
  export LD_LIBRARY_PATH="/opt/dpcpp/lib:${LD_LIBRARY_PATH:-}"
fi
if [ "$backend" = opencl ]; then
  # The oneAPI base image pins its private Intel ICD. Generic mode must use the system dispatcher so
  # every mounted vendor ICD is visible, including vendors unknown when this image was built.
  unset OCL_ICD_FILENAMES
fi
if [ "$backend" = nvidia ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

publish() {
  local source=$1 key=$2
  mkdir -p "$artifact_dir/$key"
  install -m 0755 "$source" "$artifact_dir/$key/mom.node"
}

build_oneapi() {
  # CPU flags are part of the artifact ABI. In particular, a cached developer build made with
  # -march=native must never be reused after MOM_PORTABLE_BUILD=1 is selected for packaging.
  local out=build-oneapi
  local marker="$(node -p process.version):oneapi-2026:cpu=${MOM_CPU_MARCH:-unset}:portable=${MOM_PORTABLE_BUILD:-0}"
  if [ "$(cat "$out/.node-version" 2>/dev/null || true)" != "$marker" ] ||
     [ ! -s "$out/Release/mom.node" ] ||
     find binding.gyp native sycl xmrig -type f -newer "$out/Release/mom.node" -print -quit | grep -q .; then
    rm -rf build "$out"
    CC=icx CXX=icpx node-gyp configure --nodedir=/usr/local -- -Dmom_sycl_impl=dpcpp
    JOBS="$jobs" CC=icx CXX=icpx node-gyp build --nodedir=/usr/local --jobs "$jobs"
    mv build "$out"; printf '%s\n' "$marker" >"$out/.node-version"
  fi
  publish "$out/Release/mom.node" oneapi
}

build_dpcpp() {
  rm -rf build
  [ ! -d build-dpcpp ] || mv build-dpcpp build
  MOM_COMBINED_TARGETS=${MOM_COMBINED_TARGETS:-spir64,nvidia_gpu_sm_80} bash scripts/combined-build.sh
  rm -rf build-dpcpp; mv build build-dpcpp
  publish build-dpcpp/Release/mom.node dpcpp
}

build_dpcpp_opencl() {
  rm -rf build
  [ ! -d build-dpcpp-opencl ] || mv build-dpcpp-opencl build
  MOM_DPCPP_IMPL=dpcpp-opencl MOM_COMBINED_TARGETS=spir64 bash scripts/combined-build.sh
  rm -rf build-dpcpp-opencl; mv build build-dpcpp-opencl
  publish build-dpcpp-opencl/Release/mom.node dpcpp-opencl
}

build_acpp() {
  local target=$1 path=$2 out=$3 key=$4
  PATH="$path/bin:$PATH" LD_LIBRARY_PATH="$path/lib:$LD_LIBRARY_PATH" \
    ACPP_TARGETS=generic ACPP_VISIBILITY_MASK="$target" MOM_ADAPTIVE_BACKEND="$target" \
    MOM_ADAPTIVE_BUILD_DIR="$out" bash scripts/adaptivecpp-entrypoint.sh true
  publish "$out/Release/mom.node" "$key"
}

case "$backend" in
  intel) build_oneapi; build_dpcpp_opencl; default=oneapi ;;
  nvidia)
    build_dpcpp
    build_dpcpp_opencl
    build_acpp cuda /opt/adaptivecpp-cuda build-acpp-cuda acpp-cuda
    default=dpcpp ;;
  amd)
    build_dpcpp_opencl
    build_acpp hip /opt/adaptivecpp-hip build-acpp-hip acpp-hip
    default=acpp-hip ;;
  opencl)
    build_dpcpp_opencl
    default=dpcpp-opencl
    export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-opencl:$opencl_device_type}" ;;
  all)
    build_oneapi; build_dpcpp; build_dpcpp_opencl
    build_acpp cuda /opt/adaptivecpp-cuda build-acpp-cuda acpp-cuda
    build_acpp hip /opt/adaptivecpp-hip build-acpp-hip acpp-hip
    default=oneapi ;;
  *) echo "Unsupported MOM_GPU_BACKEND=$backend" >&2; exit 2 ;;
esac

restore_platform_tree
trap - EXIT
rm -rf build/lin
mkdir -p build/lin/Release
cp -a "$artifact_dir/." build/lin/Release/
cp "build/lin/Release/$default/mom.node" build/lin/Release/mom.node
# The container runs as root so it can reach MSRs and GPU device nodes, but build/ is shared with
# Windows run.sh. Return both the platform tree and its parent to the checkout owner; otherwise a
# first Linux build leaves a root-owned build/ directory that prevents run.sh from creating
# build/win beside it.
chown --reference="$PWD" build
chown -R --reference="$PWD" build/lin
export MOM_NATIVE_DIR="$PWD/build/lin/Release"
export MOM_NATIVE_PATH="$MOM_NATIVE_DIR/$default/mom.node"
export MOM_NATIVE_PATH_LAUNCHER_DEFAULT="$MOM_NATIVE_PATH"
# A fresh checkout has no host node_modules. Keep `./r.sh npm test` self-contained without making
# ordinary miner/build commands pay for a package-registry check.
if [ "${1:-}" = npm ] && [ ! -x node_modules/.bin/eslint ]; then
  npm install --no-package-lock --ignore-scripts
fi
exec "$@"
