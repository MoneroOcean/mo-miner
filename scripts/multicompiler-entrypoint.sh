#!/usr/bin/env bash
set -euo pipefail

backend=${MOM_GPU_BACKEND:-intel}
build_compiler=${MOM_LINUX_BUILD_COMPILER:-all}
jobs=${MOM_BUILD_JOBS:-$(nproc)}
opencl_device_type=${MOM_OPENCL_DEVICE_TYPE:-gpu}
case "$build_compiler" in
  all|oneapi|dpcpp|dpcpp-opencl|acpp-cuda|acpp-hip) ;;
  *) echo "MOM_LINUX_BUILD_COMPILER must be all, oneapi, dpcpp, dpcpp-opencl, acpp-cuda, or acpp-hip" >&2; exit 2 ;;
esac
case "$backend:$build_compiler" in
  intel:all|intel:oneapi|intel:dpcpp-opencl|nvidia:all|nvidia:dpcpp|\
  nvidia:dpcpp-opencl|nvidia:acpp-cuda|amd:all|amd:dpcpp-opencl|amd:acpp-hip|\
  opencl:all|opencl:dpcpp-opencl|all:*) ;;
  *) echo "MOM_LINUX_BUILD_COMPILER=$build_compiler is incompatible with MOM_GPU_BACKEND=$backend" >&2; exit 2 ;;
esac
selected() {
  [ "$build_compiler" = all ] || [ "$build_compiler" = "$1" ]
}
build_compilers=()
case "$backend" in
  intel)
    selected oneapi && build_compilers+=(oneapi)
    selected dpcpp-opencl && build_compilers+=(dpcpp-opencl)
    default=${build_compiler/all/oneapi} ;;
  nvidia)
    selected dpcpp && build_compilers+=(dpcpp)
    selected dpcpp-opencl && build_compilers+=(dpcpp-opencl)
    selected acpp-cuda && build_compilers+=(acpp-cuda)
    default=${build_compiler/all/dpcpp} ;;
  amd)
    selected dpcpp-opencl && build_compilers+=(dpcpp-opencl)
    selected acpp-hip && build_compilers+=(acpp-hip)
    default=${build_compiler/all/acpp-hip} ;;
  opencl)
    build_compilers+=(dpcpp-opencl)
    default=dpcpp-opencl ;;
  all)
    selected oneapi && build_compilers+=(oneapi)
    selected dpcpp && build_compilers+=(dpcpp)
    selected dpcpp-opencl && build_compilers+=(dpcpp-opencl)
    selected acpp-cuda && build_compilers+=(acpp-cuda)
    selected acpp-hip && build_compilers+=(acpp-hip)
    default=${build_compiler/all/oneapi} ;;
  *) echo "Unsupported MOM_GPU_BACKEND=$backend" >&2; exit 2 ;;
esac
if [ "${MOM_LINUX_BUILD_PLAN_ONLY:-0}" = 1 ]; then
  echo "workers=${build_compilers[*]}"
  echo "default=$default"
  exit
fi
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
set +u
# The disk-efficient multicompiler image inherits the combined oneAPI image, whose Docker build has
# already sourced setvars. Force a clean process-local refresh instead of treating that inherited
# marker as an error; the previous Ubuntu scratch final did not carry the marker.
. /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u
compiler_first_line() {
  "$@" --version 2>&1 | sed -n '1p'
}

source_fingerprint() {
  local key=$1
  local -a inputs=(
    binding.gyp
    native
    sycl
    xmrig
    scripts/multicompiler-entrypoint.sh
    scripts/cpu-cflags.sh
    scripts/cpu-optflags.sh
    scripts/cpu-feature.sh
  )
  case "$key" in
    dpcpp|dpcpp-opencl)
      inputs+=(scripts/combined-build.sh scripts/cxx-combined.sh)
      ;;
    acpp-cuda|acpp-hip)
      inputs+=(scripts/adaptivecpp-entrypoint.sh scripts/cxx-adaptivecpp.sh)
      ;;
  esac
  find "${inputs[@]}" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum |
    awk '{print $1}'
}

# Concurrent Linux GPU measurements reuse an immutable worker produced by an earlier ordinary
# r.sh invocation. Fail closed if that worker is absent or stale and, most importantly, do not
# rename or otherwise mutate the shared build tree in this mode.
case "${MOM_REUSE_BUILT_WORKER:-0}" in
  0) ;;
  1)
    if [ "$backend" = nvidia ] || [ "$backend" = opencl ]; then
      export PATH="/opt/dpcpp/bin:$PATH"
      export LD_LIBRARY_PATH="/opt/dpcpp/lib:${LD_LIBRARY_PATH:-}"
    fi
    if [ "$backend" = opencl ]; then
      unset OCL_ICD_FILENAMES
      export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-opencl:$opencl_device_type}"
    fi
    if [ "$backend" = nvidia ]; then
      export PATH="/usr/local/cuda/bin:$PATH"
      export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    fi
    reused_worker="$PWD/build/lin/Release/$default/mom.node"
    if [ ! -s "$reused_worker" ]; then
      echo "Reusable $default worker is missing: $reused_worker" >&2
      exit 2
    fi
    fingerprint_file="$reused_worker.sources.sha256"
    recorded_fingerprint=$(cat "$fingerprint_file" 2>/dev/null || true)
    current_fingerprint=$(source_fingerprint "$default")
    if [[ ! $recorded_fingerprint =~ ^[0-9a-f]{64}$ ]] ||
       [ "$recorded_fingerprint" != "$current_fingerprint" ]; then
      echo "Reusable $default worker is stale; source fingerprint differs" >&2
      echo "Run the matching backend once without MOM_REUSE_BUILT_WORKER before retrying" >&2
      exit 2
    fi
    export MOM_NATIVE_DIR="$PWD/build/lin/Release"
    export MOM_NATIVE_PATH="$reused_worker"
    export MOM_NATIVE_PATH_LAUNCHER_DEFAULT="$MOM_NATIVE_PATH"
    if [ "${1:-}" = npm ] && ! npm ls --depth=0 --silent >/dev/null 2>&1; then
      echo "Concurrent worker reuse requires existing node_modules; run an ordinary npm task first" >&2
      exit 2
    fi
    exec "$@"
    ;;
  *)
    echo "MOM_REUSE_BUILT_WORKER must be 0 or 1" >&2
    exit 2
    ;;
esac

# node-gyp unconditionally uses a top-level build/ directory. Keep it as a scratch workspace while
# compiling. The resting tree keeps platform outputs and every reusable compiler cache together
# under build/; temporarily parking that tree prevents a Windows PE addon returned by win/run.sh
# from ever colliding with a Linux ELF addon.
platforms_hold=build-platforms-hold
if [[ -e "$platforms_hold" || -L "$platforms_hold" ]]; then
  echo "$platforms_hold exists from an interrupted build; refusing to overwrite it" >&2
  exit 1
fi
if [[ -e build || -L build ]] && [[ ! -d build || -L build ]]; then
  echo "The repository build root must be a real directory: $PWD/build" >&2
  exit 2
fi
if [ -d build ]; then mv build "$platforms_hold"; else mkdir -p "$platforms_hold"; fi
cache_root="$platforms_hold/cache"
artifact_dir="$platforms_hold/lin/Release"
active_cache=
mkdir -p "$cache_root" "$artifact_dir"
# Remove the pre-policy flat names left by older development images. Compiler workers and their
# runtimes now always live in isolated key/ directories on both operating systems.
rm -f "$artifact_dir"/mom-*.node
restore_platform_tree() {
  # Preserve a partially rebuilt object tree on ordinary command failure or Ctrl-C. It remains
  # eligible for the same marker/dependency checks on the next run instead of losing all progress.
  if [ -n "$active_cache" ] && [ -d build ]; then
    rm -rf "$active_cache"
    mkdir -p "$(dirname "$active_cache")"
    mv build "$active_cache"
  else
    rm -rf build
  fi
  active_cache=
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
  local source=$1 key=$2 destination binary_sha
  mkdir -p "$artifact_dir/$key"
  destination="$artifact_dir/$key/mom.node"
  install -m 0755 "$source" "$destination"
  source_fingerprint "$key" >"$destination.sources.sha256"
  binary_sha="$(sha256sum "$destination" | awk '{print $1}')"
  printf '%s\n' \
    'schema=1' \
    "worker=$key" \
    "sha256=$binary_sha" \
    "portable=${MOM_PORTABLE_BUILD:-0}" \
    "cpu=${MOM_CPU_MARCH:-unset}" >"$destination.build-profile"
}

build_oneapi() {
  # CPU flags are part of the artifact ABI. In particular, a cached developer build made with
  # -march=native must never be reused after MOM_PORTABLE_BUILD=1 is selected for packaging.
  local out="$cache_root/oneapi"
  local marker
  marker="$(printf '%s\n' \
    "node=$(node -p process.version)" \
    "compiler=oneapi-2026" \
    "cpu=${MOM_CPU_MARCH:-unset}" \
    "portable=${MOM_PORTABLE_BUILD:-0}" \
    "lto=${MOM_LTO:-auto}" \
    "icx_id=$(compiler_first_line icx)" \
    "icpx_id=$(compiler_first_line icpx)")"
  active_cache="$out"
  rm -rf build
  [ ! -d "$out" ] || mv "$out" build
  # Keep node-gyp's dependency/object tree across ordinary source iterations. Make already knows
  # which translation units changed; a clean configure is needed only when the compiler/CPU mode or
  # generated build graph changes. The previous source-mtime gate discarded every object whenever
  # one SYCL file changed, turning a targeted worker build into a full project rebuild.
  if [ "$(cat build/.node-version 2>/dev/null || true)" != "$marker" ] ||
     [ ! -s build/Makefile ] || [ binding.gyp -nt build/Makefile ] ||
     [ scripts/cpu-cflags.sh -nt build/Makefile ] ||
     [ scripts/cpu-optflags.sh -nt build/Makefile ] ||
     [ scripts/cpu-feature.sh -nt build/Makefile ]; then
    rm -rf build
    CC=icx CXX=icpx node-gyp configure --nodedir=/usr/local -- -Dmom_sycl_impl=dpcpp
    printf '%s\n' "$marker" >build/.node-version
  fi
  JOBS="$jobs" CC=icx CXX=icpx node-gyp build --nodedir=/usr/local --jobs "$jobs"
  rm -rf "$out"
  mv build "$out"
  active_cache=
  publish "$out/Release/mom.node" oneapi
}

build_dpcpp() {
  local out="$cache_root/dpcpp"
  active_cache="$out"
  rm -rf build
  [ ! -d "$out" ] || mv "$out" build
  bash scripts/combined-build.sh
  rm -rf "$out"
  mv build "$out"
  active_cache=
  publish "$out/Release/mom.node" dpcpp
}

build_dpcpp_opencl() {
  local out="$cache_root/dpcpp-opencl"
  active_cache="$out"
  rm -rf build
  [ ! -d "$out" ] || mv "$out" build
  MOM_DPCPP_IMPL=dpcpp-opencl MOM_COMBINED_TARGETS=spir64 MOM_INTEL_AOT_DEVICE= \
    bash scripts/combined-build.sh
  rm -rf "$out"
  mv build "$out"
  active_cache=
  publish "$out/Release/mom.node" dpcpp-opencl
}

build_acpp() {
  local target=$1 path=$2 out=$3 key=$4
  PATH="$path/bin:$PATH" LD_LIBRARY_PATH="$path/lib:$LD_LIBRARY_PATH" \
    ACPP_TARGETS=generic ACPP_VISIBILITY_MASK="$target" MOM_ADAPTIVE_BACKEND="$target" \
    MOM_ADAPTIVE_BUILD_DIR="$out" bash scripts/adaptivecpp-entrypoint.sh true
  publish "$out/Release/mom.node" "$key"
}

for compiler in "${build_compilers[@]}"; do
  case "$compiler" in
    oneapi) build_oneapi ;;
    dpcpp) build_dpcpp ;;
    dpcpp-opencl) build_dpcpp_opencl ;;
    acpp-cuda) build_acpp cuda /opt/adaptivecpp-cuda "$cache_root/acpp-cuda" acpp-cuda ;;
    acpp-hip) build_acpp hip /opt/adaptivecpp-hip "$cache_root/acpp-hip" acpp-hip ;;
  esac
done
if [ "$backend" = opencl ]; then
  export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-opencl:$opencl_device_type}"
fi

cp "$artifact_dir/$default/mom.node" "$artifact_dir/mom.node"
cp "$artifact_dir/$default/mom.node.sources.sha256" "$artifact_dir/mom.node.sources.sha256"
cp "$artifact_dir/$default/mom.node.build-profile" "$artifact_dir/mom.node.build-profile"
# The constrained container keeps root only for build-tree ownership and selected GPU device nodes,
# but build/ is shared with Windows run.sh. Return both the platform tree and its parent to the
# checkout owner; otherwise a first Linux build leaves a root-owned build/ directory that prevents
# run.sh from creating build/win beside it.
chown -R --reference="$PWD" "$platforms_hold/lin"
restore_platform_tree
trap - EXIT
chown --reference="$PWD" build
export MOM_NATIVE_DIR="$PWD/build/lin/Release"
export MOM_NATIVE_PATH="$MOM_NATIVE_DIR/$default/mom.node"
export MOM_NATIVE_PATH_LAUNCHER_DEFAULT="$MOM_NATIVE_PATH"
# A fresh checkout has no host node_modules. Keep `./r.sh npm test` self-contained without making
# ordinary miner/build commands pay for a package-registry check.
if [ "${1:-}" = npm ] && ! npm ls --depth=0 --silent >/dev/null 2>&1; then
  npm install --ignore-scripts --no-audit --no-fund
fi
exec "$@"
