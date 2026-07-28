#!/usr/bin/env bash
# Package every policy-selected Linux worker from the one multicompiler image. Each worker and its
# runtime live in an isolated directory so the incompatible oneAPI/nightly/AdaptiveCpp libraries
# never share a process. compiler-policy.js chooses the directory before spawning each worker.
set -euo pipefail

version="${1:-}"
if [ -z "$version" ] && [[ "${GITHUB_REF_NAME:-}" =~ ^v?[0-9] ]]; then version="$GITHUB_REF_NAME"; fi
[ -n "$version" ] || version="$(node -p "require('./package.json').version")"
version="${version#v}"
root="mom-v${version}"
archive="${2:-mom-v${version}-lin.tgz}"
package_dir="release-combined/$root"
libs_dir="$package_dir/libs"
build_dir=release-combined-build
node_bin="${NODE_BIN:-}"
image="${MOM_MULTICOMPILER_IMAGE:-mom-build-multicompiler}"
compiler_build_dir=build/lin/Release

compilers=(oneapi dpcpp dpcpp-opencl acpp-cuda acpp-hip)
for key in "${compilers[@]}"; do
  [ -s "$compiler_build_dir/$key/mom.node" ] || {
    echo "$compiler_build_dir/$key/mom.node is missing; run MOM_GPU_BACKEND=all ./r.sh true." >&2
    exit 1
  }
done
docker image inspect "$image" >/dev/null

rm -rf release-combined "$build_dir" "$archive"
mkdir -p "$package_dir" "$libs_dir" "$build_dir"

# SEA blobs must be produced by the exact Node executable that receives them. Distribution Node
# binaries can omit the SEA fuse even when their version nominally supports SEA, so fall back to
# the tested Node 24 executable already carried by the unified build image.
if [ -z "$node_bin" ]; then node_bin="$(command -v node)"; fi
if ! LC_ALL=C grep -aq 'NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2' "$node_bin"; then
  node_container="mom-multicompiler-node-$$"
  docker rm -f "$node_container" >/dev/null 2>&1 || true
  docker create --name "$node_container" --entrypoint /bin/true "$image" >/dev/null
  docker cp "$node_container:/usr/local/bin/node" "$build_dir/node"
  docker rm "$node_container" >/dev/null
  chmod +x "$build_dir/node"
  node_bin="$PWD/$build_dir/node"
fi

# Build the standalone Node executable payload. The Markdown policy remains an external, readable
# config and is copied beside the executable; esbuild intentionally leaves the fs read at runtime.
bundle_path="$PWD/$build_dir/mom.bundle.cjs"
blob_path="$PWD/$build_dir/mom.blob"
npx --no-install esbuild mom.js --bundle --platform=node --format=cjs \
  --banner:js="const { createRequire } = require('node:module'); require = createRequire(process.execPath);" \
  --outfile="$bundle_path"
cat >"$build_dir/sea-config.json" <<EOF
{"main":"$bundle_path","output":"$blob_path","disableExperimentalSEAWarning":true,"useCodeCache":false,"useSnapshot":false}
EOF
"$node_bin" --experimental-sea-config "$build_dir/sea-config.json"
cp "$node_bin" "$package_dir/mom-bin"
npx --no-install postject "$package_dir/mom-bin" NODE_SEA_BLOB "$blob_path" \
  --sentinel-fuse NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2
chmod +x "$package_dir/mom-bin"

cat >"$package_dir/mom" <<'EOF'
#!/usr/bin/env sh
set -eu
case "$0" in */*) script_dir=${0%/*} ;; *) script_dir=$(dirname "$(command -v "$0")") ;; esac
script_dir=$(CDPATH= cd -- "$script_dir" && pwd -P)
libs="$script_dir/libs"

# Auto-select only on single-vendor systems. On a mixed-vendor host the requested worker/device is
# ambiguous, so MOM_GPU_BACKEND=intel|nvidia|amd is intentionally explicit. `opencl` is the
# best-effort generic SPIR-V fallback for a driver not covered by those native vendor paths.
if [ -z "${MOM_GPU_BACKEND:-}" ]; then
  intel=0; nvidia=0; amd=0; opencl=0
  for vendor in /sys/bus/pci/devices/*/vendor; do
    [ -r "$vendor" ] || continue
    class_file="${vendor%/vendor}/class"
    [ -r "$class_file" ] || continue
    case "$(cat "$class_file")" in 0x03*) ;; *) continue ;; esac
    case "$(cat "$vendor")" in 0x8086) intel=1 ;; 0x10de) nvidia=1 ;; 0x1002) amd=1 ;; esac
    case "$(cat "$vendor")" in 0x8086|0x10de|0x1002) ;; *) opencl=1 ;; esac
  done
  count=$((intel+nvidia+amd+opencl))
  if [ "$count" -eq 1 ]; then
    [ "$intel" -eq 0 ] || MOM_GPU_BACKEND=intel
    [ "$nvidia" -eq 0 ] || MOM_GPU_BACKEND=nvidia
    [ "$amd" -eq 0 ] || MOM_GPU_BACKEND=amd
    [ "$opencl" -eq 0 ] || MOM_GPU_BACKEND=opencl
    export MOM_GPU_BACKEND
  fi
fi

# oneAPI remains the compatibility/default worker for legacy callers. Automatic unknown-vendor
# detection and explicit MOM_GPU_BACKEND=opencl select the open-source DPC++ SPIR-V/UR worker.
# Release CI selects the standards-only DPC++ OpenCL worker for its full SYCL CPU vector suite.
key=oneapi
if [ -n "${MOM_GPU_INDEX:-}" ]; then
  case "$MOM_GPU_INDEX" in *[!0-9]*) echo "MOM_GPU_INDEX must be a non-negative integer" >&2; exit 2 ;; esac
fi
selector_backend=
selector_default=
case "${MOM_GPU_BACKEND:-}" in
  intel)
    key=oneapi; selector_backend=level_zero; selector_default="level_zero:gpu"
    UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=${UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS:-1}
    export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS
    ;;
  nvidia) key=dpcpp; selector_backend=cuda; selector_default=cuda:* ;;
  amd)
    key=acpp-hip
    ACPP_VISIBILITY_MASK=${ACPP_VISIBILITY_MASK:-hip}
    export ACPP_VISIBILITY_MASK
    ;;
  opencl) key=dpcpp-opencl; selector_backend=opencl; selector_default=opencl:gpu ;;
esac
if [ -n "$selector_backend" ] && [ -z "${ONEAPI_DEVICE_SELECTOR:-}" ]; then
  if [ -n "${MOM_GPU_INDEX:-}" ]; then
    if [ "${MOM_GPU_BACKEND:-}" = intel ]; then
      ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
    elif [ "${MOM_GPU_BACKEND:-}" = opencl ]; then
      # Keep all ICDs visible; the addon applies the index after stable device-name sorting.
      ONEAPI_DEVICE_SELECTOR="opencl:gpu"
    else
      ONEAPI_DEVICE_SELECTOR="$selector_backend:$MOM_GPU_INDEX"
    fi
  else
    ONEAPI_DEVICE_SELECTOR="$selector_default"
  fi
fi
# Release CI uses this narrow override to select the packaged standards-only DPC++ OpenCL worker for
# its SYCL CPU gate. Normal miners never set it; the extracted archive can therefore validate an
# isolated runtime without changing the user-facing vendor policy or requiring a physical GPU.
if [ -n "${MOM_RELEASE_RUNTIME_KEY:-}" ]; then
  case "$MOM_RELEASE_RUNTIME_KEY" in
    oneapi|dpcpp|dpcpp-opencl|acpp-cuda|acpp-hip) key=$MOM_RELEASE_RUNTIME_KEY ;;
    *) echo "Unknown MOM_RELEASE_RUNTIME_KEY: $MOM_RELEASE_RUNTIME_KEY" >&2; exit 2 ;;
  esac
fi
export ONEAPI_DEVICE_SELECTOR
runtime="$libs/$key"
export MOM_NATIVE_DIR="$libs"
if [ -z "${MOM_NATIVE_PATH:-}" ]; then
  export MOM_NATIVE_PATH="$runtime/mom.node"
  export MOM_NATIVE_PATH_LAUNCHER_DEFAULT="$MOM_NATIVE_PATH"
fi
# install.sh exposes the NVIDIA source-JIT payload at this stable path. Development images already
# export CUDA_PATH, but a release host normally does not; ProgPoW uses its compiler/libdevice tools,
# while Pearl discovers NVRTC and CUDA/CCCL headers below the same root.
if [ "${MOM_GPU_BACKEND:-}" = nvidia ]; then
  if [ -z "${CUDA_PATH:-}" ] && [ -d /usr/local/cuda ]; then CUDA_PATH=/usr/local/cuda; fi
  if [ -n "${CUDA_PATH:-}" ]; then
    export CUDA_PATH
    if [ -d "$CUDA_PATH/bin" ]; then PATH="$CUDA_PATH/bin:$PATH"; export PATH; fi
  fi
fi
library_dirs="$runtime:$runtime/hipSYCL"
if [ "$key" = dpcpp-opencl ]; then library_dirs="$library_dirs:$libs/dpcpp"; fi
if [ -n "${LD_LIBRARY_PATH:-}" ]; then library_dirs="$library_dirs:$LD_LIBRARY_PATH"; fi
export LD_LIBRARY_PATH="$library_dirs"
if [ "${MOM_GPU_BACKEND:-}" != opencl ] && \
   [ -z "${OCL_ICD_FILENAMES:-}" ] && [ -f "$runtime/libintelocl.so" ]; then
  export OCL_ICD_FILENAMES="$runtime/libintelocl.so"
fi
exec "$script_dir/mom-bin" "$@"
EOF
chmod +x "$package_dir/mom"

cp package.json compiler-policy.js README.md DEVELOPMENT.md GPU-COMPILERS.md LICENSE "$package_dir/"
cp scripts/install.sh scripts/install-cutlass.sh "$package_dir/"
for key in "${compilers[@]}"; do
  mkdir -p "$libs_dir/$key"
  cp "$compiler_build_dir/$key/mom.node" "$libs_dir/$key/mom.node"
done
# Only the open DPC++ CUDA worker compiles the shared SYCL device body from source at runtime.
# The generic AdaptiveCpp workers consume their embedded SSCP IR instead, so duplicating this file
# in every runtime directory has no consumer.
cp sycl/kawpow/device.inc "$libs_dir/dpcpp/kawpow_device.inc"
cp sycl/kawpow/keccak.inc "$libs_dir/dpcpp/kawpow_keccak.inc"
# Backward-compatible fallback for callers that do not use the launcher/policy selection.
cp "$libs_dir/oneapi/mom.node" "$libs_dir/mom.node"

container="mom-multicompiler-package-$$"
docker rm -f "$container" >/dev/null 2>&1 || true
docker run -d --name "$container" --entrypoint sleep -v "$PWD:/repo:ro" "$image" infinity >/dev/null
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT

is_base_lib() {
  case "$1" in
    ld-linux*|libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|libresolv.so*|libutil.so*|\
    libgcc_s.so*|libstdc++.so*|libcuda.so*|libnvidia-ml.so*|libOpenCL.so*|libze_loader.so*|\
    libamdhip64.so*|libhiprtc.so*|libhsa-runtime64.so*|libhsakmt.so*|libdrm*.so*|libnuma.so*) return 0 ;;
  esac
  return 1
}

runtime_key=
runtime_roots=
runtime_dest=
declare -a closure_queue
declare -A closure_seen

copy_runtime_file() {
  local src="$1" name
  name="$(basename "$src")"
  is_base_lib "$name" && return 0
  if [ ! -e "$runtime_dest/$name" ]; then docker cp -L "$container:$src" "$runtime_dest/$name"; fi
  closure_queue+=("$src")
}

image_matches() {
  local expression="$1"
  docker exec "$container" bash -lc "for f in $expression; do [ -f \"\$f\" ] && [[ \"\$f\" != *-gdb.py ]] && printf '%s\\n' \"\$f\"; done"
}

copy_matches() {
  local expression="$1" src
  while IFS= read -r src; do [ -z "$src" ] || copy_runtime_file "$src"; done < <(image_matches "$expression")
}

copy_dependency_closure() {
  local index=0 current dep name output
  while [ "$index" -lt "${#closure_queue[@]}" ]; do
    current="${closure_queue[$index]}"; index=$((index+1))
    [ -z "${closure_seen[$current]:-}" ] || continue
    closure_seen[$current]=1
    output="$(docker exec "$container" bash -lc \
      "LD_LIBRARY_PATH='$runtime_roots' ldd '$current' 2>/dev/null || true")"
    while IFS= read -r dep; do
      [ -n "$dep" ] || continue
      name="$(basename "$dep")"
      is_base_lib "$name" && continue
      copy_runtime_file "$dep"
    done < <(awk '/=> \/[^ ]+/{print $3} /^\/[^( ]+/{print $1}' <<<"$output" | sort -u)
  done
}

begin_runtime() {
  runtime_key="$1"; runtime_roots="$2"; runtime_dest="$libs_dir/$runtime_key"
  closure_queue=("/repo/$compiler_build_dir/$runtime_key/mom.node")
  closure_seen=()
}

copy_oneapi_opencl_runtime() {
  copy_matches '/opt/intel/oneapi/compiler/latest/lib/libintelocl.so'
  # shellcheck disable=SC2016 # evaluated inside the build container by copy_matches
  copy_matches '$(find /opt/intel/oneapi/compiler/latest/lib /opt/intel/oneapi/tbb /opt/intel/oneapi/tcm /opt/intel/oneapi/umf -type f,l \( -name "libocl_svml_*.so" -o -name "libtbbmalloc.so.2" -o -name "libtcm.so.1" -o -name "libiomp5.so" \) 2>/dev/null)'
  local blob
  while IFS= read -r blob; do
    [ -z "$blob" ] || docker cp -L "$container:$blob" "$runtime_dest/$(basename "$blob")"
  done < <(image_matches '/opt/intel/oneapi/compiler/latest/lib/clbltfn*.rtl /opt/intel/oneapi/compiler/latest/lib/cllibrary*.rtl /opt/intel/oneapi/compiler/latest/lib/cllibrary*.o')
}

begin_runtime oneapi '/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/tbb/latest/lib:/opt/intel/oneapi/tcm/latest/lib:/opt/intel/oneapi/umf/latest/lib'
copy_matches '/opt/intel/oneapi/compiler/latest/lib/libsycl.so.9 /opt/intel/oneapi/compiler/latest/lib/libur_adapter_level_zero.so.0 /opt/intel/oneapi/compiler/latest/lib/libur_adapter_level_zero_v2.so.0 /opt/intel/oneapi/compiler/latest/lib/libur_adapter_opencl.so.0 /opt/intel/oneapi/umf/latest/lib/libumf.so.1'
copy_oneapi_opencl_runtime
copy_dependency_closure

begin_runtime dpcpp '/opt/dpcpp/lib:/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/tbb/latest/lib:/opt/intel/oneapi/tcm/latest/lib'
copy_matches '/opt/dpcpp/lib/libsycl.so.9 /opt/dpcpp/lib/libur_adapter_cuda.so.0 /opt/dpcpp/lib/libur_adapter_level_zero.so.0 /opt/dpcpp/lib/libur_adapter_level_zero_v2.so.0 /opt/dpcpp/lib/libumf.so.1 /opt/dpcpp/lib/libsycl-jit.so'
# This Linux configuration links the UR OpenCL adapter into libsycl; only Windows emits a separate
# ur_adapter_opencl.dll. The vendor's ICD and libOpenCL dispatcher remain host dependencies, so an
# unknown future vendor can supply its driver without any OpenCL-specific miner sources.
copy_dependency_closure

copy_acpp_runtime() {
  local key="$1" prefix="$2" so tool redist_bin llvm_root=/opt/llvm21-ubuntu24
  [ "$key" != acpp-hip ] || llvm_root=/opt/llvm20-ubuntu24
  begin_runtime "$key" "$prefix/lib:$prefix/lib/hipSYCL:/opt/ubuntu24-libs:$llvm_root/lib:/usr/lib/x86_64-linux-gnu"
  docker cp -L "$container:$prefix/lib/." "$runtime_dest/"
  rm -rf "$runtime_dest/cmake"
  # These workers are intentionally vendor-isolated and never select AdaptiveCpp's OpenCL
  # backend. Shipping its plugin makes a driver-only CUDA/HIP host probe libOpenCL at startup and
  # emit a misleading loader warning; the SPIR-V translator has no remaining consumer either.
  rm -f "$runtime_dest/hipSYCL/librt-backend-ocl.so" \
    "$runtime_dest/hipSYCL/llvm-to-backend/libllvm-to-spirv.so"
  # AdaptiveCpp's supported relocatable JIT layout. Without these, a package works in the dev image
  # (which has LLVM globally) but fails on a driver-only target when generic/SSCP first compiles.
  redist_bin="$runtime_dest/hipSYCL/ext/llvm/bin"
  mkdir -p "$redist_bin"
  # Preserve every compiled tool basename. In particular AdaptiveCpp's host translator searches for
  # `ld.lld`, not the multicall binary's `lld` name; omitting the symlink made the otherwise-complete
  # OpenMP JIT payload fall back to the build-only /usr/lib/llvm-21 path.
  for tool in opt llc lld ld.lld; do
    if docker exec "$container" test -x "$llvm_root/bin/$tool"; then
      docker cp -L "$container:$llvm_root/bin/$tool" "$redist_bin/$tool"
      closure_queue+=("$llvm_root/bin/$tool")
    fi
  done
  if [ "$key" = acpp-hip ]; then
    mkdir -p "$runtime_dest/hipSYCL/ext/bitcode/amdgcn"
    while IFS= read -r tool; do
      docker cp -L "$container:$tool" "$runtime_dest/hipSYCL/ext/bitcode/amdgcn/$(basename "$tool")"
    done < <(image_matches '/opt/rocm-device-libs-ubuntu24/*.bc')
  elif [ "$key" = acpp-cuda ]; then
    # Destination prescribed by AdaptiveCpp's deployment manifest. The runtime resolves this
    # relative to its own libraries, avoiding the build-image-only /usr/local/cuda path.
    mkdir -p "$runtime_dest/hipSYCL/ext/bitcode/ptx"
    docker cp -L "$container:/usr/local/cuda/nvvm/libdevice/libdevice.10.bc" \
      "$runtime_dest/hipSYCL/ext/bitcode/ptx/libdevice.10.bc"
  fi
  while IFS= read -r so; do closure_queue+=("$so"); done < <(
    docker exec "$container" find "$prefix/lib" -type f -name '*.so*' -print)
  copy_dependency_closure
}
copy_acpp_runtime acpp-cuda /opt/adaptivecpp-cuda
copy_acpp_runtime acpp-hip /opt/adaptivecpp-hip

# Keep the AdaptiveCpp runtime directories isolated for loader safety, but hard-link any
# byte-identical support files so tar stores one copy. The CUDA and HIP translators use LLVM 21 and
# LLVM 20 respectively; cmp naturally keeps those versioned JIT payloads separate.
while IFS= read -r -d '' file; do
  relative="${file#"$libs_dir/acpp-hip/"}"
  peer="$libs_dir/acpp-cuda/$relative"
  if [ -f "$peer" ] && cmp -s "$peer" "$file"; then
    rm "$file"
    ln "$peer" "$file"
  fi
done < <(find "$libs_dir/acpp-hip" -type f -print0)

# Preserve the old libs/mom.node entry point without duplicating archive data: its oneAPI runtime
# files are hard links to the isolated default directory. Policy-selected workers still prepend only
# their own compiler directory.
while IFS= read -r -d '' file; do
  name="$(basename "$file")"
  [ "$name" = mom.node ] || [ -e "$libs_dir/$name" ] || ln "$file" "$libs_dir/$name"
done < <(find "$libs_dir/oneapi" -maxdepth 1 -type f -print0)

# Validate every isolated closure. Driver/loader libraries listed in is_base_lib intentionally come
# from install.sh and are the only permitted unresolved dependencies on the packaging host.
failed=0
while IFS= read -r -d '' file; do
  relative="${file#"$libs_dir/"}"; key="${relative%%/*}"; compiler_root="$libs_dir/$key"
  dir="$(dirname "$file")"
  shared_root=
  [ "$key" != dpcpp-opencl ] || shared_root=":$libs_dir/dpcpp"
  if output="$(LD_LIBRARY_PATH="$compiler_root:$compiler_root/hipSYCL:$dir$shared_root" ldd "$file" 2>&1)"; then :; else true; fi
  unresolved="$(awk '/not found/{print $1}' <<<"$output" | while read -r lib; do is_base_lib "$lib" || echo "$lib"; done)"
  if [ -n "$unresolved" ]; then echo "$file: unresolved $unresolved" >&2; failed=1; fi
done < <(find "$libs_dir" -type f \( -name 'mom.node' -o -name '*.so' -o -name '*.so.*' \) -print0)
[ "$failed" -eq 0 ] || exit 1

tar -C release-combined -czf "$archive" "$root"
echo "$archive"
