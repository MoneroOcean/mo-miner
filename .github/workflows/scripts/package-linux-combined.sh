#!/usr/bin/env bash
# Package every policy-selected Linux worker from the prepared host toolchain. Each worker and its
# runtime live in an isolated directory so incompatible oneAPI/nightly/AdaptiveCpp libraries never
# share a process. compiler-policy.js chooses the directory before spawning each worker.
set -euo pipefail

version="${1:-}"
if [ -z "$version" ] && [ "${GITHUB_REF_TYPE:-}" = tag ] && [[ "${GITHUB_REF_NAME:-}" =~ ^v?[0-9] ]]; then version="$GITHUB_REF_NAME"; fi
[ -n "$version" ] || version="$(node -p "require('./package.json').version")"
version="${version#[vV]}"
if [[ ! "$version" =~ ^[0-9][0-9A-Za-z.-]*$ ]]; then
  echo "Invalid release version: $version" >&2
  exit 2
fi
root="mom-v${version}"
archive="${2:-mom-v${version}-lin.tgz}"
package_dir="release-combined/$root"
libs_dir="$package_dir/libs"
build_dir=release-combined-build
node_bin="${NODE_BIN:-}"
compiler_build_dir=build/lin/Release
archive_stage=

cleanup_package() {
  local status=$?
  if [ -n "$archive_stage" ]; then
    rm -rf -- "$archive_stage" || true
  fi
  return "$status"
}

compilers=(oneapi dpcpp dpcpp-opencl acpp-cuda acpp-hip)

profile_value() {
  local file=$1 field=$2
  awk -F= -v field="$field" '
    $1 == field { count++; value = substr($0, index($0, "=") + 1) }
    END { if (count == 1) print value; else exit 1 }
  ' "$file"
}

read_profile() {
  local file=$1
  schema="$(profile_value "$file" schema)" &&
    profile_worker="$(profile_value "$file" worker)" &&
    expected_sha="$(profile_value "$file" sha256)" &&
    portable="$(profile_value "$file" portable)" &&
    cpu="$(profile_value "$file" cpu)"
}

archive_path="$(realpath -m -- "$archive")"
for cleanup_root in release-combined "$build_dir"; do
  cleanup_path="$(realpath -m -- "$cleanup_root")"
  case "$archive_path" in
    "$cleanup_path"|"$cleanup_path"/*)
      echo "Archive path must not be inside cleanup directory: $archive" >&2
      exit 2
      ;;
  esac
done

if [ -d "$archive" ]; then
  echo "Archive path must not be a directory: $archive" >&2
  exit 1
fi

for key in "${compilers[@]}"; do
  worker="$compiler_build_dir/$key/mom.node"
  profile="$worker.build-profile"
  [ -s "$worker" ] || {
    echo "$worker is missing; run npm run build:release with the native SDKs installed." >&2
    exit 1
  }
  [ -s "$profile" ] || {
    echo "$profile is missing; rebuild the portable Linux workers." >&2
    exit 1
  }
  read_profile "$profile" || {
    echo "$profile has missing or duplicate fields." >&2
    exit 1
  }
  actual_sha="$(sha256sum "$worker" | awk '{print $1}')"
  if [ "$schema" != 1 ] || [ "$profile_worker" != "$key" ] ||
     [[ ! $expected_sha =~ ^[0-9a-f]{64}$ ]] || [ "$actual_sha" != "$expected_sha" ]; then
    echo "$profile does not identify $worker exactly; rebuild the portable Linux workers." >&2
    exit 1
  fi
  if [ "$portable" != 1 ] || [ "$cpu" != unset ]; then
    echo "$worker is not a portable release worker (portable=$portable, cpu=$cpu)." >&2
    echo "Run npm run build:release with the native SDKs installed before packaging." >&2
    exit 1
  fi
done
# Validate the host Node before any destructive staging cleanup. The packager never obtains a
# fallback binary from Docker; CI supplies the same official SEA-capable toolchain through scripts/dev-container.sh.
if [ -z "$node_bin" ]; then node_bin="$(command -v node || true)"; fi
if [ -z "$node_bin" ] || [ ! -x "$node_bin" ]; then
  echo "Native Linux packaging requires an executable official Node binary with SEA support" >&2
  exit 1
fi
if ! LC_ALL=C grep -aq 'NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2' "$node_bin"; then
  echo "Native Linux packaging requires an official Node binary with the SEA fuse; install the pinned Node toolchain" >&2
  exit 1
fi

# Reject invalid inputs first, but require the packaging tools before changing any output.
for tool in patchelf readelf; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "Linux release packaging requires $tool (install patchelf and binutils)." >&2
    exit 1
  }
done

trap cleanup_package EXIT
rm -rf release-combined "$build_dir"
mkdir -p "$package_dir" "$libs_dir" "$build_dir"

# Build the standalone Node executable payload. The Markdown policy remains an external, readable
# config and is copied beside the executable; esbuild intentionally leaves the fs read at runtime.
bundle_path="$PWD/$build_dir/mom.bundle.cjs"
blob_path="$PWD/$build_dir/mom.blob"
npx --no-install esbuild mom.js --bundle --platform=node --format=cjs \
  --banner:js="const { createRequire } = require('node:module'); require = createRequire(process.execPath);" \
  --outfile="$bundle_path"
"$node_bin" - "$bundle_path" "$blob_path" >"$build_dir/sea-config.json" <<'NODE_SEA_CONFIG'
const [main, output] = process.argv.slice(2);
process.stdout.write(JSON.stringify({main, output, disableExperimentalSEAWarning: true,
  useCodeCache: false, useSnapshot: false}) + "\n");
NODE_SEA_CONFIG
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
MOM_COMMAND=${MOM_COMMAND:-./mom}
export MOM_COMMAND
case "${MOM_GPU_BACKEND:-}" in
  [iI][nN][tT][eE][lL]) export MOM_GPU_BACKEND=intel ;;
  [iI][nN][tT][eE][lL]_[iI][gG][pP][uU]) export MOM_GPU_BACKEND=intel_igpu ;;
esac

# Auto-select only on single-vendor systems. On a mixed-vendor host the requested worker/device is
# ambiguous, so MOM_GPU_BACKEND=intel|intel_igpu|nvidia|amd|opencl is intentionally explicit. `opencl` is the
# best-effort generic SPIR-V fallback for a driver not covered by those native vendor paths.
if [ -z "${MOM_GPU_BACKEND:-}" ]; then
  intel=0
  nvidia=0
  amd=0
  opencl=0
  for vendor in /sys/bus/pci/devices/*/vendor; do
    [ -r "$vendor" ] || continue
    class_file="${vendor%/vendor}/class"
    [ -r "$class_file" ] || continue
    case "$(cat "$class_file")" in 0x03*) ;; *) continue ;; esac
    # QEMU's Cirrus console is display-only, not a competing compute backend.
    device_file="${vendor%/vendor}/device"
    case "$(cat "$vendor"):$(cat "$device_file" 2>/dev/null)" in
      0x1013:0x00b8) continue ;;
    esac
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
  MOM_GPU_INDEX=$(awk 'BEGIN {
    value = ENVIRON["MOM_GPU_INDEX"] + 0
    if (value > 1023) exit 2
    printf "%.0f", value
  }') || {
    echo "MOM_GPU_INDEX must be between 0 and 1023" >&2
    exit 2
  }
  export MOM_GPU_INDEX
fi

check_index_mask() {
  case "$2" in
    ''|-1|*,*) usable=0 ;;
    *) usable=1 ;;
  esac
  if [ "$MOM_GPU_INDEX" -ne 0 ] || [ "$usable" -eq 0 ]; then
    echo "MOM_GPU_INDEX cannot safely select within inherited $1; use a single-device visibility mask with MOM_GPU_INDEX=0, or omit MOM_GPU_INDEX and select a reported --job.dev" >&2
    exit 2
  fi
}
# Isolation uses the runtime's logical index, never a physical nvidia-smi row. Preserve inherited
# masks verbatim; lists can contain UUIDs or invalid tokens that truncate the visible device set.
if [ -n "${MOM_GPU_INDEX:-}" ]; then
  case "${MOM_GPU_BACKEND:-}" in
    nvidia)
      if [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
        check_index_mask CUDA_VISIBLE_DEVICES "$CUDA_VISIBLE_DEVICES"
      else
        CUDA_VISIBLE_DEVICES=$MOM_GPU_INDEX
        export CUDA_VISIBLE_DEVICES
      fi
      MOM_GPU_INDEX=0
      export MOM_GPU_INDEX
      ;;
    amd)
      if [ "${HIP_VISIBLE_DEVICES+x}" = x ]; then
        check_index_mask HIP_VISIBLE_DEVICES "$HIP_VISIBLE_DEVICES"
      fi
      if [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
        check_index_mask CUDA_VISIBLE_DEVICES "$CUDA_VISIBLE_DEVICES"
      fi
      if [ "${HIP_VISIBLE_DEVICES+x}" != x ] && [ "${CUDA_VISIBLE_DEVICES+x}" != x ]; then
        HIP_VISIBLE_DEVICES=$MOM_GPU_INDEX
        export HIP_VISIBLE_DEVICES
      fi
      MOM_GPU_INDEX=0
      export MOM_GPU_INDEX
      ;;
  esac
fi
opencl_device_type=${MOM_OPENCL_DEVICE_TYPE:-gpu}
case "$opencl_device_type" in
  gpu|cpu) ;;
  *) echo "MOM_OPENCL_DEVICE_TYPE must be gpu or cpu" >&2; exit 2 ;;
esac
selector_backend=
selector_default=
case "${MOM_GPU_BACKEND:-}" in
  intel)
    key=oneapi; selector_backend=level_zero; selector_default="level_zero:gpu"
    UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=${UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS:-1}
    export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS
    ;;
  intel_igpu)
    key=dpcpp-opencl; selector_backend=level_zero; selector_default="level_zero:gpu"
    ;;
  nvidia)
    capabilities=
    if [ -z "${MOM_NVIDIA_COMPUTE_CAPABILITY:-}" ]; then
      # The sentinel preserves trailing blank rows; a failed query must discard even valid stdout.
      capabilities=$(timeout 2 nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null && printf '.') || capabilities=
      capabilities=${capabilities%.}
    fi
    # Bootstrap runs before the JS policy can load a worker. Match its SM formats and unknown rules.
    sm=$(printf '%s' "$capabilities" | awk '
      function parse(value, parts, major, result) {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        if (value !~ /^[1-9][0-9]*(\.[0-9])?$/) return 0
        split(value, parts, ".")
        major = parts[1] + 0
        result = length(parts[2]) || major < 10 ? major * 10 + (parts[2] + 0) : major
        return result > 0 && result <= 9007199254740991 ? result : 0
      }
      BEGIN {
        capability = ENVIRON["MOM_NVIDIA_COMPUTE_CAPABILITY"]
        if (capability != "") { printf "%.0f\n", parse(capability); exit }
      }
      {
        value = parse($0)
        if (!value) unknown = 1
        else if (!minimum || value < minimum) minimum = value
      }
      END {
        if (capability == "") printf "%.0f\n", unknown ? 0 : minimum + 0
      }
    ')
    if [ -n "${MOM_NVIDIA_COMPUTE_CAPABILITY:-}" ] && [ "$sm" -eq 0 ]; then
      echo "Invalid MOM_NVIDIA_COMPUTE_CAPABILITY" >&2
      exit 2
    fi
    if [ "$sm" -lt 80 ]; then
      key=acpp-cuda; ACPP_VISIBILITY_MASK=cuda; export ACPP_VISIBILITY_MASK
    else
      key=dpcpp; selector_backend=cuda; selector_default=cuda:*
    fi
    ;;
  amd)
    key=acpp-hip
    ACPP_VISIBILITY_MASK=${ACPP_VISIBILITY_MASK:-hip}
    export ACPP_VISIBILITY_MASK
    if [ -z "${ROCM_PATH:-}" ] && [ -z "${HIP_PATH:-}" ] &&
       ! command -v hipconfig >/dev/null 2>&1 && [ -d /opt/rocm ]; then
      ROCM_PATH=/opt/rocm
      export ROCM_PATH
    fi
    ;;
  opencl)
    key=dpcpp-opencl
    selector_backend=opencl
    selector_default="opencl:$opencl_device_type"
    ;;
esac
if [ -n "$selector_backend" ] && [ -z "${ONEAPI_DEVICE_SELECTOR:-}" ]; then
  if [ -n "${MOM_GPU_INDEX:-}" ]; then
    if [ "${MOM_GPU_BACKEND:-}" = intel ] || [ "${MOM_GPU_BACKEND:-}" = intel_igpu ]; then
      ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
    elif [ "${MOM_GPU_BACKEND:-}" = opencl ]; then
      # Keep all matching ICD devices visible; the addon applies the index after stable sorting.
      ONEAPI_DEVICE_SELECTOR="$selector_default"
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
export MOM_NATIVE_DIR="$libs"
if [ -z "${MOM_NATIVE_PATH:-}" ]; then
  runtime="$libs/$key"
  export MOM_NATIVE_PATH="$runtime/mom.node"
  export MOM_NATIVE_PATH_LAUNCHER_DEFAULT="$MOM_NATIVE_PATH"
else
  # An explicit addon may belong to a different compiler cohort than the vendor default. Keep
  # that addon's runtime first so same-SONAME SYCL libraries cannot cross-load (for example,
  # dpcpp/mom.node with oneapi/libsycl.so.9).
  runtime=${MOM_RUNTIME_DIR:-}
  if [ -z "$runtime" ]; then
    case "$MOM_NATIVE_PATH" in
      */*) runtime=${MOM_NATIVE_PATH%/*} ;;
      *) runtime=. ;;
    esac
  fi
fi
# install.sh exposes the NVIDIA source-JIT payload at this stable path. The prepared toolchain
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
if [ "$runtime" = "$libs/dpcpp-opencl" ]; then
  library_dirs="$library_dirs:$libs/dpcpp"
fi
if [ -n "${LD_LIBRARY_PATH:-}" ]; then library_dirs="$library_dirs:$LD_LIBRARY_PATH"; fi
export LD_LIBRARY_PATH="$library_dirs"
if [ "${MOM_GPU_BACKEND:-}" != opencl ] && \
   [ -z "${OCL_ICD_FILENAMES:-}" ] && [ -f "$runtime/libintelocl.so" ]; then
  export OCL_ICD_FILENAMES="$runtime/libintelocl.so"
fi
exec "$script_dir/mom-bin" "$@"
EOF
chmod +x "$package_dir/mom"

cp package.json compiler-policy.js gpu-tuning.js README.md GPU-CONFIG.md LICENSE "$package_dir/"
mkdir -p "$package_dir/helper"
cp helper/hash.js "$package_dir/helper/"
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
for key in "${compilers[@]}"; do
  [ -s "$compiler_build_dir/$key/mom.node" ] || {
    echo "Native Linux packaging cannot read $compiler_build_dir/$key/mom.node" >&2
    exit 1
  }
done

is_base_lib() {
  case "$1" in
    ld-linux*|libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|libresolv.so*|libutil.so*|\
    libgcc_s.so*|libstdc++.so*|libcuda.so*|libnvidia-ml.so*|libOpenCL.so*|libze_loader.so*|\
    libamdhip64.so*|libhiprtc.so*|libamd_comgr.so*|libhsa-runtime64.so*|libhsakmt.so*|\
    libdrm*.so*|libnuma.so*) return 0 ;;
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
  if [ ! -e "$runtime_dest/$name" ]; then cp -L "$src" "$runtime_dest/$name"; fi
  closure_queue+=("$src")
}
runtime_matches() (
  # Expand trusted whitespace-separated file patterns, never evaluate shell commands.
  shopt -s nullglob
  local file
  for file in $1; do [ ! -f "$file" ] || printf '%s\n' "$file"; done
)
copy_matches() {
  local expression="$1" src
  while IFS= read -r src; do [ -z "$src" ] || copy_runtime_file "$src"; done < <(runtime_matches "$expression")
}

copy_dependency_closure() {
  local index=0 current dep name output
  while [ "$index" -lt "${#closure_queue[@]}" ]; do
    current="${closure_queue[$index]}"; index=$((index+1))
    [ -z "${closure_seen[$current]+x}" ] || continue
    closure_seen[$current]=1
    output="$(LD_LIBRARY_PATH="$runtime_roots" ldd "$current" 2>/dev/null || true)"
    while IFS= read -r dep; do
      [ -n "$dep" ] || continue
      name="$(basename "$dep")"
      is_base_lib "$name" && continue
      copy_runtime_file "$dep"
    done < <(awk '/=> \/[^ ]+/{print $3} /^\/[^ (]+/{print $1}' <<<"$output" | sort -u)
  done
}
begin_runtime() {
  runtime_key="$1"; runtime_roots="$2"; runtime_dest="$libs_dir/$runtime_key"
  closure_queue=("$compiler_build_dir/$runtime_key/mom.node")
  closure_seen=()
}
normalize_runtime_elfs() {
  local compiler_root file dir dynamic relative origin rpath old_rpath entry resolved
  compiler_root="$(realpath -e -- "$1")"
  while IFS= read -r -d '' file; do
    [ "$(LC_ALL=C head -c 4 -- "$file")" = $'\177ELF' ] || continue
    dynamic="$(LC_ALL=C readelf -d -- "$file")"
    [[ "$dynamic" == *'Dynamic section at offset'* ]] || continue
    if grep -Eq '\(NEEDED\).*\[[^]]*/' <<<"$dynamic"; then
      echo "$file: path-qualified DT_NEEDED bypasses runtime isolation" >&2
      return 1
    fi
    dir="$(dirname -- "$file")"
    relative="$(realpath --relative-to="$dir" -- "$compiler_root")"
    origin='$ORIGIN'
    [ "$relative" = . ] || origin+="/$relative"
    rpath='$ORIGIN'
    [ "$relative" = . ] || rpath+=":$origin"
    rpath+=":$origin/hipSYCL"
    [ "$(basename -- "$compiler_root")" != dpcpp-opencl ] || rpath+=":$origin/../dpcpp"
    # Keep nested plugin paths only inside this compiler closure. Absolute SDK paths and
    # empty components (the current directory) must never compete with packaged libraries.
    old_rpath="$(patchelf --print-rpath "$file")"
    while IFS= read -r entry; do
      case "$entry" in
        '$ORIGIN'|'$ORIGIN/'*) resolved="$dir${entry#'$ORIGIN'}" ;;
        '${ORIGIN}'|'${ORIGIN}/'*) resolved="$dir${entry#'${ORIGIN}'}" ;;
        *) continue ;;
      esac
      resolved="$(realpath -m -- "$resolved")"
      case "$resolved" in
        "$compiler_root"|"$compiler_root"/*)
          case ":$rpath:" in
            *":$entry:"*) ;;
            *) rpath+=":$entry" ;;
          esac
          ;;
      esac
    done < <(printf '%s\n' "$old_rpath" | tr : '\n')
    # RUNPATH honors the launcher's selected runtime prefix. Each dynamic dependency gets
    # its own relative closure because RUNPATH, unlike legacy RPATH, is not transitive.
    patchelf --set-rpath "$rpath" "$file"
  done < <(find "$compiler_root" -type f -print0)
}

copy_oneapi_opencl_runtime() {
  copy_matches '/opt/intel/oneapi/compiler/latest/lib/libintelocl.so'
  local blob
  # These libraries are loaded indirectly, so ldd alone does not discover them.
  while IFS= read -r -d '' blob; do
    copy_runtime_file "$blob"
  done < <(find /opt/intel/oneapi/compiler/latest/lib /opt/intel/oneapi/tbb \
    /opt/intel/oneapi/tcm /opt/intel/oneapi/umf -type f,l \
    \( -name 'libocl_svml_*.so' -o -name 'libtbbmalloc.so.2' -o -name 'libtcm.so.1' \
       -o -name 'libiomp5.so' \) -print0 2>/dev/null)
  while IFS= read -r blob; do
    [ -z "$blob" ] || cp -L "$blob" "$runtime_dest/$(basename "$blob")"
  done < <(runtime_matches '/opt/intel/oneapi/compiler/latest/lib/clbltfn*.rtl /opt/intel/oneapi/compiler/latest/lib/cllibrary*.rtl /opt/intel/oneapi/compiler/latest/lib/cllibrary*.o')
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
  local key="$1" prefix="$2" so tool redist_bin llvm_root identity sdk_version
  identity="$(cat "$prefix/.mom-build-identity" 2>/dev/null)" || identity=
  # install-dev records the compiler's LLVM major; CI preserves this marker and exposes the
  # relocated SDK at /usr/lib/llvm-N. Unrelated SDK directories must not select JIT tools.
  if [[ ! $identity =~ ^[[:xdigit:]]{40}:([1-9][0-9]*):(cuda|hip):([0-9][0-9.-]*):[A-Za-z0-9.-]+:([[:xdigit:]]{64}|none)$ ]] ||
     [ "acpp-${BASH_REMATCH[2]:-}" != "$key" ]; then
    echo "AdaptiveCpp $key packaging requires a valid matching .mom-build-identity; rerun install-dev.sh for this compiler" >&2
    exit 1
  fi
  llvm_root="/usr/lib/llvm-${BASH_REMATCH[1]}"
  sdk_version="${BASH_REMATCH[3]}"
  begin_runtime "$key" "$prefix/lib:$prefix/lib/hipSYCL:/opt/ubuntu24-libs:$llvm_root/lib:/usr/lib/x86_64-linux-gnu"
  cp -rL "$prefix/lib/." "$runtime_dest/"
  rm -rf "$runtime_dest/cmake"
  if [ "$key" = acpp-hip ]; then
    # Keep the driver-facing HIP stack host-owned and version-coherent. The packaged compiler/JIT
    # payload is portable across compatible ROCm 7 runtimes; partial ROCm libraries are not.
    rm -f "$runtime_dest"/libamdhip64.so* "$runtime_dest"/libhiprtc.so* \
      "$runtime_dest"/libamd_comgr.so* "$runtime_dest"/libhsa-runtime64.so* \
      "$runtime_dest"/libhsakmt.so*
  fi
  # These workers are intentionally vendor-isolated and never select AdaptiveCpp's OpenCL
  # backend. Shipping its plugin makes a driver-only CUDA/HIP host probe libOpenCL at startup and
  # emit a misleading loader warning; the SPIR-V translator has no remaining consumer either.
  rm -f "$runtime_dest/hipSYCL/librt-backend-ocl.so" \
    "$runtime_dest/hipSYCL/llvm-to-backend/libllvm-to-spirv.so"
  # AdaptiveCpp's supported relocatable JIT layout. Without these, a package works in the build
  # (which has LLVM globally) but fails on a driver-only target when generic/SSCP first compiles.
  redist_bin="$runtime_dest/hipSYCL/ext/llvm/bin"
  mkdir -p "$redist_bin"
  # Preserve every compiled tool basename. In particular AdaptiveCpp's host translator searches for
  # `ld.lld`, not the multicall binary's `lld` name; omitting the symlink made the otherwise-complete
  # OpenMP JIT payload fall back to the build-only /usr/lib/llvm-21 path.
  for tool in opt llc lld ld.lld; do
    if ! test -x "$llvm_root/bin/$tool"; then
      echo "AdaptiveCpp $key packaging requires executable LLVM JIT tool: $llvm_root/bin/$tool" >&2
      exit 1
    fi
    cp -L "$llvm_root/bin/$tool" "$redist_bin/$tool"
    closure_queue+=("$llvm_root/bin/$tool")
  done
  # Retain the compiled basename used by the CI LLVM20 HIP translator alongside native LLVM tools.
  [ "$key" != acpp-hip ] || ln -- "$redist_bin/ld.lld" "$redist_bin/ld.lld-20"
  if [ "$key" = acpp-hip ]; then
    mkdir -p "$runtime_dest/hipSYCL/ext/bitcode/amdgcn"
    local device_bitcode rocm_device_glob
    if [[ ! $sdk_version =~ ^[0-9]+(\.[0-9]+)+$ ]] ||
       { [ -n "${MOM_ROCM_VERSION:-}" ] && [ "$MOM_ROCM_VERSION" != "$sdk_version" ]; }; then
      echo 'AdaptiveCpp HIP bitcode must match the recorded ROCm SDK version' >&2
      exit 1
    fi
    # CI preserves the compiler cohort's install-relative bitcode. Native installs retain the
    # versioned SDK root; unrelated legacy directories or mutable /opt/rocm links cannot select it.
    if [ -n "$(runtime_matches "$prefix/lib/hipSYCL/ext/bitcode/amdgcn/*.bc")" ]; then
      rocm_device_glob="$prefix/lib/hipSYCL/ext/bitcode/amdgcn/*.bc"
    else
      rocm_device_glob="/opt/rocm-${sdk_version}/amdgcn/bitcode/*.bc"
    fi
    if ! device_bitcode="$(runtime_matches "$rocm_device_glob")"; then
      echo "AdaptiveCpp acpp-hip packaging could not inspect the pinned ROCm device bitcode root" >&2
      exit 1
    fi
    if [ -z "$device_bitcode" ]; then
      echo "AdaptiveCpp acpp-hip packaging found no bitcode for its recorded ROCm SDK: $sdk_version" >&2
      exit 1
    fi
    while IFS= read -r tool; do
      cp -L "$tool" "$runtime_dest/hipSYCL/ext/bitcode/amdgcn/$(basename "$tool")"
    done <<<"$device_bitcode"
  elif [ "$key" = acpp-cuda ]; then
    [[ $sdk_version =~ ^[0-9]+-[0-9]+$ ]] || {
      echo 'AdaptiveCpp CUDA bitcode requires a recorded major-minor SDK version' >&2
      exit 1
    }
    # Destination prescribed by AdaptiveCpp's deployment manifest. The runtime resolves this
    # relative to its own libraries; the recorded SDK, not a mutable cuda link, selects the source.
    mkdir -p "$runtime_dest/hipSYCL/ext/bitcode/ptx"
    cp -L "/usr/local/cuda-${sdk_version/-/.}/nvvm/libdevice/libdevice.10.bc" \
      "$runtime_dest/hipSYCL/ext/bitcode/ptx/libdevice.10.bc"
  fi
  while IFS= read -r so; do closure_queue+=("$so"); done < <(
    find "$prefix/lib" -type f -name '*.so*' -print)
  copy_dependency_closure
}
copy_acpp_runtime acpp-cuda /opt/adaptivecpp-cuda
copy_acpp_runtime acpp-hip /opt/adaptivecpp-hip

# Normalize package copies only, before aliasing/deduplication and loader validation. Original
# build workers and vendor driver libraries are never modified.
for key in "${compilers[@]}"; do
  normalize_runtime_elfs "$libs_dir/$key"
done

# Keep the AdaptiveCpp runtime directories isolated for loader safety, but hard-link any
# byte-identical support files so tar stores one copy. Compiler identities choose the LLVM majors;
# cmp naturally keeps differing JIT payloads separate.
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
ln "$libs_dir/oneapi/mom.node" "$libs_dir/mom.node"
while IFS= read -r -d '' file; do
  name="$(basename "$file")"
  [ "$name" = mom.node ] || [ -e "$libs_dir/$name" ] || ln "$file" "$libs_dir/$name"
done < <(find "$libs_dir/oneapi" -maxdepth 1 -type f -print0)

# Validate every isolated closure. Driver/loader libraries listed in is_base_lib intentionally come
# from install.sh and are the only permitted unresolved dependencies on the packaging host.
failed=0
while IFS= read -r -d '' file; do
  relative="${file#"$libs_dir/"}"
  key="${relative%%/*}"
  compiler_root="$libs_dir/$key"
  dir="$(dirname "$file")"
  shared_root=
  [ "$key" != dpcpp-opencl ] || shared_root=":$libs_dir/dpcpp"
  if output="$(LD_LIBRARY_PATH="$compiler_root:$compiler_root/hipSYCL:$dir$shared_root" ldd "$file" 2>&1)"; then :; else true; fi
  unresolved="$(awk '/not found/{print $1}' <<<"$output" | while read -r lib; do is_base_lib "$lib" || echo "$lib"; done)"
  if [ -n "$unresolved" ]; then echo "$file: unresolved $unresolved" >&2; failed=1; fi
done < <(find "$libs_dir" -type f \( -name 'mom.node' -o -name '*.so' -o -name '*.so.*' \) -print0)
[ "$failed" -eq 0 ] || exit 1

# Normal npm outputs share the disposable build stage, so a hard-killed publication is
# reclaimed by the next package run. Resolve only the parent: mv replaces a final symlink.
archive_stage_parent="$(dirname -- "$archive")"
if [ "$(realpath -m -- "$archive_stage_parent")" = "$(pwd -P)" ]; then
  archive_stage_parent="$build_dir"
fi
archive_stage="$(mktemp -d -- "$archive_stage_parent/.mom-archive.XXXXXX")"
tar -C release-combined -czf "$archive_stage/archive.tgz" "$root"
mv -fT -- "$archive_stage/archive.tgz" "$archive"
rmdir -- "$archive_stage"
archive_stage=
echo "$archive"
