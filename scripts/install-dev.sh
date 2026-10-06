#!/usr/bin/env bash
set -euo pipefail

# Canonical Linux development-toolchain installer. GPU drivers are deliberately out of scope; this
# installs the SDKs, compilers, Node/C++ build prerequisites, and all three SYCL compiler families
# used by GPU-CONFIG.md. Components are independently selectable so Docker BuildKit and hosted
# CI can cache/build the expensive source toolchains in separate jobs.

NODE_VERSION=${MOM_NODE_VERSION-24.15.0}
NODE_SHA256=${MOM_NODE_SHA256-472655581fb851559730c48763e0c9d3bc25975c59d518003fc0849d3e4ba0f6}
DPCPP_RELEASE=${MOM_DPCPP_RELEASE-nightly-2026-07-11}
DPCPP_ASSET=${MOM_DPCPP_ASSET-sycl_linux.tar.gz}
DPCPP_SHA256=${MOM_DPCPP_SHA256-7b2e774121370132f930db508196c5c4abdc6c7763867c9da26f54e5145b881c}
ADAPTIVECPP_COMMIT=${MOM_ADAPTIVECPP_COMMIT-da2463e45aa90aa36306c45abcfc05b87de51bc6}
CUDA_VERSION=${MOM_CUDA_VERSION-12-6}
ROCM_VERSION=${MOM_ROCM_VERSION-7.1.1}
LLVM_VERSION=${MOM_LLVM_VERSION-21}
# Pin direct toolchain payloads and repository trust roots; package managers verify the payloads
# fetched after these keys are installed.
LLVM_INSTALLER_URL=https://raw.githubusercontent.com/opencollab/llvm-jenkins.debian.net/6dc0d1ad7de83d0782731687fd555a7859b4da58/llvm.sh
LLVM_INSTALLER_SHA256=03878e08f47b66cc95bc4b544b0db3c6d9ce8d60e6cf2492ae357984330a9eae
ONEAPI_KEY_SHA256=db932ba032a71f732dc415eef5f4f185fc7ae768487d7a357bc2772746b6c48c
CUDA_KEYRING_SHA256=d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba
ROCM_KEY_SHA256=2de99e2354646a90d9903e2a669fc4e36b02c1bbff7075c481e12d7edab2c88b
# Bump only when build_adaptivecpp's output recipe changes. This keeps stale installed toolchains
# detectable without rebuilding LLVM for unrelated installer orchestration edits.
ADAPTIVECPP_RECIPE_VERSION=1
JOBS=${MOM_BUILD_JOBS-$(nproc)}
WORKSPACE=${MOM_DEV_WORKSPACE:-/var/tmp/mom-dev-toolchains}
KEEP_WORKSPACE=0
VALIDATE_ONLY=0
declare -a COMPONENTS=()
SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)"

usage() {
  cat <<'USAGE'
Usage: scripts/install-dev.sh [options]

Options:
  --component NAME[,NAME...]  Repeatable component selection (default: all)
  --jobs N                    Bound source/compiler parallelism
  --workspace PATH            Download/build workspace
  --keep-workspace            Preserve sources and build trees
  --validate-only             Check selected components without installing
  -h, --help                  Show this help

Components:
  base node oneapi cuda rocm dpcpp acpp-cuda acpp-hip all

`all` installs a complete mom development environment except GPU drivers. Docker/CI normally select
one source compiler per cache stage, for example:
  scripts/install-dev.sh --component acpp-cuda --jobs 2
USAGE
}

while (($#)); do
  case "$1" in
    --component)
      [[ $# -ge 2 ]] || { echo "--component requires a value" >&2; exit 2; }
      IFS=',' read -ra requested <<<"$2"
      COMPONENTS+=("${requested[@]}")
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs requires a positive integer" >&2; exit 2; }
      JOBS=$2
      shift 2
      ;;
    --workspace)
      [[ $# -ge 2 && -n "$2" ]] || { echo "--workspace requires a path" >&2; exit 2; }
      WORKSPACE=$2
      shift 2
      ;;
    --keep-workspace) KEEP_WORKSPACE=1; shift ;;
    --validate-only) VALIDATE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

invalid_config() {
  echo "Invalid installer configuration: $1" >&2
  exit 2
}

validate_config() {
  local LC_ALL=C
  [[ $NODE_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
    invalid_config "MOM_NODE_VERSION must be numeric x.y.z"
  [[ $NODE_SHA256 =~ ^[0-9A-Fa-f]{64}$ ]] ||
    invalid_config "MOM_NODE_SHA256 must be exactly 64 hexadecimal characters"
  [[ $DPCPP_RELEASE =~ ^[A-Za-z0-9._-]+$ ]] ||
    invalid_config "MOM_DPCPP_RELEASE must use ASCII letters, numbers, dot, underscore, or hyphen"
  [[ $DPCPP_ASSET =~ ^[A-Za-z0-9._-]+$ ]] ||
    invalid_config "MOM_DPCPP_ASSET must use ASCII letters, numbers, dot, underscore, or hyphen"
  [[ $DPCPP_SHA256 =~ ^[0-9A-Fa-f]{64}$ ]] ||
    invalid_config "MOM_DPCPP_SHA256 must be exactly 64 hexadecimal characters"
  [[ $ADAPTIVECPP_COMMIT =~ ^[0-9A-Fa-f]{40}$ ]] ||
    invalid_config "MOM_ADAPTIVECPP_COMMIT must be exactly 40 hexadecimal characters"
  [[ $CUDA_VERSION =~ ^[0-9]+-[0-9]+$ ]] ||
    invalid_config "MOM_CUDA_VERSION must be numeric major-minor"
  [[ $ROCM_VERSION =~ ^[0-9]+(\.[0-9]+)+$ ]] ||
    invalid_config "MOM_ROCM_VERSION must be a numeric dot-separated version"
  [[ $LLVM_VERSION =~ ^[1-9][0-9]*$ ]] ||
    invalid_config "MOM_LLVM_VERSION must be a positive decimal integer"
  [[ $JOBS =~ ^[1-9][0-9]*$ ]] ||
    invalid_config "MOM_BUILD_JOBS or --jobs must be a positive decimal integer"
}

validate_config

((${#COMPONENTS[@]})) || COMPONENTS=(all)
declare -a expanded=()
for component in "${COMPONENTS[@]}"; do
  case "$component" in
    all)
      expanded+=(base node oneapi cuda rocm dpcpp acpp-cuda acpp-hip)
      ;;
    base|node|oneapi|cuda|rocm|dpcpp|acpp-cuda|acpp-hip)
      expanded+=("$component")
      ;;
    *) echo "Unknown component: $component" >&2; exit 2 ;;
  esac
done
COMPONENTS=("${expanded[@]}")
WORKSPACE="$(realpath -m -- "$WORKSPACE")"
if [[ ! ${WORKSPACE##*/} =~ ^mom-dev-[A-Za-z0-9_.-]+$ ]]; then
  echo "--workspace must name a dedicated mom-dev-* directory" >&2
  exit 2
fi
workspace_marker="$WORKSPACE/.mom-dev-workspace"

selected() {
  local wanted=$1 item
  for item in "${COMPONENTS[@]}"; do [[ "$item" == "$wanted" ]] && return 0; done
  return 1
}

if [[ $(id -u) -ne 0 ]]; then
  sudo_args=(--component "$(IFS=,; echo "${COMPONENTS[*]}")" --jobs "$JOBS" --workspace "$WORKSPACE")
  ((KEEP_WORKSPACE)) && sudo_args+=(--keep-workspace)
  ((VALIDATE_ONLY)) && sudo_args+=(--validate-only)
  exec sudo --preserve-env=MOM_NODE_VERSION,MOM_DPCPP_RELEASE,MOM_DPCPP_ASSET,\
MOM_NODE_SHA256,MOM_DPCPP_SHA256,MOM_ADAPTIVECPP_COMMIT,MOM_CUDA_VERSION,\
MOM_ROCM_VERSION,MOM_LLVM_VERSION,MOM_BUILD_JOBS,MOM_DEV_WORKSPACE "$0" "${sudo_args[@]}"
fi

[[ -r /etc/os-release ]] || { echo "/etc/os-release is required" >&2; exit 1; }
# shellcheck disable=SC1091
. /etc/os-release
[[ ${ID:-} == ubuntu ]] || { echo "install-dev.sh currently supports Ubuntu, found ${PRETTY_NAME:-unknown}" >&2; exit 1; }
case ${VERSION_ID:-} in 24.04|26.04) ;; *) echo "Ubuntu 24.04 or 26.04 is required" >&2; exit 1 ;; esac
ARCH=$(dpkg --print-architecture)
[[ $ARCH == amd64 ]] || { echo "Only Ubuntu amd64 is currently supported" >&2; exit 1; }

APT_UPDATED=0
BASE_INSTALLED=0
apt_update() {
  ((APT_UPDATED)) || { apt-get update; APT_UPDATED=1; }
}
apt_install() {
  apt_update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}

verify_sha256() {
  local file=$1 expected=$2 actual
  actual=$(sha256sum -- "$file")
  actual=${actual%% *}
  if [[ $actual != "${expected,,}" ]]; then
    echo "SHA256 mismatch for $file: expected ${expected,,}, got $actual" >&2
    rm -f -- "$file"
    exit 1
  fi
}

install_base() {
  ((BASE_INSTALLED)) && return
  # Keep the generic base resolvable from stock Ubuntu 24.04/26.04. LLVM 21 is added from apt.llvm.org
  # only by the AdaptiveCpp components that need it.
  apt_install build-essential ca-certificates cmake curl git gnupg iputils-ping libboost-context-dev \
    libboost-fiber-dev libhwloc-dev libzstd-dev ninja-build pkg-config python3 python3-psutil \
    python3-yaml lsb-release sudo xz-utils
  BASE_INSTALLED=1
}

install_node() {
  if command -v node >/dev/null 2>&1 && [[ $(node -p process.version) == "v$NODE_VERSION" ]] &&
     command -v node-gyp >/dev/null 2>&1; then
    return
  fi
  local archive="$WORKSPACE/node-v${NODE_VERSION}-linux-x64.tar.xz"
  curl -fsSL --retry 5 "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o "$archive"
  verify_sha256 "$archive" "$NODE_SHA256"
  tar -C /usr/local --strip-components=1 -xf "$archive"
  npm install -g node-gyp@12.2.0
}

install_oneapi() {
  if [[ -x /opt/intel/oneapi/compiler/latest/bin/icpx ]] &&
     /opt/intel/oneapi/compiler/latest/bin/icpx --version 2>/dev/null | grep -q '2026\.0\.0'; then
    return
  fi
  local key=/usr/share/keyrings/oneapi-archive-keyring.gpg
  local key_source="$WORKSPACE/oneapi-repository-key.pub"
  mkdir -p "$(dirname "$key")"
  curl -fsSL --retry 5 \
    https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    -o "$key_source"
  verify_sha256 "$key_source" "$ONEAPI_KEY_SHA256"
  gpg --dearmor --yes -o "$key" "$key_source"
  printf '%s\n' "deb [signed-by=$key] https://apt.repos.intel.com/oneapi all main" \
    >/etc/apt/sources.list.d/oneAPI.list
  APT_UPDATED=0
  # The compiler package contains icx/icpx, the SYCL runtime, and the Level Zero/OpenCL adapters
  # used by mom. The oneAPI toolkit umbrella additionally installs MKL, MPI, VTune, Fortran, IPP,
  # and other multi-gigabyte products that neither the build nor release package consumes.
  apt_install intel-oneapi-compiler-dpcpp-cpp-2026.0
  ln -sfn 2026.0 /opt/intel/oneapi/compiler/latest
}

install_cuda() {
  # shellcheck disable=SC1091
  . "$SCRIPT_DIR/install-cutlass.sh"
  local cuda_dir="/usr/local/cuda-${CUDA_VERSION/-/.}"
  if ! cuda_payload_ready "$cuda_dir"; then
    # NVIDIA does not always publish a repository for a brand-new Ubuntu release immediately. The
    # CUDA 12.6 Ubuntu-24.04 SDK is glibc-compatible on 26.04 and contains no display driver packages.
    local repo_os=ubuntu2404 keyring="$WORKSPACE/cuda-keyring.deb"
    curl -fsSL --retry 5 \
      "https://developer.download.nvidia.com/compute/cuda/repos/${repo_os}/x86_64/cuda-keyring_1.1-1_all.deb" \
      -o "$keyring"
    verify_sha256 "$keyring" "$CUDA_KEYRING_SHA256"
    dpkg -i "$keyring"
    APT_UPDATED=0
    # Keep the compiler SDK narrow. cuda-toolkit also pulls profilers, GUI tools, BLAS/FFT/solver
    # libraries, and OpenCL components; mom needs only NVCC/PTXAS/libdevice, CUDA/CCCL headers,
    # cudart development stubs, and NVRTC for architecture-aware source JIT.
    apt_install "cuda-minimal-build-${CUDA_VERSION}" "cuda-nvrtc-dev-${CUDA_VERSION}"
  fi
  cuda_payload_ready "$cuda_dir" || {
    echo "CUDA SDK payload is missing: $cuda_dir" >&2
    exit 1
  }
  ln -sfnT "$cuda_dir" /usr/local/cuda
  link_points_to /usr/local/cuda "$cuda_dir" || {
    echo "CUDA SDK link does not select $cuda_dir" >&2
    exit 1
  }
  install_cutlass_headers
}

cuda_payload_ready() {
  local cuda_dir=$1
  [[ -x $cuda_dir/bin/nvcc && -x $cuda_dir/bin/ptxas && -r $cuda_dir/nvvm/libdevice/libdevice.10.bc &&
    -r $cuda_dir/include/nvrtc.h &&
    ( -r $cuda_dir/include/cuda/std/cstdint ||
      -r $cuda_dir/include/cccl/cuda/std/cstdint ) ]]
}

rocm_payload_ready() {
  local rocm_dir=$1
  [[ -x $rocm_dir/bin/hipcc && -r $rocm_dir/lib/libamdhip64.so ]]
}

install_rocm() {
  local rocm_dir="/opt/rocm-${ROCM_VERSION}"
  if ! rocm_payload_ready "$rocm_dir"; then
    local key=/usr/share/keyrings/rocm-archive-keyring.gpg
    local key_source="$WORKSPACE/rocm-repository-key.pub"
    mkdir -p "$(dirname "$key")"
    curl -fsSL --retry 5 https://repo.radeon.com/rocm/rocm.gpg.key -o "$key_source"
    verify_sha256 "$key_source" "$ROCM_KEY_SHA256"
    gpg --dearmor --yes -o "$key" "$key_source"
    printf '%s\n' "deb [arch=amd64 signed-by=$key] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
      >/etc/apt/sources.list.d/rocm.list
    printf '%s\n' 'Package: *' 'Pin: release o=repo.radeon.com' 'Pin-Priority: 600' \
      >/etc/apt/preferences.d/rocm-pin-600
    APT_UPDATED=0
    # mom needs the HIP compiler/runtime/RTC headers and device bitcode, not the multi-gigabyte BLAS,
    # FFT, solver, tensor, and collective libraries pulled in by the rocm-hip-sdk umbrella package.
    # Keep hipcc explicit: ROCm 7.1's rocm-hip-sdk metadata no longer pulls it in on a clean Noble
    # installation even though the SDK validator and native addon builds require it.
    apt_install hip-dev hipcc rocm-device-libs
  fi
  rocm_payload_ready "$rocm_dir" || {
    echo "ROCm SDK payload is missing: $rocm_dir" >&2
    exit 1
  }
  ln -sfnT "$rocm_dir" /opt/rocm
  link_points_to /opt/rocm "$rocm_dir" || {
    echo "ROCm SDK link does not select $rocm_dir" >&2
    exit 1
  }
}

marker_matches() {
  local marker=$1 expected=$2
  [[ -f $marker && ! -L $marker ]] && [[ $(<"$marker") == "$expected" ]]
}

link_points_to() {
  [[ -L $1 && $(readlink -f -- "$1") == "$2" ]]
}

replace_staged_directory() (
  local staged=$1 dest=$2 backup=""
  if [[ -e $dest || -L $dest ]]; then
    backup=$(mktemp -d "/opt/$(basename "$dest").previous.XXXXXX") || return 1
    if ! rmdir "$backup"; then
      rm -rf -- "$backup"
      return 1
    fi
    # A subshell function retains its locals while its EXIT trap runs. Restore the old payload if
    # replacement stops after moving it aside, and never delete the only recoverable copy.
    rollback_staged_directory() {
      local status=$?
      if [[ ! -e $dest && ! -L $dest && ( -e $backup || -L $backup ) ]]; then
        mv -- "$backup" "$dest" || status=1
      fi
      if [[ -e $dest || -L $dest ]]; then
        rm -rf -- "$backup" || status=1
      fi
      exit "$status"
    }
    trap rollback_staged_directory EXIT
    mv -- "$dest" "$backup" || return 1
  fi
  mv -- "$staged" "$dest" || return 1
  if [[ -n $backup ]]; then rm -rf -- "$backup" || return 1; fi
  trap - EXIT
)

dpcpp_payload_ready() {
  [[ -x /opt/dpcpp/bin/clang++ && -r /opt/dpcpp/lib/libsycl.so.9 &&
    -r /opt/dpcpp/lib/libsycl-jit.so ]]
}

dpcpp_ready() {
  marker_matches /opt/dpcpp/.mom-toolchain-sha256 "$DPCPP_SHA256" && dpcpp_payload_ready
}

install_dpcpp() {
  dpcpp_ready && return
  local url archive="$WORKSPACE/dpcpp-${DPCPP_RELEASE}.tar.gz"
  url="https://github.com/intel/llvm/releases/download/${DPCPP_RELEASE}/${DPCPP_ASSET}"
  curl -fsSL --retry 5 "$url" -o "$archive"
  verify_sha256 "$archive" "$DPCPP_SHA256"
  local staging
  staging=$(mktemp -d /opt/dpcpp.staging.XXXXXX)
  trap "rm -rf -- '$staging'" EXIT
  if ! tar -C "$staging" -xf "$archive"; then
    echo "DPC++ archive extraction failed: $archive" >&2
    exit 1
  fi
  if ! [[ -x $staging/bin/clang++ && -r $staging/lib/libsycl.so.9 &&
    -r $staging/lib/libsycl-jit.so ]]; then
    echo "DPC++ staged payload is missing required files: $staging" >&2
    exit 1
  fi
  printf '%s\n' "$DPCPP_SHA256" >"$staging/.mom-toolchain-sha256"
  if ! replace_staged_directory "$staging" /opt/dpcpp; then
    echo "DPC++ staged payload replacement failed" >&2
    exit 1
  fi
  trap - EXIT
}

ensure_llvm() {
  [[ -x /usr/bin/clang-${LLVM_VERSION} && -d /usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm ]] && return
  apt_install software-properties-common
  curl -fsSL --retry 5 "$LLVM_INSTALLER_URL" -o "$WORKSPACE/llvm.sh"
  verify_sha256 "$WORKSPACE/llvm.sh" "$LLVM_INSTALLER_SHA256"
  bash "$WORKSPACE/llvm.sh" "$LLVM_VERSION" all
  APT_UPDATED=0
  apt_install "libclang-${LLVM_VERSION}-dev" "libomp-${LLVM_VERSION}-dev" "llvm-${LLVM_VERSION}-dev"
}

build_adaptivecpp() {
  local backend=$1 dest=$2
  local cuda_dir="/usr/local/cuda-${CUDA_VERSION/-/.}" rocm_dir="/opt/rocm-${ROCM_VERSION}"
  local src="$WORKSPACE/adaptivecpp-$backend" build="$WORKSPACE/adaptivecpp-$backend-build"
  adaptivecpp_ready "$dest" "$backend" && return
  ensure_llvm
  rm -rf "$src" "$build"
  git clone --filter=blob:none https://github.com/AdaptiveCpp/AdaptiveCpp "$src"
  git -C "$src" checkout "$ADAPTIVECPP_COMMIT"
  if [[ $backend == cuda ]]; then
    [[ -x $cuda_dir/bin/ptxas ]] || { echo "CUDA SDK is required for acpp-cuda" >&2; exit 1; }
    git -C "$src" apply "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches/adaptivecpp-cuda-unloading.patch"
  else
    [[ -r $rocm_dir/lib/libamdhip64.so ]] || { echo "ROCm SDK is required for acpp-hip" >&2; exit 1; }
  fi
  local -a options=(
    -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$dest"
    -DCMAKE_C_COMPILER="clang-${LLVM_VERSION}" -DCMAKE_CXX_COMPILER="clang++-${LLVM_VERSION}"
    -DLLVM_DIR="/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm"
    -DClang_DIR="/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/clang"
    -DACPP_COMPILER_FEATURE_PROFILE=full -DACPP_EXPERIMENTAL_LLVM=ON
    -DACPP_SUBPROJECT_PARALLEL_JOBS="$JOBS" -DDEFAULT_TARGETS=generic
  )
  if [[ $backend == cuda ]]; then
    options+=(-DWITH_CUDA_BACKEND=ON -DWITH_ROCM_BACKEND=OFF
      -DCUDA_TOOLKIT_ROOT_DIR="$cuda_dir")
  else
    options+=(
      -DWITH_CUDA_BACKEND=OFF -DWITH_ROCM_BACKEND=ON
      -DROCM_DEVICE_LIBS_PATH="$rocm_dir/amdgcn/bitcode"
      -DAMDHIP64_LIBRARY="$rocm_dir/lib/libamdhip64.so"
      -DACPP_LLD_PATH="/usr/bin/ld.lld-${LLVM_VERSION}"
    )
  fi
  cmake -S "$src" -B "$build" "${options[@]}"
  cmake --build "$build" --parallel "$JOBS"
  local staging staged_dest
  staging=$(mktemp -d "/opt/adaptivecpp-${backend}.staging.XXXXXX")
  trap "rm -rf -- '$staging'" EXIT
  if ! DESTDIR="$staging" cmake --install "$build"; then
    echo "AdaptiveCpp staged install failed: $dest" >&2
    exit 1
  fi
  staged_dest="$staging$dest"
  if ! adaptivecpp_payload_ready "$staged_dest" "$backend"; then
    echo "AdaptiveCpp staged payload is missing required files: $staged_dest" >&2
    exit 1
  fi
  adaptivecpp_identity "$backend" >"$staged_dest/.mom-build-identity"
  if ! replace_staged_directory "$staged_dest" "$dest"; then
    echo "AdaptiveCpp staged payload replacement failed: $dest" >&2
    exit 1
  fi
  rm -rf -- "$staging"
  trap - EXIT
}

adaptivecpp_payload_ready() {
  local dest=$1 backend=$2
  [[ -x $dest/bin/acpp && -r $dest/lib/hipSYCL/librt-backend-${backend}.so &&
    -r $dest/lib/hipSYCL/librt-backend-omp.so ]]
}

adaptivecpp_identity() {
  local backend=$1 sdk_version patch_hash=none
  if [[ $backend == cuda ]]; then
    sdk_version=$CUDA_VERSION
    patch_hash=$(sha256sum "$SCRIPT_DIR/patches/adaptivecpp-cuda-unloading.patch")
    patch_hash=${patch_hash%% *}
  else
    sdk_version=$ROCM_VERSION
  fi
  printf '%s\n' \
    "$ADAPTIVECPP_COMMIT:$LLVM_VERSION:$backend:$sdk_version:$ADAPTIVECPP_RECIPE_VERSION:$patch_hash"
}

adaptivecpp_ready() {
  local dest=$1 backend=$2
  marker_matches "$dest/.mom-build-identity" "$(adaptivecpp_identity "$backend")" &&
    adaptivecpp_payload_ready "$dest" "$backend"
}

validate_component() {
  case "$1" in
    base) command -v cmake >/dev/null && command -v ninja >/dev/null && command -v git >/dev/null ;;
    node) [[ $(node -p process.version 2>/dev/null) == "v$NODE_VERSION" ]] && command -v node-gyp >/dev/null ;;
    oneapi) [[ -x /opt/intel/oneapi/compiler/latest/bin/icpx ]] &&
      /opt/intel/oneapi/compiler/latest/bin/icpx --version 2>/dev/null | grep -q '2026\.0\.0' ;;
    cuda)
      local cuda_dir="/usr/local/cuda-${CUDA_VERSION/-/.}"
      cuda_payload_ready "$cuda_dir" &&
        [[ -r /opt/mom/cutlass/include/cute/tensor.hpp ]] &&
        link_points_to /usr/local/cuda "$cuda_dir"
      ;;
    rocm)
      local rocm_dir="/opt/rocm-${ROCM_VERSION}"
      rocm_payload_ready "$rocm_dir" && link_points_to /opt/rocm "$rocm_dir"
      ;;
    dpcpp) dpcpp_ready ;;
    acpp-cuda) adaptivecpp_ready /opt/adaptivecpp-cuda cuda ;;
    acpp-hip) adaptivecpp_ready /opt/adaptivecpp-hip hip ;;
  esac
}

if ((VALIDATE_ONLY)); then
  failed=0
  for component in "${COMPONENTS[@]}"; do
    if validate_component "$component"; then echo "dev component ok: $component"
    else echo "dev component missing: $component" >&2; failed=1; fi
  done
  exit "$failed"
fi

if [[ -e "$WORKSPACE" || -L "$WORKSPACE" ]]; then
  if [[ ! -d "$WORKSPACE" || -L "$WORKSPACE" ]]; then
    echo "--workspace must be an owned directory" >&2
    exit 2
  fi
  if ! marker_matches "$workspace_marker" 'mom development workspace'; then
    echo "--workspace exists without the .mom-dev-workspace marker" >&2
    exit 2
  fi
else
  mkdir -p -- "$WORKSPACE"
  printf '%s\n' 'mom development workspace' >"$workspace_marker"
fi

selected base && install_base
selected node && { install_base; install_node; }
selected oneapi && { install_base; install_oneapi; }
selected cuda && { install_base; install_cuda; }
selected rocm && { install_base; install_rocm; }
selected dpcpp && { install_base; install_dpcpp; }
selected acpp-cuda && { install_base; install_cuda; build_adaptivecpp cuda /opt/adaptivecpp-cuda; }
selected acpp-hip && { install_base; install_rocm; build_adaptivecpp hip /opt/adaptivecpp-hip; }

for component in "${COMPONENTS[@]}"; do
  validate_component "$component" || { echo "Installed component failed validation: $component" >&2; exit 1; }
  echo "dev component ready: $component"
done
((KEEP_WORKSPACE)) || rm -rf -- "$WORKSPACE"
