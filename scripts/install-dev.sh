#!/usr/bin/env bash
set -euo pipefail

# Canonical Linux development-toolchain installer. GPU drivers are deliberately out of scope; this
# installs the SDKs, compilers, Node/C++ build prerequisites, and all three SYCL compiler families
# used by GPU-COMPILERS.md. Components are independently selectable so Docker BuildKit and hosted
# CI can cache/build the expensive source toolchains in separate jobs.

NODE_VERSION=${MOM_NODE_VERSION:-24.15.0}
DPCPP_RELEASE=${MOM_DPCPP_RELEASE:-nightly-2026-07-11}
DPCPP_ASSET=${MOM_DPCPP_ASSET:-sycl_linux.tar.gz}
DPCPP_SHA256=${MOM_DPCPP_SHA256:-7b2e774121370132f930db508196c5c4abdc6c7763867c9da26f54e5145b881c}
DPCPP_HIP_BRANCH=${MOM_DPCPP_HIP_BRANCH:-nightly-2026-07-11}
DPCPP_HIP_COMMIT=${MOM_DPCPP_HIP_COMMIT:-eca4d070277a1e62b196a5fddefe72bc7f98ee24}
ADAPTIVECPP_COMMIT=${MOM_ADAPTIVECPP_COMMIT:-da2463e45aa90aa36306c45abcfc05b87de51bc6}
CUDA_VERSION=${MOM_CUDA_VERSION:-12-6}
ROCM_VERSION=${MOM_ROCM_VERSION:-7.1.1}
LLVM_VERSION=${MOM_LLVM_VERSION:-21}
JOBS=${MOM_BUILD_JOBS:-$(nproc)}
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
  base node oneapi cuda rocm dpcpp dpcpp-hip acpp-cuda acpp-hip all

`all` installs a complete mom development environment except GPU drivers. Docker/CI normally select
one source compiler per cache stage, for example:
  scripts/install-dev.sh --component dpcpp-hip --jobs 2
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

((${#COMPONENTS[@]})) || COMPONENTS=(all)
declare -a expanded=()
for component in "${COMPONENTS[@]}"; do
  case "$component" in
    all)
      expanded+=(base node oneapi cuda rocm dpcpp acpp-cuda acpp-hip)
      ;;
    base|node|oneapi|cuda|rocm|dpcpp|dpcpp-hip|acpp-cuda|acpp-hip)
      expanded+=("$component")
      ;;
    *) echo "Unknown component: $component" >&2; exit 2 ;;
  esac
done
COMPONENTS=("${expanded[@]}")

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
MOM_DPCPP_SHA256,MOM_DPCPP_HIP_BRANCH,MOM_DPCPP_HIP_COMMIT,MOM_ADAPTIVECPP_COMMIT,MOM_CUDA_VERSION,\
MOM_ROCM_VERSION,MOM_LLVM_VERSION,MOM_BUILD_JOBS,MOM_DEV_WORKSPACE "$0" "${sudo_args[@]}"
fi

[[ -r /etc/os-release ]] || { echo "/etc/os-release is required" >&2; exit 1; }
# shellcheck disable=SC1091
. /etc/os-release
[[ ${ID:-} == ubuntu ]] || { echo "install-dev.sh currently supports Ubuntu, found ${PRETTY_NAME:-unknown}" >&2; exit 1; }
case ${VERSION_ID:-} in 24.04|26.04) ;; *) echo "Ubuntu 24.04 or 26.04 is required" >&2; exit 1 ;; esac
ARCH=$(dpkg --print-architecture)
[[ $ARCH == amd64 ]] || { echo "Only Ubuntu amd64 is currently supported" >&2; exit 1; }

mkdir -p "$WORKSPACE"
APT_UPDATED=0
BASE_INSTALLED=0
apt_update() {
  ((APT_UPDATED)) || { apt-get update; APT_UPDATED=1; }
}
apt_install() {
  apt_update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
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
  tar -C /usr/local --strip-components=1 -xf "$archive"
  npm install -g node-gyp@12.2.0
}

install_oneapi() {
  if [[ -x /opt/intel/oneapi/compiler/latest/bin/icpx ]] &&
     /opt/intel/oneapi/compiler/latest/bin/icpx --version 2>/dev/null | grep -q '2026\.0\.0'; then
    return
  fi
  local key=/usr/share/keyrings/oneapi-archive-keyring.gpg
  mkdir -p "$(dirname "$key")"
  curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB |
    gpg --dearmor --yes -o "$key"
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
  if [[ ! -x /usr/local/cuda/bin/ptxas ||
        ! -r /usr/local/cuda/nvvm/libdevice/libdevice.10.bc ]]; then
    # NVIDIA does not always publish a repository for a brand-new Ubuntu release immediately. The
    # CUDA 12.6 Ubuntu-24.04 SDK is glibc-compatible on 26.04 and contains no display driver packages.
    local repo_os=ubuntu2404 keyring="$WORKSPACE/cuda-keyring.deb"
    curl -fsSL --retry 5 \
      "https://developer.download.nvidia.com/compute/cuda/repos/${repo_os}/x86_64/cuda-keyring_1.1-1_all.deb" \
      -o "$keyring"
    dpkg -i "$keyring"
    APT_UPDATED=0
    # Keep the compiler SDK narrow. cuda-toolkit also pulls profilers, GUI tools, BLAS/FFT/solver
    # libraries, and OpenCL components; mom needs only NVCC/PTXAS/libdevice, CUDA/CCCL headers,
    # cudart development stubs, and NVRTC for architecture-aware source JIT.
    apt_install "cuda-minimal-build-${CUDA_VERSION}" "cuda-nvrtc-dev-${CUDA_VERSION}"
    local cuda_dir="/usr/local/cuda-${CUDA_VERSION/-/.}"
    [[ -d $cuda_dir ]] && ln -sfn "$cuda_dir" /usr/local/cuda
  fi
  install_cutlass_headers
}

install_rocm() {
  if [[ -x /opt/rocm/bin/hipcc && -r /opt/rocm/lib/libamdhip64.so ]]; then return; fi
  local key=/usr/share/keyrings/rocm-archive-keyring.gpg
  curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor --yes -o "$key"
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
  ln -sfn "/opt/rocm-${ROCM_VERSION}" /opt/rocm 2>/dev/null || true
}

install_dpcpp() {
  if [[ -x /opt/dpcpp/bin/clang++ && -r /opt/dpcpp/lib/libsycl.so.9 ]]; then return; fi
  local url archive="$WORKSPACE/dpcpp-${DPCPP_RELEASE}.tar.gz"
  url="https://github.com/intel/llvm/releases/download/${DPCPP_RELEASE}/${DPCPP_ASSET}"
  curl -fsSL --retry 5 "$url" -o "$archive"
  printf '%s  %s\n' "$DPCPP_SHA256" "$archive" | sha256sum --check --status ||
    { echo "DPC++ archive SHA256 mismatch: $archive" >&2; rm -f "$archive"; exit 1; }
  rm -rf /opt/dpcpp
  mkdir -p /opt/dpcpp
  tar -C /opt/dpcpp -xf "$archive"
}

build_dpcpp_hip() {
  [[ -x /opt/rocm/bin/hipcc || -x /opt/rocm-${ROCM_VERSION}/bin/hipcc ]] ||
    { echo "ROCm SDK is required for dpcpp-hip (select component rocm first)" >&2; exit 1; }
  if [[ -x /opt/dpcpp-amd/bin/clang++ ]]; then return; fi
  local src="$WORKSPACE/intel-llvm-hip" build="$WORKSPACE/dpcpp-hip-build"
  rm -rf "$src" "$build"
  git clone --depth=1 --filter=blob:none --single-branch --branch "$DPCPP_HIP_BRANCH" \
    https://github.com/intel/llvm "$src"
  [[ $(git -C "$src" rev-parse HEAD) == "$DPCPP_HIP_COMMIT" ]] ||
    { echo "Unexpected DPC++ HIP commit" >&2; exit 1; }
  local rocm=/opt/rocm
  CC=gcc CXX=g++ python3 "$src/buildbot/configure.py" --hip --no-assertions -s "$src" -o "$build" \
    --cmake-opt=-DUR_HIP_ROCM_DIR="$rocm" \
    --cmake-opt=-DUR_HIP_INCLUDE_DIR="$rocm/include" \
    --cmake-opt=-DUR_HIP_HSA_INCLUDE_DIR="$rocm/include/hsa" \
    --cmake-opt=-DUR_HIP_LIB_DIR="$rocm/lib"
  python3 "$src/buildbot/compile.py" -s "$src" -o "$build" -j "$JOBS" -t deploy-sycl-toolchain
  rm -rf /opt/dpcpp-amd
  mkdir -p /opt/dpcpp-amd
  cp -a "$build/install/." /opt/dpcpp-amd/
  local clang_lib
  clang_lib=$(find /opt/dpcpp-amd/lib/clang -mindepth 1 -maxdepth 1 -type d -print -quit)/lib
  ln -sfn amdgcn-amd-amdhsa-llvm "$clang_lib/amdgcn-amd-amdhsa"
}

ensure_llvm() {
  [[ -x /usr/bin/clang-${LLVM_VERSION} && -d /usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm ]] && return
  apt_install software-properties-common
  curl -fsSL https://apt.llvm.org/llvm.sh -o "$WORKSPACE/llvm.sh"
  bash "$WORKSPACE/llvm.sh" "$LLVM_VERSION" all
  APT_UPDATED=0
  apt_install "libclang-${LLVM_VERSION}-dev" "libomp-${LLVM_VERSION}-dev" "llvm-${LLVM_VERSION}-dev"
}

build_adaptivecpp() {
  local backend=$1 dest=$2
  local src="$WORKSPACE/adaptivecpp-$backend" build="$WORKSPACE/adaptivecpp-$backend-build"
  if [[ -x $dest/bin/acpp && -r $dest/lib/hipSYCL/librt-backend-${backend}.so &&
        -r $dest/lib/hipSYCL/librt-backend-omp.so ]]; then return; fi
  ensure_llvm
  rm -rf "$src" "$build"
  git clone --filter=blob:none https://github.com/AdaptiveCpp/AdaptiveCpp "$src"
  git -C "$src" checkout "$ADAPTIVECPP_COMMIT"
  if [[ $backend == cuda ]]; then
    [[ -x /usr/local/cuda/bin/ptxas ]] || { echo "CUDA SDK is required for acpp-cuda" >&2; exit 1; }
    git -C "$src" apply "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches/adaptivecpp-cuda-unloading.patch"
  else
    [[ -r /opt/rocm/lib/libamdhip64.so ]] || { echo "ROCm SDK is required for acpp-hip" >&2; exit 1; }
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
    options+=(-DWITH_CUDA_BACKEND=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda)
  else
    options+=(
      -DWITH_ROCM_BACKEND=ON -DROCM_DEVICE_LIBS_PATH=/opt/rocm/amdgcn/bitcode
      -DAMDHIP64_LIBRARY=/opt/rocm/lib/libamdhip64.so
      -DACPP_LLD_PATH="/usr/bin/ld.lld-${LLVM_VERSION}"
    )
  fi
  cmake -S "$src" -B "$build" "${options[@]}"
  cmake --build "$build" --parallel "$JOBS"
  cmake --install "$build"
}

validate_component() {
  case "$1" in
    base) command -v cmake >/dev/null && command -v ninja >/dev/null && command -v git >/dev/null ;;
    node) [[ $(node -p process.version 2>/dev/null) == "v$NODE_VERSION" ]] && command -v node-gyp >/dev/null ;;
    oneapi) [[ -x /opt/intel/oneapi/compiler/latest/bin/icpx ]] &&
      /opt/intel/oneapi/compiler/latest/bin/icpx --version 2>/dev/null | grep -q '2026\.0\.0' ;;
    cuda) [[ -x /usr/local/cuda/bin/ptxas &&
      -r /usr/local/cuda/nvvm/libdevice/libdevice.10.bc &&
      -r /usr/local/cuda/include/nvrtc.h &&
      ( -r /usr/local/cuda/include/cuda/std/cstdint ||
        -r /usr/local/cuda/include/cccl/cuda/std/cstdint ) &&
      -r /opt/mom/cutlass/include/cute/tensor.hpp ]] ;;
    rocm) [[ -x /opt/rocm/bin/hipcc && -r /opt/rocm/lib/libamdhip64.so ]] ;;
    dpcpp) [[ -x /opt/dpcpp/bin/clang++ && -r /opt/dpcpp/lib/libsycl.so.9 ]] ;;
    dpcpp-hip) [[ -x /opt/dpcpp-amd/bin/clang++ && -r /opt/dpcpp-amd/lib/libsycl.so.9 &&
      -r /opt/dpcpp-amd/lib/libur_adapter_hip.so.0 ]] ;;
    acpp-cuda) [[ -x /opt/adaptivecpp-cuda/bin/acpp &&
      -r /opt/adaptivecpp-cuda/lib/hipSYCL/librt-backend-cuda.so &&
      -r /opt/adaptivecpp-cuda/lib/hipSYCL/librt-backend-omp.so ]] ;;
    acpp-hip) [[ -x /opt/adaptivecpp-hip/bin/acpp &&
      -r /opt/adaptivecpp-hip/lib/hipSYCL/librt-backend-hip.so &&
      -r /opt/adaptivecpp-hip/lib/hipSYCL/librt-backend-omp.so ]] ;;
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

selected base && install_base
selected node && { install_base; install_node; }
selected oneapi && { install_base; install_oneapi; }
selected cuda && { install_base; install_cuda; }
selected rocm && { install_base; install_rocm; }
selected dpcpp && { install_base; install_dpcpp; }
selected dpcpp-hip && { install_base; install_rocm; build_dpcpp_hip; }
selected acpp-cuda && { install_base; install_cuda; build_adaptivecpp cuda /opt/adaptivecpp-cuda; }
selected acpp-hip && { install_base; install_rocm; build_adaptivecpp hip /opt/adaptivecpp-hip; }

for component in "${COMPONENTS[@]}"; do
  validate_component "$component" || { echo "Installed component failed validation: $component" >&2; exit 1; }
  echo "dev component ready: $component"
done
((KEEP_WORKSPACE)) || rm -rf "$WORKSPACE"
