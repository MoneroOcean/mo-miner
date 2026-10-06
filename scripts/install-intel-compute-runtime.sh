#!/usr/bin/env bash
set -euo pipefail

# Pin the image-local Intel GPU user-mode driver. The kernel driver remains host-owned.
readonly COMPUTE_RUNTIME_VERSION=26.31.39395.13
readonly IGC_VERSION=2.40.13
readonly IGC_BUILD=22418
readonly LEVEL_ZERO_VERSION=1.32.0
readonly COMPUTE_URL="https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}"
readonly IGC_URL="https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}"
readonly LEVEL_ZERO_URL="https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}"

readonly -a PACKAGE_NAMES=(
  libigdgmm12
  intel-igc-core-2
  intel-igc-opencl-2
  intel-ocloc
  libze1
  libze-dev
  libze-intel-gpu1
  intel-opencl-icd
)
readonly -a PACKAGE_VERSIONS=(
  22.10.0
  "${IGC_VERSION}"
  "${IGC_VERSION}"
  "${COMPUTE_RUNTIME_VERSION}-0"
  "${LEVEL_ZERO_VERSION}"
  "${LEVEL_ZERO_VERSION}"
  "${COMPUTE_RUNTIME_VERSION}-0"
  "${COMPUTE_RUNTIME_VERSION}-0"
)
readonly -a PACKAGE_FILES=(
  libigdgmm12_22.10.0_amd64.deb
  "intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb"
  "intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb"
  "intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb"
  "libze1_${LEVEL_ZERO_VERSION}+u24.04_amd64.deb"
  "libze-dev_${LEVEL_ZERO_VERSION}+u24.04_amd64.deb"
  "libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb"
  "intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb"
)
readonly -a PACKAGE_SHA256=(
  6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54
  ebd795e9fddf303a9b24b7f04545d8ddd9ad1f85b3d0cb1166476fab24da6d44
  4f990874efc11c3f6091a663b08aef576c4af592dcd8f12e116f8c2fc92d34d9
  12c5e61ed1dca5cbf38494e280abf88100a451580d57c44f601a17d9727e465e
  3c846af24f84a89150f6a4c6adcb4ea4ebef74dc119fe44f4e269bfaa72c7ba6
  4b783ed5fb937a55a7a0f3a8bc66af252f362e82476ebc0304da36173c9f2eb8
  1722943f81b576b9bb8d61016464208f48ce533dc3bf24ad39605293115cc289
  5a9c9e8fdca8a2f9e22754b1a4618c7babf21d7c3ab3503c680005007c7a8c44
)

[[ $(id -u) -eq 0 ]] || { echo "install-intel-compute-runtime.sh must run as root" >&2; exit 1; }
[[ $(dpkg --print-architecture) == amd64 ]] || { echo "Intel runtime requires amd64" >&2; exit 1; }
# shellcheck disable=SC1091
. /etc/os-release
[[ ${ID:-} == ubuntu && ${VERSION_ID:-} == 24.04 ]] || {
  echo "Intel runtime packages require Ubuntu 24.04" >&2
  exit 1
}

runtime_ready() {
  local index installed
  [[ -r /usr/lib/x86_64-linux-gnu/libze_loader.so.1 &&
    -r /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 ]] || return 1
  for index in "${!PACKAGE_NAMES[@]}"; do
    installed=$(dpkg-query -W -f='${Status} ${Version}' \
      "${PACKAGE_NAMES[index]}" 2>/dev/null || true)
    [[ $installed == "install ok installed ${PACKAGE_VERSIONS[index]}" ]] || return 1
  done
}

runtime_ready && exit 0

workspace=$(mktemp -d /var/tmp/mom-intel-runtime.XXXXXX)
trap 'rm -rf -- "$workspace"' EXIT
declare -a packages=()
for index in "${!PACKAGE_FILES[@]}"; do
  file="$workspace/${PACKAGE_FILES[index]}"
  case ${PACKAGE_NAMES[index]} in
    intel-igc-*) base_url=$IGC_URL ;;
    libze1|libze-dev) base_url=$LEVEL_ZERO_URL ;;
    *) base_url=$COMPUTE_URL ;;
  esac
  curl -fsSL --retry 5 "${base_url}/${PACKAGE_FILES[index]/+/%2B}" -o "$file"
  printf '%s  %s\n' "${PACKAGE_SHA256[index]}" "$file" | sha256sum -c -
  packages+=("$file")
done

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ocl-icd-libopencl1 "${packages[@]}"
runtime_ready || { echo "Pinned Intel compute runtime failed validation" >&2; exit 1; }
