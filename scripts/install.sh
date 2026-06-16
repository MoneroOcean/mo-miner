#!/usr/bin/env bash
set -euo pipefail

# Install the Intel GPU compute runtime that mom's bundled SYCL runtime needs to reach the GPU,
# straight from the distribution's own apt repositories -- no extra apt repos, no downloads. mom
# already bundles the SYCL runtime user-space (libsycl + the UR adapters + libhwloc + the OpenCL CPU
# runtime) in libs/; this only adds the host GPU driver/runtime that exposes Level-Zero / OpenCL
# devices to them. Ubuntu 24.04 / 26.04 (aim for 26.04, whose packages are new enough for Arc B-series).

if [ "$(id -u)" -ne 0 ]; then
  exec sudo -- "$0" "$@"
fi

if [ ! -r /etc/os-release ]; then
  echo "/etc/os-release is missing; unable to detect the Linux distribution." >&2
  exit 1
fi
. /etc/os-release
if [ "${ID:-}" != "ubuntu" ]; then
  echo "This installer targets Ubuntu (detected ${PRETTY_NAME:-unknown})." >&2
  echo "Install the equivalents of: intel-opencl-icd, the Level-Zero GPU driver, and the Level-Zero + OpenCL loaders." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update

# Install the OpenCL/Level-Zero runtime for the GPU vendor present. lspci may be absent on minimal
# images; if so (or if an Intel GPU is present) we install the Intel runtime, the primary target.
gpus="$(lspci 2>/dev/null | grep -iE 'vga|3d|display' || true)"
has_intel=0; has_amd=0
printf '%s' "$gpus" | grep -qiE 'intel|8086' && has_intel=1
printf '%s' "$gpus" | grep -qiE 'advanced micro devices|\[amd|\bati\b|\bamd\b' && has_amd=1

if [ "$has_amd" = 1 ] && [ "$has_intel" = 0 ]; then
  # AMD-only box: Mesa's OpenCL (rusticl) from stock apt -- no extra repos. UNTESTED on real AMD
  # hardware. rusticl is opt-in per driver, so it may need `RUSTICL_ENABLE=radeonsi` in the env; ROCm
  # (rocm-opencl-runtime, needs AMD's own apt repo) is the higher-compatibility alternative.
  # ocl-icd-libopencl1 is the OpenCL ICD loader (libOpenCL.so.1).
  pkgs="mesa-opencl-icd ocl-icd-libopencl1 clinfo"
  echo "AMD GPU detected -- installing Mesa OpenCL (rusticl) from apt: $pkgs"
  echo "NOTE: AMD support is untested. If no gpu1 appears, try 'RUSTICL_ENABLE=radeonsi ./mom algo_params', or install ROCm."
else
  # Intel (default): NEO OpenCL GPU driver + Level-Zero GPU driver + the L0/OpenCL ICD loaders.
  #   intel-opencl-icd  : Intel OpenCL GPU driver (NEO) -- cn/gpu and OpenCL GPU devices.
  #   libze-intel-gpu1  : Intel Level-Zero GPU driver (named intel-level-zero-gpu on older Ubuntu).
  #   libze1            : oneAPI Level-Zero loader (libze_loader.so.1).
  #   ocl-icd-libopencl1: OpenCL ICD loader (libOpenCL.so.1).
  pkgs="intel-opencl-icd libze1 ocl-icd-libopencl1"
  if apt-cache show libze-intel-gpu1 >/dev/null 2>&1; then pkgs="$pkgs libze-intel-gpu1"; else pkgs="$pkgs intel-level-zero-gpu"; fi
  echo "Installing Intel GPU runtime from apt: $pkgs"
fi

apt-get install -y --no-install-recommends $pkgs
ldconfig

echo "Installed GPU runtime packages:"
# Query only the packages actually selected above ($pkgs). dpkg-query -W exits nonzero if ANY queried
# name is unknown to dpkg, so listing names that weren't installed (e.g. the legacy intel-level-zero-gpu
# on 24.04, where libze-intel-gpu1 is used instead) would abort the script under `set -e` AFTER a
# successful install. The trailing `|| true` keeps this summary purely informational regardless.
dpkg-query -W -f='  ${Package}\t${Version}\n' $pkgs 2>/dev/null | sort || true
echo "Done. Run './mom algo_params' to confirm a gpu1 device is listed."
