#!/usr/bin/env bash
set -euo pipefail

# Unified Linux host-runtime installer. Auto-detects the GPU vendor(s) present (Intel / AMD / NVIDIA)
# and installs, from Ubuntu's own apt repositories (no extra apt repositories), the host driver/runtime
# that mom's bundled SYCL user-space needs to reach each device. It also installs the small compiler
# payload needed by source-JIT kernels. A box with more than one vendor gets all required runtimes.
# Ubuntu 24.04 / 26.04 (aim for 26.04, whose packages are new enough for Arc B-series).

if [ "$(id -u)" -ne 0 ]; then
  exec sudo -- "$0" "$@"
fi

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)"
if [ -f "$SCRIPT_DIR/install-cutlass.sh" ]; then
  # shellcheck disable=SC1091
  . "$SCRIPT_DIR/install-cutlass.sh"
fi

if [ ! -r /etc/os-release ]; then
  echo "/etc/os-release is missing; unable to detect the Linux distribution." >&2
  exit 1
fi
# shellcheck disable=SC1091
. /etc/os-release
if [ "${ID:-}" != "ubuntu" ]; then
  echo "This installer targets Ubuntu (detected ${PRETTY_NAME:-unknown})." >&2
  echo "Install the equivalents for your GPU:" >&2
  echo "  Intel : intel-opencl-icd + the Level-Zero GPU driver + the Level-Zero/OpenCL loaders" >&2
  echo "  AMD   : the ROCm HIP and HSA runtimes" >&2
  echo "  NVIDIA: the proprietary driver (>= 560), plus the CUDA/C++ source-JIT tools" >&2
  echo "          used by full-speed ProgPoW and PearlHash (see README.md)." >&2
  exit 1
fi

# Detect the GPU vendor(s). A host can legitimately have several vendors at once.
# Prefer sysfs so a minimal Ubuntu install does not need pciutils/lspci just to decide
# what runtime packages are required. MOM_INSTALL_GPU_VENDORS is useful for container
# validation where /sys may still expose more host GPUs than the container is meant to test.
has_intel=0; has_amd=0; has_nvidia=0; has_opencl=0
if [ -n "${MOM_INSTALL_GPU_VENDORS:-}" ]; then
  case ",${MOM_INSTALL_GPU_VENDORS,,}," in *,intel,*) has_intel=1;; esac
  case ",${MOM_INSTALL_GPU_VENDORS,,}," in *,amd,*) has_amd=1;; esac
  case ",${MOM_INSTALL_GPU_VENDORS,,}," in *,nvidia,*) has_nvidia=1;; esac
  case ",${MOM_INSTALL_GPU_VENDORS,,}," in *,opencl,*|*,unknown,*) has_opencl=1;; esac
else
  for vendor_file in /sys/bus/pci/devices/*/vendor; do
    [ -r "$vendor_file" ] || continue
    class_file="${vendor_file%/vendor}/class"
    class="$(cat "$class_file" 2>/dev/null || true)"
    case "$class" in
      0x03*) ;;
      *) continue ;;
    esac
    vendor="$(cat "$vendor_file" 2>/dev/null || true)"
    case "$vendor" in
      0x8086) has_intel=1 ;;
      0x1002) has_amd=1 ;;
      0x10de) has_nvidia=1 ;;
      *) has_opencl=1 ;;
    esac
  done

  if [ "$has_intel$has_amd$has_nvidia$has_opencl" = "0000" ] && command -v lspci >/dev/null 2>&1; then
    gpus="$(lspci 2>/dev/null | grep -iE 'vga|3d|display' || true)"
    printf '%s' "$gpus" | grep -qiE 'intel|8086' && has_intel=1
    printf '%s' "$gpus" | grep -qiE 'advanced micro devices|\[amd|\bati\b|\bamd\b' && has_amd=1
    printf '%s' "$gpus" | grep -qiE 'nvidia|10de' && has_nvidia=1
    [ -z "$gpus" ] || [ "$has_intel$has_amd$has_nvidia" != "000" ] || has_opencl=1
  fi
fi
if [ "$has_intel$has_amd$has_nvidia$has_opencl" = "0000" ]; then
  echo "No GPU detected. Nothing to install; native CPU mining needs no GPU/OpenCL runtime."
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update

reboot_needed=0

install_cuda_toolkit_payload_from_apt() {
  # Extract the CUDA toolkit/dev payload instead of apt-installing it: apt can
  # collide with NVIDIA Docker's bind-mounted driver files, but the payload is
  # enough for the SYCL source-JIT.
  local work dest deb nvrtc_builtins library
  local -a packages=(nvidia-cuda-toolkit nvidia-cuda-dev libnvrtc12 libcu++-dev)
  nvrtc_builtins="$(apt-cache search --names-only '^libnvrtc-builtins[0-9.]+' 2>/dev/null |
    awk 'NR == 1 { print $1 }')"
  [ -z "$nvrtc_builtins" ] || packages+=("$nvrtc_builtins")
  work="$(mktemp -d)"
  chmod 755 "$work"
  dest="/opt/nvidia-cuda-ubuntu"
  (
    cd "$work"
    rm -rf "$dest"
    mkdir -p "$dest"
    apt-get -o APT::Sandbox::User=root download "${packages[@]}" >/dev/null
    for deb in ./*.deb; do dpkg-deb -x "$deb" "$dest"; done
  )
  rm -rf "$work"

  mkdir -p /usr/local/cuda/bin /usr/local/cuda/nvvm /usr/local/cuda/lib64
  ln -sf "$dest/usr/bin/ptxas" /usr/local/cuda/bin/ptxas
  ln -sfn "$dest/usr/include" /usr/local/cuda/include
  ln -sfn "$dest/usr/lib/nvidia-cuda-toolkit/libdevice" /usr/local/cuda/nvvm/libdevice
  for library in "$dest"/usr/lib/x86_64-linux-gnu/libnvrtc.so* \
                 "$dest"/usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so*; do
    [ -e "$library" ] || continue
    ln -sf "$library" "/usr/local/cuda/lib64/${library##*/}"
  done
  printf '%s\n' /usr/local/cuda/lib64 >/etc/ld.so.conf.d/mom-cuda.conf
}

# ---- NVIDIA: proprietary driver plus the compiler/header payload for ProgPoW and PearlHash ----
if [ "$has_nvidia" = 1 ]; then
  echo "NVIDIA GPU detected."
  if ldconfig -p 2>/dev/null | grep -q "libcuda.so.1" || command -v nvidia-smi >/dev/null 2>&1; then
    echo "  NVIDIA driver already present."
  else
    # Prefer ubuntu-drivers' headless (--gpgpu) recommendation; else the newest headless server/open
    # metapackage plus matching nvidia-utils for nvidia-smi. --no-oem keeps this on Ubuntu's own repos.
    if apt-get install -y --no-install-recommends ubuntu-drivers-common && ubuntu-drivers --gpgpu --no-oem install; then
      reboot_needed=1
    else
      drv="$(apt-cache search 'nvidia-headless-[0-9].*server-open' 2>/dev/null \
             | grep -oE 'nvidia-headless-[0-9]+-server-open' | sort -t- -k3 -n | tail -1)"
      if [ -n "$drv" ]; then
        ver="$(printf '%s' "$drv" | sed -E 's/^nvidia-headless-([0-9]+)-server-open$/\1/')"
        utils="nvidia-utils-$ver-server"
        driver_packages=("$drv")
        apt-cache show "$utils" >/dev/null 2>&1 && driver_packages+=("$utils")
        echo "  Installing ${driver_packages[*]}"
        apt-get install -y --no-install-recommends "${driver_packages[@]}" && reboot_needed=1
      fi
    fi
    [ "$reboot_needed" = 1 ] || echo "  WARNING: could not install an NVIDIA driver via apt -- install one manually (e.g. 'sudo ubuntu-drivers install')." >&2
  fi
  # Full-speed ProgPoW and PearlHash need the source compiler, CUDA/CCCL headers, and CUTLASS.
  echo "  Installing the NVIDIA source-JIT toolchain from Ubuntu apt packages..."
  apt-get install -y --no-install-recommends ca-certificates curl g++
  install_cuda_toolkit_payload_from_apt
  if ! declare -F install_cutlass_headers >/dev/null; then
    cutlass_helper="$(mktemp)"
    curl -fsSL --retry 5 \
      https://raw.githubusercontent.com/MoneroOcean/mo-miner/master/scripts/install-cutlass.sh \
      -o "$cutlass_helper"
    # shellcheck disable=SC1090
    . "$cutlass_helper"
    rm -f "$cutlass_helper"
  fi
  install_cutlass_headers
fi

# ---- Intel: NEO OpenCL GPU driver + Level-Zero GPU driver + the L0/OpenCL ICD loaders ----
if [ "$has_intel" = 1 ]; then
  #   intel-opencl-icd  : Intel OpenCL GPU driver (NEO) -- cn/gpu and OpenCL GPU devices.
  #   libze-intel-gpu1  : Intel Level-Zero GPU driver (named intel-level-zero-gpu on older Ubuntu).
  #   libze1            : oneAPI Level-Zero loader (libze_loader.so.1).
  #   ocl-icd-libopencl1: OpenCL ICD loader (libOpenCL.so.1).
  intel_packages=(intel-opencl-icd libze1 ocl-icd-libopencl1)
  if apt-cache show libze-intel-gpu1 >/dev/null 2>&1; then
    intel_packages+=(libze-intel-gpu1)
  else
    intel_packages+=(intel-level-zero-gpu)
  fi
  echo "Intel GPU detected -- installing the Intel GPU runtime from apt: ${intel_packages[*]}"
  apt-get install -y --no-install-recommends "${intel_packages[@]}"
  dpkg-query -W -f='  ${Package}\t${Version}\n' "${intel_packages[@]}" 2>/dev/null | sort || true
fi

# ---- Unknown/future GPU vendor: mom supplies SPIR-V and the Unified Runtime OpenCL adapter; the
# hardware vendor supplies its ICD. Install only the standard dispatch loader here.
if [ "$has_opencl" = 1 ]; then
  echo "Unknown GPU vendor detected -- installing the generic OpenCL ICD loader."
  apt-get install -y --no-install-recommends ocl-icd-libopencl1
  echo "  Install the GPU vendor's OpenCL ICD if it is not already provided by its driver."
fi

# ---- AMD: distro ROCm HIP/HSA runtime for generic SYCL and source-JIT kernels ----
if [ "$has_amd" = 1 ]; then
  # Ubuntu splits HIPRTC's device builtins into a separate package but libhiprtc7 does not pull it
  # in as a hard dependency. Without it AdaptiveCpp's COMGR steps succeed and hiprtcLinkComplete()
  # still fails when an SSCP kernel is finalized.
  # libamdhip64-dev supplies the unversioned libamdhip64.so/libhiprtc.so aliases that HIPRTC opens
  # dynamically while finalizing an AdaptiveCpp SSCP image; the versioned runtime alone is not
  # sufficient even though normal ELF dependency checks pass.
  amd_packages=(libamdhip64-7 libamdhip64-dev libhiprtc7 libhiprtc-builtins7 libomp5)
  echo "AMD GPU detected -- installing the ROCm HIP runtime from apt: ${amd_packages[*]}"
  apt-get install -y --no-install-recommends "${amd_packages[@]}"
  dpkg-query -W -f='  ${Package}\t${Version}\n' "${amd_packages[@]}" 2>/dev/null | sort || true
fi

ldconfig

if [ "$reboot_needed" = 1 ]; then
  echo "Done. An NVIDIA driver was installed -- REBOOT, then run './mom algo_params' to confirm a gpu1 device is listed."
else
  echo "Done. Run './mom algo_params' to confirm a gpu1 device is listed."
fi
