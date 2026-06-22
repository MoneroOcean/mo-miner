#!/usr/bin/env bash
set -euo pipefail

# Unified Linux host-runtime installer. Auto-detects the GPU vendor(s) present (Intel / AMD / NVIDIA)
# and installs, from Ubuntu's own apt repositories (no extra repos, no downloads), the host driver/runtime
# that mom's bundled SYCL user-space (libsycl + the UR adapters + the OpenCL CPU runtime + the ProgPoW
# source-JIT, all in libs/) needs to reach each device. A box with more than one vendor gets all of them.
# Ubuntu 24.04 / 26.04 (aim for 26.04, whose packages are new enough for Arc B-series).

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
  echo "Install the equivalents for your GPU:" >&2
  echo "  Intel : intel-opencl-icd + the Level-Zero GPU driver + the Level-Zero/OpenCL loaders" >&2
  echo "  AMD   : mesa-opencl-icd (rusticl) or ROCm + the OpenCL loader" >&2
  echo "  NVIDIA: the proprietary driver (>= 560), plus the CUDA toolkit (libdevice) + g++ for" >&2
  echo "          full-speed kawpow/firopow/evrprogpow (the source-JIT; see README.md)." >&2
  exit 1
fi

# Detect the GPU vendor(s). lspci may be absent on minimal images; if nothing is detected we default to
# the Intel runtime (the primary target). A host can legitimately have several vendors at once.
gpus="$(lspci 2>/dev/null | grep -iE 'vga|3d|display' || true)"
has_intel=0; has_amd=0; has_nvidia=0
printf '%s' "$gpus" | grep -qiE 'intel|8086' && has_intel=1
printf '%s' "$gpus" | grep -qiE 'advanced micro devices|\[amd|\bati\b|\bamd\b' && has_amd=1
printf '%s' "$gpus" | grep -qiE 'nvidia|10de' && has_nvidia=1
if [ "$has_intel$has_amd$has_nvidia" = "000" ]; then
  echo "No GPU detected via lspci -- defaulting to the Intel runtime."
  has_intel=1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update

reboot_needed=0

# ---- NVIDIA: proprietary driver (for libcuda.so.1) + CUDA toolkit (libdevice) + g++ (the ProgPoW JIT) ----
if [ "$has_nvidia" = 1 ]; then
  echo "NVIDIA GPU detected."
  if ldconfig -p 2>/dev/null | grep -q "libcuda.so.1" || command -v nvidia-smi >/dev/null 2>&1; then
    echo "  NVIDIA driver already present."
  else
    # Prefer ubuntu-drivers' headless (--gpgpu) recommendation; else the newest nvidia-driver-*-server-open
    # (server = headless/datacenter, -open = Turing+ open kernel modules).
    if apt-get install -y --no-install-recommends ubuntu-drivers-common && ubuntu-drivers install --gpgpu; then
      reboot_needed=1
    else
      drv="$(apt-cache search 'nvidia-driver-[0-9].*server-open' 2>/dev/null \
             | grep -oE 'nvidia-driver-[0-9]+-server-open' | sort -t- -k3 -n | tail -1)"
      if [ -n "$drv" ]; then echo "  Installing $drv"; apt-get install -y "$drv" && reboot_needed=1; fi
    fi
    [ "$reboot_needed" = 1 ] || echo "  WARNING: could not install an NVIDIA driver via apt -- install one manually (e.g. 'sudo ubuntu-drivers install')." >&2
  fi
  # Full-speed kawpow/firopow/evrprogpow recompile their kernel at runtime (source-JIT), which needs a
  # CUDA toolkit (libdevice + ptxas) AND a host C++ toolchain (g++/libstdc++). Without them those three
  # fall back to a correct but ~3x slower AOT kernel (~4 vs ~13.8 MH/s on an L4); every other algo is
  # unaffected. Installing nvidia-cuda-toolkit + g++ lets the JIT auto-detect the toolkit -- verified to
  # restore 13.8 MH/s on an L4. (No NVIDIA apt repo needed; Ubuntu's nvidia-cuda-toolkit suffices.)
  echo "  Installing the ProgPoW source-JIT toolchain (nvidia-cuda-toolkit + g++)..."
  apt-get install -y --no-install-recommends nvidia-cuda-toolkit g++
  # Belt-and-suspenders: expose the toolkit under /usr/local/cuda (the canonical path the JIT probes),
  # in case a clang/JIT build does not auto-detect Ubuntu's split apt layout.
  ld="$(find /usr -name 'libdevice*.bc' 2>/dev/null | head -1 || true)"
  if [ -n "$ld" ] && [ ! -e /usr/local/cuda/nvvm/libdevice ]; then
    mkdir -p /usr/local/cuda/nvvm/libdevice /usr/local/cuda/bin
    ln -sf "$ld" "/usr/local/cuda/nvvm/libdevice/$(basename "$ld")"
    p="$(command -v ptxas || true)"; [ -n "$p" ] && ln -sf "$p" /usr/local/cuda/bin/ptxas
    [ -e /usr/local/cuda/include ] || ln -sf /usr/include /usr/local/cuda/include
  fi
fi

# ---- Intel: NEO OpenCL GPU driver + Level-Zero GPU driver + the L0/OpenCL ICD loaders ----
if [ "$has_intel" = 1 ]; then
  #   intel-opencl-icd  : Intel OpenCL GPU driver (NEO) -- cn/gpu and OpenCL GPU devices.
  #   libze-intel-gpu1  : Intel Level-Zero GPU driver (named intel-level-zero-gpu on older Ubuntu).
  #   libze1            : oneAPI Level-Zero loader (libze_loader.so.1).
  #   ocl-icd-libopencl1: OpenCL ICD loader (libOpenCL.so.1).
  pkgs="intel-opencl-icd libze1 ocl-icd-libopencl1"
  if apt-cache show libze-intel-gpu1 >/dev/null 2>&1; then pkgs="$pkgs libze-intel-gpu1"; else pkgs="$pkgs intel-level-zero-gpu"; fi
  echo "Intel GPU detected -- installing the Intel GPU runtime from apt: $pkgs"
  apt-get install -y --no-install-recommends $pkgs
  dpkg-query -W -f='  ${Package}\t${Version}\n' $pkgs 2>/dev/null | sort || true
elif [ "$has_amd" = 1 ]; then
  # AMD-only box: Mesa's OpenCL (rusticl) from stock apt -- no extra repos. UNTESTED on real AMD hardware.
  # rusticl is opt-in per driver (may need RUSTICL_ENABLE=radeonsi in the env); ROCm (needs AMD's apt repo)
  # is the higher-compatibility alternative. ocl-icd-libopencl1 is the OpenCL ICD loader (libOpenCL.so.1).
  pkgs="mesa-opencl-icd ocl-icd-libopencl1 clinfo"
  echo "AMD GPU detected -- installing Mesa OpenCL (rusticl) from apt: $pkgs"
  echo "NOTE: AMD support is untested. If no gpu1 appears, try 'RUSTICL_ENABLE=radeonsi ./mom algo_params', or install ROCm."
  apt-get install -y --no-install-recommends $pkgs
fi

ldconfig

if [ "$reboot_needed" = 1 ]; then
  echo "Done. An NVIDIA driver was installed -- REBOOT, then run './mom algo_params' to confirm a gpu1 device is listed."
else
  echo "Done. Run './mom algo_params' to confirm a gpu1 device is listed."
fi
