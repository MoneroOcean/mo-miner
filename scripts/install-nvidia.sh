#!/usr/bin/env bash
set -euo pipefail

# Install the NVIDIA driver that mom's bundled DPC++ CUDA runtime needs (libcuda.so.1), from
# Ubuntu's own apt repositories (the restricted component -- no extra apt repos, no downloads). mom
# already bundles the SYCL runtime user-space (libsycl + the CUDA Unified Runtime adapter + the
# ProgPoW kernel-compiler JIT) in libs/, so the driver alone is enough for CORRECT mining of every algo.
# NOTE: full-speed kawpow/firopow/evrprogpow additionally need a CUDA toolkit (libdevice + CUDA headers
# under /usr/local/cuda) AND a host C++ toolchain (g++/libstdc++) for the runtime source-JIT; without
# them those three fall back to a correct ~4 MH/s AOT kernel (every other algo is unaffected). See the
# "NVIDIA GPU install" section of README.md.
# Ubuntu 24.04 / 26.04. A reboot is required after the driver's kernel module is installed.

have() { command -v "$1" >/dev/null 2>&1; }
found_libcuda() { ldconfig -p 2>/dev/null | grep -q "libcuda.so.1"; }

if found_libcuda || have nvidia-smi; then
  echo "NVIDIA driver already present:"
  have nvidia-smi && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  /' || true
  echo "Ready. Run './mom algo_params' to confirm a gpu1 device is listed."
  exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
  exec sudo -- "$0" "$@"
fi

if [ ! -r /etc/os-release ]; then
  echo "/etc/os-release is missing; unable to detect the Linux distribution." >&2
  exit 1
fi
. /etc/os-release
if [ "${ID:-}" != "ubuntu" ]; then
  echo "This installer targets Ubuntu. Install the proprietary NVIDIA driver (>= 560 for CUDA 12.6)" >&2
  echo "from your distribution and reboot. The CUDA toolkit + g++ are only needed for full-speed" >&2
  echo "kawpow/firopow/evrprogpow (the source-JIT); see README.md 'NVIDIA GPU install'." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update

# Prefer ubuntu-drivers' headless (--gpgpu) recommendation; otherwise install the newest
# nvidia-driver-*-server-open from apt (server = headless/datacenter, -open = Turing+ open modules).
installed=0
if apt-get install -y --no-install-recommends ubuntu-drivers-common && ubuntu-drivers install --gpgpu; then
  installed=1
else
  drv="$(apt-cache search 'nvidia-driver-[0-9].*server-open' 2>/dev/null \
         | grep -oE 'nvidia-driver-[0-9]+-server-open' | sort -t- -k3 -n | tail -1)"
  if [ -n "$drv" ]; then
    echo "Installing $drv"
    apt-get install -y "$drv" && installed=1
  fi
fi

if [ "$installed" != 1 ]; then
  echo "Could not install an NVIDIA driver via apt. Install one manually (e.g. 'sudo ubuntu-drivers install')." >&2
  exit 1
fi

echo "NVIDIA driver installed. REBOOT, then run './mom algo_params' to confirm a gpu1 device is listed."
