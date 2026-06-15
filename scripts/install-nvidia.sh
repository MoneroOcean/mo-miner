#!/usr/bin/env bash
# Runtime check for the Linux/NVIDIA mom release. Unlike the Intel install.sh
# (which fetches the Intel GPU compute runtime), the NVIDIA build bundles its own
# DPC++ SYCL runtime (libsycl + the CUDA Unified Runtime adapter + the kawpow
# kernel-compiler JIT) in libs/ and only needs the system NVIDIA driver
# (libcuda.so.1) to run on the GPU.
set -eu

have() { command -v "$1" >/dev/null 2>&1; }
found_libcuda() {
  for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib /lib/x86_64-linux-gnu; do
    [ -e "$d/libcuda.so.1" ] && return 0
  done
  # Fall back to the dynamic linker cache; grep's exit status is the result.
  ldconfig -p 2>/dev/null | grep -q "libcuda.so.1"
}

echo "mom (Linux/NVIDIA) runtime check"

if found_libcuda || have nvidia-smi; then
  echo "  NVIDIA driver: detected."
  if have nvidia-smi; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /' || true
  fi
  echo "  Ready: GPU mining (gpu1...) and the SYCL CPU fallback are both available."
  exit 0
fi

cat <<'EOF'
  NVIDIA driver NOT detected (no libcuda.so.1).
  GPU mining needs the proprietary NVIDIA driver. The miner will otherwise fall
  back to the bundled SYCL CPU (OpenMP) device, which is for verification only.

  To install the driver:
    Ubuntu/Debian : sudo ubuntu-drivers install   (or: sudo apt install nvidia-driver-580)
    RHEL/Fedora   : sudo dnf install akmod-nvidia
    Other         : https://www.nvidia.com/Download/index.aspx
  A reboot is required after installing the kernel driver.

  The CUDA *toolkit* is NOT required at runtime — the DPC++ SYCL runtime is
  bundled in libs/; only the driver is needed.
EOF
exit 0
