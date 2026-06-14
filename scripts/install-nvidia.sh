#!/usr/bin/env bash
# Runtime check for the Linux/NVIDIA mo-miner release. Unlike the Intel install.sh
# (which fetches the Intel GPU compute runtime), the NVIDIA build bundles its own
# AdaptiveCpp + LLVM + CUDA libdevice runtime in libs/ and only needs the system
# NVIDIA driver (libcuda.so.1) to JIT and run on the GPU. If the driver is absent
# the miner still runs on the bundled SYCL CPU (OpenMP) device for verification.
set -eu

have() { command -v "$1" >/dev/null 2>&1; }
found_libcuda() {
  for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib /lib/x86_64-linux-gnu; do
    [ -e "$d/libcuda.so.1" ] && return 0
  done
  ldconfig -p 2>/dev/null | grep -q "libcuda.so.1" && return 0
  return 1
}

echo "mo-miner (Linux/NVIDIA) runtime check"

if found_libcuda || have nvidia-smi; then
  echo "  NVIDIA driver: detected (libcuda.so.1 present)."
  if have nvidia-smi; then nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /' || true; fi
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

  The CUDA *toolkit* is NOT required at runtime — AdaptiveCpp's libdevice and JIT
  are bundled in libs/; only the driver is needed.
EOF
exit 0
