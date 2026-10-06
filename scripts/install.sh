#!/usr/bin/env bash
set -euo pipefail

# Unified Linux host-runtime installer. Auto-detects the GPU vendor(s) present (Intel / AMD / NVIDIA)
# and installs, from Ubuntu's own apt repositories (no extra apt repositories), the host driver/runtime
# that mom's bundled SYCL user-space needs to reach each device. It also installs the small compiler
# payload needed by source-JIT kernels. A box with more than one vendor gets all required runtimes.
# Ubuntu 24.04 / 26.04 (aim for 26.04, whose packages are new enough for Arc B-series).

if [ "$(id -u)" -ne 0 ]; then
  preserve_env=""
  for variable in MOM_INSTALL_GPU_VENDORS ROCM_PATH HIP_PATH; do
    if [ -n "${!variable:-}" ]; then
      [ -n "$preserve_env" ] && preserve_env+=","
      preserve_env+="$variable"
    fi
  done
  if [ -n "$preserve_env" ]; then
    exec sudo --preserve-env="$preserve_env" -- "$0" "$@"
  fi
  exec sudo -- "$0" "$@"
fi

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)"
MOM_CUTLASS_HELPER_SHA256=8c6e84ffe5f89e1c8a42b2117e833be6bae444b4865a93ac0760391938765940
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
has_intel=0
has_amd=0
has_nvidia=0
has_opencl=0
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
# AdaptiveCpp's bundled OpenMP fallback backend links libnuma even when the selected work runs on a
# discrete CUDA/HIP device. Keep this small runtime dependency common to every detected GPU vendor.
apt-get install -y --no-install-recommends libnuma1

reboot_needed=0
gpu_groups_changed=0

# Direct-root/image installs have no invoking desktop user to modify.
configure_gpu_access_groups() {
  local user="${SUDO_USER:-}" uid="${SUDO_UID:-}" group group_csv user_groups
  local -a missing=()
  [ -n "$user" ] && [ "$user" != root ] && [[ "$uid" =~ ^[0-9]+$ ]] || return 0
  [ "$(id -u -- "$user" 2>/dev/null || true)" = "$uid" ] || return 0
  user_groups="$(id -nG -- "$user" 2>/dev/null)" || return 0
  for group in render video; do
    getent group "$group" >/dev/null 2>&1 || continue
    case " $user_groups " in *" $group "*) ;; *) missing+=("$group") ;; esac
  done
  [ "${#missing[@]}" -gt 0 ] || return 0
  group_csv="$(IFS=,; printf '%s' "${missing[*]}")"
  usermod -aG "$group_csv" -- "$user"
  echo "Added $user to the GPU access group(s): ${missing[*]}"
  gpu_groups_changed=1
}

runtime_has_library() {
  local directory="$1" library="$2" candidate
  for candidate in "$directory/$library" "$directory/$library".*; do
    [ -e "$candidate" ] && return 0
  done
  return 1
}

find_rocm_runtime() {
  local hipconfig="" resolved="" root directory
  local -a roots=("${ROCM_PATH:-}" "${HIP_PATH:-}")
  hipconfig="$(command -v hipconfig 2>/dev/null || true)"
  if [ -n "$hipconfig" ]; then
    resolved="$(readlink -f -- "$hipconfig" 2>/dev/null || true)"
    if [[ "$resolved" == */bin/hipconfig ]]; then roots+=("${resolved%/bin/hipconfig}"); fi
  fi
  roots+=(/opt/rocm)
  for root in "${roots[@]}"; do
    [ -n "$root" ] || continue
    for directory in "$root/lib" "$root/lib64"; do
      if runtime_has_library "$directory" libamdhip64.so.7 &&
         runtime_has_library "$directory" libamd_comgr.so.3 &&
         runtime_has_library "$directory" libhsa-runtime64.so.1 &&
         runtime_has_library "$directory" libhiprtc.so.7 &&
         runtime_has_library "$directory" libhiprtc-builtins.so.7; then
        printf '%s\n' "$directory"
        return 0
      fi
    done
  done
  return 1
}

install_cuda_toolkit_payload_from_apt() {
  # Extract the CUDA toolkit/dev payload instead of apt-installing it: apt can
  # collide with NVIDIA Docker's bind-mounted driver files, but the payload is
  # enough for the SYCL source-JIT.
  local dest deb library nvrtc_builtins nvrtc_library nvrtc_builtins_library staging previous=""
  local -a packages=(nvidia-cuda-toolkit nvidia-cuda-dev libnvrtc12 libcu++-dev)
  if [[ -e /usr/local/cuda || -L /usr/local/cuda ]]; then
    if [[ -x /usr/local/cuda/bin/ptxas &&
      -r /usr/local/cuda/include/cuda.h &&
      -r /usr/local/cuda/include/cuda_runtime.h &&
      -r /usr/local/cuda/include/nvrtc.h &&
      ( -r /usr/local/cuda/include/cuda/std/cstdint ||
        -r /usr/local/cuda/include/cccl/cuda/std/cstdint ) &&
      -r /usr/local/cuda/nvvm/libdevice/libdevice.10.bc ]] &&
      runtime_has_library /usr/local/cuda/lib64 libnvrtc.so &&
      runtime_has_library /usr/local/cuda/lib64 libnvrtc-builtins.so; then
      echo "  Keeping the complete CUDA source-JIT toolkit at /usr/local/cuda."
      return 0
    fi
    echo "Existing CUDA installation at /usr/local/cuda is incomplete; repair the existing CUDA installation before running the installer. It will not be replaced automatically." >&2
    return 1
  fi
  echo "  Installing the NVIDIA source-JIT toolchain from Ubuntu apt packages..."
  nvrtc_builtins="$(apt-cache search --names-only '^libnvrtc-builtins[0-9.]+' 2>/dev/null |
    awk 'NR == 1 { print $1 }')"
  [ -z "$nvrtc_builtins" ] || packages+=("$nvrtc_builtins")
  dest="/opt/nvidia-cuda-ubuntu"
  staging="$(mktemp -d /opt/nvidia-cuda-ubuntu.staging.XXXXXX)"
  chmod 755 "$staging"
  # Expand the generated path while the function-local variable is still in scope. Bash runs an
  # EXIT trap after unwinding function locals, so a deferred "$staging" reference cannot clean up.
  trap "rm -rf -- '$staging'" EXIT
  (
    cd "$staging"
    apt-get -o APT::Sandbox::User=root download "${packages[@]}" >/dev/null
  )
  for deb in "$staging"/*.deb; do
    [ -f "$deb" ] || {
      echo "CUDA apt payload did not produce a .deb archive" >&2
      exit 1
    }
    dpkg-deb -x "$deb" "$staging"
    rm -f -- "$deb"
  done
  nvrtc_library="$(find "$staging/usr/lib" -type f -name 'libnvrtc.so*' -print -quit 2>/dev/null || true)"
  nvrtc_builtins_library="$(find "$staging/usr/lib" -type f -name 'libnvrtc-builtins.so*' -print -quit 2>/dev/null || true)"
  if ! [[ -x "$staging/usr/bin/ptxas" &&
    -d "$staging/usr/include" &&
    -r "$staging/usr/include/cuda.h" &&
    -r "$staging/usr/include/cuda_runtime.h" &&
    -r "$staging/usr/include/nvrtc.h" &&
    ( -r "$staging/usr/include/cuda/std/cstdint" ||
      -r "$staging/usr/include/cccl/cuda/std/cstdint" ) &&
    -r "$staging/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc" &&
    -n "$nvrtc_library" && -n "$nvrtc_builtins_library" ]]; then
    echo "CUDA staged payload is missing ptxas, headers, libdevice, or NVRTC" >&2
    exit 1
  fi
  if [[ -e "$dest" || -L "$dest" ]]; then
    previous="$(mktemp -d /opt/nvidia-cuda-ubuntu.previous.XXXXXX)"
    rmdir "$previous"
    mv -- "$dest" "$previous"
    # Restore the prior payload if the script exits in the narrow interval before staging is
    # installed. All paths here are fixed or generated by mktemp, so embedding them is safe.
    trap "if [[ ! -e '$dest' && ! -L '$dest' ]]; then mv -- '$previous' '$dest'; fi; rm -rf -- '$staging'" EXIT
  fi
  if ! mv -- "$staging" "$dest"; then
    if [ -n "$previous" ]; then mv -- "$previous" "$dest"; fi
    echo "CUDA staged payload replacement failed" >&2
    exit 1
  fi
  if [ -n "$previous" ]; then rm -rf -- "$previous"; fi
  trap - EXIT

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
    # Use ubuntu-drivers' hardware-specific branch recommendation, but install its headless variant.
    # This matters for pre-GSP GPUs: a newer open module may exist yet be unable to drive the card.
    apt-get install -y --no-install-recommends ubuntu-drivers-common
    recommended="$(ubuntu-drivers devices 2>/dev/null |
      awk '$1 == "driver" && /recommended/ { print $3; exit }')" || true
    if [ -z "$recommended" ]; then
      recommended="$(ubuntu-drivers list 2>/dev/null |
        grep -E '^nvidia-driver-[0-9]+$' | sort -t- -k3,3n | tail -1)" || true
    fi
    if [ -z "$recommended" ]; then
      recommended="$(ubuntu-drivers list --gpgpu 2>/dev/null |
        grep -E '^nvidia-driver-[0-9]+-server$' | sort -t- -k3,3n | tail -1)" || true
    fi
    drv="${recommended/nvidia-driver-/nvidia-headless-}"
    apt-cache show "$drv" >/dev/null 2>&1 || drv=""
    if [ -n "$drv" ]; then
      base="${drv%-open}"
      ver="$(printf '%s' "$base" | sed -E 's/^nvidia-headless-([0-9]+)(-server)?$/\1/')"
      suffix="${base#nvidia-headless-$ver}"
      utils="nvidia-utils-$ver$suffix"
      driver_packages=("$drv")
      apt-cache show "$utils" >/dev/null 2>&1 && driver_packages+=("$utils")
      echo "  Installing ${driver_packages[*]}"
      apt-get install -y --no-install-recommends "${driver_packages[@]}" && reboot_needed=1
    fi
    [ "$reboot_needed" = 1 ] || echo "  WARNING: could not install an NVIDIA driver via apt -- install one manually (e.g. 'sudo ubuntu-drivers install')." >&2
  fi
  # Full-speed ProgPoW and PearlHash need the source compiler, CUDA/CCCL headers, and CUTLASS.
  apt-get install -y --no-install-recommends ca-certificates curl g++
  install_cuda_toolkit_payload_from_apt
  if ! declare -F install_cutlass_headers >/dev/null; then
    cutlass_helper="$(mktemp)"
    trap "rm -f -- '$cutlass_helper'" EXIT
    curl -fsSL --retry 5 \
      https://raw.githubusercontent.com/MoneroOcean/mo-miner/v0.9.0/scripts/install-cutlass.sh \
      -o "$cutlass_helper"
    printf '%s  %s\n' "$MOM_CUTLASS_HELPER_SHA256" "$cutlass_helper" | sha256sum -c - >/dev/null
    # shellcheck disable=SC1090
    . "$cutlass_helper"
    rm -f -- "$cutlass_helper"
    trap - EXIT
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
  # Preserve a coherent vendor toolkit when one is already installed. Otherwise Ubuntu splits
  # HIPRTC's device builtins from libhiprtc7, and its development package supplies the unversioned
  # aliases used by dynamic source-JIT paths, so install the complete distro set together.
  apt-get install -y --no-install-recommends libomp5
  if amd_runtime="$(find_rocm_runtime)"; then
    echo "AMD GPU detected -- keeping the coherent ROCm core at $amd_runtime."
  else
    amd_packages=(libamdhip64-7 libamdhip64-dev libhiprtc7 libhiprtc-builtins7)
    echo "AMD GPU detected -- installing the ROCm HIP runtime from apt: ${amd_packages[*]}"
    apt-get install -y --no-install-recommends "${amd_packages[@]}"
    dpkg-query -W -f='  ${Package}\t${Version}\n' "${amd_packages[@]}" 2>/dev/null | sort || true
  fi
fi

ldconfig
configure_gpu_access_groups

if [ "$reboot_needed" = 1 ]; then
  echo "Done. An NVIDIA driver was installed -- REBOOT, then run './mom algorithms' to confirm a gpu1 device is listed."
elif [ "$gpu_groups_changed" = 1 ]; then
  echo "Done. Sign out and back in, then run './mom algorithms' to confirm a gpu1 device is listed."
else
  echo "Done. Run './mom algorithms' to confirm a gpu1 device is listed."
fi
