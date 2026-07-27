#!/usr/bin/env bash
set -euo pipefail

dpcpp=${MOM_DPCPP_ROOT:-/opt/dpcpp}
compiler="$dpcpp/bin/clang++"
target=${MOM_AMD_TARGET:-gfx1200}
rocm_libs=${MOM_ROCM_DEVICE_LIBS:-/usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode}
rocm_root=${MOM_ROCM_ROOT:-${ROCM_PATH:-/opt/rocm}}

is_link=0
is_sycl=0
for arg in "$@"; do
  case "$arg" in
    -shared|*.node) is_link=1 ;;
    sycl/*.cpp|*/sycl/*.cpp) is_sycl=1 ;;
  esac
done

if [ "$is_link" = 1 ] || [ "$is_sycl" = 1 ]; then
  args=(
    -fsycl
    -fsycl-targets=amdgcn-amd-amdhsa
    -Xsycl-target-backend=amdgcn-amd-amdhsa "--offload-arch=$target"
    "--rocm-device-lib-path=$rocm_libs"
    "-I$rocm_root/include"
  )
  if [ "$is_link" = 1 ]; then
    # intel/llvm PR #21385 exposed a pass-order regression: the default AMDGPU LTO pipeline runs
    # GlobalOffsetPass after AMDGPUAttributor, retaining hidden kernel arguments that HIP does not
    # populate (launch error 401). Running GlobalOffsetPass first produces the same valid hidden-
    # argument layout as the modern LLVM pipeline without requiring AdaptiveCpp.
    args+=(
      -Xoffload-linker=amdgcn-amd-amdhsa
      '--lto-newpm-passes=globaloffset,lto<O3>'
      "-L$rocm_root/lib"
      "-Wl,-rpath,$dpcpp/lib"
    )
  fi
  exec "$compiler" "${args[@]}" "$@"
fi

exec "${MOM_AMD_HOST_CXX:-clang++-21}" "$@"
