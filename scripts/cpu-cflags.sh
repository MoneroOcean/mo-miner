#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

intel_ax_flags() {
  local compiler="${CXX:-${CC:-c++}}"
  if "$compiler" --version 2>/dev/null | grep -q "Intel(R) oneAPI DPC++/C++ Compiler"; then
    printf ' %s' "-axCORE-AVX2,CORE-AVX512,ROCKETLAKE"
  fi
}

# ARM flags are architecture-detected, ignoring MOM_CPU_MARCH (which is x86-only).
if "$SCRIPT_DIR/cpu-feature.sh" arm64; then
  echo "-march=armv8-a+crypto -flax-vector-conversions"
  exit 0
elif "$SCRIPT_DIR/cpu-feature.sh" arm; then
  echo "-mfpu=neon -flax-vector-conversions"
  exit 0
fi

case "${MOM_CPU_MARCH:-}" in
  # Explicit "native" or unset both build native, except an unset value with
  # MOM_PORTABLE_BUILD=1, which selects a portable baseline + Intel -ax flags.
  ""|native)
    if [ -z "${MOM_CPU_MARCH:-}" ] && [ "${MOM_PORTABLE_BUILD:-}" = "1" ]; then
      echo "-march=x86-64 -mtune=generic -maes$(intel_ax_flags)"
    else
      echo "-march=native"
    fi
    ;;
  x86-64|x86-64-v2|x86-64-v3|x86-64-v4|rocketlake)
    echo "-march=${MOM_CPU_MARCH} -mtune=generic -maes"
    ;;
  *)
    echo "Unsupported MOM_CPU_MARCH=${MOM_CPU_MARCH}" >&2
    exit 1
    ;;
esac
