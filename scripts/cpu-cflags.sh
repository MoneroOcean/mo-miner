#!/usr/bin/env bash

set -e

case "${MOM_CPU_MARCH:-}" in
  # Explicit "native" or unset both build native, except an unset value with
  # MOM_PORTABLE_BUILD=1, which selects a portable x86-64 baseline. Do not use
  # Intel -ax here: it may optimize static helpers directly for the build CPU
  # instead of emitting a guarded dispatch, making a release built on AVX-512
  # crash on otherwise-supported CPUs. The explicitly dispatched Argon2/Blake
  # variants and RandomX JIT/assembly retain their architecture optimizations.
  ""|native)
    if [ -z "${MOM_CPU_MARCH:-}" ] && [ "${MOM_PORTABLE_BUILD:-}" = "1" ]; then
      echo "-march=x86-64 -mtune=generic -maes"
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
