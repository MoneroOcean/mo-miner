#!/usr/bin/env bash
set -euo pipefail

is_link=0
src=
for arg in "$@"; do
  case "$arg" in
    -shared|*.node) is_link=1 ;;
    *.cpp|*.cc|*.cxx|*.C) src=$arg ;;
  esac
done

if [ "$is_link" = 1 ] || [[ "$src" == sycl/*.cpp || "$src" == */sycl/*.cpp ]]; then
  args=()
  for arg in "$@"; do
    # node-gyp adds DPC++'s driver switch globally. AdaptiveCpp's compiler wrapper
    # owns SYCL target selection through ACPP_TARGETS and must not receive it.
    [ "$arg" = -fsycl ] || args+=("$arg")
  done
  if [ "${MOM_ADAPTIVE_BACKEND:-}" = hip ]; then
    rocm_root=${MOM_ROCM_ROOT:-${ROCM_PATH:-/opt/rocm}}
    args=("-I$rocm_root/include" "-L$rocm_root/lib" "${args[@]}")
  fi
  exec acpp "${args[@]}"
fi
exec "${MOM_ADAPTIVE_HOST_CXX:-clang++-21}" "$@"
