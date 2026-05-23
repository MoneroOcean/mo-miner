#!/usr/bin/env bash

set -e

case "$1" in
  cflags|ldflags) ;;
  *)
    echo "Usage: $0 cflags|ldflags" >&2
    exit 1
    ;;
esac

flags="-O3 -ffast-math -funroll-loops -fmerge-all-constants"

case "${MOMINER_LTO:-auto}" in
  auto)
    flags="-flto $flags"
    ;;
  thin)
    flags="-flto=thin $flags"
    ;;
  1|full|on|true|yes)
    flags="-flto $flags"
    ;;
  0|off|false|no)
    ;;
  *)
    echo "Unsupported MOMINER_LTO=${MOMINER_LTO}" >&2
    exit 1
    ;;
esac

if [ "$1" = "cflags" ]; then
  echo "-DXMRIG_FEATURE_ASM $flags"
else
  echo "$flags"
fi
