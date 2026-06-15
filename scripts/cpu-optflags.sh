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

# LTO defaults on ("auto"); MOM_LTO=thin uses ThinLTO, off-ish values disable it.
case "${MOM_LTO:-auto}" in
  auto|1|full|on|true|yes) flags="-flto $flags";;
  thin)                    flags="-flto=thin $flags";;
  0|off|false|no)          ;;  # no-op: build without LTO
  *)
    echo "Unsupported MOM_LTO=${MOM_LTO}" >&2
    exit 1
    ;;
esac

# cflags additionally carry the ASM-feature define; ldflags get the bare flags.
[ "$1" = cflags ] && flags="-DXMRIG_FEATURE_ASM $flags"
echo "$flags"
