#!/usr/bin/env bash

# Headers used by PearlHash's optional NVIDIA source-JIT kernel. Keep this separate from install.sh so
# release hosts, development containers, and CI provision the exact same architecture-neutral source.
MOM_CUTLASS_VERSION=v4.6.1
MOM_CUTLASS_SHA256=455d9ba37d57cb214d67b5d1a6070441244b378bcacb2e916c3b86f2a9b02e1c
MOM_CUTLASS_URL="https://github.com/NVIDIA/cutlass/archive/refs/tags/${MOM_CUTLASS_VERSION}.tar.gz"

install_cutlass_headers() {
  local destination="${1:-/opt/mom/cutlass}"
  local marker="$destination/.mom-version"
  if [ -f "$destination/include/cute/tensor.hpp" ] &&
     [ "$(cat "$marker" 2>/dev/null || true)" = "$MOM_CUTLASS_SHA256" ]; then
    echo "  CUTLASS $MOM_CUTLASS_VERSION headers are already installed."
    return
  fi

  local work archive
  work="$(mktemp -d)"
  archive="$work/cutlass.tar.gz"
  curl -fsSL --retry 5 "$MOM_CUTLASS_URL" -o "$archive"
  echo "$MOM_CUTLASS_SHA256  $archive" | sha256sum -c - >/dev/null
  rm -rf "$destination"
  mkdir -p "$destination"
  tar -xzf "$archive" -C "$destination" --strip-components=1 \
    "cutlass-${MOM_CUTLASS_VERSION#v}/include"
  printf '%s\n' "$MOM_CUTLASS_SHA256" >"$marker"
  rm -rf "$work"
  echo "  Installed CUTLASS $MOM_CUTLASS_VERSION headers."
}
