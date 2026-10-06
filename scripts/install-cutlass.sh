#!/usr/bin/env bash

# Headers used by PearlHash's optional NVIDIA source-JIT kernel. Keep this separate from install.sh so
# release hosts, development containers, and CI provision the exact same architecture-neutral source.
MOM_CUTLASS_VERSION=v4.6.1
MOM_CUTLASS_SHA256=455d9ba37d57cb214d67b5d1a6070441244b378bcacb2e916c3b86f2a9b02e1c
MOM_CUTLASS_URL="https://github.com/NVIDIA/cutlass/archive/refs/tags/${MOM_CUTLASS_VERSION}.tar.gz"

validate_cutlass_destination() {
  local destination="$1"
  local marker="$destination/.mom-version"
  if [ -L "$destination" ]; then
    echo "Refusing CUTLASS destination symlink: $destination" >&2
    return 1
  fi
  if [ -e "$destination" ] && [ ! -d "$destination" ]; then
    echo "Refusing CUTLASS destination that is not a directory: $destination" >&2
    return 1
  fi
  if [ -L "$marker" ]; then
    echo "Refusing CUTLASS version marker symlink: $marker" >&2
    return 1
  fi
  if [ -e "$marker" ] && [ ! -f "$marker" ]; then
    echo "Refusing CUTLASS version marker that is not a regular file: $marker" >&2
    return 1
  fi
}

install_cutlass_headers() {
  local destination="${1:-/opt/mom/cutlass}"
  local marker="$destination/.mom-version"
  validate_cutlass_destination "$destination" || return 1
  if [ -f "$destination/include/cute/tensor.hpp" ] &&
     [ "$(cat "$marker" 2>/dev/null || true)" = "$MOM_CUTLASS_SHA256" ]; then
    echo "  CUTLASS $MOM_CUTLASS_VERSION headers are already installed."
    return
  fi

  local work archive
  work="$(mktemp -d)" || return 1
  archive="$work/cutlass.tar.gz"
  if ! (
    curl -fsSL --retry 5 "$MOM_CUTLASS_URL" -o "$archive" || exit 1
    echo "$MOM_CUTLASS_SHA256  $archive" | sha256sum -c - >/dev/null || exit 1
    validate_cutlass_destination "$destination" || exit 1
    mkdir -p -- "$destination" || exit 1
    tar -xzf "$archive" -C "$destination" --strip-components=1 \
      "cutlass-${MOM_CUTLASS_VERSION#v}/include" || exit 1
    validate_cutlass_destination "$destination" || exit 1
    printf '%s\n' "$MOM_CUTLASS_SHA256" >"$marker" || exit 1
  ); then
    rm -rf -- "$work"
    return 1
  fi
  rm -rf -- "$work" || return 1
  echo "  Installed CUTLASS $MOM_CUTLASS_VERSION headers."
}
