#!/usr/bin/env bash

# Headers used by PearlHash's optional NVIDIA source-JIT kernel. Keep this separate from install.sh so
# release hosts, development containers, and CI provision the exact same architecture-neutral source.
MOM_CUTLASS_VERSION=v4.6.1
MOM_CUTLASS_SHA256=455d9ba37d57cb214d67b5d1a6070441244b378bcacb2e916c3b86f2a9b02e1c
MOM_CUTLASS_LICENSE_SHA256=ad1513ca983da737a6843364ad19e8424a56d860736c2da773b8c5be527e1475
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
  if [ -L "$destination/LICENSE.txt" ] || { [ -e "$destination/LICENSE.txt" ] && [ ! -f "$destination/LICENSE.txt" ]; }; then
    echo "Refusing CUTLASS license that is not a regular file: $destination/LICENSE.txt" >&2
    return 1
  fi
}

install_cutlass_headers() {
  local destination="${1:-/opt/mom/cutlass}"
  local marker="$destination/.mom-version"
  local reuse_headers=0
  validate_cutlass_destination "$destination" || return 1
  if [ -e "$destination/LICENSE.txt" ] &&
     [ "$(sha256sum "$destination/LICENSE.txt" | awk '{print $1}')" != "$MOM_CUTLASS_LICENSE_SHA256" ]; then
    echo "Existing CUTLASS $MOM_CUTLASS_VERSION LICENSE.txt has changed; refusing to overwrite it." >&2
    return 1
  fi
  if [ -f "$destination/include/cute/tensor.hpp" ] &&
     [ "$(cat "$marker" 2>/dev/null || true)" = "$MOM_CUTLASS_SHA256" ]; then
    if [ -f "$destination/LICENSE.txt" ]; then
      echo "  CUTLASS $MOM_CUTLASS_VERSION headers and license are already installed."
      return
    fi
    reuse_headers=1
  fi

  local work archive
  work="$(mktemp -d)" || return 1
  archive="$work/cutlass.tar.gz"
  if ! (
    curl -fsSL --retry 5 "$MOM_CUTLASS_URL" -o "$archive" || exit 1
    echo "$MOM_CUTLASS_SHA256  $archive" | sha256sum -c - >/dev/null || exit 1
    validate_cutlass_destination "$destination" || exit 1
    mkdir -p -- "$destination" || exit 1
    if [ "$reuse_headers" -eq 1 ]; then
      tar -xzf "$archive" -C "$work" --strip-components=1 \
        "cutlass-${MOM_CUTLASS_VERSION#v}/LICENSE.txt" || exit 1
      [ "$(sha256sum "$work/LICENSE.txt" | awk '{print $1}')" = "$MOM_CUTLASS_LICENSE_SHA256" ] || exit 1
      install -m 0644 -- "$work/LICENSE.txt" "$destination/LICENSE.txt" || exit 1
    else
      tar -xzf "$archive" -C "$destination" --strip-components=1 \
        "cutlass-${MOM_CUTLASS_VERSION#v}/include" \
        "cutlass-${MOM_CUTLASS_VERSION#v}/LICENSE.txt" || exit 1
    fi
    [ "$(sha256sum "$destination/LICENSE.txt" | awk '{print $1}')" = "$MOM_CUTLASS_LICENSE_SHA256" ] || exit 1
    validate_cutlass_destination "$destination" || exit 1
    printf '%s\n' "$MOM_CUTLASS_SHA256" >"$marker" || exit 1
  ); then
    rm -rf -- "$work"
    return 1
  fi
  rm -rf -- "$work" || return 1
  echo "  Installed CUTLASS $MOM_CUTLASS_VERSION headers."
}
