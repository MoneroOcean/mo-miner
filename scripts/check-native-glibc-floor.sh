#!/usr/bin/env bash
# Run a standalone Linux mom's `algorithms` command, without network or GPUs, in the oldest supported
# glibc distributions. Images are pinned by digest; the binary is mounted read-only.
# Usage: scripts/check-native-glibc-floor.sh /absolute/mom
set -euo pipefail
binary=${1:-}
if [[ $# -ne 1 || $binary != /* || ! -f $binary ]]; then
  echo "Usage: scripts/check-native-glibc-floor.sh /absolute/mom" >&2
  exit 2
fi
images=(
  "ubuntu:20.04@sha256:8feb4d8ca5354def3d8fce243717141ce31e2c428701f6682bd2fafe15388214" # glibc 2.31
  "debian:11@sha256:6f519a81440354a85eb592c5f32109ab80605f6b892455983a6f618bf87fabe9"    # glibc 2.31
  "rockylinux:9@sha256:03303863808e9cd79f0c58c833db3e5f9cde9f2abe16804a267b05dc54af86e4" # glibc 2.34
)
# Artifact transport drops the executable bit; run a private executable copy.
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
install -m 0755 "$binary" "$work/mom"
status=0
for image in "${images[@]}"; do
  if output=$(docker run --rm --network none --read-only --tmpfs /tmp -e HOME=/tmp -w /tmp \
      -v "$work/mom:/opt/mom/mom:ro" "$image" /opt/mom/mom algorithms 2>&1) &&
      grep -q '^MOM_ALGORITHMS {' <<<"$output"; then
    echo "[glibc-floor] PASS ${image%%@*}"
  else
    echo "[glibc-floor] FAIL ${image%%@*}"
    printf '%s\n' "$output" | tail -n 20
    status=1
  fi
done
exit "$status"
