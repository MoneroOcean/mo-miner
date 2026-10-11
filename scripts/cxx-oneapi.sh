#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/build-helpers.sh"
compile=0 sycl=0
for arg in "$@"; do
  case "$arg" in -c) compile=1 ;; -fsycl) sycl=1 ;; esac
done
if [ "$compile:$sycl" = 1:1 ]; then
  mom_compile_sycl_dependencies spir64 "$script_dir/linux-build-state.js" icpx "$@"
else
  exec icpx "$@"
fi
