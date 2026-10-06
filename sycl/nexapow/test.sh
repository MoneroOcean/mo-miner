#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/mom-nexapow.XXXXXX")
trap 'rm -rf "$out"' EXIT

host_cxx=${CXX:-c++}
"$host_cxx" -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -DMOM_NEXAPOW_HOST_TEST \
  "$root/nexapow.cpp" -o "$out/recorded"
"$out/recorded"
"$host_cxx" -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  "$root/test_fixed_base.cpp" -o "$out/host"
"$out/host"

source_level_private_bytes=$((49 + 32 + 32 + 64 + 32 + 80 + 113 + 32 + 32 + 32 + 97 + 33 + 32 + 64 * 4 + 18 * 8))
printf 'source-level named-array estimate (not register allocation): %d bytes\n' \
  "$source_level_private_bytes"

compiled=0
dpcpp_target=${NEXAPOW_DPCPP_TARGET:-spir64}
dpcpp=${NEXAPOW_DPCPP:-${DPCPP:-}}
if [ -z "$dpcpp" ]; then dpcpp=$(command -v dpcpp 2>/dev/null || true); fi
if [ -z "$dpcpp" ] && [ -x /opt/dpcpp/bin/clang++ ]; then dpcpp=/opt/dpcpp/bin/clang++; fi
if [ -n "$dpcpp" ]; then
  printf 'DPC++: %s\n' "$dpcpp"
  read -r -a dpcpp_flags <<< "${NEXAPOW_DPCPP_FLAGS:-}"
  # This is compile-only; no device is selected or launched.
  "$dpcpp" -std=c++17 -O2 -fsycl -fsycl-targets="$dpcpp_target" \
    "${dpcpp_flags[@]}" -c "$root/test_sycl.cpp" -o "$out/dpcpp.o"
  printf 'DPC++ device object: %s bytes\n' "$(stat -c %s "$out/dpcpp.o")"
  "$dpcpp" -std=c++17 -O2 -fsycl -fsycl-targets="$dpcpp_target" \
    "${dpcpp_flags[@]}" "$root/runtime_harness.cpp" -o "$out/runtime-dpcpp"
  printf 'DPC++ runtime harness: %s bytes (compiled, not run)\n' "$(stat -c %s "$out/runtime-dpcpp")"
  if [ "${NEXAPOW_DEVICE_REPORT:-0}" = 1 ]; then
    "$dpcpp" -std=c++17 -O2 -fsycl -fsycl-targets="$dpcpp_target" -fsycl-device-only \
      -S -emit-llvm "${dpcpp_flags[@]}" "$root/test_sycl.cpp" -o "$out/dpcpp-device.ll"
    printf 'DPC++ device LLVM IR: %s bytes; static alloca lines: %s\n' \
      "$(stat -c %s "$out/dpcpp-device.ll")" \
      "$(grep -c '= alloca ' "$out/dpcpp-device.ll" || true)"
  fi
  compiled=1
fi

acpp=${NEXAPOW_ACPP:-${ACPP:-}}
acpp_targets=${NEXAPOW_ACPP_TARGETS:-generic}
if [ -z "$acpp" ]; then acpp=$(command -v acpp 2>/dev/null || true); fi
if [ -z "$acpp" ]; then
  for candidate in /opt/adaptivecpp-cuda/bin/acpp /opt/adaptivecpp-hip/bin/acpp; do
    if [ -x "$candidate" ]; then acpp=$candidate; break; fi
  done
fi
if [ -n "$acpp" ]; then
  printf 'AdaptiveCpp: %s\n' "$acpp"
  read -r -a acpp_flags <<< "${NEXAPOW_ACPP_FLAGS:-}"
  "$acpp" -std=c++17 -O2 --acpp-targets="$acpp_targets" -c "$root/test_sycl.cpp" \
    "${acpp_flags[@]}" -o "$out/acpp.o"
  printf 'AdaptiveCpp object: %s bytes\n' "$(stat -c %s "$out/acpp.o")"
  "$acpp" -std=c++17 -O2 --acpp-targets="$acpp_targets" "$root/runtime_harness.cpp" \
    "${acpp_flags[@]}" -o "$out/runtime-acpp"
  printf 'AdaptiveCpp runtime harness: %s bytes (compiled, not run)\n' "$(stat -c %s "$out/runtime-acpp")"
  compiled=1
fi

if [ "$compiled" = 0 ]; then
  echo 'no DPC++ or AdaptiveCpp compiler found' >&2
  exit 2
fi
echo 'nexapow host and compile-only SYCL checks passed'
