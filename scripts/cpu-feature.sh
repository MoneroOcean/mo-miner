#!/usr/bin/env bash

# Exit 0 if this CPU/OS has the requested feature, non-zero otherwise.
# Callers (binding.gyp, cpu-cflags.sh) use it as a boolean: `cpu-feature.sh X && ...`

set -e
QUERY="$1"

# Map a feature flag to the substring grep'd from the OS feature list. The pattern
# is the flag name itself, except "aes" needs a leading space so it can't match
# "vaes"/"pclmulqdq aes..." neighbours. An unknown flag aborts with an error.
feature_pattern() {
  case "$1" in
    aes) printf ' aes';;
    msr|sse2|ssse3|sse4_1|xop|avx2|avx512f|vaes) printf '%s' "$1";;
    *) echo "Unrecognized CPU feature check: $QUERY"; exit 1;;
  esac
}

case "$(uname -a)" in
  # macOS: feature flags live in sysctl machdep.cpu.features (case-insensitive).
  Darwin*) sysctl -n machdep.cpu.features | grep -i "$(feature_pattern "$QUERY")" >/dev/null;;
  # Linux: arch from `uname -a`, feature flags from /proc/cpuinfo.
  *) case "$QUERY" in
       arm64)  uname -a | grep "aarch64" >/dev/null;;
       arm)    uname -a | grep "armv7"   >/dev/null;;
       x86_64) uname -a | grep "x86_64"  >/dev/null;;
       *) grep "$(feature_pattern "$QUERY")" /proc/cpuinfo >/dev/null;;
     esac;;
esac
