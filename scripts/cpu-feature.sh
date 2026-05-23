#!/usr/bin/env bash

set -e
QUERY="$1"

check_mac() {
  case "$1" in
    msr)     sysctl -n machdep.cpu.features | grep -i msr >/dev/null;;
    sse2)    sysctl -n machdep.cpu.features | grep -i sse2 >/dev/null;;
    ssse3)   sysctl -n machdep.cpu.features | grep -i ssse3 >/dev/null;;
    sse4_1)  sysctl -n machdep.cpu.features | grep -i sse4_1 >/dev/null;;
    xop)     sysctl -n machdep.cpu.features | grep -i xop >/dev/null;;
    avx2)    sysctl -n machdep.cpu.features | grep -i avx2 >/dev/null;;
    avx512f) sysctl -n machdep.cpu.features | grep -i avx512f >/dev/null;;
    vaes)    sysctl -n machdep.cpu.features | grep -i vaes >/dev/null;;
    aes)     sysctl -n machdep.cpu.features | grep -i " aes" >/dev/null;;
    *) echo "Unrecognized CPU feature check: $QUERY"; exit 1;;
  esac
}

check_linux() {
  CPUINFO=$(cat /proc/cpuinfo)
  case "$1" in
    arm64)   uname -a | grep "aarch64" >/dev/null;;
    arm)     uname -a | grep "armv7"   >/dev/null;;
    x86_64)  uname -a | grep "x86_64"  >/dev/null;;
    msr)     grep msr      <<<$CPUINFO >/dev/null;;
    sse2)    grep sse2     <<<$CPUINFO >/dev/null;;
    ssse3)   grep ssse3    <<<$CPUINFO >/dev/null;;
    sse4_1)  grep sse4_1   <<<$CPUINFO >/dev/null;;
    xop)     grep xop      <<<$CPUINFO >/dev/null;;
    avx2)    grep avx2     <<<$CPUINFO >/dev/null;;
    avx512f) grep avx512f  <<<$CPUINFO >/dev/null;;
    vaes)    grep vaes     <<<$CPUINFO >/dev/null;;
    aes)     grep " aes"   <<<$CPUINFO >/dev/null;;
    *) echo "Unrecognized CPU feature check: $QUERY"; exit 1;;
  esac
}

case "$(uname -a)" in
  Darwin*) check_mac   "$QUERY";;
  *)       check_linux "$QUERY";;
esac
