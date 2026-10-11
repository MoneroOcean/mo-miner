#!/usr/bin/env bash

# Consume only verified npm scaffolding before compiler lookup. A real nested npm command
# reinstates the anchor; compiler wrappers inherit its removal and do not repeat this work.
if [ -n "${npm_package_json:-}" ]; then
  if mom_normalized_path=$(node "${BASH_SOURCE[0]%/*}/build-environment.js" --path && printf '.'); then
    export PATH="${mom_normalized_path%.}"
    unset npm_package_json
  else
    mom_normalize_status=$?
    if [ "$mom_normalize_status" -ne 3 ]; then return "$mom_normalize_status"; fi
  fi
  unset mom_normalized_path mom_normalize_status
fi

# Override Gyp's Make variable, not compiler flags: GCC keeps -MMD semantics even with a later
# -MD. This records system headers without changing code generation or generated Makefiles.
mom_make_dependency_args=('DEPFLAGS=-MD -MF $(depfile).raw')

# DPC++'s host -MD graph can omit device-only includes. Each frontend emits its own graph
# during the same compile, then merge before Gyp consumes the single raw dependency rule.
mom_compile_sycl_dependencies() {
  local targets=$1 helper=$2 depfile= arg previous= target leaf
  shift 2
  local -a command=("$@") leaves=() extra=() device_targets=()
  local -A seen=()
  for arg in "$@"; do
    if [ "$previous" = -MF ]; then depfile=$arg; fi
    case "$arg" in -MF?*) depfile=${arg#-MF} ;; esac
    previous=$arg
  done
  if [ -z "$depfile" ]; then "$@"; return; fi
  if [ -L "$depfile" ]; then return 2; fi
  rm -f -- "$depfile"
  IFS=, read -r -a device_targets <<< "$targets"
  for target in "${device_targets[@]}"; do
    case "$target" in nvidia_gpu_sm_*) target=nvptx64-nvidia-cuda ;; esac
    [[ "$target" =~ ^[a-zA-Z0-9_-]+$ ]] || return 2
    if [ "${seen[$target]:-0}" = 1 ]; then continue; fi
    seen[$target]=1
    leaf="$depfile.device-$target"
    if [ -L "$leaf" ] || [ -L "$depfile" ]; then return 2; fi
    rm -f -- "$leaf"
    leaves+=("$leaf")
    extra+=("-Xsycl-target-frontend=$target" "-MF$leaf")
  done
  "${command[@]}" "${extra[@]}" || return $?
  node "$helper" merge-deps "$depfile" "${leaves[@]}"
}

# Check before parking/moving caches: a symlinked parent redirects later recursive removal.
mom_assert_linux_cache_paths() {
  local root=$1 key candidate
  for key in '' oneapi dpcpp dpcpp-opencl acpp-cuda acpp-hip; do
    candidate="$root${key:+/$key}"
    if [ -L "$candidate" ] || { [ -e "$candidate" ] && [ ! -d "$candidate" ]; }; then
      echo 'Linux compiler cache paths must be real directories' >&2
      return 2
    fi
  done
}

# Run noisy compiler commands quietly on success while preserving complete diagnostics on failure.
# Set MOM_BUILD_VERBOSE=1 for an interactive full build log.
mom_run_quiet() {
  local label="$1"
  shift
  mkdir -p "$(dirname "$MOM_BUILD_LOG")"
  if [ "${MOM_BUILD_VERBOSE:-0}" = 1 ]; then
    echo "$label"
    "$@"
    return
  fi
  if ! "$@" >"$MOM_BUILD_LOG" 2>&1; then
    echo "$label failed" >&2
    cat "$MOM_BUILD_LOG" >&2
    return 1
  fi
}
