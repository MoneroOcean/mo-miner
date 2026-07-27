#!/usr/bin/env bash

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
