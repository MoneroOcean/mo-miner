#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[native-ci] Refusing unbounded producer launch: %s\n' "$1" >&2
  exit 2
}

script_path="$(readlink -f -- "${BASH_SOURCE[0]}")" || fail 'cannot resolve wrapper path'
script_dir="$(dirname -- "$script_path")"

if [[ "${1:-}" == "--verify" ]]; then
  [[ $# -ge 5 ]] || fail 'invalid internal scope invocation'
  expected_uid="$2"
  [[ "$expected_uid" =~ ^[1-9][0-9]*$ && "$EUID" -ne 0 && "$(id -u)" == "$expected_uid" ]] ||
    fail 'scope must return to the original non-root user'
  verifier_node="$3"
  [[ "$verifier_node" == /* && -x "$verifier_node" ]] || fail 'invalid verifier executable'
  shift 3
  [[ "$1" == '--' ]] || fail 'invalid internal command delimiter'
  shift
  (($# > 0)) || fail 'missing producer command'
  # Verify before restoring producer PATH or Node preloads; neither may replace this check.
  /usr/bin/env -i "$verifier_node" "$script_dir/native-ci-memory-envelope.js" ||
    fail 'kernel memory envelope verification failed'
  # Restore the caller's environment only after dropping privileges. NUL framing preserves
  # multiline values without exposing CI secrets in argv, files, or a root process's loader.
  while IFS= read -r -d '' entry; do
    name="${entry%%=*}"
    [[ "$entry" == *=* && "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || fail 'invalid producer environment key'
    export "$entry" || fail 'cannot restore producer environment'
  done
  exec "$@"
fi

(($# > 0)) || fail 'missing producer command'
((EUID != 0)) || fail 'invoke as the original non-root workflow user'

original_uid="$(id -u)" || fail 'cannot determine original uid'
original_user="$(id -un)" || fail 'cannot determine original username'
[[ "$original_uid" =~ ^[1-9][0-9]*$ && "$original_user" =~ ^[A-Za-z_][A-Za-z0-9_.-]*\$?$ &&
  "$(id -u "$original_user")" == "$original_uid" ]] || fail 'cannot validate original non-root identity'
verifier_node="$(type -P node)" || fail 'cannot find the workflow Node executable'
verifier_node="$(readlink -f -- "$verifier_node")" || fail 'cannot resolve the workflow Node executable'
[[ "$verifier_node" == /* && -x "$verifier_node" ]] || fail 'invalid workflow Node executable'

/usr/bin/env -i "$verifier_node" "$script_dir/native-ci-memory-envelope.js" --host-headroom ||
  fail 'host lacks more than 12 GiB plus 512 MiB of available memory'

sudo_path=/usr/bin/sudo
systemd_run_path=/usr/bin/systemd-run
runuser_path=/usr/sbin/runuser
for executable in "$sudo_path" "$systemd_run_path" "$runuser_path"; do
  [[ "$executable" == /* && -x "$executable" ]] || fail 'required scope executable is not an absolute executable path'
done

/usr/bin/env -0 | "$sudo_path" -n "$systemd_run_path" --scope --quiet --same-dir \
  -p MemoryMax=12884901888 -p MemorySwapMax=0 -- \
  "$runuser_path" --user "$original_user" -- \
  "$script_path" --verify "$original_uid" "$verifier_node" -- "$@"
