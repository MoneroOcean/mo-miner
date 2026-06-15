#!/usr/bin/env bash
set -euo pipefail

archive="${1:?Usage: .github/workflows/scripts/test-release-linux.sh <archive> [suite]}"
suite="${2:-all}"
node_bin="${NODE_BIN:-$(command -v node)}"
work_dir="${MOM_RELEASE_TEST_DIR:-release-test}"

escape_github_message() {
  local value="$1"
  value="${value//'%'/'%25'}"
  value="${value//$'\r'/'%0D'}"
  value="${value//$'\n'/'%0A'}"
  printf '%s' "$value"
}

# Print a one-line message to stderr and abort.
die() {
  echo "$1" >&2
  exit 1
}

# Like die(), but also emit a GitHub Actions error annotation when running in CI.
fail() {
  local title="$1" message="$2"
  if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    printf '::error title=%s::%s\n' \
      "$(escape_github_message "$title")" \
      "$(escape_github_message "$message")" >&2
  fi
  die "$message"
}

# Run a command, capturing combined stdout+stderr into the global CAPTURE_OUT
# and its exit status into CAPTURE_RC (without tripping set -e). The command
# runs in a subshell, so a leading `cd` stays contained.
CAPTURE_OUT=""
CAPTURE_RC=0
capture() {
  set +e
  CAPTURE_OUT="$( "$@" 2>&1 )"
  CAPTURE_RC=$?
  set -e
}

rm -rf "$work_dir"
mkdir -p "$work_dir"

archive_list="$(mktemp)"
trap 'rm -f "$archive_list"' EXIT
tar -tzf "$archive" >"$archive_list"

root="$(sed -n '1p' "$archive_list" | cut -d/ -f1)"
[ -n "$root" ] || die "Unable to determine archive root for $archive."
if grep -Eq '(^|/)tests(/|$)' "$archive_list"; then
  die "Release archive must not contain tests/."
fi

tar -C "$work_dir" -xzf "$archive"
package_dir="$work_dir/$root"
libs_dir="$package_dir/libs"
[ ! -d "$package_dir/tests" ] || die "Extracted release package unexpectedly contains tests/."
[ -f "$libs_dir/mom.node" ] || die "Extracted release package is missing libs/mom.node."

check_ldd() {
  local file output failed=0
  while IFS= read -r -d "" file; do
    output="$(LD_LIBRARY_PATH="$PWD/$libs_dir:$PWD/$package_dir" ldd "$file" 2>&1 || true)"
    # The unified package's CUDA UR adapter links the user-provided driver libs libcuda.so.1 /
    # libnvidia-ml.so.1. These are intentionally NOT bundled (see the base_libs list in
    # package-linux-combined.sh: the driver supplies them) and are absent on a GPU-less CI runner,
    # so their "not found" is expected, not a packaging defect — ignore those lines.
    if grep -vE '^[[:space:]]*(libcuda|libnvidia-ml)\.so' <<<"$output" | grep -q "not found"; then
      echo "$output" >&2
      failed=1
    fi
  done < <(
    find "$package_dir" "$libs_dir" -maxdepth 1 -type f \
      \( -name "mom-bin" -o -name "mom.node" -o -name "*.so" -o -name "*.so.*" \) \
      -print0
  )
  return "$failed"
}

check_ldd

cp -r tests "$package_dir/"

system_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PATH="$PWD/$package_dir:$system_path"
if [ -f "$libs_dir/libintelocl.so" ]; then
  export OCL_ICD_FILENAMES="$PWD/$libs_dir/libintelocl.so"
fi

# Run the extracted ./mom from inside the package dir, with LD_LIBRARY_PATH
# unset so the loader must find the bundled libs via rpath alone.
run_mom() { (cd "$package_dir" && env -u LD_LIBRARY_PATH "$@" ./mom algo_params); }

capture run_mom
smoke_output="$CAPTURE_OUT"
if [ "$CAPTURE_RC" -ne 0 ]; then
  smoke_exit="$CAPTURE_RC"
  capture run_mom MOM_DEBUG_STARTUP=1
  fail "Linux release smoke test failed" "$(cat <<EOF
Direct executable smoke test failed with exit code $smoke_exit.

Output:
$smoke_output

Debug exit code: $CAPTURE_RC
Debug output:
$CAPTURE_OUT
EOF
)"
fi
if ! grep -q '^MOM_ALGO_PARAMS ' <<<"$smoke_output"; then
  fail "Linux release smoke test missing marker" "$(printf '%s\n%s' \
    "Direct executable smoke test did not print algo params marker." "$smoke_output")"
fi
# Validate that every algo advertises a usable device string (non-empty and not
# a disabled "*0"/"^0" entry).
capture "$node_bin" -e '
const fs = require("node:fs");
const marker = fs.readFileSync(0, "utf8").split(/\r?\n/).find((line) => line.startsWith("MOM_ALGO_PARAMS "));
const params = JSON.parse(marker.slice("MOM_ALGO_PARAMS ".length));
for (const [algo, dev] of Object.entries(params)) {
  if (typeof dev !== "string" || !dev || /(?:^|,)[^,]*(?:\*0|\^0)(?:,|$)/.test(dev)) {
    console.error(`Invalid algo params for ${algo}: ${dev}`);
    process.exit(1);
  }
}
' <<<"$smoke_output"
if [ "$CAPTURE_RC" -ne 0 ]; then
  fail "Linux release algo params invalid" "$(cat <<EOF
Algo params validation failed with exit code $CAPTURE_RC.

Validation output:
$CAPTURE_OUT

Smoke output:
$smoke_output
EOF
)"
fi

case "$suite" in
  all|cpu|gpu|sycl-cpu)
    (cd "$package_dir" && "$node_bin" tests/run_hash.js "$suite") ;;
  *)
    die "Unknown release test suite: $suite" ;;
esac
