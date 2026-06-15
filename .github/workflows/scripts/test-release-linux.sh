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

emit_github_error() {
  local title="$1"
  local message="$2"
  if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    printf '::error title=%s::%s\n' \
      "$(escape_github_message "$title")" \
      "$(escape_github_message "$message")" >&2
  fi
}

rm -rf "$work_dir"
mkdir -p "$work_dir"

archive_list="$(mktemp)"
trap 'rm -f "$archive_list"' EXIT
tar -tzf "$archive" >"$archive_list"

root="$(sed -n '1p' "$archive_list" | cut -d/ -f1)"
if [ -z "$root" ]; then
  echo "Unable to determine archive root for $archive." >&2
  exit 1
fi
if grep -Eq '(^|/)tests(/|$)' "$archive_list"; then
  echo "Release archive must not contain tests/." >&2
  exit 1
fi

tar -C "$work_dir" -xzf "$archive"
package_dir="$work_dir/$root"
libs_dir="$package_dir/libs"
if [ -d "$package_dir/tests" ]; then
  echo "Extracted release package unexpectedly contains tests/." >&2
  exit 1
fi
if [ ! -f "$libs_dir/mom.node" ]; then
  echo "Extracted release package is missing libs/mom.node." >&2
  exit 1
fi

check_ldd() {
  local file output failed=0
  while IFS= read -r -d "" file; do
    output="$(LD_LIBRARY_PATH="$PWD/$libs_dir:$PWD/$package_dir" ldd "$file" 2>&1 || true)"
    if grep -q "not found" <<<"$output"; then
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

set +e
smoke_output="$(cd "$package_dir" && env -u LD_LIBRARY_PATH ./mom algo_params 2>&1)"
smoke_exit=$?
set -e
if [ "$smoke_exit" -ne 0 ]; then
  set +e
  debug_output="$(cd "$package_dir" && env -u LD_LIBRARY_PATH MOM_DEBUG_STARTUP=1 ./mom algo_params 2>&1)"
  debug_exit=$?
  set -e
  smoke_message=$(
    cat <<EOF
Direct executable smoke test failed with exit code $smoke_exit.

Output:
$smoke_output

Debug exit code: $debug_exit
Debug output:
$debug_output
EOF
  )
  emit_github_error "Linux release smoke test failed" "$smoke_message"
  echo "$smoke_message" >&2
  exit 1
fi
if ! grep -q '^MOM_ALGO_PARAMS ' <<<"$smoke_output"; then
  emit_github_error "Linux release smoke test missing marker" "$smoke_output"
  echo "Direct executable smoke test did not print algo params marker." >&2
  echo "$smoke_output" >&2
  exit 1
fi
set +e
validation_output="$(printf '%s\n' "$smoke_output" | "$node_bin" -e '
const fs = require("node:fs");
const marker = fs.readFileSync(0, "utf8").split(/\r?\n/).find((line) => line.startsWith("MOM_ALGO_PARAMS "));
const params = JSON.parse(marker.slice("MOM_ALGO_PARAMS ".length));
for (const [algo, dev] of Object.entries(params)) {
  if (typeof dev !== "string" || !dev || /(?:^|,)[^,]*(?:\*0|\^0)(?:,|$)/.test(dev)) {
    console.error(`Invalid algo params for ${algo}: ${dev}`);
    process.exit(1);
  }
}
' 2>&1)"
validation_exit=$?
set -e
if [ "$validation_exit" -ne 0 ]; then
  validation_message=$(
    cat <<EOF
Algo params validation failed with exit code $validation_exit.

Validation output:
$validation_output

Smoke output:
$smoke_output
EOF
  )
  emit_github_error "Linux release algo params invalid" "$validation_message"
  echo "$validation_message" >&2
  exit 1
fi

case "$suite" in
  all|cpu|gpu|sycl-cpu)
    (cd "$package_dir" && "$node_bin" tests/run_hash.js "$suite") ;;
  *)
    echo "Unknown release test suite: $suite" >&2
    exit 1
    ;;
esac
