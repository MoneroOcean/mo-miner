#!/usr/bin/env bash
set -euo pipefail

archive="${1:?Usage: .github/workflows/scripts/test-release-linux.sh <archive> [suite]}"
suite="${2:-all}"
if [ "$suite" = opencl ]; then
  export MOM_GPU_BACKEND=opencl
  export MOM_OPENCL_DEVICE_TYPE="${MOM_OPENCL_DEVICE_TYPE:-cpu}"
fi
if [ "$suite" = sycl-cpu ]; then
  # Select the packaged standards-only OpenCL worker before the launcher smoke test as well as the
  # vector suite. Otherwise a host with an Intel GPU can make QEMU probe Level Zero/DRM before the
  # CPU-only selector is applied, making this nominally hardware-independent gate abort.
  export MOM_RELEASE_RUNTIME_KEY=dpcpp-opencl
  export MOM_GPU_BACKEND=opencl
  export MOM_OPENCL_DEVICE_TYPE=cpu
  export ONEAPI_DEVICE_SELECTOR=opencl:cpu
fi
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
# Canonicalize once so later runtime paths stay valid whether MOM_RELEASE_TEST_DIR is absolute or
# relative. Prefixing $PWD to an already-absolute work directory produced a nonexistent OpenCL ICD
# path and let the SYCL-CPU package smoke skip instead of exercising the bundled runtime.
package_dir="$(cd "$work_dir/$root" && pwd -P)"
libs_dir="$package_dir/libs"
[ ! -d "$package_dir/tests" ] || die "Extracted release package unexpectedly contains tests/."
[ -f "$libs_dir/mom.node" ] || die "Extracted release package is missing libs/mom.node."
for compiler in oneapi dpcpp dpcpp-opencl acpp-cuda acpp-hip; do
  [ -f "$libs_dir/$compiler/mom.node" ] || die "Release is missing $compiler/mom.node."
done
[ -f "$package_dir/install.sh" ] || die "Release is missing install.sh."
[ -f "$package_dir/install-cutlass.sh" ] || die "Release is missing install-cutlass.sh."
bash -n "$package_dir/install.sh"
bash -n "$package_dir/install-cutlass.sh"
# Exercise the extracted installer once in CI. An explicit no-device selection validates its root
# relaunch, platform detection, and packaged helper loading without installing host GPU packages.
if [ "${GITHUB_ACTIONS:-}" = true ] && [ "$suite" = cpu ]; then
  sudo env MOM_INSTALL_GPU_VENDORS=none bash "$package_dir/install.sh"
fi
grep -aq 'Failed to load libOpenCL.so.1' "$libs_dir/dpcpp/libsycl.so.9" ||
  die "Release DPC++ runtime is missing its static Unified Runtime OpenCL adapter."

check_ldd() {
  local file dir relative compiler compiler_root output failed=0
  while IFS= read -r -d "" file; do
    dir="$(dirname "$file")"
    relative="${file#"$libs_dir/"}"; compiler="${relative%%/*}"; compiler_root="$libs_dir/$compiler"
    output="$(LD_LIBRARY_PATH="$compiler_root:$compiler_root/hipSYCL:$dir:$libs_dir:$package_dir" ldd "$file" 2>&1 || true)"
    # `ldd` can resolve a library by name yet still reject it because the package was built on a
    # newer distribution. Treat symbol-version failures as hard packaging defects; checking only
    # "not found" previously let an Ubuntu 26 libtinfo/libsycl closure reach Ubuntu 24 releases.
    if grep -Eq 'version .*(GLIBC|GLIBCXX|CXXABI)_[0-9.]+.*not found' <<<"$output"; then
      echo "$output" >&2
      failed=1
    fi
    # The unified package's CUDA UR adapter links the user-provided driver libs libcuda.so.1 /
    # libnvidia-ml.so.1. These are intentionally NOT bundled (see the base_libs list in
    # package-linux-combined.sh: the driver supplies them) and are absent on a GPU-less CI runner,
    # so their "not found" is expected, not a packaging defect — ignore those lines.
    if grep -vE '^[[:space:]]*(libcuda|libnvidia-ml|libOpenCL|libze_loader|libamdhip64|libhiprtc|libhsa-runtime64|libhsakmt|libdrm|libnuma)\.so' <<<"$output" | grep -q "not found"; then
      echo "$output" >&2
      failed=1
    fi
  done < <(
    find "$package_dir" "$libs_dir" -maxdepth 4 -type f \
      \( -name "mom-bin" -o -name "mom.node" -o -name "*.so" -o -name "*.so.*" \) \
      -print0
  )
  return "$failed"
}

check_ldd

# GitHub's x64 runners do not promise a CPU vendor, while Intel's OpenCL CPU runtime intentionally
# enumerates only Intel CPUs. QEMU user-mode gives the exact packaged executable an Intel CPUID on
# any x86-64 host; the kernels and bundled runtime remain unchanged, and Haswell is old enough to
# keep this a conservative portability gate. The wrapper is created only in the extracted test
# directory after dependency closure was checked, never in the release archive itself.
if [ "${MOM_RELEASE_EMULATE_INTEL_CPU:-0}" = 1 ]; then
  command -v qemu-x86_64 >/dev/null ||
    die "MOM_RELEASE_EMULATE_INTEL_CPU=1 requires qemu-x86_64."
  mv "$package_dir/mom-bin" "$package_dir/mom-bin.real"
  cat >"$package_dir/mom-bin" <<'EOF'
#!/usr/bin/env sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd -P)
exec env QEMU_CPU="${QEMU_CPU:-Haswell-noTSX-IBRS}" \
  qemu-x86_64 "$script_dir/mom-bin.real" "$@"
EOF
  chmod 0755 "$package_dir/mom-bin"
fi

cp -r tests "$package_dir/"
mkdir -p "$package_dir/scripts"
cp scripts/validate-portable-opencl.js "$package_dir/scripts/"

system_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PATH="$package_dir:$system_path"
# Point the generic OpenCL CPU gate at the archive's Intel CPU ICD explicitly. OCL_ICD_FILENAMES is
# not implemented by every system loader; OCL_ICD_VENDORS with a private one-line manifest is. This
# also prevents unrelated host GPU ICDs from making a CPU-only CI gate appear to pass accidentally.
if { [ "$suite" = opencl ] && [ "${MOM_OPENCL_DEVICE_TYPE:-gpu}" = cpu ]; } ||
   [ "$suite" = sycl-cpu ]; then
  # Test workers run with the extracted package as cwd, so the loader needs an absolute vendor
  # directory. A relative MOM_RELEASE_TEST_DIR otherwise works for the launcher smoke but becomes a
  # nonexistent package-relative path when tests spawn their own miner processes.
  icd_dir="$(cd "$work_dir" && pwd -P)/opencl-icd"
  mkdir -p "$icd_dir"
  printf '%s\n' "$(readlink -f "$libs_dir/oneapi/libintelocl.so")" >"$icd_dir/intel64.icd"
  export OCL_ICD_VENDORS="$icd_dir"
  unset OCL_ICD_FILENAMES
elif [ -z "${OCL_ICD_FILENAMES:-}" ] && [ -f "$libs_dir/oneapi/libintelocl.so" ]; then
  export OCL_ICD_FILENAMES="$libs_dir/oneapi/libintelocl.so"
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
# shellcheck disable=SC2016 # JavaScript is intentionally single-quoted shell data
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
if { [ "$suite" = gpu ] || { [ "$suite" = opencl ] && [ "${MOM_OPENCL_DEVICE_TYPE:-gpu}" != cpu ]; }; } &&
   ! grep -Eq ':"gpu[0-9]+' <<<"$smoke_output"; then
  fail "Linux release GPU discovery missing" \
    "The $suite suite requires launcher-time GPU discovery, but algo_params returned no GPU job."
fi

case "$suite" in
  all|cpu|gpu|sycl-cpu|opencl)
    if [ "$suite" = gpu ]; then export MOM_REQUIRE_GPU_TESTS=1; fi
    if [ "$suite" = sycl-cpu ]; then
      export MOM_REQUIRE_SYCL_CPU_TESTS=1
      # Exercise every CPU-sized GPU algorithm vector from the extracted archive. These cases avoid
      # production-size DAGs but prove that the complete portable kernel set and runtime closure JIT.
      # Use the standards-only SPIR-V worker: unlike AdaptiveCpp's OpenMP backend it has the same
      # semantics as the generic OpenCL deployment path and passes the complete vector set.
    fi
    if [ "$suite" = opencl ] && [ "${MOM_OPENCL_DEVICE_TYPE:-gpu}" = cpu ]; then
      export MOM_REQUIRE_SYCL_CPU_TESTS=1
    fi
    (cd "$package_dir" && "$node_bin" tests/run_hash.js "$suite") ;;
  *)
    die "Unknown release test suite: $suite" ;;
esac
