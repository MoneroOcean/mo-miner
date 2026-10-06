#!/usr/bin/env bash
set -euo pipefail

archive="${1:?Usage: .github/workflows/scripts/test-release-linux.sh <archive> [suite]}"
suite="${2:-all}"
case "$suite" in
  all|cpu|gpu|gpu-discrete|gpu-integrated|gpu-multi|gpu-portable-cpu) ;;
  *) echo "Unknown release test suite: $suite" >&2; exit 2 ;;
esac
# Correctness/archive validation never needs privileged RandomX MSR tuning. Disable it here rather
# than relying on each caller so unprivileged hosts skip the optimization without noisy diagnostics.
export MOM_SKIP_MSR=1
if [ "$suite" = gpu-portable-cpu ]; then
  # Select the packaged standards-only OpenCL worker before the launcher smoke test as well as the
  # vector suite. Otherwise a host with an Intel GPU can make QEMU probe Level Zero/DRM before the
  # CPU-only selector is applied, making this nominally hardware-independent gate abort.
  export MOM_RELEASE_RUNTIME_KEY=dpcpp-opencl
  export MOM_GPU_BACKEND=opencl
  export MOM_OPENCL_DEVICE_TYPE=cpu
  export ONEAPI_DEVICE_SELECTOR=opencl:cpu
fi
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(CDPATH= cd -- "$script_dir/../../.." && pwd -P)"
node_bin="${NODE_BIN:-$(command -v node)}"
work_dir="$(realpath -m -- "${MOM_RELEASE_TEST_DIR:-mom-release-test}")"
if [[ ! ${work_dir##*/} =~ ^mom-release-[A-Za-z0-9_.-]+$ ]]; then
  echo "MOM_RELEASE_TEST_DIR must name a dedicated mom-release-* directory" >&2
  exit 2
fi
workspace_marker="$work_dir/.mom-release-test-workspace"

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

version="$("$node_bin" -e 'process.stdout.write(require(process.argv[1]).version)' "$repo_root/package.json")"
case "$version" in
  ''|*[!0-9A-Za-z.-]*) die "Invalid package version: $version" ;;
esac
expected_root="mom-v$version"
if [[ -L "$archive" || ! -f "$archive" ]]; then
  die "Release archive must be a regular file."
fi
archive="$(realpath -- "$archive")" || die "Unable to resolve release archive."
if [[ -L "$archive" || ! -f "$archive" ]]; then
  die "Release archive must be a regular file."
fi
cd -- "$repo_root"

private_archive="$work_dir/.mom-release-input.tgz"
case "$archive" in
  "$work_dir"|"$work_dir"/*) die "Release archive must be outside the test workspace." ;;
esac

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

if [[ -e "$work_dir" || -L "$work_dir" ]]; then
  if [[ ! -d "$work_dir" || -L "$work_dir" ]]; then
    echo "MOM_RELEASE_TEST_DIR must be an owned directory" >&2
    exit 2
  fi
  if [[ ! -f "$workspace_marker" || -L "$workspace_marker" ]] ||
     [[ $(<"$workspace_marker") != 'mom release test workspace' ]]; then
    echo "MOM_RELEASE_TEST_DIR exists without the .mom-release-test-workspace marker" >&2
    exit 2
  fi
  rm -rf -- "$work_dir"
fi
mkdir -p -m 700 -- "$work_dir"
printf '%s\n' 'mom release test workspace' >"$workspace_marker"
cp -- "$archive" "$private_archive"
archive="$private_archive"

archive_list="$(mktemp)"
trap 'rm -f "$archive_list"' EXIT
tar -tzf "$archive" >"$archive_list"
if grep -Eq '(^/|(^|/)\.\.(/|$))' "$archive_list"; then
  die "Release archive contains an unsafe path."
fi
root="$expected_root"
if grep -Eq '(^|/)tests(/|$)' "$archive_list"; then
  die "Release archive must not contain tests/."
fi
grep -Eq '(^|/)DEVELOPMENT\.md$' "$archive_list" &&
  die "Release archive must not contain DEVELOPMENT.md."

python3 - "$archive" "$work_dir" "$root" <<'PY'
import os
import sys
import posixpath
import tarfile
from pathlib import PurePosixPath

archive, destination, expected_root = sys.argv[1:]

def normalize_hardlink(linkname):
    raw = linkname.replace("\\", "/")
    if not raw or raw.startswith("/") or ":" in raw:
        raise tarfile.FilterError("hardlink target is absolute or noncanonical")
    candidate = raw if raw == expected_root or raw.startswith(expected_root + "/") \
        else f"{expected_root}/{raw}"
    candidate = posixpath.normpath(candidate)
    if candidate == expected_root or not candidate.startswith(expected_root + "/"):
        raise tarfile.FilterError("hardlink target escapes the package root")
    return candidate

try:
    with tarfile.open(archive, "r:gz") as payload:
        members = payload.getmembers()
        seen = set()
        safe_members = []
        has_root_entry = False
        for member in members:
            raw_name = member.name
            name = raw_name.rstrip("/")
            parts = PurePosixPath(name).parts
            if (not name or raw_name.startswith("/") or "\\" in raw_name or
                    "//" in raw_name or
                    any(part in (".", "..") for part in raw_name.split("/")) or
                    any(part in (".", "..") for part in parts)):
                raise tarfile.FilterError(f"noncanonical member: {raw_name!r}")
            if not parts or parts[0] != expected_root:
                raise tarfile.FilterError(
                    f"member is outside the expected package root: {raw_name!r}"
                )
            if name in seen:
                raise tarfile.FilterError(f"duplicate member: {raw_name!r}")
            seen.add(name)
            if name == expected_root:
                if not member.isdir():
                    raise tarfile.FilterError("archive must contain an explicit package root directory")
                has_root_entry = True
            if member.issym() or not (member.isfile() or member.isdir() or member.islnk()):
                raise tarfile.FilterError(
                    f"unsupported member type for {raw_name!r}"
                )
            if member.islnk():
                member = member.replace(linkname=normalize_hardlink(member.linkname), deep=False)
            filtered = tarfile.data_filter(member, destination)
            if filtered is None:
                raise tarfile.FilterError(f"archive member was filtered: {raw_name!r}")
            safe_members.append(filtered)
        if not has_root_entry:
            raise tarfile.FilterError("archive must contain an explicit package root directory")
        if os.environ.get("MOM_RELEASE_ARCHIVE_VALIDATION_ONLY", "0") != "1":
            payload.extractall(destination, members=safe_members, filter="data")
except (OSError, tarfile.TarError, ValueError) as error:
    raise SystemExit(f"Release archive contains an unsafe member: {error}") from error
PY
# Focused archive tests set this after the structural checks; release CI leaves it unset.
if [ "${MOM_RELEASE_ARCHIVE_VALIDATION_ONLY:-0}" = 1 ]; then
  exit 0
fi
# Canonicalize once so later runtime paths stay valid whether MOM_RELEASE_TEST_DIR is absolute or
# relative. Prefixing $PWD to an already-absolute work directory produced a nonexistent OpenCL ICD
# path and let the SYCL-CPU package smoke skip instead of exercising the bundled runtime.
package_dir="$(cd "$work_dir/$root" && pwd -P)"
libs_dir="$package_dir/libs"
for sidecar in kawpow_device.inc kawpow_keccak.inc; do
  [ -s "$libs_dir/dpcpp/$sidecar" ] ||
    die "Release is missing nonempty dpcpp/$sidecar."
done
[ ! -d "$package_dir/tests" ] || die "Extracted release package unexpectedly contains tests/."
[ -f "$package_dir/GPU-CONFIG.md" ] || die "Release is missing GPU-CONFIG.md."
[ -f "$package_dir/gpu-tuning.js" ] || die "Release is missing gpu-tuning.js."
[ -f "$package_dir/helper/hash.js" ] || die "Release is missing helper/hash.js."
[ ! -e "$package_dir/DEVELOPMENT.md" ] || die "Release unexpectedly contains DEVELOPMENT.md."
[ -f "$libs_dir/mom.node" ] || die "Extracted release package is missing libs/mom.node."
for compiler in oneapi dpcpp dpcpp-opencl acpp-cuda acpp-hip; do
  [ -f "$libs_dir/$compiler/mom.node" ] || die "Release is missing $compiler/mom.node."
done
[ -f "$libs_dir/dpcpp/libsycl-jit.so" ] || die "Release is missing dpcpp/libsycl-jit.so."
for tool in opt llc lld ld.lld; do
  [ -x "$libs_dir/acpp-cuda/hipSYCL/ext/llvm/bin/$tool" ] ||
    die "Release is missing executable acpp-cuda/hipSYCL/ext/llvm/bin/$tool."
  [ -x "$libs_dir/acpp-hip/hipSYCL/ext/llvm/bin/$tool" ] ||
    die "Release is missing executable acpp-hip/hipSYCL/ext/llvm/bin/$tool."
done
[ -x "$libs_dir/acpp-hip/hipSYCL/ext/llvm/bin/ld.lld-20" ] ||
  die "Release is missing executable acpp-hip/hipSYCL/ext/llvm/bin/ld.lld-20."
cmp -s "$libs_dir/acpp-hip/hipSYCL/ext/llvm/bin/ld.lld" \
  "$libs_dir/acpp-hip/hipSYCL/ext/llvm/bin/ld.lld-20" ||
  die "Release HIP linker alias differs from the staged LLVM linker."
for bitcode in \
  acpp-cuda/hipSYCL/bitcode/libkernel-sscp-host-full.bc \
  acpp-cuda/hipSYCL/bitcode/libkernel-sscp-ptx-full.bc \
  acpp-cuda/hipSYCL/ext/bitcode/ptx/libdevice.10.bc \
  acpp-hip/hipSYCL/bitcode/libkernel-sscp-host-full.bc \
  acpp-hip/hipSYCL/bitcode/libkernel-sscp-amdgpu-amdhsa-full.bc \
  acpp-hip/hipSYCL/ext/bitcode/amdgcn/ockl.bc; do
  [ -f "$libs_dir/$bitcode" ] || die "Release is missing $bitcode."
done
if [ -n "$(find "$libs_dir/acpp-hip" \( -type f -o -type l \) \
  \( -name 'libamdhip64.so*' -o -name 'libhiprtc.so*' -o -name 'libamd_comgr.so*' \
     -o -name 'libhsa-runtime64.so*' -o -name 'libhsakmt.so*' \) -print -quit)" ]; then
  die "Release must not bundle a partial host ROCm runtime."
fi
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
    if grep -vE '^[[:space:]]*(libcuda|libnvidia-ml|libOpenCL|libze_loader|libamdhip64|libhiprtc|libamd_comgr|libhsa-runtime64|libhsakmt|libdrm|libnuma)\.so' <<<"$output" | grep -q "not found"; then
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
  cat >"$package_dir/mom-qemu-preload.cjs" <<'EOF'
"use strict";

const path = require("node:path");
// QEMU does not follow exec: route Node's worker launches through the emulator wrapper too.
process.execPath = path.join(__dirname, "mom-bin");
EOF
  chmod 0644 "$package_dir/mom-qemu-preload.cjs"
cat >"$package_dir/mom-bin" <<'EOF'
#!/usr/bin/env sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd -P)
# Worker processes inherit this marker. Re-enter QEMU directly so their IPC/stdout streams and
# native exit status remain transparent; only the top-level wrapper needs teardown normalization.
if [ "${MOM_RELEASE_QEMU_CHILD:-0}" = 1 ]; then
  exec env QEMU_CPU="${QEMU_CPU:-Haswell-noTSX-IBRS}" \
    qemu-x86_64 "$script_dir/mom-bin.real" "$@"
fi
stdout=$(mktemp)
stderr=$(mktemp)
trap 'rm -f "$stdout" "$stderr"' EXIT
preload_path="$script_dir/mom-qemu-preload.cjs"
preload_quoted=$(printf '%s' "$preload_path" | sed 's/\\/\\\\/g; s/"/\\"/g')
set +e
env QEMU_CPU="${QEMU_CPU:-Haswell-noTSX-IBRS}" \
  MOM_RELEASE_QEMU_CHILD=1 \
  NODE_OPTIONS="${NODE_OPTIONS:-} --require \"$preload_quoted\"" \
  qemu-x86_64 "$script_dir/mom-bin.real" "$@" >"$stdout" 2>"$stderr"
status=$?
set -e
cat "$stdout"
cat "$stderr" >&2
# Intel's OpenCL CPU runtime can deliver SIGSEGV while QEMU tears it down after the process has
# already completed and reported a valid vector. Normalize only that exact emulator-only outcome;
# a signal before PASSED, any other exit, or any native run remains a hard test failure.
if [ "$status" -eq 139 ] &&
   grep -qx 'PASSED' "$stdout" &&
   grep -q 'qemu: uncaught target signal 11 (Segmentation fault)' "$stderr"; then
  exit 0
fi
exit "$status"
EOF
  chmod 0755 "$package_dir/mom-bin"
fi

cp -r tests "$package_dir/"
mkdir -p "$package_dir/scripts"
cp scripts/validate-portable-opencl.js scripts/windows-command.js "$package_dir/scripts/"

system_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PATH="$package_dir:$system_path"
# Point the generic OpenCL CPU gate at the archive's Intel CPU ICD explicitly. OCL_ICD_FILENAMES is
# not implemented by every system loader; OCL_ICD_VENDORS with a private one-line manifest is. This
# also prevents unrelated host GPU ICDs from making a CPU-only CI gate appear to pass accidentally.
if [ "$suite" = gpu-portable-cpu ]; then
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
run_mom() { (cd "$package_dir" && env -u LD_LIBRARY_PATH "$@" ./mom algorithms); }

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
if ! grep -q '^MOM_ALGORITHMS ' <<<"$smoke_output"; then
  fail "Linux release smoke test missing marker" "$(printf '%s\n%s' \
    "Direct executable smoke test did not print algorithms marker." "$smoke_output")"
fi
# Validate that every algo advertises a usable device string (non-empty and not
# a disabled "*0"/"^0" entry).
# shellcheck disable=SC2016 # JavaScript is intentionally single-quoted shell data
capture "$node_bin" -e '
const fs = require("node:fs");
const marker = fs.readFileSync(0, "utf8").split(/\r?\n/).find((line) => line.startsWith("MOM_ALGORITHMS "));
const params = JSON.parse(marker.slice("MOM_ALGORITHMS ".length));
for (const [algo, dev] of Object.entries(params)) {
  if (typeof dev !== "string" || !dev || /(?:^|,)[^,]*(?:\*0|\^0)(?:,|$)/.test(dev)) {
    console.error(`Invalid algorithms entry for ${algo}: ${dev}`);
    process.exit(1);
  }
}
' <<<"$smoke_output"
if [ "$CAPTURE_RC" -ne 0 ]; then
  fail "Linux release algorithms invalid" "$(cat <<EOF
Algorithms validation failed with exit code $CAPTURE_RC.

Validation output:
$CAPTURE_OUT

Smoke output:
$smoke_output
EOF
)"
fi
if [[ "$suite" = gpu || "$suite" = gpu-discrete || "$suite" = gpu-integrated ||
      "$suite" = gpu-multi ]] &&
  ! grep -Eq ':"gpu[0-9]+' <<<"$smoke_output"; then
  fail "Linux release GPU discovery missing" \
    "The $suite suite requires launcher-time GPU discovery, but algorithms returned no GPU job."
fi
if [ "$suite" = gpu-integrated ] &&
   ! grep -Eiq '^gpu[0-9]+: .*Intel.*\[integrated\]$' <<<"$smoke_output"; then
  fail "Linux release integrated GPU discovery missing" \
    "The gpu-integrated suite requires an Intel integrated GPU, but algorithms reported none."
fi

if [[ "$suite" = gpu || "$suite" = gpu-discrete || "$suite" = gpu-integrated ||
      "$suite" = gpu-multi ]]; then
  export MOM_REQUIRE_GPU_TESTS=1
fi
if [ "$suite" = gpu-integrated ]; then
  export MOM_REQUIRE_INTEGRATED_GPU_TESTS=1
fi
if [ "$suite" = gpu-multi ]; then
  export MOM_REQUIRE_MULTI_GPU_TESTS=1
fi
if [ "$suite" = gpu-portable-cpu ]; then
  export MOM_REQUIRE_PORTABLE_CPU_TESTS=1
  # Exercise every CPU-sized GPU algorithm vector from the extracted archive. These cases avoid
  # production-size DAGs but prove that the complete portable kernel set and runtime closure JIT.
  # Use the standards-only SPIR-V worker: unlike AdaptiveCpp's OpenMP backend it has the same
  # semantics as the generic OpenCL deployment path and passes the complete vector set.
fi
if [ "$suite" = cpu ]; then
  (cd "$package_dir" && env \
    MOM_NATIVE_PATH="$libs_dir/oneapi/mom.node" \
    LD_LIBRARY_PATH="$libs_dir/oneapi" \
    "$node_bin" tests/run_hash.js "$suite")
else
  (cd "$package_dir" && "$node_bin" tests/run_hash.js "$suite")
fi
