#!/usr/bin/env bash
set -euo pipefail
exec 3>&2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="${MOM_DEPLOY_TARGET:-all}"
DEPLOY_SKIP_VECTORS="${MOM_DEPLOY_SKIP_VECTORS:-0}"
WIN_RUN="${WIN_RUN:-$HOME/win/run.sh}"
WIN_MOM_DEV_BASE="${WIN_MOM_DEV_BASE:-$HOME/cache/win/images/win-mom-dev-base.qcow2}"
SUDO_PASSWORD="${SUDO_PASSWORD:-}"
RELEASE_VERSION="${MOM_RELEASE_VERSION:-$(node -p "require('./package.json').version")}"
RELEASE_VERSION="${RELEASE_VERSION#[vV]}"
if [[ ! "$RELEASE_VERSION" =~ ^[0-9][0-9A-Za-z.-]*$ ]]; then
  echo "Invalid release version: $RELEASE_VERSION" >&2
  exit 2
fi
RELEASE_DIR="mom-v${RELEASE_VERSION}"
LINUX_ARCHIVE="${RELEASE_DIR}-lin.tgz"
WINDOWS_ARCHIVE="${RELEASE_DIR}-win.zip"
WINDOWS_STAGE="build/deploy-win"
declare -a DEPLOY_RESULTS=()
declare -a TEST_COUNTS=(0 0 0 0)
declare -a REQUESTED_LANES=()
declare -A RECORDED_RESULTS=()
DEPLOY_SKIP_REASON=""
ACTIVE_LANE_PGID=""
INTERRUPTED_STATUS=0
# Give win/run.sh enough time to tear down a VM and its mounts before escalation.
INTERRUPT_GRACE_SECONDS=60

case "$TARGET" in
  all) REQUESTED_LANES=(
    nvidia-linux intel-linux amd-linux nvidia-windows intel-windows amd-windows multi-windows
  ) ;;
  linux) REQUESTED_LANES=(nvidia-linux intel-linux amd-linux) ;;
  windows) REQUESTED_LANES=(nvidia-windows intel-windows amd-windows multi-windows) ;;
  linux-nvidia|linux-intel|linux-amd) REQUESTED_LANES=("${TARGET#linux-}-linux") ;;
  windows-nvidia|windows-intel|windows-amd) REQUESTED_LANES=("${TARGET#windows-}-windows") ;;
  *) echo "Unknown MOM_DEPLOY_TARGET: $TARGET" >&2; exit 2 ;;
esac
case "$DEPLOY_SKIP_VECTORS" in 0|1) ;; *) echo "MOM_DEPLOY_SKIP_VECTORS must be 0 or 1" >&2; exit 2;; esac

skip() {
  DEPLOY_SKIP_REASON="$*"
  printf '  ➖ %s\n' "$*"
}

record_result() {
  local platform="$1" status="$2" detail="${3:-}" synthetic="${4:-0}" icon counter
  case "$status" in PASS) icon=✔;; FAIL) icon=✖;; SKIP) icon=➖;; esac
  DEPLOY_RESULTS+=("  $icon $platform${detail:+ ($detail)}")
  RECORDED_RESULTS["$platform"]=1
  if (( synthetic )); then
    TEST_COUNTS[0]=$((TEST_COUNTS[0] + 1))
    case "$status" in PASS) counter=1;; FAIL) counter=2;; SKIP) counter=3;; esac
    TEST_COUNTS[$counter]=$((TEST_COUNTS[counter] + 1))
  fi
}

interrupt() {
  local signal="$1" status="$2" i
  (( INTERRUPTED_STATUS )) && return
  INTERRUPTED_STATUS="$status"
  printf '\n  ✖ %s received; stopping active deployment phase\n' "$signal" >&3
  if [ -z "$ACTIVE_LANE_PGID" ]; then
    record_pending_interrupted
    exit "$status"
  fi
  kill -s "$signal" -- "-$ACTIVE_LANE_PGID" 2>/dev/null || true
  for ((i=0; i<INTERRUPT_GRACE_SECONDS; i++)); do
    kill -0 -- "-$ACTIVE_LANE_PGID" 2>/dev/null || return
    sleep 1 || true
  done
  kill -KILL -- "-$ACTIVE_LANE_PGID" 2>/dev/null || true
}

run_active() {
  local status
  set +m
  (
    set +m
    export ROOT_DIR TARGET WIN_RUN WIN_MOM_DEV_BASE SUDO_PASSWORD RELEASE_VERSION RELEASE_DIR \
      LINUX_ARCHIVE WINDOWS_ARCHIVE WINDOWS_STAGE DEPLOY_SKIP_VECTORS
    export -f skip has_drm_vendor linux_gpu_available package_linux_release build_linux_release \
      windows_available run_windows_root package_windows_release test_linux_release test_windows_release \
      test_windows_multi_release \
      stream_lane
    trap - INT TERM QUIT
    exec setsid bash -c 'set +e; set -u -o pipefail; "$@"' deploy-active "$@"
  ) & ACTIVE_LANE_PGID=$!
  wait "$ACTIVE_LANE_PGID"; status=$?
  if (( INTERRUPTED_STATUS )); then
    wait "$ACTIVE_LANE_PGID" 2>/dev/null || true
    status=$INTERRUPTED_STATUS
  fi
  ACTIVE_LANE_PGID=""
  return "$status"
}

stream_lane() {
  local log="$1"
  shift
  if [ "$DEPLOY_SKIP_VECTORS" = 1 ]; then
    printf '  ➖ Stability checks skipped (MOM_DEPLOY_SKIP_VECTORS=1)\n' | tee -a "$log"
    printf 'MOM_TEST_SUMMARY 1 0 0 1\n' >>"$log"
  else
    : >"$log"
  fi
  "$@" 2>&1 | tee -a "$log" | awk '
    /^[[:space:]]*✔ Stability checks passed$/ ||
    /^[[:space:]]*[✔✖] [^:]+: .*% of README/ ||
    /^[[:space:]]*✖ [^:]+: (release discovery reported no GPU tuning|no hashrate was reported)$/ ||
    / received; stopping the VM / {print; fflush()}'
}

run_lane() {
  local platform="$1" status=0 log i fails_before reason outcome synthetic summary_seen=0
  local lane_pass=0 lane_fail=0 lane_skip=0
  shift
  DEPLOY_SKIP_REASON=""
  fails_before=${TEST_COUNTS[2]}
  log="$(mktemp)"
  printf '\n▶ %s\n' "$platform"
  set +e
  run_active stream_lane "$log" "$@"
  status=$?
  set -e
  while read -r _ _tests _pass _fail _skip; do
    lane_pass=$((lane_pass + _pass))
    lane_fail=$((lane_fail + _fail))
    lane_skip=$((lane_skip + _skip))
    i=0
    for value in $_tests $_pass $_fail $_skip; do
      TEST_COUNTS[$i]=$((TEST_COUNTS[i] + value))
      i=$((i + 1))
    done
  done < <(grep '^MOM_TEST_SUMMARY ' "$log" || true)
  grep -q '^MOM_TEST_SUMMARY ' "$log" && summary_seen=1 || true
  if (( INTERRUPTED_STATUS )); then
    outcome=FAIL reason=interrupted synthetic="$(( TEST_COUNTS[2] == fails_before ))"
    printf '  ✖ %s (interrupted)\n' "$platform"
  elif (( status != 0 || TEST_COUNTS[2] > fails_before )); then
    reason="exit $status"
    (( status == 0 )) && reason="tests failed"
    outcome=FAIL synthetic="$(( TEST_COUNTS[2] == fails_before ))"
    printf '  ✖ %s (%s)\n' "$platform" "$reason"
  elif (( lane_pass == 0 && lane_fail == 0 && lane_skip > 0 )) ||
       { (( ! summary_seen )) && grep -q '^[[:space:]]*➖ ' "$log"; }; then
    reason="$(sed -n 's/^[[:space:]]*➖ //p' "$log" | tail -1)"
    outcome=SKIP synthetic="$(( ! summary_seen ))"
    printf '  ➖ %s (%s)\n' "$platform" "$reason"
  elif (( ! summary_seen )); then
    outcome=FAIL reason="test summary missing" synthetic=1
    printf '  ✖ %s (%s)\n' "$platform" "$reason"
  else
    outcome=PASS reason="" synthetic="$(( ! summary_seen ))"
    printf '  ✔ %s\n' "$platform"
  fi
  record_result "$platform" "$outcome" "$reason" "$synthetic"
  if (( status != 0 || TEST_COUNTS[2] > fails_before )); then
    awk '!/^MOM_TEST_SUMMARY / && !/^[[:space:]]*✔ Stability checks passed$/ &&
      !/^[[:space:]]*[✔✖] [^:]+: .*% of README/ &&
      !/^[[:space:]]*✖ [^:]+: (release discovery reported no GPU tuning|no hashrate was reported)$/ &&
      !/^[[:space:]]*➖ [^:]+: no GPU is available/ &&
      !/^[[:space:]]*➖ Stability checks skipped / &&
      !/ received; stopping the VM /' "$log" >&2
  fi
  rm -f "$log"
  (( INTERRUPTED_STATUS )) || return 0
  return "$INTERRUPTED_STATUS"
}

quiet_phase() {
  local label="$1" log status=0 reason
  shift; log="$(mktemp)"; printf '\n▶ %s\n' "$label"
  run_active "$@" >"$log" 2>&1 || status=$?
  reason="$(sed -n 's/^[[:space:]]*➖ //p' "$log" | tail -1)"
  [ -z "$reason" ] || DEPLOY_SKIP_REASON="$reason"
  if (( status == 0 )); then
    printf '  ✔ %s\n' "$label"
  elif (( INTERRUPTED_STATUS )); then
    printf '  ✖ %s (interrupted)\n' "$label"
  elif [ -n "$reason" ]; then
    printf '  ➖ %s (%s)\n' "$label" "$reason"
  else
    printf '  ✖ %s (exit %d)\n' "$label" "$status"
    cat "$log" >&2
  fi
  rm -f "$log"
  return "$status"
}

record_unbuilt() {
  local os="$1" status=FAIL reason="$1 release build failed" vendor
  shift
  if [ -n "$DEPLOY_SKIP_REASON" ]; then status=SKIP; reason="$DEPLOY_SKIP_REASON"; fi
  for vendor in "$@"; do
    [ "$status" = SKIP ] && printf '  ➖ %s-%s (%s)\n' "$vendor" "$os" "$reason"
    record_result "$vendor-$os" "$status" "$reason" 1
  done
}

record_pending_interrupted() {
  local platform
  for platform in "${REQUESTED_LANES[@]}"; do
    [ -n "${RECORDED_RESULTS[$platform]+x}" ] ||
      record_result "$platform" FAIL interrupted 1
  done
}

finish() {
  local exit_status=$?
  (( INTERRUPTED_STATUS )) && exit_status=$INTERRUPTED_STATUS
  (( INTERRUPTED_STATUS )) && record_pending_interrupted
  trap - EXIT INT TERM
  printf '\n▶ Deployment summary\n'
  printf '%s\n' "${DEPLOY_RESULTS[@]}"
  printf 'ℹ tests %d\nℹ pass %d\nℹ fail %d\nℹ skipped %d\n' "${TEST_COUNTS[@]}"
  if (( exit_status == 0 )) && { (( TEST_COUNTS[2] > 0 )) ||
    printf '%s\n' "${DEPLOY_RESULTS[@]}" | grep -q '✖'; }; then
    exit_status=1
  fi
  exit "$exit_status"
}
trap finish EXIT
trap 'interrupt INT 130' INT
trap 'interrupt TERM 143' TERM

has_drm_vendor() {
  local expected="${1,,}" vendor
  for vendor in /sys/class/drm/card*/device/vendor; do
    [ -r "$vendor" ] || continue
    [ "$(tr '[:upper:]' '[:lower:]' <"$vendor")" = "$expected" ] && return 0
  done
  return 1
}

package_linux_release() (
  local container="" node=""
  trap '[ -z "$node" ] || rm -f -- "$node"; [ -z "$container" ] || docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
  if NODE_BIN="$(command -v node)" .github/workflows/scripts/package-linux-combined.sh \
      "$RELEASE_VERSION" "$LINUX_ARCHIVE" >/dev/null 2>&1; then
    return 0
  fi
  container="$(docker create --entrypoint sleep mom-build-multicompiler infinity)" || return
  node="$(mktemp /tmp/mom-release-node.XXXXXX)" || return
  docker cp "$container:/usr/local/bin/node" "$node" || return
  chmod +x "$node"
  NODE_BIN="$node" .github/workflows/scripts/package-linux-combined.sh \
    "$RELEASE_VERSION" "$LINUX_ARCHIVE"
)

build_linux_release() {
  if [ "${MOM_DEPLOY_REUSE_ARCHIVE:-0}" = 1 ] && [ -f "$LINUX_ARCHIVE" ]; then
    echo "[deploy] Reusing $LINUX_ARCHIVE"
    return
  fi
  if ! command -v docker >/dev/null; then
    skip "Docker is unavailable; Linux deployment tests need a clean container"
    return 1
  fi
  echo "[deploy] Building Linux release binary"
  MOM_PORTABLE_BUILD=1 MOM_GPU_BACKEND=all ./r.sh true || return
  echo "[deploy] Packaging Linux release archive"
  package_linux_release
}

linux_gpu_available() {
  case "$1" in
    nvidia)
      [ -d /proc/driver/nvidia/gpus ] ||
        { command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; }
      ;;
    intel) has_drm_vendor 0x8086 ;;
    amd) has_drm_vendor 0x1002 && [ -e /dev/kfd ] ;;
  esac
}

test_linux_release() {
  local vendor="$1" platform="$1-linux" evidence_base="" evidence_root=""
  local -a devices=() docker_gpu=() selector=() verthash=() evidence=()
  if ! linux_gpu_available "$vendor"; then
    skip "no usable Linux $vendor GPU"
    return
  fi
  case "$vendor" in
    nvidia)
      docker_gpu=(--gpus all)
      selector=(MOM_GPU_BACKEND=nvidia)
      ;;
    intel)
      devices=(--device=/dev/dri:/dev/dri)
      selector=(MOM_GPU_BACKEND=intel ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0)
      ;;
    amd)
      devices=(--device=/dev/kfd --device=/dev/dri --group-add video)
      selector=(MOM_GPU_BACKEND=amd)
      ;;
  esac
  if [ -n "${MOM_VERTHASH_DATA:-}" ]; then
    [ -f "$MOM_VERTHASH_DATA" ] || {
      echo "MOM_VERTHASH_DATA is not a regular file: $MOM_VERTHASH_DATA" >&2
      return 1
    }
    verthash=(-v "$MOM_VERTHASH_DATA:/verthash.dat:ro" -e MOM_VERTHASH_DATA=/verthash.dat)
  fi
  if [ -n "${MOM_RELEASE_PERF_EVIDENCE_DIR:-}" ]; then
    if [ -L "$MOM_RELEASE_PERF_EVIDENCE_DIR" ]; then
      echo "MOM_RELEASE_PERF_EVIDENCE_DIR must not be a symlink" >&2
      return 1
    fi
    evidence_base="$(realpath -m -- "$MOM_RELEASE_PERF_EVIDENCE_DIR")" || return
    if [[ "$evidence_base" == *:* ]] ||
       { [ -e "$evidence_base" ] && [ ! -d "$evidence_base" ]; }; then
      echo "MOM_RELEASE_PERF_EVIDENCE_DIR must be an owned directory without ':'" >&2
      return 1
    fi
    mkdir -p -m 700 -- "$evidence_base" || return
    if [ ! -O "$evidence_base" ]; then
      echo "MOM_RELEASE_PERF_EVIDENCE_DIR must be owned by the current user" >&2
      return 1
    fi
    chmod 700 -- "$evidence_base" || return
    evidence_root="$evidence_base/$platform"
    if [ -L "$evidence_root" ]; then
      echo "Platform performance evidence directory must not be a symlink" >&2
      return 1
    fi
    mkdir -p -m 700 -- "$evidence_root" || return
    if [ ! -O "$evidence_root" ]; then
      echo "Platform performance evidence directory must be owned by the current user" >&2
      return 1
    fi
    chmod 700 -- "$evidence_root" || return
    evidence=(-v "$evidence_root:/mom-perf-evidence"
      -e MOM_RELEASE_PERF_EVIDENCE_DIR=/mom-perf-evidence
      -e MOM_RELEASE_PERF_EVIDENCE_UID="$(id -u)"
      -e MOM_RELEASE_PERF_EVIDENCE_GID="$(id -g)")
  fi

  echo "[deploy] Testing every $vendor GPU vector and README performance gate in clean Ubuntu 26.04"
  docker run --rm "${docker_gpu[@]}" "${devices[@]}" "${verthash[@]}" "${evidence[@]}" \
    -v "$ROOT_DIR:/repo:ro" \
    -e MOM_RELEASE_ARCHIVE="$LINUX_ARCHIVE" -e MOM_RELEASE_DIR="$RELEASE_DIR" \
    -e MOM_SKIP_MSR=1 -e MOM_GPU_TEST_ALGO="${MOM_DEPLOY_ALGO:-}" \
    -e MOM_DEPLOY_ALGO="${MOM_DEPLOY_ALGO:-}" \
    ubuntu:26.04 bash -lc '
      set -euo pipefail
      export DEBIAN_FRONTEND=noninteractive
      apt-get update >/dev/null
      apt-get install -y --no-install-recommends ca-certificates nodejs python3 >/dev/null
      cd /tmp
      tar -xzf "/repo/$MOM_RELEASE_ARCHIVE"
      cd "$MOM_RELEASE_DIR"
      MOM_INSTALL_GPU_VENDORS="'"$vendor"'" ./install.sh >/tmp/mom-install.log
      export '"${selector[*]}"'
      export MOM_GPU_TEST_VENDORS="'"$vendor"'"
      export MOM_REQUIRE_GPU_TESTS=1
      export MOM_BENCHMARK_GRACEFUL_ONLY=1
      release_path="$PWD"
      cd /repo
      performance_args=()
      if [ -n "${MOM_DEPLOY_ALGO:-}" ]; then
        performance_args+=(--algo "$MOM_DEPLOY_ALGO")
      fi
      if [ "'"$DEPLOY_SKIP_VECTORS"'" != 1 ]; then
        vector_log=$(mktemp); status=0
        NODE_BIN="$(command -v node)" MOM_RELEASE_TEST_DIR=/tmp/mom-release-vectors \
          .github/workflows/scripts/test-release-linux.sh "$MOM_RELEASE_ARCHIVE" gpu-discrete \
          >"$vector_log" 2>&1 || status=$?
        awk '\''/^ *ℹ (tests|pass|fail|skipped) [0-9]+$/ {v[$2]=$3} END {
          if (v["tests"]) print "MOM_TEST_SUMMARY",v["tests"],v["pass"],v["fail"],v["skipped"]
        }'\'' "$vector_log"
        if [ "$status" = 0 ]; then echo "  ✔ Stability checks passed"
        else cat "$vector_log"; exit "$status"; fi
        rm -f "$vector_log"
      fi
      performance_status=0
      node scripts/check-release-performance.js \
        --miner "$release_path/mom" --readme README.md \
        --platform "'"$platform"'" --margin 0.05 "${performance_args[@]}" || performance_status=$?
      if [ -n "${MOM_RELEASE_PERF_EVIDENCE_DIR:-}" ]; then
        chown -R "$MOM_RELEASE_PERF_EVIDENCE_UID:$MOM_RELEASE_PERF_EVIDENCE_GID" \
          "$MOM_RELEASE_PERF_EVIDENCE_DIR"
      fi
      exit "$performance_status"
    '
}

windows_available() {
  [ -x "$WIN_RUN" ] && return 0
  skip "$WIN_RUN is unavailable; Windows deployment tests are optional"
  return 1
}

run_windows_root() {
  if [ -n "$SUDO_PASSWORD" ]; then
    printf '%s\n' "$SUDO_PASSWORD" | sudo -S -p '' -- "$@"
  else
    sudo -n -- "$@"
  fi
}

package_windows_release() {
  windows_available || return 1
  if [ "${MOM_DEPLOY_REUSE_ARCHIVE:-0}" = 1 ] && [ -f "$WINDOWS_ARCHIVE" ]; then
    echo "[deploy] Reusing $WINDOWS_ARCHIVE"
    return
  fi
  echo "[deploy] Building and packaging Windows release archive"
  run_windows_root env GPU_GROUP=none WIN_MOM_RUN_BASE="$WIN_MOM_DEV_BASE" "$WIN_RUN" \
    --download build/win --download "$WINDOWS_ARCHIVE" -- \
    powershell -NoProfile -ExecutionPolicy Bypass -Command \
    'npm install --ignore-scripts; if ($LASTEXITCODE) { exit $LASTEXITCODE }; & .github\workflows\scripts\build-windows-multicompiler.ps1; if ($LASTEXITCODE) { exit $LASTEXITCODE }; & .github\workflows\scripts\package-windows.ps1; if ($LASTEXITCODE) { exit $LASTEXITCODE }' || return
  run_windows_root chown -R "$(id -u):$(id -g)" build/win "$WINDOWS_ARCHIVE"
}

test_windows_release() {
  local vendor="$1" group="$1" platform="$1-windows"
  [ "$vendor" = intel ] && group=arc
  mkdir -p "$WINDOWS_STAGE"
  cp -f "$WINDOWS_ARCHIVE" "$WINDOWS_STAGE/$WINDOWS_ARCHIVE"
  echo "[deploy] Testing every Windows $vendor GPU vector and README performance gate"
  run_windows_root env GPU_GROUP="$group" "$WIN_RUN" \
    --release --download "$WINDOWS_STAGE" -- \
    env MOM_SKIP_MSR=1 MOM_DEPLOY_SKIP_VECTORS="$DEPLOY_SKIP_VECTORS" \
      MOM_DEPLOY_ALGO="${MOM_DEPLOY_ALGO:-}" \
      MOM_GPU_TEST_ALGO="${MOM_DEPLOY_ALGO:-}" \
      MOM_KAWPOW_DAG_CHUNK_NODES="${MOM_KAWPOW_DAG_CHUNK_NODES:-}" \
    powershell -NoProfile -ExecutionPolicy Bypass -File \
      scripts\\test-windows-release-gpu.ps1 \
      -Archive "${WINDOWS_STAGE//\//\\}\\$WINDOWS_ARCHIVE" -Platform "$platform"
}

test_windows_multi_release() {
  if [ "$DEPLOY_SKIP_VECTORS" = 1 ]; then
    skip "Windows mixed-vendor correctness skipped with the vector gates"
    return
  fi
  if [ -n "${MOM_DEPLOY_ALGO:-}" ]; then
    skip "Windows mixed-vendor correctness is outside a single-algorithm deployment"
    return
  fi
  mkdir -p "$WINDOWS_STAGE"
  cp -f "$WINDOWS_ARCHIVE" "$WINDOWS_STAGE/$WINDOWS_ARCHIVE"
  echo "[deploy] Testing concurrent NVIDIA and AMD workers from the Windows release"
  run_windows_root env GPU_GROUP=all "$WIN_RUN" \
    --release --download "$WINDOWS_STAGE" -- \
    env MOM_SKIP_MSR=1 MOM_GPU_BACKEND=nvidia MOM_GPU_TEST_VENDORS=nvidia,amd \
      MOM_REQUIRE_MULTI_GPU_TESTS=1 \
    powershell -NoProfile -ExecutionPolicy Bypass -File \
      .github\\workflows\\scripts\\test-release-windows.ps1 \
      -Archive "${WINDOWS_STAGE//\//\\}\\$WINDOWS_ARCHIVE" -Suite gpu-multi || return
  printf 'MOM_TEST_SUMMARY 1 1 0 0\n'
}

run_platform() {
  local os="$1" build="$2" test="$3" i
  shift 3
  local -a vendors=("$@")
  DEPLOY_SKIP_REASON=""
  if ! quiet_phase "Build ${os^} release" "$build"; then
    if (( ! INTERRUPTED_STATUS )); then
      record_unbuilt "$os" "${vendors[@]}"
      if [ "$os" = windows ] && { [ "$TARGET" = all ] || [ "$TARGET" = windows ]; }; then
        record_unbuilt windows multi
      fi
    fi
    return "$INTERRUPTED_STATUS"
  fi
  for ((i=0; i<${#vendors[@]}; i++)); do
    if ! run_lane "${vendors[i]}-$os" "$test" "${vendors[i]}"; then
      (( INTERRUPTED_STATUS )) || return 1
      return "$INTERRUPTED_STATUS"
    fi
  done
}

case "$TARGET" in
  all) run_platform linux build_linux_release test_linux_release nvidia intel amd
       run_platform windows package_windows_release test_windows_release nvidia intel amd
       run_lane multi-windows test_windows_multi_release ;;
  linux) run_platform linux build_linux_release test_linux_release nvidia intel amd ;;
  windows) run_platform windows package_windows_release test_windows_release nvidia intel amd
           run_lane multi-windows test_windows_multi_release ;;
  linux-*) run_platform linux build_linux_release test_linux_release "${TARGET#linux-}" ;;
  windows-*) run_platform windows package_windows_release test_windows_release "${TARGET#windows-}" ;;
esac
