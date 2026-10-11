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
ACTIVE_LANE_WINDOWS=0
SUDO_REFRESH_PGID=""
INTERRUPTED_STATUS=0
# Linux workers have no VM/cache-export cleanup protocol.
INTERRUPT_GRACE_SECONDS=60

case "$TARGET" in
  all) REQUESTED_LANES=(
    intel-linux nvidia-linux amd-linux multi-linux intel-windows nvidia-windows amd-windows multi-windows
  ) ;;
  linux) REQUESTED_LANES=(intel-linux nvidia-linux amd-linux multi-linux) ;;
  windows) REQUESTED_LANES=(intel-windows nvidia-windows amd-windows multi-windows) ;;
  linux-nvidia|linux-intel|linux-amd) REQUESTED_LANES=("${TARGET#linux-}-linux") ;;
  windows-nvidia|windows-intel|windows-amd) REQUESTED_LANES=("${TARGET#windows-}-windows") ;;
  *) echo "Unknown MOM_DEPLOY_TARGET: $TARGET" >&2; exit 2 ;;
esac
case "$DEPLOY_SKIP_VECTORS" in 0|1) ;; *) echo "MOM_DEPLOY_SKIP_VECTORS must be 0 or 1" >&2; exit 2;; esac
case "${MOM_FORCE_REBUILD:-0}" in 0|1) ;; *) echo "MOM_FORCE_REBUILD must be 0 or 1" >&2; exit 2;; esac
if [ "${MOM_FORCE_REBUILD:-0}" = 1 ] && [ "${MOM_DEPLOY_REUSE_ARCHIVE:-0}" = 1 ]; then
  echo "MOM_FORCE_REBUILD cannot be combined with MOM_DEPLOY_REUSE_ARCHIVE" >&2
  exit 2
fi

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
  # sudo's PTY monitor reliably relays TERM, not the ignored background INT.
  if (( ACTIVE_LANE_WINDOWS )); then
    kill -TERM -- "-$ACTIVE_LANE_PGID" 2>/dev/null || true
  else
    kill -s "$signal" -- "-$ACTIVE_LANE_PGID" 2>/dev/null || true
  fi
  if (( ACTIVE_LANE_WINDOWS )); then
    # The runner owns shutdown/export deadlines. Killing its cleanup loses the
    # guest cache and can leave passthrough recovery incomplete.
    while kill -0 -- "-$ACTIVE_LANE_PGID" 2>/dev/null; do sleep 1 || true; done
    return
  fi
  for ((i=0; i<INTERRUPT_GRACE_SECONDS; i++)); do
    kill -0 -- "-$ACTIVE_LANE_PGID" 2>/dev/null || return
    sleep 1 || true
  done
  kill -KILL -- "-$ACTIVE_LANE_PGID" 2>/dev/null || true
}

prepare_windows_auth() {
  case "$TARGET" in all|windows|windows-*) ;; *) return 0 ;; esac
  [ -x "$WIN_RUN" ] && [ "$(id -u)" != 0 ] && [ -z "$SUDO_PASSWORD" ] || return 0
  # Ask before lengthy Linux work, never midway through an unattended run.
  if ! sudo -n -v 2>/dev/null; then
    if [ -t 0 ] && [ -t 3 ]; then
      printf '[deploy] Authenticate now for the Windows deployment phases.\n' >&3
      sudo -v >&3 2>&3 || return
    else
      printf 'Windows deployment needs sudo authentication; run from an interactive terminal or supply SUDO_PASSWORD.\n' >&2
      return 1
    fi
  fi
  # Retain the caller's tty-scoped timestamp. A separate group lets finish stop
  # both the refresher and its sleep without touching the active VM group.
  # Quiet the expected sleep-termination notice; refresh failures still report on fd 3.
  set -m
  (
    set +m
    trap - EXIT
    trap 'exit 0' INT TERM
    while kill -0 "$$" 2>/dev/null; do
      sleep 60 || exit
      kill -0 "$$" 2>/dev/null || exit
      sudo -n -v 2>/dev/null || {
        printf '[deploy] Sudo authentication expired or was revoked; Windows phases will fail without prompting.\n' >&3
        exit 1
      }
    done
  ) </dev/null 2>/dev/null & SUDO_REFRESH_PGID=$!
  set +m
}

run_active() {
  local status windows_phase=0
  case "$1" in
    package_windows_release) windows_phase=1 ;;
    stream_lane) [[ "${3:-}" == test_windows_* ]] && windows_phase=1 ;;
  esac
  ACTIVE_LANE_WINDOWS=$windows_phase
  if (( windows_phase )) && [ -x "$WIN_RUN" ] && [ "$(id -u)" != 0 ] &&
     [ -z "$SUDO_PASSWORD" ]; then
    # Revoked credentials must fail clearly, not prompt after hours of testing.
    if ! sudo -n -v 2>/dev/null; then
      printf 'Windows deployment sudo authentication is no longer valid; rerun with MOM_DEPLOY_TARGET=windows.\n' >&2
      return 1
    fi
  fi
  # Windows sudo commands must share the caller's tty-scoped timestamp. Job
  # control gives their worker a separate process group without losing that tty.
  if (( windows_phase )); then set -m; else set +m; fi
  (
    set +m
    export ROOT_DIR TARGET WIN_RUN WIN_MOM_DEV_BASE SUDO_PASSWORD RELEASE_VERSION RELEASE_DIR \
      LINUX_ARCHIVE WINDOWS_ARCHIVE WINDOWS_STAGE DEPLOY_SKIP_VECTORS
    export -f skip has_drm_vendor linux_gpu_available build_linux_release \
      windows_available run_windows_root package_windows_release test_linux_release test_windows_release \
      test_windows_multi_release \
      stream_lane
    trap - INT TERM QUIT
    if (( windows_phase )); then
      exec bash -c 'set +e; set -u -o pipefail; "$@"' deploy-active "$@"
    fi
    exec setsid bash -c 'set +e; set -u -o pipefail; "$@"' deploy-active "$@"
  ) & ACTIVE_LANE_PGID=$!
  set +m
  wait "$ACTIVE_LANE_PGID"; status=$?
  if (( INTERRUPTED_STATUS )); then
    wait "$ACTIVE_LANE_PGID" 2>/dev/null || true
    status=$INTERRUPTED_STATUS
  fi
  ACTIVE_LANE_PGID=""
  ACTIVE_LANE_WINDOWS=0
  return "$status"
}

stream_lane() {
  local log="$1" windows_lane=0
  shift
  [[ "$1" == test_windows_* ]] && windows_lane=1
  if [ "$DEPLOY_SKIP_VECTORS" = 1 ]; then
    printf '  ➖ Stability checks skipped (MOM_DEPLOY_SKIP_VECTORS=1)\n' | tee -a "$log"
    printf 'MOM_TEST_SUMMARY 1 0 0 1\n' >>"$log"
  else
    : >"$log"
  fi
  # Keep Windows readers alive while sudo relays cancellation and exports logs.
  "$@" 2>&1 | ( (( windows_lane )) && trap '' INT TERM; exec tee -a "$log" ) | (
    (( windows_lane )) && trap '' INT TERM
    exec awk '
    /Compute core error:/ {print "[perf] Compute core error reported; see private debug logs"; fflush(); next}
    /^[[:space:]]*✔ Stability checks passed$/ ||
    /^[[:space:]]*[✔✖] [^:]+: .*% of README/ ||
    /^[[:space:]]*✖ [^:]+: (release discovery reported no GPU tuning|no hashrate was reported)$/ ||
    /^\[perf\] / ||
    /^Release command deadline reached / ||
    /^Release benchmark still waiting for cooperative cleanup/ ||
    /^\[deploy\] Debug logs: / ||
    / received; stopping the VM / {print; fflush()}'
  )
}

run_lane() {
  local platform="$1" status=0 log i value line fails_before reason outcome synthetic summary_seen=0 summary_invalid=0
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
  while IFS= read -r line; do
    summary_seen=1
    # Native Windows output may traverse more than one CRLF-converting PTY.
    while [[ "$line" == *$'\r' ]]; do line="${line%$'\r'}"; done
    if [[ ! "$line" =~ ^MOM_TEST_SUMMARY\ ([0-9]{1,9})\ ([0-9]{1,9})\ ([0-9]{1,9})\ ([0-9]{1,9})$ ]]; then
      summary_invalid=1
      continue
    fi
    if (( 10#${BASH_REMATCH[1]} == 0 ||
          10#${BASH_REMATCH[1]} != 10#${BASH_REMATCH[2]} + 10#${BASH_REMATCH[3]} + 10#${BASH_REMATCH[4]} )); then
      summary_invalid=1
      continue
    fi
    lane_pass=$((lane_pass + 10#${BASH_REMATCH[2]}))
    lane_fail=$((lane_fail + 10#${BASH_REMATCH[3]}))
    lane_skip=$((lane_skip + 10#${BASH_REMATCH[4]}))
    for i in 0 1 2 3; do
      value=$((10#${BASH_REMATCH[i + 1]}))
      TEST_COUNTS[$i]=$((TEST_COUNTS[i] + value))
    done
  done < <(grep '^MOM_TEST_SUMMARY ' "$log" || true)
  if (( INTERRUPTED_STATUS )); then
    outcome=FAIL reason=interrupted synthetic="$(( TEST_COUNTS[2] == fails_before ))"
    printf '  ✖ %s (interrupted)\n' "$platform"
  elif (( summary_invalid )); then
    outcome=FAIL reason="invalid test summary" synthetic=1
    printf '  ✖ %s (%s)\n' "$platform" "$reason"
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
      !/^\[deploy\] Debug logs: / &&
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
  if [ -n "$SUDO_REFRESH_PGID" ]; then
    kill -TERM -- "-$SUDO_REFRESH_PGID" 2>/dev/null || true
    wait "$SUDO_REFRESH_PGID" 2>/dev/null || true
  fi
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

build_linux_release() {
  if [ "${MOM_DEPLOY_REUSE_ARCHIVE:-0}" = 1 ] && [ -f "$LINUX_ARCHIVE" ]; then
    echo "[deploy] Reusing $LINUX_ARCHIVE"
    return
  fi
  if ! command -v docker >/dev/null; then
    skip "Docker is unavailable; Linux deployment tests need a clean container"
    return 1
  fi
  echo "[deploy] Building and packaging Linux release archive"
  # The container entrypoint builds once before executing npm; package in that same toolchain.
  MOM_PORTABLE_BUILD=1 MOM_GPU_BACKEND=all MOM_LINUX_BUILD_COMPILER=all MOM_DOCKER_GPUS=0 \
    MOM_REUSE_BUILT_WORKER=0 MOM_RELEASE_VERSION="$RELEASE_VERSION" \
    ./scripts/dev-container.sh npm run build:release -- --package-only
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
  local vendors="$vendor" suite=gpu-discrete uuid vendor_file node name
  local -a devices=() docker_gpu=() selector=() verthash=() evidence=()
  if [ "$vendor" = multi ]; then
    if [ "$DEPLOY_SKIP_VECTORS" = 1 ] || [ -n "${MOM_DEPLOY_ALGO:-}" ]; then
      skip "Linux mixed-vendor correctness requires the full vector gates"
      return
    fi
    if ! { has_drm_vendor 0x10de || linux_gpu_available nvidia; } || ! has_drm_vendor 0x1002; then
      skip "Linux mixed-vendor correctness requires NVIDIA and AMD hardware"
      return
    fi
    if ! linux_gpu_available amd; then
      echo "AMD hardware is present but its Linux GPU device interface is unavailable" >&2
      return 1
    fi
  elif ! linux_gpu_available "$vendor"; then
    skip "no usable Linux $vendor GPU"
    return
  fi
  case "$vendor" in
    multi)
      # Admit one NVIDIA GPU and only the first AMD GPU's DRM nodes, never Intel.
      uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1)" || return
      if [[ ! "$uuid" =~ ^GPU-[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}$ ]]; then
        echo "Cannot select a NVIDIA GPU for mixed-vendor correctness" >&2
        return 1
      fi
      for vendor_file in /sys/class/drm/card*/device/vendor; do
        [ -r "$vendor_file" ] || continue
        [ "$(tr '[:upper:]' '[:lower:]' <"$vendor_file")" = 0x1002 ] || continue
        for node in "${vendor_file%/vendor}"/drm/*; do
          name="${node##*/}"
          [[ "$name" =~ ^(card[0-9]+|renderD[0-9]+)$ ]] || continue
          [ -c "/dev/dri/$name" ] && devices+=("--device=/dev/dri/$name")
        done
        break
      done
      if [ "${#devices[@]}" = 0 ]; then
        echo "Cannot select AMD DRM nodes for mixed-vendor correctness" >&2
        return 1
      fi
      docker_gpu=(--gpus "device=$uuid")
      devices+=(--device=/dev/kfd --group-add video)
      vendors=nvidia,amd
      suite=gpu-multi
      selector=(MOM_GPU_BACKEND=nvidia MOM_REQUIRE_MULTI_GPU_TESTS=1)
      ;;
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

  echo "[deploy] Testing $vendors release GPU suite $suite in clean Ubuntu 26.04"
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
      MOM_INSTALL_GPU_VENDORS="'"$vendors"'" ./install.sh >/tmp/mom-install.log
      export '"${selector[*]}"'
      export MOM_GPU_TEST_VENDORS="'"$vendors"'"
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
          .github/workflows/scripts/test-release-linux.sh "$MOM_RELEASE_ARCHIVE" '"$suite"' \
          >"$vector_log" 2>&1 || status=$?
        awk '\''/^ *ℹ (tests|pass|fail|skipped) [0-9]+$/ {v[$2]=$3} END {
          if (v["tests"]) print "MOM_TEST_SUMMARY",v["tests"],v["pass"],v["fail"],v["skipped"]
        }'\'' "$vector_log"
        if [ "$status" = 0 ]; then echo "  ✔ Stability checks passed"
        else cat "$vector_log"; exit "$status"; fi
        rm -f "$vector_log"
      fi
      [ "'"$suite"'" != gpu-multi ] || exit 0
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
  # A builtin wait receives TERM immediately even when the runner is waiting on
  # SSH. Signal only our freshly created session, then reap its complete cleanup.
  local -a command=(bash -c '
    child=""; cancelled=0; forwarded=0
    cancel() {
      trap "" INT TERM
      cancelled=1
      if [ -n "$child" ] && (( ! forwarded )); then
        forwarded=1
        # Before setsid runs, the owned child has no separate group yet.
        kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
      fi
    }
    trap cancel INT TERM
    setsid "$@" <&0 & child=$!
    (( cancelled )) && cancel
    wait "$child"; status=$?
    if (( cancelled )); then wait "$child"; status=143; fi
    exit "$status"
  ' deploy-root "$@")
  if [ "$(id -u)" = 0 ]; then
    "${command[@]}"
  elif [ -n "$SUDO_PASSWORD" ]; then
    printf '%s\n' "$SUDO_PASSWORD" | sudo -S -p '' -- "${command[@]}"
  else
    sudo -n -- "${command[@]}"
  fi
}

package_windows_release() {
  windows_available || return 1
  if [ "${MOM_DEPLOY_REUSE_ARCHIVE:-0}" = 1 ] && [ -f "$WINDOWS_ARCHIVE" ]; then
    echo "[deploy] Reusing $WINDOWS_ARCHIVE"
    return
  fi
  echo "[deploy] Building and packaging Windows release archive"
  # The explicit dev base supplies toolchains; --release disables the VM's implicit prebuild.
  # Keep ownership restoration in the same privileged command: a long VM build
  # can outlive sudo's timestamp, but does not revoke an already running command.
  run_windows_root bash -c '
    owner="$1"; archive="$2"; shift 2
    "$@" || exit
    [ -f "$archive" ] || {
      printf "Windows release build did not produce %s\\n" "$archive" >&2
      exit 1
    }
    chown -R "$owner" build/win "$archive"
  ' deploy-windows-package "$(id -u):$(id -g)" "$WINDOWS_ARCHIVE" \
    env GPU_GROUP=none WIN_MOM_RUN_BASE="$WIN_MOM_DEV_BASE" "$WIN_RUN" \
    --release --download build/win --download "$WINDOWS_ARCHIVE" -- \
    env MOM_RELEASE_VERSION="$RELEASE_VERSION" MOM_FORCE_REBUILD="${MOM_FORCE_REBUILD:-0}" \
    powershell -NoProfile -ExecutionPolicy Bypass -Command \
    'npm install --ignore-scripts; if ($LASTEXITCODE) { exit $LASTEXITCODE }; npm run build:release; if ($LASTEXITCODE) { exit $LASTEXITCODE }' || return
}

test_windows_release() {
  local vendor="$1" group="$1" platform="$1-windows" status=0
  [ "$vendor" = intel ] && group=arc
  mkdir -p "$WINDOWS_STAGE"
  cp -f "$WINDOWS_ARCHIVE" "$WINDOWS_STAGE/$WINDOWS_ARCHIVE" || return
  echo "[deploy] Testing every Windows $vendor GPU vector and README performance gate"
  run_windows_root env GPU_GROUP="$group" "$WIN_RUN" \
    --release --download "$WINDOWS_STAGE" -- \
    env MOM_SKIP_MSR=1 MOM_DEPLOY_DEBUG="${MOM_DEPLOY_DEBUG:-}" \
      MOM_DEPLOY_SKIP_VECTORS="$DEPLOY_SKIP_VECTORS" \
      MOM_DEPLOY_ALGO="${MOM_DEPLOY_ALGO:-}" \
      MOM_GPU_TEST_ALGO="${MOM_DEPLOY_ALGO:-}" \
      MOM_KAWPOW_DAG_CHUNK_NODES="${MOM_KAWPOW_DAG_CHUNK_NODES:-}" \
    powershell -NoProfile -ExecutionPolicy Bypass -File \
      scripts\\test-windows-release-gpu.ps1 \
      -Archive "${WINDOWS_STAGE//\//\\}\\$WINDOWS_ARCHIVE" -Platform "$platform" || status=$?
  # The VM round trip can replace directory modes; keep raw diagnostics private on this host.
  if [ -d "$WINDOWS_STAGE/debug" ]; then
    [ ! -L "$WINDOWS_STAGE/debug" ] && chmod 700 "$WINDOWS_STAGE/debug" || return 1
    if [ "${MOM_DEPLOY_DEBUG:-}" = 1 ]; then
      printf '[deploy] Debug logs: %s/debug\n' "$WINDOWS_STAGE"
    fi
  fi
  return "$status"
}

test_windows_multi_release() {
  local vector_log status=0
  if [ "$DEPLOY_SKIP_VECTORS" = 1 ]; then
    skip "Windows mixed-vendor correctness skipped with the vector gates"
    return
  fi
  if [ -n "${MOM_DEPLOY_ALGO:-}" ]; then
    skip "Windows mixed-vendor correctness is outside a single-algorithm deployment"
    return
  fi
  mkdir -p "$WINDOWS_STAGE"
  cp -f "$WINDOWS_ARCHIVE" "$WINDOWS_STAGE/$WINDOWS_ARCHIVE" || return
  echo "[deploy] Testing concurrent NVIDIA and AMD workers from the Windows release"
  vector_log="$(mktemp)" || return
  run_windows_root env GPU_GROUP=discrete "$WIN_RUN" \
    --release --download "$WINDOWS_STAGE" -- \
    env MOM_SKIP_MSR=1 MOM_GPU_BACKEND=nvidia MOM_GPU_TEST_VENDORS=nvidia,amd \
      MOM_REQUIRE_MULTI_GPU_TESTS=1 \
    powershell -NoProfile -ExecutionPolicy Bypass -File \
      .github\\workflows\\scripts\\test-release-windows.ps1 \
      -Archive "${WINDOWS_STAGE//\//\\}\\$WINDOWS_ARCHIVE" -Suite gpu-multi >"$vector_log" 2>&1 || status=$?
  cat "$vector_log"
  # PowerShell 5.1 can decode the spec reporter's UTF-8 info symbol as Γä╣.
  awk '{sub(/\r$/, "")} /^[[:space:]]*(ℹ|Γä╣|#) (tests|pass|fail|skipped) [0-9]+$/ {
    if (length($3) <= 9) v[$2]=$3
  } END {
    if (v["tests"] && "pass" in v && "fail" in v && "skipped" in v)
      print "MOM_TEST_SUMMARY",v["tests"],v["pass"],v["fail"],v["skipped"]
  }' "$vector_log"
  rm -f "$vector_log"
  return "$status"
}

run_platform() {
  local os="$1" build="$2" test="$3" i
  shift 3
  local -a vendors=("$@")
  DEPLOY_SKIP_REASON=""
  if ! quiet_phase "Build ${os^} release" "$build"; then
    if (( ! INTERRUPTED_STATUS )); then
      record_unbuilt "$os" "${vendors[@]}"
      if [ "$TARGET" = all ] || [ "$TARGET" = "$os" ]; then
        record_unbuilt "$os" multi
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
  if [ "$TARGET" = all ] || [ "$TARGET" = "$os" ]; then
    if [ "$os" = windows ]; then run_lane multi-windows test_windows_multi_release
    else run_lane multi-linux test_linux_release multi; fi
  fi
}

if ! prepare_windows_auth; then
  for platform in "${REQUESTED_LANES[@]}"; do
    case "$platform" in
      *-windows) record_result "$platform" FAIL "sudo authentication failed" 1 ;;
      *) record_result "$platform" SKIP "Windows sudo preflight failed" 1 ;;
    esac
  done
  exit 1
fi

case "$TARGET" in
  all) run_platform linux build_linux_release test_linux_release intel nvidia amd
       run_platform windows package_windows_release test_windows_release intel nvidia amd ;;
  linux) run_platform linux build_linux_release test_linux_release intel nvidia amd ;;
  windows) run_platform windows package_windows_release test_windows_release intel nvidia amd ;;
  linux-*) run_platform linux build_linux_release test_linux_release "${TARGET#linux-}" ;;
  windows-*) run_platform windows package_windows_release test_windows_release "${TARGET#windows-}" ;;
esac
