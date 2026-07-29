#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="${MOM_DEPLOY_TARGET:-all}"
WIN_RUN="${WIN_RUN:-$HOME/win/run.sh}"
WIN_MOM_DEV_BASE="${WIN_MOM_DEV_BASE:-$HOME/cache/win/images/win-mom-dev.qcow2}"
SUDO_PASSWORD="${SUDO_PASSWORD:-sap}"
RELEASE_VERSION="${MOM_RELEASE_VERSION:-$(node -p "require('./package.json').version")}"
RELEASE_DIR="mom-v${RELEASE_VERSION}"
LINUX_ARCHIVE="${RELEASE_DIR}-lin.tgz"
WINDOWS_ARCHIVE="${RELEASE_DIR}-win.zip"
WINDOWS_STAGE="build/deploy-win"

case "$TARGET" in
  all|linux|windows|linux-nvidia|linux-intel|linux-amd|windows-nvidia|windows-intel|windows-amd) ;;
  *) echo "Unknown MOM_DEPLOY_TARGET: $TARGET" >&2; exit 2 ;;
esac

skip() {
  printf '[deploy] SKIP: %s\n' "$*"
}

has_drm_vendor() {
  local expected="${1,,}" vendor
  for vendor in /sys/class/drm/card*/device/vendor; do
    [ -r "$vendor" ] || continue
    [ "$(tr '[:upper:]' '[:lower:]' <"$vendor")" = "$expected" ] && return 0
  done
  return 1
}

package_linux_release() {
  local container=""
  if ! NODE_BIN="$(command -v node)" .github/workflows/scripts/package-linux-combined.sh >/dev/null 2>&1; then
    container="$(docker create --entrypoint sleep mom-build-multicompiler infinity)"
    if docker cp "$container:/usr/local/bin/node" /tmp/mom-release-node; then
      chmod +x /tmp/mom-release-node
      NODE_BIN=/tmp/mom-release-node .github/workflows/scripts/package-linux-combined.sh
    fi
    rm -f /tmp/mom-release-node
    docker rm -f "$container" >/dev/null 2>&1 || true
  fi
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
  echo "[deploy] Building Linux release binary"
  MOM_GPU_BACKEND=all ./r.sh true
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
  local vendor="$1" platform="$1-linux"
  local -a devices=()
  local -a docker_gpu=()
  local -a selector=()
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

  echo "[deploy] Testing every $vendor GPU vector and README performance gate in clean Ubuntu 26.04"
  docker run --rm "${docker_gpu[@]}" "${devices[@]}" -v "$ROOT_DIR:/repo:ro" \
    -e MOM_RELEASE_ARCHIVE="$LINUX_ARCHIVE" -e MOM_RELEASE_DIR="$RELEASE_DIR" \
    -e MOM_SKIP_MSR=1 \
    ubuntu:26.04 bash -lc '
      set -euo pipefail
      export DEBIAN_FRONTEND=noninteractive
      apt-get update >/dev/null
      apt-get install -y --no-install-recommends ca-certificates nodejs >/dev/null
      cd /tmp
      tar -xzf "/repo/$MOM_RELEASE_ARCHIVE"
      cd "$MOM_RELEASE_DIR"
      MOM_INSTALL_GPU_VENDORS="'"$vendor"'" ./install.sh >/tmp/mom-install.log
      export '"${selector[*]}"'
      export MOM_GPU_TEST_VENDORS="'"$vendor"'"
      export MOM_REQUIRE_GPU_TESTS=1
      release_path="$PWD"
      cd /repo
      if [ "'"${MOM_DEPLOY_SKIP_VECTORS:-0}"'" != 1 ]; then
        NODE_BIN="$(command -v node)" MOM_RELEASE_TEST_DIR=/tmp/mom-release-vectors \
          .github/workflows/scripts/test-release-linux.sh "$MOM_RELEASE_ARCHIVE" gpu-discrete
      fi
      node scripts/check-release-performance.js \
        --miner "$release_path/mom" --readme README.md \
        --platform "'"$platform"'" --margin 0.05
    '
}

windows_available() {
  if [ ! -x "$WIN_RUN" ]; then
    skip "$WIN_RUN is unavailable; Windows deployment tests are optional"
    return 1
  fi
}

package_windows_release() {
  windows_available || return 1
  if [ "${MOM_DEPLOY_REUSE_ARCHIVE:-0}" = 1 ] && [ -f "$WINDOWS_ARCHIVE" ]; then
    echo "[deploy] Reusing $WINDOWS_ARCHIVE"
    return
  fi
  echo "[deploy] Building and packaging Windows release archive"
  printf '%s\n' "$SUDO_PASSWORD" | sudo -S env GPU_GROUP=nvidia \
    WIN_MOM_RUN_BASE="$WIN_MOM_DEV_BASE" "$WIN_RUN" \
    --release --download build/win --download "$WINDOWS_ARCHIVE" -- \
    powershell -NoProfile -ExecutionPolicy Bypass -Command \
    'npm ci --ignore-scripts; if ($LASTEXITCODE) { exit $LASTEXITCODE }; & .github\workflows\scripts\build-windows-multicompiler.ps1; if ($LASTEXITCODE) { exit $LASTEXITCODE }; & .github\workflows\scripts\package-windows.ps1; if ($LASTEXITCODE) { exit $LASTEXITCODE }'
  sudo chown -R "$(id -u):$(id -g)" build/win "$WINDOWS_ARCHIVE"
}

test_windows_release() {
  local vendor="$1" group="$1" platform="$1-windows"
  [ "$vendor" = intel ] && group=arc
  mkdir -p "$WINDOWS_STAGE"
  cp -f "$WINDOWS_ARCHIVE" "$WINDOWS_STAGE/$WINDOWS_ARCHIVE"
  echo "[deploy] Testing every Windows $vendor GPU vector and README performance gate"
  printf '%s\n' "$SUDO_PASSWORD" | sudo -S env GPU_GROUP="$group" "$WIN_RUN" \
    --release --download "$WINDOWS_STAGE" -- \
    env MOM_SKIP_MSR=1 MOM_DEPLOY_SKIP_VECTORS="${MOM_DEPLOY_SKIP_VECTORS:-0}" \
      MOM_DEPLOY_ALGO="${MOM_DEPLOY_ALGO:-}" \
    powershell -NoProfile -ExecutionPolicy Bypass -File \
      scripts\\test-windows-release-gpu.ps1 \
      -Archive "${WINDOWS_STAGE//\//\\}\\$WINDOWS_ARCHIVE" -Platform "$platform"
}

run_linux() {
  if ! build_linux_release; then return; fi
  test_linux_release nvidia
  test_linux_release intel
  test_linux_release amd
}

run_windows() {
  if ! package_windows_release; then return; fi
  test_windows_release nvidia
  test_windows_release intel
  test_windows_release amd
}

case "$TARGET" in
  all) run_linux; run_windows ;;
  linux) run_linux ;;
  windows) run_windows ;;
  linux-*) if build_linux_release; then test_linux_release "${TARGET#linux-}"; fi ;;
  windows-*) if package_windows_release; then test_windows_release "${TARGET#windows-}"; fi ;;
esac
