#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="${MOM_DEPLOY_TARGET:-all}"
MIN_KAWPOW_MHS="${MOM_DEPLOY_MIN_KAWPOW_MHS:-20}"
WIN_RUN="${WIN_RUN:-$HOME/win/run.sh}"
WIN_MOM_DEV_BASE="${WIN_MOM_DEV_BASE:-$HOME/cache/win/images/win-mom-dev.qcow2}"
SUDO_PASSWORD="${SUDO_PASSWORD:-sap}"
RELEASE_VERSION="${MOM_RELEASE_VERSION:-$(node -p "require('./package.json').version")}"
RELEASE_DIR="mom-v${RELEASE_VERSION}"
LINUX_ARCHIVE="${RELEASE_DIR}-lin.tgz"
WINDOWS_ARCHIVE="${RELEASE_DIR}-win.zip"

case "$TARGET" in
  all|linux|windows|linux-nvidia|linux-intel|linux-amd|windows-nvidia|windows-intel|windows-amd) ;;
  *) echo "Unknown MOM_DEPLOY_TARGET: $TARGET" >&2; exit 2 ;;
esac

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "$1 is required" >&2; exit 1; }
}

check_kawpow_output() {
  local label="$1" output="$2" minimum="${3:-$MIN_KAWPOW_MHS}"
  python3 -c '
import re
import sys

label = sys.argv[1]
minimum = float(sys.argv[2])
text = sys.stdin.read()
matches = re.findall(r"Algo kawpow \([^)]*\) hashrate: ([0-9.]+)\s+([KMGT]?H/s)", text)
scale = {"H/s": 1e-6, "KH/s": 1e-3, "MH/s": 1.0, "GH/s": 1e3, "TH/s": 1e6}
rates = [float(v) * scale[u] for v, u in matches]
if not rates:
    print(f"{label}: no kawpow hashrate found", file=sys.stderr)
    sys.exit(1)
best = max(rates)
print(f"{label}: kawpow best {best:.2f} MH/s")
if best < minimum:
    print(f"{label}: kawpow below {minimum:.2f} MH/s", file=sys.stderr)
    sys.exit(1)
' "$label" "$minimum" <<<"$output"
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
  require_cmd docker
  echo "[deploy] Building Linux release binary"
  MOM_GPU_BACKEND=all ./r.sh true
  echo "[deploy] Packaging Linux release archive"
  package_linux_release
}

test_linux_nvidia_release() {
  echo "[deploy] Testing Linux NVIDIA release archive in clean Ubuntu 26.04 container"
  set +e
  local output
  output="$(
    docker run --rm --gpus all -v "$ROOT_DIR:/repo:ro" \
      -e MOM_TEST_NO_POOL_NETWORK=1 \
      -e MOM_RELEASE_ARCHIVE="$LINUX_ARCHIVE" -e MOM_RELEASE_DIR="$RELEASE_DIR" \
      ubuntu:26.04 bash -lc '
      set -euo pipefail
      export DEBIAN_FRONTEND=noninteractive
      apt-get update >/dev/null
      apt-get install -y --no-install-recommends ca-certificates >/dev/null
      cd /tmp
      tar -xzf "/repo/$MOM_RELEASE_ARCHIVE"
      cd "$MOM_RELEASE_DIR"
      MOM_INSTALL_GPU_VENDORS=nvidia ./install.sh >/tmp/mom-install.log
      # Exercise the release launcher exactly as a miner does. It must discover the CUDA compiler
      # payload installed above; injecting CUDA_PATH here would mask a slow ProgPoW fallback.
      timeout 180 env MOM_GPU_BACKEND=nvidia ./mom bench kawpow --job.dev gpu1*6291456
    ' 2>&1
  )"
  local status=$?
  set -e
  printf '%s\n' "$output"
  if [ "$status" -ne 0 ] && [ "$status" -ne 124 ]; then
    return "$status"
  fi
  check_kawpow_output "linux nvidia release" "$output"
}

test_linux_intel_release() {
  echo "[deploy] Testing Linux Intel release archive in clean Ubuntu 26.04 container"
  set +e
  local output
  output="$(
    docker run --rm --device=/dev/dri:/dev/dri -v "$ROOT_DIR:/repo:ro" \
      -e MOM_TEST_NO_POOL_NETWORK=1 \
      -e MOM_RELEASE_ARCHIVE="$LINUX_ARCHIVE" -e MOM_RELEASE_DIR="$RELEASE_DIR" \
      ubuntu:26.04 bash -lc '
      set -euo pipefail
      export DEBIAN_FRONTEND=noninteractive
      apt-get update >/dev/null
      apt-get install -y --no-install-recommends ca-certificates >/dev/null
      cd /tmp
      tar -xzf "/repo/$MOM_RELEASE_ARCHIVE"
      cd "$MOM_RELEASE_DIR"
      MOM_INSTALL_GPU_VENDORS=intel ./install.sh >/tmp/mom-install.log
      timeout 180 env ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 ./mom bench kawpow --job.dev gpu1*37282560
    ' 2>&1
  )"
  local status=$?
  set -e
  printf '%s\n' "$output"
  if [ "$status" -ne 0 ] && [ "$status" -ne 124 ]; then
    return "$status"
  fi
  check_kawpow_output "linux intel release" "$output"
}

test_linux_amd_release() {
  echo "[deploy] Testing Linux AMD release archive in clean Ubuntu 26.04 container"
  set +e
  local output
  output="$(
    docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
      -v "$ROOT_DIR:/repo:ro" -e MOM_TEST_NO_POOL_NETWORK=1 \
      -e MOM_RELEASE_ARCHIVE="$LINUX_ARCHIVE" \
      -e MOM_RELEASE_DIR="$RELEASE_DIR" ubuntu:26.04 bash -lc '
      set -euo pipefail
      export DEBIAN_FRONTEND=noninteractive
      apt-get update >/dev/null
      apt-get install -y --no-install-recommends ca-certificates >/dev/null
      cd /tmp
      tar -xzf "/repo/$MOM_RELEASE_ARCHIVE"
      cd "$MOM_RELEASE_DIR"
      MOM_INSTALL_GPU_VENDORS=amd ./install.sh >/tmp/mom-install.log
      timeout 180 env MOM_GPU_BACKEND=amd ./mom bench kawpow --job.dev gpu1*2796032
    ' 2>&1
  )"
  local status=$?
  set -e
  printf '%s\n' "$output"
  if [ "$status" -ne 0 ] && [ "$status" -ne 124 ]; then
    return "$status"
  fi
  check_kawpow_output "linux amd release" "$output" 19
}

package_windows_release() {
  [ -x "$WIN_RUN" ] || { echo "$WIN_RUN is required for Windows deploy tests" >&2; exit 1; }

  echo "[deploy] Building and packaging Windows release archive"
  printf '%s\n' "$SUDO_PASSWORD" | sudo -S env GPU_GROUP=nvidia WIN_MOM_RUN_BASE="$WIN_MOM_DEV_BASE" "$WIN_RUN" \
    --release --download "$WINDOWS_ARCHIVE" -- powershell -NoProfile -ExecutionPolicy Bypass -Command \
    "& .github\\workflows\\scripts\\build-windows-multicompiler.ps1; if (`$LASTEXITCODE) { exit `$LASTEXITCODE }; & .github\\workflows\\scripts\\package-windows.ps1"
}

test_windows_nvidia_release() {
  echo "[deploy] Testing Windows NVIDIA release archive on GPU-driver layer"
  local ps_script encoded output status
  ps_script="$(cat <<'PS'
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$dst = 'C:\mom-release-test'
Remove-Item -Recurse -Force $dst -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $dst | Out-Null
$archive = Get-Item (Join-Path (Get-Location) $env:MOM_RELEASE_ARCHIVE)
Expand-Archive -Force -Path $archive.FullName -DestinationPath $dst
$releaseDir = $archive.BaseName -replace '-win$', ''
Set-Location (Join-Path $dst $releaseDir)
& .\install.bat -InstallCudaToolkit
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$ready = (Test-Path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\ptxas.exe') -and
  (Test-Path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\nvvm\libdevice\libdevice.10.bc') -and
  (Test-Path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include\cuda.h') -and
  (Test-Path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include\cuda_runtime.h') -and
  (Test-Path 'C:\BuildTools\VC\Tools\MSVC') -and
  (Test-Path 'C:\Program Files (x86)\Windows Kits\10\Include')
Write-Host "NVIDIA_JIT_READY=$ready"
if (-not $ready) { throw 'CUDA toolkit or C++ headers did not install' }
$env:ONEAPI_DEVICE_SELECTOR = 'cuda:*'
$p = Start-Process -FilePath '.\mom.cmd' -ArgumentList @('bench', 'kawpow', '--job.dev', 'gpu1*6291456') -NoNewWindow -RedirectStandardOutput 'mom-kawpow.out' -RedirectStandardError 'mom-kawpow.err' -PassThru
if (-not $p.WaitForExit(180000)) { $p.Kill() }
$out = Get-Content -Raw -ErrorAction SilentlyContinue 'mom-kawpow.out'
$err = Get-Content -Raw -ErrorAction SilentlyContinue 'mom-kawpow.err'
Write-Host $out
if ($err) { Write-Host $err }
if (($out + $err) -notmatch 'Algo kawpow .* hashrate:') { throw 'kawpow release perf not confirmed' }
PS
)"
  encoded="$(printf '%s' "$ps_script" | iconv -f UTF-8 -t UTF-16LE | base64 -w0)"
  set +e
  output="$(printf '%s\n' "$SUDO_PASSWORD" | sudo -S env GPU_GROUP=nvidia "$WIN_RUN" --release -- \
    env "MOM_RELEASE_ARCHIVE=$WINDOWS_ARCHIVE" "MOM_TEST_NO_POOL_NETWORK=1" \
      powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand "$encoded" 2>&1)"
  status=$?
  set -e
  printf '%s\n' "$output"
  if [ "$status" -ne 0 ]; then
    return "$status"
  fi
  check_kawpow_output "windows nvidia release" "$output"
}

test_windows_intel_release() {
  echo "[deploy] Testing Windows Intel release archive on GPU-driver layer"
  local ps_script encoded output status
  ps_script="$(cat <<'PS'
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$dst = 'C:\mom-release-test'
Remove-Item -Recurse -Force $dst -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $dst | Out-Null
$archive = Get-Item (Join-Path (Get-Location) $env:MOM_RELEASE_ARCHIVE)
Expand-Archive -Force -Path $archive.FullName -DestinationPath $dst
$releaseDir = $archive.BaseName -replace '-win$', ''
Set-Location (Join-Path $dst $releaseDir)
& .\install.bat
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:ONEAPI_DEVICE_SELECTOR = 'level_zero:gpu'
$env:ZE_AFFINITY_MASK = '0'
$p = Start-Process -FilePath '.\mom.cmd' -ArgumentList @('bench', 'kawpow', '--job.dev', 'gpu1*37282560') -NoNewWindow -RedirectStandardOutput 'mom-kawpow.out' -RedirectStandardError 'mom-kawpow.err' -PassThru
if (-not $p.WaitForExit(180000)) { $p.Kill() }
$out = Get-Content -Raw -ErrorAction SilentlyContinue 'mom-kawpow.out'
$err = Get-Content -Raw -ErrorAction SilentlyContinue 'mom-kawpow.err'
Write-Host $out
if ($err) { Write-Host $err }
if (($out + $err) -notmatch 'Algo kawpow .* hashrate:') { throw 'kawpow release perf not confirmed' }
PS
)"
  encoded="$(printf '%s' "$ps_script" | iconv -f UTF-8 -t UTF-16LE | base64 -w0)"
  set +e
  output="$(printf '%s\n' "$SUDO_PASSWORD" | sudo -S env GPU_GROUP=arc "$WIN_RUN" --release -- \
    env "MOM_RELEASE_ARCHIVE=$WINDOWS_ARCHIVE" "MOM_TEST_NO_POOL_NETWORK=1" \
      powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand "$encoded" 2>&1)"
  status=$?
  set -e
  printf '%s\n' "$output"
  if [ "$status" -ne 0 ]; then
    return "$status"
  fi
  check_kawpow_output "windows intel release" "$output"
}

test_windows_amd_release() {
  echo "[deploy] Testing Windows AMD release archive on GPU-driver layer"
  local ps_script encoded output status
  ps_script="$(cat <<'PS'
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$dst = 'C:\mom-release-test'
Remove-Item -Recurse -Force $dst -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $dst | Out-Null
$archive = Get-Item (Join-Path (Get-Location) $env:MOM_RELEASE_ARCHIVE)
Expand-Archive -Force -Path $archive.FullName -DestinationPath $dst
$releaseDir = $archive.BaseName -replace '-win$', ''
Set-Location (Join-Path $dst $releaseDir)
& .\install.bat
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:MOM_GPU_BACKEND = 'amd'
$env:ACPP_VISIBILITY_MASK = 'hip'
$p = Start-Process -FilePath '.\mom.cmd' -ArgumentList @('bench', 'kawpow', '--job.dev', 'gpu1*2796032') -NoNewWindow -RedirectStandardOutput 'mom-kawpow.out' -RedirectStandardError 'mom-kawpow.err' -PassThru
if (-not $p.WaitForExit(240000)) { $p.Kill() }
$out = Get-Content -Raw -ErrorAction SilentlyContinue 'mom-kawpow.out'
$err = Get-Content -Raw -ErrorAction SilentlyContinue 'mom-kawpow.err'
Write-Host $out
if ($err) { Write-Host $err }
if (($out + $err) -notmatch 'Algo kawpow .* hashrate:') { throw 'AMD kawpow release perf not confirmed' }
PS
)"
  encoded="$(printf '%s' "$ps_script" | iconv -f UTF-8 -t UTF-16LE | base64 -w0)"
  set +e
  output="$(printf '%s\n' "$SUDO_PASSWORD" | sudo -S env GPU_GROUP=amd "$WIN_RUN" --release -- \
    env "MOM_RELEASE_ARCHIVE=$WINDOWS_ARCHIVE" "MOM_TEST_NO_POOL_NETWORK=1" \
      powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand "$encoded" 2>&1)"
  status=$?
  set -e
  printf '%s\n' "$output"
  [ "$status" -eq 0 ] || return "$status"
  check_kawpow_output "windows amd release" "$output" 15
}

case "$TARGET" in
  all) build_linux_release; test_linux_nvidia_release; test_linux_intel_release; test_linux_amd_release; package_windows_release; test_windows_nvidia_release; test_windows_intel_release; test_windows_amd_release ;;
  linux) build_linux_release; test_linux_nvidia_release; test_linux_intel_release; test_linux_amd_release ;;
  windows) package_windows_release; test_windows_nvidia_release; test_windows_intel_release; test_windows_amd_release ;;
  linux-nvidia) build_linux_release; test_linux_nvidia_release ;;
  linux-intel) build_linux_release; test_linux_intel_release ;;
  linux-amd) build_linux_release; test_linux_amd_release ;;
  windows-nvidia) package_windows_release; test_windows_nvidia_release ;;
  windows-intel) package_windows_release; test_windows_intel_release ;;
  windows-amd) package_windows_release; test_windows_amd_release ;;
esac
