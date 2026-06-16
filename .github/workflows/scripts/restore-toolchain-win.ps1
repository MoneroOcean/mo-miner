# Restore the prebuilt from-source intel/llvm `--cuda` DPC++ toolchain (the GitHub release asset
# produced by package-toolchain-win.ps1) so CI can build the unified spir64+nvptx sycl.dll WITHOUT the
# ~1.5 h LLVM build (which cannot finish in a GitHub-hosted Windows job: 6 h cap, 2-4 vCPU, see
# scripts/build-windows-nvidia.md). Downloads the asset, verifies its SHA256, and extracts bin/lib/include
# to -Dest. Emits the resolved toolchain dir on stdout (last line) and as $env:MOM_DPCPP_DIR / GITHUB_ENV.
param(
  [string]$Repo  = "MoneroOcean/mo-miner",
  [string]$Tag   = "toolchain-win-dpcpp-cuda",
  [string]$Asset = "dpcpp-cuda-win.tar.gz",
  [string]$Dest  = ""
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
# Default destination only when -Dest is not given: RUNNER_TEMP in CI, else under the cwd for local runs.
# (Must not clobber an explicit -Dest -- that was a bug found provisioning a dev box.)
if (-not $Dest) {
  $Dest = if ($env:RUNNER_TEMP) { Join-Path $env:RUNNER_TEMP "dpcpp-cuda-win" } else { Join-Path (Get-Location) "dpcpp-cuda-win" } }

$work = Join-Path ([System.IO.Path]::GetTempPath()) "mom-dpcpp-dl"
New-Item -ItemType Directory -Force $work | Out-Null
$tarball = Join-Path $work $Asset
$shafile = "$tarball.sha256"

# Prefer `gh` (preinstalled on GitHub runners, uses the job token); fall back to the public download URL.
$gh = Get-Command gh -ErrorAction SilentlyContinue
if ($gh) {
  & gh release download $Tag --repo $Repo --pattern "$Asset" --pattern "$Asset.sha256" --dir $work --clobber
  if ($LASTEXITCODE -ne 0) { throw "gh release download failed ($LASTEXITCODE)." }
} else {
  $base = "https://github.com/$Repo/releases/download/$Tag"
  Invoke-WebRequest "$base/$Asset"        -OutFile $tarball
  Invoke-WebRequest "$base/$Asset.sha256" -OutFile $shafile
}

# Verify integrity (the sidecar is "<sha256>  <name>").
$expected = ((Get-Content $shafile -Raw) -split '\s+')[0].Trim().ToLower()
$actual   = (Get-FileHash $tarball -Algorithm SHA256).Hash.ToLower()
if ($expected -ne $actual) { throw "SHA256 mismatch for ${Asset}: expected $expected, got $actual." }

# Extract (gzip tar; the runner's bundled tar.exe handles it). Re-extract clean each run.
if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
New-Item -ItemType Directory -Force $Dest | Out-Null
& "$env:SystemRoot\system32\tar.exe" -xzf $tarball -C $Dest
if ($LASTEXITCODE -ne 0) { throw "tar extract failed ($LASTEXITCODE)." }

$clang = Join-Path $Dest "bin\clang++.exe"
if (-not (Test-Path $clang)) { throw "clang++.exe not found under $Dest\bin after extract." }

$resolved = (Resolve-Path $Dest).Path
$env:MOM_DPCPP_DIR = $resolved
if ($env:GITHUB_ENV) { "MOM_DPCPP_DIR=$resolved" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8 }
Write-Host "Restored DPC++ CUDA toolchain to $resolved"
Write-Output $resolved
