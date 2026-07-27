# Restore the prebuilt from-source intel/llvm `--cuda` DPC++ toolchain (the GitHub release asset
# produced by package-toolchain-win.ps1) so CI can build the unified spir64+nvptx sycl.dll WITHOUT the
# ~1.5 h LLVM build (which cannot finish in a GitHub-hosted Windows job: 6 h cap, 2-4 vCPU, see
# scripts/build-windows-nvidia.md). Downloads the asset, verifies its SHA256, and extracts bin/lib/include
# to -Dest. Emits the resolved toolchain dir on stdout (last line) and as $env:MOM_DPCPP_DIR / GITHUB_ENV.
param(
  [string]$Repo  = "MoneroOcean/mo-miner",
  [string]$Tag   = "toolchain-win-dpcpp-cuda",
  [string]$Asset = "dpcpp-cuda-win.tar.gz",
  [string]$ExpectedSha256 = "7a61b81cc15484656c80d3927dfc890d14d98689d64f05e3b88f5b45a1e4bb34",
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

# The release is public. curl's bounded retries are more reliable for this 1+ GiB payload than one
# `gh release download` attempt, and make local/dev and hosted-runner behavior identical.
$url = "https://github.com/$Repo/releases/download/$Tag/$Asset"
& curl.exe -fL --retry 5 --retry-delay 5 -o $tarball $url
if ($LASTEXITCODE -ne 0) { throw "DPC++ download failed ($LASTEXITCODE): $url" }

# Verify against the source-controlled digest. A mutable release sidecar would only prove that the
# payload and sidecar changed together, and could silently replace the compiler used for releases.
$expected = $ExpectedSha256.Trim().ToLower()
$actual   = (Get-FileHash $tarball -Algorithm SHA256).Hash.ToLower()
if ($expected -ne $actual) { throw "SHA256 mismatch for ${Asset}: expected $expected, got $actual." }

# Extract (gzip tar; the runner's bundled tar.exe handles it). Re-extract clean each run.
if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
New-Item -ItemType Directory -Force $Dest | Out-Null
& "$env:SystemRoot\system32\tar.exe" -xzf $tarball -C $Dest
if ($LASTEXITCODE -ne 0) { throw "tar extract failed ($LASTEXITCODE)." }

$clang = Join-Path $Dest "bin\clang++.exe"
if (-not (Test-Path $clang)) { throw "clang++.exe not found under $Dest\bin after extract." }
# The development image persists across source revisions. Record the verified payload identity so
# install-dev.ps1 can replace a stale-but-otherwise-functional compiler instead of accepting it.
$expected | Set-Content (Join-Path $Dest '.mom-toolchain-sha256') -NoNewline

$resolved = (Resolve-Path $Dest).Path
$env:MOM_DPCPP_DIR = $resolved
if ($env:GITHUB_ENV) { "MOM_DPCPP_DIR=$resolved" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8 }
Write-Host "Restored DPC++ CUDA toolchain to $resolved"
Write-Output $resolved
