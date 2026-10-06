# Restore the prebuilt from-source intel/llvm `--cuda` DPC++ toolchain (the GitHub release asset
# produced by package-toolchain-win.ps1) so CI can build the unified spir64+nvptx sycl.dll WITHOUT the
# ~1.5 h LLVM build (which cannot finish in a GitHub-hosted Windows job: 6 h cap, 2-4 vCPU, see
# scripts/windows-multicompiler.md). Downloads the asset, verifies its SHA256, and extracts bin/lib/include
# to -Dest. Emits the resolved toolchain dir on stdout (last line) and as $env:MOM_DPCPP_DIR / GITHUB_ENV.
param(
  [string]$Repo  = "MoneroOcean/mo-miner",
  [string]$Tag   = "toolchain-win-dpcpp-cuda",
  [string]$Asset = "dpcpp-cuda-win-v7.1.1.tar.gz",
  [string]$ExpectedSha256 = "8e5e9de06ed46c5e28aac4c574811a68f0a85dbb1cb65508187b0b8cf2e573cb",
  [string]$Dest  = ""
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
. (Join-Path $repoRoot 'scripts\windows-install-helpers.ps1')
if ([string]::IsNullOrWhiteSpace($Asset) -or
    $Asset -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
  throw "Asset must be a leaf filename: $Asset"
}
if ([string]::IsNullOrWhiteSpace($ExpectedSha256) -or
    $ExpectedSha256 -notmatch '^[0-9A-Fa-f]{64}$') {
  throw 'ExpectedSha256 must be exactly 64 hexadecimal characters.'
}
$expected = $ExpectedSha256.ToLowerInvariant()
# Only recorded owned predecessors may migrate to the pinned stable-source SDK cohort.
$knownPriorExpected = '7a61b81cc15484656c80d3927dfc890d14d98689d64f05e3b88f5b45a1e4bb34'
$knownMatchedExpected = '81b116580a84ac29221c459ffa354045fae69d95338a8c1db70b76d16be7e472'
$acceptKnownPrior = $expected -eq $knownMatchedExpected -or $expected -eq '8e5e9de06ed46c5e28aac4c574811a68f0a85dbb1cb65508187b0b8cf2e573cb'
$acceptKnownMatched = $expected -eq '8e5e9de06ed46c5e28aac4c574811a68f0a85dbb1cb65508187b0b8cf2e573cb'
# Default destination only when -Dest is not given: RUNNER_TEMP in CI, else under the cwd for local runs.
# (Must not clobber an explicit -Dest -- that was a bug found provisioning a dev box.)
if (-not $Dest) {
  $Dest = if ($env:RUNNER_TEMP) { Join-Path $env:RUNNER_TEMP "dpcpp-cuda-win" } else { Join-Path (Get-Location) "dpcpp-cuda-win" } }
$Dest = [System.IO.Path]::GetFullPath($Dest)
$destComparable = $Dest.TrimEnd([char[]]@('\', '/'))
$rootComparable = ([System.IO.Path]::GetPathRoot($Dest)).TrimEnd([char[]]@('\', '/'))
if ($destComparable.Equals($rootComparable, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing to use a filesystem root as -Dest: $Dest"
}
Assert-NoReparseAncestor $Dest 'Toolchain destination'

$destItem = Get-Item -LiteralPath $Dest -Force -ErrorAction SilentlyContinue
$destExists = $null -ne $destItem
$destHasContents = $false
if ($destExists) {
  if (($destItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Refusing reparse-point toolchain destination: $Dest"
  }
  if (-not $destItem.PSIsContainer) {
    throw "Refusing toolchain destination that is not a directory: $Dest"
  }
  $destChildren = @(Get-ChildItem -LiteralPath $Dest -Force -ErrorAction Stop)
  $destHasContents = $destChildren.Count -gt 0
  if ($destHasContents) {
    $ownershipMarker = Join-Path $Dest '.mom-toolchain-sha256'
    if (-not (Test-MomMarker $ownershipMarker $expected) -and
        -not ($acceptKnownPrior -and (Test-MomMarker $ownershipMarker $knownPriorExpected)) -and
        -not ($acceptKnownMatched -and (Test-MomMarker $ownershipMarker $knownMatchedExpected))) {
      throw "Refusing to replace unowned nonempty toolchain destination: $Dest"
    }
  }
}

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("mom-dpcpp-dl-{0}" -f [Guid]::NewGuid().ToString('N'))
$workCreated = $false

try {
  [IO.Directory]::CreateDirectory($work) | Out-Null
  $workCreated = $true
  $tarball = Join-Path $work $Asset
  $staging = Join-Path $work 'staging'
  [IO.Directory]::CreateDirectory($staging) | Out-Null

  # The release is public. curl's bounded retries are more reliable for this 1+ GiB payload than one
  # `gh release download` attempt, and make local/dev and hosted-runner behavior identical.
  $url = "https://github.com/$Repo/releases/download/$Tag/$Asset"
  & curl.exe -fL --retry 5 --retry-delay 5 -o $tarball $url
  if ($LASTEXITCODE -ne 0) { throw "DPC++ download failed ($LASTEXITCODE): $url" }

  # Verify against the source-controlled digest. A mutable release sidecar would only prove that the
  # payload and sidecar changed together, and could silently replace the compiler used for releases.
  $actual   = (Get-FileHash $tarball -Algorithm SHA256).Hash.ToLower()
  if ($expected -ne $actual) { throw "SHA256 mismatch for ${Asset}: expected $expected, got $actual." }

  # Extract (gzip tar; the runner's bundled tar.exe handles it) into staging before touching Dest.
  & "$env:SystemRoot\system32\tar.exe" -xzf $tarball -C $staging
  if ($LASTEXITCODE -ne 0) { throw "tar extract failed ($LASTEXITCODE)." }

  if (-not (Test-MomDpcppToolchain $staging)) {
    throw 'DPC++ toolchain is incomplete under staging after extract.'
  }
  # The development image persists across source revisions. Record the verified payload identity so
  # install-dev.ps1 can replace a stale-but-otherwise-functional compiler instead of accepting it.
  $expected | Set-Content -LiteralPath (Join-Path $staging '.mom-toolchain-sha256') -NoNewline

  $destParent = Split-Path -Parent $Dest
  Assert-NoReparseAncestor $Dest 'Toolchain destination'
  if (-not (Test-Path -LiteralPath $destParent -PathType Container)) {
    [IO.Directory]::CreateDirectory($destParent) | Out-Null
  }

  $currentDest = Get-Item -LiteralPath $Dest -Force -ErrorAction SilentlyContinue
  if ($destExists) {
    if ($null -eq $currentDest -or
        ($currentDest.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        -not $currentDest.PSIsContainer) {
      throw "Toolchain destination changed during restore: $Dest"
    }
    $currentChildren = @(Get-ChildItem -LiteralPath $Dest -Force -ErrorAction Stop)
    if ($currentChildren.Count -gt 0) {
      $ownershipMarker = Join-Path $Dest '.mom-toolchain-sha256'
      if (-not (Test-MomMarker $ownershipMarker $expected) -and
          -not ($acceptKnownPrior -and (Test-MomMarker $ownershipMarker $knownPriorExpected)) -and
          -not ($acceptKnownMatched -and (Test-MomMarker $ownershipMarker $knownMatchedExpected))) {
        throw "Refusing to replace unowned nonempty toolchain destination: $Dest"
      }
      Remove-Item -LiteralPath $Dest -Recurse -Force
    } else {
      Remove-Item -LiteralPath $Dest -Force
    }
  } elseif ($null -ne $currentDest) {
    throw "Toolchain destination appeared during restore: $Dest"
  }
  Move-Item -LiteralPath $staging -Destination $Dest

  $resolved = (Resolve-Path $Dest).Path
  $env:MOM_DPCPP_DIR = $resolved
  if ($env:GITHUB_ENV) { "MOM_DPCPP_DIR=$resolved" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8 }
  Write-Host "Restored DPC++ CUDA toolchain to $resolved"
  Write-Output $resolved
} finally {
  if ($workCreated) {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
  }
}
