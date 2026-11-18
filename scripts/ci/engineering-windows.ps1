# Seal the Windows Engineering CPU addon and portable SYCL runtime, then re-admit both on the test runner.
# Artifact files carry the run and attempt so a re-run never reuses or collides with an earlier upload.
param(
  [Parameter(Mandatory = $true)][ValidateSet('seal-cpu', 'seal-portable', 'stage')][string]$Action
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$suffix = "$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT"
$cpuName = "mom-$suffix.node"
$portableName = "engineering-dpcpp-opencl-$suffix.zip"
$portableDir = 'build/win/compilers/dpcpp-opencl'

function Get-Sha256([string]$Path) {
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Output-Value([string]$Key, [string]$Value) {
  "$Key=$Value" | Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
}

if ($Action -eq 'seal-cpu') {
  $node = 'build/win/Release/mom.node'
  if (-not (Test-Path -LiteralPath $node -PathType Leaf)) { throw "Missing CPU addon: $node" }
  $target = Join-Path $env:RUNNER_TEMP $cpuName
  Copy-Item -LiteralPath $node -Destination $target
  Write-Output-Value 'node_sha256' (Get-Sha256 $target)
  exit 0
}

if ($Action -eq 'seal-portable') {
  $files = @(Get-ChildItem -LiteralPath $portableDir -File -Recurse)
  if (-not ($files | Where-Object Name -CEQ 'sycl.dll')) { throw 'Portable runtime omitted sycl.dll.' }
  if ($files | Where-Object Name -CEQ 'mom.node') { throw 'Portable runtime must not contain mom.node.' }
  $archive = Join-Path $env:RUNNER_TEMP $portableName
  Compress-Archive -Path (Join-Path $portableDir '*') -DestinationPath $archive -CompressionLevel NoCompression
  Write-Output-Value 'runtime_sha256' (Get-Sha256 $archive)
  exit 0
}

$cpuNode = Join-Path $env:RUNNER_TEMP "incoming/$cpuName"
$runtimeZip = Join-Path $env:RUNNER_TEMP "incoming/$portableName"
foreach ($value in @($env:CPU_NODE_SHA256, $env:PORTABLE_RUNTIME_SHA256)) {
  if ($value -notmatch '^[0-9a-f]{64}$') { throw 'Producer SHA-256 output is malformed.' }
}
if ((Get-Sha256 $cpuNode) -ne $env:CPU_NODE_SHA256) { throw 'Downloaded CPU addon SHA-256 mismatch.' }
if ((Get-Sha256 $runtimeZip) -ne $env:PORTABLE_RUNTIME_SHA256) { throw 'Downloaded portable runtime SHA-256 mismatch.' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($runtimeZip)
try {
  $names = @($zip.Entries | ForEach-Object FullName)
  if ($names -notcontains 'sycl.dll' -or $names -contains 'mom.node' -or
      @($names | Where-Object { $_ -notmatch '^[A-Za-z0-9_.+-]+$' -or $_ -in @('.', '..') }).Count -gt 0 -or
      @($names | Select-Object -Unique).Count -ne $names.Count) {
    throw 'Portable runtime archive has an unexpected member set.'
  }
} finally { $zip.Dispose() }
New-Item -ItemType Directory -Force $portableDir, 'build/win/Release' | Out-Null
Expand-Archive -LiteralPath $runtimeZip -DestinationPath $portableDir -Force
$staged = @(Get-ChildItem -LiteralPath $portableDir -File -Recurse)
if (@($staged | Where-Object Name -CEQ 'mom.node').Count -ne 0 -or @($staged | Where-Object Name -CEQ 'sycl.dll').Count -ne 1) {
  throw 'Staged portable runtime is incomplete or contains a node addon.'
}
Copy-Item -LiteralPath $cpuNode -Destination 'build/win/Release/mom.node' -Force
Copy-Item -LiteralPath $cpuNode -Destination (Join-Path $portableDir 'mom.node') -Force
