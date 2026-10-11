param(
  [Parameter(Mandatory = $true)][string]$Archive,
  [Parameter(Mandatory = $true)]
  [ValidateSet("intel-windows", "nvidia-windows", "amd-windows")]
  [string]$Platform,
  [switch]$SkipAbsentGpu
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$root = (Get-Location).Path
$archivePath = (Get-Item (Join-Path $root $Archive)).FullName
$tempBase = if ($env:TEMP) { $env:TEMP } else { [IO.Path]::GetTempPath() }
$work = Join-Path $tempBase ("mom-release-deploy-{0}" -f [Guid]::NewGuid().ToString('N'))
$deployDebug = $env:MOM_DEPLOY_DEBUG -eq "1"
# Normal auxiliary captures share extraction cleanup; debug captures survive it.
$debugDir = if ($deployDebug) {
  Join-Path $root ("build\deploy-win\debug\{0}-{1}" -f $Platform, [Guid]::NewGuid().ToString('N'))
} else { Join-Path $work "diagnostics" }
$vectorLogPath = Join-Path $debugDir "vectors.log"
$previousEvidenceDir = $env:MOM_RELEASE_PERF_EVIDENCE_DIR
$workCreated = $false

try {
  [IO.Directory]::CreateDirectory($work) | Out-Null
  $workCreated = $true
  New-Item -ItemType Directory -Path $debugDir -ErrorAction Stop | Out-Null
  if ($deployDebug -and -not $env:MOM_RELEASE_PERF_EVIDENCE_DIR) {
    $env:MOM_RELEASE_PERF_EVIDENCE_DIR = Join-Path $debugDir "performance"
  }
  if ($deployDebug) {
    Write-Host "[deploy] Debug logs: $debugDir"
    [ordered]@{
      startedAt = [DateTime]::UtcNow.ToString('o'); platform = $Platform
      archiveSha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
      readmeSha256 = (Get-FileHash -LiteralPath (Join-Path $root 'README.md') -Algorithm SHA256).Hash
      gateSha256 = (Get-FileHash -LiteralPath (Join-Path $root 'scripts\check-release-performance.js') -Algorithm SHA256).Hash
      helperSha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
      zelhashProfilerPresent = (Test-Path Env:MOM_ZELHASH_PROF)
      zelhashPerfLoggerPresent = (Test-Path Env:MOM_ZELHASH_PERF)
      zelhashSlots = $(if ($env:MOM_ZELHASH_SLOTS -match '^[0-9]{1,10}$') { $env:MOM_ZELHASH_SLOTS } else { $null })
      nativeOverridePresent = (Test-Path Env:MOM_NATIVE_PATH)
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $debugDir 'context.json') -Encoding UTF8
  }
  Expand-Archive -Force -Path $archivePath -DestinationPath $work
  $releaseName = [IO.Path]::GetFileNameWithoutExtension($archivePath) -replace "-win$", ""
  $release = Join-Path $work $releaseName

  $vendor = $Platform.Split("-")[0]
  $env:MOM_GPU_BACKEND = $vendor
  $env:MOM_GPU_TEST_VENDORS = $vendor
  $env:MOM_REQUIRE_GPU_TESTS = "1"
  if ($env:MOM_DEPLOY_ALGO) {$env:MOM_GPU_TEST_ALGO = $env:MOM_DEPLOY_ALGO}
  if ($vendor -eq "intel") {
    $env:ONEAPI_DEVICE_SELECTOR = "level_zero:gpu"
    # MoM filters discrete Intel devices; Level Zero ordinal 0 may instead be the host's iGPU.
  }
  if ($vendor -eq "amd") {
    $env:ACPP_VISIBILITY_MASK = "hip"
    $env:ACPP_APPDB_DIR = Join-Path $release "build\win\.acpp"
  }

  Push-Location $release
  $installCapturePreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $global:LASTEXITCODE = $null
    $installLog = & .\install.bat 2>&1
    $installExit = if ($null -eq $global:LASTEXITCODE) { 1 } else { $global:LASTEXITCODE }
    $installLog | Out-File -LiteralPath (Join-Path $debugDir 'install.log') -Encoding UTF8 -ErrorAction Stop
    if ($installExit -ne 0) {
      $installLog | Write-Host
      throw "Release installer failed with exit code $installExit"
    }
  } finally {
    $ErrorActionPreference = $installCapturePreference
    Pop-Location
  }

  if ($SkipAbsentGpu) {
    $discoveryPreference = $ErrorActionPreference
    try {
      # Native stderr is diagnostic; only the exit status and canonical report decide admission.
      $ErrorActionPreference = "Continue"
      $global:LASTEXITCODE = $null
      $discoveryOutput = & (Join-Path $release "mom.cmd") algorithms 2>&1
      $discoveryExit = if ($null -eq $global:LASTEXITCODE) { 1 } else { $global:LASTEXITCODE }
    } finally {
      $ErrorActionPreference = $discoveryPreference
    }
    if ($discoveryExit -ne 0) {
      $discoveryOutput | Write-Host
      throw "Windows GPU discovery failed with exit code $discoveryExit"
    }
    $markers = @($discoveryOutput | Where-Object { "$_" -match '^MOM_ALGORITHMS ' })
    if ($markers.Count -ne 1) { throw "Windows GPU discovery requires one MOM_ALGORITHMS report" }
    $json = ("$($markers[0])" -replace '^MOM_ALGORITHMS ', '').Trim()
    # PowerShell can unwrap a one-element JSON array; require an object before conversion.
    if (-not $json.StartsWith('{') -or -not $json.EndsWith('}')) {
      throw "Windows GPU discovery returned an invalid algorithms object"
    }
    $params = $json | ConvertFrom-Json
    if ($params -isnot [pscustomobject] -or @($params.PSObject.Properties).Count -eq 0) {
      throw "Windows GPU discovery returned an invalid algorithms object"
    }
    $gpuParams = $false
    foreach ($prop in $params.PSObject.Properties) {
      if ($prop.Value -isnot [string] -or -not $prop.Value -or
          $prop.Value -notmatch '^(cpu\d*|gpu\d+)(?=$|[*^:\[,])' -or
          $prop.Value -match '(^|,)[^,]*(\*0|\^0)(,|$)' -or
          $prop.Value -match '[\r\n\x00]') {
        throw "Windows GPU discovery returned an invalid algorithms entry"
      }
      if ($prop.Value -match '(^|,)gpu\d+') { $gpuParams = $true }
    }
    $gpuDevices = @($discoveryOutput | Where-Object { "$_" -match '^gpu\d+:\s+.+$' })
    if (-not $gpuParams) {
      if ($gpuDevices.Count -ne 0) { throw "Windows GPU discovery reported devices without GPU jobs" }
      Write-Host "SKIP ${Platform}: no usable discrete GPU discovered"
      # Exit 3 is reserved for a successful, validated native-only absence report.
      # VM callers omit SkipAbsentGpu and retain the strict required-GPU gates.
      exit 3
    }
    if ($gpuDevices.Count -eq 0) { throw "Windows GPU discovery reported jobs without GPU devices" }
  }

  if ($env:MOM_DEPLOY_SKIP_VECTORS -ne "1") {
    $vectorExit = 0
    $vectorCapturePreference = $ErrorActionPreference
    try {
      # The vector script mutates loader paths; contain them before performance worker launch.
      # Windows PowerShell native stderr is diagnostic output, not an exit-code verdict.
      $ErrorActionPreference = "Continue"
      # Native commands update global status, even when a caller shadows it locally.
      $global:LASTEXITCODE = $null
      $vectorPowerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
      & $vectorPowerShell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $root ".github\workflows\scripts\test-release-windows.ps1") `
        -Archive $archivePath -Suite gpu-discrete *> $vectorLogPath
      $vectorExit = if ($null -eq $global:LASTEXITCODE) { 1 } else { $global:LASTEXITCODE }
    } catch {
      $_ | Out-String | Add-Content -LiteralPath $vectorLogPath
      $vectorExit = 1
    } finally {
      $ErrorActionPreference = $vectorCapturePreference
    }
    $vectorLog = Get-Content -LiteralPath $vectorLogPath
    $counts = @{}
    $vectorLog | ForEach-Object {
      if ("$_" -match '(tests|pass|fail|skipped) (\d+)$') {
        $counts[$Matches[1]] = $Matches[2]
      }
    }
    if ($counts.tests) {
      Write-Host "MOM_TEST_SUMMARY $($counts.tests) $($counts.pass) $($counts.fail) $($counts.skipped)"
    }
    if ($vectorExit -ne 0) {
      $vectorLog | Write-Host
      throw "Windows GPU vector gate failed with exit code $vectorExit"
    }
    Write-Host "  $([char]0x2714) Stability checks passed"
  }

  $node = Join-Path $release "mom-node.exe"
  $miner = Join-Path $release "mom.cmd"
  $gate = Join-Path $root "scripts\check-release-performance.js"
  $readme = Join-Path $root "README.md"
  $env:MOM_BENCHMARK_GRACEFUL_ONLY = "1"
  $gateArgs = @($gate, "--miner", $miner, "--readme", $readme,
    "--platform", $Platform, "--margin", "0.05")
  if ($env:MOM_DEPLOY_ALGO) {$gateArgs += @("--algo", $env:MOM_DEPLOY_ALGO)}
  $gateCapturePreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $global:LASTEXITCODE = $null
    # PowerShell 5.1 wraps native stderr in ErrorRecord objects. Preserve the message text so
    # progress/deadline lines reach the host filter without an injected PowerShell stack trace.
    & $node @gateArgs 2>&1 | ForEach-Object { $_.ToString() } |
      Tee-Object -FilePath (Join-Path $debugDir 'performance.log') -ErrorAction Stop
    $gateExit = if ($null -eq $global:LASTEXITCODE) { 1 } else { $global:LASTEXITCODE }
  } finally {
    $ErrorActionPreference = $gateCapturePreference
  }
  if ($gateExit -ne 0) {throw "Windows performance gate failed with exit code $gateExit"}
} finally {
  $env:MOM_RELEASE_PERF_EVIDENCE_DIR = $previousEvidenceDir
  if ($workCreated) {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
  }
}
