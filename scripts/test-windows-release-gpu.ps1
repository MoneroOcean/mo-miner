param(
  [Parameter(Mandatory = $true)][string]$Archive,
  [Parameter(Mandatory = $true)]
  [ValidateSet("intel-windows", "nvidia-windows", "amd-windows")]
  [string]$Platform
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$root = (Get-Location).Path
$archivePath = (Get-Item (Join-Path $root $Archive)).FullName
$tempBase = if ($env:TEMP) { $env:TEMP } else { [IO.Path]::GetTempPath() }
$work = Join-Path $tempBase ("mom-release-deploy-{0}" -f [Guid]::NewGuid().ToString('N'))
$vectorLogPath = Join-Path $tempBase ("mom-vector-gate-{0}.log" -f [Guid]::NewGuid().ToString('N'))
$workCreated = $false
$vectorLogCreated = $false

try {
  [IO.Directory]::CreateDirectory($work) | Out-Null
  $workCreated = $true
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
    $env:ZE_AFFINITY_MASK = "0"
  }
  if ($vendor -eq "amd") {
    $env:ACPP_VISIBILITY_MASK = "hip"
    $env:ACPP_APPDB_DIR = Join-Path $release "build\win\.acpp"
  }

  Push-Location $release
  try {
    $installLog = & .\install.bat 2>&1
    if ($LASTEXITCODE -ne 0) {
      $installLog | Write-Host
      throw "Release installer failed with exit code $LASTEXITCODE"
    }
  } finally {
    Pop-Location
  }

  if ($env:MOM_DEPLOY_SKIP_VECTORS -ne "1") {
    $vectorExit = 0
    $vectorLogCreated = $true
    try {
      & (Join-Path $root ".github\workflows\scripts\test-release-windows.ps1") `
        -Archive $archivePath -Suite gpu-discrete *> $vectorLogPath
      $vectorExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    } catch {
      $_ | Out-String | Add-Content -LiteralPath $vectorLogPath
      $vectorExit = 1
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
  & $node @gateArgs
  if ($LASTEXITCODE -ne 0) {throw "Windows performance gate failed with exit code $LASTEXITCODE"}
} finally {
  if ($vectorLogCreated) {
    Remove-Item -LiteralPath $vectorLogPath -Force -ErrorAction SilentlyContinue
  }
  if ($workCreated) {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
  }
}
