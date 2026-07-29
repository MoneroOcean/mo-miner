param(
  [Parameter(Mandatory = $true)][string]$Archive,
  [Parameter(Mandatory = $true)]
  [ValidateSet("intel-windows", "nvidia-windows", "amd-windows")]
  [string]$Platform
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = (Get-Location).Path
$archivePath = (Get-Item (Join-Path $root $Archive)).FullName
$work = Join-Path $env:TEMP "mom-release-deploy"

Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $work | Out-Null
Expand-Archive -Force -Path $archivePath -DestinationPath $work
$releaseName = [IO.Path]::GetFileNameWithoutExtension($archivePath) -replace "-win$", ""
$release = Join-Path $work $releaseName

$vendor = $Platform.Split("-")[0]
$env:MOM_GPU_BACKEND = $vendor
$env:MOM_GPU_TEST_VENDORS = $vendor
$env:MOM_REQUIRE_GPU_TESTS = "1"
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
  & .\install.bat
  if ($LASTEXITCODE -ne 0) {throw "Release installer failed with exit code $LASTEXITCODE"}
} finally {
  Pop-Location
}

if ($env:MOM_DEPLOY_SKIP_VECTORS -ne "1") {
  & (Join-Path $root ".github\workflows\scripts\test-release-windows.ps1") `
    -Archive $archivePath -Suite gpu-discrete
  if ($LASTEXITCODE -ne 0) {throw "Windows GPU vector gate failed with exit code $LASTEXITCODE"}
}

$node = Join-Path $release "mom-node.exe"
$miner = Join-Path $release "mom.cmd"
$gate = Join-Path $root "scripts\check-release-performance.js"
$readme = Join-Path $root "README.md"
$gateArgs = @($gate, "--miner", $miner, "--readme", $readme,
  "--platform", $Platform, "--margin", "0.05")
if ($env:MOM_DEPLOY_ALGO) {$gateArgs += @("--algo", $env:MOM_DEPLOY_ALGO)}
& $node @gateArgs
if ($LASTEXITCODE -ne 0) {throw "Windows performance gate failed with exit code $LASTEXITCODE"}
