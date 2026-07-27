param(
  [string]$ToolchainDir = 'C:\llvm-amd\build\install',
  [string]$HipPath = 'C:\Program Files\AMD\ROCm\7.1',
  [string]$AmdArch = 'gfx1200',
  [switch]$FullGpuSuite
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo

$env:MOM_DPCPP_DIR = $ToolchainDir
$env:HIP_PATH = $HipPath
$env:MOM_AMD_TARGET = $AmdArch
$env:MOM_GPU_BACKEND = 'amd'
$env:MOM_GPU_TEST_VENDORS = 'amd'
$env:ONEAPI_DEVICE_SELECTOR = 'hip:*'
$env:PATH = "$ToolchainDir\bin;$HipPath\bin;$env:PATH"

# Build the normal Windows host addon first, then replace its Intel-only SYCL DLL with the unified
# spir64+AMDGPU image. The latter applies the bug1 GlobalOffsetPass ordering workaround at device link.
if (Test-Path "$repo\build") { Remove-Item "$repo\build" -Recurse -Force }
& "$repo\.github\workflows\scripts\build-windows.ps1"
if ($LASTEXITCODE -ne 0) { throw "Windows host build failed ($LASTEXITCODE)" }
& "$repo\.github\workflows\scripts\build-sycl-cuda-win.ps1" `
  -ToolchainDir $ToolchainDir -CudaPath '' -HipPath $HipPath -AmdArch $AmdArch
if ($LASTEXITCODE -ne 0) { throw "DPC++ AMD SYCL build failed ($LASTEXITCODE)" }

node mom.js algo_params
if ($LASTEXITCODE -ne 0) { throw "AMD device discovery failed ($LASTEXITCODE)" }
if ($FullGpuSuite) {
  npm run test:gpu-discrete
  if ($LASTEXITCODE -ne 0) { throw "AMD GPU suite failed ($LASTEXITCODE)" }
}
