# Build the miner's SYCL DLL with AdaptiveCpp's generic/SSCP target. The same
# portable device image is JIT-compiled by either the CUDA or HIP runtime.
param(
  [string]$RepoRoot = '',
  [Parameter(Mandatory=$true)][string]$ToolchainDir,
  [ValidateSet('hip','cuda')][string]$Backend = 'hip',
  [string]$HipPath = 'C:\Program Files\AMD\ROCm\7.1',
  [string]$CudaPath = $env:CUDA_PATH,
  [string]$OutDir = 'build\Release'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if (-not $RepoRoot) {
  # Windows PowerShell 5.1 does not populate $PSScriptRoot while evaluating parameter defaults.
  # Resolve it after parameter binding so this script works both directly and through the wrappers.
  $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
}
Set-Location $RepoRoot

# The VM may launch PowerShell from a generic or x86 Visual Studio environment.
# AdaptiveCpp invokes clang/lld underneath, so an x86 LIB here silently selects
# the wrong CRT and only fails at the final x64 DLL link. Always import vcvars64.
$vcvars = 'C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }
$vcenv = & cmd.exe /d /s /c "call `"$vcvars`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'vcvars64.bat failed' }
foreach ($line in $vcenv) {
  if ($line -match '^([^=]+)=(.*)$') {
    [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
  }
}

function Invoke-Checked([scriptblock]$Command, [string]$Name) {
  & $Command
  if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

$acpp = Join-Path $ToolchainDir 'bin\acpp'
$clang = Join-Path $ToolchainDir 'bin\clang.exe'
if (-not (Test-Path $acpp)) { throw "AdaptiveCpp wrapper not found at $acpp" }
if (-not (Test-Path $clang)) { throw "AdaptiveCpp clang not found at $clang" }

$backendDefines = @()
if ($Backend -eq 'hip') { $backendDefines += '-DMOM_SYCL_HAS_HIP' }
if ($Backend -eq 'cuda') { $backendDefines += '-DMOM_SYCL_ADAPTIVECPP_CUDA' }
# Do not define MOM_SYCL_HAS_CUDA for AdaptiveCpp. That macro enables the
# DPC++ runtime-compiled sycl::kernel ProgPoW path; AdaptiveCpp generic/SSCP
# must use the portable kernel path, as does the Linux AdaptiveCpp build.
$backendFlags = @()
if ($Backend -eq 'hip') {
  if (-not (Test-Path "$HipPath\lib\amdhip64.lib")) { throw "HIP SDK is incomplete at $HipPath" }
  $env:ROCM_PATH = $HipPath
  $env:HIP_PATH = $HipPath
  $env:PATH = "$HipPath\bin;$env:PATH"
  $backendFlags = @("-I$HipPath\include", "-L$HipPath\lib", '-lamdhip64')
} elseif ($CudaPath) {
  $env:CUDA_PATH = $CudaPath
  $env:PATH = "$CudaPath\bin;$env:PATH"
  $backendFlags = @("-I$CudaPath\include", "-L$CudaPath\lib\x64", '-lnvrtc', '-lcuda')
}
$env:ACPP_VISIBILITY_MASK = $Backend
$env:PATH = "$ToolchainDir\bin;$env:PATH"

$obj = Join-Path $RepoRoot "obj-acpp-$Backend"
Remove-Item $obj -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $obj, (Join-Path $RepoRoot $OutDir) | Out-Null
$common = @('--acpp-targets=generic','-std=c++20','-O3','-ffp-contract=off','-DNDEBUG','-D_CRT_SECURE_NO_WARNINGS',
  '-DMOM_SYCL_BUILD','-DMOM_SYCL_ADAPTIVECPP') + $backendDefines + @('-DNOMINMAX',
  '-DWIN32_LEAN_AND_MEAN','-fno-strict-aliasing',"-I$(Join-Path $RepoRoot 'xmrig')")
if ($Backend -eq 'hip') { $common += '-D__HIP_PLATFORM_AMD__', "-I$HipPath\include" }
$sources = [ordered]@{
  lib         = 'sycl\lib.cpp'
  ethash      = 'sycl\etchash\ethash.cpp'
  etchash     = 'sycl\etchash\etchash.cpp'
  autolykos2  = 'sycl\autolykos2\autolykos2.cpp'
  pearlhash   = 'sycl\pearlhash\pearlhash.cpp'
  c29         = 'sycl\c29\c29.cpp'
  cn_gpu      = 'sycl\cn_gpu\cn_gpu.cpp'
  kawpow      = 'sycl\kawpow\kawpow.cpp'
  fishhash    = 'sycl\fishhash\fishhash.cpp'
  zelhash     = 'sycl\zelhash\zelhash.cpp'
  beamhash3   = 'sycl\beamhash3\beamhash3.cpp'
  blake2b     = 'sycl\c29\blake2b.cpp'
}
$objects = @()
foreach ($entry in $sources.GetEnumerator()) {
  $source = $entry.Key
  $object = Join-Path $obj "$source.obj"
  Invoke-Checked { & python.exe $acpp @common -c $entry.Value -o $object } "AdaptiveCpp compile $source"
  $objects += $object
}
foreach ($source in @('xmrig\base\crypto\sha3.cpp','xmrig\base\crypto\keccak.cpp')) {
  $name = [IO.Path]::GetFileNameWithoutExtension($source)
  $object = Join-Path $obj "$name.obj"
  Invoke-Checked { & python.exe $acpp @common -c $source -o $object } "AdaptiveCpp compile $source"
  $objects += $object
}
$blakeObject = Join-Path $obj 'blake2brx.obj'
Invoke-Checked { & $clang -O3 -DNDEBUG -D_CRT_SECURE_NO_WARNINGS "-I$(Join-Path $RepoRoot 'xmrig')" -c `
  'xmrig\crypto\randomx\blake2\blake2b.c' -o $blakeObject } 'compile blake2b.c'
$objects += $blakeObject

$out = Join-Path $RepoRoot (Join-Path $OutDir 'sycl.dll')
Invoke-Checked { & python.exe $acpp '--acpp-targets=generic' -shared @objects @backendFlags -o $out } 'AdaptiveCpp link sycl.dll'
Write-Host ("Built {0} ({1:N1} MB, AdaptiveCpp generic/{2})" -f $out, ((Get-Item $out).Length/1MB), $Backend)
