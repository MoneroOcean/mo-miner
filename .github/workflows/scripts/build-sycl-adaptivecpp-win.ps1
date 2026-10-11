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
. (Join-Path $RepoRoot 'scripts\windows-incremental-build.ps1')
Initialize-MomBuildEnvironment $RepoRoot
. (Join-Path $RepoRoot 'scripts\import-vcvars.ps1')
. (Join-Path $RepoRoot 'scripts\windows-sycl-sources.ps1')
. (Join-Path $RepoRoot 'scripts\windows-install-helpers.ps1')

# The VM may launch PowerShell from a generic or x86 Visual Studio environment.
# AdaptiveCpp invokes clang/lld underneath, so an x86 LIB here silently selects
# the wrong CRT and only fails at the final x64 DLL link. Always import vcvars64.
Import-MomVcVars64

function Invoke-Checked([scriptblock]$Command, [string]$Name) {
  & $Command
  if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

$acpp = Join-Path $ToolchainDir 'bin\acpp'
$clang = Join-Path $ToolchainDir 'bin\clang.exe'
if (-not (Test-Path $acpp)) { throw "AdaptiveCpp wrapper not found at $acpp" }
if (-not (Test-Path $clang)) { throw "AdaptiveCpp clang not found at $clang" }

$backendDefines = @()
if ($Backend -eq 'hip') { $backendDefines += '-DMOM_SYCL_HAS_HIP', '-DMOM_OCTOPUS_HAS_SYCL_NATIVE' }
if ($Backend -eq 'cuda') { $backendDefines += '-DMOM_SYCL_ADAPTIVECPP_CUDA' }
# Do not define MOM_SYCL_HAS_CUDA for AdaptiveCpp. That macro enables the
# DPC++ runtime-compiled sycl::kernel ProgPoW path; AdaptiveCpp generic/SSCP
# must use the portable kernel path, as does the Linux AdaptiveCpp build.
$backendFlags = @()
if ($Backend -eq 'hip') {
  if (-not (Test-MomHipSdk $HipPath)) { throw "HIP SDK is incomplete at $HipPath" }
  $HipPath = (Resolve-Path -LiteralPath $HipPath).Path
  $env:ROCM_PATH = $HipPath
  $env:HIP_PATH = $HipPath
  $env:PATH = "$HipPath\bin;$env:PATH"
  $backendFlags = @("-I$HipPath\include", "-L$HipPath\lib", '-lamdhip64')
} else {
  if (-not (Test-MomCudaCompilerSdk $CudaPath)) { throw "CUDA SDK is incomplete at $CudaPath" }
  $CudaPath = (Resolve-Path -LiteralPath $CudaPath).Path
  $env:CUDA_PATH = $CudaPath
  $env:PATH = "$CudaPath\bin;$env:PATH"
  $backendFlags = @("-I$CudaPath\include", "-L$CudaPath\lib\x64", '-lnvrtc', '-lcuda')
}
$env:ACPP_VISIBILITY_MASK = $Backend
$env:PATH = "$ToolchainDir\bin;$env:PATH"

$obj = Resolve-MomBuildPath (Join-Path $RepoRoot "build\.incremental\sycl-acpp-$Backend")
$publishedOutput = Resolve-MomBuildPath (Join-Path $OutDir 'sycl.dll')
$out = Join-Path $obj 'sycl.dll'
$buildRoot = Resolve-MomBuildPath 'build'
Assert-MomBuildArtifactPath $out $buildRoot
Assert-MomBuildArtifactPath $publishedOutput $buildRoot
New-Item -ItemType Directory -Force $obj, (Split-Path -Parent $publishedOutput) | Out-Null
$publishSeparate = (Resolve-MomBuildPath $publishedOutput) -ne (Resolve-MomBuildPath $out)
if ($publishSeparate) { Remove-Item -LiteralPath $publishedOutput -Force -ErrorAction SilentlyContinue }
$python = (Get-Command python.exe -ErrorAction Stop).Source
$buildContext = Initialize-MomIncrementalBuild $obj @($python, $acpp, $clang, $PSCommandPath,
  (Join-Path $RepoRoot 'scripts\windows-incremental-build.ps1')) @($ToolchainDir,
  $(if ($Backend -eq 'hip') { $HipPath } else { $CudaPath }), $env:VCToolsInstallDir, $env:WindowsSdkDir)
function Invoke-IncrementalCompile([string]$Exe, [string[]]$CompilerArgs, [string]$Source, [string]$Object, [string]$Name) {
  $task = New-MomCompileTask $Exe $CompilerArgs $Source $Object $Name $buildContext
  if ($task.Skip) { Write-Host "  [cached] $Name"; return }
  Remove-Item -LiteralPath $publishedOutput -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $out, (Join-Path $obj 'link.json') -Force -ErrorAction SilentlyContinue
  & $task.Exe @($task.Args)
  if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
  Complete-MomCompileTask $task
}
$common = @('--acpp-targets=generic','-std=c++20','-O3','-ffp-contract=off','-DNDEBUG','-D_CRT_SECURE_NO_WARNINGS',
  '-DMOM_SYCL_BUILD','-DMOM_SYCL_ADAPTIVECPP') + $backendDefines + @('-DNOMINMAX',
  '-DWIN32_LEAN_AND_MEAN','-fno-strict-aliasing',"-I$(Join-Path $RepoRoot 'xmrig')")
if ($Backend -eq 'hip') { $common += '-D__HIP_PLATFORM_AMD__', "-I$HipPath\include" }
if ($Backend -eq 'cuda' -and $CudaPath) { $common += "-I$CudaPath\include" }
$sources = Get-MomWindowsSyclSources
$objects = @()
foreach ($entry in $sources.GetEnumerator()) {
  $source = $entry.Key
  $object = Join-Path $obj "$source.obj"
  Invoke-IncrementalCompile $python (@($acpp) + $common + @('-c', $entry.Value, '-o', $object)) $entry.Value $object "AdaptiveCpp compile $source"
  $objects += $object
}
foreach ($source in @('xmrig\base\crypto\sha3.cpp','xmrig\base\crypto\keccak.cpp')) {
  $name = [IO.Path]::GetFileNameWithoutExtension($source)
  $object = Join-Path $obj "$name.obj"
  Invoke-IncrementalCompile $python (@($acpp) + $common + @('-c', $source, '-o', $object)) $source $object "AdaptiveCpp compile $source"
  $objects += $object
}
$blakeObject = Join-Path $obj 'blake2brx.obj'
Invoke-IncrementalCompile $clang @('-O3', '-DNDEBUG', '-D_CRT_SECURE_NO_WARNINGS', "-I$(Join-Path $RepoRoot 'xmrig')", '-c',
  'xmrig\crypto\randomx\blake2\blake2b.c', '-o', $blakeObject) 'xmrig\crypto\randomx\blake2\blake2b.c' $blakeObject 'compile blake2b.c'
$objects += $blakeObject
$hoohashObject = Join-Path $obj 'hoohash_host.obj'
Invoke-IncrementalCompile $clang @('-O2', '-fno-fast-math', '-ffp-contract=off', '-fno-builtin', '-DNDEBUG',
  '-D_CRT_SECURE_NO_WARNINGS', '-c', 'sycl\hoohash\host_math.c', '-o', $hoohashObject) 'sycl\hoohash\host_math.c' $hoohashObject 'compile hoohash host math'
$objects += $hoohashObject

Invoke-MomIncrementalLink $python (@($acpp, '--acpp-targets=generic', '-shared') + $objects + $backendFlags + @('-o', $out)) $objects $out (Join-Path $obj 'link.json') $buildContext
Remove-MomObsoleteCompileOutputs $obj $objects
if ($publishSeparate) { Copy-Item -LiteralPath $out -Destination $publishedOutput -Force }
Write-Host ("Built {0} ({1:N1} MB, AdaptiveCpp generic/{2})" -f $out, ((Get-Item $out).Length/1MB), $Backend)
