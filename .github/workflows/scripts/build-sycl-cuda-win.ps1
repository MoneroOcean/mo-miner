# Build the UNIFIED sycl.dll (Intel spir64 + NVIDIA nvptx in one DLL) with the from-source intel/llvm
# `--cuda` clang restored by restore-toolchain-win.ps1, then drop it into build\Release so packaging
# ships it instead of the Intel-only (MSBuild/icx) sycl.dll. This is the Windows counterpart of the Linux
# combined build's clang `-fsycl` device step; the kernel sources are byte-identical (the only Windows
# source delta is kawpow_jit.inc's module-dir lookup). Mirrors the validated bring-up buildsycl.bat.
param(
  [string]$RepoRoot     = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path,
  [string]$ToolchainDir = $env:MOM_DPCPP_DIR,
  [string]$CudaPath     = $env:CUDA_PATH,
  [string]$CudaArch     = "nvidia_gpu_sm_80",   # single low arch; driver JITs PTX forward to the real GPU
  [string]$OutDir       = "build\Release"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location $RepoRoot

if (-not $ToolchainDir) { throw "ToolchainDir not set (run restore-toolchain-win.ps1 first, or pass -ToolchainDir)." }
$clang  = Join-Path $ToolchainDir "bin\clang++.exe"
$clangc = Join-Path $ToolchainDir "bin\clang.exe"
if (-not (Test-Path $clang)) { throw "clang++.exe not found at $clang." }
if (-not $CudaPath -or -not (Test-Path $CudaPath)) {
  throw "CUDA_PATH ('$CudaPath') is unset/missing; the nvptx target + kawpow JIT need CUDA libdevice."
}

# The clang driver links sycl.dll with lld-link + the MSVC CRT/Windows SDK, so it needs the MSVC build
# environment. Import vcvars64 into this process (mirrors how build-windows.ps1 imports oneAPI setvars).
function Import-VcVars64 {
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  $vsRoot = $null
  if (Test-Path $vswhere) {
    $vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null | Select-Object -First 1
  }
  $candidates = @()
  if ($vsRoot) { $candidates += (Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat") }
  $candidates += "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  $vcvars = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $vcvars) { throw "vcvars64.bat not found (need VS 2022 C++ build tools)." }
  $lines = & cmd.exe /d /s /c "call `"$vcvars`" >nul && set"
  if ($LASTEXITCODE -ne 0) { throw "vcvars64 failed ($LASTEXITCODE)." }
  foreach ($line in $lines) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process") }
  }
}
Import-VcVars64
$env:CUDA_PATH = $CudaPath
$env:PATH = "$CudaPath\bin;$env:PATH"

$obj = Join-Path $RepoRoot "obj"
if (Test-Path $obj) { Remove-Item -Recurse -Force $obj }
New-Item -ItemType Directory -Force $obj | Out-Null
New-Item -ItemType Directory -Force $OutDir | Out-Null

$inc = "-I" + (Join-Path $RepoRoot "xmrig")
# SYCL device TU flags (match buildsycl.bat / the Linux dpcpp-combined build) and the host-helper flags.
$F = @("-std=c++20","-O3","-ffp-contract=off","-fsycl-embed-ir","-DNDEBUG","-DMOM_SYCL_BUILD",
       "-DNOMINMAX","-DWIN32_LEAN_AND_MEAN","-DMOM_SYCL_HAS_CUDA","-DMOM_PEARL_HAS_ESIMD",
       "-fno-strict-aliasing",$inc)
$H = @("-std=c++20","-O3","-DNDEBUG","-DMOM_SYCL_BUILD","-DNOMINMAX","-DWIN32_LEAN_AND_MEAN",$inc)
$targets = "spir64,$CudaArch"

function Invoke-Clang { param([string[]]$ClangArgs, [string]$What)
  # NB: do not name this $Args -- that is a reserved PowerShell automatic variable and the splat
  # would silently expand to nothing ("clang++: error: no input files").
  & $clang @ClangArgs
  if ($LASTEXITCODE -ne 0) { throw "clang failed compiling $What ($LASTEXITCODE)." }
}

# Main SYCL TUs -> spir64 + nvptx.
$main = @(
  "lib",
  "ethash",
  "etchash",
  "autolykos2",
  "pearl",
  "c29",
  "cn-gpu",
  "kawpow",
  "kheavyhash",
  "fishhash",
  "equihash125_4",
  "beamhash3",
  "blake2b"
)
$objs = @()
foreach ($s in $main) {
  $o = Join-Path $obj "$s.obj"
  Write-Host "  [spir64+nvptx] $s"
  Invoke-Clang (@("-fsycl","-fsycl-targets=$targets") + $F + @("-c","sycl\$s.cpp","-o",$o)) $s
  $objs += $o
}
# pearl ESIMD TU -> spir64 only (ESIMD can't share -fsycl-targets with nvptx; dispatched at runtime).
$pe = Join-Path $obj "pearl_esimd.obj"
Write-Host "  [spir64] pearl_esimd"
Invoke-Clang (@("-fsycl","-fsycl-targets=spir64") + $F + @("-DPEARL_ESIMD","-c","sycl\pearl_esimd.cpp","-o",$pe)) "pearl_esimd"
$objs += $pe
# Host helpers the sycl target also needs (no -fsycl).
$sha3 = Join-Path $obj "sha3.obj"; Invoke-Clang ($H + @("-c","xmrig\base\crypto\sha3.cpp","-o",$sha3)) "sha3"
$keccak = Join-Path $obj "keccak.obj"; Invoke-Clang ($H + @("-c","xmrig\base\crypto\keccak.cpp","-o",$keccak)) "keccak"
$objs += $sha3, $keccak
$b2b = Join-Path $obj "blake2brx.obj"
& $clangc "-O3" "-DNDEBUG" $inc "-c" "xmrig\crypto\randomx\blake2\blake2b.c" "-o" $b2b
if ($LASTEXITCODE -ne 0) { throw "clang failed compiling blake2b.c ($LASTEXITCODE)." }
$objs += $b2b

# Link the unified sycl.dll.
$out = Join-Path $OutDir "sycl.dll"
Write-Host "  [link] $out"
& $clang "-fsycl" "-fsycl-targets=$targets" "-shared" @objs "-o" $out
if ($LASTEXITCODE -ne 0) { throw "clang failed linking sycl.dll ($LASTEXITCODE)." }
if (-not (Test-Path $out)) { throw "sycl.dll was not produced at $out." }
Write-Host ("Built unified sycl.dll ({0:N1} MB, spir64+$CudaArch)" -f ((Get-Item $out).Length/1MB))
