# Build the unified sycl.dll (Intel spir64 + NVIDIA nvptx, optionally AMD amdgcn) with the from-source
# intel/llvm clang restored by restore-toolchain-win.ps1, then drop it into build\Release so packaging
# ships it instead of the Intel-only (MSBuild/icx) sycl.dll. This is the Windows counterpart of the Linux
# combined build's clang `-fsycl` device step; the kernel sources are byte-identical (the only Windows
# source delta is kawpow_jit.inc's module-dir lookup). Mirrors the validated bring-up buildsycl.bat.
param(
  [string]$RepoRoot     = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path,
  [string]$ToolchainDir = $env:MOM_DPCPP_DIR,
  [string]$CudaPath     = $env:CUDA_PATH,
  [string]$CudaArch     = "nvidia_gpu_sm_80",   # single low arch; driver JITs PTX forward to the real GPU
  [string]$HipPath      = $env:HIP_PATH,
  [string]$AmdArch      = $env:MOM_AMD_TARGET,
  [string]$OutDir       = "build\Release",
  [int]$Jobs            = 0,
  [switch]$PortableOpencl
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location $RepoRoot

function Resolve-BuildJobs([int]$Requested) {
  $jobs = $Requested
  if ($jobs -lt 1 -and $env:MOM_BUILD_JOBS) {
    [void][int]::TryParse($env:MOM_BUILD_JOBS, [ref]$jobs)
  }
  if ($jobs -lt 1) {
    $jobs = [Environment]::ProcessorCount
  }
  return [Math]::Max(1, $jobs)
}

if (-not $ToolchainDir) { throw "ToolchainDir not set (run restore-toolchain-win.ps1 first, or pass -ToolchainDir)." }
$clang  = Join-Path $ToolchainDir "bin\clang++.exe"
$clangc = Join-Path $ToolchainDir "bin\clang.exe"
if (-not (Test-Path $clang)) { throw "clang++.exe not found at $clang." }
$withCuda = $CudaPath -and (Test-Path $CudaPath)

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
if ($withCuda) {
  $env:CUDA_PATH = (Resolve-Path $CudaPath).Path
  $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
}
$withHip = $HipPath -and (Test-Path $HipPath)
if ($PortableOpencl -and ($withCuda -or $withHip)) {
  throw 'PortableOpencl is a standards-only SPIR-V build and cannot include CUDA or HIP targets.'
}
if ($withHip -and -not $AmdArch) { $AmdArch = "gfx1200" }
if ($withHip) {
  $env:HIP_PATH = (Resolve-Path $HipPath).Path
  $env:PATH = "$env:HIP_PATH\bin;$env:PATH"
  Write-Host "AMD HIP target enabled: $AmdArch ($env:HIP_PATH)"
}
$buildJobs = Resolve-BuildJobs $Jobs
Write-Host "MOM_BUILD_JOBS = $buildJobs"

$obj = Join-Path $RepoRoot "obj"
if (Test-Path $obj) { Remove-Item -Recurse -Force $obj }
New-Item -ItemType Directory -Force $obj | Out-Null
New-Item -ItemType Directory -Force $OutDir | Out-Null

$inc = "-I" + (Join-Path $RepoRoot "xmrig")
# SYCL device TU flags (match buildsycl.bat / the Linux dpcpp-combined build) and the host-helper flags.
$F = @("-std=c++20","-O3","-ffp-contract=off","-DNDEBUG","-D_CRT_SECURE_NO_WARNINGS","-DMOM_SYCL_BUILD",
       "-DNOMINMAX","-DWIN32_LEAN_AND_MEAN","-fno-strict-aliasing",$inc)
$portableSpirvArgs = @()
if ($PortableOpencl) {
  # Device IR is translated and embedded during this standards-only link.
  $F += @("-DMOM_SYCL_PORTABLE_OPENCL", "-fno-sycl-rdc", "-fsycl-device-code-split=per_kernel",
          "-fno-sycl-instrument-device-code")
  $portableSpirvArgs = @(
    "-Xspirv-translator=spir64",
    "--spirv-ext=-SPV_INTEL_memory_access_aliasing,-SPV_KHR_expect_assume,-SPV_KHR_linkonce_odr"
  )
} else {
  $F += "-DMOM_PEARLHASH_HAS_ESIMD"
}
$H = @("-std=c++20","-O3","-DNDEBUG","-D_CRT_SECURE_NO_WARNINGS","-DMOM_SYCL_BUILD","-DNOMINMAX","-DWIN32_LEAN_AND_MEAN",$inc)
$targetList = @("spir64")
if ($withCuda) {
  $targetList += $CudaArch
  $F += "-DMOM_SYCL_HAS_CUDA"
}
if ($withHip) {
  $clangResource = Get-ChildItem (Join-Path $ToolchainDir 'lib\clang') -Directory |
    Sort-Object Name -Descending | Select-Object -First 1
  $amdLibspirv = if ($clangResource) {
    Join-Path $clangResource.FullName 'lib\amdgcn-amd-amdhsa-llvm\libspirv.l32.signed_char.bc'
  } else { $null }
  if (-not $amdLibspirv -or -not (Test-Path $amdLibspirv)) {
    throw "AMDGPU libspirv was not found under $ToolchainDir\lib\clang"
  }
  $ocml = Get-ChildItem $env:HIP_PATH -Recurse -Filter 'ocml.bc' -File -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if (-not $ocml) { throw "ROCm device libraries were not found under $env:HIP_PATH" }
  $rocmDeviceLib = $ocml.Directory.FullName
  $targetList += "amdgcn-amd-amdhsa"
  $F += @("-DMOM_SYCL_HAS_HIP", "-D__HIP_PLATFORM_AMD__", "--rocm-path=$env:HIP_PATH",
          "-fsycl-libspirv-path=$amdLibspirv", "-I$env:HIP_PATH\include")
}
$targets = $targetList -join ','
$amdBackendArgs = if ($withHip) {
  @("-Xsycl-target-backend=amdgcn-amd-amdhsa", "--offload-arch=$AmdArch")
} else { @() }

function New-ClangTask {
  param([string]$Exe, [string[]]$ClangArgs, [string]$What)
  [PSCustomObject]@{ Exe = $Exe; Args = $ClangArgs; What = $What }
}

function Complete-ClangJob {
  param([System.Management.Automation.Job]$Job)
  $receiveErrors = @()
  $output = Receive-Job -Job $Job -ErrorAction SilentlyContinue -ErrorVariable receiveErrors
  if ($output) { $output | ForEach-Object { Write-Host $_ } }
  if ($receiveErrors) { $receiveErrors | ForEach-Object { Write-Host $_.ToString() } }
  $ok = $Job.State -eq 'Completed'
  $name = $Job.Name
  Remove-Job -Job $Job -Force
  if (-not $ok) { throw "clang failed compiling $name." }
}

function Invoke-ClangTasks {
  param([object[]]$Tasks, [int]$Throttle)
  if ($Throttle -le 1) {
    foreach ($task in $Tasks) {
      Write-Host "  [$($task.What)]"
      & $task.Exe @($task.Args)
      if ($LASTEXITCODE -ne 0) { throw "clang failed compiling $($task.What) ($LASTEXITCODE)." }
    }
    return
  }

  $running = @()
  try {
    foreach ($task in $Tasks) {
      while ($running.Count -ge $Throttle) {
        $done = Wait-Job -Job $running -Any
        $doneId = $done.Id
        Complete-ClangJob $done
        $running = @($running | Where-Object { $_.Id -ne $doneId })
      }

      Write-Host "  [start] $($task.What)"
      $taskJson = @{
        Exe = $task.Exe
        Args = @($task.Args)
        What = $task.What
        WorkDir = $RepoRoot
      } | ConvertTo-Json -Compress -Depth 4
      $job = Start-Job -Name $task.What -ScriptBlock {
        param([string]$TaskJson)
        $task = $TaskJson | ConvertFrom-Json
        $clangArgs = @($task.Args | ForEach-Object { [string]$_ })
        Set-Location ([string]$task.WorkDir)
        & ([string]$task.Exe) @clangArgs 2>&1 | ForEach-Object { $_ }
        if ($LASTEXITCODE -ne 0) { throw "clang failed compiling $($task.What) ($LASTEXITCODE)." }
      } -ArgumentList $taskJson
      $running += $job
    }

    while ($running.Count -gt 0) {
      $done = Wait-Job -Job $running -Any
      $doneId = $done.Id
      Complete-ClangJob $done
      $running = @($running | Where-Object { $_.Id -ne $doneId })
    }
  } catch {
    foreach ($job in $running) {
      Stop-Job -Job $job -ErrorAction SilentlyContinue
      Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    }
    throw
  }
}

# Main SYCL TUs -> spir64 + nvptx. Keep stable object names for the linker while source paths follow
# the algorithm directories used by binding.gyp.
$main = [ordered]@{
  lib         = "sycl\lib.cpp"
  ethash      = "sycl\etchash\ethash.cpp"
  etchash     = "sycl\etchash\etchash.cpp"
  autolykos2  = "sycl\autolykos2\autolykos2.cpp"
  pearlhash   = "sycl\pearlhash\pearlhash.cpp"
  c29         = "sycl\c29\c29.cpp"
  cn_gpu      = "sycl\cn_gpu\cn_gpu.cpp"
  kawpow      = "sycl\kawpow\kawpow.cpp"
  fishhash    = "sycl\fishhash\fishhash.cpp"
  zelhash     = "sycl\zelhash\zelhash.cpp"
  beamhash3   = "sycl\beamhash3\beamhash3.cpp"
  blake2b     = "sycl\c29\blake2b.cpp"
}
$objs = @()
$compileTasks = @()
foreach ($entry in $main.GetEnumerator()) {
  $s = $entry.Key
  $o = Join-Path $obj "$s.obj"
  $compileTasks += New-ClangTask $clang (@("-fsycl","-fsycl-targets=$targets") + $amdBackendArgs +
    $F + @("-c",$entry.Value,"-o",$o)) "$targets $s"
  $objs += $o
}
# PearlHash ESIMD TU -> spir64 only (ESIMD can't share -fsycl-targets with nvptx; dispatched at runtime).
if (-not $PortableOpencl) {
  $pe = Join-Path $obj "pearlhash_esimd.obj"
  $compileTasks += New-ClangTask $clang (@("-fsycl","-fsycl-targets=spir64") + $F +
    @("-c","sycl\pearlhash\esimd.cpp","-o",$pe)) "spir64 pearlhash_esimd"
  $objs += $pe
}
# Host helpers the sycl target also needs (no -fsycl).
$sha3 = Join-Path $obj "sha3.obj"; $compileTasks += New-ClangTask $clang ($H + @("-c","xmrig\base\crypto\sha3.cpp","-o",$sha3)) "sha3"
$keccak = Join-Path $obj "keccak.obj"; $compileTasks += New-ClangTask $clang ($H + @("-c","xmrig\base\crypto\keccak.cpp","-o",$keccak)) "keccak"
$objs += $sha3, $keccak
$b2b = Join-Path $obj "blake2brx.obj"
$compileTasks += New-ClangTask $clangc @("-O3","-DNDEBUG",$inc,"-c","xmrig\crypto\randomx\blake2\blake2b.c","-o",$b2b) "blake2b.c"
$objs += $b2b
Invoke-ClangTasks $compileTasks $buildJobs

# Link the unified sycl.dll.
$out = Join-Path $OutDir "sycl.dll"
Write-Host "  [link] $out"
$linkArgs = @("-fsycl", "-fsycl-targets=$targets") + $amdBackendArgs + $portableSpirvArgs
if ($PortableOpencl) {
  $linkArgs += @("-fno-sycl-rdc", "-fsycl-device-code-split=per_kernel",
                 "-fno-sycl-instrument-device-code")
}
if ($withHip) {
  # intel/llvm #21385 workaround: run GlobalOffsetPass before AMDGPUAttributor so the runtime ABI
  # does not retain hidden global-offset arguments that HIP cannot populate.
  $linkArgs += @("-Xoffload-linker=amdgcn-amd-amdhsa", "--lto-newpm-passes=globaloffset,lto<O3>",
                 "--rocm-path=$env:HIP_PATH", "--rocm-device-lib-path=$rocmDeviceLib",
                 "-fsycl-libspirv-path=$amdLibspirv",
                 "-L$env:HIP_PATH\lib", "-lamdhip64")
}
& $clang @linkArgs "-shared" @objs "-o" $out
if ($LASTEXITCODE -ne 0) { throw "clang failed linking sycl.dll ($LASTEXITCODE)." }
if (-not (Test-Path $out)) { throw "sycl.dll was not produced at $out." }
$profile = if ($PortableOpencl) { 'portable OpenCL/Level Zero' } else { $targets }
Write-Host ("Built unified sycl.dll ({0:N1} MB, $profile)" -f ((Get-Item $out).Length/1MB))
