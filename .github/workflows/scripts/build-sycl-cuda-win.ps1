# Build the unified sycl.dll (Intel spir64 + NVIDIA nvptx) with the from-source
# intel/llvm clang restored by restore-toolchain-win.ps1, then drop it into build\Release so packaging
# ships it instead of the Intel-only (MSBuild/icx) sycl.dll. This is the Windows counterpart of the Linux
# combined build's clang `-fsycl` device step; the kernel sources are byte-identical (the only Windows
# source delta is kawpow_jit.inc's module-dir lookup). Mirrors the validated Linux dpcpp-combined build.
param(
  [string]$RepoRoot     = '',
  [string]$ToolchainDir = $env:MOM_DPCPP_DIR,
  [string]$CudaPath     = $env:CUDA_PATH,
  [string]$CudaArch     = "nvidia_gpu_sm_80",   # single low arch; driver JITs PTX forward to the real GPU
  [string]$OutDir       = "build\Release",
  [int]$Jobs            = 0,
  [switch]$PortableOpencl,
  [switch]$RequireCuda
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (-not $RepoRoot) {
  # Windows PowerShell 5.1 does not populate $PSScriptRoot while evaluating parameter defaults.
  $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
} else {
  $RepoRoot = (Resolve-Path $RepoRoot).Path
}
Set-Location $RepoRoot
. (Join-Path $RepoRoot 'scripts\import-vcvars.ps1')
. (Join-Path $RepoRoot 'scripts\windows-sycl-sources.ps1')
. (Join-Path $RepoRoot 'scripts\windows-install-helpers.ps1')

function Resolve-BuildJobs([int]$Requested) {
  if ($Requested -lt 0) {
    throw "Jobs must be zero or a positive integer, got '$Requested'"
  }
  $jobs = $Requested
  if ($jobs -eq 0 -and $null -ne $env:MOM_BUILD_JOBS -and $env:MOM_BUILD_JOBS -ne '') {
    $parsed = 0
    if (-not [int]::TryParse($env:MOM_BUILD_JOBS, [ref]$parsed) -or $parsed -lt 1) {
      throw "MOM_BUILD_JOBS must be a positive integer, got '$env:MOM_BUILD_JOBS'"
    }
    $jobs = $parsed
  }
  if ($jobs -eq 0) {
    $jobs = [Environment]::ProcessorCount
  }
  return [Math]::Max(1, $jobs)
}

$buildJobs = Resolve-BuildJobs $Jobs

if (-not $ToolchainDir) { throw "ToolchainDir not set (run restore-toolchain-win.ps1 first, or pass -ToolchainDir)." }
$clang  = Join-Path $ToolchainDir "bin\clang++.exe"
$clangc = Join-Path $ToolchainDir "bin\clang.exe"
$lld    = Join-Path $ToolchainDir "bin\lld-link.exe"
foreach ($compilerTool in @($clang, $clangc, $lld)) {
  if (-not (Test-Path -LiteralPath $compilerTool -PathType Leaf)) {
    throw "Required compiler tool not found at $compilerTool."
  }
}
$withCuda = $false
if ($CudaPath) {
  if (-not (Test-Path -LiteralPath $CudaPath -PathType Container)) {
    throw "CudaPath does not resolve to an existing directory: $CudaPath"
  }
  $CudaPath = (Resolve-Path -LiteralPath $CudaPath).Path
  if (-not (Test-MomCudaCompilerSdk $CudaPath)) {
    throw "CUDA compiler SDK is incomplete at $CudaPath."
  }
  $withCuda = $true
}
if ($RequireCuda -and -not $withCuda) {
  throw 'CUDA is required, but no usable CudaPath was provided.'
}

# The clang driver links sycl.dll with lld-link + the MSVC CRT/Windows SDK.
Import-MomVcVars64
if ($withCuda) {
  $env:CUDA_PATH = (Resolve-Path $CudaPath).Path
  $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
}
if ($PortableOpencl -and $withCuda) {
  throw 'PortableOpencl is a standards-only SPIR-V build and cannot include a CUDA target.'
}
Write-Host "MOM_BUILD_JOBS = $buildJobs"

$obj = Join-Path $RepoRoot "obj"
if (Test-Path $obj) { Remove-Item -Recurse -Force $obj }
New-Item -ItemType Directory -Force $obj | Out-Null
New-Item -ItemType Directory -Force $OutDir | Out-Null

$inc = "-I" + (Join-Path $RepoRoot "xmrig")
# SYCL device TU flags (match the Linux dpcpp-combined build) and the host-helper flags.
$F = @("-std=c++20","-O3","-ffp-contract=off","-DNDEBUG","-D_CRT_SECURE_NO_WARNINGS","-DMOM_SYCL_BUILD",
       "-DNOMINMAX","-DWIN32_LEAN_AND_MEAN","-fno-strict-aliasing",$inc)
$portableSpirvArgs = @()
if ($PortableOpencl) {
  # Device IR is translated and embedded during this standards-only link.
  $F += @("-DMOM_SYCL_PORTABLE_OPENCL", "-DMOM_NEXAPOW_PORTABLE_FIELD32", "-fno-sycl-rdc", "-fsycl-device-code-split=per_kernel",
          "-fno-sycl-instrument-device-code")
  # SPIR-V 1.3 has core subgroup operations. The pinned CI translator otherwise falls back to
  # SPV_INTEL_subgroups when that extension is enabled, or rejects standard SYCL collectives when
  # it is disabled. Forward each translator option separately: -X consumes exactly one argument.
  $portableSpirvArgs = @(
    "-Xspirv-translator=spir64",
    "--spirv-max-version=1.3",
    "-Xspirv-translator=spir64",
    "--spirv-ext=-SPV_INTEL_memory_access_aliasing,-SPV_INTEL_subgroups,-SPV_KHR_expect_assume,-SPV_KHR_linkonce_odr"
  )
} else {
  $F += "-DMOM_PEARLHASH_HAS_ESIMD"
}
$H = @("-std=c++20","-O3","-DNDEBUG","-D_CRT_SECURE_NO_WARNINGS","-DMOM_SYCL_BUILD","-DNOMINMAX","-DWIN32_LEAN_AND_MEAN",$inc)
$targetList = @("spir64")
if ($withCuda) {
  $targetList += $CudaArch
  $F += @("-DMOM_SYCL_HAS_CUDA",
          "-DMOM_OCTOPUS_HAS_SYCL_NATIVE",
          "-DMOM_NEXAPOW_SYCL_NATIVE_FIELD", "-I$env:CUDA_PATH\include")
}
$targets = $targetList -join ','

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
$main = Get-MomWindowsSyclSources
$objs = @()
$compileTasks = @()
foreach ($entry in $main.GetEnumerator()) {
  $s = $entry.Key
  $o = Join-Path $obj "$s.obj"
  $compileTasks += New-ClangTask $clang (@("-fsycl","-fsycl-targets=$targets") +
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
$hoo = Join-Path $obj "hoohash_host.obj"
$compileTasks += New-ClangTask $clangc @("-O2","-fno-fast-math","-ffp-contract=off","-fno-builtin",
  "-DNDEBUG","-D_CRT_SECURE_NO_WARNINGS","-c","sycl\hoohash\host_math.c","-o",$hoo) "hoohash host math"
$objs += $hoo
Invoke-ClangTasks $compileTasks $buildJobs

# Link the unified sycl.dll.
$out = Join-Path $OutDir "sycl.dll"
Write-Host "  [link] $out"
$linkArgs = @("-fsycl", "-fsycl-targets=$targets") + $portableSpirvArgs
if ($PortableOpencl) {
  $linkArgs += @("-fno-sycl-rdc", "-fsycl-device-code-split=per_kernel",
                 "-fno-sycl-instrument-device-code")
}
& $clang @linkArgs "-shared" @objs "-o" $out
if ($LASTEXITCODE -ne 0) { throw "clang failed linking sycl.dll ($LASTEXITCODE)." }
if (-not (Test-Path $out)) { throw "sycl.dll was not produced at $out." }
$profile = if ($PortableOpencl) { 'portable OpenCL/Level Zero' } else { $targets }
Write-Host ("Built unified sycl.dll ({0:N1} MB, $profile)" -f ((Get-Item $out).Length/1MB))
