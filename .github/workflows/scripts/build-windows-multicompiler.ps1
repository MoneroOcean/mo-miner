param(
  [ValidateSet('all','intel','nvidia','amd','opencl')][string]$Backend = 'all',
  [ValidateSet('all','oneapi','dpcpp','portable','adaptivecpp')][string]$Compiler = 'all',
  [string]$DpcppDir = $(if ($env:MOM_DPCPP_DIR) { $env:MOM_DPCPP_DIR } else { 'C:\Tools\dpcpp' }),
  [string]$AcppCudaDir = 'C:\Tools\acpp-cuda',
  [string]$AcppHipDir = 'C:\Tools\acpp-amd',
  [string]$HipPath = 'C:\Program Files\AMD\ROCm\7.1',
  [string]$CudaPath = $env:CUDA_PATH
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
Set-Location $repo
. "$PSScriptRoot\windows-dll-deps.ps1"
$supportedCompilers = switch ($Backend) {
  'all'    { @('all') }
  'intel'  { @('all','oneapi','dpcpp','portable') }
  'nvidia' { @('all','dpcpp','portable','adaptivecpp') }
  'amd'    { @('all','portable','adaptivecpp') }
  'opencl' { @('all','portable') }
}
if ($Compiler -notin $supportedCompilers) {
  throw "Compiler '$Compiler' is not available for backend '$Backend'."
}
$platformRoot = Join-Path $repo 'build'
$platformsHold = Join-Path $repo 'build-platforms-hold'
$completedBuild = Join-Path $repo 'build-windows-completed'
if ((Test-Path $platformsHold) -or (Test-Path $completedBuild)) {
  throw 'A previous Windows build staging directory still exists; preserve or remove it before rebuilding.'
}

function Resolve-RepoPath([string]$Path) {
  if (-not $Path) { return '' }
  if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
  [IO.Path]::GetFullPath((Join-Path $repo $Path))
}

$DpcppDir = Resolve-RepoPath $DpcppDir
$AcppCudaDir = Resolve-RepoPath $AcppCudaDir
$AcppHipDir = Resolve-RepoPath $AcppHipDir
$HipPath = Resolve-RepoPath $HipPath
$CudaPath = Resolve-RepoPath $CudaPath

if (Test-Path $platformRoot) {
  # node-gyp owns the top-level build directory while this script runs, so preserve the existing
  # Linux/Windows outputs in a sibling. Reusable local toolchains also live below build/. Remap
  # those caller-supplied paths into the held tree before moving it; otherwise a valid
  # build/toolchains/... input disappears halfway through release assembly.
  $separator = [IO.Path]::DirectorySeparatorChar
  $buildPrefix = $platformRoot.TrimEnd($separator) + $separator
  foreach ($name in @('DpcppDir','AcppCudaDir','AcppHipDir','HipPath','CudaPath')) {
    $value = Get-Variable -Name $name -ValueOnly
    if ($value -and $value.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase)) {
      Set-Variable -Name $name -Value (Join-Path $platformsHold $value.Substring($buildPrefix.Length))
    }
  }
  Move-Item $platformRoot $platformsHold
}
New-Item -ItemType Directory -Force $platformRoot | Out-Null
# AdaptiveCpp's application database is runtime state, not node-gyp output. Carry it through a
# source rebuild so build/win can be round-tripped by run.sh without repeating every SSCP JIT.
$cachedAppDb = Join-Path $platformsHold 'win\.acpp'
if (Test-Path $cachedAppDb) { Copy-Item $cachedAppDb (Join-Path $platformRoot '.acpp') -Recurse }
$out = Join-Path $repo 'build\compilers'
# A targeted developer rebuild updates one compiler or vendor set in place. Release/CI builds use
# Backend=all, Compiler=all and start from an empty compiler tree, so stale workers cannot enter
# artifacts. run.sh normally requests one passed-through vendor and must preserve the other cached
# workers instead of turning build/win into a single-vendor tree.
if ($Compiler -ne 'all' -or $Backend -ne 'all') {
  $savedCompilers = Join-Path $platformsHold 'win\compilers'
  if (Test-Path $savedCompilers) {
    New-Item -ItemType Directory -Force $out | Out-Null
    Copy-Item (Join-Path $savedCompilers '*') $out -Recurse -Force
  }
}
# node-gyp requires this temporary top-level build directory. The finally block publishes the
# complete PE tree as build/win and restores build/lin untouched, even when compilation fails.

function Save-Compiler(
  [string]$Name,
  [string]$ToolchainDir = '',
  [string]$RuntimeToolsDir = '',
  [string]$DeviceLibDir = ''
) {
  $dest = Join-Path $out $Name
  New-Item -ItemType Directory -Force $dest | Out-Null
  Copy-Item build\Release\mom.node, build\Release\sycl.dll $dest
  if ($ToolchainDir) {
    # Snapshot the production accelerator and required host backend, not the entire installation.
    # The AdaptiveCpp bin tree also contains libclang/LLVM-C/LTO and every enabled backend; none is
    # used by SSCP's external opt/llc/lld pipeline. Copying all of it added 80--100 MiB compressed per worker and
    # made a nominally CUDA-only directory carry OpenMP/HIP code (and vice versa).
    $runtimeBackend = if ($Name -eq 'acpp-cuda') { 'cuda' } elseif ($Name -eq 'acpp-hip') { 'hip' }
      else { throw "Unknown AdaptiveCpp worker $Name" }
    $bin = (Resolve-Path (Join-Path $ToolchainDir 'bin')).Path
    $runtimeFiles = @(
      (Join-Path $bin 'acpp-common.dll'),
      (Join-Path $bin 'acpp-rt.dll'),
      (Join-Path $bin 'libomp.dll'),
      # AdaptiveCpp's backend manager requires one host backend even when every submitted kernel
      # targets CUDA or HIP. Omitting the OMP plugin lets the GPU plugin load, then terminates on the
      # first runtime construction with "No CPU backend has been loaded."
      (Join-Path $bin 'hipSYCL\rt-backend-omp.dll'),
      (Join-Path $bin "hipSYCL\rt-backend-$runtimeBackend.dll")
    )
    foreach ($file in $runtimeFiles) {
      if (-not (Test-Path $file)) { throw "AdaptiveCpp runtime file missing: $file" }
      $resolved = (Resolve-Path $file).Path
      $relative = $resolved.Substring($bin.Length).TrimStart('\')
      $target = Join-Path $dest $relative
      New-Item -ItemType Directory -Force (Split-Path -Parent $target) | Out-Null
      Copy-Item $resolved $target -Force
    }
    $bitcode = Join-Path $ToolchainDir 'lib\hipSYCL\bitcode'
    if (Test-Path $bitcode) {
      $deviceLibrary = if ($runtimeBackend -eq 'cuda') { 'libkernel-sscp-ptx-full.bc' }
        else { 'libkernel-sscp-amdgpu-amdhsa-full.bc' }
      $redistDeviceLib = Join-Path $dest 'hipSYCL\bitcode'
      New-Item -ItemType Directory -Force $redistDeviceLib | Out-Null
      # The OpenMP host backend is part of each AdaptiveCpp runtime and is used by deployment CPU
      # smokes. Generic/SSCP JIT needs its host builtins just as CUDA/HIP need their device builtins.
      foreach ($library in @('libkernel-sscp-host-full.bc', $deviceLibrary)) {
        $libraryPath = Join-Path $bitcode $library
        if (-not (Test-Path $libraryPath)) {
          throw "AdaptiveCpp device library missing: $libraryPath"
        }
        Copy-Item $libraryPath $redistDeviceLib -Force
      }
    }
    # Generic/SSCP JIT invokes opt/llc at runtime. AdaptiveCpp looks here before its compiled-in
    # development path, making the release relocatable. HIP deliberately uses ROCm's LLVM 21 tools.
    if (-not $RuntimeToolsDir) { $RuntimeToolsDir = Join-Path $ToolchainDir 'bin' }
    $redistBin = Join-Path $dest 'hipSYCL\ext\llvm\bin'
    New-Item -ItemType Directory -Force $redistBin | Out-Null
    # AdaptiveCpp's Windows configuration records lld-link, not the byte-identical lld alias.
    foreach ($tool in @('opt.exe','llc.exe','lld-link.exe')) {
      $source = Join-Path $RuntimeToolsDir $tool
      if (-not (Test-Path $source)) { throw "AdaptiveCpp runtime tool missing: $source" }
      Copy-Item $source $redistBin -Force
    }
    if ($DeviceLibDir -and (Test-Path (Join-Path $DeviceLibDir 'ockl.bc'))) {
      $redistBitcode = Join-Path $dest 'hipSYCL\ext\bitcode\amdgcn'
      New-Item -ItemType Directory -Force $redistBitcode | Out-Null
      Copy-Item (Join-Path $DeviceLibDir '*.bc') $redistBitcode -Force
    }
    if ($runtimeBackend -eq 'cuda') {
      # AdaptiveCpp checks this relocatable deployment-manifest location before its compiled-in
      # CUDA toolkit path. Keep the release driver-only: SSCP needs this small LLVM bitcode library
      # to JIT kernels, but it does not need a system-wide CUDA toolkit.
      $libdevice = Get-ChildItem -Path @(
        (Join-Path $ToolchainDir 'lib\hipSYCL\ext\bitcode\ptx\libdevice.10.bc'),
        $(if ($CudaPath) { Join-Path $CudaPath 'nvvm\libdevice\libdevice.10.bc' })
      ) -File -ErrorAction SilentlyContinue | Select-Object -First 1
      if (-not $libdevice) { throw 'AdaptiveCpp CUDA libdevice.10.bc was not found.' }
      $redistPtx = Join-Path $dest 'hipSYCL\ext\bitcode\ptx'
      New-Item -ItemType Directory -Force $redistPtx | Out-Null
      Copy-Item $libdevice.FullName (Join-Path $redistPtx 'libdevice.10.bc') -Force
    }
  }
}

function Save-DpcppRuntime(
  [string]$Name = 'dpcpp',
  [string]$RuntimeToolchainDir = $DpcppDir
) {
  $dest = Join-Path $out $Name
  $savedDpcpp = $env:MOM_DPCPP_DIR
  $savedAcpp = $env:MOM_ACPP_DIR
  $savedHip = $env:HIP_PATH
  $savedRocm = $env:ROCM_PATH
  $savedOneApi = $env:ONEAPI_ROOT
  try {
    $env:MOM_DPCPP_DIR = $RuntimeToolchainDir
    Remove-Item Env:MOM_ACPP_DIR, Env:ONEAPI_ROOT -ErrorAction SilentlyContinue
    Remove-Item Env:HIP_PATH, Env:ROCM_PATH -ErrorAction SilentlyContinue
    # The compiler policy puts only this worker directory on PATH. Stage the nightly SYCL runtime,
    # dynamically loaded UR adapters and their DLL closure here so source-tree tests can run before
    # package-windows.ps1 creates the final release layout.
    Copy-MominerOptionalRuntimeFiles -PackageDir $dest
    $entries = @(Get-ChildItem $dest -File | Where-Object { $_.Extension -in @('.dll', '.node') } |
      ForEach-Object { $_.FullName })
    Copy-MominerDllClosure -PackageDir $dest -EntryPaths $entries
  } finally {
    if ($savedDpcpp) { $env:MOM_DPCPP_DIR = $savedDpcpp } else { Remove-Item Env:MOM_DPCPP_DIR -ErrorAction SilentlyContinue }
    if ($savedAcpp) { $env:MOM_ACPP_DIR = $savedAcpp } else { Remove-Item Env:MOM_ACPP_DIR -ErrorAction SilentlyContinue }
    if ($savedHip) { $env:HIP_PATH = $savedHip } else { Remove-Item Env:HIP_PATH -ErrorAction SilentlyContinue }
    if ($savedRocm) { $env:ROCM_PATH = $savedRocm } else { Remove-Item Env:ROCM_PATH -ErrorAction SilentlyContinue }
    if ($savedOneApi) { $env:ONEAPI_ROOT = $savedOneApi } else { Remove-Item Env:ONEAPI_ROOT -ErrorAction SilentlyContinue }
  }
}

try {
  & "$PSScriptRoot\build-windows.ps1"
  if ($LASTEXITCODE -ne 0) { throw 'Windows host/oneAPI build failed' }
  if ($Backend -in @('all','intel')) {
    if ($Compiler -in @('all','oneapi')) { Save-Compiler oneapi }

    # Intel must be a real second compiler snapshot, not another invocation of the oneAPI addon.
    # Build open DPC++ for generic SPIR-V/Level Zero without CUDA or HIP target dependencies.
    if ($Compiler -in @('all','dpcpp')) {
      & "$PSScriptRoot\build-sycl-cuda-win.ps1" -ToolchainDir $DpcppDir -CudaPath '' -HipPath ''
      if ($LASTEXITCODE -ne 0) { throw 'open-source DPC++ Intel build failed' }
      $env:MOM_DPCPP_DIR = $DpcppDir
      Save-Compiler dpcpp
      Save-DpcppRuntime
    }

  }

  # The same standards-only SPIR-V image serves generic OpenCL on every vendor and Level Zero on
  # Intel integrated GPUs. Keep it in targeted vendor builds so any config can select a portable
  # backend without first rebuilding under MOM_GPU_BACKEND=opencl.
  if ($Backend -in @('all','intel','nvidia','amd','opencl') -and $Compiler -in @('all','portable')) {
    & "$PSScriptRoot\build-sycl-cuda-win.ps1" -ToolchainDir $DpcppDir -CudaPath '' -HipPath '' -PortableOpencl
    if ($LASTEXITCODE -ne 0) { throw 'portable OpenCL/Level Zero DPC++ build failed' }
    Save-Compiler dpcpp-opencl
    # A targeted portable rebuild may select a newer/pinned DPC++ over a preserved build/win tree.
    # Always refresh the shared runtime from that same toolchain; mixing a newly compiled addon with
    # an older sycl9/UR loader can pass compilation yet fail during device teardown.
    Save-DpcppRuntime
  }

  if ($Backend -in @('all','nvidia') -and $Compiler -in @('all','dpcpp')) {
    & "$PSScriptRoot\build-sycl-cuda-win.ps1" -ToolchainDir $DpcppDir -CudaPath $CudaPath -HipPath ''
    if ($LASTEXITCODE -ne 0) { throw 'open-source DPC++ build failed' }
  # DPC++ workers need the runtime DLLs, not a redistributable compiler. package-windows.ps1 copies
  # its release runtime/adapters from this canonical tree and resolves their transitive closure.
  # Treating DPC++ like AdaptiveCpp here used to add opt/llc/lld plus every debug adapter/runtime to
  # the release (hundreds of MiB with no runtime purpose).
    $env:MOM_DPCPP_DIR = $DpcppDir
    Save-Compiler dpcpp
    Save-DpcppRuntime

  }
  if ($Backend -in @('all','nvidia') -and $Compiler -in @('all','adaptivecpp')) {
    & "$PSScriptRoot\build-sycl-adaptivecpp-win.ps1" -ToolchainDir $AcppCudaDir -Backend cuda -CudaPath $CudaPath
    if ($LASTEXITCODE -ne 0) { throw 'AdaptiveCpp CUDA build failed' }
    Save-Compiler acpp-cuda $AcppCudaDir
  }

  if ($Backend -in @('all','amd') -and $Compiler -in @('all','adaptivecpp')) {
    $amdDeviceLib = Get-ChildItem $HipPath -Recurse -Filter 'ockl.bc' -File | Select-Object -First 1
    if (-not $amdDeviceLib) { throw "ROCm device libraries not found under $HipPath" }
    & "$PSScriptRoot\build-sycl-adaptivecpp-win.ps1" -ToolchainDir $AcppHipDir -Backend hip -HipPath $HipPath
    if ($LASTEXITCODE -ne 0) { throw 'AdaptiveCpp HIP build failed' }
    Save-Compiler acpp-hip $AcppHipDir (Join-Path $HipPath 'bin') $amdDeviceLib.Directory.FullName
  }

  Write-Host 'Built Windows compiler workers:'
  Get-ChildItem $out -Directory | ForEach-Object {
    $compilerDir = $_
    $bytes = (Get-ChildItem -LiteralPath $compilerDir.FullName -File -Recurse |
      Measure-Object -Property Length -Sum).Sum
    Write-Host ("  {0}: {1:N1} MB" -f $compilerDir.Name, ($bytes / 1MB))
  }
} finally {
  if (Test-Path $platformRoot) { Move-Item $platformRoot $completedBuild }
  if (Test-Path $platformsHold) {
    Move-Item $platformsHold $platformRoot
  } else {
    New-Item -ItemType Directory -Force $platformRoot | Out-Null
  }
  if (Test-Path $completedBuild) {
    $windowsBuild = Join-Path $platformRoot 'win'
    Remove-Item $windowsBuild -Recurse -Force -ErrorAction SilentlyContinue
    Move-Item $completedBuild $windowsBuild
  }
}
