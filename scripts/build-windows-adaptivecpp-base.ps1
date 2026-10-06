param(
  [Parameter(Mandatory = $true)][string]$InstallDir,
  [string]$Workspace = 'C:\mom-dev-bootstrap',
  [int]$Jobs = 2
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($PSVersionTable.PSVersion.Major -ge 7) { $PSNativeCommandUseErrorActionPreference = $true }
if ($Jobs -lt 1) { throw '-Jobs must be a positive integer' }
. (Join-Path $PSScriptRoot 'windows-install-helpers.ps1')
. (Join-Path $PSScriptRoot 'import-vcvars.ps1')
if (-not [IO.Path]::IsPathRooted($InstallDir)) {
  $InstallDir = Join-Path (Get-Location).Path $InstallDir
}
$InstallDir = [IO.Path]::GetFullPath($InstallDir)
$Workspace = [IO.Path]::GetFullPath($Workspace)
$trimChars = [char[]]@('\', '/')
$installLeaf = [IO.Path]::GetFileName($InstallDir.TrimEnd($trimChars))
$workspaceLeaf = [IO.Path]::GetFileName($Workspace.TrimEnd($trimChars))
if ($installLeaf -notmatch '^acpp-[A-Za-z0-9_.-]+$') {
  throw '-InstallDir must name a dedicated acpp-* directory'
}
if ($workspaceLeaf -notmatch '^mom-dev-[A-Za-z0-9_.-]+$') {
  throw '-Workspace must name a dedicated mom-dev-* directory'
}
foreach ($path in @($InstallDir, $Workspace)) {
  $trimmed = $path.TrimEnd($trimChars)
  $root = [IO.Path]::GetPathRoot($path).TrimEnd($trimChars)
  if ([string]::Equals($trimmed, $root, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'AdaptiveCpp paths must not be filesystem roots'
  }
  $item = Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
  if ($item -and -not $item.PSIsContainer) {
    throw "AdaptiveCpp path must be a real directory: $path"
  }
  Assert-NoReparseAncestor $path 'AdaptiveCpp path'
}
$installItem = Get-Item -LiteralPath $InstallDir -Force -ErrorAction SilentlyContinue
if ($installItem) {
  $invalidMarker = -not (Test-MomMarker (Join-Path $InstallDir '.mom-acpp-toolchain') `
    'mom AdaptiveCpp toolchain')
  if ($invalidMarker -and
      (Get-ChildItem -LiteralPath $InstallDir -Force | Select-Object -First 1)) {
    throw '-InstallDir exists without the .mom-acpp-toolchain ownership marker'
  }
}
if (-not (Test-MomMarker (Join-Path $Workspace '.mom-dev-workspace') `
    'mom development workspace')) {
  throw '-Workspace must be initialized by scripts\install-dev.bat'
}
$separator = [IO.Path]::DirectorySeparatorChar
$installPrefix = $InstallDir.TrimEnd($trimChars) + $separator
$workspacePrefix = $Workspace.TrimEnd($trimChars) + $separator
if ($InstallDir.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase) -or
    $Workspace.StartsWith($installPrefix, [StringComparison]::OrdinalIgnoreCase)) {
  throw '-InstallDir and -Workspace must be separate directories'
}

$adaptiveCppCommit = 'da2463e45aa90aa36306c45abcfc05b87de51bc6'
$llvmTag = 'llvmorg-20.1.8'
$llvmCommit = '87f0227cb60147a26a1eeb4fb06e3b505e9c7261'
$bootstrapUrl = 'https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.7/LLVM-19.1.7-win64.exe'
$bootstrapSha256 = 'f19ae5bc4823ac69ec01dc2ded503ec80a04ad2208dda1595d1f0413c148ef90'
$bootstrapDir = Join-Path $Workspace 'llvm-bootstrap'
$bootstrapMarker = Join-Path $bootstrapDir '.mom-llvm-bootstrap'
$adaptiveCppSource = Join-Path $Workspace 'AdaptiveCpp-base-src'
$llvmSource = Join-Path $Workspace 'llvm-project'
$buildDir = Join-Path $Workspace 'AdaptiveCpp-base-build'

Import-MomVcVars64
New-Item -ItemType Directory -Force $Workspace | Out-Null

# Bootstrap LLVM only compiles the pinned LLVM 20 tree. It is not shipped in the final toolchain.
$bootstrapInstaller = Join-Path $Workspace 'LLVM-19.1.7-win64.exe'
$bootstrapReady = (Test-MomMarker $bootstrapMarker $bootstrapSha256) -and
  (Test-Path (Join-Path $bootstrapDir 'bin\clang-cl.exe') -PathType Leaf) -and
  (Test-Path (Join-Path $bootstrapDir 'bin\lld-link.exe') -PathType Leaf)
if (-not $bootstrapReady) {
  if (-not (Test-Path $bootstrapInstaller)) { Download $bootstrapUrl $bootstrapInstaller }
  Assert-Sha256 $bootstrapInstaller $bootstrapSha256
  Remove-Item $bootstrapDir -Recurse -Force -ErrorAction SilentlyContinue
  $process = Start-Process $bootstrapInstaller -ArgumentList @('/S',"/D=$bootstrapDir") `
    -Wait -PassThru
  if ($process.ExitCode -ne 0) { throw "LLVM bootstrap install failed: $($process.ExitCode)" }
  foreach ($required in @('bin\clang-cl.exe', 'bin\lld-link.exe')) {
    if (-not (Test-Path (Join-Path $bootstrapDir $required) -PathType Leaf)) {
      throw "LLVM bootstrap install is missing $required"
    }
  }
  Set-Content -LiteralPath $bootstrapMarker -Value $bootstrapSha256 -NoNewline
}
$env:Path = "$bootstrapDir\bin;$env:Path"
$env:CC = 'clang-cl'
$env:CXX = 'clang-cl'

Remove-Item $InstallDir, $adaptiveCppSource, $llvmSource, $buildDir `
  -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $InstallDir | Out-Null
Set-Content -LiteralPath (Join-Path $InstallDir '.mom-acpp-toolchain') `
  -Value 'mom AdaptiveCpp toolchain' -NoNewline

Invoke-Checked 'git.exe' @(
  'clone','--quiet','--filter=blob:none','https://github.com/AdaptiveCpp/AdaptiveCpp.git',
  $adaptiveCppSource
)
Invoke-Checked 'git.exe' @('-C',$adaptiveCppSource,'checkout','--quiet',$adaptiveCppCommit)
Invoke-Checked 'git.exe' @(
  '-C',$adaptiveCppSource,'submodule','update','--init','--recursive','--depth','1'
)
Invoke-Checked 'git.exe' @(
  'clone','--quiet','--filter=blob:none','--depth','1','--branch',$llvmTag,
  'https://github.com/llvm/llvm-project.git',$llvmSource
)
$actualLlvm = (& git.exe -C $llvmSource rev-parse HEAD).Trim()
if ($actualLlvm -ne $llvmCommit) {
  throw "LLVM $llvmTag resolved to $actualLlvm instead of $llvmCommit"
}

$cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else {
  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
}
$subprojectJobs = [Math]::Min(4, $Jobs)
$cmakeArgs = @(
  '-S', (Join-Path $llvmSource 'llvm'),
  '-B', $buildDir,
  '-G', 'Ninja',
  '-DCMAKE_BUILD_TYPE=Release',
  "-DCMAKE_INSTALL_PREFIX=$InstallDir",
  # One shared source build serves both native Windows GPU addons. Keep AMDGPU and NVPTX in the
  # installed LLVM so the later HIP addon can lower SSCP IR without rebuilding LLVM from scratch.
  '-DLLVM_TARGETS_TO_BUILD=X86;AMDGPU;NVPTX',
  '-DLLVM_ENABLE_PROJECTS=clang;openmp;lld;compiler-rt',
  # DIA is unnecessary for release compilation and requires Visual Studio's optional ATL headers.
  '-DLLVM_ENABLE_DIA_SDK=OFF',
  '-DLLVM_PARALLEL_LINK_JOBS=1',
  '-DLLVM_EXTERNAL_PROJECTS=AdaptiveCpp',
  "-DLLVM_EXTERNAL_ADAPTIVECPP_SOURCE_DIR=$adaptiveCppSource",
  '-DLLVM_ADAPTIVECPP_LINK_INTO_TOOLS=ON',
  '-DWITH_CUDA_BACKEND=ON',
  '-DWITH_ROCM_BACKEND=OFF',
  '-DWITH_OPENCL_BACKEND=OFF',
  '-DWITH_VULKAN_BACKEND=OFF',
  # Windows Clang has complete std::filesystem support; do not build/cache a redundant Boost tree.
  '-DACPP_FILESYSTEM_SEARCH_OPTIONS=Final',
  # Do not bake the ephemeral CI host's exact CPU ISA into the shipped OpenMP backend.
  '-DACPP_HOST_FORCE_MCPU_TARGET=x86-64',
  '-DLLVM_TOOL_BUGPOINT_BUILD=OFF',
  '-DOPENMP_ENABLE_LIBOMPTARGET=OFF',
  '-DLLVM_INCLUDE_TESTS=OFF',
  "-DCUDA_TOOLKIT_ROOT_DIR=$cuda",
  "-DACPP_SUBPROJECT_PARALLEL_JOBS=$subprojectJobs"
)
Invoke-Checked 'cmake.exe' $cmakeArgs
Invoke-Checked 'cmake.exe' @('--build',$buildDir,'--target','install','--parallel',"$Jobs")

foreach ($required in @(
  'bin\acpp','bin\clang-cl.exe','bin\llc.exe','bin\libomp.dll',
  'bin\hipSYCL\rt-backend-cuda.dll','bin\hipSYCL\rt-backend-omp.dll'
)) {
  if (-not (Test-Path (Join-Path $InstallDir $required))) {
    throw "AdaptiveCpp source build is missing $required"
  }
}
$registeredTargets = (& (Join-Path $InstallDir 'bin\llc.exe') --version) -join "`n"
foreach ($target in @('amdgcn', 'nvptx', 'x86')) {
  if ($registeredTargets -notmatch "(?m)^\s*$target\s+-") {
    throw "AdaptiveCpp source build LLVM is missing the $target target"
  }
}
Remove-Item $adaptiveCppSource, $llvmSource, $buildDir, $bootstrapDir, $bootstrapInstaller `
  -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "AdaptiveCpp Windows CUDA/base toolchain ready at $InstallDir"
