param(
  [string]$RocmStage = $PSScriptRoot,
  [string]$RocmPath = 'C:\Program Files\AMD\ROCm\7.1',
  [string]$SourceDir = 'C:\llvm-amd',
  [string]$BuildDir = 'C:\llvm-amd\build',
  [string]$InstallDir = '',
  [string]$Tag = 'nightly-2026-07-11',
  [string]$Commit = 'eca4d070277a1e62b196a5fddefe72bc7f98ee24',
  [int]$Jobs = 0
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($Jobs -lt 1) { $Jobs = [Environment]::ProcessorCount }

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

function Replace-RequiredText([string]$Path, [string]$Old, [string]$New, [string]$Label) {
  $text = (Get-Content -Raw $Path).Replace("`r`n", "`n")
  if ($text.Contains($New)) { return }
  if (-not $text.Contains($Old)) { throw "DPC++ workaround no longer matches upstream: $Label" }
  Set-Content -Path $Path -Value ($text.Replace($Old, $New)) -Encoding UTF8
}

# These are the only HIP SDK components needed by clang, the UR HIP adapter and the miner's HIPRTC
# ProgPoW path. The display driver remains untouched.
if (-not (Test-Path "$RocmPath\lib\amdhip64.lib")) {
  foreach ($name in @('ROCm_SDK_Core.msi', 'ROCm_RTC_RT.msi', 'ROCm_RTC_Dev.msi')) {
    $msi = Join-Path $RocmStage $name
    if (-not (Test-Path $msi)) { throw "Missing HIP SDK component: $msi" }
    # msiexec may start with System32 as its working directory; always give it an absolute path.
    $msi = (Resolve-Path $msi).Path
    $p = Start-Process msiexec.exe -ArgumentList @('/i', "`"$msi`"", '/qn', '/norestart') -Wait -PassThru
    if ($p.ExitCode -notin @(0, 3010)) { throw "$name install failed with exit code $($p.ExitCode)" }
  }
}

$env:ROCM_PATH = $RocmPath
$env:HIP_PATH = $RocmPath
if (-not (Test-Path "$RocmPath\lib\amdhip64.lib")) { throw "HIP SDK is incomplete at $RocmPath" }

# Do not put ROCm's device-oriented clang first: on Windows it is invoked with GNU driver semantics
# and cannot link a host CRT executable. The existing DPC++ clang-cl is also configured -nostdlib;
# bootstrap the new LLVM with the VS 2022 host compiler imported by the runner's vcvars environment.
$hostCompiler = Get-ChildItem 'C:\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe' -File -ErrorAction SilentlyContinue |
  Sort-Object FullName -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName }
if (-not $hostCompiler) {
  $hostCompiler = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe' -File -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName }
}
if (-not $hostCompiler) { throw 'The VS 2022 x64 cl.exe host compiler was not found' }

if (-not (Test-Path (Join-Path $SourceDir '.git'))) {
  Invoke-Checked { git clone --depth 1 --filter=blob:none --single-branch --branch $Tag https://github.com/intel/llvm $SourceDir } 'git clone'
}
$actual = (& git -C $SourceDir rev-parse HEAD).Trim()
if ($actual -ne $Commit) { throw "intel/llvm commit mismatch: expected $Commit, got $actual" }

# Windows HIP 7.1 returns zero for hipDeviceAttributeManagedMemory even though hipMallocManaged and
# hipFree succeed. UR consequently reports no shared-USM aspect and DPC++ rejects malloc_shared()
# before calling HIP. Advertise ordinary access plus device atomics on Windows; concurrent-access
# bits remain governed by the real driver attribute, and the miner synchronizes every host/device
# handoff. The complete hash-vector gate exercises the shared result atomics on the target driver.
$hipDevice = Join-Path $SourceDir 'unified-runtime\source\adapters\hip\device.cpp'
$sharedOld = '    // query if/how the device can access managed memory associated to it' + "`n" +
  '    ur_device_usm_access_capability_flags_t Value = {};' + "`n" +
  '    if (getAttribute(hDevice, hipDeviceAttributeManagedMemory)) {'
$sharedNew = '    // query if/how the device can access managed memory associated to it' + "`n" +
  '    ur_device_usm_access_capability_flags_t Value = {};' + "`n" +
  '#if defined(_WIN32)' + "`n" +
  '    Value = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |' + "`n" +
  '            UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS;' + "`n" +
  '#endif' + "`n" +
  '    if (getAttribute(hDevice, hipDeviceAttributeManagedMemory)) {'
Replace-RequiredText $hipDevice $sharedOld $sharedNew 'Windows HIP managed-memory capability'

# Intel's documentation still labels Windows HIP unsupported, but current UR contains explicit WIN32
# amdhip64/COMGR import-library handling. Build that code path with AMDGPU enabled instead of relying on
# the official Windows archive, which omits both AMDGPU and ur_adapter_hip.dll.
$configure = @(
  "$SourceDir\buildbot\configure.py", '--hip', '--no-assertions', '--cmake-gen', 'Ninja',
  '-s', $SourceDir, '-o', $BuildDir,
  "--cmake-opt=-DCMAKE_C_COMPILER=$hostCompiler",
  "--cmake-opt=-DCMAKE_CXX_COMPILER=$hostCompiler",
  '--cmake-opt=-DSYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB=OFF',
  "--cmake-opt=-DUR_USE_EXTERNAL_UMF=OFF",
  "--cmake-opt=-DUR_HIP_ROCM_DIR=$RocmPath",
  "--cmake-opt=-DUR_HIP_INCLUDE_DIR=$RocmPath\include",
  "--cmake-opt=-DUR_HIP_HSA_INCLUDE_DIRS=$RocmPath\include",
  "--cmake-opt=-DUR_HIP_LIB_DIR=$RocmPath\lib"
)
Invoke-Checked { python @configure } 'DPC++ configure'
$compile = { python "$SourceDir\buildbot\compile.py" -s $SourceDir -o $BuildDir -j $Jobs -t deploy-sycl-toolchain }
& $compile
if ($LASTEXITCODE -ne 0) {
  # The Windows deploy target currently requests the Debug import-library name even when the
  # internally built UMF install exports only umf.lib. The binaries are identical here; provide
  # the generated alias and let Ninja resume instead of rebuilding the toolchain.
  $umfd = Join-Path $BuildDir 'unified-runtimed\install\lib\umfd.lib'
  $umf = Get-ChildItem $BuildDir -Recurse -Filter 'umf.lib' -File -ErrorAction SilentlyContinue |
    Sort-Object @{ Expression = { if ($_.FullName -like '*unified-runtimed*') { 0 } else { 1 } } }, FullName |
    Select-Object -First 1
  if (-not $umf) { throw "DPC++ build failed and no generated umf.lib was found under $BuildDir" }
  New-Item -ItemType Directory -Force (Split-Path $umfd) | Out-Null
  Copy-Item $umf.FullName $umfd -Force
  & $compile
  if ($LASTEXITCODE -ne 0) { throw "DPC++ build failed with exit code $LASTEXITCODE" }
}

$deployDir = Join-Path $BuildDir 'install'
$required = @('bin\clang++.exe', 'bin\sycl9.dll', 'bin\sycl-jit.dll', 'bin\ur_adapter_hip.dll',
              'lib\clang\23\lib\amdgcn-amd-amdhsa-llvm\libspirv.l32.signed_char.bc')
foreach ($relative in $required) {
  if (-not (Test-Path (Join-Path $deployDir $relative))) { throw "DPC++ deploy omitted $relative" }
}
if ($InstallDir) {
  $parent = Split-Path -Parent $InstallDir
  if ($parent) { New-Item -ItemType Directory -Force $parent | Out-Null }
  Remove-Item $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
  Move-Item $deployDir $InstallDir
  $deployDir = $InstallDir
}
Write-Host "DPC++ AMD Windows toolchain ready at $deployDir"
