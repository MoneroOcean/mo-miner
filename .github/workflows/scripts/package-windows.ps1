param(
  [string]$Version = "",
  [string]$Archive = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if ($PSVersionTable.PSVersion.Major -ge 7) {
  $PSNativeCommandUseErrorActionPreference = $true
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
Set-Location $repoRoot
if (-not $env:ONEAPI_ROOT) {
  $env:ONEAPI_ROOT = "C:\Program Files (x86)\Intel\oneAPI"
}

if (-not $Version) {
  $Version = if ($env:GITHUB_REF_TYPE -eq 'tag' -and $env:GITHUB_REF_NAME -match '^v?[0-9]') {
    $env:GITHUB_REF_NAME
  } else {
    (Get-Content package.json | ConvertFrom-Json).version
  }
}
$Version = $Version -replace '^[vV]', ''
if ($Version -notmatch '^[0-9][0-9A-Za-z.-]*$') {
  throw "Invalid release version: $Version"
}

. "$PSScriptRoot/windows-dll-deps.ps1"

$root = "mom-v$Version"
if (-not $Archive) {
  $Archive = "mom-v$Version-win.zip"
}
$archiveFullPath = [IO.Path]::GetFullPath($Archive)
foreach ($cleanupRoot in @("release", "release-build")) {
  $cleanupFullPath = [IO.Path]::GetFullPath($cleanupRoot)
  $cleanupDescendantPrefix = $cleanupFullPath + [IO.Path]::DirectorySeparatorChar
  if ([string]::Equals($archiveFullPath, $cleanupFullPath, [StringComparison]::OrdinalIgnoreCase) -or
      $archiveFullPath.StartsWith($cleanupDescendantPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Archive path must not be inside cleanup directory: $Archive"
  }
}
$packageDir = "release/$root"
$libsDir = Join-Path $packageDir "libs"
$nodeExe = if ($env:NODE_BIN) { $env:NODE_BIN } else { (Get-Command node.exe -ErrorAction Stop | Select-Object -First 1).Source }

function Assert-BuildArtifact {
  param([string]$Path, [string]$Reason)
  if (-not (Test-Path $Path)) {
    throw "$Path is missing; $Reason"
  }
}

function Assert-WindowsPeArtifact {
  param([string]$Path, [string]$Reason)
  Assert-BuildArtifact $Path $Reason
  $stream = [IO.File]::OpenRead((Resolve-Path $Path).Path)
  try {
    if ($stream.Length -lt 2 -or $stream.ReadByte() -ne 0x4d -or $stream.ReadByte() -ne 0x5a) {
      throw "$Path is not a Windows PE binary; rebuild it on Windows before packaging."
    }
  } finally {
    $stream.Dispose()
  }
}

if (Test-Path -LiteralPath $Archive -PathType Container) {
  throw "Archive path must not be a directory: $Archive"
}
$archiveItem = Get-Item -LiteralPath $Archive -Force -ErrorAction SilentlyContinue
if ($archiveItem -and -not $archiveItem.PSIsContainer -and
    (($archiveItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
  throw "Archive path must not be a reparse point: $Archive"
}

$multiCompilerDir = "build/win/compilers"
foreach ($compiler in @('oneapi', 'dpcpp', 'dpcpp-opencl', 'acpp-cuda', 'acpp-hip')) {
  Assert-WindowsPeArtifact (Join-Path $multiCompilerDir "$compiler\mom.node") "build the $compiler native addon before packaging."
  Assert-WindowsPeArtifact (Join-Path $multiCompilerDir "$compiler\sycl.dll") "Windows release packages require the $compiler SYCL support."
}

Remove-Item -Recurse -Force release, release-build -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $packageDir, $libsDir, release-build | Out-Null

$bundlePath = (Resolve-Path release-build).Path + "\mom.bundle.cjs"
& npx.cmd --no-install esbuild mom.js `
  --bundle `
  --platform=node `
  --format=cjs `
  --outfile="$bundlePath"
if ($LASTEXITCODE -ne 0) {
  throw "esbuild failed with exit code $LASTEXITCODE. Run npm install --ignore-scripts first."
}
Assert-BuildArtifact $bundlePath "esbuild did not produce the release bundle."
Copy-Item $nodeExe "$packageDir/mom-node.exe"
Copy-Item $bundlePath "$packageDir/mom.bundle.cjs"
@'
@echo off
setlocal
set "MOM_DIR=%~dp0"
set "MOM_LIBS=%MOM_DIR%libs"
set "PATH=%MOM_LIBS%;%MOM_DIR%;%CD%;%PATH%"
if not defined CUDA_PATH if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\ptxas.exe" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if defined CUDA_PATH if exist "%CUDA_PATH%\bin" set "PATH=%CUDA_PATH%\bin;%PATH%"
if not defined VSCMD_VER if exist "C:\BuildTools\Common7\Tools\VsDevCmd.bat" call "C:\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul
if not defined VSCMD_VER if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul
if not defined MOM_COMMAND set "MOM_COMMAND=mom"
if not defined MOM_GPU_BACKEND for /f "usebackq delims=" %%V in (`powershell.exe -NoProfile -Command "$ids=@((Get-CimInstance Win32_VideoController).PNPDeviceID); $pci=@(); foreach($id in $ids){if($id -match 'VEN_([0-9A-Fa-f]{4})'){$pci+=$Matches[1].ToUpperInvariant()}}; $v=@(); if($pci -contains '1002'){$v+='amd'}; if($pci -contains '10DE'){$v+='nvidia'}; if($pci -contains '8086'){$v+='intel'}; if($pci.Where({$_ -notin @('1002','10DE','8086')}).Count){$v+='opencl'}; if($v.Count -eq 1){$v[0]}"`) do set "MOM_GPU_BACKEND=%%V"
if defined MOM_GPU_INDEX for /f "delims=0123456789" %%I in ("%MOM_GPU_INDEX%") do (
  echo MOM_GPU_INDEX must be a non-negative integer
  exit /b 2
)
set "MOM_NATIVE_DIR=%MOM_LIBS%"
if not defined MOM_NATIVE_PATH set "MOM_NATIVE_PATH_WAS_UNSET=1"
if /I "%MOM_GPU_BACKEND%"=="intel" if not defined MOM_NATIVE_PATH set "MOM_NATIVE_PATH=%MOM_LIBS%\oneapi\mom.node"
if /I "%MOM_GPU_BACKEND%"=="intel" if not defined UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS set "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1"
rem Keep every Intel GPU visible: the addon applies MOM_GPU_INDEX after stable hardware-name sorting.
if /I "%MOM_GPU_BACKEND%"=="intel" if not defined ONEAPI_DEVICE_SELECTOR set "ONEAPI_DEVICE_SELECTOR=level_zero:gpu"
rem Only trusted policy keys and newly isolated numeric masks cross the cmd assignment boundary.
rem Inherited masks stay untouched; query failure selects portable, invalid indexed masks abort.
set "MOM_VENDOR_RUNTIME="
if /I "%MOM_GPU_BACKEND%"=="nvidia" set "MOM_VENDOR_RUNTIME=pending"
if /I "%MOM_GPU_BACKEND%"=="amd" set "MOM_VENDOR_RUNTIME=pending"
if defined MOM_VENDOR_RUNTIME for /f "usebackq tokens=1,* delims==" %%K in (`^""%MOM_DIR%mom-node.exe" -e "const p=require(process.env.MOM_DIR+'compiler-policy.js');try{const e=p.vendorDeviceEnv(process.env);const g=p.gpuFromEnv();e.MOM_VENDOR_RUNTIME=p.selection('',g,'win32',g==='nvidia'?(p.nvidiaComputeCapability()??0):null).key;for(const[k,v]of Object.entries(e))console.log(k+'='+v);}catch(e){console.error(e.message);console.log('MOM_VENDOR_RUNTIME=error');process.exitCode=2;}"^"`) do set "%%K=%%L"
if /I "%MOM_VENDOR_RUNTIME%"=="pending" exit /b 2
if /I "%MOM_VENDOR_RUNTIME%"=="error" exit /b 2
if defined MOM_VENDOR_RUNTIME if not defined MOM_NATIVE_PATH set "MOM_NATIVE_PATH=%MOM_LIBS%\%MOM_VENDOR_RUNTIME%\mom.node"
if /I "%MOM_VENDOR_RUNTIME%"=="acpp-cuda" if not defined ACPP_VISIBILITY_MASK set "ACPP_VISIBILITY_MASK=cuda"
if /I "%MOM_VENDOR_RUNTIME%"=="dpcpp" if defined MOM_GPU_INDEX if not defined ONEAPI_DEVICE_SELECTOR set "ONEAPI_DEVICE_SELECTOR=cuda:%MOM_GPU_INDEX%"
if /I "%MOM_VENDOR_RUNTIME%"=="dpcpp" if not defined ONEAPI_DEVICE_SELECTOR set "ONEAPI_DEVICE_SELECTOR=cuda:gpu"
if /I "%MOM_VENDOR_RUNTIME%"=="acpp-hip" if not defined ACPP_VISIBILITY_MASK set "ACPP_VISIBILITY_MASK=hip"
if defined MOM_OPENCL_DEVICE_TYPE if /I not "%MOM_OPENCL_DEVICE_TYPE%"=="gpu" if /I not "%MOM_OPENCL_DEVICE_TYPE%"=="cpu" (
  echo MOM_OPENCL_DEVICE_TYPE must be gpu or cpu
  exit /b 2
)
if /I "%MOM_GPU_BACKEND%"=="opencl" if not defined MOM_NATIVE_PATH set "MOM_NATIVE_PATH=%MOM_LIBS%\dpcpp-opencl\mom.node"
if /I "%MOM_GPU_BACKEND%"=="opencl" if /I "%MOM_OPENCL_DEVICE_TYPE%"=="cpu" if not defined ONEAPI_DEVICE_SELECTOR set "ONEAPI_DEVICE_SELECTOR=opencl:cpu"
if /I "%MOM_GPU_BACKEND%"=="opencl" if not defined ONEAPI_DEVICE_SELECTOR set "ONEAPI_DEVICE_SELECTOR=opencl:gpu"
rem UR's Windows proxy loader only searches beside its own DLL. Force the one isolated adapter
rem selected by this launcher; drive-letter paths must be quoted inside UR_ADAPTERS_FORCE_LOAD.
rem Quote DLL values directly: outer SET quotes would expose path punctuation.
if /I "%MOM_GPU_BACKEND%"=="intel" if not defined UR_ADAPTERS_FORCE_LOAD set UR_ADAPTERS_FORCE_LOAD="%MOM_LIBS%\oneapi\ur_adapter_level_zero_v2.dll","%MOM_LIBS%\oneapi\ur_adapter_opencl.dll"
if /I "%MOM_VENDOR_RUNTIME%"=="dpcpp" if not defined UR_ADAPTERS_FORCE_LOAD set UR_ADAPTERS_FORCE_LOAD="%MOM_LIBS%\dpcpp\ur_adapter_cuda.dll"
if /I "%MOM_GPU_BACKEND%"=="opencl" if not defined UR_ADAPTERS_FORCE_LOAD set UR_ADAPTERS_FORCE_LOAD="%MOM_LIBS%\dpcpp-opencl\ur_adapter_opencl.dll"
if not defined MOM_NATIVE_PATH set "MOM_NATIVE_PATH=%MOM_LIBS%\oneapi\mom.node"
if defined MOM_NATIVE_PATH_WAS_UNSET set "MOM_NATIVE_PATH_LAUNCHER_DEFAULT=%MOM_NATIVE_PATH%"
set "MOM_NATIVE_PATH_WAS_UNSET="
if not defined MOM_RUNTIME_DIR if /I "%MOM_GPU_BACKEND%"=="intel" set "MOM_RUNTIME_DIR=%MOM_LIBS%\oneapi"
if not defined MOM_RUNTIME_DIR if defined MOM_VENDOR_RUNTIME set "MOM_RUNTIME_DIR=%MOM_LIBS%\%MOM_VENDOR_RUNTIME%"
if not defined MOM_RUNTIME_DIR if /I "%MOM_GPU_BACKEND%"=="opencl" set "MOM_RUNTIME_DIR=%MOM_LIBS%\dpcpp-opencl"
if not defined MOM_RUNTIME_DIR set "MOM_RUNTIME_DIR=%MOM_LIBS%\oneapi"
rem Share oneAPI JIT dependencies; the selected runtime's sycl.dll must win PATH lookup.
if /I "%MOM_GPU_BACKEND%"=="opencl" set "PATH=%MOM_LIBS%\oneapi;%PATH%"
if defined MOM_RUNTIME_DIR set "PATH=%MOM_RUNTIME_DIR%;%MOM_RUNTIME_DIR%\hipSYCL;%PATH%"
if /I not "%MOM_GPU_BACKEND%"=="opencl" if not defined OCL_ICD_FILENAMES for %%F in ("%MOM_LIBS%\intelocl*.dll") do if exist "%%~fF" set "OCL_ICD_FILENAMES=%%~fF"
"%MOM_DIR%mom-node.exe" "%MOM_DIR%mom.bundle.cjs" %*
exit /b %ERRORLEVEL%
'@ | Set-Content -Encoding ascii "$packageDir/mom.cmd"

$releaseFiles = @(
  'package.json', 'compiler-policy.js', 'gpu-tuning.js', 'README.md', 'GPU-CONFIG.md',
  'LICENSE', 'scripts/install.bat', 'scripts/install.ps1', 'scripts/install-cutlass.ps1'
)
Copy-Item $releaseFiles "$packageDir/"
New-Item -ItemType Directory -Force "$packageDir/helper" | Out-Null
Copy-Item 'helper/hash.js' "$packageDir/helper/"
Copy-Item "$multiCompilerDir/*" $libsDir -Recurse -Force
# Older cached build trees predate the AdaptiveCpp deployment-manifest copy. Repair them while
# packaging; new builds already snapshot libdevice in this relocatable, upstream-defined path.
$acppCudaLibdevice = Join-Path $libsDir 'acpp-cuda\hipSYCL\ext\bitcode\ptx\libdevice.10.bc'
if (-not (Test-Path $acppCudaLibdevice)) {
  $cudaRoots = @($env:CUDA_PATH, 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6') |
    Where-Object { $_ }
  $libdevice = Get-ChildItem -Path ($cudaRoots | ForEach-Object {
    Join-Path $_ 'nvvm\libdevice\libdevice.10.bc'
  }) -File -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $libdevice) {
    throw 'AdaptiveCpp CUDA packaging requires libdevice.10.bc.'
  }
  New-Item -ItemType Directory -Force (Split-Path -Parent $acppCudaLibdevice) | Out-Null
  Copy-Item $libdevice.FullName $acppCudaLibdevice -Force
}
foreach ($requiredRuntime in @(
  'dpcpp-opencl\mom.node',
  'dpcpp-opencl\sycl.dll',
  'acpp-cuda\libomp.dll',
  'acpp-cuda\hipSYCL\rt-backend-cuda.dll',
  'acpp-cuda\hipSYCL\rt-backend-omp.dll',
  'acpp-cuda\hipSYCL\bitcode\libkernel-sscp-host-full.bc',
  'acpp-cuda\hipSYCL\ext\bitcode\ptx\libdevice.10.bc',
  'acpp-hip\libomp.dll',
  'acpp-hip\hipSYCL\rt-backend-hip.dll',
  'acpp-hip\hipSYCL\rt-backend-omp.dll',
  'acpp-hip\hipSYCL\bitcode\libkernel-sscp-host-full.bc'
)) {
  if (-not (Test-Path (Join-Path $libsDir $requiredRuntime))) {
    throw "Windows release worker is missing required compiler runtime: $requiredRuntime"
  }
}
# Source snapshots carry compiler-matched runtimes. Retain only the two native binaries before
# copying each exact release cohort; portable-local nightly adapters/JIT must not survive.
foreach ($worker in @('dpcpp', 'dpcpp-opencl')) {
  Get-ChildItem (Join-Path $libsDir $worker) -Force |
    Where-Object { $_.Name -notin @('mom.node', 'sycl.dll') } |
    Remove-Item -Recurse -Force
}
$hipPackageDir = Join-Path $libsDir 'acpp-hip'
Copy-MominerHipDynamicRuntimeFiles -PackageDir $hipPackageDir
$hiprtcCompiler = Get-ChildItem $hipPackageDir -Filter 'hiprtc*.dll' -File `
  -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike 'hiprtc-builtins*' } |
  Select-Object -First 1
if (-not $hiprtcCompiler) {
  throw 'Windows AMD source-JIT package is missing the HIPRTC compiler DLL.'
}
foreach ($pattern in @('hiprtc-builtins*.dll', 'amd_comgr0*.dll')) {
  if (-not (Get-ChildItem $hipPackageDir -Filter $pattern -File `
      -ErrorAction SilentlyContinue | Select-Object -First 1)) {
    throw "Windows AMD source-JIT package is missing $pattern."
  }
}
# DPC++ adapters are loaded dynamically, so dumpbin cannot discover them from sycl.dll. Copy the
# release runtimes explicitly here; build-windows-multicompiler intentionally does not snapshot
# compiler executables or debug DLLs into build/win/compilers/dpcpp.
$savedDpcppAcpp = $env:MOM_ACPP_DIR
$savedHipPath = $env:HIP_PATH
$savedDpcppRocm = $env:ROCM_PATH
$savedDpcppOneApi = $env:ONEAPI_ROOT
$savedDpcppToolchain = $env:MOM_DPCPP_DIR
# The DPC++ policy worker carries CUDA plus the vendor-neutral SPIR-V/OpenCL fallback. Keep its own
# UR runtimes/adapters while resolving ordinary dependencies later through the closure. Intel's
# OpenCL device compiler remains in the isolated oneAPI worker; duplicating it here would add
# roughly 250 MiB without helping third-party system ICDs.
$dpcppToolchain = if ($env:MOM_DPCPP_CUDA_DIR) { $env:MOM_DPCPP_CUDA_DIR }
  elseif ($savedDpcppToolchain) { $savedDpcppToolchain }
  elseif (Test-Path 'C:\Tools\dpcpp\bin') { 'C:\Tools\dpcpp' }
  else { $null }
if (-not $dpcppToolchain) { throw 'The open DPC++ runtime toolchain was not found.' }
$dpcppLoader = Join-Path $dpcppToolchain 'bin\ur_loader.dll'
Assert-WindowsPeArtifact $dpcppLoader 'restore the matching DPC++ Unified Runtime loader before packaging.'
$env:MOM_DPCPP_DIR = $dpcppToolchain
Remove-Item Env:MOM_ACPP_DIR, Env:HIP_PATH, Env:ROCM_PATH, Env:ONEAPI_ROOT -ErrorAction SilentlyContinue
Copy-MominerOptionalRuntimeFiles -PackageDir (Join-Path $libsDir 'dpcpp') `
  -SourceRoots @(Get-MominerDpcppBinDir)
# Equal UR API versions do not imply matching DDI layouts. Never replace this toolchain's loader
# or fill a missing adapter from oneAPI; its SYCL/proxy/loader/adapters must remain one cohort.
Copy-Item $dpcppLoader (Join-Path $libsDir 'dpcpp\ur_loader.dll') -Force
if ($savedDpcppAcpp) { $env:MOM_ACPP_DIR = $savedDpcppAcpp }
if ($savedHipPath) { $env:HIP_PATH = $savedHipPath }
if ($savedDpcppRocm) { $env:ROCM_PATH = $savedDpcppRocm }
if ($savedDpcppOneApi) { $env:ONEAPI_ROOT = $savedDpcppOneApi }
if ($savedDpcppToolchain) { $env:MOM_DPCPP_DIR = $savedDpcppToolchain }
else { Remove-Item Env:MOM_DPCPP_DIR -ErrorAction SilentlyContinue }

# oneAPI UR adapters are dlopen() dependencies and therefore absent from dumpbin's closure.
# Copy them with DPC++/AdaptiveCpp roots temporarily hidden so same-named runtimes cannot leak
# into the oneAPI directory. The other compiler snapshots already carry their toolchain DLLs.
$savedDpcpp = $env:MOM_DPCPP_DIR
$savedAcpp = $env:MOM_ACPP_DIR
$savedHip = $env:HIP_PATH
Remove-Item Env:MOM_DPCPP_DIR, Env:MOM_ACPP_DIR, Env:HIP_PATH -ErrorAction SilentlyContinue
$oneApiCoreRuntimes = @(
  'sycl9.dll', 'ur_win_proxy_loader.dll', 'ur_loader.dll', 'OpenCL.dll',
  'ur_adapter_opencl.dll', 'ur_adapter_level_zero_v2.dll',
  'umf.dll', 'libhwloc-15.dll', 'libmmd.dll'
)
# The optional copier keeps existing names. Refresh only staged cohort files so stale snapshots
# cannot mix with this selected oneAPI installation; missing source files must fail below.
foreach ($runtime in ($oneApiCoreRuntimes + @('sycl-jit.dll'))) {
  $staged = Join-Path $libsDir "oneapi\$runtime"
  if (Test-Path -LiteralPath $staged) { Remove-Item -LiteralPath $staged -Force }
}
Copy-MominerOptionalRuntimeFiles -PackageDir (Join-Path $libsDir 'oneapi') `
  -SourceRoots @(Get-MominerOneApiBinDirs)
if ($savedDpcpp) { $env:MOM_DPCPP_DIR = $savedDpcpp }
if ($savedAcpp) { $env:MOM_ACPP_DIR = $savedAcpp }
if ($savedHip) { $env:HIP_PATH = $savedHip }

# The portable worker uses the qualified oneAPI SYCL/UR cohort. Keep its loader and adapters together;
# mixing a nightly adapter with oneAPI's loader can dispatch buffer writes to the wrong DDI slot.
foreach ($runtime in $oneApiCoreRuntimes) {
  $source = Join-Path $libsDir "oneapi\$runtime"
  Assert-WindowsPeArtifact $source 'the portable worker requires the complete oneAPI runtime cohort.'
  Copy-Item $source (Join-Path $libsDir "dpcpp-opencl\$runtime") -Force
}
# A stale portable-local JIT would win DLL lookup ahead of the shared, matching oneAPI library.
$portableJit = Join-Path $libsDir 'dpcpp-opencl\sycl-jit.dll'
if (Test-Path -LiteralPath $portableJit) { Remove-Item -LiteralPath $portableJit -Force }
# The large matching JIT library is shared through the launcher's oneAPI PATH, not duplicated.
Assert-WindowsPeArtifact (Join-Path $libsDir 'oneapi\sycl-jit.dll') 'the portable worker requires the matching oneAPI JIT library.'

foreach ($runtime in @(
  'dpcpp\sycl9.dll',
  'dpcpp\ur_win_proxy_loader.dll',
  'dpcpp\ur_loader.dll',
  'dpcpp\ur_adapter_cuda.dll',
  'dpcpp\ur_adapter_opencl.dll',
  'dpcpp\ur_adapter_level_zero_v2.dll',
  'dpcpp-opencl\sycl9.dll',
  'dpcpp-opencl\ur_win_proxy_loader.dll',
  'dpcpp-opencl\umf.dll',
  'dpcpp-opencl\libhwloc-15.dll',
  'dpcpp-opencl\libmmd.dll',
  'dpcpp-opencl\ur_loader.dll',
  'dpcpp-opencl\ur_adapter_opencl.dll',
  'dpcpp-opencl\ur_adapter_level_zero_v2.dll',
  'dpcpp-opencl\OpenCL.dll'
)) {
  if (-not (Test-Path (Join-Path $libsDir $runtime))) {
    throw "Windows release worker is missing required Unified Runtime file: $runtime"
  }
}
# Ship the KawPow source-JIT assets beside the DPC++ worker when present.
if (Test-Path "sycl/kawpow/device.inc") {
  # The CUDA source-JIT locates this relative to the loaded DPC++ sycl.dll.
  Copy-Item "sycl/kawpow/device.inc" (Join-Path $libsDir 'dpcpp\kawpow_device.inc')
  Copy-Item "sycl/kawpow/keccak.inc" (Join-Path $libsDir 'dpcpp\kawpow_keccak.inc')
}
$entryPaths = @("$packageDir/mom-node.exe")
Copy-MominerDllClosure -PackageDir $libsDir -EntryPaths $entryPaths
foreach ($compilerDir in Get-ChildItem $libsDir -Directory) {
  $runtimeEntries = Get-ChildItem $compilerDir.FullName `
    -Include '*.dll','opt.exe','llc.exe','lld.exe','lld-link.exe' -File -Recurse
  if ($compilerDir.Name -eq 'acpp-hip') {
    # Current snapshots contain the selected accelerator plugin plus AdaptiveCpp's required OMP
    # host plugin; retain this filter for older trees so their irrelevant CUDA plugin cannot pull
    # cudart into an otherwise HIP-only closure.
    $runtimeEntries = $runtimeEntries | Where-Object { $_.Name -ne 'rt-backend-cuda.dll' }
  } elseif ($compilerDir.Name -eq 'acpp-cuda') {
    # Symmetric compatibility filter for pre-trim NVIDIA snapshots.
    $runtimeEntries = $runtimeEntries | Where-Object { $_.Name -ne 'rt-backend-hip.dll' }
  }
  $entries = @((Join-Path $compilerDir.FullName 'mom.node'), (Join-Path $compilerDir.FullName 'sycl.dll')) +
    @($runtimeEntries | ForEach-Object FullName)
  if (Test-Path $entries[0]) { Copy-MominerDllClosure -PackageDir $compilerDir.FullName -EntryPaths $entries }
}

if (Test-Path "$packageDir/tests") {
  throw "Release package unexpectedly contains tests/."
}

$archiveDirectory = [IO.Path]::GetDirectoryName($archiveFullPath)
do {
  $archiveTempPath = Join-Path $archiveDirectory ".mom-archive-$([Guid]::NewGuid().ToString('N')).zip"
} while (Test-Path -LiteralPath $archiveTempPath)
try {
  Compress-Archive -Path $packageDir -DestinationPath $archiveTempPath
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $packageRootEntry = (Split-Path -Leaf $packageDir) + '/'
  $zipArchive = [IO.Compression.ZipFile]::Open($archiveTempPath, [IO.Compression.ZipArchiveMode]::Update)
  try {
    $rootEntries = @($zipArchive.Entries | Where-Object {
      $_.FullName.Replace('\', '/') -eq $packageRootEntry
    })
    if ($rootEntries.Count -eq 0) {
      [void]$zipArchive.CreateEntry($packageRootEntry)
    } elseif ($rootEntries.Count -gt 1) {
      throw "Temporary archive contains duplicate package-root entries: $packageRootEntry"
    }
  } finally {
    $zipArchive.Dispose()
  }
  if (Test-Path -LiteralPath $archiveFullPath -PathType Leaf) {
    # NullString preserves a null backup path; PowerShell otherwise passes an empty string.
    [IO.File]::Replace($archiveTempPath, $archiveFullPath, [NullString]::Value)
  } else {
    [IO.File]::Move($archiveTempPath, $archiveFullPath)
  }
} finally {
  if (Test-Path -LiteralPath $archiveTempPath -PathType Leaf) {
    Remove-Item -Force -LiteralPath $archiveTempPath -ErrorAction SilentlyContinue
  }
}
Write-Output $Archive
