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

. "$PSScriptRoot/windows-dll-deps.ps1"

if (-not $Version) {
  $Version = if ($env:GITHUB_REF_NAME -and $env:GITHUB_REF_NAME -match '^v?[0-9]') {
    $env:GITHUB_REF_NAME
  } else {
    (Get-Content package.json | ConvertFrom-Json).version
  }
}
$Version = $Version -replace '^v', ''

$root = "mom-v$Version"
if (-not $Archive) {
  $Archive = "mom-v$Version-win.zip"
}
$packageDir = "release/$root"
$libsDir = Join-Path $packageDir "libs"
$nodeExe = if ($env:NODE_BIN) { $env:NODE_BIN } else { (Get-Command node.exe).Source }

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

Assert-WindowsPeArtifact "build/win/Release/mom.node" "build the native addon before packaging."
Assert-WindowsPeArtifact "build/win/Release/sycl.dll" "Windows release packages require SYCL support."
$multiCompilerDir = "build/win/compilers"
$hasMultiCompiler = Test-Path "$multiCompilerDir/oneapi/mom.node"

Remove-Item -Recurse -Force release, release-build, $Archive -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $packageDir, $libsDir, release-build | Out-Null

$bundlePath = (Resolve-Path release-build).Path + "\mom.bundle.cjs"
npx --no-install esbuild mom.js `
  --bundle `
  --platform=node `
  --format=cjs `
  --outfile="$bundlePath"
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
if /I "%MOM_GPU_BACKEND%"=="intel" if defined MOM_GPU_INDEX if not defined ONEAPI_DEVICE_SELECTOR set "ONEAPI_DEVICE_SELECTOR=level_zero:gpu"
if /I "%MOM_GPU_BACKEND%"=="intel" if not defined ONEAPI_DEVICE_SELECTOR set "ONEAPI_DEVICE_SELECTOR=level_zero:gpu"
if /I "%MOM_GPU_BACKEND%"=="nvidia" if not defined MOM_NATIVE_PATH set "MOM_NATIVE_PATH=%MOM_LIBS%\dpcpp\mom.node"
if /I "%MOM_GPU_BACKEND%"=="nvidia" if defined MOM_GPU_INDEX if not defined ONEAPI_DEVICE_SELECTOR set "ONEAPI_DEVICE_SELECTOR=cuda:%MOM_GPU_INDEX%"
if /I "%MOM_GPU_BACKEND%"=="nvidia" if not defined ONEAPI_DEVICE_SELECTOR set "ONEAPI_DEVICE_SELECTOR=cuda:gpu"
if /I "%MOM_GPU_BACKEND%"=="amd" if not defined MOM_NATIVE_PATH set "MOM_NATIVE_PATH=%MOM_LIBS%\acpp-hip\mom.node"
if /I "%MOM_GPU_BACKEND%"=="amd" if defined MOM_GPU_INDEX if not defined HIP_VISIBLE_DEVICES set "HIP_VISIBLE_DEVICES=%MOM_GPU_INDEX%"
if /I "%MOM_GPU_BACKEND%"=="amd" if not defined ACPP_VISIBILITY_MASK set "ACPP_VISIBILITY_MASK=hip"
if defined MOM_OPENCL_DEVICE_TYPE if /I not "%MOM_OPENCL_DEVICE_TYPE%"=="gpu" if /I not "%MOM_OPENCL_DEVICE_TYPE%"=="cpu" (
  echo MOM_OPENCL_DEVICE_TYPE must be gpu or cpu
  exit /b 2
)
if /I "%MOM_GPU_BACKEND%"=="opencl" if not defined MOM_NATIVE_PATH set "MOM_NATIVE_PATH=%MOM_LIBS%\dpcpp-opencl\mom.node"
if /I "%MOM_GPU_BACKEND%"=="opencl" if /I "%MOM_OPENCL_DEVICE_TYPE%"=="cpu" if not defined ONEAPI_DEVICE_SELECTOR set "ONEAPI_DEVICE_SELECTOR=opencl:cpu"
if /I "%MOM_GPU_BACKEND%"=="opencl" if not defined ONEAPI_DEVICE_SELECTOR set "ONEAPI_DEVICE_SELECTOR=opencl:gpu"
rem UR's Windows proxy loader only searches beside its own DLL. Force the one isolated adapter
rem selected by this launcher; drive-letter paths must be quoted inside UR_ADAPTERS_FORCE_LOAD.
if /I "%MOM_GPU_BACKEND%"=="intel" if not defined UR_ADAPTERS_FORCE_LOAD set "UR_ADAPTERS_FORCE_LOAD="%MOM_LIBS%\oneapi\ur_adapter_level_zero_v2.dll","%MOM_LIBS%\oneapi\ur_adapter_opencl.dll""
if /I "%MOM_GPU_BACKEND%"=="nvidia" if not defined UR_ADAPTERS_FORCE_LOAD set "UR_ADAPTERS_FORCE_LOAD="%MOM_LIBS%\dpcpp\ur_adapter_cuda.dll""
if /I "%MOM_GPU_BACKEND%"=="opencl" if not defined UR_ADAPTERS_FORCE_LOAD set "UR_ADAPTERS_FORCE_LOAD="%MOM_LIBS%\dpcpp-opencl\ur_adapter_opencl.dll""
if not defined MOM_NATIVE_PATH set "MOM_NATIVE_PATH=%MOM_LIBS%\oneapi\mom.node"
if defined MOM_NATIVE_PATH_WAS_UNSET set "MOM_NATIVE_PATH_LAUNCHER_DEFAULT=%MOM_NATIVE_PATH%"
set "MOM_NATIVE_PATH_WAS_UNSET="
if not defined MOM_RUNTIME_DIR if /I "%MOM_GPU_BACKEND%"=="intel" set "MOM_RUNTIME_DIR=%MOM_LIBS%\oneapi"
if not defined MOM_RUNTIME_DIR if /I "%MOM_GPU_BACKEND%"=="nvidia" set "MOM_RUNTIME_DIR=%MOM_LIBS%\dpcpp"
if not defined MOM_RUNTIME_DIR if /I "%MOM_GPU_BACKEND%"=="amd" set "MOM_RUNTIME_DIR=%MOM_LIBS%\acpp-hip"
if not defined MOM_RUNTIME_DIR if /I "%MOM_GPU_BACKEND%"=="opencl" set "MOM_RUNTIME_DIR=%MOM_LIBS%\dpcpp-opencl"
if not defined MOM_RUNTIME_DIR set "MOM_RUNTIME_DIR=%MOM_LIBS%\oneapi"
if defined MOM_RUNTIME_DIR set "PATH=%MOM_RUNTIME_DIR%;%MOM_RUNTIME_DIR%\hipSYCL;%PATH%"
if /I "%MOM_GPU_BACKEND%"=="opencl" set "PATH=%MOM_LIBS%\dpcpp;%PATH%"
if /I not "%MOM_GPU_BACKEND%"=="opencl" if not defined OCL_ICD_FILENAMES for %%F in ("%MOM_LIBS%\intelocl*.dll") do if exist "%%~fF" set "OCL_ICD_FILENAMES=%%~fF"
"%MOM_DIR%mom-node.exe" "%MOM_DIR%mom.bundle.cjs" %*
exit /b %ERRORLEVEL%
'@ | Set-Content -Encoding ascii "$packageDir/mom.cmd"

$releaseFiles = @(
  'package.json', 'compiler-policy.js', 'README.md', 'DEVELOPMENT.md', 'GPU-COMPILERS.md',
  'LICENSE', 'scripts/install.bat', 'scripts/install.ps1', 'scripts/install-cutlass.ps1'
)
Copy-Item $releaseFiles "$packageDir/"
if ($hasMultiCompiler) {
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
  # Accept build directories made before the runtime-only DPC++ snapshot change, but never ship
  # their compiler/debug payload. The two linked binaries below plus the explicit runtime copy are
  # the complete production input.
  Get-ChildItem (Join-Path $libsDir 'dpcpp') -Force |
    Where-Object { $_.Name -notin @('mom.node', 'sycl.dll') } |
    Remove-Item -Recurse -Force
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
  $savedDpcppHip = $env:HIP_PATH
  $savedDpcppRocm = $env:ROCM_PATH
  $savedDpcppOneApi = $env:ONEAPI_ROOT
  $savedDpcppToolchain = $env:MOM_DPCPP_DIR
  # The DPC++ policy worker carries CUDA plus the vendor-neutral SPIR-V/OpenCL fallback. Keep its own
  # UR runtimes/adapters while resolving ordinary dependencies later through the closure. Intel's
  # OpenCL device compiler remains in the isolated oneAPI worker; duplicating it here would add
  # roughly 250 MiB without helping third-party system ICDs.
  $dpcppToolchain = if ($env:MOM_DPCPP_CUDA_DIR) { $env:MOM_DPCPP_CUDA_DIR }
    elseif (Test-Path 'C:\Tools\dpcpp\bin') { 'C:\Tools\dpcpp' }
    else { $savedDpcppToolchain }
  if (-not $dpcppToolchain) { throw 'The open DPC++ runtime toolchain was not found.' }
  $env:MOM_DPCPP_DIR = $dpcppToolchain
  Remove-Item Env:MOM_ACPP_DIR, Env:HIP_PATH, Env:ROCM_PATH, Env:ONEAPI_ROOT -ErrorAction SilentlyContinue
  Copy-MominerOptionalRuntimeFiles -PackageDir (Join-Path $libsDir 'dpcpp')
  # The pinned CUDA-enabled DPC++ snapshot does not build its own OpenCL UR adapter. The same
  # standards-only worker passed through oneAPI's adapter in the development image; copy that one
  # small, stable UR plugin explicitly so a driver-only release host has the identical path without
  # inheriting oneAPI's Intel OpenCL device-compiler payload.
  $dpcppOpenclAdapter = Join-Path $libsDir 'dpcpp\ur_adapter_opencl.dll'
  if (-not (Test-Path $dpcppOpenclAdapter) -and $savedDpcppOneApi) {
    $adapter = Get-ChildItem -Path @(
      (Join-Path $savedDpcppOneApi 'compiler\latest\bin\ur_adapter_opencl.dll'),
      (Join-Path $savedDpcppOneApi 'compiler\latest\bin\compiler\ur_adapter_opencl.dll')
    ) -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($adapter) { Copy-Item $adapter.FullName $dpcppOpenclAdapter }
  }
  if (-not (Test-Path $dpcppOpenclAdapter)) {
    throw 'The generic DPC++ worker requires ur_adapter_opencl.dll, but no compatible adapter was found.'
  }
  if ($savedDpcppAcpp) { $env:MOM_ACPP_DIR = $savedDpcppAcpp }
  if ($savedDpcppHip) { $env:HIP_PATH = $savedDpcppHip }
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
  Copy-MominerOptionalRuntimeFiles -PackageDir (Join-Path $libsDir 'oneapi')
  if ($savedDpcpp) { $env:MOM_DPCPP_DIR = $savedDpcpp }
  if ($savedAcpp) { $env:MOM_ACPP_DIR = $savedAcpp }
  if ($savedHip) { $env:HIP_PATH = $savedHip }

  # The Windows SYCL runtime's proxy loads ur_loader.dll from the directory containing sycl9.dll;
  # PATH is intentionally ignored. The open DPC++ toolchains omit that redistributable DLL and use
  # oneAPI's ABI-compatible loader in development, so make the dependency explicit in every worker.
  $urLoader = Join-Path $libsDir 'oneapi\ur_loader.dll'
  if (-not (Test-Path $urLoader)) { throw 'The oneAPI Unified Runtime loader was not packaged.' }
  foreach ($worker in @('dpcpp','dpcpp-opencl')) {
    Copy-Item $urLoader (Join-Path $libsDir "$worker\ur_loader.dll") -Force
  }
  # The portable worker has its own sycl9/proxy pair, so its selected adapter and OpenCL loader must
  # be colocated as well. This is a few MiB and avoids duplicating CUDA JIT/compiler payloads.
  foreach ($runtime in @('OpenCL.dll','ur_adapter_opencl.dll','ur_adapter_level_zero_v2.dll')) {
    $source = Join-Path $libsDir "dpcpp\$runtime"
    if (-not (Test-Path $source)) { throw "The generic DPC++ runtime is missing $runtime." }
    Copy-Item $source (Join-Path $libsDir "dpcpp-opencl\$runtime") -Force
  }
} else {
  Copy-Item build/win/Release/mom.node, build/win/Release/sycl.dll "$libsDir/"
  # Preserve the historical flat layout only for the explicitly supported single-worker build.
  # Unified packages always launch an isolated compiler directory and would otherwise duplicate the
  # complete oneAPI runtime at the archive root without any consumer.
  $savedRootDpcpp = $env:MOM_DPCPP_DIR
  $savedRootAcpp = $env:MOM_ACPP_DIR
  $savedRootHip = $env:HIP_PATH
  $savedRootRocm = $env:ROCM_PATH
  Remove-Item Env:MOM_DPCPP_DIR, Env:MOM_ACPP_DIR, Env:HIP_PATH, Env:ROCM_PATH -ErrorAction SilentlyContinue
  Copy-MominerOptionalRuntimeFiles -PackageDir $libsDir
  if ($savedRootDpcpp) { $env:MOM_DPCPP_DIR = $savedRootDpcpp }
  if ($savedRootAcpp) { $env:MOM_ACPP_DIR = $savedRootAcpp }
  if ($savedRootHip) { $env:HIP_PATH = $savedRootHip }
  if ($savedRootRocm) { $env:ROCM_PATH = $savedRootRocm }
}
if ($hasMultiCompiler) {
  foreach ($runtime in @(
    'dpcpp\ur_loader.dll',
    'dpcpp-opencl\ur_loader.dll',
    'dpcpp-opencl\ur_adapter_opencl.dll',
    'dpcpp-opencl\ur_adapter_level_zero_v2.dll',
    'dpcpp-opencl\OpenCL.dll'
  )) {
    if (-not (Test-Path (Join-Path $libsDir $runtime))) {
      throw "Windows release worker is missing required Unified Runtime file: $runtime"
    }
  }
}
# Combined (Intel+NVIDIA) build: the kawpow CUDA source-JIT reads kawpow_device.inc beside the module at
# runtime (else it falls back to the slower AOT kernel). Ship it whenever the source checkout has it;
# Intel-only packages ignore the extra file.
if (Test-Path "sycl/kawpow/device.inc") {
  if ($hasMultiCompiler) {
    # The CUDA source-JIT locates this relative to the loaded DPC++ sycl.dll.
    Copy-Item "sycl/kawpow/device.inc" (Join-Path $libsDir 'dpcpp\kawpow_device.inc')
    Copy-Item "sycl/kawpow/keccak.inc" (Join-Path $libsDir 'dpcpp\kawpow_keccak.inc')
  } else {
    Copy-Item "sycl/kawpow/device.inc" (Join-Path $libsDir 'kawpow_device.inc')
    Copy-Item "sycl/kawpow/keccak.inc" (Join-Path $libsDir 'kawpow_keccak.inc')
  }
}
$entryPaths = @("$packageDir/mom-node.exe")
if (-not $hasMultiCompiler) { $entryPaths += "$libsDir/mom.node", "$libsDir/sycl.dll" }
Copy-MominerDllClosure -PackageDir $libsDir -EntryPaths $entryPaths
if ($hasMultiCompiler) {
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
}

if (Test-Path "$packageDir/tests") {
  throw "Release package unexpectedly contains tests/."
}

Compress-Archive -Path $packageDir -DestinationPath $Archive
Write-Output $Archive
