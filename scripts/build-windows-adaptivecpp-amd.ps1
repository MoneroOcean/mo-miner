param(
  [string]$Workspace = $PSScriptRoot,
  [string]$BaseToolchain = 'C:\Tools\acpp-cuda'
)
$ErrorActionPreference = 'Stop'
Set-Location $Workspace

$buildJobs = 8
if ($env:MOM_BUILD_JOBS) {
  $parsedBuildJobs = 0
  if (-not [int]::TryParse($env:MOM_BUILD_JOBS, [ref]$parsedBuildJobs) -or $parsedBuildJobs -lt 1) {
    throw "MOM_BUILD_JOBS must be a positive integer, got '$env:MOM_BUILD_JOBS'"
  }
  $buildJobs = $parsedBuildJobs
}
$subprojectJobs = [Math]::Min(4, $buildJobs)

function Invoke-Checked {
  param([Parameter(Mandatory = $true)][scriptblock]$Command)
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $($LASTEXITCODE): $Command"
  }
}

function Replace-RequiredText([string]$Text, [string]$Old, [string]$New, [string]$Label) {
  if (-not $Text.Contains($Old)) { throw "AdaptiveCpp workaround no longer matches upstream: $Label" }
  $Text.Replace($Old, $New)
}

# The runner starts in a generic VS prompt whose LIB can point at x86 CRTs.
# Import the x64 environment explicitly before any CMake try-link or DLL link.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsRoot = if (Test-Path $vswhere) {
  & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath |
    Select-Object -First 1
} else { $null }
$vcvars = @(
  $(if ($vsRoot) { Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat' }),
  'C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $vcvars) { throw 'vcvars64.bat not found' }
$vcenv = & cmd.exe /d /s /c "call `"$vcvars`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'vcvars64.bat failed' }
foreach ($line in $vcenv) {
  if ($line -match '^([^=]+)=(.*)$') {
    [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
  }
}

$rocm = 'C:\Program Files\AMD\ROCm\7.1'
if (-not (Test-Path "$rocm\lib\amdhip64.lib")) {
  foreach ($msi in @('ROCm_SDK_Core.msi', 'ROCm_RTC_RT.msi', 'ROCm_RTC_Dev.msi')) {
    $path = Join-Path $PWD $msi
    if (-not (Test-Path $path)) { throw "Required HIP SDK installer is missing: $path" }
    $process = Start-Process msiexec.exe -ArgumentList @('/i', "`"$path`"", '/qn', '/norestart') -Wait -PassThru
    Write-Host "INSTALL $msi EXIT $($process.ExitCode)"
    if ($process.ExitCode -notin @(0, 3010)) { exit $process.ExitCode }
  }
}
$acpp = Join-Path $PWD 'acpp-toolchain'
$source = Join-Path $PWD 'AdaptiveCpp-src'
$build = Join-Path $PWD 'AdaptiveCpp-build-hip'
$env:PATH = "$rocm\bin;$acpp\bin;$env:PATH"
$env:ROCM_PATH = $rocm

Remove-Item -Recurse -Force $acpp, $source, $build -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $acpp, $source, $build | Out-Null
if (-not (Test-Path (Join-Path $BaseToolchain 'bin\acpp'))) {
  throw "AdaptiveCpp base toolchain is missing: $BaseToolchain"
}
Copy-Item (Join-Path $BaseToolchain '*') $acpp -Recurse -Force
Invoke-Checked { & "$env:SystemRoot\System32\tar.exe" -xzf (Join-Path $PWD 'AdaptiveCpp-src.tar.gz') -C $source }

# Keep imported LLVM exports relocatable if an older locally cached base still refers to Visual
# Studio Enterprise. The current pinned source build disables DIA and therefore needs no rewrite.
$llvmExports = Join-Path $acpp 'lib\cmake\llvm\LLVMExports.cmake'
$llvmExportsText = Get-Content -Raw $llvmExports
$buildToolsDia = 'C:/BuildTools/DIA SDK/lib/amd64/diaguids.lib'
if (Test-Path $buildToolsDia) {
  $llvmExportsText = $llvmExportsText.Replace(
    'C:/Program Files/Microsoft Visual Studio/2022/Enterprise/DIA SDK/lib/amd64/diaguids.lib',
    $buildToolsDia)
}
Set-Content -Path $llvmExports -Value $llvmExportsText -Encoding UTF8

Write-Host 'ROCm compiler/tool inventory:'
Get-ChildItem "$rocm\bin" -File |
  Where-Object { $_.Name -match '^(clang|hipcc|hipInfo|llvm-link|opt|lld)' } |
  Select-Object -ExpandProperty Name

$deviceLib = Get-ChildItem $rocm -Recurse -Filter ockl.bc -File | Select-Object -First 1
if (-not $deviceLib) { throw 'ROCm device library ockl.bc was not found' }
$deviceLibDir = $deviceLib.DirectoryName
Write-Host "ROCm device libraries: $deviceLibDir"
$windowsLibraries = 'kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib'

# Standalone Windows builds incorrectly require a monolithic LLVM.lib before selecting the component
# libraries shipped by the source-built base, so skip only that redundant check.
$backendCmake = Join-Path $source 'src\compiler\llvm-to-backend\CMakeLists.txt'
$backendText = (Get-Content -Raw $backendCmake).Replace("`r`n", "`n")
$backendOld = 'if(NOT ACPP_LLVM_COMPONENT)' + "`n    find_library(LLVM_LIBRARY"
$backendNew = 'if(NOT ACPP_LLVM_COMPONENT AND NOT WIN32)' + "`n    find_library(LLVM_LIBRARY"
$backendText = Replace-RequiredText $backendText $backendOld $backendNew `
  'skip redundant monolithic LLVM lookup on Windows'
Set-Content -Path $backendCmake -Value $backendText -Encoding UTF8

# The generic frontend is already linked into the source-built Clang. Build only the new JIT
# translator from src/compiler, avoiding duplicate frontend LLVM component targets from that base.
$compilerCmake = Join-Path $source 'src\compiler\CMakeLists.txt'
$compilerText = (Get-Content -Raw $compilerCmake).Replace("`r`n", "`n")
$compilerOpenOld = 'if(WITH_ACCELERATED_CPU OR WITH_SSCP_COMPILER)' + "`n  set(CBS_PLUGIN"
$compilerOpenNew = 'if(NOT ACPP_BACKEND_ONLY)' + "`nif(WITH_ACCELERATED_CPU OR WITH_SSCP_COMPILER)`n  set(CBS_PLUGIN"
$compilerText = Replace-RequiredText $compilerText $compilerOpenOld $compilerOpenNew `
  'exclude the prebuilt CBS/frontend targets from the backend-only build'
$compilerCloseOld = 'endif()' + "`n`n`nadd_subdirectory(llvm-to-backend)"
$compilerCloseNew = 'endif()' + "`nendif()`n`nadd_subdirectory(llvm-to-backend)"
$compilerText = Replace-RequiredText $compilerText $compilerCloseOld $compilerCloseNew `
  'close the backend-only frontend exclusion'
Set-Content -Path $compilerCmake -Value $compilerText -Encoding UTF8

# Build gfx12-capable AMDGPU libkernel bitcode with the HIP SDK's ROCm LLVM 21. The shared LLVM 20
# base includes AMDGPU for AdaptiveCpp's translator, but ROCm's compiler is the authoritative match
# for the Windows HIP 7.1 device-library payload.
$libkernelCmake = Join-Path $source 'src\libkernel\sscp\CMakeLists.txt'
$libkernelText = Get-Content -Raw $libkernelCmake
$libkernelOld = 'COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-link -o=${linked_output} ${output_files}'
$libkernelNew = 'COMMAND "${ROCM_PATH}/bin/llvm-link.exe" -o=${linked_output} ${output_files}'
$libkernelText = Replace-RequiredText $libkernelText $libkernelOld $libkernelNew `
  'use ROCm LLVM for AMDGPU libkernel linking'
Set-Content -Path $libkernelCmake -Value $libkernelText -Encoding UTF8

# This addon is GPU-only. Upstream currently hard-enables the OpenMP host
# backend, whose Windows discovery is unrelated and fails without libomp.
$rootCmake = Join-Path $source 'CMakeLists.txt'
$rootCmakeText = Get-Content -Raw $rootCmake
$rootCmakeText = Replace-RequiredText $rootCmakeText 'set(WITH_CPU_BACKEND true)' `
  'set(WITH_CPU_BACKEND false)' 'disable the unrelated Windows OpenMP backend build'
Set-Content -Path $rootCmake -Value $rootCmakeText -Encoding UTF8

$cmakeArgs = @(
  '-S', $source,
  '-B', $build,
  '-G', 'Ninja',
  '-DCMAKE_BUILD_TYPE=Release',
  '-DCMAKE_TRY_COMPILE_CONFIGURATION=Release',
  '-DCMAKE_C_COMPILER_WORKS=TRUE',
  '-DCMAKE_CXX_COMPILER_WORKS=TRUE',
  '-DCMAKE_SIZEOF_VOID_P=8',
  # clang-cl's forced try-compile bypass also prevents FindFilesystem from proving
  # the MSVC STL link. The real backend links it below via the explicit SDK libs.
  '-DCXX_FILESYSTEM_HAVE_FS=TRUE',
  '-DCXX_FILESYSTEM_HEADER=filesystem',
  '-DCXX_FILESYSTEM_NAMESPACE=std::filesystem',
  '-DCXX_FILESYSTEM_NO_LINK_NEEDED=TRUE',
  '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL',
  "-DCMAKE_C_STANDARD_LIBRARIES=$windowsLibraries",
  "-DCMAKE_CXX_STANDARD_LIBRARIES=$windowsLibraries",
  "-DCMAKE_INSTALL_PREFIX=$acpp",
  "-DCMAKE_C_COMPILER=$acpp\bin\clang-cl.exe",
  "-DCMAKE_CXX_COMPILER=$acpp\bin\clang-cl.exe",
  "-DCMAKE_PREFIX_PATH=$acpp;$rocm",
  "-DLLVM_DIR=$acpp\lib\cmake\llvm",
  "-DClang_DIR=$acpp\lib\cmake\clang",
  '-DACPP_FILESYSTEM_SEARCH_OPTIONS=Final',
  "-DROCM_PATH=$rocm",
  "-DHIP_DIR=$rocm\lib\cmake\hip",
  "-DROCM_DEVICE_LIBS_PATH=$deviceLibDir",
  "-DCLANG_EXECUTABLE_PATH=$rocm\bin\clang++.exe",
  "-DCLANG_INCLUDE_PATH=$rocm\lib\clang\21",
  "-DACPP_OPT_PATH=$rocm\bin\opt.exe",
  "-DACPP_LLC_PATH=$acpp\bin\llc.exe",
  "-DACPP_LLD_PATH=$acpp\bin\lld-link.exe",
  '-DACPP_COMPILER_FEATURE_PROFILE=full',
  '-DACPP_BACKEND_ONLY=ON',
  '-DACPP_EXPERIMENTAL_LLVM=ON',
  '-DACPP_HOST_FORCE_MCPU_TARGET=x86-64',
  '-DWITH_ROCM_BACKEND=ON',
  '-DWITH_CUDA_BACKEND=OFF',
  '-DWITH_OPENCL_BACKEND=OFF',
  '-DWITH_LEVEL_ZERO_BACKEND=OFF',
  '-DWITH_VULKAN_BACKEND=OFF',
  "-DACPP_SUBPROJECT_PARALLEL_JOBS=$subprojectJobs"
)
Invoke-Checked { & cmake.exe @cmakeArgs }
Invoke-Checked { & cmake.exe --build $build --target rt-backend-hip libkernel-sscp-amdgpu-amdhsa --parallel $buildJobs }

$hipBackend = Get-ChildItem $build -Recurse -Filter rt-backend-hip.dll -File | Select-Object -First 1
$amdBitcode = Get-ChildItem $build -Recurse -Filter libkernel-sscp-amdgpu-amdhsa-full.bc -File | Select-Object -First 1
if (-not $hipBackend) { throw 'rt-backend-hip.dll was not produced' }
if (-not $amdBitcode) { throw 'AMDGPU libkernel bitcode was not produced' }
New-Item -ItemType Directory -Force "$acpp\bin\hipSYCL", "$acpp\lib\hipSYCL\bitcode" | Out-Null
Copy-Item -Force $hipBackend.FullName "$acpp\bin\hipSYCL\rt-backend-hip.dll"
Copy-Item -Force $amdBitcode.FullName "$acpp\lib\hipSYCL\bitcode\libkernel-sscp-amdgpu-amdhsa-full.bc"
Write-Host "Installed HIP backend: $($hipBackend.FullName)"
Write-Host "Installed AMDGPU bitcode: $($amdBitcode.FullName)"

$env:ACPP_VISIBILITY_MASK = 'hip'
Remove-Item -Recurse -Force (Join-Path $env:LOCALAPPDATA 'acpp') -ErrorAction SilentlyContinue
Write-Host 'AdaptiveCpp device inventory:'
Invoke-Checked { & "$acpp\bin\acpp-info.exe" }
Write-Host "AdaptiveCpp Windows HIP toolchain ready at $acpp"
