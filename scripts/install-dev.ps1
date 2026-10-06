param(
  [Alias('Components')][string[]]$Component = @('all'),
  [int]$Jobs = $(if ($env:MOM_BUILD_JOBS) { [int]$env:MOM_BUILD_JOBS } else { [Environment]::ProcessorCount }),
  [string]$DpcppDir = 'C:\Tools\dpcpp',
  [string]$AcppCudaDir = 'C:\Tools\acpp-cuda',
  [string]$AcppHipDir = 'C:\Tools\acpp-amd',
  [string]$HipDir = 'C:\Program Files\AMD\ROCm\7.1',
  [string]$CudaDir = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6',
  [string]$DpcppSha256 = '8e5e9de06ed46c5e28aac4c574811a68f0a85dbb1cb65508187b0b8cf2e573cb',
  [string]$Workspace = 'C:\mom-dev-bootstrap',
  [switch]$ValidateOnly,
  [switch]$KeepWorkspace
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($PSVersionTable.PSVersion.Major -ge 7) { $PSNativeCommandUseErrorActionPreference = $true }
if ($Jobs -lt 1) { throw '-Jobs must be a positive integer' }
$env:MOM_BUILD_JOBS = [string]$Jobs
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'windows-install-helpers.ps1')
. (Join-Path $PSScriptRoot "install-cutlass.ps1")

function Resolve-FromRepo([string]$Path) {
  if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
  [IO.Path]::GetFullPath((Join-Path $repo $Path))
}

# Build helpers deliberately change into their private workspaces. Resolve every caller-supplied
# location first so a relative path continues to mean "relative to the repository", as it does for
# the rest of the development and VM tooling.
$DpcppDir = Resolve-FromRepo $DpcppDir
$AcppCudaDir = Resolve-FromRepo $AcppCudaDir
$AcppHipDir = Resolve-FromRepo $AcppHipDir
$HipDir = Resolve-FromRepo $HipDir
$CudaDir = Resolve-FromRepo $CudaDir
$Workspace = Resolve-FromRepo $Workspace
$pathTrimChars = [char[]]@('\', '/')
$pathSeparator = [IO.Path]::DirectorySeparatorChar
$Workspace = $Workspace.TrimEnd($pathTrimChars)
$workspaceLeaf = [IO.Path]::GetFileName($Workspace)
if ($workspaceLeaf -notmatch '^mom-dev-[A-Za-z0-9_.-]+$') {
  throw '-Workspace must name a dedicated mom-dev-* directory'
}
$workspaceMarker = Join-Path $Workspace '.mom-dev-workspace'
Assert-NoReparseAncestor $Workspace 'Development workspace'

function Initialize-Workspace {
  $workspaceItem = Get-Item -LiteralPath $Workspace -Force -ErrorAction SilentlyContinue
  if ($workspaceItem) {
    if (-not $workspaceItem.PSIsContainer) { throw '-Workspace must be an owned directory' }
    if ($workspaceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
      throw '-Workspace must not be a reparse point'
    }
    if (-not (Test-MomMarker $workspaceMarker 'mom development workspace')) {
      throw '-Workspace exists without the .mom-dev-workspace marker'
    }
    return
  }
  [IO.Directory]::CreateDirectory($Workspace) | Out-Null
  Set-Content -LiteralPath $workspaceMarker -Value 'mom development workspace' -NoNewline
}

# Versions and destinations are shared by local VM provisioning and hosted CI. GPU drivers are the
# only deliberate external prerequisite; CUDA/HIP below are compiler SDK payloads, not display drivers.
$nodeVersion = '24.15.0'
$nodeSha256 = 'feffb8e5cb5ac47f793666636d496ef3e975be82c84c4da5d20e6aa8fa4eb806'
$momToolsRoot = 'C:\Tools\mom'
$pythonVersion = '3.12.10'
$pythonSha256 = '4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3'
$pythonRoot = Join-Path $momToolsRoot "python-$pythonVersion"
$gitVersion = '2.50.1'
$gitSha256 = '6f672aebe9e488a246efd6875f9197dbc0d9a40100e218acc3877cba2b206c45'
$gitRoot = Join-Path $momToolsRoot "git-$gitVersion"
$sevenZipSha256 = 'ec6af1ea0367d16dde6639a89a080a524cebc4d4bedfe00ed0cac4b865a918d8'
$adaptiveCppCommit = 'da2463e45aa90aa36306c45abcfc05b87de51bc6'
$hipSdkUrl = 'https://download.amd.com/developer/eula/rocm-hub/AMD-Software-PRO-Edition-26.Q1-Win11-For-HIP.exe'
$hipSdkSha256 = 'f9e1fd7ae6004ce448ef39dcac2c3b45fed741f2d83210259bfda61b86f78f84'
$cudaUrl = 'https://developer.download.nvidia.com/compute/cuda/12.6.0/network_installers/cuda_12.6.0_windows_network.exe'
$cudaNvrtcUrl = 'https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-13.1.115-archive.zip'
$cudaNvrtcSha256 = '4ddd5a1e34fd62bb41e78c0725edfbf5609f4ecedd7fe118eddf2148b097fc91'
$oneApiUrl = 'https://registrationcenter-download.intel.com/akdlm/IRC_NAS/bae85ab1-cfcd-4251-8d42-a0c27949ea33/intel-oneapi-toolkit-2026.0.0.193.exe'
$oneApiPackage = 'intel-oneapi-toolkit'
$oneApiComponents = 'intel.oneapi.win.cpp-dpcpp-common'
$openClCpuUrl = 'https://registrationcenter-download.intel.com/akdlm/IRC_NAS/ad824c04-01c8-4ae5-b5e8-164a04f67609/w_opencl_runtime_p_2025.3.1.762.exe'

$requested = [System.Collections.Generic.List[string]]::new()
foreach ($entry in $Component) {
  foreach ($name in ($entry -split ',')) {
    if ($name.Trim()) { $requested.Add($name.Trim().ToLowerInvariant()) }
  }
}
if (-not $requested.Count) { $requested.Add('all') }
$valid = @('base','node','oneapi','opencl-cpu','cuda','hip','dpcpp','acpp-cuda','acpp-hip')
$defaultComponents = @('base','node','oneapi','opencl-cpu','cuda','hip','dpcpp','acpp-cuda','acpp-hip')
$components = [System.Collections.Generic.List[string]]::new()
foreach ($name in $requested) {
  if ($name -eq 'all') { foreach ($item in $defaultComponents) { if (-not $components.Contains($item)) { $components.Add($item) } } }
  elseif ($valid -contains $name) { if (-not $components.Contains($name)) { $components.Add($name) } }
  else { throw "Unknown development component '$name'. Valid: all, $($valid -join ', ')" }
}

if (($components -contains 'cuda') -or ($components -contains 'acpp-cuda') -or
    ($components -contains 'acpp-hip')) {
  Assert-NoReparseAncestor $CudaDir 'CUDA SDK destination'
}
if ($components -contains 'dpcpp') {
  Assert-NoReparseAncestor $DpcppDir 'DPC++ destination'
}
if (($components -contains 'hip') -or ($components -contains 'acpp-hip')) {
  Assert-NoReparseAncestor $HipDir 'HIP SDK destination'
}

$acppDestinations = @()
if (($components -contains 'acpp-cuda') -or ($components -contains 'acpp-hip')) {
  $acppDestinations += $AcppCudaDir
}
if ($components -contains 'acpp-hip') { $acppDestinations += $AcppHipDir }
foreach ($path in ($acppDestinations | Select-Object -Unique)) {
  $trimmed = $path.TrimEnd($pathTrimChars)
  $root = [IO.Path]::GetPathRoot($path).TrimEnd($pathTrimChars)
  if ([string]::Equals($trimmed, $root, [StringComparison]::OrdinalIgnoreCase) -or
      [IO.Path]::GetFileName($trimmed) -notmatch '^acpp-[A-Za-z0-9_.-]+$') {
    throw 'AdaptiveCpp destinations must be dedicated acpp-* directories, not filesystem roots'
  }
  $item = Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
  if ($item -and -not $item.PSIsContainer) {
    throw "AdaptiveCpp destination must be a real directory: $path"
  }
  Assert-NoReparseAncestor $path 'AdaptiveCpp destination'
  $workspacePrefix = $Workspace + $pathSeparator
  $destinationPrefix = $trimmed + $pathSeparator
  if ($trimmed.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase) -or
      $Workspace.StartsWith($destinationPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "AdaptiveCpp destination must be separate from -Workspace: $path"
  }
  if ($item) {
    if (-not (Test-MomMarker (Join-Path $path '.mom-acpp-toolchain') `
        'mom AdaptiveCpp toolchain')) {
      $contents = Get-ChildItem -LiteralPath $path -Force | Select-Object -First 1
      if ($contents) {
        throw "AdaptiveCpp destination exists without its ownership marker: $path"
      }
    }
  }
}
$uniqueAcppDestinations = @($acppDestinations | Select-Object -Unique)
if ($uniqueAcppDestinations.Count -eq 2 -and
    -not [string]::Equals($uniqueAcppDestinations[0], $uniqueAcppDestinations[1],
      [StringComparison]::OrdinalIgnoreCase)) {
  $first = $uniqueAcppDestinations[0].TrimEnd($pathTrimChars)
  $second = $uniqueAcppDestinations[1].TrimEnd($pathTrimChars)
  if ($first.StartsWith($second + $pathSeparator, [StringComparison]::OrdinalIgnoreCase) -or
      $second.StartsWith($first + $pathSeparator, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'AdaptiveCpp CUDA and HIP destinations must be equal or separate directories'
  }
}

function Test-Administrator {
  $principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
  $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}
function Require-Administrator {
  if (-not (Test-Administrator)) { throw 'Run scripts\install-dev.bat from an elevated Administrator prompt.' }
}
function Add-MachinePath([string]$Entry) {
  if (-not (Test-Path $Entry)) { return }
  $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
  $parts = @($machine -split ';' | Where-Object { $_ })
  if ($parts -notcontains $Entry) {
    [Environment]::SetEnvironmentVariable('Path', (($parts + $Entry) -join ';'), 'Machine')
  }
  if (($env:Path -split ';') -notcontains $Entry) { $env:Path = "$Entry;$env:Path" }
  # Each GitHub Actions step starts a fresh process with the runner's original PATH. In particular,
  # free-windows-disk.ps1 removes the hosted Node tool cache before this installer supplies the
  # pinned replacement under Program Files. Persist every installed path for subsequent build and
  # package steps instead of relying on the machine environment to be reloaded implicitly.
  if ($env:GITHUB_PATH) { $Entry | Out-File $env:GITHUB_PATH -Append -Encoding utf8 }
}
function Export-DevelopmentEnvironment([string]$Name, [string]$Value) {
  [Environment]::SetEnvironmentVariable($Name, $Value, 'Machine')
  Set-Item "Env:$Name" $Value
  if ($env:GITHUB_ENV) { "$Name=$Value" | Out-File $env:GITHUB_ENV -Append -Encoding utf8 }
}
function Install-Msi([string]$Path, [string[]]$Properties = @()) {
  Require-Administrator
  # Callers authenticate direct downloads or the signed/hash-pinned container they extract from.
  $process = Start-Process msiexec.exe -ArgumentList (@('/i',"`"$Path`"",'/qn','/norestart') + $Properties) -Wait -PassThru
  if ($process.ExitCode -notin @(0,3010,1641)) { throw "MSI $Path failed with exit code $($process.ExitCode)" }
}
function Test-VsBuildTools {
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vswhere) {
    $root = & $vswhere -latest -products * `
      -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
      -requires Microsoft.VisualStudio.Component.Windows11SDK.26100 `
      -requires Microsoft.VisualStudio.Component.VC.CMake.Project `
      -property installationPath | Select-Object -First 1
    if ($root -and (Test-Path (Join-Path $root 'VC\Auxiliary\Build\vcvars64.bat'))) { return $true }
  }
  $false
}
function Test-Python {
  $python = Join-Path $pythonRoot 'python.exe'
  if (-not (Test-Path $python)) { return $false }
  $reported = (& $python --version 2>&1 | Select-Object -First 1).ToString().Trim()
  ($LASTEXITCODE -eq 0) -and ($reported -eq "Python $pythonVersion")
}
function Test-Git {
  $git = Join-Path $gitRoot 'cmd\git.exe'
  if (-not (Test-Path $git)) { return $false }
  & $git --version >$null 2>&1
  $LASTEXITCODE -eq 0
}
function Install-7Zip {
  if (-not (Get-Command 7z.exe -ErrorAction SilentlyContinue)) {
    $sevenZip = Join-Path $Workspace '7zip.msi'
    Download 'https://www.7-zip.org/a/7z2409-x64.msi' $sevenZip
    Assert-Sha256 $sevenZip $sevenZipSha256
    Install-Msi $sevenZip
  }
  Add-MachinePath 'C:\Program Files\7-Zip'
}
function Install-Base {
  Require-Administrator
  [IO.Directory]::CreateDirectory($momToolsRoot) | Out-Null
  if (-not (Test-VsBuildTools)) {
    $installer = Join-Path $Workspace 'vs_BuildTools.exe'
    Download 'https://aka.ms/vs/17/release/vs_BuildTools.exe' $installer
    Assert-Authenticode $installer
    $arguments = @(
      '--quiet','--wait','--norestart','--nocache','--installPath','C:\BuildTools',
      '--add','Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
      '--add','Microsoft.VisualStudio.Component.Windows11SDK.26100',
      '--add','Microsoft.VisualStudio.Component.VC.CMake.Project'
    )
    $process = Start-Process $installer -ArgumentList $arguments -Wait -PassThru
    if ($process.ExitCode -notin @(0,3010)) { throw "Visual Studio Build Tools failed: $($process.ExitCode)" }
  }

  # Do not trust Get-Command here: a fresh Windows installation exposes the Microsoft Store
  # WindowsApps `python.exe` alias, which exists on PATH but exits with 9009. Use the pinned,
  # registry-independent embeddable distribution so hosted-runner disk cleanup cannot leave the
  # python.org MSI registered with its tool-cache payload already removed.
  if (-not (Test-Python)) {
    $python = Join-Path $Workspace "python-$pythonVersion-embed-amd64.zip"
    Download "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-embed-amd64.zip" $python
    Assert-Sha256 $python $pythonSha256
    Remove-Item $pythonRoot -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $pythonRoot | Out-Null
    Invoke-Checked "$env:SystemRoot\System32\tar.exe" @('-xf',$python,'-C',$pythonRoot)
    # The embeddable archive enables isolated `_pth` mode by default. LLVM/OpenMP build helpers
    # import sibling modules (for example message-converter.py -> libomputils.py), which requires
    # normal script-directory resolution just like a regular python.org installation.
    Remove-Item (Join-Path $pythonRoot 'python312._pth') -Force
  }
  Add-MachinePath $pythonRoot
  Add-MachinePath (Join-Path $pythonRoot 'Scripts')

  if (-not (Test-Git)) {
    $mingit = Join-Path $Workspace 'mingit.zip'
    Download "https://github.com/git-for-windows/git/releases/download/v$gitVersion.windows.1/MinGit-$gitVersion-64-bit.zip" $mingit
    Assert-Sha256 $mingit $gitSha256
    Remove-Item $gitRoot -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $gitRoot | Out-Null
    Invoke-Checked "$env:SystemRoot\System32\tar.exe" @('-xf',$mingit,'-C',$gitRoot)
  }
  Add-MachinePath (Join-Path $gitRoot 'cmd')

  Install-7Zip
  Add-MachinePath 'C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
  Add-MachinePath 'C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'
}
function Install-Node {
  if (-not (Test-Node)) {
    $msi = Join-Path $Workspace "node-v$nodeVersion-x64.msi"
    Download "https://nodejs.org/dist/v$nodeVersion/node-v$nodeVersion-x64.msi" $msi
    Assert-Sha256 $msi $nodeSha256
    Install-Msi $msi
  }
  Add-MachinePath 'C:\Program Files\nodejs'
  $npmPrefix = ((& npm.cmd prefix -g) | Select-Object -Last 1).Trim()
  if ($LASTEXITCODE -ne 0 -or -not $npmPrefix) { throw 'Unable to resolve the global npm prefix' }
  Add-MachinePath $npmPrefix
  if (-not (Get-Command node-gyp.cmd -ErrorAction SilentlyContinue)) {
    Invoke-Checked 'npm.cmd' @('install','-g','node-gyp@12.2.0')
  }
}
function Test-Node {
  # MSI installers run later in an `all` setup may refresh the machine PATH
  # without updating this PowerShell process. Validate the pinned installation
  # directly so a successful Node setup is not mistaken for a missing one.
  $node = 'C:\Program Files\nodejs\node.exe'
  $npm = 'C:\Program Files\nodejs\npm.cmd'
  if (-not ((Test-Path $node) -and (Test-Path $npm))) { return $false }
  $npmPrefix = ((& $npm prefix -g) | Select-Object -Last 1).Trim()
  if ($LASTEXITCODE -ne 0 -or -not $npmPrefix) { return $false }
  ((& $node --version) -eq "v$nodeVersion") -and
    (Test-Path (Join-Path $npmPrefix 'node_modules\node-gyp\bin\node-gyp.js'))
}
function Test-OneApi {
  $compiler = 'C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icx.exe'
  (Test-Path $compiler) -and ((Get-Item $compiler).VersionInfo.ProductVersion -like '2026.0*')
}
function Get-OpenClCpuRuntime {
  $sharedRoot = 'C:\Program Files (x86)\Common Files\Intel\Shared Libraries'
  $registry = 'HKLM:\SOFTWARE\Khronos\OpenCL\Vendors'
  if (Test-Path $registry) {
    $properties = Get-ItemProperty $registry
    foreach ($property in $properties.PSObject.Properties) {
      # The oneAPI compiler also registers bin\intelocl64.dll, but that is Intel's GPU ICD and
      # does not provide a CPU device. Only accept the standalone CPU runtime installed below.
      if ($property.Name.StartsWith($sharedRoot, [StringComparison]::OrdinalIgnoreCase) -and
          $property.Name -like '*intelocl64.dll' -and (Test-Path $property.Name)) {
        return $property.Name
      }
    }
  }
  Get-ChildItem $sharedRoot `
    -Filter intelocl64.dll -File -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
}
function Install-OpenClCpu {
  if (Get-OpenClCpuRuntime) { return }
  Require-Administrator
  # This runtime needs an extractor, not Visual Studio, Python, Git, or the compiler SDKs.
  Install-7Zip
  $installer = Join-Path $Workspace 'opencl-cpu-runtime.exe'
  Download $openClCpuUrl $installer
  Assert-Authenticode $installer
  # Intel wraps one ordinary MSI in StubWebImage.exe. Its documented silent wrapper can remain
  # alive indefinitely after the MSI transaction on headless Windows/CI. Extract and install that
  # signed payload directly so setup has deterministic completion and exit-code handling.
  $extractDir = Join-Path $Workspace 'opencl-cpu-runtime'
  Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force $extractDir | Out-Null
  Invoke-Checked '7z.exe' @('x','-y','-bso0',"-o$extractDir",$installer)
  $msi = Get-ChildItem $extractDir -Filter '*.msi' -File | Select-Object -First 1
  if (-not $msi) { throw 'Intel CPU OpenCL wrapper did not contain an MSI payload.' }
  Install-Msi $msi.FullName
  if (-not (Get-OpenClCpuRuntime)) { throw 'Intel CPU OpenCL runtime did not register its ICD.' }
}
function Install-OneApi {
  if (-not (Test-OneApi)) {
    Require-Administrator
    Invoke-Checked (Join-Path $repo '.github\workflows\scripts\install-oneapi.bat') @(
      $oneApiUrl,$oneApiComponents,$oneApiPackage)
  }
  Export-DevelopmentEnvironment ONEAPI_ROOT 'C:\Program Files (x86)\Intel\oneAPI'
  Install-OpenClCpu
}
function Test-CudaNvrtc {
  (Test-Path (Join-Path $CudaDir 'bin\nvrtc64_130_0.dll')) -and
    (Test-Path (Join-Path $CudaDir 'bin\nvrtc-builtins64_131.dll'))
}
function Install-CudaNvrtc {
  if (Test-CudaNvrtc) { return }
  Require-Administrator
  $archive = Join-Path $Workspace 'cuda-nvrtc-13.1.115.zip'
  Download $cudaNvrtcUrl $archive
  Assert-Sha256 $archive $cudaNvrtcSha256
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $zip = [System.IO.Compression.ZipFile]::OpenRead($archive)
  try {
    foreach ($name in @('nvrtc64_130_0.dll','nvrtc-builtins64_131.dll')) {
      $entry = $zip.Entries | Where-Object { $_.Name -eq $name } | Select-Object -First 1
      if (-not $entry) { throw "CUDA NVRTC archive does not contain $name" }
      [System.IO.Compression.ZipFileExtensions]::ExtractToFile(
        $entry, (Join-Path $CudaDir "bin\$name"), $true)
    }
  } finally {
    $zip.Dispose()
  }
  if (-not (Test-CudaNvrtc)) { throw 'CUDA 13 NVRTC runtime install failed validation' }
}
function Install-Cuda {
  if (-not (Test-MomCudaCompilerSdk $CudaDir)) {
    Require-Administrator
    $installer = Join-Path $Workspace 'cuda-network.exe'
    Download $cudaUrl $installer
    Assert-Authenticode $installer
    $process = Start-Process $installer -ArgumentList @(
      '-s','nvcc_12.6','cudart_12.6','nvrtc_12.6','nvrtc_dev_12.6','nvml_dev_12.6'
    ) -Wait -PassThru
    if ($process.ExitCode -notin @(0,3010)) { throw "CUDA toolkit install failed: $($process.ExitCode)" }
  }
  Install-CudaNvrtc
  if (-not (Test-MomCccl $CudaDir)) { Require-Administrator; Install-MomCccl $CudaDir }
  if (-not (Test-MomCutlass)) { Require-Administrator; Install-MomCutlass }
  Export-DevelopmentEnvironment CUDA_PATH $CudaDir
  Add-MachinePath (Join-Path $CudaDir 'bin')
}
function Test-Dpcpp {
  $marker = Join-Path $DpcppDir '.mom-toolchain-sha256'
  (Test-MomDpcppToolchain $DpcppDir) -and
    (Test-MomMarker $marker $DpcppSha256.ToLowerInvariant())
}
function Install-Dpcpp {
  if (-not (Test-Dpcpp)) {
    & (Join-Path $repo '.github\workflows\scripts\restore-toolchain-win.ps1') `
      -Dest $DpcppDir -ExpectedSha256 $DpcppSha256
  }
  Export-DevelopmentEnvironment MOM_DPCPP_DIR $DpcppDir
}
function Install-AcppCuda {
  if (Test-Acpp $AcppCudaDir cuda) { return }
  Install-Cuda
  & (Join-Path $repo 'scripts\build-windows-adaptivecpp-base.ps1') `
    -InstallDir $AcppCudaDir -Workspace $Workspace -Jobs $Jobs
}
function Install-Hip {
  if (-not (Test-MomHipSdk $HipDir)) {
    Require-Administrator
    $sdkExe = Join-Path $Workspace 'hip-sdk.exe'
    Download $hipSdkUrl $sdkExe
    Assert-Sha256 $sdkExe $hipSdkSha256
    $sevenZip = (Get-Command 7z.exe -ErrorAction Stop | Select-Object -First 1).Source
    $sdkItems = @(
      'Packages/Apps/ROCmSDKPackages/SDKCore/ROCm_SDK_Core.msi',
      'Packages/Apps/ROCmSDKPackages/RTCDevelopment/ROCm_RTC_Dev.msi',
      'Packages/Apps/ROCmSDKPackages/RTCRuntime/ROCm_RTC_RT.msi'
    )
    # The SDK payload is one solid 7z stream. Extract all three members in one pass; invoking 7-Zip
    # once per MSI decompresses the same 1.6 GiB archive three times on every cold CI/VM setup.
    Invoke-Checked $sevenZip (@('e','-y',"-o$Workspace",$sdkExe) + $sdkItems)
    foreach ($msi in @('ROCm_SDK_Core.msi','ROCm_RTC_Dev.msi','ROCm_RTC_RT.msi')) {
      Install-Msi (Join-Path $Workspace $msi)
    }
  }
  if (-not (Test-MomHipSdk $HipDir)) { throw "HIP SDK install failed validation: $HipDir" }
  Export-DevelopmentEnvironment HIP_PATH $HipDir
  Export-DevelopmentEnvironment ROCM_PATH $HipDir
}
function Install-AcppHip {
  if (Test-Acpp $AcppHipDir hip) { return }
  Install-AcppCuda
  Install-Hip
  $sourceRepo = Join-Path $Workspace 'AdaptiveCpp-repo'
  Remove-Item $sourceRepo -Recurse -Force -ErrorAction SilentlyContinue
  Invoke-Checked 'git.exe' @('clone','--quiet','--filter=blob:none','https://github.com/AdaptiveCpp/AdaptiveCpp.git',$sourceRepo)
  Invoke-Checked 'git.exe' @('-C',$sourceRepo,'checkout',$adaptiveCppCommit)
  Invoke-Checked 'git.exe' @('-C',$sourceRepo,'archive','--format=tar.gz','-o',
    (Join-Path $Workspace 'AdaptiveCpp-src.tar.gz'),$adaptiveCppCommit)
  & (Join-Path $repo 'scripts\build-windows-adaptivecpp-amd.ps1') `
    -Workspace $Workspace -BaseToolchain $AcppCudaDir -HipPath $HipDir
  Remove-Item $AcppHipDir -Recurse -Force -ErrorAction SilentlyContinue
  Move-Item (Join-Path $Workspace 'acpp-toolchain') $AcppHipDir
}

function Test-AcppLlvmTargets([string]$Path) {
  $llc = Join-Path $Path 'bin\llc.exe'
  if (-not (Test-Path $llc)) { return $false }
  try {
    $targets = (& $llc --version 2>$null) -join "`n"
    if ($LASTEXITCODE -ne 0) { return $false }
    foreach ($target in @('amdgcn', 'nvptx', 'x86')) {
      if ($targets -notmatch "(?m)^\s*$target\s+-") { return $false }
    }
    return $true
  } catch {
    return $false
  }
}

function Test-Acpp([string]$Path, [string]$Backend) {
  $devicePayload = switch ($Backend) {
    cuda {
      (Test-Path (Join-Path $Path 'bin\hipSYCL\llvm-to-backend\llvm-to-ptx-tool.exe')) -and
        (Test-Path (Join-Path $Path 'lib\hipSYCL\bitcode\libkernel-sscp-ptx-full.bc'))
    }
    hip {
      Test-Path (Join-Path $Path 'lib\hipSYCL\bitcode\libkernel-sscp-amdgpu-amdhsa-full.bc')
    }
    default { $false }
  }
  $devicePayload -and
    (Test-MomMarker (Join-Path $Path '.mom-acpp-toolchain') `
      'mom AdaptiveCpp toolchain') -and
    (Test-Path (Join-Path $Path 'bin\acpp')) -and
    (Test-Path (Join-Path $Path 'bin\clang.exe') -PathType Leaf) -and
    (Test-Path (Join-Path $Path 'bin\acpp-common.dll') -PathType Leaf) -and
    (Test-Path (Join-Path $Path 'bin\acpp-rt.dll') -PathType Leaf) -and
    (Test-Path (Join-Path $Path 'bin\opt.exe') -PathType Leaf) -and
    (Test-Path (Join-Path $Path 'bin\llc.exe') -PathType Leaf) -and
    (Test-Path (Join-Path $Path 'bin\lld-link.exe') -PathType Leaf) -and
    (Test-Path (Join-Path $Path 'lib\hipSYCL\bitcode\libkernel-sscp-host-full.bc') -PathType Leaf) -and
    (Test-Path (Join-Path $Path 'bin\libomp.dll')) -and
    (Test-Path (Join-Path $Path 'bin\hipSYCL\rt-backend-omp.dll')) -and
    (Test-Path (Join-Path $Path "bin\hipSYCL\rt-backend-$Backend.dll")) -and
    (Test-AcppLlvmTargets $Path)
}

function Test-Component([string]$Name) {
  switch ($Name) {
    base { return (Test-VsBuildTools) -and (Test-Python) -and (Test-Git) -and
      [bool](Get-Command cmake.exe -ErrorAction SilentlyContinue) -and
      [bool](Get-Command ninja.exe -ErrorAction SilentlyContinue) -and
      [bool](Get-Command 7z.exe -ErrorAction SilentlyContinue) }
    node { return Test-Node }
    oneapi { return (Test-OneApi) -and [bool](Get-OpenClCpuRuntime) }
    'opencl-cpu' { return [bool](Get-OpenClCpuRuntime) }
    cuda {
      return (Test-MomCudaCompilerSdk $CudaDir) -and
        (Test-CudaNvrtc) -and
        (Test-MomCccl $CudaDir) -and
        (Test-MomCutlass)
    }
    hip { return Test-MomHipSdk $HipDir }
    dpcpp { return Test-Dpcpp }
    'acpp-cuda' { return Test-Acpp $AcppCudaDir cuda }
    'acpp-hip' { return Test-Acpp $AcppHipDir hip }
  }
}

if ($ValidateOnly) {
  $missing = @($components | Where-Object { -not (Test-Component $_) })
  foreach ($name in $components) { Write-Host "dev component $name`: $(if (Test-Component $name) {'ok'} else {'missing'})" }
  if ($missing.Count) { throw "Missing development components: $($missing -join ', ')" }
  exit 0
}

Require-Administrator
Initialize-Workspace

if ($components -contains 'base') {
  Install-Base
}
if ($components -contains 'node') {
  Install-Base
  Install-Node
}
if ($components -contains 'oneapi') {
  Install-Base
  Install-OneApi
}
if ($components -contains 'opencl-cpu') {
  Install-OpenClCpu
}
if ($components -contains 'cuda') {
  Install-Base
  Install-Cuda
}
if ($components -contains 'hip') {
  Install-Base
  Install-Hip
}
if ($components -contains 'dpcpp') {
  Install-Base
  Install-Dpcpp
}
if ($components -contains 'acpp-cuda') {
  Install-Base
  Install-Cuda
  Install-AcppCuda
}
if ($components -contains 'acpp-hip') {
  Install-Base
  Install-Hip
  Install-AcppHip
}

foreach ($name in $components) {
  if (-not (Test-Component $name)) { throw "Installed component failed validation: $name" }
  Write-Host "dev component ready: $name"
}
[IO.Directory]::CreateDirectory('C:\Tools') | Out-Null
@(
  'mom Windows multi-compiler development environment',
  "oneAPI=C:\Program Files (x86)\Intel\oneAPI",
  "DPCPP=$DpcppDir",
  "AdaptiveCpp-CUDA=$AcppCudaDir",
  "AdaptiveCpp-HIP=$AcppHipDir",
  "CUDA=$CudaDir",
  "HIP=$HipDir"
) | Set-Content 'C:\Tools\mom-toolchains.txt' -Encoding UTF8
if (-not $KeepWorkspace) {
  # Build helpers may change the process working directory into their workspace.
  # Leave it before removing the owned tree; Windows cannot delete the current directory.
  Set-Location -LiteralPath $repo
  Assert-NoReparseAncestor $Workspace 'Development workspace'
  if (-not (Test-MomMarker $workspaceMarker 'mom development workspace')) {
    throw 'Development workspace ownership changed before cleanup'
  }
  Remove-Item -LiteralPath $Workspace -Recurse -Force
  if (Test-Path -LiteralPath $Workspace) {
    throw "Development workspace cleanup failed: $Workspace"
  }
}
