# Windows GPU drivers are prerequisites: Intel Level Zero, NVIDIA CUDA, AMD HIP, and other vendors'
# OpenCL ICDs come from their respective display drivers. The release bundles Node.js and isolated
# oneAPI, DPC++, AdaptiveCpp CUDA/HIP runtimes. On NVIDIA, this script automatically installs missing
# CUDA/C++ pieces used by the ProgPoW and PearlHash source-JIT paths.

param(
  [string]$CudaToolkitUrl = "",
  # Force provisioning even without an NVIDIA device (for CI/build hosts).
  [switch]$InstallCudaToolkit,
  [switch]$SkipCudaToolkit,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if ($PSVersionTable.PSVersion.Major -ge 7) {
  $PSNativeCommandUseErrorActionPreference = $true
}

function Test-Administrator {
  $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
  return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Assert-Authenticode([string]$Path) {
  $signature = Get-AuthenticodeSignature -LiteralPath $Path
  if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
    throw "Authenticode verification failed for $Path`: $($signature.Status)"
  }
}

$tempRoot = if ($env:TEMP) { $env:TEMP } elseif ($env:TMPDIR) { $env:TMPDIR } else { "/tmp" }
$fallbackCudaToolkitUrl = "https://developer.download.nvidia.com/compute/cuda/12.6.0/network_installers/cuda_12.6.0_windows_network.exe"
$cudaToolkitRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$cudaNvrtcUrl = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-13.1.115-archive.zip"
$cudaNvrtcSha256 = "4ddd5a1e34fd62bb41e78c0725edfbf5609f4ecedd7fe118eddf2148b097fc91"
$openClCpuUrl = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/ad824c04-01c8-4ae5-b5e8-164a04f67609/w_opencl_runtime_p_2025.3.1.762.exe"
$sevenZipUrl = "https://www.7-zip.org/a/7z2409-x64.msi"
$sevenZipSha256 = "ec6af1ea0367d16dde6639a89a080a524cebc4d4bedfe00ed0cac4b865a918d8"
$fallbackVsBuildToolsUrl = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
$cudaToolkitPackages = @(
  "nvcc_12.6", "cudart_12.6", "nvrtc_12.6", "nvrtc_dev_12.6"
)

function Get-NvidiaPnpDevice {
  Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue |
    Where-Object { $_.PNPDeviceID -like "PCI\VEN_10DE*" } |
    Select-Object -First 1
}

function Get-AmdPnpDevice {
  Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue |
    Where-Object { $_.PNPDeviceID -like "PCI\VEN_1002*" } |
    Select-Object -First 1
}

function Install-MsiPackage([string]$Path) {
  if (-not (Test-Administrator)) {
    throw "Run install.bat from an elevated Administrator command prompt."
  }
  $install = Start-Process msiexec.exe `
    -ArgumentList @('/i', "`"$Path`"", '/qn', '/norestart') -Wait -PassThru
  if ($install.ExitCode -notin @(0, 3010, 1641)) {
    throw "MSI install failed with exit code $($install.ExitCode): $Path"
  }
}

function Get-OpenClCpuRuntime {
  $sharedRoot = "C:\Program Files (x86)\Common Files\Intel\Shared Libraries"
  $registry = "HKLM:\SOFTWARE\Khronos\OpenCL\Vendors"
  if (Test-Path $registry) {
    $properties = Get-ItemProperty $registry
    foreach ($property in $properties.PSObject.Properties) {
      if ($property.Name.StartsWith($sharedRoot, [StringComparison]::OrdinalIgnoreCase) -and
          $property.Name -like '*intelocl64.dll' -and (Test-Path $property.Name)) {
        return $property.Name
      }
    }
  }
  Get-ChildItem $sharedRoot -Filter intelocl64.dll -File -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
}

function Get-7ZipExecutable {
  $command = Get-Command 7z.exe -ErrorAction SilentlyContinue
  if ($command) { return $command.Source }
  $installed = "C:\Program Files\7-Zip\7z.exe"
  if (Test-Path $installed) { return $installed }
  return $null
}

function Install-OpenClCpuRuntime {
  if (Get-OpenClCpuRuntime) {
    Write-Host "Intel CPU OpenCL runtime is already available."
    return
  }
  if (-not (Test-Administrator)) {
    throw "Run install.bat from an elevated Administrator command prompt."
  }

  $workspace = Join-Path $tempRoot ("mom-opencl-cpu-{0}" -f [Guid]::NewGuid().ToString('N'))
  [IO.Directory]::CreateDirectory($workspace) | Out-Null
  try {
    $sevenZip = Get-7ZipExecutable
    if (-not $sevenZip) {
      $sevenZipMsi = Join-Path $workspace '7zip.msi'
      Invoke-WebRequest -UseBasicParsing -Uri $sevenZipUrl -OutFile $sevenZipMsi
      $actual = (Get-FileHash -Algorithm SHA256 $sevenZipMsi).Hash.ToLowerInvariant()
      if ($actual -ne $sevenZipSha256) {
        throw "7-Zip MSI SHA256 mismatch: expected $sevenZipSha256, got $actual"
      }
      Install-MsiPackage $sevenZipMsi
      $sevenZip = Get-7ZipExecutable
      if (-not $sevenZip) { throw '7-Zip installation completed but 7z.exe was not found.' }
    }

    $installer = Join-Path $workspace 'opencl-cpu-runtime.exe'
    Invoke-WebRequest -UseBasicParsing -Uri $openClCpuUrl -OutFile $installer
    Assert-Authenticode $installer
    # Intel's silent wrapper can outlive its MSI on headless systems; installing the signed payload
    # directly gives setup one deterministic process and exit code.
    $extractDir = Join-Path $workspace 'payload'
    [IO.Directory]::CreateDirectory($extractDir) | Out-Null
    & $sevenZip x -y -bso0 "-o$extractDir" $installer
    if ($LASTEXITCODE -ne 0) { throw "7-Zip extraction failed with exit code $LASTEXITCODE" }
    $msi = Get-ChildItem $extractDir -Filter '*.msi' -File | Select-Object -First 1
    if (-not $msi) { throw 'Intel CPU OpenCL wrapper did not contain an MSI payload.' }
    Install-MsiPackage $msi.FullName
  } finally {
    Remove-Item -LiteralPath $workspace -Recurse -Force -ErrorAction SilentlyContinue
  }
  if (-not (Get-OpenClCpuRuntime)) {
    throw 'Intel CPU OpenCL runtime did not register its ICD.'
  }
  Write-Host "Intel CPU OpenCL runtime installed."
}

function Get-NvidiaSmi {
  $cmd = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }

  $system32 = Join-Path $env:WINDIR "System32\nvidia-smi.exe"
  if (Test-Path $system32) { return $system32 }

  $nvSmi = "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
  if (Test-Path $nvSmi) { return $nvSmi }

  return $null
}

function Get-CudaToolkitRoot {
  $roots = @()
  if ($env:CUDA_PATH) { $roots += $env:CUDA_PATH }
  if ($env:CUDA_HOME) { $roots += $env:CUDA_HOME }
  $roots += $cudaToolkitRoot

  foreach ($root in $roots | Select-Object -Unique) {
    if (-not $root) { continue }
    $hasNvrtc = [bool](Get-ChildItem (Join-Path $root "bin") -Filter "nvrtc64_*.dll" `
      -File -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ((Test-Path (Join-Path $root "bin\ptxas.exe")) -and
        (Test-Path (Join-Path $root "nvvm\libdevice\libdevice.10.bc")) -and
        (Test-Path (Join-Path $root "include\cuda.h")) -and
        (Test-Path (Join-Path $root "include\cuda_runtime.h")) -and
        (Test-Path (Join-Path $root "include\nvrtc.h")) -and $hasNvrtc) {
      return $root
    }
  }
  return $null
}

function Test-CudaToolkitReady {
  return [bool](Get-CudaToolkitRoot)
}

function Test-CudaNvrtc {
  param([Parameter(Mandatory = $true)][string]$CudaRoot)
  return ((Test-Path (Join-Path $CudaRoot "bin\nvrtc64_130_0.dll")) -and
    (Test-Path (Join-Path $CudaRoot "bin\nvrtc-builtins64_131.dll")))
}

function Install-CudaNvrtc {
  param([Parameter(Mandatory = $true)][string]$CudaRoot)
  if (Test-CudaNvrtc $CudaRoot) {
    Write-Host "CUDA 13 NVRTC runtime is already available; skipping download."
    return
  }
  if (-not (Test-Administrator)) {
    throw "Run install.bat from an elevated Administrator command prompt."
  }

  $archive = Join-Path $tempRoot ("mom-cuda-nvrtc-{0}.zip" -f [Guid]::NewGuid().ToString('N'))
  Invoke-WebRequest -UseBasicParsing -Uri $cudaNvrtcUrl -OutFile $archive
  $actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
  if ($actual -ne $cudaNvrtcSha256) {
    Remove-Item $archive -Force -ErrorAction SilentlyContinue
    throw "CUDA NVRTC archive SHA256 mismatch: expected $cudaNvrtcSha256, got $actual"
  }

  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $zip = [System.IO.Compression.ZipFile]::OpenRead($archive)
  try {
    foreach ($name in @("nvrtc64_130_0.dll", "nvrtc-builtins64_131.dll")) {
      $entry = $zip.Entries | Where-Object { $_.Name -eq $name } | Select-Object -First 1
      if (-not $entry) { throw "CUDA NVRTC archive does not contain $name" }
      [System.IO.Compression.ZipFileExtensions]::ExtractToFile(
        $entry, (Join-Path $CudaRoot "bin\$name"), $true)
    }
  } finally {
    $zip.Dispose()
    Remove-Item $archive -Force -ErrorAction SilentlyContinue
  }
  if (-not (Test-CudaNvrtc $CudaRoot)) {
    throw "CUDA 13 NVRTC runtime install failed validation."
  }
  Write-Host "CUDA 13 NVRTC runtime installed for architecture-aware PearlHash source compilation."
}

function Test-CppToolchainReady {
  $cppRoots = @(
    "C:\BuildTools\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
  )
  $hasTypeTraits = $false
  foreach ($root in $cppRoots) {
    if (Get-ChildItem -Path $root -Filter type_traits -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1) {
      $hasTypeTraits = $true
      break
    }
  }

  $ucrtRoot = "C:\Program Files (x86)\Windows Kits\10\Include"
  $hasStringH = [bool](Get-ChildItem -Path $ucrtRoot -Filter string.h -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1)
  return ($hasTypeTraits -and $hasStringH)
}

function Install-CppToolchain {
  if (Test-CppToolchainReady) {
    Write-Host "Visual C++ headers are already available; skipping Build Tools install."
    return
  }
  if (-not (Test-Administrator)) {
    throw "Run install.bat from an elevated Administrator command prompt."
  }

  $vsDir = Join-Path $tempRoot ("mom-vs-build-tools-{0}" -f [Guid]::NewGuid().ToString('N'))
  $vsInstaller = Join-Path $vsDir "vs_BuildTools.exe"
  [IO.Directory]::CreateDirectory($vsDir) | Out-Null

  try {
    Write-Host "Downloading Visual Studio Build Tools from $fallbackVsBuildToolsUrl"
    Invoke-WebRequest -UseBasicParsing -Uri $fallbackVsBuildToolsUrl -OutFile $vsInstaller
    Assert-Authenticode $vsInstaller

    $args = @(
      "--quiet", "--wait", "--norestart", "--nocache",
      "--installPath", "C:\BuildTools",
      "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
      "--add", "Microsoft.VisualStudio.Component.Windows11SDK.26100"
    )
    $install = Start-Process -FilePath $vsInstaller -ArgumentList $args -Wait -PassThru
    if ($install.ExitCode -ne 0 -and $install.ExitCode -ne 3010) {
      throw "Visual Studio Build Tools install failed with exit code $($install.ExitCode)."
    }
  } finally {
    Remove-Item -Recurse -Force $vsDir -ErrorAction SilentlyContinue
  }

  if (-not (Test-CppToolchainReady)) {
    throw "Visual Studio Build Tools install completed but C++ or Windows SDK headers were not found."
  }
  Write-Host "Visual C++ and Windows SDK headers are available."
}

function Install-CudaToolkit([switch]$Force) {
  if (-not $Force -and -not (Get-NvidiaPnpDevice)) {
    Write-Host "No NVIDIA PCI device was detected; skipping CUDA toolkit install."
    return
  }
  if (Test-CudaToolkitReady) {
    Write-Host "CUDA toolkit headers, ptxas, and libdevice are already available; skipping CUDA toolkit install."
    return
  }
  if (-not (Test-Administrator)) {
    throw "Run install.bat from an elevated Administrator command prompt."
  }

  $url = if ($CudaToolkitUrl) { $CudaToolkitUrl } else { $fallbackCudaToolkitUrl }
  $cudaDir = Join-Path $tempRoot ("mom-cuda-toolkit-{0}" -f [Guid]::NewGuid().ToString('N'))
  $cudaInstaller = Join-Path $cudaDir "cuda-toolkit-network.exe"
  [IO.Directory]::CreateDirectory($cudaDir) | Out-Null

  try {
    Write-Host "Downloading CUDA toolkit network installer from $url"
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $cudaInstaller
    Assert-Authenticode $cudaInstaller

    $install = Start-Process -FilePath $cudaInstaller -ArgumentList (@("-s") + $cudaToolkitPackages) -Wait -PassThru
    if ($install.ExitCode -ne 0 -and $install.ExitCode -ne 3010) {
      throw "CUDA toolkit install failed with exit code $($install.ExitCode)."
    }
  } finally {
    Remove-Item -Recurse -Force $cudaDir -ErrorAction SilentlyContinue
  }

  $cudaRoot = $cudaToolkitRoot
  if (Test-Path $cudaRoot) {
    $env:CUDA_PATH = $cudaRoot
    $env:Path = (Join-Path $cudaRoot "bin") + ";" + $env:Path
  }
  if (-not (Test-CudaToolkitReady)) {
    throw "CUDA toolkit install completed but required headers, ptxas, or libdevice were not found."
  }
  Write-Host "CUDA toolkit installation completed."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$cutlassHelperSha256 = "319405c20146e2ad20324d2ae895e222c5e5b58fdad0cccb93d360995aada893"
$needsCudaSupport = [bool]((Get-NvidiaPnpDevice) -or $InstallCudaToolkit)
if ($needsCudaSupport) {
  $helper = Join-Path $scriptDir "install-cutlass.ps1"
  if (Test-Path $helper) {
    . $helper
  } else {
    $helper = Join-Path $tempRoot ("mom-install-cutlass-{0}.ps1" -f [Guid]::NewGuid().ToString('N'))
    try {
      Invoke-WebRequest -UseBasicParsing `
        -Uri "https://raw.githubusercontent.com/MoneroOcean/mo-miner/v0.9.0/scripts/install-cutlass.ps1" `
        -OutFile $helper
      $actual = (Get-FileHash -Algorithm SHA256 $helper).Hash.ToLowerInvariant()
      if ($actual -ne $cutlassHelperSha256) {
        throw "CUTLASS helper SHA256 mismatch: expected $cutlassHelperSha256, got $actual"
      }
      . $helper
    } finally {
      Remove-Item $helper -Force -ErrorAction SilentlyContinue
    }
  }
}

function Install-CutlassHeaders {
  if (-not (Test-MomCutlass)) {
    if (-not (Test-Administrator)) {
      throw "Run install.bat from an elevated Administrator command prompt."
    }
    Install-MomCutlass
  }
}

function Install-CcclHeaders {
  param([Parameter(Mandatory = $true)][string]$CudaRoot)
  if (-not (Test-MomCccl $CudaRoot)) {
    if (-not (Test-Administrator)) {
      throw "Run install.bat from an elevated Administrator command prompt."
    }
    Install-MomCccl $CudaRoot
  }
}
if (Get-ChildItem -Path (Join-Path $scriptDir "libs") -Filter "ur_adapter_level_zero*.dll" -File -Recurse -ErrorAction SilentlyContinue) {
  Write-Host "Bundled Level Zero UR adapter is present."
} else {
  Write-Host "Bundled Level Zero UR adapter was not found; update the mom release package if GPU Level Zero is unavailable."
}
if (Get-ChildItem -Path (Join-Path $scriptDir "libs") -Filter "intelocl*.dll" -File -Recurse -ErrorAction SilentlyContinue) {
  Write-Host "Bundled Intel OpenCL runtime is present."
} else {
  Write-Host "Bundled Intel OpenCL runtime was not found; update the mom release package if Intel OpenCL is unavailable."
}

if ($DryRun) {
  Write-Host "GPU drivers are prerequisites and are not installed by this script."
  if ((Get-NvidiaPnpDevice) -or $InstallCudaToolkit) {
    $smi = Get-NvidiaSmi
    if ($smi) {
      Write-Host "nvidia-smi is available at $smi."
    } else {
      Write-Host "nvidia-smi was not found; install the NVIDIA driver before running NVIDIA GPU mining."
    }
    $resolvedCudaToolkitUrl = if ($CudaToolkitUrl) { $CudaToolkitUrl } else { $fallbackCudaToolkitUrl }
    Write-Host "CUDA toolkit URL: $resolvedCudaToolkitUrl"
    Write-Host "CUDA toolkit packages: $($cudaToolkitPackages -join ', ')"
    Write-Host "CUDA toolkit ready: $(Test-CudaToolkitReady)"
    $detectedCudaRoot = Get-CudaToolkitRoot
    Write-Host "CUDA 13 NVRTC ready: $(if ($detectedCudaRoot) { Test-CudaNvrtc $detectedCudaRoot } else { $false })"
    Write-Host "CCCL headers ready: $(if ($detectedCudaRoot) { Test-MomCccl $detectedCudaRoot } else { $false })"
    Write-Host "CUTLASS headers ready: $(Test-MomCutlass)"
  } else {
    Write-Host "No NVIDIA PCI device detected."
  }
  if (Get-AmdPnpDevice) {
    Write-Host "AMD GPU detected; the bundled HIP workers use the installed AMD display driver."
  }
  Write-Host "Intel CPU OpenCL runtime ready: $([bool](Get-OpenClCpuRuntime))"
  Write-Host "Dry run completed; no packages were installed."
  exit 0
}

if ($InstallCudaToolkit) {
  Install-CudaToolkit -Force
  $cudaRoot = Get-CudaToolkitRoot
  Install-CudaNvrtc $cudaRoot
  Install-CcclHeaders $cudaRoot
  Install-CutlassHeaders
  Install-CppToolchain
} elseif ((Get-NvidiaPnpDevice) -and -not $SkipCudaToolkit) {
  Install-CudaToolkit
  $cudaRoot = Get-CudaToolkitRoot
  Install-CudaNvrtc $cudaRoot
  Install-CcclHeaders $cudaRoot
  Install-CutlassHeaders
  Install-CppToolchain
} elseif ((Get-NvidiaPnpDevice) -and $SkipCudaToolkit) {
  Write-Host "NVIDIA CUDA/C++ provisioning was explicitly skipped."
}

Install-OpenClCpuRuntime

Write-Host "GPU drivers are prerequisites: native Intel/NVIDIA/AMD support and other vendors' OpenCL ICDs come from their current display drivers."
Write-Host "GPU runtime setup completed."
