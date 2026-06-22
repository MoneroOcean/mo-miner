# Windows Level Zero GPU support is installed by the Intel Graphics Driver.
# NVIDIA CUDA support is installed by the NVIDIA driver; the CUDA toolkit is
# only needed for runtime SYCL-source JIT paths such as the ProgPoW kernels.
# The release package already bundles the oneAPI SYCL/UR runtime DLLs. This
# script installs the Intel OpenCL CPU runtime silently so CPU SYCL/OpenCL
# devices are available on Windows release systems.

param(
  [string]$OpenClRuntimeUrl = "",
  [string]$NvidiaDriverUrl = "",
  [string]$CudaToolkitUrl = "",
  [switch]$InstallIntelGraphicsDriver,
  [switch]$InstallNvidiaDriver,
  [switch]$InstallCudaToolkit,
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

$tempRoot = if ($env:TEMP) { $env:TEMP } elseif ($env:TMPDIR) { $env:TMPDIR } else { "/tmp" }
$profileRoot = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
$runtimeDir = Join-Path $tempRoot "mom-opencl-runtime"
$installer = Join-Path $runtimeDir "opencl-runtime.exe"
$extractRoot = Join-Path $profileRoot "Downloads\Intel"
$openClRuntimePage = "https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html"
$fallbackOpenClRuntimeUrl = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/ad824c04-01c8-4ae5-b5e8-164a04f67609/w_opencl_runtime_p_2025.3.1.762.exe"
$fallbackNvidiaDriverUrl = "https://us.download.nvidia.com/tesla/596.36/596.36-data-center-tesla-desktop-winserver-2022-2025-dch-international.exe"
$fallbackCudaToolkitUrl = "https://developer.download.nvidia.com/compute/cuda/12.6.0/network_installers/cuda_12.6.0_windows_network.exe"
$cudaToolkitPackages = @("nvcc_12.6", "cudart_12.6", "nvrtc_12.6", "nvrtc_dev_12.6")

function Resolve-OpenClRuntimeUrl {
  if ($OpenClRuntimeUrl) {
    return $OpenClRuntimeUrl
  }

  try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri $openClRuntimePage
    # Scrape the runtime .exe links (use $found, not the $Matches automatic var).
    $found = [regex]::Matches(
      $response.Content,
      'https://registrationcenter-download\.intel\.com/[^"''<>\s]+/w_opencl_runtime_p_[0-9.]+\.exe'
    )
    $url = @($found.Value | Sort-Object -Unique) | Select-Object -First 1
    if ($url) {
      return $url
    }
  } catch {
    Write-Warning "Unable to discover latest Intel OpenCL CPU runtime URL: $($_.Exception.Message)"
  }

  Write-Warning "Falling back to known Intel OpenCL CPU runtime URL. Pass -OpenClRuntimeUrl to override."
  return $fallbackOpenClRuntimeUrl
}

function Install-IntelGraphicsDriver {
  $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
  if (-not $winget) {
    Write-Warning "winget.exe was not found; skipping Intel Graphics Driver update."
    return
  }

  $arguments = @(
    "upgrade",
    "--id", "Intel.IntelGraphicsDriver",
    "--exact",
    "--silent",
    "--accept-package-agreements",
    "--accept-source-agreements",
    "--disable-interactivity"
  )
  $runWinget = { (Start-Process -FilePath $winget.Source -ArgumentList $arguments -Wait -PassThru).ExitCode }

  $upgradeExitCode = & $runWinget
  if ($upgradeExitCode -eq 0) {
    Write-Host "Intel Graphics Driver update completed."
    return
  }

  # winget 'upgrade' fails when the driver is not yet installed; retry with 'install'.
  Write-Warning "Intel Graphics Driver winget update exited with $upgradeExitCode; trying install."
  $arguments[0] = "install"
  $installExitCode = & $runWinget
  if ($installExitCode -ne 0) {
    Write-Warning "Intel Graphics Driver winget install exited with $installExitCode."
  }
}

function Get-NvidiaPnpDevice {
  Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue |
    Where-Object { $_.PNPDeviceID -like "PCI\VEN_10DE*" } |
    Select-Object -First 1
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

function Install-NvidiaDriver {
  if (-not (Get-NvidiaPnpDevice)) {
    Write-Host "No NVIDIA PCI device was detected; skipping NVIDIA driver install."
    return
  }

  $smi = Get-NvidiaSmi
  if ($smi) {
    Write-Host "nvidia-smi is already available at $smi; skipping NVIDIA driver install."
    return
  }

  $url = if ($NvidiaDriverUrl) { $NvidiaDriverUrl } else { $fallbackNvidiaDriverUrl }
  $driverDir = Join-Path $tempRoot "mom-nvidia-driver"
  $driverInstaller = Join-Path $driverDir "nvidia-driver.exe"
  Remove-Item -Recurse -Force $driverDir -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force $driverDir | Out-Null

  Write-Host "Downloading NVIDIA driver from $url"
  Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $driverInstaller

  $install = Start-Process -FilePath $driverInstaller -ArgumentList @("-s", "-noreboot") -Wait -PassThru
  if ($install.ExitCode -ne 0 -and $install.ExitCode -ne 3010) {
    throw "NVIDIA driver install failed with exit code $($install.ExitCode)."
  }

  Write-Host "NVIDIA driver installation completed. Reboot if nvidia-smi is still unavailable."
}

function Test-CudaToolkitReady {
  $roots = @()
  if ($env:CUDA_PATH) { $roots += $env:CUDA_PATH }
  $roots += "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

  foreach ($root in $roots | Select-Object -Unique) {
    if (-not $root) { continue }
    if ((Test-Path (Join-Path $root "bin\ptxas.exe")) -and
        (Test-Path (Join-Path $root "nvvm\libdevice\libdevice.10.bc"))) {
      return $true
    }
  }
  return $false
}

function Install-CudaToolkit {
  if (-not (Get-NvidiaPnpDevice)) {
    Write-Host "No NVIDIA PCI device was detected; skipping CUDA toolkit install."
    return
  }
  if (Test-CudaToolkitReady) {
    Write-Host "CUDA toolkit ptxas/libdevice are already available; skipping CUDA toolkit install."
    return
  }

  $url = if ($CudaToolkitUrl) { $CudaToolkitUrl } else { $fallbackCudaToolkitUrl }
  $cudaDir = Join-Path $tempRoot "mom-cuda-toolkit"
  $cudaInstaller = Join-Path $cudaDir "cuda-toolkit-network.exe"
  Remove-Item -Recurse -Force $cudaDir -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force $cudaDir | Out-Null

  Write-Host "Downloading CUDA toolkit network installer from $url"
  Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $cudaInstaller

  $install = Start-Process -FilePath $cudaInstaller -ArgumentList (@("-s") + $cudaToolkitPackages) -Wait -PassThru
  if ($install.ExitCode -ne 0 -and $install.ExitCode -ne 3010) {
    throw "CUDA toolkit install failed with exit code $($install.ExitCode)."
  }

  $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
  if (Test-Path $cudaRoot) {
    $env:CUDA_PATH = $cudaRoot
    $env:Path = (Join-Path $cudaRoot "bin") + ";" + $env:Path
  }
  Write-Host "CUDA toolkit installation completed."
}

$resolvedOpenClRuntimeUrl = Resolve-OpenClRuntimeUrl

if ($DryRun) {
  Write-Host "Intel OpenCL CPU runtime URL: $resolvedOpenClRuntimeUrl"
  if (Get-Command winget.exe -ErrorAction SilentlyContinue) {
    Write-Host "winget.exe is available for optional Intel Graphics Driver updates."
  } else {
    Write-Host "winget.exe was not found; optional Intel Graphics Driver update would be skipped."
  }
  if (Get-NvidiaPnpDevice) {
    $resolvedNvidiaDriverUrl = if ($NvidiaDriverUrl) { $NvidiaDriverUrl } else { $fallbackNvidiaDriverUrl }
    $resolvedCudaToolkitUrl = if ($CudaToolkitUrl) { $CudaToolkitUrl } else { $fallbackCudaToolkitUrl }
    Write-Host "NVIDIA PCI device detected."
    Write-Host "NVIDIA driver URL: $resolvedNvidiaDriverUrl"
    Write-Host "CUDA toolkit URL: $resolvedCudaToolkitUrl"
    Write-Host "CUDA toolkit packages: $($cudaToolkitPackages -join ', ')"
  } else {
    Write-Host "No NVIDIA PCI device detected."
  }
  Write-Host "Dry run completed; no packages were installed."
  exit 0
}

if (-not (Test-Administrator)) {
  throw "Run install.bat from an elevated Administrator command prompt."
}

if ($InstallNvidiaDriver) {
  Install-NvidiaDriver
}

if ($InstallCudaToolkit) {
  Install-CudaToolkit
}

function Stop-OpenClRuntimeInstallers {
  # Match only the actual installer EXECUTABLES by image path. Do NOT match on CommandLine containing
  # the runtime name/URL: when the URL is passed via -OpenClRuntimeUrl, this very script's process
  # (and its parent shell) carry that string on their command line, so a CommandLine match would make
  # the script Stop-Process itself and exit with no error. Excluding $PID is a further safety net.
  Get-CimInstance Win32_Process |
    Where-Object {
      $_.ProcessId -ne $PID -and (
        $_.ExecutablePath -eq $installer -or
        $_.ExecutablePath -like "*\w_opencl_runtime_p_*.exe" -or
        (
          $_.Name -ieq "msiexec.exe" -and
          $_.CommandLine -like "*\w_opencl_runtime_p_*.msi*" -and
          $_.CommandLine -notmatch "(/qn|/quiet)"
        )
      )
    } |
    ForEach-Object {
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Remove-Item -Recurse -Force $runtimeDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $runtimeDir | Out-Null

Write-Host "Downloading Intel OpenCL CPU runtime from $resolvedOpenClRuntimeUrl"
Invoke-WebRequest -UseBasicParsing -Uri $resolvedOpenClRuntimeUrl -OutFile $installer

Stop-OpenClRuntimeInstallers
Remove-Item -Recurse -Force $extractRoot -ErrorAction SilentlyContinue
# The installer self-extracts the MSI into $extractRoot; we poll for it below
# rather than waiting on the process, so its handle is intentionally discarded.
Start-Process -FilePath $installer -ArgumentList @("-s", "-a", "--silent", "--eula", "accept") | Out-Null

$deadline = (Get-Date).AddMinutes(3)
$msi = $null
while ((Get-Date) -lt $deadline) {
  $msi = Get-ChildItem -Path $extractRoot -Filter "w_opencl_runtime_p_*.msi" -File -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if ($msi) {
    break
  }
  Start-Sleep -Seconds 1
}

Stop-OpenClRuntimeInstallers

if (-not $msi) {
  throw "Unable to extract Intel OpenCL CPU runtime MSI from $installer."
}

$install = Start-Process -FilePath msiexec.exe -ArgumentList @("/i", $msi.FullName, "/qn", "/norestart") -Wait -PassThru

# 3010 = ERROR_SUCCESS_REBOOT_REQUIRED: the install succeeded, just needs a reboot.
if ($install.ExitCode -ne 0 -and $install.ExitCode -ne 3010) {
  throw "Intel OpenCL CPU runtime install failed with exit code $($install.ExitCode)."
}

if ($InstallIntelGraphicsDriver) {
  Install-IntelGraphicsDriver
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Get-ChildItem -Path (Join-Path $scriptDir "libs") -Filter "ur_adapter_level_zero*.dll" -File -ErrorAction SilentlyContinue) {
  Write-Host "Bundled Level Zero UR adapter is present."
} else {
  Write-Host "Bundled Level Zero UR adapter was not found; update the mom release package if GPU Level Zero is unavailable."
}

Write-Host "Intel OpenCL CPU runtime installation completed."
Write-Host "Intel GPU Level Zero runtime is provided by the Intel Graphics Driver."
Write-Host "Run install.ps1 -InstallIntelGraphicsDriver to try a non-interactive winget driver update."
Write-Host "Run install.ps1 -InstallNVIDIADriver -InstallCudaToolkit on NVIDIA Windows hosts that need driver/toolkit setup."
