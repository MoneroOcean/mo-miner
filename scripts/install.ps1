# Windows Level Zero GPU support is installed by the Intel Graphics Driver.
# The release package already bundles the oneAPI Level Zero loader and SYCL
# runtime DLLs. This script installs the Intel OpenCL CPU runtime silently so
# CPU SYCL/OpenCL devices are available on Windows release systems.

param(
  [string]$OpenClRuntimeUrl = "",
  [switch]$InstallIntelGraphicsDriver,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if ($PSVersionTable.PSVersion.Major -ge 7) {
  $PSNativeCommandUseErrorActionPreference = $true
}

function Test-Administrator {
  $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
  $principal = New-Object Security.Principal.WindowsPrincipal($identity)
  return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

$tempRoot = if ($env:TEMP) { $env:TEMP } elseif ($env:TMPDIR) { $env:TMPDIR } else { "/tmp" }
$profileRoot = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
$runtimeDir = Join-Path $tempRoot "mo-miner-opencl-runtime"
$installer = Join-Path $runtimeDir "opencl-runtime.exe"
$extractRoot = Join-Path $profileRoot "Downloads\Intel"
$openClRuntimePage = "https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html"
$fallbackOpenClRuntimeUrl = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/ad824c04-01c8-4ae5-b5e8-164a04f67609/w_opencl_runtime_p_2025.3.1.762.exe"

function Resolve-OpenClRuntimeUrl {
  if ($OpenClRuntimeUrl) {
    return $OpenClRuntimeUrl
  }

  try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri $openClRuntimePage
    $matches = [regex]::Matches(
      $response.Content,
      'https://registrationcenter-download\.intel\.com/[^"''<>\s]+/w_opencl_runtime_p_[0-9.]+\.exe'
    )
    $urls = @(
      $matches |
        ForEach-Object { $_.Value } |
        Sort-Object -Unique
    )
    if ($urls.Count -gt 0) {
      return $urls[0]
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

  $upgrade = Start-Process -FilePath $winget.Source -ArgumentList $arguments -Wait -PassThru
  if ($upgrade.ExitCode -eq 0) {
    Write-Host "Intel Graphics Driver update completed."
    return
  }

  Write-Warning "Intel Graphics Driver winget update exited with $($upgrade.ExitCode); trying install."
  $arguments[0] = "install"
  $install = Start-Process -FilePath $winget.Source -ArgumentList $arguments -Wait -PassThru
  if ($install.ExitCode -ne 0) {
    Write-Warning "Intel Graphics Driver winget install exited with $($install.ExitCode)."
  }
}

$resolvedOpenClRuntimeUrl = Resolve-OpenClRuntimeUrl

if ($DryRun) {
  Write-Host "Intel OpenCL CPU runtime URL: $resolvedOpenClRuntimeUrl"
  if (Get-Command winget.exe -ErrorAction SilentlyContinue) {
    Write-Host "winget.exe is available for optional Intel Graphics Driver updates."
  } else {
    Write-Host "winget.exe was not found; optional Intel Graphics Driver update would be skipped."
  }
  Write-Host "Dry run completed; no packages were installed."
  exit 0
}

if (-not (Test-Administrator)) {
  throw "Run install.bat from an elevated Administrator command prompt."
}

function Stop-OpenClRuntimeInstallers {
  Get-CimInstance Win32_Process |
    Where-Object {
      $_.ExecutablePath -eq $installer -or
      $_.CommandLine -like "*w_opencl_runtime_p_*" -or
      $_.CommandLine -like "*opencl-runtime.exe*"
    } |
    ForEach-Object {
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Remove-Item -Recurse -Force $runtimeDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $runtimeDir | Out-Null

Write-Host "Downloading Intel OpenCL CPU runtime from $resolvedOpenClRuntimeUrl"
Invoke-WebRequest `
  -UseBasicParsing `
  -Uri $resolvedOpenClRuntimeUrl `
  -OutFile $installer

Stop-OpenClRuntimeInstallers
Remove-Item -Recurse -Force $extractRoot -ErrorAction SilentlyContinue
$extract = Start-Process `
  -FilePath $installer `
  -ArgumentList @("-s", "-a", "--silent", "--eula", "accept") `
  -PassThru

$deadline = (Get-Date).AddMinutes(3)
$msi = $null
while ((Get-Date) -lt $deadline) {
  $msi = Get-ChildItem `
    -Path $extractRoot `
    -Filter "w_opencl_runtime_p_*.msi" `
    -File `
    -Recurse `
    -ErrorAction SilentlyContinue |
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

$install = Start-Process `
  -FilePath msiexec.exe `
  -ArgumentList @("/i", $msi.FullName, "/qn", "/norestart") `
  -Wait `
  -PassThru

if ($install.ExitCode -ne 0 -and $install.ExitCode -ne 3010) {
  throw "Intel OpenCL CPU runtime install failed with exit code $($install.ExitCode)."
}

if ($InstallIntelGraphicsDriver) {
  Install-IntelGraphicsDriver
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$libsDir = Join-Path $scriptDir "libs"
if (Test-Path (Join-Path $libsDir "ze_loader.dll")) {
  Write-Host "Bundled Level Zero loader is present."
} else {
  Write-Host "Bundled Level Zero loader was not found; update the mo-miner release package if GPU Level Zero is unavailable."
}

Write-Host "Intel OpenCL CPU runtime installation completed."
Write-Host "Intel GPU Level Zero runtime is provided by the Intel Graphics Driver."
Write-Host "Run install.ps1 -InstallIntelGraphicsDriver to try a non-interactive winget driver update."
