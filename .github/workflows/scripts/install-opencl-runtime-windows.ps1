param(
  [string]$Url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/ad824c04-01c8-4ae5-b5e8-164a04f67609/w_opencl_runtime_p_2025.3.1.762.exe"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if ($PSVersionTable.PSVersion.Major -ge 7) {
  $PSNativeCommandUseErrorActionPreference = $true
}

$runtimeDir = Join-Path $env:TEMP "mom-opencl-runtime"
$installer = Join-Path $runtimeDir "opencl-runtime.exe"
$extractRoot = Join-Path $env:USERPROFILE "Downloads\Intel"

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

Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $installer

Stop-OpenClRuntimeInstallers
Remove-Item -Recurse -Force $extractRoot -ErrorAction SilentlyContinue

# The installer self-extracts the MSI into $extractRoot and then re-launches it
# interactively, so run it non-blocking and poll for the MSI instead of waiting.
Start-Process -FilePath $installer -ArgumentList "-s", "-a", "--silent", "--eula", "accept"

$deadline = (Get-Date).AddMinutes(3)
$msi = $null
while ((Get-Date) -lt $deadline) {
  $msi = Get-ChildItem -Path $extractRoot -Filter "w_opencl_runtime_p_*.msi" -File -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if ($msi) { break }
  Start-Sleep -Seconds 1
}

# Kill the bootstrapper's interactive MSI re-launch before we drive msiexec ourselves.
Stop-OpenClRuntimeInstallers

if (-not $msi) {
  throw "Unable to extract Intel OpenCL CPU runtime MSI from $installer."
}

$msiProc = Start-Process -FilePath msiexec.exe `
  -ArgumentList "/i", $msi.FullName, "/qn", "/norestart" -Wait -PassThru

# 3010 == ERROR_SUCCESS_REBOOT_REQUIRED: a successful install that wants a reboot.
if ($msiProc.ExitCode -ne 0 -and $msiProc.ExitCode -ne 3010) {
  throw "Intel OpenCL CPU runtime install failed with exit code $($msiProc.ExitCode)."
}
