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

$root = "mo-miner-v$Version"
if (-not $Archive) {
  $Archive = "mo-miner-v$Version-win.zip"
}
$packageDir = "release/$root"
$libsDir = Join-Path $packageDir "libs"
$nodeExe = if ($env:NODE_BIN) { $env:NODE_BIN } else { (Get-Command node.exe).Source }

if (-not (Test-Path "build/Release/mo-miner.node")) {
  throw "build/Release/mo-miner.node is missing; build the native addon before packaging."
}
if (-not (Test-Path "build/Release/sycl.dll")) {
  throw "build/Release/sycl.dll is missing; Windows release packages require SYCL support."
}

Remove-Item -Recurse -Force release, release-build, $Archive -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $packageDir, $libsDir, release-build | Out-Null

$bundlePath = (Resolve-Path release-build).Path + "\mo-miner.bundle.cjs"
npx --yes esbuild@0.28.0 mo-miner.js `
  --bundle `
  --platform=node `
  --format=cjs `
  --outfile="$bundlePath"
Copy-Item $nodeExe "$packageDir/mo-miner-node.exe"
Copy-Item $bundlePath "$packageDir/mo-miner.bundle.cjs"
@'
@echo off
setlocal
set "MOMINER_DIR=%~dp0"
set "MOMINER_LIBS=%MOMINER_DIR%libs"
set "PATH=%MOMINER_LIBS%;%MOMINER_DIR%;%CD%;%PATH%"
if not defined MOMINER_COMMAND set "MOMINER_COMMAND=mo-miner"
if not defined OCL_ICD_FILENAMES for %%F in ("%MOMINER_LIBS%\intelocl*.dll") do if exist "%%~fF" set "OCL_ICD_FILENAMES=%%~fF"
"%MOMINER_DIR%mo-miner-node.exe" "%MOMINER_DIR%mo-miner.bundle.cjs" %*
exit /b %ERRORLEVEL%
'@ | Set-Content -Encoding ascii "$packageDir/mo-miner.cmd"

Copy-Item package.json, README.md, LICENSE, install.bat, install.ps1 "$packageDir/"
Copy-Item build/Release/mo-miner.node "$libsDir/"
Copy-Item build/Release/sycl.dll "$libsDir/"

Copy-MominerOptionalRuntimeFiles -PackageDir $libsDir
$entryPaths = @("$packageDir/mo-miner-node.exe", "$libsDir/mo-miner.node", "$libsDir/sycl.dll")
Copy-MominerDllClosure -PackageDir $libsDir -EntryPaths $entryPaths

if (Test-Path "$packageDir/tests") {
  throw "Release package unexpectedly contains tests/."
}

Compress-Archive -Path $packageDir -DestinationPath $Archive
Write-Output $Archive
