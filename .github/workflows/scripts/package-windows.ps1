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

if (-not (Test-Path "build/Release/mom.node")) {
  throw "build/Release/mom.node is missing; build the native addon before packaging."
}
if (-not (Test-Path "build/Release/sycl.dll")) {
  throw "build/Release/sycl.dll is missing; Windows release packages require SYCL support."
}

Remove-Item -Recurse -Force release, release-build, $Archive -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $packageDir, $libsDir, release-build | Out-Null

$bundlePath = (Resolve-Path release-build).Path + "\mom.bundle.cjs"
npx --yes esbuild@0.28.0 mom.js `
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
if not defined MOM_COMMAND set "MOM_COMMAND=mom"
if not defined OCL_ICD_FILENAMES for %%F in ("%MOM_LIBS%\intelocl*.dll") do if exist "%%~fF" set "OCL_ICD_FILENAMES=%%~fF"
"%MOM_DIR%mom-node.exe" "%MOM_DIR%mom.bundle.cjs" %*
exit /b %ERRORLEVEL%
'@ | Set-Content -Encoding ascii "$packageDir/mom.cmd"

Copy-Item package.json, README.md, LICENSE "$packageDir/"
Copy-Item scripts/install.bat "$packageDir/install.bat"
Copy-Item scripts/install.ps1 "$packageDir/install.ps1"
Copy-Item build/Release/mom.node "$libsDir/"
Copy-Item build/Release/sycl.dll "$libsDir/"

Copy-MominerOptionalRuntimeFiles -PackageDir $libsDir
$entryPaths = @("$packageDir/mom-node.exe", "$libsDir/mom.node", "$libsDir/sycl.dll")
Copy-MominerDllClosure -PackageDir $libsDir -EntryPaths $entryPaths

if (Test-Path "$packageDir/tests") {
  throw "Release package unexpectedly contains tests/."
}

Compress-Archive -Path $packageDir -DestinationPath $Archive
Write-Output $Archive
