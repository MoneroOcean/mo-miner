$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

trap {
  if ($env:GITHUB_ACTIONS) {
    # GitHub Actions workflow commands require %/CR/LF percent-encoded in the message.
    $message = $_.Exception.Message.Replace('%', '%25').Replace("`r", '%0D').Replace("`n", '%0A')
    Write-Host "::error title=Windows build failed::$message"
  }
  break
}

function Invoke-MominerNative {
  param([scriptblock]$Command, [string]$Name)
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Name failed with exit code $LASTEXITCODE."
  }
}

function Find-MominerNodeGyp {
  $candidates = @()

  $nodeRoot = Split-Path (Get-Command node -ErrorAction Stop).Source -Parent
  $candidates += Join-Path $nodeRoot "node_modules\npm\node_modules\node-gyp\bin\node-gyp.js"

  $npmCommand = Get-Command npm -ErrorAction SilentlyContinue
  if ($npmCommand) {
    $npmRoot = Split-Path $npmCommand.Source -Parent
    $candidates += Join-Path $npmRoot "node_modules\npm\node_modules\node-gyp\bin\node-gyp.js"
  }

  $globalRootOutput = & npm root -g 2>$null
  if ($LASTEXITCODE -eq 0 -and $globalRootOutput) {
    $globalRoot = ($globalRootOutput | Select-Object -First 1).Trim()
    $candidates += Join-Path $globalRoot "npm\node_modules\node-gyp\bin\node-gyp.js"
    $candidates += Join-Path $globalRoot "node-gyp\bin\node-gyp.js"
  }

  foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }
  }
  return $null
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
Set-Location $repoRoot
if (-not $env:ONEAPI_ROOT) {
  $env:ONEAPI_ROOT = "C:\Program Files (x86)\Intel\oneAPI"
}
if (-not $env:VS2022INSTALLDIR -and (Test-Path "C:\BuildTools")) {
  $env:VS2022INSTALLDIR = "C:\BuildTools"
}

Invoke-MominerNative { node -v } "node"
Invoke-MominerNative { npm -v } "npm"
Invoke-MominerNative { python --version } "python"

$nodeGyp = Find-MominerNodeGyp
if (-not $nodeGyp) {
  Invoke-MominerNative { npm install -g node-gyp@12.2.0 } "npm install node-gyp"
  $nodeGyp = Find-MominerNodeGyp
}
if (-not $nodeGyp) {
  throw "Unable to locate node-gyp."
}
Write-Host "Using node-gyp at $nodeGyp"
Invoke-MominerNative { node $nodeGyp --version } "node-gyp version"

$setvars = Join-Path $env:ONEAPI_ROOT "setvars.bat"
$envLines = & cmd.exe /d /s /c "call `"$setvars`" intel64 --force >nul && set"
if ($LASTEXITCODE -ne 0) {
  throw "oneAPI setvars failed with exit code $LASTEXITCODE."
}
foreach ($line in $envLines) {
  if ($line -match '^([^=]+)=(.*)$') {
    [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
  }
}
# The Intel MSBuild toolsets read $(ICInstallDir)/$(IDPCInstallDir) to locate icx. Point them at the REAL
# compiler dir: the `compiler\latest` junction does NOT survive actions/cache restore in CI (it exists on
# a normal install), which makes the toolset fail "Could not expand ICInstallDir". Derive it from icx's
# actual location, then fall back to `latest`, then the newest versioned `compiler\<ver>` dir.
$compilerDir = $null
$icx = (Get-Command icx.exe -ErrorAction SilentlyContinue).Source
if ($icx) {
  # ...\compiler\<ver>\bin\compiler\icx.exe -> ...\compiler\<ver>
  $compilerDir = Split-Path (Split-Path (Split-Path $icx -Parent) -Parent) -Parent
}
if (-not $compilerDir -or -not (Test-Path (Join-Path $compilerDir "bin"))) {
  $latest = Join-Path $env:ONEAPI_ROOT "compiler\latest"
  if (Test-Path (Join-Path $latest "bin")) {
    $compilerDir = $latest
  } else {
    $verDir = Get-ChildItem (Join-Path $env:ONEAPI_ROOT "compiler") -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match '^[0-9]' } | Sort-Object Name -Descending | Select-Object -First 1
    if ($verDir) { $compilerDir = $verDir.FullName }
  }
}
if (-not $compilerDir) { throw "Could not locate the oneAPI compiler directory under $env:ONEAPI_ROOT." }
$compilerDir = $compilerDir.TrimEnd('\') + "\"
$env:ICInstallDir = $compilerDir
$env:IDPCInstallDir = $compilerDir
Write-Host "ICInstallDir = $compilerDir"

Invoke-MominerNative { icx --version } "icx"

Invoke-MominerNative { node $nodeGyp configure --msvs_version=2022 } "node-gyp configure"
# MSBuild is on PATH inside a VS Developer/Native-Tools shell (and on CI runners), but not on a bare
# VS Build Tools box. Fall back to locating it via vswhere so a local Windows build works either way.
$msbuildCmd = Get-Command MSBuild.exe -ErrorAction SilentlyContinue
if ($msbuildCmd) {
  $msbuild = $msbuildCmd.Source
} else {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vswhere) {
    # NB: do not add -requires Microsoft.Component.MSBuild -- VS Build Tools doesn't report that exact
    # component id, and it makes -find return nothing. -products * covers BuildTools + full VS.
    $msbuild = & $vswhere -latest -products * -find "MSBuild\**\Bin\MSBuild.exe" |
      Select-Object -First 1
  }
  if (-not $msbuild) {
    throw "MSBuild.exe not found on PATH or via vswhere; run from a VS Developer/Native Tools prompt."
  }
}
# Capture output (rather than letting it stream) so the tail can be re-shown in the failure message.
$msbuildOutput = & $msbuild build\mom.vcxproj /clp:Verbosity=minimal /nologo /nodeReuse:false /p:Configuration=Release /p:Platform=x64 2>&1
$msbuildOutput | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) {
  $tail = ($msbuildOutput | Select-Object -Last 80) -join "`n"
  throw "MSBuild failed with exit code $LASTEXITCODE.`n$tail"
}

New-Item -ItemType Directory -Force build\Release | Out-Null
$releasePath = (Resolve-Path build\Release).Path
Get-ChildItem build -Recurse -Include *.node,*.dll |
  Where-Object { $_.FullName -ne (Join-Path $releasePath $_.Name) } |
  Copy-Item -Destination build\Release -Force

if (-not (Test-Path build\Release\mom.node)) {
  throw "build\Release\mom.node was not created."
}
