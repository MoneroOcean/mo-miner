$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

trap {
  if ($env:GITHUB_ACTIONS) {
    $message = $_.Exception.Message.Replace('%', '%25').Replace("`r", '%0D').Replace("`n", '%0A')
    Write-Host "::error title=Windows build failed::$message"
  }
  break
}

function Invoke-MominerNative {
  param(
    [Parameter(Mandatory = $true)]
    [scriptblock]$Command,

    [Parameter(Mandatory = $true)]
    [string]$Name
  )

  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Name failed with exit code $LASTEXITCODE."
  }
}

function Find-MominerNodeGyp {
  $candidates = New-Object System.Collections.Generic.List[string]

  $nodeCommand = Get-Command node -ErrorAction Stop
  $nodeRoot = Split-Path $nodeCommand.Source -Parent
  $candidates.Add((Join-Path $nodeRoot "node_modules\npm\node_modules\node-gyp\bin\node-gyp.js"))

  $npmCommand = Get-Command npm -ErrorAction SilentlyContinue
  if ($npmCommand) {
    $npmRoot = Split-Path $npmCommand.Source -Parent
    $candidates.Add((Join-Path $npmRoot "node_modules\npm\node_modules\node-gyp\bin\node-gyp.js"))
  }

  $globalRootOutput = & npm root -g 2>$null
  if ($LASTEXITCODE -eq 0 -and $globalRootOutput) {
    $globalRoot = ($globalRootOutput | Select-Object -First 1).Trim()
    $candidates.Add((Join-Path $globalRoot "npm\node_modules\node-gyp\bin\node-gyp.js"))
    $candidates.Add((Join-Path $globalRoot "node-gyp\bin\node-gyp.js"))
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
$env:ICInstallDir = Join-Path $env:ONEAPI_ROOT "compiler\latest\"
$env:IDPCInstallDir = Join-Path $env:ONEAPI_ROOT "compiler\latest\"

Invoke-MominerNative { icx --version } "icx"

Invoke-MominerNative { node $nodeGyp configure --msvs_version=2022 } "node-gyp configure"
$msbuild = (Get-Command MSBuild.exe -ErrorAction Stop).Source
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
