param(
  [Parameter(Mandatory = $true)][string]$SourceRoot,
  [Parameter(Mandatory = $true)][string]$OutputRoot,
  [switch]$ExpectPluralFailure,
  [switch]$ExpectBuildFailure
)

$ErrorActionPreference = 'Stop'
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
if ([IO.Path]::GetFileName($OutputRoot) -notmatch '^mom-executable-[A-Za-z0-9_.-]+$' -or
    (Test-Path -LiteralPath $OutputRoot)) {
  throw 'OutputRoot must be a fresh dedicated mom-executable-* directory'
}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
Set-Location -LiteralPath $OutputRoot

function Read-ProductionAst([string]$Relative) {
  $tokens = $null
  $errors = $null
  $ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $SourceRoot $Relative), [ref]$tokens, [ref]$errors)
  if ($errors.Count) { throw "Production script parse failed: $Relative" }
  return $ast
}
function Assert-True([bool]$Condition, [string]$Message) {
  if (-not $Condition) { throw $Message }
}
function Test-Case([string]$Name, [scriptblock]$Body, [bool]$ShouldFail = $false) {
  $failed = $false
  try { & $Body } catch {
    if (-not $ShouldFail) { throw }
    $failed = $true
    Write-Output "EXPECTED_FAIL $Name`: $($_.Exception.Message)"
  }
  Assert-True ($failed -eq $ShouldFail) "$Name`: unexpected pass/fail state"
  if (-not $failed) { Write-Output "PASS $Name" }
}
function Assert-Missing([scriptblock]$Body, [string]$Text) {
  try { & $Body } catch {
    Assert-True ($_.Exception.Message.Contains($Text)) "Unexpected missing-tool error: $_"
    return
  }
  throw "Missing executable was accepted: $Text"
}

# Only pure definitions and the exact consumer statements are loaded. Installer entry points,
# package cleanup/build/archive phases and real executable discovery are never evaluated.
. (Join-Path $SourceRoot 'scripts\windows-install-helpers.ps1')
. (Join-Path $SourceRoot '.github\workflows\scripts\windows-dll-deps.ps1')
$install = Read-ProductionAst 'scripts\install-dev.ps1'
$hip = $install.Find({ param($node)
  $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
  $node.Name -eq 'Install-Hip'
}, $true)
Assert-True ($null -ne $hip) 'Install-Hip consumer is missing'
. ([scriptblock]::Create($hip.Extent.Text))
$package = Read-ProductionAst '.github\workflows\scripts\package-windows.ps1'
$lookup = $package.Find({ param($node)
  $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
  $node.Left.Extent.Text -eq '$nodeExe'
}, $true)
$copy = $package.Find({ param($node)
  $node -is [System.Management.Automation.Language.CommandAst] -and
  $node.GetCommandName() -eq 'Copy-Item' -and $node.CommandElements.Count -gt 1 -and
  $node.CommandElements[1].Extent.Text -eq '$nodeExe'
}, $true)
Assert-True ($null -ne $lookup -and $null -ne $copy) 'Node packaging consumer is missing'
$lookupNode = [scriptblock]::Create($lookup.Extent.Text)
$copyNode = [scriptblock]::Create($copy.Extent.Text)
$build = Read-ProductionAst '.github\workflows\scripts\build-windows.ps1'
$findGyp = $build.Find({ param($node)
  $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
  $node.Name -eq 'Find-MominerNodeGyp'
}, $true)
$msbuildLookup = $build.Find({ param($node)
  $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
  $node.Left.Extent.Text -eq '$msbuildCmd'
}, $true)
$msbuildSelect = $build.Find({ param($node)
  $node -is [System.Management.Automation.Language.IfStatementAst] -and
  $node.Clauses[0].Item1.Extent.Text -eq '$msbuildCmd'
}, $true)
$msbuildInvoke = $build.Find({ param($node)
  $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
  $node.Left.Extent.Text -eq '$msbuildOutput'
}, $true)
$buildTrap = $build.Find({ param($node)
  $node -is [System.Management.Automation.Language.TrapStatementAst]
}, $true)
Assert-True ($null -ne $findGyp -and $null -ne $msbuildLookup -and $null -ne $msbuildSelect -and
  $null -ne $msbuildInvoke -and $null -ne $buildTrap) 'Build discovery consumers are missing'
. ([scriptblock]::Create($findGyp.Extent.Text))
$selectMsbuild = [scriptblock]::Create($msbuildLookup.Extent.Text + "`n" +
  $msbuildSelect.Extent.Text + "`n" + $msbuildInvoke.Extent.Text)
$trapFixture = [scriptblock]::Create($buildTrap.Extent.Text +
  "`nWrite-Error -Message 'BUILD_ERROR_RECORD_FIXTURE' -ErrorId 'MomBuildFixture' -Category InvalidOperation")

$nodeFirst = Join-Path $OutputRoot 'node-first\node.exe'
$nodeSecond = Join-Path $OutputRoot 'node-second\node.exe'
$nodePinned = Join-Path $OutputRoot 'node-explicit\node.exe'
$npmFirst = Join-Path $OutputRoot 'npm-first\npm.cmd'
$npmSecond = Join-Path $OutputRoot 'npm-second\npm.cmd'
$npmEmpty = Join-Path $OutputRoot 'npm-empty\npm.cmd'
$packageDir = Join-Path $OutputRoot 'package'
foreach ($file in @($nodeFirst, $nodeSecond, $nodePinned, $npmFirst, $npmSecond, $npmEmpty)) {
  New-Item -ItemType Directory -Path (Split-Path -Parent $file) | Out-Null
  [IO.File]::WriteAllText($file, [IO.Path]::GetFileName([IO.Path]::GetDirectoryName($file)))
}
New-Item -ItemType Directory -Path $packageDir | Out-Null
$env:WINDIR = Join-Path $OutputRoot 'Windows'
${env:ProgramFiles(x86)} = Join-Path $OutputRoot 'ProgramFiles-x86'
New-Item -ItemType Directory -Path (Join-Path $env:WINDIR 'System32') | Out-Null
$Workspace = $OutputRoot
$HipDir = Join-Path $OutputRoot 'hip'
$hipSdkUrl = 'fixture://never-downloaded'
$hipSdkSha256 = 'fixture-not-hashed'
$entry = Join-Path $packageDir 'fixture.dll'
[IO.File]::WriteAllText($entry, 'not a native DLL')

function Get-Command {
  [CmdletBinding()]param([string]$Name)
  $script:LookupCalls++
  if ($script:Missing) {
    if ($ErrorActionPreference -eq 'Stop') { throw "DISCOVERY_MISSING: $Name" }
    return
  }
  $sources = switch ($Name) {
    '7z.exe' { 'First-SevenZip'; 'Second-SevenZip' }
    'dumpbin.exe' { 'First-DumpBin'; 'Second-DumpBin' }
    'node.exe' { $nodeFirst; $nodeSecond }
    'node' { if ($script:HasBundledNode) { $nodeFirst } else { $nodePinned }; $nodeSecond }
    'npm' { if ($script:HasBundledNpm) { $npmFirst } else { $npmEmpty }; $npmSecond }
    'MSBuild.exe' { 'First-MSBuild'; 'Second-MSBuild' }
    default { throw "Unexpected executable lookup: $Name" }
  }
  if ($Name -ne $script:PluralTool) { $sources = @($sources)[0] }
  foreach ($source in $sources) { [pscustomobject]@{ Source = $source } }
}
function First-SevenZip { $script:SevenZipCalls++; $global:LASTEXITCODE = 0 }
function Second-SevenZip { throw 'Second 7z candidate was executed' }
function First-DumpBin {
  $script:DumpBinCalls++
  '    kernel32.dll'
  $global:LASTEXITCODE = 0
}
function Second-DumpBin { throw 'Second dumpbin candidate was executed' }
function npm {
  Assert-True ($args.Count -eq 2 -and $args[0] -eq 'root' -and $args[1] -eq '-g') 'Unexpected optional npm query'
  $script:NpmCalls++
  if ($script:NpmThrows) { throw 'OPTIONAL_NPM_QUERY_FAILED' }
  $global:LASTEXITCODE = $script:NpmStatus
  return $script:NpmRoot
}
function First-MSBuild { $script:MsbuildCalls++; $global:LASTEXITCODE = 0; 'mock build output' }
function Second-MSBuild { throw 'Second MSBuild candidate was executed' }
function Get-ChildItem {
  [CmdletBinding()]param([object]$Path, [string]$Filter, [switch]$File,
    [switch]$Directory, [switch]$Recurse)
}
function Get-MominerDpcppBinDir { }
function Get-MominerAdaptiveCppBinDir { }
function Get-MominerCudaBinDir { }
function Get-MominerHipBinDir { }
function Get-MominerOneApiBinDirs { }
function Require-Administrator { }
function Download { }
function Assert-Sha256 { }
function Export-DevelopmentEnvironment { }
function Test-MomHipSdk { return $script:HipValidated }
function Install-Msi {
  $script:MsiCalls++
  if ($script:MsiCalls -eq 3) { $script:HipValidated = $true }
}

$script:Missing = $false
$script:PluralTool = '7z.exe'
$script:LookupCalls = 0
$script:SevenZipCalls = 0
$script:MsiCalls = 0
$script:HipValidated = $false
Test-Case '7z plural actual Install-Hip/Invoke-Checked' {
  Install-Hip
  Assert-True ($script:SevenZipCalls -eq 1 -and $script:MsiCalls -eq 3) '7z invocation was not scalar'
} $ExpectPluralFailure
$script:Missing = $true
$script:SevenZipCalls = 0
$script:MsiCalls = 0
$script:HipValidated = $false
Test-Case '7z missing actual Install-Hip' {
  Assert-Missing { Install-Hip } 'DISCOVERY_MISSING: 7z.exe'
  Assert-True ($script:SevenZipCalls -eq 0 -and $script:MsiCalls -eq 0) 'Missing 7z reached execution'
}
$script:Missing = $false
$script:PluralTool = 'dumpbin.exe'
$script:DumpBinCalls = 0
Test-Case 'dumpbin plural actual DLL closure' {
  Invoke-MominerDllClosure -PackageDir $packageDir -EntryPaths @($entry)
  Assert-True ($script:DumpBinCalls -eq 1) 'dumpbin invocation was not scalar'
} $ExpectPluralFailure
$script:Missing = $true
Test-Case 'dumpbin missing actual fallback' {
  Assert-Missing { Get-MominerDumpBin } 'dumpbin.exe was not found'
}
$env:NODE_BIN = ''
$script:Missing = $false
$script:PluralTool = 'node.exe'
Test-Case 'node plural actual package lookup/copy' {
  . $lookupNode
  . $copyNode
  Assert-True ([IO.File]::ReadAllText((Join-Path $packageDir 'mom-node.exe')) -eq 'node-first') 'Node copy did not retain the selected first candidate'
} $ExpectPluralFailure
$script:Missing = $true
Test-Case 'node missing actual package lookup' {
  Assert-Missing { . $lookupNode } 'DISCOVERY_MISSING: node.exe'
}
$env:NODE_BIN = $nodePinned
$script:LookupCalls = 0
Test-Case 'node explicit path bypasses ambient discovery' {
  . $lookupNode
  . $copyNode
  Assert-True ($script:LookupCalls -eq 0) 'Explicit NODE_BIN performed ambient discovery'
  Assert-True ([IO.File]::ReadAllText((Join-Path $packageDir 'mom-node.exe')) -eq 'node-explicit') 'Explicit NODE_BIN precedence changed'
}
$script:Missing = $false
Test-Case 'node plural actual DLL search roots' {
  $roots = @(Get-MominerDllSearchRoots $packageDir)
  Assert-True ($roots -contains (Split-Path -Parent $nodeFirst)) 'Selected Node directory is absent'
  Assert-True ($roots -notcontains (Split-Path -Parent $nodeSecond)) 'Unselected Node directory crossed into closure roots'
} $ExpectPluralFailure
$script:Missing = $true
Test-Case 'node missing optional DLL search roots' {
  $roots = @(Get-MominerDllSearchRoots $packageDir)
  Assert-True ($roots -contains $packageDir) 'Missing optional Node discovery lost package root'
  Assert-True ($roots -notcontains (Split-Path -Parent $nodeFirst)) 'Missing optional Node discovery added a Node directory'
}
$bundledNode = Join-Path (Split-Path -Parent $nodeFirst) 'node_modules\npm\node_modules\node-gyp\bin\node-gyp.js'
$bundledNpm = Join-Path (Split-Path -Parent $npmFirst) 'node_modules\npm\node_modules\node-gyp\bin\node-gyp.js'
$globalRoot = Join-Path $OutputRoot 'global-modules'
$globalGyp = Join-Path $globalRoot 'node-gyp\bin\node-gyp.js'
foreach ($file in @($bundledNode, $bundledNpm, $globalGyp)) {
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $file) | Out-Null
  [IO.File]::WriteAllText($file, 'fixture, never executed')
}
$script:Missing = $false
$script:NpmThrows = $true
$script:HasBundledNode = $true
$script:HasBundledNpm = $true
foreach ($plural in @('', 'node', 'npm')) {
  $script:PluralTool = $plural
  $script:HasBundledNode = $plural -ne 'npm'
  $script:NpmCalls = 0
  Test-Case "node-gyp bundled preference with plural '$plural' and throwing optional query" {
    $expected = if ($script:HasBundledNode) { $bundledNode } else { $bundledNpm }
    Assert-True ((Find-MominerNodeGyp) -eq $expected) 'Bundled node-gyp preference changed'
    Assert-True ($script:NpmCalls -eq 0) 'Bundled lookup executed the optional global query'
  } $ExpectBuildFailure
}
$script:HasBundledNode = $false
$script:HasBundledNpm = $false
$script:PluralTool = ''
$script:NpmThrows = $false
$script:NpmRoot = $globalRoot
$script:NpmStatus = 0
$script:NpmCalls = 0
Test-Case 'node-gyp custom global root fallback' {
  Assert-True ((Find-MominerNodeGyp) -eq $globalGyp) 'Custom global node-gyp fallback was lost'
  Assert-True ($script:NpmCalls -eq 1) 'Global fallback did not query exactly once'
}
Test-Case 'node-gyp absent or failed global root returns no candidate' {
  foreach ($status in @(0, 3)) {
    $script:NpmStatus = $status
    $script:NpmRoot = if ($status) { $globalRoot } else { $null }
    Assert-True ($null -eq (Find-MominerNodeGyp)) 'An unavailable global root produced a candidate'
  }
}
$script:PluralTool = 'MSBuild.exe'
$script:MsbuildCalls = 0
$buildJobs = 2
Test-Case 'MSBuild plural actual lookup and invocation' {
  . $selectMsbuild
  Assert-True ($script:MsbuildCalls -eq 1 -and $msbuildOutput -eq 'mock build output') 'MSBuild invocation was not scalar'
} $ExpectBuildFailure
$script:Missing = $true
Test-Case 'MSBuild missing actual fallback' {
  Assert-Missing { . $selectMsbuild } 'MSBuild.exe not found on PATH or via vswhere'
  Assert-True ($script:MsbuildCalls -eq 1) 'Missing MSBuild reached execution'
}
$env:GITHUB_ACTIONS = ''
Test-Case 'build trap preserves original ErrorRecord' {
  try { & $trapFixture } catch {
    Assert-True ($_.Exception.Message -eq 'BUILD_ERROR_RECORD_FIXTURE' -and
      $_.FullyQualifiedErrorId -like 'MomBuildFixture*' -and
      $_.CategoryInfo.Category -eq 'InvalidOperation') 'Build trap replaced original error identity'
    return
  }
  throw 'Build trap swallowed the error'
} $ExpectBuildFailure
Write-Output 'EXECUTABLE_DISCOVERY_FIXTURE_PASSED cases=17 native=false installer=false network=false'
