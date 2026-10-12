"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const test = require("node:test");

const repo = path.resolve(__dirname, "..");
const scripts = path.join(repo, ".github/workflows/scripts");
const source = fs.readFileSync(path.join(scripts, "build-windows-multicompiler.ps1"), "utf8");
const oneapi = fs.readFileSync(path.join(scripts, "build-windows.ps1"), "utf8");

test.describe("Windows incremental build orchestration", {concurrency: false}, () => {

  test("Windows packaging reclaims interrupted default ZIP staging and preserves atomic publication", {
    skip: process.platform !== "win32",
  }, () => {
    const packageSource = fs.readFileSync(path.join(scripts, "package-windows.ps1"), "utf8");
    const start = packageSource.indexOf("$archiveDirectory = [IO.Path]::GetDirectoryName($archiveFullPath)");
    const end = packageSource.indexOf("\nWrite-Output $Archive", start);
    const selectionEnd = packageSource.indexOf("\ntry {\n  Compress-Archive", start);
    assert(start > 0 && selectionEnd > start && end > selectionEnd);
    const cleanup = packageSource.match(/^Remove-Item -Recurse -Force release, release-build -ErrorAction SilentlyContinue$/m);
    assert(cleanup);
    // Use one absolute identity across Node and PowerShell, not a TEMP path alias.
    const fixture = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "mom-package-retention-")));
    /** @param {string} value */
    const quote = value => `'${value.replace(/'/g, "''")}'`;
    const driver = path.join(fixture, "driver.ps1");
    const driverSource = String.raw`
$ErrorActionPreference = 'Stop'
Set-Location __DIRECTORY__
$repoRoot = (Get-Location).ProviderPath
$packageDir = 'release/mom-fixture'
$archiveFullPath = Join-Path $repoRoot 'mom-fixture-win.zip'
$publication = __PUBLICATION__
function Assert-Fixture($Condition, [string]$Name) { if (-not $Condition) { throw $Name } }
New-Item -ItemType Directory -Force $packageDir, 'release-build', 'custom' | Out-Null
Set-Content $archiveFullPath 'previous final archive'
$originalHash = (Get-FileHash $archiveFullPath -Algorithm SHA256).Hash
. ([scriptblock]::Create(__SELECTION__))
Assert-Fixture ((Split-Path -Parent $archiveTempPath) -eq (Join-Path $repoRoot 'release-build')) 'default temp escaped staging'
Set-Content $archiveTempPath 'interrupted temporary archive'
$abandoned = $archiveTempPath
$customOrphan = Join-Path $repoRoot 'custom/.mom-archive-0123456789abcdef0123456789abcdef.zip'
Set-Content $customOrphan 'unrelated custom temporary archive'
__CLEANUP__
New-Item -ItemType Directory -Force $packageDir, 'release-build' | Out-Null
Set-Content (Join-Path $packageDir 'fixture-marker.txt') 'current package'
Assert-Fixture (-not (Test-Path -LiteralPath $abandoned)) 'interrupted staging retained on retry'
Assert-Fixture (Test-Path -LiteralPath $customOrphan) 'custom directory swept on retry'
Assert-Fixture ((Get-FileHash $archiveFullPath -Algorithm SHA256).Hash -eq $originalHash) 'retry reset changed final archive'
. ([scriptblock]::Create($publication))
$publishedHash = (Get-FileHash $archiveFullPath -Algorithm SHA256).Hash
Assert-Fixture ($publishedHash -ne $originalHash) 'successful replacement retained old archive'
Assert-Fixture (@(Get-ChildItem release-build -Filter '.mom-archive-*.zip').Count -eq 0) 'successful temp retained'
# Deny delete-sharing to exercise an actual failed File.Replace, not a mocked publisher.
$locked = [IO.File]::Open($archiveFullPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
$rejected = $false
try { . ([scriptblock]::Create($publication)) } catch { $rejected = $true } finally { $locked.Dispose() }
Assert-Fixture $rejected 'locked final archive admitted'
Assert-Fixture ((Get-FileHash $archiveFullPath -Algorithm SHA256).Hash -eq $publishedHash) 'failed publication changed final archive'
Assert-Fixture (@(Get-ChildItem release-build -Filter '.mom-archive-*.zip').Count -eq 0) 'failed publication temp retained'
function Compress-Archive([string]$Path, [string]$DestinationPath) {
  Set-Content $DestinationPath 'partial archive'
  throw 'Fixture compression failure'
}
$rejected = $false
try { . ([scriptblock]::Create($publication)) } catch { $rejected = $_.Exception.Message -eq 'Fixture compression failure' }
Remove-Item Function:Compress-Archive
Assert-Fixture $rejected 'wrong compression failure'
Assert-Fixture ((Get-FileHash $archiveFullPath -Algorithm SHA256).Hash -eq $publishedHash) 'failed compression changed final archive'
Assert-Fixture (@(Get-ChildItem release-build -Filter '.mom-archive-*.zip').Count -eq 0) 'failed compression temp retained'
$archiveFullPath = Join-Path $repoRoot 'custom/mom-fixture-win.zip'
. ([scriptblock]::Create(__SELECTION__))
Assert-Fixture ((Split-Path -Parent $archiveTempPath) -eq (Join-Path $repoRoot 'custom')) 'custom temp not adjacent'
. ([scriptblock]::Create($publication))
Assert-Fixture (Test-Path -LiteralPath $archiveFullPath -PathType Leaf) 'custom publication missing'
Assert-Fixture (Test-Path -LiteralPath $customOrphan) 'custom orphan changed'
Assert-Fixture (-not (Test-Path -LiteralPath $archiveTempPath)) 'custom publication temp retained'
Write-Output 'PACKAGE_RETENTION_FIXTURE_OK'
`.replaceAll("__DIRECTORY__", () => quote(fixture))
      .replaceAll("__PUBLICATION__", () => quote(packageSource.slice(start, end)))
      .replaceAll("__SELECTION__", () => quote(packageSource.slice(start, selectionEnd)))
      .replace("__CLEANUP__", () => cleanup[0]);
    try {
      fs.writeFileSync(driver, driverSource);
      const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver],
        {encoding: "utf8", windowsHide: true, timeout: 30000});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.match(result.stdout, /PACKAGE_RETENTION_FIXTURE_OK/);
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  });

  test("every Windows build entrypoint normalizes before compiler environment setup", () => {
    const combinedInitialize = source.indexOf("Initialize-MomBuildEnvironment $repo");
    const oneapiInitialize = oneapi.indexOf("Initialize-MomBuildEnvironment $repoRoot");
    assert(combinedInitialize > 0 && combinedInitialize < source.indexOf("$supportedCompilers ="));
    assert(oneapiInitialize > 0 && oneapiInitialize < oneapi.indexOf("Invoke-MominerNative { node -v }"));
    for (const name of ["build-sycl-cuda-win.ps1", "build-sycl-adaptivecpp-win.ps1"]) {
      const worker = fs.readFileSync(path.join(scripts, name), "utf8");
      const initialize = worker.indexOf("Initialize-MomBuildEnvironment $RepoRoot");
      assert(initialize > 0 && initialize < worker.indexOf("scripts\\import-vcvars.ps1"));
    }
    const helper = fs.readFileSync(path.join(repo, "scripts/windows-incremental-build.ps1"), "utf8");
    assert.match(helper, /build-environment\.js'\) --json/);
    assert.match(helper, /\$normalizeExit = \$global:LASTEXITCODE/);
    assert(helper.indexOf("if (-not $env:npm_package_json)") < helper.indexOf("$global:LASTEXITCODE = $null"));
    assert(helper.indexOf("if ($normalizeExit -eq 3)") < helper.indexOf("$env:PATH = $normalizedPath"));
  });

  test("Windows environment boundary consumes only successful npm scaffolds and rejects missing native status", {
    skip: process.platform !== "win32",
  }, () => {
    const fixture = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "mom-build-environment-")));
    const helper = fs.readFileSync(path.join(repo, "scripts/windows-incremental-build.ps1"), "utf8");
    const start = helper.indexOf("function Initialize-MomBuildEnvironment");
    const end = helper.indexOf("\nfunction Remove-MomAbandonedBuildReceipts", start);
    assert(start >= 0 && end > start);
    /** @param {string} value */
    const quote = value => `'${value.replace(/'/g, "''")}'`;
    try {
      const driver = path.join(fixture, "driver.ps1");
      fs.writeFileSync(driver, [
        "$ErrorActionPreference = 'Stop'",
        helper.slice(start, end),
        `$repo = ${quote(repo)}`,
        "function node {",
        "  param([string]$Script, [string]$Option)",
        "  if (-not $Script.EndsWith('build-environment.js') -or $Option -ne '--json') { throw 'Wrong normalization CLI' }",
        "  if ($null -ne $script:fixtureExit) { $global:LASTEXITCODE = $script:fixtureExit }",
        "  return $script:fixtureOutput",
        "}",
        "foreach ($shadow in @(0, $null)) {",
        "  $local:LASTEXITCODE = $shadow",
        "  $env:PATH = 'before;; intentional bin '; $env:npm_package_json = 'fixture-anchor'",
        "  $script:fixtureExit = 0; $script:fixtureOutput = '\"canonical;\\u03b2; ;\"'",
        "  Initialize-MomBuildEnvironment $repo",
        "  if ($env:PATH -ne ('canonical;' + [char]0x03b2 + '; ;') -or $env:npm_package_json) { throw 'Successful scaffold not consumed exactly' }",
        "  $script:fixtureExit = 17; Initialize-MomBuildEnvironment $repo",
        "  if ($env:PATH -ne ('canonical;' + [char]0x03b2 + '; ;')) { throw 'Repeated boundary changed normalized PATH' }",
        "  $env:PATH = 'before;; intentional bin '; $env:npm_package_json = 'fixture-anchor'",
        "  $script:fixtureExit = 3; $script:fixtureOutput = $null",
        "  Initialize-MomBuildEnvironment $repo",
        "  if ($env:PATH -ne 'before;; intentional bin ' -or $env:npm_package_json -ne 'fixture-anchor') { throw 'No-op changed caller environment' }",
        "  foreach ($exit in @(17, $null, 0)) {",
        "    $script:fixtureExit = $exit; $script:fixtureOutput = if ($exit -eq 0 -and $null -ne $exit) { '{}' } else { '\"changed\"' }",
        "    $rejected = $false; try { Initialize-MomBuildEnvironment $repo } catch { $rejected = $true }",
        "    if (-not $rejected -or $env:PATH -ne 'before;; intentional bin ' -or $env:npm_package_json -ne 'fixture-anchor') { throw 'Failed normalization changed environment or was admitted' }",
        "  }",
        "  $script:fixtureExit = 0; $script:fixtureOutput = '\"\"'",
        "  Initialize-MomBuildEnvironment $repo",
        "  if ($env:PATH -and $env:PATH -ne '') { throw 'Empty caller PATH rejected' }",
        "}",
      ].join("\n"));
      const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver],
        {encoding: "utf8", timeout: 30000, windowsHide: true});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stdout + result.stderr);
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  });

  test("Windows caches preserve stable node-gyp paths and isolate worker DLL publication", () => {
    const admissionGuard = source.indexOf("Assert-MomBuildArtifactPath (Join-Path $platformRoot '.mom-stage.obj') $platformRoot");
    const publicationGuard = source.indexOf("Assert-MomBuildOutputTree $windowsBuild $platformRoot");
    assert(admissionGuard > 0 && admissionGuard < source.indexOf("Move-Item $platformRoot $platformsHold"));
    assert(publicationGuard > 0 && publicationGuard < source.indexOf("Remove-Item $windowsBuild"));
    assert.match(source, /Move-Item -LiteralPath \$savedWindowsBuild \$platformRoot/);
    assert.match(source, /try \{\s+\$savedWindowsBuild = Join-Path \$platformsHold 'win'/);
    assert.match(source, /\$Compiler -eq 'all' -and \$Backend -eq 'all'/);
    for (const variant of ["sycl-dpcpp-portable", "sycl-dpcpp-cuda", "sycl-dpcpp-spir64",
      "sycl-acpp-cuda", "sycl-acpp-hip"]) {
      assert(source.includes(`-OutDir 'build\\.incremental\\${variant}'`));
      assert(source.includes(`-SyclDll 'build\\.incremental\\${variant}\\sycl.dll'`));
    }
    assert.match(oneapi, /TrackFileAccess=true/);
    assert.doesNotMatch(oneapi, /TrackFileAccess=false/);
    assert(oneapi.includes("'UseMultiToolTask=false'"));
    assert(oneapi.includes("/p:UseMultiToolTask=false /p:TrackFileAccess=true"));
    assert.doesNotMatch(oneapi, /UseMultiToolTask=true/);
    assert.match(oneapi, /if \(\$configureNeeded\) \{\s+Clear-MomMsbuildRelease[^\n]+\n\s+Remove-Item[^\n]+\n\s+Invoke-MominerNative/);
    assert.match(oneapi, /MOM_FORCE_REBUILD -eq '1'/);
    assert.match(oneapi, /if \(\$configureNeeded\) \{\s+Clear-MomMsbuildRelease/);
    assert(oneapi.indexOf("node $nodeGyp install --ensure") < oneapi.indexOf("$toolchainRoots ="));
    assert(oneapi.indexOf("$toolchainRoots =") < oneapi.indexOf("$context = Initialize-MomIncrementalBuild"));
    assert.match(oneapi, /configure --msvs_version=2022 @configureHeaderArgs/);
    assert.match(oneapi, /\$incrementalCachePath \+ '\\'/);
    for (const field of ["Node = $nodeIdentity", "Inventory = $sourceInventory", "Binding = Get-MomBuildHash",
      "Script = Get-MomBuildHash", "$env:npm_config_nodedir", "$env:GYP_DEFINES"]) {
      assert(oneapi.includes(field));
    }
  });

  test("Windows configure gate skips edits but invalidates membership and force changes", {
    skip: process.platform !== "win32",
  }, () => {
    const fixture = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "mom-incremental-configure-")));
    const start = oneapi.indexOf("# Keep node-gyp's generated projects");
    const end = oneapi.indexOf("# Capture output", start);
    assert(start >= 0 && end > start);
    const block = oneapi.slice(start, end);
    const headerFunctionStart = oneapi.indexOf("function Get-MomNodeHeaderInputs");
    const headerFunctionEnd = oneapi.indexOf("\nfunction Find-MominerNodeGyp", headerFunctionStart);
    assert(headerFunctionStart >= 0 && headerFunctionEnd > headerFunctionStart);
    const headerFunction = oneapi.slice(headerFunctionStart, headerFunctionEnd);
    /** @param {string} value */
    const quote = value => `'${value.replace(/'/g, "''")}'`;
    try {
      fs.mkdirSync(path.join(fixture, "native"));
      fs.mkdirSync(path.join(fixture, "build"));
      fs.writeFileSync(path.join(fixture, "native/input.cpp"), "original");
      fs.writeFileSync(path.join(fixture, "binding.gyp"), "{}");
      fs.writeFileSync(path.join(fixture, "build/mom.vcxproj"), "fixture");
      const driver = path.join(fixture, "driver.ps1");
      fs.writeFileSync(driver, [
        "$ErrorActionPreference = 'Stop'",
        `. ${quote(path.join(repo, "scripts/windows-incremental-build.ps1"))}`,
        `. ${quote(path.join(repo, "scripts/windows-msbuild-cache.ps1"))}`,
        headerFunction,
        `Set-Location ${quote(fixture)}`,
        `$repoRoot = ${quote(fixture)}`,
        "$compilerDir = ''; $nodeGyp = ''; $msbuild = ''; $configureCalls = 0; $ensureCalls = 0",
        "$env:LOCALAPPDATA = Join-Path $repoRoot 'localappdata'; $env:npm_config_nodedir = ''; $env:npm_config_devdir = ''; $env:npm_config_target = ''",
        "function icx {}",
        "function Initialize-MomIncrementalBuild { param([string]$CacheDir); if (-not (Test-Path $nodeHeaders)) { throw 'Header fingerprint ran before admission' }; New-Item -ItemType Directory -Force $CacheDir | Out-Null; return 'fixture-tool-context' }",
        "function Invoke-MominerNative {",
        "  param([scriptblock]$Command, [string]$Name)",
        "  if ($Name -eq 'node-gyp configure') { $script:configureCalls++; return }",
        "  if ($Name -ne 'node-gyp install headers') { throw 'Unexpected fixture native call' }",
        "  $script:ensureCalls++; New-Item -ItemType Directory -Force $nodeHeaders | Out-Null",
        "  $marker = Join-Path $nodeHeaders 'installVersion'; if (-not (Test-Path $marker)) { Set-Content $marker '11' }",
        "  $libDir = Join-Path $nodeHeaders 'x64'; New-Item -ItemType Directory -Force $libDir | Out-Null",
        "  $lib = Join-Path $libDir 'node.lib'; if (-not (Test-Path $lib)) { Set-Content $lib 'node-import-library' }",
        "}",
        "$env:MOM_FORCE_REBUILD = ''",
        block,
        "if ($configureCalls -ne 1) { throw 'Missing initial configure' }",
        "Write-MomBuildState $configureState @{ Key = $configureKey; Graph = $generatedGraphKey }",
        "$input = Join-Path $repoRoot 'native\\input.cpp'; $object = Join-Path $repoRoot 'build/Release/input.obj'",
        "New-Item -ItemType Directory -Force (Split-Path -Parent $object) | Out-Null; Set-Content $object 'fixture-object'",
        "Write-MomBuildState $trackedStatePath @{ Sources=$sourceHashes; Outputs=@(@{Path=$object;Hash=Get-MomBuildHash $object;Inputs=$sourceHashes}) }",
        "$nodeLib = Join-Path $nodeHeaders 'x64\\node.lib'; $marker = Join-Path $nodeHeaders 'installVersion'",
        "if ($nodeLib -notin @($nodeHeaderInputs.Path)) { throw 'Selected Node import library was not fingerprinted' }",
        "(Get-Item $nodeLib).LastWriteTimeUtc = [DateTime]::UtcNow.AddMinutes(5); (Get-Item $marker).LastWriteTimeUtc = [DateTime]::UtcNow.AddMinutes(5)",
        block,
        "if ($configureCalls -ne 1) { throw 'Identical downloaded Node SDK bytes with fresh mtimes invalidated cache' }",
        "Set-Content native\\input.cpp 'edited'",
        block,
        "if ($configureCalls -ne 1 -or $msbuildTarget -ne '/t:Build') { throw 'Ordinary edit reconfigured' }",
        "Write-MomBuildState $configureState @{ Key = $configureKey; Graph = $generatedGraphKey }",
        "Set-Content native\\added.inc 'added'",
        block,
        "if ($configureCalls -ne 2 -or $msbuildTarget -ne '/t:Rebuild') { throw 'Inventory change reused configure' }",
        "Write-MomBuildState $configureState @{ Key = $configureKey; Graph = $generatedGraphKey }",
        "$env:MOM_FORCE_REBUILD = '1'",
        block,
        "if ($configureCalls -ne 3) { throw 'Force control ignored' }",
        "$env:MOM_FORCE_REBUILD = ''",
        "Set-Content build\\mom.vcxproj 'corrupt-generated-project'",
        block,
        "if ($configureCalls -ne 4) { throw 'Corrupted generated project reused configure identity' }",
        "$ensureBefore = $ensureCalls; $explicit = Join-Path $repoRoot 'explicit-headers'",
        "New-Item -ItemType Directory $explicit | Out-Null; $env:npm_config_nodedir = $explicit",
        block,
        "if ($ensureCalls -ne $ensureBefore -or $nodeHeaders -ne $explicit) { throw 'Explicit Node headers were downloaded or ignored' }",
        "$env:npm_config_nodedir = ''; $env:npm_config_target = 'v99.1.0'; $env:npm_config_devdir = Join-Path $repoRoot 'selected-cache'",
        block,
        "if ($nodeHeaders -ne (Join-Path $env:npm_config_devdir '99.1.0') -or '--target=99.1.0' -notin $configureHeaderArgs) { throw 'Selected Node target/cache was not admitted' }",
        "$beforeLibraryEdit = $configureCalls; Set-Content (Join-Path $nodeHeaders 'x64\\node.lib') 'changed-node-library'",
        block,
        "if ($configureCalls -ne ($beforeLibraryEdit + 1)) { throw 'Changed Node import library did not invalidate configuration' }",
      ].join("\n"));
      const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver],
        {encoding: "utf8", timeout: 30000, windowsHide: true});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stdout + result.stderr);
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  });

  test("Windows tracker receipts invalidate only matched objects and corrupt outputs", {
    skip: process.platform !== "win32",
  }, () => {
    const fixture = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "mom-incremental-tracker-")));
    /** @param {string} value */
    const quote = value => `'${value.replace(/'/g, "''")}'`;
    try {
      const driver = path.join(fixture, "driver.ps1");
      fs.writeFileSync(driver, [
        "$ErrorActionPreference = 'Stop'",
        `. ${quote(path.join(repo, "scripts/windows-incremental-build.ps1"))}`,
        `. ${quote(path.join(repo, "scripts/windows-msbuild-cache.ps1"))}`,
        `$fixture = ${quote(fixture)}`,
        "$build = Join-Path $fixture 'build'; $logs = Join-Path $build 'Release\\obj\\mom.tlog'",
        "New-Item -ItemType Directory -Force $logs | Out-Null",
        "$a = Join-Path $fixture 'a.cpp'; $b = Join-Path $fixture 'b.cpp'; $header = Join-Path $fixture 'shared.inc'",
        "$oa = Join-Path $build 'Release\\a.obj'; $ob = Join-Path $build 'Release\\b.obj'",
        "Set-Content $a 'a'; Set-Content $b 'b'; Set-Content $header 'header'",
        "$secondHeader = Join-Path $fixture 'other.inc'; Set-Content $secondHeader 'other-header'",
        "Set-Content $oa 'object-a'; Set-Content $ob 'object-b'",
        "Set-Content (Join-Path $logs 'CL.read.1.tlog') @(('^' + $a),$header,('^' + $b),('^' + $a),$secondHeader) -Encoding Unicode",
        "Set-Content (Join-Path $logs 'CL.write.1.tlog') @(('^' + $a),$oa,('^' + $b),$ob) -Encoding Unicode",
        "$rows = @(Get-MomTrackedOutputs $build)",
        "if ($rows.Count -ne 2) { throw 'Tracker did not preserve two independent outputs' }",
        "$sdkDir = Join-Path $build 'toolchains'; New-Item -ItemType Directory $sdkDir | Out-Null",
        "$sdkDll = Join-Path $sdkDir 'victim.dll'; Set-Content $sdkDll 'keep SDK'",
        "Add-Content (Join-Path $logs 'CL.write.1.tlog') $sdkDll -Encoding Unicode",
        "$rows = @(Get-MomTrackedOutputs $build)",
        "if ($rows.Count -ne 2 -or $sdkDll -in @($rows.Path)) { throw 'Tracker admitted an SDK output' }",
        "$rejected = $false",
        "try { Repair-MomTrackedOutputs @{Outputs=@(@{Path=$oa;Hash='invalid';Inputs=@()},@{Path=$sdkDll;Hash='invalid';Inputs=@()})} $build } catch { $rejected=$true }",
        "if (-not $rejected -or (Get-Content $sdkDll) -ne 'keep SDK' -or (Get-Content $oa) -ne 'object-a') { throw 'Invalid receipt mutated SDK or Release before rejection' }",
        "$aRow = @($rows | Where-Object { $_.Path -eq $oa })[0]",
        "if ($header -notin @($aRow.Inputs.Path) -or $secondHeader -notin @($aRow.Inputs.Path)) { throw 'Duplicate tracker roots discarded dependencies' }",
        "$mtime = (Get-Item $header).LastWriteTimeUtc",
        "Set-Content $header 'backdated-header'; (Get-Item $header).LastWriteTimeUtc = $mtime",
        "Repair-MomTrackedOutputs @{Outputs=$rows} $build",
        "if ((Test-Path $oa) -or -not (Test-Path $ob)) { throw 'Dependency edit invalidated wrong objects' }",
        "Set-Content $oa 'rebuilt-object-a'",
        "$rows = @(Get-MomTrackedOutputs $build)",
        "Set-Content $ob 'corrupt-object-b'",
        "Repair-MomTrackedOutputs @{Outputs=$rows} $build",
        "if (-not (Test-Path $oa) -or (Test-Path $ob)) { throw 'Corruption invalidated wrong outputs' }",
        "Set-Content $ob 'rebuilt-object-b'; Remove-Item -LiteralPath $secondHeader",
        "$rejected = $false; try { Get-MomTrackedOutputs $build | Out-Null } catch { $rejected = $true }",
        "if (-not $rejected) { throw 'Missing consumed dependency narrowed signed graph' }",
        "$outside = Join-Path $fixture 'outside.obj'; Set-Content $outside 'keep'",
        "$escaped = $build + '\\..\\outside.obj'; $rejected = $false",
        "try { Repair-MomTrackedOutputs @{Outputs=@(@{Path=$escaped;Hash='invalid';Inputs=@()})} $build } catch { $rejected=$true }",
        "if (-not $rejected -or -not (Test-Path $outside)) { throw 'Unsafe tracker output path was accepted' }",
        "$outsideDir = Join-Path $fixture 'outside'; New-Item -ItemType Directory $outsideDir | Out-Null",
        "$outsideObject = Join-Path $outsideDir 'victim.obj'; Set-Content $outsideObject 'keep'",
        "$junction = Join-Path $build 'Release\\redirected'",
        "New-Item -ItemType Junction -Path $junction -Value $outsideDir | Out-Null",
        "$redirected = Join-Path $junction 'victim.obj'; $rejected = $false",
        "try { Repair-MomTrackedOutputs @{Outputs=@(@{Path=$redirected;Hash='invalid';Inputs=@()})} $build } catch { $rejected=$true }",
        "if (-not $rejected -or (Get-Content $outsideObject) -ne 'keep') { throw 'Junction output escaped build ownership' }",
      ].join("\n"));
      const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver],
        {encoding: "utf8", timeout: 30000, windowsHide: true});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stdout + result.stderr);
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  });

  test("oneAPI graph refresh clears only local generated Release and refuses nested junctions", {
    skip: process.platform !== "win32",
  }, () => {
    const fixture = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "mom-msbuild-retention-")));
    /** @param {string} value @returns {string} */
    const quote = value => "'" + value.replaceAll("'", "''") + "'";
    const driver = path.join(fixture, "driver.ps1");
    const driverSource = String.raw`
$ErrorActionPreference = 'Stop'
. __HELPER__
Set-Location __DIRECTORY__
function Assert-Fixture($Condition, [string]$Name) { if (-not $Condition) { throw $Name } }
$build = Join-Path __DIRECTORY__ 'build'
$expectedBuild = [IO.Path]::GetFullPath((Join-Path (Get-Location).ProviderPath 'build'))
Assert-Fixture ($build -eq $expectedBuild) "fixture build identity changed: expected=$expectedBuild actual=$build"
$release = Join-Path $build 'Release'
New-Item -ItemType Directory -Force (Join-Path $release 'obj/old-target') | Out-Null
Set-Content (Join-Path $release 'obj/old-target/orphan.obj') 'obsolete object'
Set-Content (Join-Path $release 'obj/old-target/orphan.tlog') 'obsolete tracker'
foreach ($leaf in @('.incremental/sycl-dpcpp-cuda','compilers/preserved','cutlass','.acpp')) {
  New-Item -ItemType Directory -Force (Join-Path $build $leaf) | Out-Null
  Set-Content (Join-Path $build "$leaf/marker") 'preserve'
}
Clear-MomMsbuildRelease $build
Assert-Fixture (-not (Test-Path -LiteralPath $release)) 'generated Release retained'
foreach ($leaf in @('.incremental/sycl-dpcpp-cuda','compilers/preserved','cutlass','.acpp')) {
  Assert-Fixture ((Get-Content (Join-Path $build "$leaf/marker")) -eq 'preserve') 'unrelated tree removed'
}
$outside = Join-Path __DIRECTORY__ 'outside'
New-Item -ItemType Directory -Force $outside, (Join-Path $release 'obj') | Out-Null
Set-Content (Join-Path $outside 'marker') 'outside'
Set-Content (Join-Path $release 'retained.obj') 'release preflight'
New-Item -ItemType Junction -Path (Join-Path $release 'obj/redirected') -Value $outside | Out-Null
$rejected = $false
try { Clear-MomMsbuildRelease $build } catch { $rejected = $true }
Assert-Fixture $rejected 'nested junction admitted'
Assert-Fixture ((Get-Content (Join-Path $outside 'marker')) -eq 'outside') 'junction target removed'
Assert-Fixture ((Get-Content (Join-Path $release 'retained.obj')) -eq 'release preflight') 'preflight partially deleted Release'
$rejected = $false
try { Clear-MomMsbuildRelease $outside } catch { $rejected = $true }
Assert-Fixture $rejected 'foreign build root admitted'
Assert-Fixture ((Get-Content (Join-Path $outside 'marker')) -eq 'outside') 'foreign output removed'
Write-Output 'MSBUILD_RETENTION_FIXTURE_OK'
`.replaceAll("__HELPER__", () => quote(path.join(repo, "scripts/windows-msbuild-cache.ps1")))
      .replaceAll("__DIRECTORY__", () => quote(fixture));
    try {
      fs.writeFileSync(driver, driverSource);
      const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver],
        {encoding: "utf8", windowsHide: true, timeout: 30000});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.match(result.stdout, /MSBUILD_RETENTION_FIXTURE_OK/);
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  });

  test("Windows staging rejects root and restored destination junctions before mutation", {
    skip: process.platform !== "win32",
  }, () => {
    const fixture = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "mom-staging-boundary-")));
    /** @param {string} value @returns {string} */
    const quote = value => "'" + value.replaceAll("'", "''") + "'";
    const admission = source.slice(source.indexOf("$platformRoot ="), source.indexOf("function Resolve-RepoPath"));
    const publishStart = source.indexOf("$windowsBuild = Join-Path $platformRoot 'win'");
    const publication = source.slice(publishStart, source.indexOf("\n  }\n}", publishStart));
    assert(admission.includes("Assert-MomBuildArtifactPath"));
    assert(publication.includes("Assert-MomBuildOutputTree"));
    const driver = path.join(fixture, "driver.ps1");
    const driverSource = String.raw`
$ErrorActionPreference = 'Stop'
. __HELPER__
function Assert-Fixture($Condition, [string]$Name) { if (-not $Condition) { throw $Name } }
foreach ($kind in @('ordinary','root-junction')) {
  $repo = Join-Path __DIRECTORY__ $kind
  $outside = Join-Path __DIRECTORY__ "outside-$kind"
  New-Item -ItemType Directory -Force $repo, (Join-Path $outside 'win') | Out-Null
  Set-Content (Join-Path $outside 'win/sentinel') 'external'
  if ($kind -eq 'root-junction') { New-Item -ItemType Junction -Path (Join-Path $repo 'build') -Value $outside | Out-Null }
  else { New-Item -ItemType Directory (Join-Path $repo 'build') | Out-Null }
  $rejected = $false
  try { & ([scriptblock]::Create(__ADMISSION__)) } catch { $rejected = $true }
  Assert-Fixture ($rejected -eq ($kind -eq 'root-junction')) 'initial root admission changed'
  Assert-Fixture ((Get-Content (Join-Path $outside 'win/sentinel')) -eq 'external') 'external root mutated'
  Assert-Fixture (-not (Test-Path (Join-Path $repo 'build-platforms-hold'))) 'root staged before validation'
}
foreach ($kind in @('ordinary','destination-junction','nested-junction')) {
  $platformRoot = Join-Path __DIRECTORY__ "publish-$kind"
  $outside = Join-Path __DIRECTORY__ "outside-publish-$kind"
  $completedBuild = Join-Path __DIRECTORY__ "completed-$kind"
  New-Item -ItemType Directory -Force $platformRoot, $outside, $completedBuild | Out-Null
  Set-Content (Join-Path $outside 'sentinel') 'external'
  Set-Content (Join-Path $completedBuild 'new-output') 'pending'
  $destination = Join-Path $platformRoot 'win'
  if ($kind -eq 'destination-junction') { New-Item -ItemType Junction -Path $destination -Value $outside | Out-Null }
  else {
    New-Item -ItemType Directory $destination | Out-Null
    Set-Content (Join-Path $destination 'old-output') 'retained'
    if ($kind -eq 'nested-junction') { New-Item -ItemType Junction -Path (Join-Path $destination 'nested') -Value $outside | Out-Null }
  }
  $rejected = $false
  try { & ([scriptblock]::Create(__PUBLICATION__)) } catch { $rejected = $true }
  Assert-Fixture ($rejected -eq ($kind -ne 'ordinary')) 'restored destination admission changed'
  Assert-Fixture ((Get-Content (Join-Path $outside 'sentinel')) -eq 'external') 'external destination mutated'
  if ($rejected) {
    Assert-Fixture ((Get-Content (Join-Path $completedBuild 'new-output')) -eq 'pending') 'pending tree moved before validation'
    if ($kind -eq 'nested-junction') { Assert-Fixture ((Get-Content (Join-Path $destination 'old-output')) -eq 'retained') 'destination partially deleted' }
  } else { Assert-Fixture ((Get-Content (Join-Path $destination 'new-output')) -eq 'pending') 'ordinary publication failed' }
}
Write-Output 'STAGING_BOUNDARY_FIXTURE_OK'
`.replaceAll("__HELPER__", () => quote(path.join(repo, "scripts/windows-msbuild-cache.ps1")))
      .replaceAll("__DIRECTORY__", () => quote(fixture)).replace("__ADMISSION__", () => quote(admission))
      .replace("__PUBLICATION__", () => quote(publication));
    try {
      fs.writeFileSync(driver, driverSource);
      const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver],
        {encoding: "utf8", windowsHide: true, timeout: 30000});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.match(result.stdout, /STAGING_BOUNDARY_FIXTURE_OK/);
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  });

  for (const failure of ["none", "compile", "after-root-staging", "after-win-restore"]) {
    test(`Windows orchestration restores objects and the other platform after ${failure}`, {
      skip: process.platform !== "win32",
    }, () => {
      const fixture = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "mom-incremental-orchestration-")));
      const dir = path.join(fixture, ".github/workflows/scripts");
      fs.mkdirSync(dir, {recursive: true});
      fs.mkdirSync(path.join(fixture, "scripts"));
      fs.mkdirSync(path.join(fixture, "build/win/Release"), {recursive: true});
      fs.mkdirSync(path.join(fixture, "build/win/obj"));
      fs.mkdirSync(path.join(fixture, "build/win/.incremental/oneapi"), {recursive: true});
      fs.mkdirSync(path.join(fixture, "build/win/compilers/preserved"), {recursive: true});
      fs.mkdirSync(path.join(fixture, "build/lin"));
      fs.writeFileSync(path.join(fixture, "build/win/obj/input.obj"), "object");
      fs.writeFileSync(path.join(fixture, "build/win/obj/input.tlog"), "tracker-log");
      fs.writeFileSync(path.join(fixture, "build/win/.incremental/oneapi/configure.json"), "configure-receipt");
      fs.writeFileSync(path.join(fixture, "build/win/compilers/preserved/sycl.dll"), "other-worker");
      fs.writeFileSync(path.join(fixture, "build/lin/marker"), "linux");
      fs.writeFileSync(path.join(dir, "windows-dll-deps.ps1"), [
        "function Copy-MominerOptionalRuntimeFiles {",
        "  param([string]$PackageDir, [string[]]$SourceRoots)",
        "  if ($PackageDir -ne (Join-Path $out 'oneapi') -or $SourceRoots.Count -ne 1 -or $SourceRoots[0] -ne (Join-Path $repo 'fixture-oneapi-bin')) { throw 'Unexpected fixture runtime export' }",
        "}",
        "function Copy-MominerDllClosure {",
        "  param([string]$PackageDir, [string[]]$EntryPaths)",
        "  if ($PackageDir -ne (Join-Path $out 'oneapi') -or $EntryPaths.Count -ne 2) { throw 'Unexpected fixture DLL closure' }",
        "  foreach ($entry in $EntryPaths) { if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) { throw 'Missing fixture runtime entry' } }",
        "}",
      ].join("\n"));
      fs.writeFileSync(path.join(fixture, "scripts/windows-install-helpers.ps1"),
        "function Get-MominerOneApiBinDirs { Join-Path $repo 'fixture-oneapi-bin' }\n");
      for (const name of ["windows-msbuild-cache.ps1", "windows-incremental-build.ps1", "build-environment.js"]) {
        fs.copyFileSync(path.join(repo, "scripts", name), path.join(fixture, "scripts", name));
      }
      const failureAnchor = failure === "after-root-staging" ? "$savedWindowsBuild = Join-Path $platformsHold 'win'"
        : failure === "after-win-restore" ? "Move-Item -LiteralPath $savedWindowsBuild $platformRoot" : null;
      if (failureAnchor) {
        assert.equal(source.split(failureAnchor).length, 2, "Failure injection must target exactly one staging operation");
      }
      const fixtureSource = failureAnchor ? source.replace(failureAnchor,
        () => failureAnchor + "\n  throw 'Fixture staging failure'") : source;
      fs.writeFileSync(path.join(dir, "build-windows-multicompiler.ps1"), fixtureSource);
      fs.writeFileSync(path.join(dir, "build-windows.ps1"), [
        "$ErrorActionPreference = 'Stop'",
        "if (-not (Test-Path 'build\\obj\\input.obj')) { throw 'Object tree was not restored' }",
        "if (Test-Path 'build\\win') { throw 'Nested source-relative build tree' }",
        "Set-Content build\\Release\\mom.node 'current-host'",
        "Set-Content build\\Release\\sycl.dll 'current-oneapi'",
        ...(failure === "compile" ? ["throw 'Fixture compilation failure'"] : []),
      ].join("\n"));
      try {
        const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
          "-File", path.join(dir, "build-windows-multicompiler.ps1"), "-Backend", "intel", "-Compiler", "oneapi"],
        {encoding: "utf8", timeout: 30000, windowsHide: true});
        assert.equal(result.error, undefined);
        assert.equal(result.status === 0, failure === "none", result.stdout + result.stderr);
        if (failureAnchor) { assert.match(result.stdout + result.stderr, /Fixture staging failure/); }
        if (failure === "compile") { assert.match(result.stdout + result.stderr, /Fixture compilation failure/); }
        assert.equal(fs.readFileSync(path.join(fixture, "build/lin/marker"), "utf8"), "linux");
        assert.equal(fs.readFileSync(path.join(fixture, "build/win/obj/input.obj"), "utf8"), "object");
        assert.equal(fs.readFileSync(path.join(fixture, "build/win/obj/input.tlog"), "utf8"), "tracker-log");
        assert.equal(fs.readFileSync(path.join(fixture, "build/win/.incremental/oneapi/configure.json"), "utf8"), "configure-receipt");
        assert.equal(fs.readFileSync(path.join(fixture, "build/win/compilers/preserved/sycl.dll"), "utf8"), "other-worker");
        assert(!fs.existsSync(path.join(fixture, "build-platforms-hold")));
        assert(!fs.existsSync(path.join(fixture, "build-windows-completed")));
        if (failure === "none") {
          assert.equal(fs.readFileSync(path.join(fixture, "build/win/compilers/oneapi/sycl.dll"), "utf8").trim(), "current-oneapi");
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });
  }
});
