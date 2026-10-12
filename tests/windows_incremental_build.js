"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const test = require("node:test");

const repo = path.resolve(__dirname, "..");
const helper = path.join(repo, "scripts/windows-incremental-build.ps1");

test.describe("Windows incremental builds", {concurrency: false}, () => {
  test("MSBuild signs outputs only after complete tracker admission", () => {
    const source = fs.readFileSync(path.join(repo, ".github/workflows/scripts/build-windows.ps1"), "utf8");
    const coverage = source.indexOf("Assert-MomTrackedOutputCoverage $trackedOutputs");
    const publish = source.indexOf("Write-MomBuildState $trackedStatePath");
    assert(coverage > 0 && publish > coverage);
    assert.match(source.slice(coverage, publish), /catch \{\s+Remove-Item -LiteralPath \$trackedStatePath[^\n]+\n\s+throw/);
    const tracker = fs.readFileSync(path.join(repo, "scripts/windows-msbuild-cache.ps1"), "utf8");
    const parser = tracker.slice(tracker.indexOf("function Get-MomTrackedOutputs"));
    assert(parser.indexOf("Assert-MomBuildOutputTree $release $BuildRoot") < parser.indexOf("-File -Recurse"));
  });

  test("MSBuild tracker admission rejects missing intermediates and empty inputs before publication", {skip: process.platform !== "win32"}, () => {
    // Canonicalize only the new private root; leave deliberate junction/escape paths untouched.
    const fixture = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-tracker-coverage-")));
    /** @param {string} value @returns {string} */
    const quote = value => "'" + value.replaceAll("'", "''") + "'";
    const source = fs.readFileSync(path.join(repo, ".github/workflows/scripts/build-windows.ps1"), "utf8").replace(/\r\n/g, "\n");
    const start = source.indexOf("try {\n  $trackedOutputs = @(Get-MomTrackedOutputs");
    const end = source.indexOf("Write-MomBuildState $trackedStatePath", start);
    assert(start >= 0 && end > start, "tracker publication boundary missing");
    const block = source.slice(start, source.indexOf("\n", end) < 0 ? source.length : source.indexOf("\n", end));
    const driver = path.join(fixture, "driver.ps1");
    const script = String.raw`
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. __HELPER__
$repoRoot = __ROOT__
$canonicalRoot = [IO.Path]::GetFullPath($repoRoot)
if ($canonicalRoot -ne $repoRoot) { throw "fixture root identity changed: expected=$repoRoot actual=$canonicalRoot" }
$build = Join-Path $repoRoot 'build'
$release = Join-Path $build 'Release'
$logs = Join-Path $release 'obj/tracker'
New-Item -ItemType Directory -Force $logs | Out-Null
$source = Join-Path $repoRoot 'input.cpp'
Set-Content -LiteralPath $source 'source'
$outputs = @('a.obj', 'b.pch', 'c.res') | ForEach-Object { Join-Path $release "obj/$_" }
foreach ($output in $outputs) { Set-Content -LiteralPath $output 'compiled' }
foreach ($name in @('mom.node', 'sycl.dll')) { Set-Content -LiteralPath (Join-Path $release $name) 'linked' }
$read = Join-Path $logs 'CL.read.1.tlog'
$write = Join-Path $logs 'CL.write.1.tlog'
Set-Content -LiteralPath $read -Value @(('^' + $source), $source) -Encoding Unicode
$trackedStatePath = Join-Path $repoRoot 'outputs.json'
$currentSources = @(@{ Path = $source; Hash = Get-MomBuildHash $source })
function Assert-Fixture($Condition, [string]$Message) { if (-not $Condition) { throw $Message } }
function Admit-Fixture([bool]$Expected) {
  Set-Content -LiteralPath $trackedStatePath 'old receipt'
  $rejected = $false
  try { __BLOCK__ } catch { $rejected = $true }
  Assert-Fixture ($rejected -ne $Expected) 'unexpected tracker admission verdict'
  Assert-Fixture ((Test-Path -LiteralPath $trackedStatePath) -eq $Expected) 'invalid tracker receipt retained or published'
}
# Positive control includes every object/precompiled header/resource output.
Set-Content -LiteralPath $write -Value (@('^' + $source) + $outputs) -Encoding Unicode
Admit-Fixture $true
Assert-Fixture (@((Read-MomBuildState $trackedStatePath).Outputs).Count -eq 5) 'complete graph receipt missing outputs'
$rows = @(Get-MomTrackedOutputs $build)
foreach ($row in $rows) { foreach ($inputRow in $row.Inputs) {
  Assert-Fixture ($inputRow.Path -is [string]) 'dependency path is not a string'
  Assert-Fixture (@($inputRow.Path.PSObject.Properties | Where-Object { $_.MemberType -eq 'NoteProperty' }).Count -eq 0) 'dependency path retained provider metadata'
} }
$json = @{ Outputs = $rows } | ConvertTo-Json -Depth 12 -Compress
Assert-Fixture ($json.Length -lt 10000) 'tiny tracker receipt expanded unexpectedly'
$parsed = $json | ConvertFrom-Json
Assert-Fixture (@($parsed.Outputs).Count -eq 3) 'serialized tracker outputs changed'
foreach ($row in $parsed.Outputs) { foreach ($inputRow in $row.Inputs) {
  Assert-Fixture ($inputRow.Path -is [string] -and $inputRow.Hash -match '^[A-F0-9]{64}$') 'serialized dependency schema changed'
} }
# Each generated extension is required, not only .obj; unmatched composite roots fail closed too.
foreach ($missing in $outputs) {
  Set-Content -LiteralPath $write -Value (@('^' + $source) + @($outputs | Where-Object { $_ -ne $missing })) -Encoding Unicode
  Admit-Fixture $false
}
Set-Content -LiteralPath $write -Value (@('^' + $source + '|' + $source) + $outputs) -Encoding Unicode
Admit-Fixture $false
$emptyInputs = @($outputs | ForEach-Object { @{ Path = $_; Inputs = @() } })
$rejected = $false
try { Assert-MomTrackedOutputCoverage $emptyInputs $build } catch { $rejected = $true }
Assert-Fixture $rejected 'empty tracked inputs admitted'
$outside = Join-Path $repoRoot 'outside'
New-Item -ItemType Directory $outside | Out-Null
$sentinel = Join-Path $outside 'sentinel.obj'
Set-Content -LiteralPath $sentinel 'outside sentinel'
$sentinelHash = Get-MomBuildHash $sentinel
$junction = Join-Path $release 'junction'
& cmd.exe /d /c ('mklink /J "{0}" "{1}"' -f $junction, $outside) | Out-Null
Assert-Fixture ($global:LASTEXITCODE -eq 0) 'junction fixture failed'
Set-Content -LiteralPath $write -Value (@('^' + $source) + $outputs) -Encoding Unicode
Admit-Fixture $false
Assert-Fixture ((Get-MomBuildHash $sentinel) -eq $sentinelHash) 'outside sentinel changed'
[IO.Directory]::Delete($junction)
Admit-Fixture $true
foreach ($output in $outputs) { Remove-Item -LiteralPath $output }
Remove-Item -LiteralPath $read, $write
Admit-Fixture $false
Write-Output 'TRACKER_COVERAGE_FIXTURE_OK'
`.replaceAll("__HELPER__", () => quote(helper)).replaceAll("__ROOT__", () => quote(fixture)).replace("__BLOCK__", () => block);
    try {
      fs.writeFileSync(driver, script);
      const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver],
        {encoding: "utf8", windowsHide: true, timeout: 30000});
      assert.equal(result.error, undefined, result.stdout + result.stderr);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.match(result.stdout, /TRACKER_COVERAGE_FIXTURE_OK/);
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  });

  for (const shadow of ["0", "$null"]) {
    for (const status of [0, 17, null]) {
      test(`Windows incremental linker status ${status ?? "missing"} ignores local shadow ${shadow}`, {skip: process.platform !== "win32"}, () => {
        const fixture = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-link-status-")));
        /** @param {string} value @returns {string} */
        const quote = value => "'" + value.replaceAll("'", "''") + "'";
        const driver = path.join(fixture, "driver.ps1");
        const script = String.raw`
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. __HELPER__
$object = Join-Path __ROOT__ 'source.obj'
$output = Join-Path __ROOT__ 'worker.dll'
$state = "$output.json"
Set-Content -LiteralPath $object 'object'
function Invoke-FixtureLink {
  Set-Content -LiteralPath $output 'linked'
  __NATIVE__
}
function Invoke-ScopedLink {
  $LASTEXITCODE = __SHADOW__
  $rejected = $false
  try { Invoke-MomIncrementalLink 'Invoke-FixtureLink' @() @($object) $output $state 'context' } catch { $rejected = $true }
  if ($rejected -ne __REJECTED__) { throw 'incorrect link status admission' }
  if ((Test-Path -LiteralPath $state) -ne (-not __REJECTED__)) { throw 'incorrect link receipt publication' }
  # Negative control: the old unqualified boundary reads this local shadow.
  $global:LASTEXITCODE = $null
  Invoke-FixtureLink
  if ($LASTEXITCODE -ne __SHADOW__) { throw 'negative control lost local shadow' }
  if ($global:LASTEXITCODE -ne __STATUS__) { throw 'native fixture status mismatch' }
}
Invoke-ScopedLink
Write-Output 'LINK_STATUS_FIXTURE_OK'
`.replace("__HELPER__", () => quote(helper)).replaceAll("__ROOT__", () => quote(fixture))
          .replaceAll("__SHADOW__", () => shadow).replaceAll("__REJECTED__", () => status === 0 ? "$false" : "$true")
          .replace("__NATIVE__", () => status === null ? "# No native status was produced."
            : `& powershell.exe -NoProfile -NonInteractive -Command 'exit ${status}'`)
          .replace("__STATUS__", () => status === null ? "$null" : String(status));
        try {
          fs.writeFileSync(driver, script);
          const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver],
            {encoding: "utf8", windowsHide: true, timeout: 30000});
          assert.equal(result.error, undefined, result.stdout + result.stderr);
          assert.equal(result.status, 0, result.stdout + result.stderr);
          assert.match(result.stdout, /LINK_STATUS_FIXTURE_OK/);
        } finally {
          fs.rmSync(fixture, {recursive: true, force: true});
        }
      });
    }
  }

  test("fingerprint keys are identical across fresh PowerShell processes", {skip: process.platform !== "win32"}, () => {
    /** @param {string} expression @returns {string} */
    function key(expression) {
      const command = `. '${helper.replaceAll("'", "''")}'; Get-MomBuildKey (${expression})`;
      const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", command],
        {encoding: "utf8", windowsHide: true, timeout: 30000});
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.match(result.stdout.trim(), /^[A-F0-9]{64}$/);
      return result.stdout.trim();
    }
    assert.equal(key("@{ b=@(@{ y=2; x=1 }); a='value' }"), key("@{ a='value'; b=@(@{ x=1; y=2 }) }"));
  });

  test("Windows SYCL workers keep per-variant objects and use compiler dependencies", () => {
    const source = fs.readFileSync(helper, "utf8");
    assert.match(source, /'-MD', '-MF', \$dep, '-MT', 'mom'/);
    assert.match(source, /Get-MomBuildHash \$_\.Path/);
    assert.match(source, /MOM_FORCE_REBUILD/);
    for (const name of ["build-sycl-cuda-win.ps1", "build-sycl-adaptivecpp-win.ps1"]) {
      const worker = fs.readFileSync(path.join(repo, ".github/workflows/scripts", name), "utf8");
      assert.match(worker, /build\\\.incremental\\sycl-/);
      assert.match(worker, /Invoke-MomIncrementalLink/);
      assert.match(worker, /Complete-MomCompileTask/);
      assert(worker.indexOf("Remove-MomObsoleteCompileOutputs") > worker.indexOf("Invoke-MomIncrementalLink"));
      assert.doesNotMatch(worker, /Remove-Item \$obj -Recurse|Remove-Item -Recurse -Force \$obj/);
    }
  });

  test("Windows publication guards precede output mutations", () => {
    for (const name of ["build-sycl-cuda-win.ps1", "build-sycl-adaptivecpp-win.ps1"]) {
      const source = fs.readFileSync(path.join(repo, ".github/workflows/scripts", name), "utf8");
      const outputGuard = source.indexOf("Assert-MomBuildArtifactPath $publishedOutput $buildRoot");
      assert(outputGuard > 0);
      assert(outputGuard < source.indexOf("New-Item -ItemType Directory -Force $obj"));
      assert(outputGuard < source.indexOf("Remove-Item -LiteralPath $publishedOutput"));
    }
    const source = fs.readFileSync(path.join(repo, ".github/workflows/scripts/build-windows-multicompiler.ps1"), "utf8");
    assert(source.indexOf("Assert-MomBuildOutputTree $dest $platformRoot") < source.indexOf("Remove-Item -LiteralPath $dest"));
    assert(source.indexOf("Assert-MomBuildOutputTree $out $platformRoot") < source.indexOf("Remove-Item -LiteralPath $out"));
  });

  test("Windows publication rejects junction escapes without changing external sentinels", {skip: process.platform !== "win32"}, () => {
    const fixture = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-publication-")));
    /** @param {string} value @returns {string} */
    const quote = value => "'" + value.replaceAll("'", "''") + "'";
    /** @param {string} source @param {string} startText @param {string} endText @returns {string} */
    const extract = (source, startText, endText) => {
      const start = source.indexOf(startText), end = source.indexOf(endText, start);
      assert(start >= 0 && end > start, "production publication boundary is missing");
      return source.slice(start, end);
    };
    const cuda = fs.readFileSync(path.join(repo, ".github/workflows/scripts/build-sycl-cuda-win.ps1"), "utf8");
    const acpp = fs.readFileSync(path.join(repo, ".github/workflows/scripts/build-sycl-adaptivecpp-win.ps1"), "utf8");
    const multi = fs.readFileSync(path.join(repo, ".github/workflows/scripts/build-windows-multicompiler.ps1"), "utf8");
    const blocks = [extract(cuda, "$variant = if", "$buildContext = Initialize-"),
      extract(acpp, "$obj = ", "$python = ")];
    const save = extract(multi, "function Save-Compiler(", "function Save-DpcppRuntime(");
    const reset = extract(multi, "$out = Join-Path $repo 'build\\compilers'", "  # node-gyp requires");
    const driver = path.join(fixture, "driver.ps1");
    const source = String.raw`
$ErrorActionPreference = 'Stop'
. __HELPER__
function Assert-Fixture($Condition, [string]$Name) { if (-not $Condition) { throw $Name } }
function Resolve-RepoPath([string]$Path) { Resolve-MomBuildPath $Path }
__SAVE__
$blocks = @(__BLOCKS__)
$index = 0
foreach ($block in $blocks) {
  $kinds = @('ordinary','published-junction','cache-junction')
  if ($block.Contains('sycl-acpp-')) { $kinds += 'relative-repo' }
  foreach ($kind in $kinds) {
    $index++
    $RepoRoot = Join-Path __DIRECTORY__ "worker-$index"
    New-Item -ItemType Directory -Force (Join-Path $RepoRoot 'build') | Out-Null
    Set-Location $RepoRoot
    $outside = Join-Path __DIRECTORY__ "outside-worker-$index"
    New-Item -ItemType Directory $outside | Out-Null
    Set-Content (Join-Path $outside 'sycl.dll') 'external worker sentinel'
    $OutDir = 'build/Release'; $Backend = 'hip'; $PortableOpencl = $false; $withCuda = $false
    if ($kind -eq 'relative-repo') { $RepoRoot = '.' }
    if ($kind -eq 'published-junction') {
      New-Item -ItemType Junction -Path 'build/Release' -Value $outside | Out-Null
    } elseif ($kind -eq 'cache-junction') {
      New-Item -ItemType Junction -Path 'build/.incremental' -Value $outside | Out-Null
    }
    $rejected = $false
    try { & ([scriptblock]::Create($block)) } catch { $rejected = $true }
    Assert-Fixture ($rejected -eq ($kind -notin @('ordinary','relative-repo'))) 'worker publication admission changed'
    Assert-Fixture ((Get-Content (Join-Path $outside 'sycl.dll')) -eq 'external worker sentinel') 'external DLL mutated'
    Assert-Fixture (@(Get-ChildItem $outside -Force).Count -eq 1) 'worker created external output directories'
    if ($kind -in @('ordinary','relative-repo')) { Assert-Fixture (Test-Path 'build/Release') 'ordinary output directory missing' }
  }
}
foreach ($kind in @('ordinary','ancestor-junction','destination-junction','nested-junction','root-nested-junction','root-ordinary')) {
  $repo = Join-Path __DIRECTORY__ $kind
  $platformRoot = Join-Path $repo 'build'
  $out = Join-Path $platformRoot 'compilers'
  $dest = Join-Path $out 'oneapi'
  New-Item -ItemType Directory -Force (Join-Path $platformRoot 'Release') | Out-Null
  Set-Location $repo
  Set-Content 'build/Release/mom.node' 'new node'
  Set-Content 'build/Release/sycl.dll' 'new DLL'
  $outside = Join-Path __DIRECTORY__ "outside-$kind"
  New-Item -ItemType Directory -Force (Join-Path $outside 'oneapi') | Out-Null
  Set-Content (Join-Path $outside 'oneapi/mom.node') 'external compiler sentinel'
  if ($kind -eq 'ancestor-junction') {
    New-Item -ItemType Junction -Path $out -Value $outside | Out-Null
  } else {
    New-Item -ItemType Directory -Force $out | Out-Null
    if ($kind -eq 'destination-junction') {
      New-Item -ItemType Junction -Path $dest -Value (Join-Path $outside 'oneapi') | Out-Null
    } else {
      New-Item -ItemType Directory $dest | Out-Null
      Set-Content (Join-Path $dest 'old.node') 'old local output'
      if ($kind -in @('nested-junction','root-nested-junction')) {
        New-Item -ItemType Junction -Path (Join-Path $dest 'nested') -Value $outside | Out-Null
      }
    }
  }
  $rejected = $false; $Compiler = 'all'; $Backend = 'all'
  try {
    if ($kind -like 'root-*') { & ([scriptblock]::Create(__RESET__)) }
    else { Save-Compiler oneapi }
  } catch { $rejected = $true }
  Assert-Fixture ($rejected -eq ($kind -notin @('ordinary','root-ordinary'))) 'compiler publication admission changed'
  Assert-Fixture ((Get-Content (Join-Path $outside 'oneapi/mom.node')) -eq 'external compiler sentinel') 'external compiler output mutated'
  Assert-Fixture (@(Get-ChildItem (Join-Path $outside 'oneapi') -Force).Count -eq 1) 'compiler copied external output'
  if ($rejected -and $kind -in @('nested-junction','root-nested-junction')) {
    Assert-Fixture ((Get-Content (Join-Path $dest 'old.node')) -eq 'old local output') 'recursive deletion began before validation'
  }
  if ($kind -eq 'ordinary') { Assert-Fixture ((Get-Content (Join-Path $dest 'mom.node')) -eq 'new node') 'ordinary publication failed' }
  if ($kind -eq 'root-ordinary') { Assert-Fixture (-not (Test-Path $out)) 'ordinary root reset failed' }
}
Write-Output 'PUBLICATION_FIXTURE_OK'
`.replaceAll("__HELPER__", () => quote(helper)).replaceAll("__DIRECTORY__", () => quote(fixture))
      .replace("__SAVE__", () => save).replace("__BLOCKS__", () => blocks.map(quote).join(","))
      .replace("__RESET__", () => quote(reset));
    try {
      fs.writeFileSync(driver, source);
      const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver],
        {encoding: "utf8", windowsHide: true, timeout: 30000});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.match(result.stdout, /PUBLICATION_FIXTURE_OK/);
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  });

  test("Windows retention prunes only obsolete current-variant outputs and abandoned receipts", {skip: process.platform !== "win32"}, () => {
    const fixture = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-retention-")));
    /** @param {string} value @returns {string} */
    const quote = value => "'" + value.replaceAll("'", "''") + "'";
    const driver = path.join(fixture, "driver.ps1");
    const source = String.raw`
$ErrorActionPreference = 'Stop'
. __HELPER__
Set-Location __DIRECTORY__
function Assert-Fixture($Condition, [string]$Name) { if (-not $Condition) { throw $Name } }
$cache = Join-Path __DIRECTORY__ 'build/.incremental/sycl-dpcpp-cuda'
$other = Join-Path __DIRECTORY__ 'build/.incremental/sycl-dpcpp-portable'
New-Item -ItemType Directory -Force $cache, $other, (Join-Path $cache 'nested') | Out-Null
$current = Join-Path $cache 'current.obj'
foreach ($name in @('current.obj','current.obj.d','current.obj.json','current.obj.spir64.d',
    'obsolete.obj','obsolete.obj.d','obsolete.obj.json','obsolete.obj.spir64.d','obsolete.obj.nvptx64-nvidia-cuda.d',
    'sycl.dll','sycl.lib','sycl.exp','link.json','notes.txt','notes.tmp','unrelated.d')) {
  Set-Content (Join-Path $cache $name) $name
}
Set-Content (Join-Path $other 'obsolete.obj') 'other variant'
Set-Content (Join-Path $cache 'nested/obsolete.obj') 'unowned nested output'
$temp = Join-Path $cache 'obsolete.obj.json.0123456789abcdef0123456789abcdef.tmp'
$otherTemp = Join-Path $cache 'user.json.0123456789abcdef0123456789abcdef.tmp'
Set-Content $temp 'abandoned receipt'
Set-Content $otherTemp 'unowned temporary'
Remove-MomAbandonedBuildReceipts $cache
Assert-Fixture (-not (Test-Path -LiteralPath $temp)) 'abandoned receipt retained'
Assert-Fixture (Test-Path -LiteralPath (Join-Path $cache 'obsolete.obj')) 'admission removed compile progress'
Remove-MomObsoleteCompileOutputs $cache @($current)
foreach ($name in @('obsolete.obj','obsolete.obj.d','obsolete.obj.json','obsolete.obj.spir64.d','obsolete.obj.nvptx64-nvidia-cuda.d')) {
  Assert-Fixture (-not (Test-Path -LiteralPath (Join-Path $cache $name))) 'obsolete compile output retained'
}
foreach ($name in @('current.obj','current.obj.d','current.obj.json','current.obj.spir64.d','sycl.dll','sycl.lib','sycl.exp',
    'link.json','notes.txt','notes.tmp','unrelated.d','nested/obsolete.obj')) {
  Assert-Fixture (Test-Path -LiteralPath (Join-Path $cache $name)) 'current or unowned output removed'
}
Assert-Fixture (Test-Path -LiteralPath $otherTemp) 'unowned receipt-like temporary removed'
Assert-Fixture ((Get-Content (Join-Path $other 'obsolete.obj')) -eq 'other variant') 'other variant removed'
$outside = Join-Path __DIRECTORY__ 'outside'
New-Item -ItemType Directory -Force $outside | Out-Null
Set-Content (Join-Path $outside 'obsolete.obj') 'outside'
$junction = Join-Path __DIRECTORY__ 'build/.incremental/redirected'
New-Item -ItemType Junction -Path $junction -Value $outside | Out-Null
$rejected = $false
try { Remove-MomObsoleteCompileOutputs $junction @() } catch { $rejected = $true }
Assert-Fixture $rejected 'junction cache admitted'
Assert-Fixture ((Get-Content (Join-Path $outside 'obsolete.obj')) -eq 'outside') 'outside cache output removed'
Write-Output 'RETENTION_FIXTURE_OK'
`.replaceAll("__HELPER__", () => quote(helper)).replaceAll("__DIRECTORY__", () => quote(fixture));
    try {
      fs.writeFileSync(driver, source);
      const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver],
        {encoding: "utf8", windowsHide: true, timeout: 30000});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.match(result.stdout, /RETENTION_FIXTURE_OK/);
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  });

  test("PowerShell incremental receipts reject changed inputs, corruption and interrupted builds", {skip: process.platform !== "win32"}, () => {
    const fixture = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-incremental-")));
    /** @param {string} value @returns {string} */
    const quote = value => "'" + value.replaceAll("'", "''") + "'";
    const driver = path.join(fixture, "driver.ps1");
    const source = String.raw`
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. __HELPER__
Set-Location __DIRECTORY__
[Environment]::CurrentDirectory = [IO.Path]::GetTempPath()
function Assert-Fixture($Condition, [string]$Name) { if (-not $Condition) { throw $Name } }
$source = Join-Path __DIRECTORY__ 'source with spaces.cpp'
$header = Join-Path __DIRECTORY__ 'included.inc'
$deviceHeader = Join-Path __DIRECTORY__ 'device only.inc'
$object = Join-Path __DIRECTORY__ 'source.obj'
$output = Join-Path __DIRECTORY__ 'sycl.dll'
Set-Content $source 'source'
Set-Content $header 'header'
Set-Content $deviceHeader 'device-only header'
$resolvedSource = Resolve-MomBuildPath 'source with spaces.cpp'
Assert-Fixture ($resolvedSource -eq $source) "relative path used native cwd: expected=$source actual=$resolvedSource provider=$((Get-Location).ProviderPath) native=$([Environment]::CurrentDirectory)"
$script:compiles = 0
$script:links = 0
$script:fail = $false
$script:mutateLibrary = $false
$script:mutateAuxiliary = $false
function Invoke-FixtureCompiler {
  $script:compiles++
  $argList = @($args)
  if (@($argList | Where-Object { $_ -isnot [string] }).Count -ne 0) { throw 'Fixture compiler requires scalar string arguments' }
  $depIndex = [Array]::IndexOf($argList, '-MF') + 1
  $objIndex = [Array]::IndexOf($argList, '-o') + 1
  if ($depIndex -lt 1 -or $depIndex -ge $argList.Count -or $objIndex -lt 1 -or $objIndex -ge $argList.Count) {
    throw 'Fixture compiler requires explicit dependency and object arguments'
  }
  $dep = $argList[$depIndex]
  $obj = $argList[$objIndex]
  if ($script:fail) { Set-Content $obj 'partial'; $global:LASTEXITCODE = 1; return }
  Set-Content $obj "object $script:compiles"
  $slash = [string][char]92
  $escaped = $source.Replace($slash, $slash + $slash).Replace(' ', $slash + ' ')
  $escapedHeader = $header.Replace($slash, $slash + $slash).Replace(' ', $slash + ' ')
  Set-Content $dep ("mom: $escaped $escapedHeader" + [Environment]::NewLine + "mom: $escaped") -Encoding UTF8
  $escapedDevice = $deviceHeader.Replace($slash, $slash + $slash).Replace(' ', $slash + ' ')
  foreach ($deviceDep in @($argList | Where-Object { $_ -like '-MF?*' })) {
    Set-Content -LiteralPath $deviceDep.Substring(3) -Value "mom: $escaped $escapedDevice" -Encoding UTF8
  }
  $global:LASTEXITCODE = 0
}
function Invoke-FixtureLink {
  $script:links++
  Set-Content $output "linked $script:links"
  if ($script:mutateLibrary) { Set-Content $selectedLib 'library three' }
  if ($script:mutateAuxiliary) { Set-Content $auxiliary 'helper three' }
  $global:LASTEXITCODE = 0
  if ($script:fail) { $global:LASTEXITCODE = 1 }
}
function Compile-Fixture([string]$Context = 'context', [string]$Flag = '-O3') {
  # Each call models a fresh production build invocation.
  $script:MomBuildSnapshot = @()
  $task = New-MomCompileTask 'Invoke-FixtureCompiler' @($Flag, '-c', $source, '-o', $object) $source $object 'fixture' $Context
  if (-not $task.Skip) {
    # Named-variable splatting preserves scalar argv for a PowerShell function mock.
    $sourceBeforeCompile = Get-MomBuildHash $task.Source
    $compilerArgs = $task.Args
    & $task.Exe @compilerArgs
    if ($LASTEXITCODE -ne 0) { return $false }
    Assert-Fixture ((Get-MomBuildHash $task.Source) -eq $sourceBeforeCompile) 'fixture compiler changed source'
    try { Complete-MomCompileTask $task } catch {
      $completionFailure = $_
      try {
        $sourceRows = @($task.Before | Where-Object { $_.Path -eq $task.Source })
        $hashEqual = $sourceRows.Count -eq 1 -and $sourceRows[0].Hash -eq (Get-MomBuildHash $task.Source)
        Write-Host "COMPILE_FIXTURE_FAILURE ORDINAL=$script:compiles BEFORE_SOURCE_COUNT=$($sourceRows.Count) BEFORE_SOURCE_HASH_EQUAL=$([int]$hashEqual)"
      } catch { Write-Host "COMPILE_FIXTURE_FAILURE ORDINAL=$script:compiles DIAGNOSTIC_AVAILABLE=0" }
      throw $completionFailure
    }
  }
  return $true
}
Assert-Fixture (Compile-Fixture) 'first compile'
function Reject-ExternalEdit {
  Set-Content $header 'external edit after task admission'
  $rejected = $false
  try { Invoke-MomIncrementalLink 'Invoke-FixtureLink' @('-shared', $object, '-o', $output) @($object) $output "$output.json" 'context' } catch { $rejected = $true }
  Assert-Fixture $rejected 'external dependency edit after admission accepted'
  Assert-Fixture (-not (Test-Path "$output.json")) 'external edit published receipt'
  Set-Content $header 'header'
}
Reject-ExternalEdit
Assert-Fixture (Compile-Fixture) 'unchanged compile'
Reject-ExternalEdit
Assert-Fixture ($script:compiles -eq 1) 'unchanged object not skipped'
$snapshotBefore = @($script:MomBuildSnapshot)
$script:MomBuildSnapshot += @{ Path = $header; Hash = ('0' * 64) }
$rejected = $false
try { Assert-MomBuildInputs } catch { $rejected = $true }
Assert-Fixture $rejected 'contradictory dependency admissions were discarded'
$script:MomBuildSnapshot = $snapshotBefore
Invoke-MomIncrementalLink 'Invoke-FixtureLink' @('-shared', $object, '-o', $output) @($object) $output "$output.json" 'context'
Invoke-MomIncrementalLink 'Invoke-FixtureLink' @('-shared', $object, '-o', $output) @($object) $output "$output.json" 'context'
Assert-Fixture ($script:links -eq 1) 'unchanged link not skipped'
$headerTime = (Get-Item $header).LastWriteTimeUtc
Set-Content $header 'changed header'
(Get-Item $header).LastWriteTimeUtc = $headerTime
Assert-Fixture (Compile-Fixture) 'header rebuild'
Assert-Fixture ($script:compiles -eq 2) 'transitive inc change not rebuilt'
Invoke-MomIncrementalLink 'Invoke-FixtureLink' @('-shared', $object, '-o', $output) @($object) $output "$output.json" 'context'
Assert-Fixture ($script:links -eq 2) 'object change not relinked'
Set-Content $output 'corrupt DLL'
Invoke-MomIncrementalLink 'Invoke-FixtureLink' @('-shared', $object, '-o', $output) @($object) $output "$output.json" 'context'
Assert-Fixture ($script:links -eq 3) 'corrupt DLL admitted'
$script:fail = $true
$linkFailed = $false
try { Invoke-MomIncrementalLink 'Invoke-FixtureLink' @('-changed', $object, '-o', $output) @($object) $output "$output.json" 'context' } catch { $linkFailed = $true }
Assert-Fixture $linkFailed 'failed link accepted'
Assert-Fixture (-not (Test-Path "$output.json")) 'failed link receipt retained'
$script:fail = $false
Invoke-MomIncrementalLink 'Invoke-FixtureLink' @('-changed', $object, '-o', $output) @($object) $output "$output.json" 'context'
Assert-Fixture ($script:links -eq 5) 'failed link not retried'
Set-Content $object 'corrupt'
Assert-Fixture (Compile-Fixture) 'corrupt object rebuild'
Assert-Fixture ($script:compiles -eq 3) 'corrupt object admitted'
Assert-Fixture (Compile-Fixture 'changed toolchain') 'toolchain rebuild'
Assert-Fixture (Compile-Fixture 'changed toolchain' '-O2') 'command rebuild'
Assert-Fixture ($script:compiles -eq 5) 'command context not invalidated'
$script:fail = $true
Set-Content $header 'failure trigger'
Assert-Fixture (-not (Compile-Fixture)) 'failed compile accepted'
Assert-Fixture (-not (Test-Path "$object.json")) 'failure receipt retained'
$script:fail = $false
Assert-Fixture (Compile-Fixture) 'failed compile retry'
Assert-Fixture ($script:compiles -eq 7) 'partial output admitted'
$env:MOM_FORCE_REBUILD = '1'
Assert-Fixture (Compile-Fixture) 'forced compile'
Assert-Fixture ($script:compiles -eq 8) 'force ignored'
$env:MOM_FORCE_REBUILD = ''
$task = New-MomCompileTask 'Invoke-FixtureCompiler' @('-O3', '-c', $source, '-o', $object) $source $object 'fixture' 'midbuild'
$sourceBeforeCompile = Get-MomBuildHash $task.Source
$compilerArgs = $task.Args
& $task.Exe @compilerArgs
Assert-Fixture ((Get-MomBuildHash $task.Source) -eq $sourceBeforeCompile) 'fixture compiler changed source before deliberate mutation'
$sourceTime = (Get-Item $source).LastWriteTimeUtc
Set-Content $source 'changed while compiling'
(Get-Item $source).LastWriteTimeUtc = $sourceTime
$rejected = $false
try { Complete-MomCompileTask $task } catch { $rejected = $true }
Assert-Fixture $rejected 'midbuild source change admitted'
Assert-Fixture (-not (Test-Path "$object.json")) 'midbuild failure receipt published'
foreach ($targets in @('spir64', 'spir64,nvidia_gpu_sm_80')) {
  $flags = @('-fsycl', "-fsycl-targets=$targets", '-c', $source, '-o', $object)
  $task = New-MomCompileTask 'Invoke-FixtureCompiler' $flags $source $object 'device fixture' 'device graph'
  Assert-Fixture ($task.DepFiles.Count -eq (1 + ($targets -split ',').Count)) 'wrong number of frontend dependency files'
  Assert-Fixture ('-Xsycl-target-frontend=spir64' -in $task.Args) 'missing SPIR frontend dependency flag'
  if ($targets.Contains('nvidia_gpu_')) {
    Assert-Fixture ('-Xsycl-target-frontend=nvptx64-nvidia-cuda' -in $task.Args) 'CUDA architecture was not normalized to frontend triple'
  }
  $compilerArgs = $task.Args
  & $task.Exe @compilerArgs
  Complete-MomCompileTask $task
  $state = Read-MomBuildState $task.State
  Assert-Fixture (@($state.Dependencies | Where-Object { $_.Path -eq $deviceHeader }).Count -eq 1) 'device-only dependencies missing or duplicated'
  $warm = New-MomCompileTask 'Invoke-FixtureCompiler' $flags $source $object 'device fixture' 'device graph'
  Assert-Fixture $warm.Skip 'unchanged complete device graph not skipped'
  $deviceTime = (Get-Item $deviceHeader).LastWriteTimeUtc
  Add-Content $deviceHeader 'backdated device edit'
  (Get-Item $deviceHeader).LastWriteTimeUtc = $deviceTime
  $task = New-MomCompileTask 'Invoke-FixtureCompiler' $flags $source $object 'device fixture' 'device graph'
  Assert-Fixture (-not $task.Skip) 'backdated device-only header was ignored'
  Assert-Fixture (@($task.DepFiles | Where-Object { Test-Path -LiteralPath $_ }).Count -eq 0) 'stale frontend dependency files retained'
  $compilerArgs = $task.Args
  & $task.Exe @compilerArgs
  Remove-Item -LiteralPath $task.DepFiles[-1]
  $rejected = $false
  try { Complete-MomCompileTask $task } catch { $rejected = $true }
  Assert-Fixture ($rejected -and -not (Test-Path $task.State)) 'missing device graph admitted'
  & $task.Exe @compilerArgs
  Set-Content -LiteralPath $task.DepFiles[-1] -Value 'unexpected: source.cpp'
  $rejected = $false
  try { Complete-MomCompileTask $task } catch { $rejected = $true }
  Assert-Fixture ($rejected -and -not (Test-Path $task.State)) 'malformed device graph admitted'
  & $task.Exe @compilerArgs
  Complete-MomCompileTask $task
}
$sdk = Join-Path __DIRECTORY__ 'sdk'
$cache = Join-Path __DIRECTORY__ 'build/.incremental/fixture'
New-Item -ItemType Directory -Force (Join-Path $sdk 'include'), (Join-Path $sdk 'nvvm/libdevice') | Out-Null
$libdevice = Join-Path $sdk 'nvvm/libdevice/libdevice.10.bc'
Set-Content $libdevice 'bitcode version one'
$env:PATH = 'fixture;other'
$env:LIB = ''
$context = Initialize-MomIncrementalBuild $cache @(__HELPER__) @($sdk)
$env:PATH = 'fixture;fixture;other'
Assert-Fixture ((Initialize-MomIncrementalBuild $cache @(__HELPER__) @($sdk)) -eq $context) 'duplicate PATH caused miss'
Set-Content (Join-Path $sdk 'include/new.h') 'new header'
$next = Initialize-MomIncrementalBuild $cache @(__HELPER__) @($sdk)
Assert-Fixture ($next -ne $context) 'include membership change admitted'
$libTime = (Get-Item $libdevice).LastWriteTimeUtc
Set-Content $libdevice 'bitcode version two'
(Get-Item $libdevice).LastWriteTimeUtc = $libTime
Assert-Fixture ((Initialize-MomIncrementalBuild $cache @(__HELPER__) @($sdk)) -ne $next) 'restored-timestamp libdevice change admitted'
$sdkBin = Join-Path $sdk 'bin'
New-Item -ItemType Directory $sdkBin | Out-Null
foreach ($name in @('ptxas.exe', 'nvlink.exe', 'sycl-post-link.exe', 'llvm-spirv.exe', 'clang-offload-wrapper.exe', 'clang-offload-bundler.exe')) {
  $auxiliary = Join-Path $sdkBin $name
  Set-Content $auxiliary 'helper one'
  $context = Initialize-MomIncrementalBuild $cache @(__HELPER__) @($sdk)
  Assert-Fixture (@($script:MomBuildSnapshot | Where-Object { $_.Path -eq $auxiliary }).Count -eq 1) 'auxiliary missing or duplicated in snapshot'
  $auxiliaryTime = (Get-Item $auxiliary).LastWriteTimeUtc
  $auxiliarySize = (Get-Item $auxiliary).Length
  Set-Content $auxiliary 'helper two'
  (Get-Item $auxiliary).LastWriteTimeUtc = $auxiliaryTime.AddDays(-1)
  Assert-Fixture ((Get-Item $auxiliary).Length -eq $auxiliarySize) 'auxiliary fixture changed size'
  Assert-Fixture ((Initialize-MomIncrementalBuild $cache @(__HELPER__) @($sdk)) -ne $context) 'same-size backdated auxiliary change admitted'
}
$context = Initialize-MomIncrementalBuild $cache @(__HELPER__) @($sdk)
$script:mutateAuxiliary = $true
$rejected = $false
try { Invoke-MomIncrementalLink 'Invoke-FixtureLink' @('-auxiliary-change') @($object) $output "$output.json" $context } catch { $rejected = $true }
Assert-Fixture $rejected 'auxiliary changed during link admitted'
Assert-Fixture (-not (Test-Path "$output.json")) 'changed-auxiliary link receipt published'
$script:mutateAuxiliary = $false
$libDir = Join-Path __DIRECTORY__ 'selected-libs'
New-Item -ItemType Directory $libDir | Out-Null
$selectedLib = Join-Path $libDir 'selected.lib'
Set-Content $selectedLib 'library one'
$env:LIB = $libDir
$context = Initialize-MomIncrementalBuild $cache @(__HELPER__) @($sdk)
$env:LIB = "$libDir;$libDir"
Assert-Fixture ((Initialize-MomIncrementalBuild $cache @(__HELPER__) @($sdk)) -eq $context) 'duplicate LIB directory caused miss'
$selectedTime = (Get-Item $selectedLib).LastWriteTimeUtc
$selectedSize = (Get-Item $selectedLib).Length
Set-Content $selectedLib 'library two'
(Get-Item $selectedLib).LastWriteTimeUtc = $selectedTime.AddDays(-1)
Assert-Fixture ((Get-Item $selectedLib).Length -eq $selectedSize) 'selected library fixture changed size'
$next = Initialize-MomIncrementalBuild $cache @(__HELPER__) @($sdk)
Assert-Fixture ($next -ne $context) 'same-size backdated selected library change admitted'
$script:mutateLibrary = $true
$rejected = $false
try { Invoke-MomIncrementalLink 'Invoke-FixtureLink' @('-library-change') @($object) $output "$output.json" $next } catch { $rejected = $true }
Assert-Fixture $rejected 'library changed during link admitted'
Assert-Fixture (-not (Test-Path "$output.json")) 'changed-library link receipt published'
$script:mutateLibrary = $false
$context = Initialize-MomIncrementalBuild $cache @(__HELPER__) @($sdk)
$libraryCountBefore = @($script:MomBuildSnapshot | Where-Object { $_.Path -eq $selectedLib }).Count
Assert-Fixture ($libraryCountBefore -eq 1) 'selected library missing or duplicated in snapshot'
$newLib = Join-Path $libDir 'new.lib'
Set-Content $newLib 'new candidate'
$rejected = $false
try { Assert-MomBuildInputs } catch { $rejected = $true }
Assert-Fixture $rejected 'library membership changed during build admitted'
$next = Initialize-MomIncrementalBuild $cache @(__HELPER__) @($sdk)
$libraryCountAfter = @($script:MomBuildSnapshot | Where-Object { $_.Path -in @($selectedLib, $newLib) }).Count
Write-Host "SDK_FIXTURE LIBRARY_INPUTS_BEFORE=$libraryCountBefore LIBRARY_INPUTS_AFTER=$libraryCountAfter LIBRARY_KEY_CHANGED=$([int]($next -ne $context))"
Assert-Fixture ($libraryCountAfter -eq 2) 'new library candidate missing or duplicated in snapshot'
Assert-Fixture ($next -ne $context) 'new library candidate reused key'
Write-Output 'INCREMENTAL_FIXTURE_OK'
`.replaceAll("__HELPER__", () => quote(helper)).replaceAll("__DIRECTORY__", () => quote(fixture));
    try {
      fs.writeFileSync(driver, source);
      const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver],
        {encoding: "utf8", windowsHide: true, cwd: os.tmpdir(), timeout: 30000});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.match(result.stdout, /INCREMENTAL_FIXTURE_OK/);
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  });
});
