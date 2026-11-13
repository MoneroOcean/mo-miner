"use strict";

const test = require("node:test"), {describe} = require("node:test"), assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const fs = require("node:fs"), path = require("node:path");
const os = require("node:os");

const root = path.resolve(__dirname, "..");
const helper = fs.readFileSync(path.join(root, ".github/workflows/scripts/prepare-native-windows-ci.ps1"), "utf8");
const installer = fs.readFileSync(path.join(root, "scripts/install-dev.ps1"), "utf8");

describe("Windows native CI tool preparation", () => {
  test("imports the selected x64 VC environment before resolving any stage tools", () => {
    assert.match(helper, /ValidateSet\('core', 'io', 'hermes'\)/);
    const importFile = helper.indexOf("scripts\\import-vcvars.ps1");
    const importCall = helper.indexOf("Import-MomVcVars64");
    const firstToolLookup = helper.indexOf("Get-Command -Name");
    assert.ok(importFile >= 0 && importFile < importCall);
    assert.ok(importCall >= 0 && importCall < firstToolLookup);
    assert.match(helper, /vswhere and vcvars64 selected different Visual Studio installations/);
  });

  test("Core dispatch preserves its established explicit tool environment without unrelated tools", () => {
    const core = helper.slice(helper.indexOf("  'core' {"), helper.indexOf("  'io' {"));
    for (const name of ["NODE_GYP", "MSBUILD", "DUMPBIN", "PYTHON", "COMPILER"]) {
      assert.match(core, new RegExp(`MOM_NATIVE_WINDOWS_CORE_${name}`));
    }
    assert.match(core, /Invoke-NativeStage 'windows-native-core'/);
    assert.doesNotMatch(core, /perl|ninja|cmake/i);
  });

  test("I/O and Hermes publish only their producer-specific explicit Windows tool maps", () => {
    const io = helper.slice(helper.indexOf("  'io' {"), helper.indexOf("  'hermes' {"));
    const hermes = helper.slice(helper.indexOf("  'hermes' {"));
    assert.match(io, /openssl = \[ordered\]@\{/);
    for (const key of ["cmake", "cc", "ninja", "ar", "link", "make", "perl", "rc", "tar"]) {
      assert.match(io, new RegExp(`\\b${key}\\s*=`));
    }
    assert.match(io, /Invoke-NativeStage 'windows-controller-io'/);
    for (const key of ["git", "cmake", "cc", "cxx", "ninja"]) {
      assert.match(hermes, new RegExp(`\\b${key}\\s*=`));
    }
    assert.match(hermes, /Invoke-NativeStage 'windows-controller-hermes'/);
    assert.match(hermes, /cxx = \$compiler/);
  });

  test("Controller stages verify Intel 2026.2 and prefer the pinned native Perl runtime", () => {
    const intel = helper.slice(helper.indexOf("function Get-OneApiCompiler"), helper.indexOf("function Get-NativePerl"));
    const perl = helper.slice(helper.indexOf("function Get-NativePerl"), helper.indexOf("function Get-VcTool"));
    assert.match(intel, /VersionInfo\.ProductVersion/);
    assert.match(intel, /\^2026\\\.2/);
    assert.match(perl, /MOM_NATIVE_PERL_ROOT/);
    assert.match(perl, /'C:\\Strawberry'/);
    assert.match(perl, /if \(\$exitCode -ne 0 -or \(\[string\]::Join\('', \$reported\)\)\.Trim\(\) -cne 'MSWin32'\)/);
    assert.match(perl, /throw 'Selected Perl is not a working native Windows/);
    assert.match(perl, /finally \{\s*\$env:Path = \$oldPath\s*\}/);
  });

  test("The setup helper is offline and never installs or downloads tools", () => {
    assert.doesNotMatch(helper, /\b(?:Invoke-WebRequest|Invoke-RestMethod|winget|choco|Install-Package|npm\s+install)\b/i);
    assert.match(helper, /'C:\\Strawberry'/);
    assert.match(helper, /Join-Path \$perlBin 'perl\.exe'/);
    assert.doesNotMatch(helper, /Git\\usr\\bin\\perl|msys[^\r\n]*perl/i);
  });

  test("explicit tool paths are canonicalized before producer admission and version checks", () => {
    const resolver = helper.slice(helper.indexOf("function Resolve-RequiredFile"), helper.indexOf("function Resolve-Application"));
    assert.match(resolver, /\$nodePath = 'C:\\Program Files\\nodejs\\node\.exe'/);
    assert.match(resolver, /realpathSync\(process\.argv\[2\]\)/);
    assert.match(resolver, /'node:fs' \$fullPath/);
    assert.match(resolver, /\$nodeExit -ne 0 -or \$resolvedRows\.Count -ne 1/);
    assert.match(resolver, /\$canonicalPath\.Length -gt 4096/);
    assert.match(resolver, /\$canonicalPath -match '\[\\x00-\\x1F\\x7F\]'/);
    assert.match(resolver, /\[IO\.Path\]::IsPathRooted\(\$canonicalPath\)/);
    assert.doesNotMatch(resolver, /"\$Name:/);
    assert.match(resolver, /canonicalize \$\(\$Name\):/);
    const compiler = helper.slice(helper.indexOf("function Get-OneApiCompiler"), helper.indexOf("function Get-NativePerl"));
    assert.ok(compiler.indexOf("Resolve-RequiredFile") < compiler.indexOf("VersionInfo.ProductVersion"));
    assert.ok(compiler.indexOf("VersionInfo.ProductVersion") < compiler.indexOf("lib\\libircmt.lib"));
    assert.ok(compiler.indexOf("lib\\libircmt.lib") < compiler.indexOf("$env:LIB ="));
    assert.match(compiler, /OrdinalIgnoreCase/);
    assert.match(compiler, /Split-Path -Parent \(Split-Path -Parent \$compiler\)/);
  });

  test("base readiness shares its six exact checks and reports missing check names", () => {
    const checks = installer.slice(installer.indexOf("function Get-BaseMissingComponents"), installer.indexOf("function Test-Component"));
    for (const name of ["Test-VsBuildTools", "Test-Python", "Test-Git", "cmake.exe", "ninja.exe", "7z.exe"]) {
      assert.ok(checks.includes(name), `base readiness must check ${name}`);
    }
    const base = installer.slice(installer.indexOf("function Test-Component"), installer.indexOf("if ($ValidateOnly)"));
    assert.match(base, /base\s*\{\s*return\s*\(@\(Get-BaseMissingComponents\)\.Count\s*-eq\s*0\)\s*\}/);
    assert.match(installer, /dev component base missing checks:/);
    assert.match(installer, /Installed component failed validation: base \(missing checks:/);
  });

  test("Python readiness buffers native output before checking the process exit code", () => {
    const python = installer.slice(installer.indexOf("function Test-Python"), installer.indexOf("function Test-Git"));
    assert.match(python, /\$output = @\(& \$python --version 2>&1\)/);
    assert.match(python, /\$exitCode = \$LASTEXITCODE/);
    assert.ok(python.indexOf("$exitCode = $LASTEXITCODE") < python.indexOf("$output[0].ToString()"));
    assert.doesNotMatch(python, /Select-Object\s+-First/);
    assert.match(python, /\$exitCode -eq 0/);
    assert.match(python, /\$reported -eq "Python \$pythonVersion"/);
  });

  test("Test-Python waits for a delayed native version probe to finish", {skip: process.platform !== "win32"}, () => {
    const versionMatch = /^\$pythonVersion = '([^']+)'$/m.exec(installer);
    if (!versionMatch) {
      throw new Error("Pinned Python version was not found in install-dev.ps1");
    }
    const pythonVersion = versionMatch[1];
    if (!pythonVersion) {
      throw new Error("Pinned Python version capture was empty");
    }
    /** @param {string} value */
    const psQuote = value => `'${value.replace(/'/g, "''")}'`;
    const script = `
$ErrorActionPreference = 'Stop'
$tokens = $null; $parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(${psQuote(path.join(root, "scripts/install-dev.ps1"))}, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'install-dev.ps1 parse failed' }
$fn = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-Python' }, $true)
if (-not $fn) { throw 'Test-Python was not found in the parsed source' }
. ([scriptblock]::Create($fn.Extent.Text))
$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ('mom-python-readiness-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot | Out-Null
$script:pythonRoot = $tempRoot
$script:pythonVersion = ${psQuote(pythonVersion)}
$env:MOM_TEST_PYTHON_MARKER = Join-Path $tempRoot 'completed.marker'
$fixture = Join-Path $tempRoot 'python.exe'
$source = @'
using System;
using System.IO;
using System.Threading;
public static class DelayedPythonVersionFixture {
  public static void Main() {
    Console.WriteLine(${JSON.stringify(`Python ${pythonVersion}`)});
    Console.Out.Flush();
    Thread.Sleep(750);
    File.WriteAllText(Environment.GetEnvironmentVariable("MOM_TEST_PYTHON_MARKER"), "complete");
  }
}
'@
try {
  Add-Type -TypeDefinition $source -Language CSharp -OutputType ConsoleApplication -OutputAssembly $fixture
  if (-not (Test-Python)) { throw 'Delayed Python version fixture failed readiness' }
  if (-not (Test-Path -LiteralPath $env:MOM_TEST_PYTHON_MARKER -PathType Leaf)) {
    throw 'Test-Python returned before the native version process completed'
  }
  if ((Get-Content -LiteralPath $env:MOM_TEST_PYTHON_MARKER -Raw) -cne 'complete') {
    throw 'Delayed Python version fixture wrote unexpected completion data'
  }
} finally {
  Remove-Item Env:MOM_TEST_PYTHON_MARKER -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
`;
    const result = spawnSync("powershell.exe", ["-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", Buffer.from(script, "utf16le").toString("base64")], {
      encoding: "utf8", timeout: 30000, windowsHide: true
    });
    assert.equal(result.error, undefined, result.error?.message);
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
  });

  test("PowerShell base readiness names each failed check without invoking installers", {skip: process.platform !== "win32"}, () => {
    /** @param {string} value */
    const psQuote = value => `'${value.replace(/'/g, "''")}'`;
    const script = `
$ErrorActionPreference = 'Stop'
$tokens = $null; $parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(${psQuote(path.join(root, "scripts/install-dev.ps1"))}, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'install-dev.ps1 parse failed' }
$fn = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-BaseMissingComponents' }, $true)
if (-not $fn) { throw 'Get-BaseMissingComponents was not found in the parsed source' }
$definition = [scriptblock]::Create($fn.Extent.Text)
. $definition
$script:ready = @{}
function Test-VsBuildTools { return [bool]$script:ready['visual-studio-build-tools'] }
function Test-Python { return [bool]$script:ready['python'] }
function Test-Git { return [bool]$script:ready['git'] }
function Get-Command {
  [CmdletBinding()]
  param([string]$Name)
  if ($script:ready[$Name]) { return [pscustomobject]@{ Name = $Name } }
  return $null
}
$required = @('visual-studio-build-tools','python','git','cmake.exe','ninja.exe','7z.exe')
foreach ($missing in $required) {
  $script:ready = @{}
  foreach ($name in $required) { $script:ready[$name] = $true }
  $script:ready[$missing] = $false
  $actual = @(Get-BaseMissingComponents)
  if ($actual.Count -ne 1 -or $actual[0] -cne $missing) { throw "Incorrect missing-check report for $missing" }
}
$script:ready = @{}
foreach ($name in $required) { $script:ready[$name] = $true }
if (@(Get-BaseMissingComponents).Count -ne 0) { throw 'Ready base checks reported a missing dependency' }
`;
    const result = spawnSync("powershell.exe", ["-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", Buffer.from(script, "utf16le").toString("base64")], {
      encoding: "utf8", timeout: 15000, windowsHide: true
    });
    assert.equal(result.error, undefined, result.error?.message);
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
  });

  test("PowerShell tool resolver returns the canonical target of a directory junction", {skip: process.platform !== "win32"}, () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-realpath-"));
    try {
      const targetDir = path.join(tempRoot, "compiler", "2026.2", "bin");
      const aliasDir = path.join(tempRoot, "compiler", "latest");
      fs.mkdirSync(targetDir, {recursive: true});
      const targetFile = path.join(targetDir, "icx-cl.exe");
      const runtimeLibraryFile = path.join(path.dirname(targetDir), "lib", "libircmt.lib");
      const redirectedLibraryDir = path.join(tempRoot, "other-compiler-runtime");
      fs.writeFileSync(targetFile, "fixture executable path only\n");
      fs.mkdirSync(path.dirname(runtimeLibraryFile), {recursive: true});
      fs.writeFileSync(runtimeLibraryFile, "fixture Intel runtime library\n");
      fs.mkdirSync(redirectedLibraryDir, {recursive: true});
      fs.writeFileSync(path.join(redirectedLibraryDir, "libircmt.lib"), "redirected fixture library\n");
      fs.symlinkSync(path.dirname(targetDir), aliasDir, "junction");
      const aliasFile = path.join(aliasDir, "bin", "icx-cl.exe");
      const missingFile = path.join(aliasDir, "bin", "missing.exe");
      const expected = fs.realpathSync(targetFile);
      const expectedLibraryDir = path.dirname(fs.realpathSync(runtimeLibraryFile));
      /** @param {string} value */
      const psQuote = value => `'${value.replace(/'/g, "''")}'`;
      const script = `
$ErrorActionPreference = 'Stop'
$tokens = $null; $parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(${psQuote(path.join(root, ".github/workflows/scripts/prepare-native-windows-ci.ps1"))}, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ('Windows CI helper parse failed: ' + (($parseErrors | ForEach-Object { $_.Message }) -join '; ')) }
$resolver = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Resolve-RequiredFile' }, $true)
$compilerFunction = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-OneApiCompiler' }, $true)
if (-not $resolver -or -not $compilerFunction) { throw 'Required Windows CI helper functions were not found in the parsed source' }
. ([scriptblock]::Create($resolver.Extent.Text))
. ([scriptblock]::Create($compilerFunction.Extent.Text))
$script:productVersion = '2026.2.0.661'
$script:expectedCompiler = ${psQuote(expected)}
function Get-Item {
  [CmdletBinding()]
  param([string]$LiteralPath)
  if ($LiteralPath -cne $script:expectedCompiler) { throw "Unexpected version lookup path: $LiteralPath" }
  return [pscustomobject]@{ VersionInfo = [pscustomobject]@{ ProductVersion = $script:productVersion } }
}
$env:ONEAPI_ROOT = ${psQuote(tempRoot)}
$actual = Resolve-RequiredFile ${psQuote(aliasFile)} 'fixture compiler'
if ($actual -cne ${psQuote(expected)}) { throw "Junction was not canonicalized: $actual" }
$rejected = $false
try { $null = Resolve-RequiredFile ${psQuote(missingFile)} 'missing fixture' } catch { $rejected = $true }
if (-not $rejected) { throw 'A missing target behind the junction was accepted' }
$runtimeLibraryDir = ${psQuote(expectedLibraryDir)}
$env:LIB = "C:\\VS\\lib;$($runtimeLibraryDir.ToUpperInvariant());C:\\SDK\\lib;$runtimeLibraryDir"
$selected = Get-OneApiCompiler
if ($selected -cne ${psQuote(expected)}) { throw "Wrong compiler selected: $selected" }
$libEntries = @($env:LIB -split ';')
if ($libEntries.Count -ne 3 -or $libEntries[0] -ine $runtimeLibraryDir -or
    $libEntries[1] -cne 'C:\\VS\\lib' -or $libEntries[2] -cne 'C:\\SDK\\lib') {
  throw "Intel runtime LIB path was not prepended once while preserving existing LIB: $($env:LIB)"
}
$beforeRepeat = $env:LIB
$null = Get-OneApiCompiler
if ($env:LIB -cne $beforeRepeat) { throw 'Repeated compiler resolution duplicated its runtime LIB path' }
$env:LIB = 'C:\\KeepOne;C:\\KeepTwo'
$beforeMissingLibrary = $env:LIB
Remove-Item -LiteralPath ${psQuote(runtimeLibraryFile)}
$missingLibraryRejected = $false
try { $null = Get-OneApiCompiler } catch { $missingLibraryRejected = $true }
if (-not $missingLibraryRejected -or $env:LIB -cne $beforeMissingLibrary) {
  throw 'A missing Intel runtime library was accepted or mutated LIB'
}
Set-Content -LiteralPath ${psQuote(runtimeLibraryFile)} -Value 'fixture Intel runtime library' -NoNewline
$script:productVersion = '2025.1.0'
$beforeWrongVersion = $env:LIB
$wrongVersionRejected = $false
try { $null = Get-OneApiCompiler } catch { $wrongVersionRejected = $true }
if (-not $wrongVersionRejected -or $env:LIB -cne $beforeWrongVersion) {
  throw 'A mismatched Intel compiler version was accepted or mutated LIB'
}
$script:productVersion = '2026.2.0.661'
$compilerRoot = Split-Path -Parent (Split-Path -Parent $script:expectedCompiler)
$runtimePath = Join-Path $compilerRoot 'lib'
$savedRuntimePath = Join-Path $compilerRoot 'lib.saved'
Move-Item -LiteralPath $runtimePath -Destination $savedRuntimePath
New-Item -ItemType Junction -Path $runtimePath -Target ${psQuote(redirectedLibraryDir)} | Out-Null
$env:LIB = 'C:\\KeepBeforeRedirect'
$beforeRedirect = $env:LIB
$redirectRejected = $false
try { $null = Get-OneApiCompiler } catch { $redirectRejected = $true }
if (-not $redirectRejected -or $env:LIB -cne $beforeRedirect) {
  throw 'A redirected Intel runtime directory was accepted or mutated LIB'
}
`;
      const result = spawnSync("powershell.exe", ["-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", Buffer.from(script, "utf16le").toString("base64")], {
        encoding: "utf8", timeout: 15000, windowsHide: true
      });
      assert.equal(result.error, undefined, result.error?.message);
      assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
    } finally {
      fs.rmSync(tempRoot, {recursive: true, force: true});
    }
  });
});
