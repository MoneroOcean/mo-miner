"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const WINDOWS = process.platform === "win32";
const repo = path.resolve(__dirname, "..");
const DPCPP_ASSET = "dpcpp-cuda-win-v7.1.1.tar.gz";
const DPCPP_SHA256 = "8e5e9de06ed46c5e28aac4c574811a68f0a85dbb1cb65508187b0b8cf2e573cb";
const PRIOR_DPCPP_SHA256 = "7a61b81cc15484656c80d3927dfc890d14d98689d64f05e3b88f5b45a1e4bb34";
const MATCHED_DPCPP_SHA256 = "81b116580a84ac29221c459ffa354045fae69d95338a8c1db70b76d16be7e472";

/** @param {string} driver @param {string[]} args */
function runPowerShell(driver, args) {
  const result = spawnSync("powershell.exe", [
    "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver, ...args,
  ], {encoding: "utf8", windowsHide: true, timeout: 30000});
  assert.equal(result.error, undefined, result.error?.message);
  return result;
}

/** @returns {string} */
function makeFixture() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-script-safety-"));
}

/** @returns {string} */
function makeRepoFixture() {
  return fs.mkdtempSync(path.join(repo, ".windows-script-safety-"));
}

/** @param {string} fixture @param {string} source @returns {string} */
function makeDriver(fixture, source) {
  const driver = path.join(fixture, "driver.ps1");
  fs.writeFileSync(driver, source);
  return driver;
}

for (const fixtureCase of [
  {name: "benign linker warning", output: "LINK : warning LNK4099: PDB was not found", code: 0, accepted: true},
  {name: "ordinary compiler warning", output: "source.cpp(1): warning C4100: unreferenced parameter", code: 0, accepted: true},
  {name: "SYCL command text", output: "icx-cl.exe /fsycl /clang:-fsycl-device-code-split=per_kernel /link /DLL", code: 0, accepted: true},
  {name: "ignored SYCL flag", output: "LINK : warning LNK4044: unrecognized option '/fsycl'; ignored", code: 0, accepted: false},
  {name: "ignored split policy", output: "LINK : warning LNK4044: unrecognized option '/clang:-fsycl-device-code-split=per_kernel'; ignored", code: 0, accepted: false},
  {name: "case-insensitive ignored option", output: "LINK : warning lnk4044: unrecognized option '/required'; ignored", code: 0, accepted: false},
  {name: "native failure despite later status overwrite", output: "MSBuild failed", code: 7, accepted: false},
  {name: "missing native status despite stale success", output: "No native status", code: null, accepted: false},
]) {
  test(`Windows MSBuild publication guard: ${fixtureCase.name}`, {skip: !WINDOWS}, () => {
    const source = fs.readFileSync(path.join(repo, ".github", "workflows", "scripts", "build-windows.ps1"), "utf8");
    const start = source.indexOf("$msbuildCapturePreference = ");
    const end = source.indexOf("\nNew-Item -ItemType Directory -Force build\\Release", start);
    assert(start >= 0 && end > start, "MSBuild publication boundary is missing");
    const block = source.slice(start, end);
    const fixture = makeFixture();
    try {
      const prefix = [
        "$ErrorActionPreference = 'Stop'",
        "$msbuild = 'Invoke-FixtureMSBuild'",
        "$buildJobs = 1",
        "function Invoke-FixtureMSBuild {",
        `  [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${Buffer.from(fixtureCase.output).toString("base64")}'))`,
        ...(fixtureCase.code === null ? [] : [`  $global:LASTEXITCODE = ${fixtureCase.code}`]),
        "}",
        // Prove the native status is captured before any subsequent output handling.
        "function Write-Host { $global:LASTEXITCODE = 0 }",
        "$global:LASTEXITCODE = 0",
      ].join("\n");
      /** @param {string} body */
      const run = body => runPowerShell(makeDriver(fixture,
        prefix + "\n" + body + "\nWrite-Output 'FIXTURE_ARTIFACT_BOUNDARY'\n"), []);
      const result = run(block);
      assert.equal(result.status === 0, fixtureCase.accepted, result.stdout + result.stderr);
      assert.equal(result.stdout.includes("FIXTURE_ARTIFACT_BOUNDARY"), fixtureCase.accepted);
      if (!fixtureCase.accepted && fixtureCase.code === 0) {
        assert.match(result.stdout + result.stderr, /MSBuild ignored a compiler\/linker option/);
        // Negative control executes the original unguarded success path, not a text assertion.
        const guardStart = block.indexOf("# An ignored linker option");
        assert(guardStart >= 0, "ignored-option guard is missing");
        const before = run(block.slice(0, guardStart));
        assert.equal(before.status, 0, before.stdout + before.stderr);
        assert(before.stdout.includes("FIXTURE_ARTIFACT_BOUNDARY"));
      }
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  });
}

test("Windows MSBuild SYCL hook forwards the exact driver property before publication", {skip: !WINDOWS}, () => {
  const source = fs.readFileSync(path.join(repo, ".github", "workflows", "scripts", "build-windows.ps1"), "utf8");
  const start = source.indexOf("$msbuildCapturePreference = ");
  const end = source.indexOf("\nNew-Item -ItemType Directory -Force build\\Release", start);
  assert(start >= 0 && end > start, "MSBuild publication boundary is missing");
  const block = source.slice(start, end);
  const argument = '"/p:DPCPPLINKOptions=/clang:-fsycl-device-code-split=per_kernel"';
  const fixture = makeFixture();
  try {
    const prefix = [
      "$ErrorActionPreference = 'Stop'",
      "$msbuild = 'Invoke-FixtureMSBuild'",
      "$buildJobs = 1",
      "function Invoke-FixtureMSBuild {",
      "  $expected = '/p:DPCPPLINKOptions=/clang:-fsycl-device-code-split=per_kernel'",
      "  if (@($args | Where-Object { [string]$_ -ceq $expected }).Count -ne 1) {",
      "    $global:LASTEXITCODE = 9; Write-Output 'FIXTURE_MISSING_DRIVER_PROPERTY'; return",
      "  }",
      "  $global:LASTEXITCODE = 0; Write-Output 'FIXTURE_DRIVER_PROPERTY_FORWARDED'",
      "}",
    ].join("\n");
    /** @param {string} body */
    const run = body => runPowerShell(makeDriver(fixture,
      prefix + "\n" + body + "\nWrite-Output 'FIXTURE_ARTIFACT_BOUNDARY'\n"), []);
    const after = run(block);
    assert.equal(after.status, 0, after.stdout + after.stderr);
    assert(after.stdout.includes("FIXTURE_DRIVER_PROPERTY_FORWARDED"));
    assert(after.stdout.includes("FIXTURE_ARTIFACT_BOUNDARY"));
    // The same runtime fixture rejects the previous producer, which did not forward this property.
    const beforeBlock = block.replace(argument, "");
    assert.notEqual(beforeBlock, block);
    const before = run(beforeBlock);
    assert.notEqual(before.status, 0);
    assert(before.stdout.includes("FIXTURE_MISSING_DRIVER_PROPERTY"));
    assert(!before.stdout.includes("FIXTURE_ARTIFACT_BOUNDARY"));
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

for (const fixtureCase of [
  {name: "stderr with success", stderr: "FIXTURE_NATIVE_STDERR", code: 0, accepted: true},
  {name: "stderr with exit 17", stderr: "FIXTURE_NATIVE_STDERR", code: 17, accepted: false},
  {name: "ignored option on stderr", stderr: "LINK : warning LNK4044: unrecognized option '/required'; ignored", code: 0, accepted: false},
  {name: "missing executable with stale success", stderr: "", code: null, accepted: false},
]) {
  test(`Windows MSBuild native stderr capture: ${fixtureCase.name}`, {skip: !WINDOWS}, () => {
    const source = fs.readFileSync(path.join(repo, ".github", "workflows", "scripts", "build-windows.ps1"), "utf8");
    const start = source.indexOf("$msbuildCapturePreference = ");
    const end = source.indexOf("\nNew-Item -ItemType Directory -Force build\\Release", start);
    assert(start >= 0 && end > start, "MSBuild publication boundary is missing");
    const block = source.slice(start, end);
    const invocation = block.match(/\$msbuildOutput = & \$msbuild [\s\S]*? 2>&1/);
    assert(invocation, "native MSBuild capture invocation is missing");
    const fixture = makeFixture();
    try {
      const nativeScript = path.join(fixture, "native-stderr.cjs");
      fs.writeFileSync(nativeScript,
        `process.exitCode = ${fixtureCase.code ?? 0}; process.stderr.write(${JSON.stringify(fixtureCase.stderr + "\n")});\n`);
      const executable = fixtureCase.code === null ? path.join(fixture, "missing.exe") : process.execPath;
      /** @param {string} value */
      const quote = value => "'" + value.replace(/'/g, "''") + "'";
      const nativeCall = `$msbuildOutput = & $msbuild ${quote(nativeScript)} 2>&1`;
      const afterBlock = block.replace(invocation[0], () => nativeCall);
      /** @param {string} body */
      const run = body => {
        const driver = makeDriver(fixture, [
          "$ErrorActionPreference = 'Stop'",
          "$global:LASTEXITCODE = 0",
          `$msbuild = ${quote(executable)}`,
          "$published = $false; $failure = ''; $errorId = ''; $msbuildExitCode = $null",
          "try {",
          body,
          "  $published = $true",
          "} catch { $failure = $_.Exception.Message; $errorId = $_.FullyQualifiedErrorId }",
          "Write-Output ('FIXTURE_RESULT ' + ([PSCustomObject]@{published=$published; status=$msbuildExitCode; preference=[string]$ErrorActionPreference; failure=$failure; errorId=$errorId} | ConvertTo-Json -Compress))",
        ].join("\n"));
        const result = runPowerShell(driver, []);
        assert.equal(result.status, 0, result.stdout + result.stderr);
        const line = result.stdout.split(/\r?\n/).find(value => value.startsWith("FIXTURE_RESULT "));
        assert(line, result.stdout + result.stderr);
        return {result, observed: JSON.parse(line.slice("FIXTURE_RESULT ".length))};
      };
      const after = run(afterBlock);
      assert.equal(after.observed.published, fixtureCase.accepted, after.result.stdout + after.result.stderr);
      assert.equal(after.observed.preference, "Stop");
      if (fixtureCase.code !== null) {
        assert.equal(after.observed.status, fixtureCase.code);
        assert((after.result.stdout + after.result.stderr).includes(fixtureCase.stderr));
      }
      if (fixtureCase.code === 17) {
        assert.match(after.observed.failure, /MSBuild failed with exit code 17/);
      } else if (fixtureCase.code === null) {
        assert.equal(after.observed.status, null);
        assert(after.observed.failure.length > 0);
      } else if (!fixtureCase.accepted) {
        assert.match(after.observed.failure, /MSBuild ignored a compiler\/linker option/);
      } else {
        // Execute the original Stop capture with the same real native process: it rejects stderr+0.
        const tail = block.slice(block.indexOf("$msbuildOutput | ForEach-Object"));
        const before = run(nativeCall + "\n$msbuildExitCode = $LASTEXITCODE\n" + tail);
        assert.equal(before.observed.published, false);
        assert.match(before.observed.errorId, /NativeCommandError/);
        assert.equal(before.observed.preference, "Stop");
      }
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  });
}

test("Windows executable discovery selects one candidate and preserves explicit paths", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const result = runPowerShell(
      path.join(repo, "tests", "fixtures", "windows_executable_discovery.ps1"), [
        "-SourceRoot", repo,
        "-OutputRoot", path.join(fixture, "mom-executable-discovery"),
      ]);
    const output = result.stdout + result.stderr;
    assert.equal(result.status, 0, output);
    assert.equal(result.stdout.split(/\r?\n/).filter(line => line.startsWith("PASS ")).length, 17);
    assert.match(result.stdout,
      /^EXECUTABLE_DISCOVERY_FIXTURE_PASSED cases=17 native=false installer=false network=false\r?$/m);
    assert.doesNotMatch(output, /EXPECTED_FAIL/);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

/** @returns {string} */
function extractWindowsArchiveRunBlock() {
  const workflow = fs.readFileSync(
    path.join(repo, ".github", "workflows", "build-release-artifacts.yml"), "utf8");
  const windowsJob = workflow.indexOf("  warm-windows-acpp-cuda:");
  assert.notEqual(windowsJob, -1, "Windows release job is missing");
  const archiveTest = workflow.indexOf("      - name: Test extracted release archive", windowsJob);
  assert.notEqual(archiveTest, -1, "Windows archive test step is missing");
  const marker = "        run: |\n";
  const markerPosition = workflow.indexOf(marker, archiveTest);
  assert.notEqual(markerPosition, -1, "Windows archive test run block is missing");
  const lines = workflow.slice(markerPosition + marker.length).split("\n");
  const body = [];
  for (const line of lines) {
    if (!line.startsWith("          ")) {
      break;
    }
    body.push(line.slice(10));
  }
  assert.ok(body.length > 0, "Windows archive test run block is empty");
  return body.join("\n");
}

/** @param {string} workflowBlock @returns {string} */
function makeArchiveWorkflowDriverSource(workflowBlock) {
  return [
    "param(",
    "  [string]$ArchiveDirectory,",
    "  [string]$Events,",
    "  [int]$CpuStatus,",
    "  [int]$GpuStatus",
    ")",
    "$ErrorActionPreference = 'Stop'",
    "Set-Location -LiteralPath $ArchiveDirectory",
    "$script:Invocation = 0",
    "function powershell {",
    "  $script:Invocation++",
    "  $suiteIndex = [Array]::IndexOf([string[]]$args, '-Suite')",
    "  if ($suiteIndex -lt 0 -or $suiteIndex + 1 -ge $args.Count) { throw 'suite argument missing' }",
    "  Add-Content -LiteralPath $Events -Value $args[$suiteIndex + 1]",
    "  if ($script:Invocation -eq 1) { $global:LASTEXITCODE = $CpuStatus }",
    "  else { $global:LASTEXITCODE = $GpuStatus }",
    "}",
    workflowBlock,
  ].join("\n") + "\n";
}

test("Windows archive workflow gates the portable suite on the CPU suite exit status", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const archiveDirectory = path.join(fixture, "archive");
    const packageVersion = require(path.join(repo, "package.json")).version;
    const archive = path.join(archiveDirectory, `mom-v${packageVersion}-win.zip`);
    const events = path.join(fixture, "events.txt");
    fs.mkdirSync(archiveDirectory);
    fs.copyFileSync(path.join(repo, "package.json"),
      path.join(archiveDirectory, "package.json"));
    fs.writeFileSync(archive, "fixture archive");
    const driver = makeDriver(fixture,
      makeArchiveWorkflowDriverSource(extractWindowsArchiveRunBlock()));
    /** @type {Array<
     * {cpu: number, gpu: number, pass: true, calls: string[], error: null} |
     * {cpu: number, gpu: number, pass: false, calls: string[], error: RegExp}
     * >} */
    const cases = [
      {cpu: 1, gpu: 0, pass: false, calls: ["cpu"], error: /CPU release suite failed with exit code 1/},
      {
        cpu: 0, gpu: 1, pass: false, calls: ["cpu", "gpu-portable-cpu"],
        error: /GPU portable CPU release suite failed with exit code 1/,
      },
      {cpu: 0, gpu: 0, pass: true, calls: ["cpu", "gpu-portable-cpu"], error: null},
    ];
    for (const entry of cases) {
      fs.writeFileSync(events, "");
      const result = runPowerShell(driver, [
        "-ArchiveDirectory", archiveDirectory,
        "-Events", events,
        "-CpuStatus", String(entry.cpu),
        "-GpuStatus", String(entry.gpu),
      ]);
      const calls = fs.readFileSync(events, "utf8").split(/\r?\n/).filter(Boolean);
      assert.deepEqual(calls, entry.calls,
        "suite calls for " + entry.cpu + "," + entry.gpu + ": " +
        result.stdout + "\n" + result.stderr);
      if (entry.pass) {
        assert.equal(result.status, 0, result.stdout + result.stderr);
      } else {
        assert.notEqual(result.status, 0, result.stdout + result.stderr);
        assert.match(result.stdout + result.stderr, entry.error);
      }
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows release packager uses only tag refs as implicit versions", {
  skip: !WINDOWS,
}, () => {
  const source = fs.readFileSync(
    path.join(repo, ".github", "workflows", "scripts", "package-windows.ps1"), "utf8");
  const start = source.indexOf("if (-not $Version) {");
  const end = source.indexOf('\n. "$PSScriptRoot/windows-dll-deps.ps1"', start);
  assert.notEqual(start, -1, "version selection is missing");
  assert.notEqual(end, -1, "version selection boundary is missing");
  const fixture = makeFixture();
  try {
    fs.copyFileSync(path.join(repo, "package.json"), path.join(fixture, "package.json"));
    const driver = makeDriver(fixture, [
      "param([string]$RefType, [string]$RefName, [string]$Version)",
      "$ErrorActionPreference = 'Stop'",
      "Set-Location -LiteralPath $PSScriptRoot",
      "$env:GITHUB_REF_TYPE = $RefType",
      "$env:GITHUB_REF_NAME = $RefName",
      source.slice(start, end),
      "Write-Output $Version",
    ].join("\n"));
    const packageVersion = require(path.join(repo, "package.json")).version;
    const cases = [
      {type: "branch", ref: "123/merge", supplied: "", expected: packageVersion},
      {type: "branch", ref: "0.8-maintenance", supplied: "", expected: packageVersion},
      {type: "branch", ref: "master", supplied: "", expected: packageVersion},
      {type: "tag", ref: "v9.8.7", supplied: "", expected: "9.8.7"},
      {type: "", ref: "", supplied: "", expected: packageVersion},
      {type: "branch", ref: "123/merge", supplied: "7.6.5", expected: "7.6.5"},
      {type: "tag", ref: "v1/bad", supplied: "", expected: null},
      {type: "branch", ref: "master", supplied: "1/bad", expected: null},
    ];
    for (const entry of cases) {
      const args = [];
      if (entry.type) { args.push("-RefType", entry.type); }
      if (entry.ref) { args.push("-RefName", entry.ref); }
      if (entry.supplied) { args.push("-Version", entry.supplied); }
      const result = runPowerShell(driver, args);
      assert.equal(result.signal, null, result.stdout + result.stderr);
      if (entry.expected === null) {
        assert.equal(result.status, 1, result.stdout + result.stderr);
        assert.match(result.stdout + result.stderr, /Invalid release version/);
      } else {
        assert.equal(result.status, 0, result.stdout + result.stderr);
        assert.equal(result.stdout.trim(), entry.expected);
      }
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

const acppBaseBoundaryDriverSource = `
param([string]$ScriptPath, [string]$InstallDir, [string]$Workspace)
$ErrorActionPreference = 'Stop'
function git.exe {
  throw 'BUILD_STOP'
}
try {
  & $ScriptPath -InstallDir $InstallDir -Workspace $Workspace
  exit 71
} catch {
  if (-not $_.Exception.Message.Contains('BUILD_STOP')) {
    [Console]::Error.WriteLine(($_ | Out-String))
    exit 72
  }
  $marker = Join-Path $InstallDir '.mom-acpp-toolchain'
  if ((Test-Path -LiteralPath $marker -PathType Leaf) -and
      ((Get-Content -LiteralPath $marker -Raw) -eq 'mom AdaptiveCpp toolchain')) {
    exit 0
  }
  exit 73
}
`;

const acppBootstrapRetryDriverSource = `
param([string]$ScriptPath, [string]$InstallDir, [string]$Workspace)
$ErrorActionPreference = 'Stop'
function curl.exe { throw 'BOOTSTRAP_DOWNLOAD' }
function git.exe { throw 'GIT_CALLED' }
try {
  & $ScriptPath -InstallDir $InstallDir -Workspace $Workspace
  exit 81
} catch {
  if ($_.Exception.Message.Contains('BOOTSTRAP_DOWNLOAD')) { exit 0 }
  [Console]::Error.WriteLine(($_ | Out-String))
  exit 82
}
`;

const installDevOpenClCpuTailDriverSource = String.raw`
param([string]$ScriptPath, [string]$ToolsRoot)
$ErrorActionPreference = 'Stop'
$source = [IO.File]::ReadAllText($ScriptPath)
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput(
  $source, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) { throw 'INSTALLER_AST_PARSE_FAILED' }

function Get-TopLevelCommandName($statement) {
  if ($statement -isnot [System.Management.Automation.Language.PipelineAst]) { return $null }
  $elements = @($statement.PipelineElements)
  if ($elements.Count -ne 1 -or
      $elements[0] -isnot [System.Management.Automation.Language.CommandAst]) { return $null }
  return $elements[0].GetCommandName()
}

$statements = @($ast.EndBlock.Statements)
$boundary = -1
for ($index = 0; $index + 1 -lt $statements.Count; $index++) {
  $first = Get-TopLevelCommandName $statements[$index]
  $second = Get-TopLevelCommandName $statements[$index + 1]
  if ($first -eq 'Require-Administrator' -and $second -eq 'Initialize-Workspace') {
    if ($boundary -ge 0) { throw 'INSTALLER_TAIL_BOUNDARY_AMBIGUOUS' }
    $boundary = $statements[$index].Extent.StartOffset
  }
}
if ($boundary -lt 0) { throw 'INSTALLER_TAIL_BOUNDARY_MISSING' }

$literalToolsRoot = $ToolsRoot.Replace("'", "''")
$tail = $source.Substring($boundary)
$safeTail = $tail.Replace('C:\Tools', $literalToolsRoot)
if ($safeTail.Contains('C:\Tools')) { throw 'REAL_TOOLS_PATH_REMAINED' }

$script:RuntimeInstalled = $false
$script:InstallCalls = 0
function Require-Administrator { }
function Initialize-Workspace { }
function Install-Base { throw 'UNREQUESTED_BUILD_TOOLS' }
function Install-OpenClCpu {
  $script:RuntimeInstalled = $true
  $script:InstallCalls++
}
function Test-Component([string]$Name) {
  if ($Name -ne 'opencl-cpu') { throw 'UNREQUESTED_COMPONENT' }
  return [bool]$script:RuntimeInstalled
}

$components = @('opencl-cpu')
$KeepWorkspace = $true
Invoke-Expression $safeTail
$manifest = Join-Path $ToolsRoot 'mom-toolchains.txt'
if (-not $script:RuntimeInstalled) { throw 'OPENCL_CPU_RUNTIME_NOT_INSTALLED' }
if ($script:InstallCalls -ne 1) { throw 'OPENCL_CPU_DISPATCH_COUNT_INVALID' }
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
  throw 'TOOLS_MANIFEST_NOT_CREATED'
}
`;

/** @param {string} driver @param {string[]} args @param {string} label */
function assertRejected(driver, args, label) {
  const result = runPowerShell(driver, args);
  assert.equal(result.status, 0,
    `${label}: expected a guarded rejection, got ${result.status}\n${result.stdout}\n${result.stderr}`);
}

test("Windows release script propagates terminating PowerShell failures", {
  skip: !WINDOWS,
}, () => {
  const result = runPowerShell(
    path.join(repo, ".github/workflows/scripts/test-release-windows.ps1"),
    ["-Archive", path.join(repo, "missing-release-fixture.zip")],
  );
  assert.notEqual(result.status, 0, result.stdout + result.stderr);
  assert.match(result.stdout + result.stderr, /missing-release-fixture\.zip/);
});

test("Windows toolchain readiness checks the artifacts consumed by each build", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  const cuda = path.join(fixture, "cuda");
  const hip = path.join(fixture, "hip");
  const dpcpp = path.join(fixture, "dpcpp");
  /** @param {string} root @param {string[]} files */
  const writeFiles = (root, files) => {
    for (const file of files) {
      const destination = path.join(root, ...file.split("/"));
      fs.mkdirSync(path.dirname(destination), {recursive: true});
      fs.writeFileSync(destination, "fixture");
    }
  };
  writeFiles(cuda, [
    "bin/nvcc.exe", "bin/ptxas.exe", "nvvm/libdevice/libdevice.10.bc",
    "include/cuda.h", "include/cuda_runtime.h", "include/nvrtc.h",
    "lib/x64/cuda.lib", "lib/x64/cudart_static.lib", "lib/x64/nvrtc.lib",
  ]);
  writeFiles(hip, [
    "lib/amdhip64.lib", "include/hip/hip_runtime_api.h", "include/hip/hiprtc.h",
    "bin/clang++.exe", "bin/llvm-link.exe", "bin/opt.exe", "bin/llc.exe",
    "bin/lld-link.exe", "amdgcn/bitcode/ockl.bc",
  ]);
  writeFiles(dpcpp, [
    "bin/clang++.exe", "bin/clang.exe", "bin/lld-link.exe", "bin/sycl-post-link.exe",
    "bin/llvm-spirv.exe", "bin/llvm-link.exe", "bin/clang-linker-wrapper.exe",
    "bin/clang-offload-wrapper.exe", "bin/sycl9.dll", "bin/sycl-jit.dll",
    "bin/ur_loader.dll", "bin/ur_adapter_cuda.dll", "bin/ur_adapter_opencl.dll",
    "bin/ur_adapter_level_zero_v2.dll",
  ]);
  const driver = makeDriver(fixture, `
param([string]$Helper, [string]$Cuda, [string]$Hip, [string]$Dpcpp)
$ErrorActionPreference = 'Stop'
. $Helper
[Console]::Write(('cuda={0};hip={1};dpcpp={2}' -f
  (Test-MomCudaCompilerSdk $Cuda), (Test-MomHipSdk $Hip), (Test-MomDpcppToolchain $Dpcpp)))
`);
  /** @returns {string} */
  const status = () => {
    const result = runPowerShell(driver, [
      "-Helper", path.join(repo, "scripts/windows-install-helpers.ps1"),
      "-Cuda", cuda, "-Hip", hip, "-Dpcpp", dpcpp,
    ]);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    return result.stdout.trim();
  };
  try {
    assert.equal(status(), "cuda=False;hip=True;dpcpp=False",
      "a CUDA SDK without the stable UR adapter's NVML prerequisite must not qualify");
    writeFiles(cuda, ["include/nvml.h"]);
    assert.equal(status(), "cuda=False;hip=True;dpcpp=False");
    writeFiles(cuda, ["lib/x64/nvml.lib"]);
    assert.equal(status(), "cuda=True;hip=True;dpcpp=False");
    fs.rmSync(path.join(cuda, "include/nvml.h"));
    assert.equal(status(), "cuda=False;hip=True;dpcpp=False");
    writeFiles(cuda, ["include/nvml.h"]);
    fs.rmSync(path.join(cuda, "lib/x64/nvml.lib"));
    assert.equal(status(), "cuda=False;hip=True;dpcpp=False");
    writeFiles(cuda, ["lib/x64/nvml.lib"]);
    writeFiles(dpcpp, ["bin/ur_win_proxy_loader.dll"]);
    assert.equal(status(), "cuda=True;hip=True;dpcpp=True");
    fs.rmSync(path.join(dpcpp, "bin/ur_loader.dll"));
    assert.equal(status(), "cuda=True;hip=True;dpcpp=False",
      "a proxy without its matching real UR loader must not qualify the toolchain");
    const archive = path.join(fixture, "previous-toolchain.tar.gz");
    fs.writeFileSync(archive, "preserve existing asset");
    fs.writeFileSync(archive + ".sha256", "preserve existing digest");
    fs.mkdirSync(path.join(dpcpp, "include"));
    fs.mkdirSync(path.join(dpcpp, "lib"));
    const packaging = runPowerShell(
      path.join(repo, ".github", "workflows", "scripts", "package-toolchain-win.ps1"),
      ["-BuildDir", dpcpp, "-OutFile", archive]);
    assert.notEqual(packaging.status, 0, packaging.stdout + packaging.stderr);
    assert.match(packaging.stdout + packaging.stderr, /matching real UR loader/);
    assert.equal(fs.readFileSync(archive, "utf8"), "preserve existing asset");
    assert.equal(fs.readFileSync(archive + ".sha256", "utf8"), "preserve existing digest");
    writeFiles(dpcpp, ["bin/ur_loader.dll"]);
    fs.rmSync(path.join(cuda, "lib/x64/cudart_static.lib"));
    fs.rmSync(path.join(hip, "include/hip/hiprtc.h"));
    assert.equal(status(), "cuda=False;hip=False;dpcpp=True");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows portable source runtime ownership is explicit", () => {
  const build = fs.readFileSync(
    path.join(repo, ".github/workflows/scripts/build-windows-multicompiler.ps1"), "utf8");
  const portable = build.match(/Save-Compiler dpcpp-opencl\r?\n([\s\S]*?)\r?\n {2}\}/)?.[1];
  assert.ok(portable, "portable source-build branch is missing");
  assert.match(portable, /^ {4}Save-DpcppRuntime -Name dpcpp-opencl$/m);
  assert.doesNotMatch(portable, /^ {4}Save-DpcppRuntime\s*$/m,
    "portable rebuild must not refresh the preserved CUDA worker");
});

test("Windows portable source runtime staging and consumer preserve the CUDA sibling", {
  skip: !WINDOWS,
}, () => {
  const source = fs.readFileSync(
    path.join(repo, ".github/workflows/scripts/build-windows-multicompiler.ps1"), "utf8");
  const start = source.indexOf("function Save-DpcppRuntime(");
  const end = source.indexOf("\n}\n\ntry {", start);
  const portable = source.match(/Save-Compiler dpcpp-opencl\r?\n([\s\S]*?)\r?\n {2}\}/)?.[1];
  const call = portable?.match(/^ {4}(Save-DpcppRuntime[^\r\n]*)$/m)?.[1];
  assert.ok(start >= 0 && end > start && call, "source runtime staging contract is missing");
  const consumer = fs.readFileSync(
    path.join(repo, "scripts/test-windows-current-multicompiler.ps1"), "utf8");
  const selectorStart = consumer.indexOf("function Clear-SelectorEnvironment {");
  const selectorEnd = consumer.indexOf("\nfunction Invoke-GpuSuite(", selectorStart);
  const consumerStart = consumer.indexOf("function Invoke-PortableSuite(");
  const consumerEnd = consumer.indexOf("\n}\n\n$compilerLanes", consumerStart);
  assert.ok(selectorStart >= 0 && selectorEnd > selectorStart &&
    consumerStart >= 0 && consumerEnd > consumerStart, "portable suite consumer is missing");
  for (const sibling of [false, true]) {
    const fixture = makeFixture();
    try {
      const cohort = ["sycl9.dll", "sycl-jit.dll", "ur_win_proxy_loader.dll", "ur_loader.dll",
        "ur_adapter_opencl.dll", "ur_adapter_level_zero_v2.dll"];
      /** @type {Array<[string, string, string[]]>} */
      const families = [
        ["toolchain/bin", "portable-current", cohort],
        ["workers/dpcpp-opencl", "portable-stale", [...cohort, "mom.node", "sycl.dll"]],
      ];
      if (sibling) {
        families.push(["workers/dpcpp", "cuda-preserved",
          [...cohort, "mom.node", "sycl.dll", "ur_adapter_cuda.dll"]]);
      }
      for (const [relative, family, files] of families) {
        const directory = path.join(fixture, relative);
        fs.mkdirSync(directory, {recursive: true});
        for (const file of files) {
          fs.writeFileSync(path.join(directory, file), `MZ${family}:${file}`);
        }
      }
      const driver = makeDriver(fixture, String.raw`
param([string]$Root, [string]$Helper, [switch]$DropPortableRuntime)
$ErrorActionPreference = 'Stop'
. $Helper
$out = Join-Path $Root 'workers'
$DpcppDir = Join-Path $Root 'toolchain'
$nativeDir = $out
$basePath = 'fixture-base-path'
$script:NodeCalls = 0
Remove-Item Env:MOM_DPCPP_DIR, Env:MOM_ACPP_DIR, Env:HIP_PATH, Env:ROCM_PATH, Env:ONEAPI_ROOT, Env:CUDA_PATH -ErrorAction SilentlyContinue
# Bound discovery to the synthetic cohort; do not inspect the host or load native code.
function Get-MominerCudaBinDir { return @() }
function Get-MominerHipBinDir { return @() }
function Copy-MominerDllClosure {
  param([string]$PackageDir, [string[]]$EntryPaths)
  if ($PackageDir -ne (Join-Path $out 'dpcpp-opencl')) { throw 'closure targeted the CUDA sibling' }
}
function node.exe {
  if ($args.Count -ne 2 -or $args[0] -ne '.\tests\run_hash.js' -or $args[1] -ne 'gpu') {
    throw 'unexpected portable suite invocation'
  }
  $portable = Join-Path $out 'dpcpp-opencl'
  if ($env:Path -ne "$portable;$basePath") { throw 'portable suite borrowed another runtime' }
  if ($env:MOM_NATIVE_PATH -or $env:MOM_GPU_TEST_VENDORS -or
      $env:ONEAPI_DEVICE_SELECTOR -or $env:ZE_AFFINITY_MASK -or $env:ACPP_VISIBILITY_MASK -or
      $env:MOM_COMPILER_POLICY_STRICT -ne '1' -or $env:MOM_REQUIRE_PORTABLE_CPU_TESTS -ne '1' -or
      $env:MOM_GPU_BACKEND -ne 'opencl') { throw 'portable suite selector policy changed' }
  foreach ($file in @('sycl9.dll', 'sycl-jit.dll', 'ur_win_proxy_loader.dll', 'ur_loader.dll',
                     'ur_adapter_opencl.dll', 'ur_adapter_level_zero_v2.dll')) {
    if ([IO.File]::ReadAllText((Join-Path $portable $file)) -ne "MZportable-current:$file") {
      throw "portable suite used a foreign cohort: $file"
    }
  }
  $script:NodeCalls++
  $global:LASTEXITCODE = 0
}
` + source.slice(start, end + 2) + "\n" + call + "\n" +
      consumer.slice(selectorStart, selectorEnd) + consumer.slice(consumerStart, consumerEnd + 2) + String.raw`
if ($DropPortableRuntime) { Remove-Item -LiteralPath (Join-Path $out 'dpcpp-opencl\sycl9.dll') }
$env:MOM_NATIVE_PATH = 'stale-addon'
$env:MOM_GPU_TEST_VENDORS = 'cuda'
$env:ONEAPI_DEVICE_SELECTOR = 'stale'
$env:ZE_AFFINITY_MASK = 'stale'
$env:ACPP_VISIBILITY_MASK = 'stale'
Invoke-PortableSuite 'fixture'
if ($script:NodeCalls -ne 1) { throw 'portable suite did not reach its runner exactly once' }
`);
      const result = runPowerShell(driver, ["-Root", fixture, "-Helper",
        path.join(repo, ".github/workflows/scripts/windows-dll-deps.ps1")]);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      for (const file of cohort) {
        assert.equal(fs.readFileSync(path.join(fixture, "workers/dpcpp-opencl", file), "utf8"),
          `MZportable-current:${file}`);
      }
      if (sibling) {
        const missing = runPowerShell(driver, ["-Root", fixture, "-Helper",
          path.join(repo, ".github/workflows/scripts/windows-dll-deps.ps1"), "-DropPortableRuntime"]);
        assert.notEqual(missing.status, 0, missing.stdout + missing.stderr);
        assert.match(missing.stdout + missing.stderr,
          /Portable DPC\+\+ worker dependency is missing:[\s\S]*?dpcpp-opencl[\\/]sycl9\.dll/);
        for (const file of [...cohort, "mom.node", "sycl.dll", "ur_adapter_cuda.dll"]) {
          assert.equal(fs.readFileSync(path.join(fixture, "workers/dpcpp", file), "utf8"),
            `MZcuda-preserved:${file}`);
        }
      } else {
        assert.equal(fs.existsSync(path.join(fixture, "workers/dpcpp")), false,
          "a targeted portable build must not need or create a CUDA sibling");
      }
      for (const file of ["mom.node", "sycl.dll"]) {
        assert.equal(fs.readFileSync(path.join(fixture, "workers/dpcpp-opencl", file), "utf8"),
          `MZportable-stale:${file}`);
      }
    } finally {
      fs.rmSync(fixture, {recursive: true, force: true});
    }
  }
});

test("Windows packaging preserves distinct coherent UR runtime cohorts", (t) => {
  const source = fs.readFileSync(
    path.join(repo, ".github", "workflows", "scripts", "package-windows.ps1"), "utf8");
  const assertions = source.slice(source.indexOf("function Assert-BuildArtifact"),
    source.indexOf("if (Test-Path -LiteralPath $Archive"));
  const cleanupStart = source.indexOf("foreach ($worker in @('dpcpp', 'dpcpp-opencl')) {");
  const cleanupEnd = source.indexOf("$hipPackageDir = ", cleanupStart);
  assert.ok(cleanupStart >= 0 && cleanupEnd > cleanupStart,
    "release packaging must strip both source runtime snapshots");
  const start = source.indexOf("$savedDpcppAcpp = ");
  const end = source.indexOf("# Ship the KawPow source-JIT assets", start);
  assert.ok(assertions.startsWith("function Assert-BuildArtifact") && start >= 0 && end > start);
  if (!WINDOWS) {
    t.skip("actual runtime-copy behavior requires Windows PowerShell");
    return;
  }
  const fixture = makeFixture();
  try {
    const driver = makeDriver(fixture, String.raw`
param([string]$Root, [string]$Helper)
$ErrorActionPreference = 'Stop'
. $Helper
$libsDir = Join-Path $Root 'libs'
$env:ONEAPI_ROOT = Join-Path $Root 'oneapi'
$env:MOM_DPCPP_DIR = Join-Path $Root 'nightly'
Remove-Item Env:MOM_DPCPP_CUDA_DIR, Env:MOM_ACPP_DIR, Env:HIP_PATH, Env:ROCM_PATH, Env:CUDA_PATH -ErrorAction SilentlyContinue
` + assertions + source.slice(cleanupStart, cleanupEnd) + source.slice(start, end));
    const portable = ["sycl9.dll", "ur_win_proxy_loader.dll", "ur_loader.dll", "OpenCL.dll",
      "ur_adapter_opencl.dll", "ur_adapter_level_zero_v2.dll", "umf.dll", "libhwloc-15.dll", "libmmd.dll"];
    const nightly = [...portable, "sycl-jit.dll", "ur_adapter_cuda.dll"];
    const modes = ["complete", "missing-nightly-loader", "missing-nightly-adapter", "missing-oneapi-adapter",
      "missing-oneapi-proxy", "missing-oneapi-umf", "missing-oneapi-hwloc",
      "missing-oneapi-jit", "non-pe-oneapi-loader"];
    for (const mode of modes) {
      const root = path.join(fixture, mode);
      /** @type {Array<[string, string, string[]]>} */
      const families = [
        [path.join(root, "nightly", "bin"), "nightly", nightly],
        [path.join(root, "oneapi", "compiler", "latest", "bin"), "oneapi", [...portable, "sycl-jit.dll"]],
        [path.join(root, "libs", "oneapi"), "stale-nightly", [...portable, "sycl-jit.dll"]],
        [path.join(root, "libs", "dpcpp"), "stale-nightly", [...nightly, "mom.node", "sycl.dll", "source-only.txt"]],
        [path.join(root, "libs", "dpcpp-opencl"), "stale-nightly", [...nightly, "mom.node", "sycl.dll", "source-only.txt"]],
      ];
      for (const [directory, family, files] of families) {
        fs.mkdirSync(directory, {recursive: true});
        for (const file of files) {
          fs.writeFileSync(path.join(directory, file), `MZ${family}:${file}`);
        }
      }
      for (const worker of ["dpcpp", "dpcpp-opencl"]) {
        fs.mkdirSync(path.join(root, "libs", worker, "compiler-only"));
        fs.writeFileSync(path.join(root, "libs", worker, "compiler-only", "nested.dll"), "stale");
      }
      if (mode === "missing-nightly-loader") {
        fs.rmSync(path.join(root, "nightly", "bin", "ur_loader.dll"));
      } else if (mode === "missing-nightly-adapter") {
        fs.rmSync(path.join(root, "nightly", "bin", "ur_adapter_opencl.dll"));
      } else if (mode === "missing-oneapi-adapter") {
        fs.rmSync(path.join(root, "oneapi", "compiler", "latest", "bin", "ur_adapter_opencl.dll"));
      } else if (mode === "missing-oneapi-proxy") {
        fs.rmSync(path.join(root, "oneapi", "compiler", "latest", "bin", "ur_win_proxy_loader.dll"));
      } else if (mode === "missing-oneapi-umf") {
        fs.rmSync(path.join(root, "oneapi", "compiler", "latest", "bin", "umf.dll"));
      } else if (mode === "missing-oneapi-hwloc") {
        fs.rmSync(path.join(root, "oneapi", "compiler", "latest", "bin", "libhwloc-15.dll"));
      } else if (mode === "missing-oneapi-jit") {
        fs.rmSync(path.join(root, "oneapi", "compiler", "latest", "bin", "sycl-jit.dll"));
      } else if (mode === "non-pe-oneapi-loader") {
        fs.writeFileSync(path.join(root, "oneapi", "compiler", "latest", "bin", "ur_loader.dll"),
          "not a PE binary");
      }
      const result = runPowerShell(driver, ["-Root", root, "-Helper",
        path.join(repo, ".github", "workflows", "scripts", "windows-dll-deps.ps1")]);
      const output = result.stdout + result.stderr;
      if (mode !== "complete") {
        assert.notEqual(result.status, 0, `${mode}: missing runtime unexpectedly accepted\n${output}`);
        assert.match(output, mode === "missing-nightly-loader"
          ? /matching DPC\+\+ Unified Runtime loader/ : mode === "missing-nightly-adapter"
            ? /missing required Unified Runtime file/ : mode === "non-pe-oneapi-loader"
              ? /not a Windows PE binary/ : mode === "missing-oneapi-jit"
                ? /matching oneAPI JIT library/ : /complete oneAPI runtime cohort/);
        continue;
      }
      assert.equal(result.status, 0, output);
      for (const file of nightly) {
        assert.equal(fs.readFileSync(path.join(root, "libs", "dpcpp", file), "utf8"),
          `MZnightly:${file}`, `DPC++ must preserve its own ${file}`);
      }
      for (const file of portable) {
        assert.equal(fs.readFileSync(path.join(root, "libs", "oneapi", file), "utf8"),
          `MZoneapi:${file}`, `oneAPI must replace a stale snapshot ${file}`);
        assert.equal(fs.readFileSync(path.join(root, "libs", "dpcpp-opencl", file), "utf8"),
          `MZoneapi:${file}`, `portable must replace a stale nightly ${file}`);
      }
      assert.equal(fs.readFileSync(path.join(root, "libs", "oneapi", "sycl-jit.dll"), "utf8"),
        "MZoneapi:sycl-jit.dll");
      assert.equal(fs.existsSync(path.join(root, "libs", "dpcpp-opencl", "sycl-jit.dll")), false,
        "the portable worker must share the matching JIT without duplicating it");
      assert.equal(fs.existsSync(path.join(root, "libs", "dpcpp-opencl", "ur_adapter_cuda.dll")), false,
        "portable release must not retain its source-local nightly CUDA adapter");
      for (const worker of ["dpcpp", "dpcpp-opencl"]) {
        for (const file of ["mom.node", "sycl.dll"]) {
          assert.equal(fs.readFileSync(path.join(root, "libs", worker, file), "utf8"),
            `MZstale-nightly:${file}`, "runtime refresh must preserve native binaries");
        }
        assert.equal(fs.existsSync(path.join(root, "libs", worker, "source-only.txt")), false);
        assert.equal(fs.existsSync(path.join(root, "libs", worker, "compiler-only")), false);
      }
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows release workspace rejects a reparse ancestor before extraction", {
  skip: !WINDOWS,
}, (/** @type {import("node:test").TestContext} */ t) => {
  const fixture = makeFixture();
  try {
    const target = path.join(fixture, "target");
    const parent = path.join(fixture, "release-parent");
    const sentinel = path.join(target, "do-not-delete.txt");
    fs.mkdirSync(target);
    fs.writeFileSync(sentinel, "sentinel");
    if (!tryMakeJunction(parent, target)) {
      t.skip("junction creation is unavailable");
      return;
    }
    const result = spawnSync("powershell.exe", [
      "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
      path.join(repo, ".github/workflows/scripts/test-release-windows.ps1"),
      "-Archive", path.join(fixture, "missing.zip"),
    ], {
      encoding: "utf8",
      windowsHide: true,
      timeout: 30000,
      env: {...process.env, MOM_RELEASE_TEST_DIR: path.join(parent, "mom-release-work")},
    });
    assert.notEqual(result.status, 0, result.stdout + result.stderr);
    assert.match(result.stdout + result.stderr, /reparse point/);
    assert.equal(fs.readFileSync(sentinel, "utf8"), "sentinel");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows release validator accepts backslash roots and rejects unsafe inventories", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  const releaseRoot = `mom-v${require(path.join(repo, "package.json")).version}`;
  const archiveBuilder = makeDriver(fixture, `
param([string]$Archive, [string]$Mode)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
$stream = [IO.File]::Open($Archive, [IO.FileMode]::CreateNew)
$zip = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Create)
try {
  $names = if ($Mode -eq 'duplicate') {
    @('${releaseRoot}/payload', '${releaseRoot}/payload')
  } elseif ($Mode -eq 'multiple') {
    @('${releaseRoot}/payload', 'other-root/payload')
  } elseif ($Mode -eq 'missing-root') {
    @('${releaseRoot}/payload')
  } else {
    @('${releaseRoot}\\', '${releaseRoot}\\payload')
  }
  foreach ($name in $names) {
    $entry = $zip.CreateEntry($name)
    if ($name.EndsWith('\\')) { continue }
    $writer = [IO.StreamWriter]::new($entry.Open())
    try { $writer.Write('fixture') } finally { $writer.Dispose() }
  }
} finally {
  $zip.Dispose()
  $stream.Dispose()
}
`);
  const validator = path.join(repo, ".github/workflows/scripts/test-release-windows.ps1");
  try {
    /** @type {Array<[string, RegExp | null]>} */
    const cases = [
      ["duplicate", /duplicate member/i],
      ["multiple", /expected package root/i],
      ["missing-root", /explicit package root directory/i],
      ["backslash-root", null],
    ];
    for (const [mode, expected] of cases) {
      const archive = path.join(fixture, `${mode}.zip`);
      const created = runPowerShell(archiveBuilder, ["-Archive", archive, "-Mode", mode]);
      assert.equal(created.status, 0, created.stdout + created.stderr);
      const result = spawnSync("powershell.exe", [
        "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", validator,
        "-Archive", archive,
      ], {
        encoding: "utf8",
        windowsHide: true,
        timeout: 30000,
        env: {...process.env,
          MOM_RELEASE_ARCHIVE_VALIDATION_ONLY: "1",
          MOM_RELEASE_TEST_DIR: path.join(fixture, `mom-release-${mode}-work`)},
      });
      if (expected) {
        assert.notEqual(result.status, 0, result.stdout + result.stderr);
        assert.match(result.stdout + result.stderr, expected);
      } else {
        assert.equal(result.status, 0, result.stdout + result.stderr);
      }
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows AdaptiveCpp builders reject broad destinations and unowned workspaces", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const systemDrive = process.env["SystemDrive"];
    assert.ok(systemDrive);
    const baseBuilder = path.join(repo, "scripts", "build-windows-adaptivecpp-base.ps1");
    const amdBuilder = path.join(repo, "scripts", "build-windows-adaptivecpp-amd.ps1");
    const unownedWorkspace = path.join(fixture, "mom-dev-unowned");
    const sentinel = path.join(unownedWorkspace, "do-not-delete.txt");
    fs.mkdirSync(unownedWorkspace);
    fs.writeFileSync(sentinel, "sentinel");

    const broad = runPowerShell(baseBuilder, [
      "-InstallDir", `${systemDrive}\\`, "-Workspace", unownedWorkspace,
    ]);
    assert.notEqual(broad.status, 0, broad.stdout + broad.stderr);
    assert.match(broad.stdout + broad.stderr, /dedicated acpp-/);

    /** @type {Array<{script: string, args: string[]}>} */
    const cases = [
      {script: baseBuilder, args: ["-InstallDir", path.join(fixture, "acpp-test"),
        "-Workspace", unownedWorkspace]},
      {script: amdBuilder, args: ["-Workspace", unownedWorkspace,
        "-BaseToolchain", path.join(fixture, "acpp-base")]},
    ];
    for (const entry of cases) {
      const result = runPowerShell(entry.script, entry.args);
      assert.notEqual(result.status, 0, result.stdout + result.stderr);
      assert.match(result.stdout + result.stderr, /initialized by scripts\\install-dev\.bat/);
    }
    assert.equal(fs.readFileSync(sentinel, "utf8"), "sentinel");

    const ownedWorkspace = path.join(fixture, "mom-dev-owned");
    const unownedToolchain = path.join(fixture, "acpp-unowned");
    const toolchainSentinel = path.join(unownedToolchain, "do-not-delete.txt");
    fs.mkdirSync(ownedWorkspace);
    fs.writeFileSync(path.join(ownedWorkspace, ".mom-dev-workspace"),
      "mom development workspace");
    fs.mkdirSync(unownedToolchain);
    fs.writeFileSync(toolchainSentinel, "sentinel");
    const unownedDestination = runPowerShell(baseBuilder, [
      "-InstallDir", unownedToolchain, "-Workspace", ownedWorkspace,
    ]);
    assert.notEqual(unownedDestination.status, 0,
      unownedDestination.stdout + unownedDestination.stderr);
    assert.match(unownedDestination.stdout + unownedDestination.stderr, /ownership marker/);
    assert.equal(fs.readFileSync(toolchainSentinel, "utf8"), "sentinel");

    fs.writeFileSync(path.join(unownedToolchain, ".mom-acpp-toolchain"), "wrong owner");
    const unownedBase = runPowerShell(amdBuilder, [
      "-Workspace", ownedWorkspace, "-BaseToolchain", unownedToolchain,
    ]);
    assert.notEqual(unownedBase.status, 0, unownedBase.stdout + unownedBase.stderr);
    assert.match(unownedBase.stdout + unownedBase.stderr, /owned acpp-\* toolchain/);
    assert.equal(fs.readFileSync(toolchainSentinel, "utf8"), "sentinel");

    const nestedBase = path.join(ownedWorkspace, "acpp-toolchain");
    const nestedSentinel = path.join(nestedBase, "do-not-delete.txt");
    fs.mkdirSync(nestedBase);
    fs.writeFileSync(path.join(nestedBase, ".mom-acpp-toolchain"),
      "mom AdaptiveCpp toolchain");
    fs.writeFileSync(nestedSentinel, "sentinel");
    const nestedBaseResult = runPowerShell(amdBuilder, [
      "-Workspace", ownedWorkspace, "-BaseToolchain", nestedBase,
    ]);
    assert.notEqual(nestedBaseResult.status, 0,
      nestedBaseResult.stdout + nestedBaseResult.stderr);
    assert.match(nestedBaseResult.stdout + nestedBaseResult.stderr, /must be separate directories/);
    assert.equal(fs.readFileSync(nestedSentinel, "utf8"), "sentinel");

    const nestedHip = path.join(ownedWorkspace, "acpp-hip");
    fs.mkdirSync(nestedHip);
    fs.writeFileSync(path.join(unownedToolchain, ".mom-acpp-toolchain"),
      "mom AdaptiveCpp toolchain");
    fs.writeFileSync(path.join(nestedHip, ".mom-acpp-toolchain"),
      "mom AdaptiveCpp toolchain");
    const nestedDestination = runPowerShell(path.join(repo, "scripts", "install-dev.ps1"), [
      "-ValidateOnly", "-Component", "acpp-hip", "-Workspace", ownedWorkspace,
      "-AcppCudaDir", unownedToolchain, "-AcppHipDir", nestedHip,
    ]);
    assert.notEqual(nestedDestination.status, 0,
      nestedDestination.stdout + nestedDestination.stderr);
    assert.match(nestedDestination.stdout + nestedDestination.stderr,
      /must be separate from -Workspace/);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows standalone opencl-cpu dispatch creates its tools directory before the manifest write", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const toolsRoot = path.join(fixture, "tools-created-by-installer-tail");
    const manifest = path.join(toolsRoot, "mom-toolchains.txt");
    assert.equal(fs.existsSync(toolsRoot), false);
    const driver = makeDriver(fixture, installDevOpenClCpuTailDriverSource);
    const args = [
      "-ScriptPath", path.join(repo, "scripts", "install-dev.ps1"),
      "-ToolsRoot", toolsRoot,
    ];

    const first = runPowerShell(driver, args);
    assert.equal(first.status, 0, first.stdout + first.stderr);
    assert.equal(fs.statSync(toolsRoot).isDirectory(), true);
    assert.equal(fs.statSync(manifest).isFile(), true);

    const second = runPowerShell(driver, args);
    assert.equal(second.status, 0, second.stdout + second.stderr);
    assert.equal(fs.statSync(manifest).isFile(), true);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows AdaptiveCpp base owns an empty destination before compiling", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const workspace = path.join(fixture, "mom-dev-interrupted");
    const install = path.join(fixture, "acpp-interrupted");
    fs.mkdirSync(path.join(workspace, "llvm-bootstrap", "bin"), {recursive: true});
    fs.writeFileSync(path.join(workspace, ".mom-dev-workspace"),
      "mom development workspace");
    fs.writeFileSync(path.join(workspace, "llvm-bootstrap", "bin", "clang-cl.exe"), "fixture");
    fs.writeFileSync(path.join(workspace, "llvm-bootstrap", "bin", "lld-link.exe"), "fixture");
    fs.writeFileSync(path.join(workspace, "llvm-bootstrap", ".mom-llvm-bootstrap"),
      "f19ae5bc4823ac69ec01dc2ded503ec80a04ad2208dda1595d1f0413c148ef90");
    fs.mkdirSync(install);
    const driver = makeDriver(fixture, acppBaseBoundaryDriverSource);
    const result = runPowerShell(driver, [
      "-ScriptPath", path.join(repo, "scripts", "build-windows-adaptivecpp-base.ps1"),
      "-InstallDir", install, "-Workspace", workspace,
    ]);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.equal(fs.readFileSync(path.join(install, ".mom-acpp-toolchain"), "utf8"),
      "mom AdaptiveCpp toolchain");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows AdaptiveCpp base retries an incomplete LLVM bootstrap", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const workspace = path.join(fixture, "mom-dev-partial-bootstrap");
    const install = path.join(fixture, "acpp-partial-bootstrap");
    fs.mkdirSync(path.join(workspace, "llvm-bootstrap", "bin"), {recursive: true});
    fs.writeFileSync(path.join(workspace, ".mom-dev-workspace"),
      "mom development workspace");
    fs.writeFileSync(path.join(workspace, "llvm-bootstrap", "bin", "clang-cl.exe"), "partial");
    const driver = makeDriver(fixture, acppBootstrapRetryDriverSource);
    const result = runPowerShell(driver, [
      "-ScriptPath", path.join(repo, "scripts", "build-windows-adaptivecpp-base.ps1"),
      "-InstallDir", install, "-Workspace", workspace,
    ]);
    assert.equal(result.status, 0, result.stdout + result.stderr);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows DPCPP CUDA build fails closed when CUDA is required but missing", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const toolchain = path.join(fixture, "toolchain");
    fs.mkdirSync(path.join(toolchain, "bin"), {recursive: true});
    fs.writeFileSync(path.join(toolchain, "bin", "clang++.exe"), "fixture");
    fs.writeFileSync(path.join(toolchain, "bin", "clang.exe"), "fixture");
    fs.writeFileSync(path.join(toolchain, "bin", "lld-link.exe"), "fixture");
    const result = runPowerShell(
      path.join(repo, ".github/workflows/scripts/build-sycl-cuda-win.ps1"),
      ["-ToolchainDir", toolchain, "-CudaPath", path.join(fixture, "missing-cuda"),
        "-RequireCuda"],
    );
    assert.notEqual(result.status, 0, result.stdout + result.stderr);
    assert.match(result.stdout + result.stderr, /CudaPath does not resolve/);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows DPCPP CUDA build rejects a nonpositive MOM_BUILD_JOBS", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const driver = makeDriver(fixture, `
param([string]$ScriptPath)
$ErrorActionPreference = 'Stop'
$env:MOM_BUILD_JOBS = '0'
try {
  & $ScriptPath -ToolchainDir (Join-Path $PSScriptRoot 'missing-toolchain') -CudaPath ''
  exit 91
} catch {
  if ($_.Exception.Message.Contains('MOM_BUILD_JOBS must be a positive integer')) { exit 0 }
  [Console]::Error.WriteLine(($_ | Out-String))
  exit 92
}
`);
    const result = runPowerShell(driver, [
      "-ScriptPath", path.join(repo, ".github/workflows/scripts/build-sycl-cuda-win.ps1"),
    ]);
    assert.equal(result.status, 0, result.stdout + result.stderr);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows AdaptiveCpp builder rejects a reparse destination before replacement", {
  skip: !WINDOWS,
}, (/** @type {import("node:test").TestContext} */ t) => {
  const fixture = makeFixture();
  try {
    const target = path.join(fixture, "target");
    const destination = path.join(fixture, "acpp-junction");
    const workspace = path.join(fixture, "mom-dev-builder");
    const sentinel = path.join(target, "do-not-delete.txt");
    fs.mkdirSync(target);
    fs.mkdirSync(workspace);
    fs.writeFileSync(path.join(workspace, ".mom-dev-workspace"),
      "mom development workspace");
    fs.writeFileSync(sentinel, "sentinel");
    if (!tryMakeJunction(destination, target)) {
      t.skip("junction creation is unavailable");
      return;
    }
    const result = runPowerShell(
      path.join(repo, "scripts", "build-windows-adaptivecpp-base.ps1"),
      ["-InstallDir", destination, "-Workspace", workspace],
    );
    assert.notEqual(result.status, 0, result.stdout + result.stderr);
    assert.match(result.stdout + result.stderr, /must not use a reparse point/);
    assert.equal(fs.readFileSync(sentinel, "utf8"), "sentinel");

    const parentTarget = path.join(fixture, "parent-target");
    const parentLink = path.join(fixture, "parent-junction");
    const nestedDestination = path.join(parentLink, "acpp-nested");
    const parentSentinel = path.join(parentTarget, "do-not-delete.txt");
    fs.mkdirSync(parentTarget);
    fs.writeFileSync(parentSentinel, "sentinel");
    if (!tryMakeJunction(parentLink, parentTarget)) {
      t.skip("parent junction creation is unavailable");
      return;
    }
    const nested = runPowerShell(
      path.join(repo, "scripts", "build-windows-adaptivecpp-base.ps1"),
      ["-InstallDir", nestedDestination, "-Workspace", workspace],
    );
    assert.notEqual(nested.status, 0, nested.stdout + nested.stderr);
    assert.match(nested.stdout + nested.stderr, /must not use a reparse point/);
    assert.equal(fs.readFileSync(parentSentinel, "utf8"), "sentinel");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

/** @param {string} link @param {string} target @returns {boolean} */
function tryMakeJunction(link, target) {
  try {
    fs.symlinkSync(target, link, "junction");
    return true;
  } catch {
    return false;
  }
}

/** @param {string} link @param {string} target @returns {boolean} */
function tryMakeFileSymlink(link, target) {
  try {
    fs.symlinkSync(target, link, "file");
    return true;
  } catch {
    return false;
  }
}

const outputDriverSource = `
param([string]$ScriptPath, [string]$OutputDirectory, [string]$ExpectedText)
$ErrorActionPreference = 'Stop'
function node.exe {
  throw 'HEAVY_CALLED'
}
try {
  & $ScriptPath -SkipBuild -SkipCompilerGates -SkipGpuGates -OutputDirectory $OutputDirectory
  exit 41
} catch {
  if ($_.Exception.Message.Contains('HEAVY_CALLED')) { exit 42 }
  if ($_.Exception.Message.Contains($ExpectedText)) { exit 0 }
  $message = $_ | Out-String
  [Console]::Error.WriteLine($message)
  exit 42
}
`;

const ownedOutputDriverSource = `
param([string]$ScriptPath, [string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
function node.exe {
  throw 'PREBUILD_STOP'
}
try {
  & $ScriptPath -SkipBuild -SkipCompilerGates -SkipGpuGates -OutputDirectory $OutputDirectory
  exit 61
} catch {
  if ($_.Exception.Message.Contains('PREBUILD_STOP')) { exit 0 }
  $message = $_ | Out-String
  [Console]::Error.WriteLine($message)
  exit 62
}
`;

const cutlassDriverSource = `
param([string]$ScriptPath, [string]$Destination, [string]$ExpectedText)
$ErrorActionPreference = 'Stop'
. $ScriptPath
function Invoke-WebRequest {
  param([switch]$UseBasicParsing, [string]$Uri, [string]$OutFile)
  throw 'NETWORK_CALLED'
}
try {
  Install-MomCutlass -Destination $Destination
  exit 41
} catch {
  if ($_.Exception.Message.Contains('NETWORK_CALLED')) { exit 42 }
  if ($_.Exception.Message.Contains($ExpectedText)) { exit 0 }
  $message = $_ | Out-String
  [Console]::Error.WriteLine($message)
  exit 43
}
`;

const toolchainDriverSource = `
param(
  [string]$ScriptPath,
  [string]$Destination,
  [string]$ExpectedText,
  [string]$Asset = 'dpcpp-cuda-win.tar.gz',
  [string]$ExpectedSha256 = ('0' * 64)
)
$ErrorActionPreference = 'Stop'
function curl.exe {
  throw 'NETWORK_CALLED'
}
try {
  & $ScriptPath -Dest $Destination -Asset $Asset -ExpectedSha256 $ExpectedSha256
  exit 51
} catch {
  if ($_.Exception.Message.Contains('NETWORK_CALLED')) { exit 52 }
  if ($_.Exception.Message.Contains($ExpectedText)) { exit 0 }
  $message = $_ | Out-String
  [Console]::Error.WriteLine($message)
  exit 53
}
`;


const toolchainDefaultsDriverSource = `
param([string]$Restore, [string]$Install, [string]$Package)
$ErrorActionPreference = 'Stop'
function Get-StringDefault([string]$Path, [string]$Name) {
  $tokens = $null
  $errors = $null
  $ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $Path, [ref]$tokens, [ref]$errors)
  if ($errors.Count) { throw "SCRIPT_PARSE_FAILED: $Path" }
  $parameter = @($ast.ParamBlock.Parameters | Where-Object {
    $_.Name.VariablePath.UserPath -eq $Name
  })
  if ($parameter.Count -ne 1) { throw "PARAMETER_NOT_UNIQUE: $Name" }
  return $parameter[0].DefaultValue.SafeGetValue()
}
[ordered]@{
  asset = Get-StringDefault $Restore 'Asset'
  restoreSha256 = Get-StringDefault $Restore 'ExpectedSha256'
  installSha256 = Get-StringDefault $Install 'DpcppSha256'
  outFile = Get-StringDefault $Package 'OutFile'
} | ConvertTo-Json -Compress
`;

const toolchainRestoreDriverSource = `
param(
  [string]$ScriptPath,
  [string]$Destination,
  [string]$Archive,
  [switch]$ChangeMarker,
  [switch]$WrongDigest
)
$ErrorActionPreference = 'Stop'
function curl.exe {
  $outputIndex = [Array]::IndexOf([string[]]$args, '-o')
  if ($outputIndex -lt 0 -or $outputIndex + 1 -ge $args.Count) { throw 'OUTPUT_ARGUMENT_MISSING' }
  if ($args[-1] -ne 'https://github.com/MoneroOcean/mo-miner/releases/download/toolchain-win-dpcpp-cuda/dpcpp-cuda-win-v7.1.1.tar.gz') {
    throw 'UNEXPECTED_ASSET_URL'
  }
  Copy-Item -LiteralPath $Archive -Destination $args[$outputIndex + 1]
  if ($ChangeMarker) {
    $marker = Join-Path $Destination '.mom-toolchain-sha256'
    Set-Content -LiteralPath $marker -Value ('f' * 64) -NoNewline
  }
  $global:LASTEXITCODE = 0
}
# Only the download/hash boundary is mocked; real staging validation and replacement run unchanged.
function Get-FileHash {
  param([string]$Path, [string]$Algorithm)
  if ([IO.Path]::GetFileName($Path) -ne 'dpcpp-cuda-win-v7.1.1.tar.gz' -or $Algorithm -ne 'SHA256') {
    throw 'UNEXPECTED_HASH_REQUEST'
  }
  $hash = if ($WrongDigest) { 'f' * 64 } else { '8e5e9de06ed46c5e28aac4c574811a68f0a85dbb1cb65508187b0b8cf2e573cb' }
  [pscustomobject]@{ Hash = $hash }
}
& $ScriptPath -Dest $Destination
`;

const packageBoundaryDriverSource = `
param([string]$ScriptPath, [string]$RepoRoot, [string]$Archive)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $RepoRoot
$env:NODE_BIN = Join-Path $RepoRoot 'node.exe'
function Remove-Item {
  [CmdletBinding()]
  param(
    [Parameter(Position = 0, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
    [object[]]$Path,
    [object[]]$LiteralPath,
    [switch]$Recurse,
    [switch]$Force
  )
  throw 'FIRST_CLEANUP'
}
function npx.cmd {
  throw 'TOOL_CALLED'
}
try {
  & $ScriptPath -Version '1.2.3' -Archive $Archive
  exit 71
} catch {
  if ($_.Exception.Message.Contains('FIRST_CLEANUP')) { exit 0 }
  if ($_.Exception.Message.Contains('TOOL_CALLED')) { exit 72 }
  [Console]::Error.WriteLine(($_ | Out-String))
  exit 73
}
`;

const packageFailureDriverSource = `
param([string]$ScriptPath, [string]$RepoRoot, [string]$Archive)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $RepoRoot
$env:NODE_BIN = Join-Path $RepoRoot 'node.exe'
function npx.cmd {
  throw 'NPM_FAILURE'
}
try {
  & $ScriptPath -Version '1.2.3' -Archive $Archive
  exit 71
} catch {
  if ($_.Exception.Message.Contains('NPM_FAILURE')) {
    [Console]::WriteLine('NPM_FAILURE')
    exit 0
  }
  [Console]::Error.WriteLine(($_ | Out-String))
  exit 72
}
`;

const windowsPackageWorkers = ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"];

/** @param {string} file @param {boolean} valid */
function writeWindowsPackageArtifact(file, valid = true) {
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, valid ? Buffer.from("MZfixture") : Buffer.from("not-pe"));
}

/** @returns {{fixture: string, script: string, archive: string, marker: string, driver: string}} */
function makeWindowsPackageFixture() {
  const fixture = makeFixture();
  const scriptDirectory = path.join(fixture, ".github", "workflows", "scripts");
  fs.mkdirSync(scriptDirectory, {recursive: true});
  for (const file of ["package-windows.ps1", "windows-dll-deps.ps1"]) {
    fs.copyFileSync(path.join(repo, ".github", "workflows", "scripts", file),
      path.join(scriptDirectory, file));
  }
  fs.writeFileSync(path.join(fixture, "node.exe"), "fixture node");
  const marker = path.join(fixture, "release", "mom-v1.2.3", "previous-archive.marker");
  fs.mkdirSync(path.dirname(marker), {recursive: true});
  fs.writeFileSync(marker, "preserve previous archive");
  const archive = path.join(fixture, "mom-v1.2.3-win.zip");
  fs.writeFileSync(archive, "preserve previous archive");
  const driver = makeDriver(fixture, packageBoundaryDriverSource);
  return {fixture, script: path.join(scriptDirectory, "package-windows.ps1"), archive, marker, driver};
}

/** @param {string} fixture @param {boolean} valid */
function writeAllWindowsPackageWorkers(fixture, valid = true) {
  for (const worker of windowsPackageWorkers) {
    writeWindowsPackageArtifact(path.join(fixture, "build", "win", "compilers", worker,
      "mom.node"), valid);
    writeWindowsPackageArtifact(path.join(fixture, "build", "win", "compilers", worker,
      "sycl.dll"), valid);
  }
}

/** @param {{fixture: string, script: string, archive: string, marker: string, driver: string}} state */
function runWindowsPackageBoundary(state) {
  return runPowerShell(state.driver, [
    "-ScriptPath", state.script, "-RepoRoot", state.fixture, "-Archive", state.archive,
  ]);
}

/** @returns {string} */
function extractWindowsArchivePublicationBlock() {
  const source = fs.readFileSync(
    path.join(repo, ".github", "workflows", "scripts", "package-windows.ps1"), "utf8");
  const startMarker = "$archiveDirectory = ";
  const endMarker = "Write-Output $Archive";
  const start = source.indexOf(startMarker);
  assert.notEqual(start, -1, "Windows archive publication block is missing");
  const end = source.indexOf(endMarker, start);
  assert.notEqual(end, -1, "Windows archive publication output is missing");
  return source.slice(start, end + endMarker.length);
}

/** @param {string} block @returns {string} */
function makeArchivePublicationDriverSource(block) {
  return [
    "param([string]$Root, [string]$Archive, [ValidateSet('success', 'partial', 'locked')] [string]$Mode)",
    "$ErrorActionPreference = 'Stop'",
    "Set-Location -LiteralPath $Root",
    "$archiveFullPath = [IO.Path]::GetFullPath($Archive)",
    "$packageDir = 'package'",
    "if ($Mode -eq 'partial') {",
    "  function Compress-Archive {",
    "    param([string]$Path, [string]$DestinationPath)",
    "    Set-Content -LiteralPath $DestinationPath -Value 'partial archive'",
    "    throw 'COMPRESS_FAILURE'",
    "  }",
    "}",
    "$script:compressionSucceeded = $false",
    "if ($Mode -eq 'locked') {",
    "  function Compress-Archive {",
    "    param([string]$Path, [string]$DestinationPath)",
    "    Microsoft.PowerShell.Archive\\Compress-Archive -Path $Path -DestinationPath $DestinationPath",
    "    $script:compressionSucceeded = $true",
    "  }",
    "}",
    "$lock = $null",
    "try {",
    "  if ($Mode -eq 'locked') {",
    "    $lock = [IO.File]::Open($archiveFullPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)",
    "  }",
    "  try {",
    block,
    "    if ($Mode -ne 'success') { exit 71 }",
    "    Add-Type -AssemblyName System.IO.Compression.FileSystem",
    "    $zip = [IO.Compression.ZipFile]::OpenRead($archiveFullPath)",
    "    try {",
    "      foreach ($entry in $zip.Entries) {",
    "        $entryName = $entry.FullName.Replace('\\', '/')",
    "        [Console]::WriteLine(('ENTRY:' + $entryName))",
    "        if ($entryName -eq 'package/mom.node') {",
    "          $stream = $entry.Open()",
    "          $reader = New-Object -TypeName System.IO.StreamReader -ArgumentList $stream",
    "          try {",
    "            [Console]::WriteLine(('CONTENT:' + $reader.ReadToEnd()))",
    "          } finally {",
    "            $reader.Dispose()",
    "            $stream.Dispose()",
    "          }",
    "        }",
    "      }",
    "    } finally {",
    "      $zip.Dispose()",
    "    }",
    "    exit 0",
    "  } catch {",
    "    if ($Mode -eq 'partial' -and $_.Exception.Message.Contains('COMPRESS_FAILURE')) {",
    "      [Console]::WriteLine('COMPRESS_FAILURE')",
    "      exit 0",
    "    }",
    "    if ($Mode -eq 'locked') {",
    "      $exception = $_.Exception.InnerException",
    "      if ($script:compressionSucceeded -and $exception -is [IO.IOException] -and",
    "          (($exception.HResult -band 0xffff) -eq 32)) {",
    "        [Console]::WriteLine('REPLACE_FAILURE')",
    "        exit 0",
    "      }",
    "      [Console]::Error.WriteLine(('UNEXPECTED_REPLACE_FAILURE: {0}' -f $_.Exception.ToString()))",
    "      exit 72",
    "    }",
    "    [Console]::Error.WriteLine(($_ | Out-String))",
    "    exit 72",
    "  }",
    "} finally {",
    "  if ($lock) { $lock.Dispose() }",
    "}",
  ].join("\n") + "\n";
}

/** @param {boolean} existing @returns {{fixture: string, archive: string, driver: string}} */
function makeArchivePublicationFixture(existing) {
  const fixture = makeFixture();
  const packageDir = path.join(fixture, "package");
  fs.mkdirSync(path.join(packageDir, "nested"), {recursive: true});
  fs.writeFileSync(path.join(packageDir, "mom.node"), "node-payload");
  fs.writeFileSync(path.join(packageDir, "nested", "manifest.txt"), "manifest-payload");
  const archive = path.join(fixture, "published.zip");
  if (existing) {
    fs.writeFileSync(archive, "old-archive");
  }
  const driver = makeDriver(fixture,
    makeArchivePublicationDriverSource(extractWindowsArchivePublicationBlock()));
  return {fixture, archive, driver};
}

/** @param {string} archive @returns {string[]} */
function archivePublicationTemps(archive) {
  return fs.readdirSync(path.dirname(archive)).filter(file =>
    /^\.mom-archive-[0-9a-f]{32}\.zip$/i.test(file));
}

/** @param {{fixture: string, archive: string, driver: string}} state @param {string} mode */
function runArchivePublication(state, mode) {
  return runPowerShell(state.driver, [
    "-Root", state.fixture, "-Archive", state.archive, "-Mode", mode,
  ]);
}

test("Windows package rejects flat-only, incomplete, and non-PE worker trees before cleanup", {
  skip: !WINDOWS,
}, () => {
  /** @type {Array<{name: string, setup: (fixture: string) => void, error: RegExp}>} */
  const cases = [
    {
      name: "flat-only",
      setup: fixture => {
        writeWindowsPackageArtifact(path.join(fixture, "build", "win", "Release", "mom.node"));
        writeWindowsPackageArtifact(path.join(fixture, "build", "win", "Release", "sycl.dll"));
      },
      error: /build[\\/]win[\\/]compilers[\\/]oneapi[\\/]mom\.node is missing/,
    },
    {
      name: "incomplete",
      setup: fixture => {
        writeWindowsPackageArtifact(path.join(fixture, "build", "win", "compilers", "oneapi",
          "mom.node"));
        writeWindowsPackageArtifact(path.join(fixture, "build", "win", "compilers", "oneapi",
          "sycl.dll"));
      },
      error: /build[\\/]win[\\/]compilers[\\/]dpcpp[\\/]mom\.node is missing/,
    },
    {
      name: "non-PE",
      setup: fixture => {
        writeWindowsPackageArtifact(path.join(fixture, "build", "win", "compilers", "oneapi",
          "mom.node"), false);
        writeWindowsPackageArtifact(path.join(fixture, "build", "win", "compilers", "oneapi",
          "sycl.dll"));
      },
      error: /not a Windows PE binary/,
    },
  ];
  for (const entry of cases) {
    const state = makeWindowsPackageFixture();
    try {
      entry.setup(state.fixture);
      const result = runWindowsPackageBoundary(state);
      const output = result.stdout + result.stderr;
      assert.notEqual(result.status, 0,
        `${entry.name}: package unexpectedly reached cleanup/tooling\n${output}`);
      assert.doesNotMatch(output, /FIRST_CLEANUP|TOOL_CALLED/,
        `${entry.name}: validation happened after cleanup/tooling\n${output}`);
      assert.match(output, entry.error, `${entry.name}: wrong validation failure\n${output}`);
      assert.equal(fs.readFileSync(state.marker, "utf8"), "preserve previous archive");
      assert.equal(fs.readFileSync(state.archive, "utf8"), "preserve previous archive");
    } finally {
      fs.rmSync(state.fixture, {recursive: true, force: true});
    }
  }
});

test("Windows package validates all five PE workers before the first cleanup", {
  skip: !WINDOWS,
}, () => {
  const state = makeWindowsPackageFixture();
  try {
    writeAllWindowsPackageWorkers(state.fixture);
    const result = runWindowsPackageBoundary(state);
    const output = result.stdout + result.stderr;
    assert.equal(result.status, 0,
      `complete worker fixture did not reach the cleanup boundary: ${result.status}\n${output}`);
    assert.doesNotMatch(output, /TOOL_CALLED/);
    assert.equal(fs.readFileSync(state.marker, "utf8"), "preserve previous archive");
    assert.equal(fs.readFileSync(state.archive, "utf8"), "preserve previous archive");
    assert.equal(fs.existsSync(path.join(state.fixture, "release-build")), false);
  } finally {
    fs.rmSync(state.fixture, {recursive: true, force: true});
  }
});

test("Windows package preserves an existing archive when bundling fails", {
  skip: !WINDOWS,
}, () => {
  const state = makeWindowsPackageFixture();
  try {
    writeAllWindowsPackageWorkers(state.fixture);
    state.driver = makeDriver(state.fixture, packageFailureDriverSource);
    const result = runWindowsPackageBoundary(state);
    const output = result.stdout + result.stderr;
    assert.equal(result.status, 0,
      `bundling failure fixture did not stop at the stubbed tool: ${result.status}\n${output}`);
    assert.match(output, /NPM_FAILURE/);
    assert.equal(fs.readFileSync(state.archive, "utf8"), "preserve previous archive");
  } finally {
    fs.rmSync(state.fixture, {recursive: true, force: true});
  }
});

test("Windows package rejects an archive file reparse point before cleanup", {
  skip: !WINDOWS,
}, (/** @type {import("node:test").TestContext} */ t) => {
  const state = makeWindowsPackageFixture();
  try {
    writeAllWindowsPackageWorkers(state.fixture);
    const target = path.join(state.fixture, "archive-target.bin");
    fs.writeFileSync(target, "reparse-target");
    fs.rmSync(state.archive);
    if (!tryMakeFileSymlink(state.archive, target)) {
      t.skip("file symlink creation is unavailable");
      return;
    }
    const result = runWindowsPackageBoundary(state);
    const output = result.stdout + result.stderr;
    assert.notEqual(result.status, 0, output);
    assert.match(output, /reparse point/);
    assert.doesNotMatch(output, /FIRST_CLEANUP|TOOL_CALLED/);
    assert.equal(fs.readFileSync(target, "utf8"), "reparse-target");
    assert.equal(fs.lstatSync(state.archive).isSymbolicLink(), true);
  } finally {
    fs.rmSync(state.fixture, {recursive: true, force: true});
  }
});

test("Windows archive publication creates or replaces a complete ZIP", {
  skip: !WINDOWS,
}, () => {
  for (const existing of [false, true]) {
    const state = makeArchivePublicationFixture(existing);
    try {
      const result = runArchivePublication(state, "success");
      const output = result.stdout + result.stderr;
      assert.equal(result.status, 0,
        `archive success fixture (${existing ? "existing" : "absent"}) failed: ` +
        `${result.status}\n${output}`);
      assert.equal((output.match(/^ENTRY:package\/$/gm) ?? []).length, 1,
        `expected one explicit package-root directory entry\n${output}`);
      assert.match(output, /ENTRY:package\/mom\.node/);
      assert.match(output, /ENTRY:package\/nested\/manifest\.txt/);
      assert.match(output, /CONTENT:node-payload/);
      assert.deepEqual(archivePublicationTemps(state.archive), []);
    } finally {
      fs.rmSync(state.fixture, {recursive: true, force: true});
    }
  }
});

test("Windows archive publication preserves output after compression failure", {
  skip: !WINDOWS,
}, () => {
  for (const existing of [false, true]) {
    const state = makeArchivePublicationFixture(existing);
    try {
      const result = runArchivePublication(state, "partial");
      const output = result.stdout + result.stderr;
      assert.equal(result.status, 0,
        `partial archive fixture (${existing ? "existing" : "absent"}) failed: ` +
        `${result.status}\n${output}`);
      assert.match(output, /COMPRESS_FAILURE/);
      assert.equal(fs.existsSync(state.archive), existing);
      if (existing) {
        assert.equal(fs.readFileSync(state.archive, "utf8"), "old-archive");
      }
      assert.deepEqual(archivePublicationTemps(state.archive), []);
    } finally {
      fs.rmSync(state.fixture, {recursive: true, force: true});
    }
  }
});

test("Windows archive publication preserves output after replacement failure", {
  skip: !WINDOWS,
}, () => {
  const state = makeArchivePublicationFixture(true);
  try {
    const result = runArchivePublication(state, "locked");
    const output = result.stdout + result.stderr;
    assert.equal(result.status, 0,
      `locked archive fixture did not report replacement failure: ${result.status}\n${output}`);
    assert.match(output, /REPLACE_FAILURE/);
    assert.equal(fs.readFileSync(state.archive, "utf8"), "old-archive");
    assert.deepEqual(archivePublicationTemps(state.archive), []);
  } finally {
    fs.rmSync(state.fixture, {recursive: true, force: true});
  }
});

test("Windows unified release rejects unowned output directories before replacement", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeRepoFixture();
  try {
    const output = path.join(fixture, "results");
    const sentinel = path.join(output, "do-not-delete.txt");
    fs.mkdirSync(output);
    fs.writeFileSync(sentinel, "sentinel");
    const driver = makeDriver(fixture, outputDriverSource);
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, "scripts", "test-windows-unified-release.ps1"),
      "-OutputDirectory", output, "-ExpectedText", "unowned",
    ], "unowned output directory");
    assert.equal(fs.readFileSync(sentinel, "utf8"), "sentinel");

    const wrongMarkerOutput = path.join(fixture, "wrong-marker-results");
    const wrongMarkerSentinel = path.join(wrongMarkerOutput, "do-not-delete.txt");
    fs.mkdirSync(wrongMarkerOutput);
    fs.writeFileSync(path.join(wrongMarkerOutput, ".mom-windows-unified-results"), "wrong-marker");
    fs.writeFileSync(wrongMarkerSentinel, "sentinel");
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, "scripts", "test-windows-unified-release.ps1"),
      "-OutputDirectory", wrongMarkerOutput, "-ExpectedText", "ownership marker",
    ], "wrong output marker");
    assert.equal(fs.readFileSync(wrongMarkerSentinel, "utf8"), "sentinel");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows unified release accepts an owned output and reaches the pre-build boundary", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeRepoFixture();
  try {
    const output = path.join(fixture, "owned-results");
    const marker = path.join(output, ".mom-windows-unified-results");
    const stale = path.join(output, "stale.txt");
    fs.mkdirSync(output);
    fs.writeFileSync(marker, "mom-windows-unified-results");
    fs.writeFileSync(stale, "stale");
    const driver = makeDriver(fixture, ownedOutputDriverSource);
    const result = runPowerShell(driver, [
      "-ScriptPath", path.join(repo, "scripts", "test-windows-unified-release.ps1"),
      "-OutputDirectory", output,
    ]);
    assert.equal(result.status, 0,
      `owned output did not reach the pre-build boundary: ${result.status}\n` +
      `${result.stdout}\n${result.stderr}`);
    assert.equal(fs.readFileSync(marker, "utf8"), "mom-windows-unified-results");
    assert.equal(fs.existsSync(stale), false);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows unified release accepts an empty output and reaches the pre-build boundary", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeRepoFixture();
  try {
    const output = path.join(fixture, "empty-results");
    fs.mkdirSync(output);
    const driver = makeDriver(fixture, ownedOutputDriverSource);
    const result = runPowerShell(driver, [
      "-ScriptPath", path.join(repo, "scripts", "test-windows-unified-release.ps1"),
      "-OutputDirectory", output,
    ]);
    assert.equal(result.status, 0,
      `empty output did not reach the pre-build boundary: ${result.status}\n` +
      `${result.stdout}\n${result.stderr}`);
    assert.equal(fs.readFileSync(path.join(output, ".mom-windows-unified-results"), "utf8"),
      "mom-windows-unified-results");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows unified release rejects a file output before replacement", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeRepoFixture();
  try {
    const output = path.join(fixture, "results-file");
    fs.writeFileSync(output, "sentinel");
    const driver = makeDriver(fixture, outputDriverSource);
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, "scripts", "test-windows-unified-release.ps1"),
      "-OutputDirectory", output, "-ExpectedText", "not a directory",
    ], "file output");
    assert.equal(fs.readFileSync(output, "utf8"), "sentinel");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows unified release rejects a reparse output before replacement when junctions are available", {
  skip: !WINDOWS,
}, (/** @type {import("node:test").TestContext} */ t) => {
  const fixture = makeRepoFixture();
  try {
    const target = path.join(fixture, "target");
    const output = path.join(fixture, "results-junction");
    const sentinel = path.join(target, "do-not-delete.txt");
    fs.mkdirSync(target);
    fs.writeFileSync(sentinel, "sentinel");
    if (!tryMakeJunction(output, target)) {
      t.skip("junction creation is unavailable");
      return;
    }
    const driver = makeDriver(fixture, outputDriverSource);
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, "scripts", "test-windows-unified-release.ps1"),
      "-OutputDirectory", output, "-ExpectedText", "reparse point",
    ], "reparse output");
    assert.equal(fs.readFileSync(sentinel, "utf8"), "sentinel");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows unified release rejects a reparse ancestor before replacement when junctions are available", {
  skip: !WINDOWS,
}, (/** @type {import("node:test").TestContext} */ t) => {
  const fixture = makeRepoFixture();
  const target = makeFixture();
  try {
    const ancestor = path.join(fixture, "results-parent");
    const output = path.join(ancestor, "results");
    const sentinel = path.join(target, "do-not-delete.txt");
    fs.writeFileSync(sentinel, "sentinel");
    if (!tryMakeJunction(ancestor, target)) {
      t.skip("junction creation is unavailable");
      return;
    }
    const driver = makeDriver(fixture, outputDriverSource);
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, "scripts", "test-windows-unified-release.ps1"),
      "-OutputDirectory", output, "-ExpectedText", "reparse-point ancestor",
    ], "reparse ancestor");
    assert.equal(fs.readFileSync(sentinel, "utf8"), "sentinel");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
    fs.rmSync(target, {recursive: true, force: true});
  }
});

test("Windows CUTLASS installer rejects file destinations before download", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const driver = makeDriver(fixture, cutlassDriverSource);
    const file = path.join(fixture, "cutlass-file");
    fs.writeFileSync(file, "sentinel");
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, "scripts", "install-cutlass.ps1"),
      "-Destination", file, "-ExpectedText", "not a directory",
    ], "CUTLASS file");
    assert.equal(fs.readFileSync(file, "utf8"), "sentinel");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows CUTLASS installer rejects a reparse destination before download when junctions are available", {
  skip: !WINDOWS,
}, (/** @type {import("node:test").TestContext} */ t) => {
  const fixture = makeFixture();
  try {
    const target = path.join(fixture, "target");
    const destination = path.join(fixture, "cutlass-junction");
    const sentinel = path.join(target, "do-not-delete.txt");
    fs.mkdirSync(target);
    fs.writeFileSync(sentinel, "sentinel");
    if (!tryMakeJunction(destination, target)) {
      t.skip("junction creation is unavailable");
      return;
    }
    const driver = makeDriver(fixture, cutlassDriverSource);
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, "scripts", "install-cutlass.ps1"),
      "-Destination", destination, "-ExpectedText", "reparse point",
    ], "CUTLASS junction");
    assert.equal(fs.readFileSync(sentinel, "utf8"), "sentinel");

    const parentTarget = path.join(fixture, "parent-target");
    const parentLink = path.join(fixture, "parent-junction");
    const nestedDestination = path.join(parentLink, "cutlass");
    const parentSentinel = path.join(parentTarget, "do-not-delete.txt");
    fs.mkdirSync(parentTarget);
    fs.writeFileSync(parentSentinel, "sentinel");
    if (!tryMakeJunction(parentLink, parentTarget)) {
      t.skip("parent junction creation is unavailable");
      return;
    }
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, "scripts", "install-cutlass.ps1"),
      "-Destination", nestedDestination, "-ExpectedText", "reparse point",
    ], "CUTLASS parent junction");
    assert.equal(fs.readFileSync(parentSentinel, "utf8"), "sentinel");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows CUTLASS installer rejects a .mom-version file symlink before download when file symlinks are available", {
  skip: !WINDOWS,
}, (/** @type {import("node:test").TestContext} */ t) => {
  const fixture = makeFixture();
  try {
    const destination = path.join(fixture, "cutlass");
    const marker = path.join(destination, ".mom-version");
    const sentinel = path.join(fixture, "external-sentinel");
    fs.mkdirSync(destination);
    fs.writeFileSync(sentinel, "sentinel");
    if (!tryMakeFileSymlink(marker, sentinel)) {
      t.skip("file symlink creation is unavailable");
      return;
    }
    const driver = makeDriver(fixture, cutlassDriverSource);
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, "scripts", "install-cutlass.ps1"),
      "-Destination", destination, "-ExpectedText", "reparse point",
    ], "CUTLASS version marker symlink");
    assert.equal(fs.readFileSync(sentinel, "utf8"), "sentinel");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows toolchain restore rejects unowned or file destinations before download", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const driver = makeDriver(fixture, toolchainDriverSource);
    const directory = path.join(fixture, "toolchain-directory");
    const directorySentinel = path.join(directory, "do-not-delete.txt");
    fs.mkdirSync(directory);
    fs.writeFileSync(directorySentinel, "sentinel");
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, ".github", "workflows", "scripts", "restore-toolchain-win.ps1"),
      "-Destination", directory, "-ExpectedText", "unowned nonempty",
    ], "toolchain directory");
    assert.equal(fs.readFileSync(directorySentinel, "utf8"), "sentinel");

    const wrongMarker = path.join(fixture, "toolchain-wrong-marker");
    const wrongMarkerSentinel = path.join(wrongMarker, "do-not-delete.txt");
    fs.mkdirSync(wrongMarker);
    fs.writeFileSync(path.join(wrongMarker, ".mom-toolchain-sha256"), "wrong");
    fs.writeFileSync(wrongMarkerSentinel, "sentinel");
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, ".github", "workflows", "scripts", "restore-toolchain-win.ps1"),
      "-Destination", wrongMarker, "-ExpectedText", "unowned nonempty",
    ], "toolchain wrong marker");
    assert.equal(fs.readFileSync(wrongMarkerSentinel, "utf8"), "sentinel");

    const file = path.join(fixture, "toolchain-file");
    fs.writeFileSync(file, "sentinel");
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, ".github", "workflows", "scripts", "restore-toolchain-win.ps1"),
      "-Destination", file, "-ExpectedText", "not a directory",
    ], "toolchain file");
    assert.equal(fs.readFileSync(file, "utf8"), "sentinel");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows toolchain restore rejects a reparse destination or filesystem root before download", {
  skip: !WINDOWS,
}, (/** @type {import("node:test").TestContext} */ t) => {
  const fixture = makeFixture();
  try {
    const driver = makeDriver(fixture, toolchainDriverSource);
    const target = path.join(fixture, "target");
    const destination = path.join(fixture, "toolchain-junction");
    const sentinel = path.join(target, "do-not-delete.txt");
    fs.mkdirSync(target);
    fs.writeFileSync(sentinel, "sentinel");
    if (tryMakeJunction(destination, target)) {
      assertRejected(driver, [
        "-ScriptPath", path.join(repo, ".github", "workflows", "scripts", "restore-toolchain-win.ps1"),
        "-Destination", destination, "-ExpectedText", "reparse point",
      ], "toolchain junction");
      assert.equal(fs.readFileSync(sentinel, "utf8"), "sentinel");
    } else {
      t.diagnostic("junction creation is unavailable; root rejection still exercised");
    }
    const parentTarget = path.join(fixture, "parent-target");
    const parentLink = path.join(fixture, "parent-junction");
    const parentSentinel = path.join(parentTarget, "do-not-delete.txt");
    fs.mkdirSync(parentTarget);
    fs.writeFileSync(parentSentinel, "sentinel");
    if (tryMakeJunction(parentLink, parentTarget)) {
      assertRejected(driver, [
        "-ScriptPath", path.join(repo, ".github", "workflows", "scripts", "restore-toolchain-win.ps1"),
        "-Destination", path.join(parentLink, "toolchain"), "-ExpectedText", "reparse point",
      ], "toolchain parent junction");
      assert.equal(fs.readFileSync(parentSentinel, "utf8"), "sentinel");
    } else {
      t.diagnostic("parent junction creation is unavailable");
    }
    const systemDrive = process.env["SystemDrive"];
    assert.ok(systemDrive);
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, ".github", "workflows", "scripts", "restore-toolchain-win.ps1"),
      "-Destination", `${systemDrive}\\`, "-ExpectedText", "filesystem root",
    ], "toolchain root");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows toolchain restore rejects unsafe asset names and digests before download", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const driver = makeDriver(fixture, toolchainDriverSource);
    const destination = path.join(fixture, "toolchain");
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, ".github", "workflows", "scripts", "restore-toolchain-win.ps1"),
      "-Destination", destination, "-ExpectedText", "leaf filename",
      "-Asset", "..\\outside.tar.gz",
    ], "toolchain asset path");
    assertRejected(driver, [
      "-ScriptPath", path.join(repo, ".github", "workflows", "scripts", "restore-toolchain-win.ps1"),
      "-Destination", destination, "-ExpectedText", "64 hexadecimal",
      "-ExpectedSha256", "not-a-digest",
    ], "toolchain digest");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows DPCPP defaults select the stable-source SDK asset", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const result = runPowerShell(makeDriver(fixture, toolchainDefaultsDriverSource), [
      "-Restore", path.join(repo, ".github/workflows/scripts/restore-toolchain-win.ps1"),
      "-Install", path.join(repo, "scripts/install-dev.ps1"),
      "-Package", path.join(repo, ".github/workflows/scripts/package-toolchain-win.ps1"),
    ]);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.deepEqual(JSON.parse(result.stdout), {
      asset: DPCPP_ASSET,
      restoreSha256: DPCPP_SHA256,
      installSha256: DPCPP_SHA256,
      outFile: "C:\\mom\\" + DPCPP_ASSET,
    });
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows toolchain restore migrates only the known owned predecessors", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  try {
    const script = path.join(repo, ".github/workflows/scripts/restore-toolchain-win.ps1");
    const payload = path.join(fixture, "payload");
    const archive = path.join(fixture, "fixture-toolchain.tar.gz");
    for (const file of [
      "clang++.exe", "clang.exe", "lld-link.exe", "sycl-post-link.exe", "llvm-spirv.exe",
      "llvm-link.exe", "clang-linker-wrapper.exe", "clang-offload-wrapper.exe", "sycl9.dll",
      "sycl-jit.dll", "ur_win_proxy_loader.dll", "ur_loader.dll", "ur_adapter_cuda.dll",
      "ur_adapter_opencl.dll", "ur_adapter_level_zero_v2.dll",
    ]) {
      fs.mkdirSync(path.join(payload, "bin"), {recursive: true});
      fs.writeFileSync(path.join(payload, "bin", file), "fixture");
    }
    const pack = makeDriver(fixture, String.raw`
param([string]$Payload, [string]$Archive)
$ErrorActionPreference = 'Stop'
& "$env:SystemRoot\system32\tar.exe" -C $Payload -czf $Archive bin
if ($LASTEXITCODE -ne 0) { throw "FIXTURE_TAR_FAILED: $LASTEXITCODE" }
`);
    const packed = runPowerShell(pack, ["-Payload", payload, "-Archive", archive]);
    assert.equal(packed.status, 0, packed.stdout + packed.stderr);
    const restoreDriver = makeDriver(
      fs.mkdtempSync(path.join(fixture, "restore-driver-")), toolchainRestoreDriverSource);
    /** @param {string} name @param {string} marker */
    const destination = (name, marker) => {
      const directory = path.join(fixture, name);
      fs.mkdirSync(directory);
      fs.writeFileSync(path.join(directory, ".mom-toolchain-sha256"), marker);
      fs.writeFileSync(path.join(directory, "old-sentinel"), "preserve until verified");
      return directory;
    };
    for (const entry of [
      {name: "current", marker: DPCPP_SHA256},
      {name: "predecessor", marker: PRIOR_DPCPP_SHA256},
      {name: "matched-predecessor", marker: MATCHED_DPCPP_SHA256},
    ]) {
      const directory = destination(entry.name, entry.marker);
      const result = runPowerShell(restoreDriver, [
        "-ScriptPath", script, "-Destination", directory, "-Archive", archive,
      ]);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.equal(fs.readFileSync(path.join(directory, ".mom-toolchain-sha256"), "utf8"),
        DPCPP_SHA256);
      assert.equal(fs.readFileSync(path.join(directory, "bin", "ur_loader.dll"), "utf8"), "fixture");
      assert.equal(fs.existsSync(path.join(directory, "old-sentinel")), false);
    }
    const rejectDriver = makeDriver(
      fs.mkdtempSync(path.join(fixture, "reject-driver-")), toolchainDriverSource);
    for (const entry of [
      {name: "unrelated", marker: "f".repeat(64), expected: DPCPP_SHA256},
      {name: "malformed", marker: PRIOR_DPCPP_SHA256 + "\n", expected: DPCPP_SHA256},
      {name: "uppercase", marker: PRIOR_DPCPP_SHA256.toUpperCase(), expected: DPCPP_SHA256},
      {name: "custom-target", marker: PRIOR_DPCPP_SHA256, expected: "0".repeat(64)},
      {name: "custom-matched-target", marker: MATCHED_DPCPP_SHA256, expected: "0".repeat(64)},
    ]) {
      const directory = destination(entry.name, entry.marker);
      assertRejected(rejectDriver, [
        "-ScriptPath", script, "-Destination", directory, "-ExpectedText", "unowned nonempty",
        "-Asset", DPCPP_ASSET, "-ExpectedSha256", entry.expected,
      ], entry.name);
      assert.equal(fs.readFileSync(path.join(directory, "old-sentinel"), "utf8"),
        "preserve until verified");
    }
    for (const entry of [
      {name: "changed-owner", flag: "-ChangeMarker", error: /unowned nonempty/},
      {name: "wrong-download", flag: "-WrongDigest", error: /SHA256 mismatch/},
    ]) {
      const directory = destination(entry.name, PRIOR_DPCPP_SHA256);
      const result = runPowerShell(restoreDriver, [
        "-ScriptPath", script, "-Destination", directory, "-Archive", archive, entry.flag,
      ]);
      assert.notEqual(result.status, 0, result.stdout + result.stderr);
      assert.match(result.stdout + result.stderr, entry.error);
      assert.equal(fs.readFileSync(path.join(directory, "old-sentinel"), "utf8"),
        "preserve until verified");
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows toolchain restore rejects a reparse ownership marker", {
  skip: !WINDOWS,
}, (/** @type {import("node:test").TestContext} */ t) => {
  const fixture = makeFixture();
  try {
    const directory = path.join(fixture, "toolchain");
    const external = path.join(fixture, "external-marker");
    fs.mkdirSync(directory);
    fs.writeFileSync(external, PRIOR_DPCPP_SHA256);
    fs.writeFileSync(path.join(directory, "old-sentinel"), "preserve");
    if (!tryMakeFileSymlink(path.join(directory, ".mom-toolchain-sha256"), external)) {
      t.skip("file symlink creation is unavailable");
      return;
    }
    assertRejected(makeDriver(fixture, toolchainDriverSource), [
      "-ScriptPath", path.join(repo, ".github/workflows/scripts/restore-toolchain-win.ps1"),
      "-Destination", directory, "-ExpectedText", "unowned nonempty",
      "-Asset", DPCPP_ASSET, "-ExpectedSha256", DPCPP_SHA256,
    ], "toolchain marker symlink");
    assert.equal(fs.readFileSync(external, "utf8"), PRIOR_DPCPP_SHA256);
    assert.equal(fs.readFileSync(path.join(directory, "old-sentinel"), "utf8"), "preserve");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});
