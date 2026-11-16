"use strict";

const {describe, test} = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const local = require("../scripts/build-local-native-sdk"), {fileHash} = require("../scripts/build-source-inputs"),
  command = require("../scripts/build-local-native-windows-command");
const repo = fs.realpathSync(path.resolve(__dirname, "..")), id = "12345678-1234-4234-9234-123456789abc",
  windowsRoot = "C:\\Users\\Fixture\\AppData\\Local\\MoM\\native-wsl-windows\\" + "a".repeat(64),
  supportWorkspace = windowsRoot + "\\support", exportWorkspace = windowsRoot + "\\exports", bridgeRoot = windowsRoot + "\\bridge",
  bridgeRunWindows = bridgeRoot + "\\run-" + id,
  oneApiRoot = "C:\\Program Files (x86)\\Intel\\oneAPI", nativePerlRoot = "C:\\Tools\\mom\\perl-5.42.3.1",
  relayEnvironment = "ONEAPI_ROOT/w:MOM_NATIVE_PERL_ROOT/w";

/** Stable Windows paths stay separate from temporary Linux fixture paths. @param {string} base */
function fixture(base) {
  const workspace = path.join(base, "native-sdk");
  local.admitWorkspace(workspace, repo);
  const paths = local.supportPaths(workspace), sourceRoot = paths.hermesSource;
  fs.mkdirSync(path.join(sourceRoot, "roots/hermes"), {recursive: true});
  fs.writeFileSync(path.join(sourceRoot, "fetch.json"), "{}\n");
  fs.writeFileSync(path.join(sourceRoot, "roots.json"), JSON.stringify({hermes: "/tmp/hermes-fetch/roots/hermes"}) + "\n");
  fs.writeFileSync(path.join(sourceRoot, "roots/hermes/config"), "[core]\n  bare = true\n");
  const bridgeRunPosix = path.join(base, "native-bridge", "run-" + id), archivePosix = path.join(bridgeRunPosix, "hermes-sources.tar"),
    inputPosix = path.join(bridgeRunPosix, "support-input.json"), outputPosix = path.join(bridgeRunPosix, "support-output.json"),
    coreRoot = path.join(base, "core"), coreArchive = path.join(coreRoot, "cpu-core.lib"), coreReceipt = path.join(coreRoot, "inputs.json"),
    ioRoot = path.join(base, "windows-support/io"), hermesRoot = path.join(base, "windows-support/hermes"),
    coreArchiveWindows = supportWorkspace + "\\cpu-core\\cpu-core.lib", coreReceiptWindows = supportWorkspace + "\\cpu-core\\inputs.json",
    ioRootWindows = exportWorkspace + "\\io", hermesRootWindows = exportWorkspace + "\\hermes",
    checkoutWindows = "C:\\Users\\Fixture\\source\\mo-miner", powershellWindows = "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
    tools = {cc: "/fixture/icx", cmake: "/fixture/cmake", cxx: "/fixture/icpx", git: "/fixture/git", ninja: "/fixture/ninja", systemdRun: "/fixture/systemd-run"};
  fs.mkdirSync(bridgeRunPosix, {recursive: true}); fs.mkdirSync(coreRoot); fs.mkdirSync(ioRoot, {recursive: true}); fs.mkdirSync(hermesRoot, {recursive: true});
  fs.writeFileSync(coreArchive, "core fixture\n"); fs.writeFileSync(coreReceipt, "original windows core receipt\n");
  const windowsToPosix = new Map([
      [bridgeRunWindows, bridgeRunPosix], [bridgeRunWindows + "\\hermes-sources.tar", archivePosix],
      [bridgeRunWindows + "\\support-input.json", inputPosix], [bridgeRunWindows + "\\support-output.json", outputPosix],
      [powershellWindows, "/fixture/powershell.exe"], [coreArchiveWindows, coreArchive], [coreReceiptWindows, coreReceipt],
      [ioRootWindows, ioRoot], [hermesRootWindows, hermesRoot],
    ]), posixToWindows = new Map([
      [repo, checkoutWindows], [archivePosix, bridgeRunWindows + "\\hermes-sources.tar"],
      [path.join(repo, "scripts/build-local-native-windows-support.ps1"), "C:\\Users\\Fixture\\source\\mo-miner\\scripts\\build-local-native-windows-support.ps1"],
    ]);
  const controllerSupport = {localBuildId: id,
    io: {root: ioRootWindows, handoffSha256: "1".repeat(64), admitted: {kind: "io", localBuildId: id}},
    hermes: {root: hermesRootWindows, handoffSha256: "2".repeat(64), admitted: {kind: "hermes", localBuildId: id}}};
  const bridgeResult = {controllerSupport, coreArchive: coreArchiveWindows, coreReceipt: {path: coreReceiptWindows, sha256: fileHash(coreReceipt)},
    supportRetirementCandidates: {core: [], hermes: [], io: [], openssl: []}, exportRetirementCandidates: []};
  const calls = /** @type {Array<{name:string,args:any[]}>} */ ([]), outputRoot = path.join(workspace, "windows-sdk");
  return {workspace, paths, sourceRoot, bridgeRunPosix, archivePosix, inputPosix, outputPosix, coreArchive, coreReceipt,
    coreArchiveWindows, supportWorkspace, exportWorkspace, bridgeRunWindows, checkoutWindows, powershellWindows, windowsToPosix,
    posixToWindows, controllerSupport, bridgeResult, tools, calls, outputRoot};
}

/** @param {import("node:test").TestContext} t @param {{failSdk?:boolean}} [options] */
function workerFixture(t, options = {}) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-windows-command-")), f = fixture(base), previousEsbuild = process.env["ESBUILD_BINARY_PATH"];
  t.after(() => {fs.rmSync(base, {recursive: true, force: true}); if (previousEsbuild === undefined) {delete process.env["ESBUILD_BINARY_PATH"];} else {process.env["ESBUILD_BINARY_PATH"] = previousEsbuild;}});
  const env = {MOM_NATIVE_BUILD_LOCK: "1", MOM_RELEASE_VERSION: "1.2.3", ONEAPI_ROOT: oneApiRoot,
      MOM_NATIVE_PERL_ROOT: nativePerlRoot}, node = {root: "/fixture/wsl-node", binary: "/fixture/esbuild",
      sha256: "3".repeat(64), version: "0.28.1", reused: false}, linuxSdk = {workspace: f.workspace,
      compilerFetch: f.paths.compilerFetch, compilerSources: f.paths.compilerSources, buildInputs: f.paths.buildInputs,
      llvmSdk: f.paths.llvmSdk, jobs: 2}, prerequisites = {workspace: f.workspace, paths: {hermesSource: f.sourceRoot}, source: linuxSdk, tools: f.tools};
  const dependencies = /** @type {any} */ ({
    platform: "linux", arch: "x64", observedNodeVersion: "v24.15.0", environment: env,
    verifyScope() {f.calls.push({name: "scope", args: []});}, verifyExt4() {f.calls.push({name: "ext4", args: []});},
    releaseVersion() {return "1.2.3";}, local: {...local, defaultWorkspace: () => f.workspace},
    localTools() {f.calls.push({name: "tools", args: []}); return f.tools;},
    preflight(/** @type {any} */ input) {f.calls.push({name: "preflight", args: [input.tools.cc, input.environment["ESBUILD_BINARY_PATH"] || ""]}); return {tools: f.tools};},
    prepareNode(/** @type {string} */ root) {f.calls.push({name: "node", args: [root]}); return node;},
    buildPrerequisites(/** @type {any} */ input, /** @type {any} */ optionsArg) {
      f.calls.push({name: "prerequisites", args: [input.workspace, input.jobs, optionsArg.environment["ESBUILD_BINARY_PATH"]]}); return prerequisites;
    },
    fileHash,
    packState(/** @type {string} */ archive) {fs.writeFileSync(archive, "tar fixture\n", {flag: "wx"}); f.calls.push({name: "pack", args: [archive]});},
    listState() {return ["hermes-fetch/fetch.json", "hermes-fetch/roots.json", "hermes-fetch/roots/hermes/config"];},
    windowsPathToWsl(/** @type {string} */ value) {const mapped = f.windowsToPosix.get(value); if (!mapped) {throw new Error("unexpected Windows fixture path " + value);} return mapped;},
    wslPathToWindows(/** @type {string} */ value) {const mapped = f.posixToWindows.get(value); if (!mapped) {throw new Error("unexpected WSL fixture path " + value);} return mapped;},
    runWindowsBridge(/** @type {any} */ input) {f.calls.push({name: "bridge", args: [input.value.hermesArchive.rootName, input.value.workspace]}); return f.bridgeResult;},
    async buildWindowsSdk(/** @type {any} */ input) {
      assert.equal(input.controllerSupport.io.root, f.windowsToPosix.get(f.controllerSupport.io.root));
      assert.equal(input.controllerSupport.hermes.root, f.windowsToPosix.get(f.controllerSupport.hermes.root));
      f.calls.push({name: "windows-sdk", args: [input.workspace, input.tools.cc, input.controllerSupport.localBuildId, input.coreArchive]});
      if (options.failSdk) {throw new Error("fixture SDK failure");}
      fs.mkdirSync(input.output, {recursive: true}); fs.writeFileSync(path.join(input.output, "native-windows-sdk.json"), "{}\n");
      return {workspace: f.workspace, configFile: path.join(input.output, "native-windows-sdk.json"), tools: f.tools};
    },
    buildDistribution(/** @type {any} */ sdk, /** @type {string} */ candidate, /** @type {any} */ envArg) {
      f.calls.push({name: "distribution", args: [sdk.configFile, candidate, envArg["MOM_BUILD_JOBS"]]}); return {candidate};
    },
    publish(/** @type {any} */ result, /** @type {string} */ destination) {
      f.calls.push({name: "publish", args: [result.candidate, destination]}); return {archive: "C:\\release\\mom-v1.2.3-win.zip", sha256: "4".repeat(64)};
    },
    console: {error(/** @type {string} */ message) {f.calls.push({name: "error", args: [message]});},
      log(/** @type {string} */ message) {f.calls.push({name: "log", args: [message]});}},
  });
  return {base, f, dependencies};
}

describe("Local Windows release coordinator", () => {
  test("native and WSL Node pin matches the maintained installer default", () => {
    const installer = fs.readFileSync(path.join(repo, "scripts/install-dev.sh"), "utf8"),
      version = /^NODE_VERSION=\$\{MOM_NODE_VERSION-([^}]+)\}$/m.exec(installer)?.[1];
    assert.equal(command.nodeVersion, "v" + version);
  });

  test("native preflight derives relay variables before composing its PowerShell hashtable", () => {
    const body = command.windowsPreflightSource(),
      payload = body.indexOf("$payload = [ordered]@{");
    assert.ok(payload > 0);
    for (const variable of ["compilerBin", "compilerVersion", "compilerRoot", "oneApiRoot", "perlBin", "perlHome", "nativePerlRoot"]) {
      const assignment = body.indexOf("$" + variable + " =");
      assert.ok(assignment >= 0 && assignment < payload, variable + " must be assigned before the payload hashtable");
    }
    assert.ok(body.indexOf("[void](Get-MomNativeBuildNode)") < payload);
  });

  test("Windows PowerShell parses the exact preflight and shared support scripts", {skip: process.platform !== "win32"}, () => {
    const parse = "$tokens=$null; $errors=$null; [void][System.Management.Automation.Language.Parser]::ParseInput([Console]::In.ReadToEnd(),[ref]$tokens,[ref]$errors); if ($errors.Count) { $errors | ForEach-Object { $_.ToString() }; exit 1 }",
      powershell = path.join(process.env["SystemRoot"] || "C:\\Windows", "System32/WindowsPowerShell/v1.0/powershell.exe");
    for (const source of [command.windowsPreflightSource(), ...["native-windows-tools.ps1", "build-local-native-windows-support.ps1"]
      .map(name => fs.readFileSync(path.join(repo, "scripts", name), "utf8"))]) {
      const result = require("node:child_process").spawnSync(powershell, ["-NoProfile", "-NonInteractive", "-Command", parse],
        {input: source, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
      assert.ifError(result.error);
      assert.equal(result.status, 0, result.stdout + result.stderr);
    }
  });

  test("native cache bootstrap creates only common parent and bridge; owners create support/export leaves", () => {
    const appData = "C:\\Users\\Fixture\\AppData\\Local", roots = command.localWindowsReleaseRoots(appData, "C:\\Users\\Fixture\\source\\mo-miner"),
      directories = new Set([appData]), created = /** @type {string[]} */ ([]), virtualFs = {
        lstatSync(/** @type {string} */ file) {return directories.has(file) ? {isDirectory: () => true, isSymbolicLink: () => false} : undefined;},
        mkdirSync(/** @type {string} */ file) {created.push(file); directories.add(file);},
        realpathSync(/** @type {string} */ file) {return file;},
      };
    command.ensureNativeWindowsReleaseRoots(roots, {fs: /** @type {any} */ (virtualFs)});
    assert.deepEqual(created, [path.win32.join(appData, "MoM"), path.win32.join(appData, "MoM", "native-wsl-windows"), roots.root, roots.bridge]);
    assert.equal(directories.has(roots.support), false); assert.equal(directories.has(roots.exports), false);
  });

  test("frontdoor orders host preflight before the WSL launch and passes argv-only locked jobs=2 settings", () => {
    const calls = /** @type {string[]} */ ([]), env = {SystemRoot: "C:\\Windows", MOM_RELEASE_VERSION: "1.2.3", MOM_FORCE_REBUILD: "yes"},
      checkout = "C:\\Users\\Fixture\\source\\mo-miner", run = {root: bridgeRunWindows, dev: 1, ino: 2}, result = command.launchLocalNativeWindowsRelease(env,
        (/** @type {string} */ executable, /** @type {string[]} */ args, /** @type {any} */ options) => {
          calls.push("spawn"); assert.equal(executable, "C:\\Windows\\System32\\wsl.exe");
          assert.ok(args.includes("MOM_BUILD_JOBS=2")); assert.ok(args.includes("MOM_FORCE_REBUILD=yes"));
          assert.ok(args.includes("ONEAPI_ROOT=" + oneApiRoot));
          assert.ok(args.includes("MOM_NATIVE_PERL_ROOT=" + nativePerlRoot));
          assert.ok(args.includes("WSLENV=" + relayEnvironment));
          assert.equal(options.env.WSLENV, relayEnvironment);
          assert.equal(options.shell, false); assert.equal(options.stdio, "inherit"); return {status: 0};
        }, /** @type {any} */ ({platform: "win32", arch: "x64", checkout, nodeVersion: "v24.15.0", releaseVersion() {calls.push("release"); return "1.2.3";},
          prepareWsl() {calls.push("wsl"); return {checkout: "/mnt/c/source/mo-miner", prefix: ["--distribution", "Ubuntu", "--exec"], executable: "C:\\Windows\\System32\\wsl.exe"};},
          preflightWindows() {
            calls.push("preflight"); return {appData: "C:\\Users\\Fixture\\AppData\\Local", oneApiRoot, nativePerlRoot,
              powershell: "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"};
          },
          ensureRoots(/** @type {any} */ roots) {calls.push("roots"); return roots;}, createId() {return id;},
          createRun(/** @type {string} */ bridge) {calls.push("run"); return {...run, root: bridge + "\\run-" + id};},
          removeRun() {calls.push("remove"); return true;}}));
    assert.equal(result, 0); assert.deepEqual(calls, ["release", "wsl", "preflight", "roots", "run", "spawn", "remove"]);
  });

  test("support bridge writes bounded JSON input and calls the existing PowerShell adapter without shell evaluation", t => {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-bridge-test-")), inputPosix = path.join(base, "input.json"),
      outputPosix = path.join(base, "output.json"), inputWindows = "C:\\fixture\\input.json", outputWindows = "C:\\fixture\\output.json",
      value = {workspace: "C:\\fixture\\support"};
    t.after(() => fs.rmSync(base, {recursive: true, force: true}));
    const result = command.runWindowsSupportBridge({inputPosix, outputPosix, inputWindows, outputWindows, value,
      powershellPosix: "/fixture/powershell.exe", scriptWindows: "C:\\repo\\support.ps1",
      environment: {ONEAPI_ROOT: oneApiRoot, MOM_NATIVE_PERL_ROOT: nativePerlRoot, WSLENV: "UNRELATED/p:ONEAPI_ROOT/p"}}, {
      windowsPathToWsl(/** @type {string} */ file) {return file === inputWindows ? inputPosix : outputPosix;},
      spawn(/** @type {string} */ executable, /** @type {string[]} */ args, /** @type {any} */ options) {
        assert.equal(executable, "/fixture/powershell.exe"); assert.deepEqual(args.slice(-4), ["-InputFile", inputWindows, "-OutputFile", outputWindows]);
        assert.equal(options.shell, false); assert.equal(options.stdio, "inherit");
        assert.equal(options.env.WSLENV, relayEnvironment);
        assert.equal(options.env.ONEAPI_ROOT, oneApiRoot);
        assert.equal(options.env.MOM_NATIVE_PERL_ROOT, nativePerlRoot);
        assert.deepEqual(JSON.parse(fs.readFileSync(inputPosix, "utf8")), value);
        fs.writeFileSync(outputPosix, JSON.stringify({accepted: true}) + "\n", {flag: "wx"}); return {status: 0};
      },
    });
    assert.deepEqual(result, {accepted: true});
  });

  test("invalid portable settings fail before WSL probes or creating a diagnostic slot", () => {
    let probes = 0;
    assert.throws(() => command.launchLocalNativeWindowsRelease({MOM_FORCE_REBUILD: "bad\nvalue"}, undefined,
      /** @type {any} */ ({platform: "win32", nodeVersion: "v24.15.0", releaseVersion: () => "1.2.3",
        prepareWsl() {probes++; throw new Error("must not probe");}})), /Invalid portable Windows-to-WSL setting/);
    assert.equal(probes, 0);
  });

  test("full WSL coordinator composes the original source/Core/handoff graph and publishes with stable stage cleanup", async t => {
    const {f, dependencies} = workerFixture(t), result = await command.runLocalWindowsWslRelease({checkout: repo, checkoutWindows: f.checkoutWindows,
        powershellWindows: f.powershellWindows, supportWorkspace: f.supportWorkspace, exportWorkspace: f.exportWorkspace,
        bridgeRunWindows: f.bridgeRunWindows}, dependencies), names = f.calls.map(row => row.name);
    assert.equal(result.archive, "C:\\release\\mom-v1.2.3-win.zip");
    assert.ok(names.indexOf("scope") < names.indexOf("preflight")); assert.ok(names.indexOf("node") < names.indexOf("prerequisites"));
    assert.ok(names.indexOf("bridge") < names.indexOf("windows-sdk")); assert.ok(names.indexOf("windows-sdk") < names.indexOf("distribution"));
    assert.ok(names.indexOf("distribution") < names.indexOf("publish"));
    assert.deepEqual(f.calls.find(row => row.name === "prerequisites")?.args, [f.workspace, 2, "/fixture/esbuild"]);
    const sdk = f.calls.find(row => row.name === "windows-sdk");
    assert.deepEqual(sdk?.args, [f.workspace, f.tools.cc, id, path.join(path.dirname(f.coreArchive), path.basename(f.coreArchive))]);
    assert.equal(fs.existsSync(path.join(f.workspace, "windows-release-work")), false);
    assert.equal(Object.hasOwn(f.bridgeResult, "runAttempt"), false);
  });

  test("failed release preserves one stable diagnostic stage and refuses to overwrite it", async t => {
    const {f, dependencies} = workerFixture(t, {failSdk: true}), input = {checkout: repo, checkoutWindows: f.checkoutWindows,
        powershellWindows: f.powershellWindows, supportWorkspace: f.supportWorkspace, exportWorkspace: f.exportWorkspace,
        bridgeRunWindows: f.bridgeRunWindows}, stage = path.join(f.workspace, "windows-release-work");
    await assert.rejects(command.runLocalWindowsWslRelease(input, dependencies), /fixture SDK failure/);
    assert.equal(fs.statSync(stage).isDirectory(), true);
    await assert.rejects(command.runLocalWindowsWslRelease(input, dependencies), /Inspect preserved local Windows release stage/);
    assert.equal(fs.statSync(stage).isDirectory(), true);
  });

  test("scope, observed Node, held lock and native roots fail before source acquisition", async t => {
    const {f, dependencies} = workerFixture(t), input = {checkout: repo, checkoutWindows: f.checkoutWindows,
      powershellWindows: f.powershellWindows, supportWorkspace: f.supportWorkspace, exportWorkspace: f.exportWorkspace,
      bridgeRunWindows: f.bridgeRunWindows};
    await assert.rejects(command.runLocalWindowsWslRelease(input, {...dependencies,
      verifyScope() {throw new Error("scope rejected");}}), /scope rejected/);
    await assert.rejects(command.runLocalWindowsWslRelease(input, {...dependencies, observedNodeVersion: "v22.22.1"}), /pinned build Node/);
    await assert.rejects(command.runLocalWindowsWslRelease(input, {...dependencies,
      environment: {...dependencies.environment, MOM_NATIVE_BUILD_LOCK: "0"}}), /held Engineering build lock/);
    await assert.rejects(command.runLocalWindowsWslRelease(input, {...dependencies,
      environment: {...dependencies.environment, ONEAPI_ROOT: "relative"}}), /selected native oneAPI and Perl roots/);
    assert.equal(f.calls.some(row => row.name === "node" || row.name === "prerequisites"), false);
  });

  test("bridge admission projects real result roots and rejects malformed retirement or changed receipts", t => {
    const {f, dependencies} = workerFixture(t), input = {supportWorkspace, exportWorkspace},
      admitted = command.admitWindowsBridgeResult(f.bridgeResult, input, dependencies);
    assert.equal(admitted.controllerSupport.io.root, f.windowsToPosix.get(f.controllerSupport.io.root));
    assert.equal(admitted.controllerSupport.hermes.root, f.windowsToPosix.get(f.controllerSupport.hermes.root));
    assert.equal(admitted.coreArchive, f.coreArchive);
    assert.throws(() => command.admitWindowsBridgeResult({...f.bridgeResult, supportRetirementCandidates: []}, input, dependencies), /unknown or missing fields/);
    assert.throws(() => command.admitWindowsBridgeResult({...f.bridgeResult,
      coreReceipt: {...f.bridgeResult.coreReceipt, sha256: "f".repeat(64)}}, input, dependencies), /failed WSL-side admission/);
  });

  test("bounded bridge JSON rejects malformed, oversized and redirected files", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-json-")), file = path.join(root, "bridge.json");
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.writeFileSync(file, '{"ok":true}\n');
    assert.deepEqual(command.readBoundedJson(file, 64), {ok: true});
    assert.throws(() => command.readBoundedJson(file, 4), /unbounded/);
    const link = path.join(root, "link.json"); fs.symlinkSync(file, link);
    assert.throws(() => command.readBoundedJson(link, 64), /aliased/);
    fs.writeFileSync(file, "{broken}");
    assert.throws(() => command.readBoundedJson(file, 64), /valid UTF-8/);
  });

  test("bridge JSON growth after opening is rejected within its allocation bound", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-json-growth-")), file = path.join(root, "bridge.json");
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.writeFileSync(file, "{}");
    const read = fs.readSync;
    let changed = false, capacity = 0;
    t.mock.method(fs, "readSync", (/** @type {number} */ fd, /** @type {Buffer} */ buffer,
      /** @type {number} */ offset, /** @type {number} */ length, /** @type {number|null} */ position) => {
      capacity = Math.max(capacity, buffer.length);
      if (!changed) {changed = true; fs.appendFileSync(file, " ".repeat(128));}
      return read(fd, buffer, offset, length, position);
    });
    assert.throws(() => command.readBoundedJson(file, 16), /changed during its bounded read/);
    assert.equal(capacity, 17);
  });
});
