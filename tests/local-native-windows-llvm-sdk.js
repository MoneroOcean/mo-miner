"use strict";

const {describe, test} = require("node:test"), assert = require("node:assert/strict"),
  fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const local = require("../scripts/build-local-native-sdk"), {fileHash, readBuildJson} = require("../scripts/build-source-inputs"),
  windowsLlvm = require("../scripts/build-native-windows-llvm-sdk"), {archiveName, archiveSha256} = require("../scripts/native-llvm-sdk"),
  windowsIntelRuntime = require("../runtime/windows-intel-runtime.json"),
  {localWindowsLlvmSdkPaths, localWindowsLlvmDescriptorTools, verifyLocalWindowsLlvmOutput, buildLocalNativeWindowsLlvmSdk} =
    require("../scripts/build-local-native-windows-llvm-sdk");

/** Inert files only; all compiler/build/audit APIs are injected and no SDK is compiled. @param {import('node:test').TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-windows-llvm-sdk-")), workspace = path.join(root, "workspace"),
    hermesRoot = path.join(root, "hermes"), windowsSdkRoot = path.join(workspace, "windows-host-sdk", "sdk"),
    support = local.supportPaths(workspace), bin = path.join(workspace, "llvm21", "sdk", "bin"),
    compilerBin = path.join(root, "intel", "compiler", "2026.2", "bin"), resource = path.join(root, "intel", "compiler", "2026.2", "lib", "clang", "23", "include"),
    tools = /** @type {any} */ ({}), selected = /** @type {any} */ ({});
  local.admitWorkspace(workspace, fs.realpathSync(path.resolve(__dirname, "..")));
  fs.mkdirSync(hermesRoot); fs.mkdirSync(windowsSdkRoot, {recursive: true}); fs.mkdirSync(support.buildInputs, {recursive: true});
  fs.mkdirSync(bin, {recursive: true}); fs.mkdirSync(compilerBin, {recursive: true}); fs.mkdirSync(resource, {recursive: true});
  /** @param {string} file @param {string} [bytes] */
  const executable = (file, bytes = path.basename(file)) => {
    fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, bytes); fs.chmodSync(file, 0o700); return file;
  };
  const lld = executable(path.join(bin, "lld")), llvmNm = executable(path.join(bin, "llvm-nm"));
  fs.symlinkSync("lld", path.join(bin, "lld-link"));
  const compiler = executable(path.join(compilerBin, "icx-cl"));
  for (const [name, file] of Object.entries({git: path.join(root, "git"), cmake: path.join(root, "cmake"), ninja: path.join(root, "ninja"),
    systemdRun: path.join(root, "systemd-run"), cc: path.join(compilerBin, "icx"), cxx: path.join(compilerBin, "icpx")})) {
    tools[name] = executable(file);
  }
  Object.assign(selected, tools);
  const files = {"bin/lld": {type: "file", bytes: fs.statSync(lld).size, sha256: fileHash(lld)},
    "bin/lld-link": {type: "symlink", target: "lld"},
    "bin/llvm-nm": {type: "file", bytes: fs.statSync(llvmNm).size, sha256: fileHash(llvmNm)}};
  const admission = {version: 1, role: "linux-llvm21-source-sdk", stage: "audited", buildOnly: true, compiled: true,
    executed: false, publicationQualified: false, sdk: {path: path.join(workspace, "llvm21", "sdk"), snapshot: {files}}};
  const input = {workspace, windowsHostSdk: {root: windowsSdkRoot, pins: {resultSha256: "1".repeat(64), descriptorSha256: "2".repeat(64),
    filesSha256: "3".repeat(64), overlaySha256: "4".repeat(64)}},
  hermes: {root: hermesRoot, localBuildId: "12345678-1234-4234-9234-123456789abc", handoffSha256: "5".repeat(64)}, tools};
  const receipt = {target: "build", archives: {llvm: {path: path.join(support.buildInputs, archiveName), sha256: archiveSha256}}},
    calls = /** @type {Array<{name:string,args:any[]}>} */ ([]), dependencies = {
      verifyScope() {calls.push({name: "scope", args: []});},
      resolveTools() {calls.push({name: "tools", args: []}); return selected;},
      /** @param {string} archiveRoot */
      verifyArchives(archiveRoot) {calls.push({name: "archives", args: [archiveRoot]}); return receipt;},
      /** @param {string} llvmRoot */
      verifyLinuxLlvm(llvmRoot) {calls.push({name: "linux", args: [llvmRoot]}); return admission;},
      /** @param {any} value */
      composeDescriptor(value) {
        calls.push({name: "compose", args: [value]});
        return {descriptor: {version: 1, source: value.hermesExpectation.localBuildId, clangClSha256: value.tools.clangCl.sha256},
          provenance: {status: "PASS", source: value.hermesExpectation.handoffSha256, clangClSha256: value.tools.clangCl.sha256}};
      },
      /** @param {string} descriptor @param {string} archive @param {string} output @param {{cmake:string,ninja:string}} cmakeTools
       * @param {number} jobs @param {NodeJS.ProcessEnv} environment */
      configure(descriptor, archive, output, cmakeTools, jobs, environment) {
        calls.push({name: "configure", args: [descriptor, archive, output, cmakeTools, jobs, environment]});
        fs.mkdirSync(output);
        fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({plan: {output, descriptorFile: descriptor, archive, jobs,
          tools: cmakeTools}}) + "\n");
      },
      /** @param {string} output @param {string} phase @param {NodeJS.ProcessEnv} environment */
      buildPhase(output, phase, environment) {calls.push({name: "build", args: [output, phase, environment]});},
      /** @param {string} output @param {NodeJS.ProcessEnv} environment */
      audit(output, environment) {calls.push({name: "audit", args: [output, environment]}); return {};},
      /** @param {string} output @param {NodeJS.ProcessEnv} environment */
      verify(output, environment) {calls.push({name: "verify", args: [output, environment]}); return {role: "verified"};},
      /** @param {string} output @param {any} result */
      verifyOutputInventory(output, result) {assert.equal(result.plan.output, output);},
      /** @type {NodeJS.ProcessEnv} */ environment: {MOM_BUILD_JOBS: "2"},
    };
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, workspace, hermesRoot, windowsSdkRoot, support, bin, compiler, resource, tools, selected, input, receipt, admission, calls, dependencies};
}

/** @param {unknown} input @param {any} dependencies */
function buildLocal(input, dependencies) {return buildLocalNativeWindowsLlvmSdk(/** @type {any} */ (input), dependencies);}

describe("Local Windows LLVM SDK coordinator", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("guards scope before input inspection", () => {
    const calls = /** @type {string[]} */ ([]);
    assert.throws(() => buildLocal(null, {
      verifyScope() {calls.push("scope");},
    }), /one workspace/);
    assert.deepEqual(calls, ["scope"]);
  });

  test("pins Intel and Linux LLVM tools to their actual selected files/admission", t => {
    const f = fixture(t), descriptorTools = localWindowsLlvmDescriptorTools(f.tools, f.admission);
    assert.equal(descriptorTools.clangCl.path, f.compiler);
    assert.equal(descriptorTools.clangCl.sha256, fileHash(f.compiler), "the selected Linux icx-cl bytes supply the descriptor pin");
    assert.notEqual(descriptorTools.clangCl.sha256, windowsIntelRuntime.compiler.sha256, "the Windows .exe pin is not used for the Linux icx-cl");
    assert.equal(descriptorTools.clangResourceInclude, f.resource);
    assert.equal(descriptorTools.lldLink.path, path.join(f.bin, "lld-link"));
    assert.equal(descriptorTools.lldLink.linkTarget, "lld");
    assert.equal(descriptorTools.llvmNm.sha256, f.admission.sdk.snapshot.files["bin/llvm-nm"].sha256);
    const redirected = structuredClone(f.admission); redirected.sdk.snapshot.files["bin/lld-link"].target = "outside";
    assert.throws(() => localWindowsLlvmDescriptorTools(f.tools, redirected), /alias differs from its admitted SDK snapshot/);
  });

  test("requires the exact public output-root receipt and phase inventory", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-windows-llvm-inventory-")), runId = "12345678-1234-4234-9234-123456789abc",
      plan = /** @type {Record<string,string>} */ (Object.fromEntries(["archiveCopy", "source", "build", "overlay", "sourceOptionsFile", "ruleOverride", "toolchainFile"]
        .map(name => [name, path.join(root, name === "archiveCopy" ? archiveName : name)])));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    for (const name of ["source", "build"]) {fs.mkdirSync(path.join(root, name));}
    const files = ["archiveCopy", "overlay", "sourceOptionsFile", "ruleOverride", "toolchainFile", "plan.json", "result.json",
      "producer-snapshot.js", "extract.log", "configure.log", "build-state.json", "result-current.json", "admitted.json", "admitted-objects.json"]
      .map(name => name in plan ? path.basename(plan[name] || name) : name);
    for (const phase of windowsLlvm.buildPhases) {files.push("build-" + phase.name + "-" + runId + ".log", "phase-" + phase.name + "-" + runId + ".json");}
    for (const name of files) {fs.writeFileSync(path.join(root, name), "fixture");}
    verifyLocalWindowsLlvmOutput(root, plan, runId);
    fs.writeFileSync(path.join(root, "unknown"), "preserve");
    assert.throws(() => verifyLocalWindowsLlvmOutput(root, plan, runId), /contains unknown entries/);
    assert.equal(fs.readFileSync(path.join(root, "unknown"), "utf8"), "preserve");
  });

  test("configures once, runs every public phase in order, and admits with the default two jobs", t => {
    const f = fixture(t), result = buildLocal(f.input, f.dependencies), paths = localWindowsLlvmSdkPaths(f.workspace),
      configure = f.calls.find(call => call.name === "configure");
    assert.ok(configure);
    assert.deepEqual(configure.args.slice(0, 5), [paths.descriptor, paths.archive, paths.output, {cmake: f.tools.cmake, ninja: f.tools.ninja}, 2]);
    assert.deepEqual(f.calls.filter(call => call.name === "build").map(call => call.args[1]), windowsLlvm.buildPhases.map(phase => phase.name));
    assert.equal(f.calls.filter(call => call.name === "audit").length, 1);
    assert.equal(f.calls.filter(call => call.name === "verify").length, 1);
    assert.equal(result.jobs, 2); assert.equal(result.reused, false); assert.equal(result.buildOnly, true);
    assert.equal(result.archive, path.join(f.support.buildInputs, archiveName));
    assert.equal(readBuildJson(paths.descriptor).source, f.input.hermes.localBuildId);
    assert.equal(fs.readFileSync(paths.provenance, "utf8").includes(f.input.hermes.handoffSha256), true);
    assert.deepEqual(fs.readdirSync(paths.root).sort(), ["descriptor.json", "output", "provenance.json"]);
    assert.equal(Object.hasOwn(result, "runAttempt"), false);
  });

  test("reuses a valid warm SDK without configure, rebuilding, or restamping", t => {
    const f = fixture(t), first = buildLocal(f.input, f.dependencies), before = f.calls.length;
    const second = buildLocal(f.input, f.dependencies), later = f.calls.slice(before).map(call => call.name);
    assert.equal(first.descriptorSha256, second.descriptorSha256);
    assert.equal(second.reused, true);
    assert.deepEqual(later.filter(name => ["configure", "build", "audit", "verify"].includes(name)), ["verify"]);
  });

  test("reuses a warm SDK recorded with another MOM_BUILD_JOBS and reports this run's selection", t => {
    const f = fixture(t), paths = localWindowsLlvmSdkPaths(f.workspace);
    buildLocal(f.input, {...f.dependencies, environment: {MOM_BUILD_JOBS: "3"}});
    assert.equal(readBuildJson(path.join(paths.output, "result.json")).plan.jobs, 3);
    const second = buildLocal(f.input, {...f.dependencies, environment: {MOM_BUILD_JOBS: "5"}});
    assert.equal(second.reused, true); assert.equal(second.jobs, 5);
    assert.equal(f.calls.filter(call => call.name === "configure").length, 1);
    assert.equal(readBuildJson(path.join(paths.output, "result.json")).plan.jobs, 3, "the recorded plan is not rewritten");
  });

  test("rejects a changed Linux icx-cl hash on warm admission", t => {
    const f = fixture(t), first = buildLocal(f.input, f.dependencies), verifyCount = f.calls.filter(call => call.name === "verify").length;
    fs.appendFileSync(f.compiler, "replacement bytes");
    assert.throws(() => buildLocal(f.input, f.dependencies), /descriptor is stale/);
    assert.equal(fs.existsSync(path.join(first.output, "result.json")), true, "the original builder output is preserved");
    assert.equal(f.calls.filter(call => call.name === "verify").length, verifyCount, "stale descriptor is rejected before warm verification");
    const recomposed = f.calls.filter(call => call.name === "compose").at(-1)?.args[0];
    assert.ok(recomposed);
    assert.equal(recomposed.tools.clangCl.sha256, fileHash(f.compiler));
  });

  test("preserves unknown, partial, descriptor-stale and admission-stale outputs", t => {
    const f = fixture(t), paths = localWindowsLlvmSdkPaths(f.workspace);
    fs.mkdirSync(paths.root); fs.writeFileSync(path.join(paths.root, "unknown"), "preserve");
    assert.throws(() => buildLocal(f.input, f.dependencies), /contains unknown entries/);
    assert.equal(fs.readFileSync(path.join(paths.root, "unknown"), "utf8"), "preserve");
    fs.unlinkSync(path.join(paths.root, "unknown")); fs.writeFileSync(paths.descriptor, "partial\n");
    assert.throws(() => buildLocal(f.input, f.dependencies), /incomplete or contains unknown entries/);
    assert.equal(fs.readFileSync(paths.descriptor, "utf8"), "partial\n");

    const f2 = fixture(t), good = buildLocal(f2.input, f2.dependencies), descriptorPath = good.descriptor;
    fs.chmodSync(descriptorPath, 0o600); fs.appendFileSync(descriptorPath, "stale\n");
    assert.throws(() => buildLocal(f2.input, f2.dependencies), /descriptor is stale/);
    assert.equal(fs.readFileSync(descriptorPath, "utf8").endsWith("stale\n"), true);

    const f3 = fixture(t), built = buildLocal(f3.input, f3.dependencies), previousVerify = f3.dependencies.verify;
    f3.dependencies.verify = () => {throw new Error("stale public admission");};
    assert.throws(() => buildLocal(f3.input, f3.dependencies), /stale public admission/);
    assert.equal(fs.existsSync(path.join(built.output, "result.json")), true);
    f3.dependencies.verify = previousVerify;
  });

  test("rejects mismatched selected tools and changed original archive receipts", t => {
    const f = fixture(t), wrongTools = {...f.input, tools: {...f.tools, cmake: "/other/cmake"}};
    assert.throws(() => buildLocal(wrongTools, f.dependencies), /differ from the current pinned selection/);
    assert.equal(fs.existsSync(localWindowsLlvmSdkPaths(f.workspace).root), false);
    const g = fixture(t), original = g.dependencies.verifyArchives; let checks = 0;
    g.dependencies.verifyArchives = (root) => {checks++; const receipt = original(root); return checks === 1 ? receipt : {...receipt, changed: true};};
    assert.throws(() => buildLocal(g.input, g.dependencies), /inputs changed during production/);
    assert.equal(fs.existsSync(localWindowsLlvmSdkPaths(g.workspace).descriptor), true);
    assert.equal(fs.existsSync(localWindowsLlvmSdkPaths(g.workspace).output), true);
  });
});
