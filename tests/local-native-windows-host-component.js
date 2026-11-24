"use strict";

const {describe, test} = require("node:test"), assert = require("node:assert/strict"),
  fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const local = require("../scripts/build-local-native-sdk"), rocm = require("../scripts/build-local-native-rocm-sdk"),
  acpp = require("../scripts/build-local-native-acpp-sdk"), windowsLlvm = require("../scripts/build-local-native-windows-llvm-sdk"),
  {fileHash} = require("../scripts/build-source-inputs"),
  {localWindowsHostComponentPaths, buildLocalNativeWindowsHostComponent} = require("../scripts/build-local-native-windows-host-component");
const repo = fs.realpathSync(path.resolve(__dirname, "..")),
  localBuildId = "12345678-1234-4234-9234-123456789abc", handoffSha256 = "5".repeat(64);

/** Inert files only; all producer/auditor boundaries are injected. No SDK is built or downloaded.
 * @param {import('node:test').TestContext} t */
function fixture(t) {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-windows-host-component-")), workspace = path.join(temp, "workspace"),
    support = local.supportPaths(workspace), rocmPaths = rocm.localRocmSdkPaths(workspace), acppPaths = acpp.localAcppSdkPaths(workspace),
    targetLlvm = windowsLlvm.localWindowsLlvmSdkPaths(workspace), sdkRoot = path.join(workspace, "windows-host-sdk", "sdk"),
    hermesRoot = path.join(temp, "windows-hermes"), dpcpp = local.localDpcppCudaState(support),
    tools = {git: "/fixture/git", cmake: "/fixture/cmake", ninja: "/fixture/ninja", systemdRun: "/fixture/systemd-run",
      cc: "/fixture/intel/icx", cxx: "/fixture/intel/icpx"}, pins = {resultSha256: "1".repeat(64), descriptorSha256: "2".repeat(64),
      filesSha256: "3".repeat(64), overlaySha256: "4".repeat(64)},
    input = {workspace, windowsHostSdk: {root: sdkRoot, pins}, hermes: {root: hermesRoot, localBuildId, handoffSha256}, tools},
    archiveBytes = Buffer.from("pinned-cudart"), archiveSha256 = fileHashBytes(archiveBytes), calls = /** @type {string[]} */ ([]),
    component = {files: {fixture: true}, provenance: {localBuildId}}, linux = {role: "linux-llvm21-source-sdk", stage: "audited",
      buildOnly: true, compiled: true, executed: false, publicationQualified: false,
      sdk: {path: path.join(support.llvmSdk, "sdk"), snapshot: {files: {}}}},
    original = {provenance: {localBuildId}, llvm: {admission: {generationId: "11111111-2222-4333-8444-555555555555"}}};
  local.admitWorkspace(workspace, repo);
  fs.mkdirSync(hermesRoot);
  for (const directory of [support.llvmSdk, support.dpcppCudaRoot, dpcpp.dpcppRoot, dpcpp.cudaRoot, path.join(dpcpp.cudaRoot, "include"),
    rocmPaths.source, rocmPaths.hipHeadersRoot, rocmPaths.resourcesRoot, rocmPaths.sdkRoot, rocmPaths.hostSdkRoot, rocmPaths.runtimeOutput]) {
    fs.mkdirSync(directory, {recursive: true});
  }
  fs.writeFileSync(dpcpp.manifest, "{}\n");
  fs.mkdirSync(path.dirname(targetLlvm.root), {recursive: true});
  let amdhipBuildId = /** @type {string | undefined} */ (undefined);
  const dependencies = /** @type {Parameters<typeof buildLocalNativeWindowsHostComponent>[1]} */ ({
    // Fixtures assert the local default; a CI job's own MOM_BUILD_JOBS must not leak in.
    environment: {MOM_BUILD_JOBS: "2"},
    verifyScope() {calls.push("scope");},
    resolveTools() {calls.push("tools"); return tools;},
    readWindowsHostSdk(root, selectedPins) {calls.push("host-sdk"); assert.equal(root, sdkRoot); assert.deepEqual(selectedPins, pins); return /** @type {any} */ ({root, pins});},
    readHermes(root, expected) {calls.push("hermes"); assert.equal(root, hermesRoot); return /** @type {any} */ ({...expected, intelRuntime: {libraries: {}}});},
    verifyLinuxLlvm(root) {calls.push("linux-llvm"); assert.equal(root, support.llvmSdk); return /** @type {any} */ (linux);},
    auditRocmComgrInputs(context) {
      calls.push("rocm-inputs"); const selected = /** @type {any} */ (context); assert.equal(selected.workspace, workspace);
      const state = rocm.localRocmBuildState(selected);
      return /** @type {any} */ ({state, hip: {include: state.hipIncludeRoot}, comgr: {sources: true}, receipt: {source: "pinned"}});
    },
    auditDpcppCuda(state, manifest) {calls.push("dpcpp-cuda"); assert.equal(state.manifest, dpcpp.manifest); assert.deepEqual(manifest, {}); return /** @type {any} */ ({manifest: state.manifest});},
    prepareAcppSources(context) {
      calls.push("acpp-sources"); assert.equal(/** @type {any} */ (context).paths.preparedCheckout, acppPaths.preparedCheckout);
      fs.mkdirSync(acppPaths.preparedCheckout, {recursive: true}); return {sources: true, prepared: true};
    },
    verifyAcppSource() {calls.push("verify-acpp-source"); return /** @type {any} */ ({revision: "da2463e45aa90aa36306c45abcfc05b87de51bc6"});},
    buildWindowsLlvm(selected) {
      calls.push("windows-llvm"); assert.equal(selected.workspace, workspace);
      fs.mkdirSync(targetLlvm.output, {recursive: true});
      writeOnce(targetLlvm.descriptor, JSON.stringify({version: 1, localBuildId}) + "\n");
      writeOnce(targetLlvm.provenance, JSON.stringify({localBuildId}) + "\n");
      return /** @type {any} */ ({workspace, root: targetLlvm.root, output: targetLlvm.output, descriptor: targetLlvm.descriptor,
        provenance: targetLlvm.provenance, archive: path.join(support.buildInputs, "llvm-project-21.1.8.src.tar.xz"),
        descriptorSha256: fileHash(targetLlvm.descriptor), provenanceSha256: fileHash(targetLlvm.provenance),
        admission: {admission: {generationId: original.llvm.admission.generationId}}, jobs: 2,
        buildOnly: true, compiled: true, executed: false, publicationQualified: false, reused: calls.filter(name => name === "windows-llvm").length > 1});
    },
    buildResources(config, output, environment) {
      calls.push("resources"); assert.equal(/** @type {any} */ (config).target, "win32");
      assert.equal(environment?.["MOM_BUILD_JOBS"], "2");
      fs.mkdirSync(path.join(output, "install"), {recursive: true}); fs.writeFileSync(path.join(output, "result.json"), "resource\n");
      return /** @type {any} */ ({version: 1, target: "win32"});
    },
    verifyDeviceResources(receipt, install) {calls.push("device-resources"); assert.equal(install, path.join(path.dirname(receipt), "install")); return /** @type {any} */ ({sha256: fileHash(receipt)});},
    configureComgr(config, environment) {
      calls.push("comgr-configure"); const selected = /** @type {any} */ (config);
      assert.equal(selected.jobs, 2); assert.equal(environment?.["MOM_AOT_BUILD_ONLY"], "1"); fs.mkdirSync(selected.output);
      return /** @type {any} */ ({version: 1, role: "windows-comgr-static-host", stage: "configured"});
    },
    buildComgr(output) {calls.push("comgr-build"); fs.writeFileSync(path.join(output, "result.json"), "comgr\n");},
    prepareHiprtcSource(source, output) {calls.push("hiprtc-source"); assert.equal(source, path.join(rocmPaths.source, "roots/clr")); fs.mkdirSync(output); return output;},
    verifyHiprtcSource(source) {calls.push("verify-hiprtc-source"); assert.equal(source, localWindowsHostComponentPaths(workspace, "fixture.zip").hiprtcSource); return {source, revision: "5b591b2ff64aac4c1c08be879b0d5f89fd713d50"};},
    buildHiprtc(config, environment) {
      calls.push("hiprtc-build"); assert.equal(config.output, localWindowsHostComponentPaths(workspace, "fixture.zip").hiprtc);
      assert.equal(environment?.["MOM_BUILD_JOBS"], "2"); fs.mkdirSync(config.output); fs.writeFileSync(path.join(config.output, "result.json"), "hiprtc\n");
      return /** @type {any} */ ({version: 1, role: "windows-hiprtc-static-host"});
    },
    cudartCatalog() {
      calls.push("cudart-catalog"); return /** @type {any} */ ({archive: {fileName: "fixture.zip", bytes: archiveBytes.length,
        sha256: archiveSha256, url: "https://example.invalid/fixture.zip"}});
    },
    downloadCudart(output) {calls.push("cudart-download"); fs.writeFileSync(output, archiveBytes); return output;},
    prepareCudart(archive, output) {
      calls.push("cudart-prepare"); assert.equal(fileHash(archive), archiveSha256);
      fs.mkdirSync(path.join(output, "runtime"), {recursive: true}); fs.writeFileSync(path.join(output, "runtime/cudart_static.lib"), "cudart");
      return /** @type {any} */ (undefined);
    },
    readCudart(output) {
      calls.push("cudart-read"); return /** @type {any} */ ({cudartStatic: path.join(output, "runtime/cudart_static.lib"),
        receipt: {source: {archivePath: localWindowsHostComponentPaths(workspace, "fixture.zip").archive, sha256: archiveSha256}}});
    },
    amdhipCatalog() {return /** @type {any} */ ({library: {fileName: "amdhip64.lib"}});},
    async prepareAmdhip(value) {
      calls.push("amdhip-prepare"); assert.match(value.localBuildId, /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i);
      assert.notEqual(value.localBuildId, localBuildId); amdhipBuildId = value.localBuildId; fs.mkdirSync(value.output);
      fs.writeFileSync(path.join(value.output, "amdhip64.lib"), "amdhip"); return /** @type {any} */ ({localBuildId: amdhipBuildId});
    },
    readAmdhip(output) {calls.push("amdhip-read"); return /** @type {any} */ ({amdhip64: path.join(output, "amdhip64.lib"), receipt: {localBuildId: amdhipBuildId}});},
    buildAcpp(config, environment) {
      calls.push("acpp-build"); assert.equal(config.jobs, 2); assert.equal(environment?.["MOM_AOT_BUILD_ONLY"], "1");
      fs.mkdirSync(config.output); fs.writeFileSync(path.join(config.output, "result.json"), "acpp\n");
      return /** @type {any} */ ({version: 1, role: "windows-acpp-static-host"});
    },
    exportComponent(value, environment) {
      calls.push("export"); assert.equal(value.output, localWindowsHostComponentPaths(workspace, "fixture.zip").component);
      assert.equal(environment?.["MOM_BUILD_JOBS"], "2"); fs.mkdirSync(value.output); fs.writeFileSync(path.join(value.output, "component.json"), "component\n");
      return /** @type {any} */ (undefined);
    },
    verifyProducers(value) {calls.push("verify-producers"); assert.equal(value.llvmOutput, targetLlvm.output); return /** @type {any} */ (original);},
    matchComponent(root) {calls.push("match-component"); assert.equal(root, localWindowsHostComponentPaths(workspace, "fixture.zip").component); return /** @type {any} */ (component);},
    readComponent(root) {calls.push("read-component"); assert.equal(root, localWindowsHostComponentPaths(workspace, "fixture.zip").component); return /** @type {any} */ (component);},
  });
  t.after(() => fs.rmSync(temp, {recursive: true, force: true}));
  return {temp, workspace, support, rocmPaths, acppPaths, targetLlvm, sdkRoot, hermesRoot, tools, pins, input, dependencies, calls,
    paths: localWindowsHostComponentPaths(workspace, "fixture.zip"), get amdhipBuildId() {return amdhipBuildId;}};
}

/** @param {Buffer} value */
function fileHashBytes(value) {return require("node:crypto").createHash("sha256").update(value).digest("hex");}
/** @param {string} file @param {string} contents */
function writeOnce(file, contents) {if (!fs.existsSync(file)) {fs.writeFileSync(file, contents, {flag: "wx"});}}

describe("Local Windows host-component coordinator", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("checks the active scope before inspecting inputs", async () => {
    const calls = /** @type {string[]} */ ([]);
    const dependencies = /** @type {any} */ ({
      verifyScope() {calls.push("scope"); throw new Error("scope missing");},
      resolveTools() {calls.push("tools"); return {};},
    });
    await assert.rejects(buildLocalNativeWindowsHostComponent(/** @type {any} */ (null), dependencies), /scope missing/);
    assert.deepEqual(calls, ["scope"]);
  });

  test("builds the original producer sequence with jobs=2 and re-admits its unchanged component", async t => {
    const f = fixture(t), result = await buildLocalNativeWindowsHostComponent(f.input, f.dependencies),
      producerCalls = f.calls.filter(name => ["resources", "comgr-configure", "comgr-build", "hiprtc-source", "hiprtc-build",
        "cudart-download", "cudart-prepare", "amdhip-prepare", "acpp-build", "export"].includes(name));
    assert.deepEqual(producerCalls, ["resources", "comgr-configure", "comgr-build", "hiprtc-source", "hiprtc-build", "cudart-download",
      "cudart-prepare", "amdhip-prepare", "acpp-build", "export"]);
    assert.equal(result.workspace, f.workspace); assert.equal(result.componentRoot, f.paths.component);
    assert.equal(result.jobs, 2); assert.equal(result.reused, false); assert.equal(result.buildOnly, true);
    assert.equal(result.compiled, true); assert.equal(result.executed, false); assert.equal(result.publicationQualified, false);
    assert.equal(Object.hasOwn(result, "runAttempt"), false);
    assert.match(f.amdhipBuildId || "", /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i);
    assert.deepEqual(fs.readdirSync(f.paths.root).sort(), ["acpp", "amdhip", "archives", "comgr", "component", "cudart", "hiprtc",
      "hiprtc-config.json", "hiprtc-source", "resources", "resources-config.json"].sort());
    assert.equal(JSON.parse(fs.readFileSync(f.paths.resourcesConfig, "utf8")).target, "win32");
    assert.equal(JSON.parse(fs.readFileSync(f.paths.hiprtcConfig, "utf8")).source, f.paths.hiprtcSource);
  });

  test("warm readmission uses original readers/matcher and never re-runs producers", async t => {
    const f = fixture(t), first = await buildLocalNativeWindowsHostComponent(f.input, f.dependencies), before = f.calls.length,
      second = await buildLocalNativeWindowsHostComponent(f.input, f.dependencies), later = f.calls.slice(before);
    assert.equal(first.componentRoot, second.componentRoot);
    assert.equal(second.reused, true);
    assert.ok(later.includes("verify-producers")); assert.ok(later.includes("match-component")); assert.ok(later.includes("read-component"));
    assert.deepEqual(later.filter(name => ["resources", "comgr-configure", "comgr-build", "hiprtc-source", "hiprtc-build", "cudart-download",
      "cudart-prepare", "amdhip-prepare", "acpp-build", "export"].includes(name)), []);
    assert.equal(fs.readFileSync(f.paths.component + "/component.json", "utf8"), "component\n");
  });

  test("rejects a Windows LLVM generation change during production", async t => {
    const f = fixture(t), deps = /** @type {any} */ (f.dependencies), originalBuild = deps.buildWindowsLlvm;
    let admissions = 0;
    const buildWithDrift = /** @type {(...args: any[]) => any} */ ((...args) => {
      const result = originalBuild(...args);
      if (++admissions === 2) {result.admission.admission.generationId = "22222222-2222-4222-8222-222222222222";}
      return result;
    });
    deps.buildWindowsLlvm = buildWithDrift;
    await assert.rejects(buildLocalNativeWindowsHostComponent(f.input, f.dependencies), /inputs changed during production/);
  });

  test("rejects unknown managed output without deleting or overwriting it", async t => {
    const f = fixture(t); fs.mkdirSync(f.paths.root); fs.writeFileSync(path.join(f.paths.root, "operator-data"), "preserve");
    await assert.rejects(buildLocalNativeWindowsHostComponent(f.input, f.dependencies), /unknown entries/);
    assert.equal(fs.readFileSync(path.join(f.paths.root, "operator-data"), "utf8"), "preserve");
    assert.deepEqual(f.calls, ["scope", "tools", "cudart-catalog"]);
  });

  test("rejects a partial managed output without repairing it", async t => {
    const f = fixture(t); fs.mkdirSync(f.paths.root); fs.mkdirSync(f.paths.resources);
    await assert.rejects(buildLocalNativeWindowsHostComponent(f.input, f.dependencies), /incomplete or contains unknown entries/);
    assert.deepEqual(fs.readdirSync(f.paths.root), ["resources"]);
    assert.deepEqual(f.calls, ["scope", "tools", "cudart-catalog"]);
  });

  test("preserves and rejects a stale descriptor on warm readmission", async t => {
    const f = fixture(t); await buildLocalNativeWindowsHostComponent(f.input, f.dependencies);
    fs.appendFileSync(f.paths.resourcesConfig, "stale\n");
    await assert.rejects(buildLocalNativeWindowsHostComponent(f.input, f.dependencies), /resource config is stale/);
    assert.equal(fs.readFileSync(f.paths.resourcesConfig, "utf8").endsWith("stale\n"), true);
    assert.equal(fs.existsSync(path.join(f.paths.component, "component.json")), true);
  });
});
