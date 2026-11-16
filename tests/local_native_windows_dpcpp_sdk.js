"use strict";

const {describe, test} = require("node:test"), assert = require("node:assert/strict"),
  fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const local = require("../scripts/build-local-native-sdk"), windowsLlvm = require("../scripts/build-local-native-windows-llvm-sdk"),
  windowsResources = require("../scripts/build-local-native-windows-jit-resources"), windowsJit = require("../scripts/build-local-native-windows-jit-sdk"),
  {fileHash} = require("../scripts/build-source-inputs"),
  {localWindowsDpcppSdkPaths, buildLocalNativeWindowsDpcppSdk} = require("../scripts/build-local-native-windows-dpcpp-sdk");

/** Inert handoff/receipt fixtures only; public build seams prevent compilers, downloads and devices from running.
 * @param {import("node:test").TestContext} t */
function fixture(t) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-windows-dpcpp-sdk-")), workspace = path.join(base, "workspace"),
    repo = fs.realpathSync(path.resolve(__dirname, ".."));
  local.admitWorkspace(workspace, repo);
  const paths = localWindowsDpcppSdkPaths(workspace), llvmPaths = windowsLlvm.localWindowsLlvmSdkPaths(workspace),
    resourcePaths = windowsResources.localWindowsJitResourcePaths(workspace), jitPaths = windowsJit.localWindowsJitSdkPaths(workspace),
    support = local.supportPaths(workspace), ioRoot = path.join(base, "controller/io"), hermesRoot = path.join(base, "controller/hermes"),
    id = "12345678-1234-4234-9234-123456789abc", handoffSha256 = "a".repeat(64), sourceFile = path.join(workspace, "windows-sources/sources.json"),
    calls = /** @type {Array<{name:string,args:any[]}>} */ ([]),
    tools = /** @type {any} */ ({cc: "/fixture/icx", cmake: "/fixture/cmake", cxx: "/fixture/icpx", git: "/fixture/git", ninja: "/fixture/ninja", systemdRun: "/fixture/systemd-run"}),
    windowsHostSdk = {workspace, root: path.join(workspace, "windows-host-sdk/sdk"), pins: {resultSha256: "b".repeat(64),
      descriptorSha256: "c".repeat(64), filesSha256: "d".repeat(64), overlaySha256: "e".repeat(64)}},
    windowsLlvmResult = {workspace, root: llvmPaths.root, output: llvmPaths.output, descriptor: llvmPaths.descriptor,
      provenance: llvmPaths.provenance, descriptorSha256: "f".repeat(64), provenanceSha256: "1".repeat(64)},
    windowsJitResources = {workspace, root: resourcePaths.root, target: "x86_64-w64-windows-gnu",
      resources: {root: path.join(resourcePaths.root, "resources"), manifestSha256: "2".repeat(64)},
      profile: {sha256: "3".repeat(64)}, emitter: {receiptSha256: "4".repeat(64)}, linux: {manifestSha256: "5".repeat(64)}},
    windowsJitResult = {workspace, root: jitPaths.root, descriptor: path.join(jitPaths.root, "descriptor.json"), descriptorSha256: "6".repeat(64),
      config: path.join(jitPaths.root, "config.json"), configSha256: "7".repeat(64), toolchain: path.join(jitPaths.root, "toolchain"),
      compilerSdk: path.join(jitPaths.root, "compiler-sdk"), compilerResultSha256: "8".repeat(64), isolation: path.join(jitPaths.root, "isolation"),
      isolationHandoffSha256: "9".repeat(64), provenance: path.join(jitPaths.root, "provenance.json"), jobs: 2, buildOnly: true,
      compiled: true, executed: false, publicationQualified: false, reused: false},
    controllerSupport = {localBuildId: id, io: {root: ioRoot, handoffSha256, admitted: {kind: "io", localBuildId: id}},
      hermes: {root: hermesRoot, handoffSha256, admitted: {kind: "hermes", localBuildId: id}}},
    input = {workspace, tools, windowsHostSdk, windowsLlvm: windowsLlvmResult, windowsJitResources, windowsJitSdk: windowsJitResult,
      controllerSupport};
  for (const directory of [llvmPaths.root, resourcePaths.root, jitPaths.root, jitPaths.isolation, support.hermesSdk, path.dirname(sourceFile),
    path.dirname(windowsHostSdk.root), ioRoot, hermesRoot, path.join(workspace, "dpcpp-cuda-inputs/dpcpp"),
    path.join(workspace, "dpcpp-cuda-inputs/cuda")]) {fs.mkdirSync(directory, {recursive: true});}
  fs.writeFileSync(sourceFile, "{\"fixture\":true}\n"); fs.writeFileSync(path.join(support.hermesSdk, "result.json"), "{\"fixture\":true}\n");
  fs.writeFileSync(path.join(jitPaths.isolation, "result.json"), "{\"isolated\":true}\n");
  /** @param {string} file */
  const readJson = file => file === llvmPaths.descriptor ? {fixture: "llvm-descriptor"} : JSON.parse(fs.readFileSync(file, "utf8")),
    componentResult = /** @type {any} */ ({link: {libraries: [path.join(paths.component, "lib/sample.lib")], wholeArchives: [], objects: []}, files: {}});
  const dependencies = {
    verifyScope() {calls.push({name: "scope", args: []});},
    resolveTools() {calls.push({name: "tools", args: []}); return /** @type {ReturnType<typeof local.localTools>} */ (tools);},
    /** @param {string} root @param {{kind:"io"|"hermes",localBuildId:string,handoffSha256:string}} expected */
    readWindowsHandoff(root, expected) {
      calls.push({name: "handoff", args: [root, expected.kind, expected.localBuildId]});
      return {kind: expected.kind, localBuildId: expected.localBuildId, handoff: path.join(root, "handoff.json"),
        handoffSha256: expected.handoffSha256, files: {}, ...(expected.kind === "io" ? {libraries: {fixture: true}} : {hermes: {fixture: true}})};
    },
    /** @param {any} value @param {any} deps */
    admitGpuInputs(value, deps) {
      calls.push({name: "gpu", args: [value.workspace, value.windowsSourceSha256]});
      assert.equal(deps.resolveTools(), tools);
      const toolRoot = path.join(workspace, "llvm21/sdk/bin"), llvmReadobj = path.join(toolRoot, "llvm-readobj");
      return {workspace, dpcppRoot: path.join(workspace, "dpcpp-cuda-inputs/dpcpp"), cudaRoot: path.join(workspace, "dpcpp-cuda-inputs/cuda"),
        tools: {toolRoot, files: {[llvmReadobj]: "0".repeat(64)}}};
    },
    /** @param {any} value */
    async buildJitSdk(value) {
      calls.push({name: "jit", args: [value.workspace, value.hermes.localBuildId]});
      return windowsJitResult;
    },
    /** @param {string} receipt */
    readLinuxHermes(receipt) {
      calls.push({name: "linuxHermes", args: [receipt]});
      assert.equal(receipt, path.join(support.hermesSdk, "result.json"));
      return {artifacts: {compiler: path.join(support.hermesSdk, "build/bin/shermes")}, files: {}};
    },
    /** @param {any} value */
    composeDescriptor(value) {
      calls.push({name: "compose", args: [value.tools.shermaot, value.tools.llvmReadobj]});
      assert.equal(value.hermesAot.artifacts.compiler, value.tools.shermaot);
      return {version: 1, fixture: "composed", tools: value.tools, allowedImports: value.allowedImports};
    },
    readJson,
    /** @param {string} sources @param {string} descriptor @param {string} dpcpp @param {string} cuda @param {string} output
     * @param {NodeJS.ProcessEnv} env */
    buildSdk(sources, descriptor, dpcpp, cuda, output, env) {
      calls.push({name: "sdk", args: [sources, descriptor, dpcpp, cuda, output, env["MOM_BUILD_JOBS"], env["MOM_AOT_DPCPP_JIT_RECEIPT"]]});
      assert.equal(env["MOM_AOT_BUILD_ONLY"], "1");
      assert.equal(env["PATH"]?.split(":")[0], path.dirname(tools.cmake));
      assert.equal(env["PATH"]?.includes("/unrelated"), false);
      assert.equal(env["LD_LIBRARY_PATH"], [path.join(dpcpp, "lib"), path.resolve(path.dirname(tools.cc), "../lib"),
        "/usr/lib/x86_64-linux-gnu"].join(":"));
      fs.mkdirSync(output); fs.writeFileSync(path.join(output, "result.json"), "{\"sdk\":true}\n");
    },
    /** @param {string} sdk @param {string} descriptor @param {string} cuda @param {string} output @param {NodeJS.ProcessEnv} env */
    prepareAdapters(sdk, descriptor, cuda, output, env) {
      calls.push({name: "adapters", args: [sdk, descriptor, cuda, output]});
      assert.equal(env["MOM_AOT_BUILD_ONLY"], "1");
      fs.mkdirSync(output); fs.writeFileSync(path.join(output, "result.json"), "{\"adapters\":true}\n");
    },
    /** @param {string} producer @param {string} cuda @param {string} output @param {NodeJS.ProcessEnv} env */
    exportComponent(producer, cuda, output, env) {
      calls.push({name: "export", args: [producer, cuda, output]});
      assert.equal(env["MOM_AOT_BUILD_ONLY"], "1");
      fs.mkdirSync(path.join(output, "lib"), {recursive: true}); fs.mkdirSync(path.join(output, "objects"));
      fs.writeFileSync(path.join(output, "component.json"), "{}\n");
      componentResult.files = {[path.join(output, "component.json")]: fileHash(path.join(output, "component.json"))};
      return componentResult;
    },
    /** @param {string} producer @param {string} cuda @param {NodeJS.ProcessEnv} env */
    producerInputs(producer, cuda, env) {
      calls.push({name: "producerInputs", args: [producer, cuda]});
      assert.equal(env["MOM_AOT_BUILD_ONLY"], "1");
      return {producer: {path: producer, sha256: fileHash(producer)}, links: {libraries: ["sample.lib"], wholeArchives: [], objects: []}, files: {}};
    },
    /** @param {string} output @param {any} original */
    matchComponent(output, original) {
      calls.push({name: "match", args: [output, original.producer.path]});
      return componentResult;
    },
  };
  t.after(() => fs.rmSync(base, {recursive: true, force: true}));
  return {base, workspace, paths, support, sourceFile, tools, input, dependencies, calls};
}

/** @param {any} f */
function build(f) {return buildLocalNativeWindowsDpcppSdk(f.input, f.dependencies);}

describe("Local Windows DPC++ SDK coordinator", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("enforces memory scope before inspecting inputs", async () => {
    const calls = /** @type {string[]} */ ([]);
    await assert.rejects(buildLocalNativeWindowsDpcppSdk(null, {verifyScope() {calls.push("scope"); throw new Error("no enforced scope");},
      resolveTools() {calls.push("tools"); return /** @type {ReturnType<typeof local.localTools>} */ ({});}}), /no enforced scope/);
    assert.deepEqual(calls, ["scope"]);
  });

  test("runs the original SDK, adapter and component producers in order", async t => {
    const f = fixture(t);
    Object.assign(f.dependencies, {environment: {PATH: "/unrelated/bin", LD_LIBRARY_PATH: "/unrelated/lib"}});
    const result = await build(f),
      stages = f.calls.filter(row => ["sdk", "adapters", "export"].includes(row.name)).map(row => row.name);
    assert.deepEqual(stages, ["sdk", "adapters", "export"]);
    assert.equal(result.root, f.paths.root); assert.equal(result.reused, false);
    assert.equal(result.jobs, 2); assert.equal(result.buildOnly, true); assert.equal(result.executed, false);
    const expectedDescriptor = {version: 1, fixture: "composed", tools: {shermaot: path.join(f.support.hermesSdk, "build/bin/shermes"),
      llvmReadobj: path.join(f.workspace, "llvm21/sdk/bin/llvm-readobj")},
    allowedImports: ["advapi32.dll", "cfgmgr32.dll", "dbghelp.dll", "kernel32.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll",
      "setupapi.dll", "shell32.dll", "shlwapi.dll", "version.dll"]};
    assert.equal(fs.readFileSync(result.descriptor, "utf8"), JSON.stringify(expectedDescriptor, null, 2) + "\n");
    assert.equal(f.calls[0]?.name, "scope");
    assert.equal(f.calls.find(row => row.name === "sdk")?.args[5], "2");
    assert.match(f.calls.find(row => row.name === "sdk")?.args[6] || "", /windows-jit-sdk\/isolation\/result\.json$/);
    assert.equal(f.calls.filter(row => row.name === "handoff").length, 4);
  });

  test("warm admission joins the installed component to the current producer without rebuilding", async t => {
    const f = fixture(t); await build(f); f.calls.length = 0;
    const result = await build(f), stages = f.calls.filter(row => ["sdk", "adapters", "export"].includes(row.name));
    assert.deepEqual(stages, []); assert.equal(result.reused, true);
    assert.equal(f.calls.filter(row => row.name === "producerInputs").length, 1);
    assert.equal(f.calls.filter(row => row.name === "match").length, 1);
    assert.equal(f.calls.filter(row => row.name === "handoff").length, 4);
  });

  test("preserves and rejects incomplete or unknown workspace output", async t => {
    const f = fixture(t); fs.mkdirSync(f.paths.root); fs.writeFileSync(f.paths.descriptor, "partial\n");
    await assert.rejects(build(f), /incomplete or contains unknown entries/);
    assert.deepEqual(fs.readdirSync(f.paths.root), ["descriptor.json"]);
    assert.equal(f.calls.filter(row => row.name === "gpu").length, 0);

    const g = fixture(t); await build(g); fs.writeFileSync(path.join(g.paths.root, "unknown"), "preserve\n");
    await assert.rejects(build(g), /unknown entries/);
    assert.equal(fs.readFileSync(path.join(g.paths.root, "unknown"), "utf8"), "preserve\n");
  });

  test("rejects descriptor drift and changed inputs after production", async t => {
    const f = fixture(t); await build(f); fs.chmodSync(f.paths.descriptor, 0o600); fs.appendFileSync(f.paths.descriptor, "stale\n");
    await assert.rejects(build(f), /descriptor is stale/);

    const g = fixture(t), originalBuildSdk = g.dependencies.buildSdk;
    g.dependencies.buildSdk = (...args) => {originalBuildSdk(...args); fs.appendFileSync(g.sourceFile, "changed\n");};
    await assert.rejects(build(g), /changed during production/);
    assert.equal(fs.existsSync(g.paths.root), true);
  });

  test("rejects handoff identity and supplied JIT result mismatches", async t => {
    const f = fixture(t); f.input.controllerSupport.hermes.admitted.localBuildId = "12345678-1234-4234-9234-123456789abd";
    await assert.rejects(build(f), /differs from its export identity/);

    const g = fixture(t); g.input.windowsJitSdk = {...g.input.windowsJitSdk, compilerResultSha256: "0".repeat(64)};
    await assert.rejects(build(g), /JIT handoff differs/);

    const h = fixture(t); h.input.windowsJitSdk = /** @type {any} */ ({...h.input.windowsJitSdk, extra: true});
    await assert.rejects(build(h), /unknown or missing fields/);
  });
});
