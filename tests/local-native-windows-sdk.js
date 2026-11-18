"use strict";

const {describe, test} = require("node:test"), assert = require("node:assert/strict"),
  fs = require("node:fs"), os = require("node:os"), path = require("node:path"), {createHash} = require("node:crypto");
const local = require("../scripts/build-local-native-sdk"), {fileHash} = require("../scripts/build-source-inputs"),
  {buildLocalNativeWindowsSdk, localWindowsSdkStagePaths, profiles} = require("../scripts/build-local-native-windows-sdk");
const repo = fs.realpathSync(path.resolve(__dirname, ".."));

/** Inert public-boundary fixtures only; no downloads, compilers, GPU, VM or full application build run.
 * @param {import("node:test").TestContext} t @param {{failGpu?:boolean,occupied?:boolean}} [options] */
function fixture(t, options = {}) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-windows-sdk-")), workspace = path.join(base, "workspace");
  local.admitWorkspace(workspace, repo);
  const support = local.supportPaths(workspace), output = path.join(workspace, "windows-sdk-stage"), paths = localWindowsSdkStagePaths(workspace, output),
    tools = {cc: "/fixture/icx", cmake: "/fixture/cmake", cxx: "/fixture/icpx", git: "/fixture/git", ninja: "/fixture/ninja", systemdRun: "/fixture/systemd-run"},
    linuxSdk = {workspace, compilerFetch: support.compilerFetch, compilerSources: support.compilerSources, buildInputs: support.buildInputs,
      llvmSdk: support.llvmSdk, jobs: 2}, id = "12345678-1234-4234-9234-123456789abc", coreArchive = path.join(base, "core/cpu-core.lib"),
    hostRoot = path.join(workspace, "windows-host-sdk/sdk"), hostPins = {descriptorSha256: "b".repeat(64), filesSha256: "c".repeat(64),
      overlaySha256: "d".repeat(64), resultSha256: "e".repeat(64)}, windowsHostSdk = {workspace, root: hostRoot, pins: hostPins},
    ioRoot = path.join(base, "controller/io"), hermesRoot = path.join(base, "controller/hermes"),
    controllerSupport = {localBuildId: id, io: {root: ioRoot, handoffSha256: "f".repeat(64), admitted: {kind: "io", localBuildId: id}},
      hermes: {root: hermesRoot, handoffSha256: "1".repeat(64), admitted: {kind: "hermes", localBuildId: id}}},
    windowsLlvm = {workspace, output: path.join(workspace, "windows-llvm-sdk/output"), descriptor: path.join(workspace, "windows-llvm-sdk/descriptor.json")},
    windowsJitResources = {workspace, target: "x86_64-w64-windows-gnu"},
    windowsJitSdk = {workspace, jobs: 2}, windowsDpcppSdk = {workspace, jobs: 2, component: {root: path.join(workspace, "windows-dpcpp-sdk/component")}},
    hostComponentRoot = path.join(workspace, "windows-host-component/component"),
    windowsHostComponent = {workspace, jobs: 2, componentRoot: hostComponentRoot,
      amdhip64: {amdhip64: path.join(workspace, "windows-host-component/amdhip/amdhip64.lib")}},
    coreDir = path.dirname(coreArchive), calls = /** @type {Array<{name:string,args:any[]}>} */ ([]),
    input = {workspace, tools, linuxSdk, controllerSupport, coreArchive, output};
  fs.mkdirSync(coreDir); fs.writeFileSync(coreArchive, "fixture core bytes\n"); fs.writeFileSync(path.join(coreDir, "inputs.json"), "{}\n");
  const llvmDescriptor = {version: 1, tools: {clangCl: "/fixture/icx-cl", lldLink: "/fixture/lld-link", llvmNm: "/fixture/llvm-nm",
    clangResourceInclude: "/fixture/clang-resource"},
  windowsSdk: {sourceRoot: "/fixture/windows-sdk/source", overlay: "/fixture/windows-sdk/overlay.yaml", includeRoots: ["/fixture/windows-sdk/include"],
    libraryRoots: ["/fixture/windows-sdk/lib"], msvcLibraries: ["/fixture/windows-sdk/lib/msvcrt.lib"],
    ucrtLibraries: ["/fixture/windows-sdk/lib/ucrt.lib"], importLibraries: ["/fixture/windows-sdk/lib/kernel32.lib"]}};
  const hermesCompiler = path.join(support.hermesSdk, "build/bin/shermes"),
    admittedController = {id, io: {libraries: {sourceInclude: "/fixture/openssl/include", ssl: "/fixture/openssl/libssl.a",
      crypto: "/fixture/openssl/libcrypto.a", libuv: "/fixture/io/libuv.lib", zlib: "/fixture/io/zlib.lib"}},
    hermes: {hermes: {source: "/fixture/hermes/source", build: "/fixture/hermes/build"},
      intelRuntime: {libRoot: "/fixture/windows-sdk/lib", libraries: {msvcrt: "/fixture/windows-sdk/lib/msvcrt.lib"}}},
    hermesInput: {root: hermesRoot, localBuildId: id, handoffSha256: controllerSupport.hermes.handoffSha256}},
    admittedGpu = {workspace, cudaRoot: "/fixture/cuda", cudaImages: {include: "/fixture/cuda-images/cuda-images.inc"},
      hip: {include: "/fixture/hip/include"}, tools: {toolRoot: "/fixture/llvm/bin"}},
    makeProfiles = () => Object.fromEntries(profiles.map(profile => {
      const config = {version: 1, profile}, configBytes = Buffer.from(JSON.stringify(config) + "\n");
      return [profile, {config, configBytes, configSha256: createHash("sha256").update(configBytes).digest("hex")}];
    }));
  if (options.occupied) {fs.mkdirSync(output); fs.writeFileSync(path.join(output, "operator-data"), "keep me\n");}
  const dependencies = /** @type {any} */ ({
    // Fixtures assert the local default; a CI job's own MOM_BUILD_JOBS must not leak in.
    environment: {MOM_BUILD_JOBS: "2"},
    verifyScope() {calls.push({name: "scope", args: []});},
    resolveTools(/** @type {NodeJS.ProcessEnv} */ environment) {calls.push({name: "tools", args: [environment["MOM_BUILD_JOBS"]]}); return tools;},
    prepareWindowsSources(/** @type {string} */ root) {
      calls.push({name: "sources", args: [root]}); return {workspace, root: path.join(workspace, "windows-sources"),
        receipt: path.join(workspace, "windows-sources/sources.json"), receiptSha256: "2".repeat(64)};
    },
    async buildHostSdk(/** @type {any} */ args, /** @type {any} */ optionsArg) {calls.push({name: "host-sdk", args: [args.workspace, optionsArg.environment["MOM_BUILD_JOBS"]]}); return windowsHostSdk;},
    admitHostSdk(/** @type {string} */ root, /** @type {any} */ pins) {calls.push({name: "host-admit", args: [root]}); return {root, pins};},
    admitControllerSupport(/** @type {any} */ value) {calls.push({name: "controller", args: [value.localBuildId]}); return admittedController;},
    buildWindowsLlvm(/** @type {any} */ args) {calls.push({name: "llvm", args: [args.hermes.localBuildId, args.windowsHostSdk.root]}); return windowsLlvm;},
    buildJitResources(/** @type {any} */ sdk, /** @type {any} */ selectedTools) {calls.push({name: "jit-resources", args: [sdk.workspace, selectedTools.cc]}); return windowsJitResources;},
    async buildJitSdk(/** @type {any} */ args) {calls.push({name: "jit-sdk", args: [args.hermes.localBuildId]}); return windowsJitSdk;},
    async buildDpcppSdk(/** @type {any} */ args) {calls.push({name: "dpcpp", args: [args.controllerSupport.localBuildId]}); return windowsDpcppSdk;},
    async buildHostComponent(/** @type {any} */ args) {calls.push({name: "host-component", args: [args.hermes.localBuildId]}); return windowsHostComponent;},
    composeGpuConfigs(/** @type {any} */ args) {
      calls.push({name: "gpu-configs", args: [args.windowsSourceSha256]});
      return {profiles: makeProfiles(), admitted: admittedGpu, buildOnly: true, executed: false, publicationQualified: false};
    },
    buildGpu(/** @type {string} */ configFile, /** @type {string} */ outputDir, /** @type {NodeJS.ProcessEnv} */ environment) {
      const config = JSON.parse(fs.readFileSync(configFile, "utf8")), profile = config.profile;
      calls.push({name: "gpu", args: [profile, environment["MOM_BUILD_JOBS"]]});
      fs.mkdirSync(outputDir, {mode: 0o700});
      if (options.failGpu) {fs.writeFileSync(path.join(outputDir, "partial.log"), "preserve partial\n"); throw new Error("fixture GPU failure");}
      const receipt = {version: 1, profile, factory: "mom_native_gpu_" + profile.replaceAll("-", "_"), configFile,
        configSha256: fileHash(configFile), sourceState: {}, sdkInputs: {}, objects: []};
      fs.writeFileSync(path.join(outputDir, "result.json"), JSON.stringify(receipt, null, 2) + "\n"); return receipt;
    },
    readLinuxHermes(/** @type {string} */ file) {calls.push({name: "linux-hermes", args: [file]}); return {artifacts: {compiler: hermesCompiler}, files: {[hermesCompiler]: "3".repeat(64)}};},
    readJson(/** @type {string} */ file) {return file === windowsLlvm.descriptor ? llvmDescriptor : JSON.parse(fs.readFileSync(file, "utf8"));},
    admitApplication(/** @type {any} */ app) {calls.push({name: "application", args: [app.core, app.gpuReceipts.length]}); return {core: app.core, graphs: profiles.map(profile => ({profile})), files: {}};},
    reviewNotices() {calls.push({name: "notices", args: []}); return {notice: {sha256: fileHash(path.join(repo, "THIRD-PARTY-NOTICES.txt"))}};},
  });
  t.after(() => fs.rmSync(base, {recursive: true, force: true}));
  return {base, workspace, support, output, paths, tools, linuxSdk, input, dependencies, calls, coreArchive, admittedGpu};
}

describe("Local Windows SDK/application graph coordinator", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("enforces memory scope before inspecting input or resolving tools", async () => {
    const calls = /** @type {string[]} */ ([]);
    await assert.rejects(buildLocalNativeWindowsSdk(null, {verifyScope() {calls.push("scope"); throw new Error("scope absent");},
      resolveTools() {calls.push("tools"); return {git: "", cmake: "", ninja: "", systemdRun: "", cc: "", cxx: ""};}}), /scope absent/);
    assert.deepEqual(calls, ["scope"]);
  });

  test("runs local adapters in order, builds all five profiles and writes a reader-validated release descriptor", async t => {
    const f = fixture(t), result = await buildLocalNativeWindowsSdk(f.input, f.dependencies), stages = f.calls
      .filter(row => ["host-sdk", "llvm", "jit-resources", "jit-sdk", "dpcpp", "host-component", "gpu"].includes(row.name))
      .map(row => row.name);
    assert.deepEqual(stages, ["host-sdk", "llvm", "jit-resources", "jit-sdk", "dpcpp", "host-component", ...profiles.map(() => "gpu")]);
    assert.deepEqual(f.calls.filter(row => row.name === "gpu").map(row => row.args[0]), profiles);
    assert.ok(f.calls.every(row => row.name !== "gpu" || row.args[1] === "2"));
    assert.equal(result.workspace, f.workspace); assert.equal(result.root, f.output); assert.equal(result.jobs, 2);
    assert.equal(result.buildOnly, true); assert.equal(result.compiled, true); assert.equal(result.executed, false);
    assert.equal(Object.hasOwn(result, "runAttempt"), false); assert.equal(result.configFile, f.paths.configFile);
    const config = JSON.parse(fs.readFileSync(result.configFile, "utf8"));
    assert.equal(config.notices.path, path.join(repo, "THIRD-PARTY-NOTICES.txt"));
    assert.equal(config.notices.sha256, fileHash(config.notices.path));
    assert.equal(config.application.core, f.coreArchive);
    assert.deepEqual(config.application.gpuReceipts.map((/** @type {string} */ file) => path.basename(path.dirname(file))), profiles);
    assert.deepEqual(fs.readdirSync(f.paths.gpuConfigs).sort(), profiles.map(profile => profile + ".json").sort());
    assert.deepEqual(fs.readdirSync(f.paths.gpuOutputs).sort(), [...profiles].sort());
    assert.equal(f.calls.filter(row => row.name === "host-admit").length, 2);
    assert.equal(f.calls.filter(row => row.name === "controller").length, 2);
    assert.equal(f.calls.filter(row => row.name === "sources").length, 2);
  });

  test("preserves an occupied stage root and rejects it before producers", async t => {
    const f = fixture(t, {occupied: true});
    await assert.rejects(buildLocalNativeWindowsSdk(f.input, f.dependencies), /fresh workspace-owned directory/);
    assert.equal(fs.readFileSync(path.join(f.output, "operator-data"), "utf8"), "keep me\n");
    assert.deepEqual(f.calls.map(row => row.name), ["scope"]);
  });

  test("preserves a partial profile build and never reuses that unknown stage root", async t => {
    const f = fixture(t, {failGpu: true});
    await assert.rejects(buildLocalNativeWindowsSdk(f.input, f.dependencies), /fixture GPU failure/);
    const profile = profiles[0];
    assert.ok(profile);
    const partial = path.join(f.paths.gpuOutputs, profile, "partial.log");
    assert.equal(fs.readFileSync(partial, "utf8"), "preserve partial\n");
    await assert.rejects(buildLocalNativeWindowsSdk(f.input, f.dependencies), /fresh workspace-owned directory/);
    assert.equal(fs.readFileSync(partial, "utf8"), "preserve partial\n");
  });
});
