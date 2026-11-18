"use strict";

const {describe, test} = require("node:test"), assert = require("node:assert/strict"),
  fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const local = require("../scripts/build-local-native-sdk"), jitStages = require("../scripts/native-jit-stages"),
  jitProducer = require("../scripts/build-native-jit-sdk"),
  isolationTools = require("../scripts/isolate-native-windows-jit"),
  {fileHash, buildFileInputs, readBuildJson} = require("../scripts/build-source-inputs"),
  {localWindowsJitSdkPaths, localWindowsJitCompilerConfig, expectedIsolationHandoff, buildLocalNativeWindowsJitSdk} =
    require("../scripts/build-local-native-windows-jit-sdk");

/** Inert receipt/tool fixtures only. No compiler, archive utility, SDK build, download or GPU is invoked.
 * @param {import('node:test').TestContext} t @param {{changeAdmission?:boolean}} [options] */
function fixture(t, options = {}) {
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-windows-jit-sdk-")), workspace = path.join(parent, "workspace");
  local.admitWorkspace(workspace, fs.realpathSync(path.resolve(__dirname, "..")));
  const paths = localWindowsJitSdkPaths(workspace), support = /** @type {ReturnType<typeof local.supportPaths>} */ ({workspace,
      compilerSources: path.join(workspace, "compiler-sources"), dpcppCudaRoot: path.join(workspace, "dpcpp-cuda-inputs")}),
    sourceDir = path.join(support.compilerSources, "sources.json"),
    sdk = path.join(support.dpcppCudaRoot, "dpcpp"), resources = path.join(workspace, "windows-jit-resources", "resources"),
    nativeTools = path.join(workspace, "linux-jit-resources", "bootstrap", "build", "bin"),
    tools = /** @type {any} */ ({}), manifestTools = /** @type {any} */ ({}), calls = /** @type {Array<{name:string,args:any[]}>} */ ([]);
  for (const dir of [support.compilerSources, sdk, resources, nativeTools, path.join(workspace, "windows-llvm-sdk"), path.join(workspace, "windows-jit-resources")]) {
    fs.mkdirSync(dir, {recursive: true});
  }
  /** @param {string} file @param {string} [contents] */
  function put(file, contents = path.basename(file)) {
    fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, contents); fs.chmodSync(file, 0o700); return file;
  }
  const toolFiles = {git: path.join(parent, "git"), cmake: path.join(parent, "cmake"), ninja: path.join(parent, "ninja"),
    systemdRun: path.join(parent, "systemd-run"), cc: path.join(parent, "icx"), cxx: path.join(parent, "icpx")};
  for (const [name, file] of Object.entries(toolFiles)) {tools[name] = put(file);}
  const llvmFiles = {llvmAr: path.join(parent, "llvm-ar"), llvmNm: path.join(parent, "llvm-nm"), llvmObjcopy: path.join(parent, "llvm-objcopy")};
  for (const [name, file] of Object.entries(llvmFiles)) {manifestTools[name] = put(file);}
  const clangTableGen = put(path.join(nativeTools, "clang-tblgen")), llvmTableGen = put(path.join(nativeTools, "llvm-tblgen")),
    descriptorFile = put(path.join(workspace, "windows-llvm-sdk", "descriptor.json"), "{}\n"),
    windowsLlvm = {workspace, root: path.dirname(descriptorFile), output: path.join(path.dirname(descriptorFile), "output"),
      descriptor: descriptorFile, provenance: path.join(path.dirname(descriptorFile), "provenance.json"), descriptorSha256: "1".repeat(64),
      provenanceSha256: "2".repeat(64)}, hermes = {root: put(path.join(workspace, "hermes-sdk")),
      localBuildId: "12345678-1234-4234-9234-123456789abc", handoffSha256: "3".repeat(64)},
    windowsHostSdk = {root: path.join(workspace, "windows-host-sdk", "sdk"), pins: {resultSha256: "4".repeat(64),
      descriptorSha256: "5".repeat(64), filesSha256: "6".repeat(64), overlaySha256: "7".repeat(64)}},
    windowsJitResources = {workspace, root: path.join(workspace, "windows-jit-resources"), target: "x86_64-w64-windows-gnu",
      resources: {root: resources, manifestSha256: "8".repeat(64)},
      profile: {sha256: "9".repeat(64)}, emitter: {receiptSha256: "a".repeat(64)}, linux: {manifestSha256: "b".repeat(64)}},
    input = {workspace, tools, hermes, windowsHostSdk, windowsLlvm, windowsJitResources};
  const context = {workspace, support, paths, windowsLlvm, windowsResources: windowsJitResources, windowsHostSdk, hermes,
    descriptor: {}, descriptorFile, hostCompiler: tools.cc, hostCompilers: {cc: tools.cc, cxx: tools.cc}, sdk, nativeTools,
    sources: sourceDir, resources, isolationTools: manifestTools,
    toolManifest: {version: 1, tools: manifestTools}, tools};
  let admissionCount = 0;
  const dependencies = {
    verifyScope() {calls.push({name: "scope", args: []});},
    resolveTools() {calls.push({name: "tools", args: []}); return tools;},
    admitInputs() {
      admissionCount++;
      calls.push({name: "admit", args: [admissionCount]});
      if (options.changeAdmission && admissionCount > 1) {
        return {...context, windowsResources: {...windowsJitResources,
          resources: {...windowsJitResources.resources, manifestSha256: "c".repeat(64)}}};
      }
      return context;
    },
    /** @param {string} _descriptor @param {string} _sdk @param {string} _nativeTools @param {string} output */
    prepareToolchain(_descriptor, _sdk, _nativeTools, output) {
      calls.push({name: "prepare", args: [output]});
      fs.mkdirSync(output, {recursive: true});
      const toolchain = path.join(output, "toolchain.cmake"); fs.writeFileSync(toolchain, "fixture toolchain\n");
      const result = {version: 1, toolchain, inputs: {[toolchain]: fileHash(toolchain)}, buildOnly: true,
        compiled: false, executed: false, publicationQualified: false};
      fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n");
      return result;
    },
    /** @param {string} _descriptor @param {string} _sdk @param {string} _nativeTools @param {string} output */
    verifyToolchain(_descriptor, _sdk, _nativeTools, output) {return readBuildJson(path.join(output, "result.json"));},
    /** @param {any} config @param {string} output @param {NodeJS.ProcessEnv} _environment @param {string} phase */
    async buildPhase(config, output, _environment, phase) {
      calls.push({name: "phase", args: [phase, config.jobs]});
      if (phase === "configure") {
        fs.mkdirSync(output); fs.mkdirSync(path.join(output, "build"));
        for (const name of ["before.json", "configured-tools.json", "commands.log", "commands.stderr.log", "stage-configure.json", "configure.log"]) {
          fs.writeFileSync(path.join(output, name), name === "before.json" ? JSON.stringify({config}, null, 2) + "\n" : "fixture\n");
        }
        return;
      }
      for (const name of ["stage-" + phase + ".json", phase + ".log", "dependencies-" + phase + ".log"]) {
        fs.writeFileSync(path.join(output, name), "fixture\n");
      }
      if (phase === "link") {
        const archive = path.join(output, "build", "sycl-jit.lib"); put(archive, "original archive fixture\n");
        fs.writeFileSync(path.join(output, "result.json"), "{\"fixture\":true}\n");
      }
    },
    /** @param {string} output @param {string} expectedSha @param {string} toolsFile */
    inspect(output, expectedSha, toolsFile) {
      calls.push({name: "inspect", args: [output, expectedSha]});
      assert.equal(expectedSha, fileHash(path.join(output, "result.json")));
      const archive = path.join(output, "build", "sycl-jit.lib"), resultPath = path.join(output, "result.json"),
        commands = path.join(output, "commands.log"), toolMap = readBuildJson(toolsFile).tools, hashes = isolationTools.hashTools(toolMap),
        evidence = {"commands.log": {path: commands, sha256: fileHash(commands)}};
      return {sdkRoot: output, resultPath, resultSha256: expectedSha, result: {proof: {path: path.join(output, "build", "proof.exe"), sha256: "d".repeat(64)},
        libraries: [{path: archive, sha256: fileHash(archive)}], recipe: jitProducer.jitSdkRecipe()},
      evidence, archives: [archive], linkLine: "fixture link", linkLineSha256: "1".repeat(64), recipeSha256: "2".repeat(64),
      inputHashes: buildFileInputs([tools.cmake]), tools: toolMap, toolsManifestSha256: fileHash(toolsFile), toolHashes: hashes,
      protectedPaths: [output, archive, toolsFile, ...Object.values(toolMap)]};
    },
    /** @param {any} prior @param {string} _toolsFile @param {string} output */
    runIsolation(prior, _toolsFile, output) {
      calls.push({name: "isolate", args: [output]}); fs.mkdirSync(output);
      const archive = path.join(output, path.basename(prior.archives[0] || "archive.lib")); put(archive, "isolated archive fixture\n");
      const map = path.join(output, "rename.map"); fs.writeFileSync(map, "old new\n");
      const sourceHashes = Object.fromEntries(["native-coff-archive.js", "native-compiler-isolation.js", "build-source-inputs.js"]
        .map(name => {const file = require.resolve("../scripts/" + name); return [file, fileHash(file)];}));
      const transformed = [{input: prior.archives[0], inputSha256: prior.result.libraries[0]?.sha256, output: archive,
          sha256: fileHash(archive), rewrittenMembers: 0, members: 1, checkedMemberSha256: "3".repeat(64)}],
        result = {version: 1, prefix: "mom_dpcpp23_", entryPoints: require("../scripts/native-compiler-isolation").dpcppEntries,
          symbols: 1, map, mapSha256: fileHash(map), transformed,
          toolHashes: Object.fromEntries(Object.values(prior.tools).map(file => [file, fileHash(file)])), sourceHashes,
          checks: {memberOrderPreserved: true, nonDirectivePayloadsPreserved: true, mappedDirectivesPreserved: true,
            symbolOrderPreserved: true, auxiliaryRecordsPreserved: true, relocationsPreserved: true},
          buildOnly: true, executed: false, publicationQualified: false};
      fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n");
      const labels = ["tool-0", "tool-1", "tool-2", "definitions-0", "isolate-0"];
      for (const label of labels) {fs.writeFileSync(path.join(output, label + ".command.json"), "{}\n"); fs.writeFileSync(path.join(output, label + ".log"), "fixture\n");}
      fs.writeFileSync(path.join(output, "handoff.json"), JSON.stringify(expectedIsolationHandoff(paths, prior, result), null, 2) + "\n");
    },
  };
  t.after(() => fs.rmSync(parent, {recursive: true, force: true}));
  return {parent, workspace, paths, support, tools, input, context, calls, dependencies, clangTableGen, llvmTableGen};
}

/** @param {unknown} input @param {any} dependencies */
function buildLocal(input, dependencies) {return buildLocalNativeWindowsJitSdk(/** @type {any} */ (input), dependencies);}

describe("Local Windows JIT SDK coordinator", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("guards the memory scope before inspecting inputs", async () => {
    const calls = /** @type {string[]} */ ([]);
    await assert.rejects(buildLocal(null, {verifyScope() {calls.push("scope");}, resolveTools() {calls.push("tools"); return {};}}), /marked workspace/);
    assert.deepEqual(calls, ["scope"]);
  });

  test("uses the public Windows target config, every maintained phase, and warm-readmits without restamping", async t => {
    const f = fixture(t), first = await buildLocal(f.input, f.dependencies), before = f.calls.length,
      config = readBuildJson(f.paths.config), second = await buildLocal(f.input, f.dependencies);
    assert.equal(config.target, "windows"); assert.equal(config.jobs, 2);
    assert.equal(config.sdk, f.context.sdk); assert.equal(config.resources, f.context.resources);
    assert.equal(config.tools.toolchain, path.join(f.paths.toolchain, "toolchain.cmake"));
    assert.deepEqual(f.calls.filter(call => call.name === "phase").map(call => call.args[0]), ["configure", ...jitStages.jitBuildStages]);
    assert.deepEqual(f.calls.slice(before).filter(call => ["phase", "prepare", "isolate"].includes(call.name)), []);
    assert.equal(first.reused, false); assert.equal(second.reused, true); assert.equal(first.compilerResultSha256, second.compilerResultSha256);
    assert.equal(fs.existsSync(path.join(f.paths.output, "result.json")), true);
    assert.equal(Object.hasOwn(second, "runAttempt"), false);
    assert.deepEqual(fs.readdirSync(f.paths.root).sort(), ["config.json", "descriptor.json", "compiler-sdk", "isolation", "llvm-tools.json", "provenance.json", "toolchain"].sort());
  });

  test("preserves unknown, partial, changed-input, and stale-result state", async t => {
    const f = fixture(t); await buildLocal(f.input, f.dependencies);
    fs.writeFileSync(path.join(f.paths.root, "unknown"), "keep\n");
    await assert.rejects(buildLocal(f.input, f.dependencies), /contains unknown entries/);
    assert.equal(fs.readFileSync(path.join(f.paths.root, "unknown"), "utf8"), "keep\n");

    const partial = fixture(t), stop = {...partial.dependencies, async buildPhase() {throw new Error("fixture interruption");}};
    await assert.rejects(buildLocal(partial.input, stop), /fixture interruption/);
    const partialContents = fs.readdirSync(partial.paths.root).sort();
    await assert.rejects(buildLocal(partial.input, partial.dependencies), /incomplete or contains unknown entries/);
    assert.deepEqual(fs.readdirSync(partial.paths.root).sort(), partialContents);

    const stale = fixture(t), built = await buildLocal(stale.input, stale.dependencies), resultFile = path.join(built.compilerSdk, "result.json");
    fs.appendFileSync(resultFile, "stale\n");
    await assert.rejects(buildLocal(stale.input, stale.dependencies), /changed after its local handoff/);
    assert.equal(fs.readFileSync(resultFile, "utf8").endsWith("stale\n"), true);

    const drift = fixture(t, {changeAdmission: true}), drifted = await buildLocal(drift.input, drift.dependencies).catch(error => error);
    assert.match(drifted.message, /inputs changed during production/);
    assert.equal(fs.existsSync(path.join(drift.paths.root, "compiler-sdk", "result.json")), true);
  });

  test("rejects malformed target config and changed toolchain inputs", t => {
    const f = fixture(t), config = /** @type {any} */ ({version: 1, target: "linux", sources: path.join(f.support.compilerSources, "sources.json"),
      sdk: f.context.sdk, resources: f.context.resources, tools: {cmake: f.tools.cmake, ninja: f.tools.ninja, toolchain: path.join(f.paths.toolchain, "toolchain.cmake")},
      inputs: [path.join(f.paths.toolchain, "toolchain.cmake")], jobs: 2});
    assert.throws(() => localWindowsJitCompilerConfig(f.support, config), /config differs/);
    assert.notEqual(fileHash(f.clangTableGen), fileHash(f.llvmTableGen));
  });
});
