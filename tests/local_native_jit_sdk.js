"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"),
  path = require("node:path"), os = require("node:os");
const local = require("../scripts/build-local-native-sdk");
const {buildLocalLinuxJitSdk, localJitSdkPaths, localJitSdkContextFromWorkspace, localJitSdkContext,
  localJitSdkConfig, localJitSdkPhasePrefix, verifyLocalStaticSdkRecipe, verifyLocalStaticDependencies, runLocalJitSdkChild} =
  require("../scripts/build-local-native-jit-sdk");
const {linuxStaticSdkRecipe} = require("../scripts/build-native-dpcpp-sdk");
const {buildFileInputs} = require("../scripts/build-source-inputs");

const repo = fs.realpathSync(path.resolve(__dirname, ".."));

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-native-jit-sdk-test-")), workspace = path.join(root, "native-sdk");
  local.admitWorkspace(workspace, repo);
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, workspace, context: localJitSdkContextFromWorkspace(workspace)};
}

/** @param {string} file */
function writeFile(file) {fs.writeFileSync(file, "fixture\n");}

/** @param {string} output @param {string} phase */
function addPhase(output, phase) {
  writeFile(path.join(output, phase === "configure" ? "stage-configure.json" : "stage-" + phase + ".json"));
  writeFile(path.join(output, phase + ".log"));
  if (phase !== "configure") {writeFile(path.join(output, "dependencies-" + phase + ".log"));}
}

/** @param {string} output */
function addConfiguredTree(output) {
  fs.mkdirSync(output); fs.mkdirSync(path.join(output, "build"));
  writeFile(path.join(output, "before.json")); writeFile(path.join(output, "configured-tools.json"));
  writeFile(path.join(output, "commands.log")); writeFile(path.join(output, "commands.stderr.log"));
  addPhase(output, "configure");
}

/** @param {string} root */
function fakeTools(root) {
  const bin = path.join(root, "tools"); fs.mkdirSync(bin);
  const executable = (/** @type {string} */ name) => {
    const file = path.join(bin, name); writeFile(file); fs.chmodSync(file, 0o755); return file;
  };
  return {git: path.join(bin, "git"), systemdRun: path.join(bin, "systemd-run"), cmake: executable("cmake"), ninja: executable("ninja"),
    cc: executable("icx"), cxx: executable("icpx")};
}

test.describe("Local Linux JIT SDK build and admission", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("uses stable workspace outputs and accepts only exact prior-stage path results", t => {
    const f = fixture(t), selected = localJitSdkContext(f.context), paths = localJitSdkPaths(f.workspace);
    assert.deepEqual(selected.paths, paths);
    assert.deepEqual(localJitSdkPaths(f.workspace), paths, "local state has no per-run generation");
    assert.equal(paths.configFile, path.join(f.workspace, "compiler-sdk.json"));
    assert.equal(paths.staticComponent, paths.staticOutput + ".component");

    const tools = fakeTools(f.root), config = localJitSdkConfig(selected, tools);
    assert.deepEqual(config, {version: 1, target: "linux", sources: path.join(f.workspace, "compiler-sources/sources.json"),
      sdk: path.join(f.workspace, "dpcpp-cuda-inputs/dpcpp"), resources: path.join(f.workspace, "linux-jit-resources/resources"),
      tools: {cmake: tools.cmake, ninja: tools.ninja, cc: tools.cc, cxx: tools.cxx}, jobs: 2});
    assert.throws(() => localJitSdkContext({...f.context, inputs: {...f.context.inputs, dpcppRoot: path.join(f.root, "other")}}),
      /exact admitted workspace paths/);
    assert.throws(() => localJitSdkContext({...f.context, resources: {...f.context.resources, target: "x86_64-w64-windows-gnu"}}),
      /exact admitted workspace paths/);
  });

  test("recognizes only a contiguous public-producer phase prefix and preserves interrupted evidence", t => {
    const f = fixture(t), output = localJitSdkPaths(f.workspace).output;
    addConfiguredTree(output);
    assert.equal(localJitSdkPhasePrefix(output), "configure");
    addPhase(output, "support");
    assert.equal(localJitSdkPhasePrefix(output), "support");

    const gap = path.join(f.workspace, "phase-gap"); addConfiguredTree(gap); addPhase(gap, "targets");
    assert.throws(() => localJitSdkPhasePrefix(gap), /contiguous prefix/);
    assert.equal(fs.readFileSync(path.join(gap, "stage-targets.json"), "utf8"), "fixture\n");

    const interrupted = path.join(f.workspace, "phase-interrupted"); addConfiguredTree(interrupted);
    writeFile(path.join(interrupted, "support.log"));
    assert.throws(() => localJitSdkPhasePrefix(interrupted), /interrupted logs/);
    assert.equal(fs.readFileSync(path.join(interrupted, "support.log"), "utf8"), "fixture\n");

    const premature = path.join(f.workspace, "phase-premature-result"); addConfiguredTree(premature);
    writeFile(path.join(premature, "result.json"));
    assert.throws(() => localJitSdkPhasePrefix(premature), /before the final link/);
  });

  test("static SDK warm admission detects edits against the public producer's current recipe", () => {
    const current = linuxStaticSdkRecipe();
    assert.doesNotThrow(() => verifyLocalStaticSdkRecipe(current));
    const graph = {...current.graph}, [first] = Object.keys(graph);
    assert.ok(first);
    graph[first || ""] = "0".repeat(64);
    assert.throws(() => verifyLocalStaticSdkRecipe({...current, graph}), /source inputs are stale/);
  });

  test("static SDK dependency admission uses the producer path-hash map and rehashes current headers", t => {
    const f = fixture(t), header = path.join(f.root, "include", "synthetic.h");
    fs.mkdirSync(path.dirname(header)); writeFile(header);
    const dependencies = buildFileInputs([header]);
    assert.deepEqual(verifyLocalStaticDependencies(dependencies), dependencies);
    assert.throws(() => verifyLocalStaticDependencies([header]), /file-hash map/);
    fs.writeFileSync(header, "changed header\n");
    assert.throws(() => verifyLocalStaticDependencies(dependencies), /inputs changed after production/);
  });

  test("runs the public local slice only through the bounded systemd scope", t => {
    const f = fixture(t), tools = fakeTools(f.root);
    /** @type {{command:string,args:string[],options:{label:string,env:NodeJS.ProcessEnv,cwd:string}}[]} */ const calls = [];
    /** @type {typeof local.runCommand} */
    function run(command, args, options) {
      calls.push({command, args, options});
      assert.equal(command, tools.systemdRun);
      assert.ok(args.includes("--scope")); assert.ok(args.includes("--quiet"));
      assert.ok(args.includes("--property=MemoryMax=" + local.memoryMax));
      assert.ok(args.includes("--property=MemorySwapMax=0"));
      assert.equal(options.env["MOM_AOT_BUILD_ONLY"], "1");
      const child = args.slice(args.indexOf("--") + 1);
      assert.equal(child[0], process.execPath);
      return {pid: 1, output: [Buffer.alloc(0), Buffer.alloc(0), Buffer.alloc(0)],
        stdout: Buffer.alloc(0), stderr: Buffer.alloc(0), status: 0, signal: null};
    }
    const result = buildLocalLinuxJitSdk(f.context, {tools, run});
    assert.deepEqual(calls.map(row => row.options.label), ["systemd memory-limit preflight", "Local Linux JIT SDK build-only slice"]);
    const probe = calls[0]?.args.slice(calls[0].args.indexOf("--") + 1) || [];
    assert.deepEqual(probe.slice(1), [path.join(repo, "scripts/build-local-native-jit-sdk.js"), "--memory-probe"]);
    const build = calls[1]?.args.slice(calls[1].args.indexOf("--") + 1) || [];
    assert.deepEqual(build.slice(1), [path.join(repo, "scripts/build-local-native-jit-sdk.js"), "--local-jit-sdk", f.workspace,
      tools.cmake, tools.ninja, tools.cc, tools.cxx]);
    assert.deepEqual(result.reused, {compilerSdk: false, compilerIsolation: false, staticDpcppSdk: false, staticComponent: false});
    assert.equal(fs.existsSync(localJitSdkPaths(f.workspace).configFile), false, "the wrapper did not write outside its bounded child");
    assert.equal(fs.existsSync(localJitSdkPaths(f.workspace).output), false);
  });

  test("forwards an enclosing-scope marker to the shared invoker and keeps both children in that scope", t => {
    const f = fixture(t), tools = fakeTools(f.root), environment = {...process.env, MOM_LOCAL_NATIVE_SCOPE: "1"},
      calls = /** @type {{command:string,args:string[],options:{label:string,env:NodeJS.ProcessEnv,cwd:string}}[]} */ ([]);
    let proofs = 0;
    const invoke = local.scopedInvoker;
    /** @type {typeof local.scopedInvoker} */
    const scopedInvoker = (selectedTools, selectedEnvironment, run) => {
      assert.equal(selectedEnvironment["MOM_LOCAL_NATIVE_SCOPE"], "1");
      return invoke(selectedTools, selectedEnvironment, run, () => {proofs++;});
    };
    t.mock.method(local, "scopedInvoker", scopedInvoker);
    /** @type {typeof local.runCommand} */
    const run = (command, args, options) => {
      calls.push({command, args, options});
      assert.equal(command, process.execPath, "the child runs directly instead of creating a sibling systemd scope");
      assert.equal(options.env["MOM_LOCAL_NATIVE_SCOPE"], "1");
      return {pid: 1, output: [Buffer.alloc(0), Buffer.alloc(0), Buffer.alloc(0)], stdout: Buffer.alloc(0), stderr: Buffer.alloc(0),
        status: 0, signal: null};
    };
    buildLocalLinuxJitSdk(f.context, {tools, run, environment});
    assert.equal(proofs, 2);
    assert.deepEqual(calls.map(row => row.options.label), ["systemd memory-limit preflight", "Local Linux JIT SDK build-only slice"]);
  });

  test("the heavy internal entry rejects a missing scope before auditing or writing", t => {
    const f = fixture(t), tools = fakeTools(f.root); let audited = false, ran = false;
    assert.throws(() => runLocalJitSdkChild(f.context, tools, {
      verifyScope() {throw new Error("cgroup limit missing");},
      auditInputs() {audited = true;},
      run() {ran = true; throw new Error("producer must not start");},
    }), /cgroup limit missing/);
    assert.equal(audited, false); assert.equal(ran, false);
    const paths = localJitSdkPaths(f.workspace);
    for (const file of [paths.configFile, paths.output, paths.isolation, paths.staticOutput, paths.staticComponent]) {
      assert.equal(fs.existsSync(file), false, file);
    }
  });
});
