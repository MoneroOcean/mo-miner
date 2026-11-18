"use strict";

const {describe, test} = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"),
  path = require("node:path"), os = require("node:os");
const local = require("../scripts/build-local-native-sdk");
const {localAcppSdkPaths, localAcppSdkContextFromWorkspace, localAcppSdkContext, localAcppSdkBuildCommand,
  assertLocalAcppSdkBindings, prepareLocalAcppSources, admitLocalAcppGpuArtifacts, buildLocalLinuxAcppSdk} = require("../scripts/build-local-native-acpp-sdk");
const {staticSyclSdkPlan} = require("../scripts/build-native-sycl-sdk");
const {fileHash} = require("../scripts/build-source-inputs");

const repo = fs.realpathSync(path.resolve(__dirname, ".."));

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-native-acpp-sdk-test-")), workspace = path.join(root, "native-sdk");
  local.admitWorkspace(workspace, repo);
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, workspace, context: localAcppSdkContextFromWorkspace(workspace)};
}

/** @param {string} root @returns {{git:string,cmake:string,ninja:string,cc:string,cxx:string}} */
function fakeTools(root) {
  const bin = path.join(root, "tools"); fs.mkdirSync(bin);
  /** @param {string} name */ const executable = name => {
    const file = path.join(bin, name); fs.writeFileSync(file, "fixture\n"); fs.chmodSync(file, 0o755); return file;
  };
  return {git: executable("git"), cmake: executable("cmake"), ninja: executable("ninja"), cc: executable("icx"), cxx: executable("icpx")};
}

/** @param {string} file @param {string} [value] */
function write(file, value = "fixture\n") {fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, value); return file;}

describe("Local static ACPP SDK slice", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("compiler plugin and generated headers must match their fresh-build capture", t => {
    const f = fixture(t), paths = localAcppSdkPaths(f.workspace),
      plugin = write(path.join(paths.sdk, "build/src/compiler/libacpp-clang.so")),
      header = write(path.join(paths.sdk, "build/include/hipSYCL/common/config.hpp")),
      receipt = write(paths.sdkReceipt, "{}"), capture = path.join(f.workspace, "acpp-gpu-artifacts.json"),
      record = {version: 1, receiptSha256: fileHash(receipt), plugin: {path: plugin, bytes: fs.statSync(plugin).size, sha256: fileHash(plugin)},
        generatedInclude: path.join(paths.sdk, "build/include"),
        generated: [{path: "hipSYCL/common/config.hpp", bytes: fs.statSync(header).size, sha256: fileHash(header)}]};
    assert.throws(() => admitLocalAcppGpuArtifacts(f.context), /capture is missing or unsafe/);
    write(capture, JSON.stringify(record));
    assert.equal(admitLocalAcppGpuArtifacts(f.context).plugin.sha256, record.plugin.sha256);
    fs.appendFileSync(plugin, "changed");
    assert.throws(() => admitLocalAcppGpuArtifacts(f.context), /differ from their original build capture/);
    write(plugin); fs.appendFileSync(header, "changed");
    assert.throws(() => admitLocalAcppGpuArtifacts(f.context), /differ from their original build capture/);
    write(header); write(path.join(paths.sdk, "build/include/untracked.hpp"));
    assert.throws(() => admitLocalAcppGpuArtifacts(f.context), /differ from their original build capture/);
    assert.deepEqual(JSON.parse(fs.readFileSync(capture, "utf8")), record, "failed admission must not restamp the capture");
  });
  test("accepts only stable compiler/ROCm workspace outputs and rejects redirected roots", t => {
    const f = fixture(t), paths = localAcppSdkPaths(f.workspace), selected = localAcppSdkContext(f.context);
    assert.deepEqual(paths, {workspace: f.workspace, fetch: path.join(f.workspace, "acpp-fetch"),
      rootsFile: path.join(f.workspace, "acpp-fetch/roots.json"), fetchReceipt: path.join(f.workspace, "acpp-fetch/fetch.json"),
      prepared: path.join(f.workspace, "acpp-sources"), preparedReceipt: path.join(f.workspace, "acpp-sources/sources.json"),
      preparedCheckout: path.join(f.workspace, "acpp-sources/acpp"), sdk: path.join(f.workspace, "acpp-cuda-hip-sdk"),
      sdkReceipt: path.join(f.workspace, "acpp-cuda-hip-sdk/result.json")});
    assert.equal(selected.rocm.rocmSdk, path.join(f.workspace, "rocm-host-sdk/runtime/sdk"));
    assert.equal(selected.rocm.comgrSdk, path.join(f.workspace, "rocm-comgr-sdk"));
    assert.throws(() => localAcppSdkContext({...f.context, extra: true}), /only admitted compiler inputs/);
    assert.throws(() => localAcppSdkContext({...f.context, rocm: {...f.context.rocm,
      rocmSdk: path.join(f.root, "foreign-rocm")}}), /exact admitted workspace paths and ROCm result/);
    assert.throws(() => localAcppSdkContext({...f.context, inputs: {...f.context.inputs,
      cudaRoot: path.join(f.root, "foreign-cuda")}}), /exact admitted workspace paths and ROCm result/);
  });

  test("delegates the combined CUDA+HIP build recipe to the public static-SYCL planner", t => {
    const f = fixture(t), tools = fakeTools(f.root), command = localAcppSdkBuildCommand(f.context, tools, {PATH: "/usr/bin"}),
      llvmSdk = path.join(f.workspace, "llvm21/sdk"), env = command.env;
    assert.deepEqual(command.plan, staticSyclSdkPlan(f.context.paths.preparedCheckout, llvmSdk,
      f.context.inputs.cudaRoot, f.context.paths.sdk, 2, env, {cc: tools.cc, cxx: tools.cxx}));
    assert.equal(command.script, "scripts/build-native-sycl-sdk.js");
    assert.deepEqual(command.args, [f.context.paths.preparedCheckout, llvmSdk, f.context.inputs.cudaRoot,
      f.context.paths.sdk, tools.cc, tools.cxx]);
    assert.equal(command.plan.backend, "cuda-hip");
    assert.deepEqual(command.plan.hip, {rocm: f.context.rocm.rocmSdk, lld: llvmSdk,
      bitcode: path.join(f.context.rocm.resources, "install/amdgcn/bitcode"), comgr: f.context.rocm.comgrReceipt});
    for (const token of ["-DWITH_CUDA_BACKEND=ON", "-DWITH_ROCM_BACKEND=ON", "-DACPP_STATIC_RUNTIME=ON"]) {
      assert.ok(command.plan.commands[0]?.includes(token), token);
    }
    assert.equal(env["MOM_AOT_BUILD_ONLY"], "1");
    assert.equal(env["MOM_AOT_OPENMP_LIBRARY"], path.join(f.workspace, "openmp-sdk/sdk/libomp.a"));
    assert.equal(env["MOM_BUILD_JOBS"], "2");
    assert.equal(env["PATH"], [...new Set([tools.cmake, tools.ninja, tools.git].map(file => path.dirname(file))), "/usr/bin"].join(path.delimiter),
      "selected build tools lead PATH without replacing the caller's system tools");
    assert.equal("RUNNER_TEMP" in env, false, "the local plan does not invent CI transport fields");
  });

  test("local launch keeps acquisition and the public producer inside the enforced scope", t => {
    const f = fixture(t), baseTools = fakeTools(f.root), tools = {...baseTools, systemdRun: path.join(f.root, "systemd-run")},
      calls = /** @type {Array<{command:string,args:string[],options:unknown}>} */ ([]),
      run = /** @type {typeof local.runCommand} */ ((command, args, options) => {
        calls.push({command, args, options}); return /** @type {ReturnType<typeof local.runCommand>} */ (/** @type {unknown} */ ({}));
      });
    const result = buildLocalLinuxAcppSdk(f.context, {tools, run});
    assert.equal(calls.length, 2);
    assert.ok(calls.every(call => call.command === tools.systemdRun));
    assert.ok(calls.every(call => call.args.includes("--scope")));
    assert.ok(calls.every(call => call.args.includes("--property=MemoryMax=" + local.localMemoryMax(
      /** @type {{env:NodeJS.ProcessEnv}} */ (call.options).env)) && call.args.includes("--property=MemorySwapMax=0")));
    assert.ok(calls[1]?.args.includes("--local-acpp-sdk"));
    assert.ok(calls[1]?.args.includes(f.workspace));
    assert.deepEqual(result.reused, {sources: false, prepared: false, sdk: false});
    assert.equal(result.buildOnly, true); assert.equal(result.executed, false); assert.equal(result.publicationQualified, false);
  });

  test("forwards an enclosing-scope marker to the shared invoker and keeps both children in that scope", t => {
    const f = fixture(t), tools = {...fakeTools(f.root), systemdRun: "/usr/bin/systemd-run"}, environment = {...process.env, MOM_LOCAL_NATIVE_SCOPE: "1"},
      calls = /** @type {Array<{command:string,args:string[],options:{label:string,env:NodeJS.ProcessEnv,cwd:string}}>} */ ([]);
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
      return /** @type {ReturnType<typeof local.runCommand>} */ (/** @type {unknown} */ ({}));
    };
    buildLocalLinuxAcppSdk(f.context, {tools, run, environment});
    assert.equal(proofs, 2);
    assert.equal(calls.length, 2);
    assert.deepEqual(calls.map(call => call.options.label), ["systemd memory-limit preflight", "Local Linux static ACPP SDK build-only slice"]);
  });

  test("preserves prepared ACPP output when its original fetch cohort is missing", t => {
    const f = fixture(t), paths = localAcppSdkPaths(f.workspace), tools = fakeTools(f.root), sentinel = write(path.join(paths.prepared, "sentinel")),
      run = /** @type {typeof local.runCommand} */ (() => {
        launched = true; return /** @type {ReturnType<typeof local.runCommand>} */ (/** @type {unknown} */ ({}));
      });
    let launched = false;
    assert.throws(() => prepareLocalAcppSources(f.context, tools, process.env, run), /without their original fetch cohort/);
    assert.equal(launched, false, "an orphaned prepared output is rejected before launching a producer");
    assert.equal(fs.readFileSync(sentinel, "utf8"), "fixture\n");
    assert.equal(fs.existsSync(paths.fetch), false, "the original fetch root is not recreated or fabricated");
  });

  test("local binding rejects a valid-shaped producer receipt redirected from admitted CUDA/HIP inputs", t => {
    const f = fixture(t), selected = localAcppSdkContext(f.context), tools = fakeTools(f.root),
      llvmSdk = path.join(f.workspace, "llvm21/sdk"), cudaRoot = selected.inputs.cudaRoot,
      comgr = {receipt: selected.rocm.comgrReceipt, roots: {llvm: llvmSdk, lld: llvmSdk,
        device: path.join(selected.rocm.resources, "install")}, archive: write(path.join(selected.rocm.comgrSdk, "build/libamd_comgr.a")), files: {}},
      cudaHeaders = Object.fromEntries(["cuda.h", "cuda_runtime_api.h"].map(name => [name, fileHash(write(path.join(cudaRoot, "include", name)))])),
      cudaStatic = Object.fromEntries(["libcudart_static.a", "libculibos.a"].map(name => {
        const file = write(path.join(cudaRoot, "lib64", name)); return [name === "libcudart_static.a" ? "cudart" : "culibos", {path: file, sha256: fileHash(file)}];
      })),
      cuda = {cuda: {root: cudaRoot, headers: {files: cudaHeaders}, staticRuntime: cudaStatic}},
      hipLibraries = ["libamdhip64.a", "libhiprtc.a", "libhsakmt-staticdrm.a", "libelf.a", "libdrm_amdgpu.a", "libdrm.a", "libnuma.a"]
        .map(name => write(path.join(selected.rocm.rocmSdk, "lib", name))),
      hsaRuntime = write(path.join(selected.rocm.rocmSdk, "lib/libhsa-runtime64.a")),
      staticLibraries = Object.values(cudaStatic).map(row => row.path), openmpArchive = write(path.join(f.workspace, "openmp-sdk/sdk/libomp.a")),
      openmp = {archive: openmpArchive},
      inputs = /** @type {any} */ ({cuda, rocm: {comgr: {comgr: {sdk: comgr}}, audit: {sdk: {prefix: selected.rocm.rocmSdk}}}, openmp,
        sourceState: {source: selected.paths.fetch, preparedReceipt: selected.paths.preparedReceipt,
          preparedCheckout: selected.paths.preparedCheckout}}),
      recordLibraries = Object.fromEntries([...hipLibraries, comgr.archive, ...staticLibraries, "/openmp/libomp.a"].map(file =>
        [file, file === "/openmp/libomp.a" ? fileHash(openmpArchive) : fileHash(file)])),
      fingerprints = Object.fromEntries([...Object.keys(cudaHeaders).map(name => [path.join(cudaRoot, "include", name), cudaHeaders[name]]),
        ...Object.values(cudaStatic).map(row => [row.path, row.sha256]), [hsaRuntime, fileHash(hsaRuntime)]]),
      record = {version: 1, backend: "cuda-hip", cudaVersion: 12060, revision: "da2463e45aa90aa36306c45abcfc05b87de51bc6",
        inputs: {source: selected.paths.preparedCheckout, build: path.join(selected.paths.sdk, "build"), revision: "da2463e45aa90aa36306c45abcfc05b87de51bc6",
          cxx: path.join(llvmSdk, "bin/clang++"), flags: [], links: ["-Wl,--whole-archive", hsaRuntime, "-Wl,--no-whole-archive"],
          libraries: [...hipLibraries, comgr.archive, ...staticLibraries, "/openmp/libomp.a"], comgr},
        hostCompilers: {cc: tools.cc, cxx: tools.cxx}, hostCompilerInputs: {}, compilerVersions: {}, fingerprints,
        libraries: recordLibraries, recipe: {}, buildOnly: true, publicationQualified: false, scope: "test"},
      sdk = /** @type {any} */ ({
        receipt: selected.paths.sdkReceipt, receiptSha256: "a".repeat(64), backend: "cuda-hip", revision: record.revision,
        cudaVersion: 12060, source: selected.paths.preparedCheckout, build: path.join(selected.paths.sdk, "build"),
        cxx: path.join(llvmSdk, "bin/clang++"), libraries: [...hipLibraries, comgr.archive, ...staticLibraries, openmpArchive],
        links: ["-Wl,--whole-archive", hsaRuntime, "-Wl,--no-whole-archive"], resolvedLinks: {},
        files: {...recordLibraries, [openmpArchive]: fileHash(openmpArchive), ...fingerprints}, pathBindings: {}, comgr, openmp});
    assert.equal(assertLocalAcppSdkBindings(f.context, sdk, record, tools, cuda, inputs), sdk);
    assert.throws(() => assertLocalAcppSdkBindings(f.context, sdk, {...record, inputs: {...record.inputs,
      comgr: {...comgr, receipt: path.join(f.root, "foreign-comgr.json")}}}, tools, cuda, inputs), /does not bind/);
    assert.throws(() => assertLocalAcppSdkBindings(f.context, sdk, {...record, fingerprints: {
      ...fingerprints, [path.join(cudaRoot, "include/cuda.h")]: "0".repeat(64)}}, tools, cuda, inputs), /does not bind/);
  });
});
