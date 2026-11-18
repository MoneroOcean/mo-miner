"use strict";

const {describe, test} = require("node:test"), assert = require("node:assert/strict"),
  fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const {fileHash} = require("../scripts/build-source-inputs");
const {localWindowsGpuTools, admitLocalWindowsDpcppGpuInputs, admitLocalWindowsGpuInputs, composeLocalWindowsGpuConfigs} =
  require("../scripts/build-local-native-windows-gpu-config");

/** Inert files only; no compiler, SDK build or device query runs.
 * @param {import('node:test').TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-windows-gpu-config-")), prefix = path.join(root, "sdk"),
    bin = path.join(prefix, "bin"), files = /** @type {Record<string,any>} */ ({});
  fs.mkdirSync(bin, {recursive: true});
  for (const name of ["llvm-nm", "llvm-readobj", "llvm-objcopy"]) {
    const file = path.join(bin, name); fs.writeFileSync(file, name); fs.chmodSync(file, 0o700);
    files["bin/" + name] = {type: "file", bytes: fs.statSync(file).size, sha256: fileHash(file)};
  }
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, bin, admission: {sdk: {path: prefix, snapshot: {files}}}};
}

/** Mock only the already-admitted API boundary, not the production admission itself. */
function cohort() {
  return {workspace: "/local/workspace", sourceRoot: "/local/windows-sources/dpcpp", sources: {},
    dpcppRoot: "/local/dpcpp", cudaRoot: "/local/cuda", dpcppManifest: {},
    compilerRoot: "/local/intel/2026.2", tools: {llvmNm: "/local/llvm/bin/llvm-nm", toolRoot: "/local/llvm/bin", files: {}},
    host: {}, cudaImages: {include: "/local/images.inc", hashes: {}}, imageAdmission: {}, files: {"/local/receipt": "a".repeat(64)},
    acpp: {source: "/local/acpp/source", generatedInclude: "/local/acpp/build/include", plugin: "/local/acpp/build/plugin.so"},
    frontend: {clang: "/local/llvm/bin/clang", clangxx: "/local/llvm/bin/clang++", resourceInclude: "/local/llvm/lib/clang/21/include"},
    hip: {include: "/local/hip/include"}};
}

describe("Local Windows DPC++ GPU descriptor admission", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("uses only exact executable LLVM tools from the admitted installation", t => {
    const f = fixture(t), selected = localWindowsGpuTools(f.root, f.admission);
    assert.equal(selected.toolRoot, f.bin); assert.equal(selected.llvmNm, path.join(f.bin, "llvm-nm"));
    assert.deepEqual(Object.keys(selected.files).sort(), ["llvm-nm", "llvm-objcopy", "llvm-readobj"].map(name => path.join(f.bin, name)));
    assert.throws(() => localWindowsGpuTools(f.root, {sdk: {path: "/foreign", snapshot: {files: {}}}}), /admitted LLVM installation/);
  });
  test("rejects modified, missing, non-executable and redirected selected tools", t => {
    const f = fixture(t), file = path.join(f.bin, "llvm-nm"), original = fs.readFileSync(file);
    fs.appendFileSync(file, "changed"); assert.throws(() => localWindowsGpuTools(f.root, f.admission), /differs from its LLVM admission/);
    fs.writeFileSync(file, original); fs.chmodSync(file, 0o600);
    assert.throws(() => localWindowsGpuTools(f.root, f.admission), /differs from its LLVM admission/);
    fs.unlinkSync(file); assert.throws(() => localWindowsGpuTools(f.root, f.admission), /differs from its LLVM admission/);
    fs.symlinkSync(path.join(f.bin, "llvm-readobj"), file);
    assert.throws(() => localWindowsGpuTools(f.root, f.admission), /differs from its LLVM admission/);
  });
  test("requires enforced memory containment before any workspace inspection", () => {
    assert.throws(() => admitLocalWindowsDpcppGpuInputs(/** @type {any} */ (null), {
      verifyScope() {throw new Error("no enforced scope");},
    }), /no enforced scope/);
  });
  test("rejects missing pins and CI identity fields instead of inventing a local handoff", () => {
    for (const value of [null, {}, {runAttempt: "123-1"}, {workspace: "/missing", windowsSourceSha256: "a".repeat(64)}]) {
      assert.throws(() => admitLocalWindowsDpcppGpuInputs(/** @type {any} */ (value), {verifyScope() {return undefined;}}), /explicit workspace\/source pins/);
    }
  });
  test("local admission joins original auditors without a CI archive or rewritten receipt", t => {
    const f = fixture(t), local = require("../scripts/build-local-native-sdk"), workspace = path.join(f.root, "workspace"),
      repo = fs.realpathSync(path.resolve(__dirname, ".."));
    local.admitWorkspace(workspace, repo);
    const paths = local.supportPaths(workspace), selected = /** @type {Record<string,any>} */ ({}),
      calls = /** @type {string[]} */ ([]);
    /** @param {string} file @param {string} [text] */
    const write = (file, text = "{}") => {fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, text); return file;};
    for (const name of ["llvm-nm", "llvm-readobj", "llvm-objcopy"]) {
      const file = write(path.join(paths.llvmSdk, "sdk/bin", name), name); fs.chmodSync(file, 0o700);
      selected["bin/" + name] = {type: "file", bytes: fs.statSync(file).size, sha256: fileHash(file)};
    }
    const source = write(path.join(workspace, "windows-sources/sources.json")),
      manifest = write(path.join(paths.dpcppCudaRoot, "inputs.json")),
      image = write(path.join(workspace, "cuda-images-handoff/cuda-images.inc"), "image fixture");
    write(image + ".json"); write(path.join(path.dirname(image), "result.json"));
    const tools = /** @type {any} */ ({cxx: "/fixture/intel/2026.2/bin/icpx"}),
      input = {workspace, windowsSdkRoot: "/fixture/windows-sdk", windowsSdkPins: /** @type {any} */ ({}),
        windowsSourceSha256: fileHash(source), tools},
      dependencies = {
        verifyScope() {calls.push("scope");}, resolveTools() {calls.push("tools"); return tools;},
        verifySources: /** @type {any} */ ((/** @type {string} */ file, /** @type {string} */ target) => {
          calls.push("sources"); assert.equal(file, source); assert.equal(target, "windows");
          return {target, prepared: {dpcpp: {root: path.join(workspace, "windows-sources/dpcpp")}}};
        }),
        verifyLlvm: /** @type {any} */ ((/** @type {string} */ root) => {
          calls.push("llvm"); assert.equal(root, paths.llvmSdk);
          return {sdk: {path: path.join(root, "sdk"), snapshot: {files: selected}}};
        }),
        auditCuda: /** @type {any} */ ((/** @type {any} */ state, /** @type {any} */ value, /** @type {string} */ sha256) => {
          calls.push("cuda"); assert.equal(state.manifest, manifest); assert.deepEqual(value, {}); assert.equal(sha256, fileHash(manifest));
        }),
        readHost: /** @type {any} */ ((/** @type {string} */ root) => {calls.push("host"); assert.equal(root, input.windowsSdkRoot); return {original: true};}),
        auditImages: /** @type {any} */ ((/** @type {any} */ state, /** @type {string} */ compiler) => {
          calls.push("images"); assert.equal(state.handoffRoot, path.dirname(image)); assert.equal(compiler, tools.cxx);
          return {fingerprints: Object.fromEntries([image, image + ".json", path.join(path.dirname(image), "result.json")].map(file => [file, fileHash(file)]))};
        }),
      };
    const admitted = admitLocalWindowsDpcppGpuInputs(input, dependencies);
    assert.deepEqual(calls, ["scope", "tools", "sources", "llvm", "cuda", "host", "images"]);
    assert.equal(admitted.files[source], input.windowsSourceSha256);
    assert.equal(admitted.compilerRoot, "/fixture/intel/2026.2");
    assert.equal(admitted.cudaImages.include, image);
    assert.equal(Object.hasOwn(admitted, "runAttempt"), false);
    calls.length = 0;
    assert.throws(() => admitLocalWindowsDpcppGpuInputs({...input, windowsSourceSha256: "0".repeat(64)}, dependencies), /differs from its pin/);
    assert.deepEqual(calls, ["scope", "tools"]);
    assert.equal(fs.readFileSync(source, "utf8"), "{}");
    assert.throws(() => admitLocalWindowsDpcppGpuInputs(input, {...dependencies,
      auditImages: /** @type {any} */ (() => ({fingerprints: {[image]: "0".repeat(64)}})),
    }), /inputs changed during admission/);
  });
  test("all five profiles share admission and the maintained recipe constructors", () => {
    const admitted = cohort(), seen = /** @type {Array<{profile:string,input:any}>} */ ([]),
      acpp = /** @type {Array<{profile:string,input:any}>} */ ([]); let audits = 0;
    const result = composeLocalWindowsGpuConfigs(/** @type {any} */ ({}), {
      admit: /** @type {any} */ (() => {audits++; return admitted;}),
      compose: /** @type {any} */ ((/** @type {string} */ profile, /** @type {any} */ input) => {
        seen.push({profile, input}); return {config: {profile}, configBytes: Buffer.from(profile), configSha256: "a".repeat(64)};
      }),
      composeAcpp: /** @type {any} */ ((/** @type {string} */ profile, /** @type {any} */ input) => {
        acpp.push({profile, input}); return {config: {profile}, configBytes: Buffer.from(profile), configSha256: "b".repeat(64)};
      }),
    });
    assert.equal(audits, 2); assert.deepEqual(seen.map(row => row.profile), ["oneapi", "dpcpp", "dpcpp-opencl"]);
    for (const {profile, input} of seen) {
      assert.equal(input.compilerRoot, profile === "oneapi" ? admitted.compilerRoot : admitted.dpcppRoot);
      assert.equal(input.host, admitted.host); assert.equal(input.sourceRoot, admitted.sourceRoot);
      assert.equal(input.cudaImages, profile === "dpcpp" ? admitted.cudaImages : null);
      assert.equal(Object.hasOwn(input, "cudaRoot"), profile === "dpcpp");
      assert.equal(Object.hasOwn(input, "runAttempt"), false);
    }
    assert.deepEqual(acpp.map(row => row.profile), ["acpp-cuda", "acpp-hip"]);
    assert.deepEqual(acpp[0]?.input.cuda, {sdk: admitted.cudaRoot, images: admitted.cudaImages.include});
    assert.equal(Object.hasOwn(acpp[0]?.input, "hipInclude"), false);
    assert.equal(acpp[1]?.input.cuda, null); assert.equal(acpp[1]?.input.hipInclude, admitted.hip.include);
    assert.deepEqual(Object.keys(result.profiles), ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"]);
    assert.equal(result.buildOnly, true); assert.equal(result.executed, false); assert.equal(result.publicationQualified, false);
  });
  test("ACPP joins its original SDK, captured artifacts, HIP headers and selected LLVM frontend", t => {
    const f = fixture(t), local = require("../scripts/build-local-native-sdk"), workspace = path.join(f.root, "workspace");
    local.admitWorkspace(workspace, fs.realpathSync(path.resolve(__dirname, "..")));
    const prefix = path.join(workspace, "llvm21/sdk"), build = path.join(workspace, "acpp-cuda-hip-sdk/build"),
      base = {...cohort(), workspace, llvm: {sdk: {path: prefix, snapshot: {files: {
        "bin/clang": {type: "symlink", target: "clang-21"}, "bin/clang++": {type: "symlink", target: "clang"},
        "bin/clang-21": {type: "file", bytes: 1, sha256: "a".repeat(64)},
        "lib/clang/21/include/stddef.h": {type: "file", bytes: 1, sha256: "b".repeat(64)},
      }}}}},
      sdk = {build, source: path.join(workspace, "acpp-sources/acpp")},
      artifacts = {generatedInclude: path.join(build, "include"), plugin: {path: path.join(build, "src/compiler/libacpp-clang.so")}},
      inputs = {rocm: {comgr: {hip: {include: path.join(workspace, "rocm-hip-headers/include")}}}},
      tools = {git: "/fixture/git", cmake: "/fixture/cmake", ninja: "/fixture/ninja", cc: "/fixture/icx", cxx: "/fixture/icpx"},
      calls = /** @type {string[]} */ ([]), dependencies = /** @type {any} */ ({
        admitBase() {calls.push("base"); return base;},
        auditAcppInputs(/** @type {any} */ context, /** @type {any} */ selectedTools) {
          calls.push("inputs"); assert.equal(context.source.workspace, workspace); assert.deepEqual(selectedTools, tools); return inputs;
        },
        auditAcpp(/** @type {any} */ _context, /** @type {any} */ _tools, /** @type {any} */ options) {
          calls.push("sdk"); assert.equal(options.auditInputs(), inputs); return sdk;
        },
        readArtifacts() {calls.push("artifacts"); return artifacts;},
        resolveFrontend(/** @type {string} */ root, /** @type {any} */ selected) {
          calls.push("frontend"); assert.equal(root, prefix);
          assert.equal(selected.files["bin/clang"].target, "clang-21");
          assert.equal(selected.files["lib/clang/21/include/stddef.h"].sha256, "b".repeat(64)); return selected;
        },
      }), input = /** @type {any} */ ({workspace, tools});
    const admitted = admitLocalWindowsGpuInputs(input, dependencies);
    assert.deepEqual(calls, ["base", "inputs", "sdk", "artifacts", "frontend"]);
    assert.equal(admitted.acpp.source, sdk.source); assert.equal(admitted.hip, inputs.rocm.comgr.hip);
    assert.equal(admitted.acpp.plugin, artifacts.plugin.path);
    dependencies.readArtifacts = () => ({...artifacts, generatedInclude: "/foreign/include"});
    assert.throws(() => admitLocalWindowsGpuInputs(input, dependencies), /different SDK build/);
  });
  test("changed producer receipts prevent returning apparently successful descriptors", () => {
    const first = cohort(), second = cohort(); second.files["/local/receipt"] = "b".repeat(64); let audits = 0;
    assert.throws(() => composeLocalWindowsGpuConfigs(/** @type {any} */ ({}), {
      admit: /** @type {any} */ (() => ++audits === 1 ? first : second),
      compose: /** @type {any} */ (() => ({})),
      composeAcpp: /** @type {any} */ (() => ({})),
    }), /producer inputs changed/);
    assert.equal(audits, 2);
  });
  test("an admission or shared-constructor failure stops without repairing inputs", () => {
    let calls = 0;
    assert.throws(() => composeLocalWindowsGpuConfigs(/** @type {any} */ ({}), {
      admit() {throw new Error("bad original SDK");}, compose: /** @type {any} */ (() => {calls++;}),
    }), /bad original SDK/);
    assert.equal(calls, 0);
    assert.throws(() => composeLocalWindowsGpuConfigs(/** @type {any} */ ({}), {
      admit: /** @type {any} */ (() => cohort()), compose() {throw new Error("header ABI mismatch");},
    }), /header ABI mismatch/);
  });
});
