"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"),
  path = require("node:path"), os = require("node:os");
const local = require("../scripts/build-local-native-sdk");
const {buildLinuxCudaImageTree} = require("../scripts/prepare-native-ci");
const {localCudaImagePaths, verifyLocalCudaCompilerInputs, verifyLocalCudaHeaderRoots, auditLocalCudaImages,
  runLocalCudaImageChild, buildLocalCudaImages} =
  require("../scripts/build-local-native-cuda-images");
const {cudaHeaderResources} = require("../scripts/native-jit-headers");
const {buildFileInputs} = require("../scripts/build-source-inputs");

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-cuda-images-")),
    compiler = path.join(root, "icpx"), header = path.join(root, "header.h"), workspace = path.join(root, "workspace");
  fs.writeFileSync(compiler, "fixture compiler"); fs.writeFileSync(header, "fixture header");
  local.admitWorkspace(workspace, fs.realpathSync(path.resolve(__dirname, "..")));
  const inputs = buildFileInputs([compiler, header]), rows = [{major: 12, minor: 6, inputs}, {major: 13, minor: 3, inputs}];
  t.after(() => fs.rmSync(root, {force: true, recursive: true}));
  return {root, compiler, header, workspace, inputs, rows};
}

/** Build tiny packs with the same public API used by the CUDA image producer.
 * @param {import("node:test").TestContext} t */
function headerFixture(t) {
  const f = fixture(t), state = localCudaImagePaths(f.workspace), cudaInclude = path.join(state.dpcpp.cudaRoot, "include"),
    roots = [state.cutlassInclude, cudaInclude], marker = "pinned CUTLASS marker\n", license = "pinned CUTLASS license\n";
  fs.mkdirSync(path.join(state.cutlassInclude, "cute"), {recursive: true});
  fs.mkdirSync(path.join(cudaInclude, "cuda/std"), {recursive: true});
  fs.mkdirSync(state.headerRoot, {recursive: true});
  fs.writeFileSync(path.join(state.cutlassInclude, "cute/tensor.hpp"), "fixture tensor header\n");
  fs.writeFileSync(path.join(cudaInclude, "cuda_runtime.h"), "fixture runtime header\n");
  fs.writeFileSync(path.join(cudaInclude, "cuda/std/cstdint"), "fixture cstdint header\n");
  fs.writeFileSync(state.cutlassMarker, marker); fs.writeFileSync(state.cutlassLicense, license);
  const {source, ...manifest} = cudaHeaderResources(roots);
  for (const pack of [state.header126, state.header133]) {
    fs.writeFileSync(pack, source);
    fs.writeFileSync(pack + ".json", JSON.stringify(manifest, null, 2) + "\n");
  }
  return {state, marker, license};
}

test.describe("Local CUDA image production and warm admission", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("uses stable workspace outputs with separate old/new compiler cohorts", t => {
    const {workspace} = fixture(t), state = localCudaImagePaths(workspace);
    assert.deepEqual(localCudaImagePaths(workspace), state);
    assert.notEqual(state.sdk126, state.linux.sdkRoot);
    assert.notEqual(state.source126, state.linux.source);
    assert.equal(state.dpcpp.workspaceRoot, workspace);
    assert.equal(state.admission, path.join(workspace, "cuda-image-admission.json"));
    assert.equal(state.noticeManifest, path.join(state.handoffRoot, "notice-sources/manifest.json"));
  });
  test("rejects relative or noncanonical workspace paths without creating them", () => {
    for (const value of ["relative", "/tmp/root/../elsewhere", "/tmp/root\nelsewhere"]) {
      assert.throws(() => localCudaImagePaths(value), /canonical and absolute/);
    }
  });
  test("rechecks the exact public manifest compiler input maps", t => {
    const f = fixture(t);
    assert.deepEqual(verifyLocalCudaCompilerInputs(f.rows, f.compiler), f.inputs);
  });
  for (const [name, rows] of Object.entries({missing: null, object: {}, empty: [], duplicate: [12, 12], modernOnly: [13, 13]})) {
    test("rejects an incomplete compiler matrix: " + name, t => {
      const f = fixture(t), value = Array.isArray(rows) ? rows.map(major => ({major, minor: major === 12 ? 6 : 3, inputs: f.inputs})) : rows;
      assert.throws(() => verifyLocalCudaCompilerInputs(value, f.compiler), /exact 12.6\/13.3 compiler matrix/);
    });
  }
  test("rejects a changed transitive header rather than restamping the cached bundle", t => {
    const f = fixture(t); fs.writeFileSync(f.header, "changed");
    assert.throws(() => verifyLocalCudaCompilerInputs(f.rows, f.compiler), /inputs changed/);
  });
  test("warm header admission catches changed CUTLASS bytes with marker and license unchanged", t => {
    const f = headerFixture(t), header = path.join(f.state.cutlassInclude, "cute/tensor.hpp");
    verifyLocalCudaHeaderRoots(f.state);
    fs.writeFileSync(header, "changed tensor header\n");
    assert.throws(() => verifyLocalCudaHeaderRoots(f.state), /Live CUDA filesystem include roots differ/);
    assert.equal(fs.readFileSync(f.state.cutlassMarker, "utf8"), f.marker);
    assert.equal(fs.readFileSync(f.state.cutlassLicense, "utf8"), f.license);
  });
  test("warm header admission catches a deleted CUTLASS header with marker and license unchanged", t => {
    const f = headerFixture(t), header = path.join(f.state.cutlassInclude, "cute/tensor.hpp");
    verifyLocalCudaHeaderRoots(f.state);
    fs.unlinkSync(header);
    assert.throws(() => verifyLocalCudaHeaderRoots(f.state), /Live CUDA filesystem include roots differ/);
    assert.equal(fs.readFileSync(f.state.cutlassMarker, "utf8"), f.marker);
    assert.equal(fs.readFileSync(f.state.cutlassLicense, "utf8"), f.license);
  });
  test("requires each compiler cohort to bind the selected host compiler", t => {
    const f = fixture(t), rows = f.rows.map(row => ({...row, inputs: buildFileInputs([f.header])}));
    assert.throws(() => verifyLocalCudaCompilerInputs(rows, f.compiler), /omit the selected host compiler/);
  });
  test("rejects array-valued compiler input maps", t => {
    const f = fixture(t);
    assert.throws(() => verifyLocalCudaCompilerInputs(f.rows.map(row => ({...row, inputs: [f.compiler]})), f.compiler),
      /omit the selected host compiler/);
  });
  test("proves actual child containment before reading a workspace or calling a producer", () => {
    let built = false;
    assert.throws(() => runLocalCudaImageChild("/missing/workspace", "/missing/compiler", {
      verifyScope() {throw new Error("scope missing");}, build() {built = true; throw new Error("must not run");},
    }), /scope missing/);
    assert.equal(built, false);
  });
  test("the public wrapper scopes both preflight and image production with memory/swap caps", t => {
    const f = fixture(t), tools = {systemdRun: "/fixture/systemd-run", git: "/fixture/git", cmake: "/fixture/cmake",
        ninja: "/fixture/ninja", cc: f.compiler, cxx: f.compiler}, calls = [];
    /** @type {typeof local.runCommand} */
    function run(command, args, options) {
      calls.push({command, args, options});
      assert.equal(command, tools.systemdRun);
      assert.ok(args.includes("--property=MemoryMax=" + local.memoryMax));
      assert.ok(args.includes("--property=MemorySwapMax=0"));
      assert.equal(options.env["MOM_AOT_BUILD_ONLY"], "1");
      return {pid: 1, output: [], status: 0, signal: null, stdout: Buffer.alloc(0), stderr: Buffer.alloc(0)};
    }
    const output = buildLocalCudaImages(f.workspace, {tools, run});
    assert.equal(calls.length, 2);
    assert.equal(output.image, path.join(f.workspace, "cuda-images-handoff/cuda-images.inc"));
    assert.equal(fs.existsSync(localCudaImagePaths(f.workspace).root), false, "mock dispatch must not build or download");
  });
  test("the actual shared producer rejects missing input provenance before fetching or writing", t => {
    const f = fixture(t), state = localCudaImagePaths(f.workspace);
    assert.throws(() => buildLinuxCudaImageTree(state, f.compiler, {}), /ENOENT/);
    for (const directory of [state.root, state.source126, state.sdk126, state.linux.source, state.linux.sdkRoot, state.handoffRoot]) {
      assert.equal(fs.existsSync(directory), false);
    }
  });
  test("the actual shared producer preserves interrupted outputs instead of replacing them", t => {
    const f = fixture(t), state = localCudaImagePaths(f.workspace), sentinel = path.join(state.root, "keep.txt");
    fs.mkdirSync(state.root); fs.writeFileSync(sentinel, "preserve");
    assert.throws(() => buildLinuxCudaImageTree(state, f.compiler, {}), /fresh same-run outputs/);
    assert.equal(fs.readFileSync(sentinel, "utf8"), "preserve");
  });
  test("warm admission refuses a redirected output root before consuming its receipts", t => {
    const f = fixture(t), state = localCudaImagePaths(f.workspace), elsewhere = path.join(f.root, "elsewhere");
    fs.mkdirSync(elsewhere); fs.symlinkSync(elsewhere, state.root, "dir");
    assert.throws(() => auditLocalCudaImages(state, f.compiler), /escaped its canonical workspace/);
  });
  for (const kind of ["producer", "handoff", "config"]) {
    test("warm admission rejects an aliased " + kind + " receipt before reading inputs", t => {
      const f = fixture(t), state = localCudaImagePaths(f.workspace), elsewhere = path.join(f.root, "receipt.json"),
        files = [state.result, path.join(state.handoffRoot, "result.json"), state.config],
        index = ["producer", "handoff", "config"].indexOf(kind);
      for (const root of [state.root, state.headerRoot, state.output, state.handoffRoot, state.noticeRoot]) {fs.mkdirSync(root, {recursive: true});}
      for (const file of files.slice(0, index)) {fs.writeFileSync(file, "{}");}
      fs.writeFileSync(elsewhere, "{}"); fs.symlinkSync(elsewhere, files[index] || "");
      assert.throws(() => auditLocalCudaImages(state, f.compiler), /receipt must be a canonical regular file/);
    });
  }
});
