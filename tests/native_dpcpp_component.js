"use strict";
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), test = require("node:test"), assert = require("node:assert/strict");
const {dpcppRuntimeGraph, dpcppComponentInputs, exportDpcppComponent} = require("../scripts/native-dpcpp-component");
const {fileHash} = require("../scripts/build-source-inputs"), {pins} = require("../scripts/probe-static-dpcpp");
const {nativeSdkLinkInputs} = require("../scripts/build-native-release");
const archives = ["libmom-sycl-static.a", "lib/libur_loader.a", "lib/libur_adapter_level_zero.a", "lib/libur_adapter_opencl.a", "lib/libze_loader.a", "deps/libhwloc.a"];
const objects = ["cuda-adapter.o", "level-zero-v2-adapter.o", "source-jit.o"];
/** An original link edge shape, not a source or library registry used by production.
 * @param {string} jit */
function graph(jit = "/jit/compiler.o") {
  return ["native-dpcpp-core:", "  input: CXX_EXECUTABLE_LINKER__native-dpcpp-core_Release", "    " + jit,
    "    CMakeFiles/native-dpcpp-core.dir/source/native_dpcpp_static.cpp.o", "    CMakeFiles/native-dpcpp-core.dir/runtime/driver-library.cpp.o",
    "    | cuda-native/adapter.o", "    | level-zero-v2-native/adapter.o", ...archives.map(n => "    | " + n),
    "    || helper-not-a-link-input", "  outputs:", "    all", ""].join("\n");
}
/** Synthetic bounded ELF/archives exercise admission, not a real SDK qualification.
 * @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-dpcpp-component-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const sdk = path.join(root, "sdk"), build = path.join(sdk, "build"), target = path.join(root, "component"); fs.mkdirSync(build, {recursive: true});
  const elf = Buffer.alloc(24); Buffer.from("7f454c460201", "hex").copy(elf); elf.writeUInt16LE(1, 16); elf.writeUInt16LE(62, 18);
  /** @param {string} name @param {string | Buffer} bytes */
  const put = (name, bytes) => {const file = path.join(root, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, bytes); return file;};
  const jit = put("jit/compiler.o", elf); for (const name of archives) {put("sdk/build/" + name, "!<arch>\n");}
  put("sdk/build/cuda-native/adapter.o", elf); put("sdk/build/level-zero-v2-native/adapter.o", elf);
  const ninja = put("ninja", "#!" + process.execPath + "\nprocess.stdout.write(" + JSON.stringify(graph(jit)) + ");\n"); fs.chmodSync(ninja, 0o700);
  const archive = path.join(build, "libmom-sycl-static.a"), producer = path.join(sdk, "result.json");
  const record = {version: 1, target: "linux", buildOnly: true, executed: false, pins, archive, sha256: fileHash(archive), inputs: {[ninja]: fileHash(ninja), [jit]: fileHash(jit)}};
  fs.writeFileSync(producer, JSON.stringify(record)); fs.writeFileSync(path.join(sdk, "before.json"), JSON.stringify({configure: ["-DMOM_DPCPP_JIT_OBJECT=" + jit]}));
  const exportIt = () => {
    const old = process.env["MOM_AOT_BUILD_ONLY"]; process.env["MOM_AOT_BUILD_ONLY"] = "1";
    try {return exportDpcppComponent(producer, ninja, target);} finally {if (old === undefined) {delete process.env["MOM_AOT_BUILD_ONLY"];} else {process.env["MOM_AOT_BUILD_ONLY"] = old;}}
  };
  return {root, sdk, build, target, put, jit, ninja, producer, record, exportIt};
}

test.describe("Actual original Ninja runtime graph selection", () => {
  test("archive membership comes from the link edge in its original order", () => {
    assert.deepEqual(dpcppRuntimeGraph(graph(), "/build", "/jit/compiler.o").archives, archives.map(n => path.join("/build", n)));
  });
  test("only the three isolated helpers become SDK objects, not fixture code", () => {
    assert.deepEqual(Object.keys(dpcppRuntimeGraph(graph(), "/build", "/jit/compiler.o").objects).sort(), [...objects].sort());
  });
  test("order-only targets cannot smuggle another archive into the SDK", () => {
    const r = dpcppRuntimeGraph(graph().replace("helper-not-a-link-input", "lib/not-a-real-input.a"), "/build", "/jit/compiler.o");
    assert.equal(r.archives.length, archives.length);
  });
  test("unknown link edges and extra graph outputs fail closed", () => {
    for (const text of [graph().replace("Release", "Debug"), graph().replace("    all", "    unexpected"), graph().replace("native-dpcpp-core:", "other:")]) {
      assert.throws(() => dpcppRuntimeGraph(text, "/build", "/jit/compiler.o"));
    }
  });
  test("missing CUDA, OpenCL, Level Zero or source JIT cannot narrow the runtime silently", () => {
    for (const text of [graph().replace("    | cuda-native/adapter.o\n", ""), graph().replace("    /jit/compiler.o\n", ""),
      graph().replace("    | lib/libur_adapter_opencl.a\n", ""), graph().replace("    | lib/libur_adapter_level_zero.a\n", "")]) {
      assert.throws(() => dpcppRuntimeGraph(text, "/build", "/jit/compiler.o"), /Incomplete/);
    }
  });
  test("duplicate names and extra executable fixture objects are rejected", () => {
    for (const text of [graph().replace("  outputs:", "    | other/libhwloc.a\n  outputs:"), graph().replace("  outputs:", "    injected.o\n  outputs:")]) {
      assert.throws(() => dpcppRuntimeGraph(text, "/build", "/jit/compiler.o"));
    }
  });
  test("traversal, unexpected external inputs and escaped shell fragments are not interpreted", () => {
    for (const input of ["../outside.a", "/outside/lib.a", "lib/evil\\name.a", "lib/name.a;execute"]) {
      assert.throws(() => dpcppRuntimeGraph(graph().replace("  outputs:", "    | " + input + "\n  outputs:"), "/build", "/jit/compiler.o"));
    }
  });
  test("metadata limits apply before graph allocation or filesystem lookup", () => {
    assert.throws(() => dpcppRuntimeGraph("x".repeat(65537), "/build", "/jit/compiler.o"), /bounded/);
    assert.throws(() => dpcppRuntimeGraph(graph(), "relative", "/jit/compiler.o"), /bounded/);
  });
});

test.describe("Self-contained static DPC++ component transport", () => {
  test("the exporter copies only runtime archives/helpers and the consumer resolves exact paths", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), r = f.exportIt(); assert.equal(r.sdkObjects.length, 3); assert.equal(Object.keys(r.files).length, archives.length + 4);
    assert.equal(r.links[0], "-Wl,--whole-archive"); assert.equal(r.links[1], path.join(f.target, "lib/libmom-sycl-static.a"));
    assert.deepEqual(fs.readdirSync(f.target).sort(), ["component.json", "lib", "objects"]);
  });
  test("the installed component no longer requires its original SDK or JIT paths", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t); f.exportIt(); fs.renameSync(f.sdk, path.join(f.root, "original-sdk-away")); fs.renameSync(path.dirname(f.jit), path.join(f.root, "original-jit-away"));
    assert.equal(dpcppComponentInputs(f.target).sdkObjects.length, 3);
  });
  test("the release linker consumes the component and fingerprints every installed input", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t); f.exportIt();
    const sdk = {version: 1, cpu: {cache: f.root}, gpu: {},
      controller: {source: f.root, build: f.root, openssl: f.root, io: f.root, cc: process.execPath, nvml: f.put("nvml.h", "fixture")},
      sdk: {cxx: process.execPath, defaultCompiler: "oneapi", flags: /** @type {string[]} */ ([]), links: /** @type {string[]} */ ([]),
        libraries: /** @type {string[]} */ ([]), sdkObjects: /** @type {{object: string}[]} */ ([]), dpcppComponent: f.target}};
    const before = JSON.stringify(sdk), result = nativeSdkLinkInputs(sdk);
    assert.equal(result.sdkObjects.length, 3); assert.equal(result.links[1], path.join(f.target, "lib/libmom-sycl-static.a"));
    assert.ok(result.hashes[path.join(f.target, "component.json")]); for (const row of result.sdkObjects) {assert.ok(result.hashes[row.object]);}
    assert.equal(JSON.stringify(sdk), before);
    sdk.sdk.links.push(path.join(f.build, "lib/libur_loader.a"));
    assert.throws(() => nativeSdkLinkInputs(sdk), /Do not mix/);
  });
  test("a mismatched Ninja is rejected before it can touch the old build log", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t); fs.appendFileSync(f.ninja, "throw Error('must not run');\n"); assert.throws(f.exportIt, /matching Ninja identity/); assert.equal(fs.existsSync(f.target), false);
  });
  test("completed exports are never overwritten", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t); f.exportIt(); assert.throws(f.exportIt, /must be new/);
  });
  test("changed staged static library bytes fail admission", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t); f.exportIt(); fs.appendFileSync(path.join(f.target, "lib/libur_loader.a"), "changed"); assert.throws(() => dpcppComponentInputs(f.target), /bytes\/type changed/);
  });
  test("staged symlinks cannot redirect static SDK inputs", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t); f.exportIt(); const file = path.join(f.target, "objects/source-jit.o"); fs.unlinkSync(file); fs.symlinkSync(f.jit, file);
    assert.throws(() => dpcppComponentInputs(f.target), /bytes\/type changed/);
  });
  test("unrecorded staged files cannot become hidden linker inputs", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t); f.exportIt(); fs.writeFileSync(path.join(f.target, "lib/extra.a"), "!<arch>\n"); assert.throws(() => dpcppComponentInputs(f.target), /Unrecorded/);
  });
  test("wrong target/pins cannot certify another runtime cohort", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t); f.exportIt(); const file = path.join(f.target, "component.json"), r = JSON.parse(fs.readFileSync(file, "utf8"));
    r.pins.source = "0".repeat(40); fs.writeFileSync(file, JSON.stringify(r)); assert.throws(() => dpcppComponentInputs(f.target), /receipt/);
  });
  test("a malicious receipt cannot add another object role", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t); f.exportIt(); const file = path.join(f.target, "component.json"), r = JSON.parse(fs.readFileSync(file, "utf8"));
    r.files["objects/extra.o"] = "0".repeat(64); fs.writeFileSync(file, JSON.stringify(r)); assert.throws(() => dpcppComponentInputs(f.target), /membership/);
  });
  test("zero-copy-looking shared libraries cannot masquerade as static archives", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t); f.exportIt(); const file = path.join(f.target, "lib/libur_loader.a"), receipt = path.join(f.target, "component.json"), r = JSON.parse(fs.readFileSync(receipt, "utf8"));
    fs.writeFileSync(file, "!<thin>\n"); r.files["lib/libur_loader.a"] = fileHash(file); fs.writeFileSync(receipt, JSON.stringify(r));
    assert.throws(() => dpcppComponentInputs(f.target), /bytes\/type changed/);
  });
});
