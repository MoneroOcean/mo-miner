"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {buildNativeArtifact} = require("../scripts/build-release"), {fileHash} = require("../scripts/build-source-inputs");
const core = require("../scripts/build-native-windows-core"), gpu = require("../scripts/build-native-windows-gpu"), controller = require("../scripts/build-native-windows-controller");
const packer = require("../scripts/pack-native-release");

/** Simulate only producer boundaries; these tests never compile or execute a candidate.
 * @param {import("node:test").TestContext} t */
function fixture(t) {
  const originalPlatform = Object.getOwnPropertyDescriptor(process, "platform");
  assert.ok(originalPlatform); Object.defineProperty(process, "platform", {value: "win32", configurable: true});
  t.after(() => Object.defineProperty(process, "platform", originalPlatform));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-source-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  fs.writeFileSync(path.join(root, "git.exe"), "fixture");
  const configFile = path.join(root, "release.json"), output = path.join(root, "output"), coreFile = path.join(root, "core.json");
  fs.writeFileSync(coreFile, '{"version":1}');
  const profiles = ["oneapi", "dpcpp-opencl", "dpcpp", "acpp-cuda", "acpp-hip"], files = Object.fromEntries(profiles.map(profile => {
    const file = path.join(root, profile + ".json"); fs.writeFileSync(file, JSON.stringify({version: 1, profile,
      ...(profile.startsWith("acpp-") ? {acpp: {git: path.join(root, "git.exe")}} : {})})); return [profile, file];
  }));
  const descriptor = {version: 1, windowsSdk: {}, application: {defaultCompiler: "oneapi", tools: {}, sdk: {}}, nativeSources: {core: coreFile, gpu: files}};
  const write = () => fs.writeFileSync(configFile, JSON.stringify(descriptor)); write();
  const steps = /** @type {string[]} */ ([]), state = {missingSdk: false, changeAfterCore: false, changeAfterGpu: false, badCore: false, badGpu: false, failController: false, changeGenerated: false, changeDuringPack: false, missingPackingReceipt: false, missingPackingLog: false};
  const tool = path.join(root, "packer"); fs.writeFileSync(tool, "packer");
  const packing = {upx: tool, upxSha256: fileHash(tool)};
  t.mock.method(core, "validateWindowsCoreDescriptor", /** @type {typeof core.validateWindowsCoreDescriptor} */ (/** @type {unknown} */ ((/** @type {unknown} */ value) => value)));
  t.mock.method(gpu, "validateWindowsGpuConfig", /** @type {typeof gpu.validateWindowsGpuConfig} */ (/** @type {unknown} */ ((/** @type {unknown} */ value) => value)));
  t.mock.method(controller, "validateWindowsControllerConfig", /** @type {typeof controller.validateWindowsControllerConfig} */ (/** @type {unknown} */ ((/** @type {unknown} */ value) => value)));
  t.mock.method(controller, "explicitInputFiles", () => state.missingSdk ? [path.join(root, "missing-sdk.lib")] : []);
  t.mock.method(core, "buildNativeWindowsCore", /** @type {typeof core.buildNativeWindowsCore} */ (/** @type {unknown} */ ((/** @type {string} */ input, /** @type {string} */ out) => {
    steps.push("core"); assert.equal(input, coreFile); fs.mkdirSync(out); const archive = path.join(out, "cpu-core.lib"); fs.writeFileSync(archive, "core");
    if (state.changeAfterCore) {fs.appendFileSync(files["oneapi"] || "", " ");}
    return {archive: {path: state.badCore ? path.join(out, "wrong.lib") : archive}};
  })));
  t.mock.method(gpu, "buildNativeWindowsGpu", /** @type {typeof gpu.buildNativeWindowsGpu} */ (/** @type {unknown} */ ((/** @type {string} */ input, /** @type {string} */ out) => {
    const selected = JSON.parse(fs.readFileSync(input, "utf8")), profile = selected.profile; steps.push(profile); fs.mkdirSync(out);
    const record = {profile, factory: state.badGpu ? "mom_native_gpu_wrong" : "mom_native_gpu_" + profile.replaceAll("-", "_")};
    fs.writeFileSync(path.join(out, "result.json"), JSON.stringify(record));
    if (state.changeAfterGpu) {fs.appendFileSync(coreFile, " ");} return record;
  })));
  t.mock.method(controller, "buildNativeWindowsController", /** @type {typeof controller.buildNativeWindowsController} */ (/** @type {unknown} */ ((/** @type {string} */ input, /** @type {string} */ out) => {
    steps.push("controller"); if (state.failController) {throw new Error("fixture controller failed");}
    const generated = JSON.parse(fs.readFileSync(input, "utf8")); assert.equal(generated.nativeSources, undefined);
    assert.equal(generated.application.core, path.join(out + ".components", "core", "cpu-core.lib")); assert.equal(generated.application.gpuReceipts.length, 5);
    fs.mkdirSync(out); const file = path.join(out, "mom.exe"); fs.writeFileSync(file, "fixture");
    const artifact = {path: file, bytes: fs.statSync(file).size, sha256: fileHash(file)};
    fs.writeFileSync(path.join(out, "result.json"), JSON.stringify({version: 1, configHash: fileHash(input), artifact,
      target: generated.target || "x86_64-pc-windows-msvc", buildOnly: true, publicationQualified: false, executed: false}, null, 2) + "\n");
    if (state.changeGenerated) {fs.appendFileSync(input, " ");}
    return {artifact, application: {}, executed: false, buildOnly: true};
  })));
  t.mock.method(packer, "packNativeRelease", /** @type {typeof packer.packNativeRelease} */ (/** @type {unknown} */ ((/** @type {Parameters<typeof packer.packNativeRelease>[0]} */ options) => {
    steps.push("pack"); fs.mkdirSync(options.output); const file = path.join(options.output, "mom.exe"); fs.copyFileSync(options.source, file);
    if (state.changeDuringPack) {fs.appendFileSync(path.join(path.dirname(options.source) + ".components", "controller.json"), " ");}
    const logs = Object.fromEntries(["pack", "integrity", "restore"].map(stage => {
      const file = path.join(options.output, stage + ".log"); fs.writeFileSync(file, "fixture " + stage + "\n", {flag: "wx"});
      return [stage, {path: file, sha256: fileHash(file)}];
    }));
    const result = {source: {path: options.source, bytes: fs.statSync(options.source).size, sha256: fileHash(options.source)},
      artifact: {path: file, bytes: fs.statSync(file).size, sha256: fileHash(file)}, tool: {path: options.upx, sha256: options.upxSha256},
      restoredSha256: fileHash(options.source), logs, buildOnly: true, publicationQualified: false};
    if (!state.missingPackingReceipt) {fs.writeFileSync(path.join(options.output, "result.json"), JSON.stringify(result), {flag: "wx"});}
    if (state.missingPackingLog) {fs.unlinkSync(path.join(options.output, "restore.log"));}
    return result;
  })));
  return {root, configFile, output, coreFile, files, descriptor, write, steps, state, run: (pack = false) => buildNativeArtifact(configFile, output, pack ? packing : undefined, {})};
}

test.describe("Windows native source-build orchestration", () => {
  test("automatic Windows Core production does not start on a Linux cross-build host", {skip: process.platform === "win32"}, t => {
    const f = fixture(t); Object.defineProperty(process, "platform", {value: "linux", configurable: true});
    assert.throws(f.run, /requires a Windows build host/); assert.equal(f.steps.length, 0); assert.ok(!fs.existsSync(f.output + ".components"));
  });
  test("existing Core and all original GPU builders run before the shared controller", t => {
    const f = fixture(t), result = f.run();
    assert.deepEqual(f.steps, ["core", "oneapi", "dpcpp-opencl", "acpp-cuda", "dpcpp", "acpp-hip", "controller"]);
    assert.equal(result.artifact.path, path.join(f.output, "mom.exe")); const receipt = JSON.parse(fs.readFileSync(path.join(f.output, "entrypoint.json"), "utf8"));
    assert.equal(Object.keys(receipt.componentHashes).length, 7); for (const [file, hash] of Object.entries(receipt.componentHashes)) {assert.equal(fileHash(file), hash);}
    assert.equal(receipt.generatedDescriptor.sha256, fileHash(receipt.generatedDescriptor.path));
    assert.deepEqual(receipt.producerReceipt, {path: path.join(f.output, "result.json"), sha256: fileHash(path.join(f.output, "result.json"))});
    assert.equal(receipt.componentHashes[receipt.producerReceipt.path], receipt.producerReceipt.sha256);
    assert.equal(receipt.publicationQualified, false);
  });
  test("symlinked output parents cannot redirect component builds into the repository", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), link = path.join(f.root, "source-link"); fs.symlinkSync(path.join(__dirname, ".."), link, "dir");
    assert.throws(() => buildNativeArtifact(f.configFile, path.join(link, "native-test")), /outside the repository/); assert.equal(f.steps.length, 0);
  });
  test("missing GPU profiles fail before any workspace or compiler", t => {
    const f = fixture(t); Reflect.deleteProperty(f.descriptor.nativeSources.gpu, "acpp-hip"); f.write(); assert.throws(f.run, /every maintained GPU/);
    assert.equal(f.steps.length, 0); assert.ok(!fs.existsSync(f.output + ".components"));
  });
  test("unexpected GPU profiles are not silently ignored", t => {
    const f = fixture(t); Object.assign(f.descriptor.nativeSources.gpu, {other: f.coreFile}); f.write(); assert.throws(f.run, /every maintained GPU/); assert.equal(f.steps.length, 0);
  });
  test("source builds cannot also provide prebuilt Core or graph receipts", t => {
    const f = fixture(t); Object.assign(f.descriptor.application, {core: f.coreFile}); f.write(); assert.throws(f.run, /source-build descriptor/); assert.equal(f.steps.length, 0);
  });
  test("GPU descriptors must identify their selected profile", t => {
    const f = fixture(t); fs.writeFileSync(f.files["oneapi"] || "", '{"version":1,"profile":"other"}'); assert.throws(f.run, /profile differs/); assert.equal(f.steps.length, 0);
  });
  test("relative component descriptors are refused", t => {
    const f = fixture(t); f.descriptor.nativeSources.core = "relative"; f.write(); assert.throws(f.run, /source-build descriptor/); assert.equal(f.steps.length, 0);
  });
  test("missing controller SDK files fail before component compilation", t => {const f = fixture(t); f.state.missingSdk = true; assert.throws(f.run, /static SDK input/); assert.equal(f.steps.length, 0);});
  test("missing Windows AdaptiveCpp Git fails before component compilation", t => {
    const f = fixture(t), file = f.files["acpp-hip"] || "", config = JSON.parse(fs.readFileSync(file, "utf8")); delete config.acpp.git; fs.writeFileSync(file, JSON.stringify(config));
    assert.throws(f.run, /explicit absolute Git/); assert.equal(f.steps.length, 0);
  });
  test("nonexistent Windows AdaptiveCpp Git fails before component compilation", t => {
    const f = fixture(t); fs.unlinkSync(path.join(f.root, "git.exe")); assert.throws(f.run, /Missing Windows AdaptiveCpp Git/); assert.equal(f.steps.length, 0);
  });
  test("existing component workspaces are preserved", t => {const f = fixture(t); fs.mkdirSync(f.output + ".components"); assert.throws(f.run, /workspace/); assert.equal(f.steps.length, 0);});
  test("changed source descriptors stop before GPU compilation", t => {const f = fixture(t); f.state.changeAfterCore = true; assert.throws(f.run, /source descriptor changed/); assert.deepEqual(f.steps, ["core"]);});
  test("changed source descriptors stop before controller compilation", t => {const f = fixture(t); f.state.changeAfterGpu = true; assert.throws(f.run, /source descriptor changed/); assert.deepEqual(f.steps, ["core", "oneapi"]);});
  test("unexpected Core archive paths cannot enter the controller", t => {const f = fixture(t); f.state.badCore = true; assert.throws(f.run, /unexpected archive/); assert.deepEqual(f.steps, ["core"]);});
  test("unexpected GPU factories cannot enter the controller", t => {const f = fixture(t); f.state.badGpu = true; assert.throws(f.run, /unexpected graph/); assert.deepEqual(f.steps, ["core", "oneapi"]);});
  test("controller failures are preserved without a release receipt", t => {const f = fixture(t); f.state.failController = true; assert.throws(f.run, /controller failed/); assert.ok(!fs.existsSync(path.join(f.output, "entrypoint.json")));});
  test("a changed generated descriptor fails before packing", t => {const f = fixture(t); f.state.changeGenerated = true; assert.throws(() => f.run(true), /descriptor changed during compilation/); assert.ok(!f.steps.includes("pack"));});
  test("a generated descriptor changed during packing cannot receive a release receipt", t => {const f = fixture(t); f.state.changeDuringPack = true; assert.throws(() => f.run(true), /descriptor changed during packing/); assert.ok(!fs.existsSync(path.join(f.output, "entrypoint.json")));});
});

test.describe("Windows source-build packing evidence", () => {
  test("packed builds bind fresh components and the complete packing receipt", t => {
    const f = fixture(t), result = f.run(true), receipt = JSON.parse(fs.readFileSync(path.join(f.output, "entrypoint.json"), "utf8"));
    assert.equal(f.steps.at(-1), "pack"); assert.equal(result.artifact.path, path.join(f.output, "mom.exe"));
    assert.equal(receipt.generatedDescriptor.path, path.join(f.output + ".unpacked.components", "controller.json"));
    assert.equal(receipt.generatedDescriptor.sha256, fileHash(receipt.generatedDescriptor.path));
    assert.equal(receipt.producerReceipt.path, path.join(f.output + ".unpacked", "result.json"));
    assert.equal(receipt.componentHashes[receipt.producerReceipt.path], receipt.producerReceipt.sha256);
    assert.equal(receipt.packingReceipt.path, path.join(f.output, "result.json"));
    assert.equal(receipt.packingReceipt.sha256, fileHash(receipt.packingReceipt.path));
  });
  for (const flag of /** @type {const} */ (["missingPackingReceipt", "missingPackingLog"])) {
    test(`rejects ${flag} without a release entrypoint`, t => {
      const f = fixture(t); f.state[flag] = true;
      assert.throws(() => f.run(true), /packer did not write a regular result receipt|packing log is missing or changed/);
      assert.ok(!fs.existsSync(path.join(f.output, "entrypoint.json")));
    });
  }
});
