"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {nativeSourceInputs, fileHash} = require("../scripts/build-source-inputs");
const {nativeApplicationGpu} = require("../scripts/probe-native-controller");
const {validateWindowsApplication, windowsApplicationInputs, verifyWindowsApplicationFiles, windowsResponseFile} = require("../scripts/native-windows-application");
const nativeCudaImages = require("../scripts/native-cuda-images");
const hostComponentReader = require("../scripts/native-windows-host-component");
const nvrtcProducer = require("../scripts/prepare-native-nvrtc-sdk");
const {coffObject: object, fixture: componentFixture, vendorNvrtcFixture} = require("./common/native-windows-component");
const repo = path.resolve(__dirname, ".."), states = {cpu: {}, gpu: {}};
const factories = nativeApplicationGpu(fs.readFileSync(path.join(repo, "runtime/gpu-dispatch.cpp"), "utf8"), {defaultCompiler: "oneapi", flags: [], links: []}).factories;

/** @param {string} file @param {unknown} value */
function json(file, value) {fs.writeFileSync(file, JSON.stringify(value));}
/** @param {import("node:test").TestContext} t @param {{embeddedCudaImages?: boolean}} [options] */
function fixture(t, options = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-application-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  /** @param {string} name @param {string | Buffer} [data] */
  const file = (name, data = "!<arch>\n") => {const selected = path.join(root, name); fs.writeFileSync(selected, data); return selected;};
  const core = file("cpu-core.lib");
  const cudaImages = options.embeddedCudaImages ? file("cuda-images.inc", "synthetic CUDA images\n") : undefined;
  if (cudaImages) {file("cuda-images.inc.json", "synthetic CUDA image receipt\n");}
  json(path.join(root, "inputs.json"), {target: "win32", gpu: true, portable: true, objectSha256: fileHash(core), sourceState: states.cpu,
    archive: {stubs: false, sha256: fileHash(core), coffTableCount: 1, machineCount: 1, x64MachineCount: 1}});
  const gpuReceipts = factories.map((factory, index) => {
    const profile = factory.slice("mom_native_gpu_".length).replaceAll("_", "-"), bytes = object();
    // Structural COFF fixtures include one unique HCF in their section, not executable device code.
    const payload = profile.startsWith("acpp-") ? Buffer.concat([bytes, Buffer.from(
      `object-id=${index + 1}\ngenerator=hipSYCL SSCP\nstart=0\nsize=4\n__acpp_hcf_binary_appendix`, "latin1"), Buffer.from([0x42, 0x43, 0xc0, 0xde])]) : bytes;
    payload.writeUInt32LE(payload.length - 60, 36);
    const selected = file(index + ".obj", payload);
    const objects = [{path: selected, sha256: fileHash(selected)}], isolation = path.join(root, index + "-isolation.json"), configFile = file(index + "-sdk.json", "{}");
    const checks = Object.fromEntries(["nonDirectivePayloadsPreserved", "nonDirectiveMetadataPreserved", "comdatSelectionPreserved", "associativeXdataPreserved", "linkerDirectiveReferencesPreserved", "externalUndefinedImportsPreserved", "privateUndefinedReferencesRenamed"].map(key => [key, true]));
    json(isolation, {prefix: "mom_" + profile.replaceAll("-", "_") + "_gpu_", entryPoints: [factory], outputs: objects, checks: {...checks, oldCollisionDefinitions: 0, oldCollisionReferences: 0}});
    const graph = path.join(root, index + "-result.json");
    const sdkInputs = {[configFile]: fileHash(configFile)};
    if (cudaImages && ["dpcpp", "acpp-cuda"].includes(profile)) {
      sdkInputs[cudaImages] = fileHash(cudaImages); sdkInputs[cudaImages + ".json"] = fileHash(cudaImages + ".json");
    }
    const compiled = cudaImages && ["dpcpp", "acpp-cuda"].includes(profile) ? [
      {file: "sycl/cn-gpu/cn-gpu.cpp", flags: ["-DMOM_EMBEDDED_CUDA_IMAGES"]},
      {file: "sycl/pearlhash/pearlhash.cpp", flags: ["-DMOM_EMBEDDED_CUDA_IMAGES"]},
    ] : undefined;
    json(graph, {factory, profile, sourceState: states.gpu, isolation, objects, sdkInputs, compiled, configFile, configSha256: fileHash(configFile)}); return graph;
  });
  /** @type {import("../scripts/native-windows-application").WindowsApplication} */
  const app = {core, gpuReceipts, defaultCompiler: "oneapi", tools: {llvmObjcopy: file("objcopy", "tool"), llvmAr: file("ar", "tool")},
    sdk: {libraries: [file("runtime.lib")], wholeArchives: [file("constructors.lib")], objects: [file("sdk.obj", object())], includeRoots: [root], nvmlHeader: file("nvml.h", "header"), hipDriverLibrary: file("amdhip64.lib"),
      ...(cudaImages ? {cudaImages} : {})}};
  /** @param {string} target @param {(value: any) => void} change */
  const edit = (target, change) => {const value = JSON.parse(fs.readFileSync(target, "utf8")); change(value); json(target, value);};
  return {root, app, edit, cudaImages};
}

/** @param {ReturnType<typeof fixture>} f */
function compactGpuHandoffs(f) {
  const states = new Map(), receipts = [], pins = [];
  for (const receipt of f.app.gpuReceipts) {
    const original = JSON.parse(fs.readFileSync(receipt, "utf8")), profile = original.profile,
      root = path.join(f.root, "mom-native-windows-gpu-" + profile + "-123456789-1"), output = path.join(root, "output"),
      isolated = path.join(output, "isolated"), configFile = path.join(root, "config.json"), graphFile = path.join(output, "result.json"),
      isolationFile = path.join(isolated, "result.json"), handoffFile = path.join(root, "handoff.json");
    fs.mkdirSync(isolated, {recursive: true}); json(configFile, {profile});
    const objects = original.objects.map((/** @type {{path:string}} */ row, /** @type {number} */ index) => {
      const objectFile = path.join(isolated, `object-${String(index).padStart(4, "0")}.obj`);
      fs.copyFileSync(row.path, objectFile); return {path: objectFile, bytes: fs.statSync(objectFile).size, sha256: fileHash(objectFile)};
    });
    const oldConfigFile = original.configFile;
    delete original.sdkInputs[oldConfigFile];
    original.sdkInputs = Object.fromEntries([
      ["/historical/sdk/z-tool.exe", "a".repeat(64)], [configFile, fileHash(configFile)],
      ...Object.entries(original.sdkInputs), ["/historical/sdk/a-header.h", "b".repeat(64)],
    ]);
    original.configFile = configFile; original.configSha256 = fileHash(configFile); original.isolation = isolationFile; original.objects = objects;
    const sourceIsolation = JSON.parse(fs.readFileSync(JSON.parse(fs.readFileSync(receipt, "utf8")).isolation, "utf8"));
    sourceIsolation.outputs = objects; json(isolationFile, sourceIsolation); json(graphFile, original);
    const handoffFiles = [configFile, graphFile, isolationFile, ...objects.map((/** @type {{path:string}} */ row) => row.path)].map(file => ({
      path: path.relative(root, file).split(path.sep).join("/"), bytes: fs.statSync(file).size, sha256: fileHash(file),
    })).sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
    const handoff = {format: "mom-windows-gpu-handoff-v1", runAttempt: "123456789-1", profile, files: handoffFiles,
      sdkInputs: Object.fromEntries(Object.keys(original.sdkInputs).sort().map(file => [file, original.sdkInputs[file]])),
      buildOnly: true, executed: false, publicationQualified: false};
    json(handoffFile, handoff); receipts.push(graphFile);
    const pin = {profile, runAttempt: handoff.runAttempt, receiptSha256: fileHash(graphFile), configSha256: fileHash(configFile), handoffSha256: fileHash(handoffFile)};
    pins.push(pin); states.set(profile, {root, graphFile, isolationFile, configFile, handoffFile, graph: original, handoff, pin, objects});
  }
  f.app.gpuReceipts = receipts; f.app.gpuHandoffs = pins;
  /** Re-pin a fixture after a deliberate graph mutation; `syncSdkMap` controls the independent handoff-map check. */
  const resign = (/** @type {string} */ profile, /** @type {boolean} */ syncSdkMap = true) => {
    const state = states.get(profile); assert.ok(state);
    json(state.graphFile, state.graph);
    if (syncSdkMap) {state.handoff.sdkInputs = Object.fromEntries(Object.keys(state.graph.sdkInputs).sort().map(file => [file, state.graph.sdkInputs[file]]));}
    const graphMember = state.handoff.files.find((/** @type {{path:string}} */ row) => row.path === "output/result.json");
    assert.ok(graphMember); graphMember.bytes = fs.statSync(state.graphFile).size; graphMember.sha256 = fileHash(state.graphFile);
    json(state.handoffFile, state.handoff);
    state.pin.receiptSha256 = fileHash(state.graphFile); state.pin.configSha256 = fileHash(state.configFile); state.pin.handoffSha256 = fileHash(state.handoffFile);
  };
  return {states, resign};
}

/** @param {import("node:test").TestContext} t */
function mockCudaImageAdmission(t) {
  t.mock.method(nativeCudaImages, "cudaImageInputs", (/** @type {string} */ file, /** @type {string} */ selectedRoot) => {
    assert.equal(selectedRoot, repo);
    return {files: {[file]: fileHash(file), [file + ".json"]: fileHash(file + ".json")}, manifest: {}};
  });
}

/** @param {string[]} receipts @param {string} profile */
function receiptFor(receipts, profile) {
  const file = receipts.find(selected => JSON.parse(fs.readFileSync(selected, "utf8")).profile === profile);
  assert.ok(file, `missing ${profile} receipt`);
  return file;
}

test.describe("Complete Windows native application admission", () => {
  test("final Windows admission rejects HCF collisions across isolated ACPP profiles", t => {
    const f = fixture(t), receipts = f.app.gpuReceipts.filter(file => JSON.parse(fs.readFileSync(file, "utf8")).profile.startsWith("acpp-"));
    assert.equal(receipts.length, 2);
    const first = JSON.parse(fs.readFileSync(receipts[0] || "", "utf8")), secondFile = receipts[1] || "",
      second = JSON.parse(fs.readFileSync(secondFile, "utf8"));
    fs.copyFileSync(first.objects[0].path, second.objects[0].path);
    second.objects[0].sha256 = fileHash(second.objects[0].path);
    f.edit(second.isolation, value => {value.outputs = second.objects;}); json(secondFile, second);
    assert.throws(() => windowsApplicationInputs(f.app, repo), /Duplicate HCF object ID/);
  });
  test("JIT identity distinguishes ordinary from whole-archive selection", t => {
    const f = fixture(t), extra = path.join(f.root, "ordinary-extra.lib");
    fs.copyFileSync(f.app.sdk.libraries[0] || "", extra);
    f.app.sdk.libraries.unshift(extra);
    const before = windowsApplicationInputs(f.app, repo).jitCacheTag;
    const moved = f.app.sdk.libraries.pop(); assert.ok(moved);
    f.app.sdk.wholeArchives.unshift(moved);
    // Concatenated bytes/order are unchanged; only the linker's selection mode differs.
    assert.notEqual(windowsApplicationInputs(f.app, repo).jitCacheTag, before);
  });
  test.before(() => {states.cpu = nativeSourceInputs(repo, "cpu"); states.gpu = nativeSourceInputs(repo, "gpu");});
  test("an installed component binds its manifest and complete payloads to the application", t => {
    const f = fixture(t), component = componentFixture(t), installed = component.produce();
    f.app.sdk.dpcppComponent = component.output; const result = windowsApplicationInputs(f.app, repo);
    for (const [file, hash] of Object.entries(installed.files)) {assert.equal(result.files[file], hash);}
    verifyWindowsApplicationFiles(result.files);
    component.edit(r => {r.producer.sha256 = "2".repeat(64);});
    assert.throws(() => verifyWindowsApplicationFiles(result.files), /input changed/);
  });
  test("embedded CUDA images bind both CUDA factory receipts and the source cohort", t => {
    mockCudaImageAdmission(t);
    const f = fixture(t, {embeddedCudaImages: true}), result = windowsApplicationInputs(f.app, repo);
    assert.ok(f.cudaImages);
    assert.equal(result.files[f.cudaImages], fileHash(f.cudaImages));
    assert.equal(result.files[f.cudaImages + ".json"], fileHash(f.cudaImages + ".json"));
    for (const profile of ["dpcpp", "acpp-cuda"]) {
      const graphFile = receiptFor(f.app.gpuReceipts, profile), graph = JSON.parse(fs.readFileSync(graphFile, "utf8"));
      assert.equal(graph.sdkInputs[f.cudaImages], fileHash(f.cudaImages));
      assert.equal(graph.sdkInputs[f.cudaImages + ".json"], fileHash(f.cudaImages + ".json"));
      for (const source of ["sycl/cn-gpu/cn-gpu.cpp", "sycl/pearlhash/pearlhash.cpp"]) {
        const compiled = graph.compiled.find((/** @type {{file: string, flags: string[]}} */ row) => row.file === source);
        assert.ok(compiled?.flags.includes("-DMOM_EMBEDDED_CUDA_IMAGES"));
      }
    }
    verifyWindowsApplicationFiles(result.files);
  });
  test("embedded image admission rejects a mixed or incomplete CUDA factory cohort", t => {
    mockCudaImageAdmission(t);
    const missingInput = fixture(t, {embeddedCudaImages: true}), acpp = receiptFor(missingInput.app.gpuReceipts, "acpp-cuda");
    assert.ok(missingInput.cudaImages);
    missingInput.edit(acpp, graph => {delete graph.sdkInputs[missingInput.cudaImages + ".json"];});
    assert.throws(() => windowsApplicationInputs(missingInput.app, repo), /did not consume the selected image cohort/);

    const mismatchedInput = fixture(t, {embeddedCudaImages: true}), mismatchedReceipt = receiptFor(mismatchedInput.app.gpuReceipts, "dpcpp"),
      imagePath = mismatchedInput.cudaImages;
    assert.ok(imagePath);
    mismatchedInput.edit(mismatchedReceipt, graph => {graph.sdkInputs[imagePath] = "0".repeat(64);});
    assert.throws(() => windowsApplicationInputs(mismatchedInput.app, repo), /component hash differs/);

    const mixedFlags = fixture(t, {embeddedCudaImages: true}), dpcpp = receiptFor(mixedFlags.app.gpuReceipts, "dpcpp");
    mixedFlags.edit(dpcpp, graph => {
      graph.compiled.find((/** @type {{file: string, flags: string[]}} */ row) => row.file === "sycl/pearlhash/pearlhash.cpp").flags = ["-DMOM_STATIC_NVRTC"];
    });
    assert.throws(() => windowsApplicationInputs(mixedFlags.app, repo), /image-only route/);

    const unbound = fixture(t, {embeddedCudaImages: true}); delete unbound.app.sdk.cudaImages;
    assert.throws(() => windowsApplicationInputs(unbound.app, repo), /require the matching sdk.cudaImages cohort/);
  });
  test("embedded image mode refuses static NVRTC archives and the transitional SDK together", t => {
    mockCudaImageAdmission(t);
    const mixedSdk = fixture(t, {embeddedCudaImages: true});
    mixedSdk.app.sdk.nvrtcSdk = "/vendor/nvrtc-sdk";
    assert.throws(() => validateWindowsApplication(mixedSdk.app, repo), /Invalid Windows application SDK/);

    const staticLibrary = fixture(t, {embeddedCudaImages: true}), nvrtc = path.join(staticLibrary.root, "nvrtc_static.lib");
    fs.writeFileSync(nvrtc, "!<arch>\n"); staticLibrary.app.sdk.libraries.push(nvrtc);
    assert.throws(() => windowsApplicationInputs(staticLibrary.app, repo), /static NVRTC libraries/);

    const headerDependency = fixture(t, {embeddedCudaImages: true}), headerReceipt = receiptFor(headerDependency.app.gpuReceipts, "acpp-cuda"),
      nvrtcHeader = path.join(headerDependency.root, "nvrtc.h");
    fs.writeFileSync(nvrtcHeader, "static NVRTC header dependency\n");
    headerDependency.edit(headerReceipt, graph => {graph.sdkInputs[nvrtcHeader] = fileHash(nvrtcHeader);});
    assert.throws(() => windowsApplicationInputs(headerDependency.app, repo), /static NVRTC dependency/);
  });
  test("the optional host component binds its manifest and every staged archive", t => {
    const f = fixture(t), hostRoot = path.join(f.root, "host-component"), library = path.join(hostRoot, "lib", "acpp.lib"),
      manifest = path.join(hostRoot, "component.json");
    fs.mkdirSync(path.dirname(library), {recursive: true}); fs.writeFileSync(library, "!<arch>\n"); fs.writeFileSync(manifest, "host component receipt\n");
    const hostFiles = {[manifest]: fileHash(manifest), [library]: fileHash(library)};
    t.mock.method(hostComponentReader, "windowsHostComponentInputs", (/** @type {string} */ root) => {
      assert.equal(root, hostRoot); return {link: {libraries: [library]}, files: hostFiles, provenance: {},
        runtimeInputs: {amdhip64: f.app.sdk.hipDriverLibrary, cudartStatic: library}};
    });
    f.app.sdk.hostComponent = hostRoot;
    const result = windowsApplicationInputs(f.app, repo);
    assert.equal(result.files[manifest], hostFiles[manifest]); assert.equal(result.files[library], hostFiles[library]);
    verifyWindowsApplicationFiles(result.files);
    fs.appendFileSync(library, "changed"); assert.throws(() => verifyWindowsApplicationFiles(result.files), /input changed/);
    fs.writeFileSync(library, "!<arch>\n"); fs.appendFileSync(manifest, "changed");
    assert.throws(() => verifyWindowsApplicationFiles(result.files), /input changed/);
  });
  test("vendor NVRTC receipt, headers, licenses, archives and tool enter the application input snapshot", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), vendor = vendorNvrtcFixture(f.root);
    t.mock.method(nvrtcProducer, "nvrtcSdkInputs", (/** @type {"linux" | "windows"} */ platform, /** @type {string} */ root) => {
      assert.equal(platform, "windows"); assert.equal(root, vendor.root);
      return {receipt: vendor.receipt, receiptFile: vendor.receiptFile, receiptSha256: vendor.receiptSha256, files: vendor.files};
    });
    f.app.sdk.nvrtcSdk = vendor.root; f.app.sdk.libraries.push(...vendor.libraries);
    const result = windowsApplicationInputs(f.app, repo);
    for (const [file, hash] of Object.entries(vendor.files)) {assert.equal(result.files[file], hash);}
    verifyWindowsApplicationFiles(result.files);
    for (const file of Object.keys(vendor.files).filter(file => file.startsWith(f.root + path.sep))) {
      const original = fs.readFileSync(file); fs.appendFileSync(file, "changed");
      assert.throws(() => verifyWindowsApplicationFiles(result.files), /input changed/);
      fs.writeFileSync(file, original);
    }
  });
  test("component bytes changed after admission cannot be rehashed as a new SDK generation", t => {
    const f = fixture(t), component = componentFixture(t), installed = component.produce(), open = fs.openSync;
    f.app.sdk.dpcppComponent = component.output; let changed = false;
    t.mock.method(fs, "openSync", /** @type {typeof fs.openSync} */ ((...args) => {
      if (!changed && args[0] === path.join(f.root, "inputs.json")) {changed = true; fs.appendFileSync(installed.link.libraries[0] || "", "changed");}
      return Reflect.apply(open, fs, args);
    }));
    assert.throws(() => windowsApplicationInputs(f.app, repo), /component hash differs/); assert.equal(changed, true);
  });
  test("a HIP import library cannot alias an archive supplied by the component", t => {
    const f = fixture(t), component = componentFixture(t), installed = component.produce();
    f.app.sdk.dpcppComponent = component.output; f.app.sdk.hipDriverLibrary = installed.link.libraries[0] || "";
    assert.throws(() => windowsApplicationInputs(f.app, repo), /Duplicate Windows application SDK archive/);
  });
  test("invalid selected component roots are rejected by the application schema", t => {
    const f = fixture(t);
    for (const value of [undefined, null, "relative", 1]) {
      f.app.sdk.dpcppComponent = /** @type {string} */ (value); assert.throws(() => validateWindowsApplication(f.app, repo), /Invalid Windows application SDK/);
    }
    delete f.app.sdk.dpcppComponent;
    for (const value of [undefined, null, "relative", 1]) {
      f.app.sdk.hostComponent = /** @type {string} */ (value); assert.throws(() => validateWindowsApplication(f.app, repo), /Invalid Windows application SDK/);
    }
    delete f.app.sdk.hostComponent;
    for (const value of [undefined, null, "relative", 1]) {
      f.app.sdk.nvrtcSdk = /** @type {string} */ (value); assert.throws(() => validateWindowsApplication(f.app, repo), /Invalid Windows application SDK/);
    }
    delete f.app.sdk.nvrtcSdk;
    f.app.sdk.hostComponent = "/valid/host-component"; /** @type {any} */ (f.app.sdk).unrecognized = true;
    assert.throws(() => validateWindowsApplication(f.app, repo), /Invalid Windows application SDK/);
  });
  test("admits exactly the maintained factory set with content-bound Core, GPU and SDK inputs", t => {
    const {app} = fixture(t), result = windowsApplicationInputs(app, repo);
    assert.deepEqual(new Set(result.graphs.map(row => row.factory)), new Set(factories)); assert.equal(result.objects.length, factories.length);
    assert.equal(result.core, app.core); assert.equal(result.definitions.length, factories.length + 1); verifyWindowsApplicationFiles(result.files);
  });
  test("compact GPU handoffs pin all profiles while retaining producer SDK hashes without reopening historical files", t => {
    const f = fixture(t), cohort = compactGpuHandoffs(f), result = windowsApplicationInputs(f.app, repo);
    assert.equal(result.graphs.length, factories.length); assert.equal(result.objects.length, factories.length);
    for (const {graphFile, handoffFile, objects} of cohort.states.values()) {
      assert.equal(result.files[graphFile], fileHash(graphFile)); assert.equal(result.files[handoffFile], fileHash(handoffFile));
      for (const row of objects) {assert.equal(result.files[row.path], row.sha256);}
    }
    assert.equal(Object.keys(result.files).some(file => file.startsWith("/historical/sdk/")), false);
    verifyWindowsApplicationFiles(result.files);
  });
  test("compact GPU handoffs reject a mixed same-run cohort", t => {
    const f = fixture(t); compactGpuHandoffs(f);
    const pins = f.app.gpuHandoffs; assert.ok(pins); const other = pins[0]; assert.ok(other); other.runAttempt = "123456789-2";
    assert.throws(() => validateWindowsApplication(f.app, repo), /every maintained profile/);
  });
  test("compact GPU handoffs continue to reject stale source, SDK-map, config and object mutations", t => {
    const staleSource = fixture(t), staleSourceCohort = compactGpuHandoffs(staleSource), sourceProfile = "oneapi",
      sourceState = staleSourceCohort.states.get(sourceProfile);
    assert.ok(sourceState); sourceState.graph.sourceState = {files: {}, graph: {}}; staleSourceCohort.resign(sourceProfile);
    assert.throws(() => windowsApplicationInputs(staleSource.app, repo), /GPU source inputs are stale/);

    const mismatchedSdk = fixture(t), sdkCohort = compactGpuHandoffs(mismatchedSdk), sdkProfile = "oneapi", sdkState = sdkCohort.states.get(sdkProfile);
    assert.ok(sdkState); sdkState.graph.sdkInputs["/historical/sdk/z-tool.exe"] = "c".repeat(64); sdkCohort.resign(sdkProfile, false);
    assert.throws(() => windowsApplicationInputs(mismatchedSdk.app, repo), /SDK map/);

    const changedConfig = fixture(t), configCohort = compactGpuHandoffs(changedConfig), configState = configCohort.states.get("oneapi");
    assert.ok(configState); fs.appendFileSync(configState.configFile, " ");
    assert.throws(() => windowsApplicationInputs(changedConfig.app, repo), /configuration|payload differs/);

    const changedObject = fixture(t), objectCohort = compactGpuHandoffs(changedObject), objectState = objectCohort.states.get("oneapi");
    assert.ok(objectState); fs.appendFileSync(objectState.objects[0]?.path || "", "changed");
    assert.throws(() => windowsApplicationInputs(changedObject.app, repo), /payload differs/);
  });
  test("rejects a missing profile rather than producing a smaller application", t => {
    const {app} = fixture(t); app.gpuReceipts.pop(); assert.throws(() => windowsApplicationInputs(app, repo), /every maintained GPU/);
  });
  test("duplicate factories cannot replace a missing profile", t => {
    const {app} = fixture(t); app.gpuReceipts[1] = app.gpuReceipts[0] || "";
    assert.throws(() => windowsApplicationInputs(app, repo), /GPU receipts/);
  });
  test("Core source changes require rebuild, not a restamped object hash", t => {
    const selected = fixture(t); selected.edit(path.join(selected.root, "inputs.json"), value => {value.sourceState = {files: {}, graph: {}};});
    assert.throws(() => windowsApplicationInputs(selected.app, repo), /Core source inputs are stale/);
  });
  test("host-tuned, stubbed or non-x64 Core archives are rejected", t => {
    const selected = fixture(t), source = path.join(selected.root, "inputs.json"), original = fs.readFileSync(source);
    /** @type {Array<(value: any) => void>} */
    const changes = [value => {value.portable = false;}, value => {value.archive.stubs = true;}, value => {value.archive.x64MachineCount = 0;}];
    for (const change of changes) {
      fs.writeFileSync(source, original); selected.edit(source, change); assert.throws(() => windowsApplicationInputs(selected.app, repo), /complete portable x64 Core/);
    }
  });
  test("GPU source, isolated object and consumed SDK changes are independently detected", t => {
    const selected = fixture(t), source = selected.app.gpuReceipts[0]; assert.ok(source); const original = fs.readFileSync(source);
    selected.edit(source, value => {value.sourceState = {files: {}, graph: {}};});
    assert.throws(() => windowsApplicationInputs(selected.app, repo), /GPU source inputs are stale/); fs.writeFileSync(source, original);
    const graph = JSON.parse(original.toString()), objectFile = graph.objects[0].path, bytes = fs.readFileSync(objectFile); fs.appendFileSync(objectFile, "changed");
    assert.throws(() => windowsApplicationInputs(selected.app, repo), /component hash differs/); fs.writeFileSync(objectFile, bytes);
    fs.appendFileSync(graph.configFile, " "); assert.throws(() => windowsApplicationInputs(selected.app, repo), /component hash differs/);
  });
  test("missing COMDAT, unwind or relocation preservation cannot pass final-link admission", t => {
    const selected = fixture(t), graph = JSON.parse(fs.readFileSync(selected.app.gpuReceipts[0] || "", "utf8")), original = fs.readFileSync(graph.isolation);
    for (const key of ["comdatSelectionPreserved", "associativeXdataPreserved", "privateUndefinedReferencesRenamed"]) {
      fs.writeFileSync(graph.isolation, original); selected.edit(graph.isolation, value => {value.checks[key] = false;});
      assert.throws(() => windowsApplicationInputs(selected.app, repo), /isolation receipt differs/);
    }
  });
  test("a changed isolation factory or object list is not hidden by valid component hashes", t => {
    const selected = fixture(t), graph = JSON.parse(fs.readFileSync(selected.app.gpuReceipts[0] || "", "utf8"));
    selected.edit(graph.isolation, value => {value.entryPoints = ["mom_native_gpu_unknown"];});
    assert.throws(() => windowsApplicationInputs(selected.app, repo), /isolation receipt differs/);
  });
  test("SDK objects cannot alias an admitted GPU object", t => {
    const selected = fixture(t), graph = JSON.parse(fs.readFileSync(selected.app.gpuReceipts[0] || "", "utf8")); selected.app.sdk.objects = [graph.objects[0].path];
    assert.throws(() => windowsApplicationInputs(selected.app, repo), /Duplicate Windows application\/SDK object/);
  });
  test("text files are not admitted as COFF SDK objects", t => {
    const selected = fixture(t); fs.writeFileSync(selected.app.sdk.objects[0] || "", "not a COFF object");
    assert.throws(() => windowsApplicationInputs(selected.app, repo), /COFF payload/);
  });
  test("shared application libraries and duplicate SDK archives are rejected", t => {
    const {app} = fixture(t); app.sdk.libraries = [path.join(path.dirname(app.core), "runtime.dll")];
    assert.throws(() => validateWindowsApplication(app, repo), /static archives/); app.sdk.libraries = [...app.sdk.wholeArchives];
    assert.throws(() => validateWindowsApplication(app, repo), /Duplicate.*SDK archive/);
  });
  test("metadata substitution after admission is detected without rebuilding source", t => {
    const selected = fixture(t), result = windowsApplicationInputs(selected.app, repo); fs.appendFileSync(selected.app.gpuReceipts[0] || "", " ");
    assert.throws(() => verifyWindowsApplicationFiles(result.files), /input changed/);
  });
  test("linker response quoting preserves literal spaces, backslashes, quotes and shell characters", () => {
    assert.equal(windowsResponseFile(["C:\\SDK path\\lib.lib", "C:\\SDK path\\", 'quote"value', "%USERPROFILE%&$(no-shell)"]),
      '"C:\\SDK path\\lib.lib"\r\n"C:\\SDK path\\\\"\r\n"quote\\"value"\r\n"%USERPROFILE%&$(no-shell)"\r\n');
  });
  test("invalid response tokens cannot inject another linker input line", () => {
    for (const value of ["bad\n/injected", "bad\rline", "bad\0byte", "x".repeat(16385)]) {assert.throws(() => windowsResponseFile([value]), /response tokens/);}
  });
});
