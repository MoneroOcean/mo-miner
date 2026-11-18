"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const esbuild = require("esbuild");
const {spawnSync} = require("node:child_process");
const {
  validateWindowsControllerConfig, windowsControllerPlan, explicitInputFiles, hashFiles, assertStableInputHashes,
  buildNativeWindowsController, runTool, validatePeHeaders, verifyCanonicalBundle, windowsToolEnvironment, dependencyHashes, windowsLongPathManifest, windowsLongPathResource,
  windowsSystemImports,
} = require("../scripts/build-native-windows-controller");
const nativeBundle = require("../scripts/native-bundle");
const hostComponentReader = require("../scripts/native-windows-host-component");
const {fileHash} = require("../scripts/build-source-inputs");
const {compileNativeObjects} = require("../scripts/native-object-cache");

const repo = path.resolve(__dirname, "..");

function config(root = "/sdk fixture/with spaces") {
  const lib = (/** @type {string} */ name) => path.join(root, name);
  return {
    version: 1,
    tools: {clangCl: lib("clang-cl"), lldLink: lib("lld-link"), shermaot: lib("shermes"), llvmNm: lib("llvm-nm"),
      llvmReadobj: lib("llvm-readobj"), llvmObjdump: lib("llvm-objdump"), clangResourceInclude: lib("clang/include")},
    hermes: {source: lib("hermes-source"), build: lib("hermes-build"), shermaot: lib("shermes")},
    windowsSdk: {overlay: lib("vfs overlay.json"), sourceRoot: lib("windows sdk"), includeRoots: [lib("msvc/include"), lib("sdk/include/ucrt"), lib("sdk/include/shared"), lib("sdk/include/um"), lib("sdk/include/winrt"), lib("exact/include")],
      libraryRoots: [lib("msvc/lib"), lib("sdk/lib/ucrt"), lib("sdk/lib/um")],
      msvcLibraries: [lib("msvc/lib/libcmt.lib"), lib("msvc/lib/libcpmt.lib"), lib("msvc/lib/oldnames.lib")],
      ucrtLibraries: [lib("msvc/lib/libvcruntime.lib"), lib("msvc/lib/libucrt.lib")],
      importLibraries: [lib("sdk/lib/um/kernel32.lib"), lib("sdk/lib/um/WS2_32.lib")],
      allowedImports: ["KERNEL32.dll", "WS2_32.dll"]},
    libraries: {sourceInclude: lib("openssl-source/include"), ssl: lib("windows-openssl/libssl.a"), crypto: lib("windows-openssl/libcrypto.a"),
      libuv: lib("libuv.lib"), zlib: lib("zlibstatic.lib")},
  };
}

/** Extract RT_MANIFEST resource ID 1 from a bounded linker fixture PE. @param {Buffer} image */
function manifestResource(image) {
  const pe = image.readUInt32LE(0x3c), optional = pe + 24, sections = image.readUInt16LE(pe + 6), sectionTable = optional + image.readUInt16LE(pe + 20);
  assert.equal(image.readUInt32LE(pe), 0x4550); assert.equal(image.readUInt16LE(optional), 0x20b);
  const directory = optional + 112 + 2 * 8, resourceRva = image.readUInt32LE(directory), resourceSize = image.readUInt32LE(directory + 4);
  assert.ok(resourceRva && resourceSize && sections > 0 && sections <= 64);
  const rows = Array.from({length: sections}, (_, index) => {
    const at = sectionTable + index * 40;
    return {rva: image.readUInt32LE(at + 12), raw: image.readUInt32LE(at + 16), offset: image.readUInt32LE(at + 20)};
  });
  /** @param {number} rva @param {number} bytes */
  const fileOffset = (rva, bytes) => {
    const row = rows.find(section => rva >= section.rva && rva - section.rva <= section.raw - bytes);
    assert.ok(row, "resource RVA must fit initialized PE section data"); return row.offset + rva - row.rva;
  };
  /** @param {number} relative @param {number} id */
  const directoryEntry = (relative, id) => {
    assert.ok(relative >= 0 && relative + 16 <= resourceSize);
    const at = fileOffset(resourceRva + relative, 16), count = image.readUInt16LE(at + 12) + image.readUInt16LE(at + 14);
    assert.ok(count > 0 && count <= 128);
    for (let index = 0; index < count; index++) {
      const entry = at + 16 + index * 8, name = image.readUInt32LE(entry), target = image.readUInt32LE(entry + 4);
      if (!(name & 0x80000000) && name === id) {assert.ok(target & 0x80000000); return target & 0x7fffffff;}
    }
    assert.fail(`Missing resource ID ${id}`);
  };
  const type = directoryEntry(0, 24), name = directoryEntry(type, 1), languageAt = fileOffset(resourceRva + name, 16),
    languageCount = image.readUInt16LE(languageAt + 12) + image.readUInt16LE(languageAt + 14);
  assert.ok(languageCount > 0 && languageCount <= 128);
  const dataTarget = image.readUInt32LE(languageAt + 20);
  assert.equal(dataTarget & 0x80000000, 0);
  const dataEntry = resourceRva + dataTarget, dataAt = fileOffset(dataEntry, 16), dataRva = image.readUInt32LE(dataAt), dataSize = image.readUInt32LE(dataAt + 4);
  assert.ok(dataTarget + 16 <= resourceSize && dataSize > 0 && dataSize <= 65536);
  const content = fileOffset(dataRva, dataSize);
  return image.subarray(content, content + dataSize);
}

/** @param {import("node:test").TestContext} t */
function libraryRootFixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-library-roots-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const selected = config(path.join(root, "sdk fixture/with spaces")), roots = selected.windowsSdk.libraryRoots;
  for (const directory of roots) {fs.mkdirSync(directory, {recursive: true});}
  const comsupp = path.join(roots[0] || "", "comsuppw.lib"), uppercase = path.join(roots[1] || "", "extra.LIB"),
    readme = path.join(roots[0] || "", "readme.txt"), nested = path.join(roots[0] || "", "nested");
  fs.writeFileSync(comsupp, "implicit default library\n"); fs.writeFileSync(uppercase, "another candidate\n");
  fs.writeFileSync(readme, "not a linker archive\n"); fs.mkdirSync(nested);
  fs.writeFileSync(path.join(nested, "not-top-level.lib"), "nested candidate\n");
  for (const file of explicitInputFiles(selected)) {
    fs.mkdirSync(path.dirname(file), {recursive: true});
    if (!fs.existsSync(file)) {fs.writeFileSync(file, "fixture input\n");}
  }
  return {root, config: selected, roots, comsupp, uppercase, readme, nested};
}

test.describe("Windows SDK library-root input snapshots", () => {
  test("unlisted top-level archives are hashed while non-library and nested files are ignored", t => {
    const fixture = libraryRootFixture(t), paths = explicitInputFiles(fixture.config);
    assert.ok(paths.includes(fixture.comsupp), "implicit comsuppw.lib is part of the bounded candidate set");
    const hashes = hashFiles(paths);
    assert.ok(paths.includes(fixture.uppercase), "the .lib suffix is case-insensitive");
    assert.equal(fixture.config.windowsSdk.msvcLibraries.includes(fixture.comsupp), false,
      "the fixture proves discovery beyond explicitly listed archives");
    assert.equal(paths.includes(fixture.readme), false);
    assert.equal(paths.includes(path.join(fixture.nested, "not-top-level.lib")), false);
    assert.equal(hashes[fixture.comsupp], fileHash(fixture.comsupp));
    const plan = windowsControllerPlan(fixture.config, path.join(fixture.root, "planned"), repo);
    assert.deepEqual(plan.link.args.filter(arg => arg.startsWith("/libpath:")),
      fixture.roots.map(directory => "/libpath:" + directory), "library search order is unchanged");
  });

  test("content changes and added or removed candidates fail exact input-map stability", t => {
    const fixture = libraryRootFixture(t), before = hashFiles(explicitInputFiles(fixture.config)),
      original = fs.readFileSync(fixture.comsupp), added = path.join(fixture.roots[0] || "", "later-default.lib");
    fs.appendFileSync(fixture.comsupp, "mutated\n");
    assert.throws(() => assertStableInputHashes(before, hashFiles(explicitInputFiles(fixture.config))), /input set or contents changed/);
    fs.writeFileSync(fixture.comsupp, original);

    fs.writeFileSync(added, "new candidate\n");
    const withAdded = hashFiles(explicitInputFiles(fixture.config));
    assert.throws(() => assertStableInputHashes(before, withAdded), /input set or contents changed/);
    fs.rmSync(added);
    assert.throws(() => assertStableInputHashes(withAdded, hashFiles(explicitInputFiles(fixture.config))), /input set or contents changed/);

    const beforeRemoval = hashFiles(explicitInputFiles(fixture.config)); fs.rmSync(fixture.comsupp);
    assert.throws(() => assertStableInputHashes(beforeRemoval, hashFiles(explicitInputFiles(fixture.config))), /input set or contents changed/);
  });

  test("non-regular and symlink archive candidates are rejected", t => {
    const fixture = libraryRootFixture(t), directory = path.join(fixture.roots[0] || "", "directory.lib");
    fs.mkdirSync(directory);
    assert.throws(() => explicitInputFiles(fixture.config), /library candidate is not a regular file/);
    fs.rmdirSync(directory);
    const target = path.join(fixture.root, "target.lib"), alias = path.join(fixture.roots[0] || "", "alias.lib");
    fs.writeFileSync(target, "target\n");
    try {fs.symlinkSync(target, alias, "file");} catch (error) {
      const code = /** @type {{code?: string}} */ (error)?.code;
      if (["EPERM", "EACCES", "ENOTSUP", "EOPNOTSUPP"].includes(code || "")) {t.skip("Host does not permit symlink fixtures"); return;}
      throw error;
    }
    assert.throws(() => explicitInputFiles(fixture.config), /library candidate is not a regular file/);
  });
});

test.describe("Windows COFF reproducibility", () => {
  test("selected Intel compiler and LLD emit identical objects and PE bytes across build times", {
    skip: (!process.env["MOM_TEST_WINDOWS_CC"] || !process.env["MOM_TEST_WINDOWS_LINKER"]) &&
      "Set MOM_TEST_WINDOWS_CC and MOM_TEST_WINDOWS_LINKER to exercise the pinned tools",
  }, async t => {
    const compiler = process.env["MOM_TEST_WINDOWS_CC"] || "", linker = process.env["MOM_TEST_WINDOWS_LINKER"] || "";
    assert.ok(path.isAbsolute(compiler) && path.isAbsolute(linker), "Select absolute pinned compiler/linker paths");
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom coff repro-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const tools = [compiler, linker].map(file => fileHash(file)), outputs = [];
    const plan = windowsControllerPlan(config(), path.join(root, "plan"), repo),
      compileStamp = plan.compile[0]?.args.filter(arg => arg === "/Brepro") || [],
      linkStamp = plan.link.args.filter(arg => arg === "/Brepro" || arg.startsWith("/timestamp:"));
    for (const name of ["first", "different-depth/second"]) {
      const cwd = path.join(root, name); fs.mkdirSync(cwd, {recursive: true});
      fs.writeFileSync(path.join(cwd, "fixture.c"), "volatile unsigned char payload[65536] = {1}; void entry(void) { payload[0] = payload[1]; }\n");
      const manifest = path.join(cwd, "long-path-aware.res");
      fs.writeFileSync(manifest, windowsLongPathResource());
      // No SDK headers, CRT, GPU or image execution: isolate the actual tools' COFF timestamps.
      for (const [command, args] of /** @type {[string, string[]][]} */ ([[compiler,
        ["--target=x86_64-pc-windows-msvc", "/nologo", "/c", "/O2", "/GS-", "/Zl", ...compileStamp, "/Fofixture.obj", "fixture.c"]],
      [linker, ["/machine:x64", "/subsystem:console", "/entry:entry", "/nodefaultlib", "/nologo", ...linkStamp,
        "/manifest:no", manifest, "/out:fixture.exe", "fixture.obj"]]])) {
        const result = spawnSync(command, args, {cwd, env: windowsToolEnvironment(), encoding: "utf8", timeout: 30000, maxBuffer: 64 << 10});
        assert.equal(result.error, undefined); assert.equal(result.signal, null); assert.equal(result.status, 0, result.stderr);
      }
      const imageFile = path.join(cwd, "fixture.exe");
      assert.ok(fs.statSync(imageFile).size < 131072, "Timestamp/compression fixture must remain bounded");
      const image = fs.readFileSync(imageFile), pe = image.readUInt32LE(0x3c), optional = pe + 24;
      assert.equal(image.readUInt16LE(optional), 0x20b);
      const embeddedManifest = manifestResource(image), manifestText = embeddedManifest.toString("utf8") + embeddedManifest.toString("utf16le");
      assert.match(manifestText, /<ws2:longPathAware>true<\/ws2:longPathAware>/);
      assert.equal(fs.existsSync(imageFile + ".manifest"), false, "long-path support must remain in the single executable");
      const upx = process.env["MOM_TEST_UPX"];
      if (upx) {
        assert.ok(path.isAbsolute(upx));
        assert.equal(fileHash(upx), require("../scripts/native-packing-tool.json").executable.sha256);
        const packed = path.join(cwd, "packed.exe"), restored = path.join(cwd, "restored.exe");
        for (const args of [["--no-env", "--best", "--lzma", "--crp-lzma-ds=134217728", "--strip-relocs=0", "--compress-exports=0", "--no-progress", "-o", packed, imageFile],
          ["--no-env", "-t", packed], ["--no-env", "-d", "-o", restored, packed]]) {
          const result = /** @type {import("node:child_process").SpawnSyncReturns<string>} */ (spawnSync(upx, args, {cwd, encoding: "utf8", timeout: 30000, maxBuffer: 64 << 10}));
          assert.equal(result.error, undefined); assert.equal(result.signal, null); assert.equal(result.status, 0, result.stderr);
        }
        // The OS consumes the packed image's manifest before UPX restores any application bytes.
        assert.deepEqual(manifestResource(fs.readFileSync(packed)), embeddedManifest);
        assert.deepEqual(manifestResource(fs.readFileSync(restored)), embeddedManifest);
      }
      assert.equal(image.readUInt32LE(optional + 112 + 6 * 8), 0,
        "Do not emit a REPRO debug directory that UPX erases during restoration");
      assert.equal(image.readUInt32LE(pe + 8), 0, "The PE link timestamp must be fixed");
      outputs.push(["fixture.obj", "fixture.exe"].map(file => fileHash(path.join(cwd, file))));
      // Make a wall-clock stamp differ even on filesystems with one-second timestamp granularity.
      if (outputs.length === 1) {await new Promise(resolve => setTimeout(resolve, 1200));}
    }
    assert.deepEqual(outputs[0], outputs[1]);
    assert.deepEqual([compiler, linker].map(file => fileHash(file)), tools, "Selected tools must not change during the comparison");
  });
});

test.describe("Windows maintained controller build recipe", () => {
  test("data-only producer and planner imports defer esbuild until bundle work", () => {
    const ciPath = path.resolve(__dirname, "../scripts/run-native-ci.js"),
      controllerPath = path.resolve(__dirname, "../scripts/build-native-windows-controller.js"),
      planOutput = path.join(os.tmpdir(), "mom-esbuild-free-plan");
    const script = `
const Module = require("node:module"), originalLoad = Module._load;
Module._load = function(request, ...args) {
  if (request === "esbuild") {throw new Error("test blocked esbuild");}
  return originalLoad.call(this, request, ...args);
};
const ci = require(${JSON.stringify(ciPath)});
if (typeof ci.runStage !== "function") {throw new Error("Native CI stage dispatcher is missing");}
const controller = require(${JSON.stringify(controllerPath)});
if (typeof controller.buildNativeWindowsController !== "function" || typeof controller.windowsControllerPlan !== "function") {
  throw new Error("Windows controller producer/planner API is missing");
}
const plan = controller.windowsControllerPlan(${JSON.stringify(config())}, ${JSON.stringify(planOutput)}, ${JSON.stringify(repo)});
if (!plan || !plan.link) {throw new Error("Windows controller planner did not return a link plan");}
try {
  controller.verifyCanonicalBundle("/tmp/entry.js", "/tmp/controller.js", Buffer.alloc(0));
  throw new Error("Bundle verification did not load esbuild");
} catch (error) {
  if (!(error instanceof Error) || error.message !== "test blocked esbuild") {throw error;}
}
console.log("planner import stayed esbuild-free; bundle verification requested esbuild");
`;
    const result = spawnSync(process.execPath, ["-e", script], {cwd: repo, encoding: "utf8", timeout: 10000, maxBuffer: 64 << 10});
    assert.equal(result.error, undefined);
    assert.equal(result.signal, null);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    assert.match(result.stdout, /planner import stayed esbuild-free; bundle verification requested esbuild/);
  });

  test("C and C++ source paths follow explicit language operands without intervening options", () => {
    const plan = windowsControllerPlan(config(), "/tmp/mom-language-operands", "/workspace");
    for (const [index, command] of plan.compile.entries()) {
      const option = index === 0 ? "/Tc" : "/Tp", at = command.args.indexOf(option), source = command.args[at + 1];
      assert.ok(at >= 0, "each maintained source has an explicit language operand");
      assert.ok(source && path.isAbsolute(source) && /\.(?:c|cpp)$/.test(source));
      assert.equal(command.args[at - 1], config().windowsSdk.overlay, "overlay options precede the inseparable source operand pair");
    }
  });
  test("full application uses the same host recipe and compiles only shared dispatch glue", () => {
    const selected = /** @type {import("../scripts/build-native-windows-controller").WindowsControllerConfig} */ (config());
    const sdk = (/** @type {string} */ name) => path.join("/sdk fixture/with spaces", name);
    selected.application = {core: sdk("cpu-core.lib"), gpuReceipts: [sdk("oneapi.json"), sdk("dpcpp.json"), sdk("opencl.json"), sdk("cuda.json"), sdk("hip.json")],
      defaultCompiler: "oneapi", tools: {llvmObjcopy: sdk("llvm-objcopy"), llvmAr: sdk("llvm-ar")},
      sdk: {libraries: [sdk("static-runtime.lib")], wholeArchives: [sdk("static-images.lib")], objects: [sdk("resources.obj")],
        includeRoots: [sdk("include")], nvmlHeader: sdk("nvml.h"), hipDriverLibrary: sdk("amdhip64.lib")}};
    const plan = windowsControllerPlan(selected, "/tmp/mom-full-windows", repo);
    assert.equal(path.basename(plan.generated.executable), "mom.exe");
    assert.equal(plan.compile.length, 12);
    assert.deepEqual(plan.sources.slice(-4), ["gpu-dispatch", "gpu-forwards", "compiler-gate", "packed-windows"]);
    assert.equal(plan.sources.includes("cuda-images"), false, "static-NVRTC and ordinary applications do not link an image registry");
    assert.equal(plan.compile.slice(1).every(command => command.args.includes("-DMOM_NATIVE_RUNTIME") && command.args.includes("/MT")), true);
    assert.equal(plan.compile.every(command => command.args.includes("-MV")), true, "all Windows compiler depfiles use the unambiguous NMake format");
    assert.equal(plan.compile[0]?.args.includes("-DMOM_NATIVE_RUNTIME"), false);
    assert.ok(plan.link.args.includes(selected.application.core));
    assert.ok(plan.link.args.includes("/wholearchive:" + selected.application.sdk.wholeArchives[0]));
    assert.ok(plan.link.args.includes(selected.application.sdk.objects[0] || ""));
    assert.ok(plan.compile.find(command => command.args.includes(path.join(repo, "runtime/nvidia-metadata.cpp")))?.args.includes("-DMOM_NATIVE_NVML_METADATA"));
    assert.equal(plan.link.args.includes(selected.application.sdk.hipDriverLibrary), false, "driver discovery is provisional, never an eager final driver import");
    const guards = plan.compile.filter(command => command.args.includes("/GS-"));
    assert.equal(guards.length, 1); assert.ok(guards[0]?.args.includes(path.join(repo, "runtime/packed-windows.cpp")));
    assert.ok(plan.link.args.includes("/entry:momPackedEntry"));
    assert.ok(plan.link.args.includes("/nodefaultlib:sycl-devicelib-host.lib"));
    assert.equal(plan.link.args.some(arg => /^\/force(?::|$)/i.test(arg)), false);
    for (const flag of ["/NXCOMPAT", "/DYNAMICBASE", "/HIGHENTROPYVA"]) {assert.ok(plan.link.args.includes(flag));}
    assert.ok(plan.link.args.includes("/timestamp:0"), "the final PE timestamp is fixed without an UPX-incompatible debug marker");
    assert.ok(!plan.link.args.includes("/Brepro"));
    assert.ok(plan.compile.every(command => command.args.includes("/Brepro")), "every controller object omits wall-clock COFF timestamps");
  });
  test("full application embeds the stable long-path capability manifest", () => {
    const selected = /** @type {import("../scripts/build-native-windows-controller").WindowsControllerConfig} */ (config());
    const sdk = (/** @type {string} */ name) => path.join("/sdk fixture/with spaces", name);
    selected.application = {core: sdk("cpu-core.lib"), gpuReceipts: [sdk("oneapi.json"), sdk("dpcpp.json"), sdk("opencl.json"), sdk("cuda.json"), sdk("hip.json")],
      defaultCompiler: "oneapi", tools: {llvmObjcopy: sdk("llvm-objcopy"), llvmAr: sdk("llvm-ar")},
      sdk: {libraries: [sdk("static-runtime.lib")], wholeArchives: [sdk("static-images.lib")], objects: [sdk("resources.obj")],
        includeRoots: [sdk("include")], nvmlHeader: sdk("nvml.h"), hipDriverLibrary: sdk("amdhip64.lib")}};
    const output = "/tmp/mom-full-windows-manifest", plan = windowsControllerPlan(selected, output, repo);
    assert.equal(plan.generated.manifest, path.join(output, "long-path-aware.res"));
    assert.ok(plan.link.args.includes("/manifest:no"), "fixed .res input must not invoke external mt.exe");
    assert.ok(plan.link.args.includes(plan.generated.manifest), "the generated resource is an explicit LLD link input");
    const resource = windowsLongPathResource();
    assert.equal(resource.readUInt32LE(32), Buffer.byteLength(windowsLongPathManifest));
    assert.equal(resource.readUInt16LE(42), 24); assert.equal(resource.readUInt16LE(46), 1);
    assert.equal(resource.subarray(64, 64 + resource.readUInt32LE(32)).toString("utf8"), windowsLongPathManifest);
    assert.match(windowsLongPathManifest, /<ws2:longPathAware>true<\/ws2:longPathAware>/);
    assert.equal(windowsLongPathManifest.includes(output), false, "the stable manifest contains no build path");
  });
  test("embedded-image applications compile one shared registry TU into the application", () => {
    const selected = /** @type {import("../scripts/build-native-windows-controller").WindowsControllerConfig} */ (config());
    const sdk = (/** @type {string} */ name) => path.join("/sdk fixture/with spaces", name);
    selected.application = {core: sdk("cpu-core.lib"), gpuReceipts: [sdk("oneapi.json"), sdk("dpcpp.json"), sdk("opencl.json"), sdk("cuda.json"), sdk("hip.json")],
      defaultCompiler: "oneapi", tools: {llvmObjcopy: sdk("llvm-objcopy"), llvmAr: sdk("llvm-ar")},
      sdk: {libraries: [sdk("static-runtime.lib")], wholeArchives: [sdk("static-images.lib")], objects: [sdk("resources.obj")],
        includeRoots: [sdk("include")], nvmlHeader: sdk("nvml.h"), hipDriverLibrary: sdk("amdhip64.lib"), cudaImages: sdk("cuda-images.inc")}};
    const output = path.join(os.tmpdir(), "mom-full-windows-images"), plan = windowsControllerPlan(selected, output, repo),
      source = path.join(repo, "sycl/cuda-images.cpp"), registryObjects = plan.sources.filter(name => name === "cuda-images");
    assert.equal(registryObjects.length, 1);
    assert.equal(plan.sources.at(-1), "cuda-images");
    const command = plan.compile.find(item => item.args.includes(source));
    assert.ok(command, "the shared maintained registry source is compiled by the application toolchain");
    assert.ok(command.args.includes("/std:c++20"));
    assert.ok(command.args.includes("-I") && command.args.includes(output), "the generated bundle is visible from the app output directory");
    assert.equal(plan.link.args.filter(file => file === plan.objects["cuda-images"]).length, 1);
    assert.equal(plan.generated.cudaImages, path.join(output, "cuda-images.inc"));
  });
  test("embedded-image linking rejects static NVRTC archives from the outer Windows SDK", () => {
    const selected = /** @type {import("../scripts/build-native-windows-controller").WindowsControllerConfig} */ (config());
    const sdk = (/** @type {string} */ name) => path.join("/sdk fixture/with spaces", name);
    selected.application = {core: sdk("cpu-core.lib"), gpuReceipts: [sdk("oneapi.json"), sdk("dpcpp.json"), sdk("opencl.json"), sdk("cuda.json"), sdk("hip.json")],
      defaultCompiler: "oneapi", tools: {llvmObjcopy: sdk("llvm-objcopy"), llvmAr: sdk("llvm-ar")},
      sdk: {libraries: [sdk("static-runtime.lib")], wholeArchives: [sdk("static-images.lib")], objects: [sdk("resources.obj")],
        includeRoots: [sdk("include")], nvmlHeader: sdk("nvml.h"), hipDriverLibrary: sdk("amdhip64.lib"), cudaImages: sdk("cuda-images.inc")}};
    selected.windowsSdk.msvcLibraries.push(sdk("nvrtc_static.lib"));
    assert.throws(() => windowsControllerPlan(selected, "/tmp/mom-full-windows-images-nvrtc", repo), /static NVRTC libraries/);

    delete selected.application.sdk.cudaImages;
    assert.doesNotThrow(() => windowsControllerPlan(selected, "/tmp/mom-full-windows-static-nvrtc", repo),
      "the transitional non-image application route keeps its existing static-NVRTC behavior");
  });
  test("ordinary OLE and device configuration remain OS libraries, not application-DLL exceptions", () => {
    const policy = windowsSystemImports(); policy.push("node.dll");
    assert.equal(windowsSystemImports().includes("node.dll"), false, "callers cannot mutate the closed OS import policy");
    const selected = config(); selected.windowsSdk.allowedImports.push("OLEAUT32.dll", "CFGMGR32.dll");
    assert.doesNotThrow(() => validateWindowsControllerConfig(selected));
    selected.windowsSdk.allowedImports.push("sycl.dll"); assert.throws(() => validateWindowsControllerConfig(selected), /ordinary system DLLs/);
  });
  test("compiler flags and SDK search paths cannot leak in through the parent environment", t => {
    const keys = ["CL", "_cl_", "LINK", "INCLUDE", "LIBPATH", "CXXFLAGS", "MOM_NATIVE_RUNTIME"], before = Object.fromEntries(keys.map(key => [key, process.env[key]]));
    t.after(() => {for (const key of keys) {if (before[key] === undefined) {delete process.env[key];} else {process.env[key] = before[key];}}});
    for (const key of keys) {process.env[key] = "unreviewed SDK override";}
    const selected = windowsToolEnvironment(), pathValue = (/** @type {NodeJS.ProcessEnv} */ environment) =>
      Object.entries(environment).find(([name]) => name.toLowerCase() === "path")?.[1];
    for (const key of keys) {assert.equal(Object.hasOwn(selected, key), false);}
    assert.equal(pathValue(selected), pathValue(process.env));
  });
  test("embedded policy identity uses target path semantics during cross compilation", () => {
    for (const root of [repo, "/source root/é-💎/mo-miner"]) {
      const plan = windowsControllerPlan(config(), "/tmp/mom-windows-policy", root);
      // The controller and native resource lookup share the target's virtual root,
      // regardless of the host checkout location or path separator.
      assert.equal(Reflect.get(plan, "policyPath"), "C:\\__mom_builtin__\\GPU-CONFIG.md");
      assert.notEqual(Reflect.get(plan, "policyPath"), path.win32.join(root, "elsewhere", "GPU-CONFIG.md"));
    }
  });

  test("the plan consumes explicit SDK paths and preserves resource-header ordering", () => {
    const selected = config(), plan = windowsControllerPlan(selected, path.join(os.tmpdir(), "mom-windows-controller"), repo);
    assert.equal(plan.config.windowsSdk.overlay, selected.windowsSdk.overlay);
    assert.equal(plan.aot.cwd, plan.output);
    assert.deepEqual(plan.aot.args.slice(-3), ["-o", "./controller.c", "./controller.js"]);
    assert.equal(JSON.parse(JSON.stringify(plan.aot)).cwd, plan.output, "relative AOT arguments retain their replay working directory");
    assert.equal(plan.link.command, selected.tools.lldLink);
    const first = plan.compile[0]; assert.ok(first);
    assert.ok(first.args.includes("-nostdinc"));
    assert.equal(first.args[first.args.indexOf("-I") + 1], selected.tools.clangResourceInclude);
    assert.ok(plan.link.args.includes(path.join("/sdk fixture/with spaces", "libuv.lib")));
    assert.ok(plan.link.args.includes("/timestamp:0"));
    assert.equal(plan.generated.manifest, undefined, "the host controller recipe keeps its existing manifest behavior");
    assert.equal(plan.link.args.some(arg => arg === "/manifest:embed" || arg.startsWith("/manifestinput:")), false);
    assert.ok(!plan.link.args.includes("/Brepro"));
    assert.ok(plan.compile.every(command => command.args.includes("/Brepro")));
    assert.doesNotMatch(plan.compile.flatMap(command => command.args).join("\n"), /intrinsics-shim|environ-shim|MOM_NATIVE_RUNTIME/);
  });

  test("generated C preserves Windows setjmp lowering while C++ flags remain cache-bound", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom win-controller-no-builtin-cache-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const planned = windowsControllerPlan(config(), path.join(root, "planned"), repo);
    const controller = planned.compile[0];
    const processSource = path.join(repo, "runtime/process.cpp");
    const processCommand = planned.compile.find(command => command.args.includes(processSource));
    assert.ok(controller);
    assert.doesNotMatch(controller.args.join(" "), /-fno-builtin|ffreestanding|\/Oi-/, "MSVC setjmp needs compiler lowering for its hidden frame argument");
    assert.ok(controller.args.includes("/Qno-builtin-memset"), "only Intel's C memset builtin mapping is disabled");
    assert.equal(processCommand?.args.includes("/Qno-builtin-memset"), false, "C++ host flags remain unchanged");
    assert.ok(processCommand?.args.includes("-fno-builtin"), "maintained process host code uses the same host policy");

    const source = path.join(root, "runtime/process.cpp"), compiler = path.join(root, "fake-clang-cl.js"), emptyPath = path.join(root, "empty-path");
    fs.mkdirSync(path.dirname(source), {recursive: true}); fs.mkdirSync(emptyPath);
    fs.writeFileSync(source, "int process_fixture() { return 0; }\n");
    fs.writeFileSync(compiler, `#!${process.execPath}\n"use strict";
const fs=require("node:fs"),path=require("node:path"),args=process.argv.slice(2);
const objectArg=args.find(arg=>arg.startsWith("-Fo")), dependencyIndex=args.indexOf("-dependency-file");
const object=objectArg&&objectArg.slice(3), depfile=dependencyIndex>=0&&args[dependencyIndex+2], source=process.env.NATIVE_CACHE_SOURCE;
if(!object||!depfile||!source)process.exit(41);
fs.mkdirSync(path.dirname(object),{recursive:true});fs.mkdirSync(path.dirname(depfile),{recursive:true});
fs.writeFileSync(object,JSON.stringify(args));fs.writeFileSync(depfile,object+': "'+source+'"\\n');
`);
    fs.chmodSync(compiler, 0o755);
    const environment = /** @type {NodeJS.ProcessEnv} */ ({...Object.fromEntries(Object.entries(windowsToolEnvironment())
      .filter(([key]) => key.toLowerCase() !== "path")), PATH: emptyPath,
    NATIVE_CACHE_SOURCE: source, MOM_NATIVE_NINJA: ""});
    /** @param {string} name @param {boolean} keepNoBuiltin */
    const run = (name, keepNoBuiltin) => {
      const output = path.join(root, name); fs.mkdirSync(output);
      const object = path.join(output, "process.obj"), depfile = path.join(output, "process.d");
      const args = /** @type {{args: string[]}} */ (processCommand).args.map(arg =>
        arg.replaceAll(planned.output, output).replaceAll(processSource, source));
      if (!keepNoBuiltin) {args.splice(args.indexOf("-fno-builtin"), 1);}
      return compileNativeObjects({repo: root, output, namespace: "windows/controller/cache-test",
        commands: [{command: process.execPath, args: [compiler, ...args], source, object, depfile, dependencyFormat: "nmake"}],
        environment, keyFiles: [compiler], includeRoots: []});
    };
    assert.equal(run("with-no-builtin", true).objects[0]?.cacheHit, false);
    assert.equal(run("without-no-builtin", false).objects[0]?.cacheHit, false,
      "changing the controller compiler flag must miss the prior object slot");
    assert.equal(run("without-no-builtin-again", false).objects[0]?.cacheHit, true);
  });

  test("explicit static CRT inputs do not trigger case-sensitive default-library lookup", () => {
    const plan = windowsControllerPlan(config(), "/tmp/mom-windows-controller", repo);
    for (const name of ["LIBCMT.lib", "OLDNAMES.lib"]) {assert.ok(plan.link.args.includes("/nodefaultlib:" + name));}
    for (const file of config().windowsSdk.msvcLibraries) {assert.ok(plan.link.args.includes(file));}
    assert.equal(plan.link.args.includes("/nodefaultlib"), false, "unrelated default libraries remain enabled");
    assert.equal(plan.link.args.some(arg => /^\/force(?::|$)/i.test(arg)), false, "unresolved symbols remain fatal");
  });

  test("keeps pinned Intel runtime inputs but excludes mixed-case basenames from final links", t => {
    const fixture = libraryRootFixture(t), selected = fixture.config,
      ircmt = path.join(fixture.roots[0] || "", "LIBIRCMT.LIB"), mmt = path.join(fixture.roots[0] || "", "LiBmMt.LiB");
    fs.writeFileSync(ircmt, "pinned Intel archive input\n"); fs.writeFileSync(mmt, "pinned Intel archive input\n");
    selected.windowsSdk.msvcLibraries.push(ircmt, mmt);
    const hashes = hashFiles(explicitInputFiles(selected)), plan = windowsControllerPlan(selected, path.join(fixture.root, "planned"), repo);
    for (const archive of [ircmt, mmt]) {
      assert.ok(Object.hasOwn(hashes, archive), "unused Intel runtime inputs remain hash-bound");
      assert.equal(plan.link.args.includes(archive), false, "the Intel host archive is not an explicit link input");
      assert.ok(plan.link.args.includes("/nodefaultlib:" + path.basename(archive).toLowerCase()), "case-varied defaultlib hints cannot re-add it");
    }
    for (const archive of selected.windowsSdk.msvcLibraries.filter(file => file !== ircmt && file !== mmt)) {
      assert.ok(plan.link.args.includes(archive), "other explicitly selected MSVC libraries remain linked");
    }
    for (const archive of selected.windowsSdk.ucrtLibraries) {assert.ok(plan.link.args.includes(archive), "UCRT remains available for ordinary math and OS support");}
    assert.ok(plan.link.args.includes("/defaultlib:libucrt.lib"));
    assert.equal(plan.link.args.some(arg => /^\/force(?::|$)/i.test(arg)), false, "unresolved symbols remain fatal");
  });

  test("simulated compiler commands receive literal argv without a shell", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-controller-command-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    /** @type {{command: string, args: string[], options: object}[]} */
    const calls = [];
    runTool("/sdk/clang-cl", ["literal;$(false)", "space value", "quote\"value"], root, path.join(root, "tool.log"),
      /** @type {any} */ ((/** @type {string} */ command, /** @type {string[]} */ args, /** @type {object} */ options) => {
        calls.push({command, args: [...args], options}); return {status: 0, signal: null};
      }));
    assert.equal(calls.length, 1); assert.equal(calls[0]?.command, "/sdk/clang-cl");
    assert.deepEqual(calls[0]?.args, ["literal;$(false)", "space value", "quote\"value"]);
    assert.equal(Reflect.get(calls[0]?.options || {}, "shell"), undefined);
  });

  test("NMake depfiles retain and hash SDK headers with spaces", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-sdk-deps-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const output = path.join(root, "out"), header = path.join(root, "sdk header.h"), dep = path.join(root, "fixture.d"), overlay = path.join(root, "overlay.json");
    fs.mkdirSync(output); fs.writeFileSync(header, "header"); fs.writeFileSync(dep, 'fixture.o: "' + header + '"\n'); fs.writeFileSync(overlay, '{"roots":[]}');
    const hashes = dependencyHashes([dep], overlay, output, Date.now() + 1000);
    assert.equal(hashes[fs.realpathSync(header)], require("../scripts/build-source-inputs").fileHash(header));
  });

  test("the production read-only bundle verifier rejects source changes with bounded diagnostics", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-controller-bundle-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const entry = path.join(root, "entry.js"), bundle = path.join(root, "controller.js");
    fs.writeFileSync(entry, "module.exports = 1;\n");
    esbuild.buildSync(nativeBundle.controllerBundleOptions(entry, bundle));
    const expected = fs.readFileSync(bundle);
    fs.writeFileSync(entry, "module.exports = 2;\n");
    assert.throws(() => verifyCanonicalBundle(entry, bundle, expected),
      /Native bundle mismatch: actual \d+ bytes SHA256 [0-9a-f]{64}; expected \d+ bytes SHA256 [0-9a-f]{64}/);
    assert.equal(fs.readFileSync(bundle).equals(expected), true, "Read-only verification must not replace the saved bundle");
  });

  test("the production read-only bundle verifier rejects missing or ambiguous outputs", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-controller-output-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const entry = path.join(root, "entry.js"), bundle = path.join(root, "controller.js");
    fs.writeFileSync(entry, "module.exports = 1;\n");
    nativeBundle(root);
    fs.writeFileSync(bundle, "saved bundle\n");
    const noOutput = /** @type {any} */ (() => ({metafile: {inputs: {}}, outputFiles: []}));
    assert.throws(() => verifyCanonicalBundle(entry, bundle, fs.readFileSync(bundle), noOutput), /exactly one output/);
    const ambiguous = /** @type {any} */ (() => ({metafile: {inputs: {}}, outputFiles: [
      {path: bundle, contents: Buffer.from("saved bundle\n")},
      {path: bundle + ".map", contents: Buffer.from("map\n")},
    ]}));
    assert.throws(() => verifyCanonicalBundle(entry, bundle, fs.readFileSync(bundle), ambiguous), /exactly one output/);
  });

  test("PE header validation rejects delay-loaded imports", () => {
    const header = `Format: COFF-x86-64
ImageOptionalHeader {
  Magic: 0x20B
  DataDirectory {
    DelayImportDescriptorRVA: 0x0
    DelayImportDescriptorSize: 0x0
  }
}`;
    assert.doesNotThrow(() => validatePeHeaders(header));
    assert.throws(() => validatePeHeaders(header.replace("DelayImportDescriptorRVA: 0x0", "DelayImportDescriptorRVA: 0x1000")),
      /delay-import directory is not supported/);
    assert.throws(() => validatePeHeaders(header.replace(/^[ ]{4}DelayImportDescriptor[^\n]+\n/gm, "")), /omitted the delay-import directory/);
  });

  test("malformed or non-explicit SDK paths fail before a build plan exists", () => {
    const selected = config();
    selected.tools.clangCl = "clang-cl";
    assert.throws(() => validateWindowsControllerConfig(selected), /absolute tool clangCl/);
    const noHermes = JSON.parse(JSON.stringify(config())); delete noHermes.hermes;
    assert.throws(() => validateWindowsControllerConfig(noHermes), /Invalid Windows controller SDK descriptor/,
      "the application/controller path still requires Hermes inputs");
    const noLibraries = JSON.parse(JSON.stringify(config())); delete noLibraries.libraries;
    assert.throws(() => validateWindowsControllerConfig(noLibraries), /Invalid Windows controller SDK descriptor/,
      "the application/controller path still requires TLS and I/O libraries");
    const noImportPolicy = JSON.parse(JSON.stringify(config())); delete noImportPolicy.windowsSdk.allowedImports;
    assert.throws(() => validateWindowsControllerConfig(noImportPolicy), /Allowed Windows imports/,
      "the application/controller path retains its explicit PE import policy");
    const withCore = /** @type {any} */ (config()); withCore.cpuCore = "/sdk/core";
    assert.throws(() => validateWindowsControllerConfig(withCore), /Core\/GPU/);
    const duplicate = config(); duplicate.windowsSdk.importLibraries.push(duplicate.windowsSdk.importLibraries[0] || "");
    assert.throws(() => validateWindowsControllerConfig(duplicate), /Duplicate/);
    const nodeImport = config(); nodeImport.windowsSdk.allowedImports = ["node.dll"];
    assert.throws(() => validateWindowsControllerConfig(nodeImport), /ordinary system DLLs/);
    const wrongTarget = /** @type {any} */ (config()); wrongTarget.target = "aarch64-pc-windows-msvc";
    assert.throws(() => validateWindowsControllerConfig(wrongTarget), /x86_64-pc-windows-msvc/);
  });

  test("an existing output collision is never overwritten", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-controller-test-"));
    try {
      const output = path.join(root, "occupied"); fs.mkdirSync(output); fs.writeFileSync(path.join(output, "keep"), "keep");
      assert.throws(() => buildNativeWindowsController(path.join(root, "missing.json"), output), /new absolute/);
      assert.equal(fs.readFileSync(path.join(output, "keep"), "utf8"), "keep");
    } finally {fs.rmSync(root, {recursive: true, force: true});}
  });

  test("a host component receipt replacement during link planning is rejected before admission", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-host-component-freeze-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const selected = /** @type {import("../scripts/build-native-windows-controller").WindowsControllerConfig} */ (config(path.join(root, "sdk"))),
      makeFile = (/** @type {string} */ file, data = "!<arch>\n") => {
        fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, data); return file;
      };
    const hostRoot = path.join(root, "host-component"), manifest = makeFile(path.join(hostRoot, "component.json"), "original receipt\n"),
      hostLibrary = makeFile(path.join(hostRoot, "lib", "acpp.lib"));
    const appSdkRoot = path.join(root, "app-sdk"), appInclude = path.join(appSdkRoot, "include"),
      core = path.join(appSdkRoot, "cpu-core.lib"), nvml = path.join(appSdkRoot, "nvml.h"), hip = path.join(appSdkRoot, "amdhip64.lib"),
      runtimeLibrary = path.join(appSdkRoot, "runtime.lib"), wholeArchive = path.join(appSdkRoot, "whole.lib"),
      llvmAr = path.join(appSdkRoot, "llvm-ar"), llvmObjcopy = path.join(appSdkRoot, "llvm-objcopy");
    selected.application = {core, gpuReceipts: ["oneapi", "dpcpp", "opencl", "cuda", "hip"].map(name => path.join(appSdkRoot, name + ".json")),
      defaultCompiler: "oneapi", tools: {llvmAr, llvmObjcopy}, sdk: {libraries: [runtimeLibrary], wholeArchives: [wholeArchive], objects: [],
        includeRoots: [appInclude], nvmlHeader: nvml, hipDriverLibrary: hip, hostComponent: hostRoot}};
    const directories = [selected.tools.clangResourceInclude, selected.hermes.source, selected.hermes.build, selected.windowsSdk.sourceRoot,
      ...selected.windowsSdk.includeRoots, ...selected.windowsSdk.libraryRoots, selected.libraries.sourceInclude, appInclude];
    for (const directory of directories) {fs.mkdirSync(directory, {recursive: true});}
    makeFile(selected.windowsSdk.overlay, JSON.stringify({roots: []}));
    const files = [selected.tools.clangCl, selected.tools.lldLink, selected.tools.shermaot, selected.tools.llvmNm, selected.tools.llvmReadobj,
      selected.tools.llvmObjdump || "", selected.libraries.ssl, selected.libraries.crypto, selected.libraries.libuv, selected.libraries.zlib,
      ...selected.windowsSdk.msvcLibraries, ...selected.windowsSdk.ucrtLibraries, ...selected.windowsSdk.importLibraries,
      path.join(selected.hermes.build, "jsi/jsi.lib"), path.join(selected.hermes.build, "lib/hermesvmlean_a.lib"),
      path.join(selected.hermes.build, "external/boost/boost_1_86_0/libs/context/boost_context.lib"),
      core, nvml, hip, runtimeLibrary, wholeArchive, llvmAr, llvmObjcopy];
    for (const file of files) {if (file) {makeFile(file, path.extname(file).toLowerCase() === ".json" ? "{}" : undefined);}}
    const hostFiles = {[manifest]: fileHash(manifest), [hostLibrary]: fileHash(hostLibrary)};
    let reads = 0;
    t.mock.method(hostComponentReader, "windowsHostComponentInputs", (/** @type {string} */ selectedRoot) => {
      assert.equal(selectedRoot, hostRoot);
      if (++reads === 1) {fs.writeFileSync(manifest, "replacement receipt\n");}
      return {link: {libraries: [hostLibrary]}, files: hostFiles, provenance: {},
        runtimeInputs: {amdhip64: hip, cudartStatic: hostLibrary}};
    });
    const descriptor = path.join(root, "windows-sdk.json"), output = path.join(root, "out");
    fs.writeFileSync(descriptor, JSON.stringify(selected));
    assert.throws(() => buildNativeWindowsController(descriptor, output, {}), /Windows application input changed/);
    assert.equal(reads, 2, "the frozen manifest must be checked after planning, before full application admission");
    assert.equal(fs.existsSync(output), false, "manifest substitution must fail before creating build output");
  });

  test("the receipt scope is build-only and never claims Windows mining or runtime integration", () => {
    const selected = config(), plan = windowsControllerPlan(selected, "/tmp/mom-windows-controller-scope", repo);
    assert.equal(plan.compile.length, 8); // AOT C plus the seven current maintained host objects.
    assert.equal(plan.link.args.includes("/entry:wmainCRTStartup"), true);
    assert.ok(!plan.sources.includes("packed-windows"));
    assert.ok(plan.compile.every(command => !command.args.includes("/GS-")));
    assert.doesNotMatch(JSON.stringify(plan), /MOM_NATIVE_RUNTIME|cpuCore|gpuInputs/);
    assert.equal(plan.config.target, undefined);
    assert.equal(plan.config.windowsSdk.allowedImports.includes("node.dll"), false);
    assert.equal(plan.compile.every(command => command.args.includes("/MT")), true);
    assert.equal(plan.compile.every(command => command.args.includes("-nostdinc")), true);
    assert.equal(plan.link.args.some(argument => /node|v8|napi/i.test(argument.replace(/^\/nodefaultlib:/i, ""))), false);
  });
});
