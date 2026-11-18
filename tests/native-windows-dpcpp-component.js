"use strict";
const fs = require("node:fs"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {windowsDpcppLinks, windowsDpcppComponentInputs, windowsDpcppSdkInputs, stageWindowsDpcppComponent,
  matchWindowsDpcppComponent, exportWindowsDpcppComponent} = require("../scripts/native-windows-dpcpp-component");
const {fileHash} = require("../scripts/build-source-inputs");
const hostComponentReader = require("../scripts/native-windows-host-component");
const {linkCommand, explicitInputFiles} = require("../scripts/build-native-windows-controller");
const {verifyWindowsApplicationFiles} = require("../scripts/native-windows-application");
const nvrtcProducer = require("../scripts/prepare-native-nvrtc-sdk");
const {required, fixture, vendorNvrtcFixture} = require("./common/native-windows-component");

/** A link-plan fixture, not an installed toolchain or executable. All inputs remain literal.
 * @param {string} root @param {import("../scripts/native-windows-application").WindowsApplication["sdk"] & {cudaImages?: string}} sdk */
function controller(root, sdk) {
  const tool = path.join(root, "tool.exe"), archive = path.join(root, "support.lib");
  return {version: 1, tools: {clangCl: tool, lldLink: tool, shermaot: tool, llvmNm: tool, llvmReadobj: tool, clangResourceInclude: root},
    hermes: {source: root, build: root, shermaot: tool},
    windowsSdk: {overlay: path.join(root, "overlay.json"), sourceRoot: root, includeRoots: [root], libraryRoots: [root],
      msvcLibraries: [archive], ucrtLibraries: [archive], importLibraries: [archive], allowedImports: ["kernel32.dll"]},
    libraries: {sourceInclude: root, ssl: archive, crypto: archive, libuv: archive, zlib: archive},
    application: {core: path.join(root, "cpu-core.lib"), gpuReceipts: [], defaultCompiler: "oneapi", tools: {llvmAr: tool, llvmObjcopy: tool}, sdk}};
}

test.describe("Windows original runtime/JIT component boundary", () => {
  test("retains complete producer order and the two constructor/object roots", t => {
    const f = fixture(t), value = windowsDpcppLinks(f.links, false); assert.equal(value, f.links);
    const r = f.produce(); assert.deepEqual(r.link.libraries.map(n => path.basename(n)), f.links.libraries.map(n => path.basename(n)));
    assert.deepEqual(r.link.wholeArchives.map(n => path.basename(n)), ["mom-sycl-static.lib", "ze_loader.lib"]);
    assert.deepEqual(r.link.objects.map(n => path.basename(n)), ["drivers.obj", "driver-aliases.obj"]);
  });
  test("cannot silently export a JIT-disabled runtime fragment", t => {
    const f = fixture(t); f.links.libraries = f.links.libraries.filter(n => !n.endsWith("sycl-jit.lib")); assert.throws(() => f.produce(), /Incomplete|membership/);
  });
  test("omitting any required runtime feature fails before creating output", t => {
    const f = fixture(t), original = [...f.links.libraries];
    for (const name of required) {f.links.libraries = original.filter(n => path.basename(n) !== name + ".lib"); assert.throws(() => f.produce(), /Incomplete|membership/); assert.equal(fs.existsSync(f.output), false);}
  });
  test("constructor roots cannot become ordinary archives", t => {
    const f = fixture(t); const n = f.links.wholeArchives.pop(); assert.ok(n); f.links.libraries.push(n); assert.throws(() => f.produce(), /Incomplete/);
  });
  test("driver aliases must stay explicit objects rather than archive-only members", t => {
    const f = fixture(t); f.links.objects.pop(); assert.throws(() => f.produce(), /Incomplete/);
  });
  test("case-insensitive archive basename collisions are rejected on Linux too", t => {
    const f = fixture(t); f.links.libraries.push(path.join(f.input, "CUDA.lib")); assert.throws(() => f.produce(), /membership/);
  });
  test("installed manifests allow only confined literal relative member names", t => {
    const f = fixture(t); f.produce();
    for (const name of ["lib/../escape.lib", "/outside.lib", "lib/evil\\name.lib", "lib/name.lib\n", "lib/payload.dll"]) {
      const link = {libraries: f.links.libraries.map(n => "lib/" + path.basename(n)), wholeArchives: ["lib/mom-sycl-static.lib", "lib/ze_loader.lib"], objects: ["objects/drivers.obj", "objects/driver-aliases.obj"]};
      link.libraries[20] = name; assert.throws(() => windowsDpcppLinks(link, true), /membership/);
    }
  });
  test("unexpected metadata and oversized closures fail without filesystem access", t => {
    const f = fixture(t); assert.throws(() => windowsDpcppLinks({...f.links, injected: true}, false), /Incomplete/);
    assert.throws(() => windowsDpcppLinks({...f.links, libraries: Array(513).fill("/lib.lib")}, false), /Incomplete/);
  });
  test("export requires explicit build-only admission before opening receipts", () => {
    assert.throws(() => exportWindowsDpcppComponent("/absent", "/cuda", "/output", {}), /MOM_AOT_BUILD_ONLY/);
  });
  test("warm matching preserves the original producer, payload order and receipt bytes", t => {
    const f = fixture(t); f.produce();
    const manifest = path.join(f.output, "component.json"), before = fs.readFileSync(manifest),
      original = {links: f.links, producer: {path: "/original/receipt.json", sha256: "1".repeat(64)},
        files: Object.fromEntries(Object.values(f.links).flat().map(file => [file, fileHash(file)]))};
    assert.deepEqual(matchWindowsDpcppComponent(f.output, original), windowsDpcppComponentInputs(f.output));
    assert.deepEqual(fs.readFileSync(manifest), before);
    assert.throws(() => matchWindowsDpcppComponent(f.output, {...original, producer: {...original.producer, sha256: "2".repeat(64)}}), /another original producer/);
    assert.throws(() => matchWindowsDpcppComponent(f.output, {...original,
      links: {...f.links, libraries: [...f.links.libraries].reverse()}}), /ordered payloads/);
  });
  test("self-consistent installed hashes cannot replace the admitted original payload", t => {
    const f = fixture(t); f.produce();
    const original = {links: f.links, producer: {path: "/original/receipt.json", sha256: "1".repeat(64)},
        files: Object.fromEntries(Object.values(f.links).flat().map(file => [file, fileHash(file)]))},
      changed = path.join(f.output, "lib/LLVM0.lib");
    fs.appendFileSync(changed, "changed");
    f.edit(value => {value.files["lib/LLVM0.lib"] = fileHash(changed);});
    assert.ok(windowsDpcppComponentInputs(f.output));
    assert.throws(() => matchWindowsDpcppComponent(f.output, original), /ordered payloads/);
  });
  test("an original payload changed after producer admission invalidates warm reuse", t => {
    const f = fixture(t); f.produce();
    const original = {links: f.links, producer: {path: "/original/receipt.json", sha256: "1".repeat(64)},
      files: Object.fromEntries(Object.values(f.links).flat().map(file => [file, fileHash(file)]))};
    fs.appendFileSync(f.links.libraries[0] || "", "changed");
    assert.throws(() => matchWindowsDpcppComponent(f.output, original), /provenance changed/);
  });
});

test.describe("Self-contained Windows static component transport", () => {
  test("exported links survive removal of original SDK paths and relocation of the component", t => {
    const f = fixture(t); f.produce(); fs.rmSync(f.input, {recursive: true}); const moved = path.join(f.root, "moved"); fs.renameSync(f.output, moved);
    const r = windowsDpcppComponentInputs(moved); assert.ok(r.link.libraries.every(n => n.startsWith(moved + path.sep))); assert.equal(Object.keys(r.files).length, 69);
  });
  test("all archives, objects and the manifest participate in the consumer fingerprint", t => {
    const f = fixture(t), r = f.produce(); assert.equal(Object.keys(r.files).length, 69); assert.ok(r.files[path.join(f.output, "component.json")]);
  });
  test("modified, missing or unrecorded payloads are refused", t => {
    const f = fixture(t); f.produce(); const p = path.join(f.output, "lib/LLVM0.lib"); fs.appendFileSync(p, "changed"); assert.throws(() => windowsDpcppComponentInputs(f.output), /changed/);
    fs.writeFileSync(p, "!<arch>\n"); const extra = path.join(f.output, "lib/extra.lib"); fs.writeFileSync(extra, "!<arch>\n"); assert.throws(() => windowsDpcppComponentInputs(f.output), /Unrecorded/);
    fs.unlinkSync(extra); fs.unlinkSync(p); assert.throws(() => windowsDpcppComponentInputs(f.output), /ENOENT/);
  });
  test("manifest membership must cover every payload exactly once", t => {
    const f = fixture(t); f.produce(); f.edit(r => {delete r.files["lib/LLVM0.lib"];}); assert.throws(() => windowsDpcppComponentInputs(f.output), /membership differs/);
  });
  test("wrong target, pins or qualification claims cannot be reclassified by transport", t => {
    const f = fixture(t); f.produce(); const p = path.join(f.output, "component.json"), original = fs.readFileSync(p);
    for (const change of [/** @param {any} r */ r => {r.kind = "dpcpp-linux";}, /** @param {any} r */ r => {r.pins.source = "0".repeat(40);}, /** @param {any} r */ r => {r.publicationQualified = true;}]) {
      fs.writeFileSync(p, original); f.edit(change); assert.throws(() => windowsDpcppComponentInputs(f.output), /receipt/);
    }
  });
  test("malformed COFF helpers cannot be accepted by matching their hash alone", t => {
    const f = fixture(t); fs.writeFileSync(f.links.objects[0] || "", "not COFF"); assert.throws(() => f.produce(), /COFF/); assert.equal(fs.existsSync(f.output), false);
  });
  test("existing output is preserved rather than silently overwritten", t => {
    const f = fixture(t); f.produce(); assert.throws(() => f.produce(), /new and canonical/); assert.ok(windowsDpcppComponentInputs(f.output));
  });
  test("source/file aliases and a symlink manifest are refused", {skip: process.platform === "win32"}, t => {
    const f = fixture(t); f.produce(); const p = path.join(f.output, "lib/LLVM0.lib"), old = path.join(f.root, "same.lib"); fs.renameSync(p, old); fs.symlinkSync(old, p);
    assert.throws(() => windowsDpcppComponentInputs(f.output), /changed/); fs.unlinkSync(p); fs.renameSync(old, p);
    const m = path.join(f.output, "component.json"), target = path.join(f.root, "manifest.json"); fs.renameSync(m, target); fs.symlinkSync(target, m);
    assert.throws(() => windowsDpcppComponentInputs(f.output), /regular local/);
  });
  test("an export cannot replace an input directory or write inside it", t => {
    const f = fixture(t); assert.throws(() => stageWindowsDpcppComponent(f.links, {path: "/p", sha256: "0".repeat(64)}, path.join(f.input, "nested")), /overlaps/);
  });
});

test.describe("Windows component source-build integration", () => {
  test("ordinary explicit SDK links retain their exact order without a component", t => {
    const f = fixture(t), r = windowsDpcppSdkInputs(f.links); assert.deepEqual(r, {link: f.links, files: {}});
    assert.notEqual(r.link.libraries, f.links.libraries);
  });
  test("CUDA image selection reaches link selection without taking over image admission", t => {
    const f = fixture(t), imageInclude = path.join(f.root, "cuda-images.inc"),
      sdk = {...f.links, includeRoots: [f.root], nvmlHeader: path.join(f.root, "nvml.h"),
        hipDriverLibrary: path.join(f.root, "amdhip64.lib"), cudaImages: imageInclude}, original = structuredClone(sdk),
      selected = windowsDpcppSdkInputs(sdk), config = controller(f.root, sdk), withoutImages = controller(f.root, {...sdk});
    delete withoutImages.application.sdk.cudaImages;
    assert.deepEqual(selected.link, f.links); assert.deepEqual(selected.files, {}); assert.deepEqual(sdk, original);
    assert.ok(explicitInputFiles(config).includes(sdk.nvmlHeader));
    assert.deepEqual(linkCommand(config, [], "/output.exe", "/output.map"),
      linkCommand(withoutImages, [], "/output.exe", "/output.map"));
  });
  test("CUDA image include selection is absolute, .inc-only, and exclusive with vendor NVRTC", t => {
    const f = fixture(t);
    for (const value of [undefined, null, "relative.inc", path.join(f.root, "images.bin"), path.join(f.root, "bad\tpath.inc")]) {
      assert.throws(() => windowsDpcppSdkInputs({...f.links, cudaImages: /** @type {string} */ (value)}), /absolute Windows CUDA image include/);
    }
    assert.throws(() => windowsDpcppSdkInputs({...f.links, cudaImages: path.join(f.root, "images.inc"), nvrtcSdk: path.join(f.root, "nvrtc")}),
      /mutually exclusive/);
  });
  test("vendor NVRTC receipt binds all three explicit libraries and original provenance without changing link order", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), vendor = vendorNvrtcFixture(f.root), before = path.join(f.root, "bcrypt.lib"), after = path.join(f.root, "cudart.lib");
    t.mock.method(nvrtcProducer, "nvrtcSdkInputs", (/** @type {"linux" | "windows"} */ platform, /** @type {string} */ root) => {
      assert.equal(platform, "windows"); assert.equal(root, vendor.root);
      return {receipt: vendor.receipt, receiptFile: vendor.receiptFile, receiptSha256: vendor.receiptSha256, files: vendor.files};
    });
    const sdk = {libraries: [before, ...vendor.libraries, after], wholeArchives: [], objects: [], nvrtcSdk: vendor.root}, selected = windowsDpcppSdkInputs(sdk);
    assert.deepEqual(selected.link.libraries, sdk.libraries);
    assert.deepEqual(selected.link.wholeArchives, []); assert.deepEqual(selected.link.objects, []);
    for (const [file, hash] of Object.entries(vendor.files)) {assert.equal(selected.files[file], hash);}
  });
  test("vendor NVRTC root requires each exact admitted static library once", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), vendor = vendorNvrtcFixture(f.root);
    t.mock.method(nvrtcProducer, "nvrtcSdkInputs", () => ({
      receipt: vendor.receipt, receiptFile: vendor.receiptFile, receiptSha256: vendor.receiptSha256, files: vendor.files
    }));
    const sdk = {libraries: vendor.libraries, wholeArchives: [], objects: [], nvrtcSdk: vendor.root};
    assert.throws(() => windowsDpcppSdkInputs({...sdk, libraries: vendor.libraries.slice(1)}), /select each admitted vendor NVRTC archive exactly once/);
    const alias = path.join(f.root, "old-sdk", "nvrtc_static.lib"); fs.mkdirSync(path.dirname(alias)); fs.writeFileSync(alias, "!<arch>\n");
    assert.throws(() => windowsDpcppSdkInputs({...sdk, libraries: [...vendor.libraries, alias]}), /select each admitted vendor NVRTC archive exactly once/);
  });
  test("a Linux-bound NVRTC root fails clearly on a native Windows host", t => {
    const f = fixture(t), platform = Object.getOwnPropertyDescriptor(process, "platform"); assert.ok(platform);
    Object.defineProperty(process, "platform", {value: "win32", configurable: true});
    try {
      assert.throws(() => windowsDpcppSdkInputs({...f.links, nvrtcSdk: path.join(f.root, "nvrtc-sdk")}), /Linux-bound.*Linux cross-build host/);
    } finally {Object.defineProperty(process, "platform", platform);}
  });
  test("one component root supplies complete runtime/JIT links and hashes", t => {
    const f = fixture(t), installed = f.produce(), r = windowsDpcppSdkInputs({libraries: [], wholeArchives: [], objects: [], dpcppComponent: f.output});
    assert.deepEqual(r, installed);
  });
  test("component inputs append to the remaining SDK without mutating the descriptor", t => {
    const f = fixture(t), installed = f.produce(), sdk = {libraries: [path.join(f.root, "other.lib")], wholeArchives: [], objects: [], dpcppComponent: f.output};
    const original = structuredClone(sdk), r = windowsDpcppSdkInputs(sdk);
    assert.deepEqual(r.link.libraries, [...sdk.libraries, ...installed.link.libraries]); assert.deepEqual(sdk, original);
  });
  test("host libraries follow explicit extras and precede the DPC++ runtime/JIT closure", t => {
    const f = fixture(t), installed = f.produce(), hostRoot = path.join(f.root, "host-component");
    const hostLibrary = path.join(hostRoot, "lib", "acpp.lib"), hostManifest = path.join(hostRoot, "component.json");
    const hostFiles = {[hostManifest]: "a".repeat(64), [hostLibrary]: "b".repeat(64)};
    t.mock.method(hostComponentReader, "windowsHostComponentInputs", (/** @type {string} */ root) => {
      assert.equal(root, hostRoot); return {link: {libraries: [hostLibrary]}, files: hostFiles, provenance: {}};
    });
    const extra = path.join(f.root, "extra.lib"), sdk = {libraries: [extra], wholeArchives: [], objects: [], hostComponent: hostRoot, dpcppComponent: f.output};
    const selected = windowsDpcppSdkInputs(sdk);
    assert.deepEqual(selected.link.libraries, [extra, hostLibrary, ...installed.link.libraries]);
    assert.deepEqual(selected.link.wholeArchives, installed.link.wholeArchives); assert.deepEqual(selected.link.objects, installed.link.objects);
    assert.equal(selected.files[hostManifest], hostFiles[hostManifest]); assert.equal(selected.files[hostLibrary], hostFiles[hostLibrary]);
    for (const [file, hash] of Object.entries(installed.files)) {assert.equal(selected.files[file], hash);}
  });
  test("host component roots are absolute and SDK selection fields are closed", t => {
    const f = fixture(t);
    for (const root of [undefined, null, "relative", "", 1]) {
      assert.throws(() => windowsDpcppSdkInputs({...f.links, hostComponent: /** @type {string} */ (root)}), /absolute Windows host component root/);
    }
    assert.throws(() => windowsDpcppSdkInputs(/** @type {any} */ ({...f.links, injected: true})), /Invalid Windows SDK component selections/);
  });
  test("overlapping host and DPC++ component receipt cohorts are rejected", t => {
    const f = fixture(t); f.produce();
    const manifest = path.join(f.output, "component.json"), hostRoot = path.join(f.root, "host-component");
    t.mock.method(hostComponentReader, "windowsHostComponentInputs", () => ({link: {libraries: []}, files: {[manifest]: "a".repeat(64)}, provenance: {}}));
    assert.throws(() => windowsDpcppSdkInputs({libraries: [], wholeArchives: [], objects: [], hostComponent: hostRoot, dpcppComponent: f.output}), /Duplicate Windows SDK component receipt\/input/);
  });
  test("constructor archives and driver objects stay in their original link categories", t => {
    const f = fixture(t); f.produce(); const r = windowsDpcppSdkInputs({libraries: [], wholeArchives: [], objects: [], dpcppComponent: f.output});
    assert.equal(r.link.wholeArchives.length, 2); assert.equal(r.link.objects.length, 2); assert.ok(!r.link.libraries.some(n => n.endsWith("ze_loader.lib")));
  });
  test("explicit undefined, null and relative component selections cannot fall back", t => {
    const f = fixture(t);
    for (const value of [undefined, null, "relative", "", 1, path.join(f.root, "tab\troot")]) {
      assert.throws(() => windowsDpcppSdkInputs({...f.links, dpcppComponent: /** @type {string} */ (value)}), /component root/);
    }
  });
  test("a missing selected component cannot silently reuse only the remaining libraries", t => {
    const f = fixture(t); assert.throws(() => windowsDpcppSdkInputs({...f.links, dpcppComponent: f.output}), /ENOENT/);
  });
  test("component mutation is detected through the final application file check", t => {
    const f = fixture(t); f.produce(); const r = windowsDpcppSdkInputs({libraries: [], wholeArchives: [], objects: [], dpcppComponent: f.output});
    f.edit(r => {r.producer.sha256 = "2".repeat(64);}); assert.throws(() => verifyWindowsApplicationFiles(r.files), /input changed/);
  });
  test("an explicit archive cannot duplicate a component payload", t => {
    const f = fixture(t), installed = f.produce();
    assert.throws(() => windowsDpcppSdkInputs({libraries: [installed.link.libraries[0] || ""], wholeArchives: [], objects: [], dpcppComponent: f.output}), /Duplicate/);
  });
  test("case-only path aliases and cross-category duplicates are rejected", t => {
    const f = fixture(t), library = f.links.libraries[0] || "";
    assert.throws(() => windowsDpcppSdkInputs({...f.links, libraries: [...f.links.libraries, library.toUpperCase()]}), /Duplicate/);
    assert.throws(() => windowsDpcppSdkInputs({...f.links, wholeArchives: [...f.links.wholeArchives, library]}), /Duplicate/);
  });
  test("independently isolated runtimes may retain the same archive basename", t => {
    const f = fixture(t), library = f.links.libraries[0] || "", other = path.join(f.root, "other", path.basename(library));
    const r = windowsDpcppSdkInputs({...f.links, libraries: [...f.links.libraries, other]}); assert.equal(r.link.libraries.at(-1), other);
  });
  test("merged lists stay bounded and require literal static archive/object paths", t => {
    const f = fixture(t); f.produce();
    assert.throws(() => windowsDpcppSdkInputs({...f.links, libraries: Array(513).fill("/file.lib")}), /bounded/);
    for (const file of ["/payload.dll", "/payload\t.obj", "/payload\n.obj", "/payload\x7f.obj"]) {
      assert.throws(() => windowsDpcppSdkInputs({...f.links, objects: [file]}), /bounded/);
    }
  });
  test("controller inputs retain host receipt and linker order across component expansion", t => {
    const f = fixture(t), installed = f.produce(), hostRoot = path.join(f.root, "host-component"),
      hostLibrary = path.join(hostRoot, "acpp.lib"), hostManifest = path.join(hostRoot, "component.json");
    t.mock.method(hostComponentReader, "windowsHostComponentInputs", () => ({link: {libraries: [hostLibrary]},
      files: {[hostManifest]: "a".repeat(64), [hostLibrary]: "b".repeat(64)}, provenance: {},
      runtimeInputs: {amdhip64: path.join(f.root, "amdhip64.lib"), cudartStatic: hostLibrary}}));
    const common = {includeRoots: [f.root], nvmlHeader: path.join(f.root, "nvml.h"), hipDriverLibrary: path.join(f.root, "amdhip64.lib")};
    const compact = controller(f.root, {...common, libraries: [], wholeArchives: [], objects: [], hostComponent: hostRoot, dpcppComponent: f.output});
    const expanded = controller(f.root, {...common, libraries: [hostLibrary, ...installed.link.libraries], wholeArchives: installed.link.wholeArchives, objects: installed.link.objects});
    assert.deepEqual(linkCommand(compact, [], "/output.exe", "/output.map"), linkCommand(expanded, [], "/output.exe", "/output.map"));
    assert.ok(explicitInputFiles(compact).includes(path.join(f.output, "component.json")));
    assert.ok(explicitInputFiles(compact).includes(hostManifest)); assert.ok(explicitInputFiles(compact).includes(hostLibrary));
  });
  test("application driver discovery cannot select a different HIP import archive", t => {
    const f = fixture(t), hostRoot = path.join(f.root, "host-component"),
      driver = path.join(hostRoot, "runtime", "amdhip64.lib"), cudart = path.join(hostRoot, "runtime", "cudart_static.lib");
    t.mock.method(hostComponentReader, "windowsHostComponentInputs", () => ({link: {libraries: [cudart]},
      files: {[driver]: "a".repeat(64), [cudart]: "b".repeat(64)}, provenance: {},
      runtimeInputs: {amdhip64: driver, cudartStatic: cudart}}));
    const sdk = {libraries: [], wholeArchives: [], objects: [], includeRoots: [f.root],
      nvmlHeader: path.join(f.root, "nvml.h"), hipDriverLibrary: driver, hostComponent: hostRoot};
    const selected = windowsDpcppSdkInputs(sdk);
    assert.deepEqual(selected.link.libraries, [cudart]);
    assert.equal(selected.files[driver], "a".repeat(64));
    assert.throws(() => windowsDpcppSdkInputs({...sdk, hipDriverLibrary: path.join(f.root, "other-amdhip64.lib")}),
      /HIP driver library must match/);
    assert.throws(() => windowsDpcppSdkInputs({...sdk, libraries: [cudart]}), /Duplicate Windows SDK/);
  });
});
