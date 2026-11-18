"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const test = require("node:test"), assert = require("node:assert/strict");
const {fileHash} = require("../scripts/build-source-inputs");
const {archiveSha256} = require("../scripts/build-native-windows-llvm21-sdk");
const {revision: comgrRevision, hostSources} = require("../scripts/build-native-windows-comgr-sdk");
const {clrRevision: hiprtcSourceRevision, sourceFiles: hiprtcSourceFiles} = require("../scripts/build-native-windows-hiprtc-sdk");
const {exportWindowsHostComponent, windowsHostComponentInputs, windowsHostComponentLlvmLayout, copyRuntimeInput,
  matchWindowsHostComponent} = require("../scripts/native-windows-host-component");

const acppRevision = "da2463e45aa90aa36306c45abcfc05b87de51bc6";
const hiprtcArchiveName = "hiprtc-host-intel.lib";
const hiprtcArchiveMembers = [...hiprtcSourceFiles.map(file => path.basename(file, ".cpp") + ".obj"), "hipRTC_header.obj"];
const pins = {version: 3, targetTriple: "x86_64-pc-windows-msvc", linkOrder: ["acpp", "hiprtc", "comgr", "llvm"],
  llvm: {sourceArchiveSha256: archiveSha256, archives: 164},
  comgr: {sourceRevision: comgrRevision, archives: 1, objects: hostSources.length},
  acpp: {sourceRevision: acppRevision, archives: 9},
  hiprtc: {sourceRevision: hiprtcSourceRevision, comgrRevision, archiveName: hiprtcArchiveName, archives: 1, objects: hiprtcArchiveMembers.length}};
const groups = ["acpp", "hiprtc", "comgr", "llvm"];
const generationId = "123e4567-e89b-42d3-a456-426614174000";

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-windows-host-component-"));
  const sourceRoot = root + "-source"; fs.mkdirSync(sourceRoot);
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  t.after(() => fs.rmSync(sourceRoot, {recursive: true, force: true}));
  const link = /** @type {Record<string, string[]>} */ ({});
  const files = /** @type {Record<string, {sha256: string, bytes: number}>} */ ({});
  const counts = /** @type {Record<string, number>} */ ({acpp: 9, hiprtc: 1, comgr: 1, llvm: 164});
  for (const group of groups) {
    const count = counts[group];
    if (count === undefined) {throw new Error("Missing fixture archive count");}
    fs.mkdirSync(path.join(root, "lib", group), {recursive: true});
    const rows = /** @type {string[]} */ ([]);
    link[group] = rows;
    for (let index = 0; index < count; index++) {
      const basename = group === "comgr" ? "amd_comgr.lib" : group === "hiprtc" ? hiprtcArchiveName : group + "-" + String(index).padStart(3, "0") + ".lib";
      const relative = "lib/" + group + "/" + String(index).padStart(3, "0") + "-" + basename;
      const file = path.join(root, relative); fs.writeFileSync(file, "!<arch>\n");
      rows.push(relative); files[relative] = {sha256: fileHash(file), bytes: fs.statSync(file).size};
    }
  }
  fs.mkdirSync(path.join(root, "runtime"));
  const runtimeInputs = /** @type {Record<string, {path:string,bytes:number,sha256:string}>} */ ({});
  for (const [name, basename] of Object.entries({cudartStatic: "cudart_static.lib", amdhip64: "amdhip64.lib"})) {
    const relative = "runtime/" + basename, source = path.join(sourceRoot, basename); fs.writeFileSync(source, "!<arch>\n");
    const proof = copyRuntimeInput({file: source, bytes: fs.statSync(source).size, sha256: fileHash(source)}, relative, root, files);
    runtimeInputs[name] = proof;
  }
  const comgrArchiveSha256 = files[link["comgr"]?.[0] || ""]?.sha256 || "e".repeat(64),
    hiprtcArchiveSha256 = files[link["hiprtc"]?.[0] || ""]?.sha256 || "0".repeat(64), resourceReceiptSha256 = "1".repeat(64);
  const manifest = {version: 3, kind: "windows-native-host-sdk", pins, link, runtimeInputs, files,
    provenance: {llvm: {generationId, receiptSha256: "a".repeat(64), admissionSha256: "b".repeat(64), objectProofSha256: "c".repeat(64)},
      comgr: {receiptSha256: "d".repeat(64), archiveSha256: comgrArchiveSha256, llvmGenerationId: generationId},
      acpp: {receiptSha256: "f".repeat(64), sourceRevision: acppRevision, llvmGenerationId: generationId,
        rocmResourceReceiptSha256: resourceReceiptSha256},
      hiprtc: {receiptSha256: "2".repeat(64), archiveSha256: hiprtcArchiveSha256, sourceRevision: hiprtcSourceRevision,
        comgrRevision, comgrReceiptSha256: "d".repeat(64), comgrArchiveSha256, resourceReceiptSha256, archiveMembers: hiprtcArchiveMembers}},
    buildOnly: true, publicationQualified: false};
  const manifestFile = path.join(root, "component.json"); fs.writeFileSync(manifestFile, JSON.stringify(manifest, null, 2) + "\n");
  /** @param {(value: any) => void} change */
  const edit = change => {const current = JSON.parse(fs.readFileSync(manifestFile, "utf8")); change(current); fs.writeFileSync(manifestFile, JSON.stringify(current, null, 2) + "\n");};
  const sources = Object.fromEntries(groups.map(group => [group, (link[group] || []).map(relative => {
    const file = path.join(sourceRoot, path.basename(relative).replace(/^\d{3}-/, ""));
    fs.copyFileSync(path.join(root, relative), file);
    return {file, sha256: fileHash(file), bytes: fs.statSync(file).size};
  })]));
  const originalRuntime = Object.fromEntries(Object.keys(runtimeInputs).map(name => {
    const file = path.join(sourceRoot, path.basename(runtimeInputs[name]?.path || ""));
    return [name, {file, sha256: fileHash(file), bytes: fs.statSync(file).size}];
  }));
  const original = /** @type {ReturnType<typeof import("../scripts/native-windows-host-component").verifyWindowsHostComponentProducers>} */
    (/** @type {unknown} */ ({sources, provenance: manifest.provenance, acpp: {runtimeInputs: originalRuntime},
      files: Object.fromEntries([...Object.values(sources).flat(), ...Object.values(originalRuntime)].map(row => [row.file, row.sha256]))}));
  return {root, manifest, manifestFile, edit, original};
}

test.describe("Relocatable Windows host SDK component", () => {
  test("warm matching joins original producer order and hashes without rewriting receipts", t => {
    const f = fixture(t), before = fs.readFileSync(f.manifestFile);
    assert.equal(matchWindowsHostComponent(f.root, f.original).link.libraries.length, 176);
    assert.deepEqual(fs.readFileSync(f.manifestFile), before);
    f.edit(value => {value.link.acpp.reverse();});
    assert.throws(() => matchWindowsHostComponent(f.root, f.original), /archive order/);
  });
  test("warm matching rejects original producer mutation", t => {
    const f = fixture(t), source = f.original.sources.acpp[0];
    assert.ok(source); fs.appendFileSync(source.file, "changed");
    assert.throws(() => matchWindowsHostComponent(f.root, f.original), /original producer.*changed/);
  });
  test("self-consistent component payload edits cannot replace original producer hashes", t => {
    const f = fixture(t), relative = f.manifest.link["acpp"]?.[0]; assert.ok(relative);
    const file = path.join(f.root, relative); fs.appendFileSync(file, "changed");
    f.edit(value => {value.files[relative] = {sha256: fileHash(file), bytes: fs.statSync(file).size};});
    assert.throws(() => matchWindowsHostComponent(f.root, f.original), /payload differs/);
  });
  test("admits the reviewed archives and CUDART after ACPP, HIPRTC, COMGR and LLVM", t => {
    const f = fixture(t), selected = windowsHostComponentInputs(f.root);
    assert.equal(selected.link.libraries.length, 176);
    assert.deepEqual(selected.link.libraries.slice(0, 2).map(file => path.basename(file)), ["000-acpp-000.lib", "001-acpp-001.lib"]);
    assert.equal(path.basename(selected.link.libraries[9] || ""), "000-hiprtc-host-intel.lib");
    assert.equal(path.basename(selected.link.libraries[10] || ""), "000-amd_comgr.lib");
    assert.equal(path.basename(selected.link.libraries[11] || ""), "000-llvm-000.lib");
    assert.equal(path.basename(selected.link.libraries[175] || ""), "cudart_static.lib");
    assert.deepEqual(selected.provenance.hiprtc.archiveMembers, hiprtcArchiveMembers);
    const cudart = selected.runtimeInputs["cudartStatic"], amdhip64 = selected.runtimeInputs["amdhip64"];
    assert.ok(cudart && amdhip64);
    assert.equal(path.basename(cudart), "cudart_static.lib");
    assert.equal(path.basename(amdhip64), "amdhip64.lib");
    assert.equal(selected.link.libraries.filter(file => file === cudart).length, 1);
    assert.ok(!selected.link.libraries.includes(amdhip64));
  });
  test("returns only relative-component paths and hashes, never producer SDK paths", t => {
    const f = fixture(t), selected = windowsHostComponentInputs(f.root);
    assert.ok(selected.link.libraries.every(file => file.startsWith(f.root + path.sep)));
    assert.equal(Object.keys(selected.files).length, 178);
    assert.equal(selected.files[f.manifestFile], fileHash(f.manifestFile));
    assert.equal(selected.provenance.llvm.generationId, generationId);
    assert.ok(Object.values(selected.runtimeInputs).every(file => file.startsWith(f.root + path.sep)));
  });
  test("rejects producer generations that no longer agree", t => {
    const f = fixture(t); f.edit(value => {value.provenance.acpp.llvmGenerationId = "223e4567-e89b-42d3-a456-426614174000";});
    assert.throws(() => windowsHostComponentInputs(f.root), /provenance/);
  });
  test("rejects HIPRTC provenance that is not tied to selected COMGR and ROCm resources", t => {
    const f = fixture(t); f.edit(value => {value.provenance.hiprtc.comgrReceiptSha256 = "3".repeat(64);});
    assert.throws(() => windowsHostComponentInputs(f.root), /provenance/);
    f.edit(value => {
      value.provenance.hiprtc.comgrReceiptSha256 = value.provenance.comgr.receiptSha256;
      value.provenance.hiprtc.resourceReceiptSha256 = "4".repeat(64);
    });
    assert.throws(() => windowsHostComponentInputs(f.root), /provenance/);
  });
  test("requires dynamic generation identifiers without pinning a private build run", t => {
    const f = fixture(t); f.edit(value => {value.provenance.llvm.generationId = "not-a-generation";});
    assert.throws(() => windowsHostComponentInputs(f.root), /provenance/);
  });
  test("rejects changed archives even if archive magic remains intact", t => {
    const f = fixture(t), selected = windowsHostComponentInputs(f.root), archive = selected.link.libraries[0];
    fs.appendFileSync(archive || "", "changed"); assert.throws(() => windowsHostComponentInputs(f.root), /changed/);
  });
  test("rejects a changed HIPRTC archive against its producer pin", t => {
    const f = fixture(t), selected = windowsHostComponentInputs(f.root), archive = selected.link.libraries[9];
    fs.appendFileSync(archive || "", "changed"); assert.throws(() => windowsHostComponentInputs(f.root), /differs|changed/);
  });
  test("rejects a missing selected archive", t => {
    const f = fixture(t), selected = windowsHostComponentInputs(f.root); fs.unlinkSync(selected.link.libraries[0] || "");
    assert.throws(() => windowsHostComponentInputs(f.root), /canonical regular file/);
  });
  test("rejects unrecorded archives and extra files", t => {
    const f = fixture(t); fs.writeFileSync(path.join(f.root, "lib", "llvm", "unrecorded.lib"), "!<arch>\n");
    assert.throws(() => windowsHostComponentInputs(f.root), /Unrecorded/);
  });
  test("requires exact relative membership for every archive group", t => {
    const f = fixture(t); f.edit(value => {value.link.llvm.pop();});
    assert.throws(() => windowsHostComponentInputs(f.root), /archive group/);
  });
  test("requires exactly one HIPRTC archive at its fixed role path", t => {
    const missing = fixture(t); missing.edit(value => {value.link.hiprtc.pop();});
    assert.throws(() => windowsHostComponentInputs(missing.root), /archive group/);
    const misplaced = fixture(t); misplaced.edit(value => {value.link.hiprtc[0] = "lib/hiprtc/000-other.lib";});
    assert.throws(() => windowsHostComponentInputs(misplaced.root), /HIPRTC component archive/);
  });
  test("rejects a file assigned to the wrong producer group", t => {
    const f = fixture(t); f.edit(value => {value.link.comgr[0] = value.link.llvm[0];});
    assert.throws(() => windowsHostComponentInputs(f.root), /archive path/);
  });
  test("rejects duplicate archive membership", t => {
    const f = fixture(t); f.edit(value => {value.link.acpp[1] = value.link.acpp[0];});
    assert.throws(() => windowsHostComponentInputs(f.root), /duplicate/);
  });
  test("rejects traversal, absolute names and non-library suffixes", t => {
    const f = fixture(t);
    for (const name of ["lib/llvm/../../escape.lib", "/outside.lib", "lib/llvm/payload.dll"]) {
      f.edit(value => {value.link.llvm[0] = name;});
      assert.throws(() => windowsHostComponentInputs(f.root), /Invalid Windows host component|archive path/);
    }
  });
  test("archive proofs require exact hashes and bounded positive sizes", t => {
    const f = fixture(t);
    f.edit(value => {value.files[value.link.acpp[0]].sha256 = "z".repeat(64);});
    assert.throws(() => windowsHostComponentInputs(f.root), /proof/);
    f.edit(value => {value.files[value.link.acpp[0]].sha256 = "a".repeat(64); value.files[value.link.acpp[0]].bytes = 0;});
    assert.throws(() => windowsHostComponentInputs(f.root), /proof/);
    f.edit(value => {value.files[value.link.acpp[0]].bytes = 256 * 1024 * 1024 + 1;});
    assert.throws(() => windowsHostComponentInputs(f.root), /proof/);
  });
  test("rejects additional manifest claims and qualified runtime state", t => {
    const f = fixture(t); f.edit(value => {value.executed = true;});
    assert.throws(() => windowsHostComponentInputs(f.root), /receipt/);
    f.edit(value => {delete value.executed; value.publicationQualified = true;});
    assert.throws(() => windowsHostComponentInputs(f.root), /receipt/);
  });
  test("requires the exact reviewed source versions and closure pins", t => {
    const f = fixture(t); f.edit(value => {value.pins.comgr.sourceRevision = "0".repeat(40);});
    assert.throws(() => windowsHostComponentInputs(f.root), /receipt/);
  });
  test("rejects symlink payloads and a symlinked component root", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), selected = windowsHostComponentInputs(f.root), archive = selected.link.libraries[0] || "";
    const moved = path.join(f.root, "payload.lib"); fs.renameSync(archive, moved); fs.symlinkSync(moved, archive);
    assert.throws(() => windowsHostComponentInputs(f.root), /canonical regular file/);
    fs.unlinkSync(archive); fs.renameSync(moved, archive);
    const rootLink = path.join(path.dirname(f.root), path.basename(f.root) + "-link"); fs.symlinkSync(f.root, rootLink);
    t.after(() => fs.unlinkSync(rootLink)); assert.throws(() => windowsHostComponentInputs(rootLink), /canonical Windows host component root/);
  });
  test("rejects unrecorded root metadata", t => {
    const f = fixture(t); fs.writeFileSync(path.join(f.root, "README"), "extra\n");
    assert.throws(() => windowsHostComponentInputs(f.root), /Unrecorded/);
  });
  test("rejects the old component version or missing required HIPRTC/runtime inputs", t => {
    const f = fixture(t); f.edit(value => {value.version = 2; value.pins.version = 2;});
    assert.throws(() => windowsHostComponentInputs(f.root), /receipt/);
    f.edit(value => {value.version = 3; value.pins.version = 3; delete value.link.hiprtc;});
    assert.throws(() => windowsHostComponentInputs(f.root), /receipt/);
    f.edit(value => {value.link.hiprtc = ["lib/hiprtc/000-hiprtc-host-intel.lib"]; delete value.runtimeInputs;});
    assert.throws(() => windowsHostComponentInputs(f.root), /receipt/);
  });
  test("requires exactly the fixed CUDART and AMDHIP64 runtime rows", t => {
    const f = fixture(t);
    f.edit(value => {delete value.runtimeInputs.amdhip64;});
    assert.throws(() => windowsHostComponentInputs(f.root), /receipt/);
    f.edit(value => {value.runtimeInputs.amdhip64 = {path: "runtime/extra.lib", bytes: 8, sha256: "a".repeat(64)};});
    assert.throws(() => windowsHostComponentInputs(f.root), /proof/);
    f.edit(value => {value.runtimeInputs.amdhip64 = {...value.runtimeInputs.cudartStatic};});
    assert.throws(() => windowsHostComponentInputs(f.root), /proof/);
    f.edit(value => {value.runtimeInputs.extra = {...value.runtimeInputs.cudartStatic};});
    assert.throws(() => windowsHostComponentInputs(f.root), /receipt/);
    const mismatchedFiles = fixture(t);
    mismatchedFiles.edit(value => {value.files[value.runtimeInputs.cudartStatic.path].sha256 = "a".repeat(64);});
    assert.throws(() => windowsHostComponentInputs(mismatchedFiles.root), /runtime input proof/);
  });
  test("rejects tampered, missing, or extra runtime files", t => {
    const tampered = fixture(t), tamperedInputs = windowsHostComponentInputs(tampered.root);
    fs.appendFileSync(tamperedInputs.runtimeInputs["cudartStatic"] || "", "changed");
    assert.throws(() => windowsHostComponentInputs(tampered.root), /changed/);
    const extra = fixture(t); fs.writeFileSync(path.join(extra.root, "runtime", "extra.lib"), "!<arch>\n");
    assert.throws(() => windowsHostComponentInputs(extra.root), /Unrecorded/);
    const missing = fixture(t), missingInputs = windowsHostComponentInputs(missing.root);
    fs.unlinkSync(missingInputs.runtimeInputs["amdhip64"] || "");
    assert.throws(() => windowsHostComponentInputs(missing.root), /canonical regular file/);
  });
  test("rejects a symlinked runtime archive", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), selected = windowsHostComponentInputs(f.root), archive = selected.runtimeInputs["cudartStatic"] || "",
      moved = path.join(f.root, "cudart_static.lib");
    fs.renameSync(archive, moved); fs.symlinkSync(moved, archive);
    assert.throws(() => windowsHostComponentInputs(f.root), /canonical regular file/);
  });
});

test.describe("Windows host component exporter preflight", {skip: process.platform !== "linux" && "Linux cross-build exporter"}, () => {
  test("runtime staging rejects unreviewed and traversing roles before copying", () => {
    const producer = {file: "/missing/cudart_static.lib", sha256: "a".repeat(64), bytes: 8};
    for (const relative of ["runtime/extra.lib", "runtime/../escape.lib", "../runtime/cudart_static.lib"]) {
      assert.throws(() => copyRuntimeInput(producer, relative, "/missing/output", {}), /runtime input path/);
    }
  });
  test("resolves refreshed LLVM identity through its admitted source/build cache", () => {
    const refreshed = windowsHostComponentLlvmLayout(/** @type {any} */ ({receipt: {plan: {
      output: "/private/refresh-output", source: "/private/refresh-output/source", reuse: {
        cacheRoot: "/private/original-cache", source: "/private/original-cache/source", build: "/private/original-cache/build",
      },
    }}}));
    assert.deepEqual(refreshed, {source: "/private/original-cache/source", build: "/private/original-cache/build"});
    const original = windowsHostComponentLlvmLayout(/** @type {any} */ ({receipt: {plan: {
      output: "/private/original-output", source: "/private/original-output/source",
    }}}));
    assert.deepEqual(original, {source: "/private/original-output/source", build: "/private/original-output/build"});
  });

  test("requires build-only authorization before opening producer paths", () => {
    assert.throws(() => exportWindowsHostComponent({llvmOutput: "/missing", comgrResult: "/missing/result.json",
      acppResult: "/missing/result.json", hiprtcResult: "/missing/hiprtc.json", output: "/new/component"}, {}), /MOM_AOT_BUILD_ONLY/);
  });
  test("rejects an incomplete explicit-root API", () => {
    assert.throws(() => exportWindowsHostComponent(/** @type {any} */ ({llvmOutput: "/missing"}), {MOM_AOT_BUILD_ONLY: "1"}), /explicit roots/);
    assert.throws(() => exportWindowsHostComponent(/** @type {any} */ ({llvmOutput: "/missing", comgrResult: "/missing/comgr.json",
      acppResult: "/missing/acpp.json", output: "/new/component"}), {MOM_AOT_BUILD_ONLY: "1"}), /explicit roots/);
  });
});
