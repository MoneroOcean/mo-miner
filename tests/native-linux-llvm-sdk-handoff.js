"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"),
  path = require("node:path"), {createHash, randomUUID} = require("node:crypto");
const {archiveName, archiveSha256} = require("../scripts/native-llvm-sdk"),
  {exportLinuxLlvmSdk, linuxLlvmSdkHandoffInputs} = require("../scripts/native-linux-llvm-sdk-handoff");

const receiptNames = ["result.json", "configured.json", "install.json", "build-complete.json"], role = "linux-llvm21-source-sdk";
/** @param {Buffer|string} value */
const digest = (value) => createHash("sha256").update(value).digest("hex");

/** @param {import("node:test").TestContext} t @param {{unsafeSymlink?:boolean}} [options] */
function fixture(t, options = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-linux-llvm-sdk-handoff-test-")), admitted = path.join(root, "admitted"),
    sdkRoot = path.join(admitted, "sdk"), bundleParent = path.join(root, "bundle-parent"), bundle = path.join(bundleParent, "linux-llvm-sdk"),
    runId = randomUUID(), planSha256 = "a".repeat(64),
    plan = {output: admitted, prefix: sdkRoot, source: path.join(root, "source"), build: path.join(root, "build"),
      archive: path.join(root, archiveName)}, configure = {generator: "fixture"}, sourceState = {fixture: true},
    sourceInputs = {[plan.archive]: archiveSha256}, toolIdentity = {fixture: true},
    snapshotFiles = /** @type {Record<string, any>} */ ({});
  fs.mkdirSync(path.join(sdkRoot, "bin"), {recursive: true});
  fs.mkdirSync(path.join(sdkRoot, "lib", "empty-directory"), {recursive: true});
  fs.mkdirSync(bundleParent);
  const addFile = (/** @type {string} */ relative, /** @type {string} */ content, /** @type {number} */ mode = 0o444) => {
    const file = path.join(sdkRoot, relative), bytes = Buffer.from(content);
    fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, bytes, {flag: "wx", mode}); fs.chmodSync(file, mode);
    snapshotFiles[relative] = {type: "file", bytes: bytes.length, sha256: digest(bytes)};
  };
  addFile("bin/llvm-config", "fixture llvm-config\n", 0o555);
  addFile("lib/libLLVM.so.21", "fixture LLVM shared library\n");
  fs.symlinkSync("libLLVM.so.21", path.join(sdkRoot, "lib/libLLVM.so"));
  snapshotFiles["lib/libLLVM.so"] = {type: "symlink", target: "libLLVM.so.21"};
  if (options.unsafeSymlink) {
    fs.symlinkSync("../../../../outside", path.join(sdkRoot, "bin/escape"));
    snapshotFiles["bin/escape"] = {type: "symlink", target: "../../../../outside"};
  }
  const sdkBytes = Object.values(snapshotFiles).reduce((sum, row) => sum +
    (row.type === "file" ? row.bytes : Buffer.byteLength(row.target)), 0),
    sdk = {path: sdkRoot, files: Object.keys(snapshotFiles).length, bytes: sdkBytes, snapshot: {files: snapshotFiles, bytes: sdkBytes}},
    writeReceipt = (/** @type {string} */ name, /** @type {unknown} */ value) => {
      const file = path.join(admitted, name); fs.writeFileSync(file, JSON.stringify(value, null, 2) + "\n", {flag: "wx", mode: 0o444});
      return digest(fs.readFileSync(file));
    };
  fs.mkdirSync(admitted, {recursive: true});
  const complete = {version: 1, role, runId, phases: [{name: "fixture", pass: true}]}, completeSha256 = writeReceipt("build-complete.json", complete),
    audit = {files: sdk.files, bytes: sdk.bytes, archives: 1, snapshot: sdk.snapshot},
    install = {version: 1, role, runId, buildCompleteSha256: completeSha256, audit}, installSha256 = writeReceipt("install.json", install),
    configured = {version: 1, role, runId, stage: "configured", archiveSha256, planSha256, plan, configure, sourceState,
      inputFiles: sourceInputs, toolIdentity};
  writeReceipt("configured.json", configured);
  const result = {version: 1, role, runId, stage: "audited", archiveSha256, catalog: {fixture: true}, sourceState, sourceInputs,
    toolIdentity, plan, planSha256, configure, buildComplete: {path: path.join(admitted, "build-complete.json"), sha256: completeSha256},
    install: {path: path.join(admitted, "install.json"), sha256: installSha256}, archives: 1, sdk, reuse: {fixture: true},
    buildOnly: true, compiled: true, executed: false, publicationQualified: false};
  writeReceipt("result.json", result);
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, admitted, sdkRoot, bundleParent, bundle, result,
    readAdmission: () => JSON.parse(fs.readFileSync(path.join(admitted, "result.json"), "utf8"))};
}

/** @param {string} file @param {unknown} value */
function replaceJson(file, value) {
  const mode = fs.statSync(file).mode & 0o777;
  fs.chmodSync(file, 0o644); fs.writeFileSync(file, JSON.stringify(value, null, 2) + "\n"); fs.chmodSync(file, mode);
}

/** @param {string} file */
function sha256(file) {return digest(fs.readFileSync(file));}

test.describe("relocatable installed Linux LLVM SDK handoff", {skip: process.platform !== "linux"}, () => {
  test("full-admits once, preserves the original receipts, then reads after the source output is removed", t => {
    const f = fixture(t), before = Object.fromEntries(receiptNames.map(name => [name, sha256(path.join(f.admitted, name))])),
      exported = exportLinuxLlvmSdk(f.admitted, f.bundle, f.readAdmission),
      moved = path.join(f.bundleParent, "relocated-sdk");
    assert.equal(exported.fileCount, 3);
    assert.equal(exported.bytes, f.result.sdk.bytes);
    for (const name of receiptNames) {
      assert.equal(sha256(path.join(f.admitted, name)), before[name], "producer leaves original receipts untouched");
      assert.ok(fs.existsSync(path.join(f.bundle, name)), "original receipt stays at the handoff root");
    }
    assert.equal(fs.existsSync(path.join(f.bundle, "receipts")), false);
    fs.renameSync(f.bundle, moved);
    fs.rmSync(f.admitted, {recursive: true});
    const admitted = linuxLlvmSdkHandoffInputs(moved, {handoffSha256: exported.handoffSha256});
    assert.equal(admitted.sdkRoot, path.join(moved, "sdk"));
    assert.equal(admitted.files["bin/llvm-config"]?.sha256, f.result.sdk.snapshot.files["bin/llvm-config"].sha256);
    assert.equal(admitted.files["lib/libLLVM.so"]?.target, "libLLVM.so.21");
    assert.ok(fs.statSync(path.join(admitted.sdkRoot, "lib/empty-directory")).isDirectory(), "empty SDK directories survive relocation");
  });

  test("rejects a wrong handoff pin", t => {
    const f = fixture(t), exported = exportLinuxLlvmSdk(f.admitted, f.bundle, f.readAdmission),
      correct = {handoffSha256: exported.handoffSha256};
    assert.throws(() => linuxLlvmSdkHandoffInputs(f.bundle, {...correct, handoffSha256: "0".repeat(64)}), /external manifest pin|invalid or belongs/);
  });

  test("rejects an SDK symlink changed to escape the bundle", t => {
    const f = fixture(t), exported = exportLinuxLlvmSdk(f.admitted, f.bundle, f.readAdmission),
      alias = path.join(f.bundle, "sdk/lib/libLLVM.so");
    fs.unlinkSync(alias); fs.symlinkSync("../../outside", alias);
    assert.throws(() => linuxLlvmSdkHandoffInputs(f.bundle, {handoffSha256: exported.handoffSha256}),
      /inside the installed SDK|missing or extra members/);
  });

  test("rejects missing and extra installed members", t => {
    const missing = fixture(t), missExport = exportLinuxLlvmSdk(missing.admitted, missing.bundle, missing.readAdmission),
      missingFile = path.join(missing.bundle, "sdk/bin/llvm-config");
    fs.unlinkSync(missingFile);
    assert.throws(() => linuxLlvmSdkHandoffInputs(missing.bundle, {handoffSha256: missExport.handoffSha256}));

    const extra = fixture(t), extraExport = exportLinuxLlvmSdk(extra.admitted, extra.bundle, extra.readAdmission);
    fs.writeFileSync(path.join(extra.bundle, "sdk/unadmitted.txt"), "extra\n", {flag: "wx", mode: 0o444});
    assert.throws(() => linuxLlvmSdkHandoffInputs(extra.bundle, {handoffSha256: extraExport.handoffSha256}),
      /differs from its admitted receipt|missing or extra members/);
  });

  test("rejects an escaping source symlink before export", t => {
    const f = fixture(t, {unsafeSymlink: true});
    assert.throws(() => exportLinuxLlvmSdk(f.admitted, f.bundle, f.readAdmission), /resolve to an existing path inside/);
  });

  test("rejects hard-linked, writable-by-others, and special-mode SDK files", t => {
    for (const mode of [0o666, 0o4555, 0o2555, 0o1555]) {
      const f = fixture(t), file = path.join(f.sdkRoot, "bin/llvm-config");
      fs.chmodSync(file, mode);
      assert.throws(() => exportLinuxLlvmSdk(f.admitted, f.bundle, f.readAdmission),
        /bounded canonical singly-linked regular file/, "rejects mode " + mode.toString(8));
    }
    const f = fixture(t), file = path.join(f.sdkRoot, "bin/llvm-config");
    fs.linkSync(file, path.join(f.root, "second-link"));
    assert.throws(() => exportLinuxLlvmSdk(f.admitted, f.bundle, f.readAdmission),
      /bounded canonical singly-linked regular file/);
  });

  test("checks receipt cross-links even when a new handoff pin is supplied", t => {
    const f = fixture(t), exported = exportLinuxLlvmSdk(f.admitted, f.bundle, f.readAdmission),
      resultPath = path.join(f.bundle, "result.json"), handoffPath = path.join(f.bundle, "handoff.json"),
      result = JSON.parse(fs.readFileSync(resultPath, "utf8")), handoff = JSON.parse(fs.readFileSync(handoffPath, "utf8"));
    result.install.sha256 = "f".repeat(64); replaceJson(resultPath, result);
    handoff.receipts["result.json"].sha256 = sha256(resultPath); handoff.original.resultSha256 = sha256(resultPath);
    replaceJson(handoffPath, handoff);
    assert.throws(() => linuxLlvmSdkHandoffInputs(f.bundle,
      {handoffSha256: sha256(handoffPath)}), /inconsistent original receipt links/);
    assert.notEqual(exported.handoffSha256, sha256(handoffPath));
  });
});
