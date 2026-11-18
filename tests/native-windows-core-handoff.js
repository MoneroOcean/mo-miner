"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {fileHash, nativeSourceInputs} = require("../scripts/build-source-inputs");
const {ioPins} = require("../scripts/build-native-io");
const {windowsNativeCoreHandoffInputs} = require("../scripts/native-windows-core-handoff");

const repo = path.resolve(__dirname, "..");
const currentSources = nativeSourceInputs(repo, "cpu");
const runAttempt = "654321-3";

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const tempRoot = fs.realpathSync(os.tmpdir()), root = fs.mkdtempSync(path.join(tempRoot, "mom-native-win-core-handoff-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const bundle = path.join(root, "native-windows-core-654321-3"); fs.mkdirSync(bundle);
  const archive = path.join(bundle, "cpu-core.lib"); fs.writeFileSync(archive, "!<arch>\n", {flag: "wx"});
  const archiveSha256 = fileHash(archive), archiveBytes = fs.statSync(archive).size, producerPath = "C:\\agent\\_temp\\core\\cpu-core.lib",
    receiptFile = path.join(bundle, "inputs.json"), handoffFile = path.join(bundle, "handoff.json");
  const receipt = {version: 1, target: "win32", gpu: true, portable: true,
    compiler: {version: "2026.2.0", target: "x86_64-pc-windows-msvc"}, objectSha256: archiveSha256, objects: [producerPath],
    archive: {path: producerPath, bytes: archiveBytes, sha256: archiveSha256, coffTableCount: 1, machineCount: 1, x64MachineCount: 1, stubs: false},
    sourceState: currentSources,
    headerHashes: {"C:\\agent\\_temp\\libuv\\include\\uv.h": "1".repeat(64),
      "C:\\agent\\_temp\\libuv\\include\\uv\\win.h": "2".repeat(64)},
    buildOnly: true, executed: false, publicationQualified: false};
  fs.writeFileSync(receiptFile, JSON.stringify(receipt) + "\n", {flag: "wx"});
  const handoff = {format: "mom-native-windows-core-handoff-v1", runAttempt: "654321-3", target: "win32", architecture: "x64",
    archive: {file: "cpu-core.lib", bytes: archiveBytes, sha256: archiveSha256},
    producer: {script: "scripts/build-native-windows-core.js", receipt: "inputs.json", receiptSha256: fileHash(receiptFile)},
    libuv: {version: "1.51.0", archiveSha256: ioPins.libuv.sha256, fetchReceiptSha256: "3".repeat(64)},
    buildOnly: true, executed: false, publicationQualified: false};
  fs.writeFileSync(handoffFile, JSON.stringify(handoff) + "\n", {flag: "wx"});
  return {bundle, archive, receiptFile, handoffFile, handoffSha256: fileHash(handoffFile), receiptBytes: fs.readFileSync(receiptFile)};
}

/** Mutations deliberately exercise malformed receipt shapes.
 * @param {ReturnType<typeof fixture>} f @param {(receipt:any)=>void} update */
function rewriteReceipt(f, update) {
  const receipt = JSON.parse(fs.readFileSync(f.receiptFile, "utf8")); update(receipt);
  fs.writeFileSync(f.receiptFile, JSON.stringify(receipt) + "\n");
  const handoff = JSON.parse(fs.readFileSync(f.handoffFile, "utf8"));
  handoff.producer.receiptSha256 = fileHash(f.receiptFile);
  fs.writeFileSync(f.handoffFile, JSON.stringify(handoff) + "\n");
  return fileHash(f.handoffFile);
}

/** @param {ReturnType<typeof fixture>} f @param {string} [pin] */
function admit(f, pin = fileHash(f.handoffFile)) {
  return windowsNativeCoreHandoffInputs(f.bundle, {runAttempt, handoffSha256: pin});
}

/** @param {ReturnType<typeof fixture>} f @param {RegExp} pattern */
function rejects(f, pattern) {assert.throws(() => admit(f), pattern);}

test.describe("Windows native Core handoff admission", () => {
  test("admits the exact source-current portable Core handoff without rewriting its receipt", t => {
    const f = fixture(t), admitted = admit(f, f.handoffSha256);
    assert.equal(admitted.kind, "core"); assert.equal(admitted.archive, f.archive);
    assert.equal(admitted.archiveSha256, fileHash(f.archive)); assert.equal(admitted.receiptSha256, fileHash(f.receiptFile));
    assert.equal(fs.readFileSync(f.receiptFile).compare(f.receiptBytes), 0);
    assert.deepEqual(Object.keys(admitted.files).sort(), [f.archive, f.handoffFile, f.receiptFile].sort());
  });

  test("requires the exact run-attempt and externally pinned handoff hash", t => {
    const f = fixture(t);
    assert.throws(() => windowsNativeCoreHandoffInputs(f.bundle, {runAttempt: "654321-4", handoffSha256: f.handoffSha256}), /producer contract/);
    assert.throws(() => admit(f, "0".repeat(64)), /producer contract/);
    assert.throws(() => windowsNativeCoreHandoffInputs(f.bundle, {runAttempt, handoffSha256: "not-a-hash"}), /external handoff SHA/);
  });

  test("rejects an archive whose bytes differ from the producer pin", t => {
    const f = fixture(t); fs.writeFileSync(f.archive, "changed\n"); rejects(f, /archive differs/);
  });

  test("rejects a non-ar static archive even when its hash is consistently rebound", t => {
    const f = fixture(t); fs.writeFileSync(f.archive, "!<thin>\n");
    const digest = fileHash(f.archive), bytes = fs.statSync(f.archive).size;
    const pin = rewriteReceipt(f, receipt => {receipt.objectSha256 = digest; receipt.archive.sha256 = digest; receipt.archive.bytes = bytes;});
    const handoff = JSON.parse(fs.readFileSync(f.handoffFile, "utf8")); handoff.archive.sha256 = digest; handoff.archive.bytes = bytes;
    fs.writeFileSync(f.handoffFile, JSON.stringify(handoff) + "\n");
    assert.throws(() => admit(f, fileHash(f.handoffFile)), /archive differs/);
    assert.notEqual(pin, fileHash(f.handoffFile));
  });

  test("rejects a receipt changed without updating its handoff receipt hash", t => {
    const f = fixture(t); fs.appendFileSync(f.receiptFile, " "); rejects(f, /producer contract/);
  });

  test("rejects invalid or oversized producer receipt JSON", t => {
    const invalid = fixture(t); fs.writeFileSync(invalid.receiptFile, "{invalid\n"); rejects(invalid, /invalid JSON/);
    const oversized = fixture(t); fs.truncateSync(oversized.receiptFile, 8 * 1024 * 1024 + 1); rejects(oversized, /bounded canonical file/);
  });

  test("rejects a changed target", t => {
    const f = fixture(t); rewriteReceipt(f, receipt => {receipt.target = "linux";}); rejects(f, /producer receipt fails/);
  });

  test("rejects a non-Intel compiler version or target", t => {
    const version = fixture(t); rewriteReceipt(version, receipt => {receipt.compiler.version = "2025.1.0";}); rejects(version, /producer receipt fails/);
    const target = fixture(t); rewriteReceipt(target, receipt => {receipt.compiler.target = "x86_64-unknown-linux-gnu";}); rejects(target, /producer receipt fails/);
  });

  test("rejects stubs and inconsistent COFF machine counts", t => {
    const stubs = fixture(t); rewriteReceipt(stubs, receipt => {receipt.archive.stubs = true;}); rejects(stubs, /producer receipt fails/);
    const counts = fixture(t); rewriteReceipt(counts, receipt => {receipt.archive.x64MachineCount = 0;}); rejects(counts, /producer receipt fails/);
  });

  test("rejects stale current-source evidence", t => {
    const f = fixture(t), pin = rewriteReceipt(f, receipt => {
      receipt.sourceState = {...receipt.sourceState, files: {...receipt.sourceState.files, "native/not-a-real-source.cpp": "4".repeat(64)}};
    });
    assert.throws(() => admit(f, pin), /source inputs are stale/);
  });

  test("requires the complete libuv header provenance pair", t => {
    const f = fixture(t), pin = rewriteReceipt(f, receipt => {
      receipt.headerHashes = {"C:\\agent\\_temp\\libuv\\include\\uv.h": "1".repeat(64)};
    });
    assert.throws(() => admit(f, pin), /omits its exact libuv/);
  });

  test("rejects unlisted bundle members", t => {
    const f = fixture(t); fs.writeFileSync(path.join(f.bundle, "unexpected.txt"), "not part of the handoff\n");
    rejects(f, /unexpected members/);
  });

  test("rejects a symlinked archive", t => {
    const f = fixture(t), target = path.join(path.dirname(f.bundle), "archive-target.lib");
    fs.renameSync(f.archive, target);
    try {fs.symlinkSync(target, f.archive, "file");} catch (error) {
      if (process.platform === "win32") {t.skip("Windows does not permit test symlink creation in this runner"); return;}
      throw error;
    }
    rejects(f, /archive differs/);
  });

  test("rejects a hard-linked archive", t => {
    const f = fixture(t);
    try {fs.linkSync(f.archive, path.join(path.dirname(f.bundle), "archive-hardlink.lib"));} catch (error) {
      if (process.platform === "win32") {t.skip("Windows does not permit test hard-link creation in this runner"); return;}
      throw error;
    }
    rejects(f, /archive differs/);
  });

  test("rejects noncanonical and symlinked bundle roots", t => {
    const f = fixture(t), noncanonical = path.dirname(f.bundle) + path.sep + "." + path.sep + path.basename(f.bundle);
    assert.throws(() => windowsNativeCoreHandoffInputs(noncanonical, {runAttempt, handoffSha256: f.handoffSha256}), /canonical root/);
    const link = path.join(path.dirname(f.bundle), "bundle-link");
    try {fs.symlinkSync(f.bundle, link, "junction");} catch (error) {
      if (process.platform === "win32") {t.skip("Windows does not permit test directory-link creation in this runner"); return;}
      throw error;
    }
    assert.throws(() => windowsNativeCoreHandoffInputs(link, {runAttempt, handoffSha256: f.handoffSha256}), /redirected/);
  });
});
