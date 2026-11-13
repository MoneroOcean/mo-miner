"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {fileHash} = require("../scripts/build-source-inputs");
const {windowsGpuHandoffInputs} = require("../scripts/native-windows-gpu-handoff");

/** @param {string} file @param {unknown} value */
function json(file, value) {fs.writeFileSync(file, JSON.stringify(value) + "\n");}

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-gpu-handoff-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const output = path.join(root, "output"), isolated = path.join(output, "isolated");
  fs.mkdirSync(isolated, {recursive: true});
  const profile = "dpcpp", runAttempt = "123456789-1", configFile = path.join(root, "config.json"),
    receiptFile = path.join(output, "result.json"), isolationFile = path.join(isolated, "result.json"),
    objectFile = path.join(isolated, "object-0000.obj"), handoffFile = path.join(root, "handoff.json");
  json(configFile, {profile}); fs.writeFileSync(objectFile, Buffer.from([0x64, 0x86, 0, 0]));
  const objects = [{path: objectFile, bytes: fs.statSync(objectFile).size, sha256: fileHash(objectFile)}];
  json(isolationFile, {profile});
  // Deliberately unsorted: producer receipts preserve original receipt insertion order.
  const sdkInputs = {"/original/producer/zeta.exe": "a".repeat(64), [configFile]: fileHash(configFile), "/original/producer/alpha.h": "b".repeat(64)};
  json(receiptFile, {version: 1, profile, factory: "mom_native_gpu_dpcpp", configFile, configSha256: fileHash(configFile),
    isolation: isolationFile, objects, sdkInputs, sourceState: {}});
  const members = [configFile, receiptFile, isolationFile, objectFile].map(file => ({
    path: path.relative(root, file).split(path.sep).join("/"), bytes: fs.statSync(file).size, sha256: fileHash(file),
  })).sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
  json(handoffFile, {format: "mom-windows-gpu-handoff-v1", runAttempt, profile, files: members,
    sdkInputs: Object.fromEntries(Object.keys(sdkInputs).sort().map(file => [file, sdkInputs[file]])),
    buildOnly: true, executed: false, publicationQualified: false});
  const expected = {profile, runAttempt, receiptSha256: fileHash(receiptFile), configSha256: fileHash(configFile), handoffSha256: fileHash(handoffFile)};
  return {root, output, isolated, profile, runAttempt, configFile, receiptFile, isolationFile, objectFile, handoffFile, expected,
    read: () => windowsGpuHandoffInputs(receiptFile, expected)};
}

test.describe("Windows GPU compact handoff admission", () => {
  test("a compact same-run handoff admits producer SDK provenance without opening those historical paths", t => {
    const f = fixture(t), result = f.read();
    assert.equal(result.profile, f.profile); assert.equal(result.runAttempt, f.runAttempt);
    assert.equal(result.graph.sdkInputs["/original/producer/zeta.exe"], "a".repeat(64));
    assert.equal(result.files[f.objectFile], fileHash(f.objectFile));
    assert.equal(result.files[f.handoffFile], f.expected.handoffSha256);
    assert.equal(Object.hasOwn(result.files, "/original/producer/zeta.exe"), false);
  });

  test("receipt, config and handoff require their distinct external pins", t => {
    const f = fixture(t);
    /** @type {Array<(pin: typeof f.expected) => void>} */
    const changes = [
      pin => {pin.receiptSha256 = "0".repeat(64);},
      pin => {pin.configSha256 = "1".repeat(64);},
      pin => {pin.handoffSha256 = "2".repeat(64);},
    ];
    for (const change of changes) {
      const selected = {...f.expected}; change(selected);
      assert.throws(() => windowsGpuHandoffInputs(f.receiptFile, selected), /pin|receipt|configuration/i);
    }
  });

  test("receipt and handoff SDK maps must match, while original receipt key order is preserved", t => {
    const f = fixture(t), graph = JSON.parse(fs.readFileSync(f.receiptFile, "utf8"));
    assert.deepEqual(Object.keys(graph.sdkInputs), ["/original/producer/zeta.exe", f.configFile, "/original/producer/alpha.h"]);
    const handoff = JSON.parse(fs.readFileSync(f.handoffFile, "utf8")); handoff.sdkInputs["/original/producer/zeta.exe"] = "c".repeat(64);
    json(f.handoffFile, handoff); f.expected.handoffSha256 = fileHash(f.handoffFile);
    assert.throws(() => f.read(), /SDK map/);
  });

  test("handoff rejects an incomplete or substituted inventory even under a new external pin", t => {
    const f = fixture(t), handoff = JSON.parse(fs.readFileSync(f.handoffFile, "utf8"));
    handoff.files.pop(); json(f.handoffFile, handoff); f.expected.handoffSha256 = fileHash(f.handoffFile);
    assert.throws(() => f.read(), /inventory/);
  });

  test("payload byte changes and extra extracted files are rejected", t => {
    const changed = fixture(t); fs.appendFileSync(changed.objectFile, "changed");
    assert.throws(() => changed.read(), /payload differs/);

    const extra = fixture(t); fs.writeFileSync(path.join(extra.isolated, "unexpected.obj"), "extra");
    assert.throws(() => extra.read(), /extra, missing or substituted/);
  });

  test("profile, run-attempt and descriptor shape cannot be substituted", t => {
    const f = fixture(t);
    assert.throws(() => windowsGpuHandoffInputs(f.receiptFile, {...f.expected, profile: "acpp-cuda"}), /pin|receipt/);
    assert.throws(() => windowsGpuHandoffInputs(f.receiptFile, {...f.expected, runAttempt: "123456789-2"}), /pin/);
    assert.throws(() => windowsGpuHandoffInputs(f.receiptFile, /** @type {any} */ ({...f.expected, extra: true})), /external SHA/);
  });
});
