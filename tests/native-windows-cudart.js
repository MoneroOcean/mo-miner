"use strict";

const fs = require("node:fs"), path = require("node:path");
const {test, describe} = require("node:test"), assert = require("node:assert/strict");
const {createHash} = require("node:crypto");
const {fileHash} = require("../scripts/build-source-inputs");
const {windowsCudartCatalog, stagePinnedZipMembers, prepareWindowsCudart, admitWindowsCudart, cudartRecipe, trustedUnzipSnapshot} = require("../scripts/prepare-native-cudart-windows");
const {tempDir} = require("./common/temp-dir");
const {storedZip} = require("./common/zip-fixture");

/** @typedef {{member:string,output:string,bytes:number,sha256:string}} CudartMember */

/** @param {import("node:test").TestContext} t */
function fixtureRoot(t) {return fs.realpathSync(tempDir(t, "mom-win-cudart-"));}

/** @param {Buffer} bytes */
function sha256(bytes) {return createHash("sha256").update(bytes).digest("hex");}

/** @returns {string} */
function findUnzip() {return "/usr/bin/unzip";}

describe("Pinned Windows CUDA Runtime inputs", {skip: process.platform !== "linux"}, () => {
  test("catalog pins exactly the CUDA 12.6 runtime archive, library, and license", () => {
    const catalog = windowsCudartCatalog();
    assert.equal(catalog.archive.url, "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.6.37-archive.zip");
    assert.equal(catalog.archive.sha256, "c66badb16921c4acb7abc38f211d530c2f290bd52c2cbdb5fb12095dc6eab0a9");
    assert.equal(catalog.cudartStatic.bytes, 2753442);
    assert.equal(catalog.cudartStatic.sha256, "7cbf0142b20001882ba94a6766d365f6d734f807ab0c8a07a0102a7020054e4e");
    assert.equal(catalog.license.bytes, 63021);
    assert.equal(catalog.license.sha256, "e2c71babfd18a8e69542dd7e9ca018f9caa438094001a58e6bc4d8c999bf0d07");
    assert.ok(Object.isFrozen(catalog) && Object.isFrozen(catalog.archive) && Object.isFrozen(catalog.license));
  });

  test("stages the two pinned plaintext members and never materializes unselected paths", t => {
    const root = fixtureRoot(t), catalog = windowsCudartCatalog(), library = Buffer.from("runtime fixture"), license = Buffer.from("license fixture"),
      rows = /** @type {[CudartMember,CudartMember]} */ ([
        {member: catalog.cudartStatic.member, output: catalog.cudartStatic.output, bytes: library.length, sha256: sha256(library)},
        {member: catalog.license.member, output: catalog.license.output, bytes: license.length, sha256: sha256(license)},
      ]), archive = path.join(root, "fixture.zip");
    fs.writeFileSync(archive, storedZip([[rows[0].member, library], ["../../escape", "ignored"], [rows[1].member, license]]));
    const staged = stagePinnedZipMembers(findUnzip(), archive, rows), libraryStage = staged[rows[0].output], licenseStage = staged[rows[1].output];
    assert.deepEqual(Object.keys(staged).sort(), rows.map(row => row.output).sort());
    assert.ok(libraryStage && licenseStage);
    assert.equal(Reflect.get(libraryStage, "bytes"), library.length);
    assert.equal(sha256(/** @type {Buffer} */ (Reflect.get(libraryStage, "data"))), sha256(library));
    assert.equal(sha256(/** @type {Buffer} */ (Reflect.get(licenseStage, "data"))), sha256(license));
    assert.equal(fs.existsSync(path.join(root, "escape")), false);
  });

  test("rejects missing or duplicate selected names, wrong member bytes, and traversal selectors", t => {
    const root = fixtureRoot(t), catalog = windowsCudartCatalog(), library = Buffer.from("runtime"), license = Buffer.from("license"),
      rows = /** @type {[CudartMember,CudartMember]} */ ([
        {member: catalog.cudartStatic.member, output: catalog.cudartStatic.output, bytes: library.length, sha256: sha256(library)},
        {member: catalog.license.member, output: catalog.license.output, bytes: license.length, sha256: sha256(license)},
      ]), writeZip = (/** @type {string} */ name, /** @type {Array<[string, Buffer]>} */ entries) => {
        const archive = path.join(root, name); fs.writeFileSync(archive, storedZip(entries)); return archive;
      }, libraryRow = /** @type {[string, Buffer]} */ ([rows[0].member, library]), licenseRow = /** @type {[string, Buffer]} */ ([rows[1].member, license]);
    assert.throws(() => stagePinnedZipMembers(findUnzip(), writeZip("missing.zip", [libraryRow]), rows), /missing or duplicated/);
    assert.throws(() => stagePinnedZipMembers(findUnzip(), writeZip("duplicate.zip", [libraryRow, libraryRow, licenseRow]), rows), /missing or duplicated/);
    const complete = writeZip("wrong-bytes.zip", [libraryRow, licenseRow]);
    assert.throws(() => stagePinnedZipMembers(findUnzip(), complete, [{...rows[0], sha256: "0".repeat(64)}, rows[1]]), /bounded pin/);
    assert.throws(() => stagePinnedZipMembers(findUnzip(), complete, [rows[0], {...rows[1], output: "../escape"}]), /unique pinned members/);
  });

  test("rejects an unpinned local ZIP before creating the output", t => {
    const root = fixtureRoot(t), archive = path.join(root, "wrong.zip"), output = path.join(root, "not-created");
    fs.writeFileSync(archive, "not the pinned archive");
    const previous = process.env["MOM_AOT_BUILD_ONLY"]; process.env["MOM_AOT_BUILD_ONLY"] = "1";
    try {assert.throws(() => prepareWindowsCudart(archive, output), /archive differs from its official pin/);} finally {
      if (previous === undefined) {delete process.env["MOM_AOT_BUILD_ONLY"];} else {process.env["MOM_AOT_BUILD_ONLY"] = previous;}
    }
    assert.equal(fs.existsSync(output), false);
  });

  test("accepts the packaged two-link unzip binary", {skip: fs.lstatSync("/usr/bin/unzip", {bigint: true}).nlink !== 2n}, () => {
    assert.equal(trustedUnzipSnapshot().nlink, 2n);
  });

  test("prepares and admits the official ZIP when explicitly supplied", {skip: !process.env["MOM_WINDOWS_CUDART_TEST_ARCHIVE"]}, t => {
    const archive = process.env["MOM_WINDOWS_CUDART_TEST_ARCHIVE"], root = fixtureRoot(t), output = path.join(root, "prepared"),
      previous = process.env["MOM_AOT_BUILD_ONLY"];
    assert.ok(archive && path.isAbsolute(archive));
    process.env["MOM_AOT_BUILD_ONLY"] = "1";
    try {
      const prepared = prepareWindowsCudart(archive, output), admitted = admitWindowsCudart(output);
      assert.equal(prepared.output, output);
      assert.deepEqual(admitted.files, prepared.files);
      assert.equal(admitted.source.sha256, windowsCudartCatalog().archive.sha256);
      assert.equal(fs.realpathSync(output), output);
    } finally {
      if (previous === undefined) {delete process.env["MOM_AOT_BUILD_ONLY"];} else {process.env["MOM_AOT_BUILD_ONLY"] = previous;}
    }
  });

  test("consumer ignores missing producer archive and never executes receipt-supplied tool paths", t => {
    const root = fixtureRoot(t), output = path.join(root, "restored"), catalog = windowsCudartCatalog(), recipe = cudartRecipe(), marker = path.join(root, "ran-tool"),
      maliciousTool = path.join(root, "receipt-tool");
    fs.writeFileSync(maliciousTool, "#!/bin/sh\nprintf executed > " + JSON.stringify(marker) + "\n", {mode: 0o700});
    const files = {"runtime/cudart_static.lib": Buffer.from("wrong library"), "licenses/cuda-cudart/LICENSE": Buffer.from("wrong license")};
    fs.mkdirSync(path.join(output, "runtime"), {recursive: true}); fs.mkdirSync(path.join(output, "licenses/cuda-cudart"), {recursive: true});
    fs.writeFileSync(path.join(output, "runtime/cudart_static.lib"), files["runtime/cudart_static.lib"]);
    fs.writeFileSync(path.join(output, "licenses/cuda-cudart/LICENSE"), files["licenses/cuda-cudart/LICENSE"]);
    const receiptFiles = {
      [catalog.cudartStatic.output]: {kind: "file", bytes: catalog.cudartStatic.bytes, sha256: catalog.cudartStatic.sha256,
        source: {archiveSha256: catalog.archive.sha256, member: catalog.cudartStatic.member}},
      [catalog.license.output]: {kind: "file", bytes: catalog.license.bytes, sha256: catalog.license.sha256,
        source: {archiveSha256: catalog.archive.sha256, member: catalog.license.member}},
    };
    const receipt = {version: 1, target: "windows-cudart", status: "PASS", networked: false, buildOnly: true, compiled: false,
      vendorPrebuilt: true, publicationQualified: false, redistributionQualified: false,
      producerSha256: fileHash(path.join(__dirname, "../scripts/prepare-native-cudart-windows.js")),
      recipe: {sha256: sha256(Buffer.from(JSON.stringify(recipe))), files: recipe.graph},
      tool: {path: maliciousTool, sha256: fileHash(maliciousTool), version: "receipt fixture"},
      source: {archivePath: path.join(root, "missing-original.zip"), url: catalog.archive.url, sha256: catalog.archive.sha256,
        bytes: catalog.archive.bytes, manifestUrl: catalog.manifestUrl}, output: "/producer/host/output", files: receiptFiles};
    fs.writeFileSync(path.join(output, "sdk.json"), JSON.stringify(receipt), {mode: 0o444});
    fs.chmodSync(path.join(output, "runtime"), 0o555); fs.chmodSync(path.join(output, "licenses/cuda-cudart"), 0o555);
    fs.chmodSync(path.join(output, "licenses"), 0o555); fs.chmodSync(output, 0o555);
    assert.throws(() => admitWindowsCudart(output), /differs from its pinned member/);
    assert.equal(fs.existsSync(marker), false);
  });
});
