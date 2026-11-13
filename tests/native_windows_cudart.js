"use strict";

const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {test, describe} = require("node:test"), assert = require("node:assert/strict");
const {createHash} = require("node:crypto");
const {fileHash} = require("../scripts/build-source-inputs");
const {windowsCudartCatalog, stagePinnedZipMembers, prepareWindowsCudart, admitWindowsCudart, cudartRecipe, trustedUnzipSnapshot} = require("../scripts/prepare-native-cudart-windows");

/** @typedef {{name:string,data:Buffer}} ZipEntry */
/** @typedef {{member:string,output:string,bytes:number,sha256:string}} CudartMember */

/** @param {Uint8Array} value */
function crc32(value) {
  let crc = 0xffffffff;
  for (const byte of value) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) {crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);}
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/** Build a tiny stored ZIP for fixed-member staging tests. @param {ZipEntry[]} entries */
function zipBytes(entries) {
  const local = [], central = []; let offset = 0;
  for (const entry of entries) {
    const name = Buffer.from(entry.name), data = entry.data, crc = crc32(data), header = Buffer.alloc(30), row = Buffer.alloc(46);
    header.writeUInt32LE(0x04034b50, 0); header.writeUInt16LE(20, 4); header.writeUInt32LE(crc, 14);
    header.writeUInt32LE(data.length, 18); header.writeUInt32LE(data.length, 22); header.writeUInt16LE(name.length, 26); local.push(header, name, data);
    row.writeUInt32LE(0x02014b50, 0); row.writeUInt16LE(0x0314, 4); row.writeUInt16LE(20, 6); row.writeUInt32LE(crc, 16);
    row.writeUInt32LE(data.length, 20); row.writeUInt32LE(data.length, 24); row.writeUInt16LE(name.length, 28);
    row.writeUInt32LE((0o100644 << 16) >>> 0, 38); row.writeUInt32LE(offset, 42); central.push(row, name);
    offset += header.length + name.length + data.length;
  }
  const directory = Buffer.concat(central), end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directory.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...local, directory, end]);
}

/** @param {string} directory */
function makeWritable(directory) {
  const stat = fs.lstatSync(directory);
  if (stat.isSymbolicLink()) {return;}
  if (stat.isDirectory()) {
    fs.chmodSync(directory, 0o700);
    for (const name of fs.readdirSync(directory)) {makeWritable(path.join(directory, name));}
  } else if (stat.isFile()) {fs.chmodSync(directory, 0o600);}
}

/** @param {import("node:test").TestContext} t */
function fixtureRoot(t) {
  const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-win-cudart-"));
  t.after(() => {makeWritable(root); fs.rmSync(root, {recursive: true, force: true});});
  return root;
}

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
    fs.writeFileSync(archive, zipBytes([{name: rows[0].member, data: library}, {name: "../../escape", data: Buffer.from("ignored")}, {name: rows[1].member, data: license}]));
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
      ]), writeZip = (/** @type {string} */ name, /** @type {ZipEntry[]} */ entries) => {
        const archive = path.join(root, name); fs.writeFileSync(archive, zipBytes(entries)); return archive;
      }, libraryRow = {name: rows[0].member, data: library}, licenseRow = {name: rows[1].member, data: license};
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
