"use strict";

const assert = require("node:assert/strict"), crypto = require("node:crypto"), fs = require("node:fs"), os = require("node:os"),
  path = require("node:path"), test = require("node:test");
const {fileHash} = require("../scripts/build-source-inputs");
const {windowsHostSdkCatalog} = require("../scripts/fetch-native-sdk-sources");
const {windowsHostSdkInputs} = require("../scripts/native-windows-host-sdk-inputs");

const catalog = windowsHostSdkCatalog();

/** @param {string | Buffer} value */
function hash(value) {return crypto.createHash("sha256").update(value).digest("hex");}

/** @param {string} file @param {unknown} value */
function writeJson(file, value) {fs.writeFileSync(file, JSON.stringify(value, null, 2) + "\n", {flag: "wx", mode: 0o644});}

/** @param {import('node:test').TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-windows-host-sdk-inputs-")), sdkRoot = path.join(root, "sdk"),
    filesPath = path.join(root, "files.json"), sdkPath = path.join(root, "sdk.json"), overlayPath = path.join(root, "vfs-overlay.json"),
    resultPath = path.join(root, "result.json"), packageKey = Object.keys(catalog.packages).find(key => /^[a-z0-9.-]{1,128}$/.test(key));
  assert.ok(packageKey, "fixture needs a real locked package key");
  const packageRow = catalog.packages[packageKey]; assert.ok(packageRow);
  fs.mkdirSync(path.join(sdkRoot, "include", "shared"), {recursive: true});
  fs.mkdirSync(path.join(sdkRoot, "lib", "ucrt"), {recursive: true});
  const selected = [
    {relative: "include/shared/fixture.h", member: "Contents/fixture/fixture.h", bytes: Buffer.from("header fixture\n")},
    {relative: "lib/ucrt/fixture.lib", member: "Contents/fixture/fixture.lib", bytes: Buffer.from("library fixture\n")}
  ];
  const origins = selected.map(row => {
    fs.writeFileSync(path.join(sdkRoot, ...row.relative.split("/")), row.bytes, {flag: "wx", mode: 0o644});
    return {package: packageKey, packageSha256: packageRow.sha256, member: row.member, sha256: hash(row.bytes), bytes: row.bytes.length};
  });
  const files = {files: selected.map((row, index) => ({path: row.relative, sha256: hash(row.bytes), bytes: row.bytes.length,
    origins: [origins[index]]})), origins: selected.map((row, index) => ({destination: row.relative, ...origins[index]}))};
  writeJson(filesPath, files);
  fs.writeFileSync(overlayPath, "{\"version\":0,\"case-sensitive\":false,\"roots\":[]}\n", {flag: "wx", mode: 0o644});
  writeJson(sdkPath, {sourceRoot: sdkRoot, overlay: overlayPath, includeRoots: ["/mom-sdk/sdk/include/shared"],
    libraryRoots: ["/mom-sdk/sdk/lib/ucrt"]});
  const result = {version: 1, status: "PASS", target: "windows-host", buildOnly: true, compiled: false, executed: false,
    publicationQualified: false, sourceCatalogPublisherAuthenticated: false, sourceCatalogBindingVerified: false,
    sourceCatalog: {sha256: catalog.sourceCatalog.sha256, publisherAuthenticated: false, channelCatalogBindingVerified: false},
    packageCount: Object.keys(catalog.packages).length, cabinetCount: Object.keys(catalog.cabinets).length,
    selectedFileCount: selected.length, roots: catalog.roots,
    descriptor: {path: sdkPath, sha256: fileHash(sdkPath)}, overlay: {path: overlayPath, sha256: fileHash(overlayPath)},
    provenance: {path: filesPath, sha256: fileHash(filesPath)}};
  writeJson(resultPath, result);
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, sdkRoot, filesPath, sdkPath, overlayPath, resultPath,
    pins: {resultSha256: fileHash(resultPath), descriptorSha256: fileHash(sdkPath), filesSha256: fileHash(filesPath), overlaySha256: fileHash(overlayPath)}};
}

/** Re-pin only synthetic metadata after an intentional fixture mutation. @param {ReturnType<typeof fixture>} f */
function repinMetadata(f) {
  const result = JSON.parse(fs.readFileSync(f.resultPath, "utf8"));
  result.descriptor.sha256 = fileHash(f.sdkPath); result.overlay.sha256 = fileHash(f.overlayPath);
  result.provenance.sha256 = fileHash(f.filesPath);
  fs.writeFileSync(f.resultPath, JSON.stringify(result, null, 2) + "\n");
  f.pins = {resultSha256: fileHash(f.resultPath), descriptorSha256: fileHash(f.sdkPath),
    filesSha256: fileHash(f.filesPath), overlaySha256: fileHash(f.overlayPath)};
}

test.describe("Windows host-SDK input admission", () => {
  test("admits exact selected files and leaves all pinned inputs unchanged", t => {
    const f = fixture(t), before = [f.resultPath, f.sdkPath, f.filesPath, f.overlayPath,
        path.join(f.sdkRoot, "include/shared/fixture.h"), path.join(f.sdkRoot, "lib/ucrt/fixture.lib")].map(fileHash),
      admitted = windowsHostSdkInputs(f.root, f.pins);
    assert.equal(admitted.root, f.root); assert.equal(admitted.sdkRoot, f.sdkRoot);
    assert.equal(admitted.files.size, 2); assert.equal(admitted.actualTree.files.length, 2);
    assert.equal(admitted.actualTree.bytes, Buffer.byteLength("header fixture\n") + Buffer.byteLength("library fixture\n"));
    assert.deepEqual([f.resultPath, f.sdkPath, f.filesPath, f.overlayPath,
      path.join(f.sdkRoot, "include/shared/fixture.h"), path.join(f.sdkRoot, "lib/ucrt/fixture.lib")].map(fileHash), before);
  });

  test("rejects malformed or wrong pins and a symlinked root alias", t => {
    const f = fixture(t), missing = {...f.pins}; Reflect.deleteProperty(missing, "overlaySha256");
    assert.throws(() => windowsHostSdkInputs(f.root, missing), /four explicit file pins/);
    assert.throws(() => windowsHostSdkInputs(f.root, {...f.pins, resultSha256: "f".repeat(64)}), /external pin/);
    const extra = {...f.pins, extra: "0".repeat(64)};
    assert.throws(() => windowsHostSdkInputs(f.root, extra), /four explicit file pins/);
    const alias = f.root + "-alias";
    try {fs.symlinkSync(f.root, alias, "dir");} catch (error) {
      if (process.platform === "win32" && ["EPERM", "EACCES", "ENOTSUP"].includes(Reflect.get(Object(error), "code"))) {
        t.skip("Windows host cannot create directory symlinks in this test environment"); return;
      }
      throw error;
    }
    t.after(() => fs.rmSync(alias, {force: true}));
    assert.throws(() => windowsHostSdkInputs(alias, f.pins), /canonical/);
  });

  test("rejects selected-file hash and byte drift", t => {
    const f = fixture(t), member = path.join(f.sdkRoot, "include/shared/fixture.h");
    fs.writeFileSync(member, "changed header\n");
    assert.throws(() => windowsHostSdkInputs(f.root, f.pins), /tree differs|pinned inventory/i);
  });

  test("rejects extra and missing selected-tree files", t => {
    const extra = fixture(t); fs.writeFileSync(path.join(extra.sdkRoot, "extra.h"), "extra\n");
    assert.throws(() => windowsHostSdkInputs(extra.root, extra.pins), /tree differs/i);
    const missing = fixture(t); fs.unlinkSync(path.join(missing.sdkRoot, "lib/ucrt/fixture.lib"));
    assert.throws(() => windowsHostSdkInputs(missing.root, missing.pins), /tree differs/i);
  });

  test("rejects case-aliased inventory paths on either filesystem kind", t => {
    const f = fixture(t), manifest = JSON.parse(fs.readFileSync(f.filesPath, "utf8")),
      alias = {...manifest.files[0], path: "INCLUDE/shared/fixture.h"};
    manifest.files.push(alias);
    manifest.origins.push({destination: alias.path, ...alias.origins[0]});
    fs.writeFileSync(f.filesPath, JSON.stringify(manifest, null, 2) + "\n");
    const result = JSON.parse(fs.readFileSync(f.resultPath, "utf8")); result.selectedFileCount = manifest.files.length;
    fs.writeFileSync(f.resultPath, JSON.stringify(result, null, 2) + "\n"); repinMetadata(f);
    assert.throws(() => windowsHostSdkInputs(f.root, f.pins), /case alias/i);
  });

  test("walkTree rejects hard-linked selected files", t => {
    const f = fixture(t), source = path.join(f.sdkRoot, "include/shared/fixture.h");
    fs.linkSync(source, path.join(f.sdkRoot, "linked.h"));
    assert.throws(() => windowsHostSdkInputs(f.root, f.pins), /hard-linked path/i);
  });

  test("walkTree rejects symbolic links in the selected SDK", t => {
    const f = fixture(t);
    try {fs.symlinkSync("fixture.h", path.join(f.sdkRoot, "include/shared/link.h"));} catch (error) {
      if (process.platform === "win32" && ["EPERM", "EACCES", "ENOTSUP"].includes(Reflect.get(Object(error), "code"))) {
        t.skip("Windows host cannot create file symlinks in this test environment"); return;
      }
      throw error;
    }
    assert.throws(() => windowsHostSdkInputs(f.root, f.pins), /link\/special\/hard-linked/i);
  });

  test("rejects origin-ledger drift even when all outer metadata pins are refreshed", t => {
    const f = fixture(t), manifest = JSON.parse(fs.readFileSync(f.filesPath, "utf8"));
    manifest.origins.pop(); fs.writeFileSync(f.filesPath, JSON.stringify(manifest, null, 2) + "\n"); repinMetadata(f);
    assert.throws(() => windowsHostSdkInputs(f.root, f.pins), /missing a source origin/i);
  });

  test("rejects empty include or library roots even when descriptor pins are refreshed", t => {
    for (const key of ["includeRoots", "libraryRoots"]) {
      const f = fixture(t), sdk = JSON.parse(fs.readFileSync(f.sdkPath, "utf8"));
      sdk[key] = []; fs.writeFileSync(f.sdkPath, JSON.stringify(sdk, null, 2) + "\n"); repinMetadata(f);
      assert.throws(() => windowsHostSdkInputs(f.root, f.pins), /outside its build-only contract/);
    }
  });

  test("requires the locked source catalog even after the result is re-pinned", t => {
    const f = fixture(t), result = JSON.parse(fs.readFileSync(f.resultPath, "utf8"));
    result.sourceCatalog.sha256 = "e".repeat(64);
    fs.writeFileSync(f.resultPath, JSON.stringify(result, null, 2) + "\n"); repinMetadata(f);
    assert.throws(() => windowsHostSdkInputs(f.root, f.pins), /outside its build-only contract/);
  });

  test("binds each coherent origin ledger to the locked package identity and digest", t => {
    for (const unknownPackage of [false, true]) {
      const f = fixture(t), manifest = JSON.parse(fs.readFileSync(f.filesPath, "utf8"));
      for (const origin of [manifest.files[0].origins[0], manifest.origins[0]]) {
        if (unknownPackage) {origin.package = "unlisted-package";} else {origin.packageSha256 = "e".repeat(64);}
      }
      fs.writeFileSync(f.filesPath, JSON.stringify(manifest, null, 2) + "\n"); repinMetadata(f);
      assert.throws(() => windowsHostSdkInputs(f.root, f.pins), /source origin is malformed/);
    }
  });
});
