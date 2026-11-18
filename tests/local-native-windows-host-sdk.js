"use strict";

const {describe, test} = require("node:test"), assert = require("node:assert/strict"),
  fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const local = require("../scripts/build-local-native-sdk");
const {fileHash} = require("../scripts/build-source-inputs");
const {buildLocalNativeWindowsHostSdk, windowsHostSdkPins} = require("../scripts/build-local-native-windows-host-sdk");
const repo = fs.realpathSync(path.resolve(__dirname, ".."));

/** @param {{after:(fn:()=>void)=>void}} t */
function fixture(t) {
  const base = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-local-windows-host-sdk-")),
    workspace = path.join(base, "workspace");
  local.admitWorkspace(workspace, repo);
  t.after(() => fs.rmSync(base, {recursive: true, force: true}));
  return {base, workspace, root: path.join(workspace, "windows-host-sdk"), archives: path.join(workspace, "windows-host-sdk/archives"),
    sdk: path.join(workspace, "windows-host-sdk/sdk")};
}

/** @param {string} root */
function writeSdkReceipts(root) {
  fs.mkdirSync(root);
  for (const [name, content] of Object.entries({"result.json": "original result\n", "sdk.json": "original descriptor\n",
    "files.json": "original inventory\n", "vfs-overlay.json": "original overlay\n"})) {
    fs.writeFileSync(path.join(root, name), content, {flag: "wx", mode: 0o444});
  }
}

/** Mock public acquire/prepare boundaries; fixtures never fetch archives or run extraction tools.
 * @param {ReturnType<typeof fixture>} f @param {{changeProof?:boolean,admissionError?:boolean,wrongArchiveBinding?:boolean}} [options] */
function dependencies(f, options = {}) {
  const calls = /** @type {string[]} */ ([]), receipt = {version: 1, target: "windows-host", status: "PASS", original: true};
  return {calls, value: /** @type {any} */ ({
    verifyScope() {calls.push("scope");},
    async fetchArchives(/** @type {string} */ output, /** @type {NodeJS.ProcessEnv} */ environment, /** @type {boolean} */ allowNetwork) {
      calls.push("fetch"); assert.equal(output, f.archives); assert.equal(allowNetwork, true);
      assert.equal(environment["MOM_AOT_BUILD_ONLY"], "1"); fs.mkdirSync(output);
      fs.writeFileSync(path.join(output, "fetch.json"), JSON.stringify(receipt) + "\n", {mode: 0o444});
      return receipt;
    },
    verifyArchives(/** @type {string} */ output) {
      calls.push("verify"); assert.equal(output, f.archives);
      const current = JSON.parse(fs.readFileSync(path.join(output, "fetch.json"), "utf8"));
      return options.changeProof && calls.filter(call => call === "verify").length > 1 ? {...current, altered: true} : current;
    },
    prepareSdk(/** @type {string} */ archives, /** @type {string} */ output, /** @type {NodeJS.ProcessEnv} */ environment) {
      calls.push("prepare"); assert.equal(archives, f.archives); assert.equal(output, f.sdk);
      assert.equal(environment["MOM_AOT_BUILD_ONLY"], "1"); writeSdkReceipts(output);
      return {ignored: "the unchanged files on disk are authoritative"};
    },
    readSdk(/** @type {string} */ root, /** @type {any} */ pins) {
      calls.push("admit"); assert.equal(root, f.sdk); assert.deepEqual(pins, windowsHostSdkPins(root));
      if (options.admissionError) {throw new Error("shared reader rejected fixture");}
      return {root, pins, sharedAdmission: true, result: {fetchReceipt: {path: options.wrongArchiveBinding ? "/foreign/fetch.json" : path.join(f.archives, "fetch.json"),
        sha256: fileHash(path.join(f.archives, "fetch.json"))}}};
    }})};
}

describe("Local Windows host-SDK producer adapter", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("runs the scope guard before inspecting input or admitting a workspace", async () => {
    const calls = /** @type {string[]} */ ([]);
    await assert.rejects(buildLocalNativeWindowsHostSdk(/** @type {any} */ (null), {
      verifyScope() {calls.push("scope"); throw new Error("scope missing");},
      admitWorkspace() {calls.push("workspace"); throw new Error("must not inspect workspace");},
    }), /scope missing/);
    assert.deepEqual(calls, ["scope"]);
  });

  test("fetches once, invokes the maintained SDK producer and admits exact original receipt pins", async t => {
    const f = fixture(t), d = dependencies(f), result = await buildLocalNativeWindowsHostSdk({workspace: f.workspace}, d.value);
    assert.deepEqual(d.calls, ["scope", "fetch", "verify", "prepare", "admit", "verify"]);
    assert.equal(result.workspace, f.workspace); assert.equal(result.root, f.sdk); assert.equal(result.archiveRoot, f.archives);
    assert.deepEqual(result.pins, windowsHostSdkPins(f.sdk)); assert.equal(/** @type {any} */ (result.admitted).sharedAdmission, true);
    assert.equal(/** @type {any} */ (result.source).original, true); assert.equal(result.buildOnly, true); assert.equal(result.compiled, false);
    assert.equal(result.publicationQualified, false); assert.equal(Object.hasOwn(result, "runAttempt"), false);
    assert.equal(fs.readFileSync(result.fetchReceipt, "utf8"), JSON.stringify({version: 1, target: "windows-host", status: "PASS", original: true}) + "\n");
  });

  test("warm reuse re-verifies both immutable archives and SDK contents without fetching or rebuilding", async t => {
    const f = fixture(t), d = dependencies(f);
    fs.mkdirSync(f.root); fs.mkdirSync(f.archives);
    fs.writeFileSync(path.join(f.archives, "fetch.json"), JSON.stringify({version: 1, target: "windows-host", status: "PASS", original: true}) + "\n");
    writeSdkReceipts(f.sdk);
    const result = await buildLocalNativeWindowsHostSdk({workspace: f.workspace}, d.value);
    assert.deepEqual(d.calls, ["scope", "verify", "admit", "verify"]);
    assert.equal(result.fetchReceiptSha256, fileHash(path.join(f.archives, "fetch.json")));
    assert.deepEqual(result.pins, windowsHostSdkPins(f.sdk));
  });

  test("preserves and rejects unknown or incomplete workspace state", async t => {
    const f = fixture(t), d = dependencies(f);
    fs.mkdirSync(f.root); fs.writeFileSync(path.join(f.root, "operator-data"), "preserve me");
    await assert.rejects(buildLocalNativeWindowsHostSdk({workspace: f.workspace}, d.value), /unknown entries/);
    assert.equal(fs.readFileSync(path.join(f.root, "operator-data"), "utf8"), "preserve me");
    assert.deepEqual(d.calls, ["scope"]);
  });

  test("rejects stale archive proof before preparing or handing off the SDK", async t => {
    const f = fixture(t), d = dependencies(f);
    fs.mkdirSync(f.root); fs.mkdirSync(f.archives); fs.writeFileSync(path.join(f.archives, "fetch.json"), "stale\n");
    const deps = {...d.value, verifyArchives() {d.calls.push("verify"); throw new Error("stale archive cohort");}};
    await assert.rejects(buildLocalNativeWindowsHostSdk({workspace: f.workspace}, deps), /stale archive cohort/);
    assert.deepEqual(d.calls, ["scope", "verify"]); assert.equal(fs.existsSync(f.sdk), false);
    assert.equal(fs.readFileSync(path.join(f.archives, "fetch.json"), "utf8"), "stale\n");
  });

  test("fails closed on an occupied invalid SDK and leaves its bytes untouched", async t => {
    const f = fixture(t), d = dependencies(f, {admissionError: true});
    fs.mkdirSync(f.root); fs.mkdirSync(f.archives);
    fs.writeFileSync(path.join(f.archives, "fetch.json"), JSON.stringify({version: 1, target: "windows-host", status: "PASS", original: true}) + "\n");
    writeSdkReceipts(f.sdk);
    await assert.rejects(buildLocalNativeWindowsHostSdk({workspace: f.workspace}, d.value), /shared reader rejected/);
    assert.deepEqual(d.calls, ["scope", "verify", "admit"]);
    assert.equal(fs.readFileSync(path.join(f.sdk, "result.json"), "utf8"), "original result\n");
  });

  test("requires the admitted SDK result to name the exact unchanged source receipt", async t => {
    const f = fixture(t), d = dependencies(f, {wrongArchiveBinding: true});
    fs.mkdirSync(f.root); fs.mkdirSync(f.archives);
    fs.writeFileSync(path.join(f.archives, "fetch.json"), JSON.stringify({version: 1, target: "windows-host", status: "PASS", original: true}) + "\n");
    writeSdkReceipts(f.sdk);
    await assert.rejects(buildLocalNativeWindowsHostSdk({workspace: f.workspace}, d.value), /lost original archive binding/);
    assert.deepEqual(d.calls, ["scope", "verify", "admit", "verify"]);
    assert.equal(fs.readFileSync(path.join(f.sdk, "result.json"), "utf8"), "original result\n");
  });

  test("rejects source-receipt drift across admission without returning pins", async t => {
    const f = fixture(t), d = dependencies(f, {changeProof: true});
    await assert.rejects(buildLocalNativeWindowsHostSdk({workspace: f.workspace}, d.value), /changed or lost original archive binding/);
    assert.deepEqual(d.calls, ["scope", "fetch", "verify", "prepare", "admit", "verify"]);
    assert.equal(fs.existsSync(path.join(f.sdk, "result.json")), true);
  });
});
