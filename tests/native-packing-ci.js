"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), assert = require("node:assert/strict"),
  test = require("node:test"), {createHash} = require("node:crypto"), {spawnSync} = require("node:child_process");
const {fileHash} = require("../scripts/build-source-inputs"), realCi = require("../scripts/prepare-native-ci");
const {prepareNativePackingTool} = require("../scripts/prepare-native-packing-tool");
const {packingPlan, packNative} = require("../scripts/ci/pack-native");
const toolCatalog = require("../scripts/native-packing-tool.json");

const suffix = "123456789-1", digest = (/** @type {string} */ value) => createHash("sha256").update(value).digest("hex");
// Tool bytes are deliberately inert fixture data, never executable host programs.
const toolArchive = "fixture upx archive", toolBytes = "fixture upx";

/** Swap the shared packing-tool catalog pins for inert fixture bytes until this test ends.
 * @param {import('node:test').TestContext} t */
function pinFixtureTool(t) {
  const saved = JSON.stringify({archive: toolCatalog.archive, executable: toolCatalog.executable});
  Object.assign(toolCatalog.archive, {bytes: Buffer.byteLength(toolArchive), sha256: digest(toolArchive)});
  Object.assign(toolCatalog.executable, {bytes: Buffer.byteLength(toolBytes), sha256: digest(toolBytes)});
  t.after(() => {
    const original = JSON.parse(saved);
    Object.assign(toolCatalog.archive, original.archive); Object.assign(toolCatalog.executable, original.executable);
  });
}

/** Build one same-run packing job input set: a real metadata tar in RUNNER_TEMP/incoming, the raw
 * download in RUNNER_TEMP/native-raw, and a private state root standing in for /tmp.
 * @param {import('node:test').TestContext} t @param {'linux'|'win32'} [target] */
function fixture(t, target = "linux") {
  pinFixtureTool(t);
  const temp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-packing-ci-")),
    platform = target === "win32" ? "windows" : "linux", binary = target === "win32" ? "mom.exe" : "mom",
    metadataName = `native-${platform}-application-build-${suffix}`, staging = path.join(temp, "staging"),
    tree = path.join(staging, metadataName), stateRoot = path.join(temp, "state"), incoming = path.join(temp, "incoming"),
    archive = path.join(incoming, metadataName + ".tar"), output = path.join(stateRoot, `native-${platform}-packed-build-${suffix}`),
    calls = /** @type {Array<{stage: string, [key: string]: any}>} */ ([]);
  t.after(() => fs.rmSync(temp, {recursive: true, force: true}));
  for (const directory of [path.join(temp, "native-raw"), tree, stateRoot, incoming]) {fs.mkdirSync(directory, {recursive: true});}
  const raw = path.join(temp, "native-raw", binary);
  fs.writeFileSync(raw, "own fixture executable");
  const save = (/** @type {string} */ name, /** @type {unknown} */ value) => fs.writeFileSync(path.join(tree, name), JSON.stringify(value));
  const hash = (/** @type {string} */ name) => fileHash(path.join(tree, name));
  save("application-config.json", {buildOnly: true}); save("result.json", {buildOnly: true});
  save("entrypoint.json", {configHash: hash("application-config.json")});
  const handoff = {version: 1, format: `mom-${platform}-application-build-handoff-v1`, runAttempt: suffix,
    target: target === "win32" ? "x86_64-pc-windows-msvc" : "x86_64-unknown-linux-gnu",
    artifact: {file: binary, bytes: fs.statSync(raw).size, sha256: fileHash(raw)},
    config: {file: "application-config.json", sha256: hash("application-config.json")},
    producerReceipt: {file: "result.json", sha256: hash("result.json")},
    entrypointReceipt: {file: "entrypoint.json", sha256: hash("entrypoint.json")},
    buildOnly: true, executed: false, publicationQualified: false};
  /** @type {NodeJS.ProcessEnv} */
  const env = {RUNNER_TEMP: temp, GITHUB_RUN_ID: "123456789", GITHUB_RUN_ATTEMPT: "1", NATIVE_PLATFORM: target,
    NATIVE_RAW_SHA256: fileHash(raw), GITHUB_OUTPUT: path.join(temp, "outputs")};
  // Seal the staged metadata into a real same-run tar and pin it like the producer job outputs.
  const seal = () => {
    save("handoff.json", handoff); fs.rmSync(archive, {force: true});
    realCi.packState(archive, staging, [metadataName], 60000);
    env["NATIVE_METADATA_SHA256"] = fileHash(archive); env["NATIVE_HANDOFF_SHA256"] = hash("handoff.json");
  };
  seal();
  const spawn = /** @type {any} */ ((/** @type {string} */ command, /** @type {string[]} */ args, /** @type {any} */ options) => {
    calls.push({stage: command, args});
    if (command === "curl") {fs.writeFileSync(args[args.indexOf("--output") + 1] || "", toolArchive);} else if (command === "tar") {
      fs.writeSync(options.stdio[1], toolBytes);
    } else {throw new Error("Unexpected external command: " + command);}
    return {status: 0};
  });
  const preparePacking = () => prepareNativePackingTool(path.join(temp, "native-packing-tool"), env, spawn);
  const io = {stateRoot,
    ci: {listState: realCi.listState,
      extractState: (/** @type {Parameters<typeof realCi.extractState>} */ ...args) => {calls.push({stage: "restore"}); realCi.extractState(...args);},
      packState: (/** @type {Parameters<typeof realCi.packState>} */ ...args) => {calls.push({stage: "seal"}); return realCi.packState(...args);}},
    prepareNativePackingTool: (/** @type {string} */ directory, /** @type {NodeJS.ProcessEnv} */ toolEnv) =>
      prepareNativePackingTool(directory, toolEnv, spawn),
    packNativeEntrypoint: /** @type {any} */ ((/** @type {any} */ options) => {
      calls.push({stage: "pack", options}); assert.equal(options.entrypointSha256, handoff.entrypointReceipt.sha256);
      assert.equal(fileHash(options.producerReceiptPath), handoff.producerReceipt.sha256);
      fs.mkdirSync(output); fs.writeFileSync(path.join(output, binary), "packed fixture");
      return {artifact: {sha256: fileHash(path.join(output, binary))}};
    })};
  return {env, calls, handoff, save, seal, raw, temp, tree, output, preparePacking,
    pack: () => {packingPlan(env); packNative(env, io);}};
}

test.describe("SDK-free native packing CI handoff", {skip: (process.platform !== "linux" || process.arch !== "x64") && "Linux x64 packing host"}, () => {
  test("official packer and its source notice cohort remain independently pinned", () => {
    const pin = toolCatalog;
    assert.equal(pin.version, "5.2.1"); assert.equal(pin.archive.bytes, 664048); assert.equal(pin.source.bytes, 1778024);
    for (const archive of [pin.archive, pin.source]) {
      assert.match(archive.url, /^https:\/\/github\.com\/upx\/upx\/releases\/download\/v5\.2\.1\/upx-5\.2\.1-/);
      assert.match(archive.sha256, /^[a-f0-9]{64}$/);
    }
    assert.deepEqual(pin.source.notices.map(row => row.member), ["upx-5.2.1-src/LICENSE", "upx-5.2.1-src/COPYING", "upx-5.2.1-src/vendor/lzma-sdk/LICENSE.txt"]);
    for (const [index, row] of pin.source.notices.entries()) {
      assert.match(row.sha256, /^[a-f0-9]{64}$/);
      if (index < 2) {assert.equal(row.sha256, pin.notices[index]?.sha256, "source and official binary distribute identical license texts");}
    }
  });
  for (const target of ["linux", "win32"]) {
    test(`${target} uses the existing packing API with the original independent pins`, t => {
      const f = fixture(t, /** @type {'linux'|'win32'} */ (target)); f.pack();
      const packed = f.calls.find(row => row.stage === "pack");
      assert.equal(packed?.["options"].target, target); assert.equal(packed?.["options"].sourceSha256, f.env["NATIVE_RAW_SHA256"]);
      assert.equal(packed?.["options"].output, f.output);
      assert.equal(packed?.["options"].sourcePath, path.join(f.temp, "native-raw", target === "win32" ? "mom.exe" : "mom"));
      assert.deepEqual(f.calls.map(row => row.stage), ["restore", "curl", "tar", "pack", "seal"]);
      const platform = target === "win32" ? "windows" : "linux", bundle = path.join(f.temp, `native-${platform}-packed-build-${suffix}.tar`);
      assert.equal(fs.readFileSync(f.env["GITHUB_OUTPUT"] || "", "utf8"),
        `executable_artifact_name=native-${platform}-packed-executable-${suffix}\n` +
        `executable_sha256=${fileHash(path.join(f.output, target === "win32" ? "mom.exe" : "mom"))}\nbuild_bundle_sha256=${fileHash(bundle)}\n`);
    });
  }
  test("the shared packer reuses admitted bytes without another download or extraction", t => {
    const f = fixture(t); f.pack();
    const count = f.calls.length, first = f.preparePacking();
    assert.equal(f.calls.length, count);
    assert.equal(first.upx, path.join(f.temp, "native-packing-tool/upx"));
  });
  test("changed cached packer bytes are rejected without replacement", t => {
    const f = fixture(t); f.pack();
    const upx = path.join(f.temp, "native-packing-tool/upx"), count = f.calls.length;
    fs.appendFileSync(upx, "changed");
    assert.throws(() => f.preparePacking(), /Pinned packing tool bytes differ/);
    assert.equal(f.calls.length, count); assert.match(fs.readFileSync(upx, "utf8"), /changed$/);
  });
  test("partial packer caches are preserved and cannot be promoted by a retry", t => {
    const f = fixture(t); f.pack();
    fs.unlinkSync(path.join(f.temp, "native-packing-tool/upx"));
    const count = f.calls.length;
    assert.throws(() => f.preparePacking(), /unknown or incomplete/);
    assert.equal(f.calls.length, count);
    assert.equal(fs.existsSync(path.join(f.temp, "native-packing-tool/upx.tar.xz")), true);
  });
  for (const [name, key, value] of [["the windows alias", "NATIVE_PLATFORM", "windows"], ["an invalid platform", "NATIVE_PLATFORM", "darwin"],
    ["attempt zero", "GITHUB_RUN_ATTEMPT", "0"], ["a non-numeric run", "GITHUB_RUN_ID", "run"], ["a missing SHA", "NATIVE_RAW_SHA256", ""],
    ["an uppercase SHA", "NATIVE_HANDOFF_SHA256", "A".repeat(64)]]) {
    test(`rejects ${name} before artifact access`, t => {
      const f = fixture(t); f.env[key || ""] = value;
      assert.throws(() => packingPlan(f.env), /exact raw executable/); assert.throws(() => f.pack(), /exact raw executable/);
      assert.equal(f.calls.length, 0);
    });
  }
  test("another run attempt cannot admit this attempt's metadata bundle", t => {
    const f = fixture(t); f.env["GITHUB_RUN_ATTEMPT"] = "2";
    assert.throws(() => f.pack(), /ENOENT/); assert.equal(f.calls.length, 0);
  });
  test("tampered raw executable fails before restore or tool acquisition", t => {
    const f = fixture(t); fs.appendFileSync(f.raw, "changed"); assert.throws(() => f.pack(), /regular-file pin/); assert.equal(f.calls.length, 0);
  });
  test("an extra file beside the raw executable fails before restore", t => {
    const f = fixture(t); fs.writeFileSync(path.join(f.temp, "native-raw", "extra"), "x");
    assert.throws(() => f.pack(), /unexpected files/); assert.equal(f.calls.length, 0);
  });
  test("extra metadata member fails before extraction", t => {
    const f = fixture(t); fs.writeFileSync(path.join(f.tree, "unexpected"), "x"); f.seal();
    assert.throws(() => f.pack(), /unexpected members/); assert.equal(f.calls.length, 0);
  });
  test("a handoff for another target cannot select the packing platform", t => {
    const f = fixture(t); f.handoff.target = "x86_64-pc-windows-msvc"; f.seal(); assert.throws(() => f.pack(), /does not bind/);
    assert.deepEqual(f.calls.map(row => row.stage), ["restore"]);
  });
  test("changed producer bytes fail before tool acquisition", t => {
    const f = fixture(t); f.save("result.json", {changed: true}); f.seal(); assert.throws(() => f.pack(), /regular-file pin/);
    assert.deepEqual(f.calls.map(row => row.stage), ["restore"]);
  });
  test("the CLI reports failures with the native-ci prefix", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/ci/pack-native.js"), "validate"],
      {encoding: "utf8", env: {PATH: process.env["PATH"] || ""}});
    assert.equal(result.status, 1); assert.match(result.stderr, /^\[native-ci\] Native packing needs/);
  });
});
