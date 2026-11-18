"use strict";

const fs = require("node:fs"), os = require("node:os"), path = require("node:path"),
  {createHash} = require("node:crypto"), test = require("node:test"), assert = require("node:assert/strict");
const {dependencyPins, checkNativeNoticeReview} = require("../scripts/check-native-notices");
const {nativePathSource} = require("../scripts/native-bundle");
const {execFileSync} = require("node:child_process");

function trackedLicenseSections() {
  const notice = fs.readFileSync(path.join(__dirname, "..", "THIRD-PARTY-NOTICES.txt"), "utf8");
  const pins = dependencyPins(), components = [...pins.archives.llvm.licenseMembers.map(({component}) => component),
    pins.runtime.linuxCudart.license.noticeComponent];
  return components.map(component => {
    const marker = "================================================================================\nComponents: " + component + "\n\n", start = notice.indexOf(marker);
    assert.notEqual(start, -1, "tracked notice must include " + component);
    const bodyStart = start + marker.length, end = notice.indexOf("\n================================================================================\nComponents: ", bodyStart);
    return notice.slice(start, end < 0 ? notice.length : end);
  }).join("\n");
}

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-notice-review-")), runtimeDirectory = path.join(root, "runtime"),
    evidenceDirectory = path.join(runtimeDirectory, "license-evidence");
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  fs.mkdirSync(evidenceDirectory, {recursive: true});
  const manifestPath = path.join(runtimeDirectory, "native-notice-review.json"), noticePath = path.join(root, "THIRD-PARTY-NOTICES.txt"),
    body = Buffer.from("Reviewed notice fixture\n" + trackedLicenseSections(), "utf8");
  const sourceManifest = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "runtime/native-notice-review.json"), "utf8"));
  for (const name of ["sdk_license.rtf", "sdk_third_party_notices.rtf"]) {
    fs.copyFileSync(path.join(__dirname, "..", "runtime/license-evidence", name), path.join(evidenceDirectory, name));
  }
  fs.writeFileSync(noticePath, body);
  const manifest = {version: 1, notice: {bytes: body.length, sha256: createHash("sha256").update(body).digest("hex")},
    redistributionEvidence: JSON.parse(JSON.stringify(sourceManifest.redistributionEvidence)),
    dependencies: JSON.parse(JSON.stringify(dependencyPins()))};
  fs.writeFileSync(manifestPath, JSON.stringify(manifest));
  return {root, manifestPath, noticePath, body, manifest};
}

/** @param {ReturnType<typeof fixture>} f */
function save(f) {fs.writeFileSync(f.manifestPath, JSON.stringify(f.manifest));}

test.describe("native notice review", () => {
  test("Git preserves pinned notice and license bytes across clean checkouts", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-notice-checkout-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const repo = path.join(__dirname, ".."), files = ["THIRD-PARTY-NOTICES.txt",
      "runtime/license-evidence/sdk_license.rtf", "runtime/license-evidence/sdk_third_party_notices.rtf"];
    const git = (/** @type {string[]} */ args) => execFileSync("git", ["-c", "core.autocrlf=true", ...args], {cwd: root, stdio: "pipe", maxBuffer: 4 << 20});
    git(["init", "--quiet", "--object-format=sha1"]);
    fs.copyFileSync(path.join(repo, ".gitattributes"), path.join(root, ".gitattributes"));
    for (const file of files) {
      fs.mkdirSync(path.dirname(path.join(root, file)), {recursive: true});
      fs.copyFileSync(path.join(repo, file), path.join(root, file));
    }
    git(["add", "--", ".gitattributes", ...files]);
    const checkout = path.join(root, "checkout"); fs.mkdirSync(checkout);
    git(["checkout-index", "--all", "--prefix=" + checkout + path.sep]);
    for (const file of files) {
      const digest = (/** @type {Buffer} */ bytes) => createHash("sha256").update(bytes).digest("hex"),
        expected = digest(fs.readFileSync(path.join(repo, file)));
      assert.equal(digest(git(["show", ":" + file])), expected, "indexed bytes: " + file);
      assert.equal(digest(fs.readFileSync(path.join(checkout, file))), expected, "checked-out bytes: " + file);
    }
  });

  test("pin snapshot covers exact native-source cohorts and semantic runtime inputs", () => {
    const pins = JSON.parse(JSON.stringify(dependencyPins()));
    assert.deepEqual(Object.keys(pins).sort(), ["archives", "buildNode", "bundledModules", "cutlass", "git", "nvrtc", "packingTool", "runtime"]);
    assert.deepEqual(Object.keys(pins.git).sort(), ["acpp", "compiler", "hermes", "linux", "rocm", "windows"]);
    assert.equal(pins.nvrtc.linux12_6.cohort, "linux-12.6");
    assert.equal(pins.nvrtc.linux13_3.version, "13.3.33");
    assert.equal(pins.archives.openssl.version, "3.5.9");
    assert.equal(pins.archives.zstd.version, "1.5.7");
    assert.match(pins.buildNode.version, /^v\d+\.\d+\.\d+$/);
    assert.match(pins.buildNode.pathSourceSha256, /^[0-9a-f]{64}$/);
    assert.equal(pins.buildNode.pathSourceSha256, createHash("sha256").update(nativePathSource().source).digest("hex"));
    assert.equal(pins.cutlass.version, "v4.6.1");
    assert.equal(pins.cutlass.licenseSha256, "ad1513ca983da737a6843364ad19e8424a56d860736c2da773b8c5be527e1475");
    assert.equal(pins.packingTool.version, "5.2.1");
    assert.equal(pins.packingTool.source.notices.some((/** @type {{member: string}} */ row) => row.member.includes("lzma-sdk")), true);
    assert.deepEqual(Object.keys(pins.bundledModules).sort(),
      ["base64-js", "buffer", "core-js-pure", "events", "ieee754", "safe-buffer", "string_decoder"]);
    assert.equal(pins.bundledModules["buffer"].version, "6.0.3");
    assert.equal(pins.bundledModules["buffer"].integrity.startsWith("sha512-"), true);
    assert.deepEqual(Object.keys(pins.runtime).sort(), ["linuxCudart", "windowsAmdhip", "windowsCudart", "windowsHostSdk", "windowsIntelRuntime"]);
    assert.equal(pins.runtime.linuxCudart.manifest.sha256, "9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf");
    assert.equal(pins.runtime.linuxCudart.archive.sha256, "f74689258a60fd9c5bdfa7679458527a55e22442691ba678dcfaeffbf4391ef9");
    assert.equal(pins.runtime.linuxCudart.libraries.cudartStatic.sha256, "5a90415e481a710fa00676f9fe1bbae703f33450c4fd3f35146a3f2980567b06");
    assert.equal(pins.runtime.linuxCudart.libraries.culibos.sha256, "5349773e44c22fd17aefd46f9f595e07eb033bd04017423f41298bf5f935900a");
    assert.equal(pins.runtime.linuxCudart.license.sha256, "e2c71babfd18a8e69542dd7e9ca018f9caa438094001a58e6bc4d8c999bf0d07");
  });

  test("LLVM Clang and LLD attributions match their pinned license members", () => {
    const notices = fs.readFileSync(path.join(__dirname, "..", "THIRD-PARTY-NOTICES.txt"), "utf8"), llvm = dependencyPins().archives.llvm;
    assert.ok(notices.includes("Copyright (c) 2007-2019 University of Illinois at Urbana-Champaign."), "Clang NCSA attribution is retained");
    assert.ok(notices.includes("Copyright (c) 2011-2019 by the contributors listed in CREDITS.TXT"), "LLD NCSA attribution is retained");
    assert.deepEqual(llvm.licenseMembers, [
      {component: "LLVM Clang subproject license (llvm-project-21.1.8.src/clang/LICENSE.TXT)",
        member: "llvm-project-21.1.8.src/clang/LICENSE.TXT", sha256: "ebcd9bbf783a73d05c53ba4d586b8d5813dcdf3bbec50265860ccc885e606f47"},
      {component: "LLVM LLD subproject license (llvm-project-21.1.8.src/lld/LICENSE.TXT)",
        member: "llvm-project-21.1.8.src/lld/LICENSE.TXT", sha256: "f7891568956e34643eb6a0db1462db30820d40d7266e2a78063f2fe233ece5a0"},
    ]);
  });

  /** @param {import("node:test").TestContext} t */
  test("matching reviewed bytes and current semantic pins pass", t => {
    const f = fixture(t), result = checkNativeNoticeReview(f.manifestPath, f.noticePath);
    assert.equal(result.notice.bytes, f.body.length);
    assert.deepEqual(result.dependencies, f.manifest.dependencies);
    assert.deepEqual(result.redistributionEvidence, f.manifest.redistributionEvidence);
  });

  /** @param {import("node:test").TestContext} t */
  test("Windows SDK redistribution evidence pins both preserved raw files", t => {
    const f = fixture(t), asset = path.join(f.root, "runtime/license-evidence/sdk_license.rtf");
    fs.unlinkSync(asset);
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /Windows SDK license evidence is missing or unreadable/);
    fs.copyFileSync(path.join(__dirname, "..", "runtime/license-evidence/sdk_license.rtf"), asset);
    fs.appendFileSync(asset, "changed");
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /Windows SDK license evidence differs from its reviewed byte pin/);
    fs.copyFileSync(path.join(__dirname, "..", "runtime/license-evidence/sdk_license.rtf"), asset);
    const notices = path.join(f.root, "runtime/license-evidence/sdk_third_party_notices.rtf");
    fs.unlinkSync(notices);
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /Windows SDK third-party notice evidence is missing or unreadable/);
    fs.copyFileSync(path.join(__dirname, "..", "runtime/license-evidence/sdk_third_party_notices.rtf"), notices);
    fs.appendFileSync(notices, "changed");
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /Windows SDK third-party notice evidence differs from its reviewed byte pin/);
  });

  /** @param {import("node:test").TestContext} t */
  test("Windows SDK version, UCRT member, and official source references are fixed", t => {
    const f = fixture(t), expected = f.manifest.redistributionEvidence.windowsHostSdk;
    expected.sdkVersion = "10.0.26100.1";
    save(f); assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /reviewed source metadata/);
    expected.sdkVersion = "10.0.26100.0"; expected.ucrtLibrary.outputMember = "sdk/lib/ucrt/other.lib";
    save(f); assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /reviewed source metadata/);
    expected.ucrtLibrary.outputMember = "sdk/lib/ucrt/libucrt.lib"; expected.redist.url = "http://example.invalid/redist";
    save(f); assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /reviewed source metadata/);
    expected.redist.url = "https://learn.microsoft.com/en-us/legal/windows-sdk/redist";
    expected.acquisitionBasis.eulaUrl = "https://example.invalid/community-eula.docx";
    save(f); assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /reviewed source metadata/);
  });

  /** @param {import("node:test").TestContext} t */
  test("changed notice body is rejected", t => {
    const f = fixture(t); fs.appendFileSync(f.noticePath, "drift\n");
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /differ from the reviewed byte pin/);
  });

  /** @param {import("node:test").TestContext} t */
  test("re-pinning changed LLVM license text does not satisfy its archive-member pin", t => {
    const f = fixture(t), changed = fs.readFileSync(f.noticePath, "utf8").replace(
      "Copyright (c) 2007-2019 University of Illinois at Urbana-Champaign.",
      "Copyright (c) 2007-2019 Unreviewed replacement.");
    fs.writeFileSync(f.noticePath, changed);
    f.manifest.notice = {bytes: Buffer.byteLength(changed), sha256: createHash("sha256").update(changed).digest("hex")};
    save(f);
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /pinned license member: llvm-project-21\.1\.8\.src\/clang\/LICENSE\.TXT/);
  });

  test("Linux CUDART license member is retained under its reviewed notice heading", t => {
    const catalog = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "runtime/linux-cudart.json"), "utf8")),
      notice = fs.readFileSync(path.join(__dirname, "..", "THIRD-PARTY-NOTICES.txt"), "utf8"),
      marker = "================================================================================\nComponents: " + catalog.license.noticeComponent + "\n\n",
      start = notice.indexOf(marker);
    assert.notEqual(start, -1);
    const bodyStart = start + marker.length, end = notice.indexOf("\n================================================================================\nComponents: ", bodyStart),
      body = Buffer.from(notice.slice(bodyStart, end < 0 ? notice.length : end));
    assert.equal(body.length, catalog.license.bytes);
    assert.equal(createHash("sha256").update(body).digest("hex"), catalog.license.sha256);
    const f = fixture(t);
    assert.doesNotThrow(() => checkNativeNoticeReview(f.manifestPath, f.noticePath));
  });

  /** @param {import("node:test").TestContext} t */
  test("re-pinning a non-newline CUDA license suffix does not satisfy its archive-member pin", t => {
    const f = fixture(t), original = fs.readFileSync(f.noticePath);
    assert.equal(original.at(-1), 0x0a, "fixture ends with the archived CUDA license's source LF");
    fs.writeFileSync(f.noticePath, Buffer.concat([original.subarray(0, -1), Buffer.from("XY", "utf8")]));
    const body = fs.readFileSync(f.noticePath);
    f.manifest.notice = {bytes: body.length, sha256: createHash("sha256").update(body).digest("hex")};
    save(f);
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /pinned license member: cuda_cudart-linux-x86_64-12\.6\.77-archive\/LICENSE/);
  });

  /** @param {import("node:test").TestContext} t */
  test("notice length and digest are both reviewed pins", t => {
    const f = fixture(t); f.manifest.notice.bytes++;
    save(f); assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /reviewed byte pin/);
  });

  /** @param {import("node:test").TestContext} t */
  test("semantic dependency drift fails the reviewed snapshot", t => {
    const f = fixture(t); f.manifest.dependencies.archives.openssl.version = "0.0.0";
    save(f); assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /dependency pins differ/);
  });

  /** @param {import("node:test").TestContext} t */
  test("omitted runtime bundle package cannot be treated as current", t => {
    const f = fixture(t); delete f.manifest.dependencies.bundledModules.buffer;
    save(f); assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /dependency pins differ/);
  });

  /** @param {import("node:test").TestContext} t */
  test("unknown extra dependency pin is rejected", t => {
    const f = fixture(t); f.manifest.dependencies.unreviewed = "1.0.0";
    save(f); assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /dependency pins differ/);
  });

  /** @param {import("node:test").TestContext} t */
  test("extra manifest fields are rejected", t => {
    const f = fixture(t); Reflect.set(f.manifest, "legalComplete", true);
    save(f); assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /unsupported schema/);
  });

  /** @param {import("node:test").TestContext} t */
  test("unsupported manifest version is rejected", t => {
    const f = fixture(t); f.manifest.version = 2;
    save(f); assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /unsupported schema/);
  });

  /** @param {import("node:test").TestContext} t */
  test("notice schema rejects additional fields and malformed hashes", t => {
    const f = fixture(t); Reflect.set(f.manifest.notice, "extra", true);
    save(f); assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /unsupported schema/);
    Reflect.deleteProperty(f.manifest.notice, "extra"); f.manifest.notice.sha256 = "A".repeat(64);
    save(f); assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /unsupported schema/);
  });

  /** @param {import("node:test").TestContext} t */
  test("malformed manifest JSON is rejected", t => {
    const f = fixture(t); fs.writeFileSync(f.manifestPath, "{");
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), SyntaxError);
  });

  /** @param {import("node:test").TestContext} t */
  test("non-UTF-8 notice bytes are rejected", t => {
    const f = fixture(t); fs.writeFileSync(f.noticePath, Buffer.from([0xff, 0xfe]));
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /valid UTF-8/);
  });

  /** @param {import("node:test").TestContext} t */
  test("relative and non-regular paths are rejected", t => {
    const f = fixture(t);
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, "relative-notices.txt"), /absolute/);
    assert.throws(() => checkNativeNoticeReview(f.root, f.noticePath), /Invalid or excessive build receipt/);
  });

  /** @param {import("node:test").TestContext} t */
  test("missing manifest path does not fall back to another review", t => {
    const f = fixture(t), missing = path.join(f.root, "missing.json");
    assert.throws(() => checkNativeNoticeReview(missing, f.noticePath), /ENOENT/);
  });

  /** @param {import("node:test").TestContext} t */
  test("notice bytes containing NUL are rejected", t => {
    const f = fixture(t); fs.writeFileSync(f.noticePath, Buffer.from([0x61, 0, 0x62]));
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /NUL/);
  });

  /** @param {import("node:test").TestContext} t */
  test("executable notice files are rejected", t => {
    const f = fixture(t);
    if (process.platform === "win32") {t.skip("Windows does not expose POSIX executable mode bits"); return;}
    fs.chmodSync(f.noticePath, 0o744);
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, f.noticePath), /non-executable/);
  });

  /** @param {import("node:test").TestContext} t */
  test("symlinked parent paths are rejected for metadata and notice files", t => {
    const f = fixture(t);
    if (process.platform === "win32") {t.skip("Directory symlinks may require elevated Windows privileges"); return;}
    const alias = path.join(f.root, "alias"); fs.symlinkSync(f.root, alias, "dir");
    assert.throws(() => checkNativeNoticeReview(path.join(alias, "runtime", path.basename(f.manifestPath)), f.noticePath), /canonical absolute path/);
    assert.throws(() => checkNativeNoticeReview(f.manifestPath, path.join(alias, path.basename(f.noticePath))), /symlink/);
  });
});
