"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"),
  path = require("node:path"), os = require("node:os");
const {fileHash} = require("../scripts/build-source-inputs");
const local = require("../scripts/build-local-native-sdk");
const {buildLocalNativeDistribution} = require("../scripts/build-local-native-distribution");
const repo = fs.realpathSync(path.resolve(__dirname, ".."));
// The default archive name carries the package version, so a release bump needs no test edit.
const {version} = require("../package.json");

/** No producer runs: retain only the actual API boundary fields inspected by the frontdoor.
 * @param {import('node:test').TestContext} t @param {"linux"|"win32"} [target] */
function fixture(t, target = "linux") {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-distribution-")), workspace = path.join(root, "sdk"),
    configFile = path.join(workspace, "application-sdk.json"), output = path.join(workspace, "candidate"),
    notices = path.join(repo, "THIRD-PARTY-NOTICES.txt"), calls = /** @type {string[]} */ ([]);
  local.admitWorkspace(workspace, repo);
  fs.writeFileSync(configFile, JSON.stringify({...(target === "win32" ? {windowsSdk: {}} : {}),
    notices: {path: notices, sha256: fileHash(notices)}}));
  const sdk = {workspace, configFile, tools: {cmake: "/fixture/cmake"}};
  /** @type {any} */ const dependencies = {
    verifyScope() {calls.push("scope");},
    review() {calls.push("notices"); return {notice: {sha256: fileHash(notices)}};},
    preparePacking() {calls.push("packing-tool"); return {upx: "/fixture/upx", upxSha256: "0".repeat(64)};},
    build(/** @type {string} */ config, /** @type {string} */ directory, /** @type {unknown} */ packing, /** @type {NodeJS.ProcessEnv} */ env) {
      calls.push("build"); assert.equal(config, configFile); assert.equal(directory, output);
      assert.equal(env["MOM_AOT_BUILD_ONLY"], "1"); assert.ok(packing);
      fs.mkdirSync(output); const executable = path.join(output, target === "win32" ? "mom.exe" : "mom"); fs.writeFileSync(executable, "inert fixture");
      return {artifact: {path: executable, sha256: fileHash(executable), bytes: fs.statSync(executable).size}};
    },
    audit(/** @type {string} */ entrypoint, /** @type {{archive?:{path:string,tool:string}}} */ options = {}) {
      calls.push(options.archive ? "archive-audit" : "binary-audit"); assert.equal(entrypoint, path.join(output, "entrypoint.json"));
      if (options.archive) {assert.equal(options.archive.tool, "/fixture/cmake");}
    },
    package(/** @type {string} */ _entrypoint, /** @type {string} */ _cmake, /** @type {string} */ version) {
      calls.push("package"); const archive = path.join(output, `mom-v${version}-${target === "win32" ? "win.zip" : "lin.tgz"}`); fs.writeFileSync(archive, "inert archive");
      return {archive: {path: archive, bytes: fs.statSync(archive).size, sha256: fileHash(archive)}};
    },
  };
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {workspace, configFile, output, sdk: /** @type {Parameters<typeof buildLocalNativeDistribution>[0]} */ (sdk), dependencies, calls};
}

test.describe("Local native distribution assembly", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("builds, audits, packages and re-audits through shared APIs before writing the checksum", t => {
    const f = fixture(t), result = buildLocalNativeDistribution(f.sdk, f.output, {}, f.dependencies);
    assert.deepEqual(f.calls, ["scope", "notices", "packing-tool", "build", "binary-audit", "package", "archive-audit"]);
    assert.equal(fs.readFileSync(result.sidecar, "utf8"), result.archive.sha256 + "  " + path.basename(result.archive.path) + "\n");
    assert.equal(result.runtimeTested, false); assert.equal(result.publicationQualified, false);
  });
  test("requires the actual memory scope before inspecting or mutating paths", t => {
    const f = fixture(t); f.dependencies.verifyScope = () => {throw new Error("scope absent");};
    assert.throws(() => buildLocalNativeDistribution(f.sdk, f.output, {}, f.dependencies), /scope absent/);
    assert.deepEqual(f.calls, []); assert.equal(fs.existsSync(f.output), false);
  });
  test("Windows-target WSL assembly delegates the unchanged descriptor to the same release pipeline", t => {
    const f = fixture(t, "win32"), original = fs.readFileSync(f.configFile),
      result = buildLocalNativeDistribution(f.sdk, f.output, {}, f.dependencies);
    assert.deepEqual(f.calls, ["scope", "notices", "packing-tool", "build", "binary-audit", "package", "archive-audit"]);
    assert.deepEqual(fs.readFileSync(f.configFile), original);
    assert.equal(path.basename(result.executable.path), "mom.exe");
    assert.equal(path.basename(result.archive.path), `mom-v${version}-win.zip`);
    assert.equal(fs.readFileSync(result.sidecar, "utf8"), `${result.archive.sha256}  mom-v${version}-win.zip\n`);
    assert.equal(result.runtimeTested, false);
    assert.equal(result.publicationQualified, false);
  });
  test("uses the shared version normalizer for an explicit release-version override", t => {
    const f = fixture(t), result = buildLocalNativeDistribution(f.sdk, f.output, {MOM_RELEASE_VERSION: "v0.9.4-test"}, f.dependencies);
    assert.equal(path.basename(result.archive.path), "mom-v0.9.4-test-lin.tgz");
    assert.match(fs.readFileSync(result.sidecar, "utf8"), /mom-v0\.9\.4-test-lin\.tgz\n$/);
  });
  test("rejects an invalid version before tool acquisition or build mutation", t => {
    const f = fixture(t);
    assert.throws(() => buildLocalNativeDistribution(f.sdk, f.output, {MOM_RELEASE_VERSION: "../bad"}, f.dependencies), /Invalid release version/);
    assert.deepEqual(f.calls, ["scope"]); assert.equal(fs.existsSync(f.output), false);
  });
  test("preserves existing candidate evidence without invoking producers", t => {
    const f = fixture(t); fs.mkdirSync(f.output); fs.writeFileSync(path.join(f.output, "keep"), "keep");
    assert.throws(() => buildLocalNativeDistribution(f.sdk, f.output, {}, f.dependencies), /fresh workspace-owned output/);
    assert.deepEqual(f.calls, ["scope"]); assert.equal(fs.readFileSync(path.join(f.output, "keep"), "utf8"), "keep");
  });
  test("rejects a stale notice binding before fetching the packer or building", t => {
    const f = fixture(t); fs.writeFileSync(f.configFile, JSON.stringify({notices: {path: "/wrong", sha256: "0".repeat(64)}}));
    assert.throws(() => buildLocalNativeDistribution(f.sdk, f.output, {}, f.dependencies), /current reviewed distribution notices/);
    assert.deepEqual(f.calls, ["scope", "notices"]);
  });
  test("a build failure does not package or write passing checksum evidence", t => {
    const f = fixture(t); f.dependencies.build = () => {throw new Error("compiler failed");};
    assert.throws(() => buildLocalNativeDistribution(f.sdk, f.output, {}, f.dependencies), /compiler failed/);
    assert.deepEqual(f.calls, ["scope", "notices", "packing-tool"]);
  });
  test("changed descriptor bytes are rejected after packaging rather than certified", t => {
    const f = fixture(t), audit = f.dependencies.audit;
    f.dependencies.audit = (/** @type {string} */ entrypoint, /** @type {any} */ options) => {
      audit(entrypoint, options); if (options?.archive) {fs.appendFileSync(f.configFile, " ");}
    };
    assert.throws(() => buildLocalNativeDistribution(f.sdk, f.output, {}, f.dependencies), /inputs changed during assembly/);
    assert.equal(fs.existsSync(path.join(f.output, `mom-v${version}-lin.tgz.sha256`)), false);
  });
});
