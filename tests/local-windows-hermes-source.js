"use strict";

const {describe, test} = require("node:test"), assert = require("node:assert/strict"),
  fs = require("node:fs"), path = require("node:path"), os = require("node:os"), {spawnSync} = require("node:child_process"),
  {createHash} = require("node:crypto");
const ci = require("../scripts/native-ci-windows-controller"), {fileHash} = require("../scripts/build-source-inputs"),
  {hermesPins} = require("../scripts/build-native-hermes"), {origins, sourceRecipe} = require("../scripts/fetch-native-sdk-sources");

const repo = fs.realpathSync(path.resolve(__dirname, "..")), tarTool = "/usr/bin/tar", git = "/usr/bin/git";

/** @param {import('node:test').TestContext} t @param {{rootName?:string,originalRoot?:string}} [options] */
function fixture(t, options = {}) {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-hermes-source-")), rootName = options.rootName || "local-hermes-source",
    originalRoot = options.originalRoot || path.join(temporary, "wsl-native-sdk/hermes-fetch/roots/hermes"),
    cohort = path.join(temporary, rootName), bare = path.join(cohort, "roots/hermes"), outputParent = path.join(temporary, "extracted"),
    extractRoot = path.join(outputParent, "source-input"), archive = path.join(temporary, "hermes-source.tar"),
    state = {kind: /** @type {"hermes"} */ ("hermes"), hermesExtractRoot: extractRoot, hermesSourceTar: archive, hermesSourceName: rootName},
    rootsBytes = Buffer.from(JSON.stringify({hermes: originalRoot}, null, 2) + "\n"), recipe = sourceRecipe("hermes"),
    recipeSha256 = createHash("sha256").update(JSON.stringify(recipe)).digest("hex"),
    fetch = {version: 1, target: "hermes", status: "PASS", networked: true, buildOnly: true, compiled: false, publicationQualified: false,
      git: {path: git, sha256: fileHash(git)}, fetcherSha256: fileHash(path.join(repo, "scripts/fetch-native-sdk-sources.js")),
      recipe: {sha256: recipeSha256, files: recipe.graph},
      components: {hermes: {origin: origins.hermes, revision: hermesPins.revision, root: originalRoot,
        gitDirectory: "bare", shallow: true, partialClone: false}}};
  fs.mkdirSync(bare, {recursive: true}); fs.mkdirSync(outputParent);
  fs.writeFileSync(path.join(bare, "config"), "[core]\n\trepositoryformatversion = 0\n\tbare = true\n", {flag: "wx"});
  fs.writeFileSync(path.join(cohort, "roots.json"), rootsBytes, {flag: "wx"});
  const fetchBytes = Buffer.from(JSON.stringify(fetch, null, 2) + "\n");
  fs.writeFileSync(path.join(cohort, "fetch.json"), fetchBytes, {flag: "wx"});
  const created = spawnSync(tarTool, ["--format=ustar", "-cf", archive, "-C", temporary, rootName], {encoding: "utf8", timeout: 30000});
  assert.equal(created.error, undefined); assert.equal(created.status, 0, created.stderr);
  t.after(() => fs.rmSync(temporary, {recursive: true, force: true}));
  return {temporary, rootName, originalRoot, cohort, bare, outputParent, extractRoot, archive, state, rootsBytes, fetchBytes};
}

/** Replace the first USTAR member name with a traversal path while keeping its header checksum valid. @param {string} archive @param {string} name */
function setFirstTarMemberName(archive, name) {
  const bytes = fs.readFileSync(archive), header = Buffer.from(bytes.subarray(0, 512));
  assert.ok(bytes.length >= 1024); assert.ok(Buffer.byteLength(name) < 100);
  header.fill(0, 0, 100); header.write(name, 0, "utf8"); header.fill(0x20, 148, 156);
  let checksum = 0;
  for (const byte of header) {checksum += byte;}
  const octal = checksum.toString(8).padStart(6, "0");
  header.write(octal, 148, "ascii"); header[154] = 0; header[155] = 0x20; header.copy(bytes, 0);
  fs.writeFileSync(archive, bytes);
}

describe("Local Windows Hermes source transport", {skip: process.platform !== "linux"}, () => {
  test("extracts the admitted local WSL root without changing original receipts and supports warm readmission", t => {
    const f = fixture(t), result = ci.extractWindowsHermesSource(f.state, tarTool, 30000, f.originalRoot),
      warm = ci.readWindowsHermesSource({hermesExtractRoot: f.extractRoot, hermesSourceName: f.rootName}, f.originalRoot);
    assert.equal(result.source, path.join(f.extractRoot, f.rootName, "roots/hermes"));
    assert.equal(result.tarSha256, fileHash(f.archive));
    assert.equal(warm.fetchSha256, fileHash(path.join(f.cohort, "fetch.json")));
    assert.equal(warm.rootsSha256, fileHash(path.join(f.cohort, "roots.json")));
    assert.deepEqual(fs.readFileSync(warm.fetchFile), f.fetchBytes);
    assert.deepEqual(fs.readFileSync(warm.rootsFile), f.rootsBytes);
    assert.deepEqual(fs.readFileSync(path.join(f.cohort, "fetch.json")), f.fetchBytes);
    assert.deepEqual(fs.readFileSync(path.join(f.cohort, "roots.json")), f.rootsBytes);
  });

  test("keeps the existing CI default /tmp source root", t => {
    const rootName = "mom-native-hermes-prerequisites-24680-1", originalRoot = "/tmp/" + rootName + "/roots/hermes",
      f = fixture(t, {rootName, originalRoot}), result = ci.extractWindowsHermesSource(f.state, tarTool, 30000);
    assert.equal(result.source, path.join(f.extractRoot, rootName, "roots/hermes"));
    assert.equal(result.fetch.components.hermes.root, originalRoot);
    assert.equal(ci.readWindowsHermesSource({hermesExtractRoot: f.extractRoot, hermesSourceName: rootName}).fetchSha256, result.fetchSha256);
  });

  test("rejects a noncanonical explicit original root before creating output", t => {
    const f = fixture(t), invalid = "/tmp/local-hermes/../elsewhere/roots/hermes";
    assert.throws(() => ci.extractWindowsHermesSource(f.state, tarTool, 30000, invalid), /canonical POSIX roots\/hermes/);
    assert.equal(fs.existsSync(f.extractRoot), false);
    assert.equal(fs.readFileSync(path.join(f.cohort, "fetch.json"), "utf8"), f.fetchBytes.toString("utf8"));
  });

  test("rejects traversal members before extraction", t => {
    const f = fixture(t); setFirstTarMemberName(f.archive, f.rootName + "/../escape");
    assert.throws(() => ci.extractWindowsHermesSource(f.state, tarTool, 30000, f.originalRoot), /unsafe paths/);
    assert.equal(fs.existsSync(f.extractRoot), false);
    assert.deepEqual(fs.readFileSync(path.join(f.cohort, "roots.json")), f.rootsBytes);
  });

  test("preserves an occupied extraction output", t => {
    const f = fixture(t), marker = path.join(f.extractRoot, "operator-data");
    fs.mkdirSync(f.extractRoot); fs.writeFileSync(marker, "preserve\n", {flag: "wx"});
    assert.throws(() => ci.extractWindowsHermesSource(f.state, tarTool, 30000, f.originalRoot), /new output/);
    assert.equal(fs.readFileSync(marker, "utf8"), "preserve\n");
  });

  test("rejects a valid but mismatched original root without changing receipts", t => {
    const f = fixture(t), otherRoot = "/mnt/other/native-sdk/hermes-fetch/roots/hermes";
    assert.throws(() => ci.extractWindowsHermesSource(f.state, tarTool, 30000, otherRoot), /do not bind the original pinned bare Git root/);
    assert.deepEqual(fs.readFileSync(path.join(f.cohort, "fetch.json")), f.fetchBytes);
    assert.deepEqual(fs.readFileSync(path.join(f.cohort, "roots.json")), f.rootsBytes);
  });
});
