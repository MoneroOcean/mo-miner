"use strict";

const {test, describe} = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {Readable} = require("node:stream"), {createHash} = require("node:crypto");
const {fetchNvrtcArchives, nvrtcArchiveCatalog} = require("../scripts/fetch-native-sdk-sources");
const {fileHash} = require("../scripts/build-source-inputs");
const {prepareNvrtcSdk, admitNvrtcSdk, nvrtcSdkInputs, nvrtcSdkRecipe, memberPlan, streamMember} = require("../scripts/prepare-native-nvrtc-sdk");
const {oracleEnvironment} = require("../scripts/probe-static-nvrtc");
const {tempDir} = require("./common/temp-dir");
const {storedZip} = require("./common/zip-fixture");

/** @typedef {"linux" | "windows"} Platform */
/** @typedef {"nvrtc" | "nvptxcompiler"} ArchiveName */
/** @typedef {Record<string, Buffer>} TestArchives */
/** @typedef {Record<string, {sha256: string}>} ArchiveHashes */
/** @typedef {{headerSymlink?: boolean, missingMember?: boolean, malformedPath?: boolean}} FixtureOptions */
/** @typedef {Array<[string, Uint8Array]>} ZipEntries */
/** @typedef {(url: string, options: {redirect: "manual", signal: AbortSignal}) => Promise<Pick<Response, "status" | "url" | "body"> & {headers: Pick<Headers, "get">}>} TestArchiveFetch */

/** @param {Buffer} data */
function sha256(data) {return createHash("sha256").update(data).digest("hex");}

/** @param {Platform} platform @param {string} fileName */
function archiveRoot(platform, fileName) {return fileName.slice(0, -(platform === "linux" ? ".tar.xz" : ".zip").length);}

/** Create only the selected synthetic vendor archives using deterministic fixture members. */
/** @param {string} root @param {Platform} platform @param {FixtureOptions} [options] @param {"linux" | "windows" | "linux-12.6"} [cohort] @returns {TestArchives} */
function buildArchives(root, platform, options = {}, cohort = platform) {
  const catalog = nvrtcArchiveCatalog(platform, undefined, cohort), plan = memberPlan(platform, cohort), /** @type {TestArchives} */ data = {};
  if (platform === "linux") {
    const base = path.join(root, "tar-roots"); fs.mkdirSync(base);
    for (const name of /** @type {ArchiveName[]} */ (Object.keys(catalog))) {
      const archiveName = catalog[name]?.fileName;
      if (!archiveName) {throw new Error("fixture archive pin is missing: " + name);}
      const rootName = archiveRoot(platform, archiveName), rootPath = path.join(base, rootName); fs.mkdirSync(rootPath);
      for (const row of plan) {
        if (!("source" in row) || row.source.archive !== name) {continue;}
        if (options.missingMember && cohort === "linux-12.6" && row.output === "lib64/libnvrtc.so.12.6.85") {continue;}
        const relative = row.source.member.slice(rootName.length + 1), output = path.join(rootPath, relative);
        fs.mkdirSync(path.dirname(output), {recursive: true});
        if (options.headerSymlink && row.output === "include/nvrtc.h") {
          fs.writeFileSync(path.join(rootPath, "include", "header-target.h"), "safe target\n");
          fs.symlinkSync("header-target.h", output);
        } else {fs.writeFileSync(output, "fixture " + row.source.member + "\n");}
      }
      const archive = path.join(root, archiveName), result = spawnSync("tar", ["-cJf", archive, "-C", base, rootName], {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.stderr); data[name] = fs.readFileSync(archive);
    }
  } else {
    for (const name of /** @type {ArchiveName[]} */ (Object.keys(catalog))) {
      const archiveName = catalog[name]?.fileName;
      if (!archiveName) {throw new Error("fixture archive pin is missing: " + name);}
      const rootName = archiveRoot(platform, archiveName), /** @type {ZipEntries} */ entries = [];
      for (const row of plan) {
        if (!("source" in row) || row.source.archive !== name) {continue;}
        if (options.missingMember && name === "nvrtc" && row.output === "lib/x64/nvrtc_static.lib") {continue;}
        entries.push([row.source.member, Buffer.from("fixture " + row.source.member + "\n")]);
      }
      if (options.malformedPath && name === "nvrtc") {entries.push([rootName + "/../escape", Buffer.from("not extracted")]);}
      data[name] = storedZip(entries);
    }
  }
  assert.ok(Object.values(data).every(bytes => bytes.length < (1 << 20)), "synthetic archives stay below 1 MiB");
  return data;
}

/** @param {import("node:test").TestContext} t @param {Platform} platform @param {FixtureOptions} [options] @param {"linux" | "windows" | "linux-12.6"} [cohort]
 * @returns {Promise<{root: string, source: string, output: string, hashes: ArchiveHashes, archives: TestArchives}>} */
async function assembledFixture(t, platform, options = {}, cohort = platform) {
  const root = tempDir(t, "mom-nvrtc-sdk-"), archives = buildArchives(root, platform, options, cohort), source = path.join(root, "verified-originals"), output = path.join(root, "assembled-sdk"),
    hashes = Object.fromEntries(Object.entries(archives).map(([name, bytes]) => [name, {sha256: sha256(bytes)}]));
  const fetch = /** @type {TestArchiveFetch} */ (async (url, _options) => {
    const name = url.includes("/cuda_nvrtc/") ? "nvrtc" : "nvptxcompiler", body = archives[name];
    if (!body) {throw new Error("unknown synthetic NVRTC URL");}
    return {status: 200, url, headers: {get: name => name === "content-length" ? String(body.length) : null}, body: Readable.toWeb(Readable.from([body]))};
  });
  await fetchNvrtcArchives(platform, source, {MOM_AOT_BUILD_ONLY: "1"}, true, {fetch, hashes, cohort});
  return {root, source, output, hashes, archives};
}

describe("offline NVRTC SDK assembly and admission", {skip: process.platform !== "linux"}, () => {
  test("shared NVRTC oracle prepends the SDK path without mutating its inherited environment", () => {
    const inherited = {PATH: "/usr/bin", LD_LIBRARY_PATH: "/intel/lib:/opt/ubuntu24-libs:"}, inheritedBefore = {...inherited};
    assert.deepEqual(oracleEnvironment("/sdk/lib64", inherited), {
      PATH: "/usr/bin", LD_LIBRARY_PATH: "/sdk/lib64:/intel/lib:/opt/ubuntu24-libs",
    });
    assert.deepEqual(inherited, inheritedBefore);

    const absent = {PATH: "/usr/bin"}, absentBefore = {...absent};
    assert.deepEqual(oracleEnvironment("/sdk/lib64", absent), {PATH: "/usr/bin", LD_LIBRARY_PATH: "/sdk/lib64"});
    assert.deepEqual(absent, absentBefore);

    const processEnvBefore = {...process.env}, defaultEnvironment = oracleEnvironment("/sdk/lib64");
    const inheritedPaths = (processEnvBefore["LD_LIBRARY_PATH"] || "").split(path.delimiter).filter(Boolean);
    assert.equal(defaultEnvironment["LD_LIBRARY_PATH"], ["/sdk/lib64", ...inheritedPaths].join(path.delimiter));
    assert.deepEqual({...process.env}, processEnvBefore);
  });

  test("a byte-limit failure waits for a TERM-ignoring extractor to be killed", async t => {
    const root = tempDir(t, "mom-nvrtc-sdk-"), pidFile = path.join(root, "child.pid"), destination = path.join(root, "partial.bin");
    const child = `require("node:fs").writeFileSync(${JSON.stringify(pidFile)},String(process.pid));` +
      "process.on('SIGTERM',()=>{});process.stdout.write('too many bytes');setInterval(()=>{},1000);";
    const started = Date.now();
    await assert.rejects(streamMember(process.execPath, ["-e", child], destination, 1, "fixture member"), /configured byte limit/);
    assert.ok(Date.now() - started >= 1800, "streamer waited for SIGKILL escalation and child close");
    assert.ok(fs.existsSync(pidFile));
  });

  for (const platform of /** @type {Platform[]} */ (["linux", "windows"])) {
    test(platform + " assembles only required pinned members and admits the read-only layout", async t => {
      const f = await assembledFixture(t, platform), receipt = await prepareNvrtcSdk(platform, f.source, f.output, {hashes: f.hashes});
      assert.equal(receipt.vendorVersion, platform === "linux" ? "13.3.33" : "13.1.115");
      assert.equal(receipt.networked, false); assert.equal(receipt.vendorPrebuilt, true); assert.equal(receipt.sourceRebuilt, false);
      assert.equal(receipt.publicationQualified, false); assert.equal(receipt.redistributionQualified, false);
      assert.deepEqual(admitNvrtcSdk(platform, f.output, {hashes: f.hashes}), receipt);
      const inputs = nvrtcSdkInputs(platform, f.output, {hashes: f.hashes}), expectedInputs = [path.join(f.output, "sdk.json"),
        ...memberPlan(platform).filter(row => "source" in row).map(row => path.join(f.output, ...row.output.split("/"))),
        path.join(receipt.source.directory, "fetch.json"), ...Object.values(receipt.source.receipt.archives).map(row => row.path), receipt.tool.path,
        ...Object.keys(receipt.recipe.files).map(relative => path.resolve(__dirname, "..", ...relative.split("/")))].sort();
      assert.deepEqual(inputs.receipt, receipt); assert.equal(inputs.receiptFile, path.join(f.output, "sdk.json"));
      assert.equal(inputs.receiptSha256, fileHash(inputs.receiptFile));
      assert.deepEqual(Object.keys(inputs.files).sort(), expectedInputs);
      for (const [file, hash] of Object.entries(inputs.files)) {assert.equal(fileHash(file), hash);}
      assert.ok(receipt.tool.sha256.match(/^[0-9a-f]{64}$/));
      assert.equal(fs.statSync(path.join(f.output, "sdk.json")).mode & 0o222, 0);
      assert.equal(fs.readFileSync(path.join(f.output, "include/nvrtc.h"), "utf8").startsWith("fixture "), true);
      assert.equal(fs.readFileSync(path.join(f.output, "include/nvPTXCompiler.h"), "utf8").startsWith("fixture "), true);
      assert.ok(fs.existsSync(path.join(f.output, "licenses/nvrtc/LICENSE")));
      assert.ok(fs.existsSync(path.join(f.output, "licenses/nvptxcompiler/LICENSE")));
      if (platform === "linux") {
        for (const name of ["libnvrtc_static.a", "libnvrtc-builtins_static.a", "libnvptxcompiler_static.a",
          "libnvrtc.so.13.3.33", "libnvrtc-builtins.so.13.3.33"]) {assert.ok(fs.existsSync(path.join(f.output, "lib64", name)));}
        assert.equal(fs.readlinkSync(path.join(f.output, "lib64/libnvrtc.so")), "libnvrtc.so.13");
        assert.equal(fs.readlinkSync(path.join(f.output, "lib64/libnvrtc.so.13")), "libnvrtc.so.13.3.33");
        assert.equal(fs.readlinkSync(path.join(f.output, "lib64/libnvrtc-builtins.so.13.3")), "libnvrtc-builtins.so.13.3.33");
      } else {
        for (const name of ["nvrtc_static.lib", "nvrtc-builtins_static.lib", "nvptxcompiler_static.lib"]) {assert.ok(fs.existsSync(path.join(f.output, "lib/x64", name)));}
      }
      assert.deepEqual(Object.keys(receipt.files).sort(), memberPlan(platform).map(row => row.output).sort());
      assert.ok(Object.values(receipt.files).every(row => "alias" in row.source ? row.source.alias === true :
        Boolean(row.source.member && row.source.archive)));
    });
  }

  test("explicit Linux 12.6 assembly admits NVRTC-only inputs and never substitutes the default cohort", async t => {
    const cohort = "linux-12.6", f = await assembledFixture(t, "linux", {}, cohort), receipt = await prepareNvrtcSdk("linux", f.source, f.output, {hashes: f.hashes, cohort});
    assert.equal(receipt.target, "nvrtc-sdk-linux-12.6"); assert.equal(receipt.platform, "linux"); assert.equal(receipt.cohort, cohort);
    assert.equal(receipt.vendorVersion, "12.6.85"); assert.deepEqual(Object.keys(receipt.source.receipt.archives), ["nvrtc"]);
    assert.deepEqual(admitNvrtcSdk("linux", f.output, {hashes: f.hashes, cohort}), receipt);
    const outputs = Object.keys(receipt.files).sort();
    assert.ok(outputs.includes("include/nvrtc.h") && outputs.includes("licenses/nvrtc/LICENSE"));
    assert.ok(outputs.includes("lib64/libnvrtc.so.12.6.85") && outputs.includes("lib64/libnvrtc-builtins.so.12.6.85"));
    assert.ok(!outputs.some(file => /nvptxcompiler|nvPTXCompiler|_static\.a/.test(file)));
    assert.equal(fs.readlinkSync(path.join(f.output, "lib64/libnvrtc.so")), "libnvrtc.so.12");
    assert.equal(fs.readlinkSync(path.join(f.output, "lib64/libnvrtc.so.12")), "libnvrtc.so.12.6.85");
    const closure = nvrtcSdkInputs("linux", f.output, {hashes: f.hashes, cohort});
    assert.deepEqual(closure.receipt, receipt); assert.ok(Object.keys(closure.files).some(file => file.endsWith("/lib64/libnvrtc.so.12.6.85")));
    assert.throws(() => nvrtcSdkInputs("linux", f.output, {hashes: f.hashes}));
    assert.throws(() => memberPlan("windows", cohort), /not available/);
    assert.throws(() => memberPlan("linux", /** @type {"linux"} */ ("")), /not available/);
  });

  test("12.6 assembly rejects an archive missing its pinned shared runtime member", async t => {
    const f = await assembledFixture(t, "linux", {missingMember: true}, "linux-12.6");
    await assert.rejects(prepareNvrtcSdk("linux", f.source, f.output, {hashes: f.hashes, cohort: "linux-12.6"}), /member is missing or duplicated/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("malformed unselected ZIP paths are never materialized; wrong member types are rejected", async t => {
    const unsafe = await assembledFixture(t, "windows", {malformedPath: true});
    await prepareNvrtcSdk("windows", unsafe.source, unsafe.output, {hashes: unsafe.hashes});
    assert.equal(fs.existsSync(path.join(unsafe.root, "escape")), false);
    assert.equal(fs.existsSync(path.join(unsafe.output, "sdk.json")), true);
    const wrongType = await assembledFixture(t, "linux", {headerSymlink: true});
    await assert.rejects(prepareNvrtcSdk("linux", wrongType.source, wrongType.output, {hashes: wrongType.hashes}), /not a regular file/);
    assert.equal(fs.existsSync(wrongType.output), false);
  });

  test("missing required original member is rejected by exact member identity", async t => {
    const f = await assembledFixture(t, "windows", {missingMember: true});
    await assert.rejects(prepareNvrtcSdk("windows", f.source, f.output, {hashes: f.hashes}), /member is missing or duplicated/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("stale producer recipe, source receipt, output bytes, and unexpected files fail read-only admission", async t => {
    const stale = await assembledFixture(t, "linux"), current = nvrtcSdkRecipe();
    const changedRecipe = () => ({...current, graph: {...current.graph, drift: "0".repeat(64)}});
    await prepareNvrtcSdk("linux", stale.source, stale.output, {hashes: stale.hashes, recipe: changedRecipe});
    assert.throws(() => nvrtcSdkInputs("linux", stale.output, {hashes: stale.hashes}), /stale producer recipe/);

    const sourceChanged = await assembledFixture(t, "windows");
    await prepareNvrtcSdk("windows", sourceChanged.source, sourceChanged.output, {hashes: sourceChanged.hashes});
    const fetchReceipt = path.join(sourceChanged.source, "fetch.json"), receipt = JSON.parse(fs.readFileSync(fetchReceipt, "utf8"));
    receipt.target = "nvrtc-linux"; fs.chmodSync(fetchReceipt, 0o644); fs.writeFileSync(fetchReceipt, JSON.stringify(receipt) + "\n"); fs.chmodSync(fetchReceipt, 0o444);
    assert.throws(() => nvrtcSdkInputs("windows", sourceChanged.output, {hashes: sourceChanged.hashes}), /valid build-only|stale|changed/);

    const archiveChanged = await assembledFixture(t, "linux");
    await prepareNvrtcSdk("linux", archiveChanged.source, archiveChanged.output, {hashes: archiveChanged.hashes});
    const acquisition = JSON.parse(fs.readFileSync(path.join(archiveChanged.source, "fetch.json"), "utf8"));
    const original = acquisition.archives.nvrtc.path; fs.chmodSync(original, 0o644);
    const fd = fs.openSync(original, "r+"); try {fs.writeSync(fd, Buffer.from("X"), 0, 1, 0);} finally {fs.closeSync(fd);}
    fs.chmodSync(original, 0o444);
    assert.throws(() => nvrtcSdkInputs("linux", archiveChanged.output, {hashes: archiveChanged.hashes}), /digest differs from its pin/);

    const receiptChanged = await assembledFixture(t, "windows");
    await prepareNvrtcSdk("windows", receiptChanged.source, receiptChanged.output, {hashes: receiptChanged.hashes});
    const sdkReceipt = path.join(receiptChanged.output, "sdk.json"), sdk = JSON.parse(fs.readFileSync(sdkReceipt, "utf8"));
    sdk.compiled = true; fs.chmodSync(sdkReceipt, 0o644); fs.writeFileSync(sdkReceipt, JSON.stringify(sdk) + "\n"); fs.chmodSync(sdkReceipt, 0o444);
    assert.throws(() => nvrtcSdkInputs("windows", receiptChanged.output, {hashes: receiptChanged.hashes}), /valid offline vendor-prebuilt/);

    const outputChanged = await assembledFixture(t, "linux");
    await prepareNvrtcSdk("linux", outputChanged.source, outputChanged.output, {hashes: outputChanged.hashes});
    const header = path.join(outputChanged.output, "include/nvrtc.h"); fs.chmodSync(header, 0o644); fs.writeFileSync(header, "tampered"); fs.chmodSync(header, 0o444);
    assert.throws(() => nvrtcSdkInputs("linux", outputChanged.output, {hashes: outputChanged.hashes}), /hash differs from receipt/);

    const extra = await assembledFixture(t, "windows");
    await prepareNvrtcSdk("windows", extra.source, extra.output, {hashes: extra.hashes});
    fs.chmodSync(extra.output, 0o755); fs.writeFileSync(path.join(extra.output, "unexpected"), "x"); fs.chmodSync(extra.output, 0o555);
    assert.throws(() => nvrtcSdkInputs("windows", extra.output, {hashes: extra.hashes}), /unexpected output/);
  });

  test("narrow extraction limits and existing outputs never overwrite data", async t => {
    const limited = await assembledFixture(t, "linux");
    await assert.rejects(prepareNvrtcSdk("linux", limited.source, limited.output, {hashes: limited.hashes, maxMemberBytes: 1}), /extraction limit/);
    assert.equal(fs.existsSync(path.join(limited.output, "sdk.json")), false);

    const totalLimited = await assembledFixture(t, "windows");
    await assert.rejects(prepareNvrtcSdk("windows", totalLimited.source, totalLimited.output,
      {hashes: totalLimited.hashes, maxTotalBytes: 1}), /extraction limit/);
    assert.equal(fs.existsSync(path.join(totalLimited.output, "sdk.json")), false);

    const f = await assembledFixture(t, "windows"); fs.mkdirSync(f.output); fs.writeFileSync(path.join(f.output, "keep"), "preserve");
    await assert.rejects(prepareNvrtcSdk("windows", f.source, f.output, {hashes: f.hashes}), /must be new/);
    assert.equal(fs.readFileSync(path.join(f.output, "keep"), "utf8"), "preserve");
    await assert.rejects(prepareNvrtcSdk("windows", f.source, f.output + "\n", {hashes: f.hashes}), /new absolute directory/);
  });
});
