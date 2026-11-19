"use strict";
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), assert = require("node:assert/strict"),
  test = require("node:test"), {createHash} = require("node:crypto"), {spawnSync} = require("node:child_process"),
  {isDeepStrictEqual} = require("node:util");
const {fileHash} = require("../scripts/build-source-inputs"), realCi = require("../scripts/native-ci-tar");
const {packagePlan, preflightPackage, packageNative} = require("../scripts/ci/package-native");

const suffix = "ci-123456789-1", digest = (/** @type {string|Buffer} */ value) => createHash("sha256").update(value).digest("hex");

/** Build the late packaging job's same-run inputs with no candidate execution: real raw/original/packed
 * tars in RUNNER_TEMP/incoming, both executables in their own download directories, a fixture
 * workspace with reviewed notices, and a private state root standing in for /tmp.
 * @param {import('node:test').TestContext} t @param {'linux'|'win32'} [target] */
function fixture(t, target = "linux") {
  const temp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-native-package-ci-")),
    platform = target === "win32" ? "windows" : "linux", binary = target === "win32" ? "mom.exe" : "mom",
    roots = {packed: `native-${platform}-packed-build-${suffix}`, original: `native-${platform}-application-build-${suffix}`},
    stateRoot = path.join(temp, "state"), staging = path.join(temp, "staging"), incoming = path.join(temp, "incoming"),
    tree = {packed: path.join(staging, roots.packed), original: path.join(staging, roots.original)},
    cmakeDirectory = path.join(temp, "cmake-bin"), cmakePath = path.join(cmakeDirectory, "cmake"),
    repoRoot = path.join(temp, "repo"), noticeBody = path.join(repoRoot, "THIRD-PARTY-NOTICES.txt"),
    noticeManifest = path.join(repoRoot, "runtime/native-notice-review.json"),
    inputDirs = {raw: path.join(temp, "native-package-inputs/raw"), packed: path.join(temp, "native-package-inputs/packed-executable")},
    calls = /** @type {Array<{stage:string, [key:string]:any}>} */ ([]);
  let expandedOverride = /** @type {number|undefined} */ (undefined), apiOverride = /** @type {string|undefined} */ (undefined);
  t.after(() => fs.rmSync(temp, {recursive: true, force: true}));
  for (const directory of [...Object.values(inputDirs), ...Object.values(tree), stateRoot, incoming, cmakeDirectory, path.dirname(noticeManifest)]) {
    fs.mkdirSync(directory, {recursive: true});
  }
  fs.writeFileSync(cmakePath, "#!/bin/sh\nprintf 'cmake version 3.99.0\\n'\n", {mode: 0o755});
  fs.writeFileSync(path.join(repoRoot, "package.json"), JSON.stringify({version: "1.2.3"}));
  const noticeBytes = Buffer.from("Fixture reviewed notices\n"), noticeDependencies = {fixture: {revision: "a".repeat(40)}},
    noticeSnapshot = {version: 1, notice: {bytes: noticeBytes.length, sha256: digest(noticeBytes)}, dependencies: noticeDependencies};
  fs.writeFileSync(noticeBody, noticeBytes);
  fs.writeFileSync(noticeManifest, JSON.stringify(noticeSnapshot, null, 2) + "\n");
  const pinnedNoticeHash = fileHash(noticeBody), pinnedManifestHash = fileHash(noticeManifest);
  fs.writeFileSync(path.join(temp, "outputs"), "");

  const rawBytes = Buffer.alloc(96, target === "linux" ? 0x41 : 0x42); rawBytes.write(target === "linux" ? "\x7fELF" : "MZ");
  const packedBytes = Buffer.concat([Buffer.from(target === "linux" ? "\x7fELF" : "MZ"), Buffer.alloc(92, 0x50)]),
    rawPath = path.join(inputDirs.raw, binary), rawSha = digest(rawBytes), packedSha = digest(packedBytes);
  fs.writeFileSync(rawPath, rawBytes, {mode: 0o755}); fs.writeFileSync(path.join(inputDirs.packed, binary), packedBytes, {mode: 0o755});
  const writeJson = (/** @type {string} */ file, /** @type {unknown} */ value) => fs.writeFileSync(file, JSON.stringify(value, null, 2) + "\n");
  const runTempConfig = path.join(temp, target === "linux" ? `native-linux-assembly-config-${suffix}.json` : `native-windows-application-config-${suffix}.json`),
    configBytes = JSON.stringify({version: 1, buildOnly: true}), configHash = digest(configBytes);

  const originalSourcePath = path.join(target === "linux" ? "/tmp" : temp,
      target === "linux" ? `mom-native-linux-application-${suffix}` : `native-windows-application-output-${suffix}`, binary),
    originalProducerPath = path.join(path.dirname(originalSourcePath), "result.json");
  fs.writeFileSync(path.join(tree.original, "application-config.json"), configBytes);
  writeJson(path.join(tree.original, "result.json"), {binary: originalSourcePath, bytes: rawBytes.length, sha256: rawSha, configHash,
    buildOnly: true, publicationQualified: false});
  const producerHash = fileHash(path.join(tree.original, "result.json"));
  writeJson(path.join(tree.original, "entrypoint.json"), {version: 1, target, configFile: runTempConfig, configHash,
    source: {path: originalSourcePath, bytes: rawBytes.length, sha256: rawSha},
    artifact: {path: originalSourcePath, bytes: rawBytes.length, sha256: rawSha}, recipe: {},
    componentHashes: {[originalProducerPath]: producerHash}, producerReceipt: {path: originalProducerPath, sha256: producerHash},
    buildOnly: true, publicationQualified: false});
  const originEntryHash = fileHash(path.join(tree.original, "entrypoint.json"));
  writeJson(path.join(tree.original, "handoff.json"), {version: 1, format: `mom-${platform}-application-build-handoff-v1`, runAttempt: suffix,
    target: target === "win32" ? "x86_64-pc-windows-msvc" : "x86_64-unknown-linux-gnu",
    artifact: {file: binary, bytes: rawBytes.length, sha256: rawSha}, config: {file: "application-config.json", sha256: configHash},
    producerReceipt: {file: "result.json", sha256: producerHash}, entrypointReceipt: {file: "entrypoint.json", sha256: originEntryHash},
    buildOnly: true, executed: false, publicationQualified: false});

  const restoredPacked = path.join(stateRoot, roots.packed), packLogs = /** @type {Record<string, {path:string, sha256:string}>} */ ({});
  fs.writeFileSync(path.join(tree.packed, binary), packedBytes, {mode: 0o755});
  for (const stage of ["pack", "integrity", "restore"]) {
    const file = path.join(tree.packed, stage + ".log"); fs.writeFileSync(file, stage + " log\n");
    packLogs[stage] = {path: path.join(restoredPacked, stage + ".log"), sha256: fileHash(file)};
  }
  const rawRestore = path.join(temp, "native-raw", binary), candidateProducerPath = path.join(temp, "native-raw", "result.json");
  writeJson(path.join(tree.packed, "result.json"), {source: {path: rawRestore, bytes: rawBytes.length, sha256: rawSha},
    artifact: {path: path.join(restoredPacked, binary), bytes: packedBytes.length, sha256: packedSha},
    buildOnly: true, publicationQualified: false, logs: packLogs});
  writeJson(path.join(tree.packed, "entrypoint.json"), {version: 1, target, configFile: runTempConfig, configHash, recipe: {},
    componentHashes: {[candidateProducerPath]: producerHash},
    source: {path: rawRestore, bytes: rawBytes.length, sha256: rawSha},
    artifact: {path: path.join(restoredPacked, binary), bytes: packedBytes.length, sha256: packedSha},
    sourceEntrypoint: {path: path.join(stateRoot, roots.original, "entrypoint.json"), sha256: originEntryHash},
    producerReceipt: {path: candidateProducerPath, sha256: producerHash},
    packingReceipt: {path: path.join(restoredPacked, "result.json"), sha256: fileHash(path.join(tree.packed, "result.json"))},
    buildOnly: true, publicationQualified: false});

  /** @type {NodeJS.ProcessEnv} */
  const env = {RUNNER_TEMP: temp, MOM_NATIVE_CI_SUFFIX: "ci-123456789-1", NATIVE_PLATFORM: target,
    RAW_EXECUTABLE_SHA256: rawSha, GITHUB_WORKSPACE: repoRoot, MOM_CMAKE_PATH: cmakePath,
    PATH: [cmakeDirectory, process.env["PATH"] || "/usr/bin:/bin"].join(path.delimiter), GITHUB_OUTPUT: path.join(temp, "outputs")};
  const tarFiles = {packed: path.join(incoming, roots.packed + ".tar"), original: path.join(incoming, roots.original + ".tar")};
  // Seal both staged trees into real same-run tars and pin them like the producer job outputs.
  const seal = () => {
    for (const kind of /** @type {Array<'packed'|'original'>} */ (["packed", "original"])) {
      fs.rmSync(tarFiles[kind], {force: true}); realCi.packState(tarFiles[kind], staging, [roots[kind]], 60000);
    }
    env["PACKED_BUILD_BUNDLE_SHA256"] = fileHash(tarFiles.packed); env["ORIGINAL_BUILD_BUNDLE_SHA256"] = fileHash(tarFiles.original);
    env["ORIGINAL_HANDOFF_SHA256"] = fileHash(path.join(tree.original, "handoff.json"));
  };
  seal();

  const io = {stateRoot, workspace: repoRoot,
    ci: {listState: realCi.listState, packState: realCi.packState,
      tarExpandedBytes: (/** @type {Parameters<typeof realCi.tarExpandedBytes>} */ ...args) => expandedOverride ?? realCi.tarExpandedBytes(...args),
      extractState: (/** @type {Parameters<typeof realCi.extractState>} */ ...args) => {calls.push({stage: "extract"}); realCi.extractState(...args);}},
    checkNativeNoticeReview: (/** @type {string|undefined} */ manifestPath, /** @type {string|undefined} */ noticePath) => {
      calls.push({stage: "notice-review", manifestPath, noticePath});
      const selectedNotice = noticePath || noticeBody,
        selected = {version: 1, notice: {bytes: fs.statSync(selectedNotice).size, sha256: fileHash(selectedNotice)}, dependencies: noticeDependencies};
      if (manifestPath !== undefined || selectedNotice !== noticeBody || fileHash(noticeManifest) !== pinnedManifestHash ||
          selected.notice.sha256 !== pinnedNoticeHash || !isDeepStrictEqual(JSON.parse(fs.readFileSync(noticeManifest, "utf8")), noticeSnapshot)) {
        throw new Error("fixture native notice review drift");
      }
      return selected;
    },
    auditNativeRelease: /** @type {any} */ ((/** @type {string} */ entrypoint, /** @type {any} */ options) => {
      calls.push({stage: options?.archive ? "archive-audit" : "build-audit", entrypoint, options});
      if (apiOverride === "pre-audit" && !options?.archive) {throw new Error("fixture pre-audit rejected");}
      if (apiOverride === "post-audit" && options?.archive) {throw new Error("fixture archive audit rejected");}
      const answer = {scope: "build-audit", target, sourceSha256: rawSha, sha256: packedSha, bytes: packedBytes.length,
        buildEvidencePassed: true, runtimeTested: false, publicationQualified: false};
      return options?.archive ? {...answer, archiveChecked: true, archiveSha256: fileHash(options.archive.path)} : answer;
    }),
    packageNativeRelease: /** @type {any} */ ((/** @type {string} */ entrypoint, /** @type {string} */ tool,
      /** @type {string} */ version, /** @type {unknown} */ spawn, /** @type {any} */ lateNotice) => {
      calls.push({stage: "package", entrypoint, tool, version, spawn, lateNotice});
      fs.mkdirSync(lateNotice.output); fs.copyFileSync(lateNotice.notices, path.join(lateNotice.output, "THIRD-PARTY-NOTICES.txt"));
      if (apiOverride === "mutate-notice-body") {fs.appendFileSync(noticeBody, "changed during package\n");}
      if (apiOverride === "mutate-notice-manifest") {fs.appendFileSync(noticeManifest, " ");}
      const archivePath = path.join(lateNotice.output, `mom-v${version}-${target === "win32" ? "win.zip" : "lin.tgz"}`);
      fs.writeFileSync(archivePath, "fixture archive"); fs.writeFileSync(path.join(lateNotice.output, "native-package.json"), "fixture package receipt\n");
      return {archive: {path: archivePath, bytes: fs.statSync(archivePath).size, sha256: fileHash(archivePath)},
        executable: {path: path.join(restoredPacked, binary), bytes: packedBytes.length, sha256: packedSha}, buildOnly: true,
        publicationQualified: false, members: apiOverride === "extra-members" ? [binary, "THIRD-PARTY-NOTICES.txt", "extra"] : [binary, "THIRD-PARTY-NOTICES.txt"]};
    })};
  return {env, calls, temp, roots, tree, inputDirs, incoming, rawPath, noticeBody, noticeManifest, cmakeDirectory, cmakePath, stateRoot, seal,
    output: () => fs.readFileSync(path.join(temp, "outputs"), "utf8"),
    preflight: () => preflightPackage(env, io), package: () => packageNative(env, io),
    setExpanded: (/** @type {number} */ value) => {expandedOverride = value;}, setApi: (/** @type {string} */ value) => {apiOverride = value;}};
}

/** @param {ReturnType<typeof fixture>} f @param {string[]} stages */
const reached = (f, stages) => f.calls.some(row => stages.includes(row.stage));

test.describe("SDK-free late native package job", {skip: process.platform !== "linux" && "Linux tar and state paths"}, () => {
  test("resolves canonical PATH CMake before downloads and reuses it for package and archive audit", t => {
    const f = fixture(t), realTool = path.join(f.cmakeDirectory, "cmake-real");
    fs.renameSync(f.cmakePath, realTool); fs.symlinkSync(realTool, f.cmakePath);
    const resolved = f.preflight();
    assert.equal(resolved, realTool); assert.equal(f.output(), `path=${realTool}\n`);
    f.env["MOM_CMAKE_PATH"] = resolved; f.package();
    assert.equal(f.calls.find(row => row.stage === "package")?.["tool"], realTool);
    assert.equal(f.calls.find(row => row.stage === "archive-audit")?.["options"]?.archive?.tool, realTool);
  });
  test("missing, failing or malformed PATH CMake fails before artifact access", t => {
    /** @type {Array<[string, string, RegExp]>} */
    const cases = [["empty", "", /CMake is required/], ["failing", "#!/bin/sh\nexit 3\n", /failed its version check/],
      ["malformed", "#!/bin/sh\nprintf 'cmake version 3.bad\\n'\n", /invalid version banner/]];
    for (const [name, script, expected] of cases) {
      const f = fixture(t), bin = path.join(f.temp, name + "-bin"); fs.mkdirSync(bin);
      if (script) {fs.writeFileSync(path.join(bin, "cmake"), script, {mode: 0o755});}
      f.env["PATH"] = script ? [bin, process.env["PATH"] || "/usr/bin:/bin"].join(path.delimiter) : bin;
      assert.throws(() => f.preflight(), expected); assert.equal(f.output(), "");
      assert.deepEqual(f.calls.map(row => row.stage), ["notice-review"]);
    }
  });
  test("package CMake guard has no missing-path or symlink bypass", t => {
    const missing = fixture(t); Reflect.deleteProperty(missing.env, "MOM_CMAKE_PATH");
    assert.throws(() => missing.package(), /Resolved CMake path is missing/);
    assert.equal(reached(missing, ["extract", "package", "archive-audit"]), false);
    const linked = fixture(t), linkPath = path.join(linked.cmakeDirectory, "cmake-link");
    fs.symlinkSync(linked.cmakePath, linkPath); linked.env["MOM_CMAKE_PATH"] = linkPath;
    assert.throws(() => linked.package(), /Resolved CMake path is missing/);
    assert.equal(reached(linked, ["extract", "package", "archive-audit"]), false);
  });
  for (const target of ["linux", "win32"]) {
    test(`${target} restores the exact pinned provenance and uses only the existing package/audit APIs`, t => {
      const f = fixture(t, /** @type {'linux'|'win32'} */ (target)); f.preflight(); f.package();
      const stages = f.calls.map(row => row.stage), platform = target === "win32" ? "windows" : "linux";
      assert.deepEqual(stages.filter(stage => stage === "extract"), ["extract", "extract"]);
      assert.deepEqual(stages.filter(stage => stage.endsWith("audit")), ["build-audit", "archive-audit"]);
      const packaged = f.calls.find(row => row.stage === "package"); assert.ok(packaged);
      assert.equal(packaged["entrypoint"], path.join(f.stateRoot, f.roots.packed, "entrypoint.json"));
      assert.equal(packaged["tool"], f.cmakePath); assert.equal(packaged["spawn"], undefined); assert.equal(packaged["version"], "1.2.3");
      assert.equal(f.calls.find(row => row.stage === "archive-audit")?.["options"]?.archive?.tool, f.cmakePath);
      assert.deepEqual(packaged["lateNotice"], {output: path.join(f.temp, `native-${platform}-package-output-${suffix}`), notices: f.noticeBody});
      const reviews = f.calls.filter(row => row.stage === "notice-review");
      assert.deepEqual(reviews.map(row => [row["manifestPath"], row["noticePath"]]), [[undefined, undefined], [undefined, f.noticeBody]]);
      const metadataName = `native-${platform}-package-metadata-${suffix}`, metadataTar = path.join(f.temp, metadataName + ".tar"),
        archiveFile = path.join(packaged["lateNotice"].output, `mom-v1.2.3-${target === "win32" ? "win.zip" : "lin.tgz"}`);
      const reviewAudit = JSON.parse(fs.readFileSync(path.join(packaged["lateNotice"].output, metadataName, "native-notice-review.json"), "utf8"));
      assert.deepEqual(realCi.listState(metadataTar, [metadataName], 60000).map(name => name.replace(/\/$/, "")).sort(),
        [metadataName, ...["build-audit.json", "native-package.json", "native-notice-review.json"].map(name => metadataName + "/" + name)].sort());
      assert.equal(reviewAudit.manifest.sha256, fileHash(f.noticeManifest));
      assert.deepEqual(reviewAudit.snapshot, {version: 1, notice: {bytes: fs.statSync(f.noticeBody).size, sha256: fileHash(f.noticeBody)},
        dependencies: {fixture: {revision: "a".repeat(40)}}});
      assert.equal(f.output(), `path=${f.cmakePath}\narchive_artifact_name=native-${platform}-application-package-${suffix}\n` +
        `archive_sha256=${fileHash(archiveFile)}\narchive_file=${archiveFile}\n` +
        `metadata_artifact_name=${metadataName}.tar\nmetadata_file=${metadataTar}\n`);
    });
  }
  test("rejects malformed platform, CI suffix, workspace and missing external pins before artifact access", t => {
    for (const [key, value] of [["NATIVE_PLATFORM", "darwin"], ["NATIVE_PLATFORM", "windows"], ["MOM_NATIVE_CI_SUFFIX", "../x"],
      ["ORIGINAL_HANDOFF_SHA256", ""], ["PACKED_BUILD_BUNDLE_SHA256", "0".repeat(63)], ["GITHUB_WORKSPACE", "relative"]]) {
      const f = fixture(t, "win32"); f.env[key || ""] = value;
      assert.throws(() => f.preflight(), /exact platform, workspace and build SHA-256 pins/);
      assert.throws(() => f.package(), /Late native package paths, pins, platform or run identity/);
      assert.equal(f.calls.length, 0);
    }
    assert.equal(packagePlan(fixture(t, "win32").env)?.platform, "windows");
  });
  test("rejects another CI suffix and stray incoming files before unpacking", t => {
    const f = fixture(t); f.env["MOM_NATIVE_CI_SUFFIX"] = "other";
    assert.throws(() => f.package(), /incoming build bundle contents/); assert.equal(reached(f, ["extract"]), false);
    const g = fixture(t); fs.writeFileSync(path.join(g.incoming, "extra.tar"), "x");
    assert.throws(() => g.package(), /incoming build bundle contents/); assert.equal(reached(g, ["extract"]), false);
  });
  test("rejects an unexpected raw download member and raw byte tampering", t => {
    const f = fixture(t); fs.writeFileSync(path.join(f.inputDirs.raw, "extra"), "not raw");
    assert.throws(() => f.package(), /raw executable artifact contents/);
    const g = fixture(t); fs.appendFileSync(g.rawPath, "changed");
    assert.throws(() => g.package(), /raw executable/); assert.equal(reached(g, ["extract"]), false);
  });
  test("rejects extra packed-build/original members and oversized expansion before extraction", t => {
    for (const kind of /** @type {Array<'packed'|'original'>} */ (["packed", "original"])) {
      const f = fixture(t); fs.writeFileSync(path.join(f.tree[kind], "unexpected"), "x"); f.seal();
      assert.throws(() => f.package(), /missing or unexpected members/); assert.equal(reached(f, ["extract"]), false);
    }
    const h = fixture(t); h.setExpanded(129 << 20);
    assert.throws(() => h.package(), /bounded expansion limit/); assert.equal(reached(h, ["extract"]), false);
  });
  test("rejects a changed bundle against its producer pin and a false original-to-packed source chain", t => {
    const f = fixture(t), pinned = f.env["PACKED_BUILD_BUNDLE_SHA256"];
    fs.writeFileSync(path.join(f.tree.packed, "pack.log"), "other log\n"); f.seal(); f.env["PACKED_BUILD_BUNDLE_SHA256"] = pinned;
    assert.throws(() => f.package(), /packed-build bundle/); assert.equal(reached(f, ["extract"]), false);
    const g = fixture(t), entry = path.join(g.tree.packed, "entrypoint.json"), candidate = JSON.parse(fs.readFileSync(entry, "utf8"));
    candidate.sourceEntrypoint.sha256 = "0".repeat(64); fs.writeFileSync(entry, JSON.stringify(candidate, null, 2) + "\n"); g.seal();
    assert.throws(() => g.package(), /Packed artifact, original handoff, raw executable or retained packing evidence/);
  });
  test("pre-package audit failure blocks packaging and archive upload outputs", t => {
    const f = fixture(t); f.setApi("pre-audit"); assert.throws(() => f.package(), /fixture pre-audit rejected/);
    assert.equal(reached(f, ["package"]), false); assert.equal(f.output(), "");
  });
  test("package result cannot broaden its member set or skip archive audit", t => {
    const f = fixture(t); f.setApi("extra-members"); assert.throws(() => f.package(), /different bytes or members/);
    assert.equal(reached(f, ["archive-audit"]), false);
    const g = fixture(t); g.setApi("post-audit"); assert.throws(() => g.package(), /fixture archive audit rejected/);
    assert.equal(g.output(), "");
  });
  test("notice review drift blocks before downloads and again at the tracked body path", t => {
    const f = fixture(t); fs.appendFileSync(f.noticeManifest, " ");
    assert.throws(() => f.preflight(), /fixture native notice review drift/);
    assert.deepEqual(f.calls.map(row => row.stage), ["notice-review"]);
    const g = fixture(t); g.preflight(); fs.appendFileSync(g.noticeBody, "changed after preflight\n");
    assert.throws(() => g.package(), /fixture native notice review drift/);
    assert.equal(reached(g, ["package", "extract"]), false);
  });
  test("body and review manifest mutation during package generation blocks archive admission", t => {
    for (const mutation of ["mutate-notice-body", "mutate-notice-manifest"]) {
      const f = fixture(t); f.setApi(mutation);
      assert.throws(() => f.package(), /changed during packaging/);
      assert.equal(reached(f, ["archive-audit"]), false); assert.equal(f.output(), "");
    }
  });
  test("the CLI reports failures with the native-ci prefix", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/ci/package-native.js"), "unknown"],
      {encoding: "utf8", env: {PATH: process.env["PATH"] || ""}});
    assert.equal(result.status, 1); assert.match(result.stderr, /^\[native-ci\] Usage: /);
  });
});
