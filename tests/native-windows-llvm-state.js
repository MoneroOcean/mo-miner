"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"),
  path = require("node:path"), {createHash} = require("node:crypto");
const ci = require("../scripts/prepare-native-ci"), {fileHash} = require("../scripts/build-source-inputs"),
  {archiveName} = require("../scripts/native-llvm-sdk"), {packWindowsNativeLlvmState, restoreWindowsNativeLlvmState} = require("../scripts/native-windows-llvm-state");

/** @param {unknown} value */
const manifestHash = value => createHash("sha256").update(JSON.stringify(value) + "\n").digest("hex");

/** @param {import('node:test').TestContext} t */
function fixture(t, {checkpoint = true} = {}) {
  const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-llvm-state-")), run = String(Date.now()) + String(process.pid), suffix = run + "-1",
    root = path.join("/tmp", "mom-native-windows-llvm21-" + suffix), outputFile = path.join(runnerTemp, "github-output"),
    summaryFile = path.join(runnerTemp, "github-summary"), archiveRoot = fs.mkdtempSync(path.join("/var/tmp", "mom-win-llvm-archive-")), now = Date.now(),
    epoch = (/** @type {number} */ minutes) => String(Math.ceil((now + minutes * 60 * 1000) / 1000)),
    environment = /** @type {NodeJS.ProcessEnv} */ ({RUNNER_TEMP: runnerTemp, GITHUB_RUN_ID: run, GITHUB_RUN_ATTEMPT: "1", GITHUB_JOB: "llvm_configure",
      MOM_NATIVE_WINDOWS_LLVM_DIR: root, MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: epoch(7 * 60),
      MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: epoch(55), MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2",
      GITHUB_OUTPUT: outputFile, GITHUB_STEP_SUMMARY: summaryFile}),
    state = ci.windowsNativeLlvmCiStatePaths(environment, "linux", "x64"), inputRoot = path.join(runnerTemp, "inputs"),
    tools = path.join(inputRoot, "tools"), sdk = path.join(inputRoot, "sdk"), include = path.join(sdk, "include"),
    mappedHeader = path.join(inputRoot, "windows.h"), overlay = path.join(inputRoot, "overlay.json"), archive = path.join(archiveRoot, archiveName),
    toolPaths = {clangCl: path.join(tools, "clang-cl"), lldLink: path.join(tools, "lld-link"), llvmNm: path.join(tools, "llvm-nm"),
      clangResourceInclude: path.join(tools, "resource")};
  fs.writeFileSync(outputFile, "", {flag: "wx"}); fs.writeFileSync(summaryFile, "", {flag: "wx"});
  fs.mkdirSync(state.root, {recursive: true, mode: 0o700});
  ci.windowsNativeLlvmCiDeadlineRecord(state, true);
  fs.mkdirSync(state.output, {recursive: true, mode: 0o700}); fs.mkdirSync(tools, {recursive: true});
  fs.mkdirSync(include, {recursive: true}); fs.mkdirSync(toolPaths.clangResourceInclude, {recursive: true});
  fs.writeFileSync(mappedHeader, "/* fixture */\n", {flag: "wx"});
  fs.writeFileSync(overlay, JSON.stringify({version: 0, "case-sensitive": false, "use-external-names": true,
    roots: [{type: "file", name: path.join(include, "windows.h"), "external-contents": mappedHeader}]}), {flag: "wx"});
  const descriptor = {version: 1, tools: toolPaths, windowsSdk: {overlay, sourceRoot: path.join(sdk, "source"),
    includeRoots: [include], libraryRoots: [path.join(sdk, "lib")], msvcLibraries: [path.join(sdk, "lib", "libcmt.lib")],
    ucrtLibraries: [path.join(sdk, "lib", "libucrt.lib")], importLibraries: [path.join(sdk, "lib", "kernel32.lib")]}};
  fs.writeFileSync(state.descriptor, JSON.stringify(descriptor) + "\n", {flag: "wx", mode: 0o444});
  const descriptorSha256 = fileHash(state.descriptor), plan = {output: state.output, descriptorFile: state.descriptor, archive,
    tools: {cmake: path.join(tools, "cmake"), ninja: path.join(tools, "ninja")}, jobs: 2};
  fs.writeFileSync(state.result, JSON.stringify({role: "windows-llvm21-host-sdk", stage: "configured", buildOnly: true, compiled: false,
    executed: false, publicationQualified: false, plan, fingerprints: {[state.descriptor]: descriptorSha256}}) + "\n", {flag: "wx", mode: 0o444});
  if (checkpoint) {
    fs.writeFileSync(path.join(state.output, "build-state.json"), JSON.stringify({version: 1, role: "fixture checkpoint", runId: suffix}) + "\n",
      {flag: "wx", mode: 0o444});
  }
  t.after(() => fs.rmSync(state.root, {recursive: true, force: true}));
  t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
  t.after(() => fs.rmSync(archiveRoot, {recursive: true, force: true}));
  return {runnerTemp, archiveRoot, environment, state, archive, now, epoch, descriptor, descriptorSha256, outputFile, summaryFile};
}

test.describe("Windows LLVM build-state transport", {skip: process.platform !== "linux"}, () => {
  test("restored state admission requires the external SDK overlay to be reconstructed first", t => {
    const f = fixture(t), overlay = f.descriptor.windowsSdk.overlay, bytes = fs.readFileSync(overlay),
      packed = packWindowsNativeLlvmState(f.archive, f.environment, f.now),
      receiver = {...f.environment, GITHUB_JOB: "llvm_build_amdgpu", MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: f.epoch(50)};
    fs.rmSync(f.state.root, {recursive: true}); fs.unlinkSync(overlay);
    assert.throws(() => restoreWindowsNativeLlvmState(f.archive, packed.manifest.archive.sha256,
      packed.manifest, packed.manifestSha256, receiver, f.now), /ENOENT.*overlay\.json/);
    fs.rmSync(f.state.root, {recursive: true}); fs.writeFileSync(overlay, bytes, {flag: "wx"});
    assert.doesNotThrow(() => restoreWindowsNativeLlvmState(f.archive, packed.manifest.archive.sha256,
      packed.manifest, packed.manifestSha256, receiver, f.now));
  });

  /** @param {ReturnType<typeof fixture>} f @param {string[]} [extraFiles] */
  const roundTrip = (f, extraFiles = []) => {
    const runDeadlineRecord = f.state.runDeadlineRecord;
    if (!runDeadlineRecord) {throw new Error("Fixture must use the cross-job run deadline contract");}
    const checkpoint = path.join(f.state.output, "build-state.json"), names = [f.state.descriptor, f.state.result,
        ...(fs.existsSync(checkpoint) ? [checkpoint] : []), runDeadlineRecord, f.state.deadlineRecord, ...extraFiles],
      before = new Map(names.map(file => [file, {sha256: fileHash(file), mtimeMs: fs.statSync(file).mtimeMs}])),
      packed = packWindowsNativeLlvmState(f.archive, f.environment, f.now), archiveSha256 = packed.manifest.archive.sha256;
    assert.equal(packed.manifest.runAttempt, f.state.suffix); assert.equal(packed.manifest.producerJob, "llvm_configure");
    assert.equal(packed.manifest.bindings.checkpointSha256, before.get(checkpoint)?.sha256 || null);
    assert.equal(packed.manifestSha256, manifestHash(packed.manifest));
    fs.rmSync(f.state.root, {recursive: true});
    const receiver = {...f.environment, GITHUB_JOB: "llvm_build_lld", MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: f.epoch(50)};
    const restored = restoreWindowsNativeLlvmState(f.archive, archiveSha256, packed.manifest, packed.manifestSha256, receiver, f.now);
    assert.equal(restored.state.root, f.state.root);
    for (const file of names) {
      assert.equal(fileHash(file), before.get(file)?.sha256);
      assert.equal(fs.statSync(file).mtimeMs, before.get(file)?.mtimeMs);
    }
    assert.ok(fs.existsSync(path.join(f.state.root, "ci-job-deadline-llvm_build_lld.json")), "receiving job gets its own immutable deadline record");
    assert.equal(fileHash(f.archive), archiveSha256);
    return {packed, receiver, archiveSha256};
  };

  test("round-trips configured state with and without a later build checkpoint", t => {
    const withCheckpoint = fixture(t), safeLink = path.join(withCheckpoint.state.output, "descriptor-link.json");
    fs.symlinkSync("../descriptor.json", safeLink);
    const result = roundTrip(withCheckpoint, [safeLink]);
    assert.ok(result.packed.manifest.bindings.checkpointSha256);
    assert.equal(fs.lstatSync(safeLink).isSymbolicLink(), true, "bounded relative link remains a link after transport");
    assert.equal(fs.readFileSync(safeLink, "utf8"), fs.readFileSync(withCheckpoint.state.descriptor, "utf8"));
    const configuredOnly = fixture(t, {checkpoint: false}), noCheckpoint = roundTrip(configuredOnly);
    assert.equal(noCheckpoint.packed.manifest.bindings.checkpointSha256, null,
      "a configure-only job is transportable before the first build phase creates a checkpoint");
  });

  test("resumes a later job after the producer job expired without extending the original run", t => {
    const f = fixture(t, {checkpoint: false}), packed = packWindowsNativeLlvmState(f.archive, f.environment, f.now),
      originalRun = fs.readFileSync(f.state.runDeadlineRecord || "");
    fs.rmSync(f.state.root, {recursive: true});
    t.mock.method(Date, "now", () => f.now + 70 * 60 * 1000);
    const receiver = {...f.environment, GITHUB_JOB: "llvm_next_job", MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: f.epoch(120)},
      restored = restoreWindowsNativeLlvmState(f.archive, packed.manifest.archive.sha256, packed.manifest,
        packed.manifestSha256, receiver);
    assert.deepEqual(fs.readFileSync(restored.state.runDeadlineRecord || ""), originalRun);
    assert.equal(JSON.parse(fs.readFileSync(restored.state.deadlineRecord, "utf8")).deadlineEpoch, f.epoch(120));
    const next = packWindowsNativeLlvmState(path.join(f.archiveRoot, "next-state.tar"), receiver);
    assert.equal(next.manifest.runDeadlineEpoch, packed.manifest.runDeadlineEpoch);
    assert.equal(next.manifest.bindings.jobDeadlines.length, 2);
  });

  test("rejects corrupt pins, stale identities, and extended or expired shared deadlines before extraction", t => {
    const f = fixture(t), packed = packWindowsNativeLlvmState(f.archive, f.environment, f.now), sha = packed.manifest.archive.sha256;
    fs.rmSync(f.state.root, {recursive: true});
    const receiver = {...f.environment, GITHUB_JOB: "llvm_build_lld", MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: f.epoch(50)};
    assert.throws(() => restoreWindowsNativeLlvmState(f.archive, "0".repeat(64), packed.manifest, packed.manifestSha256, receiver, f.now), /external same-run/);
    assert.throws(() => restoreWindowsNativeLlvmState(f.archive, sha, packed.manifest, "0".repeat(64), receiver, f.now), /external SHA-256 pin/);
    const wrongRun = /** @type {NodeJS.ProcessEnv} */ ({...receiver, GITHUB_RUN_ID: String(BigInt(f.state.suffix.split("-")[0] || "0") + 1n)});
    wrongRun["MOM_NATIVE_WINDOWS_LLVM_DIR"] = path.join("/tmp", "mom-native-windows-llvm21-" + (wrongRun["GITHUB_RUN_ID"] || "") + "-1");
    assert.throws(() => restoreWindowsNativeLlvmState(f.archive, sha, packed.manifest, packed.manifestSha256, wrongRun, f.now), /same-run archive/);
    const extendedRun = {...receiver, MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: f.epoch(7 * 60 + 1)};
    assert.throws(() => restoreWindowsNativeLlvmState(f.archive, sha, packed.manifest, packed.manifestSha256, extendedRun, f.now), /same-run archive/);
    const jobBeyondRun = {...receiver, MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: f.epoch(8 * 60)};
    assert.throws(() => restoreWindowsNativeLlvmState(f.archive, sha, packed.manifest, packed.manifestSha256, jobBeyondRun, f.now), /deadline|same-run archive/);
    const expired = {...receiver, MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: String(Math.floor(f.now / 1000) - 1)};
    assert.throws(() => restoreWindowsNativeLlvmState(f.archive, sha, packed.manifest, packed.manifestSha256, expired, f.now), /deadline is expired/);
    const tooLarge = {...packed.manifest, archive: {...packed.manifest.archive, bytes: 16 * 1024 ** 3 + 1}}, tooLargeHash = manifestHash(tooLarge);
    assert.throws(() => restoreWindowsNativeLlvmState(f.archive, sha, tooLarge, tooLargeHash, receiver, f.now), /manifest is invalid/);
    const tooExpanded = {...packed.manifest, archive: {...packed.manifest.archive, expandedBytes: 64 * 1024 ** 3 + 1}},
      tooExpandedHash = manifestHash(tooExpanded);
    assert.throws(() => restoreWindowsNativeLlvmState(f.archive, sha, tooExpanded, tooExpandedHash, receiver, f.now), /manifest is invalid/);
    assert.equal(fs.existsSync(f.state.root), false);
  });

  test("rejects an existing destination, aliased archive, and archive byte drift", t => {
    const f = fixture(t), packed = packWindowsNativeLlvmState(f.archive, f.environment, f.now), sha = packed.manifest.archive.sha256,
      receiver = {...f.environment, GITHUB_JOB: "llvm_build_lld", MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: f.epoch(50)};
    assert.throws(() => restoreWindowsNativeLlvmState(f.archive, sha, packed.manifest, packed.manifestSha256, receiver, f.now), /fresh canonical/);
    fs.rmSync(f.state.root, {recursive: true});
    const hardlink = path.join(f.archiveRoot, "linked.tar"); fs.linkSync(f.archive, hardlink);
    assert.throws(() => restoreWindowsNativeLlvmState(hardlink, sha, packed.manifest, packed.manifestSha256, receiver, f.now), /singly-linked/);
    const tampered = path.join(f.archiveRoot, "tampered.tar"); fs.copyFileSync(f.archive, tampered); fs.appendFileSync(tampered, "tamper");
    assert.throws(() => restoreWindowsNativeLlvmState(tampered, sha, packed.manifest, packed.manifestSha256, receiver, f.now), /external SHA-256 and size pins/);
    assert.equal(fileHash(f.archive), sha); assert.equal(fs.existsSync(f.state.root), false);
  });

  test("rejects absolute and escaping symlinks without retaining an archive", t => {
    const cases = /** @type {Array<[string, "absolute" | "relative"]>} */ ([
      ["absolute-link", "absolute"], ["relative-escape", "relative"],
    ]);
    for (const [name, target] of cases) {
      const f = fixture(t), link = path.join(f.state.output, name), linkTarget = target === "absolute"
        ? f.outputFile : path.relative(f.state.output, f.outputFile);
      fs.symlinkSync(linkTarget, link);
      assert.throws(() => packWindowsNativeLlvmState(f.archive, f.environment, f.now), /symlink|symbolic|escape|outside|contained|link/i);
      assert.equal(fs.existsSync(f.archive), false, "rejected external links do not leave a partial archive");
      fs.rmSync(f.state.root, {recursive: true, force: true});
      fs.rmSync(f.runnerTemp, {recursive: true, force: true});
      fs.rmSync(f.archiveRoot, {recursive: true, force: true});
    }
  });
});
