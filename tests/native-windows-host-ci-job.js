"use strict";

const assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path"),
  {spawnSync} = require("node:child_process"), {test} = require("node:test");

const repo = path.resolve(__dirname, ".."), hostJob = require("../scripts/ci/windows-host"),
  ci = require("../scripts/native-ci-tar"), {fileHash} = require("../scripts/build-source-inputs");
const {clrRevision, sourceFiles: hiprtcSourceFiles} = require("../scripts/build-native-windows-hiprtc-sdk");

/** @param {string[]} args @param {NodeJS.ProcessEnv} [environment] */
function runGit(args, environment = {}) {
  const result = spawnSync("/usr/bin/git", args, {cwd: repo, env: {...process.env, ...environment}, encoding: "utf8", timeout: 10_000});
  assert.equal(result.status, 0, result.stderr);
  return (result.stdout || "").trim();
}

/** @param {import("node:test").TestContext} t @param {string} [parent] */
function tempDir(t, parent = os.tmpdir()) {
  const directory = fs.mkdtempSync(path.join(parent, "mom-native-windows-host-job-"));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  return directory;
}

/** Unique run attempt so fixed /tmp restore roots never collide between test runs. */
function uniqueRun() {
  const runId = String(Date.now()) + String(process.pid).padStart(6, "0");
  return {runId, attempt: "2", suffix: runId + "-2"};
}

/** @param {number} offsetSeconds */
function epoch(offsetSeconds) {return String(Math.floor(Date.now() / 1000) + offsetSeconds);}

/** @param {string} value */
function allPins(value) {return Object.fromEntries(Object.values(hostJob.inputPins).map(name => [name, value]));}

/** @param {string} root */
function hostComponentFixture(root) {
  const {archiveSha256} = require("../scripts/build-native-windows-llvm-sdk"),
    {revision: comgrRevision, hostSources} = require("../scripts/build-native-windows-comgr-sdk"),
    acppRevision = "da2463e45aa90aa36306c45abcfc05b87de51bc6",
    pins = {version: 3, targetTriple: "x86_64-pc-windows-msvc", linkOrder: ["acpp", "hiprtc", "comgr", "llvm"],
      llvm: {sourceArchiveSha256: archiveSha256, archives: 164},
      comgr: {sourceRevision: comgrRevision, archives: 1, objects: hostSources.length},
      acpp: {sourceRevision: acppRevision, archives: 9},
      hiprtc: {sourceRevision: clrRevision, comgrRevision, archiveName: "hiprtc-host-intel.lib", archives: 1, objects: hiprtcSourceFiles.length + 1}},
    groups = {acpp: 9, hiprtc: 1, comgr: 1, llvm: 164}, generationId = "123e4567-e89b-42d3-a456-426614174000",
    link = /** @type {Record<"acpp" | "hiprtc" | "comgr" | "llvm", string[]>} */ ({}),
    files = /** @type {Record<string, {sha256: string, bytes: number}>} */ ({});
  fs.mkdirSync(root);
  for (const [group, count] of /** @type {Array<[keyof typeof groups, number]>} */ (Object.entries(groups))) {
    fs.mkdirSync(path.join(root, "lib", group), {recursive: true});
    link[group] = [];
    for (let index = 0; index < count; index++) {
      const basename = group === "hiprtc" ? "hiprtc-host-intel.lib" : group === "comgr" ? "amd_comgr.lib" : group + "-" + String(index).padStart(3, "0") + ".lib",
        relative = "lib/" + group + "/" + String(index).padStart(3, "0") + "-" + basename,
        file = path.join(root, relative);
      fs.writeFileSync(file, "!<arch>\n");
      link[group].push(relative);
      files[relative] = {sha256: fileHash(file), bytes: fs.statSync(file).size};
    }
  }
  const hiprtcMembers = [...hiprtcSourceFiles.map(name => path.basename(name, ".cpp") + ".obj"), "hipRTC_header.obj"],
    comgrArchiveSha256 = files[link["comgr"]?.[0] || ""]?.sha256 || "e".repeat(64),
    resourceReceiptSha256 = "2".repeat(64);
  fs.mkdirSync(path.join(root, "runtime"));
  const runtimeInputs = /** @type {Record<string, {path: string, bytes: number, sha256: string}>} */ ({});
  for (const [name, basename] of Object.entries({cudartStatic: "cudart_static.lib", amdhip64: "amdhip64.lib"})) {
    const relative = "runtime/" + basename, file = path.join(root, relative);
    fs.writeFileSync(file, "!<arch>\n");
    const proof = {path: relative, bytes: fs.statSync(file).size, sha256: fileHash(file)};
    runtimeInputs[name] = proof;
    files[relative] = {sha256: proof.sha256, bytes: proof.bytes};
  }
  const comgrReceiptSha256 = "d".repeat(64),
    manifest = {version: 3, kind: "windows-native-host-sdk", pins, link, runtimeInputs, files,
      provenance: {llvm: {generationId, receiptSha256: "a".repeat(64), admissionSha256: "b".repeat(64), objectProofSha256: "c".repeat(64)},
        comgr: {receiptSha256: comgrReceiptSha256, archiveSha256: comgrArchiveSha256, llvmGenerationId: generationId},
        acpp: {receiptSha256: "f".repeat(64), sourceRevision: acppRevision, llvmGenerationId: generationId, rocmResourceReceiptSha256: resourceReceiptSha256},
        hiprtc: {receiptSha256: "1".repeat(64), archiveSha256: files[link.hiprtc[0] || ""]?.sha256 || "e".repeat(64), sourceRevision: clrRevision,
          comgrRevision, comgrReceiptSha256, comgrArchiveSha256, resourceReceiptSha256, archiveMembers: hiprtcMembers}},
      buildOnly: true, publicationQualified: false};
  fs.writeFileSync(path.join(root, "component.json"), JSON.stringify(manifest, null, 2) + "\n");
  return manifest;
}

test.describe("Windows static host-component CI", {skip: process.platform !== "linux" && "Linux-hosted Windows build job harness"}, () => {
  test("pins validates every external pin and the job budget before writing same-run artifact names", t => {
    const directory = tempDir(t), output = path.join(directory, "output"),
      environment = {GITHUB_OUTPUT: output, MOM_NATIVE_CI_SUFFIX: "123456-2", MOM_CI_DEADLINE_EPOCH: epoch(3300),
        ...allPins("a".repeat(64))};
    fs.writeFileSync(output, "");
    hostJob.validatePins(environment);
    const text = fs.readFileSync(output, "utf8");
    assert.equal(text.trim().split("\n").length, 10);
    assert.match(text, /^linux_sdk_archive=native-linux-llvm21-sdk-123456-2\.tar$/m);
    assert.match(text, /^windows_state_archive=native-windows-llvm-state-123456-2-audited\.tar$/m);
    assert.match(text, /^windows_state_manifest=native-windows-llvm-state-123456-2-audited\.json$/m);
    assert.match(text, /^hip_headers_archive=native-rocm-hip-headers-123456-2\.tar$/m);
    assert.throws(() => hostJob.validatePins({...environment, MOM_NATIVE_LINUX_DPCPP_CUDA_MANIFEST_SHA256: "f".repeat(63)}),
      /Missing or malformed DPCPP_CUDA_MANIFEST pin/);
    assert.throws(() => hostJob.validatePins({...environment, MOM_CI_DEADLINE_EPOCH: epoch(120)}), /Insufficient budget/);
    assert.throws(() => hostJob.validatePins({...environment, MOM_CI_DEADLINE_EPOCH: epoch(-1)}), /expired MOM_CI_DEADLINE_EPOCH/);
    assert.throws(() => hostJob.validatePins({...environment, MOM_CI_DEADLINE_EPOCH: "12345"}), /Invalid or expired/);
    assert.equal(fs.readFileSync(output, "utf8"), text, "rejected pins or deadlines must not publish artifact names");
  });

  test("the job deadline is the earlier of MOM_CI_DEADLINE_EPOCH and the original Windows LLVM run deadline", () => {
    const now = Math.floor(Date.now() / 1000), environment = {MOM_CI_DEADLINE_EPOCH: String(now + 3300)};
    assert.deepEqual(hostJob.jobDeadlines(environment, String(now + 7200), now), {runEpoch: now + 7200, jobEpoch: now + 3300});
    assert.deepEqual(hostJob.jobDeadlines(environment, String(now + 600), now), {runEpoch: now + 600, jobEpoch: now + 600});
    assert.throws(() => hostJob.jobDeadlines(environment, String(now), now), /expired or exceeds eight hours/);
    assert.throws(() => hostJob.jobDeadlines(environment, String(now + 8 * 3600 + 1), now), /expired or exceeds eight hours/);
    assert.throws(() => hostJob.jobDeadlines(environment, 1234567890, now), /Invalid original Windows LLVM run deadline/);
    assert.throws(() => hostJob.jobDeadlines({MOM_CI_DEADLINE_EPOCH: String(now)}, String(now + 600), now), /MOM_CI_DEADLINE_EPOCH/);
    const derived = hostJob.jobEnvironment("/runner/temp", "123456-2", {runEpoch: now + 7200, jobEpoch: now + 3300});
    assert.equal(derived.MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH, String(now + 7200));
    assert.equal(derived.MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH, String(now + 3300));
    assert.equal(derived.MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH, String(now + 3300));
    assert.equal(derived.MOM_NATIVE_WINDOWS_HOST_SDK_OUTPUT_DIR, "/runner/temp/mom-native-windows-host-sdk-123456-2/native-windows-host-sdk-123456-2");
    assert.equal(derived.MOM_NATIVE_WINDOWS_HOST_SDK_METADATA_TAR, "/runner/temp/native-windows-host-sdk-123456-2-metadata.tar");
    assert.equal(derived.MOM_NATIVE_WINDOWS_LLVM_DESCRIPTOR_DIR, "/runner/temp/mom-native-windows-llvm21-descriptor-123456-2");
  });

  test("job admission reads the run deadline only from the externally pinned flat Windows LLVM manifest", t => {
    const runnerTemp = tempDir(t), incoming = path.join(runnerTemp, "incoming"),
      manifestFile = path.join(incoming, "native-windows-llvm-state-123456-2-audited.json"), runDeadline = epoch(7200);
    fs.mkdirSync(incoming);
    fs.writeFileSync(manifestFile, JSON.stringify({version: 1, runDeadlineEpoch: runDeadline}) + "\n");
    const environment = {RUNNER_TEMP: runnerTemp, MOM_NATIVE_CI_SUFFIX: "123456-2", MOM_CI_DEADLINE_EPOCH: epoch(3300),
        ...allPins("a".repeat(64)), MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256: fileHash(manifestFile)},
      job = hostJob.admitHostJob(environment);
    assert.equal(job.deadlines.runEpoch, Number(runDeadline));
    assert.equal(job.deadlines.jobEpoch, Number(environment.MOM_CI_DEADLINE_EPOCH));
    assert.equal(job.environment.MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH, runDeadline);
    assert.equal(job.manifestSha256, environment.MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256);
    assert.throws(() => hostJob.admitHostJob({...environment, MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256: "f".repeat(64)}),
      /differs from its external SHA-256 pin/);
    fs.rmSync(manifestFile);
    fs.writeFileSync(manifestFile, JSON.stringify({version: 1, runDeadlineEpoch: epoch(-10)}) + "\n");
    assert.throws(() => hostJob.admitHostJob({...environment, MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256: fileHash(manifestFile)}),
      /expired or exceeds eight hours/);
  });

  test("the build command fails closed on a malformed external pin before any work or output", t => {
    const directory = tempDir(t), output = path.join(directory, "output"), summary = path.join(directory, "summary");
    fs.writeFileSync(output, "sentinel\n");
    fs.writeFileSync(summary, "summary\n");
    const environment = {...process.env, MOM_NATIVE_WINDOWS_LLVM_STATE_BUNDLE_SHA256: "not-a-digest", GITHUB_OUTPUT: output,
        GITHUB_STEP_SUMMARY: summary, MOM_NATIVE_CI_SUFFIX: "123456-2", RUNNER_TEMP: directory},
      result = spawnSync(process.execPath, [path.join(repo, "scripts/ci/windows-host.js"), "build"],
        {cwd: repo, env: environment, encoding: "utf8", timeout: 10_000});
    assert.equal(result.status, 1);
    assert.match(result.stderr, /^\[native-ci\] Missing or malformed WINDOWS_STATE_BUNDLE pin$/m);
    assert.equal(fs.readFileSync(output, "utf8"), "sentinel\n");
    assert.equal(fs.readFileSync(summary, "utf8"), "summary\n");
    assert.deepEqual(fs.readdirSync(directory).sort(), ["output", "summary"]);
  });

  test("the pinned manifest reader returns exactly the hashed bounded bytes", t => {
    const directory = tempDir(t), file = path.join(directory, "manifest.json"), bytes = Buffer.from(JSON.stringify({fixture: true}) + "\n");
    fs.writeFileSync(file, bytes, {flag: "wx"});
    const pin = fileHash(file), admitted = hostJob.pinnedManifest(file, pin, 1024);
    assert.ok(admitted.bytes.equals(bytes));
    assert.equal(admitted.sha256, pin);
    assert.throws(() => hostJob.pinnedManifest(file, "f".repeat(64), 1024), /differs from its external SHA-256 pin/);
    assert.throws(() => hostJob.pinnedManifest(file, pin, 1), /bounded canonical regular file/);
    assert.throws(() => hostJob.pinnedManifest(file, pin, 2 << 20), /pin or byte bound is invalid/);
    const alias = path.join(directory, "alias.json");
    fs.linkSync(file, alias);
    assert.throws(() => hostJob.pinnedManifest(file, pin, 1024), /bounded canonical regular file/);
  });

  test("restore unpacks Linux LLVM tools only after pin admission into their fresh configure-time root", t => {
    const {suffix} = uniqueRun(), runnerTemp = tempDir(t, "/var/tmp"),
      context = {suffix, runnerTemp, sdkName: "mom-native-llvm21-sdk-" + suffix, deadlineMs: Date.now() + 600_000},
      rootName = "native-linux-llvm-tools-" + suffix, sourceBase = path.join(runnerTemp, "fixture-input"),
      sourceRoot = path.join(sourceBase, rootName), archive = path.join(runnerTemp, rootName + ".tar"),
      toolsRoot = path.join(runnerTemp, "linux-llvm-tools-extracted-" + suffix), clangTool = path.join(sourceRoot, "bin", "clang-tblgen");
    fs.mkdirSync(path.dirname(clangTool), {recursive: true});
    fs.writeFileSync(clangTool, "fixture clang-tblgen\n");
    ci.packState(archive, sourceBase, [rootName]);
    const pin = fileHash(archive);
    assert.throws(() => hostJob.restore(context, archive, "f".repeat(64), [rootName], toolsRoot, 1024, 25 * 60 * 1000, true),
      /differs from its external SHA-256 pin/);
    assert.equal(fs.existsSync(toolsRoot), false, "a wrong pin must not create the extraction root");
    const wrongBase = path.join(runnerTemp, "alternate-linux-llvm-tools-root");
    assert.throws(() => hostJob.restore(context, archive, pin, [rootName], wrongBase, 1024, 25 * 60 * 1000, true), /fresh exact same-run extraction root/);
    assert.equal(fs.existsSync(wrongBase), false);
    const wrongRootName = "alternate-linux-llvm-tools-" + suffix, wrongArchive = path.join(runnerTemp, wrongRootName + ".tar");
    fs.mkdirSync(path.join(sourceBase, wrongRootName, "bin"), {recursive: true});
    fs.writeFileSync(path.join(sourceBase, wrongRootName, "bin", "clang-tblgen"), "wrong root\n");
    ci.packState(wrongArchive, sourceBase, [wrongRootName]);
    assert.throws(() => hostJob.restore(context, wrongArchive, fileHash(wrongArchive), [wrongRootName], toolsRoot, 1024, 25 * 60 * 1000, true),
      /fresh exact same-run extraction root/);
    assert.equal(fs.existsSync(toolsRoot), false);
    fs.mkdirSync(toolsRoot, {mode: 0o700});
    const sentinel = path.join(toolsRoot, "sentinel");
    fs.writeFileSync(sentinel, "preserve\n");
    assert.throws(() => hostJob.restore(context, archive, pin, [rootName], toolsRoot, 1024, 25 * 60 * 1000, true), /fresh exact/);
    assert.equal(fs.readFileSync(sentinel, "utf8"), "preserve\n");
    assert.equal(fs.existsSync(path.join(toolsRoot, rootName)), false);
    fs.rmSync(toolsRoot, {recursive: true, force: true});
    assert.throws(() => hostJob.restore(context, archive, pin, [rootName], toolsRoot, 4, 25 * 60 * 1000, true), /expanded-size bound/);
    const restored = hostJob.restore(context, archive, pin, [rootName], toolsRoot, 1024, 25 * 60 * 1000, true);
    assert.equal(restored.expandedBytes, Buffer.byteLength("fixture clang-tblgen\n"));
    assert.equal(fs.readFileSync(path.join(toolsRoot, rootName, "bin", "clang-tblgen"), "utf8"), "fixture clang-tblgen\n");
  });

  test("restore returns the HIP header transport to its exported absolute /tmp root", t => {
    const {suffix} = uniqueRun(), runnerTemp = tempDir(t, "/var/tmp"), hipName = "native-rocm-hip-headers-" + suffix,
      context = {suffix, runnerTemp, sdkName: "mom-native-llvm21-sdk-" + suffix, deadlineMs: Date.now() + 600_000},
      sourceBase = path.join(runnerTemp, "fixture-input"), archive = path.join(runnerTemp, hipName + ".tar");
    t.after(() => fs.rmSync(path.join("/tmp", hipName), {recursive: true, force: true}));
    fs.mkdirSync(path.join(sourceBase, hipName), {recursive: true});
    fs.writeFileSync(path.join(sourceBase, hipName, "marker"), "headers");
    ci.packState(archive, sourceBase, [hipName]);
    hostJob.restore(context, archive, fileHash(archive), [hipName], "/tmp", 128 * 1024 ** 2);
    assert.equal(fs.readFileSync(path.join("/tmp", hipName, "marker"), "utf8"), "headers");
    assert.throws(() => hostJob.restore({...context, deadlineMs: Date.now() + 30_000}, archive, fileHash(archive), [hipName], "/tmp", 1024),
      /job deadline exhausted/);
  });

  test("the verified bare CLR input yields the clean ordinary checkout required by HIPRTC admission", t => {
    const directory = tempDir(t), source = path.join(directory, "source"), bare = path.join(directory, "clr.git"),
      checkout = path.join(directory, "checkout"), deadlineMs = Date.now() + 600_000;
    fs.mkdirSync(source);
    runGit(["init", "--quiet", source]);
    runGit(["-C", source, "config", "user.name", "CI fixture"]);
    runGit(["-C", source, "config", "user.email", "ci@example.invalid"]);
    for (const root of ["hipamd", "rocclr"]) {
      fs.mkdirSync(path.join(source, root));
      fs.writeFileSync(path.join(source, root, "source.txt"), root + "\n");
    }
    runGit(["-C", source, "add", "--", "hipamd", "rocclr"]);
    runGit(["-C", source, "commit", "--quiet", "-m", "fixture"]);
    runGit(["clone", "--quiet", "--bare", "--", source, bare]);
    const revision = runGit(["-C", bare, "rev-parse", "HEAD"]);
    assert.equal(runGit(["-C", bare, "rev-parse", "--is-bare-repository"]), "true");
    assert.throws(() => hostJob.checkoutPinnedSource(deadlineMs, bare, path.join(directory, "missing"), "0".repeat(40)), /HIPRTC source preparation failed/);
    assert.equal(fs.existsSync(path.join(directory, "missing")), false);
    hostJob.checkoutPinnedSource(deadlineMs, bare, checkout, revision);
    assert.equal(runGit(["-C", checkout, "rev-parse", "HEAD"]), revision);
    assert.equal(runGit(["-C", checkout, "rev-parse", "--is-inside-work-tree"]), "true");
    assert.equal(runGit(["-C", checkout, "status", "--porcelain=v1", "--untracked-files=all", "--", "hipamd", "rocclr"]), "");
  });

  test("seal re-admits, packs and rechecks an exact host-component fixture as the only artifact", t => {
    const {suffix} = uniqueRun(), root = path.join("/tmp", "native-windows-host-component-" + suffix), name = path.basename(root),
      runnerTemp = tempDir(t, "/var/tmp"), output = path.join(runnerTemp, "output"), jobDeadlineEpoch = Number(epoch(600));
    assert.equal(fs.existsSync(root), false, "fixture root must be fresh");
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    hostComponentFixture(root);
    fs.writeFileSync(output, "");
    assert.throws(() => hostJob.sealHostComponent({suffix, runnerTemp, jobDeadlineEpoch: Number(epoch(20)), outputFile: output}),
      /inherited job deadline are invalid/);
    const sealed = hostJob.sealHostComponent({suffix, runnerTemp, jobDeadlineEpoch, outputFile: output}),
      archive = path.join(runnerTemp, name + ".tar"),
      rows = Object.fromEntries(fs.readFileSync(output, "utf8").trim().split(/\r?\n/).map(line => line.split("=")));
    assert.equal(sealed.archive, archive);
    assert.equal(rows["artifact_name"], name + ".tar");
    assert.equal(rows["bundle_sha256"], fileHash(archive));
    assert.equal(rows["manifest_sha256"], fileHash(path.join(root, "component.json")));
    const members = ci.listState(archive, [name]).map(member => member.replace(/\/$/, "")).sort();
    assert.ok(members.includes(name + "/component.json"));
    assert.ok(ci.tarExpandedBytes(archive, members.length) > 0);
    const before = fs.readFileSync(output, "utf8");
    assert.throws(() => hostJob.sealHostComponent({suffix, runnerTemp, jobDeadlineEpoch, outputFile: output}),
      /differs from its fresh build-only manifest before sealing/);
    assert.equal(fs.readFileSync(output, "utf8"), before, "a second seal must not overwrite or re-publish outputs");
  });
});
