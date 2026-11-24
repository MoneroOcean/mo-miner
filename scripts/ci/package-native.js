"use strict";

// Late native packaging job: re-admit the same-run raw executable, packed executable, packed build
// bundle and original build handoff, then package and audit through the existing release APIs.
// The candidate is never executed and nothing is rebuilt.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const ci = require("../native-ci-tar"), {hasExactBuildFields, hasBuildDirectoryEntries, fileHash, readBuildJson} = require("../build-source-inputs");
const {applicationBuildFiles, releaseArtifactNames, sha256Pin, runnerPath, admitReleaseFile} = require("../native-ci-job");
const {auditNativeRelease} = require("../check-native-release"), {packageNativeRelease} = require("../package-native-release");
const {checkNativeNoticeReview} = require("../check-native-notices");

const hashPattern = /^[a-f0-9]{64}$/;
const packedFiles = ["entrypoint.json", "result.json", "pack.log", "integrity.log", "restore.log"];

/** @typedef {{version: number, notice: {bytes: number, sha256: string}, dependencies: unknown}} NoticeReview */
/** @typedef {{stateRoot: string, workspace: string, ci: Pick<typeof ci, "listState" | "extractState" | "packState" | "tarExpandedBytes">,
 *   checkNativeNoticeReview: (manifestPath?: string, noticePath?: string) => NoticeReview,
 *   auditNativeRelease: typeof auditNativeRelease, packageNativeRelease: typeof packageNativeRelease}} PackageIo */

/** Derive the exact platform, names and producer pins; nothing here touches artifacts.
 * @param {NodeJS.ProcessEnv} env */
function packagePlan(env) {
  const artifacts = releaseArtifactNames(env),
    packedBundleSha256 = sha256Pin(env, "PACKED_BUILD_BUNDLE_SHA256"), rawSha256 = sha256Pin(env, "RAW_EXECUTABLE_SHA256"),
    originalBundleSha256 = sha256Pin(env, "ORIGINAL_BUILD_BUNDLE_SHA256"), originalHandoffSha256 = sha256Pin(env, "ORIGINAL_HANDOFF_SHA256");
  if (!artifacts ||
      packedBundleSha256 === null || rawSha256 === null || originalBundleSha256 === null || originalHandoffSha256 === null) {
    return null;
  }
  return {...artifacts, packedBundleSha256, rawSha256, originalBundleSha256, originalHandoffSha256};
}

/** Resolve CMake from PATH the way `command -v` does: the first executable regular file named cmake.
 * @param {string} searchPath */
function findCmakeOnPath(searchPath) {
  for (const directory of searchPath.split(path.delimiter)) {
    const candidate = path.join(directory || ".", "cmake");
    try {
      if (!fs.statSync(candidate).isFile()) {continue;}
      fs.accessSync(candidate, fs.constants.X_OK);
      return candidate;
    } catch {continue;}
  }
  return null;
}

/** Resolve one canonical CMake executable from PATH and require a well-formed version banner.
 * @param {NodeJS.ProcessEnv} env */
function resolveCmake(env) {
  const found = findCmakeOnPath(env["PATH"] ?? "");
  if (found === null) {throw new Error("CMake is required for native package creation and archive audit.");}
  let cmake;
  try {cmake = fs.realpathSync(found);} catch {throw new Error("CMake on PATH could not be resolved to a canonical file.");}
  if (!canonicalExecutable(cmake)) {throw new Error("CMake on PATH is not a canonical absolute executable file.");}
  const result = spawnSync(cmake, ["--version"], {encoding: "utf8", timeout: 60000, maxBuffer: 1 << 20,
    stdio: ["ignore", "pipe", "pipe"]});
  if (result.error || result.signal || result.status !== 0) {throw new Error("CMake on PATH failed its version check.");}
  const banner = (result.stdout + result.stderr).replace(/\n+$/, "").split("\n")[0] || "";
  if (!/^cmake version\s[0-9]+(\.[0-9]+){1,2}([+-][0-9A-Za-z.-]+)?$/.test(banner)) {
    throw new Error("CMake on PATH returned an invalid version banner.");
  }
  return cmake;
}

/** Validate the run identity, producer pins, workspace and reviewed notices before any download,
 * then resolve CMake so a missing tool fails before artifact transfer.
 * @param {NodeJS.ProcessEnv} env @param {Pick<PackageIo, "checkNativeNoticeReview">} io */
function preflightPackage(env, io) {
  if (packagePlan(env) === null || !path.isAbsolute(env["GITHUB_WORKSPACE"] || "")) {
    throw new Error("Late native packaging needs the exact platform, workspace and build SHA-256 pins");
  }
  io.checkNativeNoticeReview();
  const cmake = resolveCmake(env);
  fs.appendFileSync(runnerPath(env, "GITHUB_OUTPUT"), `path=${cmake}\n`);
  return cmake;
}

/** Accept only an absolute, canonical, non-link executable regular file.
 * @param {string | undefined} file @returns {file is string} */
function canonicalExecutable(file) {
  if (!file || !path.isAbsolute(file) || path.resolve(file) !== file ||
      Array.from(file).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) {return false;}
  try {
    const stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file) {return false;}
    fs.accessSync(file, fs.constants.X_OK);
    return true;
  } catch {return false;}
}

/** Require a canonical real directory holding exactly the named entries.
 * @param {string} directory @param {readonly string[]} names @param {string} label */
function exactDirectory(directory, names, label) {
  const stat = fs.lstatSync(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink() || path.resolve(directory) !== directory || fs.realpathSync(directory) !== directory ||
      !hasBuildDirectoryEntries(directory, names)) {
    throw new Error("Unexpected native packaging " + label + " contents");
  }
}

/** Admit a tar whose members are exactly `expected`, whose expansion is bounded, and whose verbose
 * listing proves every member is a plain directory or regular file (no links or special files).
 * @param {PackageIo["ci"]} tar @param {string} archive @param {string[]} expected @param {number} maxExpanded */
function admitTarMembers(tar, archive, expected, maxExpanded) {
  const members = tar.listState(archive, [path.basename(expected[0] || "")], 60000),
    normalized = members.map(name => name.replace(/\/$/, "")).sort(), wanted = [...expected].sort();
  if (normalized.length !== wanted.length || normalized.join("\0") !== wanted.join("\0")) {
    throw new Error("Native packaging tar has missing or unexpected members");
  }
  if (tar.tarExpandedBytes(archive, members, 60000) > maxExpanded) {
    throw new Error("Native packaging tar exceeds its bounded expansion limit");
  }
}

/** Package the admitted packed candidate with the reviewed notices, audit before and after, and seal
 * the receipt-only metadata tar. Writes archive/metadata names and files to GITHUB_OUTPUT.
 * @param {NodeJS.ProcessEnv} env @param {PackageIo} io */
function packageNative(env, io) {
  const plan = packagePlan(env), temp = env["RUNNER_TEMP"], cmake = env["MOM_CMAKE_PATH"],
    planMismatch = "Late native package paths, pins, platform or run identity differ from the exact fresh attempt plan";
  if (!canonicalExecutable(cmake)) {throw new Error("Resolved CMake path is missing, non-executable or non-canonical");}
  if (plan === null || temp === undefined || !path.isAbsolute(temp) || path.resolve(temp) !== temp || fs.realpathSync(temp) !== temp) {
    throw new Error(planMismatch);
  }
  const {suffix, platform, target, binary} = plan, workspace = fs.realpathSync(io.workspace),
    inputRoot = path.join(temp, "native-package-inputs"), incoming = path.join(temp, "incoming"),
    inputDirs = {raw: path.join(inputRoot, "raw"), packed: path.join(inputRoot, "packed-executable")},
    tarFiles = {packed: path.join(incoming, plan.packedBuild + ".tar"), original: path.join(incoming, plan.applicationBuild + ".tar")},
    rawDirectory = path.join(temp, "native-raw"), raw = path.join(rawDirectory, binary), producer = path.join(rawDirectory, "result.json"),
    packedRoot = path.join(io.stateRoot, plan.packedBuild), originalRoot = path.join(io.stateRoot, plan.applicationBuild),
    originalEntry = path.join(originalRoot, "entrypoint.json"), originalConfig = path.join(originalRoot, "application-config.json"),
    runTempConfig = path.join(temp, target === "linux" ?
      `native-linux-assembly-config-${suffix}.json` : `native-windows-application-config-${suffix}.json`),
    entrypoint = path.join(packedRoot, "entrypoint.json"), packReceipt = path.join(packedRoot, "result.json"),
    noticeBody = path.join(workspace, "THIRD-PARTY-NOTICES.txt"), noticeManifest = path.join(workspace, "runtime/native-notice-review.json"),
    outputRoot = path.join(temp, `native-${platform}-package-output-${suffix}`),
    metadataName = `native-${platform}-package-metadata-${suffix}`, metadataRoot = path.join(outputRoot, metadataName),
    metadataTar = path.join(temp, metadataName + ".tar"), outputFile = env["GITHUB_OUTPUT"];
  if (workspace !== path.resolve(io.workspace) || workspace !== path.resolve(env["GITHUB_WORKSPACE"] || "") ||
      [rawDirectory, originalRoot, packedRoot, runTempConfig, outputRoot, metadataRoot, metadataTar].some(file => fs.existsSync(file)) ||
      outputFile === undefined || !path.isAbsolute(outputFile)) {
    throw new Error(planMismatch);
  }
  for (const directory of [...Object.values(inputDirs), incoming]) {
    if (directory !== incoming && path.dirname(directory) !== inputRoot) {throw new Error("Native download directory escaped its bounded root");}
    const stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {throw new Error("Native download root is not canonical");}
  }

  const noticeReview = io.checkNativeNoticeReview(undefined, noticeBody),
    noticeManifestInfo = admitReleaseFile(noticeManifest, null, 2 << 20, "notice review manifest"),
    reviewedNoticeInfo = admitReleaseFile(noticeBody, noticeReview.notice?.sha256, 8 << 20, "reviewed third-party notices");
  if (noticeReview.version !== 1 || !noticeReview.dependencies || typeof noticeReview.dependencies !== "object" ||
      Array.isArray(noticeReview.dependencies) || noticeReview.notice?.bytes !== reviewedNoticeInfo.bytes) {
    throw new Error("Native notice review did not admit the tracked notice body and dependency snapshot");
  }
  const assertNoticeReviewStable = () => {
    const manifest = admitReleaseFile(noticeManifest, null, 2 << 20, "notice review manifest"),
      notice = admitReleaseFile(noticeBody, null, 8 << 20, "reviewed third-party notices");
    if (manifest.sha256 !== noticeManifestInfo.sha256 || manifest.bytes !== noticeManifestInfo.bytes ||
        notice.sha256 !== noticeReview.notice.sha256 || notice.bytes !== noticeReview.notice.bytes) {
      throw new Error("Reviewed notice body or manifest changed during packaging");
    }
  };
  const noticeReviewAudit = {version: 1, manifest: {file: "runtime/native-notice-review.json",
    bytes: noticeManifestInfo.bytes, sha256: noticeManifestInfo.sha256}, snapshot: noticeReview};

  // The raw and packed executables share one file name, so each keeps its own download directory.
  exactDirectory(inputDirs.raw, [binary], "raw executable artifact");
  const rawDownload = path.join(inputDirs.raw, binary), rawInfo = admitReleaseFile(rawDownload, plan.rawSha256, (1 << 30) - 1, "raw executable");
  exactDirectory(inputDirs.packed, [binary], "packed executable artifact");
  const packedInfo = admitReleaseFile(path.join(inputDirs.packed, binary), null, (1 << 30) - 1, "packed executable");
  exactDirectory(incoming, [plan.packedBuild + ".tar", plan.applicationBuild + ".tar"], "incoming build bundle");
  admitReleaseFile(tarFiles.packed, plan.packedBundleSha256, 128 << 20, "packed-build bundle");
  admitReleaseFile(tarFiles.original, plan.originalBundleSha256, 128 << 20, "original-build bundle");

  admitTarMembers(io.ci, tarFiles.packed, [plan.packedBuild, ...[binary, ...packedFiles].map(name => `${plan.packedBuild}/${name}`)], 128 << 20);
  admitTarMembers(io.ci, tarFiles.original, [plan.applicationBuild, ...applicationBuildFiles.map(name => `${plan.applicationBuild}/${name}`)], 128 << 20);
  io.ci.extractState(tarFiles.packed, io.stateRoot, [plan.packedBuild], [plan.packedBuild], 60000);
  io.ci.extractState(tarFiles.original, io.stateRoot, [plan.applicationBuild], [plan.applicationBuild], 60000);
  exactDirectory(packedRoot, [binary, ...packedFiles], "restored packed candidate");
  exactDirectory(originalRoot, applicationBuildFiles, "restored original handoff");
  const packedArtifact = admitReleaseFile(path.join(packedRoot, binary), null, (1 << 30) - 1, "packed candidate"),
    candidateEntryInfo = admitReleaseFile(entrypoint, null, 8 << 20, "packed entrypoint"),
    packInfo = admitReleaseFile(packReceipt, null, 8 << 20, "packing receipt"),
    originInfo = admitReleaseFile(originalEntry, null, 8 << 20, "original entrypoint"),
    handoffFile = path.join(originalRoot, "handoff.json"), originProducer = path.join(originalRoot, "result.json");
  admitReleaseFile(handoffFile, plan.originalHandoffSha256, 8 << 20, "original handoff");
  const originProducerInfo = admitReleaseFile(originProducer, null, 128 << 20, "original producer receipt"),
    originConfigInfo = admitReleaseFile(originalConfig, null, 8 << 20, "original application config"),
    candidate = readBuildJson(entrypoint), origin = readBuildJson(originalEntry), handoff = readBuildJson(handoffFile),
    packing = readBuildJson(packReceipt);
  // The Linux producer always assembles under the fixed /tmp root; Windows uses its RUNNER_TEMP output.
  const originSource = path.join(target === "linux" ? "/tmp" : temp,
    target === "linux" ? `mom-native-linux-application-${suffix}` : `native-windows-application-output-${suffix}`, binary);
  if (packedArtifact.sha256 !== packedInfo.sha256 || packedArtifact.bytes !== packedInfo.bytes ||
      packInfo.sha256 !== candidate.packingReceipt?.sha256 ||
      candidate.version !== 1 || candidate.target !== target || candidate.buildOnly !== true || candidate.publicationQualified !== false ||
      candidate.artifact?.path !== path.join(packedRoot, binary) || candidate.artifact.sha256 !== packedArtifact.sha256 ||
      candidate.artifact.bytes !== packedArtifact.bytes ||
      candidate.source?.path !== raw || candidate.source.sha256 !== plan.rawSha256 || candidate.source.bytes !== rawInfo.bytes ||
      candidate.sourceEntrypoint?.path !== originalEntry || candidate.sourceEntrypoint.sha256 !== originInfo.sha256 ||
      candidate.producerReceipt?.path !== producer || candidate.producerReceipt.sha256 !== originProducerInfo.sha256 ||
      candidate.packingReceipt?.path !== packReceipt || candidate.packingReceipt.sha256 !== packInfo.sha256 ||
      candidate.configFile !== runTempConfig || candidate.configHash !== origin.configHash ||
      !handoff || handoff.version !== 1 || handoff.format !== `mom-${platform}-application-build-handoff-v1` ||
      handoff.target !== (target === "win32" ? "x86_64-pc-windows-msvc" : "x86_64-unknown-linux-gnu") ||
      handoff.buildOnly !== true || handoff.executed !== false || handoff.publicationQualified !== false ||
      handoff.artifact?.file !== binary || handoff.artifact.sha256 !== plan.rawSha256 || handoff.artifact.bytes !== rawInfo.bytes ||
      handoff.config?.file !== "application-config.json" || handoff.config.sha256 !== originConfigInfo.sha256 ||
      handoff.entrypointReceipt?.file !== "entrypoint.json" || handoff.entrypointReceipt.sha256 !== originInfo.sha256 ||
      handoff.producerReceipt?.file !== "result.json" || handoff.producerReceipt.sha256 !== originProducerInfo.sha256 ||
      origin.version !== 1 || origin.target !== target || origin.buildOnly !== true || origin.publicationQualified !== false ||
      origin.source?.path !== originSource ||
      origin.source.sha256 !== plan.rawSha256 || origin.artifact?.path !== origin.source.path ||
      origin.artifact.sha256 !== plan.rawSha256 || origin.artifact.bytes !== rawInfo.bytes ||
      origin.producerReceipt?.sha256 !== originProducerInfo.sha256 || origin.configFile !== runTempConfig ||
      origin.configHash !== originConfigInfo.sha256 ||
      packing.source?.path !== raw || packing.source.sha256 !== plan.rawSha256 || packing.source.bytes !== rawInfo.bytes ||
      packing.artifact?.path !== path.join(packedRoot, binary) || packing.artifact.sha256 !== packedArtifact.sha256 ||
      packing.artifact.bytes !== packedArtifact.bytes || packing.buildOnly !== true || packing.publicationQualified !== false ||
      !hasExactBuildFields(packing.logs, ["integrity", "pack", "restore"])) {
    throw new Error("Packed artifact, original handoff, raw executable or retained packing evidence does not join the pinned build");
  }
  for (const stage of ["pack", "integrity", "restore"]) {
    const logPath = path.join(packedRoot, stage + ".log"), row = packing.logs[stage];
    if (row?.path !== logPath || !hashPattern.test(row.sha256 || "")) {throw new Error("Packed receipt has an invalid " + stage + " log binding");}
    admitReleaseFile(logPath, row.sha256, 8 << 20, stage + " log");
  }
  fs.mkdirSync(rawDirectory, {mode: 0o700});
  fs.copyFileSync(rawDownload, raw, fs.constants.COPYFILE_EXCL);
  fs.copyFileSync(originProducer, producer, fs.constants.COPYFILE_EXCL);
  fs.copyFileSync(originalConfig, runTempConfig, fs.constants.COPYFILE_EXCL);
  if (admitReleaseFile(raw, plan.rawSha256, (1 << 30) - 1, "restored raw executable").bytes !== rawInfo.bytes ||
      admitReleaseFile(producer, originProducerInfo.sha256, 128 << 20, "unchanged producer receipt").sha256 !== originProducerInfo.sha256 ||
      admitReleaseFile(runTempConfig, originConfigInfo.sha256, 8 << 20, "restored original config").sha256 !== originConfigInfo.sha256) {
    throw new Error("Restored raw/config/producer bytes differ from their original same-run pins");
  }
  const preAudit = io.auditNativeRelease(entrypoint);
  if (preAudit.scope !== "build-audit" || preAudit.target !== target || preAudit.buildEvidencePassed !== true ||
      preAudit.runtimeTested !== false || preAudit.publicationQualified !== false ||
      preAudit.sourceSha256 !== plan.rawSha256 || preAudit.sha256 !== packedArtifact.sha256 || preAudit.bytes !== packedArtifact.bytes ||
      fileHash(entrypoint) !== candidateEntryInfo.sha256) {
    throw new Error("Existing build-only audit did not admit the packed candidate and its original build chain");
  }
  const version = JSON.parse(fs.readFileSync(path.join(workspace, "package.json"), "utf8")).version;
  if (typeof version !== "string") {throw new Error("Workspace package.json has no release version");}
  const archiveName = `mom-v${version}-${target === "win32" ? "win.zip" : "lin.tgz"}`,
    archivePath = path.join(outputRoot, archiveName), packageReceipt = path.join(outputRoot, "native-package.json");
  assertNoticeReviewStable();
  let packaged;
  try {packaged = io.packageNativeRelease(entrypoint, cmake, version, undefined, {output: outputRoot, notices: noticeBody});} finally {
    assertNoticeReviewStable();
  }
  if (packaged.archive?.path !== archivePath || packaged.archive.sha256 !== fileHash(archivePath) ||
      packaged.executable?.path !== path.join(packedRoot, binary) || packaged.executable.sha256 !== packedArtifact.sha256 ||
      packaged.executable.bytes !== packedArtifact.bytes || packaged.buildOnly !== true || packaged.publicationQualified !== false ||
      packaged.members?.join("\0") !== [binary, "THIRD-PARTY-NOTICES.txt"].join("\0")) {
    throw new Error("Existing late-notice package API returned a receipt for different bytes or members");
  }
  const archiveInfo = admitReleaseFile(archivePath, packaged.archive.sha256, 1 << 30, "native release archive"),
    packageInfo = admitReleaseFile(packageReceipt, null, 8 << 20, "native package receipt");
  if (archiveInfo.bytes !== packaged.archive.bytes ||
      admitReleaseFile(path.join(outputRoot, "THIRD-PARTY-NOTICES.txt"), noticeReview.notice.sha256, 8 << 20,
        "packaged third-party notices").bytes !== noticeReview.notice.bytes) {
    throw new Error("Late package archive bytes or selected notice body are missing");
  }
  const postAudit = io.auditNativeRelease(entrypoint, {archive: {path: archivePath, tool: cmake}});
  if (postAudit.scope !== "build-audit" || postAudit.target !== target || postAudit.archiveChecked !== true ||
      postAudit.buildEvidencePassed !== true || postAudit.runtimeTested !== false || postAudit.publicationQualified !== false ||
      postAudit.sha256 !== packedArtifact.sha256 || postAudit.archiveSha256 !== packaged.archive.sha256 ||
      fileHash(entrypoint) !== candidateEntryInfo.sha256) {
    throw new Error("Existing archive audit did not admit the exact two-member build-only package");
  }
  fs.mkdirSync(metadataRoot, {mode: 0o700});
  fs.copyFileSync(packageReceipt, path.join(metadataRoot, "native-package.json"), fs.constants.COPYFILE_EXCL);
  fs.writeFileSync(path.join(metadataRoot, "build-audit.json"), JSON.stringify({beforePackaging: preAudit, archive: postAudit}, null, 2) + "\n",
    {flag: "wx", mode: 0o600});
  fs.writeFileSync(path.join(metadataRoot, "native-notice-review.json"), JSON.stringify(noticeReviewAudit, null, 2) + "\n",
    {flag: "wx", mode: 0o600});
  admitReleaseFile(path.join(metadataRoot, "native-notice-review.json"), null, 8 << 20, "notice review snapshot");
  if (admitReleaseFile(path.join(metadataRoot, "native-package.json"), packageInfo.sha256, 8 << 20, "copied package receipt").sha256 !== packageInfo.sha256) {
    throw new Error("Package receipt changed while retaining the audit metadata");
  }
  const metadataBundle = io.ci.packState(metadataTar, outputRoot, [metadataName], 60000),
    metadataBundleInfo = admitReleaseFile(metadataTar, null, 16 << 20, "package metadata tar");
  if (!Number.isSafeInteger(metadataBundle?.bytes) || metadataBundle.bytes < 1 || metadataBundle.bytes > 16 << 20 ||
      metadataBundle.bytes !== metadataBundleInfo.bytes) {throw new Error("Native package metadata tar exceeds its bounded receipt-only size");}
  admitTarMembers(io.ci, metadataTar, [metadataName, ...["build-audit.json", "native-package.json", "native-notice-review.json"]
    .map(name => `${metadataName}/${name}`)], 16 << 20);
  fs.appendFileSync(outputFile, `archive_artifact_name=native-${platform}-application-package-${suffix}\n` +
    `archive_sha256=${packaged.archive.sha256}\narchive_file=${archivePath}\n` +
    `metadata_artifact_name=${metadataName}.tar\nmetadata_file=${metadataTar}\n`);
}

/** Run one workflow step: `preflight` before downloads, `package` after them. @param {string | undefined} command */
function main(command) {
  if (command === "preflight") {
    preflightPackage(process.env, {checkNativeNoticeReview});
  } else if (command === "package") {
    packageNative(process.env, {stateRoot: "/tmp", workspace: process.cwd(), ci, checkNativeNoticeReview, auditNativeRelease,
      packageNativeRelease});
  } else {
    throw new Error("Usage: node scripts/ci/package-native.js preflight|package");
  }
}

module.exports = {packagePlan, resolveCmake, preflightPackage, packageNative};

if (require.main === module) {
  try {main(process.argv[2]);} catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
