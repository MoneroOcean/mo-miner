"use strict";

// SDK-free native packing job: admit this CI job's raw executable and compact build metadata,
// then pack through the existing release API. No compiler, SDK or miner runs here.
const fs = require("node:fs"), path = require("node:path");
const ci = require("../native-ci-tar"), {fileHash, readBuildJson} = require("../build-source-inputs");
const {applicationBuildFiles, releaseArtifactNames, sha256Pin, runnerPath, admitReleaseFile} = require("../native-ci-job");
const {packNativeEntrypoint} = require("../build-release");
const {prepareNativePackingTool} = require("../prepare-native-packing-tool");

/** @typedef {{stateRoot: string, ci: Pick<typeof ci, "listState" | "extractState" | "packState" | "tarExpandedBytes">,
 *   packNativeEntrypoint: typeof packNativeEntrypoint,
 *   prepareNativePackingTool: (output: string, env: NodeJS.ProcessEnv) => {upx: string, upxSha256: string}}} PackIo */

/** Derive the exact artifact names and admit the independent producer pins.
 * Artifact names are never inputs: they are fixed by platform and the CI suffix.
 * @param {NodeJS.ProcessEnv} env */
function packingPlan(env) {
  const artifacts = releaseArtifactNames(env),
    rawSha256 = sha256Pin(env, "NATIVE_RAW_SHA256"), metadataSha256 = sha256Pin(env, "NATIVE_METADATA_SHA256"),
    handoffSha256 = sha256Pin(env, "NATIVE_HANDOFF_SHA256");
  if (!artifacts ||
      rawSha256 === null || metadataSha256 === null || handoffSha256 === null) {
    throw new Error("Native packing needs the exact raw executable, build metadata and independent SHA-256 pins");
  }
  return {...artifacts, rawSha256, metadataSha256, handoffSha256};
}

/** Restore the pinned metadata bundle, bind it to the raw executable, pack, and seal the packed build bundle.
 * The raw executable stays at RUNNER_TEMP/native-raw because the packing receipt records that source path.
 * @param {NodeJS.ProcessEnv} env @param {PackIo} io */
function packNative(env, io) {
  const plan = packingPlan(env), temp = runnerPath(env, "RUNNER_TEMP"), outputFile = runnerPath(env, "GITHUB_OUTPUT"),
    metadata = path.join(io.stateRoot, plan.applicationBuild), archive = path.join(temp, "incoming", plan.applicationBuild + ".tar"),
    raw = path.join(temp, "native-raw", plan.binary), output = path.join(io.stateRoot, plan.packedBuild);
  admitReleaseFile(archive, plan.metadataSha256, 128 << 20);
  const rawInfo = admitReleaseFile(raw, plan.rawSha256, (1 << 30) - 1);
  if (fs.readdirSync(path.dirname(raw)).join() !== plan.binary) {throw new Error("Raw runtime artifact contains unexpected files");}
  const members = io.ci.listState(archive, [plan.applicationBuild], 60000),
    expected = [plan.applicationBuild, ...applicationBuildFiles.map(name => plan.applicationBuild + "/" + name)].sort();
  if (members.map(name => name.replace(/\/$/, "")).sort().join("\0") !== expected.join("\0")) {
    throw new Error("Native packing metadata has unexpected members");
  }
  if (io.ci.tarExpandedBytes(archive, members, 60000) > 128 << 20) {
    throw new Error("Native packing metadata exceeds its bounded expansion limit");
  }
  io.ci.extractState(archive, io.stateRoot, [plan.applicationBuild], [plan.applicationBuild], 60000);
  const handoffFile = path.join(metadata, "handoff.json");
  admitReleaseFile(handoffFile, plan.handoffSha256, 8 << 20);
  const handoff = readBuildJson(handoffFile), target = plan.target === "win32" ? "x86_64-pc-windows-msvc" : "x86_64-unknown-linux-gnu";
  if (handoff.version !== 1 || handoff.format !== `mom-${plan.platform}-application-build-handoff-v1` ||
      handoff.target !== target || handoff.buildOnly !== true || handoff.executed !== false || handoff.publicationQualified !== false ||
      handoff.artifact?.file !== plan.binary || handoff.artifact.sha256 !== plan.rawSha256 || handoff.artifact.bytes !== rawInfo.bytes) {
    throw new Error("Native packing metadata does not bind this raw candidate and attempt");
  }
  /** @type {Array<[string, string]>} */
  const receipts = [["config", "application-config.json"], ["entrypointReceipt", "entrypoint.json"], ["producerReceipt", "result.json"]];
  for (const [key, name] of receipts) {
    if (handoff[key]?.file !== name) {throw new Error("Native packing metadata has an unexpected receipt name");}
    admitReleaseFile(path.join(metadata, name), handoff[key].sha256 ?? "", 128 << 20);
  }
  const entrypoint = readBuildJson(path.join(metadata, "entrypoint.json"));
  if (entrypoint.configHash !== handoff.config.sha256) {throw new Error("Native packing descriptor binding differs");}
  const producer = path.join(path.dirname(raw), "result.json");
  fs.copyFileSync(path.join(metadata, "result.json"), producer, fs.constants.COPYFILE_EXCL);
  // Only the pinned official packer is fetched; no compiler, SDK or miner is run here.
  const {upx, upxSha256} = io.prepareNativePackingTool(path.join(temp, "native-packing-tool"), env);
  const packed = io.packNativeEntrypoint({entrypointPath: path.join(metadata, "entrypoint.json"),
    entrypointSha256: handoff.entrypointReceipt.sha256, sourcePath: raw, sourceSha256: plan.rawSha256, producerReceiptPath: producer,
    target: plan.target, output, upx, upxSha256});
  admitReleaseFile(archive, plan.metadataSha256, 128 << 20);
  admitReleaseFile(handoffFile, plan.handoffSha256, 8 << 20);
  const buildArchive = path.join(temp, plan.packedBuild + ".tar");
  io.ci.packState(buildArchive, io.stateRoot, [plan.packedBuild], 60000);
  fs.appendFileSync(outputFile, `executable_artifact_name=native-${plan.platform}-packed-executable-${plan.suffix}\n` +
    `executable_sha256=${packed.artifact.sha256}\nbuild_bundle_sha256=${fileHash(buildArchive)}\n`);
}

/** Run one workflow step: `validate` before downloads, `pack` after them. @param {string | undefined} command */
function main(command) {
  if (command === "validate") {
    packingPlan(process.env);
  } else if (command === "pack") {
    packNative(process.env, {stateRoot: "/tmp", ci, packNativeEntrypoint, prepareNativePackingTool});
  } else {
    throw new Error("Usage: node scripts/ci/pack-native.js validate|pack");
  }
}

module.exports = {packingPlan, packNative};

if (require.main === module) {
  try {main(process.argv[2]);} catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
