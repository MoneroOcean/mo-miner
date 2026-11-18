"use strict";

// Linux native component and application CI steps: profile validation, the component tar seal,
// the diagnostics-capturing application build, and binding the raw executable to its build receipts.

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawn} = require("node:child_process");
const ciRelease = require("../native-ci-linux-release"),
  ciSources = require("../native-ci-linux-sources"),
  ciTar = require("../native-ci-tar");
const {ciSuffix} = require("../native-ci-job");
const {fileHash, readBuildJson} = require("../build-source-inputs");
const {canonicalFile, appendOutputs} = require("./seal");

const repo = path.resolve(__dirname, "../..");
// The outer stage log and the copied linker log are bounded so a runaway build cannot fill the runner.
const diagnosticsTailBytes = 1 << 20;
const linkDiagnosticsBytes = 16 << 20;

/** Validate the requested Core or GPU factory profile. @param {NodeJS.ProcessEnv} [environment] */
function componentProfile(environment = process.env) {
  const profile = environment["NATIVE_PROFILE"] || "";
  if (profile !== "core" && !ciRelease.linuxNativeGpuProfiles.includes(profile)) {
    throw new Error("Select one exact Linux Core or GPU factory profile");
  }
  return profile;
}

/** Seal one producer component tar after checking its exact root and manifest member.
 * @param {Pick<ReturnType<typeof ciSources.ciPaths>, "suffix" | "runnerTemp">} paths @param {string} profile
 * @returns {{bundle_sha256: string}} */
function sealLinuxComponent(paths, profile) {
  const state = ciRelease.linuxNativeComponentStatePaths(paths, profile);
  canonicalFile(state.archive, "Linux component handoff");
  if (!ciTar.listState(state.archive, [state.name]).includes(state.name + "/component.json")) {
    throw new Error("Linux component handoff is missing its producer manifest");
  }
  return {bundle_sha256: fileHash(state.archive)};
}

/** Validate RUNNER_TEMP as an absolute canonical directory path. @param {NodeJS.ProcessEnv} environment */
function canonicalRunnerTemp(environment) {
  const runnerTemp = environment["RUNNER_TEMP"] || "";
  if (!path.isAbsolute(runnerTemp) || path.resolve(runnerTemp) !== runnerTemp || fs.realpathSync(runnerTemp) !== runnerTemp) {
    throw new Error("RUNNER_TEMP must be an absolute canonical path");
  }
  return runnerTemp;
}

/** Run the bounded application stage with stderr merged into stdout, keep only the last 1 MiB
 * as the uploaded diagnostics log, and on failure copy the head of the linker log beside it.
 * @param {NodeJS.ProcessEnv} [environment] @param {string[]} [command] @returns {Promise<number>} the stage exit status */
async function runApplicationBuild(environment = process.env,
  command = ["scripts/run-bounded-native-ci.sh", "node", "scripts/run-native-ci.js", "stage", "linux-native-application"]) {
  const suffix = ciSuffix(environment), runnerTemp = canonicalRunnerTemp(environment);
  const child = spawn("/bin/sh", ["-c", "exec \"$@\" 2>&1", "sh", ...command], {cwd: repo, env: environment, stdio: ["ignore", "pipe", "inherit"]});
  /** @type {Buffer[]} */
  const chunks = [];
  let retained = 0;
  child.stdout.on("data", (/** @type {Buffer} */ chunk) => {
    chunks.push(chunk);
    retained += chunk.length;
    while (chunks.length > 1 && retained - (chunks[0]?.length || 0) >= diagnosticsTailBytes) {retained -= chunks.shift()?.length || 0;}
  });
  const status = await new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("close", (code, signal) => resolve(signal ? 128 + (os.constants.signals[signal] || 0) : code ?? 1));
  });
  const tail = Buffer.concat(chunks).subarray(-diagnosticsTailBytes);
  fs.writeFileSync(path.join(runnerTemp, "native-linux-application-diagnostics-" + suffix + ".log"), tail);
  process.stdout.write(tail);
  if (status !== 0) {copyLinkDiagnostics(suffix, runnerTemp);}
  return status;
}

/** Copy at most 16 MiB of a regular (non-symlink) application linker log for failure upload.
 * @param {string} suffix @param {string} runnerTemp */
function copyLinkDiagnostics(suffix, runnerTemp) {
  const linkLog = path.join("/tmp", "mom-native-linux-application-" + suffix, "application", "link.log"),
    stat = fs.lstatSync(linkLog, {throwIfNoEntry: false});
  if (!stat?.isFile()) {return;}
  const fd = fs.openSync(linkLog, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  try {
    const buffer = Buffer.alloc(Math.min(stat.size, linkDiagnosticsBytes)), bytes = fs.readSync(fd, buffer, 0, buffer.length, 0);
    fs.writeFileSync(path.join(runnerTemp, "native-linux-application-link-diagnostics-" + suffix + ".log"), buffer.subarray(0, bytes));
  } finally {fs.closeSync(fd);}
}

/** Admit a canonical regular application file and remember its identity for later drift checks.
 * @param {string} file @param {string} label @param {number} maximum */
function applicationFile(file, label, maximum) {
  const stat = canonicalFile(file, "Linux application " + label, maximum);
  return {bytes: stat.size, sha256: fileHash(file), identity: {dev: stat.dev, ino: stat.ino, size: stat.size}};
}

/** The entrypoint and producer receipt must both describe this exact executable and derived config.
 * @param {any} result @param {any} entrypoint
 * @param {{binary: string, receiptPath: string, configPath: string, bytes: number, sha256: string, receiptSha256: string, configSha256: string}} expected */
function assertApplicationReceipts(result, entrypoint, expected) {
  const {binary, receiptPath, configPath, bytes, sha256, receiptSha256, configSha256} = expected;
  if (result.binary !== binary || result.sha256 !== sha256 || result.bytes !== bytes || result.configHash !== configSha256 ||
      result.buildOnly !== true || result.publicationQualified !== false || result.application?.application !== true ||
      result.application?.entrypoint !== "mom.js" || result.application?.buildOnly !== true ||
      entrypoint.version !== 1 || entrypoint.target !== "linux" || entrypoint.configFile !== configPath || entrypoint.configHash !== configSha256 ||
      entrypoint.source?.path !== binary || entrypoint.source.sha256 !== sha256 ||
      entrypoint.artifact?.path !== binary || entrypoint.artifact.bytes !== bytes || entrypoint.artifact.sha256 !== sha256 ||
      entrypoint.producerReceipt?.path !== receiptPath || entrypoint.producerReceipt.sha256 !== receiptSha256 ||
      !entrypoint.recipe || typeof entrypoint.recipe !== "object" || Array.isArray(entrypoint.recipe) ||
      !entrypoint.componentHashes || typeof entrypoint.componentHashes !== "object" || Array.isArray(entrypoint.componentHashes) ||
      Object.keys(entrypoint.componentHashes).join("\0") !== receiptPath || entrypoint.componentHashes[receiptPath] !== receiptSha256 ||
      entrypoint.buildOnly !== true || entrypoint.publicationQualified !== false) {
    throw new Error("Linux handoff differs from the public build-only entrypoint, config or producer receipt");
  }
}

/** Plan the fresh CI work-root paths for the application handoff, validating the aggregate deadline.
 * @param {NodeJS.ProcessEnv} environment */
function applicationHandoffPlan(environment) {
  const suffix = ciSuffix(environment), runnerTemp = canonicalRunnerTemp(environment),
    deadline = Number(environment["MOM_NATIVE_APPLICATION_DEADLINE_EPOCH"]), appRoot = path.join("/tmp", "mom-native-linux-application-" + suffix),
    evidenceName = "native-linux-application-build-" + suffix;
  const plan = {suffix, deadline, appRoot, evidenceName, evidenceRoot: path.join("/tmp", evidenceName),
    archive: path.join(runnerTemp, evidenceName + ".tar"), runtimeDirectory: path.join(runnerTemp, "mom-native-runtime-" + suffix),
    binary: path.join(appRoot, "mom"), receiptPath: path.join(appRoot, "result.json"), entrypointPath: path.join(appRoot, "entrypoint.json"),
    configPath: path.join(runnerTemp, "native-linux-assembly-config-" + suffix + ".json")};
  if (!Number.isSafeInteger(deadline) || deadline * 1000 <= Date.now() || !path.isAbsolute(environment["GITHUB_OUTPUT"] || "") ||
      fs.existsSync(plan.runtimeDirectory) || fs.existsSync(plan.evidenceRoot) || fs.existsSync(plan.archive)) {
    throw new Error("Linux application handoff paths are not the fresh canonical CI work-root plan");
  }
  const rootStat = fs.lstatSync(appRoot);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(appRoot) !== appRoot) {
    throw new Error("Linux application output is not a canonical directory");
  }
  return plan;
}

/** Remaining milliseconds before the aggregate deadline, capped; refuses to start with under one second.
 * @param {number} deadline Unix seconds @param {number} capMs */
function remainingMs(deadline, capMs) {
  const milliseconds = Math.min(capMs, deadline * 1000 - Date.now());
  if (!Number.isSafeInteger(milliseconds) || milliseconds < 1000) {throw new Error("Linux application aggregate deadline expired during handoff sealing");}
  return milliseconds;
}

/** Bind the raw executable to its producer receipt, entrypoint and derived config; pack those
 * receipts plus a handoff record into a compact evidence tar; and copy the executable for runtime upload.
 * @param {NodeJS.ProcessEnv} [environment] @returns {Record<string, string | number>} the step outputs */
function bindApplicationHandoff(environment = process.env) {
  const plan = applicationHandoffPlan(environment), {suffix, binary, receiptPath, entrypointPath, configPath, evidenceRoot, evidenceName, archive} = plan;
  const artifact = applicationFile(binary, "runtime executable", (1 << 30) - 1),
    resultRecord = applicationFile(receiptPath, "producer receipt", 8 << 20),
    entrypointRecord = applicationFile(entrypointPath, "original entrypoint", 8 << 20),
    configRecord = applicationFile(configPath, "derived application config", 8 << 20);
  assertApplicationReceipts(readBuildJson(receiptPath), readBuildJson(entrypointPath), {binary, receiptPath, configPath,
    bytes: artifact.bytes, sha256: artifact.sha256, receiptSha256: resultRecord.sha256, configSha256: configRecord.sha256});
  fs.mkdirSync(evidenceRoot, {mode: 0o700});
  for (const [source, name, sha256] of /** @type {Array<[string, string, string]>} */ ([[configPath, "application-config.json", configRecord.sha256],
    [entrypointPath, "entrypoint.json", entrypointRecord.sha256], [receiptPath, "result.json", resultRecord.sha256]])) {
    const destination = path.join(evidenceRoot, name);
    fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
    if (fileHash(source) !== sha256 || fileHash(destination) !== sha256) {throw new Error("Linux application evidence copy changed: " + name);}
  }
  const handoff = {version: 1, format: "mom-linux-application-build-handoff-v1",
    target: "x86_64-unknown-linux-gnu", artifact: {file: "mom", bytes: artifact.bytes, sha256: artifact.sha256},
    config: {file: "application-config.json", sha256: configRecord.sha256},
    producerReceipt: {file: "result.json", sha256: resultRecord.sha256},
    entrypointReceipt: {file: "entrypoint.json", sha256: entrypointRecord.sha256},
    buildOnly: true, executed: false, publicationQualified: false};
  const handoffFile = path.join(evidenceRoot, "handoff.json");
  fs.writeFileSync(handoffFile, JSON.stringify(handoff, null, 2) + "\n", {flag: "wx", mode: 0o444});
  const handoffSha256 = fileHash(handoffFile), packed = ciTar.packState(archive, "/tmp", [evidenceName], remainingMs(plan.deadline, 5 * 60 * 1000)),
    expectedMembers = [evidenceName, ...["application-config.json", "entrypoint.json", "handoff.json", "result.json"].map(name => evidenceName + "/" + name)].sort(),
    members = ciTar.listState(archive, [evidenceName], remainingMs(plan.deadline, 5 * 60 * 1000)).map(name => name.replace(/\/$/, "")).sort();
  const runtimeStat = fs.lstatSync(binary), outputStat = fs.lstatSync(environment["GITHUB_OUTPUT"] || "");
  if (packed.bytes < 1 || packed.bytes > 128 << 20 || members.join("\0") !== expectedMembers.join("\0") ||
      fileHash(handoffFile) !== handoffSha256 ||
      artifact.identity.dev !== runtimeStat.dev || artifact.identity.ino !== runtimeStat.ino || artifact.identity.size !== runtimeStat.size ||
      fileHash(binary) !== artifact.sha256 || fileHash(configPath) !== configRecord.sha256 ||
      fileHash(receiptPath) !== resultRecord.sha256 || fileHash(entrypointPath) !== entrypointRecord.sha256 ||
      !outputStat.isFile() || outputStat.isSymbolicLink()) {
    throw new Error("Linux application runtime or compact evidence changed while sealing");
  }
  fs.mkdirSync(plan.runtimeDirectory, {mode: 0o700});
  const runtimeBinary = path.join(plan.runtimeDirectory, "mom");
  fs.copyFileSync(binary, runtimeBinary, fs.constants.COPYFILE_EXCL);
  fs.chmodSync(runtimeBinary, runtimeStat.mode & 0o777);
  if (fileHash(runtimeBinary) !== artifact.sha256 || fileHash(binary) !== artifact.sha256) {
    throw new Error("Runtime handoff copy differs from the pinned build receipt");
  }
  const outputs = {sha256: artifact.sha256, artifact_name: "native-linux-executable-" + suffix, executable_bytes: artifact.bytes,
    entrypoint_sha256: entrypointRecord.sha256, producer_receipt_sha256: resultRecord.sha256,
    build_handoff_artifact_name: evidenceName + ".tar", build_handoff_bundle_sha256: fileHash(archive), build_handoff_sha256: handoffSha256};
  appendOutputs(outputs, environment);
  console.log("Bound same-run Linux executable and compact build evidence: " + artifact.sha256);
  return outputs;
}

/** Dispatch one Linux application CI command. @param {string[]} args @returns {Promise<number>} the process exit status */
async function main(args) {
  const [command, ...rest] = args;
  if (rest.length) {throw new Error("Linux application CI commands take no further arguments");}
  if (command === "component-profile") {
    componentProfile();
  } else if (command === "component-seal") {
    appendOutputs(sealLinuxComponent(ciSources.ciPaths(), componentProfile()));
  } else if (command === "application-build") {
    return runApplicationBuild();
  } else if (command === "application-bind") {
    bindApplicationHandoff();
  } else {
    throw new Error("Usage: linux-application.js component-profile|component-seal|application-build|application-bind");
  }
  return 0;
}

module.exports = {componentProfile, sealLinuxComponent, runApplicationBuild, copyLinkDiagnostics, assertApplicationReceipts,
  applicationHandoffPlan, bindApplicationHandoff, main};

if (require.main === module) {
  main(process.argv.slice(2)).then(status => {process.exitCode = status;}, error => {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  });
}
