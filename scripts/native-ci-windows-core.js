"use strict";

// Native CI Windows Core build on its native runner and its compact handoff.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {ciSuffix, runNode} = require("./native-ci-job");
const {fileHash, readBuildJson, nativeSourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {ioPins, validateIoArchive} = require("./build-native-io");
const {verifyIoArchives} = require("./fetch-native-sdk-sources");
const {windowsNativeCoreHandoffInputs} = require("./native-windows-core-handoff");
const {isStaticArchive} = require("./native-audit");
const repo = path.resolve(__dirname, "..");

const windowsNativeCorePackReserveMs = 7 * 60 * 1000, windowsNativeCoreMinBuildMs = 5 * 60 * 1000,
  windowsNativeCoreMaxFetchMs = 20 * 60 * 1000, windowsNativeCoreMaxBuildMs = 40 * 60 * 1000,
  windowsNativeCoreMaxArchiveStepMs = 2 * 60 * 1000;

/** @typedef {{suffix: string, runnerTemp: string, name: string, workRoot: string, source: string, unpacked: string,
 * output: string, descriptor: string, include: string, bundle: string, archive: string}} WindowsNativeCorePaths */
/** Keep Windows Core on its native runner: its MSBuild/COFF tools and Node-gyp inputs are not
 * interchangeable with the Linux SDK stage. Every path is derived from this CI job.
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @returns {WindowsNativeCorePaths} */
function windowsNativeCoreStatePaths(environment = process.env, platform = process.platform) {
  const runnerTemp = environment["RUNNER_TEMP"] || "";
  if (platform !== "win32" || !path.isAbsolute(runnerTemp)) {throw new Error("Windows native Core CI needs a Windows runner and RUNNER_TEMP");}
  const suffix = ciSuffix(environment), name = "native-windows-core-" + suffix,
    workRoot = path.join(path.resolve(runnerTemp), "mom-native-windows-core-" + suffix);
  return {suffix, runnerTemp: path.resolve(runnerTemp), name, workRoot, source: path.join(workRoot, "io-source"),
    unpacked: path.join(workRoot, "libuv-unpacked"), output: path.join(workRoot, "producer-output"), descriptor: path.join(workRoot, "windows-core.json"),
    include: path.join(workRoot, "libuv-unpacked", ioPins.libuv.directory, "include"), bundle: path.join(path.resolve(runnerTemp), name),
    archive: path.join(path.resolve(runnerTemp), name + ".tar")};
}

/** Cap each Windows Core child process to the job-start deadline while preserving time for remaining stages.
 * @param {string} deadlineEpoch @param {"fetch" | "listing" | "extract" | "build" | "package"} phase @param {number} [nowMs]
 * @returns {{timeoutMs: number, createTimeoutMs: number, listingTimeoutMs: number, remainingMs: number}} */
function windowsNativeCoreDeadlineBudget(deadlineEpoch, phase, nowMs = Date.now()) {
  if (!/^\d{10}$/.test(deadlineEpoch) || !Number.isSafeInteger(nowMs) || nowMs < 0 ||
      !["fetch", "listing", "extract", "build", "package"].includes(phase)) {
    throw new Error("Windows Core stage needs a valid bounded job-start deadline");
  }
  const remainingMs = Number(deadlineEpoch) * 1000 - nowMs, maximumMs = 60 * 60 * 1000;
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0 || remainingMs > maximumMs) {
    throw new Error("Windows Core job deadline is expired or exceeds the 60-minute job bound");
  }
  const reserve = windowsNativeCorePackReserveMs + windowsNativeCoreMinBuildMs;
  if (phase === "fetch") {
    // Leave four minutes for archive inspection/extraction, a minimal build window, and final handoff.
    const remainingAfterFetch = 2 * windowsNativeCoreMaxArchiveStepMs + reserve;
    if (remainingMs < remainingAfterFetch + 60 * 1000) {
      throw new Error("Insufficient aggregate job time remains before fetching Windows Core inputs");
    }
    return {timeoutMs: Math.min(windowsNativeCoreMaxFetchMs, remainingMs - remainingAfterFetch), createTimeoutMs: 0, listingTimeoutMs: 0, remainingMs};
  }
  if (phase === "listing" || phase === "extract") {
    const futureArchive = phase === "listing" ? windowsNativeCoreMaxArchiveStepMs : 0;
    const reserveAfter = futureArchive + reserve;
    if (remainingMs < reserveAfter + 60 * 1000) {
      throw new Error("Insufficient aggregate job time remains for Windows Core archive preparation");
    }
    return {timeoutMs: Math.min(windowsNativeCoreMaxArchiveStepMs, remainingMs - reserveAfter), createTimeoutMs: 0, listingTimeoutMs: 0, remainingMs};
  }
  if (phase === "build") {
    if (remainingMs < reserve + 60 * 1000) {
      throw new Error("Insufficient aggregate job time remains before the Windows Core producer build");
    }
    return {timeoutMs: Math.min(windowsNativeCoreMaxBuildMs, remainingMs - windowsNativeCorePackReserveMs), createTimeoutMs: 0, listingTimeoutMs: 0, remainingMs};
  }
  if (remainingMs < windowsNativeCorePackReserveMs) {
    throw new Error("Insufficient aggregate job time remains for the Windows Core handoff");
  }
  return {timeoutMs: windowsNativeCorePackReserveMs, createTimeoutMs: 5 * 60 * 1000, listingTimeoutMs: 2 * 60 * 1000, remainingMs};
}

/** The workflow resolves VS, Python, Node-gyp and Intel tools before entering this function.
 * The public producer validates each selected file and verifies the Intel version/target.
 * @param {WindowsNativeCorePaths} state @param {NodeJS.ProcessEnv} [environment] */
function windowsNativeCoreDescriptor(state, environment = process.env) {
  const {validateWindowsCoreDescriptor, validatePortableEnvironment} = require("./build-native-windows-core");
  validatePortableEnvironment(environment);
  return validateWindowsCoreDescriptor({version: 1, nodeGyp: environment["MOM_NATIVE_WINDOWS_CORE_NODE_GYP"],
    msbuild: environment["MOM_NATIVE_WINDOWS_CORE_MSBUILD"], dumpbin: environment["MOM_NATIVE_WINDOWS_CORE_DUMPBIN"],
    python: environment["MOM_NATIVE_WINDOWS_CORE_PYTHON"], compiler: environment["MOM_NATIVE_WINDOWS_CORE_COMPILER"], libuvInclude: state.include});
}

/** Assert the consumer-side portable COFF Core contract before retaining a same-run tar.
 * `inputs.json` remains the public producer's original receipt bytes.
 * @param {WindowsNativeCorePaths} state
 * @returns {{archive: string, receiptFile: string, archiveSha256: string, receiptSha256: string, receipt: Record<string, unknown>, bytes: number}} */
function auditWindowsNativeCoreOutput(state) {
  const archive = path.join(state.output, "cpu-core.lib"), receiptFile = path.join(state.output, "inputs.json"),
    receiptStat = fs.lstatSync(receiptFile), archiveStat = fs.lstatSync(archive);
  if (!receiptStat.isFile() || receiptStat.isSymbolicLink() || !archiveStat.isFile() || archiveStat.isSymbolicLink() || !isStaticArchive(archive)) {
    throw new Error("Windows Core output is not a regular static COFF archive with a regular receipt");
  }
  const receipt = readBuildJson(receiptFile);
  const archiveSha256 = fileHash(archive), receiptSha256 = fileHash(receiptFile), fact = receipt.archive;
  if (receipt.target !== "win32" || receipt.gpu !== true || receipt.portable !== true || receipt.buildOnly !== true ||
      receipt.executed !== false || receipt.publicationQualified !== false || !receipt.compiler ||
      !/^2026\.2(?:\.|$)/.test(receipt.compiler.version || "") || receipt.compiler.target !== "x86_64-pc-windows-msvc" ||
      receipt.objectSha256 !== archiveSha256 || !fact || fact.path !== archive || fact.sha256 !== archiveSha256 || fact.bytes !== archiveStat.size ||
      fact.stubs !== false || !Number.isSafeInteger(fact.coffTableCount) || fact.coffTableCount < 1 ||
      fact.coffTableCount !== fact.machineCount || fact.machineCount !== fact.x64MachineCount) {
    throw new Error("Windows Core receipt fails the portable x64 consumer contract");
  }
  verifySourceInputs(receipt.sourceState, nativeSourceInputs(repo, "cpu"), "Windows Core CI handoff");
  if (!receipt.headerHashes || typeof receipt.headerHashes !== "object" || Array.isArray(receipt.headerHashes) ||
      !Object.keys(receipt.headerHashes).length || Object.keys(receipt.headerHashes).length > 4096) {
    throw new Error("Windows Core receipt has no bounded libuv-header input closure");
  }
  const includeStat = fs.lstatSync(state.include);
  if (!includeStat.isDirectory() || includeStat.isSymbolicLink() || fs.realpathSync(state.include).toLowerCase() !== path.resolve(state.include).toLowerCase()) {
    throw new Error("Windows Core libuv include root is redirected");
  }
  const headers = new Set();
  for (const [file, hash] of Object.entries(receipt.headerHashes)) {
    if (!path.isAbsolute(file) || typeof hash !== "string" || !/^[0-9a-f]{64}$/.test(hash)) {throw new Error("Windows Core header receipt is malformed");}
    const relative = path.relative(state.include, file), normalized = relative.replaceAll("\\", "/"), header = fs.lstatSync(file);
    if (path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep) ||
        !/\.(?:h|hpp|inc)$/i.test(file) || !header.isFile() || header.isSymbolicLink() ||
        fs.realpathSync(file).toLowerCase() !== path.resolve(file).toLowerCase() || fileHash(file) !== hash) {
      throw new Error("Windows Core header input changed or escaped the pinned libuv include root");
    }
    headers.add(normalized);
  }
  if (!headers.has("uv.h") || !headers.has("uv/win.h")) {throw new Error("Windows Core libuv header closure is incomplete");}
  return {archive, receiptFile, archiveSha256, receiptSha256, receipt, bytes: archiveStat.size};
}

/** @typedef {{core_sha256: string, receipt_sha256: string, handoff_sha256: string, bundle_sha256: string, bundle_bytes: number, tar_tool_sha256: string}} WindowsNativeCoreReport */
/** Package only the Core archive, unchanged producer receipt, and a small transport index.
 * The Windows SDK/toolchain and downloaded source archive are intentionally not uploaded.
 * @param {WindowsNativeCorePaths} state @param {string} sourceReceiptSha256 @param {string} libuvArchiveSha256
 * @param {{createTimeoutMs?: number, listingTimeoutMs?: number}} [timeouts]
 * @returns {WindowsNativeCoreReport} */
function packWindowsNativeCoreHandoff(state, sourceReceiptSha256, libuvArchiveSha256, timeouts = {}) {
  const createTimeoutMs = timeouts.createTimeoutMs ?? 5 * 60 * 1000, listingTimeoutMs = timeouts.listingTimeoutMs ?? 2 * 60 * 1000;
  if (!Number.isSafeInteger(createTimeoutMs) || createTimeoutMs < 1 || createTimeoutMs > 5 * 60 * 1000 ||
      !Number.isSafeInteger(listingTimeoutMs) || listingTimeoutMs < 1 || listingTimeoutMs > 2 * 60 * 1000) {
    throw new Error("Windows Core handoff tar timeouts are invalid");
  }
  const audited = auditWindowsNativeCoreOutput(state);
  if (libuvArchiveSha256 !== ioPins.libuv.sha256 || !/^[0-9a-f]{64}$/.test(sourceReceiptSha256 || "")) {
    throw new Error("Windows Core handoff source provenance differs from the pinned libuv input");
  }
  if (fs.existsSync(state.bundle) || fs.existsSync(state.archive)) {throw new Error("Windows Core handoff output already exists");}
  fs.mkdirSync(state.bundle, {mode: 0o700});
  const coreFile = path.join(state.bundle, "cpu-core.lib"), receiptFile = path.join(state.bundle, "inputs.json");
  fs.copyFileSync(audited.archive, coreFile, fs.constants.COPYFILE_EXCL);
  fs.copyFileSync(audited.receiptFile, receiptFile, fs.constants.COPYFILE_EXCL);
  if (fileHash(coreFile) !== audited.archiveSha256 || fileHash(receiptFile) !== audited.receiptSha256) {
    throw new Error("Windows Core handoff changed the producer output bytes");
  }
  const handoff = {format: "mom-native-windows-core-handoff-v1", target: "win32", architecture: "x64",
    archive: {file: "cpu-core.lib", bytes: audited.bytes, sha256: audited.archiveSha256},
    producer: {script: "scripts/build-native-windows-core.js", receipt: "inputs.json", receiptSha256: audited.receiptSha256},
    libuv: {version: "1.51.0", archiveSha256: libuvArchiveSha256, fetchReceiptSha256: sourceReceiptSha256},
    buildOnly: true, executed: false, publicationQualified: false};
  fs.writeFileSync(path.join(state.bundle, "handoff.json"), JSON.stringify(handoff, null, 2) + "\n", {flag: "wx", mode: 0o600});
  if (!isDeepStrictEqual(fs.readdirSync(state.bundle).sort(), ["cpu-core.lib", "handoff.json", "inputs.json"])) {
    throw new Error("Windows Core handoff has unexpected members");
  }
  const handoffFile = path.join(state.bundle, "handoff.json"), handoffSha256 = fileHash(handoffFile),
    admitted = windowsNativeCoreHandoffInputs(state.bundle, {handoffSha256});
  if (admitted.archiveSha256 !== audited.archiveSha256 || admitted.receiptSha256 !== audited.receiptSha256) {
    throw new Error("Windows Core handoff reader rejected the original producer payload");
  }
  const systemRoot = process.env["SystemRoot"] || process.env["WINDIR"] || "", tarTool = path.join(systemRoot, "System32", "tar.exe"),
    tarStat = fs.lstatSync(tarTool);
  if (!path.isAbsolute(tarTool) || !tarStat.isFile() || tarStat.isSymbolicLink() || fs.existsSync(state.archive)) {
    throw new Error("Windows Core handoff requires the Windows inbox tar tool and a new archive path");
  }
  const tarSha256 = fileHash(tarTool), pack = spawnSync(tarTool, ["-cf", state.archive, "-C", state.runnerTemp, state.name],
    {encoding: "utf8", windowsHide: true, timeout: createTimeoutMs, maxBuffer: 4 << 20});
  if (pack.error || pack.signal || pack.status !== 0) {throw new Error("Windows Core handoff tar creation failed", {cause: pack.error});}
  const listing = spawnSync(tarTool, ["-tf", state.archive], {encoding: "utf8", windowsHide: true, timeout: listingTimeoutMs, maxBuffer: 4 << 20});
  if (listing.error || listing.signal || listing.status !== 0 || typeof listing.stdout !== "string") {
    throw new Error("Windows Core handoff tar listing failed", {cause: listing.error});
  }
  const members = listing.stdout.trim().split(/\r?\n/).filter(Boolean).map(name => name.replaceAll("\\", "/").replace(/\/$/, "")).sort(),
    expected = [state.name, state.name + "/cpu-core.lib", state.name + "/handoff.json", state.name + "/inputs.json"].sort();
  if (!isDeepStrictEqual(members, expected) || fileHash(handoffFile) !== admitted.handoffSha256) {
    throw new Error("Windows Core handoff tar has unexpected members or changed source receipt");
  }
  const report = {core_sha256: audited.archiveSha256, receipt_sha256: audited.receiptSha256,
    handoff_sha256: admitted.handoffSha256, bundle_sha256: fileHash(state.archive), bundle_bytes: fs.statSync(state.archive).size, tar_tool_sha256: tarSha256};
  console.log(JSON.stringify(report));
  return report;
}

/** Fetch only the pinned libuv/zlib source cohort, then build and retain the public Windows Core. */
function buildWindowsNativeCoreCi() {
  const state = windowsNativeCoreStatePaths();
  /** @type {NodeJS.ProcessEnv} */
  const environment = {...process.env, MOM_AOT_BUILD_ONLY: "1"};
  const deadlineEpoch = environment["MOM_NATIVE_WINDOWS_CORE_DEADLINE_EPOCH"] || "";
  const githubOutput = environment["GITHUB_OUTPUT"] || "";
  if (!path.isAbsolute(githubOutput) || !fs.lstatSync(githubOutput).isFile() || fs.lstatSync(githubOutput).isSymbolicLink()) {
    throw new Error("Windows Core CI needs the workflow's regular GITHUB_OUTPUT file");
  }
  for (const output of [state.workRoot, state.bundle, state.archive]) {
    if (fs.existsSync(output)) {throw new Error("Windows native Core stage refuses an existing CI work-root output: " + path.basename(output));}
  }
  const tempStat = fs.lstatSync(state.runnerTemp);
  if (!tempStat.isDirectory() || tempStat.isSymbolicLink() || path.resolve(fs.realpathSync(state.runnerTemp)).toLowerCase() !== state.runnerTemp.toLowerCase()) {
    throw new Error("Windows Core RUNNER_TEMP must be canonical and not redirected");
  }
  const fetchBudget = windowsNativeCoreDeadlineBudget(deadlineEpoch, "fetch");
  fs.mkdirSync(state.workRoot, {mode: 0o700});
  runNode("scripts/fetch-native-sdk-sources.js", ["io", state.source, "--allow-network"], environment, fetchBudget.timeoutMs);
  verifyIoArchives(state.source);
  const fetchReceipt = path.join(state.source, "fetch.json"), sourceReceiptSha256 = fileHash(fetchReceipt),
    libuvArchive = path.join(state.source, ioPins.libuv.directory + ".tar.gz"), libuvArchiveSha256 = fileHash(libuvArchive),
    systemRoot = environment["SystemRoot"] || environment["WINDIR"] || "", tarTool = path.join(systemRoot, "System32", "tar.exe");
  if (libuvArchiveSha256 !== ioPins.libuv.sha256 || !path.isAbsolute(tarTool) || !fs.lstatSync(tarTool).isFile()) {
    throw new Error("Windows Core needs the exact libuv pin and Windows inbox tar.exe");
  }
  const listingBudget = windowsNativeCoreDeadlineBudget(deadlineEpoch, "listing");
  const listing = spawnSync(tarTool, ["-tzf", libuvArchive], {encoding: "utf8", windowsHide: true, timeout: listingBudget.timeoutMs, maxBuffer: 4 << 20});
  if (listing.error || listing.signal || listing.status !== 0 || typeof listing.stdout !== "string") {throw new Error("Pinned libuv archive listing failed", {cause: listing.error});}
  validateIoArchive(listing.stdout, ioPins.libuv.directory);
  fs.mkdirSync(state.unpacked, {mode: 0o700});
  const extractBudget = windowsNativeCoreDeadlineBudget(deadlineEpoch, "extract");
  const unpack = spawnSync(tarTool, ["-xzf", libuvArchive, "-C", state.unpacked, ioPins.libuv.directory + "/include"],
    {encoding: "utf8", windowsHide: true, timeout: extractBudget.timeoutMs, maxBuffer: 4 << 20});
  if (unpack.error || unpack.signal || unpack.status !== 0) {throw new Error("Pinned libuv header extraction failed", {cause: unpack.error});}
  const config = windowsNativeCoreDescriptor(state, environment);
  if (fs.existsSync(state.output) || fs.existsSync(state.descriptor)) {throw new Error("Windows Core producer output is not new");}
  fs.writeFileSync(state.descriptor, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  // Fail on tool/layout mistakes before the expensive public producer starts.
  require("./build-native-windows-core").windowsCorePlan(config, state.output, repo);
  const buildBudget = windowsNativeCoreDeadlineBudget(deadlineEpoch, "build");
  runNode("scripts/build-native-windows-core.js", [state.descriptor, state.output], environment, buildBudget.timeoutMs);
  verifyIoArchives(state.source);
  const packageBudget = windowsNativeCoreDeadlineBudget(deadlineEpoch, "package");
  const result = packWindowsNativeCoreHandoff(state, sourceReceiptSha256, libuvArchiveSha256, packageBudget);
  if (fileHash(fetchReceipt) !== sourceReceiptSha256 || fileHash(libuvArchive) !== libuvArchiveSha256) {
    throw new Error("Windows Core source acquisition changed after production");
  }
  fs.appendFileSync(githubOutput, `core_sha256=${result.core_sha256}\nreceipt_sha256=${result.receipt_sha256}\nhandoff_sha256=${result.handoff_sha256}\nbundle_sha256=${result.bundle_sha256}\n`, "utf8");
  return result;
}

module.exports = {auditWindowsNativeCoreOutput, buildWindowsNativeCoreCi, packWindowsNativeCoreHandoff,
  windowsNativeCoreDeadlineBudget, windowsNativeCoreDescriptor, windowsNativeCoreStatePaths};
