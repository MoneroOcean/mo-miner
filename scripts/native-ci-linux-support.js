"use strict";

// Native CI Linux support SDKs: I/O/OpenSSL, Hermes, zstd and static OpenMP producer commands and admission.
const fs = require("node:fs"), path = require("node:path");
const {isDeepStrictEqual} = require("node:util");
const {canonicalExecutable, ciBuildJobs, ioOpenSslChildTimeout, runNode, runNodeProcessGroup} = require("./native-ci-job");
const {fileHash, readBuildJson} = require("./build-source-inputs");
const {ioPins, nativeIoInputs} = require("./build-native-io");
const {nativeOpenSslInputs, opensslPin} = require("./build-native-openssl");
const {verifyZstdArchive} = require("./fetch-native-sdk-sources");
const {hermesSdkPlan} = require("./build-native-hermes");
const {zstdPin, nativeZstdInputs} = require("./build-native-zstd");
const {nativeHermesInputs} = require("./native-release-sdk-inputs");
const {nativeOpenmpInputs} = require("./build-native-openmp-sdk");
const {extractState, listState, packState, temp} = require("./native-ci-tar");
const {linuxSysrootToolchain} = require("./native-linux-sysroot");
const {hermesStatePaths, openmpStatePaths, provisionOneApi,
  restoreSources, verifyHermesSource, verifyIoSource, zstdStatePaths} = require("./native-ci-linux-sources");
const repo = path.resolve(__dirname, "..");

/** Explicit signed Intel tools for the unchanged public Hermes SDK producer. */
function hermesBuildTools() {
  return {git: canonicalExecutable("/usr/bin/git"), cmake: canonicalExecutable("/usr/bin/cmake"),
    cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx"), ninja: canonicalExecutable("/usr/bin/ninja")};
}

/** Return the exact public producer command for the fetched bare Git root, not its cohort parent.
 * @param {ReturnType<typeof hermesStatePaths>} state @param {ReturnType<typeof hermesBuildTools>} tools @param {string} jobs
 * @param {string} [toolchain] the glibc sysroot's CMake toolchain file */
function hermesBuildCommand(state, tools, jobs, toolchain) {
  const source = path.join(state.source, "roots/hermes");
  hermesSdkPlan(source, state.sdkRoot, tools, Number(jobs), "linux", toolchain);
  return {script: "scripts/build-native-hermes.js", args: [source, state.sdkRoot, tools.git, tools.cmake, tools.cc, tools.cxx, tools.ninja, jobs]};
}

/** Accept only the producer's canonical absolute patch location; tool paths and every hash stay exact.
 * @param {unknown} fingerprints @param {ReturnType<typeof hermesBuildTools>} tools @param {string[]} patches */
function assertHermesSdkFingerprints(fingerprints, tools, patches) {
  const toolNames = ["cc", "cmake", "cxx", "git", "ninja"], knownPatches = ["runtime/hermes-host-rejections.patch"];
  const canonicalAbsolute = (/** @type {unknown} */ value) => typeof value === "string" && value.length <= 4096 &&
    path.isAbsolute(value) && path.resolve(value) === value && !Array.from(value).some(char => char.charCodeAt(0) < 32 ||
      char.charCodeAt(0) >= 127 && char.charCodeAt(0) <= 159);
  if (!fingerprints || typeof fingerprints !== "object" || Array.isArray(fingerprints) ||
      Object.keys(tools).sort().join("\0") !== toolNames.join("\0") ||
      !isDeepStrictEqual(patches, knownPatches) || !Object.values(tools).every(canonicalAbsolute)) {
    throw new Error("Hermes SDK fingerprint plan is not the known bounded tool/patch set");
  }
  const recorded = /** @type {Record<string, unknown>} */ (fingerprints), recordedPaths = Object.keys(recorded),
    toolPaths = Object.values(tools), aliases = /** @type {Record<string, string>} */ ({}), allowed = new Set(toolPaths);
  if (new Set(toolPaths).size !== toolPaths.length || recordedPaths.some(file => !canonicalAbsolute(file))) {
    throw new Error("Hermes SDK fingerprints contain duplicate or noncanonical paths");
  }
  for (const relative of patches) {
    const suffix = "/" + relative, matches = recordedPaths.filter(file => file.endsWith(suffix));
    if (matches.length !== 1) {throw new Error("Hermes SDK must record exactly one canonical path for each source patch");}
    const sourcePath = matches[0] || "";
    if (allowed.has(sourcePath)) {throw new Error("Hermes SDK patch identity collides with a tool fingerprint");}
    allowed.add(sourcePath);
    aliases[relative] = sourcePath;
  }
  if (recordedPaths.length !== allowed.size || recordedPaths.some(file => !allowed.has(file))) {
    throw new Error("Hermes SDK fingerprints contain unknown or missing tool/patch entries");
  }
  for (const file of toolPaths) {
    if (recorded[file] !== fileHash(file)) {throw new Error("Hermes SDK tool fingerprint changed: " + file);}
  }
  for (const [relative, sourcePath] of Object.entries(aliases)) {
    if (recorded[sourcePath] !== fileHash(path.join(repo, relative))) {
      throw new Error("Hermes SDK patch fingerprint changed: " + relative);
    }
  }
}

/** Use the shared strict SDK admission; add only run-specific tool and command-plan identity.
 * @param {ReturnType<typeof hermesStatePaths>} state @param {ReturnType<typeof hermesBuildTools>} tools @param {string} jobs
 * @param {string} [toolchain] the glibc sysroot's CMake toolchain file */
function auditHermesSdk(state, tools, jobs, toolchain) {
  const command = hermesBuildCommand(state, tools, jobs, toolchain), receiptPath = path.join(state.sdkRoot, "result.json"),
    receipt = readBuildJson(receiptPath), before = readBuildJson(path.join(state.sdkRoot, "before.json")),
    plan = hermesSdkPlan(command.args[0] || "", state.sdkRoot, tools, Number(jobs), "linux", toolchain);
  nativeHermesInputs(receiptPath);
  assertHermesSdkFingerprints(receipt.fingerprints, tools, plan.patches);
  if (!isDeepStrictEqual(before.configure, plan.configure) || !isDeepStrictEqual(before.compile, plan.compile)) {
    throw new Error("Native Hermes SDK does not record this run's explicit Intel tools and producer command plan");
  }
  return receipt;
}

/** @typedef {{ar: string, cc: string, make: string, perl: string, ranlib: string, tar: string}} LinuxOpenSslTools */
/** @typedef {{ar: string, cc: string, make: string, mkdir: string, tar: string}} LinuxZstdTools */

/** Use only the existing signed Intel compiler route and stock build tools. @returns {LinuxOpenSslTools} */
function linuxOpenSslTools() {
  return {ar: canonicalExecutable("/usr/bin/ar"), cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    make: canonicalExecutable("/usr/bin/make"), perl: canonicalExecutable("/usr/bin/perl"),
    ranlib: canonicalExecutable("/usr/bin/ranlib"), tar: canonicalExecutable("/usr/bin/tar")};
}

/** Use only the signed Intel compiler route and explicit stock tools required by the zstd producer. @returns {LinuxZstdTools} */
function linuxZstdTools() {
  return {ar: canonicalExecutable("/usr/bin/ar"), cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    make: canonicalExecutable("/usr/bin/make"), mkdir: canonicalExecutable("/usr/bin/mkdir"), tar: canonicalExecutable("/usr/bin/tar")};
}

/** Preserve the producer's SDK-root contract; callers must not substitute lib/libzstd.a.
 * @param {Pick<ReturnType<typeof zstdStatePaths>, "archive" | "sdkRoot">} state @param {string} toolsFile */
function zstdBuildCommand(state, toolsFile) {
  return {script: "scripts/build-native-zstd.js", args: [state.archive, state.sdkRoot, toolsFile]};
}

/** Reuse producer admission, then join its original-input path to this run's verified acquisition.
 * @param {Pick<ReturnType<typeof zstdStatePaths>, "source" | "archive" | "sdkRoot">} state @param {LinuxZstdTools} tools @param {string} toolsFile
 * @param {{verifyArchive?: typeof verifyZstdArchive, sdkInputs?: typeof nativeZstdInputs}} [readers] */
function auditZstdSdk(state, tools, toolsFile, readers = {}) {
  const fetched = (readers.verifyArchive || verifyZstdArchive)(state.source), archive = fetched.archives?.["zstd"],
    sdk = (readers.sdkInputs || nativeZstdInputs)(state.sdkRoot), record = readBuildJson(sdk.receipt),
    expectedTools = Object.fromEntries(Object.entries(tools).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}]));
  if (!archive || archive.path !== state.archive || archive.sha256 !== zstdPin.sha256 ||
      record.archive?.path !== archive.path || record.archive.sha256 !== archive.sha256 ||
      sdk.archive !== path.join(state.sdkRoot, "lib/libzstd.a") || record.config?.sha256 !== fileHash(toolsFile) ||
      record.plan?.jobs !== 4 || !isDeepStrictEqual(record.tools, expectedTools) ||
      sdk.files?.[state.archive] !== archive.sha256 || sdk.files?.[sdk.receipt] !== sdk.receiptSha256 ||
      !/^[0-9a-f]{64}$/.test(sdk.files?.[sdk.archive] || "")) {
    throw new Error("Native zstd SDK does not bind this run's pinned archive, descriptor, tools and SDK root");
  }
  return sdk;
}

/** @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function ioFinalTar(paths) {return path.join(paths.runnerTemp, "native-linux-io-openssl-sdk-" + paths.suffix + ".tar");}

/** Verify both SDK receipts and exact original inputs before any same-run SDK state is retained.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths @param {number} [verifyTimeoutMs] */
function auditIoOpenSsl(paths, verifyTimeoutMs = 2 * 60 * 1000) {
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", paths.source], paths.env, verifyTimeoutMs);
  verifyIoSource(paths);
  const io = nativeIoInputs(paths.ioSdk, "linux"), openssl = nativeOpenSslInputs(paths.opensslSdk, "linux"),
    ioReceipt = readBuildJson(io.receipt), opensslReceipt = readBuildJson(path.join(paths.opensslSdk, "result.json")),
    toolsDescriptor = readBuildJson(paths.opensslTools), expectedTools = linuxOpenSslTools();
  const cmake = canonicalExecutable("/usr/bin/cmake"), ninja = canonicalExecutable("/usr/bin/ninja"),
    expectedIoFingerprints = Object.fromEntries([cmake, expectedTools.cc, ninja].map(file => [file, fileHash(file)]));
  const expectedIoInputs = {
    [path.join(paths.ioSource, "libuv-v1.51.0.tar.gz")]: ioPins.libuv.sha256,
    [path.join(paths.ioSource, "zlib-1.3.1.tar.gz")]: ioPins.zlib.sha256,
  };
  const opensslArchive = path.join(paths.source, "openssl-" + opensslPin.version + ".tar.gz");
  if (ioReceipt.target !== "linux" || ioReceipt.buildOnly !== true || ioReceipt.publicationQualified !== false ||
      !isDeepStrictEqual(ioReceipt.inputs, expectedIoInputs) || !isDeepStrictEqual(ioReceipt.fingerprints, expectedIoFingerprints) ||
      !opensslReceipt.archive || opensslReceipt.archive.path !== opensslArchive || opensslReceipt.archive.sha256 !== opensslPin.sha256 ||
      !toolsDescriptor || Object.keys(toolsDescriptor).sort().join(",") !== "jobs,tools,version" ||
      toolsDescriptor.version !== 1 || ![2, 4].includes(toolsDescriptor.jobs) || !isDeepStrictEqual(toolsDescriptor.tools, expectedTools) ||
      opensslReceipt.config?.sha256 !== fileHash(paths.opensslTools) ||
      !opensslReceipt.tools || Object.keys(opensslReceipt.tools).sort().join(",") !== Object.keys(expectedTools).sort().join(",")) {
    throw new Error("Native I/O/OpenSSL SDK receipts do not bind this run's pinned sources and tools");
  }
  for (const [name, file] of Object.entries(expectedTools)) {
    const record = opensslReceipt.tools[name];
    if (!record || record.path !== file || record.sha256 !== fileHash(file)) {throw new Error("OpenSSL SDK tool identity changed: " + name);}
  }
  if (fileHash(opensslArchive) !== opensslPin.sha256 ||
      Object.entries(expectedIoInputs).some(([file, hash]) => fileHash(file) !== hash) ||
      io.libraries.length !== 2 || openssl.libraries.length !== 2 || io.libraries.some(file => !fs.statSync(file).isFile()) ||
      openssl.libraries.some(file => !fs.statSync(file).isFile())) {
    throw new Error("Native I/O/OpenSSL SDK source or selected archive changed during admission");
  }
  return {io, openssl};
}

/** Keep the two public producer invocations bound to this run's canonical source and output paths.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths @param {string} cmake @param {string} ninja @param {LinuxOpenSslTools} tools @param {string} jobs */
function ioOpenSslBuildCommands(paths, cmake, ninja, tools, jobs) {
  return [
    {script: "scripts/build-native-io.js", args: [path.join(paths.ioSource, "libuv-v1.51.0.tar.gz"),
      path.join(paths.ioSource, "zlib-1.3.1.tar.gz"), paths.ioSdk, cmake, tools.cc, ninja, jobs]},
    {script: "scripts/build-native-openssl.js", args: [path.join(paths.source, "openssl-" + opensslPin.version + ".tar.gz"),
      paths.opensslSdk, paths.opensslTools]},
  ];
}

/** Restore and verify the same-run IO/OpenSSL sources, then build both original static SDKs.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildIoOpenSsl(paths) {
  const deadlineEpoch = paths.env["MOM_NATIVE_IO_OPENSSL_DEADLINE_EPOCH"] || "";
  ioOpenSslChildTimeout(deadlineEpoch, 60 * 60 * 1000, Date.now(), 5 * 60 * 1000);
  restoreSources(paths, deadlineEpoch);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-io-openssl-" + paths.suffix,
    ioOpenSslChildTimeout(deadlineEpoch, 15 * 60 * 1000, Date.now(), 5 * 60 * 1000));
  const cmake = canonicalExecutable("/usr/bin/cmake"), ninja = canonicalExecutable("/usr/bin/ninja"), tools = linuxOpenSslTools();
  for (const file of ["/usr/bin/python3", "/usr/bin/ld"]) {canonicalExecutable(file);}
  if (fs.existsSync(paths.ioSdkRoot)) {throw new Error("Native I/O/OpenSSL SDK output already exists");}
  fs.mkdirSync(paths.ioSdkRoot, {mode: 0o700});
  const config = {version: 1, jobs: Number(ciBuildJobs(paths.env)), tools};
  fs.writeFileSync(paths.opensslTools, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  for (const command of ioOpenSslBuildCommands(paths, cmake, ninja, tools, ciBuildJobs(paths.env))) {
    runNodeProcessGroup(command.script, command.args, paths.env,
      ioOpenSslChildTimeout(deadlineEpoch, 45 * 60 * 1000, Date.now(), 5 * 60 * 1000));
  }
  auditIoOpenSsl(paths, ioOpenSslChildTimeout(deadlineEpoch, 2 * 60 * 1000));
  packState(ioFinalTar(paths), temp, [paths.sourceName, paths.ioSourceName, paths.ioSdkName],
    ioOpenSslChildTimeout(deadlineEpoch, 5 * 60 * 1000));
}

/** Restore same-run Hermes source and build/audit the original static SDK with explicit tools.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildHermesSdk(paths) {
  const state = hermesStatePaths(paths), git = canonicalExecutable("/usr/bin/git");
  extractState(state.incomingTar, temp, [state.sourceName]);
  verifyHermesSource(paths, state, git);
  fs.unlinkSync(state.incomingTar);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-hermes-" + paths.suffix);
  const tools = hermesBuildTools();
  const toolchain = linuxSysrootToolchain(paths.env), command = hermesBuildCommand(state, tools, ciBuildJobs(paths.env), toolchain);
  runNode(command.script, command.args, paths.env, 55 * 60 * 1000);
  auditHermesSdk(state, tools, ciBuildJobs(paths.env), toolchain);
  packState(state.sdkTar, temp, [state.sourceName, state.sdkName]);
}

/** Restore the original same-run archive, build with the signed Intel 2026.2 compiler and retain the complete admitted SDK root. @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildZstdSdk(paths) {
  const state = zstdStatePaths(paths);
  extractState(state.incomingTar, temp, [state.sourceName]);
  verifyZstdArchive(state.source);
  fs.unlinkSync(state.incomingTar);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-zstd-" + paths.suffix);
  const tools = linuxZstdTools(), config = {version: 1, jobs: 4, tools};
  fs.writeFileSync(state.toolsFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  try {
    const command = zstdBuildCommand(state, state.toolsFile);
    runNode(command.script, command.args, paths.env, 25 * 60 * 1000);
    auditZstdSdk(state, tools, state.toolsFile);
    packState(state.sdkTar, temp, [state.sourceName, state.sdkName]);
  } finally {
    fs.unlinkSync(state.toolsFile);
  }
}

/** Explicit signed Intel 2026.2, CMake/Ninja and OS inspection tools for static OpenMP production. */
function openmpBuildTools() {
  return {cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    cmake: canonicalExecutable("/usr/bin/cmake"), cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx"),
    ninja: canonicalExecutable("/usr/bin/ninja"), readelf: canonicalExecutable("/usr/bin/readelf"), tar: canonicalExecutable("/usr/bin/tar")};
}

/** Return the maintained producer invocation for this run's verified original archive.
 * @param {ReturnType<typeof openmpStatePaths>} state */
function openmpBuildCommand(state) {
  return {script: "scripts/build-native-openmp-sdk.js", args: [state.archive, state.sdkRoot, state.toolsFile]};
}

/** Reuse full producer admission, then bind the artifact to this exact same-run source handoff.
 * @param {ReturnType<typeof openmpStatePaths>} state @param {Record<string,string>} tools */
function auditOpenmpSdk(state, tools) {
  const sdk = nativeOpenmpInputs(state.sdkRoot), record = readBuildJson(sdk.receipt),
    expectedTools = Object.fromEntries(Object.entries(tools).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}]));
  if (record.archive.path !== state.archive || record.fetchReceipt.path !== path.join(state.source, "fetch.json") ||
      ![2, 4].includes(record.plan.jobs) || !isDeepStrictEqual(record.tools, expectedTools) ||
      sdk.archive !== path.join(state.sdkRoot, "sdk/libomp.a") || sdk.files[state.archive] !== record.archive.sha256 ||
      sdk.files[sdk.receipt] !== sdk.receiptSha256 || sdk.files[sdk.archive] !== record.sdk.sha256) {
    throw new Error("Static OpenMP SDK does not bind this run's pinned archive, tools and ACPP handoff path");
  }
  return sdk;
}

/** Restore the same-run source state, provision Intel 2026.2 within a separate budget, and build only static libomp.a.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildOpenmpSdk(paths) {
  const state = openmpStatePaths(paths);
  if (fs.existsSync(state.sdkRoot) || fs.existsSync(state.toolsFile)) {throw new Error("Native OpenMP SDK output or tool descriptor already exists");}
  restoreSources(paths);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-openmp-" + paths.suffix, 40 * 60 * 1000);
  const tools = openmpBuildTools(), config = {version: 1, jobs: Number(ciBuildJobs(paths.env)), tools};
  fs.writeFileSync(state.toolsFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  try {
    const command = openmpBuildCommand(state);
    runNode(command.script, command.args, paths.env, 10 * 60 * 1000);
    auditOpenmpSdk(state, tools);
    packState(state.sdkTar, temp, [state.sourceName, state.sdkName]);
    listState(state.sdkTar, [state.sourceName, state.sdkName]);
  } finally {
    fs.unlinkSync(state.toolsFile);
  }
}

module.exports = {assertHermesSdkFingerprints, auditHermesSdk, auditIoOpenSsl, auditOpenmpSdk, auditZstdSdk, buildHermesSdk,
  buildIoOpenSsl, buildOpenmpSdk, buildZstdSdk, hermesBuildCommand, hermesBuildTools, ioFinalTar, ioOpenSslBuildCommands,
  linuxOpenSslTools, linuxZstdTools, openmpBuildCommand, openmpBuildTools, zstdBuildCommand};
