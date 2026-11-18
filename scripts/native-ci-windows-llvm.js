"use strict";

// Native CI Windows host SDK sources, LLVM descriptor and cross-built LLVM phases.
const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");
const {appendWorkflowOutputs, ciBuildJobs, ciSuffix, hasExactKeys, runNode, runNodeProcessGroup} = require("./native-ci-job");
const {fileHash, readBuildJson, sourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {verifyWindowsHostSdkArchives, windowsHostSdkCatalog} = require("./fetch-native-sdk-sources");
const {windowsControllerHandoffInputs, windowsLocalControllerHandoffInputs} = require("./native-windows-controller-handoff");
const {safeRelative: safeWindowsSdkRelative, defaultWindowsNativeLibraryPolicy} = require("./prepare-native-windows-sdk");
const {archiveName: llvmArchiveName, archiveSha256: llvmArchiveSha256} = require("./native-llvm-sdk");
const {isStaticArchive} = require("./native-audit");
const windowsIntelRuntimeCatalog = /** @type {import("./native-ci-windows-controller").WindowsIntelRuntimeCatalog} */ (
  require("../runtime/windows-intel-runtime.json"));
const {listState, packState, safeRelative, temp} = require("./native-ci-tar");
const {windowsControllerReadJson, windowsControllerRegularFile} = require("./native-ci-windows-controller");
const repo = path.resolve(__dirname, "..");

const windowsHostSdkArchiveMaxMs = 20 * 60 * 1000, windowsHostSdkPackageReserveMs = 4 * 60 * 1000,
  windowsHostSdkComposeReserveMs = 12 * 60 * 1000, windowsHostSdkMaxJobMs = 60 * 60 * 1000;

/** Same-run source/composition handoff for the selected Windows host-SDK roots. The vendor SDK
 * remains on this disposable runner; only receipts and path/hash metadata cross Actions.
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {string} [architecture]
 * @returns {{suffix:string,runnerTemp:string,workRoot:string,workName:string,archiveRoot:string,archiveName:string,
 * sdkOutput:string,outputName:string,metadataTar:string,deadlineEpoch:string}} */
function windowsHostSdkStatePaths(environment = process.env, platform = process.platform, architecture = process.arch) {
  const runnerTemp = environment["RUNNER_TEMP"] || "";
  if (platform !== "linux" || architecture !== "x64" || !path.isAbsolute(runnerTemp)) {
    throw new Error("Windows host-SDK CI needs a Linux x64 runner and RUNNER_TEMP");
  }
  const canonicalTemp = fs.realpathSync(runnerTemp);
  if (canonicalTemp !== path.resolve(runnerTemp) || !fs.lstatSync(canonicalTemp).isDirectory()) {throw new Error("Windows host-SDK RUNNER_TEMP is not canonical");}
  const suffix = ciSuffix(environment), workName = "mom-native-windows-host-sdk-" + suffix,
    outputName = "native-windows-host-sdk-" + suffix, workRoot = path.join(canonicalTemp, workName),
    archiveRoot = path.join(workRoot, "archives"), sdkOutput = path.join(workRoot, outputName),
    metadataTar = path.join(canonicalTemp, outputName + "-metadata.tar");
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1" || environment["MOM_BUILD_JOBS"] !== ciBuildJobs(environment) ||
      environment["MOM_NATIVE_WINDOWS_HOST_SDK_WORK_ROOT"] !== workRoot || environment["MOM_NATIVE_WINDOWS_HOST_SDK_ARCHIVE_DIR"] !== archiveRoot ||
      environment["MOM_NATIVE_WINDOWS_HOST_SDK_OUTPUT_DIR"] !== sdkOutput || environment["MOM_NATIVE_WINDOWS_HOST_SDK_METADATA_TAR"] !== metadataTar) {
    throw new Error("Windows host-SDK CI paths, job envelope, or build-only mode differ from this run");
  }
  return {suffix, runnerTemp: canonicalTemp, workRoot, workName, archiveRoot, archiveName: "archives", sdkOutput, outputName, metadataTar,
    deadlineEpoch: environment["MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH"] || ""};
}

/** Keep fetch, offline composition and metadata packaging inside one job-start deadline.
 * @param {string} deadlineEpoch @param {"fetch" | "compose" | "package"} phase @param {number} [nowMs] */
function windowsHostSdkDeadlineBudget(deadlineEpoch, phase, nowMs = Date.now()) {
  if (!/^\d{10}$/.test(deadlineEpoch) || !Number.isSafeInteger(nowMs) || nowMs < 0 ||
      !["fetch", "compose", "package"].includes(phase)) {throw new Error("Windows host-SDK CI needs a bounded job-start deadline");}
  const remainingMs = Number(deadlineEpoch) * 1000 - nowMs;
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0 || remainingMs > windowsHostSdkMaxJobMs) {
    throw new Error("Windows host-SDK job deadline is expired or exceeds the 60-minute job bound");
  }
  if (phase === "fetch") {
    const reserve = windowsHostSdkComposeReserveMs + windowsHostSdkPackageReserveMs;
    if (remainingMs < reserve + 60 * 1000) {throw new Error("Insufficient job time remains for Windows host-SDK acquisition");}
    return {timeoutMs: Math.min(windowsHostSdkArchiveMaxMs, remainingMs - reserve), remainingMs};
  }
  if (phase === "compose") {
    if (remainingMs < windowsHostSdkComposeReserveMs + windowsHostSdkPackageReserveMs) {
      throw new Error("Insufficient job time remains for offline Windows host-SDK composition");
    }
    return {timeoutMs: Math.min(45 * 60 * 1000, remainingMs - windowsHostSdkPackageReserveMs), remainingMs};
  }
  if (remainingMs < 60 * 1000) {throw new Error("Insufficient job time remains for the Windows host-SDK receipt handoff");}
  return {timeoutMs: Math.min(windowsHostSdkPackageReserveMs, remainingMs), remainingMs};
}

/** Append hash-bound outputs only after source acquisition and extraction both completed. @param {string} file @param {Record<string,string|number>} values */
function appendWindowsHostSdkOutputs(file, values) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || !path.isAbsolute(file)) {throw new Error("Windows host-SDK job output file is not regular");}
  const lines = Object.entries(values).map(([key, value]) => {
    if (!/^[a-z][a-z0-9_]*$/.test(key) || !/^(?:[a-f0-9]{64}|[0-9]+|native-windows-host-sdk-[0-9]+-[0-9]+-metadata\.tar)$/.test(String(value))) {
      throw new Error("Windows host-SDK output is malformed: " + key);
    }
    return key + "=" + value;
  });
  fs.appendFileSync(file, lines.join("\n") + "\n", "utf8");
}

/** Copy the two small immutable acquisition descriptors beside the composed receipt.
 * @param {ReturnType<typeof windowsHostSdkStatePaths>} state @param {string} fetchReceiptHash */
function retainWindowsHostSdkMetadata(state, fetchReceiptHash) {
  const receipt = path.join(state.archiveRoot, "fetch.json"), receiptStat = fs.lstatSync(receipt), lockSource = path.join(repo, "runtime/windows-host-sdk.json"),
    lockTarget = path.join(state.sdkOutput, "source-lock.json"), receiptTarget = path.join(state.sdkOutput, "source-fetch.json");
  if (!receiptStat.isFile() || receiptStat.isSymbolicLink() || fileHash(receipt) !== fetchReceiptHash ||
      fs.existsSync(lockTarget) || fs.existsSync(receiptTarget)) {throw new Error("Windows host-SDK source receipts changed or collide");}
  const copies = /** @type {Array<[string, string]>} */ ([[receipt, receiptTarget], [lockSource, lockTarget]]);
  for (const [source, target] of copies) {
    fs.copyFileSync(source, target, fs.constants.COPYFILE_EXCL); fs.chmodSync(target, 0o444);
    if (fileHash(target) !== fileHash(source)) {throw new Error("Windows host-SDK metadata copy changed bytes");}
  }
  return {receiptTarget, lockTarget};
}

/** Fetch the 42 pinned vendor archives, compose the selected roots offline, and upload receipts only. */
function buildWindowsHostSdkSourcesCi() {
  const state = windowsHostSdkStatePaths(), environment = {...process.env, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: ciBuildJobs()},
    lock = windowsHostSdkCatalog(), outputFile = process.env["GITHUB_OUTPUT"] || "", summary = process.env["GITHUB_STEP_SUMMARY"] || "";
  if (Object.keys(lock.packages).length + Object.keys(lock.cabinets).length !== 42 || !path.isAbsolute(outputFile) || !path.isAbsolute(summary)) {
    throw new Error("Windows host-SDK CI requires the fixed 42-archive cohort and workflow output files");
  }
  for (const file of [outputFile, summary]) {
    const stat = fs.lstatSync(file); if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Windows host-SDK workflow output is not regular");}
  }
  for (const target of [state.workRoot, state.metadataTar]) {
    try {
      fs.lstatSync(target); throw new Error("Windows host-SDK CI refuses an existing CI work-root output");
    } catch (error) {if (Reflect.get(Object(error), "code") !== "ENOENT") {throw error;}}
  }
  const recipeLeaves = ["scripts/native-ci-windows-llvm.js", "scripts/prepare-native-windows-sdk.js", "scripts/fetch-native-sdk-sources.js",
      "scripts/build-source-inputs.js", "runtime/windows-host-sdk.json"], recipe = sourceInputs(repo, [], recipeLeaves);
  fs.mkdirSync(state.workRoot, {mode: 0o700});
  const fetchBudget = windowsHostSdkDeadlineBudget(state.deadlineEpoch, "fetch");
  runNode("scripts/fetch-native-sdk-sources.js", ["windows-host-sdk", state.archiveRoot, "--allow-network"], environment, fetchBudget.timeoutMs);
  const fetched = verifyWindowsHostSdkArchives(state.archiveRoot), fetchReceiptPath = path.join(state.archiveRoot, "fetch.json"), fetchReceiptSha256 = fileHash(fetchReceiptPath);
  if (fetched.status !== "PASS" || fetched.target !== "windows-host" || fetched.networked !== true || fetched.buildOnly !== true ||
      fetched.compiled !== false || fetched.publicationQualified !== false ||
      Object.keys(fetched.archives || {}).length !== 42) {throw new Error("Windows host-SDK acquisition receipt is not the exact build-only cohort");}
  const composeBudget = windowsHostSdkDeadlineBudget(state.deadlineEpoch, "compose");
  const timeoutIdentity = runNodeProcessGroup("scripts/prepare-native-windows-sdk.js", [state.archiveRoot, state.sdkOutput], environment, composeBudget.timeoutMs);
  const resultFile = path.join(state.sdkOutput, "result.json"), descriptorFile = path.join(state.sdkOutput, "sdk.json"),
    overlayFile = path.join(state.sdkOutput, "vfs-overlay.json"), filesFile = path.join(state.sdkOutput, "files.json"),
    result = /** @type {{status:string,target:string,buildOnly:boolean,compiled:boolean,executed:boolean,publicationQualified:boolean,
      sourceCatalogPublisherAuthenticated:boolean,sourceCatalogBindingVerified:boolean,packageCount:number,cabinetCount:number,
      selectedFileCount:number,tools:unknown[],fetchReceipt:{path:string,sha256:string},sourceCatalog:{sha256:string},roots:string[],
      descriptor:{path:string,sha256:string},overlay:{path:string,sha256:string},provenance:{path:string,sha256:string}}} */ (readBuildJson(resultFile)),
    descriptor = /** @type {{sourceRoot:string,overlay:string,includeRoots:string[],libraryRoots:string[]}} */ (readBuildJson(descriptorFile)),
    files = /** @type {{files:Array<{path:string,sha256:string,bytes:number}>,origins:unknown[]}} */ (readBuildJson(filesFile)), sdkRoot = path.join(state.sdkOutput, "sdk");
  for (const file of [resultFile, descriptorFile, overlayFile, filesFile]) {
    const stat = fs.lstatSync(file); if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Windows host-SDK composition output is not a regular file");}
  }
  const descriptorKeys = ["includeRoots", "libraryRoots", "overlay", "sourceRoot"];
  if (!result || result.status !== "PASS" || result.target !== "windows-host" || result.buildOnly !== true || result.compiled !== false ||
      result.executed !== false || result.publicationQualified !== false || result.sourceCatalogPublisherAuthenticated !== false ||
      result.sourceCatalogBindingVerified !== false || result.packageCount !== Object.keys(lock.packages).length ||
      result.cabinetCount !== Object.keys(lock.cabinets).length || !Number.isSafeInteger(result.selectedFileCount) || result.selectedFileCount < 1 ||
      result.fetchReceipt?.path !== fetchReceiptPath || result.fetchReceipt.sha256 !== fetchReceiptSha256 ||
      result.sourceCatalog?.sha256 !== lock.sourceCatalog.sha256 || !isDeepStrictEqual(result.roots, lock.roots) ||
      result.descriptor?.path !== descriptorFile || result.descriptor.sha256 !== fileHash(descriptorFile) ||
      result.overlay?.path !== overlayFile || result.overlay.sha256 !== fileHash(overlayFile) ||
      result.provenance?.path !== filesFile || result.provenance.sha256 !== fileHash(filesFile) ||
      !descriptor || Object.keys(descriptor).sort().join("\0") !== descriptorKeys.sort().join("\0") || descriptor.sourceRoot !== sdkRoot ||
      descriptor.overlay !== overlayFile || !Array.isArray(descriptor.includeRoots) || !descriptor.includeRoots.length ||
      !Array.isArray(descriptor.libraryRoots) || !descriptor.libraryRoots.length || !files || !Array.isArray(files.files) ||
      files.files.length !== result.selectedFileCount || !files.origins || !Array.isArray(files.origins)) {
    throw new Error("Windows host-SDK composition receipt or descriptor fails the source-only contract");
  }
  const seen = new Map();
  for (const row of files.files) {
    if (!row || !safeRelative(row.path) || !/^[0-9a-f]{64}$/.test(row.sha256 || "") || !Number.isSafeInteger(row.bytes) || row.bytes < 0 || seen.has(row.path)) {
      throw new Error("Windows host-SDK per-file provenance is malformed");
    }
    seen.set(row.path, row);
  }
  const actual = require("./prepare-native-windows-sdk").walkTree(sdkRoot);
  if (actual.files.length !== seen.size || actual.bytes !== files.files.reduce((sum, row) => sum + row.bytes, 0)) {
    throw new Error("Windows host-SDK composed tree differs from its selected-file provenance");
  }
  for (const file of actual.files) {
    const row = seen.get(file.path); if (!row || row.bytes !== file.bytes || fileHash(file.full) !== row.sha256) {throw new Error("Windows host-SDK composed file changed after extraction");}
  }
  verifyWindowsHostSdkArchives(state.archiveRoot);
  verifySourceInputs(recipe, sourceInputs(repo, [], recipeLeaves), "Windows host-SDK CI source/composition recipe");
  const retained = retainWindowsHostSdkMetadata(state, fetchReceiptSha256), metadataBudget = windowsHostSdkDeadlineBudget(state.deadlineEpoch, "package");
  const handoffFile = path.join(state.sdkOutput, "handoff.json"), handoff = {version: 1, format: "mom-windows-host-sdk-ci-metadata-v1",
    target: "windows-host", status: "PASS", sourceOnly: true, buildOnly: true, compiled: false, executed: false,
    publicationQualified: false, sourceCatalogPublisherAuthenticated: false, sourceCatalogBindingVerified: false,
    vendorPayloadIncluded: false, sourceLockSha256: fileHash(path.join(repo, "runtime/windows-host-sdk.json")),
    archiveCount: Object.keys(fetched.archives).length, packageCount: result.packageCount, cabinetCount: result.cabinetCount,
    selectedFileCount: result.selectedFileCount, selectedRoots: lock.roots,
    acquisition: {receipt: "source-fetch.json", sha256: fetchReceiptSha256},
    composition: {receipt: "result.json", sha256: fileHash(resultFile), descriptor: "sdk.json", descriptorSha256: fileHash(descriptorFile),
      overlaySha256: fileHash(overlayFile), files: "files.json", filesSha256: fileHash(filesFile)},
    extractionTools: result.tools, timeoutTool: timeoutIdentity, note: "Only source-selection metadata and hashes are retained. Vendor archives and SDK payload files remain runner-local and are not uploaded."};
  fs.writeFileSync(handoffFile, JSON.stringify(handoff, null, 2) + "\n", {flag: "wx", mode: 0o444});
  if (fileHash(retained.receiptTarget) !== fetchReceiptSha256 || fileHash(retained.lockTarget) !== handoff.sourceLockSha256) {
    throw new Error("Windows host-SDK retained source metadata changed");
  }
  const metadataHashes = /** @type {Array<[string,string]>} */ (["sdk.json", "vfs-overlay.json", "files.json", "result.json", "source-fetch.json", "source-lock.json", "handoff.json"]
    .map(name => {const file = path.join(state.sdkOutput, name); return [file, fileHash(file)];}));
  const members = windowsHostSdkPackageMembers(state), packed = packState(state.metadataTar, state.workRoot, members, metadataBudget.timeoutMs),
    listed = listState(state.metadataTar, [state.outputName]), allowed = new Set(["sdk.json", "vfs-overlay.json", "files.json", "result.json", "source-fetch.json", "source-lock.json", "handoff.json"]);
  for (const member of listed) {
    const normalized = member.endsWith("/") ? member.slice(0, -1) : member;
    if (normalized === state.outputName) {continue;}
    const prefix = state.outputName + "/"; if (!normalized.startsWith(prefix)) {throw new Error("Windows host-SDK metadata tar has an unexpected root");}
    const relative = normalized.slice(prefix.length); if (!allowed.has(relative) || relative.includes("/")) {throw new Error("Windows host-SDK tar contains vendor payload or unexpected metadata");}
  }
  if (packed.bytes > 64 * 1024 * 1024 || metadataHashes.some(([file, hash]) => fileHash(file) !== hash)) {
    throw new Error("Windows host-SDK metadata tar is excessive or its receipts changed while packing");
  }
  verifySourceInputs(recipe, sourceInputs(repo, [], recipeLeaves), "Windows host-SDK CI source/composition final verification");
  const bundleSha256 = fileHash(state.metadataTar), output = {metadata_bundle_sha256: bundleSha256,
    metadata_bundle_bytes: packed.bytes, sdk_receipt_sha256: fileHash(resultFile), descriptor_sha256: fileHash(descriptorFile),
    files_provenance_sha256: fileHash(filesFile), handoff_sha256: fileHash(handoffFile), selected_file_count: result.selectedFileCount};
  appendWindowsHostSdkOutputs(outputFile, output);
  fs.appendFileSync(summary, `### Windows host SDK source/composition (metadata only)\n- Acquired and composed ${result.selectedFileCount} selected files from the exact 42-archive lock; raw vendor archives and SDK payloads remained runner-local.\n- The retained artifact contains only receipts, descriptor/overlay, and per-file provenance; no compiler, application, runtime, publication, or publisher-authentication claim is made.\n`, "utf8");
  return output;
}

/** @param {Pick<ReturnType<typeof windowsHostSdkStatePaths>, "outputName">} state */
function windowsHostSdkPackageMembers(state) {
  return ["sdk.json", "vfs-overlay.json", "files.json", "result.json", "source-fetch.json", "source-lock.json", "handoff.json"]
    .map(name => state.outputName + "/" + name);
}

/** Validate the explicit link-order policy before opening any of its input paths.
 * @param {unknown} value @returns {{msvcLibraries:string[],ucrtLibraries:string[],importLibraries:string[]}} */
function validateWindowsLlvmLibraryPolicy(value) {
  const groups = ["msvcLibraries", "ucrtLibraries", "importLibraries"];
  if (!hasExactKeys(value, groups)) {throw new Error("Windows LLVM library policy must name all ordered library groups");}
  const seen = new Set(), policy = /** @type {Record<string,unknown>} */ (value), result = /** @type {Record<string,string[]>} */ ({});
  for (const group of groups) {
    const names = policy[group];
    if (!Array.isArray(names) || names.length < 1 || names.length > 256) {throw new Error("Windows LLVM library policy group is empty or excessive: " + group);}
    result[group] = names.map(name => {
      if (typeof name !== "string" || !/^[A-Za-z0-9_.+-]{1,128}\.lib$/i.test(name) ||
          safeWindowsSdkRelative(name) !== name || name.includes("/") || seen.has(name.toLowerCase())) {
        throw new Error("Windows LLVM library policy contains an unsafe or duplicate basename");
      }
      seen.add(name.toLowerCase()); return name;
    });
  }
  return /** @type {{msvcLibraries:string[],ucrtLibraries:string[],importLibraries:string[]}} */ (result);
}

/** Hash-pin a regular Linux tool file; permit only the explicitly declared sibling alias.
 * @param {{path:string,sha256:string,linkTarget?:string}} pin @param {string} label */
function auditWindowsLlvmToolFile(pin, label) {
  const keys = pin && Object.hasOwn(pin, "linkTarget") ? ["path", "sha256", "linkTarget"] : ["path", "sha256"];
  if (!hasExactKeys(pin, keys) || !path.isAbsolute(pin.path) || /[\r\n\0]/.test(pin.path) || !/^[a-f0-9]{64}$/.test(pin.sha256)) {
    throw new Error("Windows LLVM tool pin is malformed: " + label);
  }
  const stat = fs.lstatSync(pin.path); let file = pin.path;
  if (stat.isSymbolicLink()) {
    if (typeof pin.linkTarget !== "string" || !/^[A-Za-z0-9_.+-]{1,128}$/.test(pin.linkTarget) || fs.readlinkSync(pin.path) !== pin.linkTarget) {
      throw new Error("Windows LLVM tool alias differs from its pin: " + label);
    }
    file = path.join(path.dirname(pin.path), pin.linkTarget);
  } else if (Object.hasOwn(pin, "linkTarget")) {throw new Error("Windows LLVM tool pin declares a missing alias: " + label);}
  const target = fs.lstatSync(file);
  if (!target.isFile() || target.isSymbolicLink() || target.nlink !== 1 || fs.realpathSync(file) !== path.resolve(file) ||
      target.size < 1 || target.size > (512 << 20) || fileHash(file) !== pin.sha256) {
    throw new Error("Windows LLVM tool bytes or canonical path differ from their pin: " + label);
  }
  return {path: pin.path, resolvedPath: file, bytes: target.size, sha256: pin.sha256,
    ...(Object.hasOwn(pin, "linkTarget") ? {linkTarget: pin.linkTarget} : {})};
}

/** Map one ordered basename list to exactly one selected, manifest-bound SDK archive per name.
 * @param {string[]} names @param {string[]} roots @param {Map<string,any>} files @param {string} sdkRoot @param {string} label */
function mapWindowsLlvmSdkLibraries(names, roots, files, sdkRoot, label) {
  return names.map(name => {
    const candidates = roots.map(root => root + "/" + name), found = candidates.map(candidate => files.get(candidate.normalize("NFC").toLowerCase())).filter(Boolean);
    if (found.length !== 1) {throw new Error("Windows LLVM " + label + " library must resolve to exactly one source file: " + name);}
    const row = found[0]; if (!row) {throw new Error("Windows LLVM " + label + " library was not found: " + name);}
    const file = path.join(sdkRoot, ...row.path.split("/")), stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== file ||
        stat.size !== row.bytes || fileHash(file) !== row.sha256 || !isStaticArchive(file)) {
      throw new Error("Windows LLVM selected " + label + " archive changed or is not regular: " + name);
    }
    return {path: file, manifestPath: row.path, bytes: stat.size, sha256: row.sha256,
      origins: row.origins.map((/** @type {any} */ origin) => ({package: origin.package, packageSha256: origin.packageSha256, member: origin.member}))};
  });
}

/** Read one small externally pinned JSON receipt without permitting a redirected or changing file.
 * @param {string} file @param {string} expectedSha256 @param {string} label @returns {any} */
function readWindowsLlvmPinnedJson(file, expectedSha256, label) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== path.resolve(file) ||
      stat.size < 1 || stat.size > (8 << 20) || fileHash(file) !== expectedSha256) {
    throw new Error("Windows LLVM receipt differs from its external pin: " + label);
  }
  const value = readBuildJson(file);
  if (fileHash(file) !== expectedSha256) {throw new Error("Windows LLVM receipt changed while reading: " + label);}
  return value;
}

/** Compose only the descriptor used by the maintained Windows LLVM builder. The host composer
 * remains source-only; this function neither builds LLVM nor asserts publisher authentication.
 * @param {{windowsSdkRoot:string,windowsSdkPins:{resultSha256:string,descriptorSha256:string,filesSha256:string,overlaySha256:string},
 * hermesRoot:string,hermesExpectation:{kind:"hermes",handoffSha256:string},
 * tools:{clangCl:{path:string,sha256:string},lldLink:{path:string,sha256:string,linkTarget?:string},llvmNm:{path:string,sha256:string},clangResourceInclude:string},
 * libraryPolicy?:{msvcLibraries:string[],ucrtLibraries:string[],importLibraries:string[]}}} input
 * @returns {{descriptor:Record<string,unknown>,provenance:Record<string,unknown>}} */
function composeWindowsLlvmSdkDescriptor(input) {
  return composeWindowsLlvmDescriptor(input, []);
}

/** Local builds retain the same SDK/tool/runtime admission without inventing CI provenance.
 * @param {Omit<Parameters<typeof composeWindowsLlvmSdkDescriptor>[0],"hermesExpectation"> &
 * {hermesExpectation:{kind:"hermes",localBuildId:string,handoffSha256:string}}} input */
function composeLocalWindowsLlvmSdkDescriptor(input) {
  return composeWindowsLlvmDescriptor(input, ["localBuildId"]);
}

/** Only the transport entrypoint differs; the descriptor and source policy stay shared.
 * @param {Parameters<typeof composeWindowsLlvmSdkDescriptor>[0] | Parameters<typeof composeLocalWindowsLlvmSdkDescriptor>[0]} input
 * @param {string[]} identityKeys */
function composeWindowsLlvmDescriptor(input, identityKeys) {
  const pinKeys = ["resultSha256", "descriptorSha256", "filesSha256", "overlaySha256"], toolKeys = ["clangCl", "lldLink", "llvmNm", "clangResourceInclude"],
    policy = validateWindowsLlvmLibraryPolicy(input?.libraryPolicy === undefined ? defaultWindowsNativeLibraryPolicy() : input.libraryPolicy);
  if (!input || !path.isAbsolute(input.windowsSdkRoot) || /[\r\n\0]/.test(input.windowsSdkRoot) ||
      !hasExactKeys(input.windowsSdkPins, pinKeys) || Object.values(input.windowsSdkPins).some(hash => !/^[a-f0-9]{64}$/.test(hash)) ||
      !hasExactKeys(input.hermesExpectation, ["kind", ...identityKeys, "handoffSha256"]) || input.hermesExpectation.kind !== "hermes" ||
      !hasExactKeys(input.tools, toolKeys) || typeof input.tools.clangResourceInclude !== "string" ||
      !path.isAbsolute(input.tools.clangResourceInclude) || /[\r\n\0]/.test(input.tools.clangResourceInclude)) {
    throw new Error("Windows LLVM descriptor composition requires explicit source pins and tool paths");
  }
  const handoff = "localBuildId" in input.hermesExpectation ?
    windowsLocalControllerHandoffInputs(input.hermesRoot, input.hermesExpectation) :
    windowsControllerHandoffInputs(input.hermesRoot, input.hermesExpectation);
  if (handoff.kind !== "hermes") {throw new Error("Windows LLVM descriptor requires a Hermes handoff");}
  const hermes = handoff, intelRuntime = hermes.intelRuntime;
  if (!intelRuntime) {throw new Error("Windows LLVM descriptor requires Intel runtime inputs in the Hermes handoff");}
  const {root, sdkRoot, resultPath, sdkPath, filesPath, overlayPath, sdk, files, actualTree} =
    require("./native-windows-host-sdk-inputs").windowsHostSdkInputs(input.windowsSdkRoot, input.windowsSdkPins);

  const toolProof = {clangCl: auditWindowsLlvmToolFile(input.tools.clangCl, "clangCl"),
      lldLink: auditWindowsLlvmToolFile(input.tools.lldLink, "lldLink"), llvmNm: auditWindowsLlvmToolFile(input.tools.llvmNm, "llvmNm")},
    resource = input.tools.clangResourceInclude, resourceStat = fs.lstatSync(resource);
  if (!resourceStat.isDirectory() || resourceStat.isSymbolicLink() || fs.realpathSync(resource) !== path.resolve(resource)) {
    throw new Error("Windows LLVM Clang resource include directory is not canonical");
  }
  const compilerRoot = path.resolve(path.dirname(input.tools.clangCl.path), ".."), relativeResource = path.relative(compilerRoot, resource);
  if (path.isAbsolute(relativeResource) || relativeResource === ".." || relativeResource.startsWith(".." + path.sep)) {
    throw new Error("Windows LLVM resource headers are outside the pinned compiler root");
  }
  const runtimeCatalog = windowsIntelRuntimeCatalog.libraries, runtimeByName = new Map(runtimeCatalog.map(row => [row.name.toLowerCase(), row])),
    runtimeMap = intelRuntime.libraries;
  if (!isDeepStrictEqual(Object.keys(runtimeMap).sort(), runtimeCatalog.map(row => row.name).sort())) {
    throw new Error("Windows Hermes handoff Intel library set differs from its shared pin catalog");
  }
  const msvcLibraries = /** @type {any[]} */ (policy.msvcLibraries.map((/** @type {string} */ name) => {
    const pin = runtimeByName.get(name.toLowerCase());
    if (!pin) {
      const row = mapWindowsLlvmSdkLibraries([name], ["msvc/lib"], files, sdkRoot, "MSVC")[0];
      if (!row) {throw new Error("Windows LLVM MSVC library is absent: " + name);} return row;
    }
    const file = runtimeMap[pin.name], expectedPath = path.join(path.resolve(input.hermesRoot), "intel-runtime", "lib", pin.name);
    if (!file) {throw new Error("Windows Intel runtime archive is absent from its handoff: " + pin.name);}
    const stat = fs.lstatSync(file);
    if (file !== expectedPath || !stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== file ||
        stat.size !== pin.bytes || fileHash(file) !== pin.sha256 || !isStaticArchive(file)) {
      throw new Error("Windows Intel runtime archive differs from the admitted Hermes handoff: " + pin.name);
    }
    return {path: file, manifestPath: "intel-runtime/lib/" + pin.name, bytes: stat.size, sha256: pin.sha256,
      origin: {handoffSha256: hermes.handoffSha256, catalog: "runtime/windows-intel-runtime.json"}};
  }));
  if (runtimeCatalog.some(row => !policy.msvcLibraries.some(name => name.toLowerCase() === row.name.toLowerCase()))) {
    throw new Error("Windows LLVM MSVC policy omits a pinned Intel runtime library");
  }
  const ucrtLibraries = mapWindowsLlvmSdkLibraries(policy.ucrtLibraries, ["msvc/lib", "sdk/lib/ucrt"], files, sdkRoot, "UCRT"),
    importLibraries = mapWindowsLlvmSdkLibraries(policy.importLibraries, ["sdk/lib/um"], files, sdkRoot, "Windows import"),
    intelLibraryRoot = intelRuntime.libRoot;
  const expectedSdkLibraryRoots = ["sdk/lib/ucrt", "sdk/lib/um", "msvc/lib"].map(relative => path.join(sdkRoot, ...relative.split("/")));
  if (!Array.isArray(sdk.includeRoots) || !Array.isArray(sdk.libraryRoots) || sdk.libraryRoots.length !== 3 ||
      !isDeepStrictEqual(sdk.libraryRoots, expectedSdkLibraryRoots) ||
      !sdk.libraryRoots.every((/** @type {string} */ rootPath) => path.isAbsolute(rootPath) && fs.lstatSync(rootPath).isDirectory() &&
        !fs.lstatSync(rootPath).isSymbolicLink() && fs.realpathSync(rootPath) === rootPath) ||
      sdk.libraryRoots.some((/** @type {string} */ rootPath) => rootPath.toLowerCase() === intelLibraryRoot.toLowerCase())) {
    throw new Error("Windows host-SDK library roots are redirected, incomplete or collide with Intel runtime");
  }
  const descriptor = {version: 1, tools: {clangCl: input.tools.clangCl.path, lldLink: input.tools.lldLink.path,
    llvmNm: input.tools.llvmNm.path, clangResourceInclude: resource}, windowsSdk: {sourceRoot: sdk.sourceRoot, overlay: sdk.overlay,
    includeRoots: sdk.includeRoots, libraryRoots: [...sdk.libraryRoots, intelLibraryRoot],
    msvcLibraries: msvcLibraries.map(row => row.path), ucrtLibraries: ucrtLibraries.map(row => row.path),
    importLibraries: importLibraries.map(row => row.path)}};
  require("./build-native-windows-controller").validateWindowsLlvmSdkConfig(descriptor);
  const finalSourcePins = /** @type {Array<[string,string]>} */ ([[resultPath, input.windowsSdkPins.resultSha256],
    [sdkPath, input.windowsSdkPins.descriptorSha256], [filesPath, input.windowsSdkPins.filesSha256], [overlayPath, input.windowsSdkPins.overlaySha256]]);
  for (const [file, hash] of finalSourcePins) {
    if (fileHash(file) !== hash) {throw new Error("Windows LLVM descriptor source changed during composition: " + path.basename(file));}
  }
  return {descriptor, provenance: {version: 1, role: "windows-llvm21-descriptor-input-map", status: "PASS", buildOnly: true,
    compiled: false, executed: false, publicationQualified: false,
    sourceTrust: {windowsCatalogPublisherAuthenticated: false, windowsChannelCatalogBindingVerified: false},
    windowsSdk: {root, resultSha256: input.windowsSdkPins.resultSha256, descriptorSha256: input.windowsSdkPins.descriptorSha256,
      filesSha256: input.windowsSdkPins.filesSha256, overlaySha256: input.windowsSdkPins.overlaySha256,
      selectedFileCount: actualTree.files.length, selectedBytes: actualTree.bytes},
    hermes: {root: path.resolve(input.hermesRoot),
      ...("localBuildId" in hermes ? {localBuildId: hermes.localBuildId} : {}), handoffSha256: hermes.handoffSha256,
      intelRuntime: {libraries: windowsIntelRuntimeCatalog.libraries.map(row => ({name: row.name, path: intelRuntime.libraries[row.name],
        manifestPath: row.path, bytes: row.bytes, sha256: row.sha256})),
      licenses: windowsIntelRuntimeCatalog.licenses.map(row => ({path: intelRuntime.licenses[row.path], manifestPath: row.path,
        bytes: row.bytes, sha256: row.sha256}))}},
    tools: {...toolProof, clangResourceInclude: resource}, libraryPolicy: policy,
    libraries: {msvcLibraries, ucrtLibraries, importLibraries}}};
}

/** Fixed run-scoped outputs for composing the Windows LLVM descriptor before configure.
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {string} [architecture] */
function windowsNativeLlvmDescriptorCiStatePaths(environment = process.env, platform = process.platform, architecture = process.arch) {
  const ci = windowsNativeLlvmCiStatePaths(environment, platform, architecture),
    root = path.join(ci.runnerTemp, "mom-native-windows-llvm21-descriptor-" + ci.suffix),
    windowsSdkRoot = path.join(ci.runnerTemp, "mom-native-windows-host-sdk-" + ci.suffix, "native-windows-host-sdk-" + ci.suffix),
    hermesRoot = path.join(ci.runnerTemp, "native-windows-controller-hermes-" + ci.suffix);
  if (environment["MOM_NATIVE_WINDOWS_LLVM_DESCRIPTOR_DIR"] !== root || path.dirname(root) !== ci.runnerTemp ||
      /[\r\n\0]/.test(ci.runnerTemp)) {
    throw new Error("Windows LLVM descriptor output must match the exact same-run path");
  }
  return {...ci, configureRoot: ci.root, root, descriptorOutput: path.join(root, "descriptor.json"), provenanceOutput: path.join(root, "provenance.json"),
    windowsSdkRoot, hermesRoot};
}

/** Validate a pinned composer input without mutating its source or configure roots.
 * @param {string[]} args @param {NodeJS.ProcessEnv} environment @param {NodeJS.Platform} platform @param {string} architecture @param {number} nowMs */
function windowsNativeLlvmDescriptorCiPreflight(args, environment = process.env, platform = process.platform, architecture = process.arch, nowMs = Date.now()) {
  if (args.length !== 2) {throw new Error("Windows LLVM descriptor stage needs one input JSON and its SHA-256 pin");}
  const state = windowsNativeLlvmDescriptorCiStatePaths(environment, platform, architecture),
    reports = windowsNativeLlvmCiReports(environment), budget = windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "configure", nowMs, state.runDeadlineEpoch),
    inputFile = args[0] || "", inputSha256 = args[1] || "";
  if (!/^[a-f0-9]{64}$/.test(inputSha256) || !path.isAbsolute(inputFile) || /[\r\n\0]/.test(inputFile) ||
      !windowsNativeLlvmCiContains(state.runnerTemp, inputFile) || path.resolve(fs.realpathSync(inputFile)) !== path.resolve(inputFile)) {
    throw new Error("Windows LLVM descriptor input must be a canonical pinned same-run JSON file");
  }
  const inputStat = windowsControllerRegularFile(inputFile, "Windows LLVM descriptor composition input");
  if (inputStat.nlink !== 1 || inputStat.size < 1 || inputStat.size > (8 << 20) || fileHash(inputFile) !== inputSha256) {
    throw new Error("Windows LLVM descriptor composition input differs from its external SHA-256 pin");
  }
  const input = readWindowsLlvmPinnedJson(inputFile, inputSha256, "descriptor composition input"),
    required = ["windowsSdkRoot", "windowsSdkPins", "hermesRoot", "hermesExpectation", "tools"], keys = Object.keys(input || {}).sort().join("\0");
  if (keys !== required.sort().join("\0") && keys !== [...required, "libraryPolicy"].sort().join("\0")) {
    throw new Error("Windows LLVM descriptor input has an unexpected composer schema");
  }
  if (input.windowsSdkRoot !== state.windowsSdkRoot || input.hermesRoot !== state.hermesRoot ||
      !hasExactKeys(input.hermesExpectation, ["kind", "handoffSha256"]) || input.hermesExpectation.kind !== "hermes" ||
      !hasExactKeys(input.tools, ["clangCl", "lldLink", "llvmNm", "clangResourceInclude"])) {
    throw new Error("Windows LLVM descriptor SDK/Hermes inputs must be the exact same-run roots and handoff");
  }
  const toolPins = [
    ["clangCl", input.tools.clangCl, ["path", "sha256"]],
    ["lldLink", input.tools.lldLink, Object.hasOwn(input.tools.lldLink || {}, "linkTarget") ? ["path", "sha256", "linkTarget"] : ["path", "sha256"]],
    ["llvmNm", input.tools.llvmNm, ["path", "sha256"]],
  ];
  for (const [label, pin, pinKeys] of toolPins) {
    if (!hasExactKeys(pin, pinKeys)) {throw new Error("Windows LLVM descriptor tool pin is malformed: " + label);}
  }
  if (typeof input.tools.clangResourceInclude !== "string" || !path.isAbsolute(input.tools.clangResourceInclude) ||
      /[\r\n\0]/.test(input.tools.clangResourceInclude)) {throw new Error("Windows LLVM Clang resource include path is malformed");}
  const toolIdentities = toolPins.map(([label, pin]) => auditWindowsLlvmToolFile(pin, label));
  const resourceStat = fs.lstatSync(input.tools.clangResourceInclude);
  if (!resourceStat.isDirectory() || resourceStat.isSymbolicLink() || fs.realpathSync(input.tools.clangResourceInclude) !== path.resolve(input.tools.clangResourceInclude)) {
    throw new Error("Windows LLVM Clang resource include directory is not canonical");
  }
  const inputRoots = [repo, inputFile, state.windowsSdkRoot, state.hermesRoot, ...toolIdentities.map(row => path.dirname(row.resolvedPath)),
    input.tools.clangResourceInclude];
  windowsNativeLlvmCiRejectReportInputOverlap(reports, [...inputRoots, state.root, state.configureRoot, state.output, state.descriptor]);
  const overlapsOutput = inputRoots.some(item => path.isAbsolute(item) &&
    windowsNativeLlvmCiContains(path.resolve(item), state.root) || windowsNativeLlvmCiContains(state.root, path.resolve(item))) ||
    windowsNativeLlvmCiContains(state.root, state.configureRoot) || windowsNativeLlvmCiContains(state.configureRoot, state.root);
  const outputStat = fs.lstatSync(state.root, {throwIfNoEntry: false});
  if (overlapsOutput || outputStat || [state.configureRoot, state.output, state.descriptor].some(file => fs.existsSync(file))) {
    throw new Error("Windows LLVM descriptor output must be fresh and separate from its inputs and configure state");
  }
  return {state, reports, budget, inputFile, inputSha256, inputStat, input, toolPins, toolIdentities};
}

/** Compose and retain the build-only descriptor through the existing full SDK/handoff admission.
 * The composer callback is an internal test seam; the CLI always supplies the maintained composer.
 * @param {string[]} args @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform]
 * @param {string} [architecture] @param {number} [nowMs]
 * @param {(input:any)=>{descriptor:Record<string,unknown>,provenance:Record<string,unknown>}} [composer] */
function buildWindowsNativeLlvmDescriptorCi(args, environment = process.env, platform = process.platform, architecture = process.arch,
  nowMs = Date.now(), composer = composeWindowsLlvmSdkDescriptor) {
  const preflight = windowsNativeLlvmDescriptorCiPreflight(args, environment, platform, architecture, nowMs),
    {state, reports, inputFile, inputSha256, input, toolPins, toolIdentities} = preflight,
    composed = composer(input);
  if (!composed || !composed.descriptor || typeof composed.descriptor !== "object" || Array.isArray(composed.descriptor) ||
      !composed.provenance || typeof composed.provenance !== "object" || Array.isArray(composed.provenance)) {
    throw new Error("Windows LLVM composer returned an invalid descriptor/provenance pair");
  }
  const verifyInputs = () => {
    const currentInput = windowsControllerRegularFile(inputFile, "Windows LLVM descriptor composition input");
    if (currentInput.nlink !== 1 || currentInput.dev !== preflight.inputStat.dev || currentInput.ino !== preflight.inputStat.ino ||
        currentInput.size !== preflight.inputStat.size || fileHash(inputFile) !== inputSha256) {
      throw new Error("Windows LLVM descriptor composition input changed during production");
    }
    const currentTools = toolPins.map(([label, pin]) => auditWindowsLlvmToolFile(pin, label));
    if (!isDeepStrictEqual(currentTools, toolIdentities)) {throw new Error("Windows LLVM descriptor tool identity changed during production");}
  };
  verifyInputs();
  windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "configure", Date.now(), state.runDeadlineEpoch);
  const descriptorBytes = Buffer.from(JSON.stringify(composed.descriptor, null, 2) + "\n"), descriptorSha256 = createHash("sha256").update(descriptorBytes).digest("hex"),
    provenance = {version: 1, format: "mom-windows-llvm21-descriptor-ci-v1", status: "PASS",
      deadlineEpoch: state.deadlineEpoch, buildOnly: true, compiled: false, executed: false, publicationQualified: false,
      input: {path: inputFile, sha256: inputSha256}, descriptor: {path: state.descriptorOutput, sha256: descriptorSha256}, composition: composed.provenance},
    provenanceBytes = Buffer.from(JSON.stringify(provenance, null, 2) + "\n");
  if (descriptorBytes.length > (4 << 20) || provenanceBytes.length > (8 << 20)) {throw new Error("Windows LLVM descriptor provenance exceeds its handoff size bound");}
  fs.mkdirSync(state.root, {mode: 0o700});
  fs.writeFileSync(state.descriptorOutput, descriptorBytes, {flag: "wx", mode: 0o444});
  fs.writeFileSync(state.provenanceOutput, provenanceBytes, {flag: "wx", mode: 0o444});
  const verified = composer(input);
  verifyInputs();
  windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "configure", Date.now(), state.runDeadlineEpoch);
  if (!isDeepStrictEqual(verified, composed) || fileHash(state.descriptorOutput) !== descriptorSha256 ||
      fileHash(state.provenanceOutput) !== createHash("sha256").update(provenanceBytes).digest("hex")) {
    throw new Error("Windows LLVM descriptor source, tool or retained output changed during production");
  }
  const values = {windows_llvm_descriptor_path: state.descriptorOutput, windows_llvm_descriptor_sha256: descriptorSha256,
    windows_llvm_descriptor_provenance_path: state.provenanceOutput, windows_llvm_descriptor_provenance_sha256: fileHash(state.provenanceOutput)};
  windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "outputs", Date.now(), state.runDeadlineEpoch);
  for (const report of reports) {verifyWindowsNativeLlvmCiReport(report);}
  appendWorkflowOutputs(environment, values);
  fs.appendFileSync(environment["GITHUB_STEP_SUMMARY"] || "", `### Windows LLVM 21 descriptor inputs\n- Same-run descriptor: ${state.descriptorOutput} (SHA-256 ${descriptorSha256}).\n` +
    `- Provenance: ${state.provenanceOutput} (SHA-256 ${values.windows_llvm_descriptor_provenance_sha256}).\n` +
    "- Build-only: true; compiled: false; executed: false; publication-qualified: false.\n", "utf8");
  return values;
}

const windowsNativeLlvmPackReserveMs = 60 * 1000, windowsNativeLlvmMinPhaseMs = 5 * 60 * 1000,
  windowsNativeLlvmMaxChildMs = 45 * 60 * 1000, windowsNativeLlvmMaxJobMs = 60 * 60 * 1000,
  windowsNativeLlvmMaxRunMs = 8 * 60 * 60 * 1000;

/** The run deadline is set before configure and must not be recreated by a later job.
 * @param {string} epoch @param {number} [nowMs] */
function windowsNativeLlvmCiRunDeadlineBudget(epoch, nowMs = Date.now()) {
  if (!/^\d{10}$/.test(epoch) || !Number.isSafeInteger(nowMs) || nowMs < 0) {
    throw new Error("Windows LLVM CI needs a valid run-start deadline");
  }
  const remainingMs = Number(epoch) * 1000 - nowMs;
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0 || remainingMs > windowsNativeLlvmMaxRunMs) {
    throw new Error("Windows LLVM CI run deadline is expired or exceeds the 8-hour run bound");
  }
  return {remainingMs};
}

/** Keep every LLVM phase invocation on the same caller-supplied job-start deadline.
 * @param {string} epoch @param {"configure" | "build" | "audit" | "outputs"} phase @param {number} [nowMs] @param {string} [runEpoch] */
function windowsNativeLlvmCiDeadlineBudget(epoch, phase, nowMs = Date.now(), runEpoch = "") {
  if (!/^\d{10}$/.test(epoch) || !Number.isSafeInteger(nowMs) || nowMs < 0 ||
      !["configure", "build", "audit", "outputs"].includes(phase)) {
    throw new Error("Windows LLVM CI needs a valid shared job-start deadline");
  }
  const remainingMs = Number(epoch) * 1000 - nowMs;
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0 || remainingMs > windowsNativeLlvmMaxJobMs) {
    throw new Error("Windows LLVM CI deadline is expired or exceeds the 60-minute job bound");
  }
  if (runEpoch) {
    const run = windowsNativeLlvmCiRunDeadlineBudget(runEpoch, nowMs);
    if (Number(epoch) > Number(runEpoch)) {throw new Error("Windows LLVM job deadline exceeds the original run deadline");}
    if (remainingMs > run.remainingMs) {throw new Error("Windows LLVM job deadline exceeds the remaining run limit");}
  }
  if (phase === "outputs") {
    if (remainingMs < 10 * 1000) {throw new Error("Insufficient shared job time remains to report Windows LLVM phase outputs");}
    return {timeoutMs: 0, remainingMs};
  }
  if (remainingMs < windowsNativeLlvmMinPhaseMs + windowsNativeLlvmPackReserveMs) {
    throw new Error("Insufficient shared job time remains for the Windows LLVM phase and final status reporting");
  }
  return {timeoutMs: Math.min(windowsNativeLlvmMaxChildMs, remainingMs - windowsNativeLlvmPackReserveMs), remainingMs};
}

/** @typedef {{suffix:string, runnerTemp:string, root:string, output:string, descriptor:string, deadlineRecord:string, result:string,
 * deadlineEpoch:string, jobId:string, runDeadlineEpoch:string, runDeadlineRecord:string}} WindowsNativeLlvmCiState */

/** Fixed same-run paths keep the builder's absolute receipts valid for each in-job phase call.
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {string} [architecture]
 * @returns {WindowsNativeLlvmCiState} */
function windowsNativeLlvmCiStatePaths(environment = process.env, platform = process.platform, architecture = process.arch) {
  const runnerTemp = environment["RUNNER_TEMP"] || "";
  const runDeadlineEpoch = environment["MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH"] || "", jobId = environment["GITHUB_JOB"] || "";
  if (platform !== "linux" || architecture !== "x64" || !path.isAbsolute(runnerTemp) ||
      environment["MOM_AOT_BUILD_ONLY"] !== "1" || environment["MOM_BUILD_JOBS"] !== ciBuildJobs(environment)) {
    throw new Error("Windows LLVM CI needs a Linux x64 runner, build-only mode and a valid job count");
  }
  if (!runDeadlineEpoch || !/^[A-Za-z0-9_-]{1,64}$/.test(jobId)) {
    throw new Error("Windows LLVM CI needs its original run deadline and a safe bounded GITHUB_JOB identity");
  }
  const canonicalTemp = fs.realpathSync(runnerTemp), suffix = ciSuffix(environment),
    root = path.join(temp, "mom-native-windows-llvm21-" + suffix), output = path.join(root, "output"),
    descriptor = path.join(root, "descriptor.json");
  if (canonicalTemp !== path.resolve(runnerTemp) || !fs.lstatSync(canonicalTemp).isDirectory() ||
      fs.realpathSync(temp) !== temp || environment["MOM_NATIVE_WINDOWS_LLVM_DIR"] !== root ||
      path.dirname(root) !== temp || path.resolve(root) !== root) {
    throw new Error("Windows LLVM CI paths must match the CI work-root plan");
  }
  return {suffix, runnerTemp: canonicalTemp, root, output, descriptor, result: path.join(output, "result.json"),
    deadlineEpoch: environment["MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH"] || "", jobId, runDeadlineEpoch,
    deadlineRecord: path.join(root, "ci-job-deadline-" + jobId + ".json"), runDeadlineRecord: path.join(root, "ci-run-deadline.json")};
}

/** Bind each phase call to the run deadline recorded at configure and to its job's first deadline,
 * without changing public builder receipts. Returns null when the job record is absent and not created.
 * @param {ReturnType<typeof windowsNativeLlvmCiStatePaths>} state @param {boolean} [create] */
function windowsNativeLlvmCiDeadlineRecord(state, create = false) {
  const jobId = state.jobId, runDeadlineRecord = state.runDeadlineRecord;
  const readRecord = (/** @type {string} */ file, /** @type {string} */ label) => {
    const stat = windowsControllerRegularFile(file, label);
    if (stat.nlink !== 1 || stat.size > 1024) {throw new Error(label + " is aliased or invalid");}
    return windowsControllerReadJson(file, 1024, label);
  };
  let runRecord;
  if (!fs.existsSync(runDeadlineRecord)) {
    if (!create || fs.existsSync(state.output)) {throw new Error("Windows LLVM cross-job state is missing its original run deadline record");}
    runRecord = {version: 1, runDeadlineEpoch: state.runDeadlineEpoch};
    fs.writeFileSync(runDeadlineRecord, JSON.stringify(runRecord) + "\n", {flag: "wx", mode: 0o444});
  } else {
    runRecord = readRecord(runDeadlineRecord, "Windows LLVM run deadline record");
    if (runRecord.version !== 1 || runRecord.runDeadlineEpoch !== state.runDeadlineEpoch) {
      throw new Error("Windows LLVM cross-job phase attempted to replace or reset its original run deadline");
    }
  }
  const jobStat = fs.lstatSync(state.deadlineRecord, {throwIfNoEntry: false});
  if (!jobStat) {
    if (!create) {return null;}
    const record = {version: 1, job: jobId,
      deadlineEpoch: state.deadlineEpoch, runDeadlineEpoch: state.runDeadlineEpoch};
    fs.writeFileSync(state.deadlineRecord, JSON.stringify(record) + "\n", {flag: "wx", mode: 0o444});
    return record;
  }
  const record = readRecord(state.deadlineRecord, "Windows LLVM job deadline record");
  if (record.version !== 1 || record.job !== jobId ||
      record.deadlineEpoch !== state.deadlineEpoch || record.runDeadlineEpoch !== state.runDeadlineEpoch) {
    throw new Error("Windows LLVM phase attempted to replace or reset its original job deadline");
  }
  return record;
}

/** Ensure workflow report files cannot redirect phase output or alias each other. @param {NodeJS.ProcessEnv} environment */
function windowsNativeLlvmCiReports(environment) {
  const files = [environment["GITHUB_OUTPUT"] || "", environment["GITHUB_STEP_SUMMARY"] || ""];
  if (files.some(file => !path.isAbsolute(file))) {throw new Error("Windows LLVM CI requires absolute workflow output files");}
  const stats = files.map((file, index) => {
    const stat = windowsControllerRegularFile(file, index === 0 ? "Windows LLVM GitHub output" : "Windows LLVM GitHub summary");
    if (stat.nlink !== 1) {throw new Error("Windows LLVM workflow output must be an unaliased regular file");}
    return {file, dev: stat.dev, ino: stat.ino};
  });
  if (path.resolve(files[0] || "") === path.resolve(files[1] || "")) {throw new Error("Windows LLVM workflow reports must be distinct files");}
  return /** @type {[{file:string,dev:number,ino:number},{file:string,dev:number,ino:number}]} */ (stats);
}

/** Reject a changed workflow file before writing outputs after a long build. @param {{file:string,dev:number,ino:number}} saved */
function verifyWindowsNativeLlvmCiReport(saved) {
  const stat = windowsControllerRegularFile(saved.file, "Windows LLVM workflow report");
  if (stat.nlink !== 1 || stat.dev !== saved.dev || stat.ino !== saved.ino) {
    throw new Error("Windows LLVM workflow report changed during its phase");
  }
}

/** @param {string} root @param {string} child */
function windowsNativeLlvmCiContains(root, child) {
  const relative = path.relative(root, child);
  return relative === "" || (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep));
}

/** Keep workflow report files outside the native inputs, including directory roots. @param {ReturnType<typeof windowsNativeLlvmCiReports>} reports @param {string[]} inputs */
function windowsNativeLlvmCiRejectReportInputOverlap(reports, inputs) {
  for (const report of reports) {
    for (const input of inputs) {
      if (typeof input !== "string" || !path.isAbsolute(input) || /[\r\n\0]/.test(input)) {continue;}
      const resolved = path.resolve(input);
      if (windowsNativeLlvmCiContains(resolved, report.file) || windowsNativeLlvmCiContains(report.file, resolved)) {
        throw new Error("Windows LLVM workflow reports must not overlap pinned inputs");
      }
      const stat = fs.lstatSync(resolved, {throwIfNoEntry: false});
      if (stat && stat.dev === report.dev && stat.ino === report.ino) {
        throw new Error("Windows LLVM workflow reports must not alias pinned inputs");
      }
    }
  }
}

/** @param {ReturnType<typeof import("./build-native-windows-controller").validateWindowsLlvmSdkConfig>} descriptor */
function windowsNativeLlvmCiDescriptorRoots(descriptor) {
  return [repo, descriptor.windowsSdk.sourceRoot, descriptor.windowsSdk.overlay, ...descriptor.windowsSdk.includeRoots,
    ...descriptor.windowsSdk.libraryRoots, path.dirname(descriptor.tools.clangCl), path.dirname(descriptor.tools.lldLink),
    path.dirname(descriptor.tools.llvmNm), descriptor.tools.clangResourceInclude];
}

/** Validate one explicit phase, shared deadline and its immutable same-run state before mutation.
 * Configure args are descriptor/path-SHA, source-archive/path-SHA, CMake/path-SHA, Ninja/path-SHA.
 * @param {string[]} args @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform]
 * @param {string} [architecture] @param {number} [nowMs] */
function windowsNativeLlvmCiPreflight(args, environment = process.env, platform = process.platform, architecture = process.arch, nowMs = Date.now()) {
  const phaseValue = args[0] || "";
  if (!["configure", "build", "audit"].includes(phaseValue) || args.length !== (phaseValue === "configure" ? 9 : phaseValue === "build" ? 2 : 1)) {
    throw new Error("Windows LLVM CI needs configure plus eight pinned inputs, build plus one phase, or audit only");
  }
  const phase = /** @type {"configure" | "build" | "audit"} */ (phaseValue),
    {validateWindowsLlvmSdkConfig} = require("./build-native-windows-controller"),
    {windowsLlvmSdkPlan, buildPhases} = require("./build-native-windows-llvm-sdk");
  const state = windowsNativeLlvmCiStatePaths(environment, platform, architecture), reports = windowsNativeLlvmCiReports(environment),
    budget = windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, phase, nowMs, state.runDeadlineEpoch);
  if (reports.some(report => windowsNativeLlvmCiContains(state.root, report.file))) {
    throw new Error("Windows LLVM CI state and workflow reports must not overlap");
  }
  if (phase !== "configure") {windowsNativeLlvmCiDeadlineRecord(state);}
  /** @param {string} file @param {string} label @param {number} maxBytes */
  const pathStat = (file, label, maxBytes) => {
    const stat = windowsControllerRegularFile(file, label);
    if (stat.nlink !== 1 || stat.size < 1 || stat.size > maxBytes) {throw new Error(label + " is aliased, empty or exceeds its size bound");}
    return stat;
  };
  if (phase === "configure") {
    if (args.length !== 9 || [state.root, state.output, state.descriptor].some(file => fs.existsSync(file))) {
      throw new Error("Windows LLVM configure needs eight pinned inputs and fresh same-run output paths");
    }
    const [descriptorSource = "", descriptorSha256 = "", archive = "", archiveSha256 = "", cmake = "", cmakeSha256 = "", ninja = "", ninjaSha256 = ""] = args.slice(1);
    const pinnedSha = (/** @type {string} */ value, /** @type {string} */ label) => {
      if (!/^[a-f0-9]{64}$/.test(value || "")) {throw new Error("Windows LLVM " + label + " needs a lowercase SHA-256 pin");}
    };
    const regularInput = (/** @type {string} */ file, /** @type {string} */ label, /** @type {number} */ maximum,
      /** @type {boolean} */ underRunnerTemp = false) => {
      if (!path.isAbsolute(file) || /[\r\n\0]/.test(file) || path.resolve(fs.realpathSync(file)).toLowerCase() !== path.resolve(file).toLowerCase() ||
          underRunnerTemp && !windowsNativeLlvmCiContains(state.runnerTemp, file)) {throw new Error(label + " must be a canonical same-run input");}
      return pathStat(file, label, maximum);
    };
    pinnedSha(descriptorSha256 || "", "descriptor"); pinnedSha(archiveSha256 || "", "source archive");
    pinnedSha(cmakeSha256 || "", "CMake"); pinnedSha(ninjaSha256 || "", "Ninja");
    if (archiveSha256 !== llvmArchiveSha256 || path.basename(archive || "") !== llvmArchiveName) {
      throw new Error("Windows LLVM configure requires the exact locked LLVM 21 source archive");
    }
    const descriptorStat = regularInput(descriptorSource || "", "Windows LLVM descriptor", 4 << 20, true),
      archiveStat = regularInput(archive || "", "Pinned LLVM 21 source archive", 2 * 1024 * 1024 * 1024, true),
      cmakeStat = regularInput(cmake || "", "Pinned CMake", 128 << 20), ninjaStat = regularInput(ninja || "", "Pinned Ninja", 128 << 20),
      descriptorBytes = fs.readFileSync(descriptorSource || "");
    const descriptorHash = createHash("sha256").update(descriptorBytes).digest("hex");
    if (descriptorBytes.length !== descriptorStat.size || descriptorHash !== descriptorSha256) {
      throw new Error("Windows LLVM descriptor differs from its external SHA-256 pin");
    }
    windowsNativeLlvmCiRejectReportInputOverlap(reports, [repo, descriptorSource, archive || "", cmake || "", ninja || ""]);
    let descriptorValue;
    try {descriptorValue = JSON.parse(descriptorBytes.toString("utf8"));} catch (error) {
      throw new Error("Pinned Windows LLVM descriptor is invalid JSON", {cause: error});
    }
    const descriptor = validateWindowsLlvmSdkConfig(descriptorValue), tools = {cmake, ninja},
      plan = windowsLlvmSdkPlan(descriptor, archive || "", state.output, tools, 2);
    windowsNativeLlvmCiRejectReportInputOverlap(reports, windowsNativeLlvmCiDescriptorRoots(descriptor));
    const overlapsOutput = [repo, path.dirname(archive || ""), descriptor.windowsSdk.sourceRoot, descriptor.windowsSdk.overlay,
      ...descriptor.windowsSdk.includeRoots, ...descriptor.windowsSdk.libraryRoots, path.dirname(descriptor.tools.clangCl),
      path.dirname(descriptor.tools.lldLink), path.dirname(descriptor.tools.llvmNm), descriptor.tools.clangResourceInclude]
      .some(input => path.resolve(input) === state.root || path.resolve(input).startsWith(state.root + path.sep) ||
        state.root.startsWith(path.resolve(input) + path.sep));
    if (overlapsOutput || descriptorSource === archive || descriptorSource === cmake || descriptorSource === ninja ||
        archive === cmake || archive === ninja || cmake === ninja ||
        fileHash(archive || "") !== archiveSha256 || fileHash(cmake || "") !== cmakeSha256 || fileHash(ninja || "") !== ninjaSha256 ||
        fileHash(descriptorSource || "") !== descriptorSha256) {
      throw new Error("Windows LLVM configure input identity, pin or output isolation changed during preflight");
    }
    const afterBudget = windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "configure", Date.now(), state.runDeadlineEpoch);
    return {phase, state, reports, descriptorSource: descriptorSource || "", descriptorSha256: descriptorSha256 || "",
      descriptorBytes, archive: archive || "", archiveSha256: archiveSha256 || "", tools, toolHashes: {cmake: cmakeSha256 || "", ninja: ninjaSha256 || ""},
      plan, timeoutMs: Math.min(budget.timeoutMs, afterBudget.timeoutMs), sourceStats: {descriptor: descriptorStat, archive: archiveStat,
        cmake: cmakeStat, ninja: ninjaStat}};
  }

  if (args.length !== (phase === "build" ? 2 : 1)) {throw new Error("Windows LLVM phase needs exactly one phase name, or no audit argument");}
  const rootStat = fs.lstatSync(state.root, {throwIfNoEntry: false}), outputStat = fs.lstatSync(state.output, {throwIfNoEntry: false});
  if (!rootStat?.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(state.root) !== state.root ||
      !outputStat?.isDirectory() || outputStat.isSymbolicLink() || fs.realpathSync(state.output) !== state.output) {
    throw new Error("Windows LLVM phase requires its canonical configured same-run output");
  }
  const receiptStat = pathStat(state.result, "Windows LLVM configure receipt", 64 << 20),
    receipt = windowsControllerReadJson(state.result, 64 << 20, "Windows LLVM configure receipt"), descriptorStat = pathStat(state.descriptor, "Windows LLVM descriptor", 4 << 20),
    descriptorHash = fileHash(state.descriptor), plan = receipt.plan;
  windowsNativeLlvmCiRejectReportInputOverlap(reports, [repo, state.descriptor, plan?.archive || "",
    plan?.tools?.cmake || "", plan?.tools?.ninja || ""]);
  if (!plan || plan.output !== state.output || plan.descriptorFile !== state.descriptor || plan.jobs !== Number(ciBuildJobs(environment)) ||
      receipt.role !== "windows-llvm21-host-sdk" || receipt.stage !== "configured" || receipt.buildOnly !== true || receipt.compiled !== false ||
      receipt.executed !== false || receipt.publicationQualified !== false || receipt.fingerprints?.[state.descriptor] !== descriptorHash ||
      descriptorStat.nlink !== 1 || receiptStat.nlink !== 1) {
    throw new Error("Windows LLVM phase receipt does not match the fixed same-run configure plan");
  }
  const descriptor = validateWindowsLlvmSdkConfig(windowsControllerReadJson(state.descriptor, 4 << 20, "Windows LLVM descriptor"));
  windowsNativeLlvmCiRejectReportInputOverlap(reports, windowsNativeLlvmCiDescriptorRoots(descriptor));
  windowsLlvmSdkPlan(descriptor, plan.archive, state.output, plan.tools, 2);
  if (phase === "build") {
    const selected = args[1] || "";
    if (!buildPhases.some(row => row.name === selected)) {throw new Error("Windows LLVM build phase must be one maintained aggregate");}
  } else if ([path.join(state.output, "admitted.json"), path.join(state.output, "admitted-objects.json")].some(file => fs.existsSync(file))) {
    throw new Error("Windows LLVM audit refuses to restamp existing admission receipts");
  }
  const beforeReceiptSha256 = fileHash(state.result), budgetAfter = windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, phase, Date.now(), state.runDeadlineEpoch);
  return {phase, state, reports, receipt, descriptorHash, beforeReceiptSha256, timeoutMs: Math.min(budget.timeoutMs, budgetAfter.timeoutMs)};
}

/** Invoke one original public Windows LLVM builder phase; never restamp its receipts or ship the CMake tree.
 * @param {string[]} args @param {NodeJS.ProcessEnv} [environment] */
function buildWindowsNativeLlvmCi(args, environment = process.env) {
  const input = windowsNativeLlvmCiPreflight(args, environment), {phase, state} = input,
    builder = "scripts/build-native-windows-llvm-sdk.js", buildEnvironment = {...environment, MOM_AOT_BUILD_ONLY: "1"};
  if (phase === "configure") {
    const descriptorBytes = input.descriptorBytes;
    if (!descriptorBytes) {throw new Error("Windows LLVM configure lost its prevalidated descriptor bytes");}
    fs.mkdirSync(state.root, {mode: 0o700});
    windowsNativeLlvmCiDeadlineRecord(input.state, true);
    fs.writeFileSync(state.descriptor, descriptorBytes, {flag: "wx", mode: 0o444});
    const stat = windowsControllerRegularFile(state.descriptor, "Staged Windows LLVM descriptor");
    if (stat.nlink !== 1 || fileHash(state.descriptor) !== input.descriptorSha256 ||
        fileHash(input.descriptorSource) !== input.descriptorSha256 || fileHash(input.archive) !== input.archiveSha256 ||
        fileHash(input.tools.cmake) !== input.toolHashes.cmake || fileHash(input.tools.ninja) !== input.toolHashes.ninja) {
      throw new Error("Windows LLVM configure inputs changed before the public builder started");
    }
    const childTimeoutMs = Math.min(input.timeoutMs,
      windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, phase, Date.now(), state.runDeadlineEpoch).timeoutMs);
    runNodeProcessGroup(builder, ["configure", state.descriptor, input.archive, state.output,
      input.tools.cmake, input.tools.ninja, ciBuildJobs(buildEnvironment)], buildEnvironment, childTimeoutMs);
    const receipt = windowsControllerReadJson(state.result, 64 << 20, "Windows LLVM configure receipt");
    windowsNativeLlvmCiDeadlineRecord(state);
    if (receipt.role !== "windows-llvm21-host-sdk" || receipt.stage !== "configured" || receipt.plan?.output !== state.output ||
        receipt.plan?.descriptorFile !== state.descriptor || receipt.plan?.jobs !== Number(ciBuildJobs(buildEnvironment)) || receipt.compiled !== false ||
        receipt.buildOnly !== true || receipt.executed !== false || receipt.publicationQualified !== false ||
        fileHash(state.descriptor) !== input.descriptorSha256 || fileHash(input.descriptorSource) !== input.descriptorSha256 ||
        fileHash(input.archive) !== input.archiveSha256 || fileHash(input.tools.cmake) !== input.toolHashes.cmake ||
        fileHash(input.tools.ninja) !== input.toolHashes.ninja) {
      throw new Error("Windows LLVM configure result lost its externally pinned inputs or build-only status");
    }
  } else {
    const childTimeoutMs = Math.min(input.timeoutMs,
      windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, phase, Date.now(), state.runDeadlineEpoch).timeoutMs);
    windowsNativeLlvmCiDeadlineRecord(state, true);
    runNodeProcessGroup(builder, phase === "build" ? ["build", state.output, args[1] || ""] : ["audit", state.output],
      buildEnvironment, childTimeoutMs);
    windowsNativeLlvmCiDeadlineRecord(state);
    if (fileHash(state.result) !== input.beforeReceiptSha256 || fileHash(state.descriptor) !== input.descriptorHash) {
      throw new Error("Windows LLVM builder changed the original configure receipt or descriptor");
    }
  }
  const values = /** @type {Record<string,string>} */ ({windows_llvm_phase: phase === "build" ? "build-" + (args[1] || "") : phase,
    windows_llvm_state_dir: state.output, windows_llvm_result_sha256: fileHash(state.result),
    windows_llvm_build_only: "true", windows_llvm_executed: "false", windows_llvm_publication_qualified: "false"});
  if (phase === "build") {
    const checkpoint = windowsControllerReadJson(path.join(state.output, "build-state.json"), 64 << 20, "Windows LLVM build checkpoint"),
      proof = checkpoint.phaseProofs?.at(-1), proofPath = proof?.path;
    if (proof?.name !== args[1] || typeof proofPath !== "string" || !path.isAbsolute(proofPath) ||
        windowsControllerReadJson(proofPath, 16 << 20, "Windows LLVM phase proof").pass !== true) {
      throw new Error("Windows LLVM producer did not retain the selected successful phase proof");
    }
    values["windows_llvm_phase_proof_sha256"] = fileHash(proofPath);
    values["windows_llvm_checkpoint_sha256"] = fileHash(path.join(state.output, "build-state.json"));
  } else if (phase === "audit") {
    const admission = windowsControllerReadJson(path.join(state.output, "admitted.json"), 64 << 20, "Windows LLVM admission receipt");
    if (admission.role !== "windows-llvm21-static-host-admission" || admission.buildOnly !== true || admission.compiled !== true ||
        admission.executed !== false || admission.publicationQualified !== false) {
      throw new Error("Windows LLVM audit output is not a build-only admission receipt");
    }
    values["windows_llvm_admission_sha256"] = fileHash(path.join(state.output, "admitted.json"));
    values["windows_llvm_objects_sha256"] = fileHash(path.join(state.output, "admitted-objects.json"));
  }
  windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "outputs", Date.now(), state.runDeadlineEpoch);
  for (const report of input.reports) {verifyWindowsNativeLlvmCiReport(report);}
  appendWorkflowOutputs(environment, values);
  const summary = environment["GITHUB_STEP_SUMMARY"] || "";
  fs.appendFileSync(summary, `### Windows LLVM 21 ${values["windows_llvm_phase"]}\n` +
    `- Same-run output: ${state.output}; original configure receipt SHA-256: ${values["windows_llvm_result_sha256"]}.\n` +
    "- Build-only: true; executed: false; publication-qualified: false. No SDK build-tree artifact was exported.\n", "utf8");
  console.log(`[native-ci] Completed the public Windows LLVM 21 ${values["windows_llvm_phase"]} phase; no SDK artifact was exported.`);
  return values;
}

module.exports = {auditWindowsLlvmToolFile, buildWindowsHostSdkSourcesCi, buildWindowsNativeLlvmCi,
  buildWindowsNativeLlvmDescriptorCi, composeLocalWindowsLlvmSdkDescriptor, composeWindowsLlvmSdkDescriptor,
  mapWindowsLlvmSdkLibraries, validateWindowsLlvmLibraryPolicy, windowsHostSdkDeadlineBudget, windowsHostSdkPackageMembers,
  windowsHostSdkStatePaths, windowsNativeLlvmCiDeadlineBudget, windowsNativeLlvmCiDeadlineRecord, windowsNativeLlvmCiPreflight,
  windowsNativeLlvmCiRunDeadlineBudget, windowsNativeLlvmCiStatePaths, windowsNativeLlvmDescriptorCiPreflight,
  windowsNativeLlvmDescriptorCiStatePaths};
