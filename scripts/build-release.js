"use strict";

const {spawnSync} = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const {createHash} = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");
const {releaseBuildEnvironment} = require("./build-environment");
const {hasExactBuildFields, containsBuildPath, fileHash, readBuildJsonSnapshot, readBuildText} = require("./build-source-inputs");

const repo = path.resolve(__dirname, "..");
const version = require("../package.json").version;

/**
 * @param {NodeJS.Platform} platform
 * @param {boolean} [packageOnly]
 * @param {string} [releaseVersion]
 */
function engineeringArchiveCommands(platform, packageOnly = false, releaseVersion = version) {
  const scripts = path.join(repo, "scripts");
  if (platform === "linux") {
    // Keep the existing cache lock through publication, including package-only runs.
    // Development parents already own this host lock around their entire command.
    const lifecycle = [
      // eslint-disable-next-line no-template-curly-in-string -- Bash parameter expansion.
      'if [ "${MOM_R_SH:-0}" != 1 ] && [ "${MOM_NATIVE_BUILD_LOCK:-0}" != 1 ]; then',
      "  lock_dir=$(git rev-parse --path-format=absolute --git-path mom-locks)",
      '  mkdir -p -m 700 "$lock_dir"',
      // eslint-disable-next-line no-template-curly-in-string -- Bash's numeric caller UID.
      '  exec 9>"$lock_dir/mom-r-sh-${UID}.build.lock"',
      "  flock 9",
      "  export MOM_NATIVE_BUILD_LOCK=1",
      "fi",
      'if [ "$4" != 1 ]; then bash "$1" true; fi',
      'exec bash "$2" "$3"',
    ].join("\n");
    return [["bash", "-eu", "-c", lifecycle, "--",
      path.join(repo, "scripts", "multicompiler-entrypoint.sh"),
      path.join(scripts, "package-linux-combined.sh"), releaseVersion, packageOnly ? "1" : "0"]];
  }
  if (platform === "win32") {
    const powershell = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"];
    const commands = [
      [...powershell, path.join(scripts, "build-windows-multicompiler.ps1"), "-Backend", "all"],
      [...powershell, path.join(scripts, "package-windows.ps1"), "-Version", releaseVersion],
    ];
    return packageOnly ? commands.slice(1) : commands;
  }
  throw new Error("Native release builds support Linux and Windows x86-64 only");
}

/** Normalize the deploy override before it can reach a command or archive path.
 * @param {NodeJS.ProcessEnv} env
 */
function releaseVersion(env) {
  const value = (env["MOM_RELEASE_VERSION"] || version).replace(/^[vV]/, "");
  if (!/^[0-9]/.test(value) || /[^0-9A-Za-z.-]/.test(value)) {
    throw new Error(`Invalid release version: ${value}`);
  }
  return value;
}

/**
 * @typedef {(command: string, args: string[], options: import("node:child_process").SpawnSyncOptions) =>
 *   {status: number | null, error?: Error}} ReleaseSpawn
 */

/**
 * @param {NodeJS.Platform} platform
 * @param {string} arch
 * @param {NodeJS.ProcessEnv} env
 * @param {boolean} [packageOnly]
 * @param {ReleaseSpawn} [spawn]
 * @returns {number}
 */
function buildEngineeringArchive(platform, arch, env, packageOnly = false, spawn = spawnSync) {
  if (arch !== "x64") {
    throw new Error("Native release builds support Linux and Windows x86-64 only");
  }
  const commands = engineeringArchiveCommands(platform, packageOnly, releaseVersion(env));
  const buildEnv = releaseBuildEnvironment(env);
  for (const [command, ...args] of commands) {
    if (!command) {throw new Error("Release build command is missing");}
    const result = spawn(command, args, {cwd: repo, env: buildEnv, stdio: "inherit"});
    if (result.error) {throw result.error;}
    if (result.status !== 0) {return result.status ?? 1;}
  }
  return 0;
}

/** Produce Windows Core and every registered GPU graph from their existing source builders.
 * The generated controller descriptor refers only to this run's freshly audited outputs.
 * @param {{nativeSources: {core: string, gpu: Record<string, string>}, application: {defaultCompiler: string, tools: unknown, sdk: unknown}, [key: string]: unknown}} descriptor
 * @param {string} workspace @param {NodeJS.ProcessEnv} environment */
function buildWindowsSources(descriptor, workspace, environment) {
  if (process.platform !== "win32") {throw new Error("Automatic Windows Core production requires a Windows build host");}
  const {absolutePath} = require("./build-native-release");
  const core = require("./build-native-windows-core"), gpu = require("./build-native-windows-gpu"), controller = require("./build-native-windows-controller");
  const {nativeApplicationGpu} = require("./build-native-application");
  const sources = descriptor.nativeSources, app = descriptor.application;
  if (!sources || !hasExactBuildFields(sources, ["core", "gpu"]) || !absolutePath(sources.core) ||
      !sources.gpu || typeof sources.gpu !== "object" || Array.isArray(sources.gpu) || !app ||
      !hasExactBuildFields(app, ["defaultCompiler", "sdk", "tools"])) {throw new Error("Invalid Windows source-build descriptor");}
  const factories = nativeApplicationGpu(fs.readFileSync(path.join(repo, "runtime/gpu-dispatch.cpp"), "utf8"),
    {defaultCompiler: app.defaultCompiler, flags: [], links: []}).factories;
  const keys = factories.map(name => name.replace("mom_native_gpu_", "").replaceAll("_", "-"));
  if (!hasExactBuildFields(sources.gpu, keys) ||
      keys.some(key => !absolutePath(sources.gpu[key]))) {throw new Error("Windows source build requires every maintained GPU profile");}
  const parent = path.dirname(workspace);
  if (!absolutePath(workspace) || !fs.statSync(parent, {throwIfNoEntry: false})?.isDirectory() ||
      fs.lstatSync(workspace, {throwIfNoEntry: false})) {
    throw new Error("Windows component workspace must be new and outside the repository");
  }
  // Resolve the parent before checking containment so a symlink cannot redirect build outputs into the source tree.
  if (containsBuildPath(fs.realpathSync(repo), path.join(fs.realpathSync(parent), path.basename(workspace)))) {
    throw new Error("Windows component workspace must be new and outside the repository");
  }
  /** @type {Record<string, string>} */
  const hashes = {};
  /** @param {string} file */
  const readDescriptor = file => {
    const {value, sha256} = readBuildJsonSnapshot(file);
    hashes[file] = sha256;
    return value;
  };
  core.validateWindowsCoreDescriptor(readDescriptor(sources.core));
  for (const key of keys) {
    const selected = gpu.validateWindowsGpuConfig(readDescriptor(sources.gpu[key] || ""));
    if (selected.profile !== key) {throw new Error("Windows GPU descriptor profile differs: " + key);}
    const git = gpu.windowsGpuGit(selected, process.platform);
    if (git && !fs.statSync(git, {throwIfNoEntry: false})?.isFile()) {throw new Error("Missing Windows AdaptiveCpp Git executable: " + git);}
  }
  const rank = (/** @type {string} */ key) => key === "oneapi" ? 0 : key === "dpcpp-opencl" ? 1 : key.endsWith("hip") ? 3 : 2;
  keys.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  const coreDirectory = path.join(workspace, "core"), receiptPaths = keys.map(key => path.join(workspace, "gpu", key, "result.json"));
  const template = {...descriptor, application: {...app, core: path.join(coreDirectory, "cpu-core.lib"), gpuReceipts: receiptPaths}};
  Reflect.deleteProperty(template, "nativeSources");
  const config = controller.validateWindowsControllerConfig(template);
  for (const file of controller.explicitInputFiles(config)) {
    if (!fs.statSync(file, {throwIfNoEntry: false})?.isFile()) {throw new Error("Missing Windows static SDK input: " + file);}
  }
  const unchanged = () => {for (const [file, hash] of Object.entries(hashes)) {if (fileHash(file) !== hash) {throw new Error("Windows source descriptor changed: " + file);}}};
  unchanged(); fs.mkdirSync(workspace); fs.mkdirSync(path.join(workspace, "gpu"));
  const builtCore = core.buildNativeWindowsCore(sources.core, coreDirectory, environment);
  if (builtCore.archive.path !== template.application.core) {throw new Error("Windows Core producer returned an unexpected archive");}
  unchanged();
  for (const [index, key] of keys.entries()) {
    const output = path.dirname(receiptPaths[index] || ""), record = gpu.buildNativeWindowsGpu(sources.gpu[key] || "", output, environment);
    if (record.factory !== "mom_native_gpu_" + key.replaceAll("-", "_") || record.profile !== key || !fs.existsSync(receiptPaths[index] || "")) {
      throw new Error("Windows GPU producer returned an unexpected graph: " + key);
    }
    unchanged();
  }
  const generated = path.join(workspace, "controller.json"); fs.writeFileSync(generated, JSON.stringify(template, null, 2) + "\n", {flag: "wx"});
  return {generated, generatedHash: fileHash(generated), hashes};
}

/** @param {string} output @param {string} binary @param {string} sourceHash
 * @param {{upx: string, upxSha256: string}} packing @param {unknown} value */
function bindPackingReceipt(output, binary, sourceHash, packing, value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {throw new Error("Native packing result is malformed");}
  const result = /** @type {Record<string, unknown>} */ (value), receiptPath = path.join(output, "result.json");
  const stat = fs.lstatSync(receiptPath, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink()) {throw new Error("Native packer did not write a regular result receipt");}
  const {value: saved, sha256: receiptHash} = readBuildJsonSnapshot(receiptPath);
  const source = result["source"], artifact = result["artifact"], tool = result["tool"], logs = result["logs"];
  if (!isDeepStrictEqual(saved, value) ||
      !source || typeof source !== "object" || Array.isArray(source) ||
      !artifact || typeof artifact !== "object" || Array.isArray(artifact) ||
      !tool || typeof tool !== "object" || Array.isArray(tool) ||
      !logs || typeof logs !== "object" || Array.isArray(logs)) {
    throw new Error("Native packing result receipt differs from the packed candidate");
  }
  const sourceRecord = /** @type {Record<string, unknown>} */ (source), artifactRecord = /** @type {Record<string, unknown>} */ (artifact),
    toolRecord = /** @type {Record<string, unknown>} */ (tool), logRecords = /** @type {Record<string, unknown>} */ (logs);
  if (sourceRecord["path"] !== binary || sourceRecord["bytes"] !== fs.statSync(binary).size || sourceRecord["sha256"] !== sourceHash ||
      typeof artifactRecord["path"] !== "string" || artifactRecord["path"] !== path.join(output, path.basename(binary)) ||
      artifactRecord["bytes"] !== fs.statSync(artifactRecord["path"]).size ||
      typeof artifactRecord["sha256"] !== "string" || !/^[0-9a-f]{64}$/.test(artifactRecord["sha256"]) ||
      fileHash(artifactRecord["path"]) !== artifactRecord["sha256"] ||
      toolRecord["path"] !== packing.upx || toolRecord["sha256"] !== packing.upxSha256 ||
      typeof result["restoredSha256"] !== "string" || !/^[0-9a-f]{64}$/.test(result["restoredSha256"]) ||
      result["buildOnly"] !== true || result["publicationQualified"] !== false) {
    throw new Error("Native packing result receipt does not bind its source, artifact, or tool");
  }
  const stages = ["pack", "integrity", "restore"];
  if (!hasExactBuildFields(logRecords, stages)) {throw new Error("Native packing result receipt has incomplete log evidence");}
  for (const stage of stages) {
    const entry = logRecords[stage], logPath = path.join(output, stage + ".log");
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {throw new Error("Native packing result receipt has invalid log evidence: " + stage);}
    const log = /** @type {Record<string, unknown>} */ (entry), logStat = fs.lstatSync(logPath, {throwIfNoEntry: false});
    if (!logStat?.isFile() || logStat.isSymbolicLink() || log["path"] !== logPath ||
        typeof log["sha256"] !== "string" || !/^[0-9a-f]{64}$/.test(log["sha256"]) || fileHash(logPath) !== log["sha256"]) {
      throw new Error("Native packing log is missing or changed: " + stage);
    }
  }
  if (fileHash(receiptPath) !== receiptHash) {throw new Error("Native packing result receipt changed during admission");}
  return {path: receiptPath, sha256: receiptHash};
}

/** Match both receipt formats against the build-only artifact and its producer descriptor.
 * @param {any} producer @param {{path:string, bytes:number, sha256:string}} expected
 * @param {string} configHash @param {string} [windowsTarget] */
function matchesProducerReceipt(producer, expected, configHash, windowsTarget) {
  if (!producer || typeof producer !== "object" || Array.isArray(producer) ||
      producer.buildOnly !== true || producer.publicationQualified !== false || producer.configHash !== configHash) {return false;}
  const targetMatches = windowsTarget ? producer.target === windowsTarget :
    producer.target === undefined || ["linux", "x86_64-unknown-linux-gnu"].includes(producer.target);
  const artifact = windowsTarget ? producer.artifact : {path: producer.binary, bytes: producer.bytes, sha256: producer.sha256};
  return targetMatches && artifact !== null && typeof artifact === "object" && !Array.isArray(artifact) &&
    artifact.path === expected.path && artifact.bytes === expected.bytes && artifact.sha256 === expected.sha256;
}

/** Pack an already-assembled raw candidate without reopening its SDK or invoking a builder.
 * The producer entrypoint remains immutable provenance; only its pinned raw executable and
 * transported producer receipt are consumed at their relocated paths.
 * @param {{entrypointPath:string, entrypointSha256:string, sourcePath:string, sourceSha256:string,
 *   producerReceiptPath:string, target:"linux"|"win32", output:string, upx:string, upxSha256:string}} options
 * @param {typeof spawnSync} [spawn] */
function packNativeEntrypoint(options, spawn = spawnSync) {
  const keys = ["entrypointPath", "entrypointSha256", "sourcePath", "sourceSha256", "producerReceiptPath", "target", "output", "upx", "upxSha256"];
  if (!hasExactBuildFields(options, keys)) {
    throw new Error("Native entrypoint packing requires an exact pinned handoff");
  }
  const {entrypointPath, entrypointSha256, sourcePath, sourceSha256, producerReceiptPath, target, output, upx, upxSha256} = options;
  const hashPattern = /^[0-9a-f]{64}$/;
  /** @param {unknown} value @returns {value is string} */
  const canonicalAbsolute = value => typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) &&
    path.resolve(value) === value && !/[\0\r\n]/.test(value);
  /** @param {unknown} value */
  const record = value => Boolean(value && typeof value === "object" && !Array.isArray(value));
  /** @param {string} file @param {string} label @param {number} maximum */
  const regularFile = (file, label, maximum) => {
    if (!canonicalAbsolute(file)) {
      throw new Error("Native entrypoint packing requires a canonical absolute " + label);
    }
    const stat = fs.lstatSync(file, {bigint: true}), real = fs.realpathSync(file), sameReal = process.platform === "win32" ? real.toLowerCase() === file.toLowerCase() : real === file;
    if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1n || stat.size < 1n || stat.size > BigInt(maximum) || !sameReal) {
      throw new Error("Native entrypoint packing requires a bounded, singly-linked regular " + label);
    }
    return stat;
  };
  if (!["linux", "win32"].includes(target) || ![entrypointSha256, sourceSha256, upxSha256].every(value => typeof value === "string" && hashPattern.test(value)) ||
      ![entrypointPath, sourcePath, producerReceiptPath, output, upx].every(canonicalAbsolute) || fs.existsSync(output)) {
    throw new Error("Native entrypoint packing needs matching platform pins and a fresh absolute output directory");
  }
  if (path.basename(entrypointPath) !== "entrypoint.json" || path.basename(sourcePath) !== (target === "win32" ? "mom.exe" : "mom") ||
      path.basename(producerReceiptPath) !== "result.json" || path.dirname(producerReceiptPath) !== path.dirname(sourcePath)) {
    throw new Error("Native entrypoint packing paths do not identify the expected candidate files");
  }
  const parent = path.dirname(output), parentStat = fs.lstatSync(parent, {bigint: true}), parentReal = fs.realpathSync(parent), sameParent =
    process.platform === "win32" ? parentReal.toLowerCase() === parent.toLowerCase() : parentReal === parent;
  if (!parentStat.isDirectory() || parentStat.isSymbolicLink() || !sameParent) {
    throw new Error("Native entrypoint packing output parent must be a canonical directory");
  }
  if (containsBuildPath(path.dirname(sourcePath), output) || containsBuildPath(path.dirname(entrypointPath), output)) {
    throw new Error("Native entrypoint packing output must be outside the immutable handoff directories");
  }
  regularFile(entrypointPath, "original entrypoint", 8 << 20);
  const {value: entry, sha256: entryHash} = readBuildJsonSnapshot(entrypointPath, undefined, {
    sha256: entrypointSha256, message: "Native original entrypoint differs from its external SHA-256 pin",
  });
  if (fileHash(entrypointPath) !== entryHash) {throw new Error("Native original entrypoint changed during admission");}
  if (!record(entry) || entry.version !== 1 || entry.target !== target || entry.buildOnly !== true || entry.publicationQualified !== false ||
      entry.packingReceipt !== undefined || !record(entry.source) || !record(entry.artifact) || !record(entry.producerReceipt) ||
      !record(entry.componentHashes) || !record(entry.recipe) || !canonicalAbsolute(entry.configFile) ||
      typeof entry.configHash !== "string" || !hashPattern.test(entry.configHash)) {
    throw new Error("Native original entrypoint is not a raw build-only candidate for the selected target");
  }
  const originalArtifactPath = entry.artifact.path, originalSourcePath = entry.source.path,
    originalArtifactHash = entry.artifact.sha256, originalSourceHash = entry.source.sha256, originalBytes = entry.artifact.bytes,
    originalProducerPath = entry.producerReceipt.path, originalProducerHash = entry.producerReceipt.sha256;
  if (!canonicalAbsolute(originalArtifactPath) || originalArtifactPath !== originalSourcePath ||
      typeof originalArtifactHash !== "string" || !hashPattern.test(originalArtifactHash) || originalArtifactHash !== originalSourceHash ||
      typeof originalBytes !== "number" || !Number.isSafeInteger(originalBytes) || originalBytes < 1 ||
      typeof originalProducerHash !== "string" || !hashPattern.test(originalProducerHash) ||
      !canonicalAbsolute(originalProducerPath) ||
      originalProducerPath !== path.join(path.dirname(originalArtifactPath), "result.json") ||
      entry.componentHashes[originalProducerPath] !== originalProducerHash) {
    throw new Error("Native original entrypoint does not bind its raw executable and producer receipt");
  }
  if (entry.generatedDescriptor !== undefined && (!record(entry.generatedDescriptor) || typeof entry.generatedDescriptor.path !== "string" ||
      !canonicalAbsolute(entry.generatedDescriptor.path) || typeof entry.generatedDescriptor.sha256 !== "string" || !hashPattern.test(entry.generatedDescriptor.sha256))) {
    throw new Error("Native original entrypoint has an invalid generated descriptor binding");
  }
  const sourceStat = regularFile(sourcePath, "relocated raw executable", (1 << 30) - 1), sourceHash = fileHash(sourcePath);
  if (sourceHash !== sourceSha256 || sourceHash !== originalArtifactHash || Number(sourceStat.size) !== originalBytes ||
      entry.source.bytes !== undefined && entry.source.bytes !== Number(sourceStat.size)) {
    throw new Error("Native raw executable differs from its external SHA-256 pin or original entrypoint");
  }
  const header = Buffer.alloc(4), sourceFd = fs.openSync(sourcePath, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
  try {
    if (fs.readSync(sourceFd, header, 0, header.length, 0) !== header.length ||
        (target === "win32" ? header.readUInt16LE() !== 0x5a4d : header.toString("hex") !== "7f454c46")) {
      throw new Error("Native raw executable format differs from its pinned target");
    }
  } finally {fs.closeSync(sourceFd);}
  if (fileHash(sourcePath) !== sourceHash) {throw new Error("Native raw executable changed during target admission");}
  regularFile(producerReceiptPath, "transported producer receipt", 8 << 20);
  const {value: producer, sha256: producerHash} = readBuildJsonSnapshot(producerReceiptPath, undefined, {
    sha256: originalProducerHash, message: "Transported producer receipt differs from its entrypoint hash",
  });
  if (fileHash(producerReceiptPath) !== producerHash) {throw new Error("Transported producer receipt changed during admission");}
  const rawArtifact = {path: originalArtifactPath, bytes: Number(sourceStat.size), sha256: sourceHash};
  if (!matchesProducerReceipt(producer, rawArtifact, entry.generatedDescriptor?.sha256 || entry.configHash,
    target === "win32" ? "x86_64-pc-windows-msvc" : undefined) ||
      target === "win32" && (producer.nativeRuntime !== true || producer.executed !== false)) {
    throw new Error("Transported producer receipt does not bind the pinned raw executable, target and descriptor");
  }
  const toolStat = regularFile(upx, "pinned UPX tool", 256 << 20);
  if (!(toolStat.mode & 0o111n) || fileHash(upx) !== upxSha256) {throw new Error("Native packing tool differs from its external SHA-256 pin or is not executable");}

  // Artifact transport normalizes files to 0644; UPX requires an executable ELF.
  // Restore permissions only after admitting every input; the pinned bytes stay unchanged.
  if (target === "linux") {fs.chmodSync(sourcePath, 0o755);}
  const packing = {upx, upxSha256}, packed = require("./pack-native-executable").packNativeRelease({source: sourcePath, output, ...packing}, spawn),
    packingReceipt = bindPackingReceipt(output, sourcePath, sourceHash, packing, packed);
  if (fileHash(entrypointPath) !== entryHash || fileHash(sourcePath) !== sourceHash || fileHash(producerReceiptPath) !== producerHash ||
      fileHash(upx) !== upxSha256 || fileHash(packingReceipt.path) !== packingReceipt.sha256) {
    throw new Error("Native packing inputs changed before writing the chained entrypoint");
  }
  const componentHashes = {...entry.componentHashes};
  if (originalProducerPath !== producerReceiptPath) {delete componentHashes[originalProducerPath];}
  componentHashes[producerReceiptPath] = producerHash;
  const chained = {...entry,
    source: {path: sourcePath, bytes: Number(sourceStat.size), sha256: sourceHash}, artifact: packed.artifact,
    sourceEntrypoint: {path: entrypointPath, sha256: entryHash}, producerReceipt: {path: producerReceiptPath, sha256: producerHash},
    componentHashes, packingReceipt, buildOnly: true, publicationQualified: false};
  // Distribution notices are supplied and re-admitted later, independently of this build handoff.
  Reflect.deleteProperty(chained, "notices"); Reflect.deleteProperty(chained, "noticeAssembly");
  const chainedPath = path.join(output, "entrypoint.json");
  fs.writeFileSync(chainedPath, JSON.stringify(chained, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return {entrypoint: chainedPath, entryHash: fileHash(chainedPath), source: chained.source, artifact: packed.artifact, packingReceipt};
}

/** Optional reviewed distribution data, not an application dependency or a claim of
 * license completeness. Check before compilation and preserve the exact selected bytes.
 * @param {unknown} value */
function noticeInput(value) {
  if (value === undefined) {return undefined;}
  const selected = /** @type {{path?: unknown, sha256?: unknown, receipt?: unknown} | null} */ (value);
  const keys = ["path", "sha256"], assembled = hasExactBuildFields(selected, [...keys, "receipt"]);
  if (!selected || (!assembled && !hasExactBuildFields(selected, keys)) ||
      typeof selected.path !== "string" || selected.path.length > 4096 || !path.isAbsolute(selected.path) ||
      Array.from(selected.path).some(c => c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127) ||
      typeof selected.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(selected.sha256)) {throw new Error("Invalid native distribution notice input");}
  let assembly;
  if (assembled) {
    const receipt = /** @type {{path?: unknown, sha256?: unknown} | null} */ (selected.receipt);
    if (!receipt || !hasExactBuildFields(receipt, ["path", "sha256"]) ||
        typeof receipt.path !== "string" || !path.isAbsolute(receipt.path) || typeof receipt.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(receipt.sha256)) {
      throw new Error("Invalid native distribution notice assembly receipt");
    }
    const admitted = require("./prepare-native-notices").readNativeNoticeReceipt(receipt.path, receipt.sha256);
    assembly = {receiptPath: admitted.receiptPath, receiptSha256: admitted.receiptSha256, fingerprint: admitted.fingerprint, notice: admitted.notice};
    if (assembly.notice.path !== selected.path || assembly.notice.sha256 !== selected.sha256) {
      throw new Error("Native notice assembly receipt does not bind the selected notice body");
    }
  }
  const stat = fs.lstatSync(selected.path);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 || stat.size > (8 << 20)) {throw new Error("Native notices must be bounded, nonempty regular text");}
  // Bounded reading rejects growth; hashing the UTF-8 round trip also rejects invalid encoding.
  const data = Buffer.from(readBuildText(selected.path));
  if (data.length !== stat.size || data.includes(0) || createHash("sha256").update(data).digest("hex") !== selected.sha256 ||
      fileHash(selected.path) !== selected.sha256 || assembly && assembly.notice.bytes !== data.length) {
    throw new Error("Native notice bytes or hash differ from the selected UTF-8 source");
  }
  return {path: selected.path, sha256: selected.sha256, data, ...(assembly ? {assembly} : {})};
}

/** Re-admit selected notice inputs at build boundaries; the body-only legacy form remains valid.
 * @param {unknown} value @param {ReturnType<typeof noticeInput>} original */
function verifyNoticeInput(value, original) {
  const current = noticeInput(value);
  if (!isDeepStrictEqual(current, original)) {throw new Error("Native notice receipt or body changed during the build");}
}

/** Reuse the original source builders, optionally compacting their unchanged executable.
 * SDK selection remains explicit until native SDK production and runtime parity qualify.
 * Unpacked build evidence stays in a new sibling, never overwritten or shipped as libraries.
 * @param {string} configFile @param {string} output
 * @param {{upx: string, upxSha256: string} | undefined} [packing]
 * @param {NodeJS.ProcessEnv} [environment] */
function buildNativeArtifact(configFile, output, packing, environment = process.env) {
  const {absolutePath} = require("./build-native-release");
  const {sourceInputs, verifySourceInputs} = require("./build-source-inputs");
  if (![configFile, output].every(absolutePath) || fs.existsSync(output)) {throw new Error("Select an absolute native SDK and new output directory");}
  const env = {...releaseBuildEnvironment(environment), MOM_AOT_BUILD_ONLY: "1"};
  const unpacked = packing === undefined ? output : output + ".unpacked";
  if (packing !== undefined && (!hasExactBuildFields(packing, ["upx", "upxSha256"]) || !absolutePath(packing.upx) || !/^[0-9a-f]{64}$/.test(packing.upxSha256) ||
      fileHash(packing.upx) !== packing.upxSha256 || fs.existsSync(unpacked))) {throw new Error("Native packing needs a pinned tool and a new unpacked build directory");}
  let recipe = sourceInputs(repo, [], ["scripts/build-release.js", "scripts/build-environment.js"]);
  const {value: descriptor, sha256: configHash} = readBuildJsonSnapshot(configFile);
  if (fileHash(configFile) !== configHash) {throw new Error("Native release descriptor changed during admission");}
  if (!descriptor || typeof descriptor !== "object" || Array.isArray(descriptor) || descriptor.version !== 1) {throw new Error("Invalid native release SDK descriptor");}
  const selectedNotices = noticeInput(descriptor.notices);
  const windows = Object.hasOwn(descriptor, "windowsSdk");
  // Presence only: the Windows controller (and buildWindowsSources for nativeSources) validates the
  // application's Core and per-profile GPU inputs. Without it the controller would link no miner graphs.
  if (windows && !descriptor.application) {throw new Error("Windows native releases require an application descriptor for the Core and all GPU graphs");}
  if (windows && descriptor.outputName !== undefined && descriptor.outputName !== "mom.exe") {throw new Error("Windows native release output must be mom.exe");}
  if (!windows && descriptor.nativeSources !== undefined) {throw new Error("Windows source descriptors cannot select a Linux build");}
  if (windows && descriptor.nativeSources !== undefined) {
    recipe = sourceInputs(repo, [], ["scripts/build-release.js", "scripts/build-environment.js", "scripts/build-native-windows-core.js",
      "scripts/build-native-windows-gpu.js", "scripts/build-native-windows-controller.js", "scripts/build-native-application.js"]);
  }
  // Lazy loading avoids the Windows builder's shared absolutePath import forming a cycle.
  let binary, generatedDescriptor = /** @type {{path: string, sha256: string} | undefined} */ (undefined);
  let componentHashes = /** @type {Record<string, string>} */ ({});
  if (windows) {
    const sourceBuild = descriptor.nativeSources === undefined ? null : buildWindowsSources(descriptor, unpacked + ".components", env);
    if (sourceBuild) {componentHashes = sourceBuild.hashes; generatedDescriptor = {path: sourceBuild.generated, sha256: sourceBuild.generatedHash};}
    const built = require("./build-native-windows-controller").buildNativeWindowsController(sourceBuild?.generated || configFile, unpacked, env);
    if (!built.application || built.executed !== false || built.buildOnly !== true) {throw new Error("Windows native release did not build the full application");}
    binary = built.artifact.path;
  } else {binary = require("./build-native-release").buildNativeRelease(configFile, unpacked, env).binary;}
  if (binary !== path.join(unpacked, windows ? "mom.exe" : "mom")) {throw new Error("Native source builder did not return its expected executable");}
  if (generatedDescriptor && fileHash(generatedDescriptor.path) !== generatedDescriptor.sha256) {throw new Error("Generated Windows controller descriptor changed during compilation");}
  if (fileHash(configFile) !== configHash) {throw new Error("Native release descriptor changed during compilation");}
  verifyNoticeInput(descriptor.notices, selectedNotices);
  verifySourceInputs(recipe, sourceInputs(repo, [], Object.keys(recipe.graph)), "Native release entrypoint");
  const sourceHash = fileHash(binary);
  const producerReceiptPath = path.join(unpacked, "result.json");
  if (!fs.existsSync(producerReceiptPath)) {throw new Error("Native source builder did not write its build receipt");}
  const {value: producer, sha256: producerReceiptHash} = readBuildJsonSnapshot(producerReceiptPath);
  const artifact = {path: binary, bytes: fs.statSync(binary).size, sha256: sourceHash};
  if (!matchesProducerReceipt(producer, artifact, generatedDescriptor?.sha256 || configHash,
    windows ? descriptor.target || "x86_64-pc-windows-msvc" : undefined)) {
    throw new Error("Native producer receipt does not match the build-only executable and target");
  }
  const producerReceipt = {path: producerReceiptPath, sha256: producerReceiptHash};
  componentHashes[producerReceiptPath] = producerReceiptHash;
  if (fileHash(producerReceipt.path) !== producerReceipt.sha256) {throw new Error("Native producer receipt changed before packing");}
  const result = packing ? require("./pack-native-executable").packNativeRelease({source: binary, output, ...packing}) :
    {artifact, buildOnly: true, publicationQualified: false};
  if (fileHash(producerReceipt.path) !== producerReceipt.sha256) {throw new Error("Native producer receipt changed during packing");}
  if (fileHash(configFile) !== configHash) {throw new Error("Native release descriptor changed during packing");}
  if (generatedDescriptor && fileHash(generatedDescriptor.path) !== generatedDescriptor.sha256) {throw new Error("Generated Windows controller descriptor changed during packing");}
  verifyNoticeInput(descriptor.notices, selectedNotices);
  for (const [file, hash] of Object.entries(componentHashes)) {if (fileHash(file) !== hash) {throw new Error("Windows source descriptor changed: " + file);}}
  verifySourceInputs(recipe, sourceInputs(repo, [], Object.keys(recipe.graph)), "Native release entrypoint");
  if (fileHash(binary) !== sourceHash || ("source" in result && result.source.sha256 !== sourceHash) ||
      result.artifact.path !== path.join(output, windows ? "mom.exe" : "mom") || fileHash(result.artifact.path) !== result.artifact.sha256) {
    throw new Error("Native release artifact chain changed during packing");
  }
  const packingReceipt = packing ? bindPackingReceipt(output, binary, sourceHash, packing, result) : undefined;
  let notices;
  if (selectedNotices) {
    verifyNoticeInput(descriptor.notices, selectedNotices);
    const file = path.join(output, "THIRD-PARTY-NOTICES.txt");
    fs.writeFileSync(file, selectedNotices.data, {flag: "wx"});
    if (fileHash(file) !== selectedNotices.sha256 || fileHash(selectedNotices.path) !== selectedNotices.sha256) {throw new Error("Native notice input changed during distribution staging");}
    notices = {path: file, sha256: selectedNotices.sha256, bytes: selectedNotices.data.length};
  }
  fs.writeFileSync(path.join(output, "entrypoint.json"), JSON.stringify({version: 1, configFile, configHash,
    target: windows ? "win32" : "linux", source: {path: binary, sha256: sourceHash}, artifact: result.artifact,
    recipe, componentHashes, generatedDescriptor, producerReceipt, ...(packingReceipt ? {packingReceipt} : {}),
    ...(notices ? {notices} : {}), ...(selectedNotices?.assembly ? {noticeAssembly: selectedNotices.assembly} : {}),
    buildOnly: true, publicationQualified: false}, null, 2) + "\n", {flag: "wx"});
  return result;
}

/** A configured native SDK makes the normal npm command use the same source build as
 * --native. Partial settings must not silently fall through to the old Node archive path.
 * Explicit CLI selection wins, so an installed default cannot redirect a deliberate build.
 * @param {string[]} args @param {NodeJS.ProcessEnv} env */
function nativeReleaseArguments(args, env) {
  if (args[0] === "--native" || !["MOM_NATIVE_SDK", "MOM_NATIVE_OUTPUT", "MOM_NATIVE_UPX", "MOM_NATIVE_UPX_SHA256"].some(key => env[key] !== undefined)) {return args;}
  const sdk = env["MOM_NATIVE_SDK"], output = env["MOM_NATIVE_OUTPUT"], upx = env["MOM_NATIVE_UPX"], sha = env["MOM_NATIVE_UPX_SHA256"];
  if (args.length || !sdk || !output || (upx !== undefined || sha !== undefined) && (!upx || !sha)) {
    throw new Error("Native release settings require MOM_NATIVE_SDK and MOM_NATIVE_OUTPUT, with both MOM_NATIVE_UPX and MOM_NATIVE_UPX_SHA256 for packing; do not mix archive flags");
  }
  return ["--native", sdk, output, ...(upx && sha ? ["--pack", upx, sha] : [])];
}

// The default command calls shared helpers which import this module's build APIs.
// Publish them before dispatch, including when this file is the process entrypoint.
module.exports = {buildEngineeringArchive, buildNativeArtifact, packNativeEntrypoint, matchesProducerReceipt,
  nativeReleaseArguments, engineeringArchiveCommands, releaseBuildEnvironment, releaseVersion, noticeInput};

if (require.main === module) {
  try {
    const args = nativeReleaseArguments(process.argv.slice(2), process.env);
    if (args.length === 0) {
      process.exitCode = require("./build-local-native-command").launchLocalNativeRelease();
    } else if (args[0] === "--native") {
      if (![3, 6].includes(args.length) || !args[1] || !args[2] || (args.length === 6 && (args[3] !== "--pack" || !args[4] || !args[5]))) {
        throw new Error("Usage: npm run build:release -- --native /absolute/sdk.json /new/absolute/output [--pack /pinned/upx <sha256>]");
      }
      const result = buildNativeArtifact(args[1], args[2], args.length === 6 ? {upx: args[4] || "", upxSha256: args[5] || ""} : undefined);
      console.log("[native] Built " + result.artifact.path + "; NOT executed or publication-qualified.");
    } else if (args[0] === "--engineering-archive") {
      // Transitional deployment/source tests still use Node/addon archives. Never select
      // that different product implicitly for a standalone-native production build.
      const archiveArgs = args.slice(1);
      if (archiveArgs.length > 1 || archiveArgs.length === 1 && !["--package-only", "--clean"].includes(archiveArgs[0] || "")) {
        throw new Error("Usage: npm run build:release -- --engineering-archive [--package-only | --clean]");
      }
      const env = archiveArgs[0] === "--clean" ? {...process.env, MOM_FORCE_REBUILD: "1"} : process.env;
      process.exitCode = buildEngineeringArchive(process.platform, process.arch, env, archiveArgs[0] === "--package-only");
      if (process.exitCode === 0) {
        console.log(`Built mom-v${releaseVersion(process.env)}-${process.platform === "win32" ? "win.zip" : "lin.tgz"}`);
      }
    } else {throw new Error("Usage: npm run build:release [-- --native /absolute/sdk.json /new/output]. Engineering Mode remains npm run dev; legacy Node archives require explicit --engineering-archive.");}
  } catch (error) {
    console.error(error instanceof Error ? error.message : "Native release build failed");
    process.exitCode = 1;
  }
}
