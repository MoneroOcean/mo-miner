"use strict";

// Qualification driver only: it builds nothing and never treats the build receipt as a pass.
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {hasExactBuildFields, fileHash, readBuildJson, readBuildJsonSnapshot} = require("./build-source-inputs");
const {matchesProducerReceipt} = require("./build-release");
const {validCiSuffix} = require("./native-ci-job");
const {nativeNoticeName, maximumNoticeBytes, nativeNoticeFile, validateNoticeAssemblyMetadata} = require("./package-native-release");
const {nativeDependencies, auditNativeGlibc, auditNativeVersionNeeds, auditNativeTrace} = require("./native-audit");
const {linuxGlibcBaseline} = require("./native-linux-sysroot");
const {elfReport} = require("./native-elf");
const {verifyNativePeMap} = require("./native-pe-audit");
const gpuTuning = require("../gpu-tuning");
const {hashTests} = require("../tests/vectors");

const repo = path.resolve(__dirname, "..");
const suiteTimeoutMs = 28 * 60 * 1000;
const maxOutputBytes = 16 << 20;
// Checked after strace exits; this caps memory used to read a log, not capture-time disk growth.
const maxNativeTraceBytes = 32 << 20;
const nativeTraceTimeoutMs = 5 * 60 * 1000;
const nativeTraceTool = "/usr/bin/strace";
const hashPattern = /^[0-9a-f]{64}$/;
const cpuVectorCount = hashTests.filter(definition => !definition.gpu).length;
const portableCpuVectors = hashTests.filter(definition => definition.syclCpu);
const portableCpuAlgos = new Set(portableCpuVectors.map(definition => definition.job.algo));
const gpuAlgos = new Set(hashTests.filter(definition => definition.gpu && !definition.portableOnly).map(definition => definition.job.algo));
const portableCpuTestCount = portableCpuVectors.length + portableCpuAlgos.size * 2 + 1;

/** @param {unknown} value @returns {value is Record<string, unknown>} */
function record(value) {return typeof value === "object" && value !== null && !Array.isArray(value);}

/** @param {string} value @param {string} label */
function absoluteFile(value, label) {
  if (typeof value !== "string" || value.length > 4096 || !path.isAbsolute(value) || /[\0\r\n]/.test(value)) {
    throw new Error("Native candidate requires an absolute " + label);
  }
  return path.normalize(value);
}

/** @param {string} left @param {string} right @param {string} platform */
function samePath(left, right, platform) {
  const a = path.resolve(left), b = path.resolve(right);
  return platform === "win32" ? a.toLowerCase() === b.toLowerCase() : a === b;
}

/** @param {string} base @param {string} candidate @param {string} platform */
function outsideDirectory(base, candidate, platform) {
  const relative = path.relative(base, candidate);
  return !samePath(base, candidate, platform) &&
    (relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative));
}

/** @param {string} file @param {string} expected @param {string} platform @param {string} label */
function regularPath(file, expected, platform, label) {
  const info = fs.lstatSync(file, {bigint: true});
  if (!info.isFile() || info.isSymbolicLink() || !samePath(fs.realpathSync(file), fs.realpathSync(expected), platform)) {
    throw new Error("Native candidate " + label + " must be the expected regular file");
  }
  return fileIdentity(info);
}

/** Preserve exact inode/change timestamps as serializable strings. Hashes alone miss a
 * transient rewrite whose original bytes are restored before the post-run check.
 * @param {import("node:fs").BigIntStats} info */
function fileIdentity(info) {
  if (info.size > BigInt(Number.MAX_SAFE_INTEGER)) {throw new Error("Native qualification file exceeds the safe byte-count limit");}
  return {size: Number(info.size), mode: Number(info.mode), dev: info.dev.toString(), ino: info.ino.toString(),
    nlink: Number(info.nlink), mtimeNs: info.mtimeNs.toString(), ctimeNs: info.ctimeNs.toString()};
}

/** @param {ReturnType<typeof fileIdentity>} left @param {ReturnType<typeof fileIdentity>} right */
function sameFileIdentity(left, right) {
  return left.size === right.size && left.mode === right.mode && left.dev === right.dev && left.ino === right.ino &&
    left.nlink === right.nlink && left.mtimeNs === right.mtimeNs && left.ctimeNs === right.ctimeNs;
}

/** @param {string} file @param {string} platform */
function checkExecutableFormat(file, platform) {
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
  try {
    const stat = fs.fstatSync(fd, {bigint: true}), size = platform === "win32" ? 4096 : 64, header = Buffer.alloc(size);
    if (!stat.isFile() || stat.size < size || fs.readSync(fd, header, 0, size, 0) !== size) {throw new Error("Native candidate executable is truncated");}
    if (platform === "linux") {
      // Both the raw executable and UPX loader must retain ASLR. A packed loader can omit
      // PT_INTERP, so check ET_DYN here; raw layout is also checked before compaction.
      if (header.toString("hex", 0, 4) !== "7f454c46" || header[4] !== 2 || header[5] !== 1 || header[6] !== 1 ||
          header.readUInt16LE(16) !== 3 || header.readUInt16LE(18) !== 62 || header.readUInt32LE(20) !== 1 ||
          header.readUInt16LE(52) !== 64 || header.readUInt16LE(54) !== 56 || !header.readUInt16LE(56)) {
        throw new Error("Native candidate is not a supported position-independent Linux x86-64 ELF executable");
      }
    } else if (platform === "win32") {
      const pe = header.readUInt32LE(0x3c);
      if (header.readUInt16LE(0) !== 0x5a4d || pe < 64 || pe + 26 > header.length || header.readUInt32LE(pe) !== 0x4550 ||
          header.readUInt16LE(pe + 4) !== 0x8664 || !header.readUInt16LE(pe + 6) || header.readUInt16LE(pe + 20) < 240 ||
          header.readUInt16LE(pe + 24) !== 0x20b) {
        throw new Error("Native candidate is not a supported Windows x86-64 PE executable");
      }
    } else {throw new Error("Unsupported native candidate target: " + platform);}
    return fileIdentity(stat);
  } finally {fs.closeSync(fd);}
}

/** Dependency evidence comes from the raw linked executable, never an opaque packed stub.
 * @param {string} raw @param {string} target */
function auditRawExecutable(raw, target) {
  if (target === "linux") {
    nativeDependencies(elfReport(raw, "--dynamic"));
    const versions = elfReport(raw, "--version-info");
    auditNativeGlibc(versions, linuxGlibcBaseline); auditNativeVersionNeeds(versions);
  } else {
    // The Windows release producer seals this map after the final x64 application link.
    // A packed candidate still points here through the entrypoint's separately hashed source.
    verifyNativePeMap(raw);
  }
}

/** Bind a transported raw image to the original raw entrypoint before accepting a non-sibling path.
 * @param {Record<string, unknown>} value @param {string} entrypoint @param {string} source
 * @param {ReturnType<typeof fileIdentity>} sourceInfo @param {string} platform
 * @returns {{sourcePath: string, metadata: {path: string, hash: string, info: ReturnType<typeof fileIdentity>}}} */
function admitTransportedSource(value, entrypoint, source, sourceInfo, platform) {
  /** @param {unknown} file @returns {file is string} */
  const canonicalAbsolute = file => typeof file === "string" && file.length <= 4096 && path.isAbsolute(file) &&
    path.resolve(file) === file && !/[\0\r\n]/.test(file);
  const name = platform === "win32" ? "mom.exe" : "mom", link = value["sourceEntrypoint"];
  if (!record(link) || !hasExactBuildFields(link, ["path", "sha256"]) || !canonicalAbsolute(link["path"]) ||
      path.basename(link["path"]) !== "entrypoint.json" || typeof link["sha256"] !== "string" || !hashPattern.test(link["sha256"])) {
    throw new Error("Relocated native source requires a canonical original entrypoint binding");
  }
  const originalEntrypoint = absoluteFile(link["path"], "original entrypoint"),
    originalInfo = regularPath(originalEntrypoint, originalEntrypoint, platform, "original entrypoint");
  // Canonical filesystem names follow the audit host, not the executable's target OS.
  if (samePath(originalEntrypoint, entrypoint, platform) || originalInfo.size < 1 || originalInfo.size > (8 << 20) ||
      !samePath(fs.realpathSync(originalEntrypoint), originalEntrypoint, process.platform)) {
    throw new Error("Native original entrypoint must be a bounded canonical file outside the packed entrypoint");
  }
  const {value: original, sha256: originalHash} = readBuildJsonSnapshot(originalEntrypoint, undefined, {
    sha256: link["sha256"], message: "Native original entrypoint differs from its packed source binding",
  });
  assertSnapshotUnchanged(originalEntrypoint, platform, "original entrypoint", {...originalInfo, hash: originalHash});
  const originSource = record(original) ? original["source"] : undefined,
    originArtifact = record(original) ? original["artifact"] : undefined,
    originProducer = record(original) ? original["producerReceipt"] : undefined,
    originComponents = record(original) ? original["componentHashes"] : undefined,
    sourcePath = record(originSource) ? originSource["path"] : undefined,
    artifactPath = record(originArtifact) ? originArtifact["path"] : undefined,
    sourceHash = record(originSource) ? originSource["sha256"] : undefined,
    artifactHash = record(originArtifact) ? originArtifact["sha256"] : undefined,
    artifactBytes = record(originArtifact) ? originArtifact["bytes"] : undefined,
    producerPath = record(originProducer) ? originProducer["path"] : undefined,
    producerHash = record(originProducer) ? originProducer["sha256"] : undefined;
  if (!record(original) || original["version"] !== 1 || original["target"] !== platform || original["buildOnly"] !== true ||
      original["publicationQualified"] !== false || original["packingReceipt"] !== undefined || original["sourceEntrypoint"] !== undefined ||
      !record(originSource) || !record(originArtifact) || !record(originProducer) || !record(originComponents) || !record(original["recipe"]) ||
      !canonicalAbsolute(sourcePath) || path.basename(sourcePath) !== name || sourcePath !== artifactPath ||
      typeof sourceHash !== "string" || !hashPattern.test(sourceHash) || sourceHash !== artifactHash ||
      !Number.isSafeInteger(artifactBytes) || artifactBytes !== sourceInfo.size ||
      originSource["bytes"] !== undefined && originSource["bytes"] !== sourceInfo.size ||
      typeof producerPath !== "string" || !canonicalAbsolute(producerPath) ||
      producerPath !== path.join(path.dirname(sourcePath), "result.json") ||
      typeof producerHash !== "string" || !hashPattern.test(producerHash) || originComponents[producerPath] !== producerHash) {
    throw new Error("Native original entrypoint does not bind an unpacked raw source, producer, and target");
  }
  const sourceRecord = value["source"], producerRecord = value["producerReceipt"],
    transportedProducerPath = path.join(path.dirname(source), "result.json"), components = value["componentHashes"],
    actualSourceHash = fileHash(source);
  if (!record(sourceRecord) || sourceRecord["path"] !== source || sourceRecord["bytes"] !== sourceInfo.size ||
      sourceRecord["sha256"] !== sourceHash || actualSourceHash !== sourceHash ||
      !record(producerRecord) || producerRecord["path"] !== transportedProducerPath || producerRecord["sha256"] !== producerHash ||
      value["configFile"] !== original["configFile"] || value["configHash"] !== original["configHash"] ||
      !isDeepStrictEqual(value["generatedDescriptor"], original["generatedDescriptor"]) ||
      !isDeepStrictEqual(value["recipe"], original["recipe"]) || !record(components)) {
    throw new Error("Native packed entrypoint differs from its original raw provenance chain");
  }
  const expectedComponents = {...originComponents};
  if (producerPath !== transportedProducerPath) {
    if (Object.hasOwn(expectedComponents, transportedProducerPath)) {
      throw new Error("Native transported producer receipt collides with an original component");
    }
    delete expectedComponents[producerPath];
  }
  expectedComponents[transportedProducerPath] = producerHash;
  if (!isDeepStrictEqual(components, expectedComponents)) {
    throw new Error("Native packed component hashes do not preserve the original map with only the producer relocated");
  }
  return {sourcePath, metadata: {path: originalEntrypoint, hash: originalHash, info: originalInfo}};
}

/** Admit build metadata without reopening build-only SDK libraries. These hashes bind
 * the tested bytes to their selected producer; they are not a signature or a test pass.
 * @param {Record<string, unknown>} value @param {string} source @param {string} artifact
 * @param {string} platform @param {string} [producerSource] */
function admitProvenance(value, source, artifact, platform, producerSource = source) {
  const metadata = /** @type {{path: string, hash: string, info: ReturnType<typeof fileIdentity>}[]} */ ([]);
  /** @param {unknown} reference @param {string | undefined} expected @param {string} label @param {boolean} [json] */
  const bind = (reference, expected, label, json = false) => {
    if (!record(reference) || typeof reference["path"] !== "string" || typeof reference["sha256"] !== "string" ||
        !hashPattern.test(reference["sha256"])) {throw new Error("Native provenance requires a valid " + label);}
    const file = absoluteFile(reference["path"], label);
    if (expected && !samePath(file, expected, platform)) {throw new Error("Native provenance has an unexpected " + label);}
    const info = regularPath(file, file, platform, label), hash = reference["sha256"];
    const message = "Native provenance hash differs: " + label,
      receipt = json ? readBuildJsonSnapshot(file, undefined, {sha256: hash, message}) : undefined;
    if (!receipt && fileHash(file) !== hash) {throw new Error(message);}
    metadata.push({path: file, hash, info});
    return {file, hash, value: receipt?.value};
  };
  const {value: config} = bind({path: value["configFile"], sha256: value["configHash"]}, undefined, "descriptor", true);
  let configHash = value["configHash"];
  if (value["generatedDescriptor"] !== undefined) {
    if (platform !== "win32") {throw new Error("Generated native descriptors are Windows-only");}
    bind(value["generatedDescriptor"], undefined, "generated descriptor");
    configHash = /** @type {Record<string, unknown>} */ (value["generatedDescriptor"])["sha256"];
  }
  const producer = bind(value["producerReceipt"], path.join(path.dirname(source), "result.json"), "producer receipt", true);
  const rawArtifact = {path: producerSource, sha256: fileHash(source), bytes: fs.statSync(source).size};
  if (typeof configHash !== "string" || !matchesProducerReceipt(producer.value, rawArtifact, configHash,
    platform === "win32" ? "x86_64-pc-windows-msvc" : undefined)) {
    throw new Error("Native producer receipt does not bind its raw executable, target and descriptor");
  }
  const components = value["componentHashes"];
  if (!record(components) || components[producer.file] !== producer.hash) {throw new Error("Native component hashes must include the producer receipt");}
  for (const [file, hash] of Object.entries(components)) {
    if (file !== producer.file) {bind({path: file, sha256: hash}, undefined, "component metadata");}
  }
  const packed = !samePath(source, artifact, platform);
  if (packed) {
    const {value: packing} = bind(value["packingReceipt"], path.join(path.dirname(artifact), "result.json"), "packing receipt", true);
    const matches = (/** @type {unknown} */ item, /** @type {string} */ file) => record(item) &&
      item["path"] === file && item["sha256"] === fileHash(file) && item["bytes"] === fs.statSync(file).size;
    if (!record(packing) || packing["buildOnly"] !== true || packing["publicationQualified"] !== false ||
        !matches(packing["source"], source) || !matches(packing["artifact"], artifact) ||
        typeof packing["restoredSha256"] !== "string" || !hashPattern.test(packing["restoredSha256"]) ||
        !record(packing["payloadAudit"]) || platform === "linux" &&
          (packing["restoredSha256"] !== fileHash(source) || packing["payloadAudit"]["exactOriginalBytes"] !== true) ||
        !record(packing["logs"]) || !hasExactBuildFields(packing["logs"], ["integrity", "pack", "restore"])) {
      throw new Error("Native packing receipt does not bind its executable round trip and saved logs");
    }
    for (const stage of ["pack", "integrity", "restore"]) {
      bind(packing["logs"][stage], path.join(path.dirname(artifact), stage + ".log"), "packing " + stage + " log");
    }
  } else if (value["packingReceipt"] !== undefined) {throw new Error("Unpacked native candidates cannot claim a packing receipt");}
  if (value["notices"] !== undefined) {
    const {file} = bind(value["notices"], path.join(path.dirname(artifact), nativeNoticeName), "distribution notices");
    const bytes = /** @type {Record<string, unknown>} */ (value["notices"])["bytes"];
    if (typeof bytes !== "number" || !Number.isSafeInteger(bytes) || bytes < 1 || bytes > (8 << 20) || fs.statSync(file).size !== bytes) {
      throw new Error("Native distribution notice byte count differs from its bounded metadata");
    }
  }
  // Read the descriptor only as metadata: its SDK paths need not exist on the clean test host.
  if (!record(config)) {throw new Error("Native provenance descriptor must be an object");}
  return {metadata, producer};
}

/** @param {string} entrypoint @param {string} platform @param {string} arch */
function admitCandidate(entrypoint, platform, arch) {
  const entry = absoluteFile(entrypoint, "entrypoint.json");
  if (path.basename(entry).toLowerCase() !== "entrypoint.json" || !["linux", "win32"].includes(platform) || arch !== "x64") {
    throw new Error("Native CPU qualification currently supports Linux/Windows x86-64 entrypoint.json candidates only");
  }
  const receiptInfo = regularPath(entry, entry, platform, "entrypoint.json");
  const {value, sha256: entryHash} = readBuildJsonSnapshot(entry);
  if (!record(value) || value["version"] !== 1 || value["target"] !== platform || value["buildOnly"] !== true || value["publicationQualified"] !== false ||
      !record(value["artifact"]) || !record(value["source"])) {
    throw new Error("Native entrypoint receipt is malformed, wrong-target, or not an unpublished build-only candidate");
  }
  const name = platform === "win32" ? "mom.exe" : "mom", directory = path.dirname(entry), artifactPath = path.join(directory, name);
  const artifact = value["artifact"], source = value["source"], artifactPathValue = artifact["path"], sourcePathValue = source["path"],
    artifactHash = artifact["sha256"], sourceHash = source["sha256"], artifactBytes = artifact["bytes"];
  if (typeof artifactPathValue !== "string" || typeof sourcePathValue !== "string" || typeof artifactHash !== "string" ||
      typeof sourceHash !== "string" || typeof artifactBytes !== "number") {
    throw new Error("Native entrypoint receipt has malformed executable fields");
  }
  const artifactFile = absoluteFile(artifactPathValue, "artifact path"), sourceFile = absoluteFile(sourcePathValue, "raw source path");
  const sourceSibling = path.join(directory + ".unpacked", name),
    localSource = samePath(sourceFile, artifactPath, platform) || samePath(sourceFile, sourceSibling, platform);
  if (!samePath(artifactFile, artifactPath, platform) || !hashPattern.test(artifactHash) ||
      !Number.isSafeInteger(artifactBytes) || artifactBytes < 1 || !hashPattern.test(sourceHash) ||
      !localSource && (sourcePathValue !== path.resolve(sourcePathValue) || path.basename(sourcePathValue) !== name)) {
    throw new Error("Native entrypoint receipt does not bind its expected executable and raw source");
  }
  const artifactInfo = regularPath(artifactFile, artifactPath, platform, "artifact");
  const sourceExpected = samePath(sourceFile, artifactPath, platform) ? artifactPath : localSource ? sourceSibling : sourceFile;
  const sourceInfo = regularPath(sourceFile, sourceExpected, platform, "raw source");
  if (!localSource && !samePath(fs.realpathSync(sourceFile), sourceFile, process.platform)) {
    throw new Error("Relocated native raw source must not use a symlinked parent path");
  }
  if (fileHash(artifactFile) !== artifactHash || artifactInfo.size !== artifactBytes ||
      fileHash(sourceFile) !== sourceHash ||
      samePath(sourceFile, artifactFile, platform) && sourceHash !== artifactHash) {
    throw new Error("Native entrypoint receipt hash/size differs from its executable or raw source");
  }
  const artifactHeader = checkExecutableFormat(artifactFile, platform), sourceHeader = checkExecutableFormat(sourceFile, platform);
  if (!sameFileIdentity(artifactInfo, artifactHeader) || !sameFileIdentity(sourceInfo, sourceHeader)) {
    throw new Error("Native candidate changed during format inspection");
  }
  if (platform === "linux" && !(artifactInfo.mode & 0o111)) {throw new Error("Linux native candidate is not executable");}
  const transported = localSource ? undefined : admitTransportedSource(value, entry, sourceFile, sourceInfo, platform),
    provenance = admitProvenance(value, sourceFile, artifactFile, platform, transported?.sourcePath),
    metadata = transported ? [transported.metadata, ...provenance.metadata] : provenance.metadata;
  validateNoticeAssemblyMetadata(value["noticeAssembly"], value["notices"]);
  if (fileHash(entry) !== entryHash || receiptInfo.size !== fs.statSync(entry).size) {throw new Error("Native entrypoint receipt changed during admission");}
  return {entrypoint: entry, entryHash, entrypointInfo: receiptInfo, entryRecord: value, artifact: artifactFile, artifactHash, artifactBytes,
    artifactInfo: artifactHeader, source: sourceFile, sourceHash, sourceInfo: sourceHeader, metadata, producer: provenance.producer, platform};
}

/** Build a test environment that cannot inherit an old candidate, GPU selector, pool guard
 * bypass, injected Node loader, or non-system dynamic-loader search path.
 * @param {string} executable @param {NodeJS.ProcessEnv} source @param {string} platform @returns {NodeJS.ProcessEnv} */
function testEnvironment(executable, source, platform) {
  const blocked = /^(?:MOM_|NODE_OPTIONS$|NODE_PATH$|FORCE_COLOR$|NO_COLOR$|LD_(?:LIBRARY_PATH|PRELOAD|AUDIT)$|DYLD_|CUDA_|HIP_|HSA_|ROCR_|ZE_|SYCL_|ACPP_|HIPSYCL_|ONEAPI_|OCL_ICD|OPENCL_VENDOR_PATH|VK_ICD|NVIDIA_VISIBLE_DEVICES$|GPU_DEVICE_ORDINAL$)/i;
  /** @type {NodeJS.ProcessEnv} */
  const env = {};
  for (const [key, value] of Object.entries(source)) {if (value !== undefined && !blocked.test(key)) {env[key] = value;}}
  if (platform === "win32") {
    const rootKey = Object.keys(env).find(key => ["systemroot", "windir"].includes(key.toLowerCase()));
    const root = rootKey ? env[rootKey] : undefined, pathKey = Object.keys(env).find(key => key.toLowerCase() === "path") || "Path";
    if (root) {env[pathKey] = path.join(root, "System32");}
  } else {
    // Node harnesses use process.execPath; the native candidate must not find a compiler
    // or helper executable in the build host's inherited PATH.
    env["PATH"] = "/unavailable-native-runtime-tools";
  }
  env["MOM_TEST_EXECUTABLE"] = executable;
  env["MOM_TEST_NO_POOL_NETWORK"] = "1";
  env["MOM_SKIP_MSR"] = "1";
  env["MOM_SKIP_SYCL_ALGO_PARAMS"] = "1";
  env["MOM_GPU_BACKEND"] = "";
  env["NO_COLOR"] = "1";
  return env;
}

/** Create a private location for fresh syscall logs. The CI suffix gives artifact collection an exact prefix.
 * @returns {string} */
function createNativeTraceDirectory() {
  const tempRoot = fs.realpathSync(process.env["RUNNER_TEMP"] || os.tmpdir()), suffix = process.env["MOM_NATIVE_CI_SUFFIX"],
    prefix = validCiSuffix(suffix) ? `mom-native-trace-${suffix}-` : "mom-native-trace-local-",
    directory = fs.mkdtempSync(path.join(tempRoot, prefix));
  try {
    const info = fs.lstatSync(directory), currentUid = typeof process.getuid === "function" ? process.getuid() : undefined;
    if (!info.isDirectory() || info.isSymbolicLink() || info.nlink !== 2 ||
        !samePath(fs.realpathSync(directory), directory, "linux") || !samePath(path.dirname(directory), tempRoot, "linux") ||
        (currentUid !== undefined && info.uid !== currentUid) || (info.mode & 0o077) !== 0 || fs.readdirSync(directory).length !== 0) {
      throw new Error("Native syscall trace directory is not private and owner-controlled");
    }
    return directory;
  } catch (error) {
    try {fs.rmSync(directory, {recursive: true, force: true});} catch { /* no candidate has been started */ }
    throw error;
  }
}

/** After tracing, read only a fresh, private regular log up to the bounded 32 MiB read limit.
 * @param {string} directory @param {string} file @returns {{text: string, bytes: number}} */
function readNativeTrace(directory, file) {
  const root = fs.lstatSync(directory, {bigint: true}), entry = fs.lstatSync(file, {bigint: true}),
    currentUid = typeof process.getuid === "function" ? BigInt(process.getuid()) : undefined;
  if (!root.isDirectory() || root.isSymbolicLink() || (currentUid !== undefined && root.uid !== currentUid) ||
      !samePath(fs.realpathSync(directory), directory, "linux") || path.dirname(file) !== directory ||
      !entry.isFile() || entry.isSymbolicLink() || entry.nlink !== 1n ||
      (currentUid !== undefined && entry.uid !== currentUid) || entry.size < 1n || entry.size > BigInt(maxNativeTraceBytes) ||
      !samePath(fs.realpathSync(file), file, "linux")) {
    throw new Error("Native syscall trace is missing, over the post-run read limit, or not a private regular file");
  }
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK);
  try {
    const before = fs.fstatSync(fd, {bigint: true});
    if (!before.isFile() || before.dev !== entry.dev || before.ino !== entry.ino || before.nlink !== 1n ||
        before.size !== entry.size || before.size > BigInt(maxNativeTraceBytes)) {
      throw new Error("Native syscall trace changed before bounded read");
    }
    const contents = Buffer.alloc(Number(before.size) + 1);
    let bytes = 0;
    while (bytes < contents.length) {
      const read = fs.readSync(fd, contents, bytes, contents.length - bytes, bytes);
      if (read === 0) {break;}
      bytes += read;
    }
    const after = fs.fstatSync(fd, {bigint: true});
    if (bytes !== Number(before.size) || bytes > maxNativeTraceBytes || after.dev !== before.dev || after.ino !== before.ino ||
        after.nlink !== before.nlink || after.size !== before.size || after.mtimeNs !== before.mtimeNs || after.ctimeNs !== before.ctimeNs) {
      throw new Error("Native syscall trace changed while being read");
    }
    return {text: contents.toString("utf8", 0, bytes), bytes};
  } finally {fs.closeSync(fd);}
}

/** @param {unknown} error @param {string} directory @returns {Error} */
function preserveNativeTraceError(error, directory) {
  const failure = error instanceof Error ? error : new Error("Native syscall qualification failed");
  if (!failure.message.includes(directory)) {failure.message += "; syscall traces preserved at " + directory;}
  return failure;
}

/** Run one direct candidate invocation under the fixed system tracer, then apply the same
 * exact-path/DSO guard used by native build probes. The tracee can never be a wrapper command.
 * @param {string} executable @param {string[]} invocation @param {string} tracePath @param {string} directory
 * @param {number} processCount @param {number} expectedStatus @param {typeof spawnSync} runner
 * @param {NodeJS.ProcessEnv} env @param {() => void} assertUnchanged */
function runNativeTraceInvocation(executable, invocation, tracePath, directory, processCount, expectedStatus, runner, env, assertUnchanged) {
  if (fs.existsSync(tracePath)) {throw new Error("Native syscall trace path was not fresh");}
  const args = ["-f", "-qq", "-s", "4096", "-e", "trace=file,process", "-o", tracePath, "--", executable, ...invocation];
  const result = runner(nativeTraceTool, args, {cwd: repo, env, encoding: "utf8", timeout: nativeTraceTimeoutMs,
    maxBuffer: 1 << 20, windowsHide: true});
  assertUnchanged();
  const trace = readNativeTrace(directory, tracePath);
  auditNativeTrace(trace.text, executable, processCount);
  assertUnchanged();
  const output = (result.stdout || "") + "\n" + (result.stderr || "");
  if (result.error || result.signal || result.status !== expectedStatus) {
    const status = result.error ? "error: " + result.error.message : "exit " + String(result.status) + (result.signal ? " signal " + result.signal : "");
    throw Object.assign(new Error("Native syscall tracee failed (" + status + ")"), {stdout: result.stdout || "", stderr: result.stderr || ""});
  }
  return {stdout: result.stdout || "", stderr: result.stderr || "", bytes: trace.bytes, output};
}

/** @typedef {{suite: string, suitePassed: boolean, executable: string, sha256: string, bytes: number,
 * stdout: string, stderr: string}} NativeSuiteResult */
/** Trace one invalid-directive CLI launch and a maintained low-memory CPU vector. RandomX
 * needlessly allocates a full dataset here; algorithm coverage belongs to the separate CPU suite.
 * The parsed device topology determines the exact parent-plus-workers execve count.
 * @param {{artifactHash: string, artifactBytes: number}} candidate @param {string} executable
 * @param {() => void} assertUnchanged @param {CheckOptions} options @returns {NativeSuiteResult} */
function runNativeTraceSuite(candidate, executable, assertUnchanged, options) {
  const tracer = snapshotFile(nativeTraceTool, "linux", "syscall tracer", true), tracerStat = fs.lstatSync(nativeTraceTool);
  if (!(tracer.mode & 0o111) || tracerStat.uid !== 0 || (tracerStat.mode & 0o022) !== 0) {
    throw new Error("Linux syscall tracer must be a root-owned, non-writable regular executable at " + nativeTraceTool);
  }
  const directory = createNativeTraceDirectory(), runner = options.spawn || spawnSync,
    env = testEnvironment(executable, options.env || process.env, "linux");
  try {
    const cli = runNativeTraceInvocation(executable, ["invalid-directive"], path.join(directory, "cli.strace"), directory,
      1, 1, runner, {...env, MOM_COMMAND: "./mom"}, assertUnchanged);
    const cliMarkers = ["Unknown directive invalid-directive", "--save_config:"];
    if (cliMarkers.some(marker => !cli.output.includes(marker))) {
      throw Object.assign(new Error("Native syscall CLI trace did not produce the expected invalid-directive response"),
        {stdout: cli.stdout, stderr: cli.stderr});
    }

    const vector = hashTests.find(definition => !definition.gpu && definition.job.algo === "cn/0"),
      job = vector ? {...vector.job, dev: "cpu"} : undefined;
    if (!vector || !job || typeof job.dev !== "string" || typeof vector.expected !== "string" && !Array.isArray(vector.expected)) {
      throw new Error("Native syscall trace requires a maintained CPU hash vector");
    }
    const workerProcesses = gpuTuning.parseDeviceList(job.dev, job.algo).reduce((total, entry) => total + entry.processes, 0),
      processCount = workerProcesses + 1, expected = Array.isArray(vector.expected) ? vector.expected.join("|") : vector.expected;
    if (!Number.isSafeInteger(workerProcesses) || workerProcesses < 1 || workerProcesses > 1024) {
      throw new Error("Maintained CPU trace vector has an invalid same-executable worker topology");
    }
    const cpu = runNativeTraceInvocation(executable, ["test", job.algo, expected, "--job", JSON.stringify(job)],
      path.join(directory, "cpu.strace"), directory, processCount, 0, runner, env, assertUnchanged);
    if (!cpu.output.includes("PASSED") || /\bFAILED\b/.test(cpu.output)) {
      throw Object.assign(new Error("Native syscall CPU trace did not pass the maintained " + job.algo + " vector"),
        {stdout: cpu.stdout, stderr: cpu.stderr});
    }
    assertSnapshotUnchanged(nativeTraceTool, "linux", "syscall tracer", tracer);
    return {suite: "trace", suitePassed: true, executable, sha256: candidate.artifactHash, bytes: candidate.artifactBytes,
      stdout: cli.stdout + cpu.stdout + "[native-release-check] syscall trace matched candidate-only launches: CLI=1, CPU " +
        job.algo + "=" + processCount + "; trace bytes=" + cli.bytes + "+" + cpu.bytes + "; trace logs: " + directory + "\n",
      stderr: cli.stderr + cpu.stderr};
  } catch (error) {throw preserveNativeTraceError(error, directory);}
}

/** A discrete GPU qualification gets one private AdaptiveCpp database. The test child must not
 * inherit a caller-selected cache, and a failed/aborted child may still be using this directory.
 * @returns {string} */
function createGpuAppdbDirectory() {
  const tempRoot = fs.realpathSync(os.tmpdir());
  if (!fs.statSync(tempRoot).isDirectory()) {throw new Error("System temporary path is not a directory");}
  const directory = fs.mkdtempSync(path.join(tempRoot, "mom-native-acpp-appdb-"));
  try {
    const info = fs.lstatSync(directory), currentUid = typeof process.getuid === "function" ? process.getuid() : undefined;
    const wrongOwner = currentUid !== undefined && info.uid !== currentUid,
      exposedPermissions = process.platform !== "win32" && (info.mode & 0o077) !== 0;
    if (!info.isDirectory() || info.isSymbolicLink() || !samePath(fs.realpathSync(directory), directory, process.platform) ||
        !samePath(path.dirname(directory), tempRoot, process.platform) || fs.readdirSync(directory).length !== 0 ||
        wrongOwner || exposedPermissions) {
      throw new Error("AdaptiveCpp cache directory is not private and owner-controlled");
    }
    return directory;
  } catch (error) {
    try {fs.rmSync(directory, {recursive: true, force: true});} catch { /* no child has been started */ }
    throw error;
  }
}

/** @param {unknown} error @param {string} directory @returns {Error} */
function preserveGpuAppdbError(error, directory) {
  const failure = error instanceof Error ? error : new Error("Native GPU qualification failed");
  if (!failure.message.includes(directory)) {
    failure.message += "; ACPP AppDB cache preserved at " + directory;
  }
  return failure;
}

/** Status zero alone cannot qualify an empty invocation or missing hardware. Discrete GPU lanes
 * include wrapper tests and backend variants, so their count is a floor; CPU and portable CPU
 * lanes have exact maintained test graphs.
 * @param {string} output @param {"cpu" | "gpu" | "gpu-portable-cpu"} suite @param {string} [algo] */
function assertVectorSuiteSummary(output, suite, algo = "") {
  const summary = output.match(/(?:^|\n)ℹ tests (\d+)\r?\nℹ suites (\d+)\r?\nℹ pass (\d+)\r?\nℹ fail (\d+)\r?\nℹ cancelled (\d+)\r?\nℹ skipped (\d+)(?:\r?\n|$)/);
  const count = Number(summary?.[1]), vectors = suite === "cpu" ? cpuVectorCount : suite === "gpu-portable-cpu" ? portableCpuTestCount :
    hashTests.filter(test => test.gpu && !test.portableOnly && test.job.algo === algo).length;
  const portableCoverageComplete = portableCpuVectors.length > 0 && portableCpuAlgos.size > 0 &&
    portableCpuVectors.length === portableCpuAlgos.size && portableCpuAlgos.size === gpuAlgos.size &&
    [...gpuAlgos].every(name => portableCpuAlgos.has(name));
  const countComplete = suite === "cpu" ? count === vectors : suite === "gpu-portable-cpu" ? count === vectors : count >= vectors + 2;
  if (!summary || !vectors || suite === "gpu-portable-cpu" && !portableCoverageComplete || !countComplete || Number(summary[2]) < 1 || summary[1] !== summary[3] ||
      summary[4] !== "0" || summary[5] !== "0" || summary[6] !== "0") {
    const message = suite === "cpu" ? "Native CPU test runner did not report every maintained vector passing without skips" :
      suite === "gpu" ? "Native GPU vector test runner did not report the selected algorithm passing without skips" :
        "Native portable SYCL/OpenCL CPU test runner did not report complete maintained vector coverage passing without skips";
    throw new Error(message);
  }
}

/** @param {unknown} value @param {string[]} keys @param {string} label */
function exactRecord(value, keys, label) {
  if (!record(value) || !hasExactBuildFields(value, keys)) {throw new Error("Native archive " + label + " is malformed");}
  return value;
}

/** Take a stable snapshot of a regular, non-symlink file.
 * @param {string} file @param {string} platform @param {string} label @param {boolean} [singleLink] */
function snapshotFile(file, platform, label, singleLink = false) {
  const info = regularPath(file, file, platform, label);
  if (singleLink && info.nlink !== 1) {throw new Error("Native archive " + label + " must not be hard-linked");}
  const hash = fileHash(file), after = regularPath(file, file, platform, label);
  if (!sameFileIdentity(after, info)) {
    throw new Error("Native archive " + label + " changed while being inspected");
  }
  return {...info, hash};
}

/** @param {string} file @param {string} platform @param {string} label @param {ReturnType<typeof snapshotFile>} original */
function assertSnapshotUnchanged(file, platform, label, original) {
  const current = snapshotFile(file, platform, label);
  if (current.hash !== original.hash || !sameFileIdentity(current, original)) {
    throw new Error("Native " + label + " changed during qualification");
  }
}

/** @typedef {{phase: "windows-cold-start-preflight", executable: string, args: string[], status: 1, expectedStatus: 1,
 * outputMarkers: string[], sha256: string, bytes: number,
 * identity: {mode: number, dev: string, ino: string, nlink: number, mtimeNs: string},
 * ctimeNs: {before: string, after: string}}} ColdStartPreflight */
/** Permit only the one expected Windows first-launch ctime transition before the strict suite.
 * This records an execution boundary, not proof that an OS extended attribute caused the change.
 * @param {string} executable @param {ReturnType<typeof snapshotFile>} original @param {NodeJS.ProcessEnv} environment
 * @param {typeof spawnSync} runner @returns {{snapshot: ReturnType<typeof snapshotFile>, evidence: ColdStartPreflight}} */
function windowsColdStartPreflight(executable, original, environment, runner) {
  assertSnapshotUnchanged(executable, "win32", "cold-start executable", original);
  const env = {...testEnvironment(executable, environment, "win32"), MOM_COMMAND: "./mom"}, args = ["invalid-directive"];
  const result = runner(executable, args, {cwd: repo, env, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20, windowsHide: true});
  const after = snapshotFile(executable, "win32", "cold-start executable");
  if (after.hash !== original.hash || after.size !== original.size || after.mode !== original.mode ||
      after.dev !== original.dev || after.ino !== original.ino || after.nlink !== original.nlink || after.mtimeNs !== original.mtimeNs) {
    throw new Error("Native Windows cold-start preflight changed executable identity, bytes, size, mode or mtime");
  }
  const markers = ["Unknown directive invalid-directive", "--save_config:"], output = (result.stdout || "") + "\n" + (result.stderr || "");
  if (result.error || result.signal || result.status !== 1 || typeof result.stdout !== "string" || typeof result.stderr !== "string" ||
      markers.some(marker => !output.includes(marker))) {
    const status = result.error ? "error: " + result.error.message : "exit " + String(result.status) + (result.signal ? " signal " + result.signal : "");
    throw Object.assign(new Error("Native Windows cold-start preflight did not produce the expected invalid-directive CLI response (" + status + ")"),
      {stdout: typeof result.stdout === "string" ? result.stdout : "", stderr: typeof result.stderr === "string" ? result.stderr : ""});
  }
  const evidence = /** @type {ColdStartPreflight} */ ({phase: "windows-cold-start-preflight", executable, args, status: 1, expectedStatus: 1, outputMarkers: markers,
    sha256: original.hash, bytes: original.size,
    identity: {mode: original.mode, dev: original.dev, ino: original.ino, nlink: original.nlink, mtimeNs: original.mtimeNs},
    ctimeNs: {before: original.ctimeNs, after: after.ctimeNs}});
  return {snapshot: {...original, ctimeNs: after.ctimeNs}, evidence};
}

/** Bind a flat archive to the already-admitted candidate and an explicitly selected tool.
 * List both names and file types before extraction; CMake's verbose listing marks links with
 * `h`/`l`, while regular entries begin with `-`.
 * @param {ReturnType<typeof admitCandidate>} candidate @param {{path: string, tool: string}} request
 * @param {typeof spawnSync} runner */
function admitNativeArchive(candidate, request, runner) {
  if (!request || typeof request !== "object") {throw new Error("Native archive qualification requires an archive and explicit tool path");}
  const archivePath = absoluteFile(request.path, "archive path"), selectedToolPath = absoluteFile(request.tool, "archive tool path"),
    toolPath = absoluteFile(fs.realpathSync(selectedToolPath), "resolved archive tool path");
  const expectedName = candidate.platform === "win32" ? "mom.exe" : "mom";
  const archivePattern = candidate.platform === "win32" ? /^mom-v[0-9][0-9A-Za-z.-]*-win\.zip$/ : /^mom-v[0-9][0-9A-Za-z.-]*-lin\.tgz$/;
  const receiptPath = path.join(path.dirname(archivePath), "native-package.json");
  const archive = snapshotFile(archivePath, candidate.platform, "file"), receiptFile = snapshotFile(receiptPath, candidate.platform, "receipt"), tool = snapshotFile(toolPath, candidate.platform, "tool");
  const {value: receiptValue} = readBuildJsonSnapshot(receiptPath, undefined, {
      sha256: receiptFile.hash, message: "Native archive receipt changed while being read",
    }), version = record(receiptValue) ? receiptValue["version"] : undefined,
    keys = ["archive", "buildOnly", "entryHash", "entrypoint", "executable", "exactOriginalBytes", "members", "notices", "publicationQualified", "tool", "version"],
    lateKeys = [...keys, "originNotices"];
  const requiredKeys = version === 2 ? lateKeys : keys, receiptKeys = Object.keys(record(receiptValue) ? receiptValue : {});
  if (requiredKeys.some(key => !receiptKeys.includes(key)) || receiptKeys.some(key => !requiredKeys.includes(key) && key !== "noticeAssembly")) {
    throw new Error("Native archive receipt is malformed");
  }
  exactRecord(receiptValue, receiptKeys, "receipt");
  const lateNotice = version === 2;
  const candidateDirectory = path.dirname(candidate.entrypoint), archiveDirectory = path.dirname(archivePath),
    correctLocation = lateNotice ? outsideDirectory(candidateDirectory, archiveDirectory, candidate.platform) :
      samePath(archiveDirectory, candidateDirectory, candidate.platform);
  if (!correctLocation || !archivePattern.test(path.basename(archivePath))) {
    throw new Error("Native archive must be the expected flat target archive beside its candidate or in a fresh notice package");
  }
  const receipt = /** @type {{version: unknown, buildOnly: unknown, publicationQualified: unknown, exactOriginalBytes: unknown,
   entrypoint: unknown, entryHash: unknown, archive: unknown, executable: unknown, tool: unknown, notices: unknown,
   originNotices?: unknown, noticeAssembly?: unknown, members: unknown}} */ (receiptValue);
  const packageArchive = /** @type {{bytes: number, path: string, sha256: string}} */ (exactRecord(receipt.archive, ["bytes", "path", "sha256"], "archive binding"));
  const executable = /** @type {{bytes: number, path: string, sha256: string}} */ (exactRecord(receipt.executable, ["bytes", "path", "sha256"], "executable binding"));
  const packageTool = /** @type {{path: string, sha256: string}} */ (exactRecord(receipt.tool, ["path", "sha256"], "tool binding"));
  const notices = /** @type {{bytes: number, path: string, sha256: string}} */ (exactRecord(receipt.notices, ["bytes", "path", "sha256"], "notice binding"));
  const candidateNoticeValue = candidate.entryRecord["notices"];
  /** @type {{bytes: number, path: string, sha256: string} | undefined} */
  let candidateNotices;
  if (candidateNoticeValue !== undefined) {
    candidateNotices = /** @type {{bytes: number, path: string, sha256: string}} */ (exactRecord(candidateNoticeValue, ["bytes", "path", "sha256"], "candidate notice binding"));
  }
  const candidateAssembly = validateNoticeAssemblyMetadata(candidate.entryRecord["noticeAssembly"], candidateNoticeValue),
    packageAssembly = validateNoticeAssemblyMetadata(receipt["noticeAssembly"], notices);
  // The producer's tool is historical provenance, not an executable selection. A clean host
  // may use a different CMake; only the caller-selected tool above is executed and hash-checked.
  if (typeof packageTool.path !== "string" || packageTool.path.length > 4096 ||
      !(path.posix.isAbsolute(packageTool.path) || path.win32.isAbsolute(packageTool.path)) ||
      Array.from(packageTool.path).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127) ||
      typeof packageTool.sha256 !== "string" || !hashPattern.test(packageTool.sha256)) {
    throw new Error("Native archive packaging-tool provenance is malformed");
  }
  const noticesMatch = lateNotice ?
    notices.path === path.join(path.dirname(archivePath), nativeNoticeName) &&
      isDeepStrictEqual(receipt.originNotices, candidateNotices ?? null) :
    isDeepStrictEqual(notices, candidateNotices);
  const expectedAssembly = candidateAssembly && candidateAssembly.notice.sha256 === notices.sha256 &&
    candidateAssembly.notice.bytes === notices.bytes ? candidateAssembly : undefined;
  if ((receipt.version !== 1 && receipt.version !== 2) ||
      receipt.buildOnly !== true || receipt.publicationQualified !== false || receipt.exactOriginalBytes !== true ||
      receipt.entrypoint !== candidate.entrypoint || receipt.entryHash !== candidate.entryHash ||
      packageArchive.path !== archivePath || !Number.isSafeInteger(packageArchive.bytes) || packageArchive.bytes < 1 ||
      packageArchive.bytes !== archive.size || packageArchive.sha256 !== archive.hash ||
      executable.path !== candidate.artifact || executable.bytes !== candidate.artifactBytes || executable.sha256 !== candidate.artifactHash ||
      !noticesMatch ||
      !hashPattern.test(notices.sha256) || !Number.isSafeInteger(notices.bytes) || notices.bytes < 1 || notices.bytes > maximumNoticeBytes ||
      !lateNotice && !isDeepStrictEqual(packageAssembly, expectedAssembly) ||
      lateNotice && expectedAssembly !== undefined && packageAssembly === undefined ||
      !Array.isArray(receipt.members) || receipt.members.length !== 2 || receipt.members[0] !== expectedName || receipt.members[1] !== nativeNoticeName) {
    throw new Error("Native archive receipt does not bind the selected candidate, archive and two flat members");
  }
  const notice = nativeNoticeFile(notices.path);
  const originFile = candidateNotices ? nativeNoticeFile(candidateNotices.path) : undefined;
  if (notice.bytes !== notices.bytes || notice.sha256 !== notices.sha256 ||
      originFile && (originFile.bytes !== candidateNotices?.bytes || originFile.sha256 !== candidateNotices?.sha256) ||
      fileHash(candidate.entrypoint) !== candidate.entryHash) {
    throw new Error("Native archive source notice or entrypoint differs from its package receipt");
  }
  const noticeSnapshot = snapshotFile(notices.path, candidate.platform, "notice");
  const stage = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-archive-check-"));
  try {
    const assertInputsUnchanged = () => {
      assertSnapshotUnchanged(archivePath, candidate.platform, "file", archive);
      assertSnapshotUnchanged(receiptPath, candidate.platform, "receipt", receiptFile);
      assertSnapshotUnchanged(toolPath, candidate.platform, "tool", tool);
      assertSnapshotUnchanged(notices.path, candidate.platform, "notice", noticeSnapshot);
      if (!samePath(fs.realpathSync(selectedToolPath), toolPath, candidate.platform)) {throw new Error("Native archive selected tool changed during qualification");}
    };
    /** @param {string[]} args @param {string} label */
    const runTool = (args, label) => {
      const result = runner(toolPath, ["-E", "tar", ...args], {cwd: stage, encoding: "utf8", timeout: 300000, maxBuffer: 1 << 20, windowsHide: true});
      if (result.error || result.signal || result.status !== 0 || typeof result.stdout !== "string") {
        throw new Error("Native archive " + label + " failed" + (result.error ? ": " + result.error.message : ""));
      }
      return result.stdout;
    };
    const lines = (/** @type {string} */ output, /** @type {string} */ label) => {
      const result = output.replace(/\r/g, "").split("\n");
      if (result.at(-1) === "") {result.pop();}
      if (result.length !== 2 || result.some(line => !line)) {throw new Error("Native archive " + label + " must list exactly two members");}
      return result;
    };
    const listed = lines(runTool(["tf", archivePath], "member listing"), "member listing");
    assertInputsUnchanged();
    if (listed[0] !== expectedName || listed[1] !== nativeNoticeName) {throw new Error("Native archive contains a foreign, duplicate or non-flat member");}
    const verbose = lines(runTool(["tvf", archivePath], "type listing"), "type listing");
    assertInputsUnchanged();
    const expectedSizes = [candidate.artifactBytes, notices.bytes];
    for (let index = 0; index < 2; index++) {
      const match = verbose[index]?.match(/^(-[rwxStTs-]{9})\s+\d+\s+\S+\s+\S+\s+(\d+)\s+.*$/);
      if (!match || !verbose[index]?.endsWith(" " + listed[index])) {
        throw new Error("Native archive members must be regular files; links and other entry types are forbidden");
      }
      if (candidate.platform === "linux" && index === 0 && /[sStT]/.test(match[1] || "")) {
        throw new Error("Native archive executable member must not have set-ID or sticky permission bits");
      }
      if (Number(match[2]) !== expectedSizes[index]) {throw new Error("Native archive declared member size differs from the admitted candidate");}
    }
    assertInputsUnchanged();
    runTool(["xf", archivePath, "--", expectedName, nativeNoticeName], "extraction");
    assertInputsUnchanged();
    const contents = fs.readdirSync(stage).sort();
    if (contents.length !== 2 || contents[0] !== [expectedName, nativeNoticeName].sort()[0] || contents[1] !== [expectedName, nativeNoticeName].sort()[1]) {
      throw new Error("Native archive extraction produced unexpected files");
    }
    const stageInfo = fs.lstatSync(stage), executablePath = path.join(stage, expectedName), extracted = snapshotFile(executablePath, candidate.platform, "executable member", true);
    if (extracted.size !== candidate.artifactBytes || extracted.hash !== candidate.artifactHash ||
        candidate.platform === "linux" && (!(extracted.mode & 0o111) || (extracted.mode & 0o7000) !== 0 ||
          (extracted.mode & 0o777) !== (candidate.artifactInfo.mode & 0o777))) {
      throw new Error("Native archive executable member differs from the admitted candidate or has unsafe permission bits");
    }
    const noticePath = path.join(stage, nativeNoticeName), extractedNotice = nativeNoticeFile(noticePath);
    if (fs.lstatSync(noticePath).nlink !== 1) {throw new Error("Native archive notice member must not be hard-linked");}
    if (extractedNotice.bytes !== notices.bytes || extractedNotice.sha256 !== notices.sha256) {
      throw new Error("Native archive notice member differs from its package receipt");
    }
    return {stage, stageInfo, path: archivePath, archive, receiptPath, receipt: receiptFile, selectedToolPath, toolPath, tool, executablePath, executable: extracted,
      noticePath, noticesPath: notices.path, noticeSnapshot, notices};
  } catch (error) {
    fs.rmSync(stage, {recursive: true, force: true});
    throw error;
  }
}

/** @param {ReturnType<typeof admitCandidate>} candidate @param {ReturnType<typeof admitNativeArchive>} archive */
function assertArchiveUnchanged(candidate, archive) {
  const stageInfo = fs.lstatSync(archive.stage), members = fs.readdirSync(archive.stage).sort(), expected = [candidate.platform === "win32" ? "mom.exe" : "mom", nativeNoticeName].sort();
  if (!stageInfo.isDirectory() || stageInfo.isSymbolicLink() || stageInfo.dev !== archive.stageInfo.dev || stageInfo.ino !== archive.stageInfo.ino ||
      members.length !== 2 || members[0] !== expected[0] || members[1] !== expected[1]) {
    throw new Error("Native archive extraction directory or members changed during qualification");
  }
  assertSnapshotUnchanged(archive.path, candidate.platform, "file", archive.archive);
  assertSnapshotUnchanged(archive.receiptPath, candidate.platform, "receipt", archive.receipt);
  assertSnapshotUnchanged(archive.toolPath, candidate.platform, "tool", archive.tool);
  assertSnapshotUnchanged(archive.noticesPath, candidate.platform, "notice", archive.noticeSnapshot);
  assertSnapshotUnchanged(archive.executablePath, candidate.platform, "executable member", archive.executable);
  if (!samePath(fs.realpathSync(archive.selectedToolPath), archive.toolPath, candidate.platform)) {throw new Error("Native archive selected tool changed during qualification");}
  const notice = nativeNoticeFile(archive.noticePath), originalNotice = nativeNoticeFile(archive.noticesPath);
  if (fs.lstatSync(archive.noticePath).nlink !== 1 || notice.sha256 !== archive.notices.sha256 || notice.bytes !== archive.notices.bytes ||
      originalNotice.sha256 !== archive.notices.sha256 || originalNotice.bytes !== archive.notices.bytes || fileHash(candidate.entrypoint) !== candidate.entryHash) {
    throw new Error("Native archive notice or entrypoint changed during qualification");
  }
}

/** In-process audit/spawn overrides are trusted test seams, never candidate data or CLI options.
 * @typedef {{platform?: string, arch?: string, env?: NodeJS.ProcessEnv,
 * spawn?: typeof spawnSync, audit?: (raw: string, target: string) => void,
 * archive?: {path: string, tool: string}, gpu?: {vendor: string, algo: string}}} CheckOptions */
/** @param {{artifactHash: string, artifactBytes: number}} candidate @param {"cpu" | "cli" | "gpu" | "gpu-portable-cpu" | "trace"} suite
 * @param {string} executable @param {() => void} assertUnchanged @param {CheckOptions} options @param {string} platform
 * @returns {NativeSuiteResult} */
function runNativeSuite(candidate, suite, executable, assertUnchanged, options, platform) {
  if (suite === "trace") {return runNativeTraceSuite(candidate, executable, assertUnchanged, options);}
  const runner = options.spawn || spawnSync;
  if (suite === "cli") {
    const env = {...testEnvironment(executable, options.env || process.env, platform), MOM_COMMAND: "./mom"};
    const normalize = (/** @type {string} */ text) => text.replaceAll("\r\n", "\n").replace(/^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d /gm, "");
    const failures = [], transcripts = [];
    for (const [args, marker] of /** @type {[string[], string][]} */ ([[[], "No directive specified"], [["invalid-directive"], "Unknown directive invalid-directive"]])) {
      const invoke = (/** @type {string} */ command, /** @type {string[]} */ tokens) => {
        const result = runner(command, tokens, {cwd: repo, env, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20, windowsHide: true});
        assertUnchanged();
        return result;
      };
      // This is the unchanged controller as a test oracle, never a production Node dependency.
      const expected = invoke(process.execPath, ["--max-old-space-size=128", "--require", path.join(repo, "tests/common/no-pool-network.js"), path.join(repo, "mom.js"), ...args]);
      const actual = invoke(executable, args), ok = (/** @type {typeof actual} */ result) =>
        !result.error && !result.signal && result.status === 1 && typeof result.stdout === "string" && typeof result.stderr === "string";
      const output = (expected.stdout || "") + (expected.stderr || "");
      // Exit status/one marker would miss abandoned queued output. Require a complete oracle
      // help footer too, then compare both streams without stripping arbitrary diagnostic text.
      if (!ok(expected) || !ok(actual) || !output.includes(marker) || !output.includes("--save_config:") ||
          normalize(actual.stdout || "") !== normalize(expected.stdout || "") || normalize(actual.stderr || "") !== normalize(expected.stderr || "")) {
        failures.push(args[0] || "help");
      }
      transcripts.push({args, expected: {status: expected.status, signal: expected.signal, error: expected.error?.message, stdout: expected.stdout, stderr: expected.stderr},
        actual: {status: actual.status, signal: actual.signal, error: actual.error?.message, stdout: actual.stdout, stderr: actual.stderr}});
    }
    if (failures.length) {throw Object.assign(new Error("Native CLI transcript parity failed: " + failures.join(", ")), {stdout: JSON.stringify(transcripts, null, 2), stderr: ""});}
    return {suite, suitePassed: true, executable, sha256: candidate.artifactHash, bytes: candidate.artifactBytes,
      stdout: "[native-release-check] Complete CLI output matched the original Node controller (2/2).\n", stderr: ""};
  }
  const env = testEnvironment(executable, options.env || process.env, platform);
  let appdbDirectory;
  if (suite === "gpu" && options.gpu) {
    delete env["MOM_SKIP_SYCL_ALGO_PARAMS"];
    Object.assign(env, {MOM_GPU_BACKEND: options.gpu.vendor, MOM_GPU_TEST_VENDORS: options.gpu.vendor,
      MOM_GPU_TEST_ALGO: options.gpu.algo, MOM_REQUIRE_GPU_TESTS: "1"});
    appdbDirectory = createGpuAppdbDirectory();
    env["ACPP_APPDB_DIR"] = appdbDirectory;
  } else if (suite === "gpu-portable-cpu") {
    // This lane is fixed: caller-selected algorithms, vendors, devices, and SYCL runtimes must not
    // narrow or redirect the same-artifact portable CPU check.
    delete env["MOM_SKIP_SYCL_ALGO_PARAMS"];
    Object.assign(env, {MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: "cpu", ONEAPI_DEVICE_SELECTOR: "opencl:cpu",
      MOM_REQUIRE_PORTABLE_CPU_TESTS: "1"});
  }
  const testSuite = suite === "gpu" ? "gpu-discrete" : suite;
  try {
    const result = runner(process.execPath, [path.join(repo, "tests", "run-hash.js"), testSuite], {
      cwd: repo, env, encoding: "utf8",
      timeout: suiteTimeoutMs, maxBuffer: maxOutputBytes, windowsHide: true,
    });
    assertUnchanged();
    if (result.error || result.signal || result.status !== 0) {
      const status = result.error ? "error: " + result.error.message : "exit " + String(result.status) + (result.signal ? " signal " + result.signal : "");
      throw Object.assign(new Error("Native " + (suite === "gpu" ? "GPU" : "CPU") + " vector suite failed (" + status + ")"), {
        stdout: result.stdout || "", stderr: result.stderr || "", status: result.status, signal: result.signal,
      });
    }
    try {
      assertVectorSuiteSummary(result.stdout || "", suite, options.gpu?.algo);
    } catch (error) {
      throw Object.assign(error instanceof Error ? error : new Error(String(error)), {stdout: result.stdout || "", stderr: result.stderr || ""});
    }
    const passed = {suite, suitePassed: true, executable, sha256: candidate.artifactHash, bytes: candidate.artifactBytes,
      stdout: result.stdout || "", stderr: result.stderr || ""};
    if (appdbDirectory) {
      const completedDirectory = appdbDirectory;
      fs.rmSync(completedDirectory, {recursive: true});
      appdbDirectory = undefined;
    }
    return passed;
  } catch (error) {
    if (appdbDirectory) {throw preserveGpuAppdbError(error, appdbDirectory);}
    throw error;
  }
}

/** @param {string} entrypoint @param {string} suite @param {CheckOptions} [options]
 * @returns {{suite: string, suitePassed: boolean, executable: string, sha256: string, bytes: number, stdout: string, stderr: string,
 * archive?: string, gpu?: {vendor: string, algo: string}, coldStartPreflight?: ColdStartPreflight}} */
function checkNativeRelease(entrypoint, suite, options = {}) {
  validateSuite(suite, options.gpu);
  if (suite === "trace") {throw new Error("Native syscall trace is runtime-only and requires an explicit executable SHA256");}
  const gpu = options.gpu;
  const platform = options.platform || process.platform, arch = options.arch || process.arch;
  const candidate = admitCandidate(entrypoint, platform, arch);
  (options.audit || auditRawExecutable)(candidate.source, candidate.platform);
  assertCandidateUnchanged(candidate);
  const archive = options.archive === undefined ? undefined : admitNativeArchive(candidate, options.archive, options.spawn || spawnSync);
  try {
    const executable = archive?.executablePath || candidate.artifact;
    const assertUnchanged = () => {assertCandidateUnchanged(candidate); if (archive) {assertArchiveUnchanged(candidate, archive);}};
    assertUnchanged();
    let coldStartPreflight;
    if (candidate.platform === "win32") {
      const original = archive ? archive.executable : {...candidate.artifactInfo, hash: candidate.artifactHash};
      const warmed = windowsColdStartPreflight(executable, original, options.env || process.env, options.spawn || spawnSync);
      coldStartPreflight = warmed.evidence;
      if (archive) {archive.executable = warmed.snapshot;} else {
        candidate.artifactInfo = {...candidate.artifactInfo, ctimeNs: warmed.snapshot.ctimeNs};
        if (samePath(candidate.source, candidate.artifact, candidate.platform)) {
          candidate.sourceInfo = {...candidate.sourceInfo, ctimeNs: warmed.snapshot.ctimeNs};
        }
      }
      // Formal runs are strict against the checked post-preflight ctime; other baselines stay original.
      assertUnchanged();
    }
    const result = runNativeSuite(candidate, /** @type {"cpu" | "cli" | "gpu" | "gpu-portable-cpu"} */ (suite), executable, assertUnchanged, options, platform);
    return {...result, ...(coldStartPreflight ? {coldStartPreflight} : {}), ...(gpu ? {gpu: {...gpu}} : {}),
      ...(archive ? {executable: candidate.artifact, archive: archive.path} : {})};
  } finally {if (archive) {fs.rmSync(archive.stage, {recursive: true, force: true});}}
}

/** Keep the same vector and device-selection contract at both runtime entrypoints.
 * @param {string} suite @param {CheckOptions['gpu']} gpu */
function validateSuite(suite, gpu) {
  if (!["cpu", "cli", "gpu", "gpu-portable-cpu", "trace"].includes(suite)) {throw new Error("Unsupported native qualification suite: " + suite + "; choose cpu, cli, gpu, gpu-portable-cpu or trace");}
  if (suite === "gpu" ? !gpu || !hasExactBuildFields(gpu, ["algo", "vendor"]) ||
      !["intel", "nvidia", "amd"].includes(gpu.vendor) ||
      !hashTests.some(test => test.gpu && !test.portableOnly && test.job.algo === gpu.algo) : gpu !== undefined) {
    throw new Error("Native GPU qualification requires one explicit supported vendor and algorithm; GPU settings apply only to the gpu suite");
  }
}

/** Audit original build evidence where its paths exist, without executing the target image.
 * The resulting artifact hash, not the raw-image hash, binds a later target-host runtime check.
 * @param {string} entrypoint @param {Pick<CheckOptions, "audit" | "spawn" | "archive">} [options] */
function auditNativeRelease(entrypoint, options = {}) {
  if (!record(options) || Object.keys(options).some(key => !["audit", "spawn", "archive"].includes(key))) {
    throw new Error("Build audit accepts only audit, spawn and archive options; the target comes from the candidate");
  }
  const entry = absoluteFile(entrypoint, "entrypoint.json"), target = readBuildJson(entry)?.target;
  if (typeof target !== "string" || !["linux", "win32"].includes(target)) {throw new Error("Native build audit requires a valid candidate target");}
  const candidate = admitCandidate(entry, target, "x64");
  (options.audit || auditRawExecutable)(candidate.source, candidate.platform);
  assertCandidateUnchanged(candidate);
  const archive = options.archive === undefined ? undefined : admitNativeArchive(candidate, options.archive, options.spawn || spawnSync);
  try {
    assertCandidateUnchanged(candidate); if (archive) {assertArchiveUnchanged(candidate, archive);}
    return {scope: "build-audit", target, entrypoint: candidate.entrypoint, entryHash: candidate.entryHash,
      sourceSha256: candidate.sourceHash, sha256: candidate.artifactHash, bytes: candidate.artifactBytes,
      executable: candidate.artifact, buildEvidencePassed: true, runtimeTested: false,
      archiveChecked: archive !== undefined, publicationQualified: false,
      ...(archive ? {archive: archive.path, archiveSha256: archive.archive.hash} : {})};
  } finally {if (archive) {fs.rmSync(archive.stage, {recursive: true, force: true});}}
}

/** Run the maintained suites on a transported image, never reinterpret or rewrite build paths.
 * This is runtime-only evidence: CI must pair it with the original build audit's exact hash.
 * @param {string} executable @param {string} expectedSha256 @param {string} suite
 * @param {Pick<CheckOptions, "env" | "spawn" | "gpu">} [options] */
function checkNativeExecutable(executable, expectedSha256, suite, options = {}) {
  if (!record(options) || Object.keys(options).some(key => !["env", "spawn", "gpu"].includes(key))) {
    throw new Error("Runtime-only checks accept only env, spawn and gpu options; build/target overrides are forbidden");
  }
  validateSuite(suite, options.gpu);
  const platform = process.platform;
  if (!["linux", "win32"].includes(platform) || process.arch !== "x64") {throw new Error("Native runtime checks require a Linux/Windows x64 host");}
  if (suite === "trace" && platform !== "linux") {throw new Error("Native syscall trace qualification supports Linux only");}
  if (typeof expectedSha256 !== "string" || !hashPattern.test(expectedSha256)) {throw new Error("Runtime-only checks require an explicit expected SHA256 from the builder audit");}
  const file = absoluteFile(executable, "runtime executable"), original = snapshotFile(file, platform, "runtime executable");
  if (original.hash !== expectedSha256) {throw new Error("Native runtime executable differs from the builder audit's expected SHA256");}
  checkExecutableFormat(file, platform);
  if (platform === "linux" && !(original.mode & 0o111)) {throw new Error("Linux native runtime executable is not executable");}
  let baseline = original, coldStartPreflight;
  if (platform === "win32") {
    const warmed = windowsColdStartPreflight(file, original, options.env || process.env, options.spawn || spawnSync);
    baseline = warmed.snapshot; coldStartPreflight = warmed.evidence;
  }
  const unchanged = () => assertSnapshotUnchanged(file, platform, "runtime executable", baseline);
  unchanged();
  const result = runNativeSuite({artifactHash: original.hash, artifactBytes: original.size}, /** @type {"cpu" | "cli" | "gpu" | "gpu-portable-cpu" | "trace"} */ (suite),
    file, unchanged, options, platform);
  return {...result, scope: "runtime-only", target: platform, buildProvenanceChecked: false, publicationQualified: false,
    ...(coldStartPreflight ? {coldStartPreflight} : {}),
    ...(options.gpu ? {gpu: {...options.gpu}} : {})};
}

/** @param {ReturnType<typeof admitCandidate>} candidate */
function assertCandidateUnchanged(candidate) {
  const check = (/** @type {string} */ file, /** @type {string} */ hash,
    /** @type {ReturnType<typeof fileIdentity>} */ original, /** @type {string} */ label) =>
    assertSnapshotUnchanged(file, candidate.platform, "candidate " + label, {...original, hash});
  check(candidate.entrypoint, candidate.entryHash, candidate.entrypointInfo, "entrypoint");
  check(candidate.artifact, candidate.artifactHash, candidate.artifactInfo, "artifact");
  check(candidate.source, candidate.sourceHash, candidate.sourceInfo, "raw source");
  for (const item of candidate.metadata) {check(item.path, item.hash, item.info, "provenance");}
}

if (require.main === module) {
  try {
    const [entrypoint, suite, ...args] = process.argv.slice(2);
    let expectedSha256;
    if (args[0] === "--sha256" && typeof args[1] === "string") {expectedSha256 = args[1]; args.splice(0, 2);}
    let gpu;
    if (suite === "gpu" && args[0] === "--vendor" && args[1] && args[2] === "--algo" && args[3]) {
      gpu = {vendor: args[1], algo: args[3]}; args.splice(0, 4);
    }
    let archive;
    if (args.length === 0) {archive = undefined;} else if (args.length === 4 && args[0] === "--archive" && typeof args[1] === "string" && args[2] === "--tool" && typeof args[3] === "string") {
      archive = {path: args[1], tool: args[3]};
    } else {archive = null;}
    if (!entrypoint || !suite || archive === null) {
      throw new Error("Usage: check-native-release.js /candidate/entrypoint.json audit|cpu|cli|gpu|gpu-portable-cpu [--vendor intel|nvidia|amd --algo <algorithm>] [--archive /archive --tool /cmake] | /executable cpu|cli|gpu|gpu-portable-cpu|trace --sha256 <builder-hash> [--vendor intel|nvidia|amd --algo <algorithm>]");
    }
    const options = {...(archive ? {archive} : {}), ...(gpu ? {gpu} : {})};
    if (suite === "audit" && expectedSha256 === undefined) {
      console.log(JSON.stringify(auditNativeRelease(entrypoint, options)));
    } else {
      const result = expectedSha256 === undefined ? checkNativeRelease(entrypoint, suite, options) :
        checkNativeExecutable(entrypoint, expectedSha256, suite, options);
      if (result.stdout) {process.stdout.write(result.stdout);}
      if (result.stderr) {process.stderr.write(result.stderr);}
      if (result.coldStartPreflight) {console.log(JSON.stringify({coldStartPreflight: result.coldStartPreflight}));}
      console.log("[native-release-check] " + (expectedSha256 === undefined ? "" : "runtime-only ") + result.suite +
        (result.gpu ? " " + result.gpu.vendor + "/" + result.gpu.algo : "") + " suite passed for " +
        ("archive" in result ? result.archive : result.executable) + "; SHA256=" + result.sha256 +
        "; this does not qualify untested devices/algorithms, performance or publication." +
        (expectedSha256 === undefined ? "" : " Build provenance requires the matching builder audit."));
    }
  } catch (error) {
    if (error && typeof error === "object") {
      const details = /** @type {{stdout?: string, stderr?: string}} */ (error);
      if (details.stdout) {process.stdout.write(details.stdout);}
      if (details.stderr) {process.stderr.write(details.stderr);}
    }
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  }
}

module.exports = {checkNativeRelease, auditNativeRelease, checkNativeExecutable, admitCandidate, testEnvironment, suiteTimeoutMs};
