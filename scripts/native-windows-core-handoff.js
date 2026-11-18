"use strict";

// Admit the exact three-file Windows Core transport without rewriting its producer receipt.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildText, nativeSourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {isStaticArchive} = require("./native-audit");
const {ioPins} = require("./build-native-io");

const maxArchiveBytes = 256 * 1024 * 1024, maxReceiptBytes = 8 * 1024 * 1024;
const hashPattern = /^[a-f0-9]{64}$/;
const coreFiles = ["cpu-core.lib", "handoff.json", "inputs.json"];
const repo = path.resolve(__dirname, "..");

/** @param {unknown} value @param {string[]} keys */
function hasExactKeys(value, keys) {
  return Boolean(value && typeof value === "object" && !Array.isArray(value) &&
    isDeepStrictEqual(Object.keys(value).sort(), [...keys].sort()));
}

/** @param {unknown} value @returns {value is string} */
function producerAbsolutePath(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 4096 &&
    (path.win32.isAbsolute(value) || path.posix.isAbsolute(value)) && !/[\0\r\n]/.test(value);
}

/** @param {string} file @param {number} maximum @param {string} label */
function readPinnedJson(file, maximum, label) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size < 1 || stat.size > maximum ||
      fs.realpathSync(file) !== file || path.resolve(file) !== file) {
    throw new Error("Windows Core handoff " + label + " is not a bounded canonical file");
  }
  const sha256 = fileHash(file), text = readBuildText(file, maximum), after = fs.lstatSync(file);
  if (!after.isFile() || after.isSymbolicLink() || after.nlink !== 1 || after.dev !== stat.dev || after.ino !== stat.ino ||
      after.size !== stat.size || fs.realpathSync(file) !== file || fileHash(file) !== sha256) {
    throw new Error("Windows Core handoff " + label + " changed during admission");
  }
  let value;
  try {value = JSON.parse(text);} catch (error) {throw new Error("Windows Core handoff " + label + " is invalid JSON", {cause: error});}
  return {value, sha256, identity: {dev: stat.dev, ino: stat.ino, size: stat.size}};
}

/** @param {string} file @param {{dev:number,ino:number,size:number}} identity @param {string} sha256 @param {string} label */
function verifyUnchangedFile(file, identity, sha256, label) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.dev !== identity.dev || stat.ino !== identity.ino ||
      stat.size !== identity.size || fs.realpathSync(file) !== file || fileHash(file) !== sha256) {
    throw new Error("Windows Core handoff " + label + " changed during admission");
  }
}

/**
 * Validate a restored `native-windows-core-<suffix>` directory against the
 * externally pinned, unchanged handoff.json. Header hashes remain producer provenance:
 * their original Windows SDK files are intentionally not part of this small transport.
 * @param {string} root @param {{handoffSha256:string}} expected
 */
function windowsNativeCoreHandoffInputs(root, expected) {
  if (typeof root !== "string" || !path.isAbsolute(root) || path.resolve(root) !== root ||
      !expected || Object.keys(expected).join(",") !== "handoffSha256" ||
      typeof expected.handoffSha256 !== "string" || !hashPattern.test(expected.handoffSha256)) {
    throw new Error("Windows Core handoff needs a canonical root and external handoff SHA-256");
  }
  const rootStat = fs.lstatSync(root);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(root) !== root ||
      !isDeepStrictEqual(fs.readdirSync(root).sort(), coreFiles)) {
    throw new Error("Windows Core handoff root is redirected or has unexpected members");
  }
  const handoffFile = path.join(root, "handoff.json"), receiptFile = path.join(root, "inputs.json"), archive = path.join(root, "cpu-core.lib"),
    handoffRecord = readPinnedJson(handoffFile, 1 << 20, "handoff receipt"), receiptRecord = readPinnedJson(receiptFile, maxReceiptBytes, "producer receipt"),
    handoff = handoffRecord.value, receipt = receiptRecord.value;
  if (handoffRecord.sha256 !== expected.handoffSha256 ||
      !hasExactKeys(handoff, ["format", "target", "architecture", "archive", "producer", "libuv", "buildOnly", "executed", "publicationQualified"]) ||
      handoff.format !== "mom-native-windows-core-handoff-v1" || handoff.target !== "win32" ||
      handoff.architecture !== "x64" || handoff.buildOnly !== true || handoff.executed !== false || handoff.publicationQualified !== false ||
      !hasExactKeys(handoff.archive, ["file", "bytes", "sha256"]) || handoff.archive.file !== "cpu-core.lib" ||
      !Number.isSafeInteger(handoff.archive.bytes) || handoff.archive.bytes < 8 || handoff.archive.bytes > maxArchiveBytes ||
      typeof handoff.archive.sha256 !== "string" || !hashPattern.test(handoff.archive.sha256) ||
      !hasExactKeys(handoff.producer, ["script", "receipt", "receiptSha256"]) || handoff.producer.script !== "scripts/build-native-windows-core.js" ||
      handoff.producer.receipt !== "inputs.json" || handoff.producer.receiptSha256 !== receiptRecord.sha256 ||
      !hasExactKeys(handoff.libuv, ["version", "archiveSha256", "fetchReceiptSha256"]) || handoff.libuv.version !== "1.51.0" ||
      handoff.libuv.archiveSha256 !== ioPins.libuv.sha256 || typeof handoff.libuv.fetchReceiptSha256 !== "string" ||
      !hashPattern.test(handoff.libuv.fetchReceiptSha256)) {
    throw new Error("Windows Core handoff differs from its pinned build-only producer contract");
  }
  const archiveStat = fs.lstatSync(archive);
  if (!archiveStat.isFile() || archiveStat.isSymbolicLink() || archiveStat.nlink !== 1 || archiveStat.size !== handoff.archive.bytes ||
      fs.realpathSync(archive) !== archive || !isStaticArchive(archive) || fileHash(archive) !== handoff.archive.sha256) {
    throw new Error("Windows Core handoff archive differs from its producer pin");
  }
  const fact = receipt.archive;
  if (receipt.version !== 1 || receipt.target !== "win32" || receipt.gpu !== true || receipt.portable !== true ||
      receipt.buildOnly !== true || receipt.executed !== false || receipt.publicationQualified !== false || !receipt.compiler ||
      typeof receipt.compiler.version !== "string" || !/^2026\.2(?:\.|$)/.test(receipt.compiler.version) ||
      receipt.compiler.target !== "x86_64-pc-windows-msvc" ||
      !producerAbsolutePath(fact?.path) || path.win32.basename(fact.path).toLowerCase() !== "cpu-core.lib" ||
      receipt.objectSha256 !== handoff.archive.sha256 || typeof fact.sha256 !== "string" || fact.sha256 !== handoff.archive.sha256 ||
      fact.bytes !== handoff.archive.bytes || fact.stubs !== false ||
      !Number.isSafeInteger(fact.coffTableCount) || fact.coffTableCount < 1 || fact.coffTableCount !== fact.machineCount ||
      fact.machineCount !== fact.x64MachineCount || !Array.isArray(receipt.objects) || receipt.objects.length !== 1 || receipt.objects[0] !== fact.path) {
    throw new Error("Windows Core producer receipt fails the portable x64 consumer contract");
  }
  const headerHashes = receipt.headerHashes;
  if (!headerHashes || typeof headerHashes !== "object" || Array.isArray(headerHashes) || !Object.keys(headerHashes).length ||
      Object.keys(headerHashes).length > 4096 || Object.entries(headerHashes).some(([file, hash]) => !producerAbsolutePath(file) ||
        !/\.(?:h|hpp|inc)$/i.test(file) || typeof hash !== "string" || !hashPattern.test(hash))) {
    throw new Error("Windows Core producer receipt has malformed libuv header provenance");
  }
  const headerPaths = Object.keys(headerHashes), uv = headerPaths.filter(file => /[\\/]uv\.h$/i.test(file)), win = headerPaths.filter(file => /[\\/]uv[\\/]win\.h$/i.test(file));
  if (uv.length !== 1 || win.length !== 1 || path.win32.dirname(uv[0] || "").toLowerCase() !==
      path.win32.dirname(path.win32.dirname(win[0] || "")).toLowerCase()) {
    throw new Error("Windows Core producer receipt omits its exact libuv uv.h / uv/win.h pair");
  }
  verifySourceInputs(receipt.sourceState, nativeSourceInputs(repo, "cpu"), "Windows Core handoff");
  const archiveIdentity = {dev: archiveStat.dev, ino: archiveStat.ino, size: archiveStat.size};
  verifyUnchangedFile(handoffFile, handoffRecord.identity, handoffRecord.sha256, "handoff receipt");
  verifyUnchangedFile(receiptFile, receiptRecord.identity, receiptRecord.sha256, "producer receipt");
  verifyUnchangedFile(archive, archiveIdentity, handoff.archive.sha256, "archive");
  const files = {[handoffFile]: handoffRecord.sha256, [receiptFile]: receiptRecord.sha256, [archive]: handoff.archive.sha256};
  const finalRootStat = fs.lstatSync(root);
  if (!finalRootStat.isDirectory() || finalRootStat.isSymbolicLink() || finalRootStat.dev !== rootStat.dev || finalRootStat.ino !== rootStat.ino ||
      !isDeepStrictEqual(fs.readdirSync(root).sort(), coreFiles) || fs.realpathSync(root) !== root) {
    throw new Error("Windows Core handoff membership changed during admission");
  }
  return {kind: "core", handoff: handoffFile, handoffSha256: handoffRecord.sha256,
    receiptFile, receiptSha256: receiptRecord.sha256, archive, archiveSha256: handoff.archive.sha256, bytes: handoff.archive.bytes, files};
}

module.exports = {windowsNativeCoreHandoffInputs};
