"use strict";

// Build-only archive production. Only the native image and its data-only notice file ship;
// the notice does not itself fulfill license or source obligations. SDK receipts, Node,
// addons, shared libraries and compiler tools must never enter the package.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const {hasExactBuildFields, fileHash, readBuildJsonSnapshot, readBuildText, requireBuildFields} = require("./build-source-inputs");
const {absolutePath} = require("./build-native-release");

const nativeNoticeName = "THIRD-PARTY-NOTICES.txt", maximumNoticeBytes = 8 << 20, noticeHashPattern = /^[0-9a-f]{64}$/;

/** Read only a small, regular non-symlink UTF-8 data file and detect replacement or mutation.
 * @param {string} file @returns {{bytes: number, sha256: string}} */
function nativeNoticeFile(file) {
  if (path.basename(file) !== nativeNoticeName) {throw new Error("Native package notice must be named " + nativeNoticeName);}
  const beforePath = fs.lstatSync(file, {throwIfNoEntry: false, bigint: true});
  if (!beforePath?.isFile() || beforePath.size < 1n || beforePath.size > BigInt(maximumNoticeBytes) ||
      (beforePath.mode & 0o111n) !== 0n || fs.realpathSync(file) !== file) {
    throw new Error("Native package notice must be a bounded, non-executable regular file, not a symlink");
  }
  const text = readBuildText(file, maximumNoticeBytes), data = Buffer.from(text, "utf8");
  const afterPath = fs.lstatSync(file, {throwIfNoEntry: false, bigint: true});
  if (!afterPath?.isFile() || afterPath.dev !== beforePath.dev || afterPath.ino !== beforePath.ino ||
      afterPath.size !== beforePath.size || afterPath.mtimeNs !== beforePath.mtimeNs ||
      afterPath.ctimeNs !== beforePath.ctimeNs || fs.realpathSync(file) !== file || BigInt(data.length) !== beforePath.size) {
    throw new Error("Native package notice changed while reading or is not valid UTF-8 text");
  }
  if (text.includes("\0")) {throw new Error("Native package notice must not contain NUL bytes");}
  const sha256 = createHash("sha256").update(data).digest("hex");
  if (fileHash(file) !== sha256) {throw new Error("Native package notice changed or is not valid UTF-8 text");}
  return {bytes: data.length, sha256};
}

/** Bind the required notice sidecar to the build receipt before archive creation.
 * @param {unknown} value @param {string} directory */
function nativeNotice(value, directory) {
  const expected = path.join(directory, nativeNoticeName), record = /** @type {{path?: unknown, sha256?: unknown, bytes?: unknown}} */ (value);
  if (!hasExactBuildFields(record, ["bytes", "path", "sha256"]) ||
      typeof record.path !== "string" || !absolutePath(record.path) || record.path !== expected ||
      typeof record.sha256 !== "string" || !noticeHashPattern.test(record.sha256) || typeof record.bytes !== "number" ||
      !Number.isSafeInteger(record.bytes) || record.bytes < 1 || record.bytes > maximumNoticeBytes) {
    throw new Error("Native package receipt does not bind its notices");
  }
  const file = nativeNoticeFile(expected);
  if (file.sha256 !== record.sha256 || file.bytes !== record.bytes) {throw new Error("Native package receipt does not bind its notices");}
  return {path: expected, sha256: file.sha256, bytes: file.bytes};
}

/** @typedef {{receiptPath: string, receiptSha256: string, fingerprint: string,
 * notice: {path: string, sha256: string, bytes: number}}} NoticeAssemblyMetadata */
/** Validate only the portable metadata carried forward from an admitted assembly receipt.
 * The originating receipt and license inputs may be absent on a clean qualification host.
 * @param {unknown} value @param {unknown} notices @returns {NoticeAssemblyMetadata | undefined} */
function validateNoticeAssemblyMetadata(value, notices) {
  if (value === undefined) {return undefined;}
  const {receiptPath, receiptSha256, fingerprint, notice} =
    requireBuildFields(value, ["fingerprint", "notice", "receiptPath", "receiptSha256"], "Native notice assembly binding");
  const absolute = (/** @type {string} */ file) => file.length <= 4096 &&
    (path.posix.isAbsolute(file) || path.win32.isAbsolute(file)) && !/[\0\r\n]/.test(file);
  if (typeof receiptPath !== "string" || !absolute(receiptPath) || !/(?:^|[\\/])native-notices\.json$/i.test(receiptPath) ||
      typeof fingerprint !== "string" || !noticeHashPattern.test(fingerprint) || typeof receiptSha256 !== "string" || !noticeHashPattern.test(receiptSha256)) {
    throw new Error("Native notice assembly metadata is malformed");
  }
  const {path: noticePath, sha256, bytes} = requireBuildFields(notice, ["bytes", "path", "sha256"], "Native notice assembly body");
  if (typeof noticePath !== "string" || !absolute(noticePath) || !/(?:^|[\\/])THIRD-PARTY-NOTICES\.txt$/i.test(noticePath) ||
      typeof sha256 !== "string" || !noticeHashPattern.test(sha256) || typeof bytes !== "number" ||
      !Number.isSafeInteger(bytes) || bytes < 1 || bytes > maximumNoticeBytes) {
    throw new Error("Native notice assembly body metadata is malformed");
  }
  const windowsPath = path.win32.isAbsolute(receiptPath) &&
    (receiptPath.includes("\\") || /^[A-Za-z]:/.test(receiptPath) || receiptPath.startsWith("//")), pathApi = windowsPath ? path.win32 : path.posix,
    expectedBodyPath = pathApi.join(pathApi.dirname(receiptPath), nativeNoticeName);
  if (noticePath !== expectedBodyPath) {throw new Error("Native notice assembly body path differs from its receipt sibling");}
  const candidate = /** @type {{path: unknown, sha256: unknown, bytes: unknown} | undefined} */ (notices);
  if (!candidate || typeof candidate !== "object" || candidate.sha256 !== sha256 || candidate.bytes !== bytes) {
    throw new Error("Native notice assembly metadata does not bind the candidate notice bytes");
  }
  return {receiptPath, receiptSha256, fingerprint, notice: {path: noticePath, sha256, bytes}};
}

/** Packaging can also verify the assembler receipt and its local source inputs.
 * @param {unknown} value @param {{path: string, sha256: string, bytes: number} | null} expected */
function nativeNoticeAssembly(value, expected) {
  const metadata = validateNoticeAssemblyMetadata(value, expected);
  if (!metadata) {return undefined;}
  if (!absolutePath(metadata.receiptPath) || !absolutePath(metadata.notice.path)) {
    throw new Error("Native package receipt has an invalid notice assembly path");
  }
  const {receiptPath, receiptSha256, fingerprint, notice} =
    require("./prepare-native-notices").readNativeNoticeReceipt(metadata.receiptPath, metadata.receiptSha256);
  if (!isDeepStrictEqual(metadata, {receiptPath, receiptSha256, fingerprint, notice})) {
    throw new Error("Native package notice assembly receipt is stale or differs from its body binding");
  }
  return metadata;
}

/** The late-notice package is separate output; never place it inside or over the candidate tree.
 * @param {string} output @param {string} entrypoint */
function lateNoticeOutput(output, entrypoint) {
  if (!absolutePath(output) || path.resolve(output) !== output || fs.existsSync(output)) {
    throw new Error("Late notice packaging requires a fresh absolute output directory");
  }
  const parent = path.dirname(output), inputDirectory = path.dirname(entrypoint), relative = path.relative(inputDirectory, output), realParent = fs.realpathSync(parent),
    sameParent = process.platform === "win32" ? realParent.toLowerCase() === parent.toLowerCase() : realParent === parent;
  if (!fs.statSync(parent, {throwIfNoEntry: false})?.isDirectory() || !sameParent || !relative ||
      (!relative.startsWith(".." + path.sep) && !path.isAbsolute(relative))) {
    throw new Error("Late notice output must be a fresh sibling or external to the admitted candidate directory");
  }
  return output;
}

/** Use CMake's existing cross-platform archive support, not separate Linux/Windows
 * packaging recipes. Verify the extracted bytes before atomically installing a new archive.
 * Failed stages remain for diagnosis; success removes only this invocation's private stage.
 * @param {string} entrypoint @param {string} cmake @param {string} version
 * @param {typeof spawnSync} [spawn] @param {{output: string, notices: string, receipt?: {path: string, sha256: string}}} [lateNotice] */
function packageNativeRelease(entrypoint, cmake, version, spawn = spawnSync, lateNotice) {
  if (![entrypoint, cmake].every(absolutePath) || !/^[0-9][0-9A-Za-z.-]*$/.test(version)) {throw new Error("Native packaging requires absolute receipt/tool paths and a release version");}
  const {value: record, sha256: entryHash} = readBuildJsonSnapshot(entrypoint), artifact = record?.artifact;
  const windows = record?.target === "win32", name = windows ? "mom.exe" : "mom", directory = path.dirname(entrypoint);
  if (lateNotice) {
    const keys = ["notices", "output"], receipt = lateNotice.receipt;
    if ((!hasExactBuildFields(lateNotice, keys) && !hasExactBuildFields(lateNotice, [...keys, "receipt"])) || !absolutePath(lateNotice.notices) ||
        receipt !== undefined && (!receipt || typeof receipt !== "object" || Array.isArray(receipt) ||
          !hasExactBuildFields(receipt, ["path", "sha256"]) || !absolutePath(receipt.path) ||
          typeof receipt.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(receipt.sha256))) {
      throw new Error("Late notice packaging requires an output directory, absolute notice input, and optional pinned assembler receipt");
    }
  }
  if (record?.version !== 1 || !["linux", "win32"].includes(record.target) || record.buildOnly !== true ||
      !artifact || artifact.path !== path.join(directory, name) || fs.realpathSync(artifact.path) !== artifact.path ||
      artifact.sha256 !== fileHash(artifact.path) || artifact.bytes !== fs.statSync(artifact.path).size || fileHash(entrypoint) !== entryHash) {
    throw new Error("Native package receipt does not bind its executable");
  }
  const outputDirectory = lateNotice ? lateNoticeOutput(lateNotice.output, entrypoint) : directory,
    originNotices = record.notices === undefined ? null : nativeNotice(record.notices, directory),
    originAssembly = nativeNoticeAssembly(record.noticeAssembly, originNotices),
    noticeInput = lateNotice ? nativeNoticeFile(lateNotice.notices) : undefined,
    artifactMode = fs.statSync(artifact.path).mode & 0o777;
  let lateAssembly;
  if (lateNotice?.receipt && noticeInput) {
    const admitted = require("./prepare-native-notices").readNativeNoticeReceipt(lateNotice.receipt.path, lateNotice.receipt.sha256);
    if (admitted.notice.path !== lateNotice.notices || admitted.notice.sha256 !== noticeInput.sha256 || admitted.notice.bytes !== noticeInput.bytes) {
      throw new Error("Late notice assembler receipt does not bind the exact selected notice path, bytes and hash");
    }
    lateAssembly = nativeNoticeAssembly({receiptPath: admitted.receiptPath, receiptSha256: admitted.receiptSha256,
      fingerprint: admitted.fingerprint, notice: admitted.notice}, {path: lateNotice.notices, ...noticeInput});
  }
  const header = Buffer.alloc(4), fd = fs.openSync(artifact.path, "r");
  try {
    if (fs.readSync(fd, header, 0, 4, 0) !== 4 || (windows ? header.readUInt16LE() !== 0x5a4d : header.toString("hex") !== "7f454c46")) {throw new Error("Native package executable format differs from its target");}
  } finally {fs.closeSync(fd);}
  const archive = path.join(outputDirectory, `mom-v${version}-${windows ? "win.zip" : "lin.tgz"}`), toolHash = fileHash(cmake);
  const receipt = path.join(outputDirectory, "native-package.json");
  if ([archive, receipt].some(file => fs.lstatSync(file, {throwIfNoEntry: false}))) {throw new Error("Native release archive or receipt already exists; refusing to overwrite it");}
  if (lateNotice) {fs.mkdirSync(outputDirectory, {mode: 0o700});}
  const outputNoticePath = lateNotice ? path.join(outputDirectory, nativeNoticeName) : undefined;
  if (lateNotice && outputNoticePath && noticeInput) {fs.copyFileSync(lateNotice.notices, outputNoticePath, fs.constants.COPYFILE_EXCL);}
  const selectedNotice = lateNotice && outputNoticePath && noticeInput ?
    nativeNotice({path: outputNoticePath, sha256: noticeInput.sha256, bytes: noticeInput.bytes}, outputDirectory) :
    originNotices || nativeNotice(record.notices, directory);
  if (!selectedNotice) {throw new Error("Native package receipt does not bind its notices");}
  const selectedAssembly = lateAssembly || (originAssembly && originAssembly.notice.sha256 === selectedNotice.sha256 &&
    originAssembly.notice.bytes === selectedNotice.bytes ? originAssembly : undefined);
  if (lateNotice && outputNoticePath && process.platform !== "win32") {fs.chmodSync(outputNoticePath, 0o644);}
  const stage = fs.mkdtempSync(path.join(outputDirectory, ".package-native-")), staged = path.join(stage, path.basename(archive));
  const inputDirectory = lateNotice ? path.join(stage, "input") : directory, extractDirectory = lateNotice ? path.join(stage, "extract") : stage;
  if (lateNotice && outputNoticePath && noticeInput) {
    fs.mkdirSync(inputDirectory); fs.mkdirSync(extractDirectory);
    fs.copyFileSync(artifact.path, path.join(inputDirectory, name), fs.constants.COPYFILE_EXCL);
    fs.copyFileSync(outputNoticePath, path.join(inputDirectory, nativeNoticeName), fs.constants.COPYFILE_EXCL);
    if (process.platform !== "win32") {
      fs.chmodSync(path.join(inputDirectory, name), artifactMode);
      fs.chmodSync(path.join(inputDirectory, nativeNoticeName), 0o644);
    }
    if (fileHash(path.join(inputDirectory, name)) !== artifact.sha256 || fileHash(path.join(inputDirectory, nativeNoticeName)) !== selectedNotice.sha256) {
      throw new Error("Late notice package staging changed executable or notice bytes; inspect " + stage);
    }
  }
  /** @param {string[]} args @param {string} cwd @param {string} label */
  const run = (args, cwd, label) => {
    const result = spawn(cmake, ["-E", "tar", ...args], {cwd, encoding: "utf8", timeout: 300000, maxBuffer: 64 << 10,
      env: {...process.env, TZ: "UTC", SOURCE_DATE_EPOCH: "315532800"}});
    fs.writeFileSync(path.join(stage, label + ".log"), (result.stdout || "") + (result.stderr || ""), {flag: "wx"});
    if (result.status !== 0 || result.signal || result.error) {throw new Error("Native archive stage failed: " + label + "; inspect " + stage, {cause: result.error});}
    return result.stdout;
  };
  // ZIP needs SOURCE_DATE_EPOCH above: CMake's explicit --mtime overrides only mtime,
  // leaving atime/ctime nondeterministic. Keep TGZ's existing metadata/bytes unchanged.
  run([windows ? "cf" : "czf", staged, ...(windows ? [] : ["--mtime=1980-01-01 00:00:00 UTC"]), "--format=" + (windows ? "zip" : "gnutar"), "--", name, nativeNoticeName], inputDirectory, "create");
  const members = run(["tf", staged], stage, "list").replace(/\r/g, "").split("\n").filter(Boolean);
  if (members.length !== 2 || new Set(members).size !== 2 || !members.includes(name) || !members.includes(nativeNoticeName)) {
    throw new Error("Native release archive must contain only " + name + " and " + nativeNoticeName + "; inspect " + stage);
  }
  run(["xf", staged, "--", name, nativeNoticeName], extractDirectory, "extract");
  const restored = path.join(extractDirectory, name);
  if (!fs.lstatSync(restored).isFile() || fileHash(restored) !== artifact.sha256 ||
      !windows && !(fs.statSync(restored).mode & 0o111)) {throw new Error("Native archive changed executable bytes or permissions; inspect " + stage);}
  const restoredNotice = nativeNoticeFile(path.join(extractDirectory, nativeNoticeName)), currentNotice = lateNotice && outputNoticePath && noticeInput ?
    nativeNotice({path: outputNoticePath, sha256: noticeInput.sha256, bytes: noticeInput.bytes}, outputDirectory) : nativeNotice(record.notices, directory);
  if (restoredNotice.sha256 !== selectedNotice.sha256 || restoredNotice.bytes !== selectedNotice.bytes ||
      currentNotice.sha256 !== selectedNotice.sha256 || currentNotice.bytes !== selectedNotice.bytes) {
    throw new Error("Native archive changed notice bytes or permissions; inspect " + stage);
  }
  if (fileHash(artifact.path) !== artifact.sha256 || fileHash(entrypoint) !== entryHash || fileHash(cmake) !== toolHash ||
      lateNotice && (nativeNoticeFile(lateNotice.notices).sha256 !== noticeInput?.sha256 || fileHash(outputNoticePath || "") !== selectedNotice.sha256) ||
      lateNotice && process.platform !== "win32" && (fs.statSync(artifact.path).mode & 0o777) !== artifactMode ||
      lateNotice && originNotices && nativeNotice(record.notices, directory).sha256 !== originNotices.sha256 ||
      originAssembly && !nativeNoticeAssembly(record.noticeAssembly, originNotices) ||
      lateAssembly && (!lateNotice || !noticeInput || !nativeNoticeAssembly(lateAssembly, {path: lateNotice.notices, ...noticeInput}))) {
    throw new Error("Native packaging inputs changed; inspect " + stage);
  }
  const result = {version: lateNotice ? 2 : 1, entrypoint, entryHash, ...(lateNotice ? {originNotices} : {}), executable: artifact, tool: {path: cmake, sha256: toolHash},
    archive: {path: archive, bytes: fs.statSync(staged).size, sha256: fileHash(staged)},
    notices: selectedNotice, ...(selectedAssembly ? {noticeAssembly: selectedAssembly} : {}),
    members: [name, nativeNoticeName], exactOriginalBytes: true, buildOnly: true, publicationQualified: false};
  // Hard-link installation is same-filesystem and fails atomically if another writer won.
  fs.linkSync(staged, archive);
  fs.writeFileSync(receipt, JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  fs.rmSync(stage, {recursive: true});
  return result;
}

module.exports = {nativeNoticeName, maximumNoticeBytes, nativeNoticeFile, validateNoticeAssemblyMetadata, packageNativeRelease};
if (require.main === module) {
  try {
    const [entrypoint, cmake, version, ...args] = process.argv.slice(2);
    let lateNotice;
    if (args.length) {
      if (![4, 7].includes(args.length) || args[0] !== "--output" || args[2] !== "--notices" ||
          args.length === 7 && args[4] !== "--receipt") {
        throw new Error("Late notice mode requires --output /new/output --notices /THIRD-PARTY-NOTICES.txt [--receipt /native-notices.json <sha256>]");
      }
      lateNotice = {output: args[1] || "", notices: args[3] || "",
        ...(args.length === 7 ? {receipt: {path: args[5] || "", sha256: args[6] || ""}} : {})};
    }
    if (!entrypoint || !cmake || !version) {throw new Error("Usage: package-native-release.js /candidate/entrypoint.json /absolute/cmake <version> [--output /new/output --notices /THIRD-PARTY-NOTICES.txt]");}
    const result = packageNativeRelease(entrypoint, cmake, version, spawnSync, lateNotice);
    console.log("[native] Packaged " + result.archive.path + "; executable and data-only notices verified, NOT publication-qualified.");
  } catch (error) {console.error(error instanceof Error ? error.message : error); process.exitCode = 1;}
}
