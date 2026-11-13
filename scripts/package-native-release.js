"use strict";

// Build-only archive production. Only the native image and its data-only notice file ship;
// the notice does not itself fulfill license or source obligations. SDK receipts, Node,
// addons, shared libraries and compiler tools must never enter the package.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const {fileHash, readBuildJson, readBuildText} = require("./build-source-inputs");
const {absolutePath} = require("./build-native-release");

const noticeName = "THIRD-PARTY-NOTICES.txt", maximumNoticeBytes = 8 << 20;

/** Read only a small, regular non-symlink UTF-8 data file and detect replacement or mutation.
 * @param {string} file @returns {{bytes: number, sha256: string}} */
function nativeNoticeFile(file) {
  if (path.basename(file) !== noticeName) {throw new Error("Native package notice must be named " + noticeName);}
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
  const expected = path.join(directory, noticeName), record = /** @type {{path?: unknown, sha256?: unknown, bytes?: unknown}} */ (value);
  if (!record || typeof record !== "object" || Array.isArray(record) || Object.keys(record).sort().join(",") !== "bytes,path,sha256" ||
      typeof record.path !== "string" || !absolutePath(record.path) || record.path !== expected ||
      typeof record.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(record.sha256) || typeof record.bytes !== "number" ||
      !Number.isSafeInteger(record.bytes) || record.bytes < 1 || record.bytes > maximumNoticeBytes) {
    throw new Error("Native package receipt does not bind its notices");
  }
  const file = nativeNoticeFile(expected);
  if (file.sha256 !== record.sha256 || file.bytes !== record.bytes) {throw new Error("Native package receipt does not bind its notices");}
  return {path: expected, sha256: file.sha256, bytes: file.bytes};
}

/** Re-admit the assembly receipt and bind its source body bytes to the candidate's staged notice.
 * The receipt's original path is intentionally retained as metadata; staging changes the path.
 * @param {unknown} value @param {{path: string, sha256: string, bytes: number} | null} expected */
function nativeNoticeAssembly(value, expected) {
  if (value === undefined) {return undefined;}
  const assembly = /** @type {{receiptPath?: unknown, receiptSha256?: unknown, fingerprint?: unknown, notice?: unknown} | null} */ (value);
  if (!assembly || typeof assembly !== "object" || Array.isArray(assembly) ||
      Object.keys(assembly).sort().join(",") !== "fingerprint,notice,receiptPath,receiptSha256" ||
      typeof assembly.receiptPath !== "string" || !absolutePath(assembly.receiptPath) ||
      typeof assembly.receiptSha256 !== "string" || !/^[0-9a-f]{64}$/.test(assembly.receiptSha256) ||
      typeof assembly.fingerprint !== "string" || !/^[0-9a-f]{64}$/.test(assembly.fingerprint) || !expected) {
    throw new Error("Native package receipt has an invalid notice assembly binding");
  }
  const notice = /** @type {{path?: unknown, sha256?: unknown, bytes?: unknown} | null} */ (assembly.notice);
  if (!notice || typeof notice !== "object" || Array.isArray(notice) || Object.keys(notice).sort().join(",") !== "bytes,path,sha256" ||
      typeof notice.path !== "string" || !absolutePath(notice.path) || typeof notice.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(notice.sha256) ||
      typeof notice.bytes !== "number" || !Number.isSafeInteger(notice.bytes) || notice.bytes < 1 || notice.bytes > maximumNoticeBytes ||
      notice.sha256 !== expected.sha256 || notice.bytes !== expected.bytes) {
    throw new Error("Native package notice assembly does not bind the candidate notice bytes");
  }
  const admitted = require("./prepare-native-notices").readNativeNoticeReceipt(assembly.receiptPath, assembly.receiptSha256);
  if (admitted.receiptPath !== assembly.receiptPath || admitted.receiptSha256 !== assembly.receiptSha256 ||
      admitted.fingerprint !== assembly.fingerprint || admitted.notice.path !== notice.path ||
      admitted.notice.sha256 !== notice.sha256 || admitted.notice.bytes !== notice.bytes) {
    throw new Error("Native package notice assembly receipt is stale or differs from its body binding");
  }
  return {receiptPath: admitted.receiptPath, receiptSha256: admitted.receiptSha256, fingerprint: admitted.fingerprint,
    notice: {path: admitted.notice.path, sha256: admitted.notice.sha256, bytes: admitted.notice.bytes}};
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
  const entryHash = fileHash(entrypoint), record = readBuildJson(entrypoint), artifact = record?.artifact;
  const windows = record?.target === "win32", name = windows ? "mom.exe" : "mom", directory = path.dirname(entrypoint);
  if (lateNotice) {
    const keys = Object.keys(lateNotice).sort().join(","), receipt = lateNotice.receipt;
    if (!["notices,output", "notices,output,receipt"].includes(keys) || !absolutePath(lateNotice.notices) ||
        receipt !== undefined && (!receipt || typeof receipt !== "object" || Array.isArray(receipt) ||
          Object.keys(receipt).sort().join(",") !== "path,sha256" || !absolutePath(receipt.path) ||
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
  const outputNoticePath = lateNotice ? path.join(outputDirectory, noticeName) : undefined;
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
    fs.copyFileSync(outputNoticePath, path.join(inputDirectory, noticeName), fs.constants.COPYFILE_EXCL);
    if (process.platform !== "win32") {
      fs.chmodSync(path.join(inputDirectory, name), artifactMode);
      fs.chmodSync(path.join(inputDirectory, noticeName), 0o644);
    }
    if (fileHash(path.join(inputDirectory, name)) !== artifact.sha256 || fileHash(path.join(inputDirectory, noticeName)) !== selectedNotice.sha256) {
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
  run([windows ? "cf" : "czf", staged, ...(windows ? [] : ["--mtime=1980-01-01 00:00:00 UTC"]), "--format=" + (windows ? "zip" : "gnutar"), "--", name, noticeName], inputDirectory, "create");
  const members = run(["tf", staged], stage, "list").replace(/\r/g, "").split("\n").filter(Boolean);
  if (members.length !== 2 || new Set(members).size !== 2 || !members.includes(name) || !members.includes(noticeName)) {
    throw new Error("Native release archive must contain only " + name + " and " + noticeName + "; inspect " + stage);
  }
  run(["xf", staged, "--", name, noticeName], extractDirectory, "extract");
  const restored = path.join(extractDirectory, name);
  if (!fs.lstatSync(restored).isFile() || fileHash(restored) !== artifact.sha256 ||
      !windows && !(fs.statSync(restored).mode & 0o111)) {throw new Error("Native archive changed executable bytes or permissions; inspect " + stage);}
  const restoredNotice = nativeNoticeFile(path.join(extractDirectory, noticeName)), currentNotice = lateNotice && outputNoticePath && noticeInput ?
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
    members: [name, noticeName], exactOriginalBytes: true, buildOnly: true, publicationQualified: false};
  // Hard-link installation is same-filesystem and fails atomically if another writer won.
  fs.linkSync(staged, archive);
  fs.writeFileSync(receipt, JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  fs.rmSync(stage, {recursive: true});
  return result;
}

module.exports = {nativeNoticeFile, packageNativeRelease};
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
