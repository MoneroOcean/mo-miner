"use strict";

const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"),
  {createHash} = require("node:crypto"), {TextDecoder} = require("node:util");

const {hasExactBuildFields} = require("./build-source-inputs");

const noticeName = "THIRD-PARTY-NOTICES.txt", receiptName = "native-notices.json", maximumBytes = 8 << 20, maximumInputs = 128;
const projectionName = "native-notice-projection.json", maximumArchiveBytes = 1 << 30, maximumEvidenceBytes = 1 << 30,
  maximumProjectionInputsBytes = 2 * (1 << 30), maximumTarListingBytes = 1 << 20, maximumProjectionRuntimeMs = 120000;
const separator = "================================================================================\n";

/** @typedef {{component: string, path: string, sha256: string}} NoticeInput */

/** @param {string} value @param {string} label */
function canonicalPath(value, label) {
  if (typeof value !== "string" || value.length > 4096 || !path.isAbsolute(value) || path.resolve(value) !== value || /[\0\r\n]/.test(value)) {
    throw new Error(label + " must be a canonical absolute path");
  }
  const real = fs.realpathSync(value), same = process.platform === "win32" ? real.toLowerCase() === value.toLowerCase() : real === value;
  if (!same) {throw new Error(label + " must not traverse a symlink");}
  return value;
}

/** @param {string} file @param {number} maximum @param {string} label */
function readRegularText(file, maximum, label) {
  canonicalPath(file, label);
  const beforePath = fs.lstatSync(file, {bigint: true});
  if (!beforePath.isFile() || beforePath.nlink !== 1n || beforePath.size < 1n || beforePath.size > BigInt(maximum)) {
    throw new Error(label + " must be a bounded, singly-linked regular file");
  }
  const flags = fs.constants.O_RDONLY | (fs.constants.O_NONBLOCK || 0) | (fs.constants.O_NOFOLLOW || 0), fd = fs.openSync(file, flags);
  try {
    const beforeFd = fs.fstatSync(fd, {bigint: true});
    if (!sameFile(beforePath, beforeFd)) {throw new Error(label + " changed while opening");}
    const data = Buffer.alloc(Number(beforeFd.size) + 1);
    let bytes = 0;
    for (; bytes < data.length;) {
      const count = fs.readSync(fd, data, bytes, data.length - bytes, null);
      if (!count) {break;}
      bytes += count;
    }
    const afterFd = fs.fstatSync(fd, {bigint: true}), afterPath = fs.lstatSync(file, {bigint: true});
    canonicalPath(file, label);
    if (bytes !== Number(beforeFd.size) || !sameFile(beforeFd, afterFd) || !sameFile(beforeFd, afterPath)) {
      throw new Error(label + " changed while reading");
    }
    const body = data.subarray(0, bytes);
    if (body.includes(0)) {throw new Error(label + " must not contain NUL bytes");}
    let text;
    try {text = new TextDecoder("utf-8", {fatal: true, ignoreBOM: true}).decode(body);} catch {throw new Error(label + " must be valid UTF-8");}
    if (!Buffer.from(text, "utf8").equals(body)) {throw new Error(label + " must round-trip as UTF-8");}
    return {body, stat: beforePath, sha256: createHash("sha256").update(body).digest("hex")};
  } finally {fs.closeSync(fd);}
}

/** Compare identity and mutation-sensitive metadata, excluding atime which reads may update.
 * @param {import("node:fs").BigIntStats} a @param {import("node:fs").BigIntStats} b */
function sameFile(a, b) {
  return a.isFile() && b.isFile() && a.dev === b.dev && a.ino === b.ino && a.nlink === b.nlink && a.nlink === 1n && a.size === b.size &&
    a.mode === b.mode && a.mtimeNs === b.mtimeNs && a.ctimeNs === b.ctimeNs;
}

/** Hash a bounded, canonical single-link regular file without buffering it. @param {string} file @param {number} maximum
 * @param {string} label @param {number} deadline @returns {{path:string,bytes:number,sha256:string,stat:import('node:fs').BigIntStats}} */
function digestRegularInput(file, maximum, label, deadline) {
  const active = () => {if (Date.now() >= deadline) {throw new Error("Native notice projection exceeded its aggregate deadline");}};
  active(); canonicalPath(file, label);
  const beforePath = fs.lstatSync(file, {bigint: true});
  if (!beforePath.isFile() || beforePath.nlink !== 1n || beforePath.size < 1n || beforePath.size > BigInt(maximum)) {
    throw new Error(label + " must be a bounded, singly-linked regular file");
  }
  const flags = fs.constants.O_RDONLY | (fs.constants.O_NONBLOCK || 0) | (fs.constants.O_NOFOLLOW || 0), fd = fs.openSync(file, flags);
  try {
    const beforeFd = fs.fstatSync(fd, {bigint: true});
    if (!sameFile(beforePath, beforeFd)) {throw new Error(label + " changed while opening");}
    const buffer = Buffer.allocUnsafe(64 << 10), hash = createHash("sha256"); let bytes = 0n;
    while (bytes <= beforeFd.size) {
      active();
      const remaining = beforeFd.size - bytes + 1n, count = fs.readSync(fd, buffer, 0,
        Number(remaining < BigInt(buffer.length) ? remaining : BigInt(buffer.length)), null);
      if (!count) {break;}
      bytes += BigInt(count); hash.update(buffer.subarray(0, count));
    }
    const afterFd = fs.fstatSync(fd, {bigint: true}), afterPath = fs.lstatSync(file, {bigint: true});
    canonicalPath(file, label); active();
    if (bytes !== beforeFd.size || !sameFile(beforeFd, afterFd) || !sameFile(beforeFd, afterPath)) {throw new Error(label + " changed while hashing");}
    return {path: file, bytes: Number(bytes), sha256: hash.digest("hex"), stat: beforePath};
  } finally {fs.closeSync(fd);}
}

/** @param {number} deadline */
function remainingRuntime(deadline) {
  const remaining = deadline - Date.now();
  if (remaining <= 0) {throw new Error("Native notice projection exceeded its aggregate deadline");}
  return remaining;
}

/** @param {string[]} args @param {number} maximum @param {number} deadline @returns {Buffer} */
function runTar(args, maximum, deadline) {
  // Windows bsdtar records platform xattrs in PAX headers. They do not affect the
  // exact regular-member bytes projected here; other tar diagnostics remain fatal.
  const result = spawnSync("tar", ["--warning=no-unknown-keyword", ...args], {encoding: "buffer", timeout: remainingRuntime(deadline), maxBuffer: maximum,
    windowsHide: true, env: {PATH: process.env["PATH"] || "/usr/bin:/bin", LC_ALL: "C", LANG: "C", TAR_OPTIONS: ""}});
  if (result.error || result.signal || result.status !== 0 || result.stderr?.length) {
    throw new Error("Pinned native source archive member could not be read safely", {cause: result.error});
  }
  return Buffer.isBuffer(result.stdout) ? result.stdout : Buffer.from(result.stdout || "", "utf8");
}

/** @param {unknown} row @param {number} index @param {Set<string>} labels */
function validateProjectionRow(row, index, labels) {
  if (!hasExactBuildFields(row, ["admissionFiles", "archive", "component", "member"])) {
    throw new Error("Invalid native notice archive row " + index);
  }
  const item = /** @type {{component:string,archive:{path:string,sha256:string},member:string,admissionFiles:Record<string,string>}} */ (row);
  if (typeof item.component !== "string" || Buffer.byteLength(item.component, "utf8") > 128 ||
      !/^[A-Za-z0-9][A-Za-z0-9 ._+()/-]*$/.test(item.component) || labels.has(item.component.toLowerCase())) {
    throw new Error("Native notice archive component labels must be unique bounded single-line names");
  }
  labels.add(item.component.toLowerCase());
  if (!item.archive || typeof item.archive !== "object" || Array.isArray(item.archive) ||
      !hasExactBuildFields(item.archive, ["path", "sha256"]) || typeof item.archive.path !== "string" ||
      typeof item.archive.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(item.archive.sha256)) {
    throw new Error("Invalid pinned archive identity for native notice " + item.component);
  }
  if (typeof item.member !== "string" || item.member.length > 4096 || item.member.startsWith("/") || item.member.includes(":") ||
      ["\\", "*", "?", "[", "]"].some(character => item.member.includes(character)) || /[\0\r\n]/.test(item.member) ||
      item.member.split("/").some(part => !part || part === "." || part === "..")) {
    throw new Error("Native notice archive member must be a fixed relative path without traversal or wildcards");
  }
  if (!item.admissionFiles || typeof item.admissionFiles !== "object" || Array.isArray(item.admissionFiles) ||
      Object.keys(item.admissionFiles).length < 1 || Object.keys(item.admissionFiles).length > 128) {
    throw new Error("Native notice archive admission files must be a nonempty bounded hash map");
  }
  for (const [file, sha256] of Object.entries(item.admissionFiles)) {
    if (typeof file !== "string" || typeof sha256 !== "string" || !/^[a-f0-9]{64}$/.test(sha256)) {
      throw new Error("Invalid native notice admission-file hash for " + item.component);
    }
  }
  return item;
}

/** Read one exact regular tar member and reject duplicate names and non-file entry types. @param {string} archive
 * @param {string} member @param {number} deadline @returns {Buffer} */
function readArchiveMember(archive, member, deadline) {
  const args = ["--no-wildcards", "--anchored", "-tvf", archive, "--", member];
  const listing = runTar(args, maximumTarListingBytes, deadline);
  let lines;
  try {
    lines = new TextDecoder("utf-8", {fatal: true}).decode(listing).replace(/\n$/, "").split("\n");
  } catch {throw new Error("Native notice archive listing is not valid UTF-8");}
  if (lines.length !== 1 || !lines[0] || /[\r\0]/.test(lines[0])) {
    throw new Error("Native notice archive must contain exactly one matching member");
  }
  if (lines[0]?.[0] !== "-") {throw new Error("Native notice archive member must be a regular file");}
  const body = runTar(["--no-wildcards", "--anchored", "-xOf", archive, "--", member], maximumBytes + 1, deadline);
  if (body.length < 1 || body.length > maximumBytes || body.includes(0)) {throw new Error("Native notice archive body is empty, oversized, or contains NUL bytes");}
  try {
    const text = new TextDecoder("utf-8", {fatal: true, ignoreBOM: true}).decode(body);
    if (!Buffer.from(text, "utf8").equals(body)) {throw new Error("round-trip");}
  } catch {throw new Error("Native notice archive body must be valid UTF-8");}
  return body;
}

/** Linux CI/GNU-tar transport helper; ordinary file-based notice assembly remains portable.
 * Project only caller-selected members from already-admitted, hash-pinned source archives. This binds bytes to
 * supplied admissions but does not determine notice completeness or make legal claims.
 * @param {unknown} rows @param {string} outputDirectory
 * @returns {{inputs:NoticeInput[],provenance:{path:string,sha256:string,bytes:number}}} */
function projectNativeArchiveNotices(rows, outputDirectory) {
  if (process.platform !== "linux") {throw new Error("Native archive notice projection requires the Linux CI GNU tar toolchain");}
  freshOutput(outputDirectory);
  if (!Array.isArray(rows) || rows.length < 1 || rows.length > maximumInputs) {
    throw new Error("Native notice archive rows must be a nonempty bounded ordered array");
  }
  const deadline = Date.now() + maximumProjectionRuntimeMs, labels = new Set(), validated = rows.map((row, index) => validateProjectionRow(row, index, labels));
  /** @type {Map<string, ReturnType<typeof digestRegularInput>>} */ const snapshots = new Map();
  /** @type {Map<string, string>} */ const identities = new Map(), foldedPaths = new Map(); let inputBytes = 0;
  /** @param {string} file @param {string} expected @param {string} label @param {number} maximum
   * @returns {ReturnType<typeof digestRegularInput>} */
  const verifyFile = (file, expected, label, maximum) => {
    const canonical = canonicalPath(file, label), folded = process.platform === "win32" ? canonical.toLowerCase() : canonical,
      priorPath = foldedPaths.get(folded);
    if (priorPath && priorPath !== canonical) {throw new Error("Native notice projection contains a case-aliased input path");}
    foldedPaths.set(folded, canonical);
    const prior = snapshots.get(canonical);
    if (prior) {
      if (prior.sha256 !== expected) {throw new Error("Conflicting native notice input hashes name one file: " + canonical);}
      return prior;
    }
    const snapshot = digestRegularInput(canonical, maximum, label, deadline), identity = snapshot.stat.dev + ":" + snapshot.stat.ino,
      alias = identities.get(identity);
    if (alias && alias !== canonical) {throw new Error("Native notice projection contains aliased input files");}
    identities.set(identity, canonical);
    if (snapshot.sha256 !== expected) {throw new Error(label + " SHA-256 differs");}
    inputBytes += snapshot.bytes;
    if (inputBytes > maximumProjectionInputsBytes) {throw new Error("Native notice projection input bytes exceed the aggregate limit");}
    snapshots.set(canonical, snapshot);
    return snapshot;
  };
  const admitted = validated.map(row => {
    const archive = verifyFile(row.archive.path, row.archive.sha256, "Native notice source archive", maximumArchiveBytes),
      admissionFiles = Object.entries(row.admissionFiles).sort(([a], [b]) => a.localeCompare(b)).map(([file, sha256]) => {
        verifyFile(file, sha256, "Native notice admission file", maximumEvidenceBytes);
        return {path: file, sha256};
      });
    return {row, archive, admissionFiles};
  });
  let noticeBytes = 0;
  /** @type {{component:string,body:Buffer,sha256:string,bytes:number,archive:{path:string,sha256:string},member:string,admissionFiles:{path:string,sha256:string}[]}[]} */
  const projected = admitted.map(({row, archive, admissionFiles}) => {
    const body = readArchiveMember(archive.path, row.member, deadline);
    // Enforce the cumulative bound while reading, not after buffering every selected member.
    noticeBytes += body.length;
    if (noticeBytes > maximumBytes) {throw new Error("Projected native notice bodies exceed the total size limit");}
    const sha256 = createHash("sha256").update(body).digest("hex");
    return {component: row.component, body, sha256, bytes: body.length, archive: {path: archive.path, sha256: archive.sha256},
      member: row.member, admissionFiles};
  });
  fs.mkdirSync(outputDirectory, {mode: 0o700});
  canonicalPath(outputDirectory, "Created notice projection directory");
  const inputs = projected.map((row, index) => {
    const slug = row.component.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 64) || "component",
      file = path.join(outputDirectory, String(index).padStart(3, "0") + "-" + slug + ".txt");
    writeData(file, row.body);
    const staged = readRegularText(file, maximumBytes, "Projected notice " + row.component);
    if (staged.sha256 !== row.sha256 || staged.body.length !== row.bytes || !staged.body.equals(row.body)) {
      throw new Error("Projected notice bytes changed while staging: " + row.component);
    }
    return {component: row.component, path: file, sha256: row.sha256};
  });
  for (const snapshot of snapshots.values()) {
    const current = digestRegularInput(snapshot.path, snapshot.bytes, "Native notice projection input", deadline);
    if (!sameFile(snapshot.stat, current.stat) || current.sha256 !== snapshot.sha256) {
      throw new Error("Native notice projection input changed during projection: " + snapshot.path);
    }
  }
  const provenanceValue = {version: 1, buildOnly: true, publicationQualified: false,
      inputs: projected.map((row, index) => ({component: row.component, path: inputs[index]?.path, sha256: row.sha256, bytes: row.bytes,
        archive: row.archive, member: row.member, admissionFiles: row.admissionFiles}))},
    provenanceBody = Buffer.from(JSON.stringify(provenanceValue, null, 2) + "\n", "utf8"), provenancePath = path.join(outputDirectory, projectionName);
  if (provenanceBody.length > (1 << 20)) {throw new Error("Native notice projection provenance exceeds its size limit");}
  writeData(provenancePath, provenanceBody);
  for (const snapshot of snapshots.values()) {
    const current = digestRegularInput(snapshot.path, snapshot.bytes, "Native notice projection input", deadline);
    if (!sameFile(snapshot.stat, current.stat) || current.sha256 !== snapshot.sha256) {
      throw new Error("Native notice projection input changed after staging: " + snapshot.path);
    }
  }
  return {inputs, provenance: {path: provenancePath, sha256: createHash("sha256").update(provenanceBody).digest("hex"), bytes: provenanceBody.length}};
}

/** Validate and freeze all selected source bodies before creating any output.
 * @param {unknown} value @returns {{component: string, path: string, sha256: string, bytes: number, body: Buffer, stat: import("node:fs").BigIntStats}[]} */
function noticeInputs(value) {
  if (!Array.isArray(value) || value.length < 1 || value.length > maximumInputs) {throw new Error("Notice inputs must be a nonempty bounded ordered array");}
  const seen = new Set(); let total = 0;
  return value.map((row, index) => {
    if (!hasExactBuildFields(row, ["component", "path", "sha256"])) {
      throw new Error("Invalid notice input row " + index);
    }
    const input = /** @type {NoticeInput} */ (row);
    if (typeof input.component !== "string" || Buffer.byteLength(input.component, "utf8") > 128 ||
        !/^[A-Za-z0-9][A-Za-z0-9 ._+()/-]*$/.test(input.component) || seen.has(input.component.toLowerCase())) {
      throw new Error("Notice component labels must be unique bounded single-line names");
    }
    seen.add(input.component.toLowerCase());
    if (typeof input.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(input.sha256)) {throw new Error("Invalid notice input SHA-256: " + input.component);}
    const selectedPath = canonicalPath(input.path, "Notice input path"), selected = readRegularText(selectedPath, maximumBytes, "Notice input " + input.component);
    if (selected.sha256 !== input.sha256) {throw new Error("Notice input hash differs: " + input.component);}
    total += selected.body.length;
    if (total > maximumBytes) {throw new Error("Notice source bodies exceed the total size limit");}
    return {component: input.component, path: selectedPath, sha256: selected.sha256, bytes: selected.body.length, body: selected.body, stat: selected.stat};
  });
}

/** Build exact source bodies under deterministic, readable component headings. @param {ReturnType<typeof noticeInputs>} inputs */
function noticeBody(inputs) {
  const chunks = [];
  for (const input of inputs) {
    chunks.push(Buffer.from(separator + "Components: " + input.component + "\n\n", "utf8"), input.body);
    if (input.body[input.body.length - 1] !== 0x0a) {chunks.push(Buffer.from("\n"));}
    chunks.push(Buffer.from("\n"));
  }
  const body = Buffer.concat(chunks);
  if (body.length > maximumBytes) {throw new Error("Assembled notice exceeds the size limit");}
  return body;
}

/** @param {string} output */
function freshOutput(output) {
  if (typeof output !== "string" || output.length > 4096 || !path.isAbsolute(output) || path.resolve(output) !== output ||
      /[\0\r\n]/.test(output)) {
    throw new Error("Notice output must be a fresh absolute directory");
  }
  canonicalPath(path.dirname(output), "Notice output parent");
  if (fs.lstatSync(output, {throwIfNoEntry: false})) {
    throw new Error("Notice output must be a fresh absolute directory");
  }
}

/** @param {string} file @param {Buffer} body */
function writeData(file, body) {
  const fd = fs.openSync(file, "wx", 0o644);
  try {fs.writeFileSync(fd, body); if (process.platform !== "win32") {fs.fchmodSync(fd, 0o644);} fs.fsyncSync(fd);} finally {fs.closeSync(fd);}
  const stat = fs.lstatSync(file, {bigint: true});
  if (!stat.isFile() || stat.nlink !== 1n || stat.size !== BigInt(body.length) || process.platform !== "win32" && (stat.mode & 0o777n) !== 0o644n) {
    throw new Error("Notice output was not written as a data-only regular file");
  }
}

/** @param {unknown} options @returns {boolean} */
function bundleInputsOption(options) {
  if (options === undefined) {return false;}
  if (!options || typeof options !== "object" || Array.isArray(options)) {
    throw new Error("Native notice assembly options must contain only a boolean bundleInputs flag");
  }
  const keys = Object.keys(options);
  if (keys.length === 0) {return false;}
  const value = /** @type {Record<string, unknown>} */ (options);
  if (keys.length !== 1 || keys[0] !== "bundleInputs" || typeof value["bundleInputs"] !== "boolean") {
    throw new Error("Native notice assembly options must contain only a boolean bundleInputs flag");
  }
  return value["bundleInputs"];
}

/** Assemble reviewed license texts into a fresh data-only output directory.
 * With bundleInputs, also stage the exact literal source bytes under a flat `inputs/` directory.
 * @param {unknown} value @param {string} outputDirectory @param {unknown} [options] */
function assembleNativeNotices(value, outputDirectory, options) {
  const bundleInputs = bundleInputsOption(options);
  const inputs = noticeInputs(value), body = noticeBody(inputs);
  freshOutput(outputDirectory);
  const output = path.join(outputDirectory, noticeName), receiptPath = path.join(outputDirectory, receiptName);
  const fingerprint = createHash("sha256").update(JSON.stringify({version: 1,
    inputs: inputs.map(({component, sha256, bytes}) => ({component, sha256, bytes}))})).digest("hex");
  fs.mkdirSync(outputDirectory, {mode: 0o700});
  canonicalPath(outputDirectory, "Created notice output directory");
  /** @type {ReturnType<typeof noticeInputs>} */ let stagedInputs = [];
  if (bundleInputs) {
    const inputDirectory = path.join(outputDirectory, "inputs");
    fs.mkdirSync(inputDirectory, {mode: 0o700});
    canonicalPath(inputDirectory, "Created notice input directory");
    stagedInputs = inputs.map((input, index) => {
      const name = String(index).padStart(3, "0") + "-" + input.sha256 + ".txt", file = path.join(inputDirectory, name);
      writeData(file, input.body);
      const staged = readRegularText(file, maximumBytes, "Staged notice input " + input.component);
      if (staged.sha256 !== input.sha256 || staged.body.length !== input.bytes || !staged.body.equals(input.body)) {
        throw new Error("Staged notice input bytes differ: " + input.component);
      }
      return {...input, path: file, stat: staged.stat};
    });
  }
  const sourceRecords = (bundleInputs ? stagedInputs : inputs).map(({component, path: inputPath, sha256, bytes}) =>
    ({component, path: inputPath, sha256, bytes}));
  // Keep failed fresh outputs for diagnosis; a retry must use a new directory.
  writeData(output, body);
  for (const input of inputs) {
    const current = readRegularText(input.path, maximumBytes, "Notice input " + input.component);
    if (!sameFile(input.stat, current.stat) || current.sha256 !== input.sha256 || !current.body.equals(input.body)) {
      throw new Error("Notice input changed during assembly: " + input.component);
    }
  }
  for (const input of stagedInputs) {
    const current = readRegularText(input.path, maximumBytes, "Staged notice input " + input.component);
    if (!sameFile(input.stat, current.stat) || current.sha256 !== input.sha256 || !current.body.equals(input.body)) {
      throw new Error("Staged notice input changed during assembly: " + input.component);
    }
  }
  const notice = {path: output, sha256: createHash("sha256").update(body).digest("hex"), bytes: body.length};
  const receipt = {version: 1, fingerprint, inputs: sourceRecords, notice, buildOnly: true, publicationQualified: false};
  writeData(receiptPath, Buffer.from(JSON.stringify(receipt, null, 2) + "\n", "utf8"));
  if (createHash("sha256").update(fs.readFileSync(output)).digest("hex") !== notice.sha256) {throw new Error("Assembled notice changed after writing");}
  for (const input of stagedInputs) {
    const current = readRegularText(input.path, maximumBytes, "Staged notice input " + input.component);
    if (!sameFile(input.stat, current.stat) || current.sha256 !== input.sha256 || !current.body.equals(input.body)) {
      throw new Error("Staged notice input changed after assembly: " + input.component);
    }
  }
  return receipt;
}

/** Admit the existing assembler receipt and its source bodies using a caller-supplied
 * receipt pin. This verifies provenance bytes, not license completeness.
 * @param {string} receiptPath @param {string} expectedSha256 */
function readNativeNoticeReceipt(receiptPath, expectedSha256) {
  if (typeof expectedSha256 !== "string" || !/^[a-f0-9]{64}$/.test(expectedSha256) || path.basename(receiptPath) !== receiptName) {
    throw new Error("Native notice receipt needs its external SHA-256 pin and canonical filename");
  }
  const receiptFile = readRegularText(receiptPath, 1 << 20, "Native notice receipt");
  if (receiptFile.sha256 !== expectedSha256) {throw new Error("Native notice receipt external SHA-256 differs");}
  let value;
  try {value = JSON.parse(receiptFile.body.toString("utf8"));} catch {throw new Error("Native notice receipt is not valid JSON");}
  if (!value || typeof value !== "object" || Array.isArray(value) ||
      !hasExactBuildFields(value, ["buildOnly", "fingerprint", "inputs", "notice", "publicationQualified", "version"]) ||
      value.version !== 1 || value.buildOnly !== true || value.publicationQualified !== false ||
      typeof value.fingerprint !== "string" || !/^[a-f0-9]{64}$/.test(value.fingerprint) ||
      !Array.isArray(value.inputs) || value.inputs.length < 1 || value.inputs.length > maximumInputs) {
    throw new Error("Native notice receipt has an invalid schema or qualification status");
  }
  const rows = /** @type {{component: string, path: string, sha256: string, bytes: number}[]} */ (value.inputs);
  for (const row of rows) {
    if (!hasExactBuildFields(row, ["bytes", "component", "path", "sha256"]) ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > maximumBytes) {throw new Error("Native notice receipt has an invalid input row");}
  }
  const inputs = noticeInputs(rows.map(({component, path: inputPath, sha256}) => ({component, path: inputPath, sha256})));
  if (inputs.some((input, index) => input.bytes !== rows[index]?.bytes)) {throw new Error("Native notice receipt input size differs");}
  const fingerprint = createHash("sha256").update(JSON.stringify({version: 1,
    inputs: inputs.map(({component, sha256, bytes}) => ({component, sha256, bytes}))})).digest("hex");
  if (fingerprint !== value.fingerprint) {throw new Error("Native notice receipt fingerprint differs");}
  const notice = /** @type {{path: string, sha256: string, bytes: number} | null} */ (value.notice);
  const expectedNoticePath = path.join(path.dirname(receiptPath), noticeName);
  if (!notice || !hasExactBuildFields(notice, ["bytes", "path", "sha256"]) ||
      notice.path !== expectedNoticePath || !/^[a-f0-9]{64}$/.test(notice.sha256) || !Number.isSafeInteger(notice.bytes) ||
      notice.bytes < 1 || notice.bytes > maximumBytes) {throw new Error("Native notice receipt has an invalid notice binding");}
  const body = noticeBody(inputs), noticeFile = readRegularText(notice.path, maximumBytes, "Assembled native notice");
  if (notice.bytes !== body.length || notice.sha256 !== createHash("sha256").update(body).digest("hex") ||
      noticeFile.body.length !== notice.bytes || noticeFile.sha256 !== notice.sha256 || !noticeFile.body.equals(body)) {
    throw new Error("Assembled native notice bytes differ from its receipt inputs");
  }
  for (const input of inputs) {
    const current = readRegularText(input.path, maximumBytes, "Native notice input " + input.component);
    if (!sameFile(input.stat, current.stat) || current.sha256 !== input.sha256 || !current.body.equals(input.body)) {
      throw new Error("Native notice input changed during admission: " + input.component);
    }
  }
  const receiptAfter = readRegularText(receiptPath, 1 << 20, "Native notice receipt"), noticeAfter = readRegularText(notice.path, maximumBytes, "Assembled native notice");
  if (!sameFile(receiptFile.stat, receiptAfter.stat) || receiptAfter.sha256 !== expectedSha256 || !receiptAfter.body.equals(receiptFile.body) ||
      !sameFile(noticeFile.stat, noticeAfter.stat) || noticeAfter.sha256 !== notice.sha256 || !noticeAfter.body.equals(body)) {
    throw new Error("Native notice receipt or body changed during admission");
  }
  return {receiptPath, receiptSha256: expectedSha256, fingerprint,
    notice: {path: notice.path, sha256: notice.sha256, bytes: notice.bytes},
    inputs: rows.map(({component, path: inputPath, sha256, bytes}) => ({component, path: inputPath, sha256, bytes})),
    buildOnly: true, publicationQualified: false};
}

module.exports = {assembleNativeNotices, projectNativeArchiveNotices, readNativeNoticeReceipt};
if (require.main === module) {
  try {
    const [manifestPath, outputDirectory, ...extra] = process.argv.slice(2);
    if (!manifestPath || !outputDirectory || extra.length || !path.isAbsolute(manifestPath)) {
      throw new Error("Usage: prepare-native-notices.js /ordered-inputs.json /fresh-output-directory");
    }
    const manifest = JSON.parse(readRegularText(manifestPath, 1 << 20, "Notice input manifest").body.toString("utf8"));
    if (!Array.isArray(manifest)) {throw new Error("Notice manifest must be an ordered array of {component,path,sha256} rows");}
    const result = assembleNativeNotices(manifest, outputDirectory);
    process.stdout.write("Assembled " + result.inputs.length + " notice components at " + result.notice.path + " (build-only; not publication-qualified).\n");
  } catch (error) {console.error(error instanceof Error ? error.message : error); process.exitCode = 1;}
}
