"use strict";

// Select only the pinned Windows CUDA Runtime library and its package license. This is neither a
// complete CUDA SDK nor a redistribution qualification.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const {hasExactBuildFields, fileHash, readBuildJson, readBuildJsonSnapshot, sourceInputs} = require("./build-source-inputs");

const repo = path.resolve(__dirname, ".."), catalogFile = path.join(repo, "runtime/windows-cudart.json");
const catalog = /** @type {CudartCatalog} */ (require("../runtime/windows-cudart.json"));
/** @param {object} value */
function freezeTree(value) {for (const child of Object.values(value)) {if (child && typeof child === "object") {freezeTree(child);}} Object.freeze(value);}
freezeTree(catalog);
const outputFiles = Object.freeze({"runtime/cudart_static.lib": catalog.cudartStatic, "licenses/cuda-cudart/LICENSE": catalog.license});
const archiveMaximumBytes = 4 << 20, memberMaximumBytes = 4 << 20, listingMaximumBytes = 2 << 20, commandTimeoutMs = 30000;
const recipeNames = ["scripts/prepare-native-cudart-windows.js", "runtime/windows-cudart.json", "scripts/build-source-inputs.js"];
/** @typedef {{member:string,output:string,bytes:number,sha256:string}} CudartMember */
/** @typedef {{version:number,target:string,vendor:string,releaseLabel:string,vendorVersion:string,manifestUrl:string,
 * license:{name:string,manifestPath:string,member:string,output:string,bytes:number,sha256:string},
 * archive:{fileName:string,url:string,origin:string,bytes:number,sha256:string},
 * cudartStatic:{member:string,output:string,bytes:number,sha256:string}}} CudartCatalog */
/** @typedef {{kind:"file",bytes:number,sha256:string,source:{archiveSha256:string,member:string}}} CudartOutput */
/** @typedef {{version:1,target:"windows-cudart",status:"PASS",networked:false,buildOnly:true,compiled:false,
 * vendorPrebuilt:true,publicationQualified:false,redistributionQualified:false,producerSha256:string,
 * recipe:{sha256:string,files:Record<string,string>},tool:{path:string,sha256:string,version:string},
 * source:{archivePath:string,url:string,sha256:string,bytes:number,manifestUrl:string},output:string,
 * files:Record<string,CudartOutput>}} CudartReceipt */
/** @typedef {{receipt:CudartReceipt,receiptFile:string,receiptSha256:string,cudartStatic:string,license:string,files:Record<string,string>}} CudartInputs */

/** @param {unknown} value @returns {value is string} */
function absolute(value) {
  return typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) && path.resolve(value) === value && !/[\r\n\0]/.test(value);
}

/** Producer paths are provenance strings; a Windows consumer must not resolve Linux producer paths. @param {unknown} value */
function provenancePath(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 4096 &&
    !Array.from(value).some(character => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127);
}

/** @param {string} file @param {string} label @param {number} maximum @param {bigint} [maximumLinks] */
function regularFileSnapshot(file, label, maximum, maximumLinks = 1n) {
  if (!absolute(file)) {throw new Error("Windows CUDA Runtime " + label + " path must be canonical and absolute");}
  const stat = fs.lstatSync(file, {bigint: true});
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink < 1n || stat.nlink > maximumLinks || stat.size < 1n ||
      stat.size > BigInt(maximum) || fs.realpathSync(file) !== file) {
    throw new Error("Windows CUDA Runtime " + label + " must be a bounded canonical regular file with an allowed link count: " + file);
  }
  return stat;
}

/** The distro-owned unzip binary is installed as a zip/unzip hardlink pair. */
function trustedUnzipSnapshot() {
  return regularFileSnapshot("/usr/bin/unzip", "unzip tool", 128 << 20, 2n);
}

/** @param {import("node:fs").BigIntStats} a @param {import("node:fs").BigIntStats} b */
function sameSnapshot(a, b) {
  return a.dev === b.dev && a.ino === b.ino && a.nlink === b.nlink && a.size === b.size && a.mode === b.mode && a.mtimeNs === b.mtimeNs && a.ctimeNs === b.ctimeNs;
}

/** @param {string} file @param {string} label @param {number} maximum */
function verifyDigest(file, label, maximum) {
  const before = regularFileSnapshot(file, label, maximum), sha256 = fileHash(file), after = regularFileSnapshot(file, label, maximum);
  if (!sameSnapshot(before, after)) {throw new Error("Windows CUDA Runtime " + label + " changed while hashing: " + file);}
  return {before, bytes: Number(before.size), sha256};
}

/** @param {string} output */
function newDestination(output) {
  if (!absolute(output)) {throw new Error("Windows CUDA Runtime output needs a new canonical absolute directory");}
  const parent = fs.realpathSync(path.dirname(output)), destination = path.join(parent, path.basename(output));
  let exists = false;
  try {fs.lstatSync(destination); exists = true;} catch (error) {if (Reflect.get(Object(error), "code") !== "ENOENT") {throw error;}}
  if (destination !== output || exists || destination === repo || destination.startsWith(repo + path.sep)) {
    throw new Error("Windows CUDA Runtime output must be new, canonical, and outside the repository");
  }
  return destination;
}

/** Read the repo-owned lock afresh, validate all semantic pins, and avoid a checkout-EOL hash dependency. */
function windowsCudartCatalog() {
  const before = regularFileSnapshot(catalogFile, "catalog", 64 << 10), disk = readBuildJson(catalogFile), after = regularFileSnapshot(catalogFile, "catalog", 64 << 10),
    archiveRoot = "cuda_cudart-windows-x86_64-" + catalog.vendorVersion + "-archive";
  if (!sameSnapshot(before, after) || !isDeepStrictEqual(disk, catalog) || !hasExactBuildFields(catalog,
    ["version", "target", "vendor", "releaseLabel", "vendorVersion", "manifestUrl", "license", "archive", "cudartStatic"]) ||
      catalog.version !== 1 || catalog.target !== "windows-cudart" || catalog.vendor !== "NVIDIA" || catalog.releaseLabel !== "12.6.0" ||
      catalog.vendorVersion !== "12.6.37" || catalog.manifestUrl !== "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.0.json" ||
      !hasExactBuildFields(catalog.archive, ["fileName", "url", "origin", "bytes", "sha256"]) || catalog.archive.fileName !== archiveRoot + ".zip" ||
      catalog.archive.url !== "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/" + catalog.archive.fileName ||
      catalog.archive.origin !== "https://developer.download.nvidia.com" || !Number.isSafeInteger(catalog.archive.bytes) ||
      catalog.archive.bytes < 1 || catalog.archive.bytes > archiveMaximumBytes || !/^[a-f0-9]{64}$/.test(catalog.archive.sha256) ||
      !hasExactBuildFields(catalog.license, ["name", "manifestPath", "member", "output", "bytes", "sha256"]) || catalog.license.name !== "CUDA Toolkit" ||
      catalog.license.manifestPath !== "cuda_cudart/LICENSE.txt" || catalog.license.member !== archiveRoot + "/LICENSE" ||
      catalog.license.output !== "licenses/cuda-cudart/LICENSE" || !Number.isSafeInteger(catalog.license.bytes) ||
      catalog.license.bytes < 1 || catalog.license.bytes > memberMaximumBytes || !/^[a-f0-9]{64}$/.test(catalog.license.sha256) ||
      !hasExactBuildFields(catalog.cudartStatic, ["member", "output", "bytes", "sha256"]) ||
      catalog.cudartStatic.member !== archiveRoot + "/lib/x64/cudart_static.lib" || catalog.cudartStatic.output !== "runtime/cudart_static.lib" ||
      !Number.isSafeInteger(catalog.cudartStatic.bytes) || catalog.cudartStatic.bytes < 1 || catalog.cudartStatic.bytes > memberMaximumBytes ||
      !/^[a-f0-9]{64}$/.test(catalog.cudartStatic.sha256)) {
    throw new Error("Pinned Windows CUDA Runtime catalog is invalid");
  }
  return catalog;
}

/** Windows file modes do not expose POSIX directory write bits; content/path pins remain enforced there. @param {import("node:fs").BigIntStats} stat */
function writableMode(stat) {return process.platform !== "win32" && (stat.mode & 0o222n) !== 0n;}

/** @returns {ReturnType<typeof sourceInputs>} */
function cudartRecipe() {return sourceInputs(repo, [], recipeNames);}

/** @param {string} tool @param {string[]} args @param {string} label @param {number} maximum */
function capture(tool, args, label, maximum) {
  const env = /** @type {NodeJS.ProcessEnv} */ ({...process.env, LC_ALL: "C", LANG: "C", UNZIPOPT: "", UNZIP_OPTS: "", ZIPINFOOPT: ""}),
    result = spawnSync(tool, args, {encoding: "utf8", timeout: commandTimeoutMs, maxBuffer: maximum, env});
  if (result.error || result.signal || result.status !== 0 || typeof result.stdout !== "string" || Buffer.byteLength(result.stdout) > maximum) {
    throw new Error("Windows CUDA Runtime " + label + " failed" + (result.error ? ": " + result.error.message : ""));
  }
  return result.stdout;
}

/** This is a build-time system utility. Receipts record its provenance; consumers never execute it. */
function findUnzip() {
  const file = "/usr/bin/unzip", stat = trustedUnzipSnapshot();
  if (stat.size > 128n * 1024n * 1024n) {
    throw new Error("Windows CUDA Runtime preparation requires canonical /usr/bin/unzip");
  }
  fs.accessSync(file, fs.constants.X_OK); return file;
}

/** @param {string} tool */
function toolIdentity(tool) {
  const stat = trustedUnzipSnapshot();
  if (tool !== "/usr/bin/unzip" || (stat.mode & 0o111n) === 0n) {throw new Error("Windows CUDA Runtime extractor is not executable");}
  const version = capture(tool, ["-v"], "unzip version", 64 << 10).split(/\r?\n/, 1)[0]?.trim();
  if (!version || /[\r\n\0]/.test(version)) {throw new Error("Windows CUDA Runtime extractor has no version identity");}
  return {path: tool, sha256: fileHash(tool), version};
}

/** Only literal fixed members are passed to unzip -p; no archive path is ever materialized. @param {string} tool @param {string} archive @param {CudartMember[]} rows */
function stagePinnedZipMembers(tool, archive, rows) {
  if (!Array.isArray(rows) || rows.length !== 2 || rows.some(row => !row || typeof row.member !== "string" || typeof row.output !== "string" ||
      !/^[A-Za-z0-9][A-Za-z0-9._+-]*(?:\/[A-Za-z0-9][A-Za-z0-9._+-]*)+$/.test(row.member) ||
      !/^[A-Za-z0-9][A-Za-z0-9._+-]*(?:\/[A-Za-z0-9][A-Za-z0-9._+-]*)+$/.test(row.output) ||
      row.member.split("/").some(part => part === "." || part === "..") || row.output.split("/").some(part => part === "." || part === "..") ||
      !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > memberMaximumBytes || !/^[a-f0-9]{64}$/.test(row.sha256)) ||
      new Set(rows.map(row => row.member)).size !== rows.length || new Set(rows.map(row => row.output)).size !== rows.length) {
    throw new Error("Windows CUDA Runtime staging requires exactly two unique pinned members");
  }
  const names = capture(tool, ["-Z1", archive], "ZIP member listing", listingMaximumBytes).split(/\r?\n/).filter(Boolean);
  if (!names.length || names.length > 4096 || names.some(name => name.length > 1024 || /[\0\r\n]/.test(name))) {
    throw new Error("Windows CUDA Runtime ZIP listing is invalid or excessive");
  }
  /** @type {Record<string,{bytes:number,sha256:string}>} */ const staged = {};
  for (const row of rows) {
    if (names.filter(name => name === row.member).length !== 1) {throw new Error("Pinned Windows CUDA Runtime ZIP member is missing or duplicated: " + row.member);}
    const result = spawnSync(tool, ["-p", archive, row.member], {encoding: null, timeout: commandTimeoutMs, maxBuffer: row.bytes + 1,
      env: {...process.env, UNZIPOPT: "", UNZIP_OPTS: "", ZIPINFOOPT: ""}});
    if (result.error || result.signal || result.status !== 0 || !Buffer.isBuffer(result.stdout) || result.stdout.length !== row.bytes ||
        createHash("sha256").update(result.stdout).digest("hex") !== row.sha256) {
      throw new Error("Windows CUDA Runtime ZIP member differs from its bounded pin: " + row.member);
    }
    staged[row.output] = {bytes: result.stdout.length, sha256: row.sha256};
    Object.defineProperty(staged[row.output], "data", {value: result.stdout, enumerable: false});
  }
  return staged;
}

/** @param {string} archivePath @param {string} output @returns {CudartReceipt} */
function prepareWindowsCudart(archivePath, output) {
  const selected = windowsCudartCatalog();
  if (process.platform !== "linux" || process.env["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows CUDA Runtime preparation requires Linux and MOM_AOT_BUILD_ONLY=1");}
  if (!absolute(archivePath) || archivePath === repo || archivePath.startsWith(repo + path.sep)) {throw new Error("Windows CUDA Runtime archive must be canonical and outside the repository");}
  const destination = newDestination(output), archiveBefore = regularFileSnapshot(archivePath, "input archive", archiveMaximumBytes);
  if (Number(archiveBefore.size) !== selected.archive.bytes || fileHash(archivePath) !== selected.archive.sha256) {throw new Error("Windows CUDA Runtime archive differs from its official pin");}
  const tool = findUnzip(), toolBefore = trustedUnzipSnapshot(), toolInfo = toolIdentity(tool),
    recipe = cudartRecipe(), recipeSha256 = createHash("sha256").update(JSON.stringify(recipe)).digest("hex"), producerSha256 = fileHash(__filename),
    rows = /** @type {CudartMember[]} */ ([selected.cudartStatic, selected.license]);
  if (fileHash(tool) !== toolInfo.sha256) {throw new Error("Windows CUDA Runtime extractor changed while identifying it");}
  const staged = stagePinnedZipMembers(tool, archivePath, rows);
  fs.mkdirSync(destination, {mode: 0o700});
  for (const row of rows) {
    const target = path.join(destination, ...row.output.split("/")); fs.mkdirSync(path.dirname(target), {recursive: true, mode: 0o700});
    const data = Reflect.get(staged[row.output] || {}, "data");
    if (!Buffer.isBuffer(data) || fileHash(archivePath) !== selected.archive.sha256) {throw new Error("Windows CUDA Runtime source changed before promotion");}
    fs.writeFileSync(target, data, {flag: "wx", mode: 0o600}); fs.chmodSync(target, 0o444);
  }
  for (const relative of ["licenses", "licenses/cuda-cudart", "runtime"]) {fs.chmodSync(path.join(destination, ...relative.split("/")), 0o555);}
  const files = /** @type {Record<string,CudartOutput>} */ (Object.fromEntries(rows.map(row => [row.output,
    {kind: "file", bytes: row.bytes, sha256: row.sha256, source: {archiveSha256: selected.archive.sha256, member: row.member}}])));
  const receipt = /** @type {CudartReceipt} */ ({version: 1, target: "windows-cudart", status: "PASS", networked: false, buildOnly: true,
    compiled: false, vendorPrebuilt: true, publicationQualified: false, redistributionQualified: false, producerSha256,
    recipe: {sha256: recipeSha256, files: recipe.graph}, tool: toolInfo,
    source: {archivePath, url: selected.archive.url, sha256: selected.archive.sha256, bytes: selected.archive.bytes, manifestUrl: selected.manifestUrl},
    output: destination, files});
  const receiptFile = path.join(destination, "sdk.json");
  fs.writeFileSync(receiptFile, JSON.stringify(receipt, null, 2) + "\n", {flag: "wx", mode: 0o600}); fs.chmodSync(receiptFile, 0o444);
  fs.chmodSync(destination, 0o555);
  const archiveAfter = regularFileSnapshot(archivePath, "input archive", archiveMaximumBytes), toolAfter = trustedUnzipSnapshot();
  if (!sameSnapshot(archiveBefore, archiveAfter) || fileHash(archivePath) !== selected.archive.sha256 || !sameSnapshot(toolBefore, toolAfter) ||
      fileHash(tool) !== toolInfo.sha256 || fileHash(__filename) !== producerSha256 || !isDeepStrictEqual(cudartRecipe(), recipe)) {
    throw new Error("Windows CUDA Runtime source, extractor or recipe changed during preparation");
  }
  return receipt;
}

/** @param {string} directory @returns {CudartReceipt} */
function admitWindowsCudart(directory) {
  const selected = windowsCudartCatalog();
  if (!absolute(directory) || directory === repo || directory.startsWith(repo + path.sep)) {throw new Error("Windows CUDA Runtime admission needs a canonical restored root");}
  const rootStat = fs.lstatSync(directory, {bigint: true});
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(directory) !== directory || writableMode(rootStat) ||
      !isDeepStrictEqual(fs.readdirSync(directory).sort(), ["licenses", "runtime", "sdk.json"])) {
    throw new Error("Windows CUDA Runtime output has an invalid or writable top-level layout");
  }
  /** @type {Array<[string,import("node:fs").BigIntStats]>} */ const directories = [[directory, rootStat]];
  const selectedDirectories = /** @type {Array<[string,string[]]>} */ ([
    ["licenses", ["cuda-cudart"]], ["licenses/cuda-cudart", ["LICENSE"]], ["runtime", ["cudart_static.lib"]],
  ]);
  for (const [relative, children] of selectedDirectories) {
    const file = path.join(directory, ...relative.split("/")), stat = fs.lstatSync(file, {bigint: true});
    if (!stat.isDirectory() || stat.isSymbolicLink() || writableMode(stat) || !isDeepStrictEqual(fs.readdirSync(file).sort(), children)) {
      throw new Error("Windows CUDA Runtime output has an invalid directory: " + relative);
    }
    directories.push([file, stat]);
  }
  const receiptFile = path.join(directory, "sdk.json"), receiptStat = regularFileSnapshot(receiptFile, "receipt", 1 << 20),
    {sha256: receiptSha256, value: parsed} = readBuildJsonSnapshot(receiptFile, 1 << 20), receipt = /** @type {CudartReceipt} */ (parsed);
  const expectedFiles = Object.keys(outputFiles).sort();
  if (writableMode(receiptStat) || !hasExactBuildFields(receipt, ["version", "target", "status", "networked", "buildOnly", "compiled", "vendorPrebuilt",
    "publicationQualified", "redistributionQualified", "producerSha256", "recipe", "tool", "source", "output", "files"]) ||
      receipt.version !== 1 || receipt.target !== "windows-cudart" || receipt.status !== "PASS" || receipt.networked !== false || receipt.buildOnly !== true ||
      receipt.compiled !== false || receipt.vendorPrebuilt !== true || receipt.publicationQualified !== false || receipt.redistributionQualified !== false ||
      !/^[a-f0-9]{64}$/.test(receipt.producerSha256) || !provenancePath(receipt.output) ||
      !hasExactBuildFields(receipt.recipe, ["sha256", "files"]) || !/^[a-f0-9]{64}$/.test(receipt.recipe.sha256) ||
      !hasExactBuildFields(receipt.recipe.files, recipeNames) || Object.values(receipt.recipe.files).some(hash => typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash)) ||
      !hasExactBuildFields(receipt.source, ["archivePath", "url", "sha256", "bytes", "manifestUrl"]) || !provenancePath(receipt.source.archivePath) ||
      receipt.source.url !== selected.archive.url || receipt.source.sha256 !== selected.archive.sha256 || receipt.source.bytes !== selected.archive.bytes ||
      receipt.source.manifestUrl !== selected.manifestUrl || !hasExactBuildFields(receipt.tool, ["path", "sha256", "version"]) ||
      !provenancePath(receipt.tool.path) || !/^[a-f0-9]{64}$/.test(receipt.tool.sha256) || !provenancePath(receipt.tool.version) ||
      !hasExactBuildFields(receipt.files, expectedFiles)) {
    throw new Error("Windows CUDA Runtime receipt is stale, malformed or incomplete");
  }
  /** @type {Array<[string,import("node:fs").BigIntStats,string,number]>} */ const outputs = [];
  for (const [relative, expected] of Object.entries(outputFiles)) {
    const file = path.join(directory, ...relative.split("/")), saved = receipt.files[relative], actual = verifyDigest(file, "member", memberMaximumBytes);
    if (!saved || !hasExactBuildFields(saved, ["kind", "bytes", "sha256", "source"]) || saved.kind !== "file" || saved.bytes !== expected.bytes || saved.sha256 !== expected.sha256 ||
        !hasExactBuildFields(saved.source, ["archiveSha256", "member"]) || saved.source.archiveSha256 !== selected.archive.sha256 || saved.source.member !== expected.member ||
        actual.bytes !== expected.bytes || actual.sha256 !== expected.sha256 || writableMode(fs.lstatSync(file, {bigint: true}))) {
      throw new Error("Windows CUDA Runtime output differs from its pinned member: " + relative);
    }
    outputs.push([file, actual.before, "member", memberMaximumBytes]);
  }
  outputs.push([receiptFile, receiptStat, "receipt", 1 << 20]);
  const catalogStat = regularFileSnapshot(catalogFile, "catalog", 64 << 10), catalogSha256 = fileHash(catalogFile);
  if (fileHash(receiptFile) !== receiptSha256 || fileHash(catalogFile) !== catalogSha256 ||
      !sameSnapshot(catalogStat, regularFileSnapshot(catalogFile, "catalog", 64 << 10)) ||
      outputs.some(([file, before, label, maximum]) => !sameSnapshot(before, regularFileSnapshot(file, label, maximum))) ||
      directories.some(([directoryPath, before]) => !sameSnapshot(before, fs.lstatSync(directoryPath, {bigint: true})))) {
    throw new Error("Windows CUDA Runtime restored inputs changed during admission");
  }
  return receipt;
}

/** @param {string} output @returns {CudartInputs} */
function windowsCudartInputs(output) {
  const directory = path.resolve(output), receiptFile = path.join(directory, "sdk.json"), receipt = admitWindowsCudart(directory),
    receiptSha256 = fileHash(receiptFile), files = /** @type {Record<string,string>} */ ({[receiptFile]: receiptSha256,
      [path.join(directory, "runtime/cudart_static.lib")]: catalog.cudartStatic.sha256,
      [path.join(directory, "licenses/cuda-cudart/LICENSE")]: catalog.license.sha256,
      [catalogFile]: fileHash(catalogFile)});
  for (const [file, hash] of Object.entries(files)) {if (fileHash(file) !== hash) {throw new Error("Windows CUDA Runtime input changed after admission: " + file);}}
  return {receipt, receiptFile, receiptSha256, cudartStatic: path.join(directory, "runtime/cudart_static.lib"),
    license: path.join(directory, "licenses/cuda-cudart/LICENSE"), files};
}

if (require.main === module) {
  try {
    const [archive, output, extra] = process.argv.slice(2);
    if (!archive || !output || extra) {throw new Error("Usage: prepare-native-cudart-windows.js /pinned/cuda_cudart-12.6.37.zip /new/cudart-output");}
    const receipt = prepareWindowsCudart(archive, output);
    console.log(`[native-sdk] Prepared only pinned CUDA Runtime ${catalog.vendorVersion} library and license at ${receipt.output}; no full SDK, compilation, or publication qualification.`);
  } catch (error) {console.error(error instanceof Error ? error.message : "Windows CUDA Runtime preparation failed"); process.exitCode = 1;}
}

module.exports = {prepareWindowsCudart, admitWindowsCudart, windowsCudartInputs, windowsCudartCatalog, cudartRecipe, stagePinnedZipMembers, trustedUnzipSnapshot};
