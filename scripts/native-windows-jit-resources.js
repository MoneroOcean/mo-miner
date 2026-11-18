"use strict";

// Build the existing source-JIT resource overlay from admitted same-run producers.
// This module does not install GNU packages or transport artifacts.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildJson, sourceInputs, verifySourceInputs, buildFileInputs} = require("./build-source-inputs");
const {jitSourceInputs, buildJitEmitter} = require("./native-jit-sdk");
const {prepare: prepareJitResources, verifyJitResources, validateProfile, resourceTarget} = require("./prepare-native-sycl-jit-resources");
const {auditGnuHeaders} = require("./prepare-native-gnu-headers");
const {REQUIRED} = require("./prepare-native-jit-base");
const {verifyPreparedSdkSources} = require("./prepare-native-sdk-sources");
const {ciSuffix, validCiSuffix, ciBuildJobs} = require("./native-ci-job");

const repo = path.resolve(__dirname, ".."), linuxTarget = "x86_64-unknown-linux-gnu", windowsTarget = "x86_64-w64-windows-gnu";
// Mirrors the private RECIPE_FILES list in prepare-native-jit-base.js; the producer does not export it.
const BASE_RECIPE_FILES = ["scripts/prepare-native-jit-base.js", "scripts/prepare-native-sdk-sources.js", "scripts/native-dpcpp-pins.js",
  "scripts/build-source-inputs.js", "scripts/build-environment.js", "scripts/build-native-release.js", "scripts/native-jit-headers.js",
  "scripts/native-jit-sdk.js", "scripts/prepare-native-sycl-jit-resources.js"];
const EMITTER_RECIPE_FILES = [...new Set([...jitSourceInputs, "scripts/native-jit-sdk.js", "scripts/build-source-inputs.js", "scripts/build-environment.js"])];
const MAX_RECEIPT_FILES = 16384, MAX_COPYRIGHTS = 4 << 20, MAX_NATIVE_TOOL_BYTES = 16 << 20;
/** @typedef {{role:string,name:string,root:string,sha256:string}} InputArchive */
/** @typedef {{version:number,target:string,buildOnly:boolean,executed:boolean,publicationQualified:boolean,inputArchives:InputArchive[],
 * base:{receipt:string,sha256:string,manifest:string,manifestSha256:string},resources:{manifest:string,manifestSha256:string,files:number},
 * bootstrap:{cache:string,ninja:string}}} LinuxResult */
/** @typedef {{version:number,baseRoot:string,baseManifest:string,baseManifestSha256:string,bytes:number,files:number,buildOnly:boolean,compiled:boolean,
 * executed:boolean,publicationQualified:boolean,required:typeof REQUIRED,recipe:unknown,sourceReceipt:{path:string,sha256:string,revisions:Record<string,string>},
 * dpcpp:{sdkRoot:string,sourceCommit:string},cuda:{root:string},build:{root:string,cmakeCacheSha256:string,ninjaGraphSha256:string,
 * clangTableGenSha256:string,llvmTableGenSha256:string}}} BaseReceipt */
/** @typedef {{version:number,files:Array<{source:string,virtual:string,sha256:string}>}} BaseManifest */
/** @typedef {{version:number,fixture:string,fixtureSha256:string,compiler:string,compilerSha256:string,inputs:Record<string,string>,recipe:unknown,
 * targetHeaders:string[],buildOnly:boolean,executed:boolean,publicationQualified:boolean,scope:string}} EmitterReceipt */

/** @param {unknown} value @param {string} label */
function hash(value, label) {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) {throw new Error("Invalid " + label + " SHA-256");}
  return value;
}

/** @param {string} file @param {string} label @param {number} [maximum] */
function regularFile(file, label, maximum = 8 << 20) {
  if (!path.isAbsolute(file) || path.resolve(file) !== file) {throw new Error(label + " path is not canonical");}
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size < 1 || stat.size > maximum || fs.realpathSync(file) !== file) {
    throw new Error(label + " is not a bounded regular file: " + file);
  }
  return stat;
}

/** @param {string} directory @param {string} label */
function directory(directory, label) {
  if (!path.isAbsolute(directory) || path.resolve(directory) !== directory) {throw new Error(label + " path is not canonical");}
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!stat?.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {throw new Error(label + " is not a canonical directory");}
  return directory;
}

/** @param {string} root @param {string} file */
function within(root, file) {
  const relative = path.relative(root, file);
  return relative === "" || (relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative));
}

/** @param {unknown} value @param {string[]} keys */
function exactKeys(value, keys) {
  return !!value && typeof value === "object" && !Array.isArray(value) &&
    Object.keys(value).sort().join("\0") === [...keys].sort().join("\0");
}

/** The archive name and all producer roots carry the same CI work-root suffix. @param {string} value */
function runSuffix(value) {
  if (!validCiSuffix(value)) {throw new Error("Invalid Windows JIT CI work-root suffix");}
  return value;
}

/** @param {string} root @param {string} suffix @param {string} name */
function runRoot(root, suffix, name) {
  directory(root, name);
  if (root !== path.join("/tmp", name + "-" + suffix)) {throw new Error(name + " is not the canonical current-run root");}
  return root;
}

/** One resource artifact root retains emitter/profile evidence beside the prepared payload.
 * @param {string} suffix */
function windowsJitResourcePaths(suffix) {
  const root = path.join("/tmp", "mom-native-windows-jit-resources-" + runSuffix(suffix));
  return {root, emitter: path.join(root, "emitter"), profile: path.join(root, "resource-profile.json"), resources: path.join(root, "resources"),
    nativeTools: path.join(root, "native-tools")};
}

/** Verify Linux resource-stage lineage and its complete base manifest before reuse. @param {string} root @param {string} suffix @param {string} expectedResultSha256 */
function admitLinuxBase(root, suffix, expectedResultSha256) {
  runRoot(root, suffix, "mom-native-linux-jit-resources");
  const ciResultFile = path.join(root, "ci-result.json");
  regularFile(ciResultFile, "Linux JIT result");
  if (fileHash(ciResultFile) !== hash(expectedResultSha256, "Linux JIT result")) {throw new Error("Linux JIT result differs from its external pin");}
  const result = /** @type {LinuxResult} */ (readBuildJson(ciResultFile));
  if (result?.version !== 1 || result.target !== linuxTarget || result.buildOnly !== true || result.executed !== false || result.publicationQualified !== false) {
    throw new Error("Linux JIT artifact is not the same-run build-only resource producer");
  }
  const sourceRoot = "mom-native-linux-source-prerequisites-" + suffix,
    dpcppRoot = "mom-native-dpcpp-cuda-inputs-" + suffix, gnuRoot = "mom-native-gnu-headers-" + suffix;
  const expectedArchives = [
    {role: "linux-sources", root: sourceRoot, name: "native-linux-sdk-state-" + suffix + "-sources.tar"},
    {role: "dpcpp-cuda-inputs", root: dpcppRoot, name: "native-linux-dpcpp-cuda-inputs-" + suffix + ".tar"},
    {role: "gnu-headers", root: gnuRoot, name: "native-linux-gnu-headers-" + suffix + ".tar"},
  ];
  if (!Array.isArray(result.inputArchives) || result.inputArchives.length !== expectedArchives.length ||
      result.inputArchives.some((/** @type {InputArchive} */ row, /** @type {number} */ index) => {
        const expected = expectedArchives[index];
        return !expected || !exactKeys(row, ["role", "name", "root", "sha256"]) || row.role !== expected.role || row.root !== expected.root ||
          row.name !== expected.name || !/^[a-f0-9]{64}$/.test(row.sha256 || "");
      })) {throw new Error("Linux JIT artifact does not bind the exact same-run source, DPC++/CUDA and GNU archives");}

  const baseRoot = directory(path.join(root, "base"), "Linux JIT base output"), base = directory(path.join(baseRoot, "base"), "Linux JIT base payload"),
    baseReceiptFile = path.join(baseRoot, "result.json"),
    baseManifestFile = path.join(baseRoot, "base-manifest.json"), resourceDir = path.join(root, "resources");
  if (!result.base || result.base.receipt !== baseReceiptFile || result.base.manifest !== baseManifestFile ||
      result.resources?.manifest !== path.join(resourceDir, "manifest.json")) {throw new Error("Linux JIT result paths do not identify the transported producer files");}
  regularFile(baseReceiptFile, "Linux JIT base receipt"); regularFile(baseManifestFile, "Linux JIT base manifest");
  if (fileHash(baseReceiptFile) !== hash(result.base.sha256, "Linux JIT base receipt") ||
      fileHash(baseManifestFile) !== hash(result.base.manifestSha256, "Linux JIT base manifest")) {
    throw new Error("Linux JIT base receipt or manifest differs from its producer result");
  }
  const receipt = /** @type {BaseReceipt} */ (readBuildJson(baseReceiptFile)), manifest = /** @type {BaseManifest} */ (readBuildJson(baseManifestFile));
  if (receipt?.version !== 1 || receipt.baseRoot !== base || receipt.baseManifest !== baseManifestFile ||
      receipt.baseManifestSha256 !== result.base.manifestSha256 || receipt.buildOnly !== true || receipt.compiled !== false ||
      receipt.executed !== false || receipt.publicationQualified !== false || !isDeepStrictEqual(receipt.required, REQUIRED)) {
    throw new Error("Linux JIT base receipt is stale, incomplete or not build-only");
  }
  verifySourceInputs(receipt.recipe, sourceInputs(repo, [], BASE_RECIPE_FILES), "Linux JIT base producer");
  if (!receipt.sourceReceipt || receipt.sourceReceipt.path !== path.join("/tmp", sourceRoot, "prepared/compiler/sources.json") ||
      !/^[a-f0-9]{64}$/.test(receipt.sourceReceipt.sha256 || "") || !receipt.dpcpp ||
      receipt.dpcpp.sdkRoot !== path.join("/tmp", dpcppRoot, "dpcpp") || receipt.cuda?.root !== path.join("/tmp", dpcppRoot, "cuda") ||
      !/^[a-f0-9]{40}$/.test(receipt.dpcpp.sourceCommit || "") || receipt.sourceReceipt.revisions?.["dpcpp"] !== receipt.dpcpp.sourceCommit) {
    throw new Error("Linux JIT base is not bound to the exact same-run source and DPC++/CUDA roots");
  }
  const sourceReceiptFile = receipt.sourceReceipt.path;
  regularFile(sourceReceiptFile, "same-run compiler source receipt");
  const sourceReceiptSha256 = fileHash(sourceReceiptFile), sourceReceipt = verifyPreparedSdkSources(sourceReceiptFile, "compiler"), dpcppSource = sourceReceipt.prepared?.dpcpp,
    buildRoot = path.join(root, "bootstrap/build"), cacheFile = path.join(buildRoot, "CMakeCache.txt"), ninjaFile = path.join(buildRoot, "build.ninja");
  if (sourceReceiptSha256 !== receipt.sourceReceipt.sha256 || !dpcppSource || dpcppSource.revision !== receipt.dpcpp.sourceCommit ||
      dpcppSource.root !== path.join("/tmp", sourceRoot, "prepared/compiler/dpcpp") || receipt.build?.root !== buildRoot ||
      !/^[a-f0-9]{64}$/.test(receipt.build.clangTableGenSha256 || "") || !/^[a-f0-9]{64}$/.test(receipt.build.llvmTableGenSha256 || "") ||
      !result.bootstrap || result.bootstrap.cache !== receipt.build.cmakeCacheSha256 || result.bootstrap.ninja !== receipt.build.ninjaGraphSha256 ||
      fileHash(cacheFile) !== result.bootstrap.cache || fileHash(ninjaFile) !== result.bootstrap.ninja) {
    throw new Error("Linux JIT base TableGen inputs are not bound to the current same-run compiler source and build graph");
  }
  /** @type {Array<{name:string,key:"clangTableGenSha256"|"llvmTableGenSha256"}>} */ const nativeToolRows = [
    {name: "clang-tblgen", key: "clangTableGenSha256"},
    {name: "llvm-tblgen", key: "llvmTableGenSha256"},
  ];
  const nativeTools = nativeToolRows.map(row => {
    const file = path.join(buildRoot, "bin", row.name), stat = regularFile(file, "Linux JIT native TableGen tool", MAX_NATIVE_TOOL_BYTES),
      sha256 = hash(receipt.build[row.key], "Linux JIT native TableGen tool");
    if ((stat.mode & 0o777) !== 0o755 || fileHash(file) !== sha256) {throw new Error("Linux JIT native TableGen tool differs from its base receipt: " + row.name);}
    return {name: row.name, source: file, bytes: stat.size, sha256};
  });
  if (!manifest || manifest.version !== 1 || !Array.isArray(manifest.files) || !manifest.files.length || manifest.files.length > MAX_RECEIPT_FILES ||
      !Number.isSafeInteger(receipt.bytes) || receipt.bytes < 1 || receipt.bytes > (64 << 20) || receipt.files !== manifest.files.length ||
      !Number.isSafeInteger(result.resources.files) || result.resources.files < 1) {throw new Error("Linux JIT base manifest is malformed or excessive");}
  let total = 0;
  const seen = new Set();
  for (const row of manifest.files) {
    if (!row || typeof row.source !== "string" || typeof row.virtual !== "string" || row.source !== row.virtual ||
        row.source.startsWith("/") || row.source.includes("\\") || row.source.split("/").some((/** @type {string} */ part) => !part || part === "." || part === "..") ||
        !/^[a-f0-9]{64}$/.test(row.sha256 || "") || seen.has(row.source.toLowerCase())) {throw new Error("Invalid or duplicate Linux JIT base entry");}
    seen.add(row.source.toLowerCase());
    const file = path.join(base, ...row.source.split("/")), stat = regularFile(file, "Linux JIT base payload", 64 << 20);
    total += stat.size;
    if (total > (64 << 20) || fileHash(file) !== row.sha256) {throw new Error("Linux JIT base payload changed or exceeds bounds");}
  }
  if (total !== receipt.bytes) {throw new Error("Linux JIT base byte count changed");}
  verifyRequiredBaseManifest(manifest);
  const resources = verifyJitResources(resourceDir, linuxTarget);
  if (resources.manifestHash !== result.resources.manifestSha256 || result.resources.manifestSha256 !== fileHash(result.resources.manifest)) {
    throw new Error("Linux JIT resource manifest differs from its producer result");
  }
  return {root, result, resultSha256: expectedResultSha256, baseRoot: base, baseManifest: baseManifestFile,
    baseManifestSha256: result.base.manifestSha256, nativeTools: {sourceReceiptFile, sourceReceiptSha256, dpcppCommit: receipt.dpcpp.sourceCommit,
      baseReceiptFile, baseReceiptSha256: fileHash(baseReceiptFile), buildRoot, files: nativeTools}};
}

/** Validate the tiny host-tool payload independently from the runtime resource manifest.
 * The caller supplies source identity only after admitting the original compiler source receipt.
 * @param {string} outerRoot @param {string} suffix
 * @param {{receiptFile:string,receiptSha256:string,dpcppCommit:string}} source
 * @returns {{toolRoot:string,manifestFile:string,manifestSha256:string,files:{clangTableGen:string,llvmTableGen:string},
 * hashes:{clangTableGenSha256:string,llvmTableGenSha256:string},source:{receiptFile:string,receiptSha256:string,commit:string},
 * linux:{root:string,resultSha256:string,baseReceiptSha256:string,buildRoot:string}}} */
function validateWindowsJitNativeTools(outerRoot, suffix, source) {
  runRoot(outerRoot, suffix, "mom-native-windows-jit-resources");
  const nativeRoot = path.join(outerRoot, "native-tools"), binRoot = path.join(nativeRoot, "bin"), manifestFile = path.join(nativeRoot, "manifest.json");
  directory(nativeRoot, "Windows JIT native-tools root"); directory(binRoot, "Windows JIT native-tools bin");
  if (!isDeepStrictEqual(fs.readdirSync(nativeRoot).sort(), ["bin", "manifest.json"]) ||
      !isDeepStrictEqual(fs.readdirSync(binRoot).sort(), ["clang-tblgen", "llvm-tblgen"])) {
    throw new Error("Windows JIT native-tools payload has missing or extra entries");
  }
  regularFile(manifestFile, "Windows JIT native-tools manifest", 64 << 10);
  const manifest = readBuildJson(manifestFile), sourceRoot = "mom-native-linux-source-prerequisites-" + suffix,
    linuxRoot = path.join("/tmp", "mom-native-linux-jit-resources-" + suffix), expectedSourceReceipt = path.join("/tmp", sourceRoot, "prepared/compiler/sources.json"),
    expectedBuildRoot = path.join(linuxRoot, "bootstrap/build");
  if (!exactKeys(manifest, ["version", "kind", "buildOnly", "executed", "publicationQualified", "linux", "source", "files"]) ||
      manifest.version !== 1 || manifest.kind !== "windows-jit-native-tablegen" ||
      manifest.buildOnly !== true || manifest.executed !== false || manifest.publicationQualified !== false ||
      !exactKeys(manifest.linux, ["root", "resultSha256", "baseReceiptSha256", "buildRoot"]) ||
      manifest.linux.root !== linuxRoot || !/^[a-f0-9]{64}$/.test(manifest.linux.resultSha256 || "") ||
      !/^[a-f0-9]{64}$/.test(manifest.linux.baseReceiptSha256 || "") || manifest.linux.buildRoot !== expectedBuildRoot ||
      !exactKeys(manifest.source, ["receiptFile", "receiptSha256", "dpcppCommit"]) || manifest.source.receiptFile !== expectedSourceReceipt ||
      manifest.source.receiptFile !== source?.receiptFile || manifest.source.receiptSha256 !== source?.receiptSha256 ||
      manifest.source.dpcppCommit !== source?.dpcppCommit || !/^[a-f0-9]{64}$/.test(manifest.source.receiptSha256 || "") ||
      !/^[a-f0-9]{40}$/.test(manifest.source.dpcppCommit || "") || !Array.isArray(manifest.files) || manifest.files.length !== 2) {
    throw new Error("Windows JIT native-tools manifest is not bound to the current same-run Linux JIT source");
  }
  const expected = ["clang-tblgen", "llvm-tblgen"];
  /** @type {{clangTableGen:string,llvmTableGen:string}} */ const files = {clangTableGen: "", llvmTableGen: ""};
  let total = 0;
  for (let index = 0; index < expected.length; index++) {
    const name = expected[index], row = manifest.files[index];
    if (!row || !exactKeys(row, ["path", "bytes", "mode", "sha256"]) || row.path !== "bin/" + name ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > MAX_NATIVE_TOOL_BYTES || row.mode !== 0o755 ||
        !/^[a-f0-9]{64}$/.test(row.sha256 || "")) {throw new Error("Windows JIT native-tools file row is invalid: " + name);}
    const file = path.join(nativeRoot, ...row.path.split("/")), stat = regularFile(file, "Windows JIT native TableGen tool", MAX_NATIVE_TOOL_BYTES);
    total += stat.size;
    if (total > (24 << 20) || stat.size !== row.bytes || (stat.mode & 0o777) !== row.mode || fileHash(file) !== row.sha256) {
      throw new Error("Windows JIT native TableGen tool changed or exceeds bounds: " + name);
    }
    files[name === "clang-tblgen" ? "clangTableGen" : "llvmTableGen"] = file;
  }
  return {toolRoot: nativeRoot, manifestFile, manifestSha256: fileHash(manifestFile), files,
    hashes: {clangTableGenSha256: manifest.files[0].sha256, llvmTableGenSha256: manifest.files[1].sha256},
    source: {receiptFile: source.receiptFile, receiptSha256: source.receiptSha256, commit: source.dpcppCommit},
    linux: manifest.linux};
}

/** Copy only the two verified build-only generators beside the Windows resource payload.
 * @param {string} outerRoot @param {string} suffix
 * @param {ReturnType<typeof admitLinuxBase>} linux */
function writeWindowsJitNativeTools(outerRoot, suffix, linux) {
  const nativeRoot = path.join(outerRoot, "native-tools"), binRoot = path.join(nativeRoot, "bin");
  if (fs.lstatSync(nativeRoot, {throwIfNoEntry: false})) {throw new Error("Windows JIT native-tools output already exists");}
  fs.mkdirSync(binRoot, {recursive: true, mode: 0o700});
  const rows = linux.nativeTools.files.map(row => {
    const file = path.join(binRoot, row.name);
    fs.copyFileSync(row.source, file, fs.constants.COPYFILE_EXCL); fs.chmodSync(file, 0o755);
    const stat = regularFile(file, "copied Windows JIT native TableGen tool", MAX_NATIVE_TOOL_BYTES);
    if ((stat.mode & 0o777) !== 0o755 || stat.size !== row.bytes || fileHash(file) !== row.sha256) {
      throw new Error("Copied Windows JIT native TableGen tool differs from its admitted Linux build: " + row.name);
    }
    return {path: "bin/" + row.name, bytes: stat.size, mode: 0o755, sha256: row.sha256};
  });
  const manifest = {version: 1, kind: "windows-jit-native-tablegen",
    buildOnly: true, executed: false, publicationQualified: false,
    linux: {root: linux.root, resultSha256: linux.resultSha256, baseReceiptSha256: linux.nativeTools.baseReceiptSha256,
      buildRoot: linux.nativeTools.buildRoot},
    source: {receiptFile: linux.nativeTools.sourceReceiptFile, receiptSha256: linux.nativeTools.sourceReceiptSha256,
      dpcppCommit: linux.nativeTools.dpcppCommit}, files: rows};
  fs.writeFileSync(path.join(nativeRoot, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return validateWindowsJitNativeTools(outerRoot, suffix, {receiptFile: linux.nativeTools.sourceReceiptFile,
    receiptSha256: linux.nativeTools.sourceReceiptSha256, dpcppCommit: linux.nativeTools.dpcppCommit});
}

/** Re-admit the current same-run source receipt and its two generated native tools. @param {string} outerRoot @param {string} suffix */
function admitWindowsJitNativeTools(outerRoot, suffix) {
  const expectedSourceReceipt = path.join("/tmp", "mom-native-linux-source-prerequisites-" + runSuffix(suffix),
    "prepared/compiler/sources.json");
  regularFile(expectedSourceReceipt, "same-run compiler source receipt");
  const sourceReceiptSha256 = fileHash(expectedSourceReceipt), receipt = verifyPreparedSdkSources(expectedSourceReceipt, "compiler"),
    dpcpp = receipt.prepared?.dpcpp;
  if (!dpcpp || !/^[a-f0-9]{40}$/.test(dpcpp.revision || "")) {throw new Error("Same-run compiler source receipt omits its DPC++ revision");}
  return validateWindowsJitNativeTools(outerRoot, suffix, {receiptFile: expectedSourceReceipt, receiptSha256: sourceReceiptSha256, dpcppCommit: dpcpp.revision});
}

/** The base producer materializes SDK inputs at their virtual destinations: libdevice gains
 * a cuda/ prefix. REQUIRED.source names the SDK input, not this staged manifest path.
 * Check the inventory only after the caller has verified every payload hash.
 * @param {BaseManifest} manifest */
function verifyRequiredBaseManifest(manifest) {
  for (const [name, required] of Object.entries(REQUIRED)) {
    const row = manifest.files.find(item => item.source === required.virtual && item.virtual === required.virtual);
    if (!row || row.sha256 !== required.sha256) {throw new Error("Linux JIT base omitted its pinned " + name + " bitcode");}
  }
}

/** Verify package-owned Windows GNU headers and the adjacent captured copyright documents. @param {string} root @param {string} suffix @param {string} resultSha256 @param {string} copyrightsSha256 */
function admitWindowsGnu(root, suffix, resultSha256, copyrightsSha256) {
  runRoot(root, suffix, "mom-native-windows-gnu-headers");
  const resultFile = path.join(root, "result.json"), copyrightFile = path.join(root, "copyrights.json");
  regularFile(resultFile, "Windows GNU header result"); regularFile(copyrightFile, "Windows GNU copyrights", MAX_COPYRIGHTS);
  if (fileHash(resultFile) !== hash(resultSha256, "Windows GNU result") || fileHash(copyrightFile) !== hash(copyrightsSha256, "Windows GNU copyrights")) {
    throw new Error("Windows GNU header or copyright metadata differs from its external pin");
  }
  const {headers, notices} = auditGnuHeaders({root}, windowsTarget);
  return {root, headers, notices, resultSha256, copyrightsSha256};
}

/** Validate a fresh emitter receipt against the admitted DPC++ compiler and exact GNU header list. */
/** @param {string} emitterRoot @param {string} dpcppRoot @param {string} gnuRoot
 * @param {{files:unknown[]}} gnuHeaders @param {string} expectedSha256 */
function admitEmitter(emitterRoot, dpcppRoot, gnuRoot, gnuHeaders, expectedSha256) {
  directory(emitterRoot, "Windows JIT emitter root"); directory(dpcppRoot, "Windows DPC++ SDK root");
  const receiptFile = path.join(emitterRoot, "result.json");
  regularFile(receiptFile, "Windows JIT emitter receipt");
  if (fileHash(receiptFile) !== hash(expectedSha256, "Windows JIT emitter receipt")) {throw new Error("Windows JIT emitter receipt differs from its external pin");}
  const receipt = /** @type {EmitterReceipt} */ (readBuildJson(receiptFile)), fixture = path.join(emitterRoot, "native-kawpow-source"), compiler = path.join(dpcppRoot, "bin/clang++"),
    targetHeaders = /** @type {string[]} */ (readBuildJson(path.join(gnuRoot, "target-headers.json")));
  if (!exactKeys(receipt, ["version", "fixture", "fixtureSha256", "compiler", "compilerSha256", "inputs", "recipe", "targetHeaders",
    "buildOnly", "executed", "publicationQualified", "scope"]) || receipt.version !== 1 || receipt.fixture !== fixture ||
      receipt.compiler !== compiler || receipt.buildOnly !== true || receipt.executed !== false || receipt.publicationQualified !== false ||
      !Array.isArray(targetHeaders) || !isDeepStrictEqual(receipt.targetHeaders, targetHeaders) ||
      targetHeaders.length !== gnuHeaders.files.length || !/^[a-f0-9]{64}$/.test(receipt.fixtureSha256 || "") ||
      !/^[a-f0-9]{64}$/.test(receipt.compilerSha256 || "")) {throw new Error("Windows JIT emitter is not bound to the fresh Windows GNU header set");}
  const fixtureStat = regularFile(fixture, "Windows JIT source fixture", 64 << 20), compilerStat = fs.statSync(compiler, {throwIfNoEntry: false});
  if (!fixtureStat.size || !compilerStat?.isFile() || !within(dpcppRoot, fs.realpathSync(compiler)) ||
      fileHash(fixture) !== receipt.fixtureSha256 || fileHash(compiler) !== receipt.compilerSha256) {
    throw new Error("Windows JIT emitter fixture or selected DPC++ compiler changed");
  }
  const recipe = sourceInputs(repo, [], EMITTER_RECIPE_FILES);
  verifySourceInputs(receipt.recipe, recipe, "Windows JIT emitter producer");
  if (!receipt.inputs || typeof receipt.inputs !== "object" || Array.isArray(receipt.inputs)) {throw new Error("Windows JIT emitter input map is missing");}
  const files = Object.keys(receipt.inputs);
  if (!files.length || files.length > MAX_RECEIPT_FILES || files.some(file => !path.isAbsolute(file) || path.resolve(file) !== file ||
      !/^[a-f0-9]{64}$/.test(receipt.inputs[file] || "") || fs.realpathSync(file) !== file)) {throw new Error("Windows JIT emitter inputs are invalid or excessive");}
  const inputHashes = buildFileInputs(files);
  if (!isDeepStrictEqual(inputHashes, receipt.inputs)) {throw new Error("Windows JIT emitter inputs differ from their receipt");}
  const requiredInputs = [process.execPath, compiler, path.join(dpcppRoot, "include/sycl/sycl.hpp"), ...targetHeaders,
    ...EMITTER_RECIPE_FILES.map(name => path.join(repo, name))].map(file => fs.realpathSync(file));
  if (requiredInputs.some(file => !Object.hasOwn(receipt.inputs, file))) {throw new Error("Windows JIT emitter omitted a compiler, header or maintained source input");}
  return {root: emitterRoot, receiptFile, receiptSha256: expectedSha256, receipt, fixture, compiler};
}

/** Pure mapping from admitted producer records to the established JIT profile schema. @internal
 * @param {{baseRoot:string,baseManifest:string}} base
 * @param {{root:string,notices:Array<{package:string,path:string,sha256:string}>}} gnu
 * @param {{fixture:string,compiler:string,receiptFile:string}} emitter */
function profileFromAdmissions(base, gnu, emitter) {
  const target = resourceTarget(windowsTarget), baseRoot = base.baseRoot, gnuRoot = gnu.root;
  const profile = {version: 1, target: target.target, fixture: emitter.fixture, clang: emitter.compiler, receipt: emitter.receiptFile,
    gnuRoot, baseRoot, noticeRoot: gnuRoot, syclInclude: path.join(baseRoot, "include"),
    clangBuiltinInclude: path.join(baseRoot, "lib/clang/23/include"), baseManifest: base.baseManifest,
    headerRoots: target.roots.map(virtual => ({path: path.join(gnuRoot, ...virtual.split("/")), virtual})),
    required: Object.fromEntries(Object.entries(REQUIRED).map(([name, row]) => [name, {virtual: row.virtual, sha256: row.sha256}])),
    notices: gnu.notices.map((/** @type {{package:string,path:string,sha256:string}} */ notice) => ({source: path.relative(gnuRoot, notice.path).split(path.sep).join("/"),
      virtual: "usr/share/licenses/" + notice.package + "/copyright", sha256: notice.sha256}))};
  return validateProfile(profile);
}

/**
 * Compose the existing Windows GNU resource profile after admitting all three independent
 * producer records. Archive extraction and its external bundle hash remain the caller's job.
 * @param {{suffix:string,linuxResourcesRoot:string,linuxResultSha256:string,windowsGnuRoot:string,windowsGnuResultSha256:string,
 * windowsGnuCopyrightsSha256:string,emitterRoot:string,emitterResultSha256:string,dpcppRoot:string,outputProfile:string}} input
 */
function composeWindowsJitResourceProfile(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {throw new Error("Windows JIT resource inputs are required");}
  const paths = windowsJitResourcePaths(input.suffix), suffix = input.suffix;
  runRoot(input.linuxResourcesRoot, suffix, "mom-native-linux-jit-resources");
  runRoot(input.windowsGnuRoot, suffix, "mom-native-windows-gnu-headers");
  directory(paths.root, "Windows JIT resource output");
  const emitterRoot = directory(input.emitterRoot, "Windows JIT emitter root"), dpcppRoot = directory(input.dpcppRoot, "Windows DPC++ SDK root"),
    output = input.outputProfile;
  if (typeof output !== "string" || emitterRoot !== paths.emitter || output !== paths.profile ||
      !dpcppRoot.split(path.sep).some(part => part.endsWith(suffix))) {
    throw new Error("Windows JIT emitter, DPC++ SDK or profile path is not bound to this CI work root");
  }
  hash(input.linuxResultSha256, "Linux JIT result"); hash(input.windowsGnuResultSha256, "Windows GNU result");
  hash(input.windowsGnuCopyrightsSha256, "Windows GNU copyrights"); hash(input.emitterResultSha256, "Windows JIT emitter receipt");
  const base = admitLinuxBase(input.linuxResourcesRoot, suffix, input.linuxResultSha256),
    gnu = admitWindowsGnu(input.windowsGnuRoot, suffix, input.windowsGnuResultSha256, input.windowsGnuCopyrightsSha256),
    emitter = admitEmitter(emitterRoot, dpcppRoot, input.windowsGnuRoot, gnu.headers, input.emitterResultSha256);
  if (!path.isAbsolute(output) || path.resolve(output) !== output || fs.lstatSync(output, {throwIfNoEntry: false}) ||
      fs.realpathSync(path.dirname(output)) !== path.dirname(output) ||
      [base.root, gnu.root, emitter.root, dpcppRoot].some(root => within(root, output) || within(output, root))) {
    throw new Error("Windows JIT profile output must be a new canonical same-run sibling outside every input");
  }
  const profile = profileFromAdmissions(base, gnu, emitter);
  fs.writeFileSync(output, JSON.stringify(profile, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return {profile, profilePath: output, profileSha256: fileHash(output), baseManifestSha256: base.baseManifestSha256,
    gnuResultSha256: gnu.resultSha256, copyrightsSha256: gnu.copyrightsSha256, emitterReceiptSha256: emitter.receiptSha256};
}

/**
 * Build the Windows-target GNU source emitter and prepare the existing resource overlay.
 * The caller must first restore DPC++/CUDA inputs from the externally pinned same-run bundle;
 * this function independently re-admits that root against its externally pinned manifest before
 * invoking the compiler. It never accepts a DPC++ path based only on a run-suffix match.
 * @param {{suffix:string,linuxResourcesRoot:string,linuxResultSha256:string,windowsGnuRoot:string,
 * windowsGnuResultSha256:string,windowsGnuCopyrightsSha256:string,dpcppRoot:string,
 * dpcppBundleSha256:string,dpcppManifestSha256:string,outputRoot:string}} input
 */
function produceWindowsJitResources(input) {
  if (process.platform !== "linux" || process.arch !== "x64" || process.env["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("Windows JIT resource production requires Linux x64 and build-only mode");
  }
  ciBuildJobs();
  if (!input || typeof input !== "object" || Array.isArray(input)) {throw new Error("Windows JIT resource inputs are required");}
  const suffix = input.suffix, paths = windowsJitResourcePaths(suffix), outputRoot = input.outputRoot;
  if (outputRoot !== paths.root || ciSuffix() !== suffix) {
    throw new Error("Windows JIT resource output or workflow suffix is not bound to this CI work root");
  }

  // The caller must verify the DPC++ archive pin before restoring it. Revalidate its external
  // manifest pin and all selected payloads here before the first compiler process is spawned.
  const ciSources = require("./native-ci-linux-sources"),
    dpcppInputs = require("./native-dpcpp-cuda-inputs"), pathsCi = ciSources.ciPaths(), dpcppState = ciSources.dpcppCudaInputsStatePaths(pathsCi, process.env),
    manifestSha256 = hash(input.dpcppManifestSha256, "DPC++ manifest");
  hash(input.dpcppBundleSha256, "DPC++ bundle");
  if (pathsCi.suffix !== suffix || input.dpcppRoot !== dpcppState.dpcppRoot) {
    throw new Error("Windows JIT emitter requires the externally pinned same-run DPC++ input handoff");
  }
  const dpcppRecord = readBuildJson(dpcppState.manifest);
  dpcppInputs.auditDpcppCudaInputs(dpcppState, dpcppRecord, manifestSha256);

  const linux = admitLinuxBase(input.linuxResourcesRoot, suffix, hash(input.linuxResultSha256, "Linux JIT result")),
    gnu = admitWindowsGnu(input.windowsGnuRoot, suffix, hash(input.windowsGnuResultSha256, "Windows GNU result"),
      hash(input.windowsGnuCopyrightsSha256, "Windows GNU copyrights")), targetHeadersPath = path.join(gnu.root, "target-headers.json"),
    targetHeaders = readBuildJson(targetHeadersPath);
  if (!Array.isArray(targetHeaders) || !targetHeaders.length || targetHeaders.length > MAX_RECEIPT_FILES ||
      targetHeaders.some((/** @type {unknown} */ file) => typeof file !== "string" || !path.isAbsolute(file) || !within(gnu.root, file))) {
    throw new Error("Windows JIT emitter requires the admitted Windows GNU target-header list");
  }
  if (fs.lstatSync(outputRoot, {throwIfNoEntry: false}) || [linux.root, gnu.root, dpcppState.root].some(root => within(root, outputRoot) || within(outputRoot, root))) {
    throw new Error("Windows JIT resource output must be fresh and outside all admitted input roots");
  }

  fs.mkdirSync(outputRoot, {mode: 0o700});
  const nativeTools = writeWindowsJitNativeTools(outputRoot, suffix, linux);
  const emitterReceipt = buildJitEmitter(dpcppState.dpcppRoot, paths.emitter, process.env, targetHeaders),
    composed = composeWindowsJitResourceProfile({suffix, linuxResourcesRoot: input.linuxResourcesRoot,
      linuxResultSha256: input.linuxResultSha256, windowsGnuRoot: input.windowsGnuRoot,
      windowsGnuResultSha256: input.windowsGnuResultSha256, windowsGnuCopyrightsSha256: input.windowsGnuCopyrightsSha256,
      emitterRoot: paths.emitter, emitterResultSha256: fileHash(path.join(paths.emitter, "result.json")),
      dpcppRoot: dpcppState.dpcppRoot, outputProfile: paths.profile}),
    prepared = prepareJitResources(composed.profile, paths.resources), verified = verifyJitResources(paths.resources, windowsTarget);

  // Close the build against post-compile input drift; the original producer receipts stay intact.
  const checkedLinux = admitLinuxBase(input.linuxResourcesRoot, suffix, input.linuxResultSha256),
    checkedNativeTools = admitWindowsJitNativeTools(outputRoot, suffix);
  admitWindowsGnu(input.windowsGnuRoot, suffix, input.windowsGnuResultSha256, input.windowsGnuCopyrightsSha256);
  dpcppInputs.auditDpcppCudaInputs(dpcppState, readBuildJson(dpcppState.manifest), manifestSha256);
  const checkedEmitter = admitEmitter(paths.emitter, dpcppState.dpcppRoot, gnu.root, gnu.headers, fileHash(path.join(paths.emitter, "result.json")));
  if (checkedEmitter.receiptSha256 !== fileHash(path.join(paths.emitter, "result.json")) ||
      checkedLinux.nativeTools.files.some((row, index) => row.sha256 !== nativeTools.hashes[index === 0 ? "clangTableGenSha256" : "llvmTableGenSha256"]) ||
      checkedNativeTools.manifestSha256 !== nativeTools.manifestSha256 ||
      !isDeepStrictEqual(prepared, readBuildJson(path.join(paths.resources, "manifest.json"))) || emitterReceipt.fixture !== checkedEmitter.fixture ||
      !verified.inputs[path.join(paths.resources, "manifest.json")]) {
    throw new Error("Windows JIT resource output changed during producer admission");
  }
  return {root: paths.root, emitter: paths.emitter, profile: paths.profile, resources: paths.resources,
    emitterReceiptSha256: checkedEmitter.receiptSha256, profileSha256: composed.profileSha256,
    manifestSha256: verified.manifestHash, files: readBuildJson(path.join(paths.resources, "manifest.json")).files.length,
    nativeTools: {root: checkedNativeTools.toolRoot, manifestFile: checkedNativeTools.manifestFile, manifestSha256: checkedNativeTools.manifestSha256,
      files: checkedNativeTools.files, hashes: checkedNativeTools.hashes, source: checkedNativeTools.source},
    buildOnly: true, executed: false, publicationQualified: false};
}

module.exports = {windowsJitResourcePaths, composeWindowsJitResourceProfile, produceWindowsJitResources,
  profileFromAdmissions, admitWindowsGnu, admitEmitter, verifyRequiredBaseManifest, validateWindowsJitNativeTools, admitWindowsJitNativeTools};

if (require.main === module) {
  try {
    const args = process.argv.slice(2);
    if (args[0] === "produce") {
      if (args.length !== 11) {throw new Error("Usage: native-windows-jit-resources.js produce <ci-suffix> <linux-root> <linux-result-sha256> <windows-gnu-root> <gnu-result-sha256> <copyrights-sha256> <dpcpp-root> <dpcpp-bundle-sha256> <dpcpp-manifest-sha256> <new-output-root>");}
      const [, suffix, linuxResourcesRoot, linuxResultSha256, windowsGnuRoot, windowsGnuResultSha256, windowsGnuCopyrightsSha256,
        dpcppRoot, dpcppBundleSha256, dpcppManifestSha256, outputRoot] = /** @type {[string,string,string,string,string,string,string,string,string,string,string]} */ (args);
      const result = produceWindowsJitResources({suffix, linuxResourcesRoot, linuxResultSha256, windowsGnuRoot, windowsGnuResultSha256,
        windowsGnuCopyrightsSha256, dpcppRoot, dpcppBundleSha256, dpcppManifestSha256, outputRoot});
      process.stdout.write(JSON.stringify({root: result.root, resources: result.resources, manifestSha256: result.manifestSha256,
        buildOnly: true, executed: false, publicationQualified: false}) + "\n");
    } else {
      throw new Error("Usage: native-windows-jit-resources.js produce ...");
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : "Windows JIT resource profile admission failed"); process.exitCode = 1;
  }
}
