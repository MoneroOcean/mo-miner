"use strict";

// Admit only the selected, build-only Windows controller-support handoffs. This is not a
// complete Windows SDK reader: OpenSSL's build CLI and Hermes' Windows AOT executable are
// intentionally absent and must be admitted separately by their consumers.
const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildJson, sourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {ioPins, ioRecipeFiles, nativeIoInputs} = require("./build-native-io");
const {opensslPin, opensslBuildRecipe} = require("./build-native-openssl");
const {hermesPins, hermesRecipeFiles} = require("./build-native-hermes");
const {ioSourceRecipe, sourceRecipe, ioArchiveUrls, buildArchiveUrls} = require("./fetch-native-sdk-sources");
const {isStaticArchive} = require("./native-audit");
const intelRuntimeCatalog = /** @type {any} */ (require("../runtime/windows-intel-runtime.json"));

const repo = path.resolve(__dirname, "..");
const MAX_FILES = 50000, MAX_ENTRIES = 100000, MAX_DEPTH = 64, MAX_BYTES = 512 << 20, HEX = /^[a-f0-9]{64}$/;
const hermesRecipeLeaves = hermesRecipeFiles("win32");

/** JSON records are validated at each trust boundary. @param {unknown} value @returns {boolean} */
function object(value) {return Boolean(value && typeof value === "object" && !Array.isArray(value));}
/** @param {unknown} value @returns {value is string} */
function hash(value) {return typeof value === "string" && HEX.test(value);}
/** @param {string} value */
function hasControl(value) {return Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);}
/** Windows producer paths are evidence only; never open them on the Linux consumer. @param {unknown} value */
function windowsPath(value) {
  if (typeof value !== "string" || value.length > 4096 || !path.win32.isAbsolute(value) || hasControl(value) || /[<>|?*"]/.test(value) ||
      value.startsWith("\\\\?\\") || value.startsWith("\\\\.\\")) {return false;}
  return path.win32.relative(path.win32.parse(value).root, value).split(/[\\/]/).every(part => !part ||
    part !== "." && part !== ".." && !/[. ]$/.test(part) && !/^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)/i.test(part));
}
/** @param {unknown} name @returns {name is string} */
function safeRelative(name) {
  return typeof name === "string" && name.length > 0 && name.length <= 4096 && !name.startsWith("/") && !name.includes("\\") &&
    !hasControl(name) && name.split("/").every(part => part && part !== "." && part !== ".." && !/[<>:"|?*]/.test(part) &&
      !/[. ]$/.test(part) && !/^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)/i.test(part));
}
/** Read the entire small handoff inventory with bounded hashing and reject path aliases.
 * @param {string} root @returns {{files:Record<string,{bytes:number,sha256:string}>,dirs:Set<string>,bytes:number}} */
function inventory(root) {
  const rootStat = fs.lstatSync(root);
  if (!path.isAbsolute(root) || !rootStat.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(root) !== root) {
    throw new Error("Windows controller handoff root must be a canonical regular directory");
  }
  const files = /** @type {Record<string,{bytes:number,sha256:string}>} */ (Object.create(null)), dirs = new Set(), folded = new Set();
  let entries = 0, count = 0, bytes = 0;
  /** @param {string} directory @param {string} prefix @param {number} depth */
  const visit = (directory, prefix, depth) => {
    const handle = fs.opendirSync(directory);
    try {
      let entry;
      while ((entry = handle.readSync()) !== null) {
        if (++entries > MAX_ENTRIES || depth >= MAX_DEPTH) {throw new Error("Windows controller handoff exceeds its entry or nesting bound");}
        const relative = prefix ? prefix + "/" + entry.name : entry.name, full = path.join(directory, entry.name), key = relative.toLowerCase();
        if (!safeRelative(relative) || folded.has(key) || entry.isSymbolicLink()) {throw new Error("Windows controller handoff has an unsafe or case-aliased path");}
        folded.add(key);
        const stat = fs.lstatSync(full);
        if (entry.isDirectory() && stat.isDirectory()) {dirs.add(relative); visit(full, relative, depth + 1); continue;}
        if (!entry.isFile() || !stat.isFile() || stat.nlink !== 1 || ++count > MAX_FILES || (bytes += stat.size) > MAX_BYTES ||
            fs.realpathSync(full) !== full) {throw new Error("Windows controller handoff exceeds regular-file bounds");}
        files[relative] = {bytes: stat.size, sha256: fileHash(full)};
      }
    } finally {handle.closeSync();}
  };
  visit(root, "", 0);
  return {files, dirs, bytes};
}
/** @param {string} root @param {string} relative @param {number} [maximum] @returns {any} */
function jsonAt(root, relative, maximum = 8 << 20) {
  if (!safeRelative(relative)) {throw new Error("Unsafe Windows controller handoff receipt path");}
  const file = path.join(root, ...relative.split("/")), stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > maximum) {throw new Error("Invalid Windows controller handoff receipt");}
  return readBuildJson(file);
}
/** @param {unknown} rows @param {string} label @returns {{expected:Record<string,{bytes:number,sha256:string}>,parents:Set<string>}} */
function exactInventory(rows, label) {
  if (!Array.isArray(rows) || !rows.length || rows.length > MAX_FILES) {throw new Error("Invalid " + label + " handoff inventory");}
  const expected = /** @type {Record<string,{bytes:number,sha256:string}>} */ (Object.create(null)), folded = new Set(), parents = new Set(); let bytes = 0, prior = "";
  for (const row of rows) {
    if (!object(row) || Object.keys(row).sort().join(",") !== "bytes,path,sha256" || !safeRelative(row.path) ||
        (prior && row.path <= prior) || !Number.isSafeInteger(row.bytes) || row.bytes < 0 || !hash(row.sha256) || Object.hasOwn(expected, row.path)) {
      throw new Error("Invalid " + label + " handoff inventory row");
    }
    const key = row.path.toLowerCase();
    if (folded.has(key)) {throw new Error("Case-aliased " + label + " handoff inventory");}
    folded.add(key);
    expected[row.path] = {bytes: row.bytes, sha256: row.sha256}; bytes += row.bytes;
    if (bytes > MAX_BYTES) {throw new Error(label + " handoff inventory exceeds its byte bound");}
    for (let slash = row.path.indexOf("/"); slash >= 0; slash = row.path.indexOf("/", slash + 1)) {parents.add(row.path.slice(0, slash));}
    prior = row.path;
  }
  return {expected, parents};
}
/** Compare every file and directory, then recheck after role-specific receipt validation.
 * @param {string} root @param {any} handoff @returns {Record<string,string>} */
function verifyTree(root, handoff) {
  const listed = exactInventory(handoff.files, "Windows controller"), actual = inventory(root), expected = {...listed.expected};
  const expectedDirs = new Set(listed.parents), actualFiles = {...actual.files}; delete actualFiles["handoff.json"];
  if (!isDeepStrictEqual(actualFiles, expected) || !isDeepStrictEqual([...actual.dirs].sort(), [...expectedDirs].sort()) ||
      actual.bytes > MAX_BYTES) {throw new Error("Windows controller handoff tree differs from its pinned inventory");}
  return Object.fromEntries(Object.entries(actual.files).map(([relative, row]) => [path.join(root, ...relative.split("/")), row.sha256]));
}
/** @param {string} root @param {string} relative @param {string} expected */
function receiptHash(root, relative, expected) {
  const file = path.join(root, ...relative.split("/"));
  if (!hash(expected) || fileHash(file) !== expected) {throw new Error("Windows controller handoff receipt hash differs: " + relative);}
}
/** @param {any} receipt @param {ReturnType<typeof sourceInputs>} recipe @param {string} label */
function currentRecipe(receipt, recipe, label) {verifySourceInputs(receipt, recipe, label);}
/** @param {any} value @param {string[]} keys */
function hasKeys(value, keys) {return object(value) && isDeepStrictEqual(Object.keys(value).sort(), [...keys].sort());}
/** @returns {{compiler:{version:string,sha256:string},libraries:Array<{name:string,path:string,bytes:number,sha256:string}>,licenses:Array<{path:string,bytes:number,sha256:string}>}} */
function readIntelRuntimePins() {
  if (!hasKeys(intelRuntimeCatalog, ["version", "compiler", "libraries", "licenses"]) || intelRuntimeCatalog.version !== 1 ||
      !hasKeys(intelRuntimeCatalog.compiler, ["version", "sha256"]) || typeof intelRuntimeCatalog.compiler.version !== "string" ||
      !intelRuntimeCatalog.compiler.version || intelRuntimeCatalog.compiler.version.length > 32 || !hash(intelRuntimeCatalog.compiler.sha256) ||
      !Array.isArray(intelRuntimeCatalog.libraries) || !intelRuntimeCatalog.libraries.length || intelRuntimeCatalog.libraries.length > 64 ||
      !Array.isArray(intelRuntimeCatalog.licenses) || !intelRuntimeCatalog.licenses.length || intelRuntimeCatalog.licenses.length > 64) {
    throw new Error("Invalid pinned Windows Intel runtime catalog");
  }
  let bytes = 0;
  const paths = new Set();
  const libraries = intelRuntimeCatalog.libraries.map((/** @type {any} */ row) => {
    if (!hasKeys(row, ["name", "source", "path", "bytes", "sha256"]) || typeof row.name !== "string" ||
        !/^[A-Za-z0-9_.+-]+\.lib$/i.test(row.name) || row.source !== "lib/" + row.name ||
        row.path !== "intel-runtime/lib/" + row.name || !safeRelative(row.source) || !safeRelative(row.path) ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > MAX_BYTES || !hash(row.sha256) || paths.has(row.path.toLowerCase())) {
      throw new Error("Invalid pinned Windows Intel runtime library row");
    }
    paths.add(row.path.toLowerCase()); bytes += row.bytes;
    if (bytes > MAX_BYTES) {throw new Error("Pinned Windows Intel runtime catalog exceeds its byte bound");}
    return {name: row.name, path: row.path, bytes: row.bytes, sha256: row.sha256};
  });
  const licenses = intelRuntimeCatalog.licenses.map((/** @type {any} */ row) => {
    if (!hasKeys(row, ["source", "path", "bytes", "sha256"]) || !safeRelative(row.source) ||
        row.path !== "intel-runtime/licenses/" + row.source || !safeRelative(row.path) || !Number.isSafeInteger(row.bytes) ||
        row.bytes < 1 || row.bytes > MAX_BYTES || !hash(row.sha256) || paths.has(row.path.toLowerCase())) {
      throw new Error("Invalid pinned Windows Intel runtime license row");
    }
    paths.add(row.path.toLowerCase()); bytes += row.bytes;
    if (bytes > MAX_BYTES) {throw new Error("Pinned Windows Intel runtime catalog exceeds its byte bound");}
    return {path: row.path, bytes: row.bytes, sha256: row.sha256};
  });
  return {compiler: intelRuntimeCatalog.compiler, libraries, licenses};
}
const intelRuntimePins = readIntelRuntimePins();
/** @param {any[]} actual @param {string[]} expected @param {string} label */
function exactPaths(actual, expected, label) {
  const paths = actual.map(row => row.path).sort(), wanted = [...expected].sort();
  if (!isDeepStrictEqual(paths, wanted)) {throw new Error("Unexpected " + label + " handoff members");}
}
/** @param {any} row @param {string} target @param {ReturnType<typeof sourceInputs>} recipe
 * @param {Record<string,{fileName:string,url:string,sha256:string,maxBytes:number}>} expectedArchives
 * @returns {Record<string,{path:string,url:string,sha256:string,bytes:number}>} */
function fetchReceipt(row, target, recipe, expectedArchives) {
  // The fetcher digest records acquisition history; current recipes and archive pins admit the receipt.
  if (!object(row) || row.version !== 1 || row.target !== target || row.status !== "PASS" || row.networked !== true || row.buildOnly !== true ||
      row.compiled !== false || row.publicationQualified !== false || !hash(row.fetcherSha256) ||
      !hasKeys(row.recipe, ["sha256", "files"]) || row.recipe.sha256 !== createHash("sha256").update(JSON.stringify(recipe)).digest("hex") ||
      !isDeepStrictEqual(row.recipe.files, recipe.graph) || !hasKeys(row.archives, Object.keys(expectedArchives))) {
    throw new Error("Windows controller source receipt is stale or incomplete: " + target);
  }
  for (const [name, expected] of Object.entries(expectedArchives)) {
    const item = row.archives[name];
    if (!object(item) || !windowsPath(item.path) || path.win32.basename(item.path).toLowerCase() !== expected.fileName.toLowerCase() ||
        item.url !== expected.url || item.sha256 !== expected.sha256 || !Number.isSafeInteger(item.bytes) || item.bytes < 1 || item.bytes > expected.maxBytes) {
      throw new Error("Windows controller source receipt pin differs: " + target + "/" + name);
    }
  }
  return row.archives;
}

/** @param {string} root @param {any} handoff @returns {{libraries:{sourceInclude:string,ssl:string,crypto:string,libuv:string,zlib:string,libuvInclude:string,zlibInclude:string,zlibGeneratedInclude:string}}} */
function ioInputs(root, handoff) {
  const ioReceipt = jsonAt(root, "io-sdk/result.json"), ioFetch = jsonAt(root, "provenance/io-fetch.json"),
    ssl = jsonAt(root, "openssl-sdk/result.json"), sslFetch = jsonAt(root, "provenance/openssl-fetch.json"), tools = jsonAt(root, "openssl-sdk/producer-tools.json"),
    ioRecipe = ioSourceRecipe(), opensslRecipe = opensslBuildRecipe(), archives = fetchReceipt(ioFetch, "io", ioRecipe, {
      libuv: {fileName: ioPins.libuv.directory + ".tar.gz", url: ioArchiveUrls.libuv, sha256: ioPins.libuv.sha256, maxBytes: 128 << 20},
      zlib: {fileName: ioPins.zlib.directory + ".tar.gz", url: ioArchiveUrls.zlib, sha256: ioPins.zlib.sha256, maxBytes: 128 << 20},
    }), sslArchives = fetchReceipt(sslFetch, "windows-openssl", opensslRecipe, {
      openssl: {fileName: "openssl-" + opensslPin.version + ".tar.gz", url: buildArchiveUrls.openssl, sha256: opensslPin.sha256, maxBytes: 128 << 20},
    });
  const libuvArchive = archives["libuv"], zlibArchive = archives["zlib"], opensslArchive = sslArchives["openssl"];
  if (!libuvArchive || !zlibArchive || !opensslArchive) {throw new Error("Windows controller source archive cohort is incomplete");}
  receiptHash(root, "provenance/io-fetch.json", handoff.sourceReceipts[0].sha256);
  receiptHash(root, "provenance/openssl-fetch.json", handoff.sourceReceipts[1].sha256);
  receiptHash(root, "io-sdk/result.json", handoff.producers[0].receiptSha256);
  receiptHash(root, "openssl-sdk/result.json", handoff.producers[1].receiptSha256);
  const ioRoot = path.join(root, "io-sdk"), io = nativeIoInputs(ioRoot, "win32"), ioRecord = ioReceipt;
  if (ioRecord.publicationQualified !== false || !hasKeys(ioRecord.inputs, [libuvArchive.path, zlibArchive.path]) ||
      ioRecord.inputs[libuvArchive.path] !== ioPins.libuv.sha256 || ioRecord.inputs[zlibArchive.path] !== ioPins.zlib.sha256 ||
      !object(ioRecord.fingerprints) || Object.keys(ioRecord.fingerprints).length !== 3 || Object.entries(ioRecord.fingerprints).some(([file, sha]) => !windowsPath(file) || !hash(sha))) {
    throw new Error("Windows controller I/O receipt does not bind the current original inputs and tool fingerprints");
  }
  if (new Set(Object.keys(ioRecord.fingerprints).map(file => path.win32.normalize(file).toLowerCase())).size !== 3) {
    throw new Error("Windows controller I/O tool fingerprints contain a case alias");
  }
  currentRecipe(ioRecord.recipe, sourceInputs(repo, [], ioRecipeFiles), "Windows I/O SDK producer");

  const opensslRecipeCurrent = opensslBuildRecipe();
  if (ssl.version !== 1 || ssl.platform !== "win32" || ssl.target !== "VC-WIN64A" || ssl.opensslVersion !== opensslPin.version ||
      ssl.sourceDateEpoch !== opensslPin.sourceDateEpoch ||
      ssl.archive?.sha256 !== opensslPin.sha256 || ssl.archive?.path !== opensslArchive.path || ssl.archive.bytes !== opensslArchive.bytes ||
      ssl.buildOnly !== true || ssl.executed !== false || ssl.publicationQualified !== false || !hash(ssl.config?.sha256) ||
      !object(ssl.source) || !Number.isSafeInteger(ssl.source.files) || ssl.source.files < 1 || !Number.isSafeInteger(ssl.source.bytes) || ssl.source.bytes < 1 ||
      ssl.source.files > 100000 || ssl.source.bytes > (1 << 30) || !hash(ssl.source.sha256) || ssl.source.notice?.path !== "source/LICENSE.txt" || !hash(ssl.source.notice.sha256) ||
      !windowsPath(ssl.archive.path) || ssl.archive.bytes > (128 << 20) || !Number.isSafeInteger(ssl.archive.members) || ssl.archive.members < 1 || ssl.archive.members > 100000) {
    throw new Error("Invalid partial Windows OpenSSL producer receipt");
  }
  currentRecipe(ssl.recipe, opensslRecipeCurrent, "Windows OpenSSL producer");
  const toolNames = ["ar", "cc", "link", "make", "perl", "rc", "tar"], configPath = path.join(root, "openssl-sdk/producer-tools.json");
  if (!hasKeys(tools, ["version", "jobs", "tools"]) || tools.version !== 1 || ![2, 4].includes(tools.jobs) || !hasKeys(tools.tools, toolNames) ||
      fileHash(configPath) !== ssl.config.sha256 || !hasKeys(ssl.tools, toolNames) || toolNames.some(name =>
    !windowsPath(tools.tools[name]) || ssl.tools[name]?.path !== tools.tools[name] || !hash(ssl.tools[name]?.sha256))) {
    throw new Error("Windows OpenSSL tool receipt is not bound to its unchanged build descriptor");
  }
  const artifacts = ssl.artifacts, selected = Object.create(null);
  if (!object(artifacts) || Object.keys(artifacts).length > 100000) {throw new Error("Invalid partial Windows OpenSSL artifact receipt");}
  for (const [name, sha] of Object.entries(artifacts)) {
    if (!hash(sha) || !(name.startsWith("include/openssl/") || ["lib/libssl.lib", "lib/libcrypto.lib", "bin/openssl.exe"].includes(name))) {
      throw new Error("Unexpected Windows OpenSSL producer artifact: " + name);
    }
    if (name.startsWith("include/openssl/")) {selected["openssl-sdk/" + name] = sha;}
  }
  for (const name of ["lib/libssl.lib", "lib/libcrypto.lib"]) {selected["openssl-sdk/" + name] = artifacts[name];}
  if (!artifacts["bin/openssl.exe"] || !artifacts["lib/libssl.lib"] || !artifacts["lib/libcrypto.lib"] ||
      !selected["openssl-sdk/include/openssl/opensslv.h"]) {throw new Error("Windows OpenSSL handoff omits selected headers or libraries");}
  const handoffRows = Object.fromEntries(handoff.files.map((/** @type {{path:string,bytes:number,sha256:string}} */ row) => [row.path, row.sha256]));
  for (const [relative, sha] of Object.entries(selected)) {
    if (handoffRows[relative] !== sha) {throw new Error("Windows OpenSSL selected file differs from its producer receipt: " + relative);}
  }
  exactPaths(handoff.files, ["io-sdk/result.json", ...Object.keys(ioRecord.artifacts).map(name => "io-sdk/" + name),
    "openssl-sdk/result.json", "openssl-sdk/producer-tools.json", "openssl-sdk/source/LICENSE.txt", ...Object.keys(selected),
    "provenance/io-fetch.json", "provenance/openssl-fetch.json"], "I/O/OpenSSL");
  const license = path.join(root, "openssl-sdk/source/LICENSE.txt"), licenseHash = fileHash(license), versionHeader = path.join(root, "openssl-sdk/include/openssl/opensslv.h");
  if (fs.statSync(versionHeader).size > (1 << 20) || licenseHash !== ssl.source.notice.sha256 || !fs.readFileSync(versionHeader, "utf8").includes('OPENSSL_VERSION_STR "' + opensslPin.version + '"') ||
      !isStaticArchive(path.join(root, "openssl-sdk/lib/libssl.lib")) || !isStaticArchive(path.join(root, "openssl-sdk/lib/libcrypto.lib"))) {
    throw new Error("Windows OpenSSL partial static/header closure differs from its pinned producer");
  }
  return {libraries: {sourceInclude: path.join(root, "openssl-sdk/include"), ssl: path.join(root, "openssl-sdk/lib/libssl.lib"),
    crypto: path.join(root, "openssl-sdk/lib/libcrypto.lib"), libuv: path.join(ioRoot, "lib/uv_a.lib"), zlib: path.join(ioRoot, "lib/zlibstatic.lib"),
    libuvInclude: io.include, zlibInclude: io.include, zlibGeneratedInclude: io.include}};
}

/** @typedef {{libRoot:string,libraries:Record<string,string>,licenses:Record<string,string>}} IntelRuntimeInputs */
/** Validate the optional Intel runtime manifest. Actual bytes are admitted by verifyTree below.
 * @param {any} handoff @param {string} cc @param {Record<string,string>} fingerprints
 * @returns {{compiler:{path:string,sha256:string},libraries:typeof intelRuntimePins.libraries,licenses:typeof intelRuntimePins.licenses} | undefined} */
function validateIntelRuntimeMetadata(handoff, cc, fingerprints) {
  if (!object(handoff)) {throw new Error("Invalid Windows controller handoff metadata");}
  if (!Object.hasOwn(handoff, "intelRuntime")) {return undefined;}
  if (!windowsPath(cc) || !object(fingerprints) || !Array.isArray(handoff.files)) {
    throw new Error("Invalid optional Windows Intel runtime compiler fingerprints");
  }
  const runtime = handoff.intelRuntime, compilerKey = Object.keys(fingerprints).find(file =>
      path.win32.normalize(file).toLowerCase() === path.win32.normalize(cc).toLowerCase()),
    expectedFiles = [...intelRuntimePins.libraries, ...intelRuntimePins.licenses];
  if (!hasKeys(runtime, ["compiler", "libraries", "licenses"]) ||
      !hasKeys(runtime.compiler, ["path", "sha256"]) ||
      !windowsPath(runtime.compiler.path) || path.win32.normalize(runtime.compiler.path).toLowerCase() !== path.win32.normalize(cc).toLowerCase() ||
      !compilerKey || !hash(fingerprints[compilerKey]) || runtime.compiler.sha256 !== fingerprints[compilerKey] ||
      runtime.compiler.sha256 !== intelRuntimePins.compiler.sha256 || !Array.isArray(runtime.libraries) ||
      !isDeepStrictEqual(runtime.libraries, intelRuntimePins.libraries) || !Array.isArray(runtime.licenses) ||
      !isDeepStrictEqual(runtime.licenses, intelRuntimePins.licenses)) {
    throw new Error("Invalid optional Windows Intel runtime compiler or payload pins");
  }
  for (const expected of expectedFiles) {
    const row = handoff.files.find((/** @type {any} */ file) => object(file) && file.path === expected.path);
    if (!row || row.bytes !== expected.bytes || row.sha256 !== expected.sha256) {
      throw new Error("Windows Intel runtime payload differs from its pinned handoff: " + expected.path);
    }
  }
  return {compiler: runtime.compiler, libraries: intelRuntimePins.libraries, licenses: intelRuntimePins.licenses};
}

/** @param {string} root @param {any} handoff @returns {{hermes:{source:string,build:string,license?:string},intelRuntime?:IntelRuntimeInputs}} */
function hermesInputs(root, handoff) {
  const result = jsonAt(root, "hermes/result.json"), before = jsonAt(root, "hermes/before.json"), fetch = jsonAt(root, "provenance/hermes-fetch.json"),
    roots = jsonAt(root, "provenance/hermes-roots.json"),
    recipe = sourceInputs(repo, [], hermesRecipeLeaves),
    fetchRecipe = sourceRecipe("hermes");
  receiptHash(root, "hermes/result.json", handoff.producers[0].receiptSha256);
  receiptHash(root, "hermes/before.json", handoff.producers[0].planSha256);
  receiptHash(root, "provenance/hermes-fetch.json", handoff.sourceReceipts[0].sha256);
  receiptHash(root, "provenance/hermes-roots.json", handoff.sourceReceipts[0].rootsSha256);
  if (result.version !== 1 || result.target !== "win32" || !isDeepStrictEqual(result.pins, hermesPins) || result.buildOnly !== true ||
      result.publicationQualified !== false || !windowsPath(result.source) || !windowsPath(result.build) || !object(result.artifacts) ||
      path.win32.basename(result.source).toLowerCase() !== "source" || path.win32.basename(result.build).toLowerCase() !== "build" ||
      path.win32.dirname(result.source).toLowerCase() !== path.win32.dirname(result.build).toLowerCase() ||
      !object(result.sources) || Object.keys(result.sources).length > 100000 || !object(result.fingerprints) ||
      !hasKeys(before, ["pins", "recipe", "fingerprints", "sources", "configure", "compile"]) ||
      !isDeepStrictEqual(before.pins, hermesPins) || !isDeepStrictEqual(before.recipe, result.recipe) ||
      !isDeepStrictEqual(before.fingerprints, result.fingerprints) || !isDeepStrictEqual(before.sources, result.sources) ||
      !Array.isArray(before.configure) || before.configure.length < 1 || before.configure.length > 128 ||
      before.configure.some((/** @type {unknown} */ arg) => typeof arg !== "string" || arg.length > 4096 || /[\r\n\0]/.test(arg)) ||
      !Array.isArray(before.compile) || before.compile.length < 1 || before.compile.length > 128 ||
      before.compile.some((/** @type {unknown} */ arg) => typeof arg !== "string" || arg.length > 4096 || /[\r\n\0]/.test(arg))) {throw new Error("Invalid same-run Windows Hermes producer receipt");}
  currentRecipe(result.recipe, recipe, "Windows Hermes producer");
  const expectedPatchHashes = Object.fromEntries(hermesRecipeLeaves.filter(name => name.endsWith(".patch"))
    .map(name => [name, fileHash(path.join(repo, name))]));
  // The fetcher hash records acquisition history; offline admission separately pins the current fetch and compiled Hermes recipes.
  if (fetch.version !== 1 || fetch.target !== "hermes" || fetch.status !== "PASS" || fetch.networked !== true || fetch.buildOnly !== true ||
      fetch.compiled !== false || fetch.publicationQualified !== false || !hash(fetch.fetcherSha256) ||
      !hasKeys(fetch.recipe, ["sha256", "files"]) || fetch.recipe.sha256 !== createHash("sha256").update(JSON.stringify(fetchRecipe)).digest("hex") ||
      !isDeepStrictEqual(fetch.recipe.files, fetchRecipe.graph) || !hasKeys(fetch.components, ["hermes"]) ||
      fetch.components.hermes?.origin !== "https://github.com/facebook/hermes.git" || fetch.components.hermes?.revision !== hermesPins.revision ||
      fetch.components.hermes?.gitDirectory !== "bare" || fetch.components.hermes?.shallow !== true || fetch.components.hermes?.partialClone !== false ||
      !object(fetch.git) || typeof fetch.git.path !== "string" || !path.posix.isAbsolute(fetch.git.path) || !hash(fetch.git.sha256) ||
      !hasKeys(roots, ["hermes"]) || roots.hermes !== fetch.components.hermes.root || typeof roots.hermes !== "string" ||
      !path.posix.isAbsolute(roots.hermes) || roots.hermes.split("/").some((/** @type {string} */ part) => part === "." || part === "..")) {
    throw new Error("Windows Hermes source receipt does not bind its pinned original source");
  }
  // One driver may serve C and C++; the role and alias checks below remain exact.
  const artifactKeys = ["compiler", "context", "jsi", "public", "runtime"];
  if (!hasKeys(result.artifacts, artifactKeys) || Object.entries(result.sources).some(([name, sha]) => !safeRelative(name) || !hash(sha)) ||
      Object.keys(result.fingerprints).length < 6 || Object.keys(result.fingerprints).length > 7 ||
      Object.entries(result.fingerprints).some(([file, sha]) => !windowsPath(file) || !hash(sha))) {
    throw new Error("Invalid Windows Hermes source, artifact or tool-fingerprint inventory");
  }
  const fingerprintPaths = new Set(Object.keys(result.fingerprints).map(file => path.win32.normalize(file).toLowerCase()));
  if (fingerprintPaths.size !== Object.keys(result.fingerprints).length) {throw new Error("Windows Hermes tool fingerprints contain a case alias");}
  const patches = ["runtime/hermes-host-rejections.patch", "runtime/hermes-clang-cl.patch"];
  for (const relative of patches) {
    const suffix = "\\" + relative.replaceAll("/", "\\"), matches = Object.entries(result.fingerprints).filter(([file]) =>
      path.win32.normalize(file).toLowerCase().endsWith(suffix.toLowerCase()));
    if (matches.length !== 1 || matches[0]?.[1] !== expectedPatchHashes[relative]) {throw new Error("Windows Hermes patch fingerprint is stale: " + relative);}
  }
  const cmake = before.configure[0], cc = before.configure.find((/** @type {string} */ arg) => arg.startsWith("-DCMAKE_C_COMPILER:FILEPATH="))?.slice(28),
    cxx = before.configure.find((/** @type {string} */ arg) => arg.startsWith("-DCMAKE_CXX_COMPILER:FILEPATH="))?.slice(30),
    ninja = before.configure.find((/** @type {string} */ arg) => arg.startsWith("-DCMAKE_MAKE_PROGRAM:FILEPATH="))?.slice(30), knownTools = [cmake, cc, cxx, ninja];
  if (before.compile[0] !== cmake || knownTools.some(file => !windowsPath(file) || !fingerprintPaths.has(path.win32.normalize(file).toLowerCase())) ||
      fingerprintPaths.size !== new Set(knownTools.map(file => path.win32.normalize(file).toLowerCase())).size + 3 ||
      Object.keys(result.fingerprints).filter(file => !knownTools.some(tool => typeof tool === "string" &&
        path.win32.normalize(tool).toLowerCase() === path.win32.normalize(file).toLowerCase()) &&
        !patches.some(patch => path.win32.normalize(file).toLowerCase().endsWith("\\" + patch.replaceAll("/", "\\").toLowerCase()))).length !== 1) {
    throw new Error("Windows Hermes tool fingerprints do not match the recorded producer commands");
  }
  const tails = /** @type {Record<string,string>} */ ({compiler: "bin/shermes.exe", context: "external/boost/boost_1_86_0/libs/context/boost_context.lib",
    jsi: "jsi/jsi.lib", public: "public/hermes/Public/hermesPublic.lib", runtime: "lib/hermesvmlean_a.lib"});
  for (const name of artifactKeys) {
    const item = result.artifacts[name];
    const tail = tails[name];
    if (!tail || !object(item) || !windowsPath(item.path) || path.win32.normalize(item.path).toLowerCase() !==
        path.win32.normalize(path.win32.join(result.build, ...tail.split("/"))).toLowerCase() || !hash(item.sha256) || !Number.isSafeInteger(item.bytes) || item.bytes < 1) {
      throw new Error("Invalid Windows Hermes artifact metadata: " + name);
    }
  }
  const selectedArtifacts = /** @type {Array<[string,string]>} */ ([
    ["runtime", "hermes/build/lib/hermesvmlean_a.lib"], ["jsi", "hermes/build/jsi/jsi.lib"],
    ["context", "hermes/build/external/boost/boost_1_86_0/libs/context/boost_context.lib"],
  ]);
  for (const [name, relative] of selectedArtifacts) {
    const item = result.artifacts[name], row = handoff.files.find((/** @type {{path:string,bytes:number,sha256:string}} */ file) => file.path === relative), target = path.join(root, ...relative.split("/"));
    if (!row || row.sha256 !== item.sha256 || row.bytes !== item.bytes || path.win32.basename(item.path).toLowerCase() !== path.posix.basename(relative).toLowerCase() || !isStaticArchive(target)) {
      throw new Error("Windows Hermes selected archive differs from its producer receipt: " + name);
    }
  }
  const selectedSources = handoff.files.filter((/** @type {{path:string,bytes:number,sha256:string}} */ row) => row.path.startsWith("hermes/source/")),
    licenseRow = handoff.files.find((/** @type {{path:string,bytes:number,sha256:string}} */ row) => row.path === "hermes/source/LICENSE");
  if (licenseRow && (!hash(result.sources.LICENSE) || licenseRow.sha256 !== result.sources.LICENSE)) {
    throw new Error("Windows Hermes root LICENSE differs from the same-run source inventory");
  }
  const expectedSources = Object.entries(result.sources).filter(([name]) => name.startsWith("include/") || name.startsWith("API/") ||
      name.startsWith("public/") || Boolean(licenseRow) && name === "LICENSE"),
    actualSources = Object.fromEntries(selectedSources.map((/** @type {{path:string,bytes:number,sha256:string}} */ row) => [row.path.slice("hermes/source/".length), row.sha256])),
    intelRuntimeMetadata = validateIntelRuntimeMetadata(handoff, /** @type {string} */ (cc), result.fingerprints),
    intelRuntime = intelRuntimeMetadata && {libRoot: path.join(root, "intel-runtime/lib"),
      libraries: Object.fromEntries(intelRuntimeMetadata.libraries.map(row => [row.name, path.join(root, ...row.path.split("/"))])),
      licenses: Object.fromEntries(intelRuntimeMetadata.licenses.map(row => [row.path, path.join(root, ...row.path.split("/"))]))};
  if (intelRuntimeMetadata && intelRuntimeMetadata.libraries.some(row => !isStaticArchive(path.join(root, ...row.path.split("/"))))) {
    throw new Error("Windows Intel runtime library is not a regular static archive");
  }
  if (!selectedSources.length || !isDeepStrictEqual(actualSources, Object.fromEntries(expectedSources)) ||
      !handoff.files.some((/** @type {{path:string,bytes:number,sha256:string}} */ row) => row.path.startsWith("hermes/build/lib/config/"))) {
    throw new Error("Windows Hermes selected source/config files differ from the same-run producer closure");
  }
  exactPaths(handoff.files, ["hermes/result.json", "hermes/before.json", "provenance/hermes-fetch.json", "provenance/hermes-roots.json",
    ...selectedSources.map((/** @type {{path:string,bytes:number,sha256:string}} */ row) => row.path),
    ...handoff.files.filter((/** @type {{path:string,bytes:number,sha256:string}} */ row) => row.path.startsWith("hermes/build/lib/config/"))
      .map((/** @type {{path:string,bytes:number,sha256:string}} */ row) => row.path),
    "hermes/build/lib/hermesvmlean_a.lib", "hermes/build/jsi/jsi.lib",
    "hermes/build/external/boost/boost_1_86_0/libs/context/boost_context.lib",
    ...(intelRuntimeMetadata ? [...intelRuntimeMetadata.libraries, ...intelRuntimeMetadata.licenses].map(row => row.path) : [])], "Hermes");
  return {hermes: {source: path.join(root, "hermes/source"), build: path.join(root, "hermes/build"),
    ...(licenseRow ? {license: path.join(root, "hermes/source/LICENSE")} : {})}, ...(intelRuntime ? {intelRuntime} : {})};
}

/** @typedef {{kind:"io",handoffSha256:string} | {kind:"hermes",handoffSha256:string}} HandoffExpectation
 * @typedef {{kind:"io",handoff:string,handoffSha256:string,files:Record<string,string>,libraries:ReturnType<typeof ioInputs>["libraries"]} |
 * {kind:"hermes",handoff:string,handoffSha256:string,files:Record<string,string>,hermes:ReturnType<typeof hermesInputs>["hermes"],intelRuntime?:IntelRuntimeInputs}} HandoffInputs
 * @typedef {HandoffInputs} SupportInputs
 * @typedef {{kind:"io"|"hermes",localBuildId:string,handoffSha256:string}} LocalHandoffExpectation
 * Admit a regular, exact, externally hash-pinned Windows controller-support handoff.
 * @param {string} root @param {HandoffExpectation} expected @returns {HandoffInputs} */
function windowsControllerHandoffInputs(root, expected) {
  if (!object(expected) || !hasKeys(expected, ["kind", "handoffSha256"]) || !["io", "hermes"].includes(expected.kind) ||
      !hash(expected.handoffSha256) || typeof root !== "string" || !path.isAbsolute(root)) {
    throw new Error("Windows controller handoff needs its exact role and producer hash");
  }
  return admitWindowsControllerHandoff(root, expected, {});
}

/** Local transport carries the local build identity; CI handoffs are bound by their producer hash alone.
 * Original Windows receipts and all content/source/tool checks are shared unchanged.
 * @param {string} root @param {LocalHandoffExpectation} expected
 * @returns {SupportInputs & {localBuildId:string}} */
function windowsLocalControllerHandoffInputs(root, expected) {
  if (!object(expected) || !hasKeys(expected, ["kind", "localBuildId", "handoffSha256"]) || !["io", "hermes"].includes(expected.kind) ||
      !/^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/.test(expected.localBuildId) ||
      !hash(expected.handoffSha256) || typeof root !== "string" || !path.isAbsolute(root)) {
    throw new Error("Local Windows controller handoff needs its exact role, local build identity and producer hash");
  }
  return {...admitWindowsControllerHandoff(root, expected, {localBuildId: expected.localBuildId}), localBuildId: expected.localBuildId};
}

/** Both transport entrypoints admit the same selected closure; identity changes no source policy.
 * Windows producer paths remain evidence only and are never rewritten for WSL.
 * @param {string} root @param {{kind:"io"|"hermes",handoffSha256:string}} expected
 * @param {{localBuildId?:string}} identity @returns {SupportInputs} */
function admitWindowsControllerHandoff(root, expected, identity) {
  root = path.resolve(root);
  const handoffFile = path.join(root, "handoff.json"), stat = fs.lstatSync(handoffFile);
  if (!stat.isFile() || stat.isSymbolicLink() || fileHash(handoffFile) !== expected.handoffSha256) {throw new Error("Windows controller handoff producer hash differs");}
  const handoff = readBuildJson(handoffFile), commonKeys = ["version", "format", ...Object.keys(identity), "target", "architecture", "kind", "buildOnly",
      "executed", "publicationQualified", "scope", "producers", "sourceReceipts", "omitted", "files"],
    handoffKeys = expected.kind === "hermes" && object(handoff) && Object.hasOwn(handoff, "intelRuntime") ? [...commonKeys, "intelRuntime"] : commonKeys;
  if (!hasKeys(handoff, handoffKeys) || handoff.version !== 1 || handoff.format !== "mom-windows-controller-support-handoff-v1" ||
      Object.entries(identity).some(([key, value]) => handoff[key] !== value) || handoff.kind !== expected.kind || handoff.target !== "win32" || handoff.architecture !== "x64" ||
      handoff.buildOnly !== true || handoff.executed !== false || handoff.publicationQualified !== false || typeof handoff.scope !== "string" ||
      !handoff.scope || handoff.scope.length > 2048 || !Array.isArray(handoff.producers) || !Array.isArray(handoff.sourceReceipts) || !Array.isArray(handoff.omitted)) {
    throw new Error("Windows controller handoff role or build-only boundary differs");
  }
  if (expected.kind === "io") {
    if (handoff.producers.length !== 2 || !hasKeys(handoff.producers[0], ["script", "receipt", "receiptSha256"]) ||
        !hasKeys(handoff.producers[1], ["script", "receipt", "receiptSha256"]) || handoff.producers[0].script !== "scripts/build-native-io.js" ||
        handoff.producers[0].receipt !== "io-sdk/result.json" || !hash(handoff.producers[0].receiptSha256) ||
        handoff.producers[1].script !== "scripts/build-native-openssl.js" || handoff.producers[1].receipt !== "openssl-sdk/result.json" ||
        !hash(handoff.producers[1].receiptSha256) || handoff.sourceReceipts.length !== 2 ||
        !hasKeys(handoff.sourceReceipts[0], ["file", "sha256", "archives"]) || !hasKeys(handoff.sourceReceipts[1], ["file", "sha256", "archives"]) ||
        handoff.sourceReceipts[0].file !== "provenance/io-fetch.json" || handoff.sourceReceipts[1].file !== "provenance/openssl-fetch.json" ||
        !hash(handoff.sourceReceipts[0].sha256) || !hash(handoff.sourceReceipts[1].sha256) ||
        !isDeepStrictEqual(handoff.sourceReceipts[0].archives, {libuv: ioPins.libuv.sha256, zlib: ioPins.zlib.sha256}) ||
        !isDeepStrictEqual(handoff.sourceReceipts[1].archives, {openssl: opensslPin.sha256}) ||
        !isDeepStrictEqual(handoff.omitted, ["OpenSSL source/build trees", "openssl.exe", "compiler/build tools", "DLLs"])) {
      throw new Error("Windows controller I/O/OpenSSL handoff producer or source receipts differ");
    }
  } else if (handoff.producers.length !== 1 || !hasKeys(handoff.producers[0], ["script", "receipt", "receiptSha256", "plan", "planSha256"]) ||
      handoff.producers[0].script !== "scripts/build-native-hermes.js" || handoff.producers[0].receipt !== "hermes/result.json" ||
      !hash(handoff.producers[0].receiptSha256) || handoff.producers[0].plan !== "hermes/before.json" || !hash(handoff.producers[0].planSha256) ||
      handoff.sourceReceipts.length !== 1 || !hasKeys(handoff.sourceReceipts[0], ["file", "sha256", "rootsFile", "rootsSha256", "revision"]) ||
      handoff.sourceReceipts[0].file !== "provenance/hermes-fetch.json" || !hash(handoff.sourceReceipts[0].sha256) ||
      handoff.sourceReceipts[0].rootsFile !== "provenance/hermes-roots.json" || !hash(handoff.sourceReceipts[0].rootsSha256) ||
      handoff.sourceReceipts[0].revision !== hermesPins.revision ||
      !isDeepStrictEqual(handoff.omitted, ["Hermes Git/build trees", "shermes.exe", "hermesPublic.lib", "compiler/build tools", "DLLs"])) {
    throw new Error("Windows controller Hermes handoff producer or source receipts differ");
  }
  const files = verifyTree(root, handoff), role = /** @type {any} */ (expected.kind === "io" ? ioInputs(root, handoff) : hermesInputs(root, handoff));
  const after = inventory(root);
  if (fileHash(handoffFile) !== expected.handoffSha256 || !isDeepStrictEqual(Object.fromEntries(Object.entries(after.files)), Object.fromEntries(Object.entries(files).map(([file, sha256]) =>
    [path.relative(root, file).replaceAll("\\", "/"), {bytes: fs.statSync(file).size, sha256}]))) ||
      !isDeepStrictEqual([...after.dirs].sort(), [...exactInventory(handoff.files, "Windows controller").parents].sort())) {
    throw new Error("Windows controller handoff changed during admission");
  }
  if (expected.kind === "io") {
    return {kind: "io", handoff: handoffFile, handoffSha256: expected.handoffSha256, files,
      libraries: /** @type {ReturnType<typeof ioInputs>["libraries"]} */ (role.libraries)};
  }
  return {kind: "hermes", handoff: handoffFile, handoffSha256: expected.handoffSha256, files,
    hermes: /** @type {ReturnType<typeof hermesInputs>["hermes"]} */ (role.hermes),
    ...(role.intelRuntime ? {intelRuntime: /** @type {IntelRuntimeInputs} */ (role.intelRuntime)} : {})};
}

module.exports = {windowsControllerHandoffInputs, windowsLocalControllerHandoffInputs, validateIntelRuntimeMetadata};
