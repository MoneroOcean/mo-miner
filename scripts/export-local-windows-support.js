"use strict";

// Export only the selected Windows controller-support closure for WSL consumption.
const fs = require("node:fs"), path = require("node:path"), {createHash, randomUUID} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildJson, sourceInputs} = require("./build-source-inputs");
const {ioPins} = require("./build-native-io"), {opensslPin} = require("./build-native-openssl"), {hermesPins} = require("./build-native-hermes");
const {windowsLocalControllerHandoffInputs} = require("./native-windows-controller-handoff");
const {copyWindowsControllerPath, copyWindowsControllerIntelRuntime, windowsControllerPackageInventory} = require("./prepare-native-ci");
const windowsSupport = require("./build-local-native-windows-support");
const repo = fs.realpathSync(path.resolve(__dirname, "..")), uuid = /^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/;
const exportMarker = ".mom-native-windows-support-export.json", exportMarkerValue = {version: 1, format: "mom-native-windows-support-export-cache-v1", checkout: repo},
  exportRecipeFiles = ["scripts/export-local-windows-support.js", "scripts/build-local-native-windows-support.js", "scripts/build-native-windows-core.js", "scripts/native-windows-controller-handoff.js",
    "scripts/prepare-native-ci.js", "scripts/build-source-inputs.js", "scripts/build-native-io.js", "scripts/build-native-openssl.js",
    "scripts/build-native-hermes.js", "scripts/fetch-native-sdk-sources.js", "scripts/native-audit.js", "runtime/windows-intel-runtime.json"],
  sourceReceiptKeys = ["hermesFetch", "hermesRoots", "ioFetch", "opensslFetch"], maxExportGenerations = 128;
/** @typedef {{files:Array<{path:string,bytes:number,sha256:string}>,bytes:number}} WindowsControllerFileLedger */

/** @param {string} left @param {string} right */
function samePath(left, right) {return process.platform === "win32" ? left.toLowerCase() === right.toLowerCase() : left === right;}
/** @param {string} a @param {string} b */
function overlaps(a, b) {
  /** @param {string} parent @param {string} child */
  const contains = (parent, child) => {const rel = path.relative(parent, child); return !rel || !path.isAbsolute(rel) && rel !== ".." && !rel.startsWith(".." + path.sep);};
  return contains(a, b) || contains(b, a);
}
/** @param {string} file @param {string} label */
function canonicalFile(file, label) {
  if (typeof file !== "string" || !path.isAbsolute(file)) {throw new Error(label + " must be an absolute canonical regular file");}
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || !samePath(fs.realpathSync(file), file)) {
    throw new Error(label + " must be an absolute canonical regular file");
  }
  return file;
}
/** @param {string} root @param {string[]} protectedPaths */
function freshRoot(root, protectedPaths) {
  if (typeof root !== "string" || !path.isAbsolute(root)) {throw new Error("Windows support export root must be absolute");}
  const parent = path.dirname(root), parentStat = fs.lstatSync(parent, {throwIfNoEntry: false});
  if (!path.isAbsolute(root) || path.resolve(root) !== root || /[\r\n\0]/.test(root) || !parentStat?.isDirectory() || parentStat.isSymbolicLink() ||
      !samePath(fs.realpathSync(parent), parent) || overlaps(root, repo) || protectedPaths.some(file => overlaps(root, file)) ||
      fs.lstatSync(root, {throwIfNoEntry: false})) {throw new Error("Windows support export root must be new, canonical, and separate from its inputs");}
  return root;
}
/** @param {any} support @param {any} options */
function exportLocalWindowsSupport(support, options) {
  if (!support?.plan || !options || !uuid.test(options.localBuildId || "") ||
      !options.roots || Object.keys(options.roots).sort().join(",") !== "hermes,io" ||
      !options.sourceReceipts || Object.keys(options.sourceReceipts).sort().join(",") !== "hermesFetch,hermesRoots,ioFetch,opensslFetch") {
    throw new Error("Windows support export needs the finished producer result, UUID, role roots and original source receipts");
  }
  const plan = support.plan, io = support.io, ssl = support.openssl, hermes = support.hermes;
  if (io?.root !== plan.io?.output || ssl?.root !== plan.openssl?.output || hermes?.root !== plan.hermes?.output ||
      io.receipt !== path.join(plan.io.output, "result.json") || ssl.receipt !== path.join(plan.openssl.output, "result.json") ||
      hermes.receipt !== path.join(plan.hermes.output, "result.json") || hermes.before !== path.join(plan.hermes.output, "before.json") ||
      !isDeepStrictEqual(hermes.result, readBuildJson(hermes.receipt))) {throw new Error("Support result no longer matches its admitted roots or Hermes receipt");}
  const hermesSelected = ["source/LICENSE", "source/include", "source/API", "source/public", "build/lib/config", "build/lib/hermesvmlean_a.lib",
      "build/jsi/jsi.lib", "build/external/boost/boost_1_86_0/libs/context/boost_context.lib"],
    plannedRuntime = plan.hermes.runtimeEntries || [], admittedRuntime = hermes.runtimeEntries || [];
  if (!isDeepStrictEqual(plan.selected.hermes, [...hermesSelected, "before.json", "result.json"]) ||
      !Array.isArray(plannedRuntime) || !Array.isArray(admittedRuntime) || !isDeepStrictEqual(plannedRuntime, admittedRuntime)) {
    throw new Error("Hermes support plan differs from the selected controller handoff layout");
  }
  const sourceFiles = Object.values(options.sourceReceipts).map((file, index) => canonicalFile(file, "Original source receipt " + index)),
    supportFiles = [io.receipt, ssl.receipt, hermes.receipt, hermes.before],
    protectedPaths = [...supportFiles, ...sourceFiles, plan.sources.libuv, plan.sources.zlib, plan.sources.openssl, plan.sources.hermes,
      plan.io.output, plan.openssl.output, plan.hermes.output],
    roots = {io: freshRoot(options.roots.io, protectedPaths), hermes: freshRoot(options.roots.hermes, protectedPaths)};
  if (overlaps(roots.io, roots.hermes)) {throw new Error("I/O and Hermes export roots must be distinct");}

  const unchanged = new Map();
  /** @param {string} file @param {string} [expected] */
  const admitFile = (file, expected) => {
    canonicalFile(file, "Admitted support input");
    const actual = fileHash(file);
    if (expected && actual !== expected) {throw new Error("Admitted support input changed: " + file);}
    const previous = unchanged.get(file);
    if (previous && previous !== actual) {throw new Error("Support input changed during export: " + file);}
    unchanged.set(file, actual); return actual;
  };
  for (const [file, sha] of Object.entries(io.artifacts || {}).map(([relative, sha]) => [path.join(io.root, ...relative.split("/")), sha])) {admitFile(file, sha);}
  for (const [file, sha] of Object.entries(ssl.files || {})) {admitFile(file, sha);}
  const artifactTails = /** @type {Record<string, string>} */ ({compiler: "build/bin/shermes.exe", context: "build/external/boost/boost_1_86_0/libs/context/boost_context.lib",
    jsi: "build/jsi/jsi.lib", public: "build/public/hermes/Public/hermesPublic.lib", runtime: "build/lib/hermesvmlean_a.lib"});
  for (const [name, row] of Object.entries(hermes.result.artifacts || {})) {
    const artifactPath = artifactTails[name];
    if (!artifactPath) {throw new Error("Hermes admission contains an unknown artifact: " + name);}
    const source = path.join(plan.hermes.output, ...artifactPath.split("/"));
    admitFile(source, row.sha256);
    if (!Number.isSafeInteger(row.bytes) || fs.statSync(source).size !== row.bytes) {throw new Error("Hermes admitted artifact size changed: " + name);}
  }
  for (const [relative, sha] of Object.entries(hermes.result.sources || {})) {admitFile(path.join(plan.hermes.output, "source", ...relative.split("/")), sha);}
  for (const file of [...supportFiles, ...sourceFiles]) {admitFile(file);}
  if (fileHash(io.receipt) !== support.io.receiptSha256 || fileHash(ssl.receipt) !== support.openssl.receiptSha256) {
    throw new Error("Support producer receipt changed after admission");
  }

  fs.mkdirSync(roots.io, {mode: 0o700}); fs.mkdirSync(roots.hermes, {mode: 0o700});
  const ioLedger = /** @type {WindowsControllerFileLedger} */ ({files: [], bytes: 0}),
    hermesLedger = /** @type {WindowsControllerFileLedger} */ ({files: [], bytes: 0});
  /** @param {string} source @param {string} target @param {string} relative @param {WindowsControllerFileLedger} ledger */
  const copy = (source, target, relative, ledger) => copyWindowsControllerPath(source, target, relative, ledger);
  for (const [relative, sha] of Object.entries(io.artifacts).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) {
    if ((!relative.startsWith("include/") && !relative.startsWith("lib/")) || relative.includes("\\") || relative.split("/").some(part => !part || part === "." || part === "..")) {
      throw new Error("I/O SDK admission contains a non-selected handoff path");
    }
    copy(path.join(io.root, ...relative.split("/")), path.join(roots.io, "io-sdk", ...relative.split("/")), "io-sdk/" + relative, ioLedger);
    if (fileHash(path.join(roots.io, "io-sdk", ...relative.split("/"))) !== sha) {throw new Error("Copied I/O artifact differs from its admission");}
  }
  copy(io.receipt, path.join(roots.io, "io-sdk/result.json"), "io-sdk/result.json", ioLedger);
  const sslReceipt = readBuildJson(ssl.receipt), config = plan.openssl.config,
    toolsRecord = {version: config.version, jobs: config.jobs, tools: config.tools}, toolsBytes = JSON.stringify(toolsRecord),
    toolsSha256 = createHash("sha256").update(toolsBytes).digest("hex");
  if (sslReceipt.config?.sha256 !== toolsSha256) {throw new Error("OpenSSL descriptor does not match its unchanged producer receipt");}
  const sslMembers = Object.keys(ssl.artifacts || {}).filter(name => name.startsWith("include/") || name === "lib/libcrypto.lib" || name === "lib/libssl.lib").sort();
  if (!sslMembers.includes("lib/libcrypto.lib") || !sslMembers.includes("lib/libssl.lib") ||
      Object.keys(ssl.artifacts).some(name => !name.startsWith("include/") && !["bin/openssl.exe", "lib/libcrypto.lib", "lib/libssl.lib"].includes(name))) {
    throw new Error("OpenSSL admission omits a selected library or contains an unexpected artifact");
  }
  for (const relative of sslMembers) {
    copy(path.join(ssl.root, ...relative.split("/")), path.join(roots.io, "openssl-sdk", ...relative.split("/")), "openssl-sdk/" + relative, ioLedger);
  }
  copy(path.join(ssl.root, "source/LICENSE.txt"), path.join(roots.io, "openssl-sdk/source/LICENSE.txt"), "openssl-sdk/source/LICENSE.txt", ioLedger);
  copy(ssl.receipt, path.join(roots.io, "openssl-sdk/result.json"), "openssl-sdk/result.json", ioLedger);
  fs.writeFileSync(path.join(roots.io, "openssl-sdk/producer-tools.json"), toolsBytes, {flag: "wx", mode: 0o600});
  copy(options.sourceReceipts.ioFetch, path.join(roots.io, "provenance/io-fetch.json"), "provenance/io-fetch.json", ioLedger);
  copy(options.sourceReceipts.opensslFetch, path.join(roots.io, "provenance/openssl-fetch.json"), "provenance/openssl-fetch.json", ioLedger);

  for (const relative of hermesSelected) {
    copy(path.join(plan.hermes.output, ...relative.split("/")), path.join(roots.hermes, "hermes", ...relative.split("/")), "hermes/" + relative, hermesLedger);
  }
  copy(hermes.before, path.join(roots.hermes, "hermes/before.json"), "hermes/before.json", hermesLedger);
  copy(hermes.receipt, path.join(roots.hermes, "hermes/result.json"), "hermes/result.json", hermesLedger);
  copy(options.sourceReceipts.hermesFetch, path.join(roots.hermes, "provenance/hermes-fetch.json"), "provenance/hermes-fetch.json", hermesLedger);
  copy(options.sourceReceipts.hermesRoots, path.join(roots.hermes, "provenance/hermes-roots.json"), "provenance/hermes-roots.json", hermesLedger);
  let intelRuntime;
  if (hermes.runtimeEntries?.length) {
    const runtime = copyWindowsControllerIntelRuntime(plan.tools.hermes.cc, roots.hermes, hermesLedger),
      planned = /** @type {Array<{source:string,sha256:string}>} */ (hermes.runtimeEntries).map(row => ({path: row.source, sha256: row.sha256}))
        .sort((a, b) => a.path.localeCompare(b.path)),
      copied = runtime.sourceFiles.map(row => ({path: row.path, sha256: row.sha256})).sort((a, b) => a.path.localeCompare(b.path));
    if (!isDeepStrictEqual(planned, copied)) {throw new Error("Windows Intel runtime selection differs from the admitted support plan");}
    intelRuntime = runtime.record;
  }

  const ioResult = finish("io", roots.io, options.localBuildId, ["scripts/build-native-io.js", "scripts/build-native-openssl.js"],
      ["provenance/io-fetch.json", "provenance/openssl-fetch.json"], ioLedger, {}),
    hermesResult = finish("hermes", roots.hermes, options.localBuildId, ["scripts/build-native-hermes.js"],
      ["provenance/hermes-fetch.json", "provenance/hermes-roots.json"], hermesLedger, {intelRuntime});
  for (const [file, sha] of unchanged) {if (fileHash(file) !== sha) {throw new Error("Original support/source receipt changed during export: " + file);}}
  return {localBuildId: options.localBuildId, io: ioResult, hermes: hermesResult};
}

/** @param {"io"|"hermes"} kind @param {string} root @param {string} localBuildId @param {string[]} scripts
 * @param {string[]} sourceFiles @param {{files:Array<{path:string,bytes:number,sha256:string}>,bytes:number}} ledger @param {Record<string,any>} extra */
function finish(kind, root, localBuildId, scripts, sourceFiles, ledger, extra) {
  const firstSource = sourceFiles[0], secondSource = sourceFiles[1];
  if (typeof firstSource !== "string" || typeof secondSource !== "string") {throw new Error("Windows export needs both original role source receipts");}
  const packageState = windowsControllerPackageInventory(root, "native-windows-controller-" + kind), producers = kind === "io" ? [
      {script: scripts[0], receipt: "io-sdk/result.json", receiptSha256: fileHash(path.join(root, "io-sdk/result.json"))},
      {script: scripts[1], receipt: "openssl-sdk/result.json", receiptSha256: fileHash(path.join(root, "openssl-sdk/result.json"))},
    ] : [{script: scripts[0], receipt: "hermes/result.json", receiptSha256: fileHash(path.join(root, "hermes/result.json")),
      plan: "hermes/before.json", planSha256: fileHash(path.join(root, "hermes/before.json"))}],
    sourceReceipts = kind === "io" ? [
      {file: firstSource, sha256: fileHash(path.join(root, ...firstSource.split("/"))), archives: {libuv: ioPins.libuv.sha256, zlib: ioPins.zlib.sha256}},
      {file: secondSource, sha256: fileHash(path.join(root, ...secondSource.split("/"))), archives: {openssl: opensslPin.sha256}},
    ] : [{file: firstSource, sha256: fileHash(path.join(root, ...firstSource.split("/"))), rootsFile: secondSource,
      rootsSha256: fileHash(path.join(root, ...secondSource.split("/"))), revision: hermesPins.revision}],
    scope = kind === "io" ? "Selected Windows libuv/zlib and OpenSSL headers/static archives plus unchanged producer and source receipts; no full-SDK or application qualification." :
      "Selected Windows Hermes headers/config/static libraries" + (extra["intelRuntime"] ? " and pinned Intel 2026.2 static runtime/license files" : "") +
        " plus unchanged producer/source receipts; no AOT compiler or full-SDK re-admission.",
    omitted = kind === "io" ? ["OpenSSL source/build trees", "openssl.exe", "compiler/build tools", "DLLs"] :
      ["Hermes Git/build trees", "shermes.exe", "hermesPublic.lib", "compiler/build tools", "DLLs"],
    handoff = {version: 1, format: "mom-windows-controller-support-handoff-v1", localBuildId, target: "win32", architecture: "x64", kind,
      buildOnly: true, executed: false, publicationQualified: false, scope, producers, sourceReceipts, omitted,
      ...(kind === "hermes" && extra["intelRuntime"] ? {intelRuntime: extra["intelRuntime"]} : {}), files: packageState.files};
  if (ledger.bytes > (512 << 20) || packageState.bytes > (512 << 20)) {throw new Error("Windows selected support export exceeds its 512 MiB bound");}
  fs.writeFileSync(path.join(root, "handoff.json"), JSON.stringify(handoff, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const handoffSha256 = fileHash(path.join(root, "handoff.json")), admitted = windowsLocalControllerHandoffInputs(root, {kind, localBuildId, handoffSha256});
  return {root, handoff: path.join(root, "handoff.json"), handoffSha256, files: packageState.files.length, bytes: packageState.bytes, admitted};
}

/** @param {unknown} value @param {string[]} keys @param {string} label */
function exactRecord(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value) || !isDeepStrictEqual(Object.keys(value).sort(), [...keys].sort())) {
    throw new Error(label + " has unknown or missing fields");
  }
  return /** @type {Record<string, any>} */ (value);
}
/** @param {unknown} value */
function isHash(value) {return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);}
/** @param {unknown} value */
function isLocalBuildId(value) {return typeof value === "string" && uuid.test(value);}
/** @param {unknown} value @param {string[]} keys @param {string} label */
function exactHashes(value, keys, label) {
  const record = exactRecord(value, keys, label);
  if (Object.values(record).some(hash => !isHash(hash))) {throw new Error(label + " contains an invalid SHA-256");}
  return /** @type {Record<string, string>} */ (record);
}
/** @param {string} directory @param {string} label */
function canonicalDirectory(directory, label) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!stat?.isDirectory() || stat.isSymbolicLink() || !samePath(fs.realpathSync(directory), directory)) {
    throw new Error(label + " must be a canonical directory");
  }
  return directory;
}
/** @param {string} directory @param {string[]} expected @param {string} label */
function exactEntries(directory, expected, label) {
  const entries = fs.readdirSync(directory).sort();
  if (!isDeepStrictEqual(entries, [...expected].sort())) {throw new Error(label + " is partial or contains unknown state; preserve it for inspection");}
}
/** @param {string} file @param {string} label @param {number} [maximum] */
function canonicalRecordFile(file, label, maximum = 1 << 20) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size > maximum || !samePath(fs.realpathSync(file), file)) {
    throw new Error(label + " must be a canonical, unaliased regular file");
  }
  return stat;
}
/** @param {string} workspace @param {any} support @param {Record<string,string>} receipts */
function checkExportWorkspacePath(workspace, support, receipts) {
  const plan = support.plan, protectedPaths = [repo, support.workspace, plan.sources.libuv, plan.sources.zlib, plan.sources.openssl, plan.sources.hermes,
    plan.io.output, plan.openssl.output, plan.hermes.output, ...Object.values(receipts)];
  if (typeof workspace !== "string" || !path.isAbsolute(workspace) || path.resolve(workspace) !== workspace || /[\r\n\0]/.test(workspace) ||
      protectedPaths.some(file => typeof file !== "string" || overlaps(workspace, file))) {
    throw new Error("Windows support export cache must be canonical and outside the checkout, producers and source receipts");
  }
  const parent = path.dirname(workspace);
  canonicalDirectory(parent, "Windows support export cache parent");
  const stat = fs.lstatSync(workspace, {throwIfNoEntry: false});
  if (stat && (!stat.isDirectory() || stat.isSymbolicLink() || !samePath(fs.realpathSync(workspace), workspace))) {
    throw new Error("Windows support export cache must be a canonical directory");
  }
}
/** @param {any} support @param {Record<string,string>} sourceReceipts
 * @param {{planSupport?:typeof windowsSupport.windowsSupportPlan,admitIo?:typeof windowsSupport.admitIoOutput,
 * admitOpenSsl?:typeof windowsSupport.admitOpenSslOutput,admitHermes?:typeof windowsSupport.admitHermesOutput,
 * ioAdmission?:Parameters<typeof windowsSupport.admitIoOutput>[1],openSslAdmission?:Parameters<typeof windowsSupport.admitOpenSslOutput>[1],
 * hermesAdmission?:Parameters<typeof windowsSupport.admitHermesOutput>[1]}} dependencies */
function admitCurrentSupport(support, sourceReceipts, dependencies = {}) {
  if (!support || typeof support !== "object" || !support.plan || typeof support.workspace !== "string" ||
      !support.plan.sources || !support.plan.tools) {throw new Error("Local Windows export needs the current Windows support producer result");}
  const plan = (dependencies.planSupport || windowsSupport.windowsSupportPlan)({workspace: support.workspace, sources: support.plan.sources, tools: support.plan.tools});
  if (!isDeepStrictEqual(plan, support.plan)) {throw new Error("Current Windows support producer plan differs from the supplied result");}
  for (const role of /** @type {const} */ (["io", "openssl", "hermes"])) {
    if (!isHash(plan[role]?.key)) {throw new Error("Current Windows support plan omits a content-addressed " + role + " generation key");}
  }
  const io = (dependencies.admitIo || windowsSupport.admitIoOutput)(plan, dependencies.ioAdmission),
    ssl = (dependencies.admitOpenSsl || windowsSupport.admitOpenSslOutput)(plan, dependencies.openSslAdmission),
    hermes = (dependencies.admitHermes || windowsSupport.admitHermesOutput)(plan, dependencies.hermesAdmission);
  if (support.io?.root !== plan.io.output || support.io.receipt !== io.receipt || support.io.receiptSha256 !== io.receiptSha256 ||
      !isDeepStrictEqual(support.io.artifacts, io.artifacts) || support.openssl?.root !== plan.openssl.output ||
      support.openssl.receipt !== ssl.receipt || support.openssl.receiptSha256 !== ssl.receiptSha256 ||
      !isDeepStrictEqual(support.openssl.artifacts, ssl.artifacts) || !isDeepStrictEqual(support.openssl.files, ssl.files) ||
      support.hermes?.root !== plan.hermes.output || support.hermes.receipt !== hermes.receipt || support.hermes.before !== hermes.before ||
      !isDeepStrictEqual(support.hermes.result, hermes.result) || !isDeepStrictEqual(support.hermes.runtimeEntries, plan.hermes.runtimeEntries)) {
    throw new Error("Current Windows support result differs from original producer re-admission");
  }
  for (const [name, file] of Object.entries(sourceReceipts)) {canonicalFile(file, "Original " + name + " source receipt");}
  return {plan, io, ssl, hermes};
}
/** @param {string} root @param {Record<string,string>} files @param {Map<string,string>} expected @param {string} label */
function compareHandoffFiles(root, files, expected, label) {
  const actual = new Map();
  for (const [file, hash] of Object.entries(files)) {
    const relative = path.relative(root, file).split(path.sep).join("/");
    if (relative === "handoff.json") {continue;}
    if (!relative || path.isAbsolute(relative) || relative === ".." || relative.startsWith("../") || actual.has(relative)) {
      throw new Error(label + " handoff contains an aliased or out-of-root file");
    }
    actual.set(relative, hash);
  }
  const difference = [...expected].find(([relative, hash]) => actual.get(relative) !== hash)?.[0] ||
    [...actual.keys()].find(relative => !expected.has(relative));
  if (actual.size !== expected.size || difference) {
    throw new Error(label + " handoff payload differs from current original producer inputs" + (difference ? ": " + difference : ""));
  }
}
/** @param {Map<string,string>} expected @param {string} relative @param {string} file @param {string} sha256 @param {string} label */
function admitSelectedFile(expected, relative, file, sha256, label) {
  canonicalFile(file, label);
  if (!isHash(sha256) || fileHash(file) !== sha256) {throw new Error(label + " changed after original producer admission");}
  if (expected.has(relative)) {throw new Error("Duplicate selected " + label + " handoff path");}
  expected.set(relative, sha256);
}
/** @param {any} current @param {Record<string,string>} sourceReceipts @param {Record<string,string>} receiptHashes
 * @param {{io:any,hermes:any}} selected */
function joinCurrentHandoffs(current, sourceReceipts, receiptHashes, selected) {
  const {plan, io, ssl, hermes} = current, expectedIo = new Map(), expectedHermes = new Map(),
    ioRoot = selected.io.root, hermesRoot = selected.hermes.root;
  for (const [relative, sha256] of Object.entries(io.artifacts)) {
    admitSelectedFile(expectedIo, "io-sdk/" + relative, path.join(plan.io.output, ...relative.split("/")), sha256, "Windows I/O artifact");
  }
  admitSelectedFile(expectedIo, "io-sdk/result.json", io.receipt, io.receiptSha256, "Windows I/O receipt");
  const sslConfig = {version: plan.openssl.config.version, jobs: plan.openssl.config.jobs, tools: plan.openssl.config.tools},
    toolsBytes = JSON.stringify(sslConfig), toolsSha256 = createHash("sha256").update(toolsBytes).digest("hex"), sslMembers = Object.keys(ssl.artifacts)
      .filter(name => name.startsWith("include/") || name === "lib/libcrypto.lib" || name === "lib/libssl.lib").sort();
  for (const relative of sslMembers) {
    admitSelectedFile(expectedIo, "openssl-sdk/" + relative, path.join(plan.openssl.output, ...relative.split("/")), ssl.artifacts[relative], "Windows OpenSSL artifact");
  }
  const license = path.join(plan.openssl.output, "source/LICENSE.txt"), licenseSha256 = ssl.files[license];
  admitSelectedFile(expectedIo, "openssl-sdk/source/LICENSE.txt", license, licenseSha256, "Windows OpenSSL license");
  admitSelectedFile(expectedIo, "openssl-sdk/result.json", ssl.receipt, ssl.receiptSha256, "Windows OpenSSL receipt");
  expectedIo.set("openssl-sdk/producer-tools.json", toolsSha256);
  admitSelectedFile(expectedIo, "provenance/io-fetch.json", sourceReceipts["ioFetch"] || "", receiptHashes["ioFetch"] || "", "Original I/O source receipt");
  admitSelectedFile(expectedIo, "provenance/openssl-fetch.json", sourceReceipts["opensslFetch"] || "", receiptHashes["opensslFetch"] || "", "Original OpenSSL source receipt");

  const result = hermes.result, sourceSelection = Object.entries(result.sources).filter(([name]) => name === "LICENSE" ||
      name.startsWith("include/") || name.startsWith("API/") || name.startsWith("public/"));
  for (const [relative, sha256] of sourceSelection) {
    admitSelectedFile(expectedHermes, "hermes/source/" + relative, path.join(plan.hermes.output, "source", ...relative.split("/")), sha256, "Windows Hermes source");
  }
  const artifactTails = {runtime: "build/lib/hermesvmlean_a.lib", jsi: "build/jsi/jsi.lib", context: "build/external/boost/boost_1_86_0/libs/context/boost_context.lib"};
  for (const [name, tail] of Object.entries(artifactTails)) {
    const artifact = result.artifacts[name];
    admitSelectedFile(expectedHermes, "hermes/" + tail, path.join(plan.hermes.output, ...tail.split("/")), artifact?.sha256, "Windows Hermes " + name + " archive");
  }
  const configRoot = path.join(plan.hermes.output, "build/lib/config"), configInventory = windowsControllerPackageInventory(configRoot, "hermes-config");
  for (const row of configInventory.files) {
    admitSelectedFile(expectedHermes, "hermes/build/lib/config/" + row.path, path.join(configRoot, ...row.path.split("/")), row.sha256, "Windows Hermes generated config");
  }
  admitSelectedFile(expectedHermes, "hermes/before.json", hermes.before, fileHash(hermes.before), "Windows Hermes build plan");
  admitSelectedFile(expectedHermes, "hermes/result.json", hermes.receipt, fileHash(hermes.receipt), "Windows Hermes producer receipt");
  admitSelectedFile(expectedHermes, "provenance/hermes-fetch.json", sourceReceipts["hermesFetch"] || "", receiptHashes["hermesFetch"] || "", "Original Hermes source receipt");
  admitSelectedFile(expectedHermes, "provenance/hermes-roots.json", sourceReceipts["hermesRoots"] || "", receiptHashes["hermesRoots"] || "", "Original Hermes roots receipt");
  for (const entry of plan.hermes.runtimeEntries || []) {
    admitSelectedFile(expectedHermes, entry.path, entry.source, entry.sha256, "Pinned Intel runtime input");
  }
  compareHandoffFiles(ioRoot, selected.io.admitted.files, expectedIo, "Windows I/O/OpenSSL");
  compareHandoffFiles(hermesRoot, selected.hermes.admitted.files, expectedHermes, "Windows Hermes");
}
/** @typedef {{version:number,planKeys:{io:string,openssl:string,hermes:string},sourceReceiptHashes:{hermesFetch:string,hermesRoots:string,ioFetch:string,opensslFetch:string},exportRecipeSha256:string}} WindowsExportIdentity */
/** @param {WindowsExportIdentity} identity */
function generationKey(identity) {return createHash("sha256").update(JSON.stringify(identity)).digest("hex");}
/** A historical generation is only a retirement candidate, not current-source proof.
 * Read its sealed identity without applying today's recipes to yesterday's payload.
 * @param {string} root */
function generationRecord(root) {
  canonicalDirectory(root, "Windows support export generation");
  exactEntries(root, ["commit.json", "hermes", "io"], "Windows support export generation");
  const recordFile = path.join(root, "commit.json"), recordStat = canonicalRecordFile(recordFile, "Windows support export commit"),
    record = readBuildJson(recordFile);
  exactRecord(record, ["version", "generationKey", "planKeys", "sourceReceiptHashes", "exportRecipeSha256", "localBuildId", "handoffSha256"], "Windows support export commit");
  const planKeys = /** @type {WindowsExportIdentity["planKeys"]} */ (exactHashes(record.planKeys, ["io", "openssl", "hermes"], "Windows support export plan keys")),
    sourceReceiptHashes = /** @type {WindowsExportIdentity["sourceReceiptHashes"]} */ (exactHashes(record.sourceReceiptHashes, sourceReceiptKeys, "Windows support export source receipt pins")),
    pins = /** @type {{io:string,hermes:string}} */ (exactHashes(record.handoffSha256, ["io", "hermes"], "Windows support handoff pins")),
    checkedIdentity = {version: record.version, planKeys, sourceReceiptHashes, exportRecipeSha256: record.exportRecipeSha256};
  if (record.version !== 1 || !isHash(record.exportRecipeSha256) || !isLocalBuildId(record.localBuildId) ||
      record.generationKey !== generationKey(checkedIdentity) || path.basename(root) !== record.generationKey || recordStat.nlink !== 1) {
    throw new Error("Windows support export commit differs from its stable generation identity");
  }
  for (const role of ["io", "hermes"]) {canonicalDirectory(path.join(root, role), "Windows support historical role");}
  return {record, checkedIdentity, pins, sourceReceiptHashes};
}
/** @param {string} root @param {string|undefined} localBuildId @param {{io:string,hermes:string}|undefined} handoffSha256
 * @param {WindowsExportIdentity|undefined} identity
 * @param {typeof windowsLocalControllerHandoffInputs} readHandoff */
function admitGeneration(root, localBuildId, handoffSha256, identity, readHandoff = windowsLocalControllerHandoffInputs) {
  const {record, checkedIdentity, pins, sourceReceiptHashes} = generationRecord(root);
  if (identity !== undefined && !isDeepStrictEqual(checkedIdentity, identity) || localBuildId !== undefined && record.localBuildId !== localBuildId ||
      handoffSha256 !== undefined && !isDeepStrictEqual(pins, handoffSha256)) {
    throw new Error("Windows support export commit differs from its current generation identity");
  }
  const io = readHandoff(path.join(root, "io"), {kind: "io", localBuildId: record.localBuildId, handoffSha256: pins["io"]}),
    hermes = readHandoff(path.join(root, "hermes"), {kind: "hermes", localBuildId: record.localBuildId, handoffSha256: pins["hermes"]});
  if (io.kind !== "io" || hermes.kind !== "hermes" || io.localBuildId !== record.localBuildId || hermes.localBuildId !== record.localBuildId ||
      io.files[path.join(root, "io/provenance/io-fetch.json")] !== sourceReceiptHashes["ioFetch"] ||
      io.files[path.join(root, "io/provenance/openssl-fetch.json")] !== sourceReceiptHashes["opensslFetch"] ||
      hermes.files[path.join(root, "hermes/provenance/hermes-fetch.json")] !== sourceReceiptHashes["hermesFetch"] ||
      hermes.files[path.join(root, "hermes/provenance/hermes-roots.json")] !== sourceReceiptHashes["hermesRoots"]) {
    throw new Error("Windows support export commit does not bind both selected handoffs and original receipts");
  }
  return {record, io, hermes};
}
/** @param {string} workspace @param {any} support @param {Record<string,string>} receipts */
function admitExportWorkspace(workspace, support, receipts) {
  checkExportWorkspacePath(workspace, support, receipts);
  const marker = path.join(workspace, exportMarker), stat = fs.lstatSync(workspace, {throwIfNoEntry: false});
  if (!stat) {
    fs.mkdirSync(workspace, {mode: 0o700});
    fs.writeFileSync(marker, JSON.stringify(exportMarkerValue) + "\n", {flag: "wx", mode: 0o600});
  } else {
    canonicalDirectory(workspace, "Windows support export cache");
    canonicalRecordFile(marker, "Windows support export cache marker", 4096);
    if (fs.readFileSync(marker, "utf8") !== JSON.stringify(exportMarkerValue) + "\n") {
      throw new Error("Windows support export cache marker belongs to another checkout or format");
    }
  }
  const entries = fs.readdirSync(workspace, {withFileTypes: true});
  if (entries.length > maxExportGenerations + 1 || entries.some(entry => entry.name !== exportMarker && !/^[a-f0-9]{64}$/.test(entry.name))) {
    throw new Error("Windows support export cache has unknown or excessive entries; preserve it for inspection");
  }
  return entries.filter(entry => entry.name !== exportMarker).map(entry => {
    const root = path.join(workspace, entry.name);
    if (!entry.isDirectory() || entry.isSymbolicLink() || !samePath(fs.realpathSync(root), root)) {
      throw new Error("Windows support export generation is redirected or not a directory");
    }
    const {record} = generationRecord(root);
    return {generationKey: entry.name, localBuildId: record.localBuildId, roots: {io: path.join(root, "io"), hermes: path.join(root, "hermes")},
      handoffSha256: record.handoffSha256, currentSourceAdmitted: false};
  });
}
/** Export the current support closure or re-admit its existing generation, then delete older
 * generations; retirementCandidates lists the deleted ones.
 * @param {any} support @param {{workspace:string,sourceReceipts:{hermesFetch:string,hermesRoots:string,ioFetch:string,opensslFetch:string}}} options
 * @param {{planSupport?:typeof windowsSupport.windowsSupportPlan,admitIo?:typeof windowsSupport.admitIoOutput,
 * admitOpenSsl?:typeof windowsSupport.admitOpenSslOutput,admitHermes?:typeof windowsSupport.admitHermesOutput,
 * ioAdmission?:Parameters<typeof windowsSupport.admitIoOutput>[1],openSslAdmission?:Parameters<typeof windowsSupport.admitOpenSslOutput>[1],
 * hermesAdmission?:Parameters<typeof windowsSupport.admitHermesOutput>[1],export?:typeof exportLocalWindowsSupport,
 * readHandoff?:typeof windowsLocalControllerHandoffInputs,id?:()=>string}} [dependencies]
 * @returns {ReturnType<typeof exportLocalWindowsSupport> & {generationKey:string,reused:boolean,retirementCandidates:Array<any>}} */
function exportOrAdmitLocalWindowsSupport(support, options, dependencies = {}) {
  if (!options || Object.keys(options).sort().join(",") !== "sourceReceipts,workspace" || typeof options.workspace !== "string" || !options.sourceReceipts ||
      Object.keys(options.sourceReceipts).sort().join(",") !== sourceReceiptKeys.slice().sort().join(",")) {
    throw new Error("Windows support export admission needs one stable cache workspace and four original source receipts");
  }
  const receipts = /** @type {Record<string,string>} */ (options.sourceReceipts), current = admitCurrentSupport(support, receipts, dependencies),
    receiptHashes = /** @type {WindowsExportIdentity["sourceReceiptHashes"]} */ (Object.fromEntries(sourceReceiptKeys.map(name => [name, fileHash(receipts[name] || "")]))),
    planKeys = /** @type {WindowsExportIdentity["planKeys"]} */ (Object.fromEntries(["io", "openssl", "hermes"].map(name => [name, current.plan[/** @type {"io"|"openssl"|"hermes"} */ (name)].key]))),
    exportRecipeSha256 = createHash("sha256").update(JSON.stringify(sourceInputs(repo, [], exportRecipeFiles))).digest("hex"),
    identity = /** @type {WindowsExportIdentity} */ ({version: 1, planKeys, sourceReceiptHashes: receiptHashes, exportRecipeSha256}), key = generationKey(identity),
    readHandoff = dependencies.readHandoff || windowsLocalControllerHandoffInputs,
    generations = admitExportWorkspace(options.workspace, support, receipts), existing = generations.find(row => row.generationKey === key);
  if (existing) {
    const generationRoot = path.join(options.workspace, key), checked = admitGeneration(generationRoot, undefined, undefined, identity, readHandoff),
      result = {localBuildId: checked.record.localBuildId,
        io: handoffResult(path.join(generationRoot, "io"), checked.record["handoffSha256"]["io"], checked.io),
        hermes: handoffResult(path.join(generationRoot, "hermes"), checked.record["handoffSha256"]["hermes"], checked.hermes)};
    joinCurrentHandoffs(current, receipts, receiptHashes, result);
    const latest = admitCurrentSupport(support, receipts, dependencies), latestChecked = admitGeneration(generationRoot, undefined, undefined, identity, readHandoff),
      latestResult = {localBuildId: latestChecked.record.localBuildId,
        io: handoffResult(path.join(generationRoot, "io"), latestChecked.record["handoffSha256"]["io"], latestChecked.io),
        hermes: handoffResult(path.join(generationRoot, "hermes"), latestChecked.record["handoffSha256"]["hermes"], latestChecked.hermes)};
    if (!isDeepStrictEqual(current.plan, latest.plan)) {throw new Error("Original Windows support plan changed during warm handoff admission");}
    joinCurrentHandoffs(latest, receipts, receiptHashes, latestResult);
    assertSourceReceiptsUnchanged(receipts, receiptHashes, "warm export admission");
    return {...latestResult, generationKey: key, reused: true, retirementCandidates: retireGenerations(options.workspace, generations, key)};
  }

  if (generations.length >= maxExportGenerations) {throw new Error("Windows support export cache reached its generation bound; preserve it for inspection");}

  const id = (dependencies.id || randomUUID)();
  if (!/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(id)) {
    throw new Error("Windows support export needs a UUIDv4 local build identity");
  }
  const generationRoot = path.join(options.workspace, key);
  fs.mkdirSync(generationRoot, {mode: 0o700});
  const roots = {io: path.join(generationRoot, "io"), hermes: path.join(generationRoot, "hermes")},
    exported = (dependencies.export || exportLocalWindowsSupport)(support, {localBuildId: id, roots, sourceReceipts: receipts});
  if (exported.localBuildId !== id || exported.io?.root !== roots.io || exported.hermes?.root !== roots.hermes) {
    throw new Error("Windows support exporter returned unexpected generation roots or identity");
  }
  const after = admitCurrentSupport(support, receipts, dependencies), result = {localBuildId: id,
    io: handoffResult(roots.io, exported.io.handoffSha256, exported.io.admitted),
    hermes: handoffResult(roots.hermes, exported.hermes.handoffSha256, exported.hermes.admitted)};
  if (!isDeepStrictEqual(current.plan, after.plan)) {throw new Error("Original Windows support plan changed during handoff export");}
  joinCurrentHandoffs(after, receipts, receiptHashes, result);
  assertSourceReceiptsUnchanged(receipts, receiptHashes, "handoff export");
  const pins = {io: result.io.handoffSha256, hermes: result.hermes.handoffSha256}, record = {...identity, generationKey: key, localBuildId: id, handoffSha256: pins};
  fs.writeFileSync(path.join(generationRoot, "commit.json"), JSON.stringify(record, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const latest = admitCurrentSupport(support, receipts, dependencies), latestChecked = admitGeneration(generationRoot, id, pins, identity, readHandoff),
    latestResult = {localBuildId: id,
      io: handoffResult(roots.io, pins.io, latestChecked.io), hermes: handoffResult(roots.hermes, pins.hermes, latestChecked.hermes)};
  if (!isDeepStrictEqual(current.plan, latest.plan)) {throw new Error("Original Windows support plan changed during handoff commit");}
  joinCurrentHandoffs(latest, receipts, receiptHashes, latestResult);
  assertSourceReceiptsUnchanged(receipts, receiptHashes, "handoff commit");
  return {...latestResult, generationKey: key, reused: false, retirementCandidates: retireGenerations(options.workspace, generations, key)};
}
/** Delete every admitted generation except the current one and return the deleted rows. Only the
 * current generation is read by this build, and the held Engineering build lock serializes builds
 * of one checkout. admitExportWorkspace proved each row is a direct 64-hex canonical directory; the
 * re-check here rejects a swap since then, and rmSync unlinks nested links instead of following them.
 * @param {string} workspace @param {ReturnType<typeof admitExportWorkspace>} generations @param {string} current */
function retireGenerations(workspace, generations, current) {
  const retired = generations.filter(row => row.generationKey !== current);
  for (const row of retired) {
    canonicalDirectory(path.join(workspace, row.generationKey), "Retired Windows support export generation");
    fs.rmSync(path.join(workspace, row.generationKey), {recursive: true});
  }
  return retired;
}
/** @param {Record<string,string>} receipts @param {WindowsExportIdentity["sourceReceiptHashes"]} hashes @param {string} phase */
function assertSourceReceiptsUnchanged(receipts, hashes, phase) {
  for (const name of sourceReceiptKeys) {
    const file = Reflect.get(receipts, name), expected = Reflect.get(hashes, name);
    if (typeof file !== "string" || typeof expected !== "string" || fileHash(file) !== expected) {
      throw new Error("Original source receipt changed during " + phase);
    }
  }
}
/** @param {string} root @param {string} handoffSha256 @param {any} admitted */
function handoffResult(root, handoffSha256, admitted) {
  const handoff = path.join(root, "handoff.json"), files = Object.keys(admitted.files).filter(file => file !== handoff),
    bytes = files.reduce((total, file) => total + fs.statSync(file).size, 0);
  if (files.length > 50000 || bytes > (512 << 20)) {throw new Error("Windows support handoff exceeds its public admission bounds");}
  return {root, handoff: path.join(root, "handoff.json"), handoffSha256, files: files.length, bytes, admitted};
}

module.exports = {exportLocalWindowsSupport, exportOrAdmitLocalWindowsSupport};
