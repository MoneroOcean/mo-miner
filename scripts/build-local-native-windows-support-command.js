"use strict";

// Native-Windows half of the WSL release build. Original source receipts survive
// transport unchanged; only selected support artifacts are exported back to WSL.
const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto");
const support = require("./build-local-native-windows-support"), sources = require("./fetch-native-sdk-sources"),
  ci = require("./native-ci-windows-controller"), exporter = require("./build-local-native-windows-support-export");
const {fileHash} = require("./build-source-inputs"), {opensslPin} = require("./build-native-openssl");
const repo = fs.realpathSync(path.resolve(__dirname, "..")), maxInputBytes = 64 << 10, maxToolsBytes = 128 << 10, maxOutputBytes = 16 << 20;
const inputKeys = ["exportWorkspace", "hermesArchive", "workspace"], archiveKeys = ["originalRoot", "path", "rootName", "sha256"],
  toolKeys = {core: ["compiler", "dumpbin", "msbuild", "nodeGyp", "python"], io: ["cc", "cmake", "ninja"],
    openssl: ["ar", "cc", "link", "make", "perl", "rc", "tar"], hermes: ["cc", "cmake", "cxx", "git", "ninja"]};
/** @typedef {{workspace:string,exportWorkspace:string,hermesArchive:{path:string,sha256:string,rootName:string,originalRoot:string}}} WindowsSupportCommandInput */
/** @typedef {import('./build-local-native-windows-support').WindowsSupportTools} WindowsSupportTools */

/** @param {string} left @param {string} right */
function samePath(left, right) {return process.platform === "win32" ? left.toLowerCase() === right.toLowerCase() : left === right;}
/** @param {string} a @param {string} b */
function overlaps(a, b) {
  /** @param {string} parent @param {string} child */
  const contains = (parent, child) => {const rel = path.relative(parent, child); return !rel || !path.isAbsolute(rel) && rel !== ".." && !rel.startsWith(".." + path.sep);};
  return contains(a, b) || contains(b, a);
}
/** @param {unknown} value @param {string[]} keys @param {string} label */
function exactObject(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).sort().join(",") !== [...keys].sort().join(",")) {
    throw new Error(label + " has unknown or missing fields");
  }
  return /** @type {Record<string, any>} */ (value);
}
/** @param {string} file @param {string} label */
function canonicalJsonFile(file, label) {
  if (typeof file !== "string" || !path.isAbsolute(file) || path.resolve(file) !== file || /[\r\n\0]/.test(file) || overlaps(file, repo)) {
    throw new Error(label + " must be an absolute canonical path outside the checkout");
  }
  const parent = path.dirname(file), parentStat = fs.lstatSync(parent, {throwIfNoEntry: false}), stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!parentStat?.isDirectory() || parentStat.isSymbolicLink() || !samePath(fs.realpathSync(parent), parent) ||
      !stat?.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size > maxInputBytes || !samePath(fs.realpathSync(file), file)) {
    throw new Error(label + " must be a bounded, canonical, unaliased regular JSON file");
  }
  return stat;
}
/** @param {fs.Stats} left @param {fs.Stats} right */
function sameSnapshot(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.nlink === right.nlink && left.size === right.size &&
    left.mtimeMs === right.mtimeMs && left.ctimeMs === right.ctimeMs;
}
/** @param {string} file */
function readStableInput(file) {
  const before = canonicalJsonFile(file, "Windows support command input"), fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
  let bytes;
  try {
    const opened = fs.fstatSync(fd);
    if (!opened.isFile() || !sameSnapshot(before, opened)) {throw new Error("Windows support command input changed while opening");}
    // Bound allocation even if a writer grows the file after the size check.
    const buffer = Buffer.alloc(maxInputBytes + 1); let offset = 0;
    while (offset < buffer.length) {
      const count = fs.readSync(fd, buffer, offset, buffer.length - offset, null);
      if (!count) {break;}
      offset += count;
    }
    bytes = buffer.subarray(0, offset);
    const after = fs.fstatSync(fd), currentPath = fs.lstatSync(file, {throwIfNoEntry: false});
    if (bytes.length > maxInputBytes || !currentPath?.isFile() || currentPath.isSymbolicLink() || currentPath.nlink !== 1 ||
        !sameSnapshot(before, after) || !sameSnapshot(before, currentPath) || !samePath(fs.realpathSync(file), file)) {
      throw new Error("Windows support command input changed during its bounded read");
    }
  } finally {fs.closeSync(fd);}
  let value;
  try {value = JSON.parse(new TextDecoder("utf-8", {fatal: true}).decode(bytes));} catch (error) {throw new Error("Windows support command input must be valid UTF-8 JSON", {cause: error});}
  const parsed = exactObject(value, inputKeys, "Windows support command input");
  const archive = exactObject(parsed["hermesArchive"], archiveKeys, "Original Hermes archive descriptor");
  for (const [name, file] of Object.entries({workspace: parsed["workspace"], exportWorkspace: parsed["exportWorkspace"], archive: archive["path"]})) {
    if (typeof file !== "string" || !path.isAbsolute(file) || /[\r\n\0]/.test(file) || file.length > 4096) {
      throw new Error("Windows support command " + name + " path must be absolute and bounded");
    }
  }
  if (typeof archive["sha256"] !== "string" || !/^[a-f0-9]{64}$/.test(archive["sha256"]) ||
      typeof archive["rootName"] !== "string" || !archive["rootName"] || archive["rootName"].length > 4096 || /[\r\n\0]/.test(archive["rootName"]) ||
      typeof archive["originalRoot"] !== "string" || !archive["originalRoot"] || archive["originalRoot"].length > 4096 || /[\r\n\0]/.test(archive["originalRoot"])) {
    throw new Error("Original Hermes archive descriptor has invalid pins or source provenance");
  }
  return {input: /** @type {WindowsSupportCommandInput} */ (parsed), sha256: createHash("sha256").update(bytes).digest("hex"), stat: before};
}
/** @param {string} file @param {{sha256:string,stat:fs.Stats}} snapshot */
function assertStableInput(file, snapshot) {
  const current = canonicalJsonFile(file, "Windows support command input");
  if (!sameSnapshot(snapshot.stat, current) || fileHash(file) !== snapshot.sha256) {
    throw new Error("Windows support command input changed during the native build");
  }
}
/** @param {Buffer} bytes @param {number} limit @param {string} label */
function decodeBoundedJson(bytes, limit, label) {
  if (bytes.length === 0 || bytes.length > limit) {throw new Error(label + " is empty or exceeds its byte bound");}
  let value;
  try {value = JSON.parse(new TextDecoder("utf-8", {fatal: true}).decode(bytes));} catch (error) {throw new Error(label + " must be valid UTF-8 JSON", {cause: error});}
  return value;
}
/** @param {unknown} value */
function validateToolsJson(value) {
  const tools = exactObject(value, Object.keys(toolKeys), "Native Windows tools JSON");
  for (const [role, keys] of Object.entries(toolKeys)) {
    const selected = exactObject(tools[role], keys, "Native Windows " + role + " tools");
    for (const [name, file] of Object.entries(selected)) {
      if (typeof file !== "string" || file.length > 4096 || /[\r\n\0]/.test(file) || !path.win32.isAbsolute(file)) {
        throw new Error("Native Windows " + role + " tool " + name + " must be a bounded absolute Windows path");
      }
    }
  }
  return /** @type {WindowsSupportTools} */ (tools);
}
/** @param {NodeJS.ReadableStream & {isTTY?:boolean}} stream */
async function readBoundedTools(stream) {
  if (Reflect.get(stream, "isTTY") === true) {throw new Error("Native Windows tools JSON must be piped to stdin");}
  const chunks = []; let bytes = 0;
  for await (const chunk of stream) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    bytes += buffer.length;
    if (bytes > maxToolsBytes) {throw new Error("Native Windows tools JSON exceeds its byte bound");}
    chunks.push(buffer);
  }
  return validateToolsJson(decodeBoundedJson(Buffer.concat(chunks), maxToolsBytes, "Native Windows tools JSON"));
}
/** @param {string} file @param {string} inputFile @param {WindowsSupportCommandInput} input */
function validateNewOutput(file, inputFile, input) {
  if (typeof file !== "string" || !path.isAbsolute(file) || path.resolve(file) !== file || /[\r\n\0]/.test(file) || overlaps(file, repo) ||
      overlaps(file, inputFile) || [input["workspace"], input["exportWorkspace"], input["hermesArchive"].path].some(root => typeof root !== "string" || overlaps(file, root))) {
    throw new Error("Windows support result path must be a new canonical file outside the checkout and inputs");
  }
  const parent = path.dirname(file), stat = fs.lstatSync(parent, {throwIfNoEntry: false});
  if (!stat?.isDirectory() || stat.isSymbolicLink() || !samePath(fs.realpathSync(parent), parent) || fs.lstatSync(file, {throwIfNoEntry: false})) {
    throw new Error("Windows support result path must be new under a canonical existing directory");
  }
}
/** @param {unknown} value */
function validateCommandResult(value) {
  const result = exactObject(value, ["controllerSupport", "coreArchive", "coreReceipt", "exportRetirementCandidates", "supportRetirementCandidates"],
      "Windows support command result"),
    controller = exactObject(result["controllerSupport"], ["hermes", "io", "localBuildId"], "Controller support result"),
    localBuildId = controller["localBuildId"];
  if (typeof localBuildId !== "string" || !/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(localBuildId)) {
    throw new Error("Windows support command result has an invalid local build identity");
  }
  for (const kind of /** @type {const} */ (["io", "hermes"])) {
    const row = exactObject(controller[kind], ["admitted", "handoffSha256", "root"], "Controller " + kind + " result"),
      admitted = exactObject(row["admitted"], ["kind", "localBuildId"], "Controller " + kind + " admission");
    if (admitted["kind"] !== kind || admitted["localBuildId"] !== localBuildId || typeof row["root"] !== "string" || !path.isAbsolute(row["root"]) ||
        typeof row["handoffSha256"] !== "string" || !/^[a-f0-9]{64}$/.test(row["handoffSha256"])) {
      throw new Error("Windows support command result has an invalid " + kind + " handoff reference");
    }
  }
  if (typeof result["coreArchive"] !== "string" || !path.isAbsolute(result["coreArchive"]) ||
      !Array.isArray(result["exportRetirementCandidates"])) {throw new Error("Windows support command result is incomplete");}
  exactObject(result["coreReceipt"], ["path", "sha256"], "Windows Core receipt reference");
  return result;
}
/** @param {string} file @param {unknown} value */
function writeExclusiveJson(file, value) {
  const bytes = Buffer.from(JSON.stringify(value) + "\n", "utf8");
  if (bytes.length > maxOutputBytes) {throw new Error("Windows support command result exceeds its byte bound");}
  let fd, owned;
  try {
    fd = fs.openSync(file, "wx", 0o600); owned = fs.fstatSync(fd);
    let offset = 0;
    while (offset < bytes.length) {
      const count = fs.writeSync(fd, bytes, offset, bytes.length - offset);
      if (count <= 0) {throw new Error("Unable to complete Windows support command result write");}
      offset += count;
    }
    fs.fsyncSync(fd); fs.closeSync(fd); fd = undefined;
    const written = fs.lstatSync(file);
    if (!written.isFile() || written.isSymbolicLink() || written.nlink !== 1 || written.dev !== owned.dev || written.ino !== owned.ino ||
        written.size !== bytes.length || !samePath(fs.realpathSync(file), file)) {
      throw new Error("Windows support result file changed during its exclusive write");
    }
  } catch (error) {
    if (fd !== undefined) {try {fs.closeSync(fd);} catch { /* best-effort close before rollback */}}
    if (owned) {
      const current = fs.lstatSync(file, {throwIfNoEntry: false});
      if (current?.isFile() && current.dev === owned.dev && current.ino === owned.ino) {fs.unlinkSync(file);}
    }
    throw error;
  }
}

/** Strict process boundary for the local PowerShell bridge; no command text is evaluated.
 * @param {string[]} argv @param {{stdin?:(NodeJS.ReadableStream & {isTTY?:boolean}),buildCommand?:(input:any,dependencies?:any)=>Promise<any>,buildDependencies?:any}} [dependencies] */
async function runLocalWindowsSupportCommandCli(argv, dependencies = {}) {
  if (!Array.isArray(argv) || argv.length !== 3 || argv[0] !== "--run") {
    throw new Error("Usage: build-local-native-windows-support-command.js --run <input.json> <new-output.json>");
  }
  const inputFile = argv[1], outputFile = argv[2];
  if (typeof inputFile !== "string" || typeof outputFile !== "string") {throw new Error("Windows support command paths must be supplied");}
  const snapshot = readStableInput(inputFile);
  validateNewOutput(outputFile, inputFile, snapshot.input);
  const tools = await readBoundedTools(dependencies.stdin || process.stdin), request = {...snapshot.input, tools},
    result = validateCommandResult(await (dependencies.buildCommand || buildLocalWindowsSupportCommand)(request, dependencies.buildDependencies));
  assertStableInput(inputFile, snapshot);
  writeExclusiveJson(outputFile, result);
  return result;
}

/** @param {string} root @param {string} label */
function sourceDirectory(root, label) {
  const stat = fs.lstatSync(root, {throwIfNoEntry: false});
  if (stat && (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(root).toLowerCase() !== path.resolve(root).toLowerCase())) {
    throw new Error(label + " is redirected or not a directory");
  }
  return Boolean(stat);
}

/** Prepare the native archive cohorts and the verified transported bare Hermes source.
 * No receipt is converted from POSIX to Windows paths or regenerated on warm reuse.
 * @param {{workspace:string,exportWorkspace:string,hermesArchive:{path:string,sha256:string,rootName:string,originalRoot:string},
 * tools:import('./build-local-native-windows-support').WindowsSupportTools}} input
 * @param {{platform?:NodeJS.Platform,admitWorkspace?:typeof support.admitWindowsSupportWorkspace,
 * validateTools?:typeof support.validateWindowsSupportTools,
 * fetchIo?:typeof sources.fetchIoArchives,verifyIo?:typeof sources.verifyIoArchives,
 * fetchOpenSsl?:typeof sources.fetchWindowsOpenSslArchives,verifyOpenSsl?:typeof sources.verifyWindowsOpenSslArchives,
 * extractHermes?:typeof ci.extractWindowsHermesSource,readHermes?:(state:any,originalRoot:string)=>any,
 * buildSupport?:typeof support.buildLocalNativeWindowsSupport,exportSupport?:(value:any,options:any)=>any,retireSupport?:(candidates:any)=>any,
 * environment?:NodeJS.ProcessEnv}} [dependencies] */
async function buildLocalWindowsSupportCommand(input, dependencies = {}) {
  if ((dependencies.platform || process.platform) !== "win32" || process.arch !== "x64") {
    throw new Error("Local native Windows support production requires Windows x64");
  }
  if (!input || Object.keys(input).sort().join(",") !== "exportWorkspace,hermesArchive,tools,workspace" ||
      !input.hermesArchive || Object.keys(input.hermesArchive).sort().join(",") !== "originalRoot,path,rootName,sha256" ||
      !/^[a-f0-9]{64}$/.test(input.hermesArchive.sha256 || "") ||
      !ci.windowsHermesSourceName(input.hermesArchive.rootName) || !ci.canonicalHermesOriginalRoot(input.hermesArchive.originalRoot) ||
      ![input.workspace, input.exportWorkspace, input.hermesArchive.path].every(file => typeof file === "string" && path.isAbsolute(file) && !/[\r\n\0]/.test(file))) {
    throw new Error("Local Windows support needs explicit workspaces, tools and the original pinned Hermes source archive");
  }
  const relative = path.relative(input.workspace, input.exportWorkspace), reverse = path.relative(input.exportWorkspace, input.workspace),
    separate = (/** @type {string} */ value) => path.isAbsolute(value) || value === ".." || value.startsWith(".." + path.sep);
  if (!separate(relative) || !separate(reverse)) {throw new Error("Windows producer and selected-export workspaces must be disjoint");}
  const environment = /** @type {NodeJS.ProcessEnv} */ ({...process.env, ...(dependencies.environment || {}),
      MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2", OMP_NUM_THREADS: "2"}),
    tools = (dependencies.validateTools || support.validateWindowsSupportTools)(input.tools, "win32"),
    paths = (dependencies.admitWorkspace || support.admitWindowsSupportWorkspace)(input.workspace), root = path.join(paths.workspace, "sources");
  if (!sourceDirectory(root, "Windows support source cache")) {fs.mkdirSync(root, {mode: 0o700});}
  const ioRoot = path.join(root, "io"), sslRoot = path.join(root, "openssl"),
    hermesRoot = path.join(root, "hermes-" + input.hermesArchive.sha256), archive = input.hermesArchive.path,
    archiveStat = fs.lstatSync(archive);
  if (!archiveStat.isFile() || archiveStat.isSymbolicLink() || archiveStat.size < 1024 || archiveStat.size > 2 * 1024 ** 3 ||
      fs.realpathSync(archive).toLowerCase() !== path.resolve(archive).toLowerCase() || fileHash(archive) !== input.hermesArchive.sha256) {
    throw new Error("Transported Hermes source archive differs from its original WSL pin");
  }
  if (!sourceDirectory(ioRoot, "Windows I/O source cache")) {await (dependencies.fetchIo || sources.fetchIoArchives)(ioRoot, environment, true);}
  const io = (dependencies.verifyIo || sources.verifyIoArchives)(ioRoot);
  if (!sourceDirectory(sslRoot, "Windows OpenSSL source cache")) {await (dependencies.fetchOpenSsl || sources.fetchWindowsOpenSslArchives)(sslRoot, environment, true);}
  const ssl = (dependencies.verifyOpenSsl || sources.verifyWindowsOpenSslArchives)(sslRoot),
    hermesState = {kind: /** @type {"hermes"} */ ("hermes"), hermesSourceTar: archive,
      hermesSourceName: input.hermesArchive.rootName, hermesExtractRoot: hermesRoot},
    hermes = sourceDirectory(hermesRoot, "Windows Hermes source cache") ?
      (dependencies.readHermes || ci.readWindowsHermesSource)({hermesExtractRoot: hermesRoot,
        hermesSourceName: input.hermesArchive.rootName}, input.hermesArchive.originalRoot) :
      (dependencies.extractHermes || ci.extractWindowsHermesSource)(hermesState,
        path.join(environment["SystemRoot"] || "C:\\Windows", "System32/tar.exe"), 5 * 60 * 1000, input.hermesArchive.originalRoot),
    libuv = io.archives["libuv"]?.path, zlib = io.archives["zlib"]?.path, openssl = ssl.archives["openssl"]?.path;
  if (!libuv || !zlib || !openssl || path.basename(openssl) !== "openssl-" + opensslPin.version + ".tar.gz" ||
      fileHash(archive) !== input.hermesArchive.sha256) {throw new Error("Windows support source cohort changed during preparation");}
  const built = (dependencies.buildSupport || support.buildLocalNativeWindowsSupport)({workspace: input.workspace,
      sources: {libuv, zlib, openssl, hermes: hermes.source}, tools}, {environment}),
    exported = (dependencies.exportSupport || exporter.exportOrAdmitLocalWindowsSupport)(built, {workspace: input.exportWorkspace,
      sourceReceipts: {ioFetch: path.join(ioRoot, "fetch.json"), opensslFetch: path.join(sslRoot, "fetch.json"),
        hermesFetch: hermes.fetchFile, hermesRoots: hermes.rootsFile}});
  if (fileHash(archive) !== input.hermesArchive.sha256) {throw new Error("Original Hermes archive changed during Windows support build");}
  /** Only location/identity crosses the process boundary; WSL re-admits the full handoffs.
   * @param {any} row @param {"io"|"hermes"} kind */
  const reference = (row, kind) => ({root: row.root, handoffSha256: row.handoffSha256,
    admitted: {kind, localBuildId: exported.localBuildId}});
  return {controllerSupport: {localBuildId: exported.localBuildId, io: reference(exported.io, "io"), hermes: reference(exported.hermes, "hermes")},
    coreArchive: built.core.archive, coreReceipt: {path: built.core.receipt, sha256: fileHash(built.core.receipt)},
    supportRetirementCandidates: (dependencies.retireSupport || support.retireManagedOutputs)(built.managedOutputCandidates),
    exportRetirementCandidates: exported.retirementCandidates};
}

module.exports = {sourceDirectory, buildLocalWindowsSupportCommand, runLocalWindowsSupportCommandCli};
if (require.main === module) {
  runLocalWindowsSupportCommandCli(process.argv.slice(2)).catch(error => {
    console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1;
  });
}
