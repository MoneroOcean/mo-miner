"use strict";

// Reconstruct the selected Windows host SDK roots from the exact locked archive cohort.
// This is extraction/composition only: it does not install Microsoft tooling or admit an app.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {buildPathsOverlap, fileHash, sourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {windowsHostSdkCatalog, verifyWindowsHostSdkArchives} = require("./fetch-native-sdk-sources");
const windowsIntelRuntimeCatalog = require("../runtime/windows-intel-runtime.json");
const repo = path.resolve(__dirname, ".."), MAX_ENTRIES = 50000, MAX_EXPANDED = 2 * 1024 * 1024 * 1024,
  MAX_INPUT = 384 * 1024 * 1024, MAX_TOOL_OUTPUT = 32 * 1024 * 1024, MAX_LISTINGS = 64 * 1024 * 1024,
  MAX_TABLE_OUTPUT = 128 * 1024 * 1024,
  MAX_TOOL_MS = 5 * 60 * 1000, MAX_WALL_MS = 45 * 60 * 1000;
const EXPECTED_ROOTS = Object.freeze(["sdk/include/ucrt", "sdk/include/shared", "sdk/include/um", "sdk/include/winrt",
  "sdk/lib/ucrt", "sdk/lib/um", "msvc/include", "msvc/lib"]);
const INCLUDE_ROOTS = Object.freeze(["/mom-sdk/msvc/include", "/mom-sdk/sdk/include/ucrt", "/mom-sdk/sdk/include/shared",
  "/mom-sdk/sdk/include/um", "/mom-sdk/sdk/include/winrt"]);
const TOOL_PATHS = Object.freeze({sevenZip: "/usr/bin/7z", msiInfo: "/usr/bin/msiinfo", msiExtract: "/usr/bin/msiextract"});
const DEFAULT_MSVC_LIBRARIES = Object.freeze(["libcmt.lib", "libcpmt.lib", "oldnames.lib", "delayimp.lib"]);
const INTEL_MSVC_LINK_ORDER = Object.freeze(["libircmt.lib", "libmmt.lib", "svml_dispmt.lib", "libdecimal.lib", "libirc.lib"]);
const DEFAULT_UCRT_LIBRARIES = Object.freeze(["libvcruntime.lib", "libucrt.lib"]);
const DEFAULT_WINDOWS_IMPORT_LIBRARIES = Object.freeze(["AdvAPI32.Lib", "Crypt32.Lib", "DbgHelp.Lib", "DnsAPI.Lib", "Gdi32.Lib",
  "iphlpapi.lib", "kernel32.Lib", "ncrypt.lib", "ntdll.lib", "Ole32.Lib", "powrprof.lib", "psapi.lib", "RpcRT4.Lib",
  "Secur32.Lib", "SetupAPI.Lib", "shell32.lib", "ShLwApi.Lib", "User32.Lib", "UserEnv.Lib", "Uuid.Lib", "Version.Lib",
  "winhttp.lib", "WinMM.Lib", "WS2_32.Lib", "WSock32.Lib", "OleAut32.Lib", "cfgmgr32.lib", "runtimeobject.lib"]);
/** @typedef {Record<string, string>} MsiTableRow */
/** @typedef {{columns: string[], rows: MsiTableRow[]}} MsiTable */
/** @typedef {{file: string, relative: string, bytes: number, sequence: number}} MsiFileRow */
/** @typedef {{path: string, directory: boolean, bytes: number}} ArchiveEntry */
/** @typedef {{type: string, physical: number, tail: number, entries: ArchiveEntry[], expandedBytes: number}} ArchiveListing */
/** @typedef {{path: string, full: string, bytes: number}} TreeFile */
/** @typedef {{files: TreeFile[], entries: number, bytes: number}} Tree */

/** The current admitted MT/full-native baseline. Return fresh arrays so callers can safely
 * layer an explicit project-specific policy without mutating this maintained default. */
function defaultWindowsNativeLibraryPolicy() {
  const catalogNames = Array.isArray(windowsIntelRuntimeCatalog.libraries) ?
    windowsIntelRuntimeCatalog.libraries.map(row => row?.name) : [];
  const expected = [...INTEL_MSVC_LINK_ORDER].sort();
  if (windowsIntelRuntimeCatalog.version !== 1 || catalogNames.length !== expected.length ||
      catalogNames.some(name => typeof name !== "string" || !/^[A-Za-z0-9_.+-]+\.lib$/i.test(name)) ||
      [...catalogNames].sort().join("\0") !== expected.join("\0")) {
    throw new Error("Windows Intel MT library order differs from the shared runtime catalog");
  }
  return {msvcLibraries: [...DEFAULT_MSVC_LIBRARIES, ...INTEL_MSVC_LINK_ORDER],
    ucrtLibraries: [...DEFAULT_UCRT_LIBRARIES], importLibraries: [...DEFAULT_WINDOWS_IMPORT_LIBRARIES]};
}

/** @param {unknown} value */
function safeRelative(value) {
  if (typeof value !== "string" || !value || value.length > 4096 || /[\0\r\n]/.test(value)) {throw new Error("Unsafe Windows SDK path");}
  const normalized = value.replace(/\\/g, "/");
  if (normalized.startsWith("/") || /^[A-Za-z]:/.test(normalized) || normalized.includes(":")) {throw new Error("Absolute/ADS Windows SDK path rejected");}
  const parts = normalized.split("/");
  if (parts.some(part => !part || part === "." || part === ".." || Buffer.byteLength(part) > 255 || /[<>|?*"]/.test(part) || /[. ]$/.test(part) ||
      /^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$/i.test(part))) {throw new Error("Unsafe Windows SDK path component");}
  return parts.join("/");
}

/** @param {string} value */
function fold(value) {return value.normalize("NFC").toLowerCase();}

/** Keep receipt ordering independent of the host locale. @param {string} a @param {string} b */
function compare(a, b) {return a < b ? -1 : a > b ? 1 : 0;}

/** MSI IDT exports have a column header, type row, primary-key declaration, and tabular data. @param {string} text @param {string} table */
function parseIdt(text, table) {
  if (typeof text !== "string" || Buffer.byteLength(text) > 8 * 1024 * 1024) {throw new Error("MSI table exceeds bounded metadata size: " + table);}
  const lines = text.replace(/\r\n/g, "\n").split("\n"); if (lines.at(-1) === "") {lines.pop();}
  if (lines.length < 3 || lines.length > MAX_ENTRIES + 3) {throw new Error("MSI table row count is invalid: " + table);}
  const columns = (lines[0] || "").split("\t"), types = (lines[1] || "").split("\t"), primary = (lines[2] || "").split("\t");
  if (columns.some(name => !name) || new Set(columns).size !== columns.length || types.length !== columns.length || primary[0] !== table ||
      primary.length < 2 || primary.slice(1).some(name => !columns.includes(name))) {throw new Error("Malformed MSI table schema: " + table);}
  const rows = lines.slice(3).map(line => {
    const cells = line.split("\t"); if (!line || cells.length !== columns.length) {throw new Error("Malformed MSI table row: " + table);}
    return Object.fromEntries(columns.map((name, index) => [name, cells[index] || ""]));
  });
  return /** @type {MsiTable} */ ({columns, rows});
}

/** MSI short|long names use the long spelling; Directory.DefaultDir may also have target:source.
 * @param {string} value @param {boolean} allowDot @param {boolean} directory */
function idtName(value, allowDot = false, directory = false) {
  if (!value) {throw new Error("Empty MSI path-table name");}
  const target = directory ? value.split(":", 1)[0] || "" : value;
  const name = target.includes("|") ? target.slice(target.lastIndexOf("|") + 1) : target;
  return allowDot && name === "." ? "" : safeRelative(name);
}

/** Resolve standard Windows Installer Directory/Component/File tables without executing the MSI.
 * @param {string} directoryText @param {string} componentText @param {string} fileText
 * @returns {{rows:MsiFileRow[],expandedBytes:number}} */
function buildMsiFilePaths(directoryText, componentText, fileText) {
  const dirs = parseIdt(directoryText, "Directory"), components = parseIdt(componentText, "Component"), files = parseIdt(fileText, "File");
  if (!["Directory", "Directory_Parent", "DefaultDir"].every(key => dirs.columns.includes(key)) ||
      !["Component", "Directory_"].every(key => components.columns.includes(key)) ||
      !["File", "Component_", "FileName", "FileSize", "Sequence"].every(key => files.columns.includes(key))) {throw new Error("Required MSI path columns are absent");}
  const byDirectory = new Map();
  for (const row of dirs.rows) {if (!row["Directory"] || byDirectory.has(row["Directory"])) {throw new Error("Duplicate MSI Directory identifier");} byDirectory.set(row["Directory"], row);}
  const resolved = new Map(), active = new Set();
  /** @param {string} id @returns {string} */
  function resolve(id) {
    if (resolved.has(id)) {return resolved.get(id);} const row = byDirectory.get(id);
    if (!row || active.has(id)) {throw new Error("Missing/cyclic MSI Directory parent");}
    active.add(id); const parent = row["Directory_Parent"] ? resolve(row["Directory_Parent"]) : "", child = idtName(row["DefaultDir"] || "", true, true);
    const value = [parent, child].filter(Boolean).join("/"); active.delete(id); resolved.set(id, value); return value;
  }
  for (const id of byDirectory.keys()) {resolve(id);}
  const byComponent = new Map();
  for (const row of components.rows) {if (!row["Component"] || byComponent.has(row["Component"]) || !resolved.has(row["Directory_"])) {throw new Error("Malformed MSI Component reference");} byComponent.set(row["Component"], row);}
  const seen = new Set(); let expandedBytes = 0;
  const rows = files.rows.map(row => {
    const component = byComponent.get(row["Component_"]), directory = component && resolved.get(component["Directory_"]);
    if (directory === undefined || !row["File"] || !/^\d+$/.test(row["FileSize"] || "") || !/^\d+$/.test(row["Sequence"] || "")) {throw new Error("Malformed MSI File row");}
    const relative = safeRelative([directory, idtName(row["FileName"] || "")].filter(Boolean).join("/")), key = fold(relative), bytes = Number(row["FileSize"]), sequence = Number(row["Sequence"]);
    if (seen.has(key) || !Number.isSafeInteger(bytes) || bytes < 0 || !Number.isSafeInteger(sequence) || sequence < 1) {throw new Error("Duplicate/invalid MSI File entry");}
    seen.add(key); expandedBytes += bytes;
    if (!Number.isSafeInteger(expandedBytes) || expandedBytes > MAX_EXPANDED) {throw new Error("MSI expanded size exceeds cap");}
    return {file: row["File"], relative, bytes, sequence};
  });
  const paths = new Set(rows.map(row => fold(row.relative)));
  for (const row of rows) {
    const parts = fold(row.relative).split("/");
    for (let index = 1; index < parts.length; index++) {if (paths.has(parts.slice(0, index).join("/"))) {throw new Error("MSI file path descends through another file");}}
  }
  return {rows, expandedBytes};
}

/** Parse a bounded 7-Zip -slt archive listing and reject unsafe or colliding entries before extraction.
 * @param {string} text @param {"zip"|"cab"} expectedType @returns {ArchiveListing} */
function parse7zListing(text, expectedType) {
  if (typeof text !== "string" || Buffer.byteLength(text) > MAX_TOOL_OUTPUT) {throw new Error("Archive listing exceeds output cap");}
  const blocks = text.replace(/\r\n/g, "\n").split(/\n\n+/), entries = [], seen = new Map(), directories = new Set(); let archive = null, expandedBytes = 0;
  for (const block of blocks) {
    if (!/^Path = /m.test(block)) {continue;}
    const fields = /** @type {Record<string, string>} */ (Object.create(null));
    for (const line of block.split("\n")) {
      const match = /^([^=]+) = (.*)$/.exec(line); if (!match) {continue;}
      const key = match[1]?.trim(), value = match[2];
      if (!key || value === undefined || Object.hasOwn(fields, key)) {throw new Error("Malformed/duplicate archive listing field");}
      fields[key] = value;
    }
    if (fields["Type"] && fields["Physical Size"]) {
      if (archive || entries.length || !path.isAbsolute(fields["Path"] || "") || fields["Type"].toLowerCase() !== expectedType) {throw new Error("Unexpected archive header/type");}
      const physical = Number(fields["Physical Size"]), tail = Number(fields["Tail Size"] || "0");
      if (!Number.isSafeInteger(physical) || physical < 0 || !Number.isSafeInteger(tail) || tail < 0 || (tail && expectedType !== "cab")) {throw new Error("Invalid archive physical size");}
      archive = {type: expectedType, physical, tail}; continue;
    }
    if (!archive) {throw new Error("Archive member precedes header");}
    if (["Symbolic Link", "Hard Link", "Reparse Point"].some(name => Object.hasOwn(fields, name)) || /^l/i.test(fields["Attributes"] || "")) {
      throw new Error("Archive link/reparse entry rejected");
    }
    let directory;
    if (fields["Folder"] === "+" || fields["Folder"] === "-") {directory = fields["Folder"] === "+";} else if (expectedType === "cab" && fields["Folder"] === undefined && fields["Attributes"] === "A") {directory = false;} else {throw new Error("Archive member has no explicit file/directory type");}
    if (expectedType === "cab" && (directory || fields["Attributes"] !== "A")) {throw new Error("Unexpected CAB member attributes");}
    const rawPath = fields["Path"] || "", member = safeRelative(directory ? rawPath.replace(/[\\/]$/, "") : rawPath);
    const key = fold(member), bytes = directory ? 0 : Number(fields["Size"]);
    if (seen.has(key) || (!directory && directories.has(key)) || !Number.isSafeInteger(bytes) || bytes < 0) {throw new Error("Duplicate or malformed archive member");}
    const parts = key.split("/");
    for (let index = 1; index < parts.length; index++) {if (seen.get(parts.slice(0, index).join("/")) === false) {throw new Error("Archive member descends through a file");}}
    for (let index = 1; index < parts.length; index++) {directories.add(parts.slice(0, index).join("/"));}
    if (directory) {directories.add(key);}
    seen.set(key, directory); expandedBytes += bytes;
    if (!Number.isSafeInteger(expandedBytes) || expandedBytes > MAX_EXPANDED || entries.length >= MAX_ENTRIES) {throw new Error("Archive exceeds entry/expanded-size cap");}
    entries.push({path: member, directory, bytes});
  }
  if (!archive || !entries.length) {throw new Error("Archive listing is incomplete");}
  return /** @type {ArchiveListing} */ ({...archive, entries, expandedBytes});
}

/** Match a validated extracted file against one fixed lock selection, case-insensitively.
 * @param {{sourcePrefix:string,destinationPrefix:string,kind:string}} selection @param {string} source @returns {string|null} */
function mapSelectedMember(selection, source) {
  const src = safeRelative(source), prefix = safeRelative(selection.sourcePrefix), dest = safeRelative(selection.destinationPrefix), sf = fold(src), pf = fold(prefix);
  if (selection.kind === "file") {return sf === pf ? dest : null;}
  const base = pf + "/"; if (!sf.startsWith(base)) {return null;}
  return safeRelative(dest + "/" + src.slice(prefix.length + 1));
}

/** MSI TARGETDIR is SourceDir in the IDT tables while msiextract may emit either that
 * symbolic root or its Windows Kits install path. Normalize only this pinned SDK subtree. @param {string} value */
function normalizeMsiSource(value) {
  let relative = safeRelative(value);
  if (relative.toLowerCase().startsWith("sourcedir/")) {relative = relative.slice("SourceDir/".length);}
  if (/^Program Files\//i.test(relative)) {relative = relative.slice("Program Files/".length);}
  if (!/^Windows Kits\/10\//i.test(relative)) {throw new Error("MSI member escaped the selected x64 Windows Kits source tree");}
  return "Program Files/" + relative;
}

/** Map one MSI file across its Media sequence and exact locked external cabinet relation.
 * @param {string} mediaText @param {string} msiKey @param {any} lock @param {MsiFileRow[]} fileRows
 * @returns {{diskId:string,firstSequence:number,lastSequence:number,cabinet:any,cabinetKey:string}[]} */
function validateMsiMedia(mediaText, msiKey, lock, fileRows) {
  const table = parseIdt(mediaText, "Media");
  if (!["DiskId", "LastSequence", "Cabinet"].every(key => table.columns.includes(key))) {throw new Error("MSI Media table columns are missing");}
  /** @type {{diskId:string,cabinet:string}[]} */ const expected = lock.msiCabinets[msiKey];
  const cabinets = new Map(expected.map(row => [row.diskId, lock.cabinets[row.cabinet]]));
  /** @type {{diskId:string,firstSequence:number,lastSequence:number,cabinet:any,cabinetKey:string}[]} */ const rows = [];
  /** @type {{firstSequence:number,lastSequence:number}[]} */ const ranges = [];
  const disks = new Set(); let previous = 0;
  for (const row of table.rows) {
    const diskId = row["DiskId"] || "", lastSequence = row["LastSequence"] || "", cabinetName = row["Cabinet"] || "";
    if (!/^\d+$/.test(diskId) || !/^\d+$/.test(lastSequence) || disks.has(diskId)) {throw new Error("Malformed MSI Media row");}
    const last = Number(lastSequence); if (last < previous) {throw new Error("MSI Media sequence is not increasing");}
    if (cabinetName && !cabinetName.startsWith("#")) {
      const relation = expected.find(item => item.diskId === diskId), cab = cabinets.get(diskId), vendorName = cab?.vendorFileName?.replace(/\\/g, "/").split("/").at(-1);
      const actualName = safeRelative(cabinetName.replace(/\\/g, "/")).split("/").at(-1);
      if (!cab || fold(vendorName || "") !== fold(actualName || "")) {throw new Error("MSI Media cabinet differs from lock: " + msiKey);}
      if (!relation) {throw new Error("MSI Media row is not bound to the fixed lock: " + msiKey);}
      rows.push({diskId, firstSequence: previous + 1, lastSequence: last, cabinet: cab, cabinetKey: relation.cabinet});
    } else if (cabinets.has(diskId)) {throw new Error("Locked external MSI cabinet is absent from Media table: " + msiKey);}
    ranges.push({firstSequence: previous + 1, lastSequence: last}); disks.add(diskId); previous = last;
  }
  if (rows.length !== expected.length || fileRows.some(file => !ranges.some(range => file.sequence >= range.firstSequence && file.sequence <= range.lastSequence))) {
    throw new Error("MSI Media rows do not cover exact locked cabinets/file sequences: " + msiKey);
  }
  return rows;
}

/** @param {string} root @returns {Tree} */
function walkTree(root) {
  /** @type {TreeFile[]} */ const files = []; const seen = new Set(); let entries = 0, bytes = 0;
  const rootStat = fs.lstatSync(root);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) {throw new Error("Extracted tree root is not a real directory");}
  /** @param {string} directory @param {string} prefix */
  function walk(directory, prefix) {
    for (const name of fs.readdirSync(directory).sort()) {
      const relative = safeRelative(prefix ? prefix + "/" + name : name), full = path.join(directory, name), st = fs.lstatSync(full);
      if (st.isSymbolicLink() || (!st.isDirectory() && !st.isFile()) || (st.isFile() && st.nlink !== 1)) {throw new Error("Extracted link/special/hard-linked path rejected");}
      const key = fold(relative); if (seen.has(key)) {throw new Error("Extracted case collision");} seen.add(key);
      if (++entries > MAX_ENTRIES) {throw new Error("Extracted tree exceeds entry cap");}
      if (st.isDirectory()) {walk(full, relative);} else {bytes += st.size; if (!Number.isSafeInteger(bytes) || bytes > MAX_EXPANDED) {throw new Error("Extracted tree exceeds expanded-byte cap");} files.push({path: relative, full, bytes: st.size});}
    }
  }
  walk(root, ""); return {files, entries, bytes};
}

/** @param {string} output @param {string} label @param {string} command @param {string[]} args @param {number} deadline @param {string} [cwd] */
function runTool(output, label, command, args, deadline, cwd) {
  const remain = Math.min(MAX_TOOL_MS, deadline - Date.now()); if (remain <= 0) {throw new Error("Windows SDK extraction wall deadline exceeded");}
  const result = spawnSync(command, args, {cwd, encoding: "utf8", timeout: remain, maxBuffer: MAX_TOOL_OUTPUT,
    env: {PATH: "/usr/bin:/bin", HOME: "/tmp", LC_ALL: "C", LANG: "C"}});
  const stdout = result.stdout || "", stderr = result.stderr || "";
  if (Buffer.byteLength(stdout) + Buffer.byteLength(stderr) > MAX_TOOL_OUTPUT || /\0/.test(stdout + stderr)) {throw new Error("Extraction tool output is malformed or oversized: " + label);}
  fs.writeFileSync(path.join(output, "logs", label + ".stdout.txt"), stdout, {flag: "wx"});
  fs.writeFileSync(path.join(output, "logs", label + ".stderr.txt"), stderr, {flag: "wx"});
  if (result.error || result.signal || result.status !== 0) {throw new Error("Windows SDK extraction tool failed: " + label);}
  return {stdout, stderr};
}

/** @param {string} executable @param {string[]} args @param {string} expect */
function toolIdentity(executable, args, expect) {
  const real = fs.realpathSync(executable), st = fs.statSync(real);
  if (!path.isAbsolute(real) || !st.isFile()) {throw new Error("Extraction tool is not a canonical regular file");}
  const result = spawnSync(real, args, {encoding: "utf8", timeout: 30000, maxBuffer: 64 * 1024,
    env: {PATH: "/usr/bin:/bin", HOME: "/tmp", LC_ALL: "C", LANG: "C"}});
  const output = (result.stdout || "") + (result.stderr || "");
  if (result.error || result.status !== 0 || !output.includes(expect)) {throw new Error("Unexpected Windows SDK extraction tool version: " + path.basename(executable));}
  return {path: real, sha256: fileHash(real), bytes: st.size, version: output.trim().slice(0, 256)};
}

/** @param {string} msi @param {string} table @param {string} output @param {number} deadline @param {{value:number}} total */
function exportMsiTable(msi, table, output, deadline, total) {
  const text = runTool(output, "msi-" + path.basename(msi, ".msi") + "-" + table.toLowerCase(), TOOL_PATHS.msiInfo, ["export", msi, table], deadline).stdout;
  total.value += Buffer.byteLength(text);
  if (!Number.isSafeInteger(total.value) || total.value > MAX_TABLE_OUTPUT) {throw new Error("Aggregate MSI table exports exceed bound");}
  return text;
}

/** @param {string} source @param {string} outputRoot @param {string} destination @param {Map<string, {path:string,sha256:string,bytes:number,origins:object[]}>} claims @param {Map<string,string>} directories @param {object} origin */
function addSelectedFile(source, outputRoot, destination, claims, directories, origin) {
  const rel = safeRelative(destination), st = fs.lstatSync(source);
  if (!st.isFile() || st.isSymbolicLink() || st.nlink !== 1) {throw new Error("Selected SDK file is not a private regular file");}
  const sha256 = fileHash(source), key = fold(rel), prior = claims.get(key);
  if (directories.has(key)) {throw new Error("SDK file collides with a case-insensitive directory: " + rel);}
  const parts = rel.split("/");
  for (let index = 1; index < parts.length; index++) {
    const directory = parts.slice(0, index).join("/"), directoryKey = fold(directory), spelling = directories.get(directoryKey);
    if (claims.has(directoryKey) || (spelling !== undefined && spelling !== directory)) {throw new Error("SDK destination has a case/file path collision: " + rel);}
    directories.set(directoryKey, directory);
  }
  if (prior) {
    if (prior.sha256 !== sha256 || prior.bytes !== st.size) {throw new Error("Conflicting case-insensitive SDK destination: " + rel);}
    prior.origins.push(origin); return;
  }
  const target = path.join(outputRoot, ...rel.split("/"));
  fs.mkdirSync(path.dirname(target), {recursive: true});
  if (fs.existsSync(target)) {throw new Error("Unexpected preexisting SDK output path");}
  fs.copyFileSync(source, target, fs.constants.COPYFILE_EXCL); fs.chmodSync(target, 0o644); fs.utimesSync(target, 0, 0);
  if (fileHash(target) !== sha256) {throw new Error("SDK selected-file copy failed hash verification");}
  claims.set(key, {path: rel, sha256, bytes: st.size, origins: [origin]});
}

/** Use the first physical spelling retained by composition, not a later same-byte alias.
 * @param {{path:string}[]} files @param {string} sourceRoot */
function headerOverlayEntries(files, sourceRoot) {
  return files.filter(row => row.path.startsWith("msvc/include/") || row.path.startsWith("sdk/include/"))
    .map(row => ({type: "file", name: "/mom-sdk/" + row.path, "external-contents": path.join(sourceRoot, ...row.path.split("/"))}));
}

/** @param {string} archiveRoot @param {string} work
 * @returns {{lock:any,receiptPath:string,receiptHash:string,archives:Record<string,{path:string,sha256:string,bytes:number,staged:string}>,totalBytes:number}} */
function verifyArchiveInputs(archiveRoot, work) {
  const receiptPath = path.join(archiveRoot, "fetch.json"), receiptHash = fileHash(receiptPath),
    receipt = /** @type {any} */ (verifyWindowsHostSdkArchives(archiveRoot)), lock = /** @type {any} */ (windowsHostSdkCatalog());
  if (receipt.version !== 1 || receipt.target !== "windows-host" || receipt.status !== "PASS" || receipt.buildOnly !== true ||
      receipt.compiled !== false || receipt.publicationQualified !== false) {
    throw new Error("Windows host-SDK archive receipt scope is invalid");
  }
  const archiveRows = {...Object.fromEntries(Object.entries(lock.packages).map(([key, row]) => ["package." + key, row])),
    ...Object.fromEntries(Object.entries(lock.cabinets).map(([key, row]) => ["cab." + key, row]))};
  if (!isDeepStrictEqual(Object.keys(receipt.archives).sort(), Object.keys(archiveRows).sort())) {throw new Error("Windows SDK receipt archive set differs from lock");}
  let totalBytes = 0;
  /** @type {Record<string,{path:string,sha256:string,bytes:number,staged:string}>} */ const archives = {};
  for (const [key, row] of Object.entries(archiveRows).sort(([a], [b]) => compare(a, b))) {
    const item = receipt.archives[key], file = item?.path;
    if (typeof file !== "string") {throw new Error("Windows SDK archive receipt path is absent: " + key);}
    const st = fs.lstatSync(file);
    if (item.url !== row.url || item.sha256 !== row.sha256 || item.bytes !== st.size || st.size > row.maxBytes ||
        !st.isFile() || st.isSymbolicLink() || st.nlink !== 1 || fileHash(file) !== row.sha256 || path.dirname(file) !== fs.realpathSync(archiveRoot) ||
        path.basename(file) !== row.fileName) {throw new Error("Windows SDK archive receipt row differs from locked bytes: " + key);}
    totalBytes += st.size; if (!Number.isSafeInteger(totalBytes) || totalBytes > MAX_INPUT) {throw new Error("Windows SDK archive inputs exceed aggregate cap");}
    const staged = path.join(work, row.fileName); fs.copyFileSync(file, staged, fs.constants.COPYFILE_EXCL);
    if (fileHash(staged) !== row.sha256) {throw new Error("Staged Windows SDK archive hash mismatch");}
    archives[key] = {path: file, sha256: row.sha256, bytes: st.size, staged};
  }
  if (fileHash(receiptPath) !== receiptHash) {throw new Error("Windows SDK fetch receipt changed during verification");}
  return {lock, receiptPath, receiptHash, archives, totalBytes};
}

/** @param {string} archiveRoot @param {string} output @param {NodeJS.ProcessEnv} [environment]
 * @returns {any} */
function prepareWindowsHostSdk(archiveRoot, output, environment = process.env) {
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows host SDK preparation requires Linux and MOM_AOT_BUILD_ONLY=1");}
  if (![archiveRoot, output].every(value => typeof value === "string" && path.isAbsolute(value) && !/[\r\n\0]/.test(value))) {throw new Error("Windows host SDK paths must be absolute");}
  const archiveReal = fs.realpathSync(archiveRoot), parent = fs.realpathSync(path.dirname(output)), destination = path.join(parent, path.basename(output));
  const work = destination + ".work";
  if (destination !== path.resolve(output) || destination === repo || destination.startsWith(repo + path.sep) ||
      (() => {try {fs.lstatSync(destination); return true;} catch {return false;}})() ||
      (() => {try {fs.lstatSync(work); return true;} catch {return false;}})()) {throw new Error("Windows host SDK output/work paths must be new and outside the repository");}
  if (buildPathsOverlap(archiveReal, destination) || buildPathsOverlap(archiveReal, work)) {throw new Error("Windows host SDK output overlaps archive inputs");}
  const recipeLeaves = ["scripts/prepare-native-windows-sdk.js", "scripts/fetch-native-sdk-sources.js", "scripts/build-source-inputs.js", "runtime/windows-host-sdk.json"];
  const recipe = sourceInputs(repo, [], recipeLeaves), fetchReceipt = path.join(archiveReal, "fetch.json");
  const sourceReceiptHash = fileHash(fetchReceipt), toolRows = [
    toolIdentity(TOOL_PATHS.sevenZip, ["i"], "7-Zip 23.01"), toolIdentity(TOOL_PATHS.msiInfo, ["--version"], "0.103"),
    toolIdentity(TOOL_PATHS.msiExtract, ["--version"], "0.103")];
  fs.mkdirSync(destination, {mode: 0o700}); fs.mkdirSync(path.join(destination, "logs")); fs.mkdirSync(work, {mode: 0o700});
  const deadline = Date.now() + MAX_WALL_MS, listingBytes = {value: 0};
  const msiTableBytes = {value: 0};
  /** @type {{archive:Record<string,ArchiveListing>,msi:Record<string,{rows:MsiFileRow[],expandedBytes:number,mediaRows:{diskId:string,firstSequence:number,lastSequence:number,cabinet:any,cabinetKey:string}[]}>}} */
  const metadata = {archive: {}, msi: {}};
  const inputs = verifyArchiveInputs(archiveReal, work);
  if (!isDeepStrictEqual(inputs.lock.roots, EXPECTED_ROOTS)) {throw new Error("Windows host SDK lock root selection changed");}
  /** @param {string} name @param {string} key @param {"zip"|"cab"} type @returns {ArchiveListing} */
  const listingFor = (name, key, type) => {
    const file = inputs.archives[key]?.staged; if (!file) {throw new Error("Locked SDK archive is absent: " + key);}
    const outputRows = runTool(destination, "list-" + name, TOOL_PATHS.sevenZip, ["l", "-slt", file], deadline), bytes = Buffer.byteLength(outputRows.stdout);
    listingBytes.value += bytes; if (listingBytes.value > MAX_LISTINGS) {throw new Error("Aggregate archive listings exceed bound");}
    const listing = parse7zListing(outputRows.stdout, type), size = fs.statSync(file).size;
    if (listing.physical + listing.tail !== size) {throw new Error("Archive listing physical/tail size differs from locked bytes");}
    return listing;
  };
  for (const [key, row] of Object.entries(inputs.lock.packages)) {
    if (row.format === "vsix") {metadata.archive["package." + key] = listingFor("package-" + key, "package." + key, "zip");} else {
      const packageArchive = inputs.archives["package." + key];
      if (!packageArchive) {throw new Error("Locked Windows SDK MSI is absent: " + key);}
      const directory = exportMsiTable(packageArchive.staged, "Directory", destination, deadline, msiTableBytes),
        component = exportMsiTable(packageArchive.staged, "Component", destination, deadline, msiTableBytes),
        files = exportMsiTable(packageArchive.staged, "File", destination, deadline, msiTableBytes),
        media = exportMsiTable(packageArchive.staged, "Media", destination, deadline, msiTableBytes);
      const projection = buildMsiFilePaths(directory, component, files), mediaRows = validateMsiMedia(media, key, inputs.lock, projection.rows);
      metadata.msi[key] = {...projection, mediaRows};
    }
  }
  for (const key of Object.keys(inputs.lock.cabinets)) {metadata.archive["cab." + key] = listingFor("cab-" + key, "cab." + key, "cab");}
  let listedArchiveBytes = 0, listedArchiveEntries = 0, msiTableBytesDeclared = 0, msiTableFiles = 0;
  for (const listing of Object.values(metadata.archive)) {
    listedArchiveBytes += listing.expandedBytes; listedArchiveEntries += listing.entries.length;
  }
  for (const msi of Object.values(metadata.msi)) {
    msiTableBytesDeclared += msi.expandedBytes; msiTableFiles += msi.rows.length;
  }
  if (!Number.isSafeInteger(listedArchiveBytes) || listedArchiveBytes > MAX_EXPANDED || listedArchiveEntries > MAX_ENTRIES ||
      !Number.isSafeInteger(msiTableBytesDeclared) || msiTableBytesDeclared > MAX_EXPANDED || msiTableFiles > MAX_ENTRIES) {
    throw new Error("Aggregate Windows SDK archive metadata exceeds expansion/entry bounds");
  }
  if (sourceReceiptHash !== fileHash(fetchReceipt)) {throw new Error("Windows SDK input receipt changed before extraction");}
  /** @type {Map<string,{package:string,sourcePrefix:string,destinationPrefix:string,kind:string}[]>} */ const selectedPackages = new Map();
  /** @type {{package:string,sourcePrefix:string,destinationPrefix:string,kind:string}[]} */ const fixedSelections = inputs.lock.selections;
  for (const selection of fixedSelections) {const rows = selectedPackages.get(selection.package) || []; rows.push(selection); selectedPackages.set(selection.package, rows);}
  const extractionTotals = {files: 0, entries: 0, bytes: 0};
  /** @type {Map<string,{path:string,sha256:string,bytes:number,origins:object[]}>} */ const claims = new Map();
  /** @type {Map<string,string>} */ const directoryClaims = new Map();
  /** @type {object[]} */ const provenance = [];
  const selectorHits = new Map(fixedSelections.map(selection =>
    [[selection.package, selection.sourcePrefix, selection.destinationPrefix, selection.kind].join("\0"), 0]));
  for (const [key, row] of Object.entries(inputs.lock.packages).sort(([a], [b]) => compare(a, b))) {
    const selections = selectedPackages.get(key) || [], archive = inputs.archives["package." + key], treeRoot = path.join(work, "extract-" + key);
    if (!archive) {throw new Error("Locked Windows SDK package is absent: " + key);}
    fs.mkdirSync(treeRoot);
    if (row.format === "vsix") {
      const listed = metadata.archive["package." + key];
      if (!listed) {throw new Error("VSIX archive was not preflighted: " + key);}
      runTool(destination, "extract-package-" + key, TOOL_PATHS.sevenZip, ["x", "-y", "-o" + treeRoot, archive.staged], deadline);
      const tree = walkTree(treeRoot), actual = new Map(tree.files.map(file => [fold(file.path), file]));
      const expected = listed.entries.filter(entry => !entry.directory);
      if (actual.size !== expected.length || expected.some(entry => actual.get(fold(entry.path))?.bytes !== entry.bytes)) {throw new Error("VSIX extracted files differ from preflight listing: " + key);}
      extractionTotals.files += tree.files.length; extractionTotals.entries += tree.entries; extractionTotals.bytes += tree.bytes;
    } else {
      const msi = metadata.msi[key]; if (!msi) {throw new Error("MSI was not preflighted: " + key);}
      const staged = path.join(work, "msi-" + key); fs.mkdirSync(staged);
      const msiPath = path.join(staged, row.fileName); fs.copyFileSync(archive.staged, msiPath, fs.constants.COPYFILE_EXCL);
      for (const media of msi.mediaRows) {
        const cab = media.cabinet, source = inputs.archives["cab." + media.cabinetKey]?.staged;
        if (!source) {throw new Error("MSI locked cabinet input is absent: " + key);}
        const cabinetName = cab.vendorFileName.replace(/\\/g, "/").split("/").at(-1), target = path.join(staged, cabinetName || "");
        if (fs.existsSync(target)) {throw new Error("Duplicate staged MSI CAB name");} fs.copyFileSync(source, target, fs.constants.COPYFILE_EXCL);
      }
      runTool(destination, "extract-msi-" + key, TOOL_PATHS.msiExtract, ["--directory", treeRoot, msiPath], deadline, staged);
      const tree = walkTree(treeRoot), expected = new Map(msi.rows.map(item => [fold(normalizeMsiSource(item.relative)), item]));
      if (tree.files.length !== expected.size || tree.files.some(item => expected.get(fold(normalizeMsiSource(item.path)))?.bytes !== item.bytes)) {throw new Error("MSI extracted files differ from File/Component/Directory tables: " + key);}
      extractionTotals.files += tree.files.length; extractionTotals.entries += tree.entries; extractionTotals.bytes += tree.bytes;
    }
    if (extractionTotals.files > MAX_ENTRIES || extractionTotals.entries > MAX_ENTRIES || extractionTotals.bytes > MAX_EXPANDED) {throw new Error("Aggregate Windows SDK extraction exceeds limits");}
    const tree = walkTree(treeRoot);
    for (const file of tree.files) {
      const source = row.format === "msi" ? normalizeMsiSource(file.path) : file.path;
      for (const selection of selections) {
        const target = mapSelectedMember(selection, source);
        if (target === null) {continue;}
        const selectorKey = [selection.package, selection.sourcePrefix, selection.destinationPrefix, selection.kind].join("\0");
        selectorHits.set(selectorKey, (selectorHits.get(selectorKey) || 0) + 1);
        const origin = {package: key, packageSha256: row.sha256, member: file.path, sha256: fileHash(file.full), bytes: file.bytes};
        addSelectedFile(file.full, path.join(destination, "sdk"), target, claims, directoryClaims, origin);
        provenance.push({destination: target, ...origin});
      }
    }
  }
  if ([...selectorHits.values()].some(count => count < 1)) {throw new Error("A locked Windows SDK selection matched no archive members");}
  for (const root of EXPECTED_ROOTS) {fs.mkdirSync(path.join(destination, "sdk", ...root.split("/")), {recursive: true, mode: 0o755});}
  const selectedRows = [...claims.values()].sort((a, b) => compare(fold(a.path), fold(b.path)));
  // Build the VFS from the physical selected headers, with stable Windows virtual names.
  const overlay = path.join(destination, "vfs-overlay.json"), sourceRoot = path.join(destination, "sdk");
  const overlayRoots = headerOverlayEntries(selectedRows, sourceRoot);
  fs.writeFileSync(overlay, JSON.stringify({version: 0, "case-sensitive": false, roots: overlayRoots}, null, 2) + "\n", {flag: "wx", mode: 0o644});
  const sdkDescriptor = {sourceRoot, overlay, includeRoots: [...INCLUDE_ROOTS], libraryRoots: EXPECTED_ROOTS.filter(root => root === "msvc/lib" || root.startsWith("sdk/lib/")).map(root => path.join(sourceRoot, ...root.split("/")))};
  const descriptorFile = path.join(destination, "sdk.json"); fs.writeFileSync(descriptorFile, JSON.stringify(sdkDescriptor, null, 2) + "\n", {flag: "wx", mode: 0o644});
  fs.writeFileSync(path.join(destination, "files.json"), JSON.stringify({files: selectedRows, origins: provenance}, null, 2) + "\n", {flag: "wx", mode: 0o644});
  const currentRecipe = sourceInputs(repo, [], recipeLeaves); verifySourceInputs(recipe, currentRecipe, "Windows host-SDK extraction recipe");
  for (const [key, archive] of Object.entries(inputs.archives)) {if (fileHash(archive.path) !== archive.sha256) {throw new Error("Windows SDK archive changed during extraction: " + key);}}
  for (const tool of toolRows) {if (fileHash(tool.path) !== tool.sha256) {throw new Error("Windows SDK extraction tool changed during preparation: " + path.basename(tool.path));}}
  if (fileHash(fetchReceipt) !== inputs.receiptHash) {throw new Error("Windows SDK archive receipt changed during extraction");}
  // Only this invocation's fresh, canonical scratch is disposable; failures retain diagnostics.
  if (fs.realpathSync(work) !== work || !fs.lstatSync(work).isDirectory() || fs.lstatSync(work).isSymbolicLink()) {throw new Error("Windows SDK scratch identity changed");}
  fs.rmSync(work, {recursive: true});
  const result = {version: 1, status: "PASS", target: "windows-host", buildOnly: true, compiled: false, executed: false, publicationQualified: false,
    sourceCatalog: inputs.lock.sourceCatalog, sourceCatalogPublisherAuthenticated: false, sourceCatalogBindingVerified: false,
    fetchReceipt: {path: inputs.receiptPath, sha256: inputs.receiptHash}, recipe, tools: toolRows, archiveBytes: inputs.totalBytes,
    intermediateWorkDirectory: work, intermediateWorkRemoved: true,
    packageCount: Object.keys(inputs.lock.packages).length, cabinetCount: Object.keys(inputs.lock.cabinets).length,
    extracted: extractionTotals, selectedFileCount: selectedRows.length, headerOverlayFileCount: overlayRoots.length,
    roots: EXPECTED_ROOTS, descriptor: {path: descriptorFile, sha256: fileHash(descriptorFile)}, overlay: {path: overlay, sha256: fileHash(overlay)},
    provenance: {path: path.join(destination, "files.json"), sha256: fileHash(path.join(destination, "files.json"))},
    note: "Offline selected-root reconstruction only; not a complete Windows installation, compiler/runtime admission, publisher authentication, or publication qualification."};
  fs.writeFileSync(path.join(destination, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx", mode: 0o644});
  return result;
}

if (require.main === module) {
  try {
    const [archives, output, extra] = process.argv.slice(2);
    if (!archives || !output || extra) {throw new Error("Usage: prepare-native-windows-sdk.js /verified/archive-root /new/output");}
    const result = prepareWindowsHostSdk(archives, output);
    console.log(`[native-sdk] Reconstructed ${result.selectedFileCount} selected Windows SDK files; extraction only, not admitted or publication-qualified.`);
  } catch (error) {console.error(error instanceof Error ? error.message : "Windows SDK extraction failed"); process.exitCode = 1;}
}

module.exports = {safeRelative, parseIdt, buildMsiFilePaths, parse7zListing, mapSelectedMember, normalizeMsiSource,
  validateMsiMedia, walkTree, addSelectedFile, headerOverlayEntries, prepareWindowsHostSdk, defaultWindowsNativeLibraryPolicy};
