"use strict";

// Build the maintained Windows Core archive through its opt-in GYP target. This lane only
// stages sources/configures MSVS/builds the archive; it never executes a candidate worker.
const fs = require("node:fs");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {createHash, randomBytes} = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");
const {fileHash, walk, nativeSourceInputs, verifySourceInputs, readBuildJson, readBuildText} = require("./build-source-inputs");
const {gpuDeclarations} = require("./native-gpu-table");

/** @typedef {{status: number | null, signal: NodeJS.Signals | null, error?: Error | undefined, stdout?: unknown, stderr?: unknown}} ToolResult */
/** @typedef {(command: string, args: readonly string[], options: import("node:child_process").SpawnSyncOptions) => ToolResult} ToolSpawn */

const repo = path.resolve(__dirname, "..");
const target = "win32", maxArchiveBytes = 256 << 20;
const junctionNames = ["native", "sycl", "xmrig"];
const gpuReferences = ["algo_params", "pearlhash_claim"];
const gpuApiNames = new Set(gpuDeclarations(fs.readFileSync(path.join(repo, "sycl/lib.h"), "utf8")).map(row => row.name));
const maxSymbolBytes = 16 << 20, maxHeaderFiles = 4096, maxHeaderBytes = 32 << 20;
const windowsCoreCacheVersion = 1, maxCoreCacheBytes = 16 << 20, maxIncludeEntries = 200000, maxIncludeBytes = 1 << 30;
const forbiddenProject = /<ProjectReference\b|<Reference\b|NODE_GYP_MODULE_NAME|USING_(?:UV|V8)_SHARED|BUILDING_NODE_EXTENSION|\bNAPI(?:_|\b)|(?:node|v8|sycl)\.(?:dll|lib|vcxproj|sln)\b|_HAS_EXCEPTIONS\s*=\s*0|<WholeProgramOptimization>\s*true\s*<\/WholeProgramOptimization>|WholeProgramOptimization\s*[:=]\s*true|<LinkTimeCodeGeneration>\s*(?!false\s*<)[^<]+<\/LinkTimeCodeGeneration>|\bLTCG\b|(?:^|[<>\s])\/GL(?:[<>\s]|$)/i;
const forbiddenCompilerEnvironment = new Set(["CL", "_CL_", "LINK", "_LINK_", "MOM_CPU_MARCH", "ICXCFG", "__INTEL_PRE_CFLAGS", "__INTEL_POST_CFLAGS"]);

/** @param {string} left @param {string} right */
function sameNativePath(left, right) {
  return process.platform === "win32" ? left.toLowerCase() === right.toLowerCase() : left === right;
}

/** @param {string} file @returns {{cl: string, lib: string, ml64: string, version: string, include: string}} */
function msvcToolchain(file) {
  const toolDirectory = path.dirname(file), x64 = path.basename(toolDirectory).toLowerCase() === "x64";
  const hostDirectory = path.dirname(toolDirectory), hostX64 = path.basename(hostDirectory).toLowerCase() === "hostx64";
  const binDirectory = path.dirname(hostDirectory), bin = path.basename(binDirectory).toLowerCase() === "bin";
  const versionDirectory = path.dirname(binDirectory), msvcDirectory = path.dirname(versionDirectory);
  const version = path.basename(versionDirectory), msvc = path.basename(msvcDirectory).toLowerCase() === "msvc";
  if (!x64 || !hostX64 || !bin || !msvc || !/^14(?:\.\d+)+$/.test(version)) {
    throw new Error("Windows Core dumpbin must be under an MSVC/14.x/bin/HostX64/x64 tool directory");
  }
  const tools = {cl: path.join(toolDirectory, "cl.exe"), lib: path.join(toolDirectory, "lib.exe"), ml64: path.join(toolDirectory, "ml64.exe")};
  for (const [name, tool] of Object.entries(tools)) {requireFile(tool, `Windows Core ${name}`);}
  return {...tools, version, include: path.join(msvcDirectory, version, "include")};
}

/** @param {unknown} value @returns {value is string} */
function absolutePath(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 4096 && path.isAbsolute(value) &&
    !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
}
/** @param {string} file @param {string} label */
function requireFile(file, label) {
  if (!absolutePath(file) || !fs.statSync(file, {throwIfNoEntry: false})?.isFile()) {throw new Error(`${label} must be an absolute regular file`);}
  return file;
}
/** @param {string} directory @param {string} label */
function requireDirectory(directory, label) {
  if (!absolutePath(directory) || !fs.statSync(directory, {throwIfNoEntry: false})?.isDirectory()) {throw new Error(`${label} must be an absolute directory`);}
  return directory;
}
/** @typedef {{version: 1, nodeGyp: string, msbuild: string, dumpbin: string, python: string, libuvInclude: string, compiler?: string}} WindowsCoreConfig */
/** @param {unknown} value @returns {WindowsCoreConfig} */
function validateWindowsCoreDescriptor(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {throw new Error("Invalid Windows Core descriptor");}
  const config = /** @type {Record<string, unknown>} */ (value);
  if (config["version"] !== 1) {throw new Error("Invalid Windows Core descriptor");}
  const keys = Object.keys(config).sort(), expected = [...(Object.hasOwn(config, "compiler") ? ["compiler"] : []), "dumpbin", "libuvInclude", "msbuild", "nodeGyp", "python", "version"];
  if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) {throw new Error("Windows Core descriptor has unknown or missing fields");}
  const nodeGyp = config["nodeGyp"], msbuild = config["msbuild"], dumpbin = config["dumpbin"], python = config["python"], libuvInclude = config["libuvInclude"];
  if (typeof nodeGyp !== "string" || !/\.js$/i.test(nodeGyp)) {throw new Error("Windows Core nodeGyp must name an absolute JS file");}
  if (typeof msbuild !== "string" || !/\.exe$/i.test(msbuild)) {throw new Error("Windows Core msbuild must name an absolute executable");}
  if (typeof dumpbin !== "string" || !/\.exe$/i.test(dumpbin)) {throw new Error("Windows Core dumpbin must name an absolute executable");}
  requireFile(nodeGyp, "Windows Core nodeGyp"); requireFile(msbuild, "Windows Core msbuild"); requireFile(dumpbin, "Windows Core dumpbin");
  if (typeof python !== "string") {throw new Error("Windows Core python must be an explicit absolute file");} requireFile(python, "Windows Core python");
  if (typeof libuvInclude !== "string") {throw new Error("Windows Core libuvInclude must be an absolute directory");}
  requireDirectory(libuvInclude, "Windows Core libuvInclude");
  requireFile(path.join(libuvInclude, "uv.h"), "Windows Core libuv uv.h");
  requireFile(path.join(libuvInclude, "uv", "win.h"), "Windows Core libuv uv/win.h");
  const compiler = config["compiler"];
  // MSBuild interprets property separators/macros even inside a literal child argv token.
  if (Object.hasOwn(config, "compiler") && (typeof compiler !== "string" || !absolutePath(compiler) ||
      !/^icx-cl\.exe$/i.test(path.basename(compiler)) || /[;%$]/.test(compiler))) {throw new Error("Windows Core compiler must name an explicit Intel icx-cl.exe path");}
  if (typeof compiler === "string") {requireFile(compiler, "Windows Core Intel compiler");}
  return ({version: 1, nodeGyp, msbuild, dumpbin, python, libuvInclude, ...(typeof compiler === "string" ? {compiler} : {})});
}
/** Do not infer Intel or target compatibility from an executable's filename.
 * @param {string} text */
function intelCoreCompilerVersion(text) {
  const version = text.match(/Intel\(R\)[^\r\n]*Compiler[^\r\n]*\b(20\d\d\.\d+(?:\.\d+)?)\b/)?.[1];
  if (!version || !/^Target:\s*x86_64-pc-windows-msvc\s*$/m.test(text)) {throw new Error("Windows Core compiler is not an Intel Windows x64 frontend");}
  return {version, target: "x86_64-pc-windows-msvc"};
}
/** @param {string} output @param {string} root @returns {string} */
function validateOutput(output, root) {
  if (!absolutePath(output) || fs.existsSync(output)) {throw new Error("Windows Core output must be a new absolute directory");}
  requireDirectory(root, "Windows Core repository root");
  const parent = path.dirname(path.resolve(output)); requireDirectory(parent, "Windows Core output parent");
  const realRoot = fs.realpathSync(root), realParent = fs.realpathSync(parent), candidate = path.join(realParent, path.basename(output));
  const relative = path.relative(realRoot, candidate);
  if (!relative || (!relative.startsWith(".." + path.sep) && !path.isAbsolute(relative))) {throw new Error("Windows Core output must be outside the maintained repository");}
  return path.resolve(output);
}
/** @param {string} directory @returns {string[]} */
function libuvHeaderFiles(directory) {
  /** @type {string[]} */
  const files = []; const root = path.resolve(directory); let bytes = 0;
  const add = (/** @type {string} */ file) => {
    if (!/\.(?:h|hpp|inc)$/i.test(file)) {return;}
    const stat = fs.statSync(file, {throwIfNoEntry: false});
    if (!stat?.isFile()) {throw new Error("Windows Core libuv header is not a regular file: " + file);}
    files.push(file); bytes += stat.size;
    if (files.length > maxHeaderFiles || bytes > maxHeaderBytes) {throw new Error("Windows Core libuv header closure is excessive");}
  };
  // The include root can be /usr/include or Node's header SDK. Only libuv is an input;
  // do not enumerate unrelated SDK/system headers to discover this small closure.
  add(path.join(root, "uv.h")); walk(path.join(root, "uv"), add);
  for (const required of [path.join(root, "uv.h"), path.join(root, "uv", "win.h")]) {
    if (!files.includes(required)) {throw new Error("Windows Core libuv header closure is incomplete: " + required);}
  }
  return files.sort();
}
/** @param {string[]} files @returns {Record<string, string>} */
function hashFiles(files) {return Object.fromEntries([...new Set(files)].sort().map(file => [file, fileHash(file)]));}
/** @param {string} output */
function corePaths(output) {
  const source = path.join(output, "source"), build = path.join(source, "build");
  return {source, build, devdir: path.join(output, "devdir"), binding: path.join(source, "binding.gyp"),
    project: path.join(build, "mom_native_core.vcxproj"), archive: path.join(build, "Release", "mom_native_core.lib"),
    exportedArchive: path.join(output, "cpu-core.lib"), msbuildUserExtensions: path.join(output, "msbuild-empty"),
    configureLog: path.join(output, "node-gyp-configure.log"), msbuildLog: path.join(output, "msbuild.log"), symbolsLog: path.join(output, "dumpbin-symbols.log")};
}

/** A single replaceable GYP/MSBuild graph; input identities live in state, never in directory names. */
/** @param {string} root */
function windowsCoreCacheRoot(root = repo) {
  if (!sameNativePath(fs.realpathSync(root), path.resolve(root))) {throw new Error("Windows Core repository root is not canonical");}
  let current = path.resolve(root);
  for (const part of ["build", "cache", "native-windows-core", "v1"]) {
    current = path.join(current, part);
    let stat;
    try {
      stat = fs.lstatSync(current);
    } catch (error) {
      if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {throw error;}
      try {fs.mkdirSync(current, {mode: 0o700});} catch (mkdirError) {
        if (!(mkdirError instanceof Error) || !("code" in mkdirError) || mkdirError.code !== "EEXIST") {throw mkdirError;}
      }
      stat = fs.lstatSync(current);
    }
    if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("Windows Core cache path contains a symlink or non-directory: " + current);}
    if (!sameNativePath(fs.realpathSync(current), current)) {throw new Error("Windows Core cache path is redirected: " + current);}
  }
  return current;
}

/** Same-host concurrent builds fail closed; an abandoned lock also requires explicit recovery. */
/** @param {string} cacheRoot */
function acquireWindowsCoreCacheLock(cacheRoot) {
  const lock = path.join(cacheRoot, "build.lock"), token = randomBytes(16).toString("hex");
  try {fs.writeFileSync(lock, JSON.stringify({token, pid: process.pid, started: Date.now()}), {flag: "wx", mode: 0o600});} catch (error) {
    if (error instanceof Error && "code" in error && error.code === "EEXIST") {
      const stat = fs.lstatSync(lock);
      if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Windows Core cache lock is unsafe", {cause: error});}
      throw new Error("Windows Core cache is already locked or has an unresolved owner", {cause: error});
    }
    throw error;
  }
  return () => {
    const stat = fs.lstatSync(lock);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 1024) {throw new Error("Windows Core cache lock changed ownership");}
    const receipt = readCoreCacheJson(lock, 1024);
    if (receipt.token !== token) {throw new Error("Windows Core cache lock token changed");}
    fs.unlinkSync(lock);
  };
}

/** @param {string} file @param {number} [maximum] */
function readCoreCacheJson(file, maximum = maxCoreCacheBytes) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 || stat.size > maximum) {throw new Error("Invalid Windows Core cache receipt: " + file);}
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

/** @param {string} file @param {unknown} value */
function writeCoreCacheJson(file, value) {
  const text = JSON.stringify(value) + "\n";
  if (Buffer.byteLength(text) > maxCoreCacheBytes) {throw new Error("Windows Core cache receipt is excessive");}
  const temporary = file + ".tmp-" + randomBytes(8).toString("hex");
  fs.writeFileSync(temporary, text, {flag: "wx", mode: 0o600});
  try {
    try {
      const stat = fs.lstatSync(file);
      if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Windows Core cache receipt was replaced: " + file);}
      fs.unlinkSync(file);
    } catch (error) {if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {throw error;}}
    fs.renameSync(temporary, file);
  } finally {if (fs.existsSync(temporary)) {fs.unlinkSync(temporary);}}
}

/** @param {string[]} roots @returns {{complete: boolean, roots: string[], sha256: string}} */
function windowsCoreIncludeMembership(roots) {
  let complete = roots.length > 0, bytes = 0; const entries = [], selected = [], selectedIdentities = new Set();
  for (const input of [...new Set(roots)].sort()) {
    if (!path.isAbsolute(input)) {complete = false; continue;}
    try {
      const real = fs.realpathSync(input), rootStat = fs.lstatSync(input);
      if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || !sameNativePath(real, path.resolve(input))) {complete = false; continue;}
      const identity = process.platform === "win32" ? real.toLowerCase() : real;
      if (selectedIdentities.has(identity)) {continue;}
      selectedIdentities.add(identity);
      selected.push(real);
      const pending = [{directory: real, relative: ""}];
      while (pending.length) {
        const current = pending.pop(); if (!current) {continue;}
        for (const item of fs.readdirSync(current.directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
          const child = path.join(current.directory, item.name), relative = current.relative ? current.relative + "/" + item.name : item.name;
          if (item.isSymbolicLink()) {complete = false; continue;}
          if (!item.isDirectory() && !item.isFile()) {complete = false; continue;}
          let digest = "";
          if (item.isFile()) {
            const stat = fs.lstatSync(child); bytes += stat.size;
            if (bytes > maxIncludeBytes) {return {complete: false, roots: selected, sha256: ""};}
            digest = fileHash(child);
          }
          entries.push([real.toLowerCase(), relative.toLowerCase(), item.isDirectory() ? "d" : "f", digest]);
          if (entries.length > maxIncludeEntries) {return {complete: false, roots: selected, sha256: ""};}
          if (item.isDirectory()) {pending.push({directory: child, relative});}
        }
      }
    } catch (error) {
      if (!(error instanceof Error) || !("code" in error) || !["ENOENT", "ENOTDIR"].includes(String(error.code))) {throw error;}
      complete = false;
    }
  }
  entries.sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
  return {complete, roots: selected.sort(), sha256: createHash("sha256").update(JSON.stringify(entries)).digest("hex")};
}

/** @param {WindowsCoreConfig} config @param {ReturnType<typeof msvcToolchain>} msvc @param {NodeJS.ProcessEnv} environment */
function windowsCoreIncludeRoots(config, msvc, environment) {
  const roots = [config.libuvInclude, msvc.include], include = environment["INCLUDE"] || "";
  const entries = include.split(";").map(value => value.trim().replace(/^"|"$/g, "")).filter(Boolean);
  let complete = entries.length > 0;
  for (const entry of entries) {if (path.isAbsolute(entry)) {roots.push(entry);} else {complete = false;}}
  const sdk = environment["WindowsSdkDir"], version = (environment["WindowsSDKVersion"] || "").replace(/[\\/]+$/, "");
  if (!sdk || !version || !path.isAbsolute(sdk)) {complete = false;} else {
    const selected = new Set();
    for (const leaf of ["ucrt", "shared", "um", "winrt", "cppwinrt"]) {
      const root = path.join(sdk, "Include", version, leaf);
      if (fs.statSync(root, {throwIfNoEntry: false})?.isDirectory()) {roots.push(root); selected.add(leaf);}
    }
    if (!["ucrt", "shared", "um"].every(leaf => selected.has(leaf))) {complete = false;}
  }
  const inventory = windowsCoreIncludeMembership([...new Set(roots)]);
  return {...inventory, complete: complete && inventory.complete};
}

/** Avoid PS5's adapted-property sort on hashtables; order/dedupe by the actual Path and Hash values. */
function windowsCoreTrackerRowsScript() {
  return [
    "$inputs = @($outputs | ForEach-Object { $_['Inputs'] } | Sort-Object -Property { [string] $_['Path'] }, { [string] $_['Hash'] } -Unique | ForEach-Object { @{ Path = $_['Path']; Hash = $_['Hash'] } })",
    "$artifacts = @($outputs | ForEach-Object { @{ Path = $_['Path']; Hash = $_['Hash'] } } | Sort-Object -Property { [string] $_['Path'] }, { [string] $_['Hash'] } -Unique)",
    "$toolchain = @($script:MomBuildSnapshot | Where-Object { -not ([string] $_['Path']).StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) } | Sort-Object -Property { [string] $_['Path'] }, { [string] $_['Hash'] } -Unique | ForEach-Object { @{ Path = $_['Path']; Hash = $_['Hash'] } })",
  ].join("\n");
}

/** Keep the cache census limited to the static Core's actual search inputs.
 * The maintained Core target is a static archive and does not resolve ambient LIB candidates;
 * actual MSBuild TLog dependencies are still retained, and the original LIB remains in the
 * build environment identity and the environment passed to MSBuild. */
/** @param {NodeJS.ProcessEnv} environment @returns {NodeJS.ProcessEnv} */
function windowsCoreTrackerEnvironment(environment) {
  const copy = {...environment};
  for (const key of Object.keys(copy)) {if (key.toUpperCase() === "LIB") {delete copy[key];}}
  copy["LIB"] = "";
  return copy;
}

/** Convert the maintained Windows MSBuild tracker receipt to compact, content-hashed inputs. */
/** @param {string} buildRoot @param {string} cacheRoot @param {ReturnType<typeof windowsCorePlan>} plan
 * @param {NodeJS.ProcessEnv} environment @param {boolean} requireComplete @param {string} [logFile] */
function windowsCoreTrackerSnapshot(buildRoot, cacheRoot, plan, environment, requireComplete, logFile) {
  const systemRoot = environment["SystemRoot"] || environment["WINDIR"];
  if (!systemRoot || !path.isAbsolute(systemRoot)) {throw new Error("Windows PowerShell root is unknown; Core tracker cache is unavailable");}
  const powershell = path.join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  requireFile(powershell, "Windows Core PowerShell");
  /** @param {string} value */
  const psQuote = value => "'" + value.replaceAll("'", "''") + "'";
  const helper = path.join(repo, "scripts", "windows-incremental-build.ps1"), tracker = path.join(repo, "scripts", "windows-msbuild-cache.ps1");
  const toolPaths = [...new Set([...Object.keys(plan.toolHashes), powershell])].map(psQuote).join(",");
  /** @param {string | undefined} value @returns {value is string} */
  const isDirectoryPath = value => typeof value === "string" && path.isAbsolute(value) && Boolean(fs.statSync(value, {throwIfNoEntry: false})?.isDirectory());
  const roots = [...new Set([environment["VCToolsInstallDir"], environment["WindowsSdkDir"], path.dirname(plan.msvc.include)]
    .filter(isDirectoryPath))].map(psQuote).join(",");
  const script = ["$ErrorActionPreference = 'Stop'", "$ProgressPreference = 'SilentlyContinue'",
    "$OutputEncoding = New-Object System.Text.UTF8Encoding($false)", "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)",
    `. ${psQuote(helper)}`, `. ${psQuote(tracker)}`,
    `$context = Initialize-MomIncrementalBuild ${psQuote(cacheRoot)} @(${toolPaths}) @(${roots})`,
    `$release = Join-Path ${psQuote(buildRoot)} 'Release'`,
    `Assert-MomBuildOutputTree $release ${psQuote(buildRoot)}`,
    "$outputs = @(); $complete = $false",
    `try { $outputs = @(Get-MomTrackedOutputs ${psQuote(buildRoot)}); if ($outputs.Count -gt 0) { Assert-MomTrackedOutputCoverage $outputs ${psQuote(buildRoot)}; $complete = $true } } catch { $outputs = @(); $complete = $false }`,
    ...(requireComplete ? ["if (-not $complete) { throw 'MSBuild tracker graph is incomplete.' }"] : []),
    `$prefix = [IO.Path]::GetFullPath(${psQuote(repo)}).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar`,
    windowsCoreTrackerRowsScript(),
    "$snapshot = @{ Complete = $complete; Membership = $script:MomBuildMembership; Toolchain = $toolchain; Inputs = $inputs; Artifacts = $artifacts }",
    "[Console]::Out.WriteLine((ConvertTo-Json -InputObject $snapshot -Compress -Depth 8))"].join("\n");
  const result = spawnSync(powershell, ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
    {cwd: repo, env: windowsCoreTrackerEnvironment(environment), encoding: "utf8", timeout: 300000, maxBuffer: 64 << 20, windowsHide: true});
  if (logFile) {
    const log = JSON.stringify({command: powershell, args: ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
      status: result.status, signal: result.signal, error: result.error?.message, stdout: result.stdout || "", stderr: result.stderr || ""}) + "\n";
    fs.writeFileSync(logFile, log, {flag: "wx"});
  }
  if (result.error || result.signal || result.status !== 0) {
    const detail = String(result.stderr || result.stdout || result.error?.message || "").slice(0, 2048);
    throw new Error("Windows Core MSBuild tracker admission failed (status " + result.status + (detail ? "): " + detail : ")"), {cause: result.error});
  }
  let snapshot;
  try {snapshot = JSON.parse(result.stdout || "");} catch (error) {
    throw new Error("Windows Core MSBuild tracker returned invalid UTF-8 JSON: " + String(result.stdout || "").slice(0, 512), {cause: error});
  }
  if (!snapshot || typeof snapshot !== "object" || typeof snapshot.Complete !== "boolean" || typeof snapshot.Membership !== "string" ||
      !Array.isArray(snapshot.Inputs) || !Array.isArray(snapshot.Artifacts) || !Array.isArray(snapshot.Toolchain)) {
    throw new Error("Windows Core MSBuild tracker returned a malformed receipt");
  }
  /** @param {unknown} value @returns {Record<string, string>} */
  const normalize = value => {
    const rows = Array.isArray(value) ? value : value ? [value] : [], result = /** @type {Record<string, string>} */ ({});
    for (const row of rows) {
      if (!row || typeof row !== "object") {throw new Error("Windows Core tracker returned an invalid file identity");}
      const record = /** @type {Record<string, unknown>} */ (row), selected = record["Path"], hash = record["Hash"];
      if (typeof selected !== "string" || !path.isAbsolute(selected) || typeof hash !== "string" || !/^[a-f0-9]{64}$/i.test(hash)) {
        throw new Error("Windows Core tracker returned an invalid file identity");
      }
      const file = fs.realpathSync(selected).toLowerCase(), digest = hash.toLowerCase();
      if (Object.hasOwn(result, file) && result[file] !== digest) {throw new Error("Windows Core tracker aliases disagree: " + file);}
      result[file] = digest;
    }
    const ordered = /** @type {Record<string, string>} */ ({});
    for (const file of Object.keys(result).sort()) {
      const hash = result[file]; if (hash !== undefined) {ordered[file] = hash;}
    }
    return ordered;
  };
  const inputs = normalize(snapshot.Inputs), artifacts = normalize(snapshot.Artifacts), toolchain = normalize(snapshot.Toolchain);
  return {complete: snapshot.Complete, membership: snapshot.Membership, inputs, artifacts, toolchain, powershellSha256: fileHash(powershell)};
}

/** @param {NodeJS.ProcessEnv} environment */
function windowsCoreEnvironmentSha256(environment) {
  const normalized = Object.fromEntries(Object.entries(environment).sort(([a], [b]) => a.localeCompare(b)));
  return createHash("sha256").update(JSON.stringify(normalized)).digest("hex");
}

/** @param {ReturnType<typeof nativeSourceInputs>} sourceState @param {string} root */
function windowsCoreSourceMtimes(sourceState, root) {
  const result = /** @type {Record<string, number>} */ ({});
  for (const relative of Object.keys(sourceState.files).sort()) {
    const file = path.join(root, relative), stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Windows Core source is not a regular file: " + file);}
    result[relative] = stat.mtimeMs;
  }
  return result;
}

/** @param {unknown} left @param {unknown} right */
function cacheJsonEqual(left, right) {return isDeepStrictEqual(left, right);}

/** Keep ordinary newer source edits incremental; every incomplete or non-source drift rebuilds. */
/** @param {{state?: any, current: any, marker: boolean, projectExists: boolean, archiveValid: boolean, forceRebuild?: boolean}} value */
function windowsCoreCacheDecision({state, current, marker, projectExists, archiveValid, forceRebuild = false}) {
  const configure = !state || marker || forceRebuild || !projectExists || state.version !== windowsCoreCacheVersion ||
    !cacheJsonEqual(state.staticIdentity, current.staticIdentity) || state.projectSha256 !== current.projectSha256;
  /** @param {string} reason */
  const rebuild = reason => ({configure, target: "Rebuild", reason});
  if (forceRebuild) {return rebuild("explicit cold-build control");}
  if (configure) {return rebuild(marker ? "interrupted prior build" : "work graph identity changed or absent");}
  if (!state.incrementalSafe || !current.incrementalSafe) {return rebuild("complete source/SDK dependency inventory is unavailable");}
  if (!archiveValid || !current.tracker.complete || !state.tracker?.complete) {return rebuild("archive or tracker graph is incomplete");}
  if (!cacheJsonEqual(state.tracker.artifacts, current.tracker.artifacts)) {return rebuild("tracked output changed outside MSBuild");}
  if (!cacheJsonEqual(state.tracker.toolchain, current.tracker.toolchain)) {return rebuild("toolchain input changed");}
  if (state.sourceState?.graph && !cacheJsonEqual(state.sourceState.graph, current.sourceState.graph)) {return rebuild("build recipe or graph input changed");}
  const previousFiles = state.sourceState.files || {}, currentFiles = current.sourceState.files || {};
  if (!cacheJsonEqual(Object.keys(previousFiles).sort(), Object.keys(currentFiles).sort())) {return rebuild("source/header membership changed");}
  const changed = Object.keys(currentFiles).filter(file => previousFiles[file] !== currentFiles[file]);
  const dependencyDiff = Object.keys(current.tracker.inputs).filter(file => state.tracker.inputs[file] !== current.tracker.inputs[file]);
  const removedDependencies = Object.keys(state.tracker.inputs).filter(file => !Object.hasOwn(current.tracker.inputs, file));
  if (!changed.length && !dependencyDiff.length && !removedDependencies.length) {return {configure: false, target: "Build", reason: "inputs unchanged"};}
  if (!changed.length || removedDependencies.length) {return rebuild("untracked or removed dependency changed");}
  const changedSources = changed.filter(file => /\.(?:c|cc|cpp|cxx|asm)$/i.test(file));
  if (changedSources.length !== changed.length) {return rebuild("header or non-source input changed");}
  const dependencySet = new Set(Object.keys(state.tracker.inputs));
  if (changedSources.some(file => !dependencySet.has(path.join(current.repo, file).toLowerCase()))) {return rebuild("changed source is not tracked by MSBuild");}
  const changedDependencySources = dependencyDiff.filter(file => changedSources.some(source => path.join(current.repo, source).toLowerCase() === file));
  if (changedDependencySources.length !== dependencyDiff.length) {return rebuild("tracked header or SDK input changed");}
  if (changedSources.some(file => !(current.sourceMtimes[file] > state.lastSuccessMs))) {return rebuild("changed source is backdated");}
  return {configure: false, target: "Build", reason: "tracked source-only edit"};
}

/** Permit outputs changed by the current MSBuild target, but not changed compiler inputs.
 * MSBuild can report generated .obj files as inputs to its static-library task. */
/** @param {{inputs: Record<string, string>, artifacts: Record<string, string>}} before
 * @param {{inputs: Record<string, string>, artifacts: Record<string, string>}} after */
function verifyWindowsCoreTrackerInputs(before, after) {
  const generated = new Set([...Object.keys(before.artifacts), ...Object.keys(after.artifacts)]);
  for (const [file, hash] of Object.entries(before.inputs)) {
    if (!generated.has(file) && Object.hasOwn(after.inputs, file) && after.inputs[file] !== hash) {
      throw new Error("Windows Core compiler dependency changed during compilation: " + file);
    }
  }
}

/** @param {ReturnType<typeof windowsCorePlan>} plan @param {string} root */
function windowsCoreWorkPlan(plan, root) {
  const paths = corePaths(root);
  /** @param {string} value */
  const replace = value => value.replaceAll(plan.output, root);
  return {...plan, output: root, paths,
    configure: {...plan.configure, args: plan.configure.args.map(replace)},
    build: {...plan.build, args: plan.build.args.map(replace)},
    audit: {...plan.audit, args: plan.audit.args.map(replace)}};
}

/** @param {ReturnType<typeof windowsCorePlan>} plan @param {string} descriptorSha256 @param {string} environmentSha256
 * @param {ReturnType<typeof windowsCoreIncludeRoots>} include @param {ReturnType<typeof windowsCoreTrackerSnapshot>} tracker
 * @param {ReturnType<typeof corePaths>} workPaths */
function windowsCoreCacheView(plan, descriptorSha256, environmentSha256, include, tracker, workPaths) {
  const membership = {files: Object.keys(plan.sourceState.files).sort(), graph: Object.keys(plan.sourceState.graph).sort()};
  const gypFiles = Object.fromEntries(Object.entries(plan.sourceState.files).filter(([file]) => /\.gypi?$/i.test(file)).sort(([a], [b]) => a.localeCompare(b)));
  const sourceMembership = [...new Set([...membership.files, ...membership.graph])].sort();
  const staticIdentity = {descriptorSha256, recipeHashes: plan.recipeHashes, toolHashes: plan.toolHashes, headerHashes: plan.headerHashes,
    environmentSha256, includeRoots: include.roots, includeMembershipSha256: include.sha256, includeComplete: include.complete,
    toolchainMembership: tracker.membership, toolchainInputs: tracker.toolchain, powershellSha256: tracker.powershellSha256,
    sourceMembership, graph: plan.sourceState.graph, gypFiles};
  let projectSha256 = null, archiveSha256 = null;
  const outputs = /** @type {Array<[string, "project" | "archive"]>} */ ([[workPaths.project, "project"], [workPaths.archive, "archive"]]);
  for (const [file, label] of outputs) {
    try {
      const stat = fs.lstatSync(file);
      if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Windows Core cached " + label + " is unsafe: " + file);}
      if (label === "project") {projectSha256 = fileHash(file);} else {archiveSha256 = fileHash(file);}
    } catch (error) {if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {throw error;}}
  }
  return {repo: plan.root, staticIdentity, sourceState: plan.sourceState, sourceMtimes: windowsCoreSourceMtimes(plan.sourceState, plan.root),
    tracker, projectSha256, archiveSha256,
    incrementalSafe: include.complete && tracker.complete && (!plan.config.compiler || plan.intelConfigs.length > 0)};
}

/** @param {string} file */
function readWindowsCoreCacheState(file) {
  try {
    const state = readCoreCacheJson(file);
    return state && state.version === windowsCoreCacheVersion && state.staticIdentity && state.sourceState && state.tracker ? state : null;
  } catch (error) {
    if (error instanceof SyntaxError) {return null;}
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {return null;}
    throw error;
  }
}

/** @param {string} file */
function windowsCoreMarkerPresent(file) {
  try {
    const stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 4096) {throw new Error("Windows Core in-progress marker is unsafe");}
    return true;
  } catch (error) {if (error instanceof Error && "code" in error && error.code === "ENOENT") {return false;} throw error;}
}

/** @param {string} file */
function markWindowsCoreInProgress(file) {
  if (windowsCoreMarkerPresent(file)) {return;}
  fs.writeFileSync(file, JSON.stringify({pid: process.pid, started: Date.now(), token: randomBytes(16).toString("hex")}), {flag: "wx", mode: 0o600});
}

/** @param {string} file */
function clearWindowsCoreInProgress(file) {
  if (!windowsCoreMarkerPresent(file)) {return;}
  fs.unlinkSync(file);
}

/** Rebuilds own only `Release`; validate every descendant before pruning orphaned MSBuild objects. */
/** @param {string} buildRoot */
function clearWindowsCoreRelease(buildRoot) {
  const root = path.resolve(buildRoot), rootStat = fs.lstatSync(root);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || !sameNativePath(fs.realpathSync(root), root)) {throw new Error("Windows Core build root is unsafe");}
  const release = path.join(root, "Release");
  let releaseStat;
  try {releaseStat = fs.lstatSync(release);} catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {return;}
    throw error;
  }
  if (!releaseStat.isDirectory() || releaseStat.isSymbolicLink()) {throw new Error("Windows Core Release output is unsafe");}
  const pending = [release];
  while (pending.length) {
    const directory = pending.pop(); if (!directory) {continue;}
    for (const entry of fs.readdirSync(directory, {withFileTypes: true})) {
      const child = path.join(directory, entry.name), stat = fs.lstatSync(child), relative = path.relative(root, child);
      if (path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep) || stat.isSymbolicLink() || (!stat.isDirectory() && !stat.isFile())) {
        throw new Error("Windows Core Release tree contains an unsafe path: " + child);
      }
      if (stat.isDirectory()) {pending.push(child);}
    }
  }
  fs.rmSync(release, {recursive: true, force: false});
}
/** @param {unknown} config @param {string} output @param {string} root */
function windowsCorePlan(config, output, root = repo) {
  const selected = validateWindowsCoreDescriptor(config), destination = validateOutput(output, root), paths = corePaths(destination);
  const sourceState = nativeSourceInputs(root, "cpu"), recipeFiles = [path.join(root, "binding.gyp"),
    path.join(root, "scripts", "build-native-windows-core.js"), path.join(root, "scripts", "windows-incremental-build.ps1"),
    path.join(root, "scripts", "windows-msbuild-cache.ps1")];
  const msvc = msvcToolchain(selected.dumpbin), headers = libuvHeaderFiles(selected.libuvInclude);
  const selectedCompiler = selected.compiler;
  const intelConfigs = selectedCompiler ? fs.readdirSync(path.dirname(selectedCompiler)).filter(name => /^(?:icx|icpx|clang|default).*\.cfg$/i.test(name))
    .map(name => requireFile(path.join(path.dirname(selectedCompiler), name), "Intel compiler config")) : [];
  const tools = [process.execPath, selected.nodeGyp, selected.msbuild, selected.dumpbin, selected.python, selected.compiler || msvc.cl, msvc.lib, msvc.ml64,
    ...intelConfigs];
  const configureArgs = [selected.nodeGyp, "configure", `--devdir=${paths.devdir}`, "--msvs-version", "2022", `--python=${selected.python}`];
  configureArgs.push("--", "-Dmom_native_core=1", `-Dmom_native_uv_include=${selected.libuvInclude}`, "-Dtarget_arch=x64", "-Dlibrary=shared_library");
  if (selected.compiler) {configureArgs.push("-Dmom_native_intel=1");}
  const msbuildArgs = [paths.project, "/m:2", "/p:BuildProjectReferences=false", "/p:Configuration=Release", "/p:Platform=x64",
    `/p:VCToolsVersion=${msvc.version}`, `/p:DirectoryBuildPropsPath=${path.join(paths.msbuildUserExtensions, "Directory.Build.props")}`,
    `/p:DirectoryBuildTargetsPath=${path.join(paths.msbuildUserExtensions, "Directory.Build.targets")}`,
    `/p:MSBuildUserExtensionsPath=${paths.msbuildUserExtensions}`, `/p:UserRootDir=${paths.msbuildUserExtensions}${path.sep}`];
  // Keep GYP's maintained source/ISA graph and MASM tooling; replace only its C/C++ frontend.
  // Disable Microsoft's multi-tool wrapper so it cannot silently invoke the default cl.exe.
  if (selected.compiler) {msbuildArgs.push(`/p:CLToolExe=${path.basename(selected.compiler)}`, `/p:CLToolPath=${path.dirname(selected.compiler)}${path.sep}`, "/p:UseMultiToolTask=false");}
  const audit = {command: selected.dumpbin, args: ["/headers", "/directives", "/symbols", paths.archive]};
  return {config: selected, root, output: destination, paths, msvc, sourceState,
    sourceInventory: [...new Set([...Object.keys(sourceState.files), ...Object.keys(sourceState.graph)])].sort(),
    recipeHashes: hashFiles(recipeFiles), toolHashes: hashFiles(tools), headerHashes: hashFiles(headers), intelConfigs,
    compiler: selected.compiler ? {command: selected.compiler, args: ["--version"]} : null,
    configure: {command: process.execPath, args: configureArgs}, build: {command: selected.msbuild, args: msbuildArgs}, audit};
}
/** @param {string} command @param {string[]} args @param {string} cwd @param {string} log
 * @param {ToolSpawn} [spawn] @param {NodeJS.ProcessEnv} [environment] @param {number} [timeout] */
function runTool(command, args, cwd, log, spawn = spawnSync, environment = process.env, timeout = 900000) {
  const fd = fs.openSync(log, "wx");
  try {
    const result = spawn(command, args, {cwd, stdio: ["ignore", fd, fd], timeout, env: environment});
    if (result.error || result.signal || result.status !== 0) {throw new Error(`Windows Core tool failed: ${path.basename(command)}`, {cause: result.error});}
    return result;
  } finally {fs.closeSync(fd);}
}
/** @param {string} command @param {string[]} args @param {string} cwd @param {string} log
 * @param {ToolSpawn} [spawn] @param {NodeJS.ProcessEnv} [environment] @param {number} [timeout] @returns {string} */
function runToolText(command, args, cwd, log, spawn = spawnSync, environment = process.env, timeout = 120000) {
  const result = spawn(command, args, {cwd, encoding: "utf8", maxBuffer: maxSymbolBytes, timeout, env: environment});
  const output = [result.stdout, result.stderr].filter(value => typeof value === "string").join("");
  fs.writeFileSync(log, output, {flag: "wx"});
  if (result.error || result.signal || result.status !== 0) {throw new Error(`Windows Core tool failed: ${path.basename(command)}`, {cause: result.error});}
  if (output.length > maxSymbolBytes) {throw new Error("Windows Core symbol output is excessive");}
  return output;
}
/** @param {string} text @param {string} tag @returns {string[]} */
function projectTagValues(text, tag) {
  const expression = new RegExp(`<${tag}\\b[^>]*>([^<]*)<\\/${tag}>`, "gi");
  return [...text.matchAll(expression)].map(match => (match[1] || "").trim());
}
/** @param {string} value @returns {string} */
function projectPath(value) {
  return value.replace(/&amp;/gi, "&").replace(/[\\/]+/g, "/").replace(/\/+$/, "").toLowerCase();
}
/** @param {string} text @returns {string[]} */
function projectIncludeDirectories(text) {
  return projectTagValues(text, "AdditionalIncludeDirectories").flatMap(value => value.split(";").map(projectPath));
}
/** @param {string} text @param {string} tag @param {string} expected */
function requireProjectValues(text, tag, expected) {
  const values = projectTagValues(text, tag);
  if (!values.length || values.some(value => value.toLowerCase() !== expected.toLowerCase())) {
    throw new Error(`Windows Core project has an unexpected ${tag} value`);
  }
}
/** @param {string} file @param {string} label @param {string} [libuvInclude] */
function inspectCoreProject(file, label = "Windows Core project", libuvInclude) {
  const text = readBuildText(file);
  const requirements = /** @type {Array<[string, string]>} */ ([ ["ConfigurationType", "StaticLibrary"], ["PlatformToolset", "v143"], ["RuntimeLibrary", "MultiThreaded"],
    ["ExceptionHandling", "Sync"], ["LanguageStandard", "stdcpp20"], ["WholeProgramOptimization", "false"]]);
  for (const [tag, expected] of requirements) {
    requireProjectValues(text, tag, expected);
  }
  const sources = [...text.matchAll(/<(?:ClCompile|MASM)\s+Include\s*=\s*"[^"]+"/gi)];
  if (!sources.length || !/\bMOM_NATIVE_RUNTIME\b/.test(text)) {
    throw new Error(`${label} is not the maintained v143 static Core project`);
  }
  if (forbiddenProject.test(text)) {throw new Error(`${label} contains Node/V8/NAPI/SYCL project or DLL inputs`);}
  if (libuvInclude) {
    const expected = projectPath(libuvInclude), includes = projectIncludeDirectories(text);
    if (!includes.includes(expected)) {throw new Error(`${label} does not include the selected libuv SDK path`);}
    const implicitNode = /(?:^|\/)(?:include\/node|deps\/uv\/include|node_root_dir|node_modules\/node(?:$|\/))/i;
    if (includes.some(include => include !== expected && implicitNode.test(include))) {
      throw new Error(`${label} contains an implicit Node or stale libuv header path`);
    }
  }
  return {bytes: Buffer.byteLength(text), sha256: fileHash(file), objectCount: sources.length};
}
/** @param {string} archive @returns {{bytes: number, mtimeMs: number, ctimeMs: number}} */
function inspectStaticArchive(archive) {
  const before = fs.statSync(archive, {throwIfNoEntry: false});
  if (!before?.isFile() || before.size < 9 || before.size > maxArchiveBytes) {throw new Error("Windows Core archive is missing or excessive");}
  const fd = fs.openSync(archive, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK), header = Buffer.alloc(8);
  try {
    if (fs.readSync(fd, header, 0, header.length, 0) !== header.length || header.toString("ascii") !== "!<arch>\n") {
      throw new Error("Windows Core output is not a static archive");
    }
    const after = fs.fstatSync(fd);
    if (after.size !== before.size || after.mtimeMs !== before.mtimeMs || after.ctimeMs !== before.ctimeMs) {throw new Error("Windows Core archive changed during inspection");}
  } finally {fs.closeSync(fd);}
  return {bytes: before.size, mtimeMs: before.mtimeMs, ctimeMs: before.ctimeMs};
}
/** @param {string} symbol @returns {string[]} */
function matchingGpuSymbols(symbol) {
  const name = symbol.match(/^(?:__imp_)?\?([A-Za-z_]\w*)@@/)?.[1];
  return name && gpuApiNames.has(name) ? [name] : [];
}
/** @param {string} symbol @returns {boolean} */
function isRuntimeSymbol(symbol) {
  // Match runtime namespace roots/C APIs, not ordinary STL names such as _Insert_node.
  return /@(?:node|v8|Napi)@@|^(?:__imp_)?_?(?:napi_|node_module_register|NODE_MODULE)|^\?v8_/i.test(symbol);
}
/** @param {unknown} value @returns {{undefinedSymbolCount: number, definedSymbolCount: number, gpuUnresolvedReferences: string[], definedGpuReferences: string[], impGpuThunks: string[], forbiddenRuntimeSymbols: string[]}} */
function parseDumpbinSymbols(value) {
  if (typeof value !== "string" || value.length < 1 || value.length > maxSymbolBytes || value.includes("\0")) {throw new Error("Windows Core symbol output is malformed or excessive");}
  const undefinedSymbols = [], definedSymbols = [], impGpuThunks = [], forbiddenRuntimeSymbols = [];
  for (const line of value.split(/\r?\n/)) {
    if (!/\bExternal\b/i.test(line) || !/\b(?:UNDEF|SECT[0-9A-F]+)\b/i.test(line)) {continue;}
    const separator = line.lastIndexOf("|"), fields = line.trim().split(/\s+/);
    // dumpbin can append a human-readable signature. Admit the actual COFF token only,
    // not words in template types, namespaces, or that optional annotation.
    const symbol = ((separator >= 0 ? line.slice(separator + 1).trim() : fields[fields.length - 1]) || "").split(/\s/)[0] || "";
    if (!symbol) {continue;}
    if (/\bUNDEF\b/i.test(line)) {undefinedSymbols.push(symbol);} else {definedSymbols.push(symbol);}
    if (symbol.startsWith("__imp_") && matchingGpuSymbols(symbol).length) {impGpuThunks.push(symbol);}
    if (isRuntimeSymbol(symbol)) {forbiddenRuntimeSymbols.push(symbol);}
  }
  const unresolved = [...new Set(undefinedSymbols.flatMap(matchingGpuSymbols))].sort();
  const defined = [...new Set(definedSymbols.flatMap(matchingGpuSymbols))].sort();
  const imp = [...new Set(impGpuThunks)].sort(), runtime = [...new Set(forbiddenRuntimeSymbols)].sort();
  if (gpuReferences.some(name => !unresolved.includes(name))) {throw new Error("Windows Core archive does not retain real GPU unresolved references");}
  if (defined.length || imp.length) {throw new Error("Windows Core symbols define or import a GPU API");}
  if (runtime.length) {throw new Error("Windows Core symbols contain Node/V8/NAPI references");}
  return {undefinedSymbolCount: undefinedSymbols.length, definedSymbolCount: definedSymbols.length, gpuUnresolvedReferences: unresolved,
    definedGpuReferences: defined, impGpuThunks: imp, forbiddenRuntimeSymbols: runtime};
}
/** @param {unknown} value @param {number} expectedObjectCount */
function parseDumpbinArchive(value, expectedObjectCount) {
  if (typeof value !== "string" || value.length < 1 || value.length > maxSymbolBytes || value.includes("\0") ||
      !Number.isSafeInteger(expectedObjectCount) || expectedObjectCount < 1) {
    throw new Error("Windows Core archive audit is malformed or missing its project object count");
  }
  const coffTableCount = (value.match(/^COFF SYMBOL TABLE\s*$/gim) || []).length;
  const machines = [...value.matchAll(/^\s*([0-9a-f]+)\s+machine\s+\(([^)\r\n]+)\)\s*$/gim)];
  const x64MachineCount = machines.filter(match => (match[1] || "").toLowerCase() === "8664" && (match[2] || "").toLowerCase() === "x64").length;
  if (coffTableCount !== expectedObjectCount || machines.length !== expectedObjectCount || x64MachineCount !== expectedObjectCount) {
    throw new Error(`Windows Core archive object/machine count mismatch: project=${expectedObjectCount}, coff=${coffTableCount}, machine=${machines.length}, x64=${x64MachineCount}`);
  }
  const defaultlibDirectives = value.split(/\r?\n/).filter(line => /^\s*\/DEFAULTLIB\s*:/i.test(line));
  const forbiddenDefaultlibs = defaultlibDirectives.filter(line => /(?:node|v8|sycl)/i.test(line));
  if (forbiddenDefaultlibs.length) {throw new Error("Windows Core archive contains Node/V8/SYCL defaultlib directives");}
  return {coffTableCount, machineCount: machines.length, x64MachineCount, defaultlibDirectives, ...parseDumpbinSymbols(value)};
}
/** @param {string} archive @param {string} dumpbin @param {string} cwd @param {string} log @param {number} expectedObjectCount
 * @param {ToolSpawn} [spawn] @param {NodeJS.ProcessEnv} [environment] */
function inspectCoreArchive(archive, dumpbin, cwd, log, expectedObjectCount, spawn = spawnSync, environment = process.env) {
  const before = inspectStaticArchive(archive);
  requireFile(dumpbin, "Windows Core dumpbin"); requireDirectory(cwd, "Windows Core dumpbin working directory");
  const output = runToolText(dumpbin, ["/headers", "/directives", "/symbols", archive], cwd, log, spawn, environment);
  const audit = parseDumpbinArchive(output, expectedObjectCount), after = inspectStaticArchive(archive);
  if (after.bytes !== before.bytes || after.mtimeMs !== before.mtimeMs || after.ctimeMs !== before.ctimeMs) {throw new Error("Windows Core archive changed during symbol inspection");}
  return {path: archive, bytes: after.bytes, sha256: fileHash(archive), symbolsLog: log, ...audit, stubs: false};
}
/** @param {string} file @param {unknown} value */
function writeJson(file, value) {fs.writeFileSync(file, JSON.stringify(value, null, 2) + "\n", {flag: "wx"});}
/** @param {Record<string, string>} before @param {Record<string, string>} after @param {string} label */
function verifyHashes(before, after, label) {if (JSON.stringify(before) !== JSON.stringify(after)) {throw new Error(`${label} changed during Windows Core build`);}}
// VM tar can materialize source junctions as ordinary copies. Recover only exact, link-free
// copies so MSBuild keeps canonical repo/tlog paths; remove verified staging junctions before export.
// This protects the cooperative cache owner, not against a hostile same-UID writer.
/** @typedef {{directories: string[], files: Record<string, string>}} CoreSourceTree */
/** @param {string} directory @returns {CoreSourceTree} */
function coreSourceTreeSnapshot(directory) {
  const root = fs.lstatSync(directory);
  if (!root.isDirectory() || root.isSymbolicLink() || !sameNativePath(fs.realpathSync(directory), path.resolve(directory))) {
    throw new Error("Windows Core source tree is redirected or is not a real directory: " + directory);
  }
  /** @type {string[]} */ const directories = [];
  /** @type {Record<string, string>} */ const files = {};
  /** @param {string} current @param {string} relative */
  const visit = (current, relative) => {
    const parent = fs.lstatSync(current);
    if (!parent.isDirectory() || parent.isSymbolicLink() || !sameNativePath(fs.realpathSync(current), path.resolve(current))) {
      throw new Error("Windows Core source tree contains a redirected directory: " + current);
    }
    if (relative) {directories.push(relative);}
    for (const entry of fs.readdirSync(current, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      if (!entry.name || entry.name === "." || entry.name === ".." || /[\\/:\0]/.test(entry.name)) {
        throw new Error("Windows Core source tree contains an unsafe name: " + path.join(current, entry.name));
      }
      const child = path.join(current, entry.name), childRelative = relative ? relative + "/" + entry.name : entry.name;
      const stat = fs.lstatSync(child);
      if (stat.isSymbolicLink()) {throw new Error("Windows Core source tree contains a link: " + child);}
      if (stat.isDirectory()) {visit(child, childRelative); continue;}
      if (!stat.isFile() || stat.nlink !== 1) {throw new Error("Windows Core source tree contains a non-regular or hard-linked file: " + child);}
      const digest = fileHash(child), after = fs.lstatSync(child);
      if (!after.isFile() || after.isSymbolicLink() || after.nlink !== 1 || after.dev !== stat.dev || after.ino !== stat.ino) {
        throw new Error("Windows Core source file changed while checking its tree: " + child);
      }
      files[childRelative] = digest;
    }
    const after = fs.lstatSync(current);
    if (!after.isDirectory() || after.isSymbolicLink() || after.dev !== parent.dev || after.ino !== parent.ino) {
      throw new Error("Windows Core source directory changed while checking its tree: " + current);
    }
  };
  visit(directory, ""); directories.sort();
  return {directories, files};
}
/** @param {string} directory @param {CoreSourceTree} expected */
function removeVerifiedCoreSourceCopy(directory, expected) {
  const owner = path.dirname(directory), ownerStat = fs.lstatSync(owner);
  if (!junctionNames.includes(path.basename(directory)) || !ownerStat.isDirectory() || ownerStat.isSymbolicLink() || !sameNativePath(fs.realpathSync(owner), path.resolve(owner))) {
    throw new Error("Windows Core restored source copy is outside its canonical staging root: " + directory);
  }
  const before = fs.lstatSync(directory), current = coreSourceTreeSnapshot(directory), after = fs.lstatSync(directory);
  if (!isDeepStrictEqual(current, expected) || !before.isDirectory() || before.isSymbolicLink() || after.dev !== before.dev || after.ino !== before.ino ||
      !sameNativePath(fs.realpathSync(directory), path.resolve(directory))) {throw new Error("Windows Core restored source copy changed before recovery: " + directory);}
  const expectedDirectories = new Set(expected.directories), removedDirectories = new Set(), removedFiles = new Set();
  /** @param {string} currentDirectory @param {string} relative */
  const removeTree = (currentDirectory, relative) => {
    const start = fs.lstatSync(currentDirectory);
    if (!start.isDirectory() || start.isSymbolicLink() || !sameNativePath(fs.realpathSync(currentDirectory), path.resolve(currentDirectory))) {
      throw new Error("Windows Core restored source directory changed during recovery: " + currentDirectory);
    }
    const entries = fs.readdirSync(currentDirectory, {withFileTypes: true});
    const opened = fs.lstatSync(currentDirectory);
    if (!opened.isDirectory() || opened.dev !== start.dev || opened.ino !== start.ino || !sameNativePath(fs.realpathSync(currentDirectory), path.resolve(currentDirectory))) {
      throw new Error("Windows Core restored source directory changed during recovery: " + currentDirectory);
    }
    for (const entry of entries) {
      if (!entry.name || entry.name === "." || entry.name === ".." || /[\\/:\0]/.test(entry.name)) {throw new Error("Unsafe restored source name: " + entry.name);}
      const child = path.join(currentDirectory, entry.name), childRelative = relative ? relative + "/" + entry.name : entry.name;
      const stat = fs.lstatSync(child);
      if (stat.isSymbolicLink()) {throw new Error("Restored source acquired a link: " + child);}
      if (stat.isDirectory()) {
        if (!expectedDirectories.has(childRelative)) {throw new Error("Restored source acquired a directory: " + child);}
        removeTree(child, childRelative);
        const after = fs.lstatSync(child);
        if (!after.isDirectory() || after.isSymbolicLink() || after.dev !== stat.dev || after.ino !== stat.ino) {throw new Error("Restored source directory changed during removal: " + child);}
        fs.rmdirSync(child); removedDirectories.add(childRelative); continue;
      }
      if (!stat.isFile() || stat.nlink !== 1 || !Object.hasOwn(expected.files, childRelative) || fileHash(child) !== expected.files[childRelative]) {
        throw new Error("Restored source acquired or changed a file: " + child);
      }
      const after = fs.lstatSync(child);
      if (!after.isFile() || after.isSymbolicLink() || after.nlink !== 1 || after.dev !== stat.dev || after.ino !== stat.ino) {throw new Error("Restored source file changed during removal: " + child);}
      fs.unlinkSync(child); removedFiles.add(childRelative);
    }
    const finish = fs.lstatSync(currentDirectory);
    if (!finish.isDirectory() || finish.isSymbolicLink() || finish.dev !== start.dev || finish.ino !== start.ino ||
        !sameNativePath(fs.realpathSync(currentDirectory), path.resolve(currentDirectory))) {throw new Error("Restored source directory changed during removal: " + currentDirectory);}
  };
  removeTree(directory, "");
  if (removedDirectories.size !== expected.directories.length || removedFiles.size !== Object.keys(expected.files).length) {
    throw new Error("Restored source membership changed during removal: " + directory);
  }
  fs.rmdirSync(directory);
}
/** @param {ReturnType<typeof windowsCorePlan>} plan @param {{junctions: Record<string, string>}} staged */
function verifyCoreSourceJunctions(plan, staged) {
  for (const directory of [path.dirname(plan.paths.source), plan.paths.source]) {
    const stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || !sameNativePath(fs.realpathSync(directory), path.resolve(directory))) {
      throw new Error("Windows Core source staging root was redirected: " + directory);
    }
  }
  for (const name of junctionNames) {
    const destination = path.join(plan.paths.source, name), expected = path.join(plan.root, name), target = staged.junctions[name];
    if (typeof target !== "string" || !sameNativePath(target, expected)) {throw new Error("Windows Core staged junction plan changed: " + destination);}
    const stat = fs.lstatSync(destination);
    if (!stat.isSymbolicLink() || !sameNativePath(fs.realpathSync(destination), fs.realpathSync(expected))) {
      throw new Error("Windows Core staged source junction changed: " + destination);
    }
  }
}
/** @param {ReturnType<typeof windowsCorePlan>} plan @param {{junctions: Record<string, string>}} staged */
function cleanupCoreSourceJunctions(plan, staged) {
  const sourceParent = path.dirname(plan.paths.source);
  if (!sameNativePath(plan.paths.source, path.join(plan.output, "source"))) {throw new Error("Windows Core source staging path changed: " + plan.paths.source);}
  for (const directory of [sourceParent, plan.paths.source]) {
    const stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || !sameNativePath(fs.realpathSync(directory), path.resolve(directory))) {
      throw new Error("Windows Core source staging root was redirected: " + directory);
    }
  }
  const remove = [];
  for (const name of Object.keys(staged.junctions)) {
    const destination = path.join(plan.paths.source, name), expected = path.join(plan.root, name), target = staged.junctions[name];
    if (!junctionNames.includes(name) || typeof target !== "string" || !sameNativePath(target, expected)) {throw new Error("Windows Core staged junction cleanup plan changed: " + destination);}
    let stat;
    try {stat = fs.lstatSync(destination);} catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") {continue;}
      throw error;
    }
    if (!stat.isSymbolicLink() || !sameNativePath(fs.realpathSync(destination), fs.realpathSync(expected))) {
      throw new Error("Windows Core refusing to remove a changed source junction: " + destination);
    }
    remove.push({destination, expected: fs.realpathSync(expected), dev: stat.dev, ino: stat.ino});
  }
  for (const entry of remove) {
    for (const directory of [sourceParent, plan.paths.source]) {
      const stat = fs.lstatSync(directory);
      if (!stat.isDirectory() || stat.isSymbolicLink() || !sameNativePath(fs.realpathSync(directory), path.resolve(directory))) {
        throw new Error("Windows Core source staging root changed during cleanup: " + directory);
      }
    }
    const stat = fs.lstatSync(entry.destination);
    if (!stat.isSymbolicLink() || stat.dev !== entry.dev || stat.ino !== entry.ino || !sameNativePath(fs.realpathSync(entry.destination), entry.expected)) {
      throw new Error("Windows Core source junction changed during cleanup: " + entry.destination);
    }
    if (process.platform === "win32") {fs.rmdirSync(entry.destination);} else {fs.unlinkSync(entry.destination);}
  }
}
/** @param {ReturnType<typeof windowsCorePlan>} plan */
function stageSourceTree(plan) {
  for (const directory of [plan.paths.source, plan.paths.devdir, plan.paths.msbuildUserExtensions]) {
    try {fs.mkdirSync(directory, {recursive: true, mode: 0o700});} catch (error) {
      if (!(error instanceof Error) || !("code" in error) || error.code !== "EEXIST") {throw error;}
    }
    const stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || !sameNativePath(fs.realpathSync(directory), directory)) {throw new Error("Windows Core work directory is redirected: " + directory);}
  }
  /** @type {Array<{name: string, destination: string, target: string, present?: boolean, recovery?: CoreSourceTree}>} */
  const junctions = [];
  for (const name of junctionNames) {
    const destination = path.join(plan.paths.source, name), target = path.join(plan.root, name);
    const targetStat = fs.lstatSync(target);
    if (!targetStat.isDirectory() || targetStat.isSymbolicLink() || !sameNativePath(fs.realpathSync(target), path.resolve(target))) {
      throw new Error("Maintained Windows Core source is redirected or is not a real directory: " + target);
    }
    try {
      const stat = fs.lstatSync(destination);
      if (stat.isSymbolicLink()) {
        if (!sameNativePath(fs.realpathSync(destination), fs.realpathSync(target))) {throw new Error("Windows Core source junction has an unexpected target: " + destination);}
        junctions.push({name, destination, target, present: true});
      } else if (stat.isDirectory() && sameNativePath(fs.realpathSync(destination), destination)) {
        const expected = coreSourceTreeSnapshot(target);
        if (!isDeepStrictEqual(coreSourceTreeSnapshot(destination), expected)) {throw new Error("Windows Core restored source copy differs from maintained source: " + destination);}
        junctions.push({name, destination, target, recovery: expected});
      } else {throw new Error("Windows Core source path is not a safe junction or restored directory: " + destination);}
    } catch (error) {
      if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {throw error;}
      junctions.push({name, destination, target});
    }
  }
  const recoveredCopies = [];
  /** @type {Record<string, string>} */ const known = Object.fromEntries(junctions.filter(entry => entry.present).map(entry => [entry.name, entry.target]));
  try {
    for (const entry of junctions) {
      if (entry.recovery) {removeVerifiedCoreSourceCopy(entry.destination, entry.recovery); recoveredCopies.push(entry.name);}
      if (!fs.existsSync(entry.destination)) {fs.symlinkSync(entry.target, entry.destination, "junction"); known[entry.name] = entry.target;}
    }
    const maintainedBinding = path.join(plan.root, "binding.gyp"), expectedHash = fileHash(maintainedBinding);
    let stagedHash = null;
    try {
      const stat = fs.lstatSync(plan.paths.binding);
      if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Windows Core staged binding.gyp is unsafe");}
      stagedHash = fileHash(plan.paths.binding);
    } catch (error) {if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {throw error;}}
    if (stagedHash !== expectedHash) {
      const temp = plan.paths.binding + ".tmp-" + randomBytes(8).toString("hex");
      fs.copyFileSync(maintainedBinding, temp, fs.constants.COPYFILE_EXCL);
      if (fs.existsSync(plan.paths.binding)) {fs.unlinkSync(plan.paths.binding);}
      fs.renameSync(temp, plan.paths.binding);
    }
    if (fileHash(plan.paths.binding) !== expectedHash) {throw new Error("Private binding.gyp copy differs from maintained input");}
    const staged = {source: plan.paths.source, junctions: Object.fromEntries(junctionNames.map(name => [name, path.join(plan.root, name)])), binding: plan.paths.binding, recoveredCopies};
    verifyCoreSourceJunctions(plan, staged);
    return staged;
  } catch (error) {
    try {
      if (Object.keys(known).length) {cleanupCoreSourceJunctions(plan, {junctions: known});}
    } catch (cleanupError) {
      if (error instanceof Error && cleanupError instanceof Error) {error.message += "; junction rollback also failed: " + cleanupError.message;} else {
        console.error("Windows Core junction rollback also failed", cleanupError);
      }
    }
    throw error;
  }
}
/** @param {NodeJS.ProcessEnv} environment */
function validatePortableEnvironment(environment) {
  for (const key of Object.keys(environment)) {
    const normalized = key.toUpperCase();
    if (forbiddenCompilerEnvironment.has(normalized) && environment[key]) {
      throw new Error(`Windows Core portable build rejects inherited ${key}`);
    }
  }
}
/** @param {NodeJS.ProcessEnv} environment @param {string} python @param {boolean} [intel] @returns {NodeJS.ProcessEnv} */
function sanitizeBuildEnvironment(environment, python, intel = false) {
  const copy = {...environment};
  for (const key of Object.keys(copy)) {
    if (forbiddenCompilerEnvironment.has(key.toUpperCase()) ||
        /^(?:npm_config_|gyp_|node_options$|node_path$|node_gyp_force_python$|python)|^(?:directorybuild(?:props|targets)path$|msbuilduserextensionspath$|cltool(?:exe|path)$|usemultitooltask$)/i.test(key) ||
        ["MOM_NATIVE_BUILD_OUTPUT", "MOM_NATIVE_BUILD_STAGE", "MOM_NATIVE_OUTPUT", "MOM_NATIVE_SDK", "MOM_NATIVE_UPX", "MOM_NATIVE_UPX_SHA256", "HOSTNAME",
          "MOM_FORCE_REBUILD", "MOM_NATIVE_CACHE"].includes(key.toUpperCase())) {
      delete copy[key];
    }
  }
  copy["PYTHON"] = python; copy["MOM_AOT_BUILD_ONLY"] = "1";
  // Use the compiler's public codegen switch, not private Intel helper shims. Unlike
  // -fno-builtin, it preserves builtin setjmp semantics; CL is hashed with the cache inputs.
  if (intel) {copy["CL"] = "/Q_no-use-libirc";}
  return copy;
}
/** @param {string} configFile @param {string} output @param {NodeJS.ProcessEnv} [environment] @param {{spawn?: ToolSpawn}} [hooks] */
function buildNativeWindowsCore(configFile, output, environment = process.env, hooks = {}) {
  if (process.platform !== "win32" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("Windows Core build requires Windows and MOM_AOT_BUILD_ONLY=1; no execution is implicit");
  }
  validatePortableEnvironment(environment);
  requireFile(configFile, "Windows Core descriptor");
  const descriptorSha256 = fileHash(configFile), config = validateWindowsCoreDescriptor(readBuildJson(configFile));
  if (fileHash(configFile) !== descriptorSha256) {throw new Error("Windows Core descriptor changed during admission");}
  const plan = windowsCorePlan(config, output, repo), cacheRoot = windowsCoreCacheRoot(repo), releaseLock = acquireWindowsCoreCacheLock(cacheRoot);
  /** @type {ReturnType<typeof stageSourceTree> | null} */ let staged = null;
  let cleanupPlan = plan;
  /** @type {unknown} */ let buildError;
  let buildFailed = false;
  try {
    validateOutput(output, repo);
    if (fileHash(configFile) !== descriptorSha256) {throw new Error("Windows Core descriptor changed while acquiring its cache slot");}
    verifySourceInputs(plan.sourceState, nativeSourceInputs(repo, "cpu"), "Windows Core pre-build");
    verifyHashes(plan.recipeHashes, hashFiles(Object.keys(plan.recipeHashes)), "Windows Core recipe inputs");
    verifyHashes(plan.toolHashes, hashFiles(Object.keys(plan.toolHashes)), "Windows Core tool inputs");
    verifyHashes(plan.headerHashes, hashFiles(libuvHeaderFiles(config.libuvInclude)), "Windows Core header inputs");

    const environmentCopy = sanitizeBuildEnvironment(environment, config.python, Boolean(config.compiler)), environmentSha256 = windowsCoreEnvironmentSha256(environmentCopy);
    const workPlan = windowsCoreWorkPlan(plan, cacheRoot), statePath = path.join(cacheRoot, "core-state.json"), markerPath = path.join(cacheRoot, "build-in-progress.json");
    cleanupPlan = workPlan;
    fs.mkdirSync(plan.output);
    const marker = windowsCoreMarkerPresent(markerPath), state = readWindowsCoreCacheState(statePath);
    staged = stageSourceTree(workPlan);
    const includeBefore = windowsCoreIncludeRoots(config, plan.msvc, environmentCopy);
    const trackerBefore = windowsCoreTrackerSnapshot(workPlan.paths.build, cacheRoot, plan, environmentCopy, false,
      path.join(plan.output, "msbuild-tracker.before.json"));
    const current = windowsCoreCacheView(plan, descriptorSha256, environmentSha256, includeBefore, trackerBefore, workPlan.paths);
    let archiveValid = false;
    if (current.archiveSha256 && state?.archiveSha256 === current.archiveSha256) {
      try {inspectStaticArchive(workPlan.paths.archive); archiveValid = true;} catch {archiveValid = false;}
    }
    const forceRebuild = environment["MOM_FORCE_REBUILD"] === "1" || environment["MOM_NATIVE_CACHE"] === "0";
    const decision = windowsCoreCacheDecision({state, current, marker, projectExists: Boolean(current.projectSha256), archiveValid, forceRebuild});
    markWindowsCoreInProgress(markerPath);

    writeJson(path.join(plan.output, "plan.json"), {target, sourceInventory: plan.sourceInventory, compiler: plan.compiler,
      configure: workPlan.configure, build: {...workPlan.build, args: [...workPlan.build.args, "/t:" + decision.target]}, audit: workPlan.audit,
      paths: plan.paths, workRoot: cacheRoot, incremental: decision});
    writeJson(path.join(plan.output, "source-hashes.before.json"), plan.sourceState); writeJson(path.join(plan.output, "tool-hashes.before.json"), plan.toolHashes);
    writeJson(path.join(plan.output, "header-hashes.before.json"), plan.headerHashes); writeJson(path.join(plan.output, "recipe-hashes.before.json"), plan.recipeHashes);

    const spawn = hooks.spawn || spawnSync;
    const compiler = plan.compiler ? {path: plan.compiler.command, ...intelCoreCompilerVersion(runToolText(plan.compiler.command,
      plan.compiler.args, workPlan.paths.source, path.join(plan.output, "compiler-version.log"), spawn, environmentCopy, 30000))} : null;
    let configured = false;
    if (decision.configure) {
      runTool(workPlan.configure.command, workPlan.configure.args, workPlan.paths.source, path.join(plan.output, "node-gyp-configure.log"), spawn, environmentCopy, 300000);
      configured = true;
    }
    const project = inspectCoreProject(workPlan.paths.project, "Windows Core project", plan.config.libuvInclude);
    writeJson(path.join(plan.output, "project-audit.json"), project);
    if (decision.target === "Rebuild") {clearWindowsCoreRelease(workPlan.paths.build);}
    const build = {...workPlan.build, args: [...workPlan.build.args, "/t:" + decision.target]};
    verifyCoreSourceJunctions(workPlan, staged);
    runTool(build.command, build.args, workPlan.paths.source, path.join(plan.output, "msbuild.log"), spawn, environmentCopy, 900000);
    verifyCoreSourceJunctions(workPlan, staged);
    const artifact = inspectCoreArchive(workPlan.paths.archive, plan.config.dumpbin, workPlan.paths.source, path.join(plan.output, "dumpbin-symbols.log"), project.objectCount, spawn, environmentCopy);
    const sourceAfter = nativeSourceInputs(repo, "cpu"); verifySourceInputs(plan.sourceState, sourceAfter, "Windows Core");
    // The public artifact and audit are fresh even when MSBuild reused its stable object graph.
    fs.copyFileSync(artifact.path, plan.paths.exportedArchive, fs.constants.COPYFILE_EXCL);
    if (fileHash(plan.paths.exportedArchive) !== artifact.sha256) {throw new Error("Exported Windows Core archive differs from audited input");}
    artifact.path = plan.paths.exportedArchive;
    const toolAfter = hashFiles(Object.keys(plan.toolHashes)), headerAfter = hashFiles(libuvHeaderFiles(plan.config.libuvInclude)), recipeAfter = hashFiles(Object.keys(plan.recipeHashes));
    verifyHashes(plan.toolHashes, toolAfter, "Windows Core tool inputs"); verifyHashes(plan.headerHashes, headerAfter, "Windows Core header inputs"); verifyHashes(plan.recipeHashes, recipeAfter, "Windows Core recipe inputs");
    if (fileHash(configFile) !== descriptorSha256) {throw new Error("Windows Core descriptor changed during compilation");}
    const includeAfter = windowsCoreIncludeRoots(config, plan.msvc, environmentCopy);
    const trackerAfter = windowsCoreTrackerSnapshot(workPlan.paths.build, cacheRoot, plan, environmentCopy, false,
      path.join(plan.output, "msbuild-tracker.after.json"));
    if (!isDeepStrictEqual(includeBefore, includeAfter) || trackerBefore.membership !== trackerAfter.membership ||
        trackerBefore.powershellSha256 !== trackerAfter.powershellSha256 || !isDeepStrictEqual(trackerBefore.toolchain, trackerAfter.toolchain)) {
      throw new Error("Windows Core SDK/toolchain inputs changed during compilation");
    }
    verifyWindowsCoreTrackerInputs(trackerBefore, trackerAfter);
    writeJson(path.join(plan.output, "source-hashes.after.json"), sourceAfter); writeJson(path.join(plan.output, "tool-hashes.after.json"), toolAfter);
    writeJson(path.join(plan.output, "header-hashes.after.json"), headerAfter); writeJson(path.join(plan.output, "recipe-hashes.after.json"), recipeAfter);

    // A cleanup error must stop before the cache state, receipt, or success marker is committed.
    verifyCoreSourceJunctions(workPlan, staged);
    cleanupCoreSourceJunctions(workPlan, staged);
    const stagedReceipt = {...staged, junctionsRemovedAfterAudit: true}; staged = null;
    const after = windowsCoreCacheView(plan, descriptorSha256, environmentSha256, includeAfter, trackerAfter, workPlan.paths);
    const incrementalSafe = after.incrementalSafe && !forceRebuild;
    const cacheState = {...after, version: windowsCoreCacheVersion, incrementalSafe, archiveSha256: artifact.sha256, lastSuccessMs: Date.now()};
    writeCoreCacheJson(statePath, cacheState);
    const cacheStateSha256 = fileHash(statePath);
    const incremental = {workRoot: cacheRoot, stateSha256: cacheStateSha256, target: decision.target, reason: decision.reason, configured,
      incrementalSafe, trackerInputs: Object.keys(trackerAfter.inputs).length, trackedOutputs: Object.keys(trackerAfter.artifacts).length,
      toolchainInputs: Object.keys(trackerAfter.toolchain).length, includeRoots: includeAfter.roots, includeMembershipSha256: includeAfter.sha256,
      powershellSha256: trackerAfter.powershellSha256, coldControl: forceRebuild};
    const receipt = {version: 1, target, gpu: true, portable: true, portability: {portable: true, reason: "Maintained generic v143 /MT /O2 /fp:strict GYP Core graph; no host-tuned flags"}, compiler,
      objectSha256: artifact.sha256, objects: [artifact.path], archive: artifact, sourceState: sourceAfter, sourceInventory: plan.sourceInventory,
      recipeHashes: recipeAfter, toolHashes: toolAfter, headerHashes: headerAfter, descriptorSha256,
      commands: [...(plan.compiler ? [plan.compiler] : []), ...(configured ? [workPlan.configure] : []), build, workPlan.audit], staged: stagedReceipt, incremental,
      buildOnly: true, executed: false, publicationQualified: false, scope: "Maintained Windows mom_native_core static archive with unresolved GPU API references; no Node/V8/NAPI/SYCL project dependency or candidate execution"};
    writeJson(path.join(plan.output, "inputs.json"), receipt);
    clearWindowsCoreInProgress(markerPath);
    return receipt;
  } catch (error) {buildError = error; buildFailed = true; throw error;} finally {
    try {
      if (staged) {cleanupCoreSourceJunctions(cleanupPlan, staged);}
    } catch (cleanupError) {
      if (buildError instanceof Error && cleanupError instanceof Error) {buildError.message += "; junction cleanup also failed: " + cleanupError.message;} else if (!buildFailed) {
        console.error("Windows Core junction cleanup failed", cleanupError);
      } else {console.error("Windows Core junction cleanup also failed", cleanupError);}
    } finally {releaseLock();}
  }
}

if (require.main === module) {
  try {
    const [configFile, output, extra] = process.argv.slice(2);
    if (!configFile || !output || extra) {throw new Error("Usage: node scripts/build-native-windows-core.js /absolute/descriptor.json /new/absolute/output");}
    buildNativeWindowsCore(configFile, output);
  } catch (error) {console.error(error instanceof Error ? error.message : "Windows Core build failed"); process.exitCode = 1;}
}

module.exports = {absolutePath, validateWindowsCoreDescriptor, intelCoreCompilerVersion, validateOutput, validatePortableEnvironment, sanitizeBuildEnvironment, windowsCorePlan,
  windowsCoreCacheRoot, acquireWindowsCoreCacheLock, windowsCoreIncludeMembership, windowsCoreIncludeRoots, windowsCoreEnvironmentSha256, windowsCoreCacheDecision, clearWindowsCoreRelease,
  windowsCoreTrackerRowsScript, windowsCoreTrackerEnvironment, verifyWindowsCoreTrackerInputs,
  runTool, runToolText, parseDumpbinSymbols, parseDumpbinArchive, inspectCoreProject, inspectCoreArchive, stageSourceTree, cleanupCoreSourceJunctions, buildNativeWindowsCore};
