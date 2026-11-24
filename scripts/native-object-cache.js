#!/usr/bin/env node
"use strict";

// Native release objects are rebuilt into a fresh evidence directory on every run. This
// cache only avoids recompiling individual C/C++ translation units; links and all admission
// checks remain owned by the producer.
const fs = require("node:fs"), path = require("node:path");
const crypto = require("node:crypto");
const {spawnSync} = require("node:child_process");
const {fileHash, sortedBuildEntries, buildDependencies} = require("./build-source-inputs");

const MAX_COMMANDS = 65536, MAX_ARGS = 65536, MAX_FILES = 200000, MAX_JSON = 8 << 20;
const generatedSuffix = /\.(?:c|cc|cpp|cxx|cuh|h|hh|hpp|hxx|inc|inl|ipp)$/i;
const orchestrationEnvironment = new Set(["MOM_NATIVE_BUILD_OUTPUT", "MOM_NATIVE_BUILD_STAGE", "MOM_NATIVE_OUTPUT", "MOM_NATIVE_SDK",
  "MOM_NATIVE_UPX", "MOM_NATIVE_UPX_SHA256", "HOSTNAME", "MOM_FORCE_REBUILD", "GITHUB_JOB",
  "MOM_NATIVE_APPLICATION_DEADLINE_EPOCH", "MOM_NATIVE_COMPONENT_DEADLINE_EPOCH",
  "MOM_NATIVE_WINDOWS_CONTROLLER_DEADLINE_EPOCH", "MOM_NATIVE_WINDOWS_CORE_DEADLINE_EPOCH",
  "MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH", "MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH",
  "MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH", "MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH",
  "CMAKE_BUILD_PARALLEL_LEVEL"]);
/** @typedef {Record<string, string>} NativeEnvironment */
/** @typedef {{selected: string, real: string, sha256: string}} ToolIdentity */
/** @typedef {{command: string, args: string[], source: string, object: string, depfile: string,
 * timeout?: number, generatedInputs?: string[], dependencyFormat?: "make" | "nmake"}} NativeCommand */
/** @typedef {{repo: string, output: string, namespace: string, commands: NativeCommand[],
 * environment?: NodeJS.ProcessEnv, ninja?: string, jobs?: number, includeRoots?: string[],
 * keyFiles?: string[],
 * finalize?: boolean, liveObjects?: string[],
 * resolveDependencies?: (input: {command: NativeCommand, depfileText: string, repo: string}) => string[]}} NativeObjectOptions */
/** @typedef {{entry: NativeCommand, source: string, object: string, depfile: string, compiler: ToolIdentity,
 * args: string[], stableSource: string, cachedObject: string, cachedDepfile: string, stateFile: string,
 * signature: string, base: {sourceSha256: string, generatedHashes: Array<[string, string]>}, slot: string,
 * id: string, objectKey: string, cacheHit: boolean, dependencies: Array<{path: string, sha256: string}>,
 * startedAt: number, status?: number | null, stdout?: string, stderr?: string, error?: string,
 * signal?: NodeJS.Signals | null, objectSha256?: string, depfileSha256?: string}} NormalizedCommand */
/** @param {string | Buffer} value */
const digest = value => crypto.createHash("sha256").update(value).digest("hex");

/** @param {string} file @param {string} label */
function regularFile(file, label) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error(label + " is not a regular file: " + file);}
  return fs.realpathSync(file);
}

/** Resolve Windows environment keys case-insensitively after process.env has been copied.
 * Reject conflicting aliases while leaving the original child environment unchanged.
 * @param {NativeEnvironment} env @param {string} name @param {NodeJS.Platform} [platform]
 * @returns {string | undefined}
 */
function environmentValue(env, name, platform = process.platform) {
  if (platform !== "win32") {return env[name];}
  const keys = Object.keys(env).filter(key => key.toLowerCase() === name.toLowerCase());
  const values = new Set(keys.map(key => env[key]));
  if (values.size > 1) {throw new Error("Conflicting case variants for Windows environment variable: " + name);}
  return keys.length ? env[keys[0] || ""] : undefined;
}

/** Resolve one explicitly selected executable without invoking a shell.
 * @param {string} name @param {NativeEnvironment} env @returns {ToolIdentity}
 */
function resolveExecutable(name, env) {
  if (typeof name !== "string" || !name || /[\0\r\n]/.test(name)) {throw new Error("Invalid native compiler path");}
  const searchPath = environmentValue(env, "PATH") || "", pathExt = environmentValue(env, "PATHEXT") || ".COM;.EXE";
  const paths = path.isAbsolute(name) || name.includes(path.sep) ? [name] :
    searchPath.split(path.delimiter).filter(Boolean).flatMap(dir => {
      const base = path.join(dir, name);
      if (process.platform !== "win32" || path.extname(name)) {return [base];}
      return [base, ...(pathExt.split(";").filter(ext => [".COM", ".EXE"].includes(ext.toUpperCase())).map(ext => base + ext.toLowerCase()))];
    });
  const candidates = paths;
  for (const candidate of candidates) {
    try {
      const selected = path.resolve(candidate), stat = fs.statSync(selected), real = fs.realpathSync(selected);
      if (!stat.isFile() || !fs.statSync(real).isFile()) {throw new Error("Native tool is not a regular file: " + selected);}
      if (process.platform !== "win32") {fs.accessSync(real, fs.constants.X_OK);}
      return {selected, real, sha256: fileHash(real)};
    } catch (error) {
      if (error instanceof Error && "code" in error && ["ENOENT", "ENOTDIR", "EACCES"].includes(String(error.code))) {continue;}
      throw error;
    }
  }
  throw new Error("Cannot resolve native tool: " + name);
}

/** Paths in a fresh output are mapped into one stable, cache-owned source-like input tree.
 * @param {string} file @param {string} label */
function ensureDirectory(file, label) {
  const resolved = path.resolve(file);
  const parsed = path.parse(resolved), components = resolved.slice(parsed.root.length).split(path.sep).filter(Boolean);
  let current = parsed.root;
  for (const component of components) {
    current = path.join(current, component);
    try {
      const stat = fs.lstatSync(current);
      if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error(label + " contains a non-directory or symlink: " + current);}
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") {fs.mkdirSync(current, {mode: 0o700});} else {throw error;}
    }
  }
  return resolved;
}

/** @param {string} root @param {string} file */
function contained(root, file) {
  const relative = path.relative(root, file);
  return relative === "" || (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep));
}

/** Mirror every source-like file in this fresh, isolated output so include probes such as
 * __has_include observe the same namespace for every object in the batch. Never follow links. */
/** @param {string} output @returns {Array<{file: string, relative: string, sha256: string}>} */
function generatedFiles(output) {
  /** @type {Array<{file: string, relative: string, sha256: string}>} */
  const files = [];
  /** @param {string} dir @param {string} prefix */
  function walk(dir, prefix) {
    for (const entry of sortedBuildEntries(dir)) {
      const file = path.join(dir, entry.name), relative = prefix ? prefix + "/" + entry.name : entry.name;
      if (entry.isDirectory()) {walk(file, relative); continue;}
      if (entry.isSymbolicLink()) {
        try {
          if (fs.statSync(file).isDirectory()) {throw new Error("Generated native output contains a directory symlink: " + file);}
        } catch (error) {
          if (error instanceof Error && error.message.startsWith("Generated native output contains")) {throw error;}
          if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {throw error;}
        }
        if (generatedSuffix.test(entry.name)) {throw new Error("Generated native output contains a source-like symlink: " + file);}
        continue;
      }
      if (!entry.isFile()) {
        if (generatedSuffix.test(entry.name)) {throw new Error("Generated native output contains a non-file source-like entry: " + file);}
        continue;
      }
      if (!generatedSuffix.test(entry.name)) {continue;}
      if (files.length >= MAX_FILES) {throw new Error("Fresh output has too many source-like generated inputs");}
      files.push({file, relative, sha256: fileHash(regularFile(file, "Generated native input"))});
    }
  }
  walk(output, "");
  return files;
}

/** Inventory selected include trees by membership only. Contents of actual dependencies are
 * hashed per object from its saved compiler depfile, avoiding a full SDK content scan. */
/** @param {string[]} roots @param {string} output @param {string[]} generated @param {string} repo */
function includeMembership(roots, output, generated, repo) {
  const result = [];
  for (const selected of [...new Set(roots)].sort()) {
    if (typeof selected !== "string" || !path.isAbsolute(selected) || /[\0\r\n]/.test(selected)) {throw new Error("Invalid native include root");}
    const root = path.resolve(selected);
    if (contained(output, root)) {
      const rel = path.relative(output, root);
      const names = generated.filter(file => contained(root, file)).map(file => path.relative(root, file).split(path.sep).join("/")).sort();
      result.push({root: rel ? "output/" + rel.split(path.sep).join("/") : "output", entries: names.map(name => [name, "file"])});
      continue;
    }
    const physical = fs.realpathSync(root), stat = fs.lstatSync(physical);
    if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("Native include root is not a real directory: " + selected);}
    const entries = [];
    const active = new Set();
    /** @param {string} dir @param {string} prefix */
    function walk(dir, prefix) {
      const realDir = fs.realpathSync(dir);
      if (active.has(realDir)) {return;}
      active.add(realDir);
      for (const entry of sortedBuildEntries(dir)) {
        if (entries.length >= MAX_FILES) {throw new Error("Native include root has too many entries");}
        const child = path.join(dir, entry.name), rel = prefix ? prefix + "/" + entry.name : entry.name;
        if (entry.isDirectory()) {entries.push([rel, "directory"]); walk(child, rel);} else if (entry.isFile()) {entries.push([rel, "file"]);} else if (entry.isSymbolicLink()) {
          entries.push([rel, "symlink", fs.readlinkSync(child)]);
          try {if (fs.statSync(child).isDirectory()) {walk(child, rel);}} catch (error) {
            if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {throw error;}
          }
        } else {throw new Error("Unsupported entry in native include root: " + child);}
      }
      active.delete(realDir);
    }
    if (physical === repo) {
      const exclusions = new Set([".git", "build", "node_modules", "vendor", "deps", "dist", "coverage", ".cache"]);
      const maintained = new Set(["native", "runtime", "sycl", "xmrig"]);
      active.add(physical);
      for (const entry of sortedBuildEntries(physical)) {
        if (entries.length >= MAX_FILES) {throw new Error("Native repository include root has too many entries");}
        const child = path.join(physical, entry.name), rel = entry.name;
        if (entry.isDirectory()) {
          entries.push([rel, "directory"]);
          if (maintained.has(entry.name) && !exclusions.has(entry.name)) {walk(child, rel);}
        } else if (entry.isFile()) {entries.push([rel, "file"]);} else if (entry.isSymbolicLink()) {entries.push([rel, "symlink", fs.readlinkSync(child)]);} else {throw new Error("Unsupported entry in native repository include root: " + child);}
      }
      active.delete(physical);
    } else {walk(physical, "");}
    result.push({root: physical, entries});
  }
  return digest(JSON.stringify(result));
}

/** Hash command environment without persisting its values in cache metadata.
 * @param {NativeEnvironment} env */
function environmentHash(env) {
  const values = Object.fromEntries(Object.keys(env).sort().map(name => [name, env[name]]));
  return digest(JSON.stringify(values));
}

/** Hash selected non-depfile inputs once for the batch (e.g. VFS overlays or plugin bitcode).
 * @param {string[]} files
 * @returns {Array<{selected: string, real: string, sha256: string}>} */
function captureKeyFiles(files) {
  if (!Array.isArray(files) || files.length > MAX_FILES ||
      files.some(file => typeof file !== "string" || !path.isAbsolute(file) || /[\0\r\n]/.test(file))) {
    throw new Error("Native key files must be a bounded list of absolute paths");
  }
  return [...new Set(files.map(file => path.resolve(file)))].sort().map(selected => {
    let real;
    try {real = regularFile(selected, "Native key file");} catch (error) {
      throw new Error("Native key file is missing or unsafe: " + selected, {cause: error});
    }
    return {selected, real, sha256: fileHash(real)};
  });
}

/** @param {string} file @param {number} [max] @returns {any} */
function readJson(file, max = MAX_JSON) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 || stat.size > max) {throw new Error("Invalid native object cache metadata: " + file);}
  const text = fs.readFileSync(file, "utf8");
  if (Buffer.byteLength(text) !== stat.size) {throw new Error("Native object cache metadata changed while reading: " + file);}
  return JSON.parse(text);
}

/** Atomically publish small metadata within an already validated cache directory.
 * @param {string} file @param {unknown} value */
function atomicJson(file, value) {
  const temp = file + ".tmp-" + process.pid + "-" + crypto.randomBytes(6).toString("hex");
  const data = JSON.stringify(value) + "\n";
  if (Buffer.byteLength(data) > MAX_JSON) {throw new Error("Native cache metadata exceeds its bound");}
  const fd = fs.openSync(temp, fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL, 0o600);
  try {fs.writeFileSync(fd, data); fs.fsyncSync(fd);} finally {fs.closeSync(fd);}
  fs.renameSync(temp, file);
}

/** Remove only cache-owned regular files; symlinks are rejected, never followed.
 * @param {string} slot */
function clearSlot(slot) {
  if (!fs.existsSync(slot)) {return;}
  const root = fs.lstatSync(slot);
  if (!root.isDirectory() || root.isSymbolicLink()) {throw new Error("Native object cache slot is not a real directory: " + slot);}
  for (const name of fs.readdirSync(slot)) {
    const file = path.join(slot, name), stat = fs.lstatSync(file);
    if (stat.isDirectory() && !stat.isSymbolicLink()) {clearSlot(file); fs.rmdirSync(file);} else if (stat.isFile() && !stat.isSymbolicLink()) {fs.unlinkSync(file);} else {throw new Error("Unexpected native object cache slot entry: " + file);}
  }
}

/** Clear all cache-owned artifacts in a slot without following links.
 * @param {string} slot */
function clearArtifacts(slot) {
  for (const name of fs.readdirSync(slot)) {
    const file = path.join(slot, name), stat = fs.lstatSync(file);
    if (stat.isDirectory() && !stat.isSymbolicLink()) {clearSlot(file); fs.rmdirSync(file);} else if (stat.isFile() && !stat.isSymbolicLink()) {fs.unlinkSync(file);} else {throw new Error("Unexpected native object cache artifact: " + file);}
  }
}

/** Keep at most one slot per logical output across successive batches.
 * @param {string} root @param {string} objectKey @param {string} id */
function updateSlotIndex(root, objectKey, id) {
  const indexFile = path.join(root, "slots.json");
  let index = /** @type {{version?: number, slots?: Record<string, string>}} */ ({});
  try {
    index = readJson(indexFile);
    if (!index || index.version !== 1 || !index.slots || typeof index.slots !== "object" || Array.isArray(index.slots)) {throw new Error("Invalid native object slot index");}
    for (const [key, value] of Object.entries(index.slots)) {
      if (typeof key !== "string" || typeof value !== "string" || !/^[a-f0-9]{40}$/.test(value)) {throw new Error("Invalid native object slot index entry");}
    }
  } catch (error) {
    if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {index = {version: 1, slots: {}};}
  }
  const slots = /** @type {Record<string, string>} */ (index.slots || {}), previous = slots[objectKey];
  slots[objectKey] = id;
  atomicJson(indexFile, {version: 1, slots});
  if (previous && previous !== id) {
    const old = path.join(root, "slots", previous);
    if (fs.existsSync(old)) {clearSlot(old); fs.rmdirSync(old);}
  }
}

/** Prune only after a caller has completed every batch in this object namespace.
 * @param {string} root @param {string} output @param {string[]} liveObjects */
function finalizeSlots(root, output, liveObjects) {
  const live = new Set(liveObjects.map(file => path.relative(output, path.resolve(file)).split(path.sep).join("/")));
  const slotsRoot = path.join(root, "slots");
  if (!fs.existsSync(slotsRoot)) {return;}
  const indexFile = path.join(root, "slots.json");
  /** @type {Record<string, string>} */
  const kept = {};
  for (const name of fs.readdirSync(slotsRoot)) {
    if (!/^[a-f0-9]{40}$/.test(name)) {throw new Error("Unexpected native object cache slot name");}
    const slot = path.join(slotsRoot, name), stat = fs.lstatSync(slot);
    if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("Unsafe native object cache slot");}
    let objectKey = "";
    try {objectKey = readJson(path.join(slot, "state.json")).objectKey;} catch { /* Interrupted or corrupt slots are misses and are pruned on finalize. */ }
    if (typeof objectKey === "string" && live.has(objectKey)) {kept[objectKey] = name;} else {clearSlot(slot); fs.rmdirSync(slot);}
  }
  atomicJson(indexFile, {version: 1, slots: kept});
}

/** Recreate a slot's generated source tree from only this command's declared inputs.
 * @param {string} root */
function resetGeneratedTree(root) {
  if (fs.existsSync(root)) {
    const stat = fs.lstatSync(root);
    if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("Generated native cache tree is unsafe");}
    clearSlot(root); fs.rmdirSync(root);
  }
  fs.mkdirSync(root, {mode: 0o700});
}

/** @param {string} root */
function removeScratch(root) {
  const stat = fs.lstatSync(root);
  if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("Native Ninja scratch directory is unsafe");}
  fs.rmSync(root, {recursive: true, force: false});
}

/** Resolve compiler dependency paths to physical files. @param {NativeCommand} command @param {string} text @param {string} repo */
function defaultDependencies(command, text, repo) {
  const format = command.dependencyFormat === "nmake" ? "nmake" : "make";
  const dependencies = buildDependencies(text, repo, format);
  if (!dependencies?.length) {throw new Error("Incomplete native compiler dependency graph for " + command.source);}
  return dependencies;
}

/** @param {string} text @param {string} needle @param {string} replacement */
function replacePath(text, needle, replacement) {
  if (!needle) {return text;}
  const variants = new Set([needle, needle.replaceAll("\\", "/"), needle.replaceAll("/", "\\")]);
  let result = text;
  const attachedPathFlags = ["-isystem", "-iquote", "-idirafter", "-imsvc", "-include", "-MF", "-MT", "-Fo", "-I", "/I", "/Fo", "/FI", "-o"];
  for (const variant of variants) {
    if (!variant) {continue;}
    let cursor = 0;
    while (true) {
      const index = result.indexOf(variant, cursor);
      if (index < 0) {break;}
      const before = result.slice(0, index), after = index + variant.length;
      const startsPath = index === 0 || /[=,:;'"]$/.test(before) || attachedPathFlags.some(flag => before.endsWith(flag));
      const endsPath = after === result.length || /[\\/,:;'"]/.test(result[after] || "");
      if (startsPath && endsPath) {
        result = result.slice(0, index) + replacement + result.slice(after);
        cursor = index + replacement.length;
      } else {cursor = after;}
    }
  }
  return result;
}

/** Replace compiler depfile tokens while preserving either Make escapes or NMake quotes.
 * @param {string} text @param {string} from @param {string} to @param {"make" | "nmake"} [format] */
function rewriteDependencyPath(text, from, to, format = "make") {
  /** @param {string} value */
  const make = value => value.replaceAll("\\", "\\\\").replaceAll(" ", "\\ ").replaceAll("#", "\\#").replaceAll(":", "\\:");
  const forms = /** @type {Array<[string, string]>} */ (format === "nmake" ? [['"' + from + '"', '"' + to + '"'], [from, to]] : [[make(from), make(to)], [from, to]]);
  let result = text;
  for (const [oldValue, newValue] of forms) {result = result.split(oldValue).join(newValue);}
  return result;
}

/** Publish one fresh candidate file without following or replacing a symlink.
 * @param {string} file @param {string | Buffer} content */
function writeFreshFile(file, content) {
  ensureDirectory(path.dirname(file), "Fresh native output path");
  if (fs.existsSync(file)) {
    const stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Fresh native output is not a regular file: " + file);}
    fs.unlinkSync(file);
  }
  const temp = file + ".tmp-" + process.pid + "-" + crypto.randomBytes(5).toString("hex");
  fs.writeFileSync(temp, content, {flag: "wx", mode: 0o600}); fs.renameSync(temp, file);
}

/** @param {string} file @param {string} output @param {string} repo */
function logicalPath(file, output, repo) {
  const resolved = path.resolve(file);
  if (contained(output, resolved)) {return "output/" + path.relative(output, resolved).split(path.sep).join("/");}
  if (contained(repo, resolved)) {return "repo/" + path.relative(repo, resolved).split(path.sep).join("/");}
  return "external/" + fs.realpathSync(resolved);
}

/** A same-namespace build lock protects slot replacement and generated Ninja state.
 * @param {string} root */
function acquireLock(root) {
  const lock = path.join(root, ".lock");
  for (let tries = 0; tries < 2; tries++) {
    try {
      fs.mkdirSync(lock, {mode: 0o700});
      atomicJson(path.join(lock, "owner.json"), {pid: process.pid, started: Date.now()});
      return () => {fs.unlinkSync(path.join(lock, "owner.json")); fs.rmdirSync(lock);};
    } catch (error) {
      if (!(error instanceof Error) || !("code" in error) || error.code !== "EEXIST") {throw error;}
      const stat = fs.lstatSync(lock);
      if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("Native object cache lock is unsafe", {cause: error});}
      let owner;
      try {owner = readJson(path.join(lock, "owner.json"), 4096);} catch {
        if (Date.now() - stat.mtimeMs < 60000) {throw new Error("Native object cache is locked by an unresolved build");}
      }
      if (owner && Number.isInteger(owner.pid)) {
        let active = false;
        try {process.kill(owner.pid, 0); active = true;} catch (probe) {
          if (!(probe instanceof Error) || (/** @type {NodeJS.ErrnoException} */ (probe)).code !== "ESRCH") {throw probe;}
        }
        if (active) {throw new Error("Native object cache is already in use by process " + owner.pid, {cause: error});}
      }
      // Only a stale helper-owned lock's exact owner file and directory are removed.
      if (fs.existsSync(path.join(lock, "owner.json"))) {fs.unlinkSync(path.join(lock, "owner.json"));}
      fs.rmdirSync(lock);
    }
  }
  throw new Error("Unable to acquire native object cache lock");
}

/** @param {string} arg */
function posixQuote(arg) {return "'" + arg.replaceAll("'", "'\\''") + "'";}
/** Windows cmd expands percent even in quotes; reject shell metacharacters we cannot safely preserve. @param {string} arg */
function windowsQuote(arg) {
  if (/[\r\n\0%&|<>^!]/.test(arg)) {throw new Error("Ninja command path cannot be safely quoted on Windows");}
  return '"' + arg.replaceAll('"', '\\"') + '"';
}

/** @param {string} value */
function ninjaPath(value) {
  if (/[\0\r\n]/.test(value)) {throw new Error("Invalid Ninja graph path");}
  return value.replaceAll("$", () => "$$").replaceAll(" ", "$ ").replaceAll(":", "$:");
}

/** Escape a shell-quoted command fragment for Ninja's variable syntax.
 * @param {string} value */
function ninjaValue(value) {return value.replaceAll("$", () => "$$");}

/**
 * Compile a batch of independent native objects, restoring each cached object into the fresh
 * output directory. Existing links, device-link, PE/ELF isolation, receipts and hash admission
 * remain the caller's responsibility.
 *
 * @param {NativeObjectOptions} options
 * @returns {{objects: Array<{object: string, depfile: string, cacheHit: boolean, status: number | null,
 *   stdout: string, stderr: string, error?: string, signal: NodeJS.Signals | null,
 *   objectSha256: string | null, depfileSha256: string | null}>, cacheRoot: string, cacheHits: number,
 *   cacheMisses: number, ninja: string | null}}
 */
function compileNativeObjects(options) {
  const repo = path.resolve(options.repo), output = path.resolve(options.output);
  if (!path.isAbsolute(options.repo) || !path.isAbsolute(options.output) || fs.realpathSync(repo) !== repo ||
      fs.realpathSync(output) !== output || !fs.lstatSync(repo).isDirectory() || !fs.lstatSync(output).isDirectory()) {
    throw new Error("Native object cache requires real absolute repo and fresh output directories");
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$/.test(options.namespace) || options.namespace.split("/").some(part => !part || part === "." || part === "..")) {
    throw new Error("Invalid native object cache namespace");
  }
  const {commands} = options;
  if (!Array.isArray(commands) || !commands.length || commands.length > MAX_COMMANDS) {throw new Error("Invalid native object compile batch");}
  if (options.includeRoots !== undefined && (!Array.isArray(options.includeRoots) || options.includeRoots.length > 4096)) {throw new Error("Invalid native include-root list");}
  if (options.keyFiles !== undefined && (!Array.isArray(options.keyFiles) || options.keyFiles.length > MAX_FILES)) {throw new Error("Invalid native key-file list");}
  if (options.jobs !== undefined && (!Number.isSafeInteger(options.jobs) || options.jobs < 1 || options.jobs > 256)) {throw new Error("Invalid native Ninja job count");}
  if ((options.finalize === true) !== Array.isArray(options.liveObjects) || (options.liveObjects && options.liveObjects.length > MAX_COMMANDS) ||
      (options.liveObjects || []).some(file => typeof file !== "string" || !path.isAbsolute(file) || !contained(output, path.resolve(file)))) {
    throw new Error("Finalizing native cache requires the complete live object list inside the fresh output");
  }
  // Candidate/container paths identify the producer invocation, not the compile. Strip them
  // from the private compiler environment as well as the key so they cannot affect outputs.
  const suppliedEnv = options.environment || process.env;
  const forceRebuild = suppliedEnv["MOM_FORCE_REBUILD"];
  if (forceRebuild !== undefined && forceRebuild !== "" && forceRebuild !== "0" && forceRebuild !== "1") {
    throw new Error("MOM_FORCE_REBUILD must be 0 or 1");
  }
  const env = /** @type {NativeEnvironment} */ (Object.fromEntries(Object.entries(suppliedEnv)
    .filter(([name, value]) => typeof value === "string" && !orchestrationEnvironment.has(name))));
  for (const [name, value] of Object.entries(env)) {
    if (typeof value !== "string" || /[\0]/.test(name + value)) {throw new Error("Invalid native build environment");}
  }
  const envHash = environmentHash(env);
  const keyFileHashes = captureKeyFiles(options.keyFiles || []);
  const keyFilePaths = new Set(keyFileHashes.map(item => item.real));
  const helperHash = fileHash(__filename), parserHash = fileHash(require.resolve("./build-source-inputs"));
  const nativeCacheParent = path.join(repo, "build", "cache", "native-objects", "v1");
  ensureDirectory(nativeCacheParent, "Native object cache path");
  const cacheRoot = ensureDirectory(path.join(nativeCacheParent, ...options.namespace.split("/")), "Native object cache namespace");
  if (contained(cacheRoot, output) || contained(output, cacheRoot)) {throw new Error("Native object cache and fresh output must be separate");}
  const release = acquireLock(cacheRoot);
  /** @type {string | null} */
  let scratch = null;
  try {
    // Remove only interrupted, private per-invocation Ninja scratch directories.
    for (const name of fs.readdirSync(cacheRoot).filter(name => name.startsWith(".run-"))) {
      const stale = path.join(cacheRoot, name), stat = fs.lstatSync(stale);
      if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("Unsafe native Ninja scratch entry");}
      fs.rmSync(stale, {recursive: true, force: false});
    }

    const declaredGenerated = new Set();
    for (const entry of commands) {
      if (!entry || typeof entry.source !== "string" || !path.isAbsolute(entry.source)) {throw new Error("Native command needs an absolute source");}
      if (contained(output, path.resolve(entry.source))) {declaredGenerated.add(path.resolve(entry.source));}
      if (entry.generatedInputs !== undefined && (!Array.isArray(entry.generatedInputs) || entry.generatedInputs.length > MAX_FILES)) {throw new Error("Invalid generated native input list");}
      for (const input of entry.generatedInputs || []) {
        if (typeof input !== "string" || !path.isAbsolute(input) || !contained(output, path.resolve(input)) || !generatedSuffix.test(input)) {
          throw new Error("Generated native inputs must be declared source-like files inside the fresh output");
        }
        declaredGenerated.add(path.resolve(input));
      }
    }
    const generatedEntries = generatedFiles(output);
    const generated = new Set(generatedEntries.map(item => item.file));
    if ([...declaredGenerated].some(file => !generated.has(file))) {throw new Error("Declared generated native input is missing or not source-like");}
    const generatedNames = generatedEntries.map(({relative}) => relative);
    if (new Set(generatedNames).size !== generatedNames.length) {throw new Error("Duplicate generated native input");}
    const membership = includeMembership(options.includeRoots || [], output, generatedEntries.map(item => item.file), repo);

    // Ninja is preferred but optional; a missing Ninja uses the same tokenized runner directly.
    /** @type {ToolIdentity | null} */
    let ninjaTool = null;
    if (options.ninja !== undefined || env["MOM_NATIVE_NINJA"]) {ninjaTool = resolveExecutable(options.ninja || env["MOM_NATIVE_NINJA"] || "", env);} else {
      try {ninjaTool = resolveExecutable("ninja", env);} catch (error) {
        if (!(error instanceof Error) || !error.message.startsWith("Cannot resolve native tool:")) {throw error;}
      }
    }
    const selectedNinja = ninjaTool?.real || null;
    const cacheDisabled = env["MOM_NATIVE_CACHE"] === "0" || forceRebuild === "1";
    const generatedRoot = path.join(cacheRoot, "generated");
    if (!cacheDisabled) {
      resetGeneratedTree(generatedRoot);
      for (const item of generatedEntries) {
        const staged = path.join(generatedRoot, item.relative);
        ensureDirectory(path.dirname(staged), "Generated native staging path");
        const temp = staged + ".tmp-" + crypto.randomBytes(5).toString("hex"); fs.copyFileSync(item.file, temp, fs.constants.COPYFILE_EXCL);
        if (fileHash(temp) !== item.sha256) {fs.unlinkSync(temp); throw new Error("Generated native input changed while staging: " + item.file);}
        fs.renameSync(temp, staged);
      }
    }
    /** @type {NormalizedCommand[]} */
    const normalized = [];
    const seenSlots = new Set();
    /** @type {Map<string, ToolIdentity>} */
    const compilerIdentities = new Map();
    for (const entry of commands) {
      if (typeof entry.command !== "string" || !Array.isArray(entry.args) || entry.args.length > MAX_ARGS ||
          entry.args.some(arg => typeof arg !== "string" || /[\0\r\n]/.test(arg) || Buffer.byteLength(arg) > (1 << 20)) ||
          !path.isAbsolute(entry.object) || !path.isAbsolute(entry.depfile) ||
          !contained(output, path.resolve(entry.object)) || !contained(output, path.resolve(entry.depfile)) ||
          path.resolve(entry.object) === path.resolve(entry.depfile) ||
          (entry.timeout !== undefined && (!Number.isSafeInteger(entry.timeout) || entry.timeout < 1 || entry.timeout > 3600000))) {
        throw new Error("Malformed native compile command");
      }
      const source = path.resolve(entry.source), object = path.resolve(entry.object), depfile = path.resolve(entry.depfile);
      if (seenSlots.has(object)) {throw new Error("Duplicate native object output: " + object);} seenSlots.add(object);
      const compiler = compilerIdentities.get(entry.command) || resolveExecutable(entry.command, env); compilerIdentities.set(entry.command, compiler);
      const id = digest(JSON.stringify([options.namespace, logicalPath(source, output, repo), path.relative(output, object).split(path.sep).join("/")])).slice(0, 40);
      const slotPath = path.join(cacheRoot, "slots", id), slot = cacheDisabled ? slotPath : ensureDirectory(slotPath, "Native object slot");
      const ext = path.extname(object) || ".o", cachedObject = path.join(slot, "object" + ext), cachedDepfile = path.join(slot, "dependencies.d"), stateFile = path.join(slot, "state.json");
      const commandGenerated = generatedEntries.filter(value => path.resolve(value.file) === source ||
        (entry.generatedInputs || []).map(file => path.resolve(file)).includes(path.resolve(value.file)));
      const stableSource = !cacheDisabled && contained(output, source) ? path.join(generatedRoot, path.relative(output, source)) : source;
      const compileObject = cacheDisabled ? object : cachedObject, compileDepfile = cacheDisabled ? depfile : cachedDepfile;
      const args = cacheDisabled ? entry.args.slice() : entry.args.map(arg => {
        let value = replacePath(arg, entry.object, compileObject);
        value = replacePath(value, entry.depfile, compileDepfile);
        value = replacePath(value, entry.source, stableSource);
        value = replacePath(value, output, generatedRoot);
        return value;
      });
      const sourceSha256 = fileHash(regularFile(stableSource, "Native source"));
      const generatedHashes = commandGenerated.map(item => /** @type {[string, string]} */ ([item.relative, item.sha256]));
      const base = {schema: 2, namespace: options.namespace, source: logicalPath(source, output, repo), sourceSha256,
        object: path.relative(output, object).split(path.sep).join("/"), compiler: {selected: compiler.selected, real: compiler.real, sha256: compiler.sha256},
        args, environment: envHash, includeMembership: membership, generatedMembership: generatedNames, generatedHashes, keyFileHashes, helperHash, parserHash,
        ninja: selectedNinja ? {path: selectedNinja, sha256: ninjaTool?.sha256 || ""} : null};
      const signature = digest(JSON.stringify(base));
      let valid = false;
      /** @type {Array<{path: string, sha256: string}>} */
      let deps = [];
      if (!cacheDisabled) {
        try {
          const receipt = readJson(stateFile);
          // A different compile recipe cannot reuse this object; avoid reading its old
          // dependency closure. Newly compiled inputs are still verified below.
          if (receipt.version === 2 && receipt.signature === signature && Array.isArray(receipt.dependencies) && receipt.dependencies.length <= MAX_FILES) {
            deps = receipt.dependencies;
            const dependenciesCurrent = deps.every(item => {
              if (!item || typeof item.path !== "string" || typeof item.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(item.sha256)) {return false;}
              const dependency = item.path.startsWith("output/") ? path.resolve(output, item.path.slice("output/".length)) : item.path;
              if (!path.isAbsolute(dependency) || (item.path.startsWith("output/") && !contained(output, dependency))) {return false;}
              return fileHash(regularFile(dependency, "Native dependency")) === item.sha256;
            });
            if (!dependenciesCurrent) {deps = [];}
            valid = dependenciesCurrent &&
              receipt.objectSha256 === fileHash(regularFile(cachedObject, "Cached native object")) &&
              receipt.depfileSha256 === fileHash(regularFile(cachedDepfile, "Cached native depfile"));
          }
        } catch (error) {
          if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {valid = false;}
        }
      }
      if (!valid && !cacheDisabled) {clearArtifacts(slot);}
      normalized.push({entry, source, object, depfile, compiler, args, stableSource, cachedObject: compileObject, cachedDepfile: compileDepfile,
        stateFile, signature, base, slot, id, objectKey: path.relative(output, object).split(path.sep).join("/"),
        cacheHit: valid, dependencies: deps, startedAt: Date.now()});
    }

    const misses = normalized.filter(item => !item.cacheHit);
    const useNinja = !cacheDisabled && misses.length > 0 && selectedNinja;
    if (useNinja) {
      const buildScratch = fs.mkdtempSync(path.join(cacheRoot, ".run-")); scratch = buildScratch; fs.chmodSync(buildScratch, 0o700);
      const manifestFile = path.join(buildScratch, "jobs.json"), graphFile = path.join(buildScratch, "build.ninja");
      const manifest = misses.map((item, index) => ({command: item.compiler.selected, args: item.args, cwd: repo, source: item.stableSource,
        object: item.cachedObject, depfile: item.cachedDepfile, timeout: item.entry.timeout || 3600000, result: path.join(buildScratch, "result-" + index + ".json")}));
      const manifestText = JSON.stringify(manifest);
      if (Buffer.byteLength(manifestText) > MAX_JSON) {throw new Error("Native Ninja manifest exceeds its bound");}
      fs.writeFileSync(manifestFile, manifestText, {mode: 0o600, flag: "wx"});
      const node = process.execPath, script = __filename;
      const quote = process.platform === "win32" ? windowsQuote : posixQuote;
      const commandLine = [node, script, "--edge", manifestFile].map(arg => ninjaValue(quote(arg))).join(" ") + " $edge";
      let graph = "ninja_required_version = 1.8\nrule native_compile\n  command = " + commandLine + "\n  description = Native object $edge\n  restat = 1\n";
      for (const [index, item] of misses.entries()) {
        graph += "build " + ninjaPath(item.cachedObject) + ": native_compile " + ninjaPath(item.stableSource) + "\n  edge = " + index + "\n";
      }
      fs.writeFileSync(graphFile, graph, {mode: 0o600, flag: "wx"});
      const jobs = options.jobs ?? 1;
      if (!Number.isSafeInteger(jobs) || jobs < 1 || jobs > 256) {throw new Error("Invalid native Ninja job count");}
      const args = ["-f", graphFile, "-v", "-j", String(jobs), ...misses.map(item => item.cachedObject)];
      const result = spawnSync(selectedNinja, args, {cwd: buildScratch, env, encoding: "utf8", timeout: 3600000, maxBuffer: 32 << 20});
      for (const [index, item] of misses.entries()) {
        const resultFile = path.join(buildScratch, "result-" + index + ".json");
        try {
          const edge = readJson(resultFile, MAX_JSON);
          item.status = edge.status; item.stdout = edge.stdout || ""; item.stderr = edge.stderr || "";
          item.error = edge.error; item.signal = edge.signal || null;
        } catch (error) {
          if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {throw error;}
          item.status = null; item.stdout = ""; item.stderr = result.stderr || "";
          item.error = result.error ? result.error.message : "Ninja did not run this native compile edge"; item.signal = result.signal || null;
        }
      }
      if (result.error || result.signal || result.status !== 0) {
        const message = result.error ? result.error.message : result.signal ? "Ninja was interrupted by " + result.signal : "Ninja failed with status " + result.status;
        for (const item of misses) {item.status = null; item.error = message; item.signal = result.signal || null;}
      }
    }

    /** @type {Array<{object: string, depfile: string, cacheHit: boolean, status: number | null, stdout: string, stderr: string,
     * error?: string, signal: NodeJS.Signals | null, objectSha256: string | null, depfileSha256: string | null}>} */
    const results = [];
    let cacheHits = 0, cacheMisses = 0;
    for (const item of normalized) {
      if (item.cacheHit) {cacheHits++;} else {
        cacheMisses++;
        if (!item.cacheHit && !useNinja) {
          const result = spawnSync(item.compiler.selected, item.args, {cwd: repo, env, encoding: "utf8", timeout: item.entry.timeout || 3600000, maxBuffer: 64 << 20});
          item.status = result.status; item.stdout = result.stdout || ""; item.stderr = result.stderr || "";
          if (result.error) {item.error = result.error.message;} else {delete item.error;}
          item.signal = result.signal;
        }
      }
      if (item.cacheHit || item.status === 0) {
        const currentCompiler = resolveExecutable(item.entry.command, env);
        if (currentCompiler.real !== item.compiler.real || currentCompiler.sha256 !== item.compiler.sha256) {
          throw new Error("Native compiler changed during object production: " + item.entry.command);
        }
        /** @type {string} */
        let admittedDepText;
        if (!cacheDisabled) {
          const objectBytes = fs.readFileSync(item.cachedObject);
          writeFreshFile(item.object, objectBytes);
          let freshDepfile = fs.readFileSync(item.cachedDepfile, "utf8");
          for (const generatedInput of generatedEntries) {
            freshDepfile = rewriteDependencyPath(freshDepfile, path.join(generatedRoot, generatedInput.relative), generatedInput.file,
              item.entry.dependencyFormat || "make");
          }
          freshDepfile = rewriteDependencyPath(freshDepfile, item.cachedObject, item.object, item.entry.dependencyFormat || "make");
          writeFreshFile(item.depfile, freshDepfile);
          admittedDepText = freshDepfile;
        } else {
          admittedDepText = fs.readFileSync(item.depfile, "utf8");
        }
        const objectSha256 = fileHash(regularFile(item.cachedObject, "Compiled native object"));
        const stableDepfileSha256 = fileHash(regularFile(item.cachedDepfile, "Compiled native depfile"));
        const depfileSha256 = fileHash(regularFile(item.depfile, "Fresh native depfile"));
        const rawDeps = (options.resolveDependencies || ((info) => defaultDependencies(info.command, info.depfileText, info.repo)))({
          command: item.entry, depfileText: admittedDepText, repo,
        });
        if (!Array.isArray(rawDeps) || !rawDeps.length || rawDeps.length > MAX_FILES) {throw new Error("Invalid resolved native dependencies for " + item.entry.source);}
        /** @type {Array<{path: string, sha256: string}>} */
        const dependencies = [];
        const previousDependencies = new Map(item.dependencies.map(previous => [previous.path, previous.sha256]));
        for (const dep of [...new Set(rawDeps)]) {
          if (typeof dep !== "string" || !path.isAbsolute(dep)) {throw new Error("Resolved native dependencies must be absolute paths");}
          let real = regularFile(dep, "Native dependency");
          if (contained(generatedRoot, real)) {real = path.join(output, path.relative(generatedRoot, real));}
          if (real === item.source || real === item.stableSource || real === item.object || real === item.cachedObject || real === item.depfile || real === item.cachedDepfile) {continue;}
          if (contained(output, real)) {
            const rel = path.relative(output, real).split(path.sep).join("/");
            if (!generated.has(real)) {throw new Error("Undeclared generated native dependency: " + real);}
            const hash = fileHash(regularFile(real, "Generated native dependency"));
            const previous = previousDependencies.get("output/" + rel);
            if (previous !== undefined && previous !== hash) {throw new Error("Native dependency changed during compilation: " + real);}
            dependencies.push({path: "output/" + rel, sha256: hash});
            continue;
          }
          if (contained(cacheRoot, real)) {
            throw new Error("Undeclared cache-local native dependency: " + real);
          }
          if (keyFilePaths.has(real)) {continue;}
          const hash = fileHash(real);
          if (previousDependencies.has(real) && previousDependencies.get(real) !== hash) {
            throw new Error("Native dependency changed during compilation: " + real);
          }
          if (fs.statSync(real).ctimeMs > item.startedAt) {throw new Error("Native dependency changed during compilation: " + real);}
          dependencies.push({path: real, sha256: hash});
        }
        if (fileHash(regularFile(item.source, "Native source")) !== item.base.sourceSha256) {throw new Error("Native source changed during compilation: " + item.entry.source);}
        if (!cacheDisabled) {
          atomicJson(item.stateFile, {version: 2, signature: item.signature, objectKey: item.objectKey,
            objectSha256, depfileSha256: stableDepfileSha256, dependencies});
          updateSlotIndex(cacheRoot, item.objectKey, item.id);
        }
        item.objectSha256 = objectSha256; item.depfileSha256 = depfileSha256;
      }
      results.push({object: item.object, depfile: item.depfile, cacheHit: item.cacheHit, status: item.status ?? (item.cacheHit ? 0 : null),
        stdout: item.stdout || "", stderr: item.stderr || "", ...(item.error ? {error: item.error} : {}), signal: item.signal || null,
        objectSha256: item.objectSha256 || null, depfileSha256: item.depfileSha256 || null});
    }
    const finalGeneratedEntries = generatedFiles(output);
    if (JSON.stringify(finalGeneratedEntries.map(({relative, sha256}) => [relative, sha256])) !==
        JSON.stringify(generatedEntries.map(({relative, sha256}) => [relative, sha256])) ||
        includeMembership(options.includeRoots || [], output, finalGeneratedEntries.map(item => item.file), repo) !== membership ||
        JSON.stringify(captureKeyFiles(options.keyFiles || [])) !== JSON.stringify(keyFileHashes)) {
      throw new Error("Native include membership, generated inputs, or key files changed during compilation");
    }
    if (!cacheDisabled && options.finalize && results.every(item => item.status === 0)) {finalizeSlots(cacheRoot, output, options.liveObjects || []);}
    if (selectedNinja) {
      const currentNinja = resolveExecutable(options.ninja || env["MOM_NATIVE_NINJA"] || "ninja", env);
      if (currentNinja.real !== selectedNinja || currentNinja.sha256 !== ninjaTool?.sha256) {throw new Error("Ninja changed during native object production");}
    }
    if (scratch) {removeScratch(scratch); scratch = null;}
    return {objects: results, cacheRoot, cacheHits, cacheMisses, ninja: selectedNinja};
  } finally {
    try {
      if (scratch && fs.existsSync(scratch)) {
        const stat = fs.lstatSync(scratch);
        if (stat.isDirectory() && !stat.isSymbolicLink()) {fs.rmSync(scratch, {recursive: true, force: false});}
      }
    } finally {release();}
  }
}

// Ninja invokes this helper with a manifest, never with compiler arguments embedded in shell
// syntax. The manifest lives in a private per-call directory and is deleted by its owner.
if (require.main === module && process.argv[2] === "--edge") {
  try {
    const [manifestFile, edgeText] = process.argv.slice(3), index = Number(edgeText);
    if (!manifestFile) {throw new Error("Missing native Ninja edge manifest");}
    const manifest = readJson(manifestFile, MAX_JSON), edge = manifest[index];
    if (!edge || !Number.isInteger(index) || !Array.isArray(edge.args) || typeof edge.command !== "string") {throw new Error("Invalid native Ninja edge");}
    const result = spawnSync(edge.command, edge.args, {cwd: edge.cwd, encoding: "utf8", timeout: edge.timeout, maxBuffer: 64 << 20});
    /** @param {string | null} value */
    const bound = value => Buffer.from(value || "").subarray(0, 1 << 20).toString("utf8");
    const record = {status: result.status, stdout: bound(result.stdout), stderr: bound(result.stderr),
      error: result.error ? result.error.message : undefined, signal: result.signal || null};
    atomicJson(edge.result, record);
    if (result.error) {process.stderr.write(result.error.message + "\n");}
    process.stdout.write(result.stdout || ""); process.stderr.write(result.stderr || "");
    process.exitCode = result.status === 0 ? 0 : 1;
  } catch (error) {process.stderr.write((error instanceof Error ? error.message : "Native compile edge failed") + "\n"); process.exitCode = 1;}
}

module.exports = {compileNativeObjects, environmentValue};
