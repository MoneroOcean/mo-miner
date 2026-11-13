"use strict";

// Produce only the pinned Linux static zstd library; the CLI and upstream build tests are out of scope.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"), {createHash} = require("node:crypto");
const {fileHash, sourceInputs, verifySourceInputs, readBuildJson} = require("./build-source-inputs");
const {nativeSdkEnvironment} = require("./build-environment");
const {isStaticArchive} = require("./native-audit");
const repo = path.resolve(__dirname, "..");
const zstdPin = Object.freeze({version: "1.5.7", sha256: "eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3"});
const sourceBase = "zstd-" + zstdPin.version;
const sourceLeaves = ["scripts/build-native-zstd.js", "scripts/build-source-inputs.js", "scripts/build-environment.js", "scripts/native-audit.js"];
const maxArchiveBytes = 16 << 20, maxExpandedBytes = 64 << 20, maxArchiveMembers = 2048,
  maxSourceBytes = 16 << 20, maxSourceFiles = 512, maxLibraryBytes = 16 << 20;
const isHash = (/** @type {unknown} */ value) => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
/** @typedef {{ar: string, cc: string, make: string, mkdir: string, tar: string}} ZstdTools */
/** @typedef {Record<keyof ZstdTools, {path: string, sha256: string}>} ZstdToolReceipt */
/** @typedef {{path: string, sha256: string, bytes: number}} ZstdOutput */

/** Makefile recipes expand these paths unquoted, so this producer deliberately rejects whitespace.
 * @param {unknown} value @returns {value is string} */
function absolute(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 4096 && path.isAbsolute(value) &&
    !/[\s;"'$%&|<>^!`\\\0]/.test(value) && !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
}

const toolNames = Object.freeze(["ar", "cc", "make", "mkdir", "tar"]);
/** @param {unknown} value @returns {ZstdTools} */
function validateTools(value) {
  const tools = /** @type {ZstdTools} */ (value);
  if (!tools || typeof tools !== "object" || Array.isArray(tools) || Object.keys(tools).sort().join(",") !== toolNames.join(",") ||
      Object.values(tools).some(file => !absolute(file))) {throw new Error("Static zstd SDK needs explicit absolute Linux tools");}
  return tools;
}

/** Pure single-library plan; CPPFLAGS selects the upstream multithreaded static target's source mode.
 * @param {string} output @param {string} archive @param {unknown} tools @param {number} jobs @param {string} [platform] */
function zstdSdkPlan(output, archive, tools, jobs, platform = process.platform) {
  if (platform !== "linux" || !absolute(output) || !absolute(archive) || !Number.isInteger(jobs) || jobs < 1 || jobs > 16) {
    throw new Error("Static zstd SDK needs Linux, absolute paths and jobs in 1..16");
  }
  const selected = validateTools(tools), source = path.join(output, "source"), build = path.join(output, "build");
  const buildArchive = path.join(build, "static", "libzstd.a"), installArchive = path.join(output, "lib", "libzstd.a");
  const extractCommand = [selected["tar"], "-xzf", archive, "-C", source, "--strip-components=1", "--no-same-owner", "--no-same-permissions", "--",
    sourceBase + "/lib", sourceBase + "/LICENSE", sourceBase + "/COPYING"];
  const buildCommand = [selected["make"], "-C", path.join(source, "lib"), "-j", String(jobs), "BUILD_DIR=" + build,
    "CC=" + selected["cc"], "AR=" + selected["ar"], "MKDIR=" + selected["mkdir"] + " -p", "CPPFLAGS=-DZSTD_MULTITHREAD",
    "CFLAGS=-O3 -DNDEBUG -fPIC", buildArchive];
  return {version: 1, platform, jobs, source, build, buildArchive, installArchive, extractCommand, buildCommand,
    target: "x86_64-linux-gnu", compiler: selected["cc"], staticMode: "ZSTD_MULTITHREAD"};
}

/** Tools/archive are opened only when they are canonical regular files; executables must run directly.
 * @param {string} file @param {string} label @param {boolean} [executable] */
function regularInput(file, label, executable = false) {
  if (!absolute(file)) {throw new Error("Static zstd " + label + " must be absolute");}
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file) {throw new Error("Static zstd " + label + " must be a canonical regular file");}
  if (executable) {fs.accessSync(file, fs.constants.X_OK);}
  return stat;
}

/** Re-open each exact tool/archive input instead of trusting hashes copied into a receipt. @param {unknown} recorded @returns {ZstdToolReceipt} */
function verifyZstdTools(recorded) {
  const tools = /** @type {ZstdToolReceipt} */ (recorded);
  if (!tools || typeof tools !== "object" || Array.isArray(tools) || Object.keys(tools).sort().join(",") !== toolNames.join(",") ||
      Object.values(tools).some(row => !row || !absolute(row.path) || !isHash(row.sha256))) {throw new Error("Static zstd SDK tool receipt is incomplete");}
  for (const [name, row] of Object.entries(tools)) {
    regularInput(row.path, "tool " + name, true);
    if (fileHash(row.path) !== row.sha256) {throw new Error("Static zstd tool changed: " + name);}
  }
  return tools;
}

/** The original archive remains an explicit admission input, not just a self-declared receipt field.
 * @param {{path: string, sha256: string, bytes: number}} archive */
function verifyZstdArchiveInput(archive) {
  if (!archive || archive.sha256 !== zstdPin.sha256 || !Number.isSafeInteger(archive.bytes) || archive.bytes < 1 || archive.bytes > maxArchiveBytes) {
    throw new Error("Invalid pinned original zstd archive identity");
  }
  const stat = regularInput(archive.path, "original source archive");
  if (stat.size !== archive.bytes || fileHash(archive.path) !== zstdPin.sha256) {throw new Error("Original pinned zstd archive changed");}
  return stat;
}

/** @param {unknown} recipe */
function verifyZstdRecipe(recipe) {
  verifySourceInputs(recipe, sourceInputs(repo, [], sourceLeaves), "Static zstd SDK producer");
}

/** Validate the complete original tar listing before selecting only lib/ and the two upstream notices.
 * The two known CLI-test links are outside the extracted cohort and are never followed.
 * @param {string} listing @param {string} verbose */
function validateArchive(listing, verbose) {
  const rows = (/** @type {string} */ value) => {const lines = value.split(/\r?\n/); if (lines.at(-1) === "") {lines.pop();} return lines;};
  const names = rows(listing), types = rows(verbose), seen = new Set(), links = new Set(), kinds = new Map();
  let extractedFiles = 0, extractedBytes = 0, expandedBytes = 0;
  if (!names.length || names.length > maxArchiveMembers || names.length !== types.length) {throw new Error("Invalid zstd source archive listing");}
  for (const [index, name] of names.entries()) {
    const typeLine = types[index] || "", normalized = name.endsWith("/") ? name.slice(0, -1) : name, parts = normalized.split("/"), kind = typeLine[0] || "";
    const safeParts = parts.every((/** @type {string} */ part) => /^[A-Za-z0-9_.+-]+$/.test(part) && part !== "." && part !== "..");
    const suffix = kind === "l" ? name + " -> zstd" : name;
    const metadata = typeLine.endsWith(suffix) ? typeLine.slice(0, -suffix.length).trim().split(/\s+/) : [];
    const size = Number(metadata[2]);
    if (!name || name.length > 4096 || /[\\\0\r\n]/.test(name) || parts[0] !== sourceBase || !safeParts || seen.has(normalized) ||
        !["-", "d", "l"].includes(kind) || (parts.length === 1 && kind !== "d") ||
        metadata.length < 5 || !Number.isSafeInteger(size) || size < 0 || (kind !== "l" && !typeLine.endsWith(name)) ||
        (kind === "l" && !typeLine.endsWith(suffix))) {
      throw new Error("Unsafe or unexpected zstd source archive member");
    }
    seen.add(normalized); kinds.set(normalized, kind);
    if (kind === "-" && (expandedBytes += size) > maxExpandedBytes) {throw new Error("Original zstd source archive exceeds its expansion bound");}
    if (normalized === sourceBase + "/LICENSE" || normalized === sourceBase + "/COPYING" || normalized === sourceBase + "/lib" ||
        normalized.startsWith(sourceBase + "/lib/")) {
      if (kind === "-") {if (++extractedFiles > maxSourceFiles || (extractedBytes += size) > maxSourceBytes) {throw new Error("Selected zstd source exceeds its extraction bounds");}}
    }
    if (kind === "l") {
      if (![sourceBase + "/tests/cli-tests/bin/unzstd", sourceBase + "/tests/cli-tests/bin/zstdcat"].includes(normalized)) {
        throw new Error("Unexpected zstd source archive link");
      }
      links.add(normalized);
    }
  }
  const required = [sourceBase, sourceBase + "/lib", sourceBase + "/lib/Makefile", sourceBase + "/lib/libzstd.mk",
    sourceBase + "/lib/zstd.h", sourceBase + "/lib/zstd_errors.h", sourceBase + "/lib/zdict.h", sourceBase + "/LICENSE", sourceBase + "/COPYING"];
  if (required.some(name => !seen.has(name)) || links.size !== 2 || kinds.get(sourceBase) !== "d" || kinds.get(sourceBase + "/lib") !== "d" ||
      ["LICENSE", "COPYING", "lib/Makefile", "lib/libzstd.mk", "lib/zstd.h", "lib/zstd_errors.h", "lib/zdict.h"].some(name => kinds.get(sourceBase + "/" + name) !== "-")) {
    throw new Error("Pinned zstd archive lacks its static library sources or notices");
  }
  return names.length;
}

/** Hash the narrow extracted source cohort and reject links, special files, and oversized trees.
 * @param {string} root */
function sourceInventory(root) {
  const files = Object.create(null), digest = createHash("sha256"); let count = 0, entries = 0, bytes = 0;
  const top = fs.readdirSync(root).sort();
  if (JSON.stringify(top) !== JSON.stringify(["COPYING", "LICENSE", "lib"])) {throw new Error("Static zstd source has unexpected top-level members");}
  /** @param {string} directory */
  function visit(directory) {
    const stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("Static zstd source contains a linked directory");}
    for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(directory, entry.name), relative = path.relative(root, file).split(path.sep).join("/");
      if (!relative || relative.length > 4096 || /[\\\0\r\n]/.test(relative) || Array.from(relative).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) {
        throw new Error("Static zstd source has an invalid path");
      }
      if (entry.isDirectory()) {
        if (++entries > maxSourceFiles * 2) {throw new Error("Static zstd source has too many entries");}
        digest.update(relative + "\0<directory>\n"); visit(file); continue;
      }
      const child = fs.lstatSync(file);
      if (!entry.isFile() || child.isSymbolicLink() || !child.isFile() || ++count > maxSourceFiles || ++entries > maxSourceFiles * 2 || (bytes += child.size) > maxSourceBytes) {
        throw new Error("Static zstd source is not a bounded regular-file tree");
      }
      const sha256 = fileHash(file); files[relative] = sha256; digest.update(relative + "\0" + sha256 + "\n");
    }
  }
  visit(root);
  if (!count) {throw new Error("Static zstd source tree is empty");}
  return {files, count, bytes, sha256: digest.digest("hex")};
}

/** Project only the two source notices already pinned by the SDK receipt.
 * @param {string} sourceRoot @param {Record<string, {path: string, sha256: string}>} recorded */
function nativeZstdNoticeInputs(sourceRoot, recorded) {
  if (!absolute(sourceRoot) || fs.realpathSync(sourceRoot) !== sourceRoot) {throw new Error("Select a canonical zstd source root for notices");}
  if (!recorded || typeof recorded !== "object" || Array.isArray(recorded)) {throw new Error("Invalid static zstd source notices");}
  return ["LICENSE", "COPYING"].map(name => {
    const notice = recorded[name], file = path.join(sourceRoot, name);
    if (notice?.path !== "source/" + name || !isHash(notice.sha256)) {throw new Error("Invalid static zstd source notice: " + name);}
    regularInput(file, name + " notice");
    if (fileHash(file) !== notice.sha256) {throw new Error("Static zstd source notice changed: " + name);}
    return {component: "zstd " + name, path: file, sha256: notice.sha256};
  });
}

/** Stage exactly the single public static archive. @param {string} root */
function stagedFiles(root) {
  const directory = path.join(root, "lib"), stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!stat?.isDirectory() || stat.isSymbolicLink() || JSON.stringify(fs.readdirSync(directory).sort()) !== JSON.stringify(["libzstd.a"])) {
    throw new Error("Static zstd SDK has unexpected staged members");
  }
  const archive = path.join(directory, "libzstd.a"), archiveStat = regularInput(archive, "installed archive");
  if (!isStaticArchive(archive) || archiveStat.size > maxLibraryBytes) {throw new Error("Static zstd output is not a bounded static archive");}
  return {"lib/libzstd.a": fileHash(archive)};
}

/** Bind the out-of-tree archive, staged archive, and receipt to one byte identity.
 * @param {string} root @param {ZstdOutput} buildOutput @param {ZstdOutput} install @param {Record<string, string>} expectedArtifacts */
function verifyZstdOutputs(root, buildOutput, install, expectedArtifacts) {
  if (buildOutput?.path !== "build/static/libzstd.a" || install?.path !== "lib/libzstd.a" ||
      !isHash(buildOutput.sha256) || !Number.isInteger(buildOutput.bytes) || buildOutput.bytes < 8 || buildOutput.bytes > maxLibraryBytes ||
      !isHash(install.sha256) || !Number.isInteger(install.bytes) || install.bytes < 8 || install.bytes > maxLibraryBytes ||
      buildOutput.sha256 !== install.sha256 || buildOutput.bytes !== install.bytes) {throw new Error("Static zstd build/install identity differs");}
  const built = path.join(root, buildOutput.path), builtStat = regularInput(built, "build archive");
  if (!isStaticArchive(built) || builtStat.size !== buildOutput.bytes || fileHash(built) !== buildOutput.sha256) {throw new Error("Static zstd build archive changed");}
  const artifacts = stagedFiles(root);
  if (!expectedArtifacts || typeof expectedArtifacts !== "object" || Array.isArray(expectedArtifacts) || Object.keys(expectedArtifacts).length !== 1 ||
      expectedArtifacts["lib/libzstd.a"] !== artifacts["lib/libzstd.a"] || install.sha256 !== artifacts["lib/libzstd.a"] ||
      install.bytes !== fs.statSync(path.join(root, "lib", "libzstd.a")).size) {throw new Error("Static zstd staged membership or bytes changed");}
  return artifacts;
}

/** @param {string[]} command @param {string} cwd @param {NodeJS.ProcessEnv} env @param {string} output @param {string} stage */
function runStage(command, cwd, env, output, stage) {
  const logPath = path.join(output, "logs", stage + ".log"), fd = fs.openSync(logPath, "wx");
  fs.writeFileSync(path.join(output, "logs", stage + ".json"), JSON.stringify(command) + "\n", {flag: "wx"});
  try {
    const result = spawnSync(command[0] || "", command.slice(1), {cwd, env, stdio: ["ignore", fd, fd], timeout: 15 * 60 * 1000});
    if (result.error || result.signal || result.status !== 0) {throw new Error("Static zstd stage failed: " + stage + "; inspect " + logPath, {cause: result.error});}
  } finally {fs.closeSync(fd);}
}

/** Build the pinned source archive with explicit tools and no host zstd, tests, CLI, or release claim.
 * @param {string} archive @param {string} output @param {{version: number, jobs: number, tools: Record<string,string>}} config
 * @param {NodeJS.ProcessEnv} [environment] @param {string} [configSha256] */
function buildNativeZstd(archive, output, config, environment = process.env, configSha256) {
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Static zstd production requires MOM_AOT_BUILD_ONLY=1");}
  if (!config || Object.keys(config).sort().join(",") !== "jobs,tools,version" || config.version !== 1 || !Number.isInteger(config.jobs) || config.jobs < 1 || config.jobs > 16) {
    throw new Error("Invalid static zstd version-1 tool descriptor");
  }
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Static zstd production supports Linux x64 only");}
  const tools = validateTools(config.tools), selectedOutput = path.resolve(output), parent = path.dirname(selectedOutput);
  const plan = zstdSdkPlan(selectedOutput, archive, tools, config.jobs, process.platform), parentStat = fs.lstatSync(parent, {throwIfNoEntry: false});
  const relative = path.relative(repo, selectedOutput);
  if (!absolute(archive) || !absolute(output) || output !== selectedOutput || !parentStat?.isDirectory() || parentStat.isSymbolicLink() ||
      fs.realpathSync(parent) !== parent || fs.lstatSync(selectedOutput, {throwIfNoEntry: false}) ||
      !relative || (!relative.startsWith(".." + path.sep) && !path.isAbsolute(relative))) {
    throw new Error("Static zstd output must be new, canonical and outside the repository");
  }
  const archiveStat = regularInput(archive, "source archive");
  if (archiveStat.size > maxArchiveBytes) {throw new Error("Static zstd source archive exceeds its size bound");}
  const toolHashes = Object.fromEntries(Object.entries(tools).map(([name, file]) => {
    regularInput(file, "tool " + name, true); return [name, {path: file, sha256: fileHash(file)}];
  }));
  const archiveSha256 = fileHash(archive);
  if (archiveSha256 !== zstdPin.sha256) {throw new Error("Static zstd archive SHA256 differs from the pinned original 1.5.7 release");}
  const configHash = configSha256 || createHash("sha256").update(JSON.stringify(config)).digest("hex");
  if (!isHash(configHash)) {throw new Error("Invalid static zstd tool descriptor digest");}
  const recipe = sourceInputs(repo, [], sourceLeaves), env = nativeSdkEnvironment(environment);
  for (const key of ["MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS", "MAKEOVERRIDES", "MAKELEVEL", "NMAKEFLAGS", "NMAKE", "AS", "ASFLAGS", "ARFLAGS", "CFLAGS", "CPPFLAGS", "LDFLAGS", "SHELL"]) {delete env[key];}
  const tar = tools["tar"], listArgs = ["-tzf", archive], verboseArgs = ["-tvzf", archive];
  const listing = spawnSync(tar, listArgs, {env, encoding: "utf8", timeout: 60000, maxBuffer: 8 << 20});
  const verbose = spawnSync(tar, verboseArgs, {env, encoding: "utf8", timeout: 60000, maxBuffer: 8 << 20});
  for (const result of [listing, verbose]) {
    if (result.error || result.signal || result.status !== 0 || result.stderr) {throw new Error("Static zstd source archive listing failed", {cause: result.error});}
  }
  const sourceEntries = validateArchive(listing.stdout || "", verbose.stdout || "");
  if (fileHash(archive) !== archiveSha256 || Object.values(toolHashes).some(row => fileHash(row.path) !== row.sha256)) {
    throw new Error("Static zstd archive or tools changed during preflight");
  }
  const versionResult = spawnSync(tools["cc"], ["--version"], {env, encoding: "utf8", timeout: 30000, maxBuffer: 16 << 10});
  const compilerText = (versionResult.stdout || "") + (versionResult.stderr || ""), compilerVersion = compilerText.trim().split(/\r?\n/, 1)[0] || "";
  if (versionResult.error || versionResult.signal || versionResult.status !== 0 || !/^Intel\(R\) oneAPI DPC\+\+\/C\+\+ Compiler 2026\.2(?:\.|\s|$)/.test(compilerVersion)) {
    throw new Error("Static zstd requires the selected Intel 2026.2 icx compiler", {cause: versionResult.error});
  }

  fs.mkdirSync(selectedOutput, {mode: 0o700}); fs.mkdirSync(path.join(selectedOutput, "logs")); fs.mkdirSync(plan.source); fs.mkdirSync(plan.build);
  fs.writeFileSync(path.join(selectedOutput, "logs", "compiler-version.log"), compilerText, {flag: "wx"});
  runStage(plan.extractCommand, selectedOutput, env, selectedOutput, "extract");
  const sourceBefore = sourceInventory(plan.source), buildDirStat = fs.lstatSync(plan.build);
  if (!buildDirStat.isDirectory() || buildDirStat.isSymbolicLink()) {throw new Error("Static zstd build directory is not a real directory");}
  runStage(plan.buildCommand, selectedOutput, env, selectedOutput, "build-static");
  const builtStat = regularInput(plan.buildArchive, "build archive");
  if (builtStat.size > maxLibraryBytes || !isStaticArchive(plan.buildArchive)) {throw new Error("Intel 2026.2 output is not a bounded static zstd archive");}
  fs.mkdirSync(path.dirname(plan.installArchive)); fs.copyFileSync(plan.buildArchive, plan.installArchive, fs.constants.COPYFILE_EXCL);
  if (!isStaticArchive(plan.installArchive) || fileHash(plan.installArchive) !== fileHash(plan.buildArchive)) {throw new Error("Staged static zstd archive changed during copy");}
  const sourceAfter = sourceInventory(plan.source);
  if (sourceBefore.sha256 !== sourceAfter.sha256 || sourceBefore.count !== sourceAfter.count || sourceBefore.bytes !== sourceAfter.bytes) {
    throw new Error("Pinned zstd source changed during the out-of-tree build");
  }
  verifySourceInputs(recipe, sourceInputs(repo, [], Object.keys(recipe.graph)), "Static zstd SDK producer");
  if (fileHash(archive) !== archiveSha256 || Object.values(toolHashes).some(row => fileHash(row.path) !== row.sha256)) {
    throw new Error("Static zstd archive or tools changed during compilation");
  }
  const artifacts = stagedFiles(selectedOutput), archiveOutput = {path: "lib/libzstd.a", sha256: artifacts["lib/libzstd.a"], bytes: fs.statSync(plan.installArchive).size};
  const buildOutput = {path: "build/static/libzstd.a", sha256: fileHash(plan.buildArchive), bytes: builtStat.size};
  const result = {version: 1, zstdVersion: zstdPin.version, platform: "linux", target: plan.target,
    archive: {path: archive, sha256: archiveSha256, bytes: archiveStat.size, members: sourceEntries},
    source: {files: sourceBefore.count, bytes: sourceBefore.bytes, sha256: sourceBefore.sha256,
      notices: {LICENSE: {path: "source/LICENSE", sha256: fileHash(path.join(plan.source, "LICENSE"))},
        COPYING: {path: "source/COPYING", sha256: fileHash(path.join(plan.source, "COPYING"))}}},
    config: {sha256: configHash}, tools: toolHashes, compiler: {version: compilerVersion}, recipe,
    plan: {jobs: plan.jobs, buildCommand: plan.buildCommand, target: plan.buildArchive, staticMode: plan.staticMode},
    buildOutput, install: archiveOutput, artifacts, buildOnly: true, executed: false, publicationQualified: false,
    scope: "Build-only static multithreaded libzstd 1.5.7 archive; no CLI, tests, shared library or release qualification."};
  fs.writeFileSync(path.join(selectedOutput, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  const inputs = nativeZstdInputs(selectedOutput);
  console.log("[native-sdk] Built pinned static zstd 1.5.7; no CLI or tests executed.");
  return {root: selectedOutput, ...inputs};
}

/** Admit only the exact source and archive closure produced by this build recipe. @param {string} root */
function nativeZstdInputs(root) {
  if (!absolute(root) || fs.realpathSync(root) !== root) {throw new Error("Select a canonical static zstd SDK root");}
  const receiptPath = path.join(root, "result.json"), receiptSha256 = fileHash(receiptPath), record = readBuildJson(receiptPath);
  const expectedKeys = ["archive", "artifacts", "buildOnly", "buildOutput", "compiler", "config", "executed", "install", "plan", "platform", "publicationQualified", "recipe", "scope", "source", "target", "tools", "version", "zstdVersion"];
  if (!record || typeof record !== "object" || Array.isArray(record) || Object.keys(record).sort().join(",") !== expectedKeys.sort().join(",") ||
      record.version !== 1 || record.zstdVersion !== zstdPin.version || record.platform !== "linux" || record.target !== "x86_64-linux-gnu" ||
      record.archive?.sha256 !== zstdPin.sha256 || !absolute(record.archive?.path) || !Number.isSafeInteger(record.archive?.bytes) ||
      record.archive.bytes < 1 || record.archive.bytes > maxArchiveBytes || !Number.isInteger(record.archive?.members) || record.archive.members < 1 ||
      record.archive.members > maxArchiveMembers || record.buildOnly !== true || record.executed !== false || record.publicationQualified !== false ||
      !isHash(record.config?.sha256) || !Number.isInteger(record.source?.files) || record.source.files < 1 || record.source.files > maxSourceFiles ||
      !Number.isInteger(record.source?.bytes) || record.source.bytes < 1 || record.source.bytes > maxSourceBytes || !isHash(record.source?.sha256) ||
      record.source.notices?.LICENSE?.path !== "source/LICENSE" || !isHash(record.source.notices.LICENSE.sha256) ||
      record.source.notices?.COPYING?.path !== "source/COPYING" || !isHash(record.source.notices.COPYING.sha256) ||
      record.compiler?.version !== undefined && !/^Intel\(R\) oneAPI DPC\+\+\/C\+\+ Compiler 2026\.2(?:\.|\s|$)/.test(record.compiler.version)) {
    throw new Error("Invalid static zstd SDK receipt");
  }
  if (record.compiler?.version === undefined || !record.plan || !Number.isInteger(record.plan.jobs) || record.plan.jobs < 1 || record.plan.jobs > 16) {
    throw new Error("Static zstd SDK tool receipt is incomplete");
  }
  const tools = verifyZstdTools(record.tools);
  verifyZstdArchiveInput(record.archive); verifyZstdRecipe(record.recipe);
  const toolPaths = Object.fromEntries(Object.entries(tools).map(([name, row]) => [name, row.path]));
  const plan = zstdSdkPlan(root, record.archive.path, toolPaths, record.plan.jobs, "linux");
  if (record.plan.staticMode !== plan.staticMode || record.plan.target !== plan.buildArchive ||
      JSON.stringify(record.plan.buildCommand) !== JSON.stringify(plan.buildCommand)) {
    throw new Error("Static zstd SDK plan or output identity is invalid");
  }
  const sourceRoot = path.join(root, "source"), source = sourceInventory(sourceRoot);
  if (source.count !== record.source.files || source.bytes !== record.source.bytes || source.sha256 !== record.source.sha256) {
    throw new Error("Static zstd source membership or bytes changed");
  }
  const notices = nativeZstdNoticeInputs(sourceRoot, record.source.notices);
  const artifacts = verifyZstdOutputs(root, record.buildOutput, record.install, record.artifacts);
  const sourceAfter = sourceInventory(sourceRoot);
  if (sourceAfter.sha256 !== source.sha256) {throw new Error("Static zstd source changed during admission");}
  const files = /** @type {Record<string, string>} */ (Object.create(null));
  const bind = (/** @type {string} */ file, /** @type {string} */ hash) => {
    if (files[file] !== undefined && files[file] !== hash) {throw new Error("Conflicting static zstd input identity: " + file);}
    files[file] = hash;
  };
  bind(record.archive.path, record.archive.sha256);
  for (const row of Object.values(tools)) {bind(row.path, row.sha256);}
  for (const [relative, hash] of Object.entries(sourceAfter.files)) {bind(path.join(sourceRoot, relative), hash);}
  bind(path.join(root, record.buildOutput.path), record.buildOutput.sha256);
  bind(path.join(root, record.install.path), record.install.sha256);
  bind(receiptPath, receiptSha256);
  for (const [file, hash] of Object.entries(files)) {
    if (fileHash(file) !== hash) {throw new Error("Static zstd input changed during admission: " + file);}
  }
  if (sourceInventory(sourceRoot).sha256 !== sourceAfter.sha256 || fileHash(receiptPath) !== receiptSha256) {
    throw new Error("Static zstd SDK inputs changed during admission");
  }
  return {archive: path.join(root, "lib", "libzstd.a"), artifacts, files, receipt: receiptPath, receiptSha256, notices};
}

if (require.main === module) {
  try {
    const [archive, output, configFile, extra] = process.argv.slice(2);
    if (!archive || !output || !configFile || extra || !absolute(configFile)) {throw new Error("Usage: MOM_AOT_BUILD_ONLY=1 node scripts/build-native-zstd.js /pinned/zstd-1.5.7.tar.gz /new/sdk /absolute/tools.json");}
    regularInput(configFile, "tool descriptor");
    const config = readBuildJson(configFile);
    buildNativeZstd(archive, output, config, process.env, fileHash(configFile));
  } catch (error) {console.error(error instanceof Error ? error.message : "Static zstd build failed"); process.exitCode = 1;}
}

module.exports = {zstdPin, toolNames, zstdSdkPlan, validateArchive, sourceInventory, stagedFiles, zstdBuildRecipe: () => sourceInputs(repo, [], sourceLeaves),
  verifyZstdArchiveInput, verifyZstdTools, verifyZstdRecipe, verifyZstdOutputs, nativeZstdNoticeInputs, nativeZstdInputs, buildNativeZstd};
