"use strict";

// Native CI Windows controller-support SDKs (I/O/OpenSSL and Hermes) built on Windows runners.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {ciBuildJobs, ciSuffix, jsonHash} = require("./native-ci-job");
const {hasExactBuildFields, fileHash, readBuildJson, readBuildJsonSnapshot, sourceInputs} = require("./build-source-inputs");
const {ioPins, ioRecipeFiles, nativeIoInputs} = require("./build-native-io");
const {nativeOpenSslInputs, opensslPin, opensslBuildRecipe} = require("./build-native-openssl");
const {verifyIoArchives, fetchIoArchives, fetchWindowsOpenSslArchives, sourceRecipe, origins} = require("./fetch-native-sdk-sources");
const {hermesPins, hermesRecipeFiles, hermesSdkPlan} = require("./build-native-hermes");
const {isStaticArchive} = require("./native-audit");
/** @typedef {{version:number,compiler:{version:string,sha256:string},libraries:Array<{name:string,source:string,path:string,bytes:number,sha256:string}>,licenses:Array<{source:string,path:string,bytes:number,sha256:string}>}} WindowsIntelRuntimeCatalog */
/** @typedef {{kind:"library",name:string,source:string,path:string,bytes:number,sha256:string}|{kind:"license",source:string,path:string,bytes:number,sha256:string}} WindowsIntelRuntimePin */
/** @typedef {WindowsIntelRuntimePin & {source:string}} WindowsIntelRuntimeEntry */
const windowsIntelRuntimeCatalog = /** @type {WindowsIntelRuntimeCatalog} */ (require("../runtime/windows-intel-runtime.json"));
const {safeRelative} = require("./native-ci-tar");
const repo = path.resolve(__dirname, "..");

/** Fresh native Windows controller-support outputs; the Hermes source tar is same-run Linux input.
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {"io" | "hermes"} [kind] */
function windowsControllerStatePaths(environment = process.env, platform = process.platform, kind = "io") {
  const runnerTemp = environment["RUNNER_TEMP"] || "";
  if (platform !== "win32" || !["io", "hermes"].includes(kind) || !path.isAbsolute(runnerTemp)) {
    throw new Error("Windows controller-support CI needs a Windows runner and RUNNER_TEMP");
  }
  const canonicalTemp = fs.realpathSync(runnerTemp), suffix = ciSuffix(environment),
    workName = "mom-native-windows-controller-" + kind + "-" + suffix, workRoot = path.join(canonicalTemp, workName),
    outputName = "native-windows-controller-" + kind + "-" + suffix,
    outputRoot = path.join(canonicalTemp, outputName), archive = path.join(canonicalTemp, outputName + ".tar"),
    expectedEnvironment = "MOM_NATIVE_WINDOWS_CONTROLLER_" + kind.toUpperCase() + "_DIR";
  const samePath = (/** @type {string} */ left, /** @type {string} */ right) => path.resolve(left).toLowerCase() === path.resolve(right).toLowerCase();
  if (!samePath(canonicalTemp, path.resolve(runnerTemp)) || !fs.lstatSync(canonicalTemp).isDirectory() ||
      !samePath(environment[expectedEnvironment] || "", workRoot) || path.dirname(workRoot) !== canonicalTemp ||
      path.dirname(outputRoot) !== canonicalTemp) {
    throw new Error("Windows controller-support paths must exactly match this CI job");
  }
  const incomingRoot = path.join(canonicalTemp, "incoming"), outputFile = environment["GITHUB_OUTPUT"] || "",
    summaryFile = environment["GITHUB_STEP_SUMMARY"] || "";
  return {kind, suffix, jobs: ciBuildJobs(environment), runnerTemp: canonicalTemp, workName, workRoot, outputName, outputRoot, archive, incomingRoot,
    ioSource: path.join(workRoot, "io-source"), opensslSource: path.join(workRoot, "openssl-source"),
    ioSdk: path.join(workRoot, "io-sdk"), opensslSdk: path.join(workRoot, "openssl-sdk"), opensslToolsFile: path.join(workRoot, "openssl-tools.json"),
    hermesSourceTar: path.join(incomingRoot, "native-hermes-state-" + suffix + "-sources.tar"),
    hermesExtractRoot: path.join(workRoot, "source-input"), hermesSourceName: "mom-native-hermes-prerequisites-" + suffix,
    hermesSource: path.join(workRoot, "source-input", "mom-native-hermes-prerequisites-" + suffix, "roots/hermes"),
    hermesSdk: path.join(workRoot, "hermes-sdk"), deadlineEpoch: environment["MOM_NATIVE_WINDOWS_CONTROLLER_DEADLINE_EPOCH"] || "",
    outputFile, summaryFile};
}

/** @param {string} epoch @param {"fetch" | "extract" | "build" | "package"} phase @param {number} [nowMs] */
function windowsControllerDeadlineBudget(epoch, phase, nowMs = Date.now()) {
  if (!/^\d{10}$/.test(epoch) || !Number.isSafeInteger(nowMs) || nowMs < 0 ||
      !["fetch", "extract", "build", "package"].includes(phase)) {
    throw new Error("Windows controller-support stage needs a valid bounded job-start deadline");
  }
  const remainingMs = Number(epoch) * 1000 - nowMs;
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0 || remainingMs > 60 * 60 * 1000) {
    throw new Error("Windows controller-support job deadline is expired or exceeds the 60-minute job bound");
  }
  const reserveMs = phase === "package" ? 30 * 1000 : 2 * 60 * 1000,
    maximumMs = {fetch: 10, extract: 5, build: 45, package: 5}[phase] * 60 * 1000;
  if (remainingMs <= reserveMs + 30 * 1000) {throw new Error("Insufficient aggregate job time remains for Windows controller-support " + phase);}
  return {timeoutMs: Math.min(maximumMs, remainingMs - reserveMs), remainingMs};
}

/** @typedef {{ar:string,cc:string,link:string,make:string,perl:string,rc:string,tar:string}} WindowsControllerOpenSslTools */
/** @typedef {{cc:string,cmake:string,ninja:string,openssl:WindowsControllerOpenSslTools}} WindowsControllerIoTools */
/** @typedef {{cc:string,cmake:string,cxx:string,git:string,ninja:string}} WindowsControllerHermesTools */

/** Parse a role-specific explicit tool table before any source acquisition or producer runs.
 * @param {NodeJS.ProcessEnv} environment @param {"io" | "hermes"} kind */
function windowsControllerTools(environment, kind) {
  const text = environment["MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS"] || "";
  if (!text || text.length > 16384) {throw new Error("Windows controller-support tools must be an explicit bounded JSON table");}
  let parsed;
  try {parsed = JSON.parse(text);} catch {throw new Error("Windows controller-support tool table is invalid JSON");}
  const keys = kind === "io" ? ["cc", "cmake", "ninja", "openssl"] : ["cc", "cmake", "cxx", "git", "ninja"];
  if (!hasExactBuildFields(parsed, keys)) {
    throw new Error("Windows controller-support tool table has an unexpected role or key set");
  }
  if (kind === "io" && (!parsed.openssl || typeof parsed.openssl !== "object" || Array.isArray(parsed.openssl) ||
      !hasExactBuildFields(parsed.openssl, ["ar", "cc", "link", "make", "perl", "rc", "tar"]))) {
    throw new Error("Windows controller I/O/OpenSSL tools must include the exact OpenSSL Windows tool set");
  }
  const files = kind === "io" ? [parsed.cc, parsed.cmake, parsed.ninja, ...Object.values(parsed.openssl)] : Object.values(parsed);
  for (const file of files) {
    if (typeof file !== "string" || file.length > 4096 || !path.isAbsolute(file) || /[\r\n\0]/.test(file)) {
      throw new Error("Windows controller-support tool path must be an explicit absolute path");
    }
    const stat = fs.lstatSync(file, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink() || path.resolve(fs.realpathSync(file)).toLowerCase() !== path.resolve(file).toLowerCase()) {
      throw new Error("Windows controller-support tool is missing, redirected or non-regular: " + file);
    }
  }
  return /** @type {WindowsControllerIoTools | WindowsControllerHermesTools} */ (parsed);
}

/** Exact public producer argv; paths stay under the fresh same-run output root.
 * @param {ReturnType<typeof windowsControllerStatePaths>} state @param {WindowsControllerIoTools} tools */
function windowsControllerIoBuildCommands(state, tools) {
  if (state.kind !== "io" || !tools?.openssl) {throw new Error("Windows I/O/OpenSSL command needs its exact role and tools");}
  return [
    {script: "scripts/build-native-io.js", args: [path.join(state.ioSource, "libuv-v1.51.0.tar.gz"),
      path.join(state.ioSource, "zlib-1.3.1.tar.gz"), state.ioSdk, tools.cmake, tools.cc, tools.ninja, state.jobs]},
    {script: "scripts/build-native-openssl.js", args: [path.join(state.opensslSource, "openssl-" + opensslPin.version + ".tar.gz"),
      state.opensslSdk, state.opensslToolsFile]},
  ];
}

/** The producer command plus the plan the producer derives from those same arguments; the receipt
 * check compares against `plan`, so the job count and paths cannot diverge from the build.
 * @param {ReturnType<typeof windowsControllerStatePaths>} state @param {WindowsControllerHermesTools} tools */
function windowsControllerHermesBuildCommand(state, tools) {
  if (state.kind !== "hermes" || !state.hermesSource || !tools) {throw new Error("Windows Hermes command needs its exact role and tools");}
  return {script: "scripts/build-native-hermes.js", args: [state.hermesSource, state.hermesSdk,
    tools.git, tools.cmake, tools.cc, tools.cxx, tools.ninja, state.jobs],
  plan: hermesSdkPlan(state.hermesSource, state.hermesSdk, tools, Number(state.jobs), "win32")};
}

/** @param {string} file @param {string} label */
function windowsControllerRegularFile(file, label) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!path.isAbsolute(file) || !stat?.isFile() || stat.isSymbolicLink() ||
      path.resolve(fs.realpathSync(file)).toLowerCase() !== path.resolve(file).toLowerCase()) {
    throw new Error(label + " must be a canonical regular file");
  }
  return stat;
}

/** @param {string} compilerFile @returns {WindowsIntelRuntimeEntry[]} */
function windowsControllerIntelRuntimeEntries(compilerFile) {
  const normalized = typeof compilerFile === "string" ? compilerFile.replaceAll("\\", "/").toLowerCase() : "";
  const catalog = windowsIntelRuntimeCatalog;
  if (!catalog || typeof catalog !== "object" || Array.isArray(catalog) ||
      !hasExactBuildFields(catalog, ["compiler", "libraries", "licenses", "version"]) || !catalog.compiler ||
      typeof catalog.compiler !== "object" || Array.isArray(catalog.compiler) ||
      !hasExactBuildFields(catalog.compiler, ["sha256", "version"]) || catalog.version !== 1 ||
      !/^\d{4}\.\d+(?:\.\d+)?$/.test(catalog.compiler.version) || !/^[a-f0-9]{64}$/.test(catalog.compiler.sha256) ||
      !Array.isArray(catalog.libraries) || catalog.libraries.length !== 5 || !Array.isArray(catalog.licenses) || catalog.licenses.length !== 3) {
    throw new Error("Intel runtime pin catalog has an unsupported shape or version");
  }
  if (typeof compilerFile !== "string" || !path.isAbsolute(compilerFile) ||
      !normalized.endsWith("/compiler/" + catalog.compiler.version + "/bin/icx-cl.exe")) {
    throw new Error("Intel runtime export requires the selected catalog icx-cl compiler path");
  }
  const compilerRoot = path.dirname(path.dirname(compilerFile));
  const entries = /** @type {WindowsIntelRuntimeEntry[]} */ ([
    ...catalog.libraries.map(entry => ({...entry, kind: "library"})),
    ...catalog.licenses.map(entry => ({...entry, kind: "license"})),
  ]);
  const destinations = new Set();
  for (const entry of entries) {
    if (!safeRelative(entry.source) || !safeRelative(entry.path) || !Number.isSafeInteger(entry.bytes) || entry.bytes < 1 ||
        !/^[a-f0-9]{64}$/.test(entry.sha256) || destinations.has(entry.path.toLowerCase()) ||
        entry.kind === "library" && (!hasExactBuildFields(entry, ["bytes", "kind", "name", "path", "sha256", "source"]) ||
          !/^[a-z0-9_.-]+\.lib$/i.test(entry.name) || entry.source !== "lib/" + entry.name || entry.path !== "intel-runtime/lib/" + entry.name) ||
        entry.kind === "license" && (!hasExactBuildFields(entry, ["bytes", "kind", "path", "sha256", "source"]) ||
          entry.path !== "intel-runtime/licenses/" + entry.source)) {
      throw new Error("Intel runtime pin catalog contains an invalid or duplicate file row");
    }
    destinations.add(entry.path.toLowerCase());
  }
  return entries.map(entry => ({...entry, source: path.join(compilerRoot, ...entry.source.split("/"))}));
}

/** @param {string} compilerFile @param {string} outputRoot @param {WindowsControllerFileLedger} ledger */
function copyWindowsControllerIntelRuntime(compilerFile, outputRoot, ledger) {
  const compilerStat = windowsControllerRegularFile(compilerFile, "Intel 2026.2 compiler"), compilerSha256 = fileHash(compilerFile),
    entries = windowsControllerIntelRuntimeEntries(compilerFile);
  if (compilerStat.size < 1 || compilerSha256 !== windowsIntelRuntimeCatalog.compiler.sha256) {
    throw new Error("Intel compiler content differs from the selected 2026.2 runtime pins");
  }
  /** @type {Array<{name:string,path:string,bytes:number,sha256:string}>} */ const libraries = [];
  /** @type {Array<{path:string,bytes:number,sha256:string}>} */ const licenses = [];
  for (const entry of entries) {
    const stat = windowsControllerRegularFile(entry.source, "Intel runtime " + (entry.kind === "library" ? entry.name : entry.path)), sha256 = fileHash(entry.source);
    if (stat.size !== entry.bytes || sha256 !== entry.sha256 || entry.kind === "library" && !isStaticArchive(entry.source)) {
      throw new Error("Intel runtime input differs from its selected 2026.2 pin: " + entry.path);
    }
    copyWindowsControllerPath(entry.source, path.join(outputRoot, ...entry.path.split("/")), entry.path, ledger);
    const row = entry.kind === "library" ? {name: entry.name, path: entry.path, bytes: stat.size, sha256} :
      {path: entry.path, bytes: stat.size, sha256};
    (entry.kind === "library" ? libraries : licenses).push(row);
  }
  const compiler = {path: compilerFile, sha256: compilerSha256}, sourceFiles = [{path: compilerFile, sha256: compilerSha256},
    ...entries.map(entry => ({path: entry.source, sha256: entry.sha256}))];
  if (sourceFiles.some(row => fileHash(row.path) !== row.sha256)) {throw new Error("Intel runtime source changed during handoff copy");}
  return {record: {compiler, libraries, licenses}, sourceFiles};
}

/** @param {string} file @param {number} maximum @param {string} label */
function windowsControllerReadJson(file, maximum, label) {
  const stat = windowsControllerRegularFile(file, label);
  if (stat.size > maximum) {throw new Error(label + " exceeds its size bound");}
  return readBuildJson(file);
}

/** @param {unknown} name @returns {name is string} */
function windowsHermesSourceName(name) {
  return typeof name === "string" && name.length <= 160 && /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name) && !/[. ]$/.test(name) &&
    !/^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)/i.test(name);
}

/** @param {string} value */
function windowsHermesHasControl(value) {
  return Array.from(value).some(char => char.charCodeAt(0) < 0x20 || char.charCodeAt(0) === 0x7f);
}

/** @param {unknown} value @returns {value is string} */
function canonicalHermesOriginalRoot(value) {
  return typeof value === "string" && value.length <= 4096 && path.posix.isAbsolute(value) && path.posix.normalize(value) === value &&
    path.posix.resolve(value) === value && !windowsHermesHasControl(value) && !value.includes("\\") && value !== "/" &&
    path.posix.basename(value) === "hermes" && path.posix.basename(path.posix.dirname(value)) === "roots";
}

/** The original WSL fetch receipts bind source paths but are never rewritten for transport.
 * @param {{hermesExtractRoot:string,hermesSourceName:string}} input @param {string} [originalRoot] */
function readWindowsHermesSource(input, originalRoot) {
  if (!input || !hasExactBuildFields(input, ["hermesExtractRoot", "hermesSourceName"]) || !windowsHermesSourceName(input.hermesSourceName) ||
      typeof input.hermesExtractRoot !== "string" || !path.isAbsolute(input.hermesExtractRoot) || path.resolve(input.hermesExtractRoot) !== input.hermesExtractRoot ||
      windowsHermesHasControl(input.hermesExtractRoot)) {
    throw new Error("Windows Hermes source reader needs its exact canonical extracted root and safe archive root name");
  }
  const expectedOriginalRoot = originalRoot === undefined ? "/tmp/" + input.hermesSourceName + "/roots/hermes" : originalRoot;
  if (!canonicalHermesOriginalRoot(expectedOriginalRoot)) {throw new Error("Hermes source original root must be a canonical POSIX roots/hermes path");}
  const extractStat = fs.lstatSync(input.hermesExtractRoot, {throwIfNoEntry: false}), parent = path.dirname(input.hermesExtractRoot),
    parentStat = fs.lstatSync(parent, {throwIfNoEntry: false});
  if (!parentStat?.isDirectory() || parentStat.isSymbolicLink() || fs.realpathSync(parent).toLowerCase() !== parent.toLowerCase() ||
      !extractStat?.isDirectory() || extractStat.isSymbolicLink() || fs.realpathSync(input.hermesExtractRoot).toLowerCase() !== input.hermesExtractRoot.toLowerCase()) {
    throw new Error("Windows Hermes extracted source root or its parent is missing or redirected");
  }
  const source = path.join(input.hermesExtractRoot, input.hermesSourceName), sourceStat = fs.lstatSync(source, {throwIfNoEntry: false});
  if (!sourceStat?.isDirectory() || sourceStat.isSymbolicLink() || fs.realpathSync(source).toLowerCase() !== source.toLowerCase()) {
    throw new Error("Windows Hermes archive root is missing or redirected");
  }
  const rootsFile = path.join(source, "roots.json"), fetchFile = path.join(source, "fetch.json"),
    roots = windowsControllerReadJson(rootsFile, 1 << 20, "Hermes source roots receipt"),
    fetch = windowsControllerReadJson(fetchFile, 4 << 20, "Hermes source fetch receipt"),
    recipe = sourceRecipe("hermes"), expectedRecipeSha = jsonHash(JSON.stringify(recipe));
  windowsControllerRegularFile(path.join(source, "roots/hermes/config"), "Extracted Hermes bare-source configuration");
  if (!hasExactBuildFields(roots, ["hermes"]) || roots.hermes !== expectedOriginalRoot || fetch.version !== 1 || fetch.target !== "hermes" ||
      fetch.status !== "PASS" || fetch.networked !== true || fetch.buildOnly !== true || fetch.compiled !== false || fetch.publicationQualified !== false ||
      !hasExactBuildFields(fetch.git, ["path", "sha256"]) || !path.posix.isAbsolute(fetch.git.path) || !/^[a-f0-9]{64}$/.test(fetch.git.sha256 || "") ||
      fetch.fetcherSha256 !== fileHash(path.join(repo, "scripts/fetch-native-sdk-sources.js")) ||
      !hasExactBuildFields(fetch.recipe, ["sha256", "files"]) || fetch.recipe.sha256 !== expectedRecipeSha || !isDeepStrictEqual(fetch.recipe.files, recipe.graph) ||
      !hasExactBuildFields(fetch.components, ["hermes"]) || fetch.components.hermes?.origin !== origins.hermes ||
      fetch.components.hermes?.revision !== hermesPins.revision || fetch.components.hermes?.root !== expectedOriginalRoot ||
      fetch.components.hermes?.gitDirectory !== "bare" || fetch.components.hermes?.shallow !== true || fetch.components.hermes?.partialClone !== false) {
    throw new Error("Hermes source receipts do not bind the original pinned bare Git root and current source recipe");
  }
  return {source: path.join(source, "roots/hermes"), rootsFile, fetchFile, fetch, fetchSha256: fileHash(fetchFile), rootsSha256: fileHash(rootsFile)};
}

/** The archive comes from the original Linux Hermes source cohort; do not fetch Git on Windows.
 * @param {Pick<ReturnType<typeof windowsControllerStatePaths>, "kind" | "hermesExtractRoot" | "hermesSourceTar" | "hermesSourceName">} state
 * @param {string} tarTool @param {number} timeoutMs @param {string} [originalRoot] */
function extractWindowsHermesSource(state, tarTool, timeoutMs, originalRoot) {
  if (!state || state.kind !== "hermes" || !windowsHermesSourceName(state.hermesSourceName) ||
      typeof state.hermesExtractRoot !== "string" || !path.isAbsolute(state.hermesExtractRoot) ||
      path.resolve(state.hermesExtractRoot) !== state.hermesExtractRoot || windowsHermesHasControl(state.hermesExtractRoot) ||
      typeof state.hermesSourceTar !== "string" || !path.isAbsolute(state.hermesSourceTar) ||
      typeof tarTool !== "string" || !path.isAbsolute(tarTool)) {
    throw new Error("Windows Hermes source extraction needs its exact role, safe root names and absolute paths");
  }
  if (originalRoot !== undefined && !canonicalHermesOriginalRoot(originalRoot)) {
    throw new Error("Hermes source original root must be a canonical POSIX roots/hermes path");
  }
  const outputParent = path.dirname(state.hermesExtractRoot), outputParentStat = fs.lstatSync(outputParent, {throwIfNoEntry: false}),
    outputStat = fs.lstatSync(state.hermesExtractRoot, {throwIfNoEntry: false});
  if (!outputParentStat?.isDirectory() || outputParentStat.isSymbolicLink() || fs.realpathSync(outputParent).toLowerCase() !== outputParent.toLowerCase() ||
      outputStat || !Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 30 * 60 * 1000) {
    throw new Error("Windows Hermes source extraction needs a safe root, canonical new output and bounded tar timeout");
  }
  windowsControllerRegularFile(tarTool, "Hermes source tar tool");
  const archiveStat = windowsControllerRegularFile(state.hermesSourceTar, "Same-run Hermes source tar");
  if (archiveStat.size < 1024 || archiveStat.size > (2 * 1024 ** 3)) {throw new Error("Same-run Hermes source tar is outside its 2 GiB bound");}
  /** @param {string[]} args @param {number} maxBuffer */
  const runTar = (args, maxBuffer) => {
    const result = spawnSync(tarTool, args, {encoding: "utf8", windowsHide: true, timeout: timeoutMs, maxBuffer});
    if (result.error || result.signal || result.status !== 0) {throw new Error("Same-run Hermes source tar operation failed", {cause: result.error});}
    return result.stdout || "";
  };
  const names = runTar(["-tf", state.hermesSourceTar], 64 << 20).split(/\r?\n/).filter(Boolean),
    verbose = runTar(["-tvf", state.hermesSourceTar], 64 << 20).split(/\r?\n/).filter(Boolean);
  if (!names.length || names.length > 200000 || names.length !== verbose.length || names.some(name => {
    const normalized = name.endsWith("/") ? name.slice(0, -1) : name;
    return !safeRelative(normalized) || normalized.split("/")[0] !== state.hermesSourceName;
  }) || verbose.some(line => !["-", "d"].includes(line[0] || ""))) {
    throw new Error("Same-run Hermes source tar has unsafe paths, links, special files or roots");
  }
  if (!names.includes(state.hermesSourceName + "/roots/hermes/config") || !names.includes(state.hermesSourceName + "/fetch.json") ||
      !names.includes(state.hermesSourceName + "/roots.json")) {throw new Error("Same-run Hermes source tar omits its bare source or original fetch receipts");}
  fs.mkdirSync(state.hermesExtractRoot, {mode: 0o700});
  runTar(["-xf", state.hermesSourceTar, "-C", state.hermesExtractRoot, "--", state.hermesSourceName], 4 << 20);
  return {...readWindowsHermesSource({hermesExtractRoot: state.hermesExtractRoot, hermesSourceName: state.hermesSourceName}, originalRoot),
    tarSha256: fileHash(state.hermesSourceTar)};
}

/** @typedef {{files:Array<{path:string,bytes:number,sha256:string}>,bytes:number}} WindowsControllerFileLedger */
/** @param {string} source @param {string} target @param {string} relative @param {WindowsControllerFileLedger} ledger */
function copyWindowsControllerPath(source, target, relative, ledger) {
  if (!safeRelative(relative)) {throw new Error("Windows controller selected path is unsafe");}
  const stat = fs.lstatSync(source);
  if (stat.isSymbolicLink() || fs.realpathSync(source).toLowerCase() !== path.resolve(source).toLowerCase()) {
    throw new Error("Windows controller SDK selection contains a symlink or redirected path");
  }
  if (stat.isDirectory()) {
    fs.mkdirSync(target, {recursive: true});
    for (const entry of fs.readdirSync(source, {withFileTypes: true}).sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0)) {
      if (!entry.isDirectory() && !entry.isFile()) {throw new Error("Windows controller SDK selection contains a special file");}
      copyWindowsControllerPath(path.join(source, entry.name), path.join(target, entry.name), relative + "/" + entry.name, ledger);
    }
    return;
  }
  if (!stat.isFile() || ledger.files.length >= 50000 || (ledger.bytes += stat.size) > (512 << 20)) {
    throw new Error("Windows controller selected files exceed their regular-file count/size bound");
  }
  fs.mkdirSync(path.dirname(target), {recursive: true});
  const sha256 = fileHash(source);
  fs.copyFileSync(source, target, fs.constants.COPYFILE_EXCL);
  const copied = fs.lstatSync(target);
  if (!copied.isFile() || copied.isSymbolicLink() || copied.size !== stat.size || fileHash(target) !== sha256) {
    throw new Error("Windows controller SDK handoff copy differs from the producer output");
  }
  ledger.files.push({path: relative, bytes: stat.size, sha256});
}

/** @param {string} root @param {string} name */
function windowsControllerPackageInventory(root, name) {
  /** @type {Array<{path:string,bytes:number,sha256:string}>} */
  const files = [], members = [name]; let bytes = 0;
  const visit = (/** @type {string} */ directory, /** @type {string} */ relative) => {
    const stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory).toLowerCase() !== path.resolve(directory).toLowerCase()) {
      throw new Error("Windows controller package contains a redirected directory");
    }
    for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0)) {
      const item = path.join(directory, entry.name), child = relative ? relative + "/" + entry.name : entry.name;
      if (!safeRelative(child)) {throw new Error("Windows controller package path is unsafe");}
      if (entry.isSymbolicLink()) {throw new Error("Windows controller package contains a symlink");}
      members.push(name + "/" + child);
      if (entry.isDirectory()) {visit(item, child); continue;}
      const itemStat = fs.lstatSync(item);
      if (!entry.isFile() || itemStat.isSymbolicLink() || !itemStat.isFile() || files.length >= 50000 || (bytes += itemStat.size) > (512 << 20)) {
        throw new Error("Windows controller package exceeds its regular-file size/count bound");
      }
      files.push({path: child, bytes: itemStat.size, sha256: fileHash(item)});
    }
  };
  visit(root, "");
  if (!files.length) {throw new Error("Windows controller package is empty");}
  return {files: files.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0), members: members.sort(), bytes};
}

/** @param {Array<{path:string,bytes:number,sha256:string}>} files @param {unknown} sources */
function verifyWindowsHermesSourceHandoff(files, sources) {
  const prefix = "hermes/source/", selected = files.filter(row => row.path.startsWith(prefix));
  if (!selected.length || !sources || typeof sources !== "object" || Array.isArray(sources) || selected.some(row => {
    const sourcePath = row.path.slice(prefix.length);
    return !safeRelative(sourcePath) || !/^[a-f0-9]{64}$/.test(row.sha256) || Reflect.get(sources, sourcePath) !== row.sha256;
  })) {throw new Error("Windows Hermes selected source files do not match the producer's tracked-source hashes");}
  return selected.length;
}

/** Pack only a selected build-only closure and its unchanged producer receipts.
 * @param {Pick<ReturnType<typeof windowsControllerStatePaths>,"archive"|"runnerTemp"|"outputName"|"outputRoot">} state
 * @param {string} tarTool @param {Record<string,unknown>} handoff */
function packWindowsControllerHandoff(state, tarTool, handoff, timeoutMs = 5 * 60 * 1000) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 5 * 60 * 1000) {throw new Error("Windows controller handoff timeout is invalid");}
  if (fs.existsSync(state.archive)) {throw new Error("Windows controller handoff tar already exists");}
  const before = windowsControllerPackageInventory(state.outputRoot, state.outputName), record = {...handoff, files: before.files};
  fs.writeFileSync(path.join(state.outputRoot, "handoff.json"), JSON.stringify(record, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const complete = windowsControllerPackageInventory(state.outputRoot, state.outputName), toolHash = fileHash(tarTool), create = spawnSync(tarTool,
    ["-cf", state.archive, "-C", state.runnerTemp, state.outputName], {encoding: "utf8", windowsHide: true,
      timeout: timeoutMs, maxBuffer: 4 << 20});
  if (create.error || create.signal || create.status !== 0) {throw new Error("Windows controller handoff tar creation failed", {cause: create.error});}
  const listing = spawnSync(tarTool, ["-tf", state.archive], {encoding: "utf8", windowsHide: true, timeout: Math.min(timeoutMs, 2 * 60 * 1000), maxBuffer: 16 << 20}),
    archiveStat = fs.lstatSync(state.archive, {throwIfNoEntry: false});
  if (listing.error || listing.signal || listing.status !== 0 || typeof listing.stdout !== "string" || fileHash(tarTool) !== toolHash ||
      !archiveStat?.isFile() || archiveStat.isSymbolicLink() || archiveStat.size > (576 << 20)) {
    throw new Error("Windows controller handoff tar listing or tool verification failed", {cause: listing.error});
  }
  const actual = listing.stdout.split(/\r?\n/).filter(Boolean).map(name => name.replaceAll("\\", "/").replace(/\/$/, "")).sort();
  if (!isDeepStrictEqual(actual, complete.members) || complete.bytes > (512 << 20)) {
    throw new Error("Windows controller handoff tar has unexpected selected members or size");
  }
  for (const file of complete.files) {
    if (fileHash(path.join(state.outputRoot, ...file.path.split("/"))) !== file.sha256) {throw new Error("Windows controller handoff changed while packing");}
  }
  return {bundleSha256: fileHash(state.archive), bundleBytes: fs.statSync(state.archive).size,
    handoffSha256: fileHash(path.join(state.outputRoot, "handoff.json")), files: before.files.length, selectedBytes: before.bytes};
}

/** @param {NodeJS.ProcessEnv} environment @param {Record<string,string>} values */
function appendWindowsControllerOutputs(environment, values) {
  const output = environment["GITHUB_OUTPUT"] || "", stat = path.isAbsolute(output) ? fs.lstatSync(output, {throwIfNoEntry: false}) : undefined;
  if (!stat?.isFile() || stat.isSymbolicLink() || Object.entries(values).some(([key, value]) =>
    !/^[a-z][a-z0-9_]*$/.test(key) || !/^(?:[a-f0-9]{64}|[0-9]+)$/.test(value))) {
    throw new Error("Windows controller-support GitHub outputs are invalid");
  }
  fs.appendFileSync(output, Object.entries(values).map(([key, value]) => key + "=" + value).join("\n") + "\n", "utf8");
}

/** @param {string} file @param {string[]} args @param {NodeJS.ProcessEnv} environment @param {number} timeoutMs */
function runWindowsControllerProducer(file, args, environment, timeoutMs) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 55 * 60 * 1000) {throw new Error("Windows controller producer timeout is invalid");}
  const result = spawnSync(process.execPath, [path.join(repo, file), ...args], {cwd: repo, env: environment, stdio: "inherit",
    windowsHide: true, timeout: timeoutMs});
  if (result.error || result.signal || result.status !== 0) {throw new Error("Windows controller public producer failed: " + file, {cause: result.error});}
}

/** @param {ReturnType<typeof windowsControllerStatePaths>} state */
function windowsControllerPreflight(state) {
  const outputStat = fs.lstatSync(state.outputFile, {throwIfNoEntry: false}), summaryStat = fs.lstatSync(state.summaryFile, {throwIfNoEntry: false});
  if (!outputStat?.isFile() || outputStat.isSymbolicLink() || !summaryStat?.isFile() || summaryStat.isSymbolicLink() ||
      [state.workRoot, state.outputRoot, state.archive].some(file => fs.existsSync(file))) {
    throw new Error("Windows controller-support stage needs new run-scoped outputs and regular workflow files");
  }
  windowsControllerDeadlineBudget(state.deadlineEpoch, "fetch");
  fs.mkdirSync(state.workRoot, {mode: 0o700});
}

/** Fetch and build original native Windows libuv/zlib and OpenSSL inputs; hand off only selected files. */
async function buildWindowsControllerIoCi() {
  const state = windowsControllerStatePaths(process.env, process.platform, "io"),
    tools = /** @type {WindowsControllerIoTools} */ (windowsControllerTools(process.env, "io")),
    environment = /** @type {NodeJS.ProcessEnv} */ ({...process.env, MOM_AOT_BUILD_ONLY: "1", OMP_NUM_THREADS: state.jobs});
  windowsControllerPreflight(state);
  const toolHashes = Object.fromEntries([tools.cc, tools.cmake, tools.ninja, ...Object.values(tools.openssl)].map(file => [file, fileHash(file)])),
    ioBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "fetch");
  const ioFetch = await fetchIoArchives(state.ioSource, environment, true, {timeoutMs: Math.min(ioBudget.timeoutMs, 10 * 60 * 1000)}),
    ioFetched = verifyIoArchives(state.ioSource), ioFetchFile = path.join(state.ioSource, "fetch.json"), ioFetchSha = fileHash(ioFetchFile),
    opensslBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "fetch"),
    opensslFetch = await fetchWindowsOpenSslArchives(state.opensslSource, environment, true,
      {timeoutMs: Math.min(opensslBudget.timeoutMs, 10 * 60 * 1000)}),
    opensslFetchFile = path.join(state.opensslSource, "fetch.json"), opensslArchive = path.join(state.opensslSource, "openssl-" + opensslPin.version + ".tar.gz"),
    opensslFetchSha = fileHash(opensslFetchFile);
  if (ioFetch.status !== "PASS" || ioFetch.target !== "io" || ioFetched.status !== "PASS" || ioFetched.target !== "io" ||
      ioFetched.publicationQualified !== false ||
      fileHash(opensslArchive) !== opensslPin.sha256 || opensslFetch.status !== "PASS" || opensslFetch.target !== "windows-openssl" ||
      opensslFetch.archives?.["openssl"]?.sha256 !== opensslPin.sha256 || opensslFetch.archives?.["openssl"]?.path !== opensslArchive) {
    throw new Error("Windows controller source acquisition did not retain its exact pinned inputs");
  }
  const opensslConfig = {version: 1, jobs: Number(state.jobs), tools: tools.openssl};
  fs.writeFileSync(state.opensslToolsFile, JSON.stringify(opensslConfig, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const sourceHashes = {[opensslArchive]: opensslPin.sha256,
      [path.join(state.ioSource, "libuv-v1.51.0.tar.gz")]: ioPins.libuv.sha256,
      [path.join(state.ioSource, "zlib-1.3.1.tar.gz")]: ioPins.zlib.sha256,
      [ioFetchFile]: ioFetchSha, [opensslFetchFile]: opensslFetchSha, [state.opensslToolsFile]: fileHash(state.opensslToolsFile)},
    ioRecipe = sourceInputs(repo, [], ioRecipeFiles),
    opensslRecipe = opensslBuildRecipe();
  const commands = windowsControllerIoBuildCommands(state, tools), ioCommand = commands[0], opensslCommand = commands[1];
  if (!ioCommand || !opensslCommand) {throw new Error("Windows controller I/O/OpenSSL producer command set is incomplete");}
  const ioBuildBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "build");
  runWindowsControllerProducer(ioCommand.script, ioCommand.args, environment, Math.min(ioBuildBudget.timeoutMs, 35 * 60 * 1000));
  const ioAdmitted = nativeIoInputs(state.ioSdk, "win32"), {receiptSha256: ioReceiptSha} = ioAdmitted,
    opensslBuildBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "build");
  runWindowsControllerProducer(opensslCommand.script, opensslCommand.args, environment, Math.min(opensslBuildBudget.timeoutMs, 45 * 60 * 1000));
  const opensslAdmitted = nativeOpenSslInputs(state.opensslSdk, "win32"), {receiptSha256: opensslReceiptSha} = opensslAdmitted;
  const {value: ioReceipt} = readBuildJsonSnapshot(ioAdmitted.receipt, undefined,
    {sha256: ioReceiptSha, message: "Windows controller I/O receipt changed after admission"});
  const {value: opensslReceipt} = readBuildJsonSnapshot(opensslAdmitted.receipt, undefined,
    {sha256: opensslReceiptSha, message: "Windows controller OpenSSL receipt changed after admission"});
  const expectedIoInputs = {
      [path.join(state.ioSource, "libuv-v1.51.0.tar.gz")]: ioPins.libuv.sha256,
      [path.join(state.ioSource, "zlib-1.3.1.tar.gz")]: ioPins.zlib.sha256},
    expectedIoTools = Object.fromEntries([tools.cmake, tools.cc, tools.ninja].map(file => [file, fileHash(file)])),
    expectedOpenSslTools = Object.fromEntries(Object.entries(tools.openssl).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}]));
  if (!isDeepStrictEqual(ioReceipt.recipe, ioRecipe) || !isDeepStrictEqual(ioReceipt.inputs, expectedIoInputs) ||
      !isDeepStrictEqual(ioReceipt.fingerprints, expectedIoTools) || ioReceipt.target !== "win32" || ioReceipt.buildOnly !== true ||
      !isDeepStrictEqual(opensslReceipt.recipe, opensslRecipe) || opensslReceipt.target !== "VC-WIN64A" ||
      opensslReceipt.platform !== "win32" || opensslReceipt.config?.sha256 !== fileHash(state.opensslToolsFile) ||
      opensslReceipt.archive?.path !== opensslArchive || opensslReceipt.archive?.sha256 !== opensslPin.sha256 ||
      !isDeepStrictEqual(opensslReceipt.tools, expectedOpenSslTools) ||
      Object.entries(sourceHashes).some(([file, sha]) => fileHash(file) !== sha) ||
      Object.entries(toolHashes).some(([file, sha]) => fileHash(file) !== sha)) {
    throw new Error("Windows controller I/O/OpenSSL source, recipe, tool or producer receipt changed");
  }
  const ledger = {files: [], bytes: 0};
  fs.mkdirSync(state.outputRoot, {mode: 0o700});
  for (const [relative, sha] of Object.entries(ioAdmitted.artifacts).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) {
    if (!relative.startsWith("include/") && !relative.startsWith("lib/")) {throw new Error("Native I/O producer selected a non-header/library handoff member");}
    if (fileHash(path.join(state.ioSdk, ...relative.split("/"))) !== sha) {throw new Error("Native I/O selected artifact changed before handoff");}
    copyWindowsControllerPath(path.join(state.ioSdk, ...relative.split("/")), path.join(state.outputRoot, "io-sdk", ...relative.split("/")),
      "io-sdk/" + relative, ledger);
  }
  const sslMembers = Object.keys(opensslAdmitted.artifacts).filter(name => name.startsWith("include/") || name === "lib/libcrypto.lib" || name === "lib/libssl.lib").sort();
  if (!sslMembers.includes("lib/libcrypto.lib") || !sslMembers.includes("lib/libssl.lib") ||
      Object.keys(opensslAdmitted.artifacts).some(name => !name.startsWith("include/") &&
        !["bin/openssl.exe", "lib/libcrypto.lib", "lib/libssl.lib"].includes(name))) {
    throw new Error("OpenSSL selected handoff omits a static library or has unexpected producer artifacts");
  }
  for (const relative of sslMembers) {
    if (fileHash(path.join(state.opensslSdk, ...relative.split("/"))) !== opensslAdmitted.artifacts[relative]) {throw new Error("OpenSSL selected artifact changed before handoff");}
    copyWindowsControllerPath(path.join(state.opensslSdk, ...relative.split("/")), path.join(state.outputRoot, "openssl-sdk", ...relative.split("/")),
      "openssl-sdk/" + relative, ledger);
  }
  copyWindowsControllerPath(path.join(state.opensslSdk, "source", "LICENSE.txt"), path.join(state.outputRoot, "openssl-sdk/source/LICENSE.txt"),
    "openssl-sdk/source/LICENSE.txt", ledger);
  copyWindowsControllerPath(ioAdmitted.receipt, path.join(state.outputRoot, "io-sdk/result.json"), "io-sdk/result.json", ledger);
  copyWindowsControllerPath(opensslAdmitted.receipt, path.join(state.outputRoot, "openssl-sdk/result.json"), "openssl-sdk/result.json", ledger);
  copyWindowsControllerPath(ioFetchFile, path.join(state.outputRoot, "provenance/io-fetch.json"), "provenance/io-fetch.json", ledger);
  copyWindowsControllerPath(opensslFetchFile, path.join(state.outputRoot, "provenance/openssl-fetch.json"), "provenance/openssl-fetch.json", ledger);
  copyWindowsControllerPath(state.opensslToolsFile, path.join(state.outputRoot, "openssl-sdk/producer-tools.json"),
    "openssl-sdk/producer-tools.json", ledger);
  if (fileHash(path.join(state.outputRoot, "io-sdk/result.json")) !== ioReceiptSha ||
      fileHash(path.join(state.outputRoot, "openssl-sdk/result.json")) !== opensslReceiptSha ||
      fileHash(path.join(state.outputRoot, "openssl-sdk/producer-tools.json")) !== opensslReceipt.config.sha256) {
    throw new Error("Windows controller-support producer receipt or tool descriptor changed during handoff");
  }
  if (Object.entries(sourceHashes).some(([file, sha]) => fileHash(file) !== sha) ||
      Object.entries(toolHashes).some(([file, sha]) => fileHash(file) !== sha)) {
    throw new Error("Windows controller-support inputs changed while packaging the handoff");
  }
  const packageBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "package"),
    packaged = packWindowsControllerHandoff(state, tools.openssl.tar, {version: 1, format: "mom-windows-controller-support-handoff-v1",
      target: "win32", architecture: "x64", kind: "io", buildOnly: true, executed: false, publicationQualified: false,
      scope: "Selected public libuv/zlib and OpenSSL headers/static archives plus unchanged receipts; not full SDK re-admission or application qualification.",
      producers: [{script: ioCommand.script, receipt: "io-sdk/result.json", receiptSha256: ioReceiptSha},
        {script: opensslCommand.script, receipt: "openssl-sdk/result.json", receiptSha256: opensslReceiptSha}],
      sourceReceipts: [{file: "provenance/io-fetch.json", sha256: ioFetchSha, archives: {libuv: ioPins.libuv.sha256, zlib: ioPins.zlib.sha256}},
        {file: "provenance/openssl-fetch.json", sha256: opensslFetchSha, archives: {openssl: opensslPin.sha256}}],
      omitted: ["OpenSSL source/build trees", "openssl.exe", "compiler/build tools", "DLLs"]}, packageBudget.timeoutMs);
  appendWindowsControllerOutputs(environment, {bundle_sha256: packaged.bundleSha256, bundle_bytes: String(packaged.bundleBytes),
    io_receipt_sha256: ioReceiptSha, openssl_receipt_sha256: opensslReceiptSha, handoff_sha256: packaged.handoffSha256,
    io_bundle_sha256: packaged.bundleSha256, io_handoff_sha256: packaged.handoffSha256});
  fs.appendFileSync(state.summaryFile, `### Native Windows controller I/O/OpenSSL support inputs\n- Built the pinned same-run Windows static producers and retained ${packaged.files} selected files (${packaged.selectedBytes} bytes).\n- This selected handoff is build-only; no complete-SDK, application, runtime or publication qualification is claimed.\n`, "utf8");
  return packaged;
}

/** Verify and build same-run Hermes source on Windows, retaining only headers/config/static libraries. */
function buildWindowsControllerHermesCi() {
  const state = windowsControllerStatePaths(process.env, process.platform, "hermes"),
    tools = /** @type {WindowsControllerHermesTools} */ (windowsControllerTools(process.env, "hermes")),
    environment = /** @type {NodeJS.ProcessEnv} */ ({...process.env, MOM_AOT_BUILD_ONLY: "1", OMP_NUM_THREADS: state.jobs});
  windowsControllerPreflight(state);
  const tarTool = path.join(environment["SystemRoot"] || environment["WINDIR"] || "", "System32", "tar.exe");
  windowsControllerRegularFile(tarTool, "Windows inbox tar tool");
  const tarSha = fileHash(tarTool), source = extractWindowsHermesSource(state, tarTool,
      windowsControllerDeadlineBudget(state.deadlineEpoch, "extract").timeoutMs), command = windowsControllerHermesBuildCommand(state, tools),
    plan = command.plan;
  if (fs.existsSync(state.hermesSdk)) {throw new Error("Windows Hermes SDK output must be new");}
  const producerRecipe = sourceInputs(repo, [], hermesRecipeFiles("win32")), fingerprints = Object.fromEntries([...Object.values(tools), ...plan.patches.map(file => path.join(repo, file))].map(file => [file, fileHash(file)])),
    buildBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "build");
  runWindowsControllerProducer(command.script, command.args, environment, Math.min(buildBudget.timeoutMs, 45 * 60 * 1000));
  const resultFile = path.join(state.hermesSdk, "result.json"), beforeFile = path.join(state.hermesSdk, "before.json"),
    result = windowsControllerReadJson(resultFile, 16 << 20, "Windows Hermes producer receipt"), before = windowsControllerReadJson(beforeFile, 16 << 20, "Windows Hermes producer plan");
  if (result.version !== 1 || result.target !== "win32" || !isDeepStrictEqual(result.pins, hermesPins) ||
      result.source !== plan.source || result.build !== plan.build || result.buildOnly !== true || result.publicationQualified !== false ||
      !isDeepStrictEqual(result.recipe, producerRecipe) || !isDeepStrictEqual(before.configure, plan.configure) || !isDeepStrictEqual(before.compile, plan.compile) ||
      !isDeepStrictEqual(result.fingerprints, fingerprints) || !isDeepStrictEqual(before.fingerprints, fingerprints) ||
      !isDeepStrictEqual(Object.keys(result.artifacts || {}).sort(), Object.keys(plan.artifacts).sort())) {
    throw new Error("Windows Hermes producer receipt differs from the exact pinned source/tool/build-only plan");
  }
  for (const [name, file] of Object.entries(plan.artifacts)) {
    const row = result.artifacts[name], stat = windowsControllerRegularFile(file, "Hermes " + name + " output");
    if (!row || row.path !== file || row.bytes !== stat.size || !/^[a-f0-9]{64}$/.test(row.sha256 || "") || fileHash(file) !== row.sha256 ||
        name !== "compiler" && !isStaticArchive(file)) {throw new Error("Windows Hermes producer artifact changed: " + name);}
  }
  if (fileHash(tarTool) !== tarSha || fileHash(source.fetchFile) !== source.fetchSha256 || fileHash(source.rootsFile) !== source.rootsSha256 ||
      fileHash(state.hermesSourceTar) !== source.tarSha256) {throw new Error("Same-run Hermes source input or Windows tar tool changed during production");}
  const ledger = {files: [], bytes: 0};
  fs.mkdirSync(state.outputRoot, {mode: 0o700});
  const intelRuntime = copyWindowsControllerIntelRuntime(tools.cc, state.outputRoot, ledger);
  if (intelRuntime.record.compiler.sha256 !== fingerprints[tools.cc]) {
    throw new Error("Intel runtime export compiler differs from the same-run Hermes tool fingerprint");
  }
  /** @type {Array<[string,string]>} */
  const selected = [
    [path.join(plan.source, "LICENSE"), "hermes/source/LICENSE"],
    [path.join(plan.source, "include"), "hermes/source/include"], [path.join(plan.source, "API"), "hermes/source/API"],
    [path.join(plan.source, "public"), "hermes/source/public"], [path.join(plan.build, "lib/config"), "hermes/build/lib/config"],
    [plan.artifacts.runtime, "hermes/build/lib/hermesvmlean_a.lib"], [plan.artifacts.jsi, "hermes/build/jsi/jsi.lib"],
    [plan.artifacts.context, "hermes/build/external/boost/boost_1_86_0/libs/context/boost_context.lib"],
  ];
  for (const [sourcePath, relative] of selected) {
    copyWindowsControllerPath(sourcePath, path.join(state.outputRoot, ...relative.split("/")), relative, ledger);
  }
  verifyWindowsHermesSourceHandoff(ledger.files, result.sources);
  const receiptFile = path.join(state.hermesSdk, "result.json"), beforeHash = fileHash(beforeFile), receiptSha = fileHash(receiptFile),
    sourceFetchTarget = path.join(state.outputRoot, "provenance/hermes-fetch.json"), sourceRootsTarget = path.join(state.outputRoot, "provenance/hermes-roots.json");
  copyWindowsControllerPath(receiptFile, path.join(state.outputRoot, "hermes/result.json"), "hermes/result.json", ledger);
  copyWindowsControllerPath(beforeFile, path.join(state.outputRoot, "hermes/before.json"), "hermes/before.json", ledger);
  copyWindowsControllerPath(source.fetchFile, sourceFetchTarget, "provenance/hermes-fetch.json", ledger);
  copyWindowsControllerPath(source.rootsFile, sourceRootsTarget, "provenance/hermes-roots.json", ledger);
  const packageBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "package"),
    packaged = packWindowsControllerHandoff(state, tarTool, {version: 1, format: "mom-windows-controller-support-handoff-v1",
      target: "win32", architecture: "x64", kind: "hermes", buildOnly: true, executed: false, publicationQualified: false,
      scope: "Selected Hermes root license, headers/config/static libraries and Intel 2026.2 static runtime/license files plus unchanged same-run source and build receipts; no AOT compiler or full-SDK re-admission.",
      producers: [{script: command.script, receipt: "hermes/result.json", receiptSha256: receiptSha, plan: "hermes/before.json", planSha256: beforeHash}],
      sourceReceipts: [{file: "provenance/hermes-fetch.json", sha256: source.fetchSha256, rootsFile: "provenance/hermes-roots.json", rootsSha256: source.rootsSha256,
        revision: hermesPins.revision}], intelRuntime: intelRuntime.record,
      omitted: ["Hermes Git/build trees", "shermes.exe", "hermesPublic.lib", "compiler/build tools", "DLLs"]}, packageBudget.timeoutMs);
  if (fileHash(tarTool) !== tarSha || fileHash(source.fetchFile) !== source.fetchSha256 || fileHash(source.rootsFile) !== source.rootsSha256 ||
      fileHash(state.hermesSourceTar) !== source.tarSha256 || Object.entries(fingerprints).some(([file, sha]) => fileHash(file) !== sha) ||
      intelRuntime.sourceFiles.some(row => fileHash(row.path) !== row.sha256) ||
      [...intelRuntime.record.libraries, ...intelRuntime.record.licenses].some(row => {
        const file = path.join(state.outputRoot, ...row.path.split("/")), stat = fs.lstatSync(file, {throwIfNoEntry: false});
        return !stat?.isFile() || stat.isSymbolicLink() || stat.size !== row.bytes || fileHash(file) !== row.sha256;
      })) {
    throw new Error("Windows Hermes source, tool or patch changed while packaging the handoff");
  }
  appendWindowsControllerOutputs(environment, {bundle_sha256: packaged.bundleSha256, bundle_bytes: String(packaged.bundleBytes),
    hermes_receipt_sha256: receiptSha, source_receipt_sha256: source.fetchSha256, handoff_sha256: packaged.handoffSha256,
    hermes_bundle_sha256: packaged.bundleSha256, hermes_handoff_sha256: packaged.handoffSha256});
  fs.appendFileSync(state.summaryFile, `### Native Windows controller Hermes support inputs\n- Built the pinned same-run Windows Hermes producer and retained ${packaged.files} selected files (${packaged.selectedBytes} bytes).\n- The Linux Hermes AOT tool remains a separate input; this handoff does not include or qualify a Windows executable for Linux.\n`, "utf8");
  return packaged;
}

module.exports = {buildWindowsControllerHermesCi, buildWindowsControllerIoCi, canonicalHermesOriginalRoot,
  copyWindowsControllerIntelRuntime, copyWindowsControllerPath, extractWindowsHermesSource, packWindowsControllerHandoff,
  readWindowsHermesSource, verifyWindowsHermesSourceHandoff, windowsControllerDeadlineBudget, windowsControllerHermesBuildCommand,
  windowsControllerIntelRuntimeEntries, windowsControllerIoBuildCommands, windowsControllerPackageInventory,
  windowsControllerReadJson, windowsControllerRegularFile, windowsControllerStatePaths, windowsControllerTools,
  windowsHermesSourceName};
