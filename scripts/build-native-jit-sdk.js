"use strict";

// Build the original source-JIT archive graph; the small emitter/plan stay independently
// usable before this expensive stage. Nothing here executes a JIT proof or queries a GPU.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {jitSdkPlan, jitSourceInputs, literal} = require("./native-jit-sdk");
const {jitBuildStages, jitBuildStage, jitBuildOutputs, jitBuildHandoff} = require("./native-jit-stages");
const {verifyPreparedSdkSources} = require("./prepare-native-sdk-sources");
const {verifyJitResources} = require("./prepare-native-sycl-jit-resources");
const {compilerArchives} = require("./native-compiler-isolation");
const {fileHash, readBuildJson, readBuildText, sourceInputs, verifySourceInputs, buildFileInputs} = require("./build-source-inputs");
const {nativeSdkEnvironment} = require("./build-environment");
const {isStaticArchive} = require("./native-audit");
const {isDeepStrictEqual} = require("node:util");
const repo = path.resolve(__dirname, "..");
/** @typedef {{version: number, target: 'linux'|'windows', sources: string, sdk: string, resources: string,
 * tools: import('./native-jit-sdk').Tools, jobs: number, inputs?: string[]}} Config */
/** @param {unknown} p @returns {p is string} */
const absolute = p => typeof p === "string" && p.length <= 4096 && path.isAbsolute(p) && !Array.from(p).some(c => c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127);

/** Only explicit build inputs, never inherited setup scripts or free-form CMake flags.
 * @param {unknown} value @returns {Config} */
function jitSdkConfig(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {throw new Error("Invalid compiler SDK configuration");}
  const c = /** @type {Config} */ (value), keys = Object.keys(c), required = ["version", "target", "sources", "sdk", "resources", "tools", "jobs"];
  const toolKeys = c.tools && Object.keys(c.tools).sort().join(",");
  if (c.version !== 1 || !["linux", "windows"].includes(c.target) || required.some(k => !Object.hasOwn(c, k)) || keys.some(k => ![...required, "inputs"].includes(k)) ||
      ![c.sources, c.sdk, c.resources].every(absolute) ||
      !["cmake,ninja", "cmake,ninja,toolchain", "cc,cmake,cxx,ninja", "cc,cmake,cxx,ninja,toolchain"].includes(toolKeys || "") ||
      !Object.values(c.tools).every(literal) || c.target === "windows" && !c.tools.toolchain || !Number.isInteger(c.jobs) || c.jobs < 1 || c.jobs > 16 ||
      c.inputs !== undefined && (!Array.isArray(c.inputs) || c.inputs.length > 16384 || !c.inputs.every(absolute))) {throw new Error("Incomplete compiler SDK inputs, tools, target or bounded jobs");}
  return c;
}

/** Keep Windows support libraries bound to the explicit, already-hashed toolchain inputs.
 * @param {Config} config @param {Record<string,string>} inputs @returns {string[]} */
function jitSupportLibraries(config, inputs) {
  if (config.target !== "windows") {return [];}
  const selected = config.inputs || [];
  if (new Set(selected).size !== selected.length) {throw new Error("Windows compiler SDK inputs contain duplicates");}
  const libraries = selected.filter(file => /\.lib$/i.test(file)).sort();
  if (libraries.some(file => !/^[a-f0-9]{64}$/.test(inputs[file] || ""))) {throw new Error("Windows support library is not bound by the selected input hashes");}
  return libraries;
}

/** Query only the selected compiler binary, and retain its exact bounded identity text.
 * @param {string} file @param {NodeJS.ProcessEnv} env @param {string} label */
function queryCompilerVersion(file, env, label) {
  const result = spawnSync(file, ["--version"], {env, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
  const version = result.stdout || "";
  if (result.status !== 0 || result.error || result.signal || !version.trim() || version.length > (1 << 20) || version.includes("\0")) {
    throw new Error("Could not query the selected " + label + " compiler version", {cause: result.error});
  }
  return version;
}

/** Bind queried versions to the already hashed binaries. Only the SDK bootstrap/embedded
 * compiler has a Clang23 requirement; an explicit host toolchain is identified, not mislabeled.
 * @param {ReturnType<typeof jitSdkPlan>} plan @param {Record<string,string>} inputs @param {Record<string,string>} versions */
function jitCompilerIdentities(plan, inputs, versions) {
  const selection = plan?.compilerIdentity, host = selection?.host;
  if (!host || !["explicit-host-override", "matching-clang23-bootstrap"].includes(host.mode) ||
      !literal(host.cc) || !literal(host.cxx) || !literal(selection.embeddedJit23)) {throw new Error("Missing selected JIT compiler identity plan");}
  /** @param {string} file */
  function identify(file) {
    const sha256 = inputs[file], version = versions[file];
    if (!/^[a-f0-9]{64}$/.test(sha256 || "") || typeof version !== "string" || !version.trim() || version.length > (1 << 20) || version.includes("\0")) {
      throw new Error("Selected JIT compiler identity is incomplete");
    }
    return {path: file, sha256, version};
  }
  const cc = identify(host.cc), cxx = identify(host.cxx), embeddedJit23 = identify(selection.embeddedJit23);
  if (host.mode === "matching-clang23-bootstrap" && [cc, cxx].some(identity => !/clang version 23\./i.test(identity.version))) {
    throw new Error("The explicit baseline path requires the matching Clang23 bootstrap compiler");
  }
  if (!/clang version 23\./i.test(embeddedJit23.version)) {throw new Error("The selected SDK must retain its embedded Clang23 JIT compiler");}
  return {host: {mode: host.mode, cc, cxx}, embeddedJit23};
}

/** Stream potentially large Ninja dependency logs with bounded lines/storage. Never evaluate
 * Make/shell text; reject stale records instead of certifying an incomplete compiler closure.
 * @param {string} file @param {string} base */
async function jitDependencyFiles(file, base) {
  if (!absolute(file) || !absolute(base) || !fs.lstatSync(file).isFile()) {throw new Error("Select a regular compiler dependency log and absolute build root");}
  const handle = await fs.promises.open(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK | fs.constants.O_NOFOLLOW), before = await handle.stat();
  if (!before.isFile() || before.size < 1 || before.size > (256 << 20)) {await handle.close(); throw new Error("Compiler dependency log exceeds bounds");}
  const buffer = Buffer.alloc(64 << 10), decoder = new TextDecoder("utf-8", {fatal: true});
  const files = new Set(); let carry = "", bytes = 0, expected = 0, seen = 0, records = 0;
  function finishRecord() {if (seen !== expected) {throw new Error("Incomplete compiler dependency record");}}
  /** @param {string} line */
  function parse(line) {
    if (line.endsWith("\r")) {line = line.slice(0, -1);}
    if (line.length > 16384 || Array.from(line).some(c => c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127)) {throw new Error("Invalid compiler dependency line");}
    if (!line) {finishRecord(); expected = 0; seen = 0; return;}
    const header = line.match(/^\S.*: #deps ([0-9]+), deps mtime [0-9]+ \(VALID\)$/);
    if (header) {
      finishRecord(); expected = Number(header[1]); seen = 0; records++;
      // MSVC PCH consumers and header-free assembly can have no direct includes. The
      // complete log must still contain actual dependencies (including the PCH's own).
      if (!Number.isInteger(expected) || expected < 0 || expected > 65536) {throw new Error("Invalid compiler dependency count");}
      return;
    }
    if (seen >= expected || !line.startsWith("    ") || !line.slice(4)) {throw new Error("Unrecognized or stale compiler dependencies");}
    seen++;
    files.add(path.resolve(base, line.slice(4)));
    if (files.size > 65536) {throw new Error("Compiler dependency file count exceeds bounds");}
  }
  try {
    for (;;) {
      const {bytesRead} = await handle.read(buffer, 0, buffer.length, null);
      if (!bytesRead) {carry += decoder.decode(); break;}
      const text = decoder.decode(buffer.subarray(0, bytesRead), {stream: true}); bytes += bytesRead;
      if (bytes > (256 << 20)) {throw new Error("Compiler dependency log grew beyond bounds");}
      const lines = (carry + text).split("\n"); carry = lines.pop() || "";
      for (const line of lines) {parse(line);}
      if (carry.length > 16384) {throw new Error("Compiler dependency line exceeds bounds");}
    }
    if (carry) {parse(carry);}
    finishRecord(); const after = await handle.stat();
    if (before.size !== after.size || before.mtimeMs !== after.mtimeMs || before.ctimeMs !== after.ctimeMs || !records || !files.size) {throw new Error("Compiler dependencies changed or are incomplete");}
    return [...files].sort();
  } finally {await handle.close();}
}

/** Capture CMake-discovered build tools before compiling, including Python and archive/link
 * utilities. Generated TableGen executables belong to the output, not an external SDK.
 * @param {string} text @param {string} build */
function jitConfiguredTools(text, build) {
  const names = /^(?:CMAKE_(?:(?:C|CXX)_COMPILER(?:_AR|_RANLIB)?|AR|RANLIB|LINKER|RC_COMPILER|MT|MAKE_PROGRAM)|_?Python\w*_EXECUTABLE|MOM_NODE|LLVM_TABLEGEN|CLANG_TABLEGEN)$/;
  const files = new Set();
  for (const line of text.split(/\r?\n/)) {
    const row = line.match(/^([^:#]+):[^=]+=(.*)$/); if (!row || !names.test(row[1] || "")) {continue;}
    const file = row[2] || "";
    if (file.endsWith("-NOTFOUND") || file === "llvm-tblgen" || file === "clang-tblgen") {continue;}
    if (!absolute(file)) {throw new Error("CMake selected a non-absolute compiler SDK tool");}
    if (!file.startsWith(build + path.sep)) {files.add(fs.realpathSync(file));}
  }
  if (!files.size) {throw new Error("Missing configured compiler SDK tools");}
  return [...files].sort();
}

/** Return the one CMake version directory containing the selected compiler metadata.
 * @param {string} build */
function jitConfiguredCompilerFiles(build) {
  if (!absolute(build)) {throw new Error("Generated CMake compiler metadata needs an absolute build root");}
  const root = path.join(build, "CMakeFiles"), rootStat = fs.lstatSync(root, {throwIfNoEntry: false});
  if (!rootStat?.isDirectory()) {throw new Error("Missing generated CMake compiler metadata directory");}
  const entries = fs.readdirSync(root, {withFileTypes: true});
  if (entries.length > 256) {throw new Error("Generated CMake compiler metadata inventory exceeds bounds");}
  const versions = entries.filter(entry => entry.isDirectory() && /^\d+(?:\.\d+)*$/.test(entry.name));
  if (versions.length !== 1) {throw new Error("Expected exactly one generated CMake compiler version directory");}
  const directory = path.join(root, versions[0]?.name || "");
  const files = {CMAKE_C_COMPILER: path.join(directory, "CMakeCCompiler.cmake"), CMAKE_CXX_COMPILER: path.join(directory, "CMakeCXXCompiler.cmake")};
  if (!Object.values(files).every(file => fs.lstatSync(file, {throwIfNoEntry: false})?.isFile())) {
    throw new Error("Missing generated CMake C/C++ compiler metadata");
  }
  return files;
}

/** Check CMake's selected cache entries before any target compile. A toolchain file may set
 * normal compiler variables that differ from the command-line cache request.
 * @param {string} text @param {{CMAKE_C_COMPILER: string, CMAKE_CXX_COMPILER: string}} expected
 * @param {{CMAKE_C_COMPILER: string, CMAKE_CXX_COMPILER: string}} generated */
function jitConfiguredCompilers(text, expected, generated) {
  if (typeof text !== "string" || text.length > (8 << 20) || !expected ||
      !generated || !literal(expected.CMAKE_C_COMPILER) || !literal(expected.CMAKE_CXX_COMPILER) ||
      ![generated.CMAKE_C_COMPILER, generated.CMAKE_CXX_COMPILER].every(value => typeof value === "string" && value.length <= (8 << 20))) {
    throw new Error("Invalid configured compiler metadata");
  }
  /** @param {keyof typeof expected} name @param {string} selected */
  function selected(name, selected) {
    const rows = text.split(/\r?\n/).filter(line => line.startsWith(name + ":"));
    const match = rows.length === 1 && rows[0]?.match(new RegExp("^" + name + ":[^=]+=(.*)$"));
    if (!match || !literal(match[1]) || match[1] !== selected) {throw new Error("CMake selected a compiler different from the requested host compiler: " + name);}
    const generatedRows = generated[name].split(/\r?\n/).filter(line => line.startsWith("set(" + name + " "));
    const generatedMatch = generatedRows.length === 1 && generatedRows[0]?.match(new RegExp("^set\\(" + name + " \"([^\"\\r\\n]+)\"\\)$"));
    if (!generatedMatch || !literal(generatedMatch[1]) || generatedMatch[1] !== selected) {
      throw new Error("CMake generated a compiler different from the requested host compiler: " + name);
    }
    return match[1];
  }
  return {CMAKE_C_COMPILER: selected("CMAKE_C_COMPILER", expected.CMAKE_C_COMPILER),
    CMAKE_CXX_COMPILER: selected("CMAKE_CXX_COMPILER", expected.CMAKE_CXX_COMPILER)};
}

/** Source/SDK production only. The caller enforces host cgroup/resource limits; no implicit
 * retry, VM, GPU, compiler installation or mutation of existing SDKs is performed.
 * Explicit phases transfer only this invocation's build tree between CI jobs, not a persistent
 * cache. Every handoff rechecks source/tool hashes AND generated outputs before executing Ninja.
 * @param {unknown} value @param {string} output @param {NodeJS.ProcessEnv} [environment]
 * @param {string} [stage] @returns {Promise<unknown>} */
async function buildJitSdk(value, output, environment = process.env, stage = "all") {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Compiler SDK production requires Linux x64 and MOM_AOT_BUILD_ONLY=1");}
  const c = jitSdkConfig(value);
  if (!["all", "configure", ...jitBuildStages].includes(stage)) {throw new Error("Select configure, support, targets, frontend, link or all");}
  if (stage === "all") {
    await buildJitSdk(c, output, environment, "configure");
    for (const phase of jitBuildStages) {await buildJitSdk(c, output, environment, phase);}
    return readBuildJson(path.join(output, "result.json"));
  }
  const initialize = stage === "configure", stat = absolute(output) ? fs.lstatSync(output, {throwIfNoEntry: false}) : null;
  if (!absolute(output) || (initialize ? stat : !stat?.isDirectory())) {
    throw new Error("Select a new absolute compiler SDK output, or its existing real directory for the next phase");
  }
  const selectedOutput = path.join(fs.realpathSync(path.dirname(output)), path.basename(output));
  if (selectedOutput !== output) {throw new Error("Compiler SDK output must have a canonical parent");}
  const beforeFile = path.join(selectedOutput, "before.json"), savedState = initialize ? null : readBuildJson(beforeFile);
  const sourcesFile = fs.realpathSync(c.sources), sources = verifyPreparedSdkSources(sourcesFile, "compiler");
  const resources = verifyJitResources(fs.realpathSync(c.resources), c.target === "windows" ? "x86_64-w64-windows-gnu" : "x86_64-unknown-linux-gnu");
  const roots = Object.fromEntries(Object.entries(sources.prepared).map(([name, source]) => [name, source.root]));
  const sdk = fs.realpathSync(c.sdk), plan = jitSdkPlan(c.target, roots, sdk, resources.overlay, selectedOutput, c.tools, c.jobs);
  const env = nativeSdkEnvironment(environment);
  for (const key of ["CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "INCLUDE", "LIB", "LIBRARY_PATH"]) {delete env[key];}
  const cc = plan.compilerIdentity.host.cc, cxx = plan.compilerIdentity.host.cxx, embeddedJit23 = plan.compilerIdentity.embeddedJit23;
  const leaves = [...new Set([...jitSourceInputs, "scripts/build-native-jit-sdk.js", "scripts/native-jit-sdk.js", "scripts/native-jit-stages.js", "scripts/prepare-native-sdk-sources.js",
    "scripts/prepare-native-sycl-jit-resources.js", "scripts/build-source-inputs.js", "scripts/build-environment.js", "scripts/native-compiler-isolation.js", "scripts/native-audit.js",
    "scripts/prepare-native-jit-toolchain.js", "runtime/driver-library.cpp"])];
  const recipe = sourceInputs(repo, [], leaves);
  const initial = buildFileInputs([sourcesFile, cc, cxx, embeddedJit23, c.tools.cmake, c.tools.ninja, process.execPath, ...Object.keys(resources.inputs),
    ...(c.inputs || []), ...(c.tools.toolchain ? [c.tools.toolchain] : [])]);
  const supportLibraries = jitSupportLibraries(c, initial);
  /** @type {Record<string,string>} */ const versions = {};
  const selectedCompilers = /** @type {[string, string][]} */ ([[cc, "host C"], [cxx, "host C++"], [embeddedJit23, "embedded Clang23 JIT"]]);
  for (const [file, label] of selectedCompilers) {
    versions[file] = queryCompilerVersion(file, env, label);
  }
  const compilerIdentities = jitCompilerIdentities(plan, initial, versions);
  if (!isDeepStrictEqual(initial, buildFileInputs(Object.keys(initial)))) {throw new Error("Selected compiler or SDK inputs changed during identity queries");}
  if (initialize) {fs.mkdirSync(selectedOutput, {mode: 0o700});}
  const state = {version: 1, config: c, plan, recipe, inputs: initial, supportLibraries,
    hostCompilerIdentity: compilerIdentities.host, embeddedJit23Identity: compilerIdentities.embeddedJit23};
  /** @param {string} name @param {unknown} data */
  function write(name, data) {
    const text = JSON.stringify(data, null, 2) + "\n";
    if (Buffer.byteLength(text) > (8 << 20)) {throw new Error("Compiler SDK production metadata exceeds bounds");}
    fs.writeFileSync(path.join(selectedOutput, name), text, {flag: "wx"});
  }
  if (initialize) {write("before.json", state);} else if (!isDeepStrictEqual(savedState, state)) {throw new Error("Compiler SDK source, configuration or tool inputs changed between phases");}
  const beforeSha256 = fileHash(beforeFile), stageFile = "stage-" + stage + ".json";
  if (fs.existsSync(path.join(selectedOutput, stageFile)) || fs.existsSync(path.join(selectedOutput, "result.json"))) {throw new Error("Compiler SDK phase already completed; refusing to overwrite evidence");}
  /** @param {string[]} command @param {string} name @param {number} timeout */
  function run(command, name, timeout) {
    const fd = fs.openSync(path.join(selectedOutput, name + ".log"), "wx");
    try {
      const result = spawnSync(command[0] || "", command.slice(1), {cwd: repo, env, stdio: ["ignore", fd, fd], timeout});
      if (result.status !== 0 || result.error || result.signal) {throw new Error("Compiler SDK stage failed: " + name + "; inspect " + selectedOutput, {cause: result.error});}
    } finally {fs.closeSync(fd);}
  }
  const startedAt = Date.now();
  if (initialize) {
    console.log("[native-sdk] Configuring original static source-JIT compiler"); run(plan.configure, "configure", 300000);
    const cache = readBuildText(path.join(plan.build, "CMakeCache.txt")), compilerFiles = jitConfiguredCompilerFiles(plan.build);
    jitConfiguredCompilers(cache, {CMAKE_C_COMPILER: cc, CMAKE_CXX_COMPILER: cxx}, {
      CMAKE_C_COMPILER: readBuildText(compilerFiles.CMAKE_C_COMPILER), CMAKE_CXX_COMPILER: readBuildText(compilerFiles.CMAKE_CXX_COMPILER)});
    const configuredTools = buildFileInputs([...jitConfiguredTools(cache, plan.build), ...Object.values(compilerFiles)]);
    write("configured-tools.json", configuredTools);
    const graph = spawnSync(c.tools.ninja, ["-C", plan.build, "-t", "commands", "native-dpcpp-jit"], {env, encoding: "utf8", timeout: 30000, maxBuffer: 32 << 20});
    const graphOutput = typeof graph.stdout === "string" ? graph.stdout : "", graphError = typeof graph.stderr === "string" ? graph.stderr : "";
    fs.writeFileSync(path.join(selectedOutput, "commands.log"), graphOutput, {flag: "wx"});
    fs.writeFileSync(path.join(selectedOutput, "commands.stderr.log"), graphError, {flag: "wx"});
    if (graph.status !== 0 || graph.error || graph.signal) {throw new Error("Original compiler archive graph query failed; raw commands.log was preserved");}
    const libraries = compilerArchives(graphOutput, plan.build, c.target === "windows", supportLibraries);
    const result = {version: 1, stage, beforeSha256, configuredSha256: fileHash(path.join(selectedOutput, "configured-tools.json")),
      commandsSha256: fileHash(path.join(selectedOutput, "commands.log")), libraries, supportLibraries, inputs: {...initial, ...configuredTools}, outputs: jitBuildOutputs(plan.build), buildOnly: true};
    if (!isDeepStrictEqual(result.inputs, buildFileInputs(Object.keys(result.inputs)))) {throw new Error("Compiler SDK configure inputs changed");}
    verifyPreparedSdkSources(sourcesFile, "compiler"); verifyJitResources(resources.directory, resources.target);
    verifySourceInputs(recipe, sourceInputs(repo, [], leaves), "Compiler SDK producer");
    write(stageFile, result); return result;
  }
  const previous = jitBuildStages.indexOf(stage) - 1, previousStage = previous < 0 ? "configure" : jitBuildStages[previous];
  const previousFile = path.join(selectedOutput, "stage-" + previousStage + ".json"), prior = readBuildJson(previousFile);
  const configuredTools = readBuildJson(path.join(selectedOutput, "configured-tools.json"));
  if (!Array.isArray(prior?.supportLibraries) || !isDeepStrictEqual(prior.supportLibraries, supportLibraries)) {
    throw new Error("Compiler SDK handoff changed or omitted the admitted support-library list");
  }
  for (const [file, hash] of Object.entries({...initial, ...configuredTools})) {
    if (prior?.inputs?.[file] !== hash) {throw new Error("Compiler SDK handoff omitted or changed a selected tool/input");}
  }
  const libraries = jitBuildHandoff(prior, {stage: previousStage || "", beforeSha256,
    configuredSha256: fileHash(path.join(selectedOutput, "configured-tools.json")), commandsSha256: fileHash(path.join(selectedOutput, "commands.log")),
    inputs: buildFileInputs(Object.keys(prior?.inputs || {})), outputs: jitBuildOutputs(plan.build)}, plan.build);
  if (!isDeepStrictEqual(libraries, compilerArchives(readBuildText(path.join(selectedOutput, "commands.log")), plan.build, c.target === "windows", supportLibraries))) {
    throw new Error("Compiler SDK handoff changed the original archive closure");
  }
  console.log("[native-sdk] Building original compiler phase " + stage);
  run(jitBuildStage(plan, stage), stage, 45 * 60 * 1000);
  run([c.tools.ninja, "-C", plan.build, "-t", "deps"], "dependencies-" + stage, 300000);
  const dependencies = await jitDependencyFiles(path.join(selectedOutput, "dependencies-" + stage + ".log"), plan.build);
  const owned = [...Object.values(roots), selectedOutput].map(root => fs.realpathSync(root) + path.sep), external = new Set(Object.keys(prior.inputs));
  for (const file of dependencies) {
    const real = fs.realpathSync(file);
    if (!owned.some(root => real.startsWith(root))) {external.add(real);}
  }
  const inputs = buildFileInputs([...external]);
  for (const [file, hash] of Object.entries(prior.inputs)) {if (inputs[file] !== hash) {throw new Error("Compiler SDK input changed during compilation: " + file);}}
  for (const file of external) {if (!Object.hasOwn(prior.inputs, file) && fs.statSync(file).ctimeMs > startedAt) {throw new Error("Discovered compiler dependency changed during compilation: " + file);}}
  verifyPreparedSdkSources(sourcesFile, "compiler"); verifyJitResources(resources.directory, resources.target);
  verifySourceInputs(recipe, sourceInputs(repo, [], leaves), "Compiler SDK producer");
  const handoff = {version: 1, stage, beforeSha256, previousSha256: fileHash(previousFile), configuredSha256: prior.configuredSha256,
    commandsSha256: prior.commandsSha256, libraries, supportLibraries, inputs, outputs: jitBuildOutputs(plan.build), buildOnly: true};
  write(stageFile, handoff);
  if (stage !== "link") {return handoff;}
  for (const file of libraries) {if (!isStaticArchive(file)) {throw new Error("Original compiler graph did not produce a static archive: " + file);}}
  const executable = path.join(plan.build, "bin/native-dpcpp-jit" + (c.target === "windows" ? ".exe" : ""));
  const result = {version: 1, kind: "original-source-jit-sdk", target: c.target, sourcesFile, resources: resources.directory, plan, supportLibraries,
    inputs, recipe, libraries: libraries.map(file => ({path: file, sha256: fileHash(file)})), proof: {path: executable, sha256: fileHash(executable)},
    compiled: true, buildOnly: true, executed: false, publicationQualified: false,
    scope: "Original pinned compiler source production; proof linked but not executed. Namespace isolation, runtime, portability, licenses and publication remain separate gates."};
  write("result.json", result);
  console.log("[native-sdk] Built original compiler SDK; not executed or publication-qualified.");
  return result;
}

module.exports = {jitSdkConfig, jitDependencyFiles, jitConfiguredTools, jitConfiguredCompilerFiles, jitConfiguredCompilers, jitCompilerIdentities,
  jitSupportLibraries, buildJitSdk};
if (require.main === module) {
  const [config, output, stage, extra] = process.argv.slice(2);
  Promise.resolve().then(() => {
    if (!config || !output || extra) {throw new Error("Usage: build-native-jit-sdk.js /absolute/config.json /absolute/output [configure|support|targets|frontend|link|all]");}
    return buildJitSdk(readBuildJson(config), output, process.env, stage);
  }).catch(error => {console.error(error instanceof Error ? error.message : "Compiler SDK production failed"); process.exitCode = 1;});
}
