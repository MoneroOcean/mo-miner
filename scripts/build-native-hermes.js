"use strict";

// Build-only SDK production from the same original Static Hermes used by the controller.
// Never modify an Engineering checkout, fetch a moving revision, or execute MoM here.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {fileHash, sourceInputs, verifySourceInputs, readBuildJson, readBuildText} = require("./build-source-inputs");
const {nativeSdkEnvironment} = require("./build-environment");
const {isStaticArchive} = require("./native-audit");
const repo = path.resolve(__dirname, "..");
const hermesPins = Object.freeze({revision: "26fbb82fad39d9d06901adb1d6d630dacc84c877",
  promiseSha256: "071f4322f9dba4fa06992065bc5f1eb9b11e49c9b1b6336d31f377b3e9801184"});
const resumeBaseline = Object.freeze({producer: "872a5c25eeacdd0d3438d7ca2cc4d9bda961f452f5f9269d3b3b42b22a16ae2a",
  sourceInputsOld: "cbe016232d5ca594d97f58657673723802173efbaf42f70e0c5de05aa57441f0",
  sourceInputsCurrent: "74e04fdb0227950aba00b27af02065f02ab700a30bf8b807f0fd1b84dc139330",
  windowsPatch: "b1afebda35a3873d60848853720f944f795559796783684c208c294d54a333d5", changedFile: "lib/VM/CMakeLists.txt"});
/** CMake values must not become lists/interpolation or command-line fragments.
 * @param {unknown} value @returns {value is string} */
const literal = value => typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) && !/[;"$]/.test(value) &&
  !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
/** @typedef {{git: string, cmake: string, cc: string, cxx: string, ninja: string}} HermesTools */

/** One original-source recipe for native Linux and Windows build hosts; Windows uses the
 * static CRT and the already-reviewed clang-cl fix. No target-name/GPU-specific policy.
 * @param {string} input @param {string} output @param {HermesTools} tools
 * @param {number} jobs @param {NodeJS.Platform} [platform] */
function hermesSdkPlan(input, output, tools, jobs, platform = process.platform) {
  if (!["linux", "win32"].includes(platform) || !tools || Object.keys(tools).sort().join(",") !== "cc,cmake,cxx,git,ninja" ||
      ![input, output, ...Object.values(tools)].every(literal) || !Number.isInteger(jobs) || jobs < 1 || jobs > 16) {
    throw new Error("Hermes SDK needs Linux/Windows, explicit absolute source/output/tools and jobs in 1..16");
  }
  for (const [parent, child] of [[input, output], [output, input], [repo, output]]) {
    const relative = path.relative(parent || "", child || "");
    if (!relative || (!relative.startsWith(".." + path.sep) && relative !== ".." && !path.isAbsolute(relative))) {
      throw new Error("Hermes SDK output must not overlap sources or the maintained repository");
    }
  }
  const source = path.join(output, "source"), build = path.join(output, "build"), windows = platform === "win32";
  const cmakePath = (/** @type {string} */ file) => file.replaceAll("\\", "/");
  const configure = [tools.cmake, "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE:STRING=Release",
    "-DCMAKE_C_COMPILER:FILEPATH=" + cmakePath(tools.cc), "-DCMAKE_CXX_COMPILER:FILEPATH=" + cmakePath(tools.cxx),
    "-DCMAKE_MAKE_PROGRAM:FILEPATH=" + cmakePath(tools.ninja),
    "-DBUILD_SHARED_LIBS:BOOL=OFF", "-DHERMES_BUILD_SHARED_JSI:BOOL=OFF", "-DHERMES_ENABLE_DEBUGGER:BOOL=OFF",
    "-DHERMES_ENABLE_INTL:BOOL=OFF", "-DHERMES_ENABLE_NAPI:BOOL=OFF", "-DHERMES_ENABLE_TEST_SUITE:BOOL=OFF", "-DHERMES_ENABLE_WERROR:BOOL=OFF",
    "-DHERMES_ENABLE_TOOLS:BOOL=ON", "-DHERMESVM_INTERNAL_JAVASCRIPT_NATIVE:BOOL=ON", "-DHERMESVM_ALLOW_JIT:STRING=0",
    ...(windows ? ["-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreaded"] : [])];
  const compile = [tools.cmake, "--build", build, "--parallel", String(jobs), "--target", "shermes", "hermesvmlean_a", "hermesPublic", "jsi", "boost_context"];
  const archive = (/** @type {string} */ name) => windows ? name + ".lib" : "lib" + name + ".a";
  return {source, build, configure, compile, patches: ["runtime/hermes-host-rejections.patch", ...(windows ? ["runtime/hermes-clang-cl.patch"] : [])],
    artifacts: {compiler: path.join(build, "bin", windows ? "shermes.exe" : "shermes"), runtime: path.join(build, "lib", archive("hermesvmlean_a")),
      public: path.join(build, "public/hermes/Public", archive("hermesPublic")), jsi: path.join(build, "jsi", archive("jsi")),
      context: path.join(build, "external/boost/boost_1_86_0/libs/context", archive("boost_context"))}};
}

/** Hash the pinned checkout's tracked bytes, not .git or generated build caches. Source
 * preparation uses a fresh local clone, so untracked input files cannot shadow SDK headers.
 * @param {string} source @param {string} names */
function hermesSourceFiles(source, names) {
  const rows = names.split("\0").filter(Boolean), files = /** @type {Record<string, string>} */ ({});
  if (!rows.length || rows.length > 100000 || new Set(rows).size !== rows.length || rows.some(name =>
    path.isAbsolute(name) || name.includes("\\") || Array.from(name).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127) ||
    name.split("/").some(part => !part || part === "." || part === ".."))) {
    throw new Error("Invalid bounded Hermes source inventory");
  }
  let bytes = 0;
  for (const name of rows) {
    const parts = name.split("/");
    let file = source;
    for (const [index, part] of parts.entries()) {
      file = path.join(file, part); const stat = fs.lstatSync(file);
      if (stat.isSymbolicLink() || (index === parts.length - 1 ? !stat.isFile() : !stat.isDirectory())) {
        throw new Error("Hermes sources must be bounded regular files");
      }
    }
    const stat = fs.statSync(file);
    if ((bytes += stat.size) > (1 << 30)) {throw new Error("Hermes sources must be bounded regular files");}
    files[name] = fileHash(file);
  }
  return files;
}

/** Resume accepts unchanged Linux sources or the scoped Windows DateUtil source delta. */
function exactKeys(/** @type {unknown} */ value, /** @type {string[]} */ keys, /** @type {string} */ label) {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).sort().join("\0") !== [...keys].sort().join("\0")) {throw new Error("Hermes resume " + label + " has unknown or missing fields");}
}
function samePath(/** @type {unknown} */ actual, /** @type {string} */ expected, /** @type {NodeJS.Platform} */ platform) {
  if (typeof actual !== "string" || !path.isAbsolute(actual)) {return false;} return platform === "win32" ? path.win32.normalize(actual).toLowerCase() === path.win32.normalize(expected).toLowerCase() : path.resolve(actual) === path.resolve(expected);
}
function validateHermesCMakeCache(/** @type {ReturnType<typeof hermesSdkPlan>} */ plan, /** @type {NodeJS.Platform} */ platform = process.platform) {
  const values = new Map();
  for (const line of readBuildText(path.join(plan.build, "CMakeCache.txt"), 1 << 20).split(/\r?\n/)) {
    const match = line.match(/^([^/#][^:]*):[^=]+=([\s\S]*)$/);
    if (match) {if (values.has(match[1])) {throw new Error("Duplicate Hermes CMake cache entry");} values.set(match[1], match[2]);}
  }
  const expected = new Map([["CMAKE_HOME_DIRECTORY", plan.source], ["CMAKE_CACHEFILE_DIR", plan.build], ["CMAKE_GENERATOR", "Ninja"]]);
  for (const arg of plan.configure) {
    const match = arg.match(/^-D([^:=]+)(?::[^=]+)?=(.*)$/);
    if (match) {expected.set(match[1] || "", match[2] || "");}
  }
  const paths = new Set(["CMAKE_HOME_DIRECTORY", "CMAKE_CACHEFILE_DIR", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_MAKE_PROGRAM"]);
  const normalized = (/** @type {string} */ value) => platform === "win32" ? path.win32.normalize(value.replaceAll("/", "\\")).toLowerCase() : path.resolve(value);
  for (const [key, value] of expected) {
    const actual = values.get(key);
    if (actual === undefined) {throw new Error("Hermes CMake cache differs from its recorded plan: " + key);}
    if (paths.has(key) ? normalized(actual) !== normalized(value) : actual !== value) {
      throw new Error("Hermes CMake cache differs from its recorded plan: " + key);
    }
  }
}
function validateHermesResumeMetadata(/** @type {any} */ result, /** @type {any} */ before,
  /** @type {ReturnType<typeof hermesSdkPlan>} */ plan, /** @type {{files: Record<string, string>, graph: Record<string, string>}} */ recipe,
  /** @type {Record<string, string>} */ fingerprints, /** @type {NodeJS.Platform} */ platform) {
  exactKeys(result, ["version", "target", "pins", "source", "build", "artifacts", "fingerprints", "sources", "recipe", "buildOnly", "publicationQualified"], "result");
  exactKeys(before, ["pins", "recipe", "fingerprints", "sources", "configure", "compile"], "before receipt");
  if (result.version !== 1 || result.target !== platform || !isDeepStrictEqual(result.pins, hermesPins) || result.buildOnly !== true || result.publicationQualified !== false ||
      !samePath(result.source, plan.source, platform) || !samePath(result.build, plan.build, platform) ||
      !isDeepStrictEqual(before.pins, hermesPins) || !isDeepStrictEqual(before.configure, plan.configure) || !isDeepStrictEqual(before.compile, plan.compile) ||
      !isDeepStrictEqual(before.sources, result.sources) || !isDeepStrictEqual(before.recipe, result.recipe) || !isDeepStrictEqual(before.fingerprints, result.fingerprints)) {
    throw new Error("Hermes resume receipt, source/build paths, pins or CMake plan differ");
  }
  exactKeys(result.recipe, ["files", "graph"], "recipe");
  if (Object.keys(result.recipe.files).length) {throw new Error("Hermes resume recipe contains unexpected source roots");}
  exactKeys(result.recipe.graph, Object.keys(recipe.graph), "recipe graph");
  const graphDelta = platform === "win32" ? new Set(["scripts/build-native-hermes.js", "scripts/build-source-inputs.js", "runtime/hermes-clang-cl.patch"])
    : new Set(["scripts/build-native-hermes.js"]);
  for (const [name, hash] of Object.entries(result.recipe.graph)) {
    if (graphDelta.has(name)) {
      const expected = name === "scripts/build-native-hermes.js" ? resumeBaseline.producer : name === "scripts/build-source-inputs.js" ? resumeBaseline.sourceInputsOld : resumeBaseline.windowsPatch;
      const current = name === "scripts/build-source-inputs.js" ? resumeBaseline.sourceInputsCurrent : recipe.graph[name];
      if (hash !== expected || recipe.graph[name] !== current || hash === current) {throw new Error("Hermes resume producer/recipe baseline is not recognized");}
    } else if (hash !== recipe.graph[name]) {throw new Error("Hermes resume recipe changed: " + name);}
  }
  exactKeys(result.fingerprints, Object.keys(fingerprints), "tool fingerprints");
  const patchPath = path.join(repo, "runtime/hermes-clang-cl.patch");
  for (const [file, hash] of Object.entries(result.fingerprints)) {
    const expected = platform === "win32" && file === patchPath ? resumeBaseline.windowsPatch : fingerprints[file];
    if (!/^[a-f0-9]{64}$/.test(hash) || hash !== expected) {
      throw new Error("Hermes resume tool/patch fingerprint changed: " + file);
    }
  }
  exactKeys(result.artifacts, Object.keys(plan.artifacts), "artifacts");
  for (const [name, file] of Object.entries(plan.artifacts)) {
    exactKeys(result.artifacts[name], ["path", "sha256", "bytes"], "artifact");
    const artifact = result.artifacts[name];
    if (!samePath(artifact.path, file, platform) || !/^[a-f0-9]{64}$/.test(artifact.sha256) || !Number.isSafeInteger(artifact.bytes) || artifact.bytes < 1) {
      throw new Error("Hermes resume artifact metadata differs: " + name);
    }
  }
}
function validateHermesResumeCache(/** @type {string} */ output, /** @type {ReturnType<typeof hermesSdkPlan>} */ plan,
  /** @type {{files: Record<string, string>, graph: Record<string, string>}} */ recipe, /** @type {Record<string, string>} */ fingerprints,
  /** @type {(args: string[]) => string} */ git, /** @type {NodeJS.Platform} */ platform = process.platform) {
  const directory = (/** @type {string} */ file) => {const stat = fs.lstatSync(file); if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("Unsafe Hermes resume directory: " + file);}};
  const regular = (/** @type {string} */ file) => {const stat = fs.lstatSync(file); if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Unsafe Hermes resume file: " + file);} return stat;};
  directory(output);
  if (fs.readdirSync(output).sort().join("\0") !== ["before.json", "build", "logs", "result.json", "source"].sort().join("\0")) {throw new Error("Hermes resume output contains unknown paths or prior generations");}
  for (const name of ["source", "build", "logs"]) {directory(path.join(output, name));}
  const resultPath = path.join(output, "result.json"), beforePath = path.join(output, "before.json"), cacheFile = path.join(plan.build, "CMakeCache.txt");
  for (const file of [resultPath, beforePath, cacheFile, path.join(plan.build, "build.ninja")]) {regular(file);}
  const result = readBuildJson(resultPath), before = readBuildJson(beforePath);
  validateHermesResumeMetadata(result, before, plan, recipe, fingerprints, platform);
  for (const [name, file] of Object.entries(plan.artifacts)) {
    const stat = regular(file), artifact = result.artifacts[name];
    if (stat.size !== artifact.bytes || fileHash(file) !== artifact.sha256 || (name !== "compiler" && !isStaticArchive(file))) {
      throw new Error("Hermes resume artifact bytes changed: " + name);
    }
  }
  validateHermesCMakeCache(plan, platform);
  const names = git(["-C", plan.source, "ls-files", "-z"]);
  // Without exclusion rules, --others includes both ignored and ordinary untracked files.
  if (git(["-C", plan.source, "ls-files", "--others", "-z"])) {throw new Error("Hermes resume source contains unknown files");}
  if (fs.existsSync(path.join(plan.source, ".git/objects/info/alternates"))) {throw new Error("Hermes resume source borrows mutable Git objects");}
  verifySourceInputs({files: result.sources, graph: {}}, {files: hermesSourceFiles(plan.source, names), graph: {}}, "Hermes resume source");
  return {resultSha256: fileHash(resultPath), result, sources: result.sources, names};
}
function copyHermesSourceChanges(/** @type {string} */ destination, /** @type {string} */ candidate,
  /** @type {Record<string, string>} */ previous, /** @type {Record<string, string>} */ next, /** @type {NodeJS.Platform} */ platform = process.platform) {
  const names = Object.keys(previous).sort();
  if (!isDeepStrictEqual(names, Object.keys(next).sort())) {throw new Error("Hermes resume tracked-source membership changed");}
  const changed = names.filter(name => previous[name] !== next[name]);
  if (platform === "linux" ? changed.length !== 0 : platform !== "win32" || changed.length !== 1 || changed[0] !== resumeBaseline.changedFile) {
    throw new Error(platform === "linux" ? "Linux Hermes resume requires unchanged sources" : "Hermes resume source delta is not the DateUtil policy only");
  }
  verifySourceInputs({files: previous, graph: {}}, {files: hermesSourceFiles(destination, names.join("\0")), graph: {}}, "Hermes resume source before update");
  verifySourceInputs({files: next, graph: {}}, {files: hermesSourceFiles(candidate, names.join("\0")), graph: {}}, "Hermes resume candidate source");
  if (platform === "win32") {for (const name of changed) {fs.copyFileSync(path.join(candidate, name), path.join(destination, name));}}
  verifySourceInputs({files: next, graph: {}}, {files: hermesSourceFiles(destination, names.join("\0")), graph: {}}, "Hermes resume source after update");
  return changed;
}

/** Compile only SDK tools/libraries, retaining the original pinned Git identity for the
 * existing source consumer. Successful receipts do not imply MoM/runtime qualification.
 * @param {string} input @param {string} output @param {HermesTools} tools
 * @param {number} jobs @param {NodeJS.ProcessEnv} [environment] @param {{resume?: boolean}} [options] */
function buildNativeHermes(input, output, tools, jobs, environment = process.env, options = {}) {
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Hermes SDK production requires MOM_AOT_BUILD_ONLY=1; no execution is implicit");}
  if (!options || Object.keys(options).some(key => key !== "resume") || (options.resume !== undefined && typeof options.resume !== "boolean")) {
    throw new Error("Invalid Hermes SDK producer options");
  }
  const resume = options.resume === true;
  if (resume && !["linux", "win32"].includes(process.platform)) {throw new Error("Hermes cache resume supports only Linux unchanged sources or the validated Windows DateUtil change");}
  const plan = hermesSdkPlan(input, output, tools, jobs), env = nativeSdkEnvironment(environment);
  if (!resume && fs.lstatSync(output, {throwIfNoEntry: false})) {throw new Error("Hermes SDK output must be new");}
  if (resume && !fs.lstatSync(output, {throwIfNoEntry: false})?.isDirectory()) {throw new Error("Hermes resume needs an existing writable cache directory");}
  const parent = fs.realpathSync(path.dirname(output)), source = fs.realpathSync(input);
  hermesSdkPlan(source, path.join(parent, path.basename(output)), tools, jobs);
  const recipe = sourceInputs(repo, [], ["scripts/build-native-hermes.js", "scripts/build-source-inputs.js", "scripts/build-environment.js",
    "scripts/native-audit.js", ...plan.patches]);
  const fingerprints = Object.fromEntries([...Object.values(tools), ...plan.patches.map(file => path.join(repo, file))].map(file => [file, fileHash(file)]));
  /** @param {string[]} args */
  const git = args => {
    const result = spawnSync(tools.git, args, {encoding: "utf8", env, timeout: 60000, maxBuffer: 8 << 20});
    if (result.status !== 0 || result.signal || result.error) {throw new Error("Pinned Hermes Git admission failed", {cause: result.error});}
    return result.stdout;
  };
  if (git(["-C", source, "rev-parse", "--verify", hermesPins.revision + "^{commit}"]).trim() !== hermesPins.revision ||
      git(["-C", source, "ls-tree", "-r", "--format=%(objectmode)", hermesPins.revision]).split(/\r?\n/).filter(Boolean).some(mode => !["100644", "100755"].includes(mode))) {
    throw new Error("Hermes source revision or regular-file tree differs from the pinned SDK");
  }
  if (!resume) {fs.mkdirSync(output, {mode: 0o700}); fs.mkdirSync(path.join(output, "logs"));}
  /** @param {string[]} command @param {string} label */
  const run = (command, label, logs = path.join(output, "logs")) => {
    const program = command[0]; if (!program) {throw new Error("Missing SDK stage command");}
    fs.writeFileSync(path.join(logs, label + ".json"), JSON.stringify(command) + "\n", {flag: "wx"});
    const log = fs.openSync(path.join(logs, label + ".log"), "wx");
    try {
      const result = spawnSync(program, command.slice(1), {cwd: output, env, stdio: ["ignore", log, log], timeout: 45 * 60 * 1000});
      if (result.status !== 0 || result.signal || result.error) {throw new Error("Hermes SDK stage failed: " + label + "; inspect " + output, {cause: result.error});}
    } finally {fs.closeSync(log);}
  };
  const prepareSource = (/** @type {string} */ target, /** @type {string} */ prefix, /** @type {string} */ logs) => {
    run([tools.git, "clone", "--no-checkout", "--no-hardlinks", "--dissociate", "--template=", "--", source, target], prefix + "clone", logs);
    run([tools.git, "-C", target, "checkout", "--detach", hermesPins.revision], prefix + "checkout", logs);
    if (fs.existsSync(path.join(target, ".git/objects/info/alternates"))) {throw new Error("Hermes SDK must not borrow mutable source-repository objects");}
    for (const [index, file] of plan.patches.entries()) {
      run([tools.git, "-C", target, "apply", "--check", path.join(repo, file)], prefix + "patch-check-" + index, logs);
      run([tools.git, "-C", target, "apply", path.join(repo, file)], prefix + "patch-" + index, logs);
    }
    if (fileHash(path.join(target, "lib/InternalJavaScript/01-Promise.js")) !== hermesPins.promiseSha256) {throw new Error("Hermes host-rejection patch differs from the shared controller SDK");}
    const names = git(["-C", target, "ls-files", "-z"]);
    if (git(["-C", target, "ls-files", "--others", "-z"])) {throw new Error("Hermes source contains unknown files");}
    return hermesSourceFiles(target, names);
  };
  let sources, resumeEvidence, candidateRoot;
  try {
    if (resume) {
      candidateRoot = fs.mkdtempSync(path.join(parent, ".hermes-resume-candidate-"));
      const candidateLogs = path.join(candidateRoot, "logs"), candidateSource = path.join(candidateRoot, "source");
      fs.mkdirSync(candidateLogs);
      const candidate = prepareSource(candidateSource, "candidate-", candidateLogs);
      const admitted = validateHermesResumeCache(output, plan, recipe, fingerprints, git);
      const linuxChanges = process.platform === "linux" ? copyHermesSourceChanges(plan.source, candidateSource, admitted.sources, candidate, "linux") : null;
      const history = path.join(output, "history"); fs.mkdirSync(history);
      for (const name of ["before.json", "result.json", "logs"]) {fs.renameSync(path.join(output, name), path.join(history, name));}
      fs.mkdirSync(path.join(output, "logs"));
      for (const name of fs.readdirSync(candidateLogs)) {fs.renameSync(path.join(candidateLogs, name), path.join(output, "logs", name));}
      const changed = linuxChanges ?? copyHermesSourceChanges(plan.source, candidateSource, admitted.sources, candidate);
      sources = candidate;
      resumeEvidence = {previousResultSha256: admitted.resultSha256, changedTrackedFiles: changed, cacheIsWorkingState: true};
      console.log(`[native-sdk] Resuming a validated ${process.platform === "linux" ? "Linux unchanged-source" : "Windows DateUtil"} Hermes Ninja cache; this directory is mutable build state, not an immutable SDK.`);
    } else {
      sources = prepareSource(plan.source, "", path.join(output, "logs"));
    }
  } finally {
    if (candidateRoot) {fs.rmSync(candidateRoot, {recursive: true, force: true});}
  }
  const before = {pins: hermesPins, recipe, fingerprints, sources, configure: plan.configure, compile: plan.compile,
    ...(resumeEvidence ? {resume: resumeEvidence} : {})};
  fs.writeFileSync(path.join(output, "before.json"), JSON.stringify(before, null, 2) + "\n", {flag: "wx"});
  console.log("[native-sdk] Configuring original Static Hermes"); run(plan.configure, "configure");
  console.log("[native-sdk] Building static runtime and source AOT compiler"); run(plan.compile, "compile");
  if (resume) {validateHermesCMakeCache(plan);}
  const cache = fs.readFileSync(path.join(plan.build, "CMakeCache.txt"), "utf8");
  for (const line of ["HERMESVM_INTERNAL_JAVASCRIPT_NATIVE:BOOL=ON", "HERMESVM_ALLOW_JIT:STRING=0", "HERMES_ENABLE_NAPI:BOOL=OFF", "HERMES_BUILD_SHARED_JSI:BOOL=OFF", "CMAKE_BUILD_TYPE:STRING=Release"]) {
    if (!cache.split(/\r?\n/).includes(line)) {throw new Error("Hermes native/static configuration drift: " + line);}
  }
  const artifacts = Object.fromEntries(Object.entries(plan.artifacts).map(([name, file]) => {
    if (name !== "compiler" && !isStaticArchive(file)) {throw new Error("Hermes runtime component is not a static archive: " + name);}
    return [name, {path: file, sha256: fileHash(file), bytes: fs.statSync(file).size}];
  }));
  for (const [file, sha] of Object.entries(fingerprints)) {if (fileHash(file) !== sha) {throw new Error("Hermes SDK tool/patch changed during compilation: " + file);}}
  verifySourceInputs({files: sources, graph: {}}, {files: hermesSourceFiles(plan.source, git(["-C", plan.source, "ls-files", "-z"])), graph: {}}, "Hermes SDK");
  verifySourceInputs(recipe, sourceInputs(repo, [], Object.keys(recipe.graph)), "Hermes SDK producer");
  const result = {version: 1, target: process.platform, pins: hermesPins, source: plan.source, build: plan.build,
    artifacts, fingerprints, sources, recipe, buildOnly: true, publicationQualified: false, ...(resumeEvidence ? {resume: resumeEvidence} : {})};
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  console.log("[native-sdk] Built original static Hermes SDK; no MoM/VM/GPU execution or publication qualification.");
  return result;
}

module.exports = {hermesPins, hermesSdkPlan, hermesSourceFiles, validateHermesCMakeCache, validateHermesResumeMetadata,
  validateHermesResumeCache, copyHermesSourceChanges, hermesResumeBaseline: resumeBaseline, buildNativeHermes};
if (require.main === module) {
  try {
    const [source, output, git, cmake, cc, cxx, ninja, jobs = "2", extra] = process.argv.slice(2);
    if (!source || !output || !git || !cmake || !cc || !cxx || !ninja || extra) {throw new Error("Usage: build-native-hermes.js /source /new/sdk /git /cmake /cc /cxx /ninja [jobs]");}
    buildNativeHermes(source, output, {git, cmake, cc, cxx, ninja}, Number(jobs));
  } catch (error) {console.error(error instanceof Error ? error.message : error); process.exitCode = 1;}
}
