"use strict";

// Local Windows-only native support producers. Source acquisition stays with the caller;
// this adapter admits pinned inputs and reuses only outputs whose public receipts still match.
const fs = require("node:fs"), path = require("node:path"), {createHash, randomUUID} = require("node:crypto"),
  {spawnSync} = require("node:child_process"), {isDeepStrictEqual} = require("node:util");
const {fileHash, sourceInputs, readBuildJson, verifySourceInputs} = require("./build-source-inputs");
const io = require("./build-native-io"), openssl = require("./build-native-openssl"), hermes = require("./build-native-hermes"),
  core = require("./build-native-windows-core"), nativeCi = require("./prepare-native-ci"), {isStaticArchive} = require("./native-audit");
const repo = fs.realpathSync(path.resolve(__dirname, "..")), markerName = ".mom-native-windows-support.json", jobs = 2,
  runtimeCatalogPath = path.join(repo, "runtime/windows-intel-runtime.json"), runtimeCatalog = require("../runtime/windows-intel-runtime.json");

/** @typedef {{core:{nodeGyp:string,msbuild:string,dumpbin:string,python:string,compiler:string},
 * io:{cmake:string,cc:string,ninja:string},
 * openssl:{ar:string,cc:string,link:string,make:string,perl:string,rc:string,tar:string},
 * hermes:{git:string,cmake:string,cc:string,cxx:string,ninja:string}}} WindowsSupportTools */
/** @typedef {{workspace:string, sources:{libuv:string,zlib:string,openssl:string,hermes:string}, tools:WindowsSupportTools}} WindowsSupportOptions */

/** @param {string} file @param {string} label */
function regularFile(file, label) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!path.isAbsolute(file) || !stat?.isFile() || stat.isSymbolicLink() || !samePath(fs.realpathSync(file), file)) {
    throw new Error(label + " must be an absolute canonical regular file: " + file);
  }
  return file;
}
/** @param {string} left @param {string} right @param {NodeJS.Platform} [platform] */
function samePath(left, right, platform = process.platform) {return platform === "win32" ? left.toLowerCase() === right.toLowerCase() : left === right;}
/** @param {string} a @param {string} b */
function overlaps(a, b) {
  /** @param {string} parent @param {string} child */
  const contains = (parent, child) => {const relative = path.relative(parent, child); return !relative || (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep));};
  return contains(a, b) || contains(b, a);
}
/** @param {unknown} value @param {string[]} keys @param {string} label */
function exactObject(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).sort().join(",") !== [...keys].sort().join(",")) {
    throw new Error(label + " has unknown or missing fields");
  }
  return /** @type {Record<string, string>} */ (value);
}
/** @param {string} workspace @param {NodeJS.Platform} targetPlatform */
function windowsSupportPaths(workspace, targetPlatform = process.platform) {
  if (targetPlatform !== "win32" || !path.isAbsolute(workspace) || path.resolve(workspace) !== workspace || /[\r\n\0]/.test(workspace) ||
      overlaps(workspace, repo)) {throw new Error("Windows support needs an explicit canonical native workspace outside the checkout");}
  return {workspace, marker: path.join(workspace, markerName), artifacts: path.join(workspace, "artifacts"),
    io: path.join(workspace, "artifacts", "io"), openssl: path.join(workspace, "artifacts", "openssl"),
    hermes: path.join(workspace, "artifacts", "hermes"), coreRuns: path.join(workspace, "core-runs"), descriptors: path.join(workspace, "descriptors")};
}

/** Exact CI role tables are kept explicit; the caller resolves VS/oneAPI tools before admission. */
/** @param {WindowsSupportTools} tools @param {NodeJS.Platform} targetPlatform */
function validateWindowsSupportTools(tools, targetPlatform) {
  if (!tools || typeof tools !== "object" || Object.keys(tools).sort().join(",") !== "core,hermes,io,openssl") {
    throw new Error("Windows support needs explicit Core, I/O, OpenSSL and Hermes tool tables");
  }
  const coreTools = /** @type {WindowsSupportTools["core"]} */ (exactObject(tools.core, ["nodeGyp", "msbuild", "dumpbin", "python", "compiler"], "Windows Core tools")),
    ioTools = /** @type {WindowsSupportTools["io"]} */ (exactObject(tools.io, ["cmake", "cc", "ninja"], "Windows I/O tools")),
    sslTools = /** @type {WindowsSupportTools["openssl"]} */ (exactObject(tools.openssl, ["ar", "cc", "link", "make", "perl", "rc", "tar"], "Windows OpenSSL tools")),
    hermesTools = /** @type {WindowsSupportTools["hermes"]} */ (exactObject(tools.hermes, ["git", "cmake", "cc", "cxx", "ninja"], "Windows Hermes tools"));
  for (const [name, table] of Object.entries({core: coreTools, io: ioTools, openssl: sslTools, hermes: hermesTools})) {
    for (const [key, file] of Object.entries(table)) {regularFile(file, "Windows " + name + " " + key);}
  }
  if ([coreTools.compiler, ioTools.cc, sslTools.cc, hermesTools.cc, hermesTools.cxx].some(file => !samePath(file, ioTools.cc, targetPlatform))) {
    throw new Error("Windows support producers must use the one selected Intel 2026.2 compiler path");
  }
  if (!/^icx-cl\.exe$/i.test(path.basename(ioTools.cc))) {throw new Error("Windows support requires the selected Intel icx-cl.exe frontend");}
  if (targetPlatform !== "win32") {throw new Error("Windows support tools are valid only on a Windows host");}
  return {core: {nodeGyp: coreTools.nodeGyp, msbuild: coreTools.msbuild, dumpbin: coreTools.dumpbin, python: coreTools.python, compiler: coreTools.compiler},
    io: {cmake: ioTools.cmake, cc: ioTools.cc, ninja: ioTools.ninja},
    openssl: {ar: sslTools.ar, cc: sslTools.cc, link: sslTools.link, make: sslTools.make, perl: sslTools.perl, rc: sslTools.rc, tar: sslTools.tar},
    hermes: {git: hermesTools.git, cmake: hermesTools.cmake, cc: hermesTools.cc, cxx: hermesTools.cxx, ninja: hermesTools.ninja}};
}

/** Read-only input/recipe/tool preflight and deterministic output identities. @param {WindowsSupportOptions} options
 * @param {{platform?:NodeJS.Platform,arch?:string,hash?:typeof fileHash,gitProbe?:(git:string,source:string)=>void,
 * compilerProbe?:(file:string)=>void,runtimeEntries?:(compiler:string)=>ReturnType<typeof nativeCi.windowsControllerIntelRuntimeEntries>}} [dependencies] */
function windowsSupportPlan(options, dependencies = {}) {
  const targetPlatform = dependencies.platform || process.platform, targetArch = dependencies.arch || process.arch,
    paths = windowsSupportPaths(options?.workspace || "", targetPlatform), hash = dependencies.hash || fileHash;
  if (targetArch !== "x64" || !options?.sources || Object.keys(options.sources).sort().join(",") !== "hermes,libuv,openssl,zlib") {
    throw new Error("Windows support requires x64 plus exact pinned source paths");
  }
  const tools = validateWindowsSupportTools(options.tools, targetPlatform), sources = {
    libuv: regularFile(options.sources.libuv, "Pinned libuv archive"), zlib: regularFile(options.sources.zlib, "Pinned zlib archive"),
    openssl: regularFile(options.sources.openssl, "Pinned OpenSSL archive"), hermes: options.sources.hermes,
  };
  if (!path.isAbsolute(sources.hermes) || /[\r\n\0]/.test(sources.hermes) || !fs.lstatSync(sources.hermes, {throwIfNoEntry: false})?.isDirectory() ||
      fs.lstatSync(sources.hermes).isSymbolicLink() || !samePath(fs.realpathSync(sources.hermes), sources.hermes)) {
    throw new Error("Pinned Hermes source must be a canonical bare Git directory");
  }
  const hashes = {libuv: hash(sources.libuv), zlib: hash(sources.zlib), openssl: hash(sources.openssl)};
  if (hashes.libuv !== io.ioPins.libuv.sha256 || hashes.zlib !== io.ioPins.zlib.sha256 || hashes.openssl !== openssl.opensslPin.sha256) {
    throw new Error("Windows support source archives differ from the pinned original releases");
  }
  const catalogFile = readBuildJson(runtimeCatalogPath), catalogRecipe = sourceInputs(repo, [], ["runtime/windows-intel-runtime.json"]);
  if (!isDeepStrictEqual(catalogFile, runtimeCatalog) || runtimeCatalog.compiler?.version !== "2026.2" ||
      hash(tools.io.cc) !== runtimeCatalog.compiler.sha256) {throw new Error("Intel compiler differs from the current pinned Windows runtime catalog");}
  (dependencies.compilerProbe || probeIntelCompiler)(tools.io.cc);
  const runtimeEntries = (dependencies.runtimeEntries || nativeCi.windowsControllerIntelRuntimeEntries)(tools.io.cc);
  if (!runtimeEntries.length) {throw new Error("Pinned Intel 2026.2 runtime catalog is empty");}
  for (const entry of runtimeEntries) {
    regularFile(entry.source, "Pinned Intel runtime " + entry.path);
    const stat = fs.statSync(entry.source);
    if (stat.size !== entry.bytes || hash(entry.source) !== entry.sha256 ||
        entry.kind === "library" && !isStaticArchive(entry.source)) {throw new Error("Intel runtime input differs from its selected pin: " + entry.path);}
  }
  const probe = dependencies.gitProbe || probeHermesGit;
  probe(tools.hermes.git, sources.hermes);
  const ioRecipe = sourceInputs(repo, [], ["scripts/build-native-io.js", "scripts/build-environment.js", "scripts/build-source-inputs.js", "scripts/native-audit.js"]),
    opensslRecipe = openssl.opensslBuildRecipe(), hermesPlan = hermes.hermesSdkPlan(sources.hermes, path.join(paths.hermes, "pending"), tools.hermes, jobs, "win32"),
    hermesRecipe = sourceInputs(repo, [], ["scripts/build-native-hermes.js", "scripts/build-source-inputs.js", "scripts/build-environment.js",
      "scripts/native-audit.js", ...hermesPlan.patches]);
  /** @param {unknown} values @param {unknown} recipe @param {unknown} inputs */
  const fingerprint = (values, recipe, inputs) => createHash("sha256").update(JSON.stringify({version: 1, inputs, tools: values, recipe})).digest("hex"),
    ioTools = Object.fromEntries(Object.values(tools.io).map(file => [file, hash(file)])),
    sslTools = Object.fromEntries(Object.entries(tools.openssl).map(([name, file]) => [name, {path: file, sha256: hash(file)}])),
    hermesTools = Object.fromEntries([...Object.values(tools.hermes), ...hermesPlan.patches.map(file => path.join(repo, file))].map(file => [file, hash(file)])),
    ioKey = fingerprint(ioTools, ioRecipe, {jobs, libuv: {path: sources.libuv, sha256: hashes.libuv}, zlib: {path: sources.zlib, sha256: hashes.zlib}}),
    sslConfig = {version: 1, jobs, tools: tools.openssl},
    sslKey = fingerprint(sslTools, opensslRecipe, {archive: {path: sources.openssl, sha256: hashes.openssl}, config: sslConfig}),
    hermesKey = fingerprint(hermesTools, hermesRecipe, {jobs, path: sources.hermes, revision: hermes.hermesPins.revision,
      runtimeCatalog: {path: runtimeCatalogPath, sha256: hash(runtimeCatalogPath)},
      runtime: runtimeEntries.map(entry => ({kind: entry.kind, source: entry.source, selected: entry.path, bytes: entry.bytes, sha256: entry.sha256,
        ...(entry.kind === "library" ? {name: entry.name} : {})}))});
  return {paths, sources, hashes, tools, recipes: {io: ioRecipe, openssl: opensslRecipe, hermes: hermesRecipe},
    io: {output: path.join(paths.io, ioKey), key: ioKey, tools: ioTools},
    openssl: {output: path.join(paths.openssl, sslKey), key: sslKey, tools: sslTools, config: sslConfig},
    hermes: {output: path.join(paths.hermes, hermesKey), key: hermesKey, tools: hermesTools, runtimeEntries, runtimeCatalog: catalogRecipe},
    selected: {io: ["include", "lib", "result.json"], openssl: ["include", "lib/libcrypto.lib", "lib/libssl.lib", "source/LICENSE.txt", "result.json"],
      hermes: ["source/LICENSE", "source/include", "source/API", "source/public", "build/lib/config", "build/lib/hermesvmlean_a.lib",
        "build/jsi/jsi.lib", "build/external/boost/boost_1_86_0/libs/context/boost_context.lib", "before.json", "result.json"]}};
}

/** @param {string} git @param {string} source */
function probeHermesGit(git, source) {
  /** @param {string[]} args */
  const run = args => {
    const result = spawnSync(git, args, {encoding: "utf8", timeout: 60000, maxBuffer: 8 << 20, windowsHide: true});
    if (result.error || result.signal || result.status !== 0) {throw new Error("Pinned Hermes Git preflight failed", {cause: result.error});}
    return result.stdout || "";
  };
  if (run(["-C", source, "remote", "get-url", "origin"]).trim() !== "https://github.com/facebook/hermes.git" ||
      run(["-C", source, "rev-parse", "--verify", hermes.hermesPins.revision + "^{commit}"]).trim() !== hermes.hermesPins.revision ||
      run(["-C", source, "ls-tree", "-r", "--format=%(objectmode)", hermes.hermesPins.revision]).split(/\r?\n/).filter(Boolean).some(mode => !["100644", "100755"].includes(mode))) {
    throw new Error("Hermes source is not the pinned regular-file Git revision");
  }
}

/** Reuse Core's Intel frontend/target parser and require the pinned compiler family. @param {string} file */
function probeIntelCompiler(file) {
  const result = spawnSync(file, ["--version"], {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20, windowsHide: true}),
    output = [result.stdout, result.stderr].filter(value => typeof value === "string").join("");
  if (result.error || result.signal || result.status !== 0) {throw new Error("Pinned Intel Windows compiler preflight failed", {cause: result.error});}
  const identity = core.intelCoreCompilerVersion(output);
  if (!/^2026\.2(?:\.|$)/.test(identity.version)) {throw new Error("Windows support requires Intel oneAPI compiler 2026.2");}
}

/** Create only a marked workspace under an existing canonical parent; never adopt arbitrary data.
 * @param {string} workspace @param {NodeJS.Platform} [targetPlatform] */
function admitWindowsSupportWorkspace(workspace, targetPlatform = process.platform) {
  const paths = windowsSupportPaths(workspace, targetPlatform), parent = path.dirname(workspace);
  if (!samePath(fs.realpathSync(parent), parent, targetPlatform)) {throw new Error("Windows support workspace parent must be canonical");}
  const existing = fs.lstatSync(workspace, {throwIfNoEntry: false});
  if (!existing) {
    fs.mkdirSync(workspace, {mode: 0o700});
    fs.writeFileSync(paths.marker, JSON.stringify({version: 1, checkout: repo}) + "\n", {flag: "wx", mode: 0o600});
  } else {
    const marker = fs.lstatSync(paths.marker, {throwIfNoEntry: false});
    if (!existing.isDirectory() || existing.isSymbolicLink() || !samePath(fs.realpathSync(workspace), workspace, targetPlatform) ||
        !marker?.isFile() || marker.isSymbolicLink() || !samePath(fs.realpathSync(paths.marker), paths.marker, targetPlatform) ||
        fs.readFileSync(paths.marker, "utf8") !== JSON.stringify({version: 1, checkout: repo}) + "\n") {
      throw new Error("Windows support workspace is unmarked or belongs to another checkout");
    }
  }
  return paths;
}

/** @param {string} directory */
function ensureDirectory(directory) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!stat) {fs.mkdirSync(directory, {mode: 0o700});} else if (!stat.isDirectory() || stat.isSymbolicLink() || !samePath(fs.realpathSync(directory), directory)) {
    throw new Error("Windows support cache path is redirected or not a directory: " + directory);
  }
}
/** @param {string} directory */
function existingOutputDirectory(directory) {
  const stat = fs.lstatSync(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink() || !samePath(fs.realpathSync(directory), directory)) {
    throw new Error("Windows support cache output is redirected or not a directory: " + directory);
  }
}
/** Enumerate only this adapter's direct content-addressed output roots. Rows that are not current
 * are retired by retireManagedOutputs() once the support build and its export succeed.
 * @param {string} parent @param {string} current */
function managedOutputCandidates(parent, current) {
  return fs.readdirSync(parent, {withFileTypes: true}).filter(entry => /^[a-f0-9]{64}$/.test(entry.name)).map(entry => {
    const root = path.join(parent, entry.name);
    if (!entry.isDirectory() || entry.isSymbolicLink() || !samePath(fs.realpathSync(root), root)) {
      throw new Error("Windows support content-addressed output is unsafe: " + root);
    }
    return {root, key: entry.name, current: samePath(root, current)};
  });
}
/** @param {string} parent @param {string} current */
function managedCoreCandidates(parent, current) {
  return fs.readdirSync(parent, {withFileTypes: true}).filter(entry => /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(entry.name)).map(entry => {
    const root = path.join(parent, entry.name), receipt = path.join(root, "inputs.json");
    if (!entry.isDirectory() || entry.isSymbolicLink() || !samePath(fs.realpathSync(root), root)) {
      throw new Error("Windows Core output candidate is unsafe: " + root);
    }
    return {root, receipt, current: samePath(root, current)};
  });
}

/** Delete every non-current support output and Core run and return the deleted rows. Outputs are
 * content-addressed and Core runs are keyed by cache state, so a later build never reads an older
 * root again; builds of one checkout are serialized by the Engineering build lock. Each root is
 * re-checked just before removal, and rmSync unlinks nested links instead of following them.
 * @param {Record<"core"|"hermes"|"io"|"openssl", {root:string,current:boolean}[]>} candidates */
function retireManagedOutputs(candidates) {
  const retired = Object.fromEntries(Object.entries(candidates).map(([kind, rows]) => [kind, rows.filter(row => !row.current)]));
  for (const row of Object.values(retired).flat()) {
    existingOutputDirectory(row.root);
    fs.rmSync(row.root, {recursive: true});
  }
  return /** @type {typeof candidates} */ (retired);
}

/** Re-admit a content-addressed I/O output against this exact source/tool/recipe identity. @param {any} plan @param {any} deps */
function admitIoOutput(plan, deps = {}) {
  const admitted = (deps.admitIo || io.nativeIoInputs)(plan.io.output, "win32"), receipt = (deps.readJson || readBuildJson)(admitted.receipt),
    expectedInputs = {[plan.sources.libuv]: io.ioPins.libuv.sha256, [plan.sources.zlib]: io.ioPins.zlib.sha256};
  if (!isDeepStrictEqual(receipt.inputs, expectedInputs) || !isDeepStrictEqual(receipt.fingerprints, plan.io.tools) ||
      !isDeepStrictEqual(receipt.recipe, plan.recipes.io) || receipt.target !== "win32" || receipt.publicationQualified !== false) {
    throw new Error("Cached Windows I/O SDK no longer binds the current inputs and recipe");
  }
  return admitted;
}
/** Re-admit a content-addressed OpenSSL output. @param {any} plan @param {any} deps */
function admitOpenSslOutput(plan, deps = {}) {
  const admitted = (deps.admitOpenSsl || openssl.nativeOpenSslInputs)(plan.openssl.output, "win32"), receipt = (deps.readJson || readBuildJson)(admitted.receipt),
    configHash = createHash("sha256").update(JSON.stringify(plan.openssl.config)).digest("hex");
  if (receipt.archive?.path !== plan.sources.openssl || receipt.archive?.sha256 !== plan.hashes.openssl || receipt.config?.sha256 !== configHash ||
      !isDeepStrictEqual(receipt.tools, plan.openssl.tools) || !isDeepStrictEqual(receipt.recipe, plan.recipes.openssl) || receipt.publicationQualified !== false) {
    throw new Error("Cached Windows OpenSSL SDK no longer binds the current archive, tools and recipe");
  }
  return admitted;
}
/** Re-admit a Hermes output using its maintained public plan/source validators. @param {any} plan @param {any} deps */
function admitHermesOutput(plan, deps = {}) {
  const readJson = deps.readJson || readBuildJson, output = plan.hermes.output,
    buildPlan = hermes.hermesSdkPlan(plan.sources.hermes, output, plan.tools.hermes, jobs, "win32"),
    result = readJson(path.join(output, "result.json")), before = readJson(path.join(output, "before.json")),
    currentRecipe = plan.recipes.hermes, currentTools = plan.hermes.tools;
  if (Object.keys(result).sort().join(",") !== "artifacts,build,buildOnly,fingerprints,pins,publicationQualified,recipe,source,sources,target,version" ||
      Object.keys(before).sort().join(",") !== "compile,configure,fingerprints,pins,recipe,sources" ||
      !isDeepStrictEqual(Object.keys(result.artifacts || {}).sort(), Object.keys(buildPlan.artifacts).sort()) ||
      result.version !== 1 || result.target !== "win32" || !isDeepStrictEqual(result.pins, hermes.hermesPins) || result.source !== buildPlan.source ||
      result.build !== buildPlan.build || result.buildOnly !== true || result.publicationQualified !== false ||
      !isDeepStrictEqual(result.recipe, currentRecipe) || !isDeepStrictEqual(result.fingerprints, currentTools) ||
      !isDeepStrictEqual(before.pins, hermes.hermesPins) || !isDeepStrictEqual(before.recipe, currentRecipe) ||
      !isDeepStrictEqual(before.fingerprints, currentTools) || !isDeepStrictEqual(before.configure, buildPlan.configure) ||
      !isDeepStrictEqual(before.compile, buildPlan.compile) || !isDeepStrictEqual(before.sources, result.sources)) {
    throw new Error("Cached Windows Hermes SDK no longer binds the current source, tools and recipe");
  }
  /** @param {string[]} args */
  const git = args => {
    const execute = deps.gitCommand || (/** @param {string} tool @param {string[]} argv */
      (tool, argv) => spawnSync(tool, argv, {encoding: "utf8", timeout: 60000, maxBuffer: 8 << 20, windowsHide: true}));
    const run = execute(plan.tools.hermes.git, args);
    if (run.error || run.signal || run.status !== 0) {throw new Error("Cached Hermes source Git admission failed", {cause: run.error});}
    return run.stdout || "";
  };
  if (git(["-C", buildPlan.source, "rev-parse", "HEAD"]).trim() !== hermes.hermesPins.revision) {
    throw new Error("Cached Hermes source is not checked out at the pinned revision");
  }
  const currentSources = hermes.hermesSourceFiles(buildPlan.source, git(["-C", buildPlan.source, "ls-files", "-z"]));
  if (git(["-C", buildPlan.source, "ls-files", "--others", "-z"])) {throw new Error("Cached Hermes source has untracked files");}
  verifySourceInputs({files: result.sources, graph: {}}, {files: currentSources, graph: {}}, "Cached Windows Hermes SDK");
  hermes.validateHermesCMakeCache(buildPlan, "win32");
  for (const [name, file] of Object.entries(buildPlan.artifacts)) {
    const row = result.artifacts?.[name], stat = fs.lstatSync(file, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink() || row?.path !== file || row.bytes !== stat.size || row.sha256 !== (deps.hash || fileHash)(file) ||
        (name !== "compiler" && !isStaticArchive(file))) {throw new Error("Cached Hermes artifact is invalid: " + name);}
  }
  return {root: output, receipt: path.join(output, "result.json"), before: path.join(output, "before.json"), result};
}

/** Produce/reuse only selected native support; Core delegates incremental work to its public MSBuild cache. @param {WindowsSupportOptions} options
 * @param {{platform?:NodeJS.Platform,arch?:string,hash?:typeof fileHash,gitProbe?:(git:string,source:string)=>void,gitCommand?:(git:string,args:string[])=>any,
 * buildIo?:(archives:any,output:string,tools:any,jobs:number,env:NodeJS.ProcessEnv)=>unknown,
 * buildOpenSsl?:(archive:string,output:string,config:any,env:NodeJS.ProcessEnv)=>unknown,
 * buildHermes?:(source:string,output:string,tools:any,jobs:number,env:NodeJS.ProcessEnv)=>unknown,
 * buildCore?:(descriptor:string,output:string,env:NodeJS.ProcessEnv)=>any,
 * currentCoreAdmission?:(descriptor:string,prospectiveOutput:string,env:NodeJS.ProcessEnv)=>any,
 * admitIo?:(output:string,platform:string)=>any,admitOpenSsl?:(output:string,platform:string)=>any,readJson?:(file:string)=>any,
 * auditCore?:(state:any)=>any,compilerProbe?:(file:string)=>void,
 * runtimeEntries?:(compiler:string)=>ReturnType<typeof nativeCi.windowsControllerIntelRuntimeEntries>,
 * environment?:NodeJS.ProcessEnv,id?:()=>string}} [dependencies] */
function buildLocalNativeWindowsSupport(options, dependencies = {}) {
  const inputEnv = dependencies.environment || process.env;
  const plan = windowsSupportPlan(options, dependencies), paths = admitWindowsSupportWorkspace(plan.paths.workspace, dependencies.platform || process.platform),
    env = {...inputEnv, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: String(jobs), OMP_NUM_THREADS: String(jobs)};
  for (const directory of [paths.artifacts, paths.io, paths.openssl, paths.hermes, paths.coreRuns, paths.descriptors]) {ensureDirectory(directory);}
  /** @param {string} output @param {() => unknown} build @param {() => any} admit */
  const obtain = (output, build, admit) => {
    const stat = fs.lstatSync(output, {throwIfNoEntry: false});
    if (stat) {existingOutputDirectory(output); return {value: admit(), reused: true};}
    build();
    return {value: admit(), reused: false};
  };
  const ioResult = obtain(plan.io.output,
      () => (dependencies.buildIo || io.buildNativeIo)({libuv: plan.sources.libuv, zlib: plan.sources.zlib}, plan.io.output, plan.tools.io, jobs, env),
      () => admitIoOutput(plan, dependencies)),
    sslResult = obtain(plan.openssl.output,
      () => (dependencies.buildOpenSsl || openssl.buildNativeOpenSsl)(plan.sources.openssl, plan.openssl.output, plan.openssl.config, env),
      () => admitOpenSslOutput(plan, dependencies)),
    hermesResult = obtain(plan.hermes.output,
      () => (dependencies.buildHermes || hermes.buildNativeHermes)(plan.sources.hermes, plan.hermes.output, plan.tools.hermes, jobs, env),
      () => admitHermesOutput(plan, dependencies)),
    ioSdk = ioResult.value, sslSdk = sslResult.value, hermesSdk = hermesResult.value;
  const id = dependencies.id || randomUUID, runId = id();
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(runId)) {
    throw new Error("Windows Core output identity must be a UUID");
  }
  const prospectiveCoreOutput = path.join(paths.coreRuns, runId),
    descriptorValue = core.validateWindowsCoreDescriptor({...plan.tools.core, version: 1, libuvInclude: ioSdk.include}),
    descriptorKey = createHash("sha256").update(JSON.stringify(descriptorValue)).digest("hex"),
    descriptor = path.join(paths.descriptors, descriptorKey + ".json");
  if (fs.lstatSync(descriptor, {throwIfNoEntry: false})) {
    const stat = fs.lstatSync(descriptor);
    if (!stat.isFile() || stat.isSymbolicLink() || !samePath(fs.realpathSync(descriptor), descriptor) ||
        !isDeepStrictEqual((dependencies.readJson || readBuildJson)(descriptor), descriptorValue)) {
      throw new Error("Windows Core descriptor path is occupied by different or unsafe inputs");
    }
  } else {fs.writeFileSync(descriptor, JSON.stringify(descriptorValue, null, 2) + "\n", {flag: "wx", mode: 0o600});}
  if (fs.lstatSync(prospectiveCoreOutput, {throwIfNoEntry: false})) {throw new Error("Windows Core output identity is already occupied");}
  const readJson = dependencies.readJson || readBuildJson, auditCore = dependencies.auditCore || nativeCi.auditWindowsNativeCoreOutput,
    forceCoreBuild = inputEnv["MOM_FORCE_REBUILD"] === "1" || inputEnv["MOM_NATIVE_CACHE"] === "0",
    currentAdmission = dependencies.currentCoreAdmission || core.windowsCoreCurrentCacheAdmission;
  /** @param {string} file */
  const readCoreCandidate = file => {
    const before = fs.lstatSync(file, {throwIfNoEntry: false});
    if (!before?.isFile() || before.isSymbolicLink() || before.nlink !== 1 || before.size < 1 || before.size > (8 << 20)) {
      throw new Error("Windows Core output candidate receipt is missing, aliased, unsafe, or excessive: " + file);
    }
    const sha256 = fileHash(file), receipt = readJson(file), after = fs.lstatSync(file, {throwIfNoEntry: false});
    if (!after?.isFile() || after.isSymbolicLink() || after.nlink !== 1 || before.dev !== after.dev || before.ino !== after.ino ||
        before.size !== after.size || before.mtimeMs !== after.mtimeMs || before.ctimeMs !== after.ctimeMs || fileHash(file) !== sha256) {
      throw new Error("Windows Core output candidate receipt changed during admission: " + file);
    }
    return {receipt, sha256};
  };
  /** @type {string} */ let coreOutput = prospectiveCoreOutput;
  /** @type {any} */ let coreReceipt;
  /** @type {any} */ let coreAdmitted;
  let coreReused = false;
  if (!forceCoreBuild) {
    const before = currentAdmission(descriptor, prospectiveCoreOutput, env);
    if (before?.reusable === true && /^[a-f0-9]{64}$/i.test(before.descriptorSha256 || "") &&
        /^[a-f0-9]{64}$/i.test(before.stateSha256 || "") && /^[a-f0-9]{64}$/i.test(before.archiveSha256 || "")) {
      const matches = [];
      for (const candidate of managedCoreCandidates(paths.coreRuns, prospectiveCoreOutput)) {
        const {receipt, sha256} = readCoreCandidate(candidate.receipt);
        if (!receipt || typeof receipt !== "object" || Array.isArray(receipt) ||
            !/^[a-f0-9]{64}$/i.test(receipt.descriptorSha256 || "") ||
            !/^[a-f0-9]{64}$/i.test(receipt.incremental?.stateSha256 || "")) {
          throw new Error("Windows Core output candidate has an incomplete producer receipt: " + candidate.root);
        }
        if (receipt.descriptorSha256 === before.descriptorSha256 && receipt.incremental.stateSha256 === before.stateSha256) {
          matches.push({candidate, receipt, receiptSha256: sha256});
        }
      }
      if (matches.length > 1) {throw new Error("Multiple Windows Core outputs claim the current cache state");}
      const match = matches[0];
      if (match) {
        const {candidate, receipt, receiptSha256} = match, archive = path.join(candidate.root, "cpu-core.lib"),
          archiveSha256 = before.archiveSha256;
        if (receipt.archive?.path !== archive || receipt.archive?.sha256 !== archiveSha256 ||
            receipt.objectSha256 !== archiveSha256) {
          throw new Error("Windows Core output receipt does not match the current cache archive");
        }
        const admitted = auditCore(/** @type {any} */ ({output: candidate.root, include: ioSdk.include}));
        if (admitted.archive !== archive || admitted.receiptFile !== candidate.receipt || admitted.archiveSha256 !== archiveSha256 ||
            admitted.receiptSha256 !== receiptSha256 || fileHash(archive) !== archiveSha256 || fileHash(candidate.receipt) !== receiptSha256) {
          throw new Error("Windows Core cached output changed during public receipt admission");
        }
        const after = currentAdmission(descriptor, prospectiveCoreOutput, env);
        if (after?.reusable !== true || after.descriptorSha256 !== before.descriptorSha256 || after.stateSha256 !== before.stateSha256 ||
            after.archiveSha256 !== before.archiveSha256) {
          throw new Error("Windows Core cache changed during local output admission");
        }
        if (fileHash(archive) !== archiveSha256 || fileHash(candidate.receipt) !== receiptSha256) {
          throw new Error("Windows Core cached output changed during cache re-admission");
        }
        coreOutput = candidate.root; coreReceipt = receipt; coreAdmitted = admitted; coreReused = true;
      }
    }
  }
  if (!coreReused) {
    coreReceipt = (dependencies.buildCore || core.buildNativeWindowsCore)(descriptor, coreOutput, env);
    coreAdmitted = auditCore(/** @type {any} */ ({output: coreOutput, include: ioSdk.include}));
  }
  const runtimeLibraries = Object.fromEntries(plan.hermes.runtimeEntries.filter(row => row.kind === "library").map(row => [row.name, row.source])),
    runtimeLicenses = Object.fromEntries(plan.hermes.runtimeEntries.filter(row => row.kind === "license").map(row => [row.path, row.source])),
    runtimeRoot = path.dirname(Object.values(runtimeLibraries)[0] || "");
  if (!Object.keys(runtimeLibraries).length) {throw new Error("Pinned Intel runtime has no selected static libraries");}
  verifySourceInputs(plan.hermes.runtimeCatalog, sourceInputs(repo, [], ["runtime/windows-intel-runtime.json"]), "Windows runtime catalog");
  if ((dependencies.hash || fileHash)(plan.tools.io.cc) !== runtimeCatalog.compiler.sha256) {throw new Error("Intel compiler changed during Windows support production");}
  for (const entry of plan.hermes.runtimeEntries) {
    const stat = fs.lstatSync(entry.source, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink() || stat.size !== entry.bytes || (dependencies.hash || fileHash)(entry.source) !== entry.sha256) {
      throw new Error("Pinned Intel runtime changed during Windows support production: " + entry.path);
    }
  }
  return {workspace: paths.workspace, reused: {io: ioResult.reused, openssl: sslResult.reused, hermes: hermesResult.reused, core: coreReused},
    core: {output: coreOutput, archive: coreAdmitted.archive, receipt: coreAdmitted.receiptFile, admitted: coreAdmitted, producerReceipt: coreReceipt},
    io: {root: plan.io.output, ...ioSdk, selected: plan.selected.io},
    openssl: {root: plan.openssl.output, ...sslSdk, selected: plan.selected.openssl},
    hermes: {root: plan.hermes.output, ...hermesSdk,
      hermes: {source: hermesSdk.result.source, build: hermesSdk.result.build, license: path.join(hermesSdk.result.source, "LICENSE")},
      intelRuntime: {libRoot: runtimeRoot, libraries: runtimeLibraries, licenses: runtimeLicenses},
      runtimeEntries: plan.hermes.runtimeEntries, selected: plan.selected.hermes.map(name => path.join(plan.hermes.output, ...name.split("/")))},
    coreCache: coreReceipt.incremental, managedOutputCandidates: {
      io: managedOutputCandidates(paths.io, plan.io.output), openssl: managedOutputCandidates(paths.openssl, plan.openssl.output),
      hermes: managedOutputCandidates(paths.hermes, plan.hermes.output), core: managedCoreCandidates(paths.coreRuns, coreOutput)}, plan};
}

module.exports = {windowsSupportPaths, validateWindowsSupportTools, windowsSupportPlan, admitWindowsSupportWorkspace,
  admitIoOutput, admitOpenSslOutput, admitHermesOutput, managedOutputCandidates, managedCoreCandidates, retireManagedOutputs,
  buildLocalNativeWindowsSupport};
