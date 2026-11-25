"use strict";

// Continue the admitted local source/LLVM/input/resource cohort through the existing
// build-only JIT and static DPC++ producers. This is deliberately not a release command.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk"), nativeCi = require("./native-ci-linux-jit");
const {verifyPreparedSdkSources} = require("./prepare-native-sdk-sources");
const {verifyLinuxLlvmAdmission} = require("./build-native-llvm-sdk");
const {auditDpcppCudaInputs} = require("./native-dpcpp-cuda-inputs");
const {jitSdkConfig} = require("./build-native-jit-sdk");
const {jitBuildStages} = require("./native-jit-stages");
const {dpcppEntries, compilerArchives} = require("./native-compiler-isolation");
const {linuxStaticSdkRecipe, linuxSdkJitInput, linuxSdkHeaders, linuxStaticSdkPlan, buildLinuxStaticSdk: _buildLinuxStaticSdk} =
  require("./build-native-dpcpp-sdk");
const {dpcppComponentInputs} = require("./native-dpcpp-component");
const {pins} = require("./native-dpcpp-pins");
const {canonicalBuildDirectory, hasExactBuildFields, fileHash, readBuildJson, verifySourceInputs, buildFileInputs} = require("./build-source-inputs");
const {isStaticArchive} = require("./native-audit");

const repo = fs.realpathSync(path.resolve(__dirname, ".."));
const phases = Object.freeze(["configure", ...jitBuildStages]);
/** @param {string} name */
const producer = name => path.join(repo, "scripts", name);
const isolationScope = "Original static compiler graph isolated; NOT execution, host-JIT bridge, GPU or release qualification";

/** Stable outputs are owned by the admitted workspace; interrupted evidence is never cleared.
 * @param {string} workspace */
function localJitSdkPaths(workspace) {
  if (typeof workspace !== "string" || !path.isAbsolute(workspace) || path.resolve(workspace) !== workspace) {
    throw new Error("Local JIT SDK workspace must be a canonical absolute path");
  }
  const output = path.join(workspace, "compiler-sdk"), isolation = path.join(workspace, "compiler-isolation"),
    staticOutput = path.join(workspace, "static-dpcpp-sdk");
  return {workspace, configFile: path.join(workspace, "compiler-sdk.json"), output, isolation,
    isolatedObject: path.join(isolation, "isolated", "compiler.o"), staticOutput, staticComponent: staticOutput + ".component"};
}

/** Build the explicit context from completed local stages without acquiring or repairing inputs.
 * @param {string} workspace */
function localJitSdkContextFromWorkspace(workspace) {
  const paths = local.supportPaths(workspace);
  return {source: {workspace, compilerFetch: paths.compilerFetch, compilerSources: paths.compilerSources,
    buildInputs: paths.buildInputs, llvmSdk: paths.llvmSdk},
  inputs: {workspace, dpcppRoot: path.join(paths.dpcppCudaRoot, "dpcpp"), cudaRoot: path.join(paths.dpcppCudaRoot, "cuda"),
    manifest: path.join(paths.dpcppCudaRoot, "inputs.json")},
  resources: {workspace, root: paths.linuxJitResources, target: "x86_64-unknown-linux-gnu"}};
}

/** Require exact outputs from the existing source, input, and resource slices.
 * @param {unknown} value */
function localJitSdkContext(value) {
  if (!value || typeof value !== "object" || Array.isArray(value) ||
      !hasExactBuildFields(value, ["inputs", "resources", "source"])) {
    throw new Error("Local JIT SDK needs only admitted source, DPC++/CUDA, and resource stage results");
  }
  const context = /** @type {{source?:{workspace?:unknown,compilerFetch?:unknown,compilerSources?:unknown,buildInputs?:unknown,llvmSdk?:unknown},
    inputs?:{workspace?:unknown,dpcppRoot?:unknown,cudaRoot?:unknown,manifest?:unknown},
    resources?:{workspace?:unknown,root?:unknown,target?:unknown}}} */ (value),
    source = context.source, inputs = context.inputs, resources = context.resources;
  if (!source || !inputs || !resources || typeof source.workspace !== "string" || source.workspace !== inputs.workspace ||
      source.workspace !== resources.workspace) {throw new Error("Local JIT SDK stage results must share one admitted workspace");}
  canonicalBuildDirectory(source.workspace, "Admitted native SDK workspace");
  canonicalBuildDirectory(path.join(source.workspace, "sources"), "Native SDK workspace source directory");
  const root = local.admitWorkspace(source.workspace, repo), paths = local.supportPaths(root), expected = localJitSdkContextFromWorkspace(root);
  if (source.compilerFetch !== paths.compilerFetch || source.compilerSources !== paths.compilerSources ||
      source.buildInputs !== paths.buildInputs || source.llvmSdk !== paths.llvmSdk ||
      inputs.dpcppRoot !== expected.inputs.dpcppRoot || inputs.cudaRoot !== expected.inputs.cudaRoot || inputs.manifest !== expected.inputs.manifest ||
      resources.root !== paths.linuxJitResources || resources.target !== expected.resources.target) {
    throw new Error("Local JIT SDK inputs do not match the exact admitted workspace paths and target");
  }
  return {source: expected.source, inputs: expected.inputs, resources: expected.resources, paths: localJitSdkPaths(root), support: paths};
}

/** @param {ReturnType<typeof localJitSdkContext>} context @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools
 * @param {number} jobs */
function localJitSdkConfig(context, tools, jobs) {
  return jitSdkConfig({version: 1, target: "linux", sources: path.join(context.support.compilerSources, "sources.json"),
    sdk: context.inputs.dpcppRoot, resources: path.join(context.support.linuxJitResources, "resources"),
    tools: {cmake: tools.cmake, ninja: tools.ninja, cc: tools.cc, cxx: tools.cxx}, jobs});
}

/** @param {string} file @param {string} label */
function regularFile(file, label) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== file) {
    throw new Error(label + " is missing or not a canonical regular file: " + file);
  }
}

/** Verify the producer's stable root map in addition to its content receipt.
 * @param {string} receiptFile @param {string} output @param {"compiler"|"linux"} target */
function verifyPreparedRoots(receiptFile, output, target) {
  const receipt = verifyPreparedSdkSources(receiptFile, target), rows = receipt.prepared;
  for (const name of Object.keys(rows)) {
    const root = path.join(output, name);
    if (rows[name]?.root !== root) {throw new Error("Prepared " + target + " source root changed: " + name);}
    canonicalBuildDirectory(root, "Prepared " + target + " source root");
  }
  return receipt;
}

/** Read-only admission for every explicit prerequisite. This runs only after the internal
 * entry has proved its actual systemd cgroup limits. @param {ReturnType<typeof localJitSdkContext>} context
 * @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools */
function auditLocalJitSdkInputs(context, tools) {
  verifyPreparedRoots(path.join(context.support.compilerSources, "sources.json"), context.support.compilerSources, "compiler");
  verifyPreparedRoots(path.join(context.support.linuxSources, "sources.json"), context.support.linuxSources, "linux");
  verifyLinuxLlvmAdmission(context.support.llvmSdk);
  auditDpcppCudaInputs(local.localDpcppCudaState(context.support), readBuildJson(context.inputs.manifest));
  local.runLocalJitResourceChild("audit", context.paths.workspace, tools);
}

/** Return the last complete phase name while rejecting gaps and impossible result ordering.
 * Receipt contents are admitted separately by the public read-only phase auditor.
 * @param {string} output @returns {string} */
function localJitSdkPhasePrefix(output) {
  canonicalBuildDirectory(output, "Local JIT compiler output");
  let missing = false, latest = null;
  const allowed = new Set(["build", "before.json", "configured-tools.json", "commands.log", "commands.stderr.log"]);
  for (const phase of phases) {
    const file = path.join(output, "stage-" + phase + ".json"),
      stageStat = fs.lstatSync(file, {throwIfNoEntry: false}), log = path.join(output, phase + ".log"),
      logStat = fs.lstatSync(log, {throwIfNoEntry: false}), dependencyLog = path.join(output, "dependencies-" + phase + ".log"),
      dependencyStat = phase === "configure" ? null : fs.lstatSync(dependencyLog, {throwIfNoEntry: false});
    if (stageStat) {
      regularFile(file, "Local JIT phase receipt");
      regularFile(log, "Local JIT phase log");
      allowed.add(path.basename(file)); allowed.add(path.basename(log));
      if (phase !== "configure") {regularFile(dependencyLog, "Local JIT dependency log"); allowed.add(path.basename(dependencyLog));}
    } else if (logStat || dependencyStat) {
      throw new Error("Local JIT phase has interrupted logs without its completion receipt; preserve the output");
    }
    if (!stageStat) {missing = true; continue;}
    if (missing) {throw new Error("Local JIT SDK phases are not a contiguous prefix; preserve the output");}
    latest = phase;
  }
  if (!latest) {throw new Error("Local JIT output is interrupted before the configure receipt; preserve the output");}
  regularFile(path.join(output, "before.json"), "Local JIT config handoff");
  regularFile(path.join(output, "configured-tools.json"), "Local JIT configured tools");
  regularFile(path.join(output, "commands.log"), "Original JIT command graph");
  regularFile(path.join(output, "commands.stderr.log"), "Original JIT command graph diagnostics");
  canonicalBuildDirectory(path.join(output, "build"), "Local JIT build tree");
  for (const name of fs.readdirSync(output)) {
    if (!allowed.has(name) && name !== "result.json") {throw new Error("Unrecognized local JIT output entry; preserve the output: " + name);}
  }
  const result = fs.lstatSync(path.join(output, "result.json"), {throwIfNoEntry: false});
  if (latest === "link") {regularFile(path.join(output, "result.json"), "Completed source-JIT result");} else if (result) {
    throw new Error("Local JIT result exists before the final link phase; preserve the output");
  }
  return latest;
}

/** Check the isolation producer's output against the admitted public compiler archive graph.
 * @param {ReturnType<typeof localJitSdkContext>} context @param {string[]} libraries */
function auditLocalCompilerIsolation(context, libraries) {
  const root = context.paths.isolation, raw = path.join(root, "compiler-raw.o"), receiptFile = path.join(root, "result.json"),
    isolatedReceiptFile = path.join(root, "isolated/result.json"), object = context.paths.isolatedObject;
  canonicalBuildDirectory(root, "Local compiler isolation output"); regularFile(raw, "Original compiler partial link");
  regularFile(receiptFile, "Compiler isolation receipt"); regularFile(isolatedReceiptFile, "Isolated compiler receipt"); regularFile(object, "Isolated compiler object");
  const result = readBuildJson(receiptFile), isolated = readBuildJson(isolatedReceiptFile), entryPoints = [...dpcppEntries];
  if (!isDeepStrictEqual(result, {archives: libraries, ...isolated, scope: isolationScope}) || !isDeepStrictEqual(isolated, {
    input: raw, object, prefix: "mom_dpcpp23_", entryPoints,
    definitions: isolated.definitions, groups: isolated.groups, renamed: isolated.renamed,
    inputSha256: fileHash(raw), objectSha256: fileHash(object)}) ||
      result.input !== raw || result.object !== object || result.prefix !== "mom_dpcpp23_" ||
      !isDeepStrictEqual(result.entryPoints, entryPoints) || result.inputSha256 !== fileHash(raw) || result.objectSha256 !== fileHash(object) ||
      !Number.isInteger(result.definitions) || result.definitions < entryPoints.length || !Number.isInteger(result.groups) || result.groups < 0 ||
      !Number.isInteger(result.renamed) || result.renamed < 1 || result.scope !== isolationScope) {
    throw new Error("Local compiler isolation evidence does not bind the original JIT graph and output bytes");
  }
  return result;
}

/** Admit only a static SDK recipe matching the current public producer source footprint.
 * @param {unknown} value */
function verifyLocalStaticSdkRecipe(value) {
  verifySourceInputs(value, linuxStaticSdkRecipe(), "Local static DPC++ SDK");
  return value;
}

/** The public static producer records exact header dependency digests as an absolute-path map.
 * @param {unknown} value */
function verifyLocalStaticDependencies(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Local static DPC++ dependencies must be a file-hash map");
  }
  const rows = /** @type {Record<string, unknown>} */ (value), files = Object.keys(rows);
  if (!files.length || files.length > 65536 || files.some(file => !path.isAbsolute(file) || path.resolve(file) !== file ||
      !/^[a-f0-9]{64}$/.test(typeof rows[file] === "string" ? rows[file] : ""))) {
    throw new Error("Local static DPC++ dependency map is malformed or exceeds bounds");
  }
  const current = buildFileInputs(files);
  if (!isDeepStrictEqual(rows, current)) {throw new Error("Local static DPC++ dependency inputs changed after production");}
  return current;
}

/** Read-only admission for the completed static DPC++ component and its source-JIT inputs.
 * @param {ReturnType<typeof localJitSdkContext>} context @param {ReturnType<typeof localJitSdkConfig>} config
 * @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools @param {string[]} libraries */
function auditLocalStaticDpcpp(context, config, tools, libraries) {
  const output = context.paths.staticOutput, resultFile = path.join(output, "result.json"), componentRoot = context.paths.staticComponent;
  canonicalBuildDirectory(output, "Local static DPC++ output"); regularFile(resultFile, "Local static DPC++ result");
  const result = /** @type {ReturnType<typeof _buildLinuxStaticSdk>} */ (readBuildJson(resultFile)),
    before = readBuildJson(path.join(output, "before.json")),
    linuxSources = verifyPreparedRoots(path.join(context.support.linuxSources, "sources.json"), context.support.linuxSources, "linux"),
    plan = linuxStaticSdkPlan(Object.fromEntries(Object.entries(linuxSources.prepared).map(([name, row]) => [name, row.root])),
      context.inputs.dpcppRoot, context.inputs.cudaRoot, context.paths.isolatedObject, output, tools, local.recordedParallelJobs(before.compile)),
    expectedInputs = {...linuxSdkJitInput(context.paths.isolatedObject),
      [path.join(context.support.linuxSources, "sources.json")]: fileHash(path.join(context.support.linuxSources, "sources.json")),
      ...Object.fromEntries([...Object.values(tools), path.join(context.inputs.dpcppRoot, "bin/clang"),
        path.join(context.inputs.dpcppRoot, "bin/clang++"), path.join(context.inputs.dpcppRoot, ".mom-toolchain-sha256")].map(file => [file, fileHash(file)]))},
    headers = linuxSdkHeaders(context.inputs.dpcppRoot, context.inputs.cudaRoot),
    dependencies = verifyLocalStaticDependencies(result.dependencies),
    compilerVersionPaths = [...new Set([path.join(context.inputs.dpcppRoot, "bin/clang++"), ...Object.values(plan.hostCompilers)])].sort(),
    component = dpcppComponentInputs(componentRoot), componentReceiptFile = path.join(componentRoot, "component.json"),
    componentReceipt = readBuildJson(componentReceiptFile), archive = path.join(plan.build, "libmom-sycl-static.a");
  regularFile(archive, "Local static DPC++ archive");
  if (result.version !== 1 || result.target !== "linux" || !isDeepStrictEqual(result.pins, pins) ||
      result.sourcesFile !== path.join(context.support.linuxSources, "sources.json") || result.buildOnly !== true ||
      result.executed !== false || result.publicationQualified !== false || result.archive !== archive ||
      !isDeepStrictEqual(before.recipe, result.recipe) ||
      !isStaticArchive(archive) || fileHash(archive) !== result.sha256 || !isDeepStrictEqual(result.inputs, expectedInputs) ||
      config.sdk !== context.inputs.dpcppRoot || config.sources !== path.join(context.support.compilerSources, "sources.json") ||
      config.resources !== path.join(context.support.linuxJitResources, "resources") ||
      !isDeepStrictEqual(config.tools, tools) ||
      !isDeepStrictEqual(result.headers, headers) || !isDeepStrictEqual(before.headers, headers) ||
      !isDeepStrictEqual(result.dependencies, dependencies) || !isDeepStrictEqual(before.inputs, expectedInputs) ||
      before.sourcesFile !== result.sourcesFile || !isDeepStrictEqual(before.hostCompilers, plan.hostCompilers) ||
      !isDeepStrictEqual(before.compilerEnvironment, plan.compilerEnvironment) ||
      !isDeepStrictEqual(before.compilerVersions, result.compilerVersions) ||
      !isDeepStrictEqual(before.configure, plan.configure) || !isDeepStrictEqual(before.compile, plan.compile) ||
      !isDeepStrictEqual(result.hostCompilers, plan.hostCompilers) ||
      !isDeepStrictEqual(Object.keys(result.compilerVersions).sort(), compilerVersionPaths) ||
      Object.values(result.compilerVersions).some(version => typeof version !== "string" || !version.trim() || version.includes("\0")) ||
      result.scope !== "Original full Linux SYCL/UR/static loader sources with isolated source JIT; no GPU, runtime, mining or publication qualification" ||
      !Array.isArray(result.audit?.artifacts) || result.audit.artifacts.length === 0 ||
      result.audit.artifacts.some((/** @type {{executable?:unknown,bytes?:unknown,sha256?:unknown}} */ row) => {
        if (typeof row.executable !== "string" || !row.executable.startsWith(output + path.sep) || !path.isAbsolute(row.executable) ||
            !Number.isInteger(row.bytes) || !/^[a-f0-9]{64}$/.test(typeof row.sha256 === "string" ? row.sha256 : "")) {return true;}
        regularFile(row.executable, "Static DPC++ audit artifact");
        return fs.statSync(row.executable).size !== row.bytes || row.sha256 !== fileHash(row.executable);
      }) ||
      componentReceipt.producer?.path !== resultFile || componentReceipt.producer?.sha256 !== fileHash(resultFile) ||
      !component.files[path.join(componentRoot, "component.json")]) {
    throw new Error("Local static DPC++ result or component does not bind the admitted source-JIT inputs");
  }
  verifyLocalStaticSdkRecipe(result.recipe);
  if (!libraries.length || !result.inputs[context.paths.isolatedObject] ||
      component.files[path.join(componentRoot, "lib/libmom-sycl-static.a")] === undefined) {
    throw new Error("Local static DPC++ component is incomplete");
  }
  return result;
}

/** Full build-only continuation, called from the internal scope-checked entry.
 * @param {unknown} context @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools
 * @param {{verifyScope?:()=>unknown,run?:typeof local.runCommand,auditInputs?:typeof auditLocalJitSdkInputs,
 *   auditCompiler?:(output:string,config:ReturnType<typeof localJitSdkConfig>,phase:string)=>unknown}} [dependencies] */
function runLocalJitSdkChild(context, tools, dependencies = {}) {
  (dependencies.verifyScope || local.verifyCurrentMemoryScope)();
  const selected = localJitSdkContext(context), paths = selected.paths,
    env = {...process.env, MOM_AOT_BUILD_ONLY: "1"}, jobs = local.localBuildJobs(env), run = dependencies.run || local.runCommand,
    auditInputs = dependencies.auditInputs || auditLocalJitSdkInputs,
    auditCompiler = dependencies.auditCompiler || nativeCi.auditLinuxJitCompilerState;
  auditInputs(selected, tools);

  const outputStat = fs.lstatSync(paths.output, {throwIfNoEntry: false}),
    configStat = fs.lstatSync(paths.configFile, {throwIfNoEntry: false});
  const isolationStat = fs.lstatSync(paths.isolation, {throwIfNoEntry: false}), staticStat = fs.lstatSync(paths.staticOutput, {throwIfNoEntry: false}),
    componentStat = fs.lstatSync(paths.staticComponent, {throwIfNoEntry: false});
  if (!outputStat && (isolationStat || staticStat || componentStat)) {throw new Error("Local JIT downstream output exists before its compiler SDK; preserve the workspace");}
  if (Boolean(staticStat) !== Boolean(componentStat) || (staticStat && !isolationStat)) {
    throw new Error("Local static DPC++ output is incomplete or out of order; preserve the workspace");
  }
  // A warm compiler keeps the job count its config recorded; that count is provenance, not an output input.
  if (configStat) {regularFile(paths.configFile, "Local JIT SDK config");}
  const recorded = configStat ? readBuildJson(paths.configFile) : undefined,
    config = localJitSdkConfig(selected, tools, configStat ? recorded?.jobs : jobs);
  if (configStat) {
    if (!isDeepStrictEqual(recorded, config)) {throw new Error("Local JIT SDK config changed; preserve the workspace");}
  } else {
    if (outputStat) {throw new Error("Local JIT compiler output exists without its original config; preserve the workspace");}
    fs.writeFileSync(paths.configFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  }
  let current = null;
  if (outputStat) {
    const last = localJitSdkPhasePrefix(paths.output);
    current = auditCompiler(paths.output, config, last || "configure");
  }
  /** @param {string[]} args @param {string} label */
  function execute(args, label) {
    run(process.execPath, args, {label, env, cwd: repo});
  }
  if (!outputStat) {
    execute([producer("build-native-jit-sdk.js"), paths.configFile, paths.output, "configure"], "Local JIT SDK configure");
    auditCompiler(paths.output, config, "configure"); current = {stage: "configure"};
  }
  for (const phase of phases.slice(phases.indexOf(/** @type {string} */ (current?.stage || "configure")) + 1)) {
    execute([producer("build-native-jit-sdk.js"), paths.configFile, paths.output, phase], "Local JIT SDK phase " + phase);
    auditCompiler(paths.output, config, phase);
  }
  const link = readBuildJson(path.join(paths.output, "stage-link.json")), libraries = compilerArchives(
    fs.readFileSync(path.join(paths.output, "commands.log"), "utf8"), path.join(paths.output, "build"));
  if (!isDeepStrictEqual(link.libraries, libraries)) {throw new Error("Local JIT SDK archive graph changed after final phase admission");}

  if (isolationStat) {auditLocalCompilerIsolation(selected, libraries);} else {
    execute([producer("native-compiler-isolation.js"), path.join(paths.output, "build"), paths.isolation], "Local compiler isolation");
    auditLocalCompilerIsolation(selected, libraries);
  }
  if (staticStat || componentStat) {
    if (!staticStat || !componentStat) {throw new Error("Local static DPC++ output is incomplete; preserve the workspace");}
    auditLocalStaticDpcpp(selected, config, tools, libraries);
  } else {
    execute([producer("build-native-dpcpp-sdk.js"), path.join(selected.support.linuxSources, "sources.json"), selected.inputs.dpcppRoot,
      selected.inputs.cudaRoot, paths.isolatedObject, paths.staticOutput, tools.cmake, tools.ninja, String(jobs), tools.cc, tools.cxx],
    "Local static DPC++ SDK");
    auditLocalStaticDpcpp(selected, config, tools, libraries);
  }
  return {workspace: paths.workspace, compilerSdk: paths.output, compilerIsolation: paths.isolation,
    isolatedObject: paths.isolatedObject, staticDpcppSdk: paths.staticOutput, staticComponent: paths.staticComponent,
    buildOnly: true, executed: false, publicationQualified: false};
}

/** Build or readmit the local source-JIT/static-DPC++ slice under the established memory cap.
 * @param {{source:unknown,inputs:unknown,resources:unknown}} context
 * @param {{run?:typeof local.runCommand,tools?:ReturnType<typeof local.localTools>,resolveTools?:typeof local.localTools,
 *   versionOf?:(file:string)=>string,environment?:NodeJS.ProcessEnv}} [dependencies] */
function buildLocalLinuxJitSdk(context, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local Linux JIT SDK currently requires Linux x64");}
  const selected = localJitSdkContext(context), paths = selected.paths,
    env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"}, resolveTools = dependencies.resolveTools || local.localTools,
    runnerTools = dependencies.tools || resolveTools(env, dependencies.versionOf),
    tools = {cmake: runnerTools.cmake, ninja: runnerTools.ninja, cc: runnerTools.cc, cxx: runnerTools.cxx},
    invoke = local.scopedInvoker(runnerTools, env, dependencies.run || local.runCommand),
    initial = Object.fromEntries([paths.output, paths.isolation, paths.staticOutput, paths.staticComponent].map(file =>
      [file, Boolean(fs.lstatSync(file, {throwIfNoEntry: false}))]));
  local.localJitToolsReceipt(tools);
  invoke("systemd memory-limit preflight", process.execPath, [__filename, "--memory-probe"]);
  invoke("Local Linux JIT SDK build-only slice", process.execPath,
    [__filename, "--local-jit-sdk", paths.workspace, tools.cmake, tools.ninja, tools.cc, tools.cxx]);
  return {workspace: paths.workspace, compilerSdk: paths.output, compilerIsolation: paths.isolation,
    isolatedObject: paths.isolatedObject, staticDpcppSdk: paths.staticOutput, staticComponent: paths.staticComponent,
    reused: {compilerSdk: initial[paths.output], compilerIsolation: initial[paths.isolation], staticDpcppSdk: initial[paths.staticOutput],
      staticComponent: initial[paths.staticComponent]},
    buildOnly: true, executed: false, publicationQualified: false};
}

/** @param {string[]} argv @param {()=>unknown} [verifyScope]
 * @param {{run?:typeof local.runCommand,auditInputs?:typeof auditLocalJitSdkInputs,
 *   auditCompiler?:(output:string,config:ReturnType<typeof localJitSdkConfig>,phase:string)=>unknown}} [dependencies] */
function main(argv, verifyScope = local.verifyCurrentMemoryScope, dependencies = {}) {
  const [command, ...args] = argv;
  if (command === "--memory-probe" && args.length === 0) {verifyScope(); return;}
  if (command === "--workspace" && args.length === 1 && args[0]) {
    const context = localJitSdkContextFromWorkspace(args[0]);
    return buildLocalLinuxJitSdk(context);
  }
  if (command !== "--local-jit-sdk" || args.length !== 5 || args.some(value => !value)) {
    throw new Error("Usage: node scripts/build-local-native-jit-sdk.js --workspace /workspace\n" +
      "       node scripts/build-local-native-jit-sdk.js --local-jit-sdk /workspace /cmake /ninja /icx /icpx");
  }
  const [workspace, cmake, ninja, cc, cxx] = args;
  if (![workspace, cmake, ninja, cc, cxx].every(value => path.isAbsolute(value || "") && path.resolve(value || "") === value)) {
    throw new Error("Local JIT SDK internal paths must be canonical and absolute");
  }
  return runLocalJitSdkChild(localJitSdkContextFromWorkspace(workspace || ""), {cmake: cmake || "", ninja: ninja || "", cc: cc || "", cxx: cxx || ""},
    {...dependencies, verifyScope});
}

if (require.main === module) {
  try {main(process.argv.slice(2));} catch (error) {console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1;}
}

module.exports = {localJitSdkPaths, localJitSdkContextFromWorkspace, localJitSdkContext, localJitSdkConfig, verifyPreparedRoots,
  localJitSdkPhasePrefix, verifyLocalStaticSdkRecipe, verifyLocalStaticDependencies, runLocalJitSdkChild, buildLocalLinuxJitSdk, main};
