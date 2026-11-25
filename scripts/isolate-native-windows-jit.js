"use strict";

// Consume a hash-bound completed Windows JIT SDK. This isolates archives only; it neither
// requalifies the mutable compiler build tree nor changes the producer's original receipt.
const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const {hasExactBuildFields, buildPathsOverlap, fileHash, readBuildJson, readBuildJsonSnapshot, readBuildText} = require("./build-source-inputs");
const {compilerArchives, dpcppEntries} = require("./native-compiler-isolation");
const {isolateCoffCompilerArchives} = require("./native-coff-archive");
const {jitBuildStages} = require("./native-jit-stages");
const {isStaticArchive} = require("./native-audit");
const {nativeSdkEnvironment} = require("./build-environment");
const {isObject: record} = require("../helper/validate");
const repo = path.resolve(__dirname, "..");
const hashPattern = /^[a-f0-9]{64}$/;

/** @param {unknown} value */
function absolutePath(value) {return typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) && path.resolve(value) === value && !/[\r\n\0]/.test(value);}
/** @typedef {{version?:unknown, kind?:unknown, target?:unknown, compiled?:unknown, buildOnly?:unknown, executed?:unknown,
 * publicationQualified?:unknown, plan?:unknown, libraries?:unknown, supportLibraries?:unknown, inputs?:unknown,
 * recipe?:unknown, proof?:unknown, sourcesFile?:unknown, resources?:unknown}} ProducerRecord */
/** @typedef {{target?:unknown, build?:unknown, compile?:unknown}} ProducerPlan */
/** @typedef {{graph?:unknown, files?:unknown}} RecipeRecord */
/** @typedef {{version?:unknown, tools?:unknown}} ToolsManifest */
/** @typedef {{version?:unknown, buildOnly?:unknown, stage?:unknown, beforeSha256?:unknown, previousSha256?:unknown,
 * configuredSha256?:unknown, commandsSha256?:unknown, libraries?:unknown, supportLibraries?:unknown, inputs?:unknown}} StageRecord */
/** @typedef {{libraries?:unknown, proof?:unknown}} ProducerArtifacts */
/** @typedef {{path?:unknown, sha256?:unknown}} ArtifactRecord */

/** Canonical paths are used for files this consumer actually reads, never path aliases. @param {string} file @param {"file"|"directory"} kind @param {string} label @param {number} [maximum] */
function canonicalPath(file, kind, label, maximum = Number.MAX_SAFE_INTEGER) {
  if (!absolutePath(file)) {throw new Error(label + " must be a normalized absolute path");}
  const stat = fs.lstatSync(file), valid = kind === "file" ? stat.isFile() : stat.isDirectory(), real = fs.realpathSync(file);
  const samePath = process.platform === "win32" ? path.resolve(real).toLowerCase() === file.toLowerCase() : real === file;
  if (!valid || stat.isSymbolicLink() || stat.size > maximum || !samePath) {throw new Error(label + " is not a canonical bounded regular " + kind + ": " + file);}
  return stat;
}

/** Keep the original public SDK result contract and its recorded external-input hashes intact. @param {unknown} value @param {string} build */
function validateProducerReceipt(value, build) {
  if (!record(value)) {throw new Error("Windows JIT producer receipt is incomplete, wrong-target or not build-only");}
  const receipt = /** @type {ProducerRecord} */ (value), plan = record(receipt.plan) ? /** @type {ProducerPlan} */ (receipt.plan) : null;
  if (receipt.version !== 1 || receipt.kind !== "original-source-jit-sdk" || receipt.target !== "windows" ||
      receipt.compiled !== true || receipt.buildOnly !== true || receipt.executed !== false || receipt.publicationQualified !== false ||
      !plan || plan.target !== "windows" || plan.build !== build || !Array.isArray(plan.compile) ||
      !Array.isArray(receipt.libraries) || !Array.isArray(receipt.supportLibraries) || !record(receipt.inputs) || !record(receipt.proof) ||
      typeof receipt.sourcesFile !== "string" || !absolutePath(receipt.sourcesFile) || typeof receipt.resources !== "string" || !absolutePath(receipt.resources)) {
    throw new Error("Windows JIT producer receipt is incomplete, wrong-target or not build-only");
  }
  validateRecipeRecord(receipt.recipe);
  const inputs = /** @type {Record<string, unknown>} */ (receipt.inputs), supportLibraries = /** @type {unknown[]} */ (receipt.supportLibraries), names = Object.keys(inputs);
  if (!names.length || names.length > 65536 || names.some(file => !absolutePath(file) || !hashPattern.test(String(inputs[file])))) {
    throw new Error("Windows JIT producer input-hash record is malformed");
  }
  if (!Object.hasOwn(inputs, receipt.sourcesFile) || supportLibraries.some((file) => typeof file !== "string" || !Object.hasOwn(inputs, file))) {
    throw new Error("Windows JIT source receipt or support libraries are not bound by producer inputs");
  }
}

/** Validate recorded source-recipe metadata without reopening producer sources. Freshness is a
 * producer-stage check; this consumer preserves the exact recipe bound by the caller's SHA. @param {unknown} recipe */
function validateRecipeRecord(recipe) {
  if (!record(recipe)) {throw new Error("Windows JIT producer source recipe is incomplete or unsafe");}
  const selected = /** @type {RecipeRecord} */ (recipe);
  if (!record(selected.graph) || !record(selected.files) || !Object.keys(selected.graph).length ||
      [...Object.entries(selected.graph), ...Object.entries(selected.files)].some(([name, sha256]) => typeof sha256 !== "string" || !hashPattern.test(sha256) ||
        !name || name.length > 4096 || name.split("/").some(part => !part || part === "." || part === "..") || /[\\:\0]/.test(name) || path.posix.isAbsolute(name))) {
    throw new Error("Windows JIT producer source recipe is incomplete or unsafe");
  }
  return recipe;
}

/** @param {unknown} value @returns {{llvmAr:string, llvmNm:string, llvmObjcopy:string}} */
function validateToolsManifest(value) {
  if (!record(value)) {throw new Error("LLVM tools manifest must name exactly llvmAr, llvmNm and llvmObjcopy");}
  const manifest = /** @type {ToolsManifest} */ (value), rawTools = manifest.tools;
  if (manifest.version !== 1 || !record(rawTools) || !hasExactBuildFields(manifest, ["tools", "version"]) ||
      !hasExactBuildFields(rawTools, ["llvmAr", "llvmNm", "llvmObjcopy"])) {throw new Error("LLVM tools manifest must name exactly llvmAr, llvmNm and llvmObjcopy");}
  const tools = /** @type {Record<string, unknown>} */ (rawTools), selected = {llvmAr: tools["llvmAr"], llvmNm: tools["llvmNm"], llvmObjcopy: tools["llvmObjcopy"]};
  for (const tool of Object.values(selected)) {
    if (typeof tool !== "string") {throw new Error("LLVM archive isolation tool path is invalid");}
    canonicalPath(tool, "file", "LLVM archive isolation tool", 256 << 20);
  }
  return /** @type {{llvmAr:string, llvmNm:string, llvmObjcopy:string}} */ (selected);
}

/** @param {{llvmAr:string, llvmNm:string, llvmObjcopy:string}} tools */
function hashTools(tools) {
  return Object.fromEntries(Object.entries(tools).map(([name, file]) => {
    canonicalPath(file, "file", "LLVM archive isolation tool", 256 << 20); return [name, fileHash(file)];
  }));
}

/** Re-derive ordered archive paths only from the recorded Ninja link command. @param {string} commands @param {string} build
 * @param {string[]} support @param {string[]} stageLibraries @param {Array<{path:string}>} producerLibraries */
function validateArchiveGraph(commands, build, support, stageLibraries, producerLibraries) {
  const lines = commands.split(/\r?\n/).filter(line => /(?:^|\s)\/out:bin\/native-dpcpp-jit\.exe(?:\s|$)/.test(line));
  const archives = compilerArchives(commands, build, true, support);
  if (lines.length !== 1 || !isDeepStrictEqual(stageLibraries, archives) || !isDeepStrictEqual(producerLibraries.map(row => row.path), archives)) {
    throw new Error("Windows JIT archive graph differs from its final producer receipt");
  }
  return {archives, linkLine: lines[0] || ""};
}

/** Validate phase ancestry using hash-bound receipt/command metadata, without scanning Ninja objects. @param {unknown[]} stages
 * @param {Record<string,string>} stageHashes @param {string} beforeSha @param {string} configuredSha @param {string} commandsSha
 * @param {string[]} libraries @param {string[]} support @param {Record<string,string>} producerInputs */
function validateStageAncestry(stages, stageHashes, beforeSha, configuredSha, commandsSha, libraries, support, producerInputs) {
  /** @type {string | undefined} */ let previous;
  for (const [index, stage] of stages.entries()) {
    const name = index === 0 ? "configure" : jitBuildStages[index - 1];
    if (!record(stage)) {throw new Error("Windows JIT phase ancestry changed at " + name);}
    const receipt = /** @type {StageRecord} */ (stage), inputs = receipt.inputs;
    if (receipt.version !== 1 || receipt.buildOnly !== true || receipt.stage !== name || receipt.beforeSha256 !== beforeSha ||
        (index === 0 ? receipt.previousSha256 !== undefined : receipt.previousSha256 !== previous) ||
        receipt.configuredSha256 !== configuredSha || receipt.commandsSha256 !== commandsSha || !isDeepStrictEqual(receipt.libraries, libraries) ||
        !isDeepStrictEqual(receipt.supportLibraries, support) || !record(inputs)) {throw new Error("Windows JIT phase ancestry changed at " + name);}
    for (const [file, sha256] of Object.entries(inputs)) {if (producerInputs[file] !== sha256) {throw new Error("Windows JIT phase inputs differ from producer receipt");}}
    previous = stageHashes[index === 0 ? "stage-configure.json" : "stage-" + name + ".json"];
  }
  const finalStage = stages.at(-1);
  if (!record(finalStage) || !isDeepStrictEqual(finalStage["inputs"], producerInputs)) {throw new Error("Windows JIT final phase inputs differ from producer receipt");}
}

/** Check current archive/proof bytes against the pinned producer's path-and-hash records. @param {unknown} value @param {string[]} archives @param {string} build */
function verifyRecordedArtifacts(value, archives, build) {
  if (!record(value)) {throw new Error("Windows JIT producer artifacts are incomplete");}
  const receipt = /** @type {ProducerArtifacts} */ (value), rows = receipt.libraries;
  if (!Array.isArray(rows) || rows.length !== archives.length || !record(receipt.proof)) {throw new Error("Windows JIT producer artifacts are incomplete");}
  for (const [index, item] of rows.entries()) {
    canonicalPath(archives[index] || "", "file", "Windows JIT compiler archive", 256 << 20);
    if (!record(item)) {throw new Error("Windows JIT archive membership or bytes differ from producer receipt");}
    const row = /** @type {ArtifactRecord} */ (item);
    if (row.path !== archives[index] || typeof row.sha256 !== "string" || !hashPattern.test(row.sha256) || !isStaticArchive(archives[index] || "") ||
        fileHash(archives[index] || "") !== row.sha256) {throw new Error("Windows JIT archive membership or bytes differ from producer receipt");}
  }
  const proof = path.join(build, "bin", "native-dpcpp-jit.exe"); canonicalPath(proof, "file", "Windows JIT link proof", 512 << 20);
  const proofRecord = /** @type {ArtifactRecord} */ (receipt.proof);
  if (proofRecord.path !== proof || typeof proofRecord.sha256 !== "string" || !hashPattern.test(proofRecord.sha256) || fileHash(proof) !== proofRecord.sha256) {
    throw new Error("Windows JIT linked proof executable changed");
  }
  return {proof, proofSha256: proofRecord.sha256};
}

/** @param {unknown} value */
function digest(value) {return createHash("sha256").update(JSON.stringify(value)).digest("hex");}

/** The upstream handoff SHA pins the immutable producer record, including its complete input map. @param {string} file @param {string} expectedSha256 */
function readPinnedProducerResult(file, expectedSha256) {
  canonicalPath(file, "file", "Windows JIT producer result", 8 << 20);
  if (!hashPattern.test(expectedSha256)) {throw new Error("Expected producer result SHA256 is invalid");}
  const {value: result, sha256} = readBuildJsonSnapshot(file, undefined,
    {sha256: expectedSha256, message: "Windows JIT producer result does not match its handoff SHA256"});
  return {result, sha256};
}

/** Consume one hash-bound producer output without depending on unrelated build-cache files. @param {string} sdkRoot
 * @param {string} expectedResultSha256 @param {string} toolsFile */
function inspectWindowsJitSdk(sdkRoot, expectedResultSha256, toolsFile) {
  canonicalPath(sdkRoot, "directory", "Windows JIT SDK root"); canonicalPath(toolsFile, "file", "LLVM tools manifest", 65536);
  /** @param {string} name */
  const file = name => path.join(sdkRoot, name), resultPath = file("result.json"), pinned = readPinnedProducerResult(resultPath, expectedResultSha256);
  const result = pinned.result, resultSha256 = pinned.sha256, build = path.join(sdkRoot, "build"); validateProducerReceipt(result, build);
  canonicalPath(build, "directory", "Windows JIT build root"); validateRecipeRecord(result.recipe);
  const tools = validateToolsManifest(readBuildJson(toolsFile)), toolHashes = hashTools(tools);
  const evidenceNames = ["before.json", "configured-tools.json", "commands.log", "stage-configure.json", ...jitBuildStages.map((stage) => "stage-" + stage + ".json")];
  const evidence = Object.fromEntries(evidenceNames.map(name => {
    const evidenceFile = file(name); canonicalPath(evidenceFile, "file", "Windows JIT producer evidence", 8 << 20);
    return [name, {path: evidenceFile, sha256: fileHash(evidenceFile)}];
  }));
  const before = readBuildJson(file("before.json")), commands = readBuildText(file("commands.log"));
  if (!record(before)) {throw new Error("Windows JIT producer prebuild evidence changed");}
  const prebuild = /** @type {{version?:unknown,config?:unknown,plan?:unknown,recipe?:unknown,supportLibraries?:unknown,inputs?:unknown}} */ (before);
  const config = record(prebuild.config) ? /** @type {{target?:unknown,sources?:unknown,resources?:unknown}} */ (prebuild.config) : null;
  const beforeInputs = prebuild.inputs;
  if (prebuild.version !== 1 || !config || config.target !== "windows" || config.sources !== result.sourcesFile || config.resources !== result.resources ||
      !isDeepStrictEqual(prebuild.plan, result.plan) || !isDeepStrictEqual(prebuild.recipe, result.recipe) ||
      !isDeepStrictEqual(prebuild.supportLibraries, result.supportLibraries) || !record(beforeInputs) ||
      Object.entries(beforeInputs).some(([name, hash]) => result.inputs[name] !== hash)) {throw new Error("Windows JIT producer prebuild evidence changed");}
  const configuredSha = evidence["configured-tools.json"]?.sha256 || "", commandsSha = evidence["commands.log"]?.sha256 || "";
  const stages = ["configure", ...jitBuildStages].map(stage => readBuildJson(file(stage === "configure" ? "stage-configure.json" : "stage-" + stage + ".json")));
  const stageHashes = Object.fromEntries(stages.map((_, index) => {
    const name = index === 0 ? "stage-configure.json" : "stage-" + jitBuildStages[index - 1] + ".json";
    return [name, evidence[name]?.sha256 || ""];
  }));
  validateStageAncestry(stages, stageHashes, evidence["before.json"]?.sha256 || "", configuredSha, commandsSha,
    stages.at(-1)?.libraries || [], result.supportLibraries, result.inputs);
  const graph = validateArchiveGraph(commands, build, result.supportLibraries, stages.at(-1)?.libraries || [], result.libraries);
  const artifacts = verifyRecordedArtifacts(result, graph.archives, build);
  const toolsManifestSha256 = fileHash(toolsFile), recipeSha256 = digest(result.recipe);
  return {sdkRoot, build, result, resultPath, resultSha256, evidence, archives: graph.archives, linkLine: graph.linkLine,
    linkLineSha256: digest(graph.linkLine), tools, toolsManifestSha256, toolHashes, artifacts, recipeSha256,
    inputHashes: result.inputs, protectedPaths: [sdkRoot, repo, build, resultPath, result.resources, path.dirname(result.sourcesFile),
      ...Object.values(evidence).map(row => row.path), toolsFile, ...Object.values(tools), ...graph.archives, artifacts.proof,
      ...Object.keys(result.recipe.graph).map(name => path.join(repo, ...name.split("/")))]};
}

/** @param {{protectedPaths:string[]}} selected @param {string} toolsFile @param {string} output */
function validateFreshOutput(selected, toolsFile, output) {
  if (!absolutePath(output) || fs.lstatSync(output, {throwIfNoEntry: false})) {throw new Error("Windows JIT archive isolation needs a fresh absolute output");}
  canonicalPath(path.dirname(output), "directory", "Windows JIT output parent");
  for (const input of [...selected.protectedPaths, toolsFile]) {if (buildPathsOverlap(input, output)) {throw new Error("Windows JIT isolation output overlaps a producer or tool input");}}
}

/** One handoff format for fresh CI isolation and read-only local cache admission.
 * The caller must first inspect the original producer and verify the isolation result.
 * @param {ReturnType<typeof inspectWindowsJitSdk>} prior
 * @param {ReturnType<typeof isolateCoffCompilerArchives>} isolation
 * @param {string} toolsFile @param {string} output */
function windowsJitIsolationHandoff(prior, isolation, toolsFile, output) {
  if (fileHash(__filename) !== priorAdapterSha256) {throw new Error("Windows JIT isolation adapter changed during execution");}
  const isolationResultPath = path.join(output, "result.json");
  return {version: 1, kind: "windows-jit-archive-isolation-handoff",
    target: "windows", buildOnly: true, executed: false, publicationQualified: false,
    producer: {result: {path: prior.resultPath, sha256: prior.resultSha256}, stages: Object.fromEntries(Object.entries(prior.evidence).map(([name, row]) => [name, row.sha256])),
      commands: {path: prior.evidence["commands.log"]?.path, sha256: prior.evidence["commands.log"]?.sha256}, linkCommand: {sha256: prior.linkLineSha256, text: prior.linkLine},
      proof: prior.result.proof, libraries: prior.result.libraries, inputHashes: prior.inputHashes, sourceRecipe: prior.result.recipe, recipeSha256: prior.recipeSha256},
    tools: {manifest: {path: toolsFile, sha256: prior.toolsManifestSha256}, files: Object.fromEntries(Object.entries(prior.tools).map(([name, file]) => [name, {path: file, sha256: prior.toolHashes[name]}]))},
    adapter: {path: __filename, sha256: priorAdapterSha256}, isolation: {result: {path: isolationResultPath, sha256: fileHash(isolationResultPath)}, sourceHashes: isolation.sourceHashes,
      mapSha256: isolation.mapSha256, archives: isolation.transformed.map(row => ({input: row.input, inputSha256: row.inputSha256, output: row.output, outputSha256: row.sha256}))},
    checks: {producerResultSha256Pinned: true, producerSourceRecipePreserved: true, exactOrderedArchiveGraph: true, archiveAndProofHashesRechecked: true, compilerInputHashesPreservedAsProducerEvidence: true}};
}

/** @param {string[]} args */
function main(args) {
  if (args.length !== 4 || args.some(value => !value)) {throw new Error("Usage: isolate-native-windows-jit.js <completed-sdk-root> <producer-result-sha256> <llvm-tools.json> <new-output>");}
  if (process.platform !== "linux" || process.arch !== "x64" || process.env["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows JIT archive isolation requires Linux x64 and MOM_AOT_BUILD_ONLY=1");}
  const [sdkRoot, expectedResultSha256, toolsFile, output] = /** @type {[string,string,string,string]} */ (args);
  const prior = inspectWindowsJitSdk(sdkRoot, expectedResultSha256, toolsFile); validateFreshOutput(prior, toolsFile, output);
  const environment = nativeSdkEnvironment(process.env);
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows JIT isolation requires a build-only Linux environment");}
  for (const key of Object.keys(process.env)) {if (!Object.hasOwn(environment, key)) {delete process.env[key];}}
  Object.assign(process.env, environment);
  const isolation = isolateCoffCompilerArchives({archives: prior.archives, tools: prior.tools, output, prefix: "mom_dpcpp23_", entryPoints: dpcppEntries});
  const current = inspectWindowsJitSdk(sdkRoot, expectedResultSha256, toolsFile);
  if (!isDeepStrictEqual(prior, current) || fileHash(__filename) !== priorAdapterSha256) {throw new Error("Windows JIT archive producer, inputs, tools or adapter changed during isolation");}
  const isolationResultPath = path.join(output, "result.json"); canonicalPath(isolationResultPath, "file", "COFF isolation result", 8 << 20);
  if (isolation.version !== 1 || Object.hasOwn(isolation, "kind") || isolation.buildOnly !== true || isolation.executed !== false ||
      isolation.publicationQualified !== false || !isDeepStrictEqual(isolation.transformed.map(row => row.input), prior.archives)) {throw new Error("COFF isolation did not preserve the complete producer archive closure");}
  const handoff = windowsJitIsolationHandoff(prior, isolation, toolsFile, output);
  fs.writeFileSync(path.join(output, "handoff.json"), JSON.stringify(handoff, null, 2) + "\n", {flag: "wx"});
  process.stdout.write("Windows JIT compiler archives isolated; producer receipt preserved; build-only, not executed or publication-qualified.\n");
}

const priorAdapterSha256 = fileHash(__filename);
module.exports = {inspectWindowsJitSdk, windowsJitIsolationHandoff, validateFreshOutput, validateProducerReceipt, validateArchiveGraph, validateStageAncestry,
  verifyRecordedArtifacts, validateToolsManifest, validateRecipeRecord, readPinnedProducerResult, hashTools};
if (require.main === module) {try {main(process.argv.slice(2));} catch (error) {console.error(error instanceof Error ? error.message : "Windows JIT archive isolation failed"); process.exitCode = 1;}}
