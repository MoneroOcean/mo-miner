#!/usr/bin/env node
"use strict";

// Build-only orchestration for the maintained NVRTC image worker. The selected host C++
// driver and every SDK file are explicit inputs; this pipeline never searches for CUDA.
const fs = require("node:fs"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {fileHash, readBuildJson} = require("./build-source-inputs");
const {cudaHeaderInputs, collectHeaders, packHeaders} = require("./native-jit-headers");
const {planCudaImages, generateCudaImageBundle, cudaImageInputs} = require("./native-cuda-images");

const maxCompilers = 16, maxInputFiles = 4096, maxJobs = 2048;
const maxImageBytes = 64 << 20, maxCommandOutput = 4 << 20, maxDiagnosticBytes = 4096;
const defaultTimeoutMs = 900000, maxTimeoutMs = 1800000;
const maintainedInputs = ["scripts/build-source-inputs.js", "scripts/native-cuda-images.cpp",
  "scripts/native-cuda-images.js", "scripts/native-jit-headers.js", "scripts/native-resources.js",
  "sycl/cn_gpu/cuda_expand_kernel.inc", "sycl/pearlhash/cuda_kernel.inc", "sycl/pearlhash/cuda_compact_kernel.inc"];

/** @typedef {{command: string, args: string[], inputs: Record<string, string>, environment?: Record<string, string>}} WorkerRecipe */
/** @typedef {{worker: WorkerRecipe, headerPack: string}} CompilerRecipe */
/** @typedef {{repo: string, output: string, compilers: CompilerRecipe[], timeoutMs: number}} BuildConfig */
/** @typedef {{major: number, minor: number, supportedArchs: number[], inputs: Record<string, string>}} CompilerQuery */
/** @typedef {{status: number | null, signal: NodeJS.Signals | null, error: Error | undefined, stdout: string, stderr: string}} ProcessResult */
/** @typedef {{recipe: CompilerRecipe, directory: string, worker: string, command: string,
 * external: Map<string, string>, header: ReturnType<typeof cudaHeaderInputs>, roots: string[], env: NodeJS.ProcessEnv,
 * query?: CompilerQuery, inputs?: Record<string, string>}} BuildState */

/** @param {unknown} value @returns {value is Record<string, unknown>} */
function record(value) {return value !== null && typeof value === "object" && !Array.isArray(value);}

/** @param {unknown} value @param {string} label @returns {string} */
function absolutePath(value, label) {
  if (typeof value !== "string" || !path.isAbsolute(value) || /[\0\r\n]/.test(value)) {
    throw new Error(`${label} must be an absolute path`);
  }
  return path.resolve(value);
}

/** @param {unknown} value @param {string} label @param {number} maximum @returns {string[]} */
function stringList(value, label, maximum) {
  if (!Array.isArray(value) || !value.length || value.length > maximum ||
      value.some((entry) => typeof entry !== "string" || !entry || /[\0\r\n]/.test(entry))) {
    throw new Error(`${label} must be a non-empty bounded string list`);
  }
  return /** @type {string[]} */ (value);
}

/** @param {unknown} value @returns {Record<string, string> | undefined} */
function workerEnvironment(value) {
  if (value === undefined) {return undefined;}
  if (!record(value) || Object.keys(value).length > 3) {throw new Error("Worker environment may only set compiler/runtime search paths");}
  /** @type {Record<string, string>} */
  const result = {};
  for (const [key, entry] of Object.entries(value)) {
    if (!["PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"].includes(key) || typeof entry !== "string" ||
        !entry || /[\0\r\n]/.test(entry) || entry.split(path.delimiter).some((root) => !root || !path.isAbsolute(root))) {
      throw new Error("Worker environment may only set absolute compiler/runtime search roots");
    }
    result[key] = entry;
  }
  return result;
}

/** @param {unknown} value @returns {BuildConfig} */
function validateConfig(value) {
  if (!record(value)) {throw new Error("CUDA image build config must be an object");}
  const repo = absolutePath(value["repo"], "Repository root"), output = absolutePath(value["output"], "Output directory");
  if (!fs.statSync(repo).isDirectory() || path.basename(output) === "" || output === repo) {
    throw new Error("CUDA image build needs an existing source root and a separate output directory");
  }
  if (!Array.isArray(value["compilers"]) || !value["compilers"].length ||
      value["compilers"].length > maxCompilers) {
    throw new Error("CUDA image build config needs a directory output and 1–16 compiler recipes");
  }
  const timeoutMs = value["timeoutMs"] === undefined ? defaultTimeoutMs : value["timeoutMs"];
  if (typeof timeoutMs !== "number" || !Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > maxTimeoutMs) {
    throw new Error("CUDA worker timeout is outside its 1–1800 second bound");
  }
  const compilers = value["compilers"].map((entry, index) => {
    if (!record(entry) || !record(entry["worker"])) {throw new Error(`Compiler recipe ${index} is incomplete`);}
    const command = absolutePath(entry["worker"]["command"], `Compiler ${index} host tool`);
    const args = stringList(entry["worker"]["args"], `Compiler ${index} host argv`, 256);
    const rawInputs = entry["worker"]["inputs"];
    if (!record(rawInputs) || Object.keys(rawInputs).length < 1 || Object.keys(rawInputs).length > maxInputFiles) {
      throw new Error(`Compiler ${index} inputs must map absolute files to SHA-256 digests`);
    }
    const inputEntries = Object.entries(rawInputs);
    const inputs = Object.fromEntries(inputEntries.map(([file, hash]) => {
      if (typeof hash !== "string" || !/^[0-9a-f]{64}$/.test(hash)) {throw new Error(`Compiler ${index} input digest is invalid`);}
      return [absolutePath(file, `Compiler ${index} input`), hash];
    }));
    const sourceMarkers = args.reduce((count, arg) => count + arg.split("{source}").length - 1, 0);
    const outputMarkers = args.reduce((count, arg) => count + arg.split("{output}").length - 1, 0);
    if (sourceMarkers !== 1 || outputMarkers !== 1 ||
        !args.some((arg) => arg === "-std=c++20" || arg === "/std:c++20") ||
        Object.keys(inputs).length !== inputEntries.length) {
      throw new Error(`Compiler ${index} argv must contain one source/output placeholder, C++20, and unique SDK inputs`);
    }
    const environment = workerEnvironment(entry["worker"]["environment"]);
    const worker = {command, args, inputs, ...(environment ? {environment} : {})};
    return {worker, headerPack: absolutePath(entry["headerPack"], `Compiler ${index} CUDA header pack`)};
  });
  return {repo, output, compilers, timeoutMs};
}

/** Resolve a configured file once, then inventory its canonical regular-file path.
 * @param {string} file @param {string} label @returns {string} */
function regularFile(file, label) {
  const real = fs.realpathSync(file), stat = fs.statSync(real);
  if (!stat.isFile()) {throw new Error(`${label} is not a regular file: ${file}`);}
  return real;
}

/** Hash a fixed list of explicit absolute files; no directory walk is performed.
 * @param {string[]} files @returns {Map<string, string>} */
function snapshotFiles(files) {
  const unique = [...new Set(files.map((file) => regularFile(file, "CUDA build input")))].sort();
  if (unique.length > maxInputFiles) {throw new Error("CUDA worker input inventory exceeds its bound");}
  return new Map(unique.map((file) => [file, fileHash(file)]));
}

/** Verify declared compiler inputs against their config pins before any worker command runs.
 * @param {Record<string, string>} declared @param {string} label @returns {Map<string, string>} */
function snapshotDeclaredInputs(declared, label) {
  const snapshot = snapshotFiles(Object.keys(declared));
  for (const [file, expected] of Object.entries(declared)) {
    const real = regularFile(file, label);
    if (snapshot.get(real) !== expected) {throw new Error(`${label} digest differs from its configured SHA-256: ${real}`);}
  }
  return snapshot;
}

/** @param {ReadonlyMap<string, string>} before @param {string} label */
function verifySnapshot(before, label) {
  for (const [file, expected] of before) {
    if (fileHash(file) !== expected) {throw new Error(`${label} changed during CUDA image production: ${file}`);}
  }
}

/** @param {string[]} roots @param {Record<string, unknown>} manifest */
function verifyHeaderRoots(roots, manifest) {
  const entries = collectHeaders(roots, true), packed = packHeaders(entries);
  const digest = require("node:crypto").createHash("sha256").update(packed).digest("hex");
  if (digest !== manifest["sha256"] || entries.length !== manifest["count"] || packed.length !== manifest["bytes"]) {
    throw new Error("Live CUDA filesystem include roots differ from the admitted header pack");
  }
}

/** @param {string} file @param {string} contents */
function writeExclusive(file, contents) {fs.writeFileSync(file, contents, {flag: "wx"});}

/** @param {string} file @param {unknown} value */
function writeJsonExclusive(file, value) {writeExclusive(file, JSON.stringify(value, null, 2) + "\n");}

/** @param {string} label @param {ProcessResult} result @returns {string} */
function processDiagnostic(label, result) {
  const detail = [result.error?.message, result.signal ? `signal=${result.signal}` : null,
    result.status === null ? "exit status unavailable" : `exit=${result.status}`,
    result.stderr, result.stdout].filter(Boolean).join("\n").trim();
  return `${label}: ${detail.slice(0, maxDiagnosticBytes) || "command failed"}`;
}

/** @param {string} command @param {string[]} args @param {NodeJS.ProcessEnv} env @param {number} timeoutMs @returns {ProcessResult} */
function runProcess(command, args, env, timeoutMs) {
  const result = spawnSync(command, args, {encoding: "utf8", env, timeout: timeoutMs, maxBuffer: maxCommandOutput});
  return {status: result.status, signal: result.signal, error: result.error, stdout: result.stdout || "", stderr: result.stderr || ""};
}

/** @param {ProcessResult} result @param {string} label */
function requireResolved(result, label) {
  if (result.error || result.signal || result.status === null) {
    throw new Error(`Unresolved CUDA worker lifecycle; stopped launching commands. ${processDiagnostic(label, result)}`);
  }
}

/** @param {string} directory @param {string} label @param {string} command @param {string[]} args
 * @param {NodeJS.ProcessEnv} env @param {number} timeoutMs @returns {ProcessResult} */
function runWithEvidence(directory, label, command, args, env, timeoutMs) {
  writeJsonExclusive(path.join(directory, `${label}.command.json`), {command, args});
  const result = runProcess(command, args, env, timeoutMs);
  const log = [result.stdout, result.stderr, result.error?.message || ""].filter(Boolean).join("\n");
  writeExclusive(path.join(directory, `${label}.log`), log);
  requireResolved(result, label);
  return result;
}

/** @param {{major: number, minor: number}} value @returns {string} */
function versionKey(value) {return `${value.major}.${value.minor}`;}

/** @param {unknown} value @returns {CompilerQuery} */
function queryRecord(value) {
  if (!record(value) || value["status"] !== "ok") {
    throw new Error("NVRTC worker --query did not return a valid public compiler query");
  }
  const nvrtc = value["nvrtc"];
  if (!record(nvrtc) || typeof nvrtc["major"] !== "number" || typeof nvrtc["minor"] !== "number" ||
      !Array.isArray(value["supportedArchs"])) {throw new Error("NVRTC worker --query did not return a valid public compiler query");}
  const supportedArchs = /** @type {number[]} */ (value["supportedArchs"]);
  return {major: nvrtc["major"], minor: nvrtc["minor"], supportedArchs, inputs: {}};
}

/** @param {unknown} value @param {string} label */
function parseJsonOutput(value, label) {
  const text = String(value).trim();
  if (!text || text.length > (1 << 20)) {throw new Error(`${label} returned empty or oversized JSON`);}
  try {return JSON.parse(text);} catch (error) {throw new Error(`${label} did not return JSON`, {cause: error});}
}

/** @param {string} repo @returns {Record<string, string>} */
function maintainedSnapshot(repo) {
  return Object.fromEntries(maintainedInputs.map((relative) => [relative, fileHash(path.join(repo, relative))]));
}

/** @param {Record<string, string>} before @param {string} repo */
function verifyMaintainedSnapshot(before, repo) {
  const after = maintainedSnapshot(repo);
  if (JSON.stringify(before) !== JSON.stringify(after)) {throw new Error("Maintained CUDA source changed while building the image worker");}
}

/** Build all explicitly selected compiler workers, query their public NVRTC APIs, compile the
 * queried image plan, and publish a no-overwrite include plus provenance sidecar.
 * @param {unknown} input @returns {{include: string, sidecar: string,
 * manifest: {compilers: Array<{major: number, minor: number}>, images: unknown[], resourceBytes: number, sourceSha256: string}}} */
function buildCudaImages(input) {
  const config = validateConfig(input), sidecar = path.join(config.output, "cuda-images.inc.json");
  const outputInclude = path.join(config.output, "cuda-images.inc");
  if (fs.existsSync(config.output)) {throw new Error(`CUDA image output already exists; refusing overwrite: ${config.output}`);}
  const parent = path.dirname(config.output);
  if (!fs.statSync(parent).isDirectory()) {throw new Error("CUDA image output parent must already be a directory");}
  fs.mkdirSync(config.output, {mode: 0o700});
  const source = path.join(config.repo, "scripts/native-cuda-images.cpp"), sourceBefore = maintainedSnapshot(config.repo);
  /** @type {BuildState[]} */
  const states = [];
  /** @type {string[]} */
  const errors = [];
  for (const [index, recipe] of config.compilers.entries()) {
    const stage = path.join(config.output, `compiler-${String(index).padStart(2, "0")}`);
    fs.mkdirSync(stage, {mode: 0o700});
    const commandInput = regularFile(recipe.worker.command, `Compiler ${index} host tool`);
    // Execute and record the exact canonical file whose bytes were admitted above.
    const command = commandInput;
    const header = cudaHeaderInputs(recipe.headerPack);
    if (!Array.isArray(header.manifest["roots"]) || !header.manifest["roots"].length ||
        header.manifest["roots"].length > 16 || header.manifest["roots"].some((root) => typeof root !== "string" || !path.isAbsolute(root))) {
      throw new Error(`Compiler ${index} header receipt has no bounded absolute filesystem roots`);
    }
    const roots = /** @type {string[]} */ (header.manifest["roots"]);
    verifyHeaderRoots(roots, header.manifest);
    const declaredInputs = snapshotDeclaredInputs(recipe.worker.inputs, `Compiler ${index} configured SDK input`);
    const external = new Map(declaredInputs);
    for (const [file, hash] of snapshotFiles([commandInput, ...Object.keys(header.files),
      path.join(config.repo, "scripts/build-native-cuda-images.js")])) {
      if (external.has(file) && external.get(file) !== hash) {throw new Error(`Compiler ${index} input changed while its inventory was captured: ${file}`);}
      external.set(file, hash);
    }
    const worker = path.join(stage, process.platform === "win32" ? "native-cuda-images.exe" : "native-cuda-images");
    const environment = {...process.env, ...(recipe.worker.environment || {})};
    const state = /** @type {BuildState} */ ({recipe, directory: stage, worker, command, external, header, roots, env: environment});
    states.push(state);
    const args = recipe.worker.args.map((arg) => arg.replaceAll("{source}", source).replaceAll("{output}", worker));
    const build = runWithEvidence(stage, "build", command, args, environment, config.timeoutMs);
    if (build.status !== 0) {
      errors.push(processDiagnostic(`NVRTC worker ${index} build`, build));
      continue;
    }
    regularFile(worker, `Compiler ${index} worker output`);
    verifySnapshot(external, `Compiler ${index} host/SDK input`);
    verifyHeaderRoots(roots, header.manifest);
    const query = runWithEvidence(stage, "query", worker, ["--query"], environment, config.timeoutMs);
    if (query.status !== 0) {
      errors.push(processDiagnostic(`NVRTC worker ${index} query`, query));
      continue;
    }
    try {state.query = queryRecord(parseJsonOutput(query.stdout, `NVRTC worker ${index} query`));} catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
    }
  }
  const queriedStates = /** @type {Array<BuildState & {query: CompilerQuery}>} */ (states.filter((state) => state.query));
  if (!queriedStates.length) {
    if (errors.length) {throw new Error(`CUDA worker setup failed:\n${errors.map((error) => `- ${error}`).join("\n")}`);}
    throw new Error("No selected compiler returned a public NVRTC query");
  }
  const queries = queriedStates.map((state) => state.query);
  /** @type {ReturnType<typeof planCudaImages>} */
  let plan;
  try {plan = planCudaImages(/** @type {CompilerQuery[]} */ (queries));} catch (error) {
    throw new Error(`NVRTC query planning failed: ${error instanceof Error ? error.message : String(error)}`, {cause: error});
  }
  if (plan.length > maxJobs) {throw new Error("Queried CUDA image plan exceeds the build cap");}
  const hasBaseline = queriedStates.some(({query}) => query.major === 12 && query.minor === 6 && query.supportedArchs.includes(80));
  const modern = queriedStates.some(({query}) =>
    (query.major > 13 || (query.major === 13 && query.minor >= 3)) && query.supportedArchs.includes(120));
  if (!hasBaseline || !modern) {
    throw new Error("Release image generation requires actual NVRTC 12.6 compute_80 and NVRTC 13.3-or-newer SM120 queries");
  }
  const byVersion = new Map(queriedStates.map((state) => [versionKey(state.query), state]));
  const headerPacks = Object.fromEntries(queriedStates.map((state) => [versionKey(state.query), state.recipe.headerPack]));
  const compilers = queriedStates.map((state) => {
    const query = state.query;
    const inputs = new Map(state.external);
    const workerSha = fileHash(state.worker);
    inputs.set(state.worker, workerSha);
    const recipePath = path.join(state.directory, "recipe.json");
    const recipeFile = {format: "mom-cuda-image-worker-v1", command: state.command, args: state.recipe.worker.args,
      worker: state.worker, workerSha256: workerSha, headerPack: state.recipe.headerPack,
      environment: state.recipe.worker.environment || {},
      roots: state.roots, query: {major: query.major, minor: query.minor, supportedArchs: query.supportedArchs},
      maintainedSources: sourceBefore, declaredInputs: state.recipe.worker.inputs, inputs: Object.fromEntries(state.external)};
    writeJsonExclusive(recipePath, recipeFile);
    inputs.set(recipePath, fileHash(recipePath));
    state.inputs = Object.fromEntries(inputs);
    return {...query, inputs: state.inputs};
  });
  const completed = new Set();
  /** @type {Array<{job: unknown, metadata: unknown, data: Buffer}>} */
  const artifacts = [];
  let rawBytes = 0;
  let suppressedErrors = 0;
  for (const [planIndex, job] of plan.entries()) {
    const version = versionKey(job.compiler), state = byVersion.get(version);
    if (!state?.query) {errors.push(`No queried worker for NVRTC ${version}; skipping ${job.profile} ${job.target}`); continue;}
    const roots = job.profile.startsWith("pearl-") ? state.roots : [];
    const artifact = path.join(state.directory, "images", `${String(planIndex).padStart(4, "0")}.${job.format}`);
    const imageDirectory = path.dirname(artifact);
    if (!fs.existsSync(imageDirectory)) {fs.mkdirSync(imageDirectory, {mode: 0o700});}
    const args = [job.profile, job.target, job.format, artifact, ...roots.flatMap((root) => ["--include", root])];
    const label = `image-${String(planIndex).padStart(4, "0")}`;
    const result = runWithEvidence(state.directory, label, state.worker, args, state.env, config.timeoutMs);
    const key = `${job.profile}|${job.target}|${job.format}|${job.role}|${version}`;
    if (result.status !== 0) {
      if (errors.length < 64) {errors.push(processDiagnostic(key, result));} else {suppressedErrors++;}
      continue;
    }
    try {
      const metadata = parseJsonOutput(result.stdout, key), stat = fs.lstatSync(artifact);
      if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error(`${key} did not create a regular image file`);}
      const size = fs.statSync(artifact).size;
      if (!Number.isSafeInteger(size) || size < 1 || size > maxImageBytes || size > maxImageBytes - rawBytes) {
        throw new Error(`${key} outputs exceed the 64 MiB image data bound`);
      }
      const data = fs.readFileSync(artifact);
      rawBytes += data.length; artifacts.push({job, metadata, data}); completed.add(key);
      writeJsonExclusive(path.join(state.directory, `${label}.result.json`), {job, output: artifact, bytes: data.length});
    } catch (error) {
      if (errors.length < 64) {errors.push(`${key}: ${error instanceof Error ? error.message : String(error)}`);} else {suppressedErrors++;}
    }
  }
  if (suppressedErrors) {errors.push(`${suppressedErrors} additional CUDA image failures omitted`);}
  if (completed.size !== plan.length || errors.length) {
    writeJsonExclusive(path.join(config.output, "failures.json"), {planned: plan.length, completed: completed.size, errors});
    throw new Error(`CUDA image compilation failed (${completed.size}/${plan.length} images):\n${errors.map((error) => `- ${error}`).join("\n")}`);
  }
  for (const state of queriedStates) {
    verifySnapshot(state.external, `NVRTC ${versionKey(state.query)} host/SDK input`);
    verifyHeaderRoots(state.roots, state.header.manifest);
    if (fileHash(state.worker) !== state.inputs?.[state.worker]) {throw new Error("NVRTC worker binary changed during image generation");}
  }
  verifyMaintainedSnapshot(sourceBefore, config.repo);
  const bundle = generateCudaImageBundle({repo: config.repo, compilers, artifacts, headerPacks});
  if (JSON.stringify(bundle.manifest.sources) !== JSON.stringify(sourceBefore)) {
    throw new Error("Maintained sources changed between worker build and CUDA image admission");
  }
  for (const state of queriedStates) {
    verifySnapshot(state.external, `NVRTC ${versionKey(state.query)} host/SDK input`);
    verifyHeaderRoots(state.roots, state.header.manifest);
    if (fileHash(state.worker) !== state.inputs?.[state.worker]) {throw new Error("NVRTC worker binary changed during image admission");}
  }
  const manifestText = JSON.stringify(bundle.manifest, null, 2) + "\n";
  let wroteInclude = false;
  try {
    writeExclusive(outputInclude, bundle.source); wroteInclude = true;
    writeExclusive(sidecar, manifestText);
    const admitted = cudaImageInputs(outputInclude, config.repo);
    if (admitted.manifest["sourceSha256"] !== fileHash(outputInclude)) {throw new Error("Published CUDA image sidecar failed admission");}
  } catch (error) {
    if (wroteInclude) {fs.unlinkSync(outputInclude);}
    throw error;
  }
  writeJsonExclusive(path.join(config.output, "result.json"), {format: "mom-cuda-image-build-v1", include: outputInclude,
    sidecar, compilerVersions: queriedStates.map((state) => versionKey(state.query)),
    plannedImages: plan.length, embeddedBytes: bundle.manifest.resourceBytes, sourceSha256: bundle.manifest.sourceSha256});
  return {include: outputInclude, sidecar, manifest: bundle.manifest};
}

/** @param {string[]} argv */
function main(argv) {
  if (argv.length !== 1 || !path.isAbsolute(argv[0] || "")) {
    throw new Error("Usage: node scripts/build-native-cuda-images.js <absolute-config.json>");
  }
  const result = buildCudaImages(readBuildJson(argv[0] || ""));
  process.stdout.write(JSON.stringify({include: result.include, sidecar: result.sidecar,
    compilerVersions: result.manifest.compilers.map(versionKey), plannedImages: result.manifest.images.length,
    embeddedBytes: result.manifest.resourceBytes}) + "\n");
}

module.exports = {buildCudaImages, validateConfig};
if (require.main === module) {
  try {main(process.argv.slice(2));} catch (error) {
    console.error(error instanceof Error ? error.message : error); process.exitCode = 1;
  }
}
