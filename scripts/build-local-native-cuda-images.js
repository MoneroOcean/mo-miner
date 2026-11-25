"use strict";

const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk"), ci = require("./native-ci-linux-cuda");
const {auditDpcppCudaInputs} = require("./native-dpcpp-cuda-inputs");
const {nvrtcSdkInputs} = require("./prepare-native-nvrtc-sdk");
const {cudaImageInputs} = require("./native-cuda-images");
const {verifyHeaderRoots} = require("./build-native-cuda-images");
const {cudaHeaderInputs} = require("./native-jit-headers");
const {fileHash, readBuildJson, buildFileInputs, sourceInputs, verifySourceInputs} = require("./build-source-inputs");
const repo = fs.realpathSync(path.resolve(__dirname, ".."));

/** Keep the two actual compiler cohorts distinct without introducing CI transport paths.
 * @param {string} workspace */
function localCudaImagePaths(workspace) {
  if (!path.isAbsolute(workspace) || path.resolve(workspace) !== workspace || /[\r\n\0]/.test(workspace)) {
    throw new Error("Local CUDA image workspace must be canonical and absolute");
  }
  const root = path.join(workspace, "cuda-images"), headerRoot = path.join(root, "headers"),
    cutlassRoot = path.join(root, "cutlass"), output = path.join(root, "bundle"),
    handoffRoot = path.join(workspace, "cuda-images-handoff"), noticeRoot = path.join(handoffRoot, "notice-sources");
  return {root, headerRoot, output, cutlassRoot, cutlassInclude: path.join(cutlassRoot, "include"),
    cutlassMarker: path.join(cutlassRoot, ".mom-version"), cutlassLicense: path.join(cutlassRoot, "LICENSE.txt"),
    header126: path.join(headerRoot, "cuda-12.6.inc"), header133: path.join(headerRoot, "cuda-13.3.inc"),
    config: path.join(root, "worker-config.json"), result: path.join(output, "result.json"),
    source126: path.join(workspace, "nvrtc-12.6-source"), sdk126: path.join(workspace, "nvrtc-12.6-sdk"),
    linux: {source: path.join(workspace, "nvrtc-13.3-source"), sdkRoot: path.join(workspace, "nvrtc-13.3-sdk")},
    handoffRoot, noticeRoot, noticeManifest: path.join(noticeRoot, "manifest.json"),
    dpcpp: local.localDpcppCudaState(local.supportPaths(workspace)),
    admission: path.join(workspace, "cuda-image-admission.json")};
}

function localCudaImageRecipe() {
  return sourceInputs(repo, [], ["scripts/build-local-native-cuda-images.js", "scripts/build-native-cuda-images.js",
    "scripts/native-ci-linux-cuda.js", "scripts/native-dpcpp-cuda-inputs.js", "scripts/install-cutlass.sh",
    "scripts/prepare-native-nvrtc-sdk.js", "scripts/fetch-native-sdk-sources.js", "scripts/probe-static-nvrtc.js"]);
}

/** The public bundle reader intentionally avoids producer paths for relocation. A warm local
 * cache additionally rechecks those original compiler/header inputs before reusing the bundle.
 * @param {unknown} rows @param {string} compiler */
function verifyLocalCudaCompilerInputs(rows, compiler) {
  if (!Array.isArray(rows) || rows.length !== 2 || rows.map(row => `${row?.major}.${row?.minor}`).sort().join(",") !== "12.6,13.3") {
    throw new Error("Local CUDA images require the exact 12.6/13.3 compiler matrix");
  }
  /** @type {Record<string,string>} */ const inputs = {};
  for (const row of rows) {
    if (!row.inputs || typeof row.inputs !== "object" || Array.isArray(row.inputs) ||
        !Object.keys(row.inputs).length || Object.keys(row.inputs).length > 4096 ||
        !Object.hasOwn(row.inputs, compiler) || row.inputs[compiler] !== fileHash(compiler)) {
      throw new Error("CUDA image compiler inputs omit the selected host compiler");
    }
    const current = buildFileInputs(Object.keys(row.inputs));
    if (!isDeepStrictEqual(row.inputs, current)) {throw new Error("Local CUDA compiler or header inputs changed");}
    for (const [file, sha256] of Object.entries(current)) {
      if (inputs[file] && inputs[file] !== sha256) {throw new Error("CUDA compiler cohorts disagree on a shared input");}
      inputs[file] = sha256;
    }
  }
  return inputs;
}

/** Rebuild the exact bounded header inventory the producer admitted from its live roots.
 * @param {ReturnType<typeof localCudaImagePaths>} state */
function verifyLocalCudaHeaderRoots(state) {
  const cudaInclude = path.join(state.dpcpp.cudaRoot, "include"), roots = [state.cutlassInclude, cudaInclude];
  if (roots.some(root => fs.realpathSync(root) !== root)) {throw new Error("CUDA image header roots must be canonical directories");}
  const expectedRoots = new Set(roots), nestedCccl = path.join(cudaInclude, "cccl");
  if (fs.existsSync(path.join(nestedCccl, "cuda/std/cstdint"))) {expectedRoots.add(fs.realpathSync(nestedCccl));}
  for (const pack of [state.header126, state.header133]) {
    const admitted = cudaHeaderInputs(pack).manifest;
    const admittedRoots = admitted.roots;
    if (!Array.isArray(admittedRoots) || admittedRoots.length !== expectedRoots.size ||
        admittedRoots.some(root => typeof root !== "string" || !expectedRoots.has(root))) {
      throw new Error("CUDA image header pack does not bind the current same-run include roots");
    }
    verifyHeaderRoots(admittedRoots, admitted);
  }
}

/** Re-admit original producers and bind both transported and original image bytes.
 * @param {ReturnType<typeof localCudaImagePaths>} state @param {string} compiler */
function auditLocalCudaImages(state, compiler) {
  for (const root of [state.root, state.headerRoot, state.output, state.handoffRoot, state.noticeRoot]) {
    const stat = fs.lstatSync(root);
    if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(root) !== root) {
      throw new Error("Local CUDA image output escaped its canonical workspace: " + root);
    }
  }
  for (const file of [state.result, path.join(state.handoffRoot, "result.json"), state.config]) {
    const stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== file) {
      throw new Error("Local CUDA image receipt must be a canonical regular file: " + file);
    }
  }
  verifyLocalCudaHeaderRoots(state);
  const dpcpp = auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest)),
    baseline = nvrtcSdkInputs("linux", state.sdk126, {cohort: "linux-12.6"}),
    modern = nvrtcSdkInputs("linux", state.linux.sdkRoot, {cohort: "linux"}),
    image = path.join(state.handoffRoot, "cuda-images.inc"), images = cudaImageInputs(image, repo),
    result = readBuildJson(state.result), copyResult = readBuildJson(path.join(state.handoffRoot, "result.json")),
    original = cudaImageInputs(path.join(state.output, "cuda-images.inc"), repo),
    inputs = verifyLocalCudaCompilerInputs(images.manifest["compilers"], compiler),
    noticePlan = ci.cudaNoticeSourcePlan(state, dpcpp, baseline, modern),
    notices = ci.auditCudaNoticeSources(state, dpcpp, images.manifest, noticePlan);
  if (!isDeepStrictEqual(images.manifest, original.manifest) || !isDeepStrictEqual(result, copyResult) ||
      result.format !== "mom-cuda-image-build-v1" || result.include !== path.join(state.output, "cuda-images.inc") ||
      result.sidecar !== result.include + ".json" || result.sourceSha256 !== images.manifest["sourceSha256"] ||
      !Array.isArray(images.manifest["images"]) || result.plannedImages !== images.manifest["images"].length ||
      result.compilerVersions?.slice().sort().join(",") !== "12.6,13.3") {
    throw new Error("Local CUDA image handoff differs from the actual producer result");
  }
  const files = [...Object.keys(inputs), ...Object.keys(images.files), ...Object.keys(original.files),
    ...Object.keys(baseline.files), ...Object.keys(modern.files), state.result, state.config,
    path.join(state.handoffRoot, "result.json"), state.dpcpp.manifest, notices.manifest,
    ...notices.files.map(row => row.path)];
  return {version: 1, recipe: localCudaImageRecipe(), fingerprints: buildFileInputs(files)};
}

/** Run this internal arm only after the actual child cgroup proves its cap. No GPU is used.
 * @param {string} workspace @param {string} compiler
 * @param {{verifyScope?:()=>unknown,build?:typeof ci.buildLinuxCudaImageTree,audit?:typeof auditLocalCudaImages}} [dependencies] */
function runLocalCudaImageChild(workspace, compiler, dependencies = {}) {
  (dependencies.verifyScope || local.verifyCurrentMemoryScope)();
  const stat = fs.lstatSync(workspace);
  if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(workspace) !== workspace) {throw new Error("CUDA image workspace must already be admitted");}
  local.admitWorkspace(workspace, repo);
  const expectedCompiler = fs.realpathSync("/opt/intel/oneapi/compiler/2026.2/bin/icpx");
  if (compiler !== expectedCompiler) {throw new Error("CUDA images require the pinned Intel 2026.2 host compiler");}
  const state = localCudaImagePaths(workspace), receipt = fs.lstatSync(state.admission, {throwIfNoEntry: false}),
    audit = dependencies.audit || auditLocalCudaImages;
  if (receipt) {
    if (!receipt.isFile() || receipt.isSymbolicLink() || receipt.nlink !== 1 || fs.realpathSync(state.admission) !== state.admission) {
      throw new Error("CUDA image admission must be a canonical regular file");
    }
    const saved = readBuildJson(state.admission);
    verifySourceInputs(saved.recipe, localCudaImageRecipe(), "Local CUDA image recipe");
    if (!isDeepStrictEqual(saved, audit(state, compiler))) {throw new Error("Local CUDA image cache changed; preserve it for diagnosis");}
    return;
  }
  (dependencies.build || ci.buildLinuxCudaImageTree)(state, compiler, {...process.env, MOM_AOT_BUILD_ONLY: "1"});
  fs.writeFileSync(state.admission, JSON.stringify(audit(state, compiler), null, 2) + "\n", {flag: "wx", mode: 0o600});
}

/** Invoke the shared image builder under the same enforced bound as the other SDK stages.
 * @param {string} workspace
 * @param {{tools?:ReturnType<typeof local.localTools>,run?:typeof local.runCommand,environment?:NodeJS.ProcessEnv}} [dependencies] */
function buildLocalCudaImages(workspace, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local CUDA image preparation requires Linux x64");}
  const env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"},
    tools = dependencies.tools || local.localTools(env),
    invoke = local.scopedInvoker(tools, env, dependencies.run || local.runCommand), state = localCudaImagePaths(workspace);
  local.admitWorkspace(workspace, repo);
  invoke("systemd memory-limit preflight", process.execPath, [__filename, "--memory-probe"]);
  invoke("Local CUDA image producer/admission", process.execPath, [__filename, "--local-cuda-images", workspace, fs.realpathSync(tools.cxx)]);
  return {workspace, handoffRoot: state.handoffRoot, image: path.join(state.handoffRoot, "cuda-images.inc")};
}

if (require.main === module) {
  try {
    const [command, workspace, compiler, extra] = process.argv.slice(2);
    if (command === "--memory-probe" && !workspace) {local.verifyCurrentMemoryScope();} else if (
      command === "--local-cuda-images" && workspace && compiler && !extra
    ) {runLocalCudaImageChild(workspace, compiler);} else {throw new Error("Internal local CUDA image producer; use the release build frontdoor");}
  } catch (error) {console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1;}
}

module.exports = {localCudaImagePaths, verifyLocalCudaCompilerInputs,
  verifyLocalCudaHeaderRoots, auditLocalCudaImages, runLocalCudaImageChild, buildLocalCudaImages};
