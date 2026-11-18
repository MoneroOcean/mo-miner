"use strict";

// Linux SDK prerequisite CI steps that need more than a plain seal: restoring the pinned source
// cohort, exporting admitted HIP headers and the ACPP frontend, sealing Linux JIT resources, and
// producing the Windows JIT resource handoff. Every input tar is read from `${RUNNER_TEMP}/incoming`.

const fs = require("node:fs");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const ciJit = require("../native-ci-linux-jit"),
  ciRocm = require("../native-ci-linux-rocm"),
  ciSources = require("../native-ci-linux-sources"),
  ciTar = require("../native-ci-tar"),
  dpcppInputs = require("../native-dpcpp-cuda-inputs"),
  {auditGnuHeaders} = require("../prepare-native-gnu-headers");
const {fileHash, readBuildJson} = require("../build-source-inputs");
const {canonicalFile, appendOutputs} = require("./seal");

const repo = path.resolve(__dirname, "../..");
const windowsTarget = "x86_64-w64-windows-gnu";
const linuxTarget = "x86_64-unknown-linux-gnu";
const windowsProducer = "scripts/native-windows-jit-resources.js";

/** @typedef {ReturnType<typeof ciSources.ciPaths>} CiPaths */

/** Restore the pinned original Linux source cohort for CUDA image admission, then drop its tar.
 * @param {CiPaths} paths @param {NodeJS.ProcessEnv} [environment] */
function restoreCudaImageSources(paths, environment = process.env) {
  const state = ciSources.linuxSourceStatePaths(paths);
  ciSources.dpcppSourceArchivePin(state.incomingTar, environment["MOM_NATIVE_LINUX_SOURCE_BUNDLE_SHA256"] || "");
  ciTar.extractState(state.incomingTar, "/tmp", [state.sourceName]);
  ciSources.verifyLinuxSources(paths, state, "/usr/bin/git");
  fs.unlinkSync(state.incomingTar);
}

/** Export the already admitted HIP headers into a small same-run tar without rebuilding ROCm.
 * @param {CiPaths} paths @returns {{bundle_sha256: string, handoff_sha256: string}} */
function exportRocmHipHeadersCi(paths) {
  const {exportRocmHipHeaders} = require("../native-rocm-hip-headers");
  const state = ciRocm.rocmComgrStatePaths(paths, paths.env), name = "native-rocm-hip-headers-" + paths.suffix,
    root = path.join("/tmp", name), archive = path.join(paths.runnerTemp, name + ".tar");
  exportRocmHipHeaders(state.hipHeadersRoot, root, paths.suffix);
  ciTar.packState(archive, "/tmp", [name]);
  return {bundle_sha256: fileHash(archive), handoff_sha256: fileHash(path.join(root, "handoff.json"))};
}

/** Export the ACPP frontend matching the installed LLVM SDK, admit it, and pack it without LLVM libraries.
 * @param {CiPaths} paths @returns {{frontend_bundle_sha256: string, frontend_handoff_sha256: string}} */
function exportAcppFrontendCi(paths) {
  const {exportLinuxLlvmAcppFrontend, linuxLlvmAcppFrontendInputs} = require("../native-linux-llvm-tools");
  const name = "native-linux-llvm-acpp-frontend-" + paths.suffix, parent = path.join(paths.runnerTemp, "acpp-frontend-export-" + paths.suffix);
  fs.mkdirSync(parent, {mode: 0o700});
  const bundle = path.join(parent, name), archive = path.join(paths.runnerTemp, name + ".tar");
  const produced = exportLinuxLlvmAcppFrontend(paths.output, bundle);
  linuxLlvmAcppFrontendInputs(bundle, {handoffSha256: produced.handoffSha256});
  ciTar.packState(archive, parent, [name]);
  return {frontend_bundle_sha256: fileHash(archive), frontend_handoff_sha256: produced.handoffSha256};
}

/** The Linux resource result must bind its unchanged build-only manifest by path, hash and file count.
 * @param {any} result @param {any} manifest @param {string} manifestFile @param {string} manifestSha256 */
function assertLinuxJitResourceResult(result, manifest, manifestFile, manifestSha256) {
  if (result.version !== 1 || result.target !== linuxTarget || result.buildOnly !== true ||
      result.executed !== false || result.publicationQualified !== false ||
      result.resources?.manifest !== manifestFile || result.resources.manifestSha256 !== manifestSha256 ||
      !Array.isArray(manifest.files) || manifest.files.length !== result.resources.files ||
      manifest.target !== linuxTarget || manifest.publicationQualified !== false) {
    throw new Error("Linux resource result does not bind its unchanged build-only resource manifest");
  }
}

/** Seal the unchanged Linux JIT resource tar together with its result and manifest receipts.
 * @param {Pick<CiPaths, "suffix" | "runnerTemp" | "linuxJitResourcesRoot">} paths
 * @returns {{resources_bundle_sha256: string, resources_manifest_sha256: string, resources_result_sha256: string}} */
function sealLinuxJitResources(paths) {
  const root = paths.linuxJitResourcesRoot, resultFile = path.join(root, "ci-result.json"),
    manifestFile = path.join(root, "resources", "manifest.json"),
    archive = path.join(paths.runnerTemp, "native-linux-jit-resources-" + paths.suffix + ".tar");
  canonicalFile(resultFile, "Linux JIT resource result", 64 << 20);
  canonicalFile(manifestFile, "Linux JIT resource manifest", 64 << 20);
  canonicalFile(archive, "Linux JIT resource bundle");
  const manifestSha256 = fileHash(manifestFile);
  assertLinuxJitResourceResult(readBuildJson(resultFile), readBuildJson(manifestFile), manifestFile, manifestSha256);
  ciTar.listState(archive, [path.basename(root)]);
  return {resources_bundle_sha256: fileHash(archive), resources_manifest_sha256: manifestSha256, resources_result_sha256: fileHash(resultFile)};
}

/** Read the nine upstream output pins the Windows JIT resource producer admits.
 * @param {NodeJS.ProcessEnv} environment */
function windowsJitInputPins(environment) {
  const names = /** @type {const} */ (["MOM_NATIVE_LINUX_SOURCE_BUNDLE_SHA256", "MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256",
    "MOM_NATIVE_LINUX_DPCPP_CUDA_MANIFEST_SHA256", "MOM_NATIVE_LINUX_JIT_RESOURCES_BUNDLE_SHA256",
    "MOM_NATIVE_LINUX_JIT_RESOURCES_RESULT_SHA256", "MOM_NATIVE_LINUX_JIT_RESOURCES_MANIFEST_SHA256",
    "MOM_NATIVE_WINDOWS_GNU_BUNDLE_SHA256", "MOM_NATIVE_WINDOWS_GNU_RESULT_SHA256", "MOM_NATIVE_WINDOWS_GNU_COPYRIGHTS_SHA256"]);
  const pins = Object.fromEntries(names.map(name => [name, environment[name] || ""]));
  if (Object.values(pins).some(value => !value)) {
    throw new Error("Same-run Linux source, DPC++/CUDA, Linux JIT, and Windows GNU output pins are required");
  }
  return /** @type {Record<typeof names[number], string>} */ (pins);
}

/** Verify one incoming tar against its external SHA-256 pin before any extraction.
 * @param {string} file @param {string} expected @param {string} label */
function pinnedArchive(file, expected, label) {
  if (!/^[a-f0-9]{64}$/.test(expected)) {throw new Error(label + " SHA-256 pin is missing");}
  return ciSources.dpcppSourceArchivePin(file, expected);
}

/** Restore one same-run root at its original /tmp path, refusing to overwrite an existing one.
 * @param {string} file @param {string} root */
function restoreRoot(file, root) {
  if (fs.existsSync(root)) {throw new Error("Refusing to overwrite same-run input root: " + root);}
  ciTar.extractState(file, "/tmp", [path.basename(root)]);
}

/** Admit the four pinned same-run inputs, run the public Windows JIT resource producer, re-admit
 * every input afterwards, and seal the produced root with a build-only CI result.
 * @param {CiPaths} paths @param {NodeJS.ProcessEnv} [environment]
 * @returns {{resources_bundle_sha256: string, resources_manifest_sha256: string, resources_result_sha256: string}} */
function produceWindowsJitResourcesCi(paths, environment = process.env) {
  const {windowsJitResourcePaths, admitWindowsJitNativeTools} = require("../native-windows-jit-resources");
  const suffix = paths.suffix, source = ciSources.linuxSourceStatePaths(paths, environment), dpcpp = ciSources.dpcppCudaInputsStatePaths(paths, environment),
    gnu = ciSources.gnuHeadersStatePaths(paths, environment, windowsTarget), linuxResources = ciJit.linuxJitResourcesStatePaths(paths, environment),
    output = windowsJitResourcePaths(suffix), incoming = path.join(paths.runnerTemp, "incoming");
  const sourceTar = path.join(incoming, path.basename(source.sourceTar)), dpcppTar = path.join(incoming, path.basename(dpcpp.sdkTar)),
    linuxResourcesTar = path.join(incoming, "native-linux-jit-resources-" + suffix + ".tar"),
    windowsGnuTar = path.join(incoming, "native-windows-gnu-headers-" + suffix + ".tar");
  const pins = windowsJitInputPins(environment);
  const archivePins = {
    source: pinnedArchive(sourceTar, pins.MOM_NATIVE_LINUX_SOURCE_BUNDLE_SHA256, "Linux source"),
    dpcpp: pinnedArchive(dpcppTar, pins.MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256, "DPC++/CUDA inputs"),
    linuxResources: pinnedArchive(linuxResourcesTar, pins.MOM_NATIVE_LINUX_JIT_RESOURCES_BUNDLE_SHA256, "Linux JIT resources"),
    windowsGnu: pinnedArchive(windowsGnuTar, pins.MOM_NATIVE_WINDOWS_GNU_BUNDLE_SHA256, "Windows GNU headers"),
  };
  restoreRoot(sourceTar, source.source);
  restoreRoot(dpcppTar, dpcpp.root);
  restoreRoot(linuxResourcesTar, linuxResources.root);
  restoreRoot(windowsGnuTar, gnu.root);
  const git = fs.realpathSync("/usr/bin/git");
  ciSources.verifyLinuxSources(paths, source, git);
  const dpcppPins = ciSources.dpcppCudaInputPins(environment);
  if (dpcppPins.bundleSha256 !== archivePins.dpcpp) {throw new Error("DPC++/CUDA inputs lack matching external pins");}
  dpcppInputs.auditDpcppCudaInputs(dpcpp, readBuildJson(dpcpp.manifest), dpcppPins.manifestSha256);
  const linuxResult = path.join(linuxResources.root, "ci-result.json"), linuxManifest = path.join(linuxResources.root, "resources", "manifest.json");
  const linuxResultSha = fileHash(linuxResult), linuxManifestSha = fileHash(linuxManifest);
  if (linuxResultSha !== pins.MOM_NATIVE_LINUX_JIT_RESOURCES_RESULT_SHA256 ||
      linuxManifestSha !== pins.MOM_NATIVE_LINUX_JIT_RESOURCES_MANIFEST_SHA256 || fs.existsSync(output.root)) {
    throw new Error("Extracted Linux JIT resource result or manifest differs from its upstream output pin");
  }
  const copyrights = path.join(gnu.root, "copyrights.json"), gnuResult = path.join(gnu.root, "result.json");
  const gnuResultSha = fileHash(gnuResult), copyrightsSha = fileHash(copyrights);
  if (gnuResultSha !== pins.MOM_NATIVE_WINDOWS_GNU_RESULT_SHA256 || copyrightsSha !== pins.MOM_NATIVE_WINDOWS_GNU_COPYRIGHTS_SHA256) {
    throw new Error("Extracted Windows GNU result or copyrights differ from their upstream output pins");
  }
  const gnuAudit = auditGnuHeaders(gnu, windowsTarget);
  const args = [path.join(repo, windowsProducer), "produce", suffix, linuxResources.root, linuxResultSha,
    gnu.root, gnuResultSha, copyrightsSha, dpcpp.dpcppRoot, dpcppPins.bundleSha256, dpcppPins.manifestSha256, output.root];
  const started = spawnSync(process.execPath, args, {cwd: repo, env: environment, stdio: "inherit", timeout: 20 * 60 * 1000});
  if (started.error || started.signal || started.status !== 0) {throw new Error("Public Windows JIT resource producer failed", {cause: started.error});}
  ciSources.verifyLinuxSources(paths, source, git);
  dpcppInputs.auditDpcppCudaInputs(dpcpp, readBuildJson(dpcpp.manifest), dpcppPins.manifestSha256);
  auditGnuHeaders(gnu, windowsTarget);
  for (const [file, sha256] of /** @type {Array<[string, string]>} */ ([[sourceTar, archivePins.source], [dpcppTar, archivePins.dpcpp],
    [linuxResourcesTar, archivePins.linuxResources], [windowsGnuTar, archivePins.windowsGnu]])) {
    if (fileHash(file) !== sha256) {throw new Error("An input tar changed during Windows JIT resource production: " + path.basename(file));}
  }
  const manifestFile = path.join(output.resources, "manifest.json"), profile = readBuildJson(output.profile),
    manifest = readBuildJson(manifestFile), emitterReceipt = path.join(output.emitter, "result.json");
  if (profile.target !== windowsTarget || manifest.target !== windowsTarget ||
      manifest.publicationQualified !== false || !Array.isArray(manifest.files) || !manifest.files.length) {
    throw new Error("Public Windows JIT producer returned an incomplete or wrong-target resource tree");
  }
  const resultFile = path.join(output.root, "ci-result.json"), nativeTools = admitWindowsJitNativeTools(output.root, suffix);
  const resultRecord = {
    version: 1, target: windowsTarget, buildOnly: true, executed: false, publicationQualified: false,
    scope: "Windows source-JIT resources and source-matched build-only generators; no compiler SDK, device, runtime, or publication qualification.",
    producer: {script: windowsProducer, sha256: fileHash(path.join(repo, windowsProducer))},
    inputs: {
      linuxSources: {artifact: path.basename(source.sourceTar), bundleSha256: archivePins.source},
      dpcppCuda: {artifact: path.basename(dpcpp.sdkTar), bundleSha256: archivePins.dpcpp, manifestSha256: dpcppPins.manifestSha256},
      linuxResources: {artifact: path.basename(linuxResources.sdkTar), bundleSha256: archivePins.linuxResources,
        resultSha256: linuxResultSha, manifestSha256: linuxManifestSha},
      windowsGnu: {artifact: "native-windows-gnu-headers-" + suffix + ".tar", bundleSha256: archivePins.windowsGnu,
        resultSha256: gnuResultSha, copyrightsSha256: copyrightsSha, notices: gnuAudit.notices.length},
    },
    emitter: {receipt: emitterReceipt, sha256: fileHash(emitterReceipt)},
    profile: {path: output.profile, sha256: fileHash(output.profile)},
    resources: {manifest: manifestFile, manifestSha256: fileHash(manifestFile), files: manifest.files.length},
    nativeTools: {root: nativeTools.toolRoot, manifest: nativeTools.manifestFile, manifestSha256: nativeTools.manifestSha256,
      files: nativeTools.files, hashes: nativeTools.hashes, source: nativeTools.source},
  };
  fs.writeFileSync(resultFile, JSON.stringify(resultRecord, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const archive = path.join(paths.runnerTemp, "mom-native-windows-jit-resources-" + suffix + ".tar");
  const packed = ciTar.packState(archive, "/tmp", [path.basename(output.root)], 5 * 60 * 1000);
  ciTar.listState(archive, [path.basename(output.root)]);
  if (canonicalFile(archive, "Windows JIT resource bundle").size !== packed.bytes) {
    throw new Error("Windows JIT resource bundle changed after packing");
  }
  return {resources_bundle_sha256: fileHash(archive), resources_manifest_sha256: resultRecord.resources.manifestSha256,
    resources_result_sha256: fileHash(resultFile)};
}

/** Dispatch one Linux SDK CI command; outputs are appended to GITHUB_OUTPUT. @param {string[]} args */
function main(args) {
  const [command, ...rest] = args;
  if (rest.length) {throw new Error("Linux SDK CI commands take no further arguments");}
  if (command === "cuda-images-sources") {
    restoreCudaImageSources(ciSources.ciPaths());
  } else if (command === "rocm-hip-headers") {
    appendOutputs(exportRocmHipHeadersCi(ciSources.ciPaths()));
  } else if (command === "acpp-frontend") {
    appendOutputs(exportAcppFrontendCi(ciSources.ciPaths()));
  } else if (command === "linux-jit-resources-seal") {
    appendOutputs(sealLinuxJitResources(ciSources.ciPaths()));
  } else if (command === "windows-jit-resources") {
    appendOutputs(produceWindowsJitResourcesCi(ciSources.ciPaths()));
  } else {
    throw new Error("Usage: linux-sdk.js cuda-images-sources|rocm-hip-headers|acpp-frontend|linux-jit-resources-seal|windows-jit-resources");
  }
}

module.exports = {restoreCudaImageSources, exportRocmHipHeadersCi, exportAcppFrontendCi, assertLinuxJitResourceResult,
  sealLinuxJitResources, windowsJitInputPins, produceWindowsJitResourcesCi, main};

if (require.main === module) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
