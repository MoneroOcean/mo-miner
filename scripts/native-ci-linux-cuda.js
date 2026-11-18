"use strict";

// Native CI NVRTC verification and Linux CUDA image bundles with their bounded license sources.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildJson} = require("./build-source-inputs");
const {verifyNvrtcArchives} = require("./fetch-native-sdk-sources");
const {nvrtcSdkInputs} = require("./prepare-native-nvrtc-sdk");
const {nvrtcCohortInfo} = require("./probe-static-nvrtc");
const {cudaHeaderInputs} = require("./native-jit-headers");
const {cudaImageInputs} = require("./native-cuda-images");
const {canonicalExecutable, hasExactKeys, runNode} = require("./native-ci-job");
const {extractState, listState, packState, safeRelative, temp} = require("./native-ci-tar");
const {dpcppCudaInputPins, dpcppCudaInputsStatePaths, dpcppSourceArchivePin, provisionOneApi} = require("./native-ci-linux-sources");
const {auditCudaPackageCopyrights, auditDpcppCudaInputs, cudaPackageNames} = require("./native-dpcpp-cuda-inputs");
const {admitLinuxLlvmOutput} = require("./native-ci-linux-llvm");
const {hermesBuildTools} = require("./native-ci-linux-support");
const repo = path.resolve(__dirname, "..");

const cutlassVersion = "v4.6.1", cutlassArchiveSha256 = "455d9ba37d57cb214d67b5d1a6070441244b378bcacb2e916c3b86f2a9b02e1c",
  cutlassLicenseSha256 = "ad1513ca983da737a6843364ad19e8424a56d860736c2da773b8c5be527e1475";

// Record terms from the runtime, static-development and NVML-header payloads.
// cuda-minimal-build is only a dependency metapackage and ships no copyright file.
/** @param {string} name */
const cudaCopyrightName = name => name.replace(/-12-6$/, "-12.6-copyright.txt");

/** NVRTC verification keeps both vendor archives and assembled libraries local to one job.
 * Only compact receipts and the Linux CPU proof outputs cross the artifact boundary.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "suffix" | "runnerTemp"> & {env?: NodeJS.ProcessEnv}} paths
 * @param {"linux" | "windows"} platform @param {NodeJS.ProcessEnv} [environment] */
function nvrtcStatePaths(paths, platform, environment = paths.env || process.env) {
  if (platform !== "linux" && platform !== "windows") {throw new Error("Native NVRTC CI target must be linux or windows");}
  const suffix = paths.suffix, key = platform.toUpperCase(), sourceName = `mom-native-nvrtc-${platform}-prerequisites-${suffix}`,
    sdkName = `mom-native-nvrtc-${platform}-sdk-${suffix}`, evidenceName = `mom-native-nvrtc-${platform}-verification-${suffix}`,
    source = path.join(temp, sourceName), sdkRoot = path.join(temp, sdkName), evidence = path.join(temp, evidenceName),
    proofParent = path.join(temp, "mom-native-nvrtc-linux-proof-" + suffix),
    verificationTar = path.join(paths.runnerTemp, `native-nvrtc-${platform}-verification-${suffix}.tar`);
  if (environment[`MOM_NATIVE_NVRTC_${key}_SOURCE_DIR`] !== source || environment[`MOM_NATIVE_NVRTC_${key}_SDK_DIR`] !== sdkRoot) {
    throw new Error(`Native NVRTC ${platform} roots must exactly match this workflow run`);
  }
  return {platform, source, sourceName, sdkRoot, sdkName,
    evidence, evidenceName, proofParent, verificationTar};
}

/** Same-run build-only CUDA image inputs and the admitted image/license-source handoff.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "suffix" | "runnerTemp">} paths @param {NodeJS.ProcessEnv} [environment] */
function cudaImagesStatePaths(paths, environment = process.env) {
  const linux = nvrtcStatePaths(paths, "linux", environment), dpcpp = dpcppCudaInputsStatePaths(paths, environment),
    source126Name = `mom-native-nvrtc-linux-12.6-prerequisites-${paths.suffix}`,
    sdk126Name = `mom-native-nvrtc-linux-12.6-sdk-${paths.suffix}`,
    source126 = path.join(temp, source126Name), sdk126 = path.join(temp, sdk126Name),
    name = `mom-native-cuda-images-${paths.suffix}`, root = path.join(temp, name),
    headerRoot = path.join(root, "headers"), cutlassRoot = path.join(root, "cutlass"),
    cutlassLicense = path.join(cutlassRoot, "LICENSE.txt"),
    output = path.join(root, "bundle"), config = path.join(root, "worker-config.json"),
    handoffName = `mom-native-cuda-images-handoff-${paths.suffix}`, handoffRoot = path.join(temp, handoffName),
    noticeRoot = path.join(handoffRoot, "notice-sources"), noticeManifest = path.join(noticeRoot, "manifest.json"),
    tar = path.join(paths.runnerTemp, `native-linux-cuda-images-${paths.suffix}.tar`);
  if (environment["MOM_NATIVE_NVRTC_LINUX_126_SOURCE_DIR"] !== source126 ||
      environment["MOM_NATIVE_NVRTC_LINUX_126_SDK_DIR"] !== sdk126 || environment["MOM_NATIVE_CUDA_IMAGES_DIR"] !== root ||
      !path.isAbsolute(paths.runnerTemp) || fs.realpathSync(paths.runnerTemp) !== paths.runnerTemp) {
    throw new Error("Native CUDA image inputs must exactly match this run's source, SDK and output roots");
  }
  return {linux, source126, source126Name, sdk126, sdk126Name, dpcpp, name, root, headerRoot,
    cutlassRoot, cutlassInclude: path.join(cutlassRoot, "include"), cutlassMarker: path.join(cutlassRoot, ".mom-version"),
    cutlassLicense, noticeRoot, noticeManifest,
    header126: path.join(headerRoot, "cuda-12.6.inc"), header133: path.join(headerRoot, "cuda-13.3.inc"),
    output, config, result: path.join(output, "result.json"), handoffName, handoffRoot, tar,
    incomingDpcppTar: path.join(paths.runnerTemp, "incoming", path.basename(dpcpp.sdkTar))};
}

/** Bind a worker to the exact NVRTC cohort, same-run CUDA 12.6 header root, licenses and compiler.
 * The public image builder then compiles actual kernels with that NVRTC/header pairing.
 * @param {ReturnType<typeof nvrtcSdkInputs>} admitted @param {{cohort: "linux" | "linux-12.6", headerPack: string,
 * cudaInclude: string, cutlassInclude: string, cutlassMarker: string, cutlassLicense: string,
 * cudaCopyrightLicenses: string[], cudaCopyrightManifest: string, compiler: string,
 * compilerIdentity: string, installer: string}} input */
function cudaImageWorkerRecipe(admitted, input) {
  const receipt = admitted.receipt, pins = nvrtcCohortInfo("linux", input.cohort), sdkRoot = receipt.output,
    sdkInclude = path.join(sdkRoot, "include"), sdkLib = path.join(sdkRoot, "lib64");
  if (receipt.platform !== "linux" || receipt.cohort !== input.cohort || receipt.vendorVersion !== pins.version ||
      receipt.vendorPrebuilt !== true || receipt.sourceRebuilt !== false || receipt.compiled !== false ||
      receipt.publicationQualified !== false || receipt.redistributionQualified !== false ||
      !admitted.files || !/^[a-f0-9]{64}$/.test(admitted.receiptSha256 || "") ||
      admitted.files[admitted.receiptFile] !== admitted.receiptSha256) {
    throw new Error("CUDA image worker needs the exact admitted Linux NVRTC cohort and build-only receipt");
  }
  const licenses = [path.join(sdkRoot, "licenses/nvrtc/LICENSE"),
    ...(input.cohort === "linux" ? [path.join(sdkRoot, "licenses/nvptxcompiler/LICENSE")] : [])];
  if (licenses.some(file => !admitted.files[file])) {throw new Error("CUDA image worker inputs must retain every admitted NVRTC license hash");}
  for (const [file, hash] of Object.entries(admitted.files)) {
    if (!/^[a-f0-9]{64}$/.test(hash) || fileHash(file) !== hash) {throw new Error("CUDA image SDK closure changed before worker configuration: " + file);}
  }
  const header = cudaHeaderInputs(input.headerPack), headerRoots = header.manifest["roots"],
    cudaInclude = fs.realpathSync(input.cudaInclude), cutlassInclude = fs.realpathSync(input.cutlassInclude),
    expectedRoots = new Set([cudaInclude, cutlassInclude]), nestedCccl = path.join(cudaInclude, "cccl");
  if (fs.existsSync(path.join(nestedCccl, "cuda/std/cstdint"))) {expectedRoots.add(fs.realpathSync(nestedCccl));}
  if (!Array.isArray(headerRoots) || headerRoots.length !== expectedRoots.size ||
      headerRoots.some(root => typeof root !== "string" || !expectedRoots.has(root)) || headerRoots.includes(fs.realpathSync(sdkInclude))) {
    throw new Error("CUDA image header pack must bind only the same-run CUDA 12.6 and pinned CUTLASS roots, not NVRTC-only headers");
  }
  const extraInputs = [input.compiler, input.compilerIdentity, input.cutlassMarker, input.cutlassLicense,
    ...input.cudaCopyrightLicenses, input.cudaCopyrightManifest, input.installer];
  /** @type {Record<string, string>} */ const inputs = {...admitted.files};
  for (const file of extraInputs) {inputs[fs.realpathSync(file)] = fileHash(file);}
  return {headerPack: input.headerPack, worker: {command: fs.realpathSync(input.compiler),
    args: ["-std=c++20", "-O2", "-I" + sdkInclude, "{source}", "-L" + sdkLib,
      "-Wl,-rpath," + sdkLib, "-lnvrtc", "-pthread", "-o", "{output}"],
    inputs, environment: {LD_LIBRARY_PATH: sdkLib}}};
}

/** @typedef {{path:string,sha256:string}} CudaNoticeSourceReference */
/** @typedef {{version:string,url:string,sha256:string,member:string,installer:CudaNoticeSourceReference,
 * archiveMarker:CudaNoticeSourceReference}} CudaNoticeArchiveSource */
/** @typedef {{component:string,version:string,sourcePath:string,file:string,sha256:string,bytes:number,
 * sourceReceipt:CudaNoticeSourceReference}} CudaNoticeSource */
/** @typedef {{format:"mom-cuda-notice-sources-v1",scope:"CUDA/NVRTC/CUTLASS source license bodies only",
 * completeNativeNotices:false,sourceArchive:CudaNoticeArchiveSource,sources:CudaNoticeSource[]}} CudaNoticeSourcePlan */

/** Bind exact CUDA/NVRTC/CUTLASS license bodies to the receipts that supplied them.
 * This subset is explicitly not a complete native-distribution notice set.
 * @param {Pick<ReturnType<typeof cudaImagesStatePaths>, "sdk126" | "cutlassMarker" | "cutlassLicense"> &
 * {dpcpp:{root:string,manifest:string},linux:{sdkRoot:string}}} state
 * @param {{cuda:{licenses:Array<{package:string,path:string,sha256:string,version:string}>,packages:Record<string,string>}}} dpcpp
 * @param {ReturnType<typeof nvrtcSdkInputs>} baseline @param {ReturnType<typeof nvrtcSdkInputs>} modern
 * @returns {CudaNoticeSourcePlan} */
function cudaNoticeSourcePlan(state, dpcpp, baseline, modern) {
  const installer = path.join(repo, "scripts/install-cutlass.sh"), installerText = fs.readFileSync(installer, "utf8"),
    archiveUrl = `https://github.com/NVIDIA/cutlass/archive/refs/tags/${cutlassVersion}.tar.gz`;
  if (!installerText.includes(`MOM_CUTLASS_VERSION=${cutlassVersion}`) ||
      !installerText.includes(`MOM_CUTLASS_SHA256=${cutlassArchiveSha256}`) ||
      !installerText.includes(`MOM_CUTLASS_LICENSE_SHA256=${cutlassLicenseSha256}`) ||
      !/MOM_CUTLASS_URL="https:\/\/github\.com\/NVIDIA\/cutlass\/archive\/refs\/tags\/\$\{MOM_CUTLASS_VERSION\}\.tar\.gz"/.test(installerText) ||
      fs.readFileSync(state.cutlassMarker, "utf8").trim() !== cutlassArchiveSha256 ||
      fileHash(state.cutlassLicense) !== cutlassLicenseSha256 || baseline.receipt.cohort !== "linux-12.6" ||
      modern.receipt.cohort !== "linux" || dpcpp.cuda.licenses.length !== cudaPackageNames.length ||
      dpcpp.cuda.licenses.some((row, index) => row.package !== cudaPackageNames[index] ||
        row.version !== dpcpp.cuda.packages[row.package] || row.path !== path.join(state.dpcpp.root, "licenses", row.package, "copyright"))) {
    throw new Error("CUDA license sources do not match the pinned same-run package, NVRTC and CUTLASS inputs");
  }
  /** @param {string} component @param {string} version @param {string} sourcePath @param {string} file
   * @param {string} sha256 @param {{path:string,sha256:string}} sourceReceipt */
  const source = (component, version, sourcePath, file, sha256, sourceReceipt) => {
    const stat = fs.lstatSync(sourcePath);
    if (!stat.isFile() || stat.isSymbolicLink() || fs.realpathSync(sourcePath) !== sourcePath || stat.size < 1 || stat.size > (2 << 20)) {
      throw new Error("CUDA notice source must be a bounded canonical regular file: " + file);
    }
    return {component, version, sourcePath, file, sha256, bytes: stat.size, sourceReceipt};
  };
  const baselineLicense = path.join(state.sdk126, "licenses/nvrtc/LICENSE"),
    modernNvrtcLicense = path.join(state.linux.sdkRoot, "licenses/nvrtc/LICENSE"),
    modernPtxLicense = path.join(state.linux.sdkRoot, "licenses/nvptxcompiler/LICENSE");
  if (baseline.files[baselineLicense] !== fileHash(baselineLicense) ||
      modern.files[modernNvrtcLicense] !== fileHash(modernNvrtcLicense) || modern.files[modernPtxLicense] !== fileHash(modernPtxLicense) ||
      dpcpp.cuda.licenses.some(row => row.sha256 !== fileHash(row.path))) {
    throw new Error("CUDA/NVRTC license body is not bound by its admitted producer receipt");
  }
  return {format: "mom-cuda-notice-sources-v1", scope: "CUDA/NVRTC/CUTLASS source license bodies only", completeNativeNotices: false,
    sourceArchive: {version: cutlassVersion.slice(1), url: archiveUrl, sha256: cutlassArchiveSha256,
      member: `cutlass-${cutlassVersion.slice(1)}/LICENSE.txt`, installer: {path: installer, sha256: fileHash(installer)},
      archiveMarker: {path: state.cutlassMarker, sha256: fileHash(state.cutlassMarker)}},
    sources: [
      ...dpcpp.cuda.licenses.map(row => source(row.package.replace(/-12-6$/, ""), row.version, row.path,
        cudaCopyrightName(row.package), row.sha256, {path: state.dpcpp.manifest, sha256: fileHash(state.dpcpp.manifest)})),
      source("nvrtc", baseline.receipt.vendorVersion, baselineLicense, "nvrtc-12.6-LICENSE.txt", baseline.files[baselineLicense],
        {path: baseline.receiptFile, sha256: baseline.receiptSha256}),
      source("nvrtc", modern.receipt.vendorVersion, modernNvrtcLicense, "nvrtc-13.3-LICENSE.txt", modern.files[modernNvrtcLicense],
        {path: modern.receiptFile, sha256: modern.receiptSha256}),
      source("nvptxcompiler", modern.receipt.vendorVersion, modernPtxLicense, "nvptxcompiler-13.3-LICENSE.txt", modern.files[modernPtxLicense],
        {path: modern.receiptFile, sha256: modern.receiptSha256}),
      source("cutlass", cutlassVersion.slice(1), state.cutlassLicense, "cutlass-4.6.1-LICENSE.txt", cutlassLicenseSha256,
        {path: state.cutlassMarker, sha256: fileHash(state.cutlassMarker)}),
    ]};
}

/** @param {Pick<ReturnType<typeof cudaImagesStatePaths>, "noticeRoot" | "noticeManifest">} state @param {CudaNoticeSourcePlan} plan */
function retainCudaNoticeSources(state, plan) {
  if (fs.lstatSync(state.noticeRoot, {throwIfNoEntry: false}) || fs.lstatSync(state.noticeManifest, {throwIfNoEntry: false}) ||
      plan.completeNativeNotices !== false || plan.sources.length !== 7) {throw new Error("CUDA notice-source handoff requires a fresh exact source bundle");}
  fs.mkdirSync(state.noticeRoot, {mode: 0o700});
  let totalBytes = 0;
  for (const row of plan.sources) {
    if (!safeRelative(row.file) || row.file.includes("/") || !path.isAbsolute(row.sourcePath) || !/^[a-f0-9]{64}$/.test(row.sha256) ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > (2 << 20) || !path.isAbsolute(row.sourceReceipt.path) ||
        !/^[a-f0-9]{64}$/.test(row.sourceReceipt.sha256)) {throw new Error("CUDA notice-source row is malformed or oversized");}
    const sourceStat = fs.lstatSync(row.sourcePath), receiptStat = fs.lstatSync(row.sourceReceipt.path);
    if (!sourceStat.isFile() || sourceStat.isSymbolicLink() || fs.realpathSync(row.sourcePath) !== row.sourcePath ||
        sourceStat.size !== row.bytes || fileHash(row.sourcePath) !== row.sha256 || !receiptStat.isFile() || receiptStat.isSymbolicLink() ||
        fileHash(row.sourceReceipt.path) !== row.sourceReceipt.sha256) {throw new Error("CUDA notice source or its receipt changed: " + row.file);}
    totalBytes += row.bytes;
    if (!Number.isSafeInteger(totalBytes) || totalBytes > (8 << 20)) {throw new Error("CUDA notice-source bundle exceeds its total byte bound");}
    const destination = path.join(state.noticeRoot, row.file);
    fs.copyFileSync(row.sourcePath, destination, fs.constants.COPYFILE_EXCL); fs.chmodSync(destination, 0o644);
    if (fileHash(destination) !== row.sha256 || fs.statSync(destination).size !== row.bytes) {throw new Error("CUDA notice copy differs from its admitted source: " + row.file);}
  }
  fs.writeFileSync(state.noticeManifest, JSON.stringify(plan, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return auditCudaNoticeFiles(state, plan);
}

/** @param {Pick<ReturnType<typeof cudaImagesStatePaths>, "noticeRoot" | "noticeManifest">} state @param {CudaNoticeSourcePlan} [expected]
 * @returns {{manifest:CudaNoticeSourcePlan,manifestSha256:string,files:Array<{path:string,sha256:string,bytes:number}>}} */
function auditCudaNoticeFiles(state, expected) {
  const rootStat = fs.lstatSync(state.noticeRoot, {throwIfNoEntry: false}), manifestStat = fs.lstatSync(state.noticeManifest, {throwIfNoEntry: false});
  if (!rootStat?.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(state.noticeRoot) !== state.noticeRoot ||
      !manifestStat?.isFile() || manifestStat.isSymbolicLink() || manifestStat.size > (64 << 10)) {
    throw new Error("CUDA notice-source artifact root or manifest is missing, unsafe or excessive");
  }
  const manifest = readBuildJson(state.noticeManifest), plan = expected || manifest;
  if (!plan || typeof plan !== "object" || Array.isArray(plan) || !Array.isArray(plan.sources) ||
      !isDeepStrictEqual(manifest, plan)) {throw new Error("CUDA notice-source manifest changed during handoff");}
  const names = ["manifest.json", ...plan.sources.map((/** @type {CudaNoticeSource} */ row) => row.file)].sort();
  if (new Set(names).size !== names.length || !isDeepStrictEqual(fs.readdirSync(state.noticeRoot).sort(), names)) {
    throw new Error("CUDA notice-source artifact has missing or unexpected files");
  }
  for (const row of plan.sources) {
    const destination = path.join(state.noticeRoot, row.file), stat = fs.lstatSync(destination);
    if (!safeRelative(row.file) || row.file.includes("/") || !/^[a-f0-9]{64}$/.test(row.sha256 || "") ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > (2 << 20) || !stat.isFile() || stat.isSymbolicLink() ||
        (stat.mode & 0o111) !== 0 || stat.size !== row.bytes || fileHash(destination) !== row.sha256) {
      throw new Error("CUDA notice-source artifact failed exact body verification: " + row.file);
    }
  }
  return {manifest, manifestSha256: fileHash(state.noticeManifest),
    files: plan.sources.map((/** @type {CudaNoticeSource} */ row) => ({path: path.join(state.noticeRoot, row.file), sha256: row.sha256, bytes: row.bytes}))};
}

/** @param {Pick<ReturnType<typeof cudaImagesStatePaths>, "noticeRoot" | "noticeManifest" | "sdk126" | "cutlassMarker" | "cutlassLicense"> &
 * {dpcpp:{root:string,manifest:string},linux:{sdkRoot:string}}} state @param {ReturnType<typeof auditDpcppCudaInputs>} dpcpp
 * @param {unknown} imageManifest @param {CudaNoticeSourcePlan} [expected]
 * @returns {{path:string,manifest:string,manifestSha256:string,files:Array<{path:string,sha256:string,bytes:number}>,completeNativeNotices:false}} */
function auditCudaNoticeSources(state, dpcpp, imageManifest, expected) {
  const filesAudit = auditCudaNoticeFiles(state, expected), manifest = filesAudit.manifest,
    compilers = imageManifest && typeof imageManifest === "object" && Array.isArray(Reflect.get(imageManifest, "compilers")) ?
      /** @type {Array<{major:number,minor:number,inputs:Record<string,string>}>} */ (Reflect.get(imageManifest, "compilers")) : [],
    compiler126 = compilers.find(row => row.major === 12 && row.minor === 6), compiler133 = compilers.find(row => row.major === 13 && row.minor === 3),
    installer = path.join(repo, "scripts/install-cutlass.sh"), installerSha256 = fileHash(installer),
    nvrtc126 = nvrtcCohortInfo("linux", "linux-12.6"), nvrtc133 = nvrtcCohortInfo("linux", "linux"),
    cutlassMarker = state.cutlassMarker, installerHash126 = compiler126?.inputs[installer], installerHash133 = compiler133?.inputs[installer];
  /** @param {Record<string,string> | undefined} inputs @param {string} file */
  const inputHash = (inputs, file) => {
    const value = inputs?.[file];
    if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) {throw new Error("CUDA image input receipt omits notice source: " + file);}
    return value;
  };
  if (compilers.length !== 2 || !compiler126 || !compiler133 || installerHash126 !== installerSha256 || installerHash133 !== installerSha256 ||
      !manifest || typeof manifest !== "object" || Array.isArray(manifest) || manifest.format !== "mom-cuda-notice-sources-v1" ||
      manifest.scope !== "CUDA/NVRTC/CUTLASS source license bodies only" || manifest.completeNativeNotices !== false ||
      !Array.isArray(manifest.sources) || manifest.sources.length !== 7 ||
      !hasExactKeys(manifest.sourceArchive, ["version", "url", "sha256", "member", "installer", "archiveMarker"])) {
    throw new Error("CUDA notice-source artifact is missing or has an invalid/incomplete manifest");
  }
  const specs = [
    ...dpcpp.cuda.licenses.map((/** @type {{package:string,version:string,path:string,sha256:string}} */ row) => ({
      component: row.package.replace(/-12-6$/, ""), version: row.version, sourcePath: row.path,
      file: cudaCopyrightName(row.package), sha256: row.sha256, sourceReceipt: {path: state.dpcpp.manifest,
        sha256: inputHash(compiler126.inputs, state.dpcpp.manifest)}, compilers: [compiler126, compiler133]})),
    {component: "nvrtc", version: nvrtc126.version, sourcePath: path.join(state.sdk126, "licenses/nvrtc/LICENSE"),
      file: "nvrtc-12.6-LICENSE.txt", sha256: inputHash(compiler126.inputs, path.join(state.sdk126, "licenses/nvrtc/LICENSE")),
      sourceReceipt: {path: path.join(state.sdk126, "sdk.json"), sha256: inputHash(compiler126.inputs, path.join(state.sdk126, "sdk.json"))}, compilers: [compiler126]},
    {component: "nvrtc", version: nvrtc133.version, sourcePath: path.join(state.linux.sdkRoot, "licenses/nvrtc/LICENSE"),
      file: "nvrtc-13.3-LICENSE.txt", sha256: inputHash(compiler133.inputs, path.join(state.linux.sdkRoot, "licenses/nvrtc/LICENSE")),
      sourceReceipt: {path: path.join(state.linux.sdkRoot, "sdk.json"), sha256: inputHash(compiler133.inputs, path.join(state.linux.sdkRoot, "sdk.json"))}, compilers: [compiler133]},
    {component: "nvptxcompiler", version: nvrtc133.version, sourcePath: path.join(state.linux.sdkRoot, "licenses/nvptxcompiler/LICENSE"),
      file: "nvptxcompiler-13.3-LICENSE.txt", sha256: inputHash(compiler133.inputs, path.join(state.linux.sdkRoot, "licenses/nvptxcompiler/LICENSE")),
      sourceReceipt: {path: path.join(state.linux.sdkRoot, "sdk.json"), sha256: inputHash(compiler133.inputs, path.join(state.linux.sdkRoot, "sdk.json"))}, compilers: [compiler133]},
    {component: "cutlass", version: cutlassVersion.slice(1), sourcePath: state.cutlassLicense,
      file: "cutlass-4.6.1-LICENSE.txt", sha256: cutlassLicenseSha256,
      sourceReceipt: {path: cutlassMarker, sha256: inputHash(compiler126.inputs, cutlassMarker)}, compilers: [compiler126, compiler133]},
  ];
  const rawSources = manifest.sources;
  if (!hasExactKeys(manifest, ["format", "scope", "completeNativeNotices", "sourceArchive", "sources"]) ||
      !isDeepStrictEqual(manifest.sourceArchive, {version: cutlassVersion.slice(1),
        url: `https://github.com/NVIDIA/cutlass/archive/refs/tags/${cutlassVersion}.tar.gz`, sha256: cutlassArchiveSha256,
        member: `cutlass-${cutlassVersion.slice(1)}/LICENSE.txt`, installer: {path: installer, sha256: installerSha256},
        archiveMarker: {path: cutlassMarker, sha256: inputHash(compiler126.inputs, cutlassMarker)}})) {
    throw new Error("CUDA notice-source manifest differs from the pinned CUTLASS archive or worker recipe");
  }
  let totalBytes = 0;
  const expectedRows = specs.map((spec, index) => {
    // Compiler requirements guide admission; they are not serialized notice-source fields.
    const {compilers: requiredCompilers, ...source} = spec;
    const row = rawSources[index], bytes = row && row.bytes;
    if (!row || typeof bytes !== "number" || !Number.isSafeInteger(bytes) || bytes < 1 || bytes > (2 << 20)) {throw new Error("CUDA notice-source body exceeds its bound");}
    for (const compiler of requiredCompilers) {
      if (inputHash(compiler.inputs, spec.sourcePath) !== spec.sha256 ||
          inputHash(compiler.inputs, spec.sourceReceipt.path) !== spec.sourceReceipt.sha256) {
        throw new Error("CUDA image worker recipe no longer binds notice source and receipt: " + spec.file);
      }
    }
    totalBytes += bytes;
    return {...source, bytes};
  });
  if (!Number.isSafeInteger(totalBytes) || totalBytes > (8 << 20) || !isDeepStrictEqual(rawSources, expectedRows) ||
      expected && !isDeepStrictEqual(manifest, expected)) {throw new Error("CUDA notice-source artifact or worker provenance changed");}
  auditCudaPackageCopyrights(state.dpcpp, dpcpp.cuda.packages, dpcpp.cuda.licenses);
  return {path: state.noticeRoot, manifest: state.noticeManifest, manifestSha256: filesAudit.manifestSha256,
    files: filesAudit.files, completeNativeNotices: false};
}

/** @typedef {Pick<ReturnType<typeof cudaImagesStatePaths>, "root" | "source126" | "sdk126" | "handoffRoot" |
 * "headerRoot" | "header126" | "header133" | "cutlassRoot" | "cutlassInclude" | "cutlassMarker" |
 * "cutlassLicense" | "output" | "config" | "result" | "noticeRoot" | "noticeManifest"> &
 * {linux:{source:string,sdkRoot:string},dpcpp:import("./native-dpcpp-cuda-inputs").DpcppCudaInputState}} LinuxCudaImageTreeState */

/** Build the maintained CUDA image and license-source cohort from admitted directories.
 * Transport, tool installation and memory containment belong to the calling frontdoor.
 * @param {LinuxCudaImageTreeState} state @param {string} compiler
 * @param {NodeJS.ProcessEnv} env @param {string} [manifestSha256] */
function buildLinuxCudaImageTree(state, compiler, env, manifestSha256) {
  const outputs = [state.root, state.source126, state.sdk126, state.linux.source, state.linux.sdkRoot,
    state.handoffRoot];
  if (outputs.some(file => fs.existsSync(file))) {throw new Error("CUDA image producer requires fresh same-run outputs");}
  const dpcppInputs = auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), manifestSha256);
  fs.mkdirSync(state.root, {mode: 0o700});
  runNode("scripts/fetch-native-sdk-sources.js", ["nvrtc-linux", state.linux.source, "--allow-network"], env, 15 * 60 * 1000);
  runNode("scripts/fetch-native-sdk-sources.js", ["nvrtc-linux-12.6", state.source126, "--allow-network"], env, 15 * 60 * 1000);
  for (const cohort of /** @type {const} */ (["linux", "linux-12.6"])) {
    const source = cohort === "linux" ? state.linux.source : state.source126,
      sdk = cohort === "linux" ? state.linux.sdkRoot : state.sdk126;
    verifyNvrtcArchives("linux", source, {cohort});
    runNode("scripts/prepare-native-nvrtc-sdk.js", [cohort, source, sdk], env, 15 * 60 * 1000);
  }
  const modern = nvrtcSdkInputs("linux", state.linux.sdkRoot, {cohort: "linux"}), baseline = nvrtcSdkInputs("linux", state.sdk126, {cohort: "linux-12.6"});
  const version = spawnSync(compiler, ["--version"],
    {encoding: "utf8", env, timeout: 30000, maxBuffer: 1 << 20});
  const identity = (version.stdout || "") + (version.stderr || "");
  if (version.error || version.signal || version.status !== 0 || !/2026\.2/.test(identity)) {
    throw new Error("CUDA image worker requires the existing Intel 2026.2 compiler", {cause: version.error});
  }
  const compilerIdentity = path.join(state.root, "icpx-version.txt");
  fs.writeFileSync(compilerIdentity, identity, {flag: "wx", mode: 0o600});
  const installed = spawnSync("/usr/bin/bash", ["-euc", '. "$1"; install_cutlass_headers "$2"', "install-cutlass",
    path.join(repo, "scripts/install-cutlass.sh"), state.cutlassRoot],
  {cwd: repo, encoding: "utf8", env, timeout: 10 * 60 * 1000, maxBuffer: 4 << 20});
  const cutlassLog = path.join(state.root, "cutlass-install.log");
  fs.writeFileSync(cutlassLog, (installed.stdout || "") + (installed.stderr || ""), {flag: "wx", mode: 0o600});
  if (installed.error || installed.signal || installed.status !== 0 ||
      fs.readFileSync(state.cutlassMarker, "utf8").trim() !== "455d9ba37d57cb214d67b5d1a6070441244b378bcacb2e916c3b86f2a9b02e1c" ||
      !fs.statSync(path.join(state.cutlassInclude, "cute/tensor.hpp")).isFile() || fileHash(state.cutlassLicense) !== cutlassLicenseSha256) {
    throw new Error("Pinned CUTLASS 4.6.1 header installation failed", {cause: installed.error});
  }
  fs.mkdirSync(state.headerRoot, {mode: 0o700});
  for (const pack of [state.header126, state.header133]) {
    runNode("scripts/native-jit-headers.js", [pack, state.cutlassInclude, path.join(state.dpcpp.cudaRoot, "include")], env, 10 * 60 * 1000);
    const admitted = cudaHeaderInputs(pack);
    if (!admitted.manifest["roots"].includes(fs.realpathSync(path.join(state.dpcpp.cudaRoot, "include")))) {
      throw new Error("CUDA image header pack lost the admitted CUDA 12.6 toolkit root");
    }
  }
  const common = {cudaInclude: path.join(state.dpcpp.cudaRoot, "include"), cutlassInclude: state.cutlassInclude,
    cutlassMarker: state.cutlassMarker, cutlassLicense: state.cutlassLicense,
    cudaCopyrightLicenses: dpcppInputs.cuda.licenses.map((/** @type {{path:string}} */ row) => row.path), cudaCopyrightManifest: state.dpcpp.manifest,
    compiler, compilerIdentity,
    installer: path.join(repo, "scripts/install-cutlass.sh")};
  const config = {repo, output: state.output, compilers: [
    cudaImageWorkerRecipe(baseline, {...common, cohort: "linux-12.6", headerPack: state.header126}),
    cudaImageWorkerRecipe(modern, {...common, cohort: "linux", headerPack: state.header133}),
  ], timeoutMs: 30 * 60 * 1000};
  fs.writeFileSync(state.config, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  runNode("scripts/build-native-cuda-images.js", [state.config], env, 50 * 60 * 1000);
  const admitted = cudaImageInputs(path.join(state.output, "cuda-images.inc"), repo), result = readBuildJson(state.result),
    compilerRows = admitted.manifest["compilers"], imageRows = admitted.manifest["images"];
  if (!Array.isArray(compilerRows) || !Array.isArray(imageRows) ||
      compilerRows.some(row => !row || typeof row !== "object" || typeof row.major !== "number" || typeof row.minor !== "number")) {
    throw new Error("CUDA image bundle compiler or image matrix is malformed");
  }
  const compilerVersions = compilerRows.map(row => `${row.major}.${row.minor}`).sort();
  if (compilerVersions.join(",") !== "12.6,13.3" || result.plannedImages !== imageRows.length ||
      result.sourceSha256 !== admitted.manifest["sourceSha256"]) {
    throw new Error("CUDA image bundle did not admit the exact 12.6/13.3 same-run compiler matrix");
  }
  fs.mkdirSync(state.handoffRoot, {mode: 0o700});
  const noticePlan = cudaNoticeSourcePlan(state, dpcppInputs, baseline, modern),
    retainedNoticeFiles = retainCudaNoticeSources(state, noticePlan),
    noticeSources = auditCudaNoticeSources(state, dpcppInputs, admitted.manifest, noticePlan);
  if (retainedNoticeFiles.manifestSha256 !== noticeSources.manifestSha256) {throw new Error("CUDA notice-source manifest changed after retention");}
  /** @type {Array<[string, string]>} */ const outputFiles = [
    [path.join(state.output, "cuda-images.inc"), "cuda-images.inc"],
    [path.join(state.output, "cuda-images.inc.json"), "cuda-images.inc.json"], [state.result, "result.json"],
  ];
  for (const [source, name] of outputFiles) {
    fs.copyFileSync(source, path.join(state.handoffRoot, name), fs.constants.COPYFILE_EXCL);
  }
  return {noticeSources, images: admitted};
}

/** Restore the pinned CI inputs, invoke the shared producer, then transport its exact outputs.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildLinuxCudaImages(paths) {
  const state = cudaImagesStatePaths(paths, paths.env), env = paths.env;
  const externalDpcppPins = dpcppCudaInputPins(env);
  if ([state.root, state.source126, state.sdk126, state.linux.source, state.linux.sdkRoot, state.handoffRoot, state.tar]
    .some(file => fs.existsSync(file))) {throw new Error("CUDA image producer requires fresh same-run outputs");}
  const incoming = fs.lstatSync(state.incomingDpcppTar, {throwIfNoEntry: false});
  if (!incoming?.isFile() || incoming.isSymbolicLink()) {throw new Error("CUDA image producer needs the same-run DPC++/CUDA 12.6 tar");}
  dpcppSourceArchivePin(state.incomingDpcppTar, externalDpcppPins.bundleSha256);
  extractState(state.incomingDpcppTar, temp, [state.dpcpp.name]);
  auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), externalDpcppPins.manifestSha256);
  fs.unlinkSync(state.incomingDpcppTar);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-cuda-images-" + paths.suffix, 25 * 60 * 1000);
  const compiler = canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx"),
    {noticeSources} = buildLinuxCudaImageTree(state, compiler, env, externalDpcppPins.manifestSha256);
  const packed = packState(state.tar, temp, [state.handoffName]), members = listState(state.tar, [state.handoffName]).map(name => name.replace(/\/$/, "")).sort(),
    expected = [state.handoffName, `${state.handoffName}/notice-sources`,
      ...["cuda-images.inc", "cuda-images.inc.json", "result.json"].map(name => `${state.handoffName}/${name}`),
      `${state.handoffName}/notice-sources/manifest.json`,
      ...noticeSources.files.map((/** @type {{path:string,sha256:string,bytes:number}} */ row) => `${state.handoffName}/notice-sources/${path.basename(row.path)}`)].sort();
  if (packed.bytes <= 0 || !isDeepStrictEqual(members, expected)) {throw new Error("CUDA image handoff tar is not the exact include, sidecar and result");}
  console.log(`[native-ci] CUDA images plus ${noticeSources.files.length} bounded same-run license sources admitted (${packed.bytes} bytes); this subset is not complete native distribution notices or publication qualification.`);
}

/** Recheck the LLVM producer's audited install inventory before using its tools for NVRTC proof.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function staticNvrtcLlvmConfig(paths) {
  const sdk = path.join(paths.output, "sdk"), llvmConfig = canonicalExecutable(path.join(sdk, "bin/llvm-config")),
    env = {...paths.env, GIT_OPTIONAL_LOCKS: "0"};
  /** @param {string[]} args */
  const query = args => {
    const result = spawnSync(llvmConfig, args, {encoding: "utf8", env, timeout: 30000, maxBuffer: 1 << 20});
    if (result.error || result.signal || result.status !== 0 || result.stderr || typeof result.stdout !== "string") {
      throw new Error("Same-run LLVM 21 tool query failed: llvm-config " + args.join(" "), {cause: result.error});
    }
    return result.stdout.trim();
  };
  const bin = path.join(sdk, "bin"), selectedBin = fs.realpathSync(query(["--bindir"]));
  if (selectedBin !== fs.realpathSync(bin) || query(["--version"]) !== "21.1.8") {
    throw new Error("Same-run LLVM SDK does not identify its own LLVM 21.1.8 tool directory");
  }
  for (const name of ["llvm-config", "clang++", "ld.lld"]) {
    const selected = canonicalExecutable(path.join(selectedBin, name)), relative = path.relative(fs.realpathSync(sdk), selected);
    if (path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep)) {
      throw new Error("NVRTC proof tool escapes the same-run LLVM SDK: " + name);
    }
  }
  return llvmConfig;
}

/** Copy only regular proof receipts, logs and generated source outputs; never SDK binaries.
 * @param {string} source @param {string} destination */
function copyNvrtcEvidenceFile(source, destination) {
  const stat = fs.lstatSync(source);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 32 * 1024 * 1024) {
    throw new Error("NVRTC verification evidence must be a bounded regular file: " + path.basename(source));
  }
  fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
}

/** @param {ReturnType<typeof nvrtcStatePaths>} state @param {string} proofDirectory
 * @param {{receipt: {platform: string, source?: {receiptSha256?: string}}, receiptFile: string, receiptSha256: string}} admitted */
function packNvrtcVerification(state, proofDirectory, admitted) {
  if (fs.existsSync(state.evidence)) {throw new Error("Native NVRTC verification evidence output already exists");}
  fs.mkdirSync(state.evidence, {mode: 0o700});
  copyNvrtcEvidenceFile(path.join(state.source, "fetch.json"), path.join(state.evidence, "fetch.json"));
  copyNvrtcEvidenceFile(admitted.receiptFile, path.join(state.evidence, "sdk.json"));
  if (state.platform === "linux") {
    const proofReceipt = path.join(proofDirectory, "result.json"), proof = readBuildJson(proofReceipt),
      expectedSource = {file: "sycl/cn-gpu/cuda-expand-kernel.inc", sha256: fileHash(path.join(repo, "sycl/cn-gpu/cuda-expand-kernel.inc"))};
    if (proof.sdk !== state.sdkRoot || proof.vendor?.receiptSha256 !== admitted.receiptSha256 ||
        !Array.isArray(proof.files) || proof.files.length !== 4 || !isDeepStrictEqual(proof.source, expectedSource)) {
      throw new Error("NVRTC CPU proof does not bind the admitted run-scoped SDK and maintained source");
    }
    const expectedOutputs = ["compute_120.ptx", "compute_80.ptx", "sm_120.cubin", "sm_80.cubin"].sort();
    if (!isDeepStrictEqual(proof.files.map((/** @type {{name?: string} | null} */ row) => row?.name).sort(), expectedOutputs)) {
      throw new Error("NVRTC CPU proof output matrix is incomplete or contains unexpected files");
    }
    const proofOutput = path.join(state.evidence, "proof"); fs.mkdirSync(proofOutput, {mode: 0o700});
    copyNvrtcEvidenceFile(proofReceipt, path.join(proofOutput, "result.json"));
    for (const directory of ["actual", "expected", "lld"]) {
      const sourceDirectory = path.join(proofDirectory, directory), destination = path.join(proofOutput, directory);
      const stat = fs.lstatSync(sourceDirectory);
      if (!stat.isDirectory() || stat.isSymbolicLink() ||
          !isDeepStrictEqual(fs.readdirSync(sourceDirectory).sort(), expectedOutputs)) {
        throw new Error("NVRTC proof output is not the exact generated-code matrix: " + directory);
      }
      fs.mkdirSync(destination, {mode: 0o700});
      for (const row of proof.files) {
        if (typeof row.name !== "string" || !/^[A-Za-z0-9_.+-]{1,128}$/.test(row.name) ||
            !Number.isSafeInteger(row.bytes) || row.bytes < 1 || !/^[0-9a-f]{64}$/.test(row.sha256 || "")) {
          throw new Error("Invalid NVRTC generated proof output record");
        }
        const file = path.join(sourceDirectory, row.name), outputStat = fs.lstatSync(file);
        if (!outputStat.isFile() || outputStat.isSymbolicLink() || outputStat.size !== row.bytes || fileHash(file) !== row.sha256) {
          throw new Error("NVRTC generated proof output changed before evidence packaging: " + row.name);
        }
        copyNvrtcEvidenceFile(file, path.join(destination, row.name));
      }
    }
    const logs = path.join(proofOutput, "logs"); fs.mkdirSync(logs, {mode: 0o700});
    for (const name of ["build.log", "oracle-build.log", "compile.log", "oracle.log", "llvm-tools.log", "lld-build.log",
      "lld.log", "arguments.log", "dependencies.log", "syscalls.log", "isolated-paths.log"]) {
      copyNvrtcEvidenceFile(path.join(proofDirectory, name), path.join(logs, name));
    }
  }
  const packed = packState(state.verificationTar, temp, [state.evidenceName]);
  if (packed.bytes > 128 * 1024 * 1024) {throw new Error("NVRTC verification evidence exceeded its bounded upload size");}
  return packed;
}

/** Fetch, assemble, and admit one target SDK in the same job; Linux also runs the strict CPU oracle.
 * Vendor archives, assembled libraries, and linked proof executables never enter the uploaded tar.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths @param {"linux" | "windows"} platform */
function verifyNvrtcTarget(paths, platform) {
  const state = nvrtcStatePaths(paths, platform, paths.env);
  if ([state.source, state.sdkRoot, state.evidence, ...(platform === "linux" ? [state.proofParent] : [])].some(item => fs.existsSync(item))) {
    throw new Error("Native NVRTC verification requires new run-scoped outputs");
  }
  runNode("scripts/fetch-native-sdk-sources.js", ["nvrtc-" + platform, state.source, "--allow-network"], paths.env, 15 * 60 * 1000);
  const fetched = verifyNvrtcArchives(platform, state.source), fetchReceipt = readBuildJson(path.join(state.source, "fetch.json"));
  if (fetched.target !== "nvrtc-" + platform || fetched.status !== "PASS" || fetched.compiled !== false ||
      fetched.publicationQualified !== false || fetchReceipt.target !== "nvrtc-" + platform) {
    throw new Error("Native NVRTC acquisition is not the pinned build-only target cohort");
  }
  runNode("scripts/prepare-native-nvrtc-sdk.js", [platform, state.source, state.sdkRoot], paths.env, 15 * 60 * 1000);
  const admitted = nvrtcSdkInputs(platform, state.sdkRoot), sdk = admitted.receipt;
  if (sdk.platform !== platform || sdk.vendorPrebuilt !== true || sdk.sourceRebuilt !== false || sdk.compiled !== false ||
      sdk.publicationQualified !== false || sdk.redistributionQualified !== false || sdk.source.receiptSha256 !== fileHash(path.join(state.source, "fetch.json"))) {
    throw new Error("Native NVRTC SDK admission must remain vendor-prebuilt and non-publication-qualified");
  }
  let proofDirectory = "";
  if (platform === "linux") {
    provisionOneApi(paths, "/var/tmp/mom-dev-native-nvrtc-" + paths.suffix);
    const tools = hermesBuildTools();
    admitLinuxLlvmOutput(paths);
    const llvmConfig = staticNvrtcLlvmConfig(paths);
    fs.mkdirSync(state.proofParent, {mode: 0o700});
    runNode("scripts/probe-static-nvrtc.js", [state.sdkRoot, state.proofParent],
      {...paths.env, CXX: tools.cxx, LLVM_CONFIG: llvmConfig}, 15 * 60 * 1000);
    const children = fs.readdirSync(state.proofParent);
    if (children.length !== 1) {throw new Error("Static NVRTC proof did not produce exactly one evidence directory");}
    proofDirectory = path.join(state.proofParent, children[0] || "");
    const proofStat = fs.lstatSync(proofDirectory);
    if (!proofStat.isDirectory() || proofStat.isSymbolicLink()) {throw new Error("Static NVRTC proof output is not a real directory");}
  }
  const packed = packNvrtcVerification(state, proofDirectory, admitted);
  console.log(`[native-ci] Verified NVRTC ${platform}; vendor SDK stays on-runner and no release qualification is implied (${packed.bytes} evidence bytes).`);
  return packed;
}

module.exports = {auditCudaNoticeFiles, auditCudaNoticeSources, buildLinuxCudaImageTree, buildLinuxCudaImages, cudaCopyrightName,
  cudaImageWorkerRecipe, cudaImagesStatePaths, cudaNoticeSourcePlan, nvrtcStatePaths, packNvrtcVerification,
  retainCudaNoticeSources, staticNvrtcLlvmConfig, verifyNvrtcTarget};
