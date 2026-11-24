"use strict";

// Native CI ROCm sources, COMGR/HIP-header and host SDKs, and the ACPP CUDA/HIP SDK.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"), {createHash} = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");
const {canonicalExecutable, ciBuildJobs, hasExactKeys, runNode} = require("./native-ci-job");
const {containsBuildPath, fileHash, readBuildJson, readBuildJsonSnapshot, readBuildText, sourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {gitSourceComponents} = require("./fetch-native-sdk-sources");
const {capturePackageCopyright} = require("./prepare-native-gnu-headers");
const {sdkSourceEnvironment} = require("./build-environment");
const {nativeRocmHostInputs, nativeStaticRuntimeInputs} = require("./native-release-sdk-inputs");
const {staticSyclInputs, runJitCacheIdentity, runHcfRepro} = require("./native-sycl-inputs");
const {sdkSourceFiles, verifySdkExport} = require("./prepare-native-sdk-sources");
const {nativeOpenmpInputs} = require("./build-native-openmp-sdk");
const {comgrDeviceInputs} = require("./build-native-comgr-sdk");
const {comgrSdkInputs} = require("./native-comgr-inputs");
const {hiprtcHeaders} = require("./build-native-rocm-resources");
const {extractState, listState, packState, runTar, safeRelative, temp} = require("./native-ci-tar");
const {acppStatePaths, dpcppCudaInputPins, dpcppCudaInputsStatePaths, dpcppSourceArchivePin, openmpStatePaths, provisionOneApi,
  rocmParserSource, rocmPinnedHostPackages, rocmSourceStatePaths, verifyAcppSource, verifyLinuxSources, verifyRocmParserSource,
  verifyRocmSources} = require("./native-ci-linux-sources");
const {auditDpcppCudaInputs} = require("./native-dpcpp-cuda-inputs");
const {admitLinuxLlvmOutput} = require("./native-ci-linux-llvm");
const {requireLinuxSysroot} = require("./native-linux-sysroot");
const {libdrmPin, nativeLibdrmInputs} = require("./build-native-libdrm");
const repo = path.resolve(__dirname, "..");

const rocmHipHeaderPatch = "scripts/patches/rocm-hip-headers-only.patch",
  rocmHipHeaderPatchSha256 = "caed551ccaa0d8c3469676d4e3a74c6cd353ed75425297398119d36eca3854e0",
  rocmHipCmakeOriginalSha256 = "4c968219e2a92609160166b73ad924ef4ce1194f451fc7afdd7c11c9c798ced6",
  rocmHipCmakePatchedSha256 = "de18cbc92de9992f90648e7df0ec591c6d23444f4827719227f5683fdcb0a7da";

const rocmResourcesRecipe = ["scripts/build-native-rocm-resources.js", "scripts/native-comgr-inputs.js", "scripts/build-native-comgr-sdk.js",
  "scripts/build-native-sycl-sdk.js", "scripts/prepare-native-sdk-sources.js", "scripts/build-source-inputs.js", "scripts/build-environment.js"];

/** Restore the exact same-run dependencies for the source-composed static ACPP CUDA/HIP SDK.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "suffix" | "runnerTemp" | "source" | "sourceName" | "ioSourceName" | "sdkRoot" | "sdkName" | "output">} paths
 * @param {NodeJS.ProcessEnv} [environment] */
function acppCudaHipSdkStatePaths(paths, environment = process.env) {
  const acpp = acppStatePaths(paths, environment), openmp = openmpStatePaths(paths, environment), rocm = rocmHostStatePaths(paths, environment),
    cudaInputs = dpcppCudaInputsStatePaths(paths, environment),
    sdkName = "mom-native-acpp-cuda-hip-sdk-" + paths.suffix, sdkRoot = path.join(temp, sdkName),
    sdkTar = path.join(paths.runnerTemp, "native-linux-acpp-cuda-hip-sdk-" + paths.suffix + ".tar");
  if (environment["MOM_NATIVE_ACPP_CUDA_HIP_SDK_DIR"] !== sdkRoot) {
    throw new Error("Native ACPP CUDA/HIP SDK output root must exactly match this workflow run");
  }
  const rocmRoots = [rocm.sourceName, rocm.llvmSourceName, rocm.llvmSdkName, rocm.hipHeadersName,
    rocm.resourcesName, rocm.sdkName, rocm.hostSdkName];
  return {acpp, openmp, rocm, cudaInputs, sdkName, sdkRoot, sdkTar, rocmRoots,
    incomingAcppTar: acpp.incomingTar, incomingOpenmpTar: path.join(paths.runnerTemp, "incoming", path.basename(openmp.sdkTar)),
    incomingRocmHostTar: path.join(paths.runnerTemp, "incoming", path.basename(rocm.hostSdkTar)),
    incomingLinuxTar: path.join(paths.runnerTemp, "incoming", path.basename(cudaInputs.linux.sourceTar)),
    incomingCudaInputsTar: path.join(paths.runnerTemp, "incoming", path.basename(cudaInputs.sdkTar)),
    llvmSdk: rocm.llvmSdk, rocmSdk: path.join(rocm.runtimeOutput, "sdk"),
    comgrReceipt: path.join(rocm.sdkRoot, "result.json"), deviceRoot: path.join(rocm.resourcesRoot, "install"),
    deviceBitcode: path.join(rocm.resourcesRoot, "install/amdgcn/bitcode"), openmpLibrary: path.join(openmp.sdkRoot, "sdk/libomp.a")};
}

/** @param {ReturnType<typeof acppCudaHipSdkStatePaths>} state @param {{cc: string, cxx: string}} tools */
function acppCudaHipSdkBuildCommand(state, tools) {
  return {script: "scripts/build-native-sycl-sdk.js", args: [state.acpp.preparedCheckout, state.llvmSdk,
    state.cudaInputs.cudaRoot, state.sdkRoot, tools.cc, tools.cxx]};
}

/** Keep the release composer’s combined-backend and exact same-run input contract explicit.
 * @param {ReturnType<typeof acppCudaHipSdkStatePaths>} state
 * @param {{hostCompilers?: {cc?: string, cxx?: string}, inputs?: {comgr?: {receipt?: string}}, fingerprints?: Record<string, string>}} record
 * @param {{libraries: string[], backend?: string, cudaVersion?: number, source?: string, build?: string, cxx?: string,
 * comgr?: {receipt?: string, roots?: {llvm?: string, lld?: string, device?: string}}, openmp?: {archive?: string}, files?: Record<string, string>}} sdk
 * @param {{cc: string, cxx: string}} tools
 * @param {{cuda?: {root?: string, headers?: {files?: Record<string, string>}, staticRuntime?: {cudart?: {path?: string, sha256?: string}, culibos?: {path?: string, sha256?: string}}}}} cudaInputs */
function assertAcppCudaHipSdk(state, record, sdk, tools, cudaInputs) {
  const cudaLibraries = cudaInputs.cuda?.staticRuntime,
    cudart = path.join(state.cudaInputs.cudaRoot, "lib64/libcudart_static.a"), culibos = path.join(state.cudaInputs.cudaRoot, "lib64/libculibos.a"),
    cudaHeaders = cudaInputs.cuda?.headers?.files;
  // The SDK receipt's cxx is LLVM clang++ for device compilation; tools stays the oneAPI host pair.
  if (!cudaLibraries || !cudaHeaders || cudaInputs.cuda?.root !== state.cudaInputs.cudaRoot ||
      ["cuda.h", "cuda_runtime_api.h"].some(name => !/^[0-9a-f]{64}$/.test(cudaHeaders[name] || "")) ||
      cudaLibraries.cudart?.path !== cudart || cudaLibraries.culibos?.path !== culibos ||
      !/^[0-9a-f]{64}$/.test(cudaLibraries.cudart.sha256 || "") || !/^[0-9a-f]{64}$/.test(cudaLibraries.culibos.sha256 || "") ||
      sdk.backend !== "cuda-hip" || sdk.cudaVersion !== 12060 || sdk.source !== state.acpp.preparedCheckout ||
      sdk.build !== path.join(state.sdkRoot, "build") || sdk.cxx !== path.join(state.llvmSdk, "bin/clang++") ||
      record.hostCompilers?.cc !== tools.cc || record.hostCompilers?.cxx !== tools.cxx ||
      record.inputs?.comgr?.receipt !== state.comgrReceipt || sdk.comgr?.roots?.llvm !== state.llvmSdk ||
      sdk.comgr?.receipt !== state.comgrReceipt || sdk.comgr?.roots?.lld !== state.llvmSdk || sdk.comgr?.roots?.device !== state.deviceRoot ||
      sdk.openmp?.archive !== state.openmpLibrary ||
      record.fingerprints?.[path.join(state.cudaInputs.cudaRoot, "include/cuda.h")] !== cudaHeaders["cuda.h"] ||
      record.fingerprints?.[path.join(state.cudaInputs.cudaRoot, "include/cuda_runtime_api.h")] !== cudaHeaders["cuda_runtime_api.h"] ||
      record.fingerprints?.[cudart] !== cudaLibraries.cudart.sha256 || record.fingerprints?.[culibos] !== cudaLibraries.culibos.sha256 ||
      sdk.files?.[cudart] !== cudaLibraries.cudart.sha256 || sdk.files?.[culibos] !== cudaLibraries.culibos.sha256) {
    throw new Error("Public ACPP CUDA/HIP SDK admission differs from the exact same-run CI inputs");
  }
  return sdk;
}

/** Admit the public combined SDK and every same-run CUDA, LLVM/ROCm/COMGR and OpenMP input.
 * @param {ReturnType<typeof acppCudaHipSdkStatePaths>} state @param {{cc: string, cxx: string}} tools
 * @param {string} [expectedDpcppManifestSha256] */
function auditAcppCudaHipSdk(state, tools, expectedDpcppManifestSha256) {
  const receiptFile = path.join(state.sdkRoot, "result.json"), record = readBuildJson(receiptFile),
    openmp = nativeOpenmpInputs(state.openmp.sdkRoot), comgr = comgrSdkInputs(state.comgrReceipt),
    cudaInputs = auditDpcppCudaInputs(state.cudaInputs, readBuildJson(state.cudaInputs.manifest), expectedDpcppManifestSha256),
    hostRecord = readBuildJson(path.join(state.rocm.runtimeOutput, "result.json"));
  comgrDeviceInputs(state.deviceRoot);
  if (openmp.archive !== state.openmpLibrary || comgr.roots.llvm !== state.llvmSdk || comgr.roots.lld !== state.llvmSdk ||
      comgr.roots.device !== state.deviceRoot || hostRecord.version !== 1 || hostRecord.buildOnly !== true ||
      hostRecord.executed !== false || hostRecord.publicationQualified !== false || hostRecord.prefix !== state.rocmSdk ||
      hostRecord.comgr?.receipt !== state.comgrReceipt || hostRecord.comgr?.roots?.llvm !== state.llvmSdk ||
      hostRecord.comgr?.roots?.lld !== state.llvmSdk || hostRecord.comgr?.roots?.device !== state.deviceRoot) {
    throw new Error("ACPP CUDA/HIP inputs do not bind the same-run LLVM, CUDA, ROCm, COMGR, resources and OpenMP roots");
  }
  const sdk = nativeStaticRuntimeInputs(receiptFile, {openmpSdkRoot: state.openmp.sdkRoot});
  return assertAcppCudaHipSdk(state, record, sdk, tools, cudaInputs);
}

/** Restore same-run source/SDK roots, build combined CUDA/HIP ACPP offline, then admit and retain its new SDK.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildAcppCudaHipSdk(paths) {
  const state = acppCudaHipSdkStatePaths(paths, paths.env), git = canonicalExecutable("/usr/bin/git");
  if (fs.existsSync(state.sdkRoot) || fs.existsSync(state.sdkTar)) {throw new Error("Native ACPP CUDA/HIP stage refuses an existing output");}
  const externalDpcppPins = dpcppCudaInputPins(paths.env);
  // When supplied by CI, authenticate transport before extraction; preserve the legacy unpinned path.
  dpcppSourceArchivePin(state.incomingCudaInputsTar, externalDpcppPins.bundleSha256);
  extractState(state.incomingAcppTar, temp, [state.acpp.sourceName]);
  verifyAcppSource(paths, state.acpp, git);
  extractState(state.incomingLinuxTar, temp, [state.cudaInputs.linux.sourceName]);
  verifyLinuxSources(paths, state.cudaInputs.linux, git);
  extractState(state.incomingCudaInputsTar, temp, [state.cudaInputs.name]);
  dpcppSourceArchivePin(state.incomingCudaInputsTar, externalDpcppPins.bundleSha256);
  auditDpcppCudaInputs(state.cudaInputs, readBuildJson(state.cudaInputs.manifest), externalDpcppPins.manifestSha256);
  extractState(state.incomingRocmHostTar, temp, state.rocmRoots);
  // OpenMP's historical handoff also carries the shared LLVM source root. That root is already
  // restored and admitted from the ROCm handoff, so validate the complete tar but restore only libomp.
  extractState(state.incomingOpenmpTar, temp, [paths.sourceName, state.openmp.sdkName], [state.openmp.sdkName]);
  verifyRocmSources(paths, state.rocm, git);
  const sourceEnv = {...sdkSourceEnvironment(paths.env), GIT_ALLOW_PROTOCOL: "file", GIT_NO_LAZY_FETCH: "1",
    GIT_TERMINAL_PROMPT: "0", GIT_OPTIONAL_LOCKS: "0"};
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", state.rocm.llvmSource], sourceEnv, 2 * 60 * 1000);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-acpp-cuda-hip-" + paths.suffix, 35 * 60 * 1000);
  nativeOpenmpInputs(state.openmp.sdkRoot); comgrSdkInputs(state.comgrReceipt); comgrDeviceInputs(state.deviceRoot);
  admitLinuxLlvmOutput(paths);
  for (const incoming of [state.incomingAcppTar, state.incomingOpenmpTar, state.incomingRocmHostTar,
    state.incomingLinuxTar, state.incomingCudaInputsTar]) {fs.unlinkSync(incoming);}
  const tools = {cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx")};
  const command = acppCudaHipSdkBuildCommand(state, tools), buildEnv = {...sourceEnv, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: ciBuildJobs(paths.env),
    MOM_AOT_SYCL_BACKEND: "cuda-hip", MOM_AOT_ROCM_STATIC_ROOT: state.rocmSdk, MOM_AOT_LLD_ROOT: state.llvmSdk,
    ROCM_DEVICE_LIBS_PATH: state.deviceBitcode, MOM_AOT_COMGR_RECEIPT: state.comgrReceipt, MOM_AOT_OPENMP_LIBRARY: state.openmpLibrary};
  runNode(command.script, command.args, buildEnv, 45 * 60 * 1000);
  const admitted = auditAcppCudaHipSdk(state, tools, externalDpcppPins.manifestSha256);
  const evidenceParent = fs.realpathSync(paths.runnerTemp), sdkRoot = fs.realpathSync(state.sdkRoot);
  if (containsBuildPath(sdkRoot, evidenceParent)) {
    throw new Error("ACPP CUDA/HIP regression evidence must stay outside the packaged SDK");
  }
  const evidence = fs.mkdtempSync(path.join(evidenceParent, "native-acpp-cuda-hip-sycl-proof-")),
    probeEnv = {...buildEnv, CXX: tools.cxx, LLVM_CONFIG: path.join(state.llvmSdk, "bin/llvm-config")},
    {source, build, cxx, flags} = staticSyclInputs(state.acpp.preparedCheckout, path.join(state.sdkRoot, "build"), evidence, "hip", probeEnv),
    cacheIdentity = runJitCacheIdentity(tools.cxx, flags, source, build, evidence, repo),
    hcfObjectRepro = runHcfRepro(cxx, flags, source, build, evidence, repo);
  fs.writeFileSync(path.join(evidence, "native-ci-regressions.json"), JSON.stringify({cacheIdentity, hcfObjectRepro}, null, 2) + "\n", {flag: "wx"});
  packState(state.sdkTar, temp, [state.sdkName]); listState(state.sdkTar, [state.sdkName]);
  console.log("[native-ci] ACPP CUDA/HIP SDK admitted: " + admitted.libraries.length + " libraries; HIP HCF and CPU-only JIT-cache proofs passed, no GPU execution or publication qualification. Evidence: " + evidence);
}

/** Same-run ROCm/COMGR build roots retain both upstream Git cohorts, the matching LLVM SDK,
 * exported HIP headers, and fresh resource/COMGR outputs under their original /tmp paths.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "suffix" | "runnerTemp" | "source" | "sourceName" | "ioSourceName" | "sdkRoot" | "sdkName" | "output">} paths
 * @param {NodeJS.ProcessEnv} [environment] */
function rocmComgrStatePaths(paths, environment = process.env) {
  const rocm = rocmSourceStatePaths(paths, environment), resourcesName = "mom-native-rocm-resources-" + paths.suffix,
    hipHeadersName = "mom-native-rocm-hip-headers-" + paths.suffix, sdkName = "mom-native-rocm-comgr-sdk-" + paths.suffix,
    resourcesRoot = path.join(temp, resourcesName), hipHeadersRoot = path.join(temp, hipHeadersName), sdkRoot = path.join(temp, sdkName),
    sourceTar = path.join(paths.runnerTemp, "native-rocm-state-" + paths.suffix + "-sources.tar"),
    llvmSdkTar = path.join(paths.runnerTemp, "native-linux-llvm21-installed-sdk-" + paths.suffix + ".tar"),
    llvmSourceTar = path.join(paths.runnerTemp, "native-llvm21-state-" + paths.suffix + "-sources.tar"),
    sdkTar = path.join(paths.runnerTemp, "native-linux-rocm-comgr-sdk-" + paths.suffix + ".tar");
  if (environment["MOM_NATIVE_ROCM_COMGR_SDK_DIR"] !== sdkRoot ||
      environment["MOM_NATIVE_ROCM_RESOURCES_DIR"] !== resourcesRoot ||
      environment["MOM_NATIVE_ROCM_HIP_HEADERS_DIR"] !== hipHeadersRoot) {
    throw new Error("Native ROCm/COMGR output roots must exactly match this workflow run");
  }
  const hipSourceRoot = path.join(hipHeadersRoot, "source"), hipCommonRoot = path.join(hipSourceRoot, "hip-common"),
    clrHipamdRoot = path.join(hipSourceRoot, "clr-hipamd"), hipBuildRoot = path.join(hipHeadersRoot, "build"),
    hipIncludeRoot = path.join(hipHeadersRoot, "include"), hipArchiveRoot = path.join(hipHeadersRoot, "archives"),
    hipInstallPrefix = path.join(hipBuildRoot, "install-prefix"), hipPatchRoot = path.join(hipHeadersRoot, "patches"),
    hipPatchFile = path.join(hipPatchRoot, path.basename(rocmHipHeaderPatch)),
    hipPatchCheckLog = path.join(hipPatchRoot, "check.json"), hipPatchApplyLog = path.join(hipPatchRoot, "apply.json");
  return {...rocm, resourcesName, resourcesRoot, hipHeadersName, hipHeadersRoot, hipIncludeRoot, hipSourceRoot, hipCommonRoot,
    clrHipamdRoot, hipBuildRoot, hipArchiveRoot, hipInstallPrefix, hipPatchRoot, hipPatchFile, hipPatchCheckLog, hipPatchApplyLog, sdkName, sdkRoot,
    sourceTar, llvmSdkTar, llvmSourceTar, incomingSourceTar: path.join(paths.runnerTemp, "incoming", path.basename(sourceTar)),
    incomingLlvmSdkTar: path.join(paths.runnerTemp, "incoming", path.basename(llvmSdkTar)),
    incomingLlvmSourceTar: path.join(paths.runnerTemp, "incoming", path.basename(llvmSourceTar)),
    llvmSourceArchiveRoots: [paths.sourceName, paths.ioSourceName],
    hipCommonTar: path.join(hipArchiveRoot, "hip-common.tar"), clrHipamdTar: path.join(hipArchiveRoot, "clr-hipamd.tar"),
    hipConfigureLog: path.join(hipHeadersRoot, "cmake-configure.log"),
    hipConfigurePlan: path.join(hipHeadersRoot, "configure-plan.json"),
    resourcesConfig: path.join(paths.runnerTemp, "rocm-resources-" + paths.suffix + ".json"),
    sdkTar, llvmSource: paths.source, llvmSourceName: paths.sourceName, llvmSdkRoot: paths.sdkRoot,
    llvmSdkName: paths.sdkName, llvmSdk: path.join(paths.output, "sdk"), output: paths.output};
}

/** Only the already-admitted source/LLVM inputs and paths consumed by the public ROCm recipe.
 * @typedef {Pick<ReturnType<typeof rocmComgrStatePaths>, "source" | "llvmSdk" | "hipHeadersRoot" | "hipSourceRoot" | "hipCommonRoot" |
 * "clrHipamdRoot" | "hipBuildRoot" | "hipArchiveRoot" | "hipInstallPrefix" | "hipIncludeRoot" | "hipPatchRoot" | "hipPatchFile" |
 * "hipPatchCheckLog" | "hipPatchApplyLog" | "hipConfigureLog" | "hipConfigurePlan" | "hipCommonTar" | "clrHipamdTar" |
 * "resourcesRoot" | "resourcesConfig" | "sdkRoot">} RocmComgrBuildState */

/** Restore the admitted COMGR cohort and retain a fresh, separately rooted ROCm host runtime SDK.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "suffix" | "runnerTemp" | "source" | "sourceName" | "ioSourceName" | "sdkRoot" | "sdkName" | "output">} paths
 * @param {NodeJS.ProcessEnv} [environment] */
function rocmHostStatePaths(paths, environment = process.env) {
  const comgr = rocmComgrStatePaths(paths, environment), hostSdkName = "mom-native-rocm-host-sdk-" + paths.suffix,
    hostSdkRoot = path.join(temp, hostSdkName), runtimeOutput = path.join(hostSdkRoot, "runtime"),
    configFile = path.join(hostSdkRoot, "rocm-host.json"),
    incomingComgrTar = path.join(paths.runnerTemp, "incoming", path.basename(comgr.sdkTar)),
    hostSdkTar = path.join(paths.runnerTemp, "native-linux-rocm-host-sdk-" + paths.suffix + ".tar");
  if (environment["MOM_NATIVE_ROCM_HOST_SDK_DIR"] !== hostSdkRoot || path.dirname(runtimeOutput) !== hostSdkRoot) {
    throw new Error("Native ROCm host SDK output root must exactly match this workflow run");
  }
  // The source-built libdrm travels inside the host SDK root because ROCm admission rehashes its archives.
  return {...comgr, hostSdkName, hostSdkRoot, runtimeOutput, configFile, incomingComgrTar, hostSdkTar,
    libdrmSource: path.join(hostSdkRoot, "libdrm-source"), libdrmSdk: path.join(hostSdkRoot, "libdrm")};
}

/** @param {Pick<RocmComgrBuildState, "source" | "llvmSdk" | "hipIncludeRoot">} state
 * @param {{cc: string, cxx: string, cmake: string}} tools @param {"linux" | "win32"} [target] */
function rocmResourcesConfig(state, tools, target = "linux") {
  if (target !== "linux" && target !== "win32") {throw new Error("Unsupported ROCm resources target");}
  return {version: 1, target, source: path.join(state.source, "roots/rocmLlvm"),
    clr: path.join(state.source, "roots/clr"), llvm: state.llvmSdk, cmake: tools.cmake, hipHeaders: state.hipIncludeRoot,
    hostCompilers: {cc: tools.cc, cxx: tools.cxx}};
}

/** Export one pinned Git tree into a new source root and verify every regular file and its membership.
 * @param {string} git @param {string} gitRoot @param {string} treeish @param {string} archive @param {string} destination
 * @param {NodeJS.ProcessEnv} env @param {string} label */
function exportRocmSourceTree(git, gitRoot, treeish, archive, destination, env, label) {
  if ([archive, destination].some(file => fs.existsSync(file))) {throw new Error("ROCm header source export refuses existing output: " + label);}
  const archived = spawnSync(git, ["-C", gitRoot, "archive", "--format=tar", "--output=" + archive, treeish],
    {env, stdio: "inherit", timeout: 5 * 60 * 1000});
  if (archived.error || archived.signal || archived.status !== 0) {throw new Error("Pinned ROCm header source archive failed: " + label, {cause: archived.error});}
  const tree = spawnSync(git, ["-C", gitRoot, "ls-tree", "-rz", treeish],
    {env, encoding: "utf8", timeout: 60_000, maxBuffer: 32 << 20});
  if (tree.error || tree.signal || tree.status !== 0 || typeof tree.stdout !== "string") {
    throw new Error("Pinned ROCm header source tree could not be read: " + label, {cause: tree.error});
  }
  fs.mkdirSync(destination, {mode: 0o700});
  runTar("extract", ["--extract", "--file", archive, "--directory", destination, "--no-same-owner", "--no-same-permissions"]);
  const files = sdkSourceFiles(destination);
  verifySdkExport(destination, files, tree.stdout);
  return {fileCount: Object.keys(files).length, filesSha256: createHash("sha256").update(JSON.stringify(files)).digest("hex"),
    treeSha256: createHash("sha256").update(tree.stdout).digest("hex")};
}

/** The pinned CLR repository stores its HIP AMD CMake project under `hipamd/`. @param {string} revision */
function rocmClrHipamdTreeish(revision) {
  if (!/^[0-9a-f]{40}$/.test(revision)) {throw new Error("Invalid pinned CLR revision");}
  return revision + ":hipamd";
}

/** Apply only the reviewed headers-only packaging guard to the verified CLR HIP subtree.
 * Preserve the exact patch copy, check/apply logs, and before/after target hashes in the tarred state.
 * @param {string} git @param {string} patchFile @param {RocmComgrBuildState} state @param {NodeJS.ProcessEnv} env */
function applyRocmHipHeadersPatch(git, patchFile, state, env) {
  const target = path.join(state.clrHipamdRoot, "CMakeLists.txt");
  if (fileHash(patchFile) !== rocmHipHeaderPatchSha256 || fileHash(target) !== rocmHipCmakeOriginalSha256 ||
      fs.existsSync(state.hipPatchRoot)) {throw new Error("Pinned CLR header-only patch inputs drifted or already exist");}
  fs.mkdirSync(state.hipPatchRoot, {mode: 0o700});
  fs.copyFileSync(patchFile, state.hipPatchFile, fs.constants.COPYFILE_EXCL);
  if (fileHash(state.hipPatchFile) !== rocmHipHeaderPatchSha256) {throw new Error("Saved CLR header-only patch copy changed");}
  /** @param {string[]} args @param {string} log */
  const runPatch = (args, log) => {
    const command = [git, "-C", state.clrHipamdRoot, "apply", "-p2", ...args, "--", state.hipPatchFile],
      result = spawnSync(command[0] || "", command.slice(1), {env, encoding: "utf8", timeout: 60_000, maxBuffer: 8 << 20});
    fs.writeFileSync(log, JSON.stringify({command, status: result.status, signal: result.signal,
      error: result.error?.message || null, stdout: result.stdout || "", stderr: result.stderr || ""}, null, 2) + "\n", {flag: "wx", mode: 0o600});
    if (result.error || result.signal || result.status !== 0) {throw new Error("Pinned CLR header-only patch failed; inspect saved patch log", {cause: result.error});}
  };
  runPatch(["--check"], state.hipPatchCheckLog);
  runPatch([], state.hipPatchApplyLog);
  const patchedSha256 = fileHash(target);
  if (patchedSha256 !== rocmHipCmakePatchedSha256) {throw new Error("CLR header-only patch produced unexpected CMake bytes");}
  /** @param {string} file */
  const relative = file => path.relative(state.hipHeadersRoot, file).split(path.sep).join("/");
  return {path: relative(state.hipPatchFile), sha256: rocmHipHeaderPatchSha256, stripComponents: 2,
    target: relative(target), originalSha256: rocmHipCmakeOriginalSha256, patchedSha256,
    checkLog: relative(state.hipPatchCheckLog), applyLog: relative(state.hipPatchApplyLog)};
}

/** Merge the upstream common and CLR AMD headers without overwrite, then require the generated
 * version header and every source-declared HIPRTC header before the LLVM resource build begins.
 * @param {string} hipCommonRoot @param {string} clrHipamdRoot @param {string} generatedHeader
 * @param {string} output @param {string} hiprtcCmake */
function assembleRocmHipHeaders(hipCommonRoot, clrHipamdRoot, generatedHeader, output, hiprtcCmake) {
  if (![hipCommonRoot, clrHipamdRoot, generatedHeader, output].every(path.isAbsolute) || fs.existsSync(output) ||
      typeof hiprtcCmake !== "string" || hiprtcCmake.length > (8 << 20)) {
    throw new Error("ROCm HIP header package inputs are missing, ambiguous or already exist");
  }
  const commonRoot = path.join(hipCommonRoot, "include/hip"), clrRoot = path.join(clrHipamdRoot, "include/hip/amd_detail"),
    common = sdkSourceFiles(commonRoot), clr = sdkSourceFiles(clrRoot), generated = fs.lstatSync(generatedHeader, {throwIfNoEntry: false});
  if (!Object.keys(common).length || !Object.keys(clr).length || Object.keys(common).some(name => name === "hip_version.h" || name.startsWith("amd_detail/")) ||
      !generated?.isFile() || generated.isSymbolicLink()) {throw new Error("ROCm HIP header package lacks common, CLR or generated-version inputs");}
  const required = hiprtcHeaders(hiprtcCmake, output);
  fs.mkdirSync(output, {mode: 0o700});
  const copy = (/** @type {string} */ sourceRoot, /** @type {string} */ relative, /** @type {string} */ destination) => {
    if (!safeRelative(relative)) {throw new Error("Unsafe pinned ROCm header path");}
    fs.mkdirSync(path.dirname(destination), {recursive: true, mode: 0o700});
    fs.copyFileSync(path.join(sourceRoot, relative), destination, fs.constants.COPYFILE_EXCL);
  };
  for (const name of Object.keys(common)) {copy(commonRoot, name, path.join(output, "hip", name));}
  for (const name of Object.keys(clr)) {copy(clrRoot, name, path.join(output, "hip/amd_detail", name));}
  copy(path.dirname(generatedHeader), "hip_version.h", path.join(output, "hip/hip_version.h"));
  for (const header of required) {
    const stat = fs.lstatSync(header, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink()) {throw new Error("Required HIPRTC header is missing from the pinned HIP/CLR package: " + path.basename(header));}
  }
  return {files: sdkSourceFiles(output), required};
}

/** The upstream configure is intentionally configure-only: the generated HIP version header is
 * required, but HIP runtime libraries are not built or installed in this prerequisite stage.
 * @param {RocmComgrBuildState} state @param {{cc: string, cxx: string, cmake: string}} tools */
function hipHeaderConfigureCommand(state, tools) {
  return [tools.cmake, "-S", state.clrHipamdRoot, "-B", state.hipBuildRoot, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=" + state.hipInstallPrefix, "-DCMAKE_C_COMPILER=" + tools.cc, "-DCMAKE_CXX_COMPILER=" + tools.cxx,
    "-DHIP_COMMON_DIR=" + state.hipCommonRoot, "-DHIP_PLATFORM:STRING=amd", "-DHIP_RUNTIME:STRING=none", "-DHIPCC_BIN_DIR:STRING=",
    "-DHIP_OFFICIAL_BUILD:BOOL=ON", "-DROCM_PATH=" + state.hipInstallPrefix];
}

/** Preserve the upstream producer's native path layout and pass only literal selected roots.
 * @param {RocmComgrBuildState} state @param {{cc: string, cxx: string, cmake: string}} tools */
function rocmComgrBuildCommands(state, tools) {
  const resources = rocmResourcesConfig(state, tools), device = path.join(state.resourcesRoot, "install");
  return [
    {script: "scripts/build-native-rocm-resources.js", args: [state.resourcesConfig, state.resourcesRoot], config: resources},
    {script: "scripts/build-native-comgr-sdk.js", args: [path.join(state.source, "roots/rocmLlvm"), state.llvmSdk, state.llvmSdk,
      device, state.sdkRoot, tools.cc, tools.cxx]},
  ];
}

/** Re-admit the CPU-only HIP/device resource receipt and public COMGR SDK receipt together.
 * @param {RocmComgrBuildState} state @param {{cc: string, cxx: string, cmake: string}} tools */
function auditRocmComgrSdk(state, tools) {
  const resources = readBuildJson(path.join(state.resourcesRoot, "result.json")), device = path.join(state.resourcesRoot, "install"),
    git = gitSourceComponents("rocm"), actualDevice = comgrDeviceInputs(device), sdk = comgrSdkInputs(path.join(state.sdkRoot, "result.json")),
    comgrRecord = readBuildJson(sdk.receipt), expectedHostCompilers = {cc: tools.cc, cxx: tools.cxx};
  if (resources.version !== 1 || resources.target !== "linux" || resources.buildOnly !== true || resources.publicationQualified !== false ||
      resources.revision !== git["rocmLlvm"]?.revision || resources.clrRevision !== git["clr"]?.revision ||
      !isDeepStrictEqual(resources.hostCompilers, expectedHostCompilers) || !isDeepStrictEqual(comgrRecord.hostCompilers, expectedHostCompilers) ||
      !isDeepStrictEqual(resources.device, actualDevice) ||
      !isDeepStrictEqual(resources.source, sdkSourceFiles(path.join(state.resourcesRoot, "source"))) ||
      !resources.header || resources.header.path !== path.join(state.resourcesRoot, "hipRTC") || fileHash(resources.header.path) !== resources.header.sha256 ||
      !resources.object || resources.object.path !== path.join(state.resourcesRoot, "hipRTC_header.o") || fileHash(resources.object.path) !== resources.object.sha256) {
    throw new Error("ROCm resource receipt does not bind the pinned Linux source export and AMD device library tree");
  }
  verifySourceInputs(resources.recipe, sourceInputs(repo, [], rocmResourcesRecipe), "ROCm resource recipe");
  for (const [label, files] of [["inputs", resources.inputs], ["dependencies", resources.dependencies]]) {
    if (!files || typeof files !== "object" || Array.isArray(files) || !Object.keys(files).length || Object.keys(files).length > 65536) {
      throw new Error("ROCm resource receipt has a missing or excessive " + label + " map");
    }
    for (const [file, sha] of Object.entries(files)) {
      if (!path.isAbsolute(file) || typeof sha !== "string" || !/^[0-9a-f]{64}$/.test(sha) || fileHash(file) !== sha) {
        throw new Error("ROCm resource input changed: " + file);
      }
    }
  }
  if (sdk.roots.llvm !== state.llvmSdk || sdk.roots.lld !== state.llvmSdk || sdk.roots.device !== device) {
    throw new Error("COMGR SDK does not use this run's same LLVM/LLD and Linux AMD device resources");
  }
  return {resources, sdk};
}

/** Build the existing CPU-only HIP headers, ROCm resources and COMGR SDK from admitted roots.
 * Source/tar admission and host provisioning remain the caller's responsibility. The LLVM callback
 * is checked before output creation and again between the resource and COMGR producers.
 * @param {RocmComgrBuildState} state @param {{cc:string,cxx:string,cmake:string}} tools
 * @param {NodeJS.ProcessEnv} environment @param {()=>unknown} verifyLlvmAdmission */
function buildRocmComgrFromInputs(state, tools, environment, verifyLlvmAdmission) {
  const freshOutputs = [state.resourcesRoot, state.hipHeadersRoot, state.sdkRoot, state.resourcesConfig];
  for (const output of freshOutputs) {
    if (typeof output !== "string" || !path.isAbsolute(output) || path.resolve(output) !== output ||
        fs.lstatSync(output, {throwIfNoEntry: false}) || fs.realpathSync(path.dirname(output)) !== path.dirname(output)) {
      throw new Error("ROCm/COMGR producer requires fresh canonical output/config paths: " + String(output));
    }
  }
  if (typeof verifyLlvmAdmission !== "function") {throw new Error("ROCm/COMGR producer requires a read-only LLVM admission callback");}
  verifyLlvmAdmission();
  const git = canonicalExecutable("/usr/bin/git"), gitEnvironment = {...sdkSourceEnvironment(environment), GIT_ALLOW_PROTOCOL: "file", GIT_NO_LAZY_FETCH: "1",
    GIT_TERMINAL_PROMPT: "0", GIT_OPTIONAL_LOCKS: "0"};
  fs.mkdirSync(state.hipHeadersRoot, {mode: 0o700});
  fs.mkdirSync(state.hipArchiveRoot, {mode: 0o700});
  fs.mkdirSync(state.hipSourceRoot, {mode: 0o700});
  const hipGitRoot = path.join(state.source, "roots/hip"), clrGitRoot = path.join(state.source, "roots/clr"),
    pins = gitSourceComponents("rocm"), hipRevision = pins["hip"]?.revision, clrRevision = pins["clr"]?.revision;
  if (!hipRevision || !clrRevision) {throw new Error("Pinned ROCm source cohort omitted HIP or CLR");}
  const hipExport = exportRocmSourceTree(git, hipGitRoot, hipRevision, state.hipCommonTar, state.hipCommonRoot, gitEnvironment, "HIP common"),
    clrTreeish = rocmClrHipamdTreeish(clrRevision),
    clrExport = exportRocmSourceTree(git, clrGitRoot, clrTreeish, state.clrHipamdTar, state.clrHipamdRoot, gitEnvironment, "CLR HIP AMD"),
    patchEvidence = applyRocmHipHeadersPatch(git, path.join(repo, rocmHipHeaderPatch), state, gitEnvironment),
    patchedClrFiles = sdkSourceFiles(state.clrHipamdRoot);
  const configure = hipHeaderConfigureCommand(state, tools), configureEnvironment = {...gitEnvironment, GIT_DIR: clrGitRoot};
  fs.writeFileSync(state.hipConfigurePlan, JSON.stringify({version: 1, command: configure, environment: {GIT_DIR: clrGitRoot},
    configureOnly: true, buildInvoked: false, installInvoked: false}, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const configureFd = fs.openSync(state.hipConfigureLog, fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY, 0o600);
  let configureResult;
  try {
    configureResult = spawnSync(configure[0] || "", configure.slice(1),
      {cwd: repo, env: configureEnvironment, stdio: ["ignore", configureFd, configureFd], timeout: 15 * 60 * 1000});
  } finally {
    fs.closeSync(configureFd);
  }
  if (configureResult.error || configureResult.signal || configureResult.status !== 0) {
    throw new Error("Pinned upstream HIP header configure failed; inspect its saved configure log", {cause: configureResult.error});
  }
  const generatedHeader = path.join(state.hipBuildRoot, "include/hip/hip_version.h"),
    hiprtcCmake = readBuildText(path.join(state.clrHipamdRoot, "src/hiprtc/CMakeLists.txt"), 1 << 20),
    headerPackage = assembleRocmHipHeaders(state.hipCommonRoot, state.clrHipamdRoot, generatedHeader, state.hipIncludeRoot, hiprtcCmake);
  /** @param {string} file */
  const relative = file => path.relative(state.hipHeadersRoot, file).split(path.sep).join("/");
  fs.writeFileSync(path.join(state.hipHeadersRoot, "header-package.json"), JSON.stringify({version: 1,
    pins: {hip: hipRevision, clr: clrRevision}, sourceExports: {
      hipCommon: {root: relative(state.hipCommonRoot), revision: hipRevision, ...hipExport},
      clrHipamd: {root: relative(state.clrHipamdRoot), revision: clrRevision, treeish: clrTreeish, ...clrExport,
        afterPatch: {fileCount: Object.keys(patchedClrFiles).length,
          filesSha256: createHash("sha256").update(JSON.stringify(patchedClrFiles)).digest("hex")}},
    }, patch: patchEvidence,
    configure: {command: configure, environment: {GIT_DIR: clrGitRoot}, configureOnly: true,
      log: relative(state.hipConfigureLog), logSha256: fileHash(state.hipConfigureLog), status: configureResult.status},
    hiprtcCmake: {path: relative(path.join(state.clrHipamdRoot, "src/hiprtc/CMakeLists.txt")), sha256: fileHash(path.join(state.clrHipamdRoot, "src/hiprtc/CMakeLists.txt"))},
    generatedHeader: {path: relative(generatedHeader), sha256: fileHash(generatedHeader)},
    files: headerPackage.files, requiredHiprtcHeaders: headerPackage.required.map(file => path.relative(state.hipIncludeRoot, file).split(path.sep).join("/")),
    buildOnly: true, publicationQualified: false}, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const commands = rocmComgrBuildCommands(state, tools), buildEnv = {...environment, MOM_BUILD_JOBS: ciBuildJobs(environment),
    GIT_ALLOW_PROTOCOL: "file", GIT_NO_LAZY_FETCH: "1", GIT_TERMINAL_PROMPT: "0", GIT_OPTIONAL_LOCKS: "0"};
  fs.writeFileSync(state.resourcesConfig, JSON.stringify(rocmResourcesConfig(state, tools), null, 2) + "\n", {flag: "wx", mode: 0o600});
  try {
    const resourceCommand = commands[0], comgrCommand = commands[1];
    if (!resourceCommand || !comgrCommand) {throw new Error("Native ROCm/COMGR command plan is incomplete");}
    runNode(resourceCommand.script, resourceCommand.args, buildEnv, 15 * 60 * 1000);
    verifyLlvmAdmission();
    runNode(comgrCommand.script, comgrCommand.args, buildEnv, 20 * 60 * 1000);
    return auditRocmComgrSdk(state, tools);
  } finally {
    if (fs.existsSync(state.resourcesConfig)) {fs.unlinkSync(state.resourcesConfig);}
  }
}

/** Restore/admit the same-run inputs, provision tools, and package the shared CPU-only ROCm producer.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildRocmComgrSdk(paths) {
  const state = rocmComgrStatePaths(paths), git = canonicalExecutable("/usr/bin/git");
  for (const output of [state.resourcesRoot, state.hipHeadersRoot, state.sdkRoot, state.resourcesConfig, state.sdkTar]) {
    if (fs.existsSync(output)) {throw new Error("Native ROCm/COMGR stage refuses an existing output: " + path.basename(output));}
  }
  extractState(state.incomingSourceTar, temp, [state.sourceName]);
  verifyRocmSources(paths, state, git);
  extractState(state.incomingLlvmSourceTar, temp, state.llvmSourceArchiveRoots, [state.llvmSourceName]);
  extractState(state.incomingLlvmSdkTar, temp, [state.llvmSdkName]);
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", state.llvmSource], paths.env, 2 * 60 * 1000);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-rocm-comgr-" + paths.suffix, 15 * 60 * 1000);
  // Retain the original pre-cleanup gate; the reusable producer also checks its own entry boundary.
  admitLinuxLlvmOutput(paths);
  for (const file of [state.incomingSourceTar, state.incomingLlvmSourceTar, state.incomingLlvmSdkTar]) {fs.unlinkSync(file);}
  const tools = {cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx"), cmake: canonicalExecutable("/usr/bin/cmake")};
  buildRocmComgrFromInputs(state, tools, paths.env, () => admitLinuxLlvmOutput(paths));
  packState(state.sdkTar, temp, [state.sourceName, state.llvmSourceName, state.llvmSdkName, state.hipHeadersName,
    state.resourcesName, state.sdkName]);
  listState(state.sdkTar, [state.sourceName, state.llvmSourceName, state.llvmSdkName, state.hipHeadersName,
    state.resourcesName, state.sdkName]);
}

/** Canonical paths accepted by the public ROCm host-runtime producer.
 * @param {Pick<ReturnType<typeof rocmHostStatePaths>, "llvmSdk">} state */
function rocmHostBuildTools(state) {
  return {ar: canonicalExecutable("/usr/bin/ar"), cmake: canonicalExecutable("/usr/bin/cmake"),
    dpkgQuery: canonicalExecutable("/usr/bin/dpkg-query"), git: canonicalExecutable("/usr/bin/git"),
    llvmObjcopy: canonicalExecutable(path.join(state.llvmSdk, "bin/llvm-objcopy")), ninja: canonicalExecutable("/usr/bin/ninja"),
    perl: canonicalExecutable("/usr/bin/perl"), pkgConfig: canonicalExecutable("/usr/bin/pkg-config"),
    python3: canonicalExecutable("/usr/bin/python3.12"), ranlib: canonicalExecutable("/usr/bin/ranlib"),
    tar: canonicalExecutable("/usr/bin/tar"), xxd: canonicalExecutable("/usr/bin/xxd"),
    cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx")};
}

/** Static OS inputs: libelf/libnuma from the glibc sysroot and libdrm from its admitted source-built SDK.
 * @param {Pick<ReturnType<typeof rocmHostStatePaths>, "source" | "llvmSdk" | "sdkRoot" | "resourcesRoot" | "libdrmSdk">} state
 * @param {ReturnType<typeof rocmHostBuildTools>} tools @param {NodeJS.ProcessEnv} environment */
function rocmHostBuildConfig(state, tools, environment) {
  const sysroot = requireLinuxSysroot(environment), libdrm = nativeLibdrmInputs(state.libdrmSdk),
    systemLibraries = path.join(sysroot.root, "usr/lib/x86_64-linux-gnu");
  return {version: 1, sources: {clr: path.join(state.source, "roots/clr"), hip: path.join(state.source, "roots/hip"),
    rocr: path.join(state.source, "roots/rocr")},
  tools: {ar: tools.ar, cmake: tools.cmake, dpkgQuery: tools.dpkgQuery, git: tools.git, llvmObjcopy: tools.llvmObjcopy,
    ninja: tools.ninja, perl: tools.perl, pkgConfig: tools.pkgConfig, python3: tools.python3, ranlib: tools.ranlib, tar: tools.tar, xxd: tools.xxd},
  hostCompilers: {cc: tools.cc, cxx: tools.cxx},
  dependencies: {drm: libdrm.drm, drmAmdgpu: libdrm.drmAmdgpu, elf: path.join(systemLibraries, "libelf.a"), numa: path.join(systemLibraries, "libnuma.a")},
  libdrm: libdrm.root, pkgConfigDir: libdrm.pkgConfigDir, comgrReceipt: path.join(state.sdkRoot, "result.json")};
}

/** Fetch (networked, from the pinned mirror release) and build the static libdrm SDK, or admit an existing one.
 * @param {{libdrmSource: string, libdrmSdk: string}} state @param {NodeJS.ProcessEnv} env */
function buildLibdrmSdk(state, env) {
  if (!fs.existsSync(state.libdrmSource)) {
    runNode("scripts/fetch-native-sdk-sources.js", ["libdrm", state.libdrmSource, "--allow-network"], env, 10 * 60 * 1000);
  }
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-libdrm", state.libdrmSource], env, 2 * 60 * 1000);
  if (!fs.existsSync(state.libdrmSdk)) {
    runNode("scripts/build-native-libdrm.js", [path.join(state.libdrmSource, libdrmPin.fileName), state.libdrmSdk,
      canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"), canonicalExecutable("/usr/bin/meson"),
      canonicalExecutable("/usr/bin/ninja"), canonicalExecutable("/usr/bin/tar")], env, 20 * 60 * 1000);
  }
  return nativeLibdrmInputs(state.libdrmSdk);
}

/** APT is only for host build prerequisites. The public producer runs afterward from local pinned inputs.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "env">} paths @param {typeof spawnSync} [runner]
 * @returns {{mode:"already-installed",packages:Record<string,string>}|undefined} */
function provisionRocmHostPackages(paths, runner = spawnSync) {
  const packageNames = ["libdrm-dev", "meson", ...Object.keys(rocmPinnedHostPackages)],
    // eslint-disable-next-line no-template-curly-in-string -- dpkg-query, not JavaScript, expands these fields.
    format = "${Package}\t${Version}\t${db:Status-Abbrev}\n",
    installed = runner("/usr/bin/dpkg-query", ["-W", "-f=" + format, ...packageNames],
      {encoding: "utf8", env: {...paths.env, LC_ALL: "C"}, timeout: 30000, maxBuffer: 1 << 20});
  if (!installed.error && !installed.signal && installed.status === 0 && typeof installed.stdout === "string") {
    const rows = installed.stdout.trimEnd().split(/\r?\n/), packages = new Map(); let valid = rows.length === packageNames.length;
    for (const row of rows) {
      const [name, version, status, ...extra] = row.split("\t"), expected = name ? rocmPinnedHostPackages[name] : undefined;
      if (!name || !packageNames.includes(name) || !version || status?.trim() !== "ii" || extra.length || packages.has(name) ||
          (expected && version !== expected)) {valid = false; break;}
      packages.set(name, version);
    }
    if (valid && packageNames.every(name => packages.has(name))) {
      return {mode: "already-installed", packages: Object.fromEntries(packages)};
    }
  }
  /** @param {string[]} args @param {number} timeout @param {string} label */
  const apt = (args, timeout, label) => {
    const result = runner("sudo", ["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", ...args],
      {cwd: repo, env: paths.env, stdio: "inherit", timeout});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error("Native ROCm host prerequisite " + label + " failed", {cause: result.error});
    }
  };
  apt(["update", "-qq"], 8 * 60 * 1000, "APT index refresh");
  const catalog = runner("/usr/bin/apt-cache", ["madison", ...Object.keys(rocmPinnedHostPackages)],
    {encoding: "utf8", env: paths.env, timeout: 30000, maxBuffer: 1 << 20});
  const available = new Set((typeof catalog.stdout === "string" ? catalog.stdout : "").split(/\r?\n/)
    .map(line => line.split("|").map(part => part.trim())).filter(parts => parts.length >= 2).map(parts => parts[0] + "=" + parts[1]));
  if (catalog.error || catalog.signal || catalog.status !== 0 ||
      Object.entries(rocmPinnedHostPackages).some(([name, version]) => !available.has(name + "=" + version))) {
    throw new Error("Pinned ROCm host prerequisite package versions are not available from the Ubuntu runner repositories");
  }
  // libdrm-dev supplies only the Debian copyright text captured for notices; meson builds the pinned libdrm source.
  apt(["install", "-y", "--no-install-recommends", "--allow-downgrades", "libdrm-dev", "meson",
    ...Object.entries(rocmPinnedHostPackages).map(([name, version]) => name + "=" + version)], 10 * 60 * 1000, "pinned OS packages");
  return undefined;
}

/** Admit an existing parser directory only when it is canonical and exactly matches the
 * complete regular-file map extracted from the checksum-pinned source distribution.
 * @param {string} destination @param {Record<string, string>} expectedFiles */
function admitRocmParserInstallation(destination, expectedFiles) {
  const stat = fs.lstatSync(destination, {throwIfNoEntry: false});
  if (!stat) {return undefined;}
  if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(destination) !== destination) {
    throw new Error("Refusing to reuse a noncanonical or redirected CppHeaderParser installation");
  }
  const installed = sdkSourceFiles(destination);
  if (!isDeepStrictEqual(expectedFiles, installed)) {throw new Error("Existing CppHeaderParser bytes differ from the pinned source distribution");}
  return installed;
}

/** Install only the verified CppHeaderParser package directory into the public producer's pinned Python path.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "runnerTemp" | "env">} paths @param {ReturnType<typeof rocmSourceStatePaths>} state */
function installRocmParserSource(paths, state) {
  const receipt = verifyRocmParserSource(state), extraction = fs.mkdtempSync(path.join(paths.runnerTemp, ".rocm-parser-install-"));
  try {
    extractState(state.pythonArchive, extraction, [rocmParserSource.directory]);
    const sourcePackage = path.join(extraction, rocmParserSource.directory, "CppHeaderParser"), sourceFiles = sdkSourceFiles(sourcePackage),
      required = ["__init__.py", "CppHeaderParser.py"];
    if (required.some(name => !Object.hasOwn(sourceFiles, name))) {throw new Error("Pinned CppHeaderParser source omitted the producer's required modules");}
    const destination = "/usr/local/lib/python3.12/dist-packages/CppHeaderParser";
    if (fs.lstatSync(path.dirname(destination), {throwIfNoEntry: false})?.isSymbolicLink()) {
      throw new Error("Refusing to use a redirected CppHeaderParser installation parent");
    }
    const preinstalled = admitRocmParserInstallation(destination, sourceFiles);
    if (preinstalled) {
      return {version: 1, sourceReceipt: {path: state.pythonSourceReceipt, sha256: fileHash(state.pythonSourceReceipt)},
        source: {path: state.pythonArchive, bytes: receipt.archive.bytes, sha256: fileHash(state.pythonArchive)},
        installed: {path: destination, files: preinstalled}};
    }
    const code = ["import pathlib,shutil,sys", "source=pathlib.Path(sys.argv[1]).resolve(strict=True)",
      "target=pathlib.Path('/usr/local/lib/python3.12/dist-packages/CppHeaderParser')",
      "if target.exists() or target.is_symlink(): raise SystemExit('CppHeaderParser target already exists')",
      "target.parent.mkdir(parents=True,exist_ok=True)", "shutil.copytree(source,target,symlinks=False)"].join("\n");
    const result = spawnSync("sudo", ["/usr/bin/python3.12", "-c", code, sourcePackage],
      {cwd: repo, env: paths.env, stdio: "inherit", timeout: 60_000});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error("Pinned offline CppHeaderParser installation failed", {cause: result.error});
    }
    const installed = admitRocmParserInstallation(destination, sourceFiles);
    if (!installed) {throw new Error("Pinned CppHeaderParser installation did not create its target directory");}
    return {version: 1, sourceReceipt: {path: state.pythonSourceReceipt, sha256: fileHash(state.pythonSourceReceipt)},
      source: {path: state.pythonArchive, bytes: receipt.archive.bytes, sha256: fileHash(state.pythonArchive)},
      installed: {path: destination, files: installed}};
  } finally {fs.rmSync(extraction, {recursive: true, force: true});}
}

/** Restore the original host package and pinned parser inputs required by strict ROCm admission.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "env" | "runnerTemp">} paths @param {ReturnType<typeof rocmHostStatePaths>} state */
function prepareRocmHostAdmission(paths, state) {
  provisionRocmHostPackages(paths);
  return installRocmParserSource(paths, state);
}

/** Admit the public static ROCm host-runtime SDK against this run's fresh COMGR/LLVM cohort.
 * @param {Pick<ReturnType<typeof rocmHostStatePaths>, "llvmSdk" | "sdkRoot" | "resourcesRoot" | "runtimeOutput">} state
 * @param {ReturnType<typeof rocmHostBuildConfig>} config */
function auditRocmHostSdk(state, config) {
  const receiptFile = path.join(state.runtimeOutput, "result.json"), sdk = nativeRocmHostInputs(receiptFile),
    record = readBuildJson(receiptFile);
  if (sdk.prefix !== path.join(state.runtimeOutput, "sdk") || record.comgr?.receipt !== config.comgrReceipt ||
      record.comgr?.roots?.llvm !== state.llvmSdk || record.comgr?.roots?.lld !== state.llvmSdk ||
      record.comgr?.roots?.device !== path.join(state.resourcesRoot, "install") ||
      !isDeepStrictEqual(record.hostCompilers, config.hostCompilers)) {
    throw new Error("Native ROCm host runtime does not bind this same-run LLVM, device and COMGR cohort");
  }
  return {sdk, record};
}

/** @typedef {{path:string,sha256:string,package:string,architecture:string,version:string,status:string}} RocmNoticeOrigin */
/** @typedef {RocmNoticeOrigin & {staged:string,stagedSha256:string}} RocmNoticeStagedOrigin */
/** @typedef {"drm"|"drmAmdgpu"|"elf"|"numa"} RocmNoticeKey */
/** @typedef {{osOrigins:Record<RocmNoticeKey,RocmNoticeOrigin>,stagedDependencies:Record<RocmNoticeKey,RocmNoticeStagedOrigin>,
 * sources:{files:Record<string,string>},sourceArchives:Record<string,{path:string,sha256:string}>,pins:{rocr:string}}} RocmNoticeReceipt */
/** @typedef {{package:string,version:string,architecture:string,sourcePath:string,archives:Array<{key:RocmNoticeKey,path:string,sha256:string}>}} RocmNoticePackageOrigin */
/** @typedef {RocmNoticePackageOrigin & {file:string,sha256:string,bytes:number}} RocmNoticePackageFile */
/** @typedef {{format:string,completeNativeNotices:boolean,runtimeReceipt:{file:string,sha256:string},packages:RocmNoticePackageFile[],
 * sourceLicense:{component:string,file:string,sha256:string,bytes:number,sourceArchive:{name:string,sha256:string},revision:string}}} RocmNoticeManifest */

/** Bind the static ROCm OS inputs to their exact Debian copyright bodies and the already-admitted hsakmt
 * source license. libelf/libnuma texts come from the sysroot packages; the upstream libdrm release has no
 * license file, so its text is the host libdrm-dev package's. This is an incomplete notice-source subset.
 * @param {RocmNoticeReceipt} record */
function rocmHostNoticeRows(record) {
  const packageInputs = /** @type {Array<[RocmNoticeKey,string]>} */ ([["drm", "libdrm-dev"], ["drmAmdgpu", "libdrm-dev"],
      ["elf", "libelf-dev"], ["numa", "libnuma-dev"]]), producer = require("./build-native-rocm-host-runtime");
  if (!record || !record.osOrigins || !record.stagedDependencies || !record.sourceArchives || !record.sources?.files ||
      !hasExactKeys(record.osOrigins, packageInputs.map(([key]) => key)) ||
      !hasExactKeys(record.stagedDependencies, packageInputs.map(([key]) => key))) {
    throw new Error("ROCm notice sources require the complete admitted static package-origin set");
  }
  /** @type {Map<string,RocmNoticePackageOrigin>} */ const grouped = new Map();
  for (const [key, packageName] of packageInputs) {
    const origin = record.osOrigins[key], staged = record.stagedDependencies[key];
    if (!origin || !staged || !hasExactKeys(origin, ["path", "sha256", "package", "architecture", "version", "status"]) ||
        origin.package !== producer.packageOwners[key] || origin.architecture !== "amd64" ||
        origin.status !== (origin.package === "libdrm" ? "source" : "sysroot") ||
        !path.isAbsolute(origin.path) || !/^[a-f0-9]{64}$/.test(origin.sha256 || "") || typeof origin.version !== "string" || !origin.version ||
        !hasExactKeys(staged, ["path", "sha256", "package", "architecture", "version", "status", "staged", "stagedSha256"])) {
      throw new Error("ROCm static archive origin is not bound to its installed package: " + key);
    }
    const {staged: stagedPath, stagedSha256, ...stagedOrigin} = staged;
    if (!isDeepStrictEqual(stagedOrigin, origin) || stagedSha256 !== origin.sha256 || !path.isAbsolute(stagedPath)) {
      throw new Error("ROCm static archive origin is not bound to its staged package input: " + key);
    }
    const row = grouped.get(packageName) || {package: packageName, version: origin.version, architecture: origin.architecture,
      sourcePath: "/usr/share/doc/" + packageName + "/copyright", archives: []};
    if (row.version !== origin.version || row.architecture !== origin.architecture) {
      throw new Error("ROCm static archives disagree on their owning package version: " + packageName);
    }
    row.archives.push({key, path: origin.path, sha256: origin.sha256}); grouped.set(packageName, row);
  }
  for (const row of grouped.values()) {row.archives.sort((a, b) => a.key.localeCompare(b.key));}
  return [...grouped.values()].sort((a, b) => a.package.localeCompare(b.package));
}

/** Capture the copyright bodies after the public ROCm producer has recorded the static archive origins.
 * @param {ReturnType<typeof rocmHostStatePaths>} state @param {ReturnType<typeof auditRocmHostSdk>} admitted
 * @param {NodeJS.ProcessEnv} env */
function captureRocmHostNoticeSources(state, admitted, env = process.env) {
  const packageRows = rocmHostNoticeRows(admitted.record), sysroot = requireLinuxSysroot(env), copied = [];
  for (const row of packageRows) {
    const originsMatch = () => row.archives.every(item => fileHash(item.path) === admitted.record.osOrigins[item.key].sha256);
    if (!row.archives.length || !originsMatch()) {
      throw new Error("ROCm static input changed between admission and copyright capture: " + row.package);
    }
    let captured;
    if (row.package === "libdrm-dev") {captured = capturePackageCopyright({root: state.hostSdkRoot}, row.package, row.version);} else {
      const source = path.join(sysroot.root, row.sourcePath), destination = path.join(state.hostSdkRoot, "licenses", row.package, "copyright");
      fs.mkdirSync(path.dirname(destination), {recursive: true, mode: 0o755});
      fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
      captured = {path: destination, sha256: fileHash(source)};
      if (fileHash(destination) !== captured.sha256) {throw new Error("Copied sysroot copyright differs from its source: " + row.package);}
    }
    const file = path.relative(state.hostSdkRoot, captured.path).split(path.sep).join("/");
    if (!originsMatch()) {
      throw new Error("ROCm static input changed while its copyright body was captured: " + row.package);
    }
    copied.push({...row, file, sha256: captured.sha256, bytes: fs.statSync(captured.path).size});
  }
  const licenseRelative = "source/rocr/libhsakmt/LICENSE.md", sourceLicense = path.join(state.runtimeOutput, licenseRelative),
    sourceRecord = admitted.record.sources.files[licenseRelative], sourceArchive = admitted.record.sourceArchives["rocr"];
  if (!sourceRecord || !sourceArchive || !/^[a-f0-9]{64}$/.test(sourceArchive.sha256 || "") ||
      !fs.lstatSync(sourceLicense, {throwIfNoEntry: false})?.isFile() || fs.lstatSync(sourceLicense).isSymbolicLink() ||
      fileHash(sourceLicense) !== sourceRecord || !fs.lstatSync(sourceArchive.path, {throwIfNoEntry: false})?.isFile() ||
      fileHash(sourceArchive.path) !== sourceArchive.sha256) {
    throw new Error("ROCm hsakmt license body is not present in the admitted same-run ROCr source snapshot");
  }
  const recordPath = path.join(state.runtimeOutput, "result.json"), manifest = {format: "mom-linux-rocm-notice-sources-v1",
    completeNativeNotices: false, runtimeReceipt: {file: "runtime/result.json", sha256: fileHash(recordPath)},
    packages: copied, sourceLicense: {component: "hsakmt-staticdrm", file: "runtime/" + licenseRelative,
      sha256: sourceRecord, bytes: fs.statSync(sourceLicense).size, sourceArchive: {name: "rocr", sha256: sourceArchive.sha256},
      revision: admitted.record.pins["rocr"]}};
  const manifestFile = path.join(state.hostSdkRoot, "rocm-notice-sources.json");
  fs.writeFileSync(manifestFile, JSON.stringify(manifest, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return auditRocmHostNoticeSources(state);
}

/** Re-admit notice-source bodies from the transported same-run ROCm host SDK.
 * @param {Pick<ReturnType<typeof rocmHostStatePaths>, "hostSdkRoot" | "runtimeOutput">} state */
function auditRocmHostNoticeSources(state) {
  const recordFile = path.join(state.runtimeOutput, "result.json"), manifestFile = path.join(state.hostSdkRoot, "rocm-notice-sources.json"),
    recordStat = fs.lstatSync(recordFile, {throwIfNoEntry: false}), manifestStat = fs.lstatSync(manifestFile, {throwIfNoEntry: false});
  if (!recordStat?.isFile() || recordStat.isSymbolicLink() || fs.realpathSync(recordFile) !== recordFile || recordStat.size > (16 << 20) ||
      !manifestStat?.isFile() || manifestStat.isSymbolicLink() || fs.realpathSync(manifestFile) !== manifestFile || manifestStat.size > (1 << 20)) {
    throw new Error("ROCm notice-source receipt or manifest is not a bounded canonical file");
  }
  const {value: parsedRecord, sha256: receiptHash} = readBuildJsonSnapshot(recordFile),
    record = /** @type {RocmNoticeReceipt} */ (parsedRecord),
    manifest = /** @type {RocmNoticeManifest} */ (readBuildJson(manifestFile)), expectedPackages = rocmHostNoticeRows(record);
  if (!hasExactKeys(manifest, ["format", "completeNativeNotices", "runtimeReceipt", "packages", "sourceLicense"]) ||
      manifest.format !== "mom-linux-rocm-notice-sources-v1" || manifest.completeNativeNotices !== false ||
      !isDeepStrictEqual(manifest.runtimeReceipt, {file: "runtime/result.json", sha256: receiptHash}) ||
      !Array.isArray(manifest.packages) || manifest.packages.length !== expectedPackages.length) {
    throw new Error("ROCm notice-source manifest is missing or not bound to the same-run producer receipt");
  }
  const packages = manifest.packages.map((/** @type {RocmNoticePackageFile} */ row, index) => {
    const expected = expectedPackages[index];
    if (!expected) {throw new Error("ROCm package copyright body is absent from its package-origin receipt");}
    const file = path.join(state.hostSdkRoot, "licenses", expected.package, "copyright"), stat = fs.lstatSync(file, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file || stat.size < 1 || stat.size > (2 << 20) ||
        !hasExactKeys(row, ["package", "version", "architecture", "sourcePath", "archives", "file", "sha256", "bytes"]) ||
        row.package !== expected.package || row.version !== expected.version || row.architecture !== expected.architecture ||
        row.sourcePath !== expected.sourcePath || !isDeepStrictEqual(row.archives, expected.archives) ||
        row.file !== "licenses/" + expected.package + "/copyright" || row.bytes !== stat.size ||
        !/^[a-f0-9]{64}$/.test(row.sha256 || "") || fileHash(file) !== row.sha256) {
      throw new Error("ROCm package copyright body or static archive origin changed: " + (expected?.package || "unknown"));
    }
    return {package: row.package, version: row.version, architecture: row.architecture, sourcePath: row.sourcePath,
      file: row.file, sha256: row.sha256, bytes: row.bytes, archives: row.archives};
  });
  const relative = "source/rocr/libhsakmt/LICENSE.md", sourceFile = path.join(state.runtimeOutput, relative), sourceArchive = record.sourceArchives?.["rocr"],
    source = manifest.sourceLicense, sourceSha256 = record.sources?.files?.[relative];
  if (!hasExactKeys(source, ["component", "file", "sha256", "bytes", "sourceArchive", "revision"]) ||
      source.component !== "hsakmt-staticdrm" || source.file !== "runtime/" + relative || source.sha256 !== sourceSha256 ||
      source.revision !== record.pins?.rocr || !sourceArchive || !isDeepStrictEqual(source.sourceArchive,
    {name: "rocr", sha256: sourceArchive.sha256}) || !/^[a-f0-9]{64}$/.test(sourceSha256 || "") ||
      !fs.lstatSync(sourceFile, {throwIfNoEntry: false})?.isFile() || fs.lstatSync(sourceFile).isSymbolicLink() ||
      fs.statSync(sourceFile).size !== source.bytes || fileHash(sourceFile) !== sourceSha256 ||
      !fs.lstatSync(sourceArchive.path, {throwIfNoEntry: false})?.isFile() || fileHash(sourceArchive.path) !== sourceArchive.sha256) {
    throw new Error("ROCm hsakmt license body differs from its admitted same-run source and archive");
  }
  return {format: manifest.format, completeNativeNotices: false,
    runtimeReceipt: {file: manifest.runtimeReceipt.file, sha256: receiptHash},
    manifest: {file: "rocm-notice-sources.json", sha256: fileHash(manifestFile)}, packages,
    sourceLicense: {component: source.component, file: source.file, sha256: source.sha256, bytes: source.bytes,
      sourceArchive: source.sourceArchive, revision: source.revision}};
}

/** Restore the same-run LLVM/COMGR cohort, provision pinned host inputs, and run the public producer offline.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildRocmHostSdk(paths) {
  const state = rocmHostStatePaths(paths), git = canonicalExecutable("/usr/bin/git"), expectedRoots = [state.sourceName, state.llvmSourceName,
    state.llvmSdkName, state.hipHeadersName, state.resourcesName, state.sdkName];
  for (const output of [state.hostSdkRoot, state.hostSdkTar]) {
    if (fs.existsSync(output)) {throw new Error("Native ROCm host SDK stage refuses an existing output: " + path.basename(output));}
  }
  extractState(state.incomingComgrTar, temp, expectedRoots);
  verifyRocmSources(paths, state, git);
  const env = {...sdkSourceEnvironment(paths.env), GIT_ALLOW_PROTOCOL: "file", GIT_NO_LAZY_FETCH: "1",
    GIT_TERMINAL_PROMPT: "0", GIT_OPTIONAL_LOCKS: "0"};
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", state.llvmSource], paths.env, 2 * 60 * 1000);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-rocm-host-" + paths.suffix, 15 * 60 * 1000);
  admitLinuxLlvmOutput(paths);
  fs.unlinkSync(state.incomingComgrTar);
  const parser = prepareRocmHostAdmission(paths, state), tools = rocmHostBuildTools(state);
  fs.mkdirSync(state.hostSdkRoot, {mode: 0o700});
  buildLibdrmSdk(state, paths.env);
  const config = rocmHostBuildConfig(state, tools, paths.env);
  auditRocmComgrSdk(state, {cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake});
  fs.writeFileSync(state.configFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  fs.writeFileSync(path.join(state.hostSdkRoot, "python-parser-install.json"), JSON.stringify(parser, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const buildEnv = {...env, MOM_BUILD_JOBS: ciBuildJobs(env), MOM_AOT_BUILD_ONLY: "1"};
  runNode("scripts/build-native-rocm-host-runtime.js", [state.configFile, state.runtimeOutput], buildEnv, 32 * 60 * 1000);
  const admitted = auditRocmHostSdk(state, config);
  captureRocmHostNoticeSources(state, admitted, paths.env);
  packState(state.hostSdkTar, temp, [...expectedRoots, state.hostSdkName]);
  listState(state.hostSdkTar, [...expectedRoots, state.hostSdkName]);
  console.log("[native-ci] ROCm host SDK admitted: " + admitted.sdk.libraries.length + " libraries, CPU-only build-only; no runtime or release qualification.");
}

module.exports = {acppCudaHipSdkBuildCommand, acppCudaHipSdkStatePaths, admitRocmParserInstallation, applyRocmHipHeadersPatch,
  assembleRocmHipHeaders, assertAcppCudaHipSdk, auditAcppCudaHipSdk, auditRocmComgrSdk, auditRocmHostNoticeSources,
  auditRocmHostSdk, buildAcppCudaHipSdk, buildRocmComgrFromInputs, buildRocmComgrSdk, buildRocmHostSdk,
  captureRocmHostNoticeSources, exportRocmSourceTree, hipHeaderConfigureCommand, installRocmParserSource,
  prepareRocmHostAdmission, provisionRocmHostPackages, rocmClrHipamdTreeish, rocmComgrBuildCommands, rocmComgrStatePaths,
  rocmHostBuildConfig, rocmHostBuildTools, rocmHostNoticeRows, rocmHostStatePaths, rocmResourcesConfig, buildLibdrmSdk};
