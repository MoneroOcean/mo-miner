"use strict";

// Assemble the already-produced Windows-target SDK/application graph for local WSL distribution.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk"), hostSdkBuilder = require("./build-local-native-windows-host-sdk"),
  windowsLlvmBuilder = require("./build-local-native-windows-llvm-sdk"), jitResourcesBuilder = require("./build-local-native-windows-jit-resources"),
  jitSdkBuilder = require("./build-local-native-windows-jit-sdk"), dpcppBuilder = require("./build-local-native-windows-dpcpp-sdk"),
  hostComponentBuilder = require("./build-local-native-windows-host-component"), gpuConfigBuilder = require("./build-local-native-windows-gpu-config"),
  gpuBuilder = require("./build-native-windows-gpu"), windowsHandoffs = require("./native-windows-controller-handoff"),
  windowsHostReader = require("./native-windows-host-sdk-inputs"), linuxHermesReader = require("./native-release-sdk-inputs"),
  applicationConfig = require("./native-windows-application-config"), windowsApplication = require("./native-windows-application"),
  windowsController = require("./build-native-windows-controller"), release = require("./build-release"),
  {checkNativeNoticeReview} = require("./check-native-notices");
const {hasExactBuildFields, requireBuildFields, verifyBuildDirectoryEntries, canonicalBuildFile, fileHash, readBuildJson} = require("./build-source-inputs");

const repo = path.resolve(__dirname, ".."),
  profiles = ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"],
  localId = /^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/;

/** @param {unknown} value @returns {value is Record<string, any>} */
function record(value) {return value !== null && typeof value === "object" && !Array.isArray(value);}

/** @param {string} workspace @param {string} output */
function localWindowsSdkStagePaths(workspace, output) {
  if (typeof output !== "string" || !path.isAbsolute(output) || path.resolve(output) !== output || /[\r\n\0]/.test(output)) {
    throw new Error("Local Windows SDK stage output must be a canonical absolute path");
  }
  const relative = path.relative(workspace, output), stat = fs.lstatSync(output, {throwIfNoEntry: false}), parent = path.dirname(output);
  if (!relative || relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative) || stat ||
      fs.realpathSync(parent) !== parent || !fs.statSync(parent).isDirectory()) {
    throw new Error("Local Windows SDK stage output must be a fresh workspace-owned directory; preserve any existing output");
  }
  const gpuConfigs = path.join(output, "gpu-configs"), gpuOutputs = path.join(output, "gpu");
  return {root: output, gpuConfigs, gpuOutputs, configFile: path.join(output, "native-windows-sdk.json")};
}

/** @param {unknown} result @param {string} workspace @param {string} label */
function assertWorkspaceResult(result, workspace, label) {
  if (!record(result) || result["workspace"] !== workspace) {throw new Error(label + " belongs to another workspace");}
}

/** Build or re-admit the local Windows SDK/application descriptor graph; no app link or execution occurs.
 * @param {unknown} input
 * @param {{verifyScope?:()=>unknown,resolveTools?:typeof local.localTools,environment?:NodeJS.ProcessEnv,
 * prepareWindowsSources?:typeof local.prepareOrAdmitWindowsSources,buildHostSdk?:typeof hostSdkBuilder.buildLocalNativeWindowsHostSdk,
 * admitHostSdk?:typeof windowsHostReader.windowsHostSdkInputs,admitControllerSupport?:(value:any)=>any,
 * buildWindowsLlvm?:typeof windowsLlvmBuilder.buildLocalNativeWindowsLlvmSdk,
 * buildJitResources?:typeof jitResourcesBuilder.buildLocalWindowsJitResources,
 * buildJitSdk?:typeof jitSdkBuilder.buildLocalNativeWindowsJitSdk,buildDpcppSdk?:typeof dpcppBuilder.buildLocalNativeWindowsDpcppSdk,
 * buildHostComponent?:typeof hostComponentBuilder.buildLocalNativeWindowsHostComponent,
 * composeGpuConfigs?:typeof gpuConfigBuilder.composeLocalWindowsGpuConfigs,buildGpu?:typeof gpuBuilder.buildNativeWindowsGpu,
 * readLinuxHermes?:typeof linuxHermesReader.nativeHermesInputs,readJson?:typeof readBuildJson,
 * composeDescriptor?:typeof applicationConfig.composeWindowsApplicationDescriptor,
 * admitApplication?:typeof windowsApplication.windowsApplicationInputs,
 * verifyApplicationFiles?:typeof windowsApplication.verifyWindowsApplicationFiles,
 * validateConfig?:typeof windowsController.validateWindowsControllerConfig,systemImports?:typeof windowsController.windowsSystemImports,
 * reviewNotices?:typeof checkNativeNoticeReview,validateNotice?:typeof release.noticeInput}} [dependencies]
 * @returns {Promise<any>} */
async function buildLocalNativeWindowsSdk(input, dependencies = {}) {
  const verifyScope = dependencies.verifyScope || local.verifyCurrentMemoryScope;
  verifyScope();
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local Windows SDK graph requires Linux x64 (WSL2)");}
  const args = requireBuildFields(input, ["controllerSupport", "coreArchive", "linuxSdk", "output", "tools", "workspace"], "Local Windows SDK input");
  if (typeof args["workspace"] !== "string") {throw new Error("Local Windows SDK workspace must be marked and canonical");}
  const workspace = local.admitWorkspace(args["workspace"], repo);
  if (workspace !== args["workspace"]) {throw new Error("Local Windows SDK workspace must be canonical and marked");}
  const jobs = local.localBuildJobs({...process.env, ...(dependencies.environment || {})}),
    paths = localWindowsSdkStagePaths(workspace, args["output"]), environment = {...process.env, ...(dependencies.environment || {}),
      MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: String(jobs), OMP_NUM_THREADS: String(jobs)},
    suppliedTools = requireBuildFields(args["tools"], ["cc", "cmake", "cxx", "git", "ninja", "systemdRun"], "Selected local tools"),
    selectedTools = (dependencies.resolveTools || local.localTools)(environment),
    tools = /** @type {ReturnType<typeof local.localTools>} */ (suppliedTools);
  if (!isDeepStrictEqual(tools, selectedTools)) {throw new Error("Local Windows SDK tools differ from the current pinned selection");}

  const support = local.supportPaths(workspace), linuxSdk = args["linuxSdk"];
  if (!record(linuxSdk)) {throw new Error("Admitted local Linux SDK result is required");}
  if (linuxSdk["workspace"] !== workspace || linuxSdk["compilerFetch"] !== support.compilerFetch || linuxSdk["compilerSources"] !== support.compilerSources ||
      linuxSdk["buildInputs"] !== support.buildInputs || linuxSdk["llvmSdk"] !== support.llvmSdk || linuxSdk["jobs"] !== jobs) {
    throw new Error("Local Windows SDK needs the current buildLocalNativeSdk result for this workspace and MOM_BUILD_JOBS");
  }
  const coreArchive = args["coreArchive"];
  canonicalBuildFile(coreArchive, "Original WSL-copied Windows Core archive", 256 << 20);
  if (!/\.lib$/i.test(coreArchive)) {throw new Error("Local Windows SDK Core input must be the original .lib archive");}
  canonicalBuildFile(path.join(path.dirname(coreArchive), "inputs.json"), "Original Windows Core producer receipt", 8 << 20);

  const prepareSources = dependencies.prepareWindowsSources || local.prepareOrAdmitWindowsSources,
    windowsSources = prepareSources(workspace, {environment}),
    expectedSourceRoot = path.join(workspace, "windows-sources"), expectedSourceReceipt = path.join(expectedSourceRoot, "sources.json");
  assertWorkspaceResult(windowsSources, workspace, "Prepared Windows sources");
  if (windowsSources.root !== expectedSourceRoot || windowsSources.receipt !== expectedSourceReceipt ||
      !/^[a-f0-9]{64}$/.test(windowsSources.receiptSha256 || "")) {
    throw new Error("Local Windows source preparation returned another source receipt");
  }

  const buildHostSdk = dependencies.buildHostSdk || hostSdkBuilder.buildLocalNativeWindowsHostSdk,
    windowsHostSdk = await buildHostSdk({workspace}, {verifyScope, environment}),
    hostSdk = {root: windowsHostSdk.root, pins: windowsHostSdk.pins};
  assertWorkspaceResult(windowsHostSdk, workspace, "Windows host SDK");
  if (hostSdk.root !== path.join(workspace, "windows-host-sdk", "sdk") ||
      !hasExactBuildFields(hostSdk.pins, ["descriptorSha256", "filesSha256", "overlaySha256", "resultSha256"]) ||
      Object.values(hostSdk.pins).some(hash => !/^[a-f0-9]{64}$/.test(hash || ""))) {
    throw new Error("Local Windows host-SDK producer returned invalid root or receipt pins");
  }
  const admitHostSdk = dependencies.admitHostSdk || windowsHostReader.windowsHostSdkInputs,
    hostAdmission = admitHostSdk(hostSdk.root, hostSdk.pins);
  if (hostAdmission.root !== hostSdk.root || !isDeepStrictEqual(hostAdmission.pins, hostSdk.pins)) {
    throw new Error("Local Windows host-SDK reader returned another admission");
  }

  const admitController = dependencies.admitControllerSupport || ((value) => dpcppBuilder.admitControllerSupport(value,
      windowsHandoffs.windowsLocalControllerHandoffInputs)), controller = admitController(args["controllerSupport"]);
  if (!localId.test(controller.id || "") || controller.id !== args["controllerSupport"]?.localBuildId ||
      controller.hermesInput?.localBuildId !== controller.id) {
    throw new Error("Local Windows controller handoffs do not retain their shared UUID identity");
  }
  const hermes = controller.hermesInput,
    buildWindowsLlvm = dependencies.buildWindowsLlvm || windowsLlvmBuilder.buildLocalNativeWindowsLlvmSdk,
    windowsLlvm = buildWindowsLlvm({workspace, windowsHostSdk: hostSdk, hermes, tools}, {verifyScope, environment});
  assertWorkspaceResult(windowsLlvm, workspace, "Windows LLVM SDK");
  if (windowsLlvm.output !== path.join(workspace, "windows-llvm-sdk", "output") ||
      windowsLlvm.descriptor !== path.join(workspace, "windows-llvm-sdk", "descriptor.json")) {
    throw new Error("Local Windows LLVM builder returned another output or descriptor");
  }

  const resourceTools = {cmake: tools.cmake, ninja: tools.ninja, cc: tools.cc, cxx: tools.cxx},
    buildJitResources = dependencies.buildJitResources || jitResourcesBuilder.buildLocalWindowsJitResources,
    windowsJitResources = buildJitResources(linuxSdk, resourceTools, {verifyScope, environment});
  assertWorkspaceResult(windowsJitResources, workspace, "Windows JIT resources");
  const buildJitSdk = dependencies.buildJitSdk || jitSdkBuilder.buildLocalNativeWindowsJitSdk,
    windowsJitSdk = await buildJitSdk({workspace, windowsHostSdk: hostSdk, hermes, tools, windowsLlvm, windowsJitResources},
      {verifyScope, resolveTools: () => tools, environment});
  assertWorkspaceResult(windowsJitSdk, workspace, "Windows JIT SDK");
  if (windowsJitSdk.jobs !== jobs) {throw new Error("Local Windows JIT SDK result differs from this run's MOM_BUILD_JOBS");}

  const buildDpcppSdk = dependencies.buildDpcppSdk || dpcppBuilder.buildLocalNativeWindowsDpcppSdk,
    windowsDpcppSdk = await buildDpcppSdk({workspace, tools, windowsHostSdk: hostSdk, windowsLlvm, windowsJitResources,
      windowsJitSdk, controllerSupport: args["controllerSupport"]}, {verifyScope, resolveTools: () => tools, environment});
  assertWorkspaceResult(windowsDpcppSdk, workspace, "Windows DPC++ SDK");
  if (windowsDpcppSdk.jobs !== jobs) {throw new Error("Local Windows DPC++ SDK result differs from this run's MOM_BUILD_JOBS");}
  const buildHostComponent = dependencies.buildHostComponent || hostComponentBuilder.buildLocalNativeWindowsHostComponent,
    windowsHostComponent = await buildHostComponent({workspace, tools, windowsHostSdk: hostSdk, hermes},
      {verifyScope, resolveTools: () => tools, environment});
  assertWorkspaceResult(windowsHostComponent, workspace, "Windows host component");
  if (windowsHostComponent.jobs !== jobs || windowsHostComponent.componentRoot !== path.join(workspace, "windows-host-component", "component") ||
      !/\.lib$/i.test(windowsHostComponent.amdhip64?.amdhip64 || "")) {
    throw new Error("Local Windows host component lacks this run's MOM_BUILD_JOBS producer result or AMDHIP driver import");
  }

  const gpuInput = {workspace, windowsSdkRoot: hostSdk.root, windowsSdkPins: hostSdk.pins,
      windowsSourceSha256: windowsSources.receiptSha256, tools},
    composeGpuConfigs = dependencies.composeGpuConfigs || gpuConfigBuilder.composeLocalWindowsGpuConfigs,
    gpuConfigs = composeGpuConfigs(gpuInput, {verifyScope, resolveTools: () => tools});
  if (!record(gpuConfigs?.profiles) || !isDeepStrictEqual(Object.keys(gpuConfigs.profiles), profiles) ||
      !record(gpuConfigs.admitted) || gpuConfigs.admitted.workspace !== workspace || gpuConfigs.buildOnly !== true ||
      gpuConfigs.executed !== false || gpuConfigs.publicationQualified !== false) {
    throw new Error("Local Windows GPU profile composer returned an incomplete five-profile cohort");
  }
  for (const profile of profiles) {
    const selected = gpuConfigs.profiles[profile];
    if (!selected || !record(selected) || !Buffer.isBuffer(selected.configBytes) || selected.configBytes.length < 1 || selected.configBytes.length > 8 << 20 ||
        !/^[a-f0-9]{64}$/.test(selected.configSha256 || "") || fileHashOf(selected.configBytes) !== selected.configSha256 ||
        selected.config?.profile !== profile) {
      throw new Error("Local Windows GPU profile config is malformed: " + profile);
    }
  }

  const stageParent = path.dirname(paths.root);
  if (fs.realpathSync(stageParent) !== stageParent) {throw new Error("Local Windows SDK stage parent is redirected");}
  fs.mkdirSync(paths.root, {mode: 0o700}); fs.mkdirSync(paths.gpuConfigs, {mode: 0o700}); fs.mkdirSync(paths.gpuOutputs, {mode: 0o700});
  const buildGpu = dependencies.buildGpu || gpuBuilder.buildNativeWindowsGpu, gpuReceipts = [];
  for (const profile of profiles) {
    const selected = gpuConfigs.profiles[profile], configFile = path.join(paths.gpuConfigs, profile + ".json"),
      output = path.join(paths.gpuOutputs, profile);
    if (!selected) {throw new Error("Local Windows GPU config disappeared: " + profile);}
    fs.writeFileSync(configFile, selected.configBytes, {flag: "wx", mode: 0o444});
    if (fileHash(configFile) !== selected.configSha256) {throw new Error("Windows GPU config changed while staging: " + profile);}
    const produced = buildGpu(configFile, output, environment), receiptFile = path.join(output, "result.json");
    canonicalBuildFile(receiptFile, "Windows GPU producer receipt");
    const receipt = (dependencies.readJson || readBuildJson)(receiptFile);
    if (!record(produced) || !isDeepStrictEqual(produced, receipt) || produced.version !== 1 || produced.profile !== profile ||
        produced.factory !== "mom_native_gpu_" + profile.replaceAll("-", "_") || produced.configFile !== configFile ||
        produced.configSha256 !== selected.configSha256 || fileHash(configFile) !== selected.configSha256) {
      throw new Error("Original Windows GPU producer did not retain the selected config and receipt: " + profile);
    }
    gpuReceipts.push(receiptFile);
  }

  const prepareAgain = prepareSources(workspace, {environment}), controllerAgain = admitController(args["controllerSupport"]),
    hostAgain = admitHostSdk(hostSdk.root, hostSdk.pins);
  if (!isDeepStrictEqual(prepareAgain, windowsSources) || !isDeepStrictEqual(controllerAgain, controller) ||
      hostAgain.root !== hostSdk.root || !isDeepStrictEqual(hostAgain.pins, hostSdk.pins)) {
    throw new Error("Local Windows SDK source, controller handoff or host-SDK inputs changed during profile compilation");
  }
  const readLinuxHermes = dependencies.readLinuxHermes || linuxHermesReader.nativeHermesInputs,
    hermesAot = readLinuxHermes(path.join(support.hermesSdk, "result.json")), admitted = gpuConfigs.admitted,
    application = {core: coreArchive, gpuReceipts, defaultCompiler: "dpcpp",
      tools: {llvmAr: path.join(support.llvmSdk, "sdk/bin/llvm-ar"), llvmObjcopy: path.join(support.llvmSdk, "sdk/bin/llvm-objcopy")},
      sdk: {libraries: [], wholeArchives: [], objects: [], includeRoots: [path.join(admitted.cudaRoot, "include"), admitted.hip.include],
        nvmlHeader: path.join(admitted.cudaRoot, "include/nvml.h"), hipDriverLibrary: windowsHostComponent.amdhip64.amdhip64,
        dpcppComponent: windowsDpcppSdk.component.root, hostComponent: windowsHostComponent.componentRoot,
        cudaImages: admitted.cudaImages.include}},
    llvmDescriptor = (dependencies.readJson || readBuildJson)(windowsLlvm.descriptor),
    composeDescriptor = dependencies.composeDescriptor || applicationConfig.composeWindowsApplicationDescriptor,
    descriptor = composeDescriptor({llvmDescriptor, io: controller.io, hermes: controller.hermes, hermesAot,
      tools: {shermaot: hermesAot.artifacts?.["compiler"], llvmReadobj: path.join(admitted.tools.toolRoot, "llvm-readobj")},
      allowedImports: (dependencies.systemImports || windowsController.windowsSystemImports)(), application}),
    reviewNotices = dependencies.reviewNotices || checkNativeNoticeReview,
    noticeReview = reviewNotices(), noticeFile = path.join(repo, "THIRD-PARTY-NOTICES.txt"),
    notices = {path: noticeFile, sha256: noticeReview.notice?.sha256},
    validateNotice = dependencies.validateNotice || release.noticeInput;
  if (!/^[a-f0-9]{64}$/.test(notices.sha256 || "")) {throw new Error("Current distribution notices have no reviewed SHA-256");}
  validateNotice(notices);
  const config = {...descriptor, notices}, validateConfig = dependencies.validateConfig || windowsController.validateWindowsControllerConfig,
    validated = validateConfig(config), admitApplication = dependencies.admitApplication || windowsApplication.windowsApplicationInputs;
  if (!record(config.application)) {throw new Error("Composed Windows release config lacks its application descriptor");}
  const applicationAdmission = admitApplication(config.application, repo);
  if (!isDeepStrictEqual(validated, config) || applicationAdmission.core !== coreArchive || !Array.isArray(applicationAdmission.graphs) ||
      applicationAdmission.graphs.length !== profiles.length) {
    throw new Error("Local Windows release config or full application reader rejected the composed graph");
  }
  (dependencies.verifyApplicationFiles || windowsApplication.verifyWindowsApplicationFiles)(applicationAdmission.files);
  const finalNoticeReview = reviewNotices();
  if (finalNoticeReview.notice?.sha256 !== notices.sha256) {throw new Error("Reviewed distribution notices changed during Windows SDK preparation");}
  validateNotice(notices);

  const expectedConfigs = profiles.map(profile => profile + ".json");
  verifyBuildDirectoryEntries(paths.gpuConfigs, expectedConfigs, "Local Windows GPU config stage");
  verifyBuildDirectoryEntries(paths.gpuOutputs, profiles, "Local Windows GPU output stage");
  const descriptorBytes = Buffer.from(JSON.stringify(config, null, 2) + "\n");
  if (descriptorBytes.length < 2 || descriptorBytes.length > 8 << 20) {throw new Error("Windows SDK descriptor exceeds its bound");}
  fs.writeFileSync(paths.configFile, descriptorBytes, {flag: "wx", mode: 0o444});
  if (!fs.readFileSync(paths.configFile).equals(descriptorBytes) || fileHash(paths.configFile) !== fileHashOf(descriptorBytes)) {
    throw new Error("Windows SDK descriptor changed while staging");
  }
  const reread = (dependencies.readJson || readBuildJson)(paths.configFile);
  if (!isDeepStrictEqual(reread, config) || !isDeepStrictEqual(validateConfig(reread), config)) {
    throw new Error("Staged Windows SDK descriptor differs from the validated release config");
  }
  validateNotice(reread.notices);
  (dependencies.verifyApplicationFiles || windowsApplication.verifyWindowsApplicationFiles)(applicationAdmission.files);
  verifyBuildDirectoryEntries(paths.root, ["gpu", "gpu-configs", "native-windows-sdk.json"], "Local Windows SDK stage");
  return {workspace, root: paths.root, configFile: paths.configFile, tools, windowsHostSdk, windowsLlvm,
    windowsJitResources, windowsJitSdk, windowsDpcppSdk, windowsHostComponent, gpuReceipts, applicationAdmission,
    jobs, buildOnly: true, compiled: true, executed: false, publicationQualified: false};
}

/** @param {Buffer} value */
function fileHashOf(value) {return require("node:crypto").createHash("sha256").update(value).digest("hex");}

module.exports = {localWindowsSdkStagePaths, buildLocalNativeWindowsSdk, profiles};
