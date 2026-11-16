"use strict";

// Build or re-admit the original static Windows host component from local WSL inputs.
// All CMake, compiler, archive and provenance rules remain in the public producers.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"),
  {randomUUID} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk"), localRocm = require("./build-local-native-rocm-sdk"),
  localAcpp = require("./build-local-native-acpp-sdk"), localWindowsLlvm = require("./build-local-native-windows-llvm-sdk"),
  nativeCi = require("./prepare-native-ci"), dpcppInputs = require("./native-dpcpp-cuda-inputs"),
  linuxLlvm = require("./build-native-llvm21-sdk"), windowsSdkReader = require("./native-windows-host-sdk-inputs"),
  controllerHandoff = require("./native-windows-controller-handoff"), rocmResources = require("./build-native-rocm-resources"),
  windowsResources = require("./build-native-windows-comgr-sdk"), windowsHiprtc = require("./build-native-windows-hiprtc-sdk"),
  windowsAcpp = require("./build-native-windows-acpp-sdk"), windowsCudart = require("./prepare-native-cudart-windows"),
  windowsAmdhip = require("./prepare-native-amdhip-windows"), hostComponent = require("./native-windows-host-component");
const {fileHash, readBuildJson} = require("./build-source-inputs");
const repo = fs.realpathSync(path.resolve(__dirname, "..")), jobs = 2,
  localId = /^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i,
  digest = /^[a-f0-9]{64}$/;

/** @param {unknown} value @param {string[]} keys @param {string} label */
function exactObject(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value) || !isDeepStrictEqual(Object.keys(value).sort(), [...keys].sort())) {
    throw new Error(label + " has unknown or missing fields");
  }
  return /** @type {Record<string, any>} */ (value);
}

/** @param {unknown} value @returns {value is string} */
function absolute(value) {
  return typeof value === "string" && path.isAbsolute(value) && path.resolve(value) === value && !/[\r\n\0]/.test(value);
}

/** @param {string} directory @param {string} label @param {boolean} [allowMissing] */
function realDirectory(directory, label, allowMissing = false) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!stat && allowMissing) {return false;}
  if (!stat?.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
    throw new Error(label + " is missing or not a canonical directory: " + directory);
  }
  return true;
}

/** @param {string} file @param {string} label */
function regularFile(file, label) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file) {
    throw new Error(label + " is missing or not a canonical regular file: " + file);
  }
  return stat;
}

/** Reject unknown, interrupted or redirected local output without repairing it. @param {string} directory
 * @param {string[]} expected @param {string} label */
function exactEntries(directory, expected, label) {
  const actual = fs.readdirSync(directory).sort();
  if (!isDeepStrictEqual(actual, [...expected].sort())) {
    throw new Error(label + " is incomplete or contains unknown state; preserve it for inspection");
  }
}

/** Stable paths under the marked workspace; original producer receipts are the only build registry.
 * @param {string} workspace @param {string} archiveName */
function localWindowsHostComponentPaths(workspace, archiveName) {
  if (!absolute(workspace) || !/^[A-Za-z0-9][A-Za-z0-9._+-]{0,159}\.zip$/.test(archiveName || "")) {
    throw new Error("Local Windows host-component workspace or pinned CUDA archive name is invalid");
  }
  const root = path.join(workspace, "windows-host-component"), archives = path.join(root, "archives");
  return {workspace, root, archives, archive: path.join(archives, archiveName), resources: path.join(root, "resources"),
    resourcesConfig: path.join(root, "resources-config.json"), comgr: path.join(root, "comgr"), hiprtcSource: path.join(root, "hiprtc-source"),
    hiprtc: path.join(root, "hiprtc"), hiprtcConfig: path.join(root, "hiprtc-config.json"), cudart: path.join(root, "cudart"),
    amdhip: path.join(root, "amdhip"), acpp: path.join(root, "acpp"), component: path.join(root, "component")};
}

/** @param {ReturnType<typeof localWindowsHostComponentPaths>} paths @param {boolean} allowMissing @param {string} archiveName */
function checkManagedRoot(paths, allowMissing, archiveName) {
  if (!realDirectory(paths.root, "Local Windows host-component managed root", allowMissing)) {return false;}
  exactEntries(paths.root, ["acpp", "amdhip", "archives", "comgr", "component", "cudart", "hiprtc", "hiprtc-config.json",
    "hiprtc-source", "resources", "resources-config.json"], "Local Windows host-component managed root");
  for (const directory of [paths.archives, paths.resources, paths.comgr, paths.hiprtcSource, paths.hiprtc, paths.cudart,
    paths.amdhip, paths.acpp, paths.component]) {realDirectory(directory, "Local Windows host-component " + path.basename(directory));}
  exactEntries(paths.archives, [archiveName], "Local Windows host-component archive cache");
  regularFile(paths.archive, "Pinned Windows CUDA archive");
  regularFile(paths.resourcesConfig, "Windows ROCm resource config");
  regularFile(paths.hiprtcConfig, "Windows HIPRTC config");
  return true;
}

/** Serialize one stable descriptor without overwriting prior state. @param {string} file @param {unknown} value */
function writeJsonOnce(file, value) {
  const bytes = Buffer.from(JSON.stringify(value, null, 2) + "\n");
  if (bytes.length < 2 || bytes.length > 8 << 20) {throw new Error("Local Windows host-component descriptor exceeds its bound");}
  fs.writeFileSync(file, bytes, {flag: "wx", mode: 0o600});
  return fileHash(file);
}

/** Require byte-identical stable config; a changed config never repairs a completed output.
 * @param {string} file @param {unknown} expected @param {string} label */
function verifyStableJson(file, expected, label) {
  regularFile(file, label);
  const bytes = Buffer.from(JSON.stringify(expected, null, 2) + "\n");
  if (!fs.readFileSync(file).equals(bytes)) {throw new Error(label + " is stale; preserve it for inspection");}
  return fileHash(file);
}

/** Preserve the original local ROCm CLR commit in a normal checkout required by the HIPRTC producer.
 * @param {string} source @param {string} output @param {string} git @param {NodeJS.ProcessEnv} env
 * @param {(command:string,args:string[],options:Record<string,unknown>)=>any} execute */
function prepareLocalHiprtcSource(source, output, git, env, execute = spawnSync) {
  const revision = windowsHiprtc.clrRevision;
  if (!absolute(source) || !absolute(output) || !absolute(git) || fs.lstatSync(output, {throwIfNoEntry: false})) {
    throw new Error("Local HIPRTC checkout requires the pinned bare CLR source and a new output path");
  }
  /** @param {string[]} args @param {string} label */
  const run = (args, label) => {
    const result = execute(git, args, {cwd: repo, env, encoding: "utf8", timeout: 120000, maxBuffer: 4 << 20});
    if (result.error || result.signal || result.status !== 0 || typeof result.stdout !== "string") {
      throw new Error("Local HIPRTC " + label + " failed", {cause: result.error});
    }
    return result.stdout.trim();
  };
  if (run(["-C", source, "rev-parse", "--verify", revision + "^{commit}"], "source pin") !== revision) {
    throw new Error("Local ROCm CLR bare source does not contain the pinned HIPRTC commit");
  }
  run(["clone", "--shared", "--no-checkout", "--", source, output], "source clone");
  run(["-C", output, "checkout", "--detach", revision], "source checkout");
  verifyLocalHiprtcSource(output, git, env, execute);
  return output;
}

/** @param {string} source @param {string} git @param {NodeJS.ProcessEnv} env
 * @param {(command:string,args:string[],options:Record<string,unknown>)=>any} execute */
function verifyLocalHiprtcSource(source, git, env, execute = spawnSync) {
  const revision = windowsHiprtc.clrRevision;
  if (!absolute(source) || !absolute(git)) {throw new Error("Local HIPRTC source paths must be canonical and absolute");}
  realDirectory(source, "Local HIPRTC source checkout");
  /** @param {string[]} args */
  const run = args => {
    const result = execute(git, args, {cwd: repo, env, encoding: "utf8", timeout: 120000, maxBuffer: 4 << 20});
    if (result.error || result.signal || result.status !== 0 || typeof result.stdout !== "string") {
      throw new Error("Local HIPRTC pinned source check failed", {cause: result.error});
    }
    return result.stdout.trim();
  };
  if (run(["-C", source, "rev-parse", "HEAD"]) !== revision || run(["-C", source, "diff", "--quiet", "HEAD", "--", "hipamd", "rocclr"]) !== "" ||
      run(["-C", source, "status", "--porcelain=v1", "--untracked-files=all", "--", "hipamd", "rocclr"]) !== "") {
    throw new Error("Local HIPRTC source checkout is changed or not the original pinned CLR revision");
  }
  return {source, revision};
}

/** Download only the public pinned CUDA Runtime archive. @param {string} output @param {any} catalog
 * @param {(command:string,args:string[],options:Record<string,unknown>)=>any} [execute] */
function downloadWindowsCudart(output, catalog, execute = spawnSync) {
  if (!absolute(output) || !catalog?.archive || !Number.isSafeInteger(catalog.archive.bytes) || catalog.archive.bytes < 1 ||
      !digest.test(catalog.archive.sha256 || "") || !/^https:\/\//.test(catalog.archive.url || "")) {
    throw new Error("Pinned Windows CUDA Runtime download descriptor is invalid");
  }
  const result = execute("/usr/bin/curl", ["--fail", "--show-error", "--location", "--proto", "=https", "--proto-redir", "=https",
    "--max-redirs", "5", "--retry", "3", "--connect-timeout", "15", "--max-time", "300", "--max-filesize", String(catalog.archive.bytes),
    "--output", output, catalog.archive.url], {cwd: path.dirname(output), env: {PATH: "/usr/bin:/bin", LC_ALL: "C", LANG: "C"},
    stdio: "inherit", timeout: 305000});
  if (result.error || result.signal || result.status !== 0) {throw new Error("Pinned Windows CUDA Runtime download failed", {cause: result.error});}
  regularFile(output, "Pinned Windows CUDA Runtime archive");
  if (fs.statSync(output).size !== catalog.archive.bytes || fileHash(output) !== catalog.archive.sha256) {
    throw new Error("Downloaded Windows CUDA Runtime archive differs from its public pin");
  }
  return output;
}

/** @param {ReturnType<typeof localRocm.localRocmSdkContext>} rocmContext */
function observedRocmResult(rocmContext) {
  const p = rocmContext.paths;
  /** @param {string} directory */
  const exists = directory => Boolean(fs.lstatSync(directory, {throwIfNoEntry: false}));
  return localRocm.rocmSdkResult(rocmContext, {reusedSources: exists(p.source),
    reusedComgr: [p.hipHeadersRoot, p.resourcesRoot, p.sdkRoot, p.comgrAdmission].every(exists),
    reusedHost: [p.hostSdkRoot, p.runtimeOutput, p.hostAdmission].every(exists)});
}

/** Re-admit original workspace source/LLVM/HIP/DPC++ inputs without rebuilding them.
 * @param {string} workspace @param {ReturnType<typeof local.localTools>} tools @param {NodeJS.ProcessEnv} env
 * @param {{verifyLinuxLlvm?:typeof linuxLlvm.verifyLinuxLlvmAdmission,auditRocmComgrInputs?:typeof localRocm.auditLocalRocmComgrInputs,
 * verifyRocmInputs?:typeof localRocm.verifyRocmBuildInputs,auditRocmSources?:typeof nativeCi.verifyRocmSources,
 * auditRocmComgr?:typeof nativeCi.auditRocmComgrSdk,auditDpcppCuda?:typeof dpcppInputs.auditDpcppCudaInputs,
 * prepareAcppSources?:typeof localAcpp.prepareLocalAcppSources,run?:typeof local.runCommand,verifyAcppSource?:typeof nativeCi.verifyAcppSource}} dependencies */
function admitWorkspaceInputs(workspace, tools, env, dependencies) {
  const support = local.supportPaths(workspace), verifyLinux = dependencies.verifyLinuxLlvm || linuxLlvm.verifyLinuxLlvmAdmission,
    linux = verifyLinux(support.llvmSdk);
  if (linux?.role !== "linux-llvm21-source-sdk" || linux.stage !== "audited" || linux.buildOnly !== true || linux.compiled !== true ||
      linux.executed !== false || linux.publicationQualified !== false || linux.sdk?.path !== path.join(support.llvmSdk, "sdk")) {
    throw new Error("Local Windows host component requires the audited original Linux LLVM 21.1.8 SDK");
  }
  const rocmContext = localRocm.localRocmSdkContext(localRocm.localRocmSdkContextFromWorkspace(workspace)),
    rocmTools = {cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake},
    auditRocm = dependencies.auditRocmComgrInputs || localRocm.auditLocalRocmComgrInputs,
    rocmAuditDependencies = /** @type {NonNullable<Parameters<typeof localRocm.auditLocalRocmComgrInputs>[2]>} */ ({});
  if (dependencies.verifyRocmInputs) {rocmAuditDependencies.verifyInputs = dependencies.verifyRocmInputs;}
  rocmAuditDependencies.verifySources = dependencies.auditRocmSources || nativeCi.verifyRocmSources;
  if (dependencies.auditRocmComgr) {rocmAuditDependencies.auditComgr = dependencies.auditRocmComgr;}
  const rocmAdmission = auditRocm(rocmContext, rocmTools, rocmAuditDependencies);
  const rocmState = rocmAdmission.state, hipHeaders = rocmAdmission.hip,
    dpcppState = local.localDpcppCudaState(support), dpcppManifest = readBuildJson(dpcppState.manifest),
    dpcpp = (dependencies.auditDpcppCuda || dpcppInputs.auditDpcppCudaInputs)(dpcppState, dpcppManifest),
    rocmResult = observedRocmResult(rocmContext), acppContext = localAcpp.localAcppSdkContextFromWorkspace(workspace, rocmResult),
    acppTools = {git: tools.git, cmake: tools.cmake, ninja: tools.ninja, cc: tools.cc, cxx: tools.cxx},
    prepareAcppSources = dependencies.prepareAcppSources || localAcpp.prepareLocalAcppSources;
  prepareAcppSources(acppContext, acppTools, env, dependencies.run || local.runCommand);
  const acppPaths = localAcpp.localAcppSdkPaths(workspace), acppSource = (dependencies.verifyAcppSource || nativeCi.verifyAcppSource)({env}, {
    source: acppPaths.fetch, preparedReceipt: acppPaths.preparedReceipt, preparedCheckout: acppPaths.preparedCheckout}, tools.git);
  realDirectory(acppPaths.preparedCheckout, "Pinned local AdaptiveCpp checkout");
  return {support, linux, rocmContext, rocmState, hipHeaders, rocmAdmission, dpcppState, dpcppManifest, dpcpp, rocmResult, acppPaths, acppSource};
}

/** @param {any} result @param {ReturnType<typeof localWindowsLlvm.localWindowsLlvmSdkPaths>} paths @param {string} workspace */
function validateWindowsLlvmResult(result, paths, workspace) {
  if (!result || result.workspace !== workspace || result.root !== paths.root || result.output !== paths.output || result.descriptor !== paths.descriptor ||
      result.provenance !== paths.provenance || result.jobs !== jobs || result.buildOnly !== true || result.compiled !== true ||
      result.executed !== false || result.publicationQualified !== false || !localId.test(result.admission?.admission?.generationId || "")) {
    throw new Error("Local Windows LLVM result is incomplete or belongs to another workspace");
  }
  regularFile(paths.descriptor, "Local Windows LLVM descriptor");
  realDirectory(paths.output, "Local Windows LLVM producer output");
  return result;
}

/** @param {any} first @param {any} second */
function sameWindowsLlvm(first, second) {
  const fields = ["workspace", "root", "output", "descriptor", "provenance", "archive", "descriptorSha256", "provenanceSha256", "jobs",
    "buildOnly", "compiled", "executed", "publicationQualified"];
  return fields.every(key => isDeepStrictEqual(first[key], second[key])) &&
    first.admission?.admission?.generationId === second.admission?.admission?.generationId &&
    fileHash(first.descriptor) === first.descriptorSha256 && fileHash(second.descriptor) === second.descriptorSha256 &&
    fileHash(first.provenance) === first.provenanceSha256 && fileHash(second.provenance) === second.provenanceSha256;
}

/** Build/re-admit the original WSL static Windows host-component path.
 * Input is the marked native workspace, the real local Windows SDK/Hermes handoffs, and the exact selected tools.
 * No CI run identity is created. Partial or stale managed output is preserved and rejected.
 * @param {{workspace:string,windowsHostSdk:{root:string,pins:{resultSha256:string,descriptorSha256:string,filesSha256:string,overlaySha256:string}},
 * hermes:{root:string,localBuildId:string,handoffSha256:string},tools:ReturnType<typeof local.localTools>}} input
 * @param {{verifyScope?:()=>unknown,admitWorkspace?:typeof local.admitWorkspace,resolveTools?:typeof local.localTools,
 * readWindowsHostSdk?:typeof windowsSdkReader.windowsHostSdkInputs,readHermes?:typeof controllerHandoff.windowsLocalControllerHandoffInputs,
 * verifyLinuxLlvm?:typeof linuxLlvm.verifyLinuxLlvmAdmission,verifyRocmInputs?:typeof localRocm.verifyRocmBuildInputs,
 * auditRocmSources?:typeof nativeCi.verifyRocmSources,auditRocmComgrInputs?:typeof localRocm.auditLocalRocmComgrInputs,
 * auditRocmComgr?:typeof nativeCi.auditRocmComgrSdk,verifyAcppSource?:typeof nativeCi.verifyAcppSource,
 * auditDpcppCuda?:typeof dpcppInputs.auditDpcppCudaInputs,prepareAcppSources?:typeof localAcpp.prepareLocalAcppSources,
 * run?:typeof local.runCommand,buildWindowsLlvm?:typeof localWindowsLlvm.buildLocalNativeWindowsLlvmSdk,
 * composeResourcesConfig?:typeof nativeCi.rocmResourcesConfig,buildResources?:typeof rocmResources.buildResources,
 * verifyDeviceResources?:typeof windowsResources.verifyWindowsDeviceResources,
 * configureComgr?:typeof windowsResources.configureWindowsComgrSdk,buildComgr?:typeof windowsResources.buildWindowsComgrSdk,
 * prepareHiprtcSource?:typeof prepareLocalHiprtcSource,verifyHiprtcSource?:typeof verifyLocalHiprtcSource,
 * buildHiprtc?:typeof windowsHiprtc.buildNativeWindowsHiprtcSdk,buildAcpp?:typeof windowsAcpp.buildWindowsAcppStaticHost,
 * cudartCatalog?:typeof windowsCudart.windowsCudartCatalog,downloadCudart?:typeof downloadWindowsCudart,
 * prepareCudart?:typeof windowsCudart.prepareWindowsCudart,readCudart?:typeof windowsCudart.windowsCudartInputs,
 * prepareAmdhip?:typeof windowsAmdhip.prepareLocalAmdhipWindows,amdhipCatalog?:typeof windowsAmdhip.windowsAmdhipCatalog,
 * readAmdhip?:typeof windowsAmdhip.windowsAmdhipInputs,
 * verifyProducers?:typeof hostComponent.verifyWindowsHostComponentProducers,matchComponent?:typeof hostComponent.matchWindowsHostComponent,
 * readComponent?:typeof hostComponent.windowsHostComponentInputs,exportComponent?:typeof hostComponent.exportWindowsHostComponent,
 * executeGit?:(command:string,args:string[],options:Record<string,unknown>)=>any,executeDownload?:typeof spawnSync,
 * environment?:NodeJS.ProcessEnv}} [dependencies] */
async function buildLocalNativeWindowsHostComponent(input, dependencies = {}) {
  (dependencies.verifyScope || local.verifyCurrentMemoryScope)();
  if (process.platform !== "linux" || process.arch !== "x64") {
    throw new Error("Local Windows host-component production requires Linux x64 (WSL2)");
  }
  exactObject(input, ["hermes", "tools", "windowsHostSdk", "workspace"], "Local Windows host-component input");
  if (!absolute(input.workspace)) {throw new Error("Local Windows host-component workspace must be canonical and absolute");}
  const hostSdk = /** @type {typeof input.windowsHostSdk} */ (exactObject(input.windowsHostSdk, ["root", "pins"], "Local Windows host-SDK input")),
    hermes = /** @type {typeof input.hermes} */ (exactObject(input.hermes, ["root", "localBuildId", "handoffSha256"], "Local Windows Hermes input")),
    tools = /** @type {typeof input.tools} */ (exactObject(input.tools, ["cc", "cmake", "cxx", "git", "ninja", "systemdRun"], "Local Windows host-component tools"));
  if (!absolute(hostSdk.root) || !hostSdk.pins || Object.keys(hostSdk.pins).sort().join(",") !== "descriptorSha256,filesSha256,overlaySha256,resultSha256" ||
      Object.values(hostSdk.pins).some(value => !digest.test(value || "")) || !absolute(hermes.root) || !localId.test(hermes.localBuildId || "") ||
      !digest.test(hermes.handoffSha256 || "") || Object.values(tools).some(file => !absolute(file))) {
    throw new Error("Local Windows host-component SDK, Hermes or tool identity is invalid");
  }

  const workspace = (dependencies.admitWorkspace || local.admitWorkspace)(input.workspace, repo),
    environment = {...process.env, ...(dependencies.environment || {}), MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: String(jobs)},
    selectedTools = (dependencies.resolveTools || local.localTools)(environment);
  if (workspace !== input.workspace || !isDeepStrictEqual(tools, selectedTools)) {
    throw new Error("Local Windows host-component tools or workspace differ from their pinned selection");
  }
  const catalog = (dependencies.cudartCatalog || windowsCudart.windowsCudartCatalog)(),
    paths = localWindowsHostComponentPaths(workspace, catalog.archive.fileName), existing = checkManagedRoot(paths, true, catalog.archive.fileName),
    sdkAdmission = (dependencies.readWindowsHostSdk || windowsSdkReader.windowsHostSdkInputs)(hostSdk.root, hostSdk.pins),
    hermesAdmission = (dependencies.readHermes || controllerHandoff.windowsLocalControllerHandoffInputs)(hermes.root,
      {kind: "hermes", localBuildId: hermes.localBuildId, handoffSha256: hermes.handoffSha256});
  if (sdkAdmission.root !== hostSdk.root || !isDeepStrictEqual(sdkAdmission.pins, hostSdk.pins) || hermesAdmission.kind !== "hermes" ||
      hermesAdmission.localBuildId !== hermes.localBuildId || hermesAdmission.handoffSha256 !== hermes.handoffSha256 || !hermesAdmission.intelRuntime) {
    throw new Error("Local Windows host-component support inputs failed their original public admissions");
  }

  const before = admitWorkspaceInputs(workspace, tools, environment, dependencies), windowsLlvmPaths = localWindowsLlvm.localWindowsLlvmSdkPaths(workspace),
    buildWindowsLlvm = dependencies.buildWindowsLlvm || localWindowsLlvm.buildLocalNativeWindowsLlvmSdk,
    makeLlvmInput = () => ({workspace, windowsHostSdk: hostSdk, hermes, tools}),
    llvmResult = validateWindowsLlvmResult(buildWindowsLlvm(makeLlvmInput(), {environment}), windowsLlvmPaths, workspace),
    composeResourcesConfig = dependencies.composeResourcesConfig || nativeCi.rocmResourcesConfig,
    resourcesConfig = composeResourcesConfig({source: before.rocmContext.paths.source, llvmSdk: before.linux.sdk.path,
      hipIncludeRoot: before.rocmState.hipIncludeRoot}, {cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake}, "win32");
  if (resourcesConfig?.target !== "win32") {throw new Error("Windows ROCm resource descriptor must use the public win32 target");}

  const common = {llvmOutput: llvmResult.output, comgrResult: path.join(paths.comgr, "result.json"),
      acppResult: path.join(paths.acpp, "result.json"), hiprtcResult: path.join(paths.hiprtc, "result.json")},
    readComponent = dependencies.readComponent || hostComponent.windowsHostComponentInputs,
    verifyProducers = dependencies.verifyProducers || hostComponent.verifyWindowsHostComponentProducers,
    matchComponent = dependencies.matchComponent || hostComponent.matchWindowsHostComponent,
    verifyDevice = dependencies.verifyDeviceResources || windowsResources.verifyWindowsDeviceResources,
    readCudart = dependencies.readCudart || windowsCudart.windowsCudartInputs,
    readAmdhip = dependencies.readAmdhip || windowsAmdhip.windowsAmdhipInputs,
    cudartOutput = paths.cudart, amdhipOutput = paths.amdhip;

  if (!existing) {
    fs.mkdirSync(paths.root, {mode: 0o700}); fs.mkdirSync(paths.archives, {mode: 0o700});
    writeJsonOnce(paths.resourcesConfig, resourcesConfig);
    (dependencies.buildResources || rocmResources.buildResources)(resourcesConfig, paths.resources, environment);

    const resourceReceipt = path.join(paths.resources, "result.json"), deviceInstall = path.join(paths.resources, "install");
    verifyDevice(resourceReceipt, deviceInstall);
    (dependencies.configureComgr || windowsResources.configureWindowsComgrSdk)({source: path.join(before.rocmContext.paths.source, "roots/rocmLlvm"),
      descriptorFile: llvmResult.descriptor, targetLlvm: llvmResult.output,
      nativeLlvm: before.linux.sdk.path, hostCompiler: tools.cc, deviceReceipt: resourceReceipt, deviceInstall,
      output: paths.comgr, jobs}, environment);
    (dependencies.buildComgr || windowsResources.buildWindowsComgrSdk)(paths.comgr, environment);

    const hiprtcSourcePrep = dependencies.prepareHiprtcSource || prepareLocalHiprtcSource,
      hiprtcSourceCheck = dependencies.verifyHiprtcSource || verifyLocalHiprtcSource,
      gitEnv = {...environment, GIT_ALLOW_PROTOCOL: "file", GIT_ATTR_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: "/dev/null",
        GIT_CONFIG_NOSYSTEM: "1", GIT_NO_LAZY_FETCH: "1", GIT_NO_REPLACE_OBJECTS: "1", GIT_OPTIONAL_LOCKS: "0", GIT_TERMINAL_PROMPT: "0"};
    hiprtcSourcePrep(before.rocmContext.paths.source + "/roots/clr", paths.hiprtcSource, tools.git, gitEnv, dependencies.executeGit || spawnSync);
    hiprtcSourceCheck(paths.hiprtcSource, tools.git, gitEnv, dependencies.executeGit || spawnSync);
    const hiprtcConfig = {source: paths.hiprtcSource, descriptorFile: llvmResult.descriptor,
      hipInclude: before.rocmState.hipIncludeRoot, comgrResult: path.join(paths.comgr, "result.json"),
      deviceReceipt: resourceReceipt, deviceInstall, llvmAr: path.join(before.linux.sdk.path, "bin/llvm-ar"), output: paths.hiprtc};
    writeJsonOnce(paths.hiprtcConfig, hiprtcConfig);
    (dependencies.buildHiprtc || windowsHiprtc.buildNativeWindowsHiprtcSdk)(hiprtcConfig, environment);

    (dependencies.downloadCudart || downloadWindowsCudart)(paths.archive, catalog, dependencies.executeDownload || spawnSync);
    if (dependencies.prepareCudart) {dependencies.prepareCudart(paths.archive, cudartOutput);} else {
      (dependencies.run || local.runCommand)(process.execPath, [path.join(repo, "scripts/prepare-native-cudart-windows.js"), paths.archive, cudartOutput],
        {label: "Pinned Windows CUDA Runtime preparation", env: environment, cwd: repo});
    }
    const cudart = readCudart(cudartOutput);
    if (cudart.cudartStatic !== path.join(cudartOutput, "runtime/cudart_static.lib") ||
        cudart.receipt?.source?.archivePath !== paths.archive || cudart.receipt?.source?.sha256 !== catalog.archive.sha256) {
      throw new Error("Local Windows CUDA Runtime output does not retain its original pinned archive provenance");
    }

    const prepareAmdhip = dependencies.prepareAmdhip || windowsAmdhip.prepareLocalAmdhipWindows, amdhipBuildId = randomUUID(),
      amdhip = await prepareAmdhip({sourceRoot: before.rocmContext.paths.source, llvmOutput: before.support.llvmSdk,
        llvmLib: path.join(before.linux.sdk.path, "bin/llvm-lib"), output: amdhipOutput, localBuildId: amdhipBuildId}, environment),
      amdhipInputs = readAmdhip(amdhipOutput), amdhipCatalog = (dependencies.amdhipCatalog || windowsAmdhip.windowsAmdhipCatalog)();
    if (amdhip?.localBuildId !== amdhipBuildId ||
        amdhipInputs.amdhip64 !== path.join(amdhipOutput, amdhipCatalog.library.fileName) ||
        amdhipInputs.receipt?.localBuildId !== amdhipBuildId) {
      throw new Error("Local AMDHIP output does not preserve the original local source/tool identity");
    }

    (dependencies.buildAcpp || windowsAcpp.buildWindowsAcppStaticHost)({source: before.acppPaths.preparedCheckout,
      descriptorFile: llvmResult.descriptor, descriptor: readBuildJson(llvmResult.descriptor), targetLlvm: llvmResult.output,
      nativeTools: path.join(before.linux.sdk.path, "bin"), cuda: before.dpcppState.cudaRoot, cudartStatic: cudart.cudartStatic,
      cudaDeviceLibs: path.join(before.dpcppState.cudaRoot, "nvvm/libdevice"), rocm: before.rocmState.hipHeadersRoot,
      amdhip64: amdhipInputs.amdhip64, rocmDeviceLibs: path.join(deviceInstall, "amdgcn/bitcode"), rocmReceipt: resourceReceipt,
      output: paths.acpp, jobs}, environment);
    (dependencies.exportComponent || hostComponent.exportWindowsHostComponent)({...common, output: paths.component}, environment);
  } else {
    verifyStableJson(paths.resourcesConfig, resourcesConfig, "Windows ROCm resource config");
    verifyStableJson(paths.hiprtcConfig, {source: paths.hiprtcSource, descriptorFile: llvmResult.descriptor,
      hipInclude: before.rocmState.hipIncludeRoot, comgrResult: path.join(paths.comgr, "result.json"),
      deviceReceipt: path.join(paths.resources, "result.json"), deviceInstall: path.join(paths.resources, "install"),
      llvmAr: path.join(before.linux.sdk.path, "bin/llvm-ar"), output: paths.hiprtc}, "Windows HIPRTC config");
    const archiveStat = regularFile(paths.archive, "Pinned Windows CUDA Runtime archive");
    if (archiveStat.size !== catalog.archive.bytes || fileHash(paths.archive) !== catalog.archive.sha256) {
      throw new Error("Local Windows CUDA archive changed after its original pin; preserve it");
    }
    const resource = verifyDevice(path.join(paths.resources, "result.json"), path.join(paths.resources, "install")),
      cudart = readCudart(cudartOutput), amdhipInputs = readAmdhip(amdhipOutput), amdhipCatalog = (dependencies.amdhipCatalog || windowsAmdhip.windowsAmdhipCatalog)();
    if (cudart.cudartStatic !== path.join(cudartOutput, "runtime/cudart_static.lib") ||
        cudart.receipt?.source?.archivePath !== paths.archive || cudart.receipt?.source?.sha256 !== catalog.archive.sha256 ||
        amdhipInputs.amdhip64 !== path.join(amdhipOutput, amdhipCatalog.library.fileName) ||
        !localId.test(amdhipInputs.receipt?.localBuildId || "")) {
      throw new Error("Local Windows CUDA/AMD runtime producer admission differs from the selected original inputs");
    }
    if (resource.sha256 !== fileHash(path.join(paths.resources, "result.json"))) {throw new Error("Windows ROCm resource receipt changed during admission");}
  }

  const after = admitWorkspaceInputs(workspace, tools, environment, dependencies),
    finalLlvm = validateWindowsLlvmResult(buildWindowsLlvm(makeLlvmInput(), {environment}), windowsLlvmPaths, workspace);
  if (!isDeepStrictEqual(before.linux, after.linux) || !isDeepStrictEqual(before.hipHeaders, after.hipHeaders) ||
      !isDeepStrictEqual(before.rocmAdmission.comgr, after.rocmAdmission.comgr) ||
      !isDeepStrictEqual(before.rocmAdmission.receipt, after.rocmAdmission.receipt) ||
      !isDeepStrictEqual(before.dpcpp, after.dpcpp) || !isDeepStrictEqual(before.acppSource, after.acppSource) ||
      !sameWindowsLlvm(llvmResult, finalLlvm)) {
    throw new Error("Local Windows host-component inputs changed during production; preserve the output");
  }
  const device = verifyDevice(path.join(paths.resources, "result.json"), path.join(paths.resources, "install")),
    original = verifyProducers(common, environment);
  if (original.llvm.admission.generationId !== llvmResult.admission.admission.generationId) {
    throw new Error("Windows host component producers differ from the locally admitted LLVM generation");
  }
  const matched = matchComponent(paths.component, original), component = readComponent(paths.component);
  if (!isDeepStrictEqual(matched, component)) {throw new Error("Windows host component changed during original producer admission");}
  checkManagedRoot(paths, false, catalog.archive.fileName);
  return {workspace, root: paths.root, componentRoot: paths.component, component, producers: original.provenance,
    windowsLlvm: {output: llvmResult.output, descriptor: llvmResult.descriptor, generationId: original.llvm.admission.generationId},
    resources: {root: paths.resources, receipt: path.join(paths.resources, "result.json"), receiptSha256: device.sha256},
    comgr: {root: paths.comgr, result: common.comgrResult}, hiprtc: {source: paths.hiprtcSource, root: paths.hiprtc, result: common.hiprtcResult},
    acpp: {root: paths.acpp, result: common.acppResult}, cudart: readCudart(cudartOutput), amdhip64: readAmdhip(amdhipOutput),
    jobs, buildOnly: true, compiled: true, executed: false, publicationQualified: false, reused: existing};
}

module.exports = {localWindowsHostComponentPaths, checkManagedRoot, prepareLocalHiprtcSource, verifyLocalHiprtcSource,
  downloadWindowsCudart, admitWorkspaceInputs, buildLocalNativeWindowsHostComponent, jobs};
