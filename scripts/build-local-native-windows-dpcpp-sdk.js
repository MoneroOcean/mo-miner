"use strict";

// Local WSL coordinator for the maintained static Windows DPC++ SDK and its relocatable component.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk"), gpu = require("./build-local-native-windows-gpu-config"),
  windowsLlvm = require("./build-local-native-windows-llvm-sdk"), windowsResources = require("./build-local-native-windows-jit-resources"),
  windowsJit = require("./build-local-native-windows-jit-sdk"),
  handoffs = require("./native-windows-controller-handoff"), hermesInputs = require("./native-release-sdk-inputs"),
  appConfig = require("./native-windows-application-config"), sdkProducer = require("./build-native-windows-sdk"),
  adapterProducer = require("./prepare-native-windows-adapters"), component = require("./native-windows-dpcpp-component");
const {hasExactBuildFields, requireBuildFields, verifyBuildDirectoryEntries, canonicalBuildDirectory, canonicalBuildFile, verifyStableBuildJson, fileHash, readBuildJson} = require("./build-source-inputs");
const repo = fs.realpathSync(path.resolve(__dirname, "..")),
  localBuildIdPattern = /^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/,
  allowedImports = ["advapi32.dll", "cfgmgr32.dll", "dbghelp.dll", "kernel32.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll",
    "setupapi.dll", "shell32.dll", "shlwapi.dll", "version.dll"];

/** @param {unknown} value @returns {value is Record<string, any>} */
function record(value) {return value !== null && typeof value === "object" && !Array.isArray(value);}

/** @param {string} workspace */
function localWindowsDpcppSdkPaths(workspace) {
  if (typeof workspace !== "string" || !path.isAbsolute(workspace) || path.resolve(workspace) !== workspace || /[\r\n\0]/.test(workspace)) {
    throw new Error("Local Windows DPC++ workspace must be canonical and absolute");
  }
  const root = path.join(workspace, "windows-dpcpp-sdk");
  return {root, descriptor: path.join(root, "descriptor.json"), sdk: path.join(root, "sdk"),
    adapters: path.join(root, "adapters"), component: path.join(root, "component")};
}

/** @param {ReturnType<typeof localWindowsDpcppSdkPaths>} paths @param {boolean} allowMissing */
function checkManagedRoot(paths, allowMissing) {
  if (!canonicalBuildDirectory(paths.root, "Local Windows DPC++ managed output", allowMissing)) {return false;}
  verifyBuildDirectoryEntries(paths.root, ["adapters", "component", "descriptor.json", "sdk"], "Local Windows DPC++ managed output");
  canonicalBuildFile(paths.descriptor, "Local Windows DPC++ descriptor");
  for (const directory of [paths.sdk, paths.adapters, paths.component]) {canonicalBuildDirectory(directory, "Local Windows DPC++ " + path.basename(directory));}
  verifyBuildDirectoryEntries(paths.component, ["component.json", "lib", "objects"], "Local Windows DPC++ component");
  for (const directory of [path.join(paths.component, "lib"), path.join(paths.component, "objects")]) {
    canonicalBuildDirectory(directory, "Local Windows DPC++ component " + path.basename(directory));
  }
  return true;
}

/** @param {any} value */
function stableJitResult(value) {
  const fields = ["workspace", "root", "descriptor", "descriptorSha256", "config", "configSha256", "toolchain", "compilerSdk", "provenance",
    "compilerResultSha256", "isolation", "isolationHandoffSha256", "jobs", "buildOnly", "compiled", "executed", "publicationQualified"];
  requireBuildFields(value, [...fields, "reused"], "Local Windows JIT SDK result");
  return Object.fromEntries(fields.map(key => [key, value[key]]));
}

/** Extract and re-admit the local IO/Hermes export returned by exportLocalWindowsSupport. @param {any} input
 * @param {typeof handoffs.windowsLocalControllerHandoffInputs} readHandoff */
function admitControllerSupport(input, readHandoff) {
  const support = requireBuildFields(input, ["localBuildId", "io", "hermes"], "Local Windows controller support"),
    id = support["localBuildId"], ioInput = support["io"], hermesInput = support["hermes"];
  if (!localBuildIdPattern.test(id || "")) {throw new Error("Local Windows controller support needs one UUID build identity");}
  /** @param {any} value @param {"io"|"hermes"} kind */
  const admit = (value, kind) => {
    if (!record(value) || typeof value["root"] !== "string" || typeof value["handoffSha256"] !== "string" ||
        value["admitted"]?.["localBuildId"] !== id || value["admitted"]?.["kind"] !== kind) {
      throw new Error("Local Windows " + kind + " handoff differs from its export identity");
    }
    return readHandoff(value["root"], {kind, localBuildId: id, handoffSha256: value["handoffSha256"]});
  };
  const io = admit(ioInput, "io"), hermes = admit(hermesInput, "hermes");
  if (io.localBuildId !== id || hermes.localBuildId !== id) {throw new Error("Local Windows support handoffs do not share one identity");}
  return {id, io, hermes, hermesInput: {root: hermesInput["root"], localBuildId: id, handoffSha256: hermesInput["handoffSha256"]}};
}

/** @param {unknown} input @param {{verifyScope?:()=>unknown,resolveTools?:typeof local.localTools,environment?:NodeJS.ProcessEnv,
 * admitGpuInputs?:typeof gpu.admitLocalWindowsDpcppGpuInputs,buildJitSdk?:typeof windowsJit.buildLocalNativeWindowsJitSdk,
 * readWindowsHandoff?:typeof handoffs.windowsLocalControllerHandoffInputs,readLinuxHermes?:typeof hermesInputs.nativeHermesInputs,
 * composeDescriptor?:typeof appConfig.composeWindowsApplicationDescriptor,buildSdk?:typeof sdkProducer.buildWindowsStaticSdk,
 * prepareAdapters?:typeof adapterProducer.prepareWindowsAdapters,exportComponent?:typeof component.exportWindowsDpcppComponent,
 * producerInputs?:typeof component.windowsDpcppProducerInputs,matchComponent?:typeof component.matchWindowsDpcppComponent,
 * readJson?:typeof readBuildJson}} [dependencies]
 * @returns {Promise<any>} */
async function buildLocalNativeWindowsDpcppSdk(input, dependencies = {}) {
  const verifyScope = dependencies.verifyScope || local.verifyCurrentMemoryScope;
  verifyScope();
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local Windows DPC++ SDK production requires Linux x64 (WSL2)");}
  const keys = ["controllerSupport", "tools", "windowsHostSdk", "windowsJitResources", "windowsJitSdk", "windowsLlvm", "workspace"];
  if (!record(input) || !hasExactBuildFields(input, keys) || typeof input["workspace"] !== "string") {
    throw new Error("Local Windows DPC++ needs the marked workspace, admitted Windows prerequisites and local controller handoffs");
  }
  const args = /** @type {{controllerSupport:any,tools:any,windowsHostSdk:any,windowsJitResources:any,windowsJitSdk:any,windowsLlvm:any,workspace:string}} */ (input),
    workspace = local.admitWorkspace(args.workspace, repo), paths = localWindowsDpcppSdkPaths(workspace), supportPaths = local.supportPaths(workspace),
    jobs = local.localBuildJobs({...process.env, ...(dependencies.environment || {})}),
    env = {...process.env, ...(dependencies.environment || {}), MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: String(jobs), OMP_NUM_THREADS: String(jobs)},
    tools = /** @type {ReturnType<typeof local.localTools>} */ (requireBuildFields(args.tools, ["cc", "cmake", "cxx", "git", "ninja", "systemdRun"], "Selected local tools")),
    resolveTools = dependencies.resolveTools || local.localTools, selectedTools = resolveTools(env);
  if (!isDeepStrictEqual(tools, selectedTools)) {throw new Error("Local Windows DPC++ tools differ from the current pinned selection");}

  if (!record(args.windowsHostSdk)) {throw new Error("Local Windows host-SDK input is malformed");}
  const hostSdk = {root: args.windowsHostSdk["root"], pins: args.windowsHostSdk["pins"]};
  if (hostSdk.root !== path.join(workspace, "windows-host-sdk", "sdk") ||
      args.windowsHostSdk["workspace"] !== undefined && args.windowsHostSdk["workspace"] !== workspace) {
    throw new Error("Local Windows DPC++ host SDK belongs to another workspace");
  }
  const windowsSourceFile = path.join(workspace, "windows-sources", "sources.json"), windowsSourceSha256 = fileHash(windowsSourceFile),
    handoffReader = dependencies.readWindowsHandoff || handoffs.windowsLocalControllerHandoffInputs,
    gpuInput = {workspace, windowsSdkRoot: hostSdk.root, windowsSdkPins: hostSdk.pins, windowsSourceSha256, tools};

  // The shared JIT coordinator re-admits the existing Windows LLVM, resources and isolated source JIT.
  // Require those prerequisites first so this final coordinator never starts or repairs their builds.
  const llvmPaths = windowsLlvm.localWindowsLlvmSdkPaths(workspace), resourcePaths = windowsResources.localWindowsJitResourcePaths(workspace),
    jitPaths = windowsJit.localWindowsJitSdkPaths(workspace);
  for (const [directory, label] of /** @type {Array<[string,string]>} */ ([[llvmPaths.root, "Windows LLVM SDK"],
    [resourcePaths.root, "Windows JIT resources"], [jitPaths.root, "Windows JIT SDK"]])) {canonicalBuildDirectory(directory, "Local " + label);}

  const admitGpu = dependencies.admitGpuInputs || gpu.admitLocalWindowsDpcppGpuInputs,
    buildJit = dependencies.buildJitSdk || windowsJit.buildLocalNativeWindowsJitSdk,
    readLinuxHermes = dependencies.readLinuxHermes || hermesInputs.nativeHermesInputs,
    compose = dependencies.composeDescriptor || appConfig.composeWindowsApplicationDescriptor,
    readJson = dependencies.readJson || readBuildJson;
  /** Re-read all original workspace and transport receipts around production or warm admission. */
  const admitInputs = async () => {
    const currentController = admitControllerSupport(args.controllerSupport, handoffReader), currentSourceSha256 = fileHash(windowsSourceFile),
      admittedGpu = admitGpu({...gpuInput, windowsSourceSha256: currentSourceSha256}, {
        verifyScope, resolveTools: () => tools}),
      admittedJit = await buildJit({workspace, windowsHostSdk: hostSdk, hermes: currentController.hermesInput,
        windowsLlvm: args.windowsLlvm, windowsJitResources: args.windowsJitResources, tools},
      {verifyScope, resolveTools: () => tools, environment: env}),
      suppliedJit = stableJitResult(args.windowsJitSdk), checkedJit = stableJitResult(admittedJit);
    if (!isDeepStrictEqual(suppliedJit, checkedJit)) {throw new Error("Local Windows JIT handoff differs from its current producer admission");}
    const hermesAot = readLinuxHermes(path.join(supportPaths.hermesSdk, "result.json")), llvmReadobj = path.join(admittedGpu.tools.toolRoot, "llvm-readobj");
    if (!admittedGpu.tools.files?.[llvmReadobj]) {throw new Error("Local Windows DPC++ descriptor needs admitted Linux llvm-readobj");}
    const descriptor = compose({llvmDescriptor: readJson(llvmPaths.descriptor), io: currentController.io, hermes: currentController.hermes,
      hermesAot, tools: {shermaot: hermesAot.artifacts["compiler"], llvmReadobj}, allowedImports});
    return {sourceSha256: currentSourceSha256, gpu: admittedGpu, jit: checkedJit, controller: currentController, hermesAot, descriptor};
  };

  const existing = checkManagedRoot(paths, true), before = await admitInputs(), descriptorSha256 = existing ?
      verifyStableBuildJson(paths.descriptor, before.descriptor, "Local Windows DPC++ descriptor") : undefined,
    sdkReceipt = path.join(paths.sdk, "result.json"), adapterReceipt = path.join(paths.adapters, "result.json"),
    componentReader = dependencies.producerInputs || component.windowsDpcppProducerInputs,
    match = dependencies.matchComponent || component.matchWindowsDpcppComponent;
  let exportedComponent;
  if (!existing) {
    const parent = path.dirname(paths.root);
    if (fs.realpathSync(parent) !== parent) {throw new Error("Local Windows DPC++ workspace parent is not canonical");}
    fs.mkdirSync(paths.root, {mode: 0o700});
    fs.writeFileSync(paths.descriptor, JSON.stringify(before.descriptor, null, 2) + "\n", {flag: "wx", mode: 0o444});
    const jitReceipt = path.join(before.jit["isolation"], "result.json");
    canonicalBuildFile(jitReceipt, "Original isolated Windows source-JIT receipt");
    // The original producer invokes CMake by name and probes DPC++; do not inherit
    // Windows/WSL PATH entries or another installation's compiler runtime libraries.
    const buildEnv = {...env, MOM_AOT_DPCPP_JIT_RECEIPT: jitReceipt,
      PATH: [...new Set([path.dirname(tools.cmake), path.dirname(tools.ninja), path.dirname(tools.cc),
        path.join(before.gpu.dpcppRoot, "bin"), before.gpu.tools.toolRoot, "/usr/bin", "/bin"])].join(":"),
      LD_LIBRARY_PATH: [path.join(before.gpu.dpcppRoot, "lib"), path.resolve(path.dirname(tools.cc), "../lib"),
        "/usr/lib/x86_64-linux-gnu"].join(":")};
    (dependencies.buildSdk || sdkProducer.buildWindowsStaticSdk)(path.join(workspace, "windows-sources", "sources.json"),
      paths.descriptor, before.gpu.dpcppRoot, before.gpu.cudaRoot, paths.sdk, buildEnv);
    (dependencies.prepareAdapters || adapterProducer.prepareWindowsAdapters)(sdkReceipt, paths.descriptor, before.gpu.cudaRoot,
      paths.adapters, buildEnv);
    exportedComponent = (dependencies.exportComponent || component.exportWindowsDpcppComponent)(adapterReceipt, before.gpu.cudaRoot,
      paths.component, buildEnv);
  }

  checkManagedRoot(paths, false);
  canonicalBuildFile(sdkReceipt, "Original Windows DPC++ SDK receipt");
  canonicalBuildFile(adapterReceipt, "Original Windows adapter receipt");
  const checkedDescriptorSha256 = verifyStableBuildJson(paths.descriptor, before.descriptor, "Local Windows DPC++ descriptor"),
    sdkReceiptSha256 = fileHash(sdkReceipt), original = componentReader(adapterReceipt, before.gpu.cudaRoot, env),
    checkedComponent = match(paths.component, original), after = await admitInputs();
  checkManagedRoot(paths, false);
  for (const [file, hash] of Object.entries(checkedComponent.files)) {
    if (fileHash(file) !== hash) {throw new Error("Local Windows DPC++ component changed during final input admission");}
  }
  if ((descriptorSha256 && descriptorSha256 !== checkedDescriptorSha256) || !isDeepStrictEqual(before, after) ||
      exportedComponent && (!isDeepStrictEqual(exportedComponent.link, checkedComponent.link) ||
        !isDeepStrictEqual(exportedComponent.files, checkedComponent.files)) || fileHash(sdkReceipt) !== sdkReceiptSha256 ||
      fileHash(paths.descriptor) !== checkedDescriptorSha256 || original.producer.path !== adapterReceipt ||
      fileHash(adapterReceipt) !== original.producer.sha256) {
    throw new Error("Local Windows DPC++ inputs or component changed during production; preserve the workspace output");
  }
  return {workspace, root: paths.root, descriptor: paths.descriptor, descriptorSha256: checkedDescriptorSha256,
    sdk: {root: paths.sdk, receipt: sdkReceipt, receiptSha256: fileHash(sdkReceipt)},
    adapters: {root: paths.adapters, receipt: adapterReceipt, receiptSha256: fileHash(adapterReceipt)},
    component: {root: paths.component, receipt: path.join(paths.component, "component.json"), link: checkedComponent.link, files: checkedComponent.files},
    jobs, buildOnly: true, compiled: true, executed: false, publicationQualified: false, reused: existing};
}

module.exports = {localWindowsDpcppSdkPaths, checkManagedRoot, admitControllerSupport, buildLocalNativeWindowsDpcppSdk};
