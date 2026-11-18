"use strict";

// Compose the maintained Linux SDK producers into the source-composed release descriptor.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk"), jit = require("./build-local-native-jit-sdk"),
  rocm = require("./build-local-native-rocm-sdk"), cuda = require("./build-local-native-cuda-images"),
  acpp = require("./build-local-native-acpp-sdk"), gnu = require("./prepare-native-gnu-headers"),
  {linuxNativeSdkDescriptorConfig, linuxNativeSdkLldPath} = require("./native-linux-sdk-config"),
  {nativeSdkConfig, nativeSdkLinkInputs} = require("./build-native-release"),
  {checkNativeNoticeReview, dependencyPins} = require("./check-native-notices");

const repo = fs.realpathSync(path.resolve(__dirname, "..")), target = "x86_64-unknown-linux-gnu";

/** Resolve small, read-only host prerequisites without creating the workspace or acquiring SDK inputs.
 * Check only the fixed ROCm Python executable here; full host-tool, Python-module, and static-archive
 * admission is deferred until LLVM has been produced inside the outer entry's verified memory scope.
 * @param {{environment?:NodeJS.ProcessEnv,tools?:ReturnType<typeof local.localTools>,resolveTools?:typeof local.localTools,
 * supportTools?:ReturnType<typeof local.localSupportTools>,resolveSupportTools?:typeof local.localSupportTools,
 * installedRoots?:{dpcpp:string,cuda:string},installedGnuPackages?:typeof gnu.installedPackages,
 * pythonFilesystem?:Pick<typeof fs,"statSync"|"accessSync">,
 * nodeVersion?:string,resolveLinker?:typeof linuxNativeSdkLldPath,
 * linker?:string,checkNotices?:typeof checkNativeNoticeReview}} [dependencies] */
function preflightLocalLinuxSdk(dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local native Linux SDK requires Linux x64");}
  const nodeVersion = dependencies.nodeVersion || process.version, expectedNode = dependencyPins().buildNode.version;
  if (nodeVersion !== expectedNode) {throw new Error(`Local native Linux SDK requires pinned Node.js ${expectedNode}; current runtime is ${nodeVersion}`);}
  const environment = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"},
    tools = dependencies.tools || (dependencies.resolveTools || local.localTools)(environment),
    supportTools = dependencies.supportTools || (dependencies.resolveSupportTools || local.localSupportTools)(),
    installedRoots = local.localDpcppCudaRoots(dependencies.installedRoots);

  // These small queries fail before any source archive, workspace, or SDK build is started.
  (dependencies.installedGnuPackages || gnu.installedPackages)();
  checkPinnedRocmPythonExecutable(dependencies.pythonFilesystem);
  checkSysrootBuildTools(dependencies.pythonFilesystem);
  const linker = dependencies.linker || (dependencies.resolveLinker || linuxNativeSdkLldPath)(tools.cxx),
    noticeReview = (dependencies.checkNotices || checkNativeNoticeReview)();
  if (!noticeReview?.notice || !/^[a-f0-9]{64}$/.test(noticeReview.notice.sha256 || "")) {
    throw new Error("Reviewed native notice input is incomplete");
  }
  return {environment, tools, supportTools, installedRoots, linker, noticeReview};
}

/** Fail fast if the ROCm host builder's fixed Python executable is absent or non-executable.
 * Importing its modules and resolving the full tool/archive cohort remains part of scoped admission.
 * @param {Pick<typeof fs,"statSync"|"accessSync">} [fileSystem] */
function checkPinnedRocmPythonExecutable(fileSystem = fs) {
  const file = "/usr/bin/python3.12";
  let stat;
  try {stat = fileSystem.statSync(file);} catch (error) {
    throw new Error("Pinned ROCm host Python 3.12 is missing or not executable: " + file, {cause: error});
  }
  if (!stat.isFile()) {throw new Error("Pinned ROCm host Python 3.12 is missing or not executable: " + file);}
  try {fileSystem.accessSync(file, fs.constants.X_OK);} catch (error) {
    throw new Error("Pinned ROCm host Python 3.12 is missing or not executable: " + file, {cause: error});
  }
}

/** The glibc sysroot is extracted with dpkg-deb and libdrm is built with meson; fail before any acquisition.
 * @param {Pick<typeof fs,"statSync"|"accessSync">} [fileSystem] */
function checkSysrootBuildTools(fileSystem = fs) {
  for (const file of ["/usr/bin/dpkg-deb", "/usr/bin/meson"]) {
    try {
      if (!fileSystem.statSync(file).isFile()) {throw new Error("not a file");}
      fileSystem.accessSync(file, fs.constants.X_OK);
    } catch (error) {throw new Error("Native Linux SDK prerequisite is missing or not executable: " + file, {cause: error});}
  }
}

/** Preserve an admitted descriptor; never replace an unknown or changed one.
 * @param {string} file @param {unknown} config */
function writeOrVerifyDescriptor(file, config) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (stat) {
    if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== file ||
        !isDeepStrictEqual(JSON.parse(fs.readFileSync(file, "utf8")), config)) {
      throw new Error("Local Linux SDK descriptor exists with different or aliased content; preserve it for diagnosis");
    }
    return;
  }
  fs.writeFileSync(file, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
}

/** Build the common local source/tool prerequisites for Linux or Windows assembly.
 * Only Linux assembly needs its own static runtime/source-JIT SDK; Windows builds
 * use the same Linux generators but produce the Windows-target runtime separately.
 * Whole-command entrypoints provide the verified aggregate MOM_BUILD_JOBS-sized no-swap scope; standalone producers enforce their own.
 * @param {{workspace?:string,linuxJit?:boolean}} [options]
 * @param {{environment?:NodeJS.ProcessEnv,tools?:ReturnType<typeof local.localTools>,resolveTools?:typeof local.localTools,
 * supportTools?:ReturnType<typeof local.localSupportTools>,resolveSupportTools?:typeof local.localSupportTools,
 * installedRoots?:{dpcpp:string,cuda:string},nodeVersion?:string,run?:typeof local.runCommand,preflight?:typeof preflightLocalLinuxSdk,
 * installedGnuPackages?:typeof gnu.installedPackages,pythonFilesystem?:Pick<typeof fs,"statSync"|"accessSync">,
 * builders?:{source?:typeof local.buildLocalNativeSdk,support?:typeof local.buildLocalNativeSupportSdk,
 * compilerInputs?:typeof local.buildLocalCompilerInputs,gnuHeaders?:typeof local.buildLocalGnuHeaders,
 * jitResources?:typeof local.buildLocalLinuxJitResources,jitSdk?:typeof jit.buildLocalLinuxJitSdk,
 * rocmSdk?:typeof rocm.buildLocalLinuxRocmSdk,cudaImages?:typeof cuda.buildLocalCudaImages,
 * acppSdk?:typeof acpp.buildLocalLinuxAcppSdk},resolveLinker?:typeof linuxNativeSdkLldPath,linker?:string,
 * checkNotices?:typeof checkNativeNoticeReview,admitConfig?:typeof nativeSdkConfig,admitLinkInputs?:typeof nativeSdkLinkInputs,
 * preflightRocmHost?:(context:ReturnType<typeof rocm.localRocmSdkContext>,tools:{cc:string,cxx:string,cmake:string},
 * environment:NodeJS.ProcessEnv)=>unknown,defaultWorkspace?:typeof local.defaultWorkspace,
 * ensureSysroot?:typeof local.ensureLocalLinuxSysroot}} [dependencies] */
function buildLocalNativePrerequisites(options = {}, dependencies = {}) {
  if (options.linuxJit !== undefined && typeof options.linuxJit !== "boolean") {throw new Error("Local Linux JIT selection must be boolean");}
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local native Linux SDK requires Linux x64");}
  const ready = (dependencies.preflight || preflightLocalLinuxSdk)(dependencies),
    workspace = options.workspace || (dependencies.defaultWorkspace || local.defaultWorkspace)(),
    root = local.admitWorkspace(workspace, repo), paths = local.supportPaths(root), builders = dependencies.builders || {},
    run = dependencies.run,
    // Every producer below compiles and links against the pinned glibc sysroot selected here.
    environment = (dependencies.ensureSysroot || local.ensureLocalLinuxSysroot)(root, ready.environment),
    common = {tools: ready.tools, environment, installedRoots: ready.installedRoots,
      ...(run ? {run} : {}), ...(dependencies.resolveTools ? {resolveTools: dependencies.resolveTools} : {})},
    source = (builders.source || local.buildLocalNativeSdk)({workspace: root}, common);
  if (source.compilerFetch !== paths.compilerFetch || source.compilerSources !== paths.compilerSources ||
      source.buildInputs !== paths.buildInputs || source.llvmSdk !== paths.llvmSdk) {
    throw new Error("Source LLVM stage returned paths outside the admitted workspace");
  }

  const rocmInput = rocm.localRocmSdkContextFromWorkspace(root), rocmContext = rocm.localRocmSdkContext(rocmInput),
    rocmTools = {cc: ready.tools.cc, cxx: ready.tools.cxx, cmake: ready.tools.cmake};
  // This host-runtime audit resolves Python/modules and pinned static archives; keep it inside
  // the whole-graph scope established by the outer entry rather than the light pre-workspace check.
  const preflightRocmHost = dependencies.preflightRocmHost ||
    ((context, tools, environment) => rocm.localRocmHostInputs(context, tools, environment));
  preflightRocmHost(rocmContext, rocmTools, environment);

  const compilerInputs = (builders.compilerInputs || local.buildLocalCompilerInputs)(source, {...common}),
    gnuHeaders = (builders.gnuHeaders || local.buildLocalGnuHeaders)(source, {...common});
  if (compilerInputs.dpcppRoot !== path.join(paths.dpcppCudaRoot, "dpcpp") ||
      compilerInputs.cudaRoot !== path.join(paths.dpcppCudaRoot, "cuda") || compilerInputs.manifest !== path.join(paths.dpcppCudaRoot, "inputs.json") ||
      gnuHeaders.root !== paths.gnuHeadersRoot || gnuHeaders.target !== target) {
    throw new Error("Local compiler or GNU inputs were not admitted at their stable workspace paths");
  }

  const support = (builders.support || local.buildLocalNativeSupportSdk)(source, {...common, supportTools: ready.supportTools});
  if (support.hermesSdk !== paths.hermesSdk || support.ioSdk !== paths.ioSdk || support.opensslSdk !== paths.opensslSdk ||
      support.zstdSdk !== paths.zstdSdk || support.openmpSdk !== paths.openmpSdk) {
    throw new Error("Static support SDK stage returned paths outside the admitted workspace");
  }

  const jitResources = (builders.jitResources || local.buildLocalLinuxJitResources)(source, {...common,
    prepareInputs: () => compilerInputs, prepareHeaders: () => gnuHeaders});
  if (jitResources.root !== paths.linuxJitResources || jitResources.target !== target) {
    throw new Error("Linux JIT resources were not admitted at their stable workspace path");
  }
  const jitContext = {source, inputs: compilerInputs, resources: jitResources},
    jitSdk = options.linuxJit ? (builders.jitSdk || jit.buildLocalLinuxJitSdk)(jitContext, common) : undefined,
    jitPaths = jit.localJitSdkPaths(root);
  if (jitSdk && jitSdk.staticComponent !== jitPaths.staticComponent) {throw new Error("Linux JIT SDK did not return the admitted static DPC++ component");}

  const rocmSdk = (builders.rocmSdk || rocm.buildLocalLinuxRocmSdk)(rocmInput, common),
    expectedRocm = rocm.rocmSdkResult(rocmContext, rocmSdk.reused);
  if (!isDeepStrictEqual(rocmSdk, expectedRocm)) {throw new Error("ROCm SDK stage result differs from its public path schema");}

  const cudaImages = (builders.cudaImages || cuda.buildLocalCudaImages)(root, common),
    imagePaths = cuda.localCudaImagePaths(root);
  if (cudaImages.handoffRoot !== imagePaths.handoffRoot || cudaImages.image !== path.join(imagePaths.handoffRoot, "cuda-images.inc")) {
    throw new Error("CUDA image stage did not return the admitted image handoff");
  }

  const acppContext = acpp.localAcppSdkContextFromWorkspace(root, rocmSdk),
    acppSdk = (builders.acppSdk || acpp.buildLocalLinuxAcppSdk)(acppContext, common),
    acppPaths = acpp.localAcppSdkPaths(root);
  if (acppSdk.sdk !== acppPaths.sdk || acppSdk.sources !== acppPaths.preparedCheckout) {
    throw new Error("ACPP SDK stage did not return its admitted source and SDK paths");
  }

  return {workspace: root, paths, tools: ready.tools, ready, common, source, compilerInputs, gnuHeaders, support,
    jitResources, jitSdk, rocmSdk, cudaImages, acppSdk};
}

/** Compose the complete Linux release SDK using the shared local prerequisite sequence.
 * @param {{workspace?:string}} [options]
 * @param {Parameters<typeof buildLocalNativePrerequisites>[1]} [dependencies] */
function buildLocalLinuxSdk(options = {}, dependencies = {}) {
  const {workspace: root, paths, ready, source, compilerInputs, gnuHeaders, support, jitSdk, rocmSdk, cudaImages, acppSdk} =
      buildLocalNativePrerequisites({...options, linuxJit: true}, dependencies),
    jitPaths = jit.localJitSdkPaths(root), imagePaths = cuda.localCudaImagePaths(root), acppPaths = acpp.localAcppSdkPaths(root);
  if (!jitSdk) {throw new Error("Linux release assembly requires its static source-JIT SDK");}
  const latestNotices = (dependencies.checkNotices || checkNativeNoticeReview)(), notices = {
      path: path.join(repo, "THIRD-PARTY-NOTICES.txt"), sha256: latestNotices.notice.sha256},
    llvmInstalled = path.join(source.llvmSdk, "sdk"), state = {
      llvmSdk: llvmInstalled, images: {handoffRoot: imagePaths.handoffRoot},
      acppSdk: {sdkRoot: acppPaths.sdk}, acpp: {preparedCheckout: acppPaths.preparedCheckout},
      dpcpp: {dpcppRoot: compilerInputs.dpcppRoot, cudaRoot: compilerInputs.cudaRoot},
      jit: {staticComponent: jitPaths.staticComponent}, hermes: {sdkRoot: paths.hermesSdk},
      rocm: {sdkRoot: rocmSdk.comgrSdk, runtimeOutput: rocmSdk.runtimeOutput},
      openmp: {sdkRoot: paths.openmpSdk}, zstd: {sdkRoot: paths.zstdSdk},
      opensslSdk: paths.opensslSdk, ioSdk: paths.ioSdk},
    config = {...linuxNativeSdkDescriptorConfig(state, {cc: ready.tools.cc, cxx: ready.tools.cxx}, ready.linker), notices},
    admitted = (dependencies.admitConfig || nativeSdkConfig)(config, repo),
    linked = (dependencies.admitLinkInputs || nativeSdkLinkInputs)(admitted.sdk);
  if (!Array.isArray(admitted.keys) || admitted.keys.length !== Object.keys(config.gpu).length || !linked || typeof linked !== "object") {
    throw new Error("Public Linux native SDK configuration or link-input admission is incomplete");
  }
  const configFile = path.join(root, "native-linux-sdk.json");
  writeOrVerifyDescriptor(configFile, config);
  return {workspace: root, configFile, tools: ready.tools, sdkPaths: {
    llvm: llvmInstalled, dpcpp: compilerInputs.dpcppRoot, cuda: compilerInputs.cudaRoot, gnuHeaders: gnuHeaders.root,
    cudaImages: cudaImages.image, jit: jitSdk.compilerSdk, staticDpcpp: jitSdk.staticDpcppSdk,
    staticDpcppComponent: jitSdk.staticComponent, acpp: acppSdk.sdk,
    hermes: support.hermesSdk, rocmComgr: rocmSdk.comgrSdk, rocmRuntime: rocmSdk.rocmSdk,
    io: support.ioSdk, openssl: support.opensslSdk, zstd: support.zstdSdk, openmp: support.openmpSdk},
  notices, profiles: admitted.keys.slice().sort(), buildOnly: true, executed: false, publicationQualified: false};
}

/** @param {string[]} argv */
function main(argv) {
  let workspace;
  for (let index = 0; index < argv.length; index++) {
    if (argv[index] === "--workspace" && argv[index + 1]) {
      workspace = argv[++index];
    } else if (argv[index] === "--help" || argv[index] === "-h") {
      console.log("Usage: node scripts/build-local-native-linux-sdk.js [--workspace /absolute/path]\n" +
        "Builds or re-admits the actual Linux source SDK graph and writes native-linux-sdk.json.\n" +
        "MOM_BUILD_JOBS (1..16, default 2) selects build parallelism and sizes the memory scope.\n" +
        "Does not build Core/GPU factories or create a release archive.");
      return;
    } else {
      throw new Error("Unknown or incomplete local Linux SDK option: " + argv[index]);
    }
  }
  const result = buildLocalLinuxSdk(workspace ? {workspace} : {});
  console.log(`[native-sdk] Linux SDK descriptor admitted at ${result.configFile}; Core, GPU factories and release packaging were not run.`);
}

if (require.main === module) {
  try {main(process.argv.slice(2));} catch (error) {console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1;}
}

module.exports = {preflightLocalLinuxSdk, buildLocalNativePrerequisites, buildLocalLinuxSdk, writeOrVerifyDescriptor, main};
