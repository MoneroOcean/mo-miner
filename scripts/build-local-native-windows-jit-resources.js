"use strict";

// Admit the local Linux JIT inputs, then stage the corresponding Windows GNU
// source-JIT resources without borrowing CI run identities or transport receipts.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk"), localJit = require("./build-local-native-jit-sdk"),
  nativeCi = require("./native-ci-linux-jit"), dpcppInputs = require("./native-dpcpp-cuda-inputs"),
  gnuHeaders = require("./prepare-native-gnu-headers"), jitSdk = require("./native-jit-sdk"),
  windowsResources = require("./native-windows-jit-resources"), resourceBuilder = require("./prepare-native-sycl-jit-resources");
const {fileHash, readBuildJson} = require("./build-source-inputs");

const repo = fs.realpathSync(path.resolve(__dirname, "..")), linuxTarget = "x86_64-unknown-linux-gnu",
  windowsTarget = "x86_64-w64-windows-gnu", oneapiCompiler = "/opt/intel/oneapi/compiler/2026.2/bin";

/** @param {string} file @param {string} label @param {number} [maximum] */
function regularFile(file, label, maximum = 16 << 20) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!path.isAbsolute(file) || path.resolve(file) !== file || !stat?.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 ||
      stat.size < 1 || stat.size > maximum || fs.realpathSync(file) !== file) {
    throw new Error(label + " is not a bounded canonical regular file: " + file);
  }
  return stat;
}

/** @param {string} root @param {string} label */
function realDirectory(root, label) {
  const stat = fs.lstatSync(root, {throwIfNoEntry: false});
  if (!path.isAbsolute(root) || path.resolve(root) !== root || !stat?.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(root) !== root) {
    throw new Error(label + " is not a canonical directory: " + root);
  }
  return root;
}

/** @param {string} workspace */
function localWindowsJitResourcePaths(workspace) {
  if (!path.isAbsolute(workspace) || path.resolve(workspace) !== workspace) {throw new Error("Local Windows JIT workspace must be canonical and absolute");}
  const root = path.join(workspace, "windows-jit-resources");
  return {root, gnu: path.join(root, "windows-gnu-headers"), emitter: path.join(root, "emitter"),
    profile: path.join(root, "resource-profile.json"), resources: path.join(root, "resources")};
}

/** Validate the exact source/LLVM paths returned by buildLocalNativeSdk and reuse the existing JIT context reader.
 * @param {unknown} value */
function admittedLocalSdkContext(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {throw new Error("Local Windows JIT needs the admitted buildLocalNativeSdk result");}
  const sdk = /** @type {{workspace?:unknown,compilerFetch?:unknown,compilerSources?:unknown,buildInputs?:unknown,llvmSdk?:unknown}} */ (value);
  if (typeof sdk.workspace !== "string") {throw new Error("Local Windows JIT needs the admitted SDK workspace path");}
  const workspace = local.admitWorkspace(sdk.workspace, repo), support = local.supportPaths(workspace);
  if (sdk.compilerFetch !== support.compilerFetch || sdk.compilerSources !== support.compilerSources ||
      sdk.buildInputs !== support.buildInputs || sdk.llvmSdk !== support.llvmSdk) {
    throw new Error("Local Windows JIT SDK result does not match the marked workspace paths");
  }
  const context = localJit.localJitSdkContext(localJit.localJitSdkContextFromWorkspace(workspace));
  return {workspace, support: context.support, inputs: context.inputs, resources: context.resources};
}

/** @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools */
function validateTools(tools) {
  if (!tools || typeof tools !== "object" || Array.isArray(tools)) {throw new Error("Local Windows JIT tools are required");}
  local.localJitToolsReceipt(tools);
  if (tools.cc !== path.join(oneapiCompiler, "icx") || tools.cxx !== path.join(oneapiCompiler, "icpx")) {
    throw new Error("Local Windows JIT resource admission requires the selected Intel 2026.2 icx/icpx tools");
  }
  for (const [name, file] of Object.entries(tools)) {
    if (fs.realpathSync(file) !== file || !fs.statSync(file).isFile()) {throw new Error("Local JIT tool is not canonical: " + name);}
    fs.accessSync(file, fs.constants.X_OK);
  }
}

/** Re-admit the original local DPC++ receipt and Linux resource tree; the digest is only a before/after drift guard,
 * not an external CI artifact pin. @param {ReturnType<typeof admittedLocalSdkContext>} context
 * @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools @param {NodeJS.ProcessEnv} env
 * @param {typeof dpcppInputs.auditDpcppCudaInputs} auditDpcpp @param {typeof nativeCi.auditLinuxJitResourceTree} auditLinux */
function auditInputs(context, tools, env, auditDpcpp, auditLinux) {
  const state = local.localDpcppCudaState(context.support), manifestStat = regularFile(state.manifest, "Local DPC++/CUDA manifest", 8 << 20),
    manifestSha256 = fileHash(state.manifest), record = readBuildJson(state.manifest);
  if (manifestStat.size < 1 || record?.source?.origin !== "workspace-prepared-receipts" || record.source.workspaceRoot !== context.workspace) {
    throw new Error("Local Windows JIT requires the original prepared-receipt DPC++ source provenance");
  }
  const dpcpp = auditDpcpp(state, record);
  const linux = auditLinux(local.localLinuxJitResourceState(context.support), tools, env);
  if (fileHash(state.manifest) !== manifestSha256 || linux?.resources?.target !== linuxTarget ||
      linux.resources.directory !== path.join(context.resources.root, "resources") || !/^[a-f0-9]{64}$/.test(linux.resources.manifestHash || "") ||
      !/^\d{8}$/.test(linux.sourceDate || "")) {
    throw new Error("Local Windows JIT source, DPC++ inputs or Linux resource admission changed");
  }
  const baseRoot = path.join(context.resources.root, "base", "base"), baseManifest = path.join(context.resources.root, "base", "base-manifest.json");
  realDirectory(baseRoot, "Admitted local Linux JIT base"); regularFile(baseManifest, "Admitted local Linux JIT base manifest", 8 << 20);
  return {dpcpp, dpcppManifestSha256: manifestSha256, linux, base: {baseRoot, baseManifest, baseManifestSha256: fileHash(baseManifest)}};
}

/** @param {ReturnType<typeof localWindowsJitResourcePaths>} paths @param {boolean} allowMissing */
function checkManagedRoot(paths, allowMissing) {
  const stat = fs.lstatSync(paths.root, {throwIfNoEntry: false});
  if (!stat) {
    if (!allowMissing) {throw new Error("Local Windows JIT managed output is missing");}
    if (fs.realpathSync(path.dirname(paths.root)) !== path.dirname(paths.root)) {throw new Error("Local Windows JIT workspace parent is not canonical");}
    return false;
  }
  realDirectory(paths.root, "Local Windows JIT managed output");
  const expected = ["emitter", "resources", "resource-profile.json", "windows-gnu-headers"].sort(), actual = fs.readdirSync(paths.root).sort();
  if (!isDeepStrictEqual(actual, expected)) {throw new Error("Local Windows JIT output is incomplete or contains unknown state; preserve it for inspection");}
  realDirectory(paths.gnu, "Local Windows GNU headers"); realDirectory(paths.emitter, "Local Windows JIT emitter");
  realDirectory(paths.resources, "Local Windows JIT resources"); regularFile(paths.profile, "Local Windows JIT resource profile", 1 << 20);
  return true;
}

/** @param {string} root */
function windowsGnuReceiptPins(root) {
  const result = path.join(root, "result.json"), copyrights = path.join(root, "copyrights.json"), targetHeaders = path.join(root, "target-headers.json");
  for (const file of [result, copyrights, targetHeaders]) {regularFile(file, "Local Windows GNU producer receipt", 8 << 20);}
  return {result: fileHash(result), copyrights: fileHash(copyrights), targetHeaders: fileHash(targetHeaders)};
}

/** Produce or re-admit the stable Windows-target resource handoff used by the local WSL coordinator.
 * No CI run identity is created; original local source, GNU, emitter and resource receipts remain authoritative.
 * @param {unknown} localSdk Result of buildLocalNativeSdk
 * @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools Selected by local.localTools
 * @param {{environment?:NodeJS.ProcessEnv,verifyScope?:()=>unknown,validateTools?:(tools:{cmake:string,ninja:string,cc:string,cxx:string})=>unknown,
 * auditDpcpp?:typeof dpcppInputs.auditDpcppCudaInputs,auditLinux?:typeof nativeCi.auditLinuxJitResourceTree,
 * buildGnu?:typeof gnuHeaders.buildGnuHeaderHandoff,auditGnu?:typeof gnuHeaders.auditGnuHeaders,
 * buildEmitter?:typeof jitSdk.buildJitEmitter,admitEmitter?:typeof windowsResources.admitEmitter,
 * prepareResources?:typeof resourceBuilder.prepare,verifyResources?:typeof resourceBuilder.verifyJitResources}} [dependencies] */
function buildLocalWindowsJitResources(localSdk, tools, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local Windows JIT resource production requires Linux x64 (WSL2)");}
  const verifyScope = dependencies.verifyScope || local.verifyCurrentMemoryScope;
  verifyScope();
  const context = admittedLocalSdkContext(localSdk), paths = localWindowsJitResourcePaths(context.workspace),
    env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2", OMP_NUM_THREADS: "2"},
    auditDpcpp = dependencies.auditDpcpp || dpcppInputs.auditDpcppCudaInputs,
    auditLinux = dependencies.auditLinux || nativeCi.auditLinuxJitResourceTree,
    buildGnu = dependencies.buildGnu || gnuHeaders.buildGnuHeaderHandoff,
    auditGnu = dependencies.auditGnu || gnuHeaders.auditGnuHeaders,
    buildEmitter = dependencies.buildEmitter || jitSdk.buildJitEmitter,
    admitEmitter = dependencies.admitEmitter || windowsResources.admitEmitter,
    prepareResources = dependencies.prepareResources || resourceBuilder.prepare,
    verifyResources = dependencies.verifyResources || resourceBuilder.verifyJitResources;
  (dependencies.validateTools || validateTools)(tools);
  const reused = checkManagedRoot(paths, true), before = auditInputs(context, tools, env, auditDpcpp, auditLinux);
  /** @type {ReturnType<typeof gnuHeaders.auditGnuHeaders>} */ let initialGnu;
  /** @type {ReturnType<typeof windowsGnuReceiptPins>} */ let initialGnuPins;
  if (!reused) {
    fs.mkdirSync(paths.root, {mode: 0o700});
    buildGnu({root: paths.gnu}, windowsTarget);
    initialGnu = auditGnu({root: paths.gnu, target: windowsTarget}, windowsTarget);
    initialGnuPins = windowsGnuReceiptPins(paths.gnu);
    const targetHeaders = readBuildJson(path.join(paths.gnu, "target-headers.json"));
    if (!Array.isArray(targetHeaders) || !targetHeaders.length || targetHeaders.length > 16384 ||
        !targetHeaders.every(file => typeof file === "string" && path.isAbsolute(file) && path.resolve(file) === file)) {
      throw new Error("Local Windows GNU producer did not emit its bounded absolute target-header list");
    }
    buildEmitter(context.inputs.dpcppRoot, paths.emitter, env, targetHeaders);
    const emitterFile = path.join(paths.emitter, "result.json"); regularFile(emitterFile, "Local Windows JIT emitter receipt");
    const emitter = admitEmitter(paths.emitter, context.inputs.dpcppRoot, paths.gnu, initialGnu.headers, fileHash(emitterFile)),
      profile = windowsResources.profileFromAdmissions(before.base, {root: paths.gnu, notices: initialGnu.notices}, emitter);
    fs.writeFileSync(paths.profile, JSON.stringify(profile, null, 2) + "\n", {flag: "wx", mode: 0o600});
    prepareResources(profile, paths.resources);
  } else {
    initialGnu = auditGnu({root: paths.gnu, target: windowsTarget}, windowsTarget);
    initialGnuPins = windowsGnuReceiptPins(paths.gnu);
  }

  checkManagedRoot(paths, false);
  const gnuAdmission = auditGnu({root: paths.gnu, target: windowsTarget}, windowsTarget),
    gnuAdmissionPins = windowsGnuReceiptPins(paths.gnu),
    emitterFile = path.join(paths.emitter, "result.json"); regularFile(emitterFile, "Local Windows JIT emitter receipt");
  const emitter = admitEmitter(paths.emitter, context.inputs.dpcppRoot,
      paths.gnu, gnuAdmission.headers, fileHash(emitterFile)),
    profile = windowsResources.profileFromAdmissions(before.base, {root: paths.gnu, notices: gnuAdmission.notices}, emitter),
    profileFile = readBuildJson(paths.profile), profileSha256 = fileHash(paths.profile);
  if (!isDeepStrictEqual(profileFile, profile)) {throw new Error("Local Windows JIT profile differs from its admitted producer inputs");}
  const resourceAdmission = verifyResources(paths.resources, windowsTarget);
  if (resourceAdmission?.target !== windowsTarget || resourceAdmission.directory !== paths.resources ||
      !/^[a-f0-9]{64}$/.test(resourceAdmission.manifestHash || "")) {
    throw new Error("Local Windows JIT resource payload failed its maintained target admission");
  }
  const after = auditInputs(context, tools, env, auditDpcpp, auditLinux), checkedGnu = auditGnu({root: paths.gnu, target: windowsTarget}, windowsTarget),
    checkedGnuPins = windowsGnuReceiptPins(paths.gnu),
    checkedEmitter = admitEmitter(paths.emitter, context.inputs.dpcppRoot, paths.gnu, checkedGnu.headers, fileHash(emitterFile)),
    checkedResources = verifyResources(paths.resources, windowsTarget);
  if (before.dpcppManifestSha256 !== after.dpcppManifestSha256 || before.base.baseManifestSha256 !== after.base.baseManifestSha256 ||
      !isDeepStrictEqual(before.dpcpp, after.dpcpp) || !isDeepStrictEqual(before.linux, after.linux) ||
      !isDeepStrictEqual(initialGnu, checkedGnu) || !isDeepStrictEqual(initialGnuPins, gnuAdmissionPins) ||
      !isDeepStrictEqual(initialGnuPins, checkedGnuPins) ||
      emitter.receiptSha256 !== checkedEmitter.receiptSha256 || profileSha256 !== fileHash(paths.profile) ||
      resourceAdmission.manifestHash !== checkedResources.manifestHash || checkedResources?.target !== windowsTarget || checkedResources.directory !== paths.resources) {
    throw new Error("Local Windows JIT inputs or outputs changed during producer admission");
  }
  return {workspace: context.workspace, root: paths.root, target: windowsTarget, gnu: {root: paths.gnu,
    receiptFile: path.join(paths.gnu, "result.json"), receiptSha256: fileHash(path.join(paths.gnu, "result.json")),
    copyrightsFile: path.join(paths.gnu, "copyrights.json"), copyrightsSha256: fileHash(path.join(paths.gnu, "copyrights.json")),
    targetHeadersFile: path.join(paths.gnu, "target-headers.json"), targetHeadersSha256: checkedGnuPins.targetHeaders, admission: checkedGnu},
  emitter: {root: paths.emitter, receiptFile: emitter.receiptFile, receiptSha256: checkedEmitter.receiptSha256,
    fixture: checkedEmitter.fixture, compiler: checkedEmitter.compiler},
  profile: {path: paths.profile, sha256: profileSha256, record: profile},
  resources: {root: paths.resources, manifestFile: path.join(paths.resources, "manifest.json"), manifestSha256: resourceAdmission.manifestHash},
  linux: {root: context.resources.root, manifestSha256: before.linux.resources.manifestHash},
  dpcpp: {root: context.inputs.dpcppRoot, manifestFile: path.join(context.support.dpcppCudaRoot, "inputs.json"),
    manifestSha256: before.dpcppManifestSha256, sourceOrigin: before.dpcpp.source},
  buildOnly: true, executed: false, publicationQualified: false, reused};
}

module.exports = {localWindowsJitResourcePaths, admittedLocalSdkContext, buildLocalWindowsJitResources};
