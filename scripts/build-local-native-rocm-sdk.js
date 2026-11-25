"use strict";

// Continue an admitted local Linux LLVM slice through the public CPU-only ROCm producers.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"),
  {createHash} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk"), ciRocm = require("./native-ci-linux-rocm"),
  ciSources = require("./native-ci-linux-sources"),
  rocmProducer = require("./build-native-rocm-host-runtime");
const {canonicalBuildDirectory, hasExactBuildFields, hasBuildDirectoryEntries, fileHash, readBuildJson} = require("./build-source-inputs");
const {sdkSourceFiles} = require("./prepare-native-sdk-sources");
const {verifyLinuxLlvmAdmission} = require("./build-native-llvm-sdk");
const {gitSourceComponents} = require("./fetch-native-sdk-sources");
const {isStaticArchive} = require("./native-audit");
const {verifyPreparedRoots} = require("./build-local-native-jit-sdk");

const repo = fs.realpathSync(path.resolve(__dirname, "..")), oneapiCompilerRoot = "/opt/intel/oneapi/compiler", sourceTarget = "rocm",
  hostPrerequisiteMessage = "ROCm host SDK needs the pinned Python 3.12.3 / CppHeaderParser 2.7.4 / PLY 3.11 environment, meson for the pinned libdrm source build, and the GL/EGL/X11 development headers; install/validate these explicitly before retrying (this local command never invokes sudo or APT).";

/** Stable workspace-owned ROCm roots; no CI run IDs, /tmp transports, or generation directories.
 * @param {string} workspace */
function localRocmSdkPaths(workspace) {
  if (typeof workspace !== "string" || !path.isAbsolute(workspace) || path.resolve(workspace) !== workspace) {
    throw new Error("Local ROCm SDK workspace must be a canonical absolute path");
  }
  const source = path.join(workspace, "rocm-sources"), hipHeadersRoot = path.join(workspace, "rocm-hip-headers"),
    resourcesRoot = path.join(workspace, "rocm-resources"), sdkRoot = path.join(workspace, "rocm-comgr-sdk"),
    hostSdkRoot = path.join(workspace, "rocm-host-sdk"), pythonSourceRoot = path.join(source, "python-source");
  return {workspace, source, rootsFile: path.join(source, "roots.json"), fetchReceipt: path.join(source, "fetch.json"),
    pythonSourceRoot, pythonArchive: path.join(pythonSourceRoot, "CppHeaderParser-2.7.4.tar.gz"),
    pythonSourceReceipt: path.join(pythonSourceRoot, "receipt.json"), hipHeadersRoot, resourcesRoot, sdkRoot,
    resourcesConfig: path.join(workspace, "rocm-resources-config.json"),
    hipSourceRoot: path.join(hipHeadersRoot, "source"), hipCommonRoot: path.join(hipHeadersRoot, "source/hip-common"),
    clrHipamdRoot: path.join(hipHeadersRoot, "source/clr-hipamd"), hipBuildRoot: path.join(hipHeadersRoot, "build"),
    hipArchiveRoot: path.join(hipHeadersRoot, "archives"), hipIncludeRoot: path.join(hipHeadersRoot, "include"),
    hipInstallPrefix: path.join(hipHeadersRoot, "build/install-prefix"), hipPatchRoot: path.join(hipHeadersRoot, "patches"),
    hipPatchFile: path.join(hipHeadersRoot, "patches/rocm-hip-headers-only.patch"),
    hipPatchCheckLog: path.join(hipHeadersRoot, "patches/check.json"), hipPatchApplyLog: path.join(hipHeadersRoot, "patches/apply.json"),
    hipConfigureLog: path.join(hipHeadersRoot, "cmake-configure.log"), hipConfigurePlan: path.join(hipHeadersRoot, "configure-plan.json"),
    hipCommonTar: path.join(hipHeadersRoot, "archives/hip-common.tar"), clrHipamdTar: path.join(hipHeadersRoot, "archives/clr-hipamd.tar"),
    hostSdkRoot, runtimeOutput: path.join(hostSdkRoot, "runtime"), hostConfig: path.join(hostSdkRoot, "rocm-host.json"),
    libdrmSource: path.join(workspace, "libdrm-fetch"), libdrmSdk: path.join(workspace, "libdrm-sdk"),
    comgrAdmission: path.join(workspace, "rocm-comgr-admission.json"), hostAdmission: path.join(workspace, "rocm-host-admission.json")};
}

/** Construct the single accepted input context from a completed source/LLVM workspace.
 * @param {string} workspace */
function localRocmSdkContextFromWorkspace(workspace) {
  const paths = local.supportPaths(workspace);
  return {source: {workspace, compilerFetch: paths.compilerFetch, compilerSources: paths.compilerSources,
    buildInputs: paths.buildInputs, llvmSdk: paths.llvmSdk}};
}

/** Reject fabricated, redirected, or cross-workspace prior-stage results. Also accepts this
 * function's own result (callers pass it on to the auditors), re-validating its source paths.
 * @param {unknown} value */
function localRocmSdkContext(value) {
  if (!hasExactBuildFields(value, ["source"]) && !hasExactBuildFields(value, ["paths", "source", "workspace"])) {
    throw new Error("Local ROCm SDK needs only an admitted source-LLVM stage result");
  }
  const row = /** @type {{source?:{workspace?:unknown,compilerFetch?:unknown,compilerSources?:unknown,buildInputs?:unknown,llvmSdk?:unknown}}} */ (value),
    source = row.source;
  if (!source || typeof source.workspace !== "string") {throw new Error("Local ROCm SDK source result is incomplete");}
  canonicalBuildDirectory(source.workspace, "Admitted native SDK workspace");
  const workspace = local.admitWorkspace(source.workspace, repo), expected = localRocmSdkContextFromWorkspace(workspace),
    paths = localRocmSdkPaths(workspace);
  if (source.compilerFetch !== expected.source.compilerFetch || source.compilerSources !== expected.source.compilerSources ||
      source.buildInputs !== expected.source.buildInputs || source.llvmSdk !== expected.source.llvmSdk) {
    throw new Error("Local ROCm SDK inputs do not match the exact admitted workspace paths");
  }
  return {workspace, source: expected.source, paths};
}

/** @param {string} file @param {string} label @param {boolean} [singleLink] */
function regularFile(file, label, singleLink = false) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || singleLink && stat.nlink !== 1 || fs.realpathSync(file) !== file) {
    throw new Error(label + " is missing or not a canonical regular file: " + file);
  }
}

/** Re-admit the exact LLVM receipt and prepared compiler source root map. @param {ReturnType<typeof localRocmSdkContext>} context
 * @param {{verifyLlvm?:typeof verifyLinuxLlvmAdmission}} [dependencies] */
function verifyRocmBuildInputs(context, dependencies = {}) {
  const paths = local.supportPaths(context.workspace);
  verifyPreparedRoots(path.join(paths.compilerSources, "sources.json"), paths.compilerSources, "compiler");
  (dependencies.verifyLlvm || verifyLinuxLlvmAdmission)(paths.llvmSdk);
}

/** Match the package inventory to the original HIP/CLR exports and current pinned source trees.
 * The shared core has no transport-free HIP-package auditor; this validates only that boundary. */
/** @param {ReturnType<typeof localRocmBuildState>} state @param {{cc:string,cxx:string,cmake:string}} tools
 * @param {string} [git] */
function auditLocalHipHeaders(state, tools, git = "/usr/bin/git") {
  canonicalBuildDirectory(state.hipHeadersRoot, "Local ROCm HIP header root");
  canonicalBuildDirectory(state.hipIncludeRoot, "Local ROCm HIP include root");
  if (!hasBuildDirectoryEntries(state.hipHeadersRoot,
    ["archives", "build", "cmake-configure.log", "configure-plan.json", "header-package.json", "include", "patches", "source"])) {
    throw new Error("Local ROCm HIP header root has missing or unexpected producer outputs");
  }
  for (const directory of [state.hipCommonRoot, state.clrHipamdRoot]) {canonicalBuildDirectory(directory, "Local ROCm HIP source export");}
  const packageFile = path.join(state.hipHeadersRoot, "header-package.json");
  regularFile(packageFile, "ROCm HIP header package receipt", true);
  const record = readBuildJson(packageFile), pins = gitSourceComponents(sourceTarget), hipRevision = pins["hip"]?.revision,
    clrRevision = pins["clr"]?.revision, common = sdkSourceFiles(state.hipCommonRoot), clr = sdkSourceFiles(state.clrHipamdRoot),
    include = sdkSourceFiles(state.hipIncludeRoot),
    keys = ["version", "pins", "sourceExports", "patch", "configure", "hiprtcCmake", "generatedHeader", "files", "requiredHiprtcHeaders", "buildOnly", "publicationQualified"];
  /** @param {Record<string,string>} value */
  const mapHash = value => createHash("sha256").update(JSON.stringify(value)).digest("hex");
  if (!hasExactBuildFields(record, keys) || record.version !== 1 || record.buildOnly !== true ||
      record.publicationQualified !== false || !isDeepStrictEqual(record.pins, {hip: hipRevision, clr: clrRevision}) ||
      !record.sourceExports || record.sourceExports.hipCommon?.root !== "source/hip-common" ||
      record.sourceExports.hipCommon.revision !== hipRevision || record.sourceExports.hipCommon.fileCount !== Object.keys(common).length ||
      record.sourceExports.hipCommon.filesSha256 !== mapHash(common) || record.sourceExports.clrHipamd?.root !== "source/clr-hipamd" ||
      record.sourceExports.clrHipamd.revision !== clrRevision || record.sourceExports.clrHipamd.treeish !== clrRevision + ":hipamd" ||
      record.sourceExports.clrHipamd.afterPatch?.fileCount !== Object.keys(clr).length ||
      record.sourceExports.clrHipamd.afterPatch.filesSha256 !== mapHash(clr) || !isDeepStrictEqual(record.files, include)) {
    throw new Error("Local ROCm HIP header package does not bind the pinned source exports and include inventory");
  }
  const packagePaths = Object.keys(include).map(name => path.join(state.hipIncludeRoot, ...name.split("/")));
  for (const file of packagePaths) {regularFile(file, "ROCm HIP header package member", true);}
  const patch = record.patch, patchSource = path.join(repo, "scripts/patches/rocm-hip-headers-only.patch");
  if (!patch || !hasExactBuildFields(patch, ["applyLog", "checkLog", "originalSha256", "patchedSha256", "path", "sha256", "stripComponents", "target"]) ||
      patch.path !== "patches/rocm-hip-headers-only.patch" || patch.sha256 !== fileHash(patchSource) || patch.stripComponents !== 2 ||
      patch.target !== "source/clr-hipamd/CMakeLists.txt" || !/^[a-f0-9]{64}$/.test(patch.originalSha256 || "") ||
      patch.patchedSha256 !== clr["CMakeLists.txt"] || patch.checkLog !== "patches/check.json" || patch.applyLog !== "patches/apply.json") {
    throw new Error("Local ROCm HIP header package patch differs from the current reviewed patch");
  }
  regularFile(state.hipPatchFile, "Saved ROCm HIP header patch", true);
  if (fileHash(state.hipPatchFile) !== patch.sha256) {throw new Error("Saved ROCm HIP header patch changed");}
  canonicalBuildDirectory(state.hipPatchRoot, "ROCm HIP patch evidence root");
  /** @type {Array<[string,boolean]>} */
  const patchLogs = [[state.hipPatchCheckLog, true], [state.hipPatchApplyLog, false]];
  for (const [file, check] of patchLogs) {
    regularFile(file, "ROCm HIP patch log", true);
    const evidence = readBuildJson(file), command = [git, "-C", state.clrHipamdRoot, "apply", "-p2", ...(check ? ["--check"] : []), "--", state.hipPatchFile];
    if (!hasExactBuildFields(evidence, ["command", "error", "signal", "status", "stderr", "stdout"]) || evidence.status !== 0 ||
        evidence.signal !== null || evidence.error !== null || typeof evidence.stdout !== "string" || typeof evidence.stderr !== "string" ||
        !isDeepStrictEqual(evidence.command, command)) {
      throw new Error("ROCm HIP patch evidence did not pass its exact public command");
    }
  }
  const expectedConfigure = ciRocm.hipHeaderConfigureCommand(state, tools), configure = record.configure;
  if (!configure || configure.status !== 0 || configure.configureOnly !== true || configure.log !== "cmake-configure.log" ||
      !isDeepStrictEqual(configure.command, expectedConfigure) ||
      !isDeepStrictEqual(configure.environment, {GIT_DIR: path.join(state.source, "roots/clr")}) ||
      configure.logSha256 !== fileHash(state.hipConfigureLog)) {
    throw new Error("Local ROCm HIP configure receipt does not match its public configure-only recipe");
  }
  regularFile(state.hipConfigurePlan, "ROCm HIP configure plan", true); regularFile(state.hipConfigureLog, "ROCm HIP configure log", true);
  if (!isDeepStrictEqual(readBuildJson(state.hipConfigurePlan), {version: 1, command: expectedConfigure,
    environment: {GIT_DIR: path.join(state.source, "roots/clr")}, configureOnly: true, buildInvoked: false, installInvoked: false})) {
    throw new Error("Local ROCm HIP configure plan is not configure-only");
  }
  const generated = record.generatedHeader, hiprtc = record.hiprtcCmake;
  if (generated?.path !== "build/include/hip/hip_version.h" || hiprtc?.path !== "source/clr-hipamd/src/hiprtc/CMakeLists.txt" ||
      !/^[a-f0-9]{64}$/.test(generated.sha256 || "") || !/^[a-f0-9]{64}$/.test(hiprtc.sha256 || "")) {
    throw new Error("Local ROCm HIP package omits its generated version or HIPRTC source identity");
  }
  for (const [relative, expected] of [[generated.path, generated.sha256], [hiprtc.path, hiprtc.sha256]]) {
    const file = path.join(state.hipHeadersRoot, relative);
    regularFile(file, "ROCm HIP package source member", true);
    if (fileHash(file) !== expected) {throw new Error("ROCm HIP package source member changed: " + relative);}
  }
  for (const [name, sha256] of Object.entries(include)) {
    const sourceName = "include/" + name, expected = name === "hip/hip_version.h" ? generated.sha256 :
      name.startsWith("hip/amd_detail/") ? clr[sourceName] : common[sourceName];
    if (sha256 !== expected) {throw new Error("Local ROCm HIP package header differs from its pinned source: " + name);}
  }
  if (!Array.isArray(record.requiredHiprtcHeaders) || record.requiredHiprtcHeaders.length < 16 || record.requiredHiprtcHeaders.length > 128 ||
      record.requiredHiprtcHeaders.some((/** @type {unknown} */ name) => typeof name !== "string" || !Object.hasOwn(include, name))) {
    throw new Error("Local ROCm HIP package has an invalid HIPRTC header list");
  }
  /** @param {string} root @param {string} treeish @param {string} expected @param {string} label */
  const verifyTree = (root, treeish, expected, label) => {
    const result = spawnSync(git, ["-C", root, "ls-tree", "-rz", treeish],
      {encoding: "utf8", env: {...process.env, GIT_ALLOW_PROTOCOL: "file", GIT_NO_LAZY_FETCH: "1"}, timeout: 60000, maxBuffer: 32 << 20});
    if (result.error || result.signal || result.status !== 0 || createHash("sha256").update(result.stdout || "").digest("hex") !== expected) {
      throw new Error("Local ROCm HIP " + label + " no longer matches its pinned Git tree", {cause: result.error});
    }
  };
  verifyTree(path.join(state.source, "roots/hip"), hipRevision || "", hipRevision ? record.sourceExports.hipCommon.treeSha256 : "", "common headers");
  verifyTree(path.join(state.source, "roots/clr"), clrRevision + ":hipamd", record.sourceExports.clrHipamd.treeSha256, "CLR headers");
  const original = spawnSync(git, ["-C", path.join(state.source, "roots/clr"), "show", clrRevision + ":hipamd/CMakeLists.txt"],
    {timeout: 60000, maxBuffer: 8 << 20});
  if (original.error || original.signal || original.status !== 0 || !Buffer.isBuffer(original.stdout) ||
      createHash("sha256").update(original.stdout).digest("hex") !== patch.originalSha256) {
    throw new Error("Local ROCm HIP patch receipt does not bind its original pinned CLR bytes", {cause: original.error});
  }
  return {package: packageFile, packageSha256: fileHash(packageFile), include: state.hipIncludeRoot,
    files: Object.fromEntries(packagePaths.map(file => [file, fileHash(file)]))};
}

/** Keep the local admission small; underlying public auditors recompute full producer inputs.
 * @param {ReturnType<typeof localRocmSdkContext>} context @param {{cc:string,cxx:string,cmake:string}} tools
 * @param {{hip:ReturnType<typeof auditLocalHipHeaders>,comgr:unknown}} admitted */
function localComgrAdmission(context, tools, admitted) {
  const state = localRocmBuildState(context), compilerReceipt = path.join(context.source.compilerSources, "sources.json"),
    llvmReceipt = path.join(context.source.llvmSdk, "result.json"), resourcesReceipt = path.join(state.resourcesRoot, "result.json"),
    sdkReceipt = path.join(state.sdkRoot, "result.json");
  return {version: 1, target: "linux-rocm-comgr", roots: {source: state.source, llvm: state.llvmSdk, hip: state.hipHeadersRoot,
    resources: state.resourcesRoot, sdk: state.sdkRoot}, inputs: {
    fetchReceipt: {path: state.fetchReceipt, sha256: fileHash(state.fetchReceipt)}, rootsReceipt: {path: state.rootsFile, sha256: fileHash(state.rootsFile)},
    parserReceipt: {path: state.pythonSourceReceipt, sha256: fileHash(state.pythonSourceReceipt)},
    compilerSources: {path: compilerReceipt, sha256: fileHash(compilerReceipt)}, llvmReceipt: {path: llvmReceipt, sha256: fileHash(llvmReceipt)},
    tools: Object.fromEntries(Object.entries(tools).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}]))},
  hip: admitted.hip, resourcesReceipt: {path: resourcesReceipt, sha256: fileHash(resourcesReceipt)},
  sdkReceipt: {path: sdkReceipt, sha256: fileHash(sdkReceipt)}, buildOnly: true, executed: false, publicationQualified: false};
}

/** @param {ReturnType<typeof localRocmSdkContext>} context */
function localRocmBuildState(context) {
  const p = context.paths, support = local.supportPaths(context.workspace);
  return {...localRocmSourceState(p), llvmSdk: path.join(support.llvmSdk, "sdk"), hipHeadersRoot: p.hipHeadersRoot,
    hipSourceRoot: p.hipSourceRoot, hipCommonRoot: p.hipCommonRoot, clrHipamdRoot: p.clrHipamdRoot,
    hipBuildRoot: p.hipBuildRoot, hipArchiveRoot: p.hipArchiveRoot, hipInstallPrefix: p.hipInstallPrefix,
    hipIncludeRoot: p.hipIncludeRoot, hipPatchRoot: p.hipPatchRoot, hipPatchFile: p.hipPatchFile,
    hipPatchCheckLog: p.hipPatchCheckLog, hipPatchApplyLog: p.hipPatchApplyLog,
    hipConfigureLog: p.hipConfigureLog, hipConfigurePlan: p.hipConfigurePlan, hipCommonTar: p.hipCommonTar,
    clrHipamdTar: p.clrHipamdTar, resourcesRoot: p.resourcesRoot, resourcesConfig: p.resourcesConfig, sdkRoot: p.sdkRoot};
}

/** @param {ReturnType<typeof localRocmSdkPaths>} paths */
function localRocmSourceState(paths) {
  return {source: paths.source, rootsFile: paths.rootsFile, fetchReceipt: paths.fetchReceipt,
    pythonSourceRoot: paths.pythonSourceRoot, pythonArchive: paths.pythonArchive, pythonSourceReceipt: paths.pythonSourceReceipt};
}

/** Admit canonical local receipt files before the public verifier parses them.
 * @param {ReturnType<typeof localRocmSdkPaths>} paths @param {NodeJS.ProcessEnv} env @param {string} git
 * @param {typeof ciSources.verifyRocmSources} [verify] */
function verifyLocalRocmSources(paths, env, git, verify = ciSources.verifyRocmSources) {
  canonicalBuildDirectory(paths.source, "Local pinned ROCm source cohort");
  regularFile(paths.fetchReceipt, "Pinned ROCm source fetch receipt", true);
  regularFile(paths.rootsFile, "Pinned ROCm source roots receipt", true);
  canonicalBuildDirectory(paths.pythonSourceRoot, "Pinned ROCm parser source directory");
  regularFile(paths.pythonArchive, "Pinned ROCm parser source archive", true);
  regularFile(paths.pythonSourceReceipt, "Pinned ROCm parser source receipt", true);
  return verify({env}, localRocmSourceState(paths), git);
}

/** Validate every input through the existing public source/LLVM and ROCm auditors. @param {ReturnType<typeof localRocmSdkContext>} context
 * @param {{cc:string,cxx:string,cmake:string}} tools
 * @param {{verifyInputs?:typeof verifyRocmBuildInputs,verifySources?:typeof ciSources.verifyRocmSources,
 *   auditComgr?:typeof ciRocm.auditRocmComgrSdk}} [dependencies] */
function auditLocalRocmComgrInputs(context, tools, dependencies = {}) {
  const selected = localRocmSdkContext(context), state = localRocmBuildState(selected), git = "/usr/bin/git",
    env = {...process.env, MOM_AOT_BUILD_ONLY: "1"};
  (dependencies.verifyInputs || verifyRocmBuildInputs)(selected);
  verifyLocalRocmSources(selected.paths, env, git, dependencies.verifySources || ciSources.verifyRocmSources);
  const hip = auditLocalHipHeaders(state, tools, git), comgr = (dependencies.auditComgr || ciRocm.auditRocmComgrSdk)(state, tools),
    expected = localComgrAdmission(selected, tools, {hip, comgr});
  return {context: selected, state, hip, comgr, receipt: expected};
}

/** Build the exact public host-runtime configuration without provisioning packages or Python modules.
 * @param {ReturnType<typeof localRocmSdkContext>} context */
function localRocmHostState(context) {
  const state = localRocmBuildState(context), p = context.paths;
  return {...state, runtimeOutput: p.runtimeOutput, hostSdkRoot: p.hostSdkRoot, configFile: p.hostConfig,
    libdrmSource: p.libdrmSource, libdrmSdk: p.libdrmSdk};
}

/** This check is intentionally non-mutating; the public builder performs exact hash/version admission.
 * @param {Pick<ReturnType<typeof ciRocm.rocmHostBuildTools>, "python3">} tools @param {NodeJS.ProcessEnv} env */
function checkRocmHostPrerequisites(tools, env) {
  const result = spawnSync(tools.python3, ["-c", "import CppHeaderParser, ply, ply.lex"],
    {encoding: "utf8", env, timeout: 30000, maxBuffer: 1 << 20});
  if (result.error || result.signal || result.status !== 0) {throw new Error(hostPrerequisiteMessage, {cause: result.error});}
}

/** Resolve only existing host-runtime prerequisites; this never provisions packages or Python.
 * @param {ReturnType<typeof localRocmSdkContext>} context @param {{cc:string,cxx:string,cmake:string}} tools
 * @param {NodeJS.ProcessEnv} env
 * @param {{hostTools?:(state:ReturnType<typeof localRocmHostState>)=>ReturnType<typeof ciRocm.rocmHostBuildTools>,
 *   hostConfig?:(state:ReturnType<typeof localRocmHostState>,tools:ReturnType<typeof ciRocm.rocmHostBuildTools>,env:NodeJS.ProcessEnv)=>ReturnType<typeof ciRocm.rocmHostBuildConfig>,
 *   verifyHostPrerequisites?:typeof checkRocmHostPrerequisites}} [dependencies] */
function localRocmHostInputs(context, tools, env, dependencies = {}) {
  const state = localRocmHostState(context);
  let hostTools, config;
  try {
    hostTools = (dependencies.hostTools || ciRocm.rocmHostBuildTools)(state);
    // The static libdrm SDK is a small pinned source build; produce or admit it before the config names it.
    if (!dependencies.hostConfig) {ciRocm.buildLibdrmSdk(state, env);}
    config = (dependencies.hostConfig || ciRocm.rocmHostBuildConfig)(state, hostTools, env);
  } catch (error) {throw new Error(hostPrerequisiteMessage, {cause: error});}
  if (tools.cmake !== hostTools.cmake || fs.realpathSync(hostTools.cc) !== hostTools.cc || fs.realpathSync(hostTools.cxx) !== hostTools.cxx ||
      fs.realpathSync(tools.cc) !== hostTools.cc || fs.realpathSync(tools.cxx) !== hostTools.cxx) {
    throw new Error("ROCm host-runtime compiler identities differ from the COMGR build tools");
  }
  (dependencies.verifyHostPrerequisites || checkRocmHostPrerequisites)(hostTools, env);
  try {
    for (const file of Object.values(config.dependencies)) {
      regularFile(file, "Pinned ROCm host static dependency");
      if (!isStaticArchive(file)) {throw new Error("ROCm host dependency is not a static archive: " + file);}
    }
  } catch (error) {throw new Error(hostPrerequisiteMessage, {cause: error});}
  return {state, tools: hostTools, config};
}

/** @param {ReturnType<typeof localRocmSdkContext>} context @param {unknown} tools
 * @param {{verifyInputs?:typeof verifyRocmBuildInputs,verifySources?:typeof ciSources.verifyRocmSources,
 *   auditComgr?:typeof ciRocm.auditRocmComgrSdk,auditHost?:typeof ciRocm.auditRocmHostSdk,
 *   verifyHostPrerequisites?:typeof checkRocmHostPrerequisites,
 *   hostTools?:(state:ReturnType<typeof localRocmHostState>)=>ReturnType<typeof ciRocm.rocmHostBuildTools>,
 *   hostConfig?:(state:ReturnType<typeof localRocmHostState>,tools:ReturnType<typeof ciRocm.rocmHostBuildTools>,env:NodeJS.ProcessEnv)=>ReturnType<typeof ciRocm.rocmHostBuildConfig>}} [dependencies] */
function auditLocalRocmSdkInputs(context, tools, dependencies = {}) {
  const selected = localRocmSdkContext(context), comgrTools = localRocmTools(tools), comgr = auditLocalRocmComgrInputs(selected, comgrTools, dependencies),
    host = localRocmHostInputs(selected, comgrTools, process.env, dependencies),
    audit = (dependencies.auditHost || ciRocm.auditRocmHostSdk)(host.state, host.config), paths = selected.paths,
    expected = localHostAdmission(selected, host.tools, audit);
  return {context: selected, comgr, state: host.state, hostTools: host.tools, config: host.config, audit, receipt: expected, paths};
}

/** Keep Intel driver spellings for COMGR; aliases may resolve only inside the pinned SDK.
 * @param {unknown} value @param {string} [compilerBin] @returns {{cc:string,cxx:string,cmake:string}} */
function localRocmTools(value, compilerBin = path.join(oneapiCompilerRoot, "2026.2/bin")) {
  if (!hasExactBuildFields(value, ["cc", "cmake", "cxx"])) {
    throw new Error("Local ROCm COMGR SDK requires only its pinned host compiler and CMake paths");
  }
  const tools = /** @type {{cc?:unknown,cxx?:unknown,cmake?:unknown}} */ (value), selected = {cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake};
  if (!path.isAbsolute(compilerBin) || path.resolve(compilerBin) !== compilerBin) {throw new Error("Pinned Intel compiler directory must be canonical");}
  canonicalBuildDirectory(compilerBin, "Pinned Intel 2026.2 compiler directory");
  const compilerRoot = path.dirname(compilerBin);
  canonicalBuildDirectory(compilerRoot, "Pinned Intel 2026.2 SDK root");
  /** @param {string} file @param {string} label */
  const compilerFile = (file, label) => {
    const stat = fs.statSync(file, {throwIfNoEntry: false});
    let real;
    try {real = fs.realpathSync(file);} catch {throw new Error(label + " is missing or resolves outside the pinned Intel 2026.2 SDK");}
    const relative = path.relative(compilerRoot, real);
    if (!stat?.isFile() || !real || relative === "" || path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep)) {
      throw new Error(label + " is missing or resolves outside the pinned Intel 2026.2 SDK");
    }
    fs.accessSync(file, fs.constants.X_OK);
  };
  if (typeof selected.cc !== "string" || typeof selected.cxx !== "string" || typeof selected.cmake !== "string" ||
      selected.cc !== path.join(compilerBin, "icx") || selected.cxx !== path.join(compilerBin, "icpx") ||
      !path.isAbsolute(selected.cmake) || path.resolve(selected.cmake) !== selected.cmake ||
      path.basename(selected.cxx) !== "icpx") {
    throw new Error("Local ROCm SDK requires Intel oneAPI 2026.2 icx/icpx and a canonical CMake executable");
  }
  compilerFile(selected.cc, "Intel icx"); compilerFile(selected.cxx, "Intel icpx alias");
  regularFile(selected.cmake, "ROCm SDK CMake"); fs.accessSync(selected.cmake, fs.constants.X_OK);
  return /** @type {{cc:string,cxx:string,cmake:string}} */ (selected);
}

/** @param {ReturnType<typeof localRocmSdkContext>} context @param {ReturnType<typeof ciRocm.rocmHostBuildTools>} tools
 * @param {ReturnType<typeof ciRocm.auditRocmHostSdk>} audit */
function localHostAdmission(context, tools, audit) {
  const state = localRocmHostState(context), configPath = context.paths.hostConfig, resultPath = path.join(state.runtimeOutput, "result.json");
  return {version: 1, target: "linux-rocm-host", root: state.hostSdkRoot,
    comgrAdmission: {path: context.paths.comgrAdmission, sha256: fileHash(context.paths.comgrAdmission)},
    config: {path: configPath, sha256: fileHash(configPath)}, tools: Object.fromEntries(Object.entries(tools).map(([name, file]) =>
      [name, {path: file, sha256: fileHash(file)}])), receipt: {path: resultPath, sha256: fileHash(resultPath)},
    prefix: audit.sdk.prefix, buildOnly: true, executed: false, publicationQualified: false};
}

/** Acquire pinned ROCm sources once and build/admit the shared COMGR and host-runtime producers.
 * All acquisition and vendor work remains inside one systemd cgroup scope.
 * @param {unknown} context @param {unknown} tools
 * @param {{run?:typeof local.runCommand,verifyScope?:()=>unknown,verifyInputs?:typeof verifyRocmBuildInputs,
 *   verifySources?:typeof ciSources.verifyRocmSources,assertFetch?:typeof ciSources.assertSdkFetchProvenance,
 *   fetchParser?:typeof ciSources.fetchRocmParserSource,buildComgr?:typeof ciRocm.buildRocmComgrFromInputs,
 *   auditComgr?:typeof ciRocm.auditRocmComgrSdk,buildHost?:(config:unknown,output:string,env:NodeJS.ProcessEnv,configFile:string)=>unknown,
 *   auditHost?:typeof ciRocm.auditRocmHostSdk,verifyHostPrerequisites?:typeof checkRocmHostPrerequisites,
 *   hostTools?:(state:ReturnType<typeof localRocmHostState>)=>ReturnType<typeof ciRocm.rocmHostBuildTools>,
 *   hostConfig?:(state:ReturnType<typeof localRocmHostState>,tools:ReturnType<typeof ciRocm.rocmHostBuildTools>,env:NodeJS.ProcessEnv)=>ReturnType<typeof ciRocm.rocmHostBuildConfig>,
 *   validateTools?:(tools:unknown)=>{cc:string,cxx:string,cmake:string}}} [dependencies] */
function runLocalRocmSdkChild(context, tools, dependencies = {}) {
  (dependencies.verifyScope || local.verifyCurrentMemoryScope)();
  const selected = localRocmSdkContext(context), comgrTools = (dependencies.validateTools || localRocmTools)(tools), paths = selected.paths,
    // The ROCm/COMGR recipe is CI's own and accepts only CI's job counts.
    env = {...process.env, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: String(local.ciRecipeJobs(local.localBuildJobs()))}, git = "/usr/bin/git",
    verifyInputs = dependencies.verifyInputs || verifyRocmBuildInputs, verifySources = dependencies.verifySources || ciSources.verifyRocmSources,
    assertFetch = dependencies.assertFetch || ciSources.assertSdkFetchProvenance, fetchParser = dependencies.fetchParser || ciSources.fetchRocmParserSource,
    run = dependencies.run || local.runCommand;
  verifyInputs(selected);
  const host = localRocmHostInputs(selected, comgrTools, env, dependencies);
  regularFile(git, "Pinned source-fetch Git executable"); fs.accessSync(git, fs.constants.X_OK);
  const sourceState = localRocmSourceState(paths);
  const reusedSources = canonicalBuildDirectory(paths.source, "Local ROCm sources", true);
  if (!reusedSources) {
    if (fs.lstatSync(paths.pythonSourceRoot, {throwIfNoEntry: false})) {throw new Error("ROCm parser source exists without its ROCm source cohort; preserve the workspace");}
    const fetcher = path.join(repo, "scripts/fetch-native-sdk-sources.js");
    run(process.execPath, [fetcher, ...ciSources.rocmSourceFetchArgs(sourceState, git)],
      {label: "Pinned ROCm Git source acquisition", env, cwd: repo});
    canonicalBuildDirectory(paths.source, "Fetched local ROCm source cohort");
    regularFile(paths.fetchReceipt, "Pinned ROCm source fetch receipt", true);
    regularFile(paths.rootsFile, "Pinned ROCm source roots receipt", true);
    assertFetch(readBuildJson(paths.fetchReceipt), readBuildJson(paths.rootsFile), "rocm", paths.source,
      fileHash(fetcher), {path: git, sha256: fileHash(git)});
    fetchParser(sourceState, env);
  }
  verifyLocalRocmSources(paths, env, git, verifySources);

  const buildState = localRocmBuildState(selected), outputFlags = [paths.hipHeadersRoot, paths.resourcesRoot, paths.sdkRoot],
    outputExists = outputFlags.map(root => canonicalBuildDirectory(root, "Local ROCm COMGR output", true)), configExists = Boolean(fs.lstatSync(paths.resourcesConfig, {throwIfNoEntry: false})),
    admissionStat = fs.lstatSync(paths.comgrAdmission, {throwIfNoEntry: false});
  let reusedComgr = outputExists.every(Boolean);
  if (!outputExists.some(Boolean)) {
    if (admissionStat || configExists) {throw new Error("ROCm COMGR admission or config exists without all producer outputs; preserve the workspace");}
    const verifyLlvm = () => verifyInputs(selected);
    (dependencies.buildComgr || ciRocm.buildRocmComgrFromInputs)(buildState, comgrTools, env, verifyLlvm);
    const audited = auditLocalRocmComgrInputs(selected, comgrTools, dependencies);
    fs.writeFileSync(paths.comgrAdmission, JSON.stringify(audited.receipt, null, 2) + "\n", {flag: "wx", mode: 0o600});
    reusedComgr = false;
  } else {
    if (!reusedComgr || configExists || !admissionStat?.isFile() || admissionStat.isSymbolicLink()) {
      throw new Error("ROCm COMGR output is partial or lacks its immutable local admission; preserve the workspace");
    }
    regularFile(paths.comgrAdmission, "Local ROCm COMGR admission receipt", true);
    const audited = auditLocalRocmComgrInputs(selected, comgrTools, dependencies);
    if (!isDeepStrictEqual(readBuildJson(paths.comgrAdmission), audited.receipt)) {
      throw new Error("Local ROCm COMGR cache fingerprints changed; preserve the workspace");
    }
    reusedComgr = true;
  }

  const hostState = host.state, hostTools = host.tools, hostConfig = host.config,
    hostPaths = [paths.hostSdkRoot, paths.runtimeOutput],
    hostExists = hostPaths.map(root => canonicalBuildDirectory(root, "Local ROCm host SDK output", true)), hostAdmissionStat = fs.lstatSync(paths.hostAdmission, {throwIfNoEntry: false});
  if (hostExists.some(Boolean)) {
    if (!hostExists.every(Boolean) || !hostAdmissionStat?.isFile() || hostAdmissionStat.isSymbolicLink() ||
        !fs.lstatSync(paths.hostConfig, {throwIfNoEntry: false})?.isFile()) {
      throw new Error("ROCm host SDK output is partial or lacks its immutable local admission; preserve the workspace");
    }
    regularFile(paths.hostAdmission, "Local ROCm host SDK admission receipt", true);
    regularFile(paths.hostConfig, "Local ROCm host SDK config", true);
    const audited = auditLocalRocmSdkInputs(selected, comgrTools, dependencies);
    if (!isDeepStrictEqual(readBuildJson(paths.hostAdmission), audited.receipt) ||
        !isDeepStrictEqual(readBuildJson(paths.hostConfig), hostConfig)) {
      throw new Error("Local ROCm host SDK cache fingerprints changed; preserve the workspace");
    }
    return rocmSdkResult(selected, {reusedSources, reusedComgr, reusedHost: true});
  }
  if (hostAdmissionStat) {throw new Error("ROCm host admission exists without its output; preserve the workspace");}
  fs.mkdirSync(paths.hostSdkRoot, {mode: 0o700});
  fs.writeFileSync(paths.hostConfig, JSON.stringify(hostConfig, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const producer = dependencies.buildHost || rocmProducer.buildRocmHostRuntime;
  producer(hostConfig, paths.runtimeOutput, env, paths.hostConfig);
  const audit = (dependencies.auditHost || ciRocm.auditRocmHostSdk)(hostState, hostConfig),
    receipt = localHostAdmission(selected, hostTools, audit);
  fs.writeFileSync(paths.hostAdmission, JSON.stringify(receipt, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return rocmSdkResult(selected, {reusedSources, reusedComgr, reusedHost: false});
}

/** @param {ReturnType<typeof localRocmSdkContext>} context @param {{reusedSources:boolean,reusedComgr:boolean,reusedHost:boolean}} reused */
function rocmSdkResult(context, reused) {
  const p = context.paths;
  return {workspace: context.workspace, sources: p.source, hipHeaders: p.hipHeadersRoot, resources: p.resourcesRoot,
    comgrSdk: p.sdkRoot, comgrReceipt: path.join(p.sdkRoot, "result.json"), runtimeOutput: p.runtimeOutput,
    rocmSdk: path.join(p.runtimeOutput, "sdk"), rocmReceipt: path.join(p.runtimeOutput, "result.json"), reused,
    buildOnly: true, executed: false, publicationQualified: false};
}

/** @typedef {{run?:typeof local.runCommand,tools?:ReturnType<typeof local.localTools>,resolveTools?:typeof local.localTools,
 * versionOf?:(file:string)=>string,environment?:NodeJS.ProcessEnv,
 * validateTools?:(tools:unknown)=>{cc:string,cxx:string,cmake:string}}} LocalRocmWrapperDependencies */

/** Build or reuse CPU-only ROCm/COMGR plus static host-runtime archives under the established cap.
 * @param {unknown} context @param {LocalRocmWrapperDependencies} [dependencies] */
function buildLocalLinuxRocmSdk(context, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local Linux ROCm SDK currently requires Linux x64");}
  const selected = localRocmSdkContext(context), env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"},
    resolved = dependencies.tools || (dependencies.resolveTools || local.localTools)(env, dependencies.versionOf),
    tools = {cc: resolved.cc, cxx: resolved.cxx, cmake: resolved.cmake}, validateTools = dependencies.validateTools || localRocmTools,
    selectedTools = validateTools(tools), invoke = local.scopedInvoker(resolved, env, dependencies.run || local.runCommand), p = selected.paths,
    reusedSources = canonicalBuildDirectory(p.source, "Local ROCm sources", true),
    reusedComgr = [p.hipHeadersRoot, p.resourcesRoot, p.sdkRoot, p.comgrAdmission].every(file => Boolean(fs.lstatSync(file, {throwIfNoEntry: false}))),
    reusedHost = [p.hostSdkRoot, p.runtimeOutput, p.hostAdmission].every(file => Boolean(fs.lstatSync(file, {throwIfNoEntry: false})));
  invoke("systemd memory-limit preflight", process.execPath, [__filename, "--memory-probe"]);
  invoke("Local Linux ROCm/COMGR and host SDK build-only slice", process.execPath,
    [__filename, "--local-rocm-sdk", selected.workspace, selectedTools.cmake, selectedTools.cc, selectedTools.cxx]);
  return rocmSdkResult(selected, {reusedSources, reusedComgr, reusedHost});
}

/** @param {string[]} argv @param {()=>unknown} [verifyScope]
 * @param {Parameters<typeof runLocalRocmSdkChild>[2]} [dependencies] */
function main(argv, verifyScope = local.verifyCurrentMemoryScope, dependencies = {}) {
  const [command, ...args] = argv;
  if (command === "--memory-probe" && args.length === 0) {verifyScope(); return;}
  if (command === "--workspace" && args.length === 1 && args[0]) {return buildLocalLinuxRocmSdk(localRocmSdkContextFromWorkspace(args[0]));}
  if (command !== "--local-rocm-sdk" || args.length !== 4 || args.some(value => !value)) {
    throw new Error("Usage: node scripts/build-local-native-rocm-sdk.js --workspace /workspace\n" +
      "       node scripts/build-local-native-rocm-sdk.js --local-rocm-sdk /workspace /cmake /icx /icpx");
  }
  const [workspace, cmake, cc, cxx] = args;
  if (![workspace, cmake, cc, cxx].every(value => path.isAbsolute(value || "") && path.resolve(value || "") === value)) {
    throw new Error("Local ROCm SDK internal paths must be canonical and absolute");
  }
  verifyScope();
  const selected = localRocmSdkContextFromWorkspace(workspace || ""), expected = local.localTools(process.env), tools = {cmake, cc, cxx};
  if (cmake !== expected.cmake || cc !== expected.cc || cxx !== expected.cxx) {throw new Error("Local ROCm SDK internal tools differ from the pinned oneAPI 2026.2 tool selection");}
  return runLocalRocmSdkChild(selected, tools, {...dependencies, verifyScope});
}

if (require.main === module) {
  try {main(process.argv.slice(2));} catch (error) {console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1;}
}

module.exports = {localRocmSdkPaths, localRocmSdkContextFromWorkspace, localRocmSdkContext, localRocmBuildState,
  verifyRocmBuildInputs, auditLocalRocmComgrInputs, auditLocalRocmSdkInputs, localRocmTools,
  localRocmHostState, localRocmHostInputs, runLocalRocmSdkChild, rocmSdkResult, buildLocalLinuxRocmSdk, main};
