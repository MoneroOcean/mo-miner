"use strict";

// Continue admitted compiler, OpenMP and ROCm source SDKs through the public static ACPP builder.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk"), jit = require("./build-local-native-jit-sdk"), rocm = require("./build-local-native-rocm-sdk"),
  ciSources = require("./native-ci-linux-sources"),
  ciSupport = require("./native-ci-linux-support");
const {verifyPreparedRoots} = jit;
const {verifyLinuxLlvmAdmission} = require("./build-native-llvm-sdk");
const {auditDpcppCudaInputs} = require("./native-dpcpp-cuda-inputs");
const {staticSyclSdkPlan} = require("./build-native-sycl-sdk");
const {nativeStaticRuntimeInputs} = require("./native-release-sdk-inputs");
const {nativeOpenmpInputs} = require("./build-native-openmp-sdk");
const {archiveName} = require("./native-llvm-sdk");
const {fileHash, readBuildJson} = require("./build-source-inputs");
const {generatedIncludeInventory} = require("./native-windows-gpu-config");

const repo = fs.realpathSync(path.resolve(__dirname, "..")), jobs = 2;
/** @param {string} name */ const producer = name => path.join(repo, "scripts", name);

/** ACPP source and SDK paths are stable workspace outputs; interrupted outputs are never removed.
 * @param {string} workspace */
function localAcppSdkPaths(workspace) {
  if (typeof workspace !== "string" || !path.isAbsolute(workspace) || path.resolve(workspace) !== workspace) {
    throw new Error("Local ACPP SDK workspace must be a canonical absolute path");
  }
  const fetch = path.join(workspace, "acpp-fetch"), prepared = path.join(workspace, "acpp-sources"), sdk = path.join(workspace, "acpp-cuda-hip-sdk");
  return {workspace, fetch, rootsFile: path.join(fetch, "roots.json"), fetchReceipt: path.join(fetch, "fetch.json"),
    prepared, preparedReceipt: path.join(prepared, "sources.json"), preparedCheckout: path.join(prepared, "acpp"), sdk,
    sdkReceipt: path.join(sdk, "result.json")};
}

/** Build the local context from stable workspace paths and the ROCm stage's exact result schema.
 * Reuse flags are informational; all producer receipts are re-admitted before use.
 * @param {string} workspace @param {ReturnType<typeof rocm.buildLocalLinuxRocmSdk>} [rocmResult] */
function localAcppSdkContextFromWorkspace(workspace, rocmResult) {
  const compiler = jit.localJitSdkContextFromWorkspace(workspace), rocmContext = rocm.localRocmSdkContextFromWorkspace(workspace),
    rocmState = rocm.localRocmSdkContext(rocmContext), paths = localAcppSdkPaths(workspace), rocmPaths = rocmState.paths;
  const currentRocm = rocmResult || rocm.rocmSdkResult(rocmState, {
    reusedSources: Boolean(fs.lstatSync(rocmPaths.source, {throwIfNoEntry: false})),
    reusedComgr: Boolean(fs.lstatSync(rocmPaths.sdkRoot, {throwIfNoEntry: false})),
    reusedHost: Boolean(fs.lstatSync(rocmPaths.runtimeOutput, {throwIfNoEntry: false})),
  });
  return {source: compiler.source, inputs: compiler.inputs, rocm: currentRocm, paths};
}

/** Require the real compiler and ROCm stage path results from one marked workspace.
 * @param {unknown} value */
function localAcppSdkContext(value) {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).sort().join(",") !== "inputs,paths,rocm,source") {
    throw new Error("Local ACPP SDK needs only admitted compiler inputs and the ROCm SDK result");
  }
  const context = /** @type {{source?:{workspace?:unknown,compilerFetch?:unknown,compilerSources?:unknown,buildInputs?:unknown,llvmSdk?:unknown},
    inputs?:{workspace?:unknown,dpcppRoot?:unknown,cudaRoot?:unknown,manifest?:unknown},rocm?:unknown,paths?:unknown}} */ (value),
    source = context.source, inputs = context.inputs;
  if (!source || !inputs || typeof source.workspace !== "string" || source.workspace !== inputs.workspace) {
    throw new Error("Local ACPP SDK stage results must share one admitted workspace");
  }
  realDirectory(source.workspace, "Admitted native SDK workspace");
  const workspace = local.admitWorkspace(source.workspace, repo), support = local.supportPaths(workspace), expected = jit.localJitSdkContextFromWorkspace(workspace),
    expectedPaths = localAcppSdkPaths(workspace), rocmContext = rocm.localRocmSdkContextFromWorkspace(workspace),
    rocmState = rocm.localRocmSdkContext(rocmContext), result = /** @type {ReturnType<typeof rocm.buildLocalLinuxRocmSdk> | undefined} */ (context.rocm),
    expectedRocm = result ? rocm.rocmSdkResult(rocmState, result.reused) : undefined;
  if (source.compilerFetch !== expected.source.compilerFetch || source.compilerSources !== expected.source.compilerSources ||
      source.buildInputs !== expected.source.buildInputs || source.llvmSdk !== expected.source.llvmSdk ||
      inputs.dpcppRoot !== expected.inputs.dpcppRoot || inputs.cudaRoot !== expected.inputs.cudaRoot || inputs.manifest !== expected.inputs.manifest ||
      !isDeepStrictEqual(context.paths, expectedPaths) || !result || !isDeepStrictEqual(result, expectedRocm)) {
    throw new Error("Local ACPP SDK inputs do not match the exact admitted workspace paths and ROCm result");
  }
  const reused = result.reused;
  if (!reused || Object.keys(reused).sort().join(",") !== "reusedComgr,reusedHost,reusedSources" ||
      Object.values(reused).some(flag => typeof flag !== "boolean")) {
    throw new Error("Local ACPP SDK received an invalid ROCm reuse record");
  }
  return {workspace, source: expected.source, inputs: expected.inputs, rocm: result, paths: expectedPaths, support,
    rocmContext: rocmState};
}

/** @param {string} directory @param {string} label */
function realDirectory(directory, label) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!stat?.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
    throw new Error(label + " is missing or not a canonical directory: " + directory);
  }
}

/** Require current compiler, CUDA, source, OpenMP and source-built ROCm receipts before ACPP work.
 * @param {unknown} context @param {ReturnType<typeof localAcppSdkTools>} tools
 * @param {{verifyLlvm?:typeof verifyLinuxLlvmAdmission,auditCuda?:typeof auditDpcppCudaInputs,
 *   auditRocm?:typeof rocm.auditLocalRocmSdkInputs,auditOpenmp?:typeof nativeOpenmpInputs}} [dependencies] */
function auditLocalAcppSdkInputs(context, tools, dependencies = {}) {
  const selected = localAcppSdkContext(context), support = selected.support;
  verifyPreparedRoots(path.join(support.compilerSources, "sources.json"), support.compilerSources, "compiler");
  verifyPreparedRoots(path.join(support.linuxSources, "sources.json"), support.linuxSources, "linux");
  (dependencies.verifyLlvm || verifyLinuxLlvmAdmission)(support.llvmSdk);
  const cuda = (dependencies.auditCuda || auditDpcppCudaInputs)(local.localDpcppCudaState(support), readBuildJson(selected.inputs.manifest)),
    rocmAudit = (dependencies.auditRocm || rocm.auditLocalRocmSdkInputs)(selected.rocmContext,
      {cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake}),
    openmp = (dependencies.auditOpenmp || nativeOpenmpInputs)(support.openmpSdk),
    openmpRecord = readBuildJson(openmp.receipt), openmpTools = ciSupport.openmpBuildTools(),
    expectedOpenmpTools = Object.fromEntries(Object.entries(openmpTools).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}])),
    sourceState = {source: selected.paths.fetch, preparedReceipt: selected.paths.preparedReceipt,
      preparedCheckout: selected.paths.preparedCheckout};
  if (openmpRecord.archive.path !== path.join(support.buildInputs, archiveName) ||
      openmpRecord.fetchReceipt.path !== path.join(support.buildInputs, "fetch.json") ||
      !isDeepStrictEqual(openmpRecord.tools, expectedOpenmpTools)) {
    throw new Error("Local OpenMP SDK does not bind the admitted workspace LLVM archive and current tools");
  }
  if (directoryExists(selected.paths.prepared, "Local ACPP prepared source output") &&
      !directoryExists(selected.paths.fetch, "Local ACPP fetch output")) {
    throw new Error("Prepared ACPP sources exist without their original fetch cohort; preserve the workspace");
  }
  if (directoryExists(selected.paths.prepared, "Local ACPP prepared source output")) {
    ciSources.verifyAcppSource({env: process.env}, sourceState, tools.git);
  }
  return {cuda, rocm: rocmAudit, openmp, sourceState};
}

/** Use the public static-SYCL planner and exact CUDA/HIP roots for the combined SDK.
 * @param {unknown} context @param {ReturnType<typeof localAcppSdkTools>} tools
 * @param {NodeJS.ProcessEnv} [environment] */
function localAcppSdkBuildEnvironment(context, tools, environment = process.env) {
  const selected = localAcppSdkContext(context), support = selected.support,
    rocmPaths = selected.rocm, toolDirectories = [...new Set([tools.cmake, tools.ninja, tools.git].map(file => path.dirname(file)))];
  return {...environment, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: String(jobs), MOM_AOT_SYCL_BACKEND: "cuda-hip",
    MOM_AOT_ROCM_STATIC_ROOT: rocmPaths.rocmSdk, MOM_AOT_LLD_ROOT: path.join(selected.source.llvmSdk, "sdk"),
    ROCM_DEVICE_LIBS_PATH: path.join(rocmPaths.resources, "install/amdgcn/bitcode"),
    MOM_AOT_COMGR_RECEIPT: rocmPaths.comgrReceipt, MOM_AOT_OPENMP_LIBRARY: path.join(support.openmpSdk, "sdk/libomp.a"),
    PATH: [...toolDirectories, environment["PATH"] || ""].filter(Boolean).join(path.delimiter)};
}

/** Return the original public producer command and plan; don't duplicate its CMake recipe.
 * @param {unknown} context @param {ReturnType<typeof localAcppSdkTools>} tools
 * @param {NodeJS.ProcessEnv} [environment] */
function localAcppSdkBuildCommand(context, tools, environment = process.env) {
  const selected = localAcppSdkContext(context), llvmSdk = path.join(selected.source.llvmSdk, "sdk"),
    env = localAcppSdkBuildEnvironment(context, tools, environment),
    plan = staticSyclSdkPlan(selected.paths.preparedCheckout, llvmSdk, selected.inputs.cudaRoot,
      selected.paths.sdk, jobs, env, {cc: tools.cc, cxx: tools.cxx});
  return {script: "scripts/build-native-sycl-sdk.js", args: [selected.paths.preparedCheckout, llvmSdk,
    selected.inputs.cudaRoot, selected.paths.sdk, tools.cc, tools.cxx], env, plan};
}

/** Bind the public receipt reader's files to the same local CUDA/HIP, OpenMP and source roots.
 * @param {unknown} context @param {ReturnType<typeof nativeStaticRuntimeInputs>} sdk
 * @param {any} record @param {ReturnType<typeof localAcppSdkTools>} tools
 * @param {ReturnType<typeof auditDpcppCudaInputs>} cuda @param {ReturnType<typeof auditLocalAcppSdkInputs>} inputs */
function assertLocalAcppSdkBindings(context, sdk, record, tools, cuda, inputs) {
  const selected = localAcppSdkContext(context), paths = selected.paths, rocmResult = selected.rocm,
    comgr = inputs.rocm.comgr.comgr.sdk, openmp = inputs.openmp,
    cudaStatic = cuda.cuda?.staticRuntime, fingerprints = record.fingerprints, libraryHashes = record.libraries,
    hipLibraries = ["libamdhip64.a", "libhiprtc.a", "libhsakmt-staticdrm.a", "libelf.a", "libdrm_amdgpu.a", "libdrm.a", "libnuma.a"]
      .map(name => path.join(rocmResult.rocmSdk, "lib", name)), hsaRuntime = path.join(rocmResult.rocmSdk, "lib/libhsa-runtime64.a");
  const expectedTopLevel = "backend,buildOnly,compilerVersions,cudaVersion,fingerprints,hostCompilerInputs,hostCompilers,inputs,libraries,publicationQualified,recipe,revision,scope,version";
  if (Object.keys(record).sort().join(",") !== expectedTopLevel || record.version !== 1 || record.backend !== "cuda-hip" ||
      record.buildOnly !== true || record.publicationQualified !== false || record.cudaVersion !== 12060 ||
      record.revision !== "da2463e45aa90aa36306c45abcfc05b87de51bc6" || record.inputs?.source !== paths.preparedCheckout ||
      record.inputs.build !== path.join(paths.sdk, "build") || record.inputs.cxx !== path.join(selected.source.llvmSdk, "sdk/bin/clang++") ||
      record.hostCompilers?.cc !== tools.cc || record.hostCompilers?.cxx !== tools.cxx ||
      sdk.receipt !== paths.sdkReceipt || sdk.source !== paths.preparedCheckout || sdk.build !== path.join(paths.sdk, "build") ||
      sdk.cxx !== path.join(selected.source.llvmSdk, "sdk/bin/clang++") || sdk.backend !== "cuda-hip" || sdk.cudaVersion !== 12060 ||
      !cudaStatic || !comgr || !openmp ||
      record.inputs.comgr?.receipt !== rocmResult.comgrReceipt || sdk.comgr?.receipt !== rocmResult.comgrReceipt ||
      !isDeepStrictEqual(record.inputs.comgr, comgr) || !isDeepStrictEqual(sdk.comgr, comgr) ||
      comgr.archive !== path.join(rocmResult.comgrSdk, "build/libamd_comgr.a") ||
      comgr.roots.llvm !== path.join(selected.source.llvmSdk, "sdk") || comgr.roots.lld !== path.join(selected.source.llvmSdk, "sdk") ||
      comgr.roots.device !== path.join(rocmResult.resources, "install") ||
      rocmResult.rocmSdk !== inputs.rocm.audit.sdk.prefix || sdk.openmp?.archive !== openmp.archive ||
      !record.inputs.libraries.includes(comgr.archive) || !sdk.libraries.includes(comgr.archive) ||
      libraryHashes[comgr.archive] !== fileHash(comgr.archive) || sdk.files[comgr.archive] !== fileHash(comgr.archive) ||
      !record.inputs.links.includes(hsaRuntime) || !sdk.links.includes(hsaRuntime) ||
      fingerprints?.[hsaRuntime] !== fileHash(hsaRuntime) || sdk.files[hsaRuntime] !== fileHash(hsaRuntime) ||
      record.inputs.libraries?.length !== sdk.libraries.length ||
      !hipLibraries.every(file => record.inputs.libraries.includes(file) && sdk.libraries.includes(file) &&
        sdk.files[file] === fileHash(file) && libraryHashes[file] === fileHash(file)) ||
      !["cuda.h", "cuda_runtime_api.h"].every(name => {
        const file = path.join(selected.inputs.cudaRoot, "include", name), digest = cuda.cuda?.headers?.files?.[name];
        return typeof digest === "string" && fingerprints?.[file] === digest && sdk.files[file] === digest;
      }) ||
      Object.values(cudaStatic).some(row => !row || !record.inputs.libraries.includes(row.path) ||
        fingerprints?.[row.path] !== row.sha256 || libraryHashes?.[row.path] !== row.sha256 || sdk.files[row.path] !== row.sha256)) {
    throw new Error("Local ACPP SDK does not bind the exact admitted CUDA/HIP, OpenMP and source roots");
  }
  return sdk;
}

/** Re-admit the build-only public ACPP output; no runtime proof, GPU execution or publication.
 * @param {unknown} context @param {ReturnType<typeof localAcppSdkTools>} tools
 * @param {{auditInputs?:typeof auditLocalAcppSdkInputs,readSdk?:typeof nativeStaticRuntimeInputs}} [dependencies] */
function auditLocalAcppSdk(context, tools, dependencies = {}) {
  const selected = localAcppSdkContext(context), inputs = (dependencies.auditInputs || auditLocalAcppSdkInputs)(context, tools),
    record = readBuildJson(selected.paths.sdkReceipt), sdk = (dependencies.readSdk || nativeStaticRuntimeInputs)(selected.paths.sdkReceipt,
      {openmpSdkRoot: selected.support.openmpSdk});
  return assertLocalAcppSdkBindings(context, sdk, record, tools, inputs.cuda, inputs);
}

/** The static-runtime receipt does not fingerprint the separately loaded compiler
 * plugin/generated headers. Capture this closure only after a fresh public build;
 * warm caches must match the original capture, never bless their current bytes.
 * @param {ReturnType<typeof localAcppSdkContext>} selected */
function localAcppGpuArtifacts(selected) {
  const plugin = path.join(selected.paths.sdk, "build/src/compiler/libacpp-clang.so"),
    generatedInclude = path.join(selected.paths.sdk, "build/include"), stat = fs.lstatSync(plugin);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size < 1 || stat.size > (128 << 20) ||
      fs.realpathSync(plugin) !== plugin) {throw new Error("Local ACPP compiler plugin is not a bounded canonical regular file");}
  const generated = generatedIncludeInventory(generatedInclude);
  if (!generated.some(row => row.path === "hipSYCL/common/config.hpp")) {throw new Error("Local ACPP generated headers omit config.hpp");}
  return {version: 1, receiptSha256: fileHash(selected.paths.sdkReceipt),
    plugin: {path: plugin, bytes: stat.size, sha256: fileHash(plugin)}, generatedInclude, generated};
}

/** Read-only admission for GPU factory inputs; callers first admit the public SDK.
 * @param {unknown} context */
function admitLocalAcppGpuArtifacts(context) {
  const selected = localAcppSdkContext(context), file = path.join(selected.workspace, "acpp-gpu-artifacts.json"),
    stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size > (1 << 20) || fs.realpathSync(file) !== file) {
    throw new Error("Local ACPP GPU artifact capture is missing or unsafe; preserve the workspace and rebuild explicitly");
  }
  const current = localAcppGpuArtifacts(selected);
  if (!isDeepStrictEqual(readBuildJson(file), current)) {throw new Error("Local ACPP plugin or generated headers differ from their original build capture");}
  return {...current, capture: file, captureSha256: fileHash(file)};
}

/** @param {unknown} value @returns {{git:string,cmake:string,ninja:string,cc:string,cxx:string}} */
function localAcppSdkTools(value) {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).sort().join(",") !== "cc,cmake,cxx,git,ninja") {
    throw new Error("Local ACPP SDK requires the exact pinned Git, CMake, Ninja and host compiler tools");
  }
  const tools = /** @type {{git?:unknown,cmake?:unknown,ninja?:unknown,cc?:unknown,cxx?:unknown}} */ (value),
    selected = {git: tools.git, cmake: tools.cmake, ninja: tools.ninja, cc: tools.cc, cxx: tools.cxx};
  if (Object.values(selected).some(file => typeof file !== "string" || !path.isAbsolute(file) || path.resolve(file) !== file)) {
    throw new Error("Local ACPP SDK tools must be canonical absolute paths");
  }
  return /** @type {{git:string,cmake:string,ninja:string,cc:string,cxx:string}} */ (selected);
}

/** @param {string} directory @param {string} label */
function directoryExists(directory, label) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!stat) {return false;}
  if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
    throw new Error(label + " is not a canonical directory: " + directory);
  }
  return true;
}

/** Fetch/prepare only the pinned ACPP source tree; never repair partial outputs.
 * @param {unknown} context @param {ReturnType<typeof localAcppSdkTools>} tools
 * @param {NodeJS.ProcessEnv} env @param {typeof local.runCommand} run */
function prepareLocalAcppSources(context, tools, env, run) {
  const selected = localAcppSdkContext(context), paths = selected.paths, fetchExisted = directoryExists(paths.fetch, "Local ACPP fetch output"),
    preparedExisted = directoryExists(paths.prepared, "Local ACPP prepared source output");
  if (preparedExisted && !fetchExisted) {
    throw new Error("Prepared ACPP sources exist without their original fetch cohort; preserve the workspace");
  }
  if (!fetchExisted) {
    run(process.execPath, [producer("fetch-native-sdk-sources.js"), "acpp", tools.git, paths.fetch, "--allow-network"],
      {label: "Pinned ACPP source acquisition", env, cwd: repo});
  }
  if (!preparedExisted) {
    run(process.execPath, [producer("prepare-native-sdk-sources.js"), "acpp", paths.rootsFile, paths.prepared],
      {label: "Offline ACPP source preparation", env, cwd: repo});
  }
  ciSources.verifyAcppSource({env}, {source: paths.fetch, preparedReceipt: paths.preparedReceipt,
    preparedCheckout: paths.preparedCheckout}, tools.git);
  return {sources: fetchExisted, prepared: preparedExisted};
}

/** Run only from the internal memory-limited entry; builds a new SDK or audits an existing one.
 * @param {unknown} context @param {unknown} tools @param {{verifyScope?:()=>unknown,run?:typeof local.runCommand,
 * auditInputs?:typeof auditLocalAcppSdkInputs,auditSdk?:typeof auditLocalAcppSdk}} [dependencies] */
function runLocalAcppSdkChild(context, tools, dependencies = {}) {
  (dependencies.verifyScope || local.verifyCurrentMemoryScope)();
  const selected = localAcppSdkContext(context), suppliedTools = localAcppSdkTools(tools), env = {...process.env, MOM_AOT_BUILD_ONLY: "1"},
    currentTools = local.localTools(env), expectedTools = {git: currentTools.git, cmake: currentTools.cmake, ninja: currentTools.ninja,
      cc: currentTools.cc, cxx: currentTools.cxx},
    run = dependencies.run || local.runCommand, auditInputs = dependencies.auditInputs || auditLocalAcppSdkInputs,
    auditSdk = dependencies.auditSdk || auditLocalAcppSdk;
  if (!isDeepStrictEqual(suppliedTools, expectedTools)) {throw new Error("Local ACPP SDK internal tools differ from the current pinned tool selection");}
  auditInputs(context, suppliedTools);
  const sources = prepareLocalAcppSources(context, suppliedTools, env, run), outputExisted = directoryExists(selected.paths.sdk, "Local ACPP SDK output"),
    capture = path.join(selected.workspace, "acpp-gpu-artifacts.json");
  if (!outputExisted && fs.lstatSync(capture, {throwIfNoEntry: false})) {
    throw new Error("Local ACPP GPU capture exists without its SDK; preserve the workspace and investigate");
  }
  if (outputExisted) {auditSdk(context, suppliedTools);} else {
    const command = localAcppSdkBuildCommand(context, suppliedTools, env);
    run(process.execPath, [producer(command.script), ...command.args], {label: "Static combined CUDA/HIP ACPP SDK", env: command.env, cwd: repo});
    auditSdk(context, suppliedTools);
    fs.writeFileSync(capture, JSON.stringify(localAcppGpuArtifacts(selected), null, 2) + "\n", {flag: "wx", mode: 0o600});
  }
  admitLocalAcppGpuArtifacts(context);
  return {workspace: selected.workspace, sources: selected.paths.preparedCheckout, sdk: selected.paths.sdk,
    reused: {...sources, sdk: outputExisted}, buildOnly: true, executed: false, publicationQualified: false};
}

/** Build or re-admit the static CUDA+HIP ACPP SDK under the established 4 GiB/no-swap scope.
 * @param {unknown} context @param {{run?:typeof local.runCommand,tools?:ReturnType<typeof local.localTools>,
 * resolveTools?:typeof local.localTools,versionOf?:(file:string)=>string,environment?:NodeJS.ProcessEnv}} [dependencies] */
function buildLocalLinuxAcppSdk(context, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local Linux ACPP SDK currently requires Linux x64");}
  const selected = localAcppSdkContext(context), env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"},
    resolved = dependencies.tools || (dependencies.resolveTools || local.localTools)(env, dependencies.versionOf),
    tools = localAcppSdkTools({git: resolved.git, cmake: resolved.cmake, ninja: resolved.ninja, cc: resolved.cc, cxx: resolved.cxx}),
    invoke = local.scopedInvoker(resolved, env, dependencies.run || local.runCommand), before = {
      sources: directoryExists(selected.paths.fetch, "Local ACPP fetch output"), prepared: directoryExists(selected.paths.prepared, "Local ACPP prepared source output"),
      sdk: directoryExists(selected.paths.sdk, "Local ACPP SDK output")};
  invoke("systemd memory-limit preflight", process.execPath, [__filename, "--memory-probe"]);
  invoke("Local Linux static ACPP SDK build-only slice", process.execPath,
    [__filename, "--local-acpp-sdk", selected.workspace, tools.git, tools.cmake, tools.ninja, tools.cc, tools.cxx]);
  return {workspace: selected.workspace, sources: selected.paths.preparedCheckout, sdk: selected.paths.sdk,
    reused: before, buildOnly: true, executed: false, publicationQualified: false};
}

/** @param {string[]} argv @param {()=>unknown} [verifyScope]
 * @param {Parameters<typeof runLocalAcppSdkChild>[2]} [dependencies] */
function main(argv, verifyScope = local.verifyCurrentMemoryScope, dependencies = {}) {
  const [command, ...args] = argv;
  if (command === "--memory-probe" && args.length === 0) {verifyScope(); return;}
  if (command === "--workspace" && args.length === 1 && args[0]) {
    return buildLocalLinuxAcppSdk(localAcppSdkContextFromWorkspace(args[0]));
  }
  if (command !== "--local-acpp-sdk" || args.length !== 6 || args.some(value => !value)) {
    throw new Error("Usage: node scripts/build-local-native-acpp-sdk.js --workspace /workspace\n" +
      "       node scripts/build-local-native-acpp-sdk.js --local-acpp-sdk /workspace /git /cmake /ninja /icx /icpx");
  }
  const [workspace, git, cmake, ninja, cc, cxx] = args;
  if (![workspace, git, cmake, ninja, cc, cxx].every(value => path.isAbsolute(value || "") && path.resolve(value || "") === value)) {
    throw new Error("Local ACPP SDK internal paths must be canonical and absolute");
  }
  const context = localAcppSdkContextFromWorkspace(workspace || "");
  return runLocalAcppSdkChild(context, {git, cmake, ninja, cc, cxx}, {...dependencies, verifyScope});
}

if (require.main === module) {
  try {main(process.argv.slice(2));} catch (error) {console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1;}
}

module.exports = {localAcppSdkPaths, localAcppSdkContextFromWorkspace, localAcppSdkContext, localAcppSdkBuildEnvironment,
  localAcppSdkBuildCommand, auditLocalAcppSdkInputs, assertLocalAcppSdkBindings, auditLocalAcppSdk,
  localAcppSdkTools, prepareLocalAcppSources, admitLocalAcppGpuArtifacts, runLocalAcppSdkChild, buildLocalLinuxAcppSdk, main, jobs};
