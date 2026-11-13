"use strict";

// Run one original source-JIT build phase per workflow job. The original builder receipts
// remain authoritative; this runner only admits current inputs and transports its mutable tree.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"), {isDeepStrictEqual} = require("node:util");
const ci = require("./prepare-native-ci"), {fileHash, readBuildJson} = require("./build-source-inputs");
const {jitSdkConfig} = require("./build-native-jit-sdk"), {jitBuildStages} = require("./native-jit-stages");
const {verifyPreparedSdkSources} = require("./prepare-native-sdk-sources");
const {verifyJitResources} = require("./prepare-native-sycl-jit-resources");
const {linuxLlvmToolHandoffInputs} = require("./native-linux-llvm-tools");
const {windowsControllerHandoffInputs} = require("./native-windows-controller-handoff");
const {windowsHostSdkInputs} = require("./native-windows-host-sdk-inputs");
const {windowsJitResourcePaths, admitWindowsJitNativeTools} = require("./native-windows-jit-resources");

const phases = Object.freeze(["configure", ...jitBuildStages]), maxConfigBytes = 8 << 20,
  maxArchiveBytes = 16 * 1024 ** 3, maxExpandedBytes = 64 * 1024 ** 3, maxMembers = 500000,
  jobLimitMs = 55 * 60 * 1000, childLimitMs = 45 * 60 * 1000,
  intelCompiler = "/opt/intel/oneapi/compiler/2026.2/bin/icx-cl";

/** Resolve the selected compiler once so config validation matches the emitted toolchain path. */
function canonicalIntelCompiler() {
  const root = fs.realpathSync("/opt/intel/oneapi/compiler/2026.2"), compiler = fs.realpathSync(intelCompiler),
    relative = path.relative(root, compiler), stat = fs.statSync(compiler);
  if (!stat.isFile() || (stat.mode & 0o111) === 0 || relative === "" || relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative)) {
    throw new Error("Selected Intel 2026.2 compiler must resolve to an executable inside its compiler root");
  }
  return compiler;
}

/** @param {string} file @param {string} expected @param {number} maximum @param {string} label */
function stableFile(file, expected, maximum, label) {
  const before = fs.lstatSync(file, {bigint: true});
  if (!path.isAbsolute(file) || /[\r\n\0]/.test(file) || !before.isFile() || before.isSymbolicLink() || before.nlink !== 1n ||
      before.size < 1n || before.size > BigInt(maximum) || fs.realpathSync(file) !== path.resolve(file)) {
    throw new Error(label + " must be a bounded canonical singly-linked regular file");
  }
  const sha256 = fileHash(file), after = fs.lstatSync(file, {bigint: true});
  if (expected && (!/^[a-f0-9]{64}$/.test(expected) || sha256 !== expected) || before.dev !== after.dev || before.ino !== after.ino ||
      before.nlink !== after.nlink || before.size !== after.size || before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs) {
    throw new Error(label + " changed or differs from its external SHA-256 pin");
  }
  return {sha256, bytes: Number(after.size)};
}

/** @param {string} directory @param {string} label */
function canonicalDirectory(directory, label) {
  const stat = fs.lstatSync(directory);
  if (!path.isAbsolute(directory) || /[\r\n\0]/.test(directory) || !stat.isDirectory() || stat.isSymbolicLink() ||
      fs.realpathSync(directory) !== path.resolve(directory)) {throw new Error(label + " must be a canonical real directory");}
  return directory;
}

/** @param {string} root @param {string} file */
function within(root, file) {
  const relative = path.relative(root, file);
  return relative === "" || relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative);
}

/** The same-source TableGen tools are distinct from the thin LLVM21 link/RC utilities.
 * @param {any} nativeTools */
function windowsJitNativeToolInputs(nativeTools) {
  const toolRoot = nativeTools?.toolRoot, manifestFile = nativeTools?.manifestFile, files = nativeTools?.files,
    hashes = nativeTools?.hashes, source = nativeTools?.source;
  if (typeof toolRoot !== "string" || !path.isAbsolute(toolRoot) || path.resolve(toolRoot) !== toolRoot ||
      path.basename(toolRoot) !== "native-tools" ||
      typeof manifestFile !== "string" || !path.isAbsolute(manifestFile) || path.resolve(manifestFile) !== manifestFile || !within(toolRoot, manifestFile) ||
      !files || Object.keys(files).sort().join(",") !== "clangTableGen,llvmTableGen" ||
      !hashes || Object.keys(hashes).sort().join(",") !== "clangTableGenSha256,llvmTableGenSha256" ||
      files.clangTableGen !== path.join(toolRoot, "bin", "clang-tblgen") || files.llvmTableGen !== path.join(toolRoot, "bin", "llvm-tblgen") ||
      typeof source?.receiptFile !== "string" || !path.isAbsolute(source.receiptFile) || path.resolve(source.receiptFile) !== source.receiptFile ||
      typeof source?.commit !== "string" || !source.commit || source.commit.length > 128 || /[\r\n\0]/.test(source.commit)) {
    throw new Error("Windows JIT native TableGen admission is incomplete");
  }
  const rows = [
    {file: manifestFile, sha256: nativeTools.manifestSha256, label: "native TableGen manifest"},
    {file: source.receiptFile, sha256: source.receiptSha256, label: "native TableGen source receipt"},
    {file: files.clangTableGen, sha256: hashes.clangTableGenSha256, label: "clang-tblgen"},
    {file: files.llvmTableGen, sha256: hashes.llvmTableGenSha256, label: "llvm-tblgen"},
  ];
  if (rows.some(row => !/^[a-f0-9]{64}$/.test(row.sha256 || "") || /[\r\n\0]/.test(row.file)) ||
      new Set(rows.map(row => row.file)).size !== rows.length || !/^[a-f0-9]{40,64}$/.test(source.commit)) {
    throw new Error("Windows JIT native TableGen inputs are not completely pinned");
  }
  return rows;
}

/** Check the partial LLVM descriptor against the independently admitted host SDK, LLVM tools,
 * and Hermes Intel-runtime handoff. The descriptor intentionally appends the Intel lib root.
 * @param {ReturnType<typeof import("./prepare-native-jit-toolchain").validateWindowsJitDescriptor>} descriptor
 * @param {{sdkRoot:string,sdk:{sourceRoot:string,overlay:string,includeRoots:string[],libraryRoots:string[]},
 * tools:{lldLink:string,llvmNm:string},intelRuntime:{libRoot:string,libraries:Record<string,string>}}} expected */
function validateWindowsJitDescriptorInputs(descriptor, expected) {
  const {windowsSdk, tools} = descriptor, libraryRoots = [...expected.sdk.libraryRoots, expected.intelRuntime.libRoot],
    runtimeLibraries = Object.values(expected.intelRuntime.libraries);
  if (expected.sdkRoot !== expected.sdk.sourceRoot || windowsSdk.sourceRoot !== expected.sdk.sourceRoot ||
      windowsSdk.overlay !== expected.sdk.overlay || !isDeepStrictEqual(windowsSdk.includeRoots, expected.sdk.includeRoots) ||
      !isDeepStrictEqual(windowsSdk.libraryRoots, libraryRoots) || tools.lldLink !== expected.tools.lldLink ||
      tools.llvmNm !== expected.tools.llvmNm || !runtimeLibraries.length || runtimeLibraries.some(file =>
    !within(expected.intelRuntime.libRoot, file) || !windowsSdk.msvcLibraries.includes(file))) {
    throw new Error("Windows JIT descriptor differs from the admitted SDK roots, LLVM tools, or Hermes Intel runtime");
  }
  return true;
}

/** @param {string} stage @param {string} configPath @param {string|undefined} configSha256
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {string} [architecture] @param {number} [nowMs] */
function windowsJitCiJobPlan(stage, configPath, configSha256, environment = process.env, platform = process.platform,
  architecture = process.arch, nowMs = Date.now()) {
  const validConfigPin = stage === "configure" ? configSha256 === undefined || configSha256 === "" : /^[a-f0-9]{64}$/.test(configSha256 || "");
  if (!phases.includes(stage) || platform !== "linux" || architecture !== "x64" || !validConfigPin ||
      environment["MOM_AOT_BUILD_ONLY"] !== "1" || environment["MOM_BUILD_JOBS"] !== "2") {
    throw new Error("Windows source-JIT CI needs a maintained phase, Linux x64, build-only mode, and two jobs");
  }
  const run = environment["GITHUB_RUN_ID"] || "", attempt = environment["GITHUB_RUN_ATTEMPT"] || "",
    baseJob = environment["GITHUB_JOB"] || "", jobId = baseJob + "_" + stage;
  if (!/^[0-9]{1,20}$/.test(run) || !/^[1-9][0-9]{0,5}$/.test(attempt) || !/^[A-Za-z0-9_-]{1,52}$/.test(baseJob) ||
      !/^[A-Za-z0-9_-]{1,64}$/.test(jobId)) {throw new Error("Windows source-JIT CI requires safe run, attempt, and phase-job identities");}
  const suffix = run + "-" + attempt, runnerTemp = environment["RUNNER_TEMP"] || "";
  canonicalDirectory(runnerTemp, "RUNNER_TEMP");
  const runDeadlineEpoch = environment["MOM_NATIVE_WINDOWS_JIT_RUN_DEADLINE_EPOCH"] ||
      (stage === "configure" ? String(Math.floor(nowMs / 1000) + 8 * 60 * 60) : ""),
    jobDeadlineEpoch = environment["MOM_NATIVE_WINDOWS_JIT_DEADLINE_EPOCH"] || "";
  ci.windowsNativeLlvmCiRunDeadlineBudget(runDeadlineEpoch, nowMs);
  const budget = ci.windowsNativeLlvmCiDeadlineBudget(jobDeadlineEpoch, stage === "configure" ? "configure" : "build", nowMs, runDeadlineEpoch);
  if (Number(jobDeadlineEpoch) > Number(runDeadlineEpoch) || budget.remainingMs > jobLimitMs) {
    throw new Error("Windows JIT phase deadline must remain within this run and its 55-minute job limit");
  }
  const rootName = "mom-native-windows-jit-" + suffix, root = path.join("/tmp", rootName), output = path.join(root, "sdk"),
    dpcppName = "mom-native-dpcpp-cuda-inputs-" + suffix,
    toolsName = "native-linux-llvm-tools-" + suffix, hermesName = "native-windows-controller-hermes-" + suffix,
    resourcesPaths = windowsJitResourcePaths(suffix), resourcesName = path.basename(resourcesPaths.root), artifactName = "native-windows-jit-sdk-" + suffix,
    resourcesArtifactRoot = resourcesPaths.root, resourcesRoot = resourcesPaths.resources,
    linux = ci.linuxSourceStatePaths({suffix, runnerTemp: fs.realpathSync(runnerTemp)}, environment),
    dpcpp = ci.dpcppCudaInputsStatePaths({suffix, runnerTemp: fs.realpathSync(runnerTemp)}, environment),
    sourceName = linux.sourceName,
    expectedConfig = path.join(fs.realpathSync(runnerTemp), "incoming/config", artifactName + ".json"),
    // Share the maintained Windows LLVM descriptor producer instead of inventing another one.
    descriptor = path.join(fs.realpathSync(runnerTemp), "mom-native-windows-llvm21-descriptor-" + suffix, "descriptor.json"),
    toolchain = path.join("/tmp", "mom-native-windows-jit-toolchain-" + suffix),
    windowsSdk = path.join(fs.realpathSync(runnerTemp), "mom-native-windows-host-sdk-" + suffix, "native-windows-host-sdk-" + suffix),
    statePhase = stage === "configure" ? "configured" : stage, index = phases.indexOf(stage),
    previousPhase = index === 0 ? "" : index === 1 ? "configured" : phases[index - 1],
    archive = path.join(fs.realpathSync(runnerTemp), artifactName + "-" + statePhase + ".tar"),
    incomingState = previousPhase ? path.join(fs.realpathSync(runnerTemp), "incoming/state", artifactName + "-" + previousPhase + ".tar") : "";
  if (path.resolve(configPath) !== expectedConfig || environment["MOM_NATIVE_WINDOWS_JIT_DIR"] !== root ||
      environment["MOM_NATIVE_WINDOWS_JIT_RESOURCES_DIR"] !== path.join("/tmp", resourcesName) ||
      environment["MOM_NATIVE_WINDOWS_JIT_DESCRIPTOR_FILE"] !== descriptor || environment["MOM_NATIVE_WINDOWS_JIT_WINDOWS_SDK_ROOT"] !== windowsSdk ||
      stage === "configure" && (environment["MOM_NATIVE_WINDOWS_JIT_STATE_PREVIOUS_PHASE"] || environment["MOM_NATIVE_WINDOWS_JIT_STATE_BUNDLE_SHA256"]) ||
      stage !== "configure" && (environment["MOM_NATIVE_WINDOWS_JIT_STATE_PREVIOUS_PHASE"] !== previousPhase ||
        !/^[a-f0-9]{64}$/.test(environment["MOM_NATIVE_WINDOWS_JIT_STATE_BUNDLE_SHA256"] || ""))) {
    throw new Error("Windows source-JIT paths or immediately preceding state do not match this run");
  }
  const pins = {
    config: configSha256 || "",
    source: environment["MOM_NATIVE_WINDOWS_JIT_SOURCE_BUNDLE_SHA256"] || "",
    dpcpp: environment["MOM_NATIVE_WINDOWS_JIT_DPCPP_BUNDLE_SHA256"] || "",
    tools: environment["MOM_NATIVE_LINUX_LLVM_TOOLS_BUNDLE_SHA256"] || "",
    toolsHandoff: environment["MOM_NATIVE_LINUX_LLVM_TOOLS_HANDOFF_SHA256"] || "",
    hermes: environment["MOM_NATIVE_WINDOWS_CONTROLLER_HERMES_BUNDLE_SHA256"] || "",
    hermesHandoff: environment["MOM_NATIVE_WINDOWS_CONTROLLER_HERMES_HANDOFF_SHA256"] || "",
    dpcppManifest: environment["MOM_NATIVE_WINDOWS_JIT_DPCPP_MANIFEST_SHA256"] || "",
    resources: environment["MOM_NATIVE_WINDOWS_JIT_RESOURCES_BUNDLE_SHA256"] || "",
    descriptor: environment["MOM_NATIVE_WINDOWS_JIT_DESCRIPTOR_SHA256"] || "",
    windowsSdk: ["RESULT", "DESCRIPTOR", "FILES", "OVERLAY"].map(key => environment[`MOM_NATIVE_WINDOWS_JIT_WINDOWS_SDK_${key}_SHA256`] || ""),
    state: environment["MOM_NATIVE_WINDOWS_JIT_STATE_BUNDLE_SHA256"] || "",
  };
  if ([pins.source, pins.dpcpp, pins.dpcppManifest, pins.tools, pins.toolsHandoff, pins.hermes, pins.hermesHandoff, pins.resources, pins.descriptor,
    ...pins.windowsSdk, ...(previousPhase ? [pins.state] : [])].some(hash => !/^[a-f0-9]{64}$/.test(hash))) {
    throw new Error("Windows source-JIT CI is missing an external source, SDK, resource or state pin");
  }
  const outputFiles = [environment["GITHUB_OUTPUT"] || "", environment["GITHUB_STEP_SUMMARY"] || ""],
    envFile = stage === "configure" && !environment["MOM_NATIVE_WINDOWS_JIT_RUN_DEADLINE_EPOCH"] ? environment["GITHUB_ENV"] || "" : "",
    reportFiles = [...outputFiles, ...(envFile ? [envFile] : [])];
  if (reportFiles.some(file => !path.isAbsolute(file) || path.resolve(file) === path.resolve(expectedConfig)) ||
      new Set(reportFiles.map(file => path.resolve(file))).size !== reportFiles.length) {
    throw new Error("Windows source-JIT workflow report files must be distinct absolute paths");
  }
  return {stage, suffix, baseJob, jobId, rootName, root, output, artifactName, archive, incomingState, previousPhase: previousPhase || null,
    runnerTemp: fs.realpathSync(runnerTemp), config: expectedConfig, descriptor, toolchain, windowsSdk, linux, dpcpp,
    resourcesArtifactRoot, resourcesRoot,
    toolsArchive: path.join(fs.realpathSync(runnerTemp), "incoming/tools", toolsName + ".tar"), toolsName,
    hermesArchive: path.join(fs.realpathSync(runnerTemp), "incoming/hermes", hermesName + ".tar"), hermesName,
    resourcesArchive: path.join(fs.realpathSync(runnerTemp), "incoming/resources", resourcesName + ".tar"), resourcesName,
    sourceArchive: linux.incomingTar, sourceName, dpcppArchive: path.join(fs.realpathSync(runnerTemp), "incoming", path.basename(dpcpp.sdkTar)), dpcppName,
    pins, stateBundleSha256: pins.state, runDeadlineEpoch, jobDeadlineEpoch, budget, reportFiles,
    environment: {...environment, GITHUB_JOB: jobId, MOM_NATIVE_WINDOWS_JIT_RUN_DEADLINE_EPOCH: runDeadlineEpoch}};
}

/** Pure checks for the original build-native-jit-sdk.js configuration. @param {unknown} value
 * @param {{sources:string,sdk:string,resources:string,toolchain:string,compiler:string,requiredInputs:string[],allowedRoots:string[]}} expected */
function validateWindowsJitConfig(value, expected) {
  const config = jitSdkConfig(value);
  if (!Array.isArray(config.inputs)) {throw new Error("Windows JIT configuration must bind its exact input files");}
  const inputs = config.inputs;
  if (config.target !== "windows" || config.jobs !== 2 || config.sources !== expected.sources || config.sdk !== expected.sdk ||
      config.resources !== expected.resources || !path.isAbsolute(expected.compiler) || path.normalize(expected.compiler) !== expected.compiler ||
      config.tools.cc !== expected.compiler || config.tools.cxx !== expected.compiler ||
      config.tools.toolchain !== expected.toolchain || config.tools.cmake !== fs.realpathSync("/usr/bin/cmake") ||
      config.tools.ninja !== fs.realpathSync("/usr/bin/ninja") || inputs.length > 16384 ||
      new Set(inputs).size !== inputs.length || expected.requiredInputs.some(file => !inputs.includes(file))) {
    throw new Error("Windows JIT configuration differs from the exact admitted same-run compiler/toolchain plan");
  }
  for (const file of inputs) {
    if (path.normalize(file) !== file || !expected.allowedRoots.some(root => within(root, file))) {
      throw new Error("Windows JIT configuration contains a private, stale or unsupported input path");
    }
  }
  return config;
}

/** Derive the original builder config only from paths returned by admitted same-run inputs
 * and the deterministic toolchain producer. Every phase recomputes it to reject stale configs.
 * @param {ReturnType<typeof windowsJitCiJobPlan>} plan
 * @param {{linux: {compilerPreparedReceipt: string}, dpcpp: {dpcppRoot: string}, compiler: string,
 * nativeJitTools: any,
 * toolchainResult: {toolchain: string, inputs: Record<string,string>}}} admitted */
function composeWindowsJitConfig(plan, admitted) {
  const toolchain = path.join(plan.toolchain, "toolchain.cmake"), cmake = fs.realpathSync("/usr/bin/cmake"),
    ninja = fs.realpathSync("/usr/bin/ninja"), compiler = admitted.compiler,
    entries = Object.entries(admitted.toolchainResult.inputs || {}), toolchainInputs = entries.map(([file]) => file),
    nativeInputs = windowsJitNativeToolInputs(admitted.nativeJitTools);
  if (admitted.toolchainResult.toolchain !== toolchain ||
      !path.isAbsolute(admitted.linux.compilerPreparedReceipt) || !path.isAbsolute(admitted.dpcpp.dpcppRoot) ||
      admitted.nativeJitTools.source.receiptFile !== admitted.linux.compilerPreparedReceipt ||
      admitted.toolchainResult.inputs[admitted.nativeJitTools.files.clangTableGen] !== admitted.nativeJitTools.hashes.clangTableGenSha256 ||
      admitted.toolchainResult.inputs[admitted.nativeJitTools.files.llvmTableGen] !== admitted.nativeJitTools.hashes.llvmTableGenSha256 ||
      !path.isAbsolute(compiler) || toolchainInputs.length > 16384 ||
      entries.some(([file, sha256]) => !path.isAbsolute(file) || !/^[a-f0-9]{64}$/.test(sha256 || ""))) {
    throw new Error("Windows JIT config can only be composed from admitted same-run inputs");
  }
  return jitSdkConfig({version: 1, target: "windows", sources: admitted.linux.compilerPreparedReceipt,
    sdk: admitted.dpcpp.dpcppRoot, resources: plan.resourcesRoot,
    tools: {cc: compiler, cxx: compiler, cmake, ninja, toolchain}, jobs: 2,
    inputs: [...new Set([...toolchainInputs, ...nativeInputs.map(row => row.file), plan.descriptor, compiler, cmake, ninja])].sort()});
}

/** Create the config directory without traversing a pre-existing symlink. @param {string} file @param {string} runnerTemp */
function ensureConfigParent(file, runnerTemp) {
  const incoming = path.join(runnerTemp, "incoming"), directory = path.dirname(file);
  for (const selected of [incoming, directory]) {
    const stat = fs.lstatSync(selected, {throwIfNoEntry: false});
    if (!stat) {
      fs.mkdirSync(selected, {mode: 0o700});
    } else if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(selected) !== selected) {
      throw new Error("Windows JIT generated-config parent must be a canonical real directory");
    }
  }
  if (directory !== path.join(incoming, "config") || !within(runnerTemp, directory)) {
    throw new Error("Windows JIT generated config escaped its same-run incoming directory");
  }
}

/** Write once, then re-read and hash the exact generated config for upload and later phases.
 * @param {ReturnType<typeof windowsJitCiJobPlan>} plan @param {ReturnType<typeof jitSdkConfig>} config */
function writeWindowsJitConfig(plan, config) {
  if (plan.stage !== "configure" || fs.existsSync(plan.config)) {throw new Error("Windows JIT config output is not fresh");}
  ensureConfigParent(plan.config, plan.runnerTemp);
  const text = JSON.stringify(config, null, 2) + "\n";
  if (Buffer.byteLength(text) > maxConfigBytes) {throw new Error("Generated Windows JIT config exceeds bounds");}
  fs.writeFileSync(plan.config, text, {flag: "wx", mode: 0o600});
  const pin = stableFile(plan.config, "", maxConfigBytes, "Generated Windows JIT configuration"), value = readBuildJson(plan.config);
  if (!isDeepStrictEqual(value, config) || stableFile(plan.config, pin.sha256, maxConfigBytes, "Generated Windows JIT configuration").bytes !== pin.bytes) {
    throw new Error("Generated Windows JIT config changed while writing");
  }
  return {sha256: pin.sha256, bytes: pin.bytes, value};
}

/** @param {string} archive @param {string} rootName @param {string} sha256 @param {string} base
 * @param {number} timeoutMs @param {number} maxExpanded */
function extractPinnedRoot(archive, rootName, sha256, base, timeoutMs, maxExpanded) {
  stableFile(archive, sha256, maxArchiveBytes, "Windows JIT input archive");
  const members = ci.listState(archive, [rootName], Math.min(timeoutMs, 5 * 60 * 1000));
  if (!members.length || members.length > maxMembers) {throw new Error("Windows JIT input archive member count is outside bounds");}
  const expanded = ci.tarExpandedBytes(archive, members.length, Math.min(timeoutMs, 15 * 60 * 1000));
  if (expanded > maxExpanded) {throw new Error("Windows JIT input archive exceeds its expanded-size bound");}
  ci.linuxNativeSdkDiskPreflight(expanded, availableBytes(base), availableBytes(path.dirname(archive)));
  if (fs.existsSync(path.join(base, rootName))) {throw new Error("Windows JIT input root already exists: " + rootName);}
  ci.extractState(archive, base, [rootName], [rootName], Math.min(timeoutMs, 30 * 60 * 1000));
  stableFile(archive, sha256, maxArchiveBytes, "Windows JIT input archive");
  return path.join(base, rootName);
}

/** @param {string} directory */
function availableBytes(directory) {
  const stat = fs.statfsSync(directory, {bigint: true}), bytes = stat.bavail * stat.bsize;
  return bytes > BigInt(Number.MAX_SAFE_INTEGER) ? Number.MAX_SAFE_INTEGER : Number(bytes);
}

/** @param {ReturnType<typeof windowsJitCiJobPlan>} plan @param {number} capMs @param {number} reserveMs */
function remaining(plan, capMs, reserveMs = 60 * 1000) {
  const end = Math.min(Number(plan.jobDeadlineEpoch) * 1000, Number(plan.runDeadlineEpoch) * 1000), timeoutMs = Math.min(capMs, end - Date.now() - reserveMs);
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000) {throw new Error("Windows JIT phase exceeded the caller's job/run deadline");}
  return timeoutMs;
}

/** @param {ReturnType<typeof windowsJitCiJobPlan>} plan @param {ReturnType<typeof validateWindowsJitConfig>} [config] */
function admitInputs(plan, config) {
  const env = plan.environment, sourceRoot = extractPinnedRoot(plan.sourceArchive, plan.sourceName, plan.pins.source, "/tmp",
      remaining(plan, 15 * 60 * 1000), 8 * 1024 ** 3),
    dpcppRoot = extractPinnedRoot(plan.dpcppArchive, plan.dpcppName, plan.pins.dpcpp, "/tmp",
      remaining(plan, 15 * 60 * 1000), 16 * 1024 ** 3),
    toolsRoot = extractPinnedRoot(plan.toolsArchive, plan.toolsName, plan.pins.tools, plan.runnerTemp,
      remaining(plan, 8 * 60 * 1000), 512 * 1024 ** 2);
  if (!toolsRoot.startsWith(plan.runnerTemp + path.sep)) {throw new Error("Linux LLVM tools extracted outside RUNNER_TEMP");}
  const tools = linuxLlvmToolHandoffInputs(toolsRoot, {runAttempt: plan.suffix, handoffSha256: plan.pins.toolsHandoff}),
    hermesRoot = extractPinnedRoot(plan.hermesArchive, plan.hermesName, plan.pins.hermes, plan.runnerTemp,
      remaining(plan, 10 * 60 * 1000), 4 * 1024 ** 3),
    hermes = windowsControllerHandoffInputs(hermesRoot, {kind: "hermes", runAttempt: plan.suffix, handoffSha256: plan.pins.hermesHandoff}),
    resourcesArtifactRoot = extractPinnedRoot(plan.resourcesArchive, plan.resourcesName, plan.pins.resources, "/tmp",
      remaining(plan, 8 * 60 * 1000), 512 * 1024 ** 2),
    resourcesRoot = path.join(resourcesArtifactRoot, "resources"),
    resources = verifyJitResources(resourcesRoot, "x86_64-w64-windows-gnu"),
    nativeJitTools = admitWindowsJitNativeTools(resourcesArtifactRoot, plan.suffix);
  const {linux, dpcpp} = plan, git = fs.realpathSync("/usr/bin/git");
  if (linux.source !== sourceRoot || dpcpp.root !== dpcppRoot) {throw new Error("Restored source/DPC++ roots are not canonical for this run");}
  const sourceReceipts = ci.verifyLinuxSources({env}, linux, git); verifyPreparedSdkSources(linux.compilerPreparedReceipt, "compiler");
  if (nativeJitTools.source.receiptFile !== linux.compilerPreparedReceipt ||
      nativeJitTools.source.receiptSha256 !== fileHash(linux.compilerPreparedReceipt) ||
      sourceReceipts.compilerPreparedReceipt?.prepared?.dpcpp?.revision !== nativeJitTools.source.commit) {
    throw new Error("Windows JIT native TableGen tools differ from the admitted same-run DPC++ source receipt");
  }
  ci.auditDpcppCudaInputs(dpcpp, readBuildJson(dpcpp.manifest), plan.pins.dpcppManifest);
  const sdkPins = {resultSha256: plan.pins.windowsSdk[0] || "", descriptorSha256: plan.pins.windowsSdk[1] || "",
      filesSha256: plan.pins.windowsSdk[2] || "", overlaySha256: plan.pins.windowsSdk[3] || ""}, host = windowsHostSdkInputs(plan.windowsSdk, sdkPins),
    descriptorPin = stableFile(plan.descriptor, plan.pins.descriptor, 8 << 20, "Windows controller descriptor"),
    {prepareWindowsJitToolchain, validateWindowsJitDescriptor} = require("./prepare-native-jit-toolchain"),
    descriptor = validateWindowsJitDescriptor(readBuildJson(plan.descriptor));
  if (hermes.kind !== "hermes") {throw new Error("Windows JIT requires the current Hermes source handoff");}
  const hermesInputs = hermes, intelRuntime = hermesInputs.intelRuntime;
  if (!intelRuntime) {throw new Error("Windows JIT requires the current Hermes Intel runtime handoff");}
  validateWindowsJitDescriptorInputs(descriptor, {sdkRoot: host.sdkRoot, sdk: host.sdk, tools, intelRuntime});
  const compiler = canonicalIntelCompiler();
  const version = spawnSync(compiler, ["--version"], {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
  if (version.error || version.signal || version.status !== 0 || !/2026\.2/.test(version.stdout || "")) {
    throw new Error("Selected Windows JIT host compiler is not Intel 2026.2", {cause: version.error});
  }
  if (descriptor.tools.clangResourceInclude !==
      "/opt/intel/oneapi/compiler/2026.2/lib/clang/23/include") {
    throw new Error("Windows JIT host compiler/resource headers differ from the pinned Intel 2026.2 runtime");
  }
  const nativeInputs = windowsJitNativeToolInputs(nativeJitTools);
  for (const row of nativeInputs) {
    stableFile(row.file, row.sha256, row.label.includes("tblgen") ? 512 * 1024 ** 2 : 8 << 20, row.label);
  }
  const toolchainResult = prepareWindowsJitToolchain(plan.descriptor, dpcpp.dpcppRoot, path.join(nativeJitTools.toolRoot, "bin"), plan.toolchain,
      {...env, MOM_AOT_BUILD_ONLY: "1"}, {cc: compiler, cxx: compiler}), toolchainInputs = Object.keys(toolchainResult.inputs);
  if (toolchainResult.inputs[nativeJitTools.files.clangTableGen] !== nativeJitTools.hashes.clangTableGenSha256 ||
      toolchainResult.inputs[nativeJitTools.files.llvmTableGen] !== nativeJitTools.hashes.llvmTableGenSha256) {
    throw new Error("Windows JIT toolchain selected different native TableGen binaries");
  }
  const allowedRoots = [path.resolve(__dirname, ".."), sourceRoot, dpcppRoot, resourcesRoot, resourcesArtifactRoot,
    tools.toolRoot, hermesRoot, nativeJitTools.toolRoot,
    plan.windowsSdk, plan.descriptor.slice(0, plan.descriptor.lastIndexOf(path.sep)), plan.toolchain,
    path.dirname(descriptor.tools.lldLink), "/opt/intel/oneapi/compiler/2026.2", "/usr/bin"];
  const expected = {sources: linux.compilerPreparedReceipt, sdk: dpcpp.dpcppRoot,
      resources: resourcesRoot, toolchain: path.join(plan.toolchain, "toolchain.cmake"), compiler,
      requiredInputs: [plan.descriptor, ...toolchainInputs, ...nativeInputs.map(row => row.file)], allowedRoots}, generatedConfig = composeWindowsJitConfig(plan,
      {linux, dpcpp, compiler, nativeJitTools, toolchainResult}),
    expectedConfig = config === undefined ? generatedConfig : validateWindowsJitConfig(config, expected);
  if (!isDeepStrictEqual(expectedConfig, generatedConfig)) {throw new Error("Windows JIT config is not the exact config derived from admitted inputs");}
  if (!resources.inputs[path.join(resourcesRoot, "manifest.json")] || descriptorPin.sha256 !== plan.pins.descriptor ||
      fileHash(plan.descriptor) !== descriptorPin.sha256 || !fs.existsSync(tools.sourceArchive) || !fs.existsSync(dpcpp.dpcppRoot)) {
    throw new Error("Windows JIT current-run source/tool/resource admission is incomplete");
  }
  return {config: expectedConfig, linux, dpcpp, tools, hermes, resources, resourcesArtifactRoot, resourcesRoot, nativeJitTools, host, toolchainResult};
}

/** @param {ReturnType<typeof windowsJitCiJobPlan>} plan @param {string[]} args */
function invokeBuilder(plan, args) {
  const timeout = fs.realpathSync("/usr/bin/timeout"), timeoutPin = fileHash(timeout), timeoutMs = remaining(plan, childLimitMs, 8 * 60 * 1000),
    result = spawnSync(timeout, ["--signal=TERM", "--kill-after=5s", Math.ceil(timeoutMs / 1000) + "s", process.execPath,
      path.join(__dirname, "build-native-jit-sdk.js"), plan.config, plan.output, ...args],
    {cwd: path.resolve(__dirname, ".."), env: {...plan.environment, MOM_AOT_BUILD_ONLY: "1"}, stdio: "inherit", timeout: timeoutMs + 10000});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error("Original Windows source-JIT builder phase failed: " + plan.stage, {cause: result.error});
  }
  if (fileHash(timeout) !== timeoutPin) {throw new Error("GNU timeout changed during the Windows JIT phase");}
}

/** @param {string} file @param {string} stage */
function phaseReceipt(file, stage) {
  const before = stableFile(file, "", 64 << 20, "Original Windows JIT phase receipt");
  const receipt = readBuildJson(file);
  if (receipt.stage !== stage || receipt.buildOnly !== true || !receipt.outputs || typeof receipt.outputs !== "object" ||
      stableFile(file, before.sha256, 64 << 20, "Original Windows JIT phase receipt").bytes !== before.bytes) {
    throw new Error("Original Windows JIT phase receipt does not bind the selected phase");
  }
  return receipt;
}

/** @param {string} file */
function freshWorkflowFile(file) {
  const stat = fs.lstatSync(file);
  if (!path.isAbsolute(file) || !stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size !== 0 || fs.realpathSync(file) !== file) {
    throw new Error("GitHub workflow outputs must be fresh canonical regular files");
  }
  return stat;
}

/** @param {string} stage @param {string} configPath @param {string|undefined} configSha256 @param {NodeJS.ProcessEnv} [environment] */
function runWindowsJitCiJob(stage, configPath, configSha256, environment = process.env) {
  const plan = windowsJitCiJobPlan(stage, configPath, configSha256, environment);
  let configPin, configValue;
  if (stage === "configure") {
    if (fs.existsSync(plan.config)) {throw new Error("Windows JIT generated config must be a fresh output");}
  } else {
    configPin = stableFile(plan.config, plan.pins.config, maxConfigBytes, "Original Windows JIT configuration");
    configValue = readBuildJson(plan.config);
    stableFile(plan.config, configPin.sha256, maxConfigBytes, "Original Windows JIT configuration");
  }
  if (fs.existsSync(plan.archive) || stage === "configure" && fs.existsSync(plan.root)) {throw new Error("Windows JIT output state is not fresh");}
  const outputStat = freshWorkflowFile(environment["GITHUB_OUTPUT"] || ""), summaryStat = freshWorkflowFile(environment["GITHUB_STEP_SUMMARY"] || ""),
    reportIds = [outputStat, summaryStat].map(stat => `${stat.dev}:${stat.ino}`);
  let envFile = "";
  if (stage === "configure" && !environment["MOM_NATIVE_WINDOWS_JIT_RUN_DEADLINE_EPOCH"]) {
    envFile = environment["GITHUB_ENV"] || "";
    const stat = freshWorkflowFile(envFile); reportIds.push(`${stat.dev}:${stat.ino}`);
  }
  if (new Set(reportIds).size !== reportIds.length) {throw new Error("Windows JIT workflow output files alias one another");}
  if (stage !== "configure") {
    stableFile(plan.incomingState, plan.stateBundleSha256, maxArchiveBytes, "Immediately preceding Windows JIT state archive");
    if (fs.existsSync(plan.root)) {throw new Error("Windows JIT state restore root already exists");}
    extractPinnedRoot(plan.incomingState, plan.rootName, plan.stateBundleSha256, "/tmp", remaining(plan, 25 * 60 * 1000), maxExpandedBytes);
    const before = readBuildJson(path.join(plan.output, "before.json")), prior = plan.previousPhase === "configured" ? "configure" : plan.previousPhase;
    if (JSON.stringify(before.config) !== JSON.stringify(configValue) || !fs.existsSync(path.join(plan.output, `stage-${prior}.json`))) {
      throw new Error("Windows JIT state does not retain the exact config and immediately preceding phase");
    }
  } else {
    fs.mkdirSync(plan.root, {recursive: false, mode: 0o700});
  }
  const runRecord = path.join(plan.root, "ci-run-deadline.json"), jobRecord = path.join(plan.root, `ci-job-deadline-${plan.jobId}.json`);
  if (stage === "configure") {
    fs.writeFileSync(runRecord, JSON.stringify({version: 1, runAttempt: plan.suffix, runDeadlineEpoch: plan.runDeadlineEpoch}) + "\n", {flag: "wx", mode: 0o444});
  } else {
    const saved = readBuildJson(runRecord);
    if (saved.version !== 1 || saved.runAttempt !== plan.suffix || saved.runDeadlineEpoch !== plan.runDeadlineEpoch) {
      throw new Error("Windows JIT phase attempted to replace the original immutable run deadline");
    }
  }
  fs.writeFileSync(jobRecord, JSON.stringify({version: 1, runAttempt: plan.suffix, job: plan.jobId,
    deadlineEpoch: plan.jobDeadlineEpoch, runDeadlineEpoch: plan.runDeadlineEpoch}) + "\n", {flag: "wx", mode: 0o444});
  const admitted = admitInputs(plan, configValue);
  if (stage === "configure") {
    const generated = writeWindowsJitConfig(plan, admitted.config);
    configPin = stableFile(plan.config, generated.sha256, maxConfigBytes, "Generated Windows JIT configuration");
  }
  if (!configPin) {throw new Error("Windows JIT config was not pinned before running the original builder");}
  ci.windowsNativeLlvmCiDeadlineBudget(plan.jobDeadlineEpoch, stage === "configure" ? "configure" : "build", Date.now(), plan.runDeadlineEpoch);
  if (envFile) {fs.appendFileSync(envFile, `MOM_NATIVE_WINDOWS_JIT_RUN_DEADLINE_EPOCH=${plan.runDeadlineEpoch}\n`, "utf8");}
  invokeBuilder(plan, [stage]);
  const before = readBuildJson(path.join(plan.output, "before.json"));
  if (JSON.stringify(before.config) !== JSON.stringify(admitted.config)) {throw new Error("Original JIT builder changed the externally pinned config");}
  const receiptPath = path.join(plan.output, `stage-${stage}.json`), receipt = phaseReceipt(receiptPath, stage),
    packed = ci.packState(plan.archive, "/tmp", [plan.rootName], remaining(plan, 10 * 60 * 1000)),
    archived = stableFile(plan.archive, "", maxArchiveBytes, "Packed Windows JIT build state");
  if (packed.bytes !== archived.bytes || archived.bytes > maxArchiveBytes || !Number.isSafeInteger(archived.bytes)) {
    throw new Error("Windows JIT phase handoff exceeds its archive size bound");
  }
  const values = {windows_jit_phase: stage === "configure" ? "configured" : stage,
    windows_jit_state_bundle_sha256: archived.sha256, windows_jit_config_sha256: configPin.sha256,
    windows_jit_build_only: "true", windows_jit_executed: "false", windows_jit_publication_qualified: "false"};
  ci.windowsNativeLlvmCiDeadlineBudget(plan.jobDeadlineEpoch, "outputs", Date.now(), plan.runDeadlineEpoch);
  const outputFile = environment["GITHUB_OUTPUT"] || "", summaryFile = environment["GITHUB_STEP_SUMMARY"] || "";
  fs.appendFileSync(outputFile, Object.entries(values).map(([key, value]) => `${key}=${value}`).join("\n") + "\n", "utf8");
  fs.appendFileSync(summaryFile, `### Windows source JIT ${stage}\n- Original builder phase receipt SHA-256: ${fileHash(receiptPath)}.\n- Build-only; not executed or publication-qualified.\n`, "utf8");
  return {...values, receiptSha256: fileHash(receiptPath), stateBytes: archived.bytes, phase: receipt.stage};
}

module.exports = {windowsJitCiJobPlan, validateWindowsJitConfig, composeWindowsJitConfig, writeWindowsJitConfig,
  validateWindowsJitDescriptorInputs, runWindowsJitCiJob};

if (require.main === module) {
  try {
    const [stage, configPath, configSha256, extra] = process.argv.slice(2);
    if (!stage || !configPath || extra || stage === "configure" && configSha256 !== undefined || stage !== "configure" && !configSha256) {
      throw new Error("Usage: run-native-windows-jit-ci.js configure <new-same-run-config.json> | support|targets|frontend|link <same-run-config.json> <external-config-sha256>");
    }
    const result = runWindowsJitCiJob(stage, configPath, configSha256);
    process.stdout.write(JSON.stringify({status: "PASS", stage: result.phase, stateBundleSha256: result.windows_jit_state_bundle_sha256,
      buildOnly: true, executed: false, publicationQualified: false}) + "\n");
  } catch (error) {
    process.stderr.write("[native-ci] Windows JIT phase failed: " + (error instanceof Error ? error.message : String(error)) + "\n");
    process.exitCode = 1;
  }
}
