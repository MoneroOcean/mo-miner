"use strict";

// One bounded same-run job at a time for the public Windows LLVM 21 builder.
// The builder's original receipts remain authoritative; this file only restores
// their exact state, re-admits current external inputs, invokes one phase or maintained group, and repacks.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util"),
  {createHash} = require("node:crypto");
const ci = require("./prepare-native-ci"), {fileHash} = require("./build-source-inputs");
const stateTransport = require("./native-windows-llvm-state");
const {archiveName, archiveSha256} = require("./native-llvm-sdk");
const {linuxLlvmToolHandoffInputs} = require("./native-linux-llvm-tools");
const {windowsControllerHandoffInputs} = require("./native-windows-controller-handoff");
const {buildPhases} = require("./build-native-windows-llvm21-sdk");

/** @type {Readonly<Record<string, {previousStage:string,outputPhase:string,phases:readonly string[]}>>} */
const groupedStages = Object.freeze({"llvm-mid": Object.freeze({previousStage: "llvm-amdgpu", outputPhase: "llvm-core",
  phases: Object.freeze(["llvm-nvptx", "llvm-x86", "llvm-core"])})});
const stages = Object.freeze(["configure", "llvm-amdgpu", ...Object.keys(groupedStages), "clang", "lld", "audit"]),
  buildPhaseOutputValues = new Set(buildPhases.map(row => "build-" + row.name));
const maxHermesArchiveBytes = 2 * 1024 ** 3, maxToolsArchiveBytes = 1 * 1024 ** 3,
  maxManifestBytes = 1 << 20, sdkComposeWindowMs = 25 * 60 * 1000;

/** @param {string} stage @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform]
 * @param {string} [architecture] @param {number} [nowMs] */
function windowsLlvmCiJobPlan(stage, environment = process.env, platform = process.platform, architecture = process.arch, nowMs = Date.now()) {
  if (!stages.includes(stage) || !Number.isSafeInteger(nowMs) || nowMs < 0) {throw new Error("Windows LLVM CI stage is not in the maintained phase order");}
  const run = environment["GITHUB_RUN_ID"] || "", attempt = environment["GITHUB_RUN_ATTEMPT"] || "",
    baseJob = environment["GITHUB_JOB"] || "", jobId = baseJob + "_" + stage;
  if (!/^[0-9]{1,20}$/.test(run) || !/^[1-9][0-9]{0,5}$/.test(attempt) ||
      !/^[A-Za-z0-9_-]{1,64}$/.test(baseJob) || !/^[A-Za-z0-9_-]{1,64}$/.test(jobId)) {
    throw new Error("Windows LLVM CI needs a bounded run, attempt, and stage-specific workflow job identity");
  }
  const group = groupedStages[stage], suffix = run + "-" + attempt,
    statePhase = stage === "configure" ? "configured" : stage === "audit" ? "audited" : group?.outputPhase || stage,
    previousStage = stage === "configure" ? null : stage === "audit" ? "lld" : group?.previousStage || buildPhases[buildPhases.findIndex(row => row.name === stage) - 1]?.name || "configured",
    previousProducerStage = stage === "clang" ? "llvm-mid" : previousStage,
    runDeadlineEpoch = environment["MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH"] ||
      (stage === "configure" ? String(Math.floor(nowMs / 1000) + 8 * 60 * 60) : ""),
    jobDeadlineEpoch = environment["MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH"] || "";
  if (!/^[0-9]{10}$/.test(runDeadlineEpoch) || !/^[0-9]{10}$/.test(jobDeadlineEpoch)) {
    throw new Error("Windows LLVM CI requires the original run deadline and a job-start deadline");
  }
  const stageEnvironment = /** @type {NodeJS.ProcessEnv} */ ({...environment,
    MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: runDeadlineEpoch, GITHUB_JOB: jobId});
  const state = ci.windowsNativeLlvmCiStatePaths(stageEnvironment, platform, architecture),
    descriptorState = ci.windowsNativeLlvmDescriptorCiStatePaths(stageEnvironment, platform, architecture),
    runnerTemp = state.runnerTemp, incomingToolsArchive = path.join(runnerTemp, "incoming/tools", `native-linux-llvm-tools-${suffix}.tar`),
    incomingHermesArchive = path.join(runnerTemp, "incoming/hermes", `native-windows-controller-hermes-${suffix}.tar`),
    incomingStateArchive = previousStage ? path.join(runnerTemp, "incoming/state", `native-windows-llvm-state-${suffix}-${previousStage}.tar`) : null,
    incomingStateManifest = previousStage ? path.join(runnerTemp, "incoming/state", `native-windows-llvm-state-${suffix}-${previousStage}.json`) : null,
    archive = path.join(runnerTemp, `native-windows-llvm-state-${suffix}-${statePhase}.tar`),
    manifest = path.join(runnerTemp, `native-windows-llvm-state-${suffix}-${statePhase}.json`),
    toolsBundleSha256 = environment["MOM_NATIVE_LINUX_LLVM_TOOLS_BUNDLE_SHA256"] || "",
    toolsHandoffSha256 = environment["MOM_NATIVE_LINUX_LLVM_TOOLS_HANDOFF_SHA256"] || "",
    hermesBundleSha256 = environment["MOM_NATIVE_WINDOWS_LLVM_HERMES_BUNDLE_SHA256"] || "",
    hermesHandoffSha256 = environment["MOM_NATIVE_WINDOWS_LLVM_HERMES_HANDOFF_SHA256"] || "",
    stateBundleSha256 = environment["MOM_NATIVE_WINDOWS_LLVM_STATE_BUNDLE_SHA256"] || "",
    stateManifestSha256 = environment["MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256"] || "",
    suppliedPreviousStage = environment["MOM_NATIVE_WINDOWS_LLVM_STATE_PREVIOUS_STAGE"] || "";
  if (stage !== "configure" && suppliedPreviousStage !== previousStage || stage === "configure" && suppliedPreviousStage) {
    throw new Error("Windows LLVM state input does not name the immediately preceding maintained stage");
  }
  const hashPins = [toolsBundleSha256, toolsHandoffSha256, hermesBundleSha256, hermesHandoffSha256,
    ...(previousStage ? [stateBundleSha256, stateManifestSha256] : [])];
  if (hashPins.some(value => !/^[a-f0-9]{64}$/.test(value))) {throw new Error("Windows LLVM CI is missing an external artifact SHA-256 pin");}
  const budget = ci.windowsNativeLlvmCiDeadlineBudget(jobDeadlineEpoch, stage === "configure" ? "configure" : stage === "audit" ? "audit" : "build",
    nowMs, runDeadlineEpoch);
  return {stage, builderPhases: group?.phases || (buildPhases.some(row => row.name === stage) ? [stage] : []),
    statePhase, previousStage, previousProducerStage, suffix,
    workflowJob: baseJob, jobId, runnerTemp, stateRoot: state.root, stateOutput: state.output, descriptorState,
    incomingToolsArchive, incomingHermesArchive, incomingStateArchive, incomingStateManifest, archive, manifest,
    toolsBundleSha256, toolsHandoffSha256, hermesBundleSha256, hermesHandoffSha256, stateBundleSha256, stateManifestSha256,
    runDeadlineEpoch, jobDeadlineEpoch, budget, environment: stageEnvironment};
}

/** Validate all same-run inputs and fresh outputs without extracting or changing state.
 * @param {string} stage @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform]
 * @param {string} [architecture] @param {number} [nowMs] */
function windowsLlvmCiJobPreflight(stage, environment = process.env, platform = process.platform, architecture = process.arch, nowMs = Date.now()) {
  const plan = windowsLlvmCiJobPlan(stage, environment, platform, architecture, nowMs), {environment: env} = plan;
  if (env["MOM_AOT_BUILD_ONLY"] !== "1" || env["MOM_BUILD_JOBS"] !== "2" ||
      env["MOM_NATIVE_WINDOWS_LLVM_DIR"] !== plan.stateRoot ||
      env["MOM_NATIVE_WINDOWS_LLVM_DESCRIPTOR_DIR"] !== plan.descriptorState.root) {
    throw new Error("Windows LLVM CI requires the fixed build-only, two-job run-attempt paths");
  }
  const outputFiles = [env["GITHUB_OUTPUT"] || "", env["GITHUB_STEP_SUMMARY"] || ""],
    reportFiles = [...outputFiles, ...(plan.stage === "configure" ? [env["GITHUB_ENV"] || ""] : [])];
  if (reportFiles.some(file => !path.isAbsolute(file)) || new Set(reportFiles.map(file => path.resolve(file))).size !== reportFiles.length) {
    throw new Error("Windows LLVM CI requires distinct canonical workflow output files");
  }
  const reportStats = reportFiles.map(file => stableRegular(file, 64 << 20, "Windows LLVM workflow output", "", true));
  const runnerTemp = stableDirectory(plan.runnerTemp, "Windows LLVM runner temp");
  if (runnerTemp !== plan.runnerTemp || path.dirname(plan.stateRoot) !== "/tmp" || fs.realpathSync("/tmp") !== "/tmp") {
    throw new Error("Windows LLVM runner and state roots are not canonical");
  }
  if (fs.existsSync(plan.stateRoot)) {
    throw new Error("Windows LLVM stage requires a fresh local state root before artifact restore");
  }
  if ([plan.archive, plan.manifest, plan.descriptorState.root].some(file => fs.existsSync(file))) {
    throw new Error("Windows LLVM stage output paths must be new");
  }
  const toolsArchive = stableRegular(plan.incomingToolsArchive, maxToolsArchiveBytes, "Thin Linux LLVM tool archive", plan.toolsBundleSha256),
    hermesArchive = stableRegular(plan.incomingHermesArchive, maxHermesArchiveBytes, "Windows Hermes source archive", plan.hermesBundleSha256);
  if (hermesArchive.bytes > maxHermesArchiveBytes) {
    throw new Error("Windows LLVM source handoff exceeds the bounded archive size");
  }
  let stateArchive = null, manifestValue = null;
  if (plan.previousStage) {
    stateArchive = stableRegular(plan.incomingStateArchive || "", 16 * 1024 ** 3, "Prior Windows LLVM state archive", plan.stateBundleSha256);
    const manifestFile = stableRegular(plan.incomingStateManifest || "", maxManifestBytes, "Prior Windows LLVM state manifest", plan.stateManifestSha256);
    manifestValue = JSON.parse(fs.readFileSync(plan.incomingStateManifest || "", "utf8"));
    const priorJobStage = plan.previousProducerStage === "configured" ? "configure" : plan.previousProducerStage;
    if (manifestFile.bytes > maxManifestBytes || manifestValue.runAttempt !== plan.suffix ||
        manifestValue.archive?.sha256 !== plan.stateBundleSha256 || manifestValue.producerJob !== `${plan.workflowJob}_${priorJobStage}`) {
      throw new Error("Prior Windows LLVM manifest does not bind the expected immediately preceding stage");
    }
  }
  const reportIds = reportFiles.map(file => {const stat = fs.lstatSync(file); return `${stat.dev}:${stat.ino}`;});
  if (new Set(reportIds).size !== reportIds.length) {throw new Error("Windows LLVM workflow output files alias one another");}
  const inputFiles = /** @type {string[]} */ ([plan.incomingToolsArchive, plan.incomingHermesArchive,
    plan.incomingStateArchive, plan.incomingStateManifest].filter(Boolean));
  if ([plan.archive, plan.manifest].some(file => outputFiles.includes(file)) ||
      inputFiles.some(file => outputFiles.includes(file) || reportIds.includes(`${fs.statSync(file).dev}:${fs.statSync(file).ino}`))) {
    throw new Error("Windows LLVM workflow reports or outputs must not alias pinned artifacts");
  }
  if (reportFiles.some((file, index) => file !== env["GITHUB_ENV"] && reportStats[index]?.bytes !== 0)) {
    throw new Error("Windows LLVM workflow output files must start empty");
  }
  ci.windowsNativeLlvmCiDeadlineBudget(plan.jobDeadlineEpoch, stage === "configure" ? "configure" : stage === "audit" ? "audit" : "build",
    nowMs, plan.runDeadlineEpoch);
  return {...plan, toolsArchive, hermesArchive, stateArchive, manifestValue};
}

/** @param {string} file @param {string} expectedSha256 @param {string} label @param {number} maximum */
function stableRegular(file, maximum, label, expectedSha256 = "", allowEmpty = false) {
  const before = fs.lstatSync(file, {bigint: true});
  if (!path.isAbsolute(file) || /[\r\n\0]/.test(file) || !before.isFile() || before.isSymbolicLink() || before.nlink !== 1n ||
      before.size < (allowEmpty ? 0n : 1n) || before.size > BigInt(maximum) || path.resolve(fs.realpathSync(file)) !== path.resolve(file)) {
    throw new Error(label + " must be a canonical singly-linked file within its bound");
  }
  const sha256 = fileHash(file), after = fs.lstatSync(file, {bigint: true});
  if (before.dev !== after.dev || before.ino !== after.ino || before.nlink !== after.nlink || before.size !== after.size ||
      before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs || expectedSha256 && sha256 !== expectedSha256) {
    throw new Error(label + " changed or differs from its external SHA-256 pin");
  }
  return {bytes: Number(after.size), sha256};
}

/** @param {string} directory @param {string} label */
function stableDirectory(directory, label) {
  const stat = fs.lstatSync(directory);
  if (!path.isAbsolute(directory) || !stat.isDirectory() || stat.isSymbolicLink() || path.resolve(fs.realpathSync(directory)) !== path.resolve(directory)) {
    throw new Error(label + " must be a canonical real directory");
  }
  return directory;
}

/** @param {string} directory */
function availableDiskBytes(directory) {
  const stat = fs.statfsSync(directory, {bigint: true});
  if (stat.bsize <= 0n) {throw new Error("Cannot determine Windows LLVM handoff extraction capacity");}
  const bytes = stat.bavail * stat.bsize;
  return bytes > BigInt(Number.MAX_SAFE_INTEGER) ? Number.MAX_SAFE_INTEGER : Number(bytes);
}

/** Derive the inner Windows host-SDK deadline without extending this job or run deadline.
 * @param {ReturnType<typeof windowsLlvmCiJobPlan>} plan @param {number} [reserveMs] @param {number} [nowMs] */
function hostSdkDeadline(plan, reserveMs = 6 * 60 * 1000, nowMs = Date.now()) {
  const now = nowMs, jobEnd = Number(plan.jobDeadlineEpoch) * 1000, runEnd = Number(plan.runDeadlineEpoch) * 1000,
    deadlineMs = Math.min(now + sdkComposeWindowMs, jobEnd - reserveMs, runEnd - reserveMs), epoch = Math.floor(deadlineMs / 1000);
  if (!Number.isSafeInteger(epoch) || epoch * 1000 - now < 17 * 60 * 1000) {
    throw new Error("Insufficient remaining stage budget to safely compose the Windows host SDK and keep phase reserve");
  }
  return String(epoch);
}

/** @param {string} file @param {string} expected @param {string} label */
function copyPinned(file, expected, label) {
  const before = stableRegular(file, 8 << 20, label, expected), bytes = fs.readFileSync(file);
  if (bytes.length !== before.bytes || fileHash(file) !== expected) {throw new Error(label + " changed while being copied");}
  return bytes;
}

/** Bind the configure consumer to the exact producer outputs before any compilation.
 * @param {Record<string,string>} generated @param {string} descriptorPath @param {string} provenancePath */
function windowsLlvmDescriptorProducerPins(generated, descriptorPath, provenancePath) {
  const descriptorSha = generated["windows_llvm_descriptor_sha256"], provenanceSha = generated["windows_llvm_descriptor_provenance_sha256"];
  if (generated["windows_llvm_descriptor_path"] !== descriptorPath || generated["windows_llvm_descriptor_provenance_path"] !== provenancePath ||
      typeof descriptorSha !== "string" || typeof provenanceSha !== "string" ||
      !/^[a-f0-9]{64}$/.test(descriptorSha) || !/^[a-f0-9]{64}$/.test(provenanceSha)) {
    throw new Error("Windows LLVM descriptor producer returned inconsistent paths or pins");
  }
  copyPinned(descriptorPath, descriptorSha, "Produced Windows LLVM descriptor");
  copyPinned(provenancePath, provenanceSha, "Produced Windows LLVM descriptor provenance");
  return {descriptorSha, provenanceSha};
}

/** Execute one phase from exact same-run artifacts. CLI callers always use the maintained admission readers.
 * @param {string} stage @param {NodeJS.ProcessEnv} [environment] */
function runWindowsLlvmCiJob(stage, environment = process.env) {
  const plan = windowsLlvmCiJobPreflight(stage, environment), env = plan.environment, suffix = plan.suffix;
  const remaining = (/** @type {number} */ capMs, reserveMs = 60 * 1000) => {
    const end = Math.min(Number(plan.jobDeadlineEpoch) * 1000, Number(plan.runDeadlineEpoch) * 1000), ms = Math.min(capMs, end - Date.now() - reserveMs);
    if (!Number.isSafeInteger(ms) || ms < 1000) {throw new Error("Windows LLVM shared job/run budget expired before " + stage);}
    return ms;
  };
  if (stage === "configure" && !environment["MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH"]) {
    if (!environment["GITHUB_ENV"] || !path.isAbsolute(environment["GITHUB_ENV"])) {
      throw new Error("Windows LLVM configure needs a workflow environment file for its immutable run deadline");
    }
    const record = `MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH=${plan.runDeadlineEpoch}\n`;
    fs.appendFileSync(environment["GITHUB_ENV"], record, "utf8");
  }
  const toolsRoot = path.join(plan.runnerTemp, `linux-llvm-tools-extracted-${suffix}`), toolsName = `native-linux-llvm-tools-${suffix}`,
    hermesName = `native-windows-controller-hermes-${suffix}`;
  if (fs.existsSync(toolsRoot) || fs.existsSync(path.join(plan.runnerTemp, hermesName))) {
    throw new Error("Windows LLVM handoff extraction root already exists");
  }
  fs.mkdirSync(toolsRoot, {mode: 0o700});
  const toolsMembers = ci.listState(plan.incomingToolsArchive, [toolsName], remaining(5 * 60 * 1000)),
    toolsExpanded = ci.tarExpandedBytes(plan.incomingToolsArchive, toolsMembers.length, remaining(5 * 60 * 1000));
  if (toolsExpanded > 512 * 1024 ** 2 || availableDiskBytes(plan.runnerTemp) < toolsExpanded + 4 * 1024 ** 3) {
    throw new Error("Thin Linux LLVM tool handoff exceeds its expanded-size or runner-temp disk bound");
  }
  ci.extractState(plan.incomingToolsArchive, toolsRoot, [toolsName], [toolsName], remaining(8 * 60 * 1000));
  const tools = linuxLlvmToolHandoffInputs(path.join(toolsRoot, toolsName), {runAttempt: suffix, handoffSha256: plan.toolsHandoffSha256});
  const hermesMembers = ci.listState(plan.incomingHermesArchive, [hermesName], remaining(5 * 60 * 1000)),
    hermesExpanded = ci.tarExpandedBytes(plan.incomingHermesArchive, hermesMembers.length, remaining(5 * 60 * 1000));
  if (hermesExpanded > 4 * 1024 ** 3 || availableDiskBytes(plan.runnerTemp) < hermesExpanded + 4 * 1024 ** 3) {
    throw new Error("Windows Hermes source handoff exceeds its expanded-size or runner-temp disk bound");
  }
  ci.extractState(plan.incomingHermesArchive, plan.runnerTemp, [hermesName], [hermesName], remaining(8 * 60 * 1000));
  const hermesRoot = path.join(plan.runnerTemp, hermesName), hermes = windowsControllerHandoffInputs(hermesRoot,
    {kind: "hermes", runAttempt: suffix, handoffSha256: plan.hermesHandoffSha256});
  if (hermes.kind !== "hermes" || !hermes.intelRuntime || tools.sourceArchive !== path.join(path.dirname(tools.toolRoot), archiveName) ||
      fileHash(tools.sourceArchive) !== archiveSha256) {
    throw new Error("Windows LLVM handoff lacks its admitted Hermes runtime or exact locked LLVM source archive");
  }
  const sdkRoot = path.join(plan.runnerTemp, `mom-native-windows-host-sdk-${suffix}`, `native-windows-host-sdk-${suffix}`);
  const hostBase = path.join(plan.runnerTemp, `mom-native-windows-host-sdk-${suffix}`),
    sdkDeadline = hostSdkDeadline(plan, stage === "audit" ? 12 * 60 * 1000 : 6 * 60 * 1000);
  Object.assign(process.env, env, {
    MOM_NATIVE_WINDOWS_HOST_SDK_WORK_ROOT: hostBase,
    MOM_NATIVE_WINDOWS_HOST_SDK_ARCHIVE_DIR: path.join(hostBase, "archives"),
    MOM_NATIVE_WINDOWS_HOST_SDK_OUTPUT_DIR: sdkRoot,
    MOM_NATIVE_WINDOWS_HOST_SDK_METADATA_TAR: path.join(plan.runnerTemp, `native-windows-host-sdk-${suffix}-metadata.tar`),
    MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH: sdkDeadline,
  });
  ci.buildWindowsHostSdkSourcesCi();
  // State admission reads the descriptor's external SDK overlay and tools, so reconstruct them first.
  if (plan.previousStage) {
    const manifestBytes = copyPinned(plan.incomingStateManifest || "", plan.stateManifestSha256, "Windows LLVM state manifest");
    const manifest = JSON.parse(manifestBytes.toString("utf8"));
    stateTransport.restoreWindowsNativeLlvmState(plan.incomingStateArchive || "", plan.stateBundleSha256, manifest,
      plan.stateManifestSha256, env);
  }
  const inputFile = path.join(plan.runnerTemp, `windows-llvm21-descriptor-input-${suffix}.json`),
    input = /** @type {Parameters<typeof ci.composeWindowsLlvmSdkDescriptor>[0]} */ ({windowsSdkRoot: sdkRoot, windowsSdkPins: {
      resultSha256: fileHash(path.join(sdkRoot, "result.json")), descriptorSha256: fileHash(path.join(sdkRoot, "sdk.json")),
      filesSha256: fileHash(path.join(sdkRoot, "files.json")), overlaySha256: fileHash(path.join(sdkRoot, "vfs-overlay.json"))},
    hermesRoot, hermesExpectation: {kind: "hermes", runAttempt: suffix, handoffSha256: plan.hermesHandoffSha256},
    tools: {clangCl: {path: "/opt/intel/oneapi/compiler/2026.2/bin/icx-cl", sha256: fileHash("/opt/intel/oneapi/compiler/2026.2/bin/icx-cl")},
      lldLink: {path: tools.lldLink, sha256: fileHash(tools.lldLink), linkTarget: "lld"},
      llvmNm: {path: tools.llvmNm, sha256: fileHash(tools.llvmNm)},
      clangResourceInclude: "/opt/intel/oneapi/compiler/2026.2/lib/clang/23/include"}}),
    inputBytes = Buffer.from(JSON.stringify(input) + "\n");
  if (inputBytes.length > (8 << 20) || fs.existsSync(inputFile)) {throw new Error("Windows LLVM descriptor recomposition input is not fresh or bounded");}
  fs.writeFileSync(inputFile, inputBytes, {flag: "wx", mode: 0o600});
  const inputSha256 = fileHash(inputFile);
  if (stage === "configure") {
    const generated = ci.buildWindowsNativeLlvmDescriptorCi([inputFile, inputSha256], env),
      // The producer owns a separate fresh root; configure retains the pinned descriptor in stateRoot.
      descriptorPath = plan.descriptorState.descriptorOutput, provenancePath = plan.descriptorState.provenanceOutput,
      {descriptorSha, provenanceSha} = windowsLlvmDescriptorProducerPins(generated, descriptorPath, provenancePath);
    const values = ci.buildWindowsNativeLlvmCi(["configure", descriptorPath, descriptorSha, tools.sourceArchive, archiveSha256,
      "/usr/bin/cmake", fileHash("/usr/bin/cmake"), "/usr/bin/ninja", fileHash("/usr/bin/ninja")], env);
    copyToState(inputFile, path.join(plan.stateRoot, "descriptor-input.json"), inputSha256, 8 << 20);
    copyToState(provenancePath, path.join(plan.stateRoot, "descriptor-provenance.json"), provenanceSha, 8 << 20);
    return finishStage(plan, values, env);
  }
  const savedInput = path.join(plan.stateRoot, "descriptor-input.json"), savedProvenance = path.join(plan.stateRoot, "descriptor-provenance.json"),
    savedInputBytes = copyPinned(savedInput, fileHash(savedInput), "Original descriptor input"),
    savedProvenanceBytes = copyPinned(savedProvenance, fileHash(savedProvenance), "Original descriptor provenance"),
    descriptor = ci.composeWindowsLlvmSdkDescriptor(input),
    originalProvenance = JSON.parse(savedProvenanceBytes.toString("utf8"));
  if (!savedInputBytes.equals(inputBytes) || originalProvenance.status !== "PASS" ||
      originalProvenance.input?.sha256 !== inputSha256 || !isDeepStrictEqual(originalProvenance.composition, descriptor.provenance) ||
      !Buffer.from(JSON.stringify(descriptor.descriptor, null, 2) + "\n").equals(copyPinned(plan.stateRoot + "/descriptor.json",
        fileHash(path.join(plan.stateRoot, "descriptor.json")), "Original Windows LLVM descriptor"))) {
    throw new Error("Recomposed Windows LLVM descriptor differs from the immutable configured inputs");
  }
  if (plan.builderPhases.length) {
    return runBuilderPhases(plan.builderPhases, phase => ci.buildWindowsNativeLlvmCi(["build", phase], env),
      values => finishStage(plan, values, env));
  }
  return finishStage(plan, ci.buildWindowsNativeLlvmCi(["audit"], env), env);
}

/** Publish only after every original builder phase has completed successfully.
 * @param {readonly string[]} phases @param {(phase:string) => Record<string,string>} build
 * @param {(values:Record<string,string>) => Record<string,string>} publish */
function runBuilderPhases(phases, build, publish) {
  if (!phases.length) {throw new Error("Windows LLVM grouped stage must contain at least one builder phase");}
  let values;
  for (const phase of phases) {values = build(phase);}
  if (!values) {throw new Error("Windows LLVM grouped stage produced no final builder result");}
  return publish(values);
}

/** @param {string} source @param {string} target @param {string} sha256 @param {number} maximum */
function copyToState(source, target, sha256, maximum) {
  const bytes = copyPinned(source, sha256, "Windows LLVM retained descriptor evidence");
  if (bytes.length > maximum) {throw new Error("Windows LLVM descriptor evidence exceeds its handoff bound");}
  fs.copyFileSync(source, target, fs.constants.COPYFILE_EXCL); fs.chmodSync(target, 0o444);
  if (fileHash(source) !== sha256 || fileHash(target) !== sha256) {throw new Error("Windows LLVM descriptor evidence changed while retained");}
}

/** @param {ReturnType<typeof windowsLlvmCiJobPlan>} plan @param {Record<string,string>} values @param {NodeJS.ProcessEnv} environment */
function finishStage(plan, values, environment) {
  const reportedPhase = values["windows_llvm_phase"];
  if (typeof reportedPhase !== "string" || !/^(?:configure|audit)$/.test(reportedPhase) && !buildPhaseOutputValues.has(reportedPhase)) {
    throw new Error("Windows LLVM workflow output is malformed: windows_llvm_phase");
  }
  const expectedPhase = plan.stage === "configure" || plan.stage === "audit" ? plan.stage : "build-" + plan.statePhase;
  if (reportedPhase !== expectedPhase) {
    throw new Error("Windows LLVM builder result does not match its maintained phase or grouped output");
  }
  const {manifest, manifestSha256} = stateTransport.packWindowsNativeLlvmState(plan.archive, plan.environment),
    bytes = Buffer.from(JSON.stringify(manifest) + "\n");
  if (manifestSha256 !== createHash("sha256").update(bytes).digest("hex") || fs.existsSync(plan.manifest)) {
    throw new Error("Windows LLVM state manifest does not match its canonical external hash");
  }
  fs.writeFileSync(plan.manifest, bytes, {flag: "wx", mode: 0o444});
  const bundleSha256 = fileHash(plan.archive), manifestFileSha256 = fileHash(plan.manifest);
  if (bundleSha256 !== manifest.archive.sha256 || manifestFileSha256 !== manifestSha256) {
    throw new Error("Windows LLVM state artifacts changed while being retained");
  }
  const output = environment["GITHUB_OUTPUT"] || "";
  const outputStat = fs.lstatSync(output);
  if (!outputStat.isFile() || outputStat.isSymbolicLink() || outputStat.nlink !== 1) {
    throw new Error("Windows LLVM workflow output is not a fresh regular file");
  }
  const result = /** @type {Record<string,string>} */ ({...values, bundle_sha256: bundleSha256, manifest_sha256: manifestFileSha256,
    run_deadline_epoch: plan.runDeadlineEpoch});
  if (plan.stage === "audit") {result["admission_sha256"] = values["windows_llvm_admission_sha256"] || "";}
  const lines = Object.entries(result).map(([key, value]) => {
    const text = String(value);
    if (!/^[a-z][a-z0-9_]*$/.test(key) ||
        (!/^(?:[a-f0-9]{64}|[0-9]{10}|true|false|build|configure|audit|\/tmp\/mom-native-windows-llvm21-[0-9-]+\/output)$/.test(text) &&
          !buildPhaseOutputValues.has(text))) {
      throw new Error("Windows LLVM workflow output is malformed: " + key);
    }
    return key + "=" + value;
  });
  fs.appendFileSync(output, lines.join("\n") + "\n", "utf8");
  fs.appendFileSync(environment["GITHUB_STEP_SUMMARY"] || "", `### Windows LLVM ${plan.stage}\n` +
    `- Same-run state artifact: ${path.basename(plan.archive)} (${bundleSha256}); external manifest ${manifestFileSha256}.\n` +
    "- Build-only: true; executed: false; publication-qualified: false. Original producer receipts were not restamped.\n", "utf8");
  return result;
}

module.exports = {windowsLlvmCiJobPlan, windowsLlvmCiJobPreflight, windowsLlvmDescriptorProducerPins, hostSdkDeadline};

if (require.main === module) {
  try {
    const [stage, ...extra] = process.argv.slice(2);
    if (extra.length || !stage || !stages.includes(stage)) {throw new Error("Usage: run-native-windows-llvm-ci.js configure|llvm-amdgpu|llvm-mid|clang|lld|audit");}
    // Report from the result: re-planning here would re-check the phase budget after the work is
    // done and would derive the job id from the GITHUB_JOB the run already rewrote.
    const result = runWindowsLlvmCiJob(stage);
    process.stdout.write(JSON.stringify({status: "PASS", stage, output: result["windows_llvm_phase"],
      bundleSha256: result["bundle_sha256"], manifestSha256: result["manifest_sha256"], buildOnly: true,
      executed: false, publicationQualified: false}) + "\n");
  } catch (error) {
    process.stderr.write(`[native-ci] Windows LLVM job failed: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
