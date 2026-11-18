"use strict";

// One job cross-builds the Windows source-JIT SDK. Before every phase the same-run Windows host SDK and
// original LLVM descriptor are recomposed from pinned inputs (compose), then the unchanged phase runner
// restores the pinned predecessor state. Phase deadlines never outlive the merged job.
const fs = require("node:fs"), path = require("node:path"), chain = require("./chain");
const ciTar = require("../native-ci-tar"),
  ciWindowsLlvm = require("../native-ci-windows-llvm"), {fileHash} = require("../build-source-inputs");
const {ciSuffix} = require("../native-ci-job");
const {linuxLlvmToolHandoffInputs} = require("../native-linux-llvm-tools");
const {windowsControllerHandoffInputs} = require("../native-windows-controller-handoff");

const phases = Object.freeze(["configure", "support", "targets", "frontend", "link"]),
  intelRoot = "/opt/intel/oneapi/compiler/2026.2";

/** Host-SDK composition gets 25 minutes but keeps 10 minutes of its phase and run for the build.
 * @param {NodeJS.ProcessEnv} env @param {number} [nowMs] */
function hostSdkDeadline(env, nowMs = Date.now()) {
  const now = Math.floor(nowMs / 1000), job = env["MOM_NATIVE_WINDOWS_JIT_DEADLINE_EPOCH"] || "",
    run = env["MOM_NATIVE_WINDOWS_JIT_RUN_DEADLINE_EPOCH"] || "";
  if (!/^[0-9]{10}$/.test(job) || !/^[0-9]{10}$/.test(run)) {throw new Error("Windows JIT composition needs phase and run deadlines");}
  const epoch = Math.min(now + 1500, Number(job) - 600, Number(run) - 600);
  if (epoch - now < 1020) {throw new Error("Insufficient bounded time for host-SDK composition and the JIT phase");}
  return String(epoch);
}

/** Extract one pinned handoff archive into a fresh root under RUNNER_TEMP.
 * @param {string} archive @param {string} name @param {string} temp @param {number} maxExpanded */
function extractHandoff(archive, name, temp, maxExpanded) {
  if (fs.existsSync(path.join(temp, name))) {throw new Error("Descriptor handoff restore roots are not fresh");}
  const members = ciTar.listState(archive, [name], 5 * 60 * 1000);
  if (!members.length || members.length > 100000 || ciTar.tarExpandedBytes(archive, members.length, 5 * 60 * 1000) > maxExpanded) {
    throw new Error("Descriptor handoff exceeds its bounds: " + name);
  }
  ciTar.extractState(archive, temp, [name], [name], 8 * 60 * 1000);
  return path.join(temp, name);
}

/** Compose the Windows host SDK and original LLVM descriptor; returns the environment the JIT phase reads.
 * @param {NodeJS.ProcessEnv} env */
function composeWindowsJitInputs(env) {
  const suffix = ciSuffix(env), temp = fs.realpathSync(env["RUNNER_TEMP"] || ""),
    workRoot = path.join(temp, `mom-native-windows-host-sdk-${suffix}`), sdkRoot = path.join(workRoot, `native-windows-host-sdk-${suffix}`),
    toolName = `native-linux-llvm-tools-${suffix}`, hermesName = `native-windows-controller-hermes-${suffix}`,
    toolArchive = path.join(temp, "incoming", `${toolName}.tar`), hermesArchive = path.join(temp, "incoming", `${hermesName}.tar`),
    toolBundlePin = env["MOM_NATIVE_LINUX_LLVM_TOOLS_BUNDLE_SHA256"] || "", toolHandoffPin = env["MOM_NATIVE_LINUX_LLVM_TOOLS_HANDOFF_SHA256"] || "",
    hermesBundlePin = env["MOM_NATIVE_WINDOWS_CONTROLLER_HERMES_BUNDLE_SHA256"] || "",
    hermesHandoffPin = env["MOM_NATIVE_WINDOWS_CONTROLLER_HERMES_HANDOFF_SHA256"] || "";
  if (fileHash(toolArchive) !== toolBundlePin || fileHash(hermesArchive) !== hermesBundlePin) {
    throw new Error("Same-run Windows descriptor tool/source archive pin mismatch");
  }
  Object.assign(process.env, {MOM_NATIVE_WINDOWS_HOST_SDK_WORK_ROOT: workRoot,
    MOM_NATIVE_WINDOWS_HOST_SDK_ARCHIVE_DIR: path.join(workRoot, "archives"), MOM_NATIVE_WINDOWS_HOST_SDK_OUTPUT_DIR: sdkRoot,
    MOM_NATIVE_WINDOWS_HOST_SDK_METADATA_TAR: path.join(temp, `native-windows-host-sdk-${suffix}-metadata.tar`),
    MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH: hostSdkDeadline(env)});
  ciWindowsLlvm.buildWindowsHostSdkSourcesCi();
  const toolRoot = extractHandoff(toolArchive, toolName, temp, 512 * 1024 ** 2),
    tools = linuxLlvmToolHandoffInputs(toolRoot, {handoffSha256: toolHandoffPin}),
    hermesRoot = extractHandoff(hermesArchive, hermesName, temp, 4 * 1024 ** 3),
    hermes = windowsControllerHandoffInputs(hermesRoot, {kind: "hermes", handoffSha256: hermesHandoffPin});
  if (hermes.kind !== "hermes" || !hermes.intelRuntime) {throw new Error("Hermes handoff omits the admitted Intel runtime");}
  const descriptorState = ciWindowsLlvm.windowsNativeLlvmDescriptorCiStatePaths(process.env),
    inputFile = path.join(temp, `windows-llvm21-descriptor-input-${suffix}.json`),
    input = {windowsSdkRoot: sdkRoot, windowsSdkPins: {
      resultSha256: fileHash(path.join(sdkRoot, "result.json")), descriptorSha256: fileHash(path.join(sdkRoot, "sdk.json")),
      filesSha256: fileHash(path.join(sdkRoot, "files.json")), overlaySha256: fileHash(path.join(sdkRoot, "vfs-overlay.json"))},
    hermesRoot, hermesExpectation: {kind: "hermes", handoffSha256: hermesHandoffPin},
    tools: {clangCl: {path: intelRoot + "/bin/icx-cl", sha256: fileHash(intelRoot + "/bin/icx-cl")},
      lldLink: {path: tools.lldLink, sha256: fileHash(tools.lldLink), linkTarget: "lld"},
      llvmNm: {path: tools.llvmNm, sha256: fileHash(tools.llvmNm)}, clangResourceInclude: intelRoot + "/lib/clang/23/include"}};
  if (fs.existsSync(inputFile)) {throw new Error("Descriptor composer input is not fresh");}
  fs.writeFileSync(inputFile, JSON.stringify(input) + "\n", {flag: "wx", mode: 0o600});
  const produced = ciWindowsLlvm.buildWindowsNativeLlvmDescriptorCi([inputFile, fileHash(inputFile)], process.env);
  if (produced["windows_llvm_descriptor_path"] !== descriptorState.descriptorOutput ||
      produced["windows_llvm_descriptor_provenance_path"] !== descriptorState.provenanceOutput) {
    throw new Error("Descriptor producer output path differs from the existing same-run contract");
  }
  // The JIT runner restores and re-admits these exact archives itself after descriptor composition.
  fs.rmSync(toolRoot, {recursive: true}); fs.rmSync(hermesRoot, {recursive: true});
  if (fileHash(toolArchive) !== toolBundlePin || fileHash(hermesArchive) !== hermesBundlePin) {
    throw new Error("A descriptor input archive changed during composition");
  }
  return {MOM_NATIVE_WINDOWS_JIT_DESCRIPTOR_FILE: descriptorState.descriptorOutput,
    MOM_NATIVE_WINDOWS_JIT_DESCRIPTOR_SHA256: produced["windows_llvm_descriptor_sha256"] || "",
    MOM_NATIVE_WINDOWS_JIT_WINDOWS_SDK_ROOT: sdkRoot,
    MOM_NATIVE_WINDOWS_JIT_WINDOWS_SDK_RESULT_SHA256: input.windowsSdkPins.resultSha256,
    MOM_NATIVE_WINDOWS_JIT_WINDOWS_SDK_DESCRIPTOR_SHA256: input.windowsSdkPins.descriptorSha256,
    MOM_NATIVE_WINDOWS_JIT_WINDOWS_SDK_FILES_SHA256: input.windowsSdkPins.filesSha256,
    MOM_NATIVE_WINDOWS_JIT_WINDOWS_SDK_OVERLAY_SHA256: input.windowsSdkPins.overlaySha256};
}

/** Per-phase deadlines, descriptor root and predecessor pin, layered over the job environment.
 * @param {import("./chain").ChainContext} context @param {{phase: string, bundle: string} | null} previous */
function phaseEnvironment(context, previous) {
  const deadline = chain.phaseDeadline(context), run = String(context.runDeadlineEpoch);
  return {...context.environment, MOM_NATIVE_WINDOWS_JIT_RUN_DEADLINE_EPOCH: run, MOM_NATIVE_WINDOWS_JIT_DEADLINE_EPOCH: deadline,
    MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: run, MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: deadline,
    MOM_NATIVE_WINDOWS_LLVM_DESCRIPTOR_DIR: path.join(context.runnerTemp, "mom-native-windows-llvm21-descriptor-" + context.suffix),
    MOM_NATIVE_WINDOWS_JIT_STATE_PREVIOUS_PHASE: previous ? previous.phase : "",
    MOM_NATIVE_WINDOWS_JIT_STATE_BUNDLE_SHA256: previous ? previous.bundle : ""};
}

/** Run all phases; the linked state archive remains in RUNNER_TEMP for upload.
 * @param {NodeJS.ProcessEnv} [environment] */
function runWindowsJitChain(environment = process.env) {
  const context = chain.chainContext(environment),
    config = path.join(context.incoming, "config", `native-windows-jit-sdk-${context.suffix}.json`);
  /** @type {{phase: string, bundle: string} | null} */
  let previous = null;
  let configSha256 = "";
  for (const phase of phases) {
    const env = phaseEnvironment(context, previous),
      composed = chain.runPhaseCommand(context, phase + "-compose", [...chain.nodeScript("scripts", "ci", "windows-jit.js"), "compose"], env).exported,
      args = phase === "configure" ? [phase, config] : [phase, config, configSha256],
      {outputs} = chain.runPhaseCommand(context, phase, [...chain.nodeScript("scripts", "run-native-windows-jit-ci.js"), ...args], {...env, ...composed}),
      statePhase = phase === "configure" ? "configured" : phase, archive = `native-windows-jit-sdk-${context.suffix}-${statePhase}.tar`;
    if (phase === "configure") {configSha256 = chain.requiredOutput(outputs, "windows_jit_config_sha256");}
    const current = {phase: statePhase, bundle: chain.requiredOutput(outputs, "windows_jit_state_bundle_sha256")};
    if (phase === "link") {
      chain.writeWorkflowOutputs(environment, {state_bundle_sha256: current.bundle, config_sha256: configSha256});
      return;
    }
    chain.handOff(context, [archive]);
    if (previous) {chain.retire(context, [`native-windows-jit-sdk-${context.suffix}-${previous.phase}.tar`]);}
    chain.resetWorkRoots(context);
    previous = current;
  }
}

module.exports = {phases, hostSdkDeadline, composeWindowsJitInputs, phaseEnvironment, runWindowsJitChain};

if (require.main === module) {
  try {
    const command = process.argv[2];
    if (process.argv.length !== 3 || !["compose", "chain"].includes(command || "")) {throw new Error("Usage: windows-jit.js compose|chain");}
    if (command === "chain") {
      runWindowsJitChain();
    } else {
      const exports = composeWindowsJitInputs(process.env);
      fs.appendFileSync(process.env["GITHUB_ENV"] || "", Object.entries(exports).map(([key, value]) => `${key}=${value}`).join("\n") + "\n");
    }
  } catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
