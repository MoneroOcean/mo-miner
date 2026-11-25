"use strict";

// One job cross-builds the Windows LLVM SDK on Linux. Each maintained phase still runs through
// run-native-windows-llvm-ci.js with its own 55-minute deadline, re-admitted Hermes/LLVM-tool inputs and
// an externally pinned predecessor state. Phase deadlines never outlive the merged job; the 8-hour run
// deadline recorded in the state also covers the downstream host-component job.
const path = require("node:path"), chain = require("./chain");

/** Maintained phase order and the state-archive suffix each phase produces. */
/** @type {ReadonlyArray<readonly [string, string]>} */
const phases = Object.freeze([["configure", "configured"], ["llvm-amdgpu", "llvm-amdgpu"], ["llvm-mid", "llvm-core"],
  ["clang", "clang"], ["lld", "lld"], ["audit", "audited"]]);

/** @param {string} suffix @param {string} output */
function stateFiles(suffix, output) {
  const stem = "native-windows-llvm-state-" + suffix + "-" + output;
  return [stem + ".tar", stem + ".json"];
}

/** Per-phase deadlines and predecessor pins, layered over the job environment.
 * @param {import("./chain").ChainContext} context @param {{output: string, bundle: string, manifest: string} | null} previous */
function phaseEnvironment(context, previous) {
  const deadline = chain.phaseDeadline(context);
  return {...context.environment, MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: String(context.runDeadlineEpoch),
    MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: deadline, MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH: deadline,
    MOM_NATIVE_WINDOWS_LLVM_DESCRIPTOR_DIR: path.join(context.runnerTemp, "mom-native-windows-llvm21-descriptor-" + context.suffix),
    MOM_NATIVE_WINDOWS_LLVM_STATE_PREVIOUS_STAGE: previous ? previous.output : "",
    MOM_NATIVE_WINDOWS_LLVM_STATE_BUNDLE_SHA256: previous ? previous.bundle : "",
    MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256: previous ? previous.manifest : ""};
}

/** Run every phase; the audited state archive and manifest remain in RUNNER_TEMP for upload.
 * @param {NodeJS.ProcessEnv} [environment] */
function runWindowsLlvmChain(environment = process.env) {
  const context = chain.chainContext(environment);
  /** @type {{output: string, bundle: string, manifest: string} | null} */
  let previous = null;
  for (const [phase, output] of phases) {
    const {outputs} = chain.runPhaseCommand(context, phase,
      [...chain.nodeScript("scripts", "run-native-windows-llvm-ci.js"), phase], phaseEnvironment(context, previous));
    const current = {output, bundle: chain.requiredOutput(outputs, "bundle_sha256"), manifest: chain.requiredOutput(outputs, "manifest_sha256")};
    if (phase === "audit") {
      chain.writeWorkflowOutputs(environment, {bundle_sha256: current.bundle, manifest_sha256: current.manifest,
        admission_sha256: chain.requiredOutput(outputs, "admission_sha256")});
      return;
    }
    chain.handOff(context, stateFiles(context.suffix, output));
    if (previous) {chain.retire(context, stateFiles(context.suffix, previous.output));}
    chain.resetWorkRoots(context);
    previous = current;
  }
}

module.exports = {phases, phaseEnvironment};

if (require.main === module) {
  try {
    if (process.argv.length !== 2) {throw new Error("Usage: windows-llvm.js");}
    runWindowsLlvmChain();
  } catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
