"use strict";

// One job builds the Linux source-JIT SDK: the public configure/support/targets/frontend/link phases,
// the proof-only runtime fixture, then the isolated static DPC++ component. Phase handoffs stay the
// same pinned local state archives the former per-phase jobs exchanged as artifacts.
const chain = require("./chain");

const phases = Object.freeze(["configure", "support", "targets", "frontend", "link", "static-dpcpp"]);

/** @param {string} suffix @param {string} phase */
function phaseArchive(suffix, phase) {return "native-linux-jit-sdk-" + suffix + "-" + (phase === "configure" ? "configured" : phase) + ".tar";}

/** Run the complete chain; uploads are the static-DPC++ state and the runtime-proof tar.
 * @param {NodeJS.ProcessEnv} [environment] */
function runLinuxJitChain(environment = process.env) {
  const context = chain.chainContext(environment), proof = "native-linux-jit-runtime-proof-" + context.suffix + ".tar",
    stage = (/** @type {string} */ name) => [...chain.nodeScript("scripts", "prepare-native-ci.js"), "stage", name];
  /** @type {Record<string, string>} */
  let proofOutputs = {};
  let previous = "";
  for (const phase of phases) {
    chain.runPhaseCommand(context, phase, stage("linux-jit-" + phase), environment);
    if (phase === "link") {proofOutputs = chain.runPhaseCommand(context, "runtime-proof", stage("linux-jit-runtime-proof"), environment).outputs;}
    if (phase === "static-dpcpp") {break;}
    const produced = phaseArchive(context.suffix, phase);
    chain.handOff(context, [produced]);
    if (previous) {chain.retire(context, [previous]);}
    chain.resetRunScopedRoots(context, [proof]);
    previous = produced;
  }
  chain.writeWorkflowOutputs(environment, {proof_sha256: chain.requiredOutput(proofOutputs, "proof_sha256"),
    result_sha256: chain.requiredOutput(proofOutputs, "result_sha256"), bundle_sha256: chain.requiredOutput(proofOutputs, "bundle_sha256"),
    bundle_bytes: String(Number(proofOutputs["bundle_bytes"] || ""))});
}

module.exports = {phases, phaseArchive, runLinuxJitChain};

if (require.main === module) {
  try {
    if (process.argv.length !== 2) {throw new Error("Usage: linux-jit.js");}
    runLinuxJitChain();
  } catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
