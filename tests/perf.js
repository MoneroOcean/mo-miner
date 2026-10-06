"use strict";

const {describe, it} = require("node:test");
const assert = require("node:assert/strict");

const {benchmarkTestTimeoutMs, formatHashrate, runMinerBench} = require("./common/miner_command");
const {perfTests} = require("./vectors");

const selectedAlgo = process.env["MOM_PERF_ALGO"] || "";
const gpuOnly = process.env["MOM_PERF_GPU_ONLY"] === "1";
const selectedTests = selectedAlgo
  ? perfTests.filter((definition) => definition.algo === selectedAlgo)
  : gpuOnly ? perfTests.filter((definition) => definition.gpu) : perfTests;

if (selectedAlgo && selectedTests.length === 0) {
  throw new Error(`Unknown perf algo: ${selectedAlgo}`);
}

// Algos whose auto config must resolve to a GPU with a named intensity.
const gpuIntensityAlgos = new Set([
  "kawpow", "firopow", "evrprogpow", "meowpow", "etchash", "octopus", "autolykos2", "verthash",
  "hoohash",
  "walahash", "equihash192_7",
  "xelishashv3", "nexapow",
]);

/** @param {string} algo @param {string | undefined} dev */
function assertGpuIntensityDev(algo, dev) {
  if (!gpuIntensityAlgos.has(algo)) {return;}
  if (typeof dev !== "string") {throw new Error(`${algo} did not report a GPU device`);}
  assert.match(dev, /(?:^|,)gpu\d+\*\[[^\]]*\bintensity=\d+/,
    `${algo} should be auto-detected on a GPU with an intensity`);
}

/** @param {{samples?: number[]}} result */
function sampleSummary(result) {
  if (!result.samples || result.samples.length <= 1) {return "";}
  return ` median of ${result.samples.length} samples [${result.samples.map(formatHashrate).join(", ")}]`;
}

describe(selectedAlgo ? `proof-of-work performance: ${selectedAlgo}`
  : gpuOnly ? "GPU proof-of-work performance" : "proof-of-work performance", () => {
  for (const definition of selectedTests) {
    it(definition.name, {timeout: benchmarkTestTimeoutMs(definition)}, async (t) => {
      const result = await runMinerBench(definition);
      if ("skipped" in result) {
        if (definition.gpu && process.env["MOM_REQUIRE_GPU_TESTS"] === "1") {
          throw new Error(`${definition.name}: ${result.reason}`);
        }
        t.skip(result.reason);
        return;
      }

      assert.ok(result.hashrate > 0, `${definition.name} reported invalid hashrate: ${result.hashrate}`);
      assertGpuIntensityDev(definition.algo, result.dev);
      t.diagnostic(`${definition.algo} (${result.dev}): ${formatHashrate(result.hashrate)}${sampleSummary(result)}`);
      if (process.env["MOM_PERF_VERBOSE"] === "1") {
        if (result.stdout.trim()) {t.diagnostic(`miner stdout:\n${result.stdout.trimEnd()}`);}
        if (result.stderr.trim()) {t.diagnostic(`miner stderr:\n${result.stderr.trimEnd()}`);}
      }
    });
  }
});
