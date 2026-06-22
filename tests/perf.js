"use strict";

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");

const { formatHashrate, runMinerBench } = require("./common/miner_command");
const { perfTests } = require("./vectors");

const selectedAlgo = process.env.MOM_PERF_ALGO || "";
const selectedTests = selectedAlgo ? perfTests.filter((definition) => definition.algo === selectedAlgo) : perfTests;

if (selectedAlgo && selectedTests.length === 0) {
  throw new Error(`Unknown perf algo: ${selectedAlgo}`);
}

// Algos whose auto config must resolve to a GPU device with an intensity (gpuN*M).
const gpuIntensityAlgos = new Set(["kawpow", "firopow", "evrprogpow", "meowpow", "etchash", "autolykos2"]);

function assertGpuIntensityDev(algo, dev) {
  if (!gpuIntensityAlgos.has(algo)) {return;}
  assert.match(dev, /(?:^|,)gpu\d+\*\d+(?:,|$)/, `${algo} should be auto-detected on a GPU with an intensity`);
}

function sampleSummary(result) {
  if (!result.samples || result.samples.length <= 1) {return "";}
  return ` median of ${result.samples.length} samples [${result.samples.map(formatHashrate).join(", ")}]`;
}

describe(selectedAlgo ? `proof-of-work performance: ${selectedAlgo}` : "proof-of-work performance", () => {
  for (const definition of selectedTests) {
    it(definition.name, { timeout: definition.timeoutMs || 3 * 60 * 1000 }, async (t) => {
      const result = await runMinerBench(definition);
      if (result.skipped) {
        t.skip(result.reason);
        return;
      }

      assert.ok(result.hashrate > 0, `${definition.name} reported invalid hashrate: ${result.hashrate}`);
      assertGpuIntensityDev(definition.algo, result.dev);
      t.diagnostic(`${definition.algo} (${result.dev}): ${formatHashrate(result.hashrate)}${sampleSummary(result)}`);
    });
  }
});
