"use strict";

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");

const { formatHashrate, runMinerBench } = require("./common/miner_command");
const { perfTests } = require("./vectors");

const selectedAlgo = process.env.MOMINER_PERF_ALGO || "";
const selectedTests = selectedAlgo ? perfTests.filter((definition) => definition.algo === selectedAlgo) : perfTests;

if (selectedAlgo && selectedTests.length === 0) {
  throw new Error(`Unknown perf algo: ${selectedAlgo}`);
}

function assertAlgoParamsDevice(definition, dev) {
  if (definition.algo !== "kawpow") return;
  assert.match(dev, /(?:^|,)gpu\d+\*\d+(?:,|$)/, "kawpow should be auto-detected on a GPU with an intensity");
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
      assertAlgoParamsDevice(definition, result.dev);
      t.diagnostic(`${definition.algo} (${result.dev}): ${formatHashrate(result.hashrate)}`);
    });
  }
});
