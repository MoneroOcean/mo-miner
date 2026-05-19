"use strict";

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");

const { getFirstSyclCpuDevice, runMinerTest } = require("./common/miner_command");
const { hashTests } = require("./vectors");

const CPU_TEST_TIMEOUT_MS = 3 * 60 * 1000;

function cloneForSyclCpu(definition, dev) {
  const copy = JSON.parse(JSON.stringify(definition));
  copy.name = `${copy.name.replace(/gpu1/g, dev)} SYCL CPU`;
  copy.gpu = false;
  copy.timeoutMs = CPU_TEST_TIMEOUT_MS;
  copy.job.dev = copy.job.dev.replace(/gpu1/g, dev);
  return copy;
}

function requiredVector(name) {
  const definition = hashTests.find((entry) => entry.name === name);
  assert.ok(definition, `Missing hash vector: ${name}`);
  return definition;
}

const syclCpuVectors = [
  requiredVector("cn/gpu gpu1*8"),
  requiredVector("c29 proofsize 42 gpu1*1"),
];

describe("SYCL CPU hash vectors", () => {
  let detectedDevice;

  async function getDevice(t) {
    if (!detectedDevice) detectedDevice = getFirstSyclCpuDevice();
    const result = await detectedDevice;
    if (result.skipped) {
      t.skip(result.reason);
      return null;
    }
    return result.dev;
  }

  for (const source of syclCpuVectors) {
    it(source.name.replace(/gpu1/g, "SYCL CPU"), { timeout: CPU_TEST_TIMEOUT_MS }, async (t) => {
      const dev = await getDevice(t);
      if (!dev) return;

      const result = await runMinerTest(cloneForSyclCpu(source, dev));
      if (result.skipped) {
        t.skip(result.reason);
        return;
      }
      assert.equal(result.skipped, false);
    });
  }
});
