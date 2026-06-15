"use strict";

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");

const { getFirstSyclCpuDevice, runMinerTest } = require("./common/miner_command");
const { hashTests } = require("./vectors");

const CPU_TEST_TIMEOUT_MS = 15 * 60 * 1000;

function cloneForSyclCpu(definition, dev) {
  const copy = JSON.parse(JSON.stringify(definition));
  copy.name = `${copy.name.replace(/gpu1/g, dev)} SYCL CPU`;
  copy.gpu = false;
  copy.timeoutMs = CPU_TEST_TIMEOUT_MS;
  copy.job.dev = copy.job.dev.replace(/gpu1/g, dev);
  return copy;
}

function requiredVector(name) {
  const definition = hashTests.find((v) => v.name === name);
  assert.ok(definition, `Missing hash vector: ${name}`);
  return definition;
}

// MOM_SYCL_CPU_SKIP lets a build opt specific algos out of SYCL-CPU hash verification
// (comma-separated algo names) when a host SYCL CPU device miscompiles them. The Intel build
// leaves it unset and runs all five; the NVIDIA build is CUDA-only (no SYCL CPU device), so it
// skips this verification entirely.
const skipAlgos = new Set(
  (process.env.MOM_SYCL_CPU_SKIP || "").split(",").map((s) => s.trim()).filter(Boolean)
);

const syclCpuVectors = [
  requiredVector("cn/gpu gpu1*8"),
  requiredVector("kawpow gpu1*256"),
  requiredVector("etchash gpu1*256"),
  requiredVector("autolykos2 gpu1*1"),
  requiredVector("c29 proofsize 42 gpu1*1"),
].filter((definition) => !skipAlgos.has(definition.job.algo));

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

  for (const definition of syclCpuVectors) {
    it(definition.name.replace(/gpu1/g, "SYCL CPU"), { timeout: CPU_TEST_TIMEOUT_MS }, async (t) => {
      const dev = await getDevice(t);
      if (!dev) return;

      // runMinerTest throws on failure, so a non-skipped return already means the vector passed.
      const result = await runMinerTest(cloneForSyclCpu(definition, dev));
      if (result.skipped) t.skip(result.reason);
    });
  }
});
