"use strict";

const {describe, it} = require("node:test");

const compilerPolicy = require("../compiler-policy");
const {getGpuDevices, runMinerTest} = require("./common/miner_command");
const {
  TEST_TIMEOUT_MS,
  cloneForDiscreteGpu,
  fastVectorFor,
  requestedVendors,
} = require("./common/gpu_test_modes");

const definition = fastVectorFor("cn/gpu");

function replaceDeviceList(testCase, devices) {
  const original = testCase.job.dev;
  const suffix = original.slice(original.indexOf("*"));
  testCase.job.dev = devices.map((device) => `${device}${suffix}`).join(",");
  testCase.name = testCase.name.replace(original, testCase.job.dev);
  return testCase;
}

async function runCombined(vendor, devices) {
  const backend = compilerPolicy.selection("cn/gpu", vendor, process.platform)?.backend || "sycl";
  const testCase = replaceDeviceList(
    cloneForDiscreteGpu(definition, vendor, devices[0], backend),
    devices
  );
  const result = await runMinerTest(testCase);
  if (result.skipped) {throw new Error(result.reason);}
}

describe("GPU multi-worker correctness", {timeout: 30 * 60 * 1000}, () => {
  for (const vendor of requestedVendors()) {
    it(`${vendor}: two workers on one discrete GPU`, {timeout: TEST_TIMEOUT_MS}, async (t) => {
      const result = await getGpuDevices(vendor, {algo: "cn/gpu", integrated: false});
      if (result.skipped) {
        if (process.env.MOM_REQUIRE_MULTI_GPU_TESTS === "1") {throw new Error(result.reason);}
        return t.skip(result.reason);
      }
      await runCombined(vendor, [result.devices[0].dev, result.devices[0].dev]);
    });

    it(`${vendor}: one worker on each discrete GPU`, {timeout: TEST_TIMEOUT_MS}, async (t) => {
      const result = await getGpuDevices(vendor, {algo: "cn/gpu", integrated: false});
      if (result.skipped || result.devices.length < 2) {
        const reason = result.skipped
          ? result.reason
          : `Only ${result.devices.length} discrete ${vendor} GPU is available`;
        if (process.env.MOM_REQUIRE_MULTI_GPU_TESTS === "1") {throw new Error(reason);}
        return t.skip(reason);
      }
      await runCombined(vendor, result.devices.map((device) => device.dev));
    });
  }
});
