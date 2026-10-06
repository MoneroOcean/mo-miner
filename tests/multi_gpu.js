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

/** @param {HashVectorDefinition} testCase @param {string[]} devices @returns {HashVectorDefinition} */
function replaceDeviceList(testCase, devices) {
  const original = testCase.job.dev;
  if (typeof original !== "string") {throw new Error("cn/gpu vector has no device");}
  const suffix = original.slice(original.indexOf("*"));
  testCase.job.dev = devices.map((device) => `${device}${suffix}`).join(",");
  testCase.name = testCase.name.replace(original, testCase.job.dev);
  return testCase;
}

/** @param {string} vendor @param {string[]} devices */
async function runCombined(vendor, devices) {
  const definition = fastVectorFor("cn/gpu");
  const first = devices[0];
  if (!definition || !first) {throw new Error("No cn/gpu device vector");}
  const backend = compilerPolicy.selection("cn/gpu", vendor, process.platform)?.backend || "sycl";
  const testCase = replaceDeviceList(
    cloneForDiscreteGpu(definition, vendor, first, backend),
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
        return t.skip(result.reason);
      }
      const first = result.devices[0];
      if (!first) {return t.skip(`No discrete ${vendor} GPU is available`);}
      await runCombined(vendor, [first.dev, first.dev]);
    });

    it(`${vendor}: one worker on each discrete GPU`, {timeout: TEST_TIMEOUT_MS}, async (t) => {
      const result = await getGpuDevices(vendor, {algo: "cn/gpu", integrated: false});
      if (result.skipped || result.devices.length < 2) {
        const reason = result.skipped
          ? result.reason
          : `Only ${result.devices.length} discrete ${vendor} GPU is available`;
        if (process.env["MOM_REQUIRE_SAME_VENDOR_MULTI_GPU_TESTS"] === "1") {
          throw new Error(reason);
        }
        return t.skip(reason);
      }
      await runCombined(vendor, result.devices.map((device) => device.dev));
    });
  }

  it("mixed-vendor: one worker on each of two discrete GPUs", {timeout: TEST_TIMEOUT_MS}, async (t) => {
    const discoveries = await Promise.all(requestedVendors().map(async (vendor) => ({
      vendor,
      result: await getGpuDevices(vendor, {algo: "cn/gpu", integrated: false}),
    })));
    const available = discoveries.flatMap(({vendor, result}) => {
      if (result.skipped) {return [];}
      const first = result.devices[0];
      return first ? [{vendor, device: first.dev}] : [];
    });
    if (available.length < 2) {
      const details = discoveries.map(({vendor, result}) => (
        result.skipped
          ? `${vendor}: ${result.reason}`
          : `${vendor}: ${result.devices.length} discrete GPU(s) available`
      ));
      const reason = `Mixed-vendor concurrency requires at least two distinct vendors with a discrete GPU; ${details.join("; ")}`;
      if (process.env["MOM_REQUIRE_MULTI_GPU_TESTS"] === "1") {throw new Error(reason);}
      return t.skip(reason);
    }
    const lanes = available.slice(0, 2);
    await Promise.all(lanes.map(({vendor, device}) => runCombined(vendor, [device])));
  });
});
