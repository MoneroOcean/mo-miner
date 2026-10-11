"use strict";

const assert = require("node:assert/strict");
const {describe, test} = require("node:test");
const policy = require("../compiler-policy");
const {addDiscreteCases, addGenericCases, orderedLanes} = require("./common/gpu_test_matrix");

/** @typedef {Parameters<typeof addDiscreteCases>[0]} MatrixPlan */
/** @param {MatrixPlan} plan */
function executionCases(plan) {
  return orderedLanes(plan).flatMap(({lane}) => [...lane.discrete, ...lane.serial]);
}

/** @param {string} _algo @param {string} vendor */
function pearlBackend(_algo, vendor) {
  return {backend: vendor === "amd" ? "native" : "sycl-native"};
}

/** @param {ReturnType<typeof executionCases>[number]} entry */
function selectedDefinition(entry) {
  const first = entry.definitions[0];
  assert(first && entry.select);
  return entry.select(first);
}

describe("GPU matrix vendor-major execution", {concurrency: false}, () => {
  test("PearlHash cross-backend cases keep Intel and NVIDIA before AMD native", async (t) => {
    t.mock.method(policy, "nvidiaComputeCapability", () => 120);
    t.mock.method(policy, "selection", pearlBackend);
    /** @type {MatrixPlan} */
    const plan = new Map();
    await addDiscreteCases(plan, "pearlhash", ["amd", "nvidia", "intel", "amd"], async (algo, vendor) => ({
      skipped: false, devices: [{dev: "gpu4", description: `${vendor} GPU`, integrated: false}],
      params: {[algo]: "gpu4"},
    }));
    const cases = executionCases(plan);
    assert.deepEqual(cases.map(({vendor}) => vendor), ["intel", "nvidia", "amd", "amd", "amd"]);
    const original = [...plan.values()].flatMap(({discrete, serial}) => [...discrete, ...serial]);
    assert.equal(cases.length, original.length);
    assert.deepEqual(new Set(cases), new Set(original), "scheduling must preserve every original case exactly once");
    assert.deepEqual(cases.map(entry => selectedDefinition(entry).job.backend),
      ["sycl-native", "sycl-native", "native", "sycl", "sycl-native"]);
  });

  test("reversed subsets and missing vendors remain ordered and retain unavailable cases", async (t) => {
    t.mock.method(policy, "nvidiaComputeCapability", () => 120);
    t.mock.method(policy, "selection", () => ({backend: "sycl"}));
    /** @type {MatrixPlan} */
    const plan = new Map();
    await addDiscreteCases(plan, "octopus", ["amd", "nvidia"], async (algo, vendor) => vendor === "nvidia"
      ? {skipped: true, reason: "fixture unavailable"}
      : {skipped: false, devices: [{dev: "gpu2", description: "AMD GPU", integrated: false}], params: {[algo]: "gpu2"}});
    const cases = executionCases(plan);
    assert.deepEqual(cases.map(({vendor}) => vendor), ["nvidia", "amd"]);
    assert.equal(cases[0]?.reason, "fixture unavailable");
    const amdCase = cases[1];
    assert(amdCase);
    assert.equal(selectedDefinition(amdCase).job.dev?.startsWith("gpu2"), true);
    assert.deepEqual(orderedLanes(new Map()), []);
  });

  test("generic OpenCL cases follow vendor order across backend groups; CPU is last", async (t) => {
    t.mock.method(policy, "nvidiaComputeCapability", () => 120);
    t.mock.method(policy, "selection", pearlBackend);
    /** @type {MatrixPlan} */
    const plan = new Map();
    await addDiscreteCases(plan, "pearlhash", ["amd", "intel"], async (algo, vendor) => ({
      skipped: false, devices: [{dev: "gpu1", description: `${vendor} GPU`, integrated: false}], params: {[algo]: "gpu1"},
    }));
    await addGenericCases(plan, "pearlhash", {
      openclGpu: async () => ({skipped: false, devices: [
        {dev: "gpu8", description: "AMD Radeon", integrated: false},
        {dev: "gpu7", description: "NVIDIA GPU", integrated: false},
        {dev: "gpu6", description: "Intel GPU", integrated: false},
      ], params: {pearlhash: "gpu8,gpu7,gpu6"}}),
      openclCpu: async () => ({skipped: true, reason: "fixture CPU unavailable"}),
      intelIntegrated: async () => ({skipped: true, reason: "fixture integrated unavailable"}),
    });
    const cases = executionCases(plan);
    assert.deepEqual(cases.map(({vendor}) => vendor), ["intel", "intel", "intel", "nvidia", "amd", "amd", "amd", "amd", "other"]);
    assert.deepEqual(cases.filter(({name}) => name.startsWith("OpenCL ")).map(({name}) => name.split(":")[0]),
      ["OpenCL intel gpu6", "OpenCL nvidia gpu7", "OpenCL amd gpu8"]);
  });
});
