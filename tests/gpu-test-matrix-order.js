"use strict";

const assert = require("node:assert/strict");
const {describe, test} = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const {runInNewContext} = require("node:vm");
const policy = require("../compiler-policy");
const {addDiscreteCases, addGenericCases, addIntegratedCases, orderedLanes} = require("./common/gpu-test-matrix");

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

/** @returns {typeof import("./common/gpu-test-matrix")} */
function standaloneMatrix() {
  const source = fs.readFileSync(path.join(__dirname, "common", "gpu-test-matrix.js"), "utf8");
  const loaded = {exports: {}};
  runInNewContext(source, {
    module: loaded,
    __dirname: path.join(__dirname, "common"),
    require: (/** @type {string} */ name) => {
      if (name === "../../compiler-policy") {return policy;}
      if (name === "./miner-command") {
        return {...require("./common/miner-command"), hasReleaseExecutable: true};
      }
      if (name === "./gpu-test-modes") {return require("./common/gpu-test-modes");}
      if (name === "../../scripts/check-portable-opencl") {
        return require("../scripts/check-portable-opencl");
      }
      return require(name);
    },
  });
  return /** @type {typeof import("./common/gpu-test-matrix")} */ (loaded.exports);
}

describe("GPU matrix vendor-major execution", {concurrency: false}, () => {
  test("required integrated coverage cannot silently skip an unsupported algorithm", async (t) => {
    t.mock.method(policy, "selection", () => ({backend: "sycl"}));
    const previous = process.env["MOM_REQUIRE_INTEGRATED_GPU_TESTS"];
    try {
      for (const required of [false, true]) {
        process.env["MOM_REQUIRE_INTEGRATED_GPU_TESTS"] = required ? "1" : "0";
        /** @type {MatrixPlan} */
        const plan = new Map();
        await addIntegratedCases(plan, "zhash", {
          openclCpu: async () => ({skipped: true, reason: "unused"}),
          openclGpu: async () => ({skipped: true, reason: "unused"}),
          intelIntegrated: async () => ({skipped: false,
            devices: [{dev: "gpu2", description: "Intel integrated GPU", integrated: true}], params: {}}),
        });
        const cases = executionCases(plan);
        assert.equal(cases.length, 1);
        assert.equal(cases[0]?.required, required);
        assert.match(cases[0]?.reason || "", /zhash is not available on Intel integrated GPU/);
        assert.equal(cases[0]?.definitions.length, 0);
      }
    } finally {
      if (previous === undefined) {delete process.env["MOM_REQUIRE_INTEGRATED_GPU_TESTS"];} else {
        process.env["MOM_REQUIRE_INTEGRATED_GPU_TESTS"] = previous;
      }
    }
  });

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

  test("standalone NVIDIA backend annotations drive lanes without a host SM query", async (t) => {
    t.mock.method(policy, "nvidiaComputeCapability", () => {
      throw new Error("standalone backend selection must not query test-host hardware");
    });
    /** @type {MatrixPlan} */
    const plan = new Map();
    const candidateMatrix = standaloneMatrix();
    await candidateMatrix.addDiscreteCases(plan, "pearlhash", ["nvidia"], async (algo) => ({
      skipped: false,
      devices: [{dev: "gpu4", description: "NVIDIA GPU", integrated: false}],
      params: {[algo]: "gpu4:auto[native]"},
    }));
    const cases = executionCases(plan);
    assert.deepEqual(cases.map(({name}) => name), [
      "nvidia gpu4: NVIDIA GPU", "nvidia gpu4: NVIDIA GPU (sycl fallback)",
      "nvidia gpu4: NVIDIA GPU (sycl-native fallback)",
    ]);
    assert.deepEqual(cases.map((entry) => selectedDefinition(entry).job.backend),
      ["native", "sycl", "sycl-native"]);
    await assert.rejects(candidateMatrix.addDiscreteCases(new Map(), "pearlhash", ["nvidia"],
      async (algo) => ({
        skipped: false,
        devices: [{dev: "gpu4", description: "NVIDIA GPU", integrated: false}],
        params: {[algo]: "gpu4"},
      })), /Standalone candidate did not report a resolved backend/);
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
      ], params: {pearlhash: "gpu8,gpu7,gpu6:auto[sycl-opencl]"}}),
      openclCpu: async () => ({skipped: true, reason: "fixture CPU unavailable"}),
      intelIntegrated: async () => ({skipped: true, reason: "fixture integrated unavailable"}),
    });
    const cases = executionCases(plan);
    assert.deepEqual(cases.map(({vendor}) => vendor), ["intel", "intel", "intel", "nvidia", "amd", "amd", "amd", "amd", "other"]);
    assert.deepEqual(cases.filter(({name}) => name.startsWith("OpenCL ")).map(({name}) => name.split(":")[0]),
      ["OpenCL intel gpu6", "OpenCL nvidia gpu7", "OpenCL amd gpu8"]);
  });
});
