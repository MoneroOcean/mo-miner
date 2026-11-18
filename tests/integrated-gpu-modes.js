"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const compilerPolicy = require("../compiler-policy");
const gpuTuning = require("../gpu-tuning");
const {addIntegratedCases} = require("./common/gpu-test-matrix");

const {
  TEST_TIMEOUT_MS,
  cloneForDiscreteGpu,
  cloneForIntelIntegrated,
  fastVectorFor,
  gpuAlgos,
} = require("./common/gpu-test-modes");

/** @param {Record<string, string | undefined>} values @param {() => void} callback */
function withEnv(values, callback) {
  const previous = new Map(Object.keys(values).map((key) => [key, process.env[key]]));
  try {
    for (const [key, value] of Object.entries(values)) {
      if (value === undefined) {delete process.env[key];} else {process.env[key] = value;}
    }
    callback();
  } finally {
    for (const [key, value] of previous) {
      if (value === undefined) {delete process.env[key];} else {process.env[key] = value;}
    }
  }
}

test.describe("Integrated GPU test modes", {concurrency: false}, () => {
  test("default integrated tests preserve consensus gold and inherited compact mode is cleared", () => {
    withEnv({MOM_INTEGRATED_PORTABLE_TEST: undefined, MOM_SYCL_PORTABLE_TEST: "1"}, () => {
      for (const algo of gpuAlgos) {
        const definition = fastVectorFor(algo);
        assert.ok(definition, `missing fast vector for ${algo}`);
        const copy = cloneForIntelIntegrated(definition, "gpu2", "sycl-native");
        const minimumTimeout = ["kawpow", "firopow", "evrprogpow", "meowpow", "etchash"]
          .includes(algo) ? 3 * 60 * 60 * 1000 : TEST_TIMEOUT_MS;
        const device = copy.job["dev"];
        const env = copy.env;

        assert.equal(copy.expected, definition.expected, `${algo} gold`);
        assert.equal(copy.gpu, definition.gpu, `${algo} device type`);
        assert.ok(typeof device === "string" && /^gpu2(?:\*|$)/.test(device), `${algo} stays on the GPU`);
        assert.equal(copy.job["backend"], "sycl-native", `${algo} backend`);
        if (algo === "c29" || algo === "zelhash") {
          assert.equal(gpuTuning.needsPrimaryTuning(device, algo), true,
            `${algo} default integrated mode keeps normal auto-tuning`);
        }
        assert.ok(!copy.name.includes("bounded"), `${algo} default label`);
        assert.ok(env, `${algo} clone env`);
        assert.equal(env["MOM_SYCL_PORTABLE_TEST"], undefined, `${algo} compact env cleanup`);
        assert.equal(env["MOM_INTEGRATED_PORTABLE_TEST"], undefined, `${algo} harness env cleanup`);
        assert.equal(copy.timeoutMs, Math.max(definition.timeoutMs || 0, minimumTimeout), `${algo} timeout`);
      }

      assert.equal(process.env["MOM_INTEGRATED_PORTABLE_TEST"], undefined);
      assert.equal(process.env["MOM_SYCL_PORTABLE_TEST"], "1");
    });
  });

  test("opt-in integrated fixtures use portable gold, bounded labels, and normal timeout", () => {
    withEnv({MOM_INTEGRATED_PORTABLE_TEST: "1", MOM_SYCL_PORTABLE_TEST: "inherited"}, () => {
      for (const algo of gpuAlgos) {
        const definition = fastVectorFor(algo);
        assert.ok(definition, `missing fast vector for ${algo}`);
        const original = JSON.stringify(definition);
        const copy = cloneForIntelIntegrated(definition, "gpu2", "sycl-native");
        const device = copy.job["dev"];
        const env = copy.env;

        assert.equal(copy.expected, definition.syclCpuExpected ?? definition.expected, `${algo} gold`);
        assert.equal(copy.gpu, definition.gpu, `${algo} device type`);
        assert.ok(typeof device === "string" && /^gpu2(?:\*|$)/.test(device), `${algo} stays on the GPU`);
        assert.equal(copy.job["backend"], "sycl-native", `${algo} backend`);
        assert.match(copy.name, /bounded portable correctness/, `${algo} label`);
        assert.ok(env, `${algo} clone env`);
        assert.equal(env["MOM_SYCL_PORTABLE_TEST"], "1", `${algo} compact fixture flag`);
        assert.equal(env["MOM_INTEGRATED_PORTABLE_TEST"], undefined, `${algo} harness env cleanup`);
        assert.equal(copy.timeoutMs, TEST_TIMEOUT_MS, `${algo} bounded timeout`);
        assert.equal(JSON.stringify(definition), original, `${algo} source vector is unchanged`);
      }

      assert.equal(process.env["MOM_INTEGRATED_PORTABLE_TEST"], "1");
      assert.equal(process.env["MOM_SYCL_PORTABLE_TEST"], "inherited");
    });
  });

  test("portable integrated discovery bypass is limited to the four bounded vectors", async (t) => {
    t.mock.method(compilerPolicy, "selection", () => ({backend: "sycl"}));
    const envKeys = ["MOM_INTEGRATED_PORTABLE_TEST", "MOM_REQUIRE_INTEGRATED_GPU_TESTS"];
    const previous = new Map(envKeys.map((key) => [key, process.env[key]]));
    delete process.env["MOM_INTEGRATED_PORTABLE_TEST"];
    delete process.env["MOM_REQUIRE_INTEGRATED_GPU_TESTS"];
    try {
      /** @param {Record<string, string>} [params] @returns {Parameters<typeof addIntegratedCases>[2]} */
      const presentDevice = (params = {}) => ({
        openclCpu: async () => ({skipped: true, reason: "unused"}),
        openclGpu: async () => ({skipped: true, reason: "unused"}),
        intelIntegrated: async () => ({skipped: false,
          devices: [{dev: "gpu2", description: "Intel UHD Graphics 750", integrated: true}],
          params}),
      });
      /** @type {Parameters<typeof addIntegratedCases>[0]} */
      const defaultPlan = new Map();
      await addIntegratedCases(defaultPlan, "c29", presentDevice());
      const defaultCases = [...defaultPlan.values()].flatMap(({serial}) => serial);
      assert.equal(defaultCases.length, 1);
      assert.match(defaultCases[0]?.reason || "", /c29 is not available on Intel UHD Graphics 750/);
      assert.equal(defaultCases[0]?.definitions.length, 0,
        "without the opt-in, existing full-mining discovery remains the gate");

      process.env["MOM_INTEGRATED_PORTABLE_TEST"] = "1";
      process.env["MOM_REQUIRE_INTEGRATED_GPU_TESTS"] = "1";
      const vectors = new Map([
        ["c29", /c29 portable edge generation/],
        ["zelhash", /zelhash gpu1 gen/],
        ["equihash192_7", /equihash192_7 gpu1 generation/],
        ["zhash", /zhash gpu1 generation/],
      ]);
      for (const [algo, expectedName] of vectors) {
        /** @type {Parameters<typeof addIntegratedCases>[0]} */
        const plan = new Map();
        await addIntegratedCases(plan, algo, presentDevice());
        const cases = [...plan.values()].flatMap(({serial}) => serial);
        assert.equal(cases.length, 1, `${algo} should retain its present-device bounded vector`);
        const entry = cases[0];
        assert.ok(entry);
        assert.equal(entry.reason, undefined, `${algo} must be attempted, not skipped as unavailable`);
        assert.match(entry.name, /bounded portable vector; full-mining support not implied/);
        assert.equal(entry.definitions.length, 1);
        const definition = fastVectorFor(algo);
        assert.ok(definition);
        assert.match(definition.name, expectedName);
        const selected = entry.select?.(definition);
        assert.ok(selected);
        assert.equal(selected.job["backend"], "sycl");
        const expectedDevice = {
          c29: "gpu2*[seed_workgroup=128;seed_blocks=16]",
          zelhash: "gpu2*[slots=4160]",
          equihash192_7: "gpu2*[intensity=1]",
          zhash: "gpu2*[intensity=1]",
        }[algo];
        const selectedDev = selected.job.dev;
        assert.equal(selectedDev, expectedDevice, `${algo} bounded launch tuning`);
        assert.ok(typeof selectedDev === "string");
        assert.equal(gpuTuning.needsPrimaryTuning(selectedDev, algo), false,
          `${algo} bounded vector must not trigger automatic GPU tuning`);
        assert.equal(selected.expected, definition.syclCpuExpected ?? definition.expected);
        assert.equal(selected.env?.["MOM_SYCL_PORTABLE_TEST"], "1");
        assert.match(selected.name, /bounded portable correctness/);
        assert.equal(process.env["MOM_INTEGRATED_PORTABLE_TEST"], "1");
      }

      /** @type {Parameters<typeof addIntegratedCases>[0]} */
      const hoohashPlan = new Map();
      await addIntegratedCases(hoohashPlan, "hoohash", presentDevice());
      const hoohashCases = [...hoohashPlan.values()].flatMap(({serial}) => serial);
      assert.equal(hoohashCases.length, 1);
      assert.match(hoohashCases[0]?.reason || "", /hoohash is not available on Intel UHD Graphics 750/);
      assert.equal(hoohashCases[0]?.required, true);
      assert.equal(hoohashCases[0]?.definitions.length, 0,
        "HooHash FP64/device-USM full-mining gates must not be bypassed");
    } finally {
      for (const [key, value] of previous) {
        if (value === undefined) {delete process.env[key];} else {process.env[key] = value;}
      }
    }
  });

  test("discrete GPU consensus vectors clear inherited compact-test switches", () => {
    withEnv({MOM_INTEGRATED_PORTABLE_TEST: "1", MOM_SYCL_PORTABLE_TEST: "1"}, () => {
      const definition = fastVectorFor("kawpow");
      assert.ok(definition);
      const copy = cloneForDiscreteGpu(definition, "intel", "gpu2", "sycl");

      assert.equal(copy.expected, definition.expected);
      assert.ok(copy.env);
      assert.equal(copy.env["MOM_SYCL_PORTABLE_TEST"], undefined);
      assert.equal(copy.env["MOM_INTEGRATED_PORTABLE_TEST"], undefined);
      assert.equal(copy.gpu, true);
      assert.equal(process.env["MOM_INTEGRATED_PORTABLE_TEST"], "1");
      assert.equal(process.env["MOM_SYCL_PORTABLE_TEST"], "1");
    });
  });
});
