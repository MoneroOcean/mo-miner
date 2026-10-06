"use strict";

const fs = require("node:fs");
const path = require("node:path");
const {repoRoot, resolveNodeRunner, spawnAndExit} = require("./common/miner_command");

// Only run the logic suite from a source checkout (opts.js is absent in release packages).
const hasLogicSuite = fs.existsSync(path.join(repoRoot, "opts.js"));
const addonSuite = ["tests/native_addon.js"];

/** @type {Record<string, string[]>} */
const suites = {
  all: [
    ...(hasLogicSuite
      ? [
        "tests/logic.js", "tests/compiler_policy.js", "tests/nvidia_compatibility.js",
        "tests/readme_performance.js",
        "tests/benchmark_scripts.js", "tests/pool_transport.js", "tests/helper_workers.js",
        ...addonSuite,
      ]
      : []),
    "tests/all.js",
  ],
  cpu: ["tests/cpu.js", ...addonSuite],
  gpu: ["tests/gpu.js"],
  "gpu-discrete": ["tests/discrete_gpu.js"],
  "gpu-fishhash-batch": ["tests/fishhash_batch.js"],
  "gpu-integrated": ["tests/integrated_gpu.js"],
  "gpu-multi": ["tests/multi_gpu.js"],
  "gpu-portable-cpu": ["tests/portable_gpu_cpu.js"],
};

const suite = process.argv[2] || "all";
const selectedSuites = suites[suite];
if (!selectedSuites) {
  console.error(`Unknown hash test suite: ${suite}`);
  process.exit(1);
}

const testArgs = [
  ...(suite === "all" ? ["--require", "./tests/common/no_pool_network.js"] : []),
  "--require",
  "./tests/common/test_output_buffer.js",
  "--test",
  "--test-reporter=./tests/common/spec_reporter.js",
  // Some suites touch the same physical GPU through different backends. Keep top-level files
  // serialized; the algorithm-centric GPU matrix controls safe device-level concurrency itself.
  "--test-concurrency=1",
  ...selectedSuites,
];

const usesVendorMatrix = suite === "all" || suite === "gpu" || suite === "gpu-discrete";
const portableOpencl = (process.env["MOM_GPU_BACKEND"] || "").toLowerCase() === "opencl";
const configuredVendors = process.env["MOM_GPU_TEST_VENDORS"];
// OpenCL owns the generic matrix. Clear inherited vendor labels before either runner can schedule
// the same physical GPU again through a native backend.
const runnerEnv = usesVendorMatrix && configuredVendors
  ? {MOM_GPU_TEST_VENDORS: portableOpencl ? undefined : configuredVendors}
  : {};
const runner = resolveNodeRunner(testArgs, runnerEnv);
spawnAndExit(runner.command, runner.args, {env: runner.env});
