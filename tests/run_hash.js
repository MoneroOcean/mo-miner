"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { repoRoot, resolveNodeRunner, spawnAndExit } = require("./common/miner_command");

// Only run the logic suite from a source checkout (opts.js is absent in release packages).
const hasLogicSuite = fs.existsSync(path.join(repoRoot, "opts.js"));

const suites = {
  all: [
    ...(hasLogicSuite
      ? [
        "tests/logic.js", "tests/compiler_policy.js", "tests/readme_performance.js",
        "tests/pool_transport.js",
      ]
      : []),
    "tests/all.js",
  ],
  cpu: ["tests/cpu.js"],
  gpu: ["tests/gpu.js"],
  "gpu-discrete": ["tests/discrete_gpu.js"],
  "gpu-multi": ["tests/multi_gpu.js"],
  "gpu-portable-cpu": ["tests/portable_gpu_cpu.js"],
};

const suite = process.argv[2] || "all";
if (!suites[suite]) {
  console.error(`Unknown hash test suite: ${suite}`);
  process.exit(1);
}

const testArgs = [
  "--require",
  "./tests/common/test_output_buffer.js",
  "--test",
  "--test-reporter=./tests/common/spec_reporter.js",
  // Some suites touch the same physical GPU through different backends. Keep top-level files
  // serialized; the algorithm-centric GPU matrix controls safe device-level concurrency itself.
  "--test-concurrency=1",
  ...suites[suite],
];

const usesVendorMatrix = suite === "all" || suite === "gpu" || suite === "gpu-discrete";
const portableOpencl = (process.env.MOM_GPU_BACKEND || "").toLowerCase() === "opencl";
const runnerEnv = usesVendorMatrix && !portableOpencl
  ? {MOM_GPU_TEST_VENDORS: process.env.MOM_GPU_TEST_VENDORS || "intel,nvidia,amd"}
  : {};
const runner = resolveNodeRunner(testArgs, runnerEnv);
spawnAndExit(runner.command, runner.args, {env: runner.env});
