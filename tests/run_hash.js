"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { repoRoot, resolveNodeRunner, spawnAndExit } = require("./common/miner_command");

const logicSuite = fs.existsSync(path.join(repoRoot, "opts.js")) ? ["tests/logic.js"] : [];

const suites = {
  all: [...logicSuite, "tests/all.js"],
  cpu: ["tests/cpu.js"],
  gpu: ["tests/gpu.js"],
  "sycl-cpu": ["tests/sycl_cpu.js"],
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
  "--test-concurrency=1",
  ...suites[suite],
];

const runner = resolveNodeRunner(testArgs);
spawnAndExit(runner.command, runner.args);
