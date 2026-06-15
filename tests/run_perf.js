"use strict";

const { resolveNodeRunner, spawnAndExit } = require("./common/miner_command");
const { perfTests } = require("./vectors");

const algo = process.argv[2];
const testArgs = [
  "--require",
  "./tests/common/test_output_buffer.js",
  "--test",
  "--test-reporter=./tests/common/spec_reporter.js",
  "--test-concurrency=1",
  "tests/perf.js",
];
const testEnv = {};

if (algo && !perfTests.some((definition) => definition.algo === algo)) {
  console.error(`Unknown perf algo: ${algo}`);
  console.error(`Available algos: ${perfTests.map((definition) => definition.algo).join(", ")}`);
  process.exit(1);
}

if (algo) testEnv.MOM_PERF_ALGO = algo;
if (process.env.MOM_PERF_SAMPLES) testEnv.MOM_PERF_SAMPLES = process.env.MOM_PERF_SAMPLES;

const runner = resolveNodeRunner(testArgs, testEnv);
spawnAndExit(runner.command, runner.args, { env: runner.env });
