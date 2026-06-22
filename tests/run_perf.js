"use strict";

const { resolveNodeRunner, spawnAndExit } = require("./common/miner_command");
const { perfTests } = require("./vectors");

const algo = process.argv[2];
const availableAlgos = perfTests.map((definition) => definition.algo);
if (algo && !availableAlgos.includes(algo)) {
  console.error(`Unknown perf algo: ${algo}`);
  console.error(`Available algos: ${availableAlgos.join(", ")}`);
  process.exit(1);
}

const testArgs = [
  "--require",
  "./tests/common/test_output_buffer.js",
  "--test",
  "--test-reporter=./tests/common/spec_reporter.js",
  "--test-concurrency=1",
  "tests/perf.js",
];
// perf.js reads these; MOM_PERF_SAMPLES is forwarded as-is when set.
const testEnv = {};
if (algo) {testEnv.MOM_PERF_ALGO = algo;}
if (process.env.MOM_PERF_SAMPLES) {testEnv.MOM_PERF_SAMPLES = process.env.MOM_PERF_SAMPLES;}

const runner = resolveNodeRunner(testArgs, testEnv);
spawnAndExit(runner.command, runner.args, { env: runner.env });
