"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { hasReleaseExecutable, repoRoot, spawnAndExit } = require("./common/miner_command");
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

if (algo) testEnv.MOMINER_PERF_ALGO = algo;

function isInsideRsh() {
  return process.env.MOMINER_R_SH === "1" || fs.existsSync("/.dockerenv");
}

const nodeRunner = { command: process.execPath, args: testArgs, env: testEnv };
let runner;
if (process.platform === "win32" || isInsideRsh() || hasReleaseExecutable) {
  runner = nodeRunner;
} else if (fs.existsSync(path.join(repoRoot, "r.sh"))) {
  const args = ["env"];
  if (algo) args.push(`MOMINER_PERF_ALGO=${algo}`);
  runner = { command: "./r.sh", args: [...args, "node", ...testArgs] };
} else {
  runner = { command: "./docker-mo-miner.sh", args: ["node", ...testArgs], env: testEnv };
}

spawnAndExit(runner.command, runner.args, { env: runner.env });
