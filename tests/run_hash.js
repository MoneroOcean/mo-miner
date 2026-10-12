"use strict";

const fs = require("node:fs");
const path = require("node:path");
const {repoRoot, resolveNodeRunner, spawnAndExit} = require("./common/miner_command");

// Release packages contain hash vectors but omit source-only policy and build tests.
const hasLogicSuite = fs.existsSync(path.join(repoRoot, "opts.js"));
const addonSuite = ["tests/native_addon.js"];
const logicSuite = [
  "tests/logic.js", "tests/compiler_policy.js", "tests/portable_dag_modes.js",
  "tests/nvidia_compatibility.js", "tests/intel_device_selection.js", "tests/gpu_test_matrix_order.js",
  "tests/pearlhash_f16_lowering.js",
  "tests/readme_performance.js", "tests/benchmark_scripts.js", "tests/windows_command.js",
  "tests/pool_transport.js", "tests/helper_workers.js", "tests/release_archive.js",
  "tests/build_release.js", "tests/build_environment.js", "tests/incremental_linux_build.js",
  "tests/windows_incremental_build.js", "tests/windows_incremental_orchestration.js",
  "tests/windows_worker_discovery.js", "tests/dev.js", "tests/native_deploy.js",
];

/** @type {Record<string, string[]>} */
const suites = {
  all: [
    ...(hasLogicSuite ? [...logicSuite, ...addonSuite] : []),
    "tests/all.js",
  ],
  logic: logicSuite,
  host: ["tests/host.js", ...logicSuite, "tests/hoohash_host.js", "tests/pearlhash_claim.js"],
  cpu: ["tests/cpu.js", ...addonSuite],
  gpu: ["tests/gpu.js"],
  "gpu-discrete": ["tests/discrete_gpu.js"],
  "gpu-fishhash-batch": ["tests/fishhash_batch.js"],
  "gpu-integrated": ["tests/integrated_gpu.js"],
  "gpu-multi": ["tests/multi_gpu.js"],
  "gpu-portable-cpu": ["tests/portable_gpu_cpu.js"],
};

const suite = process.argv[2] || "all";
const selectedSuites = Object.hasOwn(suites, suite) ? suites[suite] : undefined;
if (!selectedSuites) {
  console.error(`Unknown test suite: ${suite}. Choose: ${Object.keys(suites).join(", ")}`);
  process.exit(1);
}
const sourceOnly = suite === "logic" || suite === "host";
if (sourceOnly && !hasLogicSuite) {
  console.error(`The ${suite} suite requires a source checkout.`);
  process.exit(1);
}
if (suite === "host" && process.platform !== "linux") {
  console.error("The host suite requires Linux, Bash and a C++ compiler. Use the logic suite on Windows.");
  process.exit(1);
}

const testArgs = [
  "--require",
  "./tests/common/no_pool_network.js",
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
// Source checks must not build a development container or probe GPU runtimes.
const runner = sourceOnly
  ? {command: process.execPath, args: testArgs, env: runnerEnv}
  : resolveNodeRunner(testArgs, runnerEnv);
spawnAndExit(runner.command, runner.args, {env: runner.env});
