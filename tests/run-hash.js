"use strict";

const fs = require("node:fs");
const path = require("node:path");
const {repoRoot, resolveNodeRunner, spawnAndExit} = require("./common/miner-command");

// Release packages contain hash vectors but omit source-only policy and build tests.
const hasLogicSuite = fs.existsSync(path.join(repoRoot, "opts.js"));
// A standalone candidate has no N-API addon. Its CPU vectors exercise the real Core through
// the application; development-addon lifecycle checks remain part of Engineering Mode.
const addonSuite = process.env["MOM_TEST_EXECUTABLE"] ? [] : ["tests/native-addon.js"];
const logicSuite = [
  "tests/native-notices.js", "tests/native-msr.js", "tests/native-rocm-hip-headers.js",
  "tests/logic.js", "tests/native-runtime.js", "tests/native-application.js", "tests/native-object-cache.js", "tests/native-controller-incremental.js", "tests/native-gpu-incremental.js", "tests/native-cpu-incremental.js", "tests/native-windows-core-incremental.js", "tests/native-build-inputs.js", "tests/native-hermes-flags.js", "tests/native-tls.js", "tests/native-environment.js", "tests/native-gpu-symbols.js", "tests/native-pe-pages.js",
  "tests/native-pe-audit.js", "tests/native-release-packing.js", "tests/native-release-repro.js", "tests/native-release-entry.js", "tests/native-release-archive.js", "tests/native-release-validation.js", "tests/native-archive-contract.js", "tests/native-hermes-sdk.js", "tests/native-io-sdk.js", "tests/native-openssl-sdk.js", "tests/native-nvrtc-sdk.js", "tests/native-zstd-sdk.js", "tests/native-openmp-sdk.js", "tests/native-windows-source-release.js",
  "tests/native-worker.js", "tests/native-core.js", "tests/native-nvidia-metadata.js", "tests/native-process.js", "tests/native-path.js", "tests/native-bundle.js",
  "tests/native-services.js", "tests/native-process-io.js", "tests/native-worker-directory.js", "tests/native-output.js", "tests/native-rejections.js", "tests/native-opencl.js", "tests/native-static-adapters.js", "tests/native-level-zero-adapter.js",
  "tests/native-mining-probe.js", "tests/native-progpow.js", "tests/native-audit.js", "tests/native-driver-deps.js", "tests/native-driver-trace.js", "tests/native-resources.js", "tests/native-jit-headers.js", "tests/native-cuda-images.js", "tests/native-cuda-images-build.js", "tests/native-cuda-image-runtime.js", "tests/native-jit-link.js", "tests/native-configuration.js", "tests/native-cuda-adapter.js", "tests/native-dpcpp-graph.js", "tests/native-gpu-table.js", "tests/native-coff-isolation.js", "tests/native-gpu-dispatch.js",
  "tests/native-coff-archive.js", "tests/native-windows-application.js", "tests/native-windows-map.js", "tests/native-windows-opencl.js", "tests/native-windows-sdk-sources.js", "tests/native-windows-sdk-build.js", "tests/native-windows-sdk.js", "tests/native-windows-ci-tools.js", "tests/native-windows-adapters.js", "tests/native-windows-dpcpp-component.js", "tests/native-dpcpp-range.js", "tests/native-windows-jit.js", "tests/native-jit-resources.js", "tests/native-jit-sdk.js", "tests/native-jit-runtime.js", "tests/native-jit-stages.js", "tests/compiler-policy.js", "tests/portable-dag-modes.js",
  "tests/native-windows-application-config.js",
  "tests/native-windows-dpcpp-ci-job.js",
  "tests/native-windows-host-ci-job.js",
  "tests/native-windows-application-ci-job.js",
  "tests/native-packing-ci.js",
  "tests/native-packing-tool.js",
  "tests/native-recipe-coverage.js",
  "tests/native-package-ci.js",
  "tests/native-notice-review.js",
  "tests/native-jit-device-paths.js",
  "tests/native-jit-cache-identity.js",
  "tests/native-linux-application-metadata.js",
  "tests/native-linux-llvm-sdk-handoff.js",
  "tests/native-linux-sdk-config.js",
  "tests/native-linux-sysroot.js",
  "tests/native-libdrm-sdk.js",
  "tests/local-native-sdk.js", "tests/local-native-jit-sdk.js", "tests/local-native-rocm-sdk.js",
  "tests/local-native-acpp-sdk.js", "tests/local-native-cuda-images.js", "tests/local-native-linux-sdk.js",
  "tests/local-native-distribution.js", "tests/local-native-command.js",
  "tests/local-native-windows-support.js", "tests/local-native-windows-support-export.js", "tests/local-native-windows-gpu-config.js",
  "tests/local-native-windows-jit-resources.js", "tests/local-native-windows-host-sdk.js", "tests/local-native-windows-llvm-sdk.js",
  "tests/local-native-windows-jit-sdk.js",
  "tests/local-native-windows-dpcpp-sdk.js",
  "tests/local-native-wsl.js",
  "tests/local-native-wsl-node.js",
  "tests/local-windows-support-command.js", "tests/local-native-windows-host-component.js",
  "tests/local-windows-hermes-source.js", "tests/local-native-windows-sdk.js",
  "tests/local-native-windows-support-bridge.js",
  "tests/local-native-windows-command.js",
  "tests/native-windows-gpu-handoff.js", "tests/native-windows-cudart.js",
  "tests/nvidia-compatibility.js", "tests/intel-device-selection.js", "tests/c29-seed-unroll.js", "tests/gpu-test-matrix-order.js", "tests/integrated-gpu-modes.js",
  "tests/nvrtc-installer.js",
  "tests/pearlhash-f16-lowering.js",
  "tests/readme-performance.js", "tests/sorted-key-literals.js", "tests/benchmark-scripts.js", "tests/windows-command.js",
  "tests/pool-transport.js", "tests/helper-workers.js", "tests/release-archive.js",
  "tests/build-release.js", "tests/native-build-release.js", "tests/native-sdk-fetch.js", "tests/native-ci.js", "tests/native-ci-jobs.js", "tests/workflows.js", "tests/native-windows-llvm-state.js", "tests/native-windows-llvm-ci-job.js", "tests/native-windows-host-sdk-inputs.js", "tests/native-windows-gpu-config.js", "tests/native-windows-gpu-ci-job.js", "tests/native-linux-llvm-tools.js", "tests/native-gnu-headers.js", "tests/native-jit-base.js", "tests/native-sycl-sdk.js", "tests/native-dpcpp-sdk.js", "tests/native-dpcpp-component.js", "tests/native-comgr-sdk.js", "tests/native-rocm-host-runtime.js", "tests/native-rocm-resources.js", "tests/native-comgr-link.js", "tests/native-windows-sycl.js",
  "tests/native-release-sdk-inputs.js", "tests/native-acpp-sources.js", "tests/native-acpp-cache.js", "tests/native-acpp-allocation.js", "tests/native-windows-llvm-sdk.js", "tests/native-windows-comgr-sdk.js", "tests/native-windows-acpp-sdk.js", "tests/native-windows-host-component.js", "tests/native-windows-hiprtc-sdk.js", "tests/native-windows-amdhip.js", "tests/native-windows-jit-ci-job.js", "tests/native-windows-jit-resources.js", "tests/native-llvm-sdk.js", "tests/bounded-native-ci.js",
  "tests/native-windows-driver.js", "tests/native-windows-controller.js", "tests/native-windows-controller-handoff.js", "tests/native-windows-core-handoff.js", "tests/native-windows-core.js", "tests/native-windows-core-build.js", "tests/native-windows-gpu-build.js", "tests/native-windows-wrapper.js", "tests/build-environment.js", "tests/incremental-linux-build.js",
  "tests/windows-incremental-build.js", "tests/windows-incremental-orchestration.js",
  "tests/windows-worker-discovery.js", "tests/dev.js", "tests/native-deploy.js",
];

/** @type {Record<string, string[]>} */
const suites = {
  all: [
    ...(hasLogicSuite ? [...logicSuite, ...addonSuite] : []),
    "tests/all.js",
  ],
  logic: logicSuite,
  host: ["tests/host.js", ...logicSuite, "tests/hoohash-host.js", "tests/pearlhash-claim.js"],
  cpu: ["tests/cpu.js", ...addonSuite],
  engineering: ["tests/cpu.js", ...addonSuite, "tests/portable-gpu-cpu.js"],
  gpu: ["tests/gpu.js"],
  "gpu-discrete": ["tests/discrete-gpu.js"],
  "gpu-fishhash-batch": ["tests/fishhash-batch.js"],
  "gpu-integrated": ["tests/integrated-gpu.js"],
  "gpu-multi": ["tests/multi-gpu.js"],
  "gpu-portable-cpu": ["tests/portable-gpu-cpu.js"],
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
  "./tests/common/no-pool-network.js",
  "--require",
  "./tests/common/test-output-buffer.js",
  "--test",
  "--test-reporter=./tests/common/spec-reporter.js",
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
const vendorEnv = usesVendorMatrix && configuredVendors
  ? {MOM_GPU_TEST_VENDORS: portableOpencl ? undefined : configuredVendors}
  : {};
// Source suites assert maintained build policy; documented developer build controls (tuned CPU,
// forced rebuilds, disabled caches, plan-only or reused archives) would otherwise change it.
const developerBuildControls = ["MOM_CPU_MARCH", "MOM_FORCE_REBUILD", "MOM_NATIVE_CACHE", "MOM_LINUX_BUILD_PLAN_ONLY",
  "MOM_DPCPP_IMPL", "MOM_COMBINED_TARGETS", "MOM_DEPLOY_REUSE_ARCHIVE"];
const runnerEnv = sourceOnly
  ? {...vendorEnv, ...Object.fromEntries(developerBuildControls.map(key => [key, undefined]))}
  : vendorEnv;
// Source checks must not build a development container or probe GPU runtimes.
const runner = sourceOnly
  ? {command: process.execPath, args: testArgs, env: runnerEnv}
  : resolveNodeRunner(testArgs, runnerEnv);
spawnAndExit(runner.command, runner.args, {env: runner.env});
