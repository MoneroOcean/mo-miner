"use strict";

const fs = require("node:fs");
const path = require("node:path");
const {repoRoot, resolveNodeRunner, spawnAndExit} = require("./common/miner_command");

// Release packages contain hash vectors but omit source-only policy and build tests.
const hasLogicSuite = fs.existsSync(path.join(repoRoot, "opts.js"));
// A standalone candidate has no N-API addon. Its CPU vectors exercise the real Core through
// the application; development-addon lifecycle checks remain part of Engineering Mode.
const addonSuite = process.env["MOM_TEST_EXECUTABLE"] ? [] : ["tests/native_addon.js"];
const logicSuite = [
  "tests/native_notices.js", "tests/native_msr.js", "tests/native_rocm_hip_headers.js",
  "tests/logic.js", "tests/native_runtime.js", "tests/native_application.js", "tests/native_object_cache.js", "tests/native_controller_incremental.js", "tests/native_gpu_incremental.js", "tests/native_cpu_incremental.js", "tests/native_windows_core_incremental.js", "tests/native_build_inputs.js", "tests/native_hermes_flags.js", "tests/native_tls.js", "tests/native_environment.js", "tests/native_gpu_symbols.js", "tests/native_pe_pages.js",
  "tests/native_pe_audit.js", "tests/native_release_packing.js", "tests/native_release_repro.js", "tests/native_release_entry.js", "tests/native_release_archive.js", "tests/native_release_validation.js", "tests/native_archive_contract.js", "tests/native_hermes_sdk.js", "tests/native_io_sdk.js", "tests/native_openssl_sdk.js", "tests/native_nvrtc_sdk.js", "tests/native_zstd_sdk.js", "tests/native_openmp_sdk.js", "tests/native_windows_source_release.js",
  "tests/native_worker.js", "tests/native_core.js", "tests/native_nvidia_metadata.js", "tests/native_process.js", "tests/native_path.js", "tests/native_bundle.js",
  "tests/native_services.js", "tests/native_process_io.js", "tests/native_worker_directory.js", "tests/native_output.js", "tests/native_rejections.js", "tests/native_opencl.js", "tests/native_static_adapters.js", "tests/native_level_zero_adapter.js",
  "tests/native_mining_probe.js", "tests/native_progpow.js", "tests/native_audit.js", "tests/native_driver_deps.js", "tests/native_driver_trace.js", "tests/native_resources.js", "tests/native_jit_headers.js", "tests/native_cuda_images.js", "tests/native_cuda_images_build.js", "tests/native_cuda_image_runtime.js", "tests/native_jit_link.js", "tests/native_configuration.js", "tests/native_cuda_adapter.js", "tests/native_dpcpp_graph.js", "tests/native_gpu_table.js", "tests/native_coff_isolation.js", "tests/native_gpu_dispatch.js",
  "tests/native_coff_archive.js", "tests/native_windows_application.js", "tests/native_windows_map.js", "tests/native_windows_opencl.js", "tests/native_windows_sdk_sources.js", "tests/native_windows_sdk_build.js", "tests/native_windows_sdk.js", "tests/native_windows_ci_tools.js", "tests/native_windows_adapters.js", "tests/native_windows_dpcpp_component.js", "tests/native_dpcpp_range.js", "tests/native_windows_jit.js", "tests/native_jit_resources.js", "tests/native_jit_sdk.js", "tests/native_jit_runtime.js", "tests/native_jit_stages.js", "tests/compiler_policy.js", "tests/portable_dag_modes.js",
  "tests/native_windows_application_config.js",
  "tests/native_windows_dpcpp_ci_job.js",
  "tests/native_windows_host_ci_job.js",
  "tests/native_windows_application_ci_job.js",
  "tests/native_packing_ci.js",
  "tests/native_package_ci.js",
  "tests/native_notice_review.js",
  "tests/native_jit_device_paths.js",
  "tests/native_jit_cache_identity.js",
  "tests/native_linux_application_metadata.js",
  "tests/native_linux_llvm_sdk_handoff.js",
  "tests/native_linux_sdk_config.js",
  "tests/local_native_sdk.js", "tests/local_native_jit_sdk.js", "tests/local_native_rocm_sdk.js",
  "tests/local_native_acpp_sdk.js", "tests/local_native_cuda_images.js", "tests/local_native_linux_sdk.js",
  "tests/local_native_distribution.js", "tests/local_native_command.js",
  "tests/local_native_windows_support.js", "tests/local_native_windows_support_export.js", "tests/local_native_windows_gpu_config.js",
  "tests/local_native_windows_jit_resources.js", "tests/local_native_windows_host_sdk.js", "tests/local_native_windows_llvm_sdk.js",
  "tests/local_native_windows_jit_sdk.js",
  "tests/local_native_windows_dpcpp_sdk.js",
  "tests/local_wsl_release.js",
  "tests/local_wsl_node.js",
  "tests/local_windows_support_command.js", "tests/local_native_windows_host_component.js",
  "tests/local_windows_hermes_source.js", "tests/local_native_windows_sdk.js",
  "tests/local_native_windows_support_bridge.js",
  "tests/local_native_windows_command.js",
  "tests/native_windows_gpu_handoff.js",
  "tests/nvidia_compatibility.js", "tests/intel_device_selection.js", "tests/c29_seed_unroll.js", "tests/gpu_test_matrix_order.js", "tests/integrated_gpu_modes.js",
  "tests/nvrtc_installer.js",
  "tests/pearlhash_f16_lowering.js",
  "tests/readme_performance.js", "tests/benchmark_scripts.js", "tests/windows_command.js",
  "tests/pool_transport.js", "tests/helper_workers.js", "tests/release_archive.js",
  "tests/build_release.js", "tests/native_build_release.js", "tests/native_sdk_fetch.js", "tests/native_ci.js", "tests/native_windows_llvm_state.js", "tests/native_windows_llvm_ci_job.js", "tests/native_windows_host_sdk_inputs.js", "tests/native_windows_gpu_config.js", "tests/native_windows_gpu_ci_job.js", "tests/native_linux_llvm_tools.js", "tests/native_gnu_headers.js", "tests/native_jit_base.js", "tests/native_sycl_sdk.js", "tests/native_dpcpp_sdk.js", "tests/native_dpcpp_component.js", "tests/native_comgr_sdk.js", "tests/native_rocm_host_runtime.js", "tests/native_rocm_resources.js", "tests/native_comgr_link.js", "tests/native_windows_sycl.js",
  "tests/native_release_sdk_inputs.js", "tests/native_acpp_sources.js", "tests/native_acpp_cache.js", "tests/native_acpp_allocation.js", "tests/native_windows_llvm21_sdk.js", "tests/native_windows_comgr_sdk.js", "tests/native_windows_acpp_sdk.js", "tests/native_windows_host_component.js", "tests/native_windows_hiprtc_sdk.js", "tests/native_windows_amdhip.js", "tests/native_windows_jit_ci_job.js", "tests/native_windows_jit_resources.js", "tests/native_llvm21_sdk.js", "tests/bounded_native_ci.js",
  "tests/native_windows_driver.js", "tests/native_windows_controller.js", "tests/native_windows_controller_handoff.js", "tests/native_windows_core_handoff.js", "tests/native_windows_core.js", "tests/native_windows_core_build.js", "tests/native_windows_gpu_build.js", "tests/native_windows_wrapper.js", "tests/build_environment.js", "tests/incremental_linux_build.js",
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
  engineering: ["tests/cpu.js", ...addonSuite, "tests/portable_gpu_cpu.js"],
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
