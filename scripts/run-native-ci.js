"use strict";

// Native CI stage dispatcher used by the workflows: `node scripts/run-native-ci.js stage <name> [...]`.
const path = require("node:path");
const {runNode} = require("./native-ci-job");
const {packState, temp} = require("./native-ci-tar");
const {buildDpcppCudaInputs, buildLinuxGnuHeaders, buildWindowsGnuHeaders, ciPaths, fetchAcppSource, fetchHermesSource,
  fetchLinuxSources, fetchRocmSources, fetchZstdSource, prepareLinuxSysrootCi, sourcesTarName, verifyIoSource} = require("./native-ci-linux-sources");
const {buildLinuxLlvmCi, buildLinuxLlvmSdkHandoff, buildLinuxLlvmToolsCi} = require("./native-ci-linux-llvm");
const {buildHermesSdk, buildIoOpenSsl, buildOpenmpSdk, buildZstdSdk} = require("./native-ci-linux-support");
const {buildLinuxCudaImages, verifyNvrtcTarget} = require("./native-ci-linux-cuda");
const {admitLinuxJitRuntimeProof, buildLinuxJitResources, buildLinuxJitSdkPhase, buildLinuxJitStaticSdk,
  packLinuxJitRuntimeProof} = require("./native-ci-linux-jit");
const {buildAcppCudaHipSdk, buildRocmComgrSdk, buildRocmHostSdk} = require("./native-ci-linux-rocm");
const {buildLinuxNativeApplication, buildLinuxNativeComponent, buildLinuxNativeSdkDescriptor} = require("./native-ci-linux-release");
const {buildWindowsControllerHermesCi, buildWindowsControllerIoCi} = require("./native-ci-windows-controller");
const {buildWindowsNativeCoreCi} = require("./native-ci-windows-core");
const {buildWindowsHostSdkSourcesCi, buildWindowsNativeLlvmDescriptorCi} = require("./native-ci-windows-llvm");
const {buildWindowsGpuProfileCi} = require("./native-ci-windows-gpu");

/** Stages that take no argument beyond their name; each runs on its own native-job.yml job. */
const linuxStages = Object.freeze({
  "linux-sysroot": prepareLinuxSysrootCi, "linux-llvm": buildLinuxLlvmCi, "linux-llvm-installed-sdk": buildLinuxLlvmSdkHandoff,
  "linux-llvm-tools": buildLinuxLlvmToolsCi, "acpp-sources": fetchAcppSource, "linux-sources": fetchLinuxSources,
  "dpcpp-cuda-inputs": buildDpcppCudaInputs, "gnu-headers": buildLinuxGnuHeaders, "windows-gnu-headers": buildWindowsGnuHeaders,
  "linux-jit-resources": buildLinuxJitResources, "linux-jit-runtime-proof": packLinuxJitRuntimeProof,
  "linux-jit-runtime-admit": admitLinuxJitRuntimeProof, "linux-jit-static-dpcpp": buildLinuxJitStaticSdk, "rocm-sources": fetchRocmSources,
  "rocm-comgr-sdk": buildRocmComgrSdk, "rocm-host-sdk": buildRocmHostSdk, "hermes-sources": fetchHermesSource, "hermes-sdk": buildHermesSdk,
  "zstd-sources": fetchZstdSource, "zstd-sdk": buildZstdSdk, "openmp-sdk": buildOpenmpSdk, "acpp-cuda-hip-sdk": buildAcppCudaHipSdk,
  "cuda-images-linux": buildLinuxCudaImages, "linux-sdk-descriptor": buildLinuxNativeSdkDescriptor,
  "linux-native-application": buildLinuxNativeApplication, "io-openssl": buildIoOpenSsl,
});
const jitPhases = Object.freeze(["configure", "support", "targets", "frontend", "link"]);

/** Fetch the pinned LLVM/OpenSSL and libuv/zlib sources into one state tar. @param {ReturnType<typeof ciPaths>} paths */
function fetchLlvmSources(paths) {
  runNode("scripts/fetch-native-sdk-sources.js", ["build", paths.source, "--allow-network"], paths.env, 25 * 60 * 1000);
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", paths.source], paths.env, 2 * 60 * 1000);
  runNode("scripts/fetch-native-sdk-sources.js", ["io", paths.ioSource, "--allow-network"], paths.env, 25 * 60 * 1000);
  verifyIoSource(paths);
  packState(path.join(paths.runnerTemp, sourcesTarName(paths)), temp, [paths.sourceName, paths.ioSourceName]);
}

/** @param {string[]} args */
async function runStage(args) {
  const [stage = "", ...rest] = args;
  if (stage === "linux-native-component") {
    if (rest.length !== 1) {throw new Error("Linux native component stage requires exactly core or one GPU profile");}
    buildLinuxNativeComponent(ciPaths(), rest[0] || ""); return;
  }
  if (stage === "windows-native-gpu") {
    if (rest.length !== 3) {throw new Error("Windows GPU profile stage requires an explicit profile, config path and config SHA-256");}
    buildWindowsGpuProfileCi(rest[0] || "", rest[1] || "", rest[2] || ""); return;
  }
  if (stage === "windows-native-llvm21-descriptor") {
    if (rest.length !== 2) {throw new Error("Windows LLVM descriptor stage needs one input JSON and its SHA-256 pin");}
    buildWindowsNativeLlvmDescriptorCi(rest); return;
  }
  if (rest.length) {throw new Error("Native CI stage takes no extra arguments");}
  if (stage === "windows-native-core") {buildWindowsNativeCoreCi(); return;}
  if (stage === "windows-host-sdk-sources") {buildWindowsHostSdkSourcesCi(); return;}
  if (stage === "windows-controller-io") {await buildWindowsControllerIoCi(); return;}
  if (stage === "windows-controller-hermes") {buildWindowsControllerHermesCi(); return;}
  const paths = ciPaths();
  if (stage === "sources") {fetchLlvmSources(paths); return;}
  if (stage === "nvrtc-linux" || stage === "nvrtc-windows") {verifyNvrtcTarget(paths, stage === "nvrtc-linux" ? "linux" : "windows"); return;}
  if (stage.startsWith("linux-jit-") && jitPhases.includes(stage.slice("linux-jit-".length))) {
    buildLinuxJitSdkPhase(paths, stage.slice("linux-jit-".length)); return;
  }
  if (!Object.hasOwn(linuxStages, stage)) {throw new Error("Unknown native CI stage: " + stage);}
  linuxStages[/** @type {keyof typeof linuxStages} */ (stage)](paths);
}

module.exports = {runStage};

if (require.main === module) {
  if (process.argv[2] !== "stage" || process.argv.length < 4) {
    const stages = ["sources", "nvrtc-linux", "nvrtc-windows", ...jitPhases.map(phase => "linux-jit-" + phase), ...Object.keys(linuxStages),
      "linux-native-component <profile>", "windows-native-core", "windows-host-sdk-sources", "windows-controller-io",
      "windows-controller-hermes", "windows-native-gpu <profile> <config.json> <config-sha256>", "windows-native-llvm21-descriptor <input.json> <sha256>"];
    console.error("Usage: run-native-ci.js stage " + stages.join("|"));
    process.exitCode = 1;
  } else {
    runStage(process.argv.slice(3)).catch(error => {
      console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
      process.exitCode = 1;
    });
  }
}
