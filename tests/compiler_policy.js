"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const {test} = require("node:test");
const policyModule = require("../compiler-policy");
const opts = require("../opts");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const gpuTuning = require("../gpu-tuning");
const helper = require("../helper");
const createJobApi = require("../miner/jobs");
const {normalizeAlgoName} = require("../miner/algorithms");
const {hexWithoutPrefix} = require("../miner/submission");
const {
  cloneForDiscreteGpu, cloneForIntelIntegrated, cloneForOpenclSycl, openclSyclEnv,
} = require("./common/gpu_test_modes");
const {
  createAlgoParamsReportCache,
  isMissingGpuOutput,
  maxCapturedOutputBytes,
  parseDiscreteGpuDevices, parseGpuDevices,
  runNode, wrapWindowsCmd,
} = require("./common/miner_command");


/** @typedef {{key: string, addon: string, backend: string, pearlhashProfile: {m: number, n: number, k: number, rank: number} | null}} CompilerSelection */
/** @typedef {Record<string, string | undefined>} TestEnv */
/** @typedef {NodeJS.ProcessEnv & {MOM_SYCL_COMPILER?: string | undefined, ACPP_VISIBILITY_MASK?: string | undefined, CUDA_VISIBLE_DEVICES?: string | undefined, HIP_VISIBLE_DEVICES?: string | undefined, MOM_NATIVE_PATH?: string | undefined, MOM_RUNTIME_DIR?: string | undefined, LD_LIBRARY_PATH?: string | undefined, PATH?: string | undefined, ONEAPI_DEVICE_SELECTOR?: string | undefined, UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS?: string | undefined, MOM_NEXAPOW_NATIVE?: string | undefined, MOM_OCTOPUS_SYCL_NATIVE?: string | undefined, MOM_XELISHASHV3_SYCL_NATIVE?: string | undefined, MOM_WALAHASH_SYCL_NATIVE?: string | undefined}} CompilerEnv */

/** @param {string} algo @param {string} gpu @param {NodeJS.Platform} [platform] @param {number | null} [nvidiaSm] @returns {CompilerSelection} */
function selectPolicy(algo, gpu, platform = process.platform, nvidiaSm = null) {
  const selected = policyModule.selection(algo, gpu, platform, nvidiaSm);
  if (!selected) {throw new Error(`No compiler policy for ${algo}/${gpu}/${platform}`);}
  return selected;
}

/** @param {string} algo @param {NodeJS.ProcessEnv} [env] @param {NodeJS.Platform} [platform] @param {string} [requestedBackend] @returns {CompilerEnv} */
function workerEnvironment(algo, env = process.env, platform = process.platform, requestedBackend = "auto") {
  return policyModule.workerEnv(algo, env, platform, requestedBackend);
}

const policy = {...policyModule, selection: selectPolicy, workerEnv: workerEnvironment};

/** @param {NodeJS.ProcessEnv} [env] */
function testJobApi(env = process.env) {
  const opt = opts.create_default_opts();
  const testProcess = Object.assign(Object.create(process), {env});
  return createJobApi({
    h: helper,
    opt,
    process: testProcess,
    compilerPolicy: policyModule,
    gpuTuning,
    hexWithoutPrefix,
    normalizeAlgoName,
    messageHandler: () => undefined,
    isExiting: () => false,
    getComputeCore: () => null,
    getLastJob: () => null,
    setLastJob: () => undefined,
  });
}

/** @param {string} compiler @param {string} backend @param {string} profile */
function policyFixture(compiler = "—", backend = "—", profile = "—") {
  const lines = [
    "| Key | Linux | Windows |",
    "| --- | --- | --- |",
    "| dpcpp | dpcpp/mom.node | dpcpp/mom.node |",
    "",
    "| OS | GPU | Compiler | Backend | PearlHash MxNxK/rank |",
    "| --- | --- | --- | --- | --- |",
    `| Linux | Intel | dpcpp | sycl | ${profile} |`,
    "",
    "| Algorithm | OS | GPU | Compiler | Backend |",
    "| --- | --- | --- | --- | --- |",
  ];
  if (compiler !== "—" || backend !== "—") {
    lines.push(`| foo | Linux | Intel | ${compiler} | ${backend} |`);
  }
  return lines.join("\n");
}

async function tuneCnGpu(rates) {
  const opt = {algo_params: {"cn/gpu": {dev: "gpu1*[intensity=1536]"}}};
  const fakeHelper = {
    formatHashrate: String,
    log: () => undefined,
    log_err: assert.fail,
    repeat(fn) { fn(() => fakeHelper.repeat(fn)); },
  };
  const tuner = require("../miner/gpu_autotune")({
    h: fakeHelper,
    opt,
    gpuTuning,
    benchAlgo: (_algo, callback, dev) => callback(rates.get(dev)),
  });
  await new Promise((resolve) => tuner.tuneAlgo("cn/gpu", resolve));
  return opt.algo_params["cn/gpu"].dev;
}

test("GPU test discovery excludes integrated devices", () => {
  // Device names are deliberately arbitrary: discovery uses only SYCL's integrated marker, never
  // a model-name/PCI-ID list, so new and unlisted GPU generations are covered automatically.
  const output = [
    "gpu1: Unlisted Intel discrete accelerator via Level Zero",
    "gpu2: Unlisted Intel integrated accelerator via Level Zero [integrated]",
  ].join("\n");
  assert.deepEqual(parseDiscreteGpuDevices(output), [{
    dev: "gpu1", description: "Unlisted Intel discrete accelerator via Level Zero",
  }]);

  assert.deepEqual(parseDiscreteGpuDevices([
    "gpu1: Future AMD discrete accelerator via HIP",
    "gpu2: Future AMD integrated accelerator via HIP [integrated]",
  ].join("\n")).map((device) => device.dev), ["gpu1"]);

  assert.deepEqual(parseDiscreteGpuDevices(
    "gpu1: Future NVIDIA discrete accelerator via CUDA"
  ).map((device) => device.dev), ["gpu1"]);

  assert.deepEqual(parseGpuDevices(output, true).map((device) => device.dev), ["gpu2"]);
  assert.deepEqual(parseGpuDevices(output, null).map((device) => device.dev), ["gpu1", "gpu2"]);
});

test("reported backend annotations are not copied into GPU device specifications", () => {
  assert.deepEqual(policy.parseReportedAlgoParam("gpu1*[intensity=8]:auto[sycl-native]"),
    {dev: "gpu1*[intensity=8]", backend: "auto"});
  assert.deepEqual(policy.parseReportedAlgoParam("gpu2*[intensity=1]:sycl"),
    {dev: "gpu2*[intensity=1]", backend: "sycl"});
  assert.deepEqual(policy.parseReportedAlgoParam("cpu*8"), {dev: "cpu*8"});
});

test("GPU tuning syntax preserves per-device workers and partial overrides", () => {
  const entries = gpuTuning.parseDeviceList(
    "gpu1*[intensity=39612672;workgroup=256]^2,gpu2*[workgroup=128]", "kawpow");
  assert.equal(gpuTuning.formatDeviceList(entries),
    "gpu1*[intensity=39612672;workgroup=256]^2,gpu2*[workgroup=128]");
  assert.equal(gpuTuning.formatDeviceList(
    gpuTuning.parseDeviceList("gpu1*39612672", "kawpow")),
  "gpu1*[intensity=39612672]");
  assert.equal(gpuTuning.formatDeviceList(gpuTuning.parseDeviceList("gpu1*128", "c29")),
    "gpu1*[seed_workgroup=128]");
  assert.equal(gpuTuning.formatDeviceList(gpuTuning.parseDeviceList("gpu1*8192", "pearlhash")),
    "gpu1*[m=8192]");
  assert.equal(gpuTuning.formatDeviceList(gpuTuning.parseDeviceList("gpu1*176", "zelhash")),
    "gpu1*[slots=176]");
  assert.equal(gpuTuning.formatDeviceList(gpuTuning.parseDeviceList("gpu1*1", "zhash")),
    "gpu1*[intensity=1]");
  assert.throws(() => gpuTuning.parseDeviceList("gpu1*2", "zhash"), /intensity must be 1/);
  assert.equal(gpuTuning.formatDeviceList(
    gpuTuning.parseDeviceList("gpu1*1", "equihash192_7")), "gpu1*[intensity=1]");
  assert.throws(() => gpuTuning.parseDeviceList("gpu1*2", "equihash192_7"),
    /intensity must be 1/);
  assert.equal(gpuTuning.formatDeviceList(gpuTuning.parseDeviceList("gpu1*1", "c30")),
    "gpu1*[intensity=1]");
  assert.throws(() => gpuTuning.parseDeviceList("gpu1*2", "c30"), /intensity must be 1/);
  assert.deepEqual(gpuTuning.tuningEnvironment("zelhash", {slots: 176}), {
    MOM_ZELHASH_SLOTS: "176",
  });
  assert.equal(gpuTuning.formatDeviceList(gpuTuning.parseDeviceList("gpu1*256", "beamhash3")),
    "gpu1*[workgroup=256]");
  assert.throws(() => gpuTuning.parseDeviceList("gpu1*[]", "kawpow"), /must not be empty/);
  assert.throws(() => gpuTuning.parseDeviceList("gpu1[intensity=2]", "kawpow"),
    /invalid device entry/);
  assert.throws(() => gpuTuning.parseDeviceList("gpu1*[intensity=2]", "beamhash3"),
    /intensity/);
  assert.throws(() => gpuTuning.parseDeviceList("gpu1*[workgroup=63]", "kawpow"),
    /must be one of/);
  assert.throws(() => gpuTuning.parseDeviceList("gpu1*[intensity=1e3]", "kawpow"),
    /base-10 integer/);
  assert.throws(() => gpuTuning.validateTuning("fishhash", {intensity: true}),
    /base-10 integer/);
  assert.throws(() => gpuTuning.parseDeviceList("gpu1024", "kawpow"), /index must be at most 1023/);
  assert.throws(() => gpuTuning.parseDeviceList("gpu1^9007199254740992", "kawpow"),
    /process count must be at most 1024/);
  const dagEntry = gpuTuning.parseDeviceList("gpu1*[dag_chunk=0]", "kawpow")[0];
  const cacheEntry = gpuTuning.parseDeviceList("gpu1*[cache_block=0]", "pearlhash")[0];
  assert.ok(dagEntry && cacheEntry);
  assert.deepEqual(dagEntry.tuning, {dag_chunk: 0});
  assert.deepEqual(cacheEntry.tuning, {cache_block: 0});
});

test("empirical GPU tuning candidates stay bounded around portable heuristics", () => {
  /** @param {string} algo @param {string} dev */
  const formats = (algo, dev) => gpuTuning.autotuneCandidates(
    algo, gpuTuning.parseDeviceEntry(dev, algo)
  ).map(gpuTuning.formatDeviceEntry);
  assert.deepEqual(formats("cn/gpu", "gpu1*[intensity=1536]"), [
    "gpu1*[intensity=1536]",
    "gpu1*[intensity=768]",
    "gpu1*[intensity=1152]",
  ]);
  const autolykos = formats("autolykos2", "gpu2*[intensity=26843520;workgroup=64]^2");
  assert(autolykos.includes("gpu2*[intensity=33554176;workgroup=64]^2"));
  assert(autolykos.includes("gpu2*[intensity=26843520;workgroup=256]^2"));
  assert.deepEqual(formats("autolykos2", "gpu1*[intensity=4294967295;workgroup=64]"), [
    "gpu1*[intensity=4294967295;workgroup=64]",
    "gpu1*[intensity=2147483392;workgroup=64]",
    "gpu1*[intensity=3221225216;workgroup=64]",
    "gpu1*[intensity=4294967040;workgroup=64]",
    "gpu1*[intensity=4294967295;workgroup=32]",
    "gpu1*[intensity=4294967295;workgroup=128]",
    "gpu1*[intensity=4294967295;workgroup=256]",
  ]);
  assert.equal(formats("zelhash", "gpu1*[slots=4480]").length, 1);
  for (const algo of ["c30", "equihash192_7", "zhash"]) {
    assert.deepEqual(formats(algo, "gpu1*[intensity=1]"), ["gpu1*[intensity=1]"]);
  }
  assert(formats("beamhash3", "gpu1*[workgroup=640]")
    .every((dev) => !dev.includes("workgroup=768") && !dev.includes("workgroup=1024")));
  assert.equal(formats("kawpow", "cpu1*8").length, 1);
});

test("empirical tuner selects the fastest candidate after requiring a material baseline gain", async () => {
  const rates = new Map([
    ["gpu1*[intensity=1536]", 100],
    ["gpu1*[intensity=768]", 105],
    ["gpu1*[intensity=1152]", 106],
  ]);
  assert.equal(await tuneCnGpu(rates), "gpu1*[intensity=1152]");
});

test("empirical tuner keeps the heuristic across benchmark noise", async () => {
  const rates = new Map([
    ["gpu1*[intensity=1536]", 100],
    ["gpu1*[intensity=768]", 101],
    ["gpu1*[intensity=1152]", 101.9],
  ]);
  assert.equal(await tuneCnGpu(rates), "gpu1*[intensity=1536]");
});

test("Pearl tuning is applied independently to each native worker job", () => {
  const first = {
    algo: "pearlhash",
    dev: "gpu1",
    pearlhash_base_target: "1",
    pearlhash_n: 131072,
    pearlhash_k: 4096,
    pearlhash_rank: 256,
  };
  gpuTuning.applyNativeJobTuning(
    first, gpuTuning.parseDeviceEntry("gpu1*[m=8192;k=2048;rank=128]", "pearlhash"), "pearlhash");
  assert.deepEqual(first, {
    algo: "pearlhash",
    dev: "gpu1",
    intensity: 8192,
    pearlhash_base_target: "1",
    pearlhash_n: 8192,
    pearlhash_k: 2048,
    pearlhash_rank: 128,
    target: "0".repeat(59) + "80000",
  });
  const second = {
    algo: "pearlhash",
    dev: "gpu1",
    pearlhash_base_target: "1",
    pearlhash_n: 131072,
    pearlhash_k: 4096,
    pearlhash_rank: 256,
  };
  gpuTuning.applyNativeJobTuning(
    second,
    gpuTuning.parseDeviceEntry("gpu1*[m=16384;n=32768;rank=256]", "pearlhash"),
    "pearlhash",
  );
  assert.deepEqual(second, {
    algo: "pearlhash",
    dev: "gpu1",
    intensity: 16384,
    pearlhash_base_target: "1",
    pearlhash_n: 32768,
    pearlhash_k: 4096,
    pearlhash_rank: 256,
    target: "0".repeat(59) + "80000",
  });
});

test("thread selection preserves algorithm-specific *B shorthand until worker resolution", () => {
  assert.equal(helper.get_dev_threads("gpu1*128^2,gpu2*[workgroup=256]"), 3);
  assert.equal(helper.get_thread_dev(0, "gpu1*128^2,gpu2*[workgroup=256]"), "gpu1*128");
  assert.equal(helper.get_thread_dev(1, "gpu1*128^2,gpu2*[workgroup=256]"), "gpu1*128");
  assert.equal(helper.get_thread_dev(2, "gpu1*128^2,gpu2*[workgroup=256]"),
    "gpu2*[workgroup=256]");

  const c29 = gpuTuning.parseDeviceEntry(helper.get_thread_dev(0, "gpu1*128^2"), "c29");
  assert.deepEqual(c29.tuning, {seed_workgroup: 128});
  const job = {algo: "c29", dev: "gpu1"};
  gpuTuning.applyNativeJobTuning(job, c29, "c29");
  assert.deepEqual(job, {algo: "c29", dev: "gpu1", intensity: 1});
  assert.deepEqual(gpuTuning.tuningEnvironment("c29", c29.tuning),
    {MOM_C29_SEED_LOCAL_SIZE: "128"});
  assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[intensity=1]", "c30").tuning,
    {intensity: 1});
  assert.deepEqual(gpuTuning.tuningEnvironment("beamhash3", {workgroup: 256}), {
    MOM_BEAMHASH3_WORKGROUP: "256",
    MOM_BEAMHASH3_COMPACT_WG: "256",
  });
});

test("portable Pearl tuning maps generic controls onto relevant vendor kernels", () => {
  assert.deepEqual(gpuTuning.tuningEnvironment("pearlhash", {
    cache_block: 32, tile: "4x2",
  }), {
    MOM_PEARLHASH_AMD_DP4A_CACHE_BLOCK: "32",
    MOM_PEARLHASH_CU_BLK: "32",
    MOM_PEARLHASH_AMD_DP4A_TILE: "4x2",
  });
});

test("GPU compiler Markdown selects platform defaults and overrides", () => {
  assert.equal(policy.selection("etchash", "intel", "linux").key, "oneapi");
  assert.equal(policy.selection("fishhash", "intel", "linux").key, "oneapi");
  assert.equal(policy.selection("karlsenhashv2", "intel", "linux").key, "oneapi");
  assert.equal(policy.selection("zhash", "intel", "linux").key, "dpcpp");
  assert.equal(policy.selection("zhash", "intel", "linux").backend, "sycl");
  assert.equal(policy.selection("zhash", "intel", "win32").key, "oneapi");
  assert.equal(policy.selection("xelishashv3", "intel", "linux").backend, "sycl-native");
  assert.equal(policy.selection("xelishashv3", "intel", "win32").backend, "sycl-native");
  assert.equal(policy.selection("autolykos2", "nvidia", "linux").key, "acpp-cuda");
  assert.equal(policy.selection("c30", "nvidia", "linux").key, "acpp-cuda");
  assert.equal(policy.selection("beamhash3", "nvidia", "linux").key, "dpcpp");
  assert.equal(policy.selection("equihash192_7", "nvidia", "linux").key, "acpp-cuda");
  assert.equal(policy.selection("fishhash", "nvidia", "linux").key, "acpp-cuda");
  assert.equal(policy.selection("karlsenhashv2", "nvidia", "linux").key, "acpp-cuda");
  assert.equal(policy.selection("zelhash", "nvidia", "linux").key, "dpcpp");
  assert.equal(policy.selection("zhash", "nvidia", "linux").key, "acpp-cuda");
  assert.equal(policy.selection("zhash", "nvidia", "linux").backend, "sycl");
  assert.equal(policy.selection("pearlhash", "nvidia", "linux").backend, "native");
  assert.equal(policy.selection("octopus", "nvidia", "linux").backend, "sycl-native");
  assert.equal(policy.selection("xelishashv3", "nvidia", "linux").backend, "sycl-native");
  assert.equal(policy.selection("nexapow", "nvidia", "linux").backend, "sycl-native");
  assert.equal(policy.selection("walahash", "nvidia", "linux").backend, "sycl-native");
  assert.deepEqual(policy.selection("pearlhash", "nvidia", "linux").pearlhashProfile,
    {m: 131072, n: 524288, k: 8192, rank: 128});
  assert.equal(policy.selection("autolykos2", "nvidia", "win32").key, "acpp-cuda");
  assert.equal(policy.selection("c30", "nvidia", "win32").key, "acpp-cuda");
  assert.equal(policy.selection("beamhash3", "nvidia", "win32").key, "dpcpp");
  assert.equal(policy.selection("cn/gpu", "nvidia", "win32").key, "dpcpp");
  assert.equal(policy.selection("cn/gpu", "nvidia", "win32").backend, "native");
  assert.equal(policy.selection("fishhash", "nvidia", "win32").key, "acpp-cuda");
  assert.equal(policy.selection("karlsenhashv2", "nvidia", "win32").key, "acpp-cuda");
  assert.equal(policy.selection("etchash", "nvidia", "win32").key, "dpcpp");
  assert.equal(policy.selection("pearlhash", "nvidia", "win32").backend, "native");
  assert.equal(policy.selection("octopus", "nvidia", "win32").backend, "sycl-native");
  assert.equal(policy.selection("walahash", "nvidia", "win32").backend, "sycl-native");
  assert.deepEqual(policy.selection("pearlhash", "nvidia", "win32").pearlhashProfile,
    {m: 131072, n: 524288, k: 8192, rank: 128});
  assert.equal(policy.selection("autolykos2", "amd", "linux").key, "acpp-hip");
  assert.equal(policy.selection("beamhash3", "amd", "linux").key, "acpp-hip");
  assert.equal(policy.selection("karlsenhashv2", "amd", "linux").key, "acpp-hip");
  assert.equal(policy.selection("pearlhash", "amd", "linux").key, "acpp-hip");
  assert.equal(policy.selection("pearlhash", "amd", "linux").backend, "native");
  assert.deepEqual(policy.selection("pearlhash", "amd", "linux").pearlhashProfile,
    {m: 131072, n: 131072, k: 2048, rank: 128});
  assert.equal(policy.selection("etchash", "amd", "linux").backend, "sycl");
  assert.equal(policy.selection("autolykos2", "amd", "linux").backend, "sycl-native");
  assert.deepEqual(policy.selection("pearlhash", "intel", "linux").pearlhashProfile,
    {m: 131072, n: 131072, k: 2048, rank: 128});
  assert.deepEqual(policy.selection("pearlhash", "intel", "win32").pearlhashProfile,
    {m: 131072, n: 131072, k: 2048, rank: 128});
  assert.equal(policy.selection("etchash", "amd", "win32").key, "acpp-hip");
  assert.equal(policy.selection("pearlhash", "amd", "win32").key, "acpp-hip");
  assert.equal(policy.selection("pearlhash", "amd", "win32").backend, "native");
  assert.deepEqual(policy.selection("pearlhash", "amd", "win32").pearlhashProfile,
    {m: 131072, n: 131072, k: 2048, rank: 128});
  assert.equal(policy.selection("cn/gpu", "intel", "linux").backend, "sycl-opencl");
  assert.equal(policy.selection("etchash", "intel", "linux").backend, "sycl");
  assert.equal(policy.selection("pearlhash", "intel", "linux").backend, "sycl-native");
  assert.equal(policy.selection("walahash", "intel", "linux").backend, "sycl-native");
  assert.equal(policy.selection("walahash", "amd", "linux").backend, "sycl-native");
  assert.equal(policy.selection("walahash", "amd", "win32").backend, "sycl-native");
  assert.equal(policy.selection("octopus", "intel", "linux").backend, "sycl-native");
  assert.equal(policy.selection("fishhash", "nvidia", "linux").backend, "sycl-native");
  assert.equal(policy.selection("c29", "nvidia", "linux").backend, "sycl");
  assert.equal(policy.selection("etchash", "opencl", "linux").backend, "sycl-opencl");
  assert.equal(policy.selection("etchash", "opencl", "linux").key, "dpcpp-opencl");
  assert.equal(policy.selection("etchash", "opencl", "win32").key, "dpcpp-opencl");
});

test("Windows compiler addons live in isolated runtime directories", () => {
  assert.equal(policy.selection("etchash", "intel", "win32").addon, "oneapi/mom.node");
  assert.equal(policy.selection("etchash", "nvidia", "win32").addon, "dpcpp/mom.node");
  assert.equal(policy.selection("autolykos2", "nvidia", "win32").addon, "acpp-cuda/mom.node");
});

test("Linux compiler addons also live in isolated runtime directories", () => {
  assert.equal(policy.selection("etchash", "intel", "linux").addon, "oneapi/mom.node");
  assert.equal(policy.selection("etchash", "nvidia", "linux").addon, "dpcpp/mom.node");
  assert.equal(policy.selection("autolykos2", "amd", "linux").addon, "acpp-hip/mom.node");
  assert.equal(policy.selection("etchash", "opencl", "linux").addon, "dpcpp-opencl/mom.node");
  assert.equal(policy.selection("etchash", "opencl", "win32").addon, "dpcpp-opencl/mom.node");
});

test("Linux worker environment isolates the selected compiler runtime", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
  fs.mkdirSync(path.join(root, "acpp-cuda"));
  fs.writeFileSync(path.join(root, "acpp-cuda", "mom.node"), "test");
  const env = policy.workerEnv("autolykos2", {
    MOM_GPU_BACKEND: "nvidia", MOM_GPU_INDEX: "2", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "120",
    LD_LIBRARY_PATH: "/system/lib"
  }, "linux");
  assert.equal(env.MOM_SYCL_COMPILER, "acpp-cuda");
  assert.equal(env.ACPP_VISIBILITY_MASK, "cuda");
  assert.equal(env.CUDA_VISIBLE_DEVICES, "2");
  assert.equal(env["MOM_GPU_INDEX"], "0");
  assert.equal(env.MOM_NATIVE_PATH, path.join(root, "acpp-cuda", "mom.node"));
  assert.equal(env.MOM_RUNTIME_DIR, path.join(root, "acpp-cuda"));
  assert.equal(env.LD_LIBRARY_PATH, [path.join(root, "acpp-cuda"),
    path.join(root, "acpp-cuda", "hipSYCL"), "/system/lib"].join(path.delimiter));
  fs.rmSync(root, {recursive: true, force: true});
});

test("Windows worker environment puts only the selected compiler runtime first", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
  fs.mkdirSync(path.join(root, "acpp-hip"));
  fs.writeFileSync(path.join(root, "acpp-hip", "mom.node"), "test");
  const env = policy.workerEnv("pearlhash", {
    MOM_GPU_BACKEND: "amd", MOM_GPU_INDEX: "3", MOM_NATIVE_DIR: root,
    Path: "C:\\Windows\\System32", ROCM_PATH: "C:\\ROCm", CUDA_PATH: "C:\\CUDA",
  }, "win32");
  assert.equal(env.MOM_SYCL_COMPILER, "acpp-hip");
  assert.equal(env.ACPP_VISIBILITY_MASK, "hip");
  assert.equal(env.HIP_VISIBLE_DEVICES, "3");
  assert.equal(env["MOM_GPU_INDEX"], "0");
  assert.equal(env.MOM_RUNTIME_DIR, path.join(root, "acpp-hip"));
  assert.equal(env["Path"], [path.join(root, "acpp-hip"),
    path.join(root, "acpp-hip", "hipSYCL"), "C:\\Windows\\System32"].join(path.delimiter));
  assert.equal(Object.hasOwn(env, "PATH"), false);
  fs.rmSync(root, {recursive: true, force: true});
});

test("explicit native addon path overrides compiler policy", () => {
  assert.deepEqual(policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "amd",
    MOM_NATIVE_PATH: "C:\\custom\\mom.node",
  }, "win32"), {});
});

test("launcher default native path does not disable per-algorithm policy", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
  fs.mkdirSync(path.join(root, "acpp-cuda"));
  fs.writeFileSync(path.join(root, "acpp-cuda", "mom.node"), "test");
  const env = policy.workerEnv("autolykos2", {
    MOM_GPU_BACKEND: "nvidia",
    MOM_NVIDIA_COMPUTE_CAPABILITY: "120",
    MOM_NATIVE_DIR: root,
    MOM_NATIVE_PATH: path.join(root, "dpcpp", "mom.node"),
    MOM_NATIVE_PATH_LAUNCHER_DEFAULT: path.join(root, "dpcpp", "mom.node"),
  }, "linux");
  assert.equal(env.MOM_SYCL_COMPILER, "acpp-cuda");
  assert.equal(env.MOM_NATIVE_PATH, path.join(root, "acpp-cuda", "mom.node"));
  fs.rmSync(root, {recursive: true, force: true});
});

test("compiler workers select the backend matching their artifact", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
  for (const key of ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-hip"]) {
    fs.mkdirSync(path.join(root, key));
    fs.writeFileSync(path.join(root, key, "mom.node"), "test");
  }
  assert.equal(policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0", MOM_NATIVE_DIR: root
  }, "linux")
    .ONEAPI_DEVICE_SELECTOR, "cuda:gpu");
  assert.equal(policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0",
    MOM_GPU_INDEX: "2", MOM_NATIVE_DIR: root
  }, "linux").ONEAPI_DEVICE_SELECTOR, "cuda:0");
  assert.equal(policy.workerEnv("etchash", {MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: root}, "linux")
    .ACPP_VISIBILITY_MASK, "hip");
  assert.equal(policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "amd", MOM_GPU_INDEX: "1", MOM_NATIVE_DIR: root
  }, "linux").HIP_VISIBLE_DEVICES, "1");
  assert.equal(policy.workerEnv("autolykos2", {MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: root}, "linux")
    .ACPP_VISIBILITY_MASK, "hip");
  const intelOneapi = policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "intel", MOM_GPU_INDEX: "4", MOM_NATIVE_DIR: root
  }, "linux");
  assert.equal(intelOneapi.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  assert.equal(intelOneapi.UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS, "1");
  assert.equal(policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
  }, "linux").ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  assert.equal(policy.workerEnv("cn/gpu", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
  }, "linux").ONEAPI_DEVICE_SELECTOR, "opencl:gpu");
  assert.equal(policy.workerEnv("__control__", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
  }, "linux").ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  assert.equal(policy.workerEnv("cn/gpu", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
  }, "linux", "sycl-l0").ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  const intelKarlsen = policy.workerEnv("karlsenhashv2", {
    MOM_GPU_BACKEND: "intel", MOM_GPU_INDEX: "4", MOM_NATIVE_DIR: root
  }, "linux");
  assert.equal(intelKarlsen.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  assert.equal(intelKarlsen.UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS, "1");
  assert.equal(policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root, UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS: "0"
  }, "linux").UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS, "0");
  const windowsIntel = policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root, Path: "C:\\Windows\\System32"
  }, "win32");
  assert.equal(windowsIntel.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  assert.equal(windowsIntel.UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS, "1");
  assert.equal(policy.workerEnv("__control__", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root, Path: "C:\\Windows\\System32"
  }, "win32").ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  const windowsPortable = policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: "cpu", MOM_NATIVE_DIR: root,
    Path: "C:\\Windows\\System32"
  }, "win32");
  assert.equal(windowsPortable.ONEAPI_DEVICE_SELECTOR, "opencl:cpu");
  assert.equal(windowsPortable["Path"], [path.join(root, "dpcpp-opencl"),
    path.join(root, "dpcpp-opencl", "hipSYCL"), path.join(root, "oneapi"),
    "C:\\Windows\\System32"].join(path.delimiter));
  assert.equal(Object.hasOwn(windowsPortable, "PATH"), false);
  const opencl = policy.workerEnv("etchash", {MOM_GPU_BACKEND: "opencl", MOM_NATIVE_DIR: root}, "linux");
  assert.equal(opencl.MOM_SYCL_COMPILER, "dpcpp-opencl");
  assert.equal(opencl.ONEAPI_DEVICE_SELECTOR, "opencl:gpu");
  assert.equal(opencl.LD_LIBRARY_PATH, [path.join(root, "dpcpp-opencl"),
    path.join(root, "dpcpp-opencl", "hipSYCL"), path.join(root, "dpcpp")].join(path.delimiter));
  assert.equal(policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "opencl", MOM_GPU_INDEX: "6", MOM_NATIVE_DIR: root
  }, "linux").ONEAPI_DEVICE_SELECTOR, "opencl:gpu");
  assert.equal(policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: "cpu", MOM_NATIVE_DIR: root
  }, "linux").ONEAPI_DEVICE_SELECTOR, "opencl:cpu");
  const portableIntel = policy.workerEnv("pearlhash", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
  }, "linux", "sycl-l0");
  assert.equal(portableIntel.MOM_SYCL_COMPILER, "dpcpp-opencl");
  assert.equal(portableIntel.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  const portableIntelOpencl = policy.workerEnv("pearlhash", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
  }, "linux", "sycl-opencl");
  assert.equal(portableIntelOpencl.MOM_SYCL_COMPILER, "dpcpp-opencl");
  assert.equal(portableIntelOpencl.ONEAPI_DEVICE_SELECTOR, "opencl:gpu");
  assert.equal(policy.workerEnv("pearlhash", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
  }, "linux", "sycl").ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  assert.equal(policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "opencl", MOM_NATIVE_DIR: root
  }, "linux", "sycl-opencl").ONEAPI_DEVICE_SELECTOR, "opencl:gpu");
  assert.throws(() => policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: root
  }, "linux", "sycl-l0"), /incompatible/);
  assert.throws(() => policy.validateBackend("unknown"), /Invalid GPU backend/);
  assert.throws(() => policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: "accelerator", MOM_NATIVE_DIR: root
  }, "linux"), /Invalid MOM_OPENCL_DEVICE_TYPE/);
  assert.throws(() => policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "intel", MOM_GPU_INDEX: "not-a-number", MOM_NATIVE_DIR: root
  }, "linux"), /Invalid MOM_GPU_INDEX/);
  assert.throws(() => policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "intel", MOM_GPU_INDEX: "1024", MOM_NATIVE_DIR: root
  }, "linux"), /Invalid MOM_GPU_INDEX/);
  assert.throws(() => policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "9".repeat(400),
    MOM_NATIVE_DIR: root
  }, "linux"), /Invalid MOM_NVIDIA_COMPUTE_CAPABILITY/);
  fs.rmSync(root, {recursive: true, force: true});
});

test("Equihash 192,7 scopes wider first-round partitioning to HIP", () => {
  const source = fs.readFileSync(
    path.join(__dirname, "../sycl/equihash192_7/equihash192_7_direct_session.hpp"), "utf8");
  assert.match(source,
    /defined\(MOM_SYCL_HAS_HIP\)[\s\S]*?default_round0_partitions = 8;[\s\S]*?#else[\s\S]*?default_round0_partitions = 2;/);
  assert.match(source, /unsigned Round0Partitions = default_round0_partitions/);
});

test("Equihash 192,7 derives first split partitions from the active bucket width", () => {
  const source = fs.readFileSync(
    path.join(__dirname, "../sycl/equihash192_7/equihash192_7_direct_session.hpp"), "utf8");
  assert.match(source,
    /constexpr unsigned partitions = 1u << \(15 - ActiveRound1::bucket_bits\);/);
  assert.match(source,
    /typename ActiveArena::Round2Record, partitions>/);
});

test("Equihash 192,7 scopes split-record partitioning to Level Zero", () => {
  const source = fs.readFileSync(
    path.join(__dirname, "../sycl/equihash192_7/equihash192_7_direct_session.hpp"), "utf8");
  const splitOutput = source.indexOf("submit_collision_round_bucketed_split_output_partitioned");
  const start = source.lastIndexOf("if (level_zero_)", splitOutput);
  const end = source.indexOf("if (level_zero_)", splitOutput);
  const splitRounds = source.slice(start, end);
  assert.equal((splitRounds.match(/submit_collision_round_bucketed_split_output_partitioned</g) || []).length, 1);
  assert.equal((splitRounds.match(/submit_split_collision_round_bucketed_partitioned</g) || []).length, 1);
  assert.equal((splitRounds.match(/submit_collision_round_bucketed_split_output</g) || []).length, 1);
  assert.equal((splitRounds.match(/submit_split_collision_round_bucketed</g) || []).length, 1);
  assert.match(splitRounds,
    /constexpr unsigned partitions =\s*std::is_same_v<ActiveArena, HybridArena> \? 2 : 4;/);
  assert.match(splitRounds, /typename ActiveArena::Round3Record, partitions>/);
  assert.match(splitRounds, /typename ActiveArena::Round4Record, 4>/);
});

test("Equihash 192,7 scopes late-round partitioning to Level Zero", () => {
  const source = fs.readFileSync(
    path.join(__dirname, "../sycl/equihash192_7/equihash192_7_direct_session.hpp"), "utf8");
  const start = source.indexOf("submit_split_collision_round_bucketed<");
  const end = source.indexOf("#endif", start);
  const lateRounds = source.slice(start, end);
  assert.match(lateRounds, /if \(level_zero_\)/);
  assert.equal((lateRounds.match(/submit_collision_round_bucketed_partitioned</g) || []).length, 2);
  assert.equal((lateRounds.match(/submit_collision_round_bucketed</g) || []).length, 2);
});

test("Equihash 192,7 gates its HIP input cache on local memory", () => {
  const session = fs.readFileSync(
    path.join(__dirname, "../sycl/equihash192_7/equihash192_7_direct_session.hpp"), "utf8");
  const shared = fs.readFileSync(
    path.join(__dirname, "../sycl/zhash/equihash_sycl.hpp"), "utf8");
  assert.match(shared, /bool CacheInput = false/);
  assert.match(shared, /partitioned_collision_local_bytes/);
  assert.match(session,
    /local_mem_size>\(\) >= required_cache_local_bytes/);
  assert.match(session,
    /if \(cache_partition_inputs_\) return run_with_cache<true>\(header\);/);
});

test("HooHash portable OpenCL avoids unsupported 64-bit mul_hi", () => {
  const source = fs.readFileSync(
    path.join(__dirname, "../sycl/hoohash/hoohash.cpp"), "utf8");
  const shared = fs.readFileSync(path.join(__dirname, "../sycl/lib-internal.h"), "utf8");
  assert.match(source, /return mo_mul_hi_u64\(a, b\);/);
  assert.match(shared,
    /#if defined\(MOM_SYCL_PORTABLE_OPENCL\)\s+const uint32_t a0 = static_cast<uint16_t>\(a\), a1 = a >> 16,[\s\S]*?return p0 \+ \(\(p1 \+ p2\) << 16\) \+ \(p3 << 32\);/);
  assert.match(shared,
    /#if defined\(MOM_SYCL_ADAPTIVECPP\) \|\| defined\(MOM_SYCL_PORTABLE_OPENCL\)[\s\S]*?mo_mul_wide_u32\(a1, b1\)/);
});

test("HooHash verifies GPU filter candidates with the canonical host hash", () => {
  const source = fs.readFileSync(
    path.join(__dirname, "../sycl/hoohash/hoohash.cpp"), "utf8");
  const hostMath = fs.readFileSync(
    path.join(__dirname, "../sycl/hoohash/host_math.c"), "utf8");
  const binding = fs.readFileSync(path.join(__dirname, "../binding.gyp"), "utf8");
  const searchStart = source.indexOf("static sycl::event search(");
  const testStart = source.indexOf("static sycl::event test_hash(");
  const testEnd = source.indexOf("} // namespace mom_hoohash", testStart);
  assert.ok(searchStart >= 0 && testStart > searchStart && testEnd > testStart,
    "HooHash search/test functions must be inside the expected namespace");
  const testKernel = source.slice(testStart, testEnd);
  const searchKernel = source.slice(searchStart, testStart);
  assert.match(testKernel, /single_task<TestKernel>[\s\S]*?filter_hash\(/);
  assert.match(searchKernel, /filter_hash\([\s\S]*?meets_target\(/);
  assert.doesNotMatch(searchKernel, /canonical_hash\(/);
  assert.doesNotMatch(searchKernel, /\bis_test\b/);
  assert.match(source,
    /mom_hoohash_canonical_hash\(input, candidate_nonce, output\);[\s\S]*?if \(meets_target\(output, target\)\)/);
  assert.match(source,
    /fetch_min\(static_cast<uint32_t>\(id\)\)[\s\S]*?remaining -= consumed;[\s\S]*?first_nonce = candidate_nonce \+ 1;/);
  assert.match(source,
    /\+\+rejected_candidates == MAX_FILTER_RETRIES[\s\S]*?return 0;/);
  assert.match(hostMath,
    /return mom_hoohash_exp\(mom_hoohash_sin\(y\) \+ mom_hoohash_cos\(y\)\);[\s\S]*?return 1 \/ mom_hoohash_sqrt\(mom_hoohash_fabs\(y\) \+ 1\);/);
  assert.match(hostMath,
    /dlvsym\(RTLD_NEXT, #name, "GLIBC_2\.2\.5"\)[\s\S]*?pthread_once\(&math_once, resolve_math\);[\s\S]*?MOM_HOST_MATH_WRAPPER\(sin\)[\s\S]*?MOM_HOST_MATH_WRAPPER\(fabs\)/);
  assert.match(source,
    /if \(const char\* error = mom_hoohash_math_error\(\)\)[\s\S]*?throw std::string\(error\);/);
  assert.equal(binding.match(/"sycl\/hoohash\/host_math\.c"/g)?.length, 1);
  assert.match(binding,
    /"target_name": "hoohash_host"[\s\S]*?"-fno-fast-math"[\s\S]*?"\/fp:strict"[\s\S]*?"target_name": "sycl"[\s\S]*?"dependencies": \[ "hoohash_host" \]/);
  assert.match(hostMath,
    /if \(y == pi \/ 2 \|\| y == 3 \* pi \/ 2\)[\s\S]*?return 0;/);
});

test("Windows unified GPU workers link HooHash's strict host verifier", () => {
  const cuda = fs.readFileSync(path.join(__dirname,
    "../.github/workflows/scripts/build-sycl-cuda-win.ps1"), "utf8");
  const acpp = fs.readFileSync(path.join(__dirname,
    "../.github/workflows/scripts/build-sycl-adaptivecpp-win.ps1"), "utf8");
  for (const script of [cuda, acpp]) {
    assert.match(script, /-fno-fast-math[\s\S]*?-ffp-contract=off[\s\S]*?-fno-builtin/);
    assert.match(script, /sycl\\hoohash\\host_math\.c/);
    assert.match(script, /hoohash_host\.obj/);
  }
  assert.match(cuda, /\$objs \+= \$hoo[\s\S]*?"-shared" @objs/);
  assert.match(acpp, /\$objects \+= \$hoohashObject[\s\S]*?-shared @objects/);
});


test("Octopus native runtime failure disables repeated retries", () => {
  const source = fs.readFileSync(path.join(__dirname, "../sycl/octopus/nvidia_tensor.inc"), "utf8");
  assert.match(source, /if \(native_failed_\)\s*return false;/);
  assert.match(source, /native_failed_ = true;/);
});

test("Octopus NVIDIA DAG loads retain the measured cache policy", () => {
  const source = fs.readFileSync(path.join(__dirname, "../sycl/octopus/nvidia_tensor.inc"), "utf8");
  assert.match(source, /ld\.global\.L1::evict_first\.v4\.u32/);
});

test("Octopus portable OpenCL uses exact field multiply-add reduction", () => {
  const source = fs.readFileSync(path.join(__dirname, "../sycl/octopus/octopus.cpp"), "utf8");
  assert.match(source,
    /if constexpr \(mom_sycl_portable_opencl\) \{[\s\S]*?return mod_field\(static_cast<uint64_t>\(a\) \* b \+ c\);/);
  assert.match(source,
    /#if defined\(MOM_SYCL_ADAPTIVECPP\) \|\| defined\(MOM_SYCL_PORTABLE_OPENCL\)[\s\S]*?static_cast<uint64_t>\(static_cast<uint32_t>\(value\)\) \* RECIPROCAL/);
});

test("Verthash keeps the large portable CPU dataset in shared USM", () => {
  const source = fs.readFileSync(path.join(__dirname, "../sycl/verthash/verthash.cpp"), "utf8");
  assert.match(source,
    /const bool cpu_data = device\.is_cpu\(\);[\s\S]*?cpu_data \? sycl::malloc_shared<Uint2>[\s\S]*?: sycl::malloc_device<Uint2>/);
  assert.match(source,
    /if \(cpu_data\)\s+std::memcpy\(data, host\.data\(\), DATA_BYTES\);\s+else\s+sycl_wait_and_throw\(queue\.memcpy/);
});

test("Verthash defaults use safe Level Zero batches", () => {
  const intensity = fs.readFileSync(path.join(__dirname, "../sycl/intensity.inc"), "utf8");
  const start = intensity.indexOf("static unsigned verthash_intensity");
  const end = intensity.indexOf("\n}\n", start);
  assert.ok(start >= 0 && end > start, "Verthash intensity function must exist");
  const block = intensity.slice(start, end);
  assert.match(block, /sycl_is_level_zero_gpu\(dev\) && !is_integrated_gpu\(dev\)/);
  assert.match(block, /#ifdef _WIN32[\s\S]*return 1u << 18;[\s\S]*#else[\s\S]*return 1u << 19;/);
  assert.match(block, /return 1u << 16;/);
});

test("Xelis orders setup transfers and Windows ESIMD stages", () => {
  const source = fs.readFileSync(
    path.join(__dirname, "../sycl/xelishashv3/xelishashv3.cpp"), "utf8");
  const entry = source.slice(source.indexOf("int xelishashv3("));
  assert.match(entry,
    /sycl_wait_and_throw\(s\.queue\.memcpy\(s\.input,[\s\S]*?\n\s*sycl_wait_and_throw\(s\.queue\.memcpy\(s\.target,[\s\S]*?\n\s*sycl_wait_and_throw\(s\.queue\.memset\(s\.result,[\s\S]*?\n\s*const char\* configured/);
  assert.match(source,
    /#ifdef _WIN32\s+required = required \|\| use_esimd;\s+#endif/);
  assert.match(source,
    /fence<esimd::memory_kind::global, esimd::fence_flush_op::clean,\s+esimd::fence_scope::group>/);
});

test("Xelis portable OpenCL avoids vendor-specific 64-bit mul_hi", () => {
  const source = fs.readFileSync(
    path.join(__dirname, "../sycl/xelishashv3/xelishashv3.cpp"), "utf8");
  assert.match(source,
    /#ifdef MOM_XELISHASHV3_HOST_TEST\s+return static_cast<uint64_t>\(a\) \* b;\s+#else\s+return mo_mul_wide_u32\(a, b\);/);
  assert.match(source,
    /const uint32_t a0 = static_cast<uint32_t>\(a\), a1 = a >> 32,[\s\S]*?return mul_wide32\(a1, b1\)/);
  assert.match(source,
    /#if defined\(MOM_SYCL_ADAPTIVECPP\) \|\| defined\(MOM_SYCL_PORTABLE_OPENCL\)[\s\S]*?return mul_hi64_portable\(a, b\);[\s\S]*?#else\s+return sycl::mul_hi\(a, b\);/);
});

test("PearlHash emits subsequent winning proofs and preserves its seed lifecycle", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, () => {
  const core = fs.readFileSync(path.join(__dirname, "../native/core/execution.inc"), "utf8");
  const branch = core.match(
    /if\s*\(m_dev\s*==\s*DEV::PEARLHASH_GPU\)\s*\{\s*const\s+uint64_t\s+prev_nonce\s*=\s*m_nonce64;[\s\S]*?(?=\s*if\s*\(m_dev\s*==\s*DEV::C30_GPU\))/
  )?.[0];
  assert.ok(branch, "native PearlHash result branch is missing");
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-results-"));
  const source = path.join(fixture, "results.cpp");
  const executable = path.join(fixture, "results");
  fs.writeFileSync(source, `
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "native/job-boundary.h"
constexpr unsigned HASH_LEN = 32;
enum class DEV { PEARLHASH_GPU };
using MessageValues = std::map<std::string, std::string>;
static std::string nonce_to_hex(uint64_t nonce, unsigned width) {
  assert(width == 8);
  char hex[17];
  std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(nonce));
  return hex;
}
struct Worker {
  DEV m_dev = DEV::PEARLHASH_GPU;
  uint64_t m_nonce64 = 10, m_target = 1, found_seed = 0;
  unsigned m_thread_num = 2, m_pearlhash_seed_stride = 3, claim_calls = 0;
  std::string m_pool_id = "pool", m_worker_id = "worker", m_job_id = "job", m_job_token = "token";
  bool available = true, active = true;
  std::string proof;
  std::vector<std::pair<std::string, MessageValues>> sent;
  // Keep historical gate inputs in this mock so the regression also compiles before the fix.
  uint8_t m_input[76]{};
  unsigned m_input_len = 76, m_batch = 128, m_pearlhash_n = 128, m_pearlhash_k = 2048,
           m_pearlhash_rank = 128, m_pearlhash_cert_version = 3;
  unsigned m_pearlhash_proof_n = 0, m_pearlhash_proof_k = 0, m_pearlhash_proof_rank = 0,
           m_pearlhash_proof_cert_version = 0, m_pearlhash_proof_m = 0;
  std::string m_pearlhash_proof_pool, m_pearlhash_proof_job, m_pearlhash_proof_header;
  const char* pearlhash_claim(uint8_t* jackpot, uint32_t* factor) {
    ++claim_calls;
    if (!available) return nullptr;
    for (unsigned i = 0; i < HASH_LEN; ++i) jackpot[i] = static_cast<uint8_t>(found_seed);
    *factor = 524288;
    proof = "captured-proof-" + std::to_string(found_seed);
    return proof.c_str();
  }
  char* hash_bin2hex(const uint8_t* hash, char* hex, unsigned batch) {
    assert(batch == 0);
    for (unsigned i = 0; i < HASH_LEN; ++i) std::snprintf(hex + 2 * i, 3, "%02x", hash[i]);
    return hex;
  }
  void send_msg(const std::string& name, const MessageValues& values) { sent.emplace_back(name, values); }
  void send_error(const std::string& message) { send_msg("error", {{"message", message}}); }
  void clear_fn() { active = false; }
  void send_last_nonce(uint64_t nonce, unsigned width, const std::string& pool,
                       const std::string& job, const std::string& token) {
    send_msg("last_nonce", {{"nonce", nonce_to_hex(nonce, width)}, {"pool_id", pool},
                            {"job_id", job}, {"job_token", token}});
  }
  void run(int dev_sols, uint64_t dev_nonce) {
    found_seed = dev_nonce;
    for (unsigned once = 0; once < 1; ++once) {
      ${branch}
    }
  }
};
static void expect_result(const Worker& worker, unsigned index, uint64_t seed) {
  const auto& event = worker.sent.at(index);
  assert(event.first == "result" && event.second.size() == 8);
  const auto& result = event.second;
  assert(result.at("nonce") == nonce_to_hex(seed, 8));
  assert(result.at("plain_proof") == "captured-proof-" + std::to_string(seed));
  char byte[3];
  std::snprintf(byte, sizeof(byte), "%02x", static_cast<unsigned>(seed & 255));
  std::string jackpot;
  for (unsigned i = 0; i < HASH_LEN; ++i) jackpot += byte;
  assert(result.at("jackpot") == jackpot && result.at("adjustment_factor") == "524288");
  assert(result.at("pool_id") == worker.m_pool_id && result.at("worker_id") == worker.m_worker_id);
  assert(result.at("job_id") == worker.m_job_id && result.at("job_token") == worker.m_job_token);
}
int main() {
  Worker wins;
  wins.run(1, 10);
  wins.run(1, 16);
  assert(wins.sent.size() == 2 && wins.claim_calls == 2 && wins.m_nonce64 == 22 && wins.active);
  expect_result(wins, 0, 10);
  expect_result(wins, 1, 16);
  Worker retry;
  retry.available = false;
  retry.run(1, 10);
  retry.available = true;
  retry.run(1, 16);
  assert(retry.sent.size() == 2 && retry.claim_calls == 2 && retry.m_nonce64 == 22 && retry.active);
  assert(retry.sent[0].first == "error");
  assert(retry.sent[0].second.at("message") == "PearlHash proof claim unavailable");
  expect_result(retry, 1, 16);
  for (int solutions : {-1, 0, 2}) {
    Worker pending;
    pending.run(solutions, 10);
    assert(pending.sent.empty() && pending.claim_calls == 0 && pending.m_nonce64 == 16 && pending.active);
  }
  Worker no_target;
  no_target.m_target = 0;
  no_target.run(1, 10);
  assert(no_target.sent.empty() && no_target.claim_calls == 0 && no_target.m_nonce64 == 16);
  Worker exhausted;
  exhausted.m_nonce64 = UINT32_MAX - 5;
  exhausted.run(1, exhausted.m_nonce64);
  assert(!exhausted.active && exhausted.sent.size() == 2 && exhausted.claim_calls == 1);
  expect_result(exhausted, 0, UINT32_MAX - 5);
  assert(exhausted.sent[1].first == "last_nonce");
  assert(exhausted.sent[1].second.at("nonce") == nonce_to_hex(UINT32_MAX - 5, 8));
  assert(exhausted.sent[1].second.at("pool_id") == "pool" &&
         exhausted.sent[1].second.at("job_id") == "job" && exhausted.sent[1].second.at("job_token") == "token");
  no_target.m_nonce64 = UINT32_MAX - 5;
  no_target.run(1, no_target.m_nonce64);
  assert(!no_target.active && no_target.sent.empty());
  std::puts("PASS PearlHash subsequent proofs, claims and seed lifecycle");
}
`);
  try {
    const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      "-pedantic", "-I", path.join(__dirname, ".."), source, "-o", executable], {encoding: "utf8"});
    assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
    const result = spawnSync(executable, [], {encoding: "utf8"});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
    assert.equal(result.stdout, "PASS PearlHash subsequent proofs, claims and seed lifecycle\n");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("CN/gpu completes kernels before readback without startup or stale-batch pacing", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, () => {
  const cn = path.join(__dirname, "../sycl/cn_gpu");
  const entry = fs.readFileSync(path.join(cn, "entry.inc"), "utf8");
  const finalMarker = "  sycl::event final_event =";
  assert.equal(entry.split(finalMarker).length, 2, "one final-kernel submission");
  const finalStart = entry.indexOf(finalMarker);
  const submitEnd = entry.indexOf("\n      });", finalStart);
  const end = entry.lastIndexOf("\n}");
  assert.ok(submitEnd > finalStart && end > submitEnd && entry.slice(end).trim() === "}");
  const tail = entry.slice(submitEnd + "\n      });".length, end)
    .replaceAll("std::chrono::steady_clock::now()", "fixture::clock::now()")
    .replaceAll("std::this_thread::sleep_until", "fixture::sleep_until");
  const finalBodyStart = entry.indexOf("uint64_t* const spad = &d_spads[25 * t];", finalStart);
  const finalBodyEnd = entry.indexOf("out[3] = spad[3];", finalBodyStart);
  assert.ok(finalBodyStart > finalStart && finalBodyEnd > finalBodyStart && finalBodyEnd < submitEnd);
  const finalBody = entry.slice(finalBodyStart, finalBodyEnd + "out[3] = spad[3];".length);
  const starts = [...entry.matchAll(/const auto batch_start\s*=\s*std::chrono::steady_clock::now\(\);[^\n]*/g)];
  assert.ok(starts.length <= 1);
  const startMatch = starts[0];
  const batchStart = startMatch ? startMatch[0]
    .replaceAll("std::chrono::steady_clock::now()", "fixture::clock::now()") : "";
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cn-readback-"));
  const source = path.join(fixture, "readback.cpp");
  const executable = path.join(fixture, "readback");
  try {
    fs.writeFileSync(source, String.raw`
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Integer intrinsics for the production Keccak, not a replacement hash implementation.
namespace sycl {
using uint4 = std::array<uint32_t, 4>;
inline uint64_t rotate(const uint64_t value, const uint64_t amount) {
  return std::rotl(value, static_cast<int>(amount & 63U));
}
inline uint64_t bitselect(const uint64_t a, const uint64_t b, const uint64_t c) {
  return (a & ~c) | (b & c);
}
} // namespace sycl

inline constexpr unsigned cn_gpu_lanes_per_hash = 16;
template <typename T> inline T mo_rotate(const T value, const T amount) {
  return std::rotl(value, static_cast<int>(amount & (sizeof(T) * 8 - 1)));
}
template <typename T> inline T mo_bitselect(const T a, const T b, const T c) {
  return (a & ~c) | (b & c);
}
#include "crypto.inc"

namespace fixture {
inline int64_t now_us = 0;
inline unsigned sleep_calls = 0;
struct clock {
  static std::chrono::steady_clock::time_point now() {
    return std::chrono::steady_clock::time_point{std::chrono::microseconds{now_us}};
  }
};
inline void sleep_until(const std::chrono::steady_clock::time_point deadline) {
  ++sleep_calls;
  now_us = std::max(now_us, std::chrono::duration_cast<std::chrono::microseconds>(
                                deadline.time_since_epoch()).count());
}

constexpr size_t HASH_LEN = 32;
enum class Failure { none, final_pending, final_retired, copy_submit, copy_pending, copy_retired,
                     buffer_read };
enum class Kind { final, copy };
struct Context {
  unsigned batch;
  Failure failure;
  bool kernels_done = false;
  bool copy_done = false;
  bool consumed = false;
  bool kernels_ready_at_copy = true;
  unsigned final_waits = 0;
  unsigned copy_submits = 0;
  unsigned copy_waits = 0;
  unsigned buffer_reads = 0;
  alignas(uint64_t) std::array<uint8_t, 4 * HASH_LEN> device_bytes{};
  std::array<uint64_t, 4 * 25> spads{};
  uint8_t* copy_destination = nullptr;
  const uint8_t* copy_source = nullptr;
  size_t copy_bytes = 0;

  void finish_kernels() {
    if (kernels_done)
      return;
    if (failure == Failure::final_pending)
      throw std::runtime_error("final-pending");
    auto* const d_spads = spads.data();
    auto* const d_outputs = device_bytes.data();
    for (unsigned t = 0; t < batch; ++t) {
${finalBody}
    }
    kernels_done = true;
    now_us += 300;
    if (failure == Failure::final_retired)
      throw std::runtime_error("final-retired");
  }
};
struct Event { Context* context; Kind kind; };
struct Queue {
  Context* context;
  Event memcpy(uint8_t* destination, const uint8_t* source, const size_t bytes) {
    ++context->copy_submits;
    context->kernels_ready_at_copy &= context->kernels_done;
    if (context->failure == Failure::copy_submit)
      throw std::runtime_error("copy-submit");
    context->copy_destination = destination;
    context->copy_source = source;
    context->copy_bytes = bytes;
    return {context, Kind::copy};
  }
};
struct Buffer {
  Context* context;
  void read(uint8_t* destination, const size_t bytes) {
    ++context->buffer_reads;
    context->kernels_ready_at_copy &= context->kernels_done;
    if (context->failure == Failure::buffer_read)
      throw std::runtime_error("buffer-read");
    std::memcpy(destination, context->device_bytes.data(), bytes);
    context->copy_done = true;
  }
};
struct State {
  int device = 0;
  bool shared_io;
  // Superset lets the same fixture execute historical and fixed source blocks.
  double wait_ema_us = 0;
  bool wait_warmup_done = false;
  Buffer buffered_outputs;
};

inline void sycl_wait_and_throw(const Event event, const int) {
  auto& context = *event.context;
  if (event.kind == Kind::final) {
    ++context.final_waits;
    context.finish_kernels();
    return;
  }
  ++context.copy_waits;
  // An in-order queue eventually retires earlier kernels, but this cannot repair a
  // host driver that blocks while submitting the copy, before returning this event.
  context.finish_kernels();
  if (context.failure == Failure::copy_pending)
    throw std::runtime_error("copy-pending");
  std::memcpy(context.copy_destination, context.copy_source, context.copy_bytes);
  context.copy_done = true;
  now_us += 30;
  if (context.failure == Failure::copy_retired)
    throw std::runtime_error("copy-retired");
}

template <bool mom_sycl_portable_opencl>
void readback(State& state, Context& context, uint8_t* output, const size_t output_bytes) {
  Queue q{&context};
  [[maybe_unused]] auto* const d_outputs = context.device_bytes.data();
  const Event final_event{&context, Kind::final};
${batchStart}
${tail}
}

unsigned checks = 0;
unsigned failures = 0;
void require(const bool condition, const char* const route, const char* const label) {
  ++checks;
  if (condition)
    return;
  ++failures;
  std::fprintf(stderr, "FAIL %s: %s\n", route, label);
}

template <bool Portable>
void run(const bool shared, const unsigned batch, const Failure failure,
         const double stale_ema, const bool warmup) {
  const char* const route = Portable ? "portable-buffer" : shared ? "shared-USM" : "device-USM";
  now_us = 0;
  sleep_calls = 0;
  Context context{batch, failure};
  context.device_bytes.fill(0xd7);
  std::array<uint8_t, 4 * HASH_LEN + 16> output;
  output.fill(0x5a);
  const auto untouched = output;
  // Varied nonce-bearing inputs seed existing Keccak state; CN's preceding GPU
  // arithmetic is deliberately not modeled or claimed validated by this test.
  std::array<uint8_t, 4 * 80> inputs{};
  for (size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = static_cast<uint8_t>((i * 37 + batch * 11) & 255);
  const auto original_inputs = inputs;
  for (size_t i = 0; i < context.spads.size(); ++i)
    context.spads[i] = uint64_t{inputs[i % inputs.size()]} * 0x0101010101010101ULL + i;
  auto oracle = context.spads;
  std::array<uint8_t, 4 * HASH_LEN> expected{};
  for (unsigned t = 0; t < batch; ++t) {
    keccak(oracle.data() + 25 * t);
    std::memcpy(expected.data() + HASH_LEN * t, oracle.data() + 25 * t, HASH_LEN);
  }
  State state{0, shared, stale_ema, warmup, Buffer{&context}};
  std::string error;
  try {
    readback<Portable>(state, context, output.data(), HASH_LEN * batch);
    context.consumed = true;
  } catch (const std::exception& e) {
    error = e.what();
  }
  require(context.final_waits == 1, route, "exactly one final-kernel wait");
  require(context.kernels_ready_at_copy, route, "kernels complete before any readback submission");
  require(sleep_calls == 0, route, "no stale-EMA pre-read sleep");
  require(state.wait_ema_us == stale_ema && state.wait_warmup_done == warmup,
          route, "readback has no EMA/warmup state mutation");
  require(inputs == original_inputs, route, "nonce-bearing input bytes unchanged");
  require(std::equal(output.begin() + HASH_LEN * batch, output.end(),
                     untouched.begin() + HASH_LEN * batch), route, "padding and output guard unchanged");
  if (failure == Failure::none) {
    require(error.empty() && context.consumed, route, "successful result consumed");
    require(context.kernels_done, route, "final kernel finished");
    require(std::equal(output.begin(), output.begin() + HASH_LEN * batch, expected.begin()),
            route, "production final-Keccak/hash output bytes unchanged");
  } else {
    const char* const expected_error = failure == Failure::final_pending ? "final-pending" :
        failure == Failure::final_retired ? "final-retired" :
        failure == Failure::copy_submit ? "copy-submit" :
        failure == Failure::copy_pending ? "copy-pending" :
        failure == Failure::copy_retired ? "copy-retired" : "buffer-read";
    require(error == expected_error && !context.consumed, route, "original error prevents consumption");
    if (failure != Failure::copy_retired)
      require(output == untouched, route, "failure before readback leaves host output untouched");
    if (failure == Failure::final_pending || failure == Failure::final_retired)
      require(context.copy_submits == 0 && context.buffer_reads == 0,
              route, "final-kernel error prevents all readback");
  }
  if constexpr (Portable) {
    if (failure == Failure::none || failure == Failure::buffer_read)
      require(context.buffer_reads == 1, route, "one buffered host read");
    require(context.copy_submits == 0 && context.copy_waits == 0, route, "no USM copy for buffer route");
  } else if (shared) {
    require(context.copy_submits == 0 && context.copy_waits == 0 && context.buffer_reads == 0,
            route, "shared route uses host memcpy only");
  } else {
    if (failure == Failure::none || failure == Failure::copy_pending || failure == Failure::copy_retired)
      require(context.copy_submits == 1 && context.copy_waits == 1,
              route, "deferred device copy waited exactly once");
    if (failure == Failure::copy_submit)
      require(context.copy_submits == 1 && context.copy_waits == 0,
              route, "submission failure cannot wait a nonexistent event");
    if (failure == Failure::none)
      require(context.copy_done, route, "device copy completed before consumption");
  }
}
} // namespace fixture

int main() {
  using namespace fixture;
  for (const unsigned batch : {1U, 2U, 3U, 4U}) {
    for (const double ema : {0.0, 8000000.0}) {
      for (const bool warmup : {false, true}) {
        run<false>(false, batch, Failure::none, ema, warmup);
        run<false>(true, batch, Failure::none, ema, warmup);
        run<true>(false, batch, Failure::none, ema, warmup);
        run<true>(true, batch, Failure::none, ema, warmup);
      }
    }
  }
  for (const Failure failure : {Failure::final_pending, Failure::final_retired}) {
    run<false>(false, 3, failure, 0, false);
    run<false>(true, 3, failure, 0, false);
    run<true>(false, 3, failure, 0, false);
    run<true>(true, 3, failure, 0, false);
  }
  for (const Failure failure : {Failure::copy_submit, Failure::copy_pending, Failure::copy_retired})
    run<false>(false, 3, failure, 0, false);
  run<true>(false, 3, Failure::buffer_read, 0, false);
  std::printf("CN/GPU source-derived readback host fixture: %u checks, %u failures\n", checks, failures);
  return failures ? 1 : 0;
}

`);
    const compiled = spawnSync("c++", ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
      "-pedantic", "-I", cn, source, "-o", executable], {encoding: "utf8"});
    assert.equal(compiled.error, undefined);
    assert.equal(compiled.signal, null);
    assert.equal(compiled.status, 0, compiled.stderr);
    const result = spawnSync(executable, [], {encoding: "utf8"});
    assert.equal(result.error, undefined);
    assert.equal(result.signal, null);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, "CN/GPU source-derived readback host fixture: 802 checks, 0 failures\n");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("PearlHash Intel SYCL pacing progresses before low-CPU completion polling", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, () => {
  const host = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/host.inc"), "utf8");
  const startMarker = "    attempt_start = std::chrono::steady_clock::now();\n";
  assert.equal(host.split(startMarker).length, 2);
  const start = host.indexOf(startMarker) + startMarker.length;
  const end = host.indexOf("    st.have_header = true;", start);
  assert.ok(end > start, "complete PearlHash pacing/wait block is missing");
  const block = host.slice(start, end)
    .replaceAll("std::chrono::", "fixture::chrono::")
    .replaceAll("std::this_thread::", "fixture::this_thread::");
  const internal = fs.readFileSync(path.join(__dirname, "../sycl/lib-internal.h"), "utf8");
  const waitStartMarker = "inline void sycl_wait_and_throw(";
  assert.equal(internal.split(waitStartMarker).length, 2);
  const waitStart = internal.indexOf(waitStartMarker);
  const waitEnd = internal.indexOf("\ninline void sycl_log_cleanup_exception", waitStart);
  assert.ok(waitEnd > waitStart, "complete shared SYCL wait helper is missing");
  const wait = internal.slice(waitStart, waitEnd)
    .replaceAll("std::chrono::", "fixture::chrono::")
    .replaceAll("std::this_thread::", "fixture::this_thread::");
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-pacing-"));
  const source = path.join(fixture, "pacing.cpp");
  const executable = path.join(fixture, "pacing");
  fs.writeFileSync(source, `
#include <cassert>
#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <stdexcept>
#include <vector>
#if !defined(MOM_FIXTURE_NO_BARRIER)
#define SYCL_EXT_ONEAPI_ENQUEUE_BARRIER 1
#endif
namespace sycl::info::device { struct vendor_id {}; }
namespace sycl::info::event { struct command_execution_status {}; }
namespace sycl::info { enum class event_command_status { submitted, complete }; }
struct EventState {
  unsigned pending = 2, queries = 0, waits = 0;
  bool active = false, progressed = false, complete = false, completed_before_wait = false;
  bool status_error = false, sleep_error = false, wait_error = false;
  unsigned* host_destination = nullptr;
};
static EventState event_state;
static int error_kind;
static std::exception_ptr polling_error;
static void fail_polling(const char* message) {
  try {
    if (error_kind == 1)
      throw std::string(message);
    if (error_kind == 2)
      throw 73;
    throw std::runtime_error(message);
  } catch (...) {
    polling_error = std::current_exception();
    throw;
  }
}
enum class PearlHashSearchBackend { sycl, hip_jit, cuda_jit };
struct Device {
  bool gpu; unsigned vendor;
  bool opencl = false;
  bool is_gpu() const { return gpu; }
  template<class T> unsigned get_info() const { return vendor; }
};
struct Event {
  template<class T> sycl::info::event_command_status get_info() {
    ++event_state.queries;
    if (event_state.status_error)
      fail_polling("status-query");
    event_state.progressed = true;
    if (event_state.pending) {
      --event_state.pending;
      return sycl::info::event_command_status::submitted;
    }
    event_state.complete = true;
    return sycl::info::event_command_status::complete;
  }
  void wait_and_throw() {
    ++event_state.waits;
    event_state.completed_before_wait = event_state.complete;
    event_state.complete = true;
    if (event_state.host_destination)
      *event_state.host_destination = 1;
    if (event_state.wait_error)
      throw std::runtime_error("final-wait");
  }
};
namespace sycl { using event = Event; using device = Device; }
static constexpr bool mom_sycl_portable_opencl =
#if defined(MOM_FIXTURE_PORTABLE_OPENCL)
  true;
#else
  false;
#endif
static bool mom_is_opencl(const Device& device) { return device.opencl; }
struct Queue {
  Device device; unsigned waits = 0, barriers = 0;
  Device get_device() const { return device; }
  void wait_and_throw() { ++waits; }
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER)
  Event ext_oneapi_submit_barrier() {
    ++barriers;
    event_state.active = true;
    return {};
  }
#endif
};
struct Jit {
  unsigned waits = 0;
  void wait(double) { ++waits; }
};
struct State { double wait_ema_us; Jit hip{}, cuda{}; bool cuda_tensor = false; };
namespace fixture {
namespace chrono {
using microseconds = std::chrono::microseconds;
long elapsed;
struct steady_clock {
  using time_point = std::chrono::steady_clock::time_point;
  static time_point now() { return time_point(microseconds(elapsed)); }
};
template<class T, class U> T duration_cast(U value) { return std::chrono::duration_cast<T>(value); }
}
namespace this_thread {
unsigned calls; long duration;
std::vector<long> durations;
void sleep_for(chrono::microseconds value) {
  ++calls;
  duration = value.count();
  durations.push_back(duration);
  if (event_state.sleep_error)
    fail_polling("sleep");
  if (event_state.active) {
    assert(event_state.progressed && event_state.queries > 0);
    assert(duration == 100);
  }
}
}
}
${wait}
static void run(PearlHashSearchBackend search_backend, bool gpu, unsigned vendor,
                double ema, long elapsed, long sleep, bool opencl = false) {
  fixture::chrono::elapsed = elapsed;
  fixture::this_thread::calls = 0;
  fixture::this_thread::duration = 0;
  fixture::this_thread::durations.clear();
  event_state = {};
  auto attempt_start = fixture::chrono::steady_clock::time_point{};
  Queue q{{gpu, vendor, opencl}};
  State st{ema};
${block}
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER)
  const bool intel_poll = search_backend == PearlHashSearchBackend::sycl && gpu &&
    vendor == 0x8086 && !opencl;
#else
  const bool intel_poll = false;
#endif
  const std::vector<long> expected_sleeps = intel_poll ? std::vector<long>{100, 100} :
    sleep > 0 ? std::vector<long>{sleep} : std::vector<long>{};
  assert(fixture::this_thread::durations == expected_sleeps);
  assert(fixture::this_thread::calls == expected_sleeps.size());
  assert(q.barriers == static_cast<unsigned>(intel_poll));
  assert(q.waits == static_cast<unsigned>(search_backend == PearlHashSearchBackend::sycl && !intel_poll));
  assert(event_state.queries == (intel_poll ? 3u : 0u));
  assert(event_state.waits == static_cast<unsigned>(intel_poll));
  assert(!intel_poll || event_state.completed_before_wait);
  assert(st.hip.waits == static_cast<unsigned>(search_backend == PearlHashSearchBackend::hip_jit));
  assert(st.cuda.waits == static_cast<unsigned>(search_backend == PearlHashSearchBackend::cuda_jit));
}
static void check_helper(bool gpu, unsigned pending, const std::string& failure = {},
                         int kind = 0, bool cleanup_error = false, bool opencl = false) {
  event_state = {};
  event_state.active = true;
  event_state.pending = pending;
  event_state.status_error = failure == "status-query";
  event_state.sleep_error = failure == "sleep";
  event_state.wait_error = cleanup_error || failure == "final-wait";
  unsigned host_destination = 0;
  event_state.host_destination = &host_destination;
  error_kind = kind;
  polling_error = {};
  fixture::this_thread::calls = 0;
  fixture::this_thread::durations.clear();
  bool committed = false;
  std::exception_ptr caught;
  try {
    sycl_wait_and_throw(Event{}, Device{gpu, 0x8086, opencl});
    committed = true;
  } catch (...) {
    caught = std::current_exception();
  }
  assert(static_cast<bool>(caught) == !failure.empty());
  assert(committed == failure.empty());
  if (polling_error) {
    assert(caught == polling_error);
  } else if (caught) {
    try {
      std::rethrow_exception(caught);
    } catch (const std::runtime_error& error) {
      assert(error.what() == failure);
    }
  }
  const bool polling = gpu && !(mom_sycl_portable_opencl && opencl);
  const unsigned sleeps = !polling || event_state.status_error ? 0 :
      event_state.sleep_error ? 1 : pending;
  assert(fixture::this_thread::calls == sleeps);
  assert(fixture::this_thread::durations == std::vector<long>(sleeps, 100));
  assert(event_state.queries == (polling ? polling_error ? 1u : pending + 1 : 0u));
  assert(event_state.waits == 1 && event_state.complete && host_destination == 1);
  assert(!polling || polling_error || event_state.completed_before_wait);
}
int main() {
  using B = PearlHashSearchBackend;
  const struct { B backend; bool gpu; unsigned vendor; double ema; long elapsed, sleep; } cases[] = {
    {B::sycl, true, 0x8086, 10000, 0, 0},
    {B::sycl, true, 0x8086, 2001, 0, 0},
    {B::sycl, true, 0x8086, 2000, 0, 0},
    {B::sycl, true, 0x8086, 0, 0, 0},
    {B::sycl, false, 0x8086, 10000, 0, 9000},
    {B::sycl, true, 0x1002, 10000, 0, 9000},
    {B::sycl, true, 0x10de, 10000, 0, 9000},
    {B::sycl, true, 0, 10000, 0, 9000},
    {B::hip_jit, true, 0x8086, 10000, 0, 9000},
    {B::cuda_jit, true, 0x8086, 10000, 0, 9000},
    {B::sycl, true, 0x1002, 2000, 0, 0},
    {B::sycl, true, 0x1002, 2001, 0, 1800},
    {B::sycl, true, 0x1002, 10000, 9000, 0},
    {B::sycl, true, 0x1002, 10000, 8999, 1},
  };
  for (const auto& test : cases)
    run(test.backend, test.gpu, test.vendor, test.ema, test.elapsed, test.sleep);
  run(B::sycl, true, 0x8086, 10000, 0, 0, true);
  run(B::sycl, true, 0x1002, 10000, 0, 9000, true);
  run(B::sycl, false, 0x8086, 10000, 0, 9000, true);
  check_helper(true, 0);
  check_helper(true, 3);
  check_helper(false, 3);
  check_helper(true, 0, "final-wait");
  check_helper(false, 3, "final-wait");
  check_helper(true, 3, {}, 0, false, true);
  check_helper(true, 3, "final-wait", 0, false, true);
  for (int kind : {0, 1, 2}) {
    for (bool cleanup_error : {false, true}) {
      check_helper(true, 3, "status-query", kind, cleanup_error);
      check_helper(true, 3, "sleep", kind, cleanup_error);
    }
  }
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER)
  std::printf("PASS PearlHash actual-source pacing/polling: 17 pacing + 19 helper cases; barrier=available; ");
#else
  std::printf("PASS PearlHash actual-source pacing/polling: 17 pacing + 19 helper cases; barrier=absent; ");
#endif
  std::printf("portable_opencl=%d\\n", mom_sycl_portable_opencl);
}
`);
  try {
    for (const available of [true, false]) {
      for (const portable of [false, true]) {
        const flags = [
          ...(available ? [] : ["-DMOM_FIXTURE_NO_BARRIER=1"]),
          ...(portable ? ["-DMOM_FIXTURE_PORTABLE_OPENCL=1"] : []),
        ];
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", ...flags, source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.signal, null);
        assert.equal(compiled.status, 0, compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8"});
        assert.equal(result.error, undefined);
        assert.equal(result.signal, null);
        assert.equal(result.status, 0, result.stderr);
        assert.equal(result.stdout, "PASS PearlHash actual-source pacing/polling: 17 pacing + 19 helper cases; " +
          "barrier=" + (available ? "available" : "absent") +
          "; portable_opencl=" + Number(portable) + "\n");
      }
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("DPC++ final addon and Windows SYCL links split device code per kernel", () => {
  const build = fs.readFileSync(path.join(__dirname, "../binding.gyp"), "utf8");
  const momStart = build.indexOf('"target_name": "mom"');
  const syclStart = build.indexOf('"target_name": "sycl"', momStart);
  const syclEnd = build.indexOf('"target_name":', syclStart + 1);
  assert(momStart >= 0 && syclStart > momStart && syclEnd > syclStart);
  const finalAddon = build.slice(momStart, syclStart);
  const syclDll = build.slice(syclStart, syclEnd);
  assert.match(finalAddon,
    /mom_sycl_impl=='dpcpp' or mom_sycl_impl=='dpcpp-combined'[\s\S]*?"ldflags\+":\s*\[ "-fsycl-device-code-split=per_kernel" \]/);
  const linker = syclDll.indexOf('"VCLinkerTool"');
  assert(linker >= 0);
  assert.match(syclDll.slice(linker),
    /"AdditionalOptions":\s*\[\s*"\/DLL",\s*"\/fsycl",\s*"\/clang:-fsycl-device-code-split=per_kernel"/);
});

test("GPU availability detection preserves actionable SYCL diagnostics", () => {
  /** @param {string} stderr */
  const result = (stderr) => ({code: 1, signal: null, error: null, stdout: "", stderr});
  assert.equal(isMissingGpuOutput(result("No device of requested type was found")), true);
  assert.equal(isMissingGpuOutput(result("No SYCL GPU device is available")), true);
  assert.equal(isMissingGpuOutput(result("SYCL GPU device unavailable")), true);
  assert.equal(isMissingGpuOutput(result("No GPUs detected")), true);
  assert.equal(isMissingGpuOutput(result("XelisHashV3 SYCL error on device submission")), false);
  assert.equal(isMissingGpuOutput(result("No GPU kernel was emitted after compiler error")), false);
});

test("PearlHash routes integrated Intel GPUs away from ESIMD prefetch", () => {
  const dispatch = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/dispatch.inc"), "utf8");
  assert.match(dispatch,
    /const bool discrete_gpu = device\.is_gpu\(\) && !is_integrated_gpu\(device\);/);
  assert.match(dispatch,
    /pearlhash_esimd_route\(tuned_sycl, supported_backend, discrete_gpu, intel_matrix, width\)/);
  assert.match(dispatch, /use_portable = !cuda_sycl && esimd_width == 0u;/);
  assert(dispatch.indexOf("pearlhash_esimd_route(") < dispatch.indexOf("compute_ab("),
    "ESIMD capability gating must precede matrix layout selection");
});

test("PearlHash discovery emits complete measured profiles with a low-memory fallback", () => {
  const intensity = fs.readFileSync(path.join(__dirname, "../sycl/intensity.inc"), "utf8");
  const params = fs.readFileSync(path.join(__dirname, "../sycl/algo_params.inc"), "utf8");
  assert.ok(intensity.includes("global_mem_size>() >= 8 * GiB"));
  assert.ok(intensity.includes("pearlhash_profile_t{131072, 524288, 8192, 128}"));
  assert.ok(intensity.includes("pearlhash_profile_t{65536, 65536, 4096, 256}"));
  assert.ok(params.includes("\"*[m=\" + std::to_string(profile.m)"));
  for (const field of ["n", "k", "rank"]) {
    assert.ok(params.includes(`";${field}=" + std::to_string(profile.${field})`));
  }
});

test("compiler policy cells reject malformed and duplicate entries", () => {
  const parsed = policyModule.parse(policyFixture("dpcpp", "—"));
  const parsedRow = parsed.policies[0];
  assert.ok(parsedRow);
  assert.deepEqual(parsedRow.overrides, {foo: {compiler: "dpcpp"}});
  assert.throws(() => policyModule.parse(policyFixture("dpcpp=extra", "—")),
    /Unknown GPU compiler key: dpcpp=extra/);
  assert.throws(() => policyModule.parse(policyFixture("", "")),
    /GPU override row is empty/);
  assert.throws(() => policyModule.parse(
    `${policyFixture("dpcpp", "—")}\n| foo | Linux | Intel | — | sycl |`
  ), /Duplicate GPU override/);
  const absent = policyModule.parse(policyFixture()).policies[0];
  const dashedCompiler = policyModule.parse(policyFixture("dpcpp", "-")).policies[0];
  const dashedBackend = policyModule.parse(policyFixture("-", "sycl")).policies[0];
  assert.ok(absent && dashedCompiler && dashedBackend);
  assert.deepEqual(absent.overrides, {});
  assert.deepEqual(dashedCompiler.overrides, {foo: {compiler: "dpcpp"}});
  assert.deepEqual(dashedBackend.overrides, {foo: {backend: "sycl"}});

  const duplicateArtifact = policyFixture().replace(
    "| dpcpp | dpcpp/mom.node | dpcpp/mom.node |",
    "| dpcpp | dpcpp/mom.node | dpcpp/mom.node |\n" +
    "| dpcpp | other/mom.node | other/mom.node |"
  );
  assert.throws(() => policyModule.parse(duplicateArtifact), /duplicate GPU artifact key/i);
  const duplicatePolicy = policyFixture().replace(
    "| Linux | Intel | dpcpp | sycl | — |",
    "| Linux | Intel | dpcpp | sycl | — |\n| Linux | Intel | dpcpp | sycl | — |"
  );
  assert.throws(() => policyModule.parse(duplicatePolicy), /Duplicate GPU policy row/);
  assert.throws(() => policyModule.parse(policyFixture("missing", "—")),
    /Unknown GPU compiler key: missing/);
  assert.throws(() => policyModule.parse(policyFixture("—", "invalid")),
    /Invalid GPU backend/);
});

test("PearlHash profiles and native shapes enforce safe individual and relational bounds", () => {
  const maxDimension = 1 << 24;
  const maxM = Math.floor(0x7fffffff / 2048 / 32) * 32;
  const maxN = Math.floor(0x7fffffff / 128 / 32) * 32;
  assert.deepEqual(gpuTuning.validatePearlHashShape("128", 128, 2048, 128), {
    m: 128, n: 128, k: 2048, rank: 128,
  });
  assert.deepEqual(gpuTuning.validatePearlHashShape(160, 160, 8192, 128), {
    m: 160, n: 160, k: 8192, rank: 128,
  });
  assert.deepEqual(gpuTuning.validatePearlHashShape(131072, 524288, 8192, 128), {
    m: 131072, n: 524288, k: 8192, rank: 128,
  });
  assert.deepEqual(gpuTuning.validatePearlHashShape(maxM, 128, 2048, 128), {
    m: maxM, n: 128, k: 2048, rank: 128,
  });
  assert.deepEqual(gpuTuning.validatePearlHashShape(128, maxN, 2048, 128), {
    m: 128, n: maxN, k: 2048, rank: 128,
  });
  assert.deepEqual(gpuTuning.validatePearlHashShape(128, 128, 65536, 1024), {
    m: 128, n: 128, k: 65536, rank: 1024,
  });
  /** @type {Array<[unknown, unknown, unknown, unknown]>} */
  const invalidShapes = [
    [96, 128, 2048, 128],
    [161, 128, 2048, 128],
    [maxDimension + 32, 128, 2048, 128],
    [128, 96, 2048, 128],
    [128, 161, 2048, 128],
    [128, 128, 960, 128],
    [128, 128, 1024, 128],
    [128, 128, 1025, 128],
    [128, 128, 65600, 1024],
    [128, 128, 16384, 2048],
    [128, 128, 3072, 192],
    [maxM + 32, 128, 2048, 128],
    [128, maxN + 32, 2048, 128],
    [131072, 4194304, 2048, 128],
    [{toString: () => "128"}, 128, 2048, 128],
  ];
  for (const shape of invalidShapes) {
    assert.throws(() => Reflect.apply(gpuTuning.validatePearlHashShape, null, shape),
      /PearlHash shape/);
  }
  assert.throws(() => policyModule.parse(policyFixture("—", "—", "128x128x16/16")),
    /PearlHash profile/);
  assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[m=96]", "pearlhash"),
    /at least 128/);
  assert.deepEqual(gpuTuning.parseDeviceEntry(`gpu1*[m=${maxDimension}]`, "pearlhash").tuning,
    {m: maxDimension});
  assert.throws(() => gpuTuning.parseDeviceEntry(`gpu1*[m=${maxDimension + 32}]`, "pearlhash"),
    /at most 16777216/);
  assert.throws(() => gpuTuning.parseDeviceEntry(`gpu1*[n=${maxDimension + 32}]`, "pearlhash"),
    /at most 16777216/);
  assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[m=161]", "pearlhash"),
    /multiple of 32/);
  assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[k=960]", "pearlhash"),
    /at least 1024/);
  assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[k=1024]", "pearlhash").tuning,
    {k: 1024});
  assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[k=65536]", "pearlhash").tuning,
    {k: 65536});
  assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[k=65600]", "pearlhash"),
    /at most 65536/);
  assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[rank=1024]", "pearlhash").tuning,
    {rank: 1024});
  assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[rank=48]", "pearlhash"),
    /between 128 and 1024/);
  assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[rank=64]", "pearlhash"),
    /between 128 and 1024/);
  assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[rank=2048]", "pearlhash"),
    /between 128 and 1024/);
  assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[k=4160;rank=256]", "pearlhash").tuning,
    {k: 4160, rank: 256});
  assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[k=2048;rank=128]", "pearlhash").tuning,
    {k: 2048, rank: 128});
  assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[k=8192;rank=128]", "pearlhash").tuning,
    {k: 8192, rank: 128});
  assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[k=16640;rank=256]", "pearlhash").tuning,
    {k: 16640, rank: 256});
});

test("GPU backend and compiler policy boundaries reject prefixes and coercive objects", () => {
  assert.equal(policyModule.gpuFromEnv({MOM_GPU_BACKEND: "nvidia-extra"}), "");
  assert.equal(policyModule.gpuFromEnv({MOM_GPU_BACKEND: "NVIDIA"}), "nvidia");
  assert.equal(policyModule.gpuFromEnv({MOM_GPU_BACKEND: "all"}), "");
  assert.equal(policyModule.gpuFromEnv({MOM_GPU_BACKEND: "nvidia"}), "nvidia");
  assert.equal(policyModule.selection("etchash", "future-gpu", "linux"), null);
  assert.throws(() => policyModule.selection({toString: () => "etchash"}, "intel", "linux"),
    /algorithm must be a string/);
  assert.throws(() => policyModule.selection("etchash", {toString: () => "intel"}, "linux"),
    /GPU name must be a string/);
  assert.throws(() => policyModule.selection("etchash", "intel", {toString: () => "linux"}),
    /Platform must be a string/);
  assert.throws(() => Reflect.apply(policyModule.nvidiaComputeCapability, null, [{
    MOM_NVIDIA_COMPUTE_CAPABILITY: {toString: () => "8.0"},
  }]), /Invalid MOM_NVIDIA_COMPUTE_CAPABILITY/);

  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-contract-"));
  try {
    fs.mkdirSync(path.join(root, "oneapi"));
    fs.writeFileSync(path.join(root, "oneapi", "mom.node"), "test");
    fs.mkdirSync(path.join(root, "dpcpp-opencl"));
    fs.writeFileSync(path.join(root, "dpcpp-opencl", "mom.node"), "test");
    assert.throws(() => Reflect.apply(policyModule.workerEnv, null, ["etchash", {
      MOM_GPU_BACKEND: "intel", MOM_GPU_INDEX: {toString: () => "1"}, MOM_NATIVE_DIR: root,
    }, "linux"]), /Invalid MOM_GPU_INDEX/);
    assert.throws(() => Reflect.apply(policyModule.workerEnv, null, ["etchash", {
      MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: {toString: () => "gpu"},
      MOM_NATIVE_DIR: root,
    }, "linux"]), /Invalid MOM_OPENCL_DEVICE_TYPE/);
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test("PearlHash default M survives CPU worker tuning", () => {
  const api = testJobApi({...process.env, MOM_GPU_BACKEND: "intel"});
  const job = api.prepareBenchmarkJob({
    algo: "pearlhash", dev: "cpu", blob_hex: "00".repeat(76),
  });
  assert.equal(job.intensity, 131072);
  gpuTuning.applyNativeJobTuning(
    job, gpuTuning.parseDeviceEntry("cpu", "pearlhash"), "pearlhash"
  );
  assert.equal(job.intensity, 131072);
  assert.equal(job.pearlhash_n, 131072);
  assert.equal(job.pearlhash_k, 4096);
  assert.equal(job.pearlhash_rank, 256);

  const explicitJob = api.prepareBenchmarkJob({
    algo: "pearlhash", dev: "cpu", blob_hex: "00".repeat(76),
  });
  gpuTuning.applyNativeJobTuning(
    explicitJob, gpuTuning.parseDeviceEntry("cpu*[m=256]", "pearlhash"), "pearlhash"
  );
  assert.equal(explicitJob.intensity, 256);
  assert.equal(explicitJob.pearlhash_n, 256);
  assert.equal(explicitJob.pearlhash_k, 4096);
  assert.equal(explicitJob.pearlhash_rank, 256);

  const gpuJob = api.prepareBenchmarkJob({
    algo: "pearlhash", dev: "gpu1", blob_hex: "00".repeat(76),
  });
  assert.equal(gpuJob.pearlhash_k, 2048);
  assert.equal(gpuJob.pearlhash_rank, 128);

  const mixedJob = api.prepareBenchmarkJob({
    algo: "pearlhash",
    dev: "gpu1*[m=131072;n=524288;k=8192;rank=128],cpu",
    blob_hex: "00".repeat(76),
  });
  assert.equal(mixedJob.pearlhash_k, 4096);
  assert.equal(mixedJob.pearlhash_rank, 256);
  gpuTuning.applyNativeJobTuning(
    mixedJob,
    gpuTuning.parseDeviceEntry("gpu1*[m=131072;n=524288;k=8192;rank=128]", "pearlhash"),
    "pearlhash"
  );
  assert.equal(mixedJob.pearlhash_k, 8192);
  assert.equal(mixedJob.pearlhash_rank, 128);
});

test("CPU thread batches do not leak into algorithm tuning", () => {
  const api = testJobApi();
  api.workerRuntimeEnv("ghostrider", "cpu*8");
  assert.equal(api.workerRuntimeEnv("hoohash", "cpu1*8")["MOM_HOOHASH_INTENSITY"], "8");
});

test("Intel command-list policy keeps short staged workers off the one-core path", () => {
  const api = testJobApi();
  for (const algo of ["c29", "equihash192_7", "zhash"]) {
    const env = api.workerRuntimeEnv(algo);
    assert.equal(env["SYCL_UR_USE_LEVEL_ZERO_V2"], "0");
    assert.equal(env["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"], "0");
  }
  assert.equal(api.workerRuntimeEnv("hoohash")["SYCL_UR_USE_LEVEL_ZERO_V2"], undefined);
});

test("test-only nonce offsets stay optional but explicit invalid offsets fail", () => {
  const api = testJobApi();
  const job = api.prepareTestJob({algo: "rx/0", dev: "cpu", blob_hex: "00"});
  assert.equal(job.nonceoffset, undefined);
  const pearlJob = api.prepareTestJob({
    algo: "pearlhash", dev: "gpu1*[m=256]", blob_hex: "00".repeat(76),
  });
  assert.equal(pearlJob.nonceoffset, undefined);
  assert.equal(pearlJob.noncebytes, 8);
  assert.throws(() => api.prepareTestJob({
    algo: "rx/0", dev: "cpu", blob_hex: "00", nonceoffset: 999,
  }), /Invalid rx\/0 nonce offset/);
  assert.throws(() => api.prepareTestJob({
    algo: "kawpow", dev: "gpu1", blob_hex: "00".repeat(40), noncebytes: 8,
  }), /Invalid kawpow nonce offset/);
});

test("C30 benchmark normalization rejects a malformed legacy nonce tail", () => {
  const api = testJobApi();
  assert.throws(() => api.prepareBenchmarkJob({
    algo: "c30", dev: "gpu1", blob_hex: "ab".repeat(32) + "zz".repeat(8),
  }), /Invalid c30 job blob/);
  const job = api.prepareBenchmarkJob({
    algo: "c30", dev: "gpu1", blob_hex: "ab".repeat(32) + "0102030405060708",
  });
  assert.equal(job.blob_hex, "ab".repeat(32));
  assert.equal(job.nonce, "0807060504030201");
});

test("FishHash benchmarks use protocol-valid header and nonce layouts", () => {
  const api = testJobApi();
  const defaultJob = api.prepareBenchmarkJob({
    algo: "fishhash", dev: "gpu1", blob_hex: opts.create_default_opts().job.blob_hex,
  });
  assert.equal(defaultJob.blob_hex, "00".repeat(180));
  assert.equal(defaultJob.noncebytes, 8);
  assert.equal(defaultJob.nonceoffset, 172);

  const offlinePrefix = api.prepareBenchmarkJob({
    algo: "fishhash", dev: "gpu1", blob_hex: "ab".repeat(32),
  });
  assert.equal(offlinePrefix.blob_hex, "ab".repeat(32) + "00".repeat(8));
  assert.equal(offlinePrefix.nonceoffset, 32);

  const offlineHeader = api.prepareBenchmarkJob({
    algo: "fishhash", dev: "gpu1", blob_hex: "cd".repeat(40),
  });
  assert.equal(offlineHeader.blob_hex, "cd".repeat(40));
  assert.equal(offlineHeader.nonceoffset, 32);

  const poolHeader = api.prepareBenchmarkJob({
    algo: "fishhash", dev: "gpu1", blob_hex: "ef".repeat(180),
  });
  assert.equal(poolHeader.blob_hex, "ef".repeat(180));
  assert.equal(poolHeader.nonceoffset, 172);
});

test("tuning environments normalize validated values and native intensity is numeric", () => {
  assert.deepEqual(gpuTuning.tuningEnvironment("zelhash", {slots: "176"}), {
    MOM_ZELHASH_SLOTS: "176",
  });
  assert.throws(() => gpuTuning.tuningEnvironment("zelhash", {
    slots: {toString: () => "176"},
  }), /base-10 integer/);
  /** @type {MiningJob} */
  const defaultJob = {algo: "kawpow", dev: ""};
  gpuTuning.applyNativeJobTuning(defaultJob, {device: "gpu1", processes: 1, tuning: {}}, "kawpow");
  assert.equal(defaultJob.intensity, 1);
  /** @type {MiningJob} */
  const zeroJob = {algo: "kawpow", dev: ""};
  gpuTuning.applyNativeJobTuning(
    zeroJob, {device: "gpu1", processes: 1, tuning: {intensity: 0}}, "kawpow");
  assert.equal(zeroJob.intensity, 0);
  assert.throws(() => Reflect.apply(gpuTuning.applyNativeJobTuning, null, [
    {algo: "kawpow", dev: ""},
    {device: "gpu1", processes: 1, tuning: {intensity: {toString: () => "8"}}},
    "kawpow",
  ]), /finite number/);
});

test("NVRTC temporary programs release once across source-JIT failures", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, async (t) => {
  const root = path.join(__dirname, "../sycl");
  const header = fs.readFileSync(path.join(root, "cuda-api.h"), "utf8");
  const helpers = header.indexOf("inline void cuda_check(");
  const namespaceEnd = header.lastIndexOf("} // namespace mom");
  assert.ok(helpers >= 0 && namespaceEnd > helpers);
  const ownerStart = header.indexOf("class NvrtcProgram {");
  const owner = ownerStart < 0 ? "" : header.slice(ownerStart, helpers);
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvrtc-program-"));
  const source = path.join(fixture, "nvrtc_program.cpp");
  const executable = path.join(fixture, "nvrtc_program");
  try {
    fs.copyFileSync(path.join(__dirname, "native/nvrtc_program.cpp"), source);
    fs.writeFileSync(path.join(fixture, "api.inc"), "namespace mom {\n" + owner +
      header.slice(helpers, namespaceEnd) + "\n}\n");
    const cn = fs.readFileSync(path.join(root, "cn_gpu/cuda_jit.inc"), "utf8");
    const cnEnd = cn.indexOf("  void launch(");
    const cnDestructor = cn.indexOf("  ~CnGpuCudaExpand()");
    const cnClassEnd = cn.indexOf("\n};", cnDestructor);
    assert.ok(cnEnd > 0 && cnDestructor > cnEnd && cnClassEnd > cnDestructor);
    fs.writeFileSync(path.join(fixture, "cn.inc"), cn.slice(0, cnEnd) +
      cn.slice(cnDestructor, cnClassEnd) + "\n};\n#endif\n");
    const pearl = fs.readFileSync(path.join(root, "pearlhash/cuda_search.inc"), "utf8");
    const pearlEnd = pearl.indexOf("  void launch(");
    const pearlDestructor = pearl.indexOf("  ~PearlHashCudaSearch()");
    const pearlClassEnd = pearl.indexOf("\n};", pearlDestructor);
    assert.ok(pearlEnd > 0 && pearlDestructor > pearlEnd && pearlClassEnd > pearlDestructor);
    fs.writeFileSync(path.join(fixture, "pearl.inc"), pearl.slice(0, pearlEnd) +
      pearl.slice(pearlDestructor, pearlClassEnd) + "\n};\n");
    const flags = owner ? ["-DMOM_FIXTURE_HAS_OWNER=1"] : [];
    const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      "-pedantic", "-DMOM_SYCL_HAS_CUDA=1", ...flags, "-I", path.join(root, "cn_gpu"),
      "-I", path.join(root, "pearlhash"), source, "-o", executable], {encoding: "utf8"});
    assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
    const cases = [
      {id: 0, name: "success releases before module loading"},
      {id: 1, name: "unavailable API creates no program"},
      {id: 2, name: "failed creation with a null handle"},
      {id: 3, name: "failed creation with a returned handle"},
      {id: 4, name: "compile failure retains diagnostic fallback"},
      {id: 5, name: "log-size failure retains diagnostic fallback"},
      {id: 6, name: "log retrieval failure retains diagnostic fallback"},
      {id: 7, name: "code-size failure releases the program"},
      {id: 8, name: "code retrieval failure releases the program"},
      {id: 9, name: "log allocation failure releases the program"},
      {id: 10, name: "code allocation failure releases the program"},
      {id: 11, name: "unsupported capability creates no program"},
      {id: 12, name: "null program is not destroyed"},
      {id: 15, name: "cache hit loads without creating a program"},
      {id: 16, name: "non-CUDA device creates no program"},
    ];
    for (const [variant, name] of ["CN PTX", "Pearl CUBIN", "Pearl forward PTX"].entries()) {
      for (const {id, name: description} of cases) {
        await t.test(name + ": " + description, () => {
          const result = spawnSync(executable, [String(variant + 1), String(id)], {encoding: "utf8"});
          assert.equal(result.status, 0, result.error?.message || result.stderr);
          assert.equal(result.signal, null);
          assert.equal(result.stdout, "PASS actual-source NVRTC lifetime\n");
        });
      }
    }
    if (owner) {
      for (const {id, name, output} of [
        {id: 13, name: "empty owner/reset is a no-op", output: "PASS empty owner\n"},
        {id: 14, name: "reset and destructor do not repeat destruction", output: "PASS explicit reset\n"},
      ]) {
        await t.test(name, () => {
          const result = spawnSync(executable, ["1", String(id)], {encoding: "utf8"});
          assert.equal(result.status, 0, result.error?.message || result.stderr);
          assert.equal(result.signal, null);
          assert.equal(result.stdout, output);
        });
      }
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("CN/gpu CUDA expansion falls back only before native submission", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, async (t) => {
  const directory = path.join(__dirname, "../sycl/cn_gpu");
  const header = fs.readFileSync(path.join(directory, "../cuda-api.h"), "utf8");
  const ownerStart = header.indexOf("class NvrtcProgram {");
  const helpers = header.indexOf("inline void cuda_check(");
  assert.ok(helpers >= 0);
  const owner = ownerStart < 0 ? "" : header.slice(ownerStart, helpers);
  const entry = fs.readFileSync(path.join(directory, "entry.inc"), "utf8");
  const start = entry.indexOf("  bool native_expansion = false;");
  const end = entry.indexOf("  if (!native_expansion) {", start);
  assert.ok(start >= 0 && end > start);
  const dispatch = entry.slice(start, end);
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cn-cuda-"));
  const source = path.join(fixture, "cuda.cpp");
  const executable = path.join(fixture, "cuda");
  fs.writeFileSync(source, `
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using CUresult = int;
using CUdevice = int;
using CUmodule = void*;
using CUfunction = void*;
using CUevent = void*;
using CUcontext = void*;
using nvrtcProgram = void*;
using nvrtcResult = int;
constexpr int CUDA_SUCCESS = 0;
constexpr int CUDA_ERROR_NOT_READY = 600;
constexpr int NVRTC_SUCCESS = 0;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 1;
constexpr int CU_EVENT_DEFAULT = 0;
static int fault;
static bool driver_available = true;
struct Counts {
  unsigned retain = 0;
  unsigned unload = 0;
  unsigned release = 0;
  unsigned destroy = 0;
  unsigned launch = 0;
  unsigned record = 0;
  unsigned query = 0;
  unsigned create_program = 0;
  unsigned destroy_program = 0;
  unsigned initial_wait = 0;
  unsigned generic = 0;
} counts;
static int handles[4];
namespace sycl {
enum class backend { ext_oneapi_cuda };
struct device { bool cuda = true; };
struct queue { device get_device() const { return {}; } };
template<backend> CUdevice get_native(const device&) { return 0; }
}
static bool mom_is_cuda(const sycl::device& device) { return device.cuda; }
namespace mom {
struct CudaDriverApi {
  std::string error = "driver-unavailable";
  static CudaDriverApi& instance() {
    static CudaDriverApi api;
    return api;
  }
  bool basic_available() const { return driver_available; }
  int (*init)(unsigned) = +[](unsigned) { return 0; };
  int (*device_attribute)(int*, int, int) = +[](int* out, int, int) {
    *out = 9;
    return 0;
  };
  int (*retain_primary)(void**, int) = +[](void** out, int) {
    ++counts.retain;
    *out = &handles[0];
    return 0;
  };
  int (*set_current)(void*) = +[](void*) { return 0; };
  int (*module_load)(void**, const void*) = +[](void** out, const void*) {
    *out = &handles[1];
    return 0;
  };
  int (*module_function)(void**, void*, const char*) = +[](void** out, void*, const char*) {
    if (fault == 5) return 999;
    *out = &handles[2];
    return 0;
  };
  int (*event_create)(void**, unsigned) = +[](void** out, unsigned) {
    *out = &handles[3];
    return 0;
  };
  int (*event_destroy)(void*) = +[](void*) {
    ++counts.destroy;
    return 0;
  };
  int (*module_unload)(void*) = +[](void*) {
    ++counts.unload;
    return 0;
  };
  int (*release_primary)(int) = +[](int) {
    ++counts.release;
    return 0;
  };
  int (*launch)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                unsigned, void*, void**, void**) =
      +[](void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
          unsigned, void*, void**, void**) {
        ++counts.launch;
        return fault == 2 ? 999 : 0;
      };
  int (*event_record)(void*, void*) = +[](void*, void*) {
    ++counts.record;
    return fault == 4 ? 999 : 0;
  };
  int (*event_query)(void*) = +[](void*) {
    ++counts.query;
    if (fault == 3) return 999;
    return counts.query == 1 ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
  };
};
struct NvrtcApi {
  std::string error = "rtc-unavailable";
  static NvrtcApi& instance() {
    static NvrtcApi api;
    return api;
  }
  bool ptx_available() const { return true; }
  int (*create_program)(void**, const char*, const char*, int, const char* const*, const char* const*) =
      +[](void** out, const char*, const char*, int, const char* const*, const char* const*) {
        ++counts.create_program;
        *out = &handles[2];
        return 0;
      };
  int (*compile_program)(void*, int, const char* const*) =
      +[](void*, int, const char* const*) { return 0; };
  int (*get_ptx_size)(void*, size_t*) = +[](void*, size_t* out) {
    *out = 1;
    return 0;
  };
  int (*get_ptx)(void*, char*) = +[](void*, char* out) {
    *out = 0;
    return 0;
  };
  int (*destroy_program)(void**) = +[](void** program) {
    ++counts.destroy_program;
    *program = nullptr;
    return 0;
  };
};
${owner}
void cuda_check(const CudaDriverApi&, int status, const char* message) {
  if (status != CUDA_SUCCESS) throw std::string(message);
}
std::string rtc_error(const NvrtcApi&, int, void*, const char* message) { return message; }
namespace jit_cache {
uint64_t hash(const std::string&) { return 1; }
std::filesystem::path directory() { return {}; }
std::vector<char> read(const std::filesystem::path&) { return {}; }
void write(const std::filesystem::path&, const std::vector<char>&) {}
}
}
#include "cuda_jit.inc"
struct State {
  sycl::device device;
  CnGpuCudaExpand cuda_expand;
  bool native_notice = false;
  uint64_t storage[8]{};
  uint64_t* spads = storage;
  uint64_t* lpads = storage;
};
static constexpr bool mom_sycl_portable_opencl = false;
static void sycl_wait_and_throw(int, const sycl::device&) {
  ++counts.initial_wait;
  if (fault == 1) throw std::runtime_error("initial-event");
}
static bool dispatch(State& state) {
  sycl::queue q;
  const std::string backend = "auto";
  const int initial_keccak = 0;
  const unsigned batch_eff = 4;
${dispatch}
  if (!native_expansion) ++counts.generic;
  return native_expansion;
}
static void check(bool valid, const char* message) {
  if (!valid) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("case argument");
    const int id = std::atoi(argv[1]);
#if defined(MOM_SYCL_HAS_CUDA)
    if (id == 6) {
      fault = 5;
      {
        CnGpuCudaExpand expand;
        sycl::queue queue;
        check(!expand.ensure(queue, nullptr), "nullable startup error did not return false");
        check(expand.checked && !expand.enabled && !expand.context && !expand.module &&
              !expand.kernel && !expand.done && !expand.retained_primary,
              "partial startup resources survived");
        check(counts.retain == 1 && counts.release == 1 && counts.unload == 1 &&
              counts.destroy == 0 && counts.create_program == counts.destroy_program,
              "partial startup cleanup changed");
      }
      check(counts.release == 1 && counts.unload == 1, "startup cleanup ran twice");
    } else {
      driver_available = id != 1;
      fault = id == 3 ? 1 : id == 4 ? 2 : id == 5 ? 3 : id == 7 ? 4 : 0;
      const bool expected_fault = id == 3 || id == 4 || id == 5 || id == 7;
      {
        State state;
        bool native = false;
        std::string caught;
        try {
          native = dispatch(state);
        } catch (const std::string& error) {
          caught = error;
        } catch (const std::exception& error) {
          caught = error.what();
        }
        std::printf("OBS case=%d native=%d generic=%u caught=%s retain=%u release=%u unload=%u destroy=%u launch=%u record=%u query=%u\\n",
                    id, native, counts.generic, caught.c_str(), counts.retain, counts.release,
                    counts.unload, counts.destroy, counts.launch, counts.record, counts.query);
        if (expected_fault) {
          check(!caught.empty(), "post-submit fault was swallowed");
          check(counts.generic == 0, "post-submit fault retried generic expansion");
          check(state.cuda_expand.enabled && counts.release == 0 && counts.unload == 0 &&
                counts.destroy == 0, "post-submit resources released before owner cleanup");
          const std::string expected = id == 3 ? "initial-event" :
              id == 4 ? "cuLaunchKernel(cn/gpu)" :
              id == 5 ? "cuEventQuery(cn/gpu)" : "cuEventRecord(cn/gpu)";
          check(caught == expected, "fault identity changed");
        } else if (id == 1) {
          check(!native && counts.generic == 1 && caught.empty() && counts.launch == 0 &&
                state.native_notice, "startup unavailability did not fall back");
        } else {
          check(id == 2 && native && counts.generic == 0 && caught.empty() &&
                counts.launch == 1 && counts.record == 1 && counts.query == 2,
                "native success changed");
        }
      }
      check(counts.release == (id == 1 ? 0u : 1u) &&
            counts.unload == counts.release && counts.destroy == counts.release,
            "owner cleanup did not release resources once");
      check(counts.create_program == counts.destroy_program, "NVRTC program leaked");
    }
#else
    check(id == 8, "unsupported-worker case argument");
    {
      State state;
      check(!dispatch(state) && counts.generic == 1 && counts.initial_wait == 0,
            "unsupported worker did not fall back");
      sycl::queue queue;
      check(!state.cuda_expand.ensure(queue, nullptr), "stub nullable reason changed");
    }
#endif
    std::printf("PASS case=%d\\n", id);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\\n", error.what());
    return 1;
  }
}
`);
  const cases = [
    "startup unavailability retains generic expansion",
    "native success retains deferred owner cleanup",
    "initial SYCL event fault propagates without generic retry",
    "native launch fault propagates without generic retry",
    "native event-query fault propagates without generic retry",
    "nullable startup reason cleans partial resources once",
    "post-launch event-record fault propagates without generic retry",
    "unsupported worker retains nullable-reason fallback",
  ];
  try {
    for (const enabled of [true, false]) {
      const target = executable + (enabled ? "-enabled" : "-stub");
      const flags = enabled ? ["-DMOM_SYCL_HAS_CUDA=1"] : [];
      const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        "-pedantic", ...flags, "-I", directory, source, "-o", target], {encoding: "utf8"});
      assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
      const indexes = enabled ? cases.slice(0, 7).map((_name, index) => index) : [7];
      for (const index of indexes) {
        const name = cases[index];
        assert.ok(name);
        await t.test(name, () => {
          const result = spawnSync(target, [String(index + 1)], {encoding: "utf8"});
          assert.equal(result.status, 0, result.error?.message || result.stderr);
          assert.equal(result.signal, null);
          assert.ok(result.stdout.endsWith("PASS case=" + (index + 1) + "\n"));
        });
      }
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("HIPRTC temporary programs release once across source-JIT failures", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, async (t) => {
  const root = path.join(__dirname, "../sycl");
  const header = fs.readFileSync(path.join(root, "hiprtc-api.h"), "utf8");
  const ownerStart = header.indexOf("class HiprtcProgram {");
  const owner = ownerStart < 0 ? "" :
    header.slice(ownerStart, header.lastIndexOf("} // namespace mom")).trim();
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hiprtc-program-"));
  const source = path.join(fixture, "hiprtc_program.cpp");
  const executable = path.join(fixture, "hiprtc_program");
  try {
    fs.copyFileSync(path.join(__dirname, "native/hiprtc_program.cpp"), source);
    fs.writeFileSync(path.join(fixture, "owner.inc"), owner ? "namespace mom {\n" + owner + "\n}\n" : "");
    for (const algorithm of ["octopus", "walahash"]) {
      const caller = fs.readFileSync(path.join(root, algorithm, "amd_wmma.inc"), "utf8");
      const start = caller.indexOf("struct ");
      const end = caller.indexOf("  bool launch(");
      assert.ok(start >= 0 && end > start);
      fs.writeFileSync(path.join(fixture, algorithm + ".inc"), caller.slice(start, end) + "};\n");
    }
    const pearl = fs.readFileSync(path.join(root, "pearlhash/hip_jit.inc"), "utf8");
    const pearlEnd = pearl.indexOf("  void launch(");
    assert.ok(pearlEnd > 0);
    fs.writeFileSync(path.join(fixture, "pearl.inc"), pearl.slice(0, pearlEnd) + "#endif\n};\n");
    const flags = owner ? ["-DMOM_FIXTURE_HAS_OWNER=1"] : [];
    const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      "-pedantic", "-DMOM_SYCL_HAS_HIP=1", ...flags, "-I", path.join(root, "pearlhash"),
      source, "-o", executable], {encoding: "utf8"});
    assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
    const cases = [
      "successful compilation releases before module loading",
      "unavailable API creates no program",
      "failed creation with a null handle",
      "failed creation with a returned handle",
      "compile failure preserves diagnostic fallback",
      "log-size failure preserves diagnostic fallback",
      "log retrieval failure preserves diagnostic fallback",
      "code-size failure releases the program",
      "code retrieval failure releases the program",
      "log allocation failure releases the program",
      "code allocation failure releases the program",
      "unsupported architecture creates no program",
      "null program is not destroyed",
    ];
    for (const [algorithm, name] of ["Octopus", "WalaHash", "PearlHash"].entries()) {
      for (const [id, description] of cases.entries()) {
        await t.test(name + ": " + description, () => {
          const result = spawnSync(executable, [String(algorithm + 1), String(id)], {encoding: "utf8"});
          assert.equal(result.status, 0, result.error?.message || result.stderr);
          assert.equal(result.signal, null);
          assert.equal(result.stdout, "PASS actual-source RTC lifetime\n");
        });
      }
    }
    if (owner) {
      for (const {id, name, output} of [
        {id: 13, name: "empty owner/reset is a no-op", output: "PASS empty owner\n"},
        {id: 14, name: "reset and destructor do not repeat destruction", output: "PASS explicit reset\n"},
      ]) {
        await t.test(name, () => {
          const result = spawnSync(executable, ["1", String(id)], {encoding: "utf8"});
          assert.equal(result.status, 0, result.error?.message || result.stderr);
          assert.equal(result.signal, null);
          assert.equal(result.stdout, output);
        });
      }
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("HIP interop and Octopus submitted faults propagate without fallback", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, async (t) => {
  const root = path.join(__dirname, "../sycl");
  const header = fs.readFileSync(path.join(root, "hiprtc-api.h"), "utf8");
  const identityFunction = header.indexOf(" hip_queue_identity(");
  const identityStart = header.lastIndexOf("inline ", identityFunction);
  const identityEnd = header.indexOf("// HIPRTC is optional", identityStart);
  const octopus = fs.readFileSync(path.join(root, "octopus/nvidia_tensor.inc"), "utf8");
  const classStart = octopus.indexOf("class OctopusSyclNativeSearch {");
  const contextStart = octopus.indexOf("struct NativeGemmContext {");
  const contextEnd = octopus.indexOf("\n};", contextStart);
  const wmma = fs.readFileSync(path.join(root, "octopus/amd_wmma.inc"), "utf8");
  const wmmaStart = wmma.indexOf("  bool ensure(");
  const wmmaEnd = wmma.indexOf("    hipDeviceProp_t properties{}", wmmaStart);
  assert.ok(identityStart > 0 && identityEnd > identityStart && classStart > contextEnd &&
    contextStart > 0 && contextEnd > contextStart && wmmaStart > 0 && wmmaEnd > wmmaStart);
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hip-interop-"));
  const source = path.join(fixture, "hip_queue_identity.cpp");
  const executable = path.join(fixture, "hip_queue_identity");
  try {
    fs.copyFileSync(path.join(__dirname, "native/hip_queue_identity.cpp"), source);
    fs.writeFileSync(path.join(fixture, "identity.inc"),
      "namespace mom {\n" + header.slice(identityStart, identityEnd) + "}\n");
    fs.writeFileSync(path.join(fixture, "octopus.inc"), octopus.slice(classStart));
    fs.writeFileSync(path.join(fixture, "gemm-context.inc"), octopus.slice(contextStart, contextEnd + 3));
    fs.writeFileSync(path.join(fixture, "wmma-init.inc"), wmma.slice(wmmaStart, wmmaEnd));
    const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      "-pedantic", "-DMOM_SYCL_HAS_HIP=1", source, "-o", executable], {encoding: "utf8"});
    assert.equal(compiled.error, undefined);
    assert.equal(compiled.signal, null);
    assert.equal(compiled.status, 0, compiled.stderr);
    const cases = [];
    for (const [target, name] of ["identity", "device alias", "Octopus search"].entries()) {
      cases.push({target, stage: 0, kind: 0, name: name + " success/cache/pipeline"});
      for (const [kind, error] of [[1, "InvalidValue"], [3, "NotInitialized"], [4, "Deinitialized"],
        [201, "InvalidContext"], [400, "InvalidHandle"], [709, "ContextIsDestroyed"], [999, "Unknown"]]) {
        cases.push({target, stage: 14, kind, name: name + " rejects HIP " + error});
      }
      for (const [kind, error] of ["std::exception", "std::string", "unknown"].entries()) {
        for (const [stage, operation] of [[1, "submit"], [2, "native queue"], [3, "interop wait"]]) {
          cases.push({target, stage, kind, name: name + " " + operation + " propagates " + error});
        }
        if (target === 2) {
          for (const [stage, operation] of [[4, "B preparation"], [5, "B preparation wait"],
            [6, "A preparation"], [7, "GEMM launch"], [8, "finish launch"], [9, "completion"],
            [10, "compiler exception"]]) {
            cases.push({target, stage, kind, name: name + " " + operation + " propagates " + error});
          }
        }
      }
    }
    for (const [stage, name] of [[11, "queue allocation"], [12, "buffer allocation"],
      [13, "compiler/capability absence"], [15, "point-version cache invalidation"],
      [16, "non-HIP device fallback"]]) {
      cases.push({target: 2, stage, kind: 0, name: "Octopus retains " + name + " behavior"});
    }
    for (const {target, stage, kind, name} of cases) {
      await t.test(name, () => {
        const result = spawnSync(executable, [String(target), String(stage), String(kind)],
          {encoding: "utf8"});
        assert.equal(result.error, undefined);
        assert.equal(result.signal, null);
        assert.equal(result.status, 0, result.stderr);
        assert.equal(result.stdout, "PASS actual-source HIP interop/Octopus fault boundary\n");
      });
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("algo params probes share identical effective environments only", async () => {
  let probeCount = 0;
  const cache = createAlgoParamsReportCache((/** @type {TestEnv} */ _env) => {
    ++probeCount;
    return Promise.resolve({params: {}, stdout: "", stderr: ""});
  });
  const dpcppKawpow = {
    MOM_GPU_BACKEND: "nvidia",
    MOM_NATIVE_PATH: "runtime/dpcpp/mom.node",
    MOM_SYCL_COMPILER: "dpcpp",
    ONEAPI_DEVICE_SELECTOR: "cuda:gpu",
  };
  const dpcppEtchash = {
    ONEAPI_DEVICE_SELECTOR: "cuda:gpu",
    MOM_SYCL_COMPILER: "dpcpp",
    MOM_NATIVE_PATH: "runtime/dpcpp/mom.node",
    MOM_GPU_BACKEND: "nvidia",
  };
  const acppFishhash = {
    ...dpcppKawpow,
    MOM_NATIVE_PATH: "runtime/acpp-cuda/mom.node",
    MOM_SYCL_COMPILER: "acpp-cuda",
    ACPP_VISIBILITY_MASK: "cuda",
  };

  const [first, second] = await Promise.all([
    cache(dpcppKawpow), cache(dpcppEtchash),
  ]);
  assert.strictEqual(first, second);
  assert.equal(probeCount, 1);
  await cache(acppFishhash);
  assert.equal(probeCount, 2);

  let envProbeCount = 0;
  const envCache = createAlgoParamsReportCache((/** @type {TestEnv} */ _env) => {
    ++envProbeCount;
    return Promise.resolve({params: {}, stdout: "", stderr: ""});
  });
  const envWithSentinel = {
    ...dpcppKawpow,
    MOM_ALGO_PARAMS_CACHE_SENTINEL: "inherited",
  };
  const deletedEnvValue = await envCache({
    ...envWithSentinel,
    MOM_ALGO_PARAMS_CACHE_SENTINEL: undefined,
  });
  const emptyEnvValue = await envCache({
    ...envWithSentinel,
    MOM_ALGO_PARAMS_CACHE_SENTINEL: "",
  });
  assert.notStrictEqual(deletedEnvValue, emptyEnvValue);
  assert.equal(envProbeCount, 2);
});

test("algo params probe cache retries rejected probes without duplicating concurrent work", async () => {
  let probeCount = 0;
  let shouldReject = true;
  const cache = createAlgoParamsReportCache((/** @type {TestEnv} */ _env) => {
    ++probeCount;
    return shouldReject
      ? Promise.reject(new Error("probe failed"))
      : Promise.resolve({params: {}, stdout: "", stderr: ""});
  });
  const env = {MOM_ALGO_PARAMS_CACHE_RETRY: "1"};
  const [first, concurrent] = [cache(env), cache({...env})];
  assert.strictEqual(first, concurrent);
  assert.equal(probeCount, 1);
  await assert.rejects(first, /probe failed/);

  shouldReject = false;
  const retry = cache(env);
  assert.notStrictEqual(retry, first);
  assert.equal(probeCount, 2);
  await retry;
});

test("Pearl tuning preserves V3 targets after worker handoff", () => {
  /** @returns {MiningJob} */
  const makeJob = () => ({
    algo: "pearlhash",
    dev: "gpu1",
    intensity: 131072,
    pearlhash_base_target: "1",
    pearlhash_n: 131072,
    pearlhash_k: 4096,
    pearlhash_rank: 256,
    pearlhash_cert_version: 3,
  });
  const v3Target = helper.pearlhashTarget("1", 4096, 256, 3);

  const v3 = makeJob();
  gpuTuning.applyNativeJobTuning(
    v3, gpuTuning.parseDeviceEntry("gpu1*[m=8192]", "pearlhash"), "pearlhash");
  assert.equal(v3.target, v3Target);

  for (const rank of [128, 256]) {
    const k = rank * 16;
    const overridden = makeJob();
    gpuTuning.applyNativeJobTuning(
      overridden,
      gpuTuning.parseDeviceEntry(`gpu1*[m=8192;k=${k};rank=${rank}]`, "pearlhash"),
      "pearlhash",
    );
    assert.equal(overridden.pearlhash_k, k);
    assert.equal(overridden.pearlhash_rank, rank);
    assert.equal(overridden.target, helper.pearlhashTarget("1", k, rank, 3));
  }

  const explicit = makeJob();
  explicit.target = "ab".repeat(32);
  delete explicit.pearlhash_base_target;
  gpuTuning.applyNativeJobTuning(
    explicit,
    gpuTuning.parseDeviceEntry("gpu1*[m=8192;k=4096;rank=256]", "pearlhash"),
    "pearlhash",
  );
  assert.equal(explicit.target, "ab".repeat(32));

  for (const certVersion of [1, 2, 4]) {
    const unsupported = makeJob();
    Reflect.set(unsupported, "pearlhash_cert_version", certVersion);
    assert.throws(() => gpuTuning.applyNativeJobTuning(
      unsupported, gpuTuning.parseDeviceEntry("gpu1*[m=8192]", "pearlhash"), "pearlhash"
    ), /certificate version/);
  }
});

test("NVIDIA capability override selects compatibility paths", () => {
  assert.equal(policy.selection("pearlhash", "nvidia", "linux", 70).key, "acpp-cuda");
  assert.equal(policy.selection("pearlhash", "nvidia", "linux", 70).backend, "sycl-native");
  assert.equal(policy.selection("cn/gpu", "nvidia", "win32", 70).key, "acpp-cuda");
  assert.equal(policy.selection("cn/gpu", "nvidia", "win32", 70).backend, "sycl");
  assert.equal(policy.selection("etchash", "nvidia", "linux", 75).key, "acpp-cuda");
  assert.equal(policy.selection("etchash", "nvidia", "linux", 80).key, "dpcpp");
  assert.equal(policy.selection("octopus", "nvidia", "linux", 75).key, "acpp-cuda");
  assert.equal(policy.selection("octopus", "nvidia", "linux", 75).backend, "sycl");
  assert.equal(policy.selection("octopus", "nvidia", "linux", 80).key, "dpcpp");
  assert.equal(policy.selection("octopus", "nvidia", "linux", 80).backend, "sycl-native");
  assert.equal(policy.selection("xelishashv3", "nvidia", "linux", 75).backend, "sycl-native");
  assert.equal(policy.selection("xelishashv3", "nvidia", "linux", 80).backend, "sycl-native");
  assert.equal(policy.selection("nexapow", "nvidia", "linux", 70).backend, "sycl");
  assert.equal(policy.selection("nexapow", "nvidia", "linux", 75).backend, "sycl");
  assert.equal(policy.selection("nexapow", "nvidia", "linux", 80).backend, "sycl-native");
  assert.equal(policy.selection("walahash", "nvidia", "win32", 60).backend, "sycl");
  assert.equal(policy.selection("walahash", "nvidia", "win32", 61).backend, "sycl-native");
  assert.equal(policy.selection("walahash", "nvidia", "win32", 80).key, "dpcpp");
  assert.equal(policy.selection("walahash", "nvidia", "win32", 80).backend, "sycl-native");
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
  for (const key of ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"]) {
    fs.mkdirSync(path.join(root, key));
    fs.writeFileSync(path.join(root, key, "mom.node"), "test");
  }
  const volta = policy.workerEnv("pearlhash", {
    MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "7.0"
  }, "linux");
  assert.equal(volta.MOM_SYCL_COMPILER, "acpp-cuda");
  assert.equal(volta.ACPP_VISIBILITY_MASK, "cuda");
  const ampere = policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
  }, "linux");
  assert.equal(ampere.MOM_SYCL_COMPILER, "dpcpp");
  const turingNexa = policy.workerEnv("nexapow", {
    MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "7.5"
  }, "linux");
  assert.equal(turingNexa.MOM_SYCL_COMPILER, "acpp-cuda");
  assert.equal(turingNexa.MOM_NEXAPOW_NATIVE, undefined);
  const legacyNexa = policy.workerEnv("nexapow", {
    MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "7.0"
  }, "linux");
  assert.equal(legacyNexa.MOM_NEXAPOW_NATIVE, undefined);
  const ampereNexa = policy.workerEnv("nexapow", {
    MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
  }, "linux");
  assert.equal(ampereNexa.MOM_SYCL_COMPILER, "dpcpp");
  assert.equal(ampereNexa.MOM_NEXAPOW_NATIVE, undefined);
  const portableNexa = policy.workerEnv("nexapow", {
    MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
  }, "linux", "sycl");
  assert.equal(portableNexa.MOM_SYCL_COMPILER, "acpp-cuda");
  assert.equal(portableNexa.MOM_NEXAPOW_NATIVE, undefined);
  const syclNativeOctopus = policy.workerEnv("octopus", {
    MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
  }, "linux");
  assert.equal(syclNativeOctopus.MOM_OCTOPUS_SYCL_NATIVE, "1");
  assert.equal(policy.workerEnv("octopus", {
    MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: root
  }, "linux", "sycl-native").MOM_OCTOPUS_SYCL_NATIVE, "1");
  assert.equal(policy.workerEnv("octopus", {
    MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
  }, "linux", "sycl").MOM_OCTOPUS_SYCL_NATIVE, "0");
  const nativeXelis = policy.workerEnv("xelishashv3", {
    MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
  }, "linux");
  assert.equal(nativeXelis.MOM_XELISHASHV3_SYCL_NATIVE, "1");
  assert.equal(policy.workerEnv("xelishashv3", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
  }, "linux").MOM_XELISHASHV3_SYCL_NATIVE, "1");
  assert.equal(policy.workerEnv("xelishashv3", {
    MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
  }, "linux", "sycl").MOM_XELISHASHV3_SYCL_NATIVE, "0");
  const nativeWala = policy.workerEnv("walahash", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
  }, "linux");
  assert.equal(nativeWala.MOM_WALAHASH_SYCL_NATIVE, "1");
  const portableWala = policy.workerEnv("walahash", {
    MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
  }, "linux", "sycl");
  assert.equal(portableWala.MOM_SYCL_COMPILER, "dpcpp-opencl");
  assert.equal(portableWala.MOM_WALAHASH_SYCL_NATIVE, "0");
  assert.equal(policy.workerEnv("walahash", {
    MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: root
  }, "linux").MOM_WALAHASH_SYCL_NATIVE, "1");
  const nativeNvidiaWala = policy.workerEnv("walahash", {
    MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
  }, "win32");
  assert.equal(nativeNvidiaWala.MOM_SYCL_COMPILER, "dpcpp");
  assert.equal(nativeNvidiaWala.MOM_WALAHASH_SYCL_NATIVE, "1");
  assert.equal(policy.workerEnv("walahash", {
    MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
    MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
  }, "win32", "sycl").MOM_WALAHASH_SYCL_NATIVE, "0");
  fs.rmSync(root, {recursive: true, force: true});
});

test("SYCL workers persist kernels and preserve cache controls", (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-cache-"));
  const localBuild = path.resolve(__dirname, "..", "build", "win");
  const exists = fs.existsSync;
  let developmentBuild = false;
  t.mock.method(fs, "existsSync", (/** @type {import("node:fs").PathLike} */ file) =>
    String(file) === localBuild
      ? developmentBuild : exists(file));
  try {
    for (const key of ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-hip"]) {
      fs.mkdirSync(path.join(root, key));
      fs.writeFileSync(path.join(root, key, "mom.node"), "test");
    }
    /** @type {NodeJS.Platform[]} */
    const platforms = ["linux", "win32"];
    for (const platform of platforms) {
      for (const gpu of ["intel", "nvidia", "opencl"]) {
        const env = {MOM_NATIVE_DIR: root, MOM_GPU_BACKEND: gpu,
          MOM_NVIDIA_COMPUTE_CAPABILITY: "100", LOCALAPPDATA: "C:\\Users\\test\\AppData\\Local"};
        const defaults = policy.workerEnv("etchash", env, platform);
        assert.equal(defaults["SYCL_CACHE_PERSISTENT"], "1");
        assert.equal(defaults["SYCL_CACHE_DIR"], platform === "win32" && gpu === "intel"
          ? path.join(env.LOCALAPPDATA, "mom-sycl-cache", "oneapi", "etchash")
          : undefined);
        const custom = {...env, SYCL_CACHE_PERSISTENT: "0", SYCL_CACHE_DIR: "custom-cache"};
        const overridden = {...custom, ...policy.workerEnv("etchash", custom, platform)};
        assert.equal(overridden["SYCL_CACHE_PERSISTENT"], "0");
        assert.equal(overridden["SYCL_CACHE_DIR"], platform === "win32" && gpu === "intel"
          ? path.join("custom-cache", "oneapi", "etchash")
          : "custom-cache");
      }
    }
    const windowsIntel = {MOM_NATIVE_DIR: root, MOM_GPU_BACKEND: "intel",
      LOCALAPPDATA: "C:\\Users\\test\\AppData\\Local"};
    assert.equal(policy.workerEnv("walahash", windowsIntel, "win32")["SYCL_CACHE_DIR"],
      path.join(windowsIntel.LOCALAPPDATA, "mom-sycl-cache", "oneapi", "walahash"));
    assert.equal(policy.workerEnv("xelishashv3", windowsIntel, "win32")["SYCL_CACHE_DIR"],
      path.join(windowsIntel.LOCALAPPDATA, "mom-sycl-cache", "oneapi", "xelishashv3"));
    developmentBuild = true;
    const env = {MOM_NATIVE_DIR: root, MOM_GPU_BACKEND: "intel"};
    assert.equal(policy.workerEnv("nexapow", env, "win32")["SYCL_CACHE_DIR"],
      path.join(localBuild, ".sycl-cache", "oneapi", "nexapow"));
    const custom = {...env, SYCL_CACHE_PERSISTENT: "0", SYCL_CACHE_DIR: "retained-cache"};
    const overridden = {...custom, ...policy.workerEnv("nexapow", custom, "win32")};
    assert.equal(overridden["SYCL_CACHE_PERSISTENT"], "0");
    assert.equal(overridden["SYCL_CACHE_DIR"],
      path.join("retained-cache", "oneapi", "nexapow"));
    const amd = policy.workerEnv("etchash", {...env, MOM_GPU_BACKEND: "amd"}, "win32");
    assert.equal(amd["SYCL_CACHE_PERSISTENT"], undefined);

  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test("OpenCL helper selects the portable addon despite an inherited native override", (
  /** @type {import("node:test").TestContext} */ t,
) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-opencl-policy-"));
  const candidate = path.join(root, "acpp-cuda", "mom.node");
  const portable = path.join(root, "dpcpp-opencl", "mom.node");
  fs.mkdirSync(path.dirname(candidate), {recursive: true});
  fs.mkdirSync(path.dirname(portable), {recursive: true});
  fs.writeFileSync(candidate, "candidate");
  fs.writeFileSync(portable, "portable");
  const previousNativeDir = process.env["MOM_NATIVE_DIR"];
  const previousNativePath = process.env["MOM_NATIVE_PATH"];
  process.env["MOM_NATIVE_DIR"] = root;
  process.env["MOM_NATIVE_PATH"] = candidate;
  try {
    const env = openclSyclEnv("cpu");
    assert.equal(env["MOM_SYCL_COMPILER"], "dpcpp-opencl");
    assert.equal(env["MOM_NATIVE_PATH"], portable);
    assert.notEqual(env["MOM_NATIVE_PATH"], candidate);
    assert.equal(env["ONEAPI_DEVICE_SELECTOR"], "opencl:cpu");

    fs.rmSync(path.dirname(portable), {recursive: true});
    t.mock.method(fs, "existsSync", () => false);
    assert.throws(() => openclSyclEnv("cpu"), /Missing dpcpp-opencl\/mom\.node/);
  } finally {
    if (previousNativeDir === undefined) {
      delete process.env["MOM_NATIVE_DIR"];
    } else {
      process.env["MOM_NATIVE_DIR"] = previousNativeDir;
    }
    if (previousNativePath === undefined) {
      delete process.env["MOM_NATIVE_PATH"];
    } else {
      process.env["MOM_NATIVE_PATH"] = previousNativePath;
    }
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test("PearlHash submitted native faults propagate without fallback", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, async (t) => {
  const root = path.join(__dirname, "../sycl/pearlhash");
  const dispatch = fs.readFileSync(path.join(root, "dispatch.inc"), "utf8");
  const dispatchEnd = dispatch.indexOf("} // namespace mom_pearlhash");
  const host = fs.readFileSync(path.join(root, "host.inc"), "utf8");
  const invalidateStart = host.indexOf("  g_pf.valid = false;");
  const invalidateEnd = host.indexOf("  PearlHashState& st =", invalidateStart);
  const waitStart = host.indexOf("  if (search_backend == PearlHashSearchBackend::hip_jit ||");
  const uploadEnd = host.indexOf("  if (is_test && q.get_device().is_gpu())", waitStart);
  const drainStart = host.indexOf("\n  } catch (...)", waitStart);
  const waitEnd = drainStart < 0 ? uploadEnd : drainStart;
  const uploadStart = Math.min(host.indexOf("  if (!st.have_header ||"),
    host.indexOf("  uint8_t tgtLE[32];"));
  const readbackEnd = host.indexOf("  // search() retains", uploadEnd);
  const commitStart = host.indexOf("  if (!b.result->found)", waitEnd);
  const commitEnd = host.indexOf("\n}\n// Builds", commitStart);
  assert.ok(dispatchEnd > 0 && invalidateStart > 0 && invalidateEnd > invalidateStart &&
    waitStart > invalidateEnd && waitEnd > waitStart && commitStart > waitEnd && commitEnd > commitStart &&
    uploadStart > invalidateEnd && uploadEnd >= waitEnd && readbackEnd > uploadEnd);
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-native-fault-"));
  const source = path.join(fixture, "pearlhash_fault.cpp");
  const executable = path.join(fixture, "pearlhash_fault");
  try {
    fs.copyFileSync(path.join(__dirname, "native/pearlhash_fault.cpp"), source);
    fs.writeFileSync(path.join(fixture, "dispatch.inc"), dispatch.slice(0, dispatchEnd));
    fs.writeFileSync(path.join(fixture, "invalidate.inc"), host.slice(invalidateStart, invalidateEnd));
    fs.writeFileSync(path.join(fixture, "wait.inc"), host.slice(waitStart, waitEnd));
    fs.writeFileSync(path.join(fixture, "upload.inc"), host.slice(uploadStart, uploadEnd));
    fs.writeFileSync(path.join(fixture, "readback.inc"), host.slice(uploadEnd, readbackEnd));
    fs.writeFileSync(path.join(fixture, "commit.inc"), host.slice(commitStart, commitEnd));
    const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      "-pedantic", "-DMOM_SYCL_HAS_CUDA=1", "-DMOM_SYCL_HAS_HIP=1", source, "-o", executable],
    {encoding: "utf8"});
    assert.equal(compiled.error, undefined);
    assert.equal(compiled.signal, null);
    assert.equal(compiled.status, 0, compiled.stderr);
    const cases = [];
    for (const [hip, name] of ["CUDA", "HIP"].entries()) {
      cases.push({dispatch: 0, hip, stage: 0, stringError: 0, name: name + " wait succeeds"});
      cases.push({dispatch: 1, hip, stage: 0, stringError: 0, name: name + " native attempt succeeds"});
      cases.push({dispatch: 1, hip, stage: 1, stringError: 0, name: name + " unavailable JIT retains SYCL"});
      for (const [stringError, error] of ["std::exception", "std::string"].entries()) {
        cases.push({dispatch: 0, hip, stage: 7, stringError, name: name + " wait propagates " + error});
        for (const [stage, operation] of [[2, "queue wait"], [3, "preparation"], [6, "launch"]]) {
          cases.push({dispatch: 1, hip, stage, stringError,
            name: name + " " + operation + " propagates " + error});
        }
        if (hip) {
          for (const [stage, operation] of [[4, "A event wait"], [5, "B event wait"],
            [8, "A event query"], [9, "B event query"]]) {
            cases.push({dispatch: 1, hip, stage, stringError,
              name: name + " " + operation + " propagates " + error});
          }
        }
      }
    }
    for (const {dispatch: dispatchCase, hip, stage, stringError, name} of cases) {
      await t.test(name, () => {
        const result = spawnSync(executable,
          [String(dispatchCase), String(hip), String(stage), String(stringError)], {encoding: "utf8"});
        assert.equal(result.error, undefined);
        assert.equal(result.status, 0, result.stderr);
        assert.equal(result.signal, null);
        assert.equal(result.stdout, "PASS actual-source Pearl native fault boundary\n");
      });
    }
    const copyCases = [
      {readback: 0, stage: 0, error: 0, name: "healthy deferred target upload"},
      {readback: 1, stage: 0, error: 0, name: "healthy deferred commitment readbacks"},
    ];
    /** @type {[number, string][]} */
    const copyFaults = [[2, "upload completion"], [7, "native completion"],
      [10, "roots submission"], [11, "noise submission"],
      [12, "key upload submission"], [13, "target upload submission"],
      [14, "first commitment readback"], [15, "second commitment readback"]];
    for (const [stage, operation] of copyFaults) {
      for (let error = 0; error < 4; ++error) {
        copyCases.push({readback: stage >= 14 ? 1 : 0, stage, error,
          name: operation + " retains owners and original " + (error % 2 ? "string" : "exception") +
            (error >= 2 ? " despite cleanup failure" : "")});
      }
    }
    for (const {readback, stage, error, name} of copyCases) {
      await t.test(name, () => {
        const result = spawnSync(executable,
          ["copies", String(readback), String(stage), String(error)], {encoding: "utf8"});
        assert.equal(result.error, undefined);
        assert.equal(result.status, 0, result.stderr);
        assert.equal(result.signal, null);
        assert.equal(result.stdout, "PASS actual-source Pearl queued-copy lifetime\n");
      });
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Octopus DAG faults propagate without light-search fallback", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, async (t) => {
  const root = path.join(__dirname, "..");
  const octopus = fs.readFileSync(path.join(root, "sycl/octopus/octopus.cpp"), "utf8");
  const execution = fs.readFileSync(path.join(root, "native/core/execution.inc"), "utf8");
  const library = fs.readFileSync(path.join(root, "sycl/lib-internal.h"), "utf8");
  /** @param {string} source @param {string} start @param {string} end */
  const extract = (source, start, end) => {
    const first = source.indexOf(start);
    const last = source.indexOf(end, first);
    assert.ok(first >= 0 && last > first, start);
    return source.slice(first, last);
  };
  const catchStart = execution.indexOf("      } catch (const std::string& err) {",
    execution.indexOf("switch (m_dev)"));
  const catchEnd = execution.indexOf("      if (loop_stats)", catchStart);
  const callerStart = octopus.indexOf("int octopus(unsigned,");
  assert.ok(catchStart > 0 && catchEnd > catchStart && callerStart > 0);
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-octopus-dag-fault-"));
  const source = path.join(fixture, "octopus_dag_fault.cpp");
  const executable = path.join(fixture, "octopus_dag_fault");
  try {
    fs.copyFileSync(path.join(__dirname, "native/octopus_dag_fault.cpp"), source);
    fs.writeFileSync(path.join(fixture, "state.inc"),
      extract(octopus, "  template <typename T> void free_ptr", "\n  void release()") +
      extract(octopus, "  void ensure_dag(", "\n};\n\nstatic DeviceStateRegistry<State>& registry()"));
    fs.writeFileSync(path.join(fixture, "result.inc"),
      extract(octopus, "struct Result {", "inline uint32_t load32("));
    fs.writeFileSync(path.join(fixture, "caller.inc"), octopus.slice(callerStart));
    fs.copyFileSync(path.join(root, "sycl/octopus/validation.h"), path.join(fixture, "validation.inc"));
    fs.writeFileSync(path.join(fixture, "compute-catch.inc"), execution.slice(catchStart, catchEnd));
    fs.writeFileSync(path.join(fixture, "cleanup.inc"),
      extract(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
    const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      "-pedantic", "-DMOM_SYCL_PORTABLE_OPENCL=1", source, "-o", executable], {encoding: "utf8"});
    assert.equal(compiled.error, undefined);
    assert.equal(compiled.signal, null);
    assert.equal(compiled.status, 0, compiled.stderr);
    const modes = ["full DAG/cache/release", "unsupported memory", "null allocation",
      "allocation std::exception", "allocation std::string", "allocation unknown",
      "first build", "second build", "first completion", "second completion",
      "unresolved memory query", "completion fault plus cleanup wait fault"];
    for (const scalar of [0, 1]) {
      for (const [mode, name] of modes.entries()) {
        const kinds = mode >= 6 ? [0, 1, 2] : [mode >= 3 ? mode - 3 : 0];
        for (const kind of kinds) {
          const error = mode >= 6 ? ": " + ["std::exception", "std::string", "unknown"][kind] : "";
          await t.test((scalar ? "scalar: " : "batched: ") + name + error, () => {
            const result = spawnSync(executable, [String(mode), String(kind), String(scalar)],
              {encoding: "utf8", timeout: 2000});
            assert.equal(result.error, undefined);
            assert.equal(result.signal, null);
            assert.equal(result.status, 0, result.stderr);
            assert.equal(result.stdout, "PASS actual-source Octopus DAG fault boundary\n");
          });
        }
      }
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Etchash submitted probe faults do not select a fallback search", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, async (t) => {
  const root = path.join(__dirname, "../sycl");
  const state = fs.readFileSync(path.join(root, "etchash/state.inc"), "utf8");
  const entry = fs.readFileSync(path.join(root, "etchash/entry.inc"), "utf8");
  const device = fs.readFileSync(path.join(root, "etchash/device.inc"), "utf8");
  const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
  /** @param {string} source @param {string} start @param {string} end */
  const extract = (source, start, end) => {
    const first = source.indexOf(start);
    const last = source.indexOf(end, first);
    assert.ok(first >= 0 && last > first, start);
    return source.slice(first, last);
  };
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-etchash-probe-fault-"));
  const source = path.join(fixture, "etchash_probe_fault.cpp");
  const executable = path.join(fixture, "etchash_probe_fault");
  try {
    fs.copyFileSync(path.join(__dirname, "native/etchash_probe_fault.cpp"), source);
    fs.writeFileSync(path.join(fixture, "probe.inc"),
      extract(state, "  bool use_inline_pair()", "  explicit EtchashState("));
    fs.writeFileSync(path.join(fixture, "result.inc"),
      extract(device, "struct EtchashResult {", "struct Uint2 {"));
    fs.writeFileSync(path.join(fixture, "cleanup.inc"),
      extract(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
    fs.writeFileSync(path.join(fixture, "selection.inc"),
      extract(entry, "    sycl_wait_and_throw(\n        state.use_inline_pair()", "\n  } else {"));
    fs.writeFileSync(path.join(fixture, "commit.inc"),
      extract(entry, "  const uint32_t count = state.result->count;", "\n}"));
    const compiled = spawnSync("c++", ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
      "-pedantic", "-fsanitize=address", "-fsanitize-address-use-after-scope",
      "-fno-omit-frame-pointer", ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []),
      source, "-o", executable], {encoding: "utf8"});
    assert.equal(compiled.error, undefined);
    assert.equal(compiled.signal, null);
    assert.equal(compiled.status, 0, compiled.stderr);
    const cases = ["numeric match", "output mismatch", "mix mismatch", "count mismatch"].map(
      (name, scenario) => ({name, scenario, stage: 0, kind: 0}));
    cases.push({name: "forced noinline", scenario: 13, stage: 0, kind: 0},
      {name: "forced inline", scenario: 14, stage: 0, kind: 0});
    const errors = ["std::exception", "std::string", "unknown"];
    for (let stage = 1; stage <= 5; ++stage) {
      cases.push({name: "allocation " + stage + " null", scenario: 4, stage, kind: 0});
      for (const [kind, error] of errors.entries()) {
        cases.push({name: "allocation " + stage + ": " + error, scenario: 5, stage, kind});
      }
    }
    const operations = ["copy", "copy completion", "inline submit", "inline completion",
      "reference submit", "reference completion", "reference fault plus cleanup fault"];
    for (const [offset, operation] of operations.entries()) {
      const scenario = offset + 6;
      for (const [kind, error] of errors.entries()) {
        for (const stage of scenario === 6 ? [1, 2, 3] : [0]) {
          cases.push({name: operation + (stage ? " " + stage : "") + ": " + error,
            scenario, stage, kind});
        }
      }
    }
    for (const [kind, error] of errors.entries()) {
      cases.push({name: "successful probe cleanup: " + error, scenario: 15, stage: 0, kind});
    }
    for (const {name, scenario, stage, kind} of cases) {
      await t.test(name, () => {
        const result = spawnSync(executable, [String(scenario), String(stage), String(kind)],
          {encoding: "utf8", timeout: 2000});
        assert.equal(result.error, undefined);
        assert.equal(result.signal, null);
        assert.equal(result.status, 0, result.stderr);
        assert.equal(result.stdout, "PASS actual-source Etchash probe fault boundary\n");
      });
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("NexaPoW submitted faults abort without fallback and do not latch failed setup", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, () => {
  const root = path.join(__dirname, "../sycl/nexapow");
  const host = fs.readFileSync(path.join(root, "nexapow.cpp"), "utf8");
  const pipeline = fs.readFileSync(path.join(root, "sycl_pipeline.inc"), "utf8");
  const section = (/** @type {string} */ source, /** @type {string} */ start,
    /** @type {string} */ end) => {
    const begin = source.indexOf(start);
    assert.ok(begin >= 0 && source.indexOf(start, begin + start.length) === -1, start);
    const stop = source.indexOf(end, begin + start.length);
    assert.ok(stop > begin, end);
    return source.slice(begin, stop);
  };
  const prefix = section(pipeline, "class NexaPowSyclSearch {", "  bool search(");
  const caller = section(host,
    "int nexapow(unsigned, uint32_t, const uint8_t* input, unsigned input_size, uint8_t* output,\n" +
    "            uint8_t*, uint64_t* pnonce, const uint8_t* target, const uint8_t*, unsigned intensity,\n" +
    "            bool is_test, bool, const std::string& dev) {", "\n#else\nint nexapow");
  const support = section(host, "struct Result {", "\nstatic constexpr uint8_t") + "\n" +
    section(host, "inline bool np_equal(", "\n#ifndef MOM_NEXAPOW_HOST_TEST");
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nexapow-fault-"));
  const source = path.join(fixture, "nexapow_fault.cpp");
  const executable = path.join(fixture, "nexapow_fault");
  try {
    fs.copyFileSync(path.join(__dirname, "native/nexapow_fault.cpp"), source);
    fs.writeFileSync(path.join(fixture, "search-prefix.inc"), prefix);
    fs.writeFileSync(path.join(fixture, "caller.inc"), caller);
    fs.writeFileSync(path.join(fixture, "caller-support.inc"), support);
    const compiled = spawnSync("c++", ["-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
      "-pedantic", "-fsanitize=address", "-fsanitize-address-use-after-scope",
      "-fno-omit-frame-pointer", ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []),
      source, "-o", executable], {encoding: "utf8"});
    assert.equal(compiled.error, undefined);
    assert.equal(compiled.signal, null);
    assert.equal(compiled.status, 0, compiled.stderr);
    const result = spawnSync(executable, [], {encoding: "utf8"});
    assert.equal(result.error, undefined);
    assert.equal(result.signal, null);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.match(result.stdout, /^# tests 125$/m);
    assert.match(result.stdout, /^# pass 125$/m);
    assert.match(result.stdout, /^# fail 0$/m);
    assert.match(result.stdout, /^# skipped 0$/m);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("C30 asynchronous completion faults stop solver stages and Core commits", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, async (t) => {
  const root = path.join(__dirname, "..");
  const c30 = fs.readFileSync(path.join(root, "sycl/c30/c30.cpp"), "utf8");
  const execution = fs.readFileSync(path.join(root, "native/core/execution.inc"), "utf8");
  /** @param {string} source @param {string} start @param {string} end */
  const extract = (source, start, end) => {
    const first = source.indexOf(start);
    const last = source.indexOf(end, first);
    assert.ok(first >= 0 && last > first, start);
    return source.slice(first, last);
  };
  const catchStart = execution.indexOf("      } catch (const std::string& err) {",
    execution.indexOf("switch (m_dev)"));
  const catchEnd = execution.indexOf("      if (loop_stats)", catchStart);
  assert.ok(catchStart > 0 && catchEnd > catchStart);
  const entryStart = c30.indexOf("int c30(unsigned,");
  assert.ok(entryStart > 0);
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-c30-async-fault-"));
  const source = path.join(fixture, "c30_async_fault.cpp");
  const executable = path.join(fixture, "c30_async_fault");
  try {
    fs.copyFileSync(path.join(__dirname, "native/c30_async_fault.cpp"), source);
    fs.copyFileSync(path.join(root, "sycl/c30/c30.h"), path.join(fixture, "c30-types.inc"));
    fs.copyFileSync(path.join(root, "native/job-boundary.h"), path.join(fixture, "job-boundary.inc"));
    fs.writeFileSync(path.join(fixture, "solver.inc"),
      extract(c30, "struct Solver {", "\n} // namespace mom::c30"));
    fs.writeFileSync(path.join(fixture, "entry.inc"), c30.slice(entryStart));
    fs.writeFileSync(path.join(fixture, "active-guard.inc"),
      extract(execution, "    if (!m_has_fn) {", "\n    { // A compute function is active"));
    fs.writeFileSync(path.join(fixture, "dispatch.inc"),
      extract(execution, "          case DEV::C30_GPU:", "          case DEV::KAWPOW_GPU:"));
    fs.writeFileSync(path.join(fixture, "compute-catch.inc"), execution.slice(catchStart, catchEnd));
    fs.writeFileSync(path.join(fixture, "accounting.inc"),
      extract(execution, "      uint64_t completed = m_batch;",
        "      if (m_dev == DEV::KAWPOW_GPU || m_dev == DEV::ETCHASH_GPU) {"));
    fs.writeFileSync(path.join(fixture, "nonce-commit.inc"),
      extract(execution, "      if (m_dev == DEV::C30_GPU) {", "      if (m_nonce_bytes == 4) {"));
    const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      "-pedantic", source, "-o", executable], {encoding: "utf8"});
    assert.equal(compiled.error, undefined);
    assert.equal(compiled.signal, null);
    assert.equal(compiled.status, 0, compiled.stderr);
    const stages = ["seed", "trim", "count readback", "compact", "compact readback",
      "edge readback", "recovery", "nonce readback"];
    for (const layout of [0, 1]) {
      for (const owner of [0, 1]) {
        const cases = [{kind: 0, stage: 0, name: "empty asynchronous error lists"}];
        for (const [stage, name] of stages.entries()) {
          for (const [kind, errors] of ["one SYCL", "multiple SYCL", "non-SYCL"].entries()) {
            cases.push({kind: kind + 1, stage: stage + 1, name: name + ": " + errors});
          }
        }
        for (const {kind, stage, name} of cases) {
          await t.test((owner ? "Core" : "entry") + (layout ? " packed: " : " wide: ") + name, () => {
            const result = spawnSync(executable,
              [String(kind), String(stage), String(layout), String(owner)], {encoding: "utf8"});
            assert.equal(result.error, undefined);
            assert.equal(result.signal, null);
            assert.equal(result.status, 0, result.stderr);
            assert.equal(result.stdout, "PASS actual-source C30 asynchronous completion boundary\n");
          });
        }
      }
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("C29 submitted graph faults propagate before cycle search or accounting", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, async (t) => {
  const root = path.join(__dirname, "..");
  const search = fs.readFileSync(path.join(root, "sycl/c29/search.inc"), "utf8");
  const entry = fs.readFileSync(path.join(root, "sycl/c29/entry.inc"), "utf8");
  const state = fs.readFileSync(path.join(root, "sycl/c29/state.inc"), "utf8");
  const trim = fs.readFileSync(path.join(root, "sycl/c29/trim.inc"), "utf8");
  const execution = fs.readFileSync(path.join(root, "native/core/execution.inc"), "utf8");
  /** @param {string} source @param {string} start @param {string} end */
  const extract = (source, start, end) => {
    const first = source.indexOf(start);
    const last = source.indexOf(end, first);
    assert.ok(first >= 0 && last > first, start);
    return source.slice(first, last);
  };
  const completeSearch = search.slice(search.indexOf("struct C29SearchCompletionGuard {"));
  assert.ok(completeSearch.startsWith("struct C29SearchCompletionGuard {"));
  assert.equal(completeSearch.split("std::thread(").length, 2);
  const catchStart = execution.indexOf("      } catch (const std::string& err) {",
    execution.indexOf("switch (m_dev)"));
  const catchEnd = execution.indexOf("      if (loop_stats)", catchStart);
  assert.ok(catchStart > 0 && catchEnd > catchStart);
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-c29-submitted-fault-"));
  const source = path.join(fixture, "c29_submitted_fault.cpp");
  const executable = path.join(fixture, "c29_submitted_fault");
  try {
    fs.copyFileSync(path.join(__dirname, "native/c29_submitted_fault.cpp"), source);
    fs.writeFileSync(path.join(fixture, "search.inc"),
      completeSearch.replace("std::thread(", "TestThread("));
    fs.writeFileSync(path.join(fixture, "entry.inc"), entry);
    fs.writeFileSync(path.join(fixture, "result-state.inc"),
      extract(state, "struct C29QueuedSolution {", "static sycl::async_handler c29_exception_handler()"));
    fs.writeFileSync(path.join(fixture, "activate.inc"), state.slice(state.indexOf("static void c29_activate_job(")));
    fs.copyFileSync(path.join(root, "sycl/c29/cycle.h"), path.join(fixture, "cycle.inc"));
    fs.copyFileSync(path.join(root, "native/job-boundary.h"), path.join(fixture, "job-boundary.inc"));
    fs.writeFileSync(path.join(fixture, "seed.inc"), "(void)c29_buffers;\n++seeds;\nif (mode == 2) inject();\n");
    fs.writeFileSync(path.join(fixture, "trim.inc"),
      "++trims;\nif (mode == 3) inject();\n" +
      "const sycl::event tail_event{};\nuint32_t trimmed_edge_count = 0;\n" +
      "sycl::buffer<uint32_t, 1> buffer_trimmed_edge_count{sycl::range<1>{1}};\n" +
      "sycl::buffer<sycl::uint2, 1> buffer_trimmed_edges_u2{sycl::range<1>{64}};\n" +
      extract(trim, "profile.mark(\"before tail wait\");", "profile.finish(trimmed_edge_count);") +
      "profile.finish(trimmed_edge_count);\n");
    fs.writeFileSync(path.join(fixture, "active-guard.inc"),
      extract(execution, "    if (!m_has_fn) {", "\n    { // A compute function is active"));
    fs.writeFileSync(path.join(fixture, "dispatch.inc"),
      extract(execution, "          case DEV::C29_GPU:", "          case DEV::C30_GPU:"));
    fs.writeFileSync(path.join(fixture, "compute-catch.inc"), execution.slice(catchStart, catchEnd));
    fs.writeFileSync(path.join(fixture, "accounting.inc"),
      extract(execution, "      uint64_t completed = m_batch;",
        "      if (m_dev == DEV::KAWPOW_GPU || m_dev == DEV::ETCHASH_GPU) {"));
    const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      "-pedantic", source, "-o", executable], {encoding: "utf8"});
    assert.equal(compiled.error, undefined);
    assert.equal(compiled.signal, null);
    assert.equal(compiled.status, 0, compiled.stderr);
    const modes = ["valid graph", "device lookup", "seed submission", "trim submission",
      "tail completion", "count readback", "edge readback", "legacy thread-start failure"];
    for (const proof of [32, 42]) {
      for (const owner of [0, 1]) {
        for (const [mode, name] of modes.entries()) {
          await t.test((owner ? "Core" : "entry") + " proof" + proof + ": " + name, () => {
            const result = spawnSync(executable, [String(mode), String(proof), String(owner)],
              {encoding: "utf8"});
            assert.equal(result.error, undefined);
            assert.equal(result.signal, null);
            assert.equal(result.status, 0, result.stderr);
            assert.match(result.stdout, /PASS actual-source C29 submitted fault boundary\n$/);
          });
        }
      }
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("release packagers never recursively remove a caller-selected archive", () => {
  const scripts = path.join(__dirname, "../.github/workflows/scripts");
  const linux = fs.readFileSync(path.join(scripts, "package-linux-combined.sh"), "utf8");
  assert.match(linux, /\[ -d "\$archive" \]/);
  assert.doesNotMatch(linux, /rm -f -- "\$archive"/);
  assert.doesNotMatch(linux, /rm -rf[^\n]*"\$archive"/);

  const windows = fs.readFileSync(path.join(scripts, "package-windows.ps1"), "utf8");
  assert.match(windows, /Test-Path -LiteralPath \$Archive -PathType Container/);
  assert.doesNotMatch(windows, /Remove-Item -Force -LiteralPath \$Archive/);
  assert.doesNotMatch(windows, /Remove-Item[^\n]*-Recurse[^\n]*\$Archive/);
  assert.match(windows, /\[IO\.Path\]::GetFullPath\(\$Archive\)/);
  assert.match(windows, /\[StringComparison\]::OrdinalIgnoreCase/);
});

test("Windows package launcher keeps the selected runtime ahead of shared oneAPI", () => {
  const script = fs.readFileSync(
    path.join(__dirname, "../.github/workflows/scripts/package-windows.ps1"), "utf8");
  const runtimePathLine = String.raw`if defined MOM_RUNTIME_DIR set "PATH=%MOM_RUNTIME_DIR%;%MOM_RUNTIME_DIR%\hipSYCL;%PATH%"`;
  const sharedDpcppPathLine = String.raw`if /I "%MOM_GPU_BACKEND%"=="opencl" set "PATH=%MOM_LIBS%\oneapi;%PATH%"`;
  assert.equal(script.split(runtimePathLine).length - 1, 1);
  assert.deepEqual(script.split(/\r?\n/).filter((line) =>
    line.includes('set "PATH=%MOM_LIBS%\\oneapi;')), [sharedDpcppPathLine]);
  assert.ok(script.indexOf(sharedDpcppPathLine) < script.indexOf(runtimePathLine),
    "OpenCL fallback dependencies must be prepended before the selected runtime");
});

test("Linux combined packager protects cleanup roots before build checks", () => {
  const root = path.join(__dirname, "..");
  const script = path.join(root, ".github/workflows/scripts/package-linux-combined.sh");
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-package-safety-test-"));
  const marker = path.join(temp, "release-combined", "marker");
  const archive = path.join(temp, "release-combined", "nested", "archive.tgz");
  fs.mkdirSync(path.dirname(marker), {recursive: true});
  fs.writeFileSync(marker, "keep");
  try {
    const result = spawnSync("bash", [script, "1.2.3", archive], {
      cwd: temp, encoding: "utf8",
    });
    assert.equal(result.status, 2, result.stderr);
    assert.match(result.stderr, /Archive path must not be inside cleanup directory/);
    assert.doesNotMatch(result.stdout + result.stderr, /missing; run|docker image inspect/i);
    assert.equal(fs.readFileSync(marker, "utf8"), "keep");
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
});

test("Linux combined packager exposes every worker through its read-only container mount", {
  skip: process.platform === "win32",
}, () => {
  const root = path.join(__dirname, "..");
  const script = path.join(root, ".github/workflows/scripts/package-linux-combined.sh");
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-package-mount-test-"));
  const fakeBin = path.join(temp, "bin");
  const dockerLog = path.join(temp, "docker.log");
  fs.mkdirSync(fakeBin);
  for (const file of ["package.json", "compiler-policy.js", "gpu-tuning.js", "README.md", "GPU-CONFIG.md", "LICENSE",
    "helper/hash.js", "scripts/install.sh", "scripts/install-cutlass.sh", "sycl/kawpow/device.inc",
    "sycl/kawpow/keccak.inc"]) {
    const destination = path.join(temp, file);
    fs.mkdirSync(path.dirname(destination), {recursive: true});
    fs.writeFileSync(destination, "fixture\n");
  }
  for (const compiler of ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"]) {
    const addon = path.join(temp, "build/lin/Release", compiler, "mom.node");
    fs.mkdirSync(path.dirname(addon), {recursive: true});
    const contents = "fixture\n";
    fs.writeFileSync(addon, contents);
    fs.writeFileSync(`${addon}.build-profile`, [
      "schema=1",
      `worker=${compiler}`,
      `sha256=${createHash("sha256").update(contents).digest("hex")}`,
      "portable=1",
      "cpu=unset",
      "",
    ].join("\n"));
  }
  const fakeNode = path.join(fakeBin, "node");
  fs.writeFileSync(fakeNode, [
    "#!/bin/sh",
    "# NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2",
    "output=$(sed -n 's/.*\"output\":\"\\([^\"]*\\)\".*/\\1/p' \"$2\")",
    "[ -n \"$output\" ] || exit 1",
    ": > \"$output\"",
  ].join("\n"));
  const fakeNpx = path.join(fakeBin, "npx");
  fs.writeFileSync(fakeNpx, [
    "#!/bin/sh",
    "case \" $* \" in",
    "  *' esbuild '*)",
    "    output=",
    "    for argument do",
    "      case \"$argument\" in --outfile=*) output=$(printf '%s\\n' \"$argument\" | sed 's/^--outfile=//') ;; esac",
    "    done",
    "    [ -n \"$output\" ] || exit 1",
    "    : > \"$output\"",
    "    ;;",
    "  *' postject '*) ;;",
    "  *) exit 1 ;;",
    "esac",
  ].join("\n"));
  const fakeDocker = path.join(fakeBin, "docker");
  fs.writeFileSync(fakeDocker, [
    "#!/bin/sh",
    "printf '%s\\n' \"$*\" >> \"$MOM_DOCKER_LOG\"",
    "[ \"$1\" != image ] || exit 0",
    "[ \"$1\" != rm ] || exit 0",
    "[ \"$1\" != run ] || exit 0",
    "if [ \"$1\" = exec ]; then",
    "  case \"$*\" in *'/acpp-hip/mom.node'*) exit 1 ;; *) exit 0 ;; esac",
    "fi",
    "exit 1",
  ].join("\n"));
  for (const executable of [fakeNode, fakeNpx, fakeDocker]) {
    fs.chmodSync(executable, 0o755);
  }
  try {
    const result = spawnSync("bash", [script, "1.2.3"], {
      cwd: temp,
      encoding: "utf8",
      env: {
        ...process.env,
        MOM_DOCKER_LOG: dockerLog,
        NODE_BIN: fakeNode,
        PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] || ""}`,
      },
    });
    assert.equal(result.status, 1, result.stdout + result.stderr);
    assert.match(result.stderr,
      /Packaging container cannot read \/repo\/build\/lin\/Release\/acpp-hip\/mom\.node/);
    const invocations = fs.readFileSync(dockerLog, "utf8");
    assert.ok(invocations.includes(
      `--mount type=bind,src=${temp}/build/lin/Release,dst=/repo/build/lin/Release,readonly`));
    for (const compiler of ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"]) {
      assert.ok(invocations.includes(`test -s /repo/build/lin/Release/${compiler}/mom.node`));
    }
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
});

test("Linux combined packager rejects incomplete or non-portable worker provenance before mutation", {
  skip: process.platform === "win32",
}, () => {
  const root = path.join(__dirname, "..");
  const script = path.join(root, ".github/workflows/scripts/package-linux-combined.sh");
  const workers = ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"];
  const contents = "worker fixture\n";
  const sha256 = createHash("sha256").update(contents).digest("hex");
  /** @typedef {{name: string, target: string, kind: string, expected: RegExp}} ProfileCase */
  /** @type {ProfileCase[]} */
  const profileCases = [
    ...workers.map((worker) => ({
      name: `${worker} portable=0`, target: worker, kind: "portable-zero",
      expected: /is not a portable release worker/,
    })),
    {name: "missing sidecar", target: "oneapi", kind: "missing-sidecar",
      expected: /\.build-profile is missing/},
    {name: "missing field", target: "oneapi", kind: "missing-field",
      expected: /has missing or duplicate fields/},
    {name: "duplicate field", target: "oneapi", kind: "duplicate-field",
      expected: /has missing or duplicate fields/},
    {name: "wrong worker key", target: "oneapi", kind: "wrong-worker",
      expected: /does not identify .*mom\.node exactly/},
    {name: "invalid hash", target: "oneapi", kind: "invalid-hash",
      expected: /does not identify .*mom\.node exactly/},
    {name: "mismatched hash", target: "oneapi", kind: "mismatched-hash",
      expected: /does not identify .*mom\.node exactly/},
    {name: "cpu=native", target: "oneapi", kind: "cpu-native",
      expected: /is not a portable release worker/},
    {name: "cpu=x86-64-v4", target: "oneapi", kind: "cpu-v4",
      expected: /is not a portable release worker/},
  ];

  for (const profileCase of profileCases) {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-package-profile-test-"));
    const fakeBin = path.join(temp, "bin");
    const dockerSentinel = path.join(temp, "docker-used");
    fs.mkdirSync(fakeBin);
    const fakeDocker = path.join(fakeBin, "docker");
    fs.writeFileSync(fakeDocker, [
      "#!/bin/sh",
      'printf "%s\\n" invoked > "$MOM_TEST_DOCKER_SENTINEL"',
      "exit 0",
      "",
    ].join("\n"));
    fs.chmodSync(fakeDocker, 0o755);

    try {
      for (const worker of workers) {
        const addon = path.join(temp, "build/lin/Release", worker, "mom.node");
        fs.mkdirSync(path.dirname(addon), {recursive: true});
        fs.writeFileSync(addon, contents);
        if (profileCase.kind === "missing-sidecar" && worker === profileCase.target) {
          continue;
        }
        const lines = [
          "schema=1",
          `worker=${profileCase.kind === "wrong-worker" && worker === profileCase.target
            ? "dpcpp" : worker}`,
          `sha256=${profileCase.kind === "invalid-hash" && worker === profileCase.target
            ? "not-a-digest" : profileCase.kind === "mismatched-hash" && worker === profileCase.target
              ? "0".repeat(64) : sha256}`,
          `portable=${profileCase.kind === "portable-zero" && worker === profileCase.target
            ? "0" : "1"}`,
          `cpu=${profileCase.kind === "cpu-native" && worker === profileCase.target
            ? "native" : profileCase.kind === "cpu-v4" && worker === profileCase.target
              ? "x86-64-v4" : "unset"}`,
        ];
        if (profileCase.kind === "missing-field" && worker === profileCase.target) {
          lines.pop();
        }
        if (profileCase.kind === "duplicate-field" && worker === profileCase.target) {
          lines.push("portable=1");
        }
        fs.writeFileSync(`${addon}.build-profile`, `${lines.join("\n")}\n`);
      }

      const result = spawnSync("bash", [script, "1.2.3", "published.tgz"], {
        cwd: temp,
        encoding: "utf8",
        env: {
          ...process.env,
          MOM_TEST_DOCKER_SENTINEL: dockerSentinel,
          PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] || ""}`,
        },
      });
      assert.equal(result.status, 1, `${profileCase.name}: ${result.stdout}${result.stderr}`);
      assert.match(result.stderr, profileCase.expected);
      assert.equal(fs.existsSync(dockerSentinel), false, `${profileCase.name}: Docker was used`);
      assert.equal(fs.existsSync(path.join(temp, "release-combined")), false);
      assert.equal(fs.existsSync(path.join(temp, "release-combined-build")), false);
      assert.equal(fs.existsSync(path.join(temp, "published.tgz")), false);
    } finally {
      fs.rmSync(temp, {recursive: true, force: true});
    }
  }
});

test("release scripts reject traversal versions before build or deploy phases", () => {
  const root = path.join(__dirname, "..");
  const traversalVersion = "../release-pwn";
  const linux = spawnSync("bash", [
    path.join(root, ".github/workflows/scripts/package-linux-combined.sh"), traversalVersion,
  ], {cwd: root, encoding: "utf8"});
  assert.equal(linux.status, 2, linux.error?.message);
  assert.match(linux.stderr, /^Invalid release version:/);
  assert.doesNotMatch(linux.stdout + linux.stderr, /missing; run|▶|docker/i);

  const deploy = spawnSync("bash", [path.join(root, "scripts/test-deploy.sh")], {
    cwd: root,
    encoding: "utf8",
    env: {...process.env, MOM_DEPLOY_TARGET: "linux", MOM_RELEASE_VERSION: traversalVersion},
  });
  assert.equal(deploy.status, 2, deploy.error?.message);
  assert.match(deploy.stderr, /^Invalid release version:/);
  assert.doesNotMatch(deploy.stdout + deploy.stderr, /▶|docker|Building|Packaging/i);

  if (process.platform === "win32") {
    const windows = spawnSync("powershell.exe", [
      "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
      path.join(root, ".github/workflows/scripts/package-windows.ps1"),
      "-Version", traversalVersion,
    ], {cwd: root, encoding: "utf8"});
    assert.notEqual(windows.status, 0, windows.error?.message);
    assert.match(windows.stdout + windows.stderr, /Invalid release version:/);
    assert.doesNotMatch(windows.stdout + windows.stderr, /Building|Packaging/i);
  }
});

test("r.sh keeps process locks in private Git metadata", {
  skip: process.platform === "win32",
}, () => {
  const root = path.join(__dirname, "..");
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-r-sh-lock-test-"));
  const fakeBin = path.join(temp, "bin");
  fs.mkdirSync(fakeBin);
  fs.copyFileSync(path.join(root, "r.sh"), path.join(temp, "r.sh"));
  fs.writeFileSync(path.join(fakeBin, "docker"), "#!/bin/sh\nexit 1\n");
  fs.chmodSync(path.join(fakeBin, "docker"), 0o755);
  try {
    const initialized = spawnSync("git", ["init", "-q"], {cwd: temp, encoding: "utf8"});
    assert.equal(initialized.status, 0, initialized.stderr);
    const result = spawnSync("bash", [path.join(temp, "r.sh")], {
      cwd: temp,
      encoding: "utf8",
      env: {...process.env, PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] || ""}`},
    });
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stderr, /Docker buildx is required/);
    assert.equal(fs.existsSync(path.join(temp, ".git", "mom-locks",
      `mom-r-sh-${process.getuid?.()}.build.lock`)), true);
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
});

test("r.sh keeps OpenCL out of GPU vendors and forwards controls once", {
  skip: process.platform === "win32",
}, () => {
  const root = path.join(__dirname, "..");
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-r-sh-opencl-test-"));
  const fakeBin = path.join(temp, "bin");
  const dockerArgs = path.join(temp, "docker-args");
  const verthashData = path.join(temp, "verthash.dat");
  fs.mkdirSync(fakeBin);
  fs.writeFileSync(verthashData, "fixture");
  fs.copyFileSync(path.join(root, "r.sh"), path.join(temp, "r.sh"));
  const docker = path.join(fakeBin, "docker");
  fs.writeFileSync(docker, [
    "#!/bin/sh",
    'case "$1:$2" in',
    "  buildx:version|image:inspect) exit 0 ;;",
    "  container:inspect) exit 1 ;;",
    '  run:*) shift; printf \'%s\\n\' "$@" > "$MOM_TEST_DOCKER_ARGS"; exit 0 ;;',
    "esac",
    "exit 1",
  ].join("\n"));
  fs.chmodSync(docker, 0o755);
  try {
    const initialized = spawnSync("git", ["init", "-q"], {cwd: temp, encoding: "utf8"});
    assert.equal(initialized.status, 0, initialized.stderr);
    const env = Object.fromEntries(Object.entries(process.env)
      .filter(([key]) => !key.startsWith("MOM_")));
    Object.assign(env, {
      MOM_DOCKER_GPUS: "0",
      MOM_CONTAINER_CPUS: "2",
      MOM_CONTAINER_MEMORY: "4g",
      MOM_CONTAINER_NETWORK: "none",
      MOM_CONTAINER_PIDS: "256",
      MOM_GPU_BACKEND: "opencl",
      MOM_NEXAPOW_PROFILE: "1",
      MOM_TEST_DOCKER_ARGS: dockerArgs,
      MOM_VERTHASH_DATA: verthashData,
      PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] || ""}`,
    });
    const result = spawnSync("bash", [path.join(temp, "r.sh"), "npm", "run", "test:gpu"], {
      cwd: temp, encoding: "utf8", env,
    });
    assert.equal(result.status, 0, result.stderr);
    const args = fs.readFileSync(dockerArgs, "utf8").trim().split(/\r?\n/);
    assert.deepEqual(args.filter((arg) => arg.startsWith("MOM_GPU_BACKEND")),
      ["MOM_GPU_BACKEND=opencl"]);
    assert.equal(args.some((arg) => arg.startsWith("MOM_GPU_TEST_VENDORS")), false);
    assert.equal(args.includes("MOM_NEXAPOW_PROFILE"), true);
    assert.deepEqual(args.filter((arg) => arg.startsWith("MOM_VERTHASH_DATA")),
      ["MOM_VERTHASH_DATA=/verthash.dat"]);
    assert.equal(args.includes("--device"), false);
    assert.equal(args.includes(
      `type=bind,source=${verthashData},target=/verthash.dat,readonly`), true);
    assert.equal(args.includes("--privileged"), false);
    /** @param {string} flag */
    const flagValue = (flag) => args[args.indexOf(flag) + 1];
    assert.equal(flagValue("--cap-drop"), "ALL");
    assert.deepEqual(args.filter((_arg, index) => args[index - 1] === "--cap-add"),
      ["CHOWN", "DAC_OVERRIDE", "FOWNER"]);
    assert.equal(flagValue("--security-opt"), "no-new-privileges:true");
    assert.equal(flagValue("--cpus"), "2");
    assert.equal(flagValue("--memory"), "4g");
    assert.equal(flagValue("--memory-swap"), "4g");
    assert.equal(flagValue("--pids-limit"), "256");
    assert.equal(flagValue("--network"), "none");
    assert.equal(args.some((arg) => arg.startsWith("MOM_CONTAINER_")), false);
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
});

test("deployment lanes require a test summary before passing", {
  skip: process.platform === "win32",
}, () => {
  const root = path.join(__dirname, "..");
  const deploySource = fs.readFileSync(path.join(root, "scripts", "test-deploy.sh"), "utf8");
  assert.match(deploySource, /apt-get install[^\n]*\bpython3\b/);
  assert.match(deploySource, /MOM_VERTHASH_DATA:\/verthash\.dat:ro/);
  assert.match(deploySource, /win-mom-dev-base\.qcow2/);
  assert.doesNotMatch(deploySource, /win-mom-dev\.qcow2/);
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-deploy-summary-test-"));
  const scripts = path.join(temp, "scripts");
  const fakeBin = path.join(temp, "bin");
  fs.mkdirSync(scripts);
  fs.mkdirSync(fakeBin);
  fs.copyFileSync(path.join(root, "scripts", "test-deploy.sh"),
    path.join(scripts, "test-deploy.sh"));
  for (const command of ["docker", "nvidia-smi"]) {
    const executable = path.join(fakeBin, command);
    fs.writeFileSync(executable, "#!/bin/sh\nexit 0\n");
    fs.chmodSync(executable, 0o755);
  }
  fs.writeFileSync(path.join(temp, "mom-v1.2.3-lin.tgz"), "fixture");
  try {
    const result = spawnSync("bash", [path.join(scripts, "test-deploy.sh")], {
      cwd: temp,
      encoding: "utf8",
      env: {
        ...process.env,
        PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] || ""}`,
        MOM_DEPLOY_TARGET: "linux-nvidia",
        MOM_DEPLOY_REUSE_ARCHIVE: "1",
        MOM_RELEASE_VERSION: "1.2.3",
      },
    });
    assert.equal(result.status, 1, result.stdout + result.stderr);
    assert.match(result.stdout, /nvidia-linux \(test summary missing\)/);
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
  const multiRelease = deploySource.match(/^test_windows_multi_release\(\) \{\n[\s\S]*?\n\}\n/m)?.[0];
  assert.ok(multiRelease, "mixed-vendor release function must exist");
  for (const status of [0, 37]) {
    /** @type {import("node:child_process").SpawnSyncReturns<string>} */
    const result = spawnSync("bash", ["-c", [
      "set +e",
      "mkdir() { :; }",
      "cp() { :; }",
      "run_windows_root() { return \"$MOM_DEPLOY_FIXTURE_EXIT\"; }",
      multiRelease,
      "test_windows_multi_release",
    ].join("\n")], {
      encoding: "utf8",
      env: {...process.env, MOM_DEPLOY_FIXTURE_EXIT: String(status),
        DEPLOY_SKIP_VECTORS: "0", MOM_DEPLOY_ALGO: "", WINDOWS_STAGE: "fixture",
        WINDOWS_ARCHIVE: "fixture.zip", WIN_RUN: "fixture-only"},
    });
    assert.equal(result.status, status, result.stdout + result.stderr);
    assert.equal(result.stdout.includes("MOM_TEST_SUMMARY 1 1 0 0"), status === 0);
  }
});

test("Linux release builds force portable compiler mode", {
  skip: process.platform === "win32",
}, () => {
  const root = path.join(__dirname, "..");
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-deploy-portable-test-"));
  const scripts = path.join(temp, "scripts");
  const fakeBin = path.join(temp, "bin");
  const rLog = path.join(temp, "r-env");
  const dockerLog = path.join(temp, "docker-invocations");
  fs.mkdirSync(scripts);
  fs.mkdirSync(fakeBin);
  fs.copyFileSync(path.join(root, "scripts", "test-deploy.sh"),
    path.join(scripts, "test-deploy.sh"));
  const fakeR = path.join(temp, "r.sh");
  fs.writeFileSync(fakeR, [
    "#!/bin/sh",
    'printf "%s\\n" "$MOM_PORTABLE_BUILD" > "$MOM_TEST_R_LOG"',
    'printf "%s\\n" "$MOM_GPU_BACKEND" >> "$MOM_TEST_R_LOG"',
    "exit 42",
  ].join("\n"));
  fs.chmodSync(fakeR, 0o755);
  const fakeDocker = path.join(fakeBin, "docker");
  fs.writeFileSync(fakeDocker, [
    "#!/bin/sh",
    'printf "%s\\n" "$*" >> "$MOM_TEST_DOCKER_LOG"',
    "exit 0",
  ].join("\n"));
  fs.chmodSync(fakeDocker, 0o755);
  try {
    const result = spawnSync("bash", [path.join(scripts, "test-deploy.sh")], {
      cwd: temp,
      encoding: "utf8",
      env: {
        ...process.env,
        PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] || ""}`,
        MOM_DEPLOY_TARGET: "linux-nvidia",
        MOM_DEPLOY_REUSE_ARCHIVE: "0",
        MOM_PORTABLE_BUILD: "0",
        MOM_RELEASE_VERSION: "1.2.3",
        MOM_TEST_DOCKER_LOG: dockerLog,
        MOM_TEST_R_LOG: rLog,
      },
    });
    assert.equal(result.status, 1, result.stdout + result.stderr);
    assert.equal(fs.readFileSync(rLog, "utf8"), "1\nall\n");
    assert.equal(fs.existsSync(dockerLog), false);
    assert.equal(fs.existsSync(path.join(temp, "mom-v1.2.3-lin.tgz")), false);
    assert.match(result.stdout, /Build Linux release \(exit 42\)/);
    assert.doesNotMatch(result.stdout + result.stderr, /Packaging Linux release archive/);
    assert.doesNotMatch(result.stdout + result.stderr, /Testing every nvidia GPU/);
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
});

test("Windows AdaptiveCpp exported source silences only its expected Git probe failure", () => {
  const build = fs.readFileSync(path.join(__dirname,
    "../scripts/build-windows-adaptivecpp-amd.ps1"), "utf8");
  const oldStart = build.indexOf("$rootGitStatusOld =");
  const newStart = build.indexOf("$rootGitStatusNew =", oldStart);
  const replaceStart = build.indexOf(
    "Replace-RequiredText $rootCmakeText $rootGitStatusOld $rootGitStatusNew", newStart);
  assert.ok(oldStart >= 0 && newStart > oldStart && replaceStart > newStart,
    "the checked exported-source workaround must remain intact");
  assert.match(build.slice(oldStart, newStart), /RESULT_VARIABLE GIT_STATUS/);
  assert.doesNotMatch(build.slice(oldStart, newStart), /ERROR_QUIET/);
  assert.match(build.slice(newStart, replaceStart), /RESULT_VARIABLE GIT_STATUS[\s\S]*ERROR_QUIET/);
});

test("combined builds preserve an unchanged linked CUDA image", () => {
  const build = fs.readFileSync(
    path.join(__dirname, "../scripts/combined-build.sh"), "utf8");
  assert.match(build, /link_target="\$ROOT\/build\/Release\/obj\.target\/mom\.node"/);
  assert.match(build, /\$WRAP -nt \$link_target/);
  assert.doesNotMatch(build, /octopus_archive/);
  assert.doesNotMatch(build, /\nrm -f build\/Release\/mom\.node build\/Release\/obj\.target\/mom\.node/);
});

test("Linux runner locks mutable build state without advisory GPU locks", () => {
  const runner = fs.readFileSync(path.join(__dirname, "../r.sh"), "utf8");
  const entrypoint = fs.readFileSync(
    path.join(__dirname, "../scripts/multicompiler-entrypoint.sh"), "utf8");

  assert.match(runner, /name="mom-\$backend"/);
  assert.doesNotMatch(runner, /--privileged/);
  assert.match(runner, /--cap-drop ALL/);
  assert.match(runner, /--security-opt no-new-privileges:true/);
  assert.match(runner, /add_drm_vendor 0x8086/);
  assert.match(runner, /add_drm_vendor 0x1002/);
  assert.match(runner,
    /sycl_cache_volume=\$\{MOM_SYCL_CACHE_VOLUME:-mom-sycl-cache-\$backend\}/);
  const reuseStart = runner.indexOf('if [ "$reuse_built_worker" = 1 ]; then');
  const reuseEnd = runner.indexOf("\nelse\n  flock 9", reuseStart);
  assert.ok(reuseStart >= 0 && reuseEnd > reuseStart, "reuse lock branch must exist");
  const reuseLocks = runner.slice(reuseStart, reuseEnd);
  assert.match(reuseLocks, /flock -s 9/);
  assert.match(runner, /exec 9>"\$build_lock"/);
  assert.doesNotMatch(runner, /MOM_GPU_LOCK_KEY|gpu_lock_key|exec 8>|\.gpu-/);
  assert.equal((runner.match(/\bflock\b/g) || []).length, 2);
  assert.match(runner,
    /if \[ "\$reuse_built_worker" != 1 \]; then[\s\S]*docker_flags\+=\(-it\)[\s\S]*docker_flags\+=\(-i\)[\s\S]*fi/);
  assert.match(runner,
    /elif \[ "\$container_stdin" = 1 \]; then[\s\S]*docker_flags\+=\(-i\)/);
  assert.doesNotMatch(runner, /docker rm -f "\$name"/);
  assert.match(runner,
    /if container_running=\$\(docker container inspect[\s\S]*if \[ "\$container_running" = true \][\s\S]*exit 2/);

  const reuseBranchStart = entrypoint.search(/case "\$\{MOM_REUSE_BUILT_WORKER:-0\}" in/);
  const buildTreeMutation = entrypoint.indexOf("platforms_hold=build-platforms-hold");
  assert.ok(reuseBranchStart >= 0 && buildTreeMutation > reuseBranchStart,
    "worker reuse branch must precede build-tree mutation");
  assert.match(entrypoint, /if \[\[ -e "\$platforms_hold" \|\| -L "\$platforms_hold" \]\]/);
  assert.match(entrypoint,
    /if \[\[ -e build \|\| -L build \]\] && \[\[ ! -d build \|\| -L build \]\]; then/);
  const reuseBranch = entrypoint.slice(reuseBranchStart, buildTreeMutation);
  assert.match(reuseBranch, /reused_worker="\$PWD\/build\/lin\/Release\/\$default\/mom\.node"/);
  assert.match(reuseBranch, /\[ ! -s "\$reused_worker" \]/);
  assert.match(entrypoint,
    /source_fingerprint\(\)[\s\S]*binding\.gyp[\s\S]*native[\s\S]*sycl[\s\S]*sha256sum/);
  assert.match(reuseBranch, /fingerprint_file="\$reused_worker\.sources\.sha256"/);
  assert.match(reuseBranch,
    /recorded_fingerprint[\s\S]*current_fingerprint[\s\S]*source fingerprint differs/);
  assert.doesNotMatch(reuseBranch, /-newer/);
  assert.match(reuseBranch, /exec "\$@"/);
});

test("coinstalled ROCm versions cannot reproduce the v0.8.0 mixed-runtime crash", {
  skip: process.platform !== "linux",
}, () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
  try {
    fs.mkdirSync(path.join(root, "native", "acpp-hip"), {recursive: true});
    fs.writeFileSync(path.join(root, "native", "acpp-hip", "mom.node"), "test");

    const rocm = path.join(root, "rocm", "core-7.14");
    const rocmBin = path.join(rocm, "bin");
    const rocmLib = path.join(rocm, "lib");
    const distroRocmLib = path.join(root, "usr-lib-rocm-7.1");
    const pathBin = path.join(root, "path-bin");
    fs.mkdirSync(rocmBin, {recursive: true});
    fs.mkdirSync(rocmLib);
    fs.mkdirSync(distroRocmLib);
    fs.mkdirSync(pathBin);
    fs.writeFileSync(path.join(rocmBin, "hipconfig"), "#!/bin/sh\n", {mode: 0o755});
    fs.symlinkSync(path.join(rocmBin, "hipconfig"), path.join(pathBin, "hipconfig"));
    for (const library of [
      "libamdhip64.so.7", "libamd_comgr.so.3", "libhsa-runtime64.so.1",
      "libhiprtc.so.7", "libhiprtc-builtins.so.7",
    ]) {
      fs.writeFileSync(path.join(rocmLib, library), "test");
    }
    const amd = policy.workerEnv("kawpow", {
      MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: path.join(root, "native"),
      PATH: pathBin, LD_LIBRARY_PATH: distroRocmLib,
    }, "linux");
    assert.equal(amd.MOM_SYCL_COMPILER, "acpp-hip");
    assert.equal(amd.LD_LIBRARY_PATH, [path.join(root, "native", "acpp-hip"),
      path.join(root, "native", "acpp-hip", "hipSYCL"), rocmLib, distroRocmLib]
      .join(path.delimiter));

    const incompleteRocm = path.join(root, "incomplete-rocm");
    fs.mkdirSync(path.join(incompleteRocm, "lib"), {recursive: true});
    for (const library of [
      "libamdhip64.so.7", "libamd_comgr.so.3", "libhsa-runtime64.so.1", "libhiprtc.so.7",
    ]) {
      fs.writeFileSync(path.join(incompleteRocm, "lib", library), "test");
    }
    const fallback = policy.workerEnv("kawpow", {
      MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: path.join(root, "native"),
      ROCM_PATH: incompleteRocm, PATH: pathBin,
    }, "linux");
    assert.ok(fallback.LD_LIBRARY_PATH?.split(path.delimiter).includes(rocmLib));
    assert.ok(!fallback.LD_LIBRARY_PATH?.split(path.delimiter)
      .includes(path.join(incompleteRocm, "lib")));
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test("vendor toolkit paths remain scoped to their matching Linux worker", {
  skip: process.platform !== "linux",
}, () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
  try {
    for (const key of ["acpp-cuda", "oneapi", "dpcpp-opencl"]) {
      fs.mkdirSync(path.join(root, "native", key), {recursive: true});
      fs.writeFileSync(path.join(root, "native", key, "mom.node"), "test");
    }
    const cuda = path.join(root, "cuda");
    fs.mkdirSync(path.join(cuda, "lib64"), {recursive: true});
    fs.writeFileSync(path.join(cuda, "lib64", "libnvrtc.so.12"), "test");
    const rocm = path.join(root, "rocm");
    fs.mkdirSync(path.join(rocm, "lib"), {recursive: true});
    for (const library of [
      "libamdhip64.so.7", "libamd_comgr.so.3", "libhsa-runtime64.so.1",
    ]) {
      fs.writeFileSync(path.join(rocm, "lib", library), "test");
    }
    const nvidia = policy.workerEnv("autolykos2", {
      MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: path.join(root, "native"), CUDA_PATH: cuda,
    }, "linux");
    assert.equal(nvidia.MOM_SYCL_COMPILER, "acpp-cuda");
    assert.ok(nvidia.LD_LIBRARY_PATH?.split(path.delimiter).includes(path.join(cuda, "lib64")));
    assert.ok(policy.workerEnv("autolykos2", {
      MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: path.join(root, "native"), CUDA_HOME: cuda,
    }, "linux").LD_LIBRARY_PATH?.split(path.delimiter).includes(path.join(cuda, "lib64")));

    const intel = policy.workerEnv("etchash", {
      MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: path.join(root, "native"),
      ROCM_PATH: rocm, CUDA_PATH: cuda,
    }, "linux");
    const opencl = policy.workerEnv("etchash", {
      MOM_GPU_BACKEND: "opencl", MOM_NATIVE_DIR: path.join(root, "native"),
      ROCM_PATH: rocm, CUDA_PATH: cuda,
    }, "linux");
    assert.ok(!intel.LD_LIBRARY_PATH?.includes(path.join(rocm, "lib")));
    assert.ok(!intel.LD_LIBRARY_PATH?.includes(path.join(cuda, "lib64")));
    assert.ok(!opencl.LD_LIBRARY_PATH?.includes(path.join(rocm, "lib")));
    assert.ok(!opencl.LD_LIBRARY_PATH?.includes(path.join(cuda, "lib64")));
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test("Windows portable workers share oneAPI JIT without crossing into nightly runtimes", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-runtime-cohort-"));
  try {
    for (const key of ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-hip"]) {
      fs.mkdirSync(path.join(root, key));
      fs.writeFileSync(path.join(root, key, "mom.node"), "fixture");
    }
    fs.writeFileSync(path.join(root, "oneapi", "sycl-jit.dll"), "matching-oneapi");
    fs.writeFileSync(path.join(root, "dpcpp", "sycl-jit.dll"), "poisoned-nightly");
    const inheritedPath = path.join(root, "dpcpp");
    const base = {MOM_NATIVE_DIR: root, Path: inheritedPath, UR_ADAPTERS_FORCE_LOAD: "caller-adapter",
      OCL_ICD_FILENAMES: "caller-icd", SYCL_CACHE_PERSISTENT: "0"};
    const portable = {...base, ...policy.workerEnv("etchash", {
      ...base, MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: "cpu",
    }, "win32")};
    const portablePath = portable["Path"];
    assert.ok(portablePath);
    const jit = portablePath.split(path.delimiter).map(dir => path.join(dir, "sycl-jit.dll"))
      .find(fs.existsSync);
    assert.equal(jit, path.join(root, "oneapi", "sycl-jit.dll"));
    assert.ok(jit);
    assert.equal(fs.readFileSync(jit, "utf8"), "matching-oneapi");
    assert.equal(portablePath.split(path.delimiter).at(-1), inheritedPath);
    assert.equal(portable.UR_ADAPTERS_FORCE_LOAD, base.UR_ADAPTERS_FORCE_LOAD);
    assert.equal(portable.OCL_ICD_FILENAMES, base.OCL_ICD_FILENAMES);
    assert.equal(portable.SYCL_CACHE_PERSISTENT, "0");
    assert.deepEqual(policy.workerEnv("etchash", {...base, MOM_GPU_BACKEND: "opencl",
      MOM_NATIVE_PATH: "caller-addon"}, "win32"), {});
    /** @type {Array<[string, string]>} */
    const vendors = [["nvidia", "dpcpp"], ["amd", "acpp-hip"]];
    for (const [gpu, key] of vendors) {
      const vendor = policy.workerEnv("etchash", {...base, MOM_GPU_BACKEND: gpu,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "90", MOM_GPU_INDEX: "2"}, "win32");
      assert.equal(vendor.MOM_RUNTIME_DIR, path.join(root, key));
      assert.equal(vendor["Path"], [path.join(root, key), path.join(root, key, "hipSYCL"),
        inheritedPath].join(path.delimiter));
    }
    const linux = policy.workerEnv("etchash", {MOM_NATIVE_DIR: root, MOM_GPU_BACKEND: "opencl",
      LD_LIBRARY_PATH: "caller-linux"}, "linux");
    assert.equal(linux.LD_LIBRARY_PATH, [path.join(root, "dpcpp-opencl"),
      path.join(root, "dpcpp-opencl", "hipSYCL"), path.join(root, "dpcpp"), "caller-linux"]
      .join(path.delimiter));
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
});
