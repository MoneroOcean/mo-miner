"use strict";

const assert = require("node:assert/strict");
const {test} = require("node:test");
const policy = require("../compiler-policy");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const gpuTuning = require("../gpu-tuning");
const helper = require("../helper");
const {
  parseDiscreteGpuDevices, parseGpuDevices,
} = require("./common/miner_command");

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
  assert.deepEqual(gpuTuning.parseDeviceList(
    "gpu1*[dag_chunk=0]", "kawpow"
  )[0].tuning, {dag_chunk: 0});
  assert.deepEqual(gpuTuning.parseDeviceList(
    "gpu1*[cache_block=0]", "pearlhash"
  )[0].tuning, {cache_block: 0});
});

test("empirical GPU tuning candidates stay bounded around portable heuristics", () => {
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
  assert.equal(formats("zelhash", "gpu1*[slots=4480]").length, 1);
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
  const first = {algo: "pearlhash", pearlhash_n: 131072, pearlhash_k: 4096, pearlhash_rank: 256};
  gpuTuning.applyNativeJobTuning(
    first, gpuTuning.parseDeviceEntry("gpu1*[m=8192;k=2048]", "pearlhash"), "pearlhash");
  assert.deepEqual(first, {
    algo: "pearlhash", dev: "gpu1", intensity: 8192,
    pearlhash_n: 8192, pearlhash_k: 2048, pearlhash_rank: 256,
  });
  const second = {algo: "pearlhash", pearlhash_n: 131072, pearlhash_k: 4096, pearlhash_rank: 256};
  gpuTuning.applyNativeJobTuning(second,
    gpuTuning.parseDeviceEntry("gpu1*[m=16384;n=32768;rank=128]", "pearlhash"), "pearlhash");
  assert.deepEqual(second, {
    algo: "pearlhash", dev: "gpu1", intensity: 16384,
    pearlhash_n: 32768, pearlhash_k: 4096, pearlhash_rank: 128,
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
  const job = {algo: "c29"};
  gpuTuning.applyNativeJobTuning(job, c29, "c29");
  assert.deepEqual(job, {algo: "c29", dev: "gpu1", intensity: 1});
  assert.deepEqual(gpuTuning.tuningEnvironment("c29", c29.tuning),
    {MOM_C29_SEED_LOCAL_SIZE: "128"});
  assert.deepEqual(gpuTuning.tuningEnvironment("beamhash3", {workgroup: 256}), {
    MOM_BEAMHASH3_WORKGROUP: "256",
    MOM_BEAMHASH3_COMPACT_WG: "256",
  });
});

test("portable Pearl tuning maps generic controls onto relevant vendor kernels", () => {
  assert.deepEqual(gpuTuning.tuningEnvironment("pearlhash", {
    workgroup: 128, cache_block: 32, tile: "4x2",
  }), {
    MOM_PEARLHASH_AMD_WMMA_THREADS: "128",
    MOM_PEARLHASH_AMD_WMMA_CACHE_BLOCK: "32",
    MOM_PEARLHASH_AMD_DP4A_CACHE_BLOCK: "32",
    MOM_PEARLHASH_CU_BLK: "32",
    MOM_PEARLHASH_AMD_DP4A_TILE: "4x2",
  });
});

test("GPU compiler Markdown selects platform defaults and overrides", () => {
  assert.equal(policy.selection("etchash", "intel", "linux").key, "oneapi");
  assert.equal(policy.selection("fishhash", "intel", "linux").key, "oneapi");
  assert.equal(policy.selection("karlsenhashv2", "intel", "linux").key, "oneapi");
  assert.equal(policy.selection("autolykos2", "nvidia", "linux").key, "acpp-cuda");
  assert.equal(policy.selection("beamhash3", "nvidia", "linux").key, "dpcpp");
  assert.equal(policy.selection("fishhash", "nvidia", "linux").key, "acpp-cuda");
  assert.equal(policy.selection("karlsenhashv2", "nvidia", "linux").key, "acpp-cuda");
  assert.equal(policy.selection("zelhash", "nvidia", "linux").key, "dpcpp");
  assert.equal(policy.selection("pearlhash", "nvidia", "linux").backend, "native");
  assert.deepEqual(policy.selection("pearlhash", "nvidia", "linux").pearlhashProfile,
    {m: 65536, n: 65536, k: 4096, rank: 256});
  assert.equal(policy.selection("autolykos2", "nvidia", "win32").key, "acpp-cuda");
  assert.equal(policy.selection("beamhash3", "nvidia", "win32").key, "dpcpp");
  assert.equal(policy.selection("cn/gpu", "nvidia", "win32").key, "dpcpp");
  assert.equal(policy.selection("cn/gpu", "nvidia", "win32").backend, "native");
  assert.equal(policy.selection("fishhash", "nvidia", "win32").key, "acpp-cuda");
  assert.equal(policy.selection("karlsenhashv2", "nvidia", "win32").key, "acpp-cuda");
  assert.equal(policy.selection("etchash", "nvidia", "win32").key, "dpcpp");
  assert.equal(policy.selection("pearlhash", "nvidia", "win32").backend, "native");
  assert.deepEqual(policy.selection("pearlhash", "nvidia", "win32").pearlhashProfile,
    {m: 65536, n: 65536, k: 4096, rank: 256});
  assert.equal(policy.selection("autolykos2", "amd", "linux").key, "acpp-hip");
  assert.equal(policy.selection("beamhash3", "amd", "linux").key, "acpp-hip");
  assert.equal(policy.selection("karlsenhashv2", "amd", "linux").key, "acpp-hip");
  assert.equal(policy.selection("pearlhash", "amd", "linux").key, "acpp-hip");
  assert.equal(policy.selection("pearlhash", "amd", "linux").backend, "native");
  assert.deepEqual(policy.selection("pearlhash", "amd", "linux").pearlhashProfile,
    {m: 32768, n: 32768, k: 2048, rank: 128});
  assert.equal(policy.selection("etchash", "amd", "linux").backend, "sycl");
  assert.equal(policy.selection("autolykos2", "amd", "linux").backend, "sycl-native");
  assert.equal(policy.selection("pearlhash", "intel", "linux").pearlhashProfile, null);
  assert.equal(policy.selection("etchash", "amd", "win32").key, "acpp-hip");
  assert.equal(policy.selection("pearlhash", "amd", "win32").key, "acpp-hip");
  assert.equal(policy.selection("pearlhash", "amd", "win32").backend, "native");
  assert.deepEqual(policy.selection("pearlhash", "amd", "win32").pearlhashProfile,
    {m: 131072, n: 131072, k: 2048, rank: 128});
  assert.equal(policy.selection("cn/gpu", "intel", "linux").backend, "sycl-opencl");
  assert.equal(policy.selection("etchash", "intel", "linux").backend, "sycl");
  assert.equal(policy.selection("pearlhash", "intel", "linux").backend, "sycl-native");
  assert.equal(policy.selection("fishhash", "nvidia", "linux").backend, "sycl-native");
  assert.equal(policy.selection("c29", "nvidia", "linux").backend, "sycl");
  assert.equal(policy.selection("etchash", "opencl", "linux").backend, "sycl-opencl");
  assert.equal(policy.selection("etchash", "opencl", "linux").key, "dpcpp-opencl");
  assert.equal(policy.selection("etchash", "opencl", "linux").allocation,
    "buffers where available; USM fallback");
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
    LD_LIBRARY_PATH: "/system/lib"
  }, "linux");
  assert.equal(env.MOM_SYCL_COMPILER, "acpp-cuda");
  assert.equal(env.ACPP_VISIBILITY_MASK, "cuda");
  assert.equal(env.CUDA_VISIBLE_DEVICES, "2");
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
    Path: "C:\\Windows\\System32"
  }, "win32");
  assert.equal(env.MOM_SYCL_COMPILER, "acpp-hip");
  assert.equal(env.ACPP_VISIBILITY_MASK, "hip");
  assert.equal(env.HIP_VISIBLE_DEVICES, "3");
  assert.equal(env.MOM_RUNTIME_DIR, path.join(root, "acpp-hip"));
  assert.equal(env.PATH, [path.join(root, "acpp-hip"),
    path.join(root, "acpp-hip", "hipSYCL"), "C:\\Windows\\System32"].join(path.delimiter));
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
  assert.equal(policy.workerEnv("etchash", {MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root}, "linux")
    .ONEAPI_DEVICE_SELECTOR, "cuda:gpu");
  assert.equal(policy.workerEnv("etchash", {
    MOM_GPU_BACKEND: "nvidia", MOM_GPU_INDEX: "2", MOM_NATIVE_DIR: root
  }, "linux").ONEAPI_DEVICE_SELECTOR, "cuda:2");
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
  assert.equal(windowsPortable.PATH, [path.join(root, "dpcpp-opencl"),
    path.join(root, "dpcpp-opencl", "hipSYCL"), path.join(root, "dpcpp"),
    "C:\\Windows\\System32"].join(path.delimiter));
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
  void set_fn(decltype(nullptr)) { active = false; }
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
