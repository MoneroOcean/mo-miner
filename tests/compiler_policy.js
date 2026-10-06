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
