"use strict";

const assert = require("node:assert/strict");
const {test} = require("node:test");
const policy = require("../compiler-policy");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {
  parseDiscreteGpuDevices, parseGpuDevices,
} = require("./common/miner_command");

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
  assert.deepEqual(policy.parseReportedAlgoParam("gpu1*8:auto[sycl-native]"),
    {dev: "gpu1*8", backend: "auto"});
  assert.deepEqual(policy.parseReportedAlgoParam("gpu2*1:sycl"),
    {dev: "gpu2*1", backend: "sycl"});
  assert.deepEqual(policy.parseReportedAlgoParam("cpu*8"), {dev: "cpu*8"});
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
