"use strict";

const compilerPolicy = require("../../compiler-policy");
const {hashTests} = require("../vectors");

const TEST_TIMEOUT_MS = 15 * 60 * 1000;
// QEMU executes the packaged Intel CPU ICD on non-Intel CI hosts; bound cold JIT/emulation
// separately even though the CPU dataset fixtures use compact geometry.
const EMULATED_CPU_TIMEOUT_MS = 60 * 60 * 1000;
// UHD 750 epoch-0 KawPow DAG construction measured exactly 120 minutes; retain an hour for hashing
// and cooperative teardown instead of expiring at the setup boundary.
const INTEGRATED_DAG_TIMEOUT_MS = 3 * 60 * 60 * 1000;
const supportedVendors = ["intel", "nvidia", "amd"];
const progpowAlgos = new Set(["kawpow", "firopow", "evrprogpow", "meowpow"]);
const integratedDagAlgos = new Set([...progpowAlgos, "etchash"]);

// CPU CI exercises shared kernels with bounded work, not GPU-sized datasets/solvers. Synthetic
// dataset gold is applied only to CPU clones; GPU and iGPU vectors keep their consensus gold.
const gpuVectors = hashTests.filter((definition) => definition.gpu && !definition.portableOnly);
const fastVectors = hashTests.filter((definition) => definition.syclCpu);
const gpuAlgos = [...new Set(gpuVectors.map((definition) => definition.job.algo))];
const gpuVectorsByAlgo = new Map(gpuAlgos.map((algo) => [
  algo,
  gpuVectors.filter((definition) => definition.job.algo === algo),
]));
const fastVectorByAlgo = new Map();

for (const definition of fastVectors) {
  if (!definition.gpu) {
    throw new Error(`SYCL CPU vector is absent from GPU coverage: ${definition.name}`);
  }
  const algo = definition.job.algo;
  if (fastVectorByAlgo.has(algo)) {
    throw new Error(`${algo} must have exactly one fast portable vector`);
  }
  fastVectorByAlgo.set(algo, definition);
}
for (const algo of gpuAlgos) {
  if (!fastVectorByAlgo.has(algo)) {
    throw new Error(`${algo} must have exactly one fast portable vector`);
  }
}

function requestedVendors() {
  const configured = process.env["MOM_GPU_TEST_VENDORS"];
  if (configured) {
    const vendors = configured.split(",").map((value) => value.trim().toLowerCase()).filter(Boolean);
    if (vendors.includes("all")) {
      if (vendors.length !== 1) {
        throw new Error("MOM_GPU_TEST_VENDORS=all cannot be combined with vendors");
      }
      return supportedVendors;
    }
    const invalid = vendors.filter((vendor) => !supportedVendors.includes(vendor));
    if (invalid.length) {
      throw new Error(`Unknown MOM_GPU_TEST_VENDORS: ${invalid.join(", ")}`);
    }
    return [...new Set(vendors)];
  }
  const backend = (process.env["MOM_GPU_BACKEND"] || "").toLowerCase();
  // The OpenCL lane discovers devices through its generic compatibility matrix. Treating
  // "opencl" as a hardware vendor would either reject it or duplicate the native vendor lanes.
  if (backend === "opencl") {return [];}
  const selected = supportedVendors.find((vendor) => backend.startsWith(vendor));
  return selected ? [selected] : supportedVendors;
}

function requestedAlgos() {
  const configured = process.env["MOM_GPU_TEST_ALGO"];
  if (!configured) {return gpuAlgos;}
  if (!gpuAlgos.includes(configured)) {
    throw new Error(`Unknown MOM_GPU_TEST_ALGO: ${configured}`);
  }
  return [configured];
}

/** @param {HashVectorDefinition} definition @returns {HashVectorDefinition} */
function copyDefinition(definition) {
  return JSON.parse(JSON.stringify(definition));
}

/** @param {HashVectorDefinition} copy @param {string} dev */
function replaceDevice(copy, dev) {
  copy.name = copy.name.replace(/gpu1/g, dev);
  if (typeof copy.job.dev !== "string") {throw new Error(`Vector ${copy.name} has no device`);}
  copy.job.dev = copy.job.dev.replace(/gpu1/g, dev);
}

/** @param {HashVectorDefinition} copy @param {string} backend */
function labelBackend(copy, backend) {
  if (typeof copy.job.dev !== "string") {throw new Error(`Vector ${copy.name} has no device`);}
  copy.name = copy.name.replace(copy.job.dev, `${copy.job.dev}:${backend}`);
}

/** @param {HashVectorDefinition} definition @param {string} vendor @param {string} dev @param {string} backend @returns {HashVectorDefinition} */
function cloneForDiscreteGpu(definition, vendor, dev, backend) {
  const copy = copyDefinition(definition);
  replaceDevice(copy, dev);
  copy.job["backend"] = backend;
  labelBackend(copy, backend);
  copy.env = {
    ...copy.env,
    MOM_GPU_BACKEND: vendor,
    // The release test parent has one launcher-selected control addon. Each vendor case is a new
    // process and must derive its own control runtime before it derives device-specific tuning.
    MOM_NATIVE_PATH: undefined,
    MOM_NATIVE_PATH_LAUNCHER_DEFAULT: undefined,
  };
  // Discrete vectors must exercise the mining DAG path; CPU/portable vectors retain light mode.
  if (copy.job.algo === "octopus") {
    copy.env["MOM_OCTOPUS_TEST_FULL_DAG"] = "1";
    // The worker proves native execution only when the actual device supports its matrix ISA.
    copy.env["MOM_OCTOPUS_TEST_NATIVE"] = backend === "sycl-native" ? "1" : undefined;
  }
  return copy;
}

/** @param {"cpu" | "gpu"} deviceType @returns {Record<string, string | undefined>} */
function openclSyclEnv(deviceType) {
  const base = {
    MOM_GPU_BACKEND: "opencl",
    MOM_OPENCL_DEVICE_TYPE: deviceType,
    MOM_COMPILER_POLICY_STRICT: "1",
    // OpenCL compatibility checks must select the portable runtime, not an inherited addon.
    MOM_NATIVE_PATH: undefined,
  };
  return {...base, ...compilerPolicy.workerEnv("__control__", {...process.env, ...base})};
}

/** @param {HashVectorDefinition} definition @param {string} dev @param {"cpu" | "gpu"} deviceType @returns {HashVectorDefinition} */
function cloneForOpenclSycl(definition, dev, deviceType) {
  const copy = copyDefinition(definition);
  replaceDevice(copy, dev);
  copy.gpu = deviceType === "gpu";
  if (deviceType === "cpu") {copy.expected = definition.syclCpuExpected ?? copy.expected;}
  const minimumTimeout = deviceType === "cpu" &&
      process.env["MOM_RELEASE_EMULATE_INTEL_CPU"] === "1"
    ? EMULATED_CPU_TIMEOUT_MS
    : TEST_TIMEOUT_MS;
  copy.timeoutMs = Math.max(copy.timeoutMs || 0, minimumTimeout);
  copy.job["backend"] = "sycl-opencl";
  labelBackend(copy, copy.job["backend"]);
  copy.env = {
    ...copy.env, ...openclSyclEnv(deviceType),
    MOM_SYCL_PORTABLE_TEST: deviceType === "cpu" ? "1" : undefined,
  };
  return copy;
}

/** @param {HashVectorDefinition} definition @param {string} dev @param {string} backend @returns {HashVectorDefinition} */
function cloneForIntelIntegrated(definition, dev, backend) {
  const copy = copyDefinition(definition);
  replaceDevice(copy, dev);
  copy.job["backend"] = backend;
  labelBackend(copy, backend);
  copy.env = {
    ...copy.env,
    MOM_COMPILER_POLICY_STRICT: "1",
    MOM_GPU_BACKEND: "intel",
  };
  const minimumTimeout = integratedDagAlgos.has(copy.job.algo)
    ? INTEGRATED_DAG_TIMEOUT_MS
    : TEST_TIMEOUT_MS;
  copy.timeoutMs = Math.max(copy.timeoutMs || 0, minimumTimeout);
  return copy;
}

/** @param {string} algo @returns {HashVectorDefinition[]} */
function gpuVectorsFor(algo) {
  return gpuVectorsByAlgo.get(algo) || [];
}

/** @param {string} algo @returns {HashVectorDefinition | undefined} */
function fastVectorFor(algo) {
  return fastVectorByAlgo.get(algo);
}

/** @param {Record<string, string> | undefined} params @param {string} algo @param {string} dev */
function configuredDeviceSupports(params, algo, dev) {
  const configured = params && params[algo];
  if (!configured) {return false;}
  return configured.split(",").some((entry) => {
    const match = entry.trim().match(/^([a-z]+\d+)/i);
    return match && match[1] === dev;
  });
}

module.exports = {
  TEST_TIMEOUT_MS,
  cloneForDiscreteGpu,
  cloneForIntelIntegrated,
  cloneForOpenclSycl,
  configuredDeviceSupports,
  fastVectorFor,
  gpuAlgos,
  gpuVectorsFor,
  openclSyclEnv,
  requestedAlgos,
  requestedVendors,
  supportedVendors,
};
