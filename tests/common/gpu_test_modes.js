"use strict";

const compilerPolicy = require("../../compiler-policy");
const {intelIgpuAlgos} = require("./compatibility_algos");
const {hashTests} = require("../vectors");

const TEST_TIMEOUT_MS = 15 * 60 * 1000;
const supportedVendors = ["intel", "nvidia", "amd"];

const gpuVectors = hashTests.filter((definition) => definition.gpu);
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
  const configured = process.env.MOM_GPU_TEST_VENDORS;
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
  const backend = (process.env.MOM_GPU_BACKEND || "").toLowerCase();
  // The OpenCL lane discovers devices through its generic compatibility matrix. Treating
  // "opencl" as a hardware vendor would either reject it or duplicate the native vendor lanes.
  if (backend === "opencl") {return [];}
  const selected = supportedVendors.find((vendor) => backend.startsWith(vendor));
  return selected ? [selected] : supportedVendors;
}

function copyDefinition(definition) {
  return JSON.parse(JSON.stringify(definition));
}

function replaceDevice(copy, dev) {
  copy.name = copy.name.replace(/gpu1/g, dev);
  copy.job.dev = copy.job.dev.replace(/gpu1/g, dev);
}

function labelBackend(copy, backend) {
  copy.name = copy.name.replace(copy.job.dev, `${copy.job.dev}:${backend}`);
}

function cloneForDiscreteGpu(definition, vendor, dev, backend) {
  const copy = copyDefinition(definition);
  replaceDevice(copy, dev);
  copy.job.backend = backend;
  labelBackend(copy, backend);
  copy.env = {...copy.env, MOM_GPU_BACKEND: vendor};
  return copy;
}

function openclSyclEnv(deviceType) {
  const base = {
    MOM_GPU_BACKEND: "opencl",
    MOM_OPENCL_DEVICE_TYPE: deviceType,
    MOM_COMPILER_POLICY_STRICT: "1",
  };
  return {...base, ...compilerPolicy.workerEnv("__control__", {...process.env, ...base})};
}

function cloneForOpenclSycl(definition, dev, deviceType) {
  const copy = copyDefinition(definition);
  replaceDevice(copy, dev);
  copy.gpu = deviceType === "gpu";
  copy.timeoutMs = Math.max(copy.timeoutMs || 0, TEST_TIMEOUT_MS);
  copy.job.backend = "sycl-opencl";
  labelBackend(copy, copy.job.backend);
  copy.env = {...copy.env, ...openclSyclEnv(deviceType)};
  return copy;
}

function cloneForIntelIntegrated(definition, dev) {
  const copy = copyDefinition(definition);
  replaceDevice(copy, dev);
  copy.job.backend = "sycl-l0";
  labelBackend(copy, copy.job.backend);
  copy.env = {...copy.env, MOM_GPU_BACKEND: "intel"};
  copy.timeoutMs = Math.max(copy.timeoutMs || 0, TEST_TIMEOUT_MS);
  return copy;
}

function gpuVectorsFor(algo) {
  return gpuVectorsByAlgo.get(algo) || [];
}

function fastVectorFor(algo) {
  return fastVectorByAlgo.get(algo);
}

function supportsIntelIgpu(algo) {
  return intelIgpuAlgos.has(algo);
}

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
  requestedVendors,
  supportedVendors,
  supportsIntelIgpu,
};
