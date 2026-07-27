"use strict";

const {describe, it} = require("node:test");

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const compilerPolicy = require("../../compiler-policy");
const {validateDirectory} = require("../../scripts/validate-portable-opencl");
const {
  getFirstSyclCpuDevice,
  getGpuDevices,
  runMinerTest,
  runNode,
} = require("./miner_command");
const {
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
  supportsIntelIgpu,
} = require("./gpu_test_modes");

const BACKEND_TIMEOUT_MS = 3 * 60 * 60 * 1000;
const ALGO_TIMEOUT_MS = 4 * 60 * 60 * 1000;
const MATRIX_TIMEOUT_MS = 8 * 60 * 60 * 1000;

function addCase(plan, backend, kind, testCase) {
  if (!plan.has(backend)) {plan.set(backend, {discrete: [], serial: []});}
  plan.get(backend)[kind].push(testCase);
}

function unavailableCase(name, reason, required = false) {
  return {name, reason, required, definitions: []};
}

function deviceCase(name, definitions, select) {
  return {name, definitions, select};
}

function deviceVendor(description) {
  if (/\bNVIDIA\b/i.test(description)) {return "nvidia";}
  if (/\bAMD\b|\bRadeon\b/i.test(description)) {return "amd";}
  if (/\bIntel(?:\(R\))?\b/i.test(description)) {return "intel";}
  return "other";
}

async function runCase(t, testCase) {
  if (testCase.reason) {
    await t.test(testCase.name, {skip: testCase.required ? false : testCase.reason}, () => {
      if (testCase.required) {throw new Error(testCase.reason);}
    });
    return;
  }
  for (const definition of testCase.definitions) {
    const selected = testCase.select(definition);
    await t.test(`${testCase.name} — ${selected.name}`,
      {timeout: selected.timeoutMs || TEST_TIMEOUT_MS}, async () => {
        const result = await runMinerTest(selected);
        if (result.skipped) {throw new Error(`${testCase.name}: ${result.reason}`);}
      });
  }
}

async function runBackend(t, backend, lane) {
  await t.test(backend, {timeout: BACKEND_TIMEOUT_MS, concurrency: true}, async (backendTest) => {
    // A case owns one physical discrete device and runs all of that algorithm's vectors in order.
    // Different devices are independent and intentionally run together. Register vector tests
    // directly under the backend so the device identity is visible without another suite level.
    await Promise.all(lane.discrete.map((testCase) => runCase(backendTest, testCase)));

    // CPU and integrated GPUs share host memory and/or the desktop display. Keep every such case
    // behind the discrete batch and run them one at a time.
    for (const testCase of lane.serial) {
      await runCase(backendTest, testCase);
    }
  });
}

function selectedBackend(algo, vendor) {
  const selected = compilerPolicy.selection(algo, vendor, process.platform);
  return selected ? selected.backend : "sycl";
}

function backendGroup(backend) {
  return backend === "native" || backend === "sycl-native" ? backend : "sycl";
}

function discreteDiscovery() {
  const cache = new Map();
  return (algo, vendor) => {
    const key = `${algo}:${vendor}`;
    if (!cache.has(key)) {
      cache.set(key, getGpuDevices(vendor, {algo, integrated: false}));
    }
    return cache.get(key);
  };
}

function sharedDiscovery() {
  let openclGpu;
  let openclCpu;
  let intelIntegrated;
  return {
    openclGpu(algo) {
      if (!openclGpu) {
        const env = {MOM_OPENCL_DEVICE_TYPE: "gpu", MOM_COMPILER_POLICY_STRICT: "1"};
        openclGpu = getGpuDevices("opencl", {algo, integrated: false, env});
      }
      return openclGpu;
    },
    openclCpu() {
      if (!openclCpu) {openclCpu = getFirstSyclCpuDevice(openclSyclEnv("cpu"));}
      return openclCpu;
    },
    intelIntegrated(algo) {
      if (!intelIntegrated) {
        intelIntegrated = getGpuDevices("intel", {
          algo,
          integrated: true,
          backend: "sycl-l0",
        });
      }
      return intelIntegrated;
    },
  };
}

async function addDiscreteCases(plan, algo, vendors, discover) {
  const definitions = gpuVectorsFor(algo);
  const discoveries = await Promise.all(vendors.map(async (vendor) => ({
    vendor,
    result: await discover(algo, vendor),
  })));

  for (const {vendor, result} of discoveries) {
    const backend = selectedBackend(algo, vendor);
    if (result.skipped) {
      const required = process.env.MOM_REQUIRE_GPU_TESTS === "1" && vendors.length === 1;
      addCase(plan, backendGroup(backend), "discrete",
        unavailableCase(`${vendor} (unavailable)`, result.reason, required));
      continue;
    }

    for (const device of result.devices) {
      const name = `${vendor} ${device.dev}: ${device.description}`;
      if (!configuredDeviceSupports(result.params, algo, device.dev)) {
        addCase(plan, backendGroup(backend), "discrete", unavailableCase(name,
          `${algo} is not available on ${device.description}`));
        continue;
      }
      if (backend === "native") {
        // Native source-JIT is an optimization. Its portable SYCL recovery path remains mandatory.
        addCase(plan, "sycl", "discrete", deviceCase(`${name} (fallback)`, definitions,
          (definition) => cloneForDiscreteGpu(definition, vendor, device.dev, "sycl")));
        addCase(plan, "native", "discrete", deviceCase(name, definitions,
          (definition) => cloneForDiscreteGpu(definition, vendor, device.dev, "native")));
      } else {
        addCase(plan, backendGroup(backend), "discrete", deviceCase(name, definitions,
          (definition) => cloneForDiscreteGpu(definition, vendor, device.dev, backend)));
      }
    }
  }
}

async function addGenericCases(plan, algo, discover) {
  const definition = fastVectorFor(algo);
  const openclResult = await discover.openclGpu(algo);
  if (openclResult.skipped) {
    addCase(plan, "sycl", "discrete",
      unavailableCase("OpenCL GPU devices (unavailable)", openclResult.reason));
  } else {
    for (const device of openclResult.devices) {
      const vendor = deviceVendor(device.description);
      const name = `OpenCL ${vendor} ${device.dev}: ${device.description}`;
      // cn/gpu on Intel already selects this exact OpenCL artifact and transport. Other direct
      // lanes use a native transport or tuned artifact, so one fast OpenCL vector remains useful.
      if (["intel", "nvidia", "amd"].includes(vendor) &&
          selectedBackend(algo, vendor) === "sycl-opencl") {continue;}
      if (!configuredDeviceSupports(openclResult.params, algo, device.dev)) {
        addCase(plan, "sycl", "serial",
          unavailableCase(name, `${algo} is not available on ${device.description}`));
        continue;
      }
      // The OpenCL and native-transport entries can name the same physical GPU. Run this short
      // compatibility case after the parallel discrete batch to avoid overlapping work on it.
      addCase(plan, "sycl", "serial", deviceCase(name, [definition],
        (entry) => cloneForOpenclSycl(entry, device.dev, "gpu")));
    }
  }

  const openclCpuResult = await discover.openclCpu();
  if (openclCpuResult.skipped) {
    addCase(plan, "sycl", "serial",
      unavailableCase("CPU device (unavailable)", openclCpuResult.reason));
  } else {
    addCase(plan, "sycl", "serial",
      deviceCase(`${openclCpuResult.dev}: ${openclCpuResult.description}`, [definition],
        (entry) => cloneForOpenclSycl(entry, openclCpuResult.dev, "cpu")));
  }

  if (!supportsIntelIgpu(algo)) {return;}
  const integratedResult = await discover.intelIntegrated(algo);
  if (integratedResult.skipped) {
    addCase(plan, "sycl", "serial",
      unavailableCase("Intel integrated GPU (unavailable)", integratedResult.reason));
    return;
  }
  for (const device of integratedResult.devices) {
    const name = `intel ${device.dev}: ${device.description}`;
    if (!configuredDeviceSupports(integratedResult.params, algo, device.dev)) {
      addCase(plan, "sycl", "serial",
        unavailableCase(name, `${algo} is not available on ${device.description}`));
      continue;
    }
    addCase(plan, "sycl", "serial", deviceCase(name, [definition],
      (entry) => cloneForIntelIntegrated(entry, device.dev)));
  }
}

async function addPortableCpuCase(plan, algo, discover) {
  const definition = fastVectorFor(algo);
  const result = await discover.openclCpu();
  if (result.skipped) {
    addCase(plan, "sycl", "serial", unavailableCase("CPU device (unavailable)", result.reason));
    return;
  }
  addCase(plan, "sycl", "serial",
    deviceCase(`${result.dev}: ${result.description}`, [definition],
      (entry) => cloneForOpenclSycl(entry, result.dev, "cpu")));
}

async function openclImageGuard() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-opencl-spv-"));
  try {
    const result = await runNode(["mom.js", "algo_params"], {
      cwd: directory,
      timeoutMs: TEST_TIMEOUT_MS,
      env: {...openclSyclEnv("cpu"), SYCL_DUMP_IMAGES: "1"},
    });
    if (result.error || result.code !== 0) {
      throw new Error(`Unable to dump generic OpenCL images:\n${result.stderr || result.error}`);
    }
    const report = validateDirectory(directory);
    if (report.errors.length) {throw new Error(report.errors.join("\n"));}
  } finally {
    fs.rmSync(directory, {recursive: true, force: true});
  }
}

function defineGpuTestMatrix({discreteOnly = false, portableCpuOnly = false} = {}) {
  const vendors = requestedVendors();
  const discoverDiscrete = discreteDiscovery();
  const discoverShared = sharedDiscovery();
  const title = portableCpuOnly
    ? "Portable SYCL CPU proof-of-work hash vectors"
    : discreteOnly
      ? "GPU proof-of-work hash vectors (algorithm-centric discrete devices)"
      : "GPU proof-of-work hash vectors (algorithm-centric backend matrix)";

  describe(title, {timeout: MATRIX_TIMEOUT_MS, concurrency: 1}, () => {
    if (!discreteOnly) {
      it("generic OpenCL image guard", {timeout: TEST_TIMEOUT_MS}, openclImageGuard);
    }

    for (const algo of gpuAlgos) {
      it(algo, {timeout: ALGO_TIMEOUT_MS, concurrency: false}, async (algoTest) => {
        const plan = new Map();
        if (portableCpuOnly) {
          await addPortableCpuCase(plan, algo, discoverShared);
        } else {
          await addDiscreteCases(plan, algo, vendors, discoverDiscrete);
          if (!discreteOnly) {await addGenericCases(plan, algo, discoverShared);}
        }
        for (const backend of [...plan.keys()].sort()) {
          await runBackend(algoTest, backend, plan.get(backend));
        }
      });
    }
  });
}

module.exports = {defineGpuTestMatrix};
