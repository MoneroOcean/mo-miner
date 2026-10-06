"use strict";

const {describe, it} = require("node:test");

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const compilerPolicy = require("../../compiler-policy");
const {validateDirectory} = require("../../scripts/validate-portable-opencl");
const {
  gpuDeviceDiscoveryKey,
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
  gpuVectorsFor,
  openclSyclEnv,
  requestedAlgos,
  requestedVendors,
} = require("./gpu_test_modes");

/** @typedef {{dev: string, description: string, integrated: boolean}} GpuDevice */
/** @typedef {{skipped: true, reason: string}} SkippedDiscovery */
/** @typedef {{skipped: false, devices: GpuDevice[], params: Record<string, string>}} GpuDiscovery */
/** @typedef {SkippedDiscovery | GpuDiscovery} GpuDiscoveryResult */
/** @typedef {{skipped: false, dev: string, description: string} | SkippedDiscovery} CpuDiscoveryResult */
/** @typedef {{algo?: string, integrated?: boolean | null, backend?: string, env?: Record<string, string | undefined>}} DiscoveryOptions */
/** @typedef {{name: string, reason?: string, required?: boolean, definitions: HashVectorDefinition[], select?: (definition: HashVectorDefinition) => HashVectorDefinition}} MatrixCase */
/** @typedef {{discrete: MatrixCase[], serial: MatrixCase[]}} MatrixLane */
/** @typedef {Map<string, MatrixLane>} MatrixPlan */
/** @typedef {(algo: string, vendor: string) => Promise<GpuDiscoveryResult>} DiscreteDiscover */
/** @typedef {{openclGpu: (algo: string) => Promise<GpuDiscoveryResult>, openclCpu: () => Promise<CpuDiscoveryResult>, intelIntegrated: (algo: string, backend: string) => Promise<GpuDiscoveryResult>}} SharedDiscover */
/** @typedef {import("node:test").TestContext} TestContext */

const BACKEND_TIMEOUT_MS = 3 * 60 * 60 * 1000;
const ALGO_TIMEOUT_MS = 4 * 60 * 60 * 1000;
const MATRIX_TIMEOUT_MS = 8 * 60 * 60 * 1000;

/** @param {MatrixPlan} plan @param {string} backend @param {"discrete" | "serial"} kind @param {MatrixCase} testCase */
function addCase(plan, backend, kind, testCase) {
  if (!plan.has(backend)) {plan.set(backend, {discrete: [], serial: []});}
  const lane = plan.get(backend);
  if (!lane) {throw new Error(`Missing matrix lane ${backend}`);}
  lane[kind].push(testCase);
}

/** @param {string} name @param {string} reason @param {boolean} [required] @returns {MatrixCase} */
function unavailableCase(name, reason, required = false) {
  return {name, reason, required, definitions: []};
}

/** @param {string} name @param {HashVectorDefinition[]} definitions @param {(definition: HashVectorDefinition) => HashVectorDefinition} select @returns {MatrixCase} */
function deviceCase(name, definitions, select) {
  return {name, definitions, select};
}

/** @param {string} description */
function deviceVendor(description) {
  if (/\bNVIDIA\b/i.test(description)) {return "nvidia";}
  if (/\bAMD\b|\bRadeon\b/i.test(description)) {return "amd";}
  if (/\bIntel(?:\(R\))?\b/i.test(description)) {return "intel";}
  return "other";
}

/** @param {TestContext} t @param {MatrixCase} testCase */
async function runCase(t, testCase) {
  if (testCase.reason) {
    await t.test(testCase.name, {skip: testCase.required ? false : testCase.reason}, () => {
      if (testCase.required) {throw new Error(testCase.reason);}
    });
    return;
  }
  const select = testCase.select;
  if (!select) {throw new Error(`Test case ${testCase.name} has no selector`);}
  for (const definition of testCase.definitions) {
    const selected = select(definition);
    await t.test(`${testCase.name} — ${selected.name}`,
      {timeout: selected.timeoutMs || TEST_TIMEOUT_MS}, async () => {
        const result = await runMinerTest(selected);
        if (result.skipped) {throw new Error(`${testCase.name}: ${result.reason}`);}
      });
  }
}

/** @param {TestContext} t @param {string} backend @param {MatrixLane} lane */
async function runBackend(t, backend, lane) {
  await t.test(backend, {timeout: BACKEND_TIMEOUT_MS, concurrency: true}, async (backendTest) => {
    // A case owns one physical discrete device and runs all of that algorithm's vectors in order.
    // Different devices are independent and intentionally run together. Register vector tests
    // directly under the backend so the device identity is visible without another suite level.
    // Each case owns one physical discrete GPU. Run two cases at a time to use parallel hardware
    // without putting all three host cards under load together.
    for (let i = 0; i < lane.discrete.length; i += 2) {
      const results = await Promise.allSettled(lane.discrete.slice(i, i + 2)
        .map((testCase) => runCase(backendTest, testCase)));
      const failure = results.find((result) => result.status === "rejected");
      if (failure && failure.status === "rejected") {throw failure.reason;}
    }

    // CPU and integrated GPUs share host memory and/or the desktop display. Keep every such case
    // behind the discrete batch and run them one at a time.
    for (const testCase of lane.serial) {
      await runCase(backendTest, testCase);
    }
  });
}

/** @param {string} algo @param {string} vendor */
function selectedBackend(algo, vendor) {
  const sm = vendor === "nvidia" ? compilerPolicy.nvidiaComputeCapability(process.env) : null;
  const selected = compilerPolicy.selection(algo, vendor, process.platform,
    vendor === "nvidia" ? sm ?? 0 : null);
  return selected ? selected.backend : "sycl";
}

/** @param {string} backend */
function backendGroup(backend) {
  return backend === "native" || backend === "sycl-native" ? backend : "sycl";
}

/** @returns {(vendor: string, options?: DiscoveryOptions) => Promise<GpuDiscoveryResult>} */
function cachedGpuDiscovery() {
  /** @type {Map<string, Promise<GpuDiscoveryResult>>} */
  const cache = new Map();
  return (vendor, options = {}) => {
    const key = gpuDeviceDiscoveryKey(vendor, options);
    const cached = cache.get(key);
    if (cached) {return cached;}
    const discovery = getGpuDevices(vendor, options);
    cache.set(key, discovery);
    return discovery;
  };
}

/** @returns {DiscreteDiscover} */
function discreteDiscovery() {
  const discover = cachedGpuDiscovery();
  return (algo, vendor) => discover(vendor, {algo, integrated: false});
}

/** @returns {SharedDiscover} */
function sharedDiscovery() {
  const discover = cachedGpuDiscovery();
  /** @type {Promise<CpuDiscoveryResult> | undefined} */
  let openclCpu;
  return {
    openclGpu(algo) {
      const env = {MOM_OPENCL_DEVICE_TYPE: "gpu", MOM_COMPILER_POLICY_STRICT: "1"};
      return discover("opencl", {algo, integrated: false, env});
    },
    openclCpu() {
      if (!openclCpu) {openclCpu = getFirstSyclCpuDevice(openclSyclEnv("cpu"));}
      return openclCpu;
    },
    intelIntegrated(algo, backend) {
      const env = {MOM_COMPILER_POLICY_STRICT: "1"};
      return discover("intel", {algo, integrated: true, backend, env});
    },
  };
}

/** @param {MatrixPlan} plan @param {string} algo @param {string[]} vendors @param {DiscreteDiscover} discover */
async function addDiscreteCases(plan, algo, vendors, discover) {
  const definitions = gpuVectorsFor(algo);
  const discoveries = await Promise.all(vendors.map(async (vendor) => ({
    vendor,
    result: await discover(algo, vendor),
  })));

  for (const {vendor, result} of discoveries) {
    const backend = selectedBackend(algo, vendor);
    if (result.skipped) {
      const required = process.env["MOM_REQUIRE_GPU_TESTS"] === "1";
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
        // Native source-JIT is an optimization. Its portable and tuned SYCL fallback paths remain
        // mandatory.
        for (const fallbackBackend of ["sycl", "sycl-native"]) {
          addCase(plan, backendGroup(fallbackBackend), "discrete",
            deviceCase(`${name} (${fallbackBackend} fallback)`, definitions,
              (definition) => cloneForDiscreteGpu(
                definition, vendor, device.dev, fallbackBackend)));
        }
        addCase(plan, "native", "discrete", deviceCase(name, definitions,
          (definition) => cloneForDiscreteGpu(definition, vendor, device.dev, "native")));
      } else {
        addCase(plan, backendGroup(backend), "discrete", deviceCase(name, definitions,
          (definition) => cloneForDiscreteGpu(definition, vendor, device.dev, backend)));
      }
    }
  }
}

/** @param {MatrixPlan} plan @param {string} algo @param {SharedDiscover} discover */
async function addGenericCases(plan, algo, discover) {
  const definition = fastVectorFor(algo);
  if (!definition) {throw new Error(`No fast vector for ${algo}`);}
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

  await addIntegratedCases(plan, algo, discover);
}

/** @param {MatrixPlan} plan @param {string} algo @param {SharedDiscover} discover */
async function addIntegratedCases(plan, algo, discover) {
  const definition = fastVectorFor(algo);
  if (!definition) {throw new Error(`No fast vector for ${algo}`);}
  const backend = selectedBackend(algo, "intel");
  const requireDevice = process.env["MOM_REQUIRE_INTEGRATED_GPU_TESTS"] === "1";
  const integratedResult = await discover.intelIntegrated(algo, backend);
  if (integratedResult.skipped) {
    addCase(plan, backendGroup(backend), "serial",
      unavailableCase("Intel integrated GPU (unavailable)", integratedResult.reason, requireDevice));
    return;
  }
  if (!integratedResult.devices.length) {
    addCase(plan, backendGroup(backend), "serial",
      unavailableCase("Intel integrated GPU (unavailable)",
        "Intel integrated GPU discovery returned no devices", requireDevice));
    return;
  }
  for (const device of integratedResult.devices) {
    const name = `intel ${device.dev}: ${device.description}`;
    if (!configuredDeviceSupports(integratedResult.params, algo, device.dev)) {
      addCase(plan, backendGroup(backend), "serial",
        unavailableCase(name, `${algo} is not available on ${device.description}`));
      continue;
    }
    addCase(plan, backendGroup(backend), "serial", deviceCase(name, [definition],
      (entry) => cloneForIntelIntegrated(entry, device.dev, backend)));
  }
}

/** @param {MatrixPlan} plan @param {string} algo @param {SharedDiscover} discover */
async function addPortableCpuCase(plan, algo, discover) {
  const definition = fastVectorFor(algo);
  if (!definition) {throw new Error(`No fast vector for ${algo}`);}
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
    const result = await runNode(["mom.js", "algorithms"], {
      cwd: directory,
      timeoutMs: TEST_TIMEOUT_MS,
      env: {
        ...openclSyclEnv("cpu"),
        MOM_GPU_INDEX: undefined,
        SYCL_DUMP_IMAGES: "1",
      },
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

/** @param {{discreteOnly?: boolean, integratedOnly?: boolean, portableCpuOnly?: boolean}} [options] */
function defineGpuTestMatrix({
  discreteOnly = false, integratedOnly = false, portableCpuOnly = false,
} = {}) {
  // The portable CPU lane does not discover hardware vendors. Ignore a caller's GPU selection
  // (for example r.sh's explicit MOM_GPU_BACKEND=opencl) instead of treating a backend as a vendor.
  const vendors = portableCpuOnly || integratedOnly ? [] : requestedVendors();
  const discoverDiscrete = discreteDiscovery();
  const discoverShared = sharedDiscovery();
  const title = portableCpuOnly
    ? "Portable SYCL CPU proof-of-work hash vectors"
    : integratedOnly
      ? "Integrated GPU proof-of-work hash vectors"
      : discreteOnly
        ? "GPU proof-of-work hash vectors (algorithm-centric discrete devices)"
        : "GPU proof-of-work hash vectors (algorithm-centric backend matrix)";

  describe(title, {timeout: MATRIX_TIMEOUT_MS, concurrency: 1}, () => {
    if (!discreteOnly && !integratedOnly) {
      it("generic OpenCL image guard", {timeout: TEST_TIMEOUT_MS}, openclImageGuard);
    }

    for (const algo of requestedAlgos()) {
      it(algo, {timeout: ALGO_TIMEOUT_MS, concurrency: false}, async (algoTest) => {
        /** @type {MatrixPlan} */
        const plan = new Map();
        if (portableCpuOnly) {
          await addPortableCpuCase(plan, algo, discoverShared);
        } else if (integratedOnly) {
          await addIntegratedCases(plan, algo, discoverShared);
        } else {
          await addDiscreteCases(plan, algo, vendors, discoverDiscrete);
          if (!discreteOnly) {await addGenericCases(plan, algo, discoverShared);}
        }
        for (const backend of [...plan.keys()].sort()) {
          const lane = plan.get(backend);
          if (!lane) {throw new Error(`Missing matrix lane ${backend}`);}
          await runBackend(algoTest, backend, lane);
        }
      });
    }
  });
}

module.exports = {addDiscreteCases, addIntegratedCases, defineGpuTestMatrix};
