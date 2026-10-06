"use strict";

const fs = require("node:fs");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {validatePearlHashShape} = require("./gpu-tuning");

/** @typedef {{m: number, n: number, k: number, rank: number}} PearlHashProfile */
/** @typedef {{linux: string, win32: string}} CompilerArtifact */
/** @typedef {{compiler?: string, backend?: string}} CompilerOverride */
/**
 * @typedef {{
 *   os: string,
 *   gpu: string,
 *   defaultCompiler: string,
 *   defaultBackend: string,
 *   overrides: Record<string, CompilerOverride>,
 *   pearlhashProfile: PearlHashProfile | null,
 * }} CompilerPolicy
 */
/**
 * @typedef {{
 *   artifacts: Record<string, CompilerArtifact>,
 *   policies: CompilerPolicy[],
 * }} CompilerConfig
 */
/**
 * @typedef {{
 *   key: string,
 *   addon: string,
 *   backend: string,
 *   pearlhashProfile: PearlHashProfile | null,
 * }} CompilerSelection
 */

const policyFile = path.join(__dirname, "GPU-CONFIG.md");
/** @type {CompilerConfig | undefined} */
let cached;
/** @type {(number | null)[] | undefined} */
let cachedNvidiaSms;
const legacyNvidiaPortableAlgos = new Set([
  "cn/gpu", "pearlhash", "octopus", "nexapow",
]);
const genericBackends = new Set(["sycl", "sycl-opencl", "sycl-l0"]);
const adaptiveCppBackends = new Set(["auto", "sycl", "sycl-native", "native"]);
const dpcppCompilerKeys = new Set(["oneapi", "dpcpp", "dpcpp-opencl"]);
/** @type {Record<string, {name: string, mask: string}>} */
const adaptiveCppTargets = {
  "acpp-cuda": {name: "CUDA", mask: "cuda"},
  "acpp-hip": {name: "HIP", mask: "hip"},
};
/** @type {Record<string, string>} */
const syclNativeVariables = {
  octopus: "MOM_OCTOPUS_SYCL_NATIVE",
  xelishashv3: "MOM_XELISHASHV3_SYCL_NATIVE",
  walahash: "MOM_WALAHASH_SYCL_NATIVE",
};
const policyKeyPattern = /^[a-z0-9][a-z0-9-]*$/;

/** @param {string} line */
function cells(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((v) =>
    v.trim().replace(/`/g, "")
  );
}

/** @param {string[]} row */
function isSeparator(row) {
  return row.every((v) => /^:?-{3,}:?$/.test(v));
}

/** @param {string} markdown @returns {string[][][]} */
function tables(markdown) {
  /** @type {string[][][]} */
  const result = [];
  /** @type {string[][]} */
  let table = [];
  for (const line of markdown.split(/\r?\n/)) {
    if (/^\s*\|.*\|\s*$/.test(line)) {
      table.push(cells(line));
    } else if (table.length) {
      result.push(table);
      table = [];
    }
  }
  if (table.length) {
    result.push(table);
  }
  return result;
}

/** @param {string} value @returns {PearlHashProfile | null} */
function parsePearlHashProfile(value) {
  if (!value || value === "—" || value === "-") {
    return null;
  }
  const match = value.match(/^(\d+)x(\d+)x(\d+)\/(\d+)$/);
  if (!match) {
    throw new Error(`Invalid PearlHash profile: ${value}`);
  }
  try {
    return validatePearlHashShape(match[1], match[2], match[3], match[4], "PearlHash profile");
  } catch (error) {
    throw new Error(`Invalid PearlHash profile: ${value}`, {cause: error});
  }
}

/** @param {string | undefined} value @returns {string | undefined} */
function optionalPolicyValue(value) {
  return !value || value === "—" || value === "-" ? undefined : value;
}

/** @param {string} [markdown] @returns {CompilerConfig} */
function parse(markdown = fs.readFileSync(policyFile, "utf8")) {
  const parsed = tables(markdown).map((table) => ({
    header: table[0] || [],
    rows: table.slice(2).filter((row) => !isSeparator(row)),
  }));
  const artifactTable = parsed.find(({header}) => header[0] === "Key");
  const defaultTable = parsed.find(({header}) => header[0] === "OS");
  const overrideTable = parsed.find(({header}) => header[0] === "Algorithm");
  if (!artifactTable || !defaultTable || !overrideTable) {
    throw new Error("GPU-CONFIG.md tables are missing");
  }
  /** @type {Record<string, CompilerArtifact>} */
  const artifacts = {};
  for (const row of artifactTable.rows) {
    const [key, linux, win32] = row;
    if (!key || !linux || !win32) {throw new Error("GPU artifact row is incomplete");}
    if (!policyKeyPattern.test(key) || Object.hasOwn(artifacts, key)) {
      throw new Error(`Invalid or duplicate GPU artifact key: ${key}`);
    }
    artifacts[key] = {linux, win32};
  }
  const policyKeys = new Set();
  const policies = defaultTable.rows.map((row) => {
    const [os, gpu, defaultCompiler, defaultBackend, profile = ""] = row;
    if (!os || !gpu || !defaultCompiler || !defaultBackend) {
      throw new Error("GPU policy row is incomplete");
    }
    const normalizedOs = os.toLowerCase();
    const normalizedGpu = gpu.toLowerCase();
    const policyKey = `${normalizedOs}/${normalizedGpu}`;
    if (policyKeys.has(policyKey)) {throw new Error(`Duplicate GPU policy row: ${policyKey}`);}
    policyKeys.add(policyKey);
    if (!Object.hasOwn(artifacts, defaultCompiler)) {
      throw new Error(`Unknown GPU compiler key: ${defaultCompiler}`);
    }
    validateBackend(defaultBackend);
    return {
      os: normalizedOs, gpu: normalizedGpu, defaultCompiler, defaultBackend,
      overrides: {}, pearlhashProfile: parsePearlHashProfile(profile),
    };
  });
  const policiesByKey = new Map(policies.map((policy) =>
    [`${policy.os}/${policy.gpu}`, policy]
  ));
  for (const row of overrideTable.rows) {
    const [algo, os, gpu, compiler = "", backend = ""] = row;
    if (!algo || !os || !gpu) {throw new Error("GPU override row is incomplete");}
    const policyKey = `${os.toLowerCase()}/${gpu.toLowerCase()}`;
    const policy = policiesByKey.get(policyKey);
    if (!policy) {throw new Error(`Unknown GPU policy row: ${policyKey}`);}
    if (Object.hasOwn(policy.overrides, algo)) {
      throw new Error(`Duplicate GPU override: ${policyKey}/${algo}`);
    }
    const compilerValue = optionalPolicyValue(compiler);
    const backendValue = optionalPolicyValue(backend);
    if (!compilerValue && !backendValue) {
      throw new Error(`GPU override row is empty: ${policyKey}/${algo}`);
    }
    if (compilerValue && !Object.hasOwn(artifacts, compilerValue)) {
      throw new Error(`Unknown GPU compiler key: ${compilerValue}`);
    }
    if (backendValue) {validateBackend(backendValue);}
    /** @type {CompilerOverride} */
    const override = {};
    if (compilerValue) {override.compiler = compilerValue;}
    if (backendValue) {override.backend = backendValue;}
    Object.defineProperty(policy.overrides, algo, {
      configurable: true, enumerable: true, value: override, writable: true,
    });
  }
  return {artifacts, policies};
}

/** @param {string} platform */
function osName(platform) {
  return platform === "win32" ? "windows" : "linux";
}

// The algorithms output is user-facing and annotates each GPU job with the selected backend. Callers that
// feed a reported job back to mom must remove that annotation first; otherwise mom would append it
// again and eventually hand an invalid device string such as gpu1*[intensity=8]:auto[sycl]:auto[sycl] to a
// worker.
/** @param {unknown} value */
function parseReportedAlgoParam(value) {
  if (typeof value !== "string") {
    throw new Error("Reported algorithm parameters must be a string");
  }
  const text = value;
  const colon = text.lastIndexOf(":");
  if (colon < 0) {
    return {dev: text};
  }
  const shownBackend = text.slice(colon + 1);
  const auto = shownBackend.match(/^auto\[([^\]]+)\]$/);
  if (auto && auto[1] && validBackends.has(auto[1])) {
    return {dev: text.slice(0, colon), backend: "auto"};
  }
  if (validBackends.has(shownBackend)) {
    return {dev: text.slice(0, colon), backend: shownBackend};
  }
  return {dev: text};
}

/**
 * @param {unknown} algo
 * @param {unknown} gpu
 * @param {unknown} [platform]
 * @param {number | null} [nvidiaSm]
 * @returns {CompilerSelection | null}
 */
function selection(algo, gpu, platform = process.platform, nvidiaSm = null) {
  if (typeof algo !== "string") {throw new Error("GPU algorithm must be a string");}
  if (typeof gpu !== "string") {throw new Error("GPU name must be a string");}
  if (typeof platform !== "string") {throw new Error("Platform must be a string");}
  if (nvidiaSm !== null && (!Number.isSafeInteger(nvidiaSm) || nvidiaSm < 0)) {
    throw new Error("Invalid NVIDIA compute capability: expected a non-negative safe integer or null");
  }
  const config = cached || (cached = parse());
  const row = config.policies.find((p) => p.os === osName(platform) && p.gpu === gpu.toLowerCase());
  if (!row) {
    return null;
  }
  const override = Object.hasOwn(row.overrides, algo) ? row.overrides[algo] : undefined;
  let key = override?.compiler ?? row.defaultCompiler;
  const legacyNvidia = row.gpu === "nvidia" && nvidiaSm !== null && nvidiaSm < 80;
  if (legacyNvidia) {
    key = "acpp-cuda";
  }
  const artifact = Object.hasOwn(config.artifacts, key) ? config.artifacts[key] : undefined;
  if (!artifact) {
    throw new Error(`Unknown GPU compiler key: ${key}`);
  }
  let backend = override?.backend ?? row.defaultBackend;
  if (legacyNvidia && (legacyNvidiaPortableAlgos.has(algo) ||
      (algo === "walahash" && nvidiaSm < 61))) {
    backend = "sycl";
  }
  // Pearl's packed integer dot path needs DP4A, not the SM80 tensor/copy instructions.
  if (legacyNvidia && algo === "pearlhash" && nvidiaSm >= 61) {
    backend = "sycl-native";
  }
  return {key, addon: artifact[platform === "win32" ? "win32" : "linux"],
    backend, pearlhashProfile: row.pearlhashProfile};
}

/** @param {unknown} value @returns {number | null} */
function parseNvidiaSm(value) {
  if (typeof value !== "number" && typeof value !== "string") {
    return null;
  }
  const text = typeof value === "string" ? value.trim() : value.toString();
  const match = text.match(/^([1-9]\d*)(?:\.(\d))?$/);
  if (!match) {
    return null;
  }
  const major = Number(match[1]);
  if (!Number.isSafeInteger(major)) {
    return null;
  }
  const sm = match[2] === undefined && major >= 10
    ? major : major * 10 + Number(match[2] || 0);
  return Number.isSafeInteger(sm) && sm > 0 ? sm : null;
}

/** @param {NodeJS.ProcessEnv} [env] @returns {number | null} */
function nvidiaComputeCapability(env = process.env) {
  gpuIndex(env);
  if (env["MOM_NVIDIA_COMPUTE_CAPABILITY"]) {
    const sm = parseNvidiaSm(env["MOM_NVIDIA_COMPUTE_CAPABILITY"]);
    if (sm === null) {
      throw new Error("Invalid MOM_NVIDIA_COMPUTE_CAPABILITY");
    }
    return sm;
  }
  if (cachedNvidiaSms === undefined) {
    const result = spawnSync("nvidia-smi",
      ["--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
      {encoding: "utf8", timeout: 2000, windowsHide: true});
    if (!result.error && result.status === 0 && typeof result.stdout === "string") {
      const output = result.stdout.replace(/\r?\n$/, "");
      cachedNvidiaSms = output === "" ? [] : output.split(/\r?\n/).map(parseNvidiaSm);
    } else {
      cachedNvidiaSms = [];
    }
  }
  // nvidia-smi rows are physical devices, not CUDA-visible ordinals. Their conservative minimum
  // remains safe under inherited visibility masks and CUDA's independent enumeration order.
  return cachedNvidiaSms.length && cachedNvidiaSms.every((sm) => sm !== null)
    ? Math.min(...cachedNvidiaSms) : null;
}

const validBackends = new Set([
  "auto", "sycl", "sycl-opencl", "sycl-l0", "sycl-native", "native",
]);

/** @param {unknown} value @returns {string} */
function validateBackend(value) {
  if (value === undefined || value === null || value === "") {return "auto";}
  if (typeof value !== "string") {
    throw new Error(`Invalid GPU backend type: ${typeof value}`);
  }
  const backend = value.toLowerCase();
  if (!validBackends.has(backend)) {
    throw new Error(`Invalid GPU backend: ${value}; expected ${[...validBackends].join(", ")}`);
  }
  return backend;
}

/** @param {NodeJS.ProcessEnv} [env] */
function gpuFromEnv(env = process.env) {
  const value = (env["MOM_GPU_BACKEND"] || "").toLowerCase();
  return value === "intel" || value === "nvidia" || value === "amd" || value === "opencl"
    ? value : "";
}

/** @param {NodeJS.ProcessEnv} env @returns {string | null} */
function gpuIndex(env) {
  if (typeof env["MOM_GPU_INDEX"] === "undefined" || env["MOM_GPU_INDEX"] === "") {
    return null;
  }
  if (typeof env["MOM_GPU_INDEX"] !== "string") {
    throw new Error("Invalid MOM_GPU_INDEX: expected a string");
  }
  const index = env["MOM_GPU_INDEX"];
  const number = Number(index);
  if (!/^\d+$/.test(index) || !Number.isSafeInteger(number) || number > 1023) {
    throw new Error(`Invalid MOM_GPU_INDEX: ${index}`);
  }
  return index;
}

/** @param {NodeJS.ProcessEnv} env @returns {NodeJS.ProcessEnv} */
function vendorDeviceEnv(env) {
  const gpu = gpuFromEnv(env);
  const index = gpuIndex(env);
  if (index === null || (gpu !== "nvidia" && gpu !== "amd")) {return {};}
  const visible = gpu === "nvidia" ? "CUDA_VISIBLE_DEVICES" : "HIP_VISIBLE_DEVICES";
  // HIP also accepts CUDA_VISIBLE_DEVICES when its own mask is absent. Never replace either
  // inherited restriction, including an empty mask or a list truncated by an invalid token.
  const masks = gpu === "amd" ? [visible, "CUDA_VISIBLE_DEVICES"] : [visible];
  const inherited = masks.filter((name) => env[name] !== undefined);
  for (const name of inherited) {
    const mask = env[name];
    if (Number(index) !== 0 || typeof mask !== "string" || !mask ||
        mask === "-1" || mask.includes(",")) {
      throw new Error(`MOM_GPU_INDEX cannot safely select within inherited ${name}; ` +
        "use a single-device visibility mask with MOM_GPU_INDEX=0, or omit MOM_GPU_INDEX " +
        "and select a reported --job.dev");
    }
  }
  // Once isolated, zero is the runtime's logical ordinal. Repeated worker-policy calls must
  // preserve the one-device mask rather than reinterpret the original index against it.
  return inherited.length ? {MOM_GPU_INDEX: "0"} : {
    [visible]: String(Number(index)), MOM_GPU_INDEX: "0",
  };
}

/** @param {NodeJS.ProcessEnv} env */
function openclDeviceType(env) {
  const value = env["MOM_OPENCL_DEVICE_TYPE"];
  if (value !== undefined && value !== "" && typeof value !== "string") {
    throw new Error("Invalid MOM_OPENCL_DEVICE_TYPE: expected a string; expected gpu or cpu");
  }
  const type = (value || "gpu").toLowerCase();
  if (type !== "gpu" && type !== "cpu") {
    throw new Error(`Invalid MOM_OPENCL_DEVICE_TYPE: ${type}; expected gpu or cpu`);
  }
  return type;
}

/** @param {string} name @param {NodeJS.ProcessEnv} env @returns {string | null} */
function executableFromPath(name, env) {
  for (const directory of (env["PATH"] || "").split(path.delimiter)) {
    if (!directory) {continue;}
    const candidate = path.join(directory, name);
    try {
      fs.accessSync(candidate, fs.constants.X_OK);
      return fs.realpathSync(candidate);
    } catch {
      // Match normal PATH lookup: an absent or unusable candidate does not hide later entries.
    }
  }
  return null;
}

/** @param {string} root @param {string[]} libraries @returns {string | null} */
function runtimeLibraryDirectory(root, libraries) {
  for (const name of ["lib", "lib64"]) {
    const directory = path.join(root, name);
    let entries;
    try {
      entries = fs.readdirSync(directory);
    } catch {
      continue;
    }
    if (libraries.every((library) =>
      entries.some((entry) =>
        (entry === library || entry.startsWith(`${library}.`)) &&
        fs.existsSync(path.join(directory, entry))))) {
      return directory;
    }
  }
  return null;
}

/**
 * Keep the loader on the same vendor toolkit selected by its standard environment/tool. Otherwise
 * coinstalled ROCm versions can make hipconfig and libamdhip disagree and crash before a kernel is
 * launched. Packaged compiler libraries still come first; this only disambiguates host runtimes.
 * @param {string} gpu
 * @param {NodeJS.ProcessEnv} env
 * @returns {string | null}
 */
function linuxVendorRuntimeDirectory(gpu, env) {
  if (gpu === "amd") {
    const hipconfig = executableFromPath("hipconfig", env);
    const roots = [env["ROCM_PATH"], env["HIP_PATH"],
      hipconfig ? path.dirname(path.dirname(hipconfig)) : null];
    for (const root of roots) {
      if (root) {
        const directory = runtimeLibraryDirectory(root, [
          "libamdhip64.so.7", "libamd_comgr.so.3", "libhsa-runtime64.so.1",
          "libhiprtc.so.7", "libhiprtc-builtins.so.7",
        ]);
        if (directory) {return directory;}
      }
    }
  }
  if (gpu === "nvidia") {
    for (const root of [env["CUDA_PATH"], env["CUDA_HOME"]]) {
      if (root) {
        const directory = runtimeLibraryDirectory(root, ["libnvrtc.so"]);
        if (directory) {return directory;}
      }
    }
  }
  return null;
}

/**
 * @param {string} algo
 * @param {NodeJS.ProcessEnv} [env]
 * @param {NodeJS.Platform} [platform]
 * @param {string} [requestedBackend]
 * @returns {NodeJS.ProcessEnv}
 */
function workerEnv(algo, env = process.env, platform = process.platform, requestedBackend = "auto") {
  // An explicit addon path is an intentional compiler override (used by focused validation and
  // advanced deployments). Do not silently replace it with the table default for this algorithm.
  if (env["MOM_NATIVE_PATH"] &&
      env["MOM_NATIVE_PATH"] !== env["MOM_NATIVE_PATH_LAUNCHER_DEFAULT"]) {
    return vendorDeviceEnv(env);
  }
  const gpu = gpuFromEnv(env);
  if (!gpu) {
    return {};
  }
  const deviceEnv = vendorDeviceEnv(env);
  const sm = gpu === "nvidia" ? nvidiaComputeCapability(env) : null;
  let selected = selection(algo, gpu, platform, gpu === "nvidia" ? sm ?? 0 : null);
  if (!selected || !selected.addon || selected.addon === "—") {
    return deviceEnv;
  }
  const explicitBackend = validateBackend(requestedBackend);
  const backend = explicitBackend === "auto" ? selected.backend : explicitBackend;
  const genericBackend = genericBackends.has(backend);
  // Intel's tuned fat image contains ESIMD/DPAS code that XeLP cannot even link when an unrelated
  // generic kernel is requested. An explicit different generic transport therefore uses the
  // standards-only artifact. Keep auto on the measured policy compiler, including cn/gpu's oneAPI
  // OpenCL path. NexaPoW's explicit NVIDIA SYCL path uses the generic AdaptiveCpp image because the
  // DPC++ image intentionally contains its NVPTX-tuned field representation.
  const genericFallback = explicitBackend !== "auto" && genericBackend &&
    backend !== selected.backend && (gpu === "intel" || gpu === "opencl");
  const nvidiaNexaPortable = algo === "nexapow" && gpu === "nvidia" &&
    explicitBackend !== "auto" && genericBackend && backend !== selected.backend;
  if ((genericFallback || nvidiaNexaPortable) &&
      selected.key !== (nvidiaNexaPortable ? "acpp-cuda" : "dpcpp-opencl")) {
    const config = cached || (cached = parse());
    const key = nvidiaNexaPortable ? "acpp-cuda" : "dpcpp-opencl";
    const artifact = Object.hasOwn(config.artifacts, key) ? config.artifacts[key] : undefined;
    if (!artifact) {throw new Error(`Unknown GPU compiler key: ${key}`);}
    selected = {
      ...selected,
      key,
      addon: artifact[platform === "win32" ? "win32" : "linux"],
    };
  }
  const platformBuild = platform === "win32" ? "win" : "lin";
  const roots = [path.join(__dirname, "libs"),
    path.join(__dirname, "build", platformBuild, "Release")];
  if (env["MOM_NATIVE_DIR"]) {roots.unshift(env["MOM_NATIVE_DIR"]);}
  const addon = roots.map((root) => path.resolve(root, selected.addon)).find(fs.existsSync);
  if (!addon) {
    const missing = `Missing ${selected.addon} for ${gpu}/${algo}`;
    if (env["MOM_COMPILER_POLICY_STRICT"] === "1") {
      throw new Error(missing);
    }
    if (gpu === "nvidia" && (sm ?? 0) < 80) {
      const capability = sm === null ? "unknown NVIDIA compute capability" :
        `NVIDIA compute capability SM${sm}`;
      throw new Error(`${missing}: ${capability} cannot use the SM80-only default worker`);
    }
    return deviceEnv;
  }
  const libDir = path.dirname(addon);
  /** @type {NodeJS.ProcessEnv} */
  const result = {
    MOM_NATIVE_PATH: addon,
    MOM_RUNTIME_DIR: libDir,
    MOM_SYCL_COMPILER: selected.key,
    ...deviceEnv,
  };
  const syclNativeVariable = syclNativeVariables[algo];
  if (syclNativeVariable) {
    result[syclNativeVariable] = backend === "sycl-native" ? "1" : "0";
  }
  const localBuild = path.join(__dirname, "build", platformBuild);
  if (fs.existsSync(localBuild)) {
    result["MOM_JIT_CACHE_DIR"] = path.join(localBuild, ".jit-cache");
  }
  if (dpcppCompilerKeys.has(selected.key)) {
    // Runtime caching defaults off; avoid recompiling large GPU kernels on every miner restart.
    // Windows Intel workers isolate algorithms because sharing one persistent cache can make a
    // WalaHash compile return incorrect XelisHashV3 code. Other workers preserve their cache path.
    result["SYCL_CACHE_PERSISTENT"] = env["SYCL_CACHE_PERSISTENT"] ?? "1";
    if (platform === "win32" && gpu === "intel") {
      const cacheRoot = env["SYCL_CACHE_DIR"] || (fs.existsSync(localBuild)
        ? path.join(localBuild, ".sycl-cache")
        : env["LOCALAPPDATA"] && path.join(env["LOCALAPPDATA"], "mom-sycl-cache"));
      if (cacheRoot) {
        const algoKey = algo.replace(/[^A-Za-z0-9_.-]/g, "_");
        result["SYCL_CACHE_DIR"] = path.join(cacheRoot, selected.key, algoKey);
      }
    } else if (platform === "win32" && fs.existsSync(localBuild) &&
               env["SYCL_CACHE_DIR"] === undefined) {
      result["SYCL_CACHE_DIR"] = path.join(localBuild, ".sycl-cache");
    }
  }
  const index = gpuIndex({...env, ...result});
  // The addon chooses the compiler runtime, while these selectors choose that runtime's matching
  // device backend. Without the AdaptiveCpp mask an acpp-cuda worker can see the host Intel OpenCL
  // ICD first and try to translate its CUDA-oriented SSCP image through llvm-spirv.
  const acppTarget = adaptiveCppTargets[selected.key];
  if (acppTarget) {
    if (!adaptiveCppBackends.has(backend)) {
      throw new Error(`${backend} is incompatible with the AdaptiveCpp ${acppTarget.name} worker`);
    }
    result["ACPP_VISIBILITY_MASK"] = acppTarget.mask;
  }
  if (selected.key === "dpcpp" && gpu === "nvidia") {
    if (backend === "sycl-opencl" || backend === "sycl-l0") {
      throw new Error(`${backend} is incompatible with the DPC++ CUDA worker`);
    }
    result["ONEAPI_DEVICE_SELECTOR"] = index === null ? "cuda:gpu" : `cuda:${index}`;
  }
  if ((selected.key === "dpcpp" || selected.key === "oneapi") && gpu === "intel") {
    result["ONEAPI_DEVICE_SELECTOR"] = backend === "sycl-opencl" ? "opencl:gpu" : "level_zero:gpu";
  }
  if (gpu === "intel" && dpcppCompilerKeys.has(selected.key)) {
    // Level Zero reports a conservative per-allocation ceiling on some full-ReBAR Arc systems.
    // The runtime's relaxed-allocation descriptor removes that artificial limit while allocation
    // failure still safely rejects workloads that do not fit physical VRAM.
    result["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] =
      env["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] || "1";
  }
  if (selected.key === "dpcpp-opencl") {
    // GPU is the mining default. CPU is an explicit correctness/portability mode used to prove
    // that the same standards-only SPIR-V artifact also runs through an OpenCL CPU implementation.
    // Intel integrated GPUs use Level Zero because it is their native low-overhead interface;
    // generic/unknown vendors use the portable OpenCL contract.
    const genericDefaultL0 = backend === "sycl" && gpu === "intel";
    result["ONEAPI_DEVICE_SELECTOR"] = backend === "sycl-l0" || genericDefaultL0
      ? "level_zero:gpu"
      : `opencl:${openclDeviceType(env)}`;
  }
  // Each compiler ships beside its own SYCL runtime. Workers are separate processes, so putting
  // only the selected directory first avoids same-SONAME collisions (notably DPC++ libsycl) while
  // still allowing the policy to switch compilers when an algorithm changes.
  // Windows portable SYCL/UR uses oneAPI, including its shared JIT library; Linux keeps DPC++.
  const sharedRuntime = selected.key === "dpcpp-opencl"
    ? path.join(path.dirname(libDir), platform === "win32" ? "oneapi" : "dpcpp")
    : null;
  if (platform !== "win32") {
    const vendorRuntime = linuxVendorRuntimeDirectory(gpu, env);
    result["LD_LIBRARY_PATH"] = [libDir, path.join(libDir, "hipSYCL"), sharedRuntime,
      vendorRuntime, env["LD_LIBRARY_PATH"]]
      .filter(Boolean).join(path.delimiter);
  } else {
    // Windows loads compiler runtimes and AdaptiveCpp backend-plugin dependencies through PATH.
    // Keep the selected worker ahead of the shared package root so same-named oneAPI/DPC++/ACPP
    // DLLs cannot cross-load when policy switches compiler for the next algorithm.
    const pathKey = Object.hasOwn(env, "Path") ? "Path" :
      Object.keys(env).find((key) => key.toLowerCase() === "path") || "PATH";
    result[pathKey] = [libDir, path.join(libDir, "hipSYCL"), sharedRuntime,
      env[pathKey]]
      .filter(Boolean).join(path.delimiter);
  }
  return result;
}

module.exports = {
  parse, parseReportedAlgoParam, selection, workerEnv, gpuFromEnv, nvidiaComputeCapability,
  validateBackend, validBackends, vendorDeviceEnv,
};
