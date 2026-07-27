"use strict";

const fs = require("fs");
const path = require("path");

const policyFile = path.join(__dirname, "GPU-COMPILERS.md");
let cached;

function cells(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((v) =>
    v.trim().replace(/`/g, "")
  );
}

function isSeparator(row) {
  return row.every((v) => /^:?-{3,}:?$/.test(v));
}

function tables(markdown) {
  const result = [];
  let table = [];
  for (const line of markdown.split(/\r?\n/)) {
    if (/^\s*\|.*\|\s*$/.test(line)) {table.push(cells(line));}
    else if (table.length) { result.push(table); table = []; }
  }
  if (table.length) {result.push(table);}
  return result;
}

function parseOverrides(value, kind) {
  if (!value || value === "—" || value === "-") {return {};}
  return Object.fromEntries(value.split(",").map((entry) => {
    const [algo, compiler] = entry.trim().split("=");
    if (!algo || !compiler) {throw new Error(`Invalid GPU ${kind} override: ${entry}`);}
    return [algo, compiler];
  }));
}

function parsePearlHashProfile(value) {
  if (!value || value === "—" || value === "-") {return null;}
  const match = value.match(/^(\d+)x(\d+)x(\d+)\/(\d+)$/);
  if (!match) {throw new Error(`Invalid PearlHash profile: ${value}`);}
  return {m: Number(match[1]), n: Number(match[2]), k: Number(match[3]), rank: Number(match[4])};
}

function parseList(value) {
  if (!value || value === "—" || value === "-") {return new Set();}
  return new Set(value.split(",").map((entry) => entry.trim()).filter(Boolean));
}

function parse(markdown = fs.readFileSync(policyFile, "utf8")) {
  const parsed = tables(markdown).map((table) =>
    [table[0], table.slice(2).filter((row) => !isSeparator(row))]
  );
  const artifactTable = parsed.find(([header]) => header[0] === "Key");
  const policyTable = parsed.find(([header]) => header[0] === "OS");
  if (!artifactTable || !policyTable) {throw new Error("GPU-COMPILERS.md tables are missing");}
  const artifacts = Object.fromEntries(artifactTable[1].map((row) => [row[0], {
    compiler: row[1], linux: row[2], win32: row[3], allocation: row[4] || ""
  }]));
  const policies = policyTable[1].map((row) => ({
    os: row[0].toLowerCase(), gpu: row[1].toLowerCase(), defaultCompiler: row[2],
    overrides: parseOverrides(row[3], "compiler"), tuned: parseList(row[4]),
    backends: parseOverrides(row[5], "backend"), pearlhashProfile: parsePearlHashProfile(row[6])
  }));
  return {artifacts, policies};
}

function osName(platform) { return platform === "win32" ? "windows" : "linux"; }

// algo_params is user-facing and annotates each GPU job with the selected backend. Callers that
// feed a reported job back to mom must remove that annotation first; otherwise mom would append it
// again and eventually hand an invalid device string such as gpu1*8:auto[sycl]:auto[sycl] to a
// worker.
function parseReportedAlgoParam(value) {
  const text = String(value);
  const colon = text.lastIndexOf(":");
  if (colon < 0) {return {dev: text};}
  const shownBackend = text.slice(colon + 1);
  const auto = shownBackend.match(/^auto\[([^\]]+)\]$/);
  if (auto && validBackends.has(auto[1])) {
    return {dev: text.slice(0, colon), backend: "auto"};
  }
  if (validBackends.has(shownBackend)) {
    return {dev: text.slice(0, colon), backend: shownBackend};
  }
  return {dev: text};
}

function selection(algo, gpu, platform = process.platform) {
  const config = cached || (cached = parse());
  const row = config.policies.find((p) => p.os === osName(platform) && p.gpu === gpu.toLowerCase());
  if (!row) {return null;}
  const key = row.overrides[algo] || row.defaultCompiler;
  const artifact = config.artifacts[key];
  if (!artifact) {throw new Error(`Unknown GPU compiler key: ${key}`);}
  const backend = row.backends[algo] || row.backends["*"] ||
    (row.tuned.has(algo) ? "sycl-native" : "sycl");
  return {key, compiler: artifact.compiler, addon: artifact[platform === "win32" ? "win32" : "linux"],
    allocation: artifact.allocation, backend, pearlhashProfile: row.pearlhashProfile};
}

const validBackends = new Set([
  "auto", "sycl", "sycl-opencl", "sycl-l0", "sycl-native", "native",
]);

function validateBackend(value) {
  const backend = String(value || "auto").toLowerCase();
  if (!validBackends.has(backend)) {
    throw new Error(`Invalid GPU backend: ${value}; expected ${[...validBackends].join(", ")}`);
  }
  return backend;
}

function gpuFromEnv(env = process.env) {
  const value = (env.MOM_GPU_BACKEND || "").toLowerCase();
  if (value.startsWith("nvidia")) {return "nvidia";}
  if (value.startsWith("amd")) {return "amd";}
  if (value.startsWith("intel")) {return "intel";}
  if (value.startsWith("opencl")) {return "opencl";}
  return "";
}

function gpuIndex(env) {
  if (typeof env.MOM_GPU_INDEX === "undefined" || env.MOM_GPU_INDEX === "") {return null;}
  const index = String(env.MOM_GPU_INDEX);
  if (!/^\d+$/.test(index)) {throw new Error(`Invalid MOM_GPU_INDEX: ${index}`);}
  return index;
}

function openclDeviceType(env) {
  const type = String(env.MOM_OPENCL_DEVICE_TYPE || "gpu").toLowerCase();
  if (type !== "gpu" && type !== "cpu") {
    throw new Error(`Invalid MOM_OPENCL_DEVICE_TYPE: ${type}; expected gpu or cpu`);
  }
  return type;
}

function workerEnv(algo, env = process.env, platform = process.platform, requestedBackend = "auto") {
  // An explicit addon path is an intentional compiler override (used by focused validation and
  // advanced deployments). Do not silently replace it with the table default for this algorithm.
  if (env.MOM_NATIVE_PATH && env.MOM_NATIVE_PATH !== env.MOM_NATIVE_PATH_LAUNCHER_DEFAULT) {return {};}
  const gpu = gpuFromEnv(env);
  if (!gpu) {return {};}
  let selected = selection(algo, gpu, platform);
  if (!selected || !selected.addon || selected.addon === "—") {return {};}
  const explicitBackend = validateBackend(requestedBackend);
  const backend = explicitBackend === "auto" ? selected.backend : explicitBackend;
  const genericBackend = backend === "sycl" || backend === "sycl-opencl" ||
    backend === "sycl-l0";
  // Intel's tuned fat image contains ESIMD/DPAS code that XeLP cannot even link when an unrelated
  // generic kernel is requested. An explicit different generic transport therefore uses the
  // standards-only artifact. Keep auto on the measured policy compiler, including cn/gpu's oneAPI
  // OpenCL path, and keep AMD/NVIDIA on their native transport when no vendor OpenCL ICD exists.
  const genericFallback = explicitBackend !== "auto" && genericBackend &&
    backend !== selected.backend && (gpu === "intel" || gpu === "opencl");
  if (genericFallback && selected.key !== "dpcpp-opencl") {
    const config = cached || (cached = parse());
    const artifact = config.artifacts["dpcpp-opencl"];
    selected = {
      ...selected,
      key: "dpcpp-opencl",
      compiler: artifact.compiler,
      addon: artifact[platform === "win32" ? "win32" : "linux"],
      allocation: artifact.allocation,
    };
  }
  const platformBuild = platform === "win32" ? "win" : "lin";
  const roots = [env.MOM_NATIVE_DIR, path.join(__dirname, "libs"),
    path.join(__dirname, "build", platformBuild, "Release")]
    .filter(Boolean);
  const addon = roots.map((root) => path.resolve(root, selected.addon)).find(fs.existsSync);
  if (!addon) {
    if (env.MOM_COMPILER_POLICY_STRICT === "1") {throw new Error(`Missing ${selected.addon} for ${gpu}/${algo}`);}
    return {};
  }
  const libDir = path.dirname(addon);
  const result = {
    MOM_NATIVE_PATH: addon,
    MOM_RUNTIME_DIR: libDir,
    MOM_SYCL_COMPILER: selected.key,
  };
  const localBuild = path.join(__dirname, "build", platform === "win32" ? "win" : "lin");
  if (fs.existsSync(localBuild)) {
    result.MOM_JIT_CACHE_DIR = path.join(localBuild, ".jit-cache");
  }
  if (platform === "win32" &&
      (selected.key === "oneapi" || selected.key === "dpcpp" || selected.key === "dpcpp-opencl")) {
    // win/run.sh round-trips build/win but deliberately knows nothing about mom. Keep DPC++'s
    // device cache in that application-owned tree so a disposable VM does not JIT the same large
    // SPIR-V image again on every run. Packaged releases have no build/win and retain the runtime's
    // normal per-user cache location.
    if (fs.existsSync(localBuild)) {
      result.SYCL_CACHE_PERSISTENT = "1";
      result.SYCL_CACHE_DIR = path.join(localBuild, ".sycl-cache");
    }
  }
  const index = gpuIndex(env);
  // The addon chooses the compiler runtime, while these selectors choose that runtime's matching
  // device backend. Without the AdaptiveCpp mask an acpp-cuda worker can see the host Intel OpenCL
  // ICD first and try to translate its CUDA-oriented SSCP image through llvm-spirv.
  if (selected.key === "acpp-cuda") {
    if (backend !== "auto" && backend !== "sycl" &&
        backend !== "sycl-native" && backend !== "native") {
      throw new Error(`${backend} is incompatible with the AdaptiveCpp CUDA worker`);
    }
    result.ACPP_VISIBILITY_MASK = "cuda";
    if (index !== null) {result.CUDA_VISIBLE_DEVICES = index;}
  }
  if (selected.key === "acpp-hip") {
    if (backend !== "auto" && backend !== "sycl" &&
        backend !== "sycl-native" && backend !== "native") {
      throw new Error(`${backend} is incompatible with the AdaptiveCpp HIP worker`);
    }
    result.ACPP_VISIBILITY_MASK = "hip";
    if (index !== null) {result.HIP_VISIBLE_DEVICES = index;}
  }
  if (selected.key === "dpcpp" && gpu === "nvidia") {
    if (backend === "sycl-opencl" || backend === "sycl-l0") {
      throw new Error(`${backend} is incompatible with the DPC++ CUDA worker`);
    }
    result.ONEAPI_DEVICE_SELECTOR = index === null ? "cuda:gpu" : `cuda:${index}`;
  }
  if (selected.key === "dpcpp" && gpu === "intel") {
    result.ONEAPI_DEVICE_SELECTOR = backend === "sycl-opencl" ? "opencl:gpu" : "level_zero:gpu";
  }
  if (selected.key === "oneapi" && gpu === "intel") {
    result.ONEAPI_DEVICE_SELECTOR = backend === "sycl-opencl" ? "opencl:gpu" : "level_zero:gpu";
  }
  if (gpu === "intel" &&
      (selected.key === "oneapi" || selected.key === "dpcpp" || selected.key === "dpcpp-opencl")) {
    // Level Zero reports a conservative per-allocation ceiling on some full-ReBAR Arc systems.
    // The runtime's relaxed-allocation descriptor removes that artificial limit while allocation
    // failure still safely rejects workloads that do not fit physical VRAM.
    result.UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS = env.UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS || "1";
  }
  if (selected.key === "dpcpp-opencl") {
    // GPU is the mining default. CPU is an explicit correctness/portability mode used to prove
    // that the same standards-only SPIR-V artifact also runs through an OpenCL CPU implementation.
    // Intel integrated GPUs use Level Zero because it is their native low-overhead interface;
    // generic/unknown vendors use the portable OpenCL contract.
    const genericDefaultL0 = backend === "sycl" && gpu === "intel";
    result.ONEAPI_DEVICE_SELECTOR = backend === "sycl-l0" || genericDefaultL0
      ? "level_zero:gpu"
      : `opencl:${openclDeviceType(env)}`;
  }
  // Each compiler ships beside its own SYCL runtime. Workers are separate processes, so putting
  // only the selected directory first avoids same-SONAME collisions (notably DPC++ libsycl) while
  // still allowing the policy to switch compilers when an algorithm changes.
  const sharedDpcppRuntime = selected.key === "dpcpp-opencl"
    ? path.join(path.dirname(libDir), "dpcpp")
    : null;
  if (platform !== "win32") {
    result.LD_LIBRARY_PATH = [libDir, path.join(libDir, "hipSYCL"), sharedDpcppRuntime,
      env.LD_LIBRARY_PATH]
      .filter(Boolean).join(path.delimiter);
  } else {
    // Windows loads compiler runtimes and AdaptiveCpp backend-plugin dependencies through PATH.
    // Keep the selected worker ahead of the shared package root so same-named oneAPI/DPC++/ACPP
    // DLLs cannot cross-load when policy switches compiler for the next algorithm.
    result.PATH = [libDir, path.join(libDir, "hipSYCL"), sharedDpcppRuntime, env.PATH || env.Path]
      .filter(Boolean).join(path.delimiter);
  }
  return result;
}

module.exports = {
  parse, parseReportedAlgoParam, selection, workerEnv, gpuFromEnv, validateBackend, validBackends,
  policyFile
};
