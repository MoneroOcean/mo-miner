"use strict";

/** @typedef {{cpu_sockets: number, cpu_threads: number, cpu_l3cache: number}} CpuInfo */
/** @typedef {{
 *   existsSync(path: string): boolean,
 *   readFileSync(path: string, encoding: "utf8"): string,
 *   readdirSync(path: string): string[],
 * }} EnvironmentFileSystem */
/** @typedef {{cpus(): import("node:os").CpuInfo[]}} EnvironmentOperatingSystem */
/** @typedef {{platform: NodeJS.Platform, env: NodeJS.ProcessEnv}} EnvironmentProcess */

/**
 * @param {{
 *   fs: EnvironmentFileSystem,
 *   os: EnvironmentOperatingSystem,
 *   process: EnvironmentProcess,
 *   opt: MinerOptions,
 *   compilerPolicy: typeof import("../compiler-policy"),
 *   gpuTuning: typeof import("../gpu-tuning"),
 *   normalizeAlgoName(algo: string | null | undefined): string | null | undefined,
 *   requestedJobBackend(algo: string): string,
 *   jobBackend(algo: string): string,
 *   resolvedDeviceList(algo: string, configured: string, heuristic: string, tuning?: GpuTuning): string,
 *   configuredTuning(algo: string): GpuTuning,
 * }} dependencies
 */
module.exports = ({
  fs, os, process, opt, compilerPolicy, gpuTuning, normalizeAlgoName,
  requestedJobBackend, jobBackend, resolvedDeviceList, configuredTuning,
}) => {

  const reservedAlgoParamKeys = new Set(["__proto__", "constructor", "prototype"]);
  // Device selection and donation respect discovery's per-device/VRAM eligibility.
  /** @type {Map<string, ReturnType<typeof gpuTuning.parseDeviceList>>} */
  const discoveredDevices = new Map();

  /** @returns {CpuInfo} */
  function fallbackCpuInfo() {
    return {
      cpu_sockets: 1,
      cpu_threads: os.cpus().length || 1,
      cpu_l3cache: 0,
    };
  }

  /** @param {string} cpuinfo */
  function cpuSocketCount(cpuinfo) {
    const physical_ids = new Set();
    for (const match of cpuinfo.matchAll(/^physical id\s*:\s*(.+)$/gm)) {physical_ids.add(match[1]);}
    return physical_ids.size || 1;
  }

  /** @param {string} size_text */
  function cacheSizeBytes(size_text) {
    const size = size_text.match(/^(\d+)([KMG])$/i);
    if (!size) {return 0;}
    const unit = size[2] && size[2].toUpperCase();
    const multiplier = unit === "K" ? 1024 : unit === "M" ? 1024 * 1024 :
      unit === "G" ? 1024 * 1024 * 1024 : 0;
    const bytes = Number(size[1]) * multiplier;
    return Number.isSafeInteger(bytes) ? bytes : 0;
  }

  /** @param {string} base @param {Set<string>} l3_ids */
  function l3CacheEntryBytes(base, l3_ids) {
    try {
      if (fs.readFileSync(`${base}/type`, "utf8").trim() !== "Unified" ||
          fs.readFileSync(`${base}/level`, "utf8").trim() !== "3") {return 0;}
      const shared_cpu_list = `${base}/shared_cpu_list`;
      const id = fs.existsSync(shared_cpu_list)
        ? fs.readFileSync(shared_cpu_list, "utf8").trim() : base;
      if (l3_ids.has(id)) {return 0;}
      l3_ids.add(id);
      return cacheSizeBytes(fs.readFileSync(`${base}/size`, "utf8").trim());
    } catch {
      return 0;
    }
  }

  function l3CacheBytes() {
    let l3cache = 0;
    const l3_ids = new Set();
    const cpu_dirs = fs.readdirSync("/sys/devices/system/cpu").filter((name) => /^cpu\d+$/.test(name));
    for (const index of cpu_dirs) {
      const cache_dir = `/sys/devices/system/cpu/${index}/cache`;
      if (!fs.existsSync(cache_dir)) {continue;}
      for (const entry of fs.readdirSync(cache_dir)) {l3cache += l3CacheEntryBytes(`${cache_dir}/${entry}`, l3_ids);}
    }
    return l3cache;
  }

  function detect_cpu() {
    const fallback = fallbackCpuInfo();
    try {
      if (!hasProcCpuInfo()) {return fallback;}
      const cpuinfo = fs.readFileSync("/proc/cpuinfo", "utf8");
      const processor_count = (cpuinfo.match(/^processor\s*:/gm) || []).length;
      return {
        cpu_sockets: cpuSocketCount(cpuinfo),
        cpu_threads: processor_count || fallback.cpu_threads,
        cpu_l3cache: l3CacheBytes(),
      };
    } catch {
      return fallback;
    }
  }

  function hasProcCpuInfo() {
    return process.platform !== "win32" && fs.existsSync("/proc/cpuinfo");
  }

  function use_msr_tuning() {
    return process.platform !== "win32" && process.env["MOM_SKIP_MSR"] !== "1";
  }

  /** @param {Record<string, string>} params @returns {Array<[string, string]>} */
  function validatedAlgoParamEntries(params) {
    if (params === null || typeof params !== "object" || Array.isArray(params)) {
      throw new Error("Algorithm parameters must be an object");
    }
    const entries = Object.entries(params);
    for (const [key, value] of entries) {
      const algo = key.startsWith("@backend:") ? key.slice("@backend:".length) : key;
      if (!/^[a-z0-9][a-z0-9_./-]*$/i.test(algo) ||
          reservedAlgoParamKeys.has(algo) ||
          (key.startsWith("@") && !key.startsWith("@backend:"))) {
        throw new Error(`Invalid algorithm parameter key: ${key}`);
      }
      if (typeof value !== "string" || value === "") {
        throw new Error(`Invalid algorithm parameters for ${key}`);
      }
    }
    return entries;
  }

  /** @param {Record<string, string>} params */
  function add_algo_params(params) {
    const entries = validatedAlgoParamEntries(params);
    discoveredDevices.clear();
    for (const [key, value] of entries) {
      if (!key.startsWith("@backend:")) {continue;}
      const algo = key.slice("@backend:".length);
      const configured = Object.hasOwn(opt.algo_params, algo) ? opt.algo_params[algo] : undefined;
      if (configured && (!configured.backend || configured.backend === "auto")) {
        configured.backend = compilerPolicy.validateBackend(value);
      }
    }
    for (const [algo, rawDev] of entries) {
      if (algo.startsWith("@")) {continue;}
      const discovered = gpuTuning.parseDeviceList(rawDev, algo);
      discoveredDevices.set(algo, discovered);
      const configured = Object.hasOwn(opt.algo_params, algo) ? opt.algo_params[algo] : undefined;
      const backendKey = `@backend:${algo}`;
      const backendHint = Object.hasOwn(params, backendKey) ? params[backendKey] : "auto";
      if (!configured) {
        opt.algo_params[algo] = {
          dev: gpuTuning.formatDeviceList(discovered),
          perf: null,
          backend: compilerPolicy.validateBackend(backendHint),
          tuning: {},
        };
      } else {
        configured.dev = resolvedDeviceList(
          algo, configured.dev, rawDev, configured.tuning || {});
      }
    }
  }

  /** @param {string} algo @param {string} dev */
  function donationAlgoSupported(algo, dev) {
    const discovered = discoveredDevices.get(algo);
    if (!discovered) {return false;}
    return gpuTuning.parseDeviceList(dev, algo).every((entry) =>
      discovered.some((candidate) => candidate.device === entry.device));
  }

  /** @param {Record<string, string>} params @returns {Record<string, string>} */
  function publicAlgoParams(params) {
    const entries = validatedAlgoParamEntries(params);
    /** @type {Record<string, string>} */
    const result = {};
    for (const [algo, rawDev] of entries) {
      if (algo.startsWith("@")) {continue;}
      const configured = Object.hasOwn(opt.algo_params, algo) ? opt.algo_params[algo] : undefined;
      const dev = resolvedDeviceList(
        algo, configured && configured.dev ? configured.dev : rawDev,
        rawDev, configuredTuning(algo));
      if (!/\bgpu\d+/i.test(dev)) {
        result[algo] = dev;
        continue;
      }
      const requested = requestedJobBackend(algo);
      const backendKey = `@backend:${algo}`;
      const hinted = Object.hasOwn(params, backendKey) ? params[backendKey] : undefined;
      const resolved = requested === "auto"
        ? compilerPolicy.validateBackend(hinted || jobBackend(algo))
        : requested;
      const label = requested === "auto" && resolved !== "auto"
        ? `auto[${resolved}]` : resolved;
      result[algo] = `${dev}:${label}`;
    }
    return result;
  }

  function prepare_algo_params() {
    const algo = normalizeAlgoName(opt.job.algo);
    const requestedDev = opt.job.dev_request;
    if (!algo && requestedDev === undefined) {return true;}
    const detected = algo && Object.hasOwn(opt.algo_params, algo) ? opt.algo_params[algo] : undefined;
    let algo_param = detected || {dev: opt.job.dev, perf: null, backend: "auto", tuning: {}};
    // An explicit device, including "cpu", overrides discovery while inheriting omitted GPU tuning.
    if (algo && requestedDev !== undefined) {
      algo_param = {
        ...algo_param,
        dev: resolvedDeviceList(
          algo, requestedDev, detected ? detected.dev : requestedDev,
          algo_param.tuning || {}),
      };
    }
    if (algo) {opt.job.algo = algo;}
    // Donation must never widen the user's mining device set or process count. Retain discovery's
    // per-algorithm capability and VRAM decisions within the selected per-device process budget.
    const selectedProcesses = new Map();
    const selectedEntries = gpuTuning.parseDeviceList(
      algo ? algo_param.dev : /** @type {string} */ (requestedDev), algo || "");
    const boundedDevices = new Set();
    const requestedParts = requestedDev?.split(",") || [];
    for (const [index, entry] of selectedEntries.entries()) {
      // Bare auto selectors retain discovery's optimized worker geometry. An explicit process
      // suffix (including ^1), or repeated entries for one device, sets a concrete total budget.
      if (requestedParts[index]?.includes("^") || selectedProcesses.has(entry.device)) {
        boundedDevices.add(entry.device);
      }
      selectedProcesses.set(
        entry.device, (selectedProcesses.get(entry.device) || 0) + entry.processes
      );
    }

    /** @param {string} candidateAlgo @param {AlgoParam} candidate
     * @param {ReturnType<typeof gpuTuning.parseDeviceList>} discovery */
    function select_auto_entries(candidateAlgo, candidate, discovery) {
      const configured = gpuTuning.parseDeviceList(candidate.dev, candidateAlgo);
      const configuredDevices = new Set(configured.map((entry) => entry.device));
      // Retain saved per-device tuning, but recover actual discovery geometry for new devices.
      const heuristic = [...configured, ...discovery.filter((entry) =>
        !configuredDevices.has(entry.device))];
      /** @type {ReturnType<typeof gpuTuning.parseDeviceList>} */
      const result = [];
      for (const [index, part] of requestedParts.entries()) {
        const device = selectedEntries[index]?.device;
        if (!discovery.some((entry) => entry.device === device)) {continue;}
        let requested;
        try {
          // Keep *B in its original form until the algorithm determines its primary tuning field.
          requested = gpuTuning.parseDeviceEntry(part, candidateAlgo);
        } catch (error) {
          const reason = error instanceof Error ? error.message : String(error);
          throw new Error(`Selected device tuning is not supported for ${candidateAlgo}: ${reason}`, {cause: error});
        }
        const matches = heuristic.filter((entry) => entry.device === device);
        const bounded = boundedDevices.has(device);
        if (requested.device.startsWith("cpu")) {
          for (const match of matches) {
            result.push({
              ...requested,
              tuning: {...match.tuning, ...requested.tuning},
              processes: bounded ? requested.processes : match.processes,
            });
          }
          continue;
        }
        // Bare selectors retain each optimized worker geometry. An explicit budget repeats the
        // first profile instead of inventing an aggregate shape for algorithm-specific fields.
        const selectedMatches = bounded ? matches.slice(0, 1) : matches;
        for (const match of selectedMatches) {
          const resolved = resolvedDeviceList(candidateAlgo, part,
            gpuTuning.formatDeviceEntry(match), candidate.tuning || {});
          for (const entry of gpuTuning.parseDeviceList(resolved, candidateAlgo)) {
            result.push({...entry, processes: bounded ? entry.processes : match.processes});
          }
        }
      }
      return result;
    }

    /** @type {Record<string, AlgoParam>} */
    const compatible = {};
    for (const [candidateAlgo, candidate] of Object.entries(opt.algo_params)) {
      const discovery = discoveredDevices.get(candidateAlgo);
      if (!discovery) {continue;}
      const discovered = new Set(discovery.map((entry) => entry.device));
      const resolvedEntries = algo ? gpuTuning.parseDeviceList(candidate.dev, candidateAlgo)
        : select_auto_entries(candidateAlgo, candidate, discovery);
      const remainingProcesses = new Map(selectedProcesses);
      if (!algo) {
        for (const device of discovered) {
          if (boundedDevices.has(device)) {continue;}
          remainingProcesses.set(device, resolvedEntries.filter((entry) => entry.device === device)
            .reduce((total, entry) => total + entry.processes, 0));
        }
      }
      const entries = resolvedEntries
        .filter((entry) => discovered.has(entry.device))
        .map((entry) => {
          const remaining = remainingProcesses.get(entry.device) || 0;
          const processes = Math.min(entry.processes, remaining);
          remainingProcesses.set(entry.device, remaining - processes);
          return processes ? {...entry, processes} : null;
        })
        .filter((entry) => entry !== null);
      if (entries.length) {
        const dev = gpuTuning.formatDeviceList(entries);
        // A measured rate belongs to its effective device geometry, not a narrowed selection.
        // Disabled (zero) entries must remain disabled rather than becoming missing benchmarks.
        const previous = gpuTuning.parseDeviceList(candidate.dev, candidateAlgo)
          .map(gpuTuning.formatDeviceEntry).sort().join(",");
        const selected = entries.map(gpuTuning.formatDeviceEntry).sort().join(",");
        compatible[candidateAlgo] = {
          ...candidate,
          dev,
          perf: !algo && candidate.perf !== null && candidate.perf > 0 && previous !== selected
            ? null : candidate.perf,
        };
      }
    }
    if (algo) {compatible[algo] = algo_param;}
    opt.algo_params = compatible;
    return Object.keys(compatible).length > 0;
  }

  return {
    detect_cpu, use_msr_tuning, add_algo_params, publicAlgoParams,
    prepare_algo_params, donationAlgoSupported,
  };
};
