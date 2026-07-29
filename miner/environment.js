"use strict";

module.exports = ({
  fs, os, process, o, opt, compilerPolicy, gpuTuning, normalizeAlgoName,
  requestedJobBackend, jobBackend, resolvedDeviceList, configuredTuning,
}) => {

  function fallbackCpuInfo() {
    return {
      cpu_sockets: 1,
      cpu_threads: os.cpus().length || 1,
      cpu_l3cache: 0,
    };
  }

  function cpuSocketCount(cpuinfo) {
    const physical_ids = new Set();
    for (const match of cpuinfo.matchAll(/^physical id\s*:\s*(.+)$/gm)) {physical_ids.add(match[1]);}
    return physical_ids.size || 1;
  }

  function cacheSizeBytes(size_text) {
    const size = size_text.match(/^(\d+)([KMG])$/i);
    if (!size) {return 0;}
    const multiplier = { K: 1024, M: 1024 * 1024, G: 1024 * 1024 * 1024 }[size[2].toUpperCase()];
    return Number(size[1]) * multiplier;
  }

  function cacheSharedId(base) {
    const shared_cpu_list = `${base}/shared_cpu_list`;
    return fs.existsSync(shared_cpu_list) ? fs.readFileSync(shared_cpu_list, "utf8").trim() : base;
  }

  function l3CacheEntryBytes(base, l3_ids) {
    try {
      if (!isUnifiedL3Cache(base)) {return 0;}
      const id = cacheSharedId(base);
      if (l3_ids.has(id)) {return 0;}
      l3_ids.add(id);
      return cacheSizeBytes(fs.readFileSync(`${base}/size`, "utf8").trim());
    } catch {
      return 0;
    }
  }

  function isUnifiedL3Cache(base) {
    return fs.readFileSync(`${base}/type`, "utf8").trim() === "Unified" &&
         fs.readFileSync(`${base}/level`, "utf8").trim() === "3";
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
    if (!hasProcCpuInfo()) {return fallback;}

    const cpuinfo = fs.readFileSync("/proc/cpuinfo", "utf8");
    const processor_count = (cpuinfo.match(/^processor\s*:/gm) || []).length;
    return {
      cpu_sockets: cpuSocketCount(cpuinfo),
      cpu_threads: processor_count || fallback.cpu_threads,
      cpu_l3cache: l3CacheBytes(),
    };
  }

  function hasProcCpuInfo() {
    return process.platform !== "win32" && fs.existsSync("/proc/cpuinfo");
  }

  function use_msr_tuning() {
    return process.platform !== "win32" && process.env.MOM_SKIP_MSR !== "1";
  }

  function add_algo_params(params) {
    for (const key in params) {
      if (!key.startsWith("@backend:")) {continue;}
      const algo = key.slice("@backend:".length);
      const configured = opt.algo_params[algo];
      if (configured && (!configured.backend || configured.backend === "auto")) {
        configured.backend = compilerPolicy.validateBackend(params[key]);
      }
    }
    for (const algo in params) {
      if (algo.startsWith("@")) {continue;}
      const configured = opt.algo_params[algo];
      if (!configured) {
        opt.algo_params[algo] = {
          dev: gpuTuning.formatDeviceList(gpuTuning.parseDeviceList(params[algo], algo)),
          perf: null,
          backend: compilerPolicy.validateBackend(params[`@backend:${algo}`] || "auto"),
          tuning: {},
        };
      } else {
        configured.dev = resolvedDeviceList(
          algo, configured.dev, params[algo], configured.tuning || {});
      }
    }
  }

  function publicAlgoParams(params) {
    const result = {};
    for (const [algo, rawDev] of Object.entries(params)) {
      if (algo.startsWith("@")) {continue;}
      const configured = opt.algo_params[algo];
      const dev = resolvedDeviceList(
        algo, configured && configured.dev ? configured.dev : rawDev,
        rawDev, configuredTuning(algo));
      if (!/\bgpu\d+/i.test(dev)) {
        result[algo] = dev;
        continue;
      }
      const requested = requestedJobBackend(algo);
      const hinted = params[`@backend:${algo}`];
      const resolved = requested === "auto"
        ? compilerPolicy.validateBackend(hinted || jobBackend(algo))
        : requested;
      const label = requested === "auto" && resolved !== "auto"
        ? `auto[${resolved}]` : resolved;
      result[algo] = `${dev}:${label}`;
    }
    return result;
  }

  function use_algo_param_benchmarks() {
    return opt.bench_algo_params !== 0;
  }

  function prepare_fixed_algo_params() {
    const algo = normalizeAlgoName(opt.job.algo);
    if (!algo) {return;}
    const detected = opt.algo_params[algo];
    let algo_param = detected || {dev: opt.job.dev, perf: null, backend: "auto", tuning: {}};
    // The default job device is CPU. A non-default --job.dev on a fixed-algorithm command is an
    // explicit user selection and must override discovery while inheriting any omitted GPU tuning.
    if (opt.job.dev !== o.opt_help.job.dev[0]) {
      algo_param = {
        ...algo_param,
        dev: resolvedDeviceList(
          algo, opt.job.dev, detected ? detected.dev : opt.job.dev,
          algo_param.tuning || {}),
      };
    }
    opt.job.algo = algo;
    opt.algo_params = { [algo]: algo_param };
  }

  return {
    detect_cpu, use_msr_tuning, add_algo_params, publicAlgoParams,
    use_algo_param_benchmarks, prepare_fixed_algo_params,
  };
};
