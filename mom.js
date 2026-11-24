// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

"use strict";

const h = require("./helper.js");

/** @typedef {import("node:child_process").ChildProcessWithoutNullStreams} SubprocessWorker */
/** @typedef {import("node:cluster").Worker} ClusterWorker */
/** @typedef {{type: "subprocess", worker: SubprocessWorker} |
 *   {type: "cluster", worker: ClusterWorker}} WorkerTarget */

// Cluster workers stop here after cluster_process() initializes the compute-core side. Keeping the
// application inside this branch avoids constructing the pool and controller runtime in every worker.
if (!h.cluster_process()) {
  const fs = require("node:fs");
  const os = require("node:os");
  const compilerPolicy = require("./compiler-policy.js");
  const gpuTuning = require("./gpu-tuning.js");
  const {normalizeAlgoName} = require("./miner/algorithms");
  const {isObject, errorMessage, isPositiveNumber} = require("./helper/validate");
  const o = require("./opts.js");
  const p = require("./pool.js");
  global.opt = o.create_default_opts();

  /** @type {ComputeCore | null} */
  let computeCore = null;
  /** @type {Promise<void>} */
  let computeCoreClose = Promise.resolve();
  /** @type {((hashrate: number) => unknown) | null} */
  let algoParamsBenchCallback = null; // used to record algo_params bench data
  /** @type {MiningJob | null} */
  let lastJob = null;
  /** @type {string | null} */
  let directive = null;
  /** @type {MinerTestState} */
  const test = {
    result_hash_hex: null,
    thread_tested:   0,
    result:          ""
  };
  let exiting = false;

  const WORKER_CLOSE_GRACE_MS = 3000;
  const PROCESS_EXIT_GRACE_MS = 5000;

  // Exit through the native core's exit hook once buffered stdout and stderr have been written.
  /** @param {number} code */
  function exitAfterFlush(code) {
    const finish = () => {
      h.exit_now(code);
    };

    setImmediate(() => {
      process.stdout.write("", () => {
        process.stderr.write("", finish);
      });
    });
  }

  /** @param {string | null} algo @param {string} value */
  function normalizeTestResult(algo, value) {
    if (algo !== "c29") {return value.trim();}

    const tokens = value.trim().split(/\s+/);
    const hasEol = tokens[tokens.length - 1] === "EOL";
    if (hasEol) {tokens.pop();}

    return tokens.sort().join(" ") + (hasEol ? " EOL" : "");
  }

  /** @param {string | null} algo @param {string} value */
  function normalizeExpectedResults(algo, value) {
    return value.split("|").map((expected) => normalizeTestResult(algo, expected));
  }

  /** @param {string | null} algo @param {string} actual @param {string} expected */
  function matchesTestResult(algo, actual, expected) {
    if (algo === "c29") {
      return normalizeTestResult(algo, actual) === normalizeTestResult(algo, expected);
    }
    const actualTokens = actual.trim().split(/\s+/);
    const expectedTokens = expected.trim().split(/\s+/);
    return actualTokens.length % expectedTokens.length === 0 &&
      actualTokens.every((token, index) => token === expectedTokens[index % expectedTokens.length]);
  }

  function closeComputeCore() {
    if (!computeCore) {return;}
    if (Object.keys(global.opt.default_msrs).length) {
      computeCore.emit_to("write_msr", h.pack_msr(global.opt.default_msrs));
    }
    computeCore.emit_to("close");
    computeCore = null;
  }

  /** @param {WorkerTarget} target */
  function waitForWorkerClose(target) {
    if (target.type === "subprocess") {
      const worker = target.worker;
      if (worker.exitCode !== null || worker.signalCode !== null) {return Promise.resolve();}
      /** @type {Promise<void>} */
      return new Promise((resolve) => worker.once("close", () => resolve()));
    }
    const worker = target.worker;
    if (worker.isDead && worker.isDead()) {return Promise.resolve();}
    /** @type {Promise<void>} */
    return new Promise((resolve) => worker.once("exit", () => resolve()));
  }

  // Shut down once: restore MSRs, close the core and workers, then exit. With force, a deadline
  // ends a hung cleanup unless a controlling test harness owns that deadline. A controlled mine
  // waits for the core, the workers, and pending share replies before exiting.
  /** @param {number} code @param {boolean} [force] @returns {false} */
  function exit(code, force = true) {
    const harnessOwnsDeadline = ((directive === "bench" || directive === "mine") &&
      process.env["MOM_BENCHMARK_CONTROL_STDIN"] === "1") ||
      (directive === "test" && process.env["MOM_GPU_TEST_GRACEFUL_ONLY"] === "1");
    const controlledMine = directive === "mine" &&
      process.env["MOM_BENCHMARK_CONTROL_STDIN"] === "1";
    // A second exit() (e.g. SIGINT during shutdown) must not re-run teardown; just honor force.
    if (exiting) {
      if (force && !harnessOwnsDeadline) {exitAfterFlush(code);}
      return false;
    }
    exiting = true;
    // Explicitly controlled tests/benchmarks own the outer deadline; avoid nested forced cleanup.
    const coreClose = controlledMine ? computeCoreClose : null;
    closeComputeCore();
    const workerTargets = h.closeWorkers(force && !harnessOwnsDeadline ? WORKER_CLOSE_GRACE_MS : null);
    process.exitCode = code;
    if (controlledMine) {
      // Mine mode's controller still owns pool sockets and timers, so terminate it only after native
      // core and workers complete their natural cleanup; controller teardown releases its resources.
      Promise.all([coreClose, ...workerTargets.map(waitForWorkerClose)])
        .then(() => p.waitForPendingShares())
        .then(() => exitAfterFlush(/** @type {number} */ (process.exitCode ?? code)));
      return false;
    }
    if (force && !harnessOwnsDeadline) {
      setTimeout(() => exitAfterFlush(code), PROCESS_EXIT_GRACE_MS).unref();
    }
    return false;
  }

  /** @param {string} msg @returns {false} */
  function errorExit(msg) {
    h.log_err(msg);
    return exit(1);
  }

  const submission = require("./miner/submission");
  const parseArgs = require("./miner/cli")({o, opt: global.opt, normalizeAlgoName});
  directive = parseArgs(process.argv, test);

  const {expectedTestThreads, messageHandler, resetHashrates} = require("./miner/messages")({
    fs, h, p, opt: global.opt, submission, test, normalizeExpectedResults,
    matchesTestResult, exit, getLastJob: () => lastJob,
    getAlgoParamsBenchCallback: () => algoParamsBenchCallback,
  });

  const jobApi = require("./miner/jobs")({
    h, opt: global.opt, process, compilerPolicy, gpuTuning,
    messageHandler, isExiting: () => exiting,
    getComputeCore: () => computeCore,
    getLastJob: () => lastJob,
    setLastJob: (job) => { lastJob = job; },
  });
  const {
    setAlgoMsr, requestedJobBackend, jobBackend, resolvedDeviceList,
    configuredTuning, workerRuntimeEnv, setJob,
    prepareTestJob, prepareBenchmarkJob, defaultBenchAlgos, moneroOceanAlgos,
  } = jobApi;

  // Benchmark one algorithm on its configured (or the given) devices and call cb once with the mean
  // of `samples` measured rates, or 0 on failure or timeout. Only one benchmark runs at a time: the
  // active one owns algoParamsBenchCallback, and a stale callback is ignored.
  /**
   * @param {string} algo
   * @param {(hashrate: number) => unknown} cb
   * @param {string | null} [dev]
   * @param {number} [samples]
   */
  function benchAlgo(algo, cb, dev = null, samples = 1) {
    if (exiting) {return;}
    const configured = global.opt.algo_params[algo];
    const benchmarkDev = dev || configured?.dev;
    if (!benchmarkDev) {throw new Error(`No device is configured for ${algo}`);}
    const job = prepareBenchmarkJob({
      algo:     algo,
      dev:      benchmarkDev,
      blob_hex: global.opt.job.blob_hex,
      seed_hex: global.opt.job.seed_hex,
      pool_id:  "", // to drop last nonce messages from this job
    });
    h.recreate_threads(job.dev, messageHandler, (entry) => workerRuntimeEnv(algo, entry));
    // Live-size DAG/table builds (benchHeightByAlgo) take ~30s on a fast GPU before the
    // 60s+ measurement window even starts, so the old 2 minute cap could cut off honest runs.
    // Intel and AMD CryptoNight-GPU workers need two complete windows to reach their steady rate on
    // both operating systems. Exclude those cold windows from profitability and tuning decisions.
    const gpuVendor = compilerPolicy.gpuFromEnv(process.env);
    const warmupSamples = algo === "cn/gpu" &&
      ["intel", "intel_igpu", "amd"].includes(gpuVendor) ? 2 : 0;
    /** @type {number[]} */
    const rates = [];
    /** @param {number} hashrate */
    const finish = function(hashrate) {
      if (algoParamsBenchCallback !== record) {return;}
      algoParamsBenchCallback = null;
      resetHashrates();
      clearTimeout(timeout);
      return cb(hashrate);
    };
    /** @param {number} hashrate */
    const record = function(hashrate) {
      if (!(hashrate > 0)) {return finish(0);}
      rates.push(hashrate);
      if (rates.length < samples + warmupSamples) {return;}
      const measuredRates = rates.slice(warmupSamples);
      return finish(measuredRates.reduce((sum, rate) => sum + rate, 0) / measuredRates.length);
    };
    // Verthash may generate its dataset on the first dispatch. Native timing discards that dispatch,
    // so generation cannot dilute the reported rate; only give the one-time setup room to finish.
    const setupTimeout = algo === "verthash" ? 30*60*1000 : 0;
    // Large Pearl profiles need two long dispatches before the first timed rate, then one per sample.
    const firstSampleMinutes = algo === "pearlhash" ? 6 : 4;
    const additionalSampleMinutes = algo === "pearlhash" ? 3 : 2;
    const timeout = setTimeout(function() {
      h.log_err("Benchmark " + algo + " algo (" + job.dev + ") timeout");
      return finish(0);
    }, setupTimeout + (firstSampleMinutes +
      Math.max(0, samples + warmupSamples - 1) * additionalSampleMinutes)*60*1000);
    algoParamsBenchCallback = record;
    setAlgoMsr(algo);
    h.messageWorkers({type: "bench", job: lastJob = job});
  }

  const gpuAutotune = require("./miner/gpu-autotune")({
    h, opt: global.opt, gpuTuning, benchAlgo: benchAlgo,
  });

  // do global.opt.algo_params benchmarks if perf === null
  /** @param {() => unknown} cb */
  function benchAlgos(cb) {
    const algos = benchmarkAlgos();
    let beforeFirstBenchmark = true;
    h.repeat(function(next) {
      const algo = nextAlgoToBenchmark(algos);
      if (!algo) {
        // This is an explicitly expensive first-run action, not a persistent mining mode.
        global.opt.gpu_tune = 0;
        return cb();
      }
      if (beforeFirstBenchmark) {h.log("Doing algo benchmarks...");}
      beforeFirstBenchmark = false;
      const benchmark = () => benchAlgo(algo, function(hashrate) {
        const params = global.opt.algo_params[algo];
        if (!params) {throw new Error(`Missing benchmark parameters for ${algo}`);}
        params.perf = hashrate;
        return next();
      });
      if (global.opt.gpu_tune) {return gpuAutotune.tuneAlgo(algo, benchmark);}
      return benchmark();
    });
  }

  /** @returns {string[]} */
  function benchmarkAlgos() {
    const algos = Object.keys(global.opt.algo_params);
    if (global.opt.bench_algo_params === 2) {return algos;}
    return algos.filter((algo) => defaultBenchAlgos.has(algo));
  }

  /** @param {string[]} algos @returns {string | undefined} */
  function nextAlgoToBenchmark(algos) {
    let algo;
    // skip until next algo with null perf
    while ((algo = algos.shift())) {
      if (global.opt.algo_params[algo]?.perf === null) {break;}
    }
    return algo;
  }

  function saveConfig() {
    const saveFile = global.opt.save_config;
    if (!saveFile) {return;}
    h.log("Saving config file to " + saveFile);
    try {
      fs.writeFileSync(saveFile, JSON.stringify(o.saved_config(global.opt), null, 2), {mode: 0o600});
      if (process.platform !== "win32") {fs.chmodSync(saveFile, 0o600);}
    } catch {
      h.log_err("Error saving " + saveFile + " file");
    }
  }

  // setup all pool share report
  function scheduleShareStats() {
    setInterval(function() {
      let goodShares = 0, badShares = 0;
      for (const pool of global.opt.pools) {
        goodShares += pool.good_shares;
        badShares += pool.bad_shares;
      }
      h.log("Accepted (" + goodShares + ") / Rejected (" + badShares + ") shares");
    }, global.opt.pool_time.stats * 1000);
  }

  // if there are backup pools, try to reconnect to primary pool if it is not active
  function schedulePrimaryReconnect() {
    if (global.opt.pools.length >= (global.opt.pool_ids.donate !== null ? 3 : 2)) {
      setInterval(function() {
        switch (global.opt.pool_ids.active) {
          case global.opt.pool_ids.primary:
          case global.opt.pool_ids.donate: return;
          default: break;
        }
        const primaryPool = global.opt.pool_ids.primary;
        if (primaryPool !== null) {p.connect_pool_throttle(primaryPool, setJob);}
      }, global.opt.pool_time.primary_reconnect * 1000);
    }
  }

  // Open one donation window: allow and connect the donation pool, then switch away when it ends.
  function startDonationWindow() {
    const poolId = global.opt.pool_ids.donate;
    if (poolId === null) {return;}
    const pool = global.opt.pools[poolId];
    if (!pool) {return;}
    const duration = global.opt.pool_time.donate_length * 1000;
    const until = Date.now() + duration;
    pool.donation_until = until;
    p.connect_pool_throttle(poolId, setJob);
    setTimeout(function() {
      if (pool.donation_until !== until) {return;}
      pool.donation_until = 0;
      p.switch_pool(poolId, setJob);
    }, duration);
  }

  function scheduleDonationMining() {
    if (global.opt.pool_ids.donate !== null) {
      setInterval(startDonationWindow, global.opt.pool_time.donate_interval * 1000);
    }
  }

  function donationAlgoParams() {
    return Object.fromEntries(Object.entries(global.opt.algo_params)
      .filter(([algo, params]) => moneroOceanAlgos.has(algo) &&
      donationAlgoSupported(algo, params.dev) &&
      isPositiveNumber(params.perf)));
  }

  // Point the donation pool at the MoM proxy with the algorithms it may assign: the current fixed
  // algorithm when the proxy supports it, otherwise every measured MoneroOcean algorithm on the
  // selected devices. Without any, drop the donation pool.
  function configureDonationMining() {
    const poolId = global.opt.pool_ids.donate;
    if (poolId === null) {return;}
    const measured = donationAlgoParams();
    const primaryAlgo = normalizeAlgoName(global.opt.job.algo);
    // The unified proxy supports fixed PearlHash mining, but not MO profitability switching.
    const primaryParams = primaryAlgo &&
      (moneroOceanAlgos.has(primaryAlgo) || primaryAlgo === "pearlhash")
      ? global.opt.algo_params[primaryAlgo] || null : null;
    // Prefer the current proxy-supported algorithm so a short donation interval can reuse its setup.
    // For other external-pool algorithms, let MO choose among measured algos on the same devices.
    const selected = primaryParams && primaryAlgo &&
      donationAlgoSupported(primaryAlgo, primaryParams.dev) ? {
        [primaryAlgo]: {
          ...primaryParams,
          perf: isPositiveNumber(primaryParams.perf) ? primaryParams.perf : 1,
        },
      } : measured;
    const selectedAlgos = Object.keys(selected);
    if (!selectedAlgos.length) {
      h.log1("Donation mining is disabled because no compatible benchmark is available");
      global.opt.pools.splice(poolId, 1);
      if (global.opt.pool_ids.primary !== null && global.opt.pool_ids.primary > poolId) {
        global.opt.pool_ids.primary--;
      }
      global.opt.pool_ids.donate = null;
      return;
    }
    const pool = global.opt.pools[poolId];
    if (!pool) {throw new Error(`Unknown donation pool ${poolId}`);}
    // Route every donation algorithm through the unified MO proxy endpoint.
    Object.assign(pool, {
      url: "mom.moneroocean.stream",
      port: 20001,
      login: "user",
      is_tls: true,
      is_nicehash: false,
      tls_verify: false,
      // Pins belong to an endpoint; a user's original certificate cannot identify the donation proxy.
      tls_fingerprint: "",
      protocol: null,
      use_subscribe: false,
      pass: "mom",
      algo_params: selected,
      donation_until: 0,
    });
  }

  // Save the config, finish option setup, and connect to the primary pool with periodic timers.
  function startMining() {
    if (exiting) {return;}
    saveConfig();
    h.log2("Options: " + JSON.stringify(o.redacted_options(global.opt)));
    o.set_internal_opts(global.opt, o.opt_help);
    h.log3("Internal options: " + JSON.stringify(o.redacted_options(global.opt)));
    configureDonationMining();
    const primaryPool = global.opt.pool_ids.primary;
    if (primaryPool === null) {throw new Error("Primary pool is not configured");}
    global.opt.pool_ids.active = primaryPool;
    p.connect_pool_throttle(primaryPool, setJob);
    scheduleShareStats();
    schedulePrimaryReconnect();
    scheduleDonationMining();
  }

  function onExitSignal() {
    if (process.env["MOM_BENCHMARK_CONTROL_STDIN"] === "1") {
      process.stdin.destroy();
    }
    exit(0, true);
  }

  function installExitHandlers() {
    process.on("SIGINT", onExitSignal);
    process.on("SIGTERM", onExitSignal);
    if (process.platform === "win32") {
      process.on("SIGBREAK", onExitSignal);
    } else {
      process.on("SIGHUP", onExitSignal);
    }
  }

  // Controlled tests may close mine or bench through this pipe. Windows emulates SIGTERM by
  // terminating only the parent, so an explicit close lets compute workers and N-API/SYCL cleanup
  // hooks run before the process exits.
  function installControlPipe() {
    if (process.env["MOM_BENCHMARK_CONTROL_STDIN"] !== "1") {return;}
    const maxControlLine = 64;
    let input = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", function(chunk) {
      if (exiting) {return;}
      input += chunk;
      let eol;
      while ((eol = input.indexOf("\n")) !== -1) {
        if (eol > maxControlLine) {return onExitSignal();}
        const command = input.slice(0, eol).trim();
        input = input.slice(eol + 1);
        if (command === "close") {return onExitSignal();}
      }
      if (input.length > maxControlLine) {return onExitSignal();}
    });
    process.stdin.resume();
  }

  const environment = require("./miner/environment")({
    fs, os, process, opt: global.opt, compilerPolicy, gpuTuning, normalizeAlgoName,
    requestedJobBackend, jobBackend, resolvedDeviceList, configuredTuning,
  });
  const {
    detectCpu, useMsrTuning, addAlgoParams, publicAlgoParams,
    prepareAlgoParams, donationAlgoSupported,
  } = environment;

  function startAfterAlgoParams() {
    try {
      if (!prepareAlgoParams()) {
        return errorExit("No supported algorithms for the selected mining devices");
      }
    } catch (error) {
      return errorExit(errorMessage(error));
    }
    if (global.opt.bench_algo_params !== 0) {return benchAlgos(startMining);}
    return startMining();
  }

  // Create the controller's control core once. Late core errors (for example a refused write_msr)
  // are logged instead of crashing when no request is waiting for a response.
  function createComputeCore() {
    if (computeCore) {return computeCore;}
    // The control core performs device discovery/MSR work before an algorithm worker exists. A
    // package launcher already selected an addon and matching runtime; preserve that atomic choice.
    // A source or standalone-native launch needs that policy choice here; native mode selects a
    // linked compiler key, never an addon. A real algorithm name is deliberately not used:
    // overrides belong only to hashing workers.
    if (globalThis.__momNativeCore !== undefined || !process.env["MOM_NATIVE_PATH"]) {
      const controlEnv = compilerPolicy.workerEnv("__control__", process.env, process.platform);
      if (controlEnv["MOM_NATIVE_PATH"]) {
        controlEnv["MOM_NATIVE_PATH_LAUNCHER_DEFAULT"] = controlEnv["MOM_NATIVE_PATH"];
      }
      Object.assign(process.env, controlEnv);
    }
    const core = h.create_core();
    computeCore = core;
    // Native command failures after startup (for example write_msr under kernel lockdown) have no
    // onceCoreResponse listener; log them instead of letting EventEmitter throw on "error".
    core.from.on("error", (error) => {
      if (core.from.listenerCount("error") === 1) {h.log_err("Compute core error: " + errorMessage(error));}
    });
    computeCoreClose = new Promise((resolve) => core.from.once("close", () => {
      if (process.exitCode == null) {process.exitCode = 0;}
      resolve();
    }));
    return core;
  }

  // Wait for one named core response or error; whichever arrives first removes both listeners.
  /**
   * @param {string} name
   * @param {(value: unknown) => unknown} onResult
   * @param {(error: unknown) => unknown} onError
   */
  function onceCoreResponse(name, onResult, onError) {
    const emitter = createComputeCore().from;
    function cleanup() {
      emitter.removeListener(name, handleResult);
      emitter.removeListener("error", handleError);
    }
    /** @param {unknown} value */
    function handleResult(value) {
      cleanup();
      if (exiting) {return;}
      return onResult(value);
    }
    /** @param {unknown} value */
    function handleError(value) {
      cleanup();
      if (exiting) {return;}
      return onError(value);
    }
    emitter.once(name, handleResult);
    emitter.once("error", handleError);
  }

  /** @param {(value: unknown) => unknown} onRead @param {(error?: unknown) => unknown} onError */
  function readMsrThen(onRead, onError) {
    if (!useMsrTuning()) {return onError();}
    onceCoreResponse("read_msr", onRead, function(v) {
      if (v) {h.log("Can't access MSR: " + JSON.stringify(errorMessage(v)));}
      return onError(v);
    });
    return createComputeCore().emit_to("read_msr", h.pack_msr(global.opt.default_msrs));
  }

  /** @param {unknown} value @returns {value is Record<string, string>} */
  function isStringRecord(value) {
    return isObject(value) && Object.values(value).every((entry) => typeof entry === "string");
  }

  /** @returns {string} */
  function directAlgo() {
    const algo = normalizeAlgoName(global.opt.job.algo);
    if (!algo) {throw new Error("A mining algorithm is required");}
    return algo;
  }

  /** @returns {NativeJob} */
  function directJob() {
    const {dev_request: _devRequest, ...job} = global.opt.job;
    return {...job, algo: directAlgo(), dev: job.dev};
  }

  /** @param {Record<string, string>} params */
  function resolveDirectJobTuning(params) {
    const algo = directAlgo();
    const heuristicDev = params[algo];
    if (!heuristicDev) {
      errorExit(`No automatic GPU tuning is available for ${algo} on the selected device`);
      return false;
    }
    global.opt.job.dev = resolvedDeviceList(
      algo, global.opt.job.dev, heuristicDev, configuredTuning(algo));
    return true;
  }

  // For bench/test, fill missing GPU primary tuning and resolve an auto GPU backend from the core's
  // device discovery before starting; otherwise start immediately.
  /** @param {() => unknown} start */
  function startWithDirectJobTuning(start) {
    const algo = directAlgo();
    const needsTuning = gpuTuning.needsPrimaryTuning(global.opt.job.dev, algo);
    const autoGpuBackend = requestedJobBackend(algo) === "auto" &&
      gpuTuning.parseDeviceList(global.opt.job.dev, algo).some((entry) => entry.device.startsWith("gpu"));
    // Explicit intensity does not resolve a device-dependent backend. Use the same discovery
    // hints as mining so bench/test do not silently select a different runtime.
    if (!needsTuning && !autoGpuBackend) {return start();}
    createComputeCore();
    /** @param {unknown} value */
    const onError = function(value) {
      errorExit("Can't derive automatic GPU tuning: " + JSON.stringify(errorMessage(value)));
    };
    onceCoreResponse("algo_params", function(params) {
      if (!isStringRecord(params)) {return onError("Malformed automatic GPU tuning response");}
      if (autoGpuBackend) {addAlgoParams(params);}
      if (!needsTuning || resolveDirectJobTuning(params)) {start();}
    }, onError);
    return createComputeCore().emit_to("algo_params", detectCpu());
  }

  function startTestJob() {
    const algo = directAlgo();
    const job = prepareTestJob(directJob());
    h.recreate_threads(job.dev, messageHandler,
      (entry) => workerRuntimeEnv(algo, entry));
    h.messageWorkers({type: "test", job});
  }

  // Start the bench directive's workers, applying the algorithm's MSR preset when MSR access works.
  function startDirectBenchmark() {
    const job = prepareBenchmarkJob(directJob());
    const startBenchJob = () => {
      lastJob = job;
      h.messageWorkers({type: "bench", job});
    };
    h.recreate_threads(job.dev, messageHandler,
      (entry) => workerRuntimeEnv(job.algo, entry));
    if (!useMsrTuning()) {
      startBenchJob();
      return;
    }
    createComputeCore();
    readMsrThen(function(v) {
      if (!isStringRecord(v)) {throw new Error("Malformed MSR response");}
      global.opt.default_msrs = h.unpack_msr(v); // to restore them on exit
      setAlgoMsr(job.algo);
      startBenchJob();
    }, startBenchJob);
  }

  switch (directive) {
    case "mine":
      installExitHandlers();
      installControlPipe();
      createComputeCore();
      onceCoreResponse("algo_params", function(v) {
        if (!isStringRecord(v)) {return errorExit("Malformed algorithm parameters response");}
        addAlgoParams(v);
        return readMsrThen(function(v) {
          if (!isStringRecord(v)) {return errorExit("Malformed MSR response");}
          global.opt.default_msrs = h.unpack_msr(v);
          return startAfterAlgoParams();
        }, function() {
          global.opt.default_msrs = {};
          return startAfterAlgoParams();
        });
      }, function(v) {
        errorExit("Can't detect algo params: " + JSON.stringify(errorMessage(v)));
      });
      createComputeCore().emit_to("algo_params", detectCpu());
      break;

    case "test":
      startWithDirectJobTuning(startTestJob);
      break;

    case "bench":
      installExitHandlers();
      installControlPipe();
      startWithDirectJobTuning(startDirectBenchmark);
      break;

    case "algos":
      createComputeCore();
      onceCoreResponse("algo_params", function(v) {
        if (!isStringRecord(v)) {return errorExit("Malformed algorithm parameters response");}
        fs.writeSync(1, "MOM_ALGORITHMS " + JSON.stringify(publicAlgoParams(v)) + "\n");
        return exit(0);
      }, function(v) {
        errorExit("Can't detect algorithms: " + JSON.stringify(errorMessage(v)));
      });
      createComputeCore().emit_to("algo_params", detectCpu());
      break;
    default:
      break;
  }

  module.exports.__test = {
    expectedTestThreads, matchesTestResult, messageHandler, publicAlgoParams, startDonationWindow,
  };
}
