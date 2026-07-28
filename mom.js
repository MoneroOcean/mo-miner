// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

"use strict";

const path = require("path");
const fs   = require("fs");
const os   = require("os");
const h    = require("./helper.js");
const compilerPolicy = require("./compiler-policy.js");
const gpuTuning = require("./gpu-tuning.js");
const {normalizeAlgoName} = require("./miner/algorithms");
const o    = require("./opts.js");
const p    = require("./pool.js");

// compute core wrapper for cluster process fork
if (h.cluster_process()) {return;}

global.opt = {};

let compute_core = null;
let algo_params_bench_cb = null; // used to record algo_params bench data
let last_job = null;
let directive = null;
const test = {
  result_hash_hex: null,
  thread_tested:   0,
  result:          ""
};
let is_exiting = false;

const WORKER_CLOSE_GRACE_MS = 3000;
const PROCESS_EXIT_GRACE_MS = 5000;

o.set_default_opts(global.opt, o.opt_help);

function orDefault(value, fallback) {
  return value ? value : fallback;
}

// Like orDefault but keyed on presence, so an explicit nonceoffset of 0 is honored (not treated as falsy).
function nonceOffsetOr(prev_job, fallback) {
  return typeof prev_job.nonceoffset !== "undefined" ? prev_job.nonceoffset : fallback;
}

function firstTruthyOr(fallback, ...values) {
  return values.find(Boolean) || fallback;
}

function reallyExit(code) {
  const finish = () => {
    if (h.exit_now) {h.exit_now(code);}
    else {process.exit(code);}
  };

  setImmediate(() => {
    process.stdout.write("", () => {
      process.stderr.write("", finish);
    });
  });
}

function normalizeTestResult(algo, value) {
  if (algo !== "c29") {return value.trim();}

  const tokens = value.trim().split(/\s+/);
  const hasEol = tokens[tokens.length - 1] === "EOL";
  if (hasEol) {tokens.pop();}

  return tokens.sort().join(" ") + (hasEol ? " EOL" : "");
}

function normalizeExpectedResults(algo, value) {
  return value.split("|").map((expected) => normalizeTestResult(algo, expected));
}

function matchesTestResult(algo, actual, expected) {
  if (algo === "c29") {
    return normalizeTestResult(algo, actual) === normalizeTestResult(algo, expected);
  }
  const actualTokens = actual.trim().split(/\s+/);
  const expectedTokens = expected.trim().split(/\s+/);
  return expectedTokens.length > 0 &&
    actualTokens.length % expectedTokens.length === 0 &&
    actualTokens.every((token, index) => token === expectedTokens[index % expectedTokens.length]);
}

function forceExitByDefault() {
  return directive === "mine" || directive === "bench" || shouldExitAfterWorkerShutdown();
}

function closeComputeCore() {
  if (!compute_core) {return;}
  if (Object.keys(global.opt.default_msrs).length)
  {compute_core.emit_to("write_msr", h.pack_msr(global.opt.default_msrs));}
  compute_core.emit_to("close");
  compute_core = null;
}

function shouldExitAfterWorkerShutdown() {
  return directive === "test" || directive === "algo_params";
}

function exit(code, force = forceExitByDefault()) {
  // A second exit() (e.g. SIGINT during shutdown) must not re-run teardown; just honor force.
  if (is_exiting) {
    if (force) {reallyExit(code);}
    return false;
  }
  is_exiting = true;
  closeComputeCore();
  h.closeWorkers(force ? WORKER_CLOSE_GRACE_MS : null);
  process.exitCode = code;
  if (force) {
    // The benchmark harness must not record success if its explicit graceful close reaches this
    // emergency deadline. A normal worker drain exits before the timer; 124 identifies a stuck
    // worker/cleanup path instead of silently using the requested status and leaving GPU work behind.
    const deadlineCode = process.env.MOM_BENCHMARK_CONTROL_STDIN === "1" ? 124 : code;
    setTimeout(() => reallyExit(deadlineCode), PROCESS_EXIT_GRACE_MS).unref();
  }
  return false;
}

function err_exit(msg) {
  h.log_err(msg);
  return exit(1);
}

const submission = require("./miner/submission");
const {hexWithoutPrefix} = submission;
const parseArgs = require("./miner/cli")({h, o, opt: global.opt, path, normalizeAlgoName});
directive = parseArgs(process.argv, test);
if (!directive) {return;}

// expectedTestThreads is also exposed to the VM-based logic harness appended by tests/logic/support.
// eslint-disable-next-line no-unused-vars
const {expectedTestThreads, messageHandler} = require("./miner/messages")({
  fs, h, p, opt: global.opt, submission, test, firstTruthyOr, normalizeExpectedResults,
  matchesTestResult, exit, getLastJob: () => last_job,
  getAlgoParamsBenchCallback: () => algo_params_bench_cb,
});

const jobApi = require("./miner/jobs")({
  h, opt: global.opt, process, compilerPolicy, gpuTuning,
  orDefault, nonceOffsetOr, firstTruthyOr,
  hexWithoutPrefix, normalizeAlgoName, messageHandler,
  getComputeCore: () => compute_core,
  getLastJob: () => last_job,
  setLastJob: (job) => { last_job = job; },
});
const {
  set_algo_msr, requestedJobBackend, jobBackend, resolvedDeviceList,
  configuredTuning, addPearlHashJobFields, workerRuntimeEnv, set_job,
  prepareBenchmarkJob, defaultBenchAlgos,
} = jobApi;
function bench_algo(algo, cb) {
  const job = prepareBenchmarkJob({
    algo:     algo,
    dev:      global.opt.algo_params[algo].dev,
    blob_hex: global.opt.job.blob_hex,
    seed_hex: global.opt.job.seed_hex,
    pool_id:  "", // to drop last nonce messages from this job
  });
  h.recreate_threads(job.dev, messageHandler, (entry) => workerRuntimeEnv(algo, entry));
  // Live-size DAG/table builds (benchHeightByAlgo) take ~30s on a fast GPU before the
  // 60s+ measurement window even starts, so the old 2 minute cap could cut off honest runs.
  const timeout = setTimeout(function() {
    h.log_err("Benchmark " + algo + " algo (" + job.dev + ") timeout");
    return cb(0);
  }, 4*60*1000);
  algo_params_bench_cb = function(hashrate) { clearTimeout(timeout); return cb(hashrate); };
  set_algo_msr(algo);
  h.messageWorkers({type: "bench", job: last_job = job});
}

// do global.opt.algo_params benchmarks if perf === null
function bench_algos(cb) {
  const algos = benchmarkAlgos();
  let is_before_first_benchmark = true;
  h.repeat(function(cb_next) {
    const algo = nextAlgoToBenchmark(algos);
    if (!algo) {return cb();}
    if (is_before_first_benchmark) {h.log("Doing algo benchmarks...");}
    is_before_first_benchmark = false;
    bench_algo(algo, function(hashrate) {
      algo_params_bench_cb = null;
      global.opt.algo_params[algo].perf = hashrate;
      return cb_next();
    });
  });
}

function benchmarkAlgos() {
  const algos = Object.keys(global.opt.algo_params);
  if (global.opt.bench_algo_params === 2) {return algos;}
  return algos.filter((algo) => defaultBenchAlgos.has(algo));
}

function nextAlgoToBenchmark(algos) {
  let algo;
  // skip until next algo with null perf
  while ((algo = algos.shift()) && global.opt.algo_params[algo].perf !== null){;}
  return algo;
}

function saveConfig() {
  const save_config = global.opt.save_config;
  if (!save_config) {return;}
  delete global.opt.save_config; // a saved config should not re-save itself when later loaded
  h.log("Saving config file to " + save_config);
  fs.writeFile(save_config, JSON.stringify(o.saved_config(global.opt), null, 2), function(err) {
    if (err) {h.log_err("Error saving " + save_config + " file");}
  });
}

// setup all pool share report
function scheduleShareStats() {
  setInterval(function() {
    let good_shares = 0, bad_shares = 0;
    for (const pool_id in global.opt.pools) {
      good_shares += global.opt.pools[pool_id].good_shares;
      bad_shares += global.opt.pools[pool_id].bad_shares;
    }
    h.log("Accepted (" + good_shares + ") / Rejected (" + bad_shares + ") shares");
  }, global.opt.pool_time.stats * 1000);
}

// if there are backup pools, try to reconnect to primary pool if it is not active
function schedulePrimaryReconnect() {
  if (global.opt.pools.length >= (global.opt.pool_ids.donate !== null ? 3 : 2))
  {setInterval(function() {
    switch (global.opt.pool_ids.active) {
      case global.opt.pool_ids.primary:
      case global.opt.pool_ids.donate: return;
      default: break;
    }
    p.connect_pool_throttle(global.opt.pool_ids.primary, set_job);
  }, global.opt.pool_time.primary_reconnect * 1000);}
}

// donation mining
function scheduleDonationMining() {
  if (global.opt.pool_ids.donate !== null) {setInterval(function() {
    p.connect_pool_throttle(global.opt.pool_ids.donate, set_job);
    setTimeout(p.switch_pool, global.opt.pool_time.donate_length * 1000,
      global.opt.pool_ids.donate, set_job);
  }, global.opt.pool_time.donate_interval * 1000);}
}

function start_mining() {
  saveConfig();
  h.log2("Options: " + JSON.stringify(global.opt));
  o.set_internal_opts(global.opt, o.opt_help);
  h.log3("Internal options: " + JSON.stringify(global.opt));
  p.connect_pool_throttle(global.opt.pool_ids.active = global.opt.pool_ids.primary, set_job);
  scheduleShareStats();
  schedulePrimaryReconnect();
  scheduleDonationMining();
}

function on_exit() { exit(0, true); }

function install_exit_handlers() {
  process.on("SIGINT", on_exit);
  process.on("SIGTERM", on_exit);
  if (process.platform === "win32") {process.on("SIGBREAK", on_exit);}
  else {process.on("SIGHUP", on_exit);}
}

function install_benchmark_control() {
  if (process.env.MOM_BENCHMARK_CONTROL_STDIN !== "1") {return;}
  let input = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", function(chunk) {
    input += chunk;
    let eol;
    while ((eol = input.indexOf("\n")) !== -1) {
      const command = input.slice(0, eol).trim();
      input = input.slice(eol + 1);
      if (command === "close") {on_exit();}
    }
  });
  process.stdin.resume();
}

const environment = require("./miner/environment")({
  fs, os, process, o, opt: global.opt, compilerPolicy, gpuTuning, normalizeAlgoName,
  requestedJobBackend, jobBackend, resolvedDeviceList, configuredTuning,
});
const {
  detect_cpu, use_msr_tuning, add_algo_params, publicAlgoParams,
  use_algo_param_benchmarks, prepare_fixed_algo_params,
} = environment;

function start_after_algo_params() {
  if (use_algo_param_benchmarks()) {return bench_algos(start_mining);}
  prepare_fixed_algo_params();
  return start_mining();
}

function createComputeCore() {
  if (compute_core) {return compute_core;}
  // The control core performs device discovery/MSR work before an algorithm worker exists. In a
  // multi-compiler source tree the last build artifact is not necessarily runnable with the DLLs
  // on the ambient PATH, so load the selected GPU policy's default worker for this probe. A real
  // algorithm name is deliberately not used: overrides belong only to hashing workers.
  Object.assign(process.env, compilerPolicy.workerEnv("__control__"));
  compute_core = h.create_core();
  compute_core.from.on("close", () => { process.exitCode = 0; });
  return compute_core;
}

function readMsrThen(on_read, on_error) {
  if (!use_msr_tuning()) {return on_error();}
  compute_core.from.on("read_msr", on_read);
  compute_core.from.on("error", function(v) {
    if (v) {h.log("Can't access MSR: " + JSON.stringify(v.message));}
    return on_error(v);
  });
  compute_core.emit_to("read_msr", h.pack_msr(global.opt.default_msrs));
}

function startBenchJob() {
  h.messageWorkers({type: "bench", job: last_job = prepareBenchmarkJob(global.opt.job)});
}

function resolveDirectJobTuning(params) {
  const algo = normalizeAlgoName(global.opt.job.algo);
  const heuristicDev = params[algo];
  if (!heuristicDev) {
    err_exit(`No automatic GPU tuning is available for ${algo} on the selected device`);
    return false;
  }
  global.opt.job.dev = resolvedDeviceList(
    algo, global.opt.job.dev, heuristicDev, configuredTuning(algo));
  return true;
}

function startWithDirectJobTuning(start) {
  const algo = normalizeAlgoName(global.opt.job.algo);
  if (!gpuTuning.needsPrimaryTuning(global.opt.job.dev, algo)) {return start();}
  createComputeCore();
  const onError = function(value) {
    err_exit("Can't derive automatic GPU tuning: " +
      JSON.stringify(value && value.message ? value.message : value));
  };
  compute_core.from.once("algo_params", function(params) {
    compute_core.from.removeListener("error", onError);
    if (resolveDirectJobTuning(params)) {start();}
  });
  compute_core.from.once("error", onError);
  compute_core.emit_to("algo_params", detect_cpu());
}

function startTestJob() {
  global.opt.job.backend_request = compilerPolicy.validateBackend(global.opt.job.backend || "auto");
  global.opt.job.backend = global.opt.job.backend_request !== "auto"
    ? compilerPolicy.validateBackend(global.opt.job.backend)
    : jobBackend(normalizeAlgoName(global.opt.job.algo));
  if (normalizeAlgoName(global.opt.job.algo) === "pearlhash") {
    addPearlHashJobFields(global.opt.job);
  }
  h.recreate_threads(global.opt.job.dev, messageHandler,
    (entry) => workerRuntimeEnv(normalizeAlgoName(global.opt.job.algo), entry));
  h.messageWorkers({type: "test", job: global.opt.job});
}

function startDirectBenchmark() {
  h.recreate_threads(global.opt.job.dev, messageHandler,
    (entry) => workerRuntimeEnv(normalizeAlgoName(global.opt.job.algo), entry));
  if (!use_msr_tuning()) {
    startBenchJob();
    return;
  }
  createComputeCore();
  readMsrThen(function(v) {
    global.opt.default_msrs = h.unpack_msr(v); // to restore them on exit
    set_algo_msr(global.opt.job.algo);
    startBenchJob();
  }, startBenchJob);
}

switch (directive) {
  case "mine":
    install_exit_handlers();
    createComputeCore();
    compute_core.from.on("algo_params", function(v) {
      add_algo_params(v);
      readMsrThen(function(v) {
        global.opt.default_msrs = h.unpack_msr(v);
        start_after_algo_params();
      }, function() {
        global.opt.default_msrs = {};
        start_after_algo_params();
      });
    });
    compute_core.emit_to("algo_params", detect_cpu());
    break;

  case "test":
    startWithDirectJobTuning(startTestJob);
    break;

  case "bench":
    install_exit_handlers();
    // benchmark-gpu-algos.js uses this small control pipe because Windows emulates SIGTERM by
    // abruptly killing only the parent process. An explicit close drains every compute worker and
    // lets N-API/SYCL environment cleanup hooks run before the benchmark process exits.
    install_benchmark_control();
    startWithDirectJobTuning(startDirectBenchmark);
    break;

  case "algo_params":
    createComputeCore();
    compute_core.from.on("algo_params", function(v) {
      fs.writeSync(1, "MOM_ALGO_PARAMS " + JSON.stringify(publicAlgoParams(v)) + "\n");
      exit(0);
    });
    compute_core.from.on("error", function(v) {
      err_exit("Can't detect algo params: " + JSON.stringify(v.message ? v.message : v));
    });
    compute_core.emit_to("algo_params", detect_cpu());
    break;
  default:
    break;
}
