// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

"use strict";

const path = require("path");
const fs   = require('fs');
const os   = require("os");
const h    = require("./helper.js");
const o    = require("./opts.js");
const p    = require("./pool.js");

// compute core wrapper for cluster process fork
if (h.cluster_process()) return;

global.opt = {};

let compute_core = null;
let algo_params_bench_cb = null; // used to record algo_params bench data
let last_job = null;
let directive = null;
let test = {
  result_hash_hex: null,
  thread_tested:   0,
  result:          ""
};
let thread_hashrates = {};
let is_exiting = false;

const WORKER_CLOSE_GRACE_MS = 3000;
const PROCESS_EXIT_GRACE_MS = 5000;

o.set_default_opts(global.opt, o.opt_help);

function orDefault(value, fallback) {
  return value ? value : fallback;
}

function orDefaultFn(value, fallback) {
  return value ? value : fallback();
}

function firstTruthyOr(fallback, ...values) {
  const found = values.find(Boolean);
  return found ? found : fallback;
}

function reallyExit(code) {
  const finish = () => {
    if (h.exit_now) h.exit_now(code);
    else process.exit(code);
  };

  setImmediate(() => {
    process.stdout.write("", () => {
      process.stderr.write("", finish);
    });
  });
}

function normalizeTestResult(algo, value) {
  if (!algo.includes("c29")) return value;

  const tokens = value.trim().split(/\s+/);
  const hasEol = tokens[tokens.length - 1] === "EOL";
  if (hasEol) tokens.pop();

  return tokens.sort().join(" ") + (hasEol ? " EOL" : "");
}

function normalizeExpectedResults(algo, value) {
  return value.split("|").map((expected) => normalizeTestResult(algo, expected));
}

function forceExitByDefault() {
  return directive === "mine" || directive === "bench";
}

function closeComputeCore() {
  if (!compute_core) return;
  if (Object.keys(global.opt.default_msrs).length)
    compute_core.emit_to("write_msr", h.pack_msr(global.opt.default_msrs));
  compute_core.emit_to("close");
  compute_core = null;
}

function shouldExitImmediately() {
  return directive === "test" || directive === "algo_params";
}

function scheduleForcedExit(code) {
  setTimeout(function() {
    reallyExit(code);
  }, PROCESS_EXIT_GRACE_MS).unref();
}

function finishAlreadyExiting(code, force) {
  if (force) reallyExit(code);
  return false;
}

function workerCloseGrace(force) {
  return force ? WORKER_CLOSE_GRACE_MS : null;
}

function finishExit(code, force) {
  if (shouldExitImmediately()) reallyExit(code);
  else if (force) scheduleForcedExit(code);
  return false;
}

function exit(code, force = forceExitByDefault()) {
  if (is_exiting) return finishAlreadyExiting(code, force);
  is_exiting = true;
  closeComputeCore();
  h.closeWorkers(workerCloseGrace(force));
  process.exitCode = code;
  return finishExit(code, force);
}

function err_exit(msg) {
  h.log_err(msg);
  return exit(1);
}

function mergeConfigOptions(opt2) {
  for (const key in opt2) switch (key) {
    case "job": case "pool_time": // do not overwrite these option sets completely
      mergeNestedConfigOption(key, opt2[key]);
      break;
    default: global.opt[key] = opt2[key];
  }
}

function mergeNestedConfigOption(key, values) {
  for (const key2 in values) global.opt[key][key2] = values[key2];
}

function loadConfigFile(config_file) {
  const config_fn = path.resolve(config_file);
  h.log("Loading config file " + config_fn);
  mergeConfigOptions(require(config_fn));
}

function parsePoolUri(pool_uri) {
  const pool_uri_parts = pool_uri.split(":");
  if (pool_uri_parts.length !== 2) return o.print_help("Wrong pool URI: " + pool_uri);
  const parsed_port = parsePoolPort(pool_uri_parts[1]);
  if (!parsed_port) return o.print_help("Wrong pool port: " + pool_uri_parts[1]);
  return { url: pool_uri_parts[0], port: parsed_port.port, is_tls: parsed_port.is_tls };
}

function parsePoolPort(port_tls) {
  const m = port_tls.match(/^(\d+)((?:tls)?)$/);
  if (!m) return o.print_help("Wrong pool port: " + port_tls);
  const port = Number(m[1]);
  if (port < 1 || port > 65535) return o.print_help("Wrong pool port: " + port_tls);
  return { port, is_tls: m[2] === "tls" };
}

function addPrimaryPool(pool_uri, pool_login, pool_pass) {
  const pool = parsePoolUri(pool_uri);
  global.opt.pool_ids.primary = global.opt.pools.length;
  global.opt.pools.push(o.pool_create(pool.url, pool.port, pool.is_tls, pool_login, pool_pass));
}

function optionalPoolPass(args) {
  return args.length > 0 && !args[0].match(/^--/) ? args.shift() : "";
}

function parseMineArgs(args) {
  if (args.length < 1) return o.print_help("Directive \"mine\" needs 1+ parameters");
  const param1 = args.shift();
  if (o.is_config_file(param1)) return loadConfigFile(param1); // load config file

  // setup primary pool
  if (args.length < 1) return o.print_help("Directive \"mine\" needs 2+ parameters");
  const pool_login = args.shift();
  const pool_pass  = optionalPoolPass(args);
  return addPrimaryPool(param1, pool_login, pool_pass);
}

function parseRemainingOptions(args) {
  while (args.length) {
    const arg = args.shift();
    if (args.length >= 1 && o.parse_opt(global.opt, o.opt_help, arg, args[0], "")) args.shift();
    else return o.print_help("Unparsed option: " + arg);
  }
}

function parse_args() {
  const args = process.argv.slice(2);

  if (args.length === 0) return o.print_help("No directive specified");
  directive = args.shift();
  const parser = directiveParsers[directive];
  if (!parser) return o.print_help("Unknown directive " + directive);
  parser(args);

  parseRemainingOptions(args);
  return true;
}

function hexWithoutPrefix(value) {
  return String(value || "").replace(/^0x/i, "");
}

function normalizedFullNonce(value) {
  return hexWithoutPrefix(value).padStart(16, "0").slice(-16);
}

function ergSubmitMeta(pool, job_id) {
  return (pool.erg_submit_jobs && pool.erg_submit_jobs[job_id]) || {};
}

function ergExtraNonce2Size(pool, meta) {
  const size = Number(meta.extra_nonce2_size);
  if (Number.isInteger(size) && size >= 0 && size <= 8) return size;

  const extraNonce = hexWithoutPrefix(meta.extra_nonce || pool.extra_nonce || "");
  return Math.max(0, 8 - Math.ceil(extraNonce.length / 2));
}

function ergSubmitParams(pool, value) {
  const meta = ergSubmitMeta(pool, value.job_id);
  const nonce = normalizedFullNonce(value.nonce);
  const extraNonce2HexLength = ergExtraNonce2Size(pool, meta) * 2;
  const extraNonce2 = extraNonce2HexLength ? nonce.slice(16 - extraNonce2HexLength) : "";
  return [pool.login, value.job_id, extraNonce2, hexWithoutPrefix(meta.ntime), nonce];
}

function normalizeAlgoName(algo) {
  return algo && (algo.startsWith("c29") || algo === "cuckaroo") ? "c29" : algo;
}

function parseTestArgs(args) {
  if (args.length < 2) return o.print_help("Directive \"test\" needs two parameters");
  global.opt.job.algo = args.shift();
  test.result_hash_hex = args.shift();
}

function parseBenchArgs(args) {
  if (args.length < 1) return o.print_help("Directive \"bench\" needs one paramater");
  global.opt.job.algo = args.shift();
}

const directiveParsers = {
  mine: parseMineArgs,
  test: parseTestArgs,
  bench: parseBenchArgs,
  algo_params: function() {},
};

if (!parse_args()) return;

function handleResult(msg) {
  const params = {
    id: msg.value.worker_id, job_id: msg.value.job_id,
    nonce: msg.value.nonce, result: msg.value.hash
  };
  const pool = global.opt.pools[msg.value.pool_id];
  if (pool && pool.submit_mode === "erg") {
    return p.pool_write(msg.value.pool_id, {
      jsonrpc: "2.0",
      id: 3,
      method: "mining.submit",
      params: ergSubmitParams(pool, msg.value),
    });
  }
  if (msg.value.mix_hash) {
    const headerHash = resultHeaderHash(msg, pool);
    if (pool && pool.submit_mode === "ethproxy") {
      return p.pool_write(msg.value.pool_id, {
        jsonrpc: "2.0",
        id: 3,
        method: "eth_submitWork",
        params: ["0x" + msg.value.nonce, "0x" + headerHash.slice(0, 64), "0x" + msg.value.mix_hash],
      });
    }
    if (pool && (pool.submit_mode === "raven" || pool.submit_mode === "eth")) {
      return p.pool_write(msg.value.pool_id, {
        jsonrpc: "2.0",
        id: 3,
        method: "mining.submit",
        params: [pool.login, msg.value.job_id, "0x" + msg.value.nonce, "0x" + headerHash.slice(0, 64), "0x" + msg.value.mix_hash],
      });
    }
    params.mixhash = msg.value.mix_hash;
    if (headerHash) params.header_hash = headerHash.slice(0, 64);
  }
  if (msg.value.commitment) params.commitment = msg.value.commitment;
  if (msg.value.edges) {
    params.pow = h.edge_hex2arr(msg.value.edges);
    // for proofsize == 42 (Tari C29) we return nonce hex as usual
    if (params.pow.length !== 42) params.nonce = Number.parseInt(params.nonce, 16);
  }
  p.pool_write(msg.value.pool_id, { jsonrpc: "2.0", id: 3, method: "submit", params: params });
}

function resultHeaderHash(msg, pool) {
  if (msg.value.header_hash) return msg.value.header_hash;
  if (!pool || !pool.last_job) return "";
  if (pool.last_job.job_id && msg.value.job_id && pool.last_job.job_id !== msg.value.job_id) return "";
  return firstTruthyOr("", pool.last_job.header_hash, pool.last_job.blob, pool.last_job.blob_hex);
}

// store max last nonce for background pool job to resume it from there
function handleLastNonce(msg) {
  const pool_id = msg.value.pool_id;
  // pool_id can be "" for benchmark jobs. can not use === here since
  // global.opt.pool_ids.active is integer here
  if (!shouldStoreLastNonce(pool_id)) return;
  const prev_nonce = global.opt.pools[pool_id].last_job.nonce;
  const new_nonce  = msg.value.nonce;
  if (isNewerNonce(prev_nonce, new_nonce))
    global.opt.pools[pool_id].last_job.nonce = new_nonce;
}

function shouldStoreLastNonce(pool_id) {
  return pool_id !== "" && pool_id != global.opt.pool_ids.active &&
         global.opt.pools[pool_id].last_job;
}

function isNewerNonce(prev_nonce, new_nonce) {
  return !prev_nonce || BigInt("0x" + prev_nonce) < BigInt("0x" + new_nonce);
}

function isRandomXAlgo(algo) {
  return algo.startsWith("rx/") || algo === "panthera";
}

function expectedTestThreads(msg) {
  const threads = h.get_dev_threads(global.opt.job.dev);
  if (isRandomXAlgo(global.opt.job.algo)) {
    const batch = h.get_dev_batch(h.get_thread_dev(msg.thread_id, global.opt.job.dev));
    return batch * threads;
  }
  return global.opt.job.algo.includes("c29") ? test.result_hash_hex.trim().split(/\s+/).length : threads;
}

function handleTestResult(msg) {
  const test_threads = expectedTestThreads(msg);
  test.result = (test.result ? test.result + " " : "") + msg.value.result;
  if (++test.thread_tested < test_threads) return;

  const expectedResults = normalizeExpectedResults(global.opt.job.algo, test.result_hash_hex);
  const actualResult = normalizeTestResult(global.opt.job.algo, test.result);
  if (!expectedResults.includes(actualResult)) {
    fs.writeSync(2, "FAILED: " + test.result + " != " + test.result_hash_hex + " " + test_threads + "\n");
    return exit(1);
  }
  fs.writeSync(1, "PASSED\n");
  return exit(0);
}

function collectedHashrate() {
  const rates = Object.values(thread_hashrates).map(Number.parseFloat);
  const total_hashrate = rates.reduce((total, rate) => total + rate, 0);
  const thread_hashrate_str = rates.map(h.formatHashrate).join(", ");
  return { total_hashrate, thread_hashrate_str };
}

function handleHashrate(msg) {
  thread_hashrates[msg.thread_id] = msg.value.hashrate;
  if (Object.keys(thread_hashrates).length < h.get_dev_threads(last_job.dev)) return;

  const hashrate = collectedHashrate();
  h.log("Algo " + last_job.algo + " (" + last_job.dev + ") hashrate: " +
        h.formatHashrate(hashrate.total_hashrate) + " (" + hashrate.thread_hashrate_str + ")");
  thread_hashrates = {};
  if (algo_params_bench_cb) return algo_params_bench_cb(hashrate.total_hashrate);
}

function handleWorkerError(msg) {
  if (msg.value.message === "Ignore duplicate job") return;
  h.log_err("Compute core error: " + JSON.stringify(msg.value));
  if (test.result_hash_hex) exit(1); // exit with error
}

// handles messages sent to the master thread from worker threads
function messageHandler(msg) {
  const handler = masterMessageHandlers[msg.type];
  if (handler) return handler(msg);
  return h.log_err("Unknown master thread message: " + JSON.stringify(msg));
}

const masterMessageHandlers = {
  result:     handleResult,
  last_nonce: handleLastNonce,
  test:       handleTestResult,
  hashrate:   handleHashrate,
  error:      handleWorkerError,
};

function set_algo_msr(algo) {
  if (Object.keys(global.opt.default_msrs).length && compute_core) {
    let default_msr = h.pack_msr(global.opt.default_msrs);
    default_msr.algo = algo;
    compute_core.emit_to("write_msr", default_msr);
  }
}

function normalizedAlgo(prev_job) {
  const algo = prev_job.algo ? prev_job.algo : global.opt.job.algo;
  return normalizeAlgoName(algo);
}

function jobDev(algo) {
  const algo_param = global.opt.algo_params[algo];
  return algo_param && algo_param.dev ? algo_param.dev : global.opt.job.dev;
}

function baseJob(prev_job, algo, dev, pool_id) {
  return {
    algo:       algo,
    dev:        dev,
    seed_hex:   orDefault(prev_job.seed_hash, prev_job.seed_hex),
    target:     jobTarget(prev_job, algo),
    worker_id:  firstTruthyOr(global.opt.pools[pool_id].worker_id || global.opt.pools[pool_id].login,
                              prev_job.id, prev_job.worker_id),
    job_id:     orDefault(prev_job.job_id, ""),
    header_hash: orDefault(prev_job.header_hash, ""),
    nonce:      orDefault(prev_job.nonce, 0),
    height:     orDefault(prev_job.height, 0),
    thread_num: h.get_dev_threads(dev),
    pool_id:    pool_id,
  };
}

const nonceAt32Algos = new Set(["kawpow", "etchash", "autolykos2"]);
const currentEtcBenchHeight = 24689903; // Sampled from ETC mainnet on 2026-06-04 for live-size Etchash benchmarks.
const defaultBenchAlgos = new Set([
  "autolykos2",
  "c29",
  "cn/gpu",
  "etchash",
  "ghostrider",
  "kawpow",
  "panthera",
  "rx/0",
  "rx/2",
  "rx/arq",
]);

function isNonceAt32Algo(algo) {
  return nonceAt32Algos.has(algo);
}

function jobTarget(prev_job, algo) {
  const explicitTarget = orDefault(prev_job.target, "");
  if (!isNonceAt32Algo(algo)) return explicitTarget || h.diff2target(prev_job.difficulty);
  if (explicitTarget && !/^\d+$/.test(explicitTarget) && explicitTarget.replace(/^0x/i, "").length > 16)
    return explicitTarget.replace(/^0x/i, "").padStart(64, "0");
  if (algo === "autolykos2" && /^\d+$/.test(explicitTarget)) return h.decimalTargetToHex(explicitTarget);
  return h.ethDiff2Target(prev_job.difficulty || (explicitTarget ? h.target2diff(explicitTarget) : 1));
}

function addC29JobFields(job, prev_job) {
  job.proofsize = orDefault(prev_job.proofsize, 42);
  if (prev_job.pre_pow) { // GRIN
    job.noncebytes  = orDefault(prev_job.noncebytes, 4);
    job.blob_hex    = prev_job.pre_pow + "00".repeat(job.noncebytes);
    job.nonceoffset = prev_job.pre_pow.length / 2;
  } else if (prev_job.blob) { // TARI C29
    job.noncebytes  = orDefault(prev_job.noncebytes, 8);
    job.blob_hex    = "00".repeat(job.noncebytes) + prev_job.blob;
    job.nonceoffset = 0;
  } else {
    job.noncebytes  = prev_job.noncebytes;
    job.blob_hex    = prev_job.blob_hex;
    job.nonceoffset = prev_job.nonceoffset;
  }
}

function addEthHashJobFields(job, prev_job) {
  job.noncebytes = orDefault(prev_job.noncebytes, 8);
  job.nonceoffset = typeof prev_job.nonceoffset !== "undefined" ? prev_job.nonceoffset : 32;

  const blob = orDefault(prev_job.blob, prev_job.blob_hex);
  job.blob_hex = blob && blob.length === 64 ? blob + "0000000000000000" : blob;
}

function addStandardJobFields(job, prev_job) {
  job.noncebytes  = orDefault(prev_job.noncebytes, 4);
  job.blob_hex    = orDefault(prev_job.blob, prev_job.blob_hex);
  job.nonceoffset = typeof prev_job.nonceoffset !== "undefined" ?
                    prev_job.nonceoffset : (job.algo === "ghostrider" ? 76 : 39);
}

function addNoncePrefix(job, prev_job) {
  // we need to create nonce with xn prefix and update nicehash_mask to cover it
  const nicehash_prefix = Buffer.from(prev_job.xn, "hex").subarray(0, job.noncebytes);
  job.nicehash_mask = Buffer.alloc(job.noncebytes, 0).fill(0xFF, 0, nicehash_prefix.length).toString("hex");
  job.nonce = Buffer.concat([nicehash_prefix, Buffer.alloc(job.noncebytes - nicehash_prefix.length, 0x00)]).toString("hex");
}

function defaultNicehashMask(job, pool_id, last_job_can_be_used) {
  if (last_job_can_be_used && last_job.nicehash_mask) return last_job.nicehash_mask;
  return Buffer.alloc(job.noncebytes, 0)
    .fill(0xFF, 0, global.opt.pools[pool_id].is_nicehash ? 1 : 0)
    .toString("hex");
}

function addNonceFields(job, prev_job, pool_id) {
  if (prev_job.xn) return addNoncePrefix(job, prev_job);

  const last_job_can_be_used = last_job && last_job.algo === job.algo;
  // use existing nicehash_mask or make a new one with FF00..00 that job.noncebytes long
  job.nicehash_mask = orDefault(prev_job.nicehash_mask,
                                defaultNicehashMask(job, pool_id, last_job_can_be_used));
  job.nonce = orDefault(prev_job.nonce, reusableLastNonce(last_job_can_be_used));
}

function reusableLastNonce(last_job_can_be_used) {
  return last_job_can_be_used && last_job.nonce ? last_job.nonce : "0";
}

function workerRuntimeEnv(algo) {
  if (algo !== "c29") return {};

  // C29 submits hundreds of short SYCL kernels per second; legacy non-immediate
  // Level Zero command lists avoid the one-core immediate-list path on Intel GPUs.
  return {
    SYCL_UR_USE_LEVEL_ZERO_V2: "0",
    SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS: "0",
    MOMINER_C29_SEED_BLOCKS: process.env.MOMINER_C29_SEED_BLOCKS || "16",
  };
}

function ensureWorkersForJob(algo, dev) {
  if (!last_job || last_job.algo !== algo || last_job.dev !== dev)
    h.recreate_threads(dev, messageHandler, workerRuntimeEnv(algo));
}

// prev_job can be either job json from the pool or
// previous job restored from the pool switch (with nonce that we need to take into account)
function set_job(prev_job) {
  const algo = normalizedAlgo(prev_job);
  const dev = jobDev(algo);
  ensureWorkersForJob(algo, dev);
  const pool_id = global.opt.pool_ids.active;
  const job = baseJob(prev_job, algo, dev, pool_id);
  if (algo === "c29") addC29JobFields(job, prev_job);
  else if (isNonceAt32Algo(algo)) addEthHashJobFields(job, prev_job);
  else addStandardJobFields(job, prev_job);
  addNonceFields(job, prev_job, pool_id);
  set_algo_msr(algo);
  h.messageWorkers({type: "job", job: last_job = job});
  return job;
}

function prepareBenchmarkJob(job) {
  if (isNonceAt32Algo(job.algo)) {
    job.noncebytes = 8;
    job.nonceoffset = 32;
    if (job.blob_hex && job.blob_hex.length === 64) job.blob_hex += "0000000000000000";
  }
  if (job.algo === "etchash") {
    job.height = job.height || currentEtcBenchHeight;
    job.seed_hex = "";
  }
  return job;
}

function bench_algo(algo, cb) {
  const job = prepareBenchmarkJob({
    algo:     algo,
    dev:      global.opt.algo_params[algo].dev,
    blob_hex: global.opt.job.blob_hex,
    seed_hex: global.opt.job.seed_hex,
    pool_id:  "", // to drop last nonce messages from this job
  });
  h.recreate_threads(job.dev, messageHandler, workerRuntimeEnv(algo));
  let timeout = setTimeout(function() {
    h.log_err("Benchmark " + algo + " algo (" + job.dev + ") timeout");
    return cb(0);
  }, 2*60*1000);
  algo_params_bench_cb = function(hashrate) { clearTimeout(timeout); return cb(hashrate) };
  set_algo_msr(algo);
  h.messageWorkers({type: "bench", job: last_job = job});
}

// do global.opt.algo_params benchmarks if perf === null
function bench_algos(cb) {
  let algos = benchmarkAlgos();
  let is_before_first_benchmark = true;
  h.repeat(function(cb_next) {
    const algo = nextAlgoToBenchmark(algos);
    if (!algo) return cb();
    if (is_before_first_benchmark) h.log("Doing algo benchmarks...");
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
  if (global.opt.bench_algo_params === 2) return algos;
  return algos.filter((algo) => defaultBenchAlgos.has(algo));
}

function nextAlgoToBenchmark(algos) {
  let algo;
  // skip until next algo with null perf
  while ((algo = algos.shift()) && global.opt.algo_params[algo].perf !== null);
  return algo;
}

function saveConfig() {
  if (global.opt.save_config) {
    const save_config = global.opt.save_config;
    delete global.opt.save_config; // by default do not save config again when it will be loaded
    h.log("Saving config file to " + save_config);
    fs.writeFile(save_config, JSON.stringify(o.saved_config(global.opt), null, 2), function(err) {
      if (err) h.log_err("Error saving " + save_config + " file");
    });
  }
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
    setInterval(function() {
      switch (global.opt.pool_ids.active) {
        case global.opt.pool_ids.primary:
        case global.opt.pool_ids.donate: return;
      }
      p.connect_pool_throttle(global.opt.pool_ids.primary, set_job);
    }, global.opt.pool_time.primary_reconnect * 1000);
}

// donation mining
function scheduleDonationMining() {
  if (global.opt.pool_ids.donate !== null) setInterval(function() {
    p.connect_pool_throttle(global.opt.pool_ids.donate, set_job);
    setTimeout(p.switch_pool, global.opt.pool_time.donate_length * 1000,
               global.opt.pool_ids.donate, set_job);
  }, global.opt.pool_time.donate_interval * 1000);
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
  if (process.platform === "win32") process.on("SIGBREAK", on_exit);
  else process.on("SIGHUP", on_exit);
}

function fallbackCpuInfo() {
  return {
    cpu_sockets: 1,
    cpu_threads: os.cpus().length || 1,
    cpu_l3cache: 0,
  };
}

function cpuSocketCount(cpuinfo) {
  const physical_ids = new Set();
  for (const match of cpuinfo.matchAll(/^physical id\s*:\s*(.+)$/gm)) physical_ids.add(match[1]);
  return physical_ids.size || 1;
}

function cacheSizeBytes(size_text) {
  const size = size_text.match(/^(\d+)([KMG])$/i);
  if (!size) return 0;
  const multiplier = { K: 1024, M: 1024 * 1024, G: 1024 * 1024 * 1024 }[size[2].toUpperCase()];
  return Number(size[1]) * multiplier;
}

function cacheSharedId(base) {
  const shared_cpu_list = `${base}/shared_cpu_list`;
  return fs.existsSync(shared_cpu_list) ? fs.readFileSync(shared_cpu_list, "utf8").trim() : base;
}

function l3CacheEntryBytes(base, l3_ids) {
  try {
    if (!isUnifiedL3Cache(base)) return 0;
    const id = cacheSharedId(base);
    if (l3_ids.has(id)) return 0;
    l3_ids.add(id);
    return cacheSizeBytes(fs.readFileSync(`${base}/size`, "utf8").trim());
  } catch (_) {
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
    if (!fs.existsSync(cache_dir)) continue;
    for (const entry of fs.readdirSync(cache_dir)) l3cache += l3CacheEntryBytes(`${cache_dir}/${entry}`, l3_ids);
  }
  return l3cache;
}

function detect_cpu() {
  const fallback = fallbackCpuInfo();
  if (!hasProcCpuInfo()) return fallback;

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
  return process.platform !== "win32";
}

function add_algo_params(params) {
  for (const algo in params) {
    if (!(algo in global.opt.algo_params))
      global.opt.algo_params[algo] = { dev: params[algo], perf: null };
  }
}

function use_algo_param_benchmarks() {
  return global.opt.bench_algo_params !== 0;
}

function prepare_fixed_algo_params() {
  const algo = normalizeAlgoName(global.opt.job.algo);
  if (!algo) return;
  const algo_param = global.opt.algo_params[algo] || { dev: global.opt.job.dev, perf: null };
  global.opt.job.algo = algo;
  global.opt.algo_params = { [algo]: algo_param };
}

function start_after_algo_params() {
  if (use_algo_param_benchmarks()) return bench_algos(start_mining);
  prepare_fixed_algo_params();
  return start_mining();
}

function onComputeCoreClose() {
  process.exitCode = 0;
}

function createComputeCore() {
  compute_core = h.create_core();
  compute_core.from.on("close", onComputeCoreClose);
  return compute_core;
}

function logMsrAccessError(v) {
  if (v) h.log("Can't access MSR: " + JSON.stringify(v.message));
}

function readMsrThen(on_read, on_error) {
  if (!use_msr_tuning()) return on_error();
  compute_core.from.on("read_msr", on_read);
  compute_core.from.on("error", function(v) {
    logMsrAccessError(v);
    return on_error(v);
  });
  compute_core.emit_to("read_msr", h.pack_msr(global.opt.default_msrs));
}

function startBenchJob() {
  h.messageWorkers({type: "bench", job: last_job = prepareBenchmarkJob(global.opt.job)});
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
    h.recreate_threads(global.opt.job.dev, messageHandler, workerRuntimeEnv(normalizeAlgoName(global.opt.job.algo)));
    h.messageWorkers({type: "test", job: global.opt.job});
    break;

  case "bench":
    install_exit_handlers();
    h.recreate_threads(global.opt.job.dev, messageHandler, workerRuntimeEnv(normalizeAlgoName(global.opt.job.algo)));
    if (!use_msr_tuning()) {
      startBenchJob();
      break;
    }
    createComputeCore();
    readMsrThen(function(v) {
      global.opt.default_msrs = h.unpack_msr(v); // to restore them on exit
      set_algo_msr(global.opt.job.algo);
      startBenchJob();
    }, startBenchJob);
    break;

  case "algo_params":
    createComputeCore();
    compute_core.from.on("algo_params", function(v) {
      fs.writeSync(1, "MOMINER_ALGO_PARAMS " + JSON.stringify(v) + "\n");
      exit(0);
    });
    compute_core.from.on("error", function(v) {
      err_exit("Can't detect algo params: " + JSON.stringify(v.message ? v.message : v));
    });
    compute_core.emit_to("algo_params", detect_cpu());
    break;
}
