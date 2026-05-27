// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

"use strict";

const path = require("path");
const net  = require("net");
const tls  = require("tls");
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

function exit(code, force = directive === "mine" || directive === "bench") {
  if (is_exiting) {
    if (force) reallyExit(code);
    return false;
  }
  is_exiting = true;
  if (compute_core) {
    if (Object.keys(global.opt.default_msrs).length)
      compute_core.emit_to("write_msr", h.pack_msr(global.opt.default_msrs));
    compute_core.emit_to("close");
    compute_core = null;
  }
  h.closeWorkers(force ? WORKER_CLOSE_GRACE_MS : null);
  process.exitCode = code;
  if (directive === "test" || directive === "algo_params") {
    reallyExit(code);
  } else if (force) {
    setTimeout(function() {
      reallyExit(code);
    }, PROCESS_EXIT_GRACE_MS).unref();
  }
  return false;
}

function err_exit(msg) {
  h.log_err(msg);
  return exit(1);
}

function parse_args() {
  let args = process.argv.slice(2);

  if (args.length === 0) return o.print_help("No directive specified");
  directive = args.shift();
  switch (directive) {
   case "mine":
      if (args.length < 1) return o.print_help("Directive \"mine\" needs 1+ parameters");
      const param1 = args.shift();
      if (o.is_config_file(param1)) { // load config file
        const config_fn = path.resolve(param1);
        h.log("Loading config file " + config_fn);
        const opt2 = require(config_fn);
        for (const key in opt2) switch (key) {
          case "job": case "pool_time": // do not overwrite these option sets completely
            for (const key2 in opt2[key]) global.opt[key][key2] = opt2[key][key2];
            break;
          default: global.opt[key] = opt2[key];
        }
      } else { // setup primary pool
        if (args.length < 1) return o.print_help("Directive \"mine\" needs 2+ parameters");
        const pool_uri   = param1;
        const pool_login = args.shift();
        const pool_pass  = args.length > 0 && !args[0].match(/^--/) ? args.shift() : "";
        const pool_uri_parts = pool_uri.split(":");
        if (pool_uri_parts.length !== 2) return o.print_help("Wrong pool URI: " + pool_uri);
        const port_tls = pool_uri_parts[1];
        const m = port_tls.match(/^(\d+)((?:tls)?)$/);
        if (!m) return o.print_help("Wrong pool port: " + port_tls);
        global.opt.pool_ids.primary = global.opt.pools.length;
        global.opt.pools.push(o.pool_create(pool_uri_parts[0], parseInt(m[1]), m[2] === "tls",
                                            pool_login, pool_pass));
      }
      break;

    case "test":
      if (args.length < 2) return o.print_help("Directive \"test\" needs two parameters");
      global.opt.job.algo = args.shift();
      test.result_hash_hex = args.shift();
      break;

    case "bench":
      if (args.length < 1) return o.print_help("Directive \"bench\" needs one paramater");
      global.opt.job.algo = args.shift();
      break;

    case "algo_params":
      break;

    default: return o.print_help("Unknown directive " + directive);
  }

  while (args.length) {
    let arg_parsed = false;
    const arg = args.shift();
    if (args.length >= 1 && o.parse_opt(global.opt, o.opt_help, arg, args[0], "")) args.shift();
    else return o.print_help("Unparsed option: " + arg);
  }

  return true;
}

if (!parse_args()) return;

// handles messages sent to the master thread from worker threads
function messageHandler(msg) {
  switch (msg.type) {
    case "result":
      let params = {
        id: msg.value.worker_id, job_id: msg.value.job_id,
        nonce: msg.value.nonce, result: msg.value.hash
      };
      if (msg.value.commitment) params.commitment = msg.value.commitment;
      if (msg.value.edges) {
        params.pow = h.edge_hex2arr(msg.value.edges);
	// for proofsize == 42 (Tari C29) we return nonce hex as usual
	if (params.pow.length != 42) params.nonce = parseInt(params.nonce, 16);
      }
      p.pool_write(msg.value.pool_id, { jsonrpc: "2.0", id: 3, method: "submit", params: params });
      break;

    case "last_nonce": // store max last nonce for background pool job to resume it from there
      const pool_id = msg.value.pool_id;
      // pool_id can be "" for benchmark jobs. can not use === here since
      // global.opt.pool_ids.active is integer here
      if (pool_id === "" || pool_id == global.opt.pool_ids.active ||
          !global.opt.pools[pool_id].last_job) break;
      const prev_nonce = global.opt.pools[pool_id].last_job.nonce;
      const new_nonce  = msg.value.nonce;
      if (!prev_nonce || BigInt("0x" + prev_nonce) < BigInt("0x" + new_nonce))
        global.opt.pools[pool_id].last_job.nonce = new_nonce;
      break;

    case "test":
      const is_rx = global.opt.job.algo.includes("rx/");
      const batch = h.get_dev_batch(h.get_thread_dev(msg.thread_id, global.opt.job.dev));
      const threads = h.get_dev_threads(global.opt.job.dev);
      const test_threads = is_rx ? batch * threads : (global.opt.job.algo.includes("c29") ? test.result_hash_hex.trim().split(/\s+/).length : threads);
      test.result = (test.result ? test.result + " " : "") + msg.value.result;
      if (++test.thread_tested >= test_threads) {
        const expectedResults = normalizeExpectedResults(global.opt.job.algo, test.result_hash_hex);
        const actualResult = normalizeTestResult(global.opt.job.algo, test.result);
        if (!expectedResults.includes(actualResult)) {
          fs.writeSync(2, "FAILED: " + test.result + " != " + test.result_hash_hex + " " + test_threads + "\n");
          return exit(1);
        } else {
          fs.writeSync(1, "PASSED\n");
          return exit(0);
        }
      }
      break;

    case "hashrate":
      thread_hashrates[msg.thread_id] = msg.value.hashrate;
      if (Object.keys(thread_hashrates).length >= h.get_dev_threads(last_job.dev)) {
        let thread_hashrate_str = "";
        let total_hashrate = 0;
        for (const [thread_id, hashrate] of Object.entries(thread_hashrates)) {
          if (thread_hashrate_str !== "") thread_hashrate_str += ", ";
          const hashrate2 = parseFloat(hashrate);
          thread_hashrate_str += hashrate2.toFixed(2);
          total_hashrate += hashrate2;
        }
        h.log("Algo " + last_job.algo + " (" + last_job.dev + ") hashrate: " +
              total_hashrate.toFixed(2) + " H/s (" + thread_hashrate_str + ")");
        thread_hashrates = {};
        if (algo_params_bench_cb) return algo_params_bench_cb(total_hashrate);
      }
      break;

    case "error":
      if (msg.value.message === "Ignore duplicate job") return;
      h.log_err("Compute core error: " + JSON.stringify(msg.value));
      if (test.result_hash_hex) exit(1); // exit with error
      break;

    default:
      h.log_err("Unknown master thread message: " + JSON.stringify(msg));
  }
}

function set_algo_msr(algo) {
  if (Object.keys(global.opt.default_msrs).length && compute_core) {
    let default_msr = h.pack_msr(global.opt.default_msrs);
    default_msr.algo = algo;
    compute_core.emit_to("write_msr", default_msr);
  }
}

// prev_job can be either job json from the pool or
// previous job restored from the pool switch (with nonce that we need to take into account)
function set_job(prev_job) {
  let algo = prev_job.algo ? prev_job.algo : global.opt.job.algo;
  if (algo.startsWith("c29") || algo === "cuckaroo") algo = "c29";
  const dev = algo in global.opt.algo_params && global.opt.algo_params[algo].dev ?
              global.opt.algo_params[algo].dev : global.opt.job.dev;
  if (!last_job || last_job.algo !== algo || last_job.dev !== dev)
    h.recreate_threads(dev, messageHandler);
  const pool_id = global.opt.pool_ids.active;
  let job = {
    algo:       algo,
    dev:        dev,
    seed_hex:   prev_job.seed_hash ? prev_job.seed_hash : prev_job.seed_hex,
    target:     prev_job.target ? prev_job.target : h.diff2target(prev_job.difficulty),
    worker_id:  prev_job.id     ? prev_job.id     : (prev_job.worker_id ? prev_job.worker_id : global.opt.pools[pool_id].worker_id),
    job_id:     prev_job.job_id ? prev_job.job_id : "",
    nonce:      prev_job.nonce  ? prev_job.nonce  : 0,
    height:     prev_job.height ? prev_job.height : 0,
    thread_num: h.get_dev_threads(dev),
    pool_id:    pool_id,
  };
  if (algo === "c29") {
    job.proofsize     = prev_job.proofsize ? prev_job.proofsize : 42;
    if (prev_job.pre_pow) { // GRIN
      job.noncebytes  = prev_job.noncebytes ? prev_job.noncebytes : 4;
      job.blob_hex    = prev_job.pre_pow + "00".repeat(job.noncebytes);
      job.nonceoffset = prev_job.pre_pow.length / 2;
    } else if (prev_job.blob) { // TARI C29
      job.noncebytes  = prev_job.noncebytes ? prev_job.noncebytes : 8;
      job.blob_hex    = "00".repeat(job.noncebytes) + prev_job.blob;
      job.nonceoffset = 0;
    } else {
      job.noncebytes  = prev_job.noncebytes;
      job.blob_hex    = prev_job.blob_hex;
      job.nonceoffset = prev_job.nonceoffset;
    }
  } else {
    job.noncebytes  = prev_job.noncebytes ? prev_job.noncebytes : 4;
    job.blob_hex    = prev_job.blob ? prev_job.blob : prev_job.blob_hex;
    job.nonceoffset = typeof prev_job.nonceoffset !== "undefined" ?
                      prev_job.nonceoffset : (algo == "ghostrider" ? 76 : 39);
  }

  if (prev_job.xn) { // we need to create nonce with xn prefix and update nicehash_mask to cover it
    const nicehash_bytes = Math.ceil(prev_job.xn.length / 2);
    job.nicehash_mask = Buffer.alloc(job.noncebytes, 0).fill(0xFF, 0, Math.min(nicehash_bytes, job.noncebytes)).toString("hex");
    job.nonce = Buffer.concat([Buffer.from(prev_job.xn, "hex"), Buffer.alloc(job.noncebytes - nicehash_bytes, 0x00)]).toString("hex");
  } else {
    const last_job_can_be_used = last_job && last_job.algo === job.algo;
    // use existing nicehash_mask or make a new one with FF00..00 that job.noncebytes long
    job.nicehash_mask = prev_job.nicehash_mask ? prev_job.nicehash_mask : (
      last_job_can_be_used && last_job.nicehash_mask ? last_job.nicehash_mask :
      Buffer.alloc(job.noncebytes, 0).fill(0xFF, 0, global.opt.pools[pool_id].is_nicehash ? 1 : 0).toString("hex")
    );
    job.nonce = prev_job.nonce ? prev_job.nonce : (last_job_can_be_used && last_job.nonce ? last_job.nonce : "0");
  }
  set_algo_msr(algo);
  h.messageWorkers({type: "job", job: last_job = job});
  return job;
}

function bench_algo(algo, cb) {
  const job = {
    algo:     algo,
    dev:      global.opt.algo_params[algo].dev,
    blob_hex: global.opt.job.blob_hex,
    seed_hex: global.opt.job.seed_hex,
    pool_id:  "", // to drop last nonce messages from this job
  };
  h.recreate_threads(job.dev, messageHandler);
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
  let algos = Object.keys(global.opt.algo_params);
  let is_before_first_benchmark = true;
  h.repeat(function(cb_next) {
    let algo;
    // skip until next algo with null perf
    while ((algo = algos.shift()) && global.opt.algo_params[algo].perf !== null);
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

function start_mining() {
  if (global.opt.save_config) {
    const save_config = global.opt.save_config;
    delete global.opt.save_config; // by default do not save config again when it will be loaded
    h.log("Saving config file to " + save_config);
    fs.writeFile(save_config, JSON.stringify(o.saved_config(global.opt), null, 2), function(err) {
      if (err) h.log_err("Error saving " + save_config + " file");
    });
  }
  h.log2("Options: " + JSON.stringify(global.opt));
  o.set_internal_opts(global.opt, o.opt_help);
  h.log3("Internal options: " + JSON.stringify(global.opt));
  p.connect_pool_throttle(global.opt.pool_ids.active = global.opt.pool_ids.primary, set_job);
  // setup all pool share report
  setInterval(function() {
    let good_shares = 0, bad_shares = 0;
    for (const pool_id in global.opt.pools) {
      good_shares += global.opt.pools[pool_id].good_shares;
      bad_shares += global.opt.pools[pool_id].bad_shares;
    }
    h.log("Accepted (" + good_shares + ") / Rejected (" + bad_shares + ") shares");
  }, global.opt.pool_time.stats * 1000);
  // if there are backup pools, try to reconnect to primary pool if it is not active
  if (global.opt.pools.length >= (global.opt.pool_ids.donate !== null ? 3 : 2))
    setInterval(function() {
      switch (global.opt.pool_ids.active) {
        case global.opt.pool_ids.primary:
        case global.opt.pool_ids.donate: return;
      }
      p.connect_pool_throttle(global.opt.pool_ids.primary, set_job);
    }, global.opt.pool_time.primary_reconnect * 1000);
  // donation mining
  if (global.opt.pool_ids.donate !== null) setInterval(function() {
    p.connect_pool_throttle(global.opt.pool_ids.donate, set_job);
    setTimeout(p.switch_pool, global.opt.pool_time.donate_length * 1000,
               global.opt.pool_ids.donate, set_job);
  }, global.opt.pool_time.donate_interval * 1000);
}

function on_exit() { exit(0, true); }

function install_exit_handlers() {
  process.on("SIGINT", on_exit);
  process.on("SIGTERM", on_exit);
  if (process.platform === "win32") process.on("SIGBREAK", on_exit);
  else process.on("SIGHUP", on_exit);
}

function detect_cpu() {
  const fallback = {
    cpu_sockets: 1,
    cpu_threads: os.cpus().length || 1,
    cpu_l3cache: 0,
  };
  if (process.platform === "win32" || !fs.existsSync("/proc/cpuinfo")) return fallback;

  const cpuinfo = fs.readFileSync("/proc/cpuinfo", "utf8");
  const processor_count = (cpuinfo.match(/^processor\s*:/gm) || []).length;
  const physical_ids = new Set();
  for (const match of cpuinfo.matchAll(/^physical id\s*:\s*(.+)$/gm)) physical_ids.add(match[1]);

  let l3cache = 0;
  const l3_ids = new Set();
  for (const index of fs.readdirSync("/sys/devices/system/cpu").filter((name) => /^cpu\d+$/.test(name))) {
    const cache_dir = `/sys/devices/system/cpu/${index}/cache`;
    if (!fs.existsSync(cache_dir)) continue;
    for (const entry of fs.readdirSync(cache_dir)) {
      try {
        const base = `${cache_dir}/${entry}`;
        if (fs.readFileSync(`${base}/type`, "utf8").trim() !== "Unified") continue;
        if (fs.readFileSync(`${base}/level`, "utf8").trim() !== "3") continue;
        const size = fs.readFileSync(`${base}/size`, "utf8").trim().match(/^(\d+)([KMG])$/i);
        if (!size) continue;
        const id = fs.existsSync(`${base}/shared_cpu_list`) ? fs.readFileSync(`${base}/shared_cpu_list`, "utf8").trim() : base;
        if (l3_ids.has(id)) continue;
        l3_ids.add(id);
        const multiplier = { K: 1024, M: 1024 * 1024, G: 1024 * 1024 * 1024 }[size[2].toUpperCase()];
        l3cache += Number(size[1]) * multiplier;
      } catch (_) {}
    }
  }

  return {
    cpu_sockets: physical_ids.size || 1,
    cpu_threads: processor_count || fallback.cpu_threads,
    cpu_l3cache: l3cache,
  };
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

switch (directive) {
  case "mine":
    install_exit_handlers();
    compute_core = h.create_core();
    compute_core.from.on("close", function() { process.exitCode = 0; });
    compute_core.from.on("algo_params", function(v) {
      add_algo_params(v);
      if (!use_msr_tuning()) {
        global.opt.default_msrs = {};
        bench_algos(start_mining);
        return;
      }
      compute_core.from.on("read_msr", function(v) {
        global.opt.default_msrs = h.unpack_msr(v);
        bench_algos(start_mining);
      });
      compute_core.from.on("error", function(v) {
        h.log("Can't access MSR: " + JSON.stringify(v.message));
        global.opt.default_msrs = {}; // do not try to write it later
        bench_algos(start_mining);
      });
      compute_core.emit_to("read_msr", h.pack_msr(global.opt.default_msrs));
    });
    compute_core.emit_to("algo_params", detect_cpu());
    break;

  case "test":
    h.recreate_threads(global.opt.job.dev, messageHandler);
    h.messageWorkers({type: "test", job: global.opt.job});
    break;

  case "bench":
    install_exit_handlers();
    h.recreate_threads(global.opt.job.dev, messageHandler);
    if (!use_msr_tuning()) {
      h.messageWorkers({type: "bench", job: last_job = global.opt.job});
      break;
    }
    compute_core = h.create_core();
    compute_core.from.on("close", function() { process.exitCode = 0; });
    compute_core.from.on("read_msr", function(v) {
      global.opt.default_msrs = h.unpack_msr(v); // to restore them on exit
      set_algo_msr(global.opt.job.algo);
      h.messageWorkers({type: "bench", job: last_job = global.opt.job});
    });
    compute_core.from.on("error", function(v) {
      h.log("Can't access MSR: " + JSON.stringify(v.message));
      h.messageWorkers({type: "bench", job: last_job = global.opt.job});
    });
    compute_core.emit_to("read_msr", h.pack_msr(global.opt.default_msrs));
    break;

  case "algo_params":
    compute_core = h.create_core();
    compute_core.from.on("close", function() { process.exitCode = 0; });
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
