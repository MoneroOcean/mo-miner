// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

"use strict";

const path = require("path");
const fs   = require("fs");
const os   = require("os");
const h    = require("./helper.js");
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
let thread_hashrates = {};
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
  if (!algo.includes("c29")) {return value;}

  const tokens = value.trim().split(/\s+/);
  const hasEol = tokens[tokens.length - 1] === "EOL";
  if (hasEol) {tokens.pop();}

  return tokens.sort().join(" ") + (hasEol ? " EOL" : "");
}

function normalizeExpectedResults(algo, value) {
  return value.split("|").map((expected) => normalizeTestResult(algo, expected));
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
  if (force) {setTimeout(() => reallyExit(code), PROCESS_EXIT_GRACE_MS).unref();}
  return false;
}

function err_exit(msg) {
  h.log_err(msg);
  return exit(1);
}

function mergeConfigOptions(opt2) {
  for (const key in opt2) {switch (key) {
    case "job": case "pool_time": // do not overwrite these option sets completely
      mergeNestedConfigOption(key, opt2[key]);
      break;
    default: global.opt[key] = opt2[key];
  }}
}

function mergeNestedConfigOption(key, values) {
  for (const key2 in values) {global.opt[key][key2] = values[key2];}
}

function loadConfigFile(config_file) {
  const config_fn = path.resolve(config_file);
  h.log("Loading config file " + config_fn);
  mergeConfigOptions(require(config_fn));
  applyPearlShapeEnv();
}

// Pearl mining params (k, noise_rank) are pool-specific (HeroMiners requires 4096/256, pearlpool
// 1024/64). Let the config carry them under algo_params.pearl.{k,rank} and forward to the env the
// native kernel and pool.js (pearlNbitsBound) both read, so a config alone is turnkey -- no env vars.
function applyPearlShapeEnv() {
  const pearl = global.opt.algo_params && global.opt.algo_params.pearl;
  if (!pearl) {return;}
  if (pearl.k    && !process.env.MOM_PEARL_K)    {process.env.MOM_PEARL_K    = String(pearl.k);}
  if (pearl.rank && !process.env.MOM_PEARL_RANK) {process.env.MOM_PEARL_RANK = String(pearl.rank);}
}

function parsePoolUri(pool_uri) {
  const pool_uri_parts = pool_uri.split(":");
  if (pool_uri_parts.length !== 2) {return o.print_help("Wrong pool URI: " + pool_uri);}
  const parsed_port = parsePoolPort(pool_uri_parts[1]);
  if (!parsed_port) {return o.print_help("Wrong pool port: " + pool_uri_parts[1]);}
  return { url: pool_uri_parts[0], port: parsed_port.port, is_tls: parsed_port.is_tls };
}

function parsePoolPort(port_tls) {
  const m = port_tls.match(/^(\d+)((?:tls)?)$/);
  if (!m) {return o.print_help("Wrong pool port: " + port_tls);}
  const port = Number(m[1]);
  if (port < 1 || port > 65535) {return o.print_help("Wrong pool port: " + port_tls);}
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
  if (args.length < 1) {return o.print_help("Directive \"mine\" needs 1+ parameters");}
  const param1 = args.shift();
  if (o.is_config_file(param1)) {return loadConfigFile(param1);} // load config file

  // setup primary pool
  if (args.length < 1) {return o.print_help("Directive \"mine\" needs 2+ parameters");}
  const pool_login = args.shift();
  const pool_pass  = optionalPoolPass(args);
  return addPrimaryPool(param1, pool_login, pool_pass);
}

function parseRemainingOptions(args) {
  while (args.length) {
    const arg = args.shift();
    if (args.length >= 1 && o.parse_opt(global.opt, o.opt_help, arg, args[0], "")) {args.shift();}
    else {return o.print_help("Unparsed option: " + arg);}
  }
}

function parse_args() {
  const args = process.argv.slice(2);

  if (args.length === 0) {return o.print_help("No directive specified");}
  directive = args.shift();
  const parser = directiveParsers[directive];
  if (!parser) {return o.print_help("Unknown directive " + directive);}
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

function reverseHexBytes(value) {
  const hex = hexWithoutPrefix(value);
  return hex.length % 2 === 0 ? (hex.match(/.{2}/g) || []).reverse().join("") : hex;
}

function ergSubmitMeta(pool, job_id) {
  return (pool.erg_submit_jobs && pool.erg_submit_jobs[job_id]) || {};
}

function ergExtraNonce2Size(pool, meta) {
  const size = Number(meta.extra_nonce2_size);
  if (Number.isInteger(size) && size >= 0 && size <= 8) {return size;}

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

// Little-endian byte order of an 8-byte counter expressed as big-endian hex (the native emits the
// search counter via %016 PRIx64 = big-endian; the wire/header stores it little-endian, matching the
// memcpy of m_nonce64 into the header at nonceoffset).
function counterHexToWireLE(nonceHex) {
  return normalizedFullNonce(nonceHex).match(/.{2}/g).reverse().join("");
}

// The submit nonce2 = the 32-byte header nonce after the pool's nonce1 prefix. The header nonce is
// nonce1 (nonce1_len bytes) || nonce2; the solver advances an 8-byte counter at the start of nonce2,
// the remaining nonce2 bytes stay as the job delivered them (zeros). Returns wire-order hex.
function equihashNonce2(pool, nonceHex) {
  const job = (pool && pool.last_job) || {};
  const nonce1_len = Number(job.nonce1_len) || 0;
  // full 32-byte nonce (64 hex) lives at the end of the 280-hex header blob
  const blob = hexWithoutPrefix(job.blob || job.blob_hex || "");
  const fullNonce = blob.slice(-64).padEnd(64, "0");
  const counterLE = counterHexToWireLE(nonceHex);   // 8-byte search counter, wire (LE) order
  // nonce2 = the counter (start of nonce2) || the nonce2 tail past the counter (job-delivered zeros);
  // the nonce1 prefix (fullNonce[0 .. nonce1_len]) is intentionally excluded per ZIP-301.
  const tail = fullNonce.slice(nonce1_len * 2 + 16);
  return (counterLE + tail).padEnd(64 - nonce1_len * 2, "0");
}

function equihashSubmitNtime(pool) {
  const job = (pool && pool.last_job) || {};
  return hexWithoutPrefix(job.ntime || "");
}

function normalizeAlgoName(algo) {
  return algo && (algo.startsWith("c29") || algo === "cuckaroo") ? "c29" : algo;
}

function parseTestArgs(args) {
  if (args.length < 2) {return o.print_help("Directive \"test\" needs two parameters");}
  global.opt.job.algo = args.shift();
  test.result_hash_hex = args.shift();
}

function parseBenchArgs(args) {
  if (args.length < 1) {return o.print_help("Directive \"bench\" needs one parameter");}
  global.opt.job.algo = args.shift();
}

const directiveParsers = {
  mine: parseMineArgs,
  test: parseTestArgs,
  bench: parseBenchArgs,
  algo_params: () => undefined,
};

if (!parse_args()) {return;}

function handleResult(msg) {
  const v = msg.value;
  const pool = global.opt.pools[v.pool_id];
  const submit_mode = pool && pool.submit_mode;
  const send = (body) => p.pool_write(v.pool_id, { jsonrpc: "2.0", id: 3, ...body });

  // Pearl: the worker already built the base64 PlainProof, and the native core emits at most one
  // solution per unit of work (job_id + header), so just relay it -- no JS-side per-job dedup
  // (which would mis-fire on HeroMiners' constant job_id).
  if (submit_mode === "pearl")
  {return send({ method: "mining.submit", params: { job_id: v.job_id, plain_proof: v.plain_proof } });}
  if (submit_mode === "erg")
  {return send({ method: "mining.submit", params: ergSubmitParams(pool, v) });}
  // Equihash 125,4 (Flux/ZIP-301): mining.submit [worker, job_id, time(8hex), nonce2(hex), solution(hex)].
  // The native solver returns the 8-byte search counter (v.nonce, big-endian hex) + the 106-hex
  // 0x34-prefixed 52-byte solution (v.solution). Rebuild nonce2 = the 32-byte header nonce minus the
  // pool's nonce1 prefix, with the search counter written little-endian at its nonceoffset.
  if (submit_mode === "equihash")
  {return send({ method: "mining.submit",
    params: [pool.login, v.job_id, equihashSubmitNtime(pool), equihashNonce2(pool, v.nonce), v.solution] });}
  // Iron Fish custom OBJECT stratum: submit {miningRequestId, randomness (8-byte BE nonce), graffiti}.
  if (submit_mode === "ironfish")
  {return p.pool_write(v.pool_id, { id: 2, method: "mining.submit",
    body: { miningRequestId: v.job_id, randomness: v.nonce, graffiti: "00".repeat(32) } });}
  // Kaspa (kheavyhash) kaspa-stratum-bridge submit: mining.submit [wallet.worker, job_id, nonce_hex].
  // The native returns the winning 8-byte nonce as 16-hex big-endian (nonce_to_hex %016PRIx64); the
  // pool parses it big-endian with the extranonce as the leading bytes, which is exactly this layout.
  if (submit_mode === "kaspa")
  {return send({ method: "mining.submit", params: [pool.login, v.job_id, "0x" + v.nonce] });}
  if (submit_mode === "beam") {
    // Beam JSON-RPC `solution`: TOP-LEVEL {id, nonce(16hex), output(208hex=104B)}. The native emits the
    // nonce as the big-endian hex of the LE-stored 8-byte blob nonce, so reverse it back to the raw
    // blob byte order the pool (and the nonceprefix) expect. The 104-byte solution is already raw.
    return send({ id: v.job_id, method: "solution", nonce: reverseHexBytes(v.nonce), output: v.solution });
  }

  const params = { id: v.worker_id, job_id: v.job_id, nonce: v.nonce, result: v.hash };
  if (v.mix_hash) {
    const headerHash = resultHeaderHash(msg, pool);
    if (submit_mode === "ethproxy")
    {return send({ method: "eth_submitWork",
      params: ["0x" + v.nonce, "0x" + headerHash.slice(0, 64), "0x" + v.mix_hash] });}
    if (submit_mode === "raven" || submit_mode === "eth")
    {return send({ method: "mining.submit",
      params: [pool.login, v.job_id, "0x" + v.nonce, "0x" + headerHash.slice(0, 64), "0x" + v.mix_hash] });}
    params.mixhash = v.mix_hash;
    if (headerHash) {params.header_hash = headerHash.slice(0, 64);}
  }
  if (v.commitment) {params.commitment = v.commitment;}
  if (v.edges) {
    params.pow = h.edge_hex2arr(v.edges);
    // for proofsize == 42 (Tari C29) we return nonce hex as usual
    if (params.pow.length !== 42) {params.nonce = Number.parseInt(params.nonce, 16);}
  }
  send({ method: "submit", params: params });
}

function resultHeaderHash(msg, pool) {
  if (msg.value.header_hash) {return msg.value.header_hash;}
  if (!pool || !pool.last_job) {return "";}
  if (pool.last_job.job_id && msg.value.job_id && pool.last_job.job_id !== msg.value.job_id) {return "";}
  return firstTruthyOr("", pool.last_job.header_hash, pool.last_job.blob, pool.last_job.blob_hex);
}

// store max last nonce for background pool job to resume it from there
function handleLastNonce(msg) {
  const pool_id = msg.value.pool_id;
  // pool_id can be "" for benchmark jobs. can not use === here since
  // global.opt.pool_ids.active is integer here
  if (!shouldStoreLastNonce(pool_id)) {return;}
  const prev_nonce = global.opt.pools[pool_id].last_job.nonce;
  const new_nonce  = msg.value.nonce;
  if (isNewerNonce(prev_nonce, new_nonce))
  {global.opt.pools[pool_id].last_job.nonce = new_nonce;}
}

function shouldStoreLastNonce(pool_id) {
  // eslint-disable-next-line eqeqeq -- pool_id is "" | number; loose != is intentional coercion
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
  if (++test.thread_tested < test_threads) {return;}

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
  if (Object.keys(thread_hashrates).length < h.get_dev_threads(last_job.dev)) {return;}

  const hashrate = collectedHashrate();
  h.log("Algo " + last_job.algo + " (" + last_job.dev + ") hashrate: " +
        h.formatHashrate(hashrate.total_hashrate) + " (" + hashrate.thread_hashrate_str + ")");
  thread_hashrates = {};
  if (algo_params_bench_cb) {return algo_params_bench_cb(hashrate.total_hashrate);}
}

function handleWorkerError(msg) {
  if (msg.value.message === "Ignore duplicate job") {return;}
  h.log_err("Compute core error: " + JSON.stringify(msg.value));
  if (test.result_hash_hex) {exit(1);} // exit with error
}

// handles messages sent to the master thread from worker threads
function messageHandler(msg) {
  const handler = masterMessageHandlers[msg.type];
  if (handler) {return handler(msg);}
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
  if (!compute_core || !Object.keys(global.opt.default_msrs).length) {return;}
  const default_msr = h.pack_msr(global.opt.default_msrs);
  default_msr.algo = algo;
  compute_core.emit_to("write_msr", default_msr);
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
    difficulty: prev_job.difficulty,
    thread_num: h.get_dev_threads(dev),
    pool_id:    pool_id,
  };
}

const nonceAt32Algos = new Set(["kawpow", "firopow", "evrprogpow", "meowpow", "etchash", "autolykos2", "fishhash"]);
// Kaspa-style algos sharing the 80-byte header / 8-byte nonce at offset 72 layout.
const kHeavyHashAlgos = new Set(["kheavyhash", "karlsenhashv2", "pyrinhashv2"]);
// Heights sampled from coin mainnets so benchmark DAG/table sizes match live pool jobs
// (epoch-0 sizes overstate hashrate by ~7-10% on these algos): ETC 2026-06-04, RVN and ERG 2026-06-12.
const benchHeightByAlgo = {
  etchash:    24689903,
  kawpow:     4407982,
  firopow:    600000,
  evrprogpow: 1800000,
  meowpow:    825000,
  autolykos2: 1806198,
};
const defaultBenchAlgos = new Set([
  "autolykos2",
  "c29",
  "cn/gpu",
  "etchash",
  "ghostrider",
  "kawpow",
  "panthera",
  // Pearl (PRL) is benched by default even though it is not (yet) a MoneroOcean pool algo: the
  // GPU PoUW NoisyGEMM path is a headline number we want reported alongside the other GPU algos.
  "pearl",
  "rx/0",
  "rx/2",
  "rx/arq",
]);

function isNonceAt32Algo(algo) {
  return nonceAt32Algos.has(algo);
}

function isKHeavyHashAlgo(algo) {
  return kHeavyHashAlgos.has(algo);
}

function isEquihashAlgo(algo) {
  return algo === "equihash125_4";
}

// A deterministic 140-byte Flux header for benching the Equihash 125,4 GPU solver (mainnet block
// 400000). Each Wagner solve over this header finds 2 distinct proofs in ~2.2 s on a B580, so the
// reported Sol/s is the solver's true throughput. The 32-byte nonce lives at offset 108.
const EQUIHASH_BENCH_BLOB =
  "04000000a8675c842f7a1342fadd00cd9b4e4909526b1c0ab5a747c5529b4deb13000000" +
  "ce7d6ea2452245925fc70c3a08a3c3dd2ca4beab7481f237a19751666bfd25c3" +
  "0fd282d94b1e1a7f2c57eb3fb9e2853d990753fa137e13c99bd43f220d4fce69" +
  "90e44f5dce28421d" +
  "600000160000000000000000000000000000000000000000000000009cfd1100";

// BeamHash III M4 keystone-shaped benchmark blob: prework(32) || nonce(8) || extranonce(4).
const BEAMHASH3_BENCH_BLOB =
  "fc40996a518c221384c9f2542ca811cd66c4ccddb001ef40b9f9ba059c20352e" +
  "0100000000000000" +
  "00000000";

function jobTarget(prev_job, algo) {
  const explicitTarget = orDefault(prev_job.target, "");
  if (algo === "pearl" || isEquihashAlgo(algo)) {
    // HeroMiners-style pools precompute the verifier bound (pool.js pearlNbitsBound -> prev_job.target);
    // Flux set_target also delivers a 256-bit big-endian hex share target. When a pool does not send a
    // target, use the lenient floor(2^256 / difficulty) fallback.
    if (explicitTarget) {return hexWithoutPrefix(explicitTarget).padStart(64, "0");}
    return fullTargetFromDifficulty(prev_job.difficulty);
  }
  if (!isNonceAt32Algo(algo)) {return explicitTarget || h.diff2target(prev_job.difficulty);}
  // autolykos2 (erg) may deliver the target as a DECIMAL string. Every other nonce-at-32 algo
  // (kawpow/firopow/evrprogpow/etchash) delivers a 256-bit HEX boundary in mining.notify/set_target --
  // and that hex can legitimately be all-[0-9] digits (e.g. WoolyPooly's "0000000100..."), so the
  // digit-only "is decimal" heuristic must be applied ONLY to autolykos2, never to the hex-target algos.
  if (algo === "autolykos2" && /^\d+$/.test(explicitTarget)) {return h.decimalTargetToHex(explicitTarget);}
  if (explicitTarget && hexWithoutPrefix(explicitTarget).length > 16)
  {return hexWithoutPrefix(explicitTarget).padStart(64, "0");}
  return h.ethDiff2Target(prev_job.difficulty || (explicitTarget ? h.target2diff(explicitTarget) : 1));
}

function fullTargetFromDifficulty(difficulty) {
  const MAX = (1n << 256n) - 1n, diff = BigInt(difficulty || 1);
  return (diff > 0n ? MAX / diff : MAX).toString(16).padStart(64, "0");
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
  job.nonceoffset = nonceOffsetOr(prev_job, 32);

  const blob = orDefault(prev_job.blob, prev_job.blob_hex);
  job.blob_hex = blob && blob.length === 64 ? blob + "0000000000000000" : blob;
}

function addKHeavyHashJobFields(job, prev_job) {
  // kHeavyHash header is 80 bytes: pre_pow_hash(32) || timestamp_le(8) || zero pad(32) || nonce_le(8),
  // with an 8-byte LE nonce at offset 72. Tests/bench pass blob_hex directly; the live kaspa dialect
  // (pool.js) builds the 80-byte header from the pool's pre_pow_hash + timestamp.
  addFixedNonceBlobFields(job, prev_job, 72);
}

function addEquihashJobFields(job, prev_job) {
  // Equihash 125,4 (Flux/ZIP-301): 140-byte Zcash header with a 32-byte nonce at offset 108. The pool
  // dialect (pool.js equihashNotifyJob) builds the full 280-hex header and sets nonceoffset to
  // 108 + nonce1_len so the solver's 8-byte search counter advances inside nonce2 (after the pool's
  // fixed nonce1 prefix). Carry ntime + nonce1_len through for the mining.submit reconstruction.
  addFixedNonceBlobFields(job, prev_job, 108);
  job.ntime       = orDefault(prev_job.ntime, "");
  job.nonce1_len  = orDefault(prev_job.nonce1_len, 0);
}

function addFixedNonceBlobFields(job, prev_job, defaultOffset) {
  job.noncebytes  = orDefault(prev_job.noncebytes, 8);
  job.nonceoffset = nonceOffsetOr(prev_job, defaultOffset);
  job.blob_hex    = orDefault(prev_job.blob, prev_job.blob_hex);
}

// BeamHash III blob = prework(32) || nonce(8) || extranonce(4) = 44 bytes; the 8-byte Beam nonce sits
// at offset 32. The pool's nonceprefix (0-6 bytes) must occupy the LEADING physical bytes of that nonce
// field. The native search counter is the LE-stored uint64 there, seeded big-endian from job.nonce and
// fixed by nicehash_mask -- so both are the reverse of the desired physical {prefix || 0...} layout.
function addBeamhash3JobFields(job, prev_job, pool) {
  job.noncebytes  = 8;
  job.nonceoffset = 32;
  // prework(64hex) || nonce(16hex, zero placeholder) || extranonce(8hex, zero) = 88 hex = 44 bytes.
  const prework = orDefault(prev_job.header_hash, prev_job.blob_hex).padStart(64, "0").slice(0, 64);
  job.blob_hex = prework + "000000000000000000000000";

  // The native writes the nonce to the blob BIG-endian (set_job + the beamhash3 loop both bswap), so the
  // 8-byte nonce field's PHYSICAL bytes equal m_nonce64's bytes most-significant-first. Beam's nonceprefix
  // must occupy the LEADING physical bytes -> the HIGH bytes of m_nonce64. So job.nonce (read big-endian
  // by the native) and the nicehash mask are the plain {prefix || counter} layout, prefix at the front.
  // The low bytes are the free search counter; seed its top free byte to 1 so the first m_nonce64 is
  // never 0 (the native loop treats m_nonce64==0 as a test dispatch).
  const prefix = hexWithoutPrefix((pool && pool.beam_nonceprefix) || "").slice(0, 16);
  const prefixBytes = Math.min(prefix.length / 2, 8);
  let nonce = prefix.slice(0, prefixBytes * 2);
  if (prefixBytes < 8) {nonce += "01";}                  // seed the first free (counter) byte = 1
  job.nonce         = nonce.padEnd(16, "0");
  job.nicehash_mask = "ff".repeat(prefixBytes).padEnd(16, "0");
}

function addStandardJobFields(job, prev_job) {
  job.noncebytes  = orDefault(prev_job.noncebytes, 4);
  job.blob_hex    = orDefault(prev_job.blob, prev_job.blob_hex);
  job.nonceoffset = nonceOffsetOr(prev_job, job.algo === "ghostrider" ? 76 : 39);
}

function addNoncePrefix(job, prev_job) {
  // we need to create nonce with xn prefix and update nicehash_mask to cover it
  const nicehash_prefix = Buffer.from(prev_job.xn, "hex").subarray(0, job.noncebytes);
  job.nicehash_mask = Buffer.alloc(job.noncebytes, 0).fill(0xFF, 0, nicehash_prefix.length).toString("hex");
  job.nonce = Buffer.concat([nicehash_prefix, Buffer.alloc(job.noncebytes - nicehash_prefix.length, 0x00)]).toString("hex");
}

function defaultNicehashMask(job, pool_id, last_job_can_be_used) {
  if (last_job_can_be_used && last_job.nicehash_mask) {return last_job.nicehash_mask;}
  return Buffer.alloc(job.noncebytes, 0)
    .fill(0xFF, 0, global.opt.pools[pool_id].is_nicehash ? 1 : 0)
    .toString("hex");
}

function addNonceFields(job, prev_job, pool_id) {
  if (prev_job.xn) {return addNoncePrefix(job, prev_job);}

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
  if (algo !== "c29") {return {};}

  // C29 submits hundreds of short SYCL kernels per second; legacy non-immediate
  // Level Zero command lists avoid the one-core immediate-list path on Intel GPUs.
  return {
    SYCL_UR_USE_LEVEL_ZERO_V2: "0",
    SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS: "0",
    MOM_C29_SEED_BLOCKS: process.env.MOM_C29_SEED_BLOCKS || "16",
  };
}

function ensureWorkersForJob(algo, dev) {
  if (!last_job || last_job.algo !== algo || last_job.dev !== dev)
  {h.recreate_threads(dev, messageHandler, workerRuntimeEnv(algo));}
}

// prev_job can be either job json from the pool or
// previous job restored from the pool switch (with nonce that we need to take into account)
function set_job(prev_job) {
  const algo = normalizeAlgoName(prev_job.algo || global.opt.job.algo);
  const dev = jobDev(algo);
  ensureWorkersForJob(algo, dev);
  const pool_id = global.opt.pool_ids.active;
  const job = baseJob(prev_job, algo, dev, pool_id);
  if (algo === "c29") {addC29JobFields(job, prev_job);}
  else if (algo === "beamhash3") {addBeamhash3JobFields(job, prev_job, global.opt.pools[pool_id]);}
  else if (isKHeavyHashAlgo(algo)) {addKHeavyHashJobFields(job, prev_job);}
  else if (isEquihashAlgo(algo)) {addEquihashJobFields(job, prev_job);}
  else if (isNonceAt32Algo(algo)) {addEthHashJobFields(job, prev_job);}
  else {addStandardJobFields(job, prev_job);}
  // BeamHash III seeds its nonce from the pool nonceprefix inside addBeamhash3JobFields; the generic
  // nonce/nicehash defaults would clobber that, so only run them for the other algos.
  if (algo !== "beamhash3") {addNonceFields(job, prev_job, pool_id);}
  else {
    job.nicehash_mask = orDefault(job.nicehash_mask, "0000000000000000");
    job.nonce = orDefault(job.nonce, "0000000000000000");
  }
  set_algo_msr(algo);
  h.messageWorkers({type: "job", job: last_job = job});
  return job;
}

function prepareBenchmarkJob(job) {
  if (isNonceAt32Algo(job.algo)) {
    job.noncebytes = 8;
    job.nonceoffset = 32;
    if (job.blob_hex && job.blob_hex.length === 64) {job.blob_hex += "0000000000000000";}
  }
  if (isKHeavyHashAlgo(job.algo)) {
    // 80-byte Kaspa-style header, 8-byte nonce at offset 72 (kheavyhash: matrix from header[0..31];
    // karlsenhashv2: prePow||ts||zeros||nonce -> the FishHash DAG).
    job.noncebytes = 8;
    job.nonceoffset = 72;
    if (!job.blob_hex || job.blob_hex.length !== 160)
    {job.blob_hex = "2a".repeat(32) + "52c9f84301000000" + "00".repeat(32) + "0000000000000000";}
  }
  if (isEquihashAlgo(job.algo)) {
    // Equihash 125,4 (ZelHash/Flux): 140-byte Zcash header with a 32-byte nonce at offset 108. Bench
    // over the deterministic block-400000 header so each solve finds ~1.88 proofs and the rate is Sol/s.
    job.noncebytes = 8;
    job.nonceoffset = 108;
    if (!job.blob_hex || job.blob_hex.length !== 280) {job.blob_hex = EQUIHASH_BENCH_BLOB;}
    job.height = job.height || 400000;
  }
  if (job.algo === "beamhash3") {
    // BeamHash III: one Wagner solve per dispatch over a deterministic M4-shaped prework. Seed the
    // 8-byte nonce nonzero so the native path does not classify the dispatch as an is_test gen run.
    job.noncebytes = 8;
    job.nonceoffset = 32;
    if (!job.blob_hex || job.blob_hex.length !== 88) {job.blob_hex = BEAMHASH3_BENCH_BLOB;}
    job.nonce = job.nonce || "0100000000000000";
    job.nicehash_mask = job.nicehash_mask || "0000000000000000";
  }
  if (benchHeightByAlgo[job.algo]) {job.height = job.height || benchHeightByAlgo[job.algo];}
  if (job.algo === "etchash") {job.seed_hex = "";}
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
  return process.platform !== "win32";
}

function add_algo_params(params) {
  for (const algo in params) {
    if (!(algo in global.opt.algo_params))
    {global.opt.algo_params[algo] = { dev: params[algo], perf: null };}
  }
}

function use_algo_param_benchmarks() {
  return global.opt.bench_algo_params !== 0;
}

function prepare_fixed_algo_params() {
  const algo = normalizeAlgoName(global.opt.job.algo);
  if (!algo) {return;}
  const algo_param = global.opt.algo_params[algo] || { dev: global.opt.job.dev, perf: null };
  global.opt.job.algo = algo;
  global.opt.algo_params = { [algo]: algo_param };
}

function start_after_algo_params() {
  if (use_algo_param_benchmarks()) {return bench_algos(start_mining);}
  prepare_fixed_algo_params();
  return start_mining();
}

function createComputeCore() {
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
      fs.writeSync(1, "MOM_ALGO_PARAMS " + JSON.stringify(v) + "\n");
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
