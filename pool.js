// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

"use strict";

const net  = require("net");
const tls  = require("tls");
const h    = require("./helper.js");
const o    = require("./opts.js");
const {normalizeAlgoName} = require("./miner/algorithms");

// Correctness tests may use normal network services, but must never contact a mining pool. Capture
// the actual socket functions before logic tests substitute their in-memory sockets, so the guard
// blocks only a real pool connection and leaves the protocol fixtures testable.
const systemNetConnect = net.connect;
const systemTlsConnect = tls.connect;

const max_pool_data_buffer = 1024 * 1024;

function pool_str(pool_id) {
  const pool = global.opt.pools[pool_id];
  return pool.url + ":" + pool.port + (pool.is_tls ? "tls" : "");
}

function pool_log_str(pool_id, str) {
  return global.opt.log_level >= 1 ? "[" + pool_str(pool_id) + "] " + str : str;
}
function pool_log(pool_id, str)     { h.log(pool_log_str(pool_id, str)); }
function pool_log1(pool_id, str)    { h.log1(pool_log_str(pool_id, str)); }
function pool_log2(pool_id, str)    { h.log2(pool_log_str(pool_id, str)); }
function pool_log_err(pool_id, str) { h.log_err(pool_log_str(pool_id, str)); }

function clear_pool_connection(pool_id, socket) {
  const pool = global.opt.pools[pool_id];
  if (socket && !isCurrentPoolSocket(pool_id, socket)) {return false;}
  clearPoolKeepalive(pool);
  if (pool.socket) {pool.socket.destroy();}
  pool.socket   = null;
  pool.last_job = null;
  pool.logged_in = false;
  pool.pending_authorize = false;
  return true;
}

function clearPoolKeepalive(pool) {
  if (pool.keepalive !== null) {clearTimeout(pool.keepalive);}
  pool.keepalive = null;
}

function isCurrentPoolSocket(pool_id, socket) {
  return global.opt.pools[pool_id].socket === socket;
}

// Maps a mining algo to its stratum protocol dialect, or null if it uses the default `login` dialect.
function protocolForAlgo(algo) {
  switch (normalizeAlgoName(algo)) {
    case "kawpow":     return "raven";
    case "firopow":    return "raven";
    case "evrprogpow": return "raven";
    case "meowpow":    return "raven";
    case "etchash":    return "eth";
    case "autolykos2": return "erg";
    case "pearlhash":  return "pearlhash";
    case "fishhash":   return "ironfish";
    case "zelhash":    return "zelhash";
    case "beamhash3":  return "beam";
    case "karlsenhashv2": return "kaspa";
    default:           return null;
  }
}

// Beam packs the network difficulty into a 32-bit int (top 8 bits = order, low 24 = mantissa). The
// native beamhash3 solver re-derives it from the low 4 bytes of the 32-byte big-endian target, so the
// JS job carries the packed int there. We keep a per-pool copy so a job that omits `difficulty` (some
// pools push it only on the login/set) can still resolve a target.
function beamPackedTarget(packed) {
  const p = (packed >>> 0).toString(16).padStart(8, "0");
  return "0".repeat(56) + p;   // 64 hex = 32 bytes big-endian, packed int32 in the low 4 bytes
}

function defaultPoolProtocol() {
  const job = global.opt && global.opt.job;
  return (job && protocolForAlgo(job.algo)) || "login";
}

function poolProtocol(pool) {
  return pool.protocol || pool.inferred_protocol || defaultPoolProtocol();
}

function usesMiningSubscribe(pool) {
  const protocol = poolProtocol(pool);
  return protocol === "raven" || protocol === "eth" || protocol === "erg" ||
         protocol === "zelhash" || protocol === "kaspa";
}

function usesEthProxy(pool) {
  return poolProtocol(pool) === "ethproxy";
}

function usesIronfish(pool) {
  return poolProtocol(pool) === "ironfish";
}

// Standard Pearl handshake (HeroMiners/LuckyPool/etc.): mining.subscribe + mining.authorize
// {wallet,worker,pass}. This is the default for PearlHash pools. pearlpool.cloud uses the older single
// `login` dialect instead -- opt OUT of subscribe there with "use_subscribe": false (the MoneroOcean
// donate pool also sets it false so donation keeps using login). Both dialects push the same
// object-param PearlHash mining.notify and take the same mining.submit{job_id,plain_proof}.
function pearlhashUsesSubscribe(pool) {
  // MOM_PEARLHASH_LOGIN forces pearlpool.cloud's login dialect for CLI mining.
  // The MO donate pool opts out via use_subscribe:false, so this env never affects donation.
  if (process.env.MOM_PEARLHASH_LOGIN) {return false;}
  return poolProtocol(pool) === "pearlhash" && pool.use_subscribe !== false;
}

// Pearl difficulty is carried in the job_id suffix "<hex>_<diff>" (HeroMiners omits the difficulty
// field that pearlpool.cloud sends); used to derive the 2^256/diff kernel target.
function pearlhashDiffFromJobId(job_id) {
  const m = String(job_id || "").match(/_(\d+)$/);
  return m ? Number(m[1]) : undefined;
}

// k - k%rank (the "dot_product_length"), from the same env the native kernel reads. Defaults MUST
// match the native PearlHash k/rank defaults (4096/256) or the JS-computed jackpot bound disagrees with
// the kernel/verifier and shares come out too rare.
function pearlhashKEff() {
  const k = Number(process.env.MOM_PEARLHASH_K) || 4096;
  const rank = Number(process.env.MOM_PEARLHASH_RANK) || 256;
  return Math.floor(k / rank) * rank;
}

// HeroMiners sends the BASE target T0 (= nbits_to_difficulty(share_nbits)); the actual jackpot bound
// the verifier checks is T0 * (16*16) * (k - k%rank)  (zk-pow extract_difficulty_bound: tile_size *
// dot_product_length). pearlpool instead accepts the lenient 2^256/diff and its target field is the
// network block target (ignored).
function pearlhashNbitsBound(baseTargetHex) {
  const MAX = (1n << 256n) - 1n;
  const base = BigInt("0x" + (hexWithoutPrefix(baseTargetHex) || "0"));
  let bound = base * BigInt(16 * 16 * pearlhashKEff());
  if (bound > MAX) {bound = MAX;}
  return bound.toString(16).padStart(64, "0");
}

module.exports.pool_write = function(pool_id, json) {
  const message = JSON.stringify(json);
  const pool = global.opt.pools[pool_id];
  if (!pool.socket) {return pool_log2(pool_id, "Sent to the closed pool socket: " + message);}

  pool_log2(pool_id, "Sent to the pool: " + message);
  pool.socket.write(message + "\n");
  // sends keepalive if no submit/keepalive to pool for more than global.opt.pool_time.keepalive
  if (!pool.is_keepalive || usesMiningSubscribe(pool) || usesEthProxy(pool) || pearlhashUsesSubscribe(pool) || usesIronfish(pool)) {return;}
  clearPoolKeepalive(pool);
  pool.keepalive = setTimeout(function() {
    pool.keepalive = null;
    module.exports.pool_write(pool_id, {
      jsonrpc: "2.0", id: 2, method: "keepalive", params: {}
    });
  }, global.opt.pool_time.keepalive * 1000);
};

// soft kill pool connection
function pool_close_wait(pool_id) {
  const socket = global.opt.pools[pool_id].socket;
  if (!socket) {return;}
  pool_log1(pool_id, "Soft closing the pool connection");
  setTimeout(function() {
    // do not do soft close if this pool became active again
    if (pool_id === global.opt.pool_ids.active ||
        !isCurrentPoolSocket(pool_id, socket)) {return;}
    pool_log1(pool_id, "Soft closed the pool connection");
    clear_pool_connection(pool_id, socket);
  }, global.opt.pool_time.close_wait * 1000);
}

function poolShareStats(pool_id) {
  const pool = global.opt.pools[pool_id];
  return "(" + pool.good_shares + "/" + pool.bad_shares + ")";
}

function poolErrorText(error) {
  return error instanceof Object && typeof error.message === "string" ? ": " + error.message : "";
}

function applyLoginExtensions(pool_id, extensions) {
  if (!Array.isArray(extensions)) {return;}
  const pool = global.opt.pools[pool_id];
  if (extensions.includes("nicehash")) {pool.is_nicehash = true;}
  if (extensions.includes("keepalive")) {pool.is_keepalive = true;}
}

function algoFromPass(pool) {
  const pass = String(pool.pass || "");
  const m = pass.match(/(?:^|[~;,])(?:algo=)?(kawpow|firopow|evrprogpow|meowpow|etchash|autolykos2|pearlhash|fishhash|zelhash)(?:$|[~;,])/i);
  return m ? normalizeAlgoName(m[1]) : "";
}

function rememberPoolProtocol(pool_id, result) {
  const pool = global.opt.pools[pool_id];
  if (pool.protocol) {return;}
  const protocol = protocolForAlgo((result && result.algo) || algoFromPass(pool));
  if (!protocol) {return;}
  pool.inferred_protocol = protocol;
  if (usesMiningSubscribe(pool) || usesEthProxy(pool)) {clearPoolKeepalive(pool);}
}

const poolJobs = require("./pool/jobs")({
  h, normalizeAlgoName, poolProtocol, usesEthProxy, pearlhashUsesSubscribe,
  pearlhashDiffFromJobId, pearlhashNbitsBound, beamPackedTarget, pool_close_wait,
  pool_log, pool_str, algoFromPass, applyLoginExtensions,
  connectPoolThrottle: (...args) => module.exports.connect_pool_throttle(...args),
});
const {
  isObject, isIronfishSetTargetNotification, isRavenSetTargetNotification,
  isSetDifficultyNotification, isSetExtranonceNotification, hexWithoutPrefix,
  validExtraNonce, rememberPoolExtraNonceHex, rememberSubscribeExtraNonce,
  switchPool, handleRavenSetTarget, handleEthSetTarget, handleZelHashSetTarget,
  handleIronfishSetTarget, handleSetDifficulty, jobFromPoolMessage,
} = poolJobs;
module.exports.switch_pool = switchPool;
function jobTargetWork(job) {
  // BeamHash III carries a PACKED 32-bit network difficulty (not a 256-bit boundary); the share rate is
  // in solutions, so report the packed difficulty itself rather than decoding job.target as a boundary.
  if (job.algo === "beamhash3") {return job.difficulty ? BigInt(job.difficulty) : null;}
  if (!job.target) {return null;}
  if (job.algo === "kawpow" || job.algo === "firopow" || job.algo === "evrprogpow" || job.algo === "meowpow")
  {return h.kawpowTarget2diff(job.target);}
  // PearlHash reports the share target in GEMM MACs to match the MAC/s hashrate (so time-per-share =
  // target/hashrate). work/share = (tiles/share = 2^256/bound) * (MACs/tile = 16*16*k_eff).
  if (job.algo === "pearlhash") {return h.target256ToWork(job.target) * BigInt(16 * 16 * pearlhashKEff());}
  // etchash/autolykos2/fishhash carry a full 256-bit target too, but their hashrate is in hashes -> H/share.
  if (job.algo === "etchash" || job.algo === "autolykos2" || job.algo === "fishhash" ||
      job.algo === "zelhash" ||
      job.algo === "karlsenhashv2")
  {return h.target256ToWork(job.target);}
  return h.target2diff(job.target);
}

function jobTargetDescription(job) {
  const work = jobTargetWork(job);
  return work !== null ? h.formatHashCount(work) + "/share target" : job.difficulty + " diff";
}

function activatePoolForJob(pool_id, active_pool) {
  // only switch active pool once for its first job here
  if (pool_id === active_pool || global.opt.pools[pool_id].last_job) {return;}
  const activator = poolActivator(pool_id);
  if (activator) {activator(pool_id, active_pool);}
}

function activatePrimaryPool(pool_id, active_pool) {
  pool_log(pool_id, "Switching active pool to primary " + pool_str(pool_id) + " pool");
  pool_close_wait(active_pool);
  global.opt.pool_ids.active = pool_id;
}

function activateDonatePool(pool_id) {
  pool_log(pool_id, "Switching active pool to donate " + pool_str(pool_id) + " pool");
  global.opt.pool_ids.active = pool_id;
}

function poolActivator(pool_id) {
  switch (pool_id) {
    case global.opt.pool_ids.primary: return activatePrimaryPool;
    case global.opt.pool_ids.donate:  return activateDonatePool;
    default:                          return null;
  }
}

function handlePoolJob(pool_id, job, set_job) {
  activatePoolForJob(pool_id, global.opt.pool_ids.active);
  if (job.target) {jobTargetWork(job);} // throws early on a malformed target before we store the job

  global.opt.pools[pool_id].last_job = job;
  if (pool_id === global.opt.pool_ids.active) {
    const last_job = set_job(job);
    pool_log(pool_id, "Got new " + last_job.algo + " algo job with " +
                     jobTargetDescription(last_job) +
                     (last_job.height ? " and " + last_job.height + " height" : "")
    );
  } else {
    pool_log2(pool_id, "Storing not active pool job " + JSON.stringify(job));
  }
}

function loginSucceeded(pool_id) {
  global.opt.pools[pool_id].logged_in = true;
  return pool_log(pool_id, "Login to the pool succeeded");
}

function loginFailed(pool_id, reason) {
  global.opt.pools[pool_id].logged_in = false;
  return pool_log_err(pool_id, "Login to the pool failed" + reason);
}

function handleLoginResponse(pool_id, is_err, is_ok, err_msg, json) {
  if (is_err || json.result === false) {return loginFailed(pool_id, err_msg || ": Login rejected");}
  if (is_ok) {return loginSucceeded(pool_id);}
}

function handleSubscribeResponse(pool_id, is_err, is_ok, err_msg, json) {
  if (is_err) {return pool_log_err(pool_id, "Subscribe to the pool failed" + err_msg);}
  if (!is_ok) {return;}
  rememberSubscribeExtraNonce(pool_id, json.result);
  const pool = global.opt.pools[pool_id];
  pool.pending_authorize = true;
  return module.exports.pool_write(pool_id, {
    jsonrpc: "2.0", id: 2, method: "mining.authorize", params: [pool.login, pool.pass]
  });
}

function handleAuthorizeResponse(pool_id, is_err, is_ok, err_msg, json) {
  global.opt.pools[pool_id].pending_authorize = false;
  if (!is_err && json.result === true) {return loginSucceeded(pool_id);}
  return loginFailed(pool_id, err_msg || ": Authorization rejected");
}

function handleShareResponse(pool_id, is_err, is_ok, err_msg) {
  if (is_err || is_ok === false) {
    ++ global.opt.pools[pool_id].bad_shares;
    return pool_log_err(pool_id, "Share rejected by the pool " + poolShareStats(pool_id) + err_msg);
  }
  if (is_ok) {
    ++ global.opt.pools[pool_id].good_shares;
    return pool_log(pool_id, "Share accepted by the pool " + poolShareStats(pool_id));
  }
}

function handlePoolResponse(pool_id, json) {
  const is_err  = "error" in json && json.error !== null;
  const err_msg = is_err ? poolErrorText(json.error) : "";
  const is_ok   = "result" in json && json.result !== null && json.result !== false;
  const handler = poolResponseHandler(pool_id, json.id);
  const result = handler(pool_id, is_err, is_ok, err_msg, json);
  if (!is_err && is_ok) {rememberPoolResponseMetadata(pool_id, json.result);}
  return result;
}

function rememberPoolResponseMetadata(pool_id, result) {
  if (!isObject(result)) {return;}

  const pool = global.opt.pools[pool_id];
  if ("id" in result) {pool.worker_id = result.id;}
  rememberPoolExtraNonceHex(pool_id, result.extra_nonce);
  rememberPoolProtocol(pool_id, result);
  applyLoginExtensions(pool_id, result.extensions);
}

function ignorePoolResponse() {
  return undefined;
}

function poolResponseHandler(pool_id, id) {
  const pool = global.opt.pools[pool_id];
  if (pearlhashUsesSubscribe(pool)) {
    if (id === 1) {return ignorePoolResponse;}           // subscribe ack/err (authorize already sent)
    if (id === 2) {return pool.pending_authorize ? handleAuthorizeResponse : ignorePoolResponse;}
  } else if (usesMiningSubscribe(pool)) {
    if (id === 1) {return handleSubscribeResponse;} // mining.subscribe response
    if (id === 2) {return pool.pending_authorize ? handleAuthorizeResponse : ignorePoolResponse;}
  } else if (usesEthProxy(pool)) {
    if (id === 1) {return handleLoginResponse;} // eth_submitLogin response
    if (id === 2) {return ignorePoolResponse;} // legacy keepalive response
  } else {
    if (id === 1) {return handleLoginResponse;} // login response
    if (id === 2) {return ignorePoolResponse;} // keepalive response
  }
  return handleShareResponse; // share submit response
}

// Iron Fish handshake replies (mining.subscribed / mining.submitted) are METHOD pushes, NOT
// {id,result} responses, and Iron Fish reuses ids across messages -- so they must be matched by
// method, never routed through the id-keyed handlePoolResponse.
function handleIronfishSubscribed(pool_id, json) {
  const pool = global.opt.pools[pool_id];
  const body = isObject(json.body) ? json.body : {};
  pool.ironfish_xn = validExtraNonce(body.xn) || hexWithoutPrefix(body.xn);
  return loginSucceeded(pool_id);
}

function handleIronfishSubmitted(pool_id, json) {
  const ok = isObject(json.body) && json.body.result === true;
  return handleShareResponse(pool_id, !ok, ok, ok ? "" : ": rejected");
}

function handleIronfishMessage(pool_id, json, set_job) {
  if (poolProtocol(global.opt.pools[pool_id]) !== "ironfish") {return false;}
  if (json.method === "mining.subscribed") { handleIronfishSubscribed(pool_id, json); return true; }
  if (json.method === "mining.submitted")  { handleIronfishSubmitted(pool_id, json);  return true; }
  if (isIronfishSetTargetNotification(json)) { handleIronfishSetTarget(pool_id, json); return true; }
  const job = jobFromPoolMessage(pool_id, json);
  if (job) { handlePoolJob(pool_id, job, set_job); return true; }
  return false;
}

// Beam replies (to login and to solution submits) are `method:"result"` messages carrying a `code`
// field (0 = login OK, 1 = share accepted; anything else = error/reject) plus an optional description.
// The login reply also carries `nonceprefix` (0-6 bytes) -- the 8-byte mining nonce's leading bytes
// MUST match it, so we stash it and seed the job nonce + nicehash mask from it.
function isBeamResult(json) {
  return json.method === "result" && typeof json.code === "number";
}

function handleBeamResult(pool_id, json) {
  const pool = global.opt.pools[pool_id];
  const desc = json.description ? ": " + json.description : "";
  if (String(json.id) === "login" || "nonceprefix" in json) {
    if (json.code === 0) {
      if (typeof json.nonceprefix === "string") {pool.beam_nonceprefix = hexWithoutPrefix(json.nonceprefix);}
      if (typeof json.forkheight === "number") {pool.beam_forkheight = json.forkheight;}
      return loginSucceeded(pool_id);
    }
    return loginFailed(pool_id, desc || ": Login rejected");
  }
  // a solution-submit reply
  if (json.code === 1) {
    ++ pool.good_shares;
    return pool_log(pool_id, "Share accepted by the pool " + poolShareStats(pool_id));
  }
  ++ pool.bad_shares;
  return pool_log_err(pool_id, "Share rejected by the pool " + poolShareStats(pool_id) + desc);
}

function pool_message(pool_id, json, set_job) {
  if (poolProtocol(global.opt.pools[pool_id]) === "beam" && isBeamResult(json))
  {return handleBeamResult(pool_id, json);}
  if (handleIronfishMessage(pool_id, json, set_job)) {return;}
  if (isRavenSetTargetNotification(json)) {
    const protocol = poolProtocol(global.opt.pools[pool_id]);
    if (protocol === "eth") {return handleEthSetTarget(pool_id, json);}
    if (protocol === "zelhash") {return handleZelHashSetTarget(pool_id, json);}
    return handleRavenSetTarget(pool_id, json);
  }
  if (isSetDifficultyNotification(json)) {return handleSetDifficulty(pool_id, json);}
  if (isSetExtranonceNotification(json)) {
    rememberPoolExtraNonceHex(pool_id, json.params[0]);
    if (Number.isInteger(Number(json.params[1])))
    {global.opt.pools[pool_id].extra_nonce2_size = Number(json.params[1]);}
    return;
  }
  const job = jobFromPoolMessage(pool_id, json);
  if (job) {return handlePoolJob(pool_id, job, set_job);}
  if ("id" in json) {return handlePoolResponse(pool_id, json);}

  pool_log1(pool_id, "Unknown message from the pool: " + JSON.stringify(json));
}

const { connectPoolThrottle } = require("./pool/connection")({
  h, o, net, tls, systemNetConnect, systemTlsConnect, max_pool_data_buffer,
  clear_pool_connection, isCurrentPoolSocket, pearlhashUsesSubscribe,
  poolProtocol, pool_log, pool_log1, pool_log2, pool_log_err, pool_log_str,
  pool_message, pool_str, usesEthProxy, usesIronfish, usesMiningSubscribe,
  poolWrite: (...args) => module.exports.pool_write(...args),
  switchPool: (...args) => module.exports.switch_pool(...args),
});
module.exports.connect_pool_throttle = connectPoolThrottle;
