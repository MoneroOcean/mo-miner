// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

"use strict";

const net  = require("net");
const tls  = require("tls");
const h    = require("./helper.js");
const o    = require("./opts.js");

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
  if (socket && !isCurrentPoolSocket(pool_id, socket)) return false;
  clearPoolKeepalive(pool);
  destroyPoolSocket(pool);
  pool.socket   = null;
  pool.last_job = null;
  pool.logged_in = false;
  pool.pending_authorize = false;
  return true;
}

function clearPoolKeepalive(pool) {
  if (pool.keepalive !== null) clearTimeout(pool.keepalive);
  pool.keepalive = null;
}

function destroyPoolSocket(pool) {
  if (pool.socket) pool.socket.destroy();
}

function isCurrentPoolSocket(pool_id, socket) {
  return global.opt.pools[pool_id].socket === socket;
}

function defaultPoolProtocol() {
  const job = global.opt && global.opt.job;
  if (job && job.algo === "kawpow") return "raven";
  if (job && job.algo === "etchash") return "eth";
  if (job && job.algo === "autolykos2") return "erg";
  return "login";
}

function poolProtocol(pool) {
  return pool.protocol || pool.inferred_protocol || defaultPoolProtocol();
}

function usesMiningSubscribe(pool) {
  return poolProtocol(pool) === "raven" || poolProtocol(pool) === "eth" || poolProtocol(pool) === "erg";
}

function usesEthProxy(pool) {
  return poolProtocol(pool) === "ethproxy";
}

module.exports.pool_write = function(pool_id, json) {
  const message = JSON.stringify(json);
  const pool = global.opt.pools[pool_id];
  if (!pool.socket) return pool_log2(pool_id, "Sent to the closed pool socket: " + message);

  pool_log2(pool_id, "Sent to the pool: " + message);
  pool.socket.write(message + "\n");
  // sends keepalive if no submit/keepalive to pool for more than global.opt.pool_time.keepalive
  if (!pool.is_keepalive || usesMiningSubscribe(pool) || usesEthProxy(pool)) return;
  if (pool.keepalive !== null) clearTimeout(pool.keepalive);
  pool.keepalive = setTimeout(function() {
    pool.keepalive = null;
    module.exports.pool_write(pool_id, {
      jsonrpc: "2.0", id: 2, method: "keepalive", params: {}
    });
  }, global.opt.pool_time.keepalive * 1000);
};

// soft kill pool connection
function pool_close_wait(pool_id) {
  if (!global.opt.pools[pool_id].socket) return;
  const socket = global.opt.pools[pool_id].socket;
  pool_log1(pool_id, "Soft closing the pool connection");
  setTimeout(function() {
    // do not do soft close if this pool became active again
    if (pool_id === global.opt.pool_ids.active ||
        !isCurrentPoolSocket(pool_id, socket)) return;
    pool_log1(pool_id, "Soft closed the pool connection");
    clear_pool_connection(pool_id, socket);
  }, global.opt.pool_time.close_wait * 1000);
}

function poolShareStats(pool_id) {
  const pool = global.opt.pools[pool_id];
  return "(" + pool.good_shares + "/" + pool.bad_shares + ")";
}

function poolErrorMessage(json, is_err) {
  return is_err ? poolErrorText(json.error) : "";
}

function poolErrorText(error) {
  return error instanceof Object && typeof error.message === "string" ? ": " + error.message : "";
}

function applyLoginExtensions(pool_id, extensions) {
  if (!Array.isArray(extensions)) return;
  const pool = global.opt.pools[pool_id];
  if (extensions.includes("nicehash")) pool.is_nicehash = true;
  if (extensions.includes("keepalive")) pool.is_keepalive = true;
}

function rememberPoolExtraNonce(pool_id, result) {
  rememberPoolExtraNonceHex(pool_id, result.extra_nonce);
}

function protocolForAlgo(algo) {
  if (algo === "kawpow") return "raven";
  if (algo === "etchash") return "eth";
  if (algo === "autolykos2") return "erg";
  return null;
}

function algoFromPass(pool) {
  const pass = String(pool.pass || "");
  const m = pass.match(/(?:^|[~;,])(?:algo=)?(kawpow|etchash|autolykos2)(?:$|[~;,])/i);
  return m ? m[1].toLowerCase() : "";
}

function rememberPoolProtocol(pool_id, result) {
  const pool = global.opt.pools[pool_id];
  if (pool.protocol) return;
  const protocol = protocolForAlgo((result && result.algo) || algoFromPass(pool));
  if (!protocol) return;
  pool.inferred_protocol = protocol;
  if (usesMiningSubscribe(pool) || usesEthProxy(pool)) clearPoolKeepalive(pool);
}

function isObject(value) {
  return value instanceof Object;
}

function isJobNotification(json) {
  return json.method === "job" && isObject(json.params);
}

function isRavenJobNotification(json) {
  return json.method === "mining.notify" && Array.isArray(json.params) && json.params.length >= 6;
}

function isEthJobNotification(json) {
  return json.method === "mining.notify" && Array.isArray(json.params) && json.params.length >= 4;
}

function isErgJobNotification(json) {
  return json.method === "mining.notify" && Array.isArray(json.params) && json.params.length >= 7;
}

function isEthProxyWork(json) {
  return Array.isArray(json.result) && json.result.length >= 3;
}

function isRavenSetTargetNotification(json) {
  return json.method === "mining.set_target" && Array.isArray(json.params) && json.params.length >= 1;
}

function isSetDifficultyNotification(json) {
  return json.method === "mining.set_difficulty" && Array.isArray(json.params) && json.params.length >= 1;
}

function hexWithoutPrefix(value) {
  return String(value || "").replace(/^0x/i, "");
}

function validExtraNonce(value) {
  const hex = hexWithoutPrefix(value);
  return hex.length > 0 && hex.length % 2 === 0 && hex.length <= 16 && !/[^0-9a-f]/i.test(hex) ? hex : "";
}

function subscribeExtraNonceCandidates(result) {
  if (!Array.isArray(result)) return [];
  return Array.isArray(result[0]) || result[0] == null ? [result[1]] : result;
}

function subscribeExtraNonce2Size(result) {
  if (!Array.isArray(result) || !(Array.isArray(result[0]) || result[0] == null)) return null;
  const size = Number(result[2]);
  return Number.isInteger(size) && size >= 0 && size <= 8 ? size : null;
}

function rememberPoolExtraNonceHex(pool_id, value) {
  const extra_nonce = validExtraNonce(value);
  if (extra_nonce) global.opt.pools[pool_id].extra_nonce = extra_nonce;
}

function rememberSubscribeExtraNonce(pool_id, result) {
  rememberPoolExtraNonceHex(pool_id, subscribeExtraNonceCandidates(result).find(validExtraNonce));
  const extra_nonce2_size = subscribeExtraNonce2Size(result);
  if (extra_nonce2_size !== null) global.opt.pools[pool_id].extra_nonce2_size = extra_nonce2_size;
}

function fixedHexBytesLE(hex, bytes) {
  const padded = hex.padEnd(bytes * 2, "0").slice(0, bytes * 2);
  return padded.match(/.{2}/g).reverse().join("");
}

function ravenExtraNonce(pool) {
  return hexWithoutPrefix(pool.extra_nonce || "");
}

function ravenNonce(pool) {
  return ravenExtraNonce(pool).padEnd(16, "0").slice(0, 16);
}

function ravenNonceMask(pool) {
  return "ff".repeat(ravenExtraNonce(pool).length / 2).padEnd(16, "0");
}

function ethExtraNonce(pool) {
  return hexWithoutPrefix(pool.extra_nonce || "");
}

function ethNonce(pool) {
  return ethExtraNonce(pool).padEnd(16, "0").slice(0, 16);
}

function ethNonceMask(pool) {
  return "ff".repeat(ethExtraNonce(pool).length / 2).padEnd(16, "0");
}

function ethBlob(headerHash, pool) {
  return headerHash + fixedHexBytesLE(ethExtraNonce(pool), 8);
}

function parseHexHeight(value) {
  const hex = hexWithoutPrefix(value);
  if (!hex || /[^0-9a-f]/i.test(hex)) return 0;
  return Number.parseInt(hex, 16);
}

function ergTarget(bound) {
  return h.decimalTargetToHex(bound);
}

function rememberErgSubmitJob(pool, job) {
  if (!pool.erg_submit_jobs) pool.erg_submit_jobs = {};
  pool.erg_submit_jobs[job.job_id] = {
    extra_nonce: ethExtraNonce(pool),
    extra_nonce2_size: pool.extra_nonce2_size,
    ntime: job.ntime || "",
  };

  const jobIds = Object.keys(pool.erg_submit_jobs);
  while (jobIds.length > 16) delete pool.erg_submit_jobs[jobIds.shift()];
}

function ravenBlob(headerHash, pool) {
  return headerHash + fixedHexBytesLE(ravenExtraNonce(pool), 8);
}

function ravenTarget(pool, notifyTarget) {
  const target = hexWithoutPrefix(notifyTarget || pool.raven_target || "");
  return target.padEnd(64, "0");
}

function ethTarget(pool) {
  const target = hexWithoutPrefix(pool.eth_target || "");
  return target ? target.padStart(64, "0") : h.ethDiff2Target(pool.eth_difficulty || 1);
}

function isLoginJob(json) {
  return !("error" in json && json.error !== null) &&
         isObject(json.result) && isObject(json.result.job);
}

function loginJobWithResultMetadata(result) {
  const job = { ...result.job };
  for (const key of ["algo", "height", "seed_hash", "target", "difficulty"]) {
    if (!(key in job) && key in result) job[key] = result[key];
  }
  return job;
}

function alivePoolJob(pool_id) {
  return global.opt.pools[pool_id].last_job;
}

function activateAlivePool(pool_id, set_job, label) {
  pool_log(pool_id, "Making " + label + " pool " + pool_str(pool_id) + " active again");
  global.opt.pool_ids.active = pool_id;
  return set_job(alivePoolJob(pool_id));
}

function reactivatePrimaryPool(set_job) {
  const primary_pool = global.opt.pool_ids.primary;
  if (!alivePoolJob(primary_pool)) return null;
  return activateAlivePool(primary_pool, set_job, "the primary");
}

function reactivateBackupPool(active_pool, set_job) {
  for (const pool_id of Object.keys(global.opt.pools)) {
    // === will not work here since here we are comparing strings and integers
    if (shouldSkipBackupPool(pool_id, active_pool)) continue;
    return activateAlivePool(pool_id, set_job, "backup");
  }
  return null;
}

function shouldSkipBackupPool(pool_id, active_pool) {
  return pool_id == global.opt.pool_ids.donate || pool_id == active_pool || !alivePoolJob(pool_id);
}

function nextNonDonatePool(pool_id) {
  const donate_pool = global.opt.pool_ids.donate;
  let next_pool = pool_id + 1;
  if (next_pool < Object.keys(global.opt.pools).length) return next_pool;
  next_pool = 0;
  return next_pool === donate_pool ? next_pool + 1 : next_pool;
}

// switch active pool to the next available pool (except donate pool)
// preferring pool with already alive socket if any
module.exports.switch_pool = function(pool_id, set_job) {
  pool_close_wait(pool_id);

  const active_pool  = global.opt.pool_ids.active;
  const donate_pool  = global.opt.pool_ids.donate;

  // do not care about not active pool
  if (pool_id !== global.opt.pool_ids.active) return;

  // select already alive pool if possible, except donate pool (starting from primary pool)
  const alive_job = reactivateAlivePoolIfAny(active_pool, set_job);
  if (alive_job) return alive_job;

  // do not continue to mine on donate pool if all other pools are dead
  if (global.opt.pool_ids.active === donate_pool) h.messageWorkers({type: "pause"});

  // select the next available pool except donate pool
  pool_id = nextNonDonatePool(pool_id);
  return module.exports.connect_pool_throttle(global.opt.pool_ids.active = pool_id, set_job);
};

function reactivateAlivePoolIfAny(active_pool, set_job) {
  return reactivatePrimaryPool(set_job) || reactivateBackupPool(active_pool, set_job);
}

function handleRavenSetTarget(pool_id, json) {
  global.opt.pools[pool_id].raven_target = hexWithoutPrefix(json.params[0]);
}

function handleEthSetTarget(pool_id, json) {
  global.opt.pools[pool_id].eth_target = hexWithoutPrefix(json.params[0]);
}

function handleSetDifficulty(pool_id, json) {
  global.opt.pools[pool_id].eth_difficulty = json.params[0];
}

function jobFromPoolMessage(pool_id, json) {
  const pool = global.opt.pools[pool_id];
  if (isJobNotification(json)) {
    if (!pool.logged_in) return null;
    pool.submit_mode = null;
    return json.params;
  }
  if (poolProtocol(pool) === "raven" && isRavenJobNotification(json)) {
    if (!pool.logged_in) return null;
    const headerHash = hexWithoutPrefix(json.params[1]);
    const seedHash = hexWithoutPrefix(json.params[2]);
    const target = ravenTarget(pool, json.params[3]);
    pool.submit_mode = "raven";
    return {
      algo: json.algo || "kawpow",
      blob: ravenBlob(headerHash, pool),
      header_hash: headerHash,
      seed_hash: seedHash,
      target: target,
      job_id: json.params[0],
      height: json.params[5],
      nonce: ravenNonce(pool),
      nicehash_mask: ravenNonceMask(pool),
      noncebytes: 8,
      nonceoffset: 32,
    };
  }
  if (poolProtocol(pool) === "eth" && isEthJobNotification(json)) {
    if (!pool.logged_in) return null;
    const seedHash = hexWithoutPrefix(json.params[1]);
    const headerHash = hexWithoutPrefix(json.params[2]);
    pool.submit_mode = "eth";
    return {
      algo: json.algo || (global.opt.job && global.opt.job.algo) || "etchash",
      blob: ethBlob(headerHash, pool),
      header_hash: headerHash,
      seed_hash: seedHash,
      target: ethTarget(pool),
      job_id: json.params[0],
      nonce: ethNonce(pool),
      nicehash_mask: ethNonceMask(pool),
      noncebytes: 8,
      nonceoffset: 32,
    };
  }
  if (usesEthProxy(pool) && isEthProxyWork(json)) {
    if (!pool.logged_in) return null;
    const headerHash = hexWithoutPrefix(json.result[0]);
    const seedHash = hexWithoutPrefix(json.result[1]);
    const target = hexWithoutPrefix(json.result[2]).padStart(64, "0");
    pool.submit_mode = "ethproxy";
    return {
      algo: json.algo || (global.opt.job && global.opt.job.algo) || "etchash",
      blob: ethBlob(headerHash, pool),
      header_hash: headerHash,
      seed_hash: seedHash,
      target: target,
      job_id: headerHash,
      height: parseHexHeight(json.result[3]),
      nonce: ethNonce(pool),
      nicehash_mask: ethNonceMask(pool),
      noncebytes: 8,
      nonceoffset: 32,
    };
  }
  if (poolProtocol(pool) === "erg" && isErgJobNotification(json)) {
    if (!pool.logged_in) return null;
    const headerHash = hexWithoutPrefix(json.params[2]);
    const ntime = hexWithoutPrefix(json.params[7]);
    pool.submit_mode = "erg";
    const job = {
      algo: json.algo || (global.opt.job && global.opt.job.algo) || "autolykos2",
      blob: ethBlob(headerHash, pool),
      header_hash: headerHash,
      target: ergTarget(json.params[6]),
      job_id: json.params[0],
      height: json.params[1],
      ntime: ntime,
      nonce: ethNonce(pool),
      nicehash_mask: ethNonceMask(pool),
      noncebytes: 8,
      nonceoffset: 32,
    };
    rememberErgSubmitJob(pool, job);
    return job;
  }
  if (isLoginJob(json)) { // login job
    pool.logged_in = true;
    pool.submit_mode = null;
    if ("id" in json.result) pool.worker_id = json.result.id;
    rememberPoolExtraNonce(pool_id, json.result);
    applyLoginExtensions(pool_id, json.result.extensions);
    return loginJobWithResultMetadata(json.result);
  }
  return null;
}

function jobTargetWork(job) {
  if (!job.target) return null;
  if (job.algo === "kawpow") return h.kawpowTarget2diff(job.target);
  if (job.algo === "etchash" || job.algo === "autolykos2") return h.target256ToWork(job.target);
  return h.target2diff(job.target);
}

function jobTargetDescription(job) {
  const work = jobTargetWork(job);
  if (work !== null) return h.formatHashCount(work) + "/share target";
  return job.difficulty + " diff";
}

function validateJobTarget(job) {
  if (job.target) jobTargetWork(job);
}

function activatePoolForJob(pool_id, active_pool) {
  // only switch active pool once for its first job here
  if (pool_id === active_pool || global.opt.pools[pool_id].last_job) return;
  const activator = poolActivator(pool_id);
  if (activator) activator(pool_id, active_pool);
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
  validateJobTarget(job);

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

function handleLoginResponse(pool_id, is_err, is_ok, err_msg, json) {
  if (is_err || json.result === false) {
    global.opt.pools[pool_id].logged_in = false;
    return pool_log_err(pool_id, "Login to the pool failed" + (err_msg || ": Login rejected"));
  }
  if (is_ok) {
    global.opt.pools[pool_id].logged_in = true;
    return pool_log(pool_id, "Login to the pool succeeded");
  }
}

function handleSubscribeResponse(pool_id, is_err, is_ok, err_msg, json) {
  if (is_err) return pool_log_err(pool_id, "Subscribe to the pool failed" + err_msg);
  if (!is_ok) return;
  rememberSubscribeExtraNonce(pool_id, json.result);
  const pool = global.opt.pools[pool_id];
  pool.pending_authorize = true;
  return module.exports.pool_write(pool_id, {
    jsonrpc: "2.0", id: 2, method: "mining.authorize", params: [pool.login, pool.pass]
  });
}

function handleEthProxyLoginResponse(pool_id, is_err, is_ok, err_msg, json) {
  return handleLoginResponse(pool_id, is_err, is_ok, err_msg, json);
}

function handleAuthorizeResponse(pool_id, is_err, is_ok, err_msg, json) {
  global.opt.pools[pool_id].pending_authorize = false;
  if (!is_err && json.result === true) {
    global.opt.pools[pool_id].logged_in = true;
    return pool_log(pool_id, "Login to the pool succeeded");
  }
  global.opt.pools[pool_id].logged_in = false;
  return pool_log_err(pool_id, "Login to the pool failed" + (err_msg || ": Authorization rejected"));
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
  const err_msg = poolErrorMessage(json, is_err);
  const is_ok   = "result" in json && json.result !== null && json.result !== false;
  const handler = poolResponseHandler(pool_id, json.id);
  const result = handler(pool_id, is_err, is_ok, err_msg, json);
  if (!is_err && is_ok) rememberPoolResponseMetadata(pool_id, json.result);
  return result;
}

function rememberPoolResponseMetadata(pool_id, result) {
  if (!isObject(result)) return;

  const pool = global.opt.pools[pool_id];
  if ("id" in result) pool.worker_id = result.id;
  rememberPoolExtraNonce(pool_id, result);
  rememberPoolProtocol(pool_id, result);
  applyLoginExtensions(pool_id, result.extensions);
}

function ignorePoolResponse() {}

function poolResponseHandler(pool_id, id) {
  const pool = global.opt.pools[pool_id];
  if (usesMiningSubscribe(pool)) {
    if (id === 1) return handleSubscribeResponse; // mining.subscribe response
    if (id === 2) return pool.pending_authorize ? handleAuthorizeResponse : ignorePoolResponse;
  } else if (usesEthProxy(pool)) {
    if (id === 1) return handleEthProxyLoginResponse; // eth_submitLogin response
    if (id === 2) return ignorePoolResponse; // legacy keepalive response
  } else {
    if (id === 1) return handleLoginResponse; // login response
    if (id === 2) return ignorePoolResponse; // keepalive response
  }
  return handleShareResponse; // share submit response
}

function pool_message(pool_id, json, set_job) {
  if (isRavenSetTargetNotification(json)) {
    if (poolProtocol(global.opt.pools[pool_id]) === "eth") return handleEthSetTarget(pool_id, json);
    return handleRavenSetTarget(pool_id, json);
  }
  if (isSetDifficultyNotification(json)) return handleSetDifficulty(pool_id, json);
  const job = jobFromPoolMessage(pool_id, json);
  if (job) return handlePoolJob(pool_id, job, set_job);
  if ("id" in json) return handlePoolResponse(pool_id, json);

  pool_log1(pool_id, "Unknown message from the pool: " + JSON.stringify(json));
}

function poolTypeStr(pool_id) {
  switch (pool_id) {
    case global.opt.pool_ids.primary: return "primary";
    case global.opt.pool_ids.donate:  return "donate";
    default:                          return "backup";
  }
}

function connectSocket(pool) {
  return pool.is_tls ?
    tls.connect(pool.port, pool.url, { rejectUnauthorized: pool.tls_verify === true }) :
    net.connect(pool.port, pool.url);
}

function poolLoginParams(pool) {
  const algos = [];
  const algo_perfs = {};
  for (const algo in global.opt.algo_params) {
    if (!global.opt.algo_params[algo].perf) continue;
    algos.push(algo);
    algo_perfs[algo] = global.opt.algo_params[algo].perf;
  }
  return {
    login: pool.login, pass: pool.pass, agent: o.agent_str,
    algo: algos, "algo-perf": algo_perfs
  };
}

function parsePoolLine(pool_id, message) {
  try {
    return JSON.parse(message);
  } catch (e) {
    pool_log_err(pool_id, "Can't parse message from the pool: " + message);
    return null;
  }
}

function processPoolJson(pool_id, json, set_job, pool_err) {
  pool_log2(pool_id, "Got from the pool: " + JSON.stringify(json));
  try {
    pool_message(pool_id, json, set_job);
  } catch (err) {
    pool_err(pool_log_str(pool_id,
      "Can't process message from the pool: " + (err && err.message ? err.message : err)
    ));
    return true;
  }
  return false;
}

function handlePoolLines(pool_id, messages, set_job, pool_err) {
  for (const message of messages) {
    if (message.trim() === '') continue;
    const json = parsePoolLine(pool_id, message);
    if (shouldStopPoolLineProcessing(pool_id, json, set_job, pool_err)) return true;
  }
  return false;
}

function shouldStopPoolLineProcessing(pool_id, json, set_job, pool_err) {
  return json ? processPoolJson(pool_id, json, set_job, pool_err) : false;
}

function poolErrorHandler(pool_id, socket, set_job) {
  return function(message) {
    if (!clear_pool_connection(pool_id, socket)) return;
    h.log_err(message);
    return module.exports.switch_pool(pool_id, set_job);
  };
}

function canRetryAsEthProxy(pool) {
  return !pool.protocol && poolProtocol(pool) === "eth" &&
         global.opt.job && global.opt.job.algo === "etchash";
}

function retryAsEthProxy(pool_id, socket, set_job) {
  const pool = global.opt.pools[pool_id];
  if (!canRetryAsEthProxy(pool)) return false;
  pool_log1(pool_id, "No initial Etchash job, retrying the pool with ethproxy protocol");
  pool.inferred_protocol = "ethproxy";
  clear_pool_connection(pool_id, socket);
  module.exports.connect_pool_throttle(pool_id, set_job);
  return true;
}

function scheduleInitialJobTimeout(pool_id, socket, set_job, pool_err) {
  setTimeout(function() {
    if (!isCurrentPoolSocket(pool_id, socket)) return;
    if (global.opt.pools[pool_id].last_job) return;
    if (retryAsEthProxy(pool_id, socket, set_job)) return;
    return pool_err(pool_log_str(pool_id,
      "No initial job from from " + pool_str(pool_id) + " pool"
    ));
  }, global.opt.pool_time.first_job_wait * 1000);
}

function handlePoolConnect(pool_id, socket, pool) {
  if (!isCurrentPoolSocket(pool_id, socket)) return;
  pool_log1(pool_id, "Connected to the pool");
  const request = usesMiningSubscribe(pool) ?
    { jsonrpc: "2.0", id: 1, method: "mining.subscribe", params: [o.agent_str] } :
    usesEthProxy(pool) ?
      { jsonrpc: "2.0", id: 1, method: "eth_submitLogin", params: [pool.login, pool.pass] } :
      { jsonrpc: "2.0", id: 1, method: "login", params: poolLoginParams(pool) };
  return module.exports.pool_write(pool_id, request);
}

function splitPoolMessages(pool_data_buff) {
  const messages = pool_data_buff.split('\n');
  const incomplete_line = pool_data_buff.slice(-1) === '\n' ? '' : messages.pop();
  return { messages, incomplete_line };
}

function readPoolData(pool_id, socket, pool_data_buff, data, pool_err) {
  if (!isCurrentPoolSocket(pool_id, socket)) return null;
  const next_buff = pool_data_buff + data;
  if (next_buff.length <= max_pool_data_buffer) return next_buff;
  pool_err(pool_log_str(pool_id, "Pool message buffer limit exceeded"));
  return null;
}

function hasCompletePoolLine(pool_data_buff) {
  return pool_data_buff.indexOf('\n') !== -1;
}

function poolDataHandler(pool_id, socket, set_job, pool_err) {
  let pool_data_buff = "";
  return function(data) {
    const next_buff = readPoolData(pool_id, socket, pool_data_buff, data, pool_err);
    if (next_buff === null) return;
    if (!hasCompletePoolLine(next_buff)) {
      pool_data_buff = next_buff;
      return;
    }
    pool_data_buff = next_buff;
    const split = splitPoolMessages(pool_data_buff);
    if (handlePoolLines(pool_id, split.messages, set_job, pool_err)) return;
    pool_data_buff = split.incomplete_line;
  };
}

function connect_pool(pool_id, set_job) {
  const pool = global.opt.pools[pool_id];

  // do not connect to already connected pools
  if (pool.socket) return;

  pool_log(pool_id, "Connecting to " + poolTypeStr(pool_id) + " " + pool_str(pool_id) + " pool");
  global.opt.pools[pool_id].last_connect_time = Date.now();
  const socket = global.opt.pools[pool_id].socket = connectSocket(pool);
  global.opt.pools[pool_id].last_job = null;

  const pool_err = poolErrorHandler(pool_id, socket, set_job);
  scheduleInitialJobTimeout(pool_id, socket, set_job, pool_err);

  socket.on("connect", function () {
    handlePoolConnect(pool_id, socket, pool);
  });

  socket.on("data", poolDataHandler(pool_id, socket, set_job, pool_err));

  socket.on("end", function() {
    return pool_err(pool_log_str(pool_id, "Socket closed from the pool"));
  });

  socket.on("error", function() {
    return pool_err(pool_log_str(pool_id, "Socket error from the pool"));
  });
}

module.exports.connect_pool_throttle = function(pool_id, set_job) {
  const pool = global.opt.pools[pool_id];
  const wait_time = global.opt.pool_time.connect_throttle * 1000 -
                    (Date.now() - pool.last_connect_time);
  if (wait_time > 0) {
    pool_log(pool_id, "Waiting " + Math.floor(wait_time / 1000) + "s to connect to the pool");
    return setTimeout(connect_pool, wait_time, pool_id, set_job);
  } else return connect_pool(pool_id, set_job);
};
