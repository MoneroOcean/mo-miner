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

module.exports.pool_write = function(pool_id, json) {
  const message = JSON.stringify(json);
  if (global.opt.pools[pool_id].socket) {
    pool_log2(pool_id, "Sent to the pool: " + message);
    global.opt.pools[pool_id].socket.write(message + "\n");
    // sends keepalive if no submit/keepalive to pool for more than global.opt.pool_time.keepalive
    if (global.opt.pools[pool_id].is_keepalive) {
      if (global.opt.pools[pool_id].keepalive !== null)
        clearTimeout(global.opt.pools[pool_id].keepalive);
      global.opt.pools[pool_id].keepalive = setTimeout(function() {
        global.opt.pools[pool_id].keepalive = null;
        module.exports.pool_write(pool_id, {
          jsonrpc: "2.0", id: 2, method: "keepalive", params: {}
        });
      }, global.opt.pool_time.keepalive * 1000);
    }
  } else {
    pool_log2(pool_id, "Sent to the closed pool socket: " + message);
  }
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

function isObject(value) {
  return value instanceof Object;
}

function isJobNotification(json) {
  return json.method === "job" && isObject(json.params);
}

function isLoginJob(json) {
  return isObject(json.result) && isObject(json.result.job);
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

function jobFromPoolMessage(pool_id, json) {
  if (isJobNotification(json)) return json.params;
  if (isLoginJob(json)) { // login job
    if ("id" in json.result) global.opt.pools[pool_id].worker_id = json.result.id;
    applyLoginExtensions(pool_id, json.result.extensions);
    return json.result.job;
  }
  return null;
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

  global.opt.pools[pool_id].last_job = job;
  if (pool_id === global.opt.pool_ids.active) {
    const last_job = set_job(job);
    pool_log(pool_id, "Got new " + last_job.algo + " algo job with " +
                     (job.target ? h.target2diff(job.target) : job.difficulty) + " diff" +
	             (job.height ? " and " + job.height + " height" : "")
    );
  } else {
    pool_log2(pool_id, "Storing not active pool job " + JSON.stringify(job));
  }
}

function handleLoginResponse(pool_id, is_err, is_ok, err_msg) {
  if (is_err) return pool_log_err(pool_id, "Login to the pool failed" + err_msg);
  if (is_ok) return pool_log(pool_id, "Login to the pool succeeded");
}

function handleShareResponse(pool_id, is_err, is_ok, err_msg) {
  if (is_err) {
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
  const is_ok   = "result" in json && json.result !== null;
  const handler = poolResponseHandler(json.id);
  return handler(pool_id, is_err, is_ok, err_msg);
}

function poolResponseHandler(id) {
  if (id === 1) return handleLoginResponse; // login response
  if (id === 2) return function() {}; // keepalive response
  return handleShareResponse; // share submit response
}

function pool_message(pool_id, json, set_job) {
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

function scheduleInitialJobTimeout(pool_id, socket, pool_err) {
  setTimeout(function() {
    if (!isCurrentPoolSocket(pool_id, socket)) return;
    if (!global.opt.pools[pool_id].last_job) return pool_err(pool_log_str(pool_id,
      "No initial job from from " + pool_str(pool_id) + " pool"
    ));
  }, global.opt.pool_time.first_job_wait * 1000);
}

function handlePoolConnect(pool_id, socket, pool) {
  if (!isCurrentPoolSocket(pool_id, socket)) return;
  pool_log1(pool_id, "Connected to the pool");
  module.exports.pool_write(pool_id, {
    jsonrpc: "2.0", id: 1, method: "login", params: poolLoginParams(pool)
  });
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
  scheduleInitialJobTimeout(pool_id, socket, pool_err);

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
