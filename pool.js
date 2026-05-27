// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

"use strict";

const path = require("path");
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
  if (socket && pool.socket !== socket) return false;
  if (pool.keepalive !== null) clearTimeout(pool.keepalive);
  pool.keepalive = null;
  if (pool.socket) pool.socket.destroy();
  pool.socket   = null;
  pool.last_job = null;
  return true;
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
        global.opt.pools[pool_id].socket !== socket) return;
    pool_log1(pool_id, "Soft closed the pool connection");
    clear_pool_connection(pool_id, socket);
  }, global.opt.pool_time.close_wait * 1000);
}

// switch active pool to the next available pool (except donate pool)
// preferring pool with already alive socket if any
module.exports.switch_pool = function(pool_id, set_job) {
  pool_close_wait(pool_id);

  const active_pool  = global.opt.pool_ids.active;
  const primary_pool = global.opt.pool_ids.primary;
  const donate_pool  = global.opt.pool_ids.donate;

  // do not care about not active pool
  if (pool_id !== global.opt.pool_ids.active) return;

  // select already alive pool if possible, except donate pool (starting from primary pool)
  if (global.opt.pools[primary_pool].last_job) {
    pool_log(primary_pool, "Making the primary pool " + pool_str(primary_pool) + " active again");
    return set_job(global.opt.pools[global.opt.pool_ids.active = primary_pool].last_job);
  }
  for (const pool_id2 in global.opt.pools) {
    // === will not work here since here we are comparing strings and integers
    if (pool_id2 == donate_pool || pool_id2 == active_pool) continue;
    if (global.opt.pools[pool_id2].last_job) {
      pool_log(pool_id2, "Making backup pool " + pool_str(pool_id2) + " active again");
      global.opt.pool_ids.active = pool_id2;
      return set_job(global.opt.pools[global.opt.pool_ids.active = pool_id2].last_job);
    }
  }

  // do not continue to mine on donate pool if all other pools are dead
  if (global.opt.pool_ids.active === donate_pool) h.messageWorkers({type: "pause"});

  // select the next available pool except donate pool
  ++ pool_id;
  if (pool_id >= Object.keys(global.opt.pools).length) {
    pool_id = 0;
    if (pool_id === donate_pool) ++ pool_id;
  }
  return module.exports.connect_pool_throttle(global.opt.pool_ids.active = pool_id, set_job);
};

function pool_message(pool_id, json, set_job) {
  const active_pool = global.opt.pool_ids.active;

  let job;
  if ("method" in json && json.method === "job" &&
      "params" in json && json.params instanceof Object) {
    job = json.params;
  // login job
  } else if ("result" in json && json.result instanceof Object &&
             "job" in json.result && json.result.job instanceof Object) {
    job = json.result.job;
    if ("id" in json.result) global.opt.pools[pool_id].worker_id = json.result.id;
    if ("extensions" in json.result && Array.isArray(json.result.extensions)) {
      if (json.result.extensions.includes("nicehash"))
        global.opt.pools[pool_id].is_nicehash = true;
      if (json.result.extensions.includes("keepalive"))
        global.opt.pools[pool_id].is_keepalive = true;
    }
  }

  if (job) {
    // only switch active pool once for its first job here
    if (pool_id !== active_pool && !global.opt.pools[pool_id].last_job) switch (pool_id) {
      case global.opt.pool_ids.primary:
        pool_log(pool_id, "Switching active pool to primary " + pool_str(pool_id) + " pool");
        pool_close_wait(active_pool);
        global.opt.pool_ids.active = pool_id;
        break;
      case global.opt.pool_ids.donate:
        pool_log(pool_id, "Switching active pool to donate " + pool_str(pool_id) + " pool");
        global.opt.pool_ids.active = pool_id;
        break;
    }
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
    return;

  } else if ("id" in json) {
    const is_err  = "error" in json && json.error !== null;
    const err_msg = is_err && json.error instanceof Object && "message" in json.error &&
                    typeof json.error.message === "string" ? ": " + json.error.message : "";
    const is_ok   = "result" in json && json.result !== null;
    switch (json.id) {
      case 1: // login response
        if (is_err) {
          return pool_log_err(pool_id, "Login to the pool failed" + err_msg)
        } else if (is_ok) {
          return pool_log(pool_id, "Login to the pool succeeded");
        }
        break;

      case 2: return; // keepalive response

      default: // share submit response
        function stats_str() {
          return "(" + global.opt.pools[pool_id].good_shares + "/" +
                       global.opt.pools[pool_id].bad_shares + ")";
        }
        if (is_err) {
          ++ global.opt.pools[pool_id].bad_shares;
          return pool_log_err(pool_id, "Share rejected by the pool " + stats_str() + err_msg);
        } else if (is_ok) {
          ++ global.opt.pools[pool_id].good_shares;
          return pool_log(pool_id, "Share accepted by the pool " + stats_str());
        }
        break;
    }
  }

  pool_log1(pool_id, "Unknown message from the pool: " + JSON.stringify(json));
}

function connect_pool(pool_id, set_job) {
  const pool = global.opt.pools[pool_id];

  // do not connect to already connected pools
  if (pool.socket) return;

  let pool_type_str;
  switch (pool_id) {
    case global.opt.pool_ids.primary: pool_type_str = "primary"; break;
    case global.opt.pool_ids.donate:  pool_type_str = "donate"; break;
    default:                          pool_type_str = "backup";
  }
  pool_log(pool_id, "Connecting to " + pool_type_str + " " + pool_str(pool_id) + " pool");
  global.opt.pools[pool_id].last_connect_time = Date.now();
  const socket = global.opt.pools[pool_id].socket = pool.is_tls ?
    tls.connect(pool.port, pool.url, { rejectUnauthorized: pool.tls_verify === true }) :
    net.connect(pool.port, pool.url);
  global.opt.pools[pool_id].last_job = null;

  const pool_err = function(message) {
    if (!clear_pool_connection(pool_id, socket)) return;
    h.log_err(message);
    return module.exports.switch_pool(pool_id, set_job);
  };

  setTimeout(function() {
    if (global.opt.pools[pool_id].socket !== socket) return;
    if (!global.opt.pools[pool_id].last_job) return pool_err(pool_log_str(pool_id,
      "No initial job from from " + pool_str(pool_id) + " pool"
    ));
  }, global.opt.pool_time.first_job_wait * 1000);

  socket.on("connect", function () {
    if (global.opt.pools[pool_id].socket !== socket) return;
    pool_log1(pool_id, "Connected to the pool");
    let algos = [];
    let algo_perfs = {};
    for (const algo in global.opt.algo_params) {
      if (!global.opt.algo_params[algo].perf) continue;
      algos.push(algo);
      algo_perfs[algo] = global.opt.algo_params[algo].perf;
    }
    module.exports.pool_write(pool_id, {
      jsonrpc: "2.0", id: 1, method: "login", params: {
        login: pool.login, pass: pool.pass, agent: o.agent_str,
        algo: algos, "algo-perf": algo_perfs
      }
    });
  });

  let pool_data_buff = "";

  socket.on("data", function (data) {
    if (global.opt.pools[pool_id].socket !== socket) return;
    pool_data_buff += data;
    if (pool_data_buff.length > max_pool_data_buffer)
      return pool_err(pool_log_str(pool_id, "Pool message buffer limit exceeded"));
    if (pool_data_buff.indexOf('\n') === -1) return;
    let messages = pool_data_buff.split('\n');
    let incomplete_line = pool_data_buff.slice(-1) === '\n' ? '' : messages.pop();
    for (let i = 0; i < messages.length; i++) {
      let message = messages[i];
      if (message.trim() === '') continue;
      let json;
      try {
        json = JSON.parse(message);
      } catch (e) {
        pool_log_err(pool_id, "Can't parse message from the pool: " + message);
        continue;
      }
      pool_log2(pool_id, "Got from the pool: " + JSON.stringify(json));
      try {
        pool_message(pool_id, json, set_job);
      } catch (err) {
        return pool_err(pool_log_str(pool_id,
          "Can't process message from the pool: " + (err && err.message ? err.message : err)
        ));
      }
    }
    pool_data_buff = incomplete_line;
  });

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
    pool_log(pool_id, "Waiting " + parseInt(wait_time/1000) + "s to connect to the pool");
    return setTimeout(connect_pool, wait_time, pool_id, set_job);
  } else return connect_pool(pool_id, set_job);
};
