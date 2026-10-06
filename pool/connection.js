"use strict";

module.exports = ({
  h, o, net, tls, systemNetConnect, systemTlsConnect, max_pool_data_buffer,
  clear_pool_connection, isCurrentPoolSocket, pearlhashUsesSubscribe, normalizeAlgoName,
  poolProtocol, pool_log, pool_log1, pool_log2, pool_log_json, pool_log_err, pool_log_str,
  poolErrorText,
  pool_message, pool_str, usesCortex, usesEthProxy, usesIronfish, usesMiningSubscribe,
  poolWrite, switchPool,
}) => {

  function poolTypeStr(pool_id) {
    switch (pool_id) {
      case global.opt.pool_ids.primary: return "primary";
      case global.opt.pool_ids.donate:  return "donate";
      default:                          return "backup";
    }
  }

  function connectSocket(pool) {
    const connect = pool.is_tls ? tls.connect : net.connect;
    const systemConnect = pool.is_tls ? systemTlsConnect : systemNetConnect;
    const loopback = pool.url === "127.0.0.1" || pool.url === "::1";
    if (process.env.MOM_TEST_NO_POOL_NETWORK === "1" && connect === systemConnect && !loopback) {
      throw new Error("Pool network access is disabled during mom correctness tests");
    }
    return pool.is_tls ?
      connect(pool.port, pool.url, { rejectUnauthorized: pool.tls_verify === true }) :
      connect(pool.port, pool.url);
  }

  function poolLoginParams(pool) {
    const algos = [];
    /** @type {Record<string, number>} */
    const algo_perfs = {};
    const algoParams = pool.algo_params || global.opt.algo_params;
    const fixedAlgo = normalizeAlgoName(global.opt.job.algo);
    for (const [algo, params] of Object.entries(algoParams)) {
      const perf = params?.perf;
      const measured = typeof perf === "number" && Number.isFinite(perf) && perf > 0;
      // A fixed unbenchmarked job still advertises its capability so a proxy cannot assign a
      // default algorithm; omitting its unknown rate is safer than inventing one.
      if (!measured && algo !== fixedAlgo) {continue;}
      // The pool's historical KawPow identifier is kawpow1, and cycle-algorithm performance is
      // reported in solutions rather than raw edges.
      const poolAlgo = algo === "kawpow" ? "kawpow1" : algo;
      algos.push(poolAlgo);
      if (measured) {algo_perfs[poolAlgo] = algo === "c29" ? perf / 42 : perf;}
    }
    pool.requested_algos = algos.map((algo) => normalizeAlgoName(algo) || algo);
    pool.requested_extensions = algos.length ? ["mo-native"] : [];
    // PearlHash seed slots are meaningful only when the native object-login capability is
    // advertised.  The pool-side extension filter will retain this request only after the pool
    // acknowledges it; until then any proxy-only slot/stride fields stay inert.
    if (pool.requested_algos.includes("pearlhash")) {
      pool.requested_extensions.push("pearl-seed-split");
    }
    // The proxy treats submit-result as a connection-wide promise, including after a switch.
    if (algos.length) {
      pool.requested_extensions.push("submit-result");
    }
    return {
      login: pool.login, pass: pool.pass, agent: o.agent_str,
      algo: algos, "algo-perf": algo_perfs,
      ...(pool.requested_extensions.length ? {extensions: pool.requested_extensions} : {}),
    };
  }

  function pearlhashAuthorizeParams(pool) {
    const login = typeof pool.login === "string" ? pool.login : "";
    const separator = login.lastIndexOf(".");
    const hasWorker = separator > 0 && separator < login.length - 1;
    const wallet = hasWorker ? login.slice(0, separator) : login;
    const embeddedWorker = hasWorker ? login.slice(separator + 1) : "";
    const worker = typeof pool.worker === "string" && pool.worker.length > 0
      ? pool.worker : embeddedWorker || "mom";
    return {wallet, worker, pass: pool.pass};
  }

  function parsePoolLine(pool_id, message) {
    try {
      const json = JSON.parse(message);
      attachKaspaPrecisePrePow(json, message);
      return json;
    } catch {
      pool_log_err(pool_id, "Can't parse message from the pool: " + message);
      return null;
    }
  }

  // The Kaspa mining.notify pre-pow words are 64-bit unsigned ints that exceed Number.MAX_SAFE_INTEGER,
  // so JSON.parse silently rounds them. Re-extract the exact integer literals from the raw line (decimal
  // strings) and stash them as BigInt-safe fields the kaspa job builder reads instead of the lossy array.
  function attachKaspaPrecisePrePow(json, message) {
    if (!json || json.method !== "mining.notify" || !Array.isArray(json.params) ||
      !Array.isArray(json.params[1])) {return;}
    // params: [ "jobId", [w0,w1,w2,w3], timestamp ]. Capture the bracketed word list + the timestamp.
    const m = message.match(/"params"\s*:\s*\[\s*"[^"]*"\s*,\s*\[([^\]]*)\]\s*,\s*(\d+)/);
    if (!m) {return;}
    const words = m[1].split(",").map((s) => s.trim()).filter((s) => /^\d+$/.test(s));
    if (words.length < 4) {return;}
    // Store as exact decimal strings (NOT BigInt) so the debug logger's JSON.stringify(json) still works.
    json.__kaspa_words = words.slice(0, 4);
    json.__kaspa_timestamp = m[2];
  }

  function processPoolJson(pool_id, json, set_job, pool_err) {
    pool_log_json(pool_id, 2, "Got from the pool: ", json);
    try {
      pool_message(pool_id, json, set_job);
    } catch (err) {
      pool_err(pool_log_str(pool_id, "Can't process message from the pool") +
        poolErrorText(pool_id, err));
      return true;
    }
    return false;
  }

  function handlePoolLines(pool_id, messages, set_job, pool_err) {
    for (const message of messages) {
      if (message.trim() === "") {continue;}
      const json = parsePoolLine(pool_id, message);
      if (json && processPoolJson(pool_id, json, set_job, pool_err)) {return true;}
    }
    return false;
  }

  function poolErrorHandler(pool_id, socket, set_job) {
    return function(message) {
      if (!clear_pool_connection(pool_id, socket)) {return;}
      h.log_err(message);
      return switchPool(pool_id, set_job);
    };
  }

  function canRetryAsEthProxy(pool) {
    return !pool.protocol && poolProtocol(pool) === "eth" &&
         global.opt.job && global.opt.job.algo === "etchash";
  }

  function retryAsEthProxy(pool_id, socket, set_job) {
    const pool = global.opt.pools[pool_id];
    if (!canRetryAsEthProxy(pool)) {return false;}
    pool_log1(pool_id, "No initial Etchash job, retrying the pool with ethproxy protocol");
    pool.inferred_protocol = "ethproxy";
    clear_pool_connection(pool_id, socket);
    connectPoolThrottle(pool_id, set_job);
    return true;
  }

  function scheduleInitialJobTimeout(pool_id, socket, set_job, pool_err) {
    setTimeout(function() {
      if (!isCurrentPoolSocket(pool_id, socket)) {return;}
      if (global.opt.pools[pool_id].last_job) {return;}
      if (retryAsEthProxy(pool_id, socket, set_job)) {return;}
      return pool_err(pool_log_str(pool_id,
        "No initial job from " + pool_str(pool_id) + " pool"
      ));
    }, global.opt.pool_time.first_job_wait * 1000);
  }

  function handlePoolConnect(pool_id, socket, pool) {
    if (!isCurrentPoolSocket(pool_id, socket)) {return;}
    pool_log1(pool_id, "Connected to the pool");
    if (usesIronfish(pool)) {
    // Iron Fish custom OBJECT stratum: a single mining.subscribe push carries the wallet+worker
    // (publicAddress) and the agent; the pool replies with mining.subscribed (handled by method).
    // No separate authorize. extend:["mining.submitted"] requests the submit-result push.
      return poolWrite(pool_id, {
        id: 1, method: "mining.subscribe",
        body: { version: 3, agent: o.agent_str, publicAddress: pool.login, extend: ["mining.submitted"] }
      });
    }
    if (usesCortex(pool)) {
      // Live Cortex pools use request-style JSON-RPC rather than mining.subscribe/authorize.
      pool.pending_cortex_login = true;
      return poolWrite(pool_id, {
        id: 72, jsonrpc: "2.0", method: "ctxc_submitLogin",
        params: [pool.login], worker: pool.worker || "mom",
      });
    }
    if (pool.use_subscribe === false) {
      return poolWrite(pool_id, {
        jsonrpc: "2.0", id: 1, method: "login", params: poolLoginParams(pool)
      });
    }
    if (poolProtocol(pool) === "conflux") {
      return poolWrite(pool_id, {
        jsonrpc: "2.0", id: 1, method: "mining.subscribe",
        params: [pool.login, ""]
      });
    }
    if (pearlhashUsesSubscribe(pool)) {
    // PearlHash subscribe dialect: send subscribe AND authorize back-to-back. mining.subscribe is just a
    // handshake nicety -- HeroMiners acks it (result:true), LuckyPool rejects it ("method not
    // supported") and drops the connection if no authorize follows promptly. So don't wait on the
    // subscribe reply; authorize immediately. authorize takes OBJECT params {wallet,worker,pass}.
      pool.pending_authorize = true;
      poolWrite(pool_id, { jsonrpc: "2.0", id: 1, method: "mining.subscribe", params: [o.agent_str] });
      return poolWrite(pool_id, {
        jsonrpc: "2.0", id: 2, method: "mining.authorize",
        params: pearlhashAuthorizeParams(pool)
      });
    }
    if (poolProtocol(pool) === "beam") {
    // Beam JSON-RPC: a single `login` with the wallet/api_key (the pool replies with a `result`
    // message carrying code:0 and the nonceprefix). No mining.subscribe handshake.
      return poolWrite(pool_id, {
        jsonrpc: "2.0", id: "login", method: "login", api_key: pool.login
      });
    }
    const request = poolProtocol(pool) === "xelis" ?
      {jsonrpc: "2.0", id: 1, method: "mining.subscribe", params: [o.agent_str, ["xel/v3"]]} :
      usesMiningSubscribe(pool) ?
      { jsonrpc: "2.0", id: 1, method: "mining.subscribe", params: [o.agent_str] } :
      usesEthProxy(pool) ?
        { jsonrpc: "2.0", id: 1, method: "eth_submitLogin", params: [pool.login, pool.pass] } :
        { jsonrpc: "2.0", id: 1, method: "login", params: poolLoginParams(pool) };
    return poolWrite(pool_id, request);
  }

  function splitPoolMessages(pool_data_buff) {
    const messages = pool_data_buff.split("\n");
    const incomplete_line = pool_data_buff.slice(-1) === "\n" ? "" : messages.pop();
    return { messages, incomplete_line };
  }

  function readPoolData(pool_id, socket, pool_data_buff, data, pool_err) {
    if (!isCurrentPoolSocket(pool_id, socket)) {return null;}
    const next_buff = pool_data_buff + data;
    if (next_buff.length <= max_pool_data_buffer) {return next_buff;}
    pool_err(pool_log_str(pool_id, "Pool message buffer limit exceeded"));
    return null;
  }

  function hasCompletePoolLine(pool_data_buff) {
    return pool_data_buff.indexOf("\n") !== -1;
  }

  function poolDataHandler(pool_id, socket, set_job, pool_err) {
    let pool_data_buff = "";
    return function(data) {
      const next_buff = readPoolData(pool_id, socket, pool_data_buff, data, pool_err);
      if (next_buff === null) {return;}
      pool_data_buff = next_buff;
      if (!hasCompletePoolLine(pool_data_buff)) {return;}
      const { messages, incomplete_line } = splitPoolMessages(pool_data_buff);
      if (handlePoolLines(pool_id, messages, set_job, pool_err)) {return;}
      pool_data_buff = incomplete_line;
    };
  }

  function connect_pool(pool_id, set_job) {
    const pool = global.opt.pools[pool_id];

    // do not connect to already connected pools
    if (pool.socket) {return;}

    pool_log(pool_id, "Connecting to " + poolTypeStr(pool_id) + " " + pool_str(pool_id) + " pool");
    pool.last_connect_time = Date.now();
    const socket = connectSocket(pool);
    pool.socket = socket;
    pool.last_job = null;

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

  function connectPoolThrottle(pool_id, set_job) {
    const pool = global.opt.pools[pool_id];
    const wait_time = global.opt.pool_time.connect_throttle * 1000 -
                    (Date.now() - pool.last_connect_time);
    if (wait_time <= 0) {return connect_pool(pool_id, set_job);}
    pool_log(pool_id, "Waiting " + Math.floor(wait_time / 1000) + "s to connect to the pool");
    return setTimeout(connect_pool, wait_time, pool_id, set_job);
  }

  return {connectPoolThrottle};
};
