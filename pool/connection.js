"use strict";

const {StringDecoder} = require("node:string_decoder");
const {normalizeAlgoName} = require("../miner/algorithms");

/**
 * @param {{
 *   h: Pick<typeof import("../helper"), "log_err">,
 *   o: Pick<typeof import("../opts"), "agent_str">,
 *   net: typeof import("node:net"),
 *   tls: typeof import("node:tls"),
 *   systemNetConnect: typeof import("node:net").connect,
 *   systemTlsConnect: typeof import("node:tls").connect,
 *   max_pool_data_buffer: number,
 *   clear_pool_connection: (pool_id: number, socket: PoolSocket) => boolean,
 *   isCurrentPoolSocket: (pool_id: number, socket: PoolSocket) => boolean,
 *   pearlhashUsesSubscribe: (pool: PoolConfig) => boolean,
 *   poolProtocol: (pool: PoolConfig) => string,
 *   pool_log: (pool_id: number, message: string) => void,
 *   pool_log1: (pool_id: number, message: string) => void,
 *   pool_log_str: (pool_id: number, message: string) => string,
 *   poolErrorText: (pool_id: number, error: unknown) => string,
 *   pool_log_json: (pool_id: number, level: number, prefix: string, json: object) => void,
 *   pool_message: (pool_id: number, json: PoolMessage, set_job: (job: PoolJob) => MiningJob) => unknown,
 *   pool_str: (pool_id: number) => string,
 *   usesCortex: (pool: PoolConfig) => boolean,
 *   usesIronfish: (pool: PoolConfig) => boolean,
 *   usesMiningSubscribe: (pool: PoolConfig) => boolean,
 *   poolWrite: (pool_id: number, json: UnknownRecord) => unknown,
 *   switchPool: (pool_id: number, set_job: (job: PoolJob) => MiningJob) => unknown,
 * }} dependencies
 */
module.exports = ({
  h, o, net, tls, systemNetConnect, systemTlsConnect, max_pool_data_buffer,
  clear_pool_connection, isCurrentPoolSocket, pearlhashUsesSubscribe,
  poolProtocol, pool_log, pool_log1, pool_log_str,
  poolErrorText,
  pool_log_json,
  pool_message, pool_str, usesCortex, usesIronfish, usesMiningSubscribe,
  poolWrite, switchPool,
}) => {

  /** @param {unknown} value @returns {value is UnknownRecord} */
  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  /** @param {number} pool_id @returns {string} */
  function poolTypeStr(pool_id) {
    switch (pool_id) {
      case global.opt.pool_ids.primary: return "primary";
      case global.opt.pool_ids.donate:  return "donate";
      default:                          return "backup";
    }
  }

  /** @param {PoolConfig} pool @returns {PoolSocket} */
  function connectSocket(pool) {
    const loopback = pool.url === "127.0.0.1" || pool.url === "::1";
    if (pool.is_tls) {
      const connect = tls.connect;
      if (process.env["MOM_TEST_NO_POOL_NETWORK"] === "1" &&
          connect === systemTlsConnect && !loopback) {
        throw new Error("Pool network access is disabled during mom correctness tests");
      }
      return connect(pool.port, pool.url, {rejectUnauthorized: pool.tls_verify === true});
    }
    const connect = net.connect;
    if (process.env["MOM_TEST_NO_POOL_NETWORK"] === "1" &&
        connect === systemNetConnect && !loopback) {
      throw new Error("Pool network access is disabled during mom correctness tests");
    }
    return connect(pool.port, pool.url);
  }

  /** @param {PoolConfig} pool @returns {UnknownRecord} */
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

  /** @param {PoolConfig} pool @returns {{wallet: string, worker: string, pass: string}} */
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

  /** @param {string} message @returns {PoolMessage | undefined} */
  function parsePoolLine(message) {
    try {
      /** @type {unknown} */
      const json = JSON.parse(message);
      if (!isObject(json)) {return undefined;}
      attachKaspaPrecisePrePow(json, message);
      return json;
    } catch {
      return undefined;
    }
  }

  // The Kaspa mining.notify pre-pow words are 64-bit unsigned ints that exceed Number.MAX_SAFE_INTEGER,
  // so JSON.parse silently rounds them. Re-extract the exact integer literals from the raw line (decimal
  // strings) and stash them as BigInt-safe fields the kaspa job builder reads instead of the lossy array.
  /** @param {PoolMessage} json @param {string} message */
  function attachKaspaPrecisePrePow(json, message) {
    if (!json || json.method !== "mining.notify" || !Array.isArray(json.params) ||
      !Array.isArray(json.params[1])) {return;}
    const exact = JSON.parse(message, preserveJsonNumber);
    if (!isObject(exact) || !Array.isArray(exact["params"]) ||
      !Array.isArray(exact["params"][1])) {return;}
    const words = exact["params"][1].map((word) => exactUint64Decimal(word));
    if (words.length < 4) {return;}
    // Store as exact decimal strings (NOT BigInt) so the debug logger's JSON.stringify(json) still works.
    json.__kaspa_words = words.slice(0, 4);
    json.__kaspa_timestamp = exactUint64Decimal(exact["params"][2]);
  }

  /**
   * @param {string} _key
   * @param {unknown} value
   * @param {{source?: string}=} context
   * @returns {unknown}
   */
  function preserveJsonNumber(_key, value, context = {}) {
    return typeof value === "number" && typeof context.source === "string" ? context.source : value;
  }

  /** @param {unknown} value @returns {string} */
  function exactUint64Decimal(value) {
    const text = typeof value === "string" ? value :
      typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? String(value) : "";
    if (!/^(0|[1-9]\d*)$/.test(text) || text.length > 20) {
      throw new Error("Invalid uint64 decimal");
    }
    if (BigInt(text) > 0xffffffffffffffffn) {
      throw new Error("uint64 decimal is out of range");
    }
    return text;
  }

  /**
   * @param {number} pool_id
   * @param {PoolMessage} json
   * @param {(job: PoolJob) => MiningJob} set_job
   * @param {(message: string) => unknown} pool_err
   * @returns {boolean}
   */
  function processPoolJson(pool_id, json, set_job, pool_err) {
    pool_log_json(pool_id, 2, "Got from the pool: ", json);
    try {
      pool_message(pool_id, json, set_job);
    } catch (error) {
      pool_err(pool_log_str(pool_id, "Can't process message from the pool") +
        poolErrorText(pool_id, error));
      return true;
    }
    return false;
  }

  /**
   * @param {number} pool_id
   * @param {PoolSocket} socket
   * @param {string[]} messages
   * @param {(job: PoolJob) => MiningJob} set_job
   * @param {(message: string) => unknown} pool_err
   * @returns {boolean}
   */
  function handlePoolLines(pool_id, socket, messages, set_job, pool_err) {
    for (const message of messages) {
      // Processing a line can close or replace this socket; never consume its remaining work.
      if (!isCurrentPoolSocket(pool_id, socket)) {return true;}
      if (message.trim() === "") {continue;}
      const json = parsePoolLine(message);
      if (json === undefined) {
        pool_err(pool_log_str(pool_id,
          `Can't parse ${Buffer.byteLength(message)}-byte message from the pool`
        ));
        return true;
      }
      if (processPoolJson(pool_id, json, set_job, pool_err)) {return true;}
    }
    return !isCurrentPoolSocket(pool_id, socket);
  }

  /**
   * @param {number} pool_id
   * @param {PoolSocket} socket
   * @param {(job: PoolJob) => MiningJob} set_job
   * @returns {(message: unknown) => unknown}
   */
  function poolErrorHandler(pool_id, socket, set_job) {
    return function(message) {
      if (!clear_pool_connection(pool_id, socket)) {return;}
      h.log_err(String(message));
      return switchPool(pool_id, set_job);
    };
  }

  /**
   * @param {number} pool_id
   * @param {PoolSocket} socket
   * @param {(message: string) => unknown} pool_err
   */
  function scheduleInitialJobTimeout(pool_id, socket, pool_err) {
    setTimeout(function() {
      if (!isCurrentPoolSocket(pool_id, socket)) {return;}
      const pool = global.opt.pools[pool_id];
      if (!pool || pool.last_job) {return;}
      return pool_err(pool_log_str(pool_id,
        "No usable initial job from " + pool_str(pool_id) + " pool"
      ));
    }, global.opt.pool_time.first_job_wait * 1000);
  }

  /** @param {number} pool_id @param {PoolSocket} socket @param {PoolConfig} pool */
  function handlePoolConnect(pool_id, socket, pool) {
    if (!isCurrentPoolSocket(pool_id, socket)) {return;}
    pool_log1(pool_id, "Connected to the pool");
    if (usesIronfish(pool)) {
      // Iron Fish custom OBJECT stratum: a single mining.subscribe push carries the wallet+worker
      // (publicAddress) and the agent; the pool replies with mining.subscribed (handled by method).
      // No separate authorize. extend:["mining.submitted"] requests the submit-result push.
      return poolWrite(pool_id, {
        id: 1, method: "mining.subscribe",
        body: {version: 3, agent: o.agent_str, publicAddress: pool.login, extend: ["mining.submitted"]}
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
    if (pearlhashUsesSubscribe(pool)) {
      // PearlHash subscribe dialect: send subscribe AND authorize back-to-back. mining.subscribe is just
      // a handshake nicety -- HeroMiners acks it (result:true), LuckyPool rejects it ("method not
      // supported") and drops the connection if no authorize follows promptly. So don't wait on the
      // subscribe reply; authorize immediately. authorize takes OBJECT params {wallet,worker,pass}.
      pool.pending_authorize = true;
      poolWrite(pool_id, {jsonrpc: "2.0", id: 1, method: "mining.subscribe", params: [o.agent_str]});
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
    if (poolProtocol(pool) === "conflux") {
      return poolWrite(pool_id, {
        jsonrpc: "2.0", id: 1, method: "mining.subscribe",
        params: [pool.login, ""]
      });
    }
    let request;
    if (poolProtocol(pool) === "xelis") {
      request = {jsonrpc: "2.0", id: 1, method: "mining.subscribe", params: [o.agent_str, ["xel/v3"]]};
    } else if (usesMiningSubscribe(pool)) {
      request = {jsonrpc: "2.0", id: 1, method: "mining.subscribe", params: [o.agent_str]};
    } else {
      request = {jsonrpc: "2.0", id: 1, method: "login", params: poolLoginParams(pool)};
    }
    if (request.method === "mining.subscribe") {pool.pending_subscribe = true;}
    return poolWrite(pool_id, request);
  }

  /** @param {string} pool_data_buff @returns {{messages: string[], incomplete_line: string}} */
  function splitPoolMessages(pool_data_buff) {
    const messages = pool_data_buff.split("\n");
    const incomplete_line = pool_data_buff.endsWith("\n") ? "" : messages.pop() ?? "";
    return {messages, incomplete_line};
  }

  /**
   * @param {number} pool_id
   * @param {PoolSocket} socket
   * @param {string} pool_data_buff
   * @param {string} data
   * @param {(message: string) => unknown} pool_err
   * @returns {string | null}
   */
  function readPoolData(pool_id, socket, pool_data_buff, data, pool_err) {
    if (!isCurrentPoolSocket(pool_id, socket)) {return null;}
    const next_buff = pool_data_buff + data;
    if (Buffer.byteLength(next_buff) <= max_pool_data_buffer) {return next_buff;}
    pool_err(pool_log_str(pool_id, "Pool message buffer limit exceeded"));
    return null;
  }

  /**
   * @param {number} pool_id
   * @param {PoolSocket} socket
   * @param {(job: PoolJob) => MiningJob} set_job
   * @param {(message: string) => unknown} pool_err
   * @returns {(data: string | Buffer) => void}
   */
  function poolDataHandler(pool_id, socket, set_job, pool_err) {
    let pool_data_buff = "";
    const decoder = new StringDecoder("utf8");
    return function(data) {
      const decoded = typeof data === "string" ? data : decoder.write(data);
      const next_buff = readPoolData(
        pool_id, socket, pool_data_buff, decoded, pool_err
      );
      if (next_buff === null) {return;}
      pool_data_buff = next_buff;
      if (!pool_data_buff.includes("\n")) {return;}
      const {messages, incomplete_line} = splitPoolMessages(pool_data_buff);
      if (handlePoolLines(pool_id, socket, messages, set_job, pool_err)) {
        pool_data_buff = "";
        return;
      }
      pool_data_buff = incomplete_line;
    };
  }

  /** @param {number} pool_id @returns {boolean} */
  function donationConnectionAllowed(pool_id) {
    const pool = global.opt.pools[pool_id];
    return pool_id !== global.opt.pool_ids.donate || Boolean(pool?.donation_until &&
      pool.donation_until > Date.now());
  }

  /** @param {number} pool_id @param {(job: PoolJob) => MiningJob} set_job */
  function connect_pool(pool_id, set_job) {
    if (!donationConnectionAllowed(pool_id)) {return;}
    const pool = global.opt.pools[pool_id];
    if (!pool) {return;}

    // do not connect to already connected pools
    if (pool.socket) {return;}

    pool_log(pool_id, "Connecting to " + poolTypeStr(pool_id) + " " + pool_str(pool_id) + " pool");
    pool.last_connect_time = Date.now();
    const socket = connectSocket(pool);
    pool.socket = socket;
    pool.last_job = null;

    const pool_err = poolErrorHandler(pool_id, socket, set_job);
    scheduleInitialJobTimeout(pool_id, socket, pool_err);

    socket.on("connect", function () {
      handlePoolConnect(pool_id, socket, pool);
    });

    socket.on("data", poolDataHandler(pool_id, socket, set_job, pool_err));

    socket.on("end", function() {
      return pool_err(pool_log_str(pool_id, "Socket closed from the pool"));
    });

    socket.on("error", function(error) {
      const code = error && typeof error.code === "string" && /^[A-Z0-9_]+$/.test(error.code)
        ? ` (${error.code})` : "";
      return pool_err(pool_log_str(pool_id, "Socket error from the pool" + code));
    });
  }

  /** @param {number} pool_id @param {(job: PoolJob) => MiningJob} set_job @returns {void | NodeJS.Timeout} */
  function connectPoolThrottle(pool_id, set_job) {
    if (!donationConnectionAllowed(pool_id)) {return;}
    const pool = global.opt.pools[pool_id];
    if (!pool) {return;}
    const wait_time = global.opt.pool_time.connect_throttle * 1000 -
                    (Date.now() - pool.last_connect_time);
    if (wait_time <= 0) {return connect_pool(pool_id, set_job);}
    pool_log(pool_id, "Waiting " + Math.floor(wait_time / 1000) + "s to connect to the pool");
    return setTimeout(connect_pool, wait_time, pool_id, set_job);
  }

  return {connectPoolThrottle};
};
