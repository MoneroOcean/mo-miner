// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

"use strict";

const net = require("node:net");
const tls = require("node:tls");
const h = require("./helper.js");
const o = require("./opts.js");
const {normalizeAlgoName, PROGPOW_ALGOS} = require("./miner/algorithms");
const {hexWithoutPrefix, xelisWorkerName} = require("./miner/submission");
const {isObject} = require("./helper/validate");

// Correctness tests may use normal network services, but must never contact a mining pool. Capture
// the actual socket functions before logic tests substitute their in-memory sockets, so the guard
// blocks only a real pool connection and leaves the protocol fixtures testable.
const systemNetConnect = net.connect;
const systemTlsConnect = tls.connect;

const MAX_POOL_DATA_BUFFER = 1024 * 1024;
// Liveness deadlines. A hung pool process or a silently dropped path (NAT/conntrack expiry,
// sleep/resume, address change) leaves TCP open: the kernel notices only after ~15 min of
// retransmitting our own writes, and never for dialects that write only when a share is found.
// Plain timers also work in the native runtime, whose sockets have no setTimeout/setKeepAlive.
// - A healthy pool answers a request within seconds (xmrig allows 20 s); 60 s also covers an
//   overloaded pool. Any received line satisfies it, because it proves the path is alive.
// - Pools push work at least once per block (most also rebroadcast every 30-60 s), and login pools
//   answer keepalived every pool_time.keepalive (5 min). Twenty silent minutes is four mean block
//   intervals of the slowest supported chain (Firo, 5 min), so only a dead connection reaches it.
const POOL_RESPONSE_TIMEOUT_MS = 60 * 1000;
const POOL_IDLE_TIMEOUT_MS = 20 * 60 * 1000;
const fullTargetAlgos = new Set([
  "etchash", "octopus", "autolykos2", "fishhash", "c30", "equihash192_7", "zelhash", "zhash",
  "karlsenhashv2", "walahash", "hoohash", "verthash", "xelishashv3", "nexapow",
]);

/** @typedef {(job: PoolJob) => MiningJob} SetJobCallback */
/** @typedef {(pool_id: number, isError: boolean, isOk: boolean, errorText: string,
 * json: PoolMessage, setJob: SetJobCallback) => void} PoolResponseHandler */
/** @typedef {PoolMessage & {code: number, description?: unknown, nonceprefix?: unknown}} BeamResultMessage */

/**
 * Resolve a pool id at the boundary where pool state is consumed. Pool ids are produced by the
 * validated options layer, but this guard also keeps malformed protocol/test inputs from turning an
 * out-of-range array access into an unrelated TypeError.
 * @param {number} pool_id
 * @returns {PoolConfig}
 */
function poolAt(pool_id) {
  const pool = global.opt.pools[pool_id];
  if (!pool) {throw new Error("Invalid pool id");}
  return pool;
}

/** @param {number} pool_id @returns {string} */
function poolName(pool_id) {
  const pool = poolAt(pool_id);
  return pool.url + ":" + pool.port + (pool.is_tls ? "tls" : "");
}

/** @param {number} pool_id @param {string} str @returns {string} */
function poolLogText(pool_id, str) {
  return global.opt.log_level >= 1 ? "[" + poolName(pool_id) + "] " + str : str;
}

/** @param {number} pool_id @param {string} str @returns {void} */
function poolLog(pool_id, str) {
  h.log(poolLogText(pool_id, str));
}

/** @param {number} pool_id @param {string} str @returns {void} */
function poolLogErr(pool_id, str) {
  h.log_err(poolLogText(pool_id, str));
}

// Remove the pool's login, wallet, worker, and password (including their JSON-escaped forms) from
// text that is about to be logged. Short credentials are replaced only as whole tokens.
/** @param {number} pool_id @param {unknown} value @returns {string} */
function redactPoolText(pool_id, value) {
  const pool = poolAt(pool_id);
  const login = String(pool.login || "");
  const workerSeparator = login.lastIndexOf(".");
  let text = String(value);
  const credentials = [
    login,
    login.split(".")[0] || "",
    workerSeparator > 0 ? login.slice(0, workerSeparator) : "",
    workerSeparator > 0 ? login.slice(workerSeparator + 1) : "",
    String(pool.worker || ""),
    String(pool.pass || ""),
  ].filter(Boolean);
  const needles = [...new Set(credentials.flatMap((credential) => [
    credential, JSON.stringify(credential).slice(1, -1),
  ]))].sort((left, right) => right.length - left.length);
  for (const needle of needles) {
    if (text === needle) {
      text = "<redacted>";
    } else if (needle.length >= 4) {
      text = text.split(needle).join("<redacted>");
    } else {
      const escaped = needle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      text = text.replace(
        new RegExp(`(^|[^a-z0-9])${escaped}(?=$|[^a-z0-9])`, "gi"),
        "$1<redacted>"
      );
    }
  }
  return text;
}

/** @param {number} pool_id @param {number} level @param {string} prefix @param {PoolMessage | PoolJob | MiningJob} json @returns {void} */
function poolLogJson(pool_id, level, prefix, json) {
  if (global.opt.log_level < level) {
    return;
  }
  let message;
  try {
    message = redactPoolText(pool_id, JSON.stringify(json, (key, value) =>
      key === "plain_proof" ? "<redacted>" : value) ?? "undefined");
  } catch {
    message = '"<unprintable pool JSON>"';
  }
  (level === 1 ? h.log1 : h.log2)(poolLogText(pool_id, prefix + message));
}

// Forget the per-connection difficulty, target, and extranonce state that job builders read.
/** @param {PoolConfig} pool */
function clearPoolJobState(pool) {
  delete pool.beam_difficulty;
  delete pool.beam_nonceprefix;
  delete pool.cortex_nonce;
  delete pool.eth_difficulty;
  delete pool.eth_target;
  delete pool.extra_nonce;
  delete pool.extra_nonce2_size;
  delete pool.ironfish_target;
  delete pool.ironfish_xn;
  delete pool.kaspa_difficulty;
  delete pool.kaspa_target;
  delete pool.nexa_difficulty;
  delete pool.nexa_target;
  delete pool.pearlhash_difficulty;
  delete pool.pearlhash_proof_encodings;
  delete pool.raven_target;
  delete pool.stratum_target;
  delete pool.verthash_difficulty;
  delete pool.xelis_difficulty;
  delete pool.xelis_extra_nonce;
  delete pool.xelis_public_key;
  delete pool.zelhash_target;
  delete pool.pending_job;
}

// Tear down the pool's connection and all session state. A stale socket (one that was already
// replaced) is ignored, so its late events cannot clear the current connection.
/** @param {number} pool_id @param {PoolSocket | null} socket @returns {boolean} */
function clearPoolConnection(pool_id, socket) {
  const pool = poolAt(pool_id);
  if (socket && !isCurrentPoolSocket(pool_id, socket)) {
    return false;
  }
  clearPoolKeepalive(pool);
  clearTimeout(pool.first_job_timer);
  delete pool.first_job_timer;
  clearTimeout(pool.response_timer);
  delete pool.response_timer;
  clearTimeout(pool.idle_timer);
  delete pool.idle_timer;
  if (pool.socket) {
    pool.socket.destroy();
  }
  pool.socket = null;
  pool.last_job = null;
  clearPoolJobState(pool);
  delete pool.inferred_protocol;
  delete pool.negotiated_keepalive;
  delete pool.negotiated_nicehash;
  delete pool.worker_id;
  delete pool.extensions;
  delete pool.requested_extensions;
  delete pool.requested_algos;
  delete pool.job_algo;
  delete pool.pending_controls;
  delete pool.pending_cortex_submit_ids;
  pool.pending_cortex_login = false;
  pool.pending_cortex_work = false;
  pool.pending_subscribe = false;
  pool.logged_in = false;
  pool.pending_authorize = false;
  pool.pending_submit_count = 0;
  return true;
}

/** @param {PoolConfig} pool @returns {void} */
function clearPoolKeepalive(pool) {
  if (pool.keepalive !== null) {
    clearTimeout(pool.keepalive);
  }
  pool.keepalive = null;
}

/**
 * Arm a liveness deadline for the socket. On expiry the socket emits "timeout" with the reason,
 * which connect_pool handles like a socket error; clearPoolConnection cancels pending deadlines.
 * @param {PoolSocket} socket @param {number} delay @param {string} reason
 * @returns {NodeJS.Timeout}
 */
function poolDeadline(socket, delay, reason) {
  return setTimeout(() => socket.emit("timeout", reason), delay);
}

/**
 * Handle one line from the pool's current socket: it proves the pool is alive, so it satisfies a
 * pending response deadline and restarts the idle deadline before the message is processed.
 * @param {number} pool_id @param {PoolMessage} json @param {SetJobCallback} setJob
 */
function receivePoolMessage(pool_id, json, setJob) {
  const pool = poolAt(pool_id);
  clearTimeout(pool.response_timer);
  delete pool.response_timer;
  clearTimeout(pool.idle_timer);
  if (pool.socket) {
    pool.idle_timer = poolDeadline(pool.socket, POOL_IDLE_TIMEOUT_MS,
      `No data from the pool for ${POOL_IDLE_TIMEOUT_MS / 60000} minutes`);
  }
  return routePoolMessage(pool_id, json, setJob);
}

/** @param {number} pool_id @param {PoolSocket} socket @returns {boolean} */
function isCurrentPoolSocket(pool_id, socket) {
  return poolAt(pool_id).socket === socket;
}

// Stratum dialect of each algorithm whose pools do not use the default `login` dialect.
const protocolByAlgo = new Map([
  ["kawpow", "raven"], ["firopow", "raven"], ["evrprogpow", "raven"], ["meowpow", "raven"],
  ["equihash192_7", "zelhash"], ["zelhash", "zelhash"], ["zhash", "zelhash"],
  ["karlsenhashv2", "kaspa"], ["walahash", "kaspa"], ["hoohash", "hoosat"],
  ["etchash", "eth"], ["octopus", "conflux"], ["nexapow", "echelon"], ["c30", "cortex"],
  ["autolykos2", "erg"], ["pearlhash", "pearlhash"], ["fishhash", "ironfish"],
  ["beamhash3", "beam"], ["verthash", "verthash"], ["xelishashv3", "xelis"],
]);

/** @param {string | null | undefined} algo @returns {string | null} */
function protocolForAlgo(algo) {
  return protocolByAlgo.get(normalizeAlgoName(algo) || "") ?? null;
}

// Beam packs the network difficulty into a 32-bit int (top 8 bits = order, low 24 = mantissa). The
// native beamhash3 solver re-derives it from the low 4 bytes of the 32-byte big-endian target, so the
// JS job carries the packed int there. We keep a per-pool copy so a job that omits `difficulty` (some
// pools push it only on the login/set) can still resolve a target.
/** @param {unknown} packed @returns {string} */
function beamPackedTarget(packed) {
  if (typeof packed !== "number" &&
      (typeof packed !== "string" || !/^\d+$/.test(packed))) {
    throw new Error("Invalid Beam packed difficulty");
  }
  const numeric = Number(packed);
  if (!Number.isSafeInteger(numeric) || numeric <= 0 || numeric > 0xFFFFFFFF) {
    throw new Error("Invalid Beam packed difficulty");
  }
  const p = (numeric >>> 0).toString(16).padStart(8, "0");
  return "0".repeat(56) + p;   // 64 hex = 32 bytes big-endian, packed int32 in the low 4 bytes
}

/** @returns {string} */
function defaultPoolProtocol() {
  const protocol = protocolForAlgo(global.opt.job.algo);
  return protocol || "login";
}

/** @param {PoolConfig} pool @returns {string} */
function poolProtocol(pool) {
  if (pool.extensions?.includes("mo-native") && pool.inferred_protocol) {
    return pool.inferred_protocol;
  }
  return normalizeAlgoName(pool.protocol || pool.inferred_protocol || defaultPoolProtocol()) || "login";
}

/** @param {PoolConfig} pool @returns {boolean} */
function usesMiningSubscribe(pool) {
  const protocol = poolProtocol(pool);
  return pool.use_subscribe !== false &&
    (protocol === "raven" || protocol === "eth" || protocol === "erg" ||
     protocol === "conflux" || protocol === "zelhash" || protocol === "kaspa" ||
     protocol === "hoosat" || protocol === "verthash" ||
     protocol === "xelis" || protocol === "echelon");
}

/** @param {PoolConfig} pool @returns {boolean} */
function usesIronfish(pool) {
  return poolProtocol(pool) === "ironfish";
}

/** @param {PoolConfig} pool @returns {boolean} */
function usesCortex(pool) {
  return poolProtocol(pool) === "cortex";
}

// Standard Pearl handshake (HeroMiners/LuckyPool/etc.): mining.subscribe + mining.authorize
// {wallet,worker,pass}. This is the default for PearlHash pools. pearlpool.cloud uses the older single
// `login` dialect instead -- opt OUT of subscribe there with "use_subscribe": false (the MoneroOcean
// donate pool also sets it false so donation keeps using login). Both dialects push the same
// object-param PearlHash mining.notify and take the same mining.submit{job_id,plain_proof}.
/** @param {PoolConfig} pool @returns {boolean} */
function pearlhashUsesSubscribe(pool) {
  // MOM_PEARLHASH_LOGIN forces pearlpool.cloud's login dialect for CLI mining.
  // The MO donate pool opts out via use_subscribe:false, so this env never affects donation.
  if (process.env["MOM_PEARLHASH_LOGIN"]) {
    return false;
  }
  return poolProtocol(pool) === "pearlhash" && pool.use_subscribe !== false;
}

// Pearl difficulty is carried in the job_id suffix "<hex>_<diff>" (HeroMiners omits the difficulty
// field that pearlpool.cloud sends); used to derive the 2^256/diff kernel target.
/** @param {unknown} job_id @returns {number | undefined} */
function pearlhashDiffFromJobId(job_id) {
  const text = typeof job_id === "string" ? job_id :
    typeof job_id === "number" && Number.isSafeInteger(job_id) ? String(job_id) : "";
  const m = text.match(/_(\d+)$/);
  const difficulty = m && m[1] ? Number(m[1]) : Number.NaN;
  return Number.isSafeInteger(difficulty) && difficulty > 0 ? difficulty : undefined;
}

// Send one JSON-RPC line, arm the response deadline, and (for login dialects with keepalive)
// reschedule the next keepalived. Write failures surface asynchronously as socket errors.
/** @param {number} pool_id @param {UnknownRecord} json @returns {void} */
module.exports.pool_write = function(pool_id, json) {
  const message = JSON.stringify(json);
  const pool = poolAt(pool_id);
  if (!pool.socket) {
    return poolLogJson(pool_id, 2, "Sent to the closed pool socket: ", json);
  }

  poolLogJson(pool_id, 2, "Sent to the pool: ", json);
  if (json["id"] === 3 && (json["method"] === "mining.submit" || json["method"] === "submit")) {
    ++pool.pending_submit_count;
  }
  const socket = pool.socket;
  try {
    socket.write(message + "\n");
  } catch (error) {
    // The native runtime refuses to queue more than 32 MiB on a stalled socket (Node never
    // throws here). Report it like any stream error: asynchronously, through the socket.
    process.nextTick(() => socket.emit("error", error));
    return;
  }
  // A pong answers the pool, and Iron Fish submit results are an optional extension push.
  const expectsResponse = json["method"] !== "mining.pong" &&
    !(usesIronfish(pool) && json["method"] === "mining.submit");
  if (expectsResponse && !pool.response_timer) {
    pool.response_timer = poolDeadline(socket, POOL_RESPONSE_TIMEOUT_MS,
      `No response from the pool for ${POOL_RESPONSE_TIMEOUT_MS / 1000} s`);
  }
  // sends keepalive if no submit/keepalive to pool for more than global.opt.pool_time.keepalive
  if (!(pool.is_keepalive || pool.negotiated_keepalive) || usesMiningSubscribe(pool) ||
      pearlhashUsesSubscribe(pool) || usesIronfish(pool) || usesCortex(pool)) {
    return;
  }
  clearPoolKeepalive(pool);
  pool.keepalive = setTimeout(function() {
    pool.keepalive = null;
    const params = pool.worker_id === undefined ? {} : {id: pool.worker_id};
    module.exports.pool_write(pool_id, {
      jsonrpc: "2.0", id: 2, method: "keepalived", params
    });
  }, global.opt.pool_time.keepalive * 1000);
};

// Native cleanup may emit a final share. Let its response arrive before controlled shutdown,
// but never let an unresponsive pool keep the process alive indefinitely.
/** @param {number} [timeoutMs] @returns {Promise<void>} */
module.exports.waitForPendingShares = function(timeoutMs = 5000) {
  if (!Number.isFinite(timeoutMs) || timeoutMs < 0 || timeoutMs > 5000) {
    throw new RangeError("Share drain timeout must be between 0 and 5000 ms");
  }
  const deadline = performance.now() + timeoutMs;
  return new Promise((resolve) => {
    const poll = () => {
      const pending = global.opt.pools.some((pool) => pool.socket &&
        (pool.pending_submit_count > 0 || pool.pending_cortex_submit_ids?.size));
      if (!pending || performance.now() >= deadline) {resolve();} else {setTimeout(poll, 25);}
    };
    poll();
  });
};

// soft kill pool connection
/** @param {number} pool_id @returns {void} */
function softClosePool(pool_id) {
  const socket = poolAt(pool_id).socket;
  if (!socket) {
    return;
  }
  h.log1(poolLogText(pool_id, "Soft closing the pool connection"));
  setTimeout(function() {
    // do not do soft close if this pool became active again
    if (pool_id === global.opt.pool_ids.active ||
        !isCurrentPoolSocket(pool_id, socket)) {
      return;
    }
    h.log1(poolLogText(pool_id, "Soft closed the pool connection"));
    clearPoolConnection(pool_id, socket);
  }, global.opt.pool_time.close_wait * 1000);
}

/** @param {number} pool_id @returns {string} */
function poolShareStats(pool_id) {
  const pool = poolAt(pool_id);
  return "(" + pool.good_shares + "/" + pool.bad_shares + ")";
}

/** @param {number} pool_id @param {unknown} error @returns {string} */
function poolErrorText(pool_id, error) {
  const message = typeof error === "string" ? error :
    error instanceof Error ? error.message :
      Array.isArray(error) && typeof error[1] === "string" ? error[1] :
        isObject(error) && typeof error["message"] === "string" ? error["message"] :
          isObject(error) && typeof error["msg"] === "string" ? error["msg"] : "";
  return message ? ": " + JSON.stringify(redactPoolText(pool_id, message).slice(0, 200)) : "";
}

/** @param {number} pool_id @param {unknown} extensions @returns {void} */
function applyLoginExtensions(pool_id, extensions) {
  if (!Array.isArray(extensions)) {
    return;
  }
  const pool = poolAt(pool_id);
  pool.extensions = (pool.requested_extensions || []).filter((name) => extensions.includes(name));
  pool.negotiated_nicehash = extensions.includes("nicehash");
  pool.negotiated_keepalive = extensions.includes("keepalive");
}

/** @param {PoolConfig} pool @returns {string} */
function algoFromPass(pool) {
  const pass = String(pool.pass || "");
  const m = pass.match(/(?:^|[~;,])(?:algo=)?(kawpow|firopow|evrprogpow|meowpow|etchash|autolykos2|pearlhash|fishhash|equihash192_7|zelhash|zhash|karlsenhashv2|walahash|hoohash|verthash|xelishashv3|xel\/(?:2|3|v3)|nexapow)(?:$|[~;,])/i);
  return m ? normalizeAlgoName(m[1]) || "" : "";
}

/** @param {PoolConfig} pool @returns {string[]} */
function xelisAuthorizeParams(pool) {
  const login = String(pool.login || "");
  const separator = login.indexOf(".");
  const wallet = separator < 0 ? login : login.slice(0, separator);
  // Pools may bind submits to the authorized name, so both must use one rule.
  return [wallet, xelisWorkerName(pool), pool.pass];
}

// Infer the dialect of a pool without an explicit protocol from the algorithm named by a login
// result or push, a single offered algorithm, or the password's algo marker.
/** @param {number} pool_id @param {unknown} result @returns {void} */
function rememberPoolProtocol(pool_id, result) {
  const pool = poolAt(pool_id);
  if (pool.protocol && !pool.extensions?.includes("mo-native")) {
    return;
  }
  const offered = pool.use_subscribe === false && pool.algo_params
    ? Object.keys(pool.algo_params) : [];
  let resultAlgo = null;
  if (isObject(result)) {
    if (typeof result["algo"] === "string") {
      resultAlgo = result["algo"];
    } else if (isObject(result["job"]) && typeof result["job"]["algo"] === "string") {
      resultAlgo = result["job"]["algo"];
    }
  }
  const protocol = protocolForAlgo(
    resultAlgo || (offered.length === 1 ? offered[0] : null) || algoFromPass(pool)
  );
  if (!protocol && !resultAlgo) {
    return;
  }
  pool.inferred_protocol = protocol || "login";
  if (usesMiningSubscribe(pool)) {
    clearPoolKeepalive(pool);
  }
}

/** @param {unknown} value @returns {string} */
function messageAlgorithm(value) {
  if (!isObject(value)) {return "";}
  const nested = isObject(value["params"]) ? value["params"] :
    isObject(value["job"]) ? value["job"] : null;
  const outer = typeof value["algo"] === "string" ? normalizeAlgoName(value["algo"]) || "" : "";
  const inner = typeof nested?.["algo"] === "string" ? normalizeAlgoName(nested["algo"]) || "" : "";
  if (outer && inner && outer !== inner) {throw new Error("Conflicting pool algorithm markers");}
  return outer || inner;
}

/** @param {number} pool_id @param {string} algo */
function preparePoolFamily(pool_id, algo) {
  const pool = poolAt(pool_id);
  if (!pool.requested_algos?.includes(algo)) {
    throw new Error("Pool assigned an unadvertised algorithm: " + algo);
  }
  if (pool.job_algo && pool.job_algo !== algo) {clearPoolJobState(pool);}
  pool.job_algo = algo;
  pool.inferred_protocol = protocolForAlgo(algo) || "login";
}

/** @param {PoolConfig} pool @param {unknown} value */
function rememberExtraNonceSize(pool, value) {
  if (value === undefined) {return;}
  const size = parseExtraNonce2Size(value);
  if (size === null) {throw new Error("Invalid extra_nonce2_size");}
  pool.extra_nonce2_size = size;
}

/** @param {PoolMessage} json @returns {string} */
function controlKind(json) {
  if (isSetTargetNotification(json) || isIronfishSetTargetNotification(json)) {return "target";}
  if (isSetDifficultyNotification(json)) {return "difficulty";}
  return isSetExtranonceNotification(json) ? "extranonce" : "";
}

// Apply a set_target, set_difficulty, or set_extranonce push to the pool's current job state.
/** @param {number} pool_id @param {PoolMessage} json */
function applyPoolControl(pool_id, json) {
  const pool = poolAt(pool_id);
  if (isSetTargetNotification(json)) {return handleSetTarget(pool_id, json.params[0]);}
  if (isIronfishSetTargetNotification(json)) {return handleSetTarget(pool_id, json.body["target"]);}
  if (isSetDifficultyNotification(json)) {return handleSetDifficulty(pool_id, json);}
  if (isSetExtranonceNotification(json)) {
    if (poolProtocol(pool) === "xelis") {return rememberXelisExtranonce(pool_id, json.params);}
    const value = json.params[0];
    const prefix = validExtraNonce(value);
    if (!prefix && value !== "") {throw new Error("Invalid pool extranonce");}
    rememberPoolExtraNonceHex(pool_id, prefix);
    if (poolProtocol(pool) === "ironfish") {pool.ironfish_xn = prefix;}
    if (poolProtocol(pool) === "beam") {
      if (prefix.length > 12) {throw new Error("Invalid Beam nonce prefix");}
      pool.beam_nonceprefix = prefix;
    }
    rememberExtraNonceSize(pool, json.params[1]);
  }
}

const poolJobs = require("./pool/jobs")({
  h, normalizeAlgoName, poolAt, poolProtocol,
  pearlhashDiffFromJobId, beamPackedTarget, softClosePool,
  poolLog, poolLogErr, poolErrorText, poolName, algoFromPass,
  connectPoolThrottle: (...args) => module.exports.connect_pool_throttle(...args),
});
const {
  isIronfishSetTargetNotification, isSetTargetNotification,
  isSetDifficultyNotification, isSetExtranonceNotification, parseExtraNonce2Size,
  validExtraNonce, rememberPoolExtraNonceHex, rememberSubscribeExtraNonce,
  switchPool, handleSetTarget, rememberXelisExtranonce, handleSetDifficulty,
  jobFromPoolMessage,
} = poolJobs;
module.exports.switch_pool = switchPool;
// Expected work per share, in the unit each algorithm reports its hashrate in; null if unknown.
/** @param {PoolJob} job @returns {bigint | number | null} */
function jobTargetWork(job) {
  const target = job.target;
  if (typeof job.algo !== "string") {
    throw new Error("Pool job has no algorithm");
  }
  // BeamHash III carries a PACKED 32-bit network difficulty (not a 256-bit boundary); the share rate is
  // in solutions, so report the packed difficulty itself rather than decoding job.target as a boundary.
  if (job.algo === "beamhash3") {
    return job.difficulty ? BigInt(job.difficulty) : null;
  }
  if (!target) {
    return null;
  }
  if (PROGPOW_ALGOS.has(job.algo)) {
    return h.kawpowTarget2diff(target);
  }
  // PearlHash reports rank-128-equivalent GEMM MACs. V3 normalizes the target to rank 128, so
  // work/share = tiles/share * 16*16*k*128/rank and remains comparable across valid ranks.
  if (job.algo === "pearlhash") {
    if (typeof job.pearlhash_k !== "number" || typeof job.pearlhash_rank !== "number") {
      throw new Error("Invalid PearlHash K/rank");
    }
    return h.pearlhashTargetWork(target, job.pearlhash_k, job.pearlhash_rank);
  }
  // These algorithms carry a full 256-bit target and report hashes, so convert it to hashes/share.
  if (fullTargetAlgos.has(job.algo)) {
    return h.target256ToWork(target);
  }
  return h.target2diff(target);
}

/** @param {PoolJob} job @returns {string} */
function jobTargetDescription(job) {
  const work = jobTargetWork(job);
  return work !== null ? h.formatHashCount(work) + "/share target" : job.difficulty + " diff";
}

/** @param {number} pool_id @param {number} activePool @returns {void} */
function activatePoolForJob(pool_id, activePool) {
  // only switch active pool once for its first job here
  if (pool_id === activePool || poolAt(pool_id).last_job) {
    return;
  }
  if (pool_id === global.opt.pool_ids.primary) {
    poolLog(pool_id, "Switching active pool to primary " + poolName(pool_id) + " pool");
    softClosePool(activePool);
    global.opt.pool_ids.active = pool_id;
  } else if (pool_id === global.opt.pool_ids.donate) {
    poolLog(pool_id, "Switching active pool to donate " + poolName(pool_id) + " pool");
    global.opt.pool_ids.active = pool_id;
  }
}

// Accept a decoded pool job: validate its target, switch the active pool on a primary or donation
// pool's first job, and dispatch it to the workers when the pool is active (otherwise keep it).
/**
 * @param {number} pool_id
 * @param {PoolJob} job
 * @param {SetJobCallback} setJob
 * @returns {void}
 */
function handlePoolJob(pool_id, job, setJob) {
  const pool = poolAt(pool_id);
  const algo = normalizeAlgoName(job.algo || pool.job_algo || global.opt.job.algo);
  if (!algo) {throw new Error("Pool job has no algorithm");}
  if (pool.extensions?.includes("mo-native")) {
    if (algo !== pool.job_algo) {throw new Error("Decoded pool job does not match its algorithm marker");}
    if (job.submit_mode == null && job.xn === undefined && pool.extra_nonce !== undefined &&
        pool.extra_nonce2_size !== undefined) {
      const width = job.noncebytes ?? (algo === "c29" && job.blob ? 8 : 4);
      if ((width !== 4 && width !== 8) || pool.extra_nonce.length / 2 + pool.extra_nonce2_size !== width) {
        throw new Error("Pool extranonce does not fit the job nonce");
      }
      job.xn ??= pool.extra_nonce;
      job.extra_nonce ??= pool.extra_nonce;
      job.extra_nonce2_size ??= pool.extra_nonce2_size;
      job.noncebytes ??= width;
      if (job.xn === "") {job.nicehash_mask ??= "00".repeat(width);}
    }
  }
  job.algo = algo;
  if (!donationJobSupported(pool_id, job, setJob)) {return;}
  if (job.target) {
    // Pearl's K/rank are selected by setJob; validate the final threshold before that selection.
    if (algo === "pearlhash") {h.target256ToWork(job.target);} else {jobTargetWork(job);}
  } // throws early on a malformed target before we store the job
  activatePoolForJob(pool_id, global.opt.pool_ids.active);

  job.submit_mode ??= null;
  job.protocol = poolProtocol(pool);
  job.submit_result = pool.extensions?.includes("submit-result") === true;
  pool.last_job = job;
  if (pool_id === global.opt.pool_ids.active) {
    const miningJob = setJob(job);
    poolLog(pool_id, "Got new " + miningJob.algo + " algo job with " +
                     jobTargetDescription(miningJob) +
                     (miningJob.height ? " and " + miningJob.height + " height" : "")
    );
  } else {
    poolLogJson(pool_id, 2, "Storing not active pool job ", job);
  }
}

// Donation work is accepted only inside the donation window and for an algorithm in the donation
// capability map; anything else ends the donation connection.
/**
 * @param {number} pool_id
 * @param {PoolJob} job
 * @param {SetJobCallback} setJob
 * @returns {boolean}
 */
function donationJobSupported(pool_id, job, setJob) {
  if (pool_id !== global.opt.pool_ids.donate) {return true;}
  const pool = poolAt(pool_id);
  const algo = normalizeAlgoName(job.algo) || "";
  const inWindow = Date.now() < (pool.donation_until ?? 0);
  if (inWindow && pool.algo_params && Object.hasOwn(pool.algo_params, algo)) {
    return true;
  }
  // The donation capability map already carries discovery's device and VRAM checks. Refuse any
  // proxy/pool assignment outside it before switching pools or allocating algorithm workers.
  // A queued job can arrive as the window expires; normal expiry is not a pool error.
  if (inWindow) {
    poolLogErr(pool_id, "Donation pool returned unsupported " + String(job.algo) + " work");
  }
  if (pool_id === global.opt.pool_ids.active) {
    switchPool(pool_id, setJob);
  } else {
    clearPoolConnection(pool_id, pool.socket);
  }
  return false;
}

/** @param {number} pool_id @returns {void} */
function loginSucceeded(pool_id) {
  poolAt(pool_id).logged_in = true;
  return poolLog(pool_id, "Login to the pool succeeded");
}

/** @param {number} pool_id @param {string} reason @returns {void} */
function loginFailed(pool_id, reason) {
  poolAt(pool_id).logged_in = false;
  return poolLogErr(pool_id, "Login to the pool failed" + reason);
}

/**
 * @param {number} pool_id
 * @param {boolean} isError
 * @param {boolean} isOk
 * @param {string} errorText
 * @param {PoolMessage} _json
 * @returns {void}
 */
function handleLoginResponse(pool_id, isError, isOk, errorText, _json) {
  const pool = poolAt(pool_id);
  if (usesCortex(pool)) {pool.pending_cortex_login = false;}
  if (isError || !isOk) {
    delete pool.pending_job;
    return loginFailed(pool_id, errorText || ": Login rejected");
  }
  loginSucceeded(pool_id);
  if (usesCortex(pool)) {
    pool.pending_cortex_work = true;
    return module.exports.pool_write(pool_id, {
      id: 100, jsonrpc: "2.0", method: "ctxc_getWork", params: [""],
    });
  }
}

/**
 * @param {number} pool_id
 * @param {boolean} isError
 * @param {boolean} isOk
 * @param {string} errorText
 * @param {PoolMessage} json
 * @returns {void}
 */
function handleSubscribeResponse(pool_id, isError, isOk, errorText, json) {
  poolAt(pool_id).pending_subscribe = false;
  if (isError) {
    delete poolAt(pool_id).pending_job;
    return poolLogErr(pool_id, "Subscribe to the pool failed" + errorText);
  }
  if (!isOk) {
    delete poolAt(pool_id).pending_job;
    return;
  }
  rememberSubscribeExtraNonce(pool_id, json.result);
  const pool = poolAt(pool_id);
  if (pool.logged_in) {return;}
  pool.pending_authorize = true;
  const params = poolProtocol(pool) === "xelis" ? xelisAuthorizeParams(pool) : [pool.login, pool.pass];
  return module.exports.pool_write(pool_id, {
    jsonrpc: "2.0", id: 2, method: "mining.authorize", params
  });
}

/**
 * @param {number} pool_id
 * @param {boolean} isError
 * @param {boolean} _isOk
 * @param {string} errorText
 * @param {PoolMessage} json
 * @param {SetJobCallback} setJob
 * @returns {void}
 */
function handleAuthorizeResponse(pool_id, isError, _isOk, errorText, json, setJob) {
  const pool = poolAt(pool_id);
  pool.pending_authorize = false;
  if (!isError && json.result === true) {
    loginSucceeded(pool_id);
    const pendingJob = pool.pending_job;
    delete pool.pending_job;
    if (pendingJob) {
      return routePoolMessage(pool_id, pendingJob, setJob);
    }
    return;
  }
  delete pool.pending_job;
  return loginFailed(pool_id, errorText || ": Authorization rejected");
}

/**
 * @param {number} pool_id
 * @param {boolean} isError
 * @param {boolean} isOk
 * @param {string} errorText
 * @returns {void}
 */
function handleShareResponse(pool_id, isError, isOk, errorText) {
  const pool = poolAt(pool_id);
  if (isError || isOk === false) {
    ++pool.bad_shares;
    return poolLogErr(pool_id, "Share rejected by the pool " + poolShareStats(pool_id) + errorText);
  }
  if (isOk) {
    ++pool.good_shares;
    return poolLog(pool_id, "Share accepted by the pool " + poolShareStats(pool_id));
  }
}

/**
 * @param {number} pool_id
 * @param {boolean} isError
 * @param {boolean} _isOk
 * @param {string} errorText
 * @returns {void}
 */
function handleCortexWorkResponse(pool_id, isError, _isOk, errorText) {
  poolAt(pool_id).pending_cortex_work = false;
  return poolLogErr(pool_id, "Cortex getWork failed" +
    (isError && errorText ? errorText : ": Invalid work response"));
}

// Route an {id, result|error} reply to the handler for the request it answers.
/** @param {number} pool_id @param {PoolMessage} json @param {SetJobCallback} setJob @returns {void} */
function handlePoolResponse(pool_id, json, setJob) {
  const isError  = "error" in json && json.error !== null;
  let errorText = isError ? poolErrorText(pool_id, json.error) : "";
  let isOk   = "result" in json && json.result !== null && json.result !== false;
  const handler = poolResponseHandler(pool_id, json.id);
  // Conflux submit replies may use [accepted, reason]; array truthiness is not acceptance.
  if (handler === handleShareResponse && poolProtocol(poolAt(pool_id)) === "conflux" &&
      Array.isArray(json.result)) {
    isOk = json.result[0] === true;
    if (!isError && !isOk) {
      errorText = poolErrorText(pool_id, json.result[1]);
    }
  }
  const result = handler(pool_id, isError, isOk, errorText, json, setJob);
  if (handler !== ignorePoolResponse && !isError && isOk && json.id === 1) {
    rememberPoolResponseMetadata(pool_id, json.result);
  }
  return result;
}

// Store worker id, negotiated extensions, extranonce, and dialect from a successful login result.
/** @param {number} pool_id @param {unknown} result @returns {void} */
function rememberPoolResponseMetadata(pool_id, result) {
  if (!isObject(result)) {
    return;
  }

  const pool = poolAt(pool_id);
  const workerId = result["id"];
  if (typeof workerId === "string" ||
      (typeof workerId === "number" && Number.isSafeInteger(workerId))) {
    pool.worker_id = workerId;
  }
  applyLoginExtensions(pool_id, result["extensions"]);
  const algo = messageAlgorithm(result);
  if (algo && pool.extensions?.includes("mo-native")) {
    preparePoolFamily(pool_id, algo);
  }
  rememberPoolExtraNonceHex(pool_id, result["extra_nonce"]);
  rememberExtraNonceSize(pool, result["extra_nonce2_size"]);
  rememberPoolProtocol(pool_id, result);
}

/** @returns {undefined} */
function ignorePoolResponse() {
  return undefined;
}

/** @param {unknown} id @returns {unknown} */
function normalizedResponseId(id) {
  if (typeof id !== "string" || !/^(?:0|[1-9]\d*)$/.test(id)) {return id;}
  const numeric = Number(id);
  return Number.isSafeInteger(numeric) ? numeric : id;
}

// Pick the response handler from the reply id and the dialect's request numbering. Replies that do
// not match an outstanding request are ignored.
/** @param {number} pool_id @param {unknown} id @returns {PoolResponseHandler} */
function poolResponseHandler(pool_id, id) {
  const pool = poolAt(pool_id);
  if (typeof id !== "string" &&
      !(typeof id === "number" && Number.isSafeInteger(id))) {
    return ignorePoolResponse;
  }
  if (usesCortex(pool)) {
    if (id === 72) {
      return pool.pending_cortex_login ? handleLoginResponse : ignorePoolResponse;
    }
    if (id === 100) {
      return pool.pending_cortex_work ? handleCortexWorkResponse : ignorePoolResponse;
    }
  } else if (usesIronfish(pool)) {
    return ignorePoolResponse;
  } else if (id === 1 && pool.requested_extensions !== undefined) {
    return handleLoginResponse;
  } else if (pool.use_subscribe === false) {
    if (id === 1) {
      return handleLoginResponse;
    }
    if (id === 2) {
      return ignorePoolResponse;
    }
  } else if (poolProtocol(pool) === "conflux") {
    if (id === 1) {
      return handleLoginResponse;
    }
  } else if (pearlhashUsesSubscribe(pool)) {
    if (id === 1) {
      return ignorePoolResponse;
    } // subscribe ack/err (authorize already sent)
    if (id === 2) {
      return pool.pending_authorize ? handleAuthorizeResponse : ignorePoolResponse;
    }
  } else if (usesMiningSubscribe(pool)) {
    if (id === 1) {
      return pool.pending_subscribe ? handleSubscribeResponse : ignorePoolResponse;
    } // mining.subscribe response
    if (id === 2) {
      return pool.pending_authorize ? handleAuthorizeResponse : ignorePoolResponse;
    }
  } else {
    if (id === 1) {
      return handleLoginResponse;
    } // login response
    if (id === 2) {
      return ignorePoolResponse;
    } // keepalive response
  }
  if (usesCortex(pool)) {
    // IDs 72 and 100 are consumed above by login/getWork. Each submit uses the next safe integer
    // from 73 so concurrent replies remain distinguishable and may arrive out of order.
    return typeof id === "number" && pool.pending_cortex_submit_ids?.delete(id)
      ? handleShareResponse : ignorePoolResponse;
  }
  if (id === 3 && pool.pending_submit_count > 0) {
    --pool.pending_submit_count;
    return handleShareResponse;
  }
  return ignorePoolResponse;
}

// Iron Fish handshake replies (mining.subscribed / mining.submitted) are METHOD pushes, NOT
// {id,result} responses, and Iron Fish reuses ids across messages -- so they must be matched by
// method, never routed through the id-keyed handlePoolResponse.
/** @param {number} pool_id @param {PoolMessage} json @returns {void} */
function handleIronfishSubscribed(pool_id, json) {
  const pool = poolAt(pool_id);
  const body = isObject(json.body) ? json.body : {};
  const xn = body["xn"];
  const extraNonce = xn === undefined || xn === "" ? "" : validExtraNonce(xn);
  if (!extraNonce && xn !== undefined && xn !== "") {
    throw new Error("Invalid Iron Fish extranonce");
  }
  pool.ironfish_xn = extraNonce;
  return loginSucceeded(pool_id);
}

/** @param {number} pool_id @param {PoolMessage} json @returns {void} */
function handleIronfishSubmitted(pool_id, json) {
  const ok = isObject(json.body) && json.body["result"] === true;
  return handleShareResponse(pool_id, !ok, ok, ok ? "" : ": rejected");
}

// Handle Iron Fish handshake/submit pushes and errors; returns true when the message was consumed.
/**
 * @param {number} pool_id
 * @param {PoolMessage} json
 * @param {SetJobCallback} setJob
 * @returns {boolean}
 */
function handleIronfishMessage(pool_id, json, setJob) {
  if (poolProtocol(poolAt(pool_id)) !== "ironfish") {
    return false;
  }
  if (json.error !== undefined && json.error !== null) {
    const pool = poolAt(pool_id);
    const nestedId = isObject(json.error) ? json.error["id"] : undefined;
    const responseId = nestedId ?? json.id;
    const errMsg = poolErrorText(pool_id, json.error);
    if (json.method === "mining.subscribed") {
      delete poolAt(pool_id).pending_job;
      handleLoginResponse(pool_id, true, false, errMsg, json);
      return true;
    }
    if (json.method === "mining.submitted") {
      handleShareResponse(pool_id, true, false, errMsg);
      return true;
    }
    if (responseId === 1 || responseId === "1") {
      delete poolAt(pool_id).pending_job;
      handleLoginResponse(pool_id, true, false, errMsg, json);
      return true;
    }
    if (responseId === 2 || responseId === "2") {
      handleShareResponse(pool_id, true, false, errMsg);
      return true;
    }
    if (!pool.logged_in || pool.pending_job) {
      delete pool.pending_job;
      handleLoginResponse(pool_id, true, false, errMsg, json);
    } else {
      poolLogErr(pool_id, "Iron Fish error" + errMsg);
    }
    return true;
  }
  if (json.method === "mining.subscribed") {
    handleIronfishSubscribed(pool_id, json);
    const pool = poolAt(pool_id);
    const pendingJob = pool.pending_job;
    delete pool.pending_job;
    if (pendingJob) {routePoolMessage(pool_id, pendingJob, setJob);}
    return true;
  }
  if (json.method === "mining.submitted") {
    handleIronfishSubmitted(pool_id, json);
    return true;
  }
  return false;
}

// Beam replies (to login and to solution submits) are `method:"result"` messages carrying a `code`
// field (0 = login OK, 1 = share accepted; anything else = error/reject) plus an optional description.
// The login reply may carry `nonceprefix` (0-6 bytes) -- the 8-byte mining nonce's leading bytes
// MUST match it, so we stash a supplied prefix and seed the job nonce + nicehash mask from it.
/** @param {PoolMessage} json @returns {json is BeamResultMessage} */
function isBeamResult(json) {
  return json.method === "result" && typeof json["code"] === "number" &&
    Number.isSafeInteger(json["code"]);
}

/** @param {number} pool_id @param {BeamResultMessage} json @returns {void} */
function handleBeamResult(pool_id, json) {
  const pool = poolAt(pool_id);
  const desc = poolErrorText(pool_id, json.description);
  if (json.id === "login" || "nonceprefix" in json) {
    if (json.code === 0) {
      if (json.nonceprefix !== undefined) {
        if (typeof json.nonceprefix !== "string") {
          return loginFailed(pool_id, ": Invalid Beam nonce prefix");
        }
        const prefix = hexWithoutPrefix(json.nonceprefix);
        if (!/^(?:[0-9a-f]{2}){0,6}$/i.test(prefix)) {
          return loginFailed(pool_id, ": Invalid Beam nonce prefix");
        }
        pool.beam_nonceprefix = prefix;
      }
      return loginSucceeded(pool_id);
    }
    return loginFailed(pool_id, desc || ": Login rejected");
  }
  return handleShareResponse(pool_id, false, json.code === 1, desc);
}

// Dispatch one parsed pool message: normalize response ids, queue or apply controls (mo-native
// pools defer controls until the job for their algorithm arrives), hold jobs that arrive before
// authorization, then decode jobs or route responses. Throws on malformed protocol data; the
// connection layer turns that into a pool error.
/**
 * @param {number} pool_id
 * @param {PoolMessage} json
 * @param {SetJobCallback} setJob
 * @returns {void}
 */
function routePoolMessage(pool_id, json, setJob) {
  const responseEnvelope = typeof json.method !== "string" && Object.hasOwn(json, "id") &&
    (Object.hasOwn(json, "result") || Object.hasOwn(json, "error"));
  if (responseEnvelope) {
    const responseId = normalizedResponseId(json.id);
    if (responseId !== json.id) {json = {...json, id: responseId};}
  }
  // A login response must be routed using the dialect that sent the login. Pushed messages may
  // announce the assigned algorithm before their protocol-specific fields are decoded.
  const pool = poolAt(pool_id);
  const nativeSwitching = pool.extensions?.includes("mo-native") === true;
  const jobAnnouncement = json.method === "job" || json.method === "mining.notify";
  // A Pearl login reply is a bare `true`, so negotiated extensions may be unknown; a marked job must
  // still name an advertised algorithm before it can select its family.
  const markedAlgo = jobAnnouncement ? messageAlgorithm(json) : "";
  if (markedAlgo && pool.requested_algos?.length && !pool.requested_algos.includes(markedAlgo)) {
    throw new Error("Pool assigned an unadvertised algorithm: " + markedAlgo);
  }
  if (!nativeSwitching && typeof json.method === "string") {
    rememberPoolProtocol(pool_id, json);
  }
  const kind = controlKind(json);
  if (kind) {
    if (!nativeSwitching) {return applyPoolControl(pool_id, json);}
    const algo = messageAlgorithm(json);
    if (algo && !pool.requested_algos?.includes(algo)) {return;}
    // Retain one control per kind/advertised family (plus unmarked). A stale marked control
    // must not overwrite an unmarked setting intended for the next job.
    pool.pending_controls = (pool.pending_controls || []).filter((pending) =>
      controlKind(pending) !== kind || messageAlgorithm(pending) !== algo);
    pool.pending_controls.push(json);
    return;
  }
  if (nativeSwitching && jobAnnouncement) {
    const algo = markedAlgo || pool.job_algo;
    if (!algo) {throw new Error("Native pool job has no algorithm");}
    preparePoolFamily(pool_id, algo);
    const controls = pool.pending_controls || [];
    delete pool.pending_controls;
    for (const control of controls) {
      const markedAlgo = messageAlgorithm(control);
      if (!markedAlgo || markedAlgo === algo) {applyPoolControl(pool_id, control);}
    }
    json = {...json, algo};
  }
  if (poolProtocol(pool) === "beam" && isBeamResult(json)) {
    return handleBeamResult(pool_id, json);
  }
  if (handleIronfishMessage(pool_id, json, setJob)) {
    return;
  }
  if (poolProtocol(pool) === "xelis" && json.method === "mining.ping") {
    return module.exports.pool_write(pool_id, {jsonrpc: "2.0", id: json.id, method: "mining.pong"});
  }
  if (!pool.logged_in && jobAnnouncement && (pool.pending_authorize || usesIronfish(pool))) {
    pool.pending_job = json;
    return;
  }
  if (usesCortex(pool) && json.id === 100 && pool.pending_cortex_work &&
      "error" in json && json.error !== null) {
    return handlePoolResponse(pool_id, json, setJob);
  }
  // Miningcore answers the initial getWork as id 100, then pushes new protocol-v1 work as id 0.
  if (usesCortex(pool) && json.id === 100 && !pool.pending_cortex_work) {return;}
  const job = jobFromPoolMessage(pool_id, json);
  if (job) {
    if (usesCortex(pool) && json.id === 100) {pool.pending_cortex_work = false;}
    if (json.id === 1 && isObject(json.result) && Object.hasOwn(json.result, "job")) {
      rememberPoolResponseMetadata(pool_id, json.result);
    }
    return handlePoolJob(pool_id, job, setJob);
  }
  if (nativeSwitching && jobAnnouncement) {
    throw new Error("Invalid native pool job");
  }
  if (responseEnvelope) {
    return handlePoolResponse(pool_id, json, setJob);
  }

  poolLogJson(pool_id, 1, "Unknown message from the pool: ", json);
}

const {connectPoolThrottle} = require("./pool/connection")({
  h, o, net, tls, systemNetConnect, systemTlsConnect, max_pool_data_buffer: MAX_POOL_DATA_BUFFER,
  clear_pool_connection: clearPoolConnection, isCurrentPoolSocket, pearlhashUsesSubscribe,
  poolProtocol, pool_log: poolLog, pool_log_str: poolLogText,
  pool_log1: (pool_id, str) => h.log1(poolLogText(pool_id, str)),
  poolErrorText, pool_log_json: poolLogJson, pool_message: receivePoolMessage, pool_str: poolName,
  usesCortex, usesIronfish, usesMiningSubscribe,
  poolWrite: (...args) => module.exports.pool_write(...args),
  switchPool: (...args) => module.exports.switch_pool(...args),
});
module.exports.connect_pool_throttle = connectPoolThrottle;
