// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

"use strict";

const net = require("node:net");
const tls = require("node:tls");
const h = require("./helper.js");
const o = require("./opts.js");
const {normalizeAlgoName} = require("./miner/algorithms");

// Correctness tests may use normal network services, but must never contact a mining pool. Capture
// the actual socket functions before logic tests substitute their in-memory sockets, so the guard
// blocks only a real pool connection and leaves the protocol fixtures testable.
const systemNetConnect = net.connect;
const systemTlsConnect = tls.connect;

const max_pool_data_buffer = 1024 * 1024;
const progpowAlgos = new Set(["kawpow", "firopow", "evrprogpow", "meowpow"]);
const fullTargetAlgos = new Set([
  "etchash", "octopus", "autolykos2", "fishhash", "c30", "equihash192_7", "zelhash", "zhash",
  "karlsenhashv2", "walahash", "hoohash", "verthash", "xelishashv3", "nexapow",
]);

/** @typedef {(job: PoolJob) => MiningJob} SetJobCallback */
/** @typedef {(pool_id: number, is_err: boolean, is_ok: boolean, err_msg: string,
 * json: PoolMessage, set_job: SetJobCallback) => void} PoolResponseHandler */
/** @typedef {PoolMessage & {code: number, description?: unknown, nonceprefix?: unknown}} BeamResultMessage */

/**
 * Resolve a pool id at the boundary where pool state is consumed. Pool ids are produced by the
 * validated options layer, but this guard also keeps malformed protocol/test inputs from turning an
 * out-of-range array access into an unrelated TypeError.
 * @param {number} pool_id
 * @returns {PoolConfig}
 */
function poolAt(pool_id) {
  if (!Number.isInteger(pool_id) || pool_id < 0 || pool_id >= global.opt.pools.length) {
    throw new Error("Invalid pool id");
  }
  const pool = global.opt.pools[pool_id];
  if (!pool) {
    throw new Error("Invalid pool id");
  }
  return pool;
}

/** @param {number} pool_id @returns {string} */
function pool_str(pool_id) {
  const pool = poolAt(pool_id);
  return pool.url + ":" + pool.port + (pool.is_tls ? "tls" : "");
}

/** @param {number} pool_id @param {string} str @returns {string} */
function pool_log_str(pool_id, str) {
  return global.opt.log_level >= 1 ? "[" + pool_str(pool_id) + "] " + str : str;
}

/** @param {number} pool_id @param {string} str @returns {void} */
function pool_log(pool_id, str) {
  h.log(pool_log_str(pool_id, str));
}

/** @param {number} pool_id @param {string} str @returns {void} */
function pool_log1(pool_id, str) {
  h.log1(pool_log_str(pool_id, str));
}

/** @param {number} pool_id @param {string} str @returns {void} */
function pool_log2(pool_id, str) {
  h.log2(pool_log_str(pool_id, str));
}

/** @param {number} pool_id @param {string} str @returns {void} */
function pool_log_err(pool_id, str) {
  h.log_err(pool_log_str(pool_id, str));
}

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
function pool_log_json(pool_id, level, prefix, json) {
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
  (level === 1 ? pool_log1 : pool_log2)(pool_id, prefix + message);
}

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

/** @param {number} pool_id @param {PoolSocket | null} socket @returns {boolean} */
function clear_pool_connection(pool_id, socket) {
  const pool = poolAt(pool_id);
  if (socket && !isCurrentPoolSocket(pool_id, socket)) {
    return false;
  }
  clearPoolKeepalive(pool);
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

/** @param {number} pool_id @param {PoolSocket} socket @returns {boolean} */
function isCurrentPoolSocket(pool_id, socket) {
  return poolAt(pool_id).socket === socket;
}

// Maps a mining algo to its stratum protocol dialect, or null if it uses the default `login` dialect.
/** @param {string | null | undefined} algo @returns {string | null} */
function protocolForAlgo(algo) {
  switch (normalizeAlgoName(algo)) {
    case "kawpow":
    case "firopow":
    case "evrprogpow":
    case "meowpow":
      return "raven";
    case "etchash":
      return "eth";
    case "octopus":
      return "conflux";
    case "nexapow":
      return "echelon";
    case "c30":
      return "cortex";
    case "autolykos2":
      return "erg";
    case "pearlhash":
      return "pearlhash";
    case "fishhash":
      return "ironfish";
    case "equihash192_7":
    case "zelhash":
    case "zhash":
      return "zelhash";
    case "beamhash3":
      return "beam";
    case "karlsenhashv2":
      return "kaspa";
    case "walahash":
      return "kaspa";
    case "hoohash":
      return "hoosat";
    case "verthash":
      return "verthash";
    case "xelishashv3":
      return "xelis";
    default:
      return null;
  }
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

/** @param {number} pool_id @param {UnknownRecord} json @returns {void} */
module.exports.pool_write = function(pool_id, json) {
  const message = JSON.stringify(json);
  const pool = poolAt(pool_id);
  if (!pool.socket) {
    return pool_log_json(pool_id, 2, "Sent to the closed pool socket: ", json);
  }

  pool_log_json(pool_id, 2, "Sent to the pool: ", json);
  if (json["id"] === 3 && (json["method"] === "mining.submit" || json["method"] === "submit")) {
    ++pool.pending_submit_count;
  }
  pool.socket.write(message + "\n");
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

// soft kill pool connection
/** @param {number} pool_id @returns {void} */
function pool_close_wait(pool_id) {
  const socket = poolAt(pool_id).socket;
  if (!socket) {
    return;
  }
  pool_log1(pool_id, "Soft closing the pool connection");
  setTimeout(function() {
    // do not do soft close if this pool became active again
    if (pool_id === global.opt.pool_ids.active ||
        !isCurrentPoolSocket(pool_id, socket)) {
      return;
    }
    pool_log1(pool_id, "Soft closed the pool connection");
    clear_pool_connection(pool_id, socket);
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
  const worker = separator < 0 ? pool.worker : login.slice(separator + 1);
  return [wallet, worker || "mom", pool.pass];
}

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
  const size = typeof value === "number" ? value :
    typeof value === "string" && /^\d+$/.test(value) ? Number(value) : Number.NaN;
  if (!Number.isInteger(size) || size < 0 || size > 8) {
    throw new Error("Invalid extra_nonce2_size");
  }
  pool.extra_nonce2_size = size;
}

/** @param {PoolMessage} json @returns {string} */
function controlKind(json) {
  if (isSetTargetNotification(json) || isIronfishSetTargetNotification(json)) {return "target";}
  if (isSetDifficultyNotification(json)) {return "difficulty";}
  return isSetExtranonceNotification(json) ? "extranonce" : "";
}

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
  pearlhashDiffFromJobId, beamPackedTarget, pool_close_wait,
  pool_log, pool_str, algoFromPass,
  connectPoolThrottle: (...args) => module.exports.connect_pool_throttle(...args),
});
const {
  isObject, isIronfishSetTargetNotification, isSetTargetNotification,
  isSetDifficultyNotification, isSetExtranonceNotification, hexWithoutPrefix,
  validExtraNonce, rememberPoolExtraNonceHex, rememberSubscribeExtraNonce,
  switchPool, handleSetTarget, rememberXelisExtranonce, handleSetDifficulty,
  jobFromPoolMessage,
} = poolJobs;
module.exports.switch_pool = switchPool;
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
  if (progpowAlgos.has(job.algo)) {
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

/** @param {number} pool_id @param {number} active_pool @returns {void} */
function activatePoolForJob(pool_id, active_pool) {
  // only switch active pool once for its first job here
  if (pool_id === active_pool || poolAt(pool_id).last_job) {
    return;
  }
  if (pool_id === global.opt.pool_ids.primary) {
    pool_log(pool_id, "Switching active pool to primary " + pool_str(pool_id) + " pool");
    pool_close_wait(active_pool);
    global.opt.pool_ids.active = pool_id;
  } else if (pool_id === global.opt.pool_ids.donate) {
    pool_log(pool_id, "Switching active pool to donate " + pool_str(pool_id) + " pool");
    global.opt.pool_ids.active = pool_id;
  }
}

/**
 * @param {number} pool_id
 * @param {PoolJob} job
 * @param {SetJobCallback} set_job
 * @returns {void}
 */
function handlePoolJob(pool_id, job, set_job) {
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
  if (!donationJobSupported(pool_id, job, set_job)) {return;}
  if (job.target) {
    // Pearl's K/rank are selected by set_job; validate the final threshold before that selection.
    if (algo === "pearlhash") {h.target256ToWork(job.target);} else {jobTargetWork(job);}
  } // throws early on a malformed target before we store the job
  activatePoolForJob(pool_id, global.opt.pool_ids.active);

  job.submit_mode ??= null;
  job.protocol = poolProtocol(pool);
  job.submit_result = pool.extensions?.includes("submit-result") === true;
  pool.last_job = job;
  if (pool_id === global.opt.pool_ids.active) {
    const last_job = set_job(job);
    pool_log(pool_id, "Got new " + last_job.algo + " algo job with " +
                     jobTargetDescription(last_job) +
                     (last_job.height ? " and " + last_job.height + " height" : "")
    );
  } else {
    pool_log_json(pool_id, 2, "Storing not active pool job ", job);
  }
}

/**
 * @param {number} pool_id
 * @param {PoolJob} job
 * @param {SetJobCallback} set_job
 * @returns {boolean}
 */
function donationJobSupported(pool_id, job, set_job) {
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
    pool_log_err(pool_id, "Donation pool returned unsupported " + String(job.algo) + " work");
  }
  if (pool_id === global.opt.pool_ids.active) {
    switchPool(pool_id, set_job);
  } else {
    clear_pool_connection(pool_id, pool.socket);
  }
  return false;
}

/** @param {number} pool_id @returns {void} */
function loginSucceeded(pool_id) {
  poolAt(pool_id).logged_in = true;
  return pool_log(pool_id, "Login to the pool succeeded");
}

/** @param {number} pool_id @param {string} reason @returns {void} */
function loginFailed(pool_id, reason) {
  poolAt(pool_id).logged_in = false;
  return pool_log_err(pool_id, "Login to the pool failed" + reason);
}

/**
 * @param {number} pool_id
 * @param {boolean} is_err
 * @param {boolean} is_ok
 * @param {string} err_msg
 * @param {PoolMessage} _json
 * @returns {void}
 */
function handleLoginResponse(pool_id, is_err, is_ok, err_msg, _json) {
  const pool = poolAt(pool_id);
  if (usesCortex(pool)) {pool.pending_cortex_login = false;}
  if (is_err || !is_ok) {
    delete pool.pending_job;
    return loginFailed(pool_id, err_msg || ": Login rejected");
  }
  if (is_ok) {
    loginSucceeded(pool_id);
    if (usesCortex(pool)) {
      pool.pending_cortex_work = true;
      return module.exports.pool_write(pool_id, {
        id: 100, jsonrpc: "2.0", method: "ctxc_getWork", params: [""],
      });
    }
  }
}

/**
 * @param {number} pool_id
 * @param {boolean} is_err
 * @param {boolean} is_ok
 * @param {string} err_msg
 * @param {PoolMessage} json
 * @returns {void}
 */
function handleSubscribeResponse(pool_id, is_err, is_ok, err_msg, json) {
  poolAt(pool_id).pending_subscribe = false;
  if (is_err) {
    delete poolAt(pool_id).pending_job;
    return pool_log_err(pool_id, "Subscribe to the pool failed" + err_msg);
  }
  if (!is_ok) {
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
 * @param {boolean} is_err
 * @param {boolean} _is_ok
 * @param {string} err_msg
 * @param {PoolMessage} json
 * @param {SetJobCallback} set_job
 * @returns {void}
 */
function handleAuthorizeResponse(pool_id, is_err, _is_ok, err_msg, json, set_job) {
  const pool = poolAt(pool_id);
  pool.pending_authorize = false;
  if (!is_err && json.result === true) {
    loginSucceeded(pool_id);
    const pendingJob = pool.pending_job;
    delete pool.pending_job;
    if (pendingJob) {
      return pool_message(pool_id, pendingJob, set_job);
    }
    return;
  }
  delete pool.pending_job;
  return loginFailed(pool_id, err_msg || ": Authorization rejected");
}

/**
 * @param {number} pool_id
 * @param {boolean} is_err
 * @param {boolean} is_ok
 * @param {string} err_msg
 * @returns {void}
 */
function handleShareResponse(pool_id, is_err, is_ok, err_msg) {
  const pool = poolAt(pool_id);
  if (is_err || is_ok === false) {
    ++pool.bad_shares;
    return pool_log_err(pool_id, "Share rejected by the pool " + poolShareStats(pool_id) + err_msg);
  }
  if (is_ok) {
    ++pool.good_shares;
    return pool_log(pool_id, "Share accepted by the pool " + poolShareStats(pool_id));
  }
}

/**
 * @param {number} pool_id
 * @param {boolean} is_err
 * @param {boolean} _is_ok
 * @param {string} err_msg
 * @returns {void}
 */
function handleCortexWorkResponse(pool_id, is_err, _is_ok, err_msg) {
  poolAt(pool_id).pending_cortex_work = false;
  return pool_log_err(pool_id, "Cortex getWork failed" +
    (is_err && err_msg ? err_msg : ": Invalid work response"));
}

/** @param {number} pool_id @param {PoolMessage} json @param {SetJobCallback} set_job @returns {void} */
function handlePoolResponse(pool_id, json, set_job) {
  const is_err  = "error" in json && json.error !== null;
  let err_msg = is_err ? poolErrorText(pool_id, json.error) : "";
  let is_ok   = "result" in json && json.result !== null && json.result !== false;
  const handler = poolResponseHandler(pool_id, json.id);
  // Conflux submit replies may use [accepted, reason]; array truthiness is not acceptance.
  if (handler === handleShareResponse && poolProtocol(poolAt(pool_id)) === "conflux" &&
      Array.isArray(json.result)) {
    is_ok = json.result[0] === true;
    if (!is_err && !is_ok) {
      err_msg = poolErrorText(pool_id, json.result[1]);
    }
  }
  const result = handler(pool_id, is_err, is_ok, err_msg, json, set_job);
  if (handler !== ignorePoolResponse && !is_err && is_ok && json.id === 1) {
    rememberPoolResponseMetadata(pool_id, json.result);
  }
  return result;
}

/** @param {number} pool_id @param {unknown} result @returns {void} */
function rememberPoolResponseMetadata(pool_id, result) {
  if (!isObject(result)) {
    return;
  }

  const pool = poolAt(pool_id);
  const worker_id = result["id"];
  if (typeof worker_id === "string" ||
      (typeof worker_id === "number" && Number.isSafeInteger(worker_id))) {
    pool.worker_id = worker_id;
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

/**
 * @param {number} pool_id
 * @param {PoolMessage} json
 * @param {SetJobCallback} set_job
 * @returns {boolean}
 */
function handleIronfishMessage(pool_id, json, set_job) {
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
      pool_log_err(pool_id, "Iron Fish error" + errMsg);
    }
    return true;
  }
  if (json.method === "mining.subscribed") {
    handleIronfishSubscribed(pool_id, json);
    const pool = poolAt(pool_id);
    const pendingJob = pool.pending_job;
    delete pool.pending_job;
    if (pendingJob) {pool_message(pool_id, pendingJob, set_job);}
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

/**
 * @param {number} pool_id
 * @param {PoolMessage} json
 * @param {SetJobCallback} set_job
 * @returns {void}
 */
function pool_message(pool_id, json, set_job) {
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
    const algo = messageAlgorithm(json) || pool.job_algo;
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
  if (handleIronfishMessage(pool_id, json, set_job)) {
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
    return handlePoolResponse(pool_id, json, set_job);
  }
  // Miningcore answers the initial getWork as id 100, then pushes new protocol-v1 work as id 0.
  if (usesCortex(pool) && json.id === 100 && !pool.pending_cortex_work) {return;}
  const job = jobFromPoolMessage(pool_id, json);
  if (job) {
    if (usesCortex(pool) && json.id === 100) {pool.pending_cortex_work = false;}
    if (json.id === 1 && isObject(json.result) && Object.hasOwn(json.result, "job")) {
      rememberPoolResponseMetadata(pool_id, json.result);
    }
    return handlePoolJob(pool_id, job, set_job);
  }
  if (nativeSwitching && jobAnnouncement) {
    throw new Error("Invalid native pool job");
  }
  if (responseEnvelope) {
    return handlePoolResponse(pool_id, json, set_job);
  }

  pool_log_json(pool_id, 1, "Unknown message from the pool: ", json);
}

const {connectPoolThrottle} = require("./pool/connection")({
  h, o, net, tls, systemNetConnect, systemTlsConnect, max_pool_data_buffer,
  clear_pool_connection, isCurrentPoolSocket, pearlhashUsesSubscribe,
  poolProtocol, pool_log, pool_log1, pool_log_str,
  poolErrorText,
  pool_log_json,
  pool_message, pool_str, usesCortex, usesIronfish, usesMiningSubscribe,
  poolWrite: (...args) => module.exports.pool_write(...args),
  switchPool: (...args) => module.exports.switch_pool(...args),
});
module.exports.connect_pool_throttle = connectPoolThrottle;
