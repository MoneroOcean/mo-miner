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
  if (pool.socket) pool.socket.destroy();
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

function isCurrentPoolSocket(pool_id, socket) {
  return global.opt.pools[pool_id].socket === socket;
}

// Maps a mining algo to its stratum protocol dialect, or null if it uses the default `login` dialect.
function protocolForAlgo(algo) {
  switch (algo) {
    case "kawpow":     return "raven";
    case "firopow":    return "raven";
    case "evrprogpow": return "raven";
    case "meowpow":    return "raven";
    case "etchash":    return "eth";
    case "autolykos2": return "erg";
    case "pearl":      return "pearl";
    case "fishhash":   return "ironfish";
    case "equihash125_4": return "equihash";
    case "beamhash3":  return "beam";
    case "kheavyhash":  return "kaspa";
    case "karlsenhashv2": return "kaspa";
    case "pyrinhashv2": return "kaspa";
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
         protocol === "equihash" || protocol === "kaspa";
}

function usesEthProxy(pool) {
  return poolProtocol(pool) === "ethproxy";
}

function usesIronfish(pool) {
  return poolProtocol(pool) === "ironfish";
}

// Standard Pearl handshake (HeroMiners/LuckyPool/etc.): mining.subscribe + mining.authorize
// {wallet,worker,pass}. This is the DEFAULT for pearl pools. pearlpool.cloud uses the older single
// `login` dialect instead -- opt OUT of subscribe there with "use_subscribe": false (the MoneroOcean
// donate pool also sets it false so donation keeps using login). Both dialects push the same
// object-param pearl mining.notify and take the same mining.submit{job_id,plain_proof}.
function pearlUsesSubscribe(pool) {
  // MOM_PEARL_LOGIN env forces the legacy login dialect (pearlpool.cloud) for CLI mining.
  // The MO donate pool opts out via use_subscribe:false, so this env never affects donation.
  if (process.env.MOM_PEARL_LOGIN) return false;
  return poolProtocol(pool) === "pearl" && pool.use_subscribe !== false;
}

// Pearl difficulty is carried in the job_id suffix "<hex>_<diff>" (HeroMiners omits the difficulty
// field that pearlpool.cloud sends); used to derive the 2^256/diff kernel target.
function pearlDiffFromJobId(job_id) {
  const m = String(job_id || "").match(/_(\d+)$/);
  return m ? Number(m[1]) : undefined;
}

// k - k%rank (the "dot_product_length"), from the same env the native kernel reads. Defaults MUST
// match the native pearl_k()/pearl_rank() (4096/256) or the JS-computed jackpot bound disagrees with
// the kernel/verifier and shares come out too rare.
function pearlKEff() {
  const k = Number(process.env.MOM_PEARL_K) || 4096, rank = Number(process.env.MOM_PEARL_RANK) || 256;
  return Math.floor(k / rank) * rank;
}

// HeroMiners sends the BASE target T0 (= nbits_to_difficulty(share_nbits)); the actual jackpot bound
// the verifier checks is T0 * (16*16) * (k - k%rank)  (zk-pow extract_difficulty_bound: tile_size *
// dot_product_length). pearlpool instead accepts the lenient 2^256/diff and its target field is the
// network block target (ignored).
function pearlNbitsBound(baseTargetHex) {
  const MAX = (1n << 256n) - 1n;
  const base = BigInt("0x" + (hexWithoutPrefix(baseTargetHex) || "0"));
  let bound = base * BigInt(16 * 16 * pearlKEff());
  if (bound > MAX) bound = MAX;
  return bound.toString(16).padStart(64, "0");
}

module.exports.pool_write = function(pool_id, json) {
  const message = JSON.stringify(json);
  const pool = global.opt.pools[pool_id];
  if (!pool.socket) return pool_log2(pool_id, "Sent to the closed pool socket: " + message);

  pool_log2(pool_id, "Sent to the pool: " + message);
  pool.socket.write(message + "\n");
  // sends keepalive if no submit/keepalive to pool for more than global.opt.pool_time.keepalive
  if (!pool.is_keepalive || usesMiningSubscribe(pool) || usesEthProxy(pool) || pearlUsesSubscribe(pool) || usesIronfish(pool)) return;
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
  if (!socket) return;
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

function poolErrorText(error) {
  return error instanceof Object && typeof error.message === "string" ? ": " + error.message : "";
}

function applyLoginExtensions(pool_id, extensions) {
  if (!Array.isArray(extensions)) return;
  const pool = global.opt.pools[pool_id];
  if (extensions.includes("nicehash")) pool.is_nicehash = true;
  if (extensions.includes("keepalive")) pool.is_keepalive = true;
}

function algoFromPass(pool) {
  const pass = String(pool.pass || "");
  const m = pass.match(/(?:^|[~;,])(?:algo=)?(kawpow|firopow|evrprogpow|meowpow|etchash|autolykos2|pearl|fishhash|equihash125_4)(?:$|[~;,])/i);
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

// ZIP-301 EquihashStratum (Zcash/Flux) mining.notify carries 8 array fields:
// [job_id, version(LE 8hex), prevhash(64), merkleroot(64), reserved(64), time(8), bits(8), clean].
function isEquihashJobNotification(json) {
  return json.method === "mining.notify" && Array.isArray(json.params) && json.params.length >= 8;
}

// Beam JSON-RPC `job`: TOP-LEVEL {input(64hex), difficulty(int32), id, method:"job"} (no params array).
function isBeamJobNotification(json) {
  return json.method === "job" && typeof json.input === "string";
}

// Kaspa (kheavyhash) kaspa-stratum-bridge mining.notify carries 3 params:
// [jobId(string), [u0,u1,u2,u3] (4 uint64 LE pre-pow words), timestamp(ms uint64)]. The 4 words are the
// 32-byte BLAKE2b pre-pow hash (TS=0,Nonce=0) split into little-endian uint64s. They overflow JS
// Number, so parsePoolLine re-extracts them from the raw line into json.__kaspa_words (decimal strings).
function isKaspaJobNotification(json) {
  return json.method === "mining.notify" && Array.isArray(json.params) &&
         json.params.length >= 3 && Array.isArray(json.params[1]) && json.params[1].length >= 4;
}

// pearlpool.cloud pushes mining.notify with OBJECT params {job_id, header, target, difficulty, height, mode}
function isPearlJobNotification(json) {
  return json.method === "mining.notify" && isObject(json.params) && typeof json.params.header === "string";
}

// Iron Fish uses a custom OBJECT-based stratum: every push is {id, method, body:{...}} (NOT params).
// mining.notify carries the 180-byte block header (first 8 bytes = randomness, leading bytes == xn).
function isIronfishJobNotification(json) {
  return json.method === "mining.notify" && isObject(json.body) && typeof json.body.header === "string";
}

// Iron Fish mining.set_target carries a 64-hex BE 256-bit target in body.target.
function isIronfishSetTargetNotification(json) {
  return json.method === "mining.set_target" && isObject(json.body) && typeof json.body.target === "string";
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

// Kaspa pushes a standalone set_extranonce (NO "mining." prefix) carrying [extranonce_hex, size]; it
// also rides in the subscribe result. Either way the extranonce becomes the leading bytes of the nonce.
function isSetExtranonceNotification(json) {
  return (json.method === "set_extranonce" || json.method === "mining.set_extranonce") &&
         Array.isArray(json.params) && json.params.length >= 1;
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

function poolExtraNonce(pool) {
  return hexWithoutPrefix(pool.extra_nonce || "");
}

function poolNonce(pool) {
  return poolExtraNonce(pool).padEnd(16, "0").slice(0, 16);
}

function poolNonceMask(pool) {
  return "ff".repeat(poolExtraNonce(pool).length / 2).padEnd(16, "0");
}

function nonceAt32Blob(headerHash, pool) {
  return headerHash + fixedHexBytesLE(poolExtraNonce(pool), 8);
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
    extra_nonce: poolExtraNonce(pool),
    extra_nonce2_size: pool.extra_nonce2_size,
    ntime: job.ntime || "",
  };

  const jobIds = Object.keys(pool.erg_submit_jobs);
  while (jobIds.length > 16) delete pool.erg_submit_jobs[jobIds.shift()];
}

// Build the Equihash 125,4 (Flux/ZIP-301) job from a mining.notify. The 8 notify fields go straight
// into the 140-byte Zcash header at the fixed offsets (prev/merkle/reserved already in header byte
// order -- concat directly, NO reversal); the 32-byte nonce at offset 108 starts as nonce1 (the
// subscribe extranonce prefix) followed by a zero nonce2 region the solver fills. The solver's 8-byte
// search counter is written at nonceoffset = 108 + nonce1_len so it lands inside nonce2, never
// clobbering the pool's fixed nonce1 prefix.
function equihashNotifyJob(pool, json) {
  const p = json.params;
  const version  = hexWithoutPrefix(p[1]).padStart(8, "0").slice(0, 8);
  const prevhash = hexWithoutPrefix(p[2]).padStart(64, "0").slice(0, 64);
  const merkle   = hexWithoutPrefix(p[3]).padStart(64, "0").slice(0, 64);
  const reserved = hexWithoutPrefix(p[4]).padStart(64, "0").slice(0, 64);
  const ntime    = hexWithoutPrefix(p[5]).padStart(8, "0").slice(0, 8);
  const bits     = hexWithoutPrefix(p[6]).padStart(8, "0").slice(0, 8);

  const nonce1 = poolExtraNonce(pool);                 // pool nonce prefix (var length 2-4 bytes)
  const nonce  = (nonce1 + "0".repeat(64)).slice(0, 64); // 32-byte nonce = nonce1 || zero nonce2
  const blob   = version + prevhash + merkle + reserved + ntime + bits + nonce; // 280 hex = 140 bytes

  return {
    algo: fixedAlgoJobName(json, "equihash125_4"),
    blob: blob,
    job_id: p[0],
    ntime: ntime,
    target: pool.equihash_target,
    nonce1_len: nonce1.length / 2,   // bytes of the fixed pool prefix; the rest of the 32 B is nonce2
    noncebytes: 8,                   // the solver's incrementing search counter is 8 bytes
    nonceoffset: 108 + nonce1.length / 2,
  };
}

// Kaspa diff -> 256-bit BE share target. The kaspa-stratum-bridge DiffToTarget = maxTarget/diff where
// maxTarget = 0xFFFF...FF (28 bytes = 224 one-bits, i.e. 2^224-1). The native kheavyhash compares the
// 32-byte output little-endian against this big-endian boundary, so we pad to a 64-hex BE string.
function kaspaDiffToTarget(diff) {
  const MAX_TARGET = (1n << 224n) - 1n;
  const d = Math.max(1, Number(diff) || 1);
  // diff can be fractional; scale to keep precision then divide (target = MAX_TARGET / diff).
  const scale = 1n << 32n;
  const dScaled = BigInt(Math.round(d * Number(scale)));
  const target = dScaled > 0n ? (MAX_TARGET * scale) / dScaled : MAX_TARGET;
  return target.toString(16).padStart(64, "0").slice(-64);
}

// Build the Kaspa (kheavyhash) 80-byte header job from a mining.notify. Header layout (LE) =
//   pre_pow_hash(32) || timestamp(8) || zero(32) || nonce(8)   [native: matrix+keccak from [0..31]].
// The 4 pre-pow uint64 words go in LITTLE-endian at offsets 0,8,16,24; the timestamp LE at 32. The
// 8-byte search nonce at offset 72 is seeded so the pool's extranonce occupies its HIGH bytes (the
// pool parses the submitted nonce big-endian with the extranonce as the leading bytes). The native
// search counter advances the LOW bytes (nonce2); nicehash_mask fixes the extranonce high bytes.
function kaspaNotifyJob(pool, json) {
  const p = json.params;
  const words = json.__kaspa_words || p[1].map((v) => BigInt(v)); // BigInt-safe from raw line
  const timestamp = json.__kaspa_timestamp !== undefined ? BigInt(json.__kaspa_timestamp) : BigInt(p[2]);

  let blob = "";
  for (let i = 0; i < 4; ++i) blob += le8Hex(BigInt(words[i]));
  blob += le8Hex(timestamp);          // timestamp word (offset 32)
  blob += "00".repeat(32);            // zero padding (offsets 40..71)

  // Extranonce is the leading (high) bytes of the 8-byte nonce. Seed job.nonce with it in the high
  // bytes; the native writes the nonce LE at offset 72, so the high bytes land at the top of the field.
  const xn = poolExtraNonce(pool);                       // 0..3 byte hex (e.g. "56e0")
  const xnBytes = Math.min(xn.length / 2, 8);
  const nonceSeedHex = (xn + "0".repeat(16)).slice(0, 16); // 8-byte uint64 hex, xn in the high bytes
  blob += "0000000000000000";         // nonce placeholder at offset 72 (native re-embeds the seed)

  return {
    algo: fixedAlgoJobName(json, "kheavyhash"),
    blob: blob,                        // 160 hex = 80 bytes
    job_id: String(p[0]),
    target: pool.kaspa_target || kaspaDiffToTarget(pool.kaspa_difficulty || 1),
    difficulty: pool.kaspa_difficulty || 1,
    noncebytes: 8,
    nonceoffset: 72,
    nonce: nonceSeedHex,
    nicehash_mask: ("ff".repeat(xnBytes) + "00".repeat(8 - xnBytes)),
  };
}

// 8-byte little-endian hex of a uint64 (BigInt).
function le8Hex(value) {
  let v = BigInt(value) & ((1n << 64n) - 1n);
  let out = "";
  for (let i = 0; i < 8; ++i) { out += (v & 0xFFn).toString(16).padStart(2, "0"); v >>= 8n; }
  return out;
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
  // eslint-disable-next-line eqeqeq -- pool_id is "" | number; loose == is intentional coercion
  return pool_id == global.opt.pool_ids.donate || pool_id == active_pool || !alivePoolJob(pool_id);
}

function nextNonDonatePool(pool_id) {
  const next_pool = pool_id + 1;
  if (next_pool < Object.keys(global.opt.pools).length) return next_pool;
  // wrapped back to the first pool; skip it if it is the donate pool
  return global.opt.pool_ids.donate === 0 ? 1 : 0;
}

// switch active pool to the next available pool (except donate pool)
// preferring pool with already alive socket if any
module.exports.switch_pool = function(pool_id, set_job) {
  pool_close_wait(pool_id);

  const active_pool  = global.opt.pool_ids.active;
  const donate_pool  = global.opt.pool_ids.donate;

  // do not care about not active pool
  if (pool_id !== active_pool) return;

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

// Equihash mining.set_target carries a verbatim 64-hex BE 256-bit share target; store it as-is
// (left zero-padded to 64), like Iron Fish -- NOT left-justified the way ravenTarget treats its target.
function handleEquihashSetTarget(pool_id, json) {
  global.opt.pools[pool_id].equihash_target = hexWithoutPrefix(json.params[0]).padStart(64, "0");
}

// Iron Fish set_target carries a verbatim 64-hex BE 256-bit target; store it as-is (zero-padded on
// the left to 64 hex), unlike ravenTarget which left-justifies its share target.
function handleIronfishSetTarget(pool_id, json) {
  global.opt.pools[pool_id].ironfish_target = hexWithoutPrefix(json.body.target).padStart(64, "0");
}

function handleSetDifficulty(pool_id, json) {
  const pool = global.opt.pools[pool_id];
  pool.eth_difficulty = json.params[0];
  // var-diff pearl pools may push a standalone set_difficulty; stash it so the next pearl job picks
  // it up if the notify itself omits a diff field (otherwise jobTarget would fall back to MAX).
  if (poolProtocol(pool) === "pearl") pool.pearl_difficulty = json.params[0];
  // Kaspa pushes mining.set_difficulty [diff] (a float). Stash it and precompute the BE share target;
  // the next mining.notify (which carries no target) picks it up via kaspaNotifyJob.
  if (poolProtocol(pool) === "kaspa") {
    pool.kaspa_difficulty = json.params[0];
    pool.kaspa_target = kaspaDiffToTarget(json.params[0]);
  }
}

function nonceAt32Job(pool, job) {
  return {
    ...job,
    blob: nonceAt32Blob(job.header_hash, pool),
    nonce: poolNonce(pool),
    nicehash_mask: poolNonceMask(pool),
    noncebytes: 8,
    nonceoffset: 32,
  };
}

function fixedAlgoJobName(json, fallback) {
  return json.algo || (global.opt.job && global.opt.job.algo) || fallback;
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
    pool.submit_mode = "raven";
    return nonceAt32Job(pool, {
      // raven dialect is shared by kawpow/firopow/evrprogpow; resolve the actual algo from the job,
      // the configured global job, or the pool pass (falling back to kawpow) so firopow/evrprogpow
      // pools select the right seal/epoch instead of always hashing kawpow.
      algo: fixedAlgoJobName(json, algoFromPass(pool) || "kawpow"),
      header_hash: hexWithoutPrefix(json.params[1]),
      seed_hash: hexWithoutPrefix(json.params[2]),
      target: ravenTarget(pool, json.params[3]),
      job_id: json.params[0],
      height: json.params[5],
    });
  }
  if (poolProtocol(pool) === "eth" && isEthJobNotification(json)) {
    if (!pool.logged_in) return null;
    pool.submit_mode = "eth";
    return nonceAt32Job(pool, {
      algo: fixedAlgoJobName(json, "etchash"),
      header_hash: hexWithoutPrefix(json.params[2]),
      seed_hash: hexWithoutPrefix(json.params[1]),
      target: ethTarget(pool),
      job_id: json.params[0],
    });
  }
  if (usesEthProxy(pool) && isEthProxyWork(json)) {
    if (!pool.logged_in) return null;
    const headerHash = hexWithoutPrefix(json.result[0]);
    pool.submit_mode = "ethproxy";
    return nonceAt32Job(pool, {
      algo: fixedAlgoJobName(json, "etchash"),
      header_hash: headerHash,
      seed_hash: hexWithoutPrefix(json.result[1]),
      target: hexWithoutPrefix(json.result[2]).padStart(64, "0"),
      job_id: headerHash, // ethproxy has no job_id field; the header hash uniquely identifies the job
      height: parseHexHeight(json.result[3]),
    });
  }
  if (poolProtocol(pool) === "erg" && isErgJobNotification(json)) {
    if (!pool.logged_in) return null;
    pool.submit_mode = "erg";
    const job = nonceAt32Job(pool, {
      algo: fixedAlgoJobName(json, "autolykos2"),
      header_hash: hexWithoutPrefix(json.params[2]),
      target: ergTarget(json.params[6]),
      job_id: json.params[0],
      height: json.params[1],
      ntime: hexWithoutPrefix(json.params[7]),
    });
    rememberErgSubmitJob(pool, job);
    return job;
  }
  if (poolProtocol(pool) === "pearl" && isPearlJobNotification(json)) {
    if (!pool.logged_in) return null;
    pool.submit_mode = "pearl";
    const pp = json.params;
    return {
      algo: fixedAlgoJobName(json, "pearl"),
      blob: hexWithoutPrefix(pp.header),   // the 76-byte incomplete header (input for the kernel)
      job_id: pp.job_id,
      height: pp.height || 0,
      difficulty: pp.difficulty || pp.diff || pearlDiffFromJobId(pp.job_id) || pool.pearl_difficulty, // LuckyPool names it "diff"; var-diff may send it via set_difficulty
      // HeroMiners-style pools: precompute the verifier's jackpot bound from the base target field.
      // pearlpool-style: leave unset so jobTarget falls back to 2^256/diff.
      target: pearlUsesSubscribe(pool) ? pearlNbitsBound(pp.target) : undefined,
    };
  }
  if (poolProtocol(pool) === "equihash" && isEquihashJobNotification(json)) {
    if (!pool.logged_in) return null;
    pool.submit_mode = "equihash";
    return equihashNotifyJob(pool, json);
  }
  if (poolProtocol(pool) === "ironfish" && isIronfishJobNotification(json)) {
    if (!pool.logged_in) return null;
    pool.submit_mode = "ironfish";
    const body = json.body;
    return {
      algo: fixedAlgoJobName(json, "fishhash"),
      blob: hexWithoutPrefix(body.header), // the 180-byte block header (first 8 bytes = randomness)
      job_id: body.miningRequestId,
      noncebytes: 8,
      nonceoffset: 0,
      target: pool.ironfish_target,
      xn: pool.ironfish_xn || "",
    };
  }
  if (poolProtocol(pool) === "kaspa" && isKaspaJobNotification(json)) {
    if (!pool.logged_in) return null;
    pool.submit_mode = "kaspa";
    return kaspaNotifyJob(pool, json);
  }
  if (poolProtocol(pool) === "beam" && isBeamJobNotification(json)) {
    if (!pool.logged_in) return null;
    pool.submit_mode = "beam";
    if (typeof json.difficulty === "number") pool.beam_difficulty = json.difficulty;
    const packed = typeof json.difficulty === "number" ? json.difficulty : (pool.beam_difficulty || 0);
    return {
      algo:        "beamhash3",
      header_hash: hexWithoutPrefix(json.input),   // 64hex = 32-byte prework (goes at blob offset 0)
      job_id:      String(json.id),
      difficulty:  packed,                          // raw packed int32, for reporting
      target:      beamPackedTarget(packed),        // native re-derives the packed int from the target
    };
  }
  if (isLoginJob(json)) {
    pool.logged_in = true;
    pool.submit_mode = null;
    if ("id" in json.result) pool.worker_id = json.result.id;
    rememberPoolExtraNonceHex(pool_id, json.result.extra_nonce);
    applyLoginExtensions(pool_id, json.result.extensions);
    return loginJobWithResultMetadata(json.result);
  }
  return null;
}

function jobTargetWork(job) {
  // BeamHash III carries a PACKED 32-bit network difficulty (not a 256-bit boundary); the share rate is
  // in solutions, so report the packed difficulty itself rather than decoding job.target as a boundary.
  if (job.algo === "beamhash3") return job.difficulty ? BigInt(job.difficulty) : null;
  if (!job.target) return null;
  if (job.algo === "kawpow" || job.algo === "firopow" || job.algo === "evrprogpow" || job.algo === "meowpow")
    return h.kawpowTarget2diff(job.target);
  // pearl: report the share target in GEMM MACs to match the MAC/s hashrate (so time-per-share =
  // target/hashrate). work/share = (tiles/share = 2^256/bound) * (MACs/tile = 16*16*k_eff).
  if (job.algo === "pearl") return h.target256ToWork(job.target) * BigInt(16 * 16 * pearlKEff());
  // etchash/autolykos2/fishhash carry a full 256-bit target too, but their hashrate is in hashes -> H/share.
  if (job.algo === "etchash" || job.algo === "autolykos2" || job.algo === "fishhash" ||
      job.algo === "equihash125_4" || job.algo === "kheavyhash" ||
      job.algo === "karlsenhashv2" || job.algo === "pyrinhashv2")
    return h.target256ToWork(job.target);
  return h.target2diff(job.target);
}

function jobTargetDescription(job) {
  const work = jobTargetWork(job);
  return work !== null ? h.formatHashCount(work) + "/share target" : job.difficulty + " diff";
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
  if (job.target) jobTargetWork(job); // throws early on a malformed target before we store the job

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
  if (is_err || json.result === false) return loginFailed(pool_id, err_msg || ": Login rejected");
  if (is_ok) return loginSucceeded(pool_id);
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

function handleAuthorizeResponse(pool_id, is_err, is_ok, err_msg, json) {
  global.opt.pools[pool_id].pending_authorize = false;
  if (!is_err && json.result === true) return loginSucceeded(pool_id);
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
  if (!is_err && is_ok) rememberPoolResponseMetadata(pool_id, json.result);
  return result;
}

function rememberPoolResponseMetadata(pool_id, result) {
  if (!isObject(result)) return;

  const pool = global.opt.pools[pool_id];
  if ("id" in result) pool.worker_id = result.id;
  rememberPoolExtraNonceHex(pool_id, result.extra_nonce);
  rememberPoolProtocol(pool_id, result);
  applyLoginExtensions(pool_id, result.extensions);
}

function ignorePoolResponse() {}

function poolResponseHandler(pool_id, id) {
  const pool = global.opt.pools[pool_id];
  if (pearlUsesSubscribe(pool)) {
    if (id === 1) return ignorePoolResponse;           // subscribe ack/err (authorize already sent)
    if (id === 2) return pool.pending_authorize ? handleAuthorizeResponse : ignorePoolResponse;
  } else if (usesMiningSubscribe(pool)) {
    if (id === 1) return handleSubscribeResponse; // mining.subscribe response
    if (id === 2) return pool.pending_authorize ? handleAuthorizeResponse : ignorePoolResponse;
  } else if (usesEthProxy(pool)) {
    if (id === 1) return handleLoginResponse; // eth_submitLogin response
    if (id === 2) return ignorePoolResponse; // legacy keepalive response
  } else {
    if (id === 1) return handleLoginResponse; // login response
    if (id === 2) return ignorePoolResponse; // keepalive response
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
  if (poolProtocol(global.opt.pools[pool_id]) !== "ironfish") return false;
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
      if (typeof json.nonceprefix === "string") pool.beam_nonceprefix = hexWithoutPrefix(json.nonceprefix);
      if (typeof json.forkheight === "number") pool.beam_forkheight = json.forkheight;
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
    return handleBeamResult(pool_id, json);
  if (handleIronfishMessage(pool_id, json, set_job)) return;
  if (isRavenSetTargetNotification(json)) {
    const protocol = poolProtocol(global.opt.pools[pool_id]);
    if (protocol === "eth") return handleEthSetTarget(pool_id, json);
    if (protocol === "equihash") return handleEquihashSetTarget(pool_id, json);
    return handleRavenSetTarget(pool_id, json);
  }
  if (isSetDifficultyNotification(json)) return handleSetDifficulty(pool_id, json);
  if (isSetExtranonceNotification(json)) {
    rememberPoolExtraNonceHex(pool_id, json.params[0]);
    if (Number.isInteger(Number(json.params[1])))
      global.opt.pools[pool_id].extra_nonce2_size = Number(json.params[1]);
    return;
  }
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
    const poolAlgo = poolAlgoName(algo);
    algos.push(poolAlgo);
    algo_perfs[poolAlgo] = normalizedPoolAlgoPerf(algo, global.opt.algo_params[algo].perf);
  }
  return {
    login: pool.login, pass: pool.pass, agent: o.agent_str,
    algo: algos, "algo-perf": algo_perfs
  };
}

function poolAlgoName(algo) {
  // Historical mom KawPow perf values were already raw H/s.
  return algo === "kawpow" ? "kawpow1" : algo;
}

function normalizedPoolAlgoPerf(algo, perf) {
  // Cycle algorithms are reported to the pool in solutions per second.
  if (algo === "c29") return perf / 42;
  return perf;
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
      !Array.isArray(json.params[1])) return;
  // params: [ "jobId", [w0,w1,w2,w3], timestamp ]. Capture the bracketed word list + the timestamp.
  const m = message.match(/"params"\s*:\s*\[\s*"[^"]*"\s*,\s*\[([^\]]*)\]\s*,\s*(\d+)/);
  if (!m) return;
  const words = m[1].split(",").map((s) => s.trim()).filter((s) => /^\d+$/.test(s));
  if (words.length < 4) return;
  // Store as exact decimal strings (NOT BigInt) so the debug logger's JSON.stringify(json) still works.
  json.__kaspa_words = words.slice(0, 4);
  json.__kaspa_timestamp = m[2];
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
    if (message.trim() === "") continue;
    const json = parsePoolLine(pool_id, message);
    if (json && processPoolJson(pool_id, json, set_job, pool_err)) return true;
  }
  return false;
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
      "No initial job from " + pool_str(pool_id) + " pool"
    ));
  }, global.opt.pool_time.first_job_wait * 1000);
}

function handlePoolConnect(pool_id, socket, pool) {
  if (!isCurrentPoolSocket(pool_id, socket)) return;
  pool_log1(pool_id, "Connected to the pool");
  if (usesIronfish(pool)) {
    // Iron Fish custom OBJECT stratum: a single mining.subscribe push carries the wallet+worker
    // (publicAddress) and the agent; the pool replies with mining.subscribed (handled by method).
    // No separate authorize. extend:["mining.submitted"] requests the submit-result push.
    return module.exports.pool_write(pool_id, {
      id: 1, method: "mining.subscribe",
      body: { version: 3, agent: o.agent_str, publicAddress: pool.login, extend: ["mining.submitted"] }
    });
  }
  if (pearlUsesSubscribe(pool)) {
    // Pearl subscribe dialect: send subscribe AND authorize back-to-back. mining.subscribe is just a
    // handshake nicety -- HeroMiners acks it (result:true), LuckyPool rejects it ("method not
    // supported") and drops the connection if no authorize follows promptly. So don't wait on the
    // subscribe reply; authorize immediately. authorize takes OBJECT params {wallet,worker,pass}.
    pool.pending_authorize = true;
    module.exports.pool_write(pool_id, { jsonrpc: "2.0", id: 1, method: "mining.subscribe", params: [o.agent_str] });
    return module.exports.pool_write(pool_id, {
      jsonrpc: "2.0", id: 2, method: "mining.authorize",
      params: { wallet: pool.login, worker: pool.worker || "mom", pass: pool.pass }
    });
  }
  if (poolProtocol(pool) === "beam") {
    // Beam JSON-RPC: a single `login` with the wallet/api_key (the pool replies with a `result`
    // message carrying code:0 and the nonceprefix). No mining.subscribe handshake.
    return module.exports.pool_write(pool_id, {
      jsonrpc: "2.0", id: "login", method: "login", api_key: pool.login
    });
  }
  const request = usesMiningSubscribe(pool) ?
    { jsonrpc: "2.0", id: 1, method: "mining.subscribe", params: [o.agent_str] } :
    usesEthProxy(pool) ?
      { jsonrpc: "2.0", id: 1, method: "eth_submitLogin", params: [pool.login, pool.pass] } :
      { jsonrpc: "2.0", id: 1, method: "login", params: poolLoginParams(pool) };
  return module.exports.pool_write(pool_id, request);
}

function splitPoolMessages(pool_data_buff) {
  const messages = pool_data_buff.split("\n");
  const incomplete_line = pool_data_buff.slice(-1) === "\n" ? "" : messages.pop();
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
  return pool_data_buff.indexOf("\n") !== -1;
}

function poolDataHandler(pool_id, socket, set_job, pool_err) {
  let pool_data_buff = "";
  return function(data) {
    const next_buff = readPoolData(pool_id, socket, pool_data_buff, data, pool_err);
    if (next_buff === null) return;
    pool_data_buff = next_buff;
    if (!hasCompletePoolLine(pool_data_buff)) return;
    const { messages, incomplete_line } = splitPoolMessages(pool_data_buff);
    if (handlePoolLines(pool_id, messages, set_job, pool_err)) return;
    pool_data_buff = incomplete_line;
  };
}

function connect_pool(pool_id, set_job) {
  const pool = global.opt.pools[pool_id];

  // do not connect to already connected pools
  if (pool.socket) return;

  pool_log(pool_id, "Connecting to " + poolTypeStr(pool_id) + " " + pool_str(pool_id) + " pool");
  pool.last_connect_time = Date.now();
  const socket = pool.socket = connectSocket(pool);
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

module.exports.connect_pool_throttle = function(pool_id, set_job) {
  const pool = global.opt.pools[pool_id];
  const wait_time = global.opt.pool_time.connect_throttle * 1000 -
                    (Date.now() - pool.last_connect_time);
  if (wait_time <= 0) return connect_pool(pool_id, set_job);
  pool_log(pool_id, "Waiting " + Math.floor(wait_time / 1000) + "s to connect to the pool");
  return setTimeout(connect_pool, wait_time, pool_id, set_job);
};
