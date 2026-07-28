"use strict";

module.exports = ({
  h, normalizeAlgoName, poolProtocol, usesEthProxy, pearlhashUsesSubscribe,
  pearlhashDiffFromJobId, pearlhashNbitsBound, beamPackedTarget, pool_close_wait,
  pool_log, pool_str, algoFromPass, applyLoginExtensions, connectPoolThrottle,
}) => {

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

  // ZelHash's ZIP-301 stratum mining.notify carries 8 array fields:
  // [job_id, version(LE 8hex), prevhash(64), merkleroot(64), reserved(64), time(8), bits(8), clean].
  function isZelHashJobNotification(json) {
    return json.method === "mining.notify" && Array.isArray(json.params) && json.params.length >= 8;
  }

  // Beam JSON-RPC `job`: TOP-LEVEL {input(64hex), difficulty(int32), id, method:"job"} (no params array).
  function isBeamJobNotification(json) {
    return json.method === "job" && typeof json.input === "string";
  }

  // Kaspa-family mining.notify carries 3 params:
  // [jobId(string), [u0,u1,u2,u3] (4 uint64 LE pre-pow words), timestamp(ms uint64)]. The 4 words are the
  // 32-byte BLAKE2b pre-pow hash (TS=0,Nonce=0) split into little-endian uint64s. They overflow JS
  // Number, so parsePoolLine re-extracts them from the raw line into json.__kaspa_words (decimal strings).
  function isKaspaJobNotification(json) {
    return json.method === "mining.notify" && Array.isArray(json.params) &&
         json.params.length >= 3 && Array.isArray(json.params[1]) && json.params[1].length >= 4;
  }

  // pearlpool.cloud pushes mining.notify with OBJECT params {job_id, header, target, difficulty, height, mode}.
  function isPearlHashJobNotification(json) {
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
    if (!Array.isArray(result)) {return [];}
    return Array.isArray(result[0]) || result[0] == null ? [result[1]] : result;
  }

  function subscribeExtraNonce2Size(result) {
    if (!Array.isArray(result) || !(Array.isArray(result[0]) || result[0] == null)) {return null;}
    const size = Number(result[2]);
    return Number.isInteger(size) && size >= 0 && size <= 8 ? size : null;
  }

  function rememberPoolExtraNonceHex(pool_id, value) {
    const extra_nonce = validExtraNonce(value);
    if (extra_nonce) {global.opt.pools[pool_id].extra_nonce = extra_nonce;}
  }

  function rememberSubscribeExtraNonce(pool_id, result) {
    rememberPoolExtraNonceHex(pool_id, subscribeExtraNonceCandidates(result).find(validExtraNonce));
    const extra_nonce2_size = subscribeExtraNonce2Size(result);
    if (extra_nonce2_size !== null) {global.opt.pools[pool_id].extra_nonce2_size = extra_nonce2_size;}
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
    if (!hex || /[^0-9a-f]/i.test(hex)) {return 0;}
    return Number.parseInt(hex, 16);
  }

  function ergTarget(bound) {
    return h.decimalTargetToHex(bound);
  }

  function rememberErgSubmitJob(pool, job) {
    if (!pool.erg_submit_jobs) {pool.erg_submit_jobs = {};}
    pool.erg_submit_jobs[job.job_id] = {
      extra_nonce: poolExtraNonce(pool),
      extra_nonce2_size: pool.extra_nonce2_size,
      ntime: job.ntime || "",
    };

    const jobIds = Object.keys(pool.erg_submit_jobs);
    while (jobIds.length > 16) {delete pool.erg_submit_jobs[jobIds.shift()];}
  }

  // Build the ZelHash (Equihash 125,4, Flux/ZIP-301) job from a mining.notify. The 8 notify fields go straight
  // into the 140-byte Zcash header at the fixed offsets (prev/merkle/reserved already in header byte
  // order -- concat directly, NO reversal); the 32-byte nonce at offset 108 starts as nonce1 (the
  // subscribe extranonce prefix) followed by a zero nonce2 region the solver fills. The solver's 8-byte
  // search counter is written at nonceoffset = 108 + nonce1_len so it lands inside nonce2, never
  // clobbering the pool's fixed nonce1 prefix.
  function zelhashNotifyJob(pool, json) {
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
      algo: fixedAlgoJobName(json, "zelhash"),
      blob: blob,
      job_id: p[0],
      ntime: ntime,
      target: pool.zelhash_target,
      nonce1_len: nonce1.length / 2,   // bytes of the fixed pool prefix; the rest of the 32 B is nonce2
      noncebytes: 8,                   // the solver's incrementing search counter is 8 bytes
      nonceoffset: 108 + nonce1.length / 2,
    };
  }

  // Kaspa diff -> 256-bit BE share target. The kaspa-stratum-bridge DiffToTarget = maxTarget/diff where
  // maxTarget = 0xFFFF...FF (28 bytes = 224 one-bits, i.e. 2^224-1). KarlsenHashV2 compares its
  // 32-byte output little-endian against this big-endian boundary, so pad to a 64-hex BE string.
  function kaspaDiffToTarget(diff) {
    const MAX_TARGET = (1n << 224n) - 1n;
    const d = Math.max(1, Number(diff) || 1);
    // diff can be fractional; scale to keep precision then divide (target = MAX_TARGET / diff).
    const scale = 1n << 32n;
    const dScaled = BigInt(Math.round(d * Number(scale)));
    const target = dScaled > 0n ? (MAX_TARGET * scale) / dScaled : MAX_TARGET;
    return target.toString(16).padStart(64, "0").slice(-64);
  }

  // Build the KarlsenHashV2 80-byte header job from a Kaspa-family mining.notify. Header layout (LE) =
  //   pre_pow_hash(32) || timestamp(8) || zero(32) || nonce(8).
  // The 4 pre-pow uint64 words go in LITTLE-endian at offsets 0,8,16,24; the timestamp LE at 32. The
  // 8-byte search nonce at offset 72 is seeded so the pool's extranonce occupies its HIGH bytes (the
  // pool parses the submitted nonce big-endian with the extranonce as the leading bytes). The native
  // search counter advances the LOW bytes (nonce2); nicehash_mask fixes the extranonce high bytes.
  function kaspaNotifyJob(pool, json) {
    const p = json.params;
    const words = json.__kaspa_words || p[1].map((v) => BigInt(v)); // BigInt-safe from raw line
    const timestamp = json.__kaspa_timestamp !== undefined ? BigInt(json.__kaspa_timestamp) : BigInt(p[2]);

    let blob = "";
    for (let i = 0; i < 4; ++i) {blob += le8Hex(BigInt(words[i]));}
    blob += le8Hex(timestamp);          // timestamp word (offset 32)
    blob += "00".repeat(32);            // zero padding (offsets 40..71)

    // Extranonce is the leading (high) bytes of the 8-byte nonce. Seed job.nonce with it in the high
    // bytes; the native writes the nonce LE at offset 72, so the high bytes land at the top of the field.
    const xn = poolExtraNonce(pool);                       // 0..3 byte hex (e.g. "56e0")
    const xnBytes = Math.min(xn.length / 2, 8);
    const nonceSeedHex = (xn + "0".repeat(16)).slice(0, 16); // 8-byte uint64 hex, xn in the high bytes
    blob += "0000000000000000";         // nonce placeholder at offset 72 (native re-embeds the seed)

    return {
      algo: fixedAlgoJobName(json, "karlsenhashv2"),
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
      if (!(key in job) && key in result) {job[key] = result[key];}
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
    if (!alivePoolJob(primary_pool)) {return null;}
    return activateAlivePool(primary_pool, set_job, "the primary");
  }

  function reactivateBackupPool(active_pool, set_job) {
    for (const pool_id of Object.keys(global.opt.pools)) {
    // === will not work here since here we are comparing strings and integers
      if (shouldSkipBackupPool(pool_id, active_pool)) {continue;}
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
    if (next_pool < Object.keys(global.opt.pools).length) {return next_pool;}
    // wrapped back to the first pool; skip it if it is the donate pool
    return global.opt.pool_ids.donate === 0 ? 1 : 0;
  }

  // switch active pool to the next available pool (except donate pool)
  // preferring pool with already alive socket if any
  function switchPool(pool_id, set_job) {
    pool_close_wait(pool_id);

    const active_pool  = global.opt.pool_ids.active;
    const donate_pool  = global.opt.pool_ids.donate;

    // do not care about not active pool
    if (pool_id !== active_pool) {return;}

    // select already alive pool if possible, except donate pool (starting from primary pool)
    const alive_job = reactivateAlivePoolIfAny(active_pool, set_job);
    if (alive_job) {return alive_job;}

    // do not continue to mine on donate pool if all other pools are dead
    if (global.opt.pool_ids.active === donate_pool) {h.messageWorkers({type: "pause"});}

    // select the next available pool except donate pool
    pool_id = nextNonDonatePool(pool_id);
    global.opt.pool_ids.active = pool_id;
    return connectPoolThrottle(pool_id, set_job);
  }

  function reactivateAlivePoolIfAny(active_pool, set_job) {
    return reactivatePrimaryPool(set_job) || reactivateBackupPool(active_pool, set_job);
  }

  function handleRavenSetTarget(pool_id, json) {
    global.opt.pools[pool_id].raven_target = hexWithoutPrefix(json.params[0]);
  }

  function handleEthSetTarget(pool_id, json) {
    global.opt.pools[pool_id].eth_target = hexWithoutPrefix(json.params[0]);
  }

  // ZelHash mining.set_target carries a verbatim 64-hex BE 256-bit share target; store it as-is
  // (left zero-padded to 64), like Iron Fish -- NOT left-justified the way ravenTarget treats its target.
  function handleZelHashSetTarget(pool_id, json) {
    global.opt.pools[pool_id].zelhash_target = hexWithoutPrefix(json.params[0]).padStart(64, "0");
  }

  // Iron Fish set_target carries a verbatim 64-hex BE 256-bit target; store it as-is (zero-padded on
  // the left to 64 hex), unlike ravenTarget which left-justifies its share target.
  function handleIronfishSetTarget(pool_id, json) {
    global.opt.pools[pool_id].ironfish_target = hexWithoutPrefix(json.body.target).padStart(64, "0");
  }

  function handleSetDifficulty(pool_id, json) {
    const pool = global.opt.pools[pool_id];
    pool.eth_difficulty = json.params[0];
    // Var-diff PearlHash pools may push a standalone set_difficulty; stash it so the next job picks
    // it up if the notify itself omits a diff field (otherwise jobTarget would fall back to MAX).
    if (poolProtocol(pool) === "pearlhash") {pool.pearlhash_difficulty = json.params[0];}
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
    return normalizeAlgoName(json.algo || (global.opt.job && global.opt.job.algo) || fallback);
  }

  function jobFromPoolMessage(pool_id, json) {
    const pool = global.opt.pools[pool_id];
    if (isJobNotification(json)) {
      if (!pool.logged_in) {return null;}
      pool.submit_mode = null;
      return json.params;
    }
    if (poolProtocol(pool) === "raven" && isRavenJobNotification(json)) {
      if (!pool.logged_in) {return null;}
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
      if (!pool.logged_in) {return null;}
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
      if (!pool.logged_in) {return null;}
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
      if (!pool.logged_in) {return null;}
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
    if (poolProtocol(pool) === "pearlhash" && isPearlHashJobNotification(json)) {
      if (!pool.logged_in) {return null;}
      pool.submit_mode = "pearlhash";
      const pp = json.params;
      return {
        algo: fixedAlgoJobName(json, "pearlhash"),
        blob: hexWithoutPrefix(pp.header),   // the 76-byte incomplete header (input for the kernel)
        job_id: pp.job_id,
        height: pp.height || 0,
        difficulty: pp.difficulty || pp.diff || pearlhashDiffFromJobId(pp.job_id) || pool.pearlhash_difficulty, // LuckyPool names it "diff"; var-diff may send it via set_difficulty
        // HeroMiners-style pools: precompute the verifier's jackpot bound from the base target field.
        // pearlpool-style: leave unset so jobTarget falls back to 2^256/diff.
        target: pearlhashUsesSubscribe(pool) ? pearlhashNbitsBound(pp.target) : undefined,
      };
    }
    if (poolProtocol(pool) === "zelhash" && isZelHashJobNotification(json)) {
      if (!pool.logged_in) {return null;}
      pool.submit_mode = "zelhash";
      return zelhashNotifyJob(pool, json);
    }
    if (poolProtocol(pool) === "ironfish" && isIronfishJobNotification(json)) {
      if (!pool.logged_in) {return null;}
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
      if (!pool.logged_in) {return null;}
      pool.submit_mode = "kaspa";
      return kaspaNotifyJob(pool, json);
    }
    if (poolProtocol(pool) === "beam" && isBeamJobNotification(json)) {
      if (!pool.logged_in) {return null;}
      pool.submit_mode = "beam";
      if (typeof json.difficulty === "number") {pool.beam_difficulty = json.difficulty;}
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
      if ("id" in json.result) {pool.worker_id = json.result.id;}
      rememberPoolExtraNonceHex(pool_id, json.result.extra_nonce);
      applyLoginExtensions(pool_id, json.result.extensions);
      return loginJobWithResultMetadata(json.result);
    }
    return null;
  }

  return {
    isObject, isIronfishSetTargetNotification, isRavenSetTargetNotification,
    isSetDifficultyNotification, isSetExtranonceNotification, hexWithoutPrefix,
    validExtraNonce, rememberPoolExtraNonceHex, rememberSubscribeExtraNonce,
    switchPool, handleRavenSetTarget, handleEthSetTarget, handleZelHashSetTarget,
    handleIronfishSetTarget, handleSetDifficulty, jobFromPoolMessage,
  };
};
