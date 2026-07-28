"use strict";

module.exports = ({
  h, opt, process, compilerPolicy, gpuTuning, orDefault, nonceOffsetOr, firstTruthyOr,
  hexWithoutPrefix, normalizeAlgoName, messageHandler,
  getComputeCore, getLastJob, setLastJob,
}) => {

  function set_algo_msr(algo) {
    const compute_core = getComputeCore();
    if (!compute_core || !Object.keys(opt.default_msrs).length) {return;}
    const default_msr = h.pack_msr(opt.default_msrs);
    default_msr.algo = algo;
    compute_core.emit_to("write_msr", default_msr);
  }

  function jobDev(algo) {
    const algo_param = opt.algo_params[algo];
    return algo_param && algo_param.dev ? algo_param.dev : opt.job.dev;
  }

  function requestedJobBackend(algo) {
    const algoParam = opt.algo_params[algo];
    const defaultRequest = opt.job.backend_request || opt.job.backend;
    const configured = algoParam && algoParam.backend !== "auto"
      ? algoParam.backend
      : defaultRequest;
    return compilerPolicy.validateBackend(configured || "auto");
  }

  function jobBackend(algo) {
    const requested = requestedJobBackend(algo);
    if (requested !== "auto") {return requested;}
    const gpu = compilerPolicy.gpuFromEnv(process.env);
    const selected = gpu && compilerPolicy.selection(algo, gpu, process.platform);
    return selected ? selected.backend : "auto";
  }

  function mergeDeviceTuning(algo, heuristic, named, entry) {
    const tuning = {...heuristic, ...named, ...entry};
    // Beam's automatic workgroup depends on the selected memory layout. Discovery reports the
    // default layout's workgroup; when a user changes only the layout, leave workgroup unresolved
    // so the hashing worker derives the correct device-specific value for that layout.
    const requestedBeamLayout = entry.layout ?? named.layout;
    if (algo === "beamhash3" && requestedBeamLayout &&
      requestedBeamLayout !== "auto" &&
      entry.workgroup === undefined && named.workgroup === undefined) {
      delete tuning.workgroup;
    }
    return tuning;
  }

  function resolvedDeviceList(algo, configuredDev, heuristicDev, namedTuning = {}) {
    const configured = gpuTuning.parseDeviceList(configuredDev, algo);
    const heuristic = gpuTuning.parseDeviceList(heuristicDev, algo);
    const primaryField = gpuTuning.primaryTuningField(algo);
    const configuredCounts = new Map();
    for (const entry of configured) {
      configuredCounts.set(entry.device, (configuredCounts.get(entry.device) || 0) + 1);
    }
    const result = [];
    for (const entry of configured) {
      if (!entry.device.startsWith("gpu")) {
        result.push(entry);
        continue;
      }
      const matches = heuristic.filter((candidate) => candidate.device === entry.device);
      const explicitShape = entry.tuning[primaryField] !== undefined;
      if (!explicitShape && configuredCounts.get(entry.device) === 1 && matches.length > 1) {
        for (const candidate of matches) {
          result.push({
            ...entry,
            tuning: mergeDeviceTuning(algo, candidate.tuning, namedTuning, entry.tuning),
          });
        }
        continue;
      }
      const heuristicTuning = matches.length === 1 ? matches[0].tuning :
        matches.length > 1
          ? {intensity: matches.reduce((total, candidate) =>
            total + (candidate.tuning.intensity || 0), 0)}
          : {};
      result.push({
        ...entry,
        tuning: mergeDeviceTuning(algo, heuristicTuning, namedTuning, entry.tuning),
      });
    }
    return gpuTuning.formatDeviceList(result);
  }

  function configuredTuning(algo) {
    return (opt.algo_params[algo] && opt.algo_params[algo].tuning) || {};
  }

  function pearlhashShape() {
    const configured = opt.algo_params.pearlhash || {};
    const tuning = configured.tuning || {};
    const gpu = compilerPolicy.gpuFromEnv(process.env);
    const selected = gpu && compilerPolicy.selection("pearlhash", gpu, process.platform);
    const profile = selected && selected.pearlhashProfile;
    return {
      m: Number(tuning.m || (profile && profile.m) || 131072),
      n: Number(tuning.n || (profile && profile.n) || tuning.m ||
      (profile && profile.m) || 131072),
      k: Number(tuning.k || (profile && profile.k) || 4096),
      rank: Number(tuning.rank || (profile && profile.rank) || 256),
    };
  }

  function addPearlHashJobFields(job) {
    const shape = pearlhashShape();
    job.pearlhash_n = shape.n;
    job.pearlhash_k = shape.k;
    job.pearlhash_rank = shape.rank;
  }

  function baseJob(prev_job, algo, dev, pool_id) {
    const job = {
      algo:       algo,
      dev:        dev,
      seed_hex:   orDefault(prev_job.seed_hash, prev_job.seed_hex),
      target:     jobTarget(prev_job, algo),
      worker_id:  firstTruthyOr(opt.pools[pool_id].worker_id || opt.pools[pool_id].login,
        prev_job.id, prev_job.worker_id),
      job_id:     orDefault(prev_job.job_id, ""),
      header_hash: orDefault(prev_job.header_hash, ""),
      nonce:      orDefault(prev_job.nonce, 0),
      height:     orDefault(prev_job.height, 0),
      difficulty: prev_job.difficulty,
      thread_num: h.get_dev_threads(dev),
      pool_id:    pool_id,
      backend_request: requestedJobBackend(algo),
      backend:    jobBackend(algo),
    };
    if (algo === "pearlhash") {addPearlHashJobFields(job);}
    return job;
  }

  const nonceAt32Algos = new Set(["kawpow", "firopow", "evrprogpow", "meowpow", "etchash", "autolykos2", "fishhash"]);
  // KarlsenHashV2 uses the Kaspa 80-byte header / 8-byte nonce-at-72 layout.
  const kaspaHeaderAlgos = new Set(["karlsenhashv2"]);
  // Heights sampled from coin mainnets so benchmark DAG/table sizes match live pool jobs
  // (epoch-0 sizes overstate hashrate by ~7-10% on these algos): ETC 2026-06-04, RVN and ERG 2026-06-12.
  const benchHeightByAlgo = {
    etchash:    24689903,
    kawpow:     4407982,
    firopow:    600000,
    evrprogpow: 1800000,
    meowpow:    825000,
    autolykos2: 1806198,
  };
  const defaultBenchAlgos = new Set([
    "autolykos2",
    "c29",
    "cn/gpu",
    "etchash",
    "ghostrider",
    "kawpow",
    "panthera",
    // PearlHash (PRL) is benched by default even though it is not (yet) a MoneroOcean pool algo: the
    // GPU PoUW NoisyGEMM path is a headline number we want reported alongside the other GPU algos.
    "pearlhash",
    "rx/0",
    "rx/2",
    "rx/arq",
  ]);

  function isNonceAt32Algo(algo) {
    return nonceAt32Algos.has(algo);
  }

  function isKaspaHeaderAlgo(algo) {
    return kaspaHeaderAlgos.has(algo);
  }

  function isZelHashAlgo(algo) {
    return algo === "zelhash";
  }

  // A deterministic 140-byte Flux header for benching the Equihash 125,4 GPU solver (mainnet block
  // 400000). Each Wagner solve over this header finds 2 distinct proofs in ~2.2 s on a B580, so the
  // reported Sol/s is the solver's true throughput. The 32-byte nonce lives at offset 108.
  const ZELHASH_BENCH_BLOB =
    "04000000a8675c842f7a1342fadd00cd9b4e4909526b1c0ab5a747c5529b4deb13000000" +
  "ce7d6ea2452245925fc70c3a08a3c3dd2ca4beab7481f237a19751666bfd25c3" +
  "0fd282d94b1e1a7f2c57eb3fb9e2853d990753fa137e13c99bd43f220d4fce69" +
  "90e44f5dce28421d" +
  "600000160000000000000000000000000000000000000000000000009cfd1100";

  // BeamHash III M4 keystone-shaped benchmark blob: prework(32) || nonce(8) || extranonce(4).
  const BEAMHASH3_BENCH_BLOB =
    "fc40996a518c221384c9f2542ca811cd66c4ccddb001ef40b9f9ba059c20352e" +
  "0100000000000000" +
  "00000000";

  function jobTarget(prev_job, algo) {
    const explicitTarget = orDefault(prev_job.target, "");
    if (algo === "pearlhash" || isZelHashAlgo(algo)) {
    // HeroMiners-style pools precompute the verifier bound (pool.js pearlhashNbitsBound -> prev_job.target);
    // Flux set_target also delivers a 256-bit big-endian hex share target. When a pool does not send a
    // target, use the lenient floor(2^256 / difficulty) fallback.
      if (explicitTarget) {return hexWithoutPrefix(explicitTarget).padStart(64, "0");}
      return fullTargetFromDifficulty(prev_job.difficulty);
    }
    if (!isNonceAt32Algo(algo)) {return explicitTarget || h.diff2target(prev_job.difficulty);}
    // autolykos2 (erg) may deliver the target as a DECIMAL string. Every other nonce-at-32 algo
    // (kawpow/firopow/evrprogpow/etchash) delivers a 256-bit HEX boundary in mining.notify/set_target --
    // and that hex can legitimately be all-[0-9] digits (e.g. WoolyPooly's "0000000100..."), so the
    // digit-only "is decimal" heuristic must be applied ONLY to autolykos2, never to the hex-target algos.
    if (algo === "autolykos2" && /^\d+$/.test(explicitTarget)) {return h.decimalTargetToHex(explicitTarget);}
    if (explicitTarget && hexWithoutPrefix(explicitTarget).length > 16)
    {return hexWithoutPrefix(explicitTarget).padStart(64, "0");}
    return h.ethDiff2Target(prev_job.difficulty || (explicitTarget ? h.target2diff(explicitTarget) : 1));
  }

  function fullTargetFromDifficulty(difficulty) {
    const MAX = (1n << 256n) - 1n, diff = BigInt(difficulty || 1);
    return (diff > 0n ? MAX / diff : MAX).toString(16).padStart(64, "0");
  }

  function addC29JobFields(job, prev_job) {
    job.proofsize = orDefault(prev_job.proofsize, 42);
    if (prev_job.pre_pow) { // GRIN
      job.noncebytes  = orDefault(prev_job.noncebytes, 4);
      job.blob_hex    = prev_job.pre_pow + "00".repeat(job.noncebytes);
      job.nonceoffset = prev_job.pre_pow.length / 2;
    } else if (prev_job.blob) { // TARI C29
      job.noncebytes  = orDefault(prev_job.noncebytes, 8);
      job.blob_hex    = "00".repeat(job.noncebytes) + prev_job.blob;
      job.nonceoffset = 0;
    } else {
      job.noncebytes  = prev_job.noncebytes;
      job.blob_hex    = prev_job.blob_hex;
      job.nonceoffset = prev_job.nonceoffset;
    }
  }

  function addEthHashJobFields(job, prev_job) {
    job.noncebytes = orDefault(prev_job.noncebytes, 8);
    job.nonceoffset = nonceOffsetOr(prev_job, 32);

    const blob = orDefault(prev_job.blob, prev_job.blob_hex);
    job.blob_hex = blob && blob.length === 64 ? blob + "0000000000000000" : blob;
  }

  function addKaspaHeaderJobFields(job, prev_job) {
  // The header is pre_pow_hash(32) || timestamp_le(8) || zero pad(32) || nonce_le(8), with an
  // 8-byte LE nonce at offset 72. Tests/bench pass blob_hex directly; the live Kaspa dialect
  // (pool.js) builds the 80-byte header from the pool's pre_pow_hash + timestamp.
    addFixedNonceBlobFields(job, prev_job, 72);
  }

  function addZelHashJobFields(job, prev_job) {
  // Equihash 125,4 (Flux/ZIP-301): 140-byte Zcash header with a 32-byte nonce at offset 108. The pool
  // dialect (pool.js zelhashNotifyJob) builds the full 280-hex header and sets nonceoffset to
  // 108 + nonce1_len so the solver's 8-byte search counter advances inside nonce2 (after the pool's
  // fixed nonce1 prefix). Carry ntime + nonce1_len through for the mining.submit reconstruction.
    addFixedNonceBlobFields(job, prev_job, 108);
    job.ntime       = orDefault(prev_job.ntime, "");
    job.nonce1_len  = orDefault(prev_job.nonce1_len, 0);
  }

  function addFixedNonceBlobFields(job, prev_job, defaultOffset) {
    job.noncebytes  = orDefault(prev_job.noncebytes, 8);
    job.nonceoffset = nonceOffsetOr(prev_job, defaultOffset);
    job.blob_hex    = orDefault(prev_job.blob, prev_job.blob_hex);
  }

  // BeamHash III blob = prework(32) || nonce(8) || extranonce(4) = 44 bytes; the 8-byte Beam nonce sits
  // at offset 32. The pool's nonceprefix (0-6 bytes) must occupy the LEADING physical bytes of that nonce
  // field. The native search counter is the LE-stored uint64 there, seeded big-endian from job.nonce and
  // fixed by nicehash_mask -- so both are the reverse of the desired physical {prefix || 0...} layout.
  function addBeamhash3JobFields(job, prev_job, pool) {
    job.noncebytes  = 8;
    job.nonceoffset = 32;
    // prework(64hex) || nonce(16hex, zero placeholder) || extranonce(8hex, zero) = 88 hex = 44 bytes.
    const prework = orDefault(prev_job.header_hash, prev_job.blob_hex).padStart(64, "0").slice(0, 64);
    job.blob_hex = prework + "000000000000000000000000";

    // The native writes the nonce to the blob BIG-endian (set_job + the beamhash3 loop both bswap), so the
    // 8-byte nonce field's PHYSICAL bytes equal m_nonce64's bytes most-significant-first. Beam's nonceprefix
    // must occupy the LEADING physical bytes -> the HIGH bytes of m_nonce64. So job.nonce (read big-endian
    // by the native) and the nicehash mask are the plain {prefix || counter} layout, prefix at the front.
    // The low bytes are the free search counter; seed its top free byte to 1 so the first m_nonce64 is
    // never 0 (the native loop treats m_nonce64==0 as a test dispatch).
    const prefix = hexWithoutPrefix((pool && pool.beam_nonceprefix) || "").slice(0, 16);
    const prefixBytes = Math.min(prefix.length / 2, 8);
    let nonce = prefix.slice(0, prefixBytes * 2);
    if (prefixBytes < 8) {nonce += "01";}                  // seed the first free (counter) byte = 1
    job.nonce         = nonce.padEnd(16, "0");
    job.nicehash_mask = "ff".repeat(prefixBytes).padEnd(16, "0");
  }

  function addStandardJobFields(job, prev_job) {
    job.noncebytes  = orDefault(prev_job.noncebytes, 4);
    job.blob_hex    = orDefault(prev_job.blob, prev_job.blob_hex);
    job.nonceoffset = nonceOffsetOr(prev_job, job.algo === "ghostrider" ? 76 : 39);
  }

  function addNoncePrefix(job, prev_job) {
  // we need to create nonce with xn prefix and update nicehash_mask to cover it
    const nicehash_prefix = Buffer.from(prev_job.xn, "hex").subarray(0, job.noncebytes);
    job.nicehash_mask = Buffer.alloc(job.noncebytes, 0).fill(0xFF, 0, nicehash_prefix.length).toString("hex");
    job.nonce = Buffer.concat([nicehash_prefix, Buffer.alloc(job.noncebytes - nicehash_prefix.length, 0x00)]).toString("hex");
  }

  function defaultNicehashMask(job, pool_id, last_job_can_be_used) {
    const last_job = getLastJob();
    if (last_job_can_be_used && last_job.nicehash_mask) {return last_job.nicehash_mask;}
    return Buffer.alloc(job.noncebytes, 0)
      .fill(0xFF, 0, opt.pools[pool_id].is_nicehash ? 1 : 0)
      .toString("hex");
  }

  function addNonceFields(job, prev_job, pool_id) {
    if (prev_job.xn) {return addNoncePrefix(job, prev_job);}

    const last_job = getLastJob();
    const last_job_can_be_used = last_job && last_job.algo === job.algo;
    // use existing nicehash_mask or make a new one with FF00..00 that job.noncebytes long
    job.nicehash_mask = orDefault(prev_job.nicehash_mask,
      defaultNicehashMask(job, pool_id, last_job_can_be_used));
    job.nonce = orDefault(prev_job.nonce, reusableLastNonce(last_job_can_be_used));
  }

  function reusableLastNonce(last_job_can_be_used) {
    const last_job = getLastJob();
    return last_job_can_be_used && last_job.nonce ? last_job.nonce : "0";
  }

  function workerRuntimeEnv(algo, devEntry = null) {
  // Preserve "auto" here so compiler policy can distinguish its measured default from an explicit
  // generic fallback. The resolved backend still travels in the job and is shown in status output.
    const env = compilerPolicy.workerEnv(
      algo, process.env, process.platform, requestedJobBackend(algo));
    const entryTuning = devEntry
      ? gpuTuning.parseDeviceEntry(devEntry, algo).tuning : {};
    const tuningEnv = gpuTuning.tuningEnvironment(
      algo, {...configuredTuning(algo), ...entryTuning});
    if (algo !== "c29") {return Object.assign(env, tuningEnv);}

    // C29 submits hundreds of short SYCL kernels per second; legacy non-immediate
    // Level Zero command lists avoid the one-core immediate-list path on Intel GPUs.
    return Object.assign(env, {
      SYCL_UR_USE_LEVEL_ZERO_V2: "0",
      SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS: "0",
    }, tuningEnv);
  }

  function ensureWorkersForJob(algo, dev) {
    const last_job = getLastJob();
    if (!last_job || last_job.algo !== algo || last_job.dev !== dev)
    {h.recreate_threads(dev, messageHandler, (entry) => workerRuntimeEnv(algo, entry));}
  }

  // prev_job can be either job json from the pool or
  // previous job restored from the pool switch (with nonce that we need to take into account)
  function set_job(prev_job) {
    const algo = normalizeAlgoName(prev_job.algo || opt.job.algo);
    const dev = jobDev(algo);
    ensureWorkersForJob(algo, dev);
    const pool_id = opt.pool_ids.active;
    const job = baseJob(prev_job, algo, dev, pool_id);
    if (algo === "c29") {addC29JobFields(job, prev_job);}
    else if (algo === "beamhash3") {addBeamhash3JobFields(job, prev_job, opt.pools[pool_id]);}
    else if (isKaspaHeaderAlgo(algo)) {addKaspaHeaderJobFields(job, prev_job);}
    else if (isZelHashAlgo(algo)) {addZelHashJobFields(job, prev_job);}
    else if (isNonceAt32Algo(algo)) {addEthHashJobFields(job, prev_job);}
    else {addStandardJobFields(job, prev_job);}
    // BeamHash III seeds its nonce from the pool nonceprefix inside addBeamhash3JobFields; the generic
    // nonce/nicehash defaults would clobber that, so only run them for the other algos.
    if (algo !== "beamhash3") {addNonceFields(job, prev_job, pool_id);}
    else {
      job.nicehash_mask = orDefault(job.nicehash_mask, "0000000000000000");
      job.nonce = orDefault(job.nonce, "0000000000000000");
    }
    set_algo_msr(algo);
    setLastJob(job);
    h.messageWorkers({type: "job", job});
    return job;
  }

  function prepareBenchmarkJob(job) {
    job.backend_request = compilerPolicy.validateBackend(job.backend || "auto");
    job.backend = job.backend_request !== "auto"
      ? compilerPolicy.validateBackend(job.backend)
      : jobBackend(normalizeAlgoName(job.algo));
    if (normalizeAlgoName(job.algo) === "pearlhash") {
      addPearlHashJobFields(job);
    }
    if (isNonceAt32Algo(job.algo)) {
      job.noncebytes = 8;
      job.nonceoffset = 32;
      if (job.blob_hex && job.blob_hex.length === 64) {job.blob_hex += "0000000000000000";}
    }
    if (isKaspaHeaderAlgo(job.algo)) {
    // 80-byte Kaspa-style header, 8-byte nonce at offset 72.
      job.noncebytes = 8;
      job.nonceoffset = 72;
      if (!job.blob_hex || job.blob_hex.length !== 160)
      {job.blob_hex = "2a".repeat(32) + "52c9f84301000000" + "00".repeat(32) + "0000000000000000";}
    }
    if (isZelHashAlgo(job.algo)) {
    // Equihash 125,4 (ZelHash/Flux): 140-byte Zcash header with a 32-byte nonce at offset 108. Bench
    // over the deterministic block-400000 header so each solve finds ~1.88 proofs and the rate is Sol/s.
      job.noncebytes = 8;
      job.nonceoffset = 108;
      if (!job.blob_hex || job.blob_hex.length !== 280) {job.blob_hex = ZELHASH_BENCH_BLOB;}
      job.height = job.height || 400000;
    }
    if (job.algo === "beamhash3") {
    // BeamHash III: one Wagner solve per dispatch over a deterministic M4-shaped prework. Seed the
    // 8-byte nonce nonzero so the native path does not classify the dispatch as an is_test gen run.
      job.noncebytes = 8;
      job.nonceoffset = 32;
      if (!job.blob_hex || job.blob_hex.length !== 88) {job.blob_hex = BEAMHASH3_BENCH_BLOB;}
      job.nonce = job.nonce || "0100000000000000";
      job.nicehash_mask = job.nicehash_mask || "0000000000000000";
    }
    if (benchHeightByAlgo[job.algo]) {job.height = job.height || benchHeightByAlgo[job.algo];}
    if (job.algo === "etchash") {job.seed_hex = "";}
    return job;
  }

  return {
    set_algo_msr, requestedJobBackend, jobBackend, resolvedDeviceList,
    configuredTuning, addPearlHashJobFields, workerRuntimeEnv, set_job,
    prepareBenchmarkJob, defaultBenchAlgos,
  };
};
