"use strict";

const crypto = require("node:crypto");
const {isValidNonce} = require("./submission");
const {PROGPOW_ALGOS, ZELHASH_ALGOS, KASPA_HEADER_ALGOS} = require("./algorithms");
const {isIdentifier} = require("../helper/worker-protocol");

// Supported headers are small; leave ample room without forwarding an unbounded pool blob.
const MAX_MINING_BLOB_HEX_CHARS = 64 * 1024 * 2;

/**
 * @param {{
 *   h: typeof import("../helper"),
 *   opt: MinerOptions,
 *   process: NodeJS.Process,
 *   compilerPolicy: typeof import("../compiler-policy"),
 *   gpuTuning: typeof import("../gpu-tuning"),
 *   hexWithoutPrefix(value: unknown): string,
 *   normalizeAlgoName(algo: string | null | undefined): string | null | undefined,
 *   messageHandler: WorkerMessageHandler,
 *   isExiting(): boolean,
 *   getComputeCore(): ComputeCore | null,
 *   getLastJob(): MiningJob | null,
 *   setLastJob(job: MiningJob): void,
 * }} dependencies
 */
module.exports = ({
  h, opt, process, compilerPolicy, gpuTuning,
  hexWithoutPrefix, normalizeAlgoName, messageHandler, isExiting,
  getComputeCore, getLastJob, setLastJob,
}) => {

  // Pool job IDs may be reused, so live results also carry our own per-job identity.
  let liveJobToken = 0n;

  // Ask the control core to apply the algorithm's MSR preset over the saved defaults, if any.
  /** @param {string} algo */
  function setAlgoMsr(algo) {
    const computeCore = getComputeCore();
    if (!computeCore || !Object.keys(opt.default_msrs).length) {return;}
    const msrs = h.pack_msr(opt.default_msrs);
    msrs["algo"] = algo;
    computeCore.emit_to("write_msr", msrs);
  }

  /** @param {string} algo */
  function jobDev(algo) {
    const algoParam = opt.algo_params[algo];
    return algoParam && algoParam.dev ? algoParam.dev : opt.job.dev;
  }

  /** @param {string} algo */
  function requestedJobBackend(algo) {
    const algoParam = opt.algo_params[algo];
    const configured = algoParam && algoParam.backend !== "auto" ? algoParam.backend : opt.job.backend;
    return compilerPolicy.validateBackend(configured || "auto");
  }

  // Resolve "auto" to the compiler policy's backend for the detected GPU vendor and NVIDIA SM.
  /** @param {string} algo */
  function jobBackend(algo) {
    const requested = requestedJobBackend(algo);
    if (requested !== "auto") {return requested;}
    const gpu = compilerPolicy.gpuFromEnv(process.env);
    const sm = gpu === "nvidia" ? compilerPolicy.nvidiaComputeCapability(process.env) : null;
    const selected = gpu && compilerPolicy.selection(algo, gpu, process.platform,
      gpu === "nvidia" ? sm ?? 0 : null);
    return selected ? selected.backend : "auto";
  }

  /**
   * @param {string} algo
   * @param {GpuTuning} heuristic
   * @param {GpuTuning} named
   * @param {GpuTuning} entry
   * @returns {GpuTuning}
   */
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

  // Complete a configured device list with discovery's heuristic tuning. A bare GPU entry expands to
  // every discovered worker profile of that GPU; explicit fields always win over named tuning, which
  // wins over the heuristic.
  /**
   * @param {string} algo
   * @param {string} configuredDev
   * @param {string} heuristicDev
   * @param {GpuTuning} [namedTuning]
   */
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
      if (!explicitShape && entry.processes === 1 &&
          configuredCounts.get(entry.device) === 1 && matches.length > 1) {
        for (const candidate of matches) {
          result.push({
            ...entry,
            tuning: mergeDeviceTuning(algo, candidate.tuning, namedTuning, entry.tuning),
          });
        }
        continue;
      }
      const onlyMatch = matches[0];
      // Summing intensity only makes sense for intensity-based algorithms. Other primary fields
      // describe a profile (for example Pearl's matrix rows or C29's seed workgroup), not a sum.
      const heuristicTuning = onlyMatch && (matches.length === 1 || primaryField !== "intensity") ? onlyMatch.tuning :
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

  /** @param {string} algo @returns {GpuTuning} */
  function configuredTuning(algo) {
    return (opt.algo_params[algo] && opt.algo_params[algo].tuning) || {};
  }

  // The job-wide PearlHash matrix shape: explicit tuning, else the GPU policy profile, else defaults.
  /** @param {string | undefined} dev */
  function pearlhashShape(dev) {
    const configured = opt.algo_params["pearlhash"];
    const entries = typeof dev === "string" ? gpuTuning.parseDeviceList(dev, "pearlhash") : [];
    // CPU workers cannot share every GPU profile. Use a CPU entry as the common job shape when
    // present; complete GPU tuning is applied independently at the worker boundary.
    const baseEntry = entries.find((entry) => !entry.device.startsWith("gpu")) || entries[0];
    const entryTuning = baseEntry?.tuning || {};
    /** @type {GpuTuning} */
    const tuning = {...configured?.tuning, ...entryTuning};
    const usesGpu = baseEntry?.device.startsWith("gpu") === true;
    const gpu = usesGpu ? compilerPolicy.gpuFromEnv(process.env) : "";
    const selected = gpu ? compilerPolicy.selection("pearlhash", gpu, process.platform) : null;
    const profile = selected && selected.pearlhashProfile;
    const m = tuning.m ?? (profile && profile.m) ?? 131072;
    return {
      m,
      // An explicit M alone means a square matrix, matching per-worker tuning.
      n: tuning.n ?? (tuning.m !== undefined ? m : (profile && profile.n) ?? m),
      k: tuning.k ?? (profile && profile.k) ?? 4096,
      rank: tuning.rank ?? (profile && profile.rank) ?? 256,
    };
  }

  /** @param {unknown} value @param {boolean} decimalOnly @returns {number | null} */
  function pearlSeedValue(value, decimalOnly = false) {
    if (typeof value === "number") {
      return Number.isSafeInteger(value) && value >= 0 && value <= 0xffffffff ? value : null;
    }
    if (typeof value !== "string" || value.length > 16 ||
        !(decimalOnly ? /^\d+$/.test(value) : /^[0-9a-f]{1,16}$/i.test(value))) {return null;}
    try {
      const parsed = decimalOnly || /^\d+$/.test(value) ? BigInt(value) : BigInt("0x" + value);
      return parsed <= 0xffffffffn ? Number(parsed) : null;
    } catch {
      return null;
    }
  }

  /** @param {unknown} value @returns {number | null} */
  function pearlSeedStride(value) {
    const stride = pearlSeedValue(value, true);
    if (stride === null) {return null;}
    const wide = BigInt(stride);
    if (wide === 0n || (wide & (wide - 1n)) !== 0n) {
      return null;
    }
    return stride;
  }

  /** @param {number} slot @param {number} stride @param {number} threadCount @returns {string} */
  function randomPearlSeed(slot, stride, threadCount) {
    const maximum = 0xffffffffn - BigInt(threadCount - 1) * BigInt(stride);
    const slotWide = BigInt(slot);
    if (threadCount < 1 || maximum < slotWide) {
      throw new Error("PearlHash seed range is exhausted");
    }
    const choices = (maximum - slotWide) / BigInt(stride) + 1n;
    const random = BigInt(crypto.randomBytes(4).readUInt32BE(0));
    const start = (random % choices) * BigInt(stride) + slotWide;
    return Number(start).toString(16).padStart(8, "0");
  }

  /** @param {unknown} value @returns {number | null} */
  function pearlSeedHexValue(value) {
    if (typeof value === "number") {
      return Number.isSafeInteger(value) && value >= 0 && value <= 0xffffffff ? value : null;
    }
    if (typeof value !== "string" || !/^[0-9a-f]{1,8}$/i.test(value)) {return null;}
    return Number.parseInt(value, 16);
  }

  // Resume a PearlHash seed for the same job input and seed partition: the pool's carried nonce when
  // seed splitting was negotiated, else the controller's last seed for this pool and header.
  /** @param {PoolJob} poolJob @param {number} pool_id @param {number} slot @param {number} stride
   * @param {boolean} accepted @returns {string | undefined} */
  function reusablePearlSeed(poolJob, pool_id, slot, stride, accepted) {
    if (accepted && poolJob.nonce !== undefined) {
      const carried = pearlSeedHexValue(poolJob.nonce);
      if (carried !== null && carried % stride === slot) {
        return carried.toString(16).padStart(8, "0");
      }
    }
    const last = getLastJob();
    if (!last || last.algo !== "pearlhash" || last.pool_id !== pool_id ||
        (last.nonce_stride !== undefined) !== accepted) {return undefined;}
    const input = poolJob.blob || poolJob.blob_hex || "";
    if (!input || last.blob_hex !== input) {return undefined;}
    const lastStride = last.nonce_stride === undefined ? 1 : pearlSeedStride(last.nonce_stride);
    const lastSeed = pearlSeedHexValue(last.nonce);
    if (lastStride !== stride || lastSeed === null || lastSeed % stride !== slot) {return undefined;}
    return lastSeed.toString(16).padStart(8, "0");
  }

  // Choose the PearlHash start seed inside the negotiated slot/stride partition (slot 0, stride 1
  // without the pearl-seed-split extension). A fresh seed is random so restarts do not overlap.
  /** @param {MiningJob} job @param {PoolJob} poolJob @param {number} pool_id */
  function finalizePearlSeed(job, poolJob, pool_id) {
    const accepted = opt.pools[pool_id]?.extensions?.includes("pearl-seed-split") === true;
    const rawStride = accepted ? poolJob.nonce_stride : undefined;
    const stride = rawStride === undefined ? 1 : pearlSeedStride(rawStride);
    if (stride === null) {
      throw new Error("Invalid PearlHash seed stride");
    }
    const slot = accepted && poolJob.nonce_slot !== undefined
      ? pearlSeedValue(poolJob.nonce_slot) : 0;
    if (slot === null || slot >= stride) {
      throw new Error("Invalid PearlHash seed slot");
    }
    const resumed = reusablePearlSeed(poolJob, pool_id, slot, stride, accepted);
    job.nonce = resumed || randomPearlSeed(slot, stride, h.get_dev_threads(job.dev));
    if (accepted) {
      // Carry the validated effective stride even when the proxy omitted its default of one.
      job.nonce_stride = stride;
    } else {
      // A downstream proxy may include these fields without an acknowledged extension. Ignore
      // them, while still assigning a random direct-pool seed to avoid an all-zero overlap.
      delete job.nonce_stride;
    }
  }

  // Set the certificate version and matrix shape PearlHash jobs carry to the native core.
  /** @param {MiningJob} job */
  function addPearlHashJobFields(job) {
    const shape = pearlhashShape(job.dev);
    if (job.pearlhash_cert_version === undefined) {job.pearlhash_cert_version = 3;}
    if (job.pearlhash_cert_version !== 3) {
      throw new Error("Unsupported PearlHash certificate version");
    }
    if (shape.rank < 128) {
      throw new Error("PearlHash certificate version 3 requires rank >= 128");
    }
    job.noncebytes = 8;
    job.intensity = shape.m;
    job.pearlhash_n = shape.n;
    job.pearlhash_k = shape.k;
    job.pearlhash_rank = shape.rank;
  }

  // The algorithm-independent part of a mining job built from a pool job.
  /**
   * @param {PoolJob} poolJob
   * @param {string} algo
   * @param {string} dev
   * @param {number} pool_id
   * @returns {MiningJob}
   */
  function baseJob(poolJob, algo, dev, pool_id) {
    const pool = opt.pools[pool_id];
    if (!pool) {throw new Error(`Unknown pool ${pool_id}`);}
    /** @type {MiningJob} */
    const job = {
      algo:       algo,
      dev:        dev,
      worker_id:  poolJob.id ?? poolJob.worker_id ?? pool.worker_id ?? pool.login,
      job_id:     poolJob.job_id ?? "",
      header_hash: poolJob.header_hash || "",
      nonce:      algo === "pearlhash" ? 0 : poolJob.nonce ?? 0,
      height:     poolJob.height ?? 0,
      pool_id:    pool_id,
      backend_request: requestedJobBackend(algo),
      backend:    jobBackend(algo),
    };
    if (typeof poolJob.job_token === "string") {job.job_token = poolJob.job_token;}
    const seed = poolJob.seed_hash || poolJob.seed_hex;
    if (seed !== undefined) {job.seed_hex = seed;}
    if (poolJob.difficulty !== undefined) {job.difficulty = poolJob.difficulty;}
    if (algo === "pearlhash") {
      if (poolJob.pearlhash_cert_version !== undefined) {
        job.pearlhash_cert_version = poolJob.pearlhash_cert_version;
      }
      if (poolJob.pearlhash_base_target !== undefined) {
        job.pearlhash_base_target = poolJob.pearlhash_base_target;
      }
      addPearlHashJobFields(job);
    }
    job.target = jobTarget(poolJob, algo, job);
    return job;
  }

  // A 32-byte header hash followed by an 8-byte nonce at offset 32.
  const nonceAt32Algos = new Set([...PROGPOW_ALGOS, "etchash", "octopus", "autolykos2", "fishhash"]);
  // Heights sampled from coin mainnets so benchmark DAG/table sizes match live pool jobs
  // (epoch-0 sizes overstate hashrate by ~7-10% on these algos): ETC 2026-06-04, RVN and ERG 2026-06-12.
  /** @type {Record<string, number>} */
  const benchHeightByAlgo = {
    etchash:    24689903,
    octopus:    152521905,
    kawpow:     4407982,
    firopow:    600000,
    evrprogpow: 1800000,
    meowpow:    825000,
    autolykos2: 1806198,
  };
  const moneroOceanAlgos = new Set([
    "autolykos2",
    "c29",
    "cn/gpu",
    "etchash",
    "ghostrider",
    "kawpow",
    "panthera",
    "rx/0",
    "rx/arq",
  ]);
  const defaultBenchAlgos = new Set([
    ...moneroOceanAlgos,
    // rx/2 remains a useful compatibility benchmark. PearlHash is a headline GPU PoUW number even
    // though neither algorithm is currently selected by MoneroOcean profitability switching.
    "rx/2",
    "pearlhash",
  ]);

  // A deterministic 140-byte Flux header for benching the Equihash 125,4 GPU solver (mainnet block
  // 400000). Each Wagner solve over this header finds 2 distinct proofs in ~2.2 s on a B580, so the
  // reported Sol/s is the solver's true throughput. The 32-byte nonce lives at offset 108.
  const ZELHASH_BENCH_BLOB =
    "04000000a8675c842f7a1342fadd00cd9b4e4909526b1c0ab5a747c5529b4deb13000000" +
  "ce7d6ea2452245925fc70c3a08a3c3dd2ca4beab7481f237a19751666bfd25c3" +
  "0fd282d94b1e1a7f2c57eb3fb9e2853d990753fa137e13c99bd43f220d4fce69" +
  "90e44f5dce28421d" +
  "600000160000000000000000000000000000000000000000000000009cfd1100";

  const ZHASH_BENCH_BLOB =
    "0400000008e9694cc2120ec1b5733cc12687b609058eec4f7046a521ad1d1e3049b40000" +
    "3e7420ed6f40659de0305ef9b7ec037f4380ed9848bc1c015691c90aa16ff393" +
    "0000000000000000000000000000000000000000000000000000000000000000" +
    "c9310d5874e0001f" +
    "000000000000000000000000000000010b000000000000000000000000666666";

  const EQUIHASH192_7_BENCH_BLOB =
    "04000000ecf888bb9e8440dff1eca5ff69c277e85462f306ec785719a76e4dd20f0b0000" +
    "c450f3fd2a66b462f4133c48cc636655ac055de95072bd693514c9c7156dfa8de" +
    "2004d086a6929b60cb4e4efbbfcf41d3fda50ad985fc421c990217a1daef400c5" +
    "8d776ad03c141e8001fde00f6dcbadb169c9131b3a07c44e9b11ca00000000000000005b15db75";

  const C30_BENCH_HEADER =
    "8bbb8897a7967634e15bae662ee23e16e8c85669f3ca0a9e6584f8f4aa41f220";

  // Iron Fish pool jobs use a 180-byte header with the big-endian nonce at byte 172. Header values
  // do not affect benchmark cost, so keep the production shape deterministic and otherwise empty.
  const FISHHASH_BENCH_BLOB = "00".repeat(180);

  // BeamHash III M4 keystone-shaped benchmark blob: prework(32) || nonce(8) || extranonce(4).
  const BEAMHASH3_BENCH_BLOB =
    "fc40996a518c221384c9f2542ca811cd66c4ccddb001ef40b9f9ba059c20352e" +
  "0100000000000000" +
  "00000000";

  const NEXAPOW_BENCH_BLOB =
    "0a4ac49b2d02e3c8d12c7093255ba7c49624f9c374d9f1c2f8e37c58705e74b0" +
    "10000000000000001182dc5800000000";

  // Benchmark layouts with a fixed header size. A supplied blob of another length is replaced by
  // the deterministic default; `nonce` seeds an unset benchmark nonce.
  /** @type {Record<string, {noncebytes: number, nonceoffset: number, blob: string, nonce?: string}>} */
  const benchLayouts = {
    // BeamHash III: one Wagner solve per dispatch over a deterministic M4-shaped prework. A nonzero
    // nonce keeps the native path from classifying the dispatch as an is_test gen run.
    beamhash3: {noncebytes: 8, nonceoffset: 32, blob: BEAMHASH3_BENCH_BLOB, nonce: "0100000000000000"},
    verthash: {noncebytes: 4, nonceoffset: 76, blob: "00".repeat(80)},
    xelishashv3: {noncebytes: 8, nonceoffset: 40, blob: "00".repeat(112)},
    nexapow: {noncebytes: 8, nonceoffset: 40, blob: NEXAPOW_BENCH_BLOB, nonce: "1182dc5800000000"},
    // Equihash 192,7 counts input solves; ZHash and ZelHash count proofs.
    equihash192_7: {noncebytes: 8, nonceoffset: 108, blob: EQUIHASH192_7_BENCH_BLOB},
    zhash: {noncebytes: 8, nonceoffset: 108, blob: ZHASH_BENCH_BLOB},
    zelhash: {noncebytes: 8, nonceoffset: 108, blob: ZELHASH_BENCH_BLOB},
  };
  for (const algo of KASPA_HEADER_ALGOS) {
    benchLayouts[algo] = {
      noncebytes: 8, nonceoffset: 72,
      blob: "2a".repeat(32) + "52c9f84301000000" + "00".repeat(32) + "0000000000000000",
    };
  }

  /** @param {unknown} value @param {string} algo @returns {number | bigint} */
  function validDifficulty(value, algo) {
    if (typeof value === "bigint") {
      if (value > 0n) {return value;}
      throw new Error(`Invalid ${algo} job difficulty`);
    }
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      throw new Error(`Invalid ${algo} job difficulty`);
    }
    return value;
  }

  // The 256-bit or compact target hex the algorithm's native search compares against, from the
  // pool's target or, when absent, its difficulty.
  /** @param {PoolJob} poolJob @param {string} algo @param {Partial<MiningJob>} [job] */
  function jobTarget(poolJob, algo, job = {}) {
    const explicitTarget = poolJob.target || "";
    if (algo === "pearlhash") {
      if (poolJob.pearlhash_base_target) {
        if (!job.pearlhash_k || !job.pearlhash_rank) {
          throw new Error("PearlHash job shape is incomplete");
        }
        return h.pearlhashTarget(
          poolJob.pearlhash_base_target, job.pearlhash_k, job.pearlhash_rank,
          job.pearlhash_cert_version);
      }
      if (explicitTarget) {return hexWithoutPrefix(explicitTarget).padStart(64, "0");}
      return h.fullDiff2Target(validDifficulty(poolJob.difficulty, algo));
    }
    if (ZELHASH_ALGOS.has(algo) || algo === "nexapow") {
      // Flux and Nexa deliver a 256-bit big-endian share target. When a pool does not send one, use
      // the lenient floor(2^256 / difficulty) fallback.
      if (explicitTarget) {return hexWithoutPrefix(explicitTarget).padStart(64, "0");}
      return h.fullDiff2Target(validDifficulty(poolJob.difficulty, algo));
    }
    if (algo === "verthash") {
      if (explicitTarget) {return hexWithoutPrefix(explicitTarget).padStart(64, "0");}
      return h.ethDiff2Target(validDifficulty(poolJob.difficulty, algo), 256n);
    }
    if (!nonceAt32Algos.has(algo)) {
      if (explicitTarget) {return explicitTarget;}
      const difficulty = validDifficulty(poolJob.difficulty, algo);
      if (typeof difficulty === "number" && !Number.isInteger(difficulty)) {
        throw new Error(`Invalid ${algo} job difficulty`);
      }
      return h.diff2target(difficulty);
    }
    // Pool adapters normalize full targets to hexadecimal, including Ergo's decimal wire target.
    // Generic MoneroOcean jobs may still carry compact targets, converted through difficulty below.
    if (explicitTarget && hexWithoutPrefix(explicitTarget).length > 16) {
      return hexWithoutPrefix(explicitTarget).padStart(64, "0");
    }
    const difficulty = poolJob.difficulty ??
      (explicitTarget ? h.target2diff(explicitTarget) : undefined);
    return h.ethDiff2Target(validDifficulty(difficulty, algo));
  }

  // Lay out a C29 job for Grin (pre_pow then nonce) or Tari (nonce then blob).
  /** @param {MiningJob} job @param {PoolJob} poolJob */
  function addC29JobFields(job, poolJob) {
    job.proofsize = poolJob.proofsize || 42;
    if (poolJob.pre_pow) { // GRIN
      job.noncebytes  = poolJob.noncebytes || 4;
      job.blob_hex    = poolJob.pre_pow + "00".repeat(job.noncebytes);
      job.nonceoffset = poolJob.pre_pow.length / 2;
    } else if (poolJob.blob) { // TARI C29
      job.noncebytes  = poolJob.noncebytes || 8;
      job.blob_hex    = "00".repeat(job.noncebytes) + poolJob.blob;
      job.nonceoffset = 0;
    } else {
      if (poolJob.noncebytes !== undefined) {job.noncebytes = poolJob.noncebytes;}
      if (poolJob.blob_hex !== undefined) {job.blob_hex = poolJob.blob_hex;}
      if (poolJob.nonceoffset !== undefined) {job.nonceoffset = poolJob.nonceoffset;}
    }
  }

  /** @param {MiningJob} job @param {PoolJob} poolJob */
  function addC30JobFields(job, poolJob) {
    job.proofsize = 42;
    job.noncebytes = 8;
    job.nonceoffset = 0;
    const blob = poolJob.blob || poolJob.blob_hex;
    if (blob !== undefined) {job.blob_hex = blob;}
  }

  /** @param {MiningJob} job @param {PoolJob} poolJob */
  function addEthHashJobFields(job, poolJob) {
    job.noncebytes = poolJob.noncebytes || 8;
    job.nonceoffset = poolJob.nonceoffset !== undefined ? poolJob.nonceoffset : 32;

    const blob = poolJob.blob || poolJob.blob_hex;
    if (blob !== undefined) {
      job.blob_hex = blob.length === 64 ? blob + "0000000000000000" : blob;
    }
  }

  /** @param {MiningJob} job @param {PoolJob} poolJob */
  function addZelHashJobFields(job, poolJob) {
    // ZIP-301: the 32-byte nonce starts at byte 108. The search counter follows the pool's nonce1;
    // retain ntime and nonce1 length so submission can reconstruct nonce2.
    addFixedNonceBlobFields(job, poolJob, 108);
    job.ntime       = poolJob.ntime || "";
    job.nonce1_len  = poolJob.nonce1_len || 0;
  }

  /** @param {MiningJob} job @param {PoolJob} poolJob @param {number} defaultOffset */
  function addFixedNonceBlobFields(job, poolJob, defaultOffset) {
    job.noncebytes  = poolJob.noncebytes || 8;
    job.nonceoffset = poolJob.nonceoffset !== undefined ? poolJob.nonceoffset : defaultOffset;
    const blob = poolJob.blob || poolJob.blob_hex;
    if (blob !== undefined) {job.blob_hex = blob;}
  }

  /** @param {MiningJob} job @param {PoolJob} poolJob */
  function addVerthashJobFields(job, poolJob) {
    job.noncebytes = 4;
    job.nonceoffset = 76;
    const blob = poolJob.blob || poolJob.blob_hex;
    if (blob !== undefined) {job.blob_hex = blob;}
    job.extranonce2 = poolJob.extranonce2 || "";
    job.ntime = poolJob.ntime || "";
  }

  // BeamHash III blob = prework(32) || nonce(8) || extranonce(4). The pool's nonce prefix occupies the
  // leading physical nonce bytes. Native code stores the numeric counter in big-endian byte order.
  /** @param {MiningJob} job @param {PoolJob} poolJob @param {PoolConfig | undefined} pool */
  function addBeamhash3JobFields(job, poolJob, pool) {
    job.noncebytes  = 8;
    job.nonceoffset = 32;
    // prework(64hex) || nonce(16hex, zero placeholder) || extranonce(8hex, zero) = 88 hex = 44 bytes.
    const prework = poolJob.header_hash || poolJob.blob_hex || "";
    if (!/^[0-9a-f]{64}$/i.test(prework)) {
      throw new Error("Invalid BeamHash III prework");
    }
    job.blob_hex = prework + "000000000000000000000000";

    // The native writes the nonce to the blob BIG-endian (native set_job and its loop both bswap), so the
    // 8-byte nonce field's PHYSICAL bytes equal m_nonce64's bytes most-significant-first. Beam's nonceprefix
    // must occupy the LEADING physical bytes -> the HIGH bytes of m_nonce64. So job.nonce (read big-endian
    // by the native) and the nicehash mask are the plain {prefix || counter} layout, prefix at the front.
    // The low bytes are the free search counter; seed its top free byte to 1 so the first m_nonce64 is
    // never 0 (the native loop treats m_nonce64==0 as a test dispatch).
    const rawPrefix = poolJob.xn ?? pool?.beam_nonceprefix;
    if (rawPrefix !== undefined && typeof rawPrefix !== "string") {
      throw new Error("Invalid Beam nonce prefix");
    }
    const prefix = hexWithoutPrefix(rawPrefix || "");
    if (!/^(?:[0-9a-f]{2}){0,6}$/i.test(prefix)) {
      throw new Error("Invalid Beam nonce prefix");
    }
    const prefixBytes = prefix.length / 2;
    let nonce = prefix.slice(0, prefixBytes * 2);
    if (prefixBytes < 8) {
      nonce += "01"; // Seed the first free counter byte so native test mode is not selected.
    }
    job.nonce = poolJob.nonce === undefined ? nonce.padEnd(16, "0") : poolJob.nonce;
    if (!isValidNonce(job.nonce, 8, prefix, true)) {throw new Error("Invalid Beam nonce");}
    poolJob.xn = prefix;
    job.nicehash_mask = "ff".repeat(prefixBytes).padEnd(16, "0");
  }

  /** @param {MiningJob} job @param {PoolJob} poolJob */
  function addStandardJobFields(job, poolJob) {
    job.noncebytes  = poolJob.noncebytes || job.noncebytes || 4;
    const blob = poolJob.blob || poolJob.blob_hex;
    if (blob !== undefined) {job.blob_hex = blob;}
    job.nonceoffset = poolJob.nonceoffset !== undefined
      ? poolJob.nonceoffset : (job.algo === "ghostrider" ? 76 : 39);
  }

  // Check the blob, nonce, mask, and nonce offset before a job reaches the native core.
  /** @param {MiningJob} job @param {boolean} [requireNonceOffset] @returns {NativeJob} */
  function validateMiningBlob(job, requireNonceOffset = true) {
    if (!job.blob_hex || job.blob_hex.length > MAX_MINING_BLOB_HEX_CHARS ||
        job.blob_hex.length % 2 !== 0 || /[^0-9a-f]/i.test(job.blob_hex)) {
      throw new Error(`Invalid ${job.algo} job blob`);
    }
    const noncebytes = nonceByteCount(job);
    const nonceHexLength = noncebytes * 2;
    const nonce = job.nonce;
    if (nonce !== undefined && !isValidNonce(nonce, noncebytes)) {
      throw new Error(`Invalid ${job.algo} nonce`);
    }
    const nonceMask = job.nicehash_mask;
    if (nonceMask !== undefined &&
        (typeof nonceMask !== "string" || !/^[0-9a-f]+$/i.test(nonceMask) ||
          nonceMask.length > nonceHexLength)) {
      throw new Error(`Invalid ${job.algo} nonce mask`);
    }
    const nonceoffset = job.nonceoffset;
    const invalidNonceOffset = nonceoffset === undefined
      ? requireNonceOffset
      : !Number.isSafeInteger(nonceoffset) || nonceoffset < 0 ||
        nonceoffset + noncebytes > job.blob_hex.length / 2;
    if (invalidNonceOffset) {
      throw new Error(`Invalid ${job.algo} nonce offset`);
    }
    return Object.assign(job, {blob_hex: job.blob_hex});
  }

  // A live job additionally needs a target and the identifiers that route its results back.
  /** @param {MiningJob} job @returns {LiveNativeJob} */
  function validateLiveMiningJob(job) {
    const nativeJob = validateMiningBlob(job);
    const target = nativeJob.target;
    if (typeof target !== "string" || !/^(?:0x)?[0-9a-f]{1,64}$/i.test(target)) {
      throw new Error(`Invalid ${job.algo} target`);
    }
    const job_token = nativeJob.job_token;
    if (typeof job_token !== "string" || job_token.length === 0) {
      throw new Error(`Invalid ${job.algo} job token`);
    }
    const {job_id, pool_id, worker_id} = nativeJob;
    if (!isIdentifier(job_id)) {throw new Error(`Invalid ${job.algo} job_id`);}
    if (!isIdentifier(pool_id)) {throw new Error(`Invalid ${job.algo} pool_id`);}
    if (!isIdentifier(worker_id)) {throw new Error(`Invalid ${job.algo} worker_id`);}
    return Object.assign(nativeJob, {target, job_id, pool_id, worker_id, job_token});
  }

  /** @param {MiningJob} job */
  function nonceByteCount(job) {
    if (job.noncebytes !== 4 && job.noncebytes !== 8) {
      throw new Error(`Invalid ${job.algo} nonce size`);
    }
    return job.noncebytes;
  }

  // Seed the nonce with the pool's extranonce (xn) prefix and mask that prefix as fixed.
  /** @param {MiningJob} job @param {PoolJob} poolJob */
  function addNoncePrefix(job, poolJob) {
    // we need to create nonce with xn prefix and update nicehash_mask to cover it
    const noncebytes = nonceByteCount(job);
    const xn = poolJob.xn || "";
    if (xn.length % 2 !== 0 || /[^0-9a-f]/i.test(xn)) {throw new Error("Invalid extranonce");}
    const noncePrefix = Buffer.from(xn, "hex").subarray(0, noncebytes);
    job.nicehash_mask = Buffer.alloc(noncebytes, 0)
      .fill(0xFF, 0, noncePrefix.length).toString("hex");
    let suffix = Buffer.alloc(noncebytes - noncePrefix.length, 0x00);
    if (job.algo === "c29" && poolJob.nonce === undefined && suffix.length > 0) {
      // A same-template C29 job after an algorithm switch must not replay deterministic proof search.
      suffix = crypto.randomBytes(suffix.length);
      suffix[0] = (suffix[0] ?? 0) & 0x7f; // Leave room to advance without crossing the pool prefix.
    }
    const seed = Buffer.concat([noncePrefix, suffix]).toString("hex");
    // Resume only this accepted job's counter, keeping its protected prefix unchanged.
    job.nonce = poolJob.nonce === undefined ? seed : poolJob.nonce;
    if (!isValidNonce(job.nonce, noncebytes, xn)) {throw new Error(`Invalid ${job.algo} nonce`);}
  }

  /** @param {MiningJob} job @param {number} pool_id */
  function defaultNicehashMask(job, pool_id) {
    const pool = opt.pools[pool_id];
    if (!pool) {throw new Error(`Unknown pool ${pool_id}`);}
    const noncebytes = nonceByteCount(job);
    const nicehash = pool.is_nicehash || pool.negotiated_nicehash === true;
    if (nicehash && noncebytes === 4 &&
        job.algo !== "c29" && job.algo !== "verthash") {
      return "000000ff";
    }
    return Buffer.alloc(noncebytes, 0)
      .fill(0xFF, 0, nicehash ? 1 : 0)
      .toString("hex");
  }

  // Choose the nicehash mask and start nonce, resuming the last job's position for the same pool.
  /** @param {MiningJob} job @param {PoolJob} poolJob @param {number} pool_id */
  function addNonceFields(job, poolJob, pool_id) {
    if (poolJob.xn) {return addNoncePrefix(job, poolJob);}

    const lastJob = getLastJob();
    const reusable = lastJob && lastJob.pool_id === pool_id && lastJob.algo === job.algo
      ? lastJob : null;
    // Reuse the existing mask or choose the protocol-specific default for this nonce layout.
    job.nicehash_mask = poolJob.nicehash_mask ??
      (reusable?.nicehash_mask || defaultNicehashMask(job, pool_id));
    job.nonce = poolJob.nonce ??
      (reusable?.nonce !== undefined ? reusable.nonce : "0");
  }

  // Environment for one worker process: compiler policy plus the entry's GPU tuning variables.
  /** @param {string} algo @param {string | null} [devEntry] */
  function workerRuntimeEnv(algo, devEntry = null) {
    // Preserve "auto" so compiler policy can distinguish its measured default from an explicit
    // generic fallback. The resolved backend still travels in the job and status output.
    const env = compilerPolicy.workerEnv(
      algo, process.env, process.platform, requestedJobBackend(algo));
    const parsedEntry = devEntry ? gpuTuning.parseDeviceEntry(devEntry, algo) : null;
    const entryTuning = {...(parsedEntry?.tuning || {})};
    // cpu*B selects the native CPU hash batch/thread shape; it is not an algorithm tuning option.
    // Numbered cpuN entries are SYCL CPU devices, so retain their compute tuning.
    if (parsedEntry?.device === "cpu") {delete entryTuning.intensity;}
    const tuningEnv = gpuTuning.tuningEnvironment(
      algo, {...configuredTuning(algo), ...entryTuning});
    if (algo !== "c29" && algo !== "equihash192_7" && algo !== "zhash") {
      return Object.assign(env, tuningEnv);
    }

    // C29's short kernels and the Equihash solvers' ordered stages otherwise take the one-core
    // immediate-list path on Intel. Legacy Level Zero lists cut that CPU load without reducing
    // measured GPU work.
    return Object.assign(env, {
      SYCL_UR_USE_LEVEL_ZERO_V2: "0",
      SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS: "0",
    }, tuningEnv);
  }

  /** @param {string} algo @param {string} dev */
  function ensureWorkersForJob(algo, dev) {
    const lastJob = getLastJob();
    if (!lastJob || lastJob.algo !== algo || lastJob.dev !== dev) {
      h.recreate_threads(dev, messageHandler, (entry) => workerRuntimeEnv(algo, entry));
    }
  }

  // poolJob can be either job json from the pool or
  // previous job restored from the pool switch (with nonce that we need to take into account)
  /** @param {PoolJob} poolJob */
  function setJob(poolJob) {
    const algo = normalizeAlgoName(poolJob.algo || opt.job.algo);
    if (!algo) {throw new Error("Pool job has no algorithm");}
    poolJob.job_token = (++liveJobToken).toString();
    const dev = jobDev(algo);
    const pool_id = opt.pool_ids.active;
    const job = baseJob(poolJob, algo, dev, pool_id);
    if (algo === "c29") {
      addC29JobFields(job, poolJob);
    } else if (algo === "c30") {
      addC30JobFields(job, poolJob);
    } else if (algo === "beamhash3") {
      addBeamhash3JobFields(job, poolJob, opt.pools[pool_id]);
    } else if (KASPA_HEADER_ALGOS.has(algo)) {
      // Kaspa-family headers place their 8-byte little-endian nonce at offset 72.
      addFixedNonceBlobFields(job, poolJob, 72);
    } else if (ZELHASH_ALGOS.has(algo)) {
      addZelHashJobFields(job, poolJob);
    } else if (algo === "verthash") {
      addVerthashJobFields(job, poolJob);
    } else if (algo === "xelishashv3" || algo === "nexapow") {
      addFixedNonceBlobFields(job, poolJob, 40);
    } else if (nonceAt32Algos.has(algo)) {
      addEthHashJobFields(job, poolJob);
    } else {
      addStandardJobFields(job, poolJob);
    }
    // BeamHash III owns its prefixed nonce layout; generic defaults would clobber it.
    if (algo !== "beamhash3") {addNonceFields(job, poolJob, pool_id);}
    if (algo === "pearlhash") {finalizePearlSeed(job, poolJob, pool_id);}
    const nativeJob = validateLiveMiningJob(job);
    poolJob.noncebytes = nonceByteCount(nativeJob);
    // Pool notifications can arrive while the already-closed workers are draining.
    if (isExiting()) {return nativeJob;}
    ensureWorkersForJob(algo, dev);
    setAlgoMsr(algo);
    setLastJob(nativeJob);
    h.messageWorkers({type: "job", job: nativeJob});
    return nativeJob;
  }

  // Resolve the backend and PearlHash shape for a bench or test job that did not come from a pool.
  /** @param {MiningJob} job @returns {MiningJob} */
  function addDirectJobDefaults(job) {
    const requestedBackend = compilerPolicy.validateBackend(job.backend || "auto");
    job.backend_request = requestedBackend;
    job.backend = requestedBackend === "auto"
      ? jobBackend(normalizeAlgoName(job.algo) || job.algo)
      : requestedBackend;
    job.noncebytes ??= 4;
    if (normalizeAlgoName(job.algo) === "pearlhash") {
      // Direct pool/bench/test jobs have no negotiated downstream extension. Ignore a stray
      // proxy-only stride rather than allowing it to alter the established seed-0/stride-1 path.
      delete job.nonce_stride;
      delete job.nonce_slot;
      job.nonce ??= 0;
      addPearlHashJobFields(job);
    }
    return job;
  }

  /** @param {MiningJob} job @returns {NativeJob} */
  function prepareTestJob(job) {
    addDirectJobDefaults(job);
    // These test-only paths hash/solve the supplied blob without mutating an embedded nonce. Keep
    // the offset optional only at that boundary; every test path which reads a nonce remains strict.
    const nativeCpuOnly = gpuTuning.parseDeviceList(job.dev, job.algo)
      .every((entry) => entry.device === "cpu");
    const skipsNonce = nativeCpuOnly || job.algo === "c29" || job.algo === "cn/gpu" ||
      job.algo === "pearlhash";
    return validateMiningBlob(job, !skipsNonce);
  }

  // Complete a bench job with a live-like nonce layout and, where the algorithm needs one, a
  // deterministic default header and benchmark height. A supplied blob of the wrong size is replaced.
  /** @param {MiningJob} job @returns {NativeJob} */
  function prepareBenchmarkJob(job) {
    addDirectJobDefaults(job);
    job.nonceoffset ??= 39;
    // The native core historically supplied these defaults. Make them explicit at the JavaScript
    // boundary so every benchmark command has the same complete nonce layout as a live job.
    if (job.algo === "c30") {
      job.proofsize = 42;
      job.noncebytes = 8;
      job.nonceoffset = 0;
      if (job.blob_hex?.length === 80) {
        // Normalize the old test-vector header||nonce form to the one live search representation:
        // a 32-byte Cortex seal hash plus an external 64-bit nonce counter.
        if (/[^0-9a-f]/i.test(job.blob_hex)) {throw new Error("Invalid c30 job blob");}
        job.nonce ||= Buffer.from(job.blob_hex.slice(64), "hex").reverse().toString("hex");
        job.blob_hex = job.blob_hex.slice(0, 64);
      } else if (!job.blob_hex || job.blob_hex.length !== 64) {
        job.blob_hex = C30_BENCH_HEADER;
      }
      job.nonce = job.nonce || "1e000000d90820d4";
    }
    if (job.algo === "fishhash") {
      job.noncebytes = 8;
      if (job.blob_hex?.length === 64) {
        job.blob_hex += "0000000000000000";
      }
      if (job.blob_hex?.length === 80) {
        job.nonceoffset = 32;
      } else {
        if (job.blob_hex?.length !== 360) {
          job.blob_hex = FISHHASH_BENCH_BLOB;
        }
        // FishHash v3 stores its 8-byte randomness at the end of the 180-byte header.
        job.nonceoffset = 172;
      }
    } else if (nonceAt32Algos.has(job.algo)) {
      job.noncebytes = 8;
      job.nonceoffset = 32;
      if (job.blob_hex && job.blob_hex.length === 64) {job.blob_hex += "0000000000000000";}
    }
    const layout = Object.hasOwn(benchLayouts, job.algo) ? benchLayouts[job.algo] : undefined;
    if (layout) {
      job.noncebytes = layout.noncebytes;
      job.nonceoffset = layout.nonceoffset;
      if (job.blob_hex?.length !== layout.blob.length) {job.blob_hex = layout.blob;}
      if (layout.nonce) {job.nonce = job.nonce || layout.nonce;}
    }
    if (ZELHASH_ALGOS.has(job.algo) && job.blob_hex) {
      // The solver's search counter starts at zero inside the 32-byte header nonce.
      job.blob_hex = job.blob_hex.slice(0, 216) + "0".repeat(16) + job.blob_hex.slice(232);
      job.height = job.height || 400000;
    }
    if (job.algo === "beamhash3") {job.nicehash_mask = job.nicehash_mask || "0000000000000000";}
    // Benchmark the steady no-share path, not the all-candidate test target.
    if (job.algo === "nexapow") {job.target = "00".repeat(32);}
    const benchmarkHeight = benchHeightByAlgo[job.algo];
    if (benchmarkHeight) {job.height = job.height || benchmarkHeight;}
    if (job.algo === "etchash") {job.seed_hex = "";}
    return validateMiningBlob(job);
  }

  return {
    setAlgoMsr, requestedJobBackend, jobBackend, resolvedDeviceList,
    configuredTuning, workerRuntimeEnv, setJob,
    prepareTestJob, prepareBenchmarkJob, defaultBenchAlgos, moneroOceanAlgos,
  };
};
