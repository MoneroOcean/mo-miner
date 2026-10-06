"use strict";

const zlib = require("node:zlib");

const {MAX_PEARL_PROOF_BYTES, MAX_PEARL_PROOF_BASE64} = require("../helper/worker-protocol");
const MAX_JOB_TOKEN_CHARS = 256;
const MAX_WORKER_ID_CHARS = 4096;
const MAX_HEADER_HASH_HEX_CHARS = 256 * 2;
// CompactSize plus the largest supported Equihash proof: (192,7) at 403 bytes.
const MAX_SOLUTION_HEX_CHARS = 403 * 2;
const MAX_EDGES_HEX_CHARS = 42 * 8;

/**
 * @param {{
 *   fs: typeof import("node:fs"),
 *   h: typeof import("../helper"),
 *   p: {pool_write(poolId: number, message: UnknownRecord): unknown},
 *   opt: MinerOptions,
 *   submission: typeof import("./submission"),
 *   test: MinerTestState,
 *   normalizeExpectedResults(algo: string | null, value: string): string[],
 *   matchesTestResult(algo: string | null, actual: string, expected: string): boolean,
 *   exit(code: number, force?: boolean): false,
 *   getLastJob(): MiningJob | null,
 *   getAlgoParamsBenchCallback(): ((hashrate: number) => unknown) | null,
 * }} dependencies
 */
module.exports = ({
  fs, h, p, opt, submission, test, normalizeExpectedResults,
  matchesTestResult, exit, getLastJob, getAlgoParamsBenchCallback,
}) => {

  /** @type {Record<number, number>} */
  let thread_hashrates = Object.create(null);
  /** @type {string | null} */
  let hashrate_job_identity = null;
  const cortex_pending_limit = 4096;
  let cortex_submit_id = 72;

  function nextCortexSubmitId() {
    if (cortex_submit_id === Number.MAX_SAFE_INTEGER) {
      throw new Error("Cortex submit request id space exhausted");
    }
    cortex_submit_id++;
    // ID 100 belongs to the concurrent ctxc_getWork request and must never identify a share.
    if (cortex_submit_id === 100) {cortex_submit_id++;}
    return cortex_submit_id;
  }

  /** @param {unknown} value @returns {value is UnknownRecord} */
  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  /** @param {unknown} pool_id @returns {PoolConfig | null} */
  function poolAt(pool_id) {
    const index = typeof pool_id === "number" ? pool_id :
      typeof pool_id === "string" && /^\d+$/.test(pool_id) ? Number(pool_id) : -1;
    return Number.isSafeInteger(index) && Object.hasOwn(opt.pools, index)
      ? opt.pools[index] || null : null;
  }

  function invalidWorkerMessage() {
    return h.log_err("Invalid compute core message");
  }

  /** @param {string} proof @returns {Buffer | null} */
  function decodePearlProof(proof) {
    if (proof.length > MAX_PEARL_PROOF_BASE64) {return null;}
    const raw = Buffer.from(proof, "base64");
    return raw.length > 0 && raw.length <= MAX_PEARL_PROOF_BYTES &&
      raw.toString("base64") === proof ? raw : null;
  }

  /** @param {unknown} value @returns {value is WorkerResult} */
  function isWorkerResult(value) {
    if (!isObject(value)) {return false;}
    if (typeof value["pool_id"] !== "string" || !/^\d+$/.test(value["pool_id"]) ||
        !isJobId(value["job_id"]) ||
        typeof value["job_token"] !== "string" || value["job_token"].length === 0 ||
        value["job_token"].length > MAX_JOB_TOKEN_CHARS ||
        typeof value["worker_id"] !== "string" || value["worker_id"].length > MAX_WORKER_ID_CHARS ||
        !isNonce(value["nonce"])) {
      return false;
    }
    for (const field of ["commitment", "hash", "mix_hash"]) {
      const hex = value[field];
      if (hex !== undefined && (typeof hex !== "string" || !/^[0-9a-f]{64}$/i.test(hex))) {
        return false;
      }
    }
    const headerHash = value["header_hash"];
    if (headerHash !== undefined) {
      if (typeof headerHash !== "string") {return false;}
      const rawHeaderHash = headerHash.replace(/^0x/i, "");
      if (rawHeaderHash.length > MAX_HEADER_HASH_HEX_CHARS || !/^[0-9a-f]{64,}$/i.test(rawHeaderHash)) {
        return false;
      }
    }
    const plainProof = value["plain_proof"];
    if (plainProof !== undefined &&
        (typeof plainProof !== "string" || plainProof.length > MAX_PEARL_PROOF_BASE64)) {
      return false;
    }
    const jackpot = value["jackpot"];
    if (jackpot !== undefined &&
        (typeof jackpot !== "string" || !/^[0-9a-f]{64}$/.test(jackpot))) {return false;}
    const adjustmentFactor = value["adjustment_factor"];
    if (adjustmentFactor !== undefined &&
        (typeof adjustmentFactor !== "string" || !/^[1-9]\d*$/.test(adjustmentFactor) ||
         !Number.isSafeInteger(Number(adjustmentFactor)) || Number(adjustmentFactor) > 0xffffffff)) {
      return false;
    }
    const solution = value["solution"];
    if (solution !== undefined &&
        (typeof solution !== "string" || solution.length > MAX_SOLUTION_HEX_CHARS ||
         !/^(?:[0-9a-f]{2})+$/i.test(solution))) {
      return false;
    }
    const edges = value["edges"];
    if (edges !== undefined &&
        (typeof edges !== "string" || edges.length > MAX_EDGES_HEX_CHARS ||
         !/^(?:[0-9a-f]{8})+$/i.test(edges))) {
      return false;
    }
    return true;
  }

  /** @param {PoolConfig} pool @param {{job_id: unknown, job_token: unknown}} value */
  function matchingPoolJob(pool, value) {
    const job = pool.last_job;
    return job && job.job_token === value.job_token &&
      String(job.job_id) === String(value.job_id) ? job : null;
  }

  /** @param {WorkerEvent} msg */
  function handleResult(msg) {
    if (!isWorkerResult(msg.value)) {return invalidWorkerMessage();}
    const v = msg.value;
    const pool_id = Number(v.pool_id);
    const pool = poolAt(pool_id);
    if (!pool) {
      return invalidWorkerMessage();
    }
    const job = matchingPoolJob(pool, v);
    if (!job) {return;}
    const jobId = job.job_id;
    if (typeof jobId !== "string" && typeof jobId !== "number") {
      return invalidWorkerMessage();
    }
    const submit_mode = job.submit_mode;
    const nativeResult = job.submit_result === true && typeof v.hash === "string"
      ? v.hash.toLowerCase() : null;
    const send = (/** @type {UnknownRecord} */ body) =>
      p.pool_write(pool_id, {jsonrpc: "2.0", id: 3, ...body,
        ...(body["method"] === "mining.submit" && nativeResult !== null ? {result: nativeResult} : {})});
    const submit = (/** @type {unknown[] | UnknownRecord | null} */ params) => params
      ? send({method: "mining.submit", params}) : invalidWorkerMessage();
    // The native worker may stringify IDs; forward the stored pool ID after token/string matching.

    // PearlHash: relay each captured proof; distinct winning seeds can share a job/header.
    // Token matching above excludes stale work without suppressing subsequent proofs.
    if (submit_mode === "pearlhash") {
      if (typeof v.plain_proof !== "string" || typeof v.jackpot !== "string" ||
          typeof v.adjustment_factor !== "string") {return invalidWorkerMessage();}
      const rawProof = decodePearlProof(v.plain_proof);
      if (rawProof === null) {return invalidWorkerMessage();}
      const gzip = pool.pearlhash_proof_encodings?.includes("gzip") === true;
      let plainProof = v.plain_proof;
      if (gzip) {
        try {
          plainProof = zlib.gzipSync(rawProof).toString("base64");
        } catch {
          return invalidWorkerMessage();
        }
      }
      return send({method: "mining.submit", params: {
        job_id: jobId,
        plain_proof: plainProof,
        ...(gzip ? {proof_encoding: "gzip"} : {}),
        jackpot: v.jackpot,
        adjustment_factor: Number(v.adjustment_factor),
      }});
    }
    if (submit_mode === "erg") {
      return submit(submission.ergSubmitParams(pool, job, v));
    }
    if (submit_mode === "verthash") {
      return submit(submission.verthashSubmitParams(pool, job, v));
    }
    if (submit_mode === "echelon") {
      return submit(submission.nexaSubmitParams(pool, job, v));
    }
    // ZIP-301 Equihash: mining.submit [worker, job_id, time, nonce2, compactSize-prefixed solution].
    // Rebuild nonce2 from the native search counter and the job's fixed nonce prefix/tail.
    if (submit_mode === "zelhash") {
      if (typeof v.solution !== "string") {return invalidWorkerMessage();}
      return submit(submission.zelhashSubmitParams(pool, job, v, v.solution));
    }
    // Iron Fish custom OBJECT Stratum v3: submit {miningRequestId, randomness (8-byte BE nonce)}.
    if (submit_mode === "ironfish") {
      // Preserve the pool's request-ID type; the worker job_id is only used for matching.
      const message = {id: 2, method: "mining.submit",
        body: {miningRequestId: jobId, randomness: v.nonce},
        ...(nativeResult === null ? {} : {result: nativeResult})};
      return p.pool_write(pool_id, message);
    }
    // Kaspa-family submit: mining.submit [wallet.worker, job_id, nonce_hex].
    // The native returns the winning 8-byte nonce as 16-hex big-endian (nonce_to_hex %016PRIx64); the
    // pool parses it big-endian with the extranonce as the leading bytes, which is exactly this layout.
    if (submit_mode === "kaspa") {
      return send({method: "mining.submit", params: [pool.login, jobId, "0x" + v.nonce]});
    }
    if (submit_mode === "hoosat") {
      if (typeof v.hash !== "string") {return invalidWorkerMessage();}
      return send({method: "mining.submit", params: [pool.login, jobId, "0x" + v.nonce, v.hash]});
    }
    if (submit_mode === "xelis") {
      // Preserve an opaque numeric pool ID; the worker string is used only to match the current job.
      return send({method: "mining.submit", params: [xelisWorkerName(pool), jobId, v.nonce]});
    }
    if (submit_mode === "conflux") {
      const headerHash = resultHeaderHash(v, job);
      if (!headerHash) {return invalidWorkerMessage();}
      return send({method: "mining.submit", params: [pool.login, jobId,
        "0x" + v.nonce,
        "0x" + headerHash]});
    }
    if (submit_mode === "cortex") {
      const headerHash = resultHeaderHash(v, job);
      if (!headerHash || typeof v.edges !== "string" || v.edges.length !== 42 * 8) {
        return invalidWorkerMessage();
      }
      // C30 can find another proof before the prior response arrives. Distinct JSON-RPC IDs keep
      // those responses unambiguous even when a pool processes submissions concurrently.
      let pending = pool.pending_cortex_submit_ids;
      if (!pending) {
        pending = new Set();
        pool.pending_cortex_submit_ids = pending;
      }
      if (pending.size >= cortex_pending_limit) {
        return h.log_err("Too many pending Cortex submissions");
      }
      const requestId = nextCortexSubmitId();
      pending.add(requestId);
      return send({id: requestId, method: "ctxc_submitWork",
        params: ["0x" + v.nonce, "0x" + headerHash, "0x" + v.edges],
        worker: pool.worker || "mom"});
    }
    if (submit_mode === "beam") {
      // Beam JSON-RPC `solution`: TOP-LEVEL {id, nonce(16hex), output(208hex=104B)}. The native emits the
      // nonce as the big-endian hex of the LE-stored 8-byte blob nonce, so reverse it back to the raw
      // blob byte order the pool (and the nonceprefix) expect. The 104-byte solution is already raw.
      if (typeof v.nonce !== "string" || typeof v.solution !== "string" ||
          !/^[0-9a-f]{208}$/i.test(v.solution) ||
          !/^[0-9a-f]{16}$/i.test(v.nonce)) {
        return invalidWorkerMessage();
      }
      return send({
        id: jobId, method: "solution",
        nonce: submission.reverseHexBytes(v.nonce), output: v.solution,
      });
    }

    /** @type {SubmitParams} */
    const params = {job_id: jobId, nonce: v.nonce};
    params.id = v.worker_id;
    if (v.hash !== undefined) {params.result = v.hash;}
    if (submit_mode === "raven" || submit_mode === "eth") {
      if (typeof v.mix_hash !== "string") {return invalidWorkerMessage();}
      const headerHash = resultHeaderHash(v, job);
      if (!headerHash) {return invalidWorkerMessage();}
      return send({method: "mining.submit",
        params: [pool.login, jobId, "0x" + v.nonce, "0x" + headerHash,
          "0x" + v.mix_hash]});
    }
    if (v.mix_hash) {
      const headerHash = resultHeaderHash(v, job);
      params.mixhash = v.mix_hash;
      if (headerHash) {
        params.header_hash = headerHash;
      }
    }
    if (v.commitment) {
      params.commitment = v.commitment;
    }
    if (v.edges) {
      const proofsize = job.proofsize ?? 42;
      if (typeof proofsize !== "number" || !Number.isSafeInteger(proofsize) || proofsize <= 0 ||
          v.edges.length !== proofsize * 8) {
        return invalidWorkerMessage();
      }
      params.pow = h.edge_hex2arr(v.edges);
      // for proofsize == 42 (Tari C29) we return nonce hex as usual
      if (params.pow.length !== 42) {
        params.nonce = Number.parseInt(v.nonce, 16);
      }
    }
    if (!v.hash && !v.mix_hash && !v.commitment && !v.edges) {return invalidWorkerMessage();}
    return send({method: "submit", params});
  }

  /** @param {PoolConfig} pool */
  function xelisWorkerName(pool) {
    const login = pool.login;
    const separator = login.indexOf(".");
    return separator < 0 ? (pool.worker || "mom") : login.slice(separator + 1) || pool.worker || "mom";
  }

  /** @param {WorkerResult} value @param {PoolJob} job */
  function resultHeaderHash(value, job) {
    const headerHash = value.header_hash || job.header_hash || job.blob || job.blob_hex;
    const raw = submission.hexWithoutPrefix(headerHash);
    return raw.length >= 64 && /^[0-9a-f]+$/i.test(raw) ? raw.slice(0, 64) : "";
  }

  // store max last nonce for background pool job to resume it from there
  /** @param {WorkerEvent} msg */
  function handleLastNonce(msg) {
    const pool_id = msg.value["pool_id"];
    // Benchmark jobs have no pool; mining worker IDs are strings, unlike the active numeric ID.
    if (pool_id === "") {
      return;
    }
    const pool = poolAt(pool_id);
    const nonce = msg.value["nonce"];
    const job_id = msg.value["job_id"];
    const job_token = msg.value["job_token"];
    if (!pool || !isNonce(nonce) || !isJobId(job_id) ||
        typeof job_token !== "string" || job_token.length === 0 ||
        job_token.length > MAX_JOB_TOKEN_CHARS) {
      return invalidWorkerMessage();
    }
    if (Number(pool_id) === opt.pool_ids.active) {
      return;
    }
    const job = matchingPoolJob(pool, {job_id, job_token});
    if (!job) {return;}
    if (job.noncebytes !== undefined &&
        !submission.isValidNonce(nonce, job.noncebytes, job.xn, job.algo === "beamhash3")) {
      return invalidWorkerMessage();
    }
    if (isNewerNonce(job.nonce, nonce)) {
      job.nonce = nonce;
    }
  }

  /** @param {unknown} value @returns {value is string} */
  function isNonce(value) {
    return typeof value === "string" && /^[0-9a-f]{1,16}$/i.test(value);
  }

  /** @param {unknown} value @returns {value is string | number} */
  function isJobId(value) {
    return (typeof value === "string" && value.length > 0 && value.length <= 256) ||
      (typeof value === "number" && Number.isSafeInteger(value));
  }

  /** @param {unknown} prev_nonce @param {string} new_nonce */
  function isNewerNonce(prev_nonce, new_nonce) {
    if (typeof prev_nonce === "number" && Number.isSafeInteger(prev_nonce) && prev_nonce >= 0) {
      return BigInt(prev_nonce) < BigInt("0x" + new_nonce);
    }
    return !isNonce(prev_nonce) || BigInt("0x" + prev_nonce) < BigInt("0x" + new_nonce);
  }

  /** @param {string | null} algo */
  function isRandomXAlgo(algo) {
    return Boolean(algo && (algo.startsWith("rx/") || algo === "panthera"));
  }

  /** @param {{thread_id: number}} msg */
  function expectedTestThreads(msg) {
    const threads = h.get_dev_threads(opt.job.dev);
    if (isRandomXAlgo(opt.job.algo)) {
      if (!validThreadId(msg.thread_id, opt.job.dev)) {return 0;}
      let results = 0;
      for (const dev of opt.job.dev.split(",")) {
        results += h.get_dev_threads(dev) * h.get_dev_batch(dev);
      }
      return results;
    }
    return opt.job.algo === "c29" && test.result_hash_hex
      ? test.result_hash_hex.trim().split(/\s+/).length : threads;
  }

  /** @param {WorkerEvent} msg */
  function handleTestResult(msg) {
    const result = msg.value["result"];
    if (typeof result !== "string" || typeof test.result_hash_hex !== "string" ||
        !validThreadId(msg.thread_id, opt.job.dev)) {
      return invalidWorkerMessage();
    }
    const test_threads = expectedTestThreads(msg);
    test.result = (test.result ? test.result + " " : "") + result;
    if (++test.thread_tested < test_threads) {
      return;
    }

    const expectedResults = normalizeExpectedResults(opt.job.algo, test.result_hash_hex);
    if (!expectedResults.some(
      (expected) => matchesTestResult(opt.job.algo, test.result, expected)
    )) {
      fs.writeSync(2, "FAILED: " + test.result + " != " + test.result_hash_hex + " " + test_threads + "\n");
      return exit(1, false);
    }
    fs.writeSync(1, "PASSED\n");
    return exit(0, false);
  }

  function collectedHashrate() {
    const rates = Object.values(thread_hashrates);
    const total_hashrate = rates.reduce((total, rate) => total + rate, 0);
    const thread_hashrate_str = rates.map(h.formatHashrate).join(", ");
    return {total_hashrate, thread_hashrate_str};
  }

  /** @param {WorkerEvent} msg */
  function handleHashrate(msg) {
    const last_job = getLastJob();
    const rawRate = msg.value["hashrate"];
    const rate = typeof rawRate === "string" && rawRate.trim() ? Number(rawRate) : Number.NaN;
    if (!last_job || typeof last_job.dev !== "string" ||
        !validThreadId(msg.thread_id, last_job.dev) || !Number.isFinite(rate) || rate < 0) {
      return invalidWorkerMessage();
    }
    const job_identity = [last_job.algo, last_job.dev, last_job.backend, last_job.backend_request,
      last_job.job_token]
      .map((value) => typeof value === "string" ? value : "")
      .join("\u0000");
    if (hashrate_job_identity !== job_identity) {
      thread_hashrates = Object.create(null);
      hashrate_job_identity = job_identity;
    }
    thread_hashrates[msg.thread_id] = rate;
    if (Object.keys(thread_hashrates).length < h.get_dev_threads(last_job.dev)) {
      return;
    }

    const hashrate = collectedHashrate();
    const backend = last_job.backend_request === "auto"
      ? `auto[${last_job.backend}]` : last_job.backend;
    h.log("Algo " + last_job.algo + " (" + last_job.dev + ":" + backend + ") hashrate: " +
        h.formatHashrate(hashrate.total_hashrate) + " (" + hashrate.thread_hashrate_str + ")");
    thread_hashrates = Object.create(null);
    const callback = getAlgoParamsBenchCallback();
    if (callback) {
      return callback(hashrate.total_hashrate);
    }
    return undefined;
  }

  /** @param {unknown} thread_id @param {string} dev @returns {thread_id is number} */
  function validThreadId(thread_id, dev) {
    return typeof thread_id === "number" && Number.isSafeInteger(thread_id) &&
      thread_id >= 0 && thread_id < h.get_dev_threads(dev);
  }

  /** @param {WorkerEvent} msg */
  function handleWorkerError(msg) {
    if (msg.value["message"] === "Ignore duplicate job") {
      return;
    }
    const message = typeof msg.value["message"] === "string" ? msg.value["message"] : "Unknown error";
    h.log_err("Compute core error: " + JSON.stringify(message.slice(0, 1024)));
    if (test.result_hash_hex) {
      return exit(1);
    }
    const callback = getAlgoParamsBenchCallback();
    if (callback) {
      return callback(0);
    }
    if (msg.value["fatal"] === true) {
      // A fatal worker loss leaves normal mining without a worker to recover.
      return exit(1);
    }
    return undefined;
  }

  // handles messages sent to the master thread from worker threads
  /** @param {unknown} msg */
  function messageHandler(msg) {
    if (!h.is_worker_event(msg)) {return invalidWorkerMessage();}
    const type = msg.type;
    const handler = Object.hasOwn(masterMessageHandlers, type) ? masterMessageHandlers[type] : null;
    if (handler) {
      return handler(msg);
    }
    return h.log_err("Unknown compute core message type");
  }

  /** @type {Record<string, (message: WorkerEvent) => unknown>} */
  const masterMessageHandlers = {
    result:     handleResult,
    last_nonce: handleLastNonce,
    test:       handleTestResult,
    hashrate:   handleHashrate,
    error:      handleWorkerError,
  };

  return {
    expectedTestThreads,
    messageHandler,
    resetHashrates: () => {
      thread_hashrates = Object.create(null);
      hashrate_job_identity = null;
    },
  };
};
