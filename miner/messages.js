"use strict";

const zlib = require("node:zlib");

const MAX_PEARL_PROOF_BYTES = 8 * 1024 * 1024;
const MAX_PEARL_PROOF_BASE64 = Math.ceil(MAX_PEARL_PROOF_BYTES / 3) * 4;
const MAX_JOB_TOKEN_CHARS = 256;
const MAX_WORKER_ID_CHARS = 4096;
const MAX_HEADER_HASH_HEX_CHARS = 256 * 2;
// CompactSize plus the largest supported Equihash proof: (192,7) at 403 bytes.
const MAX_SOLUTION_HEX_CHARS = 403 * 2;
const MAX_EDGES_HEX_CHARS = 42 * 8;


module.exports = ({
  fs, h, p, opt, submission, test, firstTruthyOr, normalizeExpectedResults,
  matchesTestResult, exit, getLastJob, getAlgoParamsBenchCallback,
}) => {

  let thread_hashrates = {};

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

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function poolAt(pool_id) {
    const index = typeof pool_id === "number" ? pool_id :
      typeof pool_id === "string" && /^\d+$/.test(pool_id) ? Number(pool_id) : -1;
    return Number.isSafeInteger(index) && Object.hasOwn(opt.pools, index)
      ? opt.pools[index] || null : null;
  }

  function invalidWorkerMessage() {
    return h.log_err("Invalid compute core message");
  }

  function isWorkerResult(value) {
    if (!isObject(value)) {return false;}
    const jobId = value["job_id"];
    if (typeof value["pool_id"] !== "string" || !/^\d+$/.test(value["pool_id"]) ||
        !((typeof jobId === "string" && jobId.length > 0 && jobId.length <= 256) ||
          (typeof jobId === "number" && Number.isSafeInteger(jobId))) ||
        typeof value["job_token"] !== "string" || value["job_token"].length === 0 ||
        value["job_token"].length > MAX_JOB_TOKEN_CHARS ||
        typeof value["worker_id"] !== "string" || value["worker_id"].length > MAX_WORKER_ID_CHARS ||
        typeof value["nonce"] !== "string" ||
        !/^[0-9a-f]{1,16}$/i.test(value["nonce"])) {
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

  function matchingPoolJob(pool, value) {
    const job = pool.last_job;
    return job && job.job_token === value.job_token &&
      String(job.job_id) === String(value.job_id) ? job : null;
  }

  function decodePearlProof(proof) {
    if (proof.length > MAX_PEARL_PROOF_BASE64) {return null;}
    const raw = Buffer.from(proof, "base64");
    return raw.length > 0 && raw.length <= MAX_PEARL_PROOF_BYTES &&
      raw.toString("base64") === proof ? raw : null;
  }

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
    if (submit_mode === "ethproxy" && typeof v.mix_hash === "string") {
      const headerHash = resultHeaderHash(v, job);
      if (!headerHash) {return invalidWorkerMessage();}
      return send({method: "eth_submitWork",
        params: ["0x" + v.nonce, "0x" + headerHash, "0x" + v.mix_hash]});
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

  function xelisWorkerName(pool) {
    const login = pool.login;
    const separator = login.indexOf(".");
    return separator < 0 ? (pool.worker || "mom") : login.slice(separator + 1) || pool.worker || "mom";
  }

  function resultHeaderHash(value, job) {
    const headerHash = value.header_hash || job.header_hash || job.blob || job.blob_hex;
    const raw = submission.hexWithoutPrefix(headerHash);
    return raw.length >= 64 && /^[0-9a-f]+$/i.test(raw) ? raw.slice(0, 64) : "";
  }

  // store max last nonce for background pool job to resume it from there
  function handleLastNonce(msg) {
    const pool_id = msg.value["pool_id"];
    // pool_id can be "" for benchmark jobs. can not use === here since
    // opt.pool_ids.active is integer here
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
    if (!shouldStoreLastNonce(pool_id)) {
      return;
    }
    const job = matchingPoolJob(pool, {job_id, job_token});
    if (!job) {return;}
    if (job.noncebytes !== undefined &&
        !submission.isValidNonce(nonce, job.noncebytes, job.xn, job.algo === "beamhash3")) {
      return invalidWorkerMessage();
    }
    const prev_nonce = job.nonce;
    const new_nonce = nonce;
    if (isNewerNonce(prev_nonce, new_nonce)) {
      job.nonce = new_nonce;
    }
  }

  function shouldStoreLastNonce(pool_id) {
    return Number(pool_id) !== opt.pool_ids.active;
  }

  function isNonce(value) {
    return typeof value === "string" && /^[0-9a-f]{1,16}$/i.test(value);
  }

  function isJobId(value) {
    return (typeof value === "string" && value.length > 0 && value.length <= 256) ||
      (typeof value === "number" && Number.isSafeInteger(value));
  }

  function isNewerNonce(prev_nonce, new_nonce) {
    if (typeof prev_nonce === "number" && Number.isSafeInteger(prev_nonce) && prev_nonce >= 0) {
      return BigInt(prev_nonce) < BigInt("0x" + new_nonce);
    }
    return !isNonce(prev_nonce) || BigInt("0x" + prev_nonce) < BigInt("0x" + new_nonce);
  }

  function isRandomXAlgo(algo) {
    return algo.startsWith("rx/") || algo === "panthera";
  }

  function expectedTestThreads(msg) {
    const threads = h.get_dev_threads(opt.job.dev);
    if (isRandomXAlgo(opt.job.algo)) {
      const batch = h.get_dev_batch(h.get_thread_dev(msg.thread_id, opt.job.dev));
      return batch * threads;
    }
    return opt.job.algo === "c29" ? test.result_hash_hex.trim().split(/\s+/).length : threads;
  }

  function handleTestResult(msg) {
    const test_threads = expectedTestThreads(msg);
    test.result = (test.result ? test.result + " " : "") + msg.value.result;
    if (++test.thread_tested < test_threads) {return;}

    const expectedResults = normalizeExpectedResults(opt.job.algo, test.result_hash_hex);
    if (!expectedResults.some(
      (expected) => matchesTestResult(opt.job.algo, test.result, expected)
    )) {
      fs.writeSync(2, "FAILED: " + test.result + " != " + test.result_hash_hex + " " + test_threads + "\n");
      return exit(1);
    }
    fs.writeSync(1, "PASSED\n");
    return exit(0);
  }

  function collectedHashrate() {
    const rates = Object.values(thread_hashrates).map(Number.parseFloat);
    const total_hashrate = rates.reduce((total, rate) => total + rate, 0);
    const thread_hashrate_str = rates.map(h.formatHashrate).join(", ");
    return { total_hashrate, thread_hashrate_str };
  }

  function handleHashrate(msg) {
    const last_job = getLastJob();
    thread_hashrates[msg.thread_id] = msg.value.hashrate;
    if (Object.keys(thread_hashrates).length < h.get_dev_threads(last_job.dev)) {return;}

    const hashrate = collectedHashrate();
    const backend = last_job.backend_request === "auto"
      ? `auto[${last_job.backend}]` : last_job.backend;
    h.log("Algo " + last_job.algo + " (" + last_job.dev + ":" + backend + ") hashrate: " +
        h.formatHashrate(hashrate.total_hashrate) + " (" + hashrate.thread_hashrate_str + ")");
    thread_hashrates = {};
    const callback = getAlgoParamsBenchCallback();
    if (callback) {return callback(hashrate.total_hashrate);}
  }

  function handleWorkerError(msg) {
    if (msg.value.message === "Ignore duplicate job") {return;}
    h.log_err("Compute core error: " + JSON.stringify(msg.value));
    if (test.result_hash_hex) {exit(1);} // exit with error
    const callback = getAlgoParamsBenchCallback();
    if (callback) {return callback(0);}
  }

  // handles messages sent to the master thread from worker threads
  function messageHandler(msg) {
    const handler = masterMessageHandlers[msg.type];
    if (handler) {return handler(msg);}
    return h.log_err("Unknown master thread message: " + JSON.stringify(msg));
  }

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
    resetHashrates: () => { thread_hashrates = {}; },
  };
};
