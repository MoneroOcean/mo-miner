"use strict";

module.exports = ({
  fs, h, p, opt, submission, test, firstTruthyOr, normalizeExpectedResults,
  matchesTestResult, exit, getLastJob, getAlgoParamsBenchCallback,
}) => {

  let thread_hashrates = {};

  function handleResult(msg) {
    const v = msg.value;
    const pool = opt.pools[v.pool_id];
    const submit_mode = pool && pool.submit_mode;
    const send = (body) => p.pool_write(v.pool_id, { jsonrpc: "2.0", id: 3, ...body });

    // PearlHash: the worker already built the base64 PlainProof, and the native core emits at most one
    // solution per unit of work (job_id + header), so just relay it -- no JS-side per-job dedup
    // (which would mis-fire on HeroMiners' constant job_id).
    if (submit_mode === "pearlhash")
    {return send({ method: "mining.submit", params: { job_id: v.job_id, plain_proof: v.plain_proof } });}
    if (submit_mode === "erg")
    {return send({ method: "mining.submit", params: submission.ergSubmitParams(pool, v) });}
    // Equihash 125,4 (Flux/ZIP-301): mining.submit [worker, job_id, time(8hex), nonce2(hex), solution(hex)].
    // The native solver returns the 8-byte search counter (v.nonce, big-endian hex) + the 106-hex
    // 0x34-prefixed 52-byte solution (v.solution). Rebuild nonce2 = the 32-byte header nonce minus the
    // pool's nonce1 prefix, with the search counter written little-endian at its nonceoffset.
    if (submit_mode === "zelhash")
    {return send({ method: "mining.submit",
      params: [pool.login, v.job_id, submission.zelhashSubmitNtime(pool),
        submission.zelhashNonce2(pool, v.nonce), v.solution] });}
    // Iron Fish custom OBJECT stratum: submit {miningRequestId, randomness (8-byte BE nonce), graffiti}.
    if (submit_mode === "ironfish")
    {return p.pool_write(v.pool_id, { id: 2, method: "mining.submit",
      body: { miningRequestId: v.job_id, randomness: v.nonce, graffiti: "00".repeat(32) } });}
    // Kaspa-family submit: mining.submit [wallet.worker, job_id, nonce_hex].
    // The native returns the winning 8-byte nonce as 16-hex big-endian (nonce_to_hex %016PRIx64); the
    // pool parses it big-endian with the extranonce as the leading bytes, which is exactly this layout.
    if (submit_mode === "kaspa")
    {return send({ method: "mining.submit", params: [pool.login, v.job_id, "0x" + v.nonce] });}
    if (submit_mode === "beam") {
    // Beam JSON-RPC `solution`: TOP-LEVEL {id, nonce(16hex), output(208hex=104B)}. The native emits the
    // nonce as the big-endian hex of the LE-stored 8-byte blob nonce, so reverse it back to the raw
    // blob byte order the pool (and the nonceprefix) expect. The 104-byte solution is already raw.
      return send({
        id: v.job_id, method: "solution",
        nonce: submission.reverseHexBytes(v.nonce), output: v.solution,
      });
    }

    const params = { id: v.worker_id, job_id: v.job_id, nonce: v.nonce, result: v.hash };
    if (v.mix_hash) {
      const headerHash = resultHeaderHash(msg, pool);
      if (submit_mode === "ethproxy")
      {return send({ method: "eth_submitWork",
        params: ["0x" + v.nonce, "0x" + headerHash.slice(0, 64), "0x" + v.mix_hash] });}
      if (submit_mode === "raven" || submit_mode === "eth")
      {return send({ method: "mining.submit",
        params: [pool.login, v.job_id, "0x" + v.nonce, "0x" + headerHash.slice(0, 64), "0x" + v.mix_hash] });}
      params.mixhash = v.mix_hash;
      if (headerHash) {params.header_hash = headerHash.slice(0, 64);}
    }
    if (v.commitment) {params.commitment = v.commitment;}
    if (v.edges) {
      params.pow = h.edge_hex2arr(v.edges);
      // for proofsize == 42 (Tari C29) we return nonce hex as usual
      if (params.pow.length !== 42) {params.nonce = Number.parseInt(params.nonce, 16);}
    }
    send({ method: "submit", params: params });
  }

  function resultHeaderHash(msg, pool) {
    if (msg.value.header_hash) {return msg.value.header_hash;}
    if (!pool || !pool.last_job) {return "";}
    if (pool.last_job.job_id && msg.value.job_id && pool.last_job.job_id !== msg.value.job_id) {return "";}
    return firstTruthyOr("", pool.last_job.header_hash, pool.last_job.blob, pool.last_job.blob_hex);
  }

  // store max last nonce for background pool job to resume it from there
  function handleLastNonce(msg) {
    const pool_id = msg.value.pool_id;
    // pool_id can be "" for benchmark jobs. can not use === here since
    // opt.pool_ids.active is integer here
    if (!shouldStoreLastNonce(pool_id)) {return;}
    const prev_nonce = opt.pools[pool_id].last_job.nonce;
    const new_nonce  = msg.value.nonce;
    if (isNewerNonce(prev_nonce, new_nonce))
    {opt.pools[pool_id].last_job.nonce = new_nonce;}
  }

  function shouldStoreLastNonce(pool_id) {
  // eslint-disable-next-line eqeqeq -- pool_id is "" | number; loose != is intentional coercion
    return pool_id !== "" && pool_id != opt.pool_ids.active &&
         opt.pools[pool_id].last_job;
  }

  function isNewerNonce(prev_nonce, new_nonce) {
    return !prev_nonce || BigInt("0x" + prev_nonce) < BigInt("0x" + new_nonce);
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
