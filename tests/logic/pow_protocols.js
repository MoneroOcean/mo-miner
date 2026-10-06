"use strict";

const s = require("./support");
const {test, pool, loadMinerWithStubs, withMockPool} = s;
const zlib = require("node:zlib");
/** @type {typeof import("node:assert/strict")} */
const assert = s.assert;

test("KawPow pool jobs append the nonce field to a header hash", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const headerHash = "00".repeat(32);

  setJob({
    algo: "kawpow",
    blob: headerHash,
    difficulty: 1,
    id: "worker",
    job_id: "job",
    submit_mode: "raven",
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.ok(jobMessage);
  assert.ok(jobMessage.job);
  assert.equal(jobMessage.job.blob_hex, headerHash + "0000000000000000");
  assert.equal(jobMessage.job.noncebytes, 8);
  assert.equal(jobMessage.job.nonceoffset, 32);
  assert.equal(jobMessage.job.worker_id, "worker");
  assert.equal(Object.hasOwn(jobMessage.job, "submit_mode"), false);
});

test("KawPow stratum jobs can use pool login as worker id", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const headerHash = "00".repeat(32);

  setJob({
    algo: "kawpow",
    blob: headerHash,
    difficulty: 1,
    job_id: "job",
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.ok(jobMessage);
  assert.ok(jobMessage.job);
  assert.equal(jobMessage.job.worker_id, "user");
});

test("malformed pool jobs fail before worker replacement", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const initialMessages = miner.sentMessages.length;

  assert.throws(() => setJob({
    algo: "cn/0", blob: "00".repeat(43), job_id: "missing-difficulty",
  }), /Invalid cn\/0 job difficulty/);
  assert.throws(() => setJob({
    algo: "cn/0", difficulty: 1, job_id: "missing-blob",
  }), /Invalid cn\/0 job blob/);
  assert.throws(() => setJob({
    algo: "cn/0", blob: "00".repeat(64 * 1024 + 1), difficulty: 1, job_id: "oversized-blob",
  }), /Invalid cn\/0 job blob/);
  assert.throws(() => setJob({
    algo: "beamhash3", difficulty: 1, header_hash: "00", job_id: "short-prework",
  }), /Invalid BeamHash III prework/);
  const numericMaskJob = {
    algo: "cn/0", blob_hex: "00".repeat(43), difficulty: 1, job_id: "numeric-mask",
  };
  Reflect.set(numericMaskJob, "nicehash_mask", 1);
  assert.throws(() => setJob(numericMaskJob), /Invalid cn\/0 nonce mask/);
  assert.equal(miner.sentMessages.length, initialMessages);
});

test("pool jobs preserve an explicit zero nonce", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();

  setJob({
    algo: "cn/0", blob_hex: "00".repeat(43), difficulty: 1, job_id: "zero-nonce", nonce: 0,
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.ok(jobMessage?.job);
  assert.equal(jobMessage.job.nonce, 0);
});

test("KawPow pool jobs preserve provided nonce template", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const headerHash = "00".repeat(32);
  const extraNonce = "00000000000081ff";

  setJob({
    algo: "kawpow",
    blob: headerHash + extraNonce,
    header_hash: headerHash,
    nonce: "ff81000000000000",
    nicehash_mask: "ffff000000000000",
    difficulty: 1,
    id: "worker",
    job_id: "job",
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.ok(jobMessage);
  assert.ok(jobMessage.job);
  assert.equal(jobMessage.job.blob_hex, headerHash + extraNonce);
  assert.equal(jobMessage.job.header_hash, headerHash);
  assert.equal(jobMessage.job.nonce, "ff81000000000000");
  assert.equal(jobMessage.job.nicehash_mask, "ffff000000000000");
});

test("KawPow submit uses the header hash carried by the worker result", async () => {
  const miner = await loadMinerWithStubs();
  const oldHeaderHash = "11".repeat(32);
  const newHeaderHash = "22".repeat(32);
  miner.global.opt.pools[0].last_job = {
    job_id: "old",
    job_token: "token",
    submit_mode: "raven",
    header_hash: newHeaderHash,
  };

  miner.messageHandler({
    thread_id: 0,
    type: "result",
    value: {
      pool_id: "0",
      worker_id: "worker",
      job_id: "old",
      job_token: "token",
      nonce: "ff81000000000001",
      hash: "00".repeat(32),
      mix_hash: "33".repeat(32),
      header_hash: "0x" + oldHeaderHash + "aa",
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.equal(JSON.stringify(write.json.params), JSON.stringify([
    "user",
    "old",
    "0xff81000000000001",
    "0x" + oldHeaderHash,
    "0x" + "33".repeat(32),
  ]));
});

test("EVRProgPoW Raven completes login, target, job, and submit", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const headerHash = "00".repeat(32);
  const seedHash = "11".repeat(32);
  const shareTarget = "00000000ffff" + "00".repeat(26);

  await withMockPool({
    pool: {protocol: "raven", login: "RVNwallet.rig01"},
    opt: {job: {algo: "evrprogpow"}},
  }, async ({socket, writes, poolConfig}) => {
    miner.global.opt.pools[0] = poolConfig;
    pool.connect_pool_throttle(0, setJob);
    socket.emit("connect");
    assert.equal(writes[0]?.method, "mining.subscribe");

    socket.emit("data", Buffer.from([
      {jsonrpc: "2.0", id: 1, error: null, result: [["mining.notify", "1"], "080c", 6]},
      {jsonrpc: "2.0", id: 2, error: null, result: true},
      {method: "mining.set_target", params: [shareTarget]},
      {
        method: "mining.notify",
        params: ["203d", headerHash, seedHash, "", true, 4390582, "1b01e5f2"],
        algo: "evrprogpow",
      },
    ].map((message) => JSON.stringify(message)).join("\n") + "\n"));

    assert.equal(poolConfig.logged_in, true);
    assert.deepEqual(writes[1]?.params, ["RVNwallet.rig01", "x"]);
    const job = poolConfig.last_job;
    assert.ok(job);
    assert.equal(job.algo, "evrprogpow");
    assert.equal(job.target, shareTarget);
    assert.equal(job["submit_mode"], "raven");

    miner.messageHandler({thread_id: 0, type: "result", value: {
      pool_id: "0", worker_id: "worker", job_id: "203d", job_token: job["job_token"],
      nonce: "080c000000000001", hash: "00".repeat(32), mix_hash: "33".repeat(32),
    }});

    const write = miner.poolWrites[0];
    assert.ok(write);
    assert.deepEqual(write.json.params, [
      "RVNwallet.rig01", "203d", "0x080c000000000001", "0x" + headerHash,
      "0x" + "33".repeat(32),
    ]);
  });
});

test("Etchash submit uses Eth mining.submit format", async () => {
  const miner = await loadMinerWithStubs();
  const headerHash = "22".repeat(32);
  miner.global.opt.pools[0].last_job = {job_id: "203d", job_token: "token", submit_mode: "eth"};

  miner.messageHandler({
    thread_id: 0,
    type: "result",
    value: {
      pool_id: "0",
      worker_id: "worker",
      job_id: "203d",
      job_token: "token",
      nonce: "080c000000000001",
      hash: "00".repeat(32),
      mix_hash: "33".repeat(32),
      header_hash: "0X" + headerHash + "bb",
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.equal(JSON.stringify(write.json.params), JSON.stringify([
    "user",
    "203d",
    "0x080c000000000001",
    "0x" + headerHash,
    "0x" + "33".repeat(32),
  ]));
});

test("Octopus submit uses Conflux mining.submit format", async () => {
  const miner = await loadMinerWithStubs();
  const headerHash = "22".repeat(32);
  miner.global.opt.pools[0].login = "cfx:wallet.rig";
  miner.global.opt.pools[0].last_job = {
    submit_mode: "conflux",
    job_id: "job1", job_token: "token", header_hash: headerHash,
  };

  miner.messageHandler({
    thread_id: 0,
    type: "result",
    value: {
      pool_id: "0", worker_id: "worker", job_id: "job1", job_token: "token", nonce: "0000000000000001",
      hash: "00".repeat(32), header_hash: "0x" + headerHash + "cc",
    },
  });

  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.equal(JSON.stringify(write.json.params), JSON.stringify([
    "cfx:wallet.rig", "job1", "0x0000000000000001", "0x" + headerHash,
  ]));
});

test("generic submit uses the normalized raw pool header hash", async () => {
  const miner = await loadMinerWithStubs();
  const headerHash = "44".repeat(32);
  miner.global.opt.pools[0].last_job = {
    job_id: "job", job_token: "token", blob: "0x" + headerHash + "aa",
  };

  miner.messageHandler({thread_id: 0, type: "result", value: {
    pool_id: "0", worker_id: "worker", job_id: "job", job_token: "token", nonce: "00000001",
    hash: "00".repeat(32), mix_hash: "55".repeat(32),
  }});

  const write = miner.poolWrites[0];
  assert.ok(write);
  const params = /** @type {{header_hash?: string}} */ (write.json.params);
  assert.equal(params.header_hash, headerHash);
});

test("PearlHash pools default to V3 and advertise supported proof encodings", async () => {
  for (const version of [undefined, 1, 2, 3, "3", 4, 0, null, 2.5]) {
    await withMockPool({pool: {protocol: "pearlhash", logged_in: true, use_subscribe: true},
      opt: {job: {algo: "pearlhash"}}}, async ({socket, poolConfig}) => {
      /** @type {PoolJob[]} */
      const jobs = [];
      pool.connect_pool_throttle(0, (job) => {
        jobs.push(job);
        return s.completeMiningJob(job);
      });
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "mining.notify", params: {
          job_id: "pearl-job", header: "00".repeat(76), target: "1", height: 1,
          proof_encodings: ["none", "gzip"],
          ...(version === undefined ? {} : {cert_version: version}),
        },
      }) + "\n"));
      const expected = version === undefined ? 3 : Number(version);
      assert.deepEqual(jobs.map((job) => job.pearlhash_cert_version),
        expected === 3 ? [3] : []);
      assert.deepEqual(poolConfig["pearlhash_proof_encodings"],
        expected === 3 ? ["none", "gzip"] : undefined);
    });
  }
});

test("PearlHash login-dialect jobs preserve a supplied target", async () => {
  await withMockPool({pool: {protocol: "pearlhash", logged_in: true, use_subscribe: false},
    opt: {job: {algo: "pearlhash"}}}, async ({socket}) => {
    /** @type {PoolJob[]} */
    const jobs = [];
    pool.connect_pool_throttle(0, (job) => {
      jobs.push(job);
      return s.completeMiningJob(job);
    });
    const target = "01".repeat(32);
    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.notify", params: {
        job_id: "451610", header: "00".repeat(76), target, cert_version: 3,
      },
    }) + "\n"));
    assert.equal(jobs.length, 1);
    assert.equal(jobs[0]?.pearlhash_base_target, target);
    assert.equal(jobs[0]?.difficulty, undefined);
  });
});

test("PearlHash accepts proxy job updates after out-of-order handshake replies", async () => {
  await withMockPool({
    pool: {protocol: "pearlhash", use_subscribe: true},
    opt: {job: {algo: "pearlhash"}},
  }, async ({socket, switched, poolConfig}) => {
    /** @type {PoolJob[]} */
    const jobs = [];
    pool.connect_pool_throttle(0, (job) => {
      jobs.push(job);
      return s.completeMiningJob(job);
    });
    socket.emit("connect");
    const target = "01".repeat(32);
    /** @param {string} job_id */
    const params = (job_id) => ({
      job_id, header: "00".repeat(76), target, cert_version: 3,
    });
    socket.emit("data", Buffer.from([
      {id: 2, jsonrpc: "2.0", error: null, result: true},
      {method: "mining.notify", params: params("first"), id: null, jsonrpc: "2.0"},
      {id: 1, jsonrpc: "2.0", error: null,
        result: [["mining.notify", "session", "EthereumStratum/1.0.0"], "ff81ee", 5]},
      {method: "job", params: {...params("second"), id: "session"}},
    ].map((message) => JSON.stringify(message)).join("\n") + "\n"));

    assert.equal(poolConfig.logged_in, true);
    assert.deepEqual(jobs.map((job) => job.job_id), ["first", "second"]);
    assert.equal(switched(), false);
    assert.notEqual(socket.destroyed, true);
  });
});

test("PearlHash submit uses its canonical submit mode", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].last_job = {job_id: "job1", job_token: "token", submit_mode: "pearlhash"};
  const proof = Buffer.from("proof").toString("base64");

  miner.messageHandler({
    thread_id: 0,
    type: "result",
    value: {
      pool_id: "0",
      job_id: "job1",
      job_token: "token",
      worker_id: "worker",
      nonce: "0000000000000001",
      plain_proof: proof,
      jackpot: "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
      adjustment_factor: "524288",
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.deepEqual(JSON.parse(JSON.stringify(write.json)), {
    jsonrpc: "2.0",
    id: 3,
    method: "mining.submit",
    params: {
      job_id: "job1",
      plain_proof: proof,
      jackpot: "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
      adjustment_factor: 524288,
    },
  });
});

test("PearlHash prefers advertised gzip and leaves unadvertised proofs unchanged", async () => {
  const proof = Buffer.from("Pearl proof bytes ".repeat(256)).toString("base64");
  const result = {
    pool_id: "0",
    job_id: "job1",
    job_token: "token",
    worker_id: "worker",
    nonce: "0000000000000001",
    plain_proof: proof,
    jackpot: "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
    adjustment_factor: "524288",
  };

  const mo = await loadMinerWithStubs();
  mo.global.opt.pools[0]["pearlhash_proof_encodings"] = ["none", "gzip"];
  mo.global.opt.pools[0].last_job = {
    job_id: "job1", job_token: "token", submit_mode: "pearlhash",
  };
  mo.messageHandler({thread_id: 0, type: "result", value: result});
  mo.messageHandler({thread_id: 0, type: "result", value: {...result}});
  const first = /** @type {Record<string, unknown>} */ (mo.poolWrites[0]?.json.params);
  const second = /** @type {Record<string, unknown>} */ (mo.poolWrites[1]?.json.params);
  const compressedProof = first["plain_proof"];
  assert.equal(first["proof_encoding"], "gzip");
  assert.equal(compressedProof, second["plain_proof"]);
  assert.ok(typeof compressedProof === "string");
  assert.equal(zlib.gunzipSync(Buffer.from(compressedProof, "base64")).toString("base64"), proof);

  const external = await loadMinerWithStubs();
  external.global.opt.pools[0]["pearlhash_proof_encodings"] = ["none"];
  external.global.opt.pools[0].last_job = {
    job_id: "job1", job_token: "token", submit_mode: "pearlhash",
  };
  external.messageHandler({thread_id: 0, type: "result", value: result});
  const externalParams = /** @type {Record<string, unknown>} */ (external.poolWrites[0]?.json.params);
  assert.equal(externalParams["plain_proof"], proof);
  assert.equal(Object.hasOwn(externalParams, "proof_encoding"), false);
});

test("PearlHash claims are atomic, native-endian, and stable across retries", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].last_job = {
    job_id: "job1", job_token: "token", submit_mode: "pearlhash",
  };
  const base = {
    pool_id: "0", job_id: "job1", job_token: "token", worker_id: "worker",
    nonce: "0000000000000001", plain_proof: Buffer.from("proof").toString("base64"),
    jackpot: "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
    adjustment_factor: "65536",
  };
  for (let retry = 0; retry < 2; retry++) {
    miner.messageHandler({thread_id: 0, type: "result", value: {...base}});
  }
  assert.equal(miner.poolWrites.length, 2);
  const first = JSON.stringify(miner.poolWrites[0]?.json);
  const second = JSON.stringify(miner.poolWrites[1]?.json);
  assert.equal(second, first);
  assert.match(first, /"jackpot":"0001020304050607/);
  assert.doesNotMatch(first, /"jackpot":"1f1e1d1c1b1a1918/);
  assert.match(first, /"adjustment_factor":65536/);

  const next = {...base, nonce: "0000000000000002",
    plain_proof: Buffer.from("next proof").toString("base64"), jackpot: "12".repeat(32)};
  miner.messageHandler({thread_id: 0, type: "result", value: next});
  assert.equal(miner.poolWrites.length, 3);
  const nextParams = /** @type {Record<string, unknown>} */ (miner.poolWrites[2]?.json.params);
  assert.equal(nextParams["plain_proof"], next.plain_proof);
  assert.equal(nextParams["jackpot"], next.jackpot);
  miner.messageHandler({thread_id: 0, type: "result", value: {...next, job_token: "stale"}});
  assert.equal(miner.poolWrites.length, 3);

  for (const value of [
    {...base, jackpot: undefined},
    {...base, adjustment_factor: undefined},
    {...base, jackpot: base.jackpot.toUpperCase()},
    {...base, adjustment_factor: "0"},
    {...base, adjustment_factor: "4294967296"},
    {...base, plain_proof: "not canonical base64"},
    {...base, plain_proof: Buffer.alloc(8 * 1024 * 1024 + 1).toString("base64")},
    {...base, job_token: "stale"},
  ]) {
    miner.messageHandler({thread_id: 0, type: "result", value});
  }
  assert.equal(miner.poolWrites.length, 3);
});

test("PearlHash claim fields cannot change another algorithm's submission", async () => {
  const result = {
    pool_id: "0", worker_id: "worker", job_id: "job", job_token: "token",
    nonce: "0000000000000001", hash: "00".repeat(32), mix_hash: "11".repeat(32),
    header_hash: "22".repeat(32),
  };
  const serialize = async (extra = {}) => {
    const miner = await loadMinerWithStubs();
    miner.global.opt.pools[0].last_job = {
      job_id: "job", job_token: "token", submit_mode: "eth",
    };
    miner.messageHandler({thread_id: 0, type: "result", value: {...result, ...extra}});
    return JSON.stringify(miner.poolWrites[0]?.json);
  };
  const baseline = await serialize();
  const withClaim = await serialize({
    jackpot: "0123456789abcdef".repeat(4), adjustment_factor: "524288",
  });
  assert.equal(withClaim, baseline);
});

test("Autolykos2 submit uses Ergo mining.submit format", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].last_job = {
    job_id: "203d", job_token: "token", submit_mode: "erg",
    extra_nonce: "080c", extra_nonce2_size: 6, ntime: "00000002",
  };

  miner.messageHandler({
    thread_id: 0,
    type: "result",
    value: {
      pool_id: "0",
      worker_id: "worker",
      job_id: "203d",
      job_token: "token",
      nonce: "080c000000000001",
      hash: "00".repeat(32),
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.equal(JSON.stringify(write.json.params), JSON.stringify([
    "user",
    "203d",
    "000000000001",
    "00000002",
    "080c000000000001",
  ]));
});

test("a reused pool job ID resets the previous submission dialect", async () => {
  const miner = await loadMinerWithStubs();
  await withMockPool({pool: {protocol: "eth", logged_in: true}, opt: {job: {algo: "etchash"}}},
    async ({socket, poolConfig}) => {
      miner.global.opt.pools[0] = poolConfig;
      let token = 0;
      pool.connect_pool_throttle(0, (job) => {
        job.job_token = String(++token);
        return s.completeMiningJob(job);
      });
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "mining.notify", params: ["reused", "11".repeat(32), "22".repeat(32), true],
      }) + "\n"));
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "job", params: {
          algo: "cn/0", job_id: "reused", blob: "00".repeat(43), target: "ff".repeat(8),
        },
      }) + "\n"));
      poolConfig.inferred_protocol = "eth";
      const value = {
        pool_id: "0", worker_id: "worker", job_id: "reused", nonce: "00000001",
        hash: "33".repeat(32),
      };
      miner.messageHandler({type: "result", thread_id: 0, value: {...value, job_token: "1"}});
      assert.equal(miner.poolWrites.length, 0);
      miner.messageHandler({type: "result", thread_id: 0, value: {...value, job_token: "2"}});
      assert.equal(miner.poolWrites.length, 1);
      assert.deepEqual(miner.poolWrites[0]?.json, {
        jsonrpc: "2.0", id: 3, method: "submit",
        params: {job_id: "reused", nonce: "00000001", id: "worker", result: "33".repeat(32)},
      });
      assert.equal(poolConfig.last_job?.["submit_mode"], null);
    });
});

test("Ergo submissions retain their accepted job's nonce metadata", async () => {
  for (const {extraNonce, size, nonce2} of [
    {extraNonce: "", size: undefined, nonce2: "123456789abcdef0"},
    {extraNonce: "080c", size: 6, nonce2: "56789abcdef0"},
    {extraNonce: "0123456789abcdef", size: 0, nonce2: ""},
  ]) {
    const miner = await loadMinerWithStubs();
    await withMockPool({
      pool: {protocol: "erg", logged_in: true, extra_nonce: extraNonce,
        ...(size === undefined ? {} : {extra_nonce2_size: size})},
      opt: {job: {algo: "autolykos2"}},
    }, async ({socket, poolConfig}) => {
      miner.global.opt.pools[0] = poolConfig;
      pool.connect_pool_throttle(0, (job) => {
        job.job_token = "erg-token";
        return s.completeMiningJob(job);
      });
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "mining.notify",
        params: ["erg-job", 1, "11".repeat(32), "", "", 0, "1", "00000002"],
      }) + "\n"));
      const job = poolConfig.last_job;
      assert.ok(job);
      const {blob, nonce} = job;
      miner.global.opt.pools.push(s.mockPoolConfig());
      miner.global.opt.pool_ids.active = 1;
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "mining.set_extranonce", params: ["ffff", 2],
      }) + "\n"));
      poolConfig.inferred_protocol = "echelon";
      miner.messageHandler({type: "result", thread_id: 0, value: {
        pool_id: "0", worker_id: "worker", job_id: "erg-job", job_token: "erg-token",
        nonce: "123456789abcdef0", hash: "00".repeat(32),
      }});
      assert.equal(miner.poolWrites.length, 1);
      assert.deepEqual(miner.poolWrites[0]?.json.params,
        ["wallet", "erg-job", nonce2, "00000002", "123456789abcdef0"]);
      assert.equal(job.blob, blob);
      assert.equal(job.nonce, nonce);
    });
  }
});

for (const [decimalTarget, hexTarget] of [
  ["16", "10".padStart(64, "0")],
  ["26", "1a".padStart(64, "0")],
  [((1n << 256n) - 1n).toString(), "ff".repeat(32)],
]) {
  test(`Ergo decimal target ${decimalTarget} reaches the worker as hexadecimal`, async () => {
    const miner = await loadMinerWithStubs();
    const setJob = miner.getSetJob();
    await withMockPool({
      pool: {protocol: "erg", logged_in: true}, opt: {job: {algo: "autolykos2"}},
    }, async ({socket, poolConfig}) => {
      miner.global.opt.pools[0] = poolConfig;
      pool.connect_pool_throttle(0, setJob);
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "mining.notify",
        params: ["erg-target", 1, "11".repeat(32), "", "", 0, decimalTarget],
      }) + "\n"));
      assert.equal(poolConfig.last_job?.target, hexTarget);
      const jobs = miner.sentMessages.filter((message) => message.type === "job");
      assert.equal(jobs.length, 1);
      assert.equal(jobs[0]?.job?.target, hexTarget);
    });
  });
}

for (const {name, fields, expectedTarget} of [
  {name: "digit-only compact target", fields: {target: "10000000"},
    expectedTarget: s.helper.ethDiff2Target(s.helper.target2diff("10000000"))},
  {name: "compact target with hex letters", fields: {target: "c6100000"},
    expectedTarget: s.helper.ethDiff2Target(s.helper.target2diff("c6100000"))},
  {name: "explicit difficulty before compact target", fields: {target: "10000000", difficulty: 2},
    expectedTarget: s.helper.ethDiff2Target(2)},
  {name: "difficulty without target", fields: {difficulty: 2},
    expectedTarget: s.helper.ethDiff2Target(2)},
]) {
  test(`Autolykos MoneroOcean jobs preserve ${name} conversion`, async () => {
    const miner = await loadMinerWithStubs();
    const setJob = miner.getSetJob();
    await withMockPool({pool: {logged_in: true}}, async ({socket, poolConfig}) => {
      miner.global.opt.pools[0] = poolConfig;
      pool.connect_pool_throttle(0, setJob);
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "job", params: {
          algo: "autolykos2", job_id: "mo-target", blob: "11".repeat(32), ...fields,
        },
      }) + "\n"));
      const jobs = miner.sentMessages.filter((message) => message.type === "job");
      assert.equal(jobs.length, 1);
      assert.equal(jobs[0]?.job?.target, expectedTarget);
    });
  });
}

test("Ergo rejects zero, malformed, and out-of-range decimal targets before mining", async () => {
  for (const target of ["0", "-1", "1.5", "1e3", "0x10", "zz", (1n << 256n).toString()]) {
    await withMockPool({
      pool: {protocol: "erg", logged_in: true}, opt: {job: {algo: "autolykos2"}},
    }, async ({socket, poolConfig}) => {
      let startedJobs = 0;
      pool.connect_pool_throttle(0, (job) => {
        ++startedJobs;
        return s.completeMiningJob(job);
      });
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "mining.notify",
        params: ["invalid-target", 1, "11".repeat(32), "", "", 0, target],
      }) + "\n"));
      assert.equal(startedJobs, 0, target);
      assert.equal(poolConfig.last_job, null, target);
      assert.equal(socket.destroyed, true, target);
    });
  }
});

test("nicehash xn prefixes longer than noncebytes are truncated", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();

  assert.doesNotThrow(() => setJob({
    algo: "cn/0",
    blob_hex: "00".repeat(4),
    noncebytes: 4,
    nonceoffset: 0,
    xn: "001122334455",
    difficulty: 1,
    id: "worker",
    job_id: "job",
  }));

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.ok(jobMessage);
  assert.ok(jobMessage.job);
  assert.equal(jobMessage.job.nonce, "00112233");
  assert.equal(jobMessage.job.nicehash_mask, "ffffffff");
});

test("nicehash xn jobs preserve valid saved nonce values and types", async () => {
  const miner = await loadMinerWithStubs();
  for (const {noncebytes, xn, nonce} of [
    {noncebytes: 4, xn: "a1b2", nonce: "A1B20017"},
    {noncebytes: 4, xn: "00", nonce: "123"},
    {noncebytes: 4, xn: "00", nonce: 0},
    {noncebytes: 4, xn: "a1b2", nonce: 0xa1b20017},
    {noncebytes: 8, xn: "a1b2", nonce: "a1b2000000000017"},
    {noncebytes: 8, xn: "00", nonce: Number.MAX_SAFE_INTEGER},
    {noncebytes: 4, xn: "001122334455", nonce: "00112233"},
  ]) {
    const job = miner.getSetJob()({
      algo: "cn/0", blob_hex: "00".repeat(noncebytes), noncebytes, nonceoffset: 0,
      xn, nonce, difficulty: 1, job_id: "resume",
    });
    assert.equal(job.nonce, nonce);
    assert.equal(job.nicehash_mask, "ff".repeat(Math.min(xn.length / 2, noncebytes))
      .padEnd(noncebytes * 2, "0"));
  }
});

test("nicehash xn jobs reject invalid saved progress before dispatch", async () => {
  const miner = await loadMinerWithStubs();
  for (const {noncebytes, xn, nonce} of [
    {noncebytes: 4, xn: "a1", nonce: "a2000000"},
    {noncebytes: 4, xn: "00", nonce: "100000000"},
    {noncebytes: 4, xn: "00", nonce: 0x100000000},
    {noncebytes: 4, xn: "00", nonce: -1},
    {noncebytes: 4, xn: "00", nonce: 0.5},
    {noncebytes: 8, xn: "00", nonce: Number.MAX_SAFE_INTEGER + 1},
    {noncebytes: 8, xn: "00", nonce: "10000000000000000"},
    {noncebytes: 8, xn: "00", nonce: "0x10"},
    {noncebytes: 8, xn: "00", nonce: ""},
    {noncebytes: 8, xn: "00", nonce: "xyz"},
  ]) {
    assert.throws(() => miner.getSetJob()({
      algo: "cn/0", blob_hex: "00".repeat(noncebytes), noncebytes, nonceoffset: 0,
      xn, nonce, difficulty: 1, job_id: "invalid-resume",
    }), /Invalid .*nonce/);
  }
  assert.deepEqual(miner.sentMessages, []);
});
