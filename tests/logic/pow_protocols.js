"use strict";

const s = require("./support");
const zlib = require("node:zlib");
const { test, assert, pool, loadMinerWithStubs, withMockPool } = s;

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
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.equal(jobMessage.job.blob_hex, headerHash + "0000000000000000");
  assert.equal(jobMessage.job.noncebytes, 8);
  assert.equal(jobMessage.job.nonceoffset, 32);
  assert.equal(jobMessage.job.worker_id, "worker");
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
  assert.equal(jobMessage.job.worker_id, "user");
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
  assert.equal(jobMessage.job.blob_hex, headerHash + extraNonce);
  assert.equal(jobMessage.job.header_hash, headerHash);
  assert.equal(jobMessage.job.nonce, "ff81000000000000");
  assert.equal(jobMessage.job.nicehash_mask, "ffff000000000000");
});

test("KawPow submit uses the header hash carried by the worker result", async () => {
  const miner = await loadMinerWithStubs();
  const oldHeaderHash = "11".repeat(32);
  const newHeaderHash = "22".repeat(32);
  miner.global.opt.pools[0].submit_mode = "raven";
  miner.global.opt.pools[0].last_job = {
    job_id: "new",
    header_hash: newHeaderHash,
  };

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "old",
      nonce: "ff81000000000001",
      hash: "00".repeat(32),
      mix_hash: "33".repeat(32),
      header_hash: oldHeaderHash,
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "user",
    "old",
    "0xff81000000000001",
    "0x" + oldHeaderHash,
    "0x" + "33".repeat(32),
  ]));
});

test("Etchash submit uses Eth mining.submit format", async () => {
  const miner = await loadMinerWithStubs();
  const headerHash = "22".repeat(32);
  miner.global.opt.pools[0].submit_mode = "eth";

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "203d",
      nonce: "080c000000000001",
      hash: "00".repeat(32),
      mix_hash: "33".repeat(32),
      header_hash: headerHash,
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "user",
    "203d",
    "0x080c000000000001",
    "0x" + headerHash,
    "0x" + "33".repeat(32),
  ]));
});

test("Etchash submit uses ethproxy eth_submitWork format", async () => {
  const miner = await loadMinerWithStubs();
  const headerHash = "22".repeat(32);
  miner.global.opt.pools[0].submit_mode = "ethproxy";

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: headerHash,
      nonce: "080c000000000001",
      hash: "00".repeat(32),
      mix_hash: "33".repeat(32),
      header_hash: headerHash,
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(miner.poolWrites[0].json.method, "eth_submitWork");
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "0x080c000000000001",
    "0x" + headerHash,
    "0x" + "33".repeat(32),
  ]));
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
  miner.global.opt.pools[0].submit_mode = "erg";
  miner.global.opt.pools[0].erg_submit_jobs = {
    "203d": { extra_nonce: "080c", extra_nonce2_size: 6, ntime: "00000002" },
  };

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "203d",
      nonce: "080c000000000001",
      hash: "00".repeat(32),
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "user",
    "203d",
    "000000000001",
    "00000002",
    "080c000000000001",
  ]));
});

test("nicehash xn prefixes longer than noncebytes are truncated", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();

  assert.doesNotThrow(() => setJob({
    algo: "cn/0",
    blob_hex: "abcd",
    noncebytes: 4,
    xn: "001122334455",
    difficulty: 1,
    id: "worker",
    job_id: "job",
  }));

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.equal(jobMessage.job.nonce, "00112233");
  assert.equal(jobMessage.job.nicehash_mask, "ffffffff");
});
