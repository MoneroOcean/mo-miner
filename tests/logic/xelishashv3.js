"use strict";

const s = require("./support");
const {test, opts, pool, loadMinerWithStubs, withMockPool} = s;
/** @type {typeof import("node:assert/strict")} */
const assert = s.assert;
const unexpectedPoolJob = s.unexpectedPoolJob;
/** @type {(job: PoolJob) => MiningJob} */
const completeMiningJob = s.completeMiningJob;

/** @typedef {PoolJob} XelisJob */

test("XELIS v3 jobs infer the versioned subscribe and authorize protocol", async () => {
  /** @type {XelisJob | undefined} */
  let jobMessage;
  const jobId = "aa".repeat(16);
  const header = "11".repeat(32);
  const extraNonce = "22".repeat(32);
  const publicKey = "33".repeat(32);
  await withMockPool({
    pool: {login: "xel:wallet.worker", is_keepalive: true},
    opt: {job: {algo: "xelishashv3"}},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.deepEqual(writes[0], {
      jsonrpc: "2.0", id: 1, method: "mining.subscribe", params: [opts.agent_str, ["xel/v3"]],
    });

    socket.emit("data", Buffer.from([
      {jsonrpc: "2.0", id: 1, result: ["session", extraNonce, 32, publicKey]},
      {method: "mining.set_difficulty", params: [512]},
      {jsonrpc: "2.0", id: 2, result: true},
      {method: "mining.notify", params: [jobId, "19726D97F49", header, "XEL/V3", true]},
    ].map((message) => JSON.stringify(message)).join("\n") + "\n"));

    assert.ok(writes[1]);
    assert.ok(jobMessage);
    assert.deepEqual(writes[1].params, ["xel:wallet", "worker", "x"]);
    assert.equal(poolConfig.logged_in, true);
    assert.equal(poolConfig.xelis_extra_nonce, extraNonce);
    assert.equal(poolConfig.xelis_public_key, publicKey);
    assert.equal(jobMessage.algo, "xelishashv3");
    assert.equal(jobMessage.submit_mode, "xelis");
    assert.equal(jobMessage.job_id, jobId);
    assert.equal(jobMessage.blob,
      header + "0000019726D97F49" + "00".repeat(8) + extraNonce + publicKey);
    assert.equal(jobMessage.noncebytes, 8);
    assert.equal(jobMessage.nonceoffset, 40);
    assert.equal(jobMessage.target,
      (((1n << 256n) - 1n) / 512n).toString(16).padStart(64, "0"));

    socket.emit("data", Buffer.from(JSON.stringify({jsonrpc: "2.0", id: 4, method: "mining.ping"}) + "\n"));
    assert.ok(writes[2]);
    assert.deepEqual(writes[2], {jsonrpc: "2.0", id: 4, method: "mining.pong"});
  });
});

test("XELIS pool version and opaque job IDs preserve their wire values", async () => {
  for (const jobId of ["0", "work:7", 0]) {
    /** @type {PoolJob | null} */
    let received = null;
    await withMockPool({
      pool: {protocol: "xelis", login: "xel:wallet"},
      opt: {job: {algo: "xelishashv3"}},
    }, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, (job) => {
        received = job;
        return completeMiningJob(job);
      });
      socket.emit("connect");
      socket.emit("data", Buffer.from([
        {id: 1, result: ["", "22".repeat(32), 32, "33".repeat(32)]},
        {id: 2, result: true},
        {id: 1, method: "mining.set_difficulty", params: [250000]},
        {id: 1, method: "mining.notify", params: [jobId, "1", "11".repeat(32), "xel/3", true]},
      ].map((message) => JSON.stringify(message)).join("\n") + "\n"));
      assert.ok(received);
      assert.equal(received.job_id, jobId);
      assert.equal(received.algo, "xelishashv3");
      assert.equal(poolConfig.bad_shares, 0);
    });
  }
});

test("XELIS rejects empty and oversized opaque job IDs", async () => {
  for (const jobId of ["", "x".repeat(257)]) {
    let received = false;
    await withMockPool({
      pool: {protocol: "xelis", login: "xel:wallet"},
      opt: {job: {algo: "xelishashv3"}},
    }, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, (job) => {
        received = true;
        return completeMiningJob(job);
      });
      socket.emit("connect");
      socket.emit("data", Buffer.from([
        {id: 1, result: ["", "22".repeat(32), 32, "33".repeat(32)]},
        {id: 2, result: true},
        {id: 1, method: "mining.set_difficulty", params: [250000]},
        {id: 1, method: "mining.notify", params: [jobId, "1", "11".repeat(32), "xel/3", true]},
      ].map((message) => JSON.stringify(message)).join("\n") + "\n"));
      assert.equal(received, false);
      assert.equal(poolConfig.last_job, null);
    });
  }
});

test("XELIS legacy v3 alias accepts extranonce and public-key updates", async () => {
  /** @type {XelisJob | undefined} */
  let jobMessage;
  const jobId = "bb".repeat(16);
  const oldExtraNonce = "44".repeat(32);
  const oldPublicKey = "55".repeat(32);
  const newExtraNonce = "66".repeat(32);
  const newPublicKey = "77".repeat(32);
  await withMockPool({
    pool: {protocol: "XELIS", login: "xel:wallet"},
    opt: {job: {algo: "xelishashv3"}},
  }, async ({socket, writes}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from([
      {jsonrpc: "2.0", id: 1, result: ["session", oldExtraNonce, 32, oldPublicKey]},
      {jsonrpc: "2.0", id: 2, result: true},
      {method: "mining.set_extranonce", params: [newExtraNonce, 32, newPublicKey]},
      {method: "mining.notify", params: [jobId, "1", "aa".repeat(32), "xel/2", false]},
    ].map((message) => JSON.stringify(message)).join("\n") + "\n"));

    assert.ok(writes[1]);
    assert.ok(jobMessage);
    assert.deepEqual(writes[1].params, ["xel:wallet", "mom", "x"]);
    assert.equal(jobMessage.blob, "aa".repeat(32) + "0000000000000001" + "00".repeat(8) +
      newExtraNonce + newPublicKey);
  });
});

test("XELIS rejects a mixed stale key pair after a malformed update", async () => {
  let received = false;
  const oldExtraNonce = "44".repeat(32);
  const oldPublicKey = "55".repeat(32);
  const newExtraNonce = "66".repeat(32);
  await withMockPool({
    pool: {protocol: "xelis", login: "xel:wallet"},
    opt: {job: {algo: "xelishashv3"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, () => {
      received = true;
      throw new Error("unexpected XELIS job");
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from([
      {id: 1, result: ["session", oldExtraNonce, 32, oldPublicKey]},
      {id: 2, result: true},
      {method: "mining.set_extranonce", params: [newExtraNonce, 32, "invalid-key"]},
      {method: "mining.notify", params: ["job", "1", "aa".repeat(32), "xel/v3", true]},
    ].map((message) => JSON.stringify(message)).join("\n") + "\n"));

    assert.equal(received, false);
    assert.equal(poolConfig.xelis_extra_nonce, "");
    assert.equal(poolConfig.xelis_public_key, "");
    assert.equal(poolConfig.last_job, null);
  });
});

test("XELIS rejects invalid pool difficulty before building a job", async () => {
  const extraNonce = "22".repeat(32);
  const publicKey = "33".repeat(32);
  for (const difficulty of [0, -1, "1.5", "not-a-number", "1".repeat(79)]) {
    await withMockPool({
      pool: {protocol: "xelis", login: "xel:wallet"},
      opt: {job: {algo: "xelishashv3"}},
    }, async ({socket}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("connect");
      socket.emit("data", Buffer.from([
        {jsonrpc: "2.0", id: 1, result: ["session", extraNonce, 32, publicKey]},
        {jsonrpc: "2.0", id: 2, result: true},
        {method: "mining.set_difficulty", params: [difficulty]},
        {
          method: "mining.notify",
          params: ["aa".repeat(16), "1", "11".repeat(32), "xel/v3", true],
        },
      ].map((message) => JSON.stringify(message)).join("\n") + "\n"));
      assert.equal(socket.destroyed, true, String(difficulty));
    });
  }
});

test("XELIS keeps full 256-bit pool difficulty exact", async () => {
  /** @type {XelisJob | undefined} */
  let jobMessage;
  const difficulty = (1n << 200n).toString();
  const extraNonce = "22".repeat(32);
  const publicKey = "33".repeat(32);
  await withMockPool({
    pool: {protocol: "xelis", login: "xel:wallet"},
    opt: {job: {algo: "xelishashv3"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from([
      {jsonrpc: "2.0", id: 1, result: ["session", extraNonce, 32, publicKey]},
      {jsonrpc: "2.0", id: 2, result: true},
      {method: "mining.set_difficulty", params: [difficulty]},
      {
        method: "mining.notify",
        params: ["aa".repeat(16), "1", "11".repeat(32), "xel/v3", true],
      },
    ].map((message) => JSON.stringify(message)).join("\n") + "\n"));

    assert.ok(jobMessage);
    assert.equal(poolConfig.xelis_difficulty, difficulty);
    assert.equal(jobMessage.target,
      (((1n << 256n) - 1n) / BigInt(difficulty)).toString(16).padStart(64, "0"));
    assert.equal(jobMessage.difficulty, undefined);
  });
});

test("XELIS rejects timestamps that JavaScript cannot represent exactly", async () => {
  const extraNonce = "22".repeat(32);
  const publicKey = "33".repeat(32);
  await withMockPool({
    pool: {protocol: "xelis", login: "xel:wallet"},
    opt: {job: {algo: "xelishashv3"}},
  }, async ({socket}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    socket.emit("data", Buffer.from([
      {jsonrpc: "2.0", id: 1, result: ["session", extraNonce, 32, publicKey]},
      {jsonrpc: "2.0", id: 2, result: true},
      {method: "mining.set_difficulty", params: [1]},
      {
        method: "mining.notify",
        params: ["aa".repeat(16), Number.MAX_SAFE_INTEGER + 1, "11".repeat(32), "xel/v3", true],
      },
    ].map((message) => JSON.stringify(message)).join("\n") + "\n"));
    assert.equal(socket.destroyed, true);
  });
});

test("XELIS submit uses the worker suffix without a 0x nonce prefix", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].login = "xel:wallet.worker";
  miner.global.opt.pools[0].last_job = {job_id: "job", job_token: "token", submit_mode: "xelis"};

  miner.messageHandler({
    thread_id: 0,
    type: "result",
    value: {
      pool_id: "0", worker_id: "worker", job_id: "job", job_token: "token",
      nonce: "0011223344556677", hash: "00".repeat(32),
    },
  });

  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.deepEqual(JSON.parse(JSON.stringify(write.json)), {
    jsonrpc: "2.0", id: 3, method: "mining.submit",
    params: ["worker", "job", "0011223344556677"],
  });
});

test("XELIS submit preserves a numeric pool job ID", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].login = "xel:wallet.worker";
  miner.global.opt.pools[0].last_job = {job_id: 0, job_token: "token", submit_mode: "xelis"};

  miner.messageHandler({
    thread_id: 0,
    type: "result",
    value: {
      pool_id: "0", worker_id: "worker", job_id: "0", job_token: "token",
      nonce: "0011223344556677", hash: "00".repeat(32),
    },
  });

  assert.deepEqual(miner.poolWrites[0]?.json.params,
    ["worker", 0, "0011223344556677"]);
});

test("XELIS jobs keep the 112-byte blob and nonce offset", async () => {
  const miner = await loadMinerWithStubs();
  miner.getSetJob()({
    algo: "xelishashv3", blob: "00".repeat(112), target: "ff".repeat(32), job_id: "job", difficulty: 1,
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.ok(jobMessage);
  assert.ok(jobMessage.job);
  assert.ok(jobMessage.job.blob_hex);
  assert.equal(jobMessage.job.blob_hex.length, 224);
  assert.equal(jobMessage.job.noncebytes, 8);
  assert.equal(jobMessage.job.nonceoffset, 40);
});
