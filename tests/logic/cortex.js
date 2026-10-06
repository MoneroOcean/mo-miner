"use strict";

const s = require("./support");
const {test, helper, pool, loadMinerWithStubs, withMockPool, unexpectedPoolJob} = s;
/** @type {typeof import("node:assert/strict")} */
const assert = s.assert;
/** @type {(job: PoolJob) => MiningJob} */
const completeMiningJob = s.completeMiningJob;

/** @typedef {PoolJob} CortexJob */

test("Cortex RPC builds c30 jobs from ctxc_getWork", async () => {
  /** @type {CortexJob | null} */
  let job = null;
  let dispatched = 0;
  await withMockPool({
    pool: {protocol: "cortex", login: "0xwallet", worker: "rig"},
    opt: {job: {algo: "c30"}},
  }, async ({socket, writes}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} value */ (value) => {
      ++dispatched;
      job = value;
      return completeMiningJob(value);
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.equal(writes.length, 1);
    assert.equal(writes[0].method, "ctxc_submitLogin");
    assert.deepEqual(writes[0].params, ["0xwallet"]);
    socket.emit("data", Buffer.from('{"jsonrpc":"2.0","id":"72","result":true}\n'));
    assert.equal(writes[1]?.method, "ctxc_getWork");
    socket.emit("data", Buffer.from('{"jsonrpc":"2.0","id":72,"result":true}\n'));
    assert.equal(writes.length, 2);
    socket.emit("data", Buffer.from(JSON.stringify({jsonrpc: "2.0", id: "100", result: [
      "0x" + "ab".repeat(32), "0x" + "00".repeat(32), "0x" + "ff".repeat(32), "0x1234",
    ]}) + "\n"));
    assert.ok(job);
    assert.equal(job.algo, "c30");
    assert.equal(job.blob, "ab".repeat(32));
    assert.equal(job.target, "ff".repeat(32));
    assert.equal(job.proofsize, 42);
    assert.equal(job.height, 0x1234);
    assert.equal(typeof job.nonce, "string");
    const nonce = String(job.nonce);
    assert.match(nonce, /^[0-9a-f]{12}$/);
    assert.equal(dispatched, 1);
    socket.emit("data", Buffer.from(JSON.stringify({jsonrpc: "2.0", id: 100, result: [
      "0x" + "cd".repeat(32), "0x" + "00".repeat(32), "0x" + "ff".repeat(32), "0x1235",
    ]}) + "\n"));
    assert.equal(dispatched, 1);
    socket.emit("data", Buffer.from(JSON.stringify({jsonrpc: "2.0", id: 0, result: [
      "0x" + "ef".repeat(32), "0x" + "00".repeat(32), "0x" + "ff".repeat(32), "0x1236",
    ]}) + "\n"));
    assert.equal(dispatched, 2);
    assert.equal(job?.nonce, nonce);
  });
});

test("Cortex keeps its RPC handshake when subscribe mode is disabled", async () => {
  await withMockPool({
    pool: {protocol: "cortex", use_subscribe: false, login: "cortex-wallet"},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    assert.equal(writes[0]?.method, "ctxc_submitLogin");
    assert.equal(writes[0]?.id, 72);
    assert.equal(poolConfig["pending_cortex_login"], true);
    socket.emit("data", Buffer.from('{"id":72,"result":true}\n'));
    assert.equal(writes[1]?.method, "ctxc_getWork");
    assert.equal(writes[1]?.id, 100);
  });
});

test("Cortex getWork errors and malformed replies clear pending work and redact logs", async () => {
  const previousLogError = helper.log_err;
  /** @type {string[]} */
  const logs = [];
  helper.log_err = (message) => logs.push(message);
  try {
    let dispatched = 0;
    await withMockPool({
      pool: {protocol: "cortex", login: "cortex-wallet", pass: "cortex-secret"},
    }, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
        ++dispatched;
        return completeMiningJob(job);
      });
      socket.emit("connect");
      socket.emit("data", Buffer.from('{"id":72,"result":true}\n'));
      assert.equal(poolConfig["pending_cortex_work"], true);
      socket.emit("data", Buffer.from(JSON.stringify({
        id: 100,
        error: {message: "cortex-wallet rejected cortex-secret"},
        result: [
          "0x" + "ab".repeat(32), "0x" + "00".repeat(32), "0x" + "ff".repeat(32), "1",
        ],
      }) + "\n"));
      assert.equal(poolConfig["pending_cortex_work"], false);
      assert.equal(dispatched, 0);

      poolConfig["pending_cortex_work"] = true;
      socket.emit("data", Buffer.from('{"id":100,"result":{}}\n'));
      assert.equal(poolConfig["pending_cortex_work"], false);
    });
    assert.equal(dispatched, 0);
  } finally {
    helper.log_err = previousLogError;
  }
  assert.equal(logs.length, 2);
  assert.ok(logs.every((message) => message.includes("Cortex getWork failed")));
  assert.ok(logs.every((message) => !message.includes("cortex-wallet")));
  assert.ok(logs.every((message) => !message.includes("cortex-secret")));
  assert.ok(logs.some((message) => message.includes("<redacted>")));
  assert.ok(logs.some((message) => message.includes("Invalid work response")));
});

test("Cortex invalid login results report failure and clear login pending state", async () => {
  const previousLogError = helper.log_err;
  /** @type {string[]} */
  const logs = [];
  helper.log_err = (message) => logs.push(message);
  try {
    for (const response of [{id: 72, error: null}, {id: 72, result: null}, {id: 72, result: false}]) {
      await withMockPool({pool: {protocol: "cortex"}}, async ({socket, poolConfig}) => {
        pool.connect_pool_throttle(0, unexpectedPoolJob);
        socket.emit("connect");
        socket.emit("data", Buffer.from(JSON.stringify(response) + "\n"));
        assert.equal(poolConfig.logged_in, false);
        assert.equal(poolConfig["pending_cortex_login"], false);
      });
    }
  } finally {
    helper.log_err = previousLogError;
  }
  assert.equal(logs.length, 3);
  assert.ok(logs.every((message) => message.includes("Login to the pool failed")));
});

test("Cortex submit relays nonce, seal hash, and 42 proof edges", async () => {
  const miner = await loadMinerWithStubs();
  const poolConfig = miner.global.opt.pools[0];
  poolConfig.worker = "rig";
  poolConfig.last_job = {
    job_id: "job", job_token: "token", submit_mode: "cortex", header_hash: "0x" + "ab".repeat(32),
  };
  const edges = "00000001".repeat(42);
  miner.messageHandler({thread_id: 0, type: "result", value: {
    pool_id: "0", worker_id: "worker", job_id: "job", job_token: "token",
    nonce: "0000000000000002",
    hash: "00".repeat(32), edges,
  }});
  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.deepEqual(write.json, {
    jsonrpc: "2.0", id: 73, method: "ctxc_submitWork",
    params: ["0x0000000000000002", "0x" + "ab".repeat(32), "0x" + edges], worker: "rig",
  });
  assert.deepEqual([.../** @type {Set<number>} */ (poolConfig.pending_cortex_submit_ids)], [73]);
  miner.messageHandler({thread_id: 0, type: "result", value: {
    pool_id: "0", worker_id: "worker", job_id: "job", job_token: "token",
    nonce: "0000000000000003", hash: "00".repeat(32), edges,
  }});
  assert.equal(miner.poolWrites[1]?.json.id, 74);
  assert.deepEqual([.../** @type {Set<number>} */ (poolConfig.pending_cortex_submit_ids)], [73, 74]);
});

test("Cortex submit rejects a proof without exactly 42 edges", async () => {
  const miner = await loadMinerWithStubs();
  const poolConfig = miner.global.opt.pools[0];
  poolConfig.last_job = {
    job_id: "job", job_token: "token", submit_mode: "cortex", header_hash: "0x" + "ab".repeat(32),
  };
  miner.messageHandler({thread_id: 0, type: "result", value: {
    pool_id: "0", worker_id: "worker", job_id: "job", job_token: "token",
    nonce: "0000000000000002",
    hash: "00".repeat(32), edges: "00000001".repeat(41),
  }});
  assert.equal(miner.poolWrites.length, 0);
});

test("Cortex submit IDs respect the pending limit and skip the reserved work ID", async () => {
  const miner = await loadMinerWithStubs();
  const poolConfig = miner.global.opt.pools[0];
  poolConfig.last_job = {
    job_id: "job", job_token: "token", submit_mode: "cortex", header_hash: "0x" + "ab".repeat(32),
  };
  const value = {
    pool_id: "0", worker_id: "worker", job_id: "job", job_token: "token",
    nonce: "0000000000000002", hash: "00".repeat(32), edges: "00000001".repeat(42),
  };
  poolConfig.pending_cortex_submit_ids = new Set(
    Array.from({length: 4096}, (_unused, index) => index));
  miner.messageHandler({thread_id: 0, type: "result", value});
  assert.equal(miner.poolWrites.length, 0);
  poolConfig.pending_cortex_submit_ids.clear();

  for (let i = 0; i < 28; ++i) {
    miner.messageHandler({thread_id: 0, type: "result", value});
  }
  assert.equal(miner.poolWrites[26]?.json.id, 99);
  assert.equal(miner.poolWrites[27]?.json.id, 101);
  assert.ok(miner.poolWrites.every(({json}) => json.id !== 100));
});
