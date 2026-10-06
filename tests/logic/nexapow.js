"use strict";

const s = require("./support");
const {test, helper, opts, pool, loadMinerWithStubs, withMockPool} = s;
/** @type {typeof import("node:assert/strict")} */
const assert = s.assert;
/** @type {(job: PoolJob) => MiningJob} */
const completeMiningJob = s.completeMiningJob;

/** @typedef {PoolJob} NexaJob */

test("Nexa pools use Echelon jobs and exact Stratum difficulty targets", async () => {
  /** @type {NexaJob | null} */
  let job = null;
  const header = "e9c64cfd6711c5eaa8d20dc4c4a430b1e22141ecc1ad701471492420432d8a4b";
  await withMockPool({
    pool: {login: "nexa:wallet.worker"}, opt: {job: {algo: "nexapow"}},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} next */ (next) => {
      job = next;
      return completeMiningJob(next);
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.deepEqual(writes[0], {
      jsonrpc: "2.0", id: 1, method: "mining.subscribe", params: [opts.agent_str],
    });
    socket.emit("data", Buffer.from([
      {id: 1, result: [[["mining.set_difficulty", "session"]], "0000000000000001", 8], error: null},
      {id: 2, result: true, error: null},
      {id: null, method: "mining.set_difficulty", params: [2.5]},
      {id: null, method: "mining.notify", params: [
        "4f", header, "1b02fab2", "0000000063b1fb60", true,
      ]},
    ].map((message) => JSON.stringify(message)).join("\n") + "\n"));
    assert.ok(writes[1]);
    assert.ok(job);
    assert.deepEqual(writes[1].params, ["nexa:wallet.worker", "x"]);
    assert.equal(job["submit_mode"], "echelon");
    assert.ok(job.blob);
    assert.equal(job.blob, header + "0000000000000001" + "00".repeat(8));
    assert.ok(job.target);
    assert.equal(job.target, helper.ethDiff2Target(2.5));
    assert.equal(job.noncebytes, 8);
    assert.equal(job.nonceoffset, 40);
    assert.equal(job.submit_mode, "echelon");
    assert.equal(job.extra_nonce, "0000000000000001");
    assert.equal(job.ntime, "0000000063b1fb60");
    socket.emit("data", Buffer.from(JSON.stringify({
      id: null,
      method: "mining.notify",
      params: ["__proto__", header, "1b02fab2", "0000000063b1fb61", false],
    }) + "\n"));
    assert.equal(poolConfig.last_job?.job_id, "__proto__");
    assert.equal(poolConfig.last_job?.extra_nonce, "0000000000000001");
    assert.equal(poolConfig.last_job?.ntime, "0000000063b1fb61");
  });
});

test("Nexa numeric job IDs survive native result matching and submission", async () => {
  const miner = await loadMinerWithStubs();
  const jobId = Number.MAX_SAFE_INTEGER - 1;
  const header = "e9".repeat(32);
  const nbits = "1b02".repeat(2);
  const ntime = "63b1fb60".repeat(2);
  await withMockPool({
    pool: {
      login: "nexa:wallet.worker", protocol: "echelon", logged_in: true,
      extra_nonce: "01".repeat(8), extra_nonce2_size: 8,
    },
    opt: {job: {algo: "nexapow"}},
  }, async ({socket, poolConfig}) => {
    miner.global.opt.pools[0] = poolConfig;
    pool.connect_pool_throttle(0, miner.getSetJob());
    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.notify", params: [jobId, header, nbits, ntime, true],
    }) + "\n"));

    const job = poolConfig.last_job;
    assert.ok(job);
    assert.equal(job.job_id, jobId);
    assert.equal(typeof job.job_id, "number");
    assert.equal(job["submit_mode"], "echelon");
    const nativeJob = miner.sentMessages.find((message) => message.type === "job")?.job;
    assert.ok(nativeJob);
    assert.equal(nativeJob.job_id, jobId);
    assert.equal(typeof nativeJob.job_id, "number");
    assert.equal(typeof nativeJob["job_token"], "string");

    miner.messageHandler({thread_id: 0, type: "result", value: {
      pool_id: "0", worker_id: "worker", job_id: String(jobId),
      job_token: nativeJob["job_token"], nonce: "00".repeat(7) + "01", hash: "00".repeat(32),
    }});
    assert.equal(miner.loggedErrors.length, 0);
    assert.equal(miner.poolWrites.length, 1);
    const write = miner.poolWrites[0];
    assert.ok(write);
    assert.equal(write.json.method, "mining.submit");
    assert.ok(Array.isArray(write.json.params));
    assert.equal(write.json.params[1], jobId);
    assert.equal(typeof write.json.params[1], "number");
  });
});

test("Nexa official Echelon jobs require the clean-jobs boolean", async () => {
  let dispatched = 0;
  await withMockPool({
    pool: {protocol: "echelon", logged_in: true,
      extra_nonce: "0000000000000001", extra_nonce2_size: 8},
    opt: {job: {algo: "nexapow"}},
  }, async ({socket}) => {
    pool.connect_pool_throttle(0, (job) => {
      ++dispatched;
      return completeMiningJob(job);
    });
    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.notify",
      params: ["invalid", "00".repeat(32), "1d00ffff", "0000000063b1fb60", 1],
    }) + "\n"));
    assert.equal(dispatched, 0);
  });
});

test("Nexa submit preserves the job's complete 128-bit Echelon nonce", async () => {
  const miner = await loadMinerWithStubs();
  const poolConfig = miner.global.opt.pools[0];
  poolConfig.login = "nexa:wallet.worker";
  poolConfig.extra_nonce = "0000000000000002";
  poolConfig.last_job = {
    job_id: "old", job_token: "token", submit_mode: "echelon",
    extra_nonce: "0000000000000001", ntime: "0000000063b1fb60",
  };
  miner.messageHandler({thread_id: 0, type: "result", value: {
    pool_id: "0", worker_id: "worker", job_id: "old", job_token: "token",
    nonce: "00003e05486566fd", hash: "00".repeat(32),
  }});
  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.deepEqual(JSON.parse(JSON.stringify(write.json)), {
    jsonrpc: "2.0", id: 3, method: "mining.submit",
    params: ["nexa:wallet.worker", "old", "000000000000000100003e05486566fd", "0000000063b1fb60"],
  });
});

test("Nexa four-field jobs separate the pool prefix from the eight-byte worker nonce", async () => {
  const miner = await loadMinerWithStubs();
  const wireHeader = Buffer.from(Array.from({length: 32}, (_, index) => index));
  await withMockPool({
    pool: {protocol: "echelon", logged_in: true, extra_nonce: "10203040", extra_nonce2_size: 4},
    opt: {job: {algo: "nexapow"}},
  }, async ({socket, poolConfig}) => {
    miner.global.opt.pools[0] = poolConfig;
    pool.connect_pool_throttle(0, miner.getSetJob());
    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.notify", params: ["short-job", wireHeader.toString("hex"), 1234567, "1d00ffff"],
    }) + "\n"));
    const job = poolConfig.last_job;
    assert.ok(job);
    assert.equal(job.blob, Buffer.from(wireHeader).reverse().toString("hex") +
      "10203040" + "00".repeat(8));
    assert.equal(job.height, 1234567);
    assert.equal(job["nbits"], "1d00ffff");
    assert.equal(job.ntime, undefined);
    assert.equal(job["extra_nonce2_size"], 4);
    assert.equal(job.xn, undefined);
    const nativeJob = miner.sentMessages.find((message) => message.type === "job")?.job;
    assert.ok(nativeJob);
    assert.equal(nativeJob.blob_hex, job.blob);
    assert.equal(nativeJob.noncebytes, 8);
    assert.equal(nativeJob.nonceoffset, 36);
    assert.equal(nativeJob.nonce, "00".repeat(8));
    assert.equal(nativeJob.nicehash_mask, "00".repeat(8));
  });
});

test("Nexa four-field submission sends the complete worker nonce with the job's fixed prefix", async () => {
  const miner = await loadMinerWithStubs();
  const poolConfig = miner.global.opt.pools[0];
  poolConfig.extra_nonce = "99887766";
  poolConfig.extra_nonce2_size = 8;
  poolConfig.last_job = {
    job_id: "short-job", job_token: "short-token", submit_mode: "echelon",
    extra_nonce: "10203040", extra_nonce2_size: 4, noncebytes: 8, nonceoffset: 36,
  };
  miner.messageHandler({thread_id: 0, type: "result", value: {
    pool_id: "0", worker_id: "worker", job_id: "short-job", job_token: "short-token",
    nonce: "5060708090a0b0c0", hash: "00".repeat(32),
  }});
  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.equal(write.json.method, "mining.submit");
  assert.deepEqual(write.json.params,
    [poolConfig.login, "short-job", "10203040", "00000000", "5060708090a0b0c0"]);
});

test("Nexa four-field submission rejects an invalid fixed prefix", () => {
  const {nexaSubmitParams} = require("../../miner/submission");
  assert.throws(() => nexaSubmitParams({login: "user"},
    {extra_nonce: "102030", extra_nonce2_size: 4},
    {nonce: "5060708090a0b0c0", job_id: "short-job"}), /fixed nonce prefix/);
});

for (const [name, extraNonce, extraSize, height, bits] of [
  ["wrong prefix width", "1020", 4, 1234567, "1d00ffff"],
  ["wrong suffix width", "10203040", 8, 1234567, "1d00ffff"],
  ["invalid height", "10203040", 4, -1, "1d00ffff"],
  ["invalid compact bits", "10203040", 4, 1234567, "zz"],
]) {
  test(`Nexa four-field jobs reject ${name} before dispatch`, async () => {
    let dispatched = 0;
    await withMockPool({
      pool: {protocol: "echelon", logged_in: true,
        extra_nonce: String(extraNonce), extra_nonce2_size: Number(extraSize)},
      opt: {job: {algo: "nexapow"}},
    }, async ({socket}) => {
      pool.connect_pool_throttle(0, (job) => {
        ++dispatched;
        return completeMiningJob(job);
      });
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "mining.notify", params: ["invalid", "00".repeat(32), height, bits],
      }) + "\n"));
      assert.equal(dispatched, 0);
    });
  });
}
