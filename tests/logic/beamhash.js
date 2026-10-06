"use strict";

const s = require("./support");
const {test, pool, loadMinerWithStubs, withMockPool} = s;
/** @type {typeof import("node:assert/strict")} */
const assert = s.assert;
/** @type {(job: PoolJob) => MiningJob} */
const completeMiningJob = s.completeMiningJob;

/** @typedef {PoolJob & {header_hash?: string, difficulty?: number, target?: string}} BeamJob */

test("BeamHash III share statuses use the common accepted/rejected accounting", async () => {
  await withMockPool({
    pool: {login: "beam-wallet", protocol: "beam"},
    opt: {job: {algo: "beamhash3"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, () => {throw new Error("Unexpected Beam job");});
    socket.emit("connect");
    socket.emit("data", Buffer.from(JSON.stringify({
      id: "login", method: "result", code: 0,
    }) + "\n"));
    socket.emit("data", Buffer.from(JSON.stringify({
      id: 1, method: "result", code: 1,
    }) + "\n"));
    socket.emit("data", Buffer.from(JSON.stringify({
      id: 2, method: "result", code: 2, description: "rejected",
    }) + "\n"));
    assert.equal(poolConfig.good_shares, 1);
    assert.equal(poolConfig.bad_shares, 1);
  });
});

test("BeamHash III pools build jobs from login nonceprefix and job push", async () => {
  /** @type {BeamJob | null} */
  let jobMessage = null;
  const input = "22".repeat(32);
  await withMockPool({
    pool: {login: "beamwallet.worker", protocol: "beam"},
    opt: {job: {algo: "beamhash3"}},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    assert.equal(JSON.stringify(writes[0]), JSON.stringify({
      jsonrpc: "2.0",
      id: "login",
      method: "login",
      api_key: "beamwallet.worker",
    }));

    socket.emit("data", Buffer.from(
      JSON.stringify({
        id: "login", method: "result", code: 0, nonceprefix: "a1b2", forkheight: 321,
      }) + "\n" +
      JSON.stringify({
        id: "17", method: "job", input, difficulty: 881445,
      }) + "\n"
    ));

    assert.ok(jobMessage);
    assert.equal(poolConfig.beam_nonceprefix, "a1b2");
    assert.equal(jobMessage.algo, "beamhash3");
    assert.equal(jobMessage.job_id, "17");
    assert.equal(jobMessage.header_hash, input);
    assert.equal(jobMessage.difficulty, 881445);
    assert.ok(jobMessage.target);
    assert.equal(jobMessage.target.slice(-8), "000d7325");
  });
});

test("BeamHash III rejects malformed login nonceprefix values", async () => {
  for (const nonceprefix of ["a", "gg", "00112233445566"]) {
    await withMockPool({
      pool: {login: "beamwallet.worker", protocol: "beam"},
      opt: {job: {algo: "beamhash3"}},
    }, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, () => { throw new Error("unexpected Beam job"); });
      socket.emit("connect");
      socket.emit("data", Buffer.from(JSON.stringify({
        id: "login", method: "result", code: 0, nonceprefix,
      }) + "\n"));
      assert.equal(poolConfig.logged_in, false);
      assert.equal(poolConfig.beam_nonceprefix, undefined);
    });
  }
});

test("BeamHash III accepts an omitted zero-byte login nonceprefix", async () => {
  await withMockPool({
    pool: {login: "beamwallet.worker", protocol: "beam"},
    opt: {job: {algo: "beamhash3"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, () => { throw new Error("unexpected Beam job"); });
    socket.emit("connect");
    socket.emit("data", Buffer.from(JSON.stringify({
      id: "login", method: "result", code: 0,
    }) + "\n"));
    assert.equal(poolConfig.logged_in, true);
    assert.equal(poolConfig.beam_nonceprefix, undefined);
  });
});

test("BeamHash III clears the login nonce prefix when a socket reconnects", async () => {
  await withMockPool({
    pool: {login: "beamwallet.worker", protocol: "beam"},
    opt: {job: {algo: "beamhash3"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, () => { throw new Error("unexpected Beam job"); });
    socket.emit("connect");
    socket.emit("data", Buffer.from(JSON.stringify({
      id: "login", method: "result", code: 0, nonceprefix: "a1b2",
    }) + "\n"));
    assert.equal(poolConfig.beam_nonceprefix, "a1b2");
    socket.emit("error", new Error("reconnect"));
    assert.equal(poolConfig.beam_nonceprefix, undefined);
  });
});

test("BeamHash III job construction rejects malformed prefixes", async () => {
  const miner = await loadMinerWithStubs();
  const poolConfig = miner.global.opt.pools[0];
  poolConfig.beam_nonceprefix = "00112233445566";
  assert.throws(() => miner.getSetJob()({
    algo: "beamhash3", header_hash: "00".repeat(32), difficulty: 1, job_id: "job",
  }), /Invalid Beam nonce prefix/);
});

test("BeamHash III resumes valid saved counters and preserves its nonzero seed", async () => {
  const miner = await loadMinerWithStubs();
  for (const xn of ["", "a1b2", "001122334455"]) {
    const fields = {algo: "beamhash3", header_hash: "22".repeat(32), xn,
      difficulty: 1, job_id: "resume"};
    assert.equal(miner.getSetJob()(fields).nonce, (xn + "01").padEnd(16, "0"));
    const nonce = (xn + "02").padEnd(16, "0");
    assert.equal(miner.getSetJob()({...fields, nonce}).nonce, nonce);
  }
  assert.equal(miner.getSetJob()({
    algo: "beamhash3", header_hash: "22".repeat(32), xn: "00", nonce: 0x123,
    difficulty: 1, job_id: "numeric",
  }).nonce, 0x123);
});

test("BeamHash III rejects invalid saved counters before dispatch", async () => {
  const miner = await loadMinerWithStubs();
  for (const {xn, nonce} of [
    {xn: "", nonce: 0}, {xn: "", nonce: "0000000000000000"},
    {xn: "a1b2", nonce: "a1b3000000000001"},
    {xn: "", nonce: "10000000000000000"}, {xn: "", nonce: "xyz"},
    {xn: "", nonce: Number.MAX_SAFE_INTEGER + 1},
  ]) {
    assert.throws(() => miner.getSetJob()({
      algo: "beamhash3", header_hash: "22".repeat(32), xn, nonce,
      difficulty: 1, job_id: "invalid-resume",
    }), /Invalid .*nonce/);
  }
  assert.deepEqual(miner.sentMessages, []);
});

test("BeamHash III snapshots each accepted prefix and resumes only current-job progress", async () => {
  for (const prefix of ["a1b2", ""]) {
    const miner = await loadMinerWithStubs();
    const setJob = miner.getSetJob();
    await withMockPool({
      pool: {protocol: "beam"}, opt: {job: {algo: "beamhash3"}},
    }, async ({socket, poolConfig}) => {
      miner.global.opt.pools[0] = poolConfig;
      pool.connect_pool_throttle(0, (job) => {
        assert.equal(job.xn, poolConfig.beam_nonceprefix ?? "");
        return setJob(job);
      });
      socket.emit("connect");
      const notify = (/** @type {UnknownRecord} */ message) =>
        socket.emit("data", Buffer.from(JSON.stringify(message) + "\n"));
      notify({id: "login", method: "result", code: 0, nonceprefix: prefix});
      notify({id: "same", method: "job", input: "22".repeat(32), difficulty: 1});
      const previous = poolConfig.last_job;
      assert.ok(previous);
      const token = previous["job_token"];
      const saved = (prefix + "02").padEnd(16, "0");
      miner.global.opt.pool_ids.active = 1;
      const progress = (/** @type {string} */ nonce, job_token = token) => miner.messageHandler({
        type: "last_nonce", thread_id: 0,
        value: {pool_id: "0", job_id: "same", job_token, nonce},
      });
      progress("0000000000000000");
      assert.equal(previous.nonce, undefined);
      progress(saved);
      progress(prefix ? "ff00000000000000" : "0000000000000000");
      assert.equal(previous.nonce, saved);
      notify({id: "login", method: "result", code: 0, nonceprefix: "ccdd"});
      assert.equal(poolConfig.beam_nonceprefix, "ccdd");
      assert.equal(previous.xn, prefix);
      miner.global.opt.pool_ids.active = 0;
      const resumed = setJob(previous);
      assert.equal(resumed.nonce, saved);
      assert.equal(resumed.nicehash_mask, "ff".repeat(prefix.length / 2).padEnd(16, "0"));

      notify({id: "same", method: "job", input: "33".repeat(32), difficulty: 1});
      const next = poolConfig.last_job;
      assert.ok(next);
      assert.notEqual(next, previous);
      assert.equal(next.xn, "ccdd");
      assert.equal(miner.sentMessages.at(-1)?.job?.nonce, "ccdd010000000000");
      miner.global.opt.pool_ids.active = 1;
      progress("ccdd020000000000", resumed.job_token);
      assert.equal(next.nonce, undefined);
      progress("ccdd020000000000", next["job_token"]);
      assert.equal(next.nonce, "ccdd020000000000");
      socket.emit("error", new Error("reconnect"));
      assert.equal(poolConfig.last_job, null);
      assert.equal(poolConfig.beam_nonceprefix, undefined);
      progress("ccdd030000000000", next["job_token"]);
      assert.equal(poolConfig.last_job, null);
    });
  }
});

test("BeamHash III submit uses solution message with raw nonce byte order", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].login = "beamwallet.worker";
  miner.global.opt.pools[0].last_job = {job_id: "beam-job", job_token: "token", submit_mode: "beam"};
  const solution = "ab".repeat(104);

  miner.messageHandler({
    thread_id: 0,
    type: "result",
    value: {
      pool_id: "0",
      worker_id: "worker",
      job_id: "beam-job",
      job_token: "token",
      nonce: "a1b2000000000007",
      hash: "00".repeat(32),
      solution,
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.equal(JSON.stringify(write.json), JSON.stringify({
    jsonrpc: "2.0",
    id: "beam-job",
    method: "solution",
    nonce: "070000000000b2a1",
    output: solution,
  }));
});

test("BeamHash III rejects malformed worker result fields", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].last_job = {job_id: "beam-job", job_token: "token", submit_mode: "beam"};

  for (const value of [
    {nonce: "a1b20000000007", solution: "ab".repeat(104)},
    {nonce: 7, solution: "ab".repeat(104)},
    {nonce: "a1b2000000000007", solution: "ab".repeat(103)},
  ]) {
    assert.doesNotThrow(() => miner.messageHandler({
      thread_id: 0,
      type: "result",
      value: {
        pool_id: "0",
        worker_id: "worker",
        job_id: "beam-job",
        job_token: "token",
        ...value,
      },
    }));
  }
  assert.equal(miner.poolWrites.length, 0);
});
