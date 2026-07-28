"use strict";

const s = require("./support");
const { test, assert, pool, loadMinerWithStubs, withMockPool } = s;

test("BeamHash III pools build jobs from login nonceprefix and job push", async () => {
  let jobMessage = null;
  const input = "22".repeat(32);
  await withMockPool({
    pool: { login: "beamwallet.worker", protocol: "beam" },
    opt: { job: { algo: "beamhash3" } },
  }, async ({ socket, writes, poolConfig }) => {
    pool.connect_pool_throttle(0, (job) => { jobMessage = job; return job; });
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
        id: "beam-job", method: "job", input, difficulty: 881445,
      }) + "\n"
    ));

    assert.equal(poolConfig.beam_nonceprefix, "a1b2");
    assert.equal(poolConfig.beam_forkheight, 321);
    assert.equal(jobMessage.algo, "beamhash3");
    assert.equal(jobMessage.job_id, "beam-job");
    assert.equal(jobMessage.header_hash, input);
    assert.equal(jobMessage.difficulty, 881445);
    assert.equal(jobMessage.target.slice(-8), "000d7325");
  });
});

test("BeamHash III submit uses solution message with raw nonce byte order", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].submit_mode = "beam";
  miner.global.opt.pools[0].login = "beamwallet.worker";
  const solution = "ab".repeat(104);

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "beam-job",
      nonce: "a1b2000000000007",
      hash: "00".repeat(32),
      solution,
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(JSON.stringify(miner.poolWrites[0].json), JSON.stringify({
    jsonrpc: "2.0",
    id: "beam-job",
    method: "solution",
    nonce: "070000000000b2a1",
    output: solution,
  }));
});
