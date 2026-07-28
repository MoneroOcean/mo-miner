"use strict";

const s = require("./support");
const { test, assert, pool, loadMinerWithStubs, withMockPool } = s;

test("Iron Fish pools build fishhash jobs from object stratum notify", async () => {
  let jobMessage = null;
  const header = "11".repeat(180);
  const target = "0f".repeat(32);
  await withMockPool({
    pool: { login: "ironfishwallet.worker", protocol: "ironfish" },
    opt: { job: { algo: "fishhash" } },
  }, async ({ socket, poolConfig }) => {
    pool.connect_pool_throttle(0, (job) => { jobMessage = job; return job; });
    socket.emit("connect");

    socket.emit("data", Buffer.from(
      JSON.stringify({ method: "mining.subscribed", body: { xn: "a1b2" } }) + "\n" +
      JSON.stringify({ method: "mining.set_target", body: { target } }) + "\n" +
      JSON.stringify({ method: "mining.notify", body: { miningRequestId: 17, header } }) + "\n"
    ));

    assert.equal(poolConfig.ironfish_xn, "a1b2");
    assert.equal(poolConfig.ironfish_target, target);
    assert.equal(jobMessage.algo, "fishhash");
    assert.equal(jobMessage.job_id, 17);
    assert.equal(jobMessage.blob, header);
    assert.equal(jobMessage.target, target);
    assert.equal(jobMessage.noncebytes, 8);
    assert.equal(jobMessage.nonceoffset, 0);
  });
});

test("Iron Fish submit uses mining.submit object body", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].submit_mode = "ironfish";

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "17",
      nonce: "0000000000000005",
      hash: "00".repeat(32),
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(JSON.stringify(miner.poolWrites[0].json), JSON.stringify({
    id: 2,
    method: "mining.submit",
    body: {
      miningRequestId: "17",
      randomness: "0000000000000005",
      graffiti: "00".repeat(32),
    },
  }));
});

test("Kaspa-family pools build 80-byte jobs from exact 64-bit notify words", async () => {
  let jobMessage = null;
  await withMockPool({
    pool: { login: "kaspa:qzwallet.mom", protocol: "kaspa" },
    opt: { job: { algo: "karlsenhashv2" } },
  }, async ({ socket, writes, poolConfig }) => {
    pool.connect_pool_throttle(0, (job) => { jobMessage = job; return job; });
    socket.emit("connect");
    assert.equal(writes[0].method, "mining.subscribe");

    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":[null,"56e0"]}\n' +
      '{"jsonrpc":"2.0","id":2,"error":null,"result":true}\n' +
      '{"id":null,"method":"mining.set_difficulty","params":[4.25]}\n' +
      '{"id":null,"method":"mining.notify","params":["7a",[' +
        "18446744073709551615,9223372036854775808,72623859790382856,4909777105915057546" +
        "],1781909733171]}\n"
    ));

    assert.equal(poolConfig.extra_nonce, "56e0");
    assert.equal(poolConfig.kaspa_difficulty, 4.25);
    assert.equal(jobMessage.algo, "karlsenhashv2");
    assert.equal(jobMessage.job_id, "7a");
    assert.equal(jobMessage.nonce, "56e0000000000000");
    assert.equal(jobMessage.nicehash_mask, "ffff000000000000");
    assert.equal(jobMessage.nonceoffset, 72);
    assert.equal(jobMessage.blob.length, 160);
    assert.equal(jobMessage.blob.slice(0, 16), "ffffffffffffffff");
    assert.equal(jobMessage.blob.slice(16, 32), "0000000000000080");
    assert.equal(jobMessage.blob.slice(32, 48), "0807060504030201");
    assert.equal(jobMessage.blob.slice(48, 64), "8a9569c443082344");
    assert.equal(jobMessage.blob.slice(64, 80), "33bf18e29e010000");
    assert.equal(jobMessage.blob.slice(80, 144), "00".repeat(32));
    assert.equal(jobMessage.blob.slice(144), "0000000000000000");
  });
});

test("Kaspa-family submit uses mining.submit [wallet.worker, job_id, 0x+nonce]", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].submit_mode = "kaspa";
  miner.global.opt.pools[0].login = "kaspa:qzwallet.mom";

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "7a",
      // native nonce_to_hex(%016PRIx64): the winning 8-byte nonce big-endian; the extranonce (high
      // bytes) leads, so the pool re-parses it big-endian with no further work -- pass it through as-is.
      nonce: "56e0000000abcdef",
      hash: "00".repeat(32),
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(miner.poolWrites[0].json.method, "mining.submit");
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "kaspa:qzwallet.mom",
    "7a",
    "0x56e0000000abcdef",
  ]));
});
