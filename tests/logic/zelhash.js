"use strict";

const s = require("./support");
const { test, assert, pool, loadMinerWithStubs, withMockPool } = s;

test("ZelHash (Flux) pools build the 140-byte header from the ZIP-301 notify", async () => {
  let jobMessage = null;
  const version  = "04000000";
  const prevhash = "a8675c842f7a1342fadd00cd9b4e4909526b1c0ab5a747c5529b4deb13000000";
  const merkle   = "ce7d6ea2452245925fc70c3a08a3c3dd2ca4beab7481f237a19751666bfd25c3";
  const reserved = "0fd282d94b1e1a7f2c57eb3fb9e2853d990753fa137e13c99bd43f220d4fce69";
  const ntime    = "90e44f5d";
  const bits     = "ce28421d";
  const target   = "0000000a42ce000000000000000000000000000000000000000000000000000c";
  await withMockPool({
    pool: { is_keepalive: true, login: "t1fluxwallet.worker", protocol: "zelhash" },
    opt: { job: { algo: "zelhash" } },
    pool_time: { keepalive: 0.001, first_job_wait: 0.001 },
  }, async ({ socket, writes, poolConfig }) => {
    pool.connect_pool_throttle(0, (job) => { jobMessage = job; return job; });
    socket.emit("connect");
    assert.equal(writes[0].method, "mining.subscribe");

    // subscribe result -> [[["mining.notify",session]], NONCE1]; NONCE1 = "0a1b" (2-byte prefix)
    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":[[["mining.notify","sess"]],"0a1b"]}\n' +
      '{"jsonrpc":"2.0","id":2,"error":null,"result":true}\n' +
      '{"id":null,"method":"mining.set_target","params":["' + target + '"]}\n' +
      '{"id":null,"method":"mining.notify","params":["job1","' + version + '","' + prevhash + '","' +
        merkle + '","' + reserved + '","' + ntime + '","' + bits + '",true]}\n'
    ));

    assert.equal(writes[1].method, "mining.authorize");
    assert.deepEqual(writes[1].params, ["t1fluxwallet.worker", "x"]);
    assert.equal(poolConfig.extra_nonce, "0a1b");
    assert.equal(poolConfig.zelhash_target, target);
    assert.equal(jobMessage.algo, "zelhash");
    assert.equal(jobMessage.job_id, "job1");
    assert.equal(jobMessage.target, target);
    assert.equal(jobMessage.ntime, ntime);
    assert.equal(jobMessage.nonce1_len, 2);
    assert.equal(jobMessage.noncebytes, 8);
    assert.equal(jobMessage.nonceoffset, 110); // 108 + nonce1_len
    // 280-hex = 140 bytes; nonce field (last 64 hex) = nonce1(0a1b) || zero nonce2 region.
    assert.equal(jobMessage.blob.length, 280);
    assert.equal(jobMessage.blob.slice(0, 8), version);
    assert.equal(jobMessage.blob.slice(8, 72), prevhash);
    assert.equal(jobMessage.blob.slice(72, 136), merkle);
    assert.equal(jobMessage.blob.slice(136, 200), reserved);
    assert.equal(jobMessage.blob.slice(200, 208), ntime);
    assert.equal(jobMessage.blob.slice(208, 216), bits);
    assert.equal(jobMessage.blob.slice(216), "0a1b" + "0".repeat(60));
  });
});

test("ZelHash submit uses ZIP-301 mining.submit [worker, job_id, time, nonce2, solution]", async () => {
  const miner = await loadMinerWithStubs();
  const solution = "34" + "ab".repeat(52); // 0x34 compactSize + 52-byte compressed proof = 106 hex
  miner.global.opt.pools[0].submit_mode = "zelhash";
  miner.global.opt.pools[0].login = "t1fluxwallet.worker";
  miner.global.opt.pools[0].last_job = {
    job_id: "job1",
    ntime: "90e44f5d",
    nonce1_len: 2,
    // 280-hex header; only the 32-byte nonce (last 64 hex) is read here = nonce1 "0a1b" || zero nonce2.
    blob: "0".repeat(216) + "0a1b" + "0".repeat(60),
  };

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "job1",
      nonce: "0000000000000007", // 8-byte search counter, big-endian (native %016 PRIx64)
      hash: "00".repeat(32),
      solution: solution,
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(miner.poolWrites[0].json.method, "mining.submit");
  // nonce2 = counter(LE) || nonce2 tail past the counter; nonce1 prefix excluded.
  // counter 0x...07 -> wire LE "0700000000000000"; tail = zeros; total = 32 - 2 = 30 bytes = 60 hex.
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "t1fluxwallet.worker",
    "job1",
    "90e44f5d",
    "0700000000000000" + "0".repeat(60 - 16),
    solution,
  ]));
});
