"use strict";

const s = require("./support");
const {test, helper, pool, loadMinerWithStubs, withMockPool} = s;
/** @type {typeof import("node:assert/strict")} */
const assert = s.assert;
const unexpectedPoolJob = s.unexpectedPoolJob;
/** @type {(job: PoolJob) => MiningJob} */
const completeMiningJob = s.completeMiningJob;

/** @typedef {PoolJob} FishJob */
/** @typedef {PoolJob} KaspaJob */
/** @typedef {PoolJob} HooJob */

test("Iron Fish pools build FishHash v3 jobs from object stratum notify", async () => {
  /** @type {FishJob | undefined} */
  let jobMessage;
  const header = "11".repeat(180);
  const target = "0f".repeat(32);
  await withMockPool({
    pool: {login: "ironfishwallet.worker", protocol: "ironfish"},
    opt: {job: {algo: "fishhash"}},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    const subscribe = writes[0];
    assert.ok(subscribe);
    assert.equal(subscribe.method, "mining.subscribe");
    const subscribeBody = /** @type {UnknownRecord} */ (subscribe["body"]);
    assert.equal(subscribeBody["version"], 3);

    socket.emit("data", Buffer.from(
      JSON.stringify({method: "mining.subscribed", body: {xn: "a1b2"}}) + "\n" +
      JSON.stringify({method: "mining.set_target", body: {target}}) + "\n" +
      JSON.stringify({method: "mining.notify", body: {miningRequestId: 17, header}}) + "\n"
    ));

    assert.ok(jobMessage);
    assert.equal(poolConfig.ironfish_xn, "a1b2");
    assert.equal(poolConfig.ironfish_target, target);
    assert.equal(jobMessage.algo, "fishhash");
    assert.equal(jobMessage.job_id, 17);
    assert.equal(jobMessage.blob, header);
    assert.equal(jobMessage.target, target);
    assert.equal(jobMessage.noncebytes, 8);
    assert.equal(jobMessage.nonceoffset, 172);
  });
});

test("Iron Fish keeps its object handshake when login mode is disabled", async () => {
  await withMockPool({
    pool: {protocol: "ironfish", use_subscribe: false},
  }, async ({socket, writes}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    assert.equal(writes[0]?.method, "mining.subscribe");
    assert.equal(writes[0]?.id, 1);
  });
});

test("Iron Fish queues a pre-subscription notify and dispatches it once", async () => {
  /** @type {PoolJob | undefined} */
  let jobMessage;
  let dispatched = 0;
  const header = "11".repeat(180);
  await withMockPool({
    pool: {login: "ironfishwallet.worker", protocol: "ironfish"},
    opt: {job: {algo: "fishhash"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      ++dispatched;
      jobMessage = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.notify", body: {miningRequestId: 17, header},
    }) + "\n"));
    assert.equal(dispatched, 0);
    assert.ok(poolConfig.pending_job);

    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.subscribed", body: {xn: "a1b2"},
    }) + "\n"));
    assert.equal(poolConfig.logged_in, true);
    assert.equal(poolConfig.pending_job, undefined);
    assert.equal(dispatched, 1);
    assert.ok(jobMessage);
    assert.equal(jobMessage.job_id, 17);
    assert.equal(jobMessage.algo, "fishhash");

    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.subscribed", body: {xn: "a1b2"},
    }) + "\n"));
    assert.equal(dispatched, 1);
  });
});

test("Iron Fish clears a queued notify on subscription failure", async () => {
  const header = "11".repeat(180);
  const failures = [
    {id: 0, error: {id: 1, message: "subscription failed"}},
    {id: 1, error: "subscription failed"},
    {method: "mining.subscribed", error: "subscription failed"},
    {error: "subscription failed"},
  ];
  for (const failure of failures) {
    let dispatched = 0;
    await withMockPool({
      pool: {login: "ironfishwallet.worker", protocol: "ironfish"},
      opt: {job: {algo: "fishhash"}},
    }, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
        ++dispatched;
        return completeMiningJob(job);
      });
      socket.emit("connect");
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "mining.notify", body: {miningRequestId: 17, header},
      }) + "\n"));
      assert.equal(dispatched, 0);
      assert.ok(poolConfig.pending_job);

      socket.emit("data", Buffer.from(JSON.stringify(failure) + "\n"));
      assert.equal(poolConfig.logged_in, false);
      assert.equal(poolConfig.pending_job, undefined);
      assert.equal(dispatched, 0);
    });
  }
});

test("Iron Fish set_job applies xn at the FishHash v3 randomness offset", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  /** @type {MiningJob | undefined} */
  let nativeJob;
  const header = "11".repeat(180);
  const target = "0f".repeat(32);
  await withMockPool({
    pool: {login: "ironfishwallet.worker", protocol: "ironfish"},
    opt: {job: {algo: "fishhash"}},
  }, async ({socket, poolConfig}) => {
    miner.global.opt.pools[0] = poolConfig;
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      const prepared = setJob(job);
      nativeJob = prepared;
      return prepared;
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(
      JSON.stringify({method: "mining.subscribed", body: {xn: "a1b2"}}) + "\n" +
      JSON.stringify({method: "mining.set_target", body: {target}}) + "\n" +
      JSON.stringify({method: "mining.notify", body: {miningRequestId: 17, header}}) + "\n"
    ));
  });

  assert.ok(nativeJob);
  assert.equal(nativeJob.nonceoffset, 172);
  const nonce = nativeJob.nonce;
  assert.ok(typeof nonce === "string");
  assert.equal(nonce.slice(0, 4), "a1b2");
  const blob = nativeJob.blob_hex;
  assert.ok(typeof blob === "string");
  assert.equal(blob.slice(0, 64), header.slice(0, 64));
  assert.equal(blob.slice(344), header.slice(344));
});

test("FishHash benchmark jobs use v3 and legacy header nonce offsets", async () => {
  const defaultBenchmark = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "fishhash"],
    algoParams: {fishhash: "gpu1*[intensity=1]"},
    waitForMessageType: "bench",
  });
  const defaultMessage = defaultBenchmark.sentMessages.find((message) => message.type === "bench");
  assert.ok(defaultMessage);
  assert.ok(defaultMessage.job);
  assert.equal(defaultMessage.job.noncebytes, 8);
  assert.equal(defaultMessage.job.nonceoffset, 172);
  assert.equal(defaultMessage.job.blob_hex?.length, 360);

  const shortBenchmark = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "fishhash", "--job",
      JSON.stringify({blob_hex: "00".repeat(40)})],
    algoParams: {fishhash: "gpu1*[intensity=1]"},
    waitForMessageType: "bench",
  });
  const shortMessage = shortBenchmark.sentMessages.find((message) => message.type === "bench");
  assert.ok(shortMessage);
  assert.ok(shortMessage.job);
  assert.equal(shortMessage.job.noncebytes, 8);
  assert.equal(shortMessage.job.nonceoffset, 32);
  assert.equal(shortMessage.job.blob_hex?.length, 80);
});

test("Iron Fish submit preserves pool ID and emits the v3 body", async () => {
  const cases = [
    {job_id: 17},
    {job_id: "17"},
  ];
  for (const item of cases) {
    const miner = await loadMinerWithStubs();
    const job = {job_id: item.job_id, job_token: "token", submit_mode: "ironfish"};
    miner.global.opt.pools[0].last_job = job;
    const nonce = "0000000000000005";
    const value = {
      pool_id: "0", worker_id: "internal-worker", job_id: "17", job_token: "token", nonce,
      hash: "00".repeat(32),
    };

    miner.messageHandler({thread_id: 0, type: "result", value});

    assert.equal(miner.poolWrites.length, 1);
    const write = miner.poolWrites[0];
    assert.ok(write);
    assert.equal(write.json.id, 2);
    assert.equal(write.json.method, "mining.submit");
    assert.deepEqual(write.json["body"], {
      miningRequestId: item.job_id, randomness: nonce,
    });
    assert.equal(value.nonce, nonce);
    assert.equal(miner.global.opt.pools[0].last_job, job);
  }
});

test("Iron Fish pools reject malformed 180-byte notify headers", async () => {
  for (const header of [undefined, "00".repeat(179), "gg".repeat(180)]) {
    await withMockPool({
      pool: {login: "ironfishwallet.worker", protocol: "ironfish"},
      opt: {job: {algo: "fishhash"}},
    }, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("connect");
      const body = header === undefined ? {miningRequestId: 17} : {miningRequestId: 17, header};
      socket.emit("data", Buffer.from(JSON.stringify({method: "mining.notify", body}) + "\n"));
      assert.equal(poolConfig.last_job, null);
    });
  }
});

test("Kaspa-family pools build 80-byte jobs from exact 64-bit notify words", async () => {
  /** @type {KaspaJob | undefined} */
  let jobMessage;
  await withMockPool({
    pool: {login: "kaspa:qzwallet.mom", protocol: "kaspa"},
    opt: {job: {algo: "karlsenhashv2"}},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.equal(writes[0].method, "mining.subscribe");

    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":[null,"56e0"]}\n' +
      '{"jsonrpc":"2.0","id":2,"error":null,"result":true}\n' +
      '{"id":null,"method":"mining.set_difficulty","params":[4.25]}\n' +
      '{"id":null,"meta":{"params":["wrong",[9,9,9,9],9]},' +
      '"method":"mining.notify","params":["7\\"a",[' +
        "18446744073709551615,9223372036854775808,72623859790382856,4909777105915057546" +
        "],1781909733171]}\n"
    ));

    assert.ok(jobMessage);
    assert.equal(poolConfig.extra_nonce, "56e0");
    assert.equal(poolConfig.kaspa_difficulty, 4.25);
    assert.equal(jobMessage.algo, "karlsenhashv2");
    assert.equal(jobMessage.job_id, '7"a');
    assert.equal(jobMessage.nonce, "56e0000000000000");
    assert.equal(jobMessage.nicehash_mask, "ffff000000000000");
    assert.equal(jobMessage.nonceoffset, 72);
    assert.ok(jobMessage.blob);
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

test("Kaspa-family jobs preserve exact words with a safe numeric job ID", async () => {
  const jobId = 73;
  const words = [
    (1n << 64n) - 1n,
    1n << 63n,
    (1n << 56n) + 0x07060504030201n,
    (0x12n << 56n) | (0x34n << 48n) | 0x5678n,
  ];
  const timestamp = (1n << 53n) + 0x1234n;
  /** @param {bigint} value @returns {string} */
  function littleEndian(value) {
    const bytes = Buffer.alloc(8);
    bytes.writeBigUInt64LE(value);
    return bytes.toString("hex");
  }
  const expectedPrePow = words.map(littleEndian).join("");
  const expectedTimestamp = littleEndian(timestamp);
  const notify = [
    '{"id":null,"method":"mining.notify","params":[', String(jobId), ",[",
    words.map((word) => word.toString()).join(","), "],", timestamp.toString(), "]}\n",
  ].join("");
  /** @type {PoolJob | undefined} */
  let jobMessage;
  await withMockPool({
    pool: {login: "kaspa:qzwallet.mom", protocol: "kaspa"},
    opt: {job: {algo: "karlsenhashv2"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(
      JSON.stringify({jsonrpc: "2.0", id: 1, error: null, result: [null, "56e0"]}) + "\n" +
      JSON.stringify({jsonrpc: "2.0", id: 2, error: null, result: true}) + "\n" + notify
    ));

    assert.ok(jobMessage);
    assert.equal(typeof jobMessage.job_id, "number");
    assert.equal(jobMessage.job_id, jobId);
    assert.equal(jobMessage.algo, "karlsenhashv2");
    assert.ok(jobMessage.blob);
    assert.equal(jobMessage.blob.slice(0, 64), expectedPrePow);
    assert.equal(jobMessage.blob.slice(64, 80), expectedTimestamp);
    assert.equal(jobMessage.blob.slice(96, 160), "00".repeat(32));
    assert.equal(poolConfig.extra_nonce, "56e0");
  });
});

test("Kaspa-family jobs reject decimal words outside uint64", async () => {
  await withMockPool({
    pool: {protocol: "kaspa"},
    opt: {job: {algo: "karlsenhashv2"}},
  }, async ({socket}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("data", Buffer.from(
      '{"id":null,"method":"mining.notify","params":["7a",[' +
      "18446744073709551616,1,2,3],4]}\n"
    ));
    assert.equal(socket.destroyed, true);
  });
});

test("Kaspa-family jobs reject timestamps outside uint64", async () => {
  await withMockPool({
    pool: {protocol: "kaspa"},
    opt: {job: {algo: "karlsenhashv2"}},
  }, async ({socket}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("data", Buffer.from(
      "{\"id\":null,\"method\":\"mining.notify\",\"params\":[\"7a\",[1,2,3,4]," +
      "18446744073709551616]}\n"
    ));
    assert.equal(socket.destroyed, true);
  });
});

test("Kaspa-family exact target conversion handles huge finite difficulty", async () => {
  await withMockPool({
    pool: {protocol: "kaspa"},
    opt: {job: {algo: "karlsenhashv2"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("data", Buffer.from(
      '{"id":null,"method":"mining.set_difficulty","params":[1e300]}\n'
    ));
    assert.equal(poolConfig["kaspa_target"], "0".repeat(64));
    assert.equal(socket.destroyed, undefined);
  });
});

test("Kaspa and HooHash targets preserve fractional difficulty exactly", async () => {
  const base = (1n << 224n) - 1n;
  /** @param {bigint} value @returns {string} */
  function targetHex(value) {
    return value.toString(16).padStart(64, "0");
  }
  /** @param {string} protocol @param {number | string} difficulty @returns {Promise<string | undefined>} */
  async function targetFor(protocol, difficulty) {
    /** @type {string | undefined} */
    let target;
    await withMockPool({
      pool: {protocol},
      opt: {job: {algo: protocol === "hoosat" ? "hoohash" : "karlsenhashv2"}},
    }, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("data", Buffer.from(JSON.stringify({
        id: null, method: "mining.set_difficulty", params: [difficulty],
      }) + "\n"));
      const kaspaTarget = poolConfig["kaspa_target"];
      target = typeof kaspaTarget === "string" ? kaspaTarget : undefined;
      assert.equal(socket.destroyed, undefined);
    });
    return target;
  }

  assert.equal(await targetFor("kaspa", 1), targetHex(base));
  assert.equal(await targetFor("kaspa", 0.01), targetHex(base * 100n));
  assert.equal(await targetFor("kaspa", 0.1), targetHex(base * 10n));
  assert.equal(await targetFor("kaspa", 0.5), targetHex(base * 2n));
  assert.equal(await targetFor("kaspa", ".5"), targetHex(base * 2n));
  assert.equal(await targetFor("kaspa", "1."), targetHex(base));
  assert.equal(await targetFor("kaspa", "1.e2"), targetHex(base / 100n));
  assert.equal(await targetFor("kaspa", 2.5), targetHex(base * 2n / 5n));
  assert.equal(await targetFor("kaspa", 0.05), targetHex(base * 20n));
  assert.equal(await targetFor("hoosat", 0.5), targetHex(base * 2n));
  assert.equal(await targetFor("kaspa", 1e-100), targetHex((1n << 256n) - 1n));
  assert.equal(await targetFor("kaspa", "1.0000000000000000001"),
    targetHex(base * 10000000000000000000n / 10000000000000000001n));
  assert.equal(helper.fullDiff2Target(1, base), targetHex(base));
  assert.equal(helper.fullDiff2Target(1), targetHex((1n << 256n) - 1n));
});

test("Kaspa-family jobs reject malformed, oversized, and nonpositive difficulties", async () => {
  for (const difficulty of [
    0, -1, "bad-difficulty", `.${"0".repeat(1001)}1`, "1".repeat(4097),
  ]) {
    await withMockPool({
      pool: {protocol: "kaspa"},
      opt: {job: {algo: "karlsenhashv2"}},
    }, async ({socket}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("data", Buffer.from(JSON.stringify({
        id: null, method: "mining.set_difficulty", params: [difficulty],
      }) + "\n"));
      assert.equal(socket.destroyed, true);
    });
  }
});

test("Kaspa-family submit uses mining.submit [wallet.worker, job_id, 0x+nonce]", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].login = "kaspa:qzwallet.mom";
  miner.global.opt.pools[0].last_job = {job_id: "7a", job_token: "token", submit_mode: "kaspa"};

  miner.messageHandler({
    thread_id: 0,
    type: "result",
    value: {
      pool_id: "0",
      worker_id: "worker",
      job_id: "7a",
      job_token: "token",
      // native nonce_to_hex(%016PRIx64): the winning 8-byte nonce big-endian; the extranonce (high
      // bytes) leads, so the pool re-parses it big-endian with no further work -- pass it through as-is.
      nonce: "56e0000000abcdef",
      hash: "00".repeat(32),
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.equal(write.json.method, "mining.submit");
  assert.equal(JSON.stringify(write.json.params), JSON.stringify([
    "kaspa:qzwallet.mom",
    "7a",
    "0x56e0000000abcdef",
  ]));
});

test("fixed Hoosat pools build Kaspa-shaped jobs for HooHash", async () => {
  /** @type {HooJob | undefined} */
  let jobMessage;
  await withMockPool({
    pool: {login: "hoosat:qzwallet.mom", protocol: "hoosat"},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.equal(writes[0].method, "mining.subscribe");
    socket.emit("data", Buffer.from([
      {jsonrpc: "2.0", id: 1, error: null, result: [null, "56e0"]},
      {jsonrpc: "2.0", id: 2, error: null, result: true},
      {id: null, method: "mining.set_difficulty", params: [4.25]},
      {id: null, method: "mining.notify", params: ["7a", [1, 2, 3, 4], 1781909733171]},
    ].map((message) => JSON.stringify(message)).join("\n") + "\n"));

    assert.ok(jobMessage);
    assert.equal(poolConfig.kaspa_difficulty, 4.25);
    assert.equal(jobMessage.submit_mode, "hoosat");
    assert.equal(jobMessage.algo, "hoohash");
    assert.equal(jobMessage.nonceoffset, 72);
    assert.ok(jobMessage.blob);
    assert.equal(jobMessage.blob.length, 160);
  });
});

test("HooHash completes subscribe, authorize, job, and submit", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();

  await withMockPool({
    pool: {login: "hoosat:qzwallet.mom", protocol: "hoosat"},
    opt: {job: {algo: "hoohash"}},
  }, async ({socket, writes, poolConfig}) => {
    miner.global.opt.pools[0] = poolConfig;
    pool.connect_pool_throttle(0, setJob);
    socket.emit("connect");
    assert.equal(writes[0]?.method, "mining.subscribe");

    socket.emit("data", Buffer.from([
      {jsonrpc: "2.0", id: 1, error: null, result: [null, "56e0"]},
      {jsonrpc: "2.0", id: 2, error: null, result: true},
      {id: null, method: "mining.set_difficulty", params: [0.01]},
      {id: null, method: "mining.notify", params: ["7a", [1, 2, 3, 4], 1781909733171]},
    ].map((message) => JSON.stringify(message)).join("\n") + "\n"));

    assert.equal(poolConfig.logged_in, true);
    assert.equal(writes[1]?.method, "mining.authorize");
    assert.deepEqual(writes[1]?.params, ["hoosat:qzwallet.mom", "x"]);
    const job = poolConfig.last_job;
    assert.ok(job);
    assert.equal(job.algo, "hoohash");
    assert.equal(job["submit_mode"], "hoosat");
    assert.equal(job.target, poolConfig["kaspa_target"]);
    assert.equal(job.target, (((1n << 224n) - 1n) * 100n).toString(16).padStart(64, "0"));

    miner.messageHandler({thread_id: 0, type: "result", value: {
      pool_id: "0", worker_id: "worker", job_id: "7a", job_token: job["job_token"],
      nonce: "56e0000000abcdef", hash: "12".repeat(32),
    }});

    const write = miner.poolWrites[0];
    assert.ok(write);
    assert.deepEqual(write.json, {
      jsonrpc: "2.0", id: 3, method: "mining.submit",
      params: ["hoosat:qzwallet.mom", "7a", "0x56e0000000abcdef", "12".repeat(32)],
    });
  });
});

test("Hoosat submit includes the exact verified HooHash", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].login = "hoosat:qzwallet.mom";
  miner.global.opt.pools[0].last_job = {job_id: "7a", job_token: "token", submit_mode: "hoosat"};
  miner.messageHandler({thread_id: 0, type: "result", value: {
    pool_id: "0", worker_id: "worker", job_id: "7a", job_token: "token",
    nonce: "56e0000000abcdef", hash: "12".repeat(32),
  }});

  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.deepEqual(JSON.parse(JSON.stringify(write.json)), {
    jsonrpc: "2.0", id: 3, method: "mining.submit",
    params: ["hoosat:qzwallet.mom", "7a", "0x56e0000000abcdef", "12".repeat(32)],
  });
});
