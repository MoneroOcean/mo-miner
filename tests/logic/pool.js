"use strict";

const s = require("./support");
const { test, assert, events, tls, opts, helper, pool, noOp, loadMinerWithStubs, withMockPool, unexpectedPoolJob, completeMiningJob, mockPoolOptions } = s;
const testGlobal = /** @type {{opt: ReturnType<typeof mockPoolOptions>}} */
  (/** @type {unknown} */ (globalThis));


function keepMiningJob(job) {
  return completeMiningJob(job);
}

test("pool debug logs redact Pearl proof bodies without changing wire messages", () => {
  const previousOpt = testGlobal.opt;
  const previousLog2 = helper.log2;
  /** @type {string[]} */
  const writes = [];
  /** @type {string[]} */
  const logs = [];
  testGlobal.opt = mockPoolOptions({
    pool: {socket: {write: (message) => {writes.push(message); return true;}}},
    opt: {log_level: 2},
  });
  helper.log2 = (message) => logs.push(message);
  try {
    pool.pool_write(0, {
      id: 3, method: "mining.submit",
      params: {job_id: "job", plain_proof: "proof-payload", proof_encoding: "gzip"},
    });
  } finally {
    helper.log2 = previousLog2;
    testGlobal.opt = previousOpt;
  }
  assert.equal(writes[0], '{"id":3,"method":"mining.submit","params":{"job_id":"job",' +
    '"plain_proof":"proof-payload","proof_encoding":"gzip"}}\n');
  assert.equal(logs.length, 1);
  assert.doesNotMatch(logs[0] || "", /proof-payload/);
  assert.match(logs[0] || "", /"plain_proof":"<redacted>"/);
});

test("fixed KawPow pools use Raven stratum subscribe and authorize", async () => {
  let jobMessage = null;
  await withMockPool({
    pool: { is_keepalive: true, login: "RVNwallet.rig01" },
    pool_time: { keepalive: 0.001 },
    opt: { job: { algo: "kawpow" } },
  }, async ({ socket, writes, poolConfig }) => {
    pool.connect_pool_throttle(0, (job) => {
      jobMessage = job;
      return job;
    });
    socket.emit("connect");
    assert.equal(writes[0].method, "mining.subscribe");

    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":["0a1fa6c0","e0"]}\n' +
      '{"jsonrpc":"2.0","id":2,"error":null,"result":true}\n' +
      '{"method":"mining.notify","params":["203d","' + "00".repeat(32) + '","' + "11".repeat(32) + '","' +
      "00000000ffff0000000000000000000000000000000000000000000000000000" +
      '",true,4390582,"1b01e5f2"],"id":null,"jsonrpc":"2.0"}\n'
    ));

    assert.equal(writes[1].method, "mining.authorize");
    assert.deepEqual(writes[1].params, ["RVNwallet.rig01", "x"]);
    assert.equal(poolConfig.extra_nonce, "0a1fa6c0");
    assert.equal(jobMessage.job_id, "203d");
    assert.equal(jobMessage.blob, "00".repeat(32) + "00000000c0a61f0a");
    assert.equal(jobMessage.nonce, "0a1fa6c000000000");
    assert.equal(jobMessage.nicehash_mask, "ffffffff00000000");
    assert.equal(writes.length, 2);
  });
});

test("fixed Etchash pools use Eth stratum notify jobs", async () => {
  let jobMessage = null;
  const headerHash = "22".repeat(32);
  const seedHash = "11".repeat(32);
  await withMockPool({
    pool: { is_keepalive: true, login: "0xwallet.worker" },
    opt: { job: { algo: "etchash" } },
    pool_time: { keepalive: 0.001, first_job_wait: 0.001 },
  }, async ({ socket, writes, poolConfig }) => {
    pool.connect_pool_throttle(0, (job) => {
      jobMessage = job;
      return job;
    });
    socket.emit("connect");
    assert.equal(writes[0].method, "mining.subscribe");

    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":[[["mining.notify","1"],"080c"],"080c",6]}\n' +
      '{"jsonrpc":"2.0","method":"mining.set_difficulty","params":[1]}\n' +
      '{"jsonrpc":"2.0","id":2,"error":null,"result":true}\n' +
      '{"method":"mining.notify","params":["203d","' + seedHash + '","' + headerHash + '",true],"id":null,"jsonrpc":"2.0"}\n'
    ));

    assert.equal(writes[1].method, "mining.authorize");
    assert.deepEqual(writes[1].params, ["0xwallet.worker", "x"]);
    assert.equal(poolConfig.extra_nonce, "080c");
    assert.equal(jobMessage.algo, "etchash");
    assert.equal(jobMessage.job_id, "203d");
    assert.equal(jobMessage.seed_hash, seedHash);
    assert.equal(jobMessage.header_hash, headerHash);
    assert.equal(jobMessage.blob, headerHash + "0000000000000c08");
    assert.equal(jobMessage.nonce, "080c000000000000");
    assert.equal(jobMessage.nicehash_mask, "ffff000000000000");
    assert.equal(jobMessage.target, "00000000ffff0000000000000000000000000000000000000000000000000000");
    assert.equal(writes.length, 2);
  });
});

test("MO login-inferred Etchash ignores stale keepalive response", async () => {
  let jobMessage = null;
  const headerHash = "22".repeat(32);
  const seedHash = "11".repeat(32);
  await withMockPool({
    pool: { is_keepalive: true, pass: "x~etchash" },
    pool_time: { keepalive: 60, first_job_wait: 0.001 },
  }, async ({ socket, writes, poolConfig }) => {
    pool.connect_pool_throttle(0, (job) => {
      jobMessage = job;
      return job;
    });
    socket.emit("connect");
    assert.equal(writes[0].method, "login");
    assert.notEqual(poolConfig.keepalive, null);

    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      error: null,
      result: { id: "worker", algo: "etchash", extra_nonce: "080c" },
    }) + "\n"));

    assert.equal(poolConfig.logged_in, true);
    assert.equal(poolConfig.inferred_protocol, "eth");
    assert.equal(poolConfig.keepalive, null);

    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0",
      id: 2,
      error: { message: "Authorization rejected" },
      result: false,
    }) + "\n"));

    assert.equal(poolConfig.logged_in, true);

    socket.emit("data", Buffer.from(
      '{"method":"mining.notify","params":["203d","' + seedHash + '","' + headerHash + '",true],"algo":"etchash","id":null,"jsonrpc":"2.0"}\n'
    ));

    assert.equal(jobMessage.job_id, "203d");
    assert.equal(jobMessage.seed_hash, seedHash);
    assert.equal(jobMessage.header_hash, headerHash);
  });
});

test("fixed Etchash pools can use ethproxy work jobs", async () => {
  let jobMessage = null;
  const headerHash = "22".repeat(32);
  const seedHash = "11".repeat(32);
  const target = "000000007fffffffffffffffffffffffffffffffffffffffffffffffffffffff";
  await withMockPool({
    pool: { is_keepalive: true, login: "0xwallet.worker", protocol: "ethproxy" },
    opt: { job: { algo: "etchash" } },
    pool_time: { keepalive: 0.001, first_job_wait: 0.001 },
  }, async ({ socket, writes }) => {
    pool.connect_pool_throttle(0, (job) => {
      jobMessage = job;
      return job;
    });
    socket.emit("connect");
    assert.equal(writes[0].method, "eth_submitLogin");
    assert.deepEqual(writes[0].params, ["0xwallet.worker", "x"]);

    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":true}\n' +
      '{"id":0,"jsonrpc":"2.0","result":["0x' + headerHash + '","0x' + seedHash + '","0x' + target + '","0x1788f2d"],"algo":"etchash"}\n'
    ));

    assert.equal(jobMessage.algo, "etchash");
    assert.equal(jobMessage.job_id, headerHash);
    assert.equal(jobMessage.seed_hash, seedHash);
    assert.equal(jobMessage.header_hash, headerHash);
    assert.equal(jobMessage.blob, headerHash + "0000000000000000");
    assert.equal(jobMessage.nonce, "0000000000000000");
    assert.equal(jobMessage.nicehash_mask, "0000000000000000");
    assert.equal(jobMessage.height, 24678189);
    assert.equal(jobMessage.target, target);
    assert.equal(writes.length, 1);
  });
});

test("fixed Autolykos2 pools use Ergo stratum notify jobs", async () => {
  let jobMessage = null;
  const headerHash = "54".repeat(32);
  const bound = "7067388259113537318333190002971674063283542741642755394446115914399301849";
  await withMockPool({
    pool: { is_keepalive: true, login: "9ergwallet.worker" },
    opt: { job: { algo: "autolykos2" } },
    pool_time: { keepalive: 0.001, first_job_wait: 0.001 },
  }, async ({ socket, writes, poolConfig }) => {
    pool.connect_pool_throttle(0, (job) => {
      jobMessage = job;
      return job;
    });
    socket.emit("connect");
    assert.equal(writes[0].method, "mining.subscribe");

    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":[[["mining.notify","1"],"080c"],"080c",6]}\n' +
      '{"jsonrpc":"2.0","id":2,"error":null,"result":true}\n' +
      '{"method":"mining.notify","params":["203d",614400,"' + headerHash + '","","",2,"' + bound + '","",true],"algo":"autolykos2","id":null,"jsonrpc":"2.0"}\n'
    ));

    assert.equal(writes[1].method, "mining.authorize");
    assert.deepEqual(writes[1].params, ["9ergwallet.worker", "x"]);
    assert.equal(poolConfig.extra_nonce, "080c");
    assert.equal(poolConfig.extra_nonce2_size, 6);
    assert.equal(jobMessage.algo, "autolykos2");
    assert.equal(jobMessage.job_id, "203d");
    assert.equal(jobMessage.header_hash, headerHash);
    assert.equal(jobMessage.blob, headerHash + "0000000000000c08");
    assert.equal(jobMessage.nonce, "080c000000000000");
    assert.equal(jobMessage.nicehash_mask, "ffff000000000000");
    assert.equal(jobMessage.height, 614400);
    assert.equal(jobMessage.ntime, "");
    assert.equal(jobMessage.target, "0003fffffffffffffffffffffffffffffffaeabb739abd2280eeff497a3340d9");
    assert.equal(writes.length, 2);
  });
});

test("stale pool timeout does not destroy a replacement socket", async () => {
  const staleSocket = new events.EventEmitter();
  const replacementSocket = new events.EventEmitter();
  replacementSocket.destroy = function() { this.destroyed = true; };

  await withMockPool({
    socket: staleSocket,
    pool_time: { first_job_wait: 0.001 },
  }, async ({ poolConfig }) => {
    pool.connect_pool_throttle(0, noOp);
    poolConfig.socket = replacementSocket;
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(replacementSocket.destroyed, undefined);
  });
});

test("TLS pools verify certificates only when explicitly enabled", async () => {
  const originalConnect = tls.connect;
  const previousOpt = global.opt;
  const optionsSeen = [];
  assert.equal(opts.pool_create("pool.example", 443, true, "user").tls_verify, false);
  tls.connect = function(_port, _host, options) {
    optionsSeen.push(options);
    const socket = new events.EventEmitter();
    socket.write = noOp;
    socket.destroy = noOp;
    return socket;
  };
  global.opt = {
    log_level: 0,
    pools: [{
      url: "pool.example",
      port: 443,
      is_tls: true,
      is_keepalive: false,
      socket: null,
      keepalive: null,
      last_job: null,
      last_connect_time: 0,
    }],
    pool_ids: { active: 0, primary: 0, donate: null },
    pool_time: { first_job_wait: 0.001, connect_throttle: 0, close_wait: 60, keepalive: 60 },
    algo_params: {},
  };

  try {
    pool.connect_pool_throttle(0, noOp);
    global.opt.pools[0].socket = null;
    global.opt.pools[0].tls_verify = true;
    pool.connect_pool_throttle(0, noOp);
    global.opt.pools[0].socket = null;
    global.opt.pools[0].tls_verify = false;
    pool.connect_pool_throttle(0, noOp);
    global.opt.pools[0].socket = null;
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(optionsSeen[0].rejectUnauthorized, false);
    assert.equal(optionsSeen[1].rejectUnauthorized, true);
    assert.equal(optionsSeen[2].rejectUnauthorized, false);
  } finally {
    tls.connect = originalConnect;
    global.opt = previousOpt;
  }
});

test("malformed pool job data closes the pool instead of throwing", async () => {
  await withMockPool({
    switchPool: true,
    pool: { logged_in: true },
    pool_time: { first_job_wait: 0.001 },
  }, async ({ socket, switched }) => {
    pool.connect_pool_throttle(0, () => ({ algo: "cn/0" }));
    assert.doesNotThrow(() => {
      socket.emit("data", Buffer.from('{"method":"job","params":{"target":"zz"}}\n'));
    });
    assert.equal(socket.destroyed, true);
    assert.equal(global.opt.pools[0].socket, null);
    assert.equal(switched(), true);
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
});

test("errored login response with job does not start mining", async () => {
  await withMockPool({}, async ({ socket, poolConfig }) => {
    let jobStarted = false;
    pool.connect_pool_throttle(0, () => { jobStarted = true; });
    socket.emit("data", Buffer.from(JSON.stringify({
      id: 1,
      jsonrpc: "2.0",
      error: { message: "No double login is allowed" },
      result: {
        id: "worker",
        job: {
          algo: "autolykos2",
          blob: "00".repeat(32),
          target: "ff",
          height: 1,
        },
      },
    }) + "\n"));

    assert.equal(jobStarted, false);
    assert.equal(poolConfig.last_job, null);
  });
});

test("job notification before login success does not start mining", async () => {
  await withMockPool({}, async ({ socket, poolConfig }) => {
    let jobStarted = false;
    pool.connect_pool_throttle(0, () => { jobStarted = true; });
    socket.emit("data", Buffer.from(
      JSON.stringify({
        method: "job",
        params: {
          algo: "autolykos2",
          blob: "00".repeat(32),
          target: "ff",
          height: 1,
        },
      }) + "\n" +
      JSON.stringify({
        id: 1,
        jsonrpc: "2.0",
        error: { message: "No double login is allowed" },
        result: false,
      }) + "\n"
    ));

    assert.equal(jobStarted, false);
    assert.equal(poolConfig.last_job, null);
    assert.equal(poolConfig.logged_in, false);
  });
});

test("login job inherits height from login result metadata", async () => {
  let jobMessage = null;
  await withMockPool({}, async ({ socket }) => {
    pool.connect_pool_throttle(0, (job) => {
      jobMessage = job;
      return job;
    });
    socket.emit("data", Buffer.from(JSON.stringify({
      id: 1,
      jsonrpc: "2.0",
      error: null,
      result: {
        id: "worker",
        height: 1799914,
        job: {
          algo: "etchash",
          blob: "00".repeat(32),
          seed_hash: "11".repeat(32),
          target: "00000000ffff0000000000000000000000000000000000000000000000000000",
        },
      },
    }) + "\n"));

    assert.equal(jobMessage.height, 1799914);
  });
});

test("oversized pool line buffer closes the pool", async () => {
  await withMockPool({
    switchPool: true,
    pool_time: { first_job_wait: 0.001 },
  }, async ({ socket, switched }) => {
    pool.connect_pool_throttle(0, noOp);
    socket.emit("data", Buffer.alloc(1024 * 1024 + 1, "a"));
    assert.equal(socket.destroyed, true);
    assert.equal(global.opt.pools[0].socket, null);
    assert.equal(switched(), true);
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
});

test("KawPow login response id is reused for later notify jobs", async () => {
  let jobMessage = null;
  await withMockPool({
    pool: { pass: "~kawpow" },
    pool_time: { first_job_wait: 0.001 },
  }, async ({ socket, poolConfig }) => {
    pool.connect_pool_throttle(0, (job) => {
      jobMessage = job;
      return job;
    });
    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":{"id":"5122080","algo":"kawpow","extra_nonce":"ff81"}}\n' +
      '{"method":"mining.notify","params":["203d","' + "00".repeat(32) + '","' + "11".repeat(32) + '","' +
      "0000005eb993eef1b05c00000000000000000000000000000000000000000000" +
      '",true,4390582,"1b01e5f2"],"algo":"kawpow","id":null,"jsonrpc":"2.0"}\n'
    ));
    assert.equal(poolConfig.worker_id, "5122080");
    assert.equal(poolConfig.extra_nonce, "ff81");
    assert.equal(jobMessage.job_id, "203d");
    assert.equal(jobMessage.blob, "00".repeat(32) + "00000000000081ff");
    assert.equal(jobMessage.nonce, "ff81000000000000");
    assert.equal(jobMessage.nicehash_mask, "ffff000000000000");
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
});

test("pool share response false is counted as rejected", async () => {
  await withMockPool({}, async ({ socket, poolConfig }) => {
    pool.connect_pool_throttle(0, noOp);
    socket.emit("data", Buffer.from('{"jsonrpc":"2.0","id":3,"error":null,"result":false}\n'));
    assert.equal(poolConfig.good_shares, 0);
    assert.equal(poolConfig.bad_shares, 1);
  });
});

test("non-C29 pool jobs preserve provided blob_hex and nonceoffset", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  assert.equal(typeof setJob, "function");

  setJob({
    algo: "cn/0",
    blob_hex: "abcd",
    nonceoffset: 7,
    difficulty: 1,
    id: "worker",
    job_id: "job",
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.equal(jobMessage.job.blob_hex, "abcd");
  assert.equal(jobMessage.job.nonceoffset, 7);
});


test("fixed Octopus pools use Conflux subscribe-only stratum jobs", async () => {
  /** @type {PoolJob | undefined} */
  let jobMessage;
  const headerHash = "22".repeat(32);
  const boundary = "00000000ffff" + "00".repeat(26);
  await withMockPool({
    pool: {is_keepalive: true, login: "cfx:wallet.rig"},
    opt: {job: {algo: "octopus"}},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return keepMiningJob(job);
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.deepEqual(writes[0], {
      jsonrpc: "2.0", id: 1, method: "mining.subscribe", params: ["cfx:wallet.rig", ""],
    });

    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":true}\n' +
      '{"jsonrpc":"2.0","method":"mining.notify","params":["job1","152521905","0x' +
      headerHash + '","0x' + boundary + '"]}\n'
    ));

    assert.ok(jobMessage);
    assert.equal(poolConfig.logged_in, true);
    assert.equal(jobMessage.submit_mode, "conflux");
    assert.equal(jobMessage.algo, "octopus");
    assert.equal(jobMessage.job_id, "job1");
    assert.equal(jobMessage.height, 152521905);
    assert.equal(jobMessage.header_hash, headerHash);
    assert.equal(jobMessage.blob, headerHash + "0000000000000000");
    assert.equal(jobMessage.target, boundary);
    assert.equal(writes.length, 1);
  });
});

test("Conflux decimal notify targets reach the job as full-width hex", async () => {
  /** @type {PoolJob | undefined} */
  let jobMessage;
  const headerHash = "33".repeat(32);
  const decimalTarget = "1".repeat(64);
  const expectedTarget = BigInt(decimalTarget).toString(16).padStart(64, "0");
  await withMockPool({
    pool: {is_keepalive: true, login: "cfx:wallet.rig"},
    opt: {job: {algo: "octopus"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return keepMiningJob(job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(
      JSON.stringify({jsonrpc: "2.0", id: 1, error: null, result: true}) + "\n" +
      JSON.stringify({
        jsonrpc: "2.0", method: "mining.notify",
        params: ["decimal-job", "123", "0x" + headerHash, decimalTarget],
      }) + "\n"
    ));

    assert.ok(jobMessage);
    assert.equal(poolConfig.logged_in, true);
    assert.equal(jobMessage.submit_mode, "conflux");
    assert.equal(jobMessage.target, expectedTarget);
    assert.equal(jobMessage.target.length, 64);
    assert.match(jobMessage.target, /^[0-9a-f]{64}$/);
  });
});

test("Conflux prefixed short hex notify targets are left-padded", async () => {
  /** @type {PoolJob | undefined} */
  let jobMessage;
  const shortHexTarget = "0x" + "ab".repeat(28);
  const expectedTarget = "0".repeat(8) + "ab".repeat(28);
  await withMockPool({
    pool: {is_keepalive: true, login: "cfx:wallet.rig"},
    opt: {job: {algo: "octopus"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return keepMiningJob(job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(
      JSON.stringify({jsonrpc: "2.0", id: 1, error: null, result: true}) + "\n" +
      JSON.stringify({
        jsonrpc: "2.0", method: "mining.notify",
        params: ["short-hex-job", "123", "0x" + "55".repeat(32), shortHexTarget],
      }) + "\n"
    ));

    assert.ok(jobMessage);
    assert.equal(poolConfig.logged_in, true);
    assert.equal(jobMessage.target, expectedTarget);
    assert.equal(jobMessage.target.length, 64);
  });
});

test("Conflux invalid notify targets are rejected before dispatch", async () => {
  for (const target of [
    "0", "01", "not-decimal", (1n << 256n).toString(),
    "a".repeat(64), "0x0", "0x" + "00".repeat(32),
  ]) {
    let dispatched = false;
    await withMockPool({
      pool: {is_keepalive: true, login: "cfx:wallet.rig"},
      opt: {job: {algo: "octopus"}},
    }, async ({socket, switched}) => {
      pool.connect_pool_throttle(0, /** @param {PoolJob} _job */ (_job) => {
        dispatched = true;
        return keepMiningJob(_job);
      });
      socket.emit("connect");
      socket.emit("data", Buffer.from(
        JSON.stringify({jsonrpc: "2.0", id: 1, error: null, result: true}) + "\n" +
        JSON.stringify({
          jsonrpc: "2.0", method: "mining.notify",
          params: ["invalid-job", "123", "0x" + "44".repeat(32), target],
        }) + "\n"
      ));

      assert.equal(dispatched, false, target);
      assert.equal(socket.destroyed, true, target);
      assert.equal(switched(), true, target);
    });
  }
});

test("Conflux overlong prefixed hex notify targets are rejected before dispatch", async () => {
  let dispatched = false;
  await withMockPool({
    pool: {is_keepalive: true, login: "cfx:wallet.rig"},
    opt: {job: {algo: "octopus"}},
  }, async ({socket, switched}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} _job */ (_job) => {
      dispatched = true;
      return keepMiningJob(_job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(
      JSON.stringify({jsonrpc: "2.0", id: 1, error: null, result: true}) + "\n" +
      JSON.stringify({
        jsonrpc: "2.0", method: "mining.notify",
        params: ["overlong-hex-job", "123", "0x" + "66".repeat(32), "0x" + "ab".repeat(33)],
      }) + "\n"
    ));

    assert.equal(dispatched, false);
    assert.equal(socket.destroyed, true);
    assert.equal(switched(), true);
  });
});

test("malformed Conflux jobs log a redacted parser reason", async () => {
  const previousLogError = helper.log_err;
  /** @type {string[]} */
  const logs = [];
  helper.log_err = (message) => logs.push(message);
  try {
    await withMockPool({
      pool: {login: "cfx:secret-wallet.worker", pass: "secret-pass"},
      opt: {job: {algo: "octopus"}},
    }, async ({socket, switched}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("connect");
      socket.emit("data", Buffer.from(
        JSON.stringify({jsonrpc: "2.0", id: 1, error: null, result: true}) + "\n" +
        JSON.stringify({
          jsonrpc: "2.0", method: "mining.notify",
          params: ["malformed-job", "123", "0x" + "44".repeat(32), "0"],
        }) + "\n"
      ));
      assert.equal(socket.destroyed, true);
      assert.equal(switched(), true);
    });
  } finally {
    helper.log_err = previousLogError;
  }
  assert.equal(logs.length, 1);
  assert.ok(logs[0]);
  assert.match(logs[0], /Can't process message from the pool: "Conflux target must be positive"/);
  assert.doesNotMatch(logs[0], /secret-wallet|secret-pass/);
});

test("Conflux share result arrays require an explicit true", async () => {
  const previousLogError = helper.log_err;
  /** @type {string[]} */
  const logs = [];
  helper.log_err = (message) => logs.push(message);
  const responses = [
    {result: true, good: 1, bad: 0},
    {result: false, good: 0, bad: 1},
    {result: [true], good: 1, bad: 0},
    {result: [false, "array rejection reason"], good: 0, bad: 1},
    {result: [], good: 0, bad: 1},
    {result: [1], good: 0, bad: 1},
    {result: [true], error: {code: 31, message: "rpc rejection reason"}, good: 0, bad: 1},
    {
      result: [false, "array-reason-marker cfx-test-user.worker cfx-test-password"],
      pool: {login: "cfx-test-user.worker", pass: "cfx-test-password"}, good: 0, bad: 1,
    },
  ];
  try {
    for (const response of responses) {
      await withMockPool({pool: {protocol: "conflux", ...response.pool}}, async ({socket, poolConfig}) => {
        pool.connect_pool_throttle(0, unexpectedPoolJob);
        poolConfig["pending_submit_count"] = 1;
        socket.emit("data", Buffer.from(JSON.stringify({
          jsonrpc: "2.0", id: 3, error: response.error || null, result: response.result,
        }) + "\n"));
        assert.equal(poolConfig.good_shares, response.good);
        assert.equal(poolConfig.bad_shares, response.bad);
      });
    }
  } finally {
    helper.log_err = previousLogError;
  }
  assert.ok(logs.some((message) => message.includes("array rejection reason")));
  const rpcErrors = logs.filter((message) => message.includes("rpc rejection reason"));
  assert.equal(rpcErrors.length, 1);
  const rpcError = rpcErrors[0];
  assert.ok(rpcError);
  assert.equal(rpcError.includes("array rejection reason"), false);
  const redactedArrayReason = logs.find((message) => message.includes("array-reason-marker"));
  assert.ok(redactedArrayReason);
  assert.equal(redactedArrayReason.includes("cfx-test-user.worker"), false);
  assert.equal(redactedArrayReason.includes("cfx-test-password"), false);
  assert.match(redactedArrayReason, /<redacted>/);
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

test("pass-only Kaspa-family metadata selects the matching jobs and submit mode", async () => {
  for (const {algo, protocol} of [
    {algo: "karlsenhashv2", protocol: "kaspa"},
    {algo: "walahash", protocol: "kaspa"},
    {algo: "hoohash", protocol: "hoosat"},
  ]) {
    /** @type {PoolJob | undefined} */
    let jobMessage;
    await withMockPool({pool: {pass: `x~${algo}`, use_subscribe: false}}, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
        jobMessage = job;
        return completeMiningJob(job);
      });
      socket.emit("connect");
      socket.emit("data", Buffer.from(JSON.stringify({
        jsonrpc: "2.0", id: 1, error: null, result: {id: "worker"},
      }) + "\n"));
      socket.emit("data", Buffer.from([
        {id: null, method: "mining.set_difficulty", params: [4.25]},
        {id: null, method: "mining.notify", params: ["job", [1, 2, 3, 4], 1781909733171]},
      ].map((message) => JSON.stringify(message)).join("\n") + "\n"));
      assert.equal(poolConfig.inferred_protocol, protocol, algo);
      assert.equal(jobMessage?.submit_mode, protocol, algo);
      assert.equal(jobMessage?.algo, algo);
    });
  }
});

test("login job inherits PearlHash certificate version from result metadata", async () => {
  /** @type {PoolJob[]} */
  const jobs = [];
  await withMockPool({}, async ({socket}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobs.push(job);
      return keepMiningJob(job);
    });
    socket.emit("data", Buffer.from(JSON.stringify({
      id: 1,
      jsonrpc: "2.0",
      error: null,
      result: {
        pearlhash_cert_version: 3,
        job: {
          algo: "pearlhash",
          blob: "00".repeat(76),
          job_id: "pearl-login-job",
          pearlhash_k: 4096,
          pearlhash_rank: 256,
          target: "1".padStart(64, "0"),
        },
      },
    }) + "\n"));

    assert.equal(jobs.length, 1);
    assert.equal(jobs[0]?.pearlhash_cert_version, 3);
  });
});

test("native PearlHash login jobs use the Pearl notification parser", async () => {
  /** @type {PoolJob[]} */
  const jobs = [];
  const header = "12".repeat(76);
  const baseTarget = "0f".repeat(32);
  await withMockPool({}, async ({socket}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobs.push(job);
      return keepMiningJob(job);
    });
    socket.emit("data", Buffer.from(JSON.stringify({
      id: 1,
      jsonrpc: "2.0",
      error: null,
      result: {
        id: "worker",
        algo: "pearlhash",
        job: {
          algo: "pearlhash",
          job_id: "proxy-pearl-login",
          header,
          target: baseTarget,
          cert_version: 3,
          height: 42,
        },
      },
    }) + "\n"));
  });

  assert.equal(jobs.length, 1);
  assert.equal(jobs[0]?.algo, "pearlhash");
  assert.equal(jobs[0]?.blob, header);
  assert.equal(jobs[0]?.pearlhash_base_target, baseTarget);
  assert.equal(jobs[0]?.pearlhash_cert_version, 3);
  assert.equal(jobs[0]?.height, 42);
  assert.equal(jobs[0]?.submit_mode, "pearlhash");
  assert.equal(jobs[0]?.target, undefined);
});

test("PearlHash login inference persists for later untagged proxy jobs", async () => {
  /** @type {PoolJob[]} */
  const jobs = [];
  const header = "12".repeat(76);
  const target = "0f".repeat(32);
  await withMockPool({pool: {protocol: "login"}}, async ({socket}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobs.push(job);
      return keepMiningJob(job);
    });
    socket.emit("data", Buffer.from([
      {
        id: 1,
        jsonrpc: "2.0",
        error: null,
        result: {
          id: "worker",
          algo: "pearlhash",
          job: {job_id: "login", header, target, cert_version: 3},
        },
      },
      {
        id: null,
        jsonrpc: "2.0",
        method: "job",
        params: {job_id: "update", header, target, cert_version: 3},
      },
    ].map((message) => JSON.stringify(message)).join("\n") + "\n"));
  });

  assert.deepEqual(jobs.map((job) => job.job_id), ["login", "update"]);
  assert.deepEqual(jobs.map((job) => job.submit_mode), ["pearlhash", "pearlhash"]);
});

test("PearlHash subscribe pools distinguish base and final jackpot targets", async () => {
  const target = "00000000d1b71758e219652bd3c36113404ea4a8c154c985f06f694467381d7d";
  for (const format of ["default", "base", "jackpot"]) {
    await withMockPool({
      pool: {
        protocol: "pearlhash", use_subscribe: true,
        ...(format === "default" ? {} : {pearlhash_target_format: format}),
      },
      opt: {job: {algo: "pearlhash"}},
    }, async ({socket, poolConfig}) => {
      /** @type {PoolJob[]} */
      const jobs = [];
      pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
        jobs.push(job);
        return keepMiningJob({...job, pearlhash_k: 4096, pearlhash_rank: 256});
      });
      socket.emit("connect");
      socket.emit("data", Buffer.from('{"id":1,"result":true,"error":null}\n'));
      socket.emit("data", Buffer.from('{"id":2,"result":true,"error":null}\n'));
      socket.emit("data", Buffer.from(JSON.stringify({id: null, method: "mining.notify", params: {
        header: "00".repeat(76), job_id: "target", target: "0x" + target,
      }}) + "\n"));
      assert.equal(jobs.length, 1);
      assert.equal(jobs[0]?.target, format === "jackpot" ? target : undefined);
      assert.equal(jobs[0]?.pearlhash_base_target, format === "jackpot" ? undefined : target);
      assert.notEqual(socket.destroyed, true);
      assert.equal(poolConfig.logged_in, true);
      assert.equal(poolConfig.bad_shares, 0);
    });
  }
});

test("Pearl rejects malformed final jackpot targets before accepting a job", async () => {
  for (const target of [null, 0, {}, "", "0x", "not-hex", "f".repeat(65)]) {
    await withMockPool({
      pool: {protocol: "pearlhash", use_subscribe: true, logged_in: true, pearlhash_target_format: "jackpot"},
      opt: {job: {algo: "pearlhash"}},
    }, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("data", Buffer.from(JSON.stringify({id: null, method: "mining.notify", params: {
        header: "00".repeat(76), job_id: "invalid", target,
      }}) + "\n"));
      assert.equal(poolConfig.last_job, null);
      assert.equal(socket.destroyed, true);
    });
  }
});
