"use strict";

const s = require("./support");
const { test, assert, events, tls, opts, pool, noOp, loadMinerWithStubs, withMockPool } = s;

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
