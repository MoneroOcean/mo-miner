"use strict";

const s = require("./support");
const {
  test, events, tls, opts, helper, pool, noOp, loadMinerWithStubs,
  mockPoolConfig, mockPoolOptions, withMockPool, unexpectedPoolJob,
} = s;
/** @type {typeof import("node:assert/strict")} */
const assert = s.assert;
/** @type {(job: PoolJob) => MiningJob} */
const completeMiningJob = s.completeMiningJob;
const testGlobal = /** @type {{opt: ReturnType<typeof mockPoolOptions>}} */
  (/** @type {unknown} */ (globalThis));
/** @typedef {import("node:events").EventEmitter & {
 *   write?: (message: string) => void, destroy?: () => void, destroyed?: boolean
 * }} PoolMockSocket */

/** @param {PoolJob} job @returns {MiningJob} */
function keepMiningJob(job) {
  return completeMiningJob(job);
}

test("pool debug logs redact credentials without changing wire messages", () => {
  const previousOpt = testGlobal.opt;
  const previousLog2 = helper.log2;
  /** @type {string[]} */
  const writes = [];
  /** @type {string[]} */
  const logs = [];
  testGlobal.opt = mockPoolOptions({
    pool: {
      login: "wallet.worker", pass: "top-secret",
      socket: {
        write: (message) => {
          writes.push(message);
          return true;
        },
      },
    },
    opt: {log_level: 2},
  });
  helper.log2 = (message) => logs.push(message);
  try {
    pool.pool_write(0, {
      id: 2, method: "mining.authorize",
      params: ["wallet", "worker", "top-secret", "wallet.worker rejected top-secret\u001b[31m"],
    });
  } finally {
    helper.log2 = previousLog2;
    testGlobal.opt = previousOpt;
  }
  assert.equal(writes[0],
    '{"id":2,"method":"mining.authorize","params":["wallet","worker","top-secret",' +
    '"wallet.worker rejected top-secret\\u001b[31m"]}\n');
  assert.equal(logs.length, 1);
  assert.ok(logs[0]);
  assert.doesNotMatch(logs[0], /wallet|top-secret/);
  assert.equal(logs[0].includes(String.fromCharCode(27)), false);
  assert.match(logs[0], /<redacted>/);
});

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

test("pool debug logs redact credentials embedded in object keys", async () => {
  const previousLog2 = helper.log2;
  /** @type {string[]} */
  const logs = [];
  const password = 'top-"secret"\n';
  const hostileKey = `prl.wallet.rig rejected ${password}`;
  const message = {
    id: 1,
    result: {
      [hostileKey]: {
        [password]: "prl.wallet named-worker rig rejected",
      },
    },
  };
  helper.log2 = (value) => logs.push(value);
  try {
    await withMockPool({
      pool: {login: "prl.wallet.rig", worker: "named-worker", pass: password},
      opt: {log_level: 2},
    }, async ({socket}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("connect");
      socket.emit("data", Buffer.from(JSON.stringify(message) + "\n"));
    });
  } finally {
    helper.log2 = previousLog2;
  }
  const received = logs.find((value) => value.includes("Got from the pool:"));
  assert.ok(received);
  assert.doesNotMatch(received, /prl|wallet|rig|top|secret|named-worker/);
  assert.match(received, /<redacted>/);
  assert.ok(Object.hasOwn(message.result, hostileKey));
  assert.ok(Object.hasOwn(message.result[hostileKey], password));
});

test("pool debug logging tolerates deeply nested input", async () => {
  const previousLog2 = helper.log2;
  /** @type {string[]} */
  const logs = [];
  const depth = 10000;
  const message = '{"next":'.repeat(depth) + "null" + "}".repeat(depth) + "\n";
  helper.log2 = (value) => logs.push(value);
  try {
    await withMockPool({opt: {log_level: 2}}, async ({socket}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("connect");
      assert.doesNotThrow(() => socket.emit("data", Buffer.from(message)));
    });
  } finally {
    helper.log2 = previousLog2;
  }
  assert.ok(logs.some((value) => value.includes("<unprintable pool JSON>")));
});

test("short credentials are redacted only at text boundaries", () => {
  const previousOpt = testGlobal.opt;
  const previousLog2 = helper.log2;
  /** @type {string[]} */
  const logs = [];
  testGlobal.opt = mockPoolOptions({
    pool: {login: "u", pass: "x", socket: {write: noOp}},
    opt: {log_level: 2},
  });
  helper.log2 = (message) => logs.push(message);
  try {
    pool.pool_write(0, {id: 1, error: "rejected x", note: "extranonce"});
  } finally {
    helper.log2 = previousLog2;
    testGlobal.opt = previousOpt;
  }
  assert.ok(logs[0]);
  assert.doesNotMatch(logs[0], /rejected x/);
  assert.match(logs[0], /rejected <redacted>/);
  assert.match(logs[0], /extranonce/);
});

test("backup reactivation preserves numeric pool IDs", () => {
  const previousOpt = testGlobal.opt;
  const restored = {algo: "cn/0", dev: "cpu", job_id: "backup"};
  testGlobal.opt = mockPoolOptions();
  testGlobal.opt.pools.push(mockPoolConfig({last_job: restored}));
  /** @type {PoolJob | null} */
  let selected = null;
  try {
    pool.switch_pool(0, /** @param {PoolJob} job */ (job) => {
      selected = job;
      assert.ok(typeof job.algo === "string");
      return keepMiningJob(job);
    });
    assert.equal(testGlobal.opt.pool_ids.active, 1);
    assert.equal(selected, restored);
  } finally {
    testGlobal.opt = previousOpt;
  }
});

test("pool failover skips a donation pool at any index", () => {
  const previousOpt = testGlobal.opt;
  testGlobal.opt = mockPoolOptions({
    opt: {pool_ids: {active: 0, primary: 0, donate: 1}},
  });
  testGlobal.opt.pools.push(
    mockPoolConfig(),
    mockPoolConfig({socket: {}}),
  );
  try {
    pool.switch_pool(0, /** @param {PoolJob} job */ (job) => {
      assert.ok(typeof job.algo === "string");
      return keepMiningJob(job);
    });
    assert.equal(testGlobal.opt.pool_ids.active, 2);
  } finally {
    testGlobal.opt = previousOpt;
  }
});

test("donation jobs outside the advertised capability map are refused", async () => {
  await withMockPool({
    pool: {donation_until: Infinity, algo_params: {"rx/0": {dev: "cpu", perf: 1}}},
    pool_time: {close_wait: 0.001},
    opt: {
      pool_ids: {active: 1, primary: 1, donate: 0},
      job: {algo: null},
      algo_params: {"rx/0": {dev: "cpu", perf: 1}},
    },
  }, async ({socket}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    socket.emit("data", Buffer.from(
      '{"id":1,"error":null,"result":{"id":"w","job":' +
      '{"algo":"etchash","blob":"00","job_id":"1","target":"01"}}}\n'
    ));
    assert.equal(socket.destroyed, true);
  });
});

test("pool jobs canonicalize alias and configured algorithms before target work", async () => {
  const kawpowTarget = "00000000117edbe19772d0000000000000000000000000000000000000000000";
  for (const {job, configuredAlgo, expectedAlgo} of [
    {
      job: {algo: "kawpow1", blob: "00", job_id: "kawpow-alias", target: kawpowTarget},
      configuredAlgo: "kawpow", expectedAlgo: "kawpow",
    },
    {
      job: {algo: "cuckaroo", blob: "00", job_id: "cuckaroo-alias", target: "ffffffff"},
      configuredAlgo: null, expectedAlgo: "c29",
    },
    {
      job: {blob: "00", job_id: "fallback", target: kawpowTarget},
      configuredAlgo: "kawpow", expectedAlgo: "kawpow",
    },
  ]) {
    /** @type {PoolJob | undefined} */
    let received;
    await withMockPool({
      pool: {logged_in: true},
      opt: {job: {algo: configuredAlgo}},
    }, async ({socket}) => {
      pool.connect_pool_throttle(0, /** @param {PoolJob} next */ (next) => {
        received = next;
        return keepMiningJob(next);
      });
      socket.emit("data", Buffer.from(JSON.stringify({method: "job", params: job}) + "\n"));
    });
    assert.ok(received);
    assert.equal(received.algo, expectedAlgo);
  }
});

test("MoneroOcean login jobs canonicalize cuckaroo before mining starts", async () => {
  /** @type {PoolJob | undefined} */
  let received;
  await withMockPool({}, async ({socket}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} next */ (next) => {
      received = next;
      return keepMiningJob(next);
    });
    socket.emit("data", Buffer.from(JSON.stringify({
      id: 1,
      error: null,
      result: {
        id: "worker",
        job: {
          algo: "cuckaroo", blob: "00", job_id: "initial-cuckaroo", target: "ffffffff",
        },
      },
    }) + "\n"));
  });
  assert.ok(received);
  assert.equal(received.algo, "c29");
});

test("an active donation pool returns to the primary after unsupported work", async () => {
  const primaryJob = {algo: "rx/0", dev: "cpu", job_id: "primary"};
  /** @type {PoolJob | null} */
  let restoredJob = null;
  await withMockPool({
    pool: {donation_until: Infinity, algo_params: {"rx/0": {dev: "cpu", perf: 1}}},
    pool_time: {close_wait: 0.001},
    opt: {
      pool_ids: {active: 0, primary: 1, donate: 0},
      job: {algo: null},
      algo_params: {"rx/0": {dev: "cpu", perf: 1}},
    },
  }, async ({socket}) => {
    testGlobal.opt.pools.push(mockPoolConfig({last_job: primaryJob}));
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      restoredJob = job;
      return keepMiningJob(job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(
      '{"id":1,"error":null,"result":{"id":"w","job":' +
      '{"algo":"etchash","blob":"00","job_id":"1","target":"01"}}}\n'
    ));
    assert.equal(testGlobal.opt.pool_ids.active, 1);
    assert.equal(restoredJob, primaryJob);
  });
});

test("pool chunk stops after unsupported donation work closes its socket", async () => {
  await withMockPool({
    pool: {
      logged_in: true, donation_until: Date.now() + 10000,
      algo_params: {"rx/0": {dev: "cpu", perf: 1}},
    },
    opt: {
      pool_ids: {active: 1, primary: 1, donate: 0}, job: {algo: null},
      algo_params: {"rx/0": {dev: "cpu", perf: 1}},
    },
  }, async ({socket, poolConfig}) => {
    testGlobal.opt.pools.push(mockPoolConfig());
    /** @type {Array<string | number | undefined>} */
    const jobs = [];
    pool.connect_pool_throttle(0, (job) => {
      jobs.push(job.job_id);
      return keepMiningJob(job);
    });
    socket.emit("data", Buffer.from(
      JSON.stringify({method: "job", params: {
        algo: "etchash", blob: "00", job_id: "unsupported", target: "ffffffff",
      }}) + "\n" +
      JSON.stringify({id: 1, error: null, result: {id: "worker", job: {
        algo: "rx/0", blob: "00", job_id: "after-close", target: "ffffffff",
      }}}) + "\n"
    ));
    assert.equal(socket.destroyed, true);
    assert.equal(poolConfig.socket, null);
    assert.equal(poolConfig.logged_in, false);
    assert.equal(testGlobal.opt.pool_ids.active, 1);
    assert.deepEqual(jobs, []);
  });
});

test("pool chunk stops after its socket is replaced", async () => {
  await withMockPool({pool: {logged_in: true}}, async ({socket, poolConfig}) => {
    const replacement = /** @type {PoolMockSocket} */ (new events.EventEmitter());
    replacement.destroy = function() {this.destroyed = true;};
    /** @type {Array<string | number | undefined>} */
    const jobs = [];
    pool.connect_pool_throttle(0, (job) => {
      jobs.push(job.job_id);
      if (job.job_id === "first") {poolConfig.socket = replacement;}
      return keepMiningJob(job);
    });
    const line = (/** @type {string} */ job_id) => JSON.stringify({method: "job", params: {
      algo: "rx/0", blob: "00", job_id, target: "ffffffff",
    }}) + "\n";
    const tail = line("old-remainder");
    socket.emit("data", Buffer.from(line("first") + line("old-complete") + tail.slice(0, -5)));
    socket.emit("data", Buffer.from(tail.slice(-5)));
    socket.emit("end");
    assert.deepEqual(jobs, ["first"]);
    assert.equal(poolConfig.socket, replacement);
    assert.equal(replacement.destroyed, undefined);
    assert.equal(poolConfig.last_job?.job_id, "first");
  });
});

test("pool chunk preserves valid lines and a fragmented remainder", async () => {
  await withMockPool({pool: {logged_in: true}}, async ({socket, poolConfig}) => {
    /** @type {Array<string | number | undefined>} */
    const jobs = [];
    pool.connect_pool_throttle(0, (job) => {
      jobs.push(job.job_id);
      return keepMiningJob(job);
    });
    const line = (/** @type {string} */ job_id) => JSON.stringify({method: "job", params: {
      algo: "rx/0", blob: "00", job_id, target: "ffffffff",
    }}) + "\n";
    const tail = line("third");
    socket.emit("data", Buffer.from(line("first") + line("second") + tail.slice(0, -5)));
    assert.deepEqual(jobs, ["first", "second"]);
    socket.emit("data", Buffer.from(tail.slice(-5)));
    assert.deepEqual(jobs, ["first", "second", "third"]);
    assert.equal(poolConfig.socket, socket);
    assert.equal(socket.destroyed, undefined);
    assert.equal(poolConfig.last_job?.job_id, "third");
  });
});

test("an expired donation window does not open a pool connection", async () => {
  await withMockPool({
    pool: {donation_until: 0, algo_params: {"rx/0": {dev: "cpu", perf: 1}}},
    opt: {
      pool_ids: {active: 1, primary: 1, donate: 0},
      job: {algo: null},
      algo_params: {"rx/0": {dev: "cpu", perf: 1}},
    },
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    assert.equal(poolConfig.socket, null);
    assert.equal(socket.listenerCount("connect"), 0);
  });
});

test("late donation work after window expiry returns to primary without an error", async () => {
  const primaryJob = {algo: "rx/0", dev: "cpu", job_id: "primary"};
  const previousLogError = helper.log_err;
  /** @type {string[]} */
  const errors = [];
  helper.log_err = (message) => errors.push(message);
  try {
    await withMockPool({
      pool: {donation_until: Infinity, algo_params: {"rx/0": {dev: "cpu", perf: 1}}},
      pool_time: {close_wait: 0.001},
      opt: {pool_ids: {active: 0, primary: 1, donate: 0}, job: {algo: null}},
    }, async ({socket, poolConfig}) => {
      testGlobal.opt.pools.push(mockPoolConfig({last_job: primaryJob}));
      /** @type {PoolJob[]} */
      const jobs = [];
      pool.connect_pool_throttle(0, (job) => {
        jobs.push(job);
        return keepMiningJob(job);
      });
      socket.emit("connect");
      poolConfig.donation_until = 0;
      poolConfig.logged_in = true;
      socket.emit("data", Buffer.from(JSON.stringify({method: "job", params: {
        algo: "rx/0", blob: "00", job_id: "late", target: "ffffffff",
      }}) + "\n"));
      assert.equal(testGlobal.opt.pool_ids.active, 1);
      assert.deepEqual(jobs, [primaryJob]);
      assert.deepEqual(errors, []);
    });
  } finally {
    helper.log_err = previousLogError;
  }
});

test("a throttled donation connection rechecks the window before opening", async () => {
  const now = Date.now();
  await withMockPool({
    pool: {
      donation_until: now + 1000,
      last_connect_time: now,
      algo_params: {"rx/0": {dev: "cpu", perf: 1}},
    },
    pool_time: {connect_throttle: 0.03},
    opt: {
      pool_ids: {active: 1, primary: 1, donate: 0},
      job: {algo: null},
      algo_params: {"rx/0": {dev: "cpu", perf: 1}},
    },
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    poolConfig.donation_until = 0;
    await new Promise((resolve) => setTimeout(resolve, 40));
    assert.equal(poolConfig.socket, null);
    assert.equal(socket.listenerCount("connect"), 0);
  });
});

test("malformed pool input is not copied into logs", async () => {
  const previousLogError = helper.log_err;
  /** @type {string[]} */
  const logs = [];
  helper.log_err = (message) => logs.push(message);
  try {
    await withMockPool({}, async ({socket, switched}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("data", Buffer.from("\u001b[31msecret\n"));
      assert.equal(socket.destroyed, true);
      assert.equal(switched(), true);
    });
  } finally {
    helper.log_err = previousLogError;
  }
  assert.equal(logs.length, 1);
  assert.ok(logs[0]);
  assert.match(logs[0], /Can't parse 11-byte message from the pool/);
  assert.doesNotMatch(logs[0], /secret/);
  assert.equal(logs[0].includes(String.fromCharCode(27)), false);
});

test("socket errors log only a safe transport code", async () => {
  const previousLogError = helper.log_err;
  /** @type {string[]} */
  const logs = [];
  helper.log_err = (message) => logs.push(message);
  try {
    await withMockPool({}, async ({socket}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("error", Object.assign(new Error("remote private detail"), {code: "ECONNRESET"}));
    });
  } finally {
    helper.log_err = previousLogError;
  }
  assert.match(logs.join("\n"), /Socket error from the pool \(ECONNRESET\)/);
  assert.doesNotMatch(logs.join("\n"), /remote private detail/);
});

test("fixed KawPow pools use Raven stratum subscribe and authorize", async () => {
  /** @type {PoolJob | undefined} */
  let jobMessage;
  const shareTarget = "00000000ffff0000000000000000000000000000000000000000000000000000";
  await withMockPool({
    pool: {is_keepalive: true, login: "RVNwallet.rig01"},
    pool_time: {keepalive: 0.001},
    opt: {job: {algo: "kawpow"}},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return keepMiningJob(job);
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.equal(writes[0].method, "mining.subscribe");

    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":["0a1fa6c0","e0"]}\n' +
      '{"jsonrpc":"2.0","id":2,"error":null,"result":true}\n' +
      '{"method":"mining.set_target","params":["' + shareTarget + '"]}\n' +
      '{"method":"mining.notify","params":["203d","' + "00".repeat(32) + '","' + "11".repeat(32) + '","' +
      '",true,4390582,"1b01e5f2"],"id":null,"jsonrpc":"2.0"}\n'
    ));

    assert.ok(writes[1]);
    assert.ok(jobMessage);
    assert.equal(writes[1].method, "mining.authorize");
    assert.deepEqual(writes[1].params, ["RVNwallet.rig01", "x"]);
    assert.equal(poolConfig.extra_nonce, "0a1fa6c0");
    assert.equal(poolConfig["raven_target"], shareTarget);
    assert.equal(jobMessage.job_id, "203d");
    assert.equal(jobMessage.target, shareTarget);
    assert.equal(jobMessage.blob, "00".repeat(32) + "00000000c0a61f0a");
    assert.equal(jobMessage.nonce, "0a1fa6c000000000");
    assert.equal(jobMessage.nicehash_mask, "ffffffff00000000");
    assert.equal(writes.length, 2);
  });
});

test("pre-authorization EVR, Verthash and Pearl jobs retain only the latest announcement", async () => {
  const ravenTarget = "00000000ffff" + "00".repeat(26);
  for (const scenario of [
    {
      protocol: "raven", algo: "evrprogpow",
      control: {method: "mining.set_target", params: [ravenTarget]},
      controlField: "raven_target",
      job: (/** @type {string} */ jobId) => ({
        method: "mining.notify",
        params: [jobId, "00".repeat(32), "11".repeat(32), "", true, 1],
        algo: "evrprogpow",
      }),
    },
    {
      protocol: "pearlhash", algo: "pearlhash",
      control: {method: "mining.set_difficulty", params: [20000]},
      controlField: "pearlhash_difficulty",
      job: (/** @type {string} */ jobId) => ({
        id: null, method: "mining.notify",
        params: {header: "00".repeat(76), job_id: jobId, height: 1},
      }),
    },
    {
      protocol: "verthash", algo: "verthash",
      control: {method: "mining.set_difficulty", params: [1]},
      controlField: "verthash_difficulty",
      job: (/** @type {string} */ jobId) => ({
        method: "mining.notify",
        params: [jobId, "22".repeat(32), "", "", [], "01000000", "ffff001d", "12345678", true],
      }),
    },
  ]) {
    /** @type {PoolJob[]} */
    const jobs = [];
    await withMockPool({
      pool: {protocol: scenario.protocol},
      opt: {job: {algo: scenario.algo}},
    }, async ({socket, writes, poolConfig}) => {
      pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
        jobs.push(job);
        return keepMiningJob(job);
      });
      socket.emit("connect");
      assert.ok(writes[0]);
      assert.equal(writes[0].method, "mining.subscribe");
      socket.emit("data", Buffer.from(JSON.stringify({
        jsonrpc: "2.0", id: 1, error: null,
        result: [["mining.notify", "1"], "080c", 6],
      }) + "\n"));
      assert.ok(writes[1]);
      assert.equal(writes[1].method, "mining.authorize");

      socket.emit("data", Buffer.from(JSON.stringify(scenario.control) + "\n"));
      socket.emit("data", Buffer.from(JSON.stringify(scenario.job("old-job")) + "\n"));
      socket.emit("data", Buffer.from(JSON.stringify(scenario.job("latest-job")) + "\n"));
      assert.equal(poolConfig[scenario.controlField], scenario.control.params[0]);
      assert.equal(poolConfig.logged_in, false);
      assert.equal(poolConfig.bad_shares, 0);
      assert.equal(jobs.length, 0);
      assert.ok(poolConfig.pending_job);

      socket.emit("data", Buffer.from(JSON.stringify({
        jsonrpc: "2.0", id: 2, error: null, result: true,
      }) + "\n"));
      assert.equal(poolConfig.logged_in, true);
      assert.equal(poolConfig.bad_shares, 0);
      assert.equal(poolConfig.pending_job, undefined);
      assert.equal(jobs.length, 1);
      assert.ok(jobs[0]);
      assert.equal(jobs[0].job_id, "latest-job");
    });
  }
});

test("Pearl discards an early job on authorization failure or disconnect", async () => {
  for (const disconnected of [false, true]) {
    await withMockPool({pool: {protocol: "pearlhash"}, opt: {job: {algo: "pearlhash"}}},
      async ({socket, poolConfig}) => {
        let jobs = 0;
        pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
          ++jobs;
          return keepMiningJob(job);
        });
        socket.emit("connect");
        const notify = {id: null, method: "mining.notify", params: {
          header: "00".repeat(76), job_id: "early", difficulty: 20000,
        }};
        socket.emit("data", Buffer.from(JSON.stringify(notify) + "\n"));
        assert.ok(poolConfig.pending_job);
        if (disconnected) {
          socket.emit("error", new Error("reconnect"));
        } else {
          socket.emit("data", Buffer.from(JSON.stringify({id: 2, error: null, result: false}) + "\n"));
          socket.emit("data", Buffer.from(JSON.stringify(notify) + "\n"));
        }
        assert.equal(poolConfig.pending_job, undefined);
        assert.equal(poolConfig.logged_in, false);
        assert.equal(poolConfig.bad_shares, 0);
        assert.equal(jobs, 0);
      });
  }
});

test("pre-authorization job is discarded on authorization failure", async () => {
  await withMockPool({
    pool: {protocol: "verthash"},
    opt: {job: {algo: "verthash"}},
  }, async ({socket, poolConfig}) => {
    let jobs = 0;
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      ++jobs;
      return keepMiningJob(job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0", id: 1, error: null,
      result: [["mining.notify", "1"], "080c", 6],
    }) + "\n"));
    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.notify",
      params: ["queued-job", "22".repeat(32), "", "", [], "01000000", "ffff001d", "12345678", true],
    }) + "\n"));
    assert.equal(jobs, 0);
    assert.ok(poolConfig.pending_job);

    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0", id: 2, error: {message: "authorization rejected"}, result: false,
    }) + "\n"));
    assert.equal(poolConfig.pending_job, undefined);
    assert.equal(poolConfig.logged_in, false);
    assert.equal(jobs, 0);
  });
});

test("pool reconnect discards a pre-authorization job", async () => {
  await withMockPool({
    pool: {protocol: "raven"},
    opt: {job: {algo: "evrprogpow"}},
  }, async ({socket, poolConfig}) => {
    let jobs = 0;
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      ++jobs;
      return keepMiningJob(job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0", id: 1, error: null,
      result: [["mining.notify", "1"], "080c", 6],
    }) + "\n"));
    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.notify",
      params: ["queued-job", "00".repeat(32), "11".repeat(32), "", true, 1],
      algo: "evrprogpow",
    }) + "\n"));
    assert.ok(poolConfig.pending_job);
    socket.emit("error", new Error("reconnect"));
    assert.equal(poolConfig.pending_job, undefined);
    assert.equal(poolConfig.logged_in, false);
    assert.equal(jobs, 0);
  });
});

test("ordered subscribe authorization still delivers the following job once", async () => {
  await withMockPool({
    pool: {protocol: "verthash"},
    opt: {job: {algo: "verthash"}},
  }, async ({socket, poolConfig}) => {
    let jobs = 0;
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      ++jobs;
      return keepMiningJob(job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0", id: 1, error: null,
      result: [["mining.notify", "1"], "080c", 6],
    }) + "\n"));
    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0", id: 2, error: null, result: true,
    }) + "\n"));
    assert.equal(poolConfig.logged_in, true);
    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.set_difficulty", params: [1],
    }) + "\n"));
    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.notify",
      params: ["ordered-job", "22".repeat(32), "", "", [], "01000000", "ffff001d", "12345678", true],
    }) + "\n"));
    assert.equal(poolConfig.pending_job, undefined);
    assert.equal(jobs, 1);
  });
});

test("fixed Etchash pools use Eth stratum notify jobs", async () => {
  /** @type {PoolJob | undefined} */
  let jobMessage;
  const headerHash = "22".repeat(32);
  const seedHash = "11".repeat(32);
  await withMockPool({
    pool: {is_keepalive: true, login: "0xwallet.worker"},
    opt: {job: {algo: "etchash"}},
    pool_time: {keepalive: 0.001, first_job_wait: 0.001},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return keepMiningJob(job);
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.equal(writes[0].method, "mining.subscribe");

    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":[[["mining.notify","1"],"080c"],"080c",6]}\n' +
      '{"jsonrpc":"2.0","method":"mining.set_difficulty","params":[1]}\n' +
      '{"jsonrpc":"2.0","id":2,"error":null,"result":true}\n' +
      '{"method":"mining.notify","params":["203d","' + seedHash + '","' + headerHash + '",true],"id":null,"jsonrpc":"2.0"}\n'
    ));

    assert.ok(writes[1]);
    assert.ok(jobMessage);
    assert.equal(writes[1].method, "mining.authorize");
    assert.deepEqual(writes[1].params, ["0xwallet.worker", "x"]);
    assert.equal(poolConfig.extra_nonce, "080c");
    assert.equal(poolConfig["eth_difficulty"], 1);
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

test("late Eth inference retains a preceding mining.set_difficulty", async () => {
  /** @type {PoolJob | undefined} */
  let jobMessage;
  const headerHash = "22".repeat(32);
  const seedHash = "11".repeat(32);
  await withMockPool({
    pool: {use_subscribe: false},
    opt: {job: {}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(
      JSON.stringify({jsonrpc: "2.0", id: 1, error: null, result: {id: "worker"}}) + "\n" +
      '{"jsonrpc":"2.0","method":"mining.set_difficulty","params":[100]}\n' +
      JSON.stringify({
        jsonrpc: "2.0", id: null, method: "mining.notify",
        params: ["203d", seedHash, headerHash, true], algo: "etchash",
      }) + "\n"
    ));

    assert.ok(jobMessage);
    assert.equal(poolConfig.inferred_protocol, "eth");
    assert.equal(poolConfig["eth_difficulty"], 100);
    assert.equal(jobMessage.target, helper.ethDiff2Target(100));
    assert.equal(jobMessage.algo, "etchash");
  });
});

test("login result infers XELIS from every supported algorithm marker", async () => {
  for (const algo of ["xel/2", "xel/3", "xel/v3"]) {
    await withMockPool({
      pool: {use_subscribe: false},
      opt: {job: {}},
      pool_time: {first_job_wait: 0.05},
    }, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("data", Buffer.from(JSON.stringify({
        jsonrpc: "2.0", id: 1, error: null,
        result: {id: "worker", algo},
      }) + "\n"));
      assert.equal(poolConfig.logged_in, true);
      assert.equal(poolConfig.inferred_protocol, "xelis");
    });
  }
});

test("late Eth inference retains a preceding mining.set_target", async () => {
  /** @type {PoolJob | undefined} */
  let jobMessage;
  const headerHash = "22".repeat(32);
  const seedHash = "11".repeat(32);
  const suppliedTarget = "0x1234";
  const expectedTarget = "0".repeat(60) + "1234";
  await withMockPool({
    pool: {use_subscribe: false},
    opt: {job: {}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(
      JSON.stringify({jsonrpc: "2.0", id: 1, error: null, result: {id: "worker"}}) + "\n" +
      JSON.stringify({method: "mining.set_target", params: [suppliedTarget]}) + "\n" +
      JSON.stringify({
        jsonrpc: "2.0", id: null, method: "mining.notify",
        params: ["203d", seedHash, headerHash, true], algo: "etchash",
      }) + "\n"
    ));

    assert.ok(jobMessage);
    assert.equal(poolConfig.inferred_protocol, "eth");
    assert.equal(poolConfig.stratum_target, "1234");
    assert.equal(poolConfig["eth_target"], undefined);
    assert.equal(jobMessage.target, expectedTarget);
    assert.equal(jobMessage.algo, "etchash");
  });
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

test("MO login-inferred Etchash ignores stale keepalive response", async () => {
  /** @type {PoolJob | undefined} */
  let jobMessage;
  const headerHash = "22".repeat(32);
  const seedHash = "11".repeat(32);
  await withMockPool({
    pool: {is_keepalive: true, pass: "x~etchash"},
    pool_time: {keepalive: 60, first_job_wait: 0.001},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return keepMiningJob(job);
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.equal(writes[0].method, "login");
    assert.notEqual(poolConfig.keepalive, null);

    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      error: null,
      result: {id: "worker", algo: "etchash", extra_nonce: "080c"},
    }) + "\n"));

    assert.equal(poolConfig.logged_in, true);
    assert.equal(poolConfig.inferred_protocol, "eth");
    assert.equal(poolConfig.keepalive, null);

    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0",
      id: 2,
      error: {message: "Authorization rejected"},
      result: false,
    }) + "\n"));

    assert.equal(poolConfig.logged_in, true);

    socket.emit("data", Buffer.from(
      '{"method":"mining.notify","params":["203d","' + seedHash + '","' + headerHash + '",true],"algo":"etchash","id":null,"jsonrpc":"2.0"}\n'
    ));

    assert.ok(jobMessage);
    assert.equal(jobMessage.job_id, "203d");
    assert.equal(jobMessage.seed_hash, seedHash);
    assert.equal(jobMessage.header_hash, headerHash);
  });
});

test("login-dialect keepalived carries the upstream worker id and reschedules", async () => {
  await withMockPool({
    pool: {is_keepalive: true, use_subscribe: false},
    pool_time: {keepalive: 0.02, first_job_wait: 0.001},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    poolConfig.last_job = {};
    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0", id: 1, error: null,
      result: {id: "upstream-session", extensions: ["keepalive"]},
    }) + "\n"));
    assert.equal(poolConfig.worker_id, "upstream-session");
    await new Promise((resolve) => setTimeout(resolve, 25));

    const firstKeepalive = writes.find((message) => message.method === "keepalived");
    assert.deepEqual(firstKeepalive, {
      jsonrpc: "2.0", id: 2, method: "keepalived", params: {id: "upstream-session"},
    });
    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":2,"error":null,"result":true}\n'
    ));
    assert.equal(poolConfig.good_shares, 0);
    assert.equal(poolConfig.bad_shares, 0);

    await new Promise((resolve) => setTimeout(resolve, 25));
    assert.ok(writes.filter((message) => message.method === "keepalived").length >= 2);
  });
});

test("login-dialect keepalived omits an unavailable worker id", async () => {
  await withMockPool({
    pool: {is_keepalive: true, use_subscribe: false},
    pool_time: {keepalive: 0.01, first_job_wait: 0.001},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    poolConfig.last_job = {};
    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":{}}\n'
    ));
    await new Promise((resolve) => setTimeout(resolve, 15));
    assert.deepEqual(writes.find((message) => message.method === "keepalived"), {
      jsonrpc: "2.0", id: 2, method: "keepalived", params: {},
    });
  });
});

test("subscribe-only pools do not schedule login-dialect keepalived", async () => {
  await withMockPool({
    pool: {is_keepalive: true, protocol: "eth"},
    pool_time: {keepalive: 0.01},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    poolConfig.last_job = {};
    assert.ok(writes[0]);
    assert.equal(writes[0].method, "mining.subscribe");
    await new Promise((resolve) => setTimeout(resolve, 15));
    assert.equal(writes.some((message) => message.method === "keepalived"), false);
    assert.equal(poolConfig.keepalive, null);
  });
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

test("fixed Autolykos2 pools use Ergo stratum notify jobs", async () => {
  /** @type {PoolJob | undefined} */
  let jobMessage;
  const headerHash = "54".repeat(32);
  const bound = "7067388259113537318333190002971674063283542741642755394446115914399301849";
  await withMockPool({
    pool: {is_keepalive: true, login: "9ergwallet.worker"},
    opt: {job: {algo: "autolykos2"}},
    pool_time: {keepalive: 0.001, first_job_wait: 0.001},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return keepMiningJob(job);
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.equal(writes[0].method, "mining.subscribe");

    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":[[["mining.notify","1"],"080c"],"080c",6]}\n' +
      '{"jsonrpc":"2.0","id":2,"error":null,"result":true}\n' +
      '{"method":"mining.notify","params":["203d",614400,"' + headerHash + '","","",2,"' + bound + '","",true],"algo":"autolykos2","id":null,"jsonrpc":"2.0"}\n'
    ));

    assert.ok(writes[1]);
    assert.ok(jobMessage);
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
  /** @type {PoolMockSocket} */
  const staleSocket = /** @type {PoolMockSocket} */ (new events.EventEmitter());
  /** @type {PoolMockSocket} */
  const replacementSocket = /** @type {PoolMockSocket} */ (new events.EventEmitter());
  replacementSocket.destroy = function() { this.destroyed = true; };

  await withMockPool({
    socket: staleSocket,
    pool_time: {first_job_wait: 0.001},
  }, async ({poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    poolConfig.socket = replacementSocket;
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(replacementSocket.destroyed, undefined);
  });
});

test("queued pre-authorization job still requires timely authorization", async () => {
  await withMockPool({
    pool: {protocol: "verthash"},
    opt: {job: {algo: "verthash"}},
    pool_time: {first_job_wait: 0.005},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0", id: 1, error: null,
      result: [["mining.notify", "1"], "01020304", 4],
    }) + "\n"));
    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.notify",
      params: ["queued-job", "22".repeat(32), "", "", [], "01000000", "ffff001d", "12345678", true],
    }) + "\n"));
    assert.equal(poolConfig.logged_in, false);
    assert.ok(poolConfig.pending_job);
    await new Promise((resolve) => setTimeout(resolve, 15));
    assert.equal(socket.destroyed, true);
    assert.equal(poolConfig.pending_job, undefined);
  });
});

test("pool sockets retain both IP families with a one-second connection attempt budget", async () => {
  const net = require("node:net");
  const originalNetConnect = net.connect;
  const originalTlsConnect = tls.connect;
  const previousOpt = testGlobal.opt;
  /** @type {import("node:tls").ConnectionOptions[]} */
  const optionsSeen = [];
  /** @param {import("node:tls").ConnectionOptions} options */
  function connect(options) {
    optionsSeen.push(options);
    /** @type {PoolMockSocket} */
    const socket = /** @type {PoolMockSocket} */ (new events.EventEmitter());
    socket.write = () => undefined;
    socket.destroy = () => undefined;
    return socket;
  }
  net.connect = /** @type {typeof net.connect} */ (/** @type {unknown} */ (connect));
  tls.connect = /** @type {typeof tls.connect} */ (/** @type {unknown} */ (connect));
  try {
    for (const is_tls of [false, true]) {
      testGlobal.opt = mockPoolOptions({
        pool: {url: "pool.example", port: is_tls ? 20001 : 10001, is_tls},
        pool_time: {first_job_wait: 0.001, connect_throttle: 0},
      });
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      testGlobal.opt.pools[0].socket = null;
    }
    assert.deepEqual(optionsSeen, [
      {host: "pool.example", port: 10001, autoSelectFamilyAttemptTimeout: 1000},
      {host: "pool.example", port: 20001, autoSelectFamilyAttemptTimeout: 1000,
        rejectUnauthorized: false},
    ]);
  } finally {
    await new Promise((resolve) => setTimeout(resolve, 10));
    net.connect = originalNetConnect;
    tls.connect = originalTlsConnect;
    testGlobal.opt = previousOpt;
  }
});

test("TLS pools accept self-signed certificates unless verification is explicitly enabled", async () => {
  const originalConnect = tls.connect;
  const previousOpt = testGlobal.opt;
  /** @type {Array<{rejectUnauthorized?: boolean | undefined}>} */
  const optionsSeen = [];
  assert.equal(opts.pool_create("pool.example", 443, true, "user", "x").tls_verify, false);
  tls.connect = /** @type {typeof tls.connect} */ (/** @type {unknown} */ (
    /** @param {import("node:tls").ConnectionOptions} options */
    function(options) {
      const tlsOptions = typeof options === "object" && options !== null ? options : {};
      optionsSeen.push({rejectUnauthorized: tlsOptions.rejectUnauthorized});
      /** @type {PoolMockSocket} */
      const socket = /** @type {PoolMockSocket} */ (new events.EventEmitter());
      socket.write = () => undefined;
      socket.destroy = () => undefined;
      return socket;
    }
  ));
  testGlobal.opt = mockPoolOptions({
    pool: {
      url: "pool.example",
      port: 443,
      is_tls: true,
      is_keepalive: false,
    },
    pool_time: {first_job_wait: 0.001, connect_throttle: 0},
  });

  try {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    testGlobal.opt.pools[0].socket = null;
    testGlobal.opt.pools[0].tls_verify = true;
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    testGlobal.opt.pools[0].socket = null;
    testGlobal.opt.pools[0].tls_verify = false;
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    testGlobal.opt.pools[0].socket = null;
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.ok(optionsSeen[0]);
    assert.ok(optionsSeen[1]);
    assert.ok(optionsSeen[2]);
    assert.equal(optionsSeen[0].rejectUnauthorized, false);
    assert.equal(optionsSeen[1].rejectUnauthorized, true);
    assert.equal(optionsSeen[2].rejectUnauthorized, false);
  } finally {
    tls.connect = originalConnect;
    testGlobal.opt = previousOpt;
  }
});

test("pool extranonce updates accept the supported size endpoints", async () => {
  for (const size of [0, 8]) {
    await withMockPool({}, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "mining.set_extranonce", params: ["aabb", size],
      }) + "\n"));
      assert.equal(poolConfig.extra_nonce2_size, size);
      assert.equal(socket.destroyed, undefined);
    });
  }
});

test("pool extranonce updates reject sizes outside the supported byte range", async () => {
  for (const size of [-1, 9, "many", {}]) {
    await withMockPool({}, async ({socket}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "mining.set_extranonce", params: ["aabb", size],
      }) + "\n"));
      assert.equal(socket.destroyed, true, String(size));
    });
  }
});

test("malformed pool job data closes the pool instead of throwing", async () => {
  await withMockPool({
    pool: {logged_in: true},
    pool_time: {first_job_wait: 0.001},
  }, async ({socket, switched}) => {
    testGlobal.opt.pools.push(mockPoolConfig());
    testGlobal.opt.pool_ids.active = 1;
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    assert.doesNotThrow(() => {
      socket.emit("data", Buffer.from('{"method":"job","params":{"target":"zz"}}\n'));
    });
    assert.equal(socket.destroyed, true);
    assert.equal(testGlobal.opt.pools[0].socket, null);
    assert.equal(testGlobal.opt.pool_ids.active, 1);
    assert.equal(switched(), true);
    await new Promise((resolve) => setTimeout(resolve, 10));
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

test("array job params are not accepted as object jobs", async () => {
  await withMockPool({pool: {logged_in: true}}, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("data", Buffer.from('{"method":"job","params":[]}\n'));
    assert.equal(poolConfig.last_job, null);
  });
});

test("generic PearlHash jobs default to certificate V3 and reject other versions", async () => {
  for (const version of [undefined, 0, 1, 2, 3, 4]) {
    await withMockPool({pool: {logged_in: true}}, async ({socket, poolConfig}) => {
      let jobs = 0;
      pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
        ++jobs;
        return keepMiningJob(job);
      });
      socket.emit("data", Buffer.from(JSON.stringify({
        method: "job",
        params: {
          algo: "pearlhash", blob: "00", job_id: "pearl-job",
          ...(version === undefined ? {} : {pearlhash_cert_version: version}),
        },
      }) + "\n"));
      const accepted = version === undefined || version === 3;
      assert.equal(jobs, accepted ? 1 : 0);
      assert.equal(poolConfig.last_job === null, !accepted);
      assert.equal(socket.destroyed, accepted ? undefined : true);
    });
  }
});

test("errored login response with job does not start mining", async () => {
  const previousLogError = helper.log_err;
  /** @type {string[]} */
  const logs = [];
  helper.log_err = (message) => logs.push(message);
  try {
    await withMockPool({pool: {login: "wallet.worker", pass: "top-secret"}},
      async ({socket, poolConfig}) => {
        pool.connect_pool_throttle(0, unexpectedPoolJob);
        socket.emit("data", Buffer.from(JSON.stringify({
          id: 1,
          jsonrpc: "2.0",
          error: {message: "wallet.worker rejected top-secret\u001b[31m"},
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

        assert.equal(poolConfig.last_job, null);
      }
    );
  } finally {
    helper.log_err = previousLogError;
  }
  assert.equal(logs.length, 1);
  assert.ok(logs[0]);
  assert.doesNotMatch(logs[0], /wallet|top-secret/);
  assert.equal(logs[0].includes(String.fromCharCode(27)), false);
  assert.match(logs[0], /Login to the pool failed/);
});

test("malformed or wrong-id login jobs cannot authorize or start mining", async () => {
  const validJob = {
    algo: "etchash",
    blob: "00".repeat(32),
    job_id: "login-job",
    target: "00000000ffff0000000000000000000000000000000000000000000000000000",
  };
  for (const {id, result} of [
    {id: 2, result: {id: "worker", extra_nonce: "beef", extensions: ["nicehash"], job: {...validJob}}},
    {id: 1, result: {id: "worker", extra_nonce: "beef", extensions: ["nicehash"], job: {job_id: "malformed"}}},
  ]) {
    await withMockPool({}, async ({socket, poolConfig}) => {
      let jobs = 0;
      pool.connect_pool_throttle(0, /** @param {PoolJob} _job */ (_job) => {
        ++jobs;
        return keepMiningJob(validJob);
      });
      socket.emit("data", Buffer.from(JSON.stringify({
        id, jsonrpc: "2.0", error: null, result,
      }) + "\n"));
      assert.equal(jobs, 0);
      assert.equal(poolConfig.logged_in, false);
      assert.equal(poolConfig.worker_id, undefined);
      assert.equal(poolConfig.extra_nonce, undefined);
    });
  }
});

test("pool reconnect clears transient protocol state and submit metadata", async () => {
  const transientFields = [
    "beam_difficulty", "beam_nonceprefix", "cortex_nonce", "eth_difficulty", "eth_target", "extra_nonce",
    "extra_nonce2_size", "inferred_protocol", "ironfish_target", "ironfish_xn",
    "kaspa_difficulty", "kaspa_target", "nexa_difficulty", "nexa_target",
    "negotiated_keepalive", "negotiated_nicehash", "pearlhash_difficulty",
    "pending_cortex_submit_ids", "raven_target", "stratum_target",
    "verthash_difficulty", "worker_id",
    "xelis_difficulty", "xelis_extra_nonce", "xelis_public_key", "zelhash_target",
  ];
  await withMockPool({pool: {protocol: "eth"}}, async ({socket, poolConfig, switched}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    for (const field of transientFields) {poolConfig[field] = "transient";}
    poolConfig.last_job = {job_id: "old-job", submit_mode: "eth"};
    poolConfig.last_connect_time = 12345;
    poolConfig.good_shares = 7;
    poolConfig.bad_shares = 8;
    poolConfig.donation_until = 67890;
    const preserved = {
      last_connect_time: poolConfig.last_connect_time,
      good_shares: poolConfig.good_shares,
      bad_shares: poolConfig.bad_shares,
      donation_until: poolConfig.donation_until,
      protocol: poolConfig.protocol,
    };

    socket.emit("error", new Error("reconnect"));

    for (const field of transientFields) {assert.equal(poolConfig[field], undefined, field);}
    assert.equal(poolConfig.last_job, null);
    assert.deepEqual({
      last_connect_time: poolConfig.last_connect_time,
      good_shares: poolConfig.good_shares,
      bad_shares: poolConfig.bad_shares,
      donation_until: poolConfig.donation_until,
      protocol: poolConfig.protocol,
    }, preserved);
    assert.equal(switched(), true);
  });
});

test("job notification before login success does not start mining", async () => {
  await withMockPool({}, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
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
        error: {message: "No double login is allowed"},
        result: false,
      }) + "\n"
    ));

    assert.equal(poolConfig.last_job, null);
    assert.equal(poolConfig.logged_in, false);
  });
});

test("login job inherits height from login result metadata", async () => {
  /** @type {PoolJob[]} */
  const jobs = [];
  await withMockPool({}, async ({socket, poolConfig}) => {
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
        height: 1799914,
        job: {
          algo: "etchash",
          blob: "00".repeat(32),
          job_id: "login-job",
          seed_hash: "11".repeat(32),
          target: "00000000ffff0000000000000000000000000000000000000000000000000000",
        },
      },
    }) + "\n"));

    assert.equal(jobs.length, 1);
    const firstJob = jobs[0];
    assert.ok(firstJob);
    assert.equal(firstJob.height, 1799914);
    assert.equal(poolConfig.inferred_protocol, "eth");

    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.notify",
      params: ["notify-job", "11".repeat(32), "22".repeat(32), true],
    }) + "\n"));

    assert.equal(jobs.length, 2);
    const secondJob = jobs[1];
    assert.ok(secondJob);
    assert.equal(secondJob.job_id, "notify-job");
    assert.equal(secondJob.algo, "etchash");
  });
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

test("native PearlHash login jobs use the Pearl notification parser", async () => {
  /** @type {PoolJob[]} */
  const jobs = [];
  const header = "12".repeat(76);
  const baseTarget = "0f".repeat(32);
  await withMockPool({pool: {use_subscribe: false, pearlhash_target_format: "jackpot"}}, async ({socket}) => {
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

test("oversized pool line buffer closes the pool", async () => {
  await withMockPool({
    pool_time: {first_job_wait: 0.001},
  }, async ({socket, switched}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("data", Buffer.alloc(1024 * 1024 + 1, "a"));
    assert.equal(socket.destroyed, true);
    assert.equal(testGlobal.opt.pools[0].socket, null);
    assert.equal(switched(), true);
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
});

test("oversized generic pool-job fields close the pool", async () => {
  await withMockPool({
    pool: {logged_in: true},
    pool_time: {first_job_wait: 0.001},
  }, async ({socket, switched}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("data", Buffer.from(JSON.stringify({
      method: "job",
      params: {
        algo: "cn/0", blob: "00", job_id: "job", target: "01",
        backend_request: "x".repeat(128 * 1024 + 1),
      },
    }) + "\n"));
    assert.equal(socket.destroyed, true);
    assert.equal(switched(), true);
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
});

test("pool line limit counts UTF-8 bytes instead of JavaScript characters", async () => {
  await withMockPool({
    pool_time: {first_job_wait: 0.001},
  }, async ({socket, switched}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("data", Buffer.from("é".repeat(1024 * 512 + 1)));
    assert.equal(socket.destroyed, true);
    assert.equal(switched(), true);
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
});

test("KawPow login response id is reused for later notify jobs", async () => {
  /** @type {PoolJob | undefined} */
  let jobMessage;
  await withMockPool({
    pool: {pass: "~kawpow"},
    pool_time: {first_job_wait: 0.001},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return keepMiningJob(job);
    });
    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":{"id":"5122080","algo":"kawpow","extra_nonce":"ff81"}}\n' +
      '{"method":"mining.notify","params":["203d","' + "00".repeat(32) + '","' + "11".repeat(32) + '","' +
      "0000005eb993eef1b05c00000000000000000000000000000000000000000000" +
      '",true,4390582,"1b01e5f2"],"algo":"kawpow","id":null,"jsonrpc":"2.0"}\n'
    ));
    assert.ok(jobMessage);
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
  await withMockPool({}, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    poolConfig["pending_submit_count"] = 1;
    socket.emit("data", Buffer.from('{"jsonrpc":"2.0","id":3,"error":null,"result":false}\n'));
    assert.equal(poolConfig.good_shares, 0);
    assert.equal(poolConfig.bad_shares, 1);
  });
});

test("pool share response true is counted as accepted", async () => {
  await withMockPool({}, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    poolConfig["pending_submit_count"] = 1;
    socket.emit("data", Buffer.from('{"jsonrpc":"2.0","id":3,"error":null,"result":true}\n'));
    assert.equal(poolConfig.good_shares, 1);
    assert.equal(poolConfig.bad_shares, 0);
  });
});

test("normal submit response id 3 counts once when stringified", async () => {
  await withMockPool({}, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("data", Buffer.from('{"id":3,"result":true}\n'));
    assert.equal(poolConfig.good_shares, 0);
    poolConfig["pending_submit_count"] = 0;
    pool.pool_write(0, {jsonrpc: "2.0", id: 3, method: "mining.submit", params: []});
    assert.equal(poolConfig["pending_submit_count"], 1);
    for (const id of [4, "unknown"]) {
      socket.emit("data", Buffer.from(JSON.stringify({id, result: true}) + "\n"));
    }
    assert.equal(poolConfig.good_shares, 0);
    assert.equal(poolConfig.bad_shares, 0);
    socket.emit("data", Buffer.from('{"id":"3","result":true}\n'));
    assert.equal(poolConfig.good_shares, 1);
    assert.equal(poolConfig.bad_shares, 0);
    socket.emit("data", Buffer.from('{"id":3,"result":true}\n'));
    assert.equal(poolConfig.good_shares, 1);
  });
});

test("Cortex counts only pending submit response ids, once and out of order", async () => {
  await withMockPool({pool: {protocol: "cortex"}}, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    poolConfig.pending_cortex_submit_ids = new Set([73, 74]);
    for (const id of [4, 72, 75, 100, "073"]) {
      socket.emit("data", Buffer.from(JSON.stringify({id, result: true}) + "\n"));
    }
    assert.equal(poolConfig.good_shares, 0);
    assert.equal(poolConfig.bad_shares, 0);
    socket.emit("data", Buffer.from([
      {id: 74, result: ["11".repeat(32), "", "22".repeat(32), 1]},
      {id: 73, result: false, error: {code: -1, message: "rejected"}},
    ].map((message) => JSON.stringify(message)).join("\n") + "\n"));
    assert.equal(poolConfig.good_shares, 1);
    assert.equal(poolConfig.bad_shares, 1);
    socket.emit("data", Buffer.from('{"id":74,"result":true}\n'));
    assert.equal(poolConfig.good_shares, 1);
  });
});

test("duplicate subscribe responses do not disturb pending authorization", async () => {
  await withMockPool({
    pool: {protocol: "raven", use_subscribe: true},
    opt: {job: {algo: "kawpow"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    assert.equal(poolConfig["pending_subscribe"], true);
    socket.emit("data", Buffer.from('{"id":1,"result":true,"error":null}\n'));
    assert.equal(poolConfig["pending_subscribe"], false);
    assert.equal(poolConfig.pending_authorize, true);
    socket.emit("data", Buffer.from('{"id":1,"result":{"id":"mutated"},"error":null}\n'));
    assert.equal(poolConfig.worker_id, undefined);
    assert.equal(poolConfig.pending_authorize, true);
    socket.emit("data", Buffer.from('{"id":1,"result":null,"error":{"code":-1}}\n'));
    assert.equal(poolConfig.pending_authorize, true);
    socket.emit("data", Buffer.from('{"id":2,"result":true,"error":null}\n'));
    assert.equal(poolConfig.logged_in, true);
    assert.equal(poolConfig.pending_authorize, false);
  });
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

test("unknown notifications and invalid response IDs do not count as shares", async () => {
  await withMockPool({}, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    for (const message of [
      {method: "mining.notify", id: 3, params: []},
      {method: "mining.notify", id: null, params: []},
      {method: "mining.notify", params: []},
      {id: null, result: true},
      {id: {}, result: true},
    ]) {
      socket.emit("data", Buffer.from(JSON.stringify(message) + "\n"));
    }
    assert.equal(poolConfig.good_shares, 0);
    assert.equal(poolConfig.bad_shares, 0);
  });
});

test("Iron Fish nested errors accept string IDs and route unknown handshake errors", async () => {
  const previousLogError = helper.log_err;
  /** @type {string[]} */
  const logs = [];
  helper.log_err = (message) => logs.push(message);
  try {
    await withMockPool({pool: {protocol: "ironfish", login: "ironfish-wallet", pass: "secret"}},
      async ({socket, poolConfig}) => {
        pool.connect_pool_throttle(0, unexpectedPoolJob);
        socket.emit("data", Buffer.from(JSON.stringify({
          id: 0, error: {id: 1, message: "Iron Fish login failed"},
        }) + "\n"));
        assert.equal(poolConfig.logged_in, false);
        assert.equal(poolConfig.bad_shares, 0);

        socket.emit("data", Buffer.from(JSON.stringify({
          id: 0, error: {id: "1", message: "Iron Fish string login failed"},
        }) + "\n"));
        assert.equal(poolConfig.logged_in, false);
        assert.equal(poolConfig.bad_shares, 0);

        socket.emit("data", Buffer.from(JSON.stringify({
          id: 0, error: {id: 2, message: "Iron Fish submit failed"},
        }) + "\n"));
        assert.equal(poolConfig.bad_shares, 1);

        socket.emit("data", Buffer.from(JSON.stringify({
          id: 0, error: {id: "2", message: "Iron Fish string submit failed"},
        }) + "\n"));
        assert.equal(poolConfig.bad_shares, 2);

        socket.emit("data", Buffer.from(JSON.stringify({
          id: 0, method: "mining.subscribed",
          error: {id: 2, message: "Iron Fish method login failed"},
        }) + "\n"));
        assert.equal(poolConfig.logged_in, false);
        assert.equal(poolConfig.bad_shares, 2);

        poolConfig.logged_in = true;
        socket.emit("data", Buffer.from(JSON.stringify({
          id: 0, method: "mining.submitted",
          error: {id: 1, message: "Iron Fish method submit failed"},
        }) + "\n"));
        assert.equal(poolConfig.bad_shares, 3);

        poolConfig.logged_in = true;
        socket.emit("data", Buffer.from(JSON.stringify({
          id: 0, error: {
            id: 99, message: "Iron Fish unknown error ironfish-wallet secret",
          },
        }) + "\n"));
        assert.equal(poolConfig.bad_shares, 3);
      }
    );
  } finally {
    helper.log_err = previousLogError;
  }
  assert.ok(logs.some((message) => message.includes("Login to the pool failed")));
  assert.ok(logs.some((message) => message.includes("Iron Fish submit failed")));
  assert.ok(logs.some((message) => message.includes("Iron Fish string submit failed")));
  assert.ok(logs.some((message) => message.includes("Iron Fish method submit failed")));
  const unknownError = logs.find((message) => message.includes("Iron Fish unknown error"));
  assert.ok(unknownError);
  assert.equal(unknownError.includes("ironfish-wallet"), false);
  assert.equal(unknownError.includes("secret"), false);
  assert.match(unknownError, /<redacted>/);
});

test("Stratum object and array errors are shown with credentials redacted", async () => {
  const previousLogError = helper.log_err;
  /** @type {string[]} */
  const logs = [];
  helper.log_err = (message) => logs.push(message);
  try {
    await withMockPool({pool: {
      protocol: "pearlhash", login: "pearl-wallet", pass: "pearl-secret",
    }}, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      poolConfig["pending_submit_count"] = 2;
      for (const error of [
        {msg: "pearl-wallet rejected pearl-secret"},
        [23, "pearl-wallet rejected pearl-secret", null],
      ]) {
        socket.emit("data", Buffer.from(JSON.stringify({id: 3, error}) + "\n"));
      }
      assert.equal(poolConfig.good_shares, 0);
      assert.equal(poolConfig.bad_shares, 2);
    });
  } finally {
    helper.log_err = previousLogError;
  }
  assert.ok(logs.some((message) => message.includes("rejected")));
  assert.ok(logs.some((message) => message.includes("<redacted>")));
  assert.ok(logs.every((message) => !message.includes("pearl-wallet")));
  assert.ok(logs.every((message) => !message.includes("pearl-secret")));
});

test("non-C29 pool jobs preserve provided blob_hex and nonceoffset", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  assert.equal(typeof setJob, "function");

  setJob({
    algo: "cn/0",
    blob_hex: "ab".repeat(11),
    nonceoffset: 7,
    difficulty: 1,
    id: "worker",
    job_id: "job",
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.ok(jobMessage);
  assert.ok(jobMessage.job);
  assert.equal(jobMessage.job.blob_hex, "ab".repeat(11));
  assert.equal(jobMessage.job.nonceoffset, 7);
});
