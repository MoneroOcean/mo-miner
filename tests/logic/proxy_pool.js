"use strict";

const s = require("./support");
const {test, pool, completeMiningJob, loadMinerWithStubs, withMockPool} = s;
/** @type {typeof import("node:assert/strict")} */
const assert = s.assert;

const HEADER = "11".repeat(32);
const SEED = "22".repeat(32);
const TARGET = "33".repeat(32);

/** @param {import("node:events").EventEmitter} socket @param {UnknownRecord} message */
function emitPoolMessage(socket, message) {
  socket.emit("data", Buffer.from(JSON.stringify(message) + "\n"));
}

/** @param {string[]} algos @returns {Record<string, {dev: string, perf: number, backend: string, tuning: {}}>} */
function positiveAlgoParams(algos) {
  /** @type {Record<string, {dev: string, perf: number, backend: string, tuning: {}}> } */
  const params = {};
  for (const [index, algo] of algos.entries()) {
    params[algo] = {dev: "cpu", perf: index + 1, backend: "auto", tuning: {}};
  }
  return params;
}

test("object login advertises only accepted native capabilities", async () => {
  const cases = [
    {
      name: "zero positive perfs keep legacy login",
      algoParams: {
        kawpow: {dev: "gpu1", perf: 0}, c29: {dev: "gpu1", perf: -1},
      },
      algos: [], perfs: {}, extensions: [], accepted: [], ack: [], legacyJob: true,
    },
    {
      name: "hash algorithms request both extensions",
      algoParams: positiveAlgoParams(["kawpow", "rx/0"]),
      algos: ["kawpow1", "rx/0"], perfs: {kawpow1: 1, "rx/0": 2},
      extensions: ["mo-native", "submit-result"], accepted: ["mo-native"],
      ack: ["mo-native", "future"],
    },
    {
      name: "C29 retains its divided performance",
      algoParams: {c29: {dev: "gpu1", perf: 84}},
      algos: ["c29"], perfs: {c29: 2}, extensions: ["mo-native", "submit-result"],
      accepted: ["mo-native", "submit-result"],
      ack: ["mo-native", "submit-result"],
    },
    {
      name: "mixed hash and proof capabilities request submit-result",
      algoParams: {kawpow: {dev: "gpu1", perf: 7}, c30: {dev: "gpu1", perf: 42}},
      algos: ["kawpow1", "c30"], perfs: {kawpow1: 7, c30: 42},
      extensions: ["mo-native", "submit-result"], accepted: ["mo-native", "submit-result"],
      ack: ["mo-native", "submit-result"],
    },
    {
      name: "Pearl seed split is retained only after acknowledgement",
      algoParams: {pearlhash: {dev: "gpu1", perf: 1}},
      algos: ["pearlhash"], perfs: {pearlhash: 1},
      extensions: ["mo-native", "pearl-seed-split", "submit-result"],
      accepted: ["mo-native", "submit-result"],
      ack: ["mo-native", "submit-result"],
    },
  ];

  for (const scenario of cases) {
    /** @type {PoolJob[]} */
    const jobs = [];
    await withMockPool({
      pool: {use_subscribe: false, login: "wallet", pass: "x"},
      opt: {job: {algo: "rx/0"}, algo_params: scenario.algoParams},
    }, async ({socket, writes, poolConfig}) => {
      pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
        jobs.push(job);
        return completeMiningJob(job);
      });
      socket.emit("connect");
      assert.equal(writes.length, 1, scenario.name);
      const login = writes[0];
      assert.ok(login, scenario.name);
      assert.equal(login.method, "login", scenario.name);
      const loginParams = /** @type {UnknownRecord} */ (login.params);
      assert.deepEqual(loginParams["algo"], scenario.algos, scenario.name);
      assert.deepEqual(loginParams["algo-perf"], scenario.perfs, scenario.name);
      assert.deepEqual(loginParams["extensions"], scenario.extensions.length ? scenario.extensions : undefined,
        scenario.name);
      assert.deepEqual(poolConfig["requested_algos"], scenario.algos.map((algo) =>
        algo === "kawpow1" ? "kawpow" : algo), scenario.name);
      assert.deepEqual(poolConfig["requested_extensions"], scenario.extensions, scenario.name);

      const result = {
        id: "worker",
        extensions: scenario.ack,
        ...(scenario.legacyJob ? {
          job: {algo: "cn/0", blob_hex: "00".repeat(43), difficulty: 1, job_id: "legacy"},
        } : {}),
      };
      emitPoolMessage(socket, {id: 1, jsonrpc: "2.0", error: null, result});
      assert.deepEqual(poolConfig["extensions"], scenario.accepted, scenario.name);
      assert.equal(poolConfig.logged_in, true, scenario.name);
      if (scenario.legacyJob) {
        assert.equal(jobs.length, 1, scenario.name);
        assert.equal(jobs[0]?.algo, "cn/0", scenario.name);
        assert.equal(jobs[0]?.submit_mode, null, scenario.name);
        assert.equal(jobs[0]?.protocol, "login", scenario.name);
        assert.equal(jobs[0]?.submit_result, false, scenario.name);
      }

      if (scenario.name === "hash algorithms request both extensions") {
        emitPoolMessage(socket, {id: 99, result: {extensions: ["submit-result"]}});
        emitPoolMessage(socket, {method: "pool.extensions", extensions: ["submit-result"]});
        emitPoolMessage(socket, {id: 1, result: [["mining.notify"], "aabb", 4]});
        assert.deepEqual(poolConfig["extensions"], scenario.accepted, scenario.name);
        assert.equal(writes.length, 1, scenario.name);
      }
    });
  }
});

test("Pearl seed slots are validated, randomized, and resumed only when negotiated", async () => {
  /** @param {Awaited<ReturnType<typeof loadMinerWithStubs>>} miner @param {string[]} extensions */
  const configurePearl = (miner, extensions) => {
    miner.global.opt.job.algo = "pearlhash";
    miner.global.opt.job.dev = "cpu";
    miner.global.opt.algo_params["pearlhash"] = {
      dev: "cpu", perf: 1, backend: "auto", tuning: {},
    };
    miner.global.opt.pools[0]["extensions"] = extensions;
  };
  const work = {
    algo: "pearlhash", blob_hex: "00".repeat(76), difficulty: 1,
    job_id: "pearl-1", nonce_slot: 7, nonce_stride: 256,
  };

  const negotiated = await loadMinerWithStubs();
  configurePearl(negotiated, ["mo-native", "pearl-seed-split"]);
  const setNegotiatedJob = negotiated.getSetJob();
  const first = setNegotiatedJob(work);
  const firstNonce = String(first.nonce);
  const firstSeed = Number.parseInt(firstNonce, 16);
  assert.equal(first.nonce_stride, 256);
  assert.equal(firstSeed % 256, 7);
  assert.ok(firstSeed >= 0 && firstSeed <= 0xffffffff);
  const retarget = setNegotiatedJob({...work, job_id: "pearl-2"});
  assert.equal(retarget.nonce, first.nonce);
  const resumed = setNegotiatedJob({...work, job_id: "pearl-3", nonce: firstNonce});
  assert.equal(resumed.nonce, first.nonce);

  const direct = await loadMinerWithStubs();
  configurePearl(direct, []);
  const directJob = direct.getSetJob()({...work, job_id: "direct"});
  assert.equal(directJob.nonce_stride, undefined);
  assert.match(String(directJob.nonce), /^[0-9a-f]{8}$/i);

  for (const invalid of [
    {nonce_slot: 256, nonce_stride: 256},
    {nonce_slot: 0, nonce_stride: 0},
    {nonce_slot: 0, nonce_stride: "4294967296"},
    {nonce_slot: 1, nonce_stride: 3},
  ]) {
    assert.throws(() => setNegotiatedJob({...work, ...invalid}), /PearlHash seed/);
  }

  const benchmark = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "pearlhash"], waitForMessageType: "bench",
  });
  const benchmarkJob = benchmark.sentMessages.find((message) => message.type === "bench")?.job;
  assert.ok(benchmarkJob);
  assert.equal(benchmarkJob.nonce, 0);
  assert.equal((/** @type {UnknownRecord} */ (/** @type {unknown} */ (benchmarkJob)))["nonce_stride"], undefined);
});

test("Pearl seed metadata strings are bounded before bigint conversion", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.job.algo = "pearlhash";
  miner.global.opt.algo_params["pearlhash"] = {
    dev: "cpu", perf: 1, backend: "auto", tuning: {},
  };
  const poolConfig = /** @type {import("./support").TestPoolConfig} */
    (miner.global.opt.pools[/** @type {number} */ (miner.global.opt.pool_ids.active)]);
  poolConfig["extensions"] = ["mo-native", "pearl-seed-split"];
  const setJob = miner.getSetJob();
  assert.ok(setJob);
  const work = {
    algo: "pearlhash", blob_hex: "00".repeat(76), difficulty: 1, job_id: "seed-bounds",
  };
  const originalBigInt = global.BigInt;
  let oversizedConversions = 0;
  global.BigInt = new Proxy(originalBigInt, {
    apply(target, thisArg, args) {
      if (typeof args[0] === "string" && args[0].length > 16 && /^\d+$/.test(args[0])) {
        oversizedConversions++;
      }
      return Reflect.apply(target, thisArg, args);
    },
  });
  try {
    for (const value of ["0".repeat(16) + "1", "0".repeat(100000) + "1", "9".repeat(100000)]) {
      const messageCount = miner.sentMessages.length;
      assert.throws(() => setJob({...work, nonce_stride: value}), /Invalid PearlHash seed stride/);
      assert.equal(oversizedConversions, 0);
      assert.equal(miner.sentMessages.length, messageCount);
    }
    const maximum = setJob({...work, nonce_slot: "2147483647", nonce_stride: "2147483648"});
    assert.equal(maximum.nonce_stride, 2147483648);
    assert.equal(Number.parseInt(String(maximum.nonce), 16) % 2147483648, 2147483647);
    const padded = setJob({...work, nonce_slot: "0000000000000000", nonce_stride: "0000000000000001"});
    assert.equal(padded.nonce_stride, 1);
    assert.match(String(padded.nonce), /^[0-9a-f]{8}$/i);
    const numeric = setJob({...work, nonce_slot: 2147483647, nonce_stride: 2147483648});
    assert.equal(numeric.nonce_stride, 2147483648);
    assert.throws(() => setJob({...work, nonce_slot: "0".repeat(17), nonce_stride: 1}),
      /Invalid PearlHash seed slot/);
    assert.equal(oversizedConversions, 0);
    poolConfig["extensions"] = [];
    const direct = setJob({...work, nonce_slot: "0".repeat(100000), nonce_stride: "0".repeat(100000) + "1"});
    assert.equal(direct.nonce_stride, undefined);
    const other = setJob({...work, algo: "cn/0", nonce_slot: "0".repeat(100000), nonce_stride: "0".repeat(100000) + "1"});
    assert.equal(other.algo, "cn/0");
    assert.equal(oversizedConversions, 0);
  } finally {
    global.BigInt = originalBigInt;
  }
});

test("Pearl proxy login jobs preserve the negotiated seed partition", async () => {
  /** @type {PoolJob[]} */
  const jobs = [];
  await withMockPool({
    pool: {use_subscribe: false, login: "wallet", pass: "x"},
    opt: {job: {algo: "pearlhash"}, algo_params: positiveAlgoParams(["pearlhash"])},
  }, async ({socket}) => {
    pool.connect_pool_throttle(0, (job) => {
      jobs.push(job);
      return completeMiningJob(job);
    });
    socket.emit("connect");
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {
        id: "worker", algo: "pearlhash",
        extensions: ["mo-native", "pearl-seed-split", "submit-result"],
        job: {
          algo: "pearlhash", job_id: "pearl-login", header: "00".repeat(76),
          target: "01".repeat(32), cert_version: 3, nonce_slot: 7, nonce_stride: 256,
        },
      },
    });
  });

  assert.equal(jobs.length, 1);
  assert.equal(jobs[0]?.nonce_slot, 7);
  assert.equal(jobs[0]?.nonce_stride, 256);

  await withMockPool({
    pool: {use_subscribe: false, login: "wallet", pass: "x"},
    opt: {job: {algo: "pearlhash"}, algo_params: positiveAlgoParams(["pearlhash"])},
  }, async ({socket}) => {
    pool.connect_pool_throttle(0, () => {
      assert.fail("malformed Pearl seed partition reached the miner");
    });
    socket.emit("connect");
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {
        id: "worker", algo: "pearlhash",
        extensions: ["mo-native", "pearl-seed-split", "submit-result"],
        job: {
          algo: "pearlhash", job_id: "pearl-bad", header: "00".repeat(76),
          target: "01".repeat(32), cert_version: 3, nonce_slot: 7, nonce_stride: null,
        },
      },
    });
    assert.equal(socket.destroyed, true);
  });
});

test("login metadata survives a generic job and malformed login jobs close the pool", async () => {
  /** @type {PoolJob | undefined} */
  let received;
  await withMockPool({
    pool: {use_subscribe: false, login: "wallet", pass: "x"},
    opt: {job: {algo: "rx/0"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, (job) => {
      received = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {
        id: "worker-7", algo: "rx/0", extra_nonce: "a1b2", extra_nonce2_size: 4,
        noncebytes: 8, nonceoffset: 39, xn: "a1b2", nicehash_mask: "ffff000000000000",
        job: {job_id: "login-job", blob_hex: "00".repeat(43), difficulty: 1},
      },
    });
    assert.ok(received);
    assert.equal(poolConfig.worker_id, "worker-7");
    assert.equal(received?.algo, "rx/0");
    assert.equal(received?.extra_nonce, "a1b2");
    assert.equal(received?.extra_nonce2_size, 4);
    assert.equal(received?.noncebytes, 8);
    assert.equal(received?.nonceoffset, 39);
    assert.equal(received?.xn, "a1b2");
    assert.equal(received?.nicehash_mask, "ffff000000000000");
    assert.equal(poolConfig.logged_in, true);
  });

  /** @type {PoolJob[]} */
  const jobs = [];
  await withMockPool({
    pool: {use_subscribe: false, login: "wallet", pass: "x"},
    opt: {job: {algo: "rx/0"}},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, (job) => {
      jobs.push(job);
      return completeMiningJob(job);
    });
    socket.emit("connect");
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {id: "worker-7", algo: "rx/0", job: {job_id: "bad-login"}},
    });
    assert.equal(jobs.length, 0);
    assert.equal(socket.destroyed, true);
    assert.equal(poolConfig.last_job, null);
  });
});

test("native job markers switch generic and protocol-specific families", async () => {
  const algoParams = positiveAlgoParams([
    "rx/0", "kawpow", "etchash", "autolykos2", "c29", "cn/0",
  ]);
  /** @type {PoolJob[]} */
  const jobs = [];
  await withMockPool({
    pool: {use_subscribe: false, login: "wallet", pass: "x"},
    opt: {job: {algo: "rx/0"}, algo_params: algoParams},
  }, async ({socket, writes}) => {
    pool.connect_pool_throttle(0, (job) => {
      jobs.push(job);
      return completeMiningJob(job);
    });
    socket.emit("connect");
    assert.equal(writes.length, 1);
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {id: "worker", algo: "rx/0", extensions: ["mo-native", "submit-result"]},
    });

    emitPoolMessage(socket, {
      method: "job",
      params: {algo: "rx/0", job_id: "rx", blob_hex: "00".repeat(43), difficulty: 1},
    });
    emitPoolMessage(socket, {
      method: "mining.notify", algo: "kawpow1",
      params: ["raven", HEADER, SEED, TARGET, true, 100],
    });
    emitPoolMessage(socket, {
      method: "mining.notify", algo: "etchash",
      params: ["eth", SEED, HEADER, true],
    });
    emitPoolMessage(socket, {
      method: "mining.notify", algo: "autolykos2",
      params: ["erg", 200, HEADER, "", "", 0, "1", "00000002"],
    });
    emitPoolMessage(socket, {
      method: "job",
      params: {algo: "cuckaroo", job_id: "c29", blob_hex: "44".repeat(43), difficulty: 1},
    });
    emitPoolMessage(socket, {
      method: "job",
      params: {algo: "cn/0", job_id: "cn", blob_hex: "55".repeat(43), difficulty: 1},
    });
  });

  assert.deepEqual(jobs.map((job) => [
    job.algo, job.submit_mode, job.protocol, job.submit_result,
  ]), [
    ["rx/0", null, "login", true],
    ["kawpow", "raven", "raven", true],
    ["etchash", "eth", "eth", true],
    ["autolykos2", "erg", "erg", true],
    ["c29", null, "login", true],
    ["cn/0", null, "login", true],
  ]);
});

test("native controls coalesce by marker and apply only to the next family", async () => {
  const oldTarget = "44".repeat(32);
  const newTarget = "55".repeat(32);
  /** @type {PoolJob[]} */
  const jobs = [];
  await withMockPool({
    pool: {use_subscribe: false, login: "wallet", pass: "x"},
    opt: {
      job: {algo: "rx/0"},
      algo_params: positiveAlgoParams(["kawpow", "etchash"]),
    },
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, (job) => {
      jobs.push(job);
      return completeMiningJob(job);
    });
    socket.emit("connect");
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {id: "worker", extensions: ["mo-native", "submit-result"]},
    });

    emitPoolMessage(socket, {
      method: "mining.notify", algo: "kawpow1",
      params: ["old", HEADER, SEED, TARGET, true, 100],
    });
    emitPoolMessage(socket, {method: "mining.set_target", params: ["66".repeat(32)]});
    emitPoolMessage(socket, {method: "mining.set_target", params: [newTarget]});
    emitPoolMessage(socket, {method: "mining.set_target", algo: "kawpow1", params: [oldTarget]});
    emitPoolMessage(socket, {method: "mining.set_difficulty", params: [3]});
    emitPoolMessage(socket, {method: "mining.set_difficulty", params: [4]});
    emitPoolMessage(socket, {method: "mining.set_difficulty", algo: "kawpow1", params: [5]});
    emitPoolMessage(socket, {method: "mining.set_extranonce", params: ["0102", 6]});
    emitPoolMessage(socket, {method: "mining.set_extranonce", params: ["0304", 6]});
    emitPoolMessage(socket, {method: "mining.set_extranonce", algo: "kawpow1", params: ["badc0ffe", 4]});
    const pendingControls = poolConfig["pending_controls"];
    assert.ok(Array.isArray(pendingControls));
    assert.equal(pendingControls.length, 6);

    emitPoolMessage(socket, {
      method: "mining.notify", algo: "etchash",
      params: ["new", SEED, HEADER, true],
    });
    assert.equal(jobs.length, 2);
    assert.equal(jobs[0]?.submit_mode, "raven");
    assert.equal(jobs[1]?.algo, "etchash");
    assert.equal(jobs[1]?.submit_mode, "eth");
    assert.equal(jobs[1]?.target, newTarget);
    assert.equal(jobs[1]?.difficulty, undefined);
    assert.equal(jobs[1]?.nonce, "0304000000000000");
    assert.equal(jobs[1]?.nicehash_mask, "ffff000000000000");
    assert.equal(poolConfig["eth_target"], newTarget);
    assert.equal(poolConfig["eth_difficulty"], 4);
    assert.equal(poolConfig["extra_nonce"], "0304");
    assert.equal(poolConfig["extra_nonce2_size"], 6);
    assert.equal(poolConfig["raven_target"], undefined);
  });
});

test("native nonce prefix snapshots preserve four, eight, and empty-byte controls", async () => {
  const cases = [
    {initial: "a1b2c3d4", later: "aabbccdd", initialMask: "ffffffff00000000",
      initialBlob: "00000000d4c3b2a1", laterNonce: "aabbccdd00000000",
      laterMask: "ffffffff00000000"},
    {initial: "0102030405060708", later: "aabbccdd", initialMask: "ffffffffffffffff",
      initialBlob: "0807060504030201", laterNonce: "aabbccdd00000000",
      laterMask: "ffffffff00000000"},
  ];

  for (const scenario of cases) {
    /** @type {PoolJob[]} */
    const jobs = [];
    await withMockPool({
      pool: {use_subscribe: false, login: "wallet", pass: "x"},
      opt: {job: {algo: "rx/0"}, algo_params: positiveAlgoParams(["kawpow"])},
    }, async ({socket}) => {
      pool.connect_pool_throttle(0, (job) => {
        jobs.push(job);
        return completeMiningJob(job);
      });
      socket.emit("connect");
      emitPoolMessage(socket, {
        id: 1, jsonrpc: "2.0", error: null,
        result: {id: "worker", extensions: ["mo-native", "submit-result"]},
      });
      emitPoolMessage(socket, {
        method: "mining.set_extranonce", params: [scenario.initial, 8 - scenario.initial.length / 2],
      });
      emitPoolMessage(socket, {
        method: "mining.notify", algo: "kawpow1",
        params: ["first", HEADER, SEED, TARGET, true, 100],
      });
      emitPoolMessage(socket, {
        method: "mining.set_extranonce", params: [scenario.later, 4],
      });
      emitPoolMessage(socket, {
        method: "mining.notify", algo: "kawpow1",
        params: ["second", HEADER, SEED, TARGET, true, 101],
      });
      emitPoolMessage(socket, {method: "mining.set_extranonce", params: ["", 8]});
      emitPoolMessage(socket, {
        method: "mining.notify", algo: "kawpow1",
        params: ["third", HEADER, SEED, TARGET, true, 102],
      });
    });

    assert.equal(jobs.length, 3);
    assert.equal(jobs[0]?.nonce, scenario.initial.padEnd(16, "0"));
    assert.equal(jobs[0]?.nicehash_mask, scenario.initialMask);
    assert.equal(jobs[0]?.noncebytes, 8);
    assert.equal(jobs[0]?.nonceoffset, 32);
    assert.equal(jobs[0]?.blob?.slice(-16), scenario.initialBlob);
    assert.equal(jobs[1]?.nonce, scenario.laterNonce);
    assert.equal(jobs[1]?.nicehash_mask, scenario.laterMask);
    assert.equal(jobs[1]?.noncebytes, 8);
    assert.equal(jobs[1]?.nonceoffset, 32);
    assert.equal(jobs[2]?.nonce, "0000000000000000");
    assert.equal(jobs[2]?.nicehash_mask, "0000000000000000");
    assert.equal(jobs[0]?.nonce, scenario.initial.padEnd(16, "0"));
    assert.equal(jobs[0]?.nicehash_mask, scenario.initialMask);
  }
});

test("negotiated generic jobs convert pool nonce prefixes through the real miner adapter", async () => {
  const cases = [
    {noncebytes: 4, prefix: "a1b2", remaining: 2, blobBytes: 43,
      expectedNonce: "a1b20000", expectedMask: "ffff0000"},
    {noncebytes: 8, prefix: "a1b2c3", remaining: 5, blobBytes: 47,
      expectedNonce: "a1b2c30000000000", expectedMask: "ffffff0000000000"},
  ];

  for (const scenario of cases) {
    const miner = await loadMinerWithStubs();
    const setJob = miner.getSetJob();
    /** @type {MiningJob[]} */
    const nativeJobs = [];
    await withMockPool({
      pool: {use_subscribe: false, login: "wallet", pass: "x"},
      opt: {
        job: {algo: "cn/0"},
        algo_params: {"cn/0": {dev: "cpu", perf: 1, backend: "auto", tuning: {}}},
      },
    }, async ({socket, writes, poolConfig}) => {
      miner.global.opt.pools[0] = poolConfig;
      pool.connect_pool_throttle(0, (job) => {
        const nativeJob = setJob(job);
        nativeJobs.push(nativeJob);
        return nativeJob;
      });
      socket.emit("connect");
      const login = writes[0];
      assert.ok(login);
      const loginParams = /** @type {UnknownRecord} */ (login.params);
      assert.deepEqual(loginParams["algo"], ["cn/0"]);
      emitPoolMessage(socket, {
        id: 1, jsonrpc: "2.0", error: null,
        result: {id: "worker", extensions: ["mo-native"]},
      });
      emitPoolMessage(socket, {
        method: "mining.set_extranonce", params: [scenario.prefix, scenario.remaining],
      });
      emitPoolMessage(socket, {
        method: "job",
        params: {
          algo: "cn/0", job_id: "first", blob_hex: "00".repeat(scenario.blobBytes),
          noncebytes: scenario.noncebytes, nonceoffset: 39, difficulty: 1,
        },
      });

      const firstWire = poolConfig.last_job;
      const firstNative = nativeJobs[0];
      assert.ok(firstWire);
      assert.ok(firstNative);
      assert.equal(firstWire.xn, scenario.prefix);
      assert.equal(firstWire.noncebytes, scenario.noncebytes);
      assert.equal(firstWire.nonceoffset, 39);
      assert.equal(firstNative.nonce, scenario.expectedNonce);
      assert.equal(firstNative.nicehash_mask, scenario.expectedMask);

      const replacementPrefix = scenario.noncebytes === 8 ? "ddeeff" : "ddee";
      emitPoolMessage(socket, {
        method: "mining.set_extranonce", params: [replacementPrefix, scenario.remaining],
      });
      emitPoolMessage(socket, {
        method: "job",
        params: {
          algo: "cn/0", job_id: "second", blob_hex: "11".repeat(scenario.blobBytes),
          noncebytes: scenario.noncebytes, nonceoffset: 39, difficulty: 1,
        },
      });
      assert.equal(firstWire.xn, scenario.prefix);
      assert.equal(firstWire.noncebytes, scenario.noncebytes);
      assert.equal(firstNative.nonce, scenario.expectedNonce);
      assert.equal(firstNative.nicehash_mask, scenario.expectedMask);
      assert.equal(nativeJobs.length, 2);
    });
  }
});

test("negotiated generic nonce controls reject mismatches and preserve explicit prefixes", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  /** @type {MiningJob[]} */
  const nativeJobs = [];
  await withMockPool({
    pool: {use_subscribe: false, login: "wallet", pass: "x"},
    opt: {
      job: {algo: "cn/0"},
      algo_params: {"cn/0": {dev: "cpu", perf: 1, backend: "auto", tuning: {}}},
    },
  }, async ({socket, poolConfig}) => {
    miner.global.opt.pools[0] = poolConfig;
    pool.connect_pool_throttle(0, (job) => {
      const nativeJob = setJob(job);
      nativeJobs.push(nativeJob);
      return nativeJob;
    });
    socket.emit("connect");
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {id: "worker", extensions: ["mo-native"]},
    });
    emitPoolMessage(socket, {
      method: "mining.set_extranonce", params: ["a1b2", 1],
    });
    emitPoolMessage(socket, {
      method: "job",
      params: {
        algo: "cn/0", job_id: "mismatch", blob_hex: "00".repeat(43),
        noncebytes: 4, nonceoffset: 39, difficulty: 1,
      },
    });
    assert.equal(nativeJobs.length, 0);
    assert.equal(socket.destroyed, true);
    assert.equal(poolConfig.last_job, null);
  });

  const explicit = await loadMinerWithStubs();
  const explicitSetJob = explicit.getSetJob();
  await withMockPool({
    pool: {use_subscribe: false, login: "wallet", pass: "x"},
    opt: {
      job: {algo: "cn/0"},
      algo_params: {"cn/0": {dev: "cpu", perf: 1, backend: "auto", tuning: {}}},
    },
  }, async ({socket, poolConfig}) => {
    explicit.global.opt.pools[0] = poolConfig;
    pool.connect_pool_throttle(0, explicitSetJob);
    socket.emit("connect");
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {id: "worker", extensions: ["mo-native"]},
    });
    emitPoolMessage(socket, {
      method: "mining.set_extranonce", params: ["a1b2", 2],
    });
    emitPoolMessage(socket, {
      method: "job",
      params: {
        algo: "cn/0", job_id: "explicit", blob_hex: "00".repeat(43),
        xn: "c3d4", noncebytes: 4, nonceoffset: 39, difficulty: 1,
      },
    });
    assert.equal(poolConfig.last_job?.xn, "c3d4");
    assert.equal(explicit.sentMessages.at(-1)?.job?.nonce, "c3d40000");
    assert.equal(explicit.sentMessages.at(-1)?.job?.nicehash_mask, "ffff0000");
  });
});

test("negotiated empty generic prefix uses a zero mask on a NiceHash pool", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  await withMockPool({
    pool: {use_subscribe: false, is_nicehash: true, login: "wallet", pass: "x"},
    opt: {
      job: {algo: "cn/0"},
      algo_params: {"cn/0": {dev: "cpu", perf: 1, backend: "auto", tuning: {}}},
    },
  }, async ({socket, poolConfig}) => {
    miner.global.opt.pools[0] = poolConfig;
    pool.connect_pool_throttle(0, setJob);
    socket.emit("connect");
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {id: "worker", extensions: ["mo-native"]},
    });
    emitPoolMessage(socket, {
      method: "mining.set_extranonce", params: ["", 4],
    });
    emitPoolMessage(socket, {
      method: "job",
      params: {
        algo: "cn/0", job_id: "empty", blob_hex: "00".repeat(43),
        noncebytes: 4, nonceoffset: 39, difficulty: 1,
      },
    });
    const job = miner.sentMessages.at(-1)?.job;
    assert.equal(poolConfig.last_job?.xn, "");
    assert.equal(job?.nonce, "0");
    assert.equal(job?.nicehash_mask, "00000000");
  });
});

test("reconnect clears native negotiation and pending controls", async () => {
  await withMockPool({
    pool: {
      use_subscribe: false, login: "wallet", pass: "x",
      is_keepalive: false, is_nicehash: false,
    },
    opt: {job: {algo: "rx/0"}, algo_params: positiveAlgoParams(["kawpow"])},
  }, async ({socket, poolConfig, switched}) => {
    pool.connect_pool_throttle(0, (job) => completeMiningJob(job));
    socket.emit("connect");
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {
        id: "worker", algo: "kawpow",
        extensions: ["mo-native", "submit-result", "keepalive", "nicehash"],
      },
    });
    emitPoolMessage(socket, {method: "mining.set_extranonce", params: ["aabb", 2]});
    emitPoolMessage(socket, {
      method: "mining.notify", algo: "kawpow1",
      params: ["job", HEADER, SEED, TARGET, true, 100],
    });
    emitPoolMessage(socket, {method: "mining.set_target", params: [TARGET]});
    const extensions = poolConfig["extensions"];
    assert.ok(Array.isArray(extensions));
    assert.ok(extensions.includes("mo-native"));
    assert.ok(poolConfig.last_job);
    assert.ok(poolConfig["pending_controls"]);
    assert.equal(poolConfig.is_keepalive, false);
    assert.equal(poolConfig.is_nicehash, false);
    assert.equal(poolConfig["negotiated_keepalive"], true);
    assert.equal(poolConfig["negotiated_nicehash"], true);

    socket.emit("error", new Error("test reconnect"));
    assert.equal(socket.destroyed, true);
    assert.equal(switched(), true);
    for (const key of [
      "extensions", "requested_extensions", "requested_algos", "job_algo", "pending_controls",
      "inferred_protocol", "negotiated_keepalive", "negotiated_nicehash", "worker_id",
      "extra_nonce", "extra_nonce2_size", "raven_target",
    ]) {
      assert.equal(poolConfig[key], undefined, key);
    }
    assert.equal(poolConfig.last_job, null);
    assert.equal(poolConfig.logged_in, false);
  });
});

test("MO-mapped generic algorithms retain their normalized names", async () => {
  /** @type {Array<[string, string]>} */
  const cases = [
    ["argon2/chukwav2", "argon2/chukwav2"], ["autolykos2", "autolykos2"],
    ["c29", "c29"], ["cn-heavy/xhv", "cn-heavy/xhv"], ["cn/gpu", "cn/gpu"],
    ["cn/half", "cn/half"], ["cn/r", "cn/r"], ["etchash", "etchash"],
    ["ghostrider", "ghostrider"], ["kawpow1", "kawpow"], ["panthera", "panthera"],
    ["rx/0", "rx/0"], ["rx/arq", "rx/arq"],
  ];
  const params = positiveAlgoParams(cases.map(([algo]) => algo));
  for (const [advertised, expected] of cases) {
    /** @type {PoolJob[]} */
    const jobs = [];
    await withMockPool({
      pool: {use_subscribe: false, login: "wallet", pass: "x"},
      opt: {job: {algo: "rx/0"}, algo_params: params},
    }, async ({socket}) => {
      pool.connect_pool_throttle(0, (job) => {
        jobs.push(job);
        return completeMiningJob(job);
      });
      socket.emit("connect");
      emitPoolMessage(socket, {
        id: 1, jsonrpc: "2.0", error: null,
        result: {id: "worker", extensions: ["mo-native", "submit-result"]},
      });
      emitPoolMessage(socket, {
        method: "job",
        params: {algo: advertised, job_id: advertised, blob_hex: "00".repeat(43), difficulty: 1},
      });
    });
    assert.equal(jobs.length, 1, advertised);
    assert.equal(jobs[0]?.algo, expected, advertised);
    assert.equal(jobs[0]?.submit_mode, null, advertised);
    assert.equal(jobs[0]?.submit_result, true, advertised);
  }
});

test("native switching canonicalizes XELIS protocol aliases", async () => {
  /** @type {PoolJob[]} */
  const jobs = [];
  await withMockPool({
    pool: {use_subscribe: false, login: "wallet", pass: "x"},
    opt: {job: {algo: "xelishashv3"}, algo_params: positiveAlgoParams(["xelishashv3"])},
  }, async ({socket}) => {
    pool.connect_pool_throttle(0, (job) => {
      jobs.push(job);
      return completeMiningJob(job);
    });
    socket.emit("connect");
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {id: "worker", extensions: ["mo-native", "submit-result"]},
    });
    emitPoolMessage(socket, {
      method: "job",
      params: {algo: "xel/3", job_id: "xelis-job", blob_hex: "00".repeat(112), difficulty: 1},
    });
  });
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0]?.algo, "xelishashv3");
});

test("unadvertised and conflicting native markers fail before set_job", async () => {
  for (const advertised of ["astrobwt/v2", "ethash", "flex"]) {
    /** @type {PoolJob[]} */
    const jobs = [];
    await withMockPool({
      pool: {use_subscribe: false, login: "wallet", pass: "x"},
      opt: {job: {algo: "rx/0"}, algo_params: positiveAlgoParams(["rx/0"])},
    }, async ({socket, poolConfig}) => {
      pool.connect_pool_throttle(0, (job) => {
        jobs.push(job);
        return completeMiningJob(job);
      });
      socket.emit("connect");
      emitPoolMessage(socket, {
        id: 1, jsonrpc: "2.0", error: null,
        result: {id: "worker", extensions: ["mo-native", "submit-result"]},
      });
      emitPoolMessage(socket, {
        method: "job",
        params: {algo: advertised, job_id: advertised, blob_hex: "00".repeat(43), difficulty: 1},
      });
      assert.equal(jobs.length, 0, advertised);
      assert.equal(socket.destroyed, true, advertised);
      assert.equal(poolConfig.last_job, null, advertised);
    });
  }

  /** @type {PoolJob[]} */
  const conflictJobs = [];
  await withMockPool({
    pool: {use_subscribe: false, login: "wallet", pass: "x"},
    opt: {job: {algo: "rx/0"}, algo_params: positiveAlgoParams(["rx/0", "kawpow"])},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, (job) => {
      conflictJobs.push(job);
      return completeMiningJob(job);
    });
    socket.emit("connect");
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {id: "worker", extensions: ["mo-native"]},
    });
    emitPoolMessage(socket, {
      method: "job", algo: "rx/0",
      params: {algo: "kawpow", job_id: "conflict", blob_hex: "00".repeat(43), difficulty: 1},
    });
    assert.equal(conflictJobs.length, 0);
    assert.equal(socket.destroyed, true);
    assert.equal(poolConfig.last_job, null);
  });
});

test("a malformed negotiated notify closes instead of retaining its new family", async () => {
  /** @type {PoolJob[]} */
  const jobs = [];
  await withMockPool({
    pool: {use_subscribe: false, login: "wallet", pass: "x"},
    opt: {job: {algo: "rx/0"}, algo_params: positiveAlgoParams(["rx/0", "kawpow"])},
  }, async ({socket, poolConfig}) => {
    pool.connect_pool_throttle(0, (job) => {
      jobs.push(job);
      return completeMiningJob(job);
    });
    socket.emit("connect");
    emitPoolMessage(socket, {
      id: 1, jsonrpc: "2.0", error: null,
      result: {id: "worker", algo: "rx/0", extensions: ["mo-native"]},
    });
    emitPoolMessage(socket, {
      method: "job",
      params: {algo: "rx/0", job_id: "initial", blob_hex: "00".repeat(43), difficulty: 1},
    });
    emitPoolMessage(socket, {
      method: "mining.notify", algo: "kawpow1",
      params: ["bad", "not-a-header", SEED, TARGET, true, 100],
    });
    assert.equal(jobs.length, 1);
    assert.equal(socket.destroyed, true);
    assert.equal(poolConfig.last_job, null);
    assert.equal(poolConfig["job_algo"], undefined);
    assert.equal(poolConfig["inferred_protocol"], undefined);
  });
});
