"use strict";

const s = require("./support");
const { test, assert, pool, noOp, loadMinerWithStubs, withMockPool, completeOneBenchmark } = s;

test("mine can skip algo benchmark before connecting", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node",
      "mom.js",
      "mine",
      "pool.example:1",
      "wallet",
      "x~kawpow",
      "--bench_algo_params",
      "0",
    ],
    algoParams: { kawpow: "gpu1*[intensity=1]" },
  });

  assert.equal(typeof miner.getSetJob(), "function");
  assert.deepEqual(miner.sentMessages, []);
});

test("fixed-algorithm mining honors explicit GPU selection and fills partial tuning", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "kawpow",
      "--job.dev", "gpu1*[workgroup=128]",
      "--bench_algo_params", "0",
    ],
    algoParams: {kawpow: "gpu1*[intensity=4096;workgroup=256]"},
  });

  assert.equal(
    miner.global.opt.algo_params.kawpow.dev,
    "gpu1*[intensity=4096;workgroup=128]",
  );
});

test("default algo benchmarking only includes MoneroOcean algos plus rx/2", async () => {
  const miner = await loadMinerWithStubs({
    algoParams: {
      "argon2/chukwa": "cpu",
      "argon2/chukwav2": "cpu",
      "cn-heavy/xhv": "cpu",
      "cn-pico/tlo": "cpu",
      "cn/0": "cpu",
      "etchash": "gpu1*[intensity=1]",
      "panthera": "cpu",
      "rx/2": "cpu",
    },
    waitForMessageType: "bench",
  });

  assert.equal(miner.sentMessages[0].job.algo, "etchash");
  completeOneBenchmark(miner);
  assert.equal(miner.sentMessages[1].job.algo, "panthera");
  completeOneBenchmark(miner);
  assert.equal(miner.sentMessages[2].job.algo, "rx/2");
  completeOneBenchmark(miner);
  assert.equal(miner.sentMessages.length, 3);
});

test("bench_algo_params 2 benchmarks all detected algos", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node",
      "mom.js",
      "mine",
      "pool.example:1",
      "wallet",
      "--bench_algo_params",
      "2",
    ],
    algoParams: {
      "argon2/chukwa": "cpu",
      "argon2/chukwav2": "cpu",
      "cn/0": "cpu",
    },
    waitForMessageType: "bench",
  });

  assert.equal(miner.sentMessages[0].job.algo, "argon2/chukwa");
  completeOneBenchmark(miner);
  assert.equal(miner.sentMessages[1].job.algo, "argon2/chukwav2");
  completeOneBenchmark(miner);
  assert.equal(miner.sentMessages[2].job.algo, "cn/0");
  completeOneBenchmark(miner);
  assert.equal(miner.sentMessages.length, 3);
});

test("Windows AMD cn/gpu first-run benchmark discards both cold runtime windows", async () => {
  const miner = await loadMinerWithStubs({
    platform: "win32",
    env: {MOM_GPU_BACKEND: "amd"},
    algoParams: {
      "cn/gpu": "gpu1*[intensity=768],gpu1*[intensity=768]",
      "rx/2": "cpu",
    },
    waitForMessageType: "bench",
  });

  assert.equal(miner.sentMessages[0].job.algo, "cn/gpu");
  const completeWorkerPair = (rate) => {
    miner.messageHandler({type: "hashrate", thread_id: 0, value: {hashrate: String(rate / 2)}});
    miner.messageHandler({type: "hashrate", thread_id: 1, value: {hashrate: String(rate / 2)}});
  };
  completeWorkerPair(1000);
  completeWorkerPair(2500);
  assert.equal(miner.sentMessages.length, 1);
  completeWorkerPair(3100);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(miner.global.opt.algo_params["cn/gpu"].perf, 3100);
  assert.equal(miner.sentMessages[1].job.algo, "rx/2");
});

test("gpu_tune compares two samples and benchmarks the materially faster saved tuning", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--gpu_tune", "1",
    ],
    algoParams: {etchash: "gpu1*[intensity=4096]"},
    waitForMessageType: "bench",
  });
  const finishCandidate = async (rate) => {
    completeOneBenchmark(miner, String(rate));
    completeOneBenchmark(miner, String(rate));
    await new Promise((resolve) => setImmediate(resolve));
  };

  assert.equal(miner.sentMessages[0].job.dev, "gpu1*[intensity=4096]");
  await finishCandidate(100);
  assert.equal(miner.sentMessages[1].job.dev, "gpu1*[intensity=2048]");
  await finishCandidate(90);
  assert.equal(miner.sentMessages[2].job.dev, "gpu1*[intensity=3072]");
  await finishCandidate(105);
  assert.equal(miner.sentMessages[3].job.dev, "gpu1*[intensity=5120]");
  await finishCandidate(104);

  assert.equal(miner.sentMessages[4].job.dev, "gpu1*[intensity=3072]");
  completeOneBenchmark(miner, "103");
  assert.equal(miner.global.opt.algo_params.etchash.dev, "gpu1*[intensity=3072]");
  assert.equal(miner.global.opt.algo_params.etchash.perf, 103);
  assert.equal(miner.global.opt.gpu_tune, 0);
});

test("KawPow benchmark jobs include fixed nonce metadata", async () => {
  const autoBenchmark = await loadMinerWithStubs({
    algoParams: { kawpow: "gpu1*[intensity=1]" },
    waitForMessageType: "bench",
  });
  const directBenchmark = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "kawpow"],
    waitForMessageType: "bench",
  });

  for (const miner of [autoBenchmark, directBenchmark]) {
    const benchMessage = miner.sentMessages.find((msg) => msg.type === "bench");
    assert.equal(benchMessage.job.algo, "kawpow");
    assert.equal(benchMessage.job.noncebytes, 8);
    assert.equal(benchMessage.job.nonceoffset, 32);
  }
});

test("BeamHash III benchmark jobs include fixed M4-shaped nonce metadata", async () => {
  const autoBenchmark = await loadMinerWithStubs({
    argv: ["node", "mom.js", "mine", "pool.example:1", "user", "--bench_algo_params", "2"],
    algoParams: { beamhash3: "gpu1" },
    waitForMessageType: "bench",
  });
  const directBenchmark = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "beamhash3"],
    waitForMessageType: "bench",
  });

  for (const miner of [autoBenchmark, directBenchmark]) {
    const benchMessage = miner.sentMessages.find((msg) => msg.type === "bench");
    assert.equal(benchMessage.job.algo, "beamhash3");
    assert.equal(benchMessage.job.noncebytes, 8);
    assert.equal(benchMessage.job.nonceoffset, 32);
    assert.equal(benchMessage.job.blob_hex.length, 88);
    assert.equal(benchMessage.job.nonce, "0100000000000000");
    assert.equal(benchMessage.job.nicehash_mask, "0000000000000000");
  }
});

test("Etchash benchmark uses current ETC height instead of default seed", async () => {
  const autoBenchmark = await loadMinerWithStubs({
    algoParams: { etchash: "gpu1*[intensity=1]" },
    waitForMessageType: "bench",
  });
  const directBenchmark = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "etchash"],
    waitForMessageType: "bench",
  });

  for (const miner of [autoBenchmark, directBenchmark]) {
    const benchMessage = miner.sentMessages.find((msg) => msg.type === "bench");
    assert.equal(benchMessage.job.algo, "etchash");
    assert.equal(benchMessage.job.height, 24689903);
    assert.equal(benchMessage.job.seed_hex, "");
    assert.equal(benchMessage.job.noncebytes, 8);
    assert.equal(benchMessage.job.nonceoffset, 32);
  }
});

test("pool login does not infer algo from pass when benchmarks are skipped", async () => {
  await withMockPool({
    pool: { pass: "x~kawpow" },
    opt: {
      bench_algo_params: 0,
      job: { algo: null },
      algo_params: { kawpow: { dev: "gpu1*[intensity=1]", perf: null } },
    },
  }, async ({ socket, writes }) => {
    pool.connect_pool_throttle(0, noOp);
    socket.emit("connect");
    const loginMessage = writes[0];
    assert.deepEqual(loginMessage.params.algo, []);
    assert.deepEqual(loginMessage.params["algo-perf"], {});
    assert.equal(loginMessage.params.pass, "x~kawpow");
  });
});

test("pool login advertises raw KawPow performance as kawpow1", async () => {
  await withMockPool({
    opt: {
      algo_params: {
        kawpow: { dev: "gpu1*[intensity=37282560]", perf: 20882200 },
        c29: { dev: "gpu1", perf: 2.79 },
        etchash: { dev: "gpu1*[intensity=33554432]", perf: 21090000 },
      },
    },
  }, async ({ socket, writes }) => {
    pool.connect_pool_throttle(0, noOp);
    socket.emit("connect");
    const loginParams = writes[0].params;
    const algoPerf = loginParams["algo-perf"];
    assert.equal(loginParams.algo.includes("kawpow1"), true);
    assert.equal(loginParams.algo.includes("kawpow"), false);
    assert.equal(algoPerf.kawpow1, 20882200);
    assert.equal("kawpow" in algoPerf, false);
    assert.equal(algoPerf.c29, 2.79 / 42);
    assert.equal(algoPerf.etchash, 21090000);
  });
});

test("PearlHash uses its canonical protocol name for subscribe and authorize", async () => {
  await withMockPool({
    pool: { login: "prl-wallet", worker: "rig", protocol: "pearlhash", use_subscribe: true },
    opt: { job: { algo: "pearlhash" } },
    pool_time: { first_job_wait: 0.001 },
  }, async ({ socket, writes, poolConfig }) => {
    pool.connect_pool_throttle(0, noOp);
    socket.emit("connect");

    assert.equal(writes[0].method, "mining.subscribe");
    assert.equal(writes[1].method, "mining.authorize");
    assert.deepEqual(writes[1].params, { wallet: "prl-wallet", worker: "rig", pass: "x" });
    poolConfig.last_job = {};
  });
});

test("donation pool mines a MoneroOcean algo while the rig is configured for pearlhash", async () => {
  // Regression for pearlhash + donation: the donate pool (xmrig.moneroocean.stream, protocol null) must
  // keep speaking the XMR `login` dialect and advertise a MoneroOcean-supported algo even when the
  // rig's algo context is pearlhash. Otherwise donation would either send pearlhash-protocol traffic the MO
  // pool can't serve, or advertise only pearlhash (which MO cannot assign). MO ignores pearlhash and assigns
  // the other advertised algo (rx/0) in its login result, which must parse as an rx/0 donation job.
  let donatedJob = null;
  await withMockPool({
    pool: { login: "user", pass: "x", use_subscribe: false }, // MO donate pool opts out of pearlhash subscribe
    opt: {
      bench_algo_params: 0,
      job: { algo: "pearlhash" },                            // rig context is pearlhash; must NOT leak to the donate pool
      algo_params: { pearlhash: { dev: "gpu1*[m=131072]", perf: 1 }, "rx/0": { dev: "cpu", perf: 1 } },
    },
  }, async ({ socket, writes }) => {
    pool.connect_pool_throttle(0, (job) => { donatedJob = job; return job; });
    socket.emit("connect");
    const login = writes[0];
    assert.equal(login.method, "login");                          // XMR login, not a pearlhash mining.subscribe
    assert.equal(login.params.algo.includes("rx/0"), true);      // a MoneroOcean-minable algo is offered
    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":{"id":"w","job":' +
      '{"blob":"0101","job_id":"1","target":"c6100000","algo":"rx/0","height":1,"seed_hash":"ab"}}}\n'));
    assert.equal(donatedJob.algo, "rx/0");                        // MO assigns rx/0; donation actually mines
  });
});
