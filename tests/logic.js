"use strict";

const { describe, test } = require("node:test");
const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const events = require("node:events");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");
const tls = require("node:tls");
const vm = require("node:vm");

const opts = require("../opts.js");
const helper = require("../helper.js");
const pool = require("../pool.js");
const { formatHashrate, parseFormattedHashrate } = require("./common/miner_command");
const specReporter = require("./common/spec_reporter");
const repoRoot = path.join(__dirname, "..");
const noOp = () => undefined;

async function loadMinerWithStubs(options = {}) {
  const source = fs.readFileSync(path.join(repoRoot, "mom.js"), "utf8");
  const moduleStub = { exports: {} };
  const globalStub = {};
  const coreEvents = new events.EventEmitter();
  const sentMessages = [];
  const poolWrites = [];
  let capturedSetJob = null;
  const algoParams = options.algoParams || {};
  const helperStub = {
    ...helper,
    cluster_process: () => false,
    create_core: () => ({
      from: coreEvents,
      emit_to: (name) => {
        if (name === "algo_params") {setImmediate(() => coreEvents.emit("algo_params", algoParams));}
        if (name === "read_msr") {setImmediate(() => coreEvents.emit("error", { message: "skip" }));}
      },
    }),
    recreate_threads: noOp,
    messageWorkers: (msg) => sentMessages.push(msg),
    log: noOp,
    log1: noOp,
    log2: noOp,
    log3: noOp,
    log_err: noOp,
  };
  const poolStub = {
    connect_pool_throttle: (pool_id, setJob) => { capturedSetJob = setJob; },
    pool_write: (pool_id, json) => poolWrites.push({ pool_id, json }),
  };
  const processStub = Object.create(process);
  processStub.argv = options.argv || ["node", "mom.js", "mine", "pool.example:1", "user"];
  processStub.env = { ...process.env };
  processStub.exit = (code) => { throw new Error(`unexpected exit ${code}`); };
  const detachedSetTimeout = (...args) => {
    const timer = setTimeout(...args);
    if (timer.unref) {timer.unref();}
    return timer;
  };
  const requireStub = (id) => {
    if (id === "./helper.js") {return helperStub;}
    if (id === "./pool.js") {return poolStub;}
    if (id === "./opts.js") {return opts;}
    return require(id);
  };

  vm.runInNewContext(
    `(function(require, module, exports, process, global, console, Buffer, setTimeout, clearTimeout, setInterval, setImmediate) { ${source}\nmodule.exports.__test = { expectedTestThreads, messageHandler };\n})`,
    {},
  )(requireStub, moduleStub, moduleStub.exports, processStub, globalStub, console, Buffer, detachedSetTimeout, clearTimeout, noOp, setImmediate);

  const hasExpectedMessage = () =>
    options.waitForMessageType && sentMessages.some((msg) => msg.type === options.waitForMessageType);
  for (let i = 0; i < 10 && !capturedSetJob && !hasExpectedMessage(); ++i) {
    await new Promise((resolve) => setImmediate(resolve));
  }
  return {
    getSetJob: () => capturedSetJob,
    global: globalStub,
    expectedTestThreads: moduleStub.exports.__test.expectedTestThreads,
    messageHandler: moduleStub.exports.__test.messageHandler,
    poolWrites,
    sentMessages,
  };
}

function mockPoolConfig(overrides = {}) {
  return {
    url: "pool.example",
    port: 1,
    is_tls: false,
    is_keepalive: false,
    socket: null,
    keepalive: null,
    last_job: null,
    last_connect_time: 0,
    good_shares: 0,
    bad_shares: 0,
    login: "wallet",
    pass: "x",
    logged_in: false,
    ...overrides,
  };
}

function mockPoolOptions(options = {}) {
  return {
    log_level: 0,
    job: {},
    pools: [mockPoolConfig(options.pool)],
    pool_ids: { active: 0, primary: 0, donate: null },
    pool_time: { first_job_wait: 0.001, connect_throttle: 60, close_wait: 60, keepalive: 60, ...options.pool_time },
    algo_params: {},
    ...options.opt,
  };
}

async function withMockPool(options, callback) {
  const originalConnect = net.connect;
  const originalSwitchPool = pool.switch_pool;
  const previousOpt = global.opt;
  const socket = options.socket || new events.EventEmitter();
  const writes = [];
  let switched = false;

  socket.write = options.write || function(message) { writes.push(JSON.parse(message)); };
  socket.destroy = options.destroy || function() { this.destroyed = true; };
  net.connect = function() { return socket; };
  if (options.switchPool) {pool.switch_pool = function() { switched = true; };}
  global.opt = mockPoolOptions(options);

  try {
    return await callback({ socket, writes, switched: () => switched, poolConfig: global.opt.pools[0] });
  } finally {
    for (const poolConfig of global.opt.pools) {poolConfig.socket = null;}
    await new Promise((resolve) => setTimeout(resolve, 5));
    net.connect = originalConnect;
    pool.switch_pool = originalSwitchPool;
    global.opt = previousOpt;
  }
}

function completeOneBenchmark(miner, rate = "1") {
  miner.messageHandler({ type: "hashrate", thread_id: 0, value: { hashrate: rate } });
}

describe("JavaScript logic tests", () => {
test("saved config omits job without mutating live options", () => {
  const opt = {
    job: { algo: "rx/0", dev: "cpu" },
    pools: [{ url: "pool.example", port: 443 }],
    pool_ids: { primary: 0 },
  };

  const saved = opts.saved_config(opt);

  assert.deepEqual(saved, {
    pools: [{ url: "pool.example", port: 443 }],
    pool_ids: { primary: 0 },
  });
  assert.deepEqual(opt.job, { algo: "rx/0", dev: "cpu" });
});

test("config file detection requires a .json extension", () => {
  assert.equal(opts.is_config_file("config.json"), true);
  assert.equal(opts.is_config_file("CONFIG.JSON"), true);
  assert.equal(opts.is_config_file("pooljson"), false);
  assert.equal(opts.is_config_file("config-json"), false);
});

test("default options do not share array or map references", () => {
  const optHelp = {
    pool: { _array: [{ url: "a", nested: { enabled: true } }] },
    algo_param: { _map: { rx: { dev: "cpu" } } },
  };
  const one = {};
  const two = {};

  opts.set_default_opts(one, optHelp);
  opts.set_default_opts(two, optHelp);
  one.pools[0].url = "changed";
  one.pools[0].nested.enabled = false;
  one.algo_params.rx.dev = "gpu0";

  assert.equal(two.pools[0].url, "a");
  assert.equal(two.pools[0].nested.enabled, true);
  assert.equal(two.algo_params.rx.dev, "cpu");
});

test("unparsed CLI options fail before runtime startup", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "rx/0",
    "--definitely-bad-option",
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unparsed option: --definitely-bad-option/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("JSON options reject non-object values cleanly", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "rx/0",
    "--job",
    "null",
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /JSON param must be an object/);
  assert.doesNotMatch(result.stderr, /TypeError|Cannot use 'in' operator|Cannot find module/);
});

test("numeric CLI options reject non-numeric values", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "rx/0",
    "--log_level",
    "nope",
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /param must be a number/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("numeric JSON option fields reject non-numeric values", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "rx/0",
    "--pool_time",
    JSON.stringify({ stats: "nope" }),
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /pool_time\.stats param must be a number/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("numeric option values reject negatives", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "rx/0",
    "--pool_time",
    JSON.stringify({ stats: -1 }),
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /pool_time\.stats param must be non-negative/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("JSON dev options reject invalid device specs", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "rx/0",
    "--job",
    JSON.stringify({ dev: "cpu^0" }),
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /invalid dev value: cpu\^0/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("JSON dev options accept explicit SYCL GPU platform suffixes", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "cn/gpu",
    "--job",
    JSON.stringify({ dev: "gpu1o*1280" }),
    "--pool_time",
    JSON.stringify({ stats: -1 }),
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.doesNotMatch(result.stderr, /invalid dev value: gpu1o\*1280/);
  assert.match(result.stderr, /pool_time\.stats param must be non-negative/);
});

test("mine pool URI rejects out-of-range ports", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "mine",
    "pool.example:70000",
    "user",
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Wrong pool port: 70000/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("JSON pool options reject invalid ports", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "rx/0",
    "--add.pool",
    JSON.stringify({ url: "pool.example", port: 70000, login: "user" }),
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /invalid pool port/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("JSON algo params reject invalid perf values", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "rx/0",
    "--new.algo_param.rx/0",
    JSON.stringify({ perf: "fast" }),
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /invalid perf value: fast/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("repeat schedules delayed callbacks", async () => {
  let calls = 0;

  await new Promise((resolve) => {
    helper.repeat((next) => {
      calls += 1;
      if (calls === 2) {return resolve();}
      next();
    }, 1);
  });

  assert.equal(calls, 2);
});

test("diff2target handles numeric zero difficulty", () => {
  assert.equal(helper.diff2target(0), "0000000000000000");
  assert.equal(helper.diff2target(0n), "0000000000000000");
  assert.equal(helper.diff2target(-1), "0000000000000000");
});

test("kawpowTarget2diff uses the Eth-style high target word", () => {
  assert.equal(
    helper.kawpowTarget2diff("00000000117edbe19772d0000000000000000000000000000000000000000000"),
    62845243145n
  );
});

test("256-bit targets convert to share work", () => {
  const diffOneTarget = "00000000ffff0000000000000000000000000000000000000000000000000000";
  assert.equal(helper.target256ToWork(diffOneTarget), 4295032833n);
  assert.equal(helper.formatHashCount(583796823439n), "583.80 GH");
  assert.equal(helper.formatHashCount(56546580n), "56.55 MH");
  assert.equal(helper.formatHashCount(12004n), "12.00 KH");
});

test("perf hashrate formatting uses scaled units", () => {
  assert.equal(formatHashrate(999.99), "999.99 H/s");
  assert.equal(formatHashrate(1000), "1.00 KH/s");
  assert.equal(formatHashrate(1000000), "1.00 MH/s");
  assert.equal(formatHashrate(19891722), "19.89 MH/s");
  assert.equal(formatHashrate(1200000000), "1.20 GH/s");
  assert.equal(parseFormattedHashrate("19.89", "MH/s"), 19890000);
});

test("test report duration formatting uses seconds and minutes", () => {
  assert.equal(specReporter.formatDurationMs(999.9, "999.9"), "999.9ms");
  assert.equal(specReporter.formatDurationMs(1000, "1000"), "1.00 s");
  assert.equal(specReporter.formatDurationMs(198896.794728, "198896.794728"), "3.31 min");
  assert.equal(
    specReporter.rewriteReporterDurations("  ✔ kawpow (198896.794728ms)\nℹ duration_ms 198936.440599\n"),
    "  ✔ kawpow (3.31 min)\nℹ duration 3.32 min\n",
  );
});

test("Panthera hash tests wait for every CPU batch result", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.job = { algo: "panthera", dev: "cpu*2" };

  assert.equal(miner.expectedTestThreads({ thread_id: 0 }), 2);
});

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
    algoParams: { kawpow: "gpu1*1" },
  });

  assert.equal(typeof miner.getSetJob(), "function");
  assert.deepEqual(miner.sentMessages, []);
});

test("default algo benchmarking only includes MoneroOcean algos plus rx/2", async () => {
  const miner = await loadMinerWithStubs({
    algoParams: {
      "argon2/chukwa": "cpu",
      "argon2/chukwav2": "cpu",
      "cn-heavy/xhv": "cpu",
      "cn-pico/tlo": "cpu",
      "cn/0": "cpu",
      "etchash": "gpu1*1",
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

test("KawPow benchmark jobs include fixed nonce metadata", async () => {
  const autoBenchmark = await loadMinerWithStubs({
    algoParams: { kawpow: "gpu1*1" },
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
    algoParams: { etchash: "gpu1*1" },
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
      algo_params: { kawpow: { dev: "gpu1*1", perf: null } },
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
        kawpow: { dev: "gpu1*37282560", perf: 20882200 },
        c29: { dev: "gpu1*1", perf: 2.79 },
        etchash: { dev: "gpu1*33554432", perf: 21090000 },
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

test("donation pool mines a MoneroOcean algo while the rig is configured for pearl", async () => {
  // Regression for pearl + donation: the donate pool (xmrig.moneroocean.stream, protocol null) must
  // keep speaking the XMR `login` dialect and advertise a MoneroOcean-supported algo even when the
  // rig's algo context is pearl. Otherwise donation would either send pearl-protocol traffic the MO
  // pool can't serve, or advertise only pearl (which MO cannot assign). MO ignores pearl and assigns
  // the other advertised algo (rx/0) in its login result, which must parse as an rx/0 donation job.
  let donatedJob = null;
  await withMockPool({
    pool: { login: "user", pass: "x", use_subscribe: false }, // MO donate pool opts out of pearl subscribe
    opt: {
      bench_algo_params: 0,
      job: { algo: "pearl" },                            // rig context is pearl; must NOT leak to the donate pool
      algo_params: { pearl: { dev: "gpu1*131072", perf: 1 }, "rx/0": { dev: "cpu", perf: 1 } },
    },
  }, async ({ socket, writes }) => {
    pool.connect_pool_throttle(0, (job) => { donatedJob = job; return job; });
    socket.emit("connect");
    const login = writes[0];
    assert.equal(login.method, "login");                          // XMR login, not a pearl mining.subscribe
    assert.equal(login.params.algo.includes("rx/0"), true);      // a MoneroOcean-minable algo is offered
    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":{"id":"w","job":' +
      '{"blob":"0101","job_id":"1","target":"c6100000","algo":"rx/0","height":1,"seed_hash":"ab"}}}\n'));
    assert.equal(donatedJob.algo, "rx/0");                        // MO assigns rx/0; donation actually mines
  });
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

test("KawPow pool jobs append the nonce field to a header hash", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const headerHash = "00".repeat(32);

  setJob({
    algo: "kawpow",
    blob: headerHash,
    difficulty: 1,
    id: "worker",
    job_id: "job",
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.equal(jobMessage.job.blob_hex, headerHash + "0000000000000000");
  assert.equal(jobMessage.job.noncebytes, 8);
  assert.equal(jobMessage.job.nonceoffset, 32);
  assert.equal(jobMessage.job.worker_id, "worker");
});

test("KawPow stratum jobs can use pool login as worker id", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const headerHash = "00".repeat(32);

  setJob({
    algo: "kawpow",
    blob: headerHash,
    difficulty: 1,
    job_id: "job",
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.equal(jobMessage.job.worker_id, "user");
});

test("KawPow pool jobs preserve provided nonce template", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const headerHash = "00".repeat(32);
  const extraNonce = "00000000000081ff";

  setJob({
    algo: "kawpow",
    blob: headerHash + extraNonce,
    header_hash: headerHash,
    nonce: "ff81000000000000",
    nicehash_mask: "ffff000000000000",
    difficulty: 1,
    id: "worker",
    job_id: "job",
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.equal(jobMessage.job.blob_hex, headerHash + extraNonce);
  assert.equal(jobMessage.job.header_hash, headerHash);
  assert.equal(jobMessage.job.nonce, "ff81000000000000");
  assert.equal(jobMessage.job.nicehash_mask, "ffff000000000000");
});

test("KawPow submit uses the header hash carried by the worker result", async () => {
  const miner = await loadMinerWithStubs();
  const oldHeaderHash = "11".repeat(32);
  const newHeaderHash = "22".repeat(32);
  miner.global.opt.pools[0].submit_mode = "raven";
  miner.global.opt.pools[0].last_job = {
    job_id: "new",
    header_hash: newHeaderHash,
  };

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "old",
      nonce: "ff81000000000001",
      hash: "00".repeat(32),
      mix_hash: "33".repeat(32),
      header_hash: oldHeaderHash,
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "user",
    "old",
    "0xff81000000000001",
    "0x" + oldHeaderHash,
    "0x" + "33".repeat(32),
  ]));
});

test("Etchash submit uses Eth mining.submit format", async () => {
  const miner = await loadMinerWithStubs();
  const headerHash = "22".repeat(32);
  miner.global.opt.pools[0].submit_mode = "eth";

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "203d",
      nonce: "080c000000000001",
      hash: "00".repeat(32),
      mix_hash: "33".repeat(32),
      header_hash: headerHash,
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "user",
    "203d",
    "0x080c000000000001",
    "0x" + headerHash,
    "0x" + "33".repeat(32),
  ]));
});

test("Etchash submit uses ethproxy eth_submitWork format", async () => {
  const miner = await loadMinerWithStubs();
  const headerHash = "22".repeat(32);
  miner.global.opt.pools[0].submit_mode = "ethproxy";

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: headerHash,
      nonce: "080c000000000001",
      hash: "00".repeat(32),
      mix_hash: "33".repeat(32),
      header_hash: headerHash,
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(miner.poolWrites[0].json.method, "eth_submitWork");
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "0x080c000000000001",
    "0x" + headerHash,
    "0x" + "33".repeat(32),
  ]));
});

test("Autolykos2 submit uses Ergo mining.submit format", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].submit_mode = "erg";
  miner.global.opt.pools[0].erg_submit_jobs = {
    "203d": { extra_nonce: "080c", extra_nonce2_size: 6, ntime: "00000002" },
  };

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "203d",
      nonce: "080c000000000001",
      hash: "00".repeat(32),
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "user",
    "203d",
    "000000000001",
    "00000002",
    "080c000000000001",
  ]));
});

test("nicehash xn prefixes longer than noncebytes are truncated", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();

  assert.doesNotThrow(() => setJob({
    algo: "cn/0",
    blob_hex: "abcd",
    noncebytes: 4,
    xn: "001122334455",
    difficulty: 1,
    id: "worker",
    job_id: "job",
  }));

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.equal(jobMessage.job.nonce, "00112233");
  assert.equal(jobMessage.job.nicehash_mask, "ffffffff");
});

test("Equihash125_4 (Flux) pools build the 140-byte header from the ZIP-301 notify", async () => {
  let jobMessage = null;
  const version  = "04000000";
  const prevhash = "a8675c842f7a1342fadd00cd9b4e4909526b1c0ab5a747c5529b4deb13000000";
  const merkle   = "ce7d6ea2452245925fc70c3a08a3c3dd2ca4beab7481f237a19751666bfd25c3";
  const reserved = "0fd282d94b1e1a7f2c57eb3fb9e2853d990753fa137e13c99bd43f220d4fce69";
  const ntime    = "90e44f5d";
  const bits     = "ce28421d";
  const target   = "0000000a42ce000000000000000000000000000000000000000000000000000c";
  await withMockPool({
    pool: { is_keepalive: true, login: "t1fluxwallet.worker", protocol: "equihash" },
    opt: { job: { algo: "equihash125_4" } },
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
    assert.equal(poolConfig.equihash_target, target);
    assert.equal(jobMessage.algo, "equihash125_4");
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

test("Equihash125_4 submit uses ZIP-301 mining.submit [worker, job_id, time, nonce2, solution]", async () => {
  const miner = await loadMinerWithStubs();
  const solution = "34" + "ab".repeat(52); // 0x34 compactSize + 52-byte compressed proof = 106 hex
  miner.global.opt.pools[0].submit_mode = "equihash";
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

test("Kaspa (kheavyhash) submit uses mining.submit [wallet.worker, job_id, 0x+nonce]", async () => {
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
});
