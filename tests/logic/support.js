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

const opts = require("../../opts.js");
const helper = require("../../helper.js");
const pool = require("../../pool.js");
const compilerPolicy = require("../../compiler-policy.js");
const { formatHashrate, parseFormattedHashrate } = require("../common/miner_command");
const specReporter = require("../common/spec_reporter");
const repoRoot = path.join(__dirname, "..", "..");
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
  // Give every VM-loaded miner its own signal emitter. Inheriting from the real process object also
  // inherits its internal EventEmitter state, so repeated tests otherwise leak signal handlers into
  // the test runner and eventually trigger MaxListenersExceededWarning.
  const processStub = new events.EventEmitter();
  Object.assign(processStub, {
    argv: options.argv || ["node", "mom.js", "mine", "pool.example:1", "user"],
    env: { ...process.env, ...(options.env || {}) },
    platform: options.platform || process.platform,
    stderr: process.stderr,
    stdin: process.stdin,
    stdout: process.stdout,
    exit: (code) => { throw new Error(`unexpected exit ${code}`); },
  });
  const detachedSetTimeout = (...args) => {
    const timer = setTimeout(...args);
    if (timer.unref) {timer.unref();}
    return timer;
  };
  const requireStub = (id) => {
    if (id === "./helper.js") {return helperStub;}
    if (id === "./pool.js") {return poolStub;}
    if (id === "./opts.js") {return opts;}
    if (id === "./compiler-policy.js") {return require("../../compiler-policy.js");}
    if (id === "./gpu-tuning.js") {return require("../../gpu-tuning.js");}
    if (id.startsWith("./miner/")) {return require(path.join(repoRoot, id));}
    return require(id);
  };

  vm.runInNewContext(
    `(function(require, module, exports, process, global, console, Buffer, setTimeout, clearTimeout, setInterval, setImmediate) { ${source}\nmodule.exports.__test = { expectedTestThreads, matchesTestResult, messageHandler, publicAlgoParams };\n})`,
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
    matchesTestResult: moduleStub.exports.__test.matchesTestResult,
    messageHandler: moduleStub.exports.__test.messageHandler,
    publicAlgoParams: moduleStub.exports.__test.publicAlgoParams,
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
    pool_ids: {active: 0, primary: 0, donate: null},
    pool_time: {first_job_wait: 0.001, connect_throttle: 60, close_wait: 60, keepalive: 60, ...options.pool_time},
    algo_params: {},
    bench_algo_params: 1,
    default_msrs: {},
    gpu_tune: 0,
    save_config: "",
    ...options.opt,
  };
}

async function withMockPool(options, callback) {
  const originalConnect = net.connect;
  const originalSwitchPool = pool.switch_pool;
  const previousOpt = testGlobal.opt;
  /** @type {MockSocket} */
  const socket = options.socket || /** @type {MockSocket} */ (new events.EventEmitter());
  /** @type {JsonObject[]} */
  const writes = [];
  let switched = false;

  socket.write = options.write || function(message) { writes.push(JSON.parse(message)); };
  socket.destroy = options.destroy || function() { socket.destroyed = true; };
  net.connect = /** @type {typeof net.connect} */ (/** @type {unknown} */ (function() { return socket; }));
  pool.switch_pool = function() { switched = true; };
  testGlobal.opt = mockPoolOptions(options);

  try {
    const mockPool = testGlobal.opt && testGlobal.opt.pools[0];
    if (!mockPool) {throw new Error("mock pool was not created");}
    return await callback({socket, writes, switched: () => switched, poolConfig: mockPool});
  } finally {
    if (testGlobal.opt) {
      for (const poolConfig of testGlobal.opt.pools) {
        if (poolConfig.keepalive !== null) {clearTimeout(poolConfig.keepalive);}
        poolConfig.keepalive = null;
        poolConfig.socket = null;
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
    net.connect = originalConnect;
    pool.switch_pool = originalSwitchPool;
    if (previousOpt) {testGlobal.opt = previousOpt;} else {delete testGlobal.opt;}
  }
}

function completeOneBenchmark(miner, rate = "1") {
  miner.messageHandler({ type: "hashrate", thread_id: 0, value: { hashrate: rate } });
}

module.exports = {
  describe, test, assert, spawnSync, events, fs, net, path, tls, vm,
  opts, helper, pool, compilerPolicy, formatHashrate, parseFormattedHashrate,
  specReporter, repoRoot, noOp, loadMinerWithStubs, mockPoolConfig,
  mockPoolOptions, withMockPool, completeOneBenchmark
};

const testGlobal = /** @type {{opt?: TestOptions}} */ (/** @type {unknown} */ (globalThis));
function unexpectedPoolJob(_job) {
  throw new Error("unexpected pool job");
}
function completeMiningJob(job) {
  if (typeof job.algo !== "string") {
    throw new Error("pool job is missing mining fields");
  }
  const {submit_mode: _submitMode, ...nativeJob} = job;
  return {...nativeJob, algo: job.algo, dev: typeof job.dev === "string" ? job.dev : "cpu"};
}
Object.assign(module.exports, {unexpectedPoolJob, completeMiningJob});
