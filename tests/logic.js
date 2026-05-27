"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const events = require("node:events");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");
const vm = require("node:vm");

const opts = require("../opts.js");
const helper = require("../helper.js");
const pool = require("../pool.js");
const repoRoot = path.join(__dirname, "..");

async function loadMinerWithStubs() {
  const source = fs.readFileSync(path.join(repoRoot, "mo-miner.js"), "utf8");
  const moduleStub = { exports: {} };
  const coreEvents = new events.EventEmitter();
  const sentMessages = [];
  let capturedSetJob = null;
  const helperStub = {
    ...helper,
    cluster_process: () => false,
    create_core: () => ({
      from: coreEvents,
      emit_to: (name) => {
        if (name === "algo_params") setImmediate(() => coreEvents.emit("algo_params", {}));
        if (name === "read_msr") setImmediate(() => coreEvents.emit("error", { message: "skip" }));
      },
    }),
    recreate_threads: () => {},
    messageWorkers: (msg) => sentMessages.push(msg),
    log: () => {},
    log1: () => {},
    log2: () => {},
    log3: () => {},
    log_err: () => {},
  };
  const poolStub = {
    connect_pool_throttle: (pool_id, setJob) => { capturedSetJob = setJob; },
  };
  const processStub = Object.create(process);
  processStub.argv = ["node", "mo-miner.js", "mine", "pool.example:1", "user"];
  processStub.env = { ...process.env };
  processStub.exit = (code) => { throw new Error(`unexpected exit ${code}`); };
  const requireStub = (id) => {
    if (id === "./helper.js") return helperStub;
    if (id === "./pool.js") return poolStub;
    if (id === "./opts.js") return opts;
    return require(id);
  };

  vm.runInNewContext(
    `(function(require, module, exports, process, global, console, Buffer, setTimeout, setInterval, setImmediate) { ${source}\n})`,
    {},
  )(requireStub, moduleStub, moduleStub.exports, processStub, {}, console, Buffer, setTimeout, () => {}, setImmediate);

  for (let i = 0; i < 10 && !capturedSetJob; ++i) {
    await new Promise((resolve) => setImmediate(resolve));
  }
  return { getSetJob: () => capturedSetJob, sentMessages };
}

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
    "mo-miner.js",
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

test("repeat schedules delayed callbacks", async () => {
  let calls = 0;

  await new Promise((resolve) => {
    helper.repeat((next) => {
      calls += 1;
      if (calls === 2) return resolve();
      next();
    }, 1);
  });

  assert.equal(calls, 2);
});

test("stale pool timeout does not destroy a replacement socket", async () => {
  const originalConnect = net.connect;
  const previousOpt = global.opt;
  const staleSocket = new events.EventEmitter();
  const replacementSocket = new events.EventEmitter();
  staleSocket.write = replacementSocket.write = function() {};
  staleSocket.destroy = function() { this.destroyed = true; };
  replacementSocket.destroy = function() { this.destroyed = true; };
  net.connect = function() { return staleSocket; };
  global.opt = {
    log_level: 0,
    pools: [{
      url: "pool.example",
      port: 1,
      is_tls: false,
      is_keepalive: false,
      socket: null,
      keepalive: null,
      last_job: null,
      last_connect_time: 0,
    }],
    pool_ids: { active: 0, primary: 0, donate: null },
    pool_time: { first_job_wait: 0.001, connect_throttle: 60, close_wait: 60, keepalive: 60 },
    algo_params: {},
  };

  try {
    pool.connect_pool_throttle(0, function() {});
    global.opt.pools[0].socket = replacementSocket;
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(replacementSocket.destroyed, undefined);
  } finally {
    net.connect = originalConnect;
    global.opt = previousOpt;
  }
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
