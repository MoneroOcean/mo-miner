"use strict";

const os = require("node:os");
const {EventEmitter} = require("node:events");
const {PassThrough} = require("node:stream");
const s = require("./support");
const {
  test, spawnSync, fs, path, opts, helper, pool, compilerPolicy,
  formatHashrate, parseFormattedHashrate, specReporter, repoRoot,
  loadMinerWithStubs, mockPoolOptions, unexpectedPoolJob,
} = s;
/** @type {typeof import("node:assert/strict")} */
const assert = s.assert;
const {normalizeAlgoName} = require("../../miner/algorithms");
/** @typedef {{pools?: Array<{url?: string, nested?: {enabled: boolean}}>,
 *   algo_params?: {rx?: {dev?: string}}}} CloneDefaults */
const testGlobal = /** @type {{opt: ReturnType<typeof mockPoolOptions>}} */
  (/** @type {unknown} */ (globalThis));
const makeEnvironment = require("../../miner/environment");
/** @param {NodeJS.Platform} platform @param {NodeJS.ProcessEnv} env */
function mockProcess(platform, env) {
  return {platform, env};
}
/** @param {(path: string, encoding: "utf8") => string} readFileSync
 * @param {(path: string) => string[]} readdirSync */
function mockFileSystem(readFileSync, readdirSync) {
  return {existsSync: () => true, readFileSync, readdirSync};
}
/** @param {Array<import("node:os").CpuInfo>} cpus */
function mockOperatingSystem(cpus) {
  return {cpus: () => cpus};
}
const environmentDeps = {
  fs, os, process, opt: opts.create_default_opts(), compilerPolicy,
  gpuTuning: require("../../gpu-tuning"), normalizeAlgoName,
  requestedJobBackend: () => "auto", jobBackend: () => "auto",
  resolvedDeviceList: /** @param {string} _algo @param {string} configured @param {string} raw */
    (_algo, configured, raw) => configured || raw,
  configuredTuning: () => ({}),
};

test("Cuckaroo aliases normalize to the short algorithm names", () => {
  for (const name of ["c29", "cuckaroo", "cuckaroo29", "c29xtm"]) {
    assert.equal(normalizeAlgoName(name), "c29");
  }
  for (const name of ["c30", "cuckaroo30", "c30ctx"]) {
    assert.equal(normalizeAlgoName(name), "c30");
  }
  for (const name of ["kawpow", "kawpow1", "kawpow4"]) {
    assert.equal(normalizeAlgoName(name), "kawpow");
  }
});

test("algorithm normalization preserves empty values and rejects non-strings", () => {
  assert.equal(normalizeAlgoName("CUCKAROO"), "c29");
  assert.equal(normalizeAlgoName(null), null);
  assert.equal(normalizeAlgoName(undefined), undefined);
  assert.equal(normalizeAlgoName(""), "");

  /** @type {unknown[]} */
  const invalid = [0, false, NaN, 1, true, {}, []];
  for (const value of invalid) {
    assert.throws(() => Reflect.apply(normalizeAlgoName, null, [value]), TypeError);
  }
});

test("MoM pool defaults to JSON login only for the canonical endpoint", () => {
  for (const host of [
    "mom.moneroocean.stream",
    "MOM.MONEROOCEAN.STREAM",
    "mom.moneroocean.stream.",
  ]) {
    assert.equal(opts.pool_create(host, 20001, true, "user", "pass").use_subscribe, false, host);
  }
  for (const host of [
    "mom.moneroocean.stream.evil",
    "evil.mom.moneroocean.stream",
    "mom.moneroocean.stream..",
    "other.moneroocean.stream",
  ]) {
    assert.equal(opts.pool_create(host, 20001, true, "user", "pass").use_subscribe, true, host);
  }

  const configured = opts.create_default_opts();
  opts.apply_config(configured, {
    pools: [{
      url: "MOM.MONEROOCEAN.STREAM.", port: 20001, is_tls: true, login: "user",
      use_subscribe: true,
    }],
    pool_ids: {primary: 0, donate: null},
  });
  assert.equal(configured.pools[0]?.use_subscribe, true);

  for (const [url, expected] of [
    ["mom.moneroocean.stream", false],
    ["other.moneroocean.stream", true],
  ]) {
    const defaults = opts.create_default_opts();
    opts.apply_config(defaults, {
      pools: [{url, port: 20001, is_tls: true, login: "user"}],
      pool_ids: {primary: 0, donate: null},
    });
    assert.equal(defaults.pools[0]?.use_subscribe, expected, String(url));
  }
});

test("CLI mine creates JSON-login primaries for fixed MoM algorithms", async () => {
  for (const algo of ["c29", "kawpow", "autolykos2", "etchash"]) {
    const miner = await loadMinerWithStubs({argv: [
      "node", "mom.js", "mine", "mom.moneroocean.stream:20001tls", "user",
      "--job.algo", algo, "--job.dev", "gpu1", "--bench_algo_params", "0",
    ]});
    const primaryId = miner.global.opt.pool_ids.primary;
    assert.equal(primaryId, 0, algo);
    assert.equal(miner.global.opt.job.algo, algo, algo);
    assert.equal(miner.global.opt.pools[primaryId]?.use_subscribe, false, algo);
  }
});

test("CLI pool JSON explicitly overrides the MoM handshake default", async () => {
  const miner = await loadMinerWithStubs({argv: [
    "node", "mom.js", "mine", "mom.moneroocean.stream:20001tls", "user",
    "--add.pool", JSON.stringify({
      url: "mom.moneroocean.stream", port: 20001, is_tls: true, login: "user",
      use_subscribe: true,
    }),
    "--bench_algo_params", "0",
  ]});
  assert.equal(miner.global.opt.pools[0]?.use_subscribe, false);
  assert.equal(miner.global.opt.pools[1]?.use_subscribe, true);
});

test("mine mode closes through the opt-in control pipe", async () => {
  const stdin = new PassThrough();
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "mine", "pool.example:1", "user", "--bench_algo_params", "0"],
    env: {MOM_BENCHMARK_CONTROL_STDIN: "1", MOM_SKIP_MSR: "1"},
    stdin, algoParams: {},
  });
  assert.equal(stdin.listenerCount("data"), 1);
  assert.deepEqual(miner.poolConnects, [0]);

  stdin.write("close\n");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(stdin.destroyed, true);
  assert.equal(miner.process.exitCode, 0);
  assert.deepEqual(miner.workerCloseDeadlines, [null]);
  assert.deepEqual(miner.exitCodes, []);

  miner.coreEvents.emit("close");
  await flushLifecycleCallbacks(2);
  assert.deepEqual(miner.exitCodes, [0]);
});

test("controlled mine waits for the core and live workers before exiting", async () => {
  const stdin = new PassThrough();
  /** @type {EventEmitter & {exitCode: number | null, signalCode: string | null, killed: boolean}} */
  const subprocess = Object.assign(new EventEmitter(), {
    exitCode: null, signalCode: null, killed: true,
  });
  let clusterClosed = false;
  /** @type {EventEmitter & {isDead: () => boolean}} */
  const cluster = Object.assign(new EventEmitter(), {
    isDead: () => clusterClosed,
  });
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "mine", "pool.example:1", "user", "--bench_algo_params", "0"],
    env: {MOM_BENCHMARK_CONTROL_STDIN: "1", MOM_SKIP_MSR: "1"},
    stdin, algoParams: {}, closeWorkerTargets: [
      {type: "subprocess", worker: subprocess}, {type: "cluster", worker: cluster},
    ],
  });
  subprocess.once("close", (code) => {
    if (code !== 0) {miner.process.exitCode = 1;}
  });

  stdin.end("close\n");
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(miner.workerCloseDeadlines, [null]);
  assert.deepEqual(miner.exitCodes, []);

  miner.coreEvents.emit("close");
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(miner.exitCodes, []);

  subprocess.killed = false;
  subprocess.exitCode = 7;
  subprocess.emit("close", 7, null);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(miner.exitCodes, []);

  clusterClosed = true;
  cluster.emit("exit", 0, null);
  await flushLifecycleCallbacks(2);
  assert.deepEqual(miner.exitCodes, [1]);
});

test("controlled mine handles already-closed core and workers", async () => {
  const stdin = new PassThrough();
  /** @type {EventEmitter & {exitCode: number | null, signalCode: string | null, killed: boolean}} */
  const subprocess = Object.assign(new EventEmitter(), {
    exitCode: 0, signalCode: null, killed: false,
  });
  /** @type {EventEmitter & {isDead: () => boolean}} */
  const cluster = Object.assign(new EventEmitter(), {isDead: () => true});
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "mine", "pool.example:1", "user", "--bench_algo_params", "0"],
    env: {MOM_BENCHMARK_CONTROL_STDIN: "1", MOM_SKIP_MSR: "1"},
    stdin, algoParams: {}, closeWorkerTargets: [
      {type: "subprocess", worker: subprocess}, {type: "cluster", worker: cluster},
    ],
  });
  miner.coreEvents.emit("close");

  stdin.end("close\n");
  await flushLifecycleCallbacks(2);
  assert.equal(miner.process.exitCode, 0);
  assert.deepEqual(miner.workerCloseDeadlines, [null]);
  assert.deepEqual(miner.exitCodes, [0]);
});

test("controlled mine preserves runtime error status until natural close", async () => {
  const stdin = new PassThrough();
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "mine", "pool.example:1", "user", "--bench_algo_params", "0"],
    env: {MOM_BENCHMARK_CONTROL_STDIN: "1", MOM_SKIP_MSR: "1"},
    stdin, deferCoreResponses: true,
  });
  miner.messageHandler({type: "error", thread_id: 0,
    value: {message: "runtime error", fatal: true}});
  assert.equal(miner.process.exitCode, 1);
  assert.deepEqual(miner.workerCloseDeadlines, [null]);
  assert.deepEqual(miner.exitCodes, []);

  miner.coreEvents.emit("close");
  await flushLifecycleCallbacks(2);
  assert.deepEqual(miner.exitCodes, [1]);
  stdin.end();
});

test("ordinary mine runtime errors retain the forced worker deadline", async () => {
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "mine", "pool.example:1", "user", "--bench_algo_params", "0"],
    env: {MOM_BENCHMARK_CONTROL_STDIN: undefined, MOM_SKIP_MSR: "1"},
    deferCoreResponses: true,
  });
  const error = {type: "error", thread_id: 0,
    value: {message: "runtime error", fatal: true}};
  miner.messageHandler(error);
  miner.messageHandler(error);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(miner.process.exitCode, 1);
  assert.deepEqual(miner.workerCloseDeadlines, [3000]);
  assert.deepEqual(miner.exitCodes, [1]);
});

test("ordinary mine mode does not install the opt-in control pipe", async () => {
  const stdin = new PassThrough();
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "mine", "pool.example:1", "user", "--bench_algo_params", "0"],
    env: {MOM_BENCHMARK_CONTROL_STDIN: undefined, MOM_SKIP_MSR: "1"},
    stdin, algoParams: {},
  });
  assert.equal(stdin.listenerCount("data"), 0);

  stdin.end("close\n");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(miner.process.exitCode, null);
  assert.deepEqual(miner.workerCloseDeadlines, []);
});

test("CLI rejects inherited object keys as directives", () => {
  const result = spawnSync(process.execPath, ["mom.js", "toString"], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });
  assert.equal(result.status, 1, result.stderr);
  assert.match(result.stderr, /Unknown directive toString/);
});

test("MSR tuning can be disabled for portable deployment tests", () => {
  const environment = makeEnvironment({
    ...environmentDeps, process: mockProcess("linux", {MOM_SKIP_MSR: "1"}),
  });
  assert.equal(environment.use_msr_tuning(), false);

  const normal = makeEnvironment({
    ...environmentDeps, process: mockProcess("linux", {}),
  });
  assert.equal(normal.use_msr_tuning(), true);
});

test("CPU discovery falls back when proc or sysfs is inaccessible", () => {
  const environment = makeEnvironment({
    ...environmentDeps,
    fs: mockFileSystem(
      (file, _encoding) => {
        if (file === "/proc/cpuinfo") {return "processor: 0\nprocessor: 1\n";}
        throw Object.assign(new Error("denied"), {code: "EACCES"});
      },
      () => { throw Object.assign(new Error("denied"), {code: "EACCES"}); },
    ),
    os: mockOperatingSystem([
      {model: "", speed: 0, times: {user: 0, nice: 0, sys: 0, idle: 0, irq: 0}},
      {model: "", speed: 0, times: {user: 0, nice: 0, sys: 0, idle: 0, irq: 0}},
      {model: "", speed: 0, times: {user: 0, nice: 0, sys: 0, idle: 0, irq: 0}},
    ]),
    process: mockProcess("linux", {}),
  });
  assert.deepEqual(environment.detect_cpu(), {
    cpu_sockets: 1, cpu_threads: 3, cpu_l3cache: 0,
  });
});

test("correctness mode blocks real pool sockets", () => {
  const previousOpt = testGlobal.opt;
  const previousGuard = process.env["MOM_TEST_NO_POOL_NETWORK"];
  testGlobal.opt = mockPoolOptions({pool_time: {connect_throttle: 0}});
  process.env["MOM_TEST_NO_POOL_NETWORK"] = "1";
  try {
    assert.throws(
      () => pool.connect_pool_throttle(0, unexpectedPoolJob),
      /Pool network access is disabled during mom correctness tests/
    );
  } finally {
    testGlobal.opt = previousOpt;
    if (previousGuard === undefined) {
      delete process.env["MOM_TEST_NO_POOL_NETWORK"];
    } else {
      process.env["MOM_TEST_NO_POOL_NETWORK"] = previousGuard;
    }
  }
});

test("ROCr signal-pool shutdown warning is hidden without losing worker stderr", () => {
  let filtered = helper.filterWorkerStderr("", "Warning: Resource leak detected by SharedSignalPool, 51");
  assert.equal(filtered.visible, "");
  filtered = helper.filterWorkerStderr(filtered.pending, "9 Signals leaked.\nreal warning\n");
  assert.equal(filtered.pending, "");
  assert.equal(filtered.visible, "real warning\n");

  filtered = helper.filterWorkerStderr("", "partial diagnostic", true);
  assert.equal(filtered.pending, "");
  assert.equal(filtered.visible, "partial diagnostic");

  const longLine = "x".repeat(128 * 1024);
  filtered = helper.filterWorkerStderr("", longLine);
  assert.ok(filtered.pending.length < longLine.length);
  assert.equal(filtered.visible + filtered.pending, longLine);
});

test("known colored AdaptiveCpp advisories are hidden without hiding errors or unfamiliar warnings", () => {
  const bufferWarning = "\u001b[;35m[AdaptiveCpp Warning] \u001b[0mThis application uses SYCL buffers; the SYCL " +
    "buffer-accessor model is well-known to introduce unnecessary overheads. Please consider " +
    "migrating to the SYCL2020 USM model, in particular device USM (sycl::malloc_device) combined " +
    "with in-order queues for more performance. See the AdaptiveCpp performance guide for more information: \n" +
    "https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/performance.md\n";
  const jitWarning = "\u001b[;35m[AdaptiveCpp Warning] \u001b[0mkernel_cache: This application run has " +
    "resulted in new binaries being JIT-compiled. This indicates that the runtime optimization process " +
    "has not yet reached peak performance. You may want to run the application again until this warning " +
    "no longer appears to achieve optimal performance.\n";
  const unfamiliarWarning = "[AdaptiveCpp Warning] kernel_cache: cache directory is read-only.\n";
  const ptxFallback = "'+ptx88' is not a recognized feature for this target (ignoring feature)\n";
  const otherTargetWarning = "'+ptx90' is not a recognized feature for this target (ignoring feature)\n";
  const filtered = helper.filterWorkerStderr("", bufferWarning + jitWarning + ptxFallback +
    unfamiliarWarning + otherTargetWarning + "real error\n");
  assert.equal(filtered.pending, "");
  assert.equal(filtered.visible, unfamiliarWarning + otherTargetWarning + "real error\n");
});

test("Windows HIP loader path chatter is hidden without hiding normal stdout", () => {
  assert.equal(helper.filterWorkerStdoutLine("HIP Library Path: C:\\WINDOWS\\SYSTEM32\\amdhip64_7.dll"), "");
  assert.equal(helper.filterWorkerStdoutLine("HIP Library Path failed"), "HIP Library Path failed");
  assert.equal(helper.filterWorkerStdoutLine("normal output"), "normal output");
});

test("control-core shutdown preserves a failing process status", async () => {
  const miner = await loadMinerWithStubs();
  miner.process.exitCode = 1;
  miner.coreEvents.emit("close");
  assert.equal(miner.process.exitCode, 1);
});

test("cluster workers load only the compute-core boundary", () => {
  const script = `
    const helper = require("./helper.js");
    let calls = 0;
    helper.cluster_process = () => {
      calls += 1;
      return true;
    };
    require("./mom.js");
    if (calls !== 1) { throw new Error("cluster_process calls=" + calls); }
    const forbidden = ["./opts.js", "./pool.js", "./compiler-policy.js", "./miner/cli.js"];
    const loaded = forbidden.filter((id) => require.cache[require.resolve(id)]);
    if (loaded.length) { throw new Error("controller modules loaded: " + loaded.join(",")); }
    process.stdout.write("cluster-boundary-ok");
  `;
  const result = spawnSync(process.execPath, ["-e", script], {
    cwd: repoRoot, encoding: "utf8", timeout: 5000,
  });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout, "cluster-boundary-ok");
});

test("saved config omits job without mutating live options", () => {
  const opt = opts.create_default_opts();
  opt.job.algo = "rx/0";
  opt.job.dev = "cpu";
  const poolConfig = opts.pool_create("pool.example", 443, false, "", "");
  poolConfig.socket = Object.assign(new s.events.EventEmitter(), {
    destroyed: false,
    destroy() { this.destroyed = true; },
    write() { return true; },
  });
  poolConfig.last_job = {job_id: "transient"};
  poolConfig.good_shares = 2;
  opt.pools = [poolConfig];
  opt.pool_ids = {active: 0, primary: 0, donate: null};

  const saved = opts.saved_config(opt);

  const savedPool = saved.pools[0];
  assert.ok(savedPool);
  assert.equal(savedPool["url"], "pool.example");
  assert.equal(savedPool["port"], 443);
  assert.equal(Object.hasOwn(savedPool, "socket"), false);
  assert.equal(Object.hasOwn(savedPool, "last_job"), false);
  assert.equal(Object.hasOwn(savedPool, "good_shares"), false);
  assert.equal(Object.hasOwn(saved, "job"), false);
  assert.equal(Object.hasOwn(saved, "save_config"), false);
  assert.equal(Object.hasOwn(saved.pool_ids, "active"), false);
  assert.equal(opt.job.algo, "rx/0");
  assert.equal(opt.job.dev, "cpu");
  assert.equal(opt.pools[0]?.last_job?.job_id, "transient");
});

test("debug options redact pool credentials without mutating live options", () => {
  const opt = opts.create_default_opts();
  opt.pools = [opts.pool_create("pool.example", 443, false, "wallet", "secret")];
  const redacted = opts.redacted_options(opt);
  assert.equal(redacted.pools[0]?.login, "<redacted>");
  assert.equal(redacted.pools[0]?.pass, "<redacted>");
  assert.equal(opt.pools[0]?.login, "wallet");
  assert.equal(opt.pools[0]?.pass, "secret");
});

test("compute-core debug messages redact nested credentials without changing messages", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-compute-message-"));
  const nativePath = path.join(directory, "fake-native.js");
  fs.writeFileSync(nativePath, [
    "class FakeWorker {",
    "  constructor(onMessage) { this.onMessage = onMessage; module.exports.worker = this; }",
    "  sendToCpp(name, payload) { module.exports.sent = {name, payload}; }",
    "}",
    "module.exports = {AsyncWorker: FakeWorker};",
    "",
  ].join("\n"));
  const previousNativePath = process.env["MOM_NATIVE_PATH"];
  const previousLog3 = helper.log3;
  /** @type {string[]} */
  const logs = [];
  process.env["MOM_NATIVE_PATH"] = nativePath;
  helper.log3 = (/** @type {string} */ message) => logs.push(message);
  try {
    const native = require(nativePath);
    const core = helper.create_core();
    const outgoing = {
      plain_proof: "raw-proof",
      worker_id: "worker-secret",
      login: "login-secret",
      pass: "pass-secret",
      wallet: "wallet-secret",
      nested: [{plain_proof: "nested-proof", visible: "visible-nested"}],
      visible: "visible-outgoing",
    };
    const outgoingSnapshot = structuredClone(outgoing);
    core.emit_to("job", outgoing);
    const sent = native.sent;

    const incoming = {
      plain_proof: "incoming-proof",
      auth: {worker_id: "incoming-worker", login: "incoming-login", pass: "incoming-pass",
        wallet: "incoming-wallet"},
      visible: "visible-incoming",
    };
    const incomingSnapshot = structuredClone(incoming);
    /** @type {unknown} */
    let delivered;
    core.from.once("result", (value) => { delivered = value; });
    native.worker.onMessage("result", incoming);

    core.emit_to("undefined");
    const circular = /** @type {Record<string, unknown>} */ ({visible: "circular-visible"});
    circular["self"] = circular;
    native.worker.onMessage("circular", circular);

    const text = logs.join("\n");
    for (const secret of [
      "raw-proof", "worker-secret", "login-secret", "pass-secret", "wallet-secret",
      "nested-proof", "incoming-proof", "incoming-worker", "incoming-login", "incoming-pass",
      "incoming-wallet",
    ]) {
      assert.equal(text.includes(secret), false, secret);
    }
    assert.match(text, /"visible":"visible-outgoing"/);
    assert.match(text, /"visible":"visible-incoming"/);
    assert.match(text, /message: undefined/);
    assert.match(text, /<unprintable compute message>/);
    assert.deepEqual(outgoing, outgoingSnapshot);
    assert.deepEqual(incoming, incomingSnapshot);
    assert.strictEqual(delivered, incoming);
    assert.equal(sent.payload.plain_proof, "raw-proof");
  } finally {
    if (previousNativePath === undefined) {
      delete process.env["MOM_NATIVE_PATH"];
    } else {
      process.env["MOM_NATIVE_PATH"] = previousNativePath;
    }
    helper.log3 = previousLog3;
    fs.rmSync(directory, {recursive: true, force: true});
  }
});

test("saved credential-bearing config is written with private permissions", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-save-config-"));
  const filename = path.join(directory, "config.json");
  try {
    await loadMinerWithStubs({argv: [
      "node", "mom.js", "mine", "pool.example:443tls", "wallet", "secret",
      "--bench_algo_params", "0", "--save_config", filename,
    ]});
    const saved = JSON.parse(fs.readFileSync(filename, "utf8"));
    assert.equal(saved.pools[1].login, "wallet");
    assert.equal(saved.pools[1].pass, "secret");
    if (process.platform !== "win32") {
      assert.equal(fs.statSync(filename).mode & 0o777, 0o600);
    }
  } finally {
    fs.rmSync(directory, {recursive: true, force: true});
  }
});

test("saved config round-trips through the validated JSON loader", () => {
  const original = opts.create_default_opts();
  assert.equal(opts.parse_opt(original, opts.opt_help, "--job.dev", "cpu", ""), true);
  original.pools = [{
    ...opts.pool_create("pool.example", 443, true, "wallet", "worker"),
    protocol: "conflux",
    use_subscribe: false,
  }];
  original.pool_ids = {active: 0, primary: 0, donate: null};
  original.default_msrs = {"0x1a4": {value: "0xf", mask: "0xffffffffffffffff"}};
  original.algo_params = {
    nexapow: {dev: "gpu1", perf: 123, backend: "auto", tuning: {intensity: 128}},
  };

  const saved = opts.saved_config(original);
  assert.equal(Object.hasOwn(saved, "job"), false);
  assert.equal("active" in saved.pool_ids, false);

  const loaded = opts.create_default_opts();
  opts.apply_config(loaded, JSON.parse(JSON.stringify(saved)));
  assert.equal(loaded.job.dev, "cpu");
  assert.equal(Object.hasOwn(loaded.job, "dev_request"), false);
  assert.deepEqual(loaded.pools[0], {
    url: "pool.example", port: 443, is_tls: true, protocol: "conflux", tls_verify: false,
    is_nicehash: false, is_keepalive: true, use_subscribe: false, worker: "",
    login: "wallet", pass: "worker", pearlhash_target_format: "base",
  });
  assert.deepEqual(loaded.pool_ids, {active: 0, primary: 0, donate: null});
  assert.deepEqual(loaded.default_msrs, original.default_msrs);
  assert.deepEqual(loaded.algo_params, original.algo_params);
});

test("config rejects a pool serving as both primary and donation target", () => {
  const {directory, result} = runConfigSource(JSON.stringify({
    pools: [{url: "pool.example", port: 443, login: "wallet"}],
    pool_ids: {primary: 0, donate: 0},
  }));
  try {
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /primary and pool_ids\.donate must select different pools/);
  } finally {
    fs.rmSync(directory, {recursive: true, force: true});
  }
});

/** @param {string} source */
function runConfigSource(source) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-config-test-"));
  const filename = path.join(directory, "config.json");
  fs.writeFileSync(filename, source);
  const result = spawnSync(process.execPath, ["mom.js", "mine", filename], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });
  return {directory, result};
}

test("config files reject unknown and prototype-mutating fields", () => {
  const valid = {
    pools: [{url: "pool.example", port: 443, login: "wallet"}],
    pool_ids: {primary: 0, donate: null},
  };
  const cases = [
    JSON.stringify({...valid, surprise: true}),
    JSON.stringify({...valid, pool_time: {typo: 1}}),
    JSON.stringify(valid).replace(/}$/, ',"algo_params":{"__proto__":{"dev":"gpu1"}}}'),
  ];
  for (const source of cases) {
    const {directory, result} = runConfigSource(source);
    try {
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, /(Invalid config: .*|Option --.* has unsupported field)/);
      assert.doesNotMatch(result.stderr, /TypeError|Cannot convert undefined or null/);
    } finally {
      fs.rmSync(directory, {recursive: true, force: true});
    }
  }
});

test("JSON config loading never executes JavaScript", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-config-code-test-"));
  const config = path.join(directory, "config.json");
  const marker = path.join(directory, "executed");
  fs.writeFileSync(config,
    `require("node:fs").writeFileSync(${JSON.stringify(marker)}, "bad"); module.exports = {};`);
  try {
    const result = spawnSync(process.execPath, ["mom.js", "mine", config], {
      cwd: repoRoot,
      encoding: "utf8",
      timeout: 5000,
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Can't load config file/);
    assert.equal(fs.existsSync(marker), false);
  } finally {
    fs.rmSync(directory, {recursive: true, force: true});
  }
});

test("config file detection requires a .json extension", () => {
  assert.equal(opts.is_config_file("config.json"), true);
  assert.equal(opts.is_config_file("CONFIG.JSON"), true);
  assert.equal(opts.is_config_file("pooljson"), false);
  assert.equal(opts.is_config_file("config-json"), false);
});

test("default options do not share array or map references", () => {
  const optHelp = {
    pool: {_array: [{url: "a", nested: {enabled: true}}]},
    algo_param: {_map: {rx: {dev: "cpu"}}},
  };
  /** @type {CloneDefaults} */
  const one = {};
  /** @type {CloneDefaults} */
  const two = {};

  opts.set_default_opts(one, optHelp);
  opts.set_default_opts(two, optHelp);
  const onePool = one.pools?.[0];
  const twoPool = two.pools?.[0];
  const oneRx = one.algo_params?.rx;
  const twoRx = two.algo_params?.rx;
  assert.ok(onePool && onePool.nested && twoPool && twoPool.nested && oneRx && twoRx);
  onePool.url = "changed";
  onePool.nested.enabled = false;
  oneRx.dev = "gpu0";

  assert.equal(twoPool.url, "a");
  assert.equal(twoPool.nested.enabled, true);
  assert.equal(twoRx.dev, "cpu");
});

test("internal options reject replaced template collections", () => {
  const mapHelp = {
    value: {_template: {_internal: [0, "internal"]}, _map: {}},
  };
  const arrayHelp = {
    value: {_template: {_internal: [0, "internal"]}, _array: []},
  };

  assert.throws(
    () => opts.set_internal_opts({values: null}, mapHelp),
    /Invalid internal option collection: value/,
  );
  assert.throws(
    () => opts.set_internal_opts({values: {}}, arrayHelp),
    /Invalid internal option collection: value/,
  );
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

test("job JSON rejects non-string hex and text fields", () => {
  const invalidFields = /** @type {Array<[string, unknown]>} */ ([
    ["blob_hex", 1], ["seed_hex", null], ["target", false], ["nicehash_mask", {}],
  ]);
  for (const [field, value] of invalidFields) {
    const result = spawnSync(process.execPath, [
      "mom.js", "bench", "rx/0", "--job", JSON.stringify({[field]: value}),
    ], {
      cwd: repoRoot,
      encoding: "utf8",
      timeout: 5000,
    });
    assert.notEqual(result.status, 0, field);
    assert.match(result.stderr, new RegExp("invalid " + field + " value"), field);
    assert.doesNotMatch(result.stderr, /TypeError|Cannot find module|Compute core/);
  }
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
    JSON.stringify({stats: true}),
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
    JSON.stringify({stats: -1}),
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /pool_time\.stats param must be non-negative/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("numeric option strings require plain decimal syntax", () => {
  const opt = opts.create_default_opts();
  assert.equal(opts.parse_opt(opt, opts.opt_help, "--log_level", "2", ""), true);
  assert.equal(opt.log_level, 2);
  assert.equal(opts.parse_opt(
    opt, opts.opt_help, "--pool_time", JSON.stringify({stats: "12"}), "",
  ), true);
  assert.equal(opt.pool_time.stats, 12);
  assert.equal(opts.parse_opt(
    opt, opts.opt_help, "--pool_time", JSON.stringify({stats: "0.5"}), "",
  ), true);
  assert.equal(opt.pool_time.stats, 0.5);

  for (const value of ["1e2", "0x10", "+1", " 1", "1 "]) {
    const result = spawnSync(process.execPath, [
      "mom.js", "bench", "rx/0", "--log_level", value,
    ], {
      cwd: repoRoot,
      encoding: "utf8",
      timeout: 5000,
    });
    assert.notEqual(result.status, 0, "CLI log_level=" + JSON.stringify(value));
    assert.match(result.stderr, /param must be a number/);
    assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
  }
});

test("pool timers reject hot-loop values and delays beyond the Node timer limit", () => {
  const originalPrintHelp = opts.print_help;
  opts.print_help = (message) => {throw new Error(message);};
  try {
    const positiveTimers = [
      "stats", "primary_reconnect", "first_job_wait",
      "donate_interval", "donate_length", "keepalive",
    ];
    for (const timer of positiveTimers) {
      const current = opts.create_default_opts();
      assert.throws(() => opts.parse_opt(
        current, opts.opt_help, `--pool_time.${timer}`, "0", ""
      ), /param must be positive/);
    }
    const negativeZero = opts.create_default_opts();
    assert.throws(() => opts.parse_opt(
      negativeZero, opts.opt_help, "--pool_time.keepalive", "-0", ""
    ), /param must be positive/);

    const immediate = opts.create_default_opts();
    assert.equal(opts.parse_opt(
      immediate, opts.opt_help, "--pool_time.connect_throttle", "0", ""
    ), true);
    assert.equal(opts.parse_opt(
      immediate, opts.opt_help, "--pool_time.close_wait", "0", ""
    ), true);
    assert.equal(immediate.pool_time.connect_throttle, 0);
    assert.equal(immediate.pool_time.close_wait, 0);

    const bounded = opts.create_default_opts();
    assert.equal(opts.parse_opt(
      bounded, opts.opt_help, "--pool_time.stats", "2147483.647", ""
    ), true);
    assert.equal(bounded.pool_time.stats, 2147483.647);
    assert.throws(() => opts.parse_opt(
      bounded, opts.opt_help, "--pool_time.stats", "2147483.648", ""
    ), /maximum timer delay/);
    assert.throws(() => opts.parse_opt(
      bounded, opts.opt_help, "--pool_time", JSON.stringify({donate_length: 0}), ""
    ), /pool_time\.donate_length param must be positive/);
  } finally {
    opts.print_help = originalPrintHelp;
  }
});

/** @returns {Parameters<typeof opts.saved_config>[0]} */
function optionsWithValidPool() {
  const opt = opts.create_default_opts();
  opt.pools = [opts.pool_create("pool.example", 443, true, "wallet", "worker")];
  opt.pool_ids = {active: 0, primary: 0, donate: null};
  return opt;
}

test("bounded numeric options accept every documented CLI and saved-config value", () => {
  /** @type {Array<["log_level" | "bench_algo_params" | "gpu_tune", number[]]>} */
  const documented = [
    ["log_level", [0, 1, 2, 3]],
    ["bench_algo_params", [0, 1, 2]],
    ["gpu_tune", [0, 1]],
  ];
  for (const [name, values] of documented) {
    for (const value of values) {
      const cli = opts.create_default_opts();
      assert.equal(opts.parse_opt(cli, opts.opt_help, "--" + name, String(value), ""), true);
      assert.equal(cli[name], value, "CLI " + name + "=" + value);

      const original = optionsWithValidPool();
      original[name] = value;
      const loaded = optionsWithValidPool();
      opts.apply_config(loaded, opts.saved_config(original));
      assert.equal(loaded[name], value, "saved config " + name + "=" + value);
    }
  }
});

test("bounded numeric options reject invalid CLI and saved-config values", () => {
  const invalid = /** @type {Record<string, Array<[string | number, RegExp]>>} */ ({
    log_level: [
      [-1, /param must be non-negative/],
      [1.5, /param must be a safe integer/],
      [Number.MAX_SAFE_INTEGER + 1, /param must be a safe integer/],
      ["NaN", /param must be a number/],
      ["Infinity", /param must be a number/],
      [4, /param must be at most 3/],
    ],
    bench_algo_params: [
      [-1, /param must be non-negative/],
      [1.5, /param must be a safe integer/],
      [Number.MAX_SAFE_INTEGER + 1, /param must be a safe integer/],
      ["NaN", /param must be a number/],
      ["Infinity", /param must be a number/],
      [3, /param must be at most 2/],
    ],
    gpu_tune: [
      [-1, /param must be non-negative/],
      [0.5, /param must be a safe integer/],
      [Number.MAX_SAFE_INTEGER + 1, /param must be a safe integer/],
      ["NaN", /param must be a number/],
      ["Infinity", /param must be a number/],
      [2, /param must be at most 1/],
    ],
  });
  const configBase = {
    pools: [{url: "pool.example", port: 443, login: "wallet"}],
    pool_ids: {primary: 0, donate: null},
  };

  for (const [name, values] of Object.entries(invalid)) {
    for (const [value, message] of values) {
      const cli = spawnSync(process.execPath, [
        "mom.js", "bench", "rx/0", "--" + name, String(value),
      ], {
        cwd: repoRoot,
        encoding: "utf8",
        timeout: 5000,
      });
      assert.notEqual(cli.status, 0, "CLI " + name + "=" + value);
      assert.match(cli.stderr, message, "CLI " + name + "=" + value);
      assert.doesNotMatch(cli.stderr, /Cannot find module|Compute core/);

      const {directory, result} = runConfigSource(JSON.stringify({...configBase, [name]: value}));
      try {
        assert.notEqual(result.status, 0, "saved config " + name + "=" + value);
        assert.match(result.stderr, message, "saved config " + name + "=" + value);
        assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
      } finally {
        fs.rmSync(directory, {recursive: true, force: true});
      }
    }
  }
});

test("PearlHash JSON options and config accept only certificate V3", () => {
  const originalPrintHelp = opts.print_help;
  opts.print_help = (message) => {throw new Error(message);};
  try {
    const defaults = opts.create_default_opts();
    assert.equal(opts.parse_opt(
      defaults, opts.opt_help, "--job", JSON.stringify({}), "",
    ), true);
    assert.equal(defaults.job.pearlhash_cert_version, undefined);

    const configured = optionsWithValidPool();
    opts.apply_config(configured, {job: {}});
    assert.equal(configured.job.pearlhash_cert_version, undefined);

    for (const version of [0, 1, 2, 4]) {
      assert.throws(() => opts.parse_opt(
        opts.create_default_opts(), opts.opt_help, "--job",
        JSON.stringify({pearlhash_cert_version: version}), "",
      ), /only supports PearlHash certificate version 3/);
      assert.throws(() => opts.apply_config(optionsWithValidPool(), {
        job: {pearlhash_cert_version: version},
      }), /only supports PearlHash certificate version 3/);
    }
  } finally {
    opts.print_help = originalPrintHelp;
  }
});

test("JSON dev options reject invalid device specs", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "rx/0",
    "--job",
    JSON.stringify({dev: "cpu^0"}),
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /invalid dev value: cpu\^0/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("JSON dev options reject non-numeric GPU suffixes", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "cn/gpu",
    "--job",
    JSON.stringify({dev: "gpu1x*1280"}),
    "--pool_time",
    JSON.stringify({stats: -1}),
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /invalid dev value: gpu1x\*1280/);
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

test("mine pool URI accepts bracketed IPv6 and rejects malformed endpoints", async () => {
  const miner = await loadMinerWithStubs({argv: [
    "node", "mom.js", "mine", "[2001:db8::1]:3333tls", "wallet", "--bench_algo_params", "0",
  ]});
  const primary = miner.global.opt.pool_ids.primary;
  assert.ok(primary !== null && primary !== undefined);
  const pool = miner.global.opt.pools[primary];
  assert.ok(pool);
  assert.equal(pool.url, "2001:db8::1");
  assert.equal(pool.port, 3333);
  assert.equal(pool.is_tls, true);

  for (const uri of ["2001:db8::1:3333", "[2001:db8::1:3333"]) {
    const result = spawnSync(process.execPath, [
      "mom.js", "mine", uri, "wallet", "--bench_algo_params", "0",
    ], {cwd: repoRoot, encoding: "utf8", timeout: 5000});
    assert.notEqual(result.status, 0, uri);
    assert.match(result.stderr, /Wrong pool URI/);
  }
});

test("JSON pool options reject an unknown PearlHash target format", () => {
  const result = spawnSync(process.execPath, [
    "mom.js", "bench", "rx/0", "--add.pool",
    JSON.stringify({url: "pool.example", port: 1234, login: "user", pearlhash_target_format: "guess"}),
  ], {cwd: repoRoot, encoding: "utf8", timeout: 5000});
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /pearlhash_target_format must be base or jackpot/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("JSON pool options retain both supported PearlHash target formats", () => {
  for (const format of ["base", "jackpot"]) {
    const opt = opts.create_default_opts();
    assert.equal(opts.parse_opt(opt, opts.opt_help, "--add.pool", JSON.stringify({
      url: "pool.example", port: 1234, login: "user", pearlhash_target_format: format,
    }), ""), true);
    assert.equal(opt.pools.at(-1)?.pearlhash_target_format, format);
  }
});

test("JSON pool options reject invalid ports", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "rx/0",
    "--add.pool",
    JSON.stringify({url: "pool.example", port: true, login: "user"}),
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
    JSON.stringify({perf: true}),
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /invalid perf value: true/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("JSON algo params reject unknown GPU backends", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "pearlhash",
    "--new.algo_param.pearlhash",
    JSON.stringify({dev: "gpu1", backend: "unknown"}),
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /invalid backend value: unknown/);
  assert.doesNotMatch(result.stderr, /Cannot find module|Compute core/);
});

test("CLI algo-param aliases use canonical map keys and tuning", () => {
  /** @type {Array<[string, string, Record<string, number>]>} */
  const cases = [
    ["c29", "c29", {}],
    ["cuckaroo", "c29", {}],
    ["CUCKAROO", "c29", {}],
    ["kawpow", "kawpow", {workgroup: 128}],
    ["kawpow1", "kawpow", {workgroup: 128}],
    ["KAWPOW", "kawpow", {workgroup: 128}],
  ];
  for (const [name, canonical, tuning] of cases) {
    const opt = opts.create_default_opts();
    assert.equal(opts.parse_opt(
      opt,
      opts.opt_help,
      "--new.algo_param." + name,
      JSON.stringify({dev: "gpu1", perf: 12.5, backend: "auto", tuning}),
      "",
    ), true);
    assert.deepEqual(Object.keys(opt.algo_params), [canonical]);
    assert.deepEqual(opt.algo_params[canonical], {
      dev: "gpu1", perf: 12.5, backend: "auto", tuning,
    });
  }

  const collision = opts.create_default_opts();
  for (const [name, perf] of [["cuckaroo", 1], ["c29", 2]]) {
    assert.equal(opts.parse_opt(
      collision,
      opts.opt_help,
      "--new.algo_param." + name,
      JSON.stringify({dev: "gpu1", perf, backend: "auto", tuning: {}}),
      "",
    ), true);
  }
  assert.deepEqual(Object.keys(collision.algo_params), ["c29"]);
  const collisionParam = collision.algo_params["c29"];
  assert.ok(collisionParam);
  assert.equal(collisionParam.perf, 2);
});

test("config algo-param aliases save and round-trip canonical keys", () => {
  /** @param {number} perf @param {Record<string, number>} [tuning] */
  function param(perf, tuning = {}) {
    return {dev: "gpu1", perf, backend: "auto", tuning};
  }
  const original = optionsWithValidPool();
  opts.apply_config(original, {
    algo_params: {
      c29: param(1),
      cuckaroo: param(2),
      CUCKAROO: param(3),
      kawpow: param(4),
      kawpow1: param(5, {workgroup: 128}),
      KAWPOW: param(6, {workgroup: 128}),
    },
  });
  assert.deepEqual(Object.keys(original.algo_params), ["c29", "kawpow"]);
  const originalC29 = original.algo_params["c29"];
  const originalKawpow = original.algo_params["kawpow"];
  assert.ok(originalC29 && originalKawpow);
  assert.equal(originalC29.perf, 3);
  assert.deepEqual(originalKawpow.tuning, {workgroup: 128});
  assert.equal(originalKawpow.perf, 6);

  const saved = opts.saved_config(original);
  assert.deepEqual(Object.keys(saved.algo_params), ["c29", "kawpow"]);
  const savedC29 = saved.algo_params["c29"];
  const savedKawpow = saved.algo_params["kawpow"];
  assert.ok(savedC29 && savedKawpow);
  assert.equal(savedC29.perf, 3);
  assert.equal(savedKawpow.perf, 6);
  const loaded = optionsWithValidPool();
  opts.apply_config(loaded, JSON.parse(JSON.stringify(saved)));
  assert.deepEqual(loaded.algo_params, original.algo_params);
});

test("algo-param normalization leaves MSR maps and ordinary options unchanged", () => {
  const opt = opts.create_default_opts();
  assert.equal(opts.parse_opt(
    opt,
    opts.opt_help,
    "--new.default_msr.0x1A4",
    JSON.stringify({value: "0xf"}),
    "",
  ), true);
  assert.deepEqual(Object.keys(opt.default_msrs), ["0x1A4"]);
  assert.deepEqual(opt.default_msrs["0x1A4"], {
    value: "0xf", mask: "0xFFFFFFFFFFFFFFFF",
  });
  assert.equal(opts.parse_opt(opt, opts.opt_help, "--job.algo", "CUCKAROO", ""), true);
  assert.equal(opt.job.algo, "CUCKAROO");
});

test("algo-param aliases reject empty and normalized reserved map keys", () => {
  const originalPrintHelp = opts.print_help;
  opts.print_help = (message) => {throw new Error(message);};
  try {
    assert.throws(() => opts.parse_opt(
      opts.create_default_opts(),
      opts.opt_help,
      "--new.algo_param.",
      JSON.stringify({dev: "gpu1"}),
      "",
    ), /invalid algorithm name/);
    assert.throws(() => opts.parse_opt(
      opts.create_default_opts(),
      opts.opt_help,
      "--new.algo_param.__PROTO__",
      JSON.stringify({dev: "gpu1"}),
      "",
    ), /Invalid option map key: __proto__/);
  } finally {
    opts.print_help = originalPrintHelp;
  }
});

test("PearlHash CLI algo params preserve named matrix tuning controls", () => {
  const opt = opts.create_default_opts();
  assert.equal(opts.parse_opt(
    opt,
    opts.opt_help,
    "--new.algo_param.pearlhash",
    JSON.stringify({
      dev: "gpu1*8192",
      backend: "native",
      tuning: {m: 8192, n: 32768, k: 2048, rank: 128},
    }),
    "",
  ), true);
  assert.ok(opt.algo_params["pearlhash"]);
  assert.deepEqual(opt.algo_params["pearlhash"], {
    dev: "gpu1*8192",
    perf: null,
    backend: "native",
    tuning: {m: 8192, n: 32768, k: 2048, rank: 128},
  });
});

test("PearlHash tuning rejects removed top-level shape fields", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "algorithms",
    "--new.algo_param.pearlhash",
    JSON.stringify({dev: "gpu1*8192", m: 8192}),
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /unsupported field: m/);
});

test("algorithms CLI reports mocked parameters and exits successfully", async () => {
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "algorithms"],
    algoParams: {"rx/0": "cpu"},
  });
  await flushLifecycleCallbacks();
  assert.equal(miner.process.exitCode, 0);
  assert.equal(miner.createdCoreCount(), 1);
  assert.equal(miner.writtenStdout, 'MOM_ALGORITHMS {"rx/0":"cpu"}\n');
  assert.equal(miner.writtenStdout.includes("MOM_ALGO_PARAMS"), false);
});

test("removed algo_params CLI directive is rejected", () => {
  const result = spawnSync(process.execPath, ["mom.js", "algo_params"], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 5000,
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stdout, /algorithms/);
  assert.match(result.stderr, /Unknown directive algo_params/);
});

test("algo_params reports requested and resolved GPU backends without changing CPU specs", async () => {
  const miner = await loadMinerWithStubs({env: {MOM_GPU_BACKEND: "amd"}});
  const selection = compilerPolicy.selection("pearlhash", "amd");
  assert.ok(selection);
  const profile = selection.pearlhashProfile;
  assert.ok(profile);
  miner.global.opt.algo_params = {
    autolykos2: {dev: "gpu1*[intensity=8]", perf: null, backend: "auto", tuning: {}},
    pearlhash: {dev: "gpu1*8192", perf: null, backend: "sycl", tuning: {}},
    "rx/0": {dev: "cpu*8", perf: null, backend: "auto"},
  };
  assert.deepEqual(
    JSON.parse(JSON.stringify(miner.publicAlgoParams({
      autolykos2: "gpu1*[intensity=8]",
      pearlhash: "gpu1*[m=8192]",
      "rx/0": "cpu*8",
    }))),
    {
      autolykos2: "gpu1*[intensity=8]:auto[sycl-native]",
      pearlhash: "gpu1*[m=8192]:sycl",
      "rx/0": "cpu*8",
    },
  );
});

test("discovered GPU backend hints reject unknown backends", () => {
  const environment = makeEnvironment({
    ...environmentDeps,
    opt: opts.create_default_opts(),
  });
  assert.throws(() => environment.add_algo_params({
    "cn/gpu": "gpu1*[intensity=1536]",
    "@backend:cn/gpu": "invalid-backend",
  }), /Invalid GPU backend/);
});

test("direct source launch keeps per-algorithm compiler switching enabled", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-direct-policy-"));
  const dpcpp = path.join(root, "dpcpp", "mom.node");
  const adaptiveCpp = path.join(root, "acpp-cuda", "mom.node");
  fs.mkdirSync(path.dirname(dpcpp));
  fs.mkdirSync(path.dirname(adaptiveCpp));
  fs.writeFileSync(dpcpp, "test");
  fs.writeFileSync(adaptiveCpp, "test");

  try {
    const miner = await loadMinerWithStubs({env: {
      MOM_GPU_BACKEND: "nvidia",
      MOM_NVIDIA_COMPUTE_CAPABILITY: "12.0",
      MOM_NATIVE_DIR: root,
      MOM_NATIVE_PATH: "",
      MOM_NATIVE_PATH_LAUNCHER_DEFAULT: "",
    }});
    assert.equal(miner.process.env["MOM_NATIVE_PATH"], dpcpp);
    assert.equal(miner.process.env["MOM_NATIVE_PATH_LAUNCHER_DEFAULT"], dpcpp);
    assert.equal(compilerPolicy.workerEnv(
      "autolykos2", miner.process.env, "linux"
    )["MOM_NATIVE_PATH"], adaptiveCpp);
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test("packaged launch preserves the launcher's atomic control runtime", async () => {
  const launcherAddon = path.join(repoRoot, "libs", "acpp-cuda", "mom.node");
  const launcherLibraries = path.join(repoRoot, "libs", "acpp-cuda");
  const miner = await loadMinerWithStubs({env: {
    MOM_GPU_BACKEND: "nvidia",
    MOM_NVIDIA_COMPUTE_CAPABILITY: "12.0",
    MOM_NATIVE_PATH: launcherAddon,
    MOM_NATIVE_PATH_LAUNCHER_DEFAULT: launcherAddon,
    LD_LIBRARY_PATH: launcherLibraries,
  }});

  assert.equal(miner.process.env["MOM_NATIVE_PATH"], launcherAddon);
  assert.equal(miner.process.env["LD_LIBRARY_PATH"], launcherLibraries);
});

test("GPU tuning precedence is entry, named object, then automatic heuristic", async () => {
  const miner = await loadMinerWithStubs({env: {
    MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "12.0",
  }});
  miner.global.opt.algo_params = {
    kawpow: {
      dev: "gpu1*[workgroup=128]",
      perf: null,
      backend: "auto",
      tuning: {intensity: 2000, workgroup: 256, dag_workgroup: 64},
    },
  };
  assert.deepEqual(
    JSON.parse(JSON.stringify(miner.publicAlgoParams({
      kawpow: "gpu1*[intensity=1000;workgroup=64;dag_workgroup=32]",
    }))),
    {
      kawpow:
        "gpu1*[intensity=2000;workgroup=128;dag_workgroup=64]:auto[sycl-native]",
    },
  );
});

test("explicit GPU process counts are not multiplied by duplicate heuristic matches", async () => {
  const miner = await loadMinerWithStubs({env: {
    MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "12.0",
  }});
  miner.global.opt.algo_params = {
    kawpow: {
      dev: "gpu1^2",
      perf: null,
      backend: "auto",
      tuning: {},
    },
  };
  assert.deepEqual(
    JSON.parse(JSON.stringify(miner.publicAlgoParams({kawpow: "gpu1*8,gpu1*16"}))),
    {kawpow: "gpu1*[intensity=24]^2:auto[sycl-native]"},
  );
});

test("Beam layout-only tuning keeps its layout-specific workgroup automatic", async () => {
  const miner = await loadMinerWithStubs({env: {
    MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "12.0",
  }});
  miner.global.opt.algo_params = {
    beamhash3: {
      dev: "gpu1*[layout=full]",
      perf: null,
      backend: "auto",
      tuning: {},
    },
  };
  assert.deepEqual(
    JSON.parse(JSON.stringify(miner.publicAlgoParams({
      beamhash3: "gpu1*[workgroup=256]",
    }))),
    {beamhash3: "gpu1*[layout=full]:auto[sycl-native]"},
  );
});

test("direct GPU benchmark fills omitted primary tuning from device heuristics", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "bench", "kawpow",
      "--job.dev", "gpu1*[workgroup=128]",
    ],
    algoParams: {kawpow: "gpu1*[intensity=4096;workgroup=256]"},
    waitForMessageType: "bench",
  });
  const message = miner.sentMessages.find((item) => item.type === "bench");
  assert.ok(message);
  assert.ok(message.job);
  assert.equal(message.job.dev, "gpu1*[intensity=4096;workgroup=128]");
  assert.equal(Object.hasOwn(message.job, "dev_request"), false);
});

for (const {directive, messageType} of [
  {directive: "bench", messageType: "bench"},
  {directive: "test", messageType: "test"},
]) {
  test(`direct ${directive} consumes automatic GPU backend hints`, async () => {
    const canonicalVector = require("../vectors/cpu").find(({name}) =>
      name === "cn/gpu odd batch gpu1*[intensity=1]");
    assert.ok(canonicalVector);
    const canonicalExpected = canonicalVector.expected;
    const cases = [
      {
        name: "automatic backend with omitted intensity",
        dev: "gpu1",
        backend: null,
        controlCores: 1,
        requestedBackend: "auto",
        resolvedBackend: "sycl-l0",
        discoveredBackend: "sycl-l0",
      },
      {
        name: "automatic backend with explicit intensity",
        dev: "gpu1*[intensity=1536]",
        backend: null,
        controlCores: 1,
        requestedBackend: "auto",
        resolvedBackend: "sycl-l0",
        discoveredBackend: "sycl-l0",
      },
      {
        name: "explicit backend with omitted intensity",
        dev: "gpu1",
        backend: "sycl-opencl",
        controlCores: 1,
        requestedBackend: "sycl-opencl",
        resolvedBackend: "sycl-opencl",
        discoveredBackend: undefined,
      },
      {
        name: "explicit backend with explicit intensity",
        dev: "gpu1*[intensity=1536]",
        backend: "sycl-opencl",
        controlCores: 0,
        requestedBackend: "sycl-opencl",
        resolvedBackend: "sycl-opencl",
        discoveredBackend: undefined,
      },
    ];

    for (const scenario of cases) {
      const argv = ["node", "mom.js", directive, "cn/gpu"];
      if (directive === "test") {argv.push(canonicalExpected);}
      argv.push("--job.dev", scenario.dev);
      if (scenario.backend) {argv.push("--job.backend", scenario.backend);}
      const miner = await loadMinerWithStubs({
        argv,
        env: {MOM_GPU_BACKEND: "intel", MOM_SKIP_MSR: "1"},
        algoParams: {
          "cn/gpu": "gpu1*[intensity=1536]",
          "@backend:cn/gpu": "sycl-l0",
        },
        waitForMessageType: messageType,
      });
      const message = miner.sentMessages.find((item) => item.type === messageType);
      assert.ok(message, `${directive} ${scenario.name} did not dispatch`);
      assert.ok(message.job);
      // loadMinerWithStubs emits one mocked algo_params response per control core.
      assert.equal(miner.createdCoreCount(), scenario.controlCores,
        `${directive} ${scenario.name} control discovery count`);
      assert.equal(message.job.dev, "gpu1*[intensity=1536]",
        `${directive} ${scenario.name} resolved device`);
      assert.equal(message.job["backend_request"], scenario.requestedBackend,
        `${directive} ${scenario.name} requested backend`);
      assert.equal(message.job["backend"], scenario.resolvedBackend,
        `${directive} ${scenario.name} resolved backend`);
      assert.equal(miner.global.opt.algo_params["cn/gpu"]?.backend, scenario.discoveredBackend,
        `${directive} ${scenario.name} discovered backend`);
    }
  });
}

for (const {name, job, msr, error} of [
  {name: "rejects a malformed blob before setup", job: {blob_hex: "zz"}, msr: "success",
    error: "Invalid rx/0 job blob"},
  {name: "rejects a malformed nonce before setup", job: {nonce: "zz"}, msr: "success",
    error: "Invalid rx/0 nonce"},
  {name: "dispatches once with MSRs skipped", job: {}, msr: "skip", error: null},
  {name: "dispatches its validated snapshot after MSR success", job: {}, msr: "success", error: null},
  {name: "dispatches its validated snapshot after MSR failure", job: {}, msr: "failure", error: null},
]) {
  test(`direct benchmark ${name}`, () => {
    const blob = "00".repeat(43);
    const savedMsrs = {"msr:0x1a4": "0xf,0xffffffffffffffff"};
    const script = `
      const {EventEmitter} = require("node:events");
      Object.defineProperty(process, "platform", {value: "linux"});
      process.env.MOM_SKIP_MSR = ${JSON.stringify(msr === "skip" ? "1" : "0")};
      delete process.env.MOM_BENCHMARK_CONTROL_STDIN;
      const helper = require("./helper.js");
      const coreEvents = new EventEmitter();
      const observed = {workers: 0, cores: 0, reads: 0, writes: [], jobs: [], error: null};
      helper.cluster_process = () => false;
      helper.log = () => {};
      helper.recreate_threads = () => { observed.workers++; };
      helper.closeWorkers = () => [];
      helper.exit_now = () => {};
      helper.create_core = () => {
        observed.cores++;
        return {from: coreEvents, emit_to(name, value) {
          if (name === "write_msr") { observed.writes.push(value); }
          if (name !== "read_msr") { return; }
          observed.reads++;
          // Async setup must dispatch the job already validated before setup began.
          if (${JSON.stringify(error === null)}) {
            global.opt.job.blob_hex = "zz";
            global.opt.job.nonce = "zz";
          }
          setImmediate(() => coreEvents.emit(
            ${JSON.stringify(msr === "success" ? "read_msr" : "error")},
            ${JSON.stringify(msr === "success" ? savedMsrs : {message: "MSRs unavailable"})}
          ));
        }};
      };
      helper.messageWorkers = ({type, job}) => {
        observed.jobs.push([type, job.blob_hex, job.noncebytes, job.nonceoffset]);
        setImmediate(() => process.emit("SIGTERM"));
        return 1;
      };
      process.once("uncaughtException", (error) => { observed.error = error.message; });
      process.once("beforeExit", () => process.stdout.write(JSON.stringify(observed)));
      process.argv = ["node", "mom.js", "bench", "rx/0", "--job",
        ${JSON.stringify(JSON.stringify({blob_hex: blob, ...job}))}];
      try { require("./mom.js"); } catch (error) { observed.error = error.message; }
    `;
    const result = spawnSync(process.execPath, ["-e", script], {
      cwd: repoRoot, encoding: "utf8", timeout: 5000,
    });
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(JSON.parse(result.stdout), {
      workers: error ? 0 : 1,
      cores: error || msr === "skip" ? 0 : 1,
      reads: error || msr === "skip" ? 0 : 1,
      writes: !error && msr === "success" ? [{...savedMsrs, algo: "rx/0"}, savedMsrs] : [],
      jobs: error ? [] : [["bench", blob, 4, 39]],
      error,
    });
  });
}

test("direct CPU test jobs preserve an optional nonce offset", async () => {
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "test", "rx/0", "expected", "--job.dev", "cpu"],
    waitForMessageType: "test",
  });
  const message = miner.sentMessages.find((item) => item.type === "test");
  assert.ok(message);
  assert.ok(message.job);
  assert.equal(miner.createdCoreCount(), 0);
  assert.equal(message.job.noncebytes, 4);
  assert.equal(message.job.nonceoffset, undefined);
  assert.equal(Object.hasOwn(message.job, "dev_request"), false);
});

test("native worker commands reject PearlHash certificate versions other than V3", () => {
  const workerScript = `
    process.env["MOM_CLUSTER_WORKER"] = "1";
    process.env["thread_id"] = "0";
    const {EventEmitter} = require("node:events");
    const helper = require("./helper.js");
    const from = new EventEmitter();
    helper.create_core = () => ({
      from,
      emit_to(type) {
        if (type === "close") {setImmediate(() => from.emit("close"));}
      },
    });
    helper.exit_now = (code) => process.exit(code);
    if (!helper.cluster_process()) {process.exit(2);}
  `;
  const baseJob = {algo: "pearlhash", dev: "cpu", blob_hex: "00"};
  for (const version of [undefined, null, 0, 1, 2, 4]) {
    const job = version === undefined ? baseJob : {...baseJob, pearlhash_cert_version: version};
    const result = spawnSync(process.execPath, ["-e", workerScript], {
      cwd: repoRoot,
      input: JSON.stringify({type: "bench", job}) + "\n" + JSON.stringify({type: "close"}) + "\n",
      encoding: "utf8",
      timeout: 5000,
    });
    assert.equal(result.status, version === undefined ? 0 : 1, String(version));
  }
});

test("native worker commands validate complete PearlHash shape and base target", () => {
  /** @type {NativeJob} */
  const baseJob = {
    algo: "pearlhash", dev: "cpu", blob_hex: "00", intensity: 128,
    pearlhash_n: 128, pearlhash_k: 2048, pearlhash_rank: 128,
  };
  /** @param {NativeJob} job @param {boolean} accepted */
  const check = (job, accepted) => {
    if (accepted) {
      assert.doesNotThrow(() => helper.messageWorkers({type: "bench", job}));
    } else {
      assert.throws(() => helper.messageWorkers({type: "bench", job}), /Invalid worker command/);
    }
  };

  check(baseJob, true);
  check({...baseJob, pearlhash_base_target: "0x" + "a".repeat(64)}, true);
  /** @type {("pearlhash_n" | "pearlhash_k" | "pearlhash_rank")[]} */
  const pearlShapeFields = ["pearlhash_n", "pearlhash_k", "pearlhash_rank"];
  for (const field of pearlShapeFields) {
    /** @type {NativeJob} */
    const partial = {...baseJob};
    delete partial[field];
    check(partial, false);
  }
  /** @type {NativeJob} */
  const targetOnly = {...baseJob, pearlhash_base_target: "a"};
  delete targetOnly.pearlhash_n;
  delete targetOnly.pearlhash_k;
  delete targetOnly.pearlhash_rank;
  check(targetOnly, false);
  for (const baseTarget of ["", "0x", "g", "0x" + "a".repeat(65)]) {
    check({...baseJob, pearlhash_base_target: baseTarget}, false);
  }

  const maxDimension = 1 << 24;
  const maxM = Math.floor(0x7fffffff / (baseJob.pearlhash_k || 1) / 32) * 32;
  const maxN = Math.floor(0x7fffffff / (baseJob.pearlhash_rank || 1) / 32) * 32;
  check({...baseJob, intensity: maxM}, true);
  check({...baseJob, intensity: maxM + 32}, false);
  check({...baseJob, intensity: maxDimension}, false);
  check({...baseJob, intensity: maxDimension + 32}, false);
  check({...baseJob, intensity: 160}, true);
  check({...baseJob, intensity: 161}, false);
  check({...baseJob, pearlhash_n: maxN}, true);
  check({...baseJob, pearlhash_n: maxN + 32}, false);
  check({...baseJob, pearlhash_n: maxDimension}, false);
  check({...baseJob, pearlhash_n: maxDimension + 32}, false);
  check({...baseJob, pearlhash_n: 160}, true);
  check({...baseJob, pearlhash_n: 161}, false);
  check({...baseJob, pearlhash_k: 960}, false);
  check({...baseJob, pearlhash_k: 1024}, false);
  check({...baseJob, pearlhash_k: 1025}, false);
  const maxK = (baseJob.pearlhash_rank ?? 0) * 64;
  check({...baseJob, pearlhash_k: maxK}, true);
  check({...baseJob, pearlhash_k: 8320}, true);
  check({...baseJob, pearlhash_k: 65536, pearlhash_rank: 1024}, true);
  check({...baseJob, pearlhash_k: 65600, pearlhash_rank: 1024}, false);
  check({...baseJob, pearlhash_k: 16384, pearlhash_rank: 1024}, true);
  check({...baseJob, pearlhash_rank: 64}, false);
  check({...baseJob, pearlhash_k: 32768, pearlhash_rank: 2048}, false);
  check({...baseJob, pearlhash_k: 3072, pearlhash_rank: 192}, false);
  check({...baseJob, pearlhash_k: 4160, pearlhash_rank: 256}, true);
  check({...baseJob, intensity: 131072, pearlhash_n: 524288, pearlhash_k: 8192,
    pearlhash_rank: 128}, true);
  check({...baseJob, intensity: 131072, pearlhash_n: 524288, pearlhash_k: 8320,
    pearlhash_rank: 128}, true);
  check({...baseJob, intensity: 131072, pearlhash_n: 4194304}, false);
});

test("completed direct tests close workers cooperatively on pass and mismatch", async () => {
  const passed = await loadMinerWithStubs({
    argv: ["node", "mom.js", "test", "rx/0", "expected", "--job.dev", "cpu"],
    waitForMessageType: "test",
  });
  passed.messageHandler({type: "test", thread_id: 0, value: {result: "expected"}});
  assert.equal(passed.process.exitCode, 0);
  assert.deepEqual(passed.workerCloseDeadlines, [null]);

  const failed = await loadMinerWithStubs({
    argv: ["node", "mom.js", "test", "rx/0", "expected", "--job.dev", "cpu"],
    waitForMessageType: "test",
  });
  failed.messageHandler({type: "test", thread_id: 0, value: {result: "mismatch"}});
  assert.equal(failed.process.exitCode, 1);
  assert.deepEqual(failed.workerCloseDeadlines, [null]);
});

test("observed direct test runtime errors preserve status without nested deadlines", async () => {
  const graceful = await loadMinerWithStubs({
    argv: ["node", "mom.js", "test", "rx/0", "expected", "--job.dev", "cpu"],
    env: {MOM_GPU_TEST_GRACEFUL_ONLY: "1"},
    waitForMessageType: "test",
  });
  graceful.messageHandler({type: "error", thread_id: 0, value: {message: "runtime error"}});
  graceful.messageHandler({type: "error", thread_id: 0, value: {message: "runtime error"}});
  await flushLifecycleCallbacks();
  assert.equal(graceful.process.exitCode, 1);
  assert.deepEqual(graceful.workerCloseDeadlines, [null]);
  assert.deepEqual(graceful.exitCodes, []);

  const ordinary = await loadMinerWithStubs({
    argv: ["node", "mom.js", "test", "rx/0", "expected", "--job.dev", "cpu"],
    env: {MOM_GPU_TEST_GRACEFUL_ONLY: undefined},
    waitForMessageType: "test",
  });
  ordinary.messageHandler({type: "error", thread_id: 0, value: {message: "runtime error"}});
  ordinary.messageHandler({type: "error", thread_id: 0, value: {message: "runtime error"}});
  await flushLifecycleCallbacks();
  assert.equal(ordinary.process.exitCode, 1);
  assert.deepEqual(ordinary.workerCloseDeadlines, [3000]);
  assert.deepEqual(ordinary.exitCodes, [1]);

  const nonTest = await loadMinerWithStubs({env: {MOM_GPU_TEST_GRACEFUL_ONLY: "1"}});
  nonTest.messageHandler({type: "error", thread_id: 0,
    value: {message: "runtime error", fatal: true}});
  assert.equal(nonTest.process.exitCode, 1);
  assert.deepEqual(nonTest.workerCloseDeadlines, [3000]);
});

/** @param {number} [count] */
async function flushLifecycleCallbacks(count = 2) {
  for (let i = 0; i < count; ++i) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

for (const response of [
  {name: "success", event: "algo_params", value: {"rx/0": "cpu"}},
  {name: "error", event: "error", value: {message: "late algo params"}},
]) {
  test("late algorithm-parameter " + response.name + " response after shutdown is ignored", async () => {
    const miner = await loadMinerWithStubs({deferCoreResponses: true});

    miner.process.emit("SIGTERM");
    miner.coreEvents.emit(response.event, response.value);
    await flushLifecycleCallbacks();

    assert.equal(miner.createdCoreCount(), 1);
    assert.equal(miner.recreatedThreadCount(), 0);
    assert.deepEqual(miner.sentMessages, []);
    assert.deepEqual(miner.poolConnects, []);
    assert.equal(miner.coreEvents.listenerCount("algo_params"), 0);
    assert.equal(miner.coreEvents.listenerCount("error"), 0);
    assert.deepEqual(miner.loggedErrors, []);
  });
}

for (const response of [
  {name: "success", event: "read_msr", value: {}},
  {name: "error", event: "error", value: {message: "late MSR response"}},
]) {
  test("late MSR " + response.name + " response after shutdown is ignored", async () => {
    const miner = await loadMinerWithStubs({
      deferCoreResponses: true, platform: "linux", env: {MOM_SKIP_MSR: "0"},
    });
    miner.coreEvents.emit("algo_params", {});
    assert.equal(miner.coreEvents.listenerCount("read_msr"), 1);
    assert.equal(miner.coreEvents.listenerCount("error"), 1);

    miner.process.emit("SIGTERM");
    miner.coreEvents.emit(response.event, response.value);
    await flushLifecycleCallbacks();

    assert.equal(miner.createdCoreCount(), 1);
    assert.equal(miner.recreatedThreadCount(), 0);
    assert.deepEqual(miner.sentMessages, []);
    assert.deepEqual(miner.poolConnects, []);
    assert.equal(miner.coreEvents.listenerCount("read_msr"), 0);
    assert.equal(miner.coreEvents.listenerCount("error"), 0);
    assert.deepEqual(miner.loggedErrors, []);
  });
}

test("delayed benchmark result after shutdown cannot start the next algorithm", async () => {
  const miner = await loadMinerWithStubs({
    algoParams: {etchash: "gpu1", kawpow: "gpu1"},
    waitForMessageType: "bench",
  });
  assert.equal(miner.sentMessages.filter((message) => message.type === "bench").length, 1);

  miner.process.emit("SIGTERM");
  miner.messageHandler({type: "hashrate", thread_id: 0, value: {hashrate: "1"}});
  await flushLifecycleCallbacks();

  assert.equal(miner.sentMessages.filter((message) => message.type === "bench").length, 1);
  assert.equal(miner.recreatedThreadCount(), 1);
  assert.deepEqual(miner.poolConnects, []);
});

test("delayed last benchmark result after shutdown cannot connect a pool", async () => {
  const miner = await loadMinerWithStubs({
    algoParams: {etchash: "gpu1"},
    waitForMessageType: "bench",
  });
  assert.equal(miner.sentMessages.filter((message) => message.type === "bench").length, 1);

  miner.process.emit("SIGTERM");
  miner.messageHandler({type: "hashrate", thread_id: 0, value: {hashrate: "1"}});
  await flushLifecycleCallbacks();

  assert.equal(miner.sentMessages.filter((message) => message.type === "bench").length, 1);
  assert.equal(miner.recreatedThreadCount(), 1);
  assert.deepEqual(miner.poolConnects, []);
});

test("queued tuning callback after shutdown cannot start another benchmark", async () => {
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "mine", "pool.example:1", "wallet", "--gpu_tune", "1"],
    algoParams: {etchash: "gpu1*[intensity=4096]"},
    waitForMessageType: "bench",
  });
  assert.equal(miner.sentMessages.filter((message) => message.type === "bench").length, 1);

  miner.messageHandler({type: "hashrate", thread_id: 0, value: {hashrate: "1"}});
  miner.messageHandler({type: "hashrate", thread_id: 0, value: {hashrate: "1"}});
  miner.process.emit("SIGTERM");
  await flushLifecycleCallbacks(3);

  assert.equal(miner.sentMessages.filter((message) => message.type === "bench").length, 1);
  assert.equal(miner.recreatedThreadCount(), 1);
  assert.deepEqual(miner.poolConnects, []);
});

test("pool jobs during shutdown do not recreate workers or dispatch", async () => {
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "mine", "pool.example:1", "wallet", "--bench_algo_params", "0"],
    algoParams: {"rx/0": "cpu", "rx/2": "cpu"},
  });
  const setJob = miner.getSetJob();
  setJob({algo: "rx/0", blob_hex: "00".repeat(43), difficulty: 1, job_id: "initial"});
  const initialConnects = miner.poolConnects.slice();
  const initialWorkers = miner.recreatedThreadCount();
  const initialMessages = miner.sentMessages.slice();

  miner.process.emit("SIGTERM");
  const nativeJob = setJob({
    algo: "rx/2", blob_hex: "11".repeat(43), difficulty: 1, job_id: "late",
  });

  assert.equal(nativeJob.algo, "rx/2");
  assert.equal(miner.recreatedThreadCount(), initialWorkers);
  assert.deepEqual(miner.poolConnects, initialConnects);
  assert.deepEqual(miner.sentMessages, initialMessages);
});

test("control-core one-shot responses remove their paired listeners", async () => {
  const mining = await loadMinerWithStubs();
  assert.equal(mining.coreEvents.listenerCount("read_msr"), 0);
  assert.equal(mining.coreEvents.listenerCount("error"), 0);

  const failedTuning = await loadMinerWithStubs({
    algoParamsError: true,
    argv: ["node", "mom.js", "test", "kawpow", "expected", "--job.dev", "gpu1"],
  });
  assert.equal(failedTuning.coreEvents.listenerCount("algo_params"), 0);
  assert.equal(failedTuning.coreEvents.listenerCount("error"), 0);
});

test("repeat schedules delayed callbacks", async () => {
  let calls = 0;

  /** @type {Promise<void>} */
  const repeated = new Promise((resolve) => {
    helper.repeat((next) => {
      calls += 1;
      if (calls === 2) {return resolve();}
      next();
    }, 1);
  });
  await repeated;

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
  assert.equal(helper.fullDiff2Target(1.5), "aa".repeat(32));
  assert.equal(helper.fullDiff2Target(".5"), "ff".repeat(32));
  assert.equal(helper.fullDiff2Target("1."), "ff".repeat(32));
  assert.equal(helper.fullDiff2Target("1.e1"),
    (((1n << 256n) - 1n) / 10n).toString(16).padStart(64, "0"));
  assert.throws(() => helper.fullDiff2Target("not-a-number"), /Invalid decimal value/);
});

test("scaled target conversions validate positive bigint arguments before difficulty", () => {
  assert.equal(helper.ethDiff2Target(0, 2n), "0".repeat(64));
  assert.equal(helper.fullDiff2Target(0, 2n), "ff".repeat(32));
  /** @type {unknown[]} */
  const invalid = [0n, -1n, 1, "1"];
  for (const value of invalid) {
    assert.throws(() => Reflect.apply(helper.ethDiff2Target, null, [1, value]), /positive bigint/);
    assert.throws(() => Reflect.apply(helper.fullDiff2Target, null, [1, value]), /positive bigint/);
    assert.throws(() => Reflect.apply(helper.ethDiff2Target, null, [0, value]), /positive bigint/);
    assert.throws(() => Reflect.apply(helper.fullDiff2Target, null, [0, value]), /positive bigint/);
  }
});

test("PearlHash V3 targets use the final worker matrix shape", () => {
  const amdTarget = helper.pearlhashTarget("1", 2048, 128);
  const nvidiaTarget = helper.pearlhashTarget("1", 4096, 256);

  assert.equal(amdTarget, "0".repeat(59) + "80000");
  assert.equal(nvidiaTarget, "0".repeat(59) + "80000");
  assert.equal(helper.pearlhashTargetWork(amdTarget, 2048, 128),
    (1n << 256n) - 524288n);
  assert.equal(helper.pearlhashTargetWork(nvidiaTarget, 4096, 256),
    (1n << 256n) - 524288n);
});

test("PearlHash V3 target overflow saturates at the largest usable target", () => {
  assert.equal(helper.pearlhashTarget("f".repeat(64), 2048, 128), "f".repeat(64));
});

test("PearlHash V3 targets apply rank normalization and reject legacy versions", () => {
  const target = helper.pearlhashTarget("1", 4096, 256);
  assert.equal(helper.pearlhashTarget("1", 2048, 128, 3),
    helper.pearlhashTarget("1", 2048, 128));
  assert.equal(helper.pearlhashTargetWork(target, 4096, 256),
    helper.target256ToWork(target) * 524288n);
  assert.doesNotThrow(() => helper.pearlhashTarget("1", 8192, 128));
  assert.throws(() => helper.pearlhashTarget("1", 2048, 64, 3), /requires rank >= 128/);
  assert.doesNotThrow(() => helper.pearlhashTarget("1", 4160, 256));
  assert.doesNotThrow(() => helper.pearlhashTarget("1", 8320, 128));
  assert.doesNotThrow(() => helper.pearlhashTargetWork(target, 8320, 128));
  assert.throws(() => helper.pearlhashTarget("1", 4096, 192), /rank/);
  assert.throws(() => helper.pearlhashTargetWork(target, 2147483648, 256), /PearlHash K/);
  for (const version of [0, 1, 2, 4, 2.5, NaN]) {
    assert.throws(() => helper.pearlhashTarget("1", 4096, 256, version), /certificate version/);
  }
});

test("target conversion rejects malformed and over-width values", () => {
  assert.throws(() => helper.target2diff("abc"), /Invalid target/);
  assert.throws(() => helper.target2diff("00".repeat(9)), /Invalid target/);
  assert.throws(() => helper.kawpowTarget2diff("f".repeat(65)), /Invalid target/);
  assert.throws(() => helper.target256ToWork("f".repeat(65)), /Invalid target/);
  assert.throws(() => helper.target256ToWork("not-hex"), /Invalid target/);
  assert.throws(() => helper.target256ToWork(["ff"]), /Invalid target/);
});

test("fractional PearlHash difficulty produces a full target", async () => {
  const miner = await loadMinerWithStubs();
  miner.getSetJob()({
    algo: "pearlhash", blob: "00".repeat(76), difficulty: 1.5, job_id: "job",
  });
  const message = miner.sentMessages.find((item) => item.type === "job");
  assert.ok(message);
  assert.ok(message.job);
  const job = message.job;
  assert.equal(job.target, "aa".repeat(32));
  assert.equal(job.noncebytes, 8);
});

test("PearlHash pool targets use the selected GPU profile", async () => {
  const miner = await loadMinerWithStubs({
    env: {MOM_GPU_BACKEND: "amd"},
    argv: ["node", "mom.js", "mine", "pool.example:1", "user", "--job.dev", "gpu1"],
  });
  miner.getSetJob()({
    algo: "pearlhash",
    blob: "00".repeat(76),
    difficulty: 1,
    job_id: "job",
    pearlhash_base_target: "1",
  });
  const message = miner.sentMessages.find((item) => item.type === "job");
  assert.ok(message);
  assert.ok(message.job);
  const job = message.job;

  assert.equal(job.pearlhash_k, 2048);
  assert.equal(job.pearlhash_rank, 128);
  assert.equal(job.pearlhash_cert_version, 3);
  assert.equal(job.target, "0".repeat(59) + "80000");
});

test("PearlHash final pool targets remain unchanged after GPU profile selection", async () => {
  const target = "00000000d1b71758e219652bd3c36113404ea4a8c154c985f06f694467381d7d";
  for (const backend of ["amd", "nvidia", "intel"]) {
    const miner = await loadMinerWithStubs({
      env: {MOM_GPU_BACKEND: backend},
      argv: ["node", "mom.js", "mine", "pool.example:1", "user", "--job.dev", "gpu1"],
    });
    miner.getSetJob()({algo: "pearlhash", blob: "00".repeat(76), job_id: "final", target});
    const job = miner.sentMessages.find((item) => item.type === "job")?.job;
    assert.ok(job);
    assert.equal(job.target, target, backend);
    assert.equal(job["pearlhash_base_target"], undefined);
  }
});

test("PearlHash V3 jobs reach the worker with the version and final-shape target", async () => {
  const miner = await loadMinerWithStubs({
    env: {MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "12.0"},
    argv: ["node", "mom.js", "mine", "pool.example:1", "user", "--job.dev", "gpu1"],
  });
  miner.getSetJob()({
    algo: "pearlhash", blob: "00".repeat(76), difficulty: 1, job_id: "v3-job",
    pearlhash_base_target: "1", pearlhash_cert_version: 3,
  });
  const job = miner.sentMessages.find((item) => item.type === "job")?.job;
  assert.ok(job);
  assert.equal(job.pearlhash_cert_version, 3);
  assert.equal(job.pearlhash_rank, 128);
  assert.ok(job.pearlhash_k);
  assert.equal(job.target, helper.pearlhashTarget("1", job.pearlhash_k, job.pearlhash_rank, 3));
  assert.throws(() => miner.getSetJob()({
    algo: "pearlhash", blob: "00".repeat(76), difficulty: 1, job_id: "unsupported",
    pearlhash_cert_version: 4,
  }), /certificate version/);
});

test("PearlHash V3 vector tuning reaches the worker without embedded shape fields", async () => {
  const v3 = require("../vectors/memory_hard").find(({name}) => name.startsWith("pearlhash v3 "));
  const portable = require("../vectors/equihash").find(({job}) => job.algo === "pearlhash");
  assert.ok(v3);
  assert.ok(portable);
  assert.ok(typeof v3.expected === "string");
  assert.equal(Object.hasOwn(v3.job, "pearlhash_k"), false);
  assert.equal(Object.hasOwn(v3.job, "pearlhash_rank"), false);
  assert.equal(v3.job.dev, "gpu1*[m=256;k=4096;rank=256]");
  assert.equal(v3.job.pearlhash_cert_version, 3);

  const direct = await loadMinerWithStubs({
    argv: ["node", "mom.js", "test", "pearlhash", v3.expected, "--job", JSON.stringify(v3.job)],
    env: {MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "12.0"},
    waitForMessageType: "test",
  });
  const testMessage = direct.sentMessages.find((message) => message.type === "test");
  assert.ok(testMessage);
  assert.ok(testMessage.job);
  assert.equal(testMessage.job.dev, v3.job.dev);
  assert.equal(testMessage.job.pearlhash_cert_version, 3);
  assert.equal(testMessage.job.pearlhash_k, 4096);
  assert.equal(testMessage.job.pearlhash_rank, 256);

  const live = await loadMinerWithStubs({
    argv: ["node", "mom.js", "mine", "pool.example:1", "wallet", "--bench_algo_params", "0"],
    env: {MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "12.0"},
    algoParams: {pearlhash: v3.job.dev},
  });
  const setJob = live.getSetJob();
  const v3Job = setJob({
    algo: "pearlhash", blob_hex: v3.job.blob_hex, difficulty: 1, job_id: "v3-vector",
    pearlhash_base_target: "1", pearlhash_cert_version: 3,
  });
  assert.equal(v3Job.pearlhash_cert_version, 3);
  assert.equal(v3Job.pearlhash_k, 4096);
  assert.equal(v3Job.pearlhash_rank, 256);
  assert.throws(() => setJob({
    algo: "pearlhash", blob_hex: v3.job.blob_hex, difficulty: 1, job_id: "invalid-version",
    pearlhash_cert_version: 4,
  }), /certificate version/);
  const defaultV3Job = setJob({
    algo: "pearlhash", blob_hex: portable.job.blob_hex, difficulty: 1, job_id: "v3-default",
    pearlhash_base_target: "1",
  });
  assert.equal(defaultV3Job.pearlhash_cert_version, 3);
});

test("PearlHash CLI accepts only V3 and defaults benchmarks to V3", async () => {
  const dev = "gpu1*[m=256;k=4096;rank=256]";
  /** @param {number | undefined} version */
  const bench = (version) => loadMinerWithStubs({
    argv: [
      "node", "mom.js", "bench", "pearlhash", "--job.dev", dev,
      ...(version === undefined ? [] : ["--job.pearlhash_cert_version", String(version)]),
    ],
    env: {
      MOM_GPU_BACKEND: "nvidia",
      MOM_NVIDIA_COMPUTE_CAPABILITY: "12.0",
      MOM_SKIP_MSR: "1",
    },
  });
  /** @param {Awaited<ReturnType<typeof loadMinerWithStubs>>} miner */
  const benchJob = (miner) => {
    const message = miner.sentMessages.find((item) => item.type === "bench");
    assert.ok(message);
    assert.ok(message.job);
    return message.job;
  };

  const v3 = benchJob(await bench(3));
  assert.equal(v3.pearlhash_cert_version, 3);
  assert.equal(typeof v3.pearlhash_cert_version, "number");
  assert.equal(v3.pearlhash_rank, 256);
  const implicitV3 = benchJob(await bench(undefined));
  assert.equal(implicitV3.pearlhash_cert_version, 3);
  assert.equal(implicitV3.pearlhash_rank, 256);

  const originalPrintHelp = opts.print_help;
  opts.print_help = (message) => {throw new Error(message);};
  try {
    await assert.rejects(() => bench(2.5), /param must be a safe integer/);
    await assert.rejects(() => bench(-1), /param must be non-negative/);
    for (const version of [0, 1, 2, 4]) {
      await assert.rejects(() => bench(version), /only supports PearlHash certificate version 3/);
    }
  } finally {
    opts.print_help = originalPrintHelp;
  }
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

test("malformed worker messages cannot crash or mutate the master", async () => {
  const miner = await loadMinerWithStubs();
  const backgroundJob = {
    job_id: "background", job_token: "background-token", nonce: "fffffffffffffffe",
  };
  miner.global.opt.pools.push(s.mockPoolConfig({last_job: backgroundJob}));
  const backgroundPool = miner.global.opt.pools[1];
  assert.ok(backgroundPool);
  backgroundPool.last_job = backgroundJob;
  miner.global.opt.pool_ids.active = 0;
  const errorCount = miner.loggedErrors.length;

  for (const message of [
    null,
    [],
    {type: "result", value: null},
    {type: "result", value: {pool_id: 99}},
    {type: "last_nonce", value: {pool_id: 99, nonce: "1"}},
    {type: "last_nonce", value: {pool_id: 1, nonce: "not-hex"}},
    {type: "last_nonce", value: {pool_id: 1, nonce: "1".repeat(17)}},
    {type: "last_nonce", value: {
      pool_id: 1, job_id: "x".repeat(257), job_token: "background-token", nonce: "1",
    }},
    {type: "last_nonce", value: {
      pool_id: 1, job_id: "background", job_token: "x".repeat(257), nonce: "1",
    }},
    {type: "hashrate", thread_id: "__proto__", value: {hashrate: "1"}},
    {type: "hashrate", thread_id: 0, value: {hashrate: "not-a-rate"}},
    {type: "test", thread_id: 0, value: {result: 1}},
    {type: "__proto__", value: {}},
    {type: "result", value: {
      pool_id: "0", job_id: "job", job_token: "x".repeat(257), worker_id: "worker",
      nonce: "1", hash: "11".repeat(32),
    }},
    {type: "result", value: {
      pool_id: "0", job_id: "job", job_token: "token", worker_id: "x".repeat(4097),
      nonce: "1", hash: "11".repeat(32),
    }},
    {type: "result", value: {
      pool_id: "0", job_id: "job", job_token: "token", worker_id: "worker",
      nonce: "1", header_hash: "11".repeat(258), hash: "11".repeat(32),
    }},
    {type: "result", value: {
      pool_id: "0", job_id: "job", job_token: "token", worker_id: "worker",
      nonce: "1", solution: "11".repeat(404), hash: "11".repeat(32),
    }},
    {type: "result", value: {
      pool_id: "0", job_id: "job", job_token: "token", worker_id: "worker",
      nonce: "1", edges: "00000001".repeat(43), hash: "11".repeat(32),
    }},
  ]) {
    assert.doesNotThrow(() => miner.messageHandler(message));
  }

  assert.equal(miner.poolWrites.length, 0);
  assert.equal(backgroundJob.nonce, "fffffffffffffffe");
  assert.ok(miner.loggedErrors.length > errorCount);
  for (const message of miner.loggedErrors.slice(errorCount)) {
    assert.match(message, /^(Invalid compute core message|Unknown compute core message type)$/);
  }

  miner.messageHandler({thread_id: 0, type: "last_nonce", value: {
    pool_id: "1", job_id: "background", job_token: "background-token", nonce: "1",
  }});
  assert.equal(backgroundJob.nonce, "fffffffffffffffe");
  miner.messageHandler({
    thread_id: 0, type: "last_nonce", value: {
      pool_id: "1", job_id: "stale", job_token: "background-token",
      nonce: "ffffffffffffffff",
    },
  });
  assert.equal(backgroundJob.nonce, "fffffffffffffffe");
  miner.messageHandler({
    thread_id: 0, type: "last_nonce", value: {
      pool_id: "1", job_id: "background", job_token: "background-token",
      nonce: "ffffffffffffffff",
    },
  });
  assert.equal(backgroundJob.nonce, "ffffffffffffffff");
});

test("late worker messages cannot affect a reused pool job ID", async () => {
  const miner = await loadMinerWithStubs();
  const currentJob = {job_id: "reused", job_token: "new-token", nonce: "1"};
  miner.global.opt.pools.push(s.mockPoolConfig({last_job: currentJob}));

  const result = {
    pool_id: "1", job_id: "reused", worker_id: "worker", nonce: "2",
    hash: "11".repeat(32),
  };
  miner.messageHandler({
    thread_id: 0, type: "result", value: {...result, job_token: "old-token"},
  });
  miner.messageHandler({thread_id: 0, type: "last_nonce", value: {
    pool_id: "1", job_id: "reused", job_token: "old-token", nonce: "2",
  }});
  assert.equal(miner.poolWrites.length, 0);
  assert.equal(currentJob.nonce, "1");

  miner.messageHandler({
    thread_id: 0, type: "result", value: {...result, job_token: "new-token"},
  });
  miner.messageHandler({thread_id: 0, type: "last_nonce", value: {
    pool_id: "1", job_id: "reused", job_token: "new-token", nonce: "2",
  }});
  assert.equal(miner.poolWrites.length, 1);
  assert.equal(currentJob.nonce, "2");
});

test("protocol submit modes reject incomplete worker results", async () => {
  const miner = await loadMinerWithStubs();
  const activePool = miner.global.opt.pools[0];
  assert.ok(activePool);
  const malformed = [
    {mode: "zelhash", value: {}},
    {mode: "hoosat", value: {}},
    {mode: "conflux", value: {}},
    {mode: "cortex", value: {header_hash: "11".repeat(32)}},
    {mode: "beam", value: {}},
    {mode: "raven", value: {header_hash: "22".repeat(32)}},
    {mode: "eth", value: {header_hash: "33".repeat(32)}},
  ];

  for (const {mode, value} of malformed) {
    activePool.last_job = {job_id: "job", job_token: "token", submit_mode: mode};
    miner.messageHandler({
      thread_id: 0,
      type: "result",
      value: {
        pool_id: "0", job_id: "job", job_token: "token", worker_id: "worker",
        nonce: "0000000000000001", ...value,
      },
    });
  }

  assert.equal(miner.poolWrites.length, 0);
  assert.equal(miner.loggedErrors.filter((message) =>
    message === "Invalid compute core message").length, malformed.length);
});

test("Panthera hash tests wait for every CPU batch result", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.job = {algo: "panthera", dev: "cpu*2"};

  assert.equal(miner.expectedTestThreads({thread_id: 0}), 2);
});

for (const algo of ["rx/0", "panthera"]) {
  test(`${algo} tests count mixed CPU batches independently of result order`, async () => {
    for (const order of [[0, 0, 1], [0, 1, 0], [1, 0, 0]]) {
      const miner = await loadMinerWithStubs({
        argv: ["node", "mom.js", "test", algo, "expected", "--job.dev", "cpu*2,cpu"],
        waitForMessageType: "test",
      });
      for (const [index, thread_id] of order.entries()) {
        miner.messageHandler({type: "test", thread_id, value: {result: "expected"}});
        assert.equal(miner.process.exitCode, index === order.length - 1 ? 0 : null,
          `result ${index + 1} in order ${order.join(",")}`);
      }
    }
  });

  test(`${algo} tests include repeated processes in mixed CPU batch totals`, async () => {
    const miner = await loadMinerWithStubs({
      argv: ["node", "mom.js", "test", algo, "expected", "--job.dev", "cpu*2^2,cpu^3"],
      waitForMessageType: "test",
    });
    for (let thread_id = 0; thread_id < 5; ++thread_id) {
      assert.equal(miner.expectedTestThreads({thread_id}), 7);
    }
    assert.equal(miner.expectedTestThreads({thread_id: -1}), 0);
    assert.equal(miner.expectedTestThreads({thread_id: 5}), 0);
    const order = [0, 2, 1, 3, 0, 4, 1];
    for (const [index, thread_id] of order.entries()) {
      miner.messageHandler({type: "test", thread_id, value: {result: "expected"}});
      assert.equal(miner.process.exitCode, index === order.length - 1 ? 0 : null);
    }
  });
}

test("repeated GPU workers may return identical multi-field test results", async () => {
  const miner = await loadMinerWithStubs();
  const result = "final_hash mix_hash";
  assert.equal(miner.matchesTestResult("kawpow", `${result} ${result}`, result), true);
  assert.equal(miner.matchesTestResult("kawpow", `${result} wrong_hash`, result), false);
});
