"use strict";

const {describe, test} = require("node:test");
const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const events = require("node:events");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");
const tls = require("node:tls");
const vm = require("node:vm");

const {formatHashrate, parseFormattedHashrate} = require("../common/miner_command");
const specReporter = require("../common/spec_reporter");
const repoRoot = path.join(__dirname, "..", "..");

/** @typedef {import("node:child_process").ChildProcess} ChildProcess */
/** @typedef {{method?: string, params?: unknown, id?: string | number | null, [key: string]: unknown}} JsonObject */
/** @typedef {{dev?: string, perf?: number | null, backend?: string, tuning?: JsonObject,
 *   [key: string]: unknown}} TestAlgoParam */
/** @typedef {{
 *   algo?: string | null, blob?: string, blob_hex?: string, target?: string, difficulty?: number,
 *   id?: string | number, job_id?: string | number, header_hash?: string, header?: string,
 *   nonce?: string | number,
 *   noncebytes?: number, nonceoffset?: number, proofsize?: number, height?: number,
 *   worker_id?: string | number, solution?: string, edges?: string, mix_hash?: string,
 *   nicehash_mask?: string, xn?: string, pearlhash_k?: number, pearlhash_rank?: number,
 *   pearlhash_cert_version?: number,
 *   plain_proof?: string, hash?: string, extra_nonce?: string, extranonce2?: string,
 *   ntime?: string, nonce1_len?: number, seed_hex?: string, pool_id?: number | string,
 *   dev?: string, [key: string]: unknown
 * }} TestJob
/** @typedef {{
 *   pool_id?: number, worker_id?: string | number, job_id?: string | number,
 *   nonce?: string | number, hash?: string,
 *   solution?: string, edges?: string, mix_hash?: string, header_hash?: string,
 *   plain_proof?: string, [key: string]: unknown
 * }} TestResult
/** @typedef {{type: string, job?: TestJob, value?: TestResult, [key: string]: unknown}} TestMessage */
/** @typedef {{
 *   url: string, port: number, is_tls: boolean, tls_verify?: boolean, is_nicehash?: boolean,
 *   is_keepalive: boolean, use_subscribe?: boolean, protocol?: string, inferred_protocol?: string,
 *   socket: SocketLike | null, keepalive: ReturnType<typeof setTimeout> | null,
 *   last_job: TestJob | null, pending_job?: PoolMessage, last_connect_time: number,
 *   pending_cortex_submit_ids?: Set<number>,
 *   good_shares: number, bad_shares: number,
 *   login: string, pass: string, worker?: string, worker_id?: string, logged_in: boolean,
 *   algo_params?: Record<string, TestAlgoParam>, extra_nonce?: string, extra_nonce2_size?: number,
 *   beam_nonceprefix?: string, xelis_extra_nonce?: string, xelis_public_key?: string,
 *   zelhash_target?: string, ironfish_xn?: string, ironfish_target?: string,
 *   stratum_target?: string,
 *   kaspa_difficulty?: number, pending_authorize?: boolean, donation_until?: number,
 *   xelis_difficulty?: number | string,
 *   [key: string]: unknown
 * }} TestPoolConfig
/** @typedef {{
 *   log_level: number, job: TestJob, pools: [TestPoolConfig, ...TestPoolConfig[]],
 *   pool_ids: {active?: number, primary?: number, donate?: number | null},
 *   pool_time: {first_job_wait: number, connect_throttle: number, close_wait: number, keepalive: number,
 *     donate_length?: number, donate_interval?: number, stats?: number, [key: string]: unknown},
 *   algo_params: Record<string, TestAlgoParam>, bench_algo_params: number,
 *   default_msrs: Record<string, JsonObject>, gpu_tune: number, save_config: string, [key: string]: unknown
 * }} TestOptions
/** @typedef {{
 *   algoParams?: Record<string, string>, algoParamsError?: boolean, argv?: string[],
 *   env?: NodeJS.ProcessEnv, platform?: NodeJS.Platform, stdin?: import("node:stream").Readable,
 *   waitForMessageType?: string, deferCoreResponses?: boolean,
 *   closeWorkerTargets?: Array<{type: "subprocess" | "cluster", worker: import("node:events").EventEmitter & {
 *     exitCode?: number | null, signalCode?: string | null, killed?: boolean,
 *     isDead?: () => boolean
 *   }}>
 * }} MinerStubOptions
/** @typedef {{
 *   expectedTestThreads: (message: {thread_id: number}) => number,
 *   matchesTestResult: (algo: string, actual: string, expected: string) => boolean,
 *   messageHandler: (message: unknown) => void, publicAlgoParams: (params: Record<string, string>) => JsonObject,
 *   startDonationWindow: () => void
 * }} MinerTestExports */
/** @typedef {{__test?: MinerTestExports}} VmExports */
/** @typedef {{opt: TestOptions}} TestGlobal */
/** @typedef {import("node:events").EventEmitter & {
 *   argv: string[], env: NodeJS.ProcessEnv, platform: NodeJS.Platform,
 *   stdin: import("node:stream").Readable,
 *   stdout: NodeJS.WriteStream, stderr: NodeJS.WriteStream, exitCode: number | null,
 *   exit: (code?: number) => never
 * }} StubProcess */
/** @typedef {{
 *   write?: (message: string) => void, destroy?: () => void, destroyed?: boolean
 * }} SocketLike */
/** @typedef {import("node:events").EventEmitter & SocketLike} MockSocket */
/** @type {{opt?: TestOptions}} */
const testGlobal = /** @type {{opt?: TestOptions}} */ (/** @type {unknown} */ (globalThis));
/** @type {typeof import("../../helper.js") & typeof import("../../helper/diagnostics.js") &
 *   typeof import("../../helper/hash.js")} */
const helper = require("../../helper.js");
/** @type {typeof import("../../pool.js")} */
const pool = require("../../pool.js");
/** @type {typeof import("../../opts.js")} */
const opts = require("../../opts.js");
/** @type {typeof import("../../compiler-policy.js")} */
const compilerPolicy = require("../../compiler-policy.js");
const noOp = () => undefined;

/** @param {PoolJob} _job @returns {never} */
function unexpectedPoolJob(_job) {
  throw new Error("unexpected pool job");
}

/** @param {PoolJob} job @returns {MiningJob} */
function completeMiningJob(job) {
  if (typeof job.algo !== "string") {
    throw new Error("pool job is missing mining fields");
  }
  const {submit_mode: _submitMode, ...nativeJob} = job;
  return {...nativeJob, algo: job.algo, dev: typeof job.dev === "string" ? job.dev : "cpu"};
}

/** @param {MinerStubOptions} [options] @returns {Promise<{
 *   getSetJob: () => ((job: TestJob) => LiveNativeJob), global: TestGlobal,
 *   expectedTestThreads: MinerTestExports["expectedTestThreads"],
 *   matchesTestResult: MinerTestExports["matchesTestResult"],
 *   messageHandler: MinerTestExports["messageHandler"],
 *   publicAlgoParams: MinerTestExports["publicAlgoParams"],
 *   startDonationWindow: MinerTestExports["startDonationWindow"], coreEvents: import("node:events").EventEmitter,
 *   process: StubProcess, loggedErrors: string[], loggedMessages: string[], poolConnects: number[],
 *   poolSwitches: number[], poolWrites: Array<{pool_id: number, json: JsonObject}>, sentMessages: TestMessage[],
 *   workerCloseDeadlines: number[], exitCodes: number[], recreatedThreadCount: () => number,
 *   createdCoreCount: () => number, writtenStdout: string
 * }>} */
async function loadMinerWithStubs(options = {}) {
  const source = fs.readFileSync(path.join(repoRoot, "mom.js"), "utf8");
  /** @type {{exports: VmExports}} */
  const moduleStub = {exports: {}};
  /** @type {{opt?: TestOptions}} */
  const globalStub = {};
  const coreEvents = new events.EventEmitter();
  /** @type {TestMessage[]} */
  const sentMessages = [];
  /** @type {number[]} */
  const workerCloseDeadlines = [];
  /** @type {number[]} */
  const exitCodes = [];
  let createdCoreCount = 0;
  let recreatedThreadCount = 0;
  /** @type {Array<{pool_id: number, json: JsonObject}>} */
  const poolWrites = [];
  /** @type {number[]} */
  const poolConnects = [];
  /** @type {number[]} */
  const poolSwitches = [];
  /** @type {string[]} */
  const loggedErrors = [];
  /** @type {string[]} */
  const loggedMessages = [];
  /** @type {string[]} */
  const writtenStdout = [];
  /** @type {((job: TestJob) => LiveNativeJob) | null} */
  let capturedSetJob = null;
  const algoParams = options.algoParams || {};
  const helperStub = {
    ...helper,
    cluster_process: () => false,
    create_core: () => {
      ++createdCoreCount;
      return {
        from: coreEvents,
        emit_to: /** @param {string} name */ (name) => {
          if (options.deferCoreResponses) {return;}
          if (name === "algo_params") {
            const event = options.algoParamsError ? "error" : "algo_params";
            const value = options.algoParamsError ? {message: "algo params failed"} : algoParams;
            setImmediate(() => coreEvents.emit(event, value));
          }
          if (name === "read_msr") {setImmediate(() => coreEvents.emit("error", {message: "skip"}));}
        },
      };
    },
    exit_now: /** @param {number} code */ (code) => exitCodes.push(code),
    recreate_threads: () => {++recreatedThreadCount;},
    closeWorkers: /** @param {number} deadline */ (deadline) => {
      workerCloseDeadlines.push(deadline);
      return options.closeWorkerTargets || [];
    },
    messageWorkers: /** @param {TestMessage} msg @returns {number} */
      (msg) => sentMessages.push(msg),
    log: /** @param {string} message */ (message) => loggedMessages.push(message),
    log1: noOp,
    log2: noOp,
    log3: noOp,
    log_err: /** @param {string} message */ (message) => loggedErrors.push(message),
  };
  const poolStub = {
    connect_pool_throttle: /** @param {number} pool_id @param {(job: TestJob) => LiveNativeJob} setJob */
      (pool_id, setJob) => {
        poolConnects.push(pool_id);
        capturedSetJob = setJob;
      },
    pool_write: /** @param {number} pool_id @param {JsonObject} json */
      (pool_id, json) => {
        if (typeof pool_id !== "number") {
          throw new TypeError("pool_write requires a numeric pool ID");
        }
        poolWrites.push({pool_id, json});
      },
    switch_pool: /** @param {number} pool_id */ (pool_id) => poolSwitches.push(pool_id),
  };
  // Give every VM-loaded miner its own signal emitter. Inheriting from the real process object also
  // inherits its internal EventEmitter state, so repeated tests otherwise leak signal handlers into
  // the test runner and eventually trigger MaxListenersExceededWarning.
  /** @type {StubProcess} */
  const processStub = Object.assign(new events.EventEmitter(), {
    argv: options.argv || ["node", "mom.js", "mine", "pool.example:1", "user"],
    env: {...process.env, ...(options.env || {})},
    platform: options.platform || process.platform,
    stderr: process.stderr,
    stdin: options.stdin || process.stdin,
    stdout: process.stdout,
    exitCode: null,
    exit: /** @param {number | undefined} code */ (code) => { throw new Error(`unexpected exit ${code}`); },
  });
  const fsStub = {
    ...fs,
    writeSync: /** @param {number} fd @param {string} data */ (fd, data) => {
      if (fd === 1) {
        writtenStdout.push(data);
        return data.length;
      }
      return fs.writeSync(fd, data);
    },
  };
  /** @param {(...args: never[]) => void} callback @param {number | undefined} delay
   * @param {...never} args */
  const detachedSetTimeout = function(callback, delay, ...args) {
    const timer = setTimeout(() => callback(...args), delay);
    if (timer.unref) {timer.unref();}
    return timer;
  };
  /** @param {string} id */
  const requireStub = (id) => {
    if (id === "node:fs") {return fsStub;}
    if (id === "./helper.js") {return helperStub;}
    if (id === "./pool.js") {return poolStub;}
    if (id === "./opts.js") {return opts;}
    if (id === "./compiler-policy.js") {return require("../../compiler-policy.js");}
    if (id === "./gpu-tuning.js") {return require("../../gpu-tuning.js");}
    if (id.startsWith("./miner/")) {return require(path.join(repoRoot, id));}
    return require(id);
  };

  const wrappedSource = `(function(
    require, module, exports, process, global, console, Buffer,
    setTimeout, clearTimeout, setInterval, setImmediate
  ) {
    ${source}
  })`;
  vm.runInNewContext(wrappedSource, {})(
    requireStub, moduleStub, moduleStub.exports, processStub, globalStub, console, Buffer,
    detachedSetTimeout, clearTimeout, noOp, setImmediate,
  );

  const hasExpectedMessage = () =>
    options.waitForMessageType && sentMessages.some((msg) => msg.type === options.waitForMessageType);
  for (let i = 0; i < 10 && !capturedSetJob && !hasExpectedMessage(); ++i) {
    await new Promise((resolve) => setImmediate(resolve));
  }
  const testExports = moduleStub.exports.__test;
  if (!testExports) {throw new Error("VM miner did not expose test helpers");}
  if (!globalStub.opt) {throw new Error("VM miner did not initialize global options");}
  return {
    getSetJob: () => {
      if (!capturedSetJob) {throw new Error("VM miner did not register a pool job callback");}
      return capturedSetJob;
    },
    global: /** @type {TestGlobal} */ (globalStub),
    expectedTestThreads: testExports.expectedTestThreads,
    matchesTestResult: testExports.matchesTestResult,
    messageHandler: testExports.messageHandler,
    publicAlgoParams: testExports.publicAlgoParams,
    startDonationWindow: testExports.startDonationWindow,
    coreEvents,
    process: processStub,
    loggedErrors,
    loggedMessages,
    poolConnects,
    poolSwitches,
    poolWrites,
    sentMessages,
    workerCloseDeadlines,
    exitCodes,
    recreatedThreadCount: () => recreatedThreadCount,
    createdCoreCount: () => createdCoreCount,
    writtenStdout: writtenStdout.join(""),
  };
}

/** @param {Partial<TestPoolConfig>} [overrides] @returns {TestPoolConfig} */
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

/** @param {{pool?: Partial<TestPoolConfig>, pool_time?: Partial<TestOptions["pool_time"]>, opt?: Partial<TestOptions>}} [options]
 * @returns {TestOptions} */
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

/** @param {{pool?: Partial<TestPoolConfig>, pool_time?: Partial<TestOptions["pool_time"]>, opt?: Partial<TestOptions>,
 *   socket?: MockSocket,
 *   write?: (message: string) => void, destroy?: () => void}} options
 * @param {(value: {socket: MockSocket,
 *   writes: JsonObject[], switched: () => boolean, poolConfig: TestPoolConfig}) => Promise<unknown>} callback */
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

/** @param {{messageHandler: (message: TestMessage) => void}} miner @param {string} [rate] */
function completeOneBenchmark(miner, rate = "1") {
  miner.messageHandler({type: "hashrate", thread_id: 0, value: {hashrate: rate}});
}

module.exports = {
  describe, test, assert, spawnSync, events, fs, path, tls,
  opts, helper, pool, compilerPolicy, formatHashrate, parseFormattedHashrate,
  specReporter, repoRoot, noOp, loadMinerWithStubs, mockPoolConfig,
  mockPoolOptions, withMockPool, completeOneBenchmark, unexpectedPoolJob, completeMiningJob
};
