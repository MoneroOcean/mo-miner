"use strict";

const crypto = require("node:crypto");
const {EventEmitter} = require("node:events");
const os = require("node:os");
const {PassThrough} = require("node:stream");
const s = require("./support");
const {
  test, pool, opts, noOp, fs, path, helper, unexpectedPoolJob, loadMinerWithStubs, mockPoolConfig, withMockPool,
  completeOneBenchmark,
} = s;
/** @type {typeof import("node:assert/strict")} */
const assert = s.assert;
/** @type {(job: PoolJob) => MiningJob} */
const completeMiningJob = s.completeMiningJob;
const makeMinerExecution = require("../common/miner_execution");
const {benchmarkTestTimeoutMs, benchmarkTimeoutMs} = require("../common/miner_command");
/** @typedef {EventEmitter & {stdin: PassThrough, stdout: PassThrough, stderr: PassThrough}} HarnessChild */
/** @typedef {{callback: () => void, delay: number, cleared: boolean, unref: () => undefined}} HarnessTimer */
/** @typedef {Record<string, string | undefined> & {
 *   MOM_PERF_SAMPLES?: string, MOM_TEST_MARKER?: string}} HarnessEnv */
/** @typedef {Awaited<ReturnType<typeof loadMinerWithStubs>>} StubMiner */
/** @typedef {StubMiner["sentMessages"][number]} MiningMessage */
/** @typedef {NonNullable<MiningMessage["job"]>} MiningTestJob */
/** @typedef {{dev?: string, perf?: number | null, backend?: string,
 *   tuning?: Record<string, unknown>, [key: string]: unknown}} MiningAlgoParam */
/** @typedef {{algo: string[], ["algo-perf"]: Record<string, number> & {
 *   kawpow1?: number, c29?: number, etchash?: number}, pass: string}} LoginParams */

/** @param {MiningMessage | undefined} message @returns {MiningTestJob} */
function requireJobMessage(message) {
  assert.ok(message);
  assert.ok(message.job);
  return message.job;
}

/** @param {StubMiner} miner @param {number | null | undefined} poolId */
function requirePool(miner, poolId) {
  assert.ok(poolId !== null && poolId !== undefined);
  const pool = miner.global.opt.pools[poolId];
  assert.ok(pool);
  return pool;
}

/** @param {StubMiner} miner @param {string} algo @returns {MiningAlgoParam} */
function requireAlgoParam(miner, algo) {
  const value = miner.global.opt.algo_params[algo];
  assert.ok(value);
  return /** @type {MiningAlgoParam} */ (value);
}

/** @param {{algo_params?: Record<string, MiningAlgoParam>} | undefined} pool
 * @returns {Record<string, MiningAlgoParam>} */
function requirePoolAlgoParams(pool) {
  assert.ok(pool && pool.algo_params);
  return pool.algo_params;
}

/** @param {{algo_params?: Record<string, MiningAlgoParam>} | undefined} pool @param {string} algo */
function requirePoolAlgoParam(pool, algo) {
  const params = requirePoolAlgoParams(pool);
  const value = params[algo];
  assert.ok(value);
  return value;
}

/** @param {unknown} value @returns {value is LoginParams} */
function isLoginParams(value) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {return false;}
  const record = /** @type {{algo?: unknown, pass?: unknown, ["algo-perf"]?: unknown}} */ (value);
  const performance = record["algo-perf"];
  return Array.isArray(record.algo) && record.algo.every((algo) => typeof algo === "string") &&
    typeof record.pass === "string" && typeof performance === "object" &&
    performance !== null && !Array.isArray(performance) &&
    Object.values(performance).every((rate) =>
      typeof rate === "number" && Number.isFinite(rate));
}

/** @param {{params?: unknown} | undefined} message @returns {LoginParams} */
function requireLoginParams(message) {
  assert.ok(message);
  assert.ok(isLoginParams(message.params));
  return message.params;
}

/** @param {Awaited<ReturnType<ReturnType<typeof benchmarkHarness>["execution"]["runMinerBench"]>>} result */
function requireBenchmarkResult(result) {
  assert.ok("hashrate" in result && typeof result.hashrate === "number" &&
    Number.isFinite(result.hashrate));
  return result;
}

for (const {name, options} of [
  {name: "normal mining", options: {}},
  {name: "test mode", options: {
    argv: ["node", "mom.js", "test", "rx/0", "expected", "--job.dev", "cpu"],
    waitForMessageType: "test",
  }},
]) {
  test("fatal worker errors shut down " + name, async () => {
    const miner = await loadMinerWithStubs(options);
    miner.messageHandler({type: "error", thread_id: 0, value: {message: "worker exited", fatal: true}});
    assert.equal(miner.process.exitCode, 1);
    await new Promise((resolve) => setImmediate(resolve));
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(miner.exitCodes, []);
  });
}

test("ordinary native worker errors remain recoverable", async () => {
  const miner = await loadMinerWithStubs();
  miner.messageHandler({type: "error", thread_id: 0, value: {message: "transient"}});
  assert.equal(miner.process.exitCode, null);
  assert.equal(miner.loggedErrors.at(-1), 'Compute core error: "transient"');
});

test("duplicate worker jobs remain ignored even when marked fatal", async () => {
  const miner = await loadMinerWithStubs();
  const errorCount = miner.loggedErrors.length;
  miner.messageHandler({type: "error", thread_id: 0,
    value: {message: "Ignore duplicate job", fatal: true}});
  assert.equal(miner.process.exitCode, null);
  assert.equal(miner.loggedErrors.length, errorCount);
});

test("fatal first-run benchmark records zero and starts the next algorithm", async () => {
  const miner = await loadMinerWithStubs({
    algoParams: {etchash: "gpu1", kawpow: "gpu1"},
    waitForMessageType: "bench",
  });
  assert.equal(requireJobMessage(miner.sentMessages[0]).algo, "etchash");

  miner.messageHandler({type: "error", thread_id: 0,
    value: {message: "worker exited", fatal: true}});

  assert.equal(requireAlgoParam(miner, "etchash").perf, 0);
  assert.equal(requireJobMessage(miner.sentMessages[1]).algo, "kawpow");
  assert.equal(miner.process.exitCode, null);
});

/** @param {HarnessTimer | undefined} timer */
function requireTimer(timer) {
  assert.ok(timer);
  return timer;
}

function benchmarkHarness(graceful = true, missingGpu = false, runNodeResult = {}) {
  /** @type {HarnessChild[]} */
  const children = [];
  /** @type {Array<HarnessEnv | undefined>} */
  const envs = [];
  /** @type {HarnessTimer[]} */
  const timers = [];
  /** @param {() => void} callback @param {number} delay @returns {HarnessTimer} */
  const schedule = (callback, delay) => {
    const timer = {callback, delay, cleared: false, unref: noOp};
    timers.push(timer);
    return timer;
  };
  /** @param {HarnessTimer} timer */
  const cancel = (timer) => {timer.cleared = true;};
  const execution = makeMinerExecution({
    compilerPolicy: {parseReportedAlgoParam: () => ({})},
    getAutoAlgoParams: async () => ({}),
    runNode: async (_args) => ({
      code: null, signal: null, error: null, stdout: "", stderr: "", ...runNodeResult,
    }),
    formatFailure: (title, _args, result) =>
      `${title}: ${result.error?.message || result.signal || `exit ${result.code}`}`,
    emitGitHubError: noOp,
    isMissingGpuOutput: () => missingGpu,
    spawnMiner: (_args, env) => {
      const child = Object.assign(new EventEmitter(), {
        stdin: new PassThrough(),
        stdout: new PassThrough(),
        stderr: new PassThrough(),
      });
      if (graceful) {
        child.stdin.on("data", () => setImmediate(() => child.emit("close", 0, null)));
      }
      children.push(child);
      envs.push(env);
      return child;
    },
    appendOutput: (result, stream, chunk) => {
      result[stream] = (result[stream] + chunk.toString()).slice(-1024 * 1024);
    },
    createRunResult: () => ({code: null, signal: null, error: null, stdout: "", stderr: ""}),
    killProcessTree: (child, signal) => {
      if (!signal) {setImmediate(() => child.emit("close", null, "SIGKILL"));}
      return true;
    },
    medianHashrate: (samples) => {
      const value = [...samples].sort((a, b) => a - b)[Math.floor(samples.length / 2)];
      return value;
    },
    hashrateUnitMultipliers: {"H/s": 1},
    escapeRegExp: (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
    parseFormattedHashrate: (value, unit) => unit === "H/s" ? Number(value) : Number.NaN,
    setTimeout: schedule,
    clearTimeout: cancel,
  });
  return {children, envs, execution, timers};
}

test("native Pearl GPU tests require the search checksum marker", async () => {
  const definition = {name: "pearl", gpu: true, expected: "ok", job: {algo: "pearlhash", dev: "gpu1"}};
  for (const backend of [undefined, "auto", "native", "sycl-native"]) {
    for (const marker of ["", "PEARLHASH_TEST search_checksum_match=false"]) {
      const harness = benchmarkHarness(true, false, {code: 0, stdout: `PASSED\n${marker}`});
      const selected = backend === undefined
        ? definition : {...definition, job: {...definition.job, backend}};
      await assert.rejects(harness.execution.runMinerTest(selected), /did not report a clean pass/);
    }
    const harness = benchmarkHarness(true, false, {
      code: 0, stdout: "PASSED\nPEARLHASH_TEST search_checksum_match=true",
    });
    const selected = backend === undefined
      ? definition : {...definition, job: {...definition.job, backend}};
    await harness.execution.runMinerTest(selected);
  }
});

test("checksum marker remains optional for CPU, portable Pearl, and non-Pearl tests", async () => {
  for (const definition of [
    {name: "cpu pearl", gpu: false, expected: "ok", job: {algo: "pearlhash", dev: "cpu"}},
    {name: "sycl pearl", gpu: true, expected: "ok", job: {algo: "pearlhash", dev: "gpu1", backend: "sycl"}},
    {name: "opencl pearl", gpu: true, expected: "ok", job: {algo: "pearlhash", dev: "gpu1", backend: "sycl-opencl"}},
    {name: "other", gpu: true, expected: "ok", job: {algo: "nexapow", dev: "gpu1", backend: "native"}},
  ]) {
    const harness = benchmarkHarness(true, false, {code: 0, stdout: "PASSED"});
    await harness.execution.runMinerTest(definition);
  }
});

test("automatic Pearl reference fallback passes only with explicit reference and host-seed evidence", async () => {
  const definition = {name: "pearl fallback", gpu: true, expected: "ok",
    job: {algo: "pearlhash", dev: "gpu1", backend: "auto"}};
  const seeds = "PEARLHASH_TEST search=sycl cert_version=3 host_seeds_match=true";
  const reference = "PEARLHASH_TEST reference_search=true";
  for (const markers of ["", seeds, reference, `${seeds}\n${reference.replace("true", "false")}`,
    `${seeds.replace("true", "false")}\n${reference}`]) {
    const harness = benchmarkHarness(true, false, {code: 0, stdout: `PASSED\n${markers}`});
    await assert.rejects(harness.execution.runMinerTest(definition), /did not report a clean pass/);
  }
  const harness = benchmarkHarness(true, false, {
    code: 0, stdout: `PASSED\n${seeds}`, stderr: reference,
  });
  await harness.execution.runMinerTest(definition);
});

/** @param {ReturnType<typeof benchmarkHarness>} harness @param {number} [index]
 * @returns {Promise<HarnessChild>} */
async function benchChild(harness, index = 0) {
  while (!harness.children[index]) {await new Promise((resolve) => setImmediate(resolve));}
  const child = harness.children[index];
  assert.ok(child);
  return child;
}

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
    algoParams: {kawpow: "gpu1*[intensity=1]"},
  });

  assert.equal(typeof miner.getSetJob(), "function");
  assert.deepEqual(miner.sentMessages, []);
});

test("same-algorithm pool changes do not reuse another pool's nonce state", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const firstPool = miner.global.opt.pool_ids.active;
  setJob({
    algo: "cn/0", blob_hex: "00".repeat(43), difficulty: 1,
    job_id: "first", nonce: "12345678", nicehash_mask: "ffffffff",
  });

  const secondPool = miner.global.opt.pools.push(mockPoolConfig()) - 1;
  miner.global.opt.pool_ids.active = secondPool;
  setJob({algo: "cn/0", blob_hex: "11".repeat(43), difficulty: 1, job_id: "second"});

  const job = requireJobMessage(miner.sentMessages.at(-1));
  assert.notEqual(firstPool, secondPool);
  assert.equal(job.pool_id, secondPool);
  assert.equal(job.nonce, "0");
  assert.equal(job.nicehash_mask, "00000000");
});

test("nonce state snapshots preserve same-algorithm progress and explicit overrides", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.job.dev = "cpu";
  const setJob = miner.getSetJob();
  const work = {algo: "cn/0", blob_hex: "00".repeat(43), difficulty: 1, job_id: "same"};
  const first = setJob({...work, nonce: "17", nicehash_mask: "12345678"});
  const continued = setJob(work);
  assert.equal(continued.nonce, first.nonce);
  assert.equal(continued.nicehash_mask, first.nicehash_mask);
  assert.notEqual(continued.job_token, first.job_token);
  const zero = setJob({...work, nonce: 0});
  assert.equal(zero.nonce, 0);
  assert.equal(zero.nicehash_mask, first.nicehash_mask);
  const overridden = setJob({...work, nonce: "23", nicehash_mask: "00000000"});
  assert.equal(overridden.nonce, "23");
  assert.equal(overridden.nicehash_mask, "00000000");
  const switched = setJob({...work, algo: "kawpow", blob_hex: "00".repeat(40), noncebytes: 8});
  assert.equal(switched.nonce, "0");
  assert.equal(switched.nicehash_mask, "00".repeat(8));
  const restored = setJob(work);
  assert.equal(restored.nonce, "0");
  assert.equal(restored.nicehash_mask, "00000000");
});

test("NiceHash defaults protect trailing generic four-byte nonce slots", async () => {
  for (const {algo, blob_hex, noncebytes, nonceoffset, expected} of [
    {algo: "cn/0", blob_hex: "00".repeat(43), noncebytes: 4, nonceoffset: 39,
      expected: "000000ff"},
    {algo: "cn/0", blob_hex: "00".repeat(43), noncebytes: 8, nonceoffset: 35,
      expected: "ff00000000000000"},
    {algo: "c29", blob_hex: "00".repeat(43), noncebytes: 4, nonceoffset: 39,
      expected: "ff000000"},
    {algo: "verthash", blob_hex: "00".repeat(80), noncebytes: 4, nonceoffset: 76,
      expected: "ff000000"},
  ]) {
    const miner = await loadMinerWithStubs();
    const poolConfig = miner.global.opt.pools[0];
    poolConfig.is_nicehash = true;
    miner.global.opt.job.dev = "cpu";
    const job = miner.getSetJob()({
      algo, blob_hex, noncebytes, nonceoffset, difficulty: 1, job_id: algo,
    });
    assert.equal(job.nicehash_mask, expected, algo);
  }

  const prefixed = await loadMinerWithStubs();
  prefixed.global.opt.pools[0].is_nicehash = true;
  prefixed.global.opt.job.dev = "cpu";
  const prefixedJob = prefixed.getSetJob()({
    algo: "cn/0", blob_hex: "00".repeat(43), noncebytes: 4, nonceoffset: 39,
    xn: "ab", difficulty: 1, job_id: "prefixed",
  });
  assert.equal(prefixedJob.nicehash_mask, "ff000000");

  const explicit = await loadMinerWithStubs();
  explicit.global.opt.pools[0].is_nicehash = true;
  explicit.global.opt.job.dev = "cpu";
  const explicitJob = explicit.getSetJob()({
    algo: "cn/0", blob_hex: "00".repeat(43), noncebytes: 4, nonceoffset: 39,
    nicehash_mask: "12345678", difficulty: 1, job_id: "explicit",
  });
  assert.equal(explicitJob.nicehash_mask, "12345678");
});

test("prefixed pool jobs retain valid saved progress across a pause without reusing replaced jobs", async () => {
  for (const noncebytes of [4, 8]) {
    const miner = await loadMinerWithStubs();
    const setJob = miner.getSetJob();
    const poolConfig = miner.global.opt.pools[0];
    const fields = {algo: "cn/0", blob_hex: "00".repeat(noncebytes), noncebytes,
      nonceoffset: 0, xn: "a1", difficulty: 1, job_id: "same"};
    const previous = {...fields};
    const initial = setJob(previous);
    poolConfig.last_job = previous;
    miner.global.opt.pool_ids.active = 1;
    const saved = "a1" + "00".repeat(noncebytes - 2) + "17";
    const progress = (/** @type {string} */ nonce, job_token = initial.job_token) => miner.messageHandler({
      type: "last_nonce", thread_id: 0,
      value: {pool_id: "0", job_id: "same", job_token, nonce},
    });
    progress(saved);
    for (const nonce of ["a2" + "ff".repeat(noncebytes - 1), "1" + "00".repeat(noncebytes), "xyz"]) {
      progress(nonce);
      assert.equal(poolConfig.last_job.nonce, saved);
    }
    miner.global.opt.pool_ids.active = 0;
    const resumed = setJob(previous);
    assert.equal(resumed.nonce, saved);
    assert.equal(resumed.nicehash_mask, initial.nicehash_mask);
    assert.notEqual(resumed.job_token, initial.job_token);

    const next = {...fields, blob_hex: "11".repeat(noncebytes)};
    const replacement = setJob(next);
    poolConfig.last_job = next;
    assert.equal(replacement.nonce, initial.nonce);
    miner.global.opt.pool_ids.active = 1;
    progress(saved, resumed.job_token);
    assert.equal(poolConfig.last_job.nonce, undefined);
    progress(saved, replacement.job_token);
    assert.equal(poolConfig.last_job.nonce, saved);
  }
});

test("C29 prefix jobs reseed free nonce bytes after an algorithm switch", async (t) => {
  const seededSuffixes = ["801122334455", "fedcba987654"];
  let randomCalls = 0;
  t.mock.method(crypto, "randomBytes", (/** @type {number} */ length) => {
    const suffix = seededSuffixes[randomCalls++];
    assert.ok(suffix);
    assert.equal(suffix.length, length * 2);
    return Buffer.from(suffix, "hex");
  });

  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const shared = {
    blob_hex: "00".repeat(40), noncebytes: 8, nonceoffset: 0,
    xn: "a1b2", difficulty: 1,
  };
  const first = setJob({...shared, algo: "c29", job_id: "c29-first"});
  const switched = setJob({...shared, algo: "kawpow", nonceoffset: 32, job_id: "kawpow"});
  const second = setJob({...shared, algo: "c29", job_id: "c29-second"});

  assert.equal(randomCalls, 2);
  assert.equal(first.blob_hex, shared.blob_hex);
  assert.equal(switched.blob_hex, shared.blob_hex);
  assert.equal(second.blob_hex, shared.blob_hex);
  assert.notEqual(first.job_id, switched.job_id);
  assert.notEqual(switched.job_id, second.job_id);
  assert.equal(first.nonce, "a1b2001122334455");
  assert.equal(switched.nonce, "a1b2000000000000");
  assert.equal(second.nonce, "a1b27edcba987654");
  assert.equal(first.nonceoffset, 0);
  assert.equal(switched.nonceoffset, 32);
  assert.equal(second.nonceoffset, 0);
  assert.notEqual(first.nonce, second.nonce);
  assert.equal(first.nicehash_mask, "ffff000000000000");
  assert.equal(switched.nicehash_mask, first.nicehash_mask);
  assert.equal(second.nicehash_mask, first.nicehash_mask);
});

test("C29 prefix jobs preserve explicit nonce progress without randomizing", async (t) => {
  let randomCalls = 0;
  t.mock.method(crypto, "randomBytes", () => {
    ++randomCalls;
    return Buffer.alloc(6, 0xff);
  });

  const miner = await loadMinerWithStubs();
  const nonce = "a1b2ffeeddccbbaa";
  const job = miner.getSetJob()({
    algo: "c29", blob_hex: "00".repeat(40), noncebytes: 8, nonceoffset: 0,
    xn: "a1b2", nonce, difficulty: 1, job_id: "saved",
  });

  assert.equal(job.nonce, nonce);
  assert.equal(job.nicehash_mask, "ffff000000000000");
  assert.equal(randomCalls, 0);
});

test("C29 all-prefix nonce jobs do not request a random suffix", async (t) => {
  let randomCalls = 0;
  t.mock.method(crypto, "randomBytes", () => {
    ++randomCalls;
    return Buffer.alloc(1, 0xff);
  });

  const miner = await loadMinerWithStubs();
  const job = miner.getSetJob()({
    algo: "c29", blob_hex: "00".repeat(40), noncebytes: 8, nonceoffset: 0,
    xn: "a1b2c3d4e5f60708", difficulty: 1, job_id: "all-prefix",
  });

  assert.equal(job.nonce, "a1b2c3d4e5f60708");
  assert.equal(job.nicehash_mask, "ffffffffffffffff");
  assert.equal(randomCalls, 0);
});

test("saved progress never moves backwards from a numeric nonce", async () => {
  const miner = await loadMinerWithStubs();
  const job = {algo: "cn/0", blob_hex: "00".repeat(43), nonce: 0x20,
    xn: "00", difficulty: 1, job_id: "numeric"};
  const nativeJob = miner.getSetJob()(job);
  const poolConfig = miner.global.opt.pools[0];
  poolConfig.last_job = job;
  miner.global.opt.pool_ids.active = 1;
  for (const nonce of ["10", "20", "100000000"]) {
    miner.messageHandler({type: "last_nonce", thread_id: 0, value: {
      pool_id: "0", job_id: "numeric", job_token: nativeJob.job_token, nonce,
    }});
    assert.equal(poolConfig.last_job.nonce, 0x20);
  }
});

test("hashrate aggregation drops partial samples when the worker shape changes", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet", "--bench_algo_params", "0",
    ],
    algoParams: {"cn/0": "cpu,cpu"},
  });
  const setJob = miner.getSetJob();
  setJob({algo: "cn/0", blob_hex: "00".repeat(43), difficulty: 1, job_id: "wide"});
  miner.messageHandler({type: "hashrate", thread_id: 0, value: {hashrate: "100"}});

  const params = miner.global.opt.algo_params["cn/0"];
  assert.ok(params);
  params.dev = "cpu";
  setJob({algo: "cn/0", blob_hex: "11".repeat(43), difficulty: 1, job_id: "narrow"});
  miner.messageHandler({type: "hashrate", thread_id: 0, value: {hashrate: "1"}});

  assert.match(miner.loggedMessages.at(-1) || "", /hashrate: 1\.00 H\/s/);
});

test("hashrate aggregation drops partial samples when the live job token changes", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet", "--bench_algo_params", "0",
    ],
    algoParams: {"cn/0": "cpu,cpu"},
  });
  const setJob = miner.getSetJob();
  const first = setJob({algo: "cn/0", blob_hex: "00".repeat(43), difficulty: 1, job_id: "same"});
  assert.ok(first);
  miner.messageHandler({type: "hashrate", thread_id: 0, value: {hashrate: "100"}});
  const logCount = miner.loggedMessages.length;

  const second = setJob({algo: "cn/0", blob_hex: "11".repeat(43), difficulty: 1, job_id: "same"});
  assert.ok(second);
  assert.notEqual(first["job_token"], second["job_token"]);
  miner.messageHandler({type: "hashrate", thread_id: 1, value: {hashrate: "20"}});
  assert.equal(miner.loggedMessages.length, logCount);
  miner.messageHandler({type: "hashrate", thread_id: 0, value: {hashrate: "1"}});

  assert.match(miner.loggedMessages.at(-1) || "", /hashrate: 21\.00 H\/s/);
});

test("pool job identifiers retain their wire type", async () => {
  const miner = await loadMinerWithStubs();
  miner.getSetJob()({
    algo: "cn/0", blob_hex: "00".repeat(43), difficulty: 1, job_id: 17,
  });

  assert.equal(requireJobMessage(miner.sentMessages.at(-1)).job_id, 17);
});

test("live jobs carry fresh tokens even when their wire IDs repeat", async () => {
  const miner = await loadMinerWithStubs();
  /** @type {PoolJob} */
  const first = {algo: "cn/0", blob_hex: "00".repeat(43), difficulty: 1, job_id: "same"};
  /** @type {PoolJob} */
  const second = {algo: "cn/0", blob_hex: "11".repeat(43), difficulty: 1, job_id: "same"};
  miner.getSetJob()(first);
  const firstMessage = requireJobMessage(miner.sentMessages.at(-1));
  miner.getSetJob()(second);
  const secondMessage = requireJobMessage(miner.sentMessages.at(-1));

  assert.match(first.job_token || "", /^\d+$/);
  assert.match(second.job_token || "", /^\d+$/);
  assert.equal(firstMessage["job_token"], first.job_token);
  assert.equal(secondMessage["job_token"], second.job_token);
  assert.notEqual(first.job_token, second.job_token);
});

test("worker result pool writes use numeric pool IDs", async () => {
  const miner = await loadMinerWithStubs();
  const result = {
    pool_id: "0", worker_id: "worker", job_id: "job", job_token: "token",
    nonce: "00000001", hash: "00".repeat(32),
  };
  miner.global.opt.pools[0].last_job = {job_id: "job", job_token: "token"};

  miner.messageHandler({thread_id: 0, type: "result", value: result});

  assert.equal(result.pool_id, "0");
  assert.equal(miner.poolWrites.length, 1);
  assert.equal(miner.poolWrites[0]?.pool_id, 0);
});

test("legacy compact-target jobs reject positive fractional difficulty", async () => {
  const miner = await loadMinerWithStubs();
  assert.throws(() => miner.getSetJob()({
    algo: "cn/0", blob: "00".repeat(43), difficulty: 1.5, job_id: "fractional",
  }), /Invalid cn\/0 job difficulty/);
});

for (const {name, options, expectedDev} of [
  {name: "omitted CPU device", options: [], expectedDev: "cpu*8^4"},
  {name: "explicit CPU device", options: ["--job.dev", "cpu"], expectedDev: "cpu"},
  {name: "explicit one-process CPU device", options: ["--job.dev", "cpu^1"], expectedDev: "cpu"},
  {name: "JSON CPU device", options: ["--job", '{"dev":"cpu"}'], expectedDev: "cpu"},
]) {
  test(`fixed-algorithm mining preserves ${name} selection`, async () => {
    const miner = await loadMinerWithStubs({
      argv: [
        "node", "mom.js", "mine", "pool.example:1", "wallet",
        "--job.algo", "rx/0", "--bench_algo_params", "0", ...options,
      ],
      algoParams: {"rx/0": "cpu*8^4"},
    });
    assert.equal(requireAlgoParam(miner, "rx/0").dev, expectedDev);
    const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
    assert.equal(requirePoolAlgoParam(donation, "rx/0").dev, expectedDev);
    const job = miner.getSetJob()({
      algo: "rx/0", blob: "00".repeat(43), difficulty: 1, job_id: "cpu-selection",
    });
    assert.equal(job.dev, expectedDev);
    assert.equal(Object.hasOwn(job, "dev_request"), false);
  });
}

test("config CPU selection distinguishes omission and honors later CLI overrides", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-device-config-"));
  const configFile = path.join(directory, "config.json");
  const previousLog = helper.log;
  helper.log = noOp;
  try {
    for (const {job, options, expectedDev} of [
      {job: {algo: "rx/0"}, options: [], expectedDev: "cpu*8^4"},
      {job: {algo: "rx/0", dev: "cpu"}, options: [], expectedDev: "cpu"},
      {job: {algo: "rx/0", dev: "cpu"}, options: ["--job.dev", "cpu^2"], expectedDev: "cpu^2"},
    ]) {
      fs.writeFileSync(configFile, JSON.stringify({
        pools: [{url: "pool.example", port: 1, login: "wallet"}],
        pool_ids: {primary: 0, donate: null}, bench_algo_params: 0, job,
      }));
      const miner = await loadMinerWithStubs({
        argv: ["node", "mom.js", "mine", configFile, ...options],
        algoParams: {"rx/0": "cpu*8^4"},
      });
      assert.equal(requireAlgoParam(miner, "rx/0").dev, expectedDev);
    }
  } finally {
    helper.log = previousLog;
    fs.rmSync(directory, {recursive: true, force: true});
  }
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
    requireAlgoParam(miner, "kawpow").dev,
    "gpu1*[intensity=4096;workgroup=128]",
  );
});

test("fixed-algorithm selection narrows the benchmark before it starts", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "kawpow", "--job.dev", "gpu1",
    ],
    algoParams: {
      kawpow: "gpu1*[intensity=1]",
      etchash: "gpu2*[intensity=1]",
    },
    waitForMessageType: "bench",
  });

  assert.deepEqual(Object.keys(miner.global.opt.algo_params), ["kawpow"]);
  assert.equal(requireJobMessage(miner.sentMessages[0]).algo, "kawpow");
  completeOneBenchmark(miner);
  assert.equal(miner.sentMessages.filter((message) => message.type === "bench").length, 1);
});

test("GPU donation reuses an unbenchmarked primary algo without widening its device", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "kawpow", "--job.dev", "gpu1",
      "--bench_algo_params", "0",
      "--new.algo_param.kawpow", JSON.stringify({dev: "gpu1"}),
      "--new.algo_param.etchash", JSON.stringify({dev: "gpu1", perf: 90}),
      // This stale saved entry simulates a memory-incompatible algorithm omitted by discovery.
      "--new.algo_param.c30", JSON.stringify({dev: "gpu1", perf: 200}),
      "--new.algo_param.rx/0", JSON.stringify({dev: "cpu", perf: 10}),
    ],
    algoParams: {
      kawpow: "gpu1*[intensity=1]",
      etchash: "gpu1*[intensity=1]",
      "rx/0": "cpu",
    },
  });

  assert.deepEqual(Object.keys(miner.global.opt.algo_params).sort(), ["etchash", "kawpow"]);
  const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
  assert.equal(donation.url, "mom.moneroocean.stream");
  assert.equal(donation.port, 20001);
  assert.equal(donation.is_tls, true);
  assert.equal(donation.tls_verify, false);
  assert.equal(donation.use_subscribe, false);
  assert.equal(donation.login, "user");
  assert.equal(donation.pass, "mom");
  assert.deepEqual(Object.keys(requirePoolAlgoParams(donation)), ["kawpow"]);
  assert.equal(requirePoolAlgoParam(donation, "kawpow").dev, "gpu1*[intensity=1]");
  assert.equal(requirePoolAlgoParam(donation, "kawpow").perf, 1);
});

test("fixed PearlHash donation reuses discovered GPU geometry and leaves the primary rate unmeasured", async () => {
  const dev = "gpu1*[m=131072;n=524288;k=8192;rank=128]";
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "pearlhash", "--job.dev", dev, "--job.backend", "native",
      "--bench_algo_params", "0",
    ],
    algoParams: {pearlhash: dev},
  });
  const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
  assert.equal(donation.url, "mom.moneroocean.stream");
  assert.equal(donation.port, 20001);
  assert.equal(donation.is_tls, true);
  assert.equal(donation.use_subscribe, false);
  assert.deepEqual(Object.keys(requirePoolAlgoParams(donation)), ["pearlhash"]);
  assert.equal(requirePoolAlgoParam(donation, "pearlhash").dev, dev);
  assert.equal(requirePoolAlgoParam(donation, "pearlhash").perf, 1);
  assert.equal(requireAlgoParam(miner, "pearlhash").perf, null);
  assert.equal(miner.global.opt.job["backend"], "native");
});

for (const algoParams of [{kawpow: "gpu1"}, {pearlhash: "gpu2"}]) {
  test(`fixed PearlHash donation requires discovery on its selected GPU: ${JSON.stringify(algoParams)}`, async () => {
    const miner = await loadMinerWithStubs({
      argv: [
        "node", "mom.js", "mine", "pool.example:1", "wallet",
        "--job.algo", "pearlhash", "--job.dev", "gpu1", "--bench_algo_params", "0",
      ],
      algoParams,
    });
    assert.equal(miner.global.opt.pool_ids.donate, null);
    assert.deepEqual(miner.poolConnects, [0]);
  });
}

test("fixed PearlHash donation retains a compatible measured MO fallback when Pearl was not discovered", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "pearlhash", "--job.dev", "gpu1", "--bench_algo_params", "0",
      "--new.algo_param.kawpow", JSON.stringify({dev: "gpu1", perf: 90}),
    ],
    algoParams: {kawpow: "gpu1*[intensity=1]"},
  });
  const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
  assert.deepEqual(Object.keys(requirePoolAlgoParams(donation)), ["kawpow"]);
  assert.equal(requirePoolAlgoParam(donation, "kawpow").dev, "gpu1*[intensity=1]");
  assert.equal(requirePoolAlgoParam(donation, "kawpow").perf, 90);
  assert.equal(requireAlgoParam(miner, "pearlhash").perf, null);
});

for (const algo of ["autolykos2", "c29", "cn/gpu", "etchash", "kawpow"]) {
  test(`GPU donation uses the unified proxy for ${algo}`, async () => {
    const miner = await loadMinerWithStubs({
      argv: [
        "node", "mom.js", "mine", "pool.example:1", "wallet",
        "--job.algo", algo, "--job.dev", "gpu1", "--bench_algo_params", "0",
      ],
      algoParams: {[algo]: "gpu1"},
    });

    const primary = requirePool(miner, miner.global.opt.pool_ids.primary);
    assert.equal(primary.url, "pool.example");
    assert.equal(primary.port, 1);
    const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
    assert.equal(donation.url, "mom.moneroocean.stream");
    assert.equal(donation.port, 20001);
    assert.equal(donation.login, "user");
    assert.equal(donation.is_tls, true);
    assert.equal(donation.tls_verify, false);
    assert.equal(donation.is_nicehash, false);
    assert.equal(donation.protocol, null);
    assert.equal(donation.use_subscribe, false);
    assert.equal(donation.pass, "mom");
    assert.equal(donation.donation_until, 0);
    assert.equal(requireAlgoParam(miner, algo).dev, "gpu1");
    assert.deepEqual(Object.keys(requirePoolAlgoParams(donation)), [algo]);
    assert.equal(requirePoolAlgoParam(donation, algo).dev, "gpu1");
  });
}

test("donation reconfiguration clears stale endpoint-specific modes", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nicehash-donation-"));
  const configFile = path.join(directory, "config.json");
  fs.writeFileSync(configFile, JSON.stringify({
    pools: [
      {
        url: "nicehash.example", port: 3333, login: "nicehash-wallet", pass: "x",
        is_nicehash: true, is_tls: true, tls_verify: true, use_subscribe: false,
      },
      {url: "pool.example", port: 1, login: "wallet", pass: "x"},
    ],
    pool_ids: {primary: 1, donate: 0},
    bench_algo_params: 0,
    algo_params: {"rx/0": {dev: "cpu", perf: 100}},
  }));
  try {
    const previousLog = helper.log;
    helper.log = noOp;
    let miner;
    try {
      miner = await loadMinerWithStubs({
        argv: ["node", "mom.js", "mine", configFile],
        algoParams: {"rx/0": "cpu"},
      });
    } finally {
      helper.log = previousLog;
    }
    const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
    assert.equal(donation.url, "mom.moneroocean.stream");
    assert.equal(donation.port, 20001);
    assert.equal(donation.is_nicehash, false);
    assert.equal(donation.tls_verify, false);
  } finally {
    fs.rmSync(directory, {recursive: true, force: true});
  }
});

test("explicit CPU selection limits donation to compatible devices and processes", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "argon2/chukwa", "--job.dev", "cpu", "--bench_algo_params", "0",
      "--new.algo_param.rx/0", JSON.stringify({dev: "cpu^6", perf: 100}),
      "--new.algo_param.cn/gpu", JSON.stringify({dev: "gpu1", perf: 1000}),
    ],
    algoParams: {
      "argon2/chukwa": "cpu^4",
      "rx/0": "cpu^6",
      "cn/gpu": "gpu1*[intensity=1]",
    },
  });

  const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
  assert.equal(donation.url, "mom.moneroocean.stream");
  assert.equal(donation.port, 20001);
  assert.equal(donation.is_tls, true);
  assert.equal(donation.tls_verify, false);
  assert.equal(donation.use_subscribe, false);
  assert.equal(donation.login, "user");
  assert.equal(donation.pass, "mom");
  assert.deepEqual(Object.keys(requirePoolAlgoParams(donation)), ["rx/0"]);
  assert.equal(requirePoolAlgoParam(donation, "rx/0").dev, "cpu");
});

test("mixed-device donation preserves the configured CPU and GPU union", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "verthash", "--job.dev", "cpu,gpu1",
      "--bench_algo_params", "0",
      "--new.algo_param.rx/0", JSON.stringify({dev: "cpu", perf: 100}),
      "--new.algo_param.cn/gpu", JSON.stringify({dev: "gpu1", perf: 1000}),
      "--new.algo_param.etchash", JSON.stringify({dev: "gpu2", perf: 2000}),
    ],
    algoParams: {
      verthash: "cpu,gpu1*[intensity=1]",
      "rx/0": "cpu",
      "cn/gpu": "gpu1*[intensity=1]",
      etchash: "gpu2*[intensity=1]",
    },
  });

  const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
  assert.equal(donation.url, "mom.moneroocean.stream");
  assert.deepEqual(Object.keys(requirePoolAlgoParams(donation)).sort(), ["cn/gpu", "rx/0"]);
  assert.equal(requirePoolAlgoParam(donation, "rx/0").dev, "cpu");
  assert.equal(requirePoolAlgoParam(donation, "cn/gpu").dev, "gpu1*[intensity=1]");
});

test("donation discovery disables a saved GPU algo absent from the selected discovery device", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "verthash", "--job.dev", "gpu1", "--bench_algo_params", "0",
      "--new.algo_param.cn/gpu", JSON.stringify({dev: "gpu1", perf: 100}),
    ],
    algoParams: {
      verthash: "gpu1*[intensity=1]",
      "cn/gpu": "gpu2*[intensity=1]",
    },
  });

  assert.equal(miner.global.opt.pool_ids.donate, null);
});

test("donation discovery keeps a primary GPU override out of donation", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "kawpow", "--job.dev", "gpu1", "--bench_algo_params", "0",
      "--new.algo_param.kawpow", JSON.stringify({dev: "gpu2", perf: 100}),
      "--new.algo_param.cn/gpu", JSON.stringify({dev: "gpu1", perf: 10}),
    ],
    algoParams: {
      kawpow: "gpu2*[intensity=1]",
      "cn/gpu": "gpu1*[intensity=1]",
    },
  });

  assert.equal(requireAlgoParam(miner, "kawpow").dev, "gpu1");
  const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
  assert.deepEqual(Object.keys(requirePoolAlgoParams(donation)), ["cn/gpu"]);
  assert.equal(requirePoolAlgoParam(donation, "cn/gpu").dev, "gpu1*[intensity=1]");
  assert.equal(Object.hasOwn(donation.algo_params || {}, "kawpow"), false);
});

test("donation discovery filters mismatched GPU devices during algo switching", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet", "--bench_algo_params", "0",
      "--new.algo_param.cn/gpu", JSON.stringify({dev: "gpu1", perf: 100}),
      "--new.algo_param.rx/0", JSON.stringify({dev: "cpu", perf: 10}),
    ],
    algoParams: {
      "cn/gpu": "gpu2*[intensity=1]",
      "rx/0": "cpu",
    },
  });

  const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
  assert.deepEqual(Object.keys(requirePoolAlgoParams(donation)), ["rx/0"]);
  assert.equal(requirePoolAlgoParam(donation, "rx/0").dev, "cpu");
  assert.equal(Object.hasOwn(donation.algo_params || {}, "cn/gpu"), false);
});

test("donation never exceeds the selected process count on a device", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "verthash", "--job.dev", "gpu1*[intensity=1]^2",
      "--bench_algo_params", "0",
      "--new.algo_param.cn/gpu", JSON.stringify({dev: "gpu1^4", perf: 1000}),
    ],
    algoParams: {
      verthash: "gpu1*[intensity=1]",
      "cn/gpu": "gpu1*[intensity=1]",
    },
  });

  const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
  assert.equal(requirePoolAlgoParam(donation, "cn/gpu").dev, "gpu1*[intensity=1]^2");
});

test("donation excludes an algo absent from device discovery while retaining a fallback", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "verthash", "--job.dev", "gpu1*[intensity=1]", "--bench_algo_params", "0",
      "--new.algo_param.etchash", JSON.stringify({dev: "gpu1*[dag_chunk=1]", perf: 1000}),
      "--new.algo_param.kawpow", JSON.stringify({dev: "gpu1*[intensity=1]", perf: 10}),
    ],
    // Donation selection consumes the same capability-filtered discovery map as normal mining.
    // An unavailable candidate must not be restored merely because it has a measured rate.
    algoParams: {
      verthash: "gpu1*[intensity=1]",
      kawpow: "gpu1*[intensity=1]",
    },
  });

  const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
  assert.deepEqual(Object.keys(requirePoolAlgoParams(donation)), ["kawpow"]);
  assert.equal(requirePoolAlgoParam(donation, "kawpow").dev, "gpu1*[intensity=1]");
  assert.equal(Object.hasOwn(donation.algo_params || {}, "etchash"), false);
});

test("donation fallback preserves all eligible measured algos for proxy profitability selection", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "verthash", "--job.dev", "gpu1*[intensity=1]", "--bench_algo_params", "0",
      "--new.algo_param.etchash", JSON.stringify({dev: "gpu1*[dag_chunk=1]", perf: 10}),
      "--new.algo_param.kawpow", JSON.stringify({dev: "gpu1*[intensity=1]", perf: 1000}),
      "--new.algo_param.cn/gpu", JSON.stringify({dev: "gpu1*[intensity=1]", perf: 500}),
    ],
    algoParams: {
      verthash: "gpu1*[intensity=1]",
      etchash: "gpu1*[dag_chunk=1]",
      kawpow: "gpu1*[intensity=1]",
      "cn/gpu": "gpu2*[intensity=1]",
    },
  });

  const donation = requirePool(miner, miner.global.opt.pool_ids.donate);
  assert.deepEqual(Object.keys(requirePoolAlgoParams(donation)).sort(), ["etchash", "kawpow"]);
  assert.equal(requirePoolAlgoParam(donation, "etchash").dev, "gpu1*[dag_chunk=1]");
  assert.equal(requirePoolAlgoParam(donation, "etchash").perf, 10);
  assert.equal(requirePoolAlgoParam(donation, "kawpow").dev, "gpu1*[intensity=1]");
  assert.equal(requirePoolAlgoParam(donation, "kawpow").perf, 1000);
  assert.equal(Object.hasOwn(donation.algo_params || {}, "cn/gpu"), false);
});

test("an unusable donation pool is removed from runtime failover", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "verthash", "--job.dev", "gpu1*[intensity=1]",
      "--bench_algo_params", "0",
    ],
    algoParams: {verthash: "gpu1*[intensity=1]"},
  });

  assert.equal(miner.global.opt.pool_ids.donate, null);
  assert.equal(miner.global.opt.pool_ids.primary, 0);
  assert.equal(miner.global.opt.pools.length, 1);
  assert.equal(miner.global.opt.pools[0].url, "pool.example");
  assert.deepEqual(miner.poolConnects, [0]);
});

test("a donation window connects once and returns to normal mining on expiry", async () => {
  const miner = await loadMinerWithStubs({
    argv: [
      "node", "mom.js", "mine", "pool.example:1", "wallet",
      "--job.algo", "argon2/chukwa", "--bench_algo_params", "0",
      "--new.algo_param.rx/0", JSON.stringify({dev: "cpu", perf: 100}),
    ],
    algoParams: {"argon2/chukwa": "cpu", "rx/0": "cpu"},
  });
  const donationId = miner.global.opt.pool_ids.donate;
  assert.ok(donationId !== null && donationId !== undefined);
  const donation = requirePool(miner, donationId);
  miner.global.opt.pool_time.donate_length = 0.01;

  miner.startDonationWindow();
  assert.equal(miner.poolConnects.at(-1), donationId);
  assert.ok(typeof donation.donation_until === "number");
  assert.equal(donation.donation_until > 0, true);
  const deadline = Date.now() + 500;
  while (donation.donation_until !== 0 && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.equal(donation.donation_until, 0);
  assert.equal(miner.poolSwitches.at(-1), donationId);
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

  assert.equal(requireJobMessage(miner.sentMessages[0]).algo, "etchash");
  completeOneBenchmark(miner);
  assert.equal(requireJobMessage(miner.sentMessages[1]).algo, "panthera");
  completeOneBenchmark(miner);
  assert.equal(requireJobMessage(miner.sentMessages[2]).algo, "rx/2");
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

  assert.equal(requireJobMessage(miner.sentMessages[0]).algo, "argon2/chukwa");
  completeOneBenchmark(miner);
  assert.equal(requireJobMessage(miner.sentMessages[1]).algo, "argon2/chukwav2");
  completeOneBenchmark(miner);
  assert.equal(requireJobMessage(miner.sentMessages[2]).algo, "cn/0");
  completeOneBenchmark(miner);
  assert.equal(miner.sentMessages.length, 3);
});

test("Intel and AMD cn/gpu benchmarks discard both cold windows on Linux and Windows", async () => {
  /** @type {Array<{platform: NodeJS.Platform, backend: string}>} */
  const cases = [
    {platform: "linux", backend: "intel"},
    {platform: "linux", backend: "amd"},
    {platform: "win32", backend: "intel"},
    {platform: "win32", backend: "amd"},
  ];
  for (const {platform, backend} of cases) {
    const miner = await loadMinerWithStubs({
      platform,
      env: {MOM_GPU_BACKEND: backend},
      algoParams: {
        "cn/gpu": "gpu1*[intensity=768],gpu1*[intensity=768]",
        "rx/2": "cpu",
      },
      waitForMessageType: "bench",
    });

    assert.equal(requireJobMessage(miner.sentMessages[0]).algo, "cn/gpu");
    const completeWorkerPair = (/** @type {number} */ rate) => {
      miner.messageHandler({type: "hashrate", thread_id: 0, value: {hashrate: String(rate / 2)}});
      miner.messageHandler({type: "hashrate", thread_id: 1, value: {hashrate: String(rate / 2)}});
    };
    completeWorkerPair(1000);
    completeWorkerPair(2500);
    assert.equal(miner.sentMessages.length, 1);
    completeWorkerPair(3100);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(requireAlgoParam(miner, "cn/gpu").perf, 3100);
    assert.equal(requireJobMessage(miner.sentMessages[1]).algo, "rx/2");
  }
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
  const finishCandidate = async (/** @type {number} */ rate) => {
    completeOneBenchmark(miner, String(rate));
    completeOneBenchmark(miner, String(rate));
    await new Promise((resolve) => setImmediate(resolve));
  };

  assert.equal(requireJobMessage(miner.sentMessages[0]).dev, "gpu1*[intensity=4096]");
  await finishCandidate(100);
  assert.equal(requireJobMessage(miner.sentMessages[1]).dev, "gpu1*[intensity=2048]");
  await finishCandidate(90);
  assert.equal(requireJobMessage(miner.sentMessages[2]).dev, "gpu1*[intensity=3072]");
  await finishCandidate(105);
  assert.equal(requireJobMessage(miner.sentMessages[3]).dev, "gpu1*[intensity=5120]");
  await finishCandidate(104);

  assert.equal(requireJobMessage(miner.sentMessages[4]).dev, "gpu1*[intensity=3072]");
  completeOneBenchmark(miner, "103");
  assert.equal(requireAlgoParam(miner, "etchash").dev, "gpu1*[intensity=3072]");
  assert.equal(requireAlgoParam(miner, "etchash").perf, 103);
  assert.equal(miner.global.opt.gpu_tune, 0);
});

test("KawPow benchmark jobs include fixed nonce metadata", async () => {
  const autoBenchmark = await loadMinerWithStubs({
    algoParams: {kawpow: "gpu1*[intensity=1]"},
    waitForMessageType: "bench",
  });
  const directBenchmark = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "kawpow"],
    waitForMessageType: "bench",
  });

  for (const miner of [autoBenchmark, directBenchmark]) {
    const benchMessage = miner.sentMessages.find((msg) => msg.type === "bench");
    const benchJob = requireJobMessage(benchMessage);
    assert.equal(benchJob.algo, "kawpow");
    assert.equal(benchJob.noncebytes, 8);
    assert.equal(benchJob.nonceoffset, 32);
  }
});

test("direct test jobs reject invalid nonce metadata before worker startup", async () => {
  for (const {field, value, expected} of [
    {field: "noncebytes", value: 6, expected: /Invalid kawpow nonce size/},
    {field: "nonceoffset", value: 999, expected: /Invalid kawpow nonce offset/},
  ]) {
    await assert.rejects(
      loadMinerWithStubs({
        argv: [
          "node", "mom.js", "test", "kawpow", "expected",
          "--job", JSON.stringify({[field]: value}),
        ],
      }),
      expected,
    );
  }
});

test("BeamHash III benchmark jobs include fixed M4-shaped nonce metadata", async () => {
  const autoBenchmark = await loadMinerWithStubs({
    argv: ["node", "mom.js", "mine", "pool.example:1", "user", "--bench_algo_params", "2"],
    algoParams: {beamhash3: "gpu1"},
    waitForMessageType: "bench",
  });
  const directBenchmark = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "beamhash3"],
    waitForMessageType: "bench",
  });

  for (const miner of [autoBenchmark, directBenchmark]) {
    const benchMessage = miner.sentMessages.find((msg) => msg.type === "bench");
    const benchJob = requireJobMessage(benchMessage);
    assert.equal(benchJob.algo, "beamhash3");
    assert.equal(benchJob.noncebytes, 8);
    assert.equal(benchJob.nonceoffset, 32);
    assert.ok(benchJob.blob_hex);
    assert.equal(benchJob.blob_hex.length, 88);
    assert.equal(benchJob.nonce, "0100000000000000");
    assert.equal(benchJob.nicehash_mask, "0000000000000000");
  }
});

test("C30 benchmark jobs use the official Cortex header and external nonce", async () => {
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "c30"],
    waitForMessageType: "bench",
  });
  const job = requireJobMessage(miner.sentMessages.find((msg) => msg.type === "bench"));
  assert.equal(job.proofsize, 42);
  assert.equal(job.noncebytes, 8);
  assert.equal(job.nonceoffset, 0);
  assert.ok(job.blob_hex);
  assert.equal(job.blob_hex.length, 64);
  assert.equal(job.nonce, "1e000000d90820d4");
});

test("NexaPoW benchmark jobs use the recorded Echelon work format", async () => {
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "nexapow"], waitForMessageType: "bench",
  });
  const job = requireJobMessage(miner.sentMessages.find((msg) => msg.type === "bench"));
  assert.ok(job.blob_hex);
  assert.equal(job.noncebytes, 8);
  assert.equal(job.nonceoffset, 40);
  assert.equal(job.blob_hex.length, 96);
  assert.equal(job.nonce, "1182dc5800000000");
  assert.equal(job.target, "00".repeat(32));
});

test("ZHash-family benchmark jobs use fixed 140-byte headers and nonce metadata", async () => {
  for (const algo of ["zhash", "equihash192_7"]) {
    const miner = await loadMinerWithStubs({
      argv: ["node", "mom.js", "bench", algo], waitForMessageType: "bench",
    });
    const job = requireJobMessage(miner.sentMessages.find((msg) => msg.type === "bench"));
    assert.ok(job.blob_hex);
    assert.equal(job.algo, algo);
    assert.equal(job.noncebytes, 8);
    assert.equal(job.nonceoffset, 108);
    assert.equal(job.blob_hex.length, 280);
    assert.equal(job.blob_hex.slice(216, 232), "0000000000000000");
  }
});

test("performance harness shares the scaled child timeout and teardown margin", () => {
  const previous = process.env["MOM_PERF_SAMPLES"];
  process.env["MOM_PERF_SAMPLES"] = "3";
  try {
    const definition = {name: "fake", job: {algo: "fake"}, timeoutMs: 5 * 60 * 1000};
    assert.equal(benchmarkTimeoutMs(definition), 15 * 60 * 1000);
    assert.equal(benchmarkTestTimeoutMs(definition), 16 * 60 * 1000);
  } finally {
    if (previous === undefined) {
      delete process.env["MOM_PERF_SAMPLES"];
    } else {
      process.env["MOM_PERF_SAMPLES"] = previous;
    }
  }
});

test("benchmark teardown has one owner and fails a stuck graceful shutdown", async () => {
  const previous = process.env["MOM_PERF_SAMPLES"];
  delete process.env["MOM_PERF_SAMPLES"];
  try {
    const clean = benchmarkHarness();
    const cleanRun = clean.execution.runMinerBench({name: "fake", job: {algo: "fake", dev: "gpu1"}});
    const cleanChild = await benchChild(clean);
    cleanChild.stdout.write("Algo fake (gpu1) hashrate: 0 H/s\n");
    assert.equal(clean.timers.some(({delay}) => delay === 30 * 1000), false);
    cleanChild.stdout.write("Algo fake (gpu1) hashrate: 1 H/s\n");
    assert.equal(requireBenchmarkResult(await cleanRun).hashrate, 1);
    assert.equal(clean.timers.filter(({delay}) => delay === 30 * 1000).length, 1);
    assert.equal(requireTimer(clean.timers.find(({delay}) => delay === 30 * 1000)).cleared, true);
    assert.equal(clean.timers.some(({delay}) => delay === 5000), false);

    const stuck = benchmarkHarness(false);
    const stuckRun = stuck.execution.runMinerBench({name: "fake", job: {algo: "fake", dev: "gpu1"}});
    (await benchChild(stuck)).stdout.write("Algo fake (gpu1) hashrate: 1 H/s\n");
    await new Promise((resolve) => setImmediate(resolve));
    const shutdown = requireTimer(stuck.timers.find(({delay}) => delay === 30 * 1000));
    shutdown.callback();
    await assert.rejects(stuckRun, /did not exit after 30000ms/);
    assert.equal(shutdown.cleared, true);

    const timedOut = benchmarkHarness(true, true);
    const timedOutRun = timedOut.execution.runMinerBench({
      name: "fake", gpu: true, job: {algo: "fake", dev: "gpu1"},
    });
    await benchChild(timedOut);
    requireTimer(timedOut.timers[0]).callback();
    await assert.rejects(timedOutRun, /Timed out/);
  } finally {
    if (previous === undefined) {
      delete process.env["MOM_PERF_SAMPLES"];
    } else {
      process.env["MOM_PERF_SAMPLES"] = previous;
    }
  }
});

test("controlled benchmark shutdown lets the outer harness wait for an in-flight GPU dispatch", async () => {
  const stdin = new PassThrough();
  const miner = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "xelishashv3"],
    env: {MOM_BENCHMARK_CONTROL_STDIN: "1"},
    stdin,
    waitForMessageType: "bench",
  });
  stdin.write("close\nclose\n");
  assert.deepEqual(miner.workerCloseDeadlines, [null]);
  assert.equal(miner.process.exitCode, 0);

  const oversizedStdin = new PassThrough();
  const oversized = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "xelishashv3"],
    env: {MOM_BENCHMARK_CONTROL_STDIN: "1"},
    stdin: oversizedStdin,
    waitForMessageType: "bench",
  });
  oversizedStdin.write("x".repeat(65));
  oversizedStdin.write("close\n");
  assert.deepEqual(oversized.workerCloseDeadlines, [null]);
  assert.equal(oversized.process.exitCode, 0);
});

test("benchmark samples require clean exits and isolated sequential children", async () => {
  const previous = process.env["MOM_PERF_SAMPLES"];
  process.env["MOM_PERF_SAMPLES"] = "3";
  try {
    const harness = benchmarkHarness();
    const run = harness.execution.runMinerBench({
      name: "fake", env: {MOM_TEST_MARKER: "yes"}, job: {algo: "fake", dev: "gpu1"},
    });
    for (const [index, rate] of [3, 1, 2].entries()) {
      const child = await benchChild(harness, index);
      assert.equal(harness.children.length, index + 1);
      const env = harness.envs[index];
      assert.ok(env);
      assert.equal(env.MOM_PERF_SAMPLES, undefined);
      assert.equal(env.MOM_TEST_MARKER, "yes");
      child.stdout.write(`Algo fake (gpu1) hashrate: ${rate} H/s\n`);
    }
    const result = requireBenchmarkResult(await run);
    assert.deepEqual(result.samples, [3, 1, 2]);
    assert.equal(result.hashrate, 2);

    delete process.env["MOM_PERF_SAMPLES"];
    /** @type {Array<[number | null, NodeJS.Signals | null, RegExp]>} */
    const failures = [[7, null, /exit 7/], [null, "SIGTERM", /SIGTERM/]];
    for (const [code, signal, expected] of failures) {
      const failed = benchmarkHarness();
      const failedRun = failed.execution.runMinerBench({
        name: "fake", benchSamples: 1, job: {algo: "fake", dev: "gpu1"},
      });
      const child = await benchChild(failed);
      child.stdout.write("Algo fake (gpu1) hashrate: 4 H/s\n");
      child.emit("close", code, signal);
      await assert.rejects(failedRun, expected);
    }
  } finally {
    if (previous === undefined) {
      delete process.env["MOM_PERF_SAMPLES"];
    } else {
      process.env["MOM_PERF_SAMPLES"] = previous;
    }
  }
});

test("benchmark samples latch split compute faults before and after rate selection", async () => {
  const previous = process.env["MOM_PERF_SAMPLES"];
  delete process.env["MOM_PERF_SAMPLES"];
  try {
    for (const position of ["before", "after", "trimmed", "oversized", "oversized-split"]) {
      const harness = benchmarkHarness(false);
      const run = harness.execution.runMinerBench({
        name: "fake", benchSamples: 1, job: {algo: "fake", dev: "gpu1"},
      });
      const rejected = assert.rejects(run, /Compute core error was reported/);
      const child = await benchChild(harness);
      if (position !== "before") {child.stdout.write("Algo fake (gpu1) hashrate: 100 H/s\n");}
      if (position === "oversized") {
        child.stderr.write("ERROR: Compute core error: oversized fixture\n" +
          "x".repeat(1024 * 1024 + 4096));
      } else {
        child.stderr.write("ERROR: Compute core ");
        child.stderr.write("error: controlled fixture\n" +
          (position === "oversized-split" ? "x".repeat(1024 * 1024 + 4096) : ""));
      }
      if (position === "before") {child.stdout.write("Algo fake (gpu1) hashrate: 100 H/s\n");}
      if (position === "trimmed") {child.stderr.write("x".repeat(1024 * 1024 + 4096));}
      child.emit("close", 0, null);
      await rejected;
      assert.ok(harness.timers.every((timer) => timer.cleared));
    }
  } finally {
    if (previous === undefined) {
      delete process.env["MOM_PERF_SAMPLES"];
    } else {
      process.env["MOM_PERF_SAMPLES"] = previous;
    }
  }
});

test("Etchash benchmark uses current ETC height instead of default seed", async () => {
  const autoBenchmark = await loadMinerWithStubs({
    algoParams: {etchash: "gpu1*[intensity=1]"},
    waitForMessageType: "bench",
  });
  const directBenchmark = await loadMinerWithStubs({
    argv: ["node", "mom.js", "bench", "etchash"],
    waitForMessageType: "bench",
  });

  for (const miner of [autoBenchmark, directBenchmark]) {
    const benchMessage = miner.sentMessages.find((msg) => msg.type === "bench");
    const benchJob = requireJobMessage(benchMessage);
    assert.equal(benchJob.algo, "etchash");
    assert.equal(benchJob.height, 24689903);
    assert.equal(benchJob.seed_hex, "");
    assert.equal(benchJob.noncebytes, 8);
    assert.equal(benchJob.nonceoffset, 32);
  }
});

test("pool login does not infer algo from pass when benchmarks are skipped", async () => {
  await withMockPool({
    pool: {pass: "x~kawpow"},
    opt: {
      bench_algo_params: 0,
      job: {algo: null},
      algo_params: {kawpow: {dev: "gpu1*[intensity=1]", perf: null}},
    },
  }, async ({socket, writes}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    const loginParams = requireLoginParams(writes[0]);
    assert.deepEqual(loginParams.algo, []);
    assert.deepEqual(loginParams["algo-perf"], {});
    assert.equal(loginParams.pass, "x~kawpow");
  });
});

test("pool login advertises an explicitly selected unmeasured algorithm", async () => {
  await withMockPool({
    pool: {use_subscribe: false},
    opt: {
      bench_algo_params: 0,
      job: {algo: "pearlhash"},
      algo_params: {pearlhash: {dev: "gpu1*[m=65536]", perf: null}},
    },
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    const loginParams = requireLoginParams(writes[0]);
    assert.deepEqual(loginParams.algo, ["pearlhash"]);
    assert.deepEqual(loginParams["algo-perf"], {});
    assert.deepEqual(poolConfig["requested_algos"], ["pearlhash"]);
    assert.deepEqual(poolConfig["requested_extensions"],
      ["mo-native", "pearl-seed-split", "submit-result"]);
  });
});

test("pool login advertises raw KawPow performance as kawpow1", async () => {
  await withMockPool({
    opt: {
      algo_params: {
        kawpow: {dev: "gpu1*[intensity=37282560]", perf: 20882200},
        c29: {dev: "gpu1", perf: 2.79},
        etchash: {dev: "gpu1*[intensity=33554432]", perf: 21090000},
      },
    },
  }, async ({socket, writes}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    const loginParams = requireLoginParams(writes[0]);
    const algoPerf = loginParams["algo-perf"];
    assert.equal(loginParams.algo.includes("kawpow1"), true);
    assert.equal(loginParams.algo.includes("kawpow"), false);
    assert.equal(algoPerf.kawpow1, 20882200);
    assert.equal("kawpow" in algoPerf, false);
    assert.equal(algoPerf.c29, 2.79 / 42);
    assert.equal(algoPerf.etchash, 21090000);
  });
});

test("PearlHash splits a dotted CLI login for subscribe and authorize", async () => {
  const defaultPool = opts.pool_create("pool.example", 1, false, "prl.wallet.rig", "x");
  assert.equal(defaultPool.worker, "");
  await withMockPool({
    pool: {
      login: defaultPool.login, pass: defaultPool.pass, worker: defaultPool.worker,
      protocol: "pearlhash", use_subscribe: true,
    },
    opt: {job: {algo: "pearlhash"}},
    pool_time: {first_job_wait: 0.001},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");

    assert.ok(writes[0]);
    assert.ok(writes[1]);
    assert.equal(writes[0].method, "mining.subscribe");
    assert.equal(writes[1].method, "mining.authorize");
    assert.deepEqual(writes[1].params, {wallet: "prl.wallet", worker: "rig", pass: "x"});
    poolConfig.last_job = {};
  });
});

test("PearlHash uses the normal worker fallback for an unusable login suffix", async () => {
  for (const login of ["prl-wallet", ".worker", "prl-wallet."]) {
    const defaultPool = opts.pool_create("pool.example", 1, false, login, "x");
    await withMockPool({
      pool: {
        login: defaultPool.login, pass: defaultPool.pass, worker: defaultPool.worker,
        protocol: "pearlhash", use_subscribe: true,
      },
      opt: {job: {algo: "pearlhash"}},
    }, async ({socket, writes, poolConfig}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("connect");
      assert.deepEqual(writes[1]?.params, {wallet: login, worker: "mom", pass: "x"});
      poolConfig.last_job = {};
    });
  }
});

test("PearlHash config worker overrides a dotted login worker", async () => {
  await withMockPool({
    pool: {
      login: "prl.wallet.embedded", worker: "configured", protocol: "pearlhash", use_subscribe: true,
    },
    opt: {job: {algo: "pearlhash"}},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");

    assert.ok(writes[1]);
    assert.deepEqual(writes[1].params, {wallet: "prl.wallet", worker: "configured", pass: "x"});
    poolConfig.last_job = {};
  });
});

test("non-Pearl worker consumers retain their normal default fallback", async () => {
  const defaultPool = opts.pool_create("pool.example", 1, false, "xel:wallet", "x");
  assert.equal(defaultPool.worker, "");
  await withMockPool({
    pool: {
      login: defaultPool.login, pass: defaultPool.pass, worker: defaultPool.worker,
      protocol: "xelis", use_subscribe: true,
    },
    opt: {job: {algo: "xelishashv3"}},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, unexpectedPoolJob);
    socket.emit("connect");
    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0", id: 1, error: null, result: ["session", "22".repeat(32), 32, "33".repeat(32)],
    }) + "\n"));
    assert.deepEqual(writes[1]?.params, ["xel:wallet", "mom", "x"]);
    poolConfig.last_job = {};
  });
});

test("donation pool mines a MoneroOcean algo while the rig is configured for pearlhash", async () => {
  // Donation uses MO's login dialect and advertises only pool-supported algorithms, independent of
  // the external pool's protocol.
  /** @type {MiningJob | null} */
  let donatedJob = null;
  await withMockPool({
    pool: {login: "user", pass: "x", use_subscribe: false}, // MO donate pool opts out of pearlhash subscribe
    opt: {
      bench_algo_params: 0,
      job: {algo: "pearlhash"},
      algo_params: {pearlhash: {dev: "gpu1*[m=131072]", perf: 1}, "rx/0": {dev: "cpu", perf: 1}},
    },
  }, async ({socket, writes}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      const miningJob = completeMiningJob(job);
      donatedJob = miningJob;
      return miningJob;
    });
    socket.emit("connect");
    const login = writes[0];
    assert.ok(login);
    const loginParams = requireLoginParams(login);
    assert.equal(login.method, "login");
    assert.equal(loginParams.algo.includes("rx/0"), true);
    socket.emit("data", Buffer.from(
      '{"jsonrpc":"2.0","id":1,"error":null,"result":{"id":"w","job":' +
      '{"blob":"0101","job_id":"1","target":"c6100000","algo":"rx/0","height":1,"seed_hash":"ab"}}}\n'));
    assert.ok(donatedJob);
    assert.equal(donatedJob.algo, "rx/0");
  });
});

test("direct MO donation login infers and consumes pushed Autolykos2 jobs", async () => {
  /** @type {MiningJob | null} */
  let donatedJob = null;
  const headerHash = "54".repeat(32);
  const bound = "7067388259113537318333190002971674063283542741642755394446115914399301849";
  await withMockPool({
    pool: {
      use_subscribe: false,
      algo_params: {autolykos2: {dev: "gpu1*[intensity=1]", perf: 1}},
    },
    opt: {
      job: {algo: "pearlhash"},
      algo_params: {autolykos2: {dev: "gpu1*[intensity=1]", perf: 1}},
    },
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      const miningJob = completeMiningJob(job);
      donatedJob = miningJob;
      return miningJob;
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.equal(writes[0].method, "login");
    socket.emit("data", Buffer.from([
      {jsonrpc: "2.0", id: 1, error: null, result: {id: "worker", algo: "autolykos2"}},
      {method: "mining.set_difficulty", params: [1], algo: "autolykos2"},
      {
        method: "mining.notify",
        params: ["203d", 614400, headerHash, "", "", 2, bound, "", true],
        algo: "autolykos2",
      },
    ].map((message) => {
      const encoded = JSON.stringify(message);
      if (encoded === undefined) {throw new Error("unable to encode test pool message");}
      return encoded;
    }).join("\n") + "\n"));
    assert.equal(poolConfig.inferred_protocol, "erg");
    assert.equal(poolConfig.last_job?.["submit_mode"], "erg");
    assert.ok(donatedJob);
    assert.equal(donatedJob.algo, "autolykos2");
  });
});

test("login dialect overrides subscribe-capable protocol response routing", async () => {
  /** @type {PoolJob | undefined} */
  let jobMessage;
  const seedHash = "11".repeat(32);
  const headerHash = "22".repeat(32);
  await withMockPool({
    pool: {is_keepalive: true, protocol: "eth", use_subscribe: false},
    opt: {job: {algo: "etchash"}},
  }, async ({socket, writes, poolConfig}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} job */ (job) => {
      jobMessage = job;
      return completeMiningJob(job);
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.equal(writes[0].method, "login");

    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0", id: 1, error: null, result: {id: "worker", algo: "etchash"},
    }) + "\n"));
    assert.equal(poolConfig.logged_in, true);
    assert.notEqual(poolConfig.keepalive, null);
    assert.equal(writes.length, 1);

    socket.emit("data", Buffer.from(JSON.stringify({
      jsonrpc: "2.0", id: 2, error: {message: "keepalive"}, result: false,
    }) + "\n"));
    assert.equal(poolConfig.logged_in, true);
    assert.equal(writes.length, 1);

    socket.emit("data", Buffer.from(JSON.stringify({
      method: "mining.notify", params: ["job", seedHash, headerHash, true], algo: "etchash",
    }) + "\n"));
    assert.ok(jobMessage);
    assert.equal(jobMessage.job_id, "job");
  });
});
