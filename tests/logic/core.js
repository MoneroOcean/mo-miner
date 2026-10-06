"use strict";

const s = require("./support");
const { test, assert, spawnSync, opts, helper, pool, compilerPolicy, formatHashrate, parseFormattedHashrate, specReporter, repoRoot, noOp, loadMinerWithStubs, mockPoolOptions } = s;
const {normalizeAlgoName} = require("../../miner/algorithms");
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
const environmentDeps = {};

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

test("MSR tuning can be disabled for portable deployment tests", () => {
  const environment = require("../../miner/environment")({
    process: {platform: "linux", env: {MOM_SKIP_MSR: "1"}},
  });
  assert.equal(environment.use_msr_tuning(), false);

  const normal = require("../../miner/environment")({
    process: {platform: "linux", env: {}},
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
  const previousOpt = global.opt;
  const previousGuard = process.env.MOM_TEST_NO_POOL_NETWORK;
  global.opt = mockPoolOptions({ pool_time: { connect_throttle: 0 } });
  process.env.MOM_TEST_NO_POOL_NETWORK = "1";
  try {
    assert.throws(
      () => pool.connect_pool_throttle(0, noOp),
      /Pool network access is disabled during mom correctness tests/
    );
  } finally {
    global.opt = previousOpt;
    if (previousGuard === undefined) {delete process.env.MOM_TEST_NO_POOL_NETWORK;}
    else {process.env.MOM_TEST_NO_POOL_NETWORK = previousGuard;}
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

test("JSON dev options reject non-numeric GPU suffixes", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "bench",
    "cn/gpu",
    "--job",
    JSON.stringify({ dev: "gpu1x*1280" }),
    "--pool_time",
    JSON.stringify({ stats: -1 }),
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

test("PearlHash CLI algo params preserve named matrix tuning controls", () => {
  const opt = {};
  opts.set_default_opts(opt, opts.opt_help);
  assert.equal(opts.parse_opt(
    opt,
    opts.opt_help,
    "--new.algo_param.pearlhash",
    JSON.stringify({
      dev: "gpu1*8192",
      backend: "native",
      tuning: {m: 8192, n: 32768, k: 2048, rank: 128},
    }),
  ), true);
  assert.deepEqual(opt.algo_params.pearlhash, {
    dev: "gpu1*8192",
    perf: null,
    backend: "native",
    tuning: {m: 8192, n: 32768, k: 2048, rank: 128},
  });
});

test("PearlHash tuning rejects removed top-level shape fields", () => {
  const result = spawnSync(process.execPath, [
    "mom.js",
    "algo_params",
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

test("algo_params reports requested and resolved GPU backends without changing CPU specs", async () => {
  const miner = await loadMinerWithStubs({env: {MOM_GPU_BACKEND: "amd"}});
  const profile = compilerPolicy.selection("pearlhash", "amd").pearlhashProfile;
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

test("GPU tuning precedence is entry, named object, then automatic heuristic", async () => {
  const miner = await loadMinerWithStubs({env: {MOM_GPU_BACKEND: "nvidia"}});
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

test("Beam layout-only tuning keeps its layout-specific workgroup automatic", async () => {
  const miner = await loadMinerWithStubs({env: {MOM_GPU_BACKEND: "nvidia"}});
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
  assert.equal(message.job.dev, "gpu1*[intensity=4096;workgroup=128]");
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

test("repeated GPU workers may return identical multi-field test results", async () => {
  const miner = await loadMinerWithStubs();
  const result = "final_hash mix_hash";
  assert.equal(miner.matchesTestResult("kawpow", `${result} ${result}`, result), true);
  assert.equal(miner.matchesTestResult("kawpow", `${result} wrong_hash`, result), false);
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
