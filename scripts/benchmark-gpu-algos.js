"use strict";

const {spawn, spawnSync} = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const {parseArgs} = require("node:util");
const compilerPolicy = require("../compiler-policy");
const {
  appendOutputTail, isRecord, maxCapturedOutput, reportedRates,
} = require("./readme-performance");

/** @typedef {{value: number, unit: string, value_per_second: number}} BenchmarkSample */
/** @typedef {{timestamp: number, wall_seconds: number, cpu_percent: number,
 * dispatch_percent: number, message_seconds: number, post_seconds: number,
 * iterations: number, jobs: number, max_message_ms: number, max_post_ms: number}} LoopStatsSample */
/** @typedef {{outputPath: string, label: string, samplesWanted: number, warmupSamples: number,
 * timeoutMs: number, requested: string[], backend: string | undefined,
 * dev?: string | undefined}} BenchmarkOptions */
/** @typedef {{algo: string, status: string, samples: BenchmarkSample[], dev?: string,
 * warmup_samples?: BenchmarkSample[], elapsed_ms?: number, stdout_tail?: string,
 * stderr_tail?: string, loop_stats?: LoopStatsSample[]}} BenchmarkResult */
/** @typedef {{label: string, timestamp: string, platform: string, selector: string, backend: string,
 * samples_wanted: number, warmup_samples: number, timeout_ms: number, jobs: Record<string, string>,
 * results: BenchmarkResult[]}} BenchmarkReport */

const loopStatsPattern = new RegExp([
  String.raw`\bLOOPSTAT t=(\d+) wall=([\d.]+)s cpu=(-?[\d.]+)% `,
  String.raw`dispatch=([\d.]+)% msg=([\d.]+)s post=([\d.]+)s `,
  String.raw`iters=(\d+) jobs=(\d+) max_msg=([\d.]+)ms max_post=([\d.]+)ms\b`,
].join(""), "u");

/** @param {string | number} value @param {string} name @param {number} minimum */
function integerOption(value, name, minimum) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum) {
    throw new Error(`${name} must be an integer of at least ${minimum}`);
  }
  return parsed;
}

/** @param {string[]} argv @param {string} [platform] @param {string} [arch] */
function parseOptions(argv, platform = process.platform, arch = process.arch) {
  const {values} = parseArgs({args: argv, options: {
    output: {type: "string", default: "gpu-benchmark.json"},
    label: {type: "string", default: `${platform}-${arch}`},
    samples: {type: "string", default: "3"},
    "warmup-samples": {type: "string", default: "1"},
    "timeout-ms": {type: "string", default: "150000"},
    algos: {type: "string", default: ""},
    backend: {type: "string", default: ""},
    dev: {type: "string"},
  }});
  const backend = values.backend;
  if (backend) {compilerPolicy.validateBackend(backend);}
  return {
    outputPath: path.resolve(values.output),
    label: values.label,
    samplesWanted: integerOption(values.samples, "--samples", 1),
    warmupSamples: integerOption(values["warmup-samples"], "--warmup-samples", 0),
    timeoutMs: integerOption(values["timeout-ms"], "--timeout-ms", 30000),
    requested: values.algos.split(",").map((value) => value.trim()).filter(Boolean),
    backend,
    dev: values.dev,
  };
}

/** @param {string[]} args */
function minerArgs(args) {
  return ["mom.js", ...args];
}

/** @param {number} timeoutMs @returns {Record<string, string>} */
function discoverGpuJobs(timeoutMs) {
  // Discovery initializes GPU runtimes; graceful-only mode must let it exit naturally.
  const report = spawnSync(process.execPath, minerArgs(["algorithms"]), {
    encoding: "utf8", env: process.env,
    ...(process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] === "1" ? {} : {timeout: timeoutMs}),
  });
  const text = `${report.stdout || ""}\n${report.stderr || ""}`;
  if (report.error) {throw new Error(`algorithms failed: ${report.error.message}`);}
  if (report.status !== 0) {
    throw new Error(`algorithms exited with ${report.signal || `code ${report.status}`}:\n${text}`);
  }
  const match = text.match(/MOM_ALGORITHMS\s+(\{[^\r\n]+\})/);
  if (!match) {throw new Error(`algorithms did not return MOM_ALGORITHMS:\n${text}`);}
  const json = match[1];
  if (!json) {throw new Error(`algorithms did not return MOM_ALGORITHMS:\n${text}`);}
  const params = JSON.parse(json);
  if (!isRecord(params)) {
    throw new Error("algorithms returned an invalid MOM_ALGORITHMS object");
  }
  // The algorithms output includes a human-readable backend annotation. Feed only the underlying device
  // specification back to bench; the miner will resolve and display the backend for that run.
  return Object.fromEntries(Object.entries(params)
    .filter(([, dev]) => typeof dev === "string" && /^gpu\d+/.test(dev))
    .map(([algo, reported]) => [algo, compilerPolicy.parseReportedAlgoParam(reported).dev]));
}

/** @param {import("node:child_process").ChildProcess} child @param {NodeJS.Signals} signal */
function killProcessTree(child, signal) {
  if (typeof child.pid !== "number") {
    child.kill(signal);
    return;
  }
  if (process.platform !== "win32") {
    try {process.kill(-child.pid, signal);} catch {child.kill(signal);}
    return;
  }
  const killer = spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {stdio: "ignore"});
  killer.once("error", () => child.kill(signal));
  killer.once("close", (code) => {
    if (code && child.exitCode === null && child.signalCode === null) {child.kill(signal);}
  });
}

/** @param {string} line @returns {LoopStatsSample | undefined} */
function parseLoopStats(line) {
  const match = line.match(loopStatsPattern);
  if (!match) {return undefined;}
  const sample = {
    timestamp: Number(match[1]), wall_seconds: Number(match[2]), cpu_percent: Number(match[3]),
    dispatch_percent: Number(match[4]), message_seconds: Number(match[5]),
    post_seconds: Number(match[6]), iterations: Number(match[7]), jobs: Number(match[8]),
    max_message_ms: Number(match[9]), max_post_ms: Number(match[10]),
  };
  return Object.values(sample).every(Number.isFinite) ? sample : undefined;
}

/** @param {string} algo @param {string} dev @param {BenchmarkOptions} options
 * @returns {Promise<BenchmarkResult>} */
function benchmark(algo, dev, options) {
  const {backend, samplesWanted, warmupSamples, timeoutMs} = options;
  return new Promise(resolve => {
    // Intel recovery can require a host reboot after an externally signalled compute worker. Its
    // guarded benchmark lane therefore relies only on MoM's control pipe and patient observation.
    const gracefulOnly = process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] === "1";
    const started = Date.now();
    /** @type {BenchmarkSample[]} */
    const samples = [];
    /** @type {BenchmarkSample[]} */
    const observedSamples = [];
    const captureLoopStats = Object.hasOwn(process.env, "MOM_LOOP_STATS");
    /** @type {LoopStatsSample[]} */
    const loopStats = [];
    let stdout = "";
    let stdoutPending = "";
    let stderr = "";
    let stderrPending = "";
    let settled = false;
    let requestedStatus = "";
    let fatalStatus = "";
    /** @type {NodeJS.Timeout | undefined} */
    let forceTimer;
    const jobArgs = ["bench", algo, "--job.dev", dev];
    if (backend) {jobArgs.push("--job.backend", backend);}
    const child = spawn(process.execPath, minerArgs(jobArgs), {
      env: {...process.env, MOM_BENCHMARK_CONTROL_STDIN: "1"},
      stdio: ["pipe", "pipe", "pipe"], windowsHide: true,
      detached: process.platform !== "win32",
    });
    child.stdin.on("error", error => {
      stderr += `\nbenchmark control pipe: ${error.stack || error}`;
      if (!gracefulOnly && requestedStatus && child.exitCode === null && child.signalCode === null) {
        killProcessTree(child, "SIGTERM");
      }
    });

    /** @param {string} status */
    const finish = (status) => {
      if (settled) {return;}
      settled = true;
      clearTimeout(timer);
      clearTimeout(forceTimer);
      process.off("SIGINT", interrupt);
      process.off("SIGTERM", interrupt);
      /** @type {BenchmarkResult} */
      const result = {algo, dev, status,
        warmup_samples: observedSamples.slice(0, warmupSamples), samples,
        elapsed_ms: Date.now() - started,
        stdout_tail: stdout.slice(-2000), stderr_tail: stderr.slice(-2000)};
      if (captureLoopStats) {result.loop_stats = loopStats;}
      resolve(result);
    };
    // Do not start the next algorithm until mom.js has closed and reaped every compute worker. The
    // explicit control pipe works on every OS; Windows emulates SIGTERM by abruptly killing only the
    // parent, which can skip N-API cleanup and briefly orphan an in-flight GPU worker. Retain a
    // bounded hard-kill fallback only for a genuinely stuck miner.
    /** @param {string} status */
    const requestStop = (status) => {
      if (requestedStatus || settled) {return;}
      requestedStatus = status;
      clearTimeout(timer);
      // exitCode/signalCode can become visible just before the "exit" callback. Let that callback
      // classify the real termination instead of racing it with the requested successful status.
      if (child.exitCode !== null || child.signalCode !== null) {return;}
      if (child.stdin.writable) {
        child.stdin.end("close\n");
      } else if (!gracefulOnly) {
        killProcessTree(child, "SIGTERM");
      }
      if (!gracefulOnly) {
        forceTimer = setTimeout(() => killProcessTree(child, "SIGKILL"), 10000);
      }
    };
    /** @param {string} text */
    const detectFatalOutput = (text) => {
      if (!fatalStatus && /ERROR:\s*Compute core error:/i.test(text)) {
        fatalStatus = "compute-error";
        requestStop(fatalStatus);
      }
    };
    // If a systemd/container gate is stopped, keep this parent alive long enough to forward the
    // signal and reap mom.js. Abruptly orphaning an in-flight GPU worker can look like a compiler or
    // driver reset even though only the benchmark harness was interrupted.
    const interrupt = () => requestStop("interrupted");
    process.once("SIGINT", interrupt);
    process.once("SIGTERM", interrupt);
    /** @param {string} text */
    const collectRates = (text) => {
      for (const rate of reportedRates(text, algo)) {
        if (rate.value <= 0) {continue;}
        // Keep the printed pair for reports and one prefix-independent value for comparisons.
        const sample = {
          value: rate.displayValue, unit: rate.unit, value_per_second: rate.value,
        };
        observedSamples.push(sample);
        // A miner can print once more while handling our graceful close. Never let that
        // teardown-time line replace the requested final steady sample in JSON.
        if (observedSamples.length > warmupSamples && samples.length < samplesWanted) {
          samples.push(sample);
        }
      }
      if (samples.length >= samplesWanted) {requestStop("ok");}
    };
    /** @param {Buffer | string} chunk */
    const scan = (chunk) => {
      const text = chunk.toString();
      stdout = appendOutputTail(stdout, text);
      stdoutPending += text;
      detectFatalOutput(stdoutPending);
      process.stdout.write(text);
      const boundary = Math.max(stdoutPending.lastIndexOf("\n"), stdoutPending.lastIndexOf("\r"));
      if (boundary >= 0) {
        collectRates(stdoutPending.slice(0, boundary + 1));
        stdoutPending = stdoutPending.slice(boundary + 1);
      }
      if (stdoutPending.length > maxCapturedOutput) {
        stdoutPending = stdoutPending.slice(-maxCapturedOutput);
        requestStop("oversized-output-line");
      }
    };
    child.stdout.on("data", scan);
    child.stderr.on("data", data => {
      const text = data.toString();
      stderr = appendOutputTail(stderr, text);
      detectFatalOutput(stderr);
      process.stderr.write(data);
      if (!captureLoopStats) {return;}
      stderrPending += text;
      const boundary = Math.max(stderrPending.lastIndexOf("\n"), stderrPending.lastIndexOf("\r"));
      if (boundary >= 0) {
        for (const line of stderrPending.slice(0, boundary + 1).split(/[\r\n]+/u)) {
          const sample = parseLoopStats(line);
          if (sample) {loopStats.push(sample);}
        }
        stderrPending = stderrPending.slice(boundary + 1);
      }
      if (stderrPending.length > maxCapturedOutput) {
        stderrPending = stderrPending.slice(-maxCapturedOutput);
      }
    });
    child.on("error", error => {
      stderr = appendOutputTail(stderr, `\n${error.stack || error}`);
      finish("spawn-error");
    });
    child.on("close", (code, signal) => {
      if (stdoutPending) {collectRates(stdoutPending);}
      if (captureLoopStats && stderrPending) {
        const sample = parseLoopStats(stderrPending);
        if (sample) {loopStats.push(sample);}
      }
      // Accept only the SIGTERM that this harness requested (on Windows Node emulates it by directly
      // terminating the child). A later assertion reports SIGABRT and the stuck-worker fallback
      // reports SIGKILL, so neither can be hidden by an already collected sample/requested status.
      const expectedStop = signal === "SIGTERM" && requestedStatus;
      const abnormal = signal && !expectedStop ? `signal-${signal}` :
        (code !== null && code !== 0 ? `exit-${code}` : "");
      // AdaptiveCpp can print a fatal asynchronous CUDA diagnostic yet return exit code zero after
      // the requested samples. Treat that as a failed lifecycle gate; otherwise a teardown fault is
      // silently recorded as a valid performance result and can destabilize the following GPU run.
      const runtimeDiagnostic = /\[AdaptiveCpp Error\]|cudaErrorCudartUnloading|error code\s*=\s*CUDA:4/i
        .test(stderr) ? "runtime-teardown-error" : "";
      finish(abnormal || runtimeDiagnostic || fatalStatus || requestedStatus ||
        (samples.length >= samplesWanted ? "ok" : "exit-before-samples"));
    });
    const timer = setTimeout(() =>
      requestStop(samples.length >= samplesWanted ? "ok" : "timeout"), timeoutMs);
  });
}

/** @param {BenchmarkOptions} options */
async function main(options) {
  const {outputPath, label, samplesWanted, warmupSamples, timeoutMs, requested, backend, dev} =
    options;
  fs.mkdirSync(path.dirname(outputPath), {recursive: true});
  const jobs = discoverGpuJobs(timeoutMs);
  if (Object.keys(jobs).length === 0) {
    throw new Error("algorithms did not report any GPU jobs");
  }
  const algos = requested.length ? requested : Object.keys(jobs).sort();
  if (dev) {
    if (algos.length !== 1) {
      throw new Error("--dev requires exactly one algorithm");
    }
    const algo = algos[0];
    if (!algo || !jobs[algo]) {
      throw new Error("--dev cannot override an undetected algorithm");
    }
    jobs[algo] = dev;
  }
  /** @type {BenchmarkReport} */
  const report = {
    label, timestamp: new Date().toISOString(), platform: process.platform,
    selector: process.env["ONEAPI_DEVICE_SELECTOR"] || "", backend: backend || "policy",
    samples_wanted: samplesWanted,
    warmup_samples: warmupSamples, timeout_ms: timeoutMs, jobs, results: [],
  };
  for (const algo of algos) {
    if (!jobs[algo]) {
      report.results.push({algo, status: "not-detected", samples: []});
    } else {
      console.log(`\n=== ${label}: ${algo} (${jobs[algo]}) ===`);
      report.results.push(await benchmark(algo, jobs[algo], options));
    }
    fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  }
  const failed = report.results.filter((result) => result.status !== "ok");
  if (failed.length) {
    throw new Error(`${label} failed: ${failed.map(result => `${result.algo}=${result.status}`).join(", ")}`);
  }
  console.log(`\nWrote ${outputPath}`);
}

if (require.main === module) {
  /** @type {BenchmarkOptions | undefined} */
  let options;
  try {
    options = parseOptions(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 2;
  }
  if (options) {
    main(options).catch((error) => {
      const message = error instanceof Error ? error.stack || error.message : String(error);
      console.error(message);
      process.exitCode = 1;
    });
  }
}

module.exports = {main, parseLoopStats, parseOptions};
