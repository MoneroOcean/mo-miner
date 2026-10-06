#!/usr/bin/env node
"use strict";

const {spawn} = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const {parseArgs} = require("node:util");

const compilerPolicy = require("../compiler-policy");
const gpuTuning = require("../gpu-tuning");
const {
  appendOutputTail, isRecord, maxCapturedOutput, readPerformanceFile, platformColumns, reportedRates,
} = require("./readme-performance");
const {parseLoopStats} = require("./benchmark-gpu-algos");
const {windowsCmdArgs} = require("./windows-command");

const GRACEFUL_CLOSE_TIMEOUT_MS = 10000;
const INTEL_WINDOWS_NEXAPOW_TIMEOUT_MS = 30 * 60 * 1000;
const C30_TIMEOUT_MS = 15 * 60 * 1000;
const STEADY_SAMPLE_COUNT = 3;
const STOCHASTIC_SAMPLE_COUNT = 9;
const stochasticSolutionAlgos = new Set(["beamhash3", "zelhash", "zhash"]);
const cpuAssistedAlgos = new Set(["c29", "c30"]);

/** @typedef {{value: number, displayValue: number, unit: string}} Rate */
/** @typedef {{timestamp: number, wall_seconds: number, cpu_percent: number,
 * dispatch_percent: number, message_seconds: number, post_seconds: number,
 * iterations: number, jobs: number, max_message_ms: number, max_post_ms: number}} LoopStatsSample */
/** @typedef {{platform: string, miner: string, readme: string, margin: number, timeoutMs: number,
 * algoFilter: string | undefined, devOverride: string | undefined}} ReleaseOptions */
/** @typedef {{code: number | null, signal: NodeJS.Signals | null, stdout: string, stderr: string,
 * gracefulClose: boolean, loopStats: LoopStatsSample[], loopStatsInvalid: boolean}} RunResult */
/** @typedef {{selectedRate: Rate | null, minimumRate: number, performanceOk: boolean,
 * cpuFailure: string | null, processOk: boolean, passed: boolean,
 * failure: string | null}} BenchmarkVerdict */
/** @typedef {{tests: number, passed: number, skipped: number}} Summary */
/** @typedef {{stopWhen?: (text: string) => boolean, commandTimeoutMs?: number}} RunOptions */
/** @typedef {(args: string[], options?: RunOptions) => Promise<RunResult>} Runner */

class RunFailure extends Error {
  /** @param {string} message @param {RunResult} result @param {Error} cause */
  constructor(message, result, cause) {
    super(message, {cause});
    this.name = "RunFailure";
    this.result = result;
  }
}

/** @param {{algo: string, reference: Rate | undefined}} row
 * @returns {row is {algo: string, reference: Rate}} */
function hasReference(row) {
  return row.reference !== undefined;
}

/** @param {string} platform @param {string} algo @param {number} timeoutMs
 * @param {number} warmupCount */
function benchmarkTimeout(platform, algo, timeoutMs, warmupCount) {
  let minimum = warmupCount > 1 ? 8 * 60 * 1000 : 0;
  // C30's first rate can take several minutes; later windows arrive about two minutes apart.
  if (algo === "c30") {
    minimum = Math.max(minimum, C30_TIMEOUT_MS);
  }
  // A cold B580 Windows cache can spend over 15 minutes in the device compiler. Later launches
  // reuse that cache and reach steady state normally, so keep the larger deadline case-specific.
  if (platform === "intel-windows" && algo === "nexapow") {
    minimum = Math.max(minimum, INTEL_WINDOWS_NEXAPOW_TIMEOUT_MS);
  }
  const measurementMs = Math.max(timeoutMs, minimum);
  // The miner allows one-time Verthash data generation before its first sample.
  return algo === "verthash" ? measurementMs + 30 * 60 * 1000 : measurementMs;
}

/** @param {string[]} argv @param {string} [hostPlatform] @returns {ReleaseOptions} */
function parseOptions(argv, hostPlatform = process.platform) {
  const {values} = parseArgs({args: argv, options: {
    platform: {type: "string"},
    miner: {type: "string", default: hostPlatform === "win32" ? "mom.cmd" : "mom"},
    readme: {type: "string", default: "README.md"},
    margin: {type: "string", default: "0.05"},
    "timeout-ms": {type: "string", default: "300000"},
    algo: {type: "string"},
    dev: {type: "string"},
  }});
  const platform = values.platform;
  if (typeof platform !== "string" || !platformColumns[platform]) {
    throw new Error(`--platform must be one of: ${Object.keys(platformColumns).join(", ")}`);
  }
  const margin = Number(values.margin);
  if (!Number.isFinite(margin) || margin < 0 || margin >= 1) {
    throw new Error("--margin must be a number between zero and one");
  }
  const timeoutMs = Number(values["timeout-ms"]);
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) {
    throw new Error("--timeout-ms must be a positive integer");
  }
  if (values.dev) {
    try {
      gpuTuning.parseDeviceList(values.dev, values.algo || "");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`--dev ${message}`, {cause: error});
    }
  }
  return {
    platform,
    miner: path.resolve(values.miner),
    readme: path.resolve(values.readme),
    margin,
    timeoutMs,
    algoFilter: values.algo,
    devOverride: values.dev,
  };
}

/** @param {string} file */
function isFile(file) {
  try {return fs.statSync(file).isFile();} catch {return false;}
}

/** @param {import("node:child_process").ChildProcess} child */
function isRunning(child) {
  return child.exitCode === null && child.signalCode === null;
}

/** @param {import("node:child_process").ChildProcess} child */
function hardStop(child) {
  if (!isRunning(child)) {return;}
  if (typeof child.pid !== "number") {
    child.kill("SIGKILL");
    return;
  }
  if (process.platform === "win32") {
    const killer = spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {stdio: "ignore"});
    killer.once("error", () => {if (isRunning(child)) {child.kill("SIGKILL");}});
    killer.once("close", (code) => {
      if (code && isRunning(child)) {
        child.kill("SIGKILL");
      }
    });
    return;
  }
  try {process.kill(-child.pid, "SIGKILL");} catch {child.kill("SIGKILL");}
}

/** @param {import("node:child_process").ChildProcess} child
 * @param {() => void} onForce @returns {NodeJS.Timeout | undefined} */
function stop(child, onForce) {
  if (!isRunning(child)) {return;}
  if (process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] === "1") {
    return requestGracefulClose(child, onForce);
  }
  if (process.platform === "win32") {
    hardStop(child);
    onForce();
    return;
  }
  if (typeof child.pid === "number") {
    try {process.kill(-child.pid, "SIGINT");} catch {child.kill("SIGINT");}
  } else {
    child.kill("SIGINT");
  }
  const timer = setTimeout(() => {
    onForce();
    hardStop(child);
  }, GRACEFUL_CLOSE_TIMEOUT_MS);
  timer.unref();
  return timer;
}

/** @param {import("node:child_process").ChildProcess} child
 * @param {() => void} onForce @returns {NodeJS.Timeout | undefined} */
function requestGracefulClose(child, onForce) {
  if (!isRunning(child)) {return;}
  // Guarded GPU runs must finish native cleanup without an external termination signal.
  const gracefulOnly = process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] === "1";
  if (!child.stdin || !child.stdin.writable) {
    if (!gracefulOnly) {
      onForce();
      hardStop(child);
      return;
    }
  } else {
    child.stdin.end("close\n");
  }
  if (gracefulOnly) {
    const warningTimer = setTimeout(() => {
      if (isRunning(child)) {
        console.error("Release benchmark still waiting for cooperative cleanup.");
      }
    }, GRACEFUL_CLOSE_TIMEOUT_MS);
    warningTimer.unref();
    return warningTimer;
  }
  const timer = setTimeout(() => {
    onForce();
    hardStop(child);
  }, GRACEFUL_CLOSE_TIMEOUT_MS);
  timer.unref();
  return timer;
}

/** @param {string} miner @param {number} defaultTimeoutMs @returns {Runner} */
function createRunner(miner, defaultTimeoutMs) {
  /** @param {string[]} args @returns {[string, string[]]} */
  function releaseCommand(args) {
    if (/\.js$/i.test(miner)) {return [process.execPath, [miner, ...args]];}
    if (process.platform !== "win32" || !/\.cmd$/i.test(miner)) {return [miner, args];}
    // Exercise the packaged launcher itself. Besides selecting mom-node.exe, it supplies the CUDA
    // toolkit and Visual C++ environment required by runtime-compiled ProgPoW kernels.
    return [process.env["ComSpec"] || "cmd.exe", windowsCmdArgs([miner, ...args])];
  }

  /** @param {string[]} args @param {RunOptions} [runOptions] */
  return function run(args, {stopWhen, commandTimeoutMs = defaultTimeoutMs} = {}) {
    const [command, commandArgs] = releaseCommand(args);
    const benchmark = args[0] === "bench";
    const windowsCmd = process.platform === "win32" && /\.cmd$/i.test(miner);
    return new Promise((resolve, reject) => {
      const child = spawn(command, commandArgs, {
        cwd: path.dirname(miner),
        detached: process.platform !== "win32",
        env: {...process.env, MOM_SKIP_MSR: "1",
          ...(benchmark ? {MOM_BENCHMARK_CONTROL_STDIN: "1", MOM_LOOP_STATS: "1"} : {})},
        windowsHide: true,
        windowsVerbatimArguments: windowsCmd,
        stdio: [benchmark ? "pipe" : "ignore", "pipe", "pipe"],
      });
      if (!child.stdout || !child.stderr) {
        child.once("error", () => undefined);
        hardStop(child);
        reject(new Error("release miner did not provide output streams"));
        return;
      }
      let stdout = "";
      let stderr = "";
      let stderrPending = "";
      /** @type {LoopStatsSample[]} */
      const loopStats = [];
      let loopStatsInvalid = false;
      let gracefulCloseRequested = false;
      let forced = false;
      /** @type {NodeJS.Timeout | undefined} */
      let forceTimer;
      /** @type {Error | undefined} */
      let terminationError;
      let settled = false;

      const cleanup = () => {
        clearTimeout(timeout);
        clearTimeout(forceTimer);
        process.off("SIGINT", interrupt);
        process.off("SIGTERM", interrupt);
      };
      /** @param {number | null} code @param {NodeJS.Signals | null} signal */
      const finish = (code, signal) => {
        if (settled) {return;}
        settled = true;
        cleanup();
        if (benchmark && stderrPending) {
          const sample = parseLoopStats(stderrPending);
          if (sample) {loopStats.push(sample);} else if (/\bLOOPSTAT\b/u.test(stderrPending)) {
            loopStatsInvalid = true;
          }
        }
        /** @type {RunResult} */
        const result = {code, signal, stdout, stderr, loopStats, loopStatsInvalid,
          gracefulClose: gracefulCloseRequested && !forced && code === 0 && signal === null};
        if (terminationError) {
          // Preserve setup/runtime diagnostics and a structured result for the private receipt.
          return reject(new RunFailure(
            `${terminationError.message}\n${stdout}\n${stderr}`.trim(), result, terminationError));
        }
        resolve(result);
      };
      /** @param {"stdout" | "stderr"} field @param {Buffer | string} chunk */
      const append = (field, chunk) => {
        const incoming = (field === "stdout" ? stdout : stderr).slice(-32) + chunk.toString();
        if (field === "stdout") {
          stdout = appendOutputTail(stdout, chunk);
        } else {
          stderr = appendOutputTail(stderr, chunk);
          if (benchmark) {
            stderrPending += chunk.toString();
            const boundary = Math.max(stderrPending.lastIndexOf("\n"),
              stderrPending.lastIndexOf("\r"));
            if (boundary >= 0) {
              for (const line of stderrPending.slice(0, boundary + 1).split(/[\r\n]+/u)) {
                const sample = parseLoopStats(line);
                if (sample) {loopStats.push(sample);} else if (/\bLOOPSTAT\b/u.test(line)) {
                  loopStatsInvalid = true;
                }
              }
              stderrPending = stderrPending.slice(boundary + 1);
            }
            if (stderrPending.length > maxCapturedOutput) {
              stderrPending = stderrPending.slice(-maxCapturedOutput);
            }
          }
        }
        const combined = `${stdout}\n${stderr}`;
        // Keep observing faults during cooperative close; the callback freezes its sample window.
        // Inspect faults before tail trimming can discard an oversized chunk's leading diagnostic.
        const observed = /\bCompute core error:/u.test(incoming) ? incoming : combined;
        const shouldStop = stopWhen && stopWhen(observed);
        if (!gracefulCloseRequested && shouldStop) {
          gracefulCloseRequested = true;
          clearTimeout(timeout);
          forceTimer = requestGracefulClose(child, () => { forced = true; });
        }
      };
      const interrupt = () => {
        if (terminationError && (forceTimer || forced)) {return;}
        if (!terminationError) {terminationError = new Error("Release benchmark interrupted");}
        forceTimer = stop(child, () => { forced = true; });
      };
      process.once("SIGINT", interrupt);
      process.once("SIGTERM", interrupt);
      child.stdout.on("data", (chunk) => append("stdout", chunk));
      child.stderr.on("data", (chunk) => append("stderr", chunk));
      if (child.stdin) {
        child.stdin.on("error", (error) => {
          if (!terminationError) {terminationError = error;}
        });
      }
      child.once("error", (error) => {
        if (settled) {return;}
        settled = true;
        cleanup();
        reject(error);
      });
      child.once("close", finish);
      const timeout = setTimeout(() => {
        const gracefulOnly = process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] === "1";
        console.error(
          `Release command deadline reached after ${commandTimeoutMs} ms; ` +
          (gracefulOnly ? "waiting for natural completion." : "stopping the command."),
        );
        terminationError = new Error(`Timed out after ${commandTimeoutMs} ms:`);
        // A deadline disqualifies this run; only its completion boundary may close GPU work.
        if (!gracefulOnly) {forceTimer = stop(child, () => { forced = true; });}
      }, commandTimeoutMs);
    });
  };
}

/** @param {RunResult} result */
function output(result) {
  return `${result.stdout}\n${result.stderr}`;
}

/** @param {Rate[]} rates @returns {Rate | null} */
function lastPositiveRate(rates) {
  let last = null;
  for (const rate of rates) {
    if (rate.value <= 0) {continue;}
    last = rate;
  }
  return last;
}

/** @param {string} text @param {string} algo @returns {Rate | null} */
function lastReportedRate(text, algo) {
  return lastPositiveRate(reportedRates(text, algo));
}

/** Select a robust steady rate while preserving the README unit for display.
 * Solution-count rates are discrete and skewed over short windows, so their longer sample set
 * uses a trimmed mean. Deterministic work rates use the median of three steady samples.
 * @param {Rate[]} rates @param {Rate} reference @param {boolean} stochastic @returns {Rate | null} */
function representativeRate(rates, reference, stochastic) {
  if (!rates.length || !(reference.value > 0) || !(reference.displayValue > 0)) {return null;}
  const sorted = [...rates].sort((left, right) => left.value - right.value);
  const selected = stochastic && sorted.length >= 5 ? sorted.slice(1, -1) : sorted;
  const middle = selected[Math.floor(selected.length / 2)];
  if (!middle) {return null;}
  const value = stochastic
    ? selected.reduce((sum, sample) => sum + sample.value, 0) / selected.length
    : middle.value;
  if (!(value > 0)) {return null;}
  const unitScale = reference.value / reference.displayValue;
  return {value, displayValue: Number((value / unitScale).toFixed(2)), unit: reference.unit};
}

/** @param {string} file @param {string} content */
function writePrivateEvidence(file, content) {
  let existing;
  try {existing = fs.lstatSync(file);} catch (error) {
    if ((/** @type {NodeJS.ErrnoException} */ (error)).code !== "ENOENT") {throw error;}
  }
  if (existing) {
    const error = new Error(existing.isSymbolicLink()
      ? `Evidence file must not be a symlink: ${file}`
      : `Evidence file already exists: ${file}`);
    Object.assign(error, {code: "EEXIST"});
    throw error;
  }
  const temporary = `${file}.tmp-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  let fd;
  /** @type {Error | undefined} */
  let failure;
  try {
    fd = fs.openSync(temporary, "wx", 0o600);
    fs.writeFileSync(fd, content, {encoding: "utf8"});
    fs.fsyncSync(fd);
    fs.closeSync(fd);
    fd = undefined;
    if (process.platform !== "win32") {fs.chmodSync(temporary, 0o600);}
    // Publishing a completed inode with link() is both atomic and exclusive: unlike rename(), it
    // cannot replace evidence another process created after the lstat() diagnostic above.
    fs.linkSync(temporary, file);
  } catch (error) {
    failure = /** @type {Error} */ (error);
  }
  if (fd !== undefined) {
    try {fs.closeSync(fd);} catch (error) {
      if (!failure) {failure = /** @type {Error} */ (error);}
    }
  }
  try {fs.unlinkSync(temporary);} catch (error) {
    if ((/** @type {NodeJS.ErrnoException} */ (error)).code !== "ENOENT" && !failure) {
      failure = /** @type {Error} */ (error);
    }
  }
  if (failure) {
    throw failure;
  }
}

/** @param {string | undefined} root @param {ReleaseOptions} options @param {string} algo
 * @param {RunResult} result @param {string} combined @param {number} startedAt
 * @param {number} endedAt @param {number} warmupCount @param {number} sampleCount
 * @param {number} requiredRates @param {BenchmarkVerdict} verdict */
function saveBenchmarkEvidence(root, options, algo, result, combined, startedAt, endedAt,
  warmupCount, sampleCount, requiredRates, verdict) {
  if (!root) {return;}
  let rootStat;
  try {rootStat = fs.lstatSync(root);} catch (error) {
    if ((/** @type {NodeJS.ErrnoException} */ (error)).code !== "ENOENT") {throw error;}
    fs.mkdirSync(root, {recursive: true, mode: 0o700});
    rootStat = fs.lstatSync(root);
  }
  if (rootStat.isSymbolicLink()) {throw new Error(`Evidence root must not be a symlink: ${root}`);}
  if (!rootStat.isDirectory()) {throw new Error(`Evidence root must be a directory: ${root}`);}
  if (process.platform !== "win32") {fs.chmodSync(root, 0o700);}
  const safeAlgo = encodeURIComponent(algo).replace(/\./g, "%2E");
  const base = path.join(root, safeAlgo);
  const rates = reportedRates(combined, algo);
  const cpuSamples = result.loopStats.map((sample) => ({
    timestamp: sample.timestamp,
    wall_seconds: sample.wall_seconds,
    cpu_percent: sample.cpu_percent,
    dispatch_percent: sample.dispatch_percent,
    message_seconds: sample.message_seconds,
    post_seconds: sample.post_seconds,
    iterations: sample.iterations,
    jobs: sample.jobs,
    max_message_ms: sample.max_message_ms,
    max_post_ms: sample.max_post_ms,
  }));
  writePrivateEvidence(`${base}.metadata.json`, `${JSON.stringify({
    schema: "release-performance-benchmark-v3",
    algo,
    platform: options.platform,
    startedAt: new Date(startedAt).toISOString(),
    endedAt: new Date(endedAt).toISOString(),
    elapsedMs: endedAt - startedAt,
    sampleCount: rates.length,
    positiveSampleCount: rates.filter((rate) => rate.value > 0).length,
    warmupWindows: warmupCount,
    steadySamples: sampleCount,
    requiredSamples: requiredRates,
    rates,
    windowCount: result.loopStats.length,
    cpuSamples,
    loopStatsInvalid: result.loopStatsInvalid,
    exitCode: result.code,
    signal: result.signal,
    gracefulClose: result.gracefulClose,
    stdoutBytes: Buffer.byteLength(result.stdout),
    stderrBytes: Buffer.byteLength(result.stderr),
    verdict,
  }, null, 2)}\n`);
}

/** @param {ReleaseOptions} options @param {((summary: Summary) => void) | null} [onSummary]
 * @returns {Promise<Summary>} */
async function main(options, onSummary = null) {
  const {platform, miner, readme, margin, timeoutMs, algoFilter, devOverride} = options;
  const evidenceRoot = process.env["MOM_RELEASE_PERF_EVIDENCE_DIR"];
  if (!isFile(miner)) {throw new Error(`Release miner is missing or not a file: ${miner}`);}
  if (!isFile(readme)) {throw new Error(`README is missing or not a file: ${readme}`);}
  const run = createRunner(miner, timeoutMs);
  const discovery = await run(["algorithms"]);
  const discoveryOutput = output(discovery);
  if (discovery.code !== 0) {
    throw new Error(`Release device discovery failed:\n${discoveryOutput.trim()}`);
  }
  if (!/^gpu\d+:/m.test(discoveryOutput)) {
    console.log(`  ➖ ${platform}: no GPU is available`);
    const summary = {tests: 1, passed: 0, skipped: 1};
    if (onSummary) {onSummary(summary);}
    return summary;
  }
  const marker = discoveryOutput.split(/\r?\n/)
    .find((line) => line.startsWith("MOM_ALGORITHMS "));
  if (!marker) {throw new Error("Release device discovery omitted MOM_ALGORITHMS");}
  const algoParams = JSON.parse(marker.slice("MOM_ALGORITHMS ".length));
  if (!isRecord(algoParams)) {
    throw new Error("Release device discovery returned invalid MOM_ALGORITHMS");
  }

  const rows = readPerformanceFile(readme)
    .map((row) => ({algo: row.algo, reference: row.performance[platform]}))
    .filter(hasReference)
    .filter((row) => !algoFilter || row.algo === algoFilter);
  if (!rows.length) {throw new Error(`README has no performance rows for ${platform}`);}

  /** @type {string[]} */
  const failures = [];
  /** @type {Summary} */
  const summary = {tests: rows.length, passed: 0, skipped: 0};
  if (onSummary) {onSummary(summary);}
  for (const {algo, reference} of rows) {
    const reported = algoParams[algo];
    if (typeof reported !== "string" || !/\bgpu\d+/i.test(reported)) {
      failures.push(`${algo}: release discovery reported no GPU tuning`);
      console.log(`  ✖ ${algo}: release discovery reported no GPU tuning`);
      continue;
    }
    const selected = compilerPolicy.parseReportedAlgoParam(reported);
    const minimum = reference.value * (1 - margin);
    // cn/gpu needs ten Windows, two Intel/AMD Linux, and one other warm-up window.
    let warmupCount = 1;
    if (algo === "cn/gpu") {
      if (platform.endsWith("-windows")) {
        warmupCount = 10;
      } else if (platform === "intel-linux" || platform === "amd-linux") {
        warmupCount = 2;
      }
    }
    const stochastic = stochasticSolutionAlgos.has(algo);
    const sampleCount = stochastic ? STOCHASTIC_SAMPLE_COUNT : STEADY_SAMPLE_COUNT;
    const requiredRates = warmupCount + sampleCount;
    /** @type {Rate | null} */
    let rate = null;
    /** @type {string | null} */
    let computeFailure = null;
    const benchArgs = ["bench", algo, "--job.dev", devOverride || selected.dev];
    if (algo === "pearlhash") {
      // Match the current network certificate without JSON quoting through cmd.exe.
      benchArgs.push("--job.pearlhash_cert_version", "3");
    }
    if (selected.backend) {benchArgs.push("--job.backend", selected.backend);}
    const benchmarkStartedAt = Date.now();
    /** @type {RunResult} */
    let result;
    try {
      result = await run(benchArgs, {
        commandTimeoutMs: benchmarkTimeout(platform, algo, timeoutMs, warmupCount),
        /** @param {string} text */
        stopWhen(text) {
          if (/\bCompute core error:/u.test(text)) {
            computeFailure = "Compute core error was reported";
            return true;
          }
          if (rate) {return true;}
          const rates = reportedRates(text, algo);
          const positiveRates = rates.filter((sample) => sample.value > 0);
          if (positiveRates.length < requiredRates) {return false;}
          rate = representativeRate(
            positiveRates.slice(warmupCount, requiredRates), reference, stochastic);
          return rate !== null;
        },
      });
    } catch (error) {
      if (error instanceof RunFailure) {
        const failure = error.cause instanceof Error ? error.cause.message : error.message;
        saveBenchmarkEvidence(evidenceRoot, options, algo, error.result, output(error.result),
          benchmarkStartedAt, Date.now(), warmupCount, sampleCount, requiredRates, {
            selectedRate: null,
            minimumRate: minimum,
            performanceOk: false,
            cpuFailure: null,
            processOk: false,
            passed: false,
            failure,
          });
      }
      throw error;
    }
    const benchmarkEndedAt = Date.now();
    const benchmarkOutput = output(result);
    /** @type {string | null} */
    let cpuFailure = null;
    if (!cpuAssistedAlgos.has(algo)) {
      const activeSamples = result.loopStats.filter((sample) => sample.iterations > 0);
      const postSetupSamples = activeSamples.slice(1);
      if (result.loopStatsInvalid) {
        cpuFailure = "CPU telemetry contains invalid LOOPSTAT records";
      } else if (postSetupSamples.length < 2) {
        cpuFailure = "CPU telemetry missing or has fewer than two post-setup active windows";
      } else if (postSetupSamples.some((sample) =>
        !Number.isFinite(sample.cpu_percent) || sample.cpu_percent < 0)) {
        cpuFailure = "CPU telemetry contains an invalid CPU value";
      } else {
        for (let index = 1; index < postSetupSamples.length; index += 1) {
          const previous = postSetupSamples[index - 1];
          const current = postSetupSamples[index];
          if (previous && current && previous.cpu_percent >= 80 && current.cpu_percent >= 80) {
            cpuFailure = `CPU telemetry high pair: ${previous.cpu_percent}% and ` +
              `${current.cpu_percent}% in consecutive post-setup windows`;
            break;
          }
        }
      }
    }
    const sampleWindowComplete = rate !== null;
    if (!rate) {rate = lastReportedRate(output(result), algo);}
    const processOk = result.code === 0 && !result.signal && result.gracefulClose;
    const performanceOk = sampleWindowComplete && !computeFailure &&
      Boolean(rate && rate.value >= minimum);
    const passed = processOk && performanceOk && !cpuFailure;
    saveBenchmarkEvidence(evidenceRoot, options, algo, result, benchmarkOutput,
      benchmarkStartedAt, benchmarkEndedAt, warmupCount, sampleCount, requiredRates, {
        selectedRate: rate,
        minimumRate: minimum,
        performanceOk,
        cpuFailure,
        processOk,
        passed,
        failure: computeFailure || (sampleWindowComplete ? null : "Insufficient hashrate samples"),
      });
    if (!processOk) {
      failures.push(`${algo}: benchmark exited with ${result.signal || `code ${result.code}`}\n` +
        output(result).trim() + (cpuFailure ? `\n${algo}: ${cpuFailure}` : ""));
      console.log(`  ✖ ${algo}: benchmark process failed`);
      continue;
    }
    if (!rate) {
      failures.push(`${algo}: no hashrate was reported\n${output(result).trim()}` +
        (cpuFailure ? `\n${algo}: ${cpuFailure}` : ""));
      console.log(`  ✖ ${algo}: no hashrate was reported`);
      continue;
    }
    const ratio = 100 * rate.value / reference.value;
    const ok = performanceOk && !cpuFailure;
    console.log(`  ${ok ? "✔" : "✖"} ${algo}: ${rate.displayValue} ${rate.unit} ` +
      `(${ratio.toFixed(1)}% of README, minimum ${((1 - margin) * 100).toFixed(0)}%)` +
      (cpuFailure ? `; ${cpuFailure}` : ""));
    if (cpuFailure) {failures.push(`${algo}: ${cpuFailure}`);}
    if (computeFailure) {
      failures.push(`${algo}: ${computeFailure}\n${benchmarkOutput.trim()}`);
    } else if (!sampleWindowComplete) {
      failures.push(`${algo}: insufficient hashrate samples (need ${requiredRates})`);
    } else if (!performanceOk) {
      failures.push(`${algo}: ${rate.value} H/s is below ${minimum} H/s`);
    }
    if (ok) {summary.passed++;}
  }
  if (failures.length) {
    throw new Error(`Release performance regressions:\n${failures.join("\n")}`);
  }
  return summary;
}

if (require.main === module) {
  /** @type {ReleaseOptions | undefined} */
  let options;
  try {
    options = parseOptions(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 2;
  }
  if (options) {
    /** @type {Summary | undefined} */
    let summary;
    main(options, (value) => { summary = value; }).catch((error) => {
      const message = error instanceof Error ? error.stack || error.message : String(error);
      console.error(message);
      process.exitCode = 1;
    }).finally(() => {
      if (summary) {
        console.log(`MOM_TEST_SUMMARY ${summary.tests} ${summary.passed} ` +
        `${summary.tests - summary.passed - summary.skipped} ${summary.skipped}`);
      }
    });
  }
}

module.exports = {benchmarkTimeout, lastReportedRate, main, parseOptions, reportedRates};
