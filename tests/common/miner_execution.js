"use strict";

/** @typedef {{code: number | null, signal: NodeJS.Signals | null, error: Error | null, stdout: string, stderr: string}} RunResult */
/** @typedef {{stdout?: Buffer, stderr?: Buffer}} OutputBuffers */
/** @typedef {RunResult & {[key: symbol]: OutputBuffers | undefined}} BufferedRunResult */
/** @typedef {import("node:events").EventEmitter & {
 * stdin: import("node:stream").Writable | null,
 * stdout: import("node:stream").Readable,
 * stderr: import("node:stream").Readable,
 * }} MinerProcess */
/** @typedef {{skipped: true, reason: string}} SkipResult */
/** @typedef {{skipped: false}} TestResult */
/** @typedef {{hashrate: number, dev: string | undefined, stdout: string, stderr: string}} BenchSample */
/** @typedef {{parseReportedAlgoParam: (reported: string) => Partial<HashJob>}} CompilerPolicy */
/**
 * @template {MinerProcess} TChild
 * @template TTimer
 * @typedef {{compilerPolicy: CompilerPolicy, getAutoAlgoParams: (env: TestEnvironment) => Promise<Record<string, string>>,
 * runNode: (args: string[], options?: {timeoutMs?: number | undefined, env?: TestEnvironment | undefined}) => Promise<BufferedRunResult>,
 * formatFailure: (title: string, args: string[], result: RunResult) => string,
 * emitGitHubError: (title: string, message: string) => void,
 * isMissingGpuOutput: (result: RunResult) => boolean,
 * spawnMiner: (args: string[], env?: TestEnvironment) => TChild,
 * appendOutput: (result: BufferedRunResult, stream: "stdout" | "stderr", chunk: Buffer | string) => void,
 * createRunResult: () => BufferedRunResult, killProcessTree: (child: TChild, signal?: NodeJS.Signals) => boolean,
 * medianHashrate: (samples: number[]) => number | undefined, hashrateUnitMultipliers: Record<string, number>,
 * escapeRegExp: (value: string) => string, parseFormattedHashrate: (value: string, unit: string) => number,
 * setTimeout: (callback: () => void, delay: number) => TTimer,
 * clearTimeout: (timer: TTimer) => void}}
 * ExecutionDependencies<TChild, TTimer>
 */

/** @template {MinerProcess} TChild @template TTimer
 * @param {ExecutionDependencies<TChild, TTimer>} dependencies */
module.exports = ({
  compilerPolicy, getAutoAlgoParams, runNode, formatFailure, emitGitHubError,
  isMissingGpuOutput, spawnMiner, appendOutput, createRunResult, killProcessTree,
  medianHashrate, hashrateUnitMultipliers, escapeRegExp, parseFormattedHashrate,
  setTimeout: schedule, clearTimeout: cancel,
}) => {
  const BENCHMARK_SHUTDOWN_GRACE_MS = 30 * 1000;
  const BENCHMARK_TEST_MARGIN_MS = 60 * 1000;

  /** @param {HashDefinition} definition @returns {Promise<{job: HashJob} | SkipResult>} */
  async function resolveBenchJob(definition) {
    const job = {...definition.job};
    if (!definition.autoDev) {return {job};}

    const algoParams = await getAutoAlgoParams(definition.env || {});
    const reported = algoParams[job.algo];
    if (reported) {
      Object.assign(job, compilerPolicy.parseReportedAlgoParam(reported));
      return {job};
    }

    if (definition.gpu) {
      return {skipped: true, reason: "GPU device is not available in this environment"};
    }

    throw new Error(`No auto device config detected for ${job.algo}`);
  }

  /** @param {HashDefinition} definition */
  function expectedHash(definition) {
    if (definition.expected === undefined) {throw new Error(`Vector ${definition.name} has no expected hash`);}
    return Array.isArray(definition.expected) ? definition.expected.join("|") : definition.expected;
  }

  /** @param {HashDefinition} definition @param {string[]} args @param {BufferedRunResult} result */
  async function maybeDebugRerun(definition, args, result) {
    if (process.platform !== "win32" || process.env["MOM_DEBUG_STARTUP"] ||
        result.error?.message?.startsWith("Timed out after")) {return result;}

    const debugResult = await runNode(args, {
      timeoutMs: definition.timeoutMs,
      env: {...definition.env, MOM_DEBUG_STARTUP: "1"},
    });
    return {
      ...result,
      stderr: [
        result.stderr,
        "Debug rerun:",
        formatFailure(`${definition.name} debug rerun`, args, debugResult),
      ].filter(Boolean).join("\n"),
    };
  }

  /** @param {HashDefinition} definition @param {string[]} args @param {BufferedRunResult} result @param {string} output */
  function assertMinerSuccess(definition, args, result, output) {
    if (minerFailed(result)) {
      const message = formatFailure(`${definition.name} failed`, args, result);
      emitGitHubError(definition.name, message);
      throw new Error(message);
    }
    if (!minerReportedPass(definition, result, output)) {
      const message = formatFailure(`${definition.name} did not report a clean pass`, args, result);
      emitGitHubError(definition.name, message);
      throw new Error(message);
    }
  }

  /** @param {RunResult} result */
  function minerFailed(result) {
    return result.error || result.code !== 0;
  }

  /** @param {HashDefinition} definition @param {RunResult} result @param {string} output */
  function minerReportedPass(definition, result, output) {
    if (!result.stdout.includes("PASSED") || /\bFAIL(?:ED)?\b/.test(output)) {return false;}
    const backend = definition.job.backend;
    if (definition.job.algo !== "pearlhash" || !definition.gpu ||
        ![undefined, "auto", "native", "sycl-native"].includes(backend)) {return true;}
    const lines = output.split(/\r?\n/);
    return lines.includes("PEARLHASH_TEST search_checksum_match=true") ||
      (lines.includes("PEARLHASH_TEST reference_search=true") &&
       lines.some(line => /^PEARLHASH_TEST search=sycl cert_version=\d+ host_seeds_match=true$/.test(line)));
  }

  /** @param {HashDefinition} definition @returns {Promise<TestResult | SkipResult>} */
  async function runMinerTest(definition) {
    const job = {...definition.job};
    const args = [
      "mom.js",
      "test",
      job.algo,
      expectedHash(definition),
      "--job",
      JSON.stringify(job),
    ];
    let result = await runNode(args, {timeoutMs: definition.timeoutMs, env: definition.env});

    if (!result.error && definition.gpu && isMissingGpuOutput(result)) {
      return {skipped: true, reason: "Requested SYCL device is not available in this environment"};
    }

    if (minerFailed(result)) {
      result = await maybeDebugRerun(definition, args, result);
    }
    assertMinerSuccess(definition, args, result, `${result.stdout}\n${result.stderr}`);

    return {skipped: false};
  }

  /** @param {HashDefinition} definition @returns {Promise<{hashrate: number, samples: number[], dev: string | undefined, stdout: string, stderr: string} | SkipResult>} */
  async function runMinerBench(definition) {
    const resolved = await resolveBenchJob(definition);
    if ("skipped" in resolved) {return resolved;}

    const job = resolved.job;
    const args = ["mom.js", "bench", job.algo, "--job", JSON.stringify(job)];
    const sampleCount = benchSampleCount(definition);
    const unitPattern = Object.keys(hashrateUnitMultipliers).map(escapeRegExp).join("|");
    const hashratePattern = new RegExp(
      `Algo ${escapeRegExp(job.algo)} \\(([^)]*)\\) hashrate: ([0-9.]+)\\s+(${unitPattern})`, "g"
    );
    const samples = [];
    /** @type {BufferedRunResult} */
    const outputs = {stdout: "", stderr: "", code: null, signal: null, error: null};
    let dev = job.dev;

    for (let i = 0; i < sampleCount; i++) {
      const sample = await runBenchSample(definition, args, hashratePattern);
      if ("skipped" in sample) {return sample;}
      samples.push(sample.hashrate);
      dev = sample.dev || dev;
      appendOutput(outputs, "stdout", sample.stdout);
      appendOutput(outputs, "stderr", sample.stderr);
    }
    const hashrate = medianHashrate(samples);
    if (hashrate === undefined) {throw new Error("Benchmark produced no hashrate samples");}
    return {hashrate, samples, dev, stdout: outputs.stdout, stderr: outputs.stderr};
  }

  /** @param {HashDefinition} definition @param {string[]} args @param {RegExp} hashratePattern @returns {Promise<BenchSample | SkipResult>} */
  function runBenchSample(definition, args, hashratePattern) {
    return new Promise((resolve, reject) => {
      const child = spawnMiner(args, {
        ...definition.env,
        MOM_PERF_SAMPLES: undefined,
        MOM_BENCHMARK_CONTROL_STDIN: "1",
      });
      const result = createRunResult();
      /** @type {number | undefined} */
      let hashrate;
      /** @type {string | undefined} */
      let dev;
      let stopping = false;
      let settled = false;
      /** @type {ReturnType<typeof schedule> | null} */
      let shutdownTimer = null;

      const stop = () => {
        if (stopping) {return;}
        stopping = true;
        shutdownTimer = schedule(() => {
          result.error = result.error || new Error(
            `Benchmark child did not exit after ${BENCHMARK_SHUTDOWN_GRACE_MS}ms`
          );
          result.signal = result.signal || "SIGKILL";
          killProcessTree(child);
          child.stdout.destroy();
          child.stderr.destroy();
          finish();
        }, BENCHMARK_SHUTDOWN_GRACE_MS);
        if (child.stdin?.writable) {
          child.stdin.write("close\n");
        } else {
          killProcessTree(child, "SIGINT");
        }
      };

      const finish = () => {
        if (settled) {return;}
        settled = true;
        cancel(timeout);
        if (shutdownTimer) {cancel(shutdownTimer);}
        if (!result.error && result.code === 0 && !result.signal &&
            typeof hashrate === "number" && hashrate > 0) {
          return resolve({hashrate, dev, stdout: result.stdout, stderr: result.stderr});
        }
        if (!result.error && definition.gpu && isMissingGpuOutput(result)) {
          return resolve({skipped: true, reason: "GPU device is not available in this environment"});
        }
        reject(new Error(formatFailure(
          `${definition.name} did not report a clean hashrate sample`, args, result
        )));
      };

      const timeout = schedule(() => {
        result.error = new Error(`Timed out after ${benchSampleTimeoutMs(definition)}ms`);
        stop();
      }, benchSampleTimeoutMs(definition));

      /** @param {"stdout" | "stderr"} streamName @param {Buffer | string} chunk */
      const onData = (streamName, chunk) => {
        const incoming = result[streamName].slice(-32) + chunk.toString();
        appendOutput(result, streamName, chunk);
        if (/\bCompute core error:/u.test(incoming)) {
          result.error = result.error || new Error("Compute core error was reported");
          stop();
          return;
        }
        if (hashrate) {return;}
        const matches = [...`${result.stdout}\n${result.stderr}`.matchAll(hashratePattern)];
        const match = matches.at(-1);
        if (!match) {return;}
        const matchDev = match[1];
        const matchValue = match[2];
        const matchUnit = match[3];
        if (!matchDev || !matchValue || !matchUnit) {return;}
        const rate = parseFormattedHashrate(matchValue, matchUnit);
        if (!Number.isFinite(rate) || rate <= 0) {return;}
        dev = matchDev;
        hashrate = rate;
        stop();
      };

      child.stdout.on("data", (chunk) => onData("stdout", chunk));
      child.stderr.on("data", (chunk) => onData("stderr", chunk));
      child.stdin?.on("error", (error) => { if (!stopping) {result.error = error;} });
      child.on("error", (error) => {
        result.error = error;
      });
      child.on("close", (code, signal) => {
        if (settled) {return;}
        result.code = code;
        result.signal = signal;
        finish();
      });
    });
  }

  /** @param {HashDefinition} definition */
  function benchSampleCount(definition) {
    const samples = Number.parseInt(String(process.env["MOM_PERF_SAMPLES"] || definition.benchSamples || 1), 10);
    return Number.isFinite(samples) && samples > 0 ? samples : 1;
  }

  /** @param {HashDefinition} definition */
  function benchSampleTimeoutMs(definition) {
    return definition.timeoutMs || 150 * 1000;
  }

  /** @param {HashDefinition} definition */
  function benchmarkTimeoutMs(definition) {
    return benchSampleCount(definition) * benchSampleTimeoutMs(definition);
  }

  /** @param {HashDefinition} definition */
  function benchmarkTestTimeoutMs(definition) {
    return benchmarkTimeoutMs(definition) + BENCHMARK_TEST_MARGIN_MS;
  }

  return {benchmarkTestTimeoutMs, benchmarkTimeoutMs, runMinerBench, runMinerTest};
};
