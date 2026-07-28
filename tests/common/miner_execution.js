"use strict";

module.exports = ({
  compilerPolicy, getAutoAlgoParams, runNode, formatFailure, emitGitHubError,
  isMissingGpuOutput, spawnMiner, appendOutput, createRunResult, killProcessTree,
  medianHashrate, hashrateUnitMultipliers, escapeRegExp, parseFormattedHashrate,
}) => {
  async function resolveBenchJob(definition) {
    const job = { ...definition.job };
    if (!definition.autoDev) {return { job };}

    const algoParams = await getAutoAlgoParams();
    const reported = algoParams[job.algo];
    if (reported) {
      Object.assign(job, compilerPolicy.parseReportedAlgoParam(reported));
      return { job };
    }

    if (definition.gpu) {
      return { skipped: true, reason: "GPU device is not available in this environment" };
    }

    throw new Error(`No auto device config detected for ${job.algo}`);
  }

  function expectedHash(definition) {
    return Array.isArray(definition.expected) ? definition.expected.join("|") : definition.expected;
  }

  async function maybeDebugRerun(definition, args, result) {
    if (process.platform !== "win32" || process.env.MOM_DEBUG_STARTUP) {return result;}

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

  function assertMinerSuccess(definition, args, result, output) {
    if (minerFailed(result)) {
      const message = formatFailure(`${definition.name} failed`, args, result);
      emitGitHubError(definition.name, message);
      throw new Error(message);
    }
    if (!minerReportedPass(result, output)) {
      const message = formatFailure(`${definition.name} did not report a clean pass`, args, result);
      emitGitHubError(definition.name, message);
      throw new Error(message);
    }
  }

  function minerFailed(result) {
    return result.error || result.code !== 0;
  }

  function minerReportedPass(result, output) {
    return result.stdout.includes("PASSED") && !/\bFAIL(?:ED)?\b/.test(output);
  }

  async function runMinerTest(definition) {
    const job = { ...definition.job };
    const args = [
      "mom.js",
      "test",
      job.algo,
      expectedHash(definition),
      "--job",
      JSON.stringify(job),
    ];
    let result = await runNode(args, { timeoutMs: definition.timeoutMs, env: definition.env });

    if (definition.gpu && isMissingGpuOutput(result)) {
      return { skipped: true, reason: "Requested SYCL device is not available in this environment" };
    }

    if (minerFailed(result)) {
      result = await maybeDebugRerun(definition, args, result);
    }
    assertMinerSuccess(definition, args, result, `${result.stdout}\n${result.stderr}`);

    return { skipped: false };
  }

  async function runMinerBench(definition) {
    const resolved = await resolveBenchJob(definition);
    if (resolved.skipped) {return resolved;}

    const job = resolved.job;
    const args = ["mom.js", "bench", job.algo, "--job", JSON.stringify(job)];
    const sampleCount = benchSampleCount(definition);
    const timeoutMs = definition.timeoutMs || 150 * 1000;
    const unitPattern = Object.keys(hashrateUnitMultipliers).map(escapeRegExp).join("|");
    const hashratePattern = new RegExp(
      `Algo ${escapeRegExp(job.algo)} \\(([^)]*)\\) hashrate: ([0-9.]+)\\s+(${unitPattern})`,
      "g"
    );

    return new Promise((resolve, reject) => {
      const child = spawnMiner(args);
      const result = createRunResult();
      const matchedHashrates = [];
      const matchedDevices = [];
      let stopping = false;

      const stop = () => {
        if (stopping) {return;}
        stopping = true;
        killProcessTree(child, "SIGINT");
        setTimeout(() => killProcessTree(child), 5000).unref();
      };

      const timeout = setTimeout(() => {
        result.error = new Error(`Timed out after ${timeoutMs}ms`);
        stop();
      }, timeoutMs);

      const onData = (streamName, chunk) => {
        appendOutput(result, streamName, chunk);
        const matches = [...`${result.stdout}\n${result.stderr}`.matchAll(hashratePattern)];
        for (const match of matches.slice(matchedHashrates.length)) {
          matchedDevices.push(match[1]);
          matchedHashrates.push(parseFormattedHashrate(match[2], match[3]));
        }
        if (matchedHashrates.length >= sampleCount) {stop();}
      };

      child.stdout.on("data", (chunk) => onData("stdout", chunk));
      child.stderr.on("data", (chunk) => onData("stderr", chunk));
      child.on("error", (error) => {
        result.error = error;
      });
      child.on("close", (code, signal) => {
        clearTimeout(timeout);
        result.code = code;
        result.signal = signal;
        finishBenchRun(
          definition, args, job, result, matchedHashrates, matchedDevices, sampleCount, resolve, reject
        );
      });
    });
  }

  function benchSampleCount(definition) {
    const samples = Number.parseInt(process.env.MOM_PERF_SAMPLES || definition.benchSamples || 1, 10);
    return Number.isFinite(samples) && samples > 0 ? samples : 1;
  }

  function finishBenchRun(
    definition, args, job, result, matchedHashrates, matchedDevices, sampleCount, resolve, reject
  ) {
    if (matchedHashrates.length >= sampleCount && matchedHashrates.every((rate) => rate > 0)) {
      const samples = matchedHashrates.slice(0, sampleCount);
      return resolve({
        hashrate: medianHashrate(samples),
        samples,
        dev: matchedDevices[sampleCount - 1] || job.dev,
        stdout: result.stdout,
        stderr: result.stderr,
      });
    }
    if (definition.gpu && isMissingGpuOutput(result))
    {return resolve({ skipped: true, reason: "GPU device is not available in this environment" });}
    reject(new Error(formatFailure(
      `${definition.name} did not report ${sampleCount} hashrate sample${sampleCount === 1 ? "" : "s"}`,
      args,
      result
    )));
  }

  return { runMinerBench, runMinerTest };
};
