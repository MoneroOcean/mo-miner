"use strict";

const childProcess = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const { spawn } = childProcess;

const repoRoot = path.join(__dirname, "..", "..");
const releaseExecutableNames = process.platform === "win32"
  ? ["mo-miner.exe", "mo-miner.cmd"]
  : ["mo-miner"];
const releaseExecutable = releaseExecutableNames
  .map((name) => path.join(repoRoot, name))
  .find((filePath) => fs.existsSync(filePath)) || path.join(repoRoot, releaseExecutableNames[0]);
const hasReleaseExecutable = fs.existsSync(releaseExecutable);
let autoAlgoParamsPromise = null;
let autoAlgoParamsReportPromise = null;

const hashrateUnits = [
  { value: 1000000000, suffix: "GH/s" },
  { value: 1000000, suffix: "MH/s" },
  { value: 1000, suffix: "KH/s" },
];
const hashrateUnitMultipliers = Object.fromEntries([
  ...hashrateUnits.map((unit) => [unit.suffix, unit.value]),
  ["H/s", 1],
]);

function quoteCommand(args) {
  return args
    .map((arg) => (/^[A-Za-z0-9_./:=+-]+$/.test(arg) ? arg : JSON.stringify(arg)))
    .join(" ");
}

function formatHashrate(hashrate) {
  const rate = Number.parseFloat(hashrate);
  if (!Number.isFinite(rate)) return String(hashrate);
  for (const unit of hashrateUnits) {
    if (Math.abs(rate) >= unit.value) return (rate / unit.value).toFixed(2) + " " + unit.suffix;
  }
  return rate.toFixed(2) + " H/s";
}

function parseFormattedHashrate(value, unit) {
  const rate = Number.parseFloat(value);
  const multiplier = hashrateUnitMultipliers[unit];
  return Number.isFinite(rate) && multiplier ? rate * multiplier : Number.NaN;
}

function medianHashrate(samples) {
  const sorted = samples.slice().sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function quoteWindowsCmdArg(arg) {
  if (arg.length === 0) return '""';
  if (!/[\s"&|<>()^%]/.test(arg)) return arg;
  return `"${arg.replace(/"/g, '""')}"`;
}

function wrapWindowsCmd(args) {
  return [
    process.env.ComSpec || "cmd.exe",
    ["/d", "/s", "/c", args.map(quoteWindowsCmdArg).join(" ")],
  ];
}

function formatOutput(label, text) {
  return text ? `\n${label}:\n${text.trimEnd()}` : `\n${label}: <empty>`;
}

function formatFailure(title, args, result) {
  const exitStatus = result.error
    ? `error: ${result.error.message}`
    : `exit: ${result.code}${result.signal ? ` signal: ${result.signal}` : ""}`;

  return [
    title,
    `$ ${quoteCommand(resolveMinerCommand(args))}`,
    exitStatus,
    formatOutput("stdout", result.stdout),
    formatOutput("stderr", result.stderr),
  ].join("\n");
}

function emitGitHubError(title, message) {
  if (!process.env.GITHUB_ACTIONS) return;

  const escape = (value) => value
    .replace(/%/g, "%25")
    .replace(/\r/g, "%0D")
    .replace(/\n/g, "%0A");
  process.stderr.write(`::error title=${escape(title)}::${escape(message)}\n`);
}

function resolveReleaseCommand(args) {
  if (!/\.cmd$/i.test(releaseExecutable)) return [releaseExecutable, ...args.slice(1)];

  const packageDir = path.dirname(releaseExecutable);
  const nodeExe = path.join(packageDir, "mo-miner-node.exe");
  const bundle = path.join(packageDir, "mo-miner.bundle.cjs");
  if (fs.existsSync(nodeExe) && fs.existsSync(bundle)) return [nodeExe, bundle, ...args.slice(1)];
  return wrapWindowsCmd([releaseExecutable, ...args.slice(1)]);
}

function resolveMinerCommand(args) {
  if (hasReleaseExecutable && args[0] === "mo-miner.js") return resolveReleaseCommand(args);
  return [process.execPath, ...args];
}

function spawnAndExit(command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: options.cwd || repoRoot,
    env: options.env ? { ...process.env, ...options.env } : process.env,
    stdio: "inherit",
  });

  child.on("exit", (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    process.exit(code === null ? 1 : code);
  });

  child.on("error", (error) => {
    console.error(error.message);
    process.exit(1);
  });
}

function isInsideRsh() {
  return process.env.MOMINER_R_SH === "1" || fs.existsSync("/.dockerenv");
}

function shouldUseDirectNode() {
  return process.platform === "win32" || isInsideRsh() || hasReleaseExecutable;
}

function resolveRshRunner(testArgs, env) {
  const envArgs = Object.entries(env).map(([key, value]) => `${key}=${value}`);
  const args = envArgs.length ? ["env", ...envArgs, "node", ...testArgs] : ["node", ...testArgs];
  return { command: "./r.sh", args };
}

function resolveNodeRunner(testArgs, env = {}) {
  if (shouldUseDirectNode()) return { command: process.execPath, args: testArgs, env };

  if (fs.existsSync(path.join(repoRoot, "r.sh"))) return resolveRshRunner(testArgs, env);

  return { command: "./docker-mo-miner.sh", args: ["node", ...testArgs], env };
}

function isMissingGpuOutput(result) {
  const output = `${result.stdout}\n${result.stderr}`;
  // A run that reported a clean pass clearly found its device; do not let
  // diagnostic stderr (e.g. AdaptiveCpp's SYCL-buffer notice) misclassify it.
  if (result.stdout.includes("PASSED")) return false;
  if (result.code === 0 && result.stdout.trim() === "" && result.stderr.trim() === "") return true;
  return /Unknown compute platform gpu|No device of requested type|No GPU|gpu[0-9]+.*not found|SYCL.*device/i.test(output);
}

function withTestDevice(definition) {
  return { ...definition.job };
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function childEnv(extra = {}) {
  const env = { ...process.env, ...extra };
  if (process.platform !== "win32") return env;
  return withWindowsTestPath(env);
}

function releasePathEntry(entry) {
  return hasReleaseExecutable ? entry : null;
}

function withWindowsTestPath(env) {
  return withWindowsPathEntries(env, [
    releasePathEntry(path.join(path.dirname(releaseExecutable), "libs")),
    releasePathEntry(path.dirname(releaseExecutable)),
    path.join(repoRoot, "build", "Release"),
  ]);
}

function withWindowsPathEntries(env, entries) {
  const pathKey = normalizeWindowsPathKey(env);
  const pathValue = env[pathKey] || "";
  env[pathKey] = [...entries, pathValue].filter(Boolean).join(path.delimiter);
  return env;
}

function normalizeWindowsPathKey(env) {
  const pathKey = Object.keys(env).find((key) => key.toLowerCase() === "path") || "Path";
  for (const key of Object.keys(env)) {
    if (key.toLowerCase() === "path" && key !== pathKey) delete env[key];
  }
  return pathKey;
}

function killProcessTree(child, signal = "SIGKILL") {
  if (process.platform !== "win32" || !child.pid) {
    child.kill(signal);
    return false;
  }
  const killer = childProcess.spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
    stdio: "ignore",
  });
  killer.on("error", () => child.kill(signal));
  return true;
}

function detachChild(child) {
  child.stdout.destroy();
  child.stderr.destroy();
  child.unref();
}

function createRunResult() {
  return {
    code: null,
    signal: null,
    error: null,
    stdout: "",
    stderr: "",
  };
}

function spawnMiner(args, env) {
  const command = resolveMinerCommand(args);
  return spawn(command[0], command.slice(1), {
    cwd: repoRoot,
    env: childEnv(env),
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function appendOutput(result, streamName, chunk) {
  result[streamName] += chunk.toString("utf8");
}

function runNode(args, options = {}) {
  const timeoutMs = options.timeoutMs || 5 * 60 * 1000;

  return new Promise((resolve) => {
    const child = spawnMiner(args, options.env);
    const result = createRunResult();
    let settled = false;
    let forceResolveTimeout = null;

    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      clearTimeout(forceResolveTimeout);
      resolve(result);
    };

    const timeout = setTimeout(() => {
      if (settled) return;
      result.error = new Error(`Timed out after ${timeoutMs}ms`);
      killProcessTree(child);
      forceResolveTimeout = setTimeout(() => {
        result.signal = result.signal || "SIGKILL";
        detachChild(child);
        finish();
      }, 10 * 1000);
    }, timeoutMs);

    child.stdout.on("data", (chunk) => appendOutput(result, "stdout", chunk));
    child.stderr.on("data", (chunk) => appendOutput(result, "stderr", chunk));
    child.on("error", (error) => {
      result.error = error;
    });
    child.on("close", (code, signal) => {
      result.code = code;
      result.signal = signal;
      finish();
    });
  });
}

async function getAutoAlgoParams() {
  if (!autoAlgoParamsPromise) {
    autoAlgoParamsPromise = getAutoAlgoParamsReport().then((report) => report.params);
  }

  return autoAlgoParamsPromise;
}

async function getAutoAlgoParamsReport() {
  if (!autoAlgoParamsReportPromise) {
    const args = ["tests/common/print_algo_params.js"];
    autoAlgoParamsReportPromise = runNode(args, { timeoutMs: 60 * 1000 })
      .then((result) => {
        if (result.error || result.code !== 0) {
          throw new Error(formatFailure("Unable to detect algo params", args, result));
        }

        const line = result.stdout
          .trim()
          .split(/\r?\n/)
          .reverse()
          .find((entry) => entry.startsWith("MOMINER_ALGO_PARAMS "));

        if (!line) {
          throw new Error(formatFailure("Algo params output did not contain JSON marker", args, result));
        }

        return {
          params: JSON.parse(line.slice("MOMINER_ALGO_PARAMS ".length)),
          stdout: result.stdout,
          stderr: result.stderr,
        };
      });
  }

  return autoAlgoParamsReportPromise;
}

function parseSyclCpuDevices(output) {
  const devices = [];
  for (const line of output.split(/\r?\n/)) {
    const match = line.match(/^(cpu\d+):\s+(.+)$/);
    if (match) devices.push({ dev: match[1], description: match[2] });
  }
  return devices;
}

function syclCpuUnavailable(message) {
  if (process.env.GITHUB_ACTIONS) {
    emitGitHubError("SYCL CPU device unavailable", message);
    throw new Error(message);
  }

  return {
    skipped: true,
    reason: message,
  };
}

function syclCpuDetectionFailure(error) {
  if (process.env.GITHUB_ACTIONS) {
    emitGitHubError("SYCL CPU device unavailable", error.message);
    throw error;
  }
  return {
    skipped: true,
    reason: `SYCL CPU device detection failed: ${error.message}`,
  };
}

async function getFirstSyclCpuDevice() {
  const assumedDevice = assumedSyclCpuDevice();
  if (assumedDevice) return assumedDevice;

  let report;
  try {
    report = await getAutoAlgoParamsReport();
  } catch (error) {
    return syclCpuDetectionFailure(error);
  }

  const output = `${report.stdout}\n${report.stderr}`;
  const devices = parseSyclCpuDevices(output);
  if (devices.length) return { skipped: false, ...devices[0] };

  const message = [
    "No SYCL CPU device was reported by algo_params output.",
    formatOutput("stdout", report.stdout),
    formatOutput("stderr", report.stderr),
  ].join("\n");
  return syclCpuUnavailable(missingSyclCpuMessage(message));
}

function missingSyclCpuMessage(reportMessage) {
  if (process.env.GITHUB_ACTIONS) return reportMessage;
  return "SYCL CPU device is not available in this environment";
}

function assumedSyclCpuDevice() {
  if (!process.env.MOMINER_ASSUME_SYCL_CPU) return null;
  return {
    skipped: false,
    dev: process.env.MOMINER_ASSUME_SYCL_CPU,
    description: "configured by MOMINER_ASSUME_SYCL_CPU",
  };
}

async function resolveBenchJob(definition) {
  const job = withTestDevice(definition);
  if (!definition.autoDev) return { job };

  const algoParams = await getAutoAlgoParams();
  const dev = algoParams[job.algo];
  if (dev) {
    job.dev = dev;
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
  if (process.platform !== "win32" || process.env.MOMINER_DEBUG_STARTUP) return result;

  const debugResult = await runNode(args, {
    timeoutMs: definition.timeoutMs,
    env: { MOMINER_DEBUG_STARTUP: "1" },
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
  const job = withTestDevice(definition);
  const args = [
    "mo-miner.js",
    "test",
    job.algo,
    expectedHash(definition),
    "--job",
    JSON.stringify(job),
  ];
  let result = await runNode(args, { timeoutMs: definition.timeoutMs });

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
  if (resolved.skipped) return resolved;

  const job = resolved.job;
  const args = ["mo-miner.js", "bench", job.algo, "--job", JSON.stringify(job)];
  const sampleCount = benchSampleCount(definition);
  const timeoutMs = definition.timeoutMs || 150 * 1000;
  const unitPattern = Object.keys(hashrateUnitMultipliers).map(escapeRegExp).join("|");
  const hashratePattern = new RegExp(`Algo ${escapeRegExp(job.algo)} \\([^)]*\\) hashrate: ([0-9.]+)\\s+(${unitPattern})`, "g");

  return new Promise((resolve, reject) => {
    const child = spawnMiner(args);
    const result = createRunResult();
    const matchedHashrates = [];
    let stopping = false;

    const stop = () => {
      if (stopping) return;
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
        matchedHashrates.push(parseFormattedHashrate(match[1], match[2]));
      }
      if (matchedHashrates.length >= sampleCount) stop();
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
      finishBenchRun(definition, args, job, result, matchedHashrates, sampleCount, resolve, reject);
    });
  });
}

function benchSampleCount(definition) {
  const samples = Number.parseInt(process.env.MOMINER_PERF_SAMPLES || definition.benchSamples || 1, 10);
  return Number.isFinite(samples) && samples > 0 ? samples : 1;
}

function finishBenchRun(definition, args, job, result, matchedHashrates, sampleCount, resolve, reject) {
  if (matchedHashrates.length >= sampleCount && matchedHashrates.every(isPositiveHashrate)) {
    const samples = matchedHashrates.slice(0, sampleCount);
    return resolve({ hashrate: medianHashrate(samples), samples, dev: job.dev });
  }
  if (isMissingBenchGpu(definition, result))
    return resolve({ skipped: true, reason: "GPU device is not available in this environment" });
  reject(new Error(formatFailure(
    `${definition.name} did not report ${sampleCount} hashrate sample${sampleCount === 1 ? "" : "s"}`,
    args,
    result
  )));
}

function isPositiveHashrate(hashrate) {
  return hashrate && hashrate > 0;
}

function isMissingBenchGpu(definition, result) {
  return definition.gpu && isMissingGpuOutput(result);
}

module.exports = {
  formatHashrate,
  getFirstSyclCpuDevice,
  hasReleaseExecutable,
  parseFormattedHashrate,
  repoRoot,
  resolveMinerCommand,
  resolveNodeRunner,
  runMinerBench,
  runMinerTest,
  spawnAndExit,
};
