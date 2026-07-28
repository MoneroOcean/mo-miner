"use strict";

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const compilerPolicy = require("../../compiler-policy");

const repoRoot = path.join(__dirname, "..", "..");
const releaseExecutableNames = process.platform === "win32"
  ? ["mom.exe", "mom.cmd"]
  : ["mom"];
const releaseExecutable = releaseExecutableNames
  .map((name) => path.join(repoRoot, name))
  .find((filePath) => fs.existsSync(filePath)) || path.join(repoRoot, releaseExecutableNames[0]);
const hasReleaseExecutable = fs.existsSync(releaseExecutable);
let autoAlgoParamsPromise = null;
let autoAlgoParamsReportPromise = null;

const hashrateUnits = [
  { value: 1000000000000000, suffix: "PH/s" },   // pearlhash reports GEMM throughput in TH/s+
  { value: 1000000000000, suffix: "TH/s" },
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
  if (!Number.isFinite(rate)) {return String(hashrate);}
  for (const unit of hashrateUnits) {
    if (Math.abs(rate) >= unit.value) {return `${(rate / unit.value).toFixed(2)} ${unit.suffix}`;}
  }
  return `${rate.toFixed(2)} H/s`;
}

function parseFormattedHashrate(value, unit) {
  const rate = Number.parseFloat(value);
  const multiplier = hashrateUnitMultipliers[unit];
  return Number.isFinite(rate) && multiplier ? rate * multiplier : Number.NaN;
}

function medianHashrate(samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function quoteWindowsCmdArg(arg) {
  if (arg.length === 0) {return '""';}
  if (!/[\s"&|<>()^%]/.test(arg)) {return arg;}
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
  if (!process.env.GITHUB_ACTIONS) {return;}

  const escape = (value) => value
    .replace(/%/g, "%25")
    .replace(/\r/g, "%0D")
    .replace(/\n/g, "%0A");
  process.stderr.write(`::error title=${escape(title)}::${escape(message)}\n`);
}

function resolveReleaseCommand(args) {
  if (!/\.cmd$/i.test(releaseExecutable)) {return [releaseExecutable, ...args.slice(1)];}

  const packageDir = path.dirname(releaseExecutable);
  const nodeExe = path.join(packageDir, "mom-node.exe");
  const bundle = path.join(packageDir, "mom.bundle.cjs");
  if (fs.existsSync(nodeExe) && fs.existsSync(bundle)) {return [nodeExe, bundle, ...args.slice(1)];}
  return wrapWindowsCmd([releaseExecutable, ...args.slice(1)]);
}

function resolveMinerCommand(args) {
  if (hasReleaseExecutable && args[0] === "mom.js") {return resolveReleaseCommand(args);}
  return [process.execPath, ...args];
}

function spawnAndExit(command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: options.cwd || repoRoot,
    env: options.env ? { ...process.env, ...options.env } : process.env,
    stdio: "inherit",
  });

  child.on("exit", (code, signal) => {
    if (signal) {process.kill(process.pid, signal);}
    process.exit(code === null ? 1 : code);
  });

  child.on("error", (error) => {
    console.error(error.message);
    process.exit(1);
  });
}

function isInsideRsh() {
  return process.env.MOM_R_SH === "1" || fs.existsSync("/.dockerenv");
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
  if (shouldUseDirectNode()) {return { command: process.execPath, args: testArgs, env };}

  if (fs.existsSync(path.join(repoRoot, "r.sh"))) {return resolveRshRunner(testArgs, env);}

  return { command: "./docker-mom.sh", args: ["node", ...testArgs], env };
}

function isMissingGpuOutput(result) {
  const output = `${result.stdout}\n${result.stderr}`;
  // A run that reported a clean pass clearly found its device; do not let
  // diagnostic stderr (e.g. a SYCL runtime buffer/info notice) misclassify it.
  if (result.stdout.includes("PASSED")) {return false;}
  // Preserve actionable compiler/JIT/crash diagnostics. Some of them mention a SYCL device in
  // their surrounding worker error, which the availability pattern below must not turn into a
  // misleading "device unavailable" skip.
  if (/\[AdaptiveCpp Error\]|Code object construction failed|Worker \d+ exited unexpectedly|LLVM ERROR|fatal error/i
    .test(output)) {return false;}
  if (result.code === 0 && result.stdout.trim() === "" && result.stderr.trim() === "") {return true;}
  return /Unknown compute platform gpu|No device of requested type|No GPU|gpu[0-9]+.*not found|SYCL.*device/i.test(output);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function childEnv(extra = {}) {
  const env = { ...process.env, ...extra };
  if (process.platform !== "win32") {return env;}
  return withWindowsTestPath(env);
}

function releasePathEntry(entry) {
  return hasReleaseExecutable ? entry : null;
}

function withWindowsTestPath(env) {
  return withWindowsPathEntries(env, [
    env.MOM_NATIVE_PATH ? path.dirname(env.MOM_NATIVE_PATH) : null,
    releasePathEntry(path.join(path.dirname(releaseExecutable), "libs")),
    releasePathEntry(path.dirname(releaseExecutable)),
    path.join(repoRoot, "build", "win", "Release"),
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
    if (key.toLowerCase() === "path" && key !== pathKey) {delete env[key];}
  }
  return pathKey;
}

function killProcessTree(child, signal = "SIGKILL") {
  if (process.platform !== "win32" || !child.pid) {
    child.kill(signal);
    return false;
  }
  const killer = spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
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

function spawnMiner(args, env, cwd = repoRoot) {
  if (!hasReleaseExecutable && args[0] === "mom.js") {
    args = [path.join(repoRoot, "mom.js"), ...args.slice(1)];
  }
  const command = resolveMinerCommand(args);
  return spawn(command[0], command.slice(1), {
    cwd,
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
    const child = spawnMiner(args, options.env, options.cwd);
    const result = createRunResult();
    let settled = false;
    let forceResolveTimeout = null;

    const finish = () => {
      if (settled) {return;}
      settled = true;
      clearTimeout(timeout);
      clearTimeout(forceResolveTimeout);
      resolve(result);
    };

    const timeout = setTimeout(() => {
      if (settled) {return;}
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

function detectAlgoParams(env) {
  const args = ["tests/common/print_algo_params.js"];
  return runNode(args, {timeoutMs: 60 * 1000, env}).then((result) => {
    if (result.error || result.code !== 0) {
      throw new Error(formatFailure("Unable to detect algo params", args, result));
    }

    const line = result.stdout.trim().split(/\r?\n/).reverse()
      .find((entry) => entry.startsWith("MOM_ALGO_PARAMS "));
    if (!line) {
      throw new Error(formatFailure("Algo params output did not contain JSON marker", args, result));
    }
    return {
      params: JSON.parse(line.slice("MOM_ALGO_PARAMS ".length)),
      stdout: result.stdout,
      stderr: result.stderr,
    };
  });
}

async function getAutoAlgoParamsReport(env) {
  if (env) {return detectAlgoParams(env);}
  if (!autoAlgoParamsReportPromise) {autoAlgoParamsReportPromise = detectAlgoParams();}
  return autoAlgoParamsReportPromise;
}

function parseSyclCpuDevices(output) {
  const devices = [];
  for (const line of output.split(/\r?\n/)) {
    const match = line.match(/^(cpu\d+):\s+(.+)$/);
    if (match) {devices.push({ dev: match[1], description: match[2] });}
  }
  return devices;
}

function parseGpuDevices(output, integrated = null) {
  const devices = new Map();
  for (const line of output.split(/\r?\n/)) {
    const match = line.match(/^(gpu\d+):\s+(.+)$/);
    if (!match) {continue;}
    const isIntegrated = /\s\[integrated\]$/i.test(match[2]);
    if (integrated !== null && isIntegrated !== integrated) {continue;}
    devices.set(match[1], {dev: match[1], description: match[2], integrated: isIntegrated});
  }
  return [...devices.values()];
}

function parseDiscreteGpuDevices(output) {
  return parseGpuDevices(output, false).map(({dev, description}) => ({dev, description}));
}

async function getGpuDevices(vendor, options = {}) {
  const baseEnv = {
    MOM_GPU_BACKEND: vendor,
    ...(options.env || {}),
  };
  const selectedEnv = compilerPolicy.workerEnv(
    options.algo || "etchash",
    {...process.env, ...baseEnv},
    process.platform,
    options.backend || "auto"
  );
  let report;
  try {
    report = await getAutoAlgoParamsReport({...baseEnv, ...selectedEnv});
  } catch (error) {
    return {skipped: true, reason: `${vendor} GPU discovery failed: ${error.message}`};
  }
  const integrated = Object.hasOwn(options, "integrated") ? options.integrated : false;
  const devices = parseGpuDevices(`${report.stdout}\n${report.stderr}`, integrated);
  if (!devices.length) {
    const kind = integrated === null ? "" : integrated ? " integrated" : " discrete";
    return {skipped: true, reason: `No${kind} ${vendor} GPU is available in this environment`};
  }
  return {skipped: false, devices, params: report.params};
}

function syclCpuUnavailable(message) {
  if (process.env.GITHUB_ACTIONS || process.env.MOM_REQUIRE_PORTABLE_CPU_TESTS === "1") {
    emitGitHubError("SYCL CPU device unavailable", message);
    throw new Error(message);
  }

  return {
    skipped: true,
    reason: message,
  };
}

function syclCpuDetectionFailure(error) {
  if (process.env.GITHUB_ACTIONS || process.env.MOM_REQUIRE_PORTABLE_CPU_TESTS === "1") {
    emitGitHubError("SYCL CPU device unavailable", error.message);
    throw error;
  }
  return {
    skipped: true,
    reason: `SYCL CPU device detection failed: ${error.message}`,
  };
}

async function getFirstSyclCpuDevice(env) {
  const assumedDevice = assumedSyclCpuDevice();
  if (assumedDevice) {return assumedDevice;}

  let report;
  try {
    report = await getAutoAlgoParamsReport(env);
  } catch (error) {
    return syclCpuDetectionFailure(error);
  }

  const output = `${report.stdout}\n${report.stderr}`;
  const devices = parseSyclCpuDevices(output);
  if (devices.length) {return { skipped: false, ...devices[0] };}

  const message = [
    "No SYCL CPU device was reported by algo_params output.",
    formatOutput("stdout", report.stdout),
    formatOutput("stderr", report.stderr),
  ].join("\n");
  return syclCpuUnavailable(missingSyclCpuMessage(message));
}

function missingSyclCpuMessage(reportMessage) {
  if (process.env.GITHUB_ACTIONS) {return reportMessage;}
  return "SYCL CPU device is not available in this environment";
}

function assumedSyclCpuDevice() {
  if (!process.env.MOM_ASSUME_SYCL_CPU) {return null;}
  return {
    skipped: false,
    dev: process.env.MOM_ASSUME_SYCL_CPU,
    description: "configured by MOM_ASSUME_SYCL_CPU",
  };
}

const execution = require("./miner_execution")({
  compilerPolicy, getAutoAlgoParams, runNode, formatFailure, emitGitHubError,
  isMissingGpuOutput, spawnMiner, appendOutput, createRunResult, killProcessTree,
  medianHashrate, hashrateUnitMultipliers, escapeRegExp, parseFormattedHashrate,
});

module.exports = {
  formatHashrate,
  getGpuDevices,
  getFirstSyclCpuDevice,
  parseFormattedHashrate,
  parseReportedAlgoParam: compilerPolicy.parseReportedAlgoParam,
  parseDiscreteGpuDevices,
  parseGpuDevices,
  repoRoot,
  resolveMinerCommand,
  resolveNodeRunner,
  runNode,
  spawnAndExit,
  ...execution,
};
