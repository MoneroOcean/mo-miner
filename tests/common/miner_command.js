"use strict";

const {spawn} = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const compilerPolicy = require("../../compiler-policy");
const {windowsCmdArgs} = require("../../scripts/windows-command");

/** @typedef {{cwd?: string, env?: TestEnvironment | undefined, timeoutMs?: number | undefined}} RunOptions */
/** @typedef {{code: number | null, signal: NodeJS.Signals | null, error: Error | null, stdout: string, stderr: string}} RunResult */
/** @typedef {{stdout?: Buffer, stderr?: Buffer}} OutputBuffers */
/** @typedef {RunResult & {[key: symbol]: OutputBuffers | undefined}} BufferedRunResult */
/** @typedef {import("node:child_process").ChildProcessByStdio<import("node:stream").Writable | null, import("node:stream").Readable, import("node:stream").Readable>} SpawnedMiner */
/** @typedef {{dev: string, description: string, integrated: boolean}} GpuDevice */
/** @typedef {{skipped: true, reason: string}} SkippedResult */
/** @typedef {{skipped: false, devices: GpuDevice[], params: Record<string, string>}} GpuDeviceResult */
/** @typedef {SkippedResult | GpuDeviceResult} GpuDiscoveryResult */
/** @typedef {{skipped: false, dev: string, description: string} | SkippedResult} CpuDeviceResult */
/** @typedef {{command: string, args: string[], env?: TestEnvironment,
 * windowsVerbatimArguments?: boolean}} Runner */
/** @typedef {"stdout" | "stderr"} OutputStream */
/** @typedef {{params: Record<string, string>, stdout: string, stderr: string}} AlgoParamsReport */

const repoRoot = path.join(__dirname, "..", "..");
const releaseExecutableNames = process.platform === "win32"
  ? ["mom.exe", "mom.cmd"]
  : ["mom"];
const defaultReleaseExecutable = process.platform === "win32" ? "mom.exe" : "mom";
const releaseExecutable = releaseExecutableNames
  .map((name) => path.join(repoRoot, name))
  .find((filePath) => fs.existsSync(filePath)) || path.join(repoRoot, defaultReleaseExecutable);
const hasReleaseExecutable = fs.existsSync(releaseExecutable);
const ALGO_PARAMS_TIMEOUT_MS = 5 * 60 * 1000;
const MAX_CAPTURED_OUTPUT_BYTES = 1024 * 1024;
const outputBuffers = Symbol("outputBuffers");

/** @param {TestEnvironment} [env] */
function normalizeAlgoParamsEnv(env = {}) {
  /** @type {Map<string, string>} */
  const values = new Map();
  /** @param {string} rawKey @returns {string} */
  const normalizeKey = (rawKey) => process.platform === "win32" ? rawKey.toUpperCase() : rawKey;
  for (const [rawKey, rawValue] of Object.entries(process.env)) {
    const key = normalizeKey(rawKey);
    values.set(key, rawValue == null ? "" : String(rawValue));
  }
  for (const [rawKey, rawValue] of Object.entries(env)) {
    const key = normalizeKey(rawKey);
    if (rawValue === undefined) {
      values.delete(key);
      continue;
    }
    values.set(key, rawValue == null ? "" : String(rawValue));
  }
  return JSON.stringify([...values.entries()].sort(([left], [right]) => left.localeCompare(right)));
}

/** @param {(env: TestEnvironment) => Promise<AlgoParamsReport>} probe @returns {(env: TestEnvironment) => Promise<AlgoParamsReport>} */
function createAlgoParamsReportCache(probe) {
  const cache = new Map();
  return (env) => {
    const key = normalizeAlgoParamsEnv(env);
    const cached = cache.get(key);
    if (cached) {return cached;}
    const report = probe(env);
    cache.set(key, report);
    report.catch(() => {
      if (cache.get(key) === report) {cache.delete(key);}
    });
    return report;
  };
}

const hashrateUnits = [
  {value: 1000000000000000, suffix: "PH/s"},   // pearlhash reports GEMM throughput in TH/s+
  {value: 1000000000000, suffix: "TH/s"},
  {value: 1000000000, suffix: "GH/s"},
  {value: 1000000, suffix: "MH/s"},
  {value: 1000, suffix: "KH/s"},
];
const hashrateUnitMultipliers = Object.fromEntries([
  ...hashrateUnits.map((unit) => [unit.suffix, unit.value]),
  ["H/s", 1],
]);

/** @param {Runner} runner */
function quoteCommand(runner) {
  return [runner.command, ...runner.args]
    .map((arg) => (/^[A-Za-z0-9_./:=+-]+$/.test(arg) ? arg : JSON.stringify(arg)))
    .join(" ");
}

/** @param {string | number} hashrate */
function formatHashrate(hashrate) {
  const rate = Number.parseFloat(String(hashrate));
  if (!Number.isFinite(rate)) {return String(hashrate);}
  for (const unit of hashrateUnits) {
    if (Math.abs(rate) >= unit.value) {return `${(rate / unit.value).toFixed(2)} ${unit.suffix}`;}
  }
  return `${rate.toFixed(2)} H/s`;
}

/** @param {string} value @param {string} unit */
function parseFormattedHashrate(value, unit) {
  const rate = Number.parseFloat(value);
  const multiplier = hashrateUnitMultipliers[unit];
  return Number.isFinite(rate) && multiplier ? rate * multiplier : Number.NaN;
}

/** @param {number[]} samples */
function medianHashrate(samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/** @param {string[]} args @returns {Runner} */
function wrapWindowsCmd(args) {
  return {
    command: process.env["ComSpec"] || "cmd.exe",
    args: windowsCmdArgs(args),
    windowsVerbatimArguments: true,
  };
}

/** @param {string} label @param {string} text */
function formatOutput(label, text) {
  return text ? `\n${label}:\n${text.trimEnd()}` : `\n${label}: <empty>`;
}

/** @param {string} title @param {string[]} args @param {RunResult} result */
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

/** @param {string} title @param {string} message */
function emitGitHubError(title, message) {
  if (!process.env["GITHUB_ACTIONS"]) {return;}

  /** @param {string} value */
  const escape = (value) => value
    .replace(/%/g, "%25")
    .replace(/\r/g, "%0D")
    .replace(/\n/g, "%0A");
  process.stderr.write(`::error title=${escape(title)}::${escape(message)}\n`);
}

/** @param {string[]} args @returns {Runner} */
function resolveReleaseCommand(args) {
  if (!/\.cmd$/i.test(releaseExecutable)) {
    return {command: releaseExecutable, args: args.slice(1)};
  }

  const packageDir = path.dirname(releaseExecutable);
  const nodeExe = path.join(packageDir, "mom-node.exe");
  const bundle = path.join(packageDir, "mom.bundle.cjs");
  if (fs.existsSync(nodeExe) && fs.existsSync(bundle)) {
    return {command: nodeExe, args: [bundle, ...args.slice(1)]};
  }
  return wrapWindowsCmd([releaseExecutable, ...args.slice(1)]);
}

/** @param {string[]} args @returns {Runner} */
function resolveMinerCommand(args) {
  if (hasReleaseExecutable && args[0] === "mom.js") {return resolveReleaseCommand(args);}
  return {command: process.execPath, args};
}

/** @param {string} command @param {string[]} args @param {RunOptions} [options] */
function spawnAndExit(command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: options.cwd || repoRoot,
    env: options.env ? {...process.env, ...options.env} : process.env,
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
  return process.env["MOM_R_SH"] === "1" || fs.existsSync("/.dockerenv");
}

function shouldUseDirectNode() {
  return process.platform === "win32" || isInsideRsh() || hasReleaseExecutable;
}

/** @param {string[]} testArgs @param {TestEnvironment} env @returns {Runner} */
function resolveRshRunner(testArgs, env) {
  const entries = Object.entries(env);
  const unsetArgs = entries.filter(([, value]) => value === undefined)
    .flatMap(([key]) => ["-u", key]);
  const valueArgs = entries.filter((entry) => entry[1] !== undefined)
    .map(([key, value]) => `${key}=${value}`);
  const envArgs = [...unsetArgs, ...valueArgs];
  const args = envArgs.length ? ["env", ...envArgs, "node", ...testArgs] : ["node", ...testArgs];
  return {command: "./r.sh", args};
}

/** @param {string[]} testArgs @param {TestEnvironment} [env] @returns {Runner} */
function resolveNodeRunner(testArgs, env = {}) {
  if (shouldUseDirectNode()) {return {command: process.execPath, args: testArgs, env};}

  if (fs.existsSync(path.join(repoRoot, "r.sh"))) {return resolveRshRunner(testArgs, env);}

  return {command: "./docker-mom.sh", args: ["node", ...testArgs], env};
}

/** @param {RunResult} result */
function isMissingGpuOutput(result) {
  const output = `${result.stdout}\n${result.stderr}`;
  if (result.error) {return false;}
  // A run that reported a clean pass clearly found its device; do not let
  // diagnostic stderr (e.g. a SYCL runtime buffer/info notice) misclassify it.
  if (result.stdout.includes("PASSED")) {return false;}
  // Preserve actionable compiler/JIT/crash diagnostics. Some of them mention a SYCL device in
  // their surrounding worker error, which the availability pattern below must not turn into a
  // misleading "device unavailable" skip.
  if (/\[AdaptiveCpp Error\]|Code object construction failed|Worker \d+ exited unexpectedly|LLVM ERROR|fatal error/i
    .test(output)) {return false;}
  if (result.code === 0 && result.stdout.trim() === "" && result.stderr.trim() === "") {return true;}
  return [
    /Unknown compute platform gpu|No device of requested type|gpu[0-9]+.*not found/i,
    /No SYCL (?:GPU )?device(?: is)? (?:available|found)|SYCL (?:GPU )?device (?:is )?(?:unavailable|not found)/i,
    /No GPU(?:s| devices?)?(?: (?:are|is))? (?:available|found|detected)/i,
  ].some((pattern) => pattern.test(output));
}

/** @param {string} value */
function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** @param {TestEnvironment} [extra] @returns {TestEnvironment} */
function childEnv(extra = {}) {
  const env = {...process.env};
  for (const [key, value] of Object.entries(extra)) {
    if (value === undefined) {
      delete env[key];
    } else {
      env[key] = value;
    }
  }
  if (process.platform !== "win32") {return env;}
  return withWindowsTestPath(env);
}

/** @param {string} entry @returns {string | null} */
function releasePathEntry(entry) {
  return hasReleaseExecutable ? entry : null;
}

/** @param {TestEnvironment} env @returns {TestEnvironment} */
function withWindowsTestPath(env) {
  return withWindowsPathEntries(env, [
    env["MOM_NATIVE_PATH"] ? path.dirname(env["MOM_NATIVE_PATH"]) : null,
    releasePathEntry(path.join(path.dirname(releaseExecutable), "libs")),
    releasePathEntry(path.dirname(releaseExecutable)),
    path.join(repoRoot, "build", "win", "Release"),
  ]);
}

/** @param {TestEnvironment} env @param {(string | null)[]} entries @returns {TestEnvironment} */
function withWindowsPathEntries(env, entries) {
  const pathKey = normalizeWindowsPathKey(env);
  const pathValue = env[pathKey] || "";
  env[pathKey] = [...entries, pathValue].filter(Boolean).join(path.delimiter);
  return env;
}

/** @param {TestEnvironment} env */
function normalizeWindowsPathKey(env) {
  const pathKey = Object.keys(env).find((key) => key.toLowerCase() === "path") || "Path";
  for (const key of Object.keys(env)) {
    if (key.toLowerCase() === "path" && key !== pathKey) {delete env[key];}
  }
  return pathKey;
}

/** @param {import("node:child_process").ChildProcess} child @param {NodeJS.Signals} [signal] */
function killProcessTree(child, signal = "SIGKILL") {
  if (!child.pid) {
    child.kill(signal);
    return false;
  }
  if (process.platform !== "win32") {
    // Let the parent mark its workers as expected before the hard-kill fallback reaches them.
    const pids = signal === "SIGINT" || process.platform !== "linux"
      ? [child.pid] : processTreePids(child.pid);
    for (const pid of pids) {
      try {process.kill(pid, signal);} catch { /* process already exited */ }
    }
    return false;
  }
  const killer = spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
    stdio: "ignore",
  });
  killer.on("error", () => child.kill(signal));
  return true;
}

/** @param {number} pid @param {Set<number>} [seen] @returns {number[]} */
function processTreePids(pid, seen = new Set()) {
  if (seen.has(pid)) {return [];}
  seen.add(pid);
  /** @type {number[]} */
  let children = [];
  try {
    children = fs.readFileSync(`/proc/${pid}/task/${pid}/children`, "utf8")
      .trim().split(/\s+/).filter(Boolean).map(Number);
  } catch { /* /proc is unavailable or the process exited */ }
  return [pid, ...children.flatMap(child => processTreePids(child, seen))];
}

/** @param {import("node:child_process").ChildProcess} child */
function detachChild(child) {
  child.stdout?.destroy();
  child.stderr?.destroy();
  child.unref();
}

/** @returns {BufferedRunResult} */
function createRunResult() {
  return {
    code: null,
    signal: null,
    error: null,
    stdout: "",
    stderr: "",
  };
}

/** @param {string[]} args @param {TestEnvironment} [env] @param {string} [cwd] @returns {SpawnedMiner} */
function spawnMiner(args, env, cwd = repoRoot) {
  if (!hasReleaseExecutable && args[0] === "mom.js") {
    args = [path.join(repoRoot, "mom.js"), ...args.slice(1)];
  }
  const {command, args: commandArgs, windowsVerbatimArguments} = resolveMinerCommand(args);
  const spawnEnv = childEnv(env);
  if (env?.["MOM_BENCHMARK_CONTROL_STDIN"] === "1") {
    return spawn(command, commandArgs, {
      cwd, env: spawnEnv, stdio: ["pipe", "pipe", "pipe"],
      ...(windowsVerbatimArguments ? {windowsVerbatimArguments: true} : {}),
    });
  }
  return spawn(command, commandArgs, {
    cwd, env: spawnEnv, stdio: ["ignore", "pipe", "pipe"],
    ...(windowsVerbatimArguments ? {windowsVerbatimArguments: true} : {}),
  });
}

/** @param {BufferedRunResult} result @param {OutputStream} streamName @param {Buffer | string} chunk */
function appendOutput(result, streamName, chunk) {
  let streams = result[outputBuffers];
  if (!streams) {
    const created = {};
    Object.defineProperty(result, outputBuffers, {value: created});
    streams = created;
  }
  const existing = streams[streamName] || Buffer.from(result[streamName] || "", "utf8");
  const incoming = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk), "utf8");
  let tail;
  if (incoming.length >= MAX_CAPTURED_OUTPUT_BYTES) {
    tail = Buffer.from(incoming.subarray(incoming.length - MAX_CAPTURED_OUTPUT_BYTES));
  } else if (existing.length + incoming.length > MAX_CAPTURED_OUTPUT_BYTES) {
    const keep = MAX_CAPTURED_OUTPUT_BYTES - incoming.length;
    tail = Buffer.concat([existing.subarray(-keep), incoming], MAX_CAPTURED_OUTPUT_BYTES);
  } else {
    tail = Buffer.concat([existing, incoming], existing.length + incoming.length);
  }
  streams[streamName] = tail;
  const text = tail.toString("utf8");
  const encoded = Buffer.from(text, "utf8");
  if (encoded.length <= MAX_CAPTURED_OUTPUT_BYTES) {
    result[streamName] = text;
    return;
  }

  let start = encoded.length - MAX_CAPTURED_OUTPUT_BYTES;
  while (true) {
    const byte = encoded[start];
    if (byte === undefined || (byte & 0xc0) !== 0x80) {break;}
    start++;
  }
  result[streamName] = encoded.subarray(start).toString("utf8");
}

/** @param {string[]} args @param {RunOptions} [options] @returns {Promise<BufferedRunResult>} */
function runNode(args, options = {}) {
  const timeoutMs = options.timeoutMs || 5 * 60 * 1000;

  return new Promise((resolve) => {
    const child = spawnMiner(args, options.env, options.cwd);
    const result = createRunResult();
    let settled = false;
    /** @type {ReturnType<typeof setTimeout> | null} */
    let forceResolveTimeout = null;

    const finish = () => {
      if (settled) {return;}
      settled = true;
      clearTimeout(timeout);
      if (forceResolveTimeout) {clearTimeout(forceResolveTimeout);}
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

/** @param {unknown} value @returns {value is Record<string, unknown>} */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** @param {string} json @returns {Record<string, string>} */
function parseAlgoParams(json) {
  const value = JSON.parse(json);
  if (!isRecord(value)) {throw new Error("MOM_ALGORITHMS was not an object");}
  /** @type {Record<string, string>} */
  const params = {};
  for (const [key, entry] of Object.entries(value)) {
    if (typeof entry !== "string") {
      throw new Error("MOM_ALGORITHMS was not a string-valued object");
    }
    params[key] = entry;
  }
  return params;
}

/** @param {unknown} error */
function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

/** @param {TestEnvironment} env @returns {Promise<Record<string, string>>} */
async function getAutoAlgoParams(env) {
  return (await getAutoAlgoParamsReport(env)).params;
}

/** @param {TestEnvironment} env @returns {Promise<AlgoParamsReport>} */
function detectAlgoParams(env) {
  // Invoke the miner directly. The old extra Node wrapper applied the selected compiler
  // environment twice around release launchers, allowing an inherited libsycl with the same SONAME
  // to pre-empt the runtime beside the selected addon.
  const args = ["mom.js", "algorithms"];
  return runNode(args, {timeoutMs: ALGO_PARAMS_TIMEOUT_MS, env}).then((result) => {
    if (result.error || result.code !== 0) {
      throw new Error(formatFailure("Unable to detect algorithms", args, result));
    }

    const line = result.stdout.trim().split(/\r?\n/).reverse()
      .find((entry) => entry.startsWith("MOM_ALGORITHMS "));
    if (!line) {
      throw new Error(formatFailure("Algorithms output did not contain JSON marker", args, result));
    }
    return {
      params: parseAlgoParams(line.slice("MOM_ALGORITHMS ".length)),
      stdout: result.stdout,
      stderr: result.stderr,
    };
  });
}

const getCachedAlgoParamsReport = createAlgoParamsReportCache(detectAlgoParams);

/** @param {TestEnvironment} env @returns {Promise<AlgoParamsReport>} */
async function getAutoAlgoParamsReport(env) {
  return getCachedAlgoParamsReport(env);
}

/** @param {string} output @returns {Array<{dev: string, description: string}>} */
function parseSyclCpuDevices(output) {
  /** @type {Array<{dev: string, description: string}>} */
  const devices = [];
  for (const line of output.split(/\r?\n/)) {
    const match = line.match(/^(cpu\d+):\s+(.+)$/);
    const dev = match?.[1];
    const description = match?.[2];
    if (dev && description) {devices.push({dev, description});}
  }
  return devices;
}

/** @param {string} output @param {boolean | null} [integrated] @returns {GpuDevice[]} */
function parseGpuDevices(output, integrated = null) {
  /** @type {Map<string, GpuDevice>} */
  const devices = new Map();
  for (const line of output.split(/\r?\n/)) {
    const match = line.match(/^(gpu\d+):\s+(.+)$/);
    if (!match) {continue;}
    const dev = match[1];
    const description = match[2];
    if (!dev || !description) {continue;}
    const isIntegrated = /\s\[integrated\]$/i.test(description);
    if (integrated !== null && isIntegrated !== integrated) {continue;}
    devices.set(dev, {dev, description, integrated: isIntegrated});
  }
  return [...devices.values()];
}

/** @param {string} output @returns {Array<{dev: string, description: string}>} */
function parseDiscreteGpuDevices(output) {
  return parseGpuDevices(output, false).map(({dev, description}) => ({dev, description}));
}

/** @param {string} vendor @param {{algo?: string, integrated?: boolean | null, backend?: string, env?: TestEnvironment}} [options] @returns {TestEnvironment} */
function gpuDeviceDiscoveryEnv(vendor, options = {}) {
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
  return {...baseEnv, ...selectedEnv};
}

/** @param {string} vendor @param {{algo?: string, integrated?: boolean | null, backend?: string, env?: TestEnvironment}} [options] */
function gpuDeviceDiscoveryKey(vendor, options = {}) {
  return JSON.stringify([
    normalizeAlgoParamsEnv(gpuDeviceDiscoveryEnv(vendor, options)),
    Object.hasOwn(options, "integrated") ? options.integrated : "default",
  ]);
}

/** @param {string} vendor @param {{algo?: string, integrated?: boolean | null, backend?: string, env?: TestEnvironment}} [options] @returns {Promise<GpuDiscoveryResult>} */
async function getGpuDevices(vendor, options = {}) {
  let report;
  try {
    report = await getAutoAlgoParamsReport(gpuDeviceDiscoveryEnv(vendor, options));
  } catch (error) {
    return {skipped: true, reason: `${vendor} GPU discovery failed: ${errorMessage(error)}`};
  }
  const integrated = Object.hasOwn(options, "integrated") ? options.integrated : false;
  const devices = parseGpuDevices(`${report.stdout}\n${report.stderr}`, integrated);
  if (!devices.length) {
    const kind = integrated === null ? "" : integrated ? " integrated" : " discrete";
    return {skipped: true, reason: `No${kind} ${vendor} GPU is available in this environment`};
  }
  return {skipped: false, devices, params: report.params};
}

/** @param {string} message @returns {SkippedResult} */
function syclCpuUnavailable(message) {
  if (process.env["GITHUB_ACTIONS"] || process.env["MOM_REQUIRE_PORTABLE_CPU_TESTS"] === "1") {
    emitGitHubError("SYCL CPU device unavailable", message);
    throw new Error(message);
  }

  return {
    skipped: true,
    reason: message,
  };
}

/** @param {unknown} error @returns {SkippedResult} */
function syclCpuDetectionFailure(error) {
  const message = errorMessage(error);
  if (process.env["GITHUB_ACTIONS"] || process.env["MOM_REQUIRE_PORTABLE_CPU_TESTS"] === "1") {
    emitGitHubError("SYCL CPU device unavailable", message);
    throw error;
  }
  return {
    skipped: true,
    reason: `SYCL CPU device detection failed: ${message}`,
  };
}

/** @param {TestEnvironment} env @returns {Promise<CpuDeviceResult>} */
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
  const first = devices[0];
  if (first) {return {skipped: false, ...first};}

  const message = [
    "No SYCL CPU device was reported by algorithms output.",
    formatOutput("stdout", report.stdout),
    formatOutput("stderr", report.stderr),
  ].join("\n");
  return syclCpuUnavailable(missingSyclCpuMessage(message));
}

/** @param {string} reportMessage */
function missingSyclCpuMessage(reportMessage) {
  if (process.env["GITHUB_ACTIONS"]) {return reportMessage;}
  return "SYCL CPU device is not available in this environment";
}

/** @returns {CpuDeviceResult | null} */
function assumedSyclCpuDevice() {
  if (!process.env["MOM_ASSUME_SYCL_CPU"]) {return null;}
  return {
    skipped: false,
    dev: process.env["MOM_ASSUME_SYCL_CPU"],
    description: "configured by MOM_ASSUME_SYCL_CPU",
  };
}

const execution = require("./miner_execution")({
  compilerPolicy, getAutoAlgoParams, runNode, formatFailure, emitGitHubError,
  isMissingGpuOutput, spawnMiner, appendOutput, createRunResult, killProcessTree,
  medianHashrate, hashrateUnitMultipliers, escapeRegExp, parseFormattedHashrate,
  setTimeout, clearTimeout,
});

module.exports = {
  createAlgoParamsReportCache,
  formatHashrate,
  getGpuDevices,
  gpuDeviceDiscoveryKey,
  getFirstSyclCpuDevice,
  isMissingGpuOutput,
  maxCapturedOutputBytes: MAX_CAPTURED_OUTPUT_BYTES,
  parseFormattedHashrate,
  parseDiscreteGpuDevices,
  parseGpuDevices,
  repoRoot,
  resolveNodeRunner,
  runNode,
  spawnAndExit,
  ...execution,
  wrapWindowsCmd,
};
