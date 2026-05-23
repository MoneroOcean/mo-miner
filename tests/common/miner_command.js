"use strict";

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

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

function quoteCommand(args) {
  return args
    .map((arg) => (/^[A-Za-z0-9_./:=+-]+$/.test(arg) ? arg : JSON.stringify(arg)))
    .join(" ");
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

function resolveMinerCommand(args) {
  if (hasReleaseExecutable && args[0] === "mo-miner.js") {
    if (/\.cmd$/i.test(releaseExecutable)) {
      const packageDir = path.dirname(releaseExecutable);
      const nodeExe = path.join(packageDir, "mo-miner-node.exe");
      const bundle = path.join(packageDir, "mo-miner.bundle.cjs");
      if (fs.existsSync(nodeExe) && fs.existsSync(bundle)) {
        return [nodeExe, bundle, ...args.slice(1)];
      }
      return wrapWindowsCmd([releaseExecutable, ...args.slice(1)]);
    }
    return [releaseExecutable, ...args.slice(1)];
  }
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

function isMissingGpuOutput(result) {
  const output = `${result.stdout}\n${result.stderr}`;
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

  const pathKey = Object.keys(env).find((key) => key.toLowerCase() === "path") || "Path";
  const pathValue = env[pathKey] || "";
  for (const key of Object.keys(env)) {
    if (key.toLowerCase() === "path" && key !== pathKey) delete env[key];
  }
  env[pathKey] = [
    hasReleaseExecutable ? path.join(path.dirname(releaseExecutable), "libs") : null,
    hasReleaseExecutable ? path.dirname(releaseExecutable) : null,
    path.join(repoRoot, "build", "Release"),
    pathValue,
  ]
    .filter(Boolean)
    .join(path.delimiter);
  return env;
}

function killProcessTree(child, signal = "SIGKILL") {
  if (process.platform !== "win32" || !child.pid) {
    child.kill(signal);
    return;
  }

  const killer = spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
    stdio: "ignore",
  });
  killer.on("error", () => child.kill(signal));
}

function detachChild(child) {
  child.stdout.destroy();
  child.stderr.destroy();
  child.unref();
}

function runNode(args, options = {}) {
  const timeoutMs = options.timeoutMs || 5 * 60 * 1000;

  return new Promise((resolve) => {
    const command = resolveMinerCommand(args);
    const child = spawn(command[0], command.slice(1), {
      cwd: repoRoot,
      env: childEnv(options.env),
      stdio: ["ignore", "pipe", "pipe"],
    });

    const result = {
      code: null,
      signal: null,
      error: null,
      stdout: "",
      stderr: "",
    };
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

    child.stdout.on("data", (chunk) => {
      result.stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      result.stderr += chunk.toString("utf8");
    });
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

async function getFirstSyclCpuDevice() {
  if (process.env.MOMINER_ASSUME_SYCL_CPU) {
    return {
      skipped: false,
      dev: process.env.MOMINER_ASSUME_SYCL_CPU,
      description: "configured by MOMINER_ASSUME_SYCL_CPU",
    };
  }

  let report;
  try {
    report = await getAutoAlgoParamsReport();
  } catch (error) {
    if (process.env.GITHUB_ACTIONS) {
      emitGitHubError("SYCL CPU device unavailable", error.message);
      throw error;
    }
    return {
      skipped: true,
      reason: `SYCL CPU device detection failed: ${error.message}`,
    };
  }

  const output = `${report.stdout}\n${report.stderr}`;
  const devices = parseSyclCpuDevices(output);
  if (devices.length) return { skipped: false, ...devices[0] };

  const message = [
    "No SYCL CPU device was reported by algo_params output.",
    formatOutput("stdout", report.stdout),
    formatOutput("stderr", report.stderr),
  ].join("\n");
  if (process.env.GITHUB_ACTIONS) {
    emitGitHubError("SYCL CPU device unavailable", message);
    throw new Error(message);
  }

  return {
    skipped: true,
    reason: "SYCL CPU device is not available in this environment",
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

async function runMinerTest(definition) {
  const job = withTestDevice(definition);
  const expected = Array.isArray(definition.expected)
    ? definition.expected.join("|")
    : definition.expected;
  const args = [
    "mo-miner.js",
    "test",
    job.algo,
    expected,
    "--job",
    JSON.stringify(job),
  ];
  let result = await runNode(args, { timeoutMs: definition.timeoutMs });
  const output = `${result.stdout}\n${result.stderr}`;

  if (definition.gpu && isMissingGpuOutput(result)) {
    return { skipped: true, reason: "Requested SYCL device is not available in this environment" };
  }

  if (result.error || result.code !== 0) {
    if (process.platform === "win32" && !process.env.MOMINER_DEBUG_STARTUP) {
      const debugResult = await runNode(args, {
        timeoutMs: definition.timeoutMs,
        env: { MOMINER_DEBUG_STARTUP: "1" },
      });
      result = {
        ...result,
        stderr: [
          result.stderr,
          "Debug rerun:",
          formatFailure(`${definition.name} debug rerun`, args, debugResult),
        ].filter(Boolean).join("\n"),
      };
    }
    const message = formatFailure(`${definition.name} failed`, args, result);
    emitGitHubError(definition.name, message);
    throw new Error(message);
  }
  if (!result.stdout.includes("PASSED") || /\bFAIL(?:ED)?\b/.test(output)) {
    const message = formatFailure(`${definition.name} did not report a clean pass`, args, result);
    emitGitHubError(definition.name, message);
    throw new Error(message);
  }

  return { skipped: false };
}

async function runMinerBench(definition) {
  const resolved = await resolveBenchJob(definition);
  if (resolved.skipped) return resolved;

  const job = resolved.job;
  const args = ["mo-miner.js", "bench", job.algo, "--job", JSON.stringify(job)];
  const timeoutMs = definition.timeoutMs || 150 * 1000;
  const hashratePattern = new RegExp(`Algo ${escapeRegExp(job.algo)} \\([^)]*\\) hashrate: ([0-9.]+) H\\/s`);

  return new Promise((resolve, reject) => {
    const command = resolveMinerCommand(args);
    const child = spawn(command[0], command.slice(1), {
      cwd: repoRoot,
      env: childEnv(),
      stdio: ["ignore", "pipe", "pipe"],
    });
    const result = { code: null, signal: null, error: null, stdout: "", stderr: "" };
    let matchedHashrate = null;
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
      result[streamName] += chunk.toString("utf8");
      const match = `${result.stdout}\n${result.stderr}`.match(hashratePattern);
      if (match && !matchedHashrate) {
        matchedHashrate = Number.parseFloat(match[1]);
        stop();
      }
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

      if (matchedHashrate && matchedHashrate > 0) return resolve({ hashrate: matchedHashrate, dev: job.dev });
      if (definition.gpu && isMissingGpuOutput(result)) {
        return resolve({ skipped: true, reason: "GPU device is not available in this environment" });
      }

      reject(new Error(formatFailure(`${definition.name} did not report hashrate`, args, result)));
    });
  });
}

module.exports = {
  getFirstSyclCpuDevice,
  hasReleaseExecutable,
  repoRoot,
  resolveMinerCommand,
  runMinerBench,
  runMinerTest,
  spawnAndExit,
};
