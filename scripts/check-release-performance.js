#!/usr/bin/env node
"use strict";

const {spawn} = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const compilerPolicy = require("../compiler-policy");
const {parseRate, readPerformanceFile, platformColumns} = require("./readme-performance");

function option(name, fallback) {
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : process.argv[index + 1];
}

const platform = option("--platform");
const miner = path.resolve(option("--miner", process.platform === "win32" ? "mom.cmd" : "mom"));
const readme = path.resolve(option("--readme", "README.md"));
const margin = Number(option("--margin", "0.05"));
const timeoutMs = Number(option("--timeout-ms", "300000"));
const algoFilter = option("--algo");
const devOverride = option("--dev");

if (!platformColumns[platform]) {
  console.error(`--platform must be one of: ${Object.keys(platformColumns).join(", ")}`);
  process.exit(2);
}
if (!fs.existsSync(miner)) {
  console.error(`Release miner is missing: ${miner}`);
  process.exit(2);
}
if (!(margin >= 0 && margin < 1)) {
  console.error("--margin must be between zero and one");
  process.exit(2);
}

function releaseCommand(args) {
  if (process.platform !== "win32" || !/\.cmd$/i.test(miner)) {return [miner, args];}
  // Exercise the packaged launcher itself. Besides selecting mom-node.exe, it supplies the CUDA
  // toolkit and Visual C++ environment required by runtime-compiled ProgPoW kernels; bypassing it
  // silently benchmarks the much slower ahead-of-time fallback.
  return [process.env.ComSpec || "cmd.exe", ["/d", "/s", "/c", miner, ...args]];
}

function stop(child) {
  if (child.exitCode !== null || child.signalCode !== null) {return;}
  if (process.platform === "win32") {
    spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {stdio: "ignore"});
  } else {
    try {process.kill(-child.pid, "SIGINT");} catch {child.kill("SIGINT");}
    setTimeout(() => {
      try {process.kill(-child.pid, "SIGKILL");} catch {
        // The benchmark normally exits cleanly on SIGINT before this fallback runs.
      }
    }, 10000).unref();
  }
}

function run(args, {stopWhen, commandTimeoutMs = timeoutMs} = {}) {
  const [command, commandArgs] = releaseCommand(args);
  return new Promise((resolve, reject) => {
    const child = spawn(command, commandArgs, {
      cwd: path.dirname(miner),
      detached: process.platform !== "win32",
      env: {...process.env, MOM_SKIP_MSR: "1"},
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let stoppedAfterMatch = false;

    const finish = (code, signal) => {
      clearTimeout(timeout);
      if (stoppedAfterMatch) {return resolve({code: 0, signal, stdout, stderr});}
      resolve({code, signal, stdout, stderr});
    };
    const append = (field, chunk) => {
      if (field === "stdout") {stdout += chunk;}
      else {stderr += chunk;}
      const combined = `${stdout}\n${stderr}`;
      if (!stoppedAfterMatch && stopWhen && stopWhen(combined)) {
        stoppedAfterMatch = true;
        stop(child);
      }
    };
    child.stdout.on("data", (chunk) => append("stdout", chunk));
    child.stderr.on("data", (chunk) => append("stderr", chunk));
    child.once("error", reject);
    child.once("close", finish);
    const timeout = setTimeout(() => {
      stop(child);
      reject(new Error(`Timed out after ${commandTimeoutMs} ms: ${args.join(" ")}`));
    }, commandTimeoutMs);
  });
}

function output(result) {
  return `${result.stdout}\n${result.stderr}`;
}

function reportedRates(text, algo) {
  const escaped = algo.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return [...text.matchAll(
    new RegExp(`Algo ${escaped} \\([^)]*\\) hashrate: ([0-9.]+)\\s+(g/s|Sol/s|[KMGT]?H/s)`, "gi")
  )].map((match) => parseRate(`${match[1]} ${match[2]}`));
}

function bestReportedRate(text, algo) {
  return reportedRates(text, algo)
    .reduce((best, rate) => !best || rate.value > best.value ? rate : best, null);
}

async function main() {
  const discovery = await run(["algo_params"]);
  const discoveryOutput = output(discovery);
  if (discovery.code !== 0) {
    throw new Error(`Release device discovery failed:\n${discoveryOutput.trim()}`);
  }
  if (!/^gpu\d+:/m.test(discoveryOutput)) {
    console.log(`SKIP ${platform}: no GPU is available`);
    return;
  }
  const marker = discoveryOutput.split(/\r?\n/)
    .find((line) => line.startsWith("MOM_ALGO_PARAMS "));
  if (!marker) {throw new Error("Release device discovery omitted MOM_ALGO_PARAMS");}
  const algoParams = JSON.parse(marker.slice("MOM_ALGO_PARAMS ".length));

  const rows = readPerformanceFile(readme)
    .map((row) => ({algo: row.algo, reference: row.performance[platform]}))
    .filter((row) => row.reference && (!algoFilter || row.algo === algoFilter));
  if (!rows.length) {throw new Error(`README has no performance rows for ${platform}`);}

  const failures = [];
  for (const {algo, reference} of rows) {
    const reported = algoParams[algo];
    if (!reported || !/\bgpu\d+/i.test(reported)) {
      failures.push(`${algo}: release discovery reported no GPU tuning`);
      continue;
    }
    const selected = compilerPolicy.parseReportedAlgoParam(reported);
    const minimum = reference.value * (1 - margin);
    // A fresh Windows AdaptiveCpp/HIP application database needs two complete cn/gpu windows
    // before its independent workers settle. This is a one-time JIT/runtime warm-up: subsequent
    // windows and application runs use the persistent cache and hold the steady rate.
    const coldWarmupLimit = platform === "amd-windows" && algo === "cn/gpu" ? 3 : 2;
    let rate;
    const result = await run([
      "bench", algo, "--job.dev", devOverride || selected.dev, "--job.backend", selected.backend,
    ], {
      commandTimeoutMs: coldWarmupLimit === 3 ? Math.max(timeoutMs, 8 * 60 * 1000) : timeoutMs,
      stopWhen(text) {
        const rates = reportedRates(text, algo);
        rate = rates.reduce((best, sample) =>
          !best || sample.value > best.value ? sample : best, null);
        // Most algorithms finish after their first full measurement. If that cold sample is below
        // the gate, allow steady-state samples before declaring a regression; this avoids making a
        // 5% floor depend on worker/JIT startup timing without extending successful warm runs.
        return Boolean(rate && (rate.value >= minimum || rates.length >= coldWarmupLimit));
      },
    });
    if (!rate) {rate = bestReportedRate(output(result), algo);}
    if (!rate) {
      failures.push(`${algo}: no hashrate was reported\n${output(result).trim()}`);
      continue;
    }
    const ratio = 100 * rate.value / reference.value;
    const status = rate.value >= minimum ? "PASS" : "FAIL";
    console.log(`${status} ${algo}: ${rate.displayValue} ${rate.unit} ` +
      `(${ratio.toFixed(1)}% of README, minimum ${((1 - margin) * 100).toFixed(0)}%)`);
    if (rate.value < minimum) {
      failures.push(`${algo}: ${rate.value} H/s is below ${minimum} H/s`);
    }
  }
  if (failures.length) {
    throw new Error(`Release performance regressions:\n${failures.join("\n")}`);
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
