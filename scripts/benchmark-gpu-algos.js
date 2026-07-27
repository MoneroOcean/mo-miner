"use strict";

const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const compilerPolicy = require("../compiler-policy");

const argv = process.argv.slice(2);
function option(name, fallback) {
  const index = argv.indexOf(name);
  return index >= 0 && argv[index + 1] ? argv[index + 1] : fallback;
}

const outputPath = path.resolve(option("--output", "gpu-benchmark.json"));
const label = option("--label", `${process.platform}-${process.arch}`);
const samplesWanted = Math.max(1, Number(option("--samples", "1")) || 1);
const warmupSamples = Math.max(0, Number(option("--warmup-samples", "0")) || 0);
const timeoutMs = Math.max(30000, Number(option("--timeout-ms", "150000")) || 150000);
const requested = option("--algos", "").split(",").map(value => value.trim()).filter(Boolean);
const backend = option("--backend", "");
const rateUnits = { "H/s": 1, "KH/s": 1e3, "MH/s": 1e6, "GH/s": 1e9,
  "TH/s": 1e12, "PH/s": 1e15 };

function minerArgs(args) {
  return ["mom.js", ...args];
}

function discoverGpuJobs() {
  const report = spawnSync(process.execPath, minerArgs(["algo_params"]), {
    encoding: "utf8", env: process.env, timeout: timeoutMs,
  });
  const text = `${report.stdout || ""}\n${report.stderr || ""}`;
  const match = text.match(/MOM_ALGO_PARAMS\s+(\{[^\r\n]+\})/);
  if (!match) {throw new Error(`algo_params did not return MOM_ALGO_PARAMS:\n${text}`);}
  const params = JSON.parse(match[1]);
  // algo_params includes a human-readable backend annotation. Feed only the underlying device
  // specification back to bench; the miner will resolve and display the backend for that run.
  return Object.fromEntries(Object.entries(params)
    .filter(([, dev]) => /^gpu\d+/.test(dev))
    .map(([algo, reported]) => [algo, compilerPolicy.parseReportedAlgoParam(reported).dev]));
}

function benchmark(algo, dev) {
  return new Promise(resolve => {
    const started = Date.now();
    const samples = [];
    const observedSamples = [];
    let stdout = "";
    let stderr = "";
    let settled = false;
    let requestedStatus = "";
    let forceTimer;
    const jobArgs = ["bench", algo, "--job.dev", dev];
    if (backend) {jobArgs.push("--job.backend", backend);}
    const child = spawn(process.execPath, minerArgs(jobArgs), {
      env: {...process.env, MOM_BENCHMARK_CONTROL_STDIN: "1"},
      stdio: ["pipe", "pipe", "pipe"], windowsHide: true,
    });
    child.stdin.on("error", error => {
      stderr += `\nbenchmark control pipe: ${error.stack || error}`;
      if (requestedStatus && child.exitCode === null && child.signalCode === null) {
        child.kill("SIGTERM");
      }
    });

    const finish = status => {
      if (settled) {return;}
      settled = true;
      clearTimeout(timer);
      clearTimeout(forceTimer);
      process.off("SIGINT", interrupt);
      process.off("SIGTERM", interrupt);
      resolve({ algo, dev, status, warmup_samples: observedSamples.slice(0, warmupSamples), samples,
        elapsed_ms: Date.now() - started,
        stdout_tail: stdout.slice(-2000), stderr_tail: stderr.slice(-2000) });
    };
    // Do not start the next algorithm until mom.js has closed and reaped every compute worker. The
    // explicit control pipe works on every OS; Windows emulates SIGTERM by abruptly killing only the
    // parent, which can skip N-API cleanup and briefly orphan an in-flight GPU worker. Retain a
    // bounded hard-kill fallback only for a genuinely stuck miner.
    const requestStop = status => {
      if (requestedStatus || settled) {return;}
      requestedStatus = status;
      clearTimeout(timer);
      // exitCode/signalCode can become visible just before the "exit" callback. Let that callback
      // classify the real termination instead of racing it with the requested successful status.
      if (child.exitCode !== null || child.signalCode !== null) {return;}
      if (child.stdin.writable) {child.stdin.end("close\n");}
      else {child.kill("SIGTERM");}
      forceTimer = setTimeout(() => child.kill("SIGKILL"), 10000);
    };
    // If a systemd/container gate is stopped, keep this parent alive long enough to forward the
    // signal and reap mom.js. Abruptly orphaning an in-flight GPU worker can look like a compiler or
    // driver reset even though only the benchmark harness was interrupted.
    const interrupt = () => requestStop("interrupted");
    process.once("SIGINT", interrupt);
    process.once("SIGTERM", interrupt);
    const scan = chunk => {
      stdout += chunk;
      process.stdout.write(chunk);
      const regex = /Algo\s+([^\r\n]+?)\s+hashrate:\s+([0-9.]+)\s+([^\s\r\n]+)/g;
      let match;
      while ((match = regex.exec(stdout))) {
        const key = `${match.index}:${match[2]}:${match[3]}`;
        if (!observedSamples.some(sample => sample.key === key)) {
          const value = Number(match[2]);
          const unit = match[3];
          // Miner output chooses a human-readable SI prefix independently for every sample. Keep the
          // printed pair for reports, but also persist one prefix-independent value so a warm-up that
          // crosses H/s -> KH/s cannot be mis-sorted as a 1000x regression.
          const sample = { key, value, unit,
            value_per_second: rateUnits[unit] ? value * rateUnits[unit] : null };
          observedSamples.push(sample);
          // A miner can print once more while handling our graceful close. Never let that
          // teardown-time line replace the requested final steady sample in JSON.
          if (observedSamples.length > warmupSamples && samples.length < samplesWanted) {
            samples.push(sample);
          }
        }
      }
      if (samples.length >= samplesWanted) {requestStop("ok");}
    };
    child.stdout.on("data", data => scan(data.toString()));
    child.stderr.on("data", data => { stderr += data; process.stderr.write(data); });
    child.on("error", error => { stderr += `\n${error.stack || error}`; finish("spawn-error"); });
    child.on("exit", (code, signal) => {
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
      finish(abnormal || runtimeDiagnostic || requestedStatus ||
        (samples.length >= samplesWanted ? "ok" : "exit-before-samples"));
    });
    const timer = setTimeout(() =>
      requestStop(samples.length >= samplesWanted ? "ok" : "timeout"), timeoutMs);
  });
}

async function main() {
  const jobs = discoverGpuJobs();
  if (Object.keys(jobs).length === 0) {
    throw new Error("algo_params did not report any GPU jobs");
  }
  const algos = requested.length ? requested : Object.keys(jobs).sort();
  const report = {
    label, timestamp: new Date().toISOString(), platform: process.platform,
    selector: process.env.ONEAPI_DEVICE_SELECTOR || "", backend: backend || "policy",
    samples_wanted: samplesWanted,
    warmup_samples: warmupSamples, timeout_ms: timeoutMs, jobs, results: [],
  };
  for (const algo of algos) {
    if (!jobs[algo]) {
      report.results.push({ algo, status: "not-detected", samples: [] });
    } else {
      console.log(`\n=== ${label}: ${algo} (${jobs[algo]}) ===`);
      report.results.push(await benchmark(algo, jobs[algo]));
    }
    fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  }
  const failed = report.results.filter(result => result.status !== "ok");
  if (failed.length) {
    throw new Error(`${label} failed: ${failed.map(result => `${result.algo}=${result.status}`).join(", ")}`);
  }
  console.log(`\nWrote ${outputPath}`);
}

main().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
