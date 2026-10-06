#!/usr/bin/env node
"use strict";

const mode = process.env["MOM_RELEASE_FIXTURE_MODE"] || "clean";
const args = process.argv.slice(2);
const algo = mode === "compute-error-matrix" && args[0] === "bench"
  ? args[1] || "fake" : process.env["MOM_RELEASE_FIXTURE_ALGO"] || "fake";

if (args[0] === "algorithms") {
  process.stdout.write(`gpu0:fixture\nMOM_ALGORITHMS ${JSON.stringify({[algo]: "gpu0",
    ...(mode === "compute-error-matrix" ? {good: "gpu0"} : {})})}\n`);
  process.exit(0);
}

if (args[0] !== "bench" || args[1] !== algo) {
  process.stderr.write(`unexpected arguments: ${args.join(" ")}\n`);
  process.exit(2);
}

if (algo === "pearlhash") {
  const certIndexes = args.flatMap((value, index) =>
    value === "--job.pearlhash_cert_version" ? [index] : []);
  const certIndex = certIndexes[0];
  if (certIndexes.length !== 1 || certIndex === undefined || args[certIndex + 1] !== "3" ||
      args.includes("--job")) {
    process.stderr.write("PearlHash benchmark option contract failed\n");
    process.exit(5);
  }
}

if (process.env["MOM_BENCHMARK_CONTROL_STDIN"] !== "1") {
  process.stderr.write("benchmark control stdin was not enabled\n");
  process.exit(3);
}
if (process.env["MOM_LOOP_STATS"] !== "1") {
  process.stderr.write("loop stats were not enabled\n");
  process.exit(4);
}

if (["timeout", "timeout-finite", "timeout-interrupt"].includes(mode)) {
  process.stdout.write("fixture setup started\n");
  process.stderr.write("fixture runtime still initializing\n");
}
const samples = {
  timeout: [],
  "timeout-finite": [],
  "timeout-interrupt": [],
  "compute-error": [],
  "compute-error-after-rate": [100],
  "insufficient-samples": [100],
  clean: [80, 100, 100, 100],
  transient: [100, 94, 94, 94],
  trailing: [80, 94, 94, 94, 1000, 0],
  crash: [80, 100, 100, 100],
  boundary: [80, 95, 95, 95],
  below: [80, 94.99, 94.99, 94.99],
  stochastic: [80, 90, 90, 90, 90, 90, 110, 110, 110, 110],
  "cn-gpu-ramp": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 100, 100],
}[mode] || [80, 100, 100, 100];
/** @type {Record<string, number[] | null>} */
const cpuSamplesByMode = {
  "cpu-high": [100, 80, 80],
  "cpu-spike": [100, 90, 10, 90],
  "cpu-missing": null,
  "cpu-short": [100, 10],
  "cpu-invalid": [100, -1, 10],
  "cpu-malformed": [100, 10, 10],
};
const cpuSamples = Object.hasOwn(cpuSamplesByMode, mode) ? cpuSamplesByMode[mode] : [100, 10, 10];
if (cpuSamples) {
  cpuSamples.forEach((cpu, index) => {
    const line = `LOOPSTAT t=${index + 1} wall=10.0s cpu=${cpu}% dispatch=90.0% ` +
      "msg=0.001s post=0.002s iters=1 jobs=1 max_msg=0.2ms max_post=0.3ms";
    const split = Math.floor(line.length / 2);
    process.stderr.write(line.slice(0, split));
    process.stderr.write(line.slice(split));
    if (index + 1 < cpuSamples.length) {process.stderr.write("\n");}
  });
}
if (mode === "cpu-malformed") {process.stderr.write("\nLOOPSTAT malformed");}
const reportSamples = () => {
  for (const sample of samples) {
    process.stdout.write(`Algo ${algo} (gpu0) hashrate: ${sample} H/s\n`);
  }
};
if (mode === "late-samples") {setTimeout(reportSamples, 75);} else {reportSamples();}
if (mode === "compute-error" || mode === "compute-error-after-rate") {
  process.stdout.write("ERROR: Compute core ");
  setTimeout(() => process.stdout.write("error: fixture\n"), 10);
}
if (mode === "crash") {
  process.exit(42);
}

process.stdin.setEncoding("utf8");
let input = "";
process.stdin.on("data", (chunk) => {
  input += chunk;
  if (input !== "close\n") {return;}
  if (mode === "timeout-finite") {return;}
  if (mode === "timeout-interrupt") {
    process.stdout.write("fixture control close accepted\n");
    return;
  }
  if (mode === "slow-close") {
    process.stdout.write("fixture control close accepted\n");
    setTimeout(() => {
      process.stderr.write("\nfixture natural teardown complete\n");
      process.exit(0);
    }, 150);
    return;
  }
  if (["compute-error-after-samples", "compute-error-trimmed"].includes(mode) ||
      (mode === "compute-error-matrix" && algo === "fake")) {
    process.stdout.write("ERROR: Compute core ", () => setImmediate(() => {
      process.stdout.write("error: late fixture\n", () => {
        const tail = mode === "compute-error-trimmed" ? "x".repeat(1024 * 1024 + 4096) : "";
        process.stdout.write(tail, () => process.exit(0));
      });
    }));
    return;
  }
  process.exit(0);
});
if (["timeout-finite", "timeout-interrupt"].includes(mode)) {
  // Finish independently of the parent's deadline or stdin; premature control is observable.
  setTimeout(() => {
    process.stdout.write("fixture finite work complete\n");
    process.stdout.write(`fixture control input ${input.length ? "received" : "absent"}\n`);
    if (mode === "timeout-interrupt") {
      const closes = input.split("\n").filter((line) => line === "close").length;
      process.stdout.write(`fixture control close count ${closes}\n`);
    }
    process.stdin.destroy();
  }, 150);
}
if (mode === "late-samples") {
  // Bound this CPU-only fixture even if the parent forgets to close a completed sample window.
  setTimeout(() => process.stdin.destroy(), 250);
}
if (mode === "insufficient-samples") {process.stdin.destroy();}
