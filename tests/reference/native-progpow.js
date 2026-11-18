"use strict";

// One real Core/registry survives period, seal, DAG-size and short-program shape changes.
// Reuse maintained consensus vectors; adjacent periods additionally require Node/native parity.
const vectors = require("../vectors/progpow");
const tuning = require("../../gpu-tuning");

/** @returns {Array<{job: HashJob & {dev: string}, expected?: string}>} */
function sequence() {
  /** @type {Array<{job: HashJob & {dev: string}, expected?: string}>} */
  const jobs = [];
  /** @type {Array<[string, number, number]>} */
  const variants = [["kawpow", 0, 3], ["firopow", 1, 2], ["evrprogpow", 0, 3], ["meowpow", 0, 6]];
  for (const [algo, height, next] of variants) {
    const vector = vectors.find(({job}) => job.algo === algo && job.height === height);
    if (!vector) {throw new Error("Missing ProgPoW reference: " + algo);}
    jobs.push(vector, {job: {...vector.job, height: next}}, vector);
  }
  const first = jobs[0];
  if (!first) {throw new Error("Missing first ProgPoW reference");}
  return [...jobs, first];
}

/** @typedef {{sendToCpp(name: string, values?: object): void}} ProbeWorker */
/** Exercise only the public worker API; injected workers let ordinary source tests stay CPU-only.
 * @param {{AsyncWorker: new (progress: (name: string, values: Record<string, string>) => void,
 * complete: () => void, failed: (error: Error) => void) => ProbeWorker}} [core]
 * @returns {Promise<Array<{algo: string, height: number, result: string}>>} */
async function probe(core) {
  if (!core) {
    if (process.env["MOM_AOT_GPU_OWNERSHIP_VERIFIED"] !== "1") {throw new Error("Verify fresh GPU ownership before ProgPoW execution");}
    core = typeof globalThis.__momWorkerCreate === "function" ? require("../../runtime/core")
      : require(process.env["MOM_AOT_CORE_ADDON"] || "../../build/lin/Release/dpcpp/mom.node");
  }
  if (!core) {throw new Error("Missing ProgPoW compute bridge");}
  const workerCore = core;
  const cases = sequence();
  return new Promise((resolve, reject) => {
    /** @type {Array<{algo: string, height: number, result: string}>} */
    const results = [];
    let index = 0, closing = false;
    /** @type {Error | undefined} */
    let failure;
    /** @param {Error} [error] */
    function close(error) {
      failure ||= error;
      if (!closing) {
        closing = true;
        try {worker.sendToCpp("close");} catch (error) {clearTimeout(deadline); reject(error);}
      }
    }
    function send() {
      try {
        const current = cases[index];
        if (!current) {throw new Error("Missing ProgPoW case");}
        const job = {...current.job, backend: "sycl"};
        // Match the original worker's named-tuning boundary, not a second device parser.
        tuning.applyNativeJobTuning(job, tuning.parseDeviceEntry(job.dev, job.algo), job.algo);
        worker.sendToCpp("test", job);
      } catch (error) {close(error instanceof Error ? error : new Error(String(error)));}
    }
    const worker = new workerCore.AsyncWorker((name, values) => {
      const current = cases[index], result = values["result"];
      if (closing) {failure ||= new Error("Unexpected result after closing"); return;}
      if (name !== "test" || !current || !result || !/^[0-9a-f]{64} [0-9a-f]{64}$/.test(result) ||
          current.expected && current.expected !== result) {close(new Error("Invalid ProgPoW result at " + index + ": " + name + " " + JSON.stringify(values))); return;}
      results.push({algo: current.job.algo, height: Number(current.job.height), result});
      if (++index === cases.length) {close();} else {send();}
    }, () => {
      clearTimeout(deadline);
      if (failure || index !== cases.length) {reject(failure || new Error("ProgPoW worker closed before all results"));} else {resolve(results);}
    }, close);
    const deadline = setTimeout(() => close(new Error("ProgPoW sequence exceeded its deadline")), 180000);
    send();
  });
}

module.exports = {sequence, probe};
