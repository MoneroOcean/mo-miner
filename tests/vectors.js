"use strict";

/** @type {HashVectorDefinition[]} */
const hashTests = [
  ...require("./vectors/cpu"),
  ...require("./vectors/progpow"),
  ...require("./vectors/memory_hard"),
  ...require("./vectors/equihash"),
];

const sourceJobAlgos = new Set([
  "kawpow", "firopow", "evrprogpow", "meowpow", "etchash", "octopus", "autolykos2",
  "c30", "hoohash", "walahash", "xelishashv3", "nexapow", "equihash192_7", "zhash",
]);

// Keep source blobs when the algorithm needs header data to initialize its benchmark state;
// other algos only need their name. MoM's benchmark-job preparation supplies representative
// heights for direct/perf jobs.
/** @param {HashJob} sourceJob @returns {HashJob} */
function perfJob(sourceJob) {
  const algo = sourceJob.algo;
  if (!sourceJobAlgos.has(algo)) {return {algo};}

  const job = {...sourceJob};
  delete job.dev;
  if (algo === "nexapow") {job["target"] = "00".repeat(32);}
  return job;
}

// One perf entry per distinct algo, taken from its first hash vector.
/** @type {PerfDefinition[]} */
const perfTests = [];
const seenAlgos = new Set();
for (const definition of hashTests) {
  const algo = definition.job.algo;
  if (seenAlgos.has(algo)) {continue;}
  seenAlgos.add(algo);

  /** @type {PerfDefinition} */
  const perfDefinition = {
    algo,
    autoDev: true,
    name: algo,
    timeoutMs: definition.perfTimeoutMs || definition.timeoutMs ||
      (algo === "c30" ? 6 : algo === "nexapow" ? 5 : 3) * 60 * 1000,
    job: perfJob(definition.job),
  };
  if (definition.gpu !== undefined) {perfDefinition.gpu = definition.gpu;}
  perfTests.push(perfDefinition);
}

module.exports = {
  hashTests,
  perfTests,
};
