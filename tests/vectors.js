"use strict";

const hashTests = [
  ...require("./vectors/cpu"),
  ...require("./vectors/progpow"),
  ...require("./vectors/memory_hard"),
  ...require("./vectors/equihash"),
];

const nonceAt32Algos = new Set(["kawpow", "firopow", "evrprogpow", "meowpow", "etchash", "autolykos2"]);
// Heights sampled from coin mainnets so perf DAG/table sizes match live pool jobs
// (ETC 2026-06-04, RVN and ERG 2026-06-12). Keep in sync with benchHeightByAlgo in mom.js.
const benchHeightByAlgo = {
  etchash:    24689903,
  kawpow:     4407982,
  firopow:    600000,
  evrprogpow: 1800000,
  meowpow:    825000,
  autolykos2: 1806198,
};

// Build a perf job from a hash vector's source job. Nonce-at-32 algos (see nonceAt32Algos above)
// carry a blob and need a live-sized DAG, so we keep the source job (clearing its dev for autoDev)
// and stamp the sampled height; all other algos only need the algo name.
function perfJob(sourceJob) {
  const algo = sourceJob.algo;
  if (!nonceAt32Algos.has(algo)) {return { algo };}

  const job = { ...sourceJob, dev: undefined };
  if (benchHeightByAlgo[algo]) {job.height = benchHeightByAlgo[algo];}
  return job;
}

// One perf entry per distinct algo, taken from its first hash vector.
const perfTests = [];
const seenAlgos = new Set();
for (const definition of hashTests) {
  const algo = definition.job.algo;
  if (seenAlgos.has(algo)) {continue;}
  seenAlgos.add(algo);

  perfTests.push({
    algo,
    gpu: definition.gpu,
    autoDev: true,
    name: algo,
    timeoutMs: definition.timeoutMs || 3 * 60 * 1000,
    job: perfJob(definition.job),
  });
}

module.exports = {
  hashTests,
  perfTests,
};
