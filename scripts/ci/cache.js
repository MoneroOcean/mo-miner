"use strict";

// actions/cache key for an expensive pinned SDK work root (.github/workflows/native-job.yml). The key
// hashes the producer recipe and every input its admission compares, so a hit is only a candidate: the
// producer re-admits the restored root with its normal admission and rebuilds it when that fails. A
// release never sees which way the root was obtained.
const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto");
const {fileHash, sourceInputs} = require("../build-source-inputs");
const {ciSuffix, ciBuildJobs} = require("../native-ci-job");
const {appendOutputs} = require("./seal");

const repo = path.resolve(__dirname, "../..");

/** Cached work roots by name; `inputs` must cover everything the producer's admission checks. */
const caches = Object.freeze({
  /** @param {NodeJS.ProcessEnv} environment */
  "linux-llvm": environment => {
    const llvm = require("../build-native-llvm-sdk"), {linuxLlvmTools} = require("../native-ci-linux-llvm");
    return {root: "/tmp/mom-native-llvm21-sdk-" + ciSuffix(environment), inputs: {
      recipe: sourceInputs(repo, [], [...llvm.recipeFiles]).graph, archiveSha256: llvm.archiveSha256, jobs: ciBuildJobs(environment),
      tools: Object.fromEntries(Object.entries(linuxLlvmTools).map(([name, file]) => [name, fileHash(fs.realpathSync(file))]))}};
  },
});

/** @param {string} name @param {NodeJS.ProcessEnv} [environment] @returns {{path: string, key: string}} */
function cacheKey(name, environment = process.env) {
  if (!Object.hasOwn(caches, name)) {throw new Error("Unknown native CI cache: " + name);}
  const {root, inputs} = caches[/** @type {keyof typeof caches} */ (name)](environment);
  return {path: root, key: "mom-native-" + name + "-" + createHash("sha256").update(JSON.stringify(inputs)).digest("hex")};
}

module.exports = {cacheKey};

if (require.main === module) {
  try {
    if (process.argv.length !== 3) {throw new Error("Usage: cache.js <name>");}
    appendOutputs(cacheKey(process.argv[2] || ""));
  } catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
