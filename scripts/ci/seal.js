"use strict";

// Seal same-run handoff files: `node scripts/ci/seal.js <output_key>=<file> [...]` admits each file
// and appends `<output_key>=<sha256>` to GITHUB_OUTPUT so downstream jobs can pin the exact bytes.

const fs = require("node:fs");
const path = require("node:path");
const {fileHash} = require("../build-source-inputs");
const {appendWorkflowOutputs} = require("../native-ci-job");

// The shared same-run handoff bound; it matches the producer and consumer tar admission limits.
const maxHandoffBytes = 16 * 1024 ** 3;

/** Admit one canonical, singly-linked, non-empty regular file no larger than `maxBytes`.
 * A hard link or symlink could let a later step rewrite the bytes after they were pinned.
 * @param {string} file @param {string} label @param {number} [maxBytes] @returns {fs.Stats} */
function canonicalFile(file, label, maxBytes = maxHandoffBytes) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size < 1 || stat.size > maxBytes ||
      !path.isAbsolute(file) || fs.realpathSync(file) !== file) {
    throw new Error(label + " is not a bounded canonical singly-linked regular file: " + file);
  }
  return stat;
}

/** Parse `<output_key>=<file>` arguments; keys are unique so no output can be silently overwritten.
 * @param {string[]} args @returns {Array<[string, string]>} */
function parseSealArguments(args) {
  if (!args.length) {throw new Error("Usage: seal.js <output_key>=<file> [...]");}
  const pairs = args.map(arg => {
    const separator = arg.indexOf("=");
    return /** @type {[string, string]} */ ([arg.slice(0, separator), arg.slice(separator + 1)]);
  });
  if (pairs.some(([key, file]) => !/^[a-z][a-z0-9_]*$/.test(key) || !file) || new Set(pairs.map(([key]) => key)).size !== pairs.length) {
    throw new Error("Seal arguments must be unique <output_key>=<file> pairs");
  }
  return pairs;
}

/** Admit every file before writing any output, then append all hashes together.
 * @param {string[]} args @param {NodeJS.ProcessEnv} [environment] @returns {Record<string, string>} */
function seal(args, environment = process.env) {
  const values = Object.fromEntries(parseSealArguments(args).map(([key, file]) => {
    canonicalFile(file, "Sealed handoff " + key);
    return [key, fileHash(file)];
  }));
  appendWorkflowOutputs(environment, values);
  return values;
}

module.exports = {maxHandoffBytes, canonicalFile, seal};

if (require.main === module) {
  try {
    seal(process.argv.slice(2));
  } catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
