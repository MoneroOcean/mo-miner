"use strict";

// Runtime-only release lane inputs: admit the candidate selection before any artifact access, then
// admit the downloaded executable by its build-audit SHA-256. Runs on Linux and Windows x64 runners.

const fs = require("node:fs");
const path = require("node:path");
const {fileHash} = require("../build-source-inputs");
const {appendOutputs} = require("./seal");
const {ciSuffix} = require("../native-ci-job");

/** Derive the executable artifact name and fresh download directory.
 * @param {NodeJS.ProcessEnv} environment @param {string} platform @param {string} variant */
function runtimeLocation(environment, platform, variant) {
  const suffix = ciSuffix(environment);
  const artifactPlatform = platform === "win32" ? "windows" : platform, packingPrefix = variant === "packed" ? "packed-" : "";
  const directory = path.join(environment["RUNNER_TEMP"] || "", "mom-native-runtime-" + suffix);
  return {artifactName: "native-" + artifactPlatform + "-" + packingPrefix + "executable-" + suffix, directory,
    binary: path.join(directory, platform === "win32" ? "mom.exe" : "mom")};
}

/** Validate the explicit candidate selection on this runner before any artifact download.
 * @param {NodeJS.ProcessEnv} [environment] @returns {{artifact_name: string, directory: string}} */
function admitCandidate(environment = process.env) {
  const sha256 = environment["NATIVE_EXPECTED_SHA256"] || "", platform = environment["NATIVE_PLATFORM"] || "",
    variant = environment["NATIVE_VARIANT"] || "raw";
  if (!["linux", "win32"].includes(platform)) {throw new Error("platform must be linux or win32");}
  if (!["raw", "packed"].includes(variant)) {throw new Error("variant must be raw or packed");}
  const location = runtimeLocation(environment, platform, variant);
  if (!/^[0-9a-f]{64}$/.test(sha256)) {throw new Error("expected_sha256 must be 64 lowercase hexadecimal digits");}
  const current = process.platform === "win32" ? "win32" : process.platform === "linux" ? "linux" : "unsupported";
  if (process.arch !== "x64" || current !== platform) {throw new Error("platform must match this Linux/Windows x64 runner");}
  if (!path.isAbsolute(location.directory) || fs.existsSync(location.directory)) {
    throw new Error("runtime artifact destination must be a new absolute runner-temp path");
  }
  console.log("Accepted runtime-only candidate input for " + platform + " on this x64 runner.");
  return {artifact_name: location.artifactName, directory: location.directory};
}

/** Admit the downloaded executable: the run-scoped directory holds only it, it is a bounded regular
 * file, and its bytes match the build-audit hash. Linux execute mode is restored only after that.
 * @param {NodeJS.ProcessEnv} [environment] @returns {{binary: string}} */
function admitBinary(environment = process.env) {
  const platform = environment["NATIVE_PLATFORM"] || "", expected = environment["NATIVE_EXPECTED_SHA256"] || "";
  if (!["linux", "win32"].includes(platform)) {throw new Error("platform must be linux or win32");}
  const {directory, binary} = runtimeLocation(environment, platform, "raw"), root = fs.lstatSync(directory);
  if (!root.isDirectory() || root.isSymbolicLink()) {throw new Error("runtime artifact must be extracted to its exact run-scoped directory");}
  const entries = fs.readdirSync(directory);
  if (entries.length !== 1 || entries[0] !== path.basename(binary)) {throw new Error("runtime artifact must contain only the expected executable file");}
  const stat = fs.lstatSync(binary);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size === 0 || stat.size >= (1 << 30)) {
    throw new Error("runtime artifact must be a nonempty regular executable smaller than 1 GiB");
  }
  const actual = fileHash(binary);
  if (!/^[0-9a-f]{64}$/.test(expected) || actual !== expected) {throw new Error("runtime artifact SHA-256 differs from the explicit build-audit hash");}
  if (platform === "linux" && !(stat.mode & 0o111)) {fs.chmodSync(binary, stat.mode | 0o111);}
  console.log("Selected executable SHA-256 verified: " + actual);
  return {binary};
}

/** Dispatch one runtime input command and append its outputs. @param {string[]} args */
function main(args) {
  const [command, ...rest] = args;
  if (rest.length) {throw new Error("Runtime input commands take no further arguments");}
  if (command === "candidate") {
    appendOutputs(admitCandidate());
  } else if (command === "binary") {
    appendOutputs(admitBinary());
  } else {
    throw new Error("Usage: runtime-input.js candidate|binary");
  }
}

module.exports = {runtimeLocation, admitCandidate, admitBinary, main};

if (require.main === module) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
